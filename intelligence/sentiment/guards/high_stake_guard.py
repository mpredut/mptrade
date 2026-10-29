"""LLM high-stake guard validating large BUY orders (e.g. >= 1000 EUR)."""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

from botcore import _parse_bool_flag
from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    clean_symbol_key,
    is_cache_fresh,
    normalize_order_side,
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.agy.topics import topic_spec
from state_io import read_recovering_previous

logger = logging.getLogger("intelligence.sentiment.high_stake_guard")

DEFAULT_MIN_NOTIONAL_USD = 700.0
DEFAULT_MIN_NOTIONAL_EUR = DEFAULT_MIN_NOTIONAL_USD


def _parse_ts_px_row(row: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    tf = safe_pos_float(row[0], 0.0)
    pf = safe_pos_float(row[1], 0.0)
    if tf <= 0.0 or pf <= 0.0:
        return None
    return (tf * 1000.0 if tf < 1e11 else tf), pf


def compute_24h_price_metrics(series: list) -> Dict[str, Any]:
    """Compute multi-horizon percentage changes (15m, 30m, 1h, 3h) and 24h range from a cache_24_price series."""
    if not isinstance(series, list) or len(series) < 2:
        return {}
    try:
        clean_series = [pt for row in series if (pt := _parse_ts_px_row(row)) is not None]
        if len(clean_series) < 2:
            return {}

        clean_series.sort(key=lambda x: x[0])
        now_ms = clean_series[-1][0]
        cur_px = clean_series[-1][1]

        def _px_ago(mins: float) -> float:
            target = now_ms - mins * 60000.0
            for tms, p_val in reversed(clean_series):
                if tms <= target:
                    return p_val
            return clean_series[0][1]

        def _has_sample_near(mins: float, max_gap_mins: float) -> bool:
            target = now_ms - mins * 60000.0
            min_allowed = now_ms - (mins + max_gap_mins) * 60000.0
            for tms, _ in reversed(clean_series):
                if tms <= target:
                    return tms >= min_allowed
            return False

        span_ms = now_ms - clean_series[0][0]
        sample_prices = [x[1] for x in clean_series[::max(1, len(clean_series) // 500)]]
        l24 = min(sample_prices) if sample_prices else cur_px
        h24 = max(sample_prices) if sample_prices else cur_px
        out: Dict[str, Any] = {
            "low_24h": l24,
            "high_24h": h24,
        }
        if h24 > 0:
            out["dd_24h_high_pct"] = round((cur_px / h24 - 1.0) * 100.0, 2)
        if l24 > 0:
            out["bounce_24h_low_pct"] = round((cur_px / l24 - 1.0) * 100.0, 2)
            out["span_24h_pct"] = round((h24 / l24 - 1.0) * 100.0, 2)
        if h24 > l24 > 0:
            out["pos_24h_pct"] = round(((cur_px - l24) / (h24 - l24)) * 100.0, 1)

        for key, mins, min_span_ms, max_gap_mins in (
            ("chg_1m_pct", 1.0, 60000.0, 2.5),
            ("chg_3m_pct", 3.0, 180000.0, 4.0),
            ("chg_5m_pct", 5.0, 300000.0, 6.0),
            ("chg_15m_pct", 15.0, 0.0, 0.0),
            ("chg_30m_pct", 30.0, 0.0, 0.0),
            ("chg_1h_pct", 60.0, 0.0, 0.0),
            ("chg_3h_pct", 180.0, 0.0, 0.0),
            ("chg_6h_pct", 360.0, 3600000.0 * 4, 0.0),
            ("chg_12h_pct", 720.0, 3600000.0 * 9, 0.0),
            ("chg_24h_pct", 1440.0, 3600000.0 * 18, 0.0),
        ):
            p_ref = _px_ago(mins)
            if p_ref > 0 and span_ms >= min_span_ms and (max_gap_mins <= 0.0 or _has_sample_near(mins, max_gap_mins)):
                out[key] = round((cur_px / p_ref - 1.0) * 100.0, 2)

        recent_15m = [p_v for tms, p_v in clean_series if tms >= now_ms - 15.0 * 60000.0]
        if len(recent_15m) >= 3:
            l15 = min(recent_15m)
            h15 = max(recent_15m)
            out["low_15m"] = l15
            out["high_15m"] = h15
            if h15 > l15 > 0:
                out["pos_15m_pct"] = round(((cur_px - l15) / (h15 - l15)) * 100.0, 1)
                out["span_15m_pct"] = round((h15 / l15 - 1.0) * 100.0, 2)
            tape_pts = []
            for m_off, lbl in ((15.0, "-15m"), (10.0, "-10m"), (5.0, "-5m"), (3.0, "-3m"), (1.0, "-1m"), (0.0, "now")):
                if m_off == 0.0 or (span_ms >= m_off * 60000.0 and _has_sample_near(m_off, max(3.0, m_off * 0.5))):
                    px_m = cur_px if m_off == 0.0 else _px_ago(m_off)
                    tape_pts.append(f"{lbl}:{px_m:.6g}")
            if len(tape_pts) >= 3:
                out["recent_tape_15m"] = tape_pts
        return out
    except Exception:
        return {}


def _extract_cached_items_series(data: Any, cands: list[str]) -> list:
    if not isinstance(data, dict):
        return []
    raw_items = data.get("items", {})
    if isinstance(raw_items, dict):
        for k_cand in cands:
            ser = raw_items.get(k_cand, [])
            if isinstance(ser, list) and ser:
                return ser
        if raw_items:
            first_val = next(iter(raw_items.values()), [])
            return first_val if isinstance(first_val, list) else []
    elif isinstance(raw_items, list):
        return raw_items
    return []


def load_symbol_price_series(
    cachedb_dir: str,
    symbol: str,
    current_price_items: Optional[Dict[str, Any]] = None,
    *,
    include_7d: bool = True,
) -> Tuple[list, list]:
    """Load and merge high-density 24h & 7d price series from cache_24_price, cache_365_price, cache_sparse_price, and cache_current_price."""
    from cachepaths import cache_symbol_variants, legacy_filenames_for, resolve_cachedb_filepath

    cands = [c for c in cache_symbol_variants(symbol, include_original=True) if isinstance(c, str) and "/" not in c]
    series_24h: list = []
    vals_7d: list = []

    # 1. Try primary 24h dense cache (cache_24_price_<SYM>.json and legacy variants)
    for cand in cands:
        primary_24 = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, f"cache_24_price_{cand}.json")))
        for p24_path in (primary_24, *legacy_filenames_for(primary_24)):
            if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                continue
            ser = _extract_cached_items_series(read_recovering_previous(p24_path, json.loads, default=None), cands)
            if len(ser) >= 2:
                series_24h = list(ser)
                break
        if series_24h:
            break

    def _span_ms(pts: list) -> float:
        if len(pts) < 2:
            return 0.0
        p0 = _parse_ts_px_row(pts[0])
        p1 = _parse_ts_px_row(pts[-1])
        return max(0.0, p1[0] - p0[0]) if (p0 and p1) else 0.0

    # 2. Supplement 24h (when < 18h span) and extract 7d history from cache_365_price_<SYM> / cache_sparse_price_<SYM>
    need_24h_supplement = _span_ms(series_24h) < 3600000.0 * 18
    archive_pts: list = []
    cands_upper = {str(c).strip().upper() for c in cands if isinstance(c, str) and str(c).strip()}
    for cand in (cands if (include_7d or need_24h_supplement) else ()):
        base_names = (
            f"cache_365_price_{cand}.jsonl",
            f"cache_365_price_{cand}.json",
            f"cache_sparse_price_{cand}.jsonl",
        )
        candidate_paths: list = []
        for bn in base_names:
            full_p = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, bn)))
            candidate_paths.append(full_p)
            candidate_paths.extend(legacy_filenames_for(full_p))
        for arch_path in candidate_paths:
            if not os.path.exists(arch_path):
                continue
            try:
                if arch_path.endswith(".json"):
                    aser = _extract_cached_items_series(read_recovering_previous(arch_path, json.loads, default=None), cands)
                    if len(aser) >= 2:
                        archive_pts = list(aser)
                        break
                else:
                    with open(arch_path, "rb") as af:
                        af.seek(0, os.SEEK_END)
                        fsize = af.tell()
                        read_bytes = min(fsize, 1024 * 1024)
                        af.seek(max(0, fsize - read_bytes))
                        raw_chunk = af.read().decode("utf-8", errors="replace")
                    lines = raw_chunk.splitlines()
                    if fsize > read_bytes and len(lines) > 1:
                        lines = lines[1:]
                    parsed_rows = []
                    for ln in lines:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            rec = json.loads(ln)
                        except ValueError:
                            continue
                        if isinstance(rec, dict):
                            rec_s = rec.get("s")
                            if (
                                isinstance(rec_s, str)
                                and rec_s.strip()
                                and cands_upper
                                and rec_s.strip().upper() not in cands_upper
                            ):
                                continue
                            item = rec.get("i")
                        elif isinstance(rec, list):
                            item = rec
                        else:
                            continue
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            parsed_rows.append([item[0], item[1]])
                    if len(parsed_rows) >= 2:
                        archive_pts = parsed_rows
                        break
            except Exception:
                continue
        if archive_pts:
            break

    if archive_pts:
        try:
            last_pt = _parse_ts_px_row(archive_pts[-1])
            if last_pt is not None:
                last_t = last_pt[0]
                cutoff_24h = last_t - 86400.0 * 1000.0
                cutoff_7d = last_t - 7.0 * 86400.0 * 1000.0
                pts_24 = []
                pts_7d_vals = []
                for row in archive_pts:
                    pt = _parse_ts_px_row(row)
                    if pt is None:
                        continue
                    tms, rp = pt
                    if tms >= cutoff_7d:
                        pts_7d_vals.append(rp)
                    if need_24h_supplement and tms >= cutoff_24h:
                        pts_24.append([tms, rp])
                if len(pts_7d_vals) >= 2:
                    vals_7d = pts_7d_vals
                if need_24h_supplement and pts_24:
                    series_24h = (pts_24 + series_24h) if series_24h else pts_24
        except Exception:
            pass

    # 3. Overlay ultra-recent 1-second ticks from cache_current_price.json
    if isinstance(current_price_items, dict) and current_price_items:
        for cand in cands:
            cur_item = current_price_items.get(cand)
            if isinstance(cur_item, list) and cur_item:
                series_24h = list(series_24h) + [
                    [pt[0], pt[1]]
                    for pt in cur_item
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]
                break
            elif cur_item is not None and not isinstance(cur_item, bool):
                raw_px = cur_item.get("price") or cur_item.get("c") if isinstance(cur_item, dict) else cur_item
                px_f = safe_pos_float(raw_px, 0.0)
                if px_f > 0.0 and series_24h:
                    last_pt = _parse_ts_px_row(series_24h[-1])
                    if last_pt is not None:
                        series_24h = list(series_24h) + [[last_pt[0] + 1000.0, px_f]]
                    break

    # 4. Normalize, sort, and deduplicate series_24h so callers always receive clean [tms, px] points
    if series_24h:
        cleaned_24h = [[pt[0], pt[1]] for row in series_24h if (pt := _parse_ts_px_row(row)) is not None]
        if len(cleaned_24h) >= 2:
            cleaned_24h.sort(key=lambda x: x[0])
            dedup_24h = [cleaned_24h[0]]
            for pt in cleaned_24h[1:]:
                if pt[0] > dedup_24h[-1][0]:
                    dedup_24h.append(pt)
                else:
                    dedup_24h[-1] = pt
            series_24h = dedup_24h if len(dedup_24h) >= 2 else []
        else:
            series_24h = []

    if vals_7d and series_24h:
        latest_px = safe_pos_float(series_24h[-1][1], default=None)
        if latest_px is not None and vals_7d[-1] != latest_px:
            vals_7d.append(latest_px)

    return series_24h, vals_7d


def compute_cross_btc_metrics(alt_series: Any, btc_series: Any) -> Dict[str, Any]:
    """Compute 24h ALT/BTC ratio range position, drawdown/bounce, and realized beta/correlation vs BTC."""
    if not isinstance(alt_series, list) or not isinstance(btc_series, list):
        return {}
    if len(alt_series) < 4 or len(btc_series) < 4:
        return {}
    try:
        def _clean_buckets(series: list) -> Dict[int, float]:
            return {
                int(pt[0] // 300000.0): pt[1]
                for row in series
                if (pt := _parse_ts_px_row(row)) is not None
            }

        a_map = _clean_buckets(alt_series)
        b_map = _clean_buckets(btc_series)
        if len(a_map) < 4 or len(b_map) < 4:
            return {}

        aligned = []
        for bk in sorted(a_map.keys()):
            b_px = b_map.get(bk)
            if b_px is None:
                b_px = b_map.get(bk - 1) or b_map.get(bk + 1)
            if b_px is not None and b_px > 0:
                aligned.append((bk, a_map[bk], b_px))
        if len(aligned) < 4:
            return {}

        ratios = [a_px / b_px for _, a_px, b_px in aligned if b_px > 0]
        if len(ratios) < 4:
            return {}

        r_cur = ratios[-1]
        r_low = min(ratios)
        r_high = max(ratios)
        out: Dict[str, Any] = {}
        if r_high > 0:
            out["alt_btc_dd_24h_high_pct"] = round((r_cur / r_high - 1.0) * 100.0, 2)
        if r_low > 0:
            out["alt_btc_bounce_24h_low_pct"] = round((r_cur / r_low - 1.0) * 100.0, 2)
        if r_high > r_low > 0:
            out["alt_btc_pos_24h_pct"] = round(((r_cur - r_low) / (r_high - r_low)) * 100.0, 1)

        rets_a = []
        rets_b = []
        for idx in range(1, len(aligned)):
            bk0, a0, b0 = aligned[idx - 1]
            bk1, a1, b1 = aligned[idx]
            if 0 < (bk1 - bk0) <= 48 and a0 > 0 and b0 > 0:
                rets_a.append(a1 / a0 - 1.0)
                rets_b.append(b1 / b0 - 1.0)
        n_r = len(rets_b)
        if n_r >= 6:
            mean_a = sum(rets_a) / n_r
            mean_b = sum(rets_b) / n_r
            cov = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(rets_a, rets_b)) / n_r
            var_a = sum((ra - mean_a) ** 2 for ra in rets_a) / n_r
            var_b = sum((rb - mean_b) ** 2 for rb in rets_b) / n_r
            if var_b > 1e-12:
                out["beta_btc_24h"] = round(cov / var_b, 2)
                if var_a > 1e-12:
                    corr = cov / math.sqrt(var_a * var_b)
                    out["corr_btc_24h"] = round(max(-1.0, min(1.0, corr)), 2)
        return out
    except Exception:
        return {}


def extract_symbol_recent_fills(orders_list: list) -> Optional[Dict[str, Any]]:
    """Extract last action side, active cycle avg BUY price, last BUY/SELL fills, and recent tranches."""
    if not isinstance(orders_list, list) or not orders_list:
        return None
    valid = [o for o in orders_list if isinstance(o, dict)]
    if not valid:
        return None
    from providers.base import (
        extract_order_price,
        extract_order_qty,
        extract_order_side,
        extract_timestamp_ms,
    )

    ordered = sorted(
        enumerate(valid),
        key=lambda item: (safe_nonneg_ts(extract_timestamp_ms(item[1], default=None), 0.0), item[0]),
    )
    chron = [o for _, o in ordered]
    last_buy = next((o for o in reversed(chron) if extract_order_side(o) == "BUY"), None)
    last_sell = next((o for o in reversed(chron) if extract_order_side(o) == "SELL"), None)
    last_any = chron[-1]

    def _norm_fill(o: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Any]:
        p_f = extract_order_price(o, default=0.0)
        q_f = extract_order_qty(o, default=0.0)
        ts_ms = extract_timestamp_ms(o, default=None)
        ts_raw = ts_ms if ts_ms is not None else (o.get("timestamp") if o.get("timestamp") is not None else o.get("time"))
        return (round(p_f, 6) if p_f > 0 else None), (q_f if q_f > 0 else None), ts_raw

    cost_qty = 0.0
    active_avg = 0.0
    buys_fmt: list[Dict[str, Any]] = []
    sells_fmt: list[Dict[str, Any]] = []
    for o in chron:
        side_o = extract_order_side(o)
        p_f, q_f, ts_raw = _norm_fill(o)
        if side_o == "BUY":
            buys_fmt.append({"price": p_f, "qty": q_f, "ts_ms": ts_raw})
            if q_f is not None and p_f is not None:
                new_cost_qty = cost_qty + q_f
                active_avg = ((cost_qty * active_avg) + (q_f * p_f)) / new_cost_qty if new_cost_qty > 0 else 0.0
                cost_qty = new_cost_qty
        elif side_o == "SELL":
            sells_fmt.append({"price": p_f, "qty": q_f, "ts_ms": ts_raw})
            if q_f is not None:
                if q_f >= cost_qty * 0.98 or cost_qty - q_f <= 1e-12:
                    cost_qty = 0.0
                    active_avg = 0.0
                else:
                    cost_qty -= q_f

    lb_norm = _norm_fill(last_buy) if last_buy else (None, None, None)
    ls_norm = _norm_fill(last_sell) if last_sell else (None, None, None)
    return {
        "last_action_side": extract_order_side(last_any) or last_any.get("side"),
        "active_avg_buy_px": round(active_avg, 6) if active_avg > 0 else None,
        "last_buy": (
            {"price": lb_norm[0], "qty": lb_norm[1], "ts_ms": lb_norm[2]}
            if last_buy else None
        ),
        "last_sell": (
            {"price": ls_norm[0], "qty": ls_norm[1], "ts_ms": ls_norm[2]}
            if last_sell else None
        ),
        "recent_buys": buys_fmt[-4:],
        "recent_sells": sells_fmt[-3:],
    }




def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) to provide deep context to the LLM."""
    if cachedb_dir == "cachedb":
        from cachepaths import get_cachedb_dir
        cachedb_dir = get_cachedb_dir()
    telemetry: Dict[str, Any] = {}
    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = get_orderbook_collector().fetch(symbol, allow_network=False)
        if ob_snap is not None:
            imb_f = safe_float(obj_attr(ob_snap, "imbalance_ratio", None), default=None)
            if imb_f is not None:
                bid_pct = max(0, min(100, int(round(imb_f * 100.0))))
                ask_pct = 100 - bid_pct
                telemetry["Orderbook Depth Imbalance"] = f"{imb_f:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
            for side_lbl, usd_attr, px_attr in (
                ("Ask", "largest_ask_wall_usd", "largest_ask_wall_price"),
                ("Bid", "largest_bid_wall_usd", "largest_bid_wall_price"),
            ):
                w_usd = safe_pos_float(obj_attr(ob_snap, usd_attr, 0.0), 0.0)
                w_px = safe_pos_float(obj_attr(ob_snap, px_attr, 0.0), 0.0)
                telemetry[f"Orderbook {side_lbl} Wall Ahead"] = (
                    f"${w_usd:,.0f} @ ${w_px:,.2f}"
                    if w_usd > 100_000.0 and w_px > 0.0
                    else "None detected within 2%"
                )
    except Exception:
        pass

    # 2. Perpetual Derivatives: Funding Rate & Open Interest (Pillar 2)
    try:
        from external_order_guard import get_derivatives_collector
        f_snap = get_derivatives_collector().fetch(symbol, allow_network=False)
        if f_snap is not None:
            fr_f = safe_float(obj_attr(f_snap, "funding_rate", None), default=None)
            if fr_f is not None:
                telemetry["Perpetual Funding Rate"] = f"{fr_f * 100:+.4f}% (8h)"
            oi_f = safe_pos_float(obj_attr(f_snap, "open_interest_usd", None), 0.0)
            if oi_f > 0.0:
                telemetry["Derivatives Open Interest"] = f"${oi_f:,.0f}"
    except Exception:
        pass

    # 3. Whale Positioning: Top Traders & Aggressive Flow (Pillar 2)
    try:
        from external_order_guard import get_whale_collector
        w_snap = get_whale_collector().fetch(symbol, allow_network=False)
        if w_snap is not None:
            lp_f = safe_float(obj_attr(w_snap, "top_traders_long_pct", None), default=None)
            lr_f = safe_float(obj_attr(w_snap, "top_traders_long_ratio", None), default=None)
            if lp_f is not None and lr_f is not None:
                telemetry["Whale Positioning"] = f"{lp_f * 100.0:.1f}% Top Traders Long (L/S ratio: {lr_f:.2f})"
            tr_f = safe_float(obj_attr(w_snap, "taker_buy_sell_ratio", None), default=None)
            if tr_f is not None:
                telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {tr_f:.2f} (regime: {obj_attr(w_snap, 'divergence_regime', 'neutral')})"
    except Exception:
        pass

    # 4. Technical Trend & Regime (Pillars 0 & 1)
    if regime_context is not None:
        try:
            trend = obj_attr(regime_context, "resolved_trend", None) or obj_attr(regime_context, "regime", None)
            dec = obj_attr(regime_context, "decision", None)
            gf = safe_float(obj_attr(dec, "gradient", None), default=None) if dec is not None else None
            ef = safe_float(obj_attr(dec, "epsilon", None), default=None) if dec is not None else None
            trend_str = f"{trend}" if trend else "unspecified"
            if gf is not None and ef is not None:
                trend_str += f" (gradient={gf:+.4f}, epsilon={ef:.4f})"
            telemetry["Technical Trend Regime"] = trend_str
        except Exception:
            pass

    df = safe_pos_float(dur_sec, 0.0)
    if df > 0.0:
        telemetry["Trend Duration"] = f"{df / 3600.0:.1f} hours"

    # 4B. 24h Multi-Horizon Price Evolution & Dynamic Multi-Window Slopes (shared with Reconciler)
    from cachepaths import cache_symbol_variants, canonical_base_asset
    sym_variants = cache_symbol_variants(symbol)
    try:
        cur_p_data = read_recovering_previous(os.path.join(cachedb_dir, "cache_current_price.json"), json.loads, default=None)
        cur_p_items = cur_p_data.get("items", {}) if isinstance(cur_p_data, dict) and isinstance(cur_p_data.get("items"), dict) else {}
        alt_series_24h, _ = load_symbol_price_series(
            cachedb_dir, symbol, current_price_items=cur_p_items, include_7d=False
        )
        m24 = compute_24h_price_metrics(alt_series_24h) if alt_series_24h else {}
        if m24:
            h_parts = []
            for micro_lbl, micro_k in (("1m", "chg_1m_pct"), ("3m", "chg_3m_pct"), ("5m", "chg_5m_pct")):
                if micro_k in m24 and m24.get(micro_k) is not None:
                    h_parts.append(f"{micro_lbl}={float(m24[micro_k]):+.2f}%")
            h_parts.extend([
                f"15m={m24.get('chg_15m_pct', 0):+.2f}%",
                f"30m={m24.get('chg_30m_pct', 0):+.2f}%",
                f"1h={m24.get('chg_1h_pct', 0):+.2f}%",
                f"3h={m24.get('chg_3h_pct', 0):+.2f}%",
            ])
            for opt_lbl, opt_k in (("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct")):
                if opt_k in m24 and m24.get(opt_k) is not None:
                    h_parts.append(f"{opt_lbl}={float(m24[opt_k]):+.2f}%")
            rng_str = f"24h Range=[${m24.get('low_24h')}, ${m24.get('high_24h')}]"
            geom_parts = []
            if m24.get("pos_24h_pct") is not None:
                geom_parts.append(f"pos={m24['pos_24h_pct']}%")
            if m24.get("dd_24h_high_pct") is not None:
                geom_parts.append(f"dd_high={float(m24['dd_24h_high_pct']):+.2f}%")
            if m24.get("bounce_24h_low_pct") is not None:
                geom_parts.append(f"bounce_low={float(m24['bounce_24h_low_pct']):+.2f}%")
            if m24.get("span_24h_pct") is not None:
                geom_parts.append(f"span={m24['span_24h_pct']}%")
            if geom_parts:
                rng_str += f" ({', '.join(geom_parts)})"
            if m24.get("recent_tape_15m"):
                rng_str += f", tape_15m=[{', '.join(m24['recent_tape_15m'])}]"
            telemetry["24h Multi-Horizon Price Evolution"] = f"{', '.join(h_parts)}, {rng_str}"

            if canonical_base_asset(symbol) != "BTC" and not str(symbol).upper().endswith(("_EQ", "_US_EQ")):
                btc_series_24h, _ = load_symbol_price_series(
                    cachedb_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False
                )
                btc_m24 = compute_24h_price_metrics(btc_series_24h) if btc_series_24h else {}
                if btc_m24:
                    vs_parts = [
                        f"{h_lbl}={float(m24[h_k]) - float(btc_m24[h_k]):+.2f}%"
                        for h_lbl, h_k in (
                            ("5m", "chg_5m_pct"),
                            ("15m", "chg_15m_pct"),
                            ("1h", "chg_1h_pct"),
                            ("3h", "chg_3h_pct"),
                            ("6h", "chg_6h_pct"),
                            ("12h", "chg_12h_pct"),
                            ("24h", "chg_24h_pct"),
                        )
                        if h_k in m24 and h_k in btc_m24
                    ]
                    cross_btc = compute_cross_btc_metrics(alt_series_24h, btc_series_24h)
                    cross_parts = []
                    if cross_btc.get("alt_btc_pos_24h_pct") is not None:
                        cross_parts.append(f"alt_btc_pos_24h={cross_btc['alt_btc_pos_24h_pct']}%")
                    if cross_btc.get("alt_btc_dd_24h_high_pct") is not None:
                        cross_parts.append(f"alt_btc_dd_high={float(cross_btc['alt_btc_dd_24h_high_pct']):+.2f}%")
                    if cross_btc.get("alt_btc_bounce_24h_low_pct") is not None:
                        cross_parts.append(f"alt_btc_bounce_low={float(cross_btc['alt_btc_bounce_24h_low_pct']):+.2f}%")
                    if vs_parts or cross_parts:
                        rel_str = f"vs_btc({', '.join(vs_parts)})" if vs_parts else ""
                        if cross_parts:
                            rel_str = f"{rel_str} | {', '.join(cross_parts)}" if rel_str else ", ".join(cross_parts)
                        telemetry["Relative Strength vs BTC (24h Benchmark)"] = rel_str
    except Exception:
        pass

    try:
        st_path = os.path.join(cachedb_dir, "cache_short_trend.json")
        st_data = read_recovering_previous(st_path, json.loads, default=None)
        if isinstance(st_data, dict):
            st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
            tinfo = next((st_items.get(cand) for cand in sym_variants if isinstance(st_items.get(cand), dict)), None)
            if isinstance(tinfo, dict):
                dyn = tinfo.get("dynamic", {}) if isinstance(tinfo.get("dynamic"), dict) else {}

                def _safe_slope(w_key: str, fallback_obj: Optional[Dict[str, Any]] = None) -> float:
                    w_obj = dyn.get(w_key)
                    if not isinstance(w_obj, dict) or w_obj.get("slope_full") is None:
                        w_obj = fallback_obj
                    return safe_float(w_obj.get("slope_full", 0.0), 0.0) if isinstance(w_obj, dict) else 0.0

                fast_fallback = tinfo if tinfo.get("slope_full") is not None else (
                    dyn.get("90") if isinstance(dyn.get("90"), dict) else None
                )
                telemetry["Dynamic Multi-Window Slopes"] = (
                    f"3.7m={_safe_slope('222', fast_fallback):+.5f}, 15m={_safe_slope('900'):+.5f}, 2.5h={_safe_slope('9000'):+.5f}"
                )
    except Exception:
        pass

    try:
        fills = {}
        for ord_file in ("cache_order.json", "cache_trade_kraken.json"):
            ord_path = os.path.join(cachedb_dir, ord_file)
            ord_data = read_recovering_previous(ord_path, json.loads, default=None)
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) else {}
            if not isinstance(o_items, dict):
                continue
            orders_for_sym = next((o_items.get(cand, []) for cand in sym_variants if o_items.get(cand)), [])
            fills = extract_symbol_recent_fills(orders_for_sym)
            if fills:
                break
        if fills:
            lb = fills.get("last_buy") or {}
            ls = fills.get("last_sell") or {}
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb.get('qty')} @ {lb.get('price')}, "
                f"Last SELL={ls.get('qty')} @ {ls.get('price')}"
            )
    except Exception:
        pass

    # 5. Macro Sentiment Bias & Fear/Greed Index (Pillar 3)
    try:
        m_data = read_recovering_previous(os.path.join(cachedb_dir, "sentiment_eval.json"), json.loads, default=None)
        if isinstance(m_data, dict):
            bias = str(m_data.get("market_bias") or "").strip()
            risk = str(m_data.get("risk_level") or "").strip()
            action = str(m_data.get("recommended_action") or "").strip()
            if bias or risk or action:
                telemetry["Macro Sentiment Bias"] = f"{bias or 'UNKNOWN'} (Risk: {risk or 'UNKNOWN'}, Recommendation: {action or 'NONE'})"
    except Exception:
        pass

    try:
        fg_data = read_recovering_previous(os.path.join(cachedb_dir, "fear_greed_collect.json"), json.loads, default=None)
        if isinstance(fg_data, dict):
            val = fg_data.get("value", "")
            label = fg_data.get("classification", "")
            if val is not None and not isinstance(val, bool) and str(val).strip() != "":
                lbl_str = str(label).strip() if (label is not None and not isinstance(label, bool)) else ""
                telemetry["Fear & Greed Index"] = f"{val} ({lbl_str})"
    except Exception:
        pass

    # 6. Geopolitical Threat Shield (Pillar 4)
    try:
        g_data = read_recovering_previous(os.path.join(cachedb_dir, "geopolitical_eval.json"), json.loads, default=None)
        if isinstance(g_data, dict):
            level = g_data.get("threat_level", "")
            if level is not None and not isinstance(level, bool) and str(level).strip() != "":
                telemetry["Geopolitical Threat Level"] = str(level).strip()
    except Exception:
        pass

    return telemetry


DEFAULT_HIGH_STAKE_CACHE_TTL_SEC = 600.0
_HIGH_STAKE_DECISION_CACHE: Dict[tuple[str, str, int], tuple[float, GuardDecision]] = {}


def is_llm_guard_exempt(explicit_client: bool = False) -> bool:
    """True when LLM pre-trade guards are suppressed via environment flags (backtest/replay/disabled)."""
    exp_b = _parse_bool_flag(explicit_client, False)
    return not exp_b and any(
        _parse_bool_flag(os.environ.get(k))
        for k in ("DISABLE_GEMINI_LLM", "BACKTEST_MODE", "REPLAY_MODE")
    )


def evaluate_llm_guard_response(
    guard_name: str,
    action_label: str,
    resp: Optional[Dict[str, Any]],
    *,
    fallback_action: str,
    notional_eur: float,
) -> GuardDecision:
    """SSOT converter from a pre-trade LLM JSON response (or timeout None) into a canonical GuardDecision."""
    safe_notional = safe_nonneg_float(notional_eur, 0.0)
    fb_norm = str(fallback_action or "allow").strip().lower()
    if not resp or not isinstance(resp, dict):
        if fb_norm in ("veto", "block"):
            return GuardDecision.veto(guard_name, "llm_query_timeout_fail_closed", notional_eur=safe_notional)
        if fb_norm == "downscale":
            return GuardDecision.downscale(
                guard_name, scale=0.5, reason="llm_query_timeout_downscaled", notional_eur=safe_notional
            )
        return GuardDecision.allow(guard_name, "llm_query_timeout_fallback_allowed", notional_eur=safe_notional)

    raw_dec = resp.get("decision")
    decision_str = str(raw_dec if (isinstance(raw_dec, str) and raw_dec.strip()) else "APPROVED").upper().strip()
    raw_reason = resp.get("reason")
    reason = str(raw_reason).strip() if (isinstance(raw_reason, str) and raw_reason.strip()) else "evaluated by LLM"
    default_scale = 0.5 if decision_str == "DOWNSCALE" else 1.0
    scale = clean_scale(resp.get("suggested_scale"), default_scale, allow_percentage=True)

    if decision_str in ("REJECTED", "REJECT", "VETO", "BLOCK", "DENIED", "DENY"):
        return GuardDecision.veto(
            guard_name,
            f"Gemini vetoed {action_label}: {reason}",
            notional_eur=safe_notional,
            gemini_reason=reason,
        )
    if decision_str == "DOWNSCALE":
        if scale <= 0.0 or scale >= 1.0:
            scale = 0.5
        return GuardDecision.downscale(
            guard_name,
            scale=max(0.1, min(1.0, scale)),
            reason=f"Gemini downscaled {action_label}: {reason}",
            notional_eur=safe_notional,
            gemini_reason=reason,
        )
    return GuardDecision.allow(
        guard_name,
        f"Gemini approved {action_label}: {reason}",
        notional_eur=safe_notional,
        gemini_reason=reason,
    )


class HighStakeGuard:
    """Invokes LLM reasoning before executing large BUY purchases (>= 1000 EUR).

    Sub-1000 EUR orders pass immediately with zero overhead (0 ms).
    Orders exceeding the threshold undergo real-time LLM risk vetting against
    market psychology, whale positioning, and liquidation cascades.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        min_notional_eur: float = DEFAULT_MIN_NOTIONAL_USD,
        min_notional_usd: Optional[float] = None,
        timeout_sec: float = 25.0,
        fallback_action: str = "allow",  # "allow" | "downscale" | "veto"
        cache_ttl_sec: Optional[float] = None,
    ) -> None:
        self._explicit_client = gemini_client is not None
        self.gemini_client = gemini_client or GeminiClient()
        raw_min = min_notional_usd if min_notional_usd is not None else min_notional_eur
        resolved_min = safe_pos_float(raw_min, DEFAULT_MIN_NOTIONAL_USD)
        self.min_notional_usd = resolved_min
        self.min_notional_eur = resolved_min
        self.timeout_sec = safe_pos_float(timeout_sec, 25.0)
        self.fallback_action = str(fallback_action or "allow").strip().lower()
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            self.cache_ttl_sec = safe_nonneg_float(cache_ttl_sec, 0.0)
        elif not self._explicit_client and getattr(self.gemini_client, "_custom_runner", None) is None:
            self.cache_ttl_sec = DEFAULT_HIGH_STAKE_CACHE_TTL_SEC
        else:
            self.cache_ttl_sec = 0.0

    def check(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        notional_eur: Optional[float] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order through LLM if value exceeds threshold."""
        side_norm = normalize_order_side(side)
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow("HighStakeGuard", "unsupported_side_exempt")

        notional = safe_pos_float(notional_eur, 0.0)
        if notional <= 0.0 and (notional_eur is None or safe_float(notional_eur, float("nan")) <= 0.0):
            pf = safe_pos_float(price, 0.0)
            qf = safe_pos_float(qty, 0.0)
            if pf > 0.0 and qf > 0.0:
                notional = pf * qf
        notional_f = safe_float(notional, default=None)
        if notional_f is None or notional_f < self.min_notional_eur:
            return GuardDecision.allow(
                "HighStakeGuard",
                f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} EUR)",
                notional_eur=notional,
            )

        if is_llm_guard_exempt(self._explicit_client):
            return GuardDecision.allow("HighStakeGuard", "backtest_mode_exempt", notional_eur=notional)

        now_ts = resolve_now_ts(now)
        cache_key = (clean_symbol_key(symbol), side_norm, int(notional // 100.0))
        if self.cache_ttl_sec > 0:
            cached = _HIGH_STAKE_DECISION_CACHE.get(cache_key)
            if isinstance(cached, (tuple, list)) and len(cached) >= 2 and is_cache_fresh(cached[0], now_ts=now_ts, ttl_sec=self.cache_ttl_sec):
                return cached[1]

        effective_telemetry = telemetry if isinstance(telemetry, dict) else collect_pretrade_telemetry(
            symbol, regime_context=regime_context, dur_sec=dur_sec
        )
        context_items = [
            f"- Symbol: {symbol}",
            f"- Side: {side_norm}",
            f"- Requested Price: {price}",
            f"- Requested Quantity: {qty}",
            f"- Total Notional Value: €{notional:,.2f} EUR",
        ]
        if isinstance(effective_telemetry, dict) and effective_telemetry:
            context_items.extend(f"- {k}: {v}" for k, v in effective_telemetry.items())

        context_str = "\n".join(context_items)
        if side_norm == "BUY":
            action_desc = "HIGH-VALUE BUY order"
            risk_context = (
                "evaluating whether this entry is sound or if there are critical red flags "
                "(e.g., euphoric top buying, aggressive whale distribution, severe funding imbalance, or active cascade)."
            )
            policy_rules = (
                "CORE RISK POLICIES & PRIORITY RULES:\n"
                "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions (% Top Traders Long) and Orderbook Bid Depth. Static long holdings can be trapped or liquidated. If Whale Taker Flow is below 0.85 (net sell aggression), you must NEVER issue an unrestricted APPROVED. If Whale Taker Flow is below 0.80 in a FLAT or BEAR market, you must issue REJECTED (or DOWNSCALE <= 0.25 only if at verified structural support).\n"
                "2. ORDERBOOK SUPPORT LIMITATION: Bid walls and orderbook depth are passive and can be pulled or absorbed by aggressive market selling. Do NOT use orderbook depth alone to override persistent taker selling.\n"
                "3. MACRO & GEOPOLITICAL RISK: If Macro Sentiment recommends CAUTION or Geopolitical threat is elevated, do not fight negative taker flow.\n\n"
            )
        else:
            action_desc = "HIGH-VALUE SELL order"
            risk_context = (
                "evaluating whether this sale is sound or if there are critical red flags "
                "(e.g., dumping directly into major whale support/walls, panic-selling at local bottom during whale accumulation, or excessive slippage)."
            )
            policy_rules = (
                "CORE RISK POLICIES & PRIORITY RULES:\n"
                "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions and Orderbook Ask Depth. Static short holdings can be squeezed or liquidated. If Whale Taker Flow is above 1.18 (net buy aggression), you must NEVER issue an unrestricted APPROVED for a non-stop-loss SELL into strength. If Whale Taker Flow is above 1.25 in a FLAT or BULL market during whale accumulation or short squeeze, you must issue REJECTED (or DOWNSCALE <= 0.25 only if at verified structural resistance).\n"
                "2. ORDERBOOK RESISTANCE LIMITATION: Ask walls and orderbook ask depth are passive and can be pulled or absorbed by aggressive market buying. Do NOT use orderbook ask depth alone to override persistent taker buying.\n"
                "3. CAPITULATION & SQUEEZE PROTECTION: Do not approve panic-selling into extreme negative funding or active whale accumulation at local bottoms.\n\n"
            )

        prompt = (
            "You are a principal quantitative crypto risk officer. "
            f"An automated trading bot is about to place a {action_desc}:\n\n"
            f"{context_str}\n\n"
            f"Because this order exceeds €{self.min_notional_eur:,.2f} EUR, you must evaluate if executing "
            f"{risk_context}\n\n"
            f"{policy_rules}"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "decision": "APPROVED" | "REJECTED" | "DOWNSCALE",\n'
            '  "suggested_scale": 0.0 to 1.0,\n'
            '  "reason": "Concise 1-2 sentence rationale for the decision"\n'
            "}"
        )

        thread_title = f"{topic_spec('guard_high_stake').title} — {side_norm} {symbol} €{notional:,.2f}"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=self.timeout_sec,
            thread_title=thread_title,
            topic_key="guard_high_stake",
        )
        if not resp:
            logger.warning(
                "LLM high-stake query failed or timed out for %s €%.2f. Fallback: %s",
                symbol,
                notional,
                self.fallback_action,
            )

        dec = evaluate_llm_guard_response(
            "HighStakeGuard",
            f"high-stake {side_norm}",
            resp or None,
            fallback_action=self.fallback_action,
            notional_eur=notional,
        )
        if self.cache_ttl_sec > 0:
            _HIGH_STAKE_DECISION_CACHE[cache_key] = (now_ts, dec)
        return dec
