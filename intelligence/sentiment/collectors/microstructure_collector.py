"""Spot/perpetual microstructure, 24h/7d price series loader, and liquidation cluster collector."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from botcore import extract_current_price_item
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    legacy_filenames_for,
)
from intelligence.external.collectors.orderbook_depth import round_spread_bps
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    obj_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from intelligence.sentiment.collectors.portfolio_risk_tca import extract_symbol_recent_fills
from intelligence.sentiment.collectors.quant_price_evaluators import (
    _WINDOW_TREND_SPECS,
    _auto_px_decimals,
    _build_vp_position_dict,
    _compute_horizon_range_dict,
    _parse_ts_px_row,
    _prices_diverged,
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_7d_range_metrics,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
    compute_swing_wave_metrics,
    compute_volume_profile_metrics,
    pct_delta,
    range_position_pct,
    round_opt_px,
)
from instrument_registry import casefold_get
from state_io import (
    read_json_dict_recovering as read_json_cached,
    read_tail_jsonl_records,
    resolve_cachedb_filepath,
)

logger = logging.getLogger("intelligence.sentiment.microstructure_collector")

_MICRO_PRICE_ANCHOR_KEYS: frozenset[str] = frozenset({
    "best_bid", "best_ask", "ob_mid_price", "bid_wall_px", "ask_wall_px",
    "long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px",
    "long_liq_dist_pct", "short_liq_dist_pct", "long_liq_deep_dist_pct", "short_liq_deep_dist_pct",
})


def _extract_cached_items_series(data: Any, cands: list[str]) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    raw_items = data.get("items") if "items" in data else data
    if isinstance(raw_items, dict):
        for k_cand in cands:
            if isinstance((ser := casefold_get(raw_items, k_cand, [])), list) and ser:
                return ser
        if "items" in data and len(raw_items) == 1 and cands:
            only_k, first_val = next(iter(raw_items.items()))
            if (
                isinstance(first_val, list)
                and canonical_base_asset(only_k) == canonical_base_asset(cands[0])
                and compatible_quote_suffixes(only_k) == compatible_quote_suffixes(cands[0])
            ):
                return first_val
    elif isinstance(raw_items, list):
        return raw_items
    return []


def load_symbol_price_series(
    cachedb_dir: str,
    symbol: str,
    current_price_items: Optional[Dict[str, Any]] = None,
    *,
    include_7d: bool = True,
    quotes: Optional[tuple[str, ...]] = None,
) -> Tuple[list, list]:
    """Load and merge high-density 24h & 7d price series from cache_24_price, cache_365_price, cache_sparse_price, and cache_current_price."""
    eff_quotes = quotes if quotes is not None else compatible_quote_suffixes(symbol)
    cands = [
        c for c in cache_symbol_variants(symbol, include_original=True, include_base="USD" in eff_quotes, quotes=eff_quotes)
        if isinstance(c, str) and "/" not in c
    ]
    series_24h: list = []
    vals_7d: list = []

    def _cand_paths(*basenames: str) -> list[str]:
        return [
            p
            for bn in basenames
            for full_p in (str(resolve_cachedb_filepath(os.path.join(cachedb_dir, bn))),)
            for p in (full_p, *legacy_filenames_for(full_p))
        ]

    # 1. Try primary 24h dense cache (cache_24_price_<SYM>.json and legacy variants)
    p0: Optional[Tuple[float, float]] = None
    p1: Optional[Tuple[float, float]] = None
    span_24h_ms = 0.0
    for cand in cands:
        for p24_path in _cand_paths(f"cache_24_price_{cand}.json"):
            if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                continue
            ser = _extract_cached_items_series(read_json_cached(p24_path), cands)
            if len(ser) >= 2:
                if not series_24h:
                    series_24h = list(ser)
                    p0 = _parse_ts_px_row(series_24h[0])
                    p1 = _parse_ts_px_row(series_24h[-1])
                    span_24h_ms = max(0.0, p1[0] - p0[0]) if (p0 and p1) else 0.0
                elif p0 is not None:
                    sib_p0 = _parse_ts_px_row(ser[0])
                    sib_p1 = _parse_ts_px_row(ser[-1])
                    if sib_p0 and sib_p1 and sib_p0[0] < p0[0] and not _prices_diverged(sib_p1[1], p0[1]):
                        older_pts = [r for r in ser if (pt := _parse_ts_px_row(r)) is not None and pt[0] < p0[0]]
                        if older_pts:
                            series_24h = older_pts + series_24h
                            p0 = _parse_ts_px_row(series_24h[0])
                            p1 = _parse_ts_px_row(series_24h[-1])
                            span_24h_ms = max(0.0, p1[0] - p0[0]) if (p0 and p1) else 0.0
                break
        if series_24h and span_24h_ms >= 3600000.0 * 18:
            break

    # 2. Supplement 24h (when < 18h span) and extract 7d history from cache_sparse_price_<SYM> / cache_365_price_<SYM>
    need_24h_supplement = span_24h_ms < 3600000.0 * 18
    archive_pts: list = []
    sparse_7d_pts: list = []
    cands_upper = {str(c).strip().upper() for c in cands if isinstance(c, str) and str(c).strip()}

    def _read_arch_file(arch_path: str, max_b: int = 1024 * 1024) -> list:
        if not os.path.exists(arch_path):
            return []
        try:
            if arch_path.endswith(".json"):
                aser = _extract_cached_items_series(read_json_cached(arch_path), cands)
                return list(aser) if len(aser) >= 2 else []
            parsed_rows = []
            for rec in read_tail_jsonl_records(arch_path, limit=None, max_bytes=max_b):
                if isinstance(rec, dict):
                    rec_s = rec.get("s")
                    if isinstance(rec_s, str) and rec_s.strip() and cands_upper and rec_s.strip().upper() not in cands_upper:
                        continue
                    item = rec.get("i")
                else:
                    item = rec if isinstance(rec, list) else None
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    parsed_rows.append([item[0], item[1]])
            return parsed_rows if len(parsed_rows) >= 2 else []
        except Exception:
            return []

    for cand in (cands if (include_7d or need_24h_supplement) else ()):
        primary_names = (
            (f"cache_sparse_price_{cand}.jsonl", f"cache_365_price_{cand}.jsonl", f"cache_365_price_{cand}.json")
            if not need_24h_supplement
            else (f"cache_365_price_{cand}.jsonl", f"cache_365_price_{cand}.json", f"cache_sparse_price_{cand}.jsonl")
        )
        cand_arch: list = []
        for arch_path in _cand_paths(*primary_names):
            if rows := _read_arch_file(arch_path):
                cand_arch = rows
                break
        if cand_arch:
            if not archive_pts:
                archive_pts = cand_arch
            else:
                cur_a0 = _parse_ts_px_row(archive_pts[0])
                sib_a0, sib_a1 = _parse_ts_px_row(cand_arch[0]), _parse_ts_px_row(cand_arch[-1])
                if cur_a0 and sib_a0 and sib_a1 and sib_a0[0] < cur_a0[0] and not _prices_diverged(sib_a1[1], cur_a0[1]):
                    older_arch = [r for r in cand_arch if (pt := _parse_ts_px_row(r)) is not None and pt[0] < cur_a0[0]]
                    if older_arch:
                        archive_pts = older_arch + archive_pts
            a0, a1 = _parse_ts_px_row(archive_pts[0]), _parse_ts_px_row(archive_pts[-1])
            arch_span_ms = (a1[0] - a0[0]) if (a0 and a1) else 0.0
            if include_7d and arch_span_ms < 5.0 * 86400.0 * 1000.0 and not sparse_7d_pts:
                for sp_path in _cand_paths(f"cache_sparse_price_{cand}.jsonl"):
                    if sp_rows := _read_arch_file(sp_path, max_b=512 * 1024):
                        sp0, sp1 = _parse_ts_px_row(sp_rows[0]), _parse_ts_px_row(sp_rows[-1])
                        if sp0 and sp1 and (sp1[0] - sp0[0]) > arch_span_ms:
                            sparse_7d_pts = sp_rows
                            break
            if arch_span_ms >= 5.0 * 86400.0 * 1000.0 or sparse_7d_pts:
                break

    if archive_pts:
        try:
            if (last_pt := _parse_ts_px_row(archive_pts[-1])) is not None:
                cutoff_24h, cutoff_7d = last_pt[0] - 86400.0 * 1000.0, last_pt[0] - 7.0 * 86400.0 * 1000.0
                pts_24, pts_7d_vals = [], []
                for row in archive_pts:
                    if (pt := _parse_ts_px_row(row)) is not None:
                        if pt[0] >= cutoff_7d:
                            pts_7d_vals.append(pt[1])
                        if need_24h_supplement and pt[0] >= cutoff_24h:
                            pts_24.append([pt[0], pt[1]])
                if sparse_7d_pts and (sp_last := _parse_ts_px_row(sparse_7d_pts[-1])) is not None:
                    sp_cutoff_7d = sp_last[0] - 7.0 * 86400.0 * 1000.0
                    sp_vals = [pt[1] for row in sparse_7d_pts if (pt := _parse_ts_px_row(row)) is not None and pt[0] >= sp_cutoff_7d]
                    if len(sp_vals) >= 2:
                        pts_7d_vals = sp_vals
                if len(pts_7d_vals) >= 2:
                    vals_7d = pts_7d_vals
                if need_24h_supplement and pts_24:
                    s0_pt = _parse_ts_px_row(series_24h[0]) if series_24h else None
                    if s0_pt is None or not _prices_diverged(pts_24[-1][1], s0_pt[1]):
                        series_24h = (pts_24 + series_24h) if series_24h else pts_24
        except Exception:
            pass

    # 3. Overlay ultra-recent 1-second ticks from cache_current_price.json
    if isinstance(current_price_items, dict) and current_price_items:
        last_24_pt = _parse_ts_px_row(series_24h[-1]) if series_24h else None
        for cand in cands:
            cur_item = casefold_get(current_price_items, cand)
            if isinstance(cur_item, list) and cur_item:
                cand_pts = [[pt[0], pt[1]] for pt in cur_item if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                cand_last = _parse_ts_px_row(cand_pts[-1]) if cand_pts else None
                if last_24_pt is not None and cand_last is not None and _prices_diverged(last_24_pt[1], cand_last[1]):
                    continue
                series_24h = list(series_24h) + cand_pts
                break
            elif (px_f := extract_current_price_item(cur_item)) is not None and series_24h and last_24_pt is not None:
                if not _prices_diverged(last_24_pt[1], px_f):
                    series_24h = list(series_24h) + [[last_24_pt[0] + 1000.0, px_f]]
                    break

    # 4. Normalize, sort, and deduplicate series_24h so callers always receive clean [tms, px] (or [tms, px, vol]) points
    if series_24h:
        cleaned_24h = sorted(
            (
                ([pt[0], pt[1], vol_v] if len(row) >= 3 and (vol_v := safe_pos_float(row[2], default=None)) is not None else [pt[0], pt[1]])
                for row in series_24h if (pt := _parse_ts_px_row(row)) is not None
            ),
            key=lambda x: x[0],
        )
        if len(cleaned_24h) >= 2:
            dedup_24h = [cleaned_24h[0]]
            for pt in cleaned_24h[1:]:
                if pt[0] > dedup_24h[-1][0]:
                    dedup_24h.append(pt)
                else:
                    dedup_24h[-1] = [pt[0], pt[1], dedup_24h[-1][2]] if (len(pt) < 3 and len(dedup_24h[-1]) >= 3) else pt
            series_24h = dedup_24h if len(dedup_24h) >= 2 else []
        else:
            series_24h = []

    if vals_7d and series_24h and (latest_px := safe_pos_float(series_24h[-1][1], default=None)) is not None:
        if vals_7d[-1] != latest_px and not _prices_diverged(latest_px, vals_7d[-1]):
            vals_7d.append(latest_px)

    return series_24h, vals_7d


def _first_variant_dict(mapping: Any, variants: Any, default: Any = None) -> Any:
    """Return the first dict entry in `mapping` matching any candidate key in `variants` (case-insensitive)."""
    if not isinstance(mapping, dict):
        return default
    return next((val for cand in variants if isinstance((val := casefold_get(mapping, cand)), dict)), default)


def load_symbol_microstructure(
    cache_dir: str,
    sym: str,
    read_json_fn: Any = None,
    *,
    ob: Any = None,
    deriv: Any = None,
    whale: Any = None,
) -> Optional[Dict[str, Any]]:
    """Load and normalize orderbook depth, derivatives, and whale snapshot across symbol variants."""
    read_json_fn = read_json_cached if read_json_fn is None else read_json_fn
    sym_cands = cache_symbol_variants(sym, include_base=True)

    def _first_telemetry_dict(prefix: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        for cand in sym_cands:
            if isinstance((d := read_json_fn(os.path.join(cache_dir, f"{prefix}{cand}.json"))), dict):
                return d, cand
        return None, None

    if ob is None:
        ob, ob_cand = _first_telemetry_dict("orderbook_depth_")
    else:
        raw_ob_sym = obj_attr(ob, "symbol")
        ob_cand = raw_ob_sym.strip() if isinstance(raw_ob_sym, str) and raw_ob_sym.strip() else None
    if deriv is None:
        deriv, _ = _first_telemetry_dict("derivatives_telemetry_")
    if whale is None:
        whale, _ = _first_telemetry_dict("whale_snapshot_")
    if not (ob or deriv or whale):
        return None
    m_info: Dict[str, Any] = {"_source_cand": ob_cand} if ob_cand else {}
    if ob is not None:
        ob_same_quote = (not ob_cand) or (compatible_quote_suffixes(ob_cand) == compatible_quote_suffixes(sym))
        m_info["ob_imbalance"] = obj_attr(ob, "imbalance_ratio", obj_attr(ob, "ob_imbalance"))
        ob_mid = safe_pos_float(obj_attr(ob, "mid_price", obj_attr(ob, "ob_mid_price")), default=None)
        best_bid = safe_pos_float(obj_attr(ob, "best_bid"), default=None)
        best_ask = safe_pos_float(obj_attr(ob, "best_ask"), default=None)
        if (spread_bps := round_spread_bps(obj_attr(ob, "spread_bps"), best_bid=best_bid, best_ask=best_ask, mid_price=ob_mid)) is not None:
            m_info["spread_bps"] = spread_bps
        if ob_same_quote:
            for k_px, v_px in (("ob_mid_price", ob_mid), ("best_bid", best_bid), ("best_ask", best_ask)):
                if v_px is not None:
                    m_info[k_px] = v_px
        for k_src, k_dst in (
            ("bid_depth_usd", "bid_depth_usd"), ("ask_depth_usd", "ask_depth_usd"),
            ("depth_1pct_bid_usd", "depth_1pct_bid_usd"), ("depth_1pct_ask_usd", "depth_1pct_ask_usd"),
            ("largest_bid_wall_usd", "bid_wall_usd"), ("largest_ask_wall_usd", "ask_wall_usd"),
        ):
            val = safe_nonneg_float(obj_attr(ob, k_src, obj_attr(ob, k_dst) if k_src != k_dst else None), default=None)
            m_info[k_dst] = round(val, 0) if val is not None else None
        m_info["bid_wall_px"] = obj_attr(ob, "largest_bid_wall_price", obj_attr(ob, "bid_wall_px")) if ob_same_quote else None
        m_info["ask_wall_px"] = obj_attr(ob, "largest_ask_wall_price", obj_attr(ob, "ask_wall_px")) if ob_same_quote else None
    if deriv is not None:
        m_info["funding_rate"] = obj_attr(deriv, "funding_rate")
        m_info["pred_funding_rate"] = obj_attr(deriv, "predicted_funding_rate", obj_attr(deriv, "pred_funding_rate"))
        m_info["oi_usd"] = obj_attr(deriv, "open_interest_usd", obj_attr(deriv, "oi_usd"))
        m_info["mark_price"] = safe_pos_float(obj_attr(deriv, "mark_price"), default=None)
        m_info["index_price"] = safe_pos_float(obj_attr(deriv, "index_price"), default=None)
        m_info["perp_spot_basis_bps"] = safe_float(obj_attr(deriv, "perp_spot_basis_bps"), default=None)
        m_info["funding_annualized_pct"] = safe_float(obj_attr(deriv, "funding_annualized_pct"), default=None)
    if whale is not None:
        for dst_k, src_k, alt_k in (
            ("whale_long_pct", "top_traders_long_pct", "whale_long_pct"),
            ("whale_long_ratio", "top_traders_long_ratio", "whale_long_ratio"),
            ("taker_buy_sell_ratio", "taker_buy_sell_ratio", None),
            ("taker_buy_vol_usd", "taker_buy_vol_usd", None),
            ("taker_sell_vol_usd", "taker_sell_vol_usd", None),
            ("oi_1h_chg_pct", "open_interest_1h_change_pct", "oi_1h_chg_pct"),
            ("divergence_regime", "divergence_regime", None),
            ("taker_ratio_5m", "taker_ratio_5m", None),
            ("taker_ratio_1h", "taker_ratio_1h", None),
            ("taker_ratio_4h", "taker_ratio_4h", None),
            ("cvd_5m_usd", "cvd_5m_usd", None),
            ("cvd_1h_usd", "cvd_1h_usd", None),
            ("cvd_4h_usd", "cvd_4h_usd", None),
        ):
            val_w = obj_attr(whale, src_k, obj_attr(whale, alt_k) if alt_k else None)
            if val_w is not None or dst_k not in m_info:
                m_info[dst_k] = val_w
        if not safe_pos_float(m_info.get("oi_usd"), default=None) and (w_oi := safe_pos_float(obj_attr(whale, "open_interest_usd", obj_attr(whale, "oi_usd")), default=None)) is not None:
            m_info["oi_usd"] = w_oi
        if m_info.get("funding_rate") is None and (w_fr := safe_float(obj_attr(whale, "funding_rate"), default=None)) is not None:
            m_info["funding_rate"] = w_fr
    if m_info.get("funding_annualized_pct") is None and (fr_v := safe_float(m_info.get("funding_rate"), default=None)) is not None:
        m_info["funding_annualized_pct"] = round(fr_v * 3.0 * 365.0 * 100.0, 2)
    enrich_microstructure_cvd(m_info)
    return m_info


def enrich_microstructure_cvd(minfo: Dict[str, Any], pinfo: Optional[Dict[str, Any]] = None) -> None:
    """Populate 1% orderbook depth / $2k slippage and `cvd_15m_usd`, `cvd_15m_base`, `cvd_regime`, and cross-venue spot vs perp flow regime on `minfo`."""
    if not isinstance(minfo, dict):
        return
    b_bid = safe_pos_float(minfo.get("best_bid"), default=0.0) or 0.0
    b_ask = safe_pos_float(minfo.get("best_ask"), default=0.0) or 0.0
    if b_bid > 0.0 and b_ask >= b_bid:
        if not safe_pos_float(minfo.get("ob_mid_price"), default=None):
            minfo["ob_mid_price"] = round((b_bid + b_ask) / 2.0, 6)
        if minfo.get("spread_bps") is None and (
            sp_bps := round_spread_bps(None, best_bid=b_bid, best_ask=b_ask, mid_price=safe_pos_float(minfo.get("ob_mid_price"), default=None))
        ) is not None:
            minfo["spread_bps"] = sp_bps
    for d1_k, d_k in (("depth_1pct_bid_usd", "bid_depth_usd"), ("depth_1pct_ask_usd", "ask_depth_usd")):
        if not safe_pos_float(minfo.get(d1_k), default=None) and (d_f := safe_pos_float(minfo.get(d_k), default=None)):
            minfo[d1_k] = round(d_f * (1.0 / 1.5), 0)
    if minfo.get("est_slip_2k_bps") is None:
        d1_b = safe_pos_float(minfo.get("depth_1pct_bid_usd"), default=0.0) or 0.0
        d1_a = safe_pos_float(minfo.get("depth_1pct_ask_usd"), default=0.0) or 0.0
        eff_d1 = min(d1_b, d1_a) if (d1_b > 0.0 and d1_a > 0.0) else max(d1_b, d1_a)
        if eff_d1 > 0.0:
            sp_half = (safe_nonneg_float(minfo.get("spread_bps"), default=2.0) or 2.0) * 0.5
            minfo["est_slip_2k_bps"] = round(min(250.0, max(0.5, sp_half + (2000.0 / (2.0 * eff_d1)) * 100.0)), 1)
    cvd_usd = safe_float(minfo.get("cvd_15m_usd"), default=None)
    if cvd_usd is None:
        tb_v, ts_v = safe_nonneg_float(minfo.get("taker_buy_vol_usd"), default=None), safe_nonneg_float(minfo.get("taker_sell_vol_usd"), default=None)
        if tb_v is not None and ts_v is not None and (tb_v + ts_v) > 0.0:
            cvd_usd = minfo["cvd_15m_usd"] = round(tb_v - ts_v, 0)
    if cvd_usd is None:
        return
    p_dict = pinfo if isinstance(pinfo, dict) else {}
    m_px = (
        safe_pos_float(p_dict.get("price"), default=None)
        or safe_pos_float(minfo.get("ob_mid_price"), default=None)
        or ((b_bid + b_ask) / 2.0 if b_bid > 0.0 and b_ask >= b_bid else None)
    )
    if m_px is not None and m_px > 0.0:
        minfo["cvd_15m_base"] = round(cvd_usd / m_px, 3 if m_px >= 100.0 else (2 if m_px >= 1.0 else 1))
    ob_imb = safe_nonneg_float(minfo.get("ob_imbalance"), default=0.5)
    has_chg_15m = "chg_15m_pct" in p_dict and p_dict.get("chg_15m_pct") is not None
    chg_15m = safe_float(p_dict.get("chg_15m_pct"), default=0.0) or 0.0
    div_reg = str(minfo.get("divergence_regime") or minfo.get("div_regime") or "").strip().lower()
    if not minfo.get("cvd_regime") or pinfo is not None:
        if cvd_usd > 0.0:
            minfo["cvd_regime"] = ("ask_absorbing" if chg_15m <= 0.02 else "aggressive_buying") if has_chg_15m else ("ask_absorbing" if ob_imb <= 0.44 else "aggressive_buying")
        elif cvd_usd < 0.0:
            is_bid_absorbing = (
                (chg_15m >= 0.05 or (chg_15m >= -0.05 and (ob_imb >= 0.54 or div_reg in ("short_covering", "bullish_absorption"))))
                if has_chg_15m
                else (ob_imb >= 0.58 or div_reg in ("short_covering", "bullish_absorption"))
            )
            minfo["cvd_regime"] = "spot_absorbing" if is_bid_absorbing else "aggressive_market_dumps"
        else:
            minfo["cvd_regime"] = "balanced_flow"

    # Compute Spot vs Perp CVD divergence & Perp-Spot Basis Regime (Layer II.1)
    b_depth = safe_nonneg_float(minfo.get("bid_depth_usd"), default=0.0) or 0.0
    a_depth = safe_nonneg_float(minfo.get("ask_depth_usd"), default=0.0) or 0.0
    if minfo.get("spot_cvd_15m_usd") is None and (b_depth + a_depth) > 0.0:
        ob_net_depth = b_depth - a_depth
        vel_adj = (b_depth + a_depth) * (chg_15m / 100.0) * 2.5 if has_chg_15m else 0.0
        minfo["spot_cvd_15m_usd"] = round(ob_net_depth * 0.35 + vel_adj, 0)
    spot_cvd = safe_float(minfo.get("spot_cvd_15m_usd"), default=0.0) or 0.0
    funding_r = safe_float(minfo.get("funding_rate"), default=0.0) or 0.0
    oi_chg = safe_float(minfo.get("oi_1h_chg_pct"), default=0.0) or 0.0
    if minfo.get("perp_spot_basis_bps") is None and m_px and m_px > 0.0:
        mark_p = safe_pos_float(minfo.get("mark_price"), default=None)
        if mark_p is not None and mark_p > 0.0:
            minfo["perp_spot_basis_bps"] = round(((mark_p - m_px) / m_px) * 10000.0, 2)
        elif funding_r != 0.0:
            minfo["perp_spot_basis_bps"] = round(funding_r * 10000.0 * 1.5, 2)

    if spot_cvd > 0.0 and cvd_usd < 0.0 and funding_r <= 0.0001:
        minfo["spot_perp_flow_regime"] = "INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE"
    elif spot_cvd < 0.0 and cvd_usd > 0.0 and oi_chg >= 0.25:
        minfo["spot_perp_flow_regime"] = "RETAIL_PERP_LONG_TRAP"
    elif spot_cvd > 0.0 and cvd_usd > 0.0:
        minfo["spot_perp_flow_regime"] = "SYNCHRONIZED_SPOT_PERP_BUYING"
    elif spot_cvd < 0.0 and cvd_usd < 0.0:
        minfo["spot_perp_flow_regime"] = "SYNCHRONIZED_SPOT_PERP_SELLING"
    else:
        minfo["spot_perp_flow_regime"] = "BALANCED_CROSS_VENUE_FLOW"


def _blend_structural_liq_anchor(
    cur_px: float,
    base_px: float,
    levels_and_pads: tuple[tuple[Any, float], ...],
    *,
    is_short: bool,
    min_dist: float,
    max_dist: float,
    bound_px: float,
    inner_guard_px: Optional[float] = None,
) -> float:
    """Blend base liquidation distance with validated structural stop/liquidity pool anchors."""
    cands = [
        swept
        for raw_lvl, pad in levels_and_pads
        if (lvl := safe_pos_float(raw_lvl, default=0.0) or 0.0) > 0.0
        and (swept := lvl * pad)
        and (min_dist <= (((swept - cur_px) if is_short else (cur_px - swept)) / cur_px) <= max_dist)
        and (inner_guard_px is None or (swept > inner_guard_px * 1.01 if is_short else swept < inner_guard_px * 0.99))
    ]
    blended = 0.5 * base_px + 0.5 * (sum(cands) / len(cands)) if cands else base_px
    return max(bound_px, blended) if is_short else min(bound_px, blended)


def compute_liquidation_clusters(
    pinfo: Optional[Dict[str, Any]], minfo: Optional[Dict[str, Any]], *, decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute symmetric LONG (downside) and SHORT (upside) liquidation heatmap clusters and magnet bias."""
    if not isinstance(minfo, dict):
        return {}
    p_dict = pinfo if isinstance(pinfo, dict) else {}
    cur_px = safe_pos_float(p_dict.get("price"), default=0.0) or 0.0
    b_bid = safe_pos_float(minfo.get("best_bid"), default=0.0) or 0.0
    b_ask = safe_pos_float(minfo.get("best_ask"), default=0.0) or 0.0
    ob_mid = safe_pos_float(minfo.get("ob_mid_price"), default=0.0) or ((b_bid + b_ask) / 2.0 if b_bid > 0.0 and b_ask >= b_bid else 0.0)
    if ob_mid > 0.0 and (cur_px <= 0.0 or _prices_diverged(cur_px, ob_mid)):
        cur_px = ob_mid
    if cur_px <= 0.0:
        return {}

    oi_usd = safe_pos_float(minfo.get("oi_usd"), default=0.0) or 0.0
    if oi_usd <= 0.0:
        b_depth = safe_nonneg_float(minfo.get("bid_depth_usd"), default=0.0) or 0.0
        a_depth = safe_nonneg_float(minfo.get("ask_depth_usd"), default=0.0) or 0.0
        oi_usd = (b_depth + a_depth) * 12.0 if (b_depth + a_depth) > 0.0 else 0.0
    if oi_usd <= 0.0:
        return {}

    raw_long_pct = clean_scale(minfo.get("whale_long_pct"), default=None, allow_percentage=True)
    if raw_long_pct is not None:
        raw_ls = min(0.85, max(0.15, float(raw_long_pct)))
        long_share = min(0.75, max(0.25, 0.50 + (raw_ls - 0.58) * 0.65))
    else:
        long_share = 0.50
    funding = safe_float(minfo.get("funding_rate"), default=0.0) or 0.0
    eff_long_share = min(0.80, max(0.20, long_share + max(-0.25, min(0.25, funding * 1500.0)) * 0.45))
    eff_short_share = 1.0 - eff_long_share

    vol_1h = max(0.35, min(6.0, safe_pos_float(p_dict.get("vol_1h_pct"), default=1.2) or 1.2))
    near_band_pct = max(1.2, min(4.5, vol_1h * 1.65))
    deep_band_pct = max(near_band_pct * 1.85, min(10.0, vol_1h * 3.4))

    long_near_raw = _blend_structural_liq_anchor(
        cur_px, cur_px * (1.0 - near_band_pct / 100.0),
        ((p_dict.get("val_px"), 0.995), (p_dict.get("low_24h"), 0.996), (minfo.get("bid_wall_px"), 0.997)),
        is_short=False, min_dist=0.0045, max_dist=0.065, bound_px=cur_px * (1.0 - 0.0045),
    )
    short_near_raw = _blend_structural_liq_anchor(
        cur_px, cur_px * (1.0 + near_band_pct / 100.0),
        ((p_dict.get("vah_px"), 1.005), (p_dict.get("high_24h"), 1.004), (minfo.get("ask_wall_px"), 1.003)),
        is_short=True, min_dist=0.0045, max_dist=0.065, bound_px=cur_px * (1.0 + 0.0045),
    )
    long_deep_raw = _blend_structural_liq_anchor(
        cur_px, min(cur_px * (1.0 - deep_band_pct / 100.0), long_near_raw * 0.978),
        ((p_dict.get("swing_low_px"), 0.995), (p_dict.get("low_7d"), 0.996)),
        is_short=False, min_dist=0.022, max_dist=0.14, bound_px=long_near_raw * 0.985, inner_guard_px=long_near_raw,
    )
    short_deep_raw = _blend_structural_liq_anchor(
        cur_px, max(cur_px * (1.0 + deep_band_pct / 100.0), short_near_raw * 1.022),
        ((p_dict.get("swing_high_px"), 1.005), (p_dict.get("high_7d"), 1.004)),
        is_short=True, min_dist=0.022, max_dist=0.14, bound_px=short_near_raw * 1.015, inner_guard_px=short_near_raw,
    )

    eff_dec = _auto_px_decimals(cur_px, decimals)
    long_liq_px, short_liq_px, long_liq_deep_px, short_liq_deep_px = (
        round(v, eff_dec) for v in (long_near_raw, short_near_raw, long_deep_raw, short_deep_raw)
    )
    oi_chg_1h = safe_float(minfo.get("oi_1h_chg_pct"), default=0.0) or 0.0
    oi_heat_mult = min(1.25, max(0.85, 1.0 + (oi_chg_1h / 100.0) * 1.5))

    long_liq_usd = round(oi_usd * eff_long_share * 0.014 * oi_heat_mult, 0)
    short_liq_usd = round(oi_usd * eff_short_share * 0.014 * oi_heat_mult, 0)
    long_liq_deep_usd = round(oi_usd * eff_long_share * 0.024 * oi_heat_mult, 0)
    short_liq_deep_usd = round(oi_usd * eff_short_share * 0.024 * oi_heat_mult, 0)

    long_liq_dist_pct, short_liq_dist_pct, long_liq_deep_dist_pct, short_liq_deep_dist_pct = (
        pct_delta(px_v, cur_px)
        for px_v in (long_liq_px, short_liq_px, long_liq_deep_px, short_liq_deep_px)
    )
    short_dist_sq = max(0.35, abs(short_liq_dist_pct)) ** 2
    long_dist_sq = max(0.35, abs(long_liq_dist_pct)) ** 2
    kalman_vel = safe_float(
        p_dict.get(
            "kalman_vel",
            p_dict.get("gradient_recent_fast", p_dict.get("slope_15m", p_dict.get("slope_3_7m"))),
        ),
        default=0.0,
    ) or 0.0
    chg_15m_v = safe_float(p_dict.get("chg_15m_pct"), default=0.0) or 0.0
    taker_r = safe_nonneg_float(minfo.get("taker_buy_sell_ratio"), default=1.0)
    ob_imb_v = safe_nonneg_float(minfo.get("ob_imbalance"), default=0.5)
    vel_sign = 1.0 if kalman_vel > 0.0 else (-1.0 if kalman_vel < 0.0 else 0.0)
    vel_tilt = max(
        -0.35,
        min(
            0.35,
            (chg_15m_v * 0.08) + (vel_sign * 0.06) + ((taker_r - 1.0) * 0.15) + ((ob_imb_v - 0.50) * 0.30),
        ),
    )
    short_gravity = (short_liq_usd / short_dist_sq) * (1.0 + max(-0.30, vel_tilt))
    long_gravity = (long_liq_usd / long_dist_sq) * (1.0 - min(0.30, vel_tilt))
    liq_magnet_bias = (
        "SHORT_SQUEEZE_MAGNET_UP" if short_gravity > long_gravity * 1.15
        else ("LONG_SWEEP_MAGNET_DOWN" if long_gravity > short_gravity * 1.15 else "BALANCED_LIQ_MAGNET")
    )
    return {
        "long_liq_px": long_liq_px, "long_liq_usd": long_liq_usd, "long_liq_dist_pct": long_liq_dist_pct,
        "short_liq_px": short_liq_px, "short_liq_usd": short_liq_usd, "short_liq_dist_pct": short_liq_dist_pct,
        "long_liq_deep_px": long_liq_deep_px, "long_liq_deep_usd": long_liq_deep_usd, "long_liq_deep_dist_pct": long_liq_deep_dist_pct,
        "short_liq_deep_px": short_liq_deep_px, "short_liq_deep_usd": short_liq_deep_usd, "short_liq_deep_dist_pct": short_liq_deep_dist_pct,
        "liq_magnet_bias": liq_magnet_bias,
    }


__all__ = [
    "_MICRO_PRICE_ANCHOR_KEYS",
    "_WINDOW_TREND_SPECS",
    "_auto_px_decimals",
    "_build_vp_position_dict",
    "_compute_horizon_range_dict",
    "_first_variant_dict",
    "_parse_ts_px_row",
    "_prices_diverged",
    "apply_live_24trend_windows",
    "apply_short_trend_item",
    "compute_24h_price_metrics",
    "compute_7d_range_metrics",
    "compute_cross_btc_metrics",
    "compute_liquidation_clusters",
    "compute_swing_wave_metrics",
    "compute_volume_profile_metrics",
    "enrich_microstructure_cvd",
    "extract_symbol_recent_fills",
    "load_symbol_microstructure",
    "load_symbol_price_series",
    "pct_delta",
    "range_position_pct",
    "round_opt_px",
]
