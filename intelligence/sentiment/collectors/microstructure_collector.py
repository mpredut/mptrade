"""Spot/perpetual microstructure, 24h/7d price series, volume profile, and liquidation cluster collector."""
from __future__ import annotations

import logging
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

from botcore import extract_current_price_item
from cachepaths import (
    cache_symbol_variants, canonical_base_asset, compatible_quote_suffixes, legacy_filenames_for,
)
from intelligence.external.collectors.orderbook_depth import round_spread_bps
from intelligence.macro_sentiment_ssot import (
    clean_scale, obj_attr, safe_float, safe_nonneg_float, safe_pos_float,
)
from instrument_registry import casefold_get
from providers.base import (
    _EXEC_QTY_KEYS, _UNFILLED_ORDER_STATUSES, aggregate_fills_by_order_id, extract_executed_qty,
    extract_order_id, extract_order_price, extract_order_side, extract_orig_qty,
    extract_timestamp_ms, format_actor_tag, resolve_order_origin,
)
from providers.execution_audit import extract_intent_order_type, extract_intent_side
from state_io import (
    read_json_dict_recovering as read_json_cached, read_tail_jsonl_records, resolve_cachedb_filepath,
)

logger = logging.getLogger("intelligence.sentiment.microstructure_collector")


def round_opt_px(val: Any, decimals: int = 6) -> Optional[float]:
    """Round a positive price anchor to ``decimals``, returning None when absent or non-positive."""
    vf = safe_pos_float(val, default=None)
    return round(vf, decimals) if vf is not None else None


def _parse_ts_px_row(row: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    tf = safe_pos_float(row[0], 0.0)
    pf = safe_pos_float(row[1], 0.0)
    if tf <= 0.0 or pf <= 0.0:
        return None
    return (tf * 1000.0 if tf < 1e11 else tf), pf


_MICRO_PRICE_ANCHOR_KEYS: frozenset[str] = frozenset({
    "best_bid", "best_ask", "ob_mid_price", "bid_wall_px", "ask_wall_px",
    "long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px",
    "long_liq_dist_pct", "short_liq_dist_pct", "long_liq_deep_dist_pct", "short_liq_deep_dist_pct",
})


def _prices_diverged(px_a: Any, px_b: Any, threshold: float = 0.10) -> bool:
    """Return True when two positive prices diverge by more than ``threshold`` relative to their minimum."""
    a_f = safe_pos_float(px_a, default=0.0) or 0.0
    b_f = safe_pos_float(px_b, default=0.0) or 0.0
    if a_f <= 0.0 or b_f <= 0.0:
        return False
    return abs(a_f - b_f) / min(a_f, b_f) > threshold


def _build_vp_position_dict(
    cur_px: float,
    poc_px: float,
    val_px: float,
    vah_px: float,
) -> Dict[str, Any]:
    """Compute symmetric % distances from POC/VAL/VAH, Value Area position %, and VA regime."""
    if cur_px <= 0.0 or poc_px <= 0.0 or val_px <= 0.0 or vah_px <= 0.0:
        return {}
    return {
        "poc_dist_pct": round(((cur_px / poc_px) - 1.0) * 100.0, 2),
        "vah_dist_pct": round(((cur_px / vah_px) - 1.0) * 100.0, 2),
        "val_dist_pct": round(((cur_px / val_px) - 1.0) * 100.0, 2),
        "va_pos_pct": round(((cur_px - val_px) / (vah_px - val_px)) * 100.0, 1) if vah_px > val_px else 50.0,
        "va_regime": (
            "ABOVE_VAH"
            if cur_px > vah_px
            else ("BELOW_VAL" if cur_px < val_px else ("INSIDE_VA_UPPER" if cur_px >= poc_px else "INSIDE_VA_LOWER"))
        ),
    }


def _compute_horizon_range_dict(
    cur_px: float,
    low_px: float,
    high_px: float,
    suffix: str,
) -> Dict[str, Any]:
    """Build low/high range geometry (`dd_<sfx>_high_pct`, `bounce_<sfx>_low_pct`, `span_<sfx>_pct`, `pos_<sfx>_pct`)."""
    out: Dict[str, Any] = {f"low_{suffix}": low_px, f"high_{suffix}": high_px}
    if high_px > 0:
        out[f"dd_{suffix}_high_pct"] = round((cur_px / high_px - 1.0) * 100.0, 2)
    if low_px > 0:
        out[f"bounce_{suffix}_low_pct"] = round((cur_px / low_px - 1.0) * 100.0, 2)
        out[f"span_{suffix}_pct"] = round((high_px / low_px - 1.0) * 100.0, 2)
    if high_px > low_px > 0:
        out[f"pos_{suffix}_pct"] = round(((cur_px - low_px) / (high_px - low_px)) * 100.0, 1)
    return out


def compute_volume_profile_metrics(
    series: list,
    *,
    num_bins: int = 24,
    value_area_pct: float = 0.70,
    decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute realized session Volume Profile (Point of Control POC, Value Area High VAH, Value Area Low VAL, symmetric % distances, VA position, and regime)."""
    if not isinstance(series, list) or len(series) < 4:
        return {}
    try:
        timed_rows: list[Tuple[float, float, float]] = []
        scalar_rows: list[float] = []
        for row in series:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                pt = _parse_ts_px_row(row)
                if pt is not None:
                    vol_w = safe_pos_float(row[2], 0.0) if len(row) >= 3 else 0.0
                    timed_rows.append((pt[0], pt[1], vol_w))
            elif not isinstance(row, (list, tuple)):
                px_f = safe_pos_float(row, 0.0)
                if px_f > 0.0:
                    scalar_rows.append(px_f)

        samples: list[Tuple[float, float]] = []
        prev_px: Optional[float] = None
        raw_pairs = (
            [(px_f, vol_w) for _, px_f, vol_w in sorted(timed_rows, key=lambda x: x[0])]
            if timed_rows
            else [(px_f, 0.0) for px_f in scalar_rows]
        )
        for px_f, vol_w in raw_pairs:
            if vol_w <= 0.0:
                vol_w = 1.0 + min(5.0, (abs(px_f - prev_px) / prev_px) * 1000.0) if prev_px else 1.0
            prev_px = px_f
            samples.append((px_f, vol_w))

        if len(samples) < 4:
            return {}

        if len(samples) > 4000:
            step = max(1, len(samples) // 2000)
            downsampled = samples[::step]
            if downsampled[-1][0] != samples[-1][0]:
                downsampled.append(samples[-1])
            samples = downsampled

        cur_px = samples[-1][0]
        low_px = min(p for p, _ in samples)
        high_px = max(p for p, _ in samples)
        if high_px <= low_px or low_px <= 0.0:
            return {}

        n_bins = max(8, min(64, int(num_bins)))
        bin_w = (high_px - low_px) / float(n_bins)
        if bin_w <= 0.0:
            return {}

        bin_vols = [0.0] * n_bins
        bin_px_sums = [0.0] * n_bins
        total_vol = 0.0
        for px_v, w_v in samples:
            b_idx = min(n_bins - 1, max(0, int((px_v - low_px) / bin_w)))
            bin_vols[b_idx] += w_v
            bin_px_sums[b_idx] += px_v * w_v
            total_vol += w_v

        if total_vol <= 0.0:
            return {}

        poc_idx = max(range(n_bins), key=lambda i: bin_vols[i])
        poc_raw = (
            bin_px_sums[poc_idx] / bin_vols[poc_idx]
            if bin_vols[poc_idx] > 0.0
            else (low_px + (poc_idx + 0.5) * bin_w)
        )

        target_vol = total_vol * max(0.50, min(0.90, float(value_area_pct)))
        acc_vol = bin_vols[poc_idx]
        val_idx = vah_idx = poc_idx

        while acc_vol < target_vol and (val_idx > 0 or vah_idx < n_bins - 1):
            down_vol = bin_vols[val_idx - 1] if val_idx > 0 else -1.0
            up_vol = bin_vols[vah_idx + 1] if vah_idx < n_bins - 1 else -1.0
            if up_vol > down_vol or (up_vol == down_vol and (poc_idx - val_idx) > (vah_idx - poc_idx) and vah_idx < n_bins - 1):
                vah_idx += 1
                acc_vol += up_vol
            elif val_idx > 0:
                val_idx -= 1
                acc_vol += down_vol
            elif vah_idx < n_bins - 1:
                vah_idx += 1
                acc_vol += up_vol
            else:
                break

        val_raw = min(poc_raw, low_px + val_idx * bin_w)
        vah_raw = max(poc_raw, low_px + (vah_idx + 1) * bin_w)
        eff_dec = decimals if decimals is not None else (2 if cur_px >= 100.0 else (4 if cur_px >= 1.0 else 6))
        poc_px = round(poc_raw, eff_dec)
        val_px = round(val_raw, eff_dec)
        vah_px = round(vah_raw, eff_dec)
        if vah_px <= val_px:
            vah_px = round(high_px, eff_dec)
            val_px = round(low_px, eff_dec)

        out: Dict[str, Any] = {"poc_px": poc_px, "vah_px": vah_px, "val_px": val_px}
        out.update(_build_vp_position_dict(cur_px, poc_px, val_px, vah_px))
        return out
    except Exception:
        return {}


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
        l24 = min(x[1] for x in clean_series)
        h24 = max(x[1] for x in clean_series)
        out = _compute_horizon_range_dict(cur_px, l24, h24, "24h")

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
        if span_ms >= 3600000.0 * 2.0:
            win_6h = [(tms, p_v) for tms, p_v in clean_series if tms >= now_ms - 360.0 * 60000.0]
            cutoff_1h = now_ms - 60.0 * 60000.0
            prior_pts = [p_v for tms, p_v in win_6h if tms < cutoff_1h]
            recent_pts = [p_v for tms, p_v in win_6h if tms >= cutoff_1h]
            if len(prior_pts) >= 2 and len(recent_pts) >= 2:
                high_6h = max(p_v for _, p_v in win_6h)
                prior_low = min(prior_pts)
                recent_low = min(recent_pts)
                low_6h = min(prior_low, recent_low)
                dd_low_6h = ((low_6h / high_6h) - 1.0) * 100.0 if high_6h > 0.0 else 0.0
                pos_6h = ((cur_px - low_6h) / (high_6h - low_6h)) if high_6h > low_6h else 1.0
                if dd_low_6h <= -2.0 and pos_6h <= 0.65 and prior_low > 0.0 and recent_low > 0.0:
                    eff_dec = 2 if cur_px >= 100.0 else (4 if cur_px >= 1.0 else 6)
                    if recent_low < prior_low and cur_px > prior_low:
                        out["bottom_structure_6h"] = "LIQUIDITY_SWEEP_RECLAIM"
                        out["reclaim_above_prior_low_pct"] = round(((cur_px / prior_low) - 1.0) * 100.0, 2)
                    elif recent_low >= prior_low and ((recent_low / prior_low) - 1.0) * 100.0 <= 4.5 and cur_px > recent_low * 1.002:
                        out["bottom_structure_6h"] = "HIGHER_LOW_DOUBLE_BOTTOM"
                        out["higher_low_vs_prior_pct"] = round(((recent_low / prior_low) - 1.0) * 100.0, 2)
                    elif recent_low <= prior_low and cur_px <= prior_low and ((cur_px / recent_low) - 1.0) * 100.0 <= 2.5:
                        out["bottom_structure_6h"] = "FRESH_6H_LOW_TEST"
                    if "bottom_structure_6h" in out:
                        out["prior_6h_low_px"] = round(prior_low, eff_dec)
                        out["recent_1h_low_px"] = round(recent_low, eff_dec)
        vp_metrics = compute_volume_profile_metrics(series)
        if vp_metrics:
            out.update(vp_metrics)
        return out
    except Exception:
        return {}


def _extract_cached_items_series(data: Any, cands: list[str]) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    raw_items = data.get("items") if "items" in data else data
    if isinstance(raw_items, dict):
        for k_cand in cands:
            ser = casefold_get(raw_items, k_cand, [])
            if isinstance(ser, list) and ser:
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
        c
        for c in cache_symbol_variants(
            symbol,
            include_original=True,
            include_base="USD" in eff_quotes,
            quotes=eff_quotes,
        )
        if isinstance(c, str) and "/" not in c
    ]
    series_24h: list = []
    vals_7d: list = []

    # 1. Try primary 24h dense cache (cache_24_price_<SYM>.json and legacy variants)
    for cand in cands:
        primary_24 = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, f"cache_24_price_{cand}.json")))
        for p24_path in (primary_24, *legacy_filenames_for(primary_24)):
            if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                continue
            ser = _extract_cached_items_series(read_json_cached(p24_path), cands)
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
                    aser = _extract_cached_items_series(read_json_cached(arch_path), cands)
                    if len(aser) >= 2:
                        archive_pts = list(aser)
                        break
                else:
                    parsed_rows = []
                    for rec in read_tail_jsonl_records(arch_path, limit=None, max_bytes=1024 * 1024):
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
                cand_pts = [
                    [pt[0], pt[1]]
                    for pt in cur_item
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]
                cand_last = _parse_ts_px_row(cand_pts[-1]) if cand_pts else None
                if last_24_pt is not None and cand_last is not None and _prices_diverged(last_24_pt[1], cand_last[1]):
                    continue
                series_24h = list(series_24h) + cand_pts
                break
            elif (px_f := extract_current_price_item(cur_item)) is not None:
                if series_24h and last_24_pt is not None:
                    if _prices_diverged(last_24_pt[1], px_f):
                        continue
                    series_24h = list(series_24h) + [[last_24_pt[0] + 1000.0, px_f]]
                    break

    # 4. Normalize, sort, and deduplicate series_24h so callers always receive clean [tms, px] (or [tms, px, vol]) points
    if series_24h:
        cleaned_24h = [
            (
                [pt[0], pt[1], vol_v]
                if len(row) >= 3 and (vol_v := safe_pos_float(row[2], default=None)) is not None
                else [pt[0], pt[1]]
            )
            for row in series_24h
            if (pt := _parse_ts_px_row(row)) is not None
        ]
        if len(cleaned_24h) >= 2:
            cleaned_24h.sort(key=lambda x: x[0])
            dedup_24h = [cleaned_24h[0]]
            for pt in cleaned_24h[1:]:
                if pt[0] > dedup_24h[-1][0]:
                    dedup_24h.append(pt)
                elif len(pt) < 3 and len(dedup_24h[-1]) >= 3:
                    dedup_24h[-1] = [pt[0], pt[1], dedup_24h[-1][2]]
                else:
                    dedup_24h[-1] = pt
            series_24h = dedup_24h if len(dedup_24h) >= 2 else []
        else:
            series_24h = []

    if vals_7d and series_24h:
        latest_px = safe_pos_float(series_24h[-1][1], default=None)
        if latest_px is not None and vals_7d[-1] != latest_px and not _prices_diverged(latest_px, vals_7d[-1]):
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
            mean_a = statistics.fmean(rets_a)
            mean_b = statistics.fmean(rets_b)
            var_a = statistics.pvariance(rets_a, mu=mean_a)
            var_b = statistics.pvariance(rets_b, mu=mean_b)
            cov = statistics.fmean((ra - mean_a) * (rb - mean_b) for ra, rb in zip(rets_a, rets_b))
            if var_b > 1e-12:
                out["beta_btc_24h"] = round(cov / var_b, 2)
                if var_a > 1e-12:
                    corr = cov / math.sqrt(var_a * var_b)
                    out["corr_btc_24h"] = round(max(-1.0, min(1.0, corr)), 2)
        return out
    except Exception:
        return {}


def extract_symbol_recent_fills(
    orders_list: list,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[Dict[str, tuple[Optional[str], Optional[str]]]] = None,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract last action side/actor, active cycle avg BUY price, last BUY/SELL fills, and recent tranches."""
    if not isinstance(orders_list, list) or not orders_list:
        return None
    _explicit_exec_keys = (*_EXEC_QTY_KEYS, "filled", "origSz", "resting", "msg")
    valid = [
        o
        for o in orders_list
        if isinstance(o, dict)
        and (
            extract_executed_qty(o, default=0.0) > 0.0
            or (
                str(o.get("status") or "").strip().upper() not in _UNFILLED_ORDER_STATUSES
                and not any(k in o for k in _explicit_exec_keys)
            )
        )
    ]
    if not valid:
        return None
    ai_oids: set[str] = set(known_ai_order_ids or ())
    valid_oids: set[str] = {
        str(oid_x)
        for x in valid
        if (oid_x := extract_order_id(x, default=""))
    }
    parsed_ai_intents: List[tuple[str, str, float, float, str]] = []
    for it in ai_intents or ():
        if not isinstance(it, dict) or str(it.get("status") or "").strip().upper() != "EXECUTED":
            continue
        exec_pay = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
        it_oid = extract_order_id(exec_pay, default="")
        if it_oid and str(it_oid) in valid_oids:
            continue
        i_side = extract_intent_side(it) or str(it.get("decision") or "").strip().upper()
        i_ts_s = safe_pos_float(it.get("timestamp"), default=0.0) or 0.0
        if i_side not in ("BUY", "SELL") or i_ts_s <= 0.0:
            continue
        i_sym = str(it.get("symbol") or "").strip().upper()
        i_otype = extract_intent_order_type(it)
        hist = it.get("history") if isinstance(it.get("history"), list) else []
        last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
        i_qty = (
            extract_executed_qty(exec_pay, default=0.0)
            or extract_orig_qty(exec_pay, default=0.0)
            or safe_pos_float(last_h.get("qty"), default=0.0)
            or 0.0
        )
        parsed_ai_intents.append((i_sym, i_side, i_ts_s, i_qty, i_otype))
    origin_by_oid: Dict[str, tuple[str, Optional[str], Optional[str]]] = {}
    origin_by_side_ts: Dict[tuple[str, Any], tuple[str, Optional[str], Optional[str]]] = {}
    for o in valid:
        act_v, mod_v, rule_v = resolve_order_origin(o, ai_oids, audit_origin_map)
        if (
            act_v == "BOT"
            and not mod_v
            and not rule_v
            and parsed_ai_intents
            and not o.get("clientOrderId")
            and not o.get("client_order_id")
        ):
            o_side = extract_order_side(o)
            o_ts_ms = extract_timestamp_ms(o, default=0) or 0
            o_ts_s = o_ts_ms / 1000.0 if o_ts_ms > 0 else 0.0
            o_qty = extract_executed_qty(o, default=0.0) or 0.0
            o_sym = str(o.get("symbol") or "").strip().upper()
            for i_sym, i_side, i_ts_s, i_qty, i_otype in parsed_ai_intents:
                if i_side != o_side or o_ts_s <= 0.0 or o_ts_s < i_ts_s - 5.0:
                    continue
                if i_sym and o_sym and (
                    canonical_base_asset(i_sym) != canonical_base_asset(o_sym)
                    or compatible_quote_suffixes(i_sym) != compatible_quote_suffixes(o_sym)
                ):
                    continue
                if (
                    (i_qty > 0.0 and o_qty > 0.0 and abs(o_qty - i_qty) <= max(1e-5, i_qty * 0.01) and o_ts_s <= i_ts_s + 86400.0)
                    or (i_otype == "MARKET" and o_ts_s <= i_ts_s + 120.0)
                ):
                    act_v = "AI"
                    break
        oid_v = extract_order_id(o, default="")
        if oid_v:
            if act_v == "AI":
                ai_oids.add(str(oid_v))
            prev = origin_by_oid.get(str(oid_v))
            if prev is None or (not prev[1] and mod_v) or (not prev[2] and rule_v) or act_v == "AI":
                origin_by_oid[str(oid_v)] = (
                    "AI" if (act_v == "AI" or (prev and prev[0] == "AI")) else "BOT",
                    mod_v or (prev[1] if prev else None),
                    rule_v or (prev[2] if prev else None),
                )
        else:
            origin_by_side_ts[(extract_order_side(o), extract_timestamp_ms(o, default=None))] = (act_v, mod_v, rule_v)

    if any(o.get(k) is not None for o in valid for k in ("orderId", "order_id", "ordertxid", "oid")):
        agg = aggregate_fills_by_order_id(valid)
        if agg:
            valid = agg

    def _fill_origin(o: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not isinstance(o, dict):
            return None, None, None, None
        oid_k = str(extract_order_id(o, default="") or "")
        side_ts_k = (extract_order_side(o), extract_timestamp_ms(o, default=None))
        act, mod, rule = (
            origin_by_oid.get(oid_k)
            or origin_by_side_ts.get(side_ts_k)
            or resolve_order_origin(o, ai_oids, audit_origin_map)
        )
        if act == "AI":
            return "AI", None, None, "AI"
        return "BOT", mod, rule, format_actor_tag("BOT", mod, rule)

    ordered = sorted(
        enumerate(valid),
        key=lambda item: (extract_timestamp_ms(item[1], default=0) or 0, item[0]),
    )
    chron = [o for _, o in ordered]
    last_any = chron[-1]

    def _norm_fill(o: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Any]:
        return (
            round_opt_px(extract_order_price(o, default=None), 6),
            extract_executed_qty(o, default=None),
            extract_timestamp_ms(o, default=None),
        )

    cost_qty = 0.0
    active_avg = 0.0
    buys_fmt: list[Dict[str, Any]] = []
    sells_fmt: list[Dict[str, Any]] = []
    for o in chron:
        side_o = extract_order_side(o)
        p_f, q_f, ts_raw = _norm_fill(o)
        act_o, mod_o, rule_o, tag_o = _fill_origin(o)
        item_d: Dict[str, Any] = {"price": p_f, "qty": q_f, "ts_ms": ts_raw, "actor": act_o}
        if mod_o:
            item_d["module"] = mod_o
        if rule_o:
            item_d["rule"] = rule_o
        if tag_o and tag_o != act_o:
            item_d["actor_tag"] = tag_o
        if side_o == "BUY":
            buys_fmt.append(item_d)
            if q_f is not None and p_f is not None:
                new_cost_qty = cost_qty + q_f
                active_avg = ((cost_qty * active_avg) + (q_f * p_f)) / new_cost_qty if new_cost_qty > 0 else 0.0
                cost_qty = new_cost_qty
        elif side_o == "SELL":
            sells_fmt.append(item_d)
            if q_f is not None:
                if q_f >= cost_qty * 0.98 or cost_qty - q_f <= 1e-12:
                    cost_qty = 0.0
                    active_avg = 0.0
                else:
                    cost_qty -= q_f

    last_act, last_mod, last_rule, last_tag = _fill_origin(last_any)
    out_fills: Dict[str, Any] = {
        "last_action_side": extract_order_side(last_any) or last_any.get("side"),
        "last_action_actor": last_act,
        "active_avg_buy_px": round(active_avg, 6) if active_avg > 0 else None,
        "last_buy": dict(buys_fmt[-1]) if buys_fmt else None,
        "last_sell": dict(sells_fmt[-1]) if sells_fmt else None,
        "recent_buys": buys_fmt[-4:],
        "recent_sells": sells_fmt[-3:],
    }
    if last_mod:
        out_fills["last_action_module"] = last_mod
    if last_rule:
        out_fills["last_action_rule"] = last_rule
    if last_tag and last_tag != last_act:
        out_fills["last_action_actor_tag"] = last_tag
    return out_fills


def apply_short_trend_item(
    entry: Dict[str, Any],
    tinfo: Dict[str, Any],
    *,
    fallback_price: Any = None,
    require_any_slope: bool = False,
    allow_price_update: bool = True,
) -> bool:
    """Populate short/dynamic trend slopes (90s, 3.7m, 15m, 2.5h) on a symbol telemetry dict from `cache_short_trend.json`."""
    dyn = tinfo.get("dynamic", {}) if isinstance(tinfo.get("dynamic"), dict) else {}
    d90 = dyn.get("90", {}) if isinstance(dyn.get("90"), dict) else {}
    d222 = dyn.get("222", {}) if isinstance(dyn.get("222"), dict) else {}
    if d222.get("slope_full") is None:
        if tinfo.get("slope_full") is not None:
            d222 = tinfo
        elif d90.get("slope_full") is not None:
            d222 = d90
    d900 = dyn.get("900", {}) if isinstance(dyn.get("900"), dict) else {}
    d9000 = dyn.get("9000", {}) if isinstance(dyn.get("9000"), dict) else {}
    if require_any_slope and not any(
        x is not None
        for x in (
            d222.get("slope_full"),
            d900.get("slope_full"),
            d9000.get("slope_full"),
            tinfo.get("slope_full"),
        )
    ):
        return False
    cur_px = (
        safe_pos_float(fallback_price, default=None)
        or safe_pos_float(tinfo.get("current_price"), default=None)
    )
    if allow_price_update and safe_float(entry.get("price"), 0.0) <= 0 and cur_px is not None:
        entry["price"] = cur_px
    trend = tinfo.get("final_trend", entry.get("trend", 0))
    entry["trend"] = trend
    entry["growth_coefficient"] = tinfo.get("growth_coefficient", entry.get("growth_coefficient", 0.0))
    entry["slope_full"] = tinfo.get("slope_full", 0.0)
    entry["gradient_recent"] = tinfo.get("gradient_recent", 0.0)
    for k in ("gradient_recent_fast", "epsilon", "vol_1h_pct"):
        if not require_any_slope or tinfo.get(k) is not None:
            entry[k] = tinfo.get(k)
    if not require_any_slope:
        entry["trend_fast"] = tinfo.get("trend_fast", trend)
    t_vol = safe_pos_float(tinfo.get("vol_1h_pct"), default=None)
    t_re = safe_pos_float(tinfo.get("adapt_reentry_pct"), default=None)
    if t_vol is None or (t_re is not None and abs(t_re - 2.0 * t_vol) <= 0.05):
        for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
            if tinfo.get(ak) is not None:
                entry[ak] = tinfo[ak]
    if d90.get("slope_full") is not None:
        entry["slope_90s"] = d90.get("slope_full")
        entry["trend_90s"] = d90.get("final_trend", trend)
    if d90.get("growth_coefficient") is not None:
        entry["growth_90s"] = d90.get("growth_coefficient")
    for w_dict, s_key, t_key, g_key in (
        (d222, "slope_3_7m", "trend_3_7m", "growth_3_7m"),
        (d900, "slope_15m", "trend_15m", "growth_15m"),
        (d9000, "slope_2_5h", "trend_2_5h", "growth_2_5h"),
    ):
        if w_dict.get("slope_full") is not None or require_any_slope:
            entry[s_key] = w_dict.get("slope_full", 0.0) if require_any_slope else w_dict.get("slope_full")
            entry[t_key] = w_dict.get("final_trend", trend)
        if w_dict.get("growth_coefficient") is not None:
            entry[g_key] = w_dict.get("growth_coefficient")
    return True


def apply_live_24trend_windows(
    entry: Dict[str, Any],
    lt_win: Dict[str, Any],
    fallback_price: Any = None,
) -> None:
    """Overlay fresh 1-second sampled dynamic windows (90s, 3.7m, 15m, 2.5h) from `cachedb/24trend/live_<SYM>.json`."""
    w90 = lt_win.get("90") if isinstance(lt_win.get("90"), dict) else {}
    w222 = lt_win.get("222") if isinstance(lt_win.get("222"), dict) else w90
    w900 = lt_win.get("900") if isinstance(lt_win.get("900"), dict) else {}
    w9000 = lt_win.get("9000") if isinstance(lt_win.get("9000"), dict) else {}
    live_px = (
        safe_pos_float(fallback_price, default=None)
        or safe_pos_float(w222.get("current_price"), default=None)
        or safe_pos_float(w900.get("current_price"), default=None)
    )
    if entry.get("price") is None and live_px is not None:
        entry["price"] = live_px
    entry.setdefault("trend", w222.get("final_trend", 0))
    entry.setdefault("growth_coefficient", w222.get("growth_coefficient", 0.0))
    if entry.get("gradient_recent") is None and w222.get("gradient_recent") is not None:
        entry["gradient_recent"] = w222.get("gradient_recent")
    if entry.get("gradient_recent_fast") is None:
        gf_live = (
            w90.get("gradient_recent_fast")
            or w222.get("gradient_recent_fast")
            or w90.get("gradient_recent")
        )
        if gf_live is not None:
            entry["gradient_recent_fast"] = gf_live
    if entry.get("epsilon") is None:
        eps_live = w222.get("epsilon") or w90.get("epsilon")
        if eps_live is not None:
            entry["epsilon"] = eps_live
    for w_dict, s_key, t_key, g_key, keep_existing_default in (
        (w90, "slope_90s", "trend_90s", "growth_90s", False),
        (w222, "slope_3_7m", "trend_3_7m", "growth_3_7m", True),
        (w900, "slope_15m", "trend_15m", "growth_15m", True),
        (w9000, "slope_2_5h", "trend_2_5h", "growth_2_5h", True),
    ):
        if not w_dict:
            continue
        if keep_existing_default:
            entry[s_key] = w_dict.get("slope_full", entry.get(s_key, 0.0))
            entry[t_key] = w_dict.get("final_trend", entry.get(t_key, 0))
        else:
            if w_dict.get("slope_full") is not None:
                entry[s_key] = w_dict.get("slope_full")
            if w_dict.get("final_trend") is not None:
                entry[t_key] = w_dict.get("final_trend")
        if w_dict.get("growth_coefficient") is not None:
            entry[g_key] = w_dict.get("growth_coefficient")


def compute_swing_wave_metrics(
    series_or_vals: list[Any],
    *,
    reversal_pct: float = 3.0,
    decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract raw ZigZag swing waves (>=3% reversals), active swing low/high, and rally/pullback % from a price series."""
    if not isinstance(series_or_vals, list) or len(series_or_vals) < 2:
        return {}
    vals: list[float] = []
    for item in series_or_vals:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            v = safe_float(item[1], 0.0)
        else:
            v = safe_float(item, 0.0)
        if v > 0.0:
            vals.append(v)
    if len(vals) < 2:
        return {}

    cur_v = vals[-1]
    thr = max(0.5, float(reversal_pct)) / 100.0
    pivots: list[Tuple[str, int, float]] = []
    direction = 0
    hi_px, hi_idx = vals[0], 0
    lo_px, lo_idx = vals[0], 0

    for idx, v in enumerate(vals[1:], start=1):
        if direction == 0:
            if v > hi_px:
                hi_px, hi_idx = v, idx
            if v < lo_px:
                lo_px, lo_idx = v, idx
            if lo_px > 0 and (v - lo_px) / lo_px >= thr and lo_idx < idx:
                pivots.append(("LOW", lo_idx, lo_px))
                direction = 1
                hi_px, hi_idx = v, idx
            elif hi_px > 0 and (hi_px - v) / hi_px >= thr and hi_idx < idx:
                pivots.append(("HIGH", hi_idx, hi_px))
                direction = -1
                lo_px, lo_idx = v, idx
        elif direction == 1:
            if v >= hi_px:
                hi_px, hi_idx = v, idx
            elif hi_px > 0 and (hi_px - v) / hi_px >= thr:
                pivots.append(("HIGH", hi_idx, hi_px))
                direction = -1
                lo_px, lo_idx = v, idx
        else:
            if v <= lo_px:
                lo_px, lo_idx = v, idx
            elif lo_px > 0 and (v - lo_px) / lo_px >= thr:
                pivots.append(("LOW", lo_idx, lo_px))
                direction = 1
                hi_px, hi_idx = v, idx

    chain: list[Tuple[str, int, float]] = list(pivots)
    if direction == 1 and chain and chain[-1][2] > 0:
        if (hi_px - chain[-1][2]) / chain[-1][2] >= thr:
            chain.append(("HIGH", hi_idx, hi_px))
    elif direction == -1 and chain and chain[-1][2] > 0:
        if (chain[-1][2] - lo_px) / chain[-1][2] >= thr:
            chain.append(("LOW", lo_idx, lo_px))

    swing_legs_pct: list[float] = []
    for i in range(1, len(chain)):
        p0, p1 = chain[i - 1][2], chain[i][2]
        if p0 > 0:
            swing_legs_pct.append(round(((p1 - p0) / p0) * 100.0, 2))

    major_low_px: Optional[float] = None
    if direction == 1 and pivots:
        swing_low_px = pivots[-1][2]
        swing_high_px = max(hi_px, cur_v)
        if len(pivots) >= 3 and pivots[-3][0] == "LOW" and pivots[-3][2] < swing_low_px:
            major_low_px = pivots[-3][2]
    elif direction == -1 and pivots:
        swing_high_px = pivots[-1][2]
        swing_low_px = min(lo_px, cur_v)
        if len(pivots) >= 2 and pivots[-2][0] == "LOW" and pivots[-2][2] < swing_low_px:
            major_low_px = pivots[-2][2]
    else:
        swing_low_px = min(vals)
        swing_high_px = max(vals)

    rally_pct = round(((cur_v / swing_low_px) - 1.0) * 100.0, 2) if swing_low_px > 0 else 0.0
    pullback_pct = round(((cur_v / swing_high_px) - 1.0) * 100.0, 2) if swing_high_px > 0 else 0.0
    major_rally_pct = (
        round(((cur_v / major_low_px) - 1.0) * 100.0, 2)
        if major_low_px and major_low_px > 0
        else None
    )

    eff_dec = decimals if decimals is not None else 6
    out: Dict[str, Any] = {
        "swing_low_px": round(swing_low_px, eff_dec),
        "swing_high_px": round(swing_high_px, eff_dec),
        "rally_from_swing_low_pct": rally_pct,
        "pullback_from_swing_high_pct": pullback_pct,
    }
    if major_rally_pct is not None and major_low_px is not None and major_rally_pct > rally_pct:
        out["major_swing_low_px"] = round(major_low_px, eff_dec)
        out["rally_from_major_low_pct"] = major_rally_pct
    if swing_legs_pct:
        out["recent_swings_7d"] = [f"{x:+.1f}%" for x in swing_legs_pct[-5:]]
        out["avg_swing_amp_7d_pct"] = round(
            sum(abs(x) for x in swing_legs_pct) / len(swing_legs_pct), 2
        )
    return out


def compute_7d_range_metrics(vals: list[float], decimals: Optional[int] = None) -> Dict[str, Any]:
    """Compute 7-day range geometry (low, high, drawdown, bounce, span, position) and 3%–8% swing waves from a price series."""
    if not isinstance(vals, list) or len(vals) < 2 or vals[0] <= 0:
        return {}
    raw_low, raw_high, cur_v = min(vals), max(vals), vals[-1]
    eff_dec = decimals if decimals is not None else 6
    l7_v = round(raw_low, eff_dec)
    h7_v = round(raw_high, eff_dec)
    metrics: Dict[str, Any] = {
        "chg_7d_pct": round(((cur_v - vals[0]) / vals[0]) * 100.0, 2),
        **_compute_horizon_range_dict(cur_v, l7_v, h7_v, "7d"),
    }
    metrics.update(compute_swing_wave_metrics(vals, decimals=eff_dec))
    return metrics


def _first_variant_dict(mapping: Any, variants: Any, default: Any = None) -> Any:
    """Return the first dict entry in `mapping` matching any candidate key in `variants` (case-insensitive)."""
    if not isinstance(mapping, dict):
        return default
    for cand in variants:
        val = casefold_get(mapping, cand)
        if isinstance(val, dict):
            return val
    return default


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
    if read_json_fn is None:
        read_json_fn = read_json_cached
    sym_cands = cache_symbol_variants(sym, include_base=True)

    ob_cand = None

    def _first_telemetry_dict(prefix: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        for cand in sym_cands:
            d = read_json_fn(os.path.join(cache_dir, f"{prefix}{cand}.json"))
            if isinstance(d, dict):
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
    m_info: Dict[str, Any] = {}
    if ob_cand:
        m_info["_source_cand"] = ob_cand
    if ob is not None:
        ob_same_quote = (not ob_cand) or (compatible_quote_suffixes(ob_cand) == compatible_quote_suffixes(sym))
        m_info["ob_imbalance"] = obj_attr(ob, "imbalance_ratio", obj_attr(ob, "ob_imbalance"))
        ob_mid = safe_pos_float(obj_attr(ob, "mid_price", obj_attr(ob, "ob_mid_price")), default=None)
        best_bid = safe_pos_float(obj_attr(ob, "best_bid"), default=None)
        best_ask = safe_pos_float(obj_attr(ob, "best_ask"), default=None)
        spread_bps = round_spread_bps(
            obj_attr(ob, "spread_bps"),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=ob_mid,
        )
        if spread_bps is not None:
            m_info["spread_bps"] = spread_bps
        if ob_same_quote:
            if ob_mid is not None:
                m_info["ob_mid_price"] = ob_mid
            if best_bid is not None:
                m_info["best_bid"] = best_bid
            if best_ask is not None:
                m_info["best_ask"] = best_ask
        for k_src, k_dst in (
            ("bid_depth_usd", "bid_depth_usd"),
            ("ask_depth_usd", "ask_depth_usd"),
            ("depth_1pct_bid_usd", "depth_1pct_bid_usd"),
            ("depth_1pct_ask_usd", "depth_1pct_ask_usd"),
            ("largest_bid_wall_usd", "bid_wall_usd"),
            ("largest_ask_wall_usd", "ask_wall_usd"),
        ):
            raw_v = obj_attr(ob, k_src)
            if raw_v is None and k_src != k_dst:
                raw_v = obj_attr(ob, k_dst)
            val = safe_nonneg_float(raw_v, default=None)
            m_info[k_dst] = round(val, 0) if val is not None else None
        if ob_same_quote:
            m_info["bid_wall_px"] = obj_attr(ob, "largest_bid_wall_price", obj_attr(ob, "bid_wall_px"))
            m_info["ask_wall_px"] = obj_attr(ob, "largest_ask_wall_price", obj_attr(ob, "ask_wall_px"))
        else:
            m_info["bid_wall_px"] = None
            m_info["ask_wall_px"] = None
    if deriv is not None:
        m_info["funding_rate"] = obj_attr(deriv, "funding_rate")
        m_info["pred_funding_rate"] = obj_attr(deriv, "predicted_funding_rate", obj_attr(deriv, "pred_funding_rate"))
        m_info["oi_usd"] = obj_attr(deriv, "open_interest_usd", obj_attr(deriv, "oi_usd"))
    if whale is not None:
        m_info["whale_long_pct"] = obj_attr(whale, "top_traders_long_pct", obj_attr(whale, "whale_long_pct"))
        m_info["whale_long_ratio"] = obj_attr(whale, "top_traders_long_ratio", obj_attr(whale, "whale_long_ratio"))
        m_info["taker_buy_sell_ratio"] = obj_attr(whale, "taker_buy_sell_ratio")
        m_info["taker_buy_vol_usd"] = obj_attr(whale, "taker_buy_vol_usd")
        m_info["taker_sell_vol_usd"] = obj_attr(whale, "taker_sell_vol_usd")
        m_info["oi_1h_chg_pct"] = obj_attr(whale, "open_interest_1h_change_pct", obj_attr(whale, "oi_1h_chg_pct"))
        m_info["divergence_regime"] = obj_attr(whale, "divergence_regime")
        if not safe_pos_float(m_info.get("oi_usd"), default=None):
            w_oi = safe_pos_float(obj_attr(whale, "open_interest_usd", obj_attr(whale, "oi_usd")), default=None)
            if w_oi is not None:
                m_info["oi_usd"] = w_oi
        if m_info.get("funding_rate") is None:
            w_fr = safe_float(obj_attr(whale, "funding_rate"), default=None)
            if w_fr is not None:
                m_info["funding_rate"] = w_fr
    enrich_microstructure_cvd(m_info)
    return m_info


def enrich_microstructure_cvd(
    minfo: Dict[str, Any],
    pinfo: Optional[Dict[str, Any]] = None,
) -> None:
    """Populate 1% orderbook depth / $2k slippage and `cvd_15m_usd`, `cvd_15m_base`, `cvd_regime` on `minfo`."""
    if not isinstance(minfo, dict):
        return
    b_bid = safe_pos_float(minfo.get("best_bid"), default=0.0) or 0.0
    b_ask = safe_pos_float(minfo.get("best_ask"), default=0.0) or 0.0
    if b_bid > 0.0 and b_ask >= b_bid:
        if not safe_pos_float(minfo.get("ob_mid_price"), default=None):
            minfo["ob_mid_price"] = round((b_bid + b_ask) / 2.0, 6)
        if minfo.get("spread_bps") is None:
            sp_bps = round_spread_bps(
                None,
                best_bid=b_bid,
                best_ask=b_ask,
                mid_price=safe_pos_float(minfo.get("ob_mid_price"), default=None),
            )
            if sp_bps is not None:
                minfo["spread_bps"] = sp_bps
    if not safe_pos_float(minfo.get("depth_1pct_bid_usd"), default=None) and (
        bd_f := safe_pos_float(minfo.get("bid_depth_usd"), default=None)
    ):
        minfo["depth_1pct_bid_usd"] = round(bd_f * (1.0 / 1.5), 0)
    if not safe_pos_float(minfo.get("depth_1pct_ask_usd"), default=None) and (
        ad_f := safe_pos_float(minfo.get("ask_depth_usd"), default=None)
    ):
        minfo["depth_1pct_ask_usd"] = round(ad_f * (1.0 / 1.5), 0)
    if minfo.get("est_slip_2k_bps") is None:
        d1_b = safe_pos_float(minfo.get("depth_1pct_bid_usd"), default=0.0) or 0.0
        d1_a = safe_pos_float(minfo.get("depth_1pct_ask_usd"), default=0.0) or 0.0
        eff_d1 = min(d1_b, d1_a) if (d1_b > 0.0 and d1_a > 0.0) else max(d1_b, d1_a)
        if eff_d1 > 0.0:
            sp_half = (safe_nonneg_float(minfo.get("spread_bps"), default=2.0) or 2.0) * 0.5
            minfo["est_slip_2k_bps"] = round(min(250.0, max(0.5, sp_half + (2000.0 / (2.0 * eff_d1)) * 100.0)), 1)
    cvd_usd = safe_float(minfo.get("cvd_15m_usd"), default=None)
    if cvd_usd is None:
        tb_v = safe_nonneg_float(minfo.get("taker_buy_vol_usd"), default=None)
        ts_v = safe_nonneg_float(minfo.get("taker_sell_vol_usd"), default=None)
        if tb_v is not None and ts_v is not None and (tb_v + ts_v) > 0.0:
            cvd_usd = round(tb_v - ts_v, 0)
            minfo["cvd_15m_usd"] = cvd_usd
    if cvd_usd is None:
        return
    p_dict = pinfo if isinstance(pinfo, dict) else {}
    m_px = (
        safe_pos_float(p_dict.get("price"), default=None)
        or safe_pos_float(minfo.get("ob_mid_price"), default=None)
        or ((b_bid + b_ask) / 2.0 if b_bid > 0.0 and b_ask >= b_bid else None)
    )
    if m_px is not None and m_px > 0.0:
        b_dec = 3 if m_px >= 100.0 else (2 if m_px >= 1.0 else 1)
        minfo["cvd_15m_base"] = round(cvd_usd / m_px, b_dec)
    if not minfo.get("cvd_regime") or pinfo is not None:
        ob_imb = safe_float(minfo.get("ob_imbalance"), default=0.5) or 0.5
        has_chg_15m = "chg_15m_pct" in p_dict and p_dict.get("chg_15m_pct") is not None
        chg_15m = safe_float(p_dict.get("chg_15m_pct"), default=0.0) or 0.0
        div_reg = str(minfo.get("divergence_regime") or minfo.get("div_regime") or "").strip().lower()
        if cvd_usd > 0.0:
            if has_chg_15m:
                minfo["cvd_regime"] = "ask_absorbing" if chg_15m <= 0.02 else "aggressive_buying"
            else:
                minfo["cvd_regime"] = "ask_absorbing" if ob_imb <= 0.44 else "aggressive_buying"
        elif cvd_usd < 0.0:
            if has_chg_15m:
                is_bid_absorbing = chg_15m >= 0.05 or (
                    chg_15m >= -0.05
                    and (ob_imb >= 0.54 or div_reg in ("short_covering", "bullish_absorption"))
                )
            else:
                is_bid_absorbing = ob_imb >= 0.58 or div_reg in ("short_covering", "bullish_absorption")
            minfo["cvd_regime"] = "spot_absorbing" if is_bid_absorbing else "aggressive_market_dumps"
        else:
            minfo["cvd_regime"] = "balanced_flow"


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
    cands: list[float] = []
    for raw_lvl, pad in levels_and_pads:
        lvl = safe_pos_float(raw_lvl, default=0.0) or 0.0
        if lvl <= 0.0:
            continue
        swept = lvl * pad
        dist_ratio = ((swept - cur_px) if is_short else (cur_px - swept)) / cur_px
        if not (min_dist <= dist_ratio <= max_dist):
            continue
        if inner_guard_px is not None:
            if is_short and swept <= inner_guard_px * 1.01:
                continue
            if not is_short and swept >= inner_guard_px * 0.99:
                continue
        cands.append(swept)
    blended = 0.5 * base_px + 0.5 * (sum(cands) / len(cands)) if cands else base_px
    return max(bound_px, blended) if is_short else min(bound_px, blended)


def compute_liquidation_clusters(
    pinfo: Optional[Dict[str, Any]],
    minfo: Optional[Dict[str, Any]],
    *,
    decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute symmetric LONG (downside) and SHORT (upside) liquidation heatmap clusters and magnet bias.

    Synthesizes open interest (`oi_usd`), top-trader positioning (`whale_long_pct`),
    perpetual funding rate (`funding_rate`), 1h OI change (`oi_1h_chg_pct`), realized
    1h volatility (`vol_1h_pct`), and structural liquidity/stop pools (`val_px`/`vah_px`,
    `low_24h`/`high_24h`, `bid_wall_px`/`ask_wall_px`, `swing_low_px`/`swing_high_px`).
    """
    if not isinstance(minfo, dict):
        return {}
    p_dict = pinfo if isinstance(pinfo, dict) else {}
    cur_px = safe_pos_float(p_dict.get("price"), default=0.0) or 0.0
    b_bid = safe_pos_float(minfo.get("best_bid"), default=0.0) or 0.0
    b_ask = safe_pos_float(minfo.get("best_ask"), default=0.0) or 0.0
    ob_mid = safe_pos_float(minfo.get("ob_mid_price"), default=0.0) or (
        (b_bid + b_ask) / 2.0 if b_bid > 0.0 and b_ask >= b_bid else 0.0
    )
    if ob_mid > 0.0 and (cur_px <= 0.0 or _prices_diverged(cur_px, ob_mid)):
        cur_px = ob_mid
    if cur_px <= 0.0:
        return {}

    oi_usd = safe_pos_float(minfo.get("oi_usd"), default=0.0) or 0.0
    if oi_usd <= 0.0:
        b_depth = safe_nonneg_float(minfo.get("bid_depth_usd"), default=0.0) or 0.0
        a_depth = safe_nonneg_float(minfo.get("ask_depth_usd"), default=0.0) or 0.0
        if (b_depth + a_depth) > 0.0:
            oi_usd = (b_depth + a_depth) * 12.0
    if oi_usd <= 0.0:
        return {}

    raw_long_pct = clean_scale(minfo.get("whale_long_pct"), default=0.50, allow_percentage=True)
    long_share = min(0.85, max(0.15, float(raw_long_pct if raw_long_pct is not None else 0.50)))

    funding = safe_float(minfo.get("funding_rate"), default=0.0) or 0.0
    funding_tilt = max(-0.30, min(0.30, funding * 1500.0))
    eff_long_share = min(0.88, max(0.12, long_share + funding_tilt * 0.5))
    eff_short_share = 1.0 - eff_long_share

    vol_1h = max(0.35, min(6.0, safe_pos_float(p_dict.get("vol_1h_pct"), default=1.2) or 1.2))
    near_band_pct = max(1.2, min(4.5, vol_1h * 1.65))
    deep_band_pct = max(near_band_pct * 1.85, min(10.0, vol_1h * 3.4))

    long_near_raw = _blend_structural_liq_anchor(
        cur_px,
        cur_px * (1.0 - near_band_pct / 100.0),
        ((p_dict.get("val_px"), 0.995), (p_dict.get("low_24h"), 0.996), (minfo.get("bid_wall_px"), 0.997)),
        is_short=False,
        min_dist=0.012,
        max_dist=0.065,
        bound_px=cur_px * (1.0 - 0.012),
    )
    short_near_raw = _blend_structural_liq_anchor(
        cur_px,
        cur_px * (1.0 + near_band_pct / 100.0),
        ((p_dict.get("vah_px"), 1.005), (p_dict.get("high_24h"), 1.004), (minfo.get("ask_wall_px"), 1.003)),
        is_short=True,
        min_dist=0.012,
        max_dist=0.065,
        bound_px=cur_px * (1.0 + 0.012),
    )
    long_deep_raw = _blend_structural_liq_anchor(
        cur_px,
        min(cur_px * (1.0 - deep_band_pct / 100.0), long_near_raw * 0.978),
        ((p_dict.get("swing_low_px"), 0.995), (p_dict.get("low_7d"), 0.996)),
        is_short=False,
        min_dist=0.022,
        max_dist=0.14,
        bound_px=long_near_raw * 0.985,
        inner_guard_px=long_near_raw,
    )
    short_deep_raw = _blend_structural_liq_anchor(
        cur_px,
        max(cur_px * (1.0 + deep_band_pct / 100.0), short_near_raw * 1.022),
        ((p_dict.get("swing_high_px"), 1.005), (p_dict.get("high_7d"), 1.004)),
        is_short=True,
        min_dist=0.022,
        max_dist=0.14,
        bound_px=short_near_raw * 1.015,
        inner_guard_px=short_near_raw,
    )

    eff_dec = decimals if decimals is not None else (2 if cur_px >= 100.0 else (4 if cur_px >= 1.0 else 6))
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
        round(((px_v - cur_px) / cur_px) * 100.0, 2)
        for px_v in (long_liq_px, short_liq_px, long_liq_deep_px, short_liq_deep_px)
    )
    short_gravity = short_liq_usd / max(0.35, abs(short_liq_dist_pct))
    long_gravity = long_liq_usd / max(0.35, abs(long_liq_dist_pct))
    liq_magnet_bias = (
        "SHORT_SQUEEZE_MAGNET_UP" if short_gravity > long_gravity * 1.18
        else ("LONG_SWEEP_MAGNET_DOWN" if long_gravity > short_gravity * 1.18 else "BALANCED_LIQ_MAGNET")
    )
    return {
        "long_liq_px": long_liq_px, "long_liq_usd": long_liq_usd, "long_liq_dist_pct": long_liq_dist_pct,
        "short_liq_px": short_liq_px, "short_liq_usd": short_liq_usd, "short_liq_dist_pct": short_liq_dist_pct,
        "long_liq_deep_px": long_liq_deep_px, "long_liq_deep_usd": long_liq_deep_usd, "long_liq_deep_dist_pct": long_liq_deep_dist_pct,
        "short_liq_deep_px": short_liq_deep_px, "short_liq_deep_usd": short_liq_deep_usd, "short_liq_deep_dist_pct": short_liq_deep_dist_pct,
        "liq_magnet_bias": liq_magnet_bias,
    }

