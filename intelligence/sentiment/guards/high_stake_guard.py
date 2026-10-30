"""LLM high-stake guard validating large BUY orders (e.g. >= 1000 EUR)."""
from __future__ import annotations

import glob as _glob
import logging
import math
import os
import re
import statistics
import sys as _sys
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.parse as _urlparse

from botcore import (
    _parse_bool_flag,
    extract_current_price_item,
    fetch_public_json,
    load_key_value_conf,
    load_local_module,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    get_cachedb_dir,
    legacy_filenames_for,
    normalize_venue_name,
    resolve_kraken_lookup_symbol,
    split_symbol_assets,
)
from intelligence.external.collectors.orderbook_depth import round_spread_bps
from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.internal.state.volatility import adaptive_thresholds
from intelligence.internal.triggers.kalman_trigger import KalmanTrend
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    clean_symbol_key,
    filter_valid_severity_keywords,
    get_cache_filenames,
    is_cache_fresh,
    normalize_order_side,
    obj_attr,
    resolve_now_ts,
    resolve_order_notional,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from intelligence.sentiment.gemini_client import GeminiClient
from instrument_registry import casefold_get
from orchestratorOS.agy.topics import topic_spec
from providers.base import (
    _EXEC_QTY_KEYS,
    _UNFILLED_ORDER_STATUSES,
    aggregate_fills_by_order_id,
    extract_executed_qty,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
    extract_timestamp_ms,
    extract_timestamp_s,
    format_actor_tag,
    normalize_open_order,
    resolve_order_origin,
)
from providers.execution_audit import (
    build_last_cycle_execution_status,
    collect_known_ai_order_ids,
    extract_intent_notional,
    extract_intent_order_type,
    extract_intent_side,
    summarize_recent_ai_intents,
)
from providers.quantity import resolve_assets, venue_fee_cap_quantity
from state_io import (
    read_json_dict_recovering as read_json_cached,
    read_json_recovering,
    read_last_jsonl_dict,
    read_tail_jsonl_dicts,
    read_tail_jsonl_records,
    resolve_cachedb_filepath,
    save_json_state,
)

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
        if timed_rows:
            timed_rows.sort(key=lambda x: x[0])
            for _, px_f, vol_w in timed_rows:
                if vol_w <= 0.0:
                    vol_w = 1.0 + min(5.0, (abs(px_f - prev_px) / prev_px) * 1000.0) if prev_px else 1.0
                prev_px = px_f
                samples.append((px_f, vol_w))
        else:
            for px_f in scalar_rows:
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
                        out["prior_6h_low_px"] = round(prior_low, eff_dec)
                        out["recent_1h_low_px"] = round(recent_low, eff_dec)
                        out["reclaim_above_prior_low_pct"] = round(((cur_px / prior_low) - 1.0) * 100.0, 2)
                    elif (
                        recent_low >= prior_low
                        and ((recent_low / prior_low) - 1.0) * 100.0 <= 4.5
                        and cur_px > recent_low * 1.002
                    ):
                        out["bottom_structure_6h"] = "HIGHER_LOW_DOUBLE_BOTTOM"
                        out["prior_6h_low_px"] = round(prior_low, eff_dec)
                        out["recent_1h_low_px"] = round(recent_low, eff_dec)
                        out["higher_low_vs_prior_pct"] = round(((recent_low / prior_low) - 1.0) * 100.0, 2)
                    elif recent_low <= prior_low and cur_px <= prior_low and ((cur_px / recent_low) - 1.0) * 100.0 <= 2.5:
                        out["bottom_structure_6h"] = "FRESH_6H_LOW_TEST"
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
    cur_px = tinfo.get("current_price", fallback_price)
    if allow_price_update and safe_float(entry.get("price"), 0.0) <= 0 and cur_px is not None:
        entry["price"] = cur_px
    trend = tinfo.get("final_trend", entry.get("trend", 0))
    entry["trend"] = trend
    entry["growth_coefficient"] = tinfo.get("growth_coefficient", entry.get("growth_coefficient", 0.0))
    entry["slope_full"] = tinfo.get("slope_full", 0.0)
    entry["gradient_recent"] = tinfo.get("gradient_recent", 0.0)
    if not require_any_slope:
        entry["gradient_recent_fast"] = tinfo.get("gradient_recent_fast")
        entry["epsilon"] = tinfo.get("epsilon")
        entry["trend_fast"] = tinfo.get("trend_fast", trend)
        entry["vol_1h_pct"] = tinfo.get("vol_1h_pct")
    else:
        if tinfo.get("gradient_recent_fast") is not None:
            entry["gradient_recent_fast"] = tinfo.get("gradient_recent_fast")
        if tinfo.get("epsilon") is not None:
            entry["epsilon"] = tinfo.get("epsilon")
        if tinfo.get("vol_1h_pct") is not None:
            entry["vol_1h_pct"] = tinfo.get("vol_1h_pct")
    t_vol = safe_pos_float(tinfo.get("vol_1h_pct"), default=None)
    t_re = safe_pos_float(tinfo.get("adapt_reentry_pct"), default=None)
    t_vol_aligned = t_vol is None or (t_re is not None and abs(t_re - 2.0 * t_vol) <= 0.05)
    if t_vol_aligned and tinfo.get("adapt_reentry_pct") is not None:
        entry["adapt_reentry_pct"] = tinfo.get("adapt_reentry_pct")
    if t_vol_aligned and tinfo.get("adapt_dca_pct") is not None:
        entry["adapt_dca_pct"] = tinfo.get("adapt_dca_pct")
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
    live_px = w222.get("current_price") or w900.get("current_price") or fallback_price
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

    eff_dec = (
        decimals
        if decimals is not None
        else (2 if cur_px >= 100.0 else (4 if cur_px >= 1.0 else 6))
    )
    long_liq_px = round(long_near_raw, eff_dec)
    short_liq_px = round(short_near_raw, eff_dec)
    long_liq_deep_px = round(long_deep_raw, eff_dec)
    short_liq_deep_px = round(short_deep_raw, eff_dec)

    oi_chg_1h = safe_float(minfo.get("oi_1h_chg_pct"), default=0.0) or 0.0
    oi_heat_mult = min(1.25, max(0.85, 1.0 + (oi_chg_1h / 100.0) * 1.5))

    long_liq_usd = round(oi_usd * eff_long_share * 0.014 * oi_heat_mult, 0)
    short_liq_usd = round(oi_usd * eff_short_share * 0.014 * oi_heat_mult, 0)
    long_liq_deep_usd = round(oi_usd * eff_long_share * 0.024 * oi_heat_mult, 0)
    short_liq_deep_usd = round(oi_usd * eff_short_share * 0.024 * oi_heat_mult, 0)

    long_liq_dist_pct = round(((long_liq_px - cur_px) / cur_px) * 100.0, 2)
    short_liq_dist_pct = round(((short_liq_px - cur_px) / cur_px) * 100.0, 2)
    long_liq_deep_dist_pct = round(((long_liq_deep_px - cur_px) / cur_px) * 100.0, 2)
    short_liq_deep_dist_pct = round(((short_liq_deep_px - cur_px) / cur_px) * 100.0, 2)

    short_gravity = short_liq_usd / max(0.35, abs(short_liq_dist_pct))
    long_gravity = long_liq_usd / max(0.35, abs(long_liq_dist_pct))
    if short_gravity > long_gravity * 1.18:
        liq_magnet_bias = "SHORT_SQUEEZE_MAGNET_UP"
    elif long_gravity > short_gravity * 1.18:
        liq_magnet_bias = "LONG_SWEEP_MAGNET_DOWN"
    else:
        liq_magnet_bias = "BALANCED_LIQ_MAGNET"

    return {
        "long_liq_px": long_liq_px,
        "long_liq_usd": long_liq_usd,
        "long_liq_dist_pct": long_liq_dist_pct,
        "short_liq_px": short_liq_px,
        "short_liq_usd": short_liq_usd,
        "short_liq_dist_pct": short_liq_dist_pct,
        "long_liq_deep_px": long_liq_deep_px,
        "long_liq_deep_usd": long_liq_deep_usd,
        "long_liq_deep_dist_pct": long_liq_deep_dist_pct,
        "short_liq_deep_px": short_liq_deep_px,
        "short_liq_deep_usd": short_liq_deep_usd,
        "short_liq_deep_dist_pct": short_liq_deep_dist_pct,
        "liq_magnet_bias": liq_magnet_bias,
    }


_VENUE_DEFAULT_FEE_SPECS: Dict[str, tuple[str, float, float, float]] = {
    "binance": ("VIP0", 10.0, 10.0, 5.0),
    "kraken": ("PRO_TIER0", 16.0, 26.0, 1.0),
    "hyperliquid": ("SPOT_TIER0", 4.0, 7.0, 10.0),
    "t212": ("INVEST_RTH", 15.0, 15.0, 1.0),
}


def resolve_venue_fee_and_limits(
    venue: str,
    symbol: str,
    inst: Any = None,
    *,
    price: float = 0.0,
    free_quote: Optional[float] = None,
    free_base: Optional[float] = None,
    spread_bps: Optional[float] = None,
    gateway_max_usd: float = 2000.0,
) -> Dict[str, Any]:
    """Resolve active venue fee tier, maker vs taker bps, live spread, and exchange/account order limits (0 ms, zero network calls)."""
    v_norm = normalize_venue_name(venue)
    fee_tier, maker_bps, taker_bps, min_notional = _VENUE_DEFAULT_FEE_SPECS.get(
        v_norm, ("STANDARD", 15.0, 15.0, 5.0)
    )

    px_f = safe_pos_float(price, default=0.0) or 0.0
    if v_norm == "t212":
        default_step = 0.001
        default_tick = 0.01
    elif v_norm == "hyperliquid":
        default_step = 0.0001 if px_f >= 1000.0 else 0.01
        default_tick = 0.01 if px_f >= 100.0 else (0.001 if px_f >= 1.0 else 0.0001)
    else:
        default_step = 0.0001 if px_f >= 1000.0 else (0.001 if px_f >= 50.0 else 0.01)
        default_tick = 0.01 if px_f >= 10.0 else (0.001 if px_f >= 1.0 else 0.0001)
    min_qty = default_step
    step_size = default_step
    tick_size = default_tick

    params = getattr(inst, "params", {}) if inst is not None and isinstance(getattr(inst, "params", None), dict) else {}
    if params.get("fee_tier"):
        fee_tier = str(params["fee_tier"]).strip()
    maker_bps = float(safe_nonneg_float(params.get("maker_fee_bps"), default=maker_bps))
    taker_bps = float(safe_nonneg_float(params.get("taker_fee_bps"), default=taker_bps))
    min_notional = float(safe_pos_float(params.get("min_notional"), default=min_notional))
    min_qty = float(safe_pos_float(params.get("min_qty"), default=min_qty))
    step_size = float(safe_pos_float(params.get("step_size"), default=step_size))
    tick_size = float(safe_pos_float(params.get("tick_size"), default=tick_size))

    # Non-blocking inspection of already-cached provider metadata (Kraken / Hyperliquid / Binance)
    prov = getattr(inst, "_provider", None) if inst is not None else None
    if v_norm == "kraken" and prov is not None:
        sym_u = str(symbol or "").strip().upper()
        kr_cands: list[str] = []
        for kc in (symbol, sym_u, clean_symbol_key(sym_u)):
            if isinstance(kc, str) and kc and kc not in kr_cands:
                kr_cands.append(kc)
        if "USD" in compatible_quote_suffixes(sym_u):
            kr_lk = resolve_kraken_lookup_symbol(sym_u)
            if kr_lk and kr_lk not in kr_cands:
                kr_cands.append(kr_lk)
        minqty_map = getattr(prov, "_minqty", None)
        if isinstance(minqty_map, dict):
            mq_cached = next(
                (v for kc in kr_cands if (v := safe_pos_float(minqty_map.get(kc), default=None)) is not None),
                None,
            )
            if mq_cached is not None:
                min_qty = mq_cached
        client = getattr(prov, "_cli", None) or getattr(prov, "_client", None)
        cache_dict = getattr(client, "_CACHE", None)
        if not isinstance(cache_dict, dict):
            kc_mod = _sys.modules.get("kraken_client") or _sys.modules.get("kraken.kraken_client")
            cache_dict = getattr(kc_mod, "_CACHE", None)
        if isinstance(cache_dict, dict):
            def _cache_val(ckey: Any) -> Any:
                hit = cache_dict.get(ckey)
                return hit[1] if isinstance(hit, tuple) and len(hit) >= 2 else None

            p_meta = next(
                (m for kc in kr_cands if isinstance((m := _cache_val(f"pair:{kc}")), dict)),
                None,
            )
            if p_meta is None:
                p_meta = next(
                    (
                        v
                        for kc in kr_cands
                        if isinstance((ap_res := _cache_val(("AssetPairs", (("pair", kc),)))), dict)
                        for v in ap_res.values()
                        if isinstance(v, dict)
                    ),
                    None,
                )
            if p_meta is None and isinstance((ap_all_res := _cache_val(("AssetPairs", ()))), dict):
                p_meta = next(
                    (ap_all_res[kc] for kc in kr_cands if isinstance(ap_all_res.get(kc), dict)),
                    None,
                )
            if isinstance(p_meta, dict):
                def _tier0_bps(fee_key: str) -> Optional[float]:
                    tbl = p_meta.get(fee_key)
                    if isinstance(tbl, list) and tbl and isinstance(tbl[0], (list, tuple)) and len(tbl[0]) >= 2:
                        fv = safe_nonneg_float(tbl[0][1], default=None)
                        return round(fv * 100.0, 2) if fv is not None else None
                    return None

                if (m_bps := _tier0_bps("fees_maker")) is not None:
                    maker_bps = m_bps
                if (t_bps := _tier0_bps("fees")) is not None:
                    taker_bps = t_bps
                ord_min = safe_pos_float(p_meta.get("ordermin"), default=None)
                if ord_min is not None:
                    min_qty = ord_min
                cost_min = safe_pos_float(p_meta.get("costmin"), default=None)
                if cost_min is not None:
                    min_notional = cost_min
                lot_dec = safe_nonneg_float(p_meta.get("lot_decimals"), default=None)
                if lot_dec is not None:
                    step_size = round(10.0 ** (-int(lot_dec)), 8)
                pair_dec = safe_nonneg_float(p_meta.get("pair_decimals"), default=None)
                if pair_dec is not None:
                    tick_size = round(10.0 ** (-int(pair_dec)), 8)
    elif v_norm == "hyperliquid" and prov is not None:
        client = getattr(prov, "_client", None)
        base_tok = canonical_base_asset(symbol)
        sz_dec = None
        meta_cache = getattr(prov, "_meta_cache", None) or getattr(client, "_meta_cache", None)
        if isinstance(meta_cache, dict):
            m_tok = meta_cache.get(base_tok)
            if isinstance(m_tok, dict):
                sz_dec = safe_nonneg_float(m_tok.get("szDecimals"), default=None)
        if sz_dec is None:
            info_obj = getattr(client, "info", None)
            c2a = getattr(info_obj, "coin_to_asset", None)
            a2sz = getattr(info_obj, "asset_to_sz_decimals", None)
            n2c = getattr(info_obj, "name_to_coin", None)
            if isinstance(c2a, dict) and isinstance(a2sz, dict):
                spot_pair = getattr(prov, "_spot_pair", None) or (
                    n2c.get(f"{base_tok}/USDC") if isinstance(n2c, dict) else None
                )
                for coin_k in (spot_pair, base_tok):
                    if coin_k and coin_k in c2a:
                        sz_dec = safe_nonneg_float(a2sz.get(c2a[coin_k]), default=None)
                        if sz_dec is not None:
                            break
        if sz_dec is not None:
            sz_i = int(sz_dec)
            step_size = round(10.0 ** (-sz_i), 8)
            min_qty = step_size
            if safe_pos_float(params.get("tick_size"), default=None) is None:
                max_px_dec = max(8 - sz_i, 0)
                if px_f > 0.0:
                    sig_dec = max(0, 4 - int(math.floor(math.log10(px_f))))
                    eff_px_dec = min(max_px_dec, sig_dec)
                else:
                    eff_px_dec = min(max_px_dec, 4)
                tick_size = round(10.0 ** (-eff_px_dec), 8)

    out: Dict[str, Any] = {
        "fee_tier": fee_tier,
        "maker_fee_bps": round(maker_bps, 2),
        "taker_fee_bps": round(taker_bps, 2),
        "maker_rebate_edge_bps": round(taker_bps - maker_bps, 2),
        "exchange_min_notional_usd": round(min_notional, 2),
        "min_qty": min_qty,
        "step_size": step_size,
        "tick_size": tick_size,
        "buy_buffer_pct": 99.5,
    }
    sp_rounded = round_spread_bps(spread_bps)
    if sp_rounded is not None:
        out["spread_bps"] = sp_rounded
    gw_max = safe_pos_float(gateway_max_usd, default=2000.0) or 2000.0
    fq_f = safe_nonneg_float(free_quote, default=None)
    if fq_f is not None:
        out["effective_max_buy_usd"] = round(min(gw_max, fq_f * 0.995), 2)
    fb_f = safe_nonneg_float(free_base, default=None)
    if fb_f is not None and px_f > 0.0:
        sellable_qty = venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f)
        out["effective_max_sell_usd"] = round(min(gw_max, sellable_qty * px_f), 2)
    return out


def round_opt_px(val: Any, decimals: int = 6) -> Optional[float]:
    """Round a positive price anchor to ``decimals``, returning None when absent or non-positive."""
    vf = safe_pos_float(val, default=None)
    return round(vf, decimals) if vf is not None else None


def resolve_symbol_price(
    symbol: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Optional[Dict[str, float]] = None,
    *,
    explicit_quote: Optional[str] = None,
) -> float:
    """Resolve a positive live price for ``symbol`` across quote-compatible symbol variants and optional fallback map."""
    if not symbol:
        return 0.0
    sym_quotes = compatible_quote_suffixes(symbol, explicit_quote)
    sym_variants = cache_symbol_variants(
        symbol,
        include_base="USD" in sym_quotes,
        quotes=sym_quotes,
    )
    for cand in sym_variants:
        entry = casefold_get(prices_and_trends, cand)
        cand_px = safe_pos_float(entry.get("price") if isinstance(entry, dict) else None, default=0.0) or 0.0
        if cand_px > 0:
            return cand_px
    if isinstance(fallback_prices, dict):
        for cand in sym_variants:
            cand_px = safe_pos_float(casefold_get(fallback_prices, cand), default=0.0) or 0.0
            if cand_px > 0:
                return cand_px
    return 0.0


def set_pos_anchor(
    pos: Dict[str, Any],
    px_key: str,
    pct_key: str,
    anchor_val: Any,
    cur_px: float,
    *,
    overwrite: bool = True,
) -> None:
    """Attach a cost/buy/sell/peak price anchor and its percentage delta to a position dict."""
    a_f = safe_pos_float(anchor_val, default=0.0) or 0.0
    if a_f <= 0:
        return
    if not overwrite and pos.get(px_key) is not None:
        return
    if px_key == "trailing_peak" and cur_px > a_f:
        a_f = cur_px
    pos[px_key] = round(a_f, 6)
    if cur_px > 0:
        pos[pct_key] = round(((cur_px - a_f) / a_f) * 100.0, 2)


def parse_venue_cycle_state(
    st_data: Dict[str, Any],
    symbol: str,
    prices_and_trends: Dict[str, Any],
    *,
    avg_decimals: int = 6,
    fallback_prices: Optional[Dict[str, float]] = None,
    include_meta: bool = False,
) -> Dict[str, Any]:
    """Normalize a venue cycle ``.state_*.json`` dict (Kraken, Hyperliquid, Trading212) into a uniform schema."""
    qty = safe_nonneg_float(st_data.get("qty"), default=0.0) or 0.0
    raw_cost = st_data.get("cost_usd") if st_data.get("cost_usd") is not None else st_data.get("cost")
    cost = safe_nonneg_float(raw_cost, default=0.0) or 0.0
    avg_cost = (
        round(cost / qty, avg_decimals)
        if qty > 1e-12 and cost > 0
        else (
            round_opt_px(
                st_data.get("avg_cost") or st_data.get("avg_price") or st_data.get("entry_price"),
                avg_decimals,
            )
            if qty > 1e-12
            else None
        )
    )
    px_f = resolve_symbol_price(symbol, prices_and_trends, fallback_prices)
    dd_pct = round(((px_f - avg_cost) / avg_cost) * 100.0, 2) if avg_cost and px_f > 0 else None
    last_buy_px = (
        st_data.get("last_buy_price")
        if st_data.get("last_buy_price") is not None
        else st_data.get("last_open_price")
    )
    cyc_entry: Dict[str, Any] = {
        "qty": round(qty, 6),
        "cost_usd": round(cost, 2),
        "avg_cost_px": avg_cost,
        "entry_price": round_opt_px(st_data.get("entry_price"), avg_decimals),
        "last_buy_price": round_opt_px(last_buy_px, avg_decimals),
        "last_sell_price": round_opt_px(st_data.get("last_sell_price"), avg_decimals),
        "last_exit_kind": st_data.get("last_exit_kind"),
        "dca_buys": st_data.get("dca_buys"),
        "dd_pct": dd_pct,
        "pnl_vs_avg_pct": dd_pct,
    }
    pos_peak = round_opt_px(st_data.get("pos_peak") or st_data.get("trail_peak"), avg_decimals)
    if pos_peak is not None and qty > 1e-12:
        cyc_entry["pos_peak"] = pos_peak
        if px_f > 0 and pos_peak > 0:
            eff_peak = max(pos_peak, px_f)
            cyc_entry["dd_from_peak_pct"] = round(((px_f - eff_peak) / eff_peak) * 100.0, 2)
    raw_realized = next(
        (
            st_data[rk]
            for rk in ("realized_net", "realized_net_usd", "realized_pnl", "realized_pnl_usd", "realized_gross")
            if st_data.get(rk) is not None
        ),
        None,
    )
    realized_pnl = safe_float(raw_realized, 0.0) or 0.0
    if abs(realized_pnl) > 1e-9:
        cyc_entry["realized_pnl_usd"] = round(realized_pnl, 2)
    tp_sold = st_data.get("tp_sold") if st_data.get("tp_sold") is not None else st_data.get("tp_sold_levels")
    if isinstance(tp_sold, list) and tp_sold:
        cyc_entry["tp_sold_levels"] = tp_sold
    raw_orders = st_data.get("orders")
    if isinstance(raw_orders, list) and raw_orders:
        close_orders = [
            {
                "side": o.get("side"),
                "px": o.get("px") or o.get("price") or o.get("limit"),
                "sz": o.get("sz") or o.get("qty"),
                "role": o.get("role") or o.get("kind"),
            }
            for o in raw_orders
            if isinstance(o, dict)
        ]
        if close_orders:
            cyc_entry["resting_orders"] = close_orders
    if include_meta:
        raw_entry_ts = safe_pos_float(st_data.get("entry_ts"), default=None)
        if raw_entry_ts is not None:
            cyc_entry["_entry_ts_ms"] = (
                int(round(raw_entry_ts * 1000.0)) if raw_entry_ts < 1e11 else int(round(raw_entry_ts))
            )
        raw_sell_ts = safe_pos_float(st_data.get("last_sell_ts"), default=None)
        if raw_sell_ts is not None:
            cyc_entry["_last_sell_ts_ms"] = (
                int(round(raw_sell_ts * 1000.0)) if raw_sell_ts < 1e11 else int(round(raw_sell_ts))
            )
    return cyc_entry


def harvest_venue_cycles(
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Dict[str, float],
    read_json_fn: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest normalized ``.state_*.json`` DCA cycle states for Kraken, Hyperliquid, and Trading212."""
    kraken_cycles: Dict[str, Any] = {}
    kraken_dir = os.path.join(workspace_dir, "kraken")
    if os.path.isdir(kraken_dir):
        for st_path in sorted(_glob.glob(os.path.join(kraken_dir, ".state_*.json"))):
            pair_name = os.path.basename(st_path)[len(".state_") : -len(".json")].upper()
            if not pair_name.isalnum() or not pair_name.endswith(("USD", "EUR", "USDC", "USDT")):
                continue
            st_data = read_json_fn(st_path)
            if isinstance(st_data, dict):
                kraken_cycles[pair_name] = parse_venue_cycle_state(
                    st_data,
                    pair_name,
                    prices_and_trends,
                    avg_decimals=6,
                    fallback_prices=fallback_prices,
                    include_meta=True,
                )

    hyperliquid_cycles: Dict[str, Any] = {}
    hl_dir = os.path.join(workspace_dir, "hyperliquid")
    if os.path.isdir(hl_dir):
        hl_state_paths = sorted(
            _glob.glob(os.path.join(hl_dir, ".state_*.json")),
            key=lambda p: (
                0 if os.path.basename(p).upper().endswith(("_LONG.JSON", "_SHORT.JSON")) else 1,
                os.path.basename(p),
            ),
        )
        for st_path in hl_state_paths:
            raw_tok = os.path.basename(st_path)[len(".state_") : -len(".json")].upper()
            if raw_tok.startswith("DN_") or raw_tok == "DN":
                continue
            for sfx in ("_LONG", "_SHORT"):
                if raw_tok.endswith(sfx):
                    raw_tok = raw_tok[: -len(sfx)]
                    break
            if not raw_tok.isalnum():
                continue
            pair_name = raw_tok if raw_tok.endswith("USDC") else f"{raw_tok}USDC"
            st_data = read_json_fn(st_path)
            if isinstance(st_data, dict):
                hyperliquid_cycles[pair_name] = parse_venue_cycle_state(
                    st_data,
                    pair_name,
                    prices_and_trends,
                    avg_decimals=6,
                    fallback_prices=fallback_prices,
                    include_meta=True,
                )

    all_t212_cycles: Dict[str, Any] = {}
    t212_dir = os.path.join(workspace_dir, "212trading")
    if os.path.isdir(t212_dir):
        for st_path in sorted(_glob.glob(os.path.join(t212_dir, ".state_*.json"))):
            fname = os.path.basename(st_path)
            if fname.startswith(".."):
                continue
            ticker = fname[len(".state_") : -len(".json")].upper()
            if not ticker.replace("_", "").isalnum():
                continue
            st_data = read_json_fn(st_path)
            if isinstance(st_data, dict):
                all_t212_cycles[ticker] = parse_venue_cycle_state(
                    st_data,
                    ticker,
                    prices_and_trends,
                    avg_decimals=4,
                    fallback_prices=fallback_prices,
                    include_meta=True,
                )

    return kraken_cycles, hyperliquid_cycles, all_t212_cycles


def harvest_kraken_xstocks(
    kraken_dir: str,
    prices_and_trends: Dict[str, Any],
    all_t212_cycles: Dict[str, Any],
    read_json_fn: Any,
) -> Optional[Dict[str, Any]]:
    """Harvest Kraken read-only xStocks holdings and link to Trading212 sibling cycle."""
    xst_path = os.path.join(kraken_dir, "xstock_state.json")
    xst_data = read_json_fn(xst_path) if os.path.exists(xst_path) else None
    if not isinstance(xst_data, dict):
        return None
    xstock_cfg = load_key_value_conf(kraken_dir, root_dir=kraken_dir)
    known_assets = xst_data.get("known_assets") if isinstance(xst_data.get("known_assets"), dict) else {}
    x_regex = xstock_cfg.get("xstock_regex") or "SPCX|SPACEX"
    x_holdings: Dict[str, float] = {}
    for ak, av in known_assets.items():
        ak_str = str(ak).strip()
        av_f = safe_float(av, 0.0) or 0.0
        if av_f <= 0:
            continue
        is_xstock_tok = ak_str.upper().endswith(("X.T", ".T"))
        if not is_xstock_tok:
            try:
                is_xstock_tok = bool(re.search(x_regex, ak_str, re.IGNORECASE))
            except re.error:
                is_xstock_tok = False
        if is_xstock_tok:
            x_holdings[ak_str] = round(av_f, 6)
    alloc_info = xst_data.get("allocated") if isinstance(xst_data.get("allocated"), dict) else None
    raw_pair = xst_data.get("pair")
    listed_pair = (
        str(raw_pair).strip().upper()
        if isinstance(raw_pair, str) and raw_pair.strip() and raw_pair.strip().lower() not in ("none", "null", "nan")
        else None
    )
    if not (x_holdings or alloc_info or listed_pair):
        return None
    underlying = (xstock_cfg.get("xstock_yahoo") or "SPCX").strip().upper()
    t212_sib = f"{underlying}_US_EQ"
    alloc_px = safe_pos_float(xstock_cfg.get("xstock_alloc_price"), default=None)
    live_px = resolve_symbol_price(listed_pair or underlying, prices_and_trends)
    xst_summary: Dict[str, Any] = {
        "underlying": underlying,
        "t212_sibling": t212_sib,
        "holdings": x_holdings,
        "allocated": alloc_info,
        "listed_pair": listed_pair,
        "alloc_price": alloc_px,
        "tp1_alert_pct": safe_pos_float(xstock_cfg.get("xstock_tp_alert_pct"), default=None),
        "tp2_alert_pct": safe_pos_float(xstock_cfg.get("xstock_tp2_alert_pct"), default=None),
        "sl_alert_pct": safe_pos_float(xstock_cfg.get("xstock_sl_alert_pct"), default=None),
    }
    if live_px > 0:
        xst_summary["live_price"] = live_px
        if alloc_px is not None:
            xst_summary["pnl_vs_alloc_pct"] = round(((live_px - alloc_px) / alloc_px) * 100.0, 2)
    sib_cyc = _first_variant_dict(all_t212_cycles, (t212_sib, underlying))
    if isinstance(sib_cyc, dict):
        xst_summary["t212_sibling_cycle"] = {k: v for k, v in sib_cyc.items() if not str(k).startswith("_")}
    return xst_summary


def build_cross_market_context(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    all_t212_cycles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build raw, objective cross-market telemetry (Crypto <-> US Equities) with zero pre-calculated correlations or bias."""
    all_t212_cycles = all_t212_cycles if isinstance(all_t212_cycles, dict) else {}
    ctx: Dict[str, Any] = {}
    def _bellwether_rank(s: str, p: Dict[str, Any]) -> Tuple[int, int, int]:
        return (
            1 if "USD" in compatible_quote_suffixes(s) else 0,
            1 if s.endswith(("USDC", "_US_EQ", "_EQ")) else 0,
            len(p),
        )

    by_base: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for sym, pinfo in sorted(prices_and_trends.items()):
        if not isinstance(pinfo, dict):
            continue
        b_canon = canonical_base_asset(sym)
        if not b_canon:
            continue
        prev = by_base.get(b_canon)
        if prev is None or _bellwether_rank(sym, pinfo) > _bellwether_rank(prev[0], prev[1]):
            by_base[b_canon] = (sym, pinfo)

    raw_horizon_keys = (
        "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
        "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
        "chg_7d_pct", "pos_24h_pct", "poc_px", "vah_px", "val_px", "va_regime",
        "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel",
    )

    crypto_bellwethers: Dict[str, Any] = {}
    for b_canon, (sym, pinfo) in sorted(by_base.items()):
        if sym.endswith(("_EQ", "_US_EQ")):
            continue
        px_f = safe_float(pinfo.get("price"), 0.0) or 0.0
        if px_f <= 0:
            continue
        cb_entry: Dict[str, Any] = {"symbol": sym, "price": px_f}
        for k_copy in raw_horizon_keys:
            if pinfo.get(k_copy) is not None:
                cb_entry[k_copy] = pinfo.get(k_copy)
        sym_quotes = compatible_quote_suffixes(sym)
        m_info = microstructure.get(sym)
        m_same_quote = True
        if not isinstance(m_info, dict):
            m_info, m_same_quote = next(
                (
                    (mv, compatible_quote_suffixes(mk) == sym_quotes)
                    for mk, mv in microstructure.items()
                    if canonical_base_asset(mk) == b_canon and isinstance(mv, dict)
                ),
                (None, False),
            )
        if isinstance(m_info, dict):
            for mk_copy in (
                "ob_imbalance", "spread_bps", "whale_long_pct", "funding_rate", "oi_1h_chg_pct",
                "cvd_15m_usd", "cvd_regime", "depth_1pct_bid_usd", "depth_1pct_ask_usd",
                *(("long_liq_px", "long_liq_usd", "short_liq_px", "short_liq_usd", "liq_magnet_bias") if m_same_quote else ()),
            ):
                if m_info.get(mk_copy) is not None:
                    cb_entry[mk_copy] = m_info.get(mk_copy)
        crypto_bellwethers[b_canon] = cb_entry
    if crypto_bellwethers:
        ctx["crypto_bellwethers"] = crypto_bellwethers

    equity_bellwethers: Dict[str, Any] = {}
    eq_syms = sorted(
        {s for s in prices_and_trends if str(s).upper().endswith(("_EQ", "_US_EQ"))}
        | {s for s in all_t212_cycles if str(s).upper().endswith(("_EQ", "_US_EQ"))}
    )
    for eq_sym in eq_syms:
        pinfo = prices_and_trends.get(eq_sym) if isinstance(prices_and_trends.get(eq_sym), dict) else {}
        t_cyc = _first_variant_dict(
            all_t212_cycles,
            cache_symbol_variants(eq_sym, include_original=True, include_base=True),
            default={},
        )
        eq_entry: Dict[str, Any] = {"symbol": eq_sym}
        if pinfo.get("price") is not None:
            eq_entry["price"] = pinfo.get("price")
        for k_copy in raw_horizon_keys:
            if pinfo.get(k_copy) is not None:
                eq_entry[k_copy] = pinfo.get(k_copy)
        if t_cyc:
            eq_entry["t212_cycle"] = {
                k: t_cyc.get(k)
                for k in ("qty", "avg_cost_px", "entry_price", "last_buy_price", "dd_pct")
                if t_cyc.get(k) is not None
            }
        if len(eq_entry) > 1:
            equity_bellwethers[eq_sym] = eq_entry
    if equity_bellwethers:
        ctx["equity_bellwethers"] = equity_bellwethers

    return ctx


def load_venue_trailing_maps(
    cache_dir: str,
    workspace_dir: str,
    read_json_fn: Any = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Load trailing peak maps from cachedb/trailing_state.json and kraken/trailing_state.json."""
    _rj = read_json_fn or read_json_cached
    trailing_map: Dict[str, Dict[str, Any]] = {}
    trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]] = {"binance": {}, "kraken": {}}
    for trail_venue, raw_trail, default_quote in (
        ("binance", _rj(os.path.join(cache_dir, "trailing_state.json")), ""),
        ("kraken", _rj(os.path.join(workspace_dir, "kraken", "trailing_state.json")), "USD"),
    ):
        if not isinstance(raw_trail, dict):
            continue
        for k_asset, k_st in raw_trail.items():
            k_up = str(k_asset).strip().upper()
            if not k_up:
                continue
            k_key = k_up if (not default_quote or split_symbol_assets(k_up) is not None) else f"{k_up}{default_quote}"
            norm_st: Optional[Dict[str, Any]] = None
            if isinstance(k_st, dict) and k_st.get("peak") is not None:
                norm_st = k_st
            else:
                k_pk = safe_pos_float(k_st, default=0.0) or 0.0
                if k_pk > 0:
                    norm_st = {"peak": k_pk}
            if norm_st is not None:
                trailing_map.setdefault(k_key, norm_st)
                trailing_by_venue[trail_venue].setdefault(k_key, norm_st)
    return trailing_map, trailing_by_venue


def harvest_tradable_positions_and_orders(
    inst_map: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    trailing_map: Dict[str, Dict[str, Any]],
    trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    gateway_max_usd: float = 2000.0,
    cache_dir: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Harvest spot wallet balances, trailing peaks, and open orders across tradable instruments."""
    tradable_positions: List[Dict[str, Any]] = []
    open_orders_list: List[Dict[str, Any]] = []
    venue_quote_cash: Dict[str, Dict[str, Any]] = {}
    inst_by_sym: Dict[str, Any] = {}
    balance_memo: Dict[tuple[str, str], Optional[float]] = {}
    bcache_path = (
        str(resolve_cachedb_filepath(os.path.join(cache_dir, "venue_balances_state.json")))
        if cache_dir
        else None
    )
    cached_bals: Dict[str, Any] = (
        read_json_cached(bcache_path)
        if bcache_path and os.path.exists(bcache_path)
        else {}
    ) or {}
    if not isinstance(cached_bals, dict):
        cached_bals = {}
    updated_bals: Dict[str, float] = {}

    def _query_free_bal(inst_obj: Any, asset: str) -> Optional[float]:
        if not asset:
            return None
        v_name = str(getattr(inst_obj, "provider_name", "")).lower()
        key = (v_name, asset.upper())
        if key in balance_memo:
            return balance_memo[key]
        prov = getattr(inst_obj, "_provider", None)
        free_fn = getattr(prov, "free_balance", None)
        val_f: Optional[float] = None
        if callable(free_fn):
            try:
                val_f = safe_nonneg_float(free_fn(asset), default=None)
            except Exception:
                val_f = None
        bkey = f"{v_name}:{asset.upper()}"
        if val_f is not None:
            updated_bals[bkey] = round(val_f, 8)
        elif bkey in cached_bals:
            val_f = safe_nonneg_float(cached_bals.get(bkey), default=None)
        balance_memo[key] = val_f
        return val_f

    if isinstance(inst_map, dict):
        for inst in inst_map.values():
            sym = str(getattr(inst, "symbol", "") or "").strip().upper()
            venue = str(getattr(inst, "provider_name", "") or "").strip().lower()
            if not sym or not venue:
                continue
            res_b, res_q = resolve_assets(sym, getattr(inst, "base", None), getattr(inst, "quote", None))
            base = res_b or sym
            quote = res_q or "USD"
            inst_by_sym[sym] = inst
            inst_by_sym[f"{normalize_venue_name(venue)}:{sym}"] = inst

            px = prices_and_trends.get(sym, {}).get("price")
            px_f = safe_pos_float(px, default=0.0) or 0.0
            if px_f <= 0:
                try:
                    get_px_fn = getattr(inst._provider, "get_current_price", None)
                    if callable(get_px_fn):
                        live_px = safe_pos_float(get_px_fn(sym), default=0.0) or 0.0
                        if live_px > 0:
                            px = live_px
                            px_f = live_px
                except Exception:
                    px = None
                    px_f = 0.0
            if px_f <= 0:
                px_f = resolve_symbol_price(sym, prices_and_trends, explicit_quote=quote)
                if px_f > 0:
                    px = px_f
            sym_entry = prices_and_trends.setdefault(
                sym, {"price": px if px_f > 0 else None, "trend": 0, "growth_coefficient": 0.0}
            )
            if px_f > 0 and (safe_pos_float(sym_entry.get("price"), default=0.0) or 0.0) <= 0:
                sym_entry["price"] = px

            base_free = _query_free_bal(inst, base)
            quote_free = _query_free_bal(inst, quote)
            quote_breakdown: Optional[Dict[str, float]] = None
            if quote.upper() == "USD":
                bd_fn = getattr(inst._provider, "usd_cash_breakdown", None)
                if callable(bd_fn):
                    try:
                        raw_bd = bd_fn()
                        if isinstance(raw_bd, dict) and raw_bd:
                            quote_breakdown = {
                                str(k): round(vf, 2)
                                for k, v in raw_bd.items()
                                if (vf := safe_pos_float(v, default=None)) is not None
                            } or None
                    except Exception:
                        quote_breakdown = None
            if quote_free is not None:
                cash_entry: Dict[str, Any] = {
                    "venue": venue,
                    "asset": quote,
                    "free": round(quote_free, 2),
                }
                if quote_breakdown:
                    cash_entry["breakdown"] = quote_breakdown
                venue_quote_cash[f"{venue}:{quote}"] = cash_entry

            pos_val_usd = (
                round(base_free * px_f, 2)
                if base_free is not None and px_f > 0
                else None
            )
            params = getattr(inst, "params", {}) if isinstance(getattr(inst, "params", None), dict) else {}
            v_norm = normalize_venue_name(venue)
            v_trail_map = trailing_by_venue.get(v_norm, {})
            sym_quotes = compatible_quote_suffixes(sym, quote)
            t_state = _first_variant_dict(
                v_trail_map,
                cache_symbol_variants(
                    sym,
                    include_original=True,
                    include_base="USD" in sym_quotes,
                    quotes=sym_quotes,
                ),
                default={},
            )
            if not t_state and v_norm == "binance" and isinstance((val_b := casefold_get(trailing_map, sym)), dict):
                t_state = val_b
            peak = t_state.get("peak")
            peak_f = safe_float(peak, 0.0)
            if peak_f > 0 and px_f > 0 and peak_f < px_f:
                peak_f = px_f
            dd_from_peak_pct = (
                round(((px_f - peak_f) / peak_f) * 100.0, 2)
                if px_f > 0 and peak_f > 0
                else None
            )
            pos_entry: Dict[str, Any] = {
                "venue": venue,
                "symbol": sym,
                "base": base,
                "quote": quote,
                "price": round_opt_px(px, 6) if px is not None else None,
                "free_base": round(base_free, 6) if base_free is not None else None,
                "value_usd": pos_val_usd,
                "free_quote": round(quote_free, 2) if quote_free is not None else None,
            }
            if quote_breakdown:
                pos_entry["free_quote_breakdown"] = quote_breakdown
            pos_entry.update(
                resolve_venue_fee_and_limits(
                    venue,
                    sym,
                    inst,
                    price=px_f,
                    free_quote=quote_free,
                    free_base=base_free,
                    gateway_max_usd=gateway_max_usd,
                )
            )
            min_notional_f = max(
                1.0,
                safe_float(pos_entry.get("exchange_min_notional_usd") or pos_entry.get("min_notional_usd"), 1.0),
            )
            has_held_base = base_free is None or (
                base_free > 1e-8 and (px_f <= 0.0 or base_free * px_f >= min_notional_f)
            )
            if peak is not None and peak_f > 0 and has_held_base:
                pos_entry["trailing_peak"] = round(peak_f, 6)
                pos_entry["dd_from_peak_pct"] = dd_from_peak_pct
            if params.get("trailing.pct"):
                pos_entry["trailing_pct_cfg"] = params.get("trailing.pct")
            if params.get("mt.gain") or params.get("mt.lost"):
                pos_entry["mt_gain_pct"] = params.get("mt.gain")
                pos_entry["mt_lost_pct"] = params.get("mt.lost")
            tradable_positions.append(pos_entry)

            open_fn = getattr(inst, "open_orders", None)
            if callable(open_fn):
                try:
                    raw_open = open_fn()
                    if isinstance(raw_open, list):
                        for oo in raw_open:
                            if isinstance(oo, dict):
                                open_orders_list.append(
                                    normalize_open_order(
                                        oo,
                                        venue=venue,
                                        symbol=sym,
                                        known_ai_order_ids=known_ai_order_ids,
                                    )
                                )
                except Exception:
                    pass

    if (
        bcache_path
        and updated_bals
        and ("PYTEST_CURRENT_TEST" not in os.environ or os.path.exists(bcache_path))
    ):
        try:
            merged_bals = {**cached_bals, **updated_bals}
            if merged_bals != cached_bals:
                os.makedirs(os.path.dirname(os.path.abspath(bcache_path)), exist_ok=True)
                save_json_state(bcache_path, merged_bals, indent=2)
        except Exception:
            pass

    return tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym


def enrich_cross_venue_trends(
    prices_and_trends: Dict[str, Any],
    trends_data: Any,
    kalman_data: Any,
    long_items: Any,
    priceanalysis_data: Any,
    t_trend_map: Any,
    watcher_price_items: Any,
) -> None:
    """Enrich every symbol with cross-venue short slopes, Kalman velocity, long trend, Weibull, and 7d range."""
    trends_items_map = (
        trends_data.get("items")
        if isinstance(trends_data, dict) and isinstance(trends_data.get("items"), dict)
        else (trends_data if isinstance(trends_data, dict) else {})
    )
    for sym, entry in list(prices_and_trends.items()):
        sym_quotes = compatible_quote_suffixes(sym)
        cand_keys = tuple(cache_symbol_variants(sym, include_original=True, include_base=True))

        has_slopes = any(
            entry.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h")
        )
        if not has_slopes and isinstance(trends_items_map, dict):
            for ck in cand_keys[1:]:
                tinfo = casefold_get(trends_items_map, ck)
                if isinstance(tinfo, dict) and apply_short_trend_item(
                    entry,
                    tinfo,
                    require_any_slope=True,
                    allow_price_update=(compatible_quote_suffixes(ck) == sym_quotes),
                ):
                    has_slopes = True
                    break
        if not has_slopes:
            for ck in cand_keys[1:]:
                sib = casefold_get(prices_and_trends, ck)
                if isinstance(sib, dict) and any(
                    sib.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h")
                ):
                    for k_inh in (
                        "slope_90s", "trend_90s", "growth_90s",
                        "slope_3_7m", "trend_3_7m", "growth_3_7m",
                        "slope_15m", "trend_15m", "growth_15m",
                        "slope_2_5h", "trend_2_5h", "growth_2_5h",
                        "gradient_recent_fast", "epsilon", "vol_1h_pct",
                    ):
                        if k_inh in sib and sib.get(k_inh) is not None and not entry.get(k_inh):
                            entry[k_inh] = sib[k_inh]
                    break

        if entry.get("vol_1h_pct") is None and isinstance(trends_items_map, dict):
            for ck in cand_keys:
                t_vol_info = casefold_get(trends_items_map, ck)
                if isinstance(t_vol_info, dict) and t_vol_info.get("vol_1h_pct") is not None:
                    entry["vol_1h_pct"] = t_vol_info.get("vol_1h_pct")
                    if entry.get("adapt_reentry_pct") is None and t_vol_info.get("adapt_reentry_pct") is not None:
                        entry["adapt_reentry_pct"] = t_vol_info.get("adapt_reentry_pct")
                    if entry.get("adapt_dca_pct") is None and t_vol_info.get("adapt_dca_pct") is not None:
                        entry["adapt_dca_pct"] = t_vol_info.get("adapt_dca_pct")
                    break

        if isinstance(kalman_data, dict):
            k_sym = _first_variant_dict(kalman_data, cand_keys)
            if isinstance(k_sym, dict):
                entry["kalman_vel"] = k_sym.get("kalman_vel")
                entry["kalman_vel_std"] = k_sym.get("kalman_vel_std")
                entry["kalman_trend"] = k_sym.get("kalman_trend")
                cur_vol = safe_pos_float(entry.get("vol_1h_pct"), default=None)
                k_vol = safe_pos_float(k_sym.get("vol_1h_pct"), default=None)
                vol_matches_kalman = cur_vol is None or (k_vol is not None and abs(cur_vol - k_vol) <= 0.05)
                if cur_vol is None and k_vol is not None:
                    entry["vol_1h_pct"] = k_vol
                if vol_matches_kalman and entry.get("adapt_reentry_pct") is None and k_sym.get("adapt_reentry_pct") is not None:
                    entry["adapt_reentry_pct"] = k_sym.get("adapt_reentry_pct")
                if vol_matches_kalman and entry.get("adapt_dca_pct") is None and k_sym.get("adapt_dca_pct") is not None:
                    entry["adapt_dca_pct"] = k_sym.get("adapt_dca_pct")

        eff_vol = safe_pos_float(entry.get("vol_1h_pct"), default=None)
        if eff_vol is not None and (entry.get("adapt_reentry_pct") is None or entry.get("adapt_dca_pct") is None):
            calc_re, calc_dca = adaptive_thresholds(eff_vol)
            if entry.get("adapt_reentry_pct") is None and calc_re is not None:
                entry["adapt_reentry_pct"] = calc_re
            if entry.get("adapt_dca_pct") is None and calc_dca is not None:
                entry["adapt_dca_pct"] = calc_dca
        lt_obj: Optional[Dict[str, Any]] = None
        if isinstance(long_items, dict):
            for ck in cand_keys:
                raw_lt = casefold_get(long_items, ck)
                if isinstance(raw_lt, list):
                    valid_lts = [x for x in raw_lt if isinstance(x, dict) and x.get("direction")]
                    if valid_lts:
                        lt_obj = valid_lts[-1]
                        break
                elif isinstance(raw_lt, dict) and raw_lt.get("direction"):
                    lt_obj = raw_lt
                    break
        if lt_obj is None and isinstance(priceanalysis_data, dict):
            pa_map = (
                priceanalysis_data.get("items")
                if isinstance(priceanalysis_data.get("items"), dict)
                else priceanalysis_data
            )
            if isinstance(pa_map, dict):
                lt_obj = next(
                    (
                        val
                        for ck in cand_keys
                        if isinstance((val := casefold_get(pa_map, ck)), dict) and val.get("direction")
                    ),
                    None,
                )
        if isinstance(lt_obj, dict):
            entry["long_trend_dir"] = lt_obj.get("direction")
            dur_s = safe_float(lt_obj.get("duration_seconds"), 0.0)
            entry["long_trend_hours"] = round(dur_s / 3600.0, 1)
            est_rem = safe_float(lt_obj.get("estimated_future_hours"), -1.0)
            if est_rem >= 0.0:
                entry["long_trend_est_rem_hours"] = round(est_rem, 1)
        if isinstance(t_trend_map, dict):
            tt = _first_variant_dict(t_trend_map, cand_keys)
            if isinstance(tt, dict):
                entry["weibull_median_d"] = tt.get("median_d")
                entry["weibull_p90_d"] = tt.get("p90_d")
        if isinstance(watcher_price_items, dict):
            sym_px_cur = safe_float(entry.get("price"), 0.0)
            m_series = None
            for ck in cand_keys:
                cand_ser = casefold_get(watcher_price_items, ck)
                if not isinstance(cand_ser, list) or len(cand_ser) < 2:
                    continue
                last_m_px = (
                    safe_float(cand_ser[-1][1], 0.0)
                    if isinstance(cand_ser[-1], (list, tuple)) and len(cand_ser[-1]) > 1
                    else 0.0
                )
                if ck != sym and (compatible_quote_suffixes(ck) != sym_quotes or _prices_diverged(sym_px_cur, last_m_px)):
                    continue
                m_series = cand_ser
                break
            if isinstance(m_series, list) and len(m_series) >= 2:
                vals = [
                    safe_float(pt[1], 0.0)
                    for pt in m_series
                    if isinstance(pt, list) and len(pt) > 1 and safe_float(pt[1], 0.0) > 0
                ]
                entry.update(compute_7d_range_metrics(vals))


def _compute_kalman_from_series(series_24h: List[Any]) -> Optional[Dict[str, Any]]:
    """Derive Kalman velocity (% per min), std, and trend from a 24h price series when cache_kalman_trend lacks the symbol."""
    pts: List[Tuple[float, float]] = []
    for row in series_24h:
        parsed = _parse_ts_px_row(row)
        if parsed is not None:
            pts.append((parsed[0] / 1000.0, parsed[1]))
    if len(pts) < 5:
        return None
    cutoff_s = pts[-1][0] - 7200.0
    subsampled: List[Tuple[float, float]] = []
    last_ts = -1e18
    for ts_s, px_v in pts:
        if ts_s < cutoff_s:
            continue
        if ts_s - last_ts >= 55.0:
            subsampled.append((ts_s, px_v))
            last_ts = ts_s
    if subsampled and pts[-1][0] > subsampled[-1][0]:
        subsampled.append(pts[-1])
    if len(subsampled) < 5:
        subsampled = pts[-min(len(pts), 30) :]
    if len(subsampled) < 5:
        return None
    kf = KalmanTrend(gap_reset_sec=7200.0)
    fed: List[float] = []
    last_out: Optional[Dict[str, Any]] = None
    for ts_s, px_v in subsampled[-60:]:
        fed.append(px_v)
        if len(fed) >= 3:
            diffs = [fed[i] - fed[i - 1] for i in range(1, len(fed))]
            mean_d = sum(diffs) / len(diffs)
            var_d = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
            eps_eff = max(math.sqrt(max(0.0, var_d)), px_v * 1e-4)
        else:
            eps_eff = max(px_v * 8e-4, 1e-9)
        last_out = kf.update(ts_s, px_v, eps_eff)
    if not isinstance(last_out, dict):
        return None
    return {
        "kalman_vel": last_out.get("vel"),
        "kalman_vel_std": last_out.get("vel_std"),
        "kalman_trend": last_out.get("trend"),
    }


def enrich_24h_and_microstructure(
    cache_dir: str,
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    cur_p_items: Dict[str, Any],
    inst_by_sym: Dict[str, Any],
    now_ts: float,
    read_json_fn: Any = None,
) -> None:
    """Enrich prices_and_trends with 24h/7d metrics, fallback OHLC/Yahoo chart series, and microstructure."""
    _rj = read_json_fn or read_json_cached
    btc_series_24h, _ = load_symbol_price_series(
        cache_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False
    )
    t212_dir = os.path.join(workspace_dir, "212trading")
    if os.path.isdir(t212_dir):
        for st_path in sorted(_glob.glob(os.path.join(t212_dir, ".state_*.json"))):
            fname = os.path.basename(st_path)
            if fname.startswith(".."):
                continue
            ticker = fname[len(".state_") : -len(".json")].upper()
            if ticker.replace("_", "").isalnum() and ticker not in prices_and_trends:
                prices_and_trends[ticker] = {"price": None, "trend": 0, "growth_coefficient": 0.0}
    for sym in list(prices_and_trends.keys()):
        sym_px_cur = safe_float(prices_and_trends[sym].get("price"), 0.0)
        matched_series_24h, arch_vals_7d = load_symbol_price_series(
            cache_dir, sym, current_price_items=cur_p_items
        )
        if sym_px_cur > 0 and matched_series_24h:
            last_24_px = safe_float(matched_series_24h[-1][1], 0.0)
            if _prices_diverged(sym_px_cur, last_24_px):
                matched_series_24h = []
            elif last_24_px > 0 and abs(sym_px_cur - last_24_px) > 1e-9:
                matched_series_24h = list(matched_series_24h) + [[int(now_ts * 1000), sym_px_cur]]
        if sym_px_cur > 0 and arch_vals_7d:
            last_7d_px = safe_float(arch_vals_7d[-1], 0.0)
            if _prices_diverged(sym_px_cur, last_7d_px):
                arch_vals_7d = []
        if safe_float(prices_and_trends[sym].get("price"), 0.0) <= 0.0:
            if matched_series_24h:
                prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
            elif arch_vals_7d:
                prices_and_trends[sym]["price"] = arch_vals_7d[-1]
        m24: Dict[str, Any] = compute_24h_price_metrics(matched_series_24h) or {} if matched_series_24h else {}
        if arch_vals_7d and len(arch_vals_7d) >= 2 and "chg_7d_pct" not in prices_and_trends[sym]:
            prices_and_trends[sym].update(compute_7d_range_metrics(arch_vals_7d))
        if not m24 and sym in inst_by_sym:
            prov = getattr(inst_by_sym[sym], "_provider", None)
            ohlc_fn = getattr(prov, "ohlc_series", None)
            if callable(ohlc_fn):
                try:
                    s5 = ohlc_fn(sym, 5)
                    cutoff_24h = now_ts - 86400.0
                    s5_vols = getattr(s5, "volumes", ()) or ()
                    pts_24h = [
                        (
                            [int(ts * 1000), float(px), float(s5_vols[idx])]
                            if idx < len(s5_vols) and (safe_pos_float(s5_vols[idx], 0.0) or 0.0) > 0.0
                            else [int(ts * 1000), float(px)]
                        )
                        for idx, (ts, px) in enumerate(zip(getattr(s5, "timestamps", ()), getattr(s5, "closes", ())))
                        if ts >= cutoff_24h and float(px) > 0
                    ]
                    m24 = compute_24h_price_metrics(pts_24h) or {}
                    if m24:
                        matched_series_24h = pts_24h
                    if "chg_7d_pct" not in prices_and_trends[sym]:
                        s60 = ohlc_fn(sym, 60)
                        cutoff_7d = now_ts - 7 * 86400.0
                        vals_7d = [
                            float(px)
                            for ts, px in zip(getattr(s60, "timestamps", ()), getattr(s60, "closes", ()))
                            if ts >= cutoff_7d and float(px) > 0
                        ]
                        prices_and_trends[sym].update(compute_7d_range_metrics(vals_7d))
                except Exception:
                    pass
        is_t212_inst = sym in inst_by_sym and normalize_venue_name(getattr(inst_by_sym[sym], "provider_name", "")) == "t212"
        if not m24 and (is_t212_inst or (sym.endswith(("_EQ", "_US_EQ")) and "PYTEST_CURRENT_TEST" not in os.environ)):
            try:
                _chart = load_local_module(os.path.join(workspace_dir, "212trading"), "market_data")._chart
                yahoo_sym = sym.split("_", 1)[0]
                chart_res = _chart(yahoo_sym, rng="5d", interval="5m")
                pts_5d: List[List[Any]] = []
                if isinstance(chart_res, tuple) and len(chart_res) >= 2 and isinstance(chart_res[1], list):
                    pts_5d = [
                        (
                            [int(ts_f * 1000), round(px_f, 4), safe_nonneg_float(bar[2], 0.0) or 0.0]
                            if len(bar) >= 3
                            else [int(ts_f * 1000), round(px_f, 4)]
                        )
                        for bar in chart_res[1]
                        if isinstance(bar, (list, tuple))
                        and len(bar) >= 2
                        and (ts_f := safe_pos_float(bar[0], default=None)) is not None
                        and (px_f := safe_pos_float(bar[1], default=None)) is not None
                    ]
                elif isinstance(chart_res, dict):
                    ts_list = chart_res.get("timestamp") or []
                    q_obj = ((chart_res.get("indicators") or {}).get("quote") or [{}])[0]
                    closes = q_obj.get("close") or []
                    vols = q_obj.get("volume") or []
                    pts_5d = [
                        [
                            int(ts_f * 1000),
                            round(c_f, 4),
                            safe_nonneg_float(vols[idx], 0.0) or 0.0 if idx < len(vols) else 0.0,
                        ]
                        for idx, (ts, c) in enumerate(zip(ts_list, closes))
                        if (ts_f := safe_pos_float(ts, default=None)) is not None
                        and (c_f := safe_pos_float(c, default=None)) is not None
                    ]
                if pts_5d:
                    cutoff_24h_ms = int((now_ts - 86400.0) * 1000)
                    pts_24h = [pt for pt in pts_5d if pt[0] >= cutoff_24h_ms]
                    if len(pts_24h) < 10:
                        pts_24h = list(pts_5d[-78:])
                    if (
                        pts_24h
                        and sym_px_cur > 0
                        and not _prices_diverged(sym_px_cur, pts_24h[-1][1])
                        and abs(sym_px_cur - pts_24h[-1][1]) > 1e-9
                    ):
                        pts_24h = list(pts_24h) + [[int(now_ts * 1000), sym_px_cur]]
                    m24 = compute_24h_price_metrics(pts_24h) or {}
                    if m24:
                        matched_series_24h = pts_24h
                    vals_5d = [pt[1] for pt in pts_5d]
                    if len(vals_5d) >= 2 and "chg_7d_pct" not in prices_and_trends[sym]:
                        prices_and_trends[sym].update(
                            compute_7d_range_metrics(vals_5d, decimals=4)
                        )
            except Exception:
                pass
        if m24:
            prices_and_trends[sym].update(m24)
            if matched_series_24h and safe_float(prices_and_trends[sym].get("price"), 0.0) <= 0.0:
                prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
            if prices_and_trends[sym].get("trend", 0) == 0 and "chg_1h_pct" in m24:
                c1h = safe_float(m24.get("chg_1h_pct"), 0.0)
                if c1h >= 0.25:
                    prices_and_trends[sym]["trend"] = 1
                elif c1h <= -0.25:
                    prices_and_trends[sym]["trend"] = -1
            if (
                matched_series_24h
                and btc_series_24h
                and not sym.endswith(("_EQ", "_US_EQ"))
                and canonical_base_asset(sym) != "BTC"
            ):
                cross_btc = compute_cross_btc_metrics(matched_series_24h, btc_series_24h)
                for ck in ("alt_btc_pos_24h_pct", "alt_btc_dd_24h_high_pct", "alt_btc_bounce_24h_low_pct"):
                    if cross_btc.get(ck) is not None:
                        prices_and_trends[sym][ck] = cross_btc[ck]
            if prices_and_trends[sym].get("kalman_vel") is None and matched_series_24h:
                k_fb = _compute_kalman_from_series(matched_series_24h)
                if isinstance(k_fb, dict):
                    for kk, kv in k_fb.items():
                        if kv is not None and prices_and_trends[sym].get(kk) is None:
                            prices_and_trends[sym][kk] = kv

        m_info = load_symbol_microstructure(cache_dir, sym, _rj)
        if m_info:
            is_usd_sym = "USD" in compatible_quote_suffixes(sym)
            sym_px_m = safe_float((prices_and_trends.get(sym) or {}).get("price"), 0.0)
            if (
                is_usd_sym
                and isinstance(prices_and_trends.get(sym), dict)
                and sym_px_m <= 0.0
                and safe_pos_float(m_info.get("ob_mid_price"), default=None)
            ):
                prices_and_trends[sym]["price"] = m_info["ob_mid_price"]
            elif not is_usd_sym and (
                sym_px_m <= 0.0 or _prices_diverged(sym_px_m, m_info.get("ob_mid_price"))
            ):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    m_info.pop(k_strip, None)
            enrich_microstructure_cvd(m_info, prices_and_trends.get(sym))
            liq_info = compute_liquidation_clusters(prices_and_trends.get(sym), m_info)
            if liq_info:
                m_info.update(liq_info)
            if m_info.get("spread_bps") is not None and isinstance(prices_and_trends.get(sym), dict):
                prices_and_trends[sym]["spread_bps"] = m_info["spread_bps"]
            microstructure[sym] = m_info

    if os.path.isdir(cache_dir):
        covered_micro_bases = {canonical_base_asset(k) for k in microstructure}
        ref_micro_syms: List[str] = []
        for pfx in ("orderbook_depth_", "derivatives_telemetry_", "whale_snapshot_"):
            for search_dir in (cache_dir, os.path.join(cache_dir, "telemetry")):
                for mpath in sorted(_glob.glob(os.path.join(search_dir, f"{pfx}*.json"))):
                    mfname = os.path.basename(mpath)
                    msym = mfname[len(pfx) : -len(".json")].upper()
                    if not msym or not msym.isalnum() or msym in microstructure or msym in ref_micro_syms:
                        continue
                    mbase = canonical_base_asset(msym)
                    if mbase not in covered_micro_bases:
                        ref_micro_syms.append(msym)
                        covered_micro_bases.add(mbase)
        for msym in ref_micro_syms:
            m_info_ref = load_symbol_microstructure(cache_dir, msym, _rj)
            if m_info_ref:
                msym_quotes = compatible_quote_suffixes(msym)
                p_ref = _first_variant_dict(
                    prices_and_trends,
                    cache_symbol_variants(
                        msym,
                        include_original=True,
                        include_base="USD" in msym_quotes,
                        quotes=msym_quotes,
                    ),
                )
                enrich_microstructure_cvd(m_info_ref, p_ref)
                liq_ref = compute_liquidation_clusters(p_ref, m_info_ref)
                if liq_ref:
                    m_info_ref.update(liq_ref)
                microstructure[msym] = m_info_ref


def sync_sibling_symbol_telemetry(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> None:
    """Symmetric cross-venue & sibling symbol synchronization across Binance, Kraken, Hyperliquid, and T212/xStocks."""
    range_bound_keys = {
        "low_24h", "high_24h", "pos_24h_pct", "dd_24h_high_pct", "bounce_24h_low_pct",
        "low_7d", "high_7d", "pos_7d_pct", "dd_7d_high_pct", "bounce_7d_low_pct", "span_7d_pct",
        "swing_low_px", "swing_high_px", "major_swing_low_px",
        "rally_from_swing_low_pct", "pullback_from_swing_high_pct", "rally_from_major_low_pct",
        "poc_px", "vah_px", "val_px", "poc_dist_pct", "vah_dist_pct", "val_dist_pct", "va_pos_pct", "va_regime",
        "bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px", "higher_low_vs_prior_pct", "reclaim_above_prior_low_pct",
    }
    for sym, pinfo in list(prices_and_trends.items()):
        if not isinstance(pinfo, dict):
            continue
        sym_quotes = compatible_quote_suffixes(sym)
        sym_variants = cache_symbol_variants(sym, include_original=True, include_base=True)
        for cand in sym_variants:
            if cand == sym:
                continue
            cand_same_quote_family = compatible_quote_suffixes(cand) == sym_quotes
            sib_pinfo = casefold_get(prices_and_trends, cand)
            if isinstance(sib_pinfo, dict):
                p_px = safe_float(pinfo.get("price"), 0.0)
                s_px = safe_float(sib_pinfo.get("price"), 0.0)
                if cand_same_quote_family and p_px <= 0.0 and s_px > 0.0:
                    pinfo["price"] = sib_pinfo["price"]
                    p_px = s_px
                px_diverged = (not cand_same_quote_family) or _prices_diverged(p_px, s_px)
                for k, v in sib_pinfo.items():
                    if v is None or k == "price":
                        continue
                    if px_diverged and k in range_bound_keys:
                        continue
                    if k not in pinfo or pinfo.get(k) is None:
                        pinfo[k] = v
                    elif k in ("trend", "growth_coefficient") and pinfo.get(k) in (0, 0.0) and v not in (0, 0.0):
                        pinfo[k] = v
            sib_micro = casefold_get(microstructure, cand)
            if isinstance(sib_micro, dict) and sib_micro:
                p_px_m = safe_float(pinfo.get("price"), 0.0)
                s_ob_mid = safe_float(sib_micro.get("ob_mid_price"), 0.0) or safe_float(
                    sib_pinfo.get("price") if isinstance(sib_pinfo, dict) else None, 0.0
                )
                micro_px_diverged = (not cand_same_quote_family) or _prices_diverged(p_px_m, s_ob_mid)
                if sym not in microstructure:
                    microstructure[sym] = {
                        mk: mv
                        for mk, mv in sib_micro.items()
                        if not (micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS)
                    }
                else:
                    for mk, mv in sib_micro.items():
                        if micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS:
                            continue
                        if microstructure[sym].get(mk) is None and mv is not None:
                            microstructure[sym][mk] = mv
        p_cur = safe_pos_float(pinfo.get("price"), default=0.0) or 0.0
        poc_v = safe_pos_float(pinfo.get("poc_px"), default=0.0) or 0.0
        val_v = safe_pos_float(pinfo.get("val_px"), default=0.0) or 0.0
        vah_v = safe_pos_float(pinfo.get("vah_px"), default=0.0) or 0.0
        if p_cur > 0.0 and poc_v > 0.0 and val_v > 0.0 and vah_v > val_v:
            pinfo.update(_build_vp_position_dict(p_cur, poc_v, val_v, vah_v))

    for msym, minfo in list(microstructure.items()):
        if not isinstance(minfo, dict):
            continue
        msym_quotes = compatible_quote_suffixes(msym)
        p_match = _first_variant_dict(
            prices_and_trends,
            cache_symbol_variants(
                msym,
                include_original=True,
                include_base="USD" in msym_quotes,
                quotes=msym_quotes,
            ),
        )
        liq_clusters = compute_liquidation_clusters(p_match, minfo)
        if liq_clusters:
            minfo.update(liq_clusters)
        enrich_microstructure_cvd(minfo, p_match)
        if isinstance(p_match, dict) and minfo.get("spread_bps") is not None:
            p_match.setdefault("spread_bps", minfo["spread_bps"])


def _load_recent_audit_origins(cache_dir: str) -> Dict[str, tuple[Optional[str], Optional[str]]]:
    """Load recent orderId/clientOrderId -> (module, rule) mappings from execution_audit jsonl logs."""
    out: Dict[str, tuple[Optional[str], Optional[str]]] = {}
    cand_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(cache_dir)), "logger", "execution_audit"),
        cache_dir,
    ]
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        try:
            files = sorted(
                (os.path.join(d, f) for f in os.listdir(d) if f.startswith("execution_audit") and f.endswith(".jsonl")),
                reverse=True,
            )
        except Exception:
            continue
        for fpath in files[:2]:
            for rec in read_tail_jsonl_dicts(fpath, limit=250):
                _, mod, rule = resolve_order_origin(rec)
                if not (mod or rule):
                    continue
                for k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                    kv = str(rec.get(k) or "").strip()
                    if kv:
                        out.setdefault(kv, (mod, rule))
        if out:
            break
    return out


def harvest_recent_orders_and_enrich_positions(
    cache_dir: str,
    inst_by_sym: Dict[str, Any],
    tradable_positions: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    kraken_cycles: Dict[str, Dict[str, Any]],
    hyperliquid_cycles: Dict[str, Dict[str, Any]],
    t212_cycles: Dict[str, Dict[str, Any]],
    read_json_fn: Any = None,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    gateway_max_usd: float = 2000.0,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Load recent fills across Binance, Kraken, Hyperliquid, and T212 and enrich tradable_positions with cost-basis anchors."""
    _rj = read_json_fn or read_json_cached
    audit_origin_map = _load_recent_audit_origins(cache_dir)
    recent_orders: Dict[str, Any] = {}
    venue_recent_orders: Dict[str, Dict[str, Any]] = {
        "binance": {},
        "kraken": {},
        "hyperliquid": {},
        "t212": {},
    }
    def _record_venue_fills(c_venue: str, sym_k: str, fills_d: Dict[str, Any]) -> None:
        fills_d.setdefault("venue", c_venue)
        venue_recent_orders[c_venue][sym_k] = fills_d
        if sym_k in recent_orders and c_venue != "binance":
            recent_orders[f"{c_venue}:{sym_k}"] = fills_d
        else:
            recent_orders.setdefault(sym_k, fills_d)

    for order_cache_name, cache_venue in (
        ("cache_order.json", "binance"),
        ("cache_trade_kraken.json", "kraken"),
        ("cache_trade_hyperliquid.json", "hyperliquid"),
        ("cache_trade_t212.json", "t212"),
    ):
        orders_data = _rj(os.path.join(cache_dir, order_cache_name))
        if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
            v_ai_intents = (
                [it for it in ai_intents if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == cache_venue]
                if isinstance(ai_intents, list)
                else None
            )
            for sym, o_list in orders_data["items"].items():
                fills = extract_symbol_recent_fills(
                    o_list,
                    known_ai_order_ids=known_ai_order_ids,
                    audit_origin_map=audit_origin_map,
                    ai_intents=v_ai_intents,
                )
                if fills:
                    _record_venue_fills(cache_venue, sym, fills)
    seen_hl_syms: set[str] = set()
    hl_ai_intents = (
        [it for it in ai_intents if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == "hyperliquid"]
        if isinstance(ai_intents, list)
        else None
    )
    for inst in inst_by_sym.values():
        sym = str(getattr(inst, "symbol", "")).upper()
        v_inst = normalize_venue_name(getattr(inst, "provider_name", ""))
        if (
            not sym
            or v_inst != "hyperliquid"
            or sym in seen_hl_syms
            or _first_variant_dict(venue_recent_orders["hyperliquid"], cache_symbol_variants(sym, include_base=True)) is not None
        ):
            continue
        seen_hl_syms.add(sym)
        try:
            prov = getattr(inst, "_provider", None)
            if prov is not None and hasattr(prov, "get_orders"):
                hl_orders = prov.get_orders(sym, None, 90 * 86400.0)
                fills = extract_symbol_recent_fills(
                    hl_orders,
                    known_ai_order_ids=known_ai_order_ids,
                    audit_origin_map=audit_origin_map,
                    ai_intents=hl_ai_intents,
                )
                if fills:
                    _record_venue_fills("hyperliquid", sym, fills)
        except Exception:
            pass
    if not venue_recent_orders["t212"] and cache_dir:
        audit_dir = os.path.join(os.path.dirname(os.path.abspath(cache_dir)), "logger", "execution_audit")
        if os.path.isdir(audit_dir):
            try:
                t212_by_sym: Dict[str, List[Dict[str, Any]]] = {}
                audit_files = sorted(
                    (os.path.join(audit_dir, f) for f in os.listdir(audit_dir) if f.startswith("execution_audit") and f.endswith(".jsonl"))
                )[-10:]
                for af in audit_files:
                    for rec in read_tail_jsonl_dicts(af, limit=500):
                        if str(rec.get("event") or "") not in ("order_terminal", "order_terminal_retry"):
                            continue
                        if normalize_venue_name(rec.get("venue")) != "t212":
                            continue
                        sym_t = str(rec.get("symbol") or "").strip().upper()
                        f_q = safe_pos_float(
                            rec.get("filled_qty") if rec.get("filled_qty") is not None else rec.get("qty"),
                            default=0.0,
                        ) or 0.0
                        cost_f = safe_pos_float(rec.get("cost"), default=0.0) or 0.0
                        px_f = safe_pos_float(rec.get("price"), default=None) or (
                            round(cost_f / f_q, 6) if cost_f > 0 and f_q > 0 else None
                        )
                        ts_s = safe_pos_float(rec.get("ts"), default=0.0) or 0.0
                        side_t = extract_order_side(rec) or str(rec.get("side") or "").strip().upper()
                        if sym_t and f_q > 0 and px_f is not None and px_f > 0 and ts_s > 0 and side_t in ("BUY", "SELL"):
                            t212_by_sym.setdefault(sym_t, []).append({
                                "symbol": sym_t,
                                "orderId": rec.get("order_id"),
                                "clientOrderId": rec.get("client_order_id"),
                                "side": side_t,
                                "price": px_f,
                                "qty": f_q,
                                "executedQty": f_q,
                                "timestamp": int(round(ts_s * 1000.0)),
                                "status": "FILLED",
                            })
                t212_ai_intents = (
                    [it for it in ai_intents if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == "t212"]
                    if isinstance(ai_intents, list)
                    else None
                )
                for sym_t, o_list in t212_by_sym.items():
                    fills = extract_symbol_recent_fills(
                        o_list,
                        known_ai_order_ids=known_ai_order_ids,
                        audit_origin_map=audit_origin_map,
                        ai_intents=t212_ai_intents,
                    )
                    if fills:
                        _record_venue_fills("t212", sym_t, fills)
            except Exception:
                pass
    cyc_sell_ts_map: Dict[tuple[str, str], int] = {}
    for c_venue, cyc_map, avg_dec in (
        ("kraken", kraken_cycles, 6),
        ("hyperliquid", hyperliquid_cycles, 6),
        ("t212", t212_cycles, 4),
    ):
        if not isinstance(cyc_map, dict):
            continue
        v_ro_c = venue_recent_orders.get(c_venue, {})
        for c_sym, cyc in cyc_map.items():
            if not isinstance(cyc, dict):
                continue
            entry_ts_ms = cyc.pop("_entry_ts_ms", None)
            sell_ts_ms = cyc.pop("_last_sell_ts_ms", None)
            if sell_ts_ms:
                cyc_sell_ts_map[(c_venue, str(c_sym).upper())] = int(sell_ts_ms)
            if not entry_ts_ms or cyc.get("resting_orders"):
                continue
            c_quotes = compatible_quote_suffixes(c_sym)
            c_variants = cache_symbol_variants(
                c_sym,
                include_base="USD" in c_quotes,
                quotes=c_quotes,
            )
            ro_c = _first_variant_dict(v_ro_c, (c_sym, *c_variants))
            if not isinstance(ro_c, dict):
                continue
            ls_c = ro_c.get("last_sell") if isinstance(ro_c.get("last_sell"), dict) else {}
            ls_ts_ms = safe_pos_float(ls_c.get("ts_ms"), default=0.0) or 0.0
            ls_qty = safe_pos_float(ls_c.get("qty"), default=0.0) or 0.0
            ls_px = safe_pos_float(ls_c.get("price"), default=0.0) or 0.0
            cyc_qty_cur = safe_nonneg_float(cyc.get("qty"), default=0.0) or 0.0
            r_buys_c = ro_c.get("recent_buys") if isinstance(ro_c.get("recent_buys"), list) else []
            buys_after = [
                b
                for b in r_buys_c
                if isinstance(b, dict)
                and (safe_pos_float(b.get("ts_ms"), default=0.0) or 0.0) > ls_ts_ms
                and (safe_pos_float(b.get("qty"), default=0.0) or 0.0) > 0.0
                and (safe_pos_float(b.get("price"), default=0.0) or 0.0) > 0.0
            ]
            reconciled_external_buy = bool(
                buys_after
                and cyc_qty_cur > 1e-8
                and float(entry_ts_ms) > (safe_pos_float(buys_after[-1].get("ts_ms"), default=0.0) or 0.0) + 5000.0
                and abs(sum(float(b["qty"]) for b in buys_after) - cyc_qty_cur) <= max(1e-4, cyc_qty_cur * 0.02)
            )
            if (ls_ts_ms > float(entry_ts_ms) and cyc_qty_cur > 1e-8 and ls_qty >= cyc_qty_cur * 0.95) or reconciled_external_buy:
                if buys_after:
                    new_qty = sum(float(b["qty"]) for b in buys_after)
                    new_cost = sum(float(b["qty"]) * float(b["price"]) for b in buys_after)
                    new_avg = round(new_cost / new_qty, avg_dec)
                    px_c = resolve_symbol_price(c_sym, prices_and_trends)
                    new_dd = round(((px_c - new_avg) / new_avg) * 100.0, 2) if px_c > 0 and new_avg > 0 else None
                    cyc["qty"] = round(new_qty, 6)
                    cyc["cost_usd"] = round(new_cost, 2)
                    cyc["avg_cost_px"] = new_avg
                    cyc["entry_price"] = round(float(buys_after[0]["price"]), avg_dec)
                    cyc["last_buy_price"] = round(float(buys_after[-1]["price"]), avg_dec)
                    if ls_px > 0:
                        cyc["last_sell_price"] = round(ls_px, avg_dec)
                    cyc["dca_buys"] = max(0, len(buys_after) - 1)
                    cyc["dd_pct"] = new_dd
                    cyc["pnl_vs_avg_pct"] = new_dd
                else:
                    cyc["qty"] = 0.0
                    cyc["cost_usd"] = 0.0
                    cyc["avg_cost_px"] = None
                    cyc["dd_pct"] = None
                    cyc["pnl_vs_avg_pct"] = None
                    if ls_px > 0:
                        cyc["last_sell_price"] = round(ls_px, avg_dec)
                cyc.pop("pos_peak", None)
                cyc.pop("dd_from_peak_pct", None)
    for pos in tradable_positions:
        if not isinstance(pos, dict):
            continue
        p_sym = str(pos.get("symbol") or "").upper()
        p_venue = normalize_venue_name(pos.get("venue"))
        p_quotes = compatible_quote_suffixes(p_sym, pos.get("quote"))
        p_quote_variants = cache_symbol_variants(
            p_sym,
            include_base="USD" in p_quotes,
            quotes=p_quotes,
        )
        p_variants = (*p_quote_variants, *cache_symbol_variants(p_sym))
        pos_px_f = safe_float(pos.get("price"), 0.0)
        if pos_px_f <= 0:
            cand_px = resolve_symbol_price(p_sym, prices_and_trends, explicit_quote=pos.get("quote"))
            if cand_px > 0:
                pos_px_f = cand_px
                pos["price"] = round(cand_px, 6)
                fb_val = safe_float(pos.get("free_base"), -1.0)
                if fb_val >= 0:
                    pos["value_usd"] = round(fb_val * cand_px, 2)
                pk_val = safe_float(pos.get("trailing_peak"), 0.0)
                if pk_val > 0:
                    if pk_val < cand_px:
                        pk_val = cand_px
                        pos["trailing_peak"] = round(pk_val, 6)
                    pos["dd_from_peak_pct"] = round(((cand_px - pk_val) / pk_val) * 100.0, 2)
        v_ro_map = venue_recent_orders.get(p_venue, {})
        ro = _first_variant_dict(v_ro_map, p_quote_variants)
        fb_cur = safe_float(pos.get("free_base"), -1.0)
        min_notional_f = max(
            1.0,
            safe_float(pos.get("exchange_min_notional_usd") or pos.get("min_notional_usd"), 1.0),
        )
        is_flat_balance = fb_cur >= 0.0 and (
            fb_cur <= 1e-8 or (pos_px_f > 0.0 and fb_cur * pos_px_f < min_notional_f)
        )
        if is_flat_balance:
            pos.pop("trailing_peak", None)
            pos.pop("dd_from_peak_pct", None)
        last_side_ro = str(ro.get("last_action_side") or "").upper() if isinstance(ro, dict) else ""
        is_flat_after_exit = is_flat_balance and last_side_ro == "SELL"
        if isinstance(ro, dict):
            lb = ro.get("last_buy")
            ls = ro.get("last_sell")
            if ro.get("last_action_side"):
                pos["last_action_side"] = str(ro.get("last_action_side")).upper()
            if is_flat_after_exit:
                ro.pop("active_avg_buy_px", None)
                pos.pop("trailing_peak", None)
                pos.pop("dd_from_peak_pct", None)
            else:
                set_pos_anchor(pos, "active_avg_buy_px", "pnl_vs_avg_pct", ro.get("active_avg_buy_px"), pos_px_f)
            set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", lb.get("price") if isinstance(lb, dict) else None, pos_px_f)
            set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", ls.get("price") if isinstance(ls, dict) else None, pos_px_f)
            r_buys = ro.get("recent_buys") if isinstance(ro.get("recent_buys"), list) else []
            low_buy_px = min(
                (
                    safe_float(b.get("price"), 0.0)
                    for b in r_buys
                    if isinstance(b, dict) and safe_float(b.get("price"), 0.0) > 0
                ),
                default=0.0,
            )
            if low_buy_px > 0 and not is_flat_after_exit:
                set_pos_anchor(pos, "lowest_buy_px", "pnl_vs_lowest_buy_pct", low_buy_px, pos_px_f)
        venue_cyc_maps = (
            (hyperliquid_cycles,)
            if p_venue == "hyperliquid"
            else (t212_cycles,)
            if p_venue == "t212"
            else (kraken_cycles,)
            if p_venue == "kraken"
            else ()
        )
        for cyc_map in venue_cyc_maps:
            cyc = _first_variant_dict(cyc_map, (p_sym, *p_quote_variants))
            if not isinstance(cyc, dict):
                continue
            cyc_qty = safe_nonneg_float(cyc.get("qty"), default=None)
            if pos.get("free_base") is None and cyc_qty is not None:
                pos["free_base"] = round(cyc_qty, 6)
                fb_cur = cyc_qty
                is_flat_balance = fb_cur <= 1e-8 or (pos_px_f > 0.0 and fb_cur * pos_px_f < 1.0)
                if pos_px_f > 0:
                    pos["value_usd"] = round(cyc_qty * pos_px_f, 2)
            cyc_has_open_qty = (cyc_qty > 1e-8) if cyc_qty is not None else (not is_flat_balance)
            cyc_flat = is_flat_balance and not cyc_has_open_qty
            if (is_flat_after_exit or cyc_flat) and not cyc.get("resting_orders"):
                cyc["qty"] = round(max(0.0, fb_cur), 6)
                cyc["cost_usd"] = 0.0
                cyc["avg_cost_px"] = None
                cyc["dd_pct"] = None
                if isinstance(ro, dict):
                    ro.pop("active_avg_buy_px", None)
                for _stale_k in (
                    "active_avg_buy_px",
                    "pnl_vs_avg_pct",
                    "lowest_buy_px",
                    "pnl_vs_lowest_buy_pct",
                    "trailing_peak",
                    "dd_from_peak_pct",
                ):
                    pos.pop(_stale_k, None)
                if is_flat_after_exit and pos.get("last_sell_px") is not None:
                    cyc["last_sell_price"] = pos["last_sell_px"]
            elif cyc_flat:
                pos.pop("trailing_peak", None)
                pos.pop("dd_from_peak_pct", None)
            elif not is_flat_after_exit and cyc_has_open_qty:
                cyc_avg = cyc.get("avg_cost_px") or cyc.get("entry_price") or cyc.get("last_buy_price")
                set_pos_anchor(
                    pos,
                    "active_avg_buy_px",
                    "pnl_vs_avg_pct",
                    cyc_avg,
                    pos_px_f,
                    overwrite=(p_venue == "t212" and cyc.get("avg_cost_px") is not None),
                )
                if p_venue == "t212":
                    _t212_avg = safe_pos_float(cyc.get("avg_cost_px"), default=0.0) or 0.0
                    _t212_lb = safe_pos_float(cyc.get("last_buy_price"), default=None)
                    _t212_ls = safe_pos_float(cyc.get("last_sell_price"), default=None)
                    _t212_ls_ts = cyc_sell_ts_map.get(("t212", p_sym))
                    if not isinstance(ro, dict) and (_t212_lb is not None or _t212_ls is not None):
                        ro = {
                            "last_action_side": "BUY" if cyc_has_open_qty else ("SELL" if _t212_ls is not None else "BUY"),
                            "last_action_actor": "BOT",
                            "last_action_actor_tag": "BOT",
                            "last_buy": (
                                {
                                    "price": _t212_lb,
                                    "qty": cyc.get("qty") if not cyc.get("dca_buys") and cyc_has_open_qty else None,
                                    "actor": "BOT",
                                    "actor_tag": "BOT",
                                }
                                if _t212_lb is not None
                                else None
                            ),
                            "last_sell": (
                                {
                                    "price": _t212_ls,
                                    "ts_ms": _t212_ls_ts,
                                    "actor": "BOT",
                                    "actor_tag": "BOT",
                                }
                                if _t212_ls is not None
                                else None
                            ),
                            "active_avg_buy_px": round(_t212_avg, 6) if _t212_avg > 0 else None,
                            "recent_buys": [],
                        }
                        recent_orders[p_sym] = ro
                        v_ro_map[p_sym] = ro
                    elif isinstance(ro, dict):
                        if _t212_avg > 0:
                            ro["active_avg_buy_px"] = round(_t212_avg, 6)
                        if ro.get("last_sell") is None and _t212_ls is not None:
                            ro["last_sell"] = {
                                "price": _t212_ls,
                                "ts_ms": _t212_ls_ts,
                                "actor": "BOT",
                                "actor_tag": "BOT",
                            }
                        if ro.get("last_buy") is None and _t212_lb is not None:
                            ro["last_buy"] = {
                                "price": _t212_lb,
                                "actor": "BOT",
                                "actor_tag": "BOT",
                            }
                set_pos_anchor(pos, "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct", cyc.get("avg_cost_px"), pos_px_f, overwrite=False)
                set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", cyc.get("pos_peak"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", cyc.get("last_buy_price"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", cyc.get("last_sell_price"), pos_px_f, overwrite=False)
            if cyc.get("last_buy_price") is None and pos.get("last_buy_px") is not None:
                cyc["last_buy_price"] = pos["last_buy_px"]
            if cyc.get("last_sell_price") is None and pos.get("last_sell_px") is not None:
                cyc["last_sell_price"] = pos["last_sell_px"]
        sym_spread = next(
            (
                val.get("spread_bps")
                for c in p_variants
                if isinstance((val := casefold_get(prices_and_trends, c)), dict)
                and val.get("spread_bps") is not None
            ),
            pos.get("spread_bps"),
        )
        inst_obj = inst_by_sym.get(f"{p_venue}:{p_sym}") or inst_by_sym.get(p_sym)
        pos.update(
            resolve_venue_fee_and_limits(
                p_venue,
                p_sym,
                inst_obj,
                price=pos_px_f,
                free_quote=safe_nonneg_float(pos.get("free_quote"), default=None),
                free_base=safe_nonneg_float(pos.get("free_base"), default=None),
                spread_bps=safe_nonneg_float(sym_spread, default=None),
                gateway_max_usd=gateway_max_usd,
            )
        )
    return recent_orders


def harvest_bot_runtime_states(
    cache_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
    tradable_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Harvest active retry queue, monitortrades, assetguardian, and smart_stop_loss states."""
    _rj = read_json_fn or read_json_cached
    out: Dict[str, Any] = {}
    retry_q_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "order_retry_queue.jsonl")))
    if os.path.exists(retry_q_path):
        try:
            import order_retry as _oq
        except Exception:
            _oq = None
        active_retries: List[Dict[str, Any]] = []
        for rec in read_tail_jsonl_dicts(retry_q_path, limit=50):
            lifecycle = str(rec.get("lifecycle") or "").lower()
            exp_ts = safe_float(rec.get("expires_at_ts"), 0.0)
            if _oq is not None and (rec.get("created_ts") is not None or rec.get("ttl_started_ts") is not None):
                try:
                    if _oq.is_expired(rec, now_ts):
                        continue
                except Exception:
                    pass
            if lifecycle in ("submit_pending", "pending", "active") and (exp_ts == 0.0 or exp_ts > now_ts):
                active_retries.append({
                    "venue": str(rec.get("provider_name") or "").lower(),
                    "symbol": rec.get("symbol"),
                    "side": rec.get("side"),
                    "qty": rec.get("qty"),
                    "requested_price": rec.get("requested_price"),
                    "kind": rec.get("kind"),
                    "attempts": rec.get("attempts"),
                    "last_failure_reason": rec.get("last_failure_reason"),
                })
        if active_retries:
            out["active_retry_queue"] = active_retries

    mt_state = _rj(os.path.join(cache_dir, "mt_state.json"))
    if isinstance(mt_state, dict):
        active_mt = {
            k: v
            for k in ("tp_ride_state", "sl_bounce_state", "sl_episode_state", "reentry_state", "pending_stop_exit")
            for v in (mt_state.get(k),)
            if isinstance(v, dict) and v
        }
        if active_mt:
            out["monitortrades_active_state"] = active_mt

    ag_state = _rj(os.path.join(cache_dir, "assetguardian_state.json"))
    if isinstance(ag_state, dict) and isinstance(ag_state.get("symbols"), dict) and ag_state["symbols"]:
        dust_binance_syms: set[str] = set()
        if isinstance(tradable_positions, list):
            for pos in tradable_positions:
                if not isinstance(pos, dict) or normalize_venue_name(pos.get("venue")) != "binance":
                    continue
                p_sym = str(pos.get("symbol") or "").upper()
                val_usd = safe_float(pos.get("value_usd"), -1.0)
                min_not = max(1.0, safe_float(pos.get("exchange_min_notional_usd") or pos.get("min_notional_usd"), 5.0))
                if p_sym and 0.0 <= val_usd < min_not:
                    dust_binance_syms.add(p_sym)
        cleaned_ag_syms: Dict[str, Any] = {}
        for ag_sym, ag_val in ag_state["symbols"].items():
            if not isinstance(ag_val, dict):
                continue
            sym_copy = dict(ag_val)
            if str(ag_sym).upper() in dust_binance_syms and "sell" in sym_copy:
                sell_st = sym_copy.get("sell")
                if not isinstance(sell_st, dict) or not sell_st.get("pending"):
                    sym_copy.pop("sell", None)
            if sym_copy:
                cleaned_ag_syms[ag_sym] = sym_copy
        if cleaned_ag_syms:
            out["assetguardian_state"] = cleaned_ag_syms

    ssl_state = _rj(os.path.join(cache_dir, "smart_stop_loss_state.json"))
    if isinstance(ssl_state, dict) and ssl_state:
        out["smart_stop_loss_state"] = ssl_state
    return out


MACRO_BENCHMARK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("SPY", "SPY", "usd"),
    ("QQQ", "QQQ", "usd"),
    ("ES=F", "ES=F", "usd"),
    ("NQ=F", "NQ=F", "usd"),
    ("DXY", "DX-Y.NYB", "index"),
    ("US10Y", "^TNX", "yield_pct"),
)


def format_macro_benchmarks_summary_line(data: Any) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / DXY / US10Y)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return ""
    ordered_keys = [k for k, _, _ in MACRO_BENCHMARK_SPECS if k in items]
    for extra_k in items:
        if extra_k not in ordered_keys and extra_k not in ("ts", "age_min", "summary_line", "items"):
            ordered_keys.append(extra_k)
    parts: List[str] = []
    for sym in ordered_keys:
        entry = items.get(sym)
        if not isinstance(entry, dict):
            continue
        px = safe_pos_float(entry.get("price"), default=None)
        if px is None:
            continue
        unit_kind = str(entry.get("unit") or "").lower()
        if sym == "US10Y" or unit_kind == "yield_pct":
            px_s = f"{px:.2f}%"
        elif sym == "DXY" or unit_kind == "index":
            px_s = f"{px:.2f}"
        else:
            px_s = f"${px:,.2f}"
        d_parts: List[str] = []
        for h_lbl, h_key in (("24h", "chg_24h_pct"), ("1h", "chg_1h_pct"), ("15m", "chg_15m_pct")):
            val = safe_float(entry.get(h_key), default=None)
            if val is not None:
                d_parts.append(f"{val:+.2f}% {h_lbl}")
        m_state = str(entry.get("market_state") or "").strip().upper()
        if m_state and m_state not in ("REGULAR", ""):
            d_parts.append(m_state.lower())
        delta_s = f" ({', '.join(d_parts)})" if d_parts else ""
        parts.append(f"{sym}={px_s}{delta_s}")
    return " | ".join(parts)


def collect_macro_benchmarks_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.5,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch SPY, QQQ, ES=F, NQ=F, DXY, and US10Y 24h/1h/15m deltas from Yahoo Finance and persist to cachedb."""
    eff_now = safe_pos_float(now_ts, default=None) or _time.time()
    _fetch = fetch_fn or fetch_public_json
    items: Dict[str, Dict[str, Any]] = {}
    for label, yahoo_sym, unit_kind in MACRO_BENCHMARK_SPECS:
        try:
            url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{_urlparse.quote(yahoo_sym)}?range=2d&interval=5m"
            )
            raw = _fetch(url, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"})
            if not isinstance(raw, dict):
                continue
            res_list = (raw.get("chart") or {}).get("result")
            if not isinstance(res_list, list) or not res_list or not isinstance(res_list[0], dict):
                continue
            res = res_list[0]
            meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
            ts_list = res.get("timestamp") if isinstance(res.get("timestamp"), list) else []
            q_list = ((res.get("indicators") or {}).get("quote") or [{}])
            q_obj = q_list[0] if isinstance(q_list, list) and q_list and isinstance(q_list[0], dict) else {}
            closes = q_obj.get("close") if isinstance(q_obj.get("close"), list) else []
            pts: List[tuple[float, float]] = [
                (ts_f, c_f)
                for ts, c in zip(ts_list, closes)
                if (ts_f := safe_pos_float(ts, default=None)) is not None
                and (c_f := safe_pos_float(c, default=None)) is not None
            ]
            px = (
                safe_pos_float(meta.get("fulldayPrice"), default=None)
                or safe_pos_float(meta.get("regularMarketPrice"), default=None)
                or (pts[-1][1] if pts else None)
            )
            if px is None:
                continue
            chg_24h: Optional[float] = None
            for meta_chg_k in ("fulldayChangePercent", "regularMarketChangePercent"):
                cand_chg = safe_float(meta.get(meta_chg_k), default=None)
                if cand_chg is not None and abs(cand_chg) > 1e-9:
                    chg_24h = round(cand_chg, 2)
                    break
            if chg_24h is None:
                prev_close = safe_pos_float(
                    meta.get("chartPreviousClose") or meta.get("previousClose"),
                    default=None,
                )
                if prev_close is not None and prev_close > 0:
                    chg_24h = round(((px - prev_close) / prev_close) * 100.0, 2)
                elif len(pts) >= 2 and pts[0][1] > 0:
                    chg_24h = round(((px - pts[0][1]) / pts[0][1]) * 100.0, 2)

            chg_1h: Optional[float] = None
            chg_15m: Optional[float] = None
            if len(pts) >= 2:
                last_bar_ts = pts[-1][0]
                for target_sec, attr_name in ((3600.0, "1h"), (900.0, "15m")):
                    target_ts = last_bar_ts - target_sec
                    ref_pt = min(pts, key=lambda p: abs(p[0] - target_ts))
                    if abs(ref_pt[0] - target_ts) <= target_sec * 1.5 and ref_pt[1] > 0:
                        pct_v = round(((px - ref_pt[1]) / ref_pt[1]) * 100.0, 2)
                        if attr_name == "1h":
                            chg_1h = pct_v
                        else:
                            chg_15m = pct_v

            item_entry: Dict[str, Any] = {
                "symbol": label,
                "yahoo_symbol": yahoo_sym,
                "unit": unit_kind,
                "price": round(px, 4 if unit_kind == "yield_pct" else 2),
                "chg_24h_pct": chg_24h if chg_24h is not None else 0.0,
            }
            if chg_1h is not None:
                item_entry["chg_1h_pct"] = chg_1h
            if chg_15m is not None:
                item_entry["chg_15m_pct"] = chg_15m
            if meta.get("marketState"):
                item_entry["market_state"] = str(meta.get("marketState")).strip().upper()
            if pts:
                cutoff_24h = pts[-1][0] - 86400.0
                vals_24h = [p[1] for p in pts if p[0] >= cutoff_24h] or [p[1] for p in pts]
                item_entry["low_24h"] = round(min(vals_24h), 4 if unit_kind == "yield_pct" else 2)
                item_entry["high_24h"] = round(max(vals_24h), 4 if unit_kind == "yield_pct" else 2)
            items[label] = item_entry
        except Exception:
            continue

    if not items:
        return None
    payload: Dict[str, Any] = {
        "ts": eff_now,
        "items": items,
        "summary_line": format_macro_benchmarks_summary_line({"items": items}),
    }
    if persist and cache_dir:
        try:
            save_json_state(os.path.join(cache_dir, "macro_benchmarks_collect.json"), payload, indent=2)
        except Exception:
            pass
    return payload


def apply_asset_scope_filter(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    portfolio_summary: Dict[str, Any],
    scope: str,
    inst_by_sym: Dict[str, Any],
    t212_cycles: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter prices_and_trends, microstructure, and portfolio_summary by asset_scope ('equities' | 'crypto')."""
    if scope not in ("equities", "crypto"):
        return prices_and_trends, microstructure
    want_t212 = scope == "equities"
    t212_sym_set = {str(s).upper() for s in t212_cycles} | {
        str(s).split(":", 1)[-1].upper()
        for s, iv in inst_by_sym.items()
        if normalize_venue_name(getattr(iv, "provider_name", None)) == "t212"
    }
    if want_t212:
        prices_and_trends = {
            k: v
            for k, v in prices_and_trends.items()
            if str(k).upper() in t212_sym_set or str(k).upper().endswith(("_EQ", "_US_EQ"))
        }
        microstructure = {}
        for crypto_key in (
            "total_value_usdc",
            "kraken_cycles",
            "hyperliquid_cycles",
            "hyperliquid_signal",
            "hyperliquid_readonly",
            "monitortrades_active_state",
            "assetguardian_state",
            "smart_stop_loss_state",
            "trailing_peaks",
        ):
            portfolio_summary.pop(crypto_key, None)
    else:
        prices_and_trends = {
            k: v for k, v in prices_and_trends.items()
            if not str(k).upper().endswith(("_EQ", "_US_EQ")) and str(k).upper() not in t212_sym_set
        }
        portfolio_summary.pop("t212_cycles", None)
    if isinstance(portfolio_summary.get("venue_quote_cash"), dict):
        filtered_cash = {
            ck: cv
            for ck, cv in portfolio_summary["venue_quote_cash"].items()
            if isinstance(cv, dict) and (normalize_venue_name(cv.get("venue")) == "t212") == want_t212
        }
        if filtered_cash:
            portfolio_summary["venue_quote_cash"] = filtered_cash
        else:
            portfolio_summary.pop("venue_quote_cash", None)
    if isinstance(portfolio_summary.get("recent_orders"), dict):

        def _is_eq_ro_key(rk: str, rv: Any = None) -> bool:
            if isinstance(rv, dict) and rv.get("venue"):
                return normalize_venue_name(rv.get("venue")) == "t212"
            rk_up = str(rk).upper()
            if rk_up.startswith("T212:") or rk_up.endswith(("_EQ", "_US_EQ")):
                return True
            if rk_up.startswith(("BINANCE:", "KRAKEN:", "HYPERLIQUID:")):
                return False
            return rk_up in t212_sym_set

        filtered_ro = {
            rk: rv
            for rk, rv in portfolio_summary["recent_orders"].items()
            if _is_eq_ro_key(rk, rv) == want_t212
        }
        if filtered_ro:
            portfolio_summary["recent_orders"] = filtered_ro
        else:
            portfolio_summary.pop("recent_orders", None)
    for list_key in ("tradable_positions", "recent_ai_intents", "active_retry_queue", "open_orders"):
        if list_key in portfolio_summary and isinstance(portfolio_summary[list_key], list):
            filtered_items = [
                item for item in portfolio_summary[list_key]
                if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
            ]
            if filtered_items:
                portfolio_summary[list_key] = filtered_items
            else:
                portfolio_summary.pop(list_key, None)
    if "open_limit_orders" in portfolio_summary and isinstance(portfolio_summary["open_limit_orders"], list):
        portfolio_summary["open_limit_orders"] = [
            item for item in portfolio_summary["open_limit_orders"]
            if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
        ]
    last_exec = portfolio_summary.get("last_cycle_execution_status")
    if isinstance(last_exec, dict):
        if (normalize_venue_name(last_exec.get("venue")) == "t212") != want_t212:
            portfolio_summary.pop("last_cycle_execution_status", None)
    elif isinstance(last_exec, list):
        filtered_exec = [
            item for item in last_exec
            if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
        ]
        if filtered_exec:
            portfolio_summary["last_cycle_execution_status"] = (
                filtered_exec[0] if len(filtered_exec) == 1 else filtered_exec
            )
        else:
            portfolio_summary.pop("last_cycle_execution_status", None)
    return prices_and_trends, microstructure


def _read_first_cached_dict(
    cache_dir: str,
    group_key: str,
    default_file: str,
    read_json_fn: Any = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Read the first non-empty dict matching `get_cache_filenames(group_key)` (or `default_file`) in `cache_dir`."""
    rj = read_json_cached if read_json_fn is None else read_json_fn
    for fname in (get_cache_filenames(group_key) or (default_file,)):
        fpath = os.path.join(cache_dir, fname)
        data = rj(fpath)
        if isinstance(data, dict) and data:
            return data, fpath
    return None, ""


def harvest_macro_and_news_sensors(
    cache_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Harvest Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, Macro Benchmarks, and Breaking News."""
    _rj = read_json_fn or read_json_recovering

    def _with_age_min(payload: Dict[str, Any], fpath: str) -> Dict[str, Any]:
        enriched = dict(payload)
        ref_ts = extract_timestamp_s(enriched)
        eff_path = str(resolve_cachedb_filepath(fpath))
        if ref_ts is None and os.path.exists(eff_path):
            try:
                ref_ts = os.path.getmtime(eff_path)
            except OSError:
                ref_ts = None
        if ref_ts is not None:
            enriched.setdefault("age_min", round(max(0.0, now_ts - ref_ts) / 60.0, 1))
        return enriched

    macro_sensors: Dict[str, Optional[Dict[str, Any]]] = {}
    for attr, group_key, fname in (
        ("fear_greed", "fear_greed_files", "fear_greed_collect.json"),
        ("market_breadth", "market_breadth_files", "market_breadth_collect.json"),
        ("sentiment_advisor", "sentiment_eval_files", "sentiment_eval.json"),
        ("geopolitical_threat", "geopolitical_eval_file", "geopolitical_eval.json"),
        ("macro_benchmarks", "macro_benchmarks_files", "macro_benchmarks_collect.json"),
    ):
        data, s_path = _read_first_cached_dict(cache_dir, group_key, fname, _rj)
        if data is not None:
            enriched = _with_age_min(data, s_path)
            if attr == "macro_benchmarks" and not enriched.get("summary_line"):
                summary_s = format_macro_benchmarks_summary_line(enriched)
                if summary_s:
                    enriched["summary_line"] = summary_s
            macro_sensors[attr] = enriched
    if (
        "macro_benchmarks" not in macro_sensors
        and read_json_fn is None
        and "PYTEST_CURRENT_TEST" not in os.environ
    ):
        live_mb = collect_macro_benchmarks_snapshot(cache_dir, now_ts=now_ts, timeout_sec=2.5, persist=True)
        if isinstance(live_mb, dict) and live_mb:
            macro_sensors["macro_benchmarks"] = _with_age_min(
                live_mb, os.path.join(cache_dir, "macro_benchmarks_collect.json")
            )

    breaking_news: List[Dict[str, Any]] = []
    news_data = _rj(os.path.join(cache_dir, "news_feed_collect.json"))
    if isinstance(news_data, list) and news_data:
        valid_news: List[Dict[str, Any]] = []
        for item in news_data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            raw_kws = item.get("matched_keywords") if isinstance(item.get("matched_keywords"), list) else []
            clean_kws = list(filter_valid_severity_keywords(raw_kws, title=str(item.get("title") or ""))) if raw_kws else []
            is_high_sev = bool(item.get("is_high_severity"))
            if raw_kws and not clean_kws:
                is_high_sev = False
            norm_item = dict(item)
            norm_item["matched_keywords"] = clean_kws
            norm_item["is_high_severity"] = is_high_sev
            valid_news.append(norm_item)
        valid_news.sort(
            key=lambda x: (
                1 if x.get("is_high_severity") else 0,
                safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
            ),
            reverse=True,
        )
        for item in valid_news[:6]:
            seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            age_m = round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None
            breaking_news.append({
                "title": item.get("title"),
                "source": item.get("source"),
                "is_high_severity": bool(item.get("is_high_severity")),
                "keywords": item.get("matched_keywords") or [],
                "age_min": age_m,
            })
    return macro_sensors, breaking_news


def harvest_initial_prices_and_trends(
    cache_dir: str,
    prices_data: Any,
    trends_data: Any,
    read_json_fn: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Build initial prices_and_trends map from cache_current_price, cache_short_trend, and 24trend/live_*.json."""
    _rj = read_json_fn or read_json_cached
    prices_and_trends: Dict[str, Dict[str, Any]] = {}
    price_map: Dict[str, float] = {}
    if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict):
        for sym, item in prices_data["items"].items():
            px_val = extract_current_price_item(item)
            if px_val is not None:
                price_map[sym] = px_val

    if isinstance(trends_data, dict):
        trends_items = trends_data.get("items") if isinstance(trends_data.get("items"), dict) else trends_data
        for sym, tinfo in trends_items.items():
            if sym in ("items", "fetchtime", "rest_sync_cursor", "metadata", "counts"):
                continue
            if isinstance(tinfo, dict):
                entry: Dict[str, Any] = {}
                apply_short_trend_item(entry, tinfo, fallback_price=price_map.get(sym))
                prices_and_trends[sym] = entry

    for sym, px in price_map.items():
        if sym not in prices_and_trends:
            prices_and_trends[sym] = {"price": px, "trend": 0, "growth_coefficient": 0.0}

    live_24trend_dir = str(resolve_cachedb_filepath(os.path.join(cache_dir, "24trend")))
    if os.path.isdir(live_24trend_dir):
        for lt_path in sorted(_glob.glob(os.path.join(live_24trend_dir, "live_*.json"))):
            fname = os.path.basename(lt_path)
            sym = fname[len("live_") : -len(".json")].upper()
            if not sym:
                continue
            lt_win = _rj(lt_path)
            if not isinstance(lt_win, dict):
                continue
            entry = prices_and_trends.setdefault(sym, {})
            apply_live_24trend_windows(entry, lt_win, fallback_price=price_map.get(sym))
    return prices_and_trends


def build_cycle_fallback_prices(
    microstructure: Dict[str, Any],
    watcher_price_items: Any,
) -> Dict[str, float]:
    """Build fallback price map from orderbook mid-prices and watcher_all_price series for cycle state parsing."""
    cycle_fallback_prices: Dict[str, float] = {}
    for m_sym, m_dict in microstructure.items():
        if isinstance(m_dict, dict):
            ob_mid_v = safe_pos_float(m_dict.get("ob_mid_price"), default=0.0) or 0.0
            if ob_mid_v <= 0:
                bb_v = safe_pos_float(m_dict.get("best_bid"), default=0.0) or 0.0
                ba_v = safe_pos_float(m_dict.get("best_ask"), default=0.0) or 0.0
                if bb_v > 0 and ba_v >= bb_v:
                    ob_mid_v = (bb_v + ba_v) / 2.0
            if ob_mid_v > 0:
                cycle_fallback_prices[m_sym] = ob_mid_v
    if isinstance(watcher_price_items, dict):
        for m_sym, m_ser in watcher_price_items.items():
            if isinstance(m_ser, list) and m_ser and isinstance(m_ser[-1], (list, tuple)) and len(m_ser[-1]) > 1:
                m_px = safe_pos_float(m_ser[-1][1], default=0.0) or 0.0
                if m_px > 0:
                    cycle_fallback_prices.setdefault(str(m_sym).upper(), m_px)
    return cycle_fallback_prices


def harvest_portfolio_and_venue_cycles(
    cache_dir: str,
    workspace_dir: str,
    logs_dirs: List[str],
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    watcher_price_items: Any,
    inst_by_sym: Dict[str, Any],
    read_json_fn: Any = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest portfolio value, venue cycles, Kraken xStocks, cross-market context, and portfolio_snapshot tail."""
    _rj = read_json_fn or read_json_cached
    last_av = read_last_jsonl_dict(os.path.join(cache_dir, "cache_asset_value.jsonl"), tail_bytes=2048)
    if isinstance(last_av, dict) and isinstance(last_av.get("i"), dict):
        portfolio_summary["total_value_usdc"] = last_av["i"].get("total_value_usdc")

    cycle_fallback_prices = build_cycle_fallback_prices(microstructure, watcher_price_items)
    kraken_cycles, hyperliquid_cycles, all_t212_cycles = harvest_venue_cycles(
        workspace_dir,
        prices_and_trends,
        cycle_fallback_prices,
        _rj,
    )
    if kraken_cycles:
        portfolio_summary["kraken_cycles"] = kraken_cycles
    if hyperliquid_cycles:
        portfolio_summary["hyperliquid_cycles"] = hyperliquid_cycles

    has_active_t212 = any(
        normalize_venue_name(getattr(inst, "provider_name", "")) == "t212"
        for inst in inst_by_sym.values()
    )
    t212_cycles = dict(all_t212_cycles) if has_active_t212 else {}
    if t212_cycles:
        portfolio_summary["t212_cycles"] = t212_cycles

    xst_summary = harvest_kraken_xstocks(
        os.path.join(workspace_dir, "kraken"),
        prices_and_trends,
        all_t212_cycles,
        _rj,
    )
    if xst_summary:
        portfolio_summary["kraken_xstocks"] = xst_summary

    cross_market_ctx = build_cross_market_context(
        prices_and_trends,
        microstructure,
        all_t212_cycles,
    )
    if cross_market_ctx:
        portfolio_summary["cross_market_context"] = cross_market_ctx

    for l_dir in logs_dirs:
        last_ps = read_last_jsonl_dict(os.path.join(l_dir, "portfolio_snapshot.jsonl"), tail_bytes=4096)
        if isinstance(last_ps, dict):
            if isinstance(last_ps.get("hl"), dict):
                portfolio_summary["hyperliquid_readonly"] = last_ps["hl"]
            break

    return kraken_cycles, hyperliquid_cycles, t212_cycles


def populate_market_and_portfolio_snapshot(
    snapshot: Any,
    *,
    cache_dir: str,
    workspace_dir: str,
    logs_dirs: List[str],
    intent_gateway: Any,
    scope: str,
    now_ts: float,
    read_json_fn: Any = None,
    log: Optional[Any] = None,
) -> Any:
    """Populate market, macro, microstructure, portfolio, cycles, fills, and runtime states on a TelemetrySensorSnapshot."""
    _rj = read_json_fn or read_json_recovering

    macro_sensors, snapshot.breaking_news = harvest_macro_and_news_sensors(cache_dir, now_ts, _rj)
    for attr, val in macro_sensors.items():
        setattr(snapshot, attr, val)

    prices_data = _rj(os.path.join(cache_dir, "cache_current_price.json"))
    trends_data = _rj(os.path.join(cache_dir, "cache_short_trend.json"))
    long_trends_data = _rj(os.path.join(cache_dir, "cache_long_trend.json"))
    priceanalysis_data = _rj(os.path.join(cache_dir, "priceanalysis.json"))
    t_trend_data = _rj(os.path.join(cache_dir, "cache_T_trend.json"))
    kalman_data = _rj(os.path.join(cache_dir, "cache_kalman_trend.json"))
    watcher_all_price_data = _rj(os.path.join(cache_dir, "cache_watcher_all_price.json"))
    trailing_map, trailing_by_venue = load_venue_trailing_maps(cache_dir, workspace_dir, _rj)
    long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}
    t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}
    snapshot.prices_and_trends = harvest_initial_prices_and_trends(cache_dir, prices_data, trends_data, _rj)

    intents_data = _rj(os.path.join(cache_dir, "autonomous_trade_intents_state.json"))
    if isinstance(intents_data, list) and intents_data:
        reconcile_fn = getattr(intent_gateway, "reconcile_intents", None)
        if callable(reconcile_fn):
            try:
                if reconcile_fn(intents_data):
                    save_fn = getattr(intent_gateway, "save_all_intents", None)
                    if callable(save_fn):
                        save_fn(intents_data)
            except Exception:
                pass
    known_ai_oids = collect_known_ai_order_ids(intents_data)
    gw_max_usd = safe_pos_float(getattr(intent_gateway, "max_notional_usd", None), default=2000.0) or 2000.0

    tradable_positions: List[Dict[str, Any]] = []
    open_orders_list: List[Dict[str, Any]] = []
    venue_quote_cash: Dict[str, Dict[str, Any]] = {}
    inst_by_sym: Dict[str, Any] = {}
    load_tradable_fn = getattr(intent_gateway, "load_tradable_instruments", None)
    has_inst_source = (
        os.path.exists(os.path.join(workspace_dir, "instruments.conf"))
        or os.path.exists(os.path.join(workspace_dir, "kraken", "config.env"))
        or os.path.exists(os.path.join(workspace_dir, "hyperliquid", "config.env"))
        or (scope == "equities" and os.path.isdir(os.path.join(workspace_dir, "212trading")))
    )
    if callable(load_tradable_fn) and has_inst_source:
        try:
            try:
                inst_map = load_tradable_fn(
                    only_workspace_conf=(scope != "equities"),
                    include_closed_rth=(scope == "equities"),
                    asset_scope=scope,
                )
            except TypeError:
                inst_map = load_tradable_fn(only_workspace_conf=True)
        except Exception as exc:
            if log:
                log.debug("Could not load tradable instruments for telemetry: %s", exc)
            inst_map = {}

        tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = (
            harvest_tradable_positions_and_orders(
                inst_map,
                snapshot.prices_and_trends,
                trailing_map,
                trailing_by_venue,
                known_ai_order_ids=known_ai_oids,
                gateway_max_usd=gw_max_usd,
                cache_dir=cache_dir,
            )
        )

    if tradable_positions:
        snapshot.portfolio_summary["tradable_positions"] = tradable_positions
    if open_orders_list:
        snapshot.portfolio_summary["open_orders"] = open_orders_list
        snapshot.portfolio_summary["open_limit_orders"] = list(open_orders_list)
    if venue_quote_cash:
        snapshot.portfolio_summary["venue_quote_cash"] = venue_quote_cash
    if trailing_map:
        snapshot.portfolio_summary["trailing_peaks"] = trailing_map

    watcher_price_items = (
        watcher_all_price_data.get("items", {}) if isinstance(watcher_all_price_data, dict) else {}
    )
    enrich_cross_venue_trends(
        snapshot.prices_and_trends,
        trends_data,
        kalman_data,
        long_items,
        priceanalysis_data,
        t_trend_map,
        watcher_price_items,
    )

    cur_p_items = (
        prices_data.get("items", {})
        if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict)
        else {}
    )
    enrich_24h_and_microstructure(
        cache_dir,
        workspace_dir,
        snapshot.prices_and_trends,
        snapshot.microstructure,
        cur_p_items,
        inst_by_sym,
        now_ts,
        _rj,
    )

    sync_sibling_symbol_telemetry(snapshot.prices_and_trends, snapshot.microstructure)

    kraken_cycles, hyperliquid_cycles, t212_cycles = harvest_portfolio_and_venue_cycles(
        cache_dir,
        workspace_dir,
        logs_dirs,
        snapshot.portfolio_summary,
        snapshot.prices_and_trends,
        snapshot.microstructure,
        watcher_price_items,
        inst_by_sym,
        _rj,
    )

    recent_orders = harvest_recent_orders_and_enrich_positions(
        cache_dir,
        inst_by_sym,
        snapshot.portfolio_summary.get("tradable_positions", []),
        snapshot.prices_and_trends,
        kraken_cycles,
        hyperliquid_cycles,
        t212_cycles,
        _rj,
        known_ai_order_ids=known_ai_oids,
        gateway_max_usd=gw_max_usd,
        ai_intents=intents_data if isinstance(intents_data, list) else None,
    )
    if recent_orders:
        snapshot.portfolio_summary["recent_orders"] = recent_orders

    if isinstance(intents_data, list) and intents_data:
        recent_intents_out = summarize_recent_ai_intents(
            intents_data,
            scope,
            now_ts,
            extract_intent_notional,
            recent_orders=recent_orders,
            open_orders=open_orders_list,
        )
        if recent_intents_out:
            snapshot.portfolio_summary["recent_ai_intents"] = recent_intents_out
        last_exec_status = build_last_cycle_execution_status(
            intents_data,
            scope,
            now_ts,
            extract_intent_notional,
            recent_orders=recent_orders,
            open_orders=open_orders_list,
        )
        if last_exec_status is not None:
            snapshot.portfolio_summary["last_cycle_execution_status"] = last_exec_status

    snapshot.portfolio_summary.update(
        harvest_bot_runtime_states(
            cache_dir,
            now_ts,
            _rj,
            tradable_positions=snapshot.portfolio_summary.get("tradable_positions"),
        )
    )

    snapshot.prices_and_trends, snapshot.microstructure = apply_asset_scope_filter(
        snapshot.prices_and_trends,
        snapshot.microstructure,
        snapshot.portfolio_summary,
        scope,
        inst_by_sym,
        t212_cycles,
    )
    mb_snap = getattr(snapshot, "macro_benchmarks", None)
    if isinstance(mb_snap, dict) and mb_snap and snapshot.portfolio_summary:
        snapshot.portfolio_summary["macro_benchmarks"] = mb_snap
        cmc = snapshot.portfolio_summary.get("cross_market_context")
        if isinstance(cmc, dict):
            cmc["macro_benchmarks"] = mb_snap
    if snapshot.portfolio_summary:
        snapshot.portfolio_summary.setdefault("open_limit_orders", [])
    return snapshot


def format_bottom_structure_6h_parts(
    m24: Dict[str, Any],
    btc_m24: Optional[Dict[str, Any]] = None,
    *,
    prior_label: str = "prior_6h_low",
    px_prefix: str = "$",
    check_smt: bool = True,
) -> Tuple[Optional[str], List[str]]:
    """Build 6h intraday bottom structure label and formatted metric parts (including SMT vs BTC)."""
    b_struct = m24.get("bottom_structure_6h") if isinstance(m24, dict) else None
    if not b_struct:
        return None, []
    b_parts = [
        f"{prior_label}={px_prefix}{m24.get('prior_6h_low_px')}",
        f"1h_low={px_prefix}{m24.get('recent_1h_low_px')}",
    ]
    if m24.get("higher_low_vs_prior_pct") is not None:
        b_parts.append(f"hl_diff={safe_float(m24.get('higher_low_vs_prior_pct'), 0.0):+.2f}%")
    if m24.get("reclaim_above_prior_low_pct") is not None:
        b_parts.append(f"reclaim={safe_float(m24.get('reclaim_above_prior_low_pct'), 0.0):+.2f}%")
    if check_smt and b_struct == "HIGHER_LOW_DOUBLE_BOTTOM" and isinstance(btc_m24, dict) and btc_m24:
        btc_b = btc_m24.get("bottom_structure_6h")
        btc_pos24 = safe_float(btc_m24.get("pos_24h_pct"), 100.0)
        if btc_b in ("LIQUIDITY_SWEEP_RECLAIM", "FRESH_6H_LOW_TEST") or btc_pos24 <= 35.0:
            b_parts.append("smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE")
    return str(b_struct), b_parts


def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) to provide deep context to the LLM."""
    default_cdir = get_cachedb_dir()
    if cachedb_dir == "cachedb":
        cachedb_dir = default_cdir
    sym_quotes = compatible_quote_suffixes(symbol)
    is_usd_sym = "USD" in sym_quotes
    sym_variants = cache_symbol_variants(symbol, include_base=True)
    quote_variants = cache_symbol_variants(symbol, include_base=is_usd_sym, quotes=sym_quotes)

    def _disk_telem(prefix: str) -> Optional[Dict[str, Any]]:
        for cand in sym_variants:
            d = read_json_cached(os.path.join(cachedb_dir, f"{prefix}{cand}.json"))
            if isinstance(d, dict):
                return d if d.get("symbol") else {**d, "symbol": cand}
        return None

    def _fetch_pillar2(getter_fn: Callable[[], Any], prefix: str) -> Any:
        try:
            col = getter_fn()
            col_cdir = getattr(col, "_cache_dir", None)
            if not isinstance(col_cdir, str) or os.path.abspath(col_cdir) == os.path.abspath(cachedb_dir):
                snap = col.fetch(symbol, allow_network=False)
                if snap is not None:
                    return snap
        except Exception:
            pass
        return _disk_telem(prefix)

    telemetry: Dict[str, Any] = {}
    ob_snap = None
    f_snap = None
    w_snap = None
    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = _fetch_pillar2(get_orderbook_collector, "orderbook_depth_")
        if ob_snap is not None:
            imb_f = safe_float(obj_attr(ob_snap, "imbalance_ratio", None), default=None)
            if imb_f is not None:
                bid_pct = max(0, min(100, int(round(imb_f * 100.0))))
                ask_pct = 100 - bid_pct
                telemetry["Orderbook Depth Imbalance"] = f"{imb_f:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
            ob_sym_raw = obj_attr(ob_snap, "symbol")
            ob_same_q = (
                compatible_quote_suffixes(ob_sym_raw) == sym_quotes
                if isinstance(ob_sym_raw, str) and ob_sym_raw.strip()
                else is_usd_sym
            )
            for side_lbl, usd_attr, px_attr in (
                ("Ask", "largest_ask_wall_usd", "largest_ask_wall_price"),
                ("Bid", "largest_bid_wall_usd", "largest_bid_wall_price"),
            ):
                if not ob_same_q:
                    continue
                w_usd = safe_pos_float(obj_attr(ob_snap, usd_attr, 0.0), 0.0)
                w_px = safe_pos_float(obj_attr(ob_snap, px_attr, 0.0), 0.0)
                w_px_fmt = f"{w_px:,.2f}" if w_px >= 1.0 else f"{w_px:.6f}".rstrip("0").rstrip(".")
                telemetry[f"Orderbook {side_lbl} Wall Ahead"] = (
                    f"${w_usd:,.0f} @ ${w_px_fmt}"
                    if w_usd > 100_000.0 and w_px > 0.0
                    else "None detected within 2%"
                )
    except Exception:
        pass

    # 2. Perpetual Derivatives: Funding Rate & Open Interest (Pillar 2)
    try:
        from external_order_guard import get_derivatives_collector
        f_snap = _fetch_pillar2(get_derivatives_collector, "derivatives_telemetry_")
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
        w_snap = _fetch_pillar2(get_whale_collector, "whale_snapshot_")
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
                gf_s = f"{gf:+.2e}" if 0.0 < abs(gf) < 1e-4 else f"{gf:+.4f}"
                ef_s = f"{ef:.2e}" if 0.0 < abs(ef) < 1e-4 else f"{ef:.4f}"
                trend_str += f" (gradient={gf_s}, epsilon={ef_s})"
            telemetry["Technical Trend Regime"] = trend_str
        except Exception:
            pass

    df = safe_pos_float(dur_sec, 0.0)
    if df > 0.0:
        telemetry["Trend Duration"] = f"{df / 3600.0:.1f} hours"

    # 4B. 24h Multi-Horizon Price Evolution & Dynamic Multi-Window Slopes (shared with Reconciler)
    try:
        cur_p_data = read_json_cached(os.path.join(cachedb_dir, "cache_current_price.json"))
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
            if m24.get("poc_px") is not None:
                telemetry["24h Volume Profile (POC / Value Area)"] = (
                    f"POC=${m24.get('poc_px')} ({float(m24.get('poc_dist_pct', 0.0)):+.2f}%), "
                    f"VAL=${m24.get('val_px')} ({float(m24.get('val_dist_pct', 0.0)):+.2f}%), "
                    f"VAH=${m24.get('vah_px')} ({float(m24.get('vah_dist_pct', 0.0)):+.2f}%), "
                    f"va_pos={m24.get('va_pos_pct')}%, regime={m24.get('va_regime')}"
                )
            btc_m24: Dict[str, Any] = {}
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
            if m24.get("bottom_structure_6h"):
                b_struct, b_parts = format_bottom_structure_6h_parts(
                    m24, btc_m24, prior_label="prior_6h_low", px_prefix="$", check_smt=bool(btc_m24)
                )
                if b_struct:
                    telemetry["6h Intraday Bottom Structure"] = f"{b_struct} ({', '.join(b_parts)})"

        cur_m_info = load_symbol_microstructure(
            cachedb_dir, symbol, ob=ob_snap, deriv=f_snap, whale=w_snap
        )
        if cur_m_info:
            if cur_m_info.get("spread_bps") is not None:
                telemetry["Orderbook Bid-Ask Spread"] = f"{cur_m_info['spread_bps']} bps"
            d1_b = safe_pos_float(cur_m_info.get("depth_1pct_bid_usd"), default=0.0) or 0.0
            d1_a = safe_pos_float(cur_m_info.get("depth_1pct_ask_usd"), default=0.0) or 0.0
            if d1_b > 0.0 or d1_a > 0.0:
                slip_s = (
                    f", est_slip_2k={cur_m_info['est_slip_2k_bps']} bps"
                    if cur_m_info.get("est_slip_2k_bps") is not None
                    else ""
                )
                telemetry["Orderbook Depth Within 1%"] = f"bid=${d1_b:,.0f} / ask=${d1_a:,.0f}{slip_s}"
            cur_px_est = safe_pos_float(alt_series_24h[-1][1], default=0.0) if alt_series_24h else 0.0
            if not cur_px_est or cur_px_est <= 0.0:
                cur_px_est = next(
                    (
                        px_v
                        for cand in quote_variants
                        if (px_v := extract_current_price_item(casefold_get(cur_p_items, cand))) is not None
                        and px_v > 0.0
                    ),
                    0.0,
                )
            if not is_usd_sym and (
                not cur_px_est or cur_px_est <= 0.0 or _prices_diverged(cur_px_est, cur_m_info.get("ob_mid_price"))
            ):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    cur_m_info.pop(k_strip, None)
            p_est = dict(m24)
            if cur_px_est and cur_px_est > 0:
                p_est["price"] = cur_px_est
            enrich_microstructure_cvd(cur_m_info, p_est)
            if cur_m_info.get("cvd_15m_usd") is not None:
                cvd_u = safe_float(cur_m_info.get("cvd_15m_usd"), 0.0)
                cvd_reg = cur_m_info.get("cvd_regime") or "balanced_flow"
                telemetry["15m Cumulative Volume Delta (CVD)"] = f"${cvd_u:+,.0f} ({cvd_reg})"
            liq_c = compute_liquidation_clusters(p_est, cur_m_info)
            if liq_c.get("long_liq_px") is not None and liq_c.get("short_liq_px") is not None:
                deep_parts = []
                if liq_c.get("long_liq_deep_px") is not None:
                    deep_parts.append(
                        f"deep_LONG=${liq_c['long_liq_deep_px']} (${liq_c.get('long_liq_deep_usd', 0):,.0f}, {float(liq_c.get('long_liq_deep_dist_pct', 0.0)):+.2f}%)"
                    )
                if liq_c.get("short_liq_deep_px") is not None:
                    deep_parts.append(
                        f"deep_SHORT=${liq_c['short_liq_deep_px']} (${liq_c.get('short_liq_deep_usd', 0):,.0f}, {float(liq_c.get('short_liq_deep_dist_pct', 0.0)):+.2f}%)"
                    )
                deep_sfx = f", {', '.join(deep_parts)}" if deep_parts else ""
                telemetry["Liquidation Clusters (LONG vs SHORT)"] = (
                    f"LONG liq=${liq_c['long_liq_px']} (${liq_c.get('long_liq_usd', 0):,.0f}, {float(liq_c.get('long_liq_dist_pct', 0.0)):+.2f}%), "
                    f"SHORT liq=${liq_c['short_liq_px']} (${liq_c.get('short_liq_usd', 0):,.0f}, {float(liq_c.get('short_liq_dist_pct', 0.0)):+.2f}%){deep_sfx}, "
                    f"magnet={liq_c.get('liq_magnet_bias')}"
                )
    except Exception:
        pass

    try:
        st_path = os.path.join(cachedb_dir, "cache_short_trend.json")
        st_data = read_json_cached(st_path)
        if isinstance(st_data, dict):
            st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
            tinfo = _first_variant_dict(st_items, sym_variants)
            if isinstance(tinfo, dict):
                t_entry: Dict[str, Any] = {}
                apply_short_trend_item(t_entry, tinfo)
                s37 = safe_float(t_entry.get("slope_3_7m"), 0.0)
                s15 = safe_float(t_entry.get("slope_15m"), 0.0)
                s25 = safe_float(t_entry.get("slope_2_5h"), 0.0)
                _fmt_s = lambda v: f"{v:+.2e}" if 0.0 < abs(v) < 1e-4 else f"{v:+.5f}"
                telemetry["Dynamic Multi-Window Slopes"] = (
                    f"3.7m={_fmt_s(s37)}, 15m={_fmt_s(s15)}, 2.5h={_fmt_s(s25)}"
                )
    except Exception:
        pass

    try:
        known_ai_oids = collect_known_ai_order_ids(
            read_json_recovering(os.path.join(cachedb_dir, "autonomous_trade_intents_state.json"))
        )
        audit_origin_map = _load_recent_audit_origins(cachedb_dir)
        fills: Optional[Dict[str, Any]] = None
        best_fill_ts = -1.0
        for ord_file in ("cache_order.json", "cache_trade_kraken.json", "cache_trade_hyperliquid.json", "cache_trade_t212.json"):
            ord_path = os.path.join(cachedb_dir, ord_file)
            ord_data = read_json_cached(ord_path)
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) else {}
            if not isinstance(o_items, dict):
                continue
            orders_for_sym = next((val for cand in quote_variants if (val := casefold_get(o_items, cand))), [])
            cand_fills = extract_symbol_recent_fills(
                orders_for_sym,
                known_ai_order_ids=known_ai_oids,
                audit_origin_map=audit_origin_map,
            )
            if cand_fills:
                cand_ts = max(
                    safe_nonneg_ts((cand_fills.get("last_buy") or {}).get("ts_ms"), 0.0),
                    safe_nonneg_ts((cand_fills.get("last_sell") or {}).get("ts_ms"), 0.0),
                )
                if fills is None or cand_ts > best_fill_ts:
                    fills = cand_fills
                    best_fill_ts = cand_ts
        if fills:
            lb = fills.get("last_buy") or {}
            ls = fills.get("last_sell") or {}
            lb_by = f" (by={lb.get('actor_tag') or lb.get('actor')})" if (lb.get("actor_tag") or lb.get("actor")) else ""
            ls_by = f" (by={ls.get('actor_tag') or ls.get('actor')})" if (ls.get("actor_tag") or ls.get("actor")) else ""
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb.get('qty')} @ {lb.get('price')}{lb_by}, "
                f"Last SELL={ls.get('qty')} @ {ls.get('price')}{ls_by}"
            )
    except Exception:
        pass

    # 5. Macro Sentiment Bias & Fear/Greed Index (Pillar 3)
    try:
        m_data, _ = _read_first_cached_dict(cachedb_dir, "sentiment_eval_files", "sentiment_eval.json")
        if isinstance(m_data, dict):
            bias = str(m_data.get("market_bias") or "").strip()
            risk = str(m_data.get("risk_level") or "").strip()
            action = str(m_data.get("recommended_action") or "").strip()
            if bias or risk or action:
                telemetry["Macro Sentiment Bias"] = f"{bias or 'UNKNOWN'} (Risk: {risk or 'UNKNOWN'}, Recommendation: {action or 'NONE'})"
    except Exception:
        pass

    try:
        fg_data, _ = _read_first_cached_dict(cachedb_dir, "fear_greed_files", "fear_greed_collect.json")
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
        g_data, _ = _read_first_cached_dict(cachedb_dir, "geopolitical_eval_file", "geopolitical_eval.json")
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


def resolve_llm_guard_cache_ttl(
    cache_ttl_sec: Any,
    explicit_client: bool,
    gemini_client: Any,
    default_ttl_sec: float,
) -> float:
    """Resolve effective LLM guard decision cache TTL (0.0 when custom client/runner is injected)."""
    if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
        return safe_nonneg_ts(cache_ttl_sec, 0.0)
    if not explicit_client and getattr(gemini_client, "_custom_runner", None) is None:
        return default_ttl_sec
    return 0.0


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
        timeout_sec: float = 120.0,
        fallback_action: str = "allow",  # "allow" | "downscale" | "veto"
        cache_ttl_sec: Optional[float] = None,
    ) -> None:
        self._explicit_client = gemini_client is not None
        self.gemini_client = gemini_client or GeminiClient()
        raw_min = min_notional_usd if min_notional_usd is not None else min_notional_eur
        resolved_min = safe_pos_float(raw_min, DEFAULT_MIN_NOTIONAL_USD)
        self.min_notional_usd = resolved_min
        self.min_notional_eur = resolved_min
        self.timeout_sec = safe_pos_float(timeout_sec, 120.0)
        self.fallback_action = str(fallback_action or "allow").strip().lower()
        self.cache_ttl_sec = resolve_llm_guard_cache_ttl(
            cache_ttl_sec, self._explicit_client, self.gemini_client, DEFAULT_HIGH_STAKE_CACHE_TTL_SEC
        )

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

        notional = resolve_order_notional(notional_eur, price, qty, default=0.0, fallback_on_nonfinite=False) or 0.0
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
                "3. MACRO, GEOPOLITICAL & CONTRARIAN BOTTOM ACCUMULATION: If Macro Sentiment recommends CAUTION or Geopolitical threat is elevated, do not fight negative taker flow. However, when 6h Intraday Bottom Structure confirms 'HIGHER_LOW_DOUBLE_BOTTOM' or 'LIQUIDITY_SWEEP_RECLAIM' (especially with 'smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE') AND Whale Taker Flow is >= 1.0 (or 15m CVD is positive/spot_absorbing), issue APPROVED — do NOT veto or downscale a verified contrarian bottom entry merely because lagging 24h breadth or residual geopolitical headlines remain elevated.\n\n"
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
                "3. CAPITULATION & SQUEEZE PROTECTION (6H DOUBLE-BOTTOM & RECLAIM DEFENSE): Do not approve panic-selling into extreme negative funding, active whale accumulation, or a verified 6h Intraday Bottom Structure ('HIGHER_LOW_DOUBLE_BOTTOM', 'LIQUIDITY_SWEEP_RECLAIM', or 'FRESH_6H_LOW_TEST') at local bottoms.\n\n"
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
