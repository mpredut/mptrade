"""Stateless quantitative price-series, volume-profile, swing-wave, Kalman, and multi-horizon trend evaluators."""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
import math
import os
import statistics
import time as _time
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from botcore import (
    _prices_diverged,
    is_live_network_backfill_enabled,
    is_offline_disk_cache_enabled,
    lru_dict_put,
    normalize_epoch_sec,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from pricewindow import (
    adaptive_thresholds,
    calculate_volatility_1h,
    KalmanTrend as _DefaultKalmanTrend,
)
from cachepaths import cache_ohlc_24h_filename, cache_path, canonical_base_asset, resolve_cachedb_filepath
from instrument_registry import is_equity_symbol, is_symbol_market_open, resolve_symbol_identity
from intelligence.internal.guards.guard_decision import OrderGuardEvaluatorFactory
from intelligence.macro_sentiment_ssot import (
    ALL_KINEMATIC_METRIC_KEYS,
    KINEMATIC_HORIZON_SPECS,
    SHORT_KINEMATIC_METRIC_KEYS,
    _TWO_COL_TS_PX_IDS,
    _extract_sorted_ts_px,
    _finalize_sorted_ts_px,
    parse_ts_px_row,
    _series_memo_anchor,
    _series_memo_key,
    callable_identity_anchor,
    resolve_kalman_vel_z,
    same_callable_anchor,
)
from state_io import generation_file_lock, read_json_dict, read_json_recovering, save_json_state

KalmanTrend: Any = None
_OHLC_24H_DISK_MEMO: Dict[str, Tuple[Tuple[int, int], List[List[Any]]]] = {}
_LAST_OHLC_WRITER_ID_BY_PATH: Dict[str, Any] = {}
LAST_7D_SPAN_MS_BY_SYM: Dict[str, float] = {}
LAST_7D_4H_BARS_BY_SYM: Dict[str, List[List[Any]]] = {}
_NY_TZ = ZoneInfo("America/New_York")
_MS_15M = 900_000.0
_MS_1H = 3_600_000.0
_MS_4H = 14_400_000.0
_MS_1D = 86_400_000.0


def round_opt_px(val: Any, decimals: int = 6) -> Optional[float]:
    """Round a positive price anchor to ``decimals``, returning None when absent or non-positive."""
    vf = safe_pos_float(val, default=None)
    return round(vf, decimals) if vf is not None else None


def _auto_px_decimals(px: float, decimals: Optional[int] = None) -> int:
    return decimals if decimals is not None else (2 if px >= 100.0 else (4 if px >= 1.0 else 6))


def _round_small_slope(val: float) -> float:
    return ((safe_float(f"{val:.2e}", 0.0) or 0.0) if 0.0 < abs(val) < 1e-4 else round(val, 6)) + 0.0


_BUCKET_5M_MEMO: Dict[Tuple[Any, int, Any, Any, Any], Tuple[Any, Dict[int, float]]] = {}
_PT_TS_KEY = itemgetter(0)


def _tpo_sample_at(pts: List[Tuple[float, float]], idx: int) -> Tuple[float, float]:
    px_f = pts[idx][1]
    prev_px = pts[idx - 1][1] if idx > 0 else None
    vol_w = 1.0 + min(5.0, (abs(px_f - prev_px) / prev_px) * 1000.0) if prev_px else 1.0
    return (px_f, vol_w)


def pct_delta(cur_px: float, ref_px: float, decimals: int = 2) -> float:
    """Compute percentage delta `((cur_px / ref_px) - 1.0) * 100.0` rounded to `decimals`."""
    return round((cur_px / ref_px - 1.0) * 100.0, decimals) + 0.0


def range_position_pct(cur_px: float, low_px: float, high_px: float, decimals: int = 1) -> float:
    """Compute normalized [0..100] range position `((cur_px - low_px) / (high_px - low_px)) * 100.0`."""
    return (round(((cur_px - low_px) / (high_px - low_px)) * 100.0, decimals) + 0.0) if high_px > low_px else 50.0


def build_vp_position_dict(cur_px: float, poc_px: float, val_px: float, vah_px: float) -> Dict[str, Any]:
    """Compute symmetric % distances from POC/VAL/VAH, Value Area position %, and VA regime."""
    if min(cur_px, poc_px, val_px, vah_px) <= 0.0:
        return {}
    va_pos = max(-100.0, min(200.0, range_position_pct(cur_px, val_px, vah_px)))
    return {
        "poc_dist_pct": pct_delta(cur_px, poc_px),
        "vah_dist_pct": pct_delta(cur_px, vah_px),
        "val_dist_pct": pct_delta(cur_px, val_px),
        "va_pos_pct": va_pos,
        "va_regime": (
            "ABOVE_VAH" if cur_px > vah_px
            else ("BELOW_VAL" if cur_px < val_px else ("INSIDE_VA_UPPER" if va_pos >= 50.0 else "INSIDE_VA_LOWER"))
        ),
    }


def compute_horizon_range_dict(cur_px: float, low_px: float, high_px: float, suffix: str) -> Dict[str, Any]:
    """Build low/high range geometry (`dd_<sfx>_high_pct`, `bounce_<sfx>_low_pct`, `span_<sfx>_pct`, `pos_<sfx>_pct`)."""
    out: Dict[str, Any] = {f"low_{suffix}": low_px, f"high_{suffix}": high_px}
    if high_px > 0:
        out[f"dd_{suffix}_high_pct"] = pct_delta(cur_px, high_px)
    if low_px > 0:
        out[f"bounce_{suffix}_low_pct"] = pct_delta(cur_px, low_px)
        out[f"span_{suffix}_pct"] = pct_delta(high_px, low_px)
    if high_px > low_px > 0:
        out[f"pos_{suffix}_pct"] = range_position_pct(cur_px, low_px, high_px)
    return out


def compute_volume_profile_metrics(
    series: list,
    *,
    num_bins: int = 24,
    value_area_pct: float = 0.70,
    decimals: Optional[int] = None,
    precleaned_ts_px: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Compute realized session Volume Profile (Point of Control POC, Value Area High VAH, Value Area Low VAL, symmetric % distances, VA position, and regime)."""
    if not isinstance(series, list) or len(series) < 4:
        return {}
    try:
        has_vol_or_scalar = (
            False
            if (precleaned_ts_px is not None and _TWO_COL_TS_PX_IDS.get(id(precleaned_ts_px)) is precleaned_ts_px)
            else any(not isinstance(r, (list, tuple)) or len(r) != 2 for r in series)
        )
        if not has_vol_or_scalar:
            ts_px_pts = precleaned_ts_px if precleaned_ts_px is not None else _extract_sorted_ts_px(series)
            n_pts = len(ts_px_pts)
            if n_pts < 4:
                return {}
            has_real_vol = False
            if n_pts > 4000:
                step = max(1, n_pts // 2000)
                samples = [_tpo_sample_at(ts_px_pts, idx) for idx in range(0, n_pts, step)]
                if samples[-1][0] != ts_px_pts[n_pts - 1][1]:
                    samples.append(_tpo_sample_at(ts_px_pts, n_pts - 1))
            else:
                samples = [_tpo_sample_at(ts_px_pts, idx) for idx in range(n_pts)]
        else:
            timed_rows: list[Tuple[float, float, float]] = []
            scalar_rows: list[float] = []
            for row in series:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    if (pt := parse_ts_px_row(row)) is not None:
                        timed_rows.append((pt[0], pt[1], safe_pos_float(row[2], 0.0) if len(row) >= 3 else 0.0))
                elif not isinstance(row, (list, tuple)) and (px_f := safe_pos_float(row, 0.0)) > 0.0:
                    scalar_rows.append(px_f)
            if timed_rows:
                s_rows = sorted(timed_rows, key=lambda x: x[0])
                if any(v > 0.0 for _, _, v in s_rows) and len(s_rows) >= 3:
                    clean_rows = [s_rows[0]]
                    for idx in range(1, len(s_rows) - 1):
                        _, p_c, v_c = s_rows[idx]
                        if v_c <= 0.0:
                            p_prv, p_nxt = clean_rows[-1][1], s_rows[idx + 1][1]
                            if (
                                min(p_prv, p_nxt, p_c) > 0.0
                                and abs(p_c - p_prv) / p_prv > 0.04
                                and abs(p_c - p_nxt) / p_nxt > 0.04
                                and abs(p_nxt - p_prv) / min(p_prv, p_nxt) <= 0.035
                            ):
                                continue
                        clean_rows.append(s_rows[idx])
                    clean_rows.append(s_rows[-1])
                    s_rows = clean_rows
                raw_pairs = [(px_f, vol_w) for _, px_f, vol_w in s_rows]
            else:
                raw_pairs = [(px_f, 0.0) for px_f in scalar_rows]
            n_with_vol_col = sum(1 for r in series if isinstance(r, (list, tuple)) and len(r) >= 3)
            n_pos_vol = sum(1 for _, vol_w in raw_pairs if vol_w > 0.0)
            vol_cov_fresh = True
            if timed_rows and n_pos_vol > 0:
                last_pos_idx = max((i for i, (_, _, v) in enumerate(s_rows) if v > 0.0), default=-1)
                if (
                    0 <= last_pos_idx < len(s_rows) - 1
                    and (s_rows[-1][0] - s_rows[last_pos_idx][0]) > 6 * 3600 * 1000
                    and (len(s_rows) - 1 - last_pos_idx) > n_pos_vol * 2
                ):
                    vol_cov_fresh = False
            has_real_vol = bool(
                vol_cov_fresh and n_pos_vol >= max(4, min(len(raw_pairs), max(n_with_vol_col, 16)) // 4)
            )

            vol_cap: Optional[float] = None
            if has_real_vol:
                pos_vols = sorted(v for _, v in raw_pairs if v > 0.0)
                if len(pos_vols) >= 8:
                    vol_cap = pos_vols[len(pos_vols) // 2] * 8.0
            samples = []
            prev_px: Optional[float] = None
            for px_f, vol_w in raw_pairs:
                if not has_real_vol:
                    vol_w = 1.0 + min(5.0, (abs(px_f - prev_px) / prev_px) * 1000.0) if prev_px else 1.0
                elif vol_w <= 0.0:
                    vol_w = 0.0
                elif vol_cap is not None and vol_w > vol_cap:
                    vol_w = vol_cap
                prev_px = px_f
                samples.append((px_f, vol_w))

            if len(samples) < 4:
                return {}
            if len(samples) > 4000:
                downsampled = samples[:: max(1, len(samples) // 2000)]
                samples = downsampled + ([samples[-1]] if downsampled[-1][0] != samples[-1][0] else [])

        cur_px = samples[-1][0]
        low_px, high_px = min(p for p, _ in samples), max(p for p, _ in samples)
        n_bins = max(8, min(64, safe_pos_int(num_bins, default=24)))
        bin_w = (high_px - low_px) / (n_bins * 1.0) if high_px > low_px and low_px > 0.0 else 0.0
        if bin_w <= 0.0:
            return {}

        bin_vols, bin_px_sums, total_vol = [0.0] * n_bins, [0.0] * n_bins, 0.0
        for px_v, w_v in samples:
            b_idx = min(n_bins - 1, max(0, math.trunc((px_v - low_px) / bin_w)))
            bin_vols[b_idx] += w_v
            bin_px_sums[b_idx] += px_v * w_v
            total_vol += w_v
        if total_vol <= 0.0:
            return {}

        # 3-bin kernel smoothing to prevent POC oscillation on bimodal distributions
        smoothed_vols = [0.0] * n_bins
        for i in range(n_bins):
            if n_bins == 1:
                smoothed_vols[i] = bin_vols[i]
            elif i == 0:
                smoothed_vols[i] = 0.70 * bin_vols[0] + 0.30 * bin_vols[1]
            elif i == n_bins - 1:
                smoothed_vols[i] = 0.70 * bin_vols[n_bins - 1] + 0.30 * bin_vols[n_bins - 2]
            else:
                smoothed_vols[i] = 0.25 * bin_vols[i - 1] + 0.50 * bin_vols[i] + 0.25 * bin_vols[i + 1]

        poc_idx = max(range(n_bins), key=lambda i: (smoothed_vols[i], bin_vols[i]))
        k_num, k_den = 0.0, 0.0
        for j, kw in ((poc_idx - 1, 0.25), (poc_idx, 0.50), (poc_idx + 1, 0.25)):
            if 0 <= j < n_bins and bin_vols[j] > 0.0:
                bin_mean_px = bin_px_sums[j] / bin_vols[j]
                w_eff = kw * bin_vols[j]
                k_num += bin_mean_px * w_eff
                k_den += w_eff
        poc_raw = (
            (k_num / k_den)
            if k_den > 0.0
            else (bin_px_sums[poc_idx] / bin_vols[poc_idx] if bin_vols[poc_idx] > 0.0 else (low_px + (poc_idx + 0.5) * bin_w))
        )

        hvn2_px: Optional[float] = None
        lvn_px: Optional[float] = None
        profile_shape = "UNIMODAL"
        sep_cands: list[Tuple[int, int]] = []
        for i in range(n_bins):
            if abs(i - poc_idx) < 2 or bin_vols[i] <= 0.0:
                continue
            cand_px_i = bin_px_sums[i] / bin_vols[i]
            if poc_raw > 0.0 and abs(cand_px_i - poc_raw) / poc_raw < 0.005:
                continue
            lo_b, hi_b = min(poc_idx, i), max(poc_idx, i)
            valley_indices = list(range(lo_b + 1, hi_b))
            if not valley_indices:
                continue
            lvn_i = min(valley_indices, key=lambda k: (smoothed_vols[k], bin_vols[k]))
            if smoothed_vols[lvn_i] <= 0.90 * min(smoothed_vols[poc_idx], smoothed_vols[i]):
                sep_cands.append((i, lvn_i))
        if sep_cands:
            hvn2_idx, lvn_idx = max(sep_cands, key=lambda pair: (smoothed_vols[pair[0]], bin_vols[pair[0]]))
            if smoothed_vols[poc_idx] > 0.0 and smoothed_vols[hvn2_idx] >= 0.60 * smoothed_vols[poc_idx]:
                hvn2_raw = (
                    bin_px_sums[hvn2_idx] / bin_vols[hvn2_idx]
                    if bin_vols[hvn2_idx] > 0.0
                    else (low_px + (hvn2_idx + 0.5) * bin_w)
                )
                lvn_raw = (
                    bin_px_sums[lvn_idx] / bin_vols[lvn_idx]
                    if bin_vols[lvn_idx] > 0.0
                    else (low_px + (lvn_idx + 0.5) * bin_w)
                )
                eff_dec_tmp = _auto_px_decimals(cur_px, decimals)
                hvn2_px = round(hvn2_raw, eff_dec_tmp)
                lvn_px = round(lvn_raw, eff_dec_tmp)
                if smoothed_vols[lvn_idx] <= 0.90 * min(smoothed_vols[poc_idx], smoothed_vols[hvn2_idx]):
                    profile_shape = "BIMODAL"
                if smoothed_vols[hvn2_idx] >= 0.85 * smoothed_vols[poc_idx]:
                    ref_center = 0.5 * (cur_px + (0.5 * (low_px + high_px)))
                    if abs(ref_center - hvn2_raw) < abs(ref_center - poc_raw):
                        poc_raw, hvn2_raw = hvn2_raw, poc_raw
                        poc_idx, hvn2_idx = hvn2_idx, poc_idx
                        hvn2_px = round(hvn2_raw, eff_dec_tmp)
        va_pct = safe_pos_float(value_area_pct, default=0.70) or 0.70
        target_vol = total_vol * max(0.50, min(0.90, va_pct))
        acc_vol, val_idx, vah_idx = bin_vols[poc_idx], poc_idx, poc_idx

        while acc_vol < target_vol and (val_idx > 0 or vah_idx < n_bins - 1):
            down_vol = bin_vols[val_idx - 1] if val_idx > 0 else -1.0
            up_vol = bin_vols[vah_idx + 1] if vah_idx < n_bins - 1 else -1.0
            go_up = up_vol > down_vol or (up_vol == down_vol and (poc_idx - val_idx) > (vah_idx - poc_idx) and vah_idx < n_bins - 1)
            if go_up or val_idx == 0:
                vah_idx += 1
                acc_vol += up_vol
            else:
                val_idx -= 1
                acc_vol += down_vol

        eff_dec = _auto_px_decimals(cur_px, decimals)
        step_px = max(0.5 * bin_w, 10.0 ** (-eff_dec))
        poc_px = round(poc_raw, eff_dec)
        val_px = round(min(poc_raw, low_px + val_idx * bin_w), eff_dec)
        vah_px = round(max(poc_raw, low_px + (vah_idx + 1) * bin_w), eff_dec)
        if val_px >= poc_px and poc_raw > low_px:
            val_px = round(max(low_px, poc_raw - step_px), eff_dec)
        if vah_px <= poc_px and poc_raw < high_px:
            vah_px = round(min(high_px, poc_raw + step_px), eff_dec)
        if vah_px <= val_px:
            vah_px, val_px = round(high_px, eff_dec), round(low_px, eff_dec)
        if poc_px >= vah_px and vah_px > val_px:
            poc_px = round(max(val_px + 10.0 ** (-eff_dec), vah_px - step_px), eff_dec)
        elif poc_px <= val_px and vah_px > val_px:
            poc_px = round(min(vah_px - 10.0 ** (-eff_dec), val_px + step_px), eff_dec)
        out_vp: Dict[str, Any] = {
            "poc_px": poc_px,
            "vah_px": vah_px,
            "val_px": val_px,
            "vp_source": "REAL_VOL" if has_real_vol else "TPO",
            "profile_shape": profile_shape,
            **build_vp_position_dict(cur_px, poc_px, val_px, vah_px),
        }
        if hvn2_px is not None:
            out_vp["hvn2_px"] = hvn2_px
            out_vp["sec_poc_px"] = hvn2_px
        if lvn_px is not None:
            out_vp["lvn_px"] = lvn_px
        return out_vp
    except Exception:
        return {}


_ORIG_COMPUTE_VP_METRICS = compute_volume_profile_metrics


def compute_volume_rvol_metrics(series: list) -> Dict[str, float]:
    """Compute 15m/1h spot USD turnover and relative volume (RVOL vs 24h average) from 3-col `[ts_ms, px, vol]` bars."""
    if not isinstance(series, (list, tuple)) or len(series) < 3:
        return {}
    bars: List[Tuple[float, float, float]] = []
    for row in series:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        pt = parse_ts_px_row(row)
        vol_v = safe_nonneg_float(row[2], default=None)
        if pt is None or vol_v is None:
            continue
        bars.append((pt[0], pt[1], vol_v))
    if len(bars) < 3:
        return {}
    bars.sort(key=lambda b: b[0])
    pos_bars = [b for b in bars if b[2] > 0.0]
    if len(pos_bars) < 3:
        return {}
    series_end_ms = bars[-1][0]
    vol_now_ms = pos_bars[-1][0]
    if (series_end_ms - vol_now_ms) > 3600_000.0:
        return {}
    cutoff_24h = vol_now_ms - 86_400_000.0
    w24 = [(t, px * v) for t, px, v in bars if t >= cutoff_24h and v > 0.0]
    if len(w24) < 3:
        return {}
    span_raw_ms = w24[-1][0] - w24[0][0]
    if span_raw_ms < 5_400_000.0:  # Require >= 1.5h of volume history for a stable RVOL baseline
        return {}
    diffs = sorted(b[0] - a[0] for a, b in zip(w24, w24[1:]) if 0.0 < (b[0] - a[0]) <= 3600_000.0)
    step_ms = diffs[len(diffs) // 2] if diffs else 300_000.0
    eff_span_h = min(24.0, max(1.5, (span_raw_ms + step_ms) / 3_600_000.0))
    tot_usd = sum(u for _, u in w24)
    if tot_usd <= 0.0:
        return {}
    avg_1h_usd = tot_usd / eff_span_h
    avg_15m_usd = avg_1h_usd * 0.25
    # Include bars strictly inside (vol_now_ms - horizon, vol_now_ms]; when bar step >= horizon, use latest bar.
    cut_15m = vol_now_ms - max(900_000.0, step_ms * 0.9)
    vol_15m_usd = sum(u for t, u in w24 if t > cut_15m) if step_ms <= 900_000.0 else 0.0
    cut_1h = vol_now_ms - max(3_600_000.0, step_ms * 0.9)
    vol_1h_usd = sum(u for t, u in w24 if t > cut_1h)
    out: Dict[str, float] = {
        "spot_vol_24h_avg_15m_usd": round(avg_15m_usd, 0),
    }
    if vol_15m_usd > 0.0:
        out["spot_vol_15m_usd"] = round(vol_15m_usd, 0)
        if avg_15m_usd > 0.0:
            out["rvol_15m"] = round(vol_15m_usd / avg_15m_usd, 2)
    if vol_1h_usd > 0.0:
        out["spot_vol_1h_usd"] = round(vol_1h_usd, 0)
        if avg_1h_usd > 0.0:
            out["rvol_1h"] = round(vol_1h_usd / avg_1h_usd, 2)
    return out


def _resample_ohlc_bars(
    series: Any,
    bucket_ms: float,
    *,
    precleaned_ts_px: Optional[List[Tuple[float, float]]] = None,
) -> List[List[Any]]:
    """Aggregate sorted price or OHLCV rows into ``[bar_key, open, high, low, close, first_ts, last_ts, count]`` bars."""
    if bucket_ms <= 0.0:
        return []
    has_hl = isinstance(series, (list, tuple)) and any(
        isinstance(r, (list, tuple)) and len(r) >= 5 for r in series[:8]
    )
    bars: List[List[Any]] = []
    if not has_hl:
        pts = (
            precleaned_ts_px
            if precleaned_ts_px is not None
            else (_extract_sorted_ts_px(series) if isinstance(series, (list, tuple)) else [])
        )
        for ts_ms, px in pts:
            if px <= 0.0:
                continue
            b_key = math.floor(ts_ms / bucket_ms)
            if bars and bars[-1][0] == b_key:
                cur = bars[-1]
                if px > cur[2]:
                    cur[2] = px
                elif px < cur[3]:
                    cur[3] = px
                cur[4] = px
                cur[6] = ts_ms
                cur[7] += 1
            else:
                bars.append([b_key, px, px, px, px, ts_ms, ts_ms, 1])
        return bars

    parsed_hl: List[Tuple[float, float, float, float, float]] = []
    for row in series:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        pt = parse_ts_px_row(row)
        if pt is None:
            continue
        ts_ms, close_px = pt
        raw_hi = safe_pos_float(row[3], default=None) if len(row) >= 4 else None
        raw_lo = safe_pos_float(row[4], default=None) if len(row) >= 5 else None
        raw_op = safe_pos_float(row[5], default=None) if len(row) >= 6 else None
        hi_px = max(close_px, raw_hi) if raw_hi is not None else close_px
        lo_px = min(close_px, raw_lo) if raw_lo is not None else close_px
        op_px = raw_op if raw_op is not None else close_px
        parsed_hl.append((ts_ms, op_px, hi_px, lo_px, close_px))
    parsed_hl.sort(key=itemgetter(0))
    for ts_ms, op_px, hi_px, lo_px, close_px in parsed_hl:
        b_key = math.floor(ts_ms / bucket_ms)
        if bars and bars[-1][0] == b_key:
            cur = bars[-1]
            if hi_px > cur[2]:
                cur[2] = hi_px
            if lo_px < cur[3]:
                cur[3] = lo_px
            cur[4] = close_px
            cur[6] = ts_ms
            cur[7] += 1
        else:
            bars.append([b_key, op_px, hi_px, lo_px, close_px, ts_ms, ts_ms, 1])
    return bars


def _rollup_ohlc_bars(bars: List[List[Any]], div_factor: int) -> List[List[Any]]:
    """Roll up finer OHLC bars into coarser aligned bars (e.g. 15m -> 1h via factor 4, 1h -> 4h via factor 4)."""
    if not bars or div_factor <= 1:
        return [list(b) for b in bars]
    out: List[List[Any]] = []
    for b in bars:
        ck = b[0] // div_factor
        if out and out[-1][0] == ck:
            cur = out[-1]
            if b[2] > cur[2]:
                cur[2] = b[2]
            if b[3] < cur[3]:
                cur[3] = b[3]
            cur[4] = b[4]
            if b[6] > cur[6]:
                cur[6] = b[6]
            cur[7] += b[7]
        else:
            out.append([ck, b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    return out


def _merge_ohlc_bar_lists(older: List[List[Any]], newer: List[List[Any]]) -> List[List[Any]]:
    """Merge two sorted OHLC bar lists keyed by ``bar_key``, preserving full combined high/low range on overlap."""
    if not older:
        return [list(b) for b in newer]
    if not newer:
        return [list(b) for b in older]
    merged: Dict[int, List[Any]] = {b[0]: list(b) for b in older}
    for b in newer:
        bk = b[0]
        if (prev := merged.get(bk)) is None:
            merged[bk] = list(b)
        else:
            if b[5] < prev[5]:
                prev[1] = b[1]
                prev[5] = b[5]
            if b[2] > prev[2]:
                prev[2] = b[2]
            if b[3] < prev[3]:
                prev[3] = b[3]
            if b[6] >= prev[6]:
                prev[4] = b[4]
                prev[6] = b[6]
            prev[7] += b[7]
    return [merged[k] for k in sorted(merged)]


def _wilder_atr_pct(
    bars: List[List[Any]],
    bucket_ms: float,
    cur_px: float,
    *,
    period: int = 14,
    max_gap_buckets: int = 0,
) -> Optional[float]:
    """Compute Wilder RMA ATR as a percentage of ``cur_px`` from sorted OHLC bars."""
    if cur_px <= 0.0 or len(bars) < 6:
        return None
    eff_bars = bars
    if len(bars) >= 7:
        dense_prior = sum(1 for b in bars[:-1] if b[7] >= 2)
        if dense_prior >= max(2, (len(bars) - 1) // 2) and (bars[-1][6] - bars[-1][5]) < 0.35 * bucket_ms:
            eff_bars = bars[:-1]
    if len(eff_bars) < 6:
        return None
    trs: List[float] = []
    first_hl = max(0.0, eff_bars[0][2] - eff_bars[0][3])
    if first_hl > 0.0:
        trs.append(first_hl)
    for idx in range(1, len(eff_bars)):
        prev_c = eff_bars[idx - 1][4]
        hi_v, lo_v = eff_bars[idx][2], eff_bars[idx][3]
        hl_v = max(0.0, hi_v - lo_v)
        if max_gap_buckets > 0 and (eff_bars[idx][0] - eff_bars[idx - 1][0]) > max_gap_buckets:
            trs.append(hl_v)
        else:
            trs.append(max(hl_v, abs(hi_v - prev_c), abs(lo_v - prev_c)))
    if len(trs) < 6:
        return None
    if len(trs) >= period:
        atr_val = statistics.fmean(trs[:period])
        for tr_v in trs[period:]:
            atr_val = (atr_val * (period - 1) + tr_v) / period
    else:
        atr_val = statistics.fmean(trs)
    if atr_val <= 0.0 or not math.isfinite(atr_val):
        return None
    atr_pct = (atr_val / cur_px) * 100.0
    if not math.isfinite(atr_pct) or atr_pct <= 0.0:
        return None
    return (round(atr_pct, 3) if atr_pct < 0.10 else round(atr_pct, 2)) + 0.0


def build_compact_4h_ohlc_bars(
    archive_pts: list,
    sparse_7d_pts: list,
    watcher_7d_pts: list,
) -> List[List[Any]]:
    """Build at most ~45 compact 4h ``[ts_ms, close, 0.0, high, low]`` bars from 7d archive rows (0 extra I/O)."""
    src_pts: list = []
    if watcher_7d_pts and len(watcher_7d_pts) >= 2:
        src_pts = list(watcher_7d_pts)
    elif sparse_7d_pts and archive_pts:
        a_last = parse_ts_px_row(archive_pts[-1])
        s_last = parse_ts_px_row(sparse_7d_pts[-1])
        if a_last is not None and s_last is not None:
            src_pts = (
                prepend_older_non_diverged_points(archive_pts, sparse_7d_pts)
                if a_last[0] >= s_last[0]
                else prepend_older_non_diverged_points(sparse_7d_pts, archive_pts)
            )
        else:
            src_pts = list(archive_pts or sparse_7d_pts)
    elif archive_pts:
        src_pts = list(archive_pts)
    elif sparse_7d_pts:
        src_pts = list(sparse_7d_pts)
    if len(src_pts) < 2:
        return []
    bars_4h = _resample_ohlc_bars(src_pts, _MS_4H)
    if not bars_4h:
        return []
    cutoff_key = bars_4h[-1][0] - 45
    return [
        [round(b[6]), b[4], 0.0, b[2], b[3]]
        for b in bars_4h
        if b[0] >= cutoff_key
    ]


def compute_atr_multi_horizon(
    series_24h: list,
    *,
    series_7d_ts_px: Optional[list] = None,
    cur_px: Optional[float] = None,
    is_equity: bool = False,
    period: int = 14,
    precleaned_ts_px: Optional[List[Tuple[float, float]]] = None,
    precomputed_bars_15m: Optional[List[List[Any]]] = None,
) -> Dict[str, float]:
    """Compute pure Wilder RMA ATR(14) percentages across 15m, 1h, and 4h horizons."""
    bars_15m = (
        precomputed_bars_15m
        if precomputed_bars_15m is not None
        else _resample_ohlc_bars(series_24h, _MS_15M, precleaned_ts_px=precleaned_ts_px)
    )
    if not bars_15m and not series_7d_ts_px:
        return {}
    eff_px = safe_pos_float(cur_px, default=None)
    if eff_px is None and bars_15m:
        eff_px = safe_pos_float(bars_15m[-1][4], default=None)
    if eff_px is None or eff_px <= 0.0:
        return {}

    eff_period = safe_pos_int(period, default=14) or 14
    intraday_gap_cap = 4 if is_equity else 0
    out: Dict[str, float] = {}

    if (atr_15m := _wilder_atr_pct(bars_15m, _MS_15M, eff_px, period=eff_period, max_gap_buckets=intraday_gap_cap)) is not None:
        out["atr_15m_pct"] = atr_15m

    bars_1h = _rollup_ohlc_bars(bars_15m, 4)
    bars_4h = _rollup_ohlc_bars(bars_1h, 4)
    if isinstance(series_7d_ts_px, (list, tuple)) and len(series_7d_ts_px) >= 2:
        is_compact_4h = len(series_7d_ts_px) <= 60 and all(
            isinstance(r, (list, tuple)) and len(r) >= 5 for r in series_7d_ts_px[:4]
        )
        if not is_compact_4h:
            bars_1h_7d = _resample_ohlc_bars(series_7d_ts_px, _MS_1H)
            bars_1h = _merge_ohlc_bar_lists(bars_1h_7d, bars_1h)
            bars_4h_7d = _rollup_ohlc_bars(bars_1h_7d, 4)
            bars_4h = _merge_ohlc_bar_lists(bars_4h_7d, bars_4h)
        else:
            bars_4h_7d = _resample_ohlc_bars(series_7d_ts_px, _MS_4H)
            bars_4h = _merge_ohlc_bar_lists(bars_4h_7d, bars_4h)

    if (atr_1h := _wilder_atr_pct(bars_1h, _MS_1H, eff_px, period=eff_period, max_gap_buckets=intraday_gap_cap)) is not None:
        out["atr_1h_pct"] = atr_1h
    if (atr_4h := _wilder_atr_pct(bars_4h, _MS_4H, eff_px, period=eff_period, max_gap_buckets=0)) is not None:
        out["atr_4h_pct"] = atr_4h
    return out


def compute_session_vwap_metrics(
    series_with_vol: list,
    *,
    is_equity: bool = False,
    cur_px: Optional[float] = None,
    decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute volume-weighted Session VWAP anchored to 00:00 UTC (crypto) or 09:30 NY RTH (US equities)."""
    if not isinstance(series_with_vol, (list, tuple)) or len(series_with_vol) < 2:
        return {}
    pos_bars: List[Tuple[float, float, float]] = []
    max_any_ts = 0.0
    last_any_px: Optional[float] = None
    for row in series_with_vol:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        pt = parse_ts_px_row(row)
        if pt is None:
            continue
        ts_ms, close_px = pt
        if ts_ms >= max_any_ts:
            max_any_ts = ts_ms
            last_any_px = close_px
        if len(row) < 3:
            continue
        vol_v = safe_pos_float(row[2], default=None)
        if vol_v is None:
            continue
        tp_px = close_px
        if len(row) >= 5:
            hi_v = safe_pos_float(row[3], default=None)
            lo_v = safe_pos_float(row[4], default=None)
            if hi_v is not None and lo_v is not None and hi_v >= lo_v > 0.0:
                tp_px = (hi_v + lo_v + close_px) / 3.0
        pos_bars.append((ts_ms, tp_px, vol_v))
    if len(pos_bars) < 2:
        return {}
    pos_bars.sort(key=itemgetter(0))

    if is_equity:
        latest_ny_dt = datetime.fromtimestamp(pos_bars[-1][0] / 1000.0, tz=timezone.utc).astimezone(_NY_TZ)
        target_date = latest_ny_dt.date()
        rth_bars: List[Tuple[float, float, float]] = []
        same_day_bars: List[Tuple[float, float, float]] = []
        for ts_ms, tp_px, vol_v in pos_bars:
            ny_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(_NY_TZ)
            if ny_dt.date() != target_date:
                continue
            same_day_bars.append((ts_ms, tp_px, vol_v))
            mins_ny = ny_dt.hour * 60 + ny_dt.minute
            if 9 * 60 + 30 <= mins_ny <= 16 * 60 + 5:
                rth_bars.append((ts_ms, tp_px, vol_v))
        sess_bars = rth_bars if len(rth_bars) >= 2 else same_day_bars
        anchor_lbl = "rth"
    else:
        ref_ms = max_any_ts if max_any_ts > 0.0 else pos_bars[-1][0]
        day_start_ms = math.floor(ref_ms / _MS_1D) * _MS_1D
        sess_bars = [b for b in pos_bars if b[0] >= day_start_ms]
        anchor_lbl = "utc_00"

    if len(sess_bars) < 2:
        return {}
    tot_vol = sum(b[2] for b in sess_bars)
    if tot_vol <= 0.0:
        return {}
    vwap = sum(b[1] * b[2] for b in sess_bars) / tot_vol
    if vwap <= 0.0 or not math.isfinite(vwap):
        return {}
    eff_px = safe_pos_float(cur_px, default=None) or last_any_px or sess_bars[-1][1]
    if eff_px <= 0.0:
        return {}
    eff_dec = 4 if (is_equity and decimals is None) else _auto_px_decimals(vwap, decimals)
    out: Dict[str, Any] = {
        "session_vwap_px": round(vwap, eff_dec),
        "dist_vs_session_vwap_pct": pct_delta(eff_px, vwap),
        "session_vwap_anchor": anchor_lbl,
    }
    var_w = sum(b[2] * ((b[1] - vwap) ** 2) for b in sess_bars) / tot_vol
    sigma = math.sqrt(max(0.0, var_w))
    if sigma >= vwap * 1e-4 and math.isfinite(sigma):
        out["session_vwap_sigma_pct"] = round((sigma / vwap) * 100.0, 2) + 0.0
        out["session_vwap_z"] = round((eff_px - vwap) / sigma, 2) + 0.0
    return out


def compute_intraday_session_extremes(
    series_24h: list,
    *,
    is_equity: bool = False,
    decimals: Optional[int] = None,
    precleaned_ts_px: Optional[List[Tuple[float, float]]] = None,
    precomputed_bars_15m: Optional[List[List[Any]]] = None,
) -> Dict[str, Any]:
    """Compute current UTC-day Asia/London/NY session highs & lows for 24/7 crypto assets."""
    if is_equity:
        return {}
    bars_15m = (
        precomputed_bars_15m
        if precomputed_bars_15m is not None
        else _resample_ohlc_bars(series_24h, _MS_15M, precleaned_ts_px=precleaned_ts_px)
    )
    if not bars_15m:
        return {}
    ref_ms = bars_15m[-1][6]
    cur_px = bars_15m[-1][4]
    if ref_ms <= 0.0 or cur_px <= 0.0:
        return {}
    day_start_bucket = math.floor(ref_ms / _MS_1D) * 96
    day_start_ms = day_start_bucket * _MS_15M
    eff_dec = _auto_px_decimals(cur_px, decimals)
    sessions: Dict[str, List[float]] = {}
    for s_name, b_start_off, b_end_off in (
        ("asia", 0, 32),      # 00:00 - 08:00 UTC
        ("london", 28, 64),   # 07:00 - 16:00 UTC
        ("ny", 54, 84),       # 13:30 - 21:00 UTC
    ):
        sess_start_ms = day_start_ms + b_start_off * _MS_15M
        if ref_ms < sess_start_ms + _MS_15M:
            continue
        k_lo = day_start_bucket + b_start_off
        k_hi = day_start_bucket + b_end_off
        win_bars = [b for b in bars_15m if k_lo <= b[0] < k_hi]
        if not win_bars or sum(b[7] for b in win_bars) < 3:
            continue
        lo_s = min(b[3] for b in win_bars)
        hi_s = max(b[2] for b in win_bars)
        if hi_s > 0.0 and lo_s > 0.0:
            sessions[s_name] = [round(lo_s, eff_dec), round(hi_s, eff_dec)]
    return {"session_extremes": sessions} if sessions else {}


def format_atr_parts(entry: Any) -> List[str]:
    """Format compact multi-horizon ATR percentage tokens ``['15m=...', '1h=...', '4h=...']``."""
    if not isinstance(entry, dict):
        return []
    return [
        f"{lbl}={val:.2f}%" if val >= 0.10 else f"{lbl}={val:.3f}%"
        for lbl, key in (("15m", "atr_15m_pct"), ("1h", "atr_1h_pct"), ("4h", "atr_4h_pct"))
        if (val := safe_pos_float(entry.get(key), default=None)) is not None
    ]


def format_session_vwap_summary(entry: Any, *, cur_px: Optional[float] = None, fmt_px_fn: Any = None) -> str:
    """Format compact Session VWAP summary string (e.g. ``$62,110.00 (-0.28%, z=-0.62, utc_00)``)."""
    if not isinstance(entry, dict):
        return ""
    vwap_px = safe_pos_float(entry.get("session_vwap_px"), default=None)
    if vwap_px is None:
        return ""
    dist_pct = safe_float(entry.get("dist_vs_session_vwap_pct"), default=None)
    if dist_pct is None and cur_px is not None and cur_px > 0.0:
        dist_pct = pct_delta(cur_px, vwap_px)
    meta: List[str] = []
    if dist_pct is not None:
        meta.append(f"{dist_pct:+.2f}%")
    if (z_val := safe_float(entry.get("session_vwap_z"), default=None)) is not None:
        meta.append(f"z={z_val:+.2f}")
    if (sig_pct := safe_pos_float(entry.get("session_vwap_sigma_pct"), default=None)) is not None and len(meta) < 2:
        meta.append(f"1s={sig_pct:.2f}%")
    if anc := str(entry.get("session_vwap_anchor") or "").strip():
        meta.append(anc)
    px_s = fmt_px_fn(vwap_px) if callable(fmt_px_fn) else f"${vwap_px}"
    return f"{px_s} ({', '.join(meta)})" if meta else px_s


def format_session_extremes_parts(entry: Any) -> List[str]:
    """Format compact intraday Asia/London/NY session extreme ranges ``['asia=[lo, hi]', ...]``."""
    if not isinstance(entry, dict) or not isinstance(sess_ex := entry.get("session_extremes"), dict):
        return []
    return [
        f"{lbl}=[{vals[0]}, {vals[1]}]"
        for lbl, key in (("asia", "asia"), ("ldn", "london"), ("ny", "ny"))
        if isinstance(vals := sess_ex.get(key), (list, tuple)) and len(vals) == 2
    ]


def compute_24h_price_metrics(
    series: list,
    *,
    is_equity: bool = False,
    series_7d_ts_px: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute multi-horizon percentage changes (15m, 30m, 1h, 3h), 24h range, ATR, Session VWAP, and session extremes."""
    if not isinstance(series, list) or len(series) < 2:
        return {}
    try:
        clean_series = _extract_sorted_ts_px(series)
        n_clean = len(clean_series)
        if n_clean < 2:
            return {}
        now_ms, cur_px = clean_series[-1]
        span_ms = now_ms - clean_series[0][0]
        low_24h_raw = high_24h_raw = clean_series[0][1]
        for _, px_v in clean_series:
            if px_v < low_24h_raw:
                low_24h_raw = px_v
            elif px_v > high_24h_raw:
                high_24h_raw = px_v
        out = compute_horizon_range_dict(
            cur_px,
            round(low_24h_raw, 6),
            round(high_24h_raw, 6),
            "24h",
        )
        seen_sample_tms: set[float] = set()
        dedup_sparse_samples = n_clean >= 12
        for key, mins, min_span_ms, max_gap_mins in (
            ("chg_1m_pct", 1.0, 60000.0, 1.999), ("chg_3m_pct", 3.0, 180000.0, 1.999), ("chg_5m_pct", 5.0, 300000.0, 5.2),
            ("chg_15m_pct", 15.0, 0.0, 14.99 if dedup_sparse_samples else 0.0),
            ("chg_30m_pct", 30.0, 0.0, 29.99 if dedup_sparse_samples else 0.0),
            ("chg_1h_pct", 60.0, 0.0, 119.99 if dedup_sparse_samples else 0.0),
            ("chg_3h_pct", 180.0, 0.0, 179.99 if dedup_sparse_samples else 0.0),
            ("chg_6h_pct", 360.0, 3600000.0 * 4, 0.0),
            ("chg_12h_pct", 720.0, 3600000.0 * 9, 0.0), ("chg_24h_pct", 1440.0, 3600000.0 * 18, 0.0),
        ):
            target_ms = now_ms - mins * 60000.0
            ref_idx = bisect.bisect_right(clean_series, target_ms, key=_PT_TS_KEY)
            tms, p_ref = clean_series[ref_idx - 1] if ref_idx > 0 else (None, clean_series[0][1])
            valid_back = p_ref > 0 and span_ms >= min_span_ms and (
                max_gap_mins <= 0.0 or (tms is not None and tms >= now_ms - (mins + max_gap_mins) * 60000.0)
            )
            if valid_back and dedup_sparse_samples and tms is not None and mins < 1440.0 and tms in seen_sample_tms:
                valid_back = False
            if not valid_back and 15.0 <= mins <= 180.0 and span_ms >= max(min_span_ms, mins * 60000.0):
                nxt_t = clean_series[ref_idx][0] if (tms is not None and ref_idx < n_clean) else None
                has_overnight_gap = (
                    tms is not None
                    and nxt_t is not None
                    and tms <= target_ms < nxt_t
                    and (nxt_t - tms) > 4.0 * 3600000.0
                )
                if mins <= 30.0 or has_overnight_gap:
                    max_fwd_ms = now_ms - (30.0 if mins == 60.0 else mins * 0.75) * 60000.0
                    fwd_pt: Optional[Tuple[float, float]] = None
                    for fwd_idx in range(ref_idx, n_clean):
                        cand_pt = clean_series[fwd_idx]
                        if cand_pt[0] > max_fwd_ms:
                            break
                        if cand_pt[0] not in seen_sample_tms:
                            fwd_pt = cand_pt
                            break
                    if fwd_pt is not None and fwd_pt[1] > 0:
                        tms, p_ref, valid_back = fwd_pt[0], fwd_pt[1], True
                    elif has_overnight_gap and p_ref > 0 and tms not in seen_sample_tms:
                        valid_back = True
            if valid_back:
                if tms is not None:
                    seen_sample_tms.add(tms)
                out[key] = pct_delta(cur_px, p_ref)

        idx_15m = bisect.bisect_left(clean_series, now_ms - 15.0 * 60000.0, key=_PT_TS_KEY)
        recent_15m = [clean_series[k][1] for k in range(idx_15m, n_clean)]
        if len(recent_15m) >= 2:
            eff_dec_15m = _auto_px_decimals(cur_px)
            tape_pts: list[str] = []
            tape_vals: list[float] = []
            seen_tape_tms: set[float] = set()
            for m_off, lbl in ((15.0, "-15m"), (10.0, "-10m"), (5.0, "-5m"), (3.0, "-3m"), (1.0, "-1m"), (0.0, "now")):
                if m_off == 0.0:
                    tape_pts.append(f"now:{f'{round(cur_px, eff_dec_15m):.{eff_dec_15m}f}'.rstrip('0').rstrip('.')}")
                    tape_vals.append(cur_px)
                    continue
                if span_ms < m_off * 60000.0:
                    continue
                max_tape_gap = 5.2 if m_off == 5.0 else (1.999 if m_off in (1.0, 3.0) else max(3.0, m_off * 0.5))
                tap_idx = bisect.bisect_right(clean_series, now_ms - m_off * 60000.0, key=_PT_TS_KEY)
                t_tap, p_tap = clean_series[tap_idx - 1] if tap_idx > 0 else (None, clean_series[0][1])
                if t_tap is None or t_tap < now_ms - (m_off + max_tape_gap) * 60000.0 or t_tap in seen_tape_tms:
                    continue
                seen_tape_tms.add(t_tap)
                tape_pts.append(f"{lbl}:{f'{round(p_tap, eff_dec_15m):.{eff_dec_15m}f}'.rstrip('0').rstrip('.')}")
                tape_vals.append(p_tap)
            if len(recent_15m) >= 3 or len(tape_pts) >= 3:
                all_15m_vals = recent_15m + tape_vals if len(tape_pts) >= 3 else recent_15m
                l15, h15 = round(min(all_15m_vals), eff_dec_15m), round(max(all_15m_vals), eff_dec_15m)
                out["low_15m"], out["high_15m"] = l15, h15
                if h15 > l15 > 0:
                    out["pos_15m_pct"] = range_position_pct(cur_px, l15, h15)
                    out["span_15m_pct"] = pct_delta(h15, l15)
            if len(tape_pts) >= 3:
                out["recent_tape_15m"] = tape_pts
        if span_ms >= 3600000.0 * 2.0:
            idx_6h = bisect.bisect_left(clean_series, now_ms - 360.0 * 60000.0, key=_PT_TS_KEY)
            idx_1h = bisect.bisect_left(clean_series, now_ms - 60.0 * 60000.0, key=_PT_TS_KEY, lo=idx_6h)
            if (idx_1h - idx_6h) >= 2 and (n_clean - idx_1h) >= 2:
                prior_pts = [clean_series[k][1] for k in range(idx_6h, idx_1h)]
                recent_pts = [clean_series[k][1] for k in range(idx_1h, n_clean)]
                high_6h = max(max(prior_pts), max(recent_pts))
                prior_low, recent_low = min(prior_pts), min(recent_pts)
                low_6h = min(prior_low, recent_low)
                dd_low_6h = ((low_6h / high_6h) - 1.0) * 100.0 if high_6h > 0.0 else 0.0
                pos_6h = ((cur_px - low_6h) / (high_6h - low_6h)) if high_6h > low_6h else 1.0
                dd_24h = safe_float(out.get("dd_24h_high_pct"), 0.0) or 0.0
                pos_24h = safe_float(out.get("pos_24h_pct"), 100.0)
                if pos_24h is None:
                    pos_24h = 100.0
                has_meaningful_bottom_dd = dd_low_6h <= -1.45 or (
                    dd_low_6h <= -0.85 and dd_24h <= -1.45 and pos_24h <= 35.0
                )
                if has_meaningful_bottom_dd and pos_6h <= 0.70 and prior_low > 0.0 and recent_low > 0.0:
                    eff_dec = _auto_px_decimals(cur_px)
                    if recent_low < prior_low and cur_px > prior_low:
                        out["bottom_structure_6h"] = "LIQUIDITY_SWEEP_RECLAIM"
                        out["reclaim_above_prior_low_pct"] = pct_delta(cur_px, prior_low)
                    elif (
                        recent_low >= prior_low
                        and ((recent_low / prior_low) - 1.0) * 100.0 <= 4.5
                        and (
                            cur_px > recent_low * 1.002
                            or (recent_low > prior_low * 1.002 and cur_px >= recent_low)
                            or (recent_low > prior_low and cur_px > recent_low * 1.0005 and cur_px > prior_low * 1.001)
                        )
                    ):
                        out["bottom_structure_6h"] = "HIGHER_LOW_DOUBLE_BOTTOM"
                        out["higher_low_vs_prior_pct"] = pct_delta(recent_low, prior_low)
                    elif (
                        (recent_low <= prior_low and cur_px <= prior_low and ((cur_px / recent_low) - 1.0) * 100.0 <= 2.5)
                        or (
                            recent_low >= prior_low
                            and ((recent_low / prior_low) - 1.0) * 100.0 <= 0.20
                            and ((cur_px / prior_low) - 1.0) * 100.0 <= 0.25
                        )
                    ):
                        out["bottom_structure_6h"] = "FRESH_6H_LOW_TEST"
                    if "bottom_structure_6h" in out:
                        out["prior_6h_low_px"], out["recent_1h_low_px"] = round(prior_low, eff_dec), round(recent_low, eff_dec)
        vp_metrics = (
            compute_volume_profile_metrics(series, precleaned_ts_px=clean_series)
            if compute_volume_profile_metrics is _ORIG_COMPUTE_VP_METRICS
            else compute_volume_profile_metrics(series)
        )
        out.update(vp_metrics)
        if vol_rvol := compute_volume_rvol_metrics(series):
            out.update(vol_rvol)
        bars_15m = _resample_ohlc_bars(series, _MS_15M, precleaned_ts_px=clean_series)
        if atr_m := compute_atr_multi_horizon(
            series,
            series_7d_ts_px=series_7d_ts_px,
            cur_px=cur_px,
            is_equity=is_equity,
            precleaned_ts_px=clean_series,
            precomputed_bars_15m=bars_15m,
        ):
            out.update(atr_m)
        if vwap_m := compute_session_vwap_metrics(series, is_equity=is_equity, cur_px=cur_px):
            out.update(vwap_m)
        if sess_m := compute_intraday_session_extremes(
            series,
            is_equity=is_equity,
            precleaned_ts_px=clean_series,
            precomputed_bars_15m=bars_15m,
        ):
            out.update(sess_m)
        return out
    except Exception:
        return {}


def compute_cross_btc_metrics(alt_series: Any, btc_series: Any) -> Dict[str, Any]:
    """Compute 24h ALT/BTC ratio range position, drawdown/bounce, and realized beta/correlation vs BTC."""
    if not isinstance(alt_series, list) or not isinstance(btc_series, list) or len(alt_series) < 4 or len(btc_series) < 4:
        return {}
    try:
        def _clean_buckets(series: list) -> Dict[int, float]:
            mkey = _series_memo_key(series)
            if mkey is not None and (cached_b := _BUCKET_5M_MEMO.get(mkey)) is not None:
                return cached_b[1]
            pts = _extract_sorted_ts_px(series)
            bmap = {round(pt[0] / 300000.0): pt[1] for pt in pts}
            if mkey is not None:
                lru_dict_put(_BUCKET_5M_MEMO, mkey, (_series_memo_anchor(series), bmap), 16)
            return bmap

        a_map, b_map = _clean_buckets(alt_series), _clean_buckets(btc_series)
        if len(a_map) < 4 or len(b_map) < 4:
            return {}

        aligned = [
            (bk, a_map[bk], b_px)
            for bk in sorted(a_map.keys())
            if (b_px := b_map.get(bk) or b_map.get(bk - 1) or b_map.get(bk + 1)) is not None and b_px > 0
        ]
        ratios = [a_px / b_px for _, a_px, b_px in aligned if b_px > 0]
        if len(aligned) < 4 or len(ratios) < 4:
            return {}

        r_cur, r_low, r_high = ratios[-1], min(ratios), max(ratios)
        out: Dict[str, Any] = {}
        if r_high > 0:
            out["alt_btc_dd_24h_high_pct"] = pct_delta(r_cur, r_high) + 0.0
        if r_low > 0:
            out["alt_btc_bounce_24h_low_pct"] = pct_delta(r_cur, r_low) + 0.0
        if r_high > r_low > 0:
            out["alt_btc_pos_24h_pct"] = range_position_pct(r_cur, r_low, r_high) + 0.0

        def _step_returns(s: int) -> Tuple[list[float], list[float]]:
            ra_list, rb_list = [], []
            for (bk0, a0, b0), (bk1, a1, b1) in zip(aligned, aligned[s:]):
                if 0 < (bk1 - bk0) <= 48 and a0 > 0 and b0 > 0:
                    ra_list.append(a1 / a0 - 1.0)
                    rb_list.append(b1 / b0 - 1.0)
            return ra_list, rb_list

        step = 3 if len(aligned) >= 18 else 1
        rets_a, rets_b = _step_returns(step)
        if len(rets_b) < 6 and step > 1:
            rets_a, rets_b = _step_returns(1)
        if len(rets_b) >= 6:
            mean_a, mean_b = statistics.fmean(rets_a), statistics.fmean(rets_b)
            var_a, var_b = statistics.pvariance(rets_a, mu=mean_a), statistics.pvariance(rets_b, mu=mean_b)
            cov = statistics.fmean((ra - mean_a) * (rb - mean_b) for ra, rb in zip(rets_a, rets_b))
            if var_b > 1e-12:
                out["beta_btc_24h"] = round(cov / var_b, 2) + 0.0
                if var_a > 1e-12:
                    out["corr_btc_24h"] = round(max(-1.0, min(1.0, cov / math.sqrt(var_a * var_b))), 2) + 0.0
        return out
    except Exception:
        return {}


def _extract_ordered_prices(series_or_vals: Any) -> list[float]:
    if not isinstance(series_or_vals, (list, tuple)) or len(series_or_vals) < 2:
        return []
    timed: list[Tuple[float, float]] = []
    scalars: list[float] = []
    for item in series_or_vals:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            if (pt := parse_ts_px_row(item)) is not None:
                timed.append(pt)
            elif (v := safe_pos_float(item[1], default=0.0) or 0.0) > 0.0:
                scalars.append(v)
        elif not isinstance(item, (list, tuple)) and (v := safe_pos_float(item, default=0.0) or 0.0) > 0.0:
            scalars.append(v)
    if timed and not scalars:
        return [px for _, px in sorted(timed, key=lambda x: x[0])]
    if scalars and not timed:
        return scalars
    return [
        v for item in series_or_vals
        if (v := safe_pos_float(item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item, default=0.0) or 0.0) > 0.0
    ]


def compute_swing_wave_metrics(
    series_or_vals: list[Any], *, reversal_pct: float = 3.0, decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract raw ZigZag swing waves (>=3% reversals), active swing low/high, and rally/pullback % from a price series."""
    if not isinstance(series_or_vals, (list, tuple)) or len(series_or_vals) < 2:
        return {}
    vals = _extract_ordered_prices(series_or_vals)
    if len(vals) < 2:
        return {}

    rev_pct = safe_pos_float(reversal_pct, default=3.0) or 3.0
    cur_v, thr = vals[-1], max(0.5, rev_pct) / 100.0
    pivots: list[Tuple[str, int, float]] = []
    direction, hi_px, hi_idx, lo_px, lo_idx = 0, vals[0], 0, vals[0], 0

    for idx, v in enumerate(vals[1:], start=1):
        if direction >= 0 and (v > hi_px if direction == 0 else v >= hi_px):
            hi_px, hi_idx = v, idx
        if direction <= 0 and (v < lo_px if direction == 0 else v <= lo_px):
            lo_px, lo_idx = v, idx
        if direction <= 0 and lo_px > 0 and (v - lo_px) / lo_px >= thr and lo_idx < idx:
            pivots.append(("LOW", lo_idx, lo_px))
            direction, hi_px, hi_idx = 1, v, idx
        elif direction >= 0 and hi_px > 0 and (hi_px - v) / hi_px >= thr and hi_idx < idx:
            pivots.append(("HIGH", hi_idx, hi_px))
            direction, lo_px, lo_idx = -1, v, idx

    chain: list[Tuple[str, int, float]] = list(pivots)
    if direction == 1 and chain and chain[-1][2] > 0 and (hi_px - chain[-1][2]) / chain[-1][2] >= thr:
        chain.append(("HIGH", hi_idx, hi_px))
    elif direction == -1 and chain and chain[-1][2] > 0 and (chain[-1][2] - lo_px) / chain[-1][2] >= thr:
        chain.append(("LOW", lo_idx, lo_px))

    swing_legs_pct = [pct_delta(p1[2], p0[2]) for p0, p1 in zip(chain, chain[1:]) if p0[2] > 0]
    major_low_px: Optional[float] = None
    if direction == 1 and pivots:
        swing_low_px, swing_high_px = pivots[-1][2], max(hi_px, cur_v)
        if (
            len(pivots) >= 2
            and pivots[-2][0] == "HIGH"
            and pivots[-2][2] > swing_high_px
            and abs(cur_v - swing_high_px) / max(cur_v, 1e-12) < 5e-4
        ):
            swing_high_px = pivots[-2][2]
        if len(pivots) >= 3 and pivots[-3][0] == "LOW" and pivots[-3][2] < swing_low_px:
            major_low_px = pivots[-3][2]
    elif direction == -1 and pivots:
        swing_high_px, swing_low_px = pivots[-1][2], min(lo_px, cur_v)
        if len(pivots) >= 2 and pivots[-2][0] == "LOW" and pivots[-2][2] < swing_low_px:
            if abs(cur_v - swing_low_px) / max(cur_v, 1e-12) < 5e-4:
                swing_low_px = pivots[-2][2]
                if len(pivots) >= 4 and pivots[-4][0] == "LOW" and pivots[-4][2] < swing_low_px:
                    major_low_px = pivots[-4][2]
            else:
                major_low_px = pivots[-2][2]
    else:
        swing_low_px, swing_high_px = min(vals), max(vals)

    rally_pct = pct_delta(cur_v, swing_low_px) if swing_low_px > 0 else 0.0
    pullback_pct = pct_delta(cur_v, swing_high_px) if swing_high_px > 0 else 0.0
    major_rally_pct = pct_delta(cur_v, major_low_px) if major_low_px and major_low_px > 0 else None
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
        out["avg_swing_amp_7d_pct"] = round(sum(abs(x) for x in swing_legs_pct) / len(swing_legs_pct), 2)
    return out


def compute_7d_range_metrics(vals: list[float], decimals: Optional[int] = None) -> Dict[str, Any]:
    """Compute 7-day range geometry (low, high, drawdown, bounce, span, position) and 3%-8% swing waves from a price series."""
    if not isinstance(vals, (list, tuple)):
        return {}
    clean_vals = _extract_ordered_prices(vals)
    if len(clean_vals) < 2:
        return {}
    eff_dec = decimals if decimals is not None else 6
    cur_v = clean_vals[-1]
    return {
        "chg_7d_pct": pct_delta(cur_v, clean_vals[0]),
        **compute_horizon_range_dict(cur_v, round(min(clean_vals), eff_dec), round(max(clean_vals), eff_dec), "7d"),
        **compute_swing_wave_metrics(clean_vals, decimals=eff_dec),
    }


_WINDOW_TREND_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    ("90", "slope_90s", "trend_90s", "growth_90s"),
    ("222", "slope_3_7m", "trend_3_7m", "growth_3_7m"),
    ("900", "slope_15m", "trend_15m", "growth_15m"),
    ("9000", "slope_2_5h", "trend_2_5h", "growth_2_5h"),
)


def _is_unseeded_single_sample_window(w_dict: Dict[str, Any]) -> bool:
    """Return True when a 24trend window has fewer than 2 samples and zero slope/growth."""
    return (
        (safe_pos_float(w_dict.get("n_samples"), default=2.0) or 2.0) < 2.0
        and abs(safe_float(w_dict.get("slope_full"), 0.0)) < 1e-12
        and abs(safe_float(w_dict.get("growth_coefficient"), 0.0)) < 1e-12
    )


def _resolve_window_metric_with_fallbacks(primary: Any, *fallbacks: Any) -> Any:
    """Return ``primary`` unless it is near-zero (``< 5e-8``), in which case return the first non-near-zero fallback."""
    if primary is None:
        return None
    p_f = safe_float(primary, default=None)
    if p_f is not None and abs(p_f) >= 5e-8:
        return primary if isinstance(primary, (int, float)) and not isinstance(primary, bool) else p_f
    for fb in fallbacks:
        fb_f = safe_float(fb, default=None)
        if fb_f is not None and abs(fb_f) >= 5e-8:
            return fb if isinstance(fb, (int, float)) and not isinstance(fb, bool) else fb_f
    return p_f if p_f is not None else 0.0


def is_unseeded_kalman_state(
    mapping: Any,
    *,
    treat_none_as_unseeded: bool = False,
) -> bool:
    """Return True when a Kalman state dictionary is unseeded or cold-initialized."""
    if not isinstance(mapping, dict):
        return treat_none_as_unseeded
    vel = safe_float(mapping.get("kalman_vel"), default=None)
    if vel is None:
        return treat_none_as_unseeded
    std = safe_float(mapping.get("kalman_vel_std"), default=None)
    return abs(vel) < 1e-9 or (abs(vel) < 1e-12 and (std is None or std >= 5.99))


def apply_short_trend_item(
    entry: Dict[str, Any],
    tinfo: Dict[str, Any],
    *,
    fallback_price: Any = None,
    require_any_slope: bool = False,
    allow_price_update: bool = True,
) -> bool:
    """Populate short/dynamic trend slopes (90s, 3.7m, 15m, 2.5h) on a symbol telemetry dict from `cache_short_trend.json`."""
    if not isinstance(entry, dict) or not isinstance(tinfo, dict):
        return False
    dyn = tinfo.get("dynamic", {}) if isinstance(tinfo.get("dynamic"), dict) else {}
    wins = {wk: (dyn.get(wk) if isinstance(dyn.get(wk), dict) else {}) for wk, *_ in _WINDOW_TREND_SPECS}
    if wins["222"].get("slope_full") is None:
        wins["222"] = tinfo if tinfo.get("slope_full") is not None else (wins["90"] if wins["90"].get("slope_full") is not None else wins["222"])
    if (
        require_any_slope
        and not any(x.get("slope_full") is not None for x in (wins["222"], wins["900"], wins["9000"], tinfo))
        and tinfo.get("vel_90s_pct_min") is None
    ):
        return False
    cur_px = safe_pos_float(fallback_price, default=None) or safe_pos_float(tinfo.get("current_price"), default=None)
    if allow_price_update and safe_float(entry.get("price"), 0.0) <= 0 and cur_px is not None:
        entry["price"] = cur_px
    if (st_ts := safe_pos_float(tinfo.get("ts"), default=None)) is not None:
        entry["_short_trend_ts"] = st_ts
    trend = tinfo.get("final_trend", entry.get("trend", 0))
    entry["trend"] = trend
    entry["growth_coefficient"] = tinfo.get("growth_coefficient", entry.get("growth_coefficient", 0.0))
    entry["slope_full"] = tinfo.get("slope_full", 0.0)
    entry["gradient_recent"] = tinfo.get("gradient_recent", 0.0)
    for k in ("gradient_recent_fast", "epsilon", "vol_1h_pct"):
        if not require_any_slope or tinfo.get(k) is not None:
            entry[k] = tinfo.get(k)
    _backfill_kinematic_keys(entry, (tinfo, wins["222"], wins["90"], wins["900"]))
    t_kv = safe_float(tinfo.get("kalman_vel"), default=None)
    t_ks = safe_pos_float(tinfo.get("kalman_vel_std"), default=None)
    if t_kv is not None and not (abs(t_kv) < 1e-12 and (t_ks is None or t_ks >= 5.99)):
        if entry.get("kalman_vel") is None:
            entry["kalman_vel"] = t_kv
        if entry.get("kalman_vel_std") is None and t_ks is not None:
            entry["kalman_vel_std"] = t_ks
        if entry.get("kalman_trend") is None and tinfo.get("kalman_trend") is not None:
            entry["kalman_trend"] = tinfo.get("kalman_trend")
    if abs(safe_float(entry.get("gradient_recent"), 0.0)) < 5e-8:
        for w_fb in (wins["222"], wins["90"], wins["900"]):
            if (gr_fb := safe_float(w_fb.get("gradient_recent"), default=None)) is not None and abs(gr_fb) >= 5e-8:
                entry["gradient_recent"] = gr_fb
                break
    if entry.get("gradient_recent_fast") is not None and abs(safe_float(entry.get("gradient_recent_fast"), 0.0)) < 5e-8:
        for w_fb in (wins["90"], wins["222"]):
            gf_fb = next(
                (
                    v
                    for gk in ("gradient_recent_fast", "gradient_recent", "slope_full")
                    if (v := safe_float(w_fb.get(gk), default=None)) is not None and abs(v) >= 5e-8
                ),
                None,
            )
            if gf_fb is not None:
                entry["gradient_recent_fast"] = gf_fb
                break
    if entry.get("epsilon") is not None and (safe_pos_float(entry.get("epsilon"), default=0.0) or 0.0) <= 1e-12:
        for w_fb in (wins["222"], wins["90"], wins["900"], wins["9000"]):
            if (eps_fb := safe_pos_float(w_fb.get("epsilon"), default=None)) is not None and eps_fb > 1e-12:
                entry["epsilon"] = eps_fb
                break
    if not require_any_slope:
        entry["trend_fast"] = tinfo.get("trend_fast", trend)
    t_vol, t_re = safe_pos_float(tinfo.get("vol_1h_pct"), default=None), safe_pos_float(tinfo.get("adapt_reentry_pct"), default=None)
    if t_vol is None or (t_re is not None and abs(t_re - 2.0 * t_vol) <= 0.05):
        for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
            if tinfo.get(ak) is not None:
                entry[ak] = tinfo[ak]
    for wk, s_key, t_key, g_key in _WINDOW_TREND_SPECS:
        w_dict = wins[wk]
        is_90 = wk == "90"
        if not require_any_slope and _is_unseeded_single_sample_window(w_dict):
            if wk == "222" and abs(safe_float(entry.get("gradient_recent_fast"), 0.0)) < 1e-12:
                entry["gradient_recent_fast"] = None
            continue
        if w_dict.get("slope_full") is not None or (require_any_slope and not is_90):
            s_raw = w_dict.get("slope_full", 0.0) if (require_any_slope and not is_90) else w_dict.get("slope_full")
            entry[s_key] = _resolve_window_metric_with_fallbacks(
                s_raw, w_dict.get("gradient_recent"), w_dict.get("growth_coefficient"),
            )
            entry[t_key] = w_dict.get("final_trend", trend)
        if w_dict.get("growth_coefficient") is not None:
            entry[g_key] = _resolve_window_metric_with_fallbacks(
                w_dict.get("growth_coefficient"), w_dict.get("slope_full"), w_dict.get("gradient_recent"),
            )
    return True


def apply_live_24trend_windows(entry: Dict[str, Any], lt_win: Dict[str, Any], fallback_price: Any = None) -> None:
    """Overlay fresh 1-second sampled dynamic windows (90s, 3.7m, 15m, 2.5h) from `cachedb/24trend/live_<SYM>.json`."""
    if not isinstance(entry, dict) or not isinstance(lt_win, dict):
        return
    win_src = lt_win.get("windows") if isinstance(lt_win.get("windows"), dict) else lt_win
    wins = {wk: (win_src.get(wk) if isinstance(win_src.get(wk), dict) else {}) for wk, *_ in _WINDOW_TREND_SPECS}
    lt_ts = max(
        (safe_pos_float(w.get("ts"), default=0.0) or 0.0 for w in wins.values() if isinstance(w, dict)),
        default=0.0,
    )
    if lt_ts > 0.0:
        entry["_short_trend_ts"] = lt_ts
    w90, w222, w900 = wins["90"], wins["222"] or wins["90"], wins["900"]
    wins["222"] = w222
    live_px = (
        safe_pos_float(fallback_price, default=None)
        or safe_pos_float(w222.get("current_price"), default=None)
        or safe_pos_float(w900.get("current_price"), default=None)
    )
    if entry.get("price") is None and live_px is not None:
        entry["price"] = live_px
    entry.setdefault("trend", w222.get("final_trend", 0))
    entry.setdefault("growth_coefficient", w222.get("growth_coefficient", 0.0))
    if entry.get("gradient_recent") is None or abs(safe_float(entry.get("gradient_recent"), 0.0)) < 5e-8:
        if (gr_live := w222.get("gradient_recent") or w90.get("gradient_recent") or w900.get("gradient_recent")) is not None:
            entry["gradient_recent"] = gr_live
    if entry.get("gradient_recent_fast") is None or abs(safe_float(entry.get("gradient_recent_fast"), 0.0)) < 5e-8:
        if (gf_live := w90.get("gradient_recent_fast") or w222.get("gradient_recent_fast") or w90.get("gradient_recent") or w90.get("slope_full")) is not None:
            entry["gradient_recent_fast"] = gf_live
    if (safe_pos_float(entry.get("epsilon"), default=0.0) or 0.0) <= 1e-12:
        for w_fb in (w222, w90, w900, wins["9000"]):
            if (eps_live := safe_pos_float(w_fb.get("epsilon"), default=None)) is not None and eps_live > 1e-12:
                entry["epsilon"] = eps_live
                break
    for wk, s_key, t_key, g_key in _WINDOW_TREND_SPECS:
        w_dict = wins[wk]
        if not w_dict or _is_unseeded_single_sample_window(w_dict):
            continue
        s_raw = w_dict.get("slope_full", entry.get(s_key, 0.0)) if wk != "90" else w_dict.get("slope_full")
        s_val = _resolve_window_metric_with_fallbacks(
            s_raw, w_dict.get("gradient_recent"), w_dict.get("growth_coefficient"),
        )
        if wk != "90":
            entry[s_key] = s_val
            entry[t_key] = w_dict.get("final_trend", entry.get(t_key, 0))
        else:
            if s_val is not None:
                entry[s_key] = s_val
            if w_dict.get("final_trend") is not None:
                entry[t_key] = w_dict.get("final_trend")
        if w_dict.get("growth_coefficient") is not None:
            entry[g_key] = _resolve_window_metric_with_fallbacks(
                w_dict.get("growth_coefficient"), w_dict.get("slope_full"), w_dict.get("gradient_recent"),
            )
    _backfill_kinematic_keys(entry, (w222, w90, w900, lt_win))


def compute_kalman_from_series(series_24h: List[Any]) -> Optional[Dict[str, Any]]:
    """Derive Kalman velocity (% per min), std, and trend from a 24h price series when cache_kalman_trend lacks the symbol."""
    pts = [(t_ms / 1000.0, px_v) for t_ms, px_v in _extract_sorted_ts_px(series_24h)]
    if len(pts) < 5:
        return None
    cutoff_s = pts[-1][0] - 7200.0
    subsampled_rev: List[Tuple[float, float]] = [pts[-1]]
    next_allowed_ts = pts[-1][0] - 55.0
    for ts_s, px_v in reversed(pts[:-1]):
        if ts_s < cutoff_s:
            break
        if ts_s <= next_allowed_ts:
            subsampled_rev.append((ts_s, px_v))
            next_allowed_ts = ts_s - 55.0
    subsampled = list(reversed(subsampled_rev))
    if len(subsampled) < 5:
        subsampled = pts[-min(len(pts), 30) :]
    if len(subsampled) < 5:
        return None
    kalman_cls = KalmanTrend if KalmanTrend is not None else _DefaultKalmanTrend
    if kalman_cls is None:
        return None
    kf = kalman_cls(gap_reset_sec=7200.0)
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
    return {"kalman_vel": last_out.get("vel"), "kalman_vel_std": last_out.get("vel_std"), "kalman_trend": last_out.get("trend")}


def compute_short_slopes_from_series(series_24h: List[Any]) -> Dict[str, Any]:
    """Derive fallback multi-horizon slopes (90s, 3.7m, 15m, 2.5h) and growth coefficients from a 24h series."""
    pts = [(t_ms / 1000.0, px_v) for t_ms, px_v in _extract_sorted_ts_px(series_24h) if px_v > 0]
    if len(pts) < 3:
        return {}
    t_now, p_now = pts[-1]
    out: Dict[str, Any] = {}
    seen_slope_ref_t: set[float] = set()
    has_sub3m_cadence = len(pts) < 10 or sum(
        1 for a, b in zip(pts[:-2][-5:], pts[1:-1][-5:]) if 0.0 < (b[0] - a[0]) < 180.0
    ) >= 2
    for win_sec, win_min, s_key, t_key, g_key in (
        (90.0, 1.5, "slope_90s", "trend_90s", "growth_90s"),
        (222.0, 3.7, "slope_3_7m", "trend_3_7m", "growth_3_7m"),
        (900.0, 15.0, "slope_15m", "trend_15m", "growth_15m"),
        (9000.0, 150.0, "slope_2_5h", "trend_2_5h", "growth_2_5h"),
    ):
        if win_sec <= 90.0 and not has_sub3m_cadence:
            continue
        target_t = t_now - win_sec
        if len(pts) >= 10:
            min_age_sec = 0.0 if win_sec <= 90.0 else (30.0 if (win_sec <= 222.0 and not has_sub3m_cadence) else win_sec * 0.5)
            max_age_sec = 150.0 if win_sec <= 90.0 else (win_sec * 4.0 if win_sec < 9000.0 else math.inf)
            cands = [
                pt for pt in pts[:-1]
                if pt[0] not in seen_slope_ref_t and min_age_sec <= (t_now - pt[0]) <= max_age_sec
            ]
        else:
            cands = pts[:-1]
        if not cands:
            continue
        at_or_before = [pt for pt in cands if pt[0] <= target_t] if win_sec >= 900.0 else []
        ref_t, ref_p = (
            max(at_or_before, key=lambda item: item[0])
            if at_or_before
            else min(cands, key=lambda item: abs(item[0] - target_t))
        )
        seen_slope_ref_t.add(ref_t)
        dt_min = max(win_min, (t_now - ref_t) / 60.0)
        if ref_p > 0 and dt_min > 0:
            raw_s = ((p_now - ref_p) / ref_p) / dt_min
            slope_v = _round_small_slope(raw_s)
            half_t = t_now - 0.5 * win_sec
            ref_t_h, ref_p_h = min(pts[:-1], key=lambda item: abs(item[0] - half_t))
            dt_min_h = max(0.5 * win_min, (t_now - ref_t_h) / 60.0)
            slope_half = (((p_now - ref_p_h) / ref_p_h) / dt_min_h) if (ref_p_h > 0 and dt_min_h > 0) else slope_v
            raw_g = 0.35 * slope_v + 0.65 * slope_half
            g_v = _round_small_slope(raw_g)
            out[s_key] = slope_v
            if (ref_t_h != ref_t or win_sec >= 9000.0 or len(pts) < 10) and (
                has_sub3m_cadence or (raw_g * raw_s > 0.0) or (abs(raw_s) <= 1e-12 and abs(raw_g) <= 1e-12)
            ):
                out[g_key] = g_v
            eff_g_for_trend = out.get(g_key, slope_v)
            win_move_bps = eff_g_for_trend * win_min * 1000.0
            out[t_key] = 1 if win_move_bps >= 0.25 else (-1 if win_move_bps <= -0.25 else 0)
    if "slope_3_7m" in out:
        out["gradient_recent_fast"] = out["slope_3_7m"]
    return out


def _linear_ols_slope(pts_tau_y: List[Tuple[float, float]]) -> Optional[float]:
    m = len(pts_tau_y)
    if m < 2:
        return None
    mt = sum(t for t, _ in pts_tau_y) / m
    my = sum(y for _, y in pts_tau_y) / m
    denom = sum((t - mt) ** 2 for t, _ in pts_tau_y)
    if denom <= 1e-12:
        return None
    return sum((t - mt) * (y - my) for t, y in pts_tau_y) / denom


def _guard_quadratic_kinematic_overshoot(
    pts_tau_y: List[Tuple[float, float]],
    vel: float,
    acc: float,
) -> Tuple[float, float]:
    """Clamp right-endpoint quadratic boundary ringing when velocity flips sign without any actual tail reversal."""
    if len(pts_tau_y) < 3:
        return vel, acc
    span_min = pts_tau_y[-1][0] - pts_tau_y[0][0]
    if span_min <= 1e-6:
        return vel, acc
    v_full = _linear_ols_slope(pts_tau_y)
    if v_full is None or (vel * v_full) >= 0.0:
        return vel, acc
    tau_mid = 0.5 * (pts_tau_y[0][0] + pts_tau_y[-1][0])
    recent_pts = [pt for pt in pts_tau_y if pt[0] >= tau_mid - 1e-9]
    if len(recent_pts) < 2:
        recent_pts = pts_tau_y[len(pts_tau_y) // 2 :]
    v_rec = _linear_ols_slope(recent_pts)
    if v_rec is None or (vel * v_rec) > 0.0:
        return vel, acc
    tail_cut = pts_tau_y[-1][0] - span_min / 3.0
    tail_pts = [pt for pt in pts_tau_y if pt[0] >= tail_cut - 1e-9]
    if len(tail_pts) < 2:
        tail_pts = pts_tau_y[-2:]
    dy_tail = tail_pts[-1][1] - tail_pts[0][1]
    if (vel * dy_tail) > 0.0:
        return vel, acc
    early_pts = [pt for pt in pts_tau_y if pt[0] <= tau_mid + 1e-9]
    if len(early_pts) < 2:
        early_pts = pts_tau_y[: max(2, len(pts_tau_y) // 2 + 1)]
    v_early = _linear_ols_slope(early_pts)
    eff_acc = acc
    if v_early is not None:
        acc_fd = (v_rec - v_early) / (0.5 * span_min)
        if (acc * acc_fd) <= 0.0 or abs(acc) > abs(acc_fd):
            eff_acc = acc_fd
    return v_rec, eff_acc


def _fit_quadratic_kinematics(
    pts_ms_px: List[Tuple[float, float]],
    *,
    min_span_ms: float,
) -> Optional[Tuple[float, float]]:
    """Fit continuous-time quadratic OLS y(tau) = b0 + b1*tau + b2*tau^2 at tau=0 (%/min, %/min^2)."""
    if not isinstance(pts_ms_px, (list, tuple)) or len(pts_ms_px) < 3:
        return None
    req_span_ms = safe_nonneg_float(min_span_ms, default=0.0)
    clean_pts: List[Tuple[float, float]] = []
    is_sorted = True
    prev_t = -math.inf
    for item in pts_ms_px:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t_ms = safe_float(item[0], default=None)
        px_v = safe_pos_float(item[1], default=None)
        if t_ms is None or px_v is None:
            continue
        if t_ms < prev_t:
            is_sorted = False
        prev_t = t_ms
        clean_pts.append((t_ms, px_v))
    if len(clean_pts) < 3:
        return None
    if not is_sorted:
        clean_pts.sort(key=lambda pt: pt[0])
    dedup: List[Tuple[float, float]] = []
    for t_ms, px_v in clean_pts:
        if dedup and abs(t_ms - dedup[-1][0]) < 1.0:
            dedup[-1] = (t_ms, px_v)
        else:
            dedup.append((t_ms, px_v))
    if len(dedup) < 3:
        return None
    t_start, p0 = dedup[0]
    t_now, _ = dedup[-1]
    span_ms = t_now - t_start
    if p0 <= 0.0 or span_ms < req_span_ms:
        return None
    span_min = span_ms / 60000.0
    if not any(-0.92 * span_min <= ((t_m - t_now) / 60000.0) <= -0.08 * span_min for t_m, _ in dedup[1:-1]):
        return None
    n_f = len(dedup) * 1.0
    s1 = s2 = s3 = s4 = y0 = y1 = y2 = 0.0
    pts_tau_y: List[Tuple[float, float]] = []
    for t_m, px_v in dedup:
        tau = (t_m - t_now) / 60000.0
        tau2 = tau * tau
        yv = ((px_v - p0) / p0) * 100.0
        pts_tau_y.append((tau, yv))
        s1 += tau
        s2 += tau2
        s3 += tau2 * tau
        s4 += tau2 * tau2
        y0 += yv
        y1 += tau * yv
        y2 += tau2 * yv
    det = n_f * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2)
    if abs(det) <= 1e-12 or not math.isfinite(det):
        return None
    vel = (n_f * (y1 * s4 - s3 * y2) - y0 * (s1 * s4 - s2 * s3) + s2 * (s1 * y2 - y1 * s2)) / det
    beta2 = (n_f * (s2 * y2 - y1 * s3) - s1 * (s1 * y2 - y1 * s2) + y0 * (s1 * s3 - s2 * s2)) / det
    acc = 2.0 * beta2
    if not (math.isfinite(vel) and math.isfinite(acc)):
        return None
    vel, acc = _guard_quadratic_kinematic_overshoot(pts_tau_y, vel, acc)
    if not (math.isfinite(vel) and math.isfinite(acc)):
        return None
    if abs(vel) < 1e-6 and abs(acc) < 1e-6:
        return None
    return round(vel, 5), round(acc, 5)


def _backfill_kinematic_keys(entry: Dict[str, Any], sources: Tuple[Any, ...]) -> None:
    for kk_kin in SHORT_KINEMATIC_METRIC_KEYS:
        for w_src in sources:
            if isinstance(w_src, dict) and (kv_kin := safe_float(w_src.get(kk_kin), default=None)) is not None:
                entry[kk_kin] = round(kv_kin, 5)
                break


def format_slope_val(val: Any, decimals: int = 5, *, signed: bool = True) -> str:
    """Format a polynomial slope or growth value with scientific notation for small magnitudes."""
    vf = safe_float(val, 0.0)
    if abs(vf) < 5e-8:
        vf = 0.0
    elif 0.0 < abs(vf) < 1e-4:
        return f"{vf:+.2e}" if signed else f"{vf:.2e}"
    return f"{vf:+.{decimals}f}" if signed else f"{vf:.{decimals}f}"


def format_multi_window_slopes_core(entry: Any, *, fallback_full: bool = False) -> str:
    """Format multi-horizon slopes ``(90s, 3.7m, 15m, 2.5h)`` from ``entry``."""
    if not isinstance(entry, dict):
        return ""
    s90_v = safe_float(entry.get("slope_90s"), default=None)
    s90_pfx = f"90s={format_slope_val(s90_v, 5)}, " if s90_v is not None else ""
    slope_parts = [
        f"{s_lbl}={format_slope_val(s_val, 5)}"
        for s_lbl, s_key in (
            ("90s", "slope_90s"),
            ("3.7m", "slope_3_7m"),
            ("15m", "slope_15m"),
            ("2.5h", "slope_2_5h"),
        )
        if (s_val := safe_float(entry.get(s_key), default=None)) is not None
    ]
    if (fallback_full and not slope_parts) or (
        "slope_15m" in entry and ("slope_3_7m" in entry or "slope_2_5h" not in entry)
    ):
        return (
            f"{s90_pfx}"
            f"3.7m={format_slope_val(entry.get('slope_3_7m'), 5)}, "
            f"15m={format_slope_val(entry.get('slope_15m'), 5)}, "
            f"2.5h={format_slope_val(entry.get('slope_2_5h'), 5)}"
        )
    return ", ".join(slope_parts)


def format_kinematic_parts(entry: Any, *, compact: bool = False) -> List[str]:
    """Format active ``(90s, 5m, 15m)`` velocity/acceleration and ``kalman_z`` segments from ``entry``."""
    if not isinstance(entry, dict):
        return []
    parts: List[str] = []
    for k_lbl, v_key, a_key in KINEMATIC_HORIZON_SPECS:
        vk_v = safe_float(entry.get(v_key), default=None)
        ak_v = safe_float(entry.get(a_key), default=None)
        if vk_v is not None and ak_v is not None and (abs(vk_v) >= 1e-6 or abs(ak_v) >= 1e-6):
            if compact:
                parts.extend((f"vel_{k_lbl}={vk_v:+.4f}%/m", f"acc_{k_lbl}={ak_v:+.4f}%/m^2"))
            else:
                parts.append(f"{k_lbl}: vel={vk_v:+.4f}%/m, acc={ak_v:+.4f}%/m^2")
    if parts and (kz_v := resolve_kalman_vel_z(entry)) is not None and abs(kz_v) >= 0.01:
        parts.append(f"kalman_z={kz_v:+.2f}")
    return parts


def compute_series_kinematics(series_24h: List[Any]) -> Dict[str, float]:
    """Compute instantaneous velocity (%/min) and acceleration (%/min^2) across short horizons from a 24h series."""
    pts = [pt for pt in _extract_sorted_ts_px(series_24h) if pt[1] > 0.0]
    if len(pts) < 3:
        return {}
    now_ms = pts[-1][0]
    out: Dict[str, float] = {}
    for lookbacks_ms, min_span_ms, v_key, a_key, fallback_only in (
        ((90000.0, 180000.0), 45000.0, "vel_90s_pct_min", "accel_90s_pct_min2", False),
        ((300000.0, 360000.0), 120000.0, "vel_5m_pct_min", "accel_5m_pct_min2", False),
        ((1080000.0,), 480000.0, "vel_15m_pct_min", "accel_15m_pct_min2", True),
    ):
        if fallback_only and out:
            break
        for lookback_ms in lookbacks_ms:
            if (fit := _fit_quadratic_kinematics([pt for pt in pts if pt[0] >= now_ms - lookback_ms], min_span_ms=min_span_ms)) is not None:
                out[v_key], out[a_key] = fit
                break
    return out


def _extract_ohlc_series_points(
    ohlc_obj: Any,
    *,
    min_ts: float = 0.0,
    max_ts: float = math.inf,
    require_positive_volume: bool = False,
) -> List[List[Any]]:
    """Extract `[ts_ms, close, (vol), (high, low, (open))]` rows from an OHLC series within `[min_ts, max_ts]`."""
    vols = getattr(ohlc_obj, "volumes", ()) or ()
    has_vols = isinstance(vols, (list, tuple)) and bool(vols)
    highs = getattr(ohlc_obj, "highs", ()) or ()
    lows = getattr(ohlc_obj, "lows", ()) or ()
    opens = getattr(ohlc_obj, "opens", ()) or ()
    has_hl = isinstance(highs, (list, tuple)) and isinstance(lows, (list, tuple)) and bool(highs) and bool(lows)
    has_opens = isinstance(opens, (list, tuple)) and bool(opens)
    min_s = normalize_epoch_sec(min_ts, default=0.0, allow_zero=True) or 0.0
    max_s = (normalize_epoch_sec(max_ts, default=math.inf) or math.inf) if math.isfinite(max_ts) else math.inf
    pts: List[List[Any]] = []
    for idx, (ts, px) in enumerate(zip(getattr(ohlc_obj, "timestamps", ()) or (), getattr(ohlc_obj, "closes", ()) or ())):
        ts_s, px_f = normalize_epoch_sec(ts, default=0.0) or 0.0, safe_pos_float(px, 0.0) or 0.0
        if ts_s > 0.0 and min_s <= ts_s <= max_s and px_f > 0.0:
            v_f = safe_pos_float(vols[idx], 0.0) or 0.0 if (has_vols and idx < len(vols)) else 0.0
            if require_positive_volume and v_f <= 0.0:
                continue
            ts_ms_i = round(ts_s * 1000.0)
            hi_f = safe_pos_float(highs[idx], default=None) if (has_hl and idx < len(highs)) else None
            lo_f = safe_pos_float(lows[idx], default=None) if (has_hl and idx < len(lows)) else None
            if hi_f is not None and lo_f is not None and hi_f >= lo_f > 0.0:
                row: List[Any] = [ts_ms_i, px_f, v_f, hi_f, lo_f]
                op_f = safe_pos_float(opens[idx], default=None) if (has_opens and idx < len(opens)) else None
                if op_f is not None and op_f > 0.0:
                    row.append(op_f)
                pts.append(row)
            else:
                pts.append([ts_ms_i, px_f, v_f] if v_f > 0.0 else [ts_ms_i, px_f])
    pts.sort(key=lambda x: x[0])
    return pts


def load_stitched_24h_ohlc_points(
    ohlc_fn: Any,
    sym: str,
    cutoff_24h: float,
    *,
    require_positive_volume: bool = False,
    cache_dir: Optional[str] = None,
) -> List[List[Any]]:
    """Load 5m OHLC points since `cutoff_24h` and prepend older 15m OHLC points when span < 18h."""
    allow_offline_disk = is_offline_disk_cache_enabled()
    sym_raw_s = "" if isinstance(sym, bool) else str(sym or "").strip()
    sym_u = sym_raw_s.upper()
    ident_sym = resolve_symbol_identity(sym_raw_s) if sym_raw_s else None
    sym_canon = (
        ident_sym.display_symbol
        if (ident_sym is not None and ident_sym.is_tokenized_equity and ident_sym.display_symbol)
        else sym_u
    )
    cutoff_ms = (normalize_epoch_sec(cutoff_24h, default=0.0, allow_zero=True) or 0.0) * 1000.0
    cpath = ""
    disk_fallback_pts: List[List[Any]] = []
    _, writer_anchor = callable_identity_anchor(ohlc_fn) if callable(ohlc_fn) else (None, None)
    if sym_canon and sym_canon.replace("_", "").isalnum():
        try:
            fname = cache_ohlc_24h_filename(sym_canon)
            cpath = (
                str(resolve_cachedb_filepath(os.path.join(cache_dir, fname)))
                if cache_dir
                else str(cache_path(fname))
            )
            read_cpath = cpath
            if not os.path.exists(read_cpath) and sym_canon != sym_u:
                alt_fname = cache_ohlc_24h_filename(sym_u, preserve_case=True)
                alt_cpath = (
                    str(resolve_cachedb_filepath(os.path.join(cache_dir, alt_fname)))
                    if cache_dir
                    else str(cache_path(alt_fname))
                )
                if os.path.exists(alt_cpath):
                    read_cpath = alt_cpath
            can_read_disk = (
                not callable(ohlc_fn)
                or allow_offline_disk
                or is_live_network_backfill_enabled()
                or (
                    (prev_writer := _LAST_OHLC_WRITER_ID_BY_PATH.get(read_cpath)) is not None
                    and same_callable_anchor(prev_writer, writer_anchor)
                )
            )
            if can_read_disk:
                st = os.stat(read_cpath)
                age_s = _time.time() - st.st_mtime
                max_disk_age = 7200.0 if allow_offline_disk else 3600.0
                if 0.0 <= age_s < max_disk_age:
                    file_sig = (st.st_mtime_ns, st.st_size)
                    cached_ohlc = _OHLC_24H_DISK_MEMO.get(read_cpath)
                    raw_pts = cached_ohlc[1] if (cached_ohlc is not None and cached_ohlc[0] == file_sig) else None
                    if raw_pts is None:
                        d_obj = read_json_recovering(read_cpath)
                        cand_pts = d_obj.get("points") if isinstance(d_obj, dict) else d_obj
                        if isinstance(cand_pts, list):
                            raw_pts = [
                                list(p)
                                for p in cand_pts
                                if isinstance(p, (list, tuple)) and len(p) >= 2 and (safe_pos_float(p[1], 0.0) or 0.0) > 0.0
                            ]
                            lru_dict_put(_OHLC_24H_DISK_MEMO, read_cpath, (file_sig, raw_pts), 64)
                    if raw_pts:
                        filt = [
                            list(p)
                            for p in raw_pts
                            if safe_float(p[0], default=-1e18) >= cutoff_ms - 300000.0
                            and (not require_positive_volume or (len(p) >= 3 and (safe_pos_float(p[2], 0.0) or 0.0) > 0.0))
                        ]
                        if len(filt) >= 8 and (filt[-1][0] - filt[0][0]) >= 18 * 3600 * 1000:
                            fresh_ttl = 180.0 + (sum(ord(ch) for ch in sym_u) % 5) * 30.0
                            if age_s < fresh_ttl or allow_offline_disk or not callable(ohlc_fn):
                                return filt
                            disk_fallback_pts = filt
        except Exception:
            pass
    if not callable(ohlc_fn):
        return disk_fallback_pts
    pts_all = _extract_ohlc_series_points(ohlc_fn(sym, 5), min_ts=cutoff_24h, require_positive_volume=False)
    pts_eff = [p for p in pts_all if len(p) >= 3 and (safe_pos_float(p[2], 0.0) or 0.0) > 0.0] if require_positive_volume else pts_all
    if not pts_eff or (pts_eff[-1][0] - pts_eff[0][0]) < 18 * 3600 * 1000:
        first_s5_ts = (pts_eff[0][0] / 1000.0 - 300.0) if pts_eff else math.inf
        older_15 = _extract_ohlc_series_points(
            ohlc_fn(sym, 15), min_ts=cutoff_24h, max_ts=first_s5_ts, require_positive_volume=False
        )
        pts_all = older_15 + pts_all
    if cpath and len(pts_all) >= 8 and (pts_all[-1][0] - pts_all[0][0]) >= 18 * 3600 * 1000:
        try:
            with generation_file_lock(cpath):
                now_w = _time.time()
                cur_disk = read_json_dict(cpath, default=None, repair_files=True)
                cur_ts = normalize_epoch_sec(cur_disk.get("ts") if isinstance(cur_disk, dict) else None, default=None)
                if not (is_equity_symbol(sym_u) and cur_ts is not None and -5.0 <= (now_w - cur_ts) < 240.0):
                    save_json_state(cpath, {"ts": now_w, "symbol": sym_canon, "points": pts_all})
                    st_new = os.stat(cpath)
                    lru_dict_put(
                        _OHLC_24H_DISK_MEMO,
                        cpath,
                        ((st_new.st_mtime_ns, st_new.st_size), [list(p) for p in pts_all]),
                        64,
                    )
                    if len(_LAST_OHLC_WRITER_ID_BY_PATH) >= 64:
                        _LAST_OHLC_WRITER_ID_BY_PATH.clear()
                    _LAST_OHLC_WRITER_ID_BY_PATH[cpath] = writer_anchor
        except Exception:
            pass
    res_pts = (
        [list(p) for p in pts_all if len(p) >= 3 and (safe_pos_float(p[2], 0.0) or 0.0) > 0.0]
        if require_positive_volume
        else pts_all
    )
    if (len(res_pts) < 8 or (res_pts[-1][0] - res_pts[0][0]) < 18 * 3600 * 1000) and disk_fallback_pts:
        return disk_fallback_pts
    return res_pts


def _reconcile_major_swing_low(
    target_dict: Dict[str, Any], p_cur: float, l24: Optional[float], decimals: int
) -> None:
    maj_lo = safe_pos_float(target_dict.get("major_swing_low_px"), default=None)
    if maj_lo is None or maj_lo <= 0.0:
        return
    maj_lo_new = round(min(maj_lo, p_cur, *([l24] if l24 is not None else [])), decimals)
    eff_sw_lo = safe_pos_float(target_dict.get("swing_low_px"), default=None)
    promote_to_swing = (
        eff_sw_lo is not None
        and maj_lo_new < eff_sw_lo - 1e-9
        and abs(p_cur - eff_sw_lo) / max(p_cur, 1e-12) < 5e-4
    )
    if eff_sw_lo is None or (maj_lo_new < eff_sw_lo - 1e-9 and not promote_to_swing):
        target_dict["major_swing_low_px"] = maj_lo_new
        target_dict["rally_from_major_low_pct"] = pct_delta(p_cur, maj_lo_new)
        return
    if promote_to_swing:
        target_dict.update(swing_low_px=maj_lo_new, rally_from_swing_low_pct=pct_delta(p_cur, maj_lo_new))
    for k in ("major_swing_low_px", "rally_from_major_low_pct"):
        target_dict.pop(k, None)


def recalculate_live_price_geometry(pinfo: Dict[str, Any], p_cur: float) -> None:
    """Recalculate VP position, 24h/7d/15m range bounds, swing waves, and 6h bottom structure for live price `p_cur`."""
    if not isinstance(pinfo, dict) or p_cur <= 0.0:
        return
    poc_v, val_v, vah_v = (safe_pos_float(pinfo.get(k), default=0.0) or 0.0 for k in ("poc_px", "val_px", "vah_px"))
    if poc_v > 0.0 and val_v > 0.0 and vah_v > val_v:
        pinfo.update(build_vp_position_dict(p_cur, poc_v, val_v, vah_v))
    if (vwap_v := safe_pos_float(pinfo.get("session_vwap_px"), default=None)) is not None:
        pinfo["dist_vs_session_vwap_pct"] = pct_delta(p_cur, vwap_v)
        if (sig_pct_v := safe_pos_float(pinfo.get("session_vwap_sigma_pct"), default=None)) is not None:
            sig_px_v = vwap_v * (sig_pct_v / 100.0)
            if sig_px_v > 0.0:
                pinfo["session_vwap_z"] = round((p_cur - vwap_v) / sig_px_v, 2) + 0.0
    l24 = safe_pos_float(pinfo.get("low_24h"), default=None)
    h24 = safe_pos_float(pinfo.get("high_24h"), default=None)
    if l24 is not None and h24 is not None:
        l24, h24 = round(min(l24, p_cur), 6), round(max(h24, p_cur), 6)
        pinfo.update(compute_horizon_range_dict(p_cur, l24, h24, "24h"))
        if (c24 := safe_float(pinfo.get("chg_24h_pct"), default=None)) is not None:
            dd24, b24 = safe_float(pinfo.get("dd_24h_high_pct"), default=None), safe_float(pinfo.get("bounce_24h_low_pct"), default=None)
            if dd24 is not None and c24 < dd24:
                pinfo["chg_24h_pct"] = dd24
            elif b24 is not None and c24 > b24:
                pinfo["chg_24h_pct"] = b24
    l7d = safe_pos_float(pinfo.get("low_7d"), default=None)
    h7d = safe_pos_float(pinfo.get("high_7d"), default=None)
    if l7d is not None and h7d is not None:
        eff_l7d = min(l7d, p_cur, *([l24] if l24 is not None else []))
        eff_h7d = max(h7d, p_cur, *([h24] if h24 is not None else []))
        pinfo.update(compute_horizon_range_dict(p_cur, round(eff_l7d, 6), round(eff_h7d, 6), "7d"))
        if (c7d := safe_float(pinfo.get("chg_7d_pct"), default=None)) is not None:
            dd7, b7 = safe_float(pinfo.get("dd_7d_high_pct"), default=None), safe_float(pinfo.get("bounce_7d_low_pct"), default=None)
            if dd7 is not None and c7d < dd7:
                pinfo["chg_7d_pct"] = dd7
            elif b7 is not None and c7d > b7:
                pinfo["chg_7d_pct"] = b7
    sw_lo = safe_pos_float(pinfo.get("swing_low_px"), default=None)
    sw_hi = safe_pos_float(pinfo.get("swing_high_px"), default=None)
    if sw_lo is not None and sw_hi is not None:
        if not pinfo.get("recent_swings_7d") and l24 is not None and h24 is not None:
            sw_lo_new, sw_hi_new = round(min(sw_lo, p_cur, l24), 6), round(max(sw_hi, p_cur, h24), 6)
        else:
            sw_lo_new = round(
                min(
                    sw_lo,
                    p_cur,
                    *(
                        [l24]
                        if l24 is not None
                        and l24 < sw_lo
                        and safe_pos_float(pinfo.get("major_swing_low_px"), default=None) is None
                        else []
                    ),
                ),
                6,
            )
            sw_hi_new = round(max(sw_hi, p_cur, h24 if h24 is not None and not pinfo.get("recent_swings_7d") else p_cur), 6)
        pinfo["swing_low_px"], pinfo["swing_high_px"] = sw_lo_new, sw_hi_new
        if sw_lo_new > 0.0:
            pinfo["rally_from_swing_low_pct"] = pct_delta(p_cur, sw_lo_new)
        if sw_hi_new > 0.0:
            pinfo["pullback_from_swing_high_pct"] = pct_delta(p_cur, sw_hi_new)
    _reconcile_major_swing_low(pinfo, p_cur, l24, 6)
    r1h_lo = safe_pos_float(pinfo.get("recent_1h_low_px"), default=None)
    p6h_lo = safe_pos_float(pinfo.get("prior_6h_low_px"), default=None)
    if r1h_lo is not None and p6h_lo is not None and p6h_lo > 0.0:
        eff_dec_b = _auto_px_decimals(p_cur)
        r1h_new = min(r1h_lo, p_cur)
        pinfo["recent_1h_low_px"] = round(r1h_new, eff_dec_b)
        b_struct = pinfo.get("bottom_structure_6h")
        if r1h_new <= p6h_lo and p_cur <= p6h_lo:
            for k_b in ("higher_low_vs_prior_pct", "reclaim_above_prior_low_pct"):
                pinfo.pop(k_b, None)
            if ((p_cur / r1h_new) - 1.0) * 100.0 <= 2.5:
                pinfo["bottom_structure_6h"] = "FRESH_6H_LOW_TEST"
            else:
                for k_b in ("bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px"):
                    pinfo.pop(k_b, None)
        elif r1h_new < p6h_lo and p_cur > p6h_lo and b_struct in ("HIGHER_LOW_DOUBLE_BOTTOM", "LIQUIDITY_SWEEP_RECLAIM", "FRESH_6H_LOW_TEST"):
            pinfo.pop("higher_low_vs_prior_pct", None)
            pinfo.update(
                {"bottom_structure_6h": "LIQUIDITY_SWEEP_RECLAIM", "reclaim_above_prior_low_pct": pct_delta(p_cur, p6h_lo)}
            )
        elif b_struct == "HIGHER_LOW_DOUBLE_BOTTOM" or (
            b_struct == "FRESH_6H_LOW_TEST"
            and r1h_new >= p6h_lo
            and (
                p_cur > r1h_new * 1.002
                or (r1h_new > p6h_lo and p_cur > r1h_new * 1.0005 and p_cur > p6h_lo * 1.001)
            )
        ):
            pinfo.pop("reclaim_above_prior_low_pct", None)
            pinfo.update(
                {"bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM", "higher_low_vs_prior_pct": pct_delta(r1h_new, p6h_lo)}
            )
    eff_dec_15m = _auto_px_decimals(p_cur)
    p_cur_15m = round(p_cur, eff_dec_15m)
    if isinstance(tape_15m := pinfo.get("recent_tape_15m"), list) and tape_15m:
        tape_copy = list(tape_15m)
        if isinstance(tape_copy[-1], str) and tape_copy[-1].startswith("now:"):
            now_s = f"{p_cur_15m:.{eff_dec_15m}f}".rstrip("0").rstrip(".")
            tape_copy[-1] = f"now:{now_s}"
            pinfo["recent_tape_15m"] = tape_copy
            tape_ref_by_lbl = {
                lbl_s.strip(): ref_px
                for entry in tape_copy[:-1]
                if isinstance(entry, str)
                and ":" in entry
                and (lbl_s := entry.split(":", 1)[0])
                and (ref_px := safe_pos_float(entry.split(":", 1)[1], default=None)) is not None
            }
            for tape_lbl, chg_key in (
                ("-1m", "chg_1m_pct"),
                ("-3m", "chg_3m_pct"),
                ("-5m", "chg_5m_pct"),
                ("-15m", "chg_15m_pct"),
            ):
                if chg_key in pinfo and (ref_px_t := tape_ref_by_lbl.get(tape_lbl)) is not None:
                    pinfo[chg_key] = pct_delta(p_cur_15m, ref_px_t)
            has_eps_scale = (safe_pos_float(pinfo.get("epsilon"), default=0.0) or 0.0) > 0.0
            for s_key_t, win_s_t, win_m_t, primary_lbl, fallback_lbl in (
                ("slope_90s", 90.0, 1.5, "-1m", "-3m"),
                ("slope_3_7m", 222.0, 3.7, "-3m", "-5m"),
            ):
                if s_key_t in pinfo and abs(safe_float(pinfo.get(s_key_t), 0.0)) <= 1e-12:
                    ref_t_px = tape_ref_by_lbl.get(primary_lbl) or tape_ref_by_lbl.get(fallback_lbl)
                    if ref_t_px is not None and ref_t_px > 0.0 and abs(p_cur_15m - ref_t_px) > 1e-12:
                        raw_st = ((p_cur_15m - ref_t_px) / win_s_t) if has_eps_scale else (((p_cur_15m - ref_t_px) / ref_t_px) / win_m_t)
                        pinfo[s_key_t] = _round_small_slope(raw_st)
            if (
                pinfo.get("gradient_recent_fast") is not None
                and abs(safe_float(pinfo.get("gradient_recent_fast"), 0.0)) <= 1e-12
            ):
                for fb_gf_k in ("slope_90s", "slope_3_7m"):
                    if (fb_gf := safe_float(pinfo.get(fb_gf_k), default=None)) is not None and abs(fb_gf) > 1e-12:
                        pinfo["gradient_recent_fast"] = fb_gf
                        break
    l15 = safe_pos_float(pinfo.get("low_15m"), default=None)
    h15 = safe_pos_float(pinfo.get("high_15m"), default=None)
    if l15 is not None and h15 is not None:
        l15_new, h15_new = round(min(l15, p_cur_15m), eff_dec_15m), round(max(h15, p_cur_15m), eff_dec_15m)
        pinfo["low_15m"], pinfo["high_15m"] = l15_new, h15_new
        if h15_new > l15_new > 0:
            pinfo["pos_15m_pct"] = round(min(100.0, max(0.0, ((p_cur_15m - l15_new) / (h15_new - l15_new)) * 100.0)), 1)
            pinfo["span_15m_pct"] = pct_delta(h15_new, l15_new)


def apply_24h_series_enrichment(
    sym: str,
    entry: Dict[str, Any],
    m24: Dict[str, Any],
    series_24h: List[Any],
    btc_series_24h: Optional[List[Any]],
) -> None:
    """Enrich a symbol's `prices_and_trends` entry with 24h metrics, cross-BTC beta/corr, Kalman, slopes, and 1h vol."""
    entry.update(m24)
    if series_24h and safe_float(entry.get("price"), 0.0) <= 0.0:
        parsed_pts_24 = _extract_sorted_ts_px(series_24h)
        if parsed_pts_24:
            entry["price"] = parsed_pts_24[-1][1]
        elif isinstance(series_24h[-1], (list, tuple)) and len(series_24h[-1]) >= 2:
            if (px_last_24 := safe_pos_float(series_24h[-1][1], default=None)) is not None:
                entry["price"] = px_last_24
    cur_e_px = safe_pos_float(entry.get("price"), default=None)
    l24_e, h24_e = safe_pos_float(entry.get("low_24h"), default=None), safe_pos_float(entry.get("high_24h"), default=None)
    l7d_e, h7d_e = safe_pos_float(entry.get("low_7d"), default=None), safe_pos_float(entry.get("high_7d"), default=None)
    is_eq = is_equity_symbol(sym)
    if cur_e_px is not None and l24_e is not None and h24_e is not None and l7d_e is not None and h7d_e is not None:
        if l24_e < l7d_e or h24_e > h7d_e:
            eff_dec_7d = 4 if is_eq else 6
            new_l7d = round(min(l7d_e, l24_e, cur_e_px), eff_dec_7d)
            new_h7d = round(max(h7d_e, h24_e, cur_e_px), eff_dec_7d)
            entry.update(compute_horizon_range_dict(cur_e_px, new_l7d, new_h7d, "7d"))
            if (c7d_e := safe_float(entry.get("chg_7d_pct"), default=None)) is not None:
                dd7_e, b7_e = safe_float(entry.get("dd_7d_high_pct"), default=None), safe_float(entry.get("bounce_7d_low_pct"), default=None)
                if dd7_e is not None and c7d_e < dd7_e:
                    entry["chg_7d_pct"] = dd7_e
                elif b7_e is not None and c7d_e > b7_e:
                    entry["chg_7d_pct"] = b7_e
            sw_lo_e = safe_pos_float(entry.get("swing_low_px"), default=None)
            sw_hi_e = safe_pos_float(entry.get("swing_high_px"), default=None)
            if sw_lo_e is not None and sw_hi_e is not None:
                has_swings_e = bool(entry.get("recent_swings_7d"))
                has_maj_lo_e = safe_pos_float(entry.get("major_swing_low_px"), default=None) is not None
                sw_lo_n = round(
                    min(sw_lo_e, cur_e_px, *([l24_e] if not has_swings_e or (l24_e < sw_lo_e and not has_maj_lo_e) else [])),
                    eff_dec_7d,
                )
                sw_hi_n = round(max(sw_hi_e, cur_e_px, *([h24_e] if not has_swings_e else [])), eff_dec_7d)
                entry["swing_low_px"], entry["swing_high_px"] = sw_lo_n, sw_hi_n
                if sw_lo_n > 0.0:
                    entry["rally_from_swing_low_pct"] = pct_delta(cur_e_px, sw_lo_n)
                if sw_hi_n > 0.0:
                    entry["pullback_from_swing_high_pct"] = pct_delta(cur_e_px, sw_hi_n)
            _reconcile_major_swing_low(entry, cur_e_px, l24_e, eff_dec_7d)
    if entry.get("trend", 0) == 0 and "chg_1h_pct" in m24:
        c1h = safe_float(m24.get("chg_1h_pct"), 0.0)
        if c1h >= 0.25 or c1h <= -0.25:
            entry["trend"] = 1 if c1h >= 0.25 else -1
    if series_24h and btc_series_24h and not is_eq and canonical_base_asset(sym) != "BTC":
        cross_btc = compute_cross_btc_metrics(series_24h, btc_series_24h)
        for ck in ("alt_btc_pos_24h_pct", "alt_btc_dd_24h_high_pct", "alt_btc_bounce_24h_low_pct", "beta_btc_24h", "corr_btc_24h"):
            if cross_btc.get(ck) is not None:
                entry[ck] = cross_btc[ck]
    k_cur_vel = safe_float(entry.get("kalman_vel"), default=None)
    k_cur_std = safe_float(entry.get("kalman_vel_std"), default=None)
    unseeded_kalman = is_unseeded_kalman_state(entry, treat_none_as_unseeded=True)
    if (unseeded_kalman or k_cur_std is None) and series_24h and isinstance(k_fb := compute_kalman_from_series(series_24h), dict):
        if k_cur_vel is None or k_cur_std is None or k_cur_std >= 5.99 or abs(safe_float(k_fb.get("kalman_vel"), 0.0)) >= 1e-5:
            for kk, kv in k_fb.items():
                if kv is not None and (entry.get(kk) is None or unseeded_kalman):
                    entry[kk] = kv
    resolve_kalman_vel_z(entry, populate=True)
    st_ts_e = safe_pos_float(entry.pop("_short_trend_ts", None), default=None)
    if series_24h:
        pts_s24_sorted = _extract_sorted_ts_px(series_24h)
        stale_short_ts = bool(
            st_ts_e is not None
            and pts_s24_sorted
            and ((safe_float(pts_s24_sorted[-1][0], default=0.0) or 0.0) / 1000.0 - st_ts_e) > 600.0
        )
        if stale_short_ts:
            for k_st_clr in (
                "slope_90s",
                "growth_90s",
                "trend_90s",
                "slope_3_7m",
                "growth_3_7m",
                "trend_3_7m",
                "slope_15m",
                "growth_15m",
                "trend_15m",
                "slope_2_5h",
                "growth_2_5h",
                "trend_2_5h",
                "gradient_recent_fast",
                "gradient_recent",
                *ALL_KINEMATIC_METRIC_KEYS,
            ):
                entry.pop(k_st_clr, None)
        s_2_5h = safe_float(entry.get("slope_2_5h"), default=None)
        s_15m = safe_float(entry.get("slope_15m"), default=None)
        stale_short_cache = stale_short_ts or (
            s_2_5h is not None
            and s_15m is not None
            and abs(s_2_5h - s_15m) < 1e-12
            and abs(safe_float(entry.get("growth_2_5h"), 0.0) - safe_float(entry.get("growth_15m"), 0.0)) < 1e-9
        )
        if (entry.get("vel_90s_pct_min") is None and entry.get("vel_5m_pct_min") is None) or stale_short_cache:
            for kk_s, kv_s in compute_series_kinematics(series_24h).items():
                if entry.get(kk_s) is None or stale_short_cache:
                    entry[kk_s] = kv_s
        if (entry.get("slope_3_7m") is None or stale_short_cache) and (s_fb := compute_short_slopes_from_series(series_24h)):
            for sk, sv in s_fb.items():
                if sv is None:
                    continue
                if entry.get(sk) is None or (
                    stale_short_cache
                    and (
                        sk in ("slope_2_5h", "growth_2_5h", "trend_2_5h")
                        or (sk in ("slope_90s", "growth_90s", "trend_90s") and abs(safe_float(entry.get("slope_90s"), 0.0)) < 1e-12)
                    )
                ):
                    entry[sk] = sv
            if stale_short_cache and "growth_2_5h" not in s_fb:
                entry.pop("growth_2_5h", None)
            if entry.get("gradient_recent") is None or abs(safe_float(entry.get("gradient_recent"), 0.0)) < 1e-12:
                for fb_gr_k in ("slope_3_7m", "slope_15m"):
                    if s_fb.get(fb_gr_k) is not None and abs(safe_float(s_fb.get(fb_gr_k), 0.0)) > 0.0:
                        entry["gradient_recent"] = s_fb[fb_gr_k]
                        break
        if len(series_24h) >= 20:
            all_pts_m = _extract_sorted_ts_px(series_24h)
            tail_pts_m: List[Tuple[float, float]] = []
            sub_all_m: List[Tuple[float, float]] = []
            has_dense_1h_bars = False
            vol_1h_inflated = False
            if len(all_pts_m) >= 10:
                cutoff_1h_ms = (safe_float(all_pts_m[-1][0], default=0.0) or 0.0) - 3600.0 * 1000.0
                last_t_ms = -1e18
                for t_ms, px_v in all_pts_m:
                    if (t_ms - last_t_ms) >= 55000.0:
                        sub_all_m.append((t_ms, px_v))
                        if t_ms >= cutoff_1h_ms:
                            tail_pts_m.append((t_ms, px_v))
                        last_t_ms = t_ms
                nz_tail = sum(
                    1 for i in range(1, len(tail_pts_m)) if abs(tail_pts_m[i][1] - tail_pts_m[i - 1][1]) > 1e-12
                )
                has_dense_1h_bars = len(tail_pts_m) >= 20 and nz_tail >= 10
                cur_vol_1h = safe_pos_float(entry.get("vol_1h_pct"), default=None)
                span_24h = safe_pos_float(entry.get("span_24h_pct"), default=None)
                vol_1h_inflated = bool(
                    cur_vol_1h is not None
                    and (
                        (span_24h is not None and span_24h > 0.0 and cur_vol_1h > max(span_24h * 0.85, 2.0))
                        or (len(tail_pts_m) >= 1 and nz_tail < 10 and (len(sub_all_m) >= 20 or len(all_pts_m) >= 20))
                    )
                )
                no_pos_vol_1h = cur_vol_1h is None
                if not has_dense_1h_bars and (no_pos_vol_1h or stale_short_cache or vol_1h_inflated):
                    tail_pts_m = sub_all_m[-60:] if len(sub_all_m) >= 20 else all_pts_m[-60:]
            if len(tail_pts_m) >= 20 and (
                has_dense_1h_bars
                or safe_pos_float(entry.get("vol_1h_pct"), default=None) is None
                or stale_short_cache
                or vol_1h_inflated
            ):
                t_span_ms = (safe_float(tail_pts_m[-1][0], default=0.0) or 0.0) - (
                    safe_float(tail_pts_m[0][0], default=0.0) or 0.0
                )
                dt_s_m = max(60.0, t_span_ms / (1000.0 * max(1, len(tail_pts_m) - 1)))
                tail_pxs_m = [
                    px_r for r in tail_pts_m if (px_r := safe_pos_float(r[1], default=None)) is not None
                ]
                if (v1h_m := calculate_volatility_1h(tail_pxs_m, dt_s_m)) is not None and v1h_m > 0:
                    entry["vol_1h_pct"] = v1h_m
                    re_th_m, dca_th_m = adaptive_thresholds(v1h_m)
                    if re_th_m is not None and (
                        has_dense_1h_bars
                        or safe_pos_float(entry.get("adapt_reentry_pct"), default=None) is None
                        or stale_short_cache
                        or vol_1h_inflated
                    ):
                        entry["adapt_reentry_pct"] = re_th_m
                    if dca_th_m is not None and (
                        has_dense_1h_bars
                        or safe_pos_float(entry.get("adapt_dca_pct"), default=None) is None
                        or stale_short_cache
                        or vol_1h_inflated
                    ):
                        entry["adapt_dca_pct"] = dca_th_m


_RTH_VP_ANCHOR_KEYS: Tuple[str, ...] = (
    "poc_px",
    "sec_poc_px",
    "hvn2_px",
    "lvn_px",
    "val_px",
    "vah_px",
    "profile_shape",
    "vp_source",
)
_RTH_VWAP_ANCHOR_KEYS: Tuple[str, ...] = (
    "session_vwap_px",
    "session_vwap_sigma_pct",
    "session_vwap_anchor",
)
STALE_SHORT_24H_KEYS: Tuple[str, ...] = (
    "chg_1m_pct",
    "chg_3m_pct",
    "chg_5m_pct",
    "recent_tape_15m",
    "low_15m",
    "high_15m",
    "pos_15m_pct",
    "span_15m_pct",
)
CLOSED_EQUITY_INTRADAY_KEYS: Tuple[str, ...] = (
    "chg_15m_pct",
    "chg_30m_pct",
    "chg_1h_pct",
)
CLOSED_EQUITY_QUOTE_AGE_MIN: float = 30.0
_SHORT_SLOPE_OVERRIDE_KEYS: Tuple[str, ...] = (
    "kalman_vel",
    "kalman_vel_std",
    "kalman_trend",
    "slope_90s",
    "growth_90s",
    "trend_90s",
    "slope_3_7m",
    "growth_3_7m",
    "trend_3_7m",
    "slope_15m",
    "growth_15m",
    "trend_15m",
    "slope_2_5h",
    "growth_2_5h",
    "trend_2_5h",
    "gradient_recent_fast",
    "gradient_recent",
)


def sort_ts_rows(rows: list) -> list:
    """Sort price-series rows ascending by parsed timestamp only when out-of-order rows exist."""
    if len(rows) < 2:
        return rows
    prev_t = -1.0
    for r in rows:
        pt = parse_ts_px_row(r)
        if pt is not None:
            if pt[0] < prev_t:
                keyed = [(p[0], r_item) for r_item in rows if (p := parse_ts_px_row(r_item)) is not None]
                keyed.sort(key=lambda x: x[0])
                return [r_item for _, r_item in keyed]
            prev_t = pt[0]
    return rows


def prepend_older_non_diverged_points(cur_ser: list, cand_ser: list) -> list:
    """Prepend older points from `cand_ser` to `cur_ser` when prices are non-diverged."""
    if not cur_ser or not cand_ser:
        return cur_ser
    cur_p0 = parse_ts_px_row(cur_ser[0])
    sib_p0, sib_p1 = parse_ts_px_row(cand_ser[0]), parse_ts_px_row(cand_ser[-1])
    if cur_p0 and sib_p0 and sib_p1 and sib_p0[0] < cur_p0[0] and not _prices_diverged(sib_p1[1], cur_p0[1]):
        older_pts = [r for r in cand_ser if (pt := parse_ts_px_row(r)) is not None and pt[0] < cur_p0[0]]
        if older_pts:
            return older_pts + cur_ser
    return cur_ser


def strip_off_rth_flat_poller_tail(rows: list, symbol: str, *, is_market_open_fn: Any = None) -> list:
    """Strip or downsample flat off-RTH poller tails on equity price series."""
    sym_u_load = str(symbol).strip().upper()
    is_eq_sym_load = is_equity_symbol(sym_u_load)
    if not is_eq_sym_load or len(rows) < 3:
        return rows
    if len(rows) >= 10:
        for split_idx in range(len(rows) - 1, max(4, len(rows) - 4), -1):
            p_prev = parse_ts_px_row(rows[split_idx - 1])
            p_cur = parse_ts_px_row(rows[split_idx])
            p_last = parse_ts_px_row(rows[-1])
            if (
                p_prev is not None
                and p_cur is not None
                and p_last is not None
                and (p_cur[0] - p_prev[0]) > 2 * 3600 * 1000
                and (p_last[0] - p_cur[0]) < 15 * 60 * 1000
            ):
                rows = rows[:split_idx]
                break
    last_ohlc_idx = max(
        (
            i
            for i, r in enumerate(rows)
            if isinstance(r, (list, tuple)) and len(r) >= 3 and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
        ),
        default=-1,
    )
    if 1 <= last_ohlc_idx < len(rows) - 1:
        last_ohlc_pt = parse_ts_px_row(rows[last_ohlc_idx])
        tail_pts = [pt for r in rows[last_ohlc_idx + 1 :] if (pt := parse_ts_px_row(r)) is not None]
        if (
            last_ohlc_pt is not None
            and tail_pts
            and (tail_pts[-1][0] - last_ohlc_pt[0]) > 20 * 60 * 1000
            and max(p[1] for p in tail_pts) == min(p[1] for p in tail_pts)
            and abs(tail_pts[0][1] - last_ohlc_pt[1]) <= 1e-6
        ):
            return rows[: last_ohlc_idx + 1]
    last_pt = parse_ts_px_row(rows[-1])
    if last_pt is not None and isinstance(rows[-1], (list, tuple)) and len(rows[-1]) < 3:
        flat_start = len(rows) - 1
        while flat_start > 0:
            prev_r = rows[flat_start - 1]
            prev_pt = parse_ts_px_row(prev_r)
            if (
                prev_pt is not None
                and isinstance(prev_r, (list, tuple))
                and len(prev_r) < 3
                and abs(prev_pt[1] - last_pt[1]) <= 1e-6
            ):
                flat_start -= 1
            else:
                break
        anchor_idx = flat_start
        if anchor_idx > 0:
            before_pt = parse_ts_px_row(rows[anchor_idx - 1])
            if before_pt is not None and abs(before_pt[1] - last_pt[1]) <= 1e-6:
                anchor_idx -= 1
        anchor_pt = parse_ts_px_row(rows[anchor_idx]) if 1 <= anchor_idx < len(rows) - 1 else None
        if anchor_pt is not None and (last_pt[0] - anchor_pt[0]) > 20 * 60 * 1000:
            next_flat_pt = parse_ts_px_row(rows[anchor_idx + 1]) if (anchor_idx + 1) < len(rows) else None
            gap_after_anchor_ms = (next_flat_pt[0] - anchor_pt[0]) if next_flat_pt is not None else 0.0
            rth_open_at_last = False
            if gap_after_anchor_ms <= 20 * 60 * 1000:
                try:
                    eff_open_fn = is_symbol_market_open if is_market_open_fn is None else is_market_open_fn
                    rth_open_at_last = bool(eff_open_fn(symbol, last_pt[0] / 1000.0))
                except Exception:
                    rth_open_at_last = False
            if not rth_open_at_last or gap_after_anchor_ms > 20 * 60 * 1000:
                return rows[: anchor_idx + 1]
            downsampled_flat = [rows[anchor_idx]]
            last_kept_t = anchor_pt[0]
            for r_fl in rows[anchor_idx + 1 : -1]:
                pt_fl = parse_ts_px_row(r_fl)
                if pt_fl is not None and (pt_fl[0] - last_kept_t) >= 55000.0:
                    downsampled_flat.append(r_fl)
                    last_kept_t = pt_fl[0]
            downsampled_flat.append(rows[-1])
            return rows[:anchor_idx] + downsampled_flat
    return rows


def collapse_watcher_equity_series(w_rows: list, is_eq_w: bool) -> Tuple[list, list, Optional[float]]:
    """Collapse flat runs in `cache_watcher_all_price` for equities and extract 7d points and 24h cutoff."""
    w_rows = sort_ts_rows(w_rows or [])
    if len(w_rows) < 10 or (w_last := parse_ts_px_row(w_rows[-1])) is None:
        return [], [], None
    w_24 = [pt for r in w_rows if (pt := parse_ts_px_row(r)) is not None and pt[0] >= w_last[0] - 86400.0 * 1000.0]
    if len(w_24) < 10 or (w_24[-1][0] - w_24[0][0]) < 3600.0 * 1000.0:
        return [], [], None
    if not is_eq_w:
        return list(w_rows), [], None
    w_parsed = [[pt[0], round(pt[1], 4)] for r in w_rows if (pt := parse_ts_px_row(r)) is not None]
    raw_last_ts = w_parsed[-1][0]
    watcher_7d_pts = [pt for pt in w_parsed if pt[0] >= raw_last_ts - 7.05 * 86400.0 * 1000.0]
    runs: List[List[List[float]]] = []
    for pt in w_parsed:
        if not runs or abs(pt[1] - runs[-1][-1][1]) > 1e-9:
            runs.append([pt])
        else:
            runs[-1].append(pt)
    last_active_ts = runs[-1][0][0] if len(runs[-1]) >= 4 else raw_last_ts
    watcher_cutoff_24h = (
        raw_last_ts - 86400.0 * 1000.0
        if (raw_last_ts - last_active_ts) <= 6.0 * 3600.0 * 1000.0
        else last_active_ts - 86400.0 * 1000.0
    )
    anchor_targets = (
        watcher_cutoff_24h,
        last_active_ts - 86400.0 * 1000.0,
        last_active_ts - 43200.0 * 1000.0,
        last_active_ts - 21600.0 * 1000.0,
    )
    collapsed_w: list = []
    for r_idx, r_pts in enumerate(runs):
        if len(r_pts) < 4:
            collapsed_w.extend(r_pts)
        elif r_idx == len(runs) - 1:
            collapsed_w.append(r_pts[0])
        else:
            keep_idx = {0, len(r_pts) - 1}
            for t_anc in anchor_targets:
                if r_pts[0][0] <= t_anc <= r_pts[-1][0]:
                    best_i = min(range(len(r_pts)), key=lambda i: abs(r_pts[i][0] - t_anc))
                    keep_idx.add(best_i)
            for k_i in sorted(keep_idx):
                collapsed_w.append(r_pts[k_i])
    return collapsed_w, watcher_7d_pts, watcher_cutoff_24h


def extract_archive_24h_and_7d_slices(
    archive_pts: list,
    sparse_7d_pts: list,
    watcher_7d_pts: list,
    series_24h: list,
    symbol: str,
    need_24h_supplement: bool,
    watcher_cutoff_24h: Optional[float],
) -> Tuple[list, list, float, float]:
    """Slice 24h supplement and 7d values from archive/sparse/watcher points, filtering zero-vol price spikes."""
    vals_7d: list = []
    arch_7d_end_ts = 0.0
    arch_7d_span_ms = 0.0
    if not archive_pts:
        return series_24h, vals_7d, arch_7d_end_ts, arch_7d_span_ms
    try:
        if (last_pt := parse_ts_px_row(archive_pts[-1])) is not None:
            sym_u_arch = str(symbol).strip().upper()
            if (
                is_equity_symbol(sym_u_arch)
                and series_24h
                and isinstance(series_24h[-1], (list, tuple))
                and len(series_24h[-1]) >= 3
                and isinstance(archive_pts[-1], (list, tuple))
                and len(archive_pts[-1]) >= 3
                and (s24_end_pt := parse_ts_px_row(series_24h[-1])) is not None
                and s24_end_pt[0] == last_pt[0]
                and safe_nonneg_float(series_24h[-1][2], -1.0) == safe_nonneg_float(archive_pts[-1][2], -2.0)
                and abs(s24_end_pt[1] - last_pt[1]) > 1e-4
            ):
                series_24h = list(series_24h[:-1]) + [[series_24h[-1][0], last_pt[1], *series_24h[-1][2:]]]
            cutoff_24h = watcher_cutoff_24h if watcher_cutoff_24h is not None else (last_pt[0] - 86400.0 * 1000.0)
            cutoff_7d = last_pt[0] - 7.0 * 86400.0 * 1000.0
            pts_24, pts_7d_vals, pts_7d_ts = [], [], []
            parsed_arch = [
                (
                    pt[0],
                    pt[1],
                    safe_pos_float(row[2], default=None) if isinstance(row, (list, tuple)) and len(row) >= 3 else None,
                    row[0] if isinstance(row, (list, tuple)) and row else pt[0],
                )
                for row in archive_pts
                if (pt := parse_ts_px_row(row)) is not None
            ]
            has_arch_vol = any(v is not None for _, _, v, _ in parsed_arch)
            clean_arch = []
            for idx_a, item_a in enumerate(parsed_arch):
                if has_arch_vol and 0 < idx_a < len(parsed_arch) - 1 and item_a[2] is None:
                    prev_a = clean_arch[-1] if clean_arch else parsed_arch[idx_a - 1]
                    next_idx_a = idx_a + 1
                    while (
                        next_idx_a < len(parsed_arch) - 1
                        and parsed_arch[next_idx_a][2] is None
                        and abs(parsed_arch[next_idx_a][1] - item_a[1]) <= 1e-9
                    ):
                        next_idx_a += 1
                    next_a = parsed_arch[next_idx_a]
                    if prev_a[2] is not None or next_a[2] is not None:
                        ref_mid = 0.5 * (prev_a[1] + next_a[1])
                        if (
                            ref_mid > 0.0
                            and abs(prev_a[1] - next_a[1]) / ref_mid <= 0.025
                            and abs(item_a[1] - ref_mid) / ref_mid > 0.025
                        ):
                            continue
                clean_arch.append(item_a)
            older_24_arch = []
            for pt_t, pt_p, pt_v, raw_t0 in clean_arch:
                if pt_t >= cutoff_7d:
                    pts_7d_vals.append(pt_p)
                    pts_7d_ts.append(pt_t)
                if need_24h_supplement:
                    row_24 = [raw_t0, pt_p, pt_v] if pt_v is not None else [raw_t0, pt_p]
                    if pt_t >= cutoff_24h:
                        pts_24.append(row_24)
                    elif pt_t >= last_pt[0] - 36.0 * 3600.0 * 1000.0:
                        older_24_arch.append(row_24)
            if (
                need_24h_supplement
                and older_24_arch
                and (
                    not pts_24
                    or (
                        (p24_0 := parse_ts_px_row(pts_24[0])) is not None
                        and (p24_1 := parse_ts_px_row(pts_24[-1])) is not None
                        and (p24_1[0] - p24_0[0]) < 18.0 * 3600.0 * 1000.0
                    )
                )
            ):
                pts_24 = [older_24_arch[-1], *pts_24]
            if len(watcher_7d_pts) >= 2:
                pts_7d_vals = [pt[1] for pt in watcher_7d_pts]
                pts_7d_ts = [pt[0] for pt in watcher_7d_pts]
            arch_7d_end_ts = pts_7d_ts[-1] if pts_7d_ts else last_pt[0]
            if len(pts_7d_ts) >= 2:
                arch_7d_span_ms = max(0.0, pts_7d_ts[-1] - pts_7d_ts[0])
            if sparse_7d_pts and (sp_last := parse_ts_px_row(sparse_7d_pts[-1])) is not None:
                merged_sp = (
                    prepend_older_non_diverged_points(archive_pts, sparse_7d_pts)
                    if last_pt[0] >= sp_last[0]
                    else prepend_older_non_diverged_points(sparse_7d_pts, archive_pts)
                )
                sp_cutoff_7d = max(last_pt[0], sp_last[0]) - 7.0 * 86400.0 * 1000.0
                sp_pts = [pt for row in merged_sp if (pt := parse_ts_px_row(row)) is not None and pt[0] >= sp_cutoff_7d]
                if len(sp_pts) >= 2:
                    pts_7d_vals = [pt[1] for pt in sp_pts]
                    arch_7d_end_ts = sp_pts[-1][0]
                    arch_7d_span_ms = max(0.0, sp_pts[-1][0] - sp_pts[0][0])
            if len(pts_7d_vals) >= 2:
                vals_7d = pts_7d_vals
            if need_24h_supplement and pts_24:
                series_24h = prepend_older_non_diverged_points(series_24h, pts_24) if series_24h else pts_24
    except Exception:
        pass
    return series_24h, vals_7d, arch_7d_end_ts, arch_7d_span_ms


def normalize_and_dedup_24h_series(series_24h: list, symbol: str, *, is_market_open_fn: Any = None) -> list:
    """Normalize, sort, deduplicate, filter zero-volume spikes, and strip off-RTH flat tails on 24h series."""
    if not series_24h:
        return []
    cleaned_24h = []
    prev_t, is_sorted, has_any_vol = -1.0, True, False
    for row in series_24h:
        if (pt := parse_ts_px_row(row)) is not None:
            is_sorted = is_sorted and pt[0] >= prev_t
            prev_t = pt[0]
            vol_v = safe_pos_float(row[2], default=None) if len(row) >= 3 else None
            if vol_v is not None:
                has_any_vol = True
                cleaned_24h.append([pt[0], pt[1], vol_v])
            else:
                cleaned_24h.append([pt[0], pt[1]])
    if not is_sorted:
        cleaned_24h.sort(key=lambda x: x[0])
    if len(cleaned_24h) < 2:
        return []
    dedup_24h = [cleaned_24h[0]]
    for pt in cleaned_24h[1:]:
        if pt[0] > dedup_24h[-1][0]:
            dedup_24h.append(pt)
        else:
            dedup_24h[-1] = [pt[0], pt[1], dedup_24h[-1][2]] if (len(pt) < 3 and len(dedup_24h[-1]) >= 3) else pt
    _has_pos_vol_24h = lambda row_d: len(row_d) >= 3 and (safe_pos_float(row_d[2], default=0.0) or 0.0) > 0.0
    if has_any_vol and len(dedup_24h) >= 3 and any(_has_pos_vol_24h(p) for p in dedup_24h):
        filtered_24h = [dedup_24h[0]]
        for idx_d in range(1, len(dedup_24h) - 1):
            cur_d = dedup_24h[idx_d]
            prev_d = filtered_24h[-1]
            if not _has_pos_vol_24h(cur_d):
                next_idx_d = idx_d + 1
                while (
                    next_idx_d < len(dedup_24h) - 1
                    and not _has_pos_vol_24h(dedup_24h[next_idx_d])
                    and abs(dedup_24h[next_idx_d][1] - cur_d[1]) <= 1e-9
                ):
                    next_idx_d += 1
                next_d = dedup_24h[next_idx_d]
                if _has_pos_vol_24h(prev_d) or _has_pos_vol_24h(next_d):
                    ref_mid = 0.5 * (prev_d[1] + next_d[1])
                    if (
                        ref_mid > 0.0
                        and abs(prev_d[1] - next_d[1]) / ref_mid <= 0.025
                        and abs(cur_d[1] - ref_mid) / ref_mid > 0.025
                    ):
                        continue
            filtered_24h.append(cur_d)
        filtered_24h.append(dedup_24h[-1])
        dedup_24h = strip_off_rth_flat_poller_tail(filtered_24h, symbol, is_market_open_fn=is_market_open_fn)
    elif not has_any_vol and len(dedup_24h) >= 2:
        mkey = _series_memo_key(dedup_24h)
        if mkey is not None:
            _finalize_sorted_ts_px(
                dedup_24h,
                [(r[0] * 1.0, r[1] * 1.0) for r in dedup_24h],
                mkey,
                all_two_col=True,
            )
    return dedup_24h if len(dedup_24h) >= 2 else []


def append_newer_24h_to_7d_vals(vals_7d: list, series_24h: list, arch_7d_end_ts: float) -> list:
    """Append newer non-diverged prices from `series_24h` onto `vals_7d`."""
    if not vals_7d or not series_24h:
        return vals_7d
    appended_newer = False
    if arch_7d_end_ts > 0.0:
        for pt in series_24h:
            if pt[0] > arch_7d_end_ts and (pt_px := safe_pos_float(pt[1], default=None)) is not None:
                if vals_7d[-1] != pt_px and not _prices_diverged(pt_px, vals_7d[-1]):
                    vals_7d.append(pt_px)
                appended_newer = True
    if not appended_newer and (latest_px := safe_pos_float(series_24h[-1][1], default=None)) is not None:
        if vals_7d[-1] != latest_px and not _prices_diverged(latest_px, vals_7d[-1]):
            vals_7d.append(latest_px)
    return vals_7d


class TelemetryEvaluatorInterface(ABC):
    """Pure abstract base contract (Level 1) for stateless quantitative telemetry evaluators.

    Modeled on ``CacheManagerInterface(ABC)``: evaluators receive normalized series/snapshot
    data and ``now_ts``, perform pure mathematical computation without disk/network I/O,
    and return a dictionary of computed metrics.
    """

    cls_name = "TelemetryEvaluatorInterface"

    def __init__(self) -> None:
        if ABC in type(self).__bases__:
            raise TypeError(f"Cannot instantiate abstract class {type(self).__name__}")

    @abstractmethod
    def evaluate(self, data: Any, *, now_ts: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
        """Compute and return pure quantitative metrics for ``data`` at ``now_ts``."""


class PriceSeriesEvaluatorBase(TelemetryEvaluatorInterface, ABC):
    """Abstract base contract (Level 2) for price-series, volume-profile, swing-wave, and trend evaluators."""

    cls_name = "PriceSeriesEvaluatorBase"


class VolumeProfileEvaluator(PriceSeriesEvaluatorBase):
    """Concrete evaluator (Level 3) for 24h TPO / Volume Profile (VWAP, POC, VAH, VAL) and horizon range position."""

    cls_name = "VolumeProfileEvaluator"

    def evaluate(
        self,
        data: Any,
        *,
        now_ts: Optional[float] = None,
        cur_px: Optional[float] = None,
        decimals: Optional[int] = None,
        num_bins: int = 24,
        value_area_pct: float = 0.70,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return compute_volume_profile_metrics(
            data,
            num_bins=num_bins,
            value_area_pct=value_area_pct,
            decimals=decimals,
        )


class Horizon24hPriceEvaluator(PriceSeriesEvaluatorBase):
    """Concrete evaluator (Level 3) for 24h multi-horizon price changes, tape, volatility, and volume profile."""

    cls_name = "Horizon24hPriceEvaluator"

    def evaluate(self, data: Any, *, now_ts: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
        return compute_24h_price_metrics(data)


class Structural7dWaveEvaluator(PriceSeriesEvaluatorBase):
    """Concrete evaluator (Level 3) for 7d structural range, swing-wave extrema, and BTC cross-asset beta."""

    cls_name = "Structural7dWaveEvaluator"

    def evaluate(
        self,
        data: Any,
        *,
        now_ts: Optional[float] = None,
        cur_px: Optional[float] = None,
        decimals: Optional[int] = None,
        btc_m24: Optional[Dict[str, Any]] = None,
        sym_m24: Optional[Dict[str, Any]] = None,
        btc_vals_7d: Optional[List[float]] = None,
        btc_cur_px: Optional[float] = None,
        symbol: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        out = compute_7d_range_metrics(data, decimals=decimals)
        if symbol and (btc_m24 or btc_vals_7d):
            out.update(
                compute_cross_btc_metrics(
                    symbol,
                    sym_m24 or {},
                    btc_m24 or {},
                    sym_vals_7d=data if isinstance(data, list) else None,
                    btc_vals_7d=btc_vals_7d,
                    sym_cur_px=cur_px,
                    btc_cur_px=btc_cur_px,
                )
            )
        return out


class ShortTrendAndKalmanEvaluator(PriceSeriesEvaluatorBase):
    """Concrete evaluator (Level 3) for 90s/222s/900s/9000s polynomial slopes, Kalman velocity, and live price geometry."""

    cls_name = "ShortTrendAndKalmanEvaluator"

    def evaluate(
        self,
        data: Any,
        *,
        now_ts: Optional[float] = None,
        entry: Optional[Dict[str, Any]] = None,
        short_trend_item: Optional[Dict[str, Any]] = None,
        live_24trend_doc: Optional[Dict[str, Any]] = None,
        live_px: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target = entry if isinstance(entry, dict) else {}
        if isinstance(short_trend_item, dict):
            apply_short_trend_item(target, short_trend_item)
        if isinstance(live_24trend_doc, dict):
            apply_live_24trend_windows(target, live_24trend_doc)
        if isinstance(data, list) and len(data) >= 2:
            if not any(target.get(k) is not None for k in ("slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h")):
                target.update(compute_short_slopes_from_series(data))
            if target.get("kalman_vel") is None:
                if kal := compute_kalman_from_series(data):
                    target.update(kal)
        if live_px is not None and live_px > 0.0:
            target["price"] = live_px
            recalculate_live_price_geometry(target, live_px)
        return target


class TelemetryEvaluatorFactory(OrderGuardEvaluatorFactory):
    """Singleton registry and factory for stateless telemetry evaluators (modeled on ``CacheFactory``)."""

    _instances: Dict[str, TelemetryEvaluatorInterface] = {}
    _registry: Dict[str, type[TelemetryEvaluatorInterface]] = {}
    _defaults_loaded: bool = False
    _unknown_id_label: str = "telemetry evaluator"
    _default_evaluator_modules: tuple[str, ...] = (
        "intelligence.sentiment.evaluators.microstructure_evaluator",
        "intelligence.sentiment.evaluators.portfolio_evaluator",
        "intelligence.sentiment.evaluators.tca_risk_evaluator",
        "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
        "intelligence.sentiment.evaluators.prompt_view_evaluator",
    )


for _ps_eval_id, _ps_eval_cls in (
    ("price_volume_profile", VolumeProfileEvaluator),
    ("price_horizon_24h", Horizon24hPriceEvaluator),
    ("price_structural_7d_wave", Structural7dWaveEvaluator),
    ("price_short_trend_kalman", ShortTrendAndKalmanEvaluator),
):
    TelemetryEvaluatorFactory.register(_ps_eval_id, _ps_eval_cls)


