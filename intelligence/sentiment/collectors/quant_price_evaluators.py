"""Stateless quantitative price-series, volume-profile, swing-wave, Kalman, and multi-horizon trend evaluators."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import canonical_base_asset
from intelligence.internal.state.volatility import adaptive_thresholds, calculate_volatility_1h
from intelligence.internal.triggers.kalman_trigger import KalmanTrend
from intelligence.macro_sentiment_ssot import safe_float, safe_pos_float


def round_opt_px(val: Any, decimals: int = 6) -> Optional[float]:
    """Round a positive price anchor to ``decimals``, returning None when absent or non-positive."""
    vf = safe_pos_float(val, default=None)
    return round(vf, decimals) if vf is not None else None


def _auto_px_decimals(px: float, decimals: Optional[int] = None) -> int:
    return decimals if decimals is not None else (2 if px >= 100.0 else (4 if px >= 1.0 else 6))


def _parse_ts_px_row(row: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    tf, pf = safe_pos_float(row[0], 0.0), safe_pos_float(row[1], 0.0)
    return ((tf * 1000.0 if tf < 1e11 else tf), pf) if (tf > 0.0 and pf > 0.0) else None


def _prices_diverged(px_a: Any, px_b: Any, threshold: float = 0.10) -> bool:
    """Return True when two positive prices diverge by more than ``threshold`` relative to their minimum."""
    a_f, b_f = safe_pos_float(px_a, default=0.0) or 0.0, safe_pos_float(px_b, default=0.0) or 0.0
    return (abs(a_f - b_f) / min(a_f, b_f) > threshold) if (a_f > 0.0 and b_f > 0.0) else False


def pct_delta(cur_px: float, ref_px: float, decimals: int = 2) -> float:
    """Compute percentage delta `((cur_px / ref_px) - 1.0) * 100.0` rounded to `decimals`."""
    return round((cur_px / ref_px - 1.0) * 100.0, decimals)


def range_position_pct(cur_px: float, low_px: float, high_px: float, decimals: int = 1) -> float:
    """Compute normalized [0..100] range position `((cur_px - low_px) / (high_px - low_px)) * 100.0`."""
    return round(((cur_px - low_px) / (high_px - low_px)) * 100.0, decimals) if high_px > low_px else 50.0


def _build_vp_position_dict(cur_px: float, poc_px: float, val_px: float, vah_px: float) -> Dict[str, Any]:
    """Compute symmetric % distances from POC/VAL/VAH, Value Area position %, and VA regime."""
    if min(cur_px, poc_px, val_px, vah_px) <= 0.0:
        return {}
    va_pos = range_position_pct(cur_px, val_px, vah_px)
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


def _compute_horizon_range_dict(cur_px: float, low_px: float, high_px: float, suffix: str) -> Dict[str, Any]:
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
    series: list, *, num_bins: int = 24, value_area_pct: float = 0.70, decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute realized session Volume Profile (Point of Control POC, Value Area High VAH, Value Area Low VAL, symmetric % distances, VA position, and regime)."""
    if not isinstance(series, list) or len(series) < 4:
        return {}
    try:
        timed_rows: list[Tuple[float, float, float]] = []
        scalar_rows: list[float] = []
        for row in series:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                if (pt := _parse_ts_px_row(row)) is not None:
                    timed_rows.append((pt[0], pt[1], safe_pos_float(row[2], 0.0) if len(row) >= 3 else 0.0))
            elif not isinstance(row, (list, tuple)) and (px_f := safe_pos_float(row, 0.0)) > 0.0:
                scalar_rows.append(px_f)

        samples: list[Tuple[float, float]] = []
        prev_px: Optional[float] = None
        raw_pairs = [(px_f, vol_w) for _, px_f, vol_w in sorted(timed_rows, key=lambda x: x[0])] if timed_rows else [(px_f, 0.0) for px_f in scalar_rows]
        has_real_vol = sum(1 for _, vol_w in raw_pairs if vol_w > 0.0) >= max(4, len(raw_pairs) // 4)
        for px_f, vol_w in raw_pairs:
            if vol_w <= 0.0:
                vol_w = 1.0 + min(5.0, (abs(px_f - prev_px) / prev_px) * 1000.0) if prev_px else 1.0
            prev_px = px_f
            samples.append((px_f, vol_w))

        if len(samples) < 4:
            return {}
        if len(samples) > 4000:
            downsampled = samples[:: max(1, len(samples) // 2000)]
            samples = downsampled + ([samples[-1]] if downsampled[-1][0] != samples[-1][0] else [])

        cur_px = samples[-1][0]
        low_px, high_px = min(p for p, _ in samples), max(p for p, _ in samples)
        n_bins = max(8, min(64, int(num_bins)))
        bin_w = (high_px - low_px) / float(n_bins) if high_px > low_px and low_px > 0.0 else 0.0
        if bin_w <= 0.0:
            return {}

        bin_vols, bin_px_sums, total_vol = [0.0] * n_bins, [0.0] * n_bins, 0.0
        for px_v, w_v in samples:
            b_idx = min(n_bins - 1, max(0, int((px_v - low_px) / bin_w)))
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
        sep_cands = [i for i in range(n_bins) if abs(i - poc_idx) >= 2 and bin_vols[i] > 0.0]
        if sep_cands:
            hvn2_idx = max(sep_cands, key=lambda i: (smoothed_vols[i], bin_vols[i]))
            if smoothed_vols[poc_idx] > 0.0 and smoothed_vols[hvn2_idx] >= 0.60 * smoothed_vols[poc_idx]:
                hvn2_raw = (
                    bin_px_sums[hvn2_idx] / bin_vols[hvn2_idx]
                    if bin_vols[hvn2_idx] > 0.0
                    else (low_px + (hvn2_idx + 0.5) * bin_w)
                )
                lo_b, hi_b = min(poc_idx, hvn2_idx), max(poc_idx, hvn2_idx)
                valley_indices = list(range(lo_b + 1, hi_b))
                if valley_indices:
                    lvn_idx = min(valley_indices, key=lambda i: (smoothed_vols[i], bin_vols[i]))
                    lvn_raw = (
                        bin_px_sums[lvn_idx] / bin_vols[lvn_idx]
                        if bin_vols[lvn_idx] > 0.0
                        else (low_px + (lvn_idx + 0.5) * bin_w)
                    )
                    eff_dec_tmp = _auto_px_decimals(cur_px, decimals)
                    hvn2_px = round(hvn2_raw, eff_dec_tmp)
                    lvn_px = round(lvn_raw, eff_dec_tmp)
                    if (
                        smoothed_vols[hvn2_idx] >= 0.68 * smoothed_vols[poc_idx]
                        and smoothed_vols[lvn_idx] <= 0.90 * min(smoothed_vols[poc_idx], smoothed_vols[hvn2_idx])
                    ):
                        profile_shape = "BIMODAL"
                    if smoothed_vols[hvn2_idx] >= 0.85 * smoothed_vols[poc_idx]:
                        ref_center = 0.5 * (cur_px + (0.5 * (low_px + high_px)))
                        if abs(ref_center - hvn2_raw) < abs(ref_center - poc_raw):
                            poc_raw, hvn2_raw = hvn2_raw, poc_raw
                            poc_idx, hvn2_idx = hvn2_idx, poc_idx
                            hvn2_px = round(hvn2_raw, eff_dec_tmp)
        target_vol = total_vol * max(0.50, min(0.90, float(value_area_pct)))
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
        poc_px = round(poc_raw, eff_dec)
        val_px = round(min(poc_raw, low_px + val_idx * bin_w), eff_dec)
        vah_px = round(max(poc_raw, low_px + (vah_idx + 1) * bin_w), eff_dec)
        if val_px >= poc_px and poc_raw > low_px:
            val_px = round(max(low_px, poc_raw - max(0.5 * bin_w, 10.0 ** (-eff_dec))), eff_dec)
        if vah_px <= poc_px and poc_raw < high_px:
            vah_px = round(min(high_px, poc_raw + max(0.5 * bin_w, 10.0 ** (-eff_dec))), eff_dec)
        if vah_px <= val_px:
            vah_px, val_px = round(high_px, eff_dec), round(low_px, eff_dec)
        out_vp: Dict[str, Any] = {
            "poc_px": poc_px,
            "vah_px": vah_px,
            "val_px": val_px,
            "vp_source": "REAL_VOL" if has_real_vol else "TPO",
            "profile_shape": profile_shape,
            **_build_vp_position_dict(cur_px, poc_px, val_px, vah_px),
        }
        if hvn2_px is not None:
            out_vp["hvn2_px"] = hvn2_px
            out_vp["sec_poc_px"] = hvn2_px
        if lvn_px is not None:
            out_vp["lvn_px"] = lvn_px
        if "sec_poc_px" not in out_vp:
            sec_cands = [i for i in range(n_bins) if abs(i - poc_idx) >= 2 and smoothed_vols[i] >= smoothed_vols[poc_idx] * 0.70 and bin_vols[i] > 0.0]
            if sec_cands:
                sec_idx = max(sec_cands, key=lambda i: (smoothed_vols[i], bin_vols[i]))
                sec_raw = bin_px_sums[sec_idx] / bin_vols[sec_idx] if bin_vols[sec_idx] > 0 else (low_px + (sec_idx + 0.5) * bin_w)
                out_vp["sec_poc_px"] = round(sec_raw, eff_dec)
        return out_vp
    except Exception:
        return {}


def compute_24h_price_metrics(series: list) -> Dict[str, Any]:
    """Compute multi-horizon percentage changes (15m, 30m, 1h, 3h) and 24h range from a cache_24_price series."""
    if not isinstance(series, list) or len(series) < 2:
        return {}
    try:
        clean_series = sorted((pt for row in series if (pt := _parse_ts_px_row(row)) is not None), key=lambda x: x[0])
        if len(clean_series) < 2:
            return {}
        now_ms, cur_px = clean_series[-1]
        span_ms = now_ms - clean_series[0][0]

        def _sample_at(mins: float) -> Tuple[Optional[float], float]:
            target = now_ms - mins * 60000.0
            return next(((tms, p_val) for tms, p_val in reversed(clean_series) if tms <= target), (None, clean_series[0][1]))

        def _has_sample_near(mins: float, max_gap_mins: float) -> bool:
            tms, _ = _sample_at(mins)
            return tms is not None and tms >= now_ms - (mins + max_gap_mins) * 60000.0

        out = _compute_horizon_range_dict(cur_px, min(x[1] for x in clean_series), max(x[1] for x in clean_series), "24h")
        for key, mins, min_span_ms, max_gap_mins in (
            ("chg_1m_pct", 1.0, 60000.0, 2.5), ("chg_3m_pct", 3.0, 180000.0, 4.0), ("chg_5m_pct", 5.0, 300000.0, 6.0),
            ("chg_15m_pct", 15.0, 0.0, 0.0), ("chg_30m_pct", 30.0, 0.0, 0.0), ("chg_1h_pct", 60.0, 0.0, 0.0),
            ("chg_3h_pct", 180.0, 0.0, 0.0), ("chg_6h_pct", 360.0, 3600000.0 * 4, 0.0),
            ("chg_12h_pct", 720.0, 3600000.0 * 9, 0.0), ("chg_24h_pct", 1440.0, 3600000.0 * 18, 0.0),
        ):
            _, p_ref = _sample_at(mins)
            if p_ref > 0 and span_ms >= min_span_ms and (max_gap_mins <= 0.0 or _has_sample_near(mins, max_gap_mins)):
                out[key] = pct_delta(cur_px, p_ref)

        recent_15m = [p_v for tms, p_v in clean_series if tms >= now_ms - 15.0 * 60000.0]
        if len(recent_15m) >= 3:
            l15, h15 = min(recent_15m), max(recent_15m)
            out["low_15m"], out["high_15m"] = l15, h15
            if h15 > l15 > 0:
                out["pos_15m_pct"] = range_position_pct(cur_px, l15, h15)
                out["span_15m_pct"] = pct_delta(h15, l15)
            tape_pts = [
                f"{lbl}:{(cur_px if m_off == 0.0 else _sample_at(m_off)[1]):.6g}"
                for m_off, lbl in ((15.0, "-15m"), (10.0, "-10m"), (5.0, "-5m"), (3.0, "-3m"), (1.0, "-1m"), (0.0, "now"))
                if m_off == 0.0 or (span_ms >= m_off * 60000.0 and _has_sample_near(m_off, max(3.0, m_off * 0.5)))
            ]
            if len(tape_pts) >= 3:
                out["recent_tape_15m"] = tape_pts
        if span_ms >= 3600000.0 * 2.0:
            win_6h = [(tms, p_v) for tms, p_v in clean_series if tms >= now_ms - 360.0 * 60000.0]
            cutoff_1h = now_ms - 60.0 * 60000.0
            prior_pts = [p_v for tms, p_v in win_6h if tms < cutoff_1h]
            recent_pts = [p_v for tms, p_v in win_6h if tms >= cutoff_1h]
            if len(prior_pts) >= 2 and len(recent_pts) >= 2:
                high_6h, prior_low, recent_low = max(p_v for _, p_v in win_6h), min(prior_pts), min(recent_pts)
                low_6h = min(prior_low, recent_low)
                dd_low_6h = ((low_6h / high_6h) - 1.0) * 100.0 if high_6h > 0.0 else 0.0
                pos_6h = ((cur_px - low_6h) / (high_6h - low_6h)) if high_6h > low_6h else 1.0
                if dd_low_6h <= -2.0 and pos_6h <= 0.65 and prior_low > 0.0 and recent_low > 0.0:
                    eff_dec = _auto_px_decimals(cur_px)
                    if recent_low < prior_low and cur_px > prior_low:
                        out["bottom_structure_6h"] = "LIQUIDITY_SWEEP_RECLAIM"
                        out["reclaim_above_prior_low_pct"] = pct_delta(cur_px, prior_low)
                    elif recent_low >= prior_low and ((recent_low / prior_low) - 1.0) * 100.0 <= 4.5 and cur_px > recent_low * 1.002:
                        out["bottom_structure_6h"] = "HIGHER_LOW_DOUBLE_BOTTOM"
                        out["higher_low_vs_prior_pct"] = pct_delta(recent_low, prior_low)
                    elif recent_low <= prior_low and cur_px <= prior_low and ((cur_px / recent_low) - 1.0) * 100.0 <= 2.5:
                        out["bottom_structure_6h"] = "FRESH_6H_LOW_TEST"
                    if "bottom_structure_6h" in out:
                        out["prior_6h_low_px"], out["recent_1h_low_px"] = round(prior_low, eff_dec), round(recent_low, eff_dec)
        out.update(compute_volume_profile_metrics(series))
        return out
    except Exception:
        return {}


def compute_cross_btc_metrics(alt_series: Any, btc_series: Any) -> Dict[str, Any]:
    """Compute 24h ALT/BTC ratio range position, drawdown/bounce, and realized beta/correlation vs BTC."""
    if not isinstance(alt_series, list) or not isinstance(btc_series, list) or len(alt_series) < 4 or len(btc_series) < 4:
        return {}
    try:
        def _clean_buckets(series: list) -> Dict[int, float]:
            return {int(round(pt[0] / 300000.0)): pt[1] for row in series if (pt := _parse_ts_px_row(row)) is not None}

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

        step = 3 if len(aligned) >= 18 else 1
        rets_a, rets_b = [], []
        for (bk0, a0, b0), (bk1, a1, b1) in zip(aligned, aligned[step:]):
            if 0 < (bk1 - bk0) <= 48 and a0 > 0 and b0 > 0:
                rets_a.append(a1 / a0 - 1.0)
                rets_b.append(b1 / b0 - 1.0)
        if len(rets_b) < 6 and step > 1:
            rets_a, rets_b = [], []
            for (bk0, a0, b0), (bk1, a1, b1) in zip(aligned, aligned[1:]):
                if 0 < (bk1 - bk0) <= 48 and a0 > 0 and b0 > 0:
                    rets_a.append(a1 / a0 - 1.0)
                    rets_b.append(b1 / b0 - 1.0)
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


def compute_swing_wave_metrics(
    series_or_vals: list[Any], *, reversal_pct: float = 3.0, decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract raw ZigZag swing waves (>=3% reversals), active swing low/high, and rally/pullback % from a price series."""
    if not isinstance(series_or_vals, list) or len(series_or_vals) < 2:
        return {}
    vals = [
        v for item in series_or_vals
        if (v := safe_float(item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item, 0.0)) > 0.0
    ]
    if len(vals) < 2:
        return {}

    cur_v, thr = vals[-1], max(0.5, float(reversal_pct)) / 100.0
    pivots: list[Tuple[str, int, float]] = []
    direction, hi_px, hi_idx, lo_px, lo_idx = 0, vals[0], 0, vals[0], 0

    for idx, v in enumerate(vals[1:], start=1):
        if direction == 0:
            if v > hi_px:
                hi_px, hi_idx = v, idx
            if v < lo_px:
                lo_px, lo_idx = v, idx
            if lo_px > 0 and (v - lo_px) / lo_px >= thr and lo_idx < idx:
                pivots.append(("LOW", lo_idx, lo_px))
                direction, hi_px, hi_idx = 1, v, idx
            elif hi_px > 0 and (hi_px - v) / hi_px >= thr and hi_idx < idx:
                pivots.append(("HIGH", hi_idx, hi_px))
                direction, lo_px, lo_idx = -1, v, idx
        elif direction == 1:
            if v >= hi_px:
                hi_px, hi_idx = v, idx
            elif hi_px > 0 and (hi_px - v) / hi_px >= thr:
                pivots.append(("HIGH", hi_idx, hi_px))
                direction, lo_px, lo_idx = -1, v, idx
        else:
            if v <= lo_px:
                lo_px, lo_idx = v, idx
            elif lo_px > 0 and (v - lo_px) / lo_px >= thr:
                pivots.append(("LOW", lo_idx, lo_px))
                direction, hi_px, hi_idx = 1, v, idx

    chain: list[Tuple[str, int, float]] = list(pivots)
    if direction == 1 and chain and chain[-1][2] > 0 and (hi_px - chain[-1][2]) / chain[-1][2] >= thr:
        chain.append(("HIGH", hi_idx, hi_px))
    elif direction == -1 and chain and chain[-1][2] > 0 and (chain[-1][2] - lo_px) / chain[-1][2] >= thr:
        chain.append(("LOW", lo_idx, lo_px))

    swing_legs_pct = [pct_delta(p1[2], p0[2]) for p0, p1 in zip(chain, chain[1:]) if p0[2] > 0]
    major_low_px: Optional[float] = None
    if direction == 1 and pivots:
        swing_low_px, swing_high_px = pivots[-1][2], max(hi_px, cur_v)
        if len(pivots) >= 3 and pivots[-3][0] == "LOW" and pivots[-3][2] < swing_low_px:
            major_low_px = pivots[-3][2]
    elif direction == -1 and pivots:
        swing_high_px, swing_low_px = pivots[-1][2], min(lo_px, cur_v)
        if len(pivots) >= 2 and pivots[-2][0] == "LOW" and pivots[-2][2] < swing_low_px:
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
    """Compute 7-day range geometry (low, high, drawdown, bounce, span, position) and 3%–8% swing waves from a price series."""
    if not isinstance(vals, list) or len(vals) < 2 or vals[0] <= 0:
        return {}
    eff_dec = decimals if decimals is not None else 6
    cur_v = vals[-1]
    return {
        "chg_7d_pct": pct_delta(cur_v, vals[0]),
        **_compute_horizon_range_dict(cur_v, round(min(vals), eff_dec), round(max(vals), eff_dec), "7d"),
        **compute_swing_wave_metrics(vals, decimals=eff_dec),
    }


_WINDOW_TREND_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    ("90", "slope_90s", "trend_90s", "growth_90s"),
    ("222", "slope_3_7m", "trend_3_7m", "growth_3_7m"),
    ("900", "slope_15m", "trend_15m", "growth_15m"),
    ("9000", "slope_2_5h", "trend_2_5h", "growth_2_5h"),
)


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
    wins = {wk: (dyn.get(wk) if isinstance(dyn.get(wk), dict) else {}) for wk, *_ in _WINDOW_TREND_SPECS}
    if wins["222"].get("slope_full") is None:
        wins["222"] = tinfo if tinfo.get("slope_full") is not None else (wins["90"] if wins["90"].get("slope_full") is not None else wins["222"])
    if require_any_slope and not any(x.get("slope_full") is not None for x in (wins["222"], wins["900"], wins["9000"], tinfo)):
        return False
    cur_px = safe_pos_float(fallback_price, default=None) or safe_pos_float(tinfo.get("current_price"), default=None)
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
    t_vol, t_re = safe_pos_float(tinfo.get("vol_1h_pct"), default=None), safe_pos_float(tinfo.get("adapt_reentry_pct"), default=None)
    if t_vol is None or (t_re is not None and abs(t_re - 2.0 * t_vol) <= 0.05):
        for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
            if tinfo.get(ak) is not None:
                entry[ak] = tinfo[ak]
    for wk, s_key, t_key, g_key in _WINDOW_TREND_SPECS:
        w_dict = wins[wk]
        is_90 = wk == "90"
        if w_dict.get("slope_full") is not None or (require_any_slope and not is_90):
            entry[s_key] = w_dict.get("slope_full", 0.0) if (require_any_slope and not is_90) else w_dict.get("slope_full")
            entry[t_key] = w_dict.get("final_trend", trend)
        if w_dict.get("growth_coefficient") is not None:
            entry[g_key] = w_dict.get("growth_coefficient")
    return True


def apply_live_24trend_windows(entry: Dict[str, Any], lt_win: Dict[str, Any], fallback_price: Any = None) -> None:
    """Overlay fresh 1-second sampled dynamic windows (90s, 3.7m, 15m, 2.5h) from `cachedb/24trend/live_<SYM>.json`."""
    wins = {wk: (lt_win.get(wk) if isinstance(lt_win.get(wk), dict) else {}) for wk, *_ in _WINDOW_TREND_SPECS}
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
    if entry.get("gradient_recent") is None and w222.get("gradient_recent") is not None:
        entry["gradient_recent"] = w222.get("gradient_recent")
    if entry.get("gradient_recent_fast") is None:
        if (gf_live := w90.get("gradient_recent_fast") or w222.get("gradient_recent_fast") or w90.get("gradient_recent")) is not None:
            entry["gradient_recent_fast"] = gf_live
    if entry.get("epsilon") is None and (eps_live := w222.get("epsilon") or w90.get("epsilon")) is not None:
        entry["epsilon"] = eps_live
    for wk, s_key, t_key, g_key in _WINDOW_TREND_SPECS:
        w_dict = wins[wk]
        if not w_dict:
            continue
        if wk != "90":
            entry[s_key] = w_dict.get("slope_full", entry.get(s_key, 0.0))
            entry[t_key] = w_dict.get("final_trend", entry.get(t_key, 0))
        else:
            if w_dict.get("slope_full") is not None:
                entry[s_key] = w_dict.get("slope_full")
            if w_dict.get("final_trend") is not None:
                entry[t_key] = w_dict.get("final_trend")
        if w_dict.get("growth_coefficient") is not None:
            entry[g_key] = w_dict.get("growth_coefficient")


def _compute_kalman_from_series(series_24h: List[Any]) -> Optional[Dict[str, Any]]:
    """Derive Kalman velocity (% per min), std, and trend from a 24h price series when cache_kalman_trend lacks the symbol."""
    pts = [(parsed[0] / 1000.0, parsed[1]) for row in series_24h if (parsed := _parse_ts_px_row(row)) is not None]
    if len(pts) < 5:
        return None
    cutoff_s = pts[-1][0] - 7200.0
    subsampled: List[Tuple[float, float]] = []
    last_ts = -1e18
    for ts_s, px_v in pts:
        if ts_s >= cutoff_s and ts_s - last_ts >= 55.0:
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
    return {"kalman_vel": last_out.get("vel"), "kalman_vel_std": last_out.get("vel_std"), "kalman_trend": last_out.get("trend")}


def _compute_short_slopes_from_series(series_24h: List[Any]) -> Dict[str, Any]:
    """Derive fallback multi-horizon slopes (90s, 3.7m, 15m, 2.5h) and growth coefficients from a 24h series."""
    pts = [(parsed[0] / 1000.0, parsed[1]) for row in series_24h if (parsed := _parse_ts_px_row(row)) is not None and parsed[1] > 0]
    if len(pts) < 3:
        return {}
    t_now, p_now = pts[-1]
    out: Dict[str, Any] = {}
    for win_sec, win_min, s_key, t_key, g_key in (
        (90.0, 1.5, "slope_90s", "trend_90s", "growth_90s"),
        (222.0, 3.7, "slope_3_7m", "trend_3_7m", "growth_3_7m"),
        (900.0, 15.0, "slope_15m", "trend_15m", "growth_15m"),
        (9000.0, 150.0, "slope_2_5h", "trend_2_5h", "growth_2_5h"),
    ):
        target_t = t_now - win_sec
        ref_t, ref_p = min(pts[:-1], key=lambda item: abs(item[0] - target_t))
        dt_min = max(win_min, (t_now - ref_t) / 60.0)
        if ref_p > 0 and dt_min > 0:
            slope_v = round(((p_now - ref_p) / ref_p) / dt_min, 6) + 0.0
            g_v = round(slope_v * win_min * 1000.0, 2) + 0.0
            out[s_key] = slope_v
            out[g_key] = g_v
            out[t_key] = 1 if g_v >= 0.25 else (-1 if g_v <= -0.25 else 0)
    if "slope_3_7m" in out:
        out["gradient_recent_fast"] = out["slope_3_7m"]
    return out


def _extract_ohlc_series_points(
    ohlc_obj: Any,
    *,
    min_ts: float = 0.0,
    max_ts: float = float("inf"),
    require_positive_volume: bool = False,
) -> List[List[Any]]:
    """Extract `[ts_ms, close, (vol)]` rows from an OHLC series within `[min_ts, max_ts]`."""
    vols = getattr(ohlc_obj, "volumes", ()) or ()
    has_vols = isinstance(vols, (list, tuple)) and bool(vols)
    pts: List[List[Any]] = []
    for idx, (ts, px) in enumerate(zip(getattr(ohlc_obj, "timestamps", ()), getattr(ohlc_obj, "closes", ()))):
        if min_ts <= ts <= max_ts and float(px) > 0:
            v_f = safe_pos_float(vols[idx], 0.0) or 0.0 if (has_vols and idx < len(vols)) else 0.0
            if require_positive_volume and v_f <= 0.0:
                continue
            pts.append([int(ts * 1000), float(px), v_f] if v_f > 0.0 else [int(ts * 1000), float(px)])
    return pts


def _load_stitched_24h_ohlc_points(
    ohlc_fn: Any,
    sym: str,
    cutoff_24h: float,
    *,
    require_positive_volume: bool = False,
) -> List[List[Any]]:
    """Load 5m OHLC points since `cutoff_24h` and prepend older 15m OHLC points when span < 18h."""
    pts = _extract_ohlc_series_points(ohlc_fn(sym, 5), min_ts=cutoff_24h, require_positive_volume=require_positive_volume)
    if not pts or (pts[-1][0] - pts[0][0]) < 18 * 3600 * 1000:
        first_s5_ts = (pts[0][0] / 1000.0 - 300.0) if pts else float("inf")
        pts = _extract_ohlc_series_points(
            ohlc_fn(sym, 15), min_ts=cutoff_24h, max_ts=first_s5_ts, require_positive_volume=require_positive_volume
        ) + pts
    return pts


def recalculate_live_price_geometry(pinfo: Dict[str, Any], p_cur: float) -> None:
    """Recalculate VP position, 24h/7d/15m range bounds, swing waves, and 6h bottom structure for live price `p_cur`."""
    if not isinstance(pinfo, dict) or p_cur <= 0.0:
        return
    poc_v, val_v, vah_v = (safe_pos_float(pinfo.get(k), default=0.0) or 0.0 for k in ("poc_px", "val_px", "vah_px"))
    if poc_v > 0.0 and val_v > 0.0 and vah_v > val_v:
        pinfo.update(_build_vp_position_dict(p_cur, poc_v, val_v, vah_v))
    l24 = safe_pos_float(pinfo.get("low_24h"), default=None)
    h24 = safe_pos_float(pinfo.get("high_24h"), default=None)
    if l24 is not None and h24 is not None:
        l24, h24 = round(min(l24, p_cur), 6), round(max(h24, p_cur), 6)
        pinfo.update(_compute_horizon_range_dict(p_cur, l24, h24, "24h"))
    l7d = safe_pos_float(pinfo.get("low_7d"), default=None)
    h7d = safe_pos_float(pinfo.get("high_7d"), default=None)
    if l7d is not None and h7d is not None:
        eff_l7d = min(l7d, p_cur, *([l24] if l24 is not None else []))
        eff_h7d = max(h7d, p_cur, *([h24] if h24 is not None else []))
        pinfo.update(_compute_horizon_range_dict(p_cur, round(eff_l7d, 6), round(eff_h7d, 6), "7d"))
    sw_lo = safe_pos_float(pinfo.get("swing_low_px"), default=None)
    sw_hi = safe_pos_float(pinfo.get("swing_high_px"), default=None)
    if sw_lo is not None and sw_hi is not None:
        sw_lo_new, sw_hi_new = round(min(sw_lo, p_cur), 6), round(max(sw_hi, p_cur), 6)
        pinfo["swing_low_px"], pinfo["swing_high_px"] = sw_lo_new, sw_hi_new
        if sw_lo_new > 0.0:
            pinfo["rally_from_swing_low_pct"] = pct_delta(p_cur, sw_lo_new)
        if sw_hi_new > 0.0:
            pinfo["pullback_from_swing_high_pct"] = pct_delta(p_cur, sw_hi_new)
    if (maj_lo := safe_pos_float(pinfo.get("major_swing_low_px"), default=None)) is not None and maj_lo > 0.0:
        maj_lo_new = round(min(maj_lo, p_cur), 6)
        pinfo["major_swing_low_px"] = maj_lo_new
        pinfo["rally_from_major_low_pct"] = pct_delta(p_cur, maj_lo_new)
    r1h_lo = safe_pos_float(pinfo.get("recent_1h_low_px"), default=None)
    p6h_lo = safe_pos_float(pinfo.get("prior_6h_low_px"), default=None)
    if r1h_lo is not None and p6h_lo is not None and p6h_lo > 0.0:
        eff_dec_b = _auto_px_decimals(p_cur)
        r1h_new = min(r1h_lo, p_cur)
        pinfo["recent_1h_low_px"] = round(r1h_new, eff_dec_b)
        b_struct = pinfo.get("bottom_structure_6h")
        if b_struct == "HIGHER_LOW_DOUBLE_BOTTOM":
            if r1h_new < p6h_lo and p_cur > p6h_lo:
                pinfo["bottom_structure_6h"] = "LIQUIDITY_SWEEP_RECLAIM"
                pinfo.pop("higher_low_vs_prior_pct", None)
                pinfo["reclaim_above_prior_low_pct"] = pct_delta(p_cur, p6h_lo)
            else:
                pinfo["higher_low_vs_prior_pct"] = pct_delta(r1h_new, p6h_lo)
        elif b_struct == "LIQUIDITY_SWEEP_RECLAIM":
            pinfo["reclaim_above_prior_low_pct"] = pct_delta(p_cur, p6h_lo)
    if isinstance(tape_15m := pinfo.get("recent_tape_15m"), list) and tape_15m:
        tape_copy = list(tape_15m)
        if isinstance(tape_copy[-1], str) and tape_copy[-1].startswith("now:"):
            tape_copy[-1] = f"now:{p_cur:.6g}"
            pinfo["recent_tape_15m"] = tape_copy
        l15 = safe_pos_float(pinfo.get("low_15m"), default=None)
        h15 = safe_pos_float(pinfo.get("high_15m"), default=None)
        if l15 is not None and h15 is not None:
            l15_new, h15_new = min(l15, p_cur), max(h15, p_cur)
            pinfo["low_15m"], pinfo["high_15m"] = round(l15_new, 6), round(h15_new, 6)
            if h15_new > l15_new > 0:
                pinfo["pos_15m_pct"] = round(min(100.0, max(0.0, ((p_cur - l15_new) / (h15_new - l15_new)) * 100.0)), 1)


def _apply_24h_series_enrichment(
    sym: str,
    entry: Dict[str, Any],
    m24: Dict[str, Any],
    series_24h: List[Any],
    btc_series_24h: Optional[List[Any]],
) -> None:
    """Enrich a symbol's `prices_and_trends` entry with 24h metrics, cross-BTC beta/corr, Kalman, slopes, and 1h vol."""
    entry.update(m24)
    if series_24h and safe_float(entry.get("price"), 0.0) <= 0.0:
        entry["price"] = series_24h[-1][1]
    cur_e_px = safe_pos_float(entry.get("price"), default=None)
    l24_e, h24_e = safe_pos_float(entry.get("low_24h"), default=None), safe_pos_float(entry.get("high_24h"), default=None)
    l7d_e, h7d_e = safe_pos_float(entry.get("low_7d"), default=None), safe_pos_float(entry.get("high_7d"), default=None)
    if cur_e_px is not None and l24_e is not None and h24_e is not None and l7d_e is not None and h7d_e is not None:
        if l24_e < l7d_e or h24_e > h7d_e:
            eff_dec_7d = _auto_px_decimals(cur_e_px)
            entry.update(_compute_horizon_range_dict(cur_e_px, round(min(l7d_e, l24_e, cur_e_px), eff_dec_7d), round(max(h7d_e, h24_e, cur_e_px), eff_dec_7d), "7d"))
    if entry.get("trend", 0) == 0 and "chg_1h_pct" in m24:
        c1h = safe_float(m24.get("chg_1h_pct"), 0.0)
        if c1h >= 0.25 or c1h <= -0.25:
            entry["trend"] = 1 if c1h >= 0.25 else -1
    is_eq = sym.endswith(("_EQ", "_US_EQ"))
    if series_24h and btc_series_24h and not is_eq and canonical_base_asset(sym) != "BTC":
        cross_btc = compute_cross_btc_metrics(series_24h, btc_series_24h)
        for ck in ("alt_btc_pos_24h_pct", "alt_btc_dd_24h_high_pct", "alt_btc_bounce_24h_low_pct", "beta_btc_24h", "corr_btc_24h"):
            if cross_btc.get(ck) is not None:
                entry[ck] = cross_btc[ck]
    if entry.get("kalman_vel") is None and series_24h and isinstance(k_fb := _compute_kalman_from_series(series_24h), dict):
        for kk, kv in k_fb.items():
            if kv is not None and entry.get(kk) is None:
                entry[kk] = kv
    if not is_eq and series_24h:
        stale_short_cache = (
            entry.get("slope_2_5h") is not None
            and entry.get("slope_15m") is not None
            and abs(float(entry["slope_2_5h"]) - float(entry["slope_15m"])) < 1e-12
            and abs(float(entry.get("growth_2_5h") or 0.0) - float(entry.get("growth_15m") or 0.0)) < 1e-9
        )
        if (entry.get("slope_3_7m") is None or stale_short_cache) and (s_fb := _compute_short_slopes_from_series(series_24h)):
            for sk, sv in s_fb.items():
                if sv is None:
                    continue
                if entry.get(sk) is None or (
                    stale_short_cache
                    and (
                        sk in ("slope_2_5h", "growth_2_5h", "trend_2_5h")
                        or (sk in ("slope_90s", "growth_90s", "trend_90s") and abs(float(entry.get("slope_90s") or 0.0)) < 1e-12)
                    )
                ):
                    entry[sk] = sv
        if (entry.get("vol_1h_pct") is None or stale_short_cache) and len(series_24h) >= 20:
            all_pts_m = [pt for r in series_24h if (pt := _parse_ts_px_row(r)) is not None]
            tail_pts_m: List[Tuple[float, float]] = []
            if len(all_pts_m) >= 10:
                cutoff_1h_ms = float(all_pts_m[-1][0]) - 3600.0 * 1000.0
                last_t_ms = -1e18
                for t_ms, px_v in all_pts_m:
                    if t_ms >= cutoff_1h_ms and (t_ms - last_t_ms) >= 55000.0:
                        tail_pts_m.append((t_ms, px_v))
                        last_t_ms = t_ms
                if len(tail_pts_m) < 10:
                    tail_pts_m = all_pts_m[-60:]
            if len(tail_pts_m) >= 10:
                dt_s_m = max(60.0, (float(tail_pts_m[-1][0]) - float(tail_pts_m[0][0])) / (1000.0 * max(1, len(tail_pts_m) - 1)))
                if (v1h_m := calculate_volatility_1h([float(r[1]) for r in tail_pts_m], dt_s_m)) is not None and v1h_m > 0:
                    entry["vol_1h_pct"] = v1h_m
                    re_th_m, dca_th_m = adaptive_thresholds(v1h_m)
                    if re_th_m is not None and (entry.get("adapt_reentry_pct") is None or stale_short_cache):
                        entry["adapt_reentry_pct"] = re_th_m
                    if dca_th_m is not None and (entry.get("adapt_dca_pct") is None or stale_short_cache):
                        entry["adapt_dca_pct"] = dca_th_m
