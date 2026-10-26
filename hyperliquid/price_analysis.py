#!/usr/bin/env python3
"""
price_analysis.py — port of binance/priceAnalysis.py trend analysis to Hyperliquid data.

Detect trends over TIME-defined windows rather than point counts. Use linear-regression
slope in each window and walk backward while its sign persists within noise tolerance.
Return direction, duration, and estimated continuation near half the past duration.

This runs on one-hour HL candles and therefore analyzes the real perpetual price.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from common import load_env_stack, required_int_env

load_env_stack(os.path.join(_HERE, ".env"))

MIN_POINTS_PER_WINDOW = 3


def _fast_linear_slope_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS linear slope in O(N) without LAPACK/SVD matrix overhead."""
    dx = x - x[0]
    mean_dx = float(np.mean(dx))
    dev_x = dx - mean_dx
    var_x = float(np.dot(dev_x, dev_x))
    if var_x == 0.0:
        return 0.0
    return float(np.dot(dev_x, y) / var_x)


def detect_long_term_trend(timestamps, prices, window_hours=24, step_hours=8,
                           min_consecutive_blocks=3, noise_tolerance=2,
                           min_points_per_window=MIN_POINTS_PER_WINDOW):
    """Port priceAnalysis.detect_long_term_trend with gap/density robustness."""
    if timestamps is None or prices is None:
        return None
    if any(isinstance(v, bool) for v in (window_hours, step_hours, min_consecutive_blocks, noise_tolerance, min_points_per_window)):
        return None
    try:
        wh_f = float(window_hours)
        sh_f = float(step_hours)
        mcb_i = int(min_consecutive_blocks)
        nt_i = int(noise_tolerance)
        mpw_i = int(min_points_per_window)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(wh_f) and wh_f > 0.0 and math.isfinite(sh_f) and sh_f > 0.0):
        return None
    if mcb_i < 1 or nt_i < 0 or mpw_i < 2:
        return None
    raw_ts = list(timestamps)
    raw_px = list(prices)
    if len(raw_ts) != len(raw_px) or len(raw_ts) < 2:
        return None
    pairs: list[tuple[float, float]] = []
    for t_item, p_item in zip(raw_ts, raw_px):
        if isinstance(t_item, bool) or isinstance(p_item, bool):
            continue
        try:
            tf = float(t_item)
            pf = float(p_item)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(tf) and tf > 0.0 and math.isfinite(pf) and pf > 0.0:
            pairs.append((tf, pf))
    if len(pairs) < 2:
        return None
    if any(pairs[i][0] > pairs[i + 1][0] for i in range(len(pairs) - 1)):
        pairs.sort(key=lambda item: item[0])
    timestamps = np.asarray([p[0] for p in pairs], dtype=float)
    prices = np.asarray([p[1] for p in pairs], dtype=float)

    t_end, t_first = timestamps[-1], timestamps[0]
    window_sec = wh_f * 3600.0
    step_sec = sh_f * 3600.0
    min_consecutive_blocks = mcb_i
    noise_tolerance = nt_i
    min_points_per_window = mpw_i

    def slope_h(t_lo, t_hi):
        lo = int(np.searchsorted(timestamps, t_lo, "left"))
        hi = int(np.searchsorted(timestamps, t_hi, "left"))
        if hi - lo < min_points_per_window:
            return None, (lo, hi)
        x, y = timestamps[lo:hi], prices[lo:hi]
        s = _fast_linear_slope_1d(x, y)
        if not math.isfinite(s):
            return None, (lo, hi)
        return s * 3600.0, (lo, hi)

    cur, cur_idx = slope_h(t_end - window_sec, t_end + 1.0)
    if cur is None or not math.isfinite(cur):
        return None
    current_sign = np.sign(cur) or 1.0

    blocks = [cur_idx]
    consecutive, noise = 1, 0
    confirm_lo = cur_idx[0]      # oldest point CONFIRMING the current direction
    t_ws = t_end - window_sec - step_sec
    while t_ws >= t_first:
        s, idx = slope_h(t_ws, t_ws + window_sec)
        if s is None:
            break
        if np.sign(s) == current_sign:
            blocks.append(idx); consecutive += 1; noise = 0
            confirm_lo = idx[0]
        elif noise < noise_tolerance:
            noise += 1; blocks.append(idx)           # tentatively allow trend continuation past noise
        else:
            break                                    # noise beyond tolerance ends the trend here
        t_ws -= step_sec

    # Without enough CONFIRMED blocks there is no coherent trend. A one-day bounce
    # against a four-day decline is not a four-day uptrend.
    if consecutive < min_consecutive_blocks:
        return None

    trend_start_ts = float(timestamps[confirm_lo])   # duration includes confirmed data only
    duration_seconds = t_end - trend_start_ts
    if duration_seconds <= 0:
        return None
    return {
        'direction': 'up' if current_sign > 0 else 'down',
        'start_timestamp': float(trend_start_ts),
        'duration_seconds': float(duration_seconds),
        'estimated_future_hours': float(duration_seconds / 3600.0 * 0.5),
        'current_slope_h': float(cur),
    }


def analyze(client, coin: str) -> dict | None:
    """Run analysis over HL candles for the coin."""
    window_hours = required_int_env("ANALYSIS_WINDOW_H")
    step_hours = required_int_env("ANALYSIS_STEP_H")
    if window_hours <= 0 or step_hours <= 0:
        return None
    lookback = window_hours + step_hours * 14   # enough historical windows
    candles = client.candles(coin, "1h", lookback_hours=lookback)
    pts: list[tuple[float, float]] = []
    for c in candles or []:
        if not isinstance(c, dict) or c.get("t") is None or c.get("c") is None:
            continue
        if isinstance(c.get("t"), bool) or isinstance(c.get("c"), bool):
            continue
        try:
            t_s = float(c["t"]) / 1000.0
            c_v = float(c["c"])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(t_s) and t_s > 0 and math.isfinite(c_v) and c_v > 0:
            pts.append((t_s, c_v))
    if len(pts) < window_hours:
        return None
    ts = [p[0] for p in pts]
    px = [p[1] for p in pts]
    return detect_long_term_trend(ts, px, window_hours, step_hours)


def signal(client, coin: str) -> dict:
    """Return a bot-format signal: {trend, confidence, source, detail}."""
    res = analyze(client, coin)
    if not res:
        return {"trend": "neutral", "confidence": 0.0, "source": "analysis(insuf.)"}
    # Confidence combines relative hourly slope and trend duration.
    last_px = None
    try:
        raw_mid = client.mid(coin)
        if raw_mid is not None and not isinstance(raw_mid, bool):
            px_f = float(raw_mid)
            if math.isfinite(px_f) and px_f > 0:
                last_px = px_f
    except Exception:  # noqa: BLE001
        pass
    slope_pct_h = abs(res["current_slope_h"]) / last_px * 100 if last_px is not None else 0.0
    dur_h = res["duration_seconds"] / 3600.0
    conf = round(min(1.0, slope_pct_h / 0.3 * 0.6 + min(dur_h / 48.0, 1.0) * 0.4), 2)
    return {
        "trend": res["direction"], "confidence": conf, "source": "analysis",
        "estimated_future_hours": round(res["estimated_future_hours"], 1),
        "detail": f"durata {dur_h:.0f}h, est. continuare ~{res['estimated_future_hours']:.0f}h, "
                  f"panta {res['current_slope_h']:+.4f}/h ({slope_pct_h:.2f}%/h)",
    }
