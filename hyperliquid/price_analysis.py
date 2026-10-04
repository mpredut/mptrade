#!/usr/bin/env python3
"""
price_analysis.py — port of binance/priceAnalysis.py trend analysis to Hyperliquid data.

Detect trends over TIME-defined windows rather than point counts. Use linear-regression
slope in each window and walk backward while its sign persists within noise tolerance.
Return direction, duration, and estimated continuation near half the past duration.

This runs on one-hour HL candles and therefore analyzes the real perpetual price.
"""

from __future__ import annotations

import numpy as np

from common import log, float_env

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
    timestamps = np.asarray(timestamps, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if len(timestamps) < 2:
        return None

    t_end, t_first = timestamps[-1], timestamps[0]
    window_sec = window_hours * 3600.0
    step_sec = step_hours * 3600.0

    def slope_h(t_lo, t_hi):
        lo = int(np.searchsorted(timestamps, t_lo, "left"))
        hi = int(np.searchsorted(timestamps, t_hi, "left"))
        if hi - lo < min_points_per_window:
            return None, (lo, hi)
        x, y = timestamps[lo:hi], prices[lo:hi]
        s = _fast_linear_slope_1d(x, y)
        return s * 3600.0, (lo, hi)

    cur, cur_idx = slope_h(t_end - window_sec, t_end + 1.0)
    if cur is None:
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
    window_hours = int(float_env("ANALYSIS_WINDOW_H") or 24)
    step_hours = int(float_env("ANALYSIS_STEP_H") or 8)
    lookback = window_hours + step_hours * 14   # enough historical windows
    candles = client.candles(coin, "1h", lookback_hours=lookback)
    ts = [float(c["t"]) / 1000.0 for c in candles if "t" in c]
    px = [float(c["c"]) for c in candles if "c" in c]
    if len(px) < window_hours:
        return None
    return detect_long_term_trend(ts, px, window_hours, step_hours)


def signal(client, coin: str) -> dict:
    """Return a bot-format signal: {trend, confidence, source, detail}."""
    res = analyze(client, coin)
    if not res:
        return {"trend": "neutral", "confidence": 0.0, "source": "analysis(insuf.)"}
    # Confidence combines relative hourly slope and trend duration.
    last_px = None
    try:
        last_px = client.mid(coin)
    except Exception:  # noqa: BLE001
        pass
    slope_pct_h = abs(res["current_slope_h"]) / last_px * 100 if last_px else 0.0
    dur_h = res["duration_seconds"] / 3600.0
    conf = round(min(1.0, slope_pct_h / 0.3 * 0.6 + min(dur_h / 48.0, 1.0) * 0.4), 2)
    return {
        "trend": res["direction"], "confidence": conf, "source": "analysis",
        "estimated_future_hours": round(res["estimated_future_hours"], 1),
        "detail": f"durata {dur_h:.0f}h, est. continuare ~{res['estimated_future_hours']:.0f}h, "
                  f"panta {res['current_slope_h']:+.4f}/h ({slope_pct_h:.2f}%/h)",
    }
