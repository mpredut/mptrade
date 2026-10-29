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
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)
_common_mod = sys.modules.get("common")
_common_file = getattr(_common_mod, "__file__", None) if _common_mod is not None else None
if _common_mod is None or (isinstance(_common_file, str) and os.path.dirname(os.path.abspath(_common_file)) != _HERE):
    import importlib.util as _ilu
    _c_spec = _ilu.spec_from_file_location("common", os.path.join(_HERE, "common.py"))
    _common_mod = _ilu.module_from_spec(_c_spec)
    sys.modules["common"] = _common_mod
    _c_spec.loader.exec_module(_common_mod)
load_env_stack = _common_mod.load_env_stack
required_int_env = _common_mod.required_int_env
safe_float = _common_mod.safe_float
safe_pos_float = _common_mod.safe_pos_float

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


def _exact_min_int(val: object, min_val: int) -> int | None:
    f = safe_float(val, default=None)
    if f is None or not f.is_integer():
        return None
    i = int(f)
    return i if i >= min_val else None


def detect_long_term_trend(timestamps, prices, window_hours=24, step_hours=8,
                           min_consecutive_blocks=3, noise_tolerance=2,
                           min_points_per_window=MIN_POINTS_PER_WINDOW):
    """Port priceAnalysis.detect_long_term_trend with gap/density robustness."""
    if (
        timestamps is None
        or prices is None
        or isinstance(timestamps, (bool, str, bytes))
        or isinstance(prices, (bool, str, bytes))
    ):
        return None
    wh_f = safe_pos_float(window_hours, default=None)
    sh_f = safe_pos_float(step_hours, default=None)
    mcb_i = _exact_min_int(min_consecutive_blocks, 1)
    nt_i = _exact_min_int(noise_tolerance, 0)
    mpw_i = _exact_min_int(min_points_per_window, 2)
    if wh_f is None or sh_f is None or mcb_i is None or nt_i is None or mpw_i is None:
        return None
    try:
        raw_ts = list(timestamps)
        raw_px = list(prices)
    except TypeError:
        return None
    if len(raw_ts) != len(raw_px) or len(raw_ts) < 2:
        return None
    pairs: list[tuple[float, float]] = []
    for t_item, p_item in zip(raw_ts, raw_px):
        tf = safe_pos_float(t_item, default=None)
        pf = safe_pos_float(p_item, default=None)
        if tf is not None and pf is not None:
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
    if cur is None or not math.isfinite(cur) or cur == 0.0:
        return None
    current_sign = float(np.sign(cur))

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
        if not isinstance(c, dict):
            continue
        t_ms = safe_pos_float(c.get("t"), default=None)
        c_v = safe_pos_float(c.get("c"), default=None)
        if t_ms is not None and c_v is not None:
            pts.append((t_ms / 1000.0, c_v))
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
        last_px = safe_pos_float(client.mid(coin), None)
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
