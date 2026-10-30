#!/usr/bin/env python3
"""
price_analysis.py — port of binance/priceAnalysis.py trend analysis to Hyperliquid data.

Detect trends over TIME-defined windows rather than point counts. Use linear-regression
slope in each window and walk backward while its sign persists within noise tolerance.
Return direction, duration, and estimated continuation near half the past duration.

This runs on one-hour HL candles and therefore analyzes the real perpetual price.
"""

from __future__ import annotations

import os
import sys

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
safe_pos_float = _common_mod.safe_pos_float
from priceAnalysis import detect_long_term_trend as _root_detect_long_term_trend

load_env_stack(os.path.join(_HERE, ".env"))

MIN_POINTS_PER_WINDOW = 3


def detect_long_term_trend(timestamps, prices, window_hours=24, step_hours=8,
                           min_consecutive_blocks=3, noise_tolerance=2,
                           min_points_per_window=MIN_POINTS_PER_WINDOW):
    """Port priceAnalysis.detect_long_term_trend with gap/density robustness."""
    res = _root_detect_long_term_trend(
        timestamps,
        prices,
        window_hours=window_hours,
        step_hours=step_hours,
        min_consecutive_blocks=min_consecutive_blocks,
        noise_tolerance=noise_tolerance,
        min_points_per_window=min_points_per_window,
        detection_lag_hours=0.0,
        mk_alpha=None,
    )
    if res is None:
        return None
    res.pop("blocks", None)
    return res


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
