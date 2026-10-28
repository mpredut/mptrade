#!/usr/bin/env python3
"""
signals.py — SIGNAL layer decoupled from the strategy.

The strategy reads {"trend": "up"|"down"|"neutral", "confidence": 0..1}.
SIGNAL_SOURCE selects:
  * off: always neutral, so the strategy ignores trend
  * builtin: simple HL candle trend from fast versus slow averages
  * file: JSON written by an external LSTM/price-analysis model

External file format (SIGNAL_FILE, default signal.json):
  {"trend": "down", "confidence": 0.72, "ts": 1781000000}
  - 'ts' is a Unix timestamp in seconds. Signals older than SIGNAL_MAX_AGE_MIN expire
    to neutral, preventing trades based on stale predictions.

An LSTM only needs to write signal.json after each prediction for the bot to consume.
RULE: backtest every signal before assigning real money.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

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
log = _common_mod.log
load_hl_local = _common_mod.load_hl_local
required_env = _common_mod.required_env
required_float_env = _common_mod.required_float_env
required_int_env = _common_mod.required_int_env
safe_float = _common_mod.safe_float
safe_pos_float = _common_mod.safe_pos_float

NEUTRAL = {"trend": "neutral", "confidence": 0.0, "source": "neutral"}


def _normalize_signal_trend(raw: object) -> str:
    """Normalize a signal trend token ('up'/'bull'/'bullish'/'long' -> 'up', 'down'/'bear'/'bearish'/'short' -> 'down')."""
    if not isinstance(raw, str):
        return "neutral"
    t = raw.strip().lower()
    if t in ("up", "bull", "bullish", "long"):
        return "up"
    if t in ("down", "bear", "bearish", "short"):
        return "down"
    return "neutral"


def _builtin_trend(client, coin: str) -> dict:
    fast = required_int_env("SIGNAL_FAST_H")
    slow = required_int_env("SIGNAL_SLOW_H")
    band = required_float_env("SIGNAL_BAND_PCT") / 100
    if fast <= 0 or slow <= 0 or not math.isfinite(band) or band < 0:
        return {**NEUTRAL, "source": "builtin(insuf. date)"}
    candles = client.candles(coin, "1h", lookback_hours=slow + 6)
    closes = [
        v for c in (candles or [])
        if isinstance(c, dict) and (v := safe_pos_float(c.get("c"), 0.0)) > 0
    ]
    if len(closes) < max(fast, slow):
        return {**NEUTRAL, "source": "builtin(insuf. date)"}
    mf = statistics.mean(closes[-fast:])
    ms = statistics.mean(closes[-slow:])
    diff = (mf - ms) / ms if ms else 0.0
    trend = "up" if diff > band else "down" if diff < -band else "neutral"
    conf = round(min(1.0, abs(diff) / (band * 3)), 2) if band > 0 else 0.0
    return {"trend": trend, "confidence": conf, "source": "builtin",
            "detail": f"medie{fast}h={mf:.3f} medie{slow}h={ms:.3f} diff={diff*100:+.2f}%"}


def _file_signal(path: str) -> dict:
    if not path or isinstance(path, bool) or not str(path).strip() or not os.path.isfile(str(path)):
        return {**NEUTRAL, "source": "file(missing)"}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        log(f"  ! an invalid signal file: {e}")
        return {**NEUTRAL, "source": "file(corupt)"}
    if not isinstance(d, dict):
        log("  ! an invalid signal file: top-level JSON must be an object")
        return {**NEUTRAL, "source": "file(corupt)"}
    ts = d.get("ts") if d.get("ts") is not None else d.get("timestamp")
    max_age = required_float_env("SIGNAL_MAX_AGE_MIN") * 60
    if ts is not None:
        ts_f = safe_pos_float(ts, 0.0)
        if ts_f <= 0:
            return {**NEUTRAL, "source": "file(expirat)"}
        if ts_f >= 100_000_000_000.0:
            ts_f /= 1000.0
        age = time.time() - ts_f
        if not math.isfinite(max_age) or max_age <= 0 or age > max_age or age < -60.0:
            return {**NEUTRAL, "source": "file(expirat)"}
    tr = _normalize_signal_trend(d.get("trend", "neutral"))
    conf = max(0.0, min(1.0, safe_float(d.get("confidence", 0), 0.0)))
    return {"trend": tr, "confidence": conf, "source": "file"}


def get_signal(client, coin: str) -> dict:
    src = required_env("SIGNAL_SOURCE").strip().lower()
    if src == "builtin":
        return _builtin_trend(client, coin)
    if src == "analysis":               # ported WMA/time-window slope analysis
        try:
            from price_analysis import signal as analysis_signal
        except ImportError:
            analysis_signal = load_hl_local("price_analysis").signal
        return analysis_signal(client, coin)
    if src == "file":
        return _file_signal(required_env("SIGNAL_FILE"))
    if src == "off":
        return {**NEUTRAL, "source": "off"}
    raise ValueError(f"Invalid SIGNAL_SOURCE: {src!r}")
