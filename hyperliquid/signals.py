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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from common import log, required_env, required_float_env, required_int_env

NEUTRAL = {"trend": "neutral", "confidence": 0.0, "source": "neutral"}


def _builtin_trend(client, coin: str) -> dict:
    fast = required_int_env("SIGNAL_FAST_H")
    slow = required_int_env("SIGNAL_SLOW_H")
    band = required_float_env("SIGNAL_BAND_PCT") / 100
    if fast <= 0 or slow <= 0 or not math.isfinite(band) or band < 0:
        return {**NEUTRAL, "source": "builtin(insuf. date)"}
    candles = client.candles(coin, "1h", lookback_hours=slow + 6)
    closes: list[float] = []
    for c in candles or []:
        if not isinstance(c, dict) or "c" not in c or isinstance(c.get("c"), bool):
            continue
        try:
            v = float(c["c"])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(v) and v > 0:
            closes.append(v)
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
    if not os.path.exists(path):
        return {**NEUTRAL, "source": "file(missing)"}
    try:
        with open(path, "r", encoding="utf-8") as f:
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
        if isinstance(ts, bool):
            return {**NEUTRAL, "source": "file(expirat)"}
        try:
            ts_f = float(ts)
        except (TypeError, ValueError, OverflowError):
            return {**NEUTRAL, "source": "file(expirat)"}
        if math.isfinite(ts_f) and ts_f >= 100_000_000_000.0:
            ts_f /= 1000.0
        age = time.time() - ts_f
        if not math.isfinite(ts_f) or ts_f <= 0 or age > max_age or age < -60.0:
            return {**NEUTRAL, "source": "file(expirat)"}
    raw_tr = d.get("trend", "neutral")
    tr = raw_tr.strip().lower() if isinstance(raw_tr, str) else "neutral"
    if tr not in ("up", "down", "neutral"):
        tr = "neutral"
    raw_conf = d.get("confidence", 0)
    if isinstance(raw_conf, bool):
        conf = 0.0
    else:
        try:
            conf = float(raw_conf or 0)
        except (TypeError, ValueError, OverflowError):
            conf = 0.0
    if not math.isfinite(conf):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {"trend": tr, "confidence": conf, "source": "file"}


def get_signal(client, coin: str) -> dict:
    src = required_env("SIGNAL_SOURCE").strip().lower()
    if src == "builtin":
        return _builtin_trend(client, coin)
    if src == "analysis":               # ported WMA/time-window slope analysis
        try:
            from price_analysis import signal as analysis_signal
        except ImportError:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "hyperliquid_price_analysis", os.path.join(_HERE, "price_analysis.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            analysis_signal = mod.signal
        return analysis_signal(client, coin)
    if src == "file":
        return _file_signal(required_env("SIGNAL_FILE"))
    if src == "off":
        return {**NEUTRAL, "source": "off"}
    raise ValueError(f"Invalid SIGNAL_SOURCE: {src!r}")
