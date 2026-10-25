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
    candles = client.candles(coin, "1h", lookback_hours=slow + 6)
    closes = [float(c["c"]) for c in candles if "c" in c]
    if len(closes) < slow:
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
    ts = d.get("ts") or d.get("timestamp")
    max_age = required_float_env("SIGNAL_MAX_AGE_MIN") * 60
    if ts:
        try:
            if time.time() - float(ts) > max_age:
                return {**NEUTRAL, "source": "file(expirat)"}
        except (TypeError, ValueError):
            pass
    tr = str(d.get("trend", "neutral")).lower()
    if tr not in ("up", "down", "neutral"):
        tr = "neutral"
    try:
        conf = float(d.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    return {"trend": tr, "confidence": conf, "source": "file"}


def get_signal(client, coin: str) -> dict:
    src = required_env("SIGNAL_SOURCE").lower()
    if src == "builtin":
        return _builtin_trend(client, coin)
    if src == "analysis":               # ported WMA/time-window slope analysis
        from price_analysis import signal as analysis_signal
        return analysis_signal(client, coin)
    if src == "file":
        return _file_signal(required_env("SIGNAL_FILE"))
    if src == "off":
        return {**NEUTRAL, "source": "off"}
    raise ValueError(f"Invalid SIGNAL_SOURCE: {src!r}")
