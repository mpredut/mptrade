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

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from botcore import (  # noqa: E402
    load_local_module,
    normalize_epoch_sec,
    normalize_trend_direction,
    resolve_patched_attr,
)
from priceAnalysis import detect_long_term_trend as _root_detect_long_term_trend  # noqa: E402
from state_io import read_json_dict  # noqa: E402

_common_mod = load_local_module(_HERE, "common")
log = _common_mod.log
load_env_stack = _common_mod.load_env_stack
required_env = _common_mod.required_env
required_float_env = _common_mod.required_float_env
required_int_env = _common_mod.required_int_env
safe_float = _common_mod.safe_float
safe_nonneg_float = _common_mod.safe_nonneg_float
safe_pos_float = _common_mod.safe_pos_float

load_env_stack(os.path.join(_HERE, ".env"))

MIN_POINTS_PER_WINDOW = 3
NEUTRAL = {"trend": "neutral", "confidence": 0.0, "source": "neutral"}


def detect_long_term_trend(
    timestamps,
    prices,
    window_hours=24,
    step_hours=8,
    min_consecutive_blocks=3,
    noise_tolerance=2,
    min_points_per_window=MIN_POINTS_PER_WINDOW,
):
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
    lookback = window_hours + step_hours * 14
    candles = client.candles(coin, "1h", lookback_hours=lookback)
    pts: list[tuple[float, float]] = []
    for c in candles or []:
        if not isinstance(c, dict):
            continue
        t_ms = safe_pos_float(c.get("t"), default=None)
        c_v = safe_pos_float(c.get("c"), default=None)
        if t_ms is not None and c_v is not None:
            pts.append((t_ms / 1000.0 if t_ms >= 1e11 else t_ms, c_v))
    if len(pts) < window_hours:
        return None
    return detect_long_term_trend([p[0] for p in pts], [p[1] for p in pts], window_hours, step_hours)


def signal(client, coin: str) -> dict:
    """Return a bot-format signal: {trend, confidence, source, detail}."""
    res = analyze(client, coin)
    if not res:
        return {"trend": "neutral", "confidence": 0.0, "source": "analysis(insuf.)"}
    last_px = None
    try:
        last_px = safe_pos_float(client.mid(coin), None)
    except Exception:  # noqa: BLE001
        pass
    slope_pct_h = abs(res["current_slope_h"]) / last_px * 100 if last_px is not None else 0.0
    dur_h = res["duration_seconds"] / 3600.0
    conf = round(min(1.0, slope_pct_h / 0.3 * 0.6 + min(dur_h / 48.0, 1.0) * 0.4), 2)
    return {
        "trend": res["direction"],
        "confidence": conf,
        "source": "analysis",
        "estimated_future_hours": round(res["estimated_future_hours"], 1),
        "detail": (
            f"durata {dur_h:.0f}h, est. continuare ~{res['estimated_future_hours']:.0f}h, "
            f"panta {res['current_slope_h']:+.4f}/h ({slope_pct_h:.2f}%/h)"
        ),
    }


def _normalize_signal_trend(raw: object) -> str:
    """Normalize a signal trend token ('up'/'bull'/'bullish'/'long' -> 'up', 'down'/'bear'/'bearish'/'short' -> 'down')."""
    return normalize_trend_direction(raw, default="neutral", allow_long_short=True) or "neutral"


def _builtin_trend(client, coin: str) -> dict:
    fast = required_int_env("SIGNAL_FAST_H")
    slow = required_int_env("SIGNAL_SLOW_H")
    band = required_float_env("SIGNAL_BAND_PCT") / 100
    band_f = safe_nonneg_float(band, default=None)
    if fast <= 0 or slow <= 0 or band_f is None:
        return {**NEUTRAL, "source": "builtin(insuf. date)"}
    candles = client.candles(coin, "1h", lookback_hours=slow + 6)
    closes = [
        v for c in (candles or [])
        if isinstance(c, dict) and (v := safe_pos_float(c.get("c"), 0.0)) > 0
    ]
    if len(closes) < max(fast, slow):
        return {**NEUTRAL, "source": "builtin(insuf. date)"}
    mf = statistics.fmean(closes[-fast:])
    ms = statistics.fmean(closes[-slow:])
    diff = (mf - ms) / ms if ms else 0.0
    trend = "up" if diff > band else "down" if diff < -band else "neutral"
    conf = round(min(1.0, abs(diff) / (band * 3)), 2) if band > 0 else 0.0
    return {"trend": trend, "confidence": conf, "source": "builtin",
            "detail": f"avg{fast}h={mf:.3f} avg{slow}h={ms:.3f} diff={diff*100:+.2f}%"}


def _file_signal(path: str) -> dict:
    if not path or isinstance(path, bool) or not str(path).strip() or not os.path.isfile(str(path)):
        return {**NEUTRAL, "source": "file(missing)"}
    try:
        d = read_json_dict(str(path), recover_previous=True)
    except (OSError, ValueError) as e:
        log(f"  ! an invalid signal file: {e}")
        return {**NEUTRAL, "source": "file(corrupt)"}
    ts = d.get("ts") if d.get("ts") is not None else d.get("timestamp")
    max_age = required_float_env("SIGNAL_MAX_AGE_MIN") * 60
    if ts is not None:
        ts_f = normalize_epoch_sec(ts, default=0.0) or 0.0
        if ts_f <= 0:
            return {**NEUTRAL, "source": "file(expired)"}
        age = time.time() - ts_f
        max_age_f = safe_pos_float(max_age, default=None)
        if max_age_f is None or age > max_age_f or age < -60.0:
            return {**NEUTRAL, "source": "file(expired)"}
    tr = _normalize_signal_trend(d.get("trend", "neutral"))
    conf = max(0.0, min(1.0, safe_float(d.get("confidence", 0), 0.0)))
    return {"trend": tr, "confidence": conf, "source": "file"}


_DEFAULT_ANALYSIS_SIGNAL = signal


def get_signal(client, coin: str) -> dict:
    src = required_env("SIGNAL_SOURCE").strip().lower()
    if src == "builtin":
        return _builtin_trend(client, coin)
    if src == "analysis":               # ported WMA/time-window slope analysis
        analysis_signal = resolve_patched_attr(
            "signal",
            signal,
            local_val=signal,
            defaults=(_DEFAULT_ANALYSIS_SIGNAL,),
            modules=("price_analysis", "hyperliquid.price_analysis", __name__),
            require_callable=True,
        )
        return analysis_signal(client, coin)
    if src == "file":
        return _file_signal(required_env("SIGNAL_FILE"))
    if src == "off":
        return {**NEUTRAL, "source": "off"}
    raise ValueError(f"Invalid SIGNAL_SOURCE: {src!r}")


_self_mod = sys.modules.get(__name__)
if _self_mod is not None and __name__ in (
    "signals",
    "hyperliquid.signals",
    "price_analysis",
    "hyperliquid.price_analysis",
):
    for _alias_k in ("price_analysis", "hyperliquid.price_analysis"):
        _prev_mod = sys.modules.get(_alias_k)
        if _prev_mod is None or getattr(_prev_mod, "__file__", None) == __file__:
            sys.modules[_alias_k] = _self_mod
    if __name__ == "hyperliquid.signals":
        sys.modules["hyperliquid.signals"] = _self_mod
    _hl_pkg = sys.modules.get("hyperliquid")
    if _hl_pkg is not None:
        setattr(_hl_pkg, "price_analysis", _self_mod)
        if __name__ == "hyperliquid.signals" or not hasattr(_hl_pkg, "signals"):
            setattr(_hl_pkg, "signals", _self_mod)

