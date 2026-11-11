#!/usr/bin/env python3
"""
signals.py -- SIGNAL layer decoupled from the strategy.

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
    log,
    normalize_epoch_sec,
    normalize_trend_direction,
    required_env,
    required_float_env,
    required_int_env,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from intelligence.internal.state.persistence import detect_long_term_trend as _root_detect_long_term_trend  # noqa: E402
from state_io import read_json_dict  # noqa: E402

_common_mod = load_local_module(_HERE, "common")
_hl_signals_env = _common_mod.load_hl_default_env


MIN_POINTS_PER_WINDOW = 3
NEUTRAL = {"trend": "neutral", "confidence": 0.0, "source": "neutral"}


def _extract_closed_candle_points(candles: object, *, now_sec: float | None = None) -> list[tuple[float, float]]:
    """Extract sorted, deduplicated (ts_sec, close) points, dropping unclosed candles when 'T' is present."""
    eff_now = time.time() if now_sec is None else now_sec
    by_ts: dict[float, float] = {}
    ordered_closes_no_ts: list[tuple[float, float]] = []
    for idx, c in enumerate(candles or []):
        if not isinstance(c, dict):
            continue
        if "T" in c and c.get("T") is not None:
            close_ts = normalize_epoch_sec(c.get("T"), default=None)
            if close_ts is not None and close_ts > eff_now:
                continue
        c_v = safe_pos_float(c.get("c"), default=None)
        if c_v is None:
            continue
        t_s = normalize_epoch_sec(c.get("t"), default=None)
        if t_s is not None:
            by_ts[t_s] = c_v
        else:
            ordered_closes_no_ts.append((idx * 1.0, c_v))
    if by_ts:
        return sorted(by_ts.items(), key=lambda item: item[0])
    return ordered_closes_no_ts


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


def _analyze_with_last_px(client, coin: str, env: dict[str, str] | None = None) -> tuple[dict | None, float | None]:
    eff_env = _hl_signals_env() if env is None else env
    window_hours = required_int_env("ANALYSIS_WINDOW_H", eff_env)
    step_hours = required_int_env("ANALYSIS_STEP_H", eff_env)
    if window_hours <= 0 or step_hours <= 0:
        return None, None
    lookback = window_hours + step_hours * 14
    candles = client.candles(coin, "1h", lookback_hours=lookback)
    pts = _extract_closed_candle_points(candles)
    if len(pts) < window_hours:
        return None, None
    res = detect_long_term_trend([p[0] for p in pts], [p[1] for p in pts], window_hours, step_hours)
    return res, pts[-1][1]


def analyze(client, coin: str) -> dict | None:
    """Run analysis over HL candles for the coin."""
    res, _ = _analyze_with_last_px(client, coin)
    return res


def signal(client, coin: str) -> dict:
    """Return a bot-format signal: {trend, confidence, source, detail}."""
    res = None
    fallback_px = None
    if analyze is _DEFAULT_ANALYZE:
        res, fallback_px = _analyze_with_last_px(client, coin)
    else:
        res = analyze(client, coin)
    if not res:
        return {"trend": "neutral", "confidence": 0.0, "source": "analysis(insuf.)"}
    last_px = None
    try:
        last_px = safe_pos_float(client.mid(coin), None)
    except Exception:  # noqa: BLE001
        pass
    if last_px is None:
        last_px = fallback_px
    slope_pct_h = abs(res["current_slope_h"]) / last_px * 100 if last_px is not None else 0.0
    dur_h = res["duration_seconds"] / 3600.0
    conf = round(min(1.0, slope_pct_h / 0.3 * 0.6 + min(dur_h / 48.0, 1.0) * 0.4), 2)
    return {
        "trend": res["direction"],
        "confidence": conf,
        "source": "analysis",
        "estimated_future_hours": round(res["estimated_future_hours"], 1),
        "detail": (
            f"duration {dur_h:.0f}h, est. continuation ~{res['estimated_future_hours']:.0f}h, "
            f"slope {res['current_slope_h']:+.4f}/h ({slope_pct_h:.2f}%/h)"
        ),
    }


def _normalize_signal_trend(raw: object) -> str:
    """Normalize a signal trend token ('up'/'bull'/'bullish'/'long' -> 'up', 'down'/'bear'/'bearish'/'short' -> 'down')."""
    return normalize_trend_direction(raw, default="neutral", allow_long_short=True) or "neutral"


def classify_fast_slow_closes(win_fast, win_slow, band: float) -> tuple[float, float, float, str]:
    mf = statistics.fmean(win_fast)
    ms = statistics.fmean(win_slow)
    diff = (mf - ms) / ms if ms else 0.0
    trend = "up" if diff > band else "down" if diff < -band else "neutral"
    return mf, ms, diff, trend


def _builtin_trend(client, coin: str) -> dict:
    eff_env = _hl_signals_env()
    fast = required_int_env("SIGNAL_FAST_H", eff_env)
    slow = required_int_env("SIGNAL_SLOW_H", eff_env)
    band = required_float_env("SIGNAL_BAND_PCT", eff_env) / 100
    band_f = safe_nonneg_float(band, default=None)
    if fast <= 0 or slow <= 0 or band_f is None:
        return {**NEUTRAL, "source": "builtin(insuf. data)"}
    candles = client.candles(coin, "1h", lookback_hours=slow + 6)
    pts = _extract_closed_candle_points(candles)
    closes = [p[1] for p in pts]
    if len(closes) < max(fast, slow):
        return {**NEUTRAL, "source": "builtin(insuf. data)"}
    mf, ms, diff, trend = classify_fast_slow_closes(closes[-fast:], closes[-slow:], band_f)
    if band_f > 0:
        conf = round(min(1.0, abs(diff) / (band_f * 3)), 2)
    else:
        conf = 1.0 if trend != "neutral" else 0.0
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
    has_explicit_ts = "ts" in d or "timestamp" in d
    ts = d.get("ts") if d.get("ts") is not None else d.get("timestamp")
    eff_env = _hl_signals_env()
    max_age = required_float_env("SIGNAL_MAX_AGE_MIN", eff_env) * 60
    if ts is None:
        if has_explicit_ts:
            return {**NEUTRAL, "source": "file(expired)"}
        try:
            ts_f = normalize_epoch_sec(os.path.getmtime(str(path)), default=0.0) or 0.0
        except OSError:
            return {**NEUTRAL, "source": "file(expired)"}
    else:
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


_DEFAULT_ANALYZE = analyze
_DEFAULT_ANALYSIS_SIGNAL = signal


def get_signal(client, coin: str) -> dict:
    eff_env = _hl_signals_env()
    src = required_env("SIGNAL_SOURCE", eff_env).strip().lower()
    if src == "builtin":
        return _builtin_trend(client, coin)
    if src == "analysis":               # ported WMA/time-window slope analysis
        analysis_signal = resolve_patched_attr(
            "signal",
            signal,
            local_val=signal,
            defaults=(_DEFAULT_ANALYSIS_SIGNAL,),
            modules=("signals", "hyperliquid.signals", __name__),
            require_callable=True,
        )
        return analysis_signal(client, coin)
    if src == "file":
        return _file_signal(required_env("SIGNAL_FILE", eff_env))
    if src == "off":
        return {**NEUTRAL, "source": "off"}
    raise ValueError(f"Invalid SIGNAL_SOURCE: {src!r}")


_self_mod = sys.modules.get(__name__)
if _self_mod is not None and __name__ in ("signals", "hyperliquid.signals"):
    if __name__ == "hyperliquid.signals":
        sys.modules["hyperliquid.signals"] = _self_mod
    _hl_pkg = sys.modules.get("hyperliquid")
    if _hl_pkg is not None and (__name__ == "hyperliquid.signals" or not hasattr(_hl_pkg, "signals")):
        setattr(_hl_pkg, "signals", _self_mod)

