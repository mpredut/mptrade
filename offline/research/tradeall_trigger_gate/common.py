#!/usr/bin/env python3
"""Shared instrumentation, logic factories, and isolated runner for tradeall trigger-gate experiments."""
from __future__ import annotations

from collections import deque
import json
import math
import os
import shutil
import sys
import time
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.backtests import tradeall as tb  # noqa: E402
import tradeall as ta  # noqa: E402

ORIG_TREND_STATE = ta.TrendState
ORIG_LOGIC = ta.logic
ORIG_LOGIC_SMALL = getattr(ta, "logic_small", None)
ORIG_WINDOW_ANALYZER = ta.WindowAnalyzer
ORIG_BIG_THRESHOLD = ta.PRICE_CHANGE_THRESHOLD_BIG_EUR
ORIG_TREND_TO_BE_OLD = ta.TREND_TO_BE_OLD_SECONDS

DEFAULT_SAFEBACK_SECONDS = 14 * 24 * 3600 + 60


def default_up_cond(gradient: float, slope: float) -> bool:
    return gradient > 0 and slope < 0


def default_down_cond(gradient: float, slope: float) -> bool:
    return gradient < 0 and slope > 0


def _safe_pos_f(val, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if (math.isfinite(f) and f > 0) else default
    except (TypeError, ValueError, OverflowError):
        return default


def make_instrumented_trend_state(
    stats: dict,
    *,
    confirm_threshold_override: Optional[int] = None,
    expiration_trend_time_override: Optional[float] = None,
    retry_interval_sec: float = 1800.0,
):
    """Create a fresh TrendState subclass rooted at ORIG_TREND_STATE (never chained across runs)."""
    stats = stats if isinstance(stats, dict) else {}
    retry_sec = _safe_pos_f(retry_interval_sec, 1800.0)
    exp_override = (
        _safe_pos_f(expiration_trend_time_override, 0.0)
        if expiration_trend_time_override is not None
        else None
    )
    if exp_override == 0.0:
        exp_override = None
    conf_override: Optional[int] = None
    if confirm_threshold_override is not None and not isinstance(confirm_threshold_override, bool):
        try:
            cf = float(confirm_threshold_override)
            if math.isfinite(cf) and cf >= 0:
                conf_override = int(cf)
        except (TypeError, ValueError, OverflowError):
            conf_override = None

    class InstrumentedTrendState(ORIG_TREND_STATE):
        def __init__(self, *a, **kw):
            if exp_override is not None:
                if len(a) >= 2:
                    a = (a[0], exp_override, *a[2:])
                else:
                    kw["expiration_trend_time"] = exp_override
            super().__init__(*a, **kw)
            self._fired_up = False
            self._fired_down = False
            self._last_attempt_up_ts = None
            self._last_attempt_down_ts = None
            self._confirmed_up = False
            self._confirmed_down = False

        def start_trend(self, new_state):
            stats["starts"] = stats.get("starts", 0) + 1
            self._fired_up = False
            self._fired_down = False
            self._confirmed_up = False
            self._confirmed_down = False
            return super().start_trend(new_state)

        def confirm_trend(self):
            r = super().confirm_trend()
            stats["confirms"] = stats.get("confirms", 0) + 1
            if "max_confirm_count" in stats:
                stats["max_confirm_count"] = max(stats["max_confirm_count"], self.confirm_count)
            return r

        def check_trend_expiration(self):
            was_expired = self.expired
            r = super().check_trend_expiration()
            if r and not was_expired:
                stats["expires"] = stats.get("expires", 0) + 1
            return r

        def is_trend_consistent_validated(self):
            if conf_override is None:
                r = super().is_trend_consistent_validated()
            elif not self.is_trend_a_minim_validated():
                r = False
            else:
                r = (
                    self.confirm_count > conf_override
                    and self.is_trend_uniform_confirmed()
                )
            if r and "reached_24" in stats:
                stats["reached_24"] += 1
            return r

        def is_started_trend_older_than(self, old_trend_time):
            r = super().is_started_trend_older_than(old_trend_time)
            if r and "reached_old_1_9h" in stats:
                stats["reached_old_1_9h"] += 1
            return r

        def mark_fired(self, direction: str):
            if direction == "UP" and not self._fired_up:
                self._fired_up = True
                if "fired_instances_up" in stats:
                    stats["fired_instances_up"] += 1
            elif direction == "DOWN" and not self._fired_down:
                self._fired_down = True
                if "fired_instances_down" in stats:
                    stats["fired_instances_down"] += 1

        def already_fired(self, direction: str) -> bool:
            return self._fired_up if direction == "UP" else self._fired_down

        def already_confirmed(self, direction: str) -> bool:
            return self._confirmed_up if direction == "UP" else self._confirmed_down

        def mark_confirmed(self, direction: str):
            super().mark_confirmed(direction)
            if direction == "UP":
                self._confirmed_up = True
            else:
                self._confirmed_down = True

        def can_retry(self, direction: str, ts: float) -> bool:
            last = self._last_attempt_up_ts if direction == "UP" else self._last_attempt_down_ts
            if last is None:
                return True
            if isinstance(ts, bool) or ts is None:
                return False
            try:
                ts_f = float(ts)
            except (TypeError, ValueError, OverflowError):
                return False
            return math.isfinite(ts_f) and (ts_f - last) >= retry_sec

        def mark_attempt(self, direction: str, ts: float):
            if isinstance(ts, bool) or ts is None:
                return
            try:
                ts_f = float(ts)
            except (TypeError, ValueError, OverflowError):
                return
            if not math.isfinite(ts_f):
                return
            if direction == "UP":
                self._last_attempt_up_ts = ts_f
            else:
                self._last_attempt_down_ts = ts_f

    return InstrumentedTrendState


def make_tradeall_logic(
    label: str,
    *,
    start_up_cond: Callable[[float, float], bool] = default_up_cond,
    start_down_cond: Callable[[float, float], bool] = default_down_cond,
    cooldown_mode: Optional[str] = None,
    stats: Optional[dict] = None,
):
    """Build a parameterized tradeall.logic replacement supporting optional cooldown modes.

    cooldown_mode:
      - None: raw firing on every matching tick (baseline tradeall behavior)
      - 'fire_once': at most one fire attempt per trend instance and direction
      - 'confirmed_retry': fire until confirmed once per trend instance, with retry cooldown on blocked attempts
    """

    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price, price_window=None, **kw):
        proposed_price = current_price

        def fire(direction: str, action: str, reason: str):
            if cooldown_mode == "fire_once":
                if trend_state.already_fired(direction):
                    return
                if enable:
                    ta._fire_order(
                        symbol, action, proposed_price, f"{label}_{reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )
                trend_state.mark_fired(direction)
            elif cooldown_mode == "confirmed_retry":
                if trend_state.already_confirmed(direction):
                    return
                ts = trend_state._now()
                dir_key = direction.lower()
                if not trend_state.can_retry(direction, ts):
                    if stats is not None:
                        stats[f"fire_skipped_cooldown_{dir_key}"] = stats.get(f"fire_skipped_cooldown_{dir_key}", 0) + 1
                    return
                trend_state.mark_attempt(direction, ts)
                if enable:
                    result = ta._fire_order(
                        symbol, action, proposed_price, f"{label}_{reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )
                    if result is not None:
                        trend_state.mark_confirmed(direction)
                        if stats is not None:
                            stats[f"fire_confirmed_{dir_key}"] = stats.get(f"fire_confirmed_{dir_key}", 0) + 1
                    else:
                        if stats is not None:
                            stats[f"fire_blocked_{dir_key}"] = stats.get(f"fire_blocked_{dir_key}", 0) + 1
            else:
                if enable:
                    ta._fire_order(
                        symbol, action, proposed_price, f"{label}_{reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )

        if start_up_cond(gradient, slope):
            if trend_state.is_trend_up():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    fire("UP", "BUY", "trend_confirmed_up")
            else:
                trend_state.start_trend("UP")

        if start_down_cond(gradient, slope):
            if trend_state.is_trend_down():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    fire("DOWN", "SELL", "trend_confirmed_down")
            else:
                trend_state.start_trend("DOWN")

        if slope <= 0 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire("UP", "BUY", "consistent_or_old_up")
        if slope >= 0 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire("DOWN", "SELL", "consistent_or_old_down")

        if slope <= -5.1 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    and trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire("UP", "BUY", "slope<=-5.1_and_old_down")
        if slope >= 5.1 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    and trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire("DOWN", "SELL", "slope>=5.1_and_old_up")

    return logic_variant


class LongTrendTracker:
    """Multi-hour linear-regression regime tracker evaluated at fixed intervals."""

    def __init__(self, stats: dict, window_hours: float, reeval_sec: float, min_points: int = 30):
        wh_f = _safe_pos_f(window_hours, 1.0)
        re_f = _safe_pos_f(reeval_sec, 60.0)
        mp_f = _safe_pos_f(min_points, 30.0)
        self.window_sec = max(60.0, wh_f * 3600.0)
        self.reeval_sec = max(1.0, re_f)
        self.min_points = max(2, int(mp_f)) if mp_f >= 2 else 30
        self.buf = deque()
        self.last_eval_ts: Optional[float] = None
        self.sign = 0
        self.fired_up = False
        self.fired_down = False
        self.last_attempt_up_ts: Optional[float] = None
        self.last_attempt_down_ts: Optional[float] = None
        self.stats = stats if isinstance(stats, dict) else {}

    def update(self, ts: float, price: float) -> int:
        if isinstance(ts, bool) or isinstance(price, bool) or ts is None or price is None:
            return self.sign
        try:
            ts = float(ts)
            price = float(price)
        except (TypeError, ValueError, OverflowError):
            return self.sign
        if not (math.isfinite(ts) and math.isfinite(price) and price > 0):
            return self.sign
        self.buf.append((ts, price))
        cutoff = ts - self.window_sec
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()
        if self.last_eval_ts is None or (ts - self.last_eval_ts) >= self.reeval_sec:
            self.last_eval_ts = ts
            if len(self.buf) >= self.min_points:
                prices = [p for _, p in self.buf]
                analyzer = ta.PriceTrendAnalyzer(prices)
                _, slope, _ = analyzer.linear_regression_trend()
                new_sign = 0
                if slope is not None and not isinstance(slope, bool) and math.isfinite(float(slope)):
                    new_sign = 1 if slope > 0 else (-1 if slope < 0 else 0)
                if new_sign != self.sign:
                    self.sign = new_sign
                    self.fired_up = False
                    self.fired_down = False
                    self.stats["sign_changes"] = self.stats.get("sign_changes", 0) + 1
        return self.sign


def make_tracker_logic(
    tag: str,
    tracker_getter: Callable[[str], object],
    *,
    up_reason: str = "long_trend_up",
    down_reason: str = "long_trend_down",
    retry_interval_sec: float = 1800.0,
):
    """Build a tradeall.logic replacement driven by a stateful regime tracker (LongTrendTracker / MeanReversionTracker)."""
    retry_sec = _safe_pos_f(retry_interval_sec, 1800.0)

    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price, price_window=None, **kw):
        tr = tracker_getter(symbol)
        ts_raw = trend_state._now()
        try:
            ts = 0.0 if (ts_raw is None or isinstance(ts_raw, bool)) else float(ts_raw)
            if not math.isfinite(ts):
                ts = 0.0
        except (TypeError, ValueError, OverflowError):
            ts = 0.0
        sign = tr.update(ts, current_price)
        stats = tr.stats if isinstance(getattr(tr, "stats", None), dict) else {}

        if sign > 0 and not tr.fired_up:
            can_retry = (
                tr.last_attempt_up_ts is None
                or (ts - tr.last_attempt_up_ts) >= retry_sec
            )
            if not can_retry:
                if "fire_skipped_cooldown_up" in stats:
                    stats["fire_skipped_cooldown_up"] += 1
            else:
                tr.last_attempt_up_ts = ts
                stats["fire_attempts_up"] = stats.get("fire_attempts_up", 0) + 1
                if enable:
                    result = ta._fire_order(
                        symbol, "BUY", current_price, f"{tag}_{up_reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )
                    if result is not None:
                        tr.fired_up, tr.fired_down = True, False
                        stats["fire_confirmed_up"] = stats.get("fire_confirmed_up", 0) + 1
                    else:
                        stats["fire_blocked_up"] = stats.get("fire_blocked_up", 0) + 1

        if sign < 0 and not tr.fired_down:
            can_retry = (
                tr.last_attempt_down_ts is None
                or (ts - tr.last_attempt_down_ts) >= retry_sec
            )
            if not can_retry:
                if "fire_skipped_cooldown_down" in stats:
                    stats["fire_skipped_cooldown_down"] += 1
            else:
                tr.last_attempt_down_ts = ts
                stats["fire_attempts_down"] = stats.get("fire_attempts_down", 0) + 1
                if enable:
                    result = ta._fire_order(
                        symbol, "SELL", current_price, f"{tag}_{down_reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )
                    if result is not None:
                        tr.fired_down, tr.fired_up = True, False
                        stats["fire_confirmed_down"] = stats.get("fire_confirmed_down", 0) + 1
                    else:
                        stats["fire_blocked_down"] = stats.get("fire_blocked_down", 0) + 1

    return logic_variant


def execute_tradeall_variant(
    run_id: str,
    symbol: str,
    start_ts: Optional[float],
    end_ts: Optional[float],
    *,
    source: str = "cache24",
    logic_fn=None,
    trend_state_cls=None,
    window_analyzer_cls=None,
    big_threshold: Optional[float] = None,
    trend_to_be_old_override: Optional[float] = None,
    disable_small: bool = False,
) -> tuple[dict, float]:
    """Run tb.run_backtest with scoped monkeypatches on tradeall and guaranteed restoration."""
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    try:
        if trend_state_cls is not None:
            ta.TrendState = trend_state_cls
            tb.ta.TrendState = trend_state_cls
        if logic_fn is not None:
            ta.logic = logic_fn
            tb.ta.logic = logic_fn
        if disable_small:
            noop = lambda *a, **kw: None  # noqa: E731
            ta.logic_small = noop
            tb.ta.logic_small = noop
        if window_analyzer_cls is not None:
            ta.WindowAnalyzer = window_analyzer_cls
            tb.ta.WindowAnalyzer = window_analyzer_cls
        if big_threshold is not None:
            ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = big_threshold
            tb.ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = big_threshold
        if trend_to_be_old_override is not None:
            ta.TREND_TO_BE_OLD_SECONDS = trend_to_be_old_override
            tb.ta.TREND_TO_BE_OLD_SECONDS = trend_to_be_old_override

        kwargs = {"quiet": True, "kalman_primary": False}
        if source == "cache24":
            kwargs["cache24_file"] = os.path.join(ROOT, "cachedb", f"cache_365_price_{symbol}.jsonl")
        tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, source, **kwargs)
    finally:
        ta.TrendState = ORIG_TREND_STATE
        tb.ta.TrendState = ORIG_TREND_STATE
        ta.logic = ORIG_LOGIC
        tb.ta.logic = ORIG_LOGIC
        if ORIG_LOGIC_SMALL is not None:
            ta.logic_small = ORIG_LOGIC_SMALL
            tb.ta.logic_small = ORIG_LOGIC_SMALL
        elif disable_small:
            if hasattr(ta, "logic_small"):
                delattr(ta, "logic_small")
            if hasattr(tb.ta, "logic_small"):
                delattr(tb.ta, "logic_small")
        ta.WindowAnalyzer = ORIG_WINDOW_ANALYZER
        tb.ta.WindowAnalyzer = ORIG_WINDOW_ANALYZER
        ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = ORIG_BIG_THRESHOLD
        tb.ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = ORIG_BIG_THRESHOLD
        ta.TREND_TO_BE_OLD_SECONDS = ORIG_TREND_TO_BE_OLD
        tb.ta.TREND_TO_BE_OLD_SECONDS = ORIG_TREND_TO_BE_OLD

    elapsed = time.time() - t0
    pnl_path = os.path.join(out_dir, "pnl.json")
    pnl = {}
    if os.path.exists(pnl_path):
        try:
            with open(pnl_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                pnl = loaded
        except (OSError, json.JSONDecodeError):
            pnl = {}
    return pnl, elapsed
