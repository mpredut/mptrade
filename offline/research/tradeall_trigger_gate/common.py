#!/usr/bin/env python3
"""Shared instrumentation, logic factories, and isolated runner for tradeall trigger-gate experiments."""
from __future__ import annotations

from collections import deque
import json
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


def make_instrumented_trend_state(
    stats: dict,
    *,
    confirm_threshold_override: Optional[int] = None,
    expiration_trend_time_override: Optional[float] = None,
    retry_interval_sec: float = 1800.0,
):
    """Create a fresh TrendState subclass rooted at ORIG_TREND_STATE (never chained across runs)."""

    class InstrumentedTrendState(ORIG_TREND_STATE):
        def __init__(self, *a, **kw):
            if expiration_trend_time_override is not None:
                if len(a) >= 2:
                    a = (a[0], expiration_trend_time_override, *a[2:])
                else:
                    kw["expiration_trend_time"] = expiration_trend_time_override
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
            if confirm_threshold_override is None:
                r = super().is_trend_consistent_validated()
            elif not self.is_trend_a_minim_validated():
                r = False
            else:
                r = (
                    self.confirm_count > confirm_threshold_override
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
            if direction == "UP":
                self._confirmed_up = True
            else:
                self._confirmed_down = True

        def can_retry(self, direction: str, ts: float) -> bool:
            last = self._last_attempt_up_ts if direction == "UP" else self._last_attempt_down_ts
            return last is None or (ts - last) >= retry_interval_sec

        def mark_attempt(self, direction: str, ts: float):
            if direction == "UP":
                self._last_attempt_up_ts = ts
            else:
                self._last_attempt_down_ts = ts

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

        if slope <= -5.1 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire("UP", "BUY", "slope<=-5.1_up")
        if slope >= 5.1 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire("DOWN", "SELL", "slope>=5.1_down")

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
        self.window_sec = window_hours * 3600.0
        self.reeval_sec = reeval_sec
        self.min_points = min_points
        self.buf = deque()
        self.last_eval_ts: Optional[float] = None
        self.sign = 0
        self.fired_up = False
        self.fired_down = False
        self.last_attempt_up_ts: Optional[float] = None
        self.last_attempt_down_ts: Optional[float] = None
        self.stats = stats

    def update(self, ts: float, price: float) -> int:
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
                if slope is not None:
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

    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price, price_window=None, **kw):
        tr = tracker_getter(symbol)
        ts = trend_state._now()
        sign = tr.update(ts, current_price)
        stats = tr.stats

        if sign > 0 and not tr.fired_up:
            can_retry = (
                tr.last_attempt_up_ts is None
                or (ts - tr.last_attempt_up_ts) >= retry_interval_sec
            )
            if not can_retry:
                if "fire_skipped_cooldown_up" in stats:
                    stats["fire_skipped_cooldown_up"] += 1
            else:
                tr.last_attempt_up_ts = ts
                stats["fire_attempts_up"] += 1
                if enable:
                    result = ta._fire_order(
                        symbol, "BUY", current_price, f"{tag}_{up_reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )
                    if result is not None:
                        tr.fired_up, tr.fired_down = True, False
                        stats["fire_confirmed_up"] += 1
                    else:
                        stats["fire_blocked_up"] += 1

        if sign < 0 and not tr.fired_down:
            can_retry = (
                tr.last_attempt_down_ts is None
                or (ts - tr.last_attempt_down_ts) >= retry_interval_sec
            )
            if not can_retry:
                if "fire_skipped_cooldown_down" in stats:
                    stats["fire_skipped_cooldown_down"] += 1
            else:
                tr.last_attempt_down_ts = ts
                stats["fire_attempts_down"] += 1
                if enable:
                    result = ta._fire_order(
                        symbol, "SELL", current_price, f"{tag}_{down_reason}",
                        safeback_seconds=DEFAULT_SAFEBACK_SECONDS, force=False, cancelorders=True, hours=1,
                    )
                    if result is not None:
                        tr.fired_down, tr.fired_up = True, False
                        stats["fire_confirmed_down"] += 1
                    else:
                        stats["fire_blocked_down"] += 1

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
            c365 = os.path.join(ROOT, "cachedb", f"cache_365_price_{symbol}.jsonl")
            if not os.path.exists(c365):
                legacy = os.path.join(ROOT, "cachedb", f"cache_365price_{symbol}.jsonl")
                if os.path.exists(legacy):
                    c365 = legacy
            kwargs["cache24_file"] = c365
        tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, source, **kwargs)
    finally:
        ta.TrendState = ORIG_TREND_STATE
        tb.ta.TrendState = ORIG_TREND_STATE
        ta.logic = ORIG_LOGIC
        tb.ta.logic = ORIG_LOGIC
        if ORIG_LOGIC_SMALL is not None:
            ta.logic_small = ORIG_LOGIC_SMALL
            tb.ta.logic_small = ORIG_LOGIC_SMALL
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
        with open(pnl_path, encoding="utf-8") as fh:
            pnl = json.load(fh)
    return pnl, elapsed
