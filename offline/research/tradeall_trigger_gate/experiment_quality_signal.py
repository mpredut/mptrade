#!/usr/bin/env python3
"""
Experiment 5 (isolated, it does NOT modify tradeall.py) — a user request, after
Experiments 1-4: "if I am on a larger general uptrend and I buy, I should be
in profit, right? test that."

Experiments 1-4 showed that gradient/slope_big/gradient_big are all
regressions over windows of MINUTES TO HOURS, recomputed on EVERY TICK — so
noisy on the same time scale, however "large" the window sounds (2.5h).
None of them was a GENUINE long-lived trend.

Testam aici un semnal calitativ DIFERIT:
  - LongTrendTracker: a linear regression over a window of `window_hours` hours
    (24h by default), but RECOMPUTED only once every `reeval_sec` (by default
    1800s = 30 min), not on every tick. That makes it STABLE for hours.
  - A cooldown on a CONFIRMED EXECUTION (not on an attempt): "I have already acted in
    this regime" is set ONLY if _fire_order returns a real result
    (an order placed), not merely because it was called. If an attempt is
    rejected (the Kalman gate; in live also weight limit/budget), we do NOT block
    future retries — exactly the user's requirement from the earlier discussion.

Trading rules: BUY when the signal is positive (AND we have not already executed
successfully in this regime), SELL when negative (symmetric). Without the complex
TrendState from tradeall.py (confirm_count/expiry) — that was part of the problem
(see Experiments 1/3), replaced entirely here by the regime logic above.

An honest caveat: with no existing position, a SELL rejected by the broker (spot,
to sell) is NOT marked separately as a "blocked retry" — it will retry at
every tick until the sign changes OR a position appears (from an earlier
BUY). In this backtest rejected retries cost no commission (only in
production would they consume weight limit/API time) — so they do not affect P&L here,
but it is worth noting as a simplification against production.
"""
import os
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.backtests import tradeall as tb
import tradeall as ta

STATS = {}

MIN_RETRY_INTERVAL_SEC = 1800.0


class LongTrendTracker:
    def __init__(self, tag, window_hours, reeval_sec, min_points=30):
        self.window_sec = window_hours * 3600
        self.reeval_sec = reeval_sec
        self.min_points = min_points
        self.buf = deque()
        self.last_eval_ts = None
        self.sign = 0
        self.fired_up = False
        self.fired_down = False
        self.last_attempt_up_ts = None
        self.last_attempt_down_ts = None
        self.stats = STATS.setdefault(tag, {
            "sign_changes": 0,
            "fire_attempts_up": 0, "fire_confirmed_up": 0, "fire_blocked_up": 0,
            "fire_attempts_down": 0, "fire_confirmed_down": 0, "fire_blocked_down": 0,
        })

    def update(self, ts, price):
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
                    self.stats["sign_changes"] += 1
        return self.sign


_trackers = {}


def make_quality_logic(tag, window_hours, reeval_sec):
    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price):
        tr = _trackers.get(symbol)
        if tr is None:
            tr = _trackers[symbol] = LongTrendTracker(tag, window_hours, reeval_sec)
        ts = trend_state._now()
        sign = tr.update(ts, current_price)
        stats = tr.stats

        if sign > 0 and not tr.fired_up:
            if tr.last_attempt_up_ts is None or (ts - tr.last_attempt_up_ts) >= MIN_RETRY_INTERVAL_SEC:
                tr.last_attempt_up_ts = ts
                stats["fire_attempts_up"] += 1
                if enable:
                    result = ta._fire_order(symbol, "BUY", current_price, f"{tag}_long_trend_up",
                                             safeback_seconds=14 * 24 * 3600 + 60, force=False,
                                             cancelorders=True, hours=1)
                    if result is not None:
                        tr.fired_up = True
                        tr.fired_down = False
                        stats["fire_confirmed_up"] += 1
                    else:
                        stats["fire_blocked_up"] += 1

        if sign < 0 and not tr.fired_down:
            if tr.last_attempt_down_ts is None or (ts - tr.last_attempt_down_ts) >= MIN_RETRY_INTERVAL_SEC:
                tr.last_attempt_down_ts = ts
                stats["fire_attempts_down"] += 1
                if enable:
                    result = ta._fire_order(symbol, "SELL", current_price, f"{tag}_long_trend_down",
                                             safeback_seconds=14 * 24 * 3600 + 60, force=False,
                                             cancelorders=True, hours=1)
                    if result is not None:
                        tr.fired_down = True
                        tr.fired_up = False
                        stats["fire_confirmed_down"] += 1
                    else:
                        stats["fire_blocked_down"] += 1

    return logic_variant


def run_variant(tag, window_hours, reeval_sec, symbol, start_ts, end_ts):
    _trackers.clear()
    ta.logic = make_quality_logic(tag, window_hours, reeval_sec)
    tb.ta.logic = ta.logic

    run_id = f"experiment5_{tag}_{symbol}"
    import shutil
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "cache24",
                     cache24_file=os.path.join(ROOT, "cachedb", f"cache_24price_long_{symbol}.jsonl"),
                     quiet=True, kalman_primary=False)
    elapsed = time.time() - t0

    import json
    pnl_path = os.path.join(out_dir, "pnl.json")
    pnl = json.load(open(pnl_path)) if os.path.exists(pnl_path) else {}
    sys.stderr.write(f"\n=== {tag} / {symbol} === (wall {elapsed:.1f}s)\n")
    sys.stderr.write(f"stats: {STATS[tag]}\n")
    sys.stderr.write(f"pnl: {pnl}\n")
    return STATS[tag], pnl


if __name__ == "__main__":
    from datetime import datetime

    btc_start = datetime.strptime("2026-07-14", "%Y-%m-%d").timestamp()
    tao_start = datetime.strptime("2026-07-14 19:40:00", "%Y-%m-%d %H:%M:%S").timestamp()
    seven_days = 7 * 24 * 3600

    # 7 full days for both symbols — a 24h signal needs
    # enough history to show more than 1-2 regime changes.
    run_variant("V6_trend24h_reeval30min", 24, 1800, "BTCUSDC", btc_start, btc_start + seven_days)
    run_variant("V6_trend24h_reeval30min", 24, 1800, "TAOUSDC", tao_start, tao_start + seven_days)
