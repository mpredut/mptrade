#!/usr/bin/env python3
"""
Experiment 5 (isolated, it does NOT modify tradeall.py) — a user request, after
Experiments 1-4: "if I am on a larger general uptrend and I buy, I should be
in profit, right? test that."

Experiments 1-4 showed that gradient/slope_big/gradient_big are all
regressions over windows of MINUTES TO HOURS, recomputed on EVERY TICK — so
noisy on the same time scale, however "large" the window sounds (2.5h).
None of them was a GENUINE long-lived trend.

We test a qualitatively DIFFERENT signal here:
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

from offline.research.tradeall_trigger_gate.common import (
    LongTrendTracker,
    execute_tradeall_variant,
    make_tracker_logic,
)

STATS = {}

MIN_RETRY_INTERVAL_SEC = 1800.0
_trackers = {}


def make_quality_logic(tag, window_hours, reeval_sec):
    stats = {
        "sign_changes": 0,
        "fire_attempts_up": 0, "fire_confirmed_up": 0, "fire_blocked_up": 0,
        "fire_attempts_down": 0, "fire_confirmed_down": 0, "fire_blocked_down": 0,
    }
    STATS[tag] = stats

    def get_tracker(symbol):
        tr = _trackers.get(symbol)
        if tr is None:
            tr = _trackers[symbol] = LongTrendTracker(stats, window_hours, reeval_sec)
        return tr

    return make_tracker_logic(tag, get_tracker, retry_interval_sec=MIN_RETRY_INTERVAL_SEC)


def run_variant(tag, window_hours, reeval_sec, symbol, start_ts, end_ts):
    _trackers.clear()
    pnl, elapsed = execute_tradeall_variant(
        f"experiment5_{tag}_{symbol}",
        symbol,
        start_ts,
        end_ts,
        source="cache24",
        logic_fn=make_quality_logic(tag, window_hours, reeval_sec),
        disable_small=True,
    )
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

