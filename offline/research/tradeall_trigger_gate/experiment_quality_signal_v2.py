#!/usr/bin/env python3
"""
Experiment 6 (isolated, it does NOT modify tradeall.py) — the continuation of Experiment 5,
at the user's request: "let's do 3 [fix the cooldown defect] but with
the note that it has to be partly limited. should I test over many more days again?"

Doua schimbari fata de Experimentul 5:

1. THE COOLDOWN FIXED (limited, not unlimited): Experiment 5 showed that
   "retry only if not yet CONFIRMED" is insufficient — with no position
   to sell, a rejected SELL was retried on EVERY tick (16683/3962
   pointless attempts in a week). The fix: MIN_RETRY_INTERVAL_SEC added
   between BLOCKED attempts (not between confirmed ones — a successful execution
   it still blocks until the regime changes, as before). In practice: "try,
   if it fails, wait at least X minutes before retrying", not "never"
   and not "on every tick".

2. ISTORIC MULT MAI LUNG: in loc de arhiva densa de 7 zile (cache24, ~1s/tick),
   we use the SPARSE history (`cache_price_{symbol}.jsonl`, ~7 min/tick, 329
   days available — verified: 2025-08-27 -> today). The signal being tested (a regression
   over 12-24h, recomputed rarely) does not need second-level resolution — 7 min is
   fine enough for a window of hours. The goal: a single 7-day test can
   fi noroc/ghinion; 329 de zile acopera multe regimuri diferite de piata.

Variants (on BTC and TAO, the COMPLETE history available):
  V7_trend24h_reeval30min_retrylimit  : a 24h window, re-evaluation every 30 min (as in Exp5)
  V7_trend12h_reeval15min_retrylimit  : a 12h window, re-evaluation every 15 min (faster)

MIN_RETRY_INTERVAL_SEC = 1800 (30 min) pt incercarile blocate, pe ambele variante.
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

MIN_RETRY_INTERVAL_SEC = 1800.0   # 30 min between BLOCKED attempts (not between confirmed ones)
_trackers = {}


def _make_stats(tag):
    stats = {
        "sign_changes": 0,
        "fire_attempts_up": 0, "fire_confirmed_up": 0, "fire_blocked_up": 0, "fire_skipped_cooldown_up": 0,
        "fire_attempts_down": 0, "fire_confirmed_down": 0, "fire_blocked_down": 0, "fire_skipped_cooldown_down": 0,
    }
    STATS[tag] = stats
    return stats


def make_quality_logic(tag):
    return make_tracker_logic(
        tag,
        lambda symbol: _trackers[(tag, symbol)],
        retry_interval_sec=MIN_RETRY_INTERVAL_SEC,
    )


def run_variant(tag, window_hours, reeval_sec, symbol, start_ts, end_ts):
    stats = _make_stats(tag)
    _trackers[(tag, symbol)] = LongTrendTracker(stats, window_hours, reeval_sec)
    pnl, elapsed = execute_tradeall_variant(
        f"experiment6_{tag}_{symbol}",
        symbol,
        start_ts,
        end_ts,
        source="history",
        logic_fn=make_quality_logic(tag),
        disable_small=True,
    )
    sys.stderr.write(f"\n=== {tag} / {symbol} === (wall {elapsed:.1f}s)\n")
    sys.stderr.write(f"stats: {STATS[tag]}\n")
    sys.stderr.write(f"pnl: {pnl}\n")
    return STATS[tag], pnl


if __name__ == "__main__":
    from datetime import datetime

    hist_start = datetime.strptime("2025-08-27", "%Y-%m-%d").timestamp()

    for symbol in ("BTCUSDC", "TAOUSDC"):
        run_variant("V7_trend24h_reeval30min_retrylimit", 24, 1800, symbol, hist_start, None)
        run_variant("V7_trend12h_reeval15min_retrylimit", 12, 900, symbol, hist_start, None)

    sys.stderr.write("\n\n===== ALL RUNS FINISHED =====\n")
    for tag, st in STATS.items():
        sys.stderr.write(f"{tag}: {st}\n")

