#!/usr/bin/env python3
"""
Experiment 7 (isolated, it does NOT modify tradeall.py) — an answer to 3 user questions
after Experiment 6:
  1. "is it worth committing ONLY the cooldown (without touching the thresholds)?"
  2. "do all the experiments show that the current variant is the best?"
  3. "does tightening the parameters make sense as a test?"

Aici testam, pe intreg istoricul disponibil (329 zile, cache_price_*.jsonl):

  A. "current_cooldown"      : the REAL logic() from tradeall.py (the start condition,
                                the gradient/slope_big divergence, UNCHANGED, all
                                pragurile 5.1/24-confirmari/TREND_TO_BE_OLD_SECONDS
                                UNCHANGED) plus ONLY the cooldown (a confirmed
                                execution plus a minimum interval between retries,
                                the final variant from Experiment 6). It answers
                                question 1 directly: what would happen if we
                                commit ONLY the cooldown, with no other change?

  B. "tighten_confirm48_cooldown" : as above, but the confirmation threshold for
                                is_trend_consistent_validated() DUBLAT (24->48)
                                — it tests question 3 (tightening, not relaxing).
                                The motivation: TAO accumulated 99 confirmations in
                                a single trend and easily reached the threshold of 24; a
                                threshold that is harder to reach might avoid the whole
                                problematic episode.

logic() is a FAITHFUL copy of the function in tradeall.py (every block, the
5.1 and TREND_TO_BE_OLD_SECONDS unchanged) — the ONLY difference from the real
code is the cooldown added at every point where _fire_order would be called, plus
(for variant B only) overriding is_trend_consistent_validated().
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.research.tradeall_trigger_gate.common import (
    execute_tradeall_variant,
    make_instrumented_trend_state,
    make_tradeall_logic,
)

STATS = {}

MIN_RETRY_INTERVAL_SEC = 1800.0   # 30 min between BLOCKED attempts (as in Exp6).


def make_instrumented(tag, confirm_threshold_override=None):
    stats = {
        "starts": 0, "confirms": 0, "expires": 0,
        "fire_confirmed_up": 0, "fire_blocked_up": 0, "fire_skipped_cooldown_up": 0,
        "fire_confirmed_down": 0, "fire_blocked_down": 0, "fire_skipped_cooldown_down": 0,
    }
    STATS[tag] = stats
    cls = make_instrumented_trend_state(
        stats,
        confirm_threshold_override=confirm_threshold_override,
        retry_interval_sec=MIN_RETRY_INTERVAL_SEC,
    )
    return cls, stats


def make_logic_real_with_cooldown(tag, stats):
    """A FAITHFUL copy of logic() from tradeall.py (lines 356-471 on 21-22 Jul) —
    NOTHING changed in the decision blocks, ONLY a cooldown added to each
    fire point (a confirmed execution plus a minimum interval between retries)."""
    return make_tradeall_logic(tag, cooldown_mode="confirmed_retry", stats=stats)


def run_variant(tag, confirm_threshold_override, symbol, start_ts, end_ts):
    trend_cls, stats = make_instrumented(tag, confirm_threshold_override)

    # IMPORTANT: logic() reala are constante de timp SCURTE (expirare 2.7min,
    # fresh 3.7min) — gandite pt tick-uri dese (~1s, ca in live). Istoricul
    # SPARSE (cache_price_*.jsonl, ~7min/tick) would make any trend "expire"
    # almost instantly, an artefact of the data's sparseness rather than of the real market
    # (verified: a 12h smoke test -> confirms=0, expires=1 immediately). That is why
    # we use the DENSE archive (cache24, ~1s/tick, 7 days) — the only source
    # compatible with this mechanism, even though the sample is smaller than
    # Experiment 6 (there the signal was recomputed rarely, every 30 min, compatible with
    # sparse).
    pnl, elapsed = execute_tradeall_variant(
        f"experiment7_{tag}_{symbol}",
        symbol,
        start_ts,
        end_ts,
        source="cache24",
        logic_fn=make_logic_real_with_cooldown(tag, stats),
        trend_state_cls=trend_cls,
    )
    sys.stderr.write(f"\n=== {tag} / {symbol} === (wall {elapsed:.1f}s)\n")
    sys.stderr.write(f"stats: {stats}\n")
    sys.stderr.write(f"pnl: {pnl}\n")
    return stats, pnl


if __name__ == "__main__":
    from datetime import datetime
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, choices=["current_cooldown", "tighten_confirm48_cooldown"])
    p.add_argument("--symbol", required=True)
    args = p.parse_args()

    dense_start = datetime.strptime("2026-07-14", "%Y-%m-%d").timestamp()
    threshold = 48 if args.tag == "tighten_confirm48_cooldown" else None
    run_variant(args.tag, threshold, args.symbol, dense_start, None)

