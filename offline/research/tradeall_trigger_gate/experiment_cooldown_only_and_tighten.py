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

from offline.backtests import tradeall as tb
import tradeall as ta

STATS = {}

MIN_RETRY_INTERVAL_SEC = 1800.0   # 30 min between BLOCKED attempts (as in Exp6).


def make_instrumented(tag, confirm_threshold_override=None):
    stats = STATS.setdefault(tag, {
        "starts": 0, "confirms": 0, "expires": 0,
        "fire_confirmed_up": 0, "fire_blocked_up": 0, "fire_skipped_cooldown_up": 0,
        "fire_confirmed_down": 0, "fire_blocked_down": 0, "fire_skipped_cooldown_down": 0,
    })
    Base = ta.TrendState

    class InstrumentedTrendState(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._last_attempt_up_ts = None
            self._last_attempt_down_ts = None
            self._confirmed_up = False
            self._confirmed_down = False

        def start_trend(self, new_state):
            stats["starts"] += 1
            self._confirmed_up = False
            self._confirmed_down = False
            return super().start_trend(new_state)

        def already_confirmed(self, direction):
            return self._confirmed_up if direction == "UP" else self._confirmed_down

        def mark_confirmed(self, direction):
            if direction == "UP":
                self._confirmed_up = True
            else:
                self._confirmed_down = True

        def confirm_trend(self):
            r = super().confirm_trend()
            stats["confirms"] += 1
            return r

        def check_trend_expiration(self):
            was_expired = self.expired
            r = super().check_trend_expiration()
            if r and not was_expired:
                stats["expires"] += 1
            return r

        def is_trend_consistent_validated(self):
            if confirm_threshold_override is None:
                return super().is_trend_consistent_validated()
            if not self.is_trend_a_minim_validated():
                return False
            return (self.confirm_count > confirm_threshold_override
                    and self.is_trend_uniform_confirmed())

        def can_retry(self, direction, ts):
            last = self._last_attempt_up_ts if direction == "UP" else self._last_attempt_down_ts
            return last is None or (ts - last) >= MIN_RETRY_INTERVAL_SEC

        def mark_attempt(self, direction, ts):
            if direction == "UP":
                self._last_attempt_up_ts = ts
            else:
                self._last_attempt_down_ts = ts

    return InstrumentedTrendState, stats


def make_logic_real_with_cooldown(tag, stats):
    """A FAITHFUL copy of logic() from tradeall.py (lines 356-471 on 21-22 Jul) —
    NOTHING changed in the decision blocks, ONLY a cooldown added to each
    fire point (a confirmed execution plus a minimum interval between retries)."""
    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price):
        d = 14
        h = 24
        proposed_price = current_price

        def fire(direction, action, reason):
            if trend_state.already_confirmed(direction):
                return
            ts = trend_state._now()
            if not trend_state.can_retry(direction, ts):
                stats[f"fire_skipped_cooldown_{direction.lower()}"] += 1
                return
            trend_state.mark_attempt(direction, ts)
            if enable:
                result = ta._fire_order(symbol, action, proposed_price, f"{tag}_{reason}",
                                         safeback_seconds=d * h * 3600 + 60, force=False,
                                         cancelorders=True, hours=1)
                if result is not None:
                    trend_state.mark_confirmed(direction)
                    stats[f"fire_confirmed_{direction.lower()}"] += 1
                else:
                    stats[f"fire_blocked_{direction.lower()}"] += 1

        if gradient > 0 and slope < 0:
            proposed_price = current_price
            if trend_state.is_trend_up():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    fire("UP", "BUY", "trend_confirmed_up")
            else:
                trend_state.start_trend('UP')

        if gradient < 0 and slope > 0:
            proposed_price = current_price
            if trend_state.is_trend_down():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    fire("DOWN", "SELL", "trend_confirmed_down")
            else:
                trend_state.start_trend('DOWN')

        proposed_price = current_price
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


def run_variant(tag, confirm_threshold_override, symbol, start_ts, end_ts):
    ta.TrendState, stats = make_instrumented(tag, confirm_threshold_override)
    tb.ta.TrendState = ta.TrendState
    ta.logic = make_logic_real_with_cooldown(tag, stats)
    tb.ta.logic = ta.logic

    run_id = f"experiment7_{tag}_{symbol}"
    import shutil
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    # IMPORTANT: logic() reala are constante de timp SCURTE (expirare 2.7min,
    # fresh 3.7min) — gandite pt tick-uri dese (~1s, ca in live). Istoricul
    # SPARSE (cache_price_*.jsonl, ~7min/tick) would make any trend "expire"
    # almost instantly, an artefact of the data's sparseness rather than of the real market
    # (verified: a 12h smoke test -> confirms=0, expires=1 immediately). That is why
    # we use the DENSE archive (cache24, ~1s/tick, 7 days) — the only source
    # compatible with this mechanism, even though the sample is smaller than
    # Experiment 6 (there the signal was recomputed rarely, every 30 min, compatible with
    # sparse).
    t0 = time.time()
    tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "cache24",
                     cache24_file=os.path.join(ROOT, "cachedb", f"cache_365price_{symbol}.jsonl"),
                     quiet=True, kalman_primary=False)
    elapsed = time.time() - t0

    import json
    pnl_path = os.path.join(out_dir, "pnl.json")
    pnl = json.load(open(pnl_path)) if os.path.exists(pnl_path) else {}
    sys.stderr.write(f"\n=== {tag} / {symbol} === (wall {elapsed:.1f}s)\n")
    sys.stderr.write(f"stats: {stats}\n")
    sys.stderr.write(f"pnl: {pnl}\n")


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
