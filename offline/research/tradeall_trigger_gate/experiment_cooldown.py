#!/usr/bin/env python3
"""
Experiment 3 (isolated, it does NOT modify tradeall.py) — it tests the very fix proposed
in the report (the memory note tradeall-trigger-gate-investigation.md): a cooldown,
"fire only once per trend instance" (edge-triggered, not level-triggered
as it is now), combined with a wider START condition (which in Experiment 2
produced catastrophic overtrading WITHOUT a cooldown).

Intrebare: cooldown-ul elimina overtrading-ul (fees/refire) pastrand totusi
the extra starts as distinct, useful opportunities, or whether those starts
extra ones were noise anyway (many, but worthless) even with a cooldown?

Variants (on BTC over 2 days and TAO over 12h, the same windows as Experiments 1-2):
  V0_baseline_cooldown        : conditia ACTUALA (divergenta) + cooldown fire-once
  V1_gradient_only_cooldown   : the V1 condition (the gradient sign only) plus the fire-once cooldown
  V3a_big_threshold_half_cooldown   : PRICE_CHANGE_THRESHOLD_BIG_EUR /2  + cooldown fire-once
  V3b_big_threshold_third_cooldown: PRICE_CHANGE_THRESHOLD_BIG_EUR /3  + cooldown fire-once

Cooldown-ul: un singur flag per TrendState per directie (_fired_up/_fired_down),
reset on every new start_trend(); every FIRE block (trend_confirmed,
consistent_or_old, slope>=5.1 and so on) checks the flag BEFORE calling _fire_order
and sets it afterwards. So: at most ONE real order per trend instance, regardless
cate tick-uri ramane validat.
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


def make_instrumented(tag):
    stats = {"starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0,
              "fired_instances_up": 0, "fired_instances_down": 0}
    STATS[tag] = stats
    Base = ta.TrendState

    class InstrumentedTrendState(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._fired_up = False
            self._fired_down = False

        def start_trend(self, new_state):
            stats["starts"] += 1
            self._fired_up = False
            self._fired_down = False
            return super().start_trend(new_state)

        def confirm_trend(self):
            r = super().confirm_trend()
            stats["confirms"] += 1
            stats["max_confirm_count"] = max(stats["max_confirm_count"], self.confirm_count)
            return r

        def check_trend_expiration(self):
            was_expired = self.expired
            r = super().check_trend_expiration()
            if r and not was_expired:
                stats["expires"] += 1
            return r

        def mark_fired(self, direction):
            if direction == "UP" and not self._fired_up:
                self._fired_up = True
                stats["fired_instances_up"] += 1
            elif direction == "DOWN" and not self._fired_down:
                self._fired_down = True
                stats["fired_instances_down"] += 1

        def already_fired(self, direction):
            return self._fired_up if direction == "UP" else self._fired_down

    return InstrumentedTrendState


def make_logic_cooldown(start_up_cond, start_down_cond, label):
    """A copy of logic() from tradeall.py, with TWO variations: the start condition
    (parameterised, as in Experiment 2) AND a fire-once cooldown per direction
    per trend instance (it gates ALL firing blocks, not just one)."""
    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price):
        d = 14
        h = 24
        proposed_price = current_price

        def fire_once(direction, action, reason):
            if trend_state.already_fired(direction):
                return
            if enable:
                ta._fire_order(symbol, action, proposed_price, f"{label}_{reason}",
                                safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)
            trend_state.mark_fired(direction)

        if start_up_cond(gradient, slope):
            proposed_price = current_price
            if trend_state.is_trend_up():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    fire_once("UP", "BUY", "trend_confirmed_up")
            else:
                trend_state.start_trend('UP')

        if start_down_cond(gradient, slope):
            proposed_price = current_price
            if trend_state.is_trend_down():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    fire_once("DOWN", "SELL", "trend_confirmed_down")
            else:
                trend_state.start_trend('DOWN')

        proposed_price = current_price
        if slope <= 0 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire_once("UP", "BUY", "consistent_or_old_up")
        if slope >= 0 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire_once("DOWN", "SELL", "consistent_or_old_down")

        if slope <= -5.1 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire_once("UP", "BUY", "slope<=-5.1_up")
        if slope >= 5.1 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire_once("DOWN", "SELL", "slope>=5.1_down")

        if slope <= -5.1 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    and trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire_once("UP", "BUY", "slope<=-5.1_and_old_down")
        if slope >= 5.1 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    and trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                fire_once("DOWN", "SELL", "slope>=5.1_and_old_up")

    return logic_variant


VARIANTS = {
    "V0_baseline_cooldown": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, None),
    "V1_doar_gradient_cooldown": (lambda g, s: g > 0, lambda g, s: g < 0, None),
    "V3a_big_threshold_half_cooldown": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, 2.0),
    "V3b_big_threshold_third_cooldown": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, 3.0),
}


def run_variant(tag, up_cond, down_cond, threshold_divisor, symbol, start_ts, end_ts):
    ta.TrendState = make_instrumented(tag)
    tb.ta.TrendState = ta.TrendState
    ta.logic = make_logic_cooldown(up_cond, down_cond, tag)
    tb.ta.logic = ta.logic

    orig_threshold = ta.PRICE_CHANGE_THRESHOLD_BIG_EUR
    if threshold_divisor:
        ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = orig_threshold / threshold_divisor

    run_id = f"experiment3_{tag}_{symbol}"
    import shutil
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "cache24",
                     cache24_file=os.path.join(ROOT, "cachedb", f"cache_365price_{symbol}.jsonl"),
                     quiet=True, kalman_primary=False)
    elapsed = time.time() - t0

    ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = orig_threshold

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
    btc_end = btc_start + 2 * 24 * 3600

    tao_start = datetime.strptime("2026-07-14 19:40:00", "%Y-%m-%d %H:%M:%S").timestamp()
    tao_end = tao_start + 12 * 3600

    for tag, (up, down, thr) in VARIANTS.items():
        run_variant(tag, up, down, thr, "BTCUSDC", btc_start, btc_end)

    for tag, (up, down, thr) in VARIANTS.items():
        run_variant(tag, up, down, thr, "TAOUSDC", tao_start, tao_end)
