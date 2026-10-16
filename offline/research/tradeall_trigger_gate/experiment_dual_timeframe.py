#!/usr/bin/env python3
"""
Experiment 4 (isolated, it does NOT modify tradeall.py) — a user request: "what each
variable represents does not help us much, I would like to use intuition on
the simple strategy" — that is: if both windows (small AND large) agree
about the DIRECTION, using the same CONTINUOUS measure (not one continuous plus one
rare/threshold-based), ought to be a sound "trend-following on two
orizonturi de timp".

The problem found (an answer to the question about are_close): slope_big is NOT a
continuous measure of direction — it is EXACTLY 0 (the "no large move" sentinel) until
when the price crosses a fixed threshold relative to the window extreme (WindowAnalyzer.
check_price_change). gradient (the small window) is continuous (almost never
exactly 0). "Agreement" between them (gradient>0 and slope_big>0) requires an event
RARE event (a non-zero slope_big) to coincide with a NOISY event (the sign of
gradient) — de-asta Experimentul 2 a dat aproape 0 activitate pe acord.

The fix tested here: replace slope_big with gradient_big — the SAME derivation as
gradient (PriceWindow.get_instant_trend(), semn -1/0/+1 dintr-o regresie
continuous), but computed on the LARGE WINDOW instead of the small one. Now "agreement"
means something honest: "the short-term AND the long-term trend are
agree on the direction" — a classic dual-timeframe strategy, not a
coincidenta rara.

The fire-once cooldown (from Experiment 3) is always ACTIVE here — it has already been shown
that without it any frequency increase leads to catastrophic overtrading; we do
not retest "without cooldown" so as not to waste time on a known result.

Variants (on BTC over 2 days / TAO over 12h, the same windows as Experiments 1-3):
  V5_dual_timeframe_agreement : gradient(small)>0 AND gradient_big(large)>0 -> UP
                            gradient(small)<0 AND gradient_big(large)<0 -> DOWN
                            + cooldown fire-once (ca Experimentul 3)
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

_OrigWindowAnalyzer = ta.WindowAnalyzer


class GradientBigWindowAnalyzer(_OrigWindowAnalyzer):
    """check_price_change() now returns gradient_big (a continuous sign, the same
    derivation as PriceWindow.get_instant_trend() used for the small window) in
    loc de slope_big (rar, prag-gated). Semnatura pastrata (val, pos) — pos
    (the second element) is not used by logic()."""
    def check_price_change(self, threshold):
        final_trend, growth_coefficient, slope_full, gradient_recent = self.window.get_instant_trend()
        return final_trend, 0


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
    """Identical to Experiment 3 (a copy of logic() plus the fire-once cooldown) — repeated
    here so the script stays independent, without depending on the previous file."""
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


def run_variant(tag, symbol, start_ts, end_ts):
    ta.TrendState = make_instrumented(tag)
    tb.ta.TrendState = ta.TrendState
    ta.logic = make_logic_cooldown(lambda g, s: g > 0 and s > 0,
                                    lambda g, s: g < 0 and s < 0, tag)
    tb.ta.logic = ta.logic
    ta.WindowAnalyzer = GradientBigWindowAnalyzer
    tb.ta.WindowAnalyzer = ta.WindowAnalyzer

    run_id = f"experiment4_{tag}_{symbol}"
    import shutil
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "cache24",
                     cache24_file=os.path.join(ROOT, "cachedb", f"cache_365price_{symbol}.jsonl"),
                     quiet=True, kalman_primary=False)
    elapsed = time.time() - t0

    ta.WindowAnalyzer = _OrigWindowAnalyzer
    tb.ta.WindowAnalyzer = ta.WindowAnalyzer

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

    run_variant("V5_dual_timeframe_acord", "BTCUSDC", btc_start, btc_end)
    run_variant("V5_dual_timeframe_acord", "TAOUSDC", tao_start, tao_end)

    # a longer window on BTC (7 full days, the same archive as the main A/B
    # backtests) — 2 days show too little data to distinguish "a sound,
    # rare strategy" from "a dead strategy"; over 7 days the test is far more honest.
    btc7_start = datetime.strptime("2026-07-14", "%Y-%m-%d").timestamp()
    run_variant("V5_dual_timeframe_acord_7d", "BTCUSDC", btc7_start, btc7_start + 7 * 24 * 3600)
