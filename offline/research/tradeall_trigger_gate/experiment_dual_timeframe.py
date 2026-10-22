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

from offline.research.tradeall_trigger_gate.common import (
    ORIG_WINDOW_ANALYZER,
    execute_tradeall_variant,
    make_instrumented_trend_state,
    make_tradeall_logic,
)

STATS = {}


class GradientBigWindowAnalyzer(ORIG_WINDOW_ANALYZER):
    """check_price_change() now returns gradient_big (a continuous sign, the same
    derivation as PriceWindow.get_instant_trend() used for the small window) in
    loc de slope_big (rar, prag-gated). Semnatura pastrata (val, pos) — pos
    (the second element) is not used by logic()."""
    def check_price_change(self, threshold):
        final_trend, growth_coefficient, slope_full, gradient_recent = self.window.get_short_trend()
        return final_trend, 0


def make_instrumented(tag):
    stats = {
        "starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0,
        "fired_instances_up": 0, "fired_instances_down": 0,
    }
    STATS[tag] = stats
    return make_instrumented_trend_state(stats)


def make_logic_cooldown(start_up_cond, start_down_cond, label):
    """Identical to Experiment 3 (a copy of logic() plus the fire-once cooldown)."""
    return make_tradeall_logic(
        label,
        start_up_cond=start_up_cond,
        start_down_cond=start_down_cond,
        cooldown_mode="fire_once",
    )


def run_variant(tag, symbol, start_ts, end_ts):
    pnl, elapsed = execute_tradeall_variant(
        f"experiment4_{tag}_{symbol}",
        symbol,
        start_ts,
        end_ts,
        source="cache24",
        logic_fn=make_logic_cooldown(lambda g, s: g > 0 and s > 0, lambda g, s: g < 0 and s < 0, tag),
        trend_state_cls=make_instrumented(tag),
        window_analyzer_cls=GradientBigWindowAnalyzer,
        disable_small=True,
    )
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

