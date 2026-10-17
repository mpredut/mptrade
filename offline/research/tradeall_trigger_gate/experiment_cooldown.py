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

from offline.research.tradeall_trigger_gate.common import (
    ORIG_BIG_THRESHOLD,
    execute_tradeall_variant,
    make_instrumented_trend_state,
    make_tradeall_logic,
)

STATS = {}


def make_instrumented(tag):
    stats = {
        "starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0,
        "fired_instances_up": 0, "fired_instances_down": 0,
    }
    STATS[tag] = stats
    return make_instrumented_trend_state(stats)


def make_logic_cooldown(start_up_cond, start_down_cond, label):
    """A copy of logic() from tradeall.py, with TWO variations: the start condition
    (parameterised, as in Experiment 2) AND a fire-once cooldown per direction
    per trend instance (it gates ALL firing blocks, not just one)."""
    return make_tradeall_logic(
        label,
        start_up_cond=start_up_cond,
        start_down_cond=start_down_cond,
        cooldown_mode="fire_once",
    )


VARIANTS = {
    "V0_baseline_cooldown": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, None),
    "V1_doar_gradient_cooldown": (lambda g, s: g > 0, lambda g, s: g < 0, None),
    "V3a_big_threshold_half_cooldown": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, 2.0),
    "V3b_big_threshold_third_cooldown": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, 3.0),
}


def run_variant(tag, up_cond, down_cond, threshold_divisor, symbol, start_ts, end_ts):
    big_thr = (ORIG_BIG_THRESHOLD / threshold_divisor) if threshold_divisor else None
    pnl, elapsed = execute_tradeall_variant(
        f"experiment3_{tag}_{symbol}",
        symbol,
        start_ts,
        end_ts,
        source="cache24",
        logic_fn=make_logic_cooldown(up_cond, down_cond, tag),
        trend_state_cls=make_instrumented(tag),
        big_threshold=big_thr,
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

    for tag, (up, down, thr) in VARIANTS.items():
        run_variant(tag, up, down, thr, "BTCUSDC", btc_start, btc_end)

    for tag, (up, down, thr) in VARIANTS.items():
        run_variant(tag, up, down, thr, "TAOUSDC", tao_start, tao_end)

