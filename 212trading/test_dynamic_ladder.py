#!/usr/bin/env python3
"""212trading/test_dynamic_ladder.py — Unit tests for granular and dynamic TP ladder."""

import unittest
from strategy import (
    generate_tp_ladder,
    adapt_ladder_rungs,
    StratParams,
)


class TestDynamicTPLadder(unittest.TestCase):
    def test_generate_tp_ladder_linear_10_steps(self):
        ladder = generate_tp_ladder(9.0, 40.0, 10, curve="linear")
        self.assertEqual(len(ladder), 10)
        self.assertEqual(ladder[0][0], 9.0)
        self.assertEqual(ladder[-1][0], 40.0)

        # Check levels are strictly increasing
        levels = [lvl for lvl, _ in ladder]
        self.assertEqual(levels, sorted(levels))

        # Check weights sum to 1.0 (within float tolerance)
        total_frac = sum(frac for _, frac in ladder)
        self.assertAlmostEqual(total_frac, 1.0, places=3)
        # In linear 10 steps, each fraction is 0.1
        for _, frac in ladder:
            self.assertEqual(frac, 0.1)

    def test_generate_tp_ladder_curves(self):
        # Front-heavy: earlier levels have larger fractions
        front = generate_tp_ladder(9.0, 40.0, 5, curve="front_heavy")
        self.assertEqual(len(front), 5)
        fracs_front = [frac for _, frac in front]
        self.assertEqual(fracs_front, sorted(fracs_front, reverse=True))

        # Back-heavy: later levels have larger fractions
        back = generate_tp_ladder(9.0, 40.0, 5, curve="back_heavy")
        self.assertEqual(len(back), 5)
        fracs_back = [frac for _, frac in back]
        self.assertEqual(fracs_back, sorted(fracs_back))

        # Geometric levels
        geo = generate_tp_ladder(9.0, 40.0, 5, curve="geometric")
        self.assertEqual(len(geo), 5)
        self.assertEqual(geo[0][0], 9.0)
        self.assertEqual(geo[-1][0], 40.0)

        # Bell curve: smaller at edges, bulk in the middle
        bell = generate_tp_ladder(9.0, 40.0, 5, curve="bell")
        self.assertEqual(len(bell), 5)
        fracs_bell = [frac for _, frac in bell]
        # Middle (index 2) must be the highest fraction
        self.assertEqual(fracs_bell[2], max(fracs_bell))
        # Start and end fractions must be lower than adjacent inner fractions
        self.assertLess(fracs_bell[0], fracs_bell[1])
        self.assertLess(fracs_bell[4], fracs_bell[3])
        self.assertAlmostEqual(sum(fracs_bell), 1.0, places=3)

    def test_adapt_ladder_rungs(self):
        full = generate_tp_ladder(9.0, 40.0, 10)

        # Sufficient volume: retains all 10 rungs
        adapted_full = adapt_ladder_rungs(full, held=4.57)
        self.assertEqual(len(adapted_full), 10)

        # Low volume: 0.05 shares can support at most 5 rungs of >= 0.01
        adapted_5 = adapt_ladder_rungs(full, held=0.05, min_tranche_qty=0.01)
        self.assertEqual(len(adapted_5), 5)
        self.assertEqual(adapted_5[0][0], 9.0)
        self.assertEqual(adapted_5[-1][0], 40.0)

        # Very low volume: 0.02 shares
        adapted_2 = adapt_ladder_rungs(full, held=0.02, min_tranche_qty=0.01)
        self.assertEqual(len(adapted_2), 2)
        self.assertEqual(adapted_2[0][0], 9.0)
        self.assertEqual(adapted_2[-1][0], 40.0)

        # 0 volume
        self.assertEqual(adapt_ladder_rungs(full, held=0.0), [])

    def test_from_env_range_parsing(self):
        base_env = {
            "STRATEGY_MODE": "avg_tp",
            "STRAT_CURRENCY": "USD",
            "STRAT_ENTRY_PCT": "6",
            "STRAT_ENTRY_DISCOUNT_PCT": "0.5",
            "STRAT_DCA_PCT": "10",
            "STRAT_DCA_DROP_PCT": "4.0",
            "STRAT_CHECK_MINUTES": "0.3",
            "STRAT_TAKEPROFIT_PCT": "9.0",
            "STRAT_MAX_BUDGET": "850",
            "STRAT_MAX_DCA_BUYS": "auto",
            "STRAT_ORDER_TTL_MIN": "3",
            "STRAT_STOP_LOSS_PCT": "30",
            "YAHOO_SYMBOL": "SPCX",
            "STRAT_REENTRY_DROP_PCT": "2.0",
            "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
            "STRAT_FX_FEE_PCT": "0.15",
            "STRAT_LOSS_ALERT_STEP": "1",
            "STRAT_LADDER_MIN_FREE": "6",
            "STRAT_SL_REBUY_ENABLED": "true",
            "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
            "STRAT_DCA_TREND_GATE_PCT": "0.10",
            "STRAT_TRAIL_PCT": "8",
            "STRAT_TRAIL_MIN_PROFIT_PCT": "9",
            "STRAT_REENTRY_HYBRID_ENABLED": "false",
            "STRAT_REENTRY_PEAK_RELATIVE": "false",
            "STRAT_REENTRY_PULLBACK_PCT": "1.5",
            "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false",
            "STRAT_REENTRY_PULLBACK_K": "1.0",
            "STRAT_REENTRY_PULLBACK_MIN": "0.8",
            "STRAT_REENTRY_PULLBACK_MAX": "4.0",
            "STRAT_REENTRY_TTL_HOURS": "0.0",
            "STRAT_SL_MHDSL_ENABLED": "false",
            "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
            "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0",
            "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
            "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
            "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
            "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true",
            "STRAT_TP_MDHTP_ENABLED": "false",
            "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
        }

        # Case 1: range:9:40:10
        env1 = dict(base_env, STRAT_TP_LADDER="range:9:40:10")
        p1 = StratParams.from_env(env1)
        self.assertEqual(len(p1.tp_ladder), 10)
        self.assertEqual(p1.tp_ladder[0][0], 9.0)
        self.assertEqual(p1.tp_ladder[-1][0], 40.0)

        # Case 2: 9:40:10 without range prefix
        env2 = dict(base_env, STRAT_TP_LADDER="9:40:10")
        p2 = StratParams.from_env(env2)
        self.assertEqual(len(p2.tp_ladder), 10)
        self.assertEqual(p2.tp_ladder[0][0], 9.0)

        # Case 3: STRAT_TP_LADDER_MIN / MAX / STEPS
        env3 = dict(base_env, STRAT_TP_LADDER="", STRAT_TP_LADDER_MIN="9.0",
                    STRAT_TP_LADDER_MAX="40.0", STRAT_TP_LADDER_STEPS="10")
        p3 = StratParams.from_env(env3)
        self.assertEqual(len(p3.tp_ladder), 10)
        self.assertEqual(p3.tp_ladder[0][0], 9.0)
        self.assertEqual(p3.tp_ladder[-1][0], 40.0)

        # Case 4: Backward compatibility with legacy comma-separated list
        env4 = dict(base_env, STRAT_TP_LADDER="11:33,20:33,30:34")
        p4 = StratParams.from_env(env4)
        self.assertEqual(len(p4.tp_ladder), 3)
        self.assertEqual(p4.tp_ladder[0][0], 11.0)
        self.assertEqual(p4.tp_ladder[1][0], 20.0)
        self.assertEqual(p4.tp_ladder[2][0], 30.0)


if __name__ == "__main__":
    unittest.main()
