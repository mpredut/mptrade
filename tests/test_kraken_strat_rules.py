"""The pure spot-DCA rules, shared between live and backtest. It checks the formulas
plus that are_close is IDENTICAL to botcore.are_close (proof that the refactor in strategy.py
does NOT change the LIVE behaviour)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_dca_rules as sr


class StratRulesTest(unittest.TestCase):
    def test_entry_and_tp(self):
        self.assertAlmostEqual(sr.entry_price(100.0, 0.2), 99.8)
        self.assertAlmostEqual(sr.tp_price(100.0, 3.0), 103.0)

    def test_hit_stop(self):
        self.assertTrue(sr.hit_stop(100.0, 87.0, 12.5))    # -13% >= 12.5
        self.assertFalse(sr.hit_stop(100.0, 90.0, 12.5))   # -10% < 12.5
        self.assertFalse(sr.hit_stop(100.0, 50.0, 0.0))    # sl dezactivat
        self.assertFalse(sr.hit_stop(0.0, 50.0, 12.5))     # A missing avg.

    def test_reentry_stop_bounce(self):
        # min 50.0, bounce 1.5% -> threshold 50.75; below it = blocked
        self.assertTrue(sr.reentry_stop_blocked(50.0, 50.0, 1.5, 0.0))
        self.assertFalse(sr.reentry_stop_blocked(50.8, 50.0, 1.5, 0.0))  # above the threshold -> it enters

    def test_reentry_drop(self):
        # sold at 100, drop 2% -> threshold 98; above it = blocked
        self.assertTrue(sr.reentry_drop_blocked(99.0, 100.0, 2.0, 0.0))
        self.assertFalse(sr.reentry_drop_blocked(97.0, 100.0, 2.0, 0.0))  # Below the threshold -> it enters.
        self.assertFalse(sr.reentry_drop_blocked(99.0, 100.0, 0.0, 0.0))  # drop 0 -> no barrier.
        self.assertFalse(sr.reentry_drop_blocked(99.0, 0.0, 2.0, 0.0))    # A missing last_sell.

    def test_dca_price_hit(self):
        # last_buy 100, drop 2% -> a threshold of 98
        self.assertTrue(sr.dca_price_hit(98.0, 100.0, 2.0, 0.0))    # Exactly at the threshold.
        self.assertTrue(sr.dca_price_hit(97.0, 100.0, 2.0, 0.0))    # Below the threshold.
        self.assertFalse(sr.dca_price_hit(99.0, 100.0, 2.0, 0.0))   # above the threshold, no tolerance
        self.assertTrue(sr.dca_price_hit(98.04, 100.0, 2.0, 0.05))  # in toleranta 0.05%

    def test_progressive_dca_spacing_is_safe_and_zero_preserves_live(self):
        self.assertEqual(sr.progressive_dca_drop_pct(1.25, 0.0, 9), 1.25)
        self.assertEqual(sr.progressive_dca_drop_pct(1.25, 0.25, 0), 1.25)
        self.assertEqual(sr.progressive_dca_drop_pct(1.25, 0.25, 4), 2.25)
        self.assertEqual(sr.progressive_dca_drop_pct(1.25, -1.0, 4), 1.25)

    def test_are_close_is_identical_to_botcore(self):
        import botcore
        for a, b, tol in [(50.0, 50.75, 0.05), (65.93, 65.91, 0.05), (100.0, 98.0, 0.0),
                          (99.0, 100.0, 2.0), (0.0, 0.0, 0.05), (58.42, 58.47, 0.05)]:
            self.assertEqual(sr.are_close(a, b, tol), botcore.are_close(a, b, tol),
                             f"divergenta la are_close({a},{b},{tol})")

    def test_reentry_hybrid_blocked_behavior(self):
        # Case 1: No last sell -> always unblocked
        blocked, reason = sr.reentry_hybrid_blocked(100.0, None, None, True, 2.0, 1.5)
        self.assertFalse(blocked)
        self.assertEqual(reason, "no_last_sell")

        # Case 2: TTL expired -> unblocked
        blocked, reason = sr.reentry_hybrid_blocked(
            price=120.0, last_sell=100.0, post_sell_peak=120.0,
            is_bull=False, drop_pct=2.0, pullback_pct=1.5,
            elapsed_sec=86400 * 15, ttl_sec=86400 * 14,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "ttl_expired")

        # Case 3: State B (TREND_CONFIRMED, is_bull=True)
        # Sold at 100, price reached peak 130.
        # With 1.5% pullback, threshold is 130 * (1 - 0.015) = 128.05.
        # At 129.0: price > 128.05 -> blocked waiting for pullback from peak.
        blocked, reason = sr.reentry_hybrid_blocked(
            price=129.0, last_sell=100.0, post_sell_peak=130.0,
            is_bull=True, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "bull_pullback_pending")

        # At 127.5: price <= 128.05 -> pullback met! Enters even though 127.5 > 100.0.
        blocked, reason = sr.reentry_hybrid_blocked(
            price=127.5, last_sell=100.0, post_sell_peak=130.0,
            is_bull=True, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "bull_pullback_met")

        # Pullback 0% in bull -> immediate
        blocked, reason = sr.reentry_hybrid_blocked(
            price=135.0, last_sell=100.0, post_sell_peak=135.0,
            is_bull=True, drop_pct=2.0, pullback_pct=0.0,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "bull_immediate")

        # Case 4: State C (NO_CONFIRMED_TREND, is_bull=False)
        # Sold at 100, drop required = 2.0% -> threshold = 98.0.
        # At 105.0: blocked.
        blocked, reason = sr.reentry_hybrid_blocked(
            price=105.0, last_sell=100.0, post_sell_peak=105.0,
            is_bull=False, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "range_drop_pending")

        # At 97.0: range drop met.
        blocked, reason = sr.reentry_hybrid_blocked(
            price=97.0, last_sell=100.0, post_sell_peak=100.0,
            is_bull=False, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "range_drop_met")


if __name__ == "__main__":
    unittest.main(verbosity=2)
