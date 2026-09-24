"""
Tests for the ADAPTIVE re-entry threshold in the spot DCA engine (23 Jul).

Context: investigated in offline/research/kraken_adaptive_thresholds/ — the adaptive
threshold (K_REENTRY * vol_1h) beat the fixed threshold on real data (HYPEUSD, ~30 days:
TOTAL +3.26% versus +2.20%). Promoted to a real decision through StratParams.reentry_adaptive
(False by default — enabled explicitly through STRAT_REENTRY_ADAPTIVE=true), with
fail-safe onto the fixed threshold when volatility cannot be computed (warm-up).

Coverage:
  - _effective_reentry_drop_pct(): fixed when reentry_adaptive=False (always,
    regardless of price history); falls back to fixed when adaptive=True but
    warm-up (<20 points); adaptive once there is enough history.
  - The re-entry block in step(): uses the EFFECTIVE threshold (not always the fixed one).
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

KRAKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kraken")
ROOT = os.path.dirname(KRAKEN_DIR)
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_dca as strat  # noqa: E402
from providers.strategy_executor import PairPrecision  # noqa: E402


# Keep the synthetic state file out of kraken/, next to the live pair states.
_STATE_DIR = tempfile.TemporaryDirectory(prefix="test_reentry_state_")


def tearDownModule():
    _STATE_DIR.cleanup()


def _make_strategy(tmp_pair="TESTPAIR_REENTRY", **param_overrides):
    """Build a strategy with explicit synthetic venue precision and minimal parameters."""
    client = MagicMock()
    client.pair_precision.return_value = PairPrecision(2, 6, 0.01, "TEST")
    defaults = dict(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2, dca_amount=50.0,
        dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=1.9, max_budget=1000.0,
        max_dca_buys=10, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=0.0,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2, reentry_tolerance_pct=0.05,
        reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )
    defaults.update(param_overrides)
    params = strat.StratParams(**defaults)
    return strat.Strategy(
        client, tmp_pair, params, dry_run=True,
        initial_state=strat._new_state(), state_dir=_STATE_DIR.name,
    )


class TestEffectiveReentryDropPct(unittest.TestCase):

    def test_effective_reentry_drop_pct_behaviors(self):
        with self.subTest(msg="fixed_when_adaptive_disabled"):
            s = _make_strategy(reentry_adaptive=False, reentry_drop_pct=2.2)
            pct, source = s._effective_reentry_drop_pct()
            self.assertEqual(pct, 2.2)
            self.assertEqual(source, "fix")

        with self.subTest(msg="fixed_stays_fixed_even_with_price_history"):
            s = _make_strategy(reentry_adaptive=False, reentry_drop_pct=2.2)
            for i in range(30):
                s._shadow_prices.append((i * 120.0, 100.0 + (i % 3)))
            pct, source = s._effective_reentry_drop_pct()
            self.assertEqual(pct, 2.2)
            self.assertEqual(source, "fix")

        with self.subTest(msg="adaptive_falls_back_to_fixed_during_warmup"):
            s = _make_strategy(reentry_adaptive=True, reentry_drop_pct=2.2)
            for i in range(10):
                s._shadow_prices.append((i * 120.0, 100.0 + i * 0.1))
            pct, source = s._effective_reentry_drop_pct()
            self.assertEqual(pct, 2.2)
            self.assertIn("fallback", source)
            self.assertIn("warm-up", source)

        with self.subTest(msg="adaptive_uses_volatility_when_enough_history"):
            s = _make_strategy(reentry_adaptive=True, reentry_drop_pct=2.2)
            import random
            random.seed(42)
            price = 100.0
            for i in range(30):
                price *= (1 + random.uniform(-0.01, 0.01))
                s._shadow_prices.append((i * 120.0, price))
            pct, source = s._effective_reentry_drop_pct()
            self.assertIn("adaptiv", source)
            self.assertNotEqual(pct, 2.2, "the adaptive threshold must not coincide with the fixed one by accident")
            self.assertGreater(pct, 0)

        with self.subTest(msg="adaptive_respects_shadow_k_reentry_env_override"):
            s = _make_strategy(reentry_adaptive=True, reentry_drop_pct=2.2)
            import random
            random.seed(7)
            price = 100.0
            for i in range(30):
                price *= (1 + random.uniform(-0.01, 0.01))
                s._shadow_prices.append((i * 120.0, price))
            pct_k2, _ = s._effective_reentry_drop_pct()
            os.environ["SHADOW_K_REENTRY"] = "4.0"
            try:
                pct_k4, _ = s._effective_reentry_drop_pct()
            finally:
                del os.environ["SHADOW_K_REENTRY"]
            self.assertAlmostEqual(pct_k4, pct_k2 * 2.0, places=6,
                                   msg="K=4.0 must give exactly double K=2.0 (the default), same vol_1h")


class TestReentryGateUsesEffectivePct(unittest.TestCase):
    """step() uses the EFFECTIVE threshold (fixed or adaptive), not always the fixed one."""

    def test_reentry_gate_effective_threshold(self):
        with self.subTest(msg="step_blocks_reentry_using_fixed_when_adaptive_disabled"):
            s = _make_strategy(reentry_adaptive=False, reentry_drop_pct=2.2, reentry_tolerance_pct=0.0)
            s.s["last_sell_price"] = 100.0
            s.s["qty"] = 0.0
            # price 98.5 > fixed threshold (100*0.978=97.8) -> should be blocked
            s.step(98.5)
            self.assertFalse(s._has_open("buy"), "the re-entry should have been blocked (price above the fixed threshold)")

        with self.subTest(msg="step_allows_reentry_when_price_below_fixed_threshold"):
            s = _make_strategy(reentry_adaptive=False, reentry_drop_pct=2.2, reentry_tolerance_pct=0.0)
            s.s["last_sell_price"] = 100.0
            s.s["qty"] = 0.0
            # price 97.0 < fixed threshold (97.8) -> the re-entry must be allowed
            s.step(97.0)
            self.assertTrue(s._has_open("buy"), "the re-entry should have been allowed (the price is below the fixed threshold)")


class TestStopAwareReentry(unittest.TestCase):
    """4 Aug: after a STOP-LOSS, re-entry is on RECOVERY (a bounce off the low), not on a
    further drop — otherwise the bot stays locked out when the price recovers."""

    def test_stop_and_tp_aware_reentry_behaviors(self):
        with self.subTest(msg="stop_reentry_not_stranded_on_recovery"):
            # The fixed BUG: sold at 51.19 on stop-loss, the price recovers to 55.6 (above the sale).
            # The old rule (re-enter only below 50.06) would block forever. The new one re-enters.
            s = _make_strategy(reentry_sl_bounce_pct=1.5, reentry_drop_pct=2.2, reentry_tolerance_pct=0.0)
            s.s["qty"] = 0.0
            s.s["last_sell_price"] = 51.19
            s.s["last_exit_kind"] = "STOP"
            s.s["sl_low"] = 51.19
            s.step(55.6)
            self.assertTrue(s._has_open("buy"), "after a STOP, the recovery must trigger the re-entry")

        with self.subTest(msg="stop_reentry_blocked_until_bounce_then_enters"):
            s = _make_strategy(reentry_sl_bounce_pct=1.5, reentry_tolerance_pct=0.0)
            s.s["qty"] = 0.0
            s.s["last_sell_price"] = 51.19
            s.s["last_exit_kind"] = "STOP"
            s.s["sl_low"] = 51.19
            s.step(50.0)                                  # Still falling -> it tracks the low, it does not enter.
            self.assertFalse(s._has_open("buy"))
            self.assertEqual(s.s["sl_low"], 50.0)
            s.step(50.8)                                  # +1.6% de la minim 50 -> bounce atins
            self.assertTrue(s._has_open("buy"), "bounce >= prag -> reintra")

        with self.subTest(msg="tp_exit_keeps_old_drop_below_sell_rule"):
            # after a TP (not a STOP) the old rule stands: do not rebuy higher than you sold
            s = _make_strategy(reentry_sl_bounce_pct=1.5, reentry_drop_pct=2.2, reentry_tolerance_pct=0.0)
            s.s["qty"] = 0.0
            s.s["last_sell_price"] = 100.0
            s.s["last_exit_kind"] = "TP"
            s.step(98.5)                                  # 98.5 > the threshold of 97.8 -> blocked (the old rule).
            self.assertFalse(s._has_open("buy"))
            s.step(97.0)                                  # 97.0 < 97.8 -> reintra
            self.assertTrue(s._has_open("buy"))


class TestTrailingTakeProfit(unittest.TestCase):
    """Trailing stays armed after the TP is first exceeded."""

    @staticmethod
    def _positioned_strategy(**overrides):
        params = dict(
            takeprofit_pct=5.0,
            tp_trend_hold=True,
            tp_trail_pct=3.0,
            dca_drop_pct=2.0,
        )
        params.update(overrides)
        s = _make_strategy(**params)
        s.s["qty"] = 1.0
        s.s["cost"] = 100.0
        s.s["spent"] = 100.0
        s.s["entry_price"] = 100.0
        s.s["last_buy_price"] = 100.0
        return s

    def test_trailing_take_profit_arming_and_exiting(self):
        with self.subTest(msg="regime_gate_uses_classic_tp_until_bullish"):
            s = self._positioned_strategy(tp_regime_gate=True)
            s.client.ohlc_closes.return_value = [100.0 - index for index in range(40)]
            s.step(105.5)
            sell = s._find_open("sell")
            self.assertIsNotNone(sell)
            self.assertFalse(sell["market"])
            self.assertIsNone(s.s["trail_peak"])

        with self.subTest(msg="regime_gate_arms_trailing_on_common_bull_signal"):
            s = self._positioned_strategy(tp_regime_gate=True)
            s.client.ohlc_closes.return_value = [100.0 + index for index in range(40)]
            s.step(105.5)
            self.assertEqual(s.s["trail_peak"], 105.5)
            self.assertFalse(s._has_open("sell"))

        with self.subTest(msg="pullback_below_tp_after_arming_still_exits"):
            s = self._positioned_strategy()
            s.step(105.5)       # exceeds TP=105 and arms the trailing
            self.assertEqual(s.s["trail_peak"], 105.5)
            self.assertFalse(s._has_open("sell"))
            s.step(102.0)       # A pullback of 3.32%; it is below the TP, but the trailing is already armed.
            sell = s._find_open("sell")
            self.assertIsNotNone(
                sell,
                "the armed trailing must still exit after falling back below the TP",
            )
            self.assertEqual(sell["kind"], "TP")

        with self.subTest(msg="armed_trailing_survives_a_later_bear_regime"):
            s = self._positioned_strategy(tp_regime_gate=True)
            s.client.ohlc_closes.return_value = [
                100.0 + index for index in range(40)
            ]
            s.step(105.5)
            s.client.ohlc_closes.return_value = [
                140.0 - index for index in range(40)
            ]
            s.step(102.0)
            sell = s._find_open("sell")
            self.assertIsNotNone(sell)
            self.assertTrue(sell["market"])
            self.assertEqual(sell["kind"], "TP")

        with self.subTest(msg="trailing_exit_does_not_open_dca_in_same_tick"):
            s = self._positioned_strategy(dca_drop_pct=2.0)
            s.step(105.5)
            # Makes the DCA threshold eligible at the same time as the trailing pullback. An exit
            # and a buy in the same tick would contradict each other and raise exposure by accident.
            s.s["last_buy_price"] = 110.0
            s.step(102.0)
            self.assertTrue(s._has_open("sell"))
            self.assertFalse(s._has_open("buy"))

    def test_trailing_take_profit_profit_floor_behaviors(self):
        with self.subTest(msg="adaptive_trailing_floor_never_moves_down_when_volatility_widens"):
            s = self._positioned_strategy(tp_trail_adaptive=True)
            with patch.object(s, "_effective_trail_pct", side_effect=[1.5, 3.0]):
                s.step(106.0)      # floor initial: 104.41
                protected = s.s["trail_stop"]
                self.assertAlmostEqual(protected, 104.41)
                # The volatility rises and the adaptive trailing would widen to 3%.
                # The floor already earned must not be lowered to 102.82.
                s.step(104.0)

            self.assertEqual(s.s["trail_stop"], protected)
            sell = s._find_open("sell")
            self.assertIsNotNone(sell, "the ratcheted floor must trigger the exit")
            self.assertTrue(sell["market"])
            self.assertEqual(sell["kind"], "TP")

        with self.subTest(msg="profit_floor_blocks_gap_exit_below_break_even"):
            s = self._positioned_strategy(
                stop_loss_pct=12.5,
                tp_trail_profit_floor_pct=1.0,
            )
            s.step(105.5)       # arms the trailing
            s.step(95.0)        # gap below break-even, but still above the hard stop
            self.assertIsNone(s._find_open("sell"))
            self.assertFalse(s._has_open("buy"))

        with self.subTest(msg="profit_floor_exits_on_recovery_still_below_trail_stop"):
            s = self._positioned_strategy(
                stop_loss_pct=12.5,
                tp_trail_profit_floor_pct=1.0,
            )
            s.step(105.5)
            s.step(95.0)
            s.step(101.2)       # MARKET reference 101.10 >= floor; below the trail stop 102.335
            sell = s._find_open("sell")
            self.assertIsNotNone(sell)
            self.assertTrue(sell["market"])
            self.assertEqual(sell["kind"], "TP")
            self.assertGreaterEqual(sell["price"], 101.0)

        with self.subTest(msg="hard_stop_exits_market_after_profit_floor_block"):
            s = self._positioned_strategy(
                stop_loss_pct=12.5,
                tp_trail_profit_floor_pct=1.0,
            )
            s.step(105.5)
            s.step(95.0)
            self.assertIsNone(s._find_open("sell"))

            s.step(87.0)        # below avg*(1-12.5%): MARKET STOP regardless of profit
            sell = s._find_open("sell")
            self.assertIsNotNone(sell)
            self.assertTrue(sell["market"])
            self.assertEqual(sell["kind"], "STOP")

        with self.subTest(msg="zero_profit_floor_preserves_market_trailing"):
            s = self._positioned_strategy(tp_trail_profit_floor_pct=0.0)
            s.step(105.5)
            s.step(95.0)
            sell = s._find_open("sell")
            self.assertIsNotNone(sell)
            self.assertTrue(sell["market"])


class TestProgressiveDcaSpacing(unittest.TestCase):
    def test_completed_buys_widen_the_next_dca_threshold(self):
        s = _make_strategy(
            dca_drop_pct=2.0,
            dca_spacing_growth_pct=0.5,
            reentry_tolerance_pct=0.0,
        )
        s.s.update({
            "qty": 1.0,
            "cost": 100.0,
            "spent": 100.0,
            "entry_price": 100.0,
            "last_buy_price": 100.0,
            "dca_buys": 2,
        })

        s.step(97.5)  # progressive threshold = 2% + 2x0.5% = 3%; it still does not buy
        self.assertFalse(s._has_open("buy"))
        s.client.free_balance.return_value = 1000.0
        s.step(97.0)
        self.assertTrue(s._has_open("buy"))


class TestPaperMarketReconciliation(unittest.TestCase):
    def test_paper_market_reconciliation_behaviors(self):
        with self.subTest(msg="limit_buy_waits_until_observed_price_reaches_limit"):
            s = _make_strategy(entry_discount_pct=1.0)
            with patch.object(strat, "notify"):
                s.step(100.0)
                order = s._find_open("buy")
                self.assertEqual(order["price"], 99.0)

                s.reconcile(101.0)
                self.assertTrue(s._has_open("buy"))
                self.assertEqual(s.s["qty"], 0.0)

                s.reconcile(98.5)

            self.assertFalse(s._has_open("buy"))
            self.assertGreater(s.s["qty"], 0.0)
            self.assertEqual(s.s["last_buy_price"], 99.0)

        with self.subTest(msg="stop_market_sell_fills_at_observed_price_during_drop"):
            s = _make_strategy(stop_loss_pct=10.0)
            s.s.update({
                "qty": 1.0, "cost": 100.0, "spent": 100.0,
                "entry_price": 100.0, "last_buy_price": 100.0,
            })

            with patch.object(strat, "notify"):
                s.step(80.0)
                order = s._find_open("sell")
                self.assertIsNotNone(order)
                self.assertTrue(order["market"])
                self.assertGreater(order["price"], 70.0)

                s.reconcile(70.0)

            self.assertFalse(s._has_open("sell"))
            self.assertEqual(s.s["qty"], 0.0)
            self.assertEqual(s.s["last_sell_price"], 70.0)
            self.assertLess(s.s["realized_net"], 0.0)


class TestFillAccounting(unittest.TestCase):
    """Cost basis and fees stay correct when the exit happens in tranches."""

    def test_two_partial_sells_reduce_cost_and_charge_each_fee_once(self):
        s = _make_strategy()
        buy = {"side": "buy", "kind": "ENTRY", "amount": 200.0}
        with patch.object(strat, "notify"):
            s._apply_fill(buy, vol=2.0, price=100.0, fee=0.52)

            self.assertAlmostEqual(s.s["qty"], 2.0)
            self.assertAlmostEqual(s.s["cost"], 200.0)
            self.assertAlmostEqual(s.s["realized_net"], -0.52)

            sell = {"side": "sell", "kind": "TP"}
            s._apply_fill(sell, vol=1.0, price=110.0, fee=0.286)
            self.assertAlmostEqual(s.s["qty"], 1.0)
            self.assertAlmostEqual(s.s["cost"], 100.0)
            self.assertAlmostEqual(s._avg(), 100.0)

            s._apply_fill(sell, vol=1.0, price=120.0, fee=0.312)
            self.assertAlmostEqual(s.s["qty"], 0.0)
            self.assertAlmostEqual(s.s["cost"], 0.0)
            self.assertAlmostEqual(s.s["realized_gross"], 30.0)
            self.assertAlmostEqual(s.s["fees_total"], 1.118)
            self.assertAlmostEqual(s.s["realized_net"], 28.882)


class TestHybridReentry(unittest.TestCase):
    """End-to-end tests for hybrid re-entry state transitions in spot_dca engine."""

    def test_hybrid_reentry_bull_waits_for_pullback_then_enters(self):
        s = _make_strategy(
            reentry_hybrid_enabled=True,
            reentry_pullback_pct=1.5,
            reentry_drop_pct=2.0,
        )
        s.s["last_sell_price"] = 100.0
        s.s["post_sell_peak"] = 100.0
        s.s["last_sell_ts"] = 1000.0

        # Mock regime decision to return bull
        mock_regime = MagicMock()
        mock_regime.regime = "bull"
        mock_regime.closes = [100.0] * 30
        mock_regime.is_valid = True

        with patch.object(s, "_regime_context", return_value=(mock_regime, [100.0] * 30)), \
             patch.object(s, "_regime_matches", return_value=True):
            # Step at 130.0 -> peak becomes 130.0, pullback 1.5% is 128.05.
            # 130.0 > 128.05 -> blocked waiting for pullback
            s.step(130.0, timestamp=1010.0)
            self.assertEqual(s.s["post_sell_peak"], 130.0)
            self.assertFalse(s._has_open("buy"))

            # Step at 129.0 -> still > 128.05 -> blocked
            s.step(129.0, timestamp=1020.0)
            self.assertFalse(s._has_open("buy"))

            # Step at 127.5 <= 128.05 -> pullback met!
            s.step(127.5, timestamp=1030.0)
            self.assertTrue(s._has_open("buy"))
            order = s._find_open("buy")
            self.assertEqual(order["kind"], "ENTRY")

    def test_hybrid_reentry_range_requires_sale_drop(self):
        s = _make_strategy(
            reentry_hybrid_enabled=True,
            reentry_pullback_pct=1.5,
            reentry_drop_pct=2.0,
        )
        s.s["last_sell_price"] = 100.0
        s.s["post_sell_peak"] = 100.0
        s.s["last_sell_ts"] = 1000.0

        # Regime not bull
        mock_regime = MagicMock()
        mock_regime.regime = "range"
        mock_regime.closes = [100.0] * 30
        mock_regime.is_valid = True

        with patch.object(s, "_regime_context", return_value=(mock_regime, [100.0] * 30)), \
             patch.object(s, "_regime_matches", return_value=False):
            # Step at 105.0 -> blocked
            s.step(105.0, timestamp=1010.0)
            self.assertFalse(s._has_open("buy"))

            # Step at 99.0 -> blocked (99 > 98)
            s.step(99.0, timestamp=1020.0)
            self.assertFalse(s._has_open("buy"))

            # Step at 97.5 <= 98.0 -> drop met!
            s.step(97.5, timestamp=1030.0)
            self.assertTrue(s._has_open("buy"))
            order = s._find_open("buy")
            self.assertEqual(order["kind"], "ENTRY")

    def test_hybrid_reentry_ttl_expiry(self):
        s = _make_strategy(
            reentry_hybrid_enabled=True,
            reentry_pullback_pct=1.5,
            reentry_drop_pct=2.0,
            reentry_ttl_hours=24.0,
        )
        s.s["last_sell_price"] = 100.0
        s.s["post_sell_peak"] = 100.0
        s.s["last_sell_ts"] = 1000.0

        mock_regime = MagicMock()
        mock_regime.regime = "range"
        mock_regime.closes = [100.0] * 30
        mock_regime.is_valid = True

        with patch.object(s, "_regime_context", return_value=(mock_regime, [100.0] * 30)), \
             patch.object(s, "_regime_matches", return_value=False):
            # After 1 hour, price 105 -> blocked
            s.step(105.0, timestamp=1000.0 + 3600.0)
            self.assertFalse(s._has_open("buy"))

            # After 25 hours (> 24h), price 105 -> TTL expired, unblocked!
            s.step(105.0, timestamp=1000.0 + 25 * 3600.0)
            self.assertTrue(s._has_open("buy"))
            order = s._find_open("buy")
            self.assertEqual(order["kind"], "ENTRY")


if __name__ == "__main__":
    unittest.main()

