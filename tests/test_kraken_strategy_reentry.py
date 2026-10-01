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
        with self.subTest(msg="trailing_arms_directly_at_tp"):
            s = self._positioned_strategy()
            s.client.ohlc_closes.return_value = [100.0 - index for index in range(40)]
            s.step(105.5)
            self.assertEqual(s.s["trail_peak"], 105.5)
            self.assertFalse(s._has_open("sell"))

        with self.subTest(msg="trailing_arms_on_rising_market"):
            s = self._positioned_strategy()
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


from strategies import spot_dca_rules as sr


class StratRulesTest(unittest.TestCase):
    def test_entry_and_tp(self):
        self.assertAlmostEqual(sr.entry_price(100.0, 0.2), 99.8)
        self.assertAlmostEqual(sr.tp_price(100.0, 3.0), 103.0)

    def test_hit_stop(self):
        self.assertTrue(sr.hit_stop(100.0, 87.0, 12.5))
        self.assertFalse(sr.hit_stop(100.0, 90.0, 12.5))
        self.assertFalse(sr.hit_stop(100.0, 50.0, 0.0))
        self.assertFalse(sr.hit_stop(0.0, 50.0, 12.5))

    def test_reentry_stop_bounce(self):
        self.assertTrue(sr.reentry_stop_blocked(50.0, 50.0, 1.5, 0.0))
        self.assertFalse(sr.reentry_stop_blocked(50.8, 50.0, 1.5, 0.0))

    def test_reentry_drop(self):
        self.assertTrue(sr.reentry_drop_blocked(99.0, 100.0, 2.0, 0.0))
        self.assertFalse(sr.reentry_drop_blocked(97.0, 100.0, 2.0, 0.0))
        self.assertFalse(sr.reentry_drop_blocked(99.0, 100.0, 0.0, 0.0))
        self.assertFalse(sr.reentry_drop_blocked(99.0, 0.0, 2.0, 0.0))

    def test_dca_price_hit(self):
        self.assertTrue(sr.dca_price_hit(98.0, 100.0, 2.0, 0.0))
        self.assertTrue(sr.dca_price_hit(97.0, 100.0, 2.0, 0.0))
        self.assertFalse(sr.dca_price_hit(99.0, 100.0, 2.0, 0.0))
        self.assertTrue(sr.dca_price_hit(98.04, 100.0, 2.0, 0.05))

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
                             f"divergence at are_close({a},{b},{tol})")

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
        blocked, reason = sr.reentry_hybrid_blocked(
            price=129.0, last_sell=100.0, post_sell_peak=130.0,
            is_bull=True, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "bull_pullback_pending")

        blocked, reason = sr.reentry_hybrid_blocked(
            price=127.5, last_sell=100.0, post_sell_peak=130.0,
            is_bull=True, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "bull_pullback_met")

        blocked, reason = sr.reentry_hybrid_blocked(
            price=135.0, last_sell=100.0, post_sell_peak=135.0,
            is_bull=True, drop_pct=2.0, pullback_pct=0.0,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "bull_immediate")

        # Case 4: State C (NO_CONFIRMED_TREND, is_bull=False)
        blocked, reason = sr.reentry_hybrid_blocked(
            price=105.0, last_sell=100.0, post_sell_peak=105.0,
            is_bull=False, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "range_drop_pending")

        blocked, reason = sr.reentry_hybrid_blocked(
            price=97.0, last_sell=100.0, post_sell_peak=100.0,
            is_bull=False, drop_pct=2.0, pullback_pct=1.5,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "range_drop_met")

    def test_effective_reentry_pullback_pct(self):
        st = _make_strategy("TESTPAIR_PULLBACK")
        st.p.reentry_pullback_pct = 1.5
        st.p.reentry_pullback_adaptive = False
        val, src = st._effective_reentry_pullback_pct()
        self.assertEqual(val, 1.5)
        self.assertEqual(src, "fixed")

        st.p.reentry_pullback_adaptive = True
        st.p.reentry_pullback_k = 1.0
        st.p.reentry_pullback_min = 0.8
        st.p.reentry_pullback_max = 3.5

        # Fallback when warm-up or no vol data
        st._shadow_vol_1h = lambda: None
        val, src = st._effective_reentry_pullback_pct()
        self.assertEqual(val, 1.5)
        self.assertIn("fallback", src)

        # Normal adaptive scaling: vol=2.2 -> 2.2%
        st._shadow_vol_1h = lambda: 2.2
        val, src = st._effective_reentry_pullback_pct()
        self.assertAlmostEqual(val, 2.2)
        self.assertIn("adaptive", src)

        # Clamping to minimum: vol=0.3 -> 0.8%
        st._shadow_vol_1h = lambda: 0.3
        val, src = st._effective_reentry_pullback_pct()
        self.assertAlmostEqual(val, 0.8)

        # Clamping to maximum: vol=5.0 -> 3.5%
        st._shadow_vol_1h = lambda: 5.0
        val, src = st._effective_reentry_pullback_pct()
        self.assertAlmostEqual(val, 3.5)


class TestMultiHorizonDynamicProfitRules(unittest.TestCase):
    def test_dynamic_flat_tp_pct(self):
        # Fallbacks on missing/invalid strength
        self.assertEqual(sr.dynamic_flat_tp_pct(None, base_tp_pct=5.0), 5.0)
        self.assertEqual(sr.dynamic_flat_tp_pct(-1.0, base_tp_pct=5.0), 5.0)
        self.assertEqual(sr.dynamic_flat_tp_pct(float("nan"), base_tp_pct=5.0), 5.0)

        # Deep flat (strength = 0.0) -> tp_min_pct (3.0%)
        self.assertAlmostEqual(sr.dynamic_flat_tp_pct(0.0, strength_threshold=2.0, tp_min_pct=3.0, tp_max_pct=7.0), 3.0)

        # Midpoint flat (strength = 1.0, threshold = 2.0 -> c_flat = 0.5) -> 5.0%
        self.assertAlmostEqual(sr.dynamic_flat_tp_pct(1.0, strength_threshold=2.0, tp_min_pct=3.0, tp_max_pct=7.0), 5.0)

        # Breakout boundary (strength = 2.0 -> c_flat = 1.0) -> 7.0%
        self.assertAlmostEqual(sr.dynamic_flat_tp_pct(2.0, strength_threshold=2.0, tp_min_pct=3.0, tp_max_pct=7.0), 7.0)

        # Past boundary (strength = 3.5 -> clamped c_flat = 1.0) -> 7.0%
        self.assertAlmostEqual(sr.dynamic_flat_tp_pct(3.5, strength_threshold=2.0, tp_min_pct=3.0, tp_max_pct=7.0), 7.0)

    def test_dynamic_trend_trail_pct(self):
        # Below threshold (gain 5% <= 6%) -> base_trail (8.0%)
        self.assertAlmostEqual(sr.dynamic_trend_trail_pct(5.0, base_trail_pct=8.0, min_trail_pct=3.0, ratchet_k=0.5, gain_threshold_pct=6.0), 8.0)

        # At threshold (gain 6.0%) -> 8.0%
        self.assertAlmostEqual(sr.dynamic_trend_trail_pct(6.0, base_trail_pct=8.0, min_trail_pct=3.0, ratchet_k=0.5, gain_threshold_pct=6.0), 8.0)

        # Above threshold: gain 8.0% -> 8.0 - 0.5 * 2.0 = 7.0%
        self.assertAlmostEqual(sr.dynamic_trend_trail_pct(8.0, base_trail_pct=8.0, min_trail_pct=3.0, ratchet_k=0.5, gain_threshold_pct=6.0), 7.0)

        # TAO scenario: gain 11.5% -> 8.0 - 0.5 * 5.5 = 5.25%
        self.assertAlmostEqual(sr.dynamic_trend_trail_pct(11.5, base_trail_pct=8.0, min_trail_pct=3.0, ratchet_k=0.5, gain_threshold_pct=6.0), 5.25)

        # Deep profit: gain 20.0% -> 8.0 - 0.5 * 14.0 = 1.0% -> clamped to min_trail (3.0%)
        self.assertAlmostEqual(sr.dynamic_trend_trail_pct(20.0, base_trail_pct=8.0, min_trail_pct=3.0, ratchet_k=0.5, gain_threshold_pct=6.0), 3.0)

    def test_fast_profit_reversal(self):
        avg = 100.0
        shadow = [(1000.0, 110.0), (1100.0, 112.0), (1200.0, 111.0)]

        # Gain < 2 * 5.0% = 10.0% -> False
        trig, gain, drop = sr.check_fast_profit_reversal(
            current_price=108.0, avg_cost=avg, base_tp_pct=5.0, mult=2.0,
            shadow_prices=shadow, window_sec=300.0, drop_pct=1.0, current_time=1200.0
        )
        self.assertFalse(trig)

        # Gain >= 10.0% (price 111.0 vs peak 112.0 in window)
        # Drop = (112 - 111) / 112 * 100 = 0.89% < 1.0% -> False
        trig, gain, drop = sr.check_fast_profit_reversal(
            current_price=111.0, avg_cost=avg, base_tp_pct=5.0, mult=2.0,
            shadow_prices=shadow, window_sec=300.0, drop_pct=1.0, current_time=1200.0
        )
        self.assertFalse(trig)
        self.assertAlmostEqual(gain, 11.0)
        self.assertAlmostEqual(drop, (112.0 - 111.0) / 112.0 * 100.0)

        # Drop >= 1.0% (price 110.5 vs peak 112.0 -> drop = 1.34%) -> True!
        trig, gain, drop = sr.check_fast_profit_reversal(
            current_price=110.5, avg_cost=avg, base_tp_pct=5.0, mult=2.0,
            shadow_prices=shadow, window_sec=300.0, drop_pct=1.0, current_time=1200.0
        )
        self.assertTrue(trig)
        self.assertAlmostEqual(gain, 10.5)

    def test_parabolic_surge_exhaustion(self):
        avg = 100.0
        # Not qualified for surge (gain 15% < 20%, window_move 10% < 25%)
        trig, reason = sr.check_parabolic_surge_exhaustion(
            current_price=115.0, avg_cost=avg, surge_peak=120.0,
            surge_gain_pct=20.0, exit_pullback_pct=2.0, window_move_pct=10.0, surge_move_pct=25.0
        )
        self.assertFalse(trig)

        # Qualified by position gain (current price 125, avg 100 -> gain 25% >= 20%)
        # Surge peak 126. Pullback = (126 - 125) / 126 * 100 = 0.79% < 2.0% -> False
        trig, reason = sr.check_parabolic_surge_exhaustion(
            current_price=125.0, avg_cost=avg, surge_peak=126.0,
            surge_gain_pct=20.0, exit_pullback_pct=2.0
        )
        self.assertFalse(trig)

        # Pullback >= 2.0% (current price 123.0 vs surge peak 126.0 -> 2.38% drop) -> True!
        trig, reason = sr.check_parabolic_surge_exhaustion(
            current_price=123.0, avg_cost=avg, surge_peak=126.0,
            surge_gain_pct=20.0, exit_pullback_pct=2.0
        )
        self.assertTrue(trig)
        self.assertIn("surge_pullback", reason)

    def test_slow_grind_exhaustion(self):
        avg = 100.0
        entry_t = 100000.0
        now_t = entry_t + 8 * 86400.0  # 8 days held (>= 7 days)
        shadow = [(now_t - 300, 120.0), (now_t, 118.0)]  # 120 -> 118 in 5 min (1.67% drop)

        # Duration < 7 days -> False
        trig, reason = sr.check_slow_grind_exhaustion(
            current_price=118.0, avg_cost=avg, entry_ts=entry_t, current_ts=entry_t + 5 * 86400,
            min_days=7.0, min_gain_pct=15.0, recent_peak=120.0, shadow_prices=shadow,
            flash_window_sec=900.0, flash_drop_pct=1.5, structural_drop_pct=2.5
        )
        self.assertFalse(trig)

        # Flash sensor triggered (drop 1.67% >= 1.5% in 15 min) -> True!
        trig, reason = sr.check_slow_grind_exhaustion(
            current_price=118.0, avg_cost=avg, entry_ts=entry_t, current_ts=now_t,
            min_days=7.0, min_gain_pct=15.0, recent_peak=120.0, shadow_prices=shadow,
            flash_window_sec=900.0, flash_drop_pct=1.5, structural_drop_pct=2.5
        )
        self.assertTrue(trig)
        self.assertIn("flash_drop", reason)

        # Structural sensor: recent_peak 122.0 vs current 118.5 (drop = 2.87% >= 2.5%) -> True!
        shadow_slow = [(now_t - 1800, 119.0), (now_t, 118.5)]
        trig, reason = sr.check_slow_grind_exhaustion(
            current_price=118.5, avg_cost=avg, entry_ts=entry_t, current_ts=now_t,
            min_days=7.0, min_gain_pct=15.0, recent_peak=122.0, shadow_prices=shadow_slow,
            flash_window_sec=900.0, flash_drop_pct=1.5, structural_drop_pct=2.5
        )
        self.assertTrue(trig)
        self.assertIn("structural_drop", reason)

        # SMA sensor: price < SMA -> True!
        trig, reason = sr.check_slow_grind_exhaustion(
            current_price=119.5, avg_cost=avg, entry_ts=entry_t, current_ts=now_t,
            min_days=7.0, min_gain_pct=15.0, recent_peak=120.0, shadow_prices=[(now_t, 119.5)],
            flash_window_sec=900.0, flash_drop_pct=1.5, structural_drop_pct=2.5, sma_value=120.0
        )
        self.assertTrue(trig)
        self.assertIn("sma_break", reason)


class TestSpotDCAMultiHorizonIntegration(unittest.TestCase):
    def test_fast_profit_guard_in_trend_overlay(self):
        st = _make_strategy("TESTPAIR_MHDPA_FAST")
        st.p.trend_overlay = True
        st.p.fast_profit_guard = True
        st.p.fast_profit_mult = 2.0
        st.p.fast_profit_window_min = 5.0
        st.p.fast_profit_drop_pct = 1.0
        st.p.takeprofit_pct = 5.0
        st.s["trend_mode"] = True
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0

        now = 10000.0
        st._shadow_prices.append((now - 120.0, 112.0))
        st._shadow_prices.append((now, 110.5))

        exited = []
        st._request_market_exit = lambda px, kind: exited.append((px, kind)) or True

        from market_regime import MarketRegimeDecision
        dummy_regime = MarketRegimeDecision("bull", 0.05, 0.01, 5.0, True, "bullish")
        handled = st._overlay_step(110.5, dummy_regime, [100.0] * 30, tick_time=now)
        self.assertTrue(handled)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0][1], "TP")

    def test_surge_guard_in_trend_overlay(self):
        st = _make_strategy("TESTPAIR_MHDPA_SURGE")
        st.p.trend_overlay = True
        st.p.surge_guard = True
        st.p.surge_gain_pct = 20.0
        st.p.surge_exit_pullback_pct = 2.0
        st.s["trend_mode"] = True
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0
        st.s["surge_peak"] = 125.0

        exited = []
        st._request_market_exit = lambda px, kind: exited.append((px, kind)) or True

        from market_regime import MarketRegimeDecision
        dummy_regime = MarketRegimeDecision("bull", 0.05, 0.01, 5.0, True, "bullish")
        handled = st._overlay_step(122.0, dummy_regime, [100.0] * 30, tick_time=10000.0)
        self.assertTrue(handled)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0][1], "TP")

    def test_slow_grind_guard_in_trend_overlay(self):
        st = _make_strategy("TESTPAIR_MHDPA_SLOW")
        st.p.trend_overlay = True
        st.p.slow_grind_guard = True
        st.p.slow_grind_days = 7.0
        st.p.slow_grind_min_gain_pct = 15.0
        st.p.slow_grind_flash_window_min = 15.0
        st.p.slow_grind_flash_drop_pct = 1.5
        st.s["trend_mode"] = True
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0
        now = 1000000.0
        st.s["entry_ts"] = now - 8 * 86400.0  # 8 days held
        st.s["slow_grind_peak"] = 120.0

        st._shadow_prices.append((now - 120.0, 120.0))
        st._shadow_prices.append((now, 118.0))  # 120 -> 118 is 1.67% drop >= 1.5%

        exited = []
        st._request_market_exit = lambda px, kind: exited.append((px, kind)) or True

        from market_regime import MarketRegimeDecision
        dummy_regime = MarketRegimeDecision("bull", 0.05, 0.01, 5.0, True, "bullish")
        handled = st._overlay_step(118.0, dummy_regime, [100.0] * 30, tick_time=now)
        self.assertTrue(handled)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0][1], "TP")

    def test_dynamic_trend_trail_in_trend_overlay(self):
        st = _make_strategy("TESTPAIR_MHDPA_TRAIL")
        st.p.trend_overlay = True
        st.p.trend_trail_pct = 8.0
        st.p.trend_trail_dynamic = True
        st.p.trend_trail_base_pct = 8.0
        st.p.trend_trail_min_pct = 3.0
        st.p.trend_trail_ratchet_k = 0.5
        st.p.trend_trail_gain_threshold = 6.0
        st.s["trend_mode"] = True
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0
        # Peak at 120 (gain 20% > 6%). Ratchet = 8.0 - 0.5 * 14.0 = 1.0 -> clamped to 3.0%
        st.s["trend_peak"] = 120.0

        exited = []
        st._request_market_exit = lambda px, kind: exited.append((px, kind)) or True

        from market_regime import MarketRegimeDecision
        dummy_regime = MarketRegimeDecision("bull", 0.05, 0.01, 5.0, True, "bullish")

        # Price at 116.5: 120 * (1 - 0.03) = 116.4. At 116.5 > 116.4 -> should not exit yet
        handled = st._overlay_step(116.5, dummy_regime, [100.0] * 30, tick_time=10000.0)
        self.assertTrue(handled)
        self.assertEqual(len(exited), 0)

        # Price at 116.3: 116.3 <= 116.4 -> exits under ratcheted 3% trail (whereas 8% trail would be 110.4)
        handled = st._overlay_step(116.3, dummy_regime, [100.0] * 30, tick_time=10000.0)
        self.assertTrue(handled)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0][1], "TP")

    def test_dynamic_flat_tp_in_step(self):
        st = _make_strategy("TESTPAIR_MHDPA_FLAT")
        st.p.enable_takeprofit = True
        st.p.takeprofit_pct = 5.0
        st.p.tp_dynamic_flat = True
        st.p.tp_min_pct = 3.0
        st.p.tp_max_pct = 7.0
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0

        # Mock regime as sideways with strength = 0.0 (deep quiet flat)
        from market_regime import MarketRegimeDecision
        st._regime_context = lambda: (
            MarketRegimeDecision("sideways", 0.0, 0.01, 0.0, True, "sideways"),
            [100.0] * 30,
        )

        # In deep flat, eff_tp is 3.0%. A limit SELL TP order should be placed at 103.0 (instead of 105.0)
        st.step(100.5, timestamp=1000.0)
        sells = [o for o in st.s["orders"] if o["side"] == "sell"]
        self.assertEqual(len(sells), 1)
        self.assertAlmostEqual(sells[0]["price"], 103.0)

    def test_fast_profit_guard_in_step_trailing(self):
        st = _make_strategy("TESTPAIR_STEP_FAST_GUARD")
        st.p.enable_takeprofit = True
        st.p.tp_trend_hold = True
        st.p.takeprofit_pct = 5.0
        st.p.fast_profit_guard = True
        st.p.fast_profit_mult = 2.0
        st.p.fast_profit_window_min = 5.0
        st.p.fast_profit_drop_pct = 1.0
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0

        now = 10000.0
        st._shadow_prices.append((now - 120.0, 112.0))
        st._shadow_prices.append((now, 110.5))

        exited = []
        st._request_market_exit = lambda px, kind, soft_floor=False: exited.append((px, kind)) or True

        # Price at 110.5 (+10.5% >= 2x 5%), dropped from 112.0 (drop = 1.34% >= 1.0% in 5m)
        st.step(110.5, timestamp=now)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0][1], "TP")

    def test_surge_guard_in_step_trailing(self):
        st = _make_strategy("TESTPAIR_STEP_SURGE_GUARD")
        st.p.enable_takeprofit = True
        st.p.tp_trend_hold = True
        st.p.takeprofit_pct = 5.0
        st.p.surge_guard = True
        st.p.surge_gain_pct = 20.0
        st.p.surge_exit_pullback_pct = 2.0
        st.s["qty"] = 1.0
        st.s["cost"] = 100.0
        st.s["entry_price"] = 100.0
        st.s["surge_peak"] = 125.0

        exited = []
        st._request_market_exit = lambda px, kind, soft_floor=False: exited.append((px, kind)) or True

        # Price at 122.0 (+22% >= 20%), pullback from peak 125 is 2.4% >= 2.0%
        st.step(122.0, timestamp=10000.0)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0][1], "TP")


if __name__ == "__main__":
    unittest.main()


