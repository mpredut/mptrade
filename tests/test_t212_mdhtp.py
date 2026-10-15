"""Tests for Multi-Horizon Dynamic Take-Profit (MDHTP) integration in 212trading/strategy.py."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T212_DIR = os.path.join(ROOT, "212trading")
sys.path.insert(0, T212_DIR)
sys.path.insert(0, ROOT)

_COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
_PRELOADED = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
try:
    spec = importlib.util.spec_from_file_location(
        "t212_strategy_mdhtp_test", os.path.join(T212_DIR, "strategy.py"),
    )
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    spec.loader.exec_module(strategy)
finally:
    for name in _COLLIDING:
        sys.modules.pop(name, None)
    sys.modules.update(_PRELOADED)


def _params(**overrides):
    config = {
        "STRAT_CURRENCY": "USD",
        "YAHOO_SYMBOL": "TEST",
        "STRATEGY_MODE": "avg_tp",
        "STRAT_ENTRY_PCT": "20",
        "STRAT_DCA_PCT": "10",
        "STRAT_ENTRY_DISCOUNT_PCT": "0.0",
        "STRAT_DCA_DROP_PCT": "5",
        "STRAT_TAKEPROFIT_PCT": "3",
        "STRAT_MAX_DCA_BUYS": "3",
        "STRAT_MAX_BUDGET": "500",
        "STRAT_FX_FEE_PCT": "0.15",
        "STRAT_STOP_LOSS_PCT": "20",
        "STRAT_CHECK_MINUTES": "5",
        "STRAT_ORDER_TTL_MIN": "10",
        "STRAT_REENTRY_DROP_PCT": "2.0",
        "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
        "STRAT_LOSS_ALERT_STEP": "1",
        "STRAT_LADDER_MIN_FREE": "6",
        "STRAT_SL_REBUY_ENABLED": "false",
        "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
        "STRAT_DCA_TREND_GATE_PCT": "0",
        "STRAT_TRAIL_PCT": "1.5",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "0.0",
        "STRAT_REENTRY_HYBRID_ENABLED": "true",
        "STRAT_REENTRY_PULLBACK_PCT": "1.5",
        "STRAT_REENTRY_TTL_HOURS": "24.0",
        "STRAT_SL_MHDSL_ENABLED": "false",
        "STRAT_TP_MDHTP_ENABLED": "true",
        "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
    }
    config.update(overrides)
    return strategy.StratParams.from_env(config)


class T212MDHTPTest(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.account_cash.return_value = {"free": 1000.0}

    def test_mdhtp_disabled_places_legacy_resting_limit(self):
        """When MDHTP is disabled, reaching holding price places fixed resting limit SELL."""
        params = _params(STRAT_TP_MDHTP_ENABLED="false", STRAT_TAKEPROFIT_PCT="5")
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: 1000.0,
            trend_slope_provider=lambda _sym: 0.5,
        )
        self.assertIsNone(strat.exit_manager)
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            strat.step(100.0)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual(o["side"], "SELL")
            self.assertEqual(o["kind"], "TP")
            self.assertEqual(o["limit"], 105.0)
            self.assertFalse(o.get("market", False))

    def test_mdhtp_below_tp_keeps_resting_sell_clear(self):
        """When MDHTP is enabled, prices below TP do not place resting limit orders."""
        params = _params(STRAT_TP_MDHTP_ENABLED="true", STRAT_TAKEPROFIT_PCT="5")
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: 1000.0,
            trend_slope_provider=lambda _sym: 0.5,
        )
        self.assertIsNotNone(strat.exit_manager)
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            strat.step(102.0)  # +2% < +5% TP
            # No resting sell placed
            self.assertEqual(len(strat.s["orders"]), 0)
            self.assertNotIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)

    def test_mdhtp_initiates_trend_riding_in_uptrend(self):
        """Crossing TP in confirmed uptrend arms trend riding instead of selling."""
        params = _params(STRAT_TP_MDHTP_ENABLED="true", STRAT_TAKEPROFIT_PCT="3")
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: 1000.0,
            trend_slope_provider=lambda _sym: 0.25,  # Confirmed uptrend
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            strat.step(104.0)  # +4% > +3% TP
            # RIDING: no sell placed, peak recorded in ride state and trail_peak
            self.assertEqual(len(strat.s["orders"]), 0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)
            self.assertEqual(strat.s["trail_peak"], 104.0)
            self.assertEqual(strat.exit_manager.tp_ride_state["TEST_US_EQ"]["peak_price"], 104.0)

    def test_mdhtp_peak_pullback_triggers_market_exit(self):
        """Pullback >= tp_pullback_pct from peak triggers market TP exit."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="1.5",  # 1.5% pullback exit
        )
        current_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
            trend_slope_provider=lambda _sym: 0.25,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            # Step 1: initiate at 105
            strat.step(105.0)
            self.assertEqual(strat.s["trail_peak"], 105.0)

            # Step 2: expand to new peak 110.0
            current_time = 1050.0
            strat.step(110.0)
            self.assertEqual(strat.s["trail_peak"], 110.0)

            # Step 3: pullback to 108.0 (-1.82% from 110, >= 1.5% pullback)
            current_time = 1100.0
            strat.step(108.0)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual(o["side"], "SELL")
            self.assertEqual(o["kind"], "TP")
            self.assertTrue(o.get("market", False))

    def test_mdhtp_guaranteed_profit_floor_triggers_exit(self):
        """Dropping below guaranteed profit floor triggers defensive exit."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="5.0",  # Loose trail so pullback alone doesn't trigger
            STRAT_TRAIL_MIN_PROFIT_PCT="4.0",  # Minimum 4.0% profit floor
        )
        current_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
            trend_slope_provider=lambda _sym: 0.25,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            strat.step(106.0)  # +6% gain initiated
            self.assertIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)

            # Price drops to 103.5 (+3.5% gain < 4.0% floor)
            current_time = 1050.0
            strat.step(103.5)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual(o["kind"], "TP")
            self.assertTrue(o.get("market", False))

    def test_mdhtp_stagnation_timeout_triggers_exit(self):
        """Remaining in ride state without a new peak for > stagnation_max_sec exits."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="5.0",
            STRAT_TP_STAGNATION_MAX_SEC="600.0",  # 10 minute timeout
        )
        current_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
            trend_slope_provider=lambda _sym: 0.25,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            strat.step(105.0)  # Initiated at t=1000.0
            self.assertEqual(len(strat.s["orders"]), 0)

            # Advance time by 601 seconds with stagnant price
            current_time = 1601.0
            strat.step(105.0)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual(o["kind"], "TP")
            self.assertTrue(o.get("market", False))

    def test_mdhtp_trend_reversal_triggers_exit(self):
        """When trend turns down during trend ride, exit immediately."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="5.0",
        )
        current_time = 1000.0
        trend_slope = 0.25
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
            trend_slope_provider=lambda _sym: trend_slope,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            strat.step(105.0)
            self.assertEqual(len(strat.s["orders"]), 0)

            # Trend turns down
            trend_slope = -0.15
            current_time = 1020.0
            strat.step(104.8)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual(o["kind"], "TP")
            self.assertTrue(o.get("market", False))

    def test_mdhtp_takes_precedence_over_tp_ladder(self):
        """When MDHTP is enabled, MDHTP dynamic management takes precedence over passive TP ladder."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="5.0",
            STRAT_TP_LADDER="105:20,110:30,115:50",
        )
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: 1000.0,
            trend_slope_provider=lambda _sym: 0.2,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            # Ladder would normally place limit orders; MDHTP will ride trend without placing ladder limits
            strat.step(105.0)
            # In MDHTP trend ride, no orders placed while riding
            self.assertEqual(len(strat.s["orders"]), 0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)

    def test_mdhtp_order_refusal_intent_persistence_and_retry(self):
        """When MDHTP exit order is refused, intent is persisted in tp_pending_exit and retried on next step."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="5.0",
        )
        current_time = 1000.0
        trend_slope = -0.1  # downward trend immediately signals SELL
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
            trend_slope_provider=lambda _sym: trend_slope,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"), patch.object(strat, "_place_sell", return_value=False) as mock_place_sell:
            strat.step(105.0)
            # Order was refused by mock_place_sell
            self.assertIsNotNone(strat.s.get("tp_pending_exit"))
            self.assertEqual(strat.s["tp_pending_exit"]["reason"], "gain_threshold_exceeded_and_trend_not_up")

        # Next tick: order placement succeeds
        with patch.object(strategy, "notify"), patch.object(strat, "_place_sell", return_value=True) as mock_place_sell:
            current_time = 1005.0
            strat.step(104.9)
            mock_place_sell.assert_called_once()
            # Pending exit is cleared after successful placement
            self.assertIsNone(strat.s.get("tp_pending_exit"))

    def test_mdhtp_defers_and_cancels_all_orders_including_dca_and_resizes_on_late_fill(self):
        """MDHTP exit cancels all orders (including DCA BUYs), waits for cancellation reconciliation,
        and recalculates exit quantity dynamically when held increases from late BUY fills."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_TRAIL_PCT="5.0",
        )
        current_time = 1000.0
        trend_slope = -0.1
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
            trend_slope_provider=lambda _sym: trend_slope,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        # DCA BUY order is resting on the book
        dca_order = {"id": "DCA-BUY-1", "side": "BUY", "qty": 2.0, "limit": 95.0, "kind": "DCA", "market": False}
        strat.s["orders"] = [dca_order]

        cancelled_orders = []
        def fake_cancel_all():
            cancelled_orders.extend(strat.s["orders"])
            return True

        with patch.object(strategy, "notify"), \
             patch.object(strat, "_cancel_all_orders", side_effect=fake_cancel_all), \
             patch.object(strat, "_place_sell", return_value=True) as mock_place_sell:

            # Tick 1: MDHTP triggers -> initiates cancel_all, but DCA order is still in orders (reconciling)
            strat.step(105.0)
            self.assertEqual(len(cancelled_orders), 1)
            self.assertEqual(cancelled_orders[0]["id"], "DCA-BUY-1")
            self.assertIsNotNone(strat.s.get("tp_pending_exit"))
            # Market sell is NOT sent yet while pending orders are reconciling!
            mock_place_sell.assert_not_called()

            # Tick 2: Late DCA BUY filled before cancel, increasing held from 10 to 12
            strat.s["orders"] = []
            strat.s["qty"] = 12.0
            current_time = 1005.0
            strat.step(104.9)

            # Market sell must be executed for the fresh 12 shares, NOT the old 10!
            mock_place_sell.assert_called_once()
            called_qty = mock_place_sell.call_args[0][0]
            self.assertEqual(called_qty, 12.0)
            self.assertIsNone(strat.s.get("tp_pending_exit"))

    def test_mdhtp_restart_with_accepted_market_exit_order_does_not_cancel_it(self):
        """When bot restarts with an accepted protective market exit order and pending intent, it must not cancel it."""
        params = _params(
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TAKEPROFIT_PCT="3",
        )
        current_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: current_time,
        )
        strat.s.update(
            qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0,
            tp_pending_exit={"action": "SELL", "sell_fraction": 1.0, "reason": "mdhtp_exit", "time": 990.0},
            orders=[
                {"id": "MKT-EXIT-1", "side": "SELL", "market": True, "kind": "TP", "qty": 10.0},
                {"id": "DCA-RESTING-2", "side": "BUY", "market": False, "kind": "DCA", "qty": 2.0},
            ]
        )

        cancelled = []
        def fake_cancel_specific(o):
            cancelled.append(o)
            return True

        with patch.object(strategy, "notify"), \
             patch.object(strat, "_cancel_specific", side_effect=fake_cancel_specific), \
             patch.object(strat, "_place_sell") as mock_place_sell:
            # Step after restart: should cancel DCA-RESTING-2, but NOT MKT-EXIT-1!
            strat.step(105.0)
            self.assertEqual(len(cancelled), 1)
            self.assertEqual(cancelled[0]["id"], "DCA-RESTING-2")
            mock_place_sell.assert_not_called()
            self.assertIsNotNone(strat.s.get("tp_pending_exit"))

            # Step after fill reconciles qty to 0
            strat.s["qty"] = 0.0
            strat.s["orders"] = []
            strat.step(105.0)
            self.assertIsNone(strat.s.get("tp_pending_exit"))


if __name__ == "__main__":
    unittest.main()
