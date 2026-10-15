"""Tests for Centralized PositionExitManager (MHDSL and Re-entry) integration in 212trading/strategy.py."""

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
        "t212_strategy_mhdsl_test", os.path.join(T212_DIR, "strategy.py"),
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
        "STRAT_DCA_DROP_PCT": "2",
        "STRAT_TAKEPROFIT_PCT": "3",
        "STRAT_MAX_DCA_BUYS": "3",
        "STRAT_MAX_BUDGET": "500",
        "STRAT_FX_FEE_PCT": "0.15",
        "STRAT_STOP_LOSS_PCT": "10",
        "STRAT_CHECK_MINUTES": "5",
        "STRAT_ORDER_TTL_MIN": "10",
        "STRAT_REENTRY_DROP_PCT": "2.0",
        "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
        "STRAT_LOSS_ALERT_STEP": "1",
        "STRAT_LADDER_MIN_FREE": "6",
        "STRAT_SL_REBUY_ENABLED": "false",
        "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
        "STRAT_DCA_TREND_GATE_PCT": "0",
        "STRAT_TRAIL_PCT": "0",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
        "STRAT_REENTRY_HYBRID_ENABLED": "true",
        "STRAT_REENTRY_PULLBACK_PCT": "1.5",
        "STRAT_REENTRY_TTL_HOURS": "24.0",
        "STRAT_SL_MHDSL_ENABLED": "true",
        "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
        "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
        "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
        "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
    }
    config.update(overrides)
    return strategy.StratParams.from_env(config)


class T212MHDSLTest(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.account_cash.return_value = {"free": 1000.0}

    def test_mhdsl_disabled_preserves_legacy_stop(self):
        """When STRAT_SL_MHDSL_ENABLED is false, legacy behavior executes immediately on breach."""
        params = _params(STRAT_SL_MHDSL_ENABLED="false", STRAT_STOP_LOSS_PCT="10")
        now_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
            trend_slope_provider=lambda _sym: 0.5,  # Strong uptrend
        )
        self.assertIsNone(strat.exit_manager)

        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0)
        # Price 89.0 -> 11% loss >= 10% threshold
        with patch.object(strategy, "notify"):
            triggered = strat._check_stop_loss(89.0)
            self.assertTrue(triggered)
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual(strat.s["orders"][0]["kind"], "SL")

    def test_mhdsl_trend_deferral_holds_in_uptrend(self):
        """When MHDSL is enabled, confirmed uptrend defers stop-loss within disaster buffer."""
        params = _params(
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_STOP_LOSS_PCT="10",
            STRAT_SL_DISASTER_BUFFER_RATIO="0.30",  # Disaster floor at 13% loss (87.0)
            STRAT_SL_TREND_DEFER_MAX_SEC="220.0",
        )
        now_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
            trend_slope_provider=lambda _sym: 0.25,  # Confirmed uptrend
        )
        self.assertIsNotNone(strat.exit_manager)

        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0)
        # Price 89.0 -> 11% loss (exceeds 10% SL, but within 13% disaster floor)
        with patch.object(strategy, "notify"):
            triggered = strat._check_stop_loss(89.0)
            # Deferred! No SL order placed
            self.assertFalse(triggered)
            self.assertEqual(len(strat.s["orders"]), 0)

    def test_mhdsl_hard_disaster_circuit_breaker(self):
        """Breaching the disaster floor forces immediate exit regardless of trend."""
        params = _params(
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_STOP_LOSS_PCT="10",
            STRAT_SL_DISASTER_BUFFER_RATIO="0.30",  # Disaster floor at 13% loss (87.0)
        )
        now_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
            trend_slope_provider=lambda _sym: 0.5,  # Strong uptrend
        )

        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0)
        # Price 86.0 -> 14% loss > 13% disaster threshold -> Hard Disaster Exit
        with patch.object(strategy, "notify"):
            triggered = strat._check_stop_loss(86.0)
            self.assertTrue(triggered)
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual(strat.s["orders"][0]["kind"], "SL")

    def test_mhdsl_reentry_mirror_requires_bounce_after_stop(self):
        """Post-STOP exit re-entry enforces bounce confirmation from trough."""
        params = _params(
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_REENTRY_DROP_PCT="2.0",
            STRAT_REENTRY_PULLBACK_PCT="1.5",
        )
        now_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
            trend_slope_provider=lambda _sym: -0.2,  # downtrend/chop
        )

        strat.s.update(
            qty=0.0,
            last_sell_price=90.0,
            post_sell_peak=90.0,
            post_sell_trough=90.0,
            last_exit_kind="STOP",
            last_sell_ts=now_time,
        )

        # Price continuing to drop (falling knife) -> trough tracks, entry blocked
        strat.step(85.0)
        self.assertEqual(len(strat.s["orders"]), 0)
        self.assertEqual(strat.s["post_sell_trough"], 85.0)

        # Small bounce from 85.0 to 85.5 (+0.58% < 1.5% required) -> still blocked
        strat.step(85.5)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Confirmed bounce: 85.0 * 1.016 = 86.36 >= +1.5% bounce -> unblocked
        strat.step(86.4)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_cycle_closure_clears_mhdsl_state(self):
        """When cycle closes, exit_manager state is cleanly cleared."""
        params = _params(STRAT_SL_MHDSL_ENABLED="true")
        now_time = 1000.0
        self.mock_client.get_portfolio.return_value = []
        self.mock_client.list_active_orders.return_value = []
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=False,
            clock=lambda: now_time,
        )

        strat.exit_manager.sl_bounce_state["TEST_US_EQ"] = {"state": "watching"}
        strat.s["orders"] = []
        strat.s["qty"] = 10.0
        strat.s["realized_pnl_usd"] = 15.0
        strat.s["realized_net_usd"] = 14.5
        strat.s["fees_usd"] = 0.5
        strat.s["last_sell_price"] = 105.0

        # Simulate reconcile where position becomes flat (cycle_closed)
        strat.reconcile(105.0)  # mock client reports 0 qty
        self.assertEqual(strat.s["cycle"], 2)
        self.assertNotIn("TEST_US_EQ", strat.exit_manager.sl_bounce_state)
        self.assertEqual(strat.s["sl_bounce_state"], {})

    def test_mhdsl_blocks_dca_during_bounce_deferral(self):
        """When in MHDSL bounce deferral, DCA buying is blocked."""
        params = _params(
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_STOP_LOSS_PCT="10",
            STRAT_DCA_DROP_PCT="2",
            STRAT_MAX_DCA_BUYS="3",
        )
        now_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
            trend_slope_provider=lambda _sym: 0.25,  # Confirmed uptrend defers SL
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        with patch.object(strategy, "notify"):
            # Price 89.0 -> 11% loss breaches 10% SL, but deferred due to uptrend
            strat.step(89.0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.sl_bounce_state)
            # DCA buy must be blocked
            dca_buys = [o for o in strat.s["orders"] if o.get("kind") == "DCA" or o.get("side") == "buy"]
            self.assertEqual(len(dca_buys), 0)

    def test_trailing_warmup_protects_dca_and_mhdsl_layers(self):
        """When trailing stop is configured with warm-up, ordinary dips do not exit;
        profit trailing arms only upon reaching warmup, and catastrophic stop loss cuts deep crashes."""
        params = _params(
            STRAT_TRAIL_PCT="8.0",
            STRAT_TRAIL_MIN_PROFIT_PCT="5.0",
            STRAT_STOP_LOSS_PCT="30.0",
            STRAT_DCA_DROP_PCT="2.0",
        )
        now_time = 1000.0
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)

        # 1. Price dips to 91 (-9% from 100): in warm-up (and loss 9% < 30% SL), so trailing does NOT exit
        with patch.object(strategy, "notify"):
            strat.step(91.0)
            self.assertFalse(any(o.get("kind") == "TR" for o in strat.s["orders"]))
            self.assertFalse(strat.s.get("tr_armed", False))

        # 2. Catastrophic crash to 68 (loss 32% >= 30% SL) triggers catastrophic MHDSL stop-loss
        strat.s["orders"] = []
        with patch.object(strategy, "notify"):
            strat.step(68.0)
            self.assertTrue(any(o.get("kind") == "SL" for o in strat.s["orders"]))

        # 3. If price reaches profit (106 >= 105), trailing arms and subsequently protects gains
        strat2 = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: now_time,
        )
        strat2.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)
        with patch.object(strategy, "notify"):
            strat2.step(106.0)
            self.assertTrue(strat2.s.get("tr_armed"))
            self.assertEqual(strat2.s.get("pos_peak"), 106.0)
            # Retracement from 106 to 97 (-8.5% >= 8%) triggers trailing exit
            strat2.step(97.0)
            self.assertEqual(len(strat2.s["orders"]), 1)
            self.assertEqual(strat2.s["orders"][0]["kind"], "TR")

    def test_t212_vol_1h_percentage_converted_to_ratio_for_exit_manager(self):
        """Verify _shadow_vol_1h percentage (2.5%) is converted to ratio (0.025) for PositionExitManager."""
        params = _params(
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_STOP_LOSS_PCT="10.0",
        )
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: 1000.0,
        )
        strat.s.update(qty=10.0, cost_usd=1000.0, spent_cash=1000.0, entry_price=100.0, last_buy_price=100.0)
        strat._shadow_vol_1h = lambda: 2.5  # 2.5% volatility

        captured_snapshots = []
        orig_eval = strat.exit_manager.evaluate_stop_loss

        def mock_eval(snap, custom_threshold=None):
            captured_snapshots.append(snap)
            return orig_eval(snap, custom_threshold=custom_threshold)

        strat.exit_manager.evaluate_stop_loss = mock_eval

        strat._check_stop_loss(89.0)
        self.assertEqual(len(captured_snapshots), 1)
        self.assertAlmostEqual(captured_snapshots[0].vol_1h, 0.025, places=5)
        self.assertNotAlmostEqual(captured_snapshots[0].vol_1h, 2.5, places=2)

    def test_t212_sl_disaster_buffer_max_pct_propagates_to_exit_manager(self):
        """Verify STRAT_SL_DISASTER_BUFFER_MAX_PCT parses from env and converts to ratio in PositionExitConfig."""
        params = _params(
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_SL_DISASTER_BUFFER_MAX_PCT="12.5",
        )
        self.assertEqual(params.sl_disaster_buffer_max_pct, 12.5)
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
        )
        self.assertIsNotNone(strat.exit_manager)
        self.assertAlmostEqual(strat.exit_manager.config.sl_disaster_buffer_max_pct, 0.125, places=5)



