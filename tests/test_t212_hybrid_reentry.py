"""Tests for Trading 212 Hybrid Re-Entry state transitions."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T212_DIR = os.path.join(ROOT, "212trading")
sys.path.insert(0, T212_DIR)
sys.path.insert(0, ROOT)

_COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
_PRELOADED = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
try:
    spec = importlib.util.spec_from_file_location(
        "t212_strategy_under_test", os.path.join(T212_DIR, "strategy.py"),
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
        "STRAT_TRAIL_PCT": "0",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
        "STRAT_REENTRY_HYBRID_ENABLED": "true",
        "STRAT_REENTRY_PULLBACK_PCT": "1.5",
        "STRAT_REENTRY_TTL_HOURS": "24.0",
    }
    config.update(overrides)
    return strategy.StratParams.from_env(config)


class T212HybridReentryTest(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.account_cash.return_value = {"free": 1000.0}

    def test_state_b_trend_confirmed_bypasses_sale_barrier_and_waits_pullback(self):
        """In bull trend (slope >= 0), entry above last_sell is allowed after pullback from peak."""
        params = _params(STRAT_REENTRY_HYBRID_ENABLED="true", STRAT_REENTRY_PULLBACK_PCT="1.5")
        now_time = 10000.0
        clock = lambda: now_time
        # Slope positive -> bull
        slope_provider = lambda _sym: 0.25

        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=clock,
            trend_slope_provider=slope_provider,
        )
        # Position was sold at 100.0, price then rallied to 130.0
        strat.s["last_sell_price"] = 100.0
        strat.s["post_sell_peak"] = 100.0
        strat.s["last_sell_ts"] = now_time

        # Tick 1: price rises to 130.0 -> peak tracks 130.0
        # Required pullback = 1.5% from 130 = 128.05.
        # Price 130.0 > 128.05 -> re-entry blocked pending pullback
        strat.step(130.0)
        self.assertEqual(strat.s["post_sell_peak"], 130.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Tick 2: price at 129.0 -> still > 128.05 -> blocked
        strat.step(129.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Tick 3: price pulls back to 127.5 <= 128.05 -> unblocked!
        # Even though 127.5 > 100.0 (last_sell), bull trend allows re-entry!
        strat.step(127.5)
        self.assertEqual(len(strat.s["orders"]), 1)
        order = strat.s["orders"][0]
        self.assertEqual(order["kind"], "ENTRY")
        self.assertEqual(order["limit"], 127.5)

    def test_state_c_no_confirmed_trend_enforces_sale_drop(self):
        """In non-bull trend (slope < 0), price must drop below last_sell * (1 - drop_pct)."""
        params = _params(
            STRAT_REENTRY_HYBRID_ENABLED="true",
            STRAT_REENTRY_DROP_PCT="2.0",
            STRAT_REENTRY_PULLBACK_PCT="1.5",
        )
        now_time = 10000.0
        clock = lambda: now_time
        # Slope negative -> bear/chop
        slope_provider = lambda _sym: -0.15

        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=clock,
            trend_slope_provider=slope_provider,
        )
        # Sold at 100.0. Drop required: -2% -> <= 98.0.
        strat.s["last_sell_price"] = 100.0
        strat.s["post_sell_peak"] = 100.0
        strat.s["last_sell_ts"] = now_time

        # Price at 105.0 -> blocked
        strat.step(105.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Price at 99.0 -> blocked (99 > 98)
        strat.step(99.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Price drops to 97.5 <= 98.0 -> unblocked
        strat.step(97.5)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_ttl_expiry_unblocks_stale_barrier(self):
        """After TTL hours elapse, stale sale barrier expires even if not bull."""
        params = _params(
            STRAT_REENTRY_HYBRID_ENABLED="true",
            STRAT_REENTRY_DROP_PCT="2.0",
            STRAT_REENTRY_TTL_HOURS="24.0",
        )
        current_time = 10000.0
        clock = lambda: current_time
        slope_provider = lambda _sym: -0.10

        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            clock=clock,
            trend_slope_provider=slope_provider,
        )
        strat.s["last_sell_price"] = 100.0
        strat.s["post_sell_peak"] = 100.0
        strat.s["last_sell_ts"] = current_time

        # Price at 105.0 after 1 hour -> blocked
        current_time += 3600.0
        strat.step(105.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Price at 105.0 after 25 hours (> 24h TTL) -> TTL expired, unblocks!
        current_time += 24.0 * 3600.0
        strat.step(105.0)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_adaptive_pullback_scaling(self):
        """Dynamic pullback scales with volatility: clamp(K * vol, min, max)."""
        params = _params(
            STRAT_REENTRY_HYBRID_ENABLED="true",
            STRAT_REENTRY_PULLBACK_ADAPTIVE="true",
            STRAT_REENTRY_PULLBACK_K="1.0",
            STRAT_REENTRY_PULLBACK_MIN="1.0",
            STRAT_REENTRY_PULLBACK_MAX="4.0",
        )
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=True,
            trend_slope_provider=lambda _sym: 0.15,  # Confirmed bull
        )
        strat.s["last_sell_price"] = 90.0
        strat.s["post_sell_peak"] = 100.0

        # Inject 2.5% volatility
        strat._shadow_vol_1h = lambda: 2.5
        val, desc = strat._effective_reentry_pullback_pct()
        self.assertAlmostEqual(val, 2.5)
        self.assertIn("adaptive", desc)

        # Price at 98.5 (1.5% drop from 100) -> blocked by 2.5% threshold
        strat.step(98.5)
        self.assertEqual(len(strat.s["orders"]), 0)

        # Price at 97.0 (3.0% drop from 100) -> unblocked by 2.5% threshold
        strat.step(97.0)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
