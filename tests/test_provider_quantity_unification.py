"""Unit test suite verifying unified quantity (qty) management, precision rounding,
filter refusal, and available-balance order placement across all providers.
"""

import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from binance_api import bapi_placeorder as po
from instrument import Instrument
from providers.base import MarketDataProvider
from providers.kraken_provider import KrakenProvider
from providers.hyperliquid_provider import HyperliquidProvider
from providers.market_api import BinanceProvider, MarketApi
from providers.quantity import decide_quantity
from providers.replay_provider import ReplayMarketDataProvider
from providers.strategy_executor import PairPrecision, ProviderError
from providers.t212_provider import T212Provider


class CustomProvider(MarketDataProvider):
    name = "custom"

    def __init__(self, decimals=3, min_qty=0.05, balance=100.0):
        self._decimals = decimals
        self._min_qty = min_qty
        self._balance = balance

    def supports_symbol(self, symbol: str) -> bool:
        return True

    def get_current_price(self, symbol: str):
        return 10.0

    def free_balance(self, asset: str):
        return self._balance

    def pair_precision(self, symbol: str):
        return PairPrecision(
            price_decimals=2,
            volume_decimals=self._decimals,
            order_min=self._min_qty,
            base_asset="TEST",
        )


class TestProviderQuantityPrecision(unittest.TestCase):
    def test_base_provider_precision_and_rounding(self):
        """Verify default MarketDataProvider floors quantity according to volume_decimals."""
        p = CustomProvider(decimals=2, min_qty=0.1)
        self.assertEqual(p.min_order_qty("TESTUSDC"), 0.1)
        self.assertAlmostEqual(p.round_quantity("TESTUSDC", 1.239), 1.23, places=6)
        self.assertAlmostEqual(p.round_amount("TESTUSDC", 1.239), 1.23, places=6)
        self.assertAlmostEqual(p.round_price("TESTUSDC", 10.126), 10.13, places=6)

        # Filter refusal catches dust
        self.assertIn("filter_lot_size_min", p.order_filter_refusal("TESTUSDC", "BUY", 10.0, 0.05) or "")
        self.assertIsNone(p.order_filter_refusal("TESTUSDC", "BUY", 10.0, 0.15))

    def test_kraken_provider_precision_and_filter(self):
        """Verify KrakenProvider rounds volume to lot_decimals and checks ordermin in filter."""
        p = KrakenProvider()
        fake_client = MagicMock()
        fake_client.pair_info.return_value = {
            "pair_decimals": 2,
            "lot_decimals": 4,
            "ordermin": "0.01",
            "base": "ETH",
        }
        p._client = lambda: fake_client

        pp = p.pair_precision("ETHUSD")
        self.assertIsNotNone(pp)
        self.assertEqual(pp.volume_decimals, 4)
        self.assertEqual(pp.order_min, 0.01)

        # Rounding floors to 4 decimals
        self.assertAlmostEqual(p.round_quantity("ETHUSD", 0.12349), 0.1234, places=6)
        self.assertAlmostEqual(p.round_price("ETHUSD", 3000.126), 3000.13, places=6)

        # Filter catches dust < 0.01
        refusal = p.order_filter_refusal("ETHUSD", "BUY", 3000.0, 0.005)
        self.assertIsNotNone(refusal)
        self.assertIn("filter_lot_size_min", refusal)

        # Filter allows >= 0.01
        self.assertIsNone(p.order_filter_refusal("ETHUSD", "BUY", 3000.0, 0.01))

        # preflight_order with qty=None does not raise
        p.preflight_order("ETHUSD", "SELL", None)

    def test_hyperliquid_provider_precision_and_preflight(self):
        """Verify HyperliquidProvider rounds volume to sz_decimals and handles None in preflight."""
        p = HyperliquidProvider()
        fake_hl = MagicMock()
        fake_hl.sz_decimals.return_value = 2
        p._hl = lambda: fake_hl
        p.validate_symbol = lambda s: None

        pp = p.pair_precision("PURR/USDC")
        self.assertIsNotNone(pp)
        self.assertEqual(pp.volume_decimals, 2)
        # 5 significant figures limit + (8 - szDecimals) spot decimals
        self.assertAlmostEqual(p.round_price("PURR/USDC", 1.23456789), 1.2346, places=6)
        self.assertAlmostEqual(p.round_price("PURR/USDC", 1.234568), 1.2346, places=6)

        # preflight_order with qty=None passes safely
        p.preflight_order("PURR/USDC", "BUY", None)

    def test_t212_provider_precision_and_filter(self):
        """Verify T212Provider rounds volume to 2 decimals and checks _ORDER_MIN."""
        p = T212Provider()
        pp = p.pair_precision("AAPL_US_EQ")
        self.assertIsNotNone(pp)
        self.assertEqual(pp.volume_decimals, 2)
        self.assertEqual(pp.order_min, 0.01)

        self.assertAlmostEqual(p.round_quantity("AAPL_US_EQ", 3.456), 3.45, places=6)
        self.assertAlmostEqual(p.round_price("AAPL_US_EQ", 150.126), 150.13, places=6)
        self.assertIn("filter_lot_size_min", p.order_filter_refusal("AAPL_US_EQ", "BUY", 150.0, 0.005) or "")
        self.assertIsNone(p.order_filter_refusal("AAPL_US_EQ", "BUY", 150.0, 0.05))

    def test_replay_provider_precision(self):
        """Verify ReplayMarketDataProvider provides 4-decimal volume precision."""
        p = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0)]})
        pp = p.pair_precision("BTCUSDC")
        self.assertIsNotNone(pp)
        self.assertEqual(pp.volume_decimals, 4)
        self.assertAlmostEqual(p.round_quantity("BTCUSDC", 0.12349), 0.1234, places=6)
        self.assertAlmostEqual(p.round_price("BTCUSDC", 50000.126), 50000.13, places=6)

    def test_market_data_registry_delegates_precision_methods(self):
        """Verify MarketApi forwards pair_precision, min_order_qty, round_quantity."""
        cp = CustomProvider(decimals=3, min_qty=0.02)
        registry = MarketApi([cp])

        pp = registry.pair_precision("TESTUSDC")
        self.assertIsNotNone(pp)
        self.assertEqual(pp.volume_decimals, 3)
        self.assertEqual(registry.min_order_qty("TESTUSDC"), 0.02)
        self.assertAlmostEqual(registry.round_quantity("TESTUSDC", 1.9876), 1.987, places=6)
        self.assertAlmostEqual(registry.round_amount("TESTUSDC", 1.9876), 1.987, places=6)
        self.assertAlmostEqual(registry.round_price("TESTUSDC", 10.126), 10.13, places=6)


class TestQuantityDecisionAndAvailableBalance(unittest.TestCase):
    def test_decide_quantity_floors_to_venue_precision(self):
        """Verify decide_quantity applies venue volume precision before returning."""
        p = CustomProvider(decimals=2, min_qty=0.01, balance=500.0)
        # Price 100, balance 500 -> balance cap 5.0. Requested 1.239 -> final should floor to 1.23
        decision = decide_quantity(p, "TESTUSDC", "BUY", 100.0, 1.239, apply_policy=False)
        self.assertEqual(decision.final_qty, 1.23)

    def test_decide_quantity_none_resolves_to_available_balance(self):
        """Verify requested_qty=None sizes to maximum allowable balance floored to precision."""
        p = CustomProvider(decimals=3, min_qty=0.01, balance=250.0)
        # Price 100.0, fee cap 0.0 -> balance cap 2.5. Final is 2.5 floored to 3 decimals = 2.5
        decision = decide_quantity(p, "TESTUSDC", "BUY", 100.0, None, apply_policy=False)
        self.assertEqual(decision.final_qty, 2.5)
        self.assertIsNone(decision.refuse_reason)

    def test_instrument_place_with_none_qty(self):
        """Verify Instrument.place can be called with qty=None or omitted."""
        from lock.trade_cooldown import release_trade
        release_trade("AVAIL_TEST_USDC")
        p = CustomProvider(decimals=2, min_qty=0.01, balance=500.0)
        placed_orders = []
        p.place_order = lambda sym, side, px, q, **kw: placed_orders.append((sym, side, px, q, kw)) or {"orderId": 123}
        p.guards_internally = lambda: False
        p.execution_enabled = lambda: True

        registry = MarketApi([p])
        inst = Instrument("AVAIL_TEST_USDC", symbol="AVAIL_TEST_USDC", provider="custom", api=registry)

        # Call with qty=None: sizes to available balance (500 / 100 = 5.0)
        res = inst.place("BUY", 100.0, None, bypass_profit_guard=True, wait_for_trend=False)
        self.assertIsNotNone(res)
        self.assertEqual(len(placed_orders), 1)
        sym, side, px, q, kw = placed_orders[0]
        self.assertEqual(q, 5.0)
        self.assertTrue(kw.get("_balance_verified"))


class TestBinanceTransientBalanceSoftSkip(unittest.TestCase):
    def test_soft_skip_transient_balance_failure_with_verified_qty(self):
        """Verify place_order_mechanics does not reject order if get_free_balance fails when _balance_verified is True."""
        with patch.object(po.api, "get_free_balance", return_value=None), \
             patch.object(po.api, "get_current_price", return_value=100.0), \
             patch("providers.binance_filters.BinanceOrderRules.from_symbol_info") as mock_rules_info, \
             patch.object(po, "_submit_binance_order", return_value={"orderId": 9999}) as mock_submit:

            mock_rules = MagicMock()
            mock_rules.normalize.return_value = ("2.0", "100.0")
            mock_rules_info.return_value = mock_rules

            # 1. Without _balance_verified: returns None (skipped)
            res1 = po.place_order_mechanics("BUY", "BTCUSDC", 100.0, 2.0, force=False)
            self.assertIsNone(res1)
            mock_submit.assert_not_called()

            # 2. With _balance_verified=True: soft-skips balance failure, submits verified qty
            res2 = po.place_order_mechanics("BUY", "BTCUSDC", 100.0, 2.0, force=False, _balance_verified=True)
            self.assertEqual(res2, {"orderId": 9999})
            mock_submit.assert_called_once()

    def test_binance_provider_place_order_forwards_balance_verified(self):
        """Verify BinanceProvider.place_order passes _balance_verified through to mechanics."""
        with patch.object(po.api, "get_free_balance", return_value=None), \
             patch.object(po.api, "get_current_price", return_value=100.0), \
             patch("providers.binance_filters.BinanceOrderRules.from_symbol_info") as mock_rules_info, \
             patch.object(po, "_submit_binance_order", return_value={"orderId": 8888}) as mock_submit:

            mock_rules = MagicMock()
            mock_rules.normalize.return_value = ("2.0", "100.0")
            mock_rules_info.return_value = mock_rules

            from providers.market_api import BinanceProvider
            bp = BinanceProvider()
            bp.validate_symbol = lambda s: None

            # Through BinanceProvider.place_order without _balance_verified: returns None
            res1 = bp.place_order("BTCUSDC", "BUY", 100.0, 2.0)
            self.assertIsNone(res1)
            mock_submit.assert_not_called()

            # Through BinanceProvider.place_order with _balance_verified=True: succeeds
            res2 = bp.place_order("BTCUSDC", "BUY", 100.0, 2.0, _balance_verified=True)
            self.assertEqual(res2, {"orderId": 8888})
            mock_submit.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
