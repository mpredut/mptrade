"""Unit test suite verifying unified quantity (qty) management, precision rounding,
filter refusal, and available-balance order placement across all providers.
"""

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
from providers.market_api import MarketApi
from providers.quantity import decide_quantity
from providers.replay_provider import ReplayMarketDataProvider
from providers.strategy_executor import PairPrecision
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

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs):
        return available_qty


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
        res = inst.place("BUY", 100.0, None, bypass_guards={"profit_guard"}, wait_for_trend=False)
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
             patch.object(po.client, "get_symbol_info", return_value={}), \
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
             patch.object(po.client, "get_symbol_info", return_value={}), \
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


class TestCentralizedBalanceSingleSourceOfTruth(unittest.TestCase):
    def test_decide_quantity_reuses_known_balance_without_calling_free_balance(self):
        """Verify decide_quantity never calls provider.free_balance when known_balance is passed."""
        p = CustomProvider(decimals=2, min_qty=0.01, balance=500.0)
        p.free_balance = MagicMock(side_effect=AssertionError("free_balance must not be called when known_balance is supplied"))

        decision = decide_quantity(p, "TESTUSDC", "BUY", 100.0, None, apply_policy=False, known_balance=300.0)
        self.assertEqual(decision.final_qty, 3.0)
        p.free_balance.assert_not_called()

    def test_instrument_place_queries_free_balance_exactly_once_on_buy(self):
        """Verify Instrument.place fetches free_balance exactly once throughout the entire BUY lifecycle."""
        from lock.trade_cooldown import release_trade
        release_trade("SSOT_BUY_USDC")

        p = CustomProvider(decimals=2, min_qty=0.01, balance=500.0)
        p.free_balance = MagicMock(return_value=500.0)
        placed_orders = []
        p.place_order = lambda sym, side, px, q, **kw: placed_orders.append((sym, side, px, q, kw)) or {"orderId": 111}
        p.guards_internally = lambda: False
        p.execution_enabled = lambda: True

        registry = MarketApi([p])
        inst = Instrument("SSOT_BUY_USDC", symbol="SSOT_BUY_USDC", provider="custom", api=registry)

        res = inst.place("BUY", 100.0, None, bypass_guards={"profit_guard"}, wait_for_trend=False)
        self.assertIsNotNone(res)
        self.assertEqual(p.free_balance.call_count, 1)
        p.free_balance.assert_called_once_with("USDC")
        self.assertEqual(placed_orders[0][3], 5.0)
        self.assertEqual(placed_orders[0][4].get("known_balance"), 500.0)

    def test_instrument_place_queries_free_balance_exactly_once_on_sell(self):
        """Verify Instrument.place fetches free_balance exactly once throughout the entire SELL lifecycle."""
        from lock.trade_cooldown import release_trade
        release_trade("SSOT_SELL_USDC")

        p = CustomProvider(decimals=2, min_qty=0.01, balance=2.5)
        p.free_balance = MagicMock(return_value=2.5)
        placed_orders = []
        p.place_order = lambda sym, side, px, q, **kw: placed_orders.append((sym, side, px, q, kw)) or {"orderId": 222}
        p.guards_internally = lambda: False
        p.execution_enabled = lambda: True

        registry = MarketApi([p])
        inst = Instrument("SSOT_SELL_USDC", symbol="SSOT_SELL_USDC", provider="custom",
                          base="SSOT_SELL", quote="USDC", api=registry)

        res = inst.place("SELL", 100.0, None, bypass_guards={"profit_guard"}, wait_for_trend=False)
        self.assertIsNotNone(res)
        self.assertEqual(p.free_balance.call_count, 1)
        p.free_balance.assert_called_once_with("SSOT_SELL")
        self.assertEqual(placed_orders[0][3], 2.5)
        self.assertEqual(placed_orders[0][4].get("known_balance"), 2.5)

    def test_instrument_place_fails_fast_on_zero_balance_for_both_sides(self):
        """Verify Stage 0 fails fast without proceeding to guards when balance is zero."""
        p = CustomProvider(decimals=2, min_qty=0.01, balance=0.0)
        p.free_balance = MagicMock(return_value=0.0)
        p.guards_internally = lambda: False
        p.execution_enabled = lambda: True

        registry = MarketApi([p])
        inst = Instrument("ZERO_TEST_USDC", symbol="ZERO_TEST_USDC", provider="custom", api=registry)

        # BUY with 0 quote balance fails in Stage 0
        outcome_buy = {}
        res_buy = inst.place("BUY", 100.0, 1.0, caller_owns_retry=True, _outcome_context=outcome_buy)
        self.assertIsNone(res_buy)
        self.assertEqual(outcome_buy.get("reason"), "balance_unavailable")

        # SELL with 0 base balance fails in Stage 0
        outcome_sell = {}
        res_sell = inst.place("SELL", 100.0, 1.0, caller_owns_retry=True, _outcome_context=outcome_sell)
        self.assertIsNone(res_sell)
        self.assertEqual(outcome_sell.get("reason"), "balance_unavailable")

    def test_ssot_hyperliquid_buy_and_t212_sell_clamping_in_general_pipeline(self):
        """Verify Hyperliquid 0.95 BUY reserve and T212 $6 SELL dust/remainder rules work in decide_quantity and Instrument.place."""
        from lock.trade_cooldown import release_trade
        from providers.quantity import clamp_notional_to_wallet_balance
        release_trade("HYPEUSDC")

        # 1. Hyperliquid oversized BUY via Instrument.place clamps to 0.95 * free_balance without preflight failure
        hl = HyperliquidProvider(token="HYPE", live_enabled=True)
        fake_hl = MagicMock()
        fake_hl.sz_decimals.return_value = 2
        hl._hl = lambda: fake_hl
        hl.free_balance = MagicMock(return_value=100.0)
        hl.get_current_price = MagicMock(return_value=40.0)
        placed_hl = []
        hl.place_order = lambda sym, side, px, q, **kw: placed_hl.append((sym, side, px, q)) or {"orderId": 777}

        registry_hl = MarketApi([hl])
        inst_hl = Instrument("HYPE", symbol="HYPEUSDC", provider="hyperliquid", base="HYPE", quote="USDC", api=registry_hl)
        res_hl = inst_hl.place("BUY", 40.0, 10.0, market=True, bypass_guards={"all"}, wait_for_trend=False, caller_owns_retry=True)
        self.assertIsNotNone(res_hl)
        # $100 * 0.95 / $40 = 2.375 -> floored to 2 decimals = 2.37
        self.assertAlmostEqual(placed_hl[0][3], 2.37, places=6)

        # 2. T212 full SELL with sub-2-decimal dust and partial SELL leaving <$6 remainder via decide_quantity
        t212 = T212Provider(live_enabled=True)
        t212.free_balance = MagicMock(return_value=5.15296546)
        dec_full = decide_quantity(t212, "SPCX_US_EQ", "SELL", 150.0, 10.0, apply_policy=False)
        self.assertAlmostEqual(dec_full.final_qty, 5.11, places=6)

        t212.free_balance = MagicMock(return_value=1.004)
        dec_part = decide_quantity(t212, "NVDA_US_EQ", "SELL", 100.0, 0.96, apply_policy=False)
        self.assertAlmostEqual(dec_part.final_qty, 0.94, places=6)

        # 3. clamp_notional_to_wallet_balance shares the same SSOT rules
        ok_n, usd_n, _ = clamp_notional_to_wallet_balance(
            t212, "SELL", 100.0, 96.0, symbol="NVDA_US_EQ"
        )
        self.assertTrue(ok_n)
        self.assertAlmostEqual(usd_n, 94.0, places=4)

    def test_import_providers_quantity_first_without_circular_import(self):
        """Importing providers.quantity before providers.base (as price_watcher does) must not raise ImportError."""
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-c", "from providers.quantity import resolve_assets, compute_fills_cost_basis; assert resolve_assets('BTCUSDC') == ('BTC', 'USDC')"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


from types import SimpleNamespace as _SimpleNamespace
from unittest.mock import patch as _patch
import order_guard as _order_guard
from providers.quantity import (
    balance_cap_quantity,
    fee_cap_quantity,
    resolve_assets,
)


class _QtyDecisionFakeProvider(MarketDataProvider):
    name = "fake"
    def __init__(self, balances, policy=999):
        self.balances = balances
        self.policy = policy
        self.filter_refusal = None
        self.filter_calls = []
    def get_current_price(self, symbol): return 100.0
    def supports_symbol(self, symbol): return True
    def free_balance(self, asset): return self.balances.get(asset, 0.0)
    def policy_cap_quantity(self, symbol, side, price, qty, available_qty, **kwargs):
        return min(qty, self.policy)
    def order_filter_refusal(self, symbol, side, price, qty, *, market=False,
                             enforce_business_minimum=True):
        self.filter_calls.append((
            symbol, side, price, qty, market, enforce_business_minimum))
        return self.filter_refusal


class QuantityDecisionTest(unittest.TestCase):
    def test_assets_are_resolved_once_for_long_suffixes(self):
        self.assertEqual(resolve_assets("TAOUSDC"), ("TAO", "USDC"))
        self.assertEqual(resolve_assets("BTCUSDC"), ("BTC", "USDC"))

    def test_resolver_covers_non_usdc_inventory_quotes_without_usdt(self):
        self.assertEqual(resolve_assets("TLVRON"), ("TLV", "RON"))
        self.assertEqual(resolve_assets("HYPEUSD"), ("HYPE", "USD"))
        self.assertEqual(resolve_assets("TAOUSDT"), ("TAO", "USDT"))
        self.assertEqual(resolve_assets("TAOBRL"), ("TAOBRL", None))

    def test_buy_converts_quote_balance_to_base_quantity(self):
        cap, asset = balance_cap_quantity(
            lambda a: {"USDC": 450}[a], "TAOUSDC", "BUY", 225)
        self.assertEqual((cap, asset), (2.0, "USDC"))

    def test_sell_uses_base_balance(self):
        decision = decide_quantity(
            _QtyDecisionFakeProvider({"TAO": 1.0}, policy=0.6),
            "TAOUSDC", "SELL", 225, 2.0)
        self.assertEqual(decision.requested_qty, 2.0)
        self.assertEqual(decision.balance_cap, 1.0)
        self.assertEqual(decision.policy_cap, 0.6)
        self.assertEqual(decision.final_qty, 0.6)

    def test_none_means_maximum_allowed_by_balance_and_policy(self):
        decision = decide_quantity(
            _QtyDecisionFakeProvider({"USDC": 500.0}, policy=3.0),
            "TAOUSDC", "BUY", 100.0, None)
        self.assertEqual(decision.balance_cap, 5.0)
        self.assertEqual(decision.policy_cap, 3.0)
        self.assertEqual(decision.final_qty, 3.0)

    def test_risk_exit_skips_policy_but_keeps_balance_and_fee_caps(self):
        provider = _QtyDecisionFakeProvider({"TAO": 0.4}, policy=0.01)
        provider.fee_cap_quantity = lambda *_args: 0.39
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 2.0,
            apply_policy=False)
        self.assertEqual(decision.final_qty, 0.39)

    def test_provider_filter_refuses_before_submission(self):
        provider = _QtyDecisionFakeProvider({"TAO": 1.0})
        provider.filter_refusal = "below_min_notional"
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 0.01)
        self.assertEqual(decision.final_qty, 0.0)
        self.assertEqual(decision.refuse_reason, "below_min_notional")
        self.assertEqual(
            provider.filter_calls,
            [("TAOUSDC", "SELL", 100.0, 0.01, False, True)],
        )

    def test_market_flag_reaches_filter_when_policy_is_disabled(self):
        provider = _QtyDecisionFakeProvider({"TAO": 1.0}, policy=0.0)
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 0.5,
            apply_policy=False, market=True)
        self.assertEqual(decision.final_qty, 0.5)
        self.assertTrue(provider.filter_calls[0][-2])

    def test_business_minimum_context_reaches_provider_filter(self):
        provider = _QtyDecisionFakeProvider({"TAO": 1.0})
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 0.5,
            enforce_business_minimum=False)
        self.assertEqual(decision.final_qty, 0.5)
        self.assertFalse(provider.filter_calls[0][-1])

    def test_filter_failure_is_not_silently_bypassed(self):
        provider = _QtyDecisionFakeProvider({"TAO": 1.0})
        provider.order_filter_refusal = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("rules unavailable"))
        with self.assertRaisesRegex(RuntimeError, "rules unavailable"):
            decide_quantity(provider, "TAOUSDC", "SELL", 100.0, 0.5)

    def test_none_is_error_but_zero_is_real_insufficient_balance(self):
        unavailable = decide_quantity(
            _QtyDecisionFakeProvider({"USDC": None}), "TAOUSDC", "BUY", 225, 1)
        empty = decide_quantity(
            _QtyDecisionFakeProvider({"USDC": 0}), "TAOUSDC", "BUY", 225, 1)
        self.assertEqual(unavailable.refuse_reason, "balance_unavailable")
        self.assertEqual(empty.refuse_reason, "insufficient_funds")

    def test_fee_cap_operates_on_base_quantity(self):
        self.assertAlmostEqual(fee_cap_quantity(2.0, 0.01), 2.0 / 1.01)

    def test_weight_limit_uses_precomputed_balance_without_refetch(self):
        provider = _QtyDecisionFakeProvider({"USDC": 500})
        provider.free_balance = lambda _asset: self.fail("the balance must not be re-read")
        fake_pa = _SimpleNamespace(
            get_weight_for_cash_permission_at_quant_time=lambda *_: 0.5)
        with _patch.dict("sys.modules", {"priceAnalysis": fake_pa}):
            qty = _order_guard.weight_limit(
                provider, "TAOUSDC", "BUY", 100.0, 5.0,
                available_qty=5.0)
        self.assertEqual(qty, 2.5)

    def test_partial_wallet_balance_below_min_filter_reports_insufficient_funds(self):
        provider = _QtyDecisionFakeProvider({"USDC": 4.65})
        provider.order_filter_refusal = (
            lambda _s, _side, px, q, **_kw: "below_min_notional" if q * px < 5.0 else None
        )
        decision = decide_quantity(provider, "TAOUSDC", "BUY", 100.0, 3.0)
        self.assertEqual(decision.final_qty, 0.0)
        self.assertEqual(decision.refuse_reason, "insufficient_funds")

        provider_sub_step = _QtyDecisionFakeProvider({"USDC": 0.004})
        provider_sub_step.round_amount = lambda _s, q: float(int(q * 100) / 100.0)
        decision_sub = decide_quantity(provider_sub_step, "TAOUSDC", "BUY", 100.0, 1.0)
        self.assertEqual(decision_sub.final_qty, 0.0)
        self.assertEqual(decision_sub.refuse_reason, "insufficient_funds")

        decision_dust_req = decide_quantity(provider, "TAOUSDC", "BUY", 100.0, 0.03)
        self.assertEqual(decision_dust_req.final_qty, 0.0)
        self.assertEqual(decision_dust_req.refuse_reason, "below_min_notional")


if __name__ == "__main__":
    unittest.main(verbosity=2)

