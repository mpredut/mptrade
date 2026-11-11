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
from providers.base import MarketDataProvider, aggregate_fills_by_order_id, extract_order_id
from providers.kraken_provider import KrakenProvider
from providers.hyperliquid_provider import HyperliquidProvider
from providers.market_api import MarketApi
from providers.quantity import (
    balance_cap_quantity,
    decide_quantity,
    fee_cap_quantity,
    resolve_assets,
)
from providers.replay_provider import ReplayMarketDataProvider, load_price_series
from providers.strategy_executor import PairPrecision, ProviderError, SubmissionOutcome
from providers.t212_provider import T212Provider
import botcore
import math
import pytest
from types import SimpleNamespace
import order_guard as _order_guard
from providers import base as pbase, order_ohlc_utils as oou


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
             patch("providers.quantity.BinanceOrderRules.from_symbol_info") as mock_rules_info, \
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
             patch("providers.quantity.BinanceOrderRules.from_symbol_info") as mock_rules_info, \
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
        from providers.quantity import clamp_notional_to_wallet_balance

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

        proc = subprocess.run(
            [sys.executable, "-c", "from providers.quantity import resolve_assets, compute_fills_cost_basis; assert resolve_assets('BTCUSDC') == ('BTC', 'USDC')"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)




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
        fake_pa = SimpleNamespace(
            get_weight_for_cash_permission_at_quant_time=lambda *_: 0.5)
        with patch.dict("sys.modules", {"priceAnalysis": fake_pa}):
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

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_resolve_assets_strips_separators_while_preserving_unknown_suffixes_iter5_s3():
    """[ITER5 #3] resolve_assets strips separators while preserving unknown suffixes."""
    # 3. resolve_assets strips separators while preserving unknown suffixes
    assert resolve_assets("HYPE/USDC") == ("HYPE", "USDC")
    assert resolve_assets("HYPE-USDC", base="HYPE") == ("HYPE", "USDC")
    assert resolve_assets("TAOUSDT") == ("TAO", "USDT")
    assert resolve_assets("TAOBRL") == ("TAOBRL", None)

def test_decide_quantity_with_nan_bool_policy_cap_and_nan_suggested_scale_iter27_s4():
    """[ITER27 #4] decide_quantity with NaN/bool policy_cap and NaN suggested_scale."""
    # 4. decide_quantity with NaN/bool policy_cap and NaN suggested_scale
    class DummyQtyProvider:
        def spent_asset(self, side, base, quote):
            return quote if side == "BUY" else base
        def free_balance(self, asset):
            return 1000.0
        def balance_cap_quantity(self, side, price, free):
            return free / price
        def policy_cap_quantity(self, *a, **k):
            return float("nan")
        def fee_cap_quantity(self, *a, **k):
            return 10.0
        def round_amount(self, sym, qty):
            return round(qty, 4)

    q_nan = decide_quantity(DummyQtyProvider(), "TAOUSD", "BUY", 100.0, 2.0, apply_policy=True)
    assert q_nan.refuse_reason == "policy_unavailable"
    assert q_nan.final_qty == 0.0

    class DummyQtyProviderOk(DummyQtyProvider):
        def policy_cap_quantity(self, *a, **k):
            return 5.0

    q_scale_nan = decide_quantity(
        DummyQtyProviderOk(),
        "TAOUSD",
        "BUY",
        100.0,
        2.0,
        apply_policy=True,
        regime_context=SimpleNamespace(suggested_scale=float("nan")),
    )
    assert q_scale_nan.refuse_reason is None
    assert q_scale_nan.final_qty == 2.0
    assert q_scale_nan.scale_applied is False

def test_candle_interval_rejects_bool_iter29_s1():
    """[ITER29 #1] candle_interval rejects bool."""
    from providers.strategy_executor import candle_interval
    # 1. candle_interval rejects bool
    with pytest.raises(ProviderError):
        candle_interval(True)
    with pytest.raises(ProviderError):
        candle_interval(False)
    assert candle_interval(1) == "1m"

def test_compute_weight_capped_qty_must_accept_required_qty_float_inf_auto_quantity_iter35_s1():
    """[ITER35 #1] compute_weight_capped_qty must accept required_qty=float('inf') (auto-quantity from decide_quantity)."""
    # 1. compute_weight_capped_qty must accept required_qty=float('inf') (auto-quantity from decide_quantity)
    capped_inf = _order_guard.compute_weight_capped_qty(
        weight=0.5,
        price=100.0,
        required_qty=float("inf"),
        available_qty=10.0,
        traded_24h_value=0.0,
    )
    assert math.isclose(capped_inf, 5.0)

    # required_qty=NaN or bool must still return 0.0
    assert _order_guard.compute_weight_capped_qty(0.5, 100.0, float("nan"), 10.0, 0.0) == 0.0
    assert _order_guard.compute_weight_capped_qty(0.5, 100.0, True, 10.0, 0.0) == 0.0

def test_fee_cap_quantity_nan_bool_safety_iter35_s2():
    """[ITER35 #2] fee_cap_quantity NaN/bool safety."""
    # 2. fee_cap_quantity NaN/bool safety
    assert fee_cap_quantity(True, 0.001) == 0.0
    assert fee_cap_quantity(float("nan"), 0.001) == 0.0
    assert math.isclose(fee_cap_quantity(10.0, True), 10.0)
    assert math.isclose(fee_cap_quantity(10.0, float("nan")), 10.0)

def test_marketdataprovider_round_quantity_round_price_order_filter_refusal_and_last_opposite_fill_iter38_s4():
    """[ITER38 #4] MarketDataProvider round_quantity, round_price, order_filter_refusal, and last_opposite_fill."""
    # 4. MarketDataProvider round_quantity, round_price, order_filter_refusal, and last_opposite_fill
    class _DummyProv(MarketDataProvider):
        @property
        def name(self) -> str:
            return "dummy"

        def get_current_price(self, symbol: str):
            return 100.0

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def pair_precision(self, symbol: str):
            return PairPrecision(price_decimals=2, volume_decimals=3, order_min=0.01)

        def get_orders(self, symbol: str, side, since_s: float):
            return [
                {"timestamp": 1000, "price": 50000.0, "qty": 1.0},
                {"timestamp": 2000, "price": 0.0, "qty": 1.0},
                "malformed",
            ]

    dp = _DummyProv()
    assert dp.round_quantity("BTCUSDC", True) == 0.0
    assert dp.round_quantity("BTCUSDC", float("nan")) == 0.0
    assert dp.round_price("BTCUSDC", True) == 0.0
    assert dp.round_price("BTCUSDC", float("nan")) == 0.0
    assert dp.order_filter_refusal("BTCUSDC", "BUY", 100.0, True) == "qty_zero_or_negative"
    assert dp.last_opposite_fill("BTCUSDC", "BUY") == pytest.approx(50000.0)

def test_submissionoutcome_rejects_boolean_order_id_iter39_s2():
    """[ITER39 #2] SubmissionOutcome rejects boolean order_id."""
    # 2. SubmissionOutcome rejects boolean order_id
    with pytest.raises(ValueError):
        SubmissionOutcome("accepted", order_id=True)

def test_decide_quantity_ignores_boolean_negative_round_fn_return_values_iter39_s4():
    """[ITER39 #4] decide_quantity ignores boolean/negative round_fn return values."""
    # 4. decide_quantity ignores boolean/negative round_fn return values
    prov_q = SimpleNamespace(
        name="dummy",
        free_balance=lambda asset: 1000.0,
        policy_cap_quantity=lambda sym, side, px, req, avail, **kw: avail,
        fee_cap_quantity=lambda sym, side, px, avail: avail,
        round_amount=lambda sym, q: True,
        order_filter_refusal=lambda *a, **kw: None,
    )
    qd = decide_quantity(prov_q, "BTCUSDC", "BUY", 100.0, 2.5, scale=0.5)
    assert qd.final_qty == pytest.approx(1.25)

def test_replaymarketdataprovider_and_simclock_reject_bool_nan_inputs_iter43_s4():
    """[ITER43 #4] ReplayMarketDataProvider and SimClock reject bool/NaN inputs."""
    from providers.replay_provider import SimClock
    # 4. ReplayMarketDataProvider and SimClock reject bool/NaN inputs
    with pytest.raises(ValueError):
        SimClock(True)
    with pytest.raises(ValueError):
        SimClock(float("nan"))
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0), (1060.0, 51000.0)]})
    assert rp.advance("BTCUSDC", True) is None
    assert rp.advance("BTCUSDC", 1) == 50000.0
    rp.place_order("BTCUSDC", "BUY", 50000.0, 0.1)
    assert rp.get_orders("BTCUSDC", "BUY", float("nan")) == []
    assert rp.get_orders("BTCUSDC", "BUY", True) == []
    assert rp.has_orders_since("BTCUSDC", float("nan")) is False
    assert rp.get_price_history("BTCUSDC", float("nan")) == []

def test_aggregate_fills_by_order_id_does_not_dilute_vwap_when_a_partial_iter59_s1():
    """[ITER59 #1] aggregate_fills_by_order_id does not dilute VWAP when a partial fill has price=0."""
    # 1. aggregate_fills_by_order_id does not dilute VWAP when a partial fill has price=0
    fills = [
        {"orderId": "ord-1", "side": "BUY", "price": 100.0, "qty": 2.0, "timestamp": 1000},
        {"orderId": "ord-1", "side": "BUY", "price": 0.0, "qty": 2.0, "timestamp": 2000},
    ]
    agg = aggregate_fills_by_order_id(fills)
    assert len(agg) == 1
    assert agg[0]["price"] == pytest.approx(100.0)
    assert agg[0]["qty"] == pytest.approx(4.0)

def test_marketdataprovider_round_quantity_and_round_price_clamp_negative_inputs_to_iter59_s2():
    """[ITER59 #2] MarketDataProvider.round_quantity and round_price clamp negative inputs to 0.0 and clamp large decimals."""
    # 2. MarketDataProvider.round_quantity and round_price clamp negative inputs to 0.0 and clamp large decimals
    class _DummyProvider(MarketDataProvider):
        @property
        def name(self) -> str:
            return "dummy"

        def get_current_price(self, symbol: str):
            return 1.0

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def pair_precision(self, symbol: str):
            return PairPrecision(price_decimals=50, volume_decimals=50)

    dp = _DummyProvider()
    assert dp.round_quantity("BTCUSDC", -1.5) == 0.0
    assert dp.round_price("BTCUSDC", -10.0) == 0.0
    assert dp.round_quantity("BTCUSDC", 1.23456) > 0.0

def test_replaymarketdataprovider_place_order_safely_returns_none_on_non_numeric_iter61_s3(tmp_path):
    """[ITER61 #3] ReplayMarketDataProvider.place_order safely returns None on non-numeric price/qty, and load_price_series skips ts <= 0."""
    import json
    # 3. ReplayMarketDataProvider.place_order safely returns None on non-numeric price/qty, and load_price_series skips ts <= 0
    jsonl_file = tmp_path / "prices.jsonl"
    jsonl_file.write_text(
        json.dumps({"s": "BTCUSDC", "i": [-1000, 50000.0]}) + "\n"
        + json.dumps({"s": "BTCUSDC", "i": [2000, 51000.0]}) + "\n",
        encoding="utf-8",
    )
    series = load_price_series(str(jsonl_file), "BTCUSDC")
    assert series == [(2.0, 51000.0)]
    rp = ReplayMarketDataProvider({"BTCUSDC": series})
    rp.advance("BTCUSDC")
    assert rp.place_order("BTCUSDC", "BUY", "not-a-price", 1.0) is None
    assert rp.place_order("BTCUSDC", "BUY", 51000.0, None) is None

def test_decide_quantity_must_never_downscale_sell_orders_even_when_iter85_s1():
    """[ITER85 #1] decide_quantity must never downscale SELL orders even when explicit scale=0.5 is passed (Theme 3)."""
    # 1. decide_quantity must never downscale SELL orders even when explicit scale=0.5 is passed (Theme 3)
    class _DummyProvider:
        def free_balance(self, _asset):
            return 10.0

        def policy_cap_quantity(self, *_a, **_k):
            return 10.0

        def fee_cap_quantity(self, _sym, _side, _px, bal_cap):
            return bal_cap

    prov = _DummyProvider()
    sell_dec = decide_quantity(prov, "BTCUSDC", "SELL", 65000.0, 2.0, scale=0.5)
    assert sell_dec.final_qty == 2.0
    assert sell_dec.scale_applied is False
    assert sell_dec.applied_scale == 1.0

    buy_dec = decide_quantity(prov, "BTCUSDC", "BUY", 10.0, 1.0, scale=0.5)
    assert buy_dec.final_qty == 0.5
    assert buy_dec.scale_applied is True
    assert buy_dec.applied_scale == 0.5

def test_providers_base_bool_env_strips_inline_comments_quotes_and_iter85_s2(monkeypatch):
    """[ITER85 #2] providers.base.bool_env strips inline comments/quotes and supports 1/yes/on and default fallback."""
    from providers.base import bool_env
    # 2. providers.base.bool_env strips inline comments/quotes and supports 1/yes/on and default fallback
    monkeypatch.setenv("TEST_BOOL_ENV_FLAG", "1 # inline comment")
    assert bool_env("TEST_BOOL_ENV_FLAG", default=False) is True
    monkeypatch.setenv("TEST_BOOL_ENV_FLAG", "'false'")
    assert bool_env("TEST_BOOL_ENV_FLAG", default=True) is False
    monkeypatch.setenv("TEST_BOOL_ENV_FLAG", "invalid_token")
    assert bool_env("TEST_BOOL_ENV_FLAG", default=True) is True

def test_pairprecision_rejects_non_finite_decimals_non_numeric_order_min_iter86_s1():
    """[ITER86 #1] PairPrecision rejects non-finite decimals, non-numeric order_min, and boolean base_asset with ValueError."""
    # 1. PairPrecision rejects non-finite decimals, non-numeric order_min, and boolean base_asset with ValueError
    with pytest.raises(ValueError, match="invalid price_decimals"):
        PairPrecision(price_decimals=float("inf"), volume_decimals=2, order_min=0.01)
    with pytest.raises(ValueError, match="invalid order_min"):
        PairPrecision(price_decimals=2, volume_decimals=2, order_min="not_a_float")
    with pytest.raises(ValueError, match="invalid base_asset"):
        PairPrecision(price_decimals=2, volume_decimals=2, order_min=0.01, base_asset=True)

def test_extract_order_side_string_isbuyer_false_vs_true_iter96_s1():
    """[ITER96 #1] extract_order_side string isBuyer ('false'/'0' vs 'true'/'1')."""
    from providers.order_ohlc_utils import extract_order_side
    # 1. extract_order_side string isBuyer ("false"/"0" vs "true"/"1")
    assert extract_order_side({"isBuyer": "false"}) == "SELL"
    assert extract_order_side({"isBuyer": "0"}) == "SELL"
    assert extract_order_side({"isBuyer": "true"}) == "BUY"
    assert extract_order_side({"isBuyer": "1"}) == "BUY"

def test_aggregate_fills_by_order_id_none_invalid_rows_and_bool_invalid_side_iter96_s2():
    """[ITER96 #2] aggregate_fills_by_order_id None/invalid rows and bool/invalid side."""
    # 2. aggregate_fills_by_order_id None/invalid rows and bool/invalid side
    assert aggregate_fills_by_order_id(None) == []
    assert aggregate_fills_by_order_id([{"orderId": "1", "side": "BUY", "price": 100.0, "qty": 1.0}], side=True) == []
    assert aggregate_fills_by_order_id([{"orderId": "1", "side": "BUY", "price": 100.0, "qty": 1.0}], side="INVALID") == []

def test_providers_strategy_executor_bool_rejection_in_submissionrefused_submissionoutcome_orderstatus_iter105_s3():
    """[ITER105 #3] providers.strategy_executor: bool rejection in SubmissionRefused, SubmissionOutcome, OrderStatus."""
    from providers.strategy_executor import OrderStatus, SubmissionRefused
    # 3. providers.strategy_executor: bool rejection in SubmissionRefused, SubmissionOutcome, OrderStatus
    with pytest.raises(ValueError):
        SubmissionRefused(True)
    with pytest.raises(ValueError):
        SubmissionOutcome("refused", reason=True)
    with pytest.raises(ValueError):
        OrderStatus("open", 0.0, 0.0, 0.0, venue_status=True)

def test_providers_quantity_remaining_average_cost_balance_cap_quantity_decide_quantity_iter106_s1():
    """[ITER106 #1] providers.quantity: remaining_average_cost, balance_cap_quantity, decide_quantity."""
    from providers import quantity as pq
    # 1. providers.quantity: remaining_average_cost, balance_cap_quantity, decide_quantity
    assert pq.remaining_average_cost(True, 1.0) is None
    assert pq.remaining_average_cost([{"id": True, "timestamp": 100.0, "side": "BUY", "qty": 1.0, "price": 100.0, "base_fee": 0.0, "quote_fee": 0.0}], 1.0) is None
    avg_cost = pq.remaining_average_cost(
        [{"id": "1", "timestamp": 100.0, "side": " b ", "qty": 1.0, "price": 100.0, "base_fee": 0.0, "quote_fee": 0.0}],
        1.0,
    )
    assert avg_cost == pytest.approx(100.0)
    assert pq.balance_cap_quantity(lambda _a: "bad_bal", "BTCUSDC", "BUY", 50000.0) == (None, "USDC")

    dummy_provider = SimpleNamespace(
        free_balance=lambda _a: 1000.0,
        policy_cap_quantity=lambda _sym, side, _px, req, _bal, **kw: req if side == "BUY" else 0.0,
        fee_cap_quantity=lambda _sym, _side, _px, bal: bal,
    )
    dec_zero_scale = pq.decide_quantity(
        dummy_provider, "BTCUSDC", " b ", 100.0, 2.0, scale=0.0
    )
    assert dec_zero_scale.final_qty == 0.0
    assert dec_zero_scale.scale_applied is True
    assert dec_zero_scale.applied_scale == 0.0
    assert dec_zero_scale.refuse_reason == "qty_zero_after_policy"

def test_binanceorderrules_from_symbol_info_rejects_non_iterable_filters_and_non_iter108_s1():
    """[ITER108 #1] BinanceOrderRules.from_symbol_info rejects non-iterable filters and non-str base/quote."""
    from providers.quantity import BinanceFilterError, BinanceOrderRules
    # 1. BinanceOrderRules.from_symbol_info rejects non-iterable filters and non-str base/quote
    with pytest.raises(BinanceFilterError):
        BinanceOrderRules.from_symbol_info({"filters": True})
    with pytest.raises(BinanceFilterError):
        BinanceOrderRules.from_symbol_info(
            {
                "baseAsset": 123,
                "quoteAsset": "USDC",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                ],
            }
        )

def test_botcore_providers_env_value_bool_env_extract_order_id_extract_order_fee_aggregate_fills_by_order_id_iter109_s1(tmp_path):
    """[ITER109 #1] botcore/providers: env_value, bool_env, extract_order_id, extract_order_fee, aggregate_fills_by_order_id."""
    # 1. botcore/providers: env_value, bool_env, extract_order_id, extract_order_fee, aggregate_fills_by_order_id
    assert botcore.env_value(True, "KEY") is None
    assert botcore.env_value(str(tmp_path), True) is None
    assert pbase.bool_env(True, default=True) is True
    assert pbase.bool_env("", default=False) is False
    assert pbase.extract_order_id({"orderId": {}}) is None
    assert pbase.extract_order_id({"orderId": set()}) is None
    assert pbase.extract_order_id(object()) is None

    fee_res = oou.extract_order_fee(
        {"commission": "0.1", "commissionAsset": [1, 2]},
        base_asset=True,
        quote_asset=True,
        fill_price=100.0,
    )
    assert fee_res is not None
    assert fee_res["fee_asset"] == ""
    assert pbase.aggregate_fills_by_order_id(12345) == []

def test_providers_replay_provider_load_price_series_rejects_non_str_path_or_iter109_s3(tmp_path):
    """[ITER109 #3] providers.replay_provider: load_price_series rejects non-str path or symbol."""
    from providers import replay_provider as rprov
    # 3. providers.replay_provider: load_price_series rejects non-str path or symbol
    assert rprov.load_price_series(True, "BTCUSDC") == []
    assert rprov.load_price_series(str(tmp_path), True) == []

def test_extract_order_id_rejects_nan_inf_none_null_non_integer_iter122_s1():
    """[ITER122 #1] extract_order_id rejects 'nan', 'inf', 'none', 'null', non-integer floats, and normalizes integer floats."""
    # 1. extract_order_id rejects "nan", "inf", "none", "null", non-integer floats, and normalizes integer floats
    assert extract_order_id({"orderId": "nan"}) is None
    assert extract_order_id({"orderId": "INF"}) is None
    assert extract_order_id({"orderId": "null"}) is None
    assert extract_order_id({"orderId": 123.5}) is None
    assert extract_order_id({"orderId": 12345.0}) == "12345"

def test_require_int_order_id_preserves_bit_integer_precision_iter122_s2():
    """[ITER122 #2] require_int_order_id preserves 64-bit integer precision (> 2**53)."""
    # 2. require_int_order_id preserves 64-bit integer precision (> 2**53)
    from providers.order_ohlc_utils import require_int_order_id

    big_oid = 9007199254740993  # 2**53 + 1
    assert require_int_order_id(big_oid, "order_status") == big_oid
    assert require_int_order_id(str(big_oid), "order_status") == big_oid
    with pytest.raises(ProviderError):
        require_int_order_id(123.5, "order_status")
    with pytest.raises(ProviderError):
        require_int_order_id("nan", "order_status")

def test_submissionoutcome_normalizes_integer_float_order_id_and_rejects_nan_iter123_s1():
    """[ITER123 #1] SubmissionOutcome normalizes integer float order_id and rejects NaN/'nan'/'null'/non-integer floats."""
    # 1. SubmissionOutcome normalizes integer float order_id and rejects NaN/"nan"/"null"/non-integer floats
    outcome = SubmissionOutcome("accepted", order_id=98765.0)
    assert outcome.order_id == "98765"
    for bad_oid in (float("nan"), float("inf"), 12.5, "nan", "null", "none", {"id": 1}):
        with pytest.raises(ValueError):
            SubmissionOutcome("accepted", order_id=bad_oid)

def test_replaymarketdataprovider_sanitizes_fee_pct_and_non_finite_cur_px_in_iter127_s2():
    """[ITER127 #2] ReplayMarketDataProvider sanitizes fee_pct and non-finite cur_px in place_order."""
    # 2. ReplayMarketDataProvider sanitizes fee_pct and non-finite cur_px in place_order
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, float("nan"))]}, fee_pct=True)
    assert rp._fee_pct == 0.1
    rp.advance("BTCUSDC")
    ord_res = rp.place_order("BTCUSDC", "BUY", 100.0, 1.0)
    assert ord_res is not None
    assert rp.position("BTCUSDC") == (1.0, 100.0)

def test_replay_provider_load_price_series_directory_safety_and_supports_symbol_unhashable_safety_iter151_s4(tmp_path):
    """[ITER151 #4] replay_provider: load_price_series directory safety and supports_symbol unhashable safety."""
    # 4. replay_provider: load_price_series directory safety and supports_symbol unhashable safety
    assert load_price_series(str(tmp_path), "BTCUSDC") == []
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1700000000.0, 60000.0)]})
    assert rp.supports_symbol("BTCUSDC") is True
    assert rp.supports_symbol(["BTCUSDC"]) is False  # type: ignore[arg-type]
    assert rp.supports_symbol(True) is False  # type: ignore[arg-type]

def test_providers_quantity_py_remaining_average_cost_fill_id_normalization_sentinel_iter154_s2():
    """[ITER154 #2] providers/quantity.py: remaining_average_cost fill ID normalization & sentinel rejection."""
    from providers.quantity import remaining_average_cost
    # 2. providers/quantity.py: remaining_average_cost fill ID normalization & sentinel rejection
    # Duplicate id=1001.0 vs "1001" is detected and rejected (returns None)
    assert remaining_average_cost(
        [
            {"id": 1001.0, "side": "BUY", "price": 10.0, "qty": 2.0, "timestamp": 1000, "base_fee": 0.0, "quote_fee": 0.0},
            {"id": "1001", "side": "BUY", "price": 10.0, "qty": 2.0, "timestamp": 2000, "base_fee": 0.0, "quote_fee": 0.0},
        ],
        4.0,
    ) is None
    # Sentinel id="0" is rejected (returns None)
    assert remaining_average_cost(
        [
            {"id": "0", "side": "BUY", "price": 20.0, "qty": 2.0, "timestamp": 2000, "base_fee": 0.0, "quote_fee": 0.0},
        ],
        2.0,
    ) is None
    # Valid id=1001.0 succeeds
    assert remaining_average_cost(
        [
            {"id": 1001.0, "side": "BUY", "price": 10.0, "qty": 2.0, "timestamp": 1000, "base_fee": 0.0, "quote_fee": 0.0},
        ],
        2.0,
    ) == 10.0

def test_providers_quantity_py_decide_quantity_string_boolean_parsing_iter154_s3():
    """[ITER154 #3] providers/quantity.py: decide_quantity string boolean parsing."""
    # 3. providers/quantity.py: decide_quantity string boolean parsing
    captured = {}

    class DummyProvider:
        def free_balance(self, asset):
            return 1000.0

        def policy_cap_quantity(self, *args, **kwargs):
            captured["policy_called"] = True
            return 1.0

        def fee_cap_quantity(self, symbol, side, price, balance_cap):
            return balance_cap

        def order_filter_refusal(self, symbol, side, price, qty, *, market=False, enforce_business_minimum=True):
            captured["market"] = market
            captured["enforce_business_minimum"] = enforce_business_minimum
            return None

    dec = decide_quantity(
        DummyProvider(),
        "BTCUSDC",
        "BUY",
        100.0,
        2.0,
        apply_policy="false",  # type: ignore[arg-type]
        market="false",  # type: ignore[arg-type]
        enforce_business_minimum="false",  # type: ignore[arg-type]
    )
    assert dec.final_qty == 2.0
    assert "policy_called" not in captured
    assert captured["market"] is False
    assert captured["enforce_business_minimum"] is False

def test_providers_base_py_extract_order_id_aggregate_fills_by_order_id_normalization_sentinel_rejection_iter154_s5():
    """[ITER154 #5] providers/base.py: extract_order_id & aggregate_fills_by_order_id .0 normalization & sentinel rejection."""
    # 5. providers/base.py: extract_order_id & aggregate_fills_by_order_id .0 normalization & sentinel rejection
    assert extract_order_id({"orderId": "12345.0"}) == "12345"
    agg = aggregate_fills_by_order_id([
        {"orderId": 555.0, "side": "BUY", "price": 100.0, "qty": 1.0, "timestamp": 1000},
        {"orderId": "555", "side": "BUY", "price": 200.0, "qty": 1.0, "timestamp": 2000},
        {"orderId": "0", "side": "BUY", "price": 300.0, "qty": 1.0, "timestamp": 3000},
        {"orderId": "0", "side": "BUY", "price": 400.0, "qty": 1.0, "timestamp": 4000},
    ])
    assert len(agg) == 3
    assert agg[0]["price"] == 150.0 and agg[0]["qty"] == 2.0

def test_botcore_providers_env_value_quoted_value_with_trailing_comment_iter164_s1(tmp_path):
    """[ITER164 #1] botcore/providers: env_value quoted value with trailing comment, bool_env default='false', extract_order_fee(require_explicit_asset='false')."""
    # 1. botcore/providers: env_value quoted value with trailing comment, bool_env default="false", extract_order_fee(require_explicit_asset="false")
    (tmp_path / ".env").write_text('QUOTED_KEY="my_secret#1" # trailing comment\nFLAG_KEY="true" # comment\n', encoding="utf-8")
    assert botcore.env_value(str(tmp_path), "QUOTED_KEY") == "my_secret#1"
    assert pbase.bool_env("FLAG_KEY", folder=str(tmp_path), default="false") is True
    assert pbase.bool_env("MISSING_KEY", folder=str(tmp_path), default="false") is False
    assert pbase.bool_env("MISSING_KEY", folder=str(tmp_path), default=float("nan")) is False
    fee_res = oou.extract_order_fee({"commission": 0.5, "price": 100.0}, quote_asset="USDC", require_explicit_asset="false")
    assert fee_res is not None and fee_res["fee_asset"] == "USDC"
    assert oou.extract_order_fee({"commission": 0.5, "price": 100.0}, quote_asset="USDC", require_explicit_asset="true") is None

def test_providers_quantity_resolve_assets_with_usdt_fdusd_gbp_and_iter168_s2():
    """[ITER168 #2] providers.quantity.resolve_assets with USDT/FDUSD/GBP and _parse_bool_flag."""
    from providers import quantity as qty_mod
    # 2. providers.quantity.resolve_assets with USDT/FDUSD/GBP and _parse_bool_flag
    assert qty_mod.resolve_assets("BTCUSDT") == ("BTC", "USDT")
    assert qty_mod.resolve_assets("ETHFDUSD") == ("ETH", "FDUSD")
    assert qty_mod.resolve_assets("SOLGBP") == ("SOL", "GBP")
    assert qty_mod.resolve_assets("BTCXYZ") == ("BTCXYZ", None)
    assert qty_mod._parse_bool_flag(2, default=False) is True
    assert qty_mod._parse_bool_flag("y", default=False) is True
    assert qty_mod._parse_bool_flag("n", default=True) is False
    assert qty_mod.clean_order_id_str(frozenset({"1"})) == ""

def test_providers_base_boolean_and_side_extraction_iter169_s1():
    """[ITER169 #1] providers.base boolean and side extraction."""
    # 1. providers.base boolean and side extraction
    assert pbase._parse_bool_flag("", default=True) is True
    assert pbase._parse_bool_flag("y", default=False) is True
    assert pbase._parse_bool_flag("n", default=True) is False
    assert oou.extract_order_side({"type": "A"}) == "SELL"
    assert oou.extract_order_side({"isBuyer": "y"}) == "BUY"
    assert oou.extract_order_side({"isBuyer": "n"}) == "SELL"

def test_submissionoutcome_and_orderstatus_normalization_iter170_s3():
    """[ITER170 #3] SubmissionOutcome and OrderStatus normalization."""
    from providers import strategy_executor as sexec
    # 3. SubmissionOutcome and OrderStatus normalization
    so = sexec.SubmissionOutcome(" ACCEPTED ", order_id=42, reason=" ok ")
    assert so.state == "accepted" and so.order_id == "42" and so.reason == "ok"

    st = sexec.OrderStatus(" CANCELLED ", "1.5", "150.0", "0.15", venue_status=" user_canceled ")
    assert st.status == "canceled"
    assert st.filled_qty == pytest.approx(1.5)
    assert st.venue_status == "USER_CANCELED"
    with pytest.raises(ValueError):
        sexec.OrderStatus("open", None, 0.0, 0.0)

def test_clean_order_id_strictly_validates_positive_integers_integer_floats_and_iter185_s3():
    """[ITER185 #3] clean_order_id strictly validates positive integers, integer floats, and valid numeric strings."""
    from providers.order_ohlc_utils import clean_order_id
    # 3. clean_order_id strictly validates positive integers, integer floats, and valid numeric strings
    assert clean_order_id(12345) == "12345"
    assert clean_order_id(12345.0) == "12345"
    assert clean_order_id(0) is None
    assert clean_order_id(-1) is None
    assert clean_order_id(123.45) is None
    assert clean_order_id(float("nan")) is None
    assert clean_order_id(float("inf")) is None
    assert clean_order_id(True) is None
    assert clean_order_id(False) is None
    assert clean_order_id({"id": 1}) is None
    assert clean_order_id([1]) is None
