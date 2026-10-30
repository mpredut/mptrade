import unittest
from types import SimpleNamespace
from unittest.mock import patch

import order_guard
from providers.base import MarketDataProvider
from providers.quantity import (
    balance_cap_quantity,
    decide_quantity,
    fee_cap_quantity,
    resolve_assets,
)


class Provider(MarketDataProvider):
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
            Provider({"TAO": 1.0}, policy=0.6),
            "TAOUSDC", "SELL", 225, 2.0)
        self.assertEqual(decision.requested_qty, 2.0)
        self.assertEqual(decision.balance_cap, 1.0)
        self.assertEqual(decision.policy_cap, 0.6)
        self.assertEqual(decision.final_qty, 0.6)

    def test_none_means_maximum_allowed_by_balance_and_policy(self):
        decision = decide_quantity(
            Provider({"USDC": 500.0}, policy=3.0),
            "TAOUSDC", "BUY", 100.0, None)
        self.assertEqual(decision.balance_cap, 5.0)
        self.assertEqual(decision.policy_cap, 3.0)
        self.assertEqual(decision.final_qty, 3.0)

    def test_risk_exit_skips_policy_but_keeps_balance_and_fee_caps(self):
        provider = Provider({"TAO": 0.4}, policy=0.01)
        provider.fee_cap_quantity = lambda *_args: 0.39
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 2.0,
            apply_policy=False)
        self.assertEqual(decision.final_qty, 0.39)

    def test_provider_filter_refuses_before_submission(self):
        provider = Provider({"TAO": 1.0})
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
        provider = Provider({"TAO": 1.0}, policy=0.0)
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 0.5,
            apply_policy=False, market=True)
        self.assertEqual(decision.final_qty, 0.5)
        self.assertTrue(provider.filter_calls[0][-2])

    def test_business_minimum_context_reaches_provider_filter(self):
        provider = Provider({"TAO": 1.0})
        decision = decide_quantity(
            provider, "TAOUSDC", "SELL", 100.0, 0.5,
            enforce_business_minimum=False)
        self.assertEqual(decision.final_qty, 0.5)
        self.assertFalse(provider.filter_calls[0][-1])

    def test_filter_failure_is_not_silently_bypassed(self):
        provider = Provider({"TAO": 1.0})
        provider.order_filter_refusal = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("rules unavailable"))
        with self.assertRaisesRegex(RuntimeError, "rules unavailable"):
            decide_quantity(provider, "TAOUSDC", "SELL", 100.0, 0.5)

    def test_none_is_error_but_zero_is_real_insufficient_balance(self):
        unavailable = decide_quantity(
            Provider({"USDC": None}), "TAOUSDC", "BUY", 225, 1)
        empty = decide_quantity(
            Provider({"USDC": 0}), "TAOUSDC", "BUY", 225, 1)
        self.assertEqual(unavailable.refuse_reason, "balance_unavailable")
        self.assertEqual(empty.refuse_reason, "insufficient_funds")

    def test_fee_cap_operates_on_base_quantity(self):
        self.assertAlmostEqual(fee_cap_quantity(2.0, 0.01), 2.0 / 1.01)

    def test_weight_limit_uses_precomputed_balance_without_refetch(self):
        provider = Provider({"USDC": 500})
        provider.free_balance = lambda _asset: self.fail("the balance must not be re-read")
        fake_pa = SimpleNamespace(
            get_weight_for_cash_permission_at_quant_time=lambda *_: 0.5)
        with patch.dict("sys.modules", {"priceAnalysis": fake_pa}):
            qty = order_guard.weight_limit(
                provider, "TAOUSDC", "BUY", 100.0, 5.0,
                available_qty=5.0)
        self.assertEqual(qty, 2.5)


if __name__ == "__main__":
    unittest.main()
