"""Tests for extracted Binance placement mechanics and provider hooks.

The network and dispatch functions are fully mocked. Coverage includes the
guards_internally=False route through the provider-neutral Instrument pipeline.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

mock_bapi = MagicMock()
sys.modules.setdefault("bapi", mock_bapi)
sys.modules.setdefault("bapi_trades", MagicMock())
sys.modules.setdefault("bapi_allorders", MagicMock())

from binance_api import bapi_placeorder as po
from binance_api import bapi
from providers.market_api import BinanceProvider
from providers.strategy_executor import (
    ProviderError,
    SubmissionRefused,
    capture_submission,
)

SYMBOL = "BTCUSDC"


class TestAdjustPriceAndCancelOpposite(unittest.TestCase):
    def test_adjust_and_cancel_success(self):
        with self.subTest(msg="buy nudges down and cancels low sells"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_open_orders", return_value={"1": {"price": 90.0}, "2": {"price": 110.0}}) as goo, \
                 patch.object(po.api, "cancel_order", return_value=True) as cancel:
                out = po.adjust_price_and_cancel_opposite("BUY", SYMBOL, 105.0, cancel_opposite=True)
            self.assertEqual(out, round(min(105.0, 100.0) * 0.999, 0))
            goo.assert_called_once_with("SELL", SYMBOL, strict=True)
            cancel.assert_called_once_with(SYMBOL, "1")

        with self.subTest(msg="sell nudges up and cancels high buys"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_open_orders", return_value={"1": {"price": 110.0}, "2": {"price": 90.0}}), \
                 patch.object(po.api, "cancel_order", return_value=True) as cancel:
                out = po.adjust_price_and_cancel_opposite("SELL", SYMBOL, 95.0, cancel_opposite=True)
            self.assertEqual(out, round(max(95.0, 100.0) * 1.001, 0))
            cancel.assert_called_once_with(SYMBOL, "1")

        with self.subTest(msg="no cancel when disabled"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_open_orders") as goo, \
                 patch.object(po.api, "cancel_order") as cancel:
                po.adjust_price_and_cancel_opposite("BUY", SYMBOL, 105.0, cancel_opposite=False)
            goo.assert_not_called()
            cancel.assert_not_called()

    def test_cancel_or_discovery_failure(self):
        with self.subTest(msg="cancel failure refuses replacement"):
            with patch.object(po.api, "get_open_orders", return_value={"1": {"price": 90.0}}), \
                 patch.object(po.api, "cancel_order", return_value=False):
                with self.assertRaisesRegex(SubmissionRefused, "opposing_cancel_unconfirmed"):
                    po.cancel_opposite_orders("BUY", SYMBOL, 105.0)

        with self.subTest(msg="discovery failure refuses replacement"):
            with patch.object(po.api, "get_open_orders", side_effect=RuntimeError("venue unavailable")):
                with self.assertRaisesRegex(SubmissionRefused, "opposing_order_discovery_unavailable"):
                    po.cancel_opposite_orders("BUY", SYMBOL, 105.0)


class TestOpenOrderRemainingQuantity(unittest.TestCase):
    def test_open_order_remaining_quantity(self):
        with self.subTest(msg="preserves original and exposes unfilled quantity"):
            native = [{
                "orderId": 7,
                "side": "SELL",
                "price": "100",
                "origQty": "2",
                "executedQty": "0.75",
                "time": 1_000,
            }]
            fake_client = MagicMock()
            fake_client.get_open_orders.return_value = native
            with patch.object(bapi, "client", fake_client):
                order = bapi.get_open_orders("SELL", SYMBOL)[7]

            self.assertEqual(order["quantity"], 2.0)
            self.assertEqual(order["executedQty"], 0.75)
            self.assertEqual(order["remainingQty"], 1.25)

        with self.subTest(msg="strict discovery propagates api failure"):
            fake_client = MagicMock()
            fake_client.get_open_orders.side_effect = RuntimeError("venue unavailable")
            with patch.object(bapi, "client", fake_client):
                with self.assertRaisesRegex(RuntimeError, "venue unavailable"):
                    bapi.get_open_orders("SELL", SYMBOL, strict=True)


class TestPlaceOrderMechanics(unittest.TestCase):
    def setUp(self):
        fake_client = MagicMock()
        fake_client.get_symbol_info.return_value = {
            "baseAsset": "BTC", "quoteAsset": "USDC", "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10", "applyToMarket": True},
            ],
        }
        self.client_patch = patch.object(po, "client", fake_client)
        self.client_patch.start()

    def tearDown(self):
        self.client_patch.stop()

    def test_place_order_success_cases(self):
        with self.subTest(msg="buy dispatches limit"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=1000.0), \
                 patch.object(po, "_submit_binance_order", return_value={"orderId": 42}) as submit:
                order = po.place_order_mechanics("BUY", SYMBOL, 100.0, 5.0, force=False)
            self.assertEqual(order, {"orderId": 42})
            submit.assert_called_once()
            self.assertEqual(submit.call_args.args, ("BUY", SYMBOL, 5.0))
            self.assertEqual(submit.call_args.kwargs["price"], 100.0)
            self.assertFalse(submit.call_args.kwargs["market"])

        client_id = "SD_0123456789abcdef0123456789abcdef"
        with self.subTest(msg="client order id reaches limit dispatch"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=1000.0), \
                 patch.object(po, "_submit_binance_order", return_value={"orderId": 42}) as submit:
                po.place_order_mechanics("BUY", SYMBOL, 100.0, 5.0, client_order_id=client_id)
            submit.assert_called_once()
            self.assertEqual(submit.call_args.kwargs["client_order_id"], client_id)

        with self.subTest(msg="legacy positional client order id keeps its slot"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=1000.0), \
                 patch.object(po, "_submit_binance_order", return_value={"orderId": 42}) as submit:
                po.place_order_mechanics("BUY", SYMBOL, 100.0, 5.0, False, client_id)
            self.assertEqual(submit.call_args.kwargs["client_order_id"], client_id)

        with self.subTest(msg="sell market when force"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=10.0), \
                 patch.object(po, "_submit_binance_order", return_value={"orderId": 7}) as submit:
                order = po.place_order_mechanics("SELL", SYMBOL, 100.0, 5.0, force=True)
            self.assertEqual(order, {"orderId": 7})
            submit.assert_called_once()
            self.assertTrue(submit.call_args.kwargs["market"])

    def test_place_order_refusals_and_skips(self):
        with self.subTest(msg="min notional rejected before submit with typed reason"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=1000.0), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 100.0), \
                 patch.object(po, "place_BUY_order", return_value={"orderId": 1}) as pbuy:
                with self.assertRaisesRegex(SubmissionRefused, "below_min_notional"):
                    po.place_order_mechanics("BUY", SYMBOL, 100.0, 0.5, force=False)
            pbuy.assert_not_called()

        with self.subTest(msg="protective market skips only the business minimum"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=10.0), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 100.0), \
                 patch.object(po, "_submit_binance_order", return_value={"orderId": 8}) as submit:
                order = po.place_order_mechanics(
                    "SELL", SYMBOL, 100.0, 0.2, force=True, enforce_business_minimum=False)
                with self.assertRaisesRegex(SubmissionRefused, "below_min_notional"):
                    po.place_order_mechanics(
                        "SELL", SYMBOL, 100.0, 0.05, force=True, enforce_business_minimum=False)
            self.assertEqual(order, {"orderId": 8})
            submit.assert_called_once()

        with self.subTest(msg="zero available returns none"):
            with patch.object(po.api, "get_current_price", return_value=100.0), \
                 patch.object(po.api, "get_free_balance", return_value=0.0), \
                 patch.object(po, "place_BUY_order") as pbuy:
                order = po.place_order_mechanics("BUY", SYMBOL, 100.0, 5.0)
            self.assertIsNone(order)
            pbuy.assert_not_called()


class TestBinanceProviderHooks(unittest.TestCase):
    def setUp(self):
        self.p = BinanceProvider()

    @staticmethod
    def _symbol_info(*, apply_to_market):
        return {
            "baseAsset": "BTC", "quoteAsset": "USDC", "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10",
                 "applyToMarket": apply_to_market},
            ],
        }

    def _rules_client(self, *, apply_to_market):
        module = MagicMock()
        module.get_current_price.return_value = 100.0
        module.client.get_symbol_info.return_value = self._symbol_info(
            apply_to_market=apply_to_market)
        return module

    def test_provider_hooks_misc(self):
        with self.subTest(msg="guards internally false"):
            self.assertFalse(self.p.guards_internally())

        with self.subTest(msg="adjust order price delegates"):
            with patch.object(po, "adjust_price_and_cancel_opposite", return_value=99.0) as f:
                out = self.p.adjust_order_price(SYMBOL, "BUY", 100.0, cancel_opposite=True)
            self.assertEqual(out, 99.0)
            f.assert_called_once_with("BUY", SYMBOL, 100.0, cancel_opposite=True)
            
        with self.subTest(msg="place order delegates to mechanics"):
            with patch.object(po, "place_order_mechanics", return_value={"orderId": 5}) as f:
                out = self.p.place_order(SYMBOL, "BUY", 100.0, 5.0, force=True, safeback_seconds=999, pair=None)
            self.assertEqual(out, {"orderId": 5})
            f.assert_called_once_with("BUY", SYMBOL, 100.0, 5.0, force=True, enforce_business_minimum=True)

        with self.subTest(msg="place order propagates protective minimum policy"):
            with patch.object(po, "place_order_mechanics", return_value={"orderId": 6}) as mechanics:
                out = self.p.place_order(SYMBOL, "SELL", 100.0, 0.2, force=True, enforce_business_minimum=False)
            self.assertEqual(out, {"orderId": 6})
            mechanics.assert_called_once_with("SELL", SYMBOL, 100.0, 0.2, force=True, enforce_business_minimum=False)

        with self.subTest(msg="profit guard window ref uses safeback"):
            import order_guard
            with patch.object(order_guard, "window_reference", return_value=123.0) as f:
                out = self.p.profit_guard_window_ref(SYMBOL, "BUY", 14 * 24 * 3600)
            self.assertEqual(out, 123.0)
            args = f.call_args[0]
            self.assertEqual(args[3], 14 * 24 * 3600)

    def test_order_filters(self):
        with self.subTest(msg="uses normalized limit notional"):
            module = self._rules_client(apply_to_market=False)
            with patch("providers.market_api._get_bapi", return_value=module):
                refusal = self.p.order_filter_refusal(SYMBOL, "BUY", 99.1, 0.101, market=False)
            self.assertEqual(refusal, "below_min_notional")

        with self.subTest(msg="uses sell candidate price"):
            module = self._rules_client(apply_to_market=False)
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 0):
                refusal = self.p.order_filter_refusal(SYMBOL, "SELL", 50.0, 0.1, market=False)
            self.assertIsNone(refusal)

        with self.subTest(msg="uses buy candidate price"):
            module = self._rules_client(apply_to_market=False)
            module.get_current_price.return_value = 90.0
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 0):
                refusal = self.p.order_filter_refusal(SYMBOL, "BUY", 150.0, 0.1, market=False)
            self.assertEqual(refusal, "below_min_notional")

        with self.subTest(msg="respects market applicability"):
            module = self._rules_client(apply_to_market=False)
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 0):
                refusal = self.p.order_filter_refusal(SYMBOL, "SELL", 50.0, 0.1, market=True)
            self.assertIsNone(refusal)

        with self.subTest(msg="applies enabled market minimum"):
            module = self._rules_client(apply_to_market=True)
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 0):
                refusal = self.p.order_filter_refusal(SYMBOL, "SELL", 50.0, 0.1, market=True)
            self.assertEqual(refusal, "below_min_notional")

        with self.subTest(msg="applies shared business minimum"):
            module = self._rules_client(apply_to_market=False)
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 25):
                refusal = self.p.order_filter_refusal(SYMBOL, "SELL", 100.0, 0.2, market=True)
            self.assertEqual(refusal, "below_min_notional")

        with self.subTest(msg="can skip business minimum for protective exit"):
            module = self._rules_client(apply_to_market=True)
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "PLACE_ORDER_MIN_NOTIONAL", 25):
                refusal = self.p.order_filter_refusal(SYMBOL, "SELL", 100.0, 0.2, market=True, enforce_business_minimum=False)
            self.assertIsNone(refusal)

        with self.subTest(msg="fails closed when rules are unavailable"):
            module = MagicMock()
            module.client.get_symbol_info.side_effect = RuntimeError("offline")
            with patch("providers.market_api._get_bapi", return_value=module):
                with self.assertRaisesRegex(ProviderError, "rules unavailable"):
                    self.p.order_filter_refusal(SYMBOL, "SELL", 100.0, 1.0, market=True)

    def test_audited_submits(self):
        with self.subTest(msg="maps normalization filter to refusal"):
            module = self._rules_client(apply_to_market=True)
            with patch("providers.market_api._get_bapi", return_value=module), \
                 patch.object(po, "_submit_binance_order") as submit:
                outcome = capture_submission(
                    lambda: self.p.submit_order(SYMBOL, "SELL", 0.01, price=None, market=True))
            self.assertEqual(outcome.state, "refused")
            self.assertTrue(outcome.reason.startswith("binance_filter_refused:"))
            submit.assert_not_called()

        with self.subTest(msg="keeps metadata failure non-terminal"):
            module = MagicMock()
            module.client.get_symbol_info.return_value = None
            with patch("providers.market_api._get_bapi", return_value=module):
                with self.assertRaisesRegex(ProviderError, "submit_order"):
                    self.p.submit_order(SYMBOL, "SELL", 1.0, price=None, market=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
