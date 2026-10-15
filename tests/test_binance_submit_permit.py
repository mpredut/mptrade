"""Regression tests for bounded one-shot Binance cache submit permits."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from binance_api import bapi_placeorder as placeorder
from binance_cache_health import CacheHealthStatus
from providers.strategy_executor import SubmissionRefused


class BinanceSubmitPermitTest(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.order_limit_buy.return_value = {"orderId": 11}
        self.client.order_market_buy.return_value = {"orderId": 12}
        self.client.get_symbol_info.return_value = {
            "baseAsset": "BTC",
            "quoteAsset": "USDC",
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "tickSize": "0.01",
                    "minPrice": "0.01",
                    "maxPrice": "1000000",
                },
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "0.0001",
                    "minQty": "0.0001",
                    "maxQty": "1000",
                },
            ],
        }
        self.trade_enabled = patch.object(
            placeorder.cfg, "is_trade_enabled", return_value=True)
        self.trade_enabled.start()
        self.status = CacheHealthStatus(
            True, "", 1.0, 1.0, "order-version", "trade-version")

    def tearDown(self):
        self.trade_enabled.stop()

    def _issue(self, **overrides):
        params = {
            "symbol": "BTCUSDC",
            "side": "BUY",
            "qty": 0.02,
            "price": 100.0,
            "market": False,
            "kind": "replacement",
        }
        params.update(overrides)
        with patch.object(
                placeorder, "require_account_cache_for_submit",
                return_value=self.status) as gate:
            permit = placeorder.issue_account_cache_submit_permit(
                **params, api_client=self.client)
        self.assertEqual(gate.call_count, 2)
        return permit

    def test_valid_permit_skips_second_health_read_and_is_one_shot(self):
        permit = self._issue()
        with patch.object(
                placeorder, "require_account_cache_for_submit",
                side_effect=AssertionError("a permit must not reread health")):
            result = placeorder._submit_binance_order(
                "BUY", "BTCUSDC", 0.01, price=100.0,
                permit_requested_price=100.0,
                cache_permit=permit, kind="replacement",
                api_client=self.client)
            with self.assertRaisesRegex(
                    SubmissionRefused, "account_cache_not_fresh"):
                placeorder._submit_binance_order(
                    "BUY", "BTCUSDC", 0.01, price=100.0,
                    permit_requested_price=100.0,
                    cache_permit=permit, kind="replacement",
                    api_client=self.client)

        self.assertEqual(result["orderId"], 11)
        self.client.order_limit_buy.assert_called_once()

    def test_submit_without_permit_still_requires_fresh_health(self):
        with patch.object(
                placeorder, "require_account_cache_for_submit",
                return_value=self.status) as gate:
            result = placeorder._submit_binance_order(
                "BUY", "BTCUSDC", 0.01, price=100.0,
                api_client=self.client)

        self.assertEqual(result["orderId"], 11)
        gate.assert_called_once_with()

    def test_expired_permit_refuses_before_client_call(self):
        permit = self._issue()

        with patch.object(
                placeorder.time, "monotonic",
                return_value=permit.expires_at + 0.001):
            with self.assertRaisesRegex(
                    SubmissionRefused, "account_cache_not_fresh"):
                placeorder._submit_binance_order(
                    "BUY", "BTCUSDC", 0.01, price=100.0,
                    permit_requested_price=100.0,
                    cache_permit=permit, kind="replacement",
                    api_client=self.client)

        self.client.order_limit_buy.assert_not_called()

    def test_wall_clock_expiry_refuses_after_suspend_like_monotonic_gap(self):
        issued_wall = 10_000.0
        issued_monotonic = 500.0
        with (
            patch.object(placeorder.time, "time", return_value=issued_wall),
            patch.object(
                placeorder.time, "monotonic",
                return_value=issued_monotonic),
        ):
            permit = self._issue()

        with (
            patch.object(
                placeorder.time, "time",
                return_value=(
                    issued_wall
                    + placeorder.binance_cache_health.MAX_AGE_SEC + 1.0)),
            patch.object(
                placeorder.time, "monotonic",
                return_value=issued_monotonic + 1.0),
        ):
            with self.assertRaisesRegex(
                    SubmissionRefused, "account_cache_not_fresh"):
                placeorder._submit_binance_order(
                    "BUY", "BTCUSDC", 0.01, price=100.0,
                    permit_requested_price=100.0,
                    cache_permit=permit, kind="replacement",
                    api_client=self.client)

        self.client.order_limit_buy.assert_not_called()

    def test_permit_scope_and_consumed_state_cannot_be_mutated(self):
        permit = self._issue()
        with self.assertRaisesRegex(AttributeError, "immutable"):
            permit._scope = None
        with self.assertRaisesRegex(AttributeError, "immutable"):
            permit._consumed = True
        with self.assertRaisesRegex(AttributeError, "immutable"):
            permit.expires_at = 0

    def test_near_stale_snapshot_cannot_authorize_a_pre_cancel_window(self):
        near_stale = CacheHealthStatus(
            True, "",
            placeorder.binance_cache_health.MAX_AGE_SEC - 1.0,
            1.0,
            "order-version",
            "trade-version",
        )
        with patch.object(
                placeorder, "require_account_cache_for_submit",
                return_value=near_stale):
            with self.assertRaisesRegex(
                    SubmissionRefused, "account_cache_not_fresh"):
                placeorder.issue_account_cache_submit_permit(
                    "BTCUSDC", "BUY", 0.02, 100.0,
                    market=False, kind="replacement")
        self.client.order_limit_buy.assert_not_called()

    def test_permit_rejects_a_more_expensive_buy(self):
        permit = self._issue()
        with self.assertRaisesRegex(
                SubmissionRefused, "account_cache_not_fresh"):
            placeorder._submit_binance_order(
                "BUY", "BTCUSDC", 0.01, price=100.01,
                permit_requested_price=100.0,
                cache_permit=permit, kind="replacement",
                api_client=self.client)
        self.client.order_limit_buy.assert_not_called()

    def test_sell_floor_normalization_may_move_by_at_most_one_tick(self):
        permit = self._issue(side="SELL")
        placeorder._submit_binance_order(
            "SELL", "BTCUSDC", 0.01, price=99.99,
            permit_requested_price=100.0,
            cache_permit=permit, kind="replacement",
            api_client=self.client)
        self.client.order_limit_sell.assert_called_once()

        unsafe = self._issue(side="SELL")
        with self.assertRaisesRegex(
                SubmissionRefused, "account_cache_not_fresh"):
            placeorder._submit_binance_order(
                "SELL", "BTCUSDC", 0.01, price=99.98,
                permit_requested_price=100.0,
                cache_permit=unsafe, kind="replacement",
                api_client=self.client)
        self.assertEqual(self.client.order_limit_sell.call_count, 1)

    def test_trading_disabled_after_issue_burns_and_refuses_permit(self):
        permit = self._issue()
        with patch.object(
                placeorder.cfg, "is_trade_enabled", return_value=False):
            with self.assertRaisesRegex(
                    SubmissionRefused, "trading_disabled"):
                placeorder._submit_binance_order(
                    "BUY", "BTCUSDC", 0.01, price=100.0,
                    permit_requested_price=100.0,
                    cache_permit=permit, kind="replacement",
                    api_client=self.client)
        with self.assertRaisesRegex(
                SubmissionRefused, "account_cache_not_fresh"):
            placeorder._submit_binance_order(
                "BUY", "BTCUSDC", 0.01, price=100.0,
                permit_requested_price=100.0,
                cache_permit=permit, kind="replacement",
                api_client=self.client)
        self.client.order_limit_buy.assert_not_called()

    def test_scope_mismatch_burns_permit_before_client_call(self):
        cases = (
            {"submit": {"symbol": "ETHUSDC"}},
            {"submit": {"side": "SELL"}},
            {"submit": {"qty": 0.03}},
            {"submit": {"permit_requested_price": 101.0}},
            {"submit": {"kind": "different"}},
            {
                "issue": {"market": True, "price": None},
                "submit": {"market": False, "permit_requested_price": None},
            },
        )
        for case in cases:
            with self.subTest(case=case):
                self.client.reset_mock()
                permit = self._issue(**case.get("issue", {}))
                submit = {
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "qty": 0.01,
                    "price": 100.0,
                    "market": False,
                    "permit_requested_price": 100.0,
                    "kind": "replacement",
                }
                submit.update(case["submit"])
                with self.assertRaisesRegex(
                        SubmissionRefused, "account_cache_not_fresh"):
                    placeorder._submit_binance_order(
                        submit.pop("side"), submit.pop("symbol"),
                        submit.pop("qty"), api_client=self.client,
                        cache_permit=permit, **submit)
                with self.assertRaises(SubmissionRefused):
                    placeorder._submit_binance_order(
                        "BUY", "BTCUSDC", 0.01, price=100.0,
                        permit_requested_price=100.0,
                        cache_permit=permit, kind="replacement",
                        api_client=self.client)
                self.client.order_limit_buy.assert_not_called()
                self.client.order_market_buy.assert_not_called()

    def test_cross_process_permit_refuses(self):
        permit = self._issue()
        with patch.object(placeorder.os, "getpid", return_value=permit.pid + 1):
            with self.assertRaisesRegex(
                    SubmissionRefused, "account_cache_not_fresh"):
                placeorder._submit_binance_order(
                    "BUY", "BTCUSDC", 0.01, price=100.0,
                    permit_requested_price=100.0,
                    cache_permit=permit, kind="replacement",
                    api_client=self.client)
        self.client.order_limit_buy.assert_not_called()


class BinanceSubmitGuardTest(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.order_limit_buy.return_value = {"orderId": 1}
        self.client.order_market_buy.return_value = {"orderId": 2}
        self.trade_enabled = patch.object(
            placeorder.cfg, "is_trade_enabled", return_value=True)
        self.stale_cache = patch.object(
            placeorder.binance_cache_health,
            "require_fresh_account_cache",
            side_effect=placeorder.binance_cache_health.AccountCacheNotReady(
                "trade_cache_stale"),
        )
        self.trade_enabled.start()
        self.stale_cache.start()

    def tearDown(self):
        self.stale_cache.stop()
        self.trade_enabled.stop()

    def test_stale_cache_blocks_limit_and_market_before_client_call(self):
        submissions = (
            {"price": 100.0, "market": False},
            {"market": True},
        )
        for options in submissions:
            with self.subTest(options=options):
                with self.assertRaisesRegex(
                        SubmissionRefused, "account_cache_not_fresh"):
                    placeorder._submit_binance_order(
                        "BUY", "BTCUSDC", 0.01,
                        api_client=self.client, **options)
        self.client.order_limit_buy.assert_not_called()
        self.client.order_market_buy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
