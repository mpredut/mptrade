"""Verify the BinanceProvider StrategyExecutor contract with a fake API."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers import market_api  # noqa: E402
from providers.market_api import BinanceProvider  # noqa: E402
from providers.strategy_executor import (  # noqa: E402
    StrategyExecutor, PairPrecision, ProviderError,
    SubmissionRefused,
)


class FakeClient:
    def __init__(self):
        self.order_calls = []
        self.open = []

    def get_symbol_info(self, symbol):
        return {"baseAsset": "BTC", "quoteAsset": "USDC", "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.00100000", "minQty": "0.00100000"}]}

    def get_klines(self, symbol, interval, limit):
        return [[0, "1", "1", "1", "10"], [0, "1", "1", "1", "11"], [0, "1", "1", "1", "12"]]

    def get_order(self, symbol, orderId=None, origClientOrderId=None):
        if origClientOrderId is not None:
            return {
                "orderId": 77, "clientOrderId": origClientOrderId,
                "side": "BUY", "price": "60", "origQty": "2",
                "executedQty": "0", "status": "NEW",
            }
        return {"status": "FILLED", "executedQty": "2.0", "cummulativeQuoteQty": "120.0"}

    def get_open_orders(self, symbol):
        return list(self.open)

    def get_my_trades(self, symbol, orderId):
        return [{
            "orderId": orderId, "price": "60", "commission": "0.12",
            "commissionAsset": "USDC",
        }]

    def order_market_buy(self, symbol, quantity, **kwargs):
        self.order_calls.append(("buy", symbol, quantity, kwargs))
        return {"orderId": 555}

    def order_market_sell(self, symbol, quantity, **kwargs):
        self.order_calls.append(("sell", symbol, quantity, kwargs))
        return {"orderId": 556}


class FakeBapi:
    def __init__(self):
        self.client = FakeClient()
        self.calls = []
        self.cancel_result = True

    def get_current_price(self, symbol):
        return 600.0

    def cancel_order(self, symbol, order_id):
        self.calls.append((symbol, order_id))
        return self.cancel_result


class BinanceExecutorContractTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeBapi()
        self._patch = mock.patch.object(market_api, "_bapi", self.fake)
        self._patch.start()
        from binance_api import bapi_placeorder
        self.placeorder = bapi_placeorder
        self._trade_enabled_patch = mock.patch.object(
            bapi_placeorder.cfg, "is_trade_enabled", return_value=True,
        )
        self._freshness_patch = mock.patch.object(
            bapi_placeorder.binance_cache_health, "require_fresh_account_cache",
            return_value=bapi_placeorder.binance_cache_health.CacheHealthStatus(
                ready=True, reason="ok", order_age_sec=1.0,
                trade_age_sec=1.0,
                order_cache_version="order-v1",
                trade_cache_version="trade-v1",
            ),
        )
        self._trade_enabled_patch.start()
        self.freshness_gate = self._freshness_patch.start()
        self._reader_patch = mock.patch(
            "cacheManager.ensure_account_cache_readers")
        self.reader_gate = self._reader_patch.start()
        self.p = BinanceProvider()

    def tearDown(self):
        self._freshness_patch.stop()
        self._reader_patch.stop()
        self._trade_enabled_patch.stop()
        self._patch.stop()

    def test_satisfies_protocol(self):
        self.assertIsInstance(self.p, StrategyExecutor)

    def test_pair_precision(self):
        pp = self.p.pair_precision("BTCUSDC")
        self.assertEqual(pp, PairPrecision(price_decimals=2, volume_decimals=3,
                                           order_min=0.001, base_asset="BTC"))

    def test_ohlc_closes_exclude_forming_bar(self):
        self.assertEqual(self.p.ohlc_closes("BTCUSDC", 240), [10.0, 11.0])

    def test_ohlc_series_preserves_last_completed_timestamp(self):
        self.fake.client.get_klines = lambda **_kwargs: [
            [0, "1", "1", "1", "10", "0", 1_000],
            [0, "1", "1", "1", "11", "0", 2_000],
            [0, "1", "1", "1", "12", "0", 3_000],
        ]
        series = self.p.ohlc_series("BTCUSDC", 240)
        self.assertEqual(series.closes, (10.0, 11.0))
        self.assertEqual(series.last_closed_at, 2.0)
        self.assertEqual(series.timestamps, (1.0, 2.0))

    def test_order_status_filled(self):
        st = self.p.order_status("BTCUSDC", "42")
        self.assertEqual(st.status, "closed")
        self.assertAlmostEqual(st.filled_qty, 2.0)
        self.assertAlmostEqual(st.cost, 120.0)
        self.assertAlmostEqual(st.fee, 0.12)

    def test_order_status_expired_in_match_is_terminal_with_remainder(self):
        self.fake.client.get_order = lambda **_kwargs: {
            "status": "EXPIRED_IN_MATCH",
            "executedQty": "0.75",
            "cummulativeQuoteQty": "45.0",
        }
        self.fake.client.get_my_trades = lambda **_kwargs: []

        status = self.p.order_status("BTCUSDC", "42")

        self.assertEqual(status.status, "expired")
        self.assertEqual(status.venue_status, "EXPIRED_IN_MATCH")
        self.assertTrue(status.terminal)
        self.assertAlmostEqual(2.0 - status.filled_qty, 1.25)

    def test_submit_order_market_returns_order_id(self):
        oid = self.p.submit_order("BTCUSDC", "buy", 0.01, price=None, market=True)
        self.assertEqual(oid, "555")

    def test_submit_order_market_propagates_client_order_id(self):
        client_id = "SD_0123456789abcdef0123456789abcdef"
        self.p.submit_order(
            "BTCUSDC", "buy", 0.01, market=True,
            client_order_id=client_id,
        )
        self.assertEqual(
            self.fake.client.order_calls[-1][-1]["newClientOrderId"], client_id,
        )

    def test_submit_order_without_an_orderId_raises(self):
        self.fake.client.order_market_buy = lambda symbol, quantity, **kwargs: {}
        with self.assertRaises(ProviderError):
            self.p.submit_order("BTCUSDC", "buy", 0.01, market=True)

    def test_stale_account_cache_refusal_reaches_caller_without_submission(self):
        self.freshness_gate.side_effect = (
            self.placeorder.binance_cache_health.AccountCacheNotReady(
                "trade_cache_stale")
        )
        with self.assertRaisesRegex(
                SubmissionRefused, "account_cache_not_fresh") as raised:
            self.p.submit_order("BTCUSDC", "buy", 0.01, market=True)
        self.assertEqual(raised.exception.__cause__.reason, "trade_cache_stale")
        self.assertEqual(self.fake.client.order_calls, [])

    def test_cancel_delegates(self):
        self.p.cancel_order("BTCUSDC", "42")
        self.assertIn(("BTCUSDC", 42), self.fake.calls)

    def test_unconfirmed_cancel_raises(self):
        self.fake.cancel_result = False
        with self.assertRaises(ProviderError):
            self.p.cancel_order("BTCUSDC", "42")

    def test_open_orders_keeps_the_identity_for_recovery(self):
        self.fake.client.open = [{
            "orderId": 77, "clientOrderId": "RT_abc", "side": "buy",
            "price": "60", "origQty": "2", "executedQty": "0.5",
            "status": "PARTIALLY_FILLED",
        }]
        self.assertEqual(self.p.open_orders("BTCUSDC"), [{
            "orderId": "77", "clientOrderId": "RT_abc", "side": "BUY",
            "price": 60.0, "origQty": 2.0, "executedQty": 0.5,
            "status": "PARTIALLY_FILLED",
        }])

    def test_lookup_by_client_order_id(self):
        order = self.p.order_by_client_id("BTCUSDC", "RT_abc")
        self.assertEqual(order["orderId"], 77)
        self.assertEqual(order["clientOrderId"], "RT_abc")

    def test_position_cost_basis_returns_none_for_unknown_symbol(self):
        self.assertIsNone(self.p.position_cost_basis("UNLISTEDCOIN", 1.0, 86400.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
