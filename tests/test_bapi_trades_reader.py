"""Tests for read-only trade-cache consumption in trading processes."""

import sys
import types
import unittest
from unittest import mock

from binance_api import bapi_trades


class _SnapshotLock:
    def __init__(self, manager):
        self.manager = manager
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.held = False
        # Simulate the central reader loading a newer generation immediately
        # after the consumer has copied its coherent snapshot.
        self.manager.cache["BTCUSDC"].clear()
        return False


class BapiTradesReaderTest(unittest.TestCase):
    def tearDown(self):
        bapi_trades.cache_trade_manager = None

    def test_bapi_trades_reader_behaviors(self):
        with self.subTest(msg="lazy manager is created without private synchronization"):
            manager = types.SimpleNamespace()
            get_cache_manager = mock.Mock(return_value=manager)
            cache_module = types.SimpleNamespace(get_cache_manager=get_cache_manager)

            with mock.patch.dict(sys.modules, {"cacheManager": cache_module}):
                actual = bapi_trades.init_cache_trade_manager()

            self.assertIs(actual, manager)
            get_cache_manager.assert_called_once_with("Trade", start_sync=False)

        with self.subTest(msg="recent trade reader copies one locked generation"):
            trade = {
                "symbol": "BTCUSDC",
                "id": 11,
                "orderId": 22,
                "price": "100.5",
                "qty": "0.01",
                "time": 2_000_000,
                "isBuyer": True,
            }
            manager = types.SimpleNamespace(cache={"BTCUSDC": [trade]})
            manager.lock = _SnapshotLock(manager)

            with (
                mock.patch.object(
                    bapi_trades, "init_cache_trade_manager", return_value=manager
                ),
                mock.patch.object(bapi_trades.sym, "validate_ordertype"),
                mock.patch.object(bapi_trades.sym, "validate_symbols"),
                mock.patch.object(bapi_trades.time, "time", return_value=2_001.0),
            ):
                result = bapi_trades.get_trade_orders("BUY", "BTCUSDC", 10)

            self.assertFalse(manager.lock.held)
            self.assertEqual(manager.cache["BTCUSDC"], [])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 11)
            self.assertEqual(result[0]["price"], 100.5)


if __name__ == "__main__":
    unittest.main()
