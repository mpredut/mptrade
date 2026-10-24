"""Unit tests for symmetric GO1 (daily_limit_guard), Kraken multi-fill aggregation, and daily_limit deferral."""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers.market_api import MarketApi, MarketDataProvider
from providers.kraken_provider import KrakenProvider
import order_guard
import order_retry
from instrument import Instrument


class FakeProvider(MarketDataProvider):
    """Minimal in-memory provider whose guards are handled by Instrument."""

    def __init__(self, name="FakeVenue", price=100.0):
        self._name = name
        self._price = price
        self._orders = []
        self.placed = []

    @property
    def name(self):
        return self._name

    def get_current_price(self, symbol):
        return self._price

    def supports_symbol(self, symbol):
        return False

    def free_balance(self, asset):
        return 1_000_000.0

    def get_orders(self, symbol, side, since_s):
        cutoff_ms = time.time() * 1000 - since_s * 1000
        out = [o for o in self._orders
               if o.get("symbol") == symbol and o.get("timestamp", 0) >= cutoff_ms]
        if side:
            out = [o for o in out if o.get("side") == side.upper()]
        return out

    def place_order(self, symbol, side, price, qty, **kwargs):
        payload = {"orderId": len(self.placed) + 1, "symbol": symbol, "side": side, "price": price, "qty": qty}
        self.placed.append(payload)
        return payload

    def guards_internally(self):
        return False

    def seed_trade(self, side, age_sec=0.0, price=100.0, qty=1.0, symbol="BTCUSDC"):
        self._orders.append({
            "orderId": len(self._orders) + 1,
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "qty": qty,
            "timestamp": (time.time() - age_sec) * 1000.0
        })


class TestSymmetricDailyLimitGuard(unittest.TestCase):
    def setUp(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def tearDown(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def test_recent_trades_below_daily_cap_are_allowed_symmetrically(self):
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            prov.seed_trade(side, age_sec=2.0)
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=25, safeback_sec=86400)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_daily_limit_blocks_buy_and_sell_symmetrically(self):
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            for _ in range(30):
                prov.seed_trade(side, age_sec=3600.0)
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=25, safeback_sec=86400)
            self.assertFalse(ok)
            self.assertEqual(reason, "daily_limit")

    def test_bypass_go1_or_daily_limit_or_all_skips_cap(self):
        prov = FakeProvider()
        for _ in range(30):
            prov.seed_trade("BUY", age_sec=3600.0)
        for bypass in ({"GO1"}, {"daily_limit"}, {"all"}):
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", "BUY", max_daily_trades=25, safeback_sec=86400,
                bypass_guards=bypass)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_daily_limit_is_non_failure_deferral_in_order_retry(self):
        self.assertTrue(order_retry.is_non_failure_deferral("daily_limit"))

    def test_kraken_get_orders_groups_multi_fill_orders_by_order_id(self):
        kp = KrakenProvider()
        now_ms = int(time.time() * 1000)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            cache_path = tf.name
            json.dump({
                "items": {
                    "HYPEUSD": [
                        {
                            "symbol": "HYPEUSD",
                            "id": "T1",
                            "orderId": "O-PARENT-1",
                            "side": "BUY",
                            "price": 40.0,
                            "qty": 2.0,
                            "timestamp": now_ms - 10_000,
                        },
                        {
                            "symbol": "HYPEUSD",
                            "id": "T2",
                            "orderId": "O-PARENT-1",
                            "side": "BUY",
                            "price": 43.0,
                            "qty": 1.0,
                            "timestamp": now_ms - 9_000,
                        },
                        {
                            "symbol": "HYPEUSD",
                            "id": "T3",
                            "orderId": "O-PARENT-2",
                            "side": "BUY",
                            "price": 44.0,
                            "qty": 1.5,
                            "timestamp": now_ms - 5_000,
                        },
                    ]
                },
                "fetchtime": {"HYPEUSD": now_ms},
            }, tf)
        try:
            with patch("providers.kraken_provider._KRAKEN_CACHE_FILE", cache_path):
                orders = kp.get_orders("HYPEUSD", "BUY", 3600.0)
            self.assertEqual(len(orders), 2)
            # First order is VWAP of (40*2 + 43*1) / 3 = 41.0, qty=3.0, max timestamp
            self.assertAlmostEqual(orders[0]["price"], 41.0)
            self.assertAlmostEqual(orders[0]["qty"], 3.0)
            self.assertEqual(orders[0]["timestamp"], now_ms - 9_000)
            self.assertEqual(orders[0]["side"], "BUY")
            # Second order
            self.assertAlmostEqual(orders[1]["price"], 44.0)
            self.assertAlmostEqual(orders[1]["qty"], 1.5)
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def test_instrument_place_unified_kind_and_motivation(self):
        prov = FakeProvider(name="kraken")
        api = MarketApi([prov])
        inst = Instrument(name="KRAKEN_BTC", symbol="BTCUSDC", provider="kraken",
                          base="BTC", quote="USDC", api=api)

        with tempfile.TemporaryDirectory() as tmpdir:
            qfile = os.path.join(tmpdir, "order_retry_queue.jsonl")
            lfile = os.path.join(tmpdir, "order_retry.lock")
            with patch.object(order_retry, "QUEUE_FILE", qfile), \
                 patch.object(order_retry, "LOCK_FILE", lfile), \
                 patch.object(order_guard, "profit_guard", return_value=True):
                res = inst.place("SELL", 100.0, 1.0, bypass_guards={"all"}, kind="STOP")
                self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
