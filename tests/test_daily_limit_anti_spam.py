"""Unit tests for configurable anti-spam (recent_transaction) window and side policy."""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers.market_api import MarketApi, MarketDataProvider
import order_guard
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


class TestRecentTransactionConfig(unittest.TestCase):
    def setUp(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def tearDown(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0
        for k in ("RECENT_TRANSACTION_SEC", "RECENT_TRANSACTION_SIDE",
                  "RECENT_TRANSACTION_BUY_SEC", "RECENT_TRANSACTION_SELL_SEC"):
            os.environ.pop(k, None)

    def test_normalize_recent_transaction_side(self):
        norm = order_guard.normalize_recent_transaction_side
        self.assertEqual(norm("buy"), "buy")
        self.assertEqual(norm("BUY"), "buy")
        self.assertEqual(norm("buys"), "buy")
        self.assertEqual(norm("only_buy"), "buy")

        self.assertEqual(norm("sell"), "sell")
        self.assertEqual(norm("SELL"), "sell")
        self.assertEqual(norm("only_sell"), "sell")

        self.assertEqual(norm("both"), "both")
        self.assertEqual(norm("symmetric"), "both")
        self.assertEqual(norm("all"), "both")
        self.assertEqual(norm("buy_sell"), "both")
        self.assertEqual(norm("buy,sell"), "both")
        self.assertEqual(norm("two_way"), "both")

        self.assertEqual(norm("off"), "off")
        self.assertEqual(norm("none"), "off")
        self.assertEqual(norm("disabled"), "off")
        self.assertEqual(norm("0"), "off")
        self.assertEqual(norm(False), "off")

        self.assertEqual(norm(None), "buy")
        self.assertEqual(norm("unknown_value"), "buy")

    def test_recent_transaction_sec_resolution(self):
        with patch.dict(order_guard._load_margins(), {
            "default_recent_transaction_sec": 90.0,
            "default_recent_transaction_buy_sec": 45.0,
            "default_recent_transaction_sell_sec": 30.0,
            "binance_recent_transaction_sec": 60.0,
            "binance_recent_transaction_buy_sec": 50.0,
        }, clear=False):
            # Specific venue + side
            self.assertEqual(order_guard.recent_transaction_sec_for("binance", "BUY"), 50.0)
            # Specific venue fallback for SELL
            self.assertEqual(order_guard.recent_transaction_sec_for("binance", "SELL"), 30.0)
            # Other venue with default side overrides
            self.assertEqual(order_guard.recent_transaction_sec_for("kraken", "BUY"), 45.0)
            self.assertEqual(order_guard.recent_transaction_sec_for("kraken", "SELL"), 30.0)

    def test_recent_transaction_sec_env_override(self):
        os.environ["RECENT_TRANSACTION_BUY_SEC"] = "15.0"
        self.assertEqual(order_guard.recent_transaction_sec_for("binance", "BUY"), 15.0)

        os.environ["RECENT_TRANSACTION_SEC"] = "22.0"
        self.assertEqual(order_guard.recent_transaction_sec_for("binance", "SELL"), 22.0)

    def test_recent_transaction_side_env_override(self):
        os.environ["RECENT_TRANSACTION_SIDE"] = "symmetric"
        self.assertEqual(order_guard.recent_transaction_side_for("binance"), "both")


class TestDailyLimitAntiSpamEnforcement(unittest.TestCase):
    def setUp(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def tearDown(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0
        for k in ("RECENT_TRANSACTION_SEC", "RECENT_TRANSACTION_SIDE"):
            os.environ.pop(k, None)

    def _make_provider(self, side="BUY", age_s=5.0):
        prov = FakeProvider()
        prov.seed_trade(side, age_sec=age_s)
        return prov

    def test_default_buy_policy(self):
        # Default is buy-only
        prov_buy = self._make_provider("BUY", age_s=5.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_buy, "BTCUSDC", "BUY", recent_transaction_sec=90, recent_transaction_side="buy")
        self.assertFalse(ok)
        self.assertEqual(reason, "recent_transaction")

        prov_sell = self._make_provider("SELL", age_s=5.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_sell, "BTCUSDC", "SELL", recent_transaction_sec=90, recent_transaction_side="buy")
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_symmetric_policy(self):
        # Symmetric blocks both BUY and SELL on recent trades
        prov_buy = self._make_provider("BUY", age_s=5.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_buy, "BTCUSDC", "BUY", recent_transaction_sec=90, recent_transaction_side="symmetric")
        self.assertFalse(ok)
        self.assertEqual(reason, "recent_transaction")

        prov_sell = self._make_provider("SELL", age_s=5.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_sell, "BTCUSDC", "SELL", recent_transaction_sec=90, recent_transaction_side="symmetric")
        self.assertFalse(ok)
        self.assertEqual(reason, "recent_transaction")

    def test_sell_only_policy(self):
        # Sell-only blocks SELL on recent trades, but allows BUY
        prov_buy = self._make_provider("BUY", age_s=5.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_buy, "BTCUSDC", "BUY", recent_transaction_sec=90, recent_transaction_side="sell")
        self.assertTrue(ok)
        self.assertIsNone(reason)

        prov_sell = self._make_provider("SELL", age_s=5.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_sell, "BTCUSDC", "SELL", recent_transaction_sec=90, recent_transaction_side="sell")
        self.assertFalse(ok)
        self.assertEqual(reason, "recent_transaction")

    def test_off_policy(self):
        # Off allows both even with 1s old trades
        prov_buy = self._make_provider("BUY", age_s=1.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_buy, "BTCUSDC", "BUY", recent_transaction_sec=90, recent_transaction_side="off")
        self.assertTrue(ok)

        prov_sell = self._make_provider("SELL", age_s=1.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_sell, "BTCUSDC", "SELL", recent_transaction_sec=90, recent_transaction_side="off")
        self.assertTrue(ok)

    def test_fine_grained_bypass_anti_spam(self):
        # bypass_guards={"recent_transaction"} allows recent trade
        prov_buy = self._make_provider("BUY", age_s=2.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_buy, "BTCUSDC", "BUY", recent_transaction_sec=90,
            bypass_guards={"recent_transaction"})
        self.assertTrue(ok)
        self.assertIsNone(reason)

        # But daily_limit cap still blocks if trade count is exceeded
        prov_many = FakeProvider()
        for _ in range(100):
            prov_many.seed_trade("BUY", age_sec=3600.0)
        ok, reason = order_guard.daily_limit_guard(
            prov_many, "BTCUSDC", "BUY", max_daily_trades=25, safeback_sec=86400,
            bypass_guards={"recent_transaction"})
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_limit")

    def test_instrument_place_forwarding(self):
        from providers.market_api import MarketApi
        prov = FakeProvider(name="kraken")
        prov.seed_trade("SELL", age_sec=5.0)
        api = MarketApi([prov])
        inst = Instrument(name="KRAKEN_BTC", symbol="BTCUSDC", provider="kraken",
                          base="BTC", quote="USDC", api=api)

        # By default, SELL is allowed with recent trade
        with patch.object(order_guard, "profit_guard", return_value=True):
            res = inst.place("SELL", 100.0, 1.0, bypass_profit_guard=True)
            self.assertIsNotNone(res)

        # With recent_transaction_side="both", SELL is blocked
        with patch.object(order_guard, "profit_guard", return_value=True):
            res_blocked = inst.place(
                "SELL", 100.0, 1.0, bypass_profit_guard=True,
                recent_transaction_side="both"
            )
            self.assertIsNone(res_blocked)


if __name__ == "__main__":
    unittest.main(verbosity=2)
