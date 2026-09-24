"""The per-venue switch for the historical BUY reference in order_guard.profit_guard.

With the reference on, a BUY must sit below the lowest sell of the window (minus the
margin); after a rise that anchor blocked every automated Binance buy for two weeks.
Switching it off for a venue must leave the SELL reference untouched."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_guard  # noqa: E402


class _Provider:
    def __init__(self, name, last_fill=216.28):
        self.name = name
        self._last_fill = last_fill

    def last_opposite_fill(self, _symbol, _order_type):
        return self._last_fill


def _margins(**overrides):
    base = {
        "default": 1.15, "binance": 1.15, "default_window_h": 0.0,
        "default_max_daily_trades": 25, "default_safeback_sec": 14 * 24 * 3600 + 60,
        "default_recent_transaction_sec": 180, "default_buy_reference": 1.0,
    }
    base.update(overrides)
    return base


class BuyReferenceSwitchTest(unittest.TestCase):
    def test_buy_reference_off_lets_a_buy_above_old_sells_through(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            provider = _Provider("binance")
            # TAO on 24 Sep: lowest sell of the window 216.28, price ~293.
            self.assertTrue(order_guard.profit_guard(
                provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_sell_reference_stays_active_when_buy_reference_is_off(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            provider = _Provider("binance")
            # Highest buy 276.0: a SELL below it (plus margin) is still refused.
            self.assertFalse(order_guard.profit_guard(
                provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))
            self.assertTrue(order_guard.profit_guard(
                provider, "TAOUSDC", "SELL", 290.0, 1.15, window_ref=276.0))

    def test_other_venues_keep_the_buy_reference(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            self.assertFalse(order_guard.profit_guard(
                _Provider("kraken"), "HYPEUSD", "BUY", 93.0, 1.15, window_ref=56.66))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))
            self.assertFalse(order_guard.buy_reference_enabled("binance"))

    def test_unreadable_value_keeps_the_conservative_guard(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="maybe")):
            self.assertTrue(order_guard.buy_reference_enabled("binance"))

    def test_repository_config_sets_binance_buy_reference_dynamic(self):
        with mock.patch.object(order_guard, "_MARGINS", None):
            self.assertEqual(order_guard.buy_reference_mode("binance"), "dynamic")
            self.assertEqual(order_guard.buy_reference_mode("kraken"), "on")
            self.assertTrue(order_guard.buy_reference_enabled("binance"))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))

    def test_dynamic_mode_bypasses_in_bull_trend_and_enforces_in_bear_or_flat(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference="dynamic")):
            provider = _Provider("binance")
            # In confirmed BULL trend: BUY above old sell reference is permitted
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertTrue(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # In BEAR trend: BUY above old sell reference is blocked for defense
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bear"):
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # In FLAT/UNKNOWN trend: defensive guard is maintained
            with mock.patch.object(order_guard, "_symbol_trend", return_value="unknown"):
                self.assertFalse(order_guard.profit_guard(
                    provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))

    def test_provider_as_string_and_case_insensitivity(self):
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            # Both string name and Provider object with various casings must work
            self.assertTrue(order_guard.profit_guard("binance", "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertTrue(order_guard.profit_guard("Binance", "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertTrue(order_guard.profit_guard(_Provider("Binance"), "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            self.assertFalse(order_guard.buy_reference_enabled("Binance"))
            self.assertFalse(order_guard.buy_reference_enabled("binance"))
            self.assertEqual(order_guard.buy_reference_mode("Binance"), "off")

    def test_missing_default_buy_reference_in_margins_does_not_raise(self):
        # Even if a test mock or custom dict omits default_buy_reference, no KeyError is thrown
        with mock.patch.object(order_guard, "_MARGINS", {"binance": 1.15}):
            self.assertTrue(order_guard.buy_reference_enabled("binance"))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))

    def test_real_binance_provider_integration(self):
        with mock.patch.object(order_guard, "_MARGINS", None):
            from providers.market_api import BinanceProvider
            provider = BinanceProvider()
            self.assertEqual(order_guard.buy_reference_mode(provider.name), "dynamic")
            # In BULL trend, BUY above historical sell reference is permitted on Binance
            with mock.patch.object(order_guard, "_symbol_trend", return_value="bull"):
                self.assertTrue(order_guard.profit_guard(provider, "TAOUSDC", "BUY", 293.0, 1.15, window_ref=216.28))
            # SELL below historical buy reference is still blocked
            self.assertFalse(order_guard.profit_guard(provider, "TAOUSDC", "SELL", 270.0, 1.15, window_ref=276.0))

    def test_instrument_pipeline_respects_buy_reference_switch(self):
        from instrument import Instrument
        from providers.market_api import MarketApi, MarketDataProvider

        class _MockBinanceProvider(MarketDataProvider):
            def __init__(self):
                self._name = "binance"
                self.placed = []

            @property
            def name(self):
                return self._name

            def guards_internally(self):
                return False

            def get_current_price(self, symbol):
                return 293.0

            def free_balance(self, asset):
                return 10000.0

            def get_orders(self, symbol, side, since_s):
                return []

            def profit_guard_window_ref(self, symbol, side, safeback_override=None):
                return 216.28 if side == "BUY" else 276.0

            def supports_symbol(self, symbol):
                return True

            def order_status(self, symbol, order_id):
                return {"orderId": order_id, "status": "FILLED"}

            def place_order(self, symbol, side, price, qty, **kwargs):
                self.placed.append({"side": side, "qty": qty, "price": price})
                return {"orderId": "123", "status": "FILLED"}

        p = _MockBinanceProvider()
        inst = Instrument(name="TAO", symbol="TAOUSDC", provider="binance", base="TAO", quote="USDC", api=MarketApi([p]))

        # With binance_buy_reference = 0: BUY at 293 (above past sell 216.28) is accepted
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=0.0)):
            res = inst.place("BUY", 293.0, 1.0, motivation="test_buy", cache_permit=object(), smart=False, wait_for_trend=False)
            self.assertIsNotNone(res)
            self.assertEqual(len(p.placed), 1)

        # With binance_buy_reference = 1: BUY at 293 (above past sell 216.28) is refused by profit_guard
        p.placed.clear()
        with mock.patch.object(order_guard, "_MARGINS", _margins(binance_buy_reference=1.0)):
            res = inst.place("BUY", 293.0, 1.0, motivation="test_buy", cache_permit=object(), smart=False, wait_for_trend=False)
            self.assertIsNone(res)
            self.assertEqual(len(p.placed), 0)


if __name__ == "__main__":
    unittest.main()

