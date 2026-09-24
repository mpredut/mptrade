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

    def test_repository_config_turns_the_binance_buy_reference_off(self):
        with mock.patch.object(order_guard, "_MARGINS", None):
            self.assertFalse(order_guard.buy_reference_enabled("binance"))
            self.assertTrue(order_guard.buy_reference_enabled("kraken"))


if __name__ == "__main__":
    unittest.main()
