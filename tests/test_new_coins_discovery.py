#!/usr/bin/env python3
"""Unit tests for market_monitor.new_coins_discovery."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from market_monitor.new_coins_discovery import (
    NewCoinsMonitor,
    NewCoinsFactory,
    is_trackable_symbol,
)


class TestNewCoinsDiscovery(unittest.TestCase):
    def test_is_trackable_symbol(self):
        self.assertFalse(is_trackable_symbol("USDC"))
        self.assertFalse(is_trackable_symbol("BTC"))
        self.assertFalse(is_trackable_symbol("ETH"))
        self.assertFalse(is_trackable_symbol("SPCX", name="SpaceX Derivative"))
        self.assertFalse(is_trackable_symbol("TEST$", name="Bad Char"))
        self.assertTrue(is_trackable_symbol("NEWCOIN", name="New Coin Project"))

    def test_trigger_alerts_calls_registered_callbacks(self):
        factory = MagicMock(spec=NewCoinsFactory)
        factory.get_all_new_coins.return_value = {}
        factory.get_available_sources.return_value = ["coinmarketcap"]

        monitor = NewCoinsMonitor(factory=factory)
        received_alerts = []

        def callback(alerts):
            received_alerts.extend(alerts)

        monitor.register_alerts_callback(callback)

        test_coins = [
            {
                "symbol": "FOO",
                "name": "Foo Token",
                "added_at": None,
                "price": 1.23,
                "url": "https://coinmarketcap.com/currencies/foo/",
            }
        ]

        # Should execute cleanly without UnboundLocalError
        monitor._trigger_alerts(test_coins, source_name="coinmarketcap", auto_add=False)

        self.assertEqual(len(received_alerts), 1)
        self.assertEqual(received_alerts[0]["symbol"], "FOO")
        self.assertEqual(received_alerts[0]["type"], "new_coin_discovered")
        self.assertEqual(received_alerts[0]["source"], "coinmarketcap")
        self.assertEqual(received_alerts[0]["price"], 1.23)

    def test_refresh_triggers_refresh_all_when_requested(self):
        factory = MagicMock(spec=NewCoinsFactory)
        factory.get_all_new_coins.return_value = {
            "coinmarketcap": [
                {"symbol": "BAR", "name": "Bar Token", "slug": "bar"}
            ]
        }
        monitor = NewCoinsMonitor(factory=factory)
        self.assertIn("BAR", monitor.all_symbols)

        # Calling refresh with refresh_sources=True must call factory.refresh_all
        monitor.refresh(refresh_sources=True)
        factory.refresh_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
