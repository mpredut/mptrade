#!/usr/bin/env python3
"""Tests for the fail-fast market_alerts.conf parser."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_monitor.alerts_config import load_config, resolve  # noqa: E402

SAMPLE = """
# Comment.
watch = BTC, TAO, HYPE
sources = coinmarketcap, coingecko
discover_new_coins = no
default  = 4.1 / 7.5
new_coin = 12 / 25
BTC = 6 / 10      # inline comment
ETH = 5 / 9
cooldown_minutes = 45
lookback_hours = 24
max_monitored = 20
max_new_coins = 8
new_coins_scan_seconds = 3600
price_scan_seconds = 60
"""


def _tmp(content):
    f = tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False)
    f.write(content); f.close()
    return f.name


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(_tmp(SAMPLE))

    def test_watch_and_sources(self):
        self.assertEqual(self.cfg["watch"], ["BTC", "TAO", "HYPE"])
        self.assertEqual(self.cfg["sources"], ["coinmarketcap", "coingecko"])

    def test_bucket_thresholds(self):
        ac = self.cfg["alert_config"]
        self.assertEqual(ac["default"], {"up_percent": 4.1, "down_percent": 7.5})
        self.assertEqual(ac["dynamic"], {"up_percent": 12.0, "down_percent": 25.0})

    def test_per_coin_thresholds(self):
        per = self.cfg["alert_config"]["per_coin"]
        self.assertEqual(per["BTC"], {"up_percent": 6.0, "down_percent": 10.0})
        self.assertEqual(per["ETH"], {"up_percent": 5.0, "down_percent": 9.0})

    def test_settings(self):
        self.assertEqual(self.cfg["alert_config"]["cooldown_minutes"], 45)
        self.assertEqual(self.cfg["max_new_coins"], 8)
        self.assertEqual(self.cfg["max_monitored"], 20)

    def test_malformed_value_fails_closed(self):
        with self.assertRaises(ValueError):
            load_config(_tmp(SAMPLE.replace("lookback_hours = 24", "lookback_hours = bad")))


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.ac = load_config(_tmp(SAMPLE))["alert_config"]

    def test_resolution_precedence(self):
        cases = (
            ("per-coin", "BTC", False, "up_percent", 6.0),
            ("default", "SOL", False, "up_percent", 4.1),
            ("dynamic", "FOONEW", True, "down_percent", 25.0),
            # A per-coin threshold takes precedence over the dynamic bucket.
            ("per-coin-over-dynamic", "BTC", True, "up_percent", 6.0),
        )
        for label, symbol, dynamic, field, expected in cases:
            with self.subTest(case=label):
                self.assertEqual(resolve(self.ac, symbol, is_dynamic=dynamic)[field], expected)


class TestMissing(unittest.TestCase):
    def test_missing_file_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            load_config("/does/not/exist.conf")

    def test_missing_key_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "max_monitored"):
            load_config(_tmp(SAMPLE.replace("max_monitored = 20\n", "")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
