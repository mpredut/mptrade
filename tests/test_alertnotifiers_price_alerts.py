from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_tools"))

from alertnotifiers import AlertNotifier  # noqa: E402
from gen_alert_thresholds import FLOOR, thresholds_from_prices  # noqa: E402
from market_monitor.alerts_config import load_config, resolve  # noqa: E402
from market_monitor.new_coins_discovery import (  # noqa: E402
    NewCoinsFactory,
    NewCoinsMonitor,
    is_trackable_symbol,
)
from market_monitor.pricechecker import PriceAlert, PriceChecker  # noqa: E402
from market_monitor.pricefetcher import (  # noqa: E402
    BinancePricePlatform,
    HyperliquidPricePlatform,
    PricePlatformFactory,
)
from notify_engine.alertnotifiers import _alert_identity, _delivery_fingerprint  # noqa: E402


SAMPLE_ALERTS_CONF = """
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


def _tmp_conf(content: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="alerts_conf_"), "alerts.conf")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class DummyCMCPlatform:
    platform_name = "CoinMarketCap"
    _all_listings = {"TAO": {"slug": "tao"}}


class DummyPriceFactory:
    def __init__(self):
        self._platforms = [DummyCMCPlatform()]


class DummyPriceManager:
    def __init__(self, latest_price=100.0, history=None):
        self._latest_price = latest_price
        self._history = history or []
        self.price_factory = DummyPriceFactory()

    def get_latest_price(self, symbol):
        return self._latest_price

    def get_price_history(self, symbol, limit=1000):
        return self._history

    @property
    def symbols(self):
        return []

    @property
    def original_symbols(self):
        return []


class PriceAlertLinkTests(unittest.TestCase):
    def test_price_checker_alert_includes_coinmarketcap_url(self):
        now_ms = int(time.time() * 1000)
        history = [
            {"timestamp": now_ms - 60 * 60 * 1000, "price": 90.0, "timestamp_readable": "2026-05-25T00:00:00"},
            {"timestamp": now_ms - 30 * 60 * 1000, "price": 95.0, "timestamp_readable": "2026-05-25T00:30:00"},
        ]
        checker = PriceChecker(DummyPriceManager(latest_price=100.0, history=history))

        alerts = checker.check_symbol("TAOUSDC")

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].url, "https://coinmarketcap.com/currencies/bittensor/")

    def test_calculate_24h_stats_returns_min_and_max_readable_timestamps(self):
        now_ms = int(time.time() * 1000)
        history = [
            {"timestamp": now_ms - 3 * 60 * 60 * 1000, "price": 101.0, "timestamp_readable": "2026-05-25T00:00:00"},
            {"timestamp": now_ms - 2 * 60 * 60 * 1000, "price": 99.0, "timestamp_readable": "2026-05-25T01:00:00"},
            {"timestamp": now_ms - 1 * 60 * 60 * 1000, "price": 105.0, "timestamp_readable": "2026-05-25T02:00:00"},
        ]
        checker = PriceChecker(DummyPriceManager(latest_price=110.0, history=history))

        stats = checker._calculate_24h_stats("TAOUSDC")

        self.assertEqual(stats["min_price"], 99.0)
        self.assertEqual(stats["min_price_timestamp_readable"], "2026-05-25T01:00:00")
        self.assertEqual(stats["max_price"], 105.0)
        self.assertEqual(stats["max_price_timestamp_readable"], "2026-05-25T02:00:00")

    def test_alert_notifier_uses_reference_time_in_batch_message(self):
        alert = PriceAlert(
            symbol="TAOUSDC",
            alert_type="up",
            current_price=110.0,
            reference_price=99.0,
            percent_change=11.11,
            threshold=5.0,
            reference_time="2026-05-25 01:00:00",
        )

        message = AlertNotifier.format_batch_message([alert])

        self.assertIn("(2026-05-25 01:00:00)", message)

    def test_alert_notifier_includes_coinmarketcap_url_in_batch_message(self):
        alert = PriceAlert(
            symbol="TAOUSDC",
            alert_type="up",
            current_price=100.0,
            reference_price=90.0,
            percent_change=11.11,
            threshold=5.0,
        )
        alert.url = "https://coinmarketcap.com/currencies/bittensor/"

        message = AlertNotifier.format_batch_message([alert])

        self.assertIn("Link: https://coinmarketcap.com/currencies/bittensor/", message)


class PriceCheckerTests(unittest.TestCase):
    def test_check_symbol_returns_empty_list_when_history_is_insufficient(self):
        checker = PriceChecker(DummyPriceManager(history=[]))
        self.assertEqual(checker.check_symbol("BTCUSDC"), [])

    def test_dynamic_symbols_use_separate_thresholds(self):
        now_ms = int(time.time() * 1000)
        history = [
            {"timestamp": now_ms - 2 * 60 * 60 * 1000, "price": 100.0, "timestamp_readable": "2026-05-25T00:00:00"},
            {"timestamp": now_ms - 1 * 60 * 60 * 1000, "price": 105.0, "timestamp_readable": "2026-05-25T01:00:00"},
        ]
        manager = DummyPriceManager(latest_price=105.0, history=history)
        manager.symbol_added_time = {"DYNUSDC": 1.0}

        checker = PriceChecker(manager)

        default_alerts = checker.check_symbol("BTCUSDC")
        dynamic_alerts = checker.check_symbol("DYNUSDC")

        self.assertEqual(len(default_alerts), 1)
        self.assertEqual(default_alerts[0].alert_type, "up")
        self.assertEqual(default_alerts[0].threshold, 4.1)
        self.assertEqual(dynamic_alerts, [])

    def test_price_platforms_fallback_and_error_handling(self):
        with patch("requests.get", side_effect=Exception("offline")), \
             patch("requests.post", side_effect=Exception("offline")):
            bp = BinancePricePlatform()
            self.assertTrue(bp.supports_symbol("BTCUSDC"))
            self.assertTrue(bp.supports_symbol("BTC"))

            hp = HyperliquidPricePlatform()
            self.assertTrue(hp.supports_symbol("BTCUSDC"))
            self.assertTrue(hp.supports_symbol("BTC"))

            factory = PricePlatformFactory()
            factory._platforms[0].get_price = MagicMock(return_value=None)
            factory._platforms[1].get_price = MagicMock(return_value=99999.0)
            res = factory.get_price("BTCUSDC")
            self.assertEqual(res["price"], 99999.0)
            self.assertEqual(res["platform"], "Hyperliquid")

            factory._platforms[1].get_price = MagicMock(return_value=None)
            with self.assertRaises(RuntimeError):
                factory.get_price("BTCUSDC")

            with self.assertRaises(ValueError):
                factory.get_price("NONEXISTENTCOIN999")


class TestNewCoinDictRobustness(unittest.TestCase):
    """New-coin alerts are dicts (not PriceAlert) — they must not crash."""

    NEW_COIN = {
        "type": "new_coin_discovered", "source": "coinmarketcap",
        "symbol": "RWS", "name": "Real World Services", "added_at": None,
        "price": 0.0154, "auto_added": True, "has_price": True,
        "url": "https://coinmarketcap.com/currencies/real-world-services/",
    }

    def test_alert_symbol_dict_and_obj(self):
        self.assertEqual(AlertNotifier.alert_symbol(self.NEW_COIN), "RWS")
        alert = PriceAlert("TAO", "up", 100.0, 90.0, 11.0, 5.0)
        self.assertEqual(AlertNotifier.alert_symbol(alert), "TAO")

    def test_save_to_file_handles_new_coin_dict(self):
        path = os.path.join(tempfile.mkdtemp(), "alerts.log")
        self.assertTrue(AlertNotifier.save_to_file(self.NEW_COIN, filename=path))
        self.assertIn("NEW COIN RWS", open(path, encoding="utf-8").read())

    def test_save_to_file_handles_bot_event_and_dict(self):
        path = os.path.join(tempfile.mkdtemp(), "alerts.log")
        bot_event = {
            "type": "bot_event",
            "name": "TRADE_BUY",
            "symbol": "BTCUSDT",
            "body": "Bought 0.01 BTC at $60000",
            "source": "binance",
        }
        self.assertTrue(AlertNotifier.save_to_file(bot_event, filename=path))
        content = open(path, encoding="utf-8").read()
        self.assertIn("BOT EVENT TRADE_BUY", content)
        self.assertIn("Bought 0.01 BTC at $60000", content)

        generic_dict = {"symbol": "ETHUSDT", "body": "Generic alert message"}
        self.assertTrue(AlertNotifier.save_to_file(generic_dict, filename=path))
        content = open(path, encoding="utf-8").read()
        self.assertIn("ETHUSDT: Generic alert message", content)

    def test_format_batch_mixed(self):
        alert = PriceAlert("TAO", "down", 80.0, 100.0, -20.0, 7.5)
        msg = AlertNotifier.format_batch_message([self.NEW_COIN, alert])
        self.assertIn("RWS", msg)
        self.assertIn("TAO", msg)

    def test_send_does_not_raise_on_dict(self):
        path = os.path.join(tempfile.mkdtemp(), "test_alerts.log")
        AlertNotifier.send(self.NEW_COIN, enable_console=False, enable_file=True,
                           enable_email=False, enable_phone_webhook=False, filename=path)
        self.assertTrue(os.path.exists(path))

    def test_new_coin_identity_deduplication(self):
        coin1 = dict(self.NEW_COIN, price=0.0154, url="https://coinmarketcap.com/currencies/real-world-services/?ref=1")
        coin2 = dict(self.NEW_COIN, price=0.0210, url="https://coinmarketcap.com/currencies/real-world-services/?ref=2")
        id1 = _alert_identity(coin1)
        id2 = _alert_identity(coin2)
        self.assertEqual(id1, id2)
        self.assertEqual(_delivery_fingerprint([coin1]), _delivery_fingerprint([coin2]))

    def test_non_ascii_symbol_preserved(self):
        coin = dict(self.NEW_COIN, symbol="小蝌蚪", name="小蝌蚪 Coin")
        self.assertEqual(AlertNotifier.alert_symbol(coin), "小蝌蚪")
        self.assertIn("小蝌蚪", AlertNotifier.format_batch_message([coin]))
        hdr = AlertNotifier.utf8_header("(1): 小蝌蚪")
        self.assertEqual(hdr.encode("latin-1").decode("utf-8"), "(1): 小蝌蚪")


class TestAlertsConfigLoadAndResolve(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(_tmp_conf(SAMPLE_ALERTS_CONF))
        self.ac = self.cfg["alert_config"]

    def test_watch_and_sources(self):
        self.assertEqual(self.cfg["watch"], ["BTC", "TAO", "HYPE"])
        self.assertEqual(self.cfg["sources"], ["coinmarketcap", "coingecko"])

    def test_bucket_thresholds(self):
        self.assertEqual(self.ac["default"], {"up_percent": 4.1, "down_percent": 7.5})
        self.assertEqual(self.ac["dynamic"], {"up_percent": 12.0, "down_percent": 25.0})

    def test_per_coin_thresholds(self):
        per = self.ac["per_coin"]
        self.assertEqual(per["BTC"], {"up_percent": 6.0, "down_percent": 10.0})
        self.assertEqual(per["ETH"], {"up_percent": 5.0, "down_percent": 9.0})

    def test_settings(self):
        self.assertEqual(self.ac["cooldown_minutes"], 45)
        self.assertEqual(self.cfg["max_new_coins"], 8)
        self.assertEqual(self.cfg["max_monitored"], 20)

    def test_malformed_value_fails_closed(self):
        with self.assertRaises(ValueError):
            load_config(_tmp_conf(SAMPLE_ALERTS_CONF.replace("lookback_hours = 24", "lookback_hours = bad")))

    def test_resolution_precedence(self):
        cases = (
            ("per-coin", "BTC", False, "up_percent", 6.0),
            ("per-coin-pair", "BTCUSDC", False, "up_percent", 6.0),
            ("default", "SOL", False, "up_percent", 4.1),
            ("dynamic", "FOONEW", True, "down_percent", 25.0),
            ("per-coin-over-dynamic", "BTC", True, "up_percent", 6.0),
        )
        for label, symbol, dynamic, field, expected in cases:
            with self.subTest(case=label):
                self.assertEqual(resolve(self.ac, symbol, is_dynamic=dynamic)[field], expected)
        checker = PriceChecker(DummyPriceManager(), config=self.ac)
        self.assertEqual(checker._get_thresholds_for_symbol("BTCUSDC")["up_percent"], 6.0)
        self.assertEqual(checker._get_thresholds_for_symbol("BTCUSDC")["down_percent"], 10.0)

    def test_missing_file_or_key_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            load_config("/does/not/exist.conf")
        with self.assertRaisesRegex(ValueError, "max_monitored"):
            load_config(_tmp_conf(SAMPLE_ALERTS_CONF.replace("max_monitored = 20\n", "")))


class TestAlertThresholdsModel(unittest.TestCase):
    def test_flat_series_returns_floor(self):
        up, down, upm, downm = thresholds_from_prices([100] * 40)
        self.assertEqual(up, FLOOR)
        self.assertEqual(down, FLOOR)
        self.assertEqual(upm, 0.0)

    def test_volatile_market_gives_wide_thresholds(self):
        prices = [100, 90, 115, 88, 130, 85, 120] * 6
        up, down, _, _ = thresholds_from_prices(prices, window=4)
        self.assertGreater(up, FLOOR)
        self.assertGreater(down, FLOOR)

    def test_higher_volatility_increases_threshold(self):
        calm = [100 + (i % 2) for i in range(60)]
        wild = [100 + (i % 2) * 30 for i in range(60)]
        up_calm = thresholds_from_prices(calm, window=6)[0]
        up_wild = thresholds_from_prices(wild, window=6)[0]
        self.assertGreater(up_wild, up_calm)

    def test_insufficient_data_returns_none(self):
        self.assertIsNone(thresholds_from_prices([100, 101], window=24))


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
        monitor.register_alerts_callback(received_alerts.extend)

        test_coins = [
            {
                "symbol": "FOO",
                "name": "Foo Token",
                "added_at": None,
                "price": 1.23,
                "url": "https://coinmarketcap.com/currencies/foo/",
            }
        ]
        monitor._trigger_alerts(test_coins, source_name="coinmarketcap", auto_add=False)

        self.assertEqual(len(received_alerts), 1)
        self.assertEqual(received_alerts[0]["symbol"], "FOO")
        self.assertEqual(received_alerts[0]["type"], "new_coin_discovered")
        self.assertEqual(received_alerts[0]["source"], "coinmarketcap")
        self.assertEqual(received_alerts[0]["price"], 1.23)

    def test_refresh_triggers_refresh_all_when_requested(self):
        factory = MagicMock(spec=NewCoinsFactory)
        factory.get_all_new_coins.return_value = {
            "coinmarketcap": [{"symbol": "BAR", "name": "Bar Token", "slug": "bar"}]
        }
        monitor = NewCoinsMonitor(factory=factory)
        self.assertIn("BAR", monitor.all_symbols)
        monitor.refresh(refresh_sources=True)
        factory.refresh_all.assert_called_once()

    def test_dexscreener_parses_list_response(self):
        from market_monitor.new_coins_discovery import DexScreenerSource
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "symbol": "DEXA",
                "name": "Dex Token A",
                "chainId": "solana",
                "tokenAddress": "addr123",
                "listedAt": int(time.time() * 1000),
            }
        ]
        with patch("market_monitor.new_coins_discovery.requests.get", return_value=mock_resp):
            src = DexScreenerSource()
        self.assertIn("DEXA", src.get_supported_symbols())
        self.assertEqual(len(src.get_new_coins()), 1)

    def test_factory_instantiates_only_enabled_sources(self):
        with patch("market_monitor.new_coins_discovery.CoinGeckoSource") as mock_cg, \
             patch("market_monitor.new_coins_discovery.DexScreenerSource") as mock_dex, \
             patch("market_monitor.new_coins_discovery.BinanceNewListingsSource") as mock_bin:
            mock_bin_inst = MagicMock()
            mock_bin_inst.is_available.return_value = True
            mock_bin.return_value = mock_bin_inst
            factory = NewCoinsFactory(enabled_sources=["binance"])
            mock_bin.assert_called_once()
            mock_cg.assert_not_called()
            mock_dex.assert_not_called()
            self.assertEqual(factory.get_available_sources(), ["binance"])

    def test_cleanup_old_symbols_purges_inactive_symbol_added_time(self):
        from market_monitor import pricefetcher
        pf_path = os.path.join(tempfile.mkdtemp(prefix="pf_clean_"), "cache_prices_multi.json")
        with patch.object(pricefetcher, "PricePlatformFactory"):
            mgr = pricefetcher.CacheAllPriceFetcherManager(60, ["BTC"], pf_path, read_only=True)
            mgr.symbol_added_time["OLDCOIN"] = time.time() - 30 * 86400
            mgr.symbol_preferred_source["OLDCOIN"] = "CoinMarketCap"
            self.assertNotIn("OLDCOIN", mgr.active_symbols)
            removed = mgr.cleanup_old_symbols(max_age_days=7)
            self.assertEqual(removed, ["OLDCOIN"])
            self.assertNotIn("OLDCOIN", mgr.symbol_added_time)
            self.assertNotIn("OLDCOIN", mgr.symbol_preferred_source)
            self.assertEqual(mgr.cleanup_old_symbols(max_age_days=7), [])


if __name__ == "__main__":
    unittest.main()
