import time
import unittest

from alertnotifiers import AlertNotifier
from pricechecker import PriceAlert, PriceChecker


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
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "alerts.log")
        self.assertTrue(AlertNotifier.save_to_file(self.NEW_COIN, filename=path))
        self.assertIn("NEW COIN RWS", open(path, encoding="utf-8").read())

    def test_save_to_file_handles_bot_event_and_dict(self):
        import tempfile, os
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
        # File channel active, no network — a dict must not raise
        AlertNotifier.send(self.NEW_COIN, enable_console=False, enable_file=True,
                           enable_email=False, enable_phone_webhook=False)

    def test_non_ascii_symbol_preserved(self):
        # A non-ASCII symbol (e.g. '小蝌蚪') must be preserved, not stripped
        coin = dict(self.NEW_COIN, symbol="小蝌蚪", name="小蝌蚪 Coin")
        self.assertEqual(AlertNotifier.alert_symbol(coin), "小蝌蚪")
        self.assertIn("小蝌蚪", AlertNotifier.format_batch_message([coin]))
        # UTF-8 header passthrough: UTF-8 bytes carried through latin-1, decodable back
        hdr = AlertNotifier.utf8_header("(1): 小蝌蚪")
        self.assertEqual(hdr.encode("latin-1").decode("utf-8"), "(1): 小蝌蚪")


if __name__ == "__main__":
    unittest.main()
