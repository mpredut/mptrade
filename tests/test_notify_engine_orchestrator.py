"""Consolidated tests for NotificationServer intent parsing, guard & price routing,
log.py intent bypass, and test-environment isolation.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import log
from notify_engine.alertnotifiers import notify, AlertNotifier
from notify_engine.server import NotificationServer, _topic_for_category
from market_monitor.pricechecker import PriceAlert


class TestNotifyEngineOrchestrator(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.state_file = os.path.join(self._temp_dir.name, "delivery_state.json")
        self.patcher = mock.patch.dict(os.environ, {
            "NOTIFICATION_STATE_FILE": self.state_file,
            "NTFY_TOPIC_TRADES": "ntfy-trades-test",
            "NTFY_TOPIC_GUARD": "ntfy-guard-test",
            "NTFY_TOPIC_MACRO": "ntfy-macro-test",
            "NTFY_TOPIC_ERROR": "ntfy-error-test",
            "NTFY_TOPIC_PRICE": "ntfy-price-test",
            "NTFY_DAILY_BUDGET": "500",
            "NTFY_URGENT_RESERVE": "50",
            "NOTIFICATION_DEDUP_SECONDS": "0",
            "NOTIFICATION_PRICE_DEDUP_SECONDS": "0",
            "NOTIFICATION_STARTUP_DEDUP_SECONDS": "0",
            "NOTIFICATION_URGENT_DEDUP_SECONDS": "0",
            "NOTIFICATION_GUARD_DEDUP_SECONDS": "0",
            "DISABLE_EXTERNAL_NOTIFICATIONS": "0",  # Ensure server methods test dispatch
        })
        self.patcher.start()
        self.server = NotificationServer()
        self.dispatched = []
        self.server._send_ntfy = self._record_send

    def tearDown(self):
        self.patcher.stop()
        self._temp_dir.cleanup()

    def _record_send(self, title: str, message: str, priority: str, topic: str, is_retry: bool = False) -> bool:
        self.dispatched.append({
            "title": title,
            "message": message,
            "priority": priority,
            "topic": topic,
        })
        return True

    def test_process_line_resilience(self):
        """Test pure JSON, timestamp-prefixed, bot-prefixed, and malformed lines."""
        # 1. Timestamp prefixed line
        line1 = (
            '12:04:20 {"__orchestrator_intent__": "ntfy_webhook", "alerts": ['
            '{"type": "bot_event", "name": "ADAUSD BUY 2725.09@0.24", '
            '"body": "TREND_ENTRY | q2725.09 a0.24", "source": "kraken", "symbol": "ADAUSD"}]}'
        )
        self.server.process_line(line1, "Kraken-ADA")

        # 2. Bot prefixed line
        line2 = (
            '[Kraken-HYPE] 12:04:20 {"__orchestrator_intent__": "ntfy_webhook", "alerts": ['
            '{"type": "bot_event", "name": "HYPEUSD BUY 7.07@91.97", '
            '"body": "TREND_ENTRY | q7.07 a91.97", "source": "kraken", "symbol": "HYPEUSD"}]}\n'
        )
        self.server.process_line(line2, "Kraken-HYPE")

        # 3. Pure JSON line
        line3 = json.dumps({
            "__orchestrator_intent__": "ntfy_webhook",
            "alerts": [{
                "type": "bot_event",
                "name": "TAOUSD BUY 2.26@287.30",
                "body": "TREND_ENTRY | q2.26 a287.30",
                "source": "kraken",
                "symbol": "TAOUSD",
            }]
        })
        self.server.process_line(line3, "Kraken-TAO")

        self.assertEqual(len(self.dispatched), 3)
        self.assertEqual(self.dispatched[0]["title"], "[Kraken] ADAUSD BUY 2725.09@0.24")
        self.assertEqual(self.dispatched[0]["topic"], "ntfy-trades-test")
        self.assertEqual(self.dispatched[1]["title"], "[Kraken] HYPEUSD BUY 7.07@91.97")
        self.assertEqual(self.dispatched[1]["topic"], "ntfy-trades-test")
        self.assertEqual(self.dispatched[2]["title"], "[Kraken] TAOUSD BUY 2.26@287.30")
        self.assertEqual(self.dispatched[2]["topic"], "ntfy-trades-test")

        # 4. Malformed JSON should not crash
        with self.assertLogs("root", level="WARNING") as log_capture:
            self.server.process_line('12:04:20 {"__orchestrator_intent__": "ntfy_webhook", BROKEN', "Kraken-ADA")
        self.assertEqual(len(self.dispatched), 3)
        self.assertTrue(any("Failed to decode orchestrator intent JSON" in m for m in log_capture.output))

    def test_topic_routing_comprehensive(self):
        """Verify routing for TRADES, GUARD, ERROR, and PRICE categories."""
        # TRADES
        self.assertEqual(_topic_for_category("BUY 10@100", "kraken"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("SELL 5@200", "binance"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("TREND_ENTRY", "kraken"), "ntfy-trades-test")

        # GUARD
        self.assertEqual(_topic_for_category("🛑 STOP_LOSS triggered", "kraken"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("🛡 Guard active", "kraken"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("TRAILING SELL FILLED", "kraken-trail"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("LIQUIDATION IMMINENT", "hl_bot"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("🛑 order submission quarantined", "order_retry"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("Drawdown alert", "assetguardian"), "ntfy-guard-test")

        # MACRO
        self.assertEqual(_topic_for_category("🛡 [MACRO SHADOW VETO] Would Block BUY TAOUSDC", "macro_shadow"), "ntfy-macro-test")
        self.assertEqual(_topic_for_category("🛡 [MACRO SHADOW DOWNSCALE] Would Scale 50% BUY TAOUSDC", "order_guard"), "ntfy-macro-test")

        # ERROR
        self.assertEqual(_topic_for_category("ORDER FAILED", "kraken"), "ntfy-error-test")
        self.assertEqual(_topic_for_category("CRITICAL EXCEPTION", "watchdog"), "ntfy-error-test")

        # PRICE
        self.assertEqual(_topic_for_category("BTC ▲ +5.20%", "price_alert"), "ntfy-price-test")
        self.assertEqual(_topic_for_category("Threshold reached", "pricechecker"), "ntfy-price-test")

    def test_provider_label_resolution(self):
        """Verify provider resolution strips coins and formats providers cleanly."""
        from notify_engine.server import _resolve_provider_label
        self.assertEqual(_resolve_provider_label("Kraken-ADA", "kraken"), "Kraken")
        self.assertEqual(_resolve_provider_label("Kraken-TAO", "kraken"), "Kraken")
        self.assertEqual(_resolve_provider_label("HL-bot", "hyperliquid"), "Hyperliquid")
        self.assertEqual(_resolve_provider_label("T212-bot", "t212"), "T212")
        self.assertEqual(_resolve_provider_label("rtrade", "binance"), "Binance")
        self.assertEqual(_resolve_provider_label("tradeall", ""), "Binance")
        self.assertEqual(_resolve_provider_label("Binance-trailing", ""), "Binance")
        self.assertEqual(_resolve_provider_label("Custom-PAIR", ""), "Custom")

    def test_price_alert_dispatch_formatting(self):
        """Verify PriceAlert objects and dicts create rich titles without [price_notifier] prefix."""
        alert = PriceAlert(
            symbol="TAOUSDC",
            alert_type="up",
            current_price=295.0,
            reference_price=280.0,
            percent_change=5.36,
            threshold=5.0,
            reference_time="2026-09-24 14:30:00",
        )
        self.server.dispatch_alerts([alert], bot_name="price_notifier")
        self.assertEqual(len(self.dispatched), 1)
        dispatched = self.dispatched[0]
        self.assertEqual(dispatched["title"], "TAOUSDC ▲ +5.36%")
        self.assertIn("TAOUSDC: U +5.36%", dispatched["message"])
        self.assertIn("C $295.0000", dispatched["message"])
        self.assertIn("R $280.0000", dispatched["message"])
        self.assertEqual(dispatched["topic"], "ntfy-price-test")

    def test_new_coin_alert_dispatch(self):
        """Verify new_coin_discovered alerts route to PRICE without [price_notifier] prefix."""
        coin = {
            "type": "new_coin_discovered", "source": "coinmarketcap",
            "symbol": "RWS", "name": "Real World Services", "added_at": None,
            "price": 0.0154, "auto_added": True, "has_price": True,
        }
        self.server.dispatch_alerts([coin], bot_name="price_notifier")
        self.assertEqual(len(self.dispatched), 1)
        dispatched = self.dispatched[0]
        self.assertEqual(dispatched["title"], "New Coin: RWS")
        self.assertIn("🆕: RWS - Real World Services", dispatched["message"])
        self.assertEqual(dispatched["topic"], "ntfy-price-test")

    def test_batch_alerts_dispatch_formatting(self):
        """Verify multi-coin batch alerts include all coins in title rather than only the first."""
        alert1 = PriceAlert("TAOUSDC", "up", 295.0, 280.0, 5.36, 5.0)
        alert2 = PriceAlert("BTCUSDC", "down", 84000.0, 87600.0, -4.11, 4.0)

        # Batch of 2 price alerts
        self.dispatched.clear()
        self.server.dispatch_alerts([alert1, alert2], bot_name="price_notifier")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "Price Alerts (2): TAOUSDC ▲, BTCUSDC ▼")
        self.assertIn("TAOUSDC: U +5.36%", self.dispatched[0]["message"])
        self.assertIn("BTCUSDC: D -4.11%", self.dispatched[0]["message"])

        # Batch of many price alerts exceeding 50 chars falls back to count
        self.dispatched.clear()
        many_alerts = [
            PriceAlert(f"COIN{i}USDC", "up" if i % 2 == 0 else "down", 100.0, 95.0, 5.0, 4.0)
            for i in range(10)
        ]
        self.server.dispatch_alerts(many_alerts, bot_name="price_notifier")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "Price Alerts (10 coins)")

        # Batch of multiple new coins
        self.dispatched.clear()
        coins = [
            {"type": "new_coin_discovered", "source": "cmc", "symbol": "COINA"},
            {"type": "new_coin_discovered", "source": "cmc", "symbol": "COINB"},
        ]
        self.server.dispatch_alerts(coins, bot_name="price_notifier")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "New Coins (2): COINA, COINB")

    def test_fake_and_test_symbols_are_blocked_from_delivery(self):
        """Verify test instruments like ZZZFAKEUSD are blocked before network dispatch."""
        test_alert = {
            "type": "bot_event",
            "name": "🛑 order submission quarantined BUY ZZZFAKEUSD",
            "body": "Test retry intent quarantine",
            "source": "order_retry",
            "symbol": "ZZZFAKEUSD",
        }
        self.server.dispatch_alerts([test_alert], bot_name="order_retry")
        self.assertEqual(len(self.dispatched), 0, "Fake/test symbols must never be dispatched to ntfy!")

    def test_disable_external_notifications_kill_switch(self):
        """Verify DISABLE_EXTERNAL_NOTIFICATIONS=1 blocks all external HTTP dispatch."""
        with mock.patch.dict(os.environ, {"DISABLE_EXTERNAL_NOTIFICATIONS": "1"}):
            real_server = NotificationServer()
            with mock.patch("requests.post") as mock_post:
                ok = real_server._send_ntfy("Live Title", "Live Message", "high", "live-topic")
                self.assertTrue(ok)
                mock_post.assert_not_called()

            # notify() wrapper isolation
            with mock.patch("sys.stdout.write") as mock_stdout:
                notify(title="Test", body="body", source="live", symbol="BTC")
                mock_stdout.assert_not_called()

    def test_serialization_and_log_print_bypass(self):
        """Verify log.py print bypass and direct stdout writing for orchestrator IPC."""
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print('{"__orchestrator_intent__": "ntfy_webhook", "alerts": []}')
        finally:
            sys.stdout = old_stdout
        self.assertTrue(buf.getvalue().startswith('{"__orchestrator_intent__"'))

        with mock.patch.dict(os.environ, {"MPTRADE_ORCHESTRATED": "1"}):
            buf2 = io.StringIO()
            sys.stdout = buf2
            try:
                notify(title="BTC BUY", body="Details", source="kraken", symbol="BTC")
            finally:
                sys.stdout = old_stdout
            payload = json.loads(buf2.getvalue().strip())
            self.assertEqual(payload["alerts"][0]["name"], "BTC BUY")

    def test_multiline_traceback_buffering(self):
        """Verify multi-line Python tracebacks are buffered until the final exception line."""
        tb_lines = [
            "Traceback (most recent call last):\n",
            '  File "intelligence/daemon.py", line 120, in run_cycle\n',
            "    resp = requests.get(url)\n",
            "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='news.google.com'): Read timed out.\n",
        ]
        for line in tb_lines:
            self.server.process_line(line, "intelligence_daemon")

        self.assertEqual(len(self.dispatched), 1)
        alert = self.dispatched[0]
        self.assertEqual(alert["title"], "[intelligence_daemon] Critical Errors")
        self.assertIn("Traceback (most recent call last):", alert["message"])
        self.assertIn("line 120", alert["message"])
        self.assertIn("ReadTimeout", alert["message"])
        self.assertEqual(alert["priority"], "urgent")

    def test_incomplete_traceback_flushed_on_new_log(self):
        """Verify incomplete traceback is dispatched if a subsequent log line arrives."""
        self.server.process_line("Traceback (most recent call last):\n", "bot_a")
        self.server.process_line('  File "bot.py", line 10, in foo\n', "bot_a")
        self.assertEqual(len(self.dispatched), 0)

        # Subsequent log line arrives
        self.server.process_line("2026-10-04 12:00:00 [INFO] Bot restarted\n", "bot_a")
        self.assertEqual(len(self.dispatched), 1)
        self.assertIn("Traceback (most recent call last):", self.dispatched[0]["message"])

    def test_guard_alert_deduplication_and_stable_fingerprint(self):
        """Verify that guard veto alerts containing BUY/SELL are deduplicated across cycles."""
        with mock.patch.dict(os.environ, {"NOTIFICATION_GUARD_DEDUP_SECONDS": "1800"}):
            alert1 = {
                "type": "bot_event",
                "name": "🛡 [MACRO SHADOW VETO] Would Block BUY TAOUSDC",
                "body": "Macro shock flagged: threat=CRITICAL_SHOCK, risk=0.95",
                "source": "order_guard",
                "symbol": "TAOUSDC",
            }
            alert2 = {
                "type": "bot_event",
                "name": "🛡 [MACRO SHADOW VETO] Would Block BUY TAOUSDC",
                "body": "Macro shock flagged: threat=CRITICAL_SHOCK, risk=0.92 (wording variation)",
                "source": "order_guard",
                "symbol": "TAOUSDC",
            }

            # First delivery should succeed
            self.dispatched.clear()
            self.server.dispatch_alerts([alert1], bot_name="rtrade")
            self.assertEqual(len(self.dispatched), 1)
            self.assertEqual(self.dispatched[0]["title"], "[rtrade] 🛡 [MACRO SHADOW VETO] Would Block BUY TAOUSDC")
            self.assertEqual(self.dispatched[0]["topic"], "ntfy-macro-test")

            # Second delivery within 1800s with slight body variation must be deduplicated
            self.server.dispatch_alerts([alert2], bot_name="rtrade")
            self.assertEqual(len(self.dispatched), 1, "Guard alert should be deduplicated despite BUY keyword")


