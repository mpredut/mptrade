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
            "NTFY_TOPIC_ERROR": "ntfy-error-test",
            "NTFY_TOPIC_PRICE": "ntfy-price-test",
            "NTFY_DAILY_BUDGET": "500",
            "NTFY_URGENT_RESERVE": "50",
            "NOTIFICATION_DEDUP_SECONDS": "0",
            "NOTIFICATION_PRICE_DEDUP_SECONDS": "0",
            "NOTIFICATION_STARTUP_DEDUP_SECONDS": "0",
            "NOTIFICATION_URGENT_DEDUP_SECONDS": "0",
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
        self.assertEqual(self.dispatched[0]["title"], "[Kraken-ADA] ADAUSD BUY 2725.09@0.24")
        self.assertEqual(self.dispatched[0]["topic"], "ntfy-trades-test")
        self.assertEqual(self.dispatched[1]["title"], "[Kraken-HYPE] HYPEUSD BUY 7.07@91.97")
        self.assertEqual(self.dispatched[1]["topic"], "ntfy-trades-test")
        self.assertEqual(self.dispatched[2]["title"], "[Kraken-TAO] TAOUSD BUY 2.26@287.30")
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

        # ERROR
        self.assertEqual(_topic_for_category("ORDER FAILED", "kraken"), "ntfy-error-test")
        self.assertEqual(_topic_for_category("CRITICAL EXCEPTION", "watchdog"), "ntfy-error-test")

        # PRICE
        self.assertEqual(_topic_for_category("BTC ▲ +5.20%", "price_alert"), "ntfy-price-test")
        self.assertEqual(_topic_for_category("Threshold reached", "pricechecker"), "ntfy-price-test")

    def test_price_alert_dispatch_formatting(self):
        """Verify PriceAlert objects and dicts create rich titles and bodies on the PRICE topic."""
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
        self.assertEqual(dispatched["title"], "[price_notifier] TAOUSDC ▲ +5.36%")
        self.assertIn("TAOUSDC: U +5.36%", dispatched["message"])
        self.assertIn("C $295.0000", dispatched["message"])
        self.assertIn("R $280.0000", dispatched["message"])
        self.assertEqual(dispatched["topic"], "ntfy-price-test")

    def test_new_coin_alert_dispatch(self):
        """Verify new_coin_discovered alerts route properly to the PRICE topic."""
        coin = {
            "type": "new_coin_discovered", "source": "coinmarketcap",
            "symbol": "RWS", "name": "Real World Services", "added_at": None,
            "price": 0.0154, "auto_added": True, "has_price": True,
        }
        self.server.dispatch_alerts([coin], bot_name="price_notifier")
        self.assertEqual(len(self.dispatched), 1)
        dispatched = self.dispatched[0]
        self.assertEqual(dispatched["title"], "[price_notifier] New Coin: RWS")
        self.assertIn("🆕: RWS - Real World Services", dispatched["message"])
        self.assertEqual(dispatched["topic"], "ntfy-price-test")

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
