"""Tests for NotificationServer orchestrator intent parsing, log.py intent bypass,
and alertnotifiers stdout serialization.
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from unittest import mock

import log
from notify_engine.alertnotifiers import notify
from notify_engine.server import NotificationServer, _topic_for_category


class TestNotifyEngineOrchestrator(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch.dict(os.environ, {
            "NTFY_TOPIC_TRADES": "ntfy-trades-test",
            "NTFY_TOPIC_GUARD": "ntfy-guard-test",
            "NTFY_TOPIC_ERROR": "ntfy-error-test",
            "NTFY_TOPIC_PRICE": "ntfy-price-test",
            "NTFY_DAILY_BUDGET": "500",
            "NTFY_URGENT_RESERVE": "50",
            "NOTIFICATION_DEDUP_SECONDS": "0",  # Disable dedup for test runs
        })
        self.patcher.start()
        self.server = NotificationServer()
        self.dispatched = []
        self.server._send_ntfy = self._record_send

    def tearDown(self):
        self.patcher.stop()

    def _record_send(self, title: str, message: str, priority: str, topic: str, is_retry: bool = False) -> bool:
        self.dispatched.append({
            "title": title,
            "message": message,
            "priority": priority,
            "topic": topic,
        })
        return True

    def test_process_line_with_timestamp_prefix(self):
        line = (
            '12:04:20 {"__orchestrator_intent__": "ntfy_webhook", "alerts": ['
            '{"type": "bot_event", "name": "ADAUSD BUY 2725.09@0.24", '
            '"body": "TREND_ENTRY | q2725.09 a0.24 | desf650USD", '
            '"source": "kraken", "symbol": "Kraken"}]}'
        )
        self.server.process_line(line, "Kraken-ADA")
        self.assertEqual(len(self.dispatched), 1)
        alert = self.dispatched[0]
        self.assertEqual(alert["title"], "[Kraken-ADA] ADAUSD BUY 2725.09@0.24")
        self.assertEqual(alert["message"], "TREND_ENTRY | q2725.09 a0.24 | desf650USD")
        self.assertEqual(alert["topic"], "ntfy-trades-test")

    def test_process_line_with_bot_prefix(self):
        line = (
            '[Kraken-HYPE] 12:04:20 {"__orchestrator_intent__": "ntfy_webhook", "alerts": ['
            '{"type": "bot_event", "name": "HYPEUSD BUY 7.07@91.97", '
            '"body": "TREND_ENTRY | q7.07 a91.97 | desf650USD", '
            '"source": "kraken", "symbol": "Kraken"}]}\n'
        )
        self.server.process_line(line, "Kraken-HYPE")
        self.assertEqual(len(self.dispatched), 1)
        alert = self.dispatched[0]
        self.assertEqual(alert["title"], "[Kraken-HYPE] HYPEUSD BUY 7.07@91.97")
        self.assertEqual(alert["topic"], "ntfy-trades-test")

    def test_process_line_pure_json(self):
        line = json.dumps({
            "__orchestrator_intent__": "ntfy_webhook",
            "alerts": [{
                "type": "bot_event",
                "name": "TAOUSD BUY 2.26@287.30",
                "body": "TREND_ENTRY | q2.26 a287.30",
                "source": "kraken",
                "symbol": "Kraken",
            }]
        })
        self.server.process_line(line, "Kraken-TAO")
        self.assertEqual(len(self.dispatched), 1)
        alert = self.dispatched[0]
        self.assertEqual(alert["title"], "[Kraken-TAO] TAOUSD BUY 2.26@287.30")
        self.assertEqual(alert["topic"], "ntfy-trades-test")

    def test_process_line_malformed_json_logs_warning_and_does_not_crash(self):
        line = '12:04:20 {"__orchestrator_intent__": "ntfy_webhook", INVALID JSON'
        with self.assertLogs("root", level="WARNING") as log_capture:
            self.server.process_line(line, "Kraken-ADA")
        self.assertEqual(len(self.dispatched), 0)
        self.assertTrue(any("Failed to decode orchestrator intent JSON" in m for m in log_capture.output))

    def test_topic_routing_by_category(self):
        self.assertEqual(_topic_for_category("BUY 10@100", "kraken"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("🛑 STOP_LOSS triggered", "kraken"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("ORDER FAILED", "kraken"), "ntfy-error-test")
        self.assertEqual(_topic_for_category("Threshold reached", "price_alert"), "ntfy-price-test")

    def test_log_patched_print_does_not_prefix_orchestrator_intent(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print('{"__orchestrator_intent__": "ntfy_webhook", "alerts": []}')
        finally:
            sys.stdout = old_stdout
        out = buf.getvalue()
        self.assertTrue(out.startswith('{"__orchestrator_intent__"'))

    def test_notify_serializes_clean_json_when_orchestrated(self):
        with mock.patch.dict(os.environ, {"MPTRADE_ORCHESTRATED": "1"}):
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                notify(
                    title="TEST BUY",
                    body="Body details",
                    source="kraken",
                    symbol="BTC",
                )
            finally:
                sys.stdout = old_stdout

            out = buf.getvalue()
            payload = json.loads(out.strip())
            self.assertEqual(payload.get("__orchestrator_intent__"), "ntfy_webhook")
            self.assertEqual(payload["alerts"][0]["name"], "TEST BUY")
            self.assertEqual(payload["alerts"][0]["body"], "Body details")
