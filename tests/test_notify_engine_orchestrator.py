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

from notify_engine.alertnotifiers import AlertNotifier, notify
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
            "NTFY_TOPIC_SERVER": "ntfy-server-test",
            "NTFY_TOPIC_DEADMAN": "ntfy-deadman-test",
            "NTFY_DAILY_BUDGET": "500",
            "NTFY_URGENT_RESERVE": "50",
            "NOTIFICATION_DEDUP_SECONDS": "0",
            "NOTIFICATION_PRICE_DEDUP_SECONDS": "0",
            "NOTIFICATION_STARTUP_DEDUP_SECONDS": "0",
            "NOTIFICATION_URGENT_DEDUP_SECONDS": "0",
            "NOTIFICATION_GUARD_DEDUP_SECONDS": "0",
            "NOTIFICATION_GATE_SECONDS": "0",
            "NOTIFICATION_TOPIC_BULK_SECONDS": "0",
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
        """Verify routing for TRADES, GUARD, ERROR, SERVER, and PRICE categories."""
        # TRADES (including launched orders, manual attribution, and fills)
        self.assertEqual(_topic_for_category("BUY 10@100", "kraken"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("SELL 5@200", "binance"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("TREND_ENTRY", "kraken"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("📝 SELL TAOUSDC @ 284.70 [manual]", "monitororder"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("📝 BUY TAOUSDC @ 283.92 [manual]", "monitororder"), "ntfy-trades-test")
        self.assertEqual(_topic_for_category("🎉 [Binance] Order Executed: BUY TAOUSDC", "monitortrades"), "ntfy-trades-test")

        # GUARD
        self.assertEqual(_topic_for_category("🛑 STOP_LOSS triggered", "kraken"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("🛡 Guard active", "kraken"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("TRAILING SELL FILLED", "kraken-trail"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("📉 TRAILING NVDA -5.2%", "T212"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("LIQUIDATION IMMINENT", "hl_bot"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("🛑 order submission quarantined", "order_retry"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("Drawdown alert", "assetguardian"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("⚠️ [auto_sharetrade_executor] REFUSED BUY NVDA_US_EQ", "auto_sharetrade_executor"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("🛡 [auto_cryptotrade_executor] BLOCKED BUY BTCUSDC", "auto_cryptotrade_executor"), "ntfy-guard-test")

        # MACRO
        self.assertEqual(_topic_for_category("🛡 [VETO] Would Block BUY TAOUSDC", "macro_shadow"), "ntfy-macro-test")
        self.assertEqual(_topic_for_category("🛡 [DOWNSCALE] Would Scale 50% BUY TAOUSDC", "macro_shadow"), "ntfy-macro-test")

        # SERVER / DEADMAN
        self.assertEqual(_topic_for_category("Config Reloaded", "orchestrator"), "ntfy-server-test")

        # ERROR
        self.assertEqual(_topic_for_category("ORDER FAILED", "kraken"), "ntfy-error-test")
        self.assertEqual(_topic_for_category("CRITICAL EXCEPTION", "watchdog"), "ntfy-error-test")
        self.assertEqual(_topic_for_category("MANUAL INTERVENTION REQUIRED", "order_retry"), "ntfy-error-test")
        self.assertEqual(_topic_for_category("ORDER FAILED: BUY BTC [manual]", "binance"), "ntfy-error-test")

        # PRICE
        self.assertEqual(_topic_for_category("BTC ▲ +5.20%", "price_alert"), "ntfy-price-test")
        self.assertEqual(_topic_for_category("Threshold reached", "pricechecker"), "ntfy-price-test")
        self.assertEqual(_topic_for_category("📉 NVDA -12.1%", "T212"), "ntfy-price-test")
        self.assertEqual(_topic_for_category("📈 xStock +25.0% above the allocation (250)", "xstock-watch"), "ntfy-price-test")

    def test_manual_order_dispatch_formatting_and_topic(self):
        """Verify manual trade orders route to TRADES with [Binance] prefix, not ERROR topic."""
        line = (
            '{"__orchestrator_intent__": "ntfy_webhook", "alerts": [{'
            '"type": "bot_event", "name": "📝 SELL TAOUSDC @ 284.70 [manual]", '
            '"body": "Order 12345 (manual) placed on venue.", "source": "monitororder", "symbol": "TAOUSDC"}]}'
        )
        self.server.process_line(line, "monitororder")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "[Binance] 📝 SELL TAOUSDC @ 284.70 [manual]")
        self.assertEqual(self.dispatched[0]["topic"], "ntfy-trades-test")

    def test_server_topic_requires_canonical_key(self):
        """Verify resolve_topic('SERVER') requires NTFY_TOPIC_SERVER and does not fall back to aliases."""
        from notify_engine import server
        with (
            mock.patch.dict(os.environ, {"NTFY_TOPIC_SERVER": "", "NTFY_TOPIC_DEADMAN": "fallback-deadman"}),
            mock.patch.object(server, "_ensure_env_loaded", lambda: None),
        ):
            with self.assertRaisesRegex(ValueError, "NTFY_TOPIC_SERVER"):
                server.resolve_topic("SERVER")

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
        """Verify PriceAlert objects and dicts create rich titles without [price_watcher] prefix."""
        alert = PriceAlert(
            symbol="TAOUSDC",
            alert_type="up",
            current_price=295.0,
            reference_price=280.0,
            percent_change=5.36,
            threshold=5.0,
            reference_time="2026-09-24 14:30:00",
        )
        self.server.dispatch_alerts([alert], bot_name="price_watcher")
        self.assertEqual(len(self.dispatched), 1)
        dispatched = self.dispatched[0]
        self.assertEqual(dispatched["title"], "TAOUSDC ▲ +5.36%")
        self.assertIn("TAOUSDC: U +5.36%", dispatched["message"])
        self.assertIn("C $295.0000", dispatched["message"])
        self.assertIn("R $280.0000", dispatched["message"])
        self.assertEqual(dispatched["topic"], "ntfy-price-test")

    def test_new_coin_alert_dispatch(self):
        """Verify new_coin_discovered alerts route to PRICE without [price_watcher] prefix."""
        coin = {
            "type": "new_coin_discovered", "source": "coinmarketcap",
            "symbol": "RWS", "name": "Real World Services", "added_at": None,
            "price": 0.0154, "auto_added": True, "has_price": True,
        }
        self.server.dispatch_alerts([coin], bot_name="price_watcher")
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
        self.server.dispatch_alerts([alert1, alert2], bot_name="price_watcher")
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
        self.server.dispatch_alerts(many_alerts, bot_name="price_watcher")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "Price Alerts (10 coins)")

        # Batch of multiple new coins
        self.dispatched.clear()
        coins = [
            {"type": "new_coin_discovered", "source": "cmc", "symbol": "COINA"},
            {"type": "new_coin_discovered", "source": "cmc", "symbol": "COINB"},
        ]
        self.server.dispatch_alerts(coins, bot_name="price_watcher")
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
                "name": "🛡 [VETO] Would Block BUY TAOUSDC",
                "body": "Macro shock flagged: threat=CRITICAL_SHOCK, risk=0.95",
                "source": "macro_shadow",
                "symbol": "TAOUSDC",
            }
            alert2 = {
                "type": "bot_event",
                "name": "🛡 [VETO] Would Block BUY TAOUSDC",
                "body": "Macro shock flagged: threat=CRITICAL_SHOCK, risk=0.92 (wording variation)",
                "source": "macro_shadow",
                "symbol": "TAOUSDC",
            }

            # First delivery should succeed
            self.dispatched.clear()
            self.server.dispatch_alerts([alert1], bot_name="rtrade")
            self.assertEqual(len(self.dispatched), 1)
            self.assertEqual(self.dispatched[0]["title"], "[rtrade] 🛡 [VETO] Would Block BUY TAOUSDC")
            self.assertEqual(self.dispatched[0]["topic"], "ntfy-macro-test")

            # Second delivery within 1800s with slight body variation must be deduplicated
            self.server.dispatch_alerts([alert2], bot_name="rtrade")
            self.assertEqual(len(self.dispatched), 1, "Guard alert should be deduplicated despite BUY keyword")

    def test_macro_analyzer_and_auto_topic_alerts_omit_bot_name_prefix(self):
        """Verify macro_analyzer and [auto_...] topic notifications do not prepend [macro_analyzer] or [bot_name]."""
        self.dispatched.clear()
        sentiment_alert = {
            "type": "bot_event",
            "name": "[macro_analyzer] 🔴 [auto_macro_sentiment] DEFENSIVE (BEARISH)",
            "body": "Action: DEFENSIVE · Bias: BEARISH (Risk: HIGH)",
            "source": "macro_shadow",
            "symbol": "GLOBAL",
        }
        self.server.dispatch_alerts([sentiment_alert], bot_name="macro_analyzer")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "🔴 [auto_macro_sentiment] DEFENSIVE (BEARISH)")
        self.assertEqual(self.dispatched[0]["topic"], "ntfy-macro-test")

        self.dispatched.clear()
        trade_alert = {
            "type": "bot_event",
            "name": "🚀 [auto_trade_executor] EXECUTED BUY TAOUSDC $250",
            "body": "Venue: binance · Qty: 1 @ $250.00",
            "source": "llm_trader",
            "symbol": "TAOUSDC",
        }
        self.server.dispatch_alerts([trade_alert], bot_name="ai_reconciler")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["title"], "🚀 [auto_trade_executor] EXECUTED BUY TAOUSDC $250")
        self.assertEqual(self.dispatched[0]["topic"], "ntfy-trades-test")

    def test_flush_queue_preserves_concurrent_appends_and_deduplicates_delayed_prefix(self):
        """Verify flush_queue_loop preserves concurrently appended queue lines and avoids duplicate [DELAYED] prefix."""
        import asyncio

        queue_file = os.path.join(self._temp_dir.name, "queue.jsonl")
        self.server.queue_file = queue_file
        initial_line = json.dumps({
            "__orchestrator_intent__": "email",
            "subject": "[DELAYED] First Alert",
            "message": "Body 1",
        })
        concurrent_line = json.dumps({
            "__orchestrator_intent__": "ntfy_webhook",
            "title": "Concurrent Alert",
            "message": "Body 2",
            "priority": "high",
            "topic": "ntfy-trades-test",
        })
        with open(queue_file, "w", encoding="utf-8") as f:
            f.write(initial_line + "\n")

        sent_emails = []

        def fake_send_email(subject: str, message: str, is_retry: bool = False) -> bool:
            sent_emails.append((subject, message, is_retry))
            # Simulate concurrent append while network I/O is in flight
            with open(queue_file, "a", encoding="utf-8") as f_append:
                f_append.write(concurrent_line + "\n")
            return True

        self.server._send_email = fake_send_email

        calls = 0

        async def fake_sleep(_sec):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise asyncio.CancelledError()

        with mock.patch("asyncio.sleep", side_effect=fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(self.server.flush_queue_loop())

        self.assertEqual(len(sent_emails), 1)
        self.assertEqual(sent_emails[0][0], "[DELAYED] First Alert")
        with open(queue_file, "r", encoding="utf-8") as f:
            remaining_lines = [l.strip() for l in f.readlines() if l.strip()]
        self.assertEqual(remaining_lines, [concurrent_line])

    def test_error_and_ops_alerts_deduplicate_despite_buy_sell_and_dynamic_body(self):
        """Verify error/expired/rejected/guard alerts with BUY/SELL and varying bodies or prices are deduplicated."""
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_ERROR_DEDUP_SECONDS": "1800",
            "NOTIFICATION_URGENT_DEDUP_SECONDS": "1800",
            "NOTIFICATION_GUARD_DEDUP_SECONDS": "1800",
        }):
            # 1. ORDER FAILED containing BUY and varying timestamp/error details in body
            err1 = {
                "type": "bot_event",
                "name": "ORDER FAILED: BUY BTCUSDC",
                "body": "2026-10-19 12:00:01 APIError(code=-2010): Account has insufficient balance",
                "source": "rtrade",
                "symbol": "BTCUSDC",
            }
            err2 = {
                "type": "bot_event",
                "name": "ORDER FAILED: BUY BTCUSDC",
                "body": "2026-10-19 12:00:05 APIError(code=-2010): Account has insufficient balance (retry 2)",
                "source": "rtrade",
                "symbol": "BTCUSDC",
            }
            self.dispatched.clear()
            self.server.dispatch_alerts([err1], bot_name="rtrade")
            self.assertEqual(len(self.dispatched), 1)
            self.assertEqual(self.dispatched[0]["topic"], "ntfy-error-test")

            self.server.dispatch_alerts([err2], bot_name="rtrade")
            self.assertEqual(len(self.dispatched), 1, "Repeated ORDER FAILED with BUY must be paused by dedup gate")

            # 2. Order Expired & Canceled containing BUY and dynamic price in title
            exp1 = {
                "type": "bot_event",
                "name": "⏰ Order Expired & Canceled: BUY TAOUSDC @ 284.50 [rtrade]",
                "body": "Order 111 expired after 15.0m",
                "source": "monitororder",
                "symbol": "TAOUSDC",
            }
            exp2 = {
                "type": "bot_event",
                "name": "⏰ Order Expired & Canceled: BUY TAOUSDC @ 285.10 [rtrade]",
                "body": "Order 112 expired after 15.1m",
                "source": "monitororder",
                "symbol": "TAOUSDC",
            }
            self.server.dispatch_alerts([exp1], bot_name="monitororder")
            self.assertEqual(len(self.dispatched), 2)
            self.server.dispatch_alerts([exp2], bot_name="monitororder")
            self.assertEqual(len(self.dispatched), 2, "Repeated expired order alerts for same symbol/side must be paused")

            # 3. Genuine trade execution with BUY still has 0s dedup (never blocked)
            fill1 = {
                "type": "bot_event",
                "name": "🎉 [Binance] Order Executed: BUY BTCUSDC @ $65000.00",
                "body": "Fill 1",
                "source": "monitortrades",
                "symbol": "BTCUSDC",
            }
            fill2 = {
                "type": "bot_event",
                "name": "🎉 [Binance] Order Executed: BUY BTCUSDC @ $65100.00",
                "body": "Fill 2",
                "source": "monitortrades",
                "symbol": "BTCUSDC",
            }
            self.server.dispatch_alerts([fill1], bot_name="monitortrades")
            self.server.dispatch_alerts([fill2], bot_name="monitortrades")
            self.assertEqual(len(self.dispatched), 4, "Real trade executions must never be suppressed")

    def test_process_line_rules_and_ntfy_tag_respect_pause_gate_and_ignore_transport_errors(self):
        """Verify [NTFY] tags and regex rules use persistent _reserve_delivery and ignore internal notify errors."""
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_ERROR_DEDUP_SECONDS": "1800",
            "NOTIFICATION_URGENT_DEDUP_SECONDS": "1800",
        }):
            self.dispatched.clear()
            # 1. Internal notification transport errors must be ignored to prevent feedback loops
            self.server.process_line(
                "2026-10-19 12:00:00 [ERROR] Failed to send ntfy alert (Critical Errors): 502 Bad Gateway\n",
                "orchestrator",
            )
            self.server.process_line(
                "2026-10-19 12:00:01 [ERROR] Email delivery failed: ConnectionResetError\n",
                "orchestrator",
            )
            self.assertEqual(len(self.dispatched), 0)

            # 2. [NTFY] shortcut is gated across repeated identical error messages
            self.server.process_line("[NTFY] Connection lost to upstream feed\n", "rtrade")
            self.server.process_line("[NTFY] Connection lost to upstream feed\n", "rtrade")
            self.assertEqual(len(self.dispatched), 1)

            # 3. Critical Errors regex rule is gated by persistent _reserve_delivery even if RAM cooldown is cleared
            self.server.process_line("2026-10-19 12:01:00 [ERROR] database lock timeout 1\n", "rtrade")
            self.assertEqual(len(self.dispatched), 2)
            self.server.cooldowns.clear()  # Simulate orchestrator restart clearing RAM cooldowns
            self.server.process_line("2026-10-19 12:02:00 [ERROR] database lock timeout 2\n", "rtrade")
            self.assertEqual(len(self.dispatched), 2, "Persistent delivery state must block Critical Errors after restart")

    def test_holistic_5min_common_gate_eliminates_similar_and_synthesizes_bulk(self):
        """Verify the 5-minute common gate:
        1. Sends the first notification immediately and closes the 5-minute gate.
        2. Eliminates structurally similar notifications arriving at short intervals.
        3. Buffers distinct notifications arriving within the 5-minute window and
           synthesizes/concatenates them into a single bulk message once 5 minutes elapse.
        """
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_GATE_SECONDS": "300",
            "NOTIFICATION_ERROR_DEDUP_SECONDS": "1800",
            "NOTIFICATION_GUARD_DEDUP_SECONDS": "1800",
        }):
            self.dispatched.clear()
            t0 = 1_700_000_000.0

            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0):
                # 1. First alert opens and immediately closes the 5-minute gate
                hung_1 = {
                    "type": "bot_event",
                    "name": "Alert",
                    "body": (
                        "Process globaltelemetry_collector (PID 613438) hung: "
                        "heartbeat globaltelemetry_collector.heartbeat stale by 28784s > 180s.\n"
                        "Last logs:\n  > 2026-10-19 05:10:46 [WARNING] slow query"
                    ),
                    "source": "orchestrator",
                    "symbol": "SYSTEM",
                }
                self.server.dispatch_alerts([hung_1], bot_name="orchestrator")
                self.assertEqual(len(self.dispatched), 1)
                self.assertEqual(self.dispatched[0]["title"], "[orchestrator] Alert")

            # 2. 15 seconds later: structurally similar hung alert (different PID, stale seconds, and logs)
            # Must be ELIMINATED completely (neither sent nor buffered as a new bulk item)
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 15):
                hung_2 = {
                    "type": "bot_event",
                    "name": "Alert",
                    "body": (
                        "Process globaltelemetry_collector (PID 699999) hung: "
                        "heartbeat globaltelemetry_collector.heartbeat stale by 28900s > 180s.\n"
                        "Last logs:\n  > 2026-10-19 05:11:01 [WARNING] another line"
                    ),
                    "source": "orchestrator",
                    "symbol": "SYSTEM",
                }
                self.server.dispatch_alerts([hung_2], bot_name="orchestrator")
                self.assertEqual(len(self.dispatched), 1, "Similar hung message must be eliminated")

            # 3. 30 seconds later: DIFFERENT alert #1 arrives while 5-min gate is closed -> buffered
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 30):
                diff_1 = {
                    "type": "bot_event",
                    "name": "⏰ Order Expired & Canceled: BUY TAOUSDC @ 284.50 [rtrade]",
                    "body": "Order 111 expired after 15.0m",
                    "source": "monitororder",
                    "symbol": "TAOUSDC",
                }
                self.server.dispatch_alerts([diff_1], bot_name="monitororder")
                self.assertEqual(len(self.dispatched), 1, "Different alert within 5m gate must be buffered for bulk")

            # 4. 45 seconds later: SIMILAR repeat of buffered alert #1 (different price & order ID)
            # Must be eliminated and increment repeat count (×2) on buffered item #1
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 45):
                diff_1_repeat = {
                    "type": "bot_event",
                    "name": "⏰ Order Expired & Canceled: BUY TAOUSDC @ 285.10 [rtrade]",
                    "body": "Order 112 expired after 15.1m",
                    "source": "monitororder",
                    "symbol": "TAOUSDC",
                }
                self.server.dispatch_alerts([diff_1_repeat], bot_name="monitororder")
                self.assertEqual(len(self.dispatched), 1)

            # 5. 60 seconds later: DIFFERENT alert #2 arrives -> buffered
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 60):
                diff_2 = {
                    "type": "bot_event",
                    "name": "🛡 [VETO] Would Block BUY ETHUSDC",
                    "body": "Macro shock flagged: threat=CRITICAL_SHOCK, risk=0.95",
                    "source": "macro_shadow",
                    "symbol": "ETHUSDC",
                }
                self.server.dispatch_alerts([diff_2], bot_name="rtrade")
                self.assertEqual(len(self.dispatched), 1)

            # 6. 90 seconds later: DIFFERENT alert #3 on same topic as diff_1 (ntfy-trades-test) + alert #4 on ntfy-error-test
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 90):
                diff_3_trade = {
                    "type": "bot_event",
                    "name": "🎉 Order Executed: SELL BTCUSDC @ $68000.00",
                    "body": "Filled 0.01 BTCUSDC",
                    "source": "monitortrades",
                    "symbol": "BTCUSDC",
                }
                self.server.dispatch_alerts([diff_3_trade], bot_name="monitortrades")
                self.server.process_line(
                    "2026-10-19 12:05:00 [ERROR] Kraken websocket connection reset by peer\n",
                    "kraken_bot",
                )
                self.assertEqual(len(self.dispatched), 1)

            # 7. Before 5 minutes (at t0 + 200s), flush_pending_bulk does nothing because gate is still closed
            self.assertFalse(self.server.flush_pending_bulk(now=t0 + 200))
            self.assertEqual(len(self.dispatched), 1)

            # 8. After central gate opens (at t0 + 305s), flush_pending_bulk synthesizes ONLY the
            # same-topic alerts for ntfy-trades-test (diff_1 + diff_3_trade) into a thematic bulk message,
            # while keeping diff_2 (ntfy-macro-test) and kraken_bot (ntfy-error-test) queued for subsequent gate openings!
            self.assertTrue(self.server.flush_pending_bulk(now=t0 + 305))
            self.assertEqual(len(self.dispatched), 2)
            bulk_msg = self.dispatched[1]
            self.assertTrue(bulk_msg["title"].startswith("📦 Bulk (2):"), bulk_msg["title"])
            self.assertIn("1. [Binance] ⏰ Order Expired & Canceled: BUY TAOUSDC @ 284.50 [rtrade] (×2)", bulk_msg["message"])
            self.assertIn("2. [Binance] 🎉 Order Executed: SELL BTCUSDC @ $68000.00", bulk_msg["message"])
            self.assertNotIn("Would Block BUY ETHUSDC", bulk_msg["message"])
            self.assertNotIn("Kraken websocket connection reset", bulk_msg["message"])
            self.assertEqual(bulk_msg["priority"], "urgent")
            self.assertEqual(bulk_msg["topic"], "ntfy-trades-test")

            # 9. Before the next central gate window (t0 + 450s), remaining topics stay deferred
            self.assertFalse(self.server.flush_pending_bulk(now=t0 + 450))
            self.assertEqual(len(self.dispatched), 2)

            # 10. Next central gate opening (t0 + 610s) dispatches ntfy-macro-test
            self.assertTrue(self.server.flush_pending_bulk(now=t0 + 610))
            self.assertEqual(len(self.dispatched), 3)
            macro_msg = self.dispatched[2]
            self.assertEqual(macro_msg["title"], "[rtrade] 🛡 [VETO] Would Block BUY ETHUSDC")
            self.assertEqual(macro_msg["topic"], "ntfy-macro-test")

            # 11. Next central gate opening (t0 + 915s) dispatches ntfy-error-test
            self.assertTrue(self.server.flush_pending_bulk(now=t0 + 915))
            self.assertEqual(len(self.dispatched), 4)
            err_msg = self.dispatched[3]
            self.assertEqual(err_msg["title"], "[kraken_bot] Critical Errors")
            self.assertIn("Kraken websocket connection reset by peer", err_msg["message"])
            self.assertEqual(err_msg["topic"], "ntfy-error-test")

    def test_assetguardian_and_trailing_stop_owner_tags_preserve_distinct_trades_in_bulk(self):
        from notify_engine.alertnotifiers import _is_guard_alert
        from notify_engine.server import _category_for_title_and_source

        self.assertEqual(_category_for_title_and_source("🎉 SELL ARBUSDC @ 0.169 [assetguardian]", "monitortrades"), "TRADES")
        self.assertEqual(_category_for_title_and_source("📝 SELL ARBUSDC @ 0.169 [assetguardian]", "monitororder"), "TRADES")
        self.assertFalse(_is_guard_alert([{
            "type": "bot_event",
            "name": "🎉 SELL ARBUSDC @ 0.169 [assetguardian]",
            "body": "Order 563411304 (assetguardian) executed! Filled: 152.9000 ARBUSDC (~25.84 USDC)",
            "source": "monitortrades",
            "symbol": "ARBUSDC",
        }]))

        t0 = 1_700_100_000.0
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_DEDUP_SECONDS": "900",
            "NOTIFICATION_GATE_SECONDS": "180",
            "NOTIFICATION_TOPIC_BULK_SECONDS": "300",
        }):
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "📝 SELL ARBUSDC @ 0.169 [assetguardian]",
                    "body": "Order 563411304 (assetguardian) placed on venue. Qty: 152.9000 ARBUSDC (~25.84 USDC)",
                    "source": "monitororder",
                    "symbol": "ARBUSDC",
                }], bot_name="monitororder")
                self.assertEqual(len(self.dispatched), 1)

            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 60):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🎉 SELL ARBUSDC @ 0.169 [assetguardian]",
                    "body": "Order 563411304 (assetguardian) executed! Filled: 152.9000 ARBUSDC (~25.84 USDC)",
                    "source": "monitortrades",
                    "symbol": "ARBUSDC",
                }], bot_name="monitortrades")

            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 120):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🎉 SELL ARBUSDC @ 0.1686 [assetguardian]",
                    "body": "Order 563412614 (assetguardian) executed! Filled: 122.3000 ARBUSDC (~20.62 USDC)",
                    "source": "monitortrades",
                    "symbol": "ARBUSDC",
                }], bot_name="monitortrades")

            self.assertTrue(self.server.flush_pending_bulk(now=t0 + 305))
            self.assertEqual(len(self.dispatched), 2)
            bulk_msg = self.dispatched[1]
            self.assertTrue(bulk_msg["title"].startswith("📦 Bulk (2):"), bulk_msg["title"])
            self.assertIn("Order 563411304 (assetguardian) executed! Filled: 152.9000 ARBUSDC", bulk_msg["message"])
            self.assertIn("Order 563412614 (assetguardian) executed! Filled: 122.3000 ARBUSDC", bulk_msg["message"])
            self.assertNotIn("(×2)", bulk_msg["title"])

    def test_queue_retry_count_persistence_and_edge_cases(self):
        import asyncio
        from notify_engine.alertnotifiers import AlertNotifier
        from orchestratorOS.lib import watchdog_common
        from intelligence.sentiment.gemini_client import GeminiClient

        # 1. format_new_coin_message rejects NaN/non-positive prices
        msg_nan = AlertNotifier.format_new_coin_message({"symbol": "ABC", "price": float("nan")})
        self.assertIn("Price: N/A", msg_nan)
        msg_neg = AlertNotifier.format_new_coin_message({"symbol": "ABC", "price": -5.0})
        self.assertIn("Price: N/A", msg_neg)

        # 2. dispatch_alerts handles non-list alerts and None fields without raising
        self.assertFalse(self.server.dispatch_alerts(123))  # type: ignore[arg-type]
        self.server.dispatch_alerts([{"type": "bot_event", "name": None, "body": None, "source": None}], bot_name=True)  # type: ignore[arg-type]

        # 3. flush_queue_loop persists incremented retry_count when single queued item fails
        with open(self.server.queue_file, "w", encoding="utf-8") as f:
            f.write('["not_a_dict"]\n')
            f.write(json.dumps({"__orchestrator_intent__": "ntfy_webhook", "title": "FailAlert", "message": "m", "topic": "t"}) + "\n")

        async def _run_one_flush():
            with mock.patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]), \
                 mock.patch.object(self.server, "flush_pending_bulk", return_value=False), \
                 mock.patch.object(self.server, "_send_ntfy", return_value=False):
                try:
                    await self.server.flush_queue_loop()
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run_one_flush())
        with open(self.server.queue_file, "r", encoding="utf-8") as f:
            queued_lines = [ln.strip() for ln in f if ln.strip()]
        self.assertEqual(len(queued_lines), 1)
        saved_item = json.loads(queued_lines[0])
        self.assertEqual(saved_item["retry_count"], 1)

        # 4. watchdog_common.load_state recovers from .previous
        prev_file = os.path.join(self._temp_dir.name, "wd_state.json")
        with open(prev_file + ".previous", "w", encoding="utf-8") as f:
            json.dump({"ok": 1}, f)
        self.assertEqual(watchdog_common.load_state(prev_file), {"ok": 1})

        # 5. GeminiClient custom_runner does not cache None/empty and query_json accepts dict
        calls = []
        cli = GeminiClient(custom_runner=lambda p, m, t: calls.append(1) or (None if len(calls) == 1 else '{"a": 1}'))
        self.assertIsNone(cli.query_json("p1", cache_ttl_sec=60.0))
        self.assertEqual(cli.query_json("p1", cache_ttl_sec=60.0), {"a": 1})

    def test_orchestrator_reconciler_gateway_and_monitor_safety(self):
        from orchestratorOS.supervisor.autonomous_ai_reconciler import (
            AutonomousAIReconciler,
            recent_log_lines,
        )
        from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
        from orchestratorOS.livecheck.fleet_health_monitor import FleetHealthMonitor

        # 1. recent_log_lines: handles NaN file_mtime, None lines, and non-str items in lines
        self.assertEqual(recent_log_lines(None, 1000.0, 500.0), [])  # type: ignore[arg-type]
        res_lines = recent_log_lines([None, "2026-10-27 00:00:00 [ERROR] test error", 123], float("nan"), 0.0)  # type: ignore[list-item]
        self.assertEqual(res_lines, ["2026-10-27 00:00:00 [ERROR] test error"])

        # 2. AutonomousAIReconciler & AutonomousIntentGateway: notify_enabled='false', mode=True rejection, non-dict submit/process_intent
        rec = AutonomousAIReconciler(workspace_dir=self._temp_dir.name, mode="shadow", notify_enabled="false")  # type: ignore[arg-type]
        self.assertFalse(rec.notify_enabled)
        with self.assertRaises(ValueError):
            AutonomousIntentGateway(workspace_dir=self._temp_dir.name, mode=True)  # type: ignore[arg-type]
        gw = AutonomousIntentGateway(workspace_dir=self._temp_dir.name, mode="shadow", notify_enabled="false")  # type: ignore[arg-type]
        self.assertFalse(gw.notify_enabled)
        self.assertEqual(gw.process_intent(None), {})  # type: ignore[arg-type]
        self.assertIsNone(gw.submit(None, source="test", max_mode="shadow"))  # type: ignore[arg-type]

        # 3. FleetHealthMonitor.gather_sensor_status: recovers from .previous when primary is missing or corrupt
        cachedb = os.path.join(self._temp_dir.name, "cachedb")
        os.makedirs(cachedb, exist_ok=True)
        with open(os.path.join(cachedb, "geopolitical_eval.json"), "w", encoding="utf-8") as f:
            f.write("{corrupt")
        with open(os.path.join(cachedb, "geopolitical_eval.json.previous"), "w", encoding="utf-8") as f:
            json.dump({"threat_level": "ELEVATED", "risk_score": 0.65, "summary": "recovered"}, f)
        mon = FleetHealthMonitor(workspace_dir=self._temp_dir.name)
        sensors = mon.gather_sensor_status()
        self.assertEqual(sensors["geopolitical_threat"]["threat_level"], "ELEVATED")
        self.assertEqual(sensors["geopolitical_threat"]["risk_score"], 0.65)

    def test_bulk_synthesis_strips_redundant_headers_and_preserves_full_summary(self):
        from notify_engine.alertnotifiers import synthesize_bulk_notifications
        from orchestratorOS.lib.technical_repair_ssot import format_supervisor_ntfy_notification
        from orchestratorOS.supervisor.autonomous_ai_reconciler import SupervisorAuditReport

        report = SupervisorAuditReport(
            ts=1793669701.0,
            mode="enforce",
            summary=(
                "All 19/19 fleet processes and heartbeats are fresh, Linux OS resources (CPU load 0.45, "
                "RAM 50.0%, Disk 68.1%) and wgpia0 VPN/DNS connectivity are healthy, all configuration "
                "files are valid, and zero persistent errors remain unresolved at log tails."
            ),
            diagnosed_issues=[
                {
                    "severity": "LOW",
                    "category": "NETWORK_ERROR",
                    "description": "Transient Binance CloudFront HTML error response at 02:00:10 (already self-recovered).",
                },
                {
                    "severity": "INFO",
                    "category": "GUARD_VETO",
                    "description": "17 guard enforcement and shadow test logs recorded.",
                },
                {
                    "severity": "INFO",
                    "category": "HEALTHY",
                    "description": "19/19 fleet processes active with fresh heartbeats.",
                },
            ],
            venue_actions=[{"action": "NONE", "venue": "system", "symbol": "SYSTEM"}],
            code_remediation={"has_fix": False},
            actions_executed=False,
        )
        rep_title, rep_body, rep_prio, rep_topic, _ = format_supervisor_ntfy_notification(
            report,
            "auto_technical_repair",
            "ntfy-error-1978",
            "ntfy-server-1978",
            "ntfy-server-1978",
        )
        self.assertEqual(rep_title, "🛠 [auto_technical_repair] Issues: 1 | Actions: 0")
        self.assertNotIn("Simulated/Observed", rep_body)
        self.assertIn("Primary Issue: [LOW] Transient Binance CloudFront", rep_body)

        bulk_title, bulk_body, _, _ = synthesize_bulk_notifications([
            {
                "title": "🔴 [auto_macro_sentiment] DEFENSIVE (BEARISH)",
                "body": (
                    "Action: DEFENSIVE · Bias: BEARISH (Risk: HIGH)\n"
                    "During the Asian overnight session (03:31 EET), BTCUSDC entered an 'aggressive_shorting' "
                    "regime as 1h open interest expanded +0.82% to $7.66B alongside heavy taker selling "
                    "($82.11M sell vs. $48.70M buy, ratio 0.593) and deepening BEARISH_BREADTH."
                ),
                "priority": "high",
                "topic": "ntfy-macro-1978",
                "count": 1,
            },
            {
                "title": rep_title,
                "body": rep_body,
                "priority": rep_prio,
                "topic": rep_topic,
                "count": 1,
            },
        ])
        self.assertTrue(bulk_title.startswith("📦 Bulk (2):"))
        self.assertNotIn("Action: DEFENSIVE · Bias: BEARISH", bulk_body)
        self.assertNotIn("Simulated/Observed", bulk_body)
        self.assertIn("[Risk: HIGH] During the Asian overnight session", bulk_body)
        self.assertIn("deepening BEARISH_BREADTH.", bulk_body)
        self.assertIn("all configuration files are valid", bulk_body)
        self.assertIn("Primary Issue: [LOW] Transient Binance CloudFront", bulk_body)
        self.assertIn("\n\n2. 🛠 [auto_technical_repair]", bulk_body)

        # Verify GUARD_VETO / HEALTHY tagged with MEDIUM severity do not trigger false-alarm SYSTEM error
        report_guard_med = SupervisorAuditReport(
            ts=1793669702.0,
            mode="enforce",
            summary="All fleet processes healthy; routine guard vetoes observed.",
            diagnosed_issues=[
                {"severity": "MEDIUM", "category": "GUARD_VETO", "description": "Routine guard vetoes"},
                {"severity": "MEDIUM", "category": "HEALTHY", "description": "System healthy"},
            ],
            venue_actions=[{"action": "NONE", "venue": "system", "symbol": "SYSTEM"}],
            code_remediation={"has_fix": False},
            actions_executed=False,
        )
        g_title, _, g_prio, _, g_any_err = format_supervisor_ntfy_notification(
            report_guard_med,
            "auto_technical_repair",
            "ntfy-error-1978",
            "ntfy-server-1978",
            "ntfy-server-1978",
        )
        self.assertFalse(g_any_err)
        self.assertEqual(g_prio, "default")
        self.assertEqual(g_title, "🛠 [auto_technical_repair] Issues: 0 | Actions: 0")

        # Verify bulk topic ranking prefers error/deadman over server regardless of item order
        _, _, b_prio, b_topic = synthesize_bulk_notifications([
            {"title": "Server status", "body": "ok", "priority": "default", "topic": "ntfy-server-1978", "count": 1},
            {"title": "Critical error", "body": "boom", "priority": "urgent", "topic": "ntfy-error-1978", "count": 1},
        ])
        self.assertEqual(b_prio, "urgent")
        self.assertEqual(b_topic, "ntfy-error-1978")

        # Verify 5-min per-topic batching + 10-min central single-gate dispatcher isolation
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_TOPIC_BULK_SECONDS": "300",
            "NOTIFICATION_GATE_SECONDS": "600",
        }):
            self.dispatched.clear()
            t0 = 1_750_000_000.0
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🔴 [auto_macro_sentiment] DEFENSIVE (BEARISH)",
                    "body": "Risk: HIGH\nMacro sentiment bearish.",
                    "source": "macro_shadow",
                    "symbol": "GLOBAL",
                }])
                self.assertEqual(len(self.dispatched), 1)
                self.assertEqual(self.dispatched[0]["topic"], "ntfy-macro-test")

            # At t0 + 120s (within 5m topic window AND within 10m central gate): 2nd macro alert arrives -> batched on macro topic
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 120):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🌍 [auto_macro_geopolitical] DOWNSCALE_50 (ELEVATED)",
                    "body": "Risk: 0.55 · Headlines: 12\nGeopolitical risk elevated.",
                    "source": "macro_shadow",
                    "symbol": "GLOBAL",
                }])
                self.assertEqual(len(self.dispatched), 1)

            # At t0 + 180s: server alert arrives -> queued separately under ntfy-server-test
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 180):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🛠 [auto_technical_repair] Issues: 1 | Actions: 0",
                    "body": "All 19/19 fleet processes healthy.",
                    "source": "ai_supervisor",
                    "symbol": "SYSTEM",
                }])
                self.assertEqual(len(self.dispatched), 1)

            # At t0 + 400s (after 5m topic window, but before 10m central gate): deferred by central gate
            self.assertFalse(self.server.flush_pending_bulk(now=t0 + 400))
            self.assertEqual(len(self.dispatched), 1)

            # At t0 + 605s (10m central gate opens): only ntfy-macro-test exits first; ntfy-server-test waits
            self.assertTrue(self.server.flush_pending_bulk(now=t0 + 605))
            self.assertEqual(len(self.dispatched), 2)
            self.assertEqual(self.dispatched[1]["topic"], "ntfy-macro-test")
            self.assertIn("auto_macro_geopolitical", self.dispatched[1]["title"])
            self.assertNotIn("auto_technical_repair", self.dispatched[1]["message"])

            # At t0 + 1210s (next 10m central gate opens): ntfy-server-test exits on its own topic
            self.assertTrue(self.server.flush_pending_bulk(now=t0 + 1210))
            self.assertEqual(len(self.dispatched), 3)
            self.assertEqual(self.dispatched[2]["topic"], "ntfy-server-test")
            self.assertIn("auto_technical_repair", self.dispatched[2]["title"])

        # Verify that when central gate is open (NOTIFICATION_GATE_SECONDS=0) and Topic A is still inside its
        # 5-min per-topic window, an alert on Topic B (whose topic window IS open) dispatches immediately
        # without being blocked by Topic A's non-due pending_bulk entry.
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_TOPIC_BULK_SECONDS": "300",
            "NOTIFICATION_GATE_SECONDS": "0",
        }):
            self.dispatched.clear()
            t1 = 1_760_000_000.0
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t1):
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🔴 [auto_macro_sentiment] DEFENSIVE (BEARISH)",
                    "body": "Macro 1",
                    "source": "macro_shadow",
                    "symbol": "GLOBAL",
                }])
                self.assertEqual(len(self.dispatched), 1)
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t1 + 60):
                # Buffered in pending_bulk because ntfy-macro-test topic window (300s) is still closed
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🌍 [auto_macro_geopolitical] DOWNSCALE_50 (ELEVATED)",
                    "body": "Macro 2",
                    "source": "macro_shadow",
                    "symbol": "GLOBAL",
                }])
                self.assertEqual(len(self.dispatched), 1)
                # Topic B (ntfy-server-test) has open topic window and central gate is open -> dispatches now!
                self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🛠 [auto_technical_repair] Issues: 2 | Actions: 1",
                    "body": "Server repair executed.",
                    "source": "ai_supervisor",
                    "symbol": "SYSTEM",
                }])
                self.assertEqual(len(self.dispatched), 2)
                self.assertEqual(self.dispatched[1]["topic"], "ntfy-server-test")

        # Verify bounded Producer-Consumer queue caps per-topic and total items (dropping overflow / evicting lower priority)
        from notify_engine.alertnotifiers import pass_common_gate, pop_due_bulk_batch
        with mock.patch.dict(os.environ, {
            "NOTIFICATION_TOPIC_BULK_SECONDS": "300",
            "NOTIFICATION_GATE_SECONDS": "600",
            "NOTIFICATION_TOPIC_BULK_MAX_ITEMS": "2",
            "NOTIFICATION_QUEUE_MAX_ITEMS": "3",
        }):
            t2 = 1_770_000_000.0
            # Open and close central gate with first message on macro topic
            ok, reason, _, _ = pass_common_gate("ntfy", "Init Macro", "m0", "default", "ntfy-macro-test", now=t2)
            self.assertTrue(ok)
            self.assertEqual(reason, "reserved")

            # Buffer 2 default-priority items on ntfy-macro-test (reaches topic_max=2)
            ok, r1, _, _ = pass_common_gate("ntfy", "Macro A", "body A", "default", "ntfy-macro-test", now=t2 + 10)
            self.assertFalse(ok)
            self.assertEqual(r1, "buffered_for_bulk")
            ok, r2, _, _ = pass_common_gate("ntfy", "Macro B", "body B", "default", "ntfy-macro-test", now=t2 + 20)
            self.assertFalse(ok)
            self.assertEqual(r2, "buffered_for_bulk")

            # 3rd default-priority item on ntfy-macro-test is dropped because topic_max=2 is full
            ok, r3, _, _ = pass_common_gate("ntfy", "Macro C Overflow", "body C", "default", "ntfy-macro-test", now=t2 + 30)
            self.assertFalse(ok)
            self.assertEqual(r3, "topic_queue_full_dropped")

            # Urgent item on ntfy-macro-test evicts the oldest default-priority item ("Macro A")
            ok, r4, _, _ = pass_common_gate("ntfy", "Macro D Urgent", "body D", "urgent", "ntfy-macro-test", urgent=True, now=t2 + 40)
            self.assertFalse(ok)
            self.assertEqual(r4, "buffered_for_bulk")

            # Fill global queue (queue_max=3) with 1 item on ntfy-server-test, then verify 4th item on ntfy-price-test is dropped
            ok, r5, _, _ = pass_common_gate("ntfy", "Server A", "srv A", "default", "ntfy-server-test", now=t2 + 50)
            self.assertEqual(r5, "buffered_for_bulk")
            ok, r6, _, _ = pass_common_gate("ntfy", "Price Overflow", "px", "default", "ntfy-price-test", now=t2 + 60)
            self.assertFalse(ok)
            self.assertEqual(r6, "queue_full_dropped")

            # Pop after 10m gate: macro batch contains "Macro B" and "Macro D Urgent" (not "Macro A" or "Macro C Overflow")
            batch = pop_due_bulk_batch("ntfy", now=t2 + 605)
            self.assertIsNotNone(batch)
            b_title, b_body, b_prio, b_topic = batch
            self.assertEqual(b_topic, "ntfy-macro-test")
            self.assertEqual(b_prio, "urgent")
            self.assertIn("Macro B", b_body)
            self.assertIn("Macro D Urgent", b_body)
            self.assertNotIn("Macro A", b_body)
            self.assertNotIn("Macro C Overflow", b_body)

            # Evicted ("Macro A") and dropped ("Macro C Overflow") items were never sent and must not be
            # suppressed as "duplicate" once queue space is available again
            ok_re_a, r_re_a, _, _ = pass_common_gate("ntfy", "Macro A", "body A", "default", "ntfy-macro-test", now=t2 + 610)
            self.assertEqual(r_re_a, "buffered_for_bulk")
            ok_re_c, r_re_c, _, _ = pass_common_gate("ntfy", "Macro C Overflow", "body C", "default", "ntfy-macro-test", now=t2 + 611)
            self.assertEqual(r_re_c, "buffered_for_bulk")

            # Verify mailer.send_email treats bounded-queue overflow drops as policy-handled (True)
            # so NotificationServer._send_email does not re-enqueue dropped overflow emails into notify_queue.json
            from notify_engine import mailer
            with mock.patch.object(mailer, "_config", return_value={"server": "smtp.test", "port": 587, "username": "u", "password": "p", "to": "t@test"}), \
                 mock.patch.object(mailer, "pass_common_gate", return_value=(False, "queue_full_dropped", False, None)):
                if os.path.exists(self.server.queue_file):
                    os.remove(self.server.queue_file)
                self.assertTrue(self.server._send_email("Overflow Email", "body"))
                self.assertFalse(os.path.exists(self.server.queue_file))

            # Also verify NotificationServer._enqueue_payload caps queue.jsonl to NOTIFICATION_QUEUE_MAX_ITEMS=3
            self.server._enqueue_payload("ntfy_webhook", {"title": "Q1"})
            self.server._enqueue_payload("ntfy_webhook", {"title": "Q2"})
            self.server._enqueue_payload("ntfy_webhook", {"title": "Q3"})
            self.server._enqueue_payload("ntfy_webhook", {"title": "Q4"})
            with open(self.server.queue_file, "r", encoding="utf-8") as f:
                q_titles = [json.loads(ln)["title"] for ln in f if ln.strip()]
            self.assertEqual(q_titles, ["Q2", "Q3", "Q4"])

    def test_gated_ntfy_accepted_status_email_mirror_and_blocked_pop(self):
        from notify_engine.alertnotifiers import _mark_provider_daily_limit, pass_common_gate, pop_due_bulk_batch

        with mock.patch.dict(os.environ, {
            "NOTIFICATION_TOPIC_BULK_SECONDS": "300",
            "NOTIFICATION_GATE_SECONDS": "600",
            "NOTIFICATION_ERROR_DEDUP_SECONDS": "1800",
        }):
            self.dispatched.clear()
            mirrored_emails = []
            self.server._send_email = lambda subj, body, is_retry=False, dedup=True: mirrored_emails.append((subj, body, is_retry, dedup)) or True
            t0 = 1_780_000_000.0

            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0):
                # 1. First trade alert opens and closes the 10-minute central gate
                ok1 = self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "🎉 Order Executed: BUY BTCUSDC @ $80000.00",
                    "body": "Filled 0.01 BTCUSDC",
                    "source": "monitortrades",
                    "symbol": "BTCUSDC",
                }], bot_name="monitortrades")
                self.assertTrue(ok1)
                self.assertEqual(len(self.dispatched), 1)
                self.assertEqual(mirrored_emails, [])

            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 30):
                # 2. ERROR alert arrives while ntfy central gate is closed -> buffered for bulk on ntfy,
                #    returns True (accepted by gate so callers/flush_queue_loop do not treat as transport failure),
                #    and immediately mirrors to email!
                ok_err = self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "ORDER FAILED: BUY ETHUSDC",
                    "body": "Insufficient balance",
                    "source": "rtrade",
                    "symbol": "ETHUSDC",
                }], bot_name="rtrade")
                self.assertTrue(ok_err)
                self.assertEqual(len(self.dispatched), 1)
                self.assertEqual(len(mirrored_emails), 1)
                self.assertIn("ORDER FAILED: BUY ETHUSDC", mirrored_emails[0][0])

                # 3. Duplicate of the ERROR alert within dedup window -> returns True without re-emailing
                ok_dup = self.server.dispatch_alerts([{
                    "type": "bot_event",
                    "name": "ORDER FAILED: BUY ETHUSDC",
                    "body": "Insufficient balance",
                    "source": "rtrade",
                    "symbol": "ETHUSDC",
                }], bot_name="rtrade")
                self.assertTrue(ok_dup)
                self.assertEqual(len(mirrored_emails), 1)

            # 4. If ntfy hits provider_daily_limit while pending_bulk has items, pop_due_bulk_batch must not pop and lose the batch
            with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t0 + 605):
                _mark_provider_daily_limit("ntfy", cooldown_seconds=300.0)
                self.assertIsNone(pop_due_bulk_batch("ntfy", now=t0 + 605))

            # 5. Once the provider daily limit cooldown expires (t0 + 910s), flush_pending_bulk dispatches the buffered ERROR
            #    to ntfy WITHOUT double-mirroring it to email
            real_send_ntfy = NotificationServer._send_ntfy.__get__(self.server, NotificationServer)
            self.server._send_ntfy = real_send_ntfy
            with mock.patch("requests.post") as mock_post:
                mock_resp = mock.MagicMock(status_code=200, text="", headers={})
                mock_post.return_value = mock_resp
                self.assertTrue(self.server.flush_pending_bulk(now=t0 + 910))
                self.assertEqual(mock_post.call_count, 1)
                self.assertEqual(len(mirrored_emails), 1)

                # 6. Midnight UTC rollover preserves pending_bulk while resetting daily sent counter
                midnight_before = 1_780_012_740.0  # 23:59:00 UTC
                midnight_after = 1_780_013_160.0   # 00:06:00 UTC (next day)
                ok_m1, _, _, _ = pass_common_gate("ntfy", "Pre-midnight Alpha", "body alpha", "default", "ntfy-macro-test", now=midnight_before)
                self.assertTrue(ok_m1)
                ok_m2, r_m2, _, _ = pass_common_gate("ntfy", "Pre-midnight Beta", "body beta", "default", "ntfy-macro-test", now=midnight_before + 30)
                self.assertFalse(ok_m2)
                self.assertEqual(r_m2, "buffered_for_bulk")
                # After midnight + 6m (when gate opens), buffered item from 23:59:30 UTC is still preserved and popped!
                popped_after_midnight = pop_due_bulk_batch("ntfy", now=midnight_after + 300)
                self.assertIsNotNone(popped_after_midnight)
                self.assertIn("Pre-midnight Beta", popped_after_midnight[0])

                # 7. When a second error arrives after the gate opens and synthesizes a bulk with a buffered error,
                #    _send_email receives only the single new error (not the synthesized '📦 Bulk')
                mirrored_emails.clear()
                t_err = midnight_after + 1000.0
                with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t_err):
                    self.server._gate_and_send_ntfy("Error One", "detail alpha", "urgent", "ntfy-error-test", urgent=True)
                with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t_err + 30):
                    self.server._gate_and_send_ntfy("Error Two", "detail beta", "urgent", "ntfy-error-test", urgent=True)
                with mock.patch("notify_engine.alertnotifiers.time.time", return_value=t_err + 610):
                    self.server._gate_and_send_ntfy("Error Three", "detail gamma", "urgent", "ntfy-error-test", urgent=True)
                self.assertEqual([e[0] for e in mirrored_emails], ["Error One", "Error Two", "Error Three"])


from unittest.mock import patch as _wd_patch
from notify_engine import mailer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in (("orchestratorOS", "lib"), ("orchestratorOS", "livecheck")):
    _p = os.path.join(ROOT_DIR, *_sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os_resources_watchdog as wd  # noqa: E402
import watchdog_common as wc  # noqa: E402


class _WatchdogResp:
    def __init__(self, status=200, headers=None, text=""):
        self.status_code = status
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        pass


class SendNtfyRetryTest(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {
            "NTFY_TOPIC_ERROR": "testtopic",
            "NTFY_DAILY_BUDGET": "100",
            "NTFY_URGENT_RESERVE": "20",
            "DISABLE_EXTERNAL_NOTIFICATIONS": "0",
        }, clear=False)
        self._env.start()
        self._temporary = tempfile.TemporaryDirectory()
        os.environ["NOTIFICATION_STATE_FILE"] = os.path.join(
            self._temporary.name, "notifications.json",
        )

    def tearDown(self):
        self._env.stop()
        self._temporary.cleanup()

    def test_retry_on_429_then_success(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep") as msleep:
            mpost.side_effect = [_WatchdogResp(429, {"Retry-After": "1"}), _WatchdogResp(200)]
            ok = wc.send_ntfy("title", "message")
        self.assertTrue(ok)
        self.assertEqual(mpost.call_count, 2)
        msleep.assert_called_once()

    def test_daily_429_does_not_retry_and_blocks_following_attempts(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep"):
            mpost.return_value = _WatchdogResp(429, text='{"code":42908,"error":"daily limit reached"}')
            ok = wc.send_ntfy("title", "message")
            blocked = wc.send_ntfy("second title", "second message")
        self.assertFalse(ok)
        self.assertFalse(blocked)
        self.assertEqual(mpost.call_count, 1)

    def test_success_first_try_no_sleep(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep") as msleep:
            mpost.side_effect = [_WatchdogResp(200)]
            ok = wc.send_ntfy("title", "message")
        self.assertTrue(ok)
        self.assertEqual(mpost.call_count, 1)
        msleep.assert_not_called()


def _wd_proc(pid=10, ticks=100, rss=100, command="worker.py"):
    return {f"{pid}:1": {"pid": pid, "ticks": ticks, "rss_mb": rss, "command": command}}


def test_alerts_only_after_sustained_cpu_use():
    with tempfile.TemporaryDirectory() as tmp, _wd_patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
            _wd_patch.object(wd, "CONSECUTIVE", 3), _wd_patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_wd_proc(ticks=100), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_wd_proc(ticks=160), cpu_count=1)
        wd.check_once(now=240, total_ticks=1200, processes=_wd_proc(ticks=220), cpu_count=1)
        alert.assert_not_called()
        wd.check_once(now=360, total_ticks=1300, processes=_wd_proc(ticks=280), cpu_count=1)
        alert.assert_called_once()


def test_short_memory_spike_does_not_alert():
    with tempfile.TemporaryDirectory() as tmp, _wd_patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
            _wd_patch.object(wd, "MEMORY_THRESHOLD_MB", 500), \
            _wd_patch.object(wd, "CONSECUTIVE", 3), _wd_patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_wd_proc(rss=800), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_wd_proc(ticks=101, rss=100), cpu_count=1)
        alert.assert_not_called()


def test_sustained_memory_use_alerts_and_then_recovers():
    with tempfile.TemporaryDirectory() as tmp, _wd_patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
            _wd_patch.object(wd, "MEMORY_THRESHOLD_MB", 500), \
            _wd_patch.object(wd, "CONSECUTIVE", 2), _wd_patch.object(wd, "RECOVERY_CHECKS", 2), \
            _wd_patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_wd_proc(rss=800), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_wd_proc(ticks=101, rss=800), cpu_count=1)
        wd.check_once(now=240, total_ticks=1200, processes=_wd_proc(ticks=102, rss=100), cpu_count=1)
        wd.check_once(now=360, total_ticks=1300, processes=_wd_proc(ticks=103, rss=100), cpu_count=1)
        assert alert.call_count == 2


def test_managed_process_is_restarted_after_sustained_breach():
    proc = next(iter(_wd_proc(rss=800, command="python worker.py").values()))
    state = {}
    with _wd_patch.object(wd, "AUTO_RESTART", True), \
            _wd_patch.object(wd, "_managed_script", return_value="worker.py"), \
            _wd_patch.object(wd, "_restart_allowed", return_value=True), \
            _wd_patch.object(wd.os, "kill") as kill:
        assert wd._restart_managed(proc, state, 1000) is True
        kill.assert_called_once_with(10, wd.signal.SIGTERM)
        assert state["restart_history"]["worker.py"] == [1000]


def test_unmanaged_process_is_never_restarted():
    proc = next(iter(_wd_proc(rss=800, command="foreign").values()))
    with _wd_patch.object(wd, "_managed_script", return_value=None), \
            _wd_patch.object(wd.os, "kill") as kill:
        assert wd._restart_managed(proc, {}, 1000) is None
        kill.assert_not_called()


def test_parse_proc_stat_handles_spaces_and_parens_in_comm():
    after_fields = [
        "S", "1", "100", "100", "0", "-1", "4194304",
        "10", "0", "0", "0", "140", "60",
        "0", "0", "20", "0", "1", "0",
        "987654", "2147483648", "4096", "18446744073709551615",
    ]
    stat_line = f"12345 (Web Content (2)) {' '.join(after_fields)}\n"
    comm, start_time, ticks, rss_pages = wd._parse_proc_stat(stat_line)
    assert comm == "Web Content (2)"
    assert start_time == 987654
    assert ticks == 200
    assert rss_pages == 4096


class EmailMirrorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {
            "DISABLE_EXTERNAL_NOTIFICATIONS": "0",
            "MPTRADE_ORCHESTRATED": "",
            "NTFY_TOPIC_ERROR": "t-error",
            "NTFY_TOPIC_DEADMAN": "t-deadman",
            "NTFY_TOPIC_TRADES": "t-trades",
            "SMTP_USERNAME": "bot@example.com",
            "SMTP_PASSWORD": "abcd efgh ijkl mnop",
            "ALERT_TO_EMAIL": "owner@example.com",
            "SMTP_SERVER": "smtp.gmail.com",
            "NOTIFICATION_STATE_FILE": os.path.join(self._tmp.name, "state.json"),
        }, clear=False)
        self._env.start()
        self.sent = []
        self._smtp = mock.patch.object(mailer, "_smtp_send", lambda msg, cfg: self.sent.append((msg, cfg)))
        self._smtp.start()
        self.server = NotificationServer()
        self.server.queue_file = os.path.join(self._tmp.name, "queue.jsonl")

    def tearDown(self):
        self._smtp.stop()
        self._env.stop()
        self._tmp.cleanup()

    def test_error_topic_push_is_mirrored_to_email(self):
        with mock.patch("requests.post", return_value=_WatchdogResp()):
            self.assertTrue(self.server._send_ntfy("[bot] Order FAILED", "detail", "high", "t-error"))
        self.assertEqual(len(self.sent), 1)
        msg, cfg = self.sent[0]
        self.assertEqual(msg["Subject"], "[bot] Order FAILED")
        self.assertEqual(msg["To"], "owner@example.com")
        self.assertEqual(cfg["password"], "abcdefghijklmnop")

    def test_deadman_topic_is_mirrored(self):
        with mock.patch("requests.post", return_value=_WatchdogResp()):
            self.server._send_ntfy("SERVER DOWN", "x", "urgent", "t-deadman")
        self.assertEqual(len(self.sent), 1)

    def test_deadman_skip_email_suppresses_mirroring(self):
        with mock.patch("requests.post", return_value=_WatchdogResp()):
            self.server._send_ntfy("Config Reloaded", "Reloaded", "high", "t-deadman", skip_email=True)
        self.assertEqual(self.sent, [])

    def test_email_survives_ntfy_outage(self):
        with mock.patch("requests.post", side_effect=OSError("network down")):
            self.assertFalse(self.server._send_ntfy("ERROR x", "y", "high", "t-error"))
        self.assertEqual(len(self.sent), 1)

    def test_other_topics_are_push_only(self):
        with mock.patch("requests.post", return_value=_WatchdogResp()):
            self.server._send_ntfy("BUY filled", "x", "high", "t-trades")
        self.assertEqual(self.sent, [])

    def test_ntfy_retry_does_not_re_email(self):
        with mock.patch("requests.post", return_value=_WatchdogResp()):
            self.server._send_ntfy("ERROR x", "y", "high", "t-error", is_retry=True)
        self.assertEqual(self.sent, [])

    def test_duplicate_email_is_suppressed_but_retry_is_not(self):
        self.assertTrue(self.server._send_email("S", "B"))
        self.assertTrue(self.server._send_email("S", "B"))
        self.assertTrue(self.server._send_email("S", "B", is_retry=True))
        self.assertEqual(len(self.sent), 2)

    def test_failed_email_is_queued_for_retry(self):
        with mock.patch.object(mailer, "_smtp_send", side_effect=OSError("smtp down")):
            self.assertFalse(self.server._send_email("S", "B"))
        with open(self.server.queue_file, encoding="utf-8") as fh:
            payload = json.loads(fh.readline())
        self.assertEqual(payload, {"__orchestrator_intent__": "email", "subject": "S", "message": "B"})

    def test_auth_rejected_email_is_not_queued_for_retry(self):
        import smtplib
        with mock.patch.object(
            mailer, "_smtp_send", side_effect=smtplib.SMTPAuthenticationError(535, b"BadCredentials")
        ):
            self.assertFalse(self.server._send_email("S", "B"))
        self.assertTrue(mailer.last_error_is_auth_failure())
        self.assertFalse(os.path.exists(self.server.queue_file))

    def test_orchestrated_email_intent_is_delivered(self):
        line = json.dumps({"__orchestrator_intent__": "email", "subject": "Subj", "message": "Body"})
        self.server.process_line(line, "bot")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0][0]["Subject"], "Subj")

    def test_send_email_batch_outside_orchestrator_sends_directly(self):
        alert = {"type": "bot_event", "name": "CRITICAL thing", "body": "b", "source": "s", "symbol": "X"}
        self.assertTrue(AlertNotifier.send_email_batch([alert]))
        self.assertEqual(self.sent[0][0]["Subject"], "CRITICAL thing")

    def test_cli_topic_filter(self):
        self.assertEqual(mailer.main(["--topic", "t-trades", "s", "b"]), 0)
        self.assertEqual(self.sent, [])
        self.assertEqual(mailer.main(["--topic", "t-error", "s", "b"]), 0)
        self.assertEqual(len(self.sent), 1)

    def test_trades_never_mirrored_to_email_even_when_ntfy_blocked(self):
        from notify_engine.alertnotifiers import _mark_provider_daily_limit
        _mark_provider_daily_limit("ntfy", cooldown_seconds=3600.0)

        trade_alert = {
            "type": "bot_event",
            "name": "📝 BUY TAOUSDC @ 255.50 [rtrade]",
            "body": "BUY order placed",
            "source": "rtrade",
            "symbol": "TAOUSDC",
        }
        result = self.server.dispatch_alerts([trade_alert])
        self.assertFalse(result)
        self.assertEqual(self.sent, [])

    def test_errors_still_mirrored_to_email_when_ntfy_blocked(self):
        from notify_engine.alertnotifiers import _mark_provider_daily_limit
        _mark_provider_daily_limit("ntfy", cooldown_seconds=3600.0)

        error_alert = {
            "type": "bot_event",
            "name": "Exception in bot runner",
            "body": "Fatal traceback details",
            "source": "watchdog",
            "symbol": "BTCUSDC",
        }
        result = self.server.dispatch_alerts([error_alert])
        self.assertFalse(result)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("Exception in bot runner", self.sent[0][0]["Subject"])

    def test_deadman_still_mirrored_to_email_when_ntfy_blocked(self):
        from notify_engine.alertnotifiers import _mark_provider_daily_limit
        _mark_provider_daily_limit("ntfy", cooldown_seconds=3600.0)

        server_alert = {
            "type": "bot_event",
            "name": "SERVER DOWN",
            "body": "Heartbeat missing",
            "source": "deadman",
            "symbol": "SYS",
        }
        result = self.server.dispatch_alerts([server_alert])
        self.assertFalse(result)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("SERVER DOWN", self.sent[0][0]["Subject"])

    def test_owner_tag_stripping_and_refused_guard_classification(self):
        from notify_engine.alertnotifiers import _is_guard_alert, strip_owner_tags
        from notify_engine.server import _category_for_title_and_source
        import order_retry

        self.assertEqual(strip_owner_tags("🎉 BUY BTCUSDC [ai_executor]"), "🎉 BUY BTCUSDC ")
        self.assertEqual(strip_owner_tags("📝 BUY HYPEUSD [spot_dca]"), "📝 BUY HYPEUSD ")
        self.assertEqual(strip_owner_tags("🎉 SELL TAOUSDC [trailing_stop]"), "🎉 SELL TAOUSDC ")
        self.assertEqual(strip_owner_tags("✓ Order filled: NVDA_US_EQ [order_manager]"), "✓ Order filled: NVDA_US_EQ ")
        self.assertTrue(_is_guard_alert([{"name": "⚠️ [auto_trade_executor] REFUSED BUY BTCUSDC", "source": "llm_trader"}]))
        self.assertEqual(order_retry.infer_concrete_record_owner({"intent_id": "trailing:kraken:ADA:SELL:100"}), "trailing_stop")
        self.assertEqual(order_retry.infer_concrete_record_owner({"kind": "TRAIL_SELL"}), "trailing_stop")
        self.assertEqual(order_retry.infer_concrete_record_owner({"client_order_id": "OR_123_0"}), "")
        self.assertEqual(_category_for_title_and_source("HYPEUSD listed on Kraken!", "kraken"), "PRICE")
        self.assertEqual(_category_for_title_and_source("✗ Order REJECTED: NVDA", "T212 order"), "ERROR")
        self.assertEqual(_category_for_title_and_source("✗ Order REJECTED: BUY NVDA", "T212 order"), "ERROR")
        self.assertEqual(_category_for_title_and_source("⚠ Order Skipped (Below Min): BUY ARBUSDC [rtrade]", "rtrade"), "GUARD")
        self.assertEqual(_category_for_title_and_source("[order_retry] 🛑 order-retry GIVE UP SELL TAOUSDC", "order_retry"), "ERROR")
        self.assertEqual(_category_for_title_and_source("🛑 order-retry queue corruption", "order_retry"), "ERROR")
        self.assertEqual(_category_for_title_and_source("[assetguardian] 🛑 ERROR: AI LLM Paid Argon Quota Exhausted (0%)", "assetguardian"), "ERROR")
        self.assertEqual(_category_for_title_and_source("Pilot backtest: config changed", "scheduled_pilot.py"), "SERVER")
        self.assertEqual(_category_for_title_and_source("Backtest: 3 new config proposal(s)", "scheduled_pilot.py"), "SERVER")

        # Verify legacy email intent with "alerts" list (no "message") renders body cleanly
        self.sent.clear()
        legacy_email_line = json.dumps({
            "__orchestrator_intent__": "email",
            "subject": "High sustained process resource usage",
            "alerts": [{"type": "bot_event", "symbol": "SYSTEM", "name": "High sustained process resource usage", "source": "watchdog", "body": "PID 123 — CPU 99%"}],
        })
        self.server.process_line(legacy_email_line, "os_resources_watchdog")
        self.assertEqual(len(self.sent), 1)
        self.assertIn("PID 123", self.sent[0][0].get_payload(decode=True).decode("utf-8"))

    def test_distinct_trades_within_topic_bulk_window_are_batched_not_eliminated(self):
        from notify_engine.alertnotifiers import pass_common_gate, pop_due_bulk_batch

        with mock.patch.dict(os.environ, {
            "NOTIFICATION_GATE_SECONDS": "180",
            "NOTIFICATION_TOPIC_BULK_SECONDS": "300",
        }):
            t0 = 1_750_000_000.0
            t1_alert = [{"type": "bot_event", "name": "[Binance] 🎉 BUY BTCUSDC @ 75180.00 [tradeall]", "body": "Order 1001 (tradeall) executed! Filled: 0.0066 BTCUSDC (~500.00 USDC)", "source": "monitortrades", "symbol": "BTCUSDC"}]
            t2_alert = [{"type": "bot_event", "name": "[Binance] 🎉 BUY BTCUSDC @ 74900.00 [tradeall]", "body": "Order 1002 (tradeall) executed! Filled: 0.0067 BTCUSDC (~500.00 USDC)", "source": "monitortrades", "symbol": "BTCUSDC"}]

            allow1, reason1, _, _ = pass_common_gate(
                "ntfy", t1_alert[0]["name"], t1_alert[0]["body"], "urgent", "ntfy-trades-test", t1_alert, urgent=True, now=t0
            )
            self.assertTrue(allow1)
            self.assertEqual(reason1, "reserved")

            # 60s later: distinct trade on same symbol must be queued for bulk, NOT dropped as similar_eliminated
            allow2, reason2, _, _ = pass_common_gate(
                "ntfy", t2_alert[0]["name"], t2_alert[0]["body"], "urgent", "ntfy-trades-test", t2_alert, urgent=True, now=t0 + 60.0
            )
            self.assertFalse(allow2)
            self.assertEqual(reason2, "buffered_for_bulk")

            batch = pop_due_bulk_batch("ntfy", now=t0 + 305.0)
            self.assertIsNotNone(batch)
            self.assertIn("Order 1002", batch[1])

        from notify_engine.alertnotifiers import (
            _GATE_ACCEPTED_REASONS,
            _GATE_SUPPRESSED_REASONS,
            _dedup_seconds,
            format_notify_price,
            format_notify_qty,
            format_owner_tags,
            format_trade_notification,
            format_trade_venue_tag,
            strip_owner_tags,
        )
        from notify_engine.server import NotificationServer

        self.assertEqual(format_owner_tags("tradeall"), (" [tradeall]", " (tradeall)"))
        self.assertEqual(format_owner_tags("unspecified"), ("", ""))
        self.assertEqual(format_owner_tags("order_retry"), ("", ""))
        self.assertEqual(format_owner_tags("order_retry", ignore=("unspecified", "unknown")), (" [order_retry]", " (order_retry)"))
        self.assertEqual(format_owner_tags("alt:12345"), ("", ""))
        self.assertEqual(format_trade_venue_tag("binance", "assetguardian"), "[binance:assetguardian]")
        self.assertEqual(format_trade_venue_tag("binance", "unspecified"), "[binance]")
        self.assertEqual(
            format_trade_notification("FILLED", "BUY", "TAOUSDC", "212.50", 1.7379, 370.52, "USDC", "721818232", venue="binance", owner="assetguardian"),
            ("BUY TAOUSDC@212.50 [binance:assetguardian]", "FILLED ~370.52 USDC (qty: 1.7379, order: 721818232)"),
        )
        self.assertEqual(
            format_trade_notification("PLACED", "SELL", "TAOUSDC", "224.20", 7.1932, 1612.72, "USDC", "721977719", venue="binance", owner="tradeall"),
            ("SELL TAOUSDC@224.20 [binance:tradeall]", "PLACED ~1612.72 USDC (qty: 7.1932, order: 721977719)"),
        )
        self.assertEqual(strip_owner_tags("BUY TAOUSDC@212.50 [binance:assetguardian]").strip(), "BUY TAOUSDC@212.50")
        with mock.patch.object(self.server, "_send_ntfy", return_value=True) as mock_send_ntfy:
            self.server.dispatch_alerts([{
                "type": "bot_event",
                "name": "BUY TAOUSDC@212.50 [binance:assetguardian]",
                "body": "FILLED ~370.52 USDC (qty: 1.7379, order: 721818232)",
                "source": "monitortrades",
                "symbol": "TAOUSDC",
            }], bot_name="monitortrades")
            mock_send_ntfy.assert_called_once()
            self.assertEqual(mock_send_ntfy.call_args[0][0], "BUY TAOUSDC@212.50 [binance:assetguardian]")
            self.assertEqual(mock_send_ntfy.call_args[0][3], "t-trades")
        self.assertEqual(format_notify_qty(0.015), "0.015")
        self.assertEqual(format_notify_qty(0.0), "0")
        self.assertEqual(format_notify_price(75180.5), "75,180.50")
        self.assertEqual(format_notify_price(75180.5, comma=False), "75180.50")
        self.assertEqual(format_notify_price(0.0001842), "0.0001842")
        self.assertTrue(set(_GATE_SUPPRESSED_REASONS).issubset(set(_GATE_ACCEPTED_REASONS)))
        self.assertIn("buffered_for_bulk", _GATE_ACCEPTED_REASONS)
        self.assertNotIn("buffered_for_bulk", _GATE_SUPPRESSED_REASONS)
        tb_lines = [f"line_{i}" for i in range(15)]
        collapsed = NotificationServer._collapse_traceback(tb_lines)
        self.assertIn("...", collapsed)
        self.assertTrue(collapsed.startswith("line_0\n  ...\nline_5"))
        with mock.patch.dict(os.environ, {"NOTIFICATION_MACRO_DEDUP_SECONDS": "3600", "NOTIFICATION_GUARD_DEDUP_SECONDS": "1800"}):
            macro_alert = [{"name": "🧭 [auto_macro_sentiment] HOLD (CAUTION)", "source": "macro_shadow"}]
            self.assertEqual(_dedup_seconds(macro_alert, False), 3600)
            guard_alert = [{"name": "🛡 [VETO] Would Block BUY TAOUSDC", "source": "macro_shadow"}]
            self.assertEqual(_dedup_seconds(guard_alert, False), 1800)





