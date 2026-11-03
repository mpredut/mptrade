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

from notify_engine.alertnotifiers import notify
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
        self.assertEqual(_topic_for_category("LIQUIDATION IMMINENT", "hl_bot"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("🛑 order submission quarantined", "order_retry"), "ntfy-guard-test")
        self.assertEqual(_topic_for_category("Drawdown alert", "assetguardian"), "ntfy-guard-test")

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





