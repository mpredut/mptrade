"""ERROR and DEADMAN ntfy channels are mirrored to email (notify_engine/mailer.py)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from notify_engine import mailer
from notify_engine.alertnotifiers import AlertNotifier
from notify_engine.server import NotificationServer


class _Resp:
    status_code = 200
    text = ""
    headers = {}

    def raise_for_status(self):
        pass


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
        with mock.patch("requests.post", return_value=_Resp()):
            self.assertTrue(self.server._send_ntfy("[bot] Order FAILED", "detail", "high", "t-error"))
        self.assertEqual(len(self.sent), 1)
        msg, cfg = self.sent[0]
        self.assertEqual(msg["Subject"], "[bot] Order FAILED")
        self.assertEqual(msg["To"], "owner@example.com")
        self.assertEqual(cfg["password"], "abcdefghijklmnop")  # gmail app-password spaces stripped

    def test_deadman_topic_is_mirrored(self):
        with mock.patch("requests.post", return_value=_Resp()):
            self.server._send_ntfy("SERVER DOWN", "x", "urgent", "t-deadman")
        self.assertEqual(len(self.sent), 1)

    def test_deadman_skip_email_suppresses_mirroring(self):
        with mock.patch("requests.post", return_value=_Resp()):
            self.server._send_ntfy("Config Reloaded", "Reloaded", "high", "t-deadman", skip_email=True)
        self.assertEqual(self.sent, [])

    def test_email_survives_ntfy_outage(self):
        with mock.patch("requests.post", side_effect=OSError("network down")):
            self.assertFalse(self.server._send_ntfy("ERROR x", "y", "high", "t-error"))
        self.assertEqual(len(self.sent), 1)

    def test_other_topics_are_push_only(self):
        with mock.patch("requests.post", return_value=_Resp()):
            self.server._send_ntfy("BUY filled", "x", "high", "t-trades")
        self.assertEqual(self.sent, [])

    def test_ntfy_retry_does_not_re_email(self):
        with mock.patch("requests.post", return_value=_Resp()):
            self.server._send_ntfy("ERROR x", "y", "high", "t-error", is_retry=True)
        self.assertEqual(self.sent, [])

    def test_duplicate_email_is_suppressed_but_retry_is_not(self):
        self.assertTrue(self.server._send_email("S", "B"))
        self.assertTrue(self.server._send_email("S", "B"))           # deduplicated
        self.assertTrue(self.server._send_email("S", "B", is_retry=True))
        self.assertEqual(len(self.sent), 2)

    def test_failed_email_is_queued_for_retry(self):
        with mock.patch.object(mailer, "_smtp_send", side_effect=OSError("smtp down")):
            self.assertFalse(self.server._send_email("S", "B"))
        with open(self.server.queue_file, encoding="utf-8") as fh:
            payload = json.loads(fh.readline())
        self.assertEqual(payload, {"__orchestrator_intent__": "email", "subject": "S", "message": "B"})

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
        # dispatch_alerts should return False because ntfy is blocked, and NOT send email
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


if __name__ == "__main__":
    unittest.main()
