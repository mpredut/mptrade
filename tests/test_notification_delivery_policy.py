import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock
import pytest

import alertnotifiers
from alertnotifiers import AlertNotifier

# send_phone_webhook_batch now takes the target topic explicitly (no PHONE_ALERT_URL
# / NTFY_TOPIC env fallback). These delivery-policy tests just need any valid URL.
_WEBHOOK = "https://ntfy.sh/test-topic"


class _Response:
    status_code = 200
    headers = {}
    text = "ok"


class _RejectedResponse:
    status_code = 401
    headers = {}
    text = "unauthorized"


class _DailyLimitResponse:
    status_code = 429
    headers = {}
    text = '{"code":42908,"error":"daily limit reached"}'


def _event(title="FILL BUY", body="qty=1", source="kraken"):
    return {
        "type": "bot_event", "symbol": "HYPE", "name": title,
        "source": source, "body": body,
    }


@unittest.skip("Delivery policy moved to Orchestrator")
class NotificationDeliveryPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(os.environ, {
            "NOTIFICATION_STATE_FILE": os.path.join(self.temporary.name, "state.json"),
            "NTFY_TOPIC": "test-topic",
            "NTFY_DAILY_BUDGET": "2",
            "NTFY_URGENT_RESERVE": "1",
            "EMAIL_DAILY_BUDGET": "2",
            "EMAIL_URGENT_RESERVE": "1",
        }, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def exhausted_state(self, *, provider_blocked=False):
        state = {
            "schema_version": 1,
            "date_utc": datetime.now(timezone.utc).date().isoformat(),
            "channels": {
                channel: {"sent": sent, "last": {}, "blocked": provider_blocked,
                          "budget_warning_sent": True}
                for channel, sent in (("ntfy", 100), ("email", 40))
            },
        }
        with open(os.environ["NOTIFICATION_STATE_FILE"], "w", encoding="utf-8") as handle:
            json.dump(state, handle)

    @mock.patch("alertnotifiers.requests.post", return_value=_Response())
    def test_urgent_ntfy_bypasses_exhausted_legacy_budget_without_reset(self, post):
        self.exhausted_state()
        for index in range(3):
            self.assertTrue(AlertNotifier.send_phone_webhook_batch(
                [_event("STOP-LOSS", body=f"incident {index}")], webhook_url=_WEBHOOK))
        self.assertFalse(AlertNotifier.send_phone_webhook_batch(
            [_event("FILL")], webhook_url=_WEBHOOK))
        self.assertEqual(post.call_count, 3)
        self.assertEqual(post.call_args.kwargs["headers"]["Priority"], "urgent")
        with open(os.environ["NOTIFICATION_STATE_FILE"], encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["channels"]["ntfy"]["sent"], 103)

    @mock.patch("alertnotifiers.requests.post", return_value=_Response())
    def test_zero_routine_allowance_does_not_disable_urgent_alerts(self, post):
        with mock.patch.dict(os.environ, {"NTFY_DAILY_BUDGET": "0"}):
            self.assertFalse(AlertNotifier.send_phone_webhook_batch(
                [_event()], webhook_url=_WEBHOOK))
            self.assertTrue(AlertNotifier.send_phone_webhook_batch(
                [_event("ERROR: account unavailable")], webhook_url=_WEBHOOK))
        self.assertEqual(post.call_count, 1)

    @mock.patch("alertnotifiers.requests.post", return_value=_Response())
    def test_urgent_duplicates_still_deduplicate_after_budget_exhaustion(self, post):
        self.exhausted_state()
        event = _event("STOP-LOSS")
        for _ in range(2):
            self.assertTrue(AlertNotifier.send_phone_webhook_batch([event], webhook_url=_WEBHOOK))
        self.assertEqual(post.call_count, 1)

    @mock.patch("alertnotifiers.smtplib.SMTP")
    def test_email_has_no_local_volume_limit_even_with_legacy_exhausted_state(self, smtp):
        self.exhausted_state(provider_blocked=True)
        with mock.patch.dict(os.environ, {
            "SMTP_SERVER": "smtp.example.test", "SMTP_PORT": "587",
            "SMTP_USERNAME": "from@example.test", "SMTP_PASSWORD": "secret",
            "ALERT_TO_EMAIL": "to@example.test", "EMAIL_DAILY_BUDGET": "0",
        }):
            for index, title in enumerate(("FILL", "STOP-LOSS", "FILL", "ERROR")):
                self.assertTrue(AlertNotifier.send_email_batch([_event(title, body=f"incident {index}")]))
        self.assertEqual(smtp.call_count, 4)

    @mock.patch.object(AlertNotifier, "send_email_batch", return_value=True)
    @mock.patch("alertnotifiers.requests.post", return_value=_DailyLimitResponse())
    def test_provider_quota_falls_back_with_each_original_urgent_incident(self, post, email):
        first, second = _event("STOP-LOSS", body="first incident"), _event("ERROR", body="second incident")
        self.assertFalse(AlertNotifier.send_phone_webhook_batch([first], webhook_url=_WEBHOOK))
        self.assertFalse(AlertNotifier.send_phone_webhook_batch([second], webhook_url=_WEBHOOK))
        self.assertEqual(post.call_count, 1)  # Honor the actual provider quota after its rejection.
        fallback_calls = [call for call in email.call_args_list
                          if "ntfy quota exhausted -> email" in (call.kwargs.get("subject") or "")]
        self.assertEqual([call.args[0] for call in fallback_calls], [[first], [second]])

    @mock.patch.object(AlertNotifier, "send_email_batch", return_value=False)
    @mock.patch("alertnotifiers.requests.post")
    def test_failed_email_fallback_does_not_report_phone_success(self, post, email):
        self.exhausted_state(provider_blocked=True)
        self.assertFalse(AlertNotifier.send_phone_webhook_batch(
            [_event("STOP-LOSS")], webhook_url=_WEBHOOK))
        post.assert_not_called()
        email.assert_called_once()

    @mock.patch.object(AlertNotifier, "send_email_batch")
    @mock.patch("alertnotifiers.requests.post")
    def test_routine_suppression_does_not_forward_every_event_to_email(self, post, email):
        self.exhausted_state(provider_blocked=True)
        self.assertFalse(AlertNotifier.send_phone_webhook_batch([_event()], webhook_url=_WEBHOOK))
        post.assert_not_called()
        email.assert_not_called()

    @mock.patch.object(AlertNotifier, "send_email_batch", return_value=True)
    def test_local_budget_warning_does_not_claim_urgent_delivery_is_stopped(self, email):
        AlertNotifier._send_budget_warning("ntfy", "local_daily_budget")
        body = email.call_args.args[0][0]["body"]
        self.assertIn("Routine ntfy", body)
        self.assertIn("Urgent ntfy alerts and all email remain exempt", body)

    @mock.patch("alertnotifiers.time.sleep")
    @mock.patch.object(AlertNotifier, "send_phone_webhook_batch", return_value=False)
    @mock.patch.object(AlertNotifier, "send_email_batch", return_value=True)
    def test_notify_uses_the_same_watchdog_urgency_for_email(self, email, phone, sleep):
        with mock.patch.dict(os.environ, {"ALERT_TO_EMAIL": "to@example.test"}):
            alertnotifiers.notify("Service unavailable", "details", "watchdog", "SYSTEM")
        email.assert_called_once()

    @mock.patch("alertnotifiers.time.sleep")
    @mock.patch.object(AlertNotifier, "send_phone_webhook_batch", return_value=True)
    @mock.patch.object(AlertNotifier, "send_email_batch", return_value=True)
    def test_notify_keeps_routine_email_opt_in(self, email, phone, sleep):
        with mock.patch.dict(os.environ, {"ALERT_TO_EMAIL": "to@example.test"}):
            alertnotifiers.notify("FILL", "details", "kraken", "HYPE")
        email.assert_not_called()

    @mock.patch("alertnotifiers.requests.post", return_value=_Response())
    def test_identical_ntfy_event_is_deduplicated_across_calls(self, post):
        first = AlertNotifier.send_phone_webhook_batch([_event()], webhook_url=_WEBHOOK)
        second = AlertNotifier.send_phone_webhook_batch([_event()], webhook_url=_WEBHOOK)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(post.call_count, 1)

    @mock.patch("alertnotifiers.requests.post", return_value=_Response())
    def test_normal_messages_cannot_consume_urgent_reserve(self, post):
        self.assertTrue(AlertNotifier.send_phone_webhook_batch([_event("FILL A")], webhook_url=_WEBHOOK))
        self.assertFalse(AlertNotifier.send_phone_webhook_batch([_event("FILL B")], webhook_url=_WEBHOOK))
        self.assertTrue(AlertNotifier.send_phone_webhook_batch([
            _event("ERORI WATCHDOG", source="watchdog"),
        ], webhook_url=_WEBHOOK))

        self.assertEqual(post.call_count, 2)
        with open(os.environ["NOTIFICATION_STATE_FILE"], encoding="utf-8") as handle:
            state = json.load(handle)
        self.assertEqual(state["channels"]["ntfy"]["sent"], 2)

    @mock.patch("alertnotifiers._ntfy_token", return_value="stale-token")
    @mock.patch("alertnotifiers.requests.post",
                side_effect=[_RejectedResponse(), _Response()])
    def test_rejected_ntfy_token_retries_once_without_credentials(self, post, token):
        self.assertTrue(AlertNotifier.send_phone_webhook_batch([_event()], webhook_url=_WEBHOOK))
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args_list[0].kwargs["headers"]["Authorization"],
            "Bearer stale-token")
        self.assertNotIn(
            "Authorization", post.call_args_list[1].kwargs["headers"])

    @mock.patch("alertnotifiers.smtplib.SMTP")
    def test_email_uses_same_cross_process_dedup_policy(self, smtp):
        smtp.return_value.__enter__.return_value = smtp.return_value
        alert = _event("ERORI WATCHDOG", source="watchdog")

        with mock.patch.dict(os.environ, {
            "SMTP_SERVER": "smtp.example.test",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "from@example.test",
            "SMTP_PASSWORD": "secret",
            "ALERT_TO_EMAIL": "to@example.test",
        }):
            self.assertTrue(AlertNotifier.send_email_batch([alert], subject="incident"))
            self.assertTrue(AlertNotifier.send_email_batch([alert], subject="incident"))

        self.assertEqual(smtp.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
