"""watchdog_common.send_ntfy: retries once on 429 (burst rate-limit on ntfy.sh),
honouring Retry-After instead of losing the message."""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "orchestratorOS", "lib"))

import watchdog_common as wc


class _Resp:
    def __init__(self, status, headers=None, text=""):
        self.status_code = status
        self.headers = headers or {}
        self.text = text


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
            mpost.side_effect = [_Resp(429, {"Retry-After": "1"}), _Resp(200)]
            ok = wc.send_ntfy("title", "message")
        self.assertTrue(ok)
        self.assertEqual(mpost.call_count, 2)     # retried
        msleep.assert_called_once()               # waited Retry-After

    def test_daily_429_does_not_retry_and_blocks_following_attempts(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep"):
            mpost.return_value = _Resp(429, text='{"code":42908,"error":"daily limit reached"}')
            ok = wc.send_ntfy("titlu", "mesaj")
            blocked = wc.send_ntfy("alt titlu", "alt mesaj")
        self.assertFalse(ok)
        self.assertFalse(blocked)
        self.assertEqual(mpost.call_count, 1)

    def test_success_first_try_no_sleep(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep") as msleep:
            mpost.side_effect = [_Resp(200)]
            ok = wc.send_ntfy("titlu", "mesaj")
        self.assertTrue(ok)
        self.assertEqual(mpost.call_count, 1)
        msleep.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
