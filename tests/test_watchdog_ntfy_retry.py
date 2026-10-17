"""Watchdog tests: ntfy retry/rate-limit handling and OS resource watchdog checks."""
import os
import sys
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "orchestratorOS", "lib"))
sys.path.insert(0, os.path.join(ROOT, "orchestratorOS", "livecheck"))

import os_resources_watchdog as wd  # noqa: E402
import watchdog_common as wc  # noqa: E402


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
        self.assertEqual(mpost.call_count, 2)
        msleep.assert_called_once()

    def test_daily_429_does_not_retry_and_blocks_following_attempts(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep"):
            mpost.return_value = _Resp(429, text='{"code":42908,"error":"daily limit reached"}')
            ok = wc.send_ntfy("title", "message")
            blocked = wc.send_ntfy("second title", "second message")
        self.assertFalse(ok)
        self.assertFalse(blocked)
        self.assertEqual(mpost.call_count, 1)

    def test_success_first_try_no_sleep(self):
        with mock.patch("requests.post") as mpost, mock.patch("time.sleep") as msleep:
            mpost.side_effect = [_Resp(200)]
            ok = wc.send_ntfy("title", "message")
        self.assertTrue(ok)
        self.assertEqual(mpost.call_count, 1)
        msleep.assert_not_called()


def _proc(pid=10, ticks=100, rss=100, command="worker.py"):
    return {f"{pid}:1": {"pid": pid, "ticks": ticks, "rss_mb": rss, "command": command}}


def test_alerts_only_after_sustained_cpu_use():
    with tempfile.TemporaryDirectory() as tmp, patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
            patch.object(wd, "CONSECUTIVE", 3), patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_proc(ticks=100), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_proc(ticks=160), cpu_count=1)
        wd.check_once(now=240, total_ticks=1200, processes=_proc(ticks=220), cpu_count=1)
        alert.assert_not_called()
        wd.check_once(now=360, total_ticks=1300, processes=_proc(ticks=280), cpu_count=1)
        alert.assert_called_once()


def test_short_memory_spike_does_not_alert():
    with tempfile.TemporaryDirectory() as tmp, patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
            patch.object(wd, "MEMORY_THRESHOLD_MB", 500), \
            patch.object(wd, "CONSECUTIVE", 3), patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_proc(rss=800), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_proc(ticks=101, rss=100), cpu_count=1)
        alert.assert_not_called()


def test_sustained_memory_use_alerts_and_then_recovers():
    with tempfile.TemporaryDirectory() as tmp, patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
            patch.object(wd, "MEMORY_THRESHOLD_MB", 500), \
            patch.object(wd, "CONSECUTIVE", 2), patch.object(wd, "RECOVERY_CHECKS", 2), \
            patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_proc(rss=800), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_proc(ticks=101, rss=800), cpu_count=1)
        wd.check_once(now=240, total_ticks=1200, processes=_proc(ticks=102, rss=100), cpu_count=1)
        wd.check_once(now=360, total_ticks=1300, processes=_proc(ticks=103, rss=100), cpu_count=1)
        assert alert.call_count == 2


def test_managed_process_is_restarted_after_sustained_breach():
    proc = next(iter(_proc(rss=800, command="python worker.py").values()))
    state = {}
    with patch.object(wd, "AUTO_RESTART", True), \
            patch.object(wd, "_managed_script", return_value="worker.py"), \
            patch.object(wd, "_restart_allowed", return_value=True), \
            patch.object(wd.os, "kill") as kill:
        assert wd._restart_managed(proc, state, 1000) is True
        kill.assert_called_once_with(10, wd.signal.SIGTERM)
        assert state["restart_history"]["worker.py"] == [1000]


def test_unmanaged_process_is_never_restarted():
    proc = next(iter(_proc(rss=800, command="foreign").values()))
    with patch.object(wd, "_managed_script", return_value=None), \
            patch.object(wd.os, "kill") as kill:
        assert wd._restart_managed(proc, {}, 1000) is None
        kill.assert_not_called()


def test_parse_proc_stat_handles_spaces_and_parens_in_comm():
    # Fields after comm:
    # 0:state 1..10:ppid..majflt 11:utime(140) 12:stime(60) 13..18:cutime..itrealvalue(0)
    # 19:starttime(987654) 20:vsize(2147483648) 21:rss(4096)
    after_fields = [
        "S", "1", "100", "100", "0", "-1", "4194304",
        "10", "0", "0", "0",
        "140", "60",
        "0", "0", "20", "0", "1", "0",
        "987654", "2147483648", "4096", "18446744073709551615",
    ]
    stat_line = f"12345 (Web Content (2)) {' '.join(after_fields)}\n"
    comm, start_time, ticks, rss_pages = wd._parse_proc_stat(stat_line)
    assert comm == "Web Content (2)"
    assert start_time == 987654
    assert ticks == 200
    assert rss_pages == 4096


if __name__ == "__main__":
    unittest.main(verbosity=2)
