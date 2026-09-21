import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "orchestratorOS", "livecheck"))
import os_resources_watchdog as wd


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
            patch.object(wd, "CONSECUTIVE", 3), patch.object(wd.wc, "alert") as alert:
        wd.check_once(now=0, total_ticks=1000, processes=_proc(rss=800), cpu_count=1)
        wd.check_once(now=120, total_ticks=1100, processes=_proc(ticks=101, rss=100), cpu_count=1)
        alert.assert_not_called()


def test_sustained_memory_use_alerts_and_then_recovers():
    with tempfile.TemporaryDirectory() as tmp, patch.object(wd, "STATE_FILE", os.path.join(tmp, "state")), \
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
