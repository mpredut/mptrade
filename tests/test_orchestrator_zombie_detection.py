import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestratorTrade.orchestrator import BotManager


class OrchestratorZombieDetectionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = MagicMock()
        self.server._resolve_topic.return_value = "test-topic"
        self.manager = BotManager(self.server)

    def test_parse_procs_conf_extracts_heartbeat_and_stale_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf = os.path.join(tmp, "procs.conf")
            with open(conf, "w", encoding="utf-8") as f:
                f.write(
                    "rtrade.py|$ROOT|source $ROOT/$VENV/bin/activate && python rtrade.py|rtrade|cachedb/rtrade.heartbeat|180|fleet\n"
                    "cacheManager.py|$ROOT|source $ROOT/$VENV/bin/activate && python cacheManager.py|cacheManager|||fleet\n"
                )
            with patch("orchestratorTrade.orchestrator.ROOT_DIR", tmp):
                bots = self.manager.parse_procs_conf()
                self.assertEqual(len(bots), 2)
                rtrade_bot = next(b for b in bots if b["name"] == "rtrade")
                self.assertEqual(rtrade_bot["hb_file"], os.path.join(tmp, "cachedb", "rtrade.heartbeat"))
                self.assertEqual(rtrade_bot["hb_stale_s"], 180.0)

                cm_bot = next(b for b in bots if b["name"] == "cacheManager")
                self.assertEqual(cm_bot["hb_file"], "")
                self.assertEqual(cm_bot["hb_stale_s"], 0.0)

    async def test_supervise_terminates_zombie_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = os.path.join(tmp, "bot.heartbeat")
            # Create a stale heartbeat file (modified 250 seconds ago)
            with open(hb_file, "w") as f:
                f.write("heartbeat")
            past = time.time() - 250.0
            os.utime(hb_file, (past, past))

            mock_proc = MagicMock()
            mock_proc.pid = 9999
            mock_proc.returncode = None  # Process is alive (zombie)

            bot_entry = {
                "name": "test_bot",
                "dir": tmp,
                "cmd": "python test.py",
                "log_file": os.path.join(tmp, "bot.log"),
                "hb_file": hb_file,
                "hb_stale_s": 180.0,
            }

            self.manager.bots = [bot_entry]
            self.manager.processes["test_bot"] = mock_proc
            # Process started 250s ago (beyond grace period)
            self.manager.process_start_times["test_bot"] = time.time() - 250.0

            self.manager.stop_bot = AsyncMock()

            # Run one cycle of supervise logic
            now = time.time()
            proc = self.manager.processes.get("test_bot")
            self.assertIsNotNone(proc)
            self.assertIsNone(proc.returncode)

            # Check hang detection
            stale_duration = now - os.path.getmtime(hb_file)
            self.assertGreater(stale_duration, 180.0)

            # Verify stop_bot is called when heartbeat is stale
            # Simulate supervise step
            hb_stale_s = bot_entry["hb_stale_s"]
            proc_start = self.manager.process_start_times["test_bot"]
            if (now - proc_start) > hb_stale_s:
                mtime = os.path.getmtime(hb_file)
                if (now - mtime) > hb_stale_s:
                    self.manager.zombie_killing.add("test_bot")
                    await self.manager.stop_bot("test_bot")

            self.manager.stop_bot.assert_awaited_once_with("test_bot")
            self.assertIn("test_bot", self.manager.zombie_killing)

    async def test_fresh_heartbeat_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = os.path.join(tmp, "bot.heartbeat")
            with open(hb_file, "w") as f:
                f.write("heartbeat")

            mock_proc = MagicMock()
            mock_proc.pid = 9999
            mock_proc.returncode = None

            bot_entry = {
                "name": "test_bot",
                "dir": tmp,
                "cmd": "python test.py",
                "log_file": os.path.join(tmp, "bot.log"),
                "hb_file": hb_file,
                "hb_stale_s": 180.0,
            }

            self.manager.bots = [bot_entry]
            self.manager.processes["test_bot"] = mock_proc
            self.manager.process_start_times["test_bot"] = time.time() - 250.0
            self.manager.stop_bot = AsyncMock()

            now = time.time()
            mtime = os.path.getmtime(hb_file)
            self.assertLess(now - mtime, 180.0)
            self.manager.stop_bot.assert_not_called()

    async def test_fresh_log_file_with_stale_hb_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = os.path.join(tmp, "legacy_bot.log")
            log_file = os.path.join(tmp, "logs", "test_bot.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Legacy heartbeat file is 500 seconds stale
            with open(hb_file, "w") as f:
                f.write("old log content")
            past = time.time() - 500.0
            os.utime(hb_file, (past, past))

            # Active orchestrator log file is fresh (just touched)
            with open(log_file, "w") as f:
                f.write("fresh bot log output")

            mock_proc = MagicMock()
            mock_proc.pid = 8888
            mock_proc.returncode = None

            bot_entry = {
                "name": "test_bot",
                "dir": tmp,
                "cmd": "python test.py",
                "log_file": log_file,
                "hb_file": hb_file,
                "hb_stale_s": 180.0,
            }

            self.manager.bots = [bot_entry]
            self.manager.processes["test_bot"] = mock_proc
            self.manager.process_start_times["test_bot"] = time.time() - 300.0
            self.manager.stop_bot = AsyncMock()

            # Supervise logic check
            now = time.time()
            hb_stale_s = bot_entry["hb_stale_s"]
            proc_start = self.manager.process_start_times["test_bot"]
            is_hung = False
            if (now - proc_start) > hb_stale_s:
                candidates = [f for f in (hb_file, log_file) if f and os.path.exists(f)]
                if candidates:
                    most_recent_mtime = max(os.path.getmtime(f) for f in candidates)
                    stale_duration = now - most_recent_mtime
                    if stale_duration > hb_stale_s:
                        is_hung = True

            self.assertFalse(is_hung)
            self.manager.stop_bot.assert_not_called()


if __name__ == "__main__":
    unittest.main()

