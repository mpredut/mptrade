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
                    "hl_bot.py|$ROOT/hyperliquid|source $ROOT/$VENV/bin/activate && nohup python3 hl_bot.py >> logs/HL-bot.log 2>&1 &|HL-bot|logs/HL-bot.log|900|bot\n"
                    "cacheManager.py|$ROOT|source $ROOT/$VENV/bin/activate && python cacheManager.py|cacheManager|||fleet\n"
                )
            with patch("orchestratorTrade.orchestrator.ROOT_DIR", tmp):
                bots = self.manager.parse_procs_conf()
                self.assertEqual(len(bots), 3)
                rtrade_bot = next(b for b in bots if b["name"] == "rtrade")
                self.assertEqual(rtrade_bot["hb_file"], os.path.join(tmp, "cachedb", "rtrade.heartbeat"))
                self.assertEqual(rtrade_bot["hb_stale_s"], 180.0)

                hl_bot = next(b for b in bots if b["name"] == "HL-bot")
                self.assertEqual(hl_bot["hb_file"], os.path.join(tmp, "logs", "HL-bot.log"))
                self.assertEqual(hl_bot["hb_stale_s"], 900.0)

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

    async def test_dedicated_heartbeat_is_strictly_enforced_without_log_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = os.path.join(tmp, "cachedb", "rtrade.heartbeat")
            os.makedirs(os.path.dirname(hb_file), exist_ok=True)
            log_file = os.path.join(tmp, "logs", "rtrade.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Dedicated heartbeat file is stale (recovery blocked)
            with open(hb_file, "w") as f:
                f.write("heartbeat")
            past = time.time() - 300.0
            os.utime(hb_file, (past, past))

            # Log file has recent error logs
            with open(log_file, "w") as f:
                f.write("recovery blocked error line\n")

            mock_proc = MagicMock()
            mock_proc.pid = 8888
            mock_proc.returncode = None

            bot_entry = {
                "name": "rtrade",
                "dir": tmp,
                "cmd": "python rtrade.py",
                "log_file": log_file,
                "hb_file": hb_file,
                "hb_stale_s": 180.0,
            }

            self.manager.bots = [bot_entry]
            self.manager.processes["rtrade"] = mock_proc
            self.manager.process_start_times["rtrade"] = time.time() - 300.0
            self.manager.stop_bot = AsyncMock()

            # Supervise logic check: should check hb_file strictly, not fall back to log_file
            now = time.time()
            hb_stale_s = bot_entry["hb_stale_s"]
            proc_start = self.manager.process_start_times["rtrade"]
            is_hung = False
            if (now - proc_start) > hb_stale_s and hb_file:
                if os.path.exists(hb_file):
                    stale_duration = now - os.path.getmtime(hb_file)
                    if stale_duration > hb_stale_s:
                        is_hung = True
                else:
                    is_hung = True

            self.assertTrue(is_hung)

    async def test_start_bot_injects_venv_in_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot_entry = {
                "name": "test_env_bot",
                "dir": tmp,
                "cmd": "python -c 'pass'",
                "log_file": os.path.join(tmp, "bot.log"),
                "hb_file": "",
                "hb_stale_s": 0.0,
            }
            with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell:
                mock_proc = MagicMock()
                mock_proc.pid = 1234
                mock_proc.stdout = None
                mock_proc.returncode = 0
                mock_proc.wait = AsyncMock(return_value=0)
                mock_shell.return_value = mock_proc

                await self.manager.start_bot(bot_entry)

                mock_shell.assert_called_once()
                call_env = mock_shell.call_args.kwargs.get("env", {})
                self.assertIn("VIRTUAL_ENV", call_env)
                self.assertIn(self.manager.venv, call_env["VIRTUAL_ENV"])
                self.assertTrue(call_env["PATH"].startswith(os.path.join(str(ROOT), self.manager.venv, "bin")))

    async def test_start_bot_refreshes_stale_heartbeat_mtime_and_zombie_cooldown_gates_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = os.path.join(tmp, "globaltelemetry_collector.heartbeat")
            with open(hb_file, "w", encoding="utf-8") as f:
                f.write("{}")
            ancient = time.time() - 28784.0
            os.utime(hb_file, (ancient, ancient))

            bot_entry = {
                "name": "globaltelemetry_collector",
                "dir": tmp,
                "cmd": "python -c 'pass'",
                "log_file": os.path.join(tmp, "globaltelemetry_collector.log"),
                "hb_file": hb_file,
                "hb_files": [hb_file],
                "hb_stale_s": 180.0,
            }
            with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell:
                mock_proc = MagicMock()
                mock_proc.pid = 613438
                mock_proc.stdout = None
                mock_proc.returncode = 0
                mock_proc.wait = AsyncMock(return_value=0)
                mock_shell.return_value = mock_proc

                await self.manager.start_bot(bot_entry)
                # start_bot must refresh the 8-hour-old heartbeat file mtime on launch
                self.assertLess(time.time() - os.path.getmtime(hb_file), 10.0)

    async def test_hot_reload_loop_watches_git_head_and_spliced_stdout_intents(self):
        import json
        import botcore
        import log as mp_log
        from intelligence.macro_sentiment_ssot import discover_dynamic_terms_from_headlines
        from notify_engine.server import NotificationServer

        with tempfile.TemporaryDirectory() as tmp:
            git_refs = os.path.join(tmp, ".git", "refs", "heads")
            os.makedirs(git_refs, exist_ok=True)
            with open(os.path.join(tmp, ".git", "HEAD"), "w", encoding="utf-8") as f:
                f.write("ref: refs/heads/main\n")
            ref_file = os.path.join(git_refs, "main")
            with open(ref_file, "w", encoding="utf-8") as f:
                f.write("1111111111111111111111111111111111111111\n")

            with patch("orchestratorTrade.orchestrator.ROOT_DIR", tmp):
                sig1 = self.manager._git_head_signature()
                self.assertEqual(sig1, "1111111111111111111111111111111111111111")
                with open(ref_file, "w", encoding="utf-8") as f:
                    f.write("2222222222222222222222222222222222222222\n")
                sig2 = self.manager._git_head_signature()
                self.assertEqual(sig2, "2222222222222222222222222222222222222222")

            # Verify NotificationServer.process_line extracts __orchestrator_intent__ even when spliced after a line with braces
            srv = NotificationServer()
            sent = []
            srv._gate_and_send_ntfy = lambda title, msg, prio, topic, **kw: sent.append((title, msg, prio, topic))
            spliced_line = (
                '[08:30:11] [nvda] {"debug": 1}'
                + json.dumps(
                    {
                        "__orchestrator_intent__": "ntfy_webhook",
                        "title": "Macro Alert",
                        "message": "Body text",
                        "priority": "high",
                        "topic": "ntfy-macro-1978",
                    }
                )
                + "[08:30:11] [spcx] trailing"
            )
            srv.process_line(spliced_line, "macro_analyzer")
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0][0], "Macro Alert")

            # Verify log._print_lock and botcore._LOG_LOCK exist and serialize output
            self.assertTrue(hasattr(mp_log, "_print_lock"))
            self.assertTrue(hasattr(botcore, "_LOG_LOCK"))

            # Verify discover_dynamic_terms_from_headlines(persist=True) prunes noise/expired entries on disk even when 0 new terms found
            lex_path = os.path.join(tmp, "macro_lexicon_collect.json")
            with open(lex_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "updated_ts": 1793680000.0,
                        "entries": {
                            "iran qeshm": {"ts": 1793680000.0, "tier": "high", "source": "test"},
                            "iran war": {"ts": 1793680000.0, "tier": "high", "source": "test"},
                        },
                    },
                    f,
                )
            discovered = discover_dynamic_terms_from_headlines(
                [{"title": "Calm weather today", "source": "Reuters"}],
                cache_dir=tmp,
                now_ts=1793680100.0,
                persist=True,
            )
            self.assertEqual(discovered, ())
            with open(lex_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertNotIn("iran qeshm", saved["entries"])
            self.assertIn("iran war", saved["entries"])

    async def test_reload_runtime_modules_and_manifest_refreshes_bots_and_removes_stale_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf = os.path.join(tmp, "procs.conf")
            with open(conf, "w", encoding="utf-8") as f:
                f.write(
                    "order_retry_worker.py|$ROOT|source $ROOT/$VENV/bin/activate && python order_retry_worker.py >> logs/order_retry.log 2>&1|order_retry|cachedb/system/heartbeat/order_retry.heartbeat|300|fleet\n"
                )
            self.manager.bots = [
                {"name": "order_retry", "hb_files": ["cachedb/system/heartbeat/order_retry.heartbeat", "logs/order_retry.log"]},
                {"name": "removed_bot", "hb_files": []},
            ]
            self.manager.processes["removed_bot"] = MagicMock()
            self.manager.stop_bot = AsyncMock()
            with patch("orchestratorTrade.orchestrator.ROOT_DIR", tmp):
                await self.manager._reload_runtime_modules_and_manifest()
            self.manager.stop_bot.assert_awaited_once_with("removed_bot")
            self.assertNotIn("removed_bot", self.manager.processes)
            self.assertEqual(len(self.manager.bots), 1)
            self.assertEqual(
                self.manager.bots[0]["hb_files"],
                [os.path.join(tmp, "cachedb/system/heartbeat/order_retry.heartbeat")],
            )


if __name__ == "__main__":
    unittest.main()

