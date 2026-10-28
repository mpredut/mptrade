import asyncio
from collections import defaultdict, deque
import logging
import math
import os
import re
import signal
import sys
import time
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Orchestrator: %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import botcore
botcore.load_dotenv(os.path.join(ROOT_DIR, "config.env"))
from cachepaths import organize_cachedb_subdomains, resolve_cachedb_filepath
from notify_engine.server import NotificationServer
from orchestratorOS.lib.technical_repair_ssot import parse_procs_manifest


class BotManager:
    def __init__(self, server: NotificationServer):
        self.server = server
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.log_files: Dict[str, Any] = {}
        self.bots: List[Dict[str, Any]] = []
        self.restart_backoff: Dict[str, float] = {}
        self.restart_delays: Dict[str, float] = {}
        self.starting: set[str] = set()
        self.process_start_times: Dict[str, float] = {}
        self.zombie_killing: set[str] = set()
        self.zombie_alert_cooldowns: Dict[str, float] = {}
        self.recent_lines: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.last_hb_status_log = time.time()
        self.running = True
        self.venv = self._resolve_venv()
        if "ORCHESTRATOR_VERBOSE_STDOUT" not in os.environ:
            raise KeyError("Missing required env key 'ORCHESTRATOR_VERBOSE_STDOUT' in config.env")
        self.verbose_stdout = botcore._parse_bool_flag(os.environ["ORCHESTRATOR_VERBOSE_STDOUT"], False)
        if "ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC" not in os.environ:
            raise KeyError("Missing required env key 'ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC' in config.env")
        self.zombie_cooldown_sec = botcore.required_float_env("ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC")
        if self.zombie_cooldown_sec < 0:
            raise ValueError(f"Invalid ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC: {self.zombie_cooldown_sec!r}")
        self._shutting_down = False

    @staticmethod
    def _resolve_venv() -> str:
        venv = os.environ.get("VENV")
        if not venv:
            for cand in (".venv", "myenv"):
                if os.path.exists(os.path.join(ROOT_DIR, cand, "bin", "activate")):
                    venv = cand
                    break
        return venv or "myenv"

    def parse_procs_conf(self) -> List[Dict[str, Any]]:
        conf_path = os.path.join(ROOT_DIR, "procs.conf")
        if not os.path.exists(conf_path):
            logging.error(f"Manifest not found: {conf_path}")
            return []
        return parse_procs_manifest(
            ROOT_DIR,
            venv=self.venv,
            roles=("bot", "fleet"),
            include_stdout_with_heartbeat_file=True,
            require_cmd=True,
        )

    async def _read_stream(self, stream, bot_name: str, log_file):
        verbose_stdout = self.verbose_stdout
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            clean_line = decoded.strip()
            if clean_line:
                self.recent_lines[bot_name].append(clean_line)

            # 1. Write to dedicated log file (so tail -f logs/<bot>.log works cleanly)
            if log_file:
                try:
                    log_file.write(decoded)
                    log_file.flush()
                except OSError:
                    pass

            # 2. Analyze for notifications
            try:
                self.server.process_line(decoded, bot_name)
            except Exception as exc:
                logging.debug(f"Notification process_line error for {bot_name}: {exc}")

            # 3. Print to console / journalctl ONLY for warnings, errors, or if verbose
            if verbose_stdout or any(err in decoded.upper() for err in ("ERROR", "CRITICAL", "EXCEPTION", "TRACEBACK", "RECOVERY", "WARN")):
                try:
                    sys.stdout.write(f"[{bot_name}] {decoded}")
                    sys.stdout.flush()
                except OSError:
                    pass

    async def start_bot(self, bot: Dict[str, str]):
        if not self.running:
            return
        name = bot["name"]
        cmd = bot["cmd"]
        directory = bot["dir"]

        # Clean up legacy bash wrappers (nohup, redirects, backgrounding)
        cmd = cmd.replace("nohup ", "")
        cmd = re.sub(r'>>\s*[^\s]+', '', cmd)
        cmd = cmd.replace("2>&1", "")
        cmd = cmd.removesuffix("&").strip()
        cmd = cmd.strip()

        logging.info(f"Starting {name} in {directory}: {cmd}")
        start_time = time.time()
        log_fh = None
        task = None
        try:
            fresh_env = dict(os.environ)
            config_env_path = os.path.join(ROOT_DIR, "config.env")
            if os.path.exists(config_env_path):
                try:
                    from botcore import _dotenv_pairs
                    pairs, ok = _dotenv_pairs(config_env_path)
                    if ok:
                        for k, v in pairs:
                            fresh_env[k] = v
                except Exception as env_err:
                    logging.warning(f"Could not parse fresh config.env for {name}: {env_err}")
            venv_bin = os.path.join(ROOT_DIR, self.venv, "bin")
            bot_env = {
                **fresh_env,
                "MPTRADE_ORCHESTRATED": "1",
                "PYTHONUNBUFFERED": "1",
                "VIRTUAL_ENV": os.path.join(ROOT_DIR, self.venv),
                "PATH": f"{venv_bin}:{fresh_env.get('PATH', '')}",
            }
            process = await asyncio.create_subprocess_shell(
                cmd,
                executable='/bin/bash',
                cwd=directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=bot_env
            )
            self.processes[name] = process
            self.process_start_times[name] = start_time
            self.zombie_killing.discard(name)
        except Exception as e:
            logging.error(f"Failed to launch {name}: {e}")
            self.restart_backoff[name] = time.time() + 5
            return
        finally:
            self.starting.discard(name)

        log_path = bot.get("log_file")
        if log_path:
            abs_log_path = log_path if os.path.isabs(log_path) else os.path.join(directory, log_path)
            try:
                os.makedirs(os.path.dirname(abs_log_path), exist_ok=True)
                log_fh = open(abs_log_path, "a", encoding="utf-8")
                try:
                    os.utime(abs_log_path, None)
                except OSError:
                    pass
                self.log_files[name] = log_fh
            except Exception as e:
                logging.error(f"Failed to open log {abs_log_path} for {name}: {e}")

        # Refresh mtime on existing dedicated heartbeat files on launch so a newly
        # started process is not immediately blamed for a predecessor's stale file.
        for hbf in (bot.get("hb_files") or ([bot["hb_file"]] if bot.get("hb_file") else [])):
            if hbf and os.path.exists(hbf):
                try:
                    os.utime(hbf, None)
                except OSError:
                    pass

        if process.stdout:
            task = asyncio.create_task(self._read_stream(process.stdout, name, log_fh))

        try:
            await process.wait()
        except asyncio.CancelledError:
            await self.stop_bot(name)
            raise
        finally:
            self.process_start_times.pop(name, None)
            self.zombie_killing.discard(name)
            duration = time.time() - start_time
            logging.warning(f"Bot {name} exited with code {process.returncode} (ran for {duration:.1f}s)")

            # Exponential backoff on rapid crash loop
            if duration < 5:
                prev_delay = self.restart_delays.get(name, 2.5)
                new_delay = min(prev_delay * 2, 60.0)
                self.restart_delays[name] = new_delay
                self.restart_backoff[name] = time.time() + new_delay
                logging.warning(f"Bot {name} rapid crash detected; backoff for {new_delay:.1f}s")
            else:
                self.restart_delays[name] = 5.0
                self.restart_backoff[name] = 0

            if task:
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, TimeoutError):
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                except Exception:
                    pass
            if log_fh:
                try:
                    log_fh.close()
                except Exception:
                    pass
                self.log_files.pop(name, None)

    async def stop_bot(self, name: str, sig=signal.SIGTERM):
        if name in self.processes:
            proc = self.processes[name]
            if isinstance(proc, asyncio.subprocess.Process) and proc.returncode is None:
                logging.info(f"Stopping bot {name} (pid={proc.pid})...")
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logging.warning(f"Failed to signal process group for {name}: {e}")
                    try:
                        proc.send_signal(sig)
                    except Exception:
                        pass

                try:
                    await asyncio.wait_for(proc.wait(), timeout=4.0)
                except (asyncio.TimeoutError, TimeoutError):
                    logging.warning(f"Bot {name} did not exit within 4s, sending SIGKILL...")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, Exception):
                        pass
                    try:
                        proc.kill()
                    except (ProcessLookupError, Exception):
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except (asyncio.TimeoutError, TimeoutError):
                        pass

    async def restart_bot(self, name: str):
        await self.stop_bot(name)
        self.restart_backoff[name] = 0

    async def shutdown(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        logging.info("Shutting down all bots gracefully...")
        self.running = False
        for name in list(self.processes.keys()):
            await self.stop_bot(name)

    def _hash_file(self, filepath: str) -> str:
        import hashlib
        if not os.path.exists(filepath):
            return ""
        hasher = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception:
            return ""

    async def _hot_reload_loop(self):
        """Watches config.env, procs.conf, and instruments.conf for changes to trigger restarts."""
        config_path = os.path.join(ROOT_DIR, "config.env")
        procs_path = os.path.join(ROOT_DIR, "procs.conf")
        instruments_path = os.path.join(ROOT_DIR, "instruments.conf")
        order_guard_path = os.path.join(ROOT_DIR, "order_guard.conf")
        hashes = {
            config_path: self._hash_file(config_path),
            procs_path: self._hash_file(procs_path),
            instruments_path: self._hash_file(instruments_path),
            order_guard_path: self._hash_file(order_guard_path),
        }

        while self.running:
            await asyncio.sleep(10)
            changed_paths = []
            for path in list(hashes.keys()):
                curr = self._hash_file(path)
                if curr and curr != hashes[path]:
                    logging.warning(f"Config {path} changed! Applying reload...")
                    hashes[path] = curr
                    changed_paths.append(path)

            if not changed_paths:
                continue

            if procs_path in changed_paths:
                new_bots = self.parse_procs_conf()
                if new_bots:
                    old_names = {b["name"] for b in self.bots}
                    self.bots = new_bots
                    new_names = {b["name"] for b in self.bots}
                    # Cleanly terminate bots removed from manifest
                    for removed in (old_names - new_names):
                        logging.info(f"Bot {removed} was removed from manifest. Stopping...")
                        await self.stop_bot(removed)
                        self.processes.pop(removed, None)

            for name in list(self.processes.keys()):
                await self.restart_bot(name)

    async def supervise(self):
        try:
            organize_cachedb_subdomains(os.path.join(ROOT_DIR, "cachedb"))
        except Exception as exc:
            logging.warning(f"Could not organize cachedb subdomains on startup: {exc}")
        self.bots = self.parse_procs_conf()
        if not self.bots:
            logging.error("No bots found to start.")
            return

        asyncio.create_task(self._hot_reload_loop())
        asyncio.create_task(self.server.flush_queue_loop())

        while self.running:
            now = time.time()
            if now - getattr(self, "last_hb_status_log", 0) >= 300.0:
                self.last_hb_status_log = now
                hb_statuses = []
                for b in self.bots:
                    hb_files = b.get("hb_files") or ([b["hb_file"]] if b.get("hb_file") else [])
                    if b.get("hb_stale_s", 0) > 0 and hb_files:
                        for raw_hbf in hb_files:
                            hbf = resolve_cachedb_filepath(raw_hbf)
                            if os.path.exists(hbf):
                                try:
                                    age = max(0.0, now - os.path.getmtime(hbf))
                                    hb_statuses.append(f"{b['name']} ({os.path.basename(hbf)}: {age:.1f}s/{b['hb_stale_s']:.0f}s)")
                                except OSError:
                                    hb_statuses.append(f"{b['name']} ({os.path.basename(hbf)}: missing)")
                            else:
                                hb_statuses.append(f"{b['name']} ({os.path.basename(hbf)}: missing)")
                if hb_statuses:
                    logging.info(f"Heartbeat healthcheck status: {', '.join(hb_statuses)}")

            for bot in self.bots:
                name = bot["name"]
                if name in self.starting:
                    continue
                proc = self.processes.get(name)
                if proc is None or proc.returncode is not None:
                    next_allowed = self.restart_backoff.get(name, 0)
                    if next_allowed > now + 65.0:
                        next_allowed = now
                        self.restart_backoff[name] = now
                    if now >= next_allowed:
                        self.starting.add(name)
                        asyncio.create_task(self.start_bot(bot))
                elif name not in self.zombie_killing:
                    # Zombie / Hang detection: check if process is alive but heartbeat is stale
                    hb_files = bot.get("hb_files") or ([bot["hb_file"]] if bot.get("hb_file") else [])
                    hb_stale_s = bot.get("hb_stale_s", 0.0)
                    if hb_stale_s > 0 and hb_files:
                        proc_start = self.process_start_times.get(name, now)
                        if proc_start > now:
                            proc_start = now
                            self.process_start_times[name] = now
                        if (now - proc_start) > hb_stale_s:
                            is_hung = False
                            stale_file = ""
                            max_stale_duration = 0.0

                            # Every heartbeat file must indicate life (ALL must be fresh)
                            for raw_hbf in hb_files:
                                hbf = resolve_cachedb_filepath(raw_hbf)
                                if os.path.exists(hbf):
                                    try:
                                        mtime = os.path.getmtime(hbf)
                                        duration = now - mtime
                                        if duration > hb_stale_s:
                                            is_hung = True
                                            if duration > max_stale_duration:
                                                max_stale_duration = duration
                                                stale_file = hbf
                                    except Exception as exc:
                                        logging.warning(f"Failed to check heartbeat mtime for {name} ({hbf}): {exc}")
                                else:
                                    is_hung = True
                                    duration = now - proc_start
                                    if duration > max_stale_duration:
                                        max_stale_duration = duration
                                        stale_file = hbf

                            if is_hung:
                                recent_logs = list(self.recent_lines.get(name, []))[-5:]
                                last_activity = "\n".join(f"  > {l}" for l in recent_logs) if recent_logs else "  (no recent logs)"
                                logging.critical(
                                    f"Bot {name} (PID {proc.pid}) is HUNG / ZOMBIE! "
                                    f"Heartbeat {stale_file} stale by {max_stale_duration:.1f}s (> {hb_stale_s:.0f}s threshold).\n"
                                    f"Recent bot activity:\n{last_activity}\n"
                                    f"Terminating zombie process..."
                                )
                                self.zombie_killing.add(name)
                                zombie_cd = self.zombie_cooldown_sec
                                cd_until = self.zombie_alert_cooldowns.get(name, 0.0)
                                if cd_until > now + zombie_cd + 5.0:
                                    cd_until = 0.0
                                if now >= cd_until:
                                    self.zombie_alert_cooldowns[name] = now + zombie_cd
                                    alert_body = (
                                        f"Process {name} (PID {proc.pid}) hung: heartbeat {os.path.basename(stale_file)} "
                                        f"stale by {int(max_stale_duration)}s > {int(hb_stale_s)}s.\n"
                                        f"Last logs:\n{last_activity}"
                                    )
                                    zombie_title = f"Zombie Bot Terminated: {name}"
                                    self.server._send_ntfy(
                                        zombie_title,
                                        alert_body,
                                        "urgent",
                                        self.server._resolve_topic("ERROR")
                                    )
                                asyncio.create_task(self.stop_bot(name))
            await asyncio.sleep(2)


async def main():
    server = NotificationServer()
    manager = BotManager(server)

    # Register OS signals for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(manager.shutdown()))
        except NotImplementedError:
            pass

    logging.info("Starting Orchestrator...")
    try:
        await manager.supervise()
    except asyncio.CancelledError:
        pass
    finally:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
