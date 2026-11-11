import asyncio
from collections import defaultdict, deque
import hashlib
import importlib
import logging
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
from botcore import evaluate_heartbeat_files_freshness, parse_procs_manifest
botcore.load_root_config()
from cachepaths import organize_cachedb_subdomains, resolve_cachedb_filepath
import notify_engine.alertnotifiers as _an
import notify_engine.mailer as _mailer
import notify_engine.server as _ns
from notify_engine.server import NotificationServer
from state_io import read_bytes_file


def _bot_hb_files(bot: Dict[str, Any]) -> List[str]:
    return bot.get("hb_files") or ([bot["hb_file"]] if bot.get("hb_file") else [])


def _close_process_transport(proc: Any) -> None:
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


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
        self.venv = os.environ.get("VENV") or next(
            (cand for cand in (".venv", "myenv") if os.path.exists(os.path.join(ROOT_DIR, cand, "bin", "activate"))),
            "myenv",
        )
        if "ORCHESTRATOR_VERBOSE_STDOUT" not in os.environ:
            raise KeyError("Missing required env key 'ORCHESTRATOR_VERBOSE_STDOUT' in config.env")
        self.verbose_stdout = botcore._parse_bool_flag(os.environ["ORCHESTRATOR_VERBOSE_STDOUT"], False)
        if "ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC" not in os.environ:
            raise KeyError("Missing required env key 'ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC' in config.env")
        self.zombie_cooldown_sec = botcore.required_float_env("ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC")
        if self.zombie_cooldown_sec < 0:
            raise ValueError(f"Invalid ORCHESTRATOR_ZOMBIE_COOLDOWN_SEC: {self.zombie_cooldown_sec!r}")
        self._shutting_down = False
        self._reloading = False

    def parse_procs_conf(self) -> List[Dict[str, Any]]:
        conf_path = os.path.join(ROOT_DIR, "procs.conf")
        if not os.path.exists(conf_path):
            logging.error(f"Manifest not found: {conf_path}")
            return []
        return parse_procs_manifest(
            ROOT_DIR,
            venv=self.venv,
            roles=("bot", "fleet"),
            include_stdout_with_heartbeat_file=False,
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
                    fresh_env.update(botcore.parse_dotenv(config_env_path))
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
        for raw_hbf in _bot_hb_files(bot):
            hbf = str(resolve_cachedb_filepath(raw_hbf)) if raw_hbf else ""
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
            _close_process_transport(process)

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
            _close_process_transport(proc)

    def _ordered_process_names_reversed(self) -> list[str]:
        manifest_names = [
            b["name"]
            for b in getattr(self, "bots", ())
            if isinstance(b, dict) and b.get("name") in self.processes
        ]
        seen = set(manifest_names)
        extra_names = [n for n in self.processes.keys() if n not in seen]
        ordered = list(reversed(manifest_names + extra_names))
        core_writers = [n for n in ("cacheManager",) if n in ordered]
        if core_writers:
            ordered = [n for n in ordered if n not in core_writers] + core_writers
        return ordered

    def _ordered_bots_for_start(self) -> list[Dict[str, Any]]:
        bots_list = list(getattr(self, "bots", ()) or ())
        priority = {"cacheManager": 0}
        return sorted(
            bots_list,
            key=lambda b: priority.get(b.get("name") if isinstance(b, dict) else "", 2),
        )

    async def shutdown(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        logging.info("Shutting down all bots gracefully...")
        self.running = False
        for name in self._ordered_process_names_reversed():
            await self.stop_bot(name)
            self.processes.pop(name, None)

    def _hash_file(self, filepath: str) -> str:
        data = read_bytes_file(filepath, default=None)
        return hashlib.md5(data).hexdigest() if data is not None else ""

    def _git_head_signature(self) -> str:
        git_dir = os.path.join(ROOT_DIR, ".git")
        head_bytes = read_bytes_file(os.path.join(git_dir, "HEAD"), default=None)
        if not head_bytes:
            return ""
        head_str = head_bytes.decode("utf-8", errors="replace").strip()
        if head_str.startswith("ref:"):
            ref_rel = head_str.split(":", 1)[1].strip()
            ref_bytes = read_bytes_file(os.path.join(git_dir, ref_rel), default=None)
            if ref_bytes:
                return ref_bytes.decode("utf-8", errors="replace").strip()
            packed_bytes = read_bytes_file(os.path.join(git_dir, "packed-refs"), default=None)
            if packed_bytes:
                return hashlib.md5(head_bytes + b"\n" + packed_bytes).hexdigest()
        return head_str

    async def _reload_runtime_modules_and_manifest(self) -> None:
        global evaluate_heartbeat_files_freshness, parse_procs_manifest, NotificationServer
        try:
            importlib.reload(botcore)
            cfg_path = os.path.join(ROOT_DIR, "config.env")
            if os.path.exists(cfg_path):
                botcore.load_dotenv(cfg_path, override=True)
            evaluate_heartbeat_files_freshness = botcore.evaluate_heartbeat_files_freshness
            parse_procs_manifest = botcore.parse_procs_manifest
            if type(getattr(self, "server", None)).__name__ == "NotificationServer":
                importlib.reload(_an)
                importlib.reload(_mailer)
                importlib.reload(_ns)
                NotificationServer = _ns.NotificationServer
                self.server.__class__ = _ns.NotificationServer
                self.server.ntfy_token = self.server._load_ntfy_token()
                self.server.rules = self.server._load_rules()
        except Exception as exc:
            logging.warning(f"Could not reload runtime heartbeat modules: {exc}")
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

    async def _hot_reload_loop(self):
        """Watches config files, core supervisor modules, and git HEAD for changes to trigger restarts."""
        config_path = os.path.join(ROOT_DIR, "config.env")
        procs_path = os.path.join(ROOT_DIR, "procs.conf")
        instruments_path = os.path.join(ROOT_DIR, "instruments.conf")
        order_guard_path = os.path.join(ROOT_DIR, "order_guard.conf")
        macro_conf_path = os.path.join(ROOT_DIR, "macro_intelligence.conf")
        self_path = os.path.abspath(__file__)
        botcore_path = os.path.join(ROOT_DIR, "botcore.py")
        trs_path = os.path.join(ROOT_DIR, "orchestratorOS", "lib", "technical_repair_ssot.py")
        hashes = {
            config_path: self._hash_file(config_path),
            procs_path: self._hash_file(procs_path),
            instruments_path: self._hash_file(instruments_path),
            order_guard_path: self._hash_file(order_guard_path),
            macro_conf_path: self._hash_file(macro_conf_path),
            self_path: self._hash_file(self_path),
            botcore_path: self._hash_file(botcore_path),
            trs_path: self._hash_file(trs_path),
        }
        last_git_head = self._git_head_signature()

        while self.running:
            await asyncio.sleep(10)
            changed_paths = []
            for path in list(hashes.keys()):
                curr = self._hash_file(path)
                if curr and curr != hashes[path]:
                    logging.warning(f"Config/module {path} changed! Applying reload...")
                    hashes[path] = curr
                    changed_paths.append(path)

            curr_git_head = self._git_head_signature()
            if curr_git_head and last_git_head and curr_git_head != last_git_head:
                logging.warning(f"Git HEAD changed ({last_git_head[:8]} -> {curr_git_head[:8]})! Applying reload...")
                last_git_head = curr_git_head
                changed_paths.append(".git/HEAD")
            elif curr_git_head and not last_git_head:
                last_git_head = curr_git_head

            if not changed_paths:
                continue

            if self_path in changed_paths and os.environ.get("INVOCATION_ID"):
                logging.warning("Orchestrator source changed under systemd; shutting down for clean service restart...")
                await self.shutdown()
                return

            await self._reload_runtime_modules_and_manifest()

            self._reloading = True
            try:
                for name in self._ordered_process_names_reversed():
                    await self.stop_bot(name)
                    self.restart_backoff[name] = 0
                # Start core cache writers immediately before releasing _reloading so
                # trading bots never wake up before account/price cache writers are live.
                for bot in self._ordered_bots_for_start():
                    bname = bot.get("name")
                    if bname == "cacheManager" and bname not in self.starting:
                        proc = self.processes.get(bname)
                        if proc is None or proc.returncode is not None:
                            self.starting.add(bname)
                            asyncio.create_task(self.start_bot(bot))
                await asyncio.sleep(0.5)
            finally:
                self._reloading = False

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
                    hb_files = _bot_hb_files(b)
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

            for bot in self._ordered_bots_for_start():
                if getattr(self, "_reloading", False):
                    continue
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
                    hb_files = _bot_hb_files(bot)
                    hb_stale_s = bot.get("hb_stale_s", 0.0)
                    if hb_stale_s > 0 and hb_files:
                        proc_start = self.process_start_times.get(name, now)
                        if proc_start > now:
                            proc_start = now
                            self.process_start_times[name] = now
                        if (now - proc_start) > hb_stale_s:
                            hb_fresh, _, max_stale_duration, stale_file = evaluate_heartbeat_files_freshness(
                                hb_files,
                                float(hb_stale_s),
                                now,
                                fallback_start_ts=proc_start,
                            )
                            is_hung = not hb_fresh

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
