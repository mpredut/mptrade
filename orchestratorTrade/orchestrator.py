#!/usr/bin/env python3
import asyncio
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
from notify_engine.server import NotificationServer


class BotManager:
    def __init__(self, server: NotificationServer):
        self.server = server
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.log_files: Dict[str, Any] = {}
        self.bots: List[Dict[str, str]] = []
        self.restart_backoff: Dict[str, float] = {}
        self.restart_delays: Dict[str, float] = {}
        self.starting: set[str] = set()
        self.running = True

    def parse_procs_conf(self) -> List[Dict[str, str]]:
        bots = []
        conf_path = os.path.join(ROOT_DIR, "procs.conf")
        if not os.path.exists(conf_path):
            logging.error(f"Manifest not found: {conf_path}")
            return bots

        with open(conf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) >= 7:
                    # pat | dir | start_cmd | label | hb_log | hb_stale_s | role
                    pat, dr, cmd, label, hblog, hbstale, role = parts[:7]

                    venv = os.environ.get("VENV")
                    if not venv:
                        for cand in (".venv", "myenv"):
                            if os.path.exists(os.path.join(ROOT_DIR, cand, "bin", "activate")):
                                venv = cand
                                break
                    venv = venv or "myenv"
                    dr = dr.replace("$ROOT", ROOT_DIR)
                    cmd = cmd.replace("$ROOT", ROOT_DIR).replace("$VENV", venv)

                    if cmd and role in ("bot", "fleet"):
                        name = label or pat
                        log_file = hblog.strip() if hblog else ""
                        if not log_file or log_file.endswith(".heartbeat"):
                            clean_stem = (label or pat).replace(".py", "")
                            log_file = os.path.join(ROOT_DIR, "logs", f"{clean_stem}.log")
                        bots.append({"name": name, "dir": dr, "cmd": cmd, "log_file": log_file})
        return bots

    async def _read_stream(self, stream, bot_name: str, log_file):
        verbose_stdout = os.environ.get("ORCHESTRATOR_VERBOSE_STDOUT", "0") in ("1", "true", "True")
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")

            # 1. Write to dedicated log file (so tail -f logs/<bot>.log works cleanly)
            if log_file:
                log_file.write(decoded)
                log_file.flush()

            # 2. Analyze for notifications
            self.server.process_line(decoded, bot_name)

            # 3. Print to console / journalctl ONLY for warnings, errors, or if verbose
            if verbose_stdout or any(err in decoded.upper() for err in ("ERROR", "CRITICAL", "EXCEPTION", "TRACEBACK", "RECOVERY BLOCKED", "WARN")):
                sys.stdout.write(f"[{bot_name}] {decoded}")
                sys.stdout.flush()

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
            bot_env = {**os.environ, "MPTRADE_ORCHESTRATED": "1", "PYTHONUNBUFFERED": "1"}
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
                self.log_files[name] = log_fh
            except Exception as e:
                logging.error(f"Failed to open log {abs_log_path} for {name}: {e}")

        if process.stdout:
            task = asyncio.create_task(self._read_stream(process.stdout, name, log_fh))

        try:
            await process.wait()
        except asyncio.CancelledError:
            await self.stop_bot(name)
            raise
        finally:
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
                await task
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
                    await asyncio.wait_for(proc.wait(), timeout=4.0)
                except (asyncio.TimeoutError, TimeoutError):
                    logging.warning(f"Bot {name} did not exit within 4s, sending SIGKILL...")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, Exception):
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except (asyncio.TimeoutError, TimeoutError):
                        pass

    async def restart_bot(self, name: str):
        await self.stop_bot(name)

    async def shutdown(self):
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
        hashes = {
            config_path: self._hash_file(config_path),
            procs_path: self._hash_file(procs_path),
            instruments_path: self._hash_file(instruments_path),
        }

        while self.running:
            await asyncio.sleep(10)
            for path in list(hashes.keys()):
                curr = self._hash_file(path)
                if curr and curr != hashes[path]:
                    logging.warning(f"Config {path} changed! Applying reload...")
                    hashes[path] = curr
                    if path == procs_path:
                        old_names = {b["name"] for b in self.bots}
                        self.bots = self.parse_procs_conf()
                        new_names = {b["name"] for b in self.bots}
                        # Cleanly terminate bots removed from manifest
                        for removed in (old_names - new_names):
                            logging.info(f"Bot {removed} was removed from manifest. Stopping...")
                            await self.stop_bot(removed)
                            self.processes.pop(removed, None)

                    for name in list(self.processes.keys()):
                        await self.restart_bot(name)

                    self.server._send_ntfy(
                        "Config Reloaded",
                        f"Detected change in {os.path.basename(path)}. All bots reloaded automatically.",
                        "high",
                        self.server._resolve_topic("TRADES")
                    )

    async def supervise(self):
        self.bots = self.parse_procs_conf()
        if not self.bots:
            logging.error("No bots found to start.")
            return

        asyncio.create_task(self._hot_reload_loop())
        asyncio.create_task(self.server.flush_queue_loop())

        while self.running:
            now = time.time()
            for bot in self.bots:
                name = bot["name"]
                if name in self.starting:
                    continue
                proc = self.processes.get(name)
                if proc is None or proc.returncode is not None:
                    next_allowed = self.restart_backoff.get(name, 0)
                    if now >= next_allowed:
                        self.starting.add(name)
                        asyncio.create_task(self.start_bot(bot))
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
