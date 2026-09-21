#!/usr/bin/env python3
import asyncio
import os
import sys
import json
import logging
import re
import time
import requests
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Orchestrator: %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, ROOT_DIR)\nfrom notify_engine.server import NotificationServer
class BotManager:
    def __init__(self, server: NotificationServer):
        self.server = server
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.log_files: Dict[str, Any] = {}

    def parse_procs_conf(self) -> List[Dict[str, str]]:
        bots = []
        conf_path = os.path.join(ROOT_DIR, "procs.conf")
        if not os.path.exists(conf_path):
            logging.error(f"Manifest not found: {conf_path}")
            return bots
            
        with open(conf_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) >= 7:
                    # pat | dir | start_cmd | label | hb_log | hb_stale_s | role
                    pat, dr, cmd, label, hblog, hbstale, role = parts[:7]
                    
                    dr = dr.replace("$ROOT", ROOT_DIR)
                    cmd = cmd.replace("$ROOT", ROOT_DIR).replace("$VENV", "myenv")
                    
                    if cmd and role in ("bot", "fleet"):
                        bots.append({"name": label or pat, "dir": dr, "cmd": cmd, "log_file": hblog})
        return bots

    async def _read_stream(self, stream, bot_name: str, log_file):
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            
            # 1. Print to console for orchestrator viewing
            sys.stdout.write(f"[{bot_name}] {decoded}")
            sys.stdout.flush()
            
            # 2. Write to original log file (so tail -f still works)
            if log_file:
                log_file.write(decoded)
                log_file.flush()
                
            # 3. Analyze for notifications
            self.server.process_line(decoded, bot_name)

    async def start_bot(self, bot: Dict[str, str]):
        name = bot["name"]
        cmd = bot["cmd"]
        directory = bot["dir"]
        
        # Clean up legacy bash wrappers (nohup, redirects, backgrounding)
        # e.g.: "source ... && nohup python3 hl_dca_bot.py >> log 2>&1 &"
        cmd = cmd.replace("nohup ", "")
        cmd = re.sub(r'>>\s*[^\s]+', '', cmd)
        cmd = cmd.replace("2>&1", "")
        cmd = cmd.removesuffix("&").strip()
        cmd = cmd.strip()
        
        # We wrap in bash -c because start_cmd often has `source activate && python ...`
        logging.info(f"Starting {name} in {directory}: {cmd}")
        process = await asyncio.create_subprocess_shell(
            cmd,
            executable='/bin/bash',
            cwd=directory,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        self.processes[name] = process
        
        log_fh = None
        log_path_rel = bot.get("log_file")
        if log_path_rel:
            abs_log_path = os.path.join(directory, log_path_rel)
            try:
                log_fh = open(abs_log_path, "a", encoding="utf-8")
                self.log_files[name] = log_fh
            except Exception as e:
                logging.error(f"Failed to open log {abs_log_path} for {name}: {e}")

        # Task to read stdout
        if process.stdout:
            task = asyncio.create_task(self._read_stream(process.stdout, name, log_fh))
            
        await process.wait()
        logging.warning(f"Bot {name} exited with code {process.returncode}")
        # Wait for the stream reader to finish before closing the log file
        if process.stdout:
            await task
        if log_fh:
            log_fh.close()
            
    async def restart_bot(self, name: str):
        if name in self.processes:
            logging.info(f"Restarting bot {name}...")
            proc = self.processes[name]
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

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
        """Watches config.env and procs.conf for changes to trigger restarts."""
        config_path = os.path.join(ROOT_DIR, "config.env")
        procs_path = os.path.join(ROOT_DIR, "procs.conf")
        hashes = {
            config_path: self._hash_file(config_path),
            procs_path: self._hash_file(procs_path)
        }
        
        while True:
            await asyncio.sleep(10) # check every 10s
            for path in list(hashes.keys()):
                curr = self._hash_file(path)
                if curr and curr != hashes[path]:
                    logging.warning(f"Config {path} changed! Restarting all affected bots...")
                    hashes[path] = curr
                    for name in self.processes.keys():
                        await self.restart_bot(name)
                        
                    # Also notify the user
                    self.server._send_ntfy(
                        "Config Reloaded", 
                        f"Detected change in {os.path.basename(path)}. All bots restarted automatically.", 
                        "high", 
                        self.server._resolve_topic("TRADES")
                    )

    async def supervise(self):
        bots = self.parse_procs_conf()
        if not bots:
            logging.error("No bots found to start.")
            return

        # Start hot reload watchdog
        asyncio.create_task(self._hot_reload_loop())

        while True:
            # We run the bots in an infinite loop. If they exit or get terminated (by reload), they restart.
            tasks = []
            for bot in bots:
                if bot["name"] not in self.processes or self.processes[bot["name"]].returncode is not None:
                    tasks.append(asyncio.create_task(self.start_bot(bot)))
            
            if tasks:
                await asyncio.gather(*tasks)
            await asyncio.sleep(2)

async def main():
    server = NotificationServer()
    manager = BotManager(server)
    logging.info("Starting Orchestrator...")
    await manager.supervise()

if __name__ == "__main__":
    asyncio.run(main())
