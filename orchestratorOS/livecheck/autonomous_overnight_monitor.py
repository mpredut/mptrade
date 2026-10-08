#!/usr/bin/env python3
"""Autonomous Overnight Fleet & Backtest Monitor.

Continuously tracks:
1. Orchestrator Fleet & Bot Heartbeats (all processes defined in procs.conf).
2. Risk Guards & Telemetry Sensors (Pillars 1-4, Geopolitical, Sentiment, Whale positioning).
3. Autonomous AI Supervisor & Intent Gateway status in ENFORCE mode.
4. DEV Machine Overnight Backtest Sweeps execution and progress.
5. OS Resource Health (CPU load, Memory, Disk space).

Persists periodic health status to cachedb/overnight_monitor_status.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autonomous_overnight_monitor")


@dataclass
class ProcessHealth:
    name: str
    pattern: str
    role: str
    is_running: bool
    heartbeat_fresh: bool
    age_seconds: float
    max_stale_seconds: float


@dataclass
class OvernightMonitorReport:
    ts: float
    fleet_healthy: bool
    total_processes: int
    running_processes: int
    stale_processes: int
    process_details: List[Dict[str, Any]]
    sensor_status: Dict[str, Any]
    autonomous_ai_mode: str
    dev_backtest_status: Dict[str, Any]
    system_resources: Dict[str, Any]


class AutonomousOvernightMonitor:
    """Orchestrates overnight monitoring of production trading fleet and dev backtests."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        cachedb_dir: str = "cachedb",
        logs_dir: str = "logs",
        dev_host: str = "192.168.0.138",
        dev_port: int = 32238,
        dev_user: str = "predut",
    ) -> None:
        self.workspace_dir = workspace_dir or ROOT_DIR
        self.cachedb_dir = os.path.join(self.workspace_dir, cachedb_dir)
        self.logs_dir = os.path.join(self.workspace_dir, logs_dir)
        self.procs_conf = os.path.join(self.workspace_dir, "procs.conf")
        self.status_file = os.path.join(self.cachedb_dir, "overnight_monitor_status.json")

        self.dev_host = dev_host
        self.dev_port = dev_port
        self.dev_user = dev_user

    def parse_process_manifest(self) -> List[Dict[str, Any]]:
        """Parses the single source of truth process definitions from procs.conf."""
        procs: List[Dict[str, Any]] = []
        if not os.path.exists(self.procs_conf):
            return procs

        try:
            with open(self.procs_conf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("|")
                    if len(parts) >= 7:
                        pat, dr, cmd, label, hblog, hbstale, role = [p.strip() for p in parts[:7]]
                        if role in ("bot", "fleet") and cmd:
                            procs.append({
                                "name": label or pat,
                                "pattern": pat,
                                "dir": dr.replace("$ROOT", self.workspace_dir),
                                "cmd": cmd.replace("$ROOT", self.workspace_dir),
                                "hb_log": hblog.replace("$ROOT", self.workspace_dir),
                                "hb_stale_s": float(hbstale) if hbstale.isdigit() else 0.0,
                                "role": role,
                            })
        except Exception as e:
            logger.warning("Error reading procs.conf: %s", e)

        return procs

    def _is_process_running(self, pattern: str) -> bool:
        """Checks if a process pattern is active using pgrep."""
        try:
            cmd = ["pgrep", "-f", pattern.rstrip("$")]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0
        except Exception:
            return False

    def check_processes(self, now: Optional[float] = None) -> List[ProcessHealth]:
        """Audits process running state and heartbeat timestamps."""
        now_ts = now if now is not None else time.time()
        results: List[ProcessHealth] = []

        for p in self.parse_process_manifest():
            running = self._is_process_running(p["pattern"])
            hb_log = p["hb_log"]
            stale_threshold = p["hb_stale_s"]
            age_sec = 0.0
            fresh = True

            if hb_log:
                hb_paths = [
                    (os.path.join(self.workspace_dir, f.strip()) if not os.path.isabs(f.strip()) else f.strip())
                    for f in hb_log.split(",") if f.strip()
                ]
                mtimes = []
                for hbp in hb_paths:
                    if os.path.exists(hbp):
                        try:
                            mtimes.append(os.path.getmtime(hbp))
                        except Exception:
                            pass
                if mtimes:
                    youngest = max(mtimes)
                    age_sec = max(0.0, now_ts - youngest)
                    if stale_threshold > 0 and age_sec > stale_threshold:
                        fresh = False
                elif stale_threshold > 0:
                    fresh = False

            results.append(ProcessHealth(
                name=p["name"],
                pattern=p["pattern"],
                role=p["role"],
                is_running=running,
                heartbeat_fresh=fresh,
                age_seconds=round(age_sec, 1),
                max_stale_seconds=stale_threshold,
            ))

        return results

    def gather_sensor_status(self) -> Dict[str, Any]:
        """Loads and summarizes real-time risk guards and telemetry files."""
        sensors: Dict[str, Any] = {}

        def _read(fname: str) -> Optional[Dict[str, Any]]:
            p = os.path.join(self.cachedb_dir, fname)
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        return json.load(f)
                except Exception:
                    pass
            return None

        # 1. Geopolitical Threat Sensor
        geo = _read("geopolitical_threat_eval.json")
        if geo:
            sensors["geopolitical_threat"] = {
                "threat_level": geo.get("threat_level", "NORMAL"),
                "risk_score": geo.get("risk_score", 0.0),
                "summary": geo.get("summary", ""),
            }

        # 2. Macro Sentiment Advisor
        macro = _read("sentiment_advisor_eval.json") or _read("macro_advisor_eval.json")
        if macro:
            sensors["sentiment_advisor"] = {
                "market_bias": macro.get("market_bias", "NEUTRAL"),
                "risk_level": macro.get("risk_level", "MODERATE"),
                "action": macro.get("recommended_action", "HOLD"),
            }

        # 3. Autonomous AI Supervisor State
        sup = _read("supervisor_nightly_eval.json")
        if sup:
            dec = sup.get("market_trading_decision", {})
            sensors["supervisor_eval"] = {
                "mode": sup.get("mode", "enforce"),
                "decision": dec.get("decision", "HOLD"),
                "confidence": dec.get("confidence", 0.0),
                "summary": sup.get("summary", ""),
            }

        # 4. Fear & Greed & Market Breadth
        fg = _read("fear_greed_cache.json")
        if fg:
            sensors["fear_greed"] = fg.get("value")

        mb = _read("market_breadth_cache.json")
        if mb:
            sensors["market_breadth"] = mb.get("advance_ratio")

        return sensors

    def get_autonomous_modes(self) -> Dict[str, str]:
        """Extracts operational mode settings from order_guard.conf."""
        modes: Dict[str, str] = {}
        conf_file = os.path.join(self.workspace_dir, "order_guard.conf")
        if os.path.exists(conf_file):
            try:
                with open(conf_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            modes[k.strip().lower()] = v.strip().lower()
            except Exception:
                pass
        return modes

    def check_dev_backtest_progress(self) -> Dict[str, Any]:
        """Polls the DEV server to check ongoing master suite execution status."""
        status = {
            "reachable": False,
            "running": False,
            "current_step": "N/A",
            "last_log_line": "N/A",
        }
        try:
            ssh_cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                "-p", str(self.dev_port),
                f"{self.dev_user}@{self.dev_host}",
                "pgrep -f run_overnight_master_suite.sh >/dev/null && echo 'RUNNING' || echo 'STOPPED'; "
                "tail -n 2 ~/mptrade/logs/overnight_runs/master_suite_current.log 2>/dev/null",
            ]
            res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=8)
            if res.returncode == 0:
                status["reachable"] = True
                lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
                if lines:
                    status["running"] = lines[0] == "RUNNING"
                    if len(lines) > 1:
                        status["last_log_line"] = lines[-1]
                        # Extract step if present
                        for l in reversed(lines):
                            if "[STEP " in l:
                                status["current_step"] = l
                                break
        except Exception as e:
            status["error"] = str(e)

        return status

    def gather_system_resources(self) -> Dict[str, Any]:
        """Reads local system resource utilization."""
        stats: Dict[str, Any] = {}
        try:
            # 1-min, 5-min, 15-min load average
            load1, load5, load15 = os.getloadavg()
            stats["load_avg"] = [round(load1, 2), round(load5, 2), round(load15, 2)]

            # Disk usage
            disk = shutil.disk_usage(self.workspace_dir)
            stats["disk_used_pct"] = round((disk.used / disk.total) * 100, 1)
            stats["disk_free_gb"] = round(disk.free / (1024 ** 3), 1)
        except Exception:
            pass
        return stats

    def run_cycle(self, persist: bool = True) -> OvernightMonitorReport:
        """Executes one comprehensive audit cycle and saves report."""
        now = time.time()
        proc_health = self.check_processes(now=now)
        running_cnt = sum(1 for p in proc_health if p.is_running)
        stale_cnt = sum(1 for p in proc_health if p.is_running and not p.heartbeat_fresh)
        healthy = (running_cnt == len(proc_health)) and (stale_cnt == 0)

        sensors = self.gather_sensor_status()
        modes = self.get_autonomous_modes()
        dev_status = self.check_dev_backtest_progress()
        resources = self.gather_system_resources()

        report = OvernightMonitorReport(
            ts=now,
            fleet_healthy=healthy,
            total_processes=len(proc_health),
            running_processes=running_cnt,
            stale_processes=stale_cnt,
            process_details=[asdict(p) for p in proc_health],
            sensor_status=sensors,
            autonomous_ai_mode=modes.get("supervisor_mode", "enforce"),
            dev_backtest_status=dev_status,
            system_resources=resources,
        )

        if persist:
            try:
                os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
                with open(self.status_file, "w", encoding="utf-8") as f:
                    json.dump(asdict(report), f, indent=2)
            except Exception as e:
                logger.warning("Could not persist overnight status: %s", e)

        return report

    def loop(self, interval_seconds: float = 60.0) -> None:
        """Continuous execution loop."""
        logger.info("Starting Autonomous Overnight Monitor (interval=%.0fs)...", interval_seconds)
        while True:
            try:
                report = self.run_cycle(persist=True)
                dev_run = "RUNNING" if report.dev_backtest_status.get("running") else "IDLE"
                logger.info(
                    "Monitor Tick: Fleet %d/%d running (%d stale) | AI Mode: %s | DEV Backtest: %s | Threat: %s",
                    report.running_processes,
                    report.total_processes,
                    report.stale_processes,
                    report.autonomous_ai_mode,
                    dev_run,
                    report.sensor_status.get("geopolitical_threat", {}).get("threat_level", "NORMAL"),
                )
            except Exception as e:
                logger.error("Error in monitor loop: %s", e, exc_info=True)
            time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Overnight Fleet & Backtest Monitor")
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    parser.add_argument("--interval", type=float, default=60.0, help="Loop interval in seconds")
    args = parser.parse_args()

    monitor = AutonomousOvernightMonitor()
    if args.once:
        rep = monitor.run_cycle(persist=True)
        print(json.dumps(asdict(rep), indent=2))
    else:
        monitor.loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
