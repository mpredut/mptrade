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
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from botcore import _parse_bool_flag
from orchestratorOS.lib.technical_repair_ssot import (
    clean_json_value,
    gather_basic_os_resources,
    parse_finite_float,
    parse_procs_manifest,
    read_json_recovering,
    resolve_heartbeat_paths,
)

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

        self.dev_host = str(dev_host or "192.168.0.138").strip() or "192.168.0.138"
        if isinstance(dev_port, bool) or dev_port is None:
            self.dev_port = 32238
        else:
            try:
                p_flt = float(dev_port)
                if math.isfinite(p_flt) and p_flt.is_integer():
                    p_int = int(p_flt)
                    self.dev_port = p_int if 1 <= p_int <= 65535 else 32238
                else:
                    self.dev_port = 32238
            except (TypeError, ValueError, OverflowError):
                self.dev_port = 32238
        self.dev_user = str(dev_user or "predut").strip() or "predut"

    def parse_process_manifest(self) -> List[Dict[str, Any]]:
        """Parses the single source of truth process definitions from procs.conf."""
        return parse_procs_manifest(self.workspace_dir)

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
        now_val = parse_finite_float(now, None)
        now_ts = now_val if (now_val is not None and now_val > 0) else time.time()
        results: List[ProcessHealth] = []

        for p in self.parse_process_manifest():
            running = self._is_process_running(p["pattern"])
            hb_log = p.get("hb_log", "")
            stale_threshold = p.get("hb_stale_s", 0.0)
            age_sec = 0.0
            fresh = True

            hb_paths = p.get("hb_paths")
            if not isinstance(hb_paths, list):
                hb_paths, _ = resolve_heartbeat_paths(
                    self.workspace_dir,
                    p.get("dir", self.workspace_dir),
                    p.get("name", ""),
                    p.get("pattern", ""),
                    hb_log,
                    stale_threshold,
                )

            if hb_paths:
                mtimes = []
                missing_any = False
                for hbp in hb_paths:
                    if os.path.exists(hbp):
                        try:
                            mtimes.append(os.path.getmtime(hbp))
                        except Exception:
                            missing_any = True
                    else:
                        missing_any = True
                if mtimes:
                    oldest = min(mtimes)
                    age_sec = max(0.0, now_ts - oldest)
                    if stale_threshold > 0 and (missing_any or age_sec > stale_threshold):
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
            data = read_json_recovering(os.path.join(self.cachedb_dir, fname))
            return data if isinstance(data, dict) else None

        # 1. Geopolitical Threat Sensor
        geo = _read("geopolitical_threat_eval.json")
        if geo:
            sensors["geopolitical_threat"] = {
                "threat_level": str(geo.get("threat_level") or "NORMAL"),
                "risk_score": parse_finite_float(geo.get("risk_score"), 0.0),
                "summary": str(geo.get("summary") or ""),
            }

        # 2. Macro Sentiment Advisor
        macro = _read("sentiment_advisor_eval.json") or _read("macro_advisor_eval.json")
        if macro:
            sensors["sentiment_advisor"] = {
                "market_bias": str(macro.get("market_bias") or "NEUTRAL"),
                "risk_level": str(macro.get("risk_level") or "MODERATE"),
                "action": str(macro.get("recommended_action") or "HOLD"),
            }

        # 3. Autonomous AI Supervisor State
        sup = _read("auto_cryptotrade_executor_eval.json") or _read("auto_trade_executor_eval.json")
        if sup:
            raw_dec = sup.get("market_trading_decision")
            dec = raw_dec if isinstance(raw_dec, dict) else {}
            sensors["supervisor_eval"] = {
                "mode": str(sup.get("mode") or "enforce"),
                "decision": str(dec.get("decision") or "HOLD"),
                "confidence": parse_finite_float(dec.get("confidence"), 0.0),
                "summary": str(sup.get("summary") or ""),
            }

        # 4. Fear & Greed & Market Breadth
        fg = _read("fear_greed_collect.json") or _read("fear_greed_cache.json")
        if fg:
            sensors["fear_greed"] = parse_finite_float(fg.get("value"), None)

        mb = _read("market_breadth_collect.json") or _read("market_breadth_cache.json")
        if mb:
            sensors["market_breadth"] = parse_finite_float(mb.get("advance_ratio"), None)

        return sensors

    def get_autonomous_modes(self) -> Dict[str, str]:
        """Extracts operational mode settings strictly from order_guard.conf."""
        modes: Dict[str, str] = {}
        search_dirs = [ROOT_DIR]
        if os.path.abspath(self.workspace_dir) != os.path.abspath(ROOT_DIR):
            search_dirs.append(self.workspace_dir)
        for base_dir in search_dirs:
            conf_file = os.path.join(base_dir, "order_guard.conf")
            if os.path.exists(conf_file):
                with open(conf_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.split("#", 1)[0].strip()
                        if not line or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        modes[k.strip().lower()] = v.strip().strip("\"'").lower()
        if "supervisor_mode" not in modes:
            raise KeyError("Missing required 'supervisor_mode' in order_guard.conf")
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
                "tail -n 5 ~/mptrade/logs/overnight_runs/latest_overnight.log ~/mptrade/logs/overnight_runs/master_suite_current.log 2>/dev/null",
            ]
            res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=8)
            if res.returncode == 0:
                status["reachable"] = True
                lines = [
                    l.strip()
                    for l in res.stdout.strip().split("\n")
                    if l.strip() and not (l.strip().startswith("==>") and l.strip().endswith("<=="))
                ]
                if lines:
                    status["running"] = lines[0] == "RUNNING"
                    if len(lines) > 1:
                        status["last_log_line"] = lines[-1]
                        # Extract step if present
                        for l in reversed(lines):
                            if "[STEP " in l or "[STAGE " in l:
                                status["current_step"] = l
                                break
        except Exception as e:
            status["error"] = str(e)

        return status

    def gather_system_resources(self) -> Dict[str, Any]:
        """Reads local system resource utilization."""
        return gather_basic_os_resources(self.workspace_dir)

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
            autonomous_ai_mode=modes["supervisor_mode"],
            dev_backtest_status=dev_status,
            system_resources=resources,
        )

        do_persist = _parse_bool_flag(persist, False)

        if do_persist:
            try:
                os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
                from state_io import save_json_state

                save_json_state(self.status_file, clean_json_value(asdict(report)), indent=2)
            except Exception as e:
                logger.warning("Could not persist overnight status: %s", e)

        return report

    def loop(self, interval_seconds: float = 60.0) -> None:
        """Continuous execution loop."""
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or float(interval_seconds) <= 0
        ):
            raise ValueError("interval_seconds must be a positive finite number")
        sleep_s = float(interval_seconds)
        logger.info("Starting Autonomous Overnight Monitor (interval=%.0fs)...", sleep_s)
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
            time.sleep(sleep_s)


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
