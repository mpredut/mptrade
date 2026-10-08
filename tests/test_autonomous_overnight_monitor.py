"""Unit tests for Autonomous Overnight Monitor (orchestratorOS/livecheck/autonomous_overnight_monitor.py).

Verifies:
1. Process inventory parsing from procs.conf.
2. Heartbeat freshness check.
3. Telemetry and guard state aggregation.
4. Autonomous AI status reporting.
5. DEV backtest master suite tracking.
6. JSON summary persistence.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestratorOS.livecheck.autonomous_overnight_monitor import (
    AutonomousOvernightMonitor,
    OvernightMonitorReport,
)


@pytest.fixture
def mock_workspace(tmp_path):
    # Setup mock procs.conf
    procs_file = tmp_path / "procs.conf"
    procs_file.write_text(
        "bot1.py|$ROOT|python bot1.py|bot1|logs/bot1.log|600|bot\n"
        "fleet1.py|$ROOT|python fleet1.py|fleet1|cachedb/fleet1.heartbeat|180|fleet\n"
    )

    # Setup mock cachedb and logs
    cachedb = tmp_path / "cachedb"
    cachedb.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()

    # Create dummy heartbeats
    (logs / "bot1.log").write_text("heartbeat")
    hb_file = cachedb / "fleet1.heartbeat"
    hb_file.write_text("heartbeat")

    # Create dummy sensor files
    (cachedb / "geopolitical_threat_eval.json").write_text(json.dumps({
        "threat_level": "ELEVATED",
        "risk_score": 0.65,
        "summary": "Active conflict attrition",
        "ts": time.time(),
    }))

    (cachedb / "supervisor_nightly_eval.json").write_text(json.dumps({
        "mode": "enforce",
        "summary": "Core execution nominal",
        "market_trading_decision": {"decision": "HOLD", "confidence": 0.52},
        "ts": time.time(),
    }))

    # Create dummy config
    (tmp_path / "order_guard.conf").write_text(
        "llm_guard_mode = enforce\nsupervisor_mode = enforce\nautonomous_trading_mode = enforce\n"
    )

    return tmp_path


def test_monitor_procs_inventory_parsing(mock_workspace):
    monitor = AutonomousOvernightMonitor(workspace_dir=str(mock_workspace))
    procs = monitor.parse_process_manifest()
    assert len(procs) == 2
    assert procs[0]["name"] == "bot1"
    assert procs[1]["name"] == "fleet1"


def test_monitor_gather_sensor_status(mock_workspace):
    monitor = AutonomousOvernightMonitor(workspace_dir=str(mock_workspace))
    sensors = monitor.gather_sensor_status()
    assert sensors["geopolitical_threat"]["threat_level"] == "ELEVATED"
    assert sensors["supervisor_eval"]["mode"] == "enforce"
    assert sensors["supervisor_eval"]["decision"] == "HOLD"


def test_monitor_run_cycle_report(mock_workspace, monkeypatch):
    monitor = AutonomousOvernightMonitor(workspace_dir=str(mock_workspace))
    
    # Mock DEV runner check
    monkeypatch.setattr(monitor, "check_dev_backtest_progress", lambda: {
        "reachable": True,
        "running": True,
        "current_step": "STEP 0A",
        "last_log_line": "Starting simulation in mode: FULL_COMPOSITE ...",
    })

    # Mock process running check
    monkeypatch.setattr(monitor, "_is_process_running", lambda pat: True)

    report = monitor.run_cycle(persist=True)
    assert isinstance(report, OvernightMonitorReport)
    assert report.fleet_healthy is True
    assert report.dev_backtest_status["running"] is True
    assert report.autonomous_ai_mode == "enforce"

    status_file = mock_workspace / "cachedb" / "overnight_monitor_status.json"
    assert status_file.exists()
    data = json.loads(status_file.read_text())
    assert data["autonomous_ai_mode"] == "enforce"
    assert data["fleet_healthy"] is True
