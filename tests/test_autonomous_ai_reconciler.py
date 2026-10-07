"""Unit tests for decoupled AutonomousAIReconciler."""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch
import urllib.request
import pytest

from orchestratorOS.supervisor.autonomous_ai_reconciler import (
    AutonomousAIReconciler,
    LogEventSummary,
    SupervisorAuditReport,
    StandaloneGeminiAuditor,
    TelemetrySensorSnapshot,
)


@pytest.fixture
def temp_env(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    cache_dir = workspace / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return str(workspace), str(cache_dir), str(logs_dir)


def test_reconciler_initialization_defaults(temp_env):
    workspace, cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
    )
    assert reconciler.mode == "shadow"
    assert reconciler.lookback_hours == 24.0
    assert reconciler.cache_dir == cache_dir


def test_harvest_sensors_telemetry(temp_env):
    workspace, cache_dir, logs_dir = temp_env
    # Create sample sensor files in cachedb
    fg_file = os.path.join(cache_dir, "fear_greed_cache.json")
    with open(fg_file, "w", encoding="utf-8") as f:
        json.dump({"value": 72, "classification": "Greed"}, f)

    mb_file = os.path.join(cache_dir, "market_breadth_cache.json")
    with open(mb_file, "w", encoding="utf-8") as f:
        json.dump({"advance_ratio": 0.65, "regime": "bull"}, f)

    hb_file = os.path.join(cache_dir, "macro_analyzer.heartbeat")
    with open(hb_file, "w", encoding="utf-8") as f:
        json.dump({"status": "alive"}, f)

    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    snapshot = reconciler.harvest_sensors_telemetry()

    assert snapshot.fear_greed is not None
    assert snapshot.fear_greed["value"] == 72
    assert snapshot.market_breadth is not None
    assert snapshot.market_breadth["advance_ratio"] == 0.65
    assert "macro_analyzer.heartbeat" in snapshot.active_heartbeats


def test_harvest_log_anomalies(temp_env):
    workspace, cache_dir, logs_dir = temp_env
    sample_log = os.path.join(logs_dir, "rtrade_test.log")
    with open(sample_log, "w", encoding="utf-8") as f:
        f.write("2026-10-07 12:00:00 [INFO] Bot started\n")
        f.write("2026-10-07 12:05:00 [ERROR] Connection timeout to Binance\n")
        f.write("2026-10-07 12:10:00 [TAOUSDC] BUY BLOCKED (fail-closed): guard vetoed\n")
        f.write("2026-10-07 12:15:00 [HIGH-STAKE GUARD] vetoed: funding is elevated\n")

    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    summary = reconciler.harvest_log_anomalies(lookback_hours=1.0)

    assert summary.total_scanned_files == 1
    assert summary.error_count == 2
    assert summary.warning_count == 1
    assert len(summary.critical_events) >= 1
    assert any("BUY BLOCKED" in evt for evt in summary.critical_events)


def test_synthesize_and_diagnose_mocked(temp_env):
    workspace, cache_dir, logs_dir = temp_env

    mock_resp = {
        "summary": "1 transient network timeout diagnosed; all open orders aligned.",
        "diagnosed_issues": [
            {
                "severity": "LOW",
                "category": "BOT_EXCEPTION",
                "description": "Transient REST socket timeout",
                "root_cause": "Binance gateway latency spike",
            }
        ],
        "venue_actions": [
            {
                "action": "CANCEL_ORDER",
                "venue": "binance",
                "symbol": "BTCUSDC",
                "order_id": "99999",
                "reason": "Stale order",
            }
        ],
        "code_remediation": {
            "has_fix": False,
        },
    }

    auditor = StandaloneGeminiAuditor(custom_runner=lambda prompt, model, timeout: json.dumps(mock_resp))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
    )

    sensors = TelemetrySensorSnapshot()
    logs = LogEventSummary(
        total_scanned_files=2,
        error_count=1,
        warning_count=0,
        critical_events=["Connection timeout"],
        guard_interventions=[],
    )

    diagnosis = reconciler.synthesize_and_diagnose(sensors, logs)
    assert diagnosis["summary"] == mock_resp["summary"]
    assert len(diagnosis["diagnosed_issues"]) == 1
    assert len(diagnosis["venue_actions"]) == 1


def test_shadow_mode_does_not_execute_destructive_actions(temp_env):
    workspace, cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        mode="shadow",
    )

    plan = {
        "summary": "Shadow audit",
        "venue_actions": [
            {"action": "CANCEL_ORDER", "venue": "binance", "symbol": "BTCUSDC", "order_id": "123", "reason": "stale"}
        ],
        "code_remediation": {
            "has_fix": True,
            "target_file": "some_file.py",
            "search_block": "a = 1",
            "replace_block": "a = 2",
        },
    }

    executed, tests_passed = reconciler.apply_reconciliation(plan)
    assert executed is False
    assert tests_passed is None


def test_dispatch_notification_via_urllib(temp_env):
    workspace, cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        mode="shadow",
        notify_enabled=True,
    )

    report = SupervisorAuditReport(
        ts=time.time(),
        mode="shadow",
        summary="Audit completed: 0 issues.",
        diagnosed_issues=[{"severity": "INFO", "description": "All healthy"}],
        venue_actions=[],
        code_remediation={"has_fix": False},
        actions_executed=False,
    )

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        ok = reconciler.dispatch_notification(report)
        assert ok is True
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.headers["Title"].decode("utf-8") if isinstance(req.headers["Title"], bytes) else req.headers["Title"]


def test_full_run_cycle_and_artifacts(temp_env):
    workspace, cache_dir, logs_dir = temp_env

    mock_resp = {
        "summary": "Daily audit clean.",
        "diagnosed_issues": [],
        "venue_actions": [],
        "code_remediation": {"has_fix": False},
    }

    auditor = StandaloneGeminiAuditor(custom_runner=lambda prompt, model, timeout: json.dumps(mock_resp))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
    )

    report = reconciler.run_cycle(force=True)

    assert report.mode == "shadow"
    assert report.actions_executed is False
    assert report.sensor_health is not None

    state_file = os.path.join(cache_dir, "supervisor_nightly_eval.json")
    heartbeat_file = os.path.join(cache_dir, "supervisor_nightly.heartbeat")

    assert os.path.exists(state_file)
    assert os.path.exists(heartbeat_file)

    with open(heartbeat_file, "r", encoding="utf-8") as f:
        hb = json.load(f)
    assert hb["status"] == "alive"
    assert hb["mode"] == "shadow"
