"""Unit tests for AutonomousAIReconciler."""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch
import pytest

from orchestratorOS.supervisor.autonomous_ai_reconciler import (
    AutonomousAIReconciler,
    LogEventSummary,
    SupervisorAuditReport,
)
from intelligence.sentiment.gemini_client import GeminiClient


@pytest.fixture
def temp_env(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir), str(logs_dir)


def test_reconciler_initialization_defaults(temp_env):
    cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(
        cache_dir=cache_dir,
        logs_dirs=[logs_dir],
    )
    assert reconciler.mode == "shadow"
    assert reconciler.lookback_hours == 24.0
    assert reconciler.cache_dir == cache_dir


def test_harvest_log_anomalies(temp_env):
    cache_dir, logs_dir = temp_env
    # Create sample log file
    sample_log = os.path.join(logs_dir, "rtrade_test.log")
    with open(sample_log, "w", encoding="utf-8") as f:
        f.write("2026-10-07 12:00:00 [INFO] Bot started\n")
        f.write("2026-10-07 12:05:00 [ERROR] Connection timeout to Binance\n")
        f.write("2026-10-07 12:10:00 [TAOUSDC] BUY BLOCKED (fail-closed): guard vetoed\n")
        f.write("2026-10-07 12:15:00 [HIGH-STAKE GUARD] vetoed: funding is elevated\n")

    reconciler = AutonomousAIReconciler(cache_dir=cache_dir, logs_dirs=[logs_dir])
    summary = reconciler.harvest_log_anomalies(lookback_hours=1.0)

    assert summary.total_scanned_files == 1
    assert summary.error_count == 2
    assert summary.warning_count == 1
    assert len(summary.critical_events) >= 1
    assert any("BUY BLOCKED" in evt for evt in summary.critical_events)


def test_inspect_venue_states(temp_env):
    cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(cache_dir=cache_dir, logs_dirs=[logs_dir])
    res = reconciler.inspect_venue_states()
    assert "stale_state_files" in res
    assert "active_anomalies" in res


def test_synthesize_and_diagnose_mocked(temp_env):
    cache_dir, logs_dir = temp_env

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

    client = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(mock_resp))
    reconciler = AutonomousAIReconciler(
        cache_dir=cache_dir,
        logs_dirs=[logs_dir],
        gemini_client=client,
        mode="shadow",
    )

    summary = LogEventSummary(
        total_scanned_files=2,
        error_count=1,
        warning_count=0,
        critical_events=["Connection timeout"],
        guard_interventions=[],
    )

    diagnosis = reconciler.synthesize_and_diagnose(summary, {"active_anomalies": []})
    assert diagnosis["summary"] == mock_resp["summary"]
    assert len(diagnosis["diagnosed_issues"]) == 1
    assert len(diagnosis["venue_actions"]) == 1


def test_shadow_mode_does_not_execute_destructive_actions(temp_env):
    cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(
        cache_dir=cache_dir,
        logs_dirs=[logs_dir],
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
    # In shadow mode, executed must be False (simulated only)
    assert executed is False
    assert tests_passed is None


def test_dispatch_notification_shadow_mode(temp_env):
    cache_dir, logs_dir = temp_env
    reconciler = AutonomousAIReconciler(
        cache_dir=cache_dir,
        logs_dirs=[logs_dir],
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

    with patch("notify_engine.alertnotifiers.notify") as mock_notify:
        ok = reconciler.dispatch_notification(report)
        assert ok is True
        mock_notify.assert_called_once()
        args, kwargs = mock_notify.call_args
        assert "[AI-SUPERVISOR · SHADOW]" in kwargs["title"]


def test_full_run_cycle_and_artifacts(temp_env):
    cache_dir, logs_dir = temp_env

    mock_resp = {
        "summary": "Daily audit clean.",
        "diagnosed_issues": [],
        "venue_actions": [],
        "code_remediation": {"has_fix": False},
    }

    client = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(mock_resp))
    reconciler = AutonomousAIReconciler(
        cache_dir=cache_dir,
        logs_dirs=[logs_dir],
        gemini_client=client,
        mode="shadow",
        notify_enabled=False,
    )

    report = reconciler.run_cycle(force=True)

    assert report.mode == "shadow"
    assert report.actions_executed is False

    # Check saved artifacts in cache_dir
    state_file = os.path.join(cache_dir, "supervisor_nightly_eval.json")
    heartbeat_file = os.path.join(cache_dir, "supervisor_nightly.heartbeat")

    assert os.path.exists(state_file)
    assert os.path.exists(heartbeat_file)

    with open(heartbeat_file, "r", encoding="utf-8") as f:
        hb = json.load(f)
    assert hb["status"] == "alive"
    assert hb["mode"] == "shadow"
