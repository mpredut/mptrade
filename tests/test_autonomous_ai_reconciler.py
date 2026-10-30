"""Unit tests for AutonomousAIReconciler and FleetHealthMonitor."""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from orchestratorOS.livecheck.fleet_health_monitor import (
    FleetHealthMonitor,
    OvernightMonitorReport,
)
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.supervisor.autonomous_ai_reconciler import (
    AutonomousAIReconciler,
    LogEventSummary,
    SupervisorAuditReport,
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
    workspace, cache_dir, _ = temp_env
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
    )
    assert reconciler.mode == "enforce"
    assert reconciler.lookback_hours == 24.0
    assert reconciler.cache_dir == cache_dir


def test_reconciler_invalid_supervisor_mode_fails_fast(temp_env):
    workspace, _, _ = temp_env
    with pytest.raises(ValueError, match="Invalid supervisor_mode"):
        AutonomousAIReconciler(
            workspace_dir=workspace,
            cache_dir="cachedb",
            logs_dirs=["logs"],
            mode="invalid_mode",
        )


def test_harvest_sensors_telemetry(temp_env):
    workspace, cache_dir, _ = temp_env
    fg_file = os.path.join(cache_dir, "fear_greed_collect.json")
    with open(fg_file, "w", encoding="utf-8") as f:
        json.dump({"value": 72, "classification": "Greed"}, f)

    mb_file = os.path.join(cache_dir, "market_breadth_collect.json")
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
    workspace, _, logs_dir = temp_env
    sample_log = os.path.join(logs_dir, "rtrade_test.log")
    now = time.time()
    stamp = lambda ago: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - ago))  # noqa: E731
    with open(sample_log, "w", encoding="utf-8") as f:
        f.write(f"{stamp(7200)} [ERROR] Stale DNS failure from two hours ago\n")
        f.write(f"{stamp(1500)} [INFO] Bot started\n")
        f.write(f"{stamp(1200)} [ERROR] Connection timeout to Binance\n")
        f.write(f"{stamp(900)} [TAOUSDC] BUY BLOCKED (fail-closed): guard vetoed\n")
        f.write(f"{stamp(600)} [HIGH-STAKE GUARD] vetoed: funding is elevated\n")
        f.write(f"{stamp(300)} [INFO] Autonomous AI Supervisor cycle complete: fail-closed guardrails active, 0 errors\n")
        f.write(f"{stamp(300)} [ENFORCE] auto_technical_repair: fail-closed guardrails active, 0 recvWindow errors\n")
        f.write(f"{stamp(200)} [WARNING] mailer: Email delivery skipped for '[cacheManager] Critical Errors' (SMTP auth rejected)\n")
        f.write(f"{stamp(190)} [INFO] Orchestrator: Dispatched ntfy alert to [ntfy-error-1978]: [cacheManager] Critical Errors\n")

    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    summary = reconciler.harvest_log_anomalies(lookback_hours=1.0)

    assert summary.total_scanned_files == 1
    assert summary.error_count == 2
    assert summary.warning_count == 1
    assert len(summary.critical_events) >= 1
    assert any("BUY BLOCKED" in evt for evt in summary.critical_events)
    assert not any("Stale DNS" in evt for evt in summary.critical_events)
    assert not any("auto_technical_repair" in evt or "Supervisor cycle complete" in evt for evt in summary.critical_events)


def test_recent_log_lines_handles_clock_only_stamps_and_midnight_wrap():
    from datetime import datetime
    from orchestratorOS.supervisor.autonomous_ai_reconciler import recent_log_lines

    mtime = datetime(2026, 10, 23, 0, 10, 0).timestamp()
    lines = [
        "14:54:17 Error: APIError(code=-1021): recvWindow\n",   # yesterday 14:54 -> outside 1h window
        "23:50:00 [ERROR] just before midnight\n",
        "Traceback (most recent call last):\n",                 # inherits the next later stamp
        "00:05:00 ERROR after midnight\n",
    ]
    picked = recent_log_lines(lines, mtime, mtime - 3600)
    assert picked == lines[1:]

    # DST fallback hour in Europe/Bucharest (2026-10-25 03:00-04:00 fold=0 -> fold=1):
    old_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Europe/Bucharest"
        if hasattr(time, "tzset"):
            time.tzset()
        dt_fold1 = datetime(2026, 10, 25, 3, 15, 0, fold=1)
        mtime_dst = dt_fold1.timestamp()
        dst_lines = [
            "2026-10-25 01:00:00 [ERROR] outside 1h window\n",
            "2026-10-25 03:45:00 [ERROR] fold=0 (30m before 03:15 fold=1)\n",
            "03:50:00 [ERROR] clock-only fold=0 (25m before 03:15 fold=1)\n",
            "2026-10-25 03:10:00 [ERROR] fold=1 (5m before 03:15 fold=1)\n",
        ]
        picked_dst = recent_log_lines(dst_lines, mtime_dst, mtime_dst - 3600)
        assert picked_dst == dst_lines[1:]
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        if hasattr(time, "tzset"):
            time.tzset()


def test_only_trade_executor_topic_reaches_gateway_bounded_by_supervisor_mode(temp_env):
    workspace, _, _ = temp_env
    plan = {"summary": "s", "venue_actions": [],
            "code_remediation": {"has_fix": True, "target_file": "x.py", "search_block": "a", "replace_block": "b"},
            "market_trading_decision": {"decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "confidence": 0.95}}
    auditor = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(plan))
    gateway = MagicMock()
    gateway.min_confidence, gateway.max_notional_usd = 0.85, 2000.0
    gateway.submit.return_value = {"status": "OBSERVED_SHADOW"}
    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"],
                                        gemini_auditor=auditor, mode="shadow", notify_enabled=False,
                                        intent_gateway=gateway)
    with patch.object(reconciler, "_apply_code_patch_safely") as patcher:
        repair = reconciler.run_cycle(topic_key="auto_technical_repair")
        gateway.submit.assert_not_called()
        assert repair.market_trading_decision == {} and repair.intent_emitted is None

        trade = reconciler.run_cycle(topic_key="auto_trade_executor", focus_symbols=["TAOUSDC"])
        patcher.assert_not_called()   # shadow never patches; the trade topic has no self-heal capability anyway
    gateway.submit.assert_called_once()
    assert gateway.submit.call_args.kwargs == {"source": "auto_trade_executor", "max_mode": "shadow"}
    assert trade.intent_emitted == {"status": "OBSERVED_SHADOW"} and trade.code_remediation == {}


def test_unknown_topic_fails_fast(temp_env):
    workspace, _, _ = temp_env
    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    with pytest.raises(KeyError):
        reconciler.run_cycle(topic_key="supervisor_nightly")


def test_code_remediation_can_be_disabled_in_enforce(temp_env):
    workspace, _, _ = temp_env
    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"], mode="enforce")
    plan = {"venue_actions": [], "code_remediation": {
        "has_fix": True, "target_file": "x.py", "search_block": "a", "replace_block": "b"}}
    with patch.object(reconciler, "_apply_code_patch_safely") as patcher:
        executed, tests_passed = reconciler.apply_reconciliation(plan, allow_code_remediation=False)
    patcher.assert_not_called()
    assert executed is False and tests_passed is None


def test_synthesize_and_diagnose_mocked(temp_env):
    workspace, _, _ = temp_env

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

    auditor = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(mock_resp))
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

    diagnosis = reconciler.synthesize_and_diagnose(sensors, logs, topic_key="auto_technical_repair")
    assert diagnosis["summary"] == mock_resp["summary"]
    assert len(diagnosis["diagnosed_issues"]) == 1
    assert len(diagnosis["venue_actions"]) == 1


def test_shadow_mode_does_not_execute_destructive_actions(temp_env):
    workspace, _, _ = temp_env
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
    workspace, _, _ = temp_env
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
    mock_resp.code = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        ok = reconciler.dispatch_notification(report)
        assert ok is True
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.headers["Title"].decode("utf-8") if isinstance(req.headers["Title"], bytes) else req.headers["Title"]


def test_full_run_cycle_and_artifacts(temp_env):
    workspace, cache_dir, _ = temp_env

    mock_resp = {
        "summary": "Daily audit clean.",
        "diagnosed_issues": [],
        "venue_actions": [],
        "code_remediation": {"has_fix": False},
    }

    auditor = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(mock_resp))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
    )

    report = reconciler.run_cycle(topic_key="auto_technical_repair")

    assert report.mode == "shadow"
    assert report.actions_executed is False
    assert report.sensor_health is not None

    state_file = os.path.join(cache_dir, "auto_technical_repair_eval.json")
    heartbeat_file = os.path.join(cache_dir, "auto_technical_repair.heartbeat")

    assert os.path.exists(state_file)
    assert os.path.exists(heartbeat_file)

    with open(heartbeat_file, "r", encoding="utf-8") as f:
        hb = json.load(f)
    assert hb["status"] == "alive"
    assert hb["mode"] == "shadow"

    # The trade topic writes its own artifacts and leaves the repair report untouched.
    nightly_mtime = os.path.getmtime(state_file)
    reconciler._intent_gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0)
    reconciler.run_cycle(topic_key="auto_trade_executor")
    assert os.path.exists(os.path.join(cache_dir, "auto_trade_executor_eval.json"))
    assert os.path.exists(os.path.join(cache_dir, "auto_trade_executor.heartbeat"))
    assert os.path.getmtime(state_file) == nightly_mtime


def test_harvest_prices_and_trends(temp_env):
    workspace, cache_dir, _ = temp_env

    cp_file = os.path.join(cache_dir, "cache_current_price.json")
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump({"items": {"BTCUSDC": [[123456, 85000.0]]}}, f)

    it_file = os.path.join(cache_dir, "cache_short_trend.json")
    with open(it_file, "w", encoding="utf-8") as f:
        json.dump({
            "BTCUSDC": {
                "current_price": 85000.0,
                "final_trend": 1,
                "growth_coefficient": 0.05,
                "slope_full": 0.02,
            }
        }, f)

    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    snapshot = reconciler.harvest_sensors_telemetry()

    assert "BTCUSDC" in snapshot.prices_and_trends
    assert snapshot.prices_and_trends["BTCUSDC"]["price"] == 85000.0
    assert snapshot.prices_and_trends["BTCUSDC"]["trend"] == 1
    assert snapshot.prices_and_trends["BTCUSDC"]["growth_coefficient"] == 0.05


@pytest.fixture
def mock_workspace(tmp_path):
    procs_file = tmp_path / "procs.conf"
    procs_file.write_text(
        "bot1.py|$ROOT|python bot1.py|bot1|logs/bot1.log|600|bot\n"
        "fleet1.py|$ROOT|python fleet1.py|fleet1|cachedb/fleet1.heartbeat|180|fleet\n"
    )

    cachedb = tmp_path / "cachedb"
    cachedb.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / "bot1.log").write_text("heartbeat")
    (cachedb / "fleet1.heartbeat").write_text("heartbeat")

    (cachedb / "geopolitical_eval.json").write_text(json.dumps({
        "threat_level": "ELEVATED",
        "risk_score": 0.65,
        "summary": "Active conflict attrition",
        "ts": time.time(),
    }))

    (cachedb / "auto_cryptotrade_executor_eval.json").write_text(json.dumps({
        "mode": "enforce",
        "summary": "Core execution nominal",
        "market_trading_decision": {"decision": "HOLD", "confidence": 0.52},
        "ts": time.time(),
    }))

    (tmp_path / "order_guard.conf").write_text(
        "llm_guard_mode = enforce\nsupervisor_mode = enforce\nautonomous_trading_mode = enforce\n"
    )

    return tmp_path


def test_monitor_procs_inventory_parsing(mock_workspace):
    monitor = FleetHealthMonitor(workspace_dir=str(mock_workspace))
    procs = monitor.parse_process_manifest()
    assert len(procs) == 2
    assert procs[0]["name"] == "bot1"
    assert procs[1]["name"] == "fleet1"


def test_monitor_gather_sensor_status(mock_workspace):
    monitor = FleetHealthMonitor(workspace_dir=str(mock_workspace))
    sensors = monitor.gather_sensor_status()
    assert sensors["geopolitical_threat"]["threat_level"] == "ELEVATED"
    assert sensors["supervisor_eval"]["mode"] == "enforce"
    assert sensors["supervisor_eval"]["decision"] == "HOLD"


def test_monitor_run_cycle_report(mock_workspace, monkeypatch):
    monitor = FleetHealthMonitor(workspace_dir=str(mock_workspace))
    monkeypatch.setattr(monitor, "check_dev_backtest_progress", lambda: {
        "reachable": True,
        "running": True,
        "current_step": "STEP 0A",
        "last_log_line": "Starting simulation in mode: FULL_COMPOSITE ...",
    })
    monkeypatch.setattr(monitor, "_is_process_running", lambda pat: True)

    report = monitor.run_cycle(persist=True)
    assert isinstance(report, OvernightMonitorReport)
    assert report.fleet_healthy is True
    assert report.dev_backtest_status["running"] is True
    assert report.autonomous_ai_mode == "enforce"

    status_file = mock_workspace / "cachedb" / "fleet_health_status.json"
    assert status_file.exists()
    data = json.loads(status_file.read_text())
    assert data["autonomous_ai_mode"] == "enforce"
    assert data["fleet_healthy"] is True


def test_none_venue_actions_and_routine_hold_do_not_trigger_executed_or_notify_spam(temp_env):
    workspace, cache_dir, _ = temp_env
    with open(os.path.join(cache_dir, "cache_24_price_TAOUSDC.json"), "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": [[1000000, 250.0], [1900000, 255.0]]}}, f)
    with open(os.path.join(cache_dir, "cache_short_trend.json"), "w", encoding="utf-8") as f:
        json.dump({"TAOUSDC": {"current_price": 255.0, "final_trend": 1, "growth_coefficient": 0.01,
                               "dynamic": {"900": {"slope_full": 0.005, "final_trend": 1}}}}, f)
    with open(os.path.join(cache_dir, "cache_order.json"), "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": [{"side": "SELL", "price": 258.9, "quantity": 8.68, "timestamp": 123}]}}, f)

    plan = {
        "summary": "Routine 5m check",
        "diagnosed_issues": [{"severity": "INFO", "category": "HEALTHY", "description": "ok", "root_cause": "ok"}],
        "venue_actions": [{"action": "NONE", "venue": "binance", "symbol": "TAOUSDC", "reason": "aligned"}],
        "market_trading_decision": {"decision": "HOLD", "symbol": "TAOUSDC", "venue": "binance", "confidence": 0.6},
    }
    auditor = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(plan))
    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0)
    gateway.submit.return_value = None
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="enforce",
        notify_enabled=True,
        intent_gateway=gateway,
    )
    snap = reconciler.harvest_sensors_telemetry()
    assert snap.prices_and_trends["TAOUSDC"]["chg_15m_pct"] == 2.0
    assert snap.prices_and_trends["TAOUSDC"]["slope_15m"] == 0.005
    assert snap.portfolio_summary["recent_orders"]["TAOUSDC"]["last_action_side"] == "SELL"

    with patch.object(reconciler, "dispatch_notification") as mock_notify:
        report = reconciler.run_cycle(topic_key="auto_trade_executor", focus_symbols=["TAOUSDC"])
    assert report.actions_executed is False
    assert report.notification_sent is False
    mock_notify.assert_not_called()


def test_intent_gateway_inherits_workspace_dir_and_null_plan_fields_are_safe(temp_env):
    workspace, _, _ = temp_env
    null_plan = {
        "summary": None,
        "diagnosed_issues": None,
        "venue_actions": None,
        "code_remediation": None,
        "market_trading_decision": None,
    }
    auditor = GeminiClient(custom_runner=lambda prompt, model, timeout: json.dumps(null_plan))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
    )
    assert os.path.abspath(reconciler.intent_gateway.workspace_dir) == os.path.abspath(workspace)
    report = reconciler.run_cycle(topic_key="auto_trade_executor")
    assert report.summary == "Audit complete."
    assert report.diagnosed_issues == []
    assert report.venue_actions == []
    assert report.market_trading_decision == {}


def test_extract_symbol_recent_fills_sorts_by_timestamp(temp_env):
    workspace, cache_dir, _ = temp_env
    with open(os.path.join(cache_dir, "cache_order.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "items": {
                    "TAOUSDC": [
                        {"side": "SELL", "price": 265.0, "quantity": 1.0, "timestamp": 2000},
                        {"side": "BUY", "price": 250.0, "quantity": 1.0, "timestamp": 1000},
                    ]
                }
            },
            f,
        )
    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    snap = reconciler.harvest_sensors_telemetry()
    recent = snap.portfolio_summary["recent_orders"]["TAOUSDC"]
    assert recent["last_action_side"] == "SELL"
    assert recent["last_action_actor"] == "BOT"
    assert recent["last_sell"] == {"price": 265.0, "qty": 1.0, "ts_ms": 2000, "actor": "BOT"}
    assert recent["last_buy"] == {"price": 250.0, "qty": 1.0, "ts_ms": 1000, "actor": "BOT"}


def test_apply_code_patch_safely_guards_workspace_boundary_and_restores_in_memory_on_failure(temp_env, tmp_path):
    import subprocess
    workspace, _, _ = temp_env
    reconciler = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"], mode="enforce")

    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    assert reconciler._apply_code_patch_safely(str(outside), "x = 1", "x = 2", "outside") is False
    assert outside.read_text(encoding="utf-8") == "x = 1\n"

    inside = os.path.join(workspace, "module_a.py")
    with open(inside, "w", encoding="utf-8") as f:
        f.write("val = 10\n")

    # No-op patch rejected
    assert reconciler._apply_code_patch_safely(inside, "val = 10", "val = 10", "noop") is False

    # Failed pytest restores exact pre-patch content even in a non-git workspace
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed")

    with patch("subprocess.run", side_effect=_fake_run):
        assert reconciler._apply_code_patch_safely(inside, "val = 10", "val = 20", "bad fix") is False
    with open(inside, "r", encoding="utf-8") as f:
        assert f.read() == "val = 10\n"


def test_intent_gateway_and_advisors_reject_nan_bool_and_null_json_fields(temp_env, tmp_path):
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro.news_feed_collector import NewsFeedSnapshot, NewsHeadline
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway

    workspace, _, _ = temp_env
    gw = AutonomousIntentGateway(workspace_dir=workspace, mode="shadow", min_confidence=0.80, max_notional_usd=2000.0)

    for bad_conf in (float("nan"), "NaN", True, False, None):
        assert gw.submit(
            {"decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "confidence": bad_conf},
            source="auto_trade_executor",
            max_mode="shadow",
        ) is None
        res = gw.process_intent(
            {"intent_id": "test_bad", "decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "confidence": bad_conf}
        )
        assert res["status"] == "REJECTED_LOW_CONFIDENCE"

    null_resp_client = GeminiClient(
        custom_runner=lambda p, m, t: json.dumps(
            {
                "threat_level": None,
                "risk_score": None,
                "summary": None,
                "recommended_brake": None,
                "market_bias": None,
                "risk_level": None,
                "confidence": True,
                "key_risks": None,
                "recommended_action": None,
            }
        )
    )
    geo = GeopoliticalThreatAnalyzer(
        gemini_client=null_resp_client,
        state_file=str(tmp_path / "geo.json"),
    )
    snap = NewsFeedSnapshot(
        headlines=(
            NewsHeadline(
                title="Major offensive reported in Strait",
                source="Test",
                pub_date="2026-10-24",
                url="https://example.com/1",
                matched_keywords=("war",),
                is_high_severity=True,
            ),
        ),
        high_severity_count=1,
        total_fetched=1,
        has_critical_shock_keywords=False,
        ts=100.0,
    )
    geo_res = geo.assess(snap, force_refresh=True, now_ts=100.0)
    assert geo_res.threat_level == "NORMAL"
    assert geo_res.risk_score == 0.5
    assert geo_res.recommended_brake == "NONE"

    adv = SentimentAdvisor(gemini_client=null_resp_client, cache_file=str(tmp_path / "sent.json"))
    adv_res = adv.review(force_refresh=True, now_ts=100.0)
    assert adv_res is not None
    assert adv_res.market_bias == "NEUTRAL"
    assert adv_res.risk_level == "MODERATE"
    assert adv_res.confidence == 0.5
    assert adv_res.key_risks == []
    assert adv_res.recommended_action == "HOLD"


def test_enforce_executes_cancel_order_sync_position_and_restart_bot(mock_workspace):
    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0)
    mock_inst = MagicMock()
    mock_inst.symbol = "TAOUSDC"
    mock_inst.provider_name = "binance"
    gateway.resolve_instrument.return_value = mock_inst

    reconciler = AutonomousAIReconciler(
        workspace_dir=str(mock_workspace),
        cache_dir="cachedb",
        logs_dirs=["logs"],
        mode="enforce",
        notify_enabled=False,
        intent_gateway=gateway,
    )

    plan = {
        "venue_actions": [
            {"action": "CANCEL_ORDER", "venue": "binance", "symbol": "TAOUSDC", "order_id": "778899", "reason": "orphan"},
            {"action": "SYNC_POSITION", "venue": "binance", "symbol": "TAOUSDC", "reason": "reconcile"},
            {"action": "RESTART_BOT", "venue": "fleet", "symbol": "bot1", "reason": "stale heartbeat"},
            {"action": "RESTART_BOT", "venue": "fleet", "symbol": "unauthorized_evil_proc", "reason": "must refuse"},
        ]
    }
    with patch("subprocess.run") as mock_subproc:
        executed, tests_passed = reconciler.apply_reconciliation(plan, allow_code_remediation=True)

    assert executed is True
    assert tests_passed is None
    mock_inst._api.cancel_order.assert_called_once_with("TAOUSDC", "778899", provider_name="binance")
    mock_inst._api.open_orders.assert_called_once_with("TAOUSDC", provider_name="binance")
    mock_subproc.assert_called_once_with(["pkill", "-f", "bot1.py"], check=False)


def test_dispatch_notification_routes_system_and_bot_errors_to_ntfy_error(temp_env):
    workspace, _, _ = temp_env
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        mode="enforce",
        notify_enabled=True,
    )
    reconciler.ntfy_url = ""
    reconciler.ntfy_topic_server = "ntfy-server-1978"
    reconciler.ntfy_topic_error = "ntfy-error-1978"

    mock_resp = MagicMock(status=200, code=200)
    mock_resp.__enter__.return_value = mock_resp

    # 1. Healthy report routes to ntfy-server-1978
    healthy_report = SupervisorAuditReport(
        ts=time.time(),
        mode="enforce",
        summary="All fleet bots and venues healthy.",
        diagnosed_issues=[{"severity": "INFO", "category": "HEALTHY", "description": "0 anomalies"}],
        venue_actions=[],
        code_remediation={"has_fix": False},
        actions_executed=False,
    )
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
         patch("notify_engine.alertnotifiers.pass_common_gate", return_value=(True, "ok", "fp", None)):
        assert reconciler.dispatch_notification(healthy_report) is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://ntfy.sh/ntfy-server-1978"
        title_str = req.headers["Title"].encode("latin-1").decode("utf-8")
        assert title_str == "🛠 [auto_technical_repair] Issues: 0 | Actions: 0"

    # 2. Linux OS / VPN / Memory / Disk issue routes to ntfy-server-1978 (with high priority)
    sys_error_report = SupervisorAuditReport(
        ts=time.time(),
        mode="enforce",
        summary="Disk usage elevated at 88% and VPN wgpia0 reconnected.",
        diagnosed_issues=[{"severity": "HIGH", "category": "OS_RESOURCE", "description": "Disk usage 88%"}],
        venue_actions=[],
        code_remediation={"has_fix": False},
        actions_executed=False,
    )
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
         patch("notify_engine.alertnotifiers.pass_common_gate", return_value=(True, "ok", "fp", None)):
        assert reconciler.dispatch_notification(sys_error_report) is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://ntfy.sh/ntfy-server-1978"
        title_str = req.headers["Title"].encode("latin-1").decode("utf-8")
        assert title_str == "🚨 [auto_technical_repair] SYSTEM Issues: 1 | Actions: 0"
        assert req.headers["Priority"] == "high"

    # 3. Fleet bot error / order / Python code repair routes to ntfy-error-1978
    error_report = SupervisorAuditReport(
        ts=time.time(),
        mode="enforce",
        summary="Bot rtrade threw unhandled exception; restarted.",
        diagnosed_issues=[{"severity": "HIGH", "category": "BOT_EXCEPTION", "description": "Socket hang in rtrade"}],
        venue_actions=[{"action": "RESTART_BOT", "venue": "fleet", "symbol": "rtrade", "reason": "hung"}],
        code_remediation={"has_fix": False},
        actions_executed=True,
    )
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
         patch("notify_engine.alertnotifiers.pass_common_gate", return_value=(True, "ok", "fp", None)):
        assert reconciler.dispatch_notification(error_report) is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://ntfy.sh/ntfy-error-1978"
        title_str = req.headers["Title"].encode("latin-1").decode("utf-8")
        assert title_str == "🚨 [auto_technical_repair] ERROR Issues: 1 | Actions: 1"
        assert req.headers["Priority"] == "high"

    # 4. LOW-severity note with 0 repair actions stays on ntfy-server-1978 with default priority (no false alarm)
    low_note_report = SupervisorAuditReport(
        ts=time.time(),
        mode="enforce",
        summary="All healthy; minor HTTP 429 rate-limit note handled by cache fallback.",
        diagnosed_issues=[{"severity": "LOW", "category": "BOT_EXCEPTION", "description": "CMC HTTP 429 fallback"}],
        venue_actions=[],
        code_remediation={"has_fix": False},
        actions_executed=False,
    )
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
         patch("notify_engine.alertnotifiers.pass_common_gate", return_value=(True, "ok", "fp", None)):
        assert reconciler.dispatch_notification(low_note_report) is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://ntfy.sh/ntfy-server-1978"
        title_str = req.headers["Title"].encode("latin-1").decode("utf-8")
        assert title_str == "🛠 [auto_technical_repair] Issues: 1 | Actions: 0"
        assert req.headers["Priority"] == "default"



def test_multi_coin_trading_decisions_and_full_telemetry_prompt_context(temp_env):
    workspace, cache_dir, _ = temp_env

    with open(os.path.join(cache_dir, "sentiment_eval.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "market_bias": "BEARISH",
                "risk_level": "HIGH",
                "recommended_action": "DEFENSIVE",
                "confidence": 0.88,
                "summary": "Broad washout risk",
                "key_risks": ["Breadth collapse"],
            },
            f,
        )
    with open(os.path.join(cache_dir, "geopolitical_eval.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "threat_level": "ELEVATED",
                "risk_score": 0.62,
                "recommended_brake": "DOWNSCALE_QTY",
                "summary": "Regional energy tension",
            },
            f,
        )
    with open(os.path.join(cache_dir, "market_breadth_collect.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "advance_ratio": 0.18,
                "median_change_pct": -3.1,
                "regime": "BEARISH_BREADTH",
                "advancing": 12,
                "total_symbols": 66,
            },
            f,
        )
    krk_dir = os.path.join(workspace, "kraken")
    os.makedirs(krk_dir, exist_ok=True)
    with open(os.path.join(krk_dir, ".state_HYPEUSD.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "qty": 13.68,
                "cost": 1079.47,
                "entry_price": 80.30,
                "last_buy_price": 75.55,
                "last_sell_price": 80.75,
                "last_exit_kind": "TP",
                "dca_buys": 2,
            },
            f,
        )

    # Extra cachedb telemetry feeds: breaking news, 24trend/live_*, kalman_state, cache_T_trend, 7d range, retry queue
    with open(os.path.join(cache_dir, "news_feed_collect.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "title": "Sudden escalation alert in Strait of Hormuz",
                    "source": "Bloomberg",
                    "matched_keywords": ["escalation"],
                    "is_high_severity": True,
                    "first_seen_ts": time.time() - 120.0,
                }
            ],
            f,
        )
    os.makedirs(os.path.join(cache_dir, "24trend"), exist_ok=True)
    with open(os.path.join(cache_dir, "24trend", "live_HYPEUSD.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "222": {"final_trend": -1, "growth_coefficient": -0.04, "slope_full": -0.025, "current_price": 75.50},
                "900": {"final_trend": -1, "growth_coefficient": -0.03, "slope_full": -0.018, "current_price": 75.50},
                "9000": {"final_trend": -1, "growth_coefficient": -0.01, "slope_full": -0.009, "current_price": 75.50},
            },
            f,
        )
    with open(os.path.join(cache_dir, "cache_kalman_trend.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "HYPEUSD": {
                    "kalman_vel": -0.031,
                    "kalman_vel_std": 0.012,
                    "kalman_trend": -1,
                    "vol_1h_pct": 0.74,
                    "adapt_reentry_pct": 1.48,
                }
            },
            f,
        )
    with open(os.path.join(cache_dir, "cache_long_trend.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "items": {
                    "HYPEUSD": [
                        {"direction": "down", "duration_seconds": 36000.0, "estimated_future_hours": 42.5}
                    ]
                }
            },
            f,
        )
    with open(os.path.join(cache_dir, "cache_T_trend.json"), "w", encoding="utf-8") as f:
        json.dump({"HYPEUSD": {"median_d": 3.0, "p90_d": 6.2}}, f)
    with open(os.path.join(cache_dir, "cache_watcher_all_price.json"), "w", encoding="utf-8") as f:
        json.dump({"items": {"HYPE": [[1000, 80.0], [2000, 74.5], [3000, 76.0]]}}, f)
    with open(os.path.join(cache_dir, "order_retry_queue.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "provider_name": "Binance",
            "symbol": "TAOUSDC",
            "side": "SELL",
            "qty": 3.31,
            "requested_price": 250.45,
            "kind": "monitor_order_replace",
            "attempts": 2,
            "last_failure_reason": "profit_guard",
            "lifecycle": "submit_pending",
            "expires_at_ts": time.time() + 3600.0,
        }) + "\n")

    captured_prompts = []
    plan = {
        "summary": "Multi-coin protective de-risking on BTCUSDC and HYPEUSD.",
        "diagnosed_issues": [],
        "venue_actions": [],
        "market_trading_decisions": [
            {
                "decision": "SELL",
                "symbol": "HYPEUSD",
                "venue": "kraken",
                "confidence": 0.89,
                "suggested_notional_usd": 800.0,
                "thesis": "Breakdown below 76 support",
            },
            {
                "decision": "SELL",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "confidence": 0.94,
                "suggested_notional_usd": 1800.0,
                "thesis": "Protective sell against cascading flush",
            },
        ],
    }

    def _runner(prompt, model, timeout):
        captured_prompts.append(prompt)
        return json.dumps(plan)

    auditor = GeminiClient(custom_runner=_runner)
    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    gateway.submit.side_effect = [
        {"intent_id": "i_btc", "symbol": "BTCUSDC", "status": "EXECUTED"},
        {"intent_id": "i_hype", "symbol": "HYPEUSD", "status": "EXECUTED"},
    ]
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="enforce",
        notify_enabled=False,
        intent_gateway=gateway,
    )
    report = reconciler.run_cycle(topic_key="auto_trade_executor")

    assert len(captured_prompts) == 1
    prompt_text = captured_prompts[0]
    assert "action=DEFENSIVE" in prompt_text
    assert "Breadth collapse" in prompt_text
    assert "brake=DOWNSCALE_QTY" in prompt_text
    assert "[HIGH-SEV] Sudden escalation alert in Strait of Hormuz" in prompt_text
    assert "median_chg=-3.1%" in prompt_text
    assert "7d=-5.00% 7d_range=[74.5, 80.0]" in prompt_text
    assert "vol_1h=0.74%(adapt_reentry=1.48%)" in prompt_text
    assert "est_rem=42.5h, med_d=3.0d/p90=6.2d" in prompt_text
    assert "order_retry_queue.jsonl" in prompt_text
    assert "[kraken] HYPEUSD: qty=13.68" in prompt_text
    assert "ZERO PROFIT OR STRATEGY GUARDS" in prompt_text
    assert "between 10 and 2000 USD" in prompt_text
    assert "zero artificial cooldowns or dedup locks" in prompt_text
    assert "1. Formulate symmetric autonomous trading decisions" in prompt_text
    assert "Explain the technical root cause of any bot exceptions" not in prompt_text
    assert "LOG ANOMALIES & GUARD VETOES" not in prompt_text
    assert "FLEET PROCESSES" not in prompt_text
    assert "LINUX SYSTEM RESOURCES & VPN TUNNEL HEALTH" not in prompt_text
    assert "diagnosed_issues" not in prompt_text
    assert "bullet points for: Piața Live & Pante" in prompt_text
    assert "Starea Flotei" not in prompt_text

    # Highest-confidence actionable decision (BTCUSDC @ 0.94) executes first and becomes primary summary
    assert gateway.submit.call_count == 2
    assert gateway.submit.call_args_list[0].args[0]["symbol"] == "BTCUSDC"
    assert gateway.submit.call_args_list[1].args[0]["symbol"] == "HYPEUSD"
    assert report.actions_executed is True
    assert report.market_trading_decision["symbol"] == "BTCUSDC"
    assert len(report.market_trading_decisions) == 2
    assert len(report.intents_emitted) == 2
    assert report.venue_actions == []

    # Verify auto_technical_repair receives ONLY fleet, bot heartbeats, Linux system resources, internet/VPN (wgpia0),
    # config audit, and persistent log anomalies — NEVER market prices, slopes, portfolio balances, or active positions.
    reconciler.run_cycle(topic_key="auto_technical_repair")
    repair_prompt = captured_prompts[-1]
    assert "LOG ANOMALIES & GUARD VETOES" in repair_prompt
    assert "LINUX SYSTEM RESOURCES & VPN TUNNEL HEALTH" in repair_prompt
    assert "VPN Tunnel (wgpia0):" in repair_prompt
    assert "Internet & DNS Connectivity:" in repair_prompt
    assert "Configuration Files Audit:" in repair_prompt
    assert "Starea Flotei, Boților, Sistemului Linux & VPN (wgpia0) & Loguri" in repair_prompt
    assert "diagnosed_issues" in repair_prompt
    assert "LIVE MARKET PRICES, MULTI-HORIZON TRENDS" not in repair_prompt
    assert "SPOT WALLET BALANCES, TRADABLE POSITIONS" not in repair_prompt
    assert "Portofoliu, Solduri & Poziții Active" not in repair_prompt
    assert "Piața Live & Pante (Toate Monedele)" not in repair_prompt


def test_auto_technical_repair_persistent_errors_config_patching_and_system_reboots(temp_env):
    import subprocess
    workspace, _, logs_dir = temp_env

    # 1. Persistent unresolved errors in logs & config audit
    with open(os.path.join(workspace, "order_guard.conf"), "w", encoding="utf-8") as f:
        f.write("supervisor_mode = enforce\nsupervisor_notify = false\nmalformed_line_without_equals\n")
    with open(os.path.join(workspace, "procs.conf"), "w", encoding="utf-8") as f:
        f.write("rtrade.py|$ROOT|python rtrade.py|rtrade|cachedb/rtrade.heartbeat|180|fleet\n")
    with open(os.path.join(workspace, "instruments.conf"), "w", encoding="utf-8") as f:
        f.write("[BINANCE_BTC]\nprovider = binance\nsymbol = BTCUSDC\nenabled = yes\n")

    now = time.time()
    stamp = lambda ago: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - ago))  # noqa: E731
    with open(os.path.join(logs_dir, "rtrade.log"), "w", encoding="utf-8") as f:
        f.write(f"{stamp(600)} [ERROR] Unhandled KeyError in rtrade execution loop\n")
        f.write(f"{stamp(300)} [ERROR] Unhandled KeyError in rtrade execution loop\n")
        f.write(f"{stamp(60)} [ERROR] Unhandled KeyError in rtrade execution loop\n")

    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        mode="enforce",
        notify_enabled=False,
    )
    logs = reconciler.harvest_log_anomalies(lookback_hours=1.0)
    assert len(logs.persistent_errors) == 1
    assert logs.persistent_errors[0]["file"] == "rtrade.log"
    assert logs.persistent_errors[0]["count"] == 3
    assert logs.persistent_errors[0]["unresolved_at_tail"] is True

    sensors = reconciler.harvest_sensors_telemetry(include_fleet=True)
    cfg_audit = sensors.system_and_vpn.get("config_audit", {})
    assert "order_guard.conf" in cfg_audit.get("checked_files", [])
    assert "instruments.conf" in cfg_audit.get("checked_files", [])
    assert not any("instruments.conf" in iss for iss in cfg_audit.get("issues", []))
    assert any("malformed_line_without_equals" in iss for iss in cfg_audit.get("issues", []))

    # 2. _apply_code_patch_safely allows fixing config files (order_guard.conf, procs.conf) while protecting .env
    env_secret = os.path.join(workspace, ".env")
    with open(env_secret, "w", encoding="utf-8") as f:
        f.write("API_SECRET=topsecret\n")
    assert reconciler._apply_code_patch_safely(env_secret, "topsecret", "hacked", "must refuse .env") is False

    og_conf = os.path.join(workspace, "order_guard.conf")
    with patch("subprocess.run", return_value=subprocess.CompletedProcess(["pytest"], 0, stdout="", stderr="")):
        assert reconciler._apply_code_patch_safely(
            og_conf,
            "malformed_line_without_equals\n",
            "",
            "Remove malformed line in order_guard.conf",
        ) is True
    with open(og_conf, "r", encoding="utf-8") as f:
        assert "malformed_line_without_equals" not in f.read()

    # 3. _execute_venue_or_fleet_action executes RESTART_MODULE, RESTART_FLEET, RESTART_VPN, PIN_DNS, and REBOOT_SYSTEM
    plan = {
        "venue_actions": [
            {"action": "RESTART_MODULE", "venue": "fleet", "symbol": "rtrade", "reason": "hung module"},
            {"action": "RESTART_FLEET", "venue": "fleet", "symbol": "FLEET", "reason": "fleet wedge"},
            {"action": "RESTART_VPN", "venue": "system", "symbol": "wgpia0", "reason": "vpn down"},
            {"action": "PIN_DNS", "venue": "system", "symbol": "dns", "reason": "dns failure"},
            {"action": "REBOOT_SYSTEM", "venue": "system", "symbol": "SYSTEM", "reason": "kernel deadlock"},
        ]
    }
    with patch("subprocess.run", return_value=subprocess.CompletedProcess(["cmd"], 0, stdout="", stderr="")) as mock_run:
        executed, _ = reconciler.apply_reconciliation(plan, allow_code_remediation=True)
    assert executed is True
    assert mock_run.call_count == 5
    cmds = [c.args[0] for c in mock_run.call_args_list]
    assert cmds[0] == ["pkill", "-f", "rtrade.py"]
    assert any("restart" in cmd for cmd in cmds[1:])
    assert any("reboot" in cmd for cmd in cmds[1:])


def test_auto_trade_executor_never_dispatches_sre_notifications_or_executes_venue_actions(temp_env):
    workspace, _, _ = temp_env
    plan = {
        "summary": "Trade execution turn with incidental venue_actions from LLM.",
        "diagnosed_issues": [{"severity": "HIGH", "category": "ORPHAN_ORDER", "description": "Queued retry"}],
        "venue_actions": [{"action": "CANCEL_ORDER", "venue": "binance", "symbol": "TAOUSDC", "order_id": "123"}],
        "market_trading_decisions": [
            {"decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "confidence": 0.92, "suggested_notional_usd": 400.0}
        ],
    }
    auditor = GeminiClient(custom_runner=lambda p, m, t: json.dumps(plan))
    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0, notify_enabled=True)
    gateway.submit.return_value = {"intent_id": "i_tao", "symbol": "TAOUSDC", "status": "EXECUTED"}
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="enforce",
        notify_enabled=True,
        intent_gateway=gateway,
    )
    with patch.object(reconciler, "dispatch_notification") as mock_sre_notify, \
         patch.object(reconciler, "apply_reconciliation") as mock_reconcile:
        report = reconciler.run_cycle(topic_key="auto_trade_executor")

    mock_sre_notify.assert_not_called()
    mock_reconcile.assert_not_called()
    assert report.venue_actions == []
    assert report.actions_executed is True
    assert report.notification_sent is True


def test_recent_log_lines_handles_invalid_glog_and_year_wrap():
    from datetime import datetime
    from orchestratorOS.supervisor.autonomous_ai_reconciler import recent_log_lines

    jan1_mtime = datetime(2027, 1, 1, 0, 10, 0).timestamp()
    cutoff = jan1_mtime - 3600.0  # 1 hour lookback (includes Dec 31 23:50, excludes Dec 31 20:00)
    lines = [
        "E1231 20:00:00 [ERROR] old dec 31 error\n",
        "E9999 99:99:99 [ERROR] malformed glog timestamp\n",
        "E1231 23:50:00 [ERROR] recent dec 31 error\n",
    ]
    picked = recent_log_lines(lines, jan1_mtime, cutoff)
    assert len(picked) == 2
    assert "E1231 23:50:00 [ERROR] recent dec 31 error\n" in picked
    assert "E1231 20:00:00 [ERROR] old dec 31 error\n" not in picked

    # Un-timestamped continuation lines belonging to an expired entry (< cutoff) must not leak
    # into the lookback window when a newer entry (>= cutoff) is appended later.
    multiline_old = [
        "[2026-10-26 10:30:06] [manage_agy] ERROR: Failed to recover daemon.\n",
        "W1026 10:30:02.942368 1 composite_token_storage.go:419] Failed to load token\n",
        "Error: failed to start service: exit status 1\n",
        "See systemctl --user status antigravity-cli-daemon.service for details.\n",
        "[2026-10-26 11:33:01] [manage_agy] ui-sync: restarting to surface threads.\n",
        "Remote-control daemon started.\n",
    ]
    noon_mtime = datetime(2026, 10, 26, 12, 0, 0).timestamp()
    picked_multi = recent_log_lines(multiline_old, noon_mtime, noon_mtime - 0.6 * 3600.0)
    assert picked_multi == multiline_old[4:]


def test_monitor_check_processes_requires_all_heartbeat_files_fresh(tmp_path, monkeypatch):
    (tmp_path / "procs.conf").write_text(
        "rtrade.py|$ROOT|python rtrade.py|rtrade|cachedb/rtrade.heartbeat,logs/rtrade.log|180|fleet\n"
    )
    (tmp_path / "cachedb").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    hb = tmp_path / "cachedb" / "rtrade.heartbeat"
    lg = tmp_path / "logs" / "rtrade.log"
    hb.write_text("hb")
    lg.write_text("log")
    now = 1_800_000_000.0
    # Log is fresh (now), but heartbeat is stale (now - 300s > 180s)
    os.utime(hb, (now - 300.0, now - 300.0))
    os.utime(lg, (now, now))

    monitor = FleetHealthMonitor(workspace_dir=str(tmp_path))
    monkeypatch.setattr(monitor, "_is_process_running", lambda pat: True)
    res = monitor.check_processes(now=now)
    assert len(res) == 1
    assert res[0].heartbeat_fresh is False
    assert res[0].age_seconds == 300.0


def test_harvest_and_synthesize_surfaces_active_avg_buy_tranches_and_open_orders(temp_env):
    workspace, cache_dir, _ = temp_env
    with open(os.path.join(workspace, "instruments.conf"), "w", encoding="utf-8") as f:
        f.write("TAOUSDC|binance|TAOUSDC|TAO|USDC|1\n")

    now_ms = int(time.time() * 1000)
    with open(os.path.join(cache_dir, "cache_order.json"), "w", encoding="utf-8") as f:
        json.dump({
            "items": {
                "TAOUSDC": [
                    {"orderId": 1, "side": "BUY", "status": "FILLED", "executedQty": "3.0", "cummulativeQuoteQty": "780.0", "time": now_ms - 7200_000},
                    {"orderId": 2, "side": "BUY", "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "240.0", "time": now_ms - 3600_000},
                ]
            }
        }, f)

    mock_inst = MagicMock()
    mock_inst.symbol = "TAOUSDC"
    mock_inst.provider_name = "binance"
    mock_inst.base = "TAO"
    mock_inst.quote = "USDC"
    mock_inst.params = {}
    mock_inst._provider.get_current_price.return_value = 255.0
    mock_inst._provider.free_balance.side_effect = lambda a: 4.0 if a == "TAO" else 500.0
    mock_inst.open_orders.return_value = [
        {"orderId": "999111", "side": "BUY", "price": "248.0", "origQty": "0.5", "executedQty": "0.0"}
    ]

    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    gateway.load_tradable_instruments.return_value = {"TAOUSDC": mock_inst}

    captured_prompts = []
    auditor = GeminiClient(custom_runner=lambda p, m, t: captured_prompts.append(p) or json.dumps({"summary": "ok"}))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
        intent_gateway=gateway,
    )
    sensors = reconciler.harvest_sensors_telemetry()
    ps = sensors.portfolio_summary
    assert ps["recent_orders"]["TAOUSDC"]["active_avg_buy_px"] == 255.0
    assert len(ps["recent_orders"]["TAOUSDC"]["recent_buys"]) == 2
    assert ps["open_orders"][0]["orderId"] == "999111"
    pos = ps["tradable_positions"][0]
    assert pos["active_avg_buy_px"] == 255.0
    assert pos["last_buy_px"] == 240.0
    assert pos["pnl_vs_last_buy_pct"] == 6.25

    reconciler.synthesize_and_diagnose(sensors, reconciler.harvest_log_anomalies(), topic_key="auto_trade_executor")
    assert len(captured_prompts) == 1
    prompt_text = captured_prompts[0]
    assert "avg_buy_px=255.0" in prompt_text
    assert "last_buy_px=240.0 (pnl_vs_last_buy=6.25%)" in prompt_text
    assert "Active Open Orders on Exchange (Locking Liquidity)" in prompt_text
    assert "Recent BUY Tranches=[3.0@260.0, 1.0@240.0]" in prompt_text


def test_verify_tools_monitor_night_snapshot_and_watch_decisions():
    from verify_tools import monitor_night, portfolio_snapshot, watch_trade_decisions

    # 1. monitor_night.load_procs resolves fleet labels and comma-separated hb_log paths
    procs_text = (
        "market_monitor/price_watcher.py|$ROOT|python market_monitor/price_watcher.py|price_watcher|cachedb/price_watcher.heartbeat,logs/price_watcher.log|180|fleet\n"
        "hyperliquid/bot.py|$ROOT/hyperliquid|python bot.py|HL-bot|cachedb/HL-bot.heartbeat,logs/HL-bot.log|600|bot\n"
    )
    with patch.object(monitor_night, "run", return_value=procs_text):
        procs = monitor_night.load_procs(MagicMock())
    assert len(procs) == 2
    assert procs[0].log == "logs/price_watcher.log"
    assert procs[1].log == "logs/HL-bot.log"

    # 2. portfolio_snapshot._ntfy encodes non-ASCII emojis safely for HTTP headers
    with patch("urllib.request.urlopen") as mock_open, \
         patch.dict(
             os.environ,
             {
                 "NTFY_TOPIC": "test-topic",
                 "DISABLE_EXTERNAL_NOTIFICATIONS": "0",
                 "BACKTEST_MODE": "0",
                 "REPLAY_MODE": "0",
             },
         ):
        portfolio_snapshot._ntfy("⚠️ Drawdown 5.5% (-$100)", "body")
        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        assert req.headers["Title"].encode("latin-1").decode("utf-8") == "⚠️ Drawdown 5.5% (-$100)"

    # 3. watch_trade_decisions.position picks latest BUY by timestamp
    now_ms = int(time.time() * 1000)
    fake_buys = [
        {"qty": "1.0", "price": "220.0", "timestamp": now_ms - 7200_000},
        {"qty": "1.0", "price": "260.0", "timestamp": now_ms - 1800_000},
    ]
    with patch.object(
        watch_trade_decisions.apiorders,
        "get_trade_orders",
        side_effect=lambda side, sym, maxage: fake_buys if side == "BUY" else [],
    ):
        avg_px, net_qty, n_sells, last_buy = watch_trade_decisions.position("TAOUSDC")
        assert avg_px == 240.0
        assert net_qty == 2.0
        assert n_sells == 0
        assert last_buy == 260.0

    # 4. parse_shadow, kraken_pnl_backfill, and visuals.drawprices edge-case guards
    from verify_tools import parse_shadow, kraken_pnl_backfill
    from visuals import drawprices
    import numpy as np

    assert parse_shadow._f(None) == 0.0
    assert parse_shadow._f(True) == 0.0
    assert parse_shadow._f(float("nan")) == 0.0
    assert parse_shadow._i(None) == 0
    assert parse_shadow._i(True) == 0
    assert parse_shadow._last_valid_record(['{"pair": "HYPEUSD"}', "{bad json"]) == {"pair": "HYPEUSD"}

    assert kraken_pnl_backfill._safe_float(None) == 0.0
    assert kraken_pnl_backfill._safe_float(True) == 0.0
    assert kraken_pnl_backfill._safe_float(float("nan")) == 0.0

    ts_arr, px_arr, dt_list = drawprices.to_series([(1700000000000, float("nan")), (True, 100.0), (1700000001000, 50.0)])
    assert len(px_arr) == 1 and px_arr[0] == 50.0
    scaled_empty, vmin, vmax = drawprices.min_max_scale(np.array([], dtype=np.float64))
    assert len(scaled_empty) == 0
    lin_one = drawprices.forecast_linear(ts_arr, px_arr, horizon=3)
    assert list(lin_one) == [50.0, 50.0, 50.0]



def test_harvest_and_synthesize_enriches_hyperliquid_and_t212_cycles(temp_env):
    workspace, cache_dir, _ = temp_env
    with open(os.path.join(workspace, "instruments.conf"), "w", encoding="utf-8") as f:
        f.write("")

    hl_dir = os.path.join(workspace, "hyperliquid")
    os.makedirs(hl_dir, exist_ok=True)
    with open(os.path.join(hl_dir, ".state_HYPE.json"), "w", encoding="utf-8") as f:
        json.dump({
            "qty": 10.0,
            "cost": 380.0,
            "entry_price": 38.0,
            "last_buy_price": 38.0,
            "last_sell_price": 41.2,
            "last_exit_kind": "TP",
            "dca_buys": 1,
        }, f)
    # Backup and paper state files in hyperliquid/ must be ignored
    with open(os.path.join(hl_dir, ".state_HYPE.pre_fee_reconcile_20260822_084538.json"), "w", encoding="utf-8") as f:
        json.dump({"qty": 99.0, "cost": 9999.0}, f)
    with open(os.path.join(hl_dir, ".state_HYPE_long.json"), "w", encoding="utf-8") as f:
        json.dump({"qty": 0.0, "cost": 0.0}, f)

    t212_dir = os.path.join(workspace, "212trading")
    os.makedirs(t212_dir, exist_ok=True)
    with open(os.path.join(t212_dir, ".state_SPCX_US_EQ.json"), "w", encoding="utf-8") as f:
        json.dump({
            "qty": 5.0,
            "cost_usd": 150.0,
            "entry_price": 30.0,
            "last_buy_price": 30.0,
            "last_sell_price": 31.5,
            "last_exit_kind": "TP",
            "dca_buys": 0,
            "pos_peak": 32.0,
        }, f)

    hl_inst = MagicMock()
    hl_inst.symbol = "HYPEUSDC"
    hl_inst.provider_name = "hyperliquid"
    hl_inst.base = "HYPE"
    hl_inst.quote = "USDC"
    hl_inst.params = {}
    hl_inst._provider.get_current_price.return_value = 40.0
    hl_inst._provider.free_balance.side_effect = lambda a: 10.0 if a == "HYPE" else 200.0
    hl_inst.open_orders.return_value = []

    t212_inst = MagicMock()
    t212_inst.symbol = "SPCX_US_EQ"
    t212_inst.provider_name = "t212"
    t212_inst.base = "SPCX_US_EQ"
    t212_inst.quote = "USD"
    t212_inst.params = {}
    t212_inst._provider.get_current_price.return_value = 30.9
    t212_inst._provider.free_balance.side_effect = lambda a: 5.0 if a == "SPCX_US_EQ" else 120.0
    t212_inst.open_orders.return_value = []

    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    gateway.load_tradable_instruments.return_value = {
        "HYPERLIQUID_HYPE": hl_inst,
        "T212_SPCX": t212_inst,
    }

    captured_prompts = []
    auditor = GeminiClient(custom_runner=lambda p, m, t: captured_prompts.append(p) or json.dumps({"summary": "ok"}))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
        intent_gateway=gateway,
    )
    sensors = reconciler.harvest_sensors_telemetry()
    ps = sensors.portfolio_summary
    assert list(ps["hyperliquid_cycles"].keys()) == ["HYPEUSDC"]
    assert "SPCX_US_EQ" in ps["t212_cycles"]

    pos_by_sym = {p["symbol"]: p for p in ps["tradable_positions"]}
    assert pos_by_sym["HYPEUSDC"]["active_avg_buy_px"] == 38.0
    assert pos_by_sym["HYPEUSDC"]["pnl_vs_avg_pct"] == 5.26
    assert pos_by_sym["SPCX_US_EQ"]["active_avg_buy_px"] == 30.0
    assert pos_by_sym["SPCX_US_EQ"]["pnl_vs_avg_pct"] == 3.0
    assert pos_by_sym["SPCX_US_EQ"]["trailing_peak"] == 32.0
    assert pos_by_sym["SPCX_US_EQ"]["dd_from_peak_pct"] == -3.44

    reconciler.synthesize_and_diagnose(sensors, reconciler.harvest_log_anomalies(), topic_key="auto_cryptotrade_executor")
    crypto_prompt = captured_prompts[0]
    assert "Hyperliquid Active Cycle Cost & Drawdown Anchors:" in crypto_prompt
    assert "venue_fee_per_leg=7bps" in crypto_prompt
    assert '"venue": "binance" | "kraken" | "hyperliquid"' in crypto_prompt

    reconciler.synthesize_and_diagnose(sensors, reconciler.harvest_log_anomalies(), topic_key="auto_sharetrade_executor")
    share_prompt = captured_prompts[1]
    assert "Trading212 Active Cycle Cost & Drawdown Anchors:" in share_prompt
    assert "venue_fee_per_leg=15bps" in share_prompt
    assert '"venue": "t212"' in share_prompt

    # Outside US RTH, auto_sharetrade_executor skips LLM evaluation unless force_rth_open=True
    with patch("orchestratorOS.supervisor.autonomous_ai_reconciler._is_us_rth_open", return_value=False):
        closed_rep = reconciler.run_cycle(topic_key="auto_sharetrade_executor")
    assert "closed; cycle skipped" in closed_rep.summary
    assert len(captured_prompts) == 2

    with patch("orchestratorOS.supervisor.autonomous_ai_reconciler._is_us_rth_open", return_value=False):
        forced_rep = reconciler.run_cycle(topic_key="auto_sharetrade_executor", force_rth_open=True)
    assert forced_rep.summary == "ok"
    assert len(captured_prompts) == 3

    # Verify scope="equities" vs scope="crypto" filters recent_ai_intents and active_retry_queue
    now_ts = time.time()
    with open(os.path.join(cache_dir, "autonomous_trade_intents_state.json"), "w", encoding="utf-8") as f:
        json.dump([
            {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "status": "EXECUTED", "suggested_notional_usd": 100.0, "timestamp": now_ts},
            {"decision": "BUY", "symbol": "SPCX_US_EQ", "venue": "t212", "status": "EXECUTED", "suggested_notional_usd": 50.0, "timestamp": now_ts},
        ], f)
    with open(os.path.join(cache_dir, "order_retry_queue.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"lifecycle": "pending", "expires_at_ts": now_ts + 3600, "provider_name": "binance", "symbol": "BTCUSDC", "side": "BUY", "qty": 0.001, "requested_price": 65000.0, "kind": "DCA", "attempts": 1}) + "\n")
        f.write(json.dumps({"lifecycle": "pending", "expires_at_ts": now_ts + 3600, "provider_name": "t212", "symbol": "SPCX_US_EQ", "side": "BUY", "qty": 2.0, "requested_price": 30.5, "kind": "DCA", "attempts": 1}) + "\n")

    eq_snap = reconciler.harvest_sensors_telemetry(asset_scope="equities")
    assert [i["venue"] for i in eq_snap.portfolio_summary.get("recent_ai_intents", [])] == ["t212"]
    assert [r["venue"] for r in eq_snap.portfolio_summary.get("active_retry_queue", [])] == ["t212"]

    cr_snap = reconciler.harvest_sensors_telemetry(asset_scope="crypto")
    assert [i["venue"] for i in cr_snap.portfolio_summary.get("recent_ai_intents", [])] == ["binance"]
    assert [r["venue"] for r in cr_snap.portfolio_summary.get("active_retry_queue", [])] == ["binance"]


def test_reconciler_monitor_and_verify_tools_nan_bool_and_quote_hardening(temp_env, tmp_path, monkeypatch):
    import subprocess
    from orchestratorOS.supervisor.autonomous_ai_reconciler import _normalize_trading_decisions
    from verify_tools import pnl_report, portfolio_snapshot, watch_trade_decisions

    workspace, cache_dir, _ = temp_env

    # 1. AutonomousAIReconciler constructor validates lookback_hours, mode, and string supervisor_notify
    for bad_lb in (True, False, 0, -5, float("nan"), float("inf")):
        try:
            AutonomousAIReconciler(workspace_dir=workspace, lookback_hours=bad_lb)
            raise AssertionError(f"Expected ValueError for lookback_hours={bad_lb!r}")
        except ValueError:
            pass
    try:
        AutonomousAIReconciler(workspace_dir=workspace, mode=True)
        raise AssertionError("Expected ValueError for boolean mode")
    except ValueError:
        pass

    with open(os.path.join(workspace, "order_guard.conf"), "w", encoding="utf-8") as f:
        f.write("supervisor_mode = \"shadow\"\nsupervisor_notify = true\n")
    rec_str_notify = AutonomousAIReconciler(workspace_dir=workspace, cache_dir="cachedb", logs_dirs=["logs"])
    assert rec_str_notify.mode == "shadow"
    assert rec_str_notify.notify_enabled is True

    # 2. _normalize_trading_decisions ignores boolean confidence when sorting
    decs = _normalize_trading_decisions({
        "market_trading_decisions": [
            {"venue": "binance", "symbol": "BTCUSDC", "decision": "BUY", "confidence": True},
            {"venue": "binance", "symbol": "TAOUSDC", "decision": "BUY", "confidence": 0.91},
        ]
    })
    assert [d["symbol"] for d in decs] == ["TAOUSDC", "BTCUSDC"]

    # 3. cache_current_price.json picks the latest sample item[-1] and write_audit_artifact sanitizes NaN
    with open(os.path.join(cache_dir, "cache_current_price.json"), "w", encoding="utf-8") as f:
        json.dump({"items": {"TAOUSDC": [[1000, 200.0], [2000, 265.5]]}}, f)
    snap = rec_str_notify.harvest_sensors_telemetry()
    assert snap.prices_and_trends["TAOUSDC"]["price"] == 265.5

    rec_str_notify.state_file = os.path.join(cache_dir, "auto_cryptotrade_executor_eval.json")
    rec_str_notify.heartbeat_file = os.path.join(cache_dir, "auto_cryptotrade_executor.heartbeat")
    nan_report = SupervisorAuditReport(
        ts=time.time(),
        mode="shadow",
        summary="NaN guard test",
        market_trading_decision={"symbol": "TAOUSDC", "confidence": float("nan")},
    )
    rec_str_notify.write_audit_artifact(nan_report)
    with open(rec_str_notify.state_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["market_trading_decision"]["confidence"] is None

    # 4. FleetHealthMonitor sanitizes sensor floats, strips quotes in order_guard.conf, and parses latest_overnight.log
    with open(os.path.join(cache_dir, "geopolitical_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"threat_level": None, "risk_score": True, "summary": None}, f)
    with open(os.path.join(cache_dir, "fear_greed_collect.json"), "w", encoding="utf-8") as f:
        json.dump({"value": "bad_val"}, f)
    mon = FleetHealthMonitor(workspace_dir=workspace)
    assert mon.get_autonomous_modes()["supervisor_mode"] == "shadow"
    s_stat = mon.gather_sensor_status()
    assert s_stat["geopolitical_threat"] == {"threat_level": "NORMAL", "risk_score": 0.0, "summary": ""}
    assert s_stat["fear_greed"] is None

    fake_ssh_out = (
        "RUNNING\n"
        "==> /home/predut/mptrade/logs/overnight_runs/latest_overnight.log <==\n"
        "[2026-10-26 23:10:00] [STAGE 5A] Running Core Strategy Unit Tests\n"
        "1876 passed in 14.2s\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=fake_ssh_out, stderr=""),
    )
    dev_prog = mon.check_dev_backtest_progress()
    assert dev_prog["running"] is True
    assert dev_prog["current_step"] == "[2026-10-26 23:10:00] [STAGE 5A] Running Core Strategy Unit Tests"
    assert dev_prog["last_log_line"] == "1876 passed in 14.2s"

    # 5. verify_tools: portfolio_snapshot.run_alerts, pnl_report.binance_symbol, and watch_trade_decisions.position
    alert_state_file = tmp_path / "alert_state.json"
    alert_state_file.write_text(
        json.dumps({"Kraken:HYPEUSD": {"dd": -12.0, "ts": "2099-01-01T00:00:00+00:00"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(portfolio_snapshot, "ALERT_STATE", str(alert_state_file))
    alerts_Sent = []
    monkeypatch.setattr(portfolio_snapshot, "_ntfy", lambda title, body: alerts_Sent.append((title, body)))
    fired = portfolio_snapshot.run_alerts([
        "not_a_dict",
        {"engine": "Binance:TAOUSDC", "qty": True, "dd_pct": -50.0},
        {"engine": "Kraken:HYPEUSD", "qty": 2.0, "dd_pct": -25.0, "unrealized": -50.0, "cost": 200.0},
    ])
    assert len(alerts_Sent) == 1
    assert fired == ["Kraken:HYPEUSD DD -25.0%"]

    monkeypatch.setattr(
        pnl_report.ao,
        "get_filled_orders",
        lambda side, sym, st: [
            "bad",
            {"side": "BUY", "price": "200.0", "qty": "1.5", "time": 1000},
            {"side": "BUY", "price": "-5.0", "qty": "1.0", "time": 1500},
            {"side": "SELL", "price": "220.0", "qty": "0.5", "time": 2000},
        ],
    )
    monkeypatch.setattr(pnl_report.bapi, "get_current_price", lambda sym: "230.0")
    res_pnl = pnl_report.binance_symbol("TAOUSDC")
    assert res_pnl["realized"] == 10.0
    assert res_pnl["pos"] == 1.0
    assert res_pnl["unreal"] == 30.0

    raw_orders = [
        {"executedQty": "2.0", "cummulativeQuoteQty": "500.0", "time": 1000},
        {"executedQty": "1.0", "price": "280.0", "updateTime": 2000},
    ]
    monkeypatch.setattr(
        watch_trade_decisions.apiorders,
        "get_trade_orders",
        lambda side, sym, maxage: raw_orders if side == "BUY" else [],
    )
    avg_w, net_w, sells_w, last_w = watch_trade_decisions.position("TAOUSDC")
    assert round(avg_w, 2) == 260.0
    assert net_w == 3.0
    assert last_w == 280.0


def test_vs_btc_relative_telemetry_and_capital_rotation_prompt(temp_env):
    workspace, cache_dir, _ = temp_env
    from orchestratorOS.supervisor.autonomous_ai_reconciler import (
        AutonomousAIReconciler,
        TelemetrySensorSnapshot,
    )
    from orchestratorOS.agy.topics import topic_spec
    from intelligence.sentiment.guards.high_stake_guard import (
        compute_24h_price_metrics,
        compute_cross_btc_metrics,
        load_symbol_price_series,
    )

    # 1. compute_24h_price_metrics computes 1m/3m/5m/6h/12h/24h changes, 15m tape, and 24h range geometry
    now_ms = 1_700_000_000_000.0
    series_24h = [
        [now_ms - 24 * 3600 * 1000, 100.0],
        [now_ms - 12 * 3600 * 1000, 102.0],
        [now_ms - 6 * 3600 * 1000, 104.0],
        [now_ms - 3 * 3600 * 1000, 105.0],
        [now_ms - 1 * 3600 * 1000, 108.0],
        [now_ms - 15 * 60 * 1000, 112.0],
        [now_ms - 5 * 60 * 1000, 111.0],
        [now_ms - 3 * 60 * 1000, 109.5],
        [now_ms - 1 * 60 * 1000, 109.8],
        [now_ms, 110.0],
    ]
    m24 = compute_24h_price_metrics(series_24h)
    assert m24.get("chg_24h_pct") == 10.0
    assert m24.get("chg_12h_pct") == 7.84
    assert m24.get("chg_6h_pct") == 5.77
    assert m24.get("chg_1h_pct") == 1.85
    assert m24.get("chg_5m_pct") == -0.9
    assert m24.get("chg_3m_pct") == 0.46
    assert m24.get("chg_1m_pct") == 0.18
    assert m24.get("dd_24h_high_pct") == -1.79
    assert m24.get("bounce_24h_low_pct") == 10.0
    assert m24.get("span_24h_pct") == 12.0
    assert m24.get("pos_24h_pct") == 83.3
    assert isinstance(m24.get("recent_tape_15m"), list) and len(m24["recent_tape_15m"]) >= 4

    # 1A. load_symbol_price_series reads cache_365_price_<SYM>.jsonl and overlays cache_current_price.json
    arch_365 = os.path.join(cache_dir, "cache_365_price_SOLUSDC.jsonl")
    with open(arch_365, "w", encoding="utf-8") as f:
        for h_back in (48, 24, 12, 6, 3, 1):
            f.write(json.dumps({"s": "SOLUSDC", "i": [now_ms - h_back * 3600_000, 150.0 + (48 - h_back) * 0.5]}) + "\n")
    cur_items = {
        "SOLUSDC": [
            [now_ms - 180_000, 173.0],
            [now_ms - 60_000, 173.5],
            [now_ms, 174.0],
        ]
    }
    sol_24h, sol_7d = load_symbol_price_series(cache_dir, "SOLUSDC", current_price_items=cur_items)
    assert len(sol_24h) >= 7
    assert len(sol_7d) >= 6
    sol_m24 = compute_24h_price_metrics(sol_24h)
    assert "chg_24h_pct" in sol_m24 and "chg_1m_pct" in sol_m24

    # Series shorter than 4h omits 6h/12h/24h
    series_short = [
        [now_ms - 3 * 3600 * 1000, 100.0],
        [now_ms, 105.0],
    ]
    m_short = compute_24h_price_metrics(series_short)
    assert "chg_24h_pct" not in m_short
    assert "chg_12h_pct" not in m_short
    assert "chg_6h_pct" not in m_short

    # 1B. compute_cross_btc_metrics computes ALT/BTC 24h range position, beta, and correlation
    btc_pts = [[now_ms - (10 - i) * 300_000, 80000.0 * (1.0 + 0.002 * i)] for i in range(11)]
    alt_pts = [[now_ms - (10 - i) * 300_000, 250.0 * (1.0 + 0.004 * i)] for i in range(11)]
    cross = compute_cross_btc_metrics(alt_pts, btc_pts)
    assert cross.get("alt_btc_pos_24h_pct") == 100.0
    assert cross.get("beta_btc_24h") is not None and cross["beta_btc_24h"] > 1.5
    assert cross.get("corr_btc_24h") is not None and cross["corr_btc_24h"] > 0.95

    # 2. Test synthesize_and_diagnose prompt generation with multi-horizon vs_btc, alt_btc_24h, and zero prescriptive rules
    captured_prompts = []

    def mock_query(prompt, **kwargs):
        captured_prompts.append(prompt)
        return json.dumps({
            "summary": "Market rotation test",
            "market_trading_decisions": [{"decision": "HOLD", "symbol": "TAOUSDC", "venue": "binance", "confidence": 0.9}],
        })

    auditor = MagicMock()
    auditor.query_json.side_effect = mock_query

    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir=cache_dir,
        logs_dirs=[],
        gemini_auditor=auditor,
        mode="shadow",
    )

    sensors = TelemetrySensorSnapshot(
        prices_and_trends={
            "BTCUSDC": {
                "price": 85000.0,
                "trend": 1,
                "growth_coefficient": 0.02,
                "chg_15m_pct": 0.5,
                "chg_30m_pct": 0.8,
                "chg_1h_pct": 2.0,
                "chg_3h_pct": 3.0,
                "chg_6h_pct": 3.5,
                "chg_12h_pct": 4.2,
                "chg_24h_pct": 5.0,
                "chg_7d_pct": 6.0,
                "low_24h": 80000.0,
                "high_24h": 86000.0,
                "low_7d": 78000.0,
                "high_7d": 86000.0,
            },
            "TAOUSDC": {
                "price": 260.0,
                "trend": 1,
                "growth_coefficient": 0.01,
                "chg_15m_pct": 0.2,
                "chg_30m_pct": 0.4,
                "chg_1h_pct": 1.0,
                "chg_3h_pct": 2.5,
                "chg_6h_pct": 4.5,
                "chg_12h_pct": 6.2,
                "chg_24h_pct": 8.0,
                "chg_7d_pct": 18.0,
                "low_24h": 240.0,
                "high_24h": 270.0,
                "low_7d": 215.0,
                "high_7d": 270.0,
                "alt_btc_pos_24h_pct": 74.5,
                "alt_btc_dd_24h_high_pct": -1.2,
                "alt_btc_bounce_24h_low_pct": 3.4,
                "beta_btc_24h": 1.42,
                "corr_btc_24h": 0.65,
            },
            "AAPL_US_EQ": {
                "price": 230.0,
                "trend": 0,
                "growth_coefficient": 0.0,
                "chg_15m_pct": 0.1,
                "chg_30m_pct": 0.2,
                "chg_1h_pct": 0.5,
                "chg_3h_pct": 0.6,
                "chg_24h_pct": 1.0,
                "low_24h": 228.0,
                "high_24h": 232.0,
            },
            "XBTUSD": {
                "price": 85010.0,
                "trend": 1,
                "growth_coefficient": 0.02,
                "chg_15m_pct": 0.5,
                "chg_1h_pct": 2.0,
                "chg_24h_pct": 5.0,
            },
        }
    )

    trade_spec = topic_spec("auto_trade_executor")
    reconciler.synthesize_and_diagnose(sensors, trade_spec, topic_key="auto_cryptotrade_executor")
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    # Verify full multi-horizon vs_btc and raw alt_btc_24h range geometry on TAOUSDC (without pre-calculated beta/corr)
    assert "vs_btc(15m=-0.30%, 1h=-1.00%, 3h=-0.50%, 6h=+1.00%, 12h=+2.00%, 24h=+3.00%, 7d=+12.00%)" in prompt
    assert "alt_btc_24h(pos=74.5%, dd_high=-1.20%, bounce_low=+3.40%)" in prompt
    assert "beta=" not in prompt
    assert "corr=" not in prompt
    assert "dd_high=-3.70%, bounce_low=+8.33%, span=12.5%" in prompt
    # Verify BTC and XBT do not have vs_btc
    btc_line = [ln for ln in prompt.splitlines() if "BTCUSDC" in ln][0]
    assert "vs_btc" not in btc_line
    xbt_line = [ln for ln in prompt.splitlines() if "XBTUSD" in ln][0]
    assert "vs_btc" not in xbt_line
    # Verify equities do not have vs_btc
    eq_line = [ln for ln in prompt.splitlines() if "AAPL_US_EQ" in ln][0]
    assert "vs_btc" not in eq_line
    # Verify prescriptive rotation clichés and pre-judged BULL/BEAR trend labels were removed
    assert "INTER-MARKET CAPITAL ROTATION PRINCIPLE" not in prompt
    assert "Phase 1 (BTC Breakout" not in prompt
    assert "Trend=BULL" not in prompt
    assert "Trend=BEAR" not in prompt
    assert "Growth=" not in prompt
    assert "[T=" not in prompt


def test_xbt_alias_7d_archive_with_24h_present_and_cross_venue_inheritance(tmp_path):
    """Verifies XBT<->BTC alias resolution, 7d archive extraction when 24h cache is full,
    cross-venue trends_data['items'] inheritance, and gateway price fallback to cache_24_price.
    """
    from types import SimpleNamespace
    from cachepaths import cache_symbol_variants, resolve_binance_lookup_symbol
    from intelligence.sentiment.guards.high_stake_guard import load_symbol_price_series
    from orchestratorOS.supervisor.autonomous_ai_reconciler import AutonomousAIReconciler
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway

    assert "BTCUSDC" in cache_symbol_variants("XBTUSD")
    assert "XBTUSD" in cache_symbol_variants("BTCUSDC")
    assert resolve_binance_lookup_symbol("XBTUSD") == "BTCUSDT"

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    now_ms = 1_760_000_000_000

    # 1. Write full 24h series in cache_24_price_BTCUSDC.json AND 7d history in cache_365_price_BTCUSDC.jsonl
    pts_24h = [[now_ms - (24 - i) * 3600 * 1000, 80000.0 + i * 200.0] for i in range(25)]
    (cache_dir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": pts_24h}}), encoding="utf-8"
    )
    arch_lines = [
        json.dumps({"s": "OTHERCOIN", "i": [int((now_ms - 6 * 86400 * 1000) / 1000), 999999.0]}),
        json.dumps({"s": "BTCUSDC", "i": [int((now_ms - 6 * 86400 * 1000) / 1000), 75000.0]}),
        json.dumps({"s": "BTCUSDC", "i": [int((now_ms - 3 * 86400 * 1000) / 1000), 78000.0]}),
        json.dumps({"s": "BTCUSDC", "i": [int(now_ms / 1000), 84800.0]}),
    ]
    (cache_dir / "cache_365_price_BTCUSDC.jsonl").write_text("\n".join(arch_lines) + "\n", encoding="utf-8")

    # Single-tick current price item for BTCUSDC, looked up via XBTUSD
    s24, v7d = load_symbol_price_series(
        str(cache_dir), "XBTUSD", current_price_items={"BTCUSDC": [[now_ms + 5000, 85000.0]]}
    )
    assert len(s24) >= 25
    assert s24[-1][1] == 85000.0
    # Ensure OTHERCOIN row was filtered out and 7d values end with latest live price
    assert 999999.0 not in v7d
    assert v7d[0] == 75000.0
    assert v7d[-1] == 85000.0

    # 2. Verify harvest_sensors_telemetry inherits slopes from trends_data['items'] and microstructure for XBTUSD
    (cache_dir / "cache_short_trend.json").write_text(
        json.dumps({
            "items": {
                "BTCUSDC": {
                    "current_price": None,
                    "final_trend": 1,
                    "slope_full": 0.0012,
                    "gradient_recent_fast": 0.0025,
                    "epsilon": 0.0005,
                    "vol_1h_pct": 1.4,
                    "dynamic": {
                        "222": {"slope_full": 0.003, "final_trend": 1},
                        "900": {"slope_full": 0.002, "final_trend": 1},
                        "9000": {"slope_full": 0.001, "final_trend": 1},
                    },
                },
                "XBTUSD": {"current_price": None},
            }
        }),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps({"imbalance_ratio": 0.42, "bid_depth_usd": 500000, "ask_depth_usd": 250000}),
        encoding="utf-8",
    )
    kraken_dir = tmp_path / "kraken"
    kraken_dir.mkdir(parents=True, exist_ok=True)
    (kraken_dir / ".state_XBTUSD.json").write_text(
        json.dumps({"qty": 0.01, "cost": 800.0, "entry_price": 80000.0}),
        encoding="utf-8",
    )

    reconciler = AutonomousAIReconciler(workspace_dir=str(tmp_path), mode="shadow", notify_enabled=False)
    snap = reconciler.harvest_sensors_telemetry(include_fleet=False, asset_scope="crypto")
    assert snap.prices_and_trends["BTCUSDC"]["price"] == 84800.0
    assert snap.prices_and_trends["XBTUSD"]["price"] == 84800.0
    assert snap.prices_and_trends["XBTUSD"]["slope_15m"] == 0.002
    assert snap.prices_and_trends["XBTUSD"]["chg_7d_pct"] == round(((84800.0 - 75000.0) / 75000.0) * 100.0, 2)
    assert snap.microstructure["XBTUSD"]["ob_imbalance"] == 0.42
    assert snap.portfolio_summary["kraken_cycles"]["XBTUSD"]["dd_pct"] == 6.0

    # 3. Verify gateway resolves BTCUSD -> XBTUSD on Kraken and falls back to cache_24_price when cache_current_price is absent
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path), mode="shadow", notify_enabled=False)
    fake_kraken_xbt = SimpleNamespace(
        name="KRAKEN_XBT",
        provider_name="kraken",
        symbol="XBTUSD",
        base="XBT",
        quote="USD",
        enabled=True,
        _provider=SimpleNamespace(),
    )
    resolved = gw.resolve_instrument("BTCUSD", venue="kraken", instruments_map={"KRAKEN_XBT": fake_kraken_xbt})
    assert resolved is fake_kraken_xbt
    assert gw._get_live_price(fake_kraken_xbt) == 84800.0


def test_kraken_tradable_pairs_xstocks_and_hyperliquid_long_state_symmetry(tmp_path):
    """Verify:
    1. KRAKEN_TRADABLE_PAIRS=HYPEUSD restricts Kraken tradable pairs to HYPEUSD (+ listed xStocks)
       while keeping KRAKEN_CACHE_PAIRS=HYPEUSD,TAOUSD,ADAUSD for reference data collection.
    2. Hyperliquid .state_HYPE_long.json is harvested into hyperliquid_cycles['HYPEUSDC'] with
       last_buy_price from last_open_price and resting_orders from orders.
    3. Kraken xStock holdings (SPCXx.T) and allocation parameters are harvested into kraken_xstocks
       and exposed in both crypto and equities (t212) scopes, linked with SPCX_US_EQ.
    """
    from cachepaths import cache_symbol_variants, canonical_base_asset, normalize_cache_base_asset, split_symbol_assets
    from orchestratorTrade.autonomous_intent_gateway import _discover_kraken_cache_pairs

    assert normalize_cache_base_asset("SPCXx.T") == "SPCX"
    assert canonical_base_asset("XBTUSD") == "BTC"
    assert canonical_base_asset("XXBTZUSD") == "BTC"
    assert canonical_base_asset("BTCUSDC") == "BTC"
    assert canonical_base_asset("HYPEUSDC") == "HYPE"
    assert canonical_base_asset("SPCXx.T") == "SPCX"
    assert canonical_base_asset("NVDA_US_EQ") == "NVDA"
    assert split_symbol_assets("XXBTZUSD") == ("XBT", "USD")
    assert split_symbol_assets("NVDA_US_EQ") == ("NVDA", "USD")
    assert split_symbol_assets("SPCXx.TUSD") == ("SPCX", "USD")
    assert "SPCX_US_EQ" in cache_symbol_variants("SPCXUSD")
    assert "SPCXUSD" in cache_symbol_variants("SPCX_US_EQ")

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    kraken_dir = tmp_path / "kraken"
    kraken_dir.mkdir(parents=True, exist_ok=True)
    hl_dir = tmp_path / "hyperliquid"
    hl_dir.mkdir(parents=True, exist_ok=True)
    t212_dir = tmp_path / "212trading"
    t212_dir.mkdir(parents=True, exist_ok=True)

    (kraken_dir / "config.env").write_text(
        "KRAKEN_PAIR=HYPEUSD\n"
        "KRAKEN_CACHE_PAIRS=HYPEUSD,TAOUSD,ADAUSD\n"
        "KRAKEN_TRADABLE_PAIRS=HYPEUSD\n"
        "XSTOCK_REGEX=SPCX|SPACEX\n"
        "XSTOCK_ALLOC_PRICE=141.75\n"
        "XSTOCK_TP_ALERT_PCT=41\n"
        "XSTOCK_TP2_ALERT_PCT=81\n"
        "XSTOCK_SL_ALERT_PCT=12\n"
        "XSTOCK_YAHOO=SPCX\n",
        encoding="utf-8",
    )
    (kraken_dir / "xstock_state.json").write_text(
        json.dumps({
            "known_assets": {"SPCXx.T": 15.74655, "AAPLx.T": 2e-06, "USDC": 2020.0},
            "allocated": {"asset": "SPCXx.T", "qty": 4.2786, "ts": 1781276702.0},
            "pair": None,
        }),
        encoding="utf-8",
    )
    (hl_dir / ".state_HYPE_long.json").write_text(
        json.dumps({
            "qty": 1.66,
            "cost": 98.027928,
            "entry_price": 59.052968,
            "dca_buys": 0,
            "last_open_price": 59.052968,
            "realized_pnl": 18.45,
            "tp_sold": [1, 2],
            "orders": [{"oid": 359202348244, "side": "sell", "px": 59.347, "sz": 1.66, "role": "close"}],
        }),
        encoding="utf-8",
    )
    (hl_dir / "signal.json").write_text(
        json.dumps({"regime": "BULLISH_FLOW", "score": 0.78}),
        encoding="utf-8",
    )
    (hl_dir / ".state_dn_HYPE.json").write_text(
        json.dumps({"qty": 999.0, "cost": 99999.0}),
        encoding="utf-8",
    )
    (t212_dir / ".state_SPCX_US_EQ.json").write_text(
        json.dumps({
            "qty": 2.442965,
            "cost_usd": 386.54,
            "entry_price": 158.2257,
            "last_buy_price": 158.2257,
            "dca_buys": 0,
        }),
        encoding="utf-8",
    )
    (t212_dir / ".state_NVDA_US_EQ.json").write_text(
        json.dumps({
            "qty": 3.0,
            "cost_usd": 360.0,
            "entry_price": 120.0,
            "last_buy_price": 120.0,
            "dca_buys": 1,
        }),
        encoding="utf-8",
    )
    now_ms = int(time.time() * 1000)
    (cache_dir / "cache_current_price.json").write_text(
        json.dumps({
            "items": {
                "BTCUSDC": 85000.0,
                "TAOUSDC": 340.0,
                "HYPEUSD": 62.0,
                "HYPEUSDC": 62.062,
                "SPCX_US_EQ": 170.1,
                "NVDA_US_EQ": 132.0,
            }
        }),
        encoding="utf-8",
    )
    for sym_s, base_px, pct_step in (
        ("BTCUSDC", 82000.0, 0.0015),
        ("TAOUSDC", 310.0, 0.0030),
        ("HYPEUSDC", 58.0, 0.0025),
        ("NVDA_US_EQ", 125.0, 0.0020),
        ("SPCX_US_EQ", 162.0, 0.0010),
    ):
        pts = [
            [now_ms - (24 - i) * 3_600_000, round(base_px * (1.0 + i * pct_step), 4)]
            for i in range(25)
        ]
        (cache_dir / f"cache_24_price_{sym_s}.json").write_text(
            json.dumps({"items": {sym_s: pts}}),
            encoding="utf-8",
        )
    (cache_dir / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps({"imbalance_ratio": 0.61, "bid_depth_usd": 120000, "ask_depth_usd": 78000}),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_ADAUSD.json").write_text(
        json.dumps({"imbalance_ratio": 0.44, "bid_depth_usd": 90000, "ask_depth_usd": 110000}),
        encoding="utf-8",
    )

    assert _discover_kraken_cache_pairs(str(tmp_path)) == [("HYPEUSD", "HYPE", "USD")]

    reconciler = AutonomousAIReconciler(workspace_dir=str(tmp_path), mode="shadow", notify_enabled=False)
    snap_crypto = reconciler.harvest_sensors_telemetry(include_fleet=False, asset_scope="crypto")
    hl_cyc = snap_crypto.portfolio_summary.get("hyperliquid_cycles", {})
    assert "HYPEUSDC" in hl_cyc
    assert "DN_HYPEUSDC" not in hl_cyc
    assert hl_cyc["HYPEUSDC"]["qty"] == 1.66
    assert hl_cyc["HYPEUSDC"]["last_buy_price"] == 59.052968
    assert hl_cyc["HYPEUSDC"]["realized_pnl_usd"] == 18.45
    assert hl_cyc["HYPEUSDC"]["tp_sold_levels"] == [1, 2]
    assert hl_cyc["HYPEUSDC"]["resting_orders"][0]["px"] == 59.347
    assert "hyperliquid_signal" not in snap_crypto.portfolio_summary

    # Reference microstructure for ADAUSD is harvested even when ADAUSD is not traded
    assert snap_crypto.microstructure["ADAUSD"]["ob_imbalance"] == 0.44
    assert snap_crypto.microstructure["HYPEUSDC"]["ob_imbalance"] == 0.61

    xs_crypto = snap_crypto.portfolio_summary.get("kraken_xstocks", {})
    assert xs_crypto["underlying"] == "SPCX"
    assert xs_crypto["t212_sibling"] == "SPCX_US_EQ"
    assert xs_crypto["holdings"]["SPCXx.T"] == 15.74655
    assert xs_crypto["alloc_price"] == 141.75
    assert xs_crypto["live_price"] == 170.1
    assert xs_crypto["pnl_vs_alloc_pct"] == 20.0
    assert xs_crypto["t212_sibling_cycle"]["qty"] == 2.442965

    # Verify cross_market_context provides raw multi-horizon telemetry without pre-judged correlations or sync labels
    cmc_crypto = snap_crypto.portfolio_summary.get("cross_market_context", {})
    assert "market_sync_pattern" not in cmc_crypto
    assert "cross_asset_correlations_24h" not in cmc_crypto
    assert "NVDA_US_EQ" in cmc_crypto.get("equity_bellwethers", {})
    assert cmc_crypto["equity_bellwethers"]["NVDA_US_EQ"]["t212_cycle"]["qty"] == 3.0
    assert "chg_24h_pct" in cmc_crypto["equity_bellwethers"]["NVDA_US_EQ"]

    snap_eq = reconciler.harvest_sensors_telemetry(include_fleet=False, asset_scope="equities")
    assert "kraken_xstocks" in snap_eq.portfolio_summary
    assert snap_eq.portfolio_summary["kraken_xstocks"]["holdings"]["SPCXx.T"] == 15.74655
    cmc_eq = snap_eq.portfolio_summary.get("cross_market_context", {})
    assert "BTC" in cmc_eq.get("crypto_bellwethers", {})
    assert "TAO" in cmc_eq.get("crypto_bellwethers", {})
    assert "chg_24h_pct" in cmc_eq["crypto_bellwethers"]["TAO"]
    assert "cross_asset_correlations_24h" not in cmc_eq

    # Verify prompt deduplicates HYPEUSDC + HYPEUSD in Sections 3 and 3B while showing sibling venue price & raw cross-market context
    captured_prompt = {}
    reconciler.gemini_auditor = GeminiClient(
        custom_runner=lambda prompt, model, timeout: (
            captured_prompt.update({"u": prompt}) or '{"summary":"ok","trade_intents":[]}'
        )
    )
    reconciler.synthesize_and_diagnose(snap_crypto, LogEventSummary(0, 0, 0), topic_key="auto_cryptotrade_executor")
    u_prompt = captured_prompt["u"]
    assert "* HYPEUSDC (sibling venue: HYPEUSD=$62.00 [-0.10%]): Price=$62.06" in u_prompt
    assert "\n  * HYPEUSD: Price=" not in u_prompt
    assert "* HYPEUSDC (HYPEUSD): ob_imbalance=0.61" in u_prompt
    assert "* ADAUSD: ob_imbalance=0.44" in u_prompt
    assert "realized_pnl=$18.45, tp_sold=[1, 2]" in u_prompt
    assert "READ-ONLY on Kraken — NOT API-tradable" in u_prompt
    assert "Cross-Market US Equities Telemetry (Trading212 Raw Prices, Multi-Horizon Deltas & Active Cycles)" in u_prompt
    assert "Realized 24h Cross-Asset Correlation" not in u_prompt
    assert "Hyperliquid Bot Signal Telemetry" not in u_prompt

    reconciler.synthesize_and_diagnose(snap_eq, LogEventSummary(0, 0, 0), topic_key="auto_sharetrade_executor")
    eq_prompt = captured_prompt["u"]
    assert "Cross-Market Live Crypto Telemetry (Raw 24/7 Prices, Multi-Horizon Deltas & Microstructure)" in eq_prompt
    assert "TAO decentralized AI proxy" not in eq_prompt


def test_technical_repair_ssot_normalization_and_manifest_unification(tmp_path):
    """Verify Single Source of Truth (SSOT) normalization across all auto_technical_repair targets and actions."""
    from orchestratorOS.lib.technical_repair_ssot import (
        ALL_DIAGNOSED_CATEGORIES,
        ALL_REPAIR_ACTIONS,
        build_system_admin_command,
        fleet_script_basenames,
        normalize_repair_action,
        parse_procs_manifest,
        resolve_manifest_process,
        resolve_patchable_target,
    )

    (tmp_path / "procs.conf").write_text(
        "rtrade.py|$ROOT|python rtrade.py|rtrade|cachedb/rtrade.heartbeat|180|fleet\n"
        "hyperliquid/hl_bot.py|$ROOT/hyperliquid|python hl_bot.py|HL-bot|cachedb/HL-bot.heartbeat|600|bot\n"
        "market_monitor/price_watcher.py|$ROOT|python market_monitor/price_watcher.py|price_watcher|cachedb/price_watcher.heartbeat|180|fleet\n",
        encoding="utf-8",
    )

    manifest = parse_procs_manifest(str(tmp_path), include_stdout_with_heartbeat_file=True)
    assert len(manifest) == 3
    assert fleet_script_basenames(str(tmp_path)) == {"rtrade.py", "price_watcher.py"}
    # Dedicated .heartbeat file also appends logs/<stem>.log when include_stdout_with_heartbeat_file=True
    assert any(p.endswith("logs/rtrade.log") for p in manifest[0]["hb_files"])

    # Alias/stem resolution for bots and modules
    assert resolve_manifest_process("hl_bot", str(tmp_path))["name"] == "HL-bot"
    assert resolve_manifest_process("hyperliquid/hl_bot.py", str(tmp_path))["name"] == "HL-bot"
    assert resolve_manifest_process("price_watcher.py", str(tmp_path))["name"] == "price_watcher"

    # Synonym & target alias normalization (RESTART_BOT + FLEET/wgpia0/dns/SYSTEM -> canonical actions)
    n_bot = normalize_repair_action({"action": "REBOOT_BOT", "bot": "hl_bot.py", "reason": "stale"}, str(tmp_path))
    assert n_bot["action"] == "RESTART_BOT"
    assert n_bot["symbol"] == "HL-bot"
    assert n_bot["venue"] == "fleet"
    assert n_bot["matched_process"]["clean_pattern"] == "hyperliquid/hl_bot.py"

    n_fleet = normalize_repair_action({"action": "RESTART_BOT", "symbol": "flota", "reason": "all wedged"}, str(tmp_path))
    assert n_fleet["action"] == "RESTART_FLEET"
    assert n_fleet["venue"] == "fleet"

    n_vpn = normalize_repair_action({"action": "RESTART_MODULE", "target": "wgpia0", "reason": "tunnel down"}, str(tmp_path))
    assert n_vpn["action"] == "RESTART_VPN"
    assert n_vpn["venue"] == "system"

    n_dns = normalize_repair_action({"action": "FIX_DNS", "target": "systemd-resolved"}, str(tmp_path))
    assert n_dns["action"] == "PIN_DNS"

    n_sys = normalize_repair_action({"action": "RESTART_BOT", "symbol": "intreg_sistemul"}, str(tmp_path))
    assert n_sys["action"] == "REBOOT_SYSTEM"
    assert n_sys["venue"] == "system"

    n_cancel = normalize_repair_action(
        {"action": "CANCEL_STALE_ORDER", "provider": "Trading212", "symbol": "SPCX_US_EQ", "orderId": 778899.0},
        str(tmp_path),
    )
    assert n_cancel["action"] == "CANCEL_STALE_ORDER"
    assert n_cancel["venue"] == "t212"
    assert n_cancel["order_id"] == "778899"

    # check_cache_coherence supports scalar + fetchtime, dict, and list currentprice entries + cross-venue variants
    import verify_tools.check_cache_coherence as ccc
    now_s = time.time()
    with (
        patch.object(ccc, "expected_symbols", return_value=["BTCUSDC", "HYPEUSD", "SPCX_US_EQ"]),
        patch.object(
            ccc,
            "load",
            side_effect=lambda name: (
                {
                    "items": {
                        "BTCUSDC": {"ts": now_s - 5, "current_price": 65000.0, "gradient_recent": 0.1},
                        "HYPEUSDC": {"ts": now_s - 5, "current_price": 42.5, "gradient_recent": 0.2},
                        "SPCX": {"ts": now_s - 5, "current_price": 12.3, "gradient_recent": 0.05},
                    }
                }
                if name == "cache_short_trend.json"
                else {
                    "fetchtime": {"BTCUSDC": now_s - 3, "HYPEUSDC": now_s - 4},
                    "items": {
                        "BTCUSDC": 65000.0,
                        "HYPEUSDC": {"price": 42.5, "ts": now_s - 4},
                        "SPCX": [[(now_s - 2) * 1000.0, 12.3]],
                    },
                }
            ),
        ),
    ):
        assert ccc.main() == 0

    for sys_act in ("RESTART_FLEET", "RESTART_VPN", "PIN_DNS", "REBOOT_SYSTEM"):
        cmd_desc = build_system_admin_command(sys_act, admin_bin="/nonexistent/trading-admin")
        assert cmd_desc is not None
        assert cmd_desc[0][:2] == ["sudo", "-n"]

    # Patch target boundary & secret file protection
    cfg_file = tmp_path / "order_guard.conf"
    cfg_file.write_text("supervisor_mode = enforce\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=1\n", encoding="utf-8")
    ok_path, err = resolve_patchable_target("order_guard.conf", str(tmp_path))
    assert ok_path == os.path.realpath(str(cfg_file)) and err is None
    bad_env, env_err = resolve_patchable_target(".env", str(tmp_path))
    assert bad_env is None and "protected secret" in env_err
    assert "REBOOT_SYSTEM" in ALL_REPAIR_ACTIONS
    assert "CONFIG_ERROR" in ALL_DIAGNOSED_CATEGORIES

    # Heartbeat freshness evaluation and fleet sensor status SSOT
    from orchestratorOS.lib.technical_repair_ssot import (
        evaluate_heartbeat_files_freshness,
        summarize_fleet_sensor_status,
    )
    now_t = 1_700_000_000.0
    hb1 = tmp_path / "hb1.heartbeat"
    hb1.write_text("ok", encoding="utf-8")
    os.utime(str(hb1), (now_t - 5.0, now_t - 5.0))
    hb2 = tmp_path / "hb2.heartbeat"
    hb2.write_text("ok", encoding="utf-8")
    os.utime(str(hb2), (now_t - 15.0, now_t - 15.0))

    is_fresh, age_s, max_stale, st_path = evaluate_heartbeat_files_freshness(
        [str(hb1), str(hb2)], 20.0, now_t
    )
    assert is_fresh is True
    assert age_s == 15.0

    is_fresh, _, max_stale, st_path = evaluate_heartbeat_files_freshness(
        [str(hb1), str(hb2)], 10.0, now_t
    )
    assert is_fresh is False
    assert max_stale == 15.0
    assert st_path == str(hb2)

    is_fresh, _, max_stale, st_path = evaluate_heartbeat_files_freshness(
        [str(hb1), str(tmp_path / "missing.heartbeat")], 20.0, now_t, fallback_start_ts=now_t - 30.0
    )
    assert is_fresh is False
    assert max_stale == 30.0

    (tmp_path / "geopolitical_eval.json").write_text(
        json.dumps({"threat_level": "ELEVATED", "risk_score": 0.65, "summary": "Middle East tensions"}),
        encoding="utf-8",
    )
    (tmp_path / "fear_greed_collect.json").write_text(json.dumps({"value": 28.0}), encoding="utf-8")
    sensors = summarize_fleet_sensor_status(str(tmp_path))
    assert sensors["geopolitical_threat"]["threat_level"] == "ELEVATED"
    assert sensors["geopolitical_threat"]["risk_score"] == 0.65
    assert sensors["fear_greed"] == 28.0



def test_extract_current_price_item_instrument_resolution_and_trailing_state_normalization(tmp_path):
    """Verify extract_current_price_item, canonical base priority in resolve_instrument, and trailing_state key normalization."""
    from orchestratorTrade.autonomous_intent_gateway import (
        AutonomousIntentGateway,
        extract_current_price_item,
    )
    from intelligence.sentiment.guards.high_stake_guard import (
        compute_24h_price_metrics,
        load_symbol_price_series,
    )

    # 1. extract_current_price_item handles scalar, 1-element list, [ts, px] series, and dicts
    assert extract_current_price_item(85123.5) == 85123.5
    assert extract_current_price_item([85123.5]) == 85123.5
    assert extract_current_price_item([[1700000000000, 84000.0], [1700000060000, 85500.0]]) == 85500.0
    assert extract_current_price_item({"price": "86000.25"}) == 86000.25
    assert extract_current_price_item({"c": 86100.0}) == 86100.0
    assert extract_current_price_item(True) is None
    assert extract_current_price_item(-5.0) is None
    assert extract_current_price_item([]) is None

    # 2. resolve_instrument prefers exact canonical base asset (ETH -> ETHUSDC) over prefix match (ETHFIUSDC)
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path), notify_enabled=False)
    inst_ethfi = MagicMock(symbol="ETHFIUSDC", provider_name="binance", base="ETHFI", quote="USDC")
    inst_eth = MagicMock(symbol="ETHUSDC", provider_name="binance", base="ETH", quote="USDC")
    instruments = {"ETHFIUSDC": inst_ethfi, "ETHUSDC": inst_eth}
    assert gw.resolve_instrument("ETH", "binance", instruments) is inst_eth
    assert gw.resolve_instrument("ETHFI", "binance", instruments) is inst_ethfi

    # 3. load_symbol_price_series appends scalar current_price_items entry onto 24h series
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    now_ms = 1_700_000_000_000.0
    (cache_dir / "cache_24_price_ETHUSDC.json").write_text(
        json.dumps({"items": [[now_ms - 3600_000, 3000.0], [now_ms - 60_000, 3050.0]]}),
        encoding="utf-8",
    )
    s24, _ = load_symbol_price_series(str(cache_dir), "ETHUSDC", current_price_items={"ETHUSDC": 3120.0})
    assert s24[-1][1] == 3120.0
    m24 = compute_24h_price_metrics(s24)
    assert m24["chg_1h_pct"] == 4.0

    # 4. Kraken trailing_state.json with non-USD quote or bare base resolves cleanly without TAOUSDCUSD corruption
    (tmp_path / "instruments.conf").write_text("", encoding="utf-8")
    krk_dir = tmp_path / "kraken"
    krk_dir.mkdir(exist_ok=True)
    (krk_dir / "trailing_state.json").write_text(
        json.dumps({"HYPE": 64.0, "TAOUSDC": 520.0}),
        encoding="utf-8",
    )
    (cache_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"HYPEUSD": 60.8, "TAOUSDC": 494.0}}),
        encoding="utf-8",
    )
    inst_hype = MagicMock(symbol="HYPEUSD", provider_name="kraken", base="HYPE", quote="USD", params={})
    inst_hype._provider.get_current_price.return_value = 60.8
    inst_hype._provider.free_balance.side_effect = lambda a: 5.0 if a == "HYPE" else 100.0
    inst_hype.open_orders.return_value = []

    inst_tao = MagicMock(symbol="TAOUSDC", provider_name="binance", base="TAO", quote="USDC", params={})
    inst_tao._provider.get_current_price.return_value = 494.0
    inst_tao._provider.free_balance.side_effect = lambda a: 2.0 if a == "TAO" else 100.0
    inst_tao.open_orders.return_value = []

    mock_gw = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    mock_gw.load_tradable_instruments.return_value = {"KRAKEN_HYPE": inst_hype, "TAOUSDC": inst_tao}

    reconciler = AutonomousAIReconciler(
        workspace_dir=str(tmp_path),
        cache_dir="cachedb",
        mode="shadow",
        notify_enabled=False,
        intent_gateway=mock_gw,
    )
    snap = reconciler.harvest_sensors_telemetry(include_fleet=False, asset_scope="crypto")
    pos_by_sym = {p["symbol"]: p for p in snap.portfolio_summary["tradable_positions"]}
    assert pos_by_sym["HYPEUSD"]["trailing_peak"] == 64.0
    assert pos_by_sym["HYPEUSD"]["dd_from_peak_pct"] == -5.0
    assert pos_by_sym["TAOUSDC"]["trailing_peak"] == 520.0
    assert pos_by_sym["TAOUSDC"]["dd_from_peak_pct"] == -5.0


def test_venue_alias_decision_dedup_kraken_pair_matching_and_cross_venue_fills(tmp_path):
    from cachepaths import normalize_venue_name
    from intelligence.sentiment.guards.high_stake_guard import extract_symbol_recent_fills
    from orchestratorOS.supervisor.autonomous_ai_reconciler import _normalize_trading_decisions
    from providers.kraken_provider import _kraken_pair_matches

    # 1. Venue alias normalization
    assert normalize_venue_name("trading212") == "t212"
    assert normalize_venue_name("TRADING_212") == "t212"
    assert normalize_venue_name("212") == "t212"
    assert normalize_venue_name("hl") == "hyperliquid"
    assert normalize_venue_name("Hyper_Liquid") == "hyperliquid"
    assert normalize_venue_name("krk") == "kraken"
    assert normalize_venue_name("bin") == "binance"

    # 2. _normalize_trading_decisions sorts by confidence before deduplicating canonical base + venue alias
    plan = {
        "trading_decisions": [
            {"venue": "hl", "symbol": "HYPE", "action": "BUY", "confidence": 0.88, "notional_usd": 40.0},
            {"venue": "hyperliquid", "symbol": "HYPEUSDC", "decision": "BUY", "confidence": 0.96, "notional_usd": 80.0},
            {"venue": "trading212", "symbol": "SPCX_US_EQ", "side": "SELL", "confidence": 0.91, "notional_usd": 50.0},
        ]
    }
    decs = _normalize_trading_decisions(plan)
    assert len(decs) == 2
    assert decs[0]["venue"] == "hyperliquid"
    assert decs[0]["symbol"] == "HYPEUSDC"
    assert decs[0]["decision"] == "BUY"
    assert decs[0]["confidence"] == 0.96
    assert decs[1]["venue"] == "t212"
    assert decs[1]["decision"] == "SELL"

    # 3. _kraken_pair_matches prevents substring collision (ETHUSD vs WETHUSD, BTCUSD vs WBTCUSD)
    assert _kraken_pair_matches("ETHUSD", "WETHUSD") is False
    assert _kraken_pair_matches("WETHUSD", "ETHUSD") is False
    assert _kraken_pair_matches("BTCUSD", "WBTCUSD") is False
    assert _kraken_pair_matches("HYPEUSD", "HYPE") is True
    assert _kraken_pair_matches("HYPE", "HYPEUSD") is True
    assert _kraken_pair_matches("DOGEUSD", "XXDGZUSD") is True
    assert _kraken_pair_matches("BTCUSD", "XXBTZUSD") is True
    assert _kraken_pair_matches("XTZUSD", "XTZUSD") is True
    assert _kraken_pair_matches("XTZUSD", "XTUSD") is False
    assert _kraken_pair_matches("CHZUSD", "CHUSD") is False
    assert _kraken_pair_matches("XAUTUSD", "AUTUSD") is False
    assert _kraken_pair_matches("XRPUSD", "XXRPZUSD") is True

    from cachepaths import canonical_base_asset, normalize_cache_base_asset
    assert canonical_base_asset("XXDGZUSD") == "DOGE"
    assert normalize_cache_base_asset("XXRPZUSD") == "XRP"
    assert normalize_cache_base_asset("XTZUSD") == "XTZ"
    assert normalize_cache_base_asset("CHZUSD") == "CHZ"
    assert normalize_cache_base_asset("XAUTUSD") == "XAUT"

    # 4. Cross-venue extract_symbol_recent_fills (Kraken + Hyperliquid)
    now_s = 1_700_000_000.0
    orders_list = [
        {
            "closetm": now_s - 600,
            "descr": {"type": "buy", "price": "40.0"},
            "vol_exec": "2.5",
            "cost": "100.0",
        },
        {
            "time": int((now_s - 300) * 1000),
            "dir": "Close Long",
            "px": "44.0",
            "sz": "2.5",
        },
    ]
    fills_summary = extract_symbol_recent_fills(orders_list)
    assert fills_summary is not None
    assert fills_summary["last_action_side"] == "SELL"
    assert fills_summary["last_buy"]["price"] == 40.0
    assert fills_summary["last_buy"]["qty"] == 2.5
    assert fills_summary["last_sell"]["price"] == 44.0
    assert fills_summary["last_sell"]["qty"] == 2.5

    # 5. Provider free_balance & resolve_assets symmetry (Kraken DOGE/xStocks, T212 short base, Hyperliquid symbol)
    from providers.kraken_provider import KrakenProvider
    from providers.quantity import resolve_assets
    from providers.t212_provider import T212Provider

    assert resolve_assets("SPCX_US_EQ") == ("SPCX_US_EQ", "USD")
    krk_p = KrakenProvider()
    krk_p._cli = MagicMock()
    krk_p._cli.balance.return_value = {"XXDG": "1500.0", "SPCXx.T": "15.75"}
    assert krk_p.free_balance("DOGE") == 1500.0
    assert krk_p.free_balance("SPCX") == 15.75
    assert krk_p.free_balance("SPCXx.T") == 15.75

    t212_p = T212Provider(client=MagicMock(get_portfolio=lambda: [{"ticker": "SPCX_US_EQ", "quantity": 4.25}]))
    assert t212_p.free_balance("SPCX") == 4.25
    assert t212_p.free_balance("SPCX_US_EQ") == 4.25

    # 6. Venue-isolated recent_orders prevents Kraken HYPEUSD from polluting Hyperliquid HYPEUSDC cost basis
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (tmp_path / "instruments.conf").write_text("", encoding="utf-8")
    hl_dir = tmp_path / "hyperliquid"
    hl_dir.mkdir(exist_ok=True)
    (hl_dir / ".state_HYPE.json").write_text(
        json.dumps({"qty": 10.0, "cost": 380.0, "entry_price": 38.0, "last_buy_price": 38.0}),
        encoding="utf-8",
    )
    (cache_dir / "cache_trade_kraken.json").write_text(
        json.dumps({
            "items": {
                "HYPEUSD": [{"closetm": now_s - 3600, "descr": {"type": "buy", "price": "25.0"}, "vol_exec": "2.0", "cost": "50.0"}]
            }
        }),
        encoding="utf-8",
    )
    hl_inst = MagicMock(symbol="HYPEUSDC", provider_name="hyperliquid", base="HYPE", quote="USDC", params={})
    hl_inst._provider.get_current_price.return_value = 41.8
    hl_inst._provider.free_balance.side_effect = lambda a: 10.0 if a == "HYPE" else 200.0
    hl_inst._provider.get_orders.return_value = []
    hl_inst.open_orders.return_value = []

    krk_inst = MagicMock(symbol="HYPEUSD", provider_name="kraken", base="HYPE", quote="USD", params={})
    krk_inst._provider.get_current_price.return_value = 41.8
    krk_inst._provider.free_balance.side_effect = lambda a: 2.0 if a == "HYPE" else 100.0
    krk_inst.open_orders.return_value = []

    mock_gw = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    mock_gw.load_tradable_instruments.return_value = {"HYPERLIQUID_HYPE": hl_inst, "KRAKEN_HYPE": krk_inst}
    reconciler = AutonomousAIReconciler(
        workspace_dir=str(tmp_path),
        cache_dir="cachedb",
        mode="shadow",
        notify_enabled=False,
        intent_gateway=mock_gw,
    )
    snap = reconciler.harvest_sensors_telemetry(include_fleet=False, asset_scope="crypto")
    pos_by_venue_sym = {(p["venue"], p["symbol"]): p for p in snap.portfolio_summary["tradable_positions"]}
    assert pos_by_venue_sym[("hyperliquid", "HYPEUSDC")]["active_avg_buy_px"] == 38.0
    assert pos_by_venue_sym[("hyperliquid", "HYPEUSDC")]["last_buy_px"] == 38.0
    assert pos_by_venue_sym[("kraken", "HYPEUSD")]["active_avg_buy_px"] == 25.0
    assert pos_by_venue_sym[("kraken", "HYPEUSD")]["last_buy_px"] == 25.0

    # 7. alt_btc_24h range geometry is populated without beta/corr, and submit() honors notional_usd + TOPIC_ALIASES
    now_ms = int(now_s * 1000)
    (cache_dir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": [[now_ms - i * 300_000, 80000.0 + i * 50.0] for i in range(10, -1, -1)]}),
        encoding="utf-8",
    )
    (cache_dir / "cache_24_price_HYPEUSDC.json").write_text(
        json.dumps({"items": [[now_ms - i * 300_000, 40.0 + i * 0.2] for i in range(10, -1, -1)]}),
        encoding="utf-8",
    )
    snap2 = reconciler.harvest_sensors_telemetry(include_fleet=False, asset_scope="crypto")
    hype_pinfo = snap2.prices_and_trends["HYPEUSDC"]
    assert hype_pinfo.get("alt_btc_pos_24h_pct") is not None
    assert "beta_btc_24h" not in hype_pinfo
    assert "corr_btc_24h" not in hype_pinfo

    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path), mode="shadow", notify_enabled=False)
    submitted = gw.submit(
        {"action": "BUY", "symbol": "HYPEUSDC", "venue": "hl", "confidence": 0.94, "notional_usd": 55.0},
        source="auto_trade_executor",
        max_mode="shadow",
        instruments_map={"HYPERLIQUID_HYPE": hl_inst},
    )
    assert submitted is not None
    assert submitted["suggested_notional_usd"] == 55.0
    assert submitted["venue"] == "hyperliquid"
    assert submitted["status"] == "OBSERVED_SHADOW"

    # 8. ExecutionAudit, MarketApi.provider_by_name, resolve_assets, and Kraken balance candidates normalize venue/asset aliases
    from providers.execution_audit import intent_client_order_id, new_intent_id
    from providers.kraken_provider import _kraken_balance_asset_candidates
    from providers.market_api import api as market_api
    from providers.quantity import resolve_assets
    from intelligence.internal.guards.guard_decision import (
        clean_guard_symbol_key,
        normalize_guard_side,
        safe_guard_float,
        safe_pos_guard_float,
        snap_attr,
    )

    hl_intent = new_intent_id("hl", "HYPEUSDC", "buy")
    assert hl_intent.startswith("hyperliquid-HYPEUSDC-buy-")
    assert intent_client_order_id("hl", hl_intent).startswith("0x")
    assert intent_client_order_id("bin", hl_intent).startswith("SD_")
    assert len(intent_client_order_id("krk", hl_intent)) == 32
    assert intent_client_order_id("trading212", hl_intent) is None

    for alias, expected_name in (("hl", "Hyperliquid"), ("krk", "Kraken"), ("bin", "Binance"), ("trading212", "T212"), ("212", "T212")):
        prov_obj = market_api.provider_by_name(alias)
        if prov_obj is not None:
            assert prov_obj.name == expected_name

    assert resolve_assets("XXBTZUSD") == ("XXBT", "USD")
    assert resolve_assets("XXDGZUSD") == ("XXDG", "USD")
    assert resolve_assets("XETHZUSD") == ("XETH", "USD")
    assert resolve_assets("SPCXx.T") == ("SPCXx.T", "USD")
    assert "ETH" in _kraken_balance_asset_candidates("XETH")
    assert "USD" in _kraken_balance_asset_candidates("ZUSD")
    assert "SPCX" in _kraken_balance_asset_candidates("SPCXx.T")

    assert normalize_guard_side("long") == "BUY"
    assert normalize_guard_side("short") == "SELL"
    assert normalize_guard_side("INVALID") == ""
    assert normalize_guard_side("HOLD") == ""
    from intelligence.macro_sentiment_ssot import normalize_order_side
    assert normalize_order_side("INVALID") == ""
    assert normalize_order_side("HOLD") == ""
    assert clean_guard_symbol_key("hype/usdc") == "HYPEUSDC"
    assert snap_attr({"x": 42}, "x") == 42
    assert safe_guard_float("3.5", 0.0) == 3.5
    assert safe_pos_guard_float("-1.0", 2.0) == 2.0

    # 9. instrument_registry, order_retry, rtrade_pair, and 212trading normalize venue, side, and T212 ticker aliases
    from instrument_registry import select_instruments
    from order_retry import _same_deferred_stream
    from strategies.rtrade_pair import OrderTicket, anchored_exit_price

    assert set(select_instruments("trading212", only_enabled=False)) == set(
        select_instruments("t212", only_enabled=False)
    )
    assert set(select_instruments("hl", only_enabled=False)) == set(
        select_instruments("hyperliquid", only_enabled=False)
    )
    assert _same_deferred_stream(
        {
            "lifecycle": "submit_pending",
            "last_failure_reason": "trend_deferred",
            "provider_name": "trading212",
            "symbol": "SPCX_US_EQ",
            "side": "BUY",
            "kind": "ENTRY",
        },
        {
            "lifecycle": "submit_pending",
            "last_failure_reason": "trend_deferred",
            "provider_name": "t212",
            "symbol": "SPCX_US_EQ",
            "side": "BUY",
            "kind": "ENTRY",
        },
    )
    t_long = OrderTicket(order_id="oid-long", side="long", price=100.0, qty=1.0)
    t_short = OrderTicket(order_id="oid-short", side="short", price=100.0, qty=1.0)
    assert t_long.side == "BUY"
    assert t_short.side == "SELL"
    assert anchored_exit_price("short", 100.0, 90.0, 0.01, 0.01) == anchored_exit_price(
        "SELL", 100.0, 90.0, 0.01, 0.01
    )


def test_swing_wave_metrics_fast_trends_execution_audit_and_unbiased_autonomy(tmp_path):
    from orchestratorOS.supervisor.autonomous_ai_reconciler import (
        AutonomousAIReconciler,
        GeminiClient,
        _compute_swing_wave_metrics,
    )

    # 1. Pure objective ZigZag swing wave extraction on rally (+8.68%) and subsequent drop (-7.82%)
    series_rally = [
        [1000, 0.1750],
        [2000, 0.1636],  # swing low (-6.51%)
        [3000, 0.1700],
        [4000, 0.1778],  # +8.68% rally peak
    ]
    m_up = _compute_swing_wave_metrics(series_rally)
    assert m_up["swing_low_px"] == 0.1636
    assert m_up["swing_high_px"] == 0.1778
    assert m_up["rally_from_swing_low_pct"] == 8.68
    assert m_up["pullback_from_swing_high_pct"] == 0.0
    assert "swing_action_zone" not in m_up
    assert m_up["recent_swings_7d"] == ["-6.5%", "+8.7%"]

    series_drop = series_rally + [[5000, 0.1639]]  # -7.82% pullback from 0.1778
    m_dn = _compute_swing_wave_metrics(series_drop)
    assert m_dn["swing_high_px"] == 0.1778
    assert m_dn["pullback_from_swing_high_pct"] == -7.82
    assert "swing_action_zone" not in m_dn
    assert m_dn["recent_swings_7d"] == ["-6.5%", "+8.7%", "-7.8%"]

    # 2. End-to-end prompt formatting includes raw swing_wave, 90s fast trend slopes & g_coeff,
    #    execution audit log records, and unbiased autonomous freedom (with tool/API/log access)
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (tmp_path / "instruments.conf").write_text("", encoding="utf-8")
    (cache_dir / "cache_watcher_all_price.json").write_text(
        json.dumps({"items": {"ARBUSDC": series_rally}}),
        encoding="utf-8",
    )
    (cache_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"ARBUSDC": 0.1778}}),
        encoding="utf-8",
    )
    (cache_dir / "cache_short_trend.json").write_text(
        json.dumps({
            "items": {
                "ARBUSDC": {
                    "current_price": 0.1778,
                    "final_trend": 1,
                    "slope_full": 0.00045,
                    "gradient_recent": 0.0012,
                    "gradient_recent_fast": 0.0018,
                    "epsilon": 0.0004,
                    "vol_1h_pct": 0.85,
                    "adapt_reentry_pct": 1.70,
                    "adapt_dca_pct": 2.10,
                    "dynamic": {
                        "90": {"slope_full": 0.00092, "final_trend": 1, "growth_coefficient": 0.00081},
                        "222": {"slope_full": 0.00064, "final_trend": 1, "growth_coefficient": 0.00055},
                        "900": {"slope_full": 0.00045, "final_trend": 1, "growth_coefficient": 0.00038},
                        "9000": {"slope_full": 0.00021, "final_trend": 1, "growth_coefficient": 0.00019},
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    audit_dir = tmp_path / "logger" / "execution_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "execution_audit_2026-10-29.jsonl").write_text(
        json.dumps({
            "event": "submit_accepted",
            "venue": "Binance",
            "symbol": "ARBUSDC",
            "side": "sell",
            "qty": 500.0,
            "price": 0.1778,
            "ts_utc": "2026-10-29T12:00:00+00:00",
        }) + "\n",
        encoding="utf-8",
    )

    arb_inst = MagicMock(symbol="ARBUSDC", provider_name="binance", base="ARB", quote="USDC", params={})
    arb_inst._provider.get_current_price.return_value = 0.1778
    arb_inst._provider.free_balance.side_effect = lambda a: 5000.0 if a == "ARB" else 400.0
    arb_inst.open_orders.return_value = []

    mock_gw = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    mock_gw.load_tradable_instruments.return_value = {"ARBUSDC": arb_inst}

    captured = []
    auditor = GeminiClient(
        custom_runner=lambda prompt, model, timeout: (
            captured.append(prompt)
            or json.dumps({"summary": "ok", "market_trading_decisions": []})
        )
    )
    reconciler = AutonomousAIReconciler(
        workspace_dir=str(tmp_path),
        cache_dir="cachedb",
        mode="shadow",
        notify_enabled=False,
        gemini_auditor=auditor,
        intent_gateway=mock_gw,
    )
    reconciler.run_cycle(topic_key="auto_cryptotrade_executor")
    assert len(captured) == 1
    prompt_out = captured[0]
    assert "swing_wave(low=0.1636, high=0.1778, rally_low=+8.68%, pullback_high=+0.00%)" in prompt_out
    assert "swings_7d=[-6.5%, +8.7%], avg_amp=7.59%" in prompt_out
    assert "slopes(90s=+0.00092, 3.7m=+0.00064, 15m=+0.00045, 2.5h=+0.00021, grad_rec=+0.0012, grad_fast=+0.0018/eps=0.0004)" in prompt_out
    assert "g_coeff(90s=+0.00081, 3.7m=+0.00055, 15m=+0.00038, 2.5h=+0.00019)" in prompt_out
    assert "vol_1h=0.85%(adapt_reentry=1.7%) adapt_dca=2.1%" in prompt_out
    assert "FULL ACCESS TO EVERYTHING IN `cachedb/` AS A DECISION INFORMATION SOURCE" in prompt_out
    assert "UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING" in prompt_out
    assert "50%–100%` of `value_usd`" in prompt_out
    assert "continue working in parallel in the system and can be consulted by you on demand" in prompt_out
    assert "providers.market_api" in prompt_out
    assert "cachedb_catalog" not in prompt_out
    assert "STRONG_SELL_HARVEST_ZONE" not in prompt_out
    assert "SWING_3_8_RULE" not in prompt_out
    assert "do NOT call tools, shell commands, or file readers" not in prompt_out

    # 3. _normalize_trading_decisions deduplicates per (venue, symbol) so conflicting BUY+SELL on the same coin
    #    cannot both fire in one cycle, and normalizes percentage confidence & LONG/SHORT aliases
    from orchestratorOS.supervisor.autonomous_ai_reconciler import _normalize_trading_decisions

    deduped = _normalize_trading_decisions({
        "market_trading_decisions": [
            {"venue": "binance", "symbol": "TAOUSDC", "decision": "BUY", "confidence": "88%", "suggested_notional_usd": "$200"},
            {"venue": "binance", "symbol": "TAOUSDC", "decision": "SELL", "confidence": "94%", "suggested_notional_usd": "$650"},
            {"venue": "binance", "symbol": "TAOUSDC", "decision": "HOLD", "confidence": 0.99},
        ]
    })
    assert len(deduped) == 1
    assert deduped[0]["decision"] == "SELL"
    assert deduped[0]["confidence"] == pytest.approx(0.94)
    assert deduped[0]["suggested_notional_usd"] == pytest.approx(650.0)

    # 4. recent_ai_intents filters by scope BEFORE slicing [-8:] so T212 intents are never pushed out by >8 crypto intents
    now_ts = time.time()
    mixed_intents = [
        {"decision": "SELL", "symbol": "NVDA_US_EQ", "venue": "trading212", "status": "EXECUTED",
         "suggested_notional_usd": 500.0, "clamped_notional_usd": 142.5, "timestamp": now_ts - 1800},
    ] + [
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "status": "EXECUTED",
         "suggested_notional_usd": 100.0, "timestamp": now_ts - 60 * i}
        for i in range(12)
    ]
    (cache_dir / "autonomous_trade_intents_state.json").write_text(json.dumps(mixed_intents), encoding="utf-8")
    eq_snap = reconciler.harvest_sensors_telemetry(asset_scope="equities")
    eq_intents = eq_snap.portfolio_summary.get("recent_ai_intents", [])
    assert len(eq_intents) == 1
    assert eq_intents[0]["venue"] == "t212"
    assert eq_intents[0]["notional_usd"] == pytest.approx(142.5)
    assert eq_intents[0]["age_h"] == pytest.approx(0.5, abs=0.05)


def test_prompt_telemetry_accuracy_and_cross_venue_isolation(tmp_path, monkeypatch):
    """Verify prompt data accuracy across T212 Yahoo fallback, orderbook wall distances,
    venue-isolated trailing peaks/cycles/orders, realized_net PnL, and clean None formatting."""
    from orchestratorOS.supervisor.autonomous_ai_reconciler import (
        AutonomousAIReconciler,
        GeminiClient,
        _format_venue_cycle_line,
        _parse_venue_cycle_state,
    )
    from providers.t212_provider import T212Provider

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (tmp_path / "instruments.conf").write_text("", encoding="utf-8")

    # 1. _parse_venue_cycle_state extracts realized_net, uses fallback_prices for dd_pct,
    #    rounds float noise, and _format_venue_cycle_line omits (None) / dd=None%
    cyc_ada = _parse_venue_cycle_state(
        {
            "qty": 202.0,
            "cost_usd": 45.4,
            "entry_price": 0.22928099985954736,
            "last_buy_price": 0.2248,
            "last_sell_price": 239.99,
            "last_exit_kind": None,
            "dca_buys": 1,
            "realized_net": -3.194,
        },
        "ADAUSD",
        {},
        avg_decimals=6,
        fallback_prices={"ADAUSD": 0.1975},
    )
    assert cyc_ada["entry_price"] == 0.229281
    assert cyc_ada["realized_pnl_usd"] == -3.19
    assert cyc_ada["dd_pct"] == -12.13
    fmt_line = _format_venue_cycle_line("kraken", "ADAUSD", cyc_ada)
    assert "(None)" not in fmt_line
    assert "dd=-12.13%" in fmt_line
    assert "realized_pnl=$-3.19" in fmt_line

    # 2. Cross-venue isolation: Hyperliquid HYPEUSDC & TAOUSDC do not inherit Kraken's stale trailing_peak
    #    or Binance's cache_order.json cost basis, and orderbook wall dist uses ob_mid_price when cur_px diverges
    krk_dir = tmp_path / "kraken"
    krk_dir.mkdir(exist_ok=True)
    (krk_dir / "trailing_state.json").write_text(
        json.dumps({"HYPE": {"peak": 82.23}, "TAO": {"peak": 290.75}}),
        encoding="utf-8",
    )
    hl_dir = tmp_path / "hyperliquid"
    hl_dir.mkdir(exist_ok=True)
    (hl_dir / ".state_TAO_LONG.json").write_text(
        json.dumps({
            "qty": 1.0,
            "cost": 248.46,
            "entry_price": 248.46,
            "last_buy_price": 248.46,
            "last_sell_price": 239.99,
            "pos_peak": 248.725,
            "realized_net": -3.19,
        }),
        encoding="utf-8",
    )
    now_ms = int(time.time() * 1000)
    (cache_dir / "cache_order.json").write_text(
        json.dumps({
            "items": {
                "TAOUSDC": [
                    {
                        "side": "BUY",
                        "status": "FILLED",
                        "price": "241.9",
                        "executedQty": "1.0",
                        "cummulativeQuoteQty": "241.9",
                        "time": now_ms - 3600_000,
                    }
                ]
            }
        }),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps({
            "mid_price": 78.865,
            "imbalance_ratio": -0.35,
            "bid_depth_usd": 45000.0,
            "ask_depth_usd": 93000.0,
            "largest_bid_wall_price": 78.39,
            "largest_bid_wall_usd": 36000.0,
            "largest_ask_wall_price": 79.00,
            "largest_ask_wall_usd": 90000.0,
        }),
        encoding="utf-8",
    )

    hl_hype = MagicMock(symbol="HYPEUSDC", provider_name="hyperliquid", base="HYPE", quote="USDC", params={})
    hl_hype._provider.get_current_price.return_value = 93.35
    hl_hype._provider.free_balance.side_effect = lambda a: 0.0 if a == "HYPE" else 32.5
    hl_hype._provider.get_orders.return_value = []
    hl_hype.open_orders.return_value = []

    hl_tao = MagicMock(symbol="TAOUSDC", provider_name="hyperliquid", base="TAO", quote="USDC", params={})
    hl_tao._provider.get_current_price.return_value = 228.5
    hl_tao._provider.free_balance.side_effect = lambda a: 1.0 if a == "TAO" else 32.5
    hl_tao._provider.get_orders.return_value = []
    hl_tao.open_orders.return_value = []

    mock_gw = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    mock_gw.load_tradable_instruments.return_value = {"HL_HYPE": hl_hype, "HL_TAO": hl_tao}

    captured_prompts = []
    auditor = GeminiClient(
        custom_runner=lambda prompt, model, timeout: (
            captured_prompts.append(prompt)
            or json.dumps({"summary": "ok", "market_trading_decisions": []})
        )
    )
    reconciler = AutonomousAIReconciler(
        workspace_dir=str(tmp_path),
        cache_dir="cachedb",
        mode="shadow",
        notify_enabled=False,
        gemini_auditor=auditor,
        intent_gateway=mock_gw,
    )
    reconciler.run_cycle(topic_key="auto_cryptotrade_executor")
    crypto_prompt = captured_prompts[-1]

    # Hyperliquid HYPEUSDC must NOT have Kraken's 82.23 peak or positive dd_from_peak
    assert "dd_from_peak=13." not in crypto_prompt
    # Hyperliquid TAOUSDC must use its own cycle anchor (248.46) rather than Binance's 241.9
    assert "[hyperliquid] TAOUSDC: free_TAO=1.0 ($228.5 USD), free_USDC=$32.5, px=228.5, avg_buy_px=248.46" in crypto_prompt
    # Orderbook wall distance uses ob_mid_price (78.865) instead of mismatched 93.35 (-16%)
    assert "bid_wall=78.39 ($36000.0, dist=-0.60%)" in crypto_prompt
    assert "ask_wall=79.0 ($90000.0, dist=+0.17%)" in crypto_prompt
    # Missing last_sell formats as Last SELL=none rather than Last SELL=None @ None (age=Noneh)
    assert "Last SELL=none" in crypto_prompt
    assert "Last SELL=None @ None" not in crypto_prompt

    # 3. T212 Yahoo Finance 5d/5m chart fallback runs even when T212Provider has inherited ohlc_series,
    #    and rate-limited free_balance=None falls back to .state_NVDA_US_EQ.json qty
    t212_dir = tmp_path / "212trading"
    t212_dir.mkdir(exist_ok=True)
    (t212_dir / ".state_NVDA_US_EQ.json").write_text(
        json.dumps({"qty": 0.21, "cost_usd": 42.34, "entry_price": 201.62, "last_buy_price": 201.62, "dca_buys": 0}),
        encoding="utf-8",
    )
    real_t212_prov = T212Provider(client=MagicMock())
    # Simulate HTTP 429 rate-limit returning None on free_balance("NVDA_US_EQ")
    monkeypatch.setattr(real_t212_prov, "free_balance", lambda asset: 105.08 if asset == "USD" else None)
    monkeypatch.setattr(real_t212_prov, "get_current_price", lambda sym: 214.47)
    nvda_inst = MagicMock(
        symbol="NVDA_US_EQ",
        provider_name="t212",
        base="NVDA_US_EQ",
        quote="USD",
        params={},
        _provider=real_t212_prov,
    )
    nvda_inst.open_orders.return_value = []
    mock_gw.load_tradable_instruments.return_value = {"T212_NVDA": nvda_inst}

    now_s = int(time.time())
    fake_bars = [(now_s - (20 - i) * 300, 200.0 + i * 0.7, 1000.0) for i in range(21)]
    import types
    fake_md = types.ModuleType("market_data")
    fake_md._chart = lambda sym, rng="5d", interval="5m": ({"regularMarketPrice": 214.0}, fake_bars)
    monkeypatch.setitem(os.sys.modules, "market_data", fake_md)

    reconciler.run_cycle(topic_key="auto_sharetrade_executor", force_rth_open=True)
    eq_prompt = captured_prompts[-1]
    assert "free_NVDA_US_EQ=0.21 ($45.04 USD)" in eq_prompt
    assert "($None USD)" not in eq_prompt
    assert "24h_range=" in eq_prompt
    assert "swing_wave(" in eq_prompt


def test_volume_profile_liquidation_clusters_and_venue_fee_limits_telemetry(tmp_path) -> None:
    """Verify Volume Profile (POC/VAH/VAL), symmetric LONG/SHORT Liquidation Clusters, and Maker/Taker/Spread/Limits telemetry."""
    from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
    from orchestratorOS.supervisor.autonomous_ai_reconciler import (
        AutonomousAIReconciler,
        GeminiClient,
        compute_liquidation_clusters,
        compute_volume_profile_metrics,
        resolve_venue_fee_and_limits,
    )

    # 1. Volume Profile (POC, VAL, VAH, symmetric % distances, and regime)
    now_ms = int(time.time() * 1000)
    # Build a 24h series heavily concentrated around $80,500 (POC), spanning $79,200 to $81,800, ending at $81,600
    vp_series = []
    for i in range(60):
        tms = now_ms - (60 - i) * 60_000
        if i < 10:
            px = 79_200.0 + i * 100.0
        elif i < 50:
            px = 80_400.0 + (i % 5) * 50.0
        else:
            px = 81_200.0 + (i - 50) * 60.0
        vp_series.append([tms, px, 100.0 if 10 <= i < 50 else 20.0])

    vp = compute_volume_profile_metrics(vp_series)
    assert vp["val_px"] <= vp["poc_px"] <= vp["vah_px"]
    assert 80_200.0 <= vp["poc_px"] <= 80_800.0
    assert vp["va_regime"] in ("ABOVE_VAH", "INSIDE_VA_UPPER")
    assert vp["poc_dist_pct"] > 0.0

    # 2. Symmetric Liquidation Clusters (LONG downside & SHORT upside)
    pinfo_btc = {
        "price": 80_500.0,
        "low_24h": 79_600.0,
        "high_24h": 81_500.0,
        "val_px": vp["val_px"],
        "vah_px": vp["vah_px"],
        "poc_px": vp["poc_px"],
        "swing_low_px": 77_500.0,
        "swing_high_px": 83_800.0,
        "vol_1h_pct": 1.2,
    }
    minfo_btc = {
        "ob_mid_price": 80_500.0,
        "oi_usd": 600_000_000.0,
        "whale_long_pct": 0.44,
        "funding_rate": -0.0001,
        "oi_1h_chg_pct": 2.5,
        "bid_wall_px": 79_650.0,
        "ask_wall_px": 81_450.0,
    }
    liq = compute_liquidation_clusters(pinfo_btc, minfo_btc)
    assert liq["long_liq_deep_px"] < liq["long_liq_px"] < 80_500.0 < liq["short_liq_px"] < liq["short_liq_deep_px"]
    assert liq["long_liq_dist_pct"] < 0.0 < liq["short_liq_dist_pct"]
    assert liq["short_liq_usd"] > liq["long_liq_usd"] > 0.0
    assert liq["liq_magnet_bias"] == "SHORT_SQUEEZE_MAGNET_UP"

    # 3. OrderbookSnapshot spread_bps and resolve_venue_fee_and_limits across venues
    ob_snap = OrderbookSnapshot.from_dict({
        "symbol": "BTCUSDC",
        "mid_price": 80_500.0,
        "best_bid": 80_480.0,
        "best_ask": 80_520.0,
        "bid_depth_usd": 500_000.0,
        "ask_depth_usd": 450_000.0,
        "imbalance_ratio": 0.53,
        "largest_bid_wall_usd": 150_000.0,
        "largest_bid_wall_price": 79_650.0,
        "largest_ask_wall_usd": 200_000.0,
        "largest_ask_wall_price": 81_450.0,
        "ts": time.time(),
    })
    assert ob_snap.spread_bps == pytest.approx(4.97, abs=0.05)

    hl_specs = resolve_venue_fee_and_limits(
        "hyperliquid", "HYPEUSDC", price=60.0, free_quote=500.0, free_base=10.0, spread_bps=3.2
    )
    assert hl_specs["fee_tier"] == "SPOT_TIER0"
    assert hl_specs["maker_fee_bps"] == 4.0
    assert hl_specs["taker_fee_bps"] == 7.0
    assert hl_specs["spread_bps"] == 3.2
    assert hl_specs["effective_max_buy_usd"] == 497.5
    assert hl_specs["effective_max_sell_usd"] == 600.0

    krk_specs = resolve_venue_fee_and_limits(
        "kraken", "HYPEUSD", price=60.0, free_quote=100.0, free_base=5.0, spread_bps=12.5
    )
    assert krk_specs["fee_tier"] == "PRO_TIER0"
    assert krk_specs["maker_fee_bps"] == 16.0
    assert krk_specs["taker_fee_bps"] == 26.0
    assert krk_specs["maker_rebate_edge_bps"] == 10.0

    t212_specs = resolve_venue_fee_and_limits(
        "t212", "NVDA_US_EQ", price=215.0, free_quote=400.0, free_base=1.0
    )
    assert t212_specs["fee_tier"] == "INVEST_RTH"
    assert t212_specs["maker_fee_bps"] == 15.0
    assert t212_specs["taker_fee_bps"] == 15.0
    assert t212_specs["step_size"] == 0.001
    assert t212_specs["tick_size"] == 0.01

    assert abs(liq["long_liq_dist_pct"]) >= 1.2
    assert abs(liq["short_liq_dist_pct"]) >= 1.2

    # 4. End-to-end prompt integration in Section 3A, 3B, 3C
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (tmp_path / "order_guard.conf").write_text("supervisor_mode = shadow\nsupervisor_notify = false\n", encoding="utf-8")
    (tmp_path / "config.env").write_text("ntfy_topic_guard=test_topic\n", encoding="utf-8")
    (tmp_path / "instruments.conf").write_text("BTCUSDC = binance\n", encoding="utf-8")
    (cache_dir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[pt[0], pt[1]] for pt in vp_series]}}),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps(ob_snap.to_dict()),
        encoding="utf-8",
    )
    (cache_dir / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps({"funding_rate": -0.0001, "predicted_funding_rate": -0.0001, "open_interest_usd": 600_000_000.0}),
        encoding="utf-8",
    )
    (cache_dir / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps({
            "top_traders_long_pct": 0.44,
            "top_traders_long_ratio": 0.79,
            "taker_buy_sell_ratio": 1.15,
            "open_interest_1h_change_pct": 2.5,
            "divergence_regime": "bullish_accumulation",
        }),
        encoding="utf-8",
    )

    btc_inst = MagicMock(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={})
    btc_inst._provider.get_current_price.return_value = 81_740.0
    btc_inst._provider.free_balance.side_effect = lambda a: 0.05 if a == "BTC" else 1000.0
    btc_inst.open_orders.return_value = []
    mock_gw = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    mock_gw.load_tradable_instruments.return_value = {"BIN_BTC": btc_inst}

    prompts: list[str] = []
    reconciler = AutonomousAIReconciler(
        workspace_dir=str(tmp_path),
        cache_dir="cachedb",
        mode="shadow",
        notify_enabled=False,
        gemini_auditor=GeminiClient(
            custom_runner=lambda p, m, t: prompts.append(p) or json.dumps({"summary": "ok", "market_trading_decisions": []})
        ),
        intent_gateway=mock_gw,
    )
    reconciler.run_cycle(topic_key="auto_cryptotrade_executor")
    full_prompt = prompts[-1]
    assert "vol_profile(POC=" in full_prompt
    assert "liq_clusters(long_liq=$" in full_prompt
    assert "short_liq=$" in full_prompt
    assert "spread=4.97bps (bid=80480.0/ask=80520.0)" in full_prompt
    assert "venue_fee_per_leg=10bps (tier=VIP0, maker=10bps/taker=10bps, spread=4.97bps)" in full_prompt
    assert "limits(min_notional=$5, min_qty=" in full_prompt
    assert "max_buy=$995.0, max_sell=$2000.0" in full_prompt


def test_telemetry_bugfixes_and_api_normalization(tmp_path):
    """Verifies bug fixes across spread rounding, 24h range decimation, VP timestamp filtering,
    canceled order exclusion, multi-fill aggregation, T212 sell dust cap, and flat trailing peak cleanup.
    """
    from intelligence.external.collectors.orderbook_depth import round_spread_bps
    from intelligence.sentiment.guards.high_stake_guard import (
        compute_24h_price_metrics,
        compute_liquidation_clusters,
        compute_volume_profile_metrics,
        extract_symbol_recent_fills,
        harvest_recent_orders_and_enrich_positions,
        resolve_venue_fee_and_limits,
    )

    # 1. round_spread_bps precision & inverted book protection
    assert round_spread_bps(best_bid=100_000.0, best_ask=100_000.5) == 0.05
    assert round_spread_bps(best_bid=100.0, best_ask=100.1) == 10.0
    assert round_spread_bps(best_bid=100.1, best_ask=100.0) is None

    # 2. compute_24h_price_metrics preserves final point in low_24h/high_24h on 1,000-point series
    now_ms = 1_760_000_000_000
    series_1000 = [[now_ms - (1000 - i) * 60_000, 100.0 + (i % 10)] for i in range(999)]
    series_1000.append([now_ms, 125.0])  # Index 999 (skipped by [::2] step if decimated early)
    m24 = compute_24h_price_metrics(series_1000)
    assert m24 is not None
    assert m24["high_24h"] == 125.0
    assert m24["dd_24h_high_pct"] == 0.0
    assert m24["pos_24h_pct"] == 100.0

    # 3. compute_volume_profile_metrics filters invalid boolean timestamps and sorts chronologically
    unsorted_vp = (
        [[True, 999999.0], [now_ms, 108.0]]
        + [[now_ms - (20 - i) * 60_000, 100.0 + (i % 3)] for i in range(15)]
    )
    vp = compute_volume_profile_metrics(unsorted_vp)
    assert vp["poc_px"] < 200.0
    assert vp["va_regime"] == "ABOVE_VAH"

    # 4. compute_liquidation_clusters keeps live cur_px when cur_px moves past nearby orderbook walls
    liq = compute_liquidation_clusters(
        {"price": 82_000.0, "vol_1h_pct": 1.2},
        {"ob_mid_price": 80_500.0, "bid_wall_px": 80_000.0, "ask_wall_px": 81_000.0, "oi_usd": 100_000_000.0},
    )
    assert liq["long_liq_px"] < 82_000.0 < liq["short_liq_px"]

    # 5. extract_symbol_recent_fills ignores unfilled CANCELED orders & aggregates partial fills by orderId
    fills = extract_symbol_recent_fills([
        {"orderId": 101, "side": "BUY", "status": "PARTIALLY_FILLED", "price": "50.0", "executedQty": "1.0", "time": 1000},
        {"orderId": 101, "side": "BUY", "status": "FILLED", "price": "52.0", "executedQty": "1.0", "time": 1005},
        {"orderId": 102, "side": "SELL", "status": "CANCELED", "price": "60.0", "origQty": "5.0", "executedQty": "0.0", "time": 2000},
    ])
    assert fills is not None
    assert fills["last_action_side"] == "BUY"
    assert fills["last_sell"] is None
    assert len(fills["recent_buys"]) == 1
    assert fills["recent_buys"][0]["qty"] == 2.0
    assert fills["recent_buys"][0]["price"] == 51.0

    # 6. resolve_venue_fee_and_limits applies t212_sell_balance_cap for 3rd-decimal fractional dust
    t212_dust_specs = resolve_venue_fee_and_limits(
        "t212", "NVDA_US_EQ", price=200.0, free_quote=500.0, free_base=0.155
    )
    # 0.155 shares @ $200 has 0.005 dust ($1.00 < $6.00), so $6 buffer (0.03 shares) caps sellable qty to 0.12 shares = $24.0
    assert t212_dust_specs["effective_max_sell_usd"] == 24.0

    # 7. Flat cycle position clears stale trailing_peak and Hyperliquid does not overwrite Binance recent_orders
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "cache_order.json").write_text(
        json.dumps({
            "items": {
                "TAOUSDC": [
                    {"orderId": 1, "side": "BUY", "status": "FILLED", "price": "300.0", "executedQty": "1.0", "time": 1000}
                ]
            }
        }),
        encoding="utf-8",
    )
    hl_inst = MagicMock(symbol="TAOUSDC", provider_name="hyperliquid")
    hl_inst._provider.get_orders.return_value = [
        {"oid": 2, "side": "B", "px": "310.0", "sz": "1.0", "time": 2000}
    ]
    positions = [
        {
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "base": "HYPE",
            "quote": "USD",
            "price": 43.877,
            "free_base": 0.0,
            "free_quote": 85.0,
            "trailing_peak": 43.877,
            "dd_from_peak_pct": 0.0,
        }
    ]
    ro_map = harvest_recent_orders_and_enrich_positions(
        cache_dir=str(cache_dir),
        inst_by_sym={"HL_TAO": hl_inst},
        tradable_positions=positions,
        prices_and_trends={"HYPEUSD": {"price": 43.877}},
        kraken_cycles={"HYPEUSD": {"qty": 0.0, "cost_usd": 0.0, "last_sell_price": 42.1}},
        hyperliquid_cycles={},
        t212_cycles={},
    )
    assert ro_map["TAOUSDC"]["last_buy"]["price"] == 300.0
    assert ro_map["hyperliquid:TAOUSDC"]["last_buy"]["price"] == 310.0
    assert "trailing_peak" not in positions[0]
    assert "dd_from_peak_pct" not in positions[0]

    # 8. extract_executed_qty & build_last_cycle_execution_status report OPEN_RESTING (not PARTIALLY_FILLED) on unfilled resting limit orders
    from providers.base import extract_executed_qty
    from providers.execution_audit import build_last_cycle_execution_status, extract_intent_notional

    assert extract_executed_qty({"status": "NEW", "origQty": "0.005", "executedQty": "0.0"}) == 0.0
    assert extract_executed_qty({"status": "WORKING", "orderedQuantity": 2.0, "filledQuantity": 0.0}) == 0.0
    assert extract_executed_qty({"status": "resting", "sz": "10.0", "origSz": "10.0"}) == 0.0
    assert extract_executed_qty({"status": "PARTIALLY_FILLED", "origQty": "0.005", "executedQty": "0.002"}) == 0.002

    resting_status = build_last_cycle_execution_status(
        [
            {
                "timestamp": 1793306000.0,
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "order_type": "LIMIT",
                "limit_price": 78500.0,
                "reference_price": 79200.0,
                "suggested_notional_usd": 392.5,
                "status": "EXECUTED",
                "execution_payload": {
                    "orderId": "990011",
                    "status": "NEW",
                    "price": "78500.0",
                    "origQty": "0.005",
                    "executedQty": "0.0",
                },
            }
        ],
        "crypto",
        1793306060.0,
        extract_intent_notional,
        open_orders=[
            {
                "venue": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "order_type": "LIMIT",
                "order_id": "990011",
                "price": 78500.0,
                "orig_qty": 0.005,
                "filled_qty": 0.0,
                "remaining_qty": 0.005,
                "status": "NEW",
            }
        ],
    )
    assert resting_status["status"] == "OPEN_RESTING"
    assert resting_status["fill_px"] is None
    assert resting_status["filled_qty"] is None
    assert resting_status["slippage_bps"] is None

    from providers.execution_audit import cancel_existing_ai_open_orders
    krk_inst = MagicMock(symbol="HYPEUSD")
    krk_inst.open_orders.return_value = [
        {"ordertxid": "OKRK_AI_1", "descr": {"type": "buy", "ordertype": "limit", "price": "42.0"}},
        {"ordertxid": "OKRK_SELL_KEEP", "descr": {"type": "sell", "ordertype": "limit", "price": "48.0"}},
    ]
    canceled = cancel_existing_ai_open_orders(
        krk_inst,
        "BUY",
        intents=[{"history": [{"order_id": "OKRK_AI_1"}]}],
    )
    assert canceled == ["OKRK_AI_1"]


def test_open_limit_orders_macro_benchmarks_and_last_cycle_execution_status(tmp_path):
    """Verify open_limit_orders, SPY/QQQ/DXY/US10Y/Futures 24h delta macro benchmarks, and last_cycle_execution_status."""
    from intelligence.sentiment.guards.high_stake_guard import collect_macro_benchmarks_snapshot
    from providers.t212_provider import T212Provider

    workspace = str(tmp_path)
    cache_dir = os.path.join(workspace, "cachedb")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace, "logs"), exist_ok=True)
    os.makedirs(os.path.join(workspace, "212trading"), exist_ok=True)
    with open(os.path.join(workspace, "instruments.conf"), "w", encoding="utf-8") as f:
        f.write("BTCUSDC|binance|BTC|USDC|1|own_ledger|24x7|\n")

    now_ts = time.time()
    # 1. Test collect_macro_benchmarks_snapshot with mocked Yahoo Finance chart payloads
    def _fake_fetch(url, timeout=3.5, headers=None):
        sym_map = {
            "SPY": (582.40, -0.85, "POST"),
            "QQQ": (501.20, -1.40, "POST"),
            "ES%3DF": (5845.00, -0.90, "REGULAR"),
            "NQ%3DF": (20310.00, -1.55, "REGULAR"),
            "DX-Y.NYB": (104.25, 0.32, "REGULAR"),
            "%5ETNX": (4.28, 1.12, "REGULAR"),
        }
        for key, (px, chg24, mstate) in sym_map.items():
            if key in url:
                p_1h = round(px / 1.002, 4)
                p_15m = round(px / 1.001, 4)
                return {
                    "chart": {
                        "result": [
                            {
                                "meta": {
                                    "regularMarketPrice": px,
                                    "fulldayPrice": px,
                                    "fulldayChangePercent": chg24,
                                    "marketState": mstate,
                                },
                                "timestamp": [int(now_ts - 3600), int(now_ts - 900), int(now_ts)],
                                "indicators": {"quote": [{"close": [p_1h, p_15m, px]}]},
                            }
                        ]
                    }
                }
        return {}

    mb_snap = collect_macro_benchmarks_snapshot(
        cache_dir, now_ts=None, persist=True, fetch_fn=_fake_fetch
    )
    assert mb_snap is not None
    assert "SPY=$582.40 (-0.85% 24h" in mb_snap["summary_line"]
    assert "QQQ=$501.20 (-1.40% 24h" in mb_snap["summary_line"]
    assert "ES=F=$5,845.00 (-0.90% 24h" in mb_snap["summary_line"]
    assert "NQ=F=$20,310.00 (-1.55% 24h" in mb_snap["summary_line"]
    assert "DXY=104.25 (+0.32% 24h" in mb_snap["summary_line"]
    assert "US10Y=4.28% (+1.12% 24h" in mb_snap["summary_line"]

    # 2. Test T212Provider.open_orders filtering by ticker
    fake_t212_cli = MagicMock()
    fake_t212_cli.list_active_orders.return_value = [
        {"id": 8801, "ticker": "NVDA_US_EQ", "orderedQuantity": -1.5, "filledQuantity": 0.5, "limitPrice": 135.0, "status": "PARTIALLY_FILLED", "type": "LIMIT"},
        {"id": 8802, "ticker": "SPCX_US_EQ", "orderedQuantity": 2.0, "filledQuantity": 0.0, "limitPrice": 110.0, "status": "NEW", "type": "LIMIT"},
    ]
    t212_prov = T212Provider(client=fake_t212_cli, live_enabled=True)
    nvda_open = t212_prov.open_orders("NVDA_US_EQ")
    assert len(nvda_open) == 1
    assert nvda_open[0]["orderId"] == "8801"
    assert nvda_open[0]["side"] == "SELL"
    assert nvda_open[0]["price"] == 135.0
    assert nvda_open[0]["origQty"] == 1.5
    assert nvda_open[0]["executedQty"] == 0.5

    # 3. Test last_cycle_execution_status (FILLED with slippage_bps) & open_limit_orders in telemetry + prompt
    intents_state = [
        {
            "intent_id": "intent_1_buy_btcusdc",
            "timestamp": now_ts - 600.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "confidence": 0.92,
            "suggested_notional_usd": 400.0,
            "clamped_notional_usd": 396.07,
            "order_type": "MARKET",
            "reference_price": 79204.89,
            "requested_price": 79204.89,
            "status": "EXECUTED",
            "execution_payload": {
                "orderId": 555001,
                "status": "FILLED",
                "executedQty": "0.005",
                "cummulativeQuoteQty": "396.07195",
            },
            "history": [
                {"ts": now_ts - 600.0, "action": "EXECUTED", "price": 79204.89, "qty": 0.005, "notional_usd": 396.07}
            ],
        }
    ]
    with open(os.path.join(cache_dir, "autonomous_trade_intents_state.json"), "w", encoding="utf-8") as f:
        json.dump(intents_state, f)

    mock_btc_inst = MagicMock()
    mock_btc_inst.symbol = "BTCUSDC"
    mock_btc_inst.provider_name = "binance"
    mock_btc_inst.base = "BTC"
    mock_btc_inst.quote = "USDC"
    mock_btc_inst.params = {}
    mock_btc_inst._provider.get_current_price.return_value = 79300.0
    mock_btc_inst._provider.free_balance.side_effect = lambda a: 0.025 if a == "BTC" else 1200.0
    mock_btc_inst.open_orders.return_value = [
        {
            "orderId": "777888",
            "side": "BUY",
            "type": "LIMIT",
            "price": "78500.0",
            "origQty": "0.005",
            "executedQty": "0.002",
            "status": "PARTIALLY_FILLED",
        }
    ]

    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    gateway.load_tradable_instruments.return_value = {"BTCUSDC": mock_btc_inst}
    gateway.reconcile_intents.return_value = False

    captured_prompts = []
    auditor = GeminiClient(custom_runner=lambda p, m, t: captured_prompts.append(p) or json.dumps({"summary": "ok"}))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
        intent_gateway=gateway,
    )
    sensors = reconciler.harvest_sensors_telemetry(asset_scope="crypto")
    ps = sensors.portfolio_summary

    assert len(ps["open_limit_orders"]) == 1
    oo = ps["open_limit_orders"][0]
    assert oo["venue"] == "binance"
    assert oo["symbol"] == "BTCUSDC"
    assert oo["order_id"] == "777888"
    assert oo["price"] == 78500.0
    assert oo["orig_qty"] == 0.005
    assert oo["filled_qty"] == 0.002
    assert oo["remaining_qty"] == 0.003
    assert oo["qty"] == 0.003
    assert oo["status"] == "PARTIALLY_FILLED"

    last_exec = ps["last_cycle_execution_status"]
    assert last_exec["symbol"] == "BTCUSDC"
    assert last_exec["venue"] == "binance"
    assert last_exec["status"] == "FILLED"
    assert last_exec["fill_px"] == 79214.39
    assert last_exec["ref_px"] == 79204.89
    assert last_exec["slippage_bps"] == 1.2

    reconciler.synthesize_and_diagnose(sensors, reconciler.harvest_log_anomalies(), topic_key="auto_cryptotrade_executor")
    prompt_text = captured_prompts[-1]
    assert "open_limit_orders:" in prompt_text
    assert "'order_id': '777888'" in prompt_text
    assert "last_cycle_execution_status:" in prompt_text
    assert "'fill_px': 79214.39" in prompt_text
    assert "'slippage_bps': 1.2" in prompt_text
    assert "SPY=$582.40 (-0.85% 24h" in prompt_text
    assert "US10Y=4.28% (+1.12% 24h" in prompt_text

    # 4. When no open orders exist on the exchange, Section 3C explicitly shows open_limit_orders: [] (none resting)
    mock_btc_inst.open_orders.return_value = []
    sensors_empty_oo = reconciler.harvest_sensors_telemetry(asset_scope="crypto")
    assert sensors_empty_oo.portfolio_summary["open_limit_orders"] == []
    reconciler.synthesize_and_diagnose(
        sensors_empty_oo, reconciler.harvest_log_anomalies(), topic_key="auto_cryptotrade_executor"
    )
    prompt_empty_oo = captured_prompts[-1]
    assert "Active Open Limit Orders on Exchange — open_limit_orders: [] (none resting)" in prompt_empty_oo
    assert "SUPERVISORY AUTHORITY TO AUDIT & REPAIR / REVERSE UNWANTED BOT OR FLEET TRADES" in prompt_empty_oo
    assert "by=AI" in prompt_empty_oo


def test_bot_vs_ai_fill_attribution_and_reversal_prompt(temp_env):
    """Verify interim bot fills are tagged by=BOT vs AI fills by=AI so the AI can audit and reverse unwanted bot trades."""
    workspace, cache_dir, _ = temp_env
    now_ms = int(time.time() * 1000)

    # AI bought BTCUSDC 2h ago (orderId 9001 in autonomous_trade_intents_state.json),
    # then a deterministic bot sold BTCUSDC 6m ago (orderId 9002, clientOrderId MT_STOP_1).
    with open(os.path.join(cache_dir, "autonomous_trade_intents_state.json"), "w") as f:
        json.dump([
            {
                "intent_id": "intent_ai_buy_9001",
                "timestamp": (now_ms - 7_200_000) / 1000.0,
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "order_type": "MARKET",
                "confidence": 0.92,
                "suggested_notional_usd": 400.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "9001", "status": "FILLED", "price": "78500.0", "executedQty": "0.005"},
            }
        ], f)

    with open(os.path.join(cache_dir, "cache_order.json"), "w") as f:
        json.dump({
            "items": {
                "BTCUSDC": [
                    {
                        "orderId": "9001",
                        "side": "BUY",
                        "price": 78500.0,
                        "executedQty": 0.005,
                        "status": "FILLED",
                        "timestamp": now_ms - 7_200_000,
                    },
                    {
                        "orderId": "9002",
                        "clientOrderId": "MT_STOP_1",
                        "side": "SELL",
                        "price": 78100.0,
                        "executedQty": 0.005,
                        "status": "FILLED",
                        "timestamp": now_ms - 360_000,
                    },
                ]
            }
        }, f)

    with open(os.path.join(workspace, "instruments.conf"), "w") as f:
        f.write("BTCUSDC = binance:BTC/USDC\n")

    mock_btc_inst = MagicMock()
    mock_btc_inst.symbol = "BTCUSDC"
    mock_btc_inst.provider_name = "binance"
    mock_btc_inst.base = "BTC"
    mock_btc_inst.quote = "USDC"
    mock_btc_inst.params = {}
    mock_btc_inst._provider.free_balance.side_effect = lambda asset: 0.0 if asset == "BTC" else 2000.0
    mock_btc_inst._provider.get_current_price.return_value = 78200.0
    mock_btc_inst.open_orders.return_value = [
        {"orderId": "bot_rest_1", "clientOrderId": "TO_123", "side": "BUY", "price": "76000.0", "origQty": "0.005", "executedQty": "0.0"},
        {"orderId": "ai_rest_1", "clientOrderId": "AI_123", "side": "BUY", "price": "77900.0", "origQty": "0.005", "executedQty": "0.0"},
    ]

    gateway = MagicMock(min_confidence=0.85, max_notional_usd=2000.0, min_notional_usd=10.0)
    gateway.load_tradable_instruments.return_value = {"BTCUSDC": mock_btc_inst}
    gateway.reconcile_intents.return_value = False

    captured_prompts = []
    auditor = GeminiClient(custom_runner=lambda p, m, t: captured_prompts.append(p) or json.dumps({"summary": "ok"}))
    reconciler = AutonomousAIReconciler(
        workspace_dir=workspace,
        cache_dir="cachedb",
        logs_dirs=["logs"],
        gemini_auditor=auditor,
        mode="shadow",
        notify_enabled=False,
        intent_gateway=gateway,
    )
    sensors = reconciler.harvest_sensors_telemetry(asset_scope="crypto")
    ro = sensors.portfolio_summary["recent_orders"]["BTCUSDC"]
    assert ro["last_action_side"] == "SELL"
    assert ro["last_action_actor"] == "BOT"
    assert ro["last_buy"]["actor"] == "AI"
    assert ro["last_sell"]["actor"] == "BOT"

    open_lim = {o["order_id"]: o["actor"] for o in sensors.portfolio_summary["open_limit_orders"]}
    assert open_lim == {"bot_rest_1": "BOT", "ai_rest_1": "AI"}

    reconciler.synthesize_and_diagnose(sensors, reconciler.harvest_log_anomalies(), topic_key="auto_cryptotrade_executor")
    prompt_text = captured_prompts[-1]
    assert "BTCUSDC Last Action=SELL (by=BOT" in prompt_text
    assert "by=AI)" in prompt_text
    assert "by=BOT" in prompt_text
    assert "If a bot SOLD prematurely or when it should NOT have" in prompt_text
    assert "If a bot BOUGHT prematurely or when it should NOT have" in prompt_text


def test_harvest_recent_orders_multi_venue_cache_and_symbol_collision(temp_env):
    from intelligence.sentiment.guards.high_stake_guard import harvest_recent_orders_and_enrich_positions

    _, cache_dir, _ = temp_env
    with open(os.path.join(cache_dir, "cache_order.json"), "w") as f:
        json.dump(
            {
                "items": {
                    "BTCUSDC": [
                        {"orderId": "b1", "side": "BUY", "price": "70000.0", "executedQty": "0.01", "time": 1700000001000}
                    ]
                }
            },
            f,
        )
    with open(os.path.join(cache_dir, "cache_trade_hyperliquid.json"), "w") as f:
        json.dump(
            {
                "items": {
                    "BTCUSDC": [
                        {"orderId": "hl1", "side": "BUY", "price": "71000.0", "executedQty": "0.02", "time": 1700000002000}
                    ]
                }
            },
            f,
        )
    with open(os.path.join(cache_dir, "cache_trade_t212.json"), "w") as f:
        json.dump(
            {
                "items": {
                    "NVDA_US_EQ": [
                        {"orderId": "t1", "side": "BUY", "price": "125.5", "executedQty": "2.0", "time": 1700000003000}
                    ]
                }
            },
            f,
        )

    positions = [
        {"symbol": "BTCUSDC", "venue": "hyperliquid", "free_base": 0.02, "price": 71500.0},
        {"symbol": "NVDA_US_EQ", "venue": "t212", "free_base": 2.0, "price": 128.0},
    ]
    recent = harvest_recent_orders_and_enrich_positions(
        cache_dir,
        {},
        positions,
        {},
        {},
        {},
        {},
    )
    assert recent["BTCUSDC"]["last_buy"]["price"] == 70000.0
    assert recent["hyperliquid:BTCUSDC"]["last_buy"]["price"] == 71000.0
    assert recent["NVDA_US_EQ"]["last_buy"]["price"] == 125.5
    assert positions[0]["last_buy_px"] == 71000.0
    assert positions[1]["last_buy_px"] == 125.5


def test_bot_module_rule_tag_cvd_15m_and_depth_1pct_in_prompt(temp_env):
    from intelligence.sentiment.guards.high_stake_guard import (
        extract_symbol_recent_fills,
        harvest_recent_orders_and_enrich_positions,
    )
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

    _, cache_dir, _ = temp_env
    with open(os.path.join(cache_dir, "execution_audit_2026_10_30.jsonl"), "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "order_id": "9901",
                    "client_order_id": "AGB1700000001",
                    "kind": "SL_PANIC",
                }
            )
            + "\n"
        )
    with open(os.path.join(cache_dir, "cache_order.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "9900",
                            "clientOrderId": "TA_OR_1700000000",
                            "side": "BUY",
                            "price": "68000.0",
                            "executedQty": "0.01",
                            "time": 1700000000000,
                            "rule": "kalman_primary_up",
                        },
                        {
                            "orderId": "9901",
                            "clientOrderId": "AGB1700000001",
                            "side": "SELL",
                            "price": "67500.0",
                            "executedQty": "0.01",
                            "time": 1700000005000,
                        },
                    ]
                }
            },
            f,
        )

    recent = harvest_recent_orders_and_enrich_positions(cache_dir, {}, [], {}, {}, {}, {})
    btc_ro = recent["BTCUSDC"]
    assert btc_ro["last_buy"]["actor"] == "BOT"
    assert btc_ro["last_buy"]["module"] == "tradeall"
    assert btc_ro["last_buy"]["rule"] == "kalman_primary_up"
    assert btc_ro["last_sell"]["actor"] == "BOT"
    assert btc_ro["last_sell"]["module"] == "assetguardian"
    assert btc_ro["last_sell"]["rule"] == "SL_PANIC"
    assert btc_ro["last_action_actor_tag"] == "BOT(module=assetguardian, rule=SL_PANIC)"

    from types import SimpleNamespace

    sensors = SimpleNamespace(
        prices_and_trends={"BTCUSDC": {"price": 68000.0, "chg_15m_pct": 0.08}},
        microstructure={
            "BTCUSDC": {
                "ob_imbalance": 0.62,
                "bid_depth_usd": 67500.0,
                "ask_depth_usd": 78000.0,
                "depth_1pct_bid_usd": 45000.0,
                "depth_1pct_ask_usd": 52000.0,
                "spread_bps": 2.0,
                "bid_wall_px": 67500.0,
                "bid_wall_usd": 25000.0,
                "ask_wall_px": 68500.0,
                "ask_wall_usd": 30000.0,
                "whale_long_pct": 0.58,
                "divergence_regime": "BULLISH_ABSORPTION",
                "taker_buy_sell_ratio": 1.14,
                "taker_buy_vol_usd": 1_360_000.0,
                "taker_sell_vol_usd": 680_000.0,
                "funding_rate": 0.0001,
                "pred_funding_rate": 0.0001,
                "oi_usd": 50_000_000.0,
                "oi_1h_chg_pct": 1.2,
            }
        },
        portfolio_summary={
            "total_value_usdc": 2000.0,
            "tradable_positions": [],
            "recent_orders": recent,
        },
    )
    prompt = build_supervisor_prompt(sensors, [])
    assert "by=BOT(module=assetguardian, rule=SL_PANIC)" in prompt
    assert "by=BOT(module=tradeall, rule=kalman_primary_up)" in prompt
    assert "depth_1pct=(bid=$45.0K/ask=$52.0K, slip_2k=3.2bps)" in prompt
    assert "cvd_15m=+10 BTC (+$680.0K, spot_absorbing)" in prompt


def test_pretrade_telemetry_without_24h_series_and_gateway_max_usd_forwarding(tmp_path):
    """Verify collect_pretrade_telemetry populates microstructure/liquidation clusters without 24h series and gateway_max_usd forwards."""
    from intelligence.sentiment.guards.high_stake_guard import (
        build_cycle_fallback_prices,
        collect_pretrade_telemetry,
        harvest_tradable_positions_and_orders,
        resolve_venue_fee_and_limits,
    )

    # 1. build_cycle_fallback_prices falls back to (best_bid + best_ask) / 2.0 when ob_mid_price is absent
    fb_px = build_cycle_fallback_prices(
        {"BTCUSDC": {"best_bid": 80_000.0, "best_ask": 80_200.0}},
        {},
    )
    assert fb_px["BTCUSDC"] == 80_100.0

    # 1b. resolve_venue_fee_and_limits reads Kraken _cli / AssetPairs tuple cache and _minqty
    krk_inst = MagicMock(symbol="HYPEUSD", provider_name="kraken", params={})
    krk_inst._provider._minqty = {"HYPEUSD": 0.1}
    krk_inst._provider._cli._CACHE = {
        ("AssetPairs", (("pair", "HYPEUSD"),)): (
            9999999999.0,
            {
                "HYPEUSD": {
                    "fees_maker": [[0, 0.12]],
                    "fees": [[0, 0.22]],
                    "costmin": "1.5",
                    "lot_decimals": 2,
                    "pair_decimals": 3,
                }
            },
        )
    }
    krk_limits = resolve_venue_fee_and_limits("kraken", "HYPEUSD", krk_inst, price=45.0)
    assert krk_limits["maker_fee_bps"] == 12.0
    assert krk_limits["taker_fee_bps"] == 22.0
    assert krk_limits["min_qty"] == 0.1
    assert krk_limits["exchange_min_notional_usd"] == 1.5
    assert krk_limits["step_size"] == 0.01
    assert krk_limits["tick_size"] == 0.001

    # 2. harvest_tradable_positions_and_orders forwards gateway_max_usd to resolve_venue_fee_and_limits
    inst_mock = MagicMock(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={})
    inst_mock._provider.free_balance.side_effect = lambda a: 1.0 if a == "BTC" else 5000.0
    inst_mock.open_orders.return_value = []
    positions, _, _, _ = harvest_tradable_positions_and_orders(
        {"BTCUSDC": inst_mock},
        {"BTCUSDC": {"price": 80_000.0}},
        {},
        {},
        gateway_max_usd=1250.0,
    )
    assert len(positions) == 1
    assert positions[0]["effective_max_buy_usd"] == 1250.0
    assert positions[0]["effective_max_sell_usd"] == 1250.0

    # 3. collect_pretrade_telemetry populates spread, 1% depth, CVD, and deep liquidation clusters even without 24h series
    telem_dir = tmp_path / "telemetry"
    telem_dir.mkdir(parents=True, exist_ok=True)
    (telem_dir / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps({
            "symbol": "BTCUSDC",
            "imbalance_ratio": 0.60,
            "best_bid": 79_990.0,
            "best_ask": 80_010.0,
            "bid_depth_usd": 600_000.0,
            "ask_depth_usd": 450_000.0,
            "depth_1pct_bid_usd": 400_000.0,
            "depth_1pct_ask_usd": 300_000.0,
            "largest_bid_wall_price": 79_500.0,
            "largest_bid_wall_usd": 180_000.0,
            "largest_ask_wall_price": 80_600.0,
            "largest_ask_wall_usd": 210_000.0,
        }),
        encoding="utf-8",
    )
    (telem_dir / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps({
            "symbol": "BTCUSDC",
            "funding_rate": 0.0001,
            "predicted_funding_rate": 0.00012,
            "open_interest_usd": 120_000_000.0,
        }),
        encoding="utf-8",
    )
    (telem_dir / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps({
            "symbol": "BTCUSDC",
            "top_traders_long_pct": 0.56,
            "top_traders_long_ratio": 1.27,
            "taker_buy_sell_ratio": 1.20,
            "taker_buy_vol_usd": 900_000.0,
            "taker_sell_vol_usd": 500_000.0,
            "open_interest_1h_change_pct": 1.5,
            "divergence_regime": "BULLISH_FLOW",
        }),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "autonomous_trade_intents_state.json").write_text(
        json.dumps([
            {"status": "EXECUTED", "execution_payload": {"orderId": "777001"}}
        ]),
        encoding="utf-8",
    )
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    (orders_dir / "cache_order.json").write_text(
        json.dumps({
            "items": {
                "BTCUSDC": [
                    {"orderId": "777001", "side": "BUY", "status": "FILLED", "price": "79800.0", "executedQty": "0.01", "time": 1000},
                ]
            }
        }),
        encoding="utf-8",
    )
    telem = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(tmp_path))
    assert "Orderbook Depth Imbalance" in telem
    assert "$210,000 @ $80,600.00" in telem["Orderbook Ask Wall Ahead"]
    assert "$180,000 @ $79,500.00" in telem["Orderbook Bid Wall Ahead"]
    assert "Perpetual Funding Rate" in telem
    assert "Derivatives Open Interest" in telem
    assert "Whale Positioning" in telem
    assert "Whale Taker Flow" in telem
    assert "Orderbook Bid-Ask Spread" in telem
    assert "Orderbook Depth Within 1%" in telem
    assert "15m Cumulative Volume Delta (CVD)" in telem
    assert "Liquidation Clusters (LONG vs SHORT)" in telem
    assert "deep_LONG=" in telem["Liquidation Clusters (LONG vs SHORT)"]
    assert "deep_SHORT=" in telem["Liquidation Clusters (LONG vs SHORT)"]
    assert "(by=AI)" in telem["Recent Venue Fills"]

    from types import SimpleNamespace
    from intelligence.sentiment.guards.high_stake_guard import (
        load_symbol_microstructure,
        resolve_venue_fee_and_limits,
    )

    hl_inst = SimpleNamespace(
        params={},
        _provider=SimpleNamespace(
            _spot_pair="@107",
            _client=SimpleNamespace(
                _meta_cache={},
                info=SimpleNamespace(
                    name_to_coin={"HYPE/USDC": "@107"},
                    coin_to_asset={"@107": 10107},
                    asset_to_sz_decimals={10107: 2},
                ),
            ),
        ),
    )
    hl_limits = resolve_venue_fee_and_limits("hyperliquid", "HYPEUSDC", hl_inst, price=42.5)
    assert hl_limits["step_size"] == pytest.approx(0.01)
    assert hl_limits["min_qty"] == pytest.approx(0.01)
    assert hl_limits["tick_size"] == pytest.approx(0.001)

    m_from_objs = load_symbol_microstructure(
        str(tmp_path),
        "UNLISTEDUSDC",
        ob=SimpleNamespace(
            imbalance_ratio=0.61,
            mid_price=50.0,
            best_bid=49.98,
            best_ask=50.02,
            spread_bps=8.0,
            bid_depth_usd=150000.0,
            ask_depth_usd=95000.0,
            depth_1pct_bid_usd=100000.0,
            depth_1pct_ask_usd=65000.0,
            largest_bid_wall_usd=45000.0,
            largest_ask_wall_usd=30000.0,
            largest_bid_wall_price=49.5,
            largest_ask_wall_price=50.5,
        ),
        deriv=SimpleNamespace(funding_rate=0.0001, predicted_funding_rate=0.00012, open_interest_usd=25000000.0),
        whale=SimpleNamespace(
            top_traders_long_pct=0.58,
            top_traders_long_ratio=1.38,
            taker_buy_sell_ratio=1.18,
            taker_buy_vol_usd=300000.0,
            taker_sell_vol_usd=180000.0,
            open_interest_1h_change_pct=1.2,
            divergence_regime="BULLISH_ACCUMULATION",
        ),
    )
    assert m_from_objs is not None
    assert m_from_objs["ob_imbalance"] == pytest.approx(0.61)
    assert m_from_objs["cvd_15m_usd"] == pytest.approx(120000.0)
    assert m_from_objs["oi_usd"] == pytest.approx(25000000.0)

    from intelligence.sentiment.guards.high_stake_guard import (
        harvest_macro_and_news_sensors,
        load_symbol_price_series,
    )

    (tmp_path / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps(
            {
                "BTCUSDC": [
                    [1000.0, 80000.0, 12.5],
                    [2000.0, 80100.0, 45.0],
                ]
            }
        ),
        encoding="utf-8",
    )
    series_24h, _ = load_symbol_price_series(
        str(tmp_path),
        "BTCUSDC",
        current_price_items={"BTCUSDC": [[2000.0, 80120.0]]},
    )
    assert series_24h[-1] == [2000000.0, 80120.0, 45.0]

    (tmp_path / "fear_greed_collect.json").write_text(
        json.dumps({"value": 24, "classification": "Extreme Fear"}),
        encoding="utf-8",
    )
    (tmp_path / "market_breadth_collect.json").write_text(
        json.dumps({"advance_ratio": 0.35, "regime": "WEAK"}),
        encoding="utf-8",
    )
    (tmp_path / "sentiment_eval.json").write_text(
        json.dumps({"market_bias": "NEUTRAL", "risk_level": "MODERATE"}),
        encoding="utf-8",
    )
    macro_map, _ = harvest_macro_and_news_sensors(str(tmp_path), now_ts=1793300000.0)
    assert macro_map["fear_greed"]["value"] == 24
    assert macro_map["market_breadth"]["advance_ratio"] == pytest.approx(0.35)
    assert macro_map["sentiment_advisor"]["market_bias"] == "NEUTRAL"

    from providers.base import extract_executed_qty, extract_order_side, normalize_open_order

    assert extract_order_side({"type": "limit", "descr": {"type": "buy"}}) == "BUY"
    assert extract_order_side({"dir": "Spot", "isBuyer": True}) == "BUY"
    assert extract_executed_qty({"oid": 12345, "sz": "10.0", "origSz": "10.0"}) == 0.0
    assert extract_executed_qty({"oid": 12345, "sz": "6.0", "origSz": "10.0"}) == pytest.approx(4.0)
    hl_partial = normalize_open_order(
        {"oid": 12345, "sz": "6.0", "origSz": "10.0", "limitPx": "40.0", "side": "B"},
        venue="hyperliquid",
        symbol="HYPEUSDC",
    )
    assert hl_partial is not None
    assert hl_partial["filled_qty"] == pytest.approx(4.0)
    assert hl_partial["remaining_qty"] == pytest.approx(6.0)
    assert hl_partial["status"] == "PARTIALLY_FILLED"

    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector

    w_col = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=str(tmp_path))
    with patch.object(
        WhalePositioningCollector,
        "_fetch_latest_json_row",
        side_effect=[
            ({"longShortRatio": "1.80", "longAccount": "0.64"}, [{"longShortRatio": "1.80", "longAccount": "0.64"}]),
            ({"buySellRatio": "1.10", "buyVol": "110.0", "sellVol": "100.0"}, [{"buySellRatio": "1.10", "buyVol": "110.0", "sellVol": "100.0"}]),
            (
                {"sumOpenInterest": "1000.0", "sumOpenInterestValue": "80000000.0"},
                [
                    {"sumOpenInterest": "990.0", "sumOpenInterestValue": "79000000.0"},
                    {"sumOpenInterest": "1000.0", "sumOpenInterestValue": "80000000.0"},
                ],
            ),
        ],
    ):
        w_snap = w_col.fetch("BTCUSDC", price_1h_change_pct=0.5, allow_network=True, now=1793300000.0)
    assert w_snap is not None
    assert w_snap.taker_buy_vol_usd == pytest.approx(8_800_000.0)
    assert w_snap.taker_sell_vol_usd == pytest.approx(8_000_000.0)

    from intelligence.sentiment.guards.high_stake_guard import harvest_recent_orders_and_enrich_positions
    from orchestratorOS.supervisor.prompt_formatters import _format_recent_order_line
    from providers.execution_audit import execute_enforce_intent_order

    mock_inst = MagicMock(symbol="BTCUSDC", provider_name="binance")
    mock_inst.place.return_value = {"orderId": "ok_cid"}
    for iid in (
        "intent_1793317660_sell_btcusdc",
        "intent_1793317661_sell_btcusdc",
        "intent_1793317660_sell_nvda_us_eq_1",
        "intent_1793317661_sell_nvda_us_eq_1",
    ):
        execute_enforce_intent_order(
            mock_inst,
            {"intent_id": iid},
            intent_id=iid,
            decision="SELL",
            target_qty=0.025,
            price=80000.0,
            effective_usd=2000.0,
            guard_allowed=True,
            guard_reason="PASSED",
            thesis="test",
            trade_tag="[auto_cryptotrade_executor]",
            now_ts=1793317660.0,
            bypass_guards=frozenset(),
        )
    cids = [c.kwargs["client_order_id"] for c in mock_inst.place.call_args_list]
    assert len(set(cids)) == 4
    assert all(len(c) <= 36 and c.startswith("AI_") for c in cids)

    flat_pos = [{"venue": "kraken", "symbol": "SOLUSD", "price": 150.0, "free_base": 0.0, "free_quote": 500.0}]
    krk_cyc = {
        "SOLUSD": {
            "qty": 0.0,
            "cost_usd": 0.0,
            "avg_cost_px": None,
            "entry_price": 140.0,
            "last_buy_price": 140.0,
            "pos_peak": 165.0,
            "resting_orders": [{"side": "buy", "px": 135.0, "sz": 1.0}],
        }
    }
    harvest_recent_orders_and_enrich_positions(
        str(tmp_path), {}, flat_pos, {"SOLUSD": {"price": 150.0}}, krk_cyc, {}, {}
    )
    assert "active_avg_buy_px" not in flat_pos[0]
    assert "trailing_peak" not in flat_pos[0]
    assert flat_pos[0]["last_buy_px"] == 140.0

    ro_line = _format_recent_order_line(
        "BTCUSDC",
        {"last_action_side": "BUY", "last_buy": {"qty": 0.1, "price": 80000.0, "ts_ms": 1793300000.0}},
        1793303600000,
    )
    assert "age=1.0h" in ro_line

    from instruments_config import resolve_instrument_live_price
    from intelligence.sentiment.guards.high_stake_guard import apply_asset_scope_filter
    from orchestratorOS.lib.technical_repair_ssot import (
        submit_supervisor_trading_decisions,
        summarize_fleet_sensor_status,
    )

    # 1. apply_asset_scope_filter filters recent_orders by scope (preserving equities for equities scope)
    p_eq = {
        "recent_orders": {
            "BTCUSDC": {"last_action_side": "BUY"},
            "NVDA_US_EQ": {"last_action_side": "BUY"},
            "t212:AAPL_US_EQ": {"last_action_side": "SELL"},
        },
        "positions": [],
        "open_orders": [],
    }
    p_cr = {
        "recent_orders": dict(p_eq["recent_orders"]),
        "positions": [],
        "open_orders": [],
    }
    apply_asset_scope_filter({}, {}, p_eq, "equities", {}, {})
    apply_asset_scope_filter({}, {}, p_cr, "crypto", {}, {})
    assert set(p_eq["recent_orders"].keys()) == {"NVDA_US_EQ", "t212:AAPL_US_EQ"}
    assert set(p_cr["recent_orders"].keys()) == {"BTCUSDC"}

    # 2. submit_supervisor_trading_decisions enforces asset scope isolation on BUY/SELL decisions
    from types import SimpleNamespace
    submitted_payloads = []
    mock_gw = MagicMock()
    mock_gw.submit.side_effect = lambda p, **kw: submitted_payloads.append(p) or {"status": "SHADOW_LOGGED"}
    plan_cross = {
        "market_trading_decisions": [
            {"decision": "BUY", "venue": "binance", "symbol": "BTCUSDC", "target_usd": 100.0, "confidence": 0.9},
            {"decision": "BUY", "venue": "t212", "symbol": "NVDA_US_EQ", "target_usd": 100.0, "confidence": 0.9},
        ]
    }
    submit_supervisor_trading_decisions(
        plan_cross,
        SimpleNamespace(can_trade=True, asset_scope="crypto"),
        topic_key="auto_cryptotrade_executor",
        mode="shadow",
        now_ts=1793300000.0,
        intent_gateway=mock_gw,
        is_rth_open_fn=lambda _: True,
    )
    assert [p["symbol"] for p in submitted_payloads] == ["BTCUSDC"]
    submitted_payloads.clear()
    submit_supervisor_trading_decisions(
        plan_cross,
        SimpleNamespace(can_trade=True, asset_scope="equities"),
        topic_key="auto_sharetrade_executor",
        mode="shadow",
        now_ts=1793300000.0,
        intent_gateway=mock_gw,
        is_rth_open_fn=lambda _: True,
    )
    assert [p["symbol"] for p in submitted_payloads] == ["NVDA_US_EQ"]

    # 3. summarize_fleet_sensor_status reads fallback state files and resolve_instrument_live_price uses ob_mid_price
    (tmp_path / "fear_greed_collect.json").unlink(missing_ok=True)
    (tmp_path / "fear_greed_state.json").write_text(json.dumps({"value": 33.0}), encoding="utf-8")
    (tmp_path / "orderbook_depth_ADAUSD.json").write_text(
        json.dumps({"best_bid": 0.70, "best_ask": 0.72, "mid_price": 0.71}),
        encoding="utf-8",
    )
    f_stat = summarize_fleet_sensor_status(str(tmp_path))
    assert f_stat["fear_greed"] == 33.0
    ada_inst = MagicMock(symbol="ADAUSD", base="ADA", _provider=None)
    assert resolve_instrument_live_price(ada_inst, str(tmp_path)) == pytest.approx(0.71)



