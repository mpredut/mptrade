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
    assert "cvd_15m=+10 BTC (+$680.0K, aggressive_buying)" in prompt


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

    # 4. Iteration 16: resting LIMIT without open_orders/AI fill stays OPEN_RESTING, ignores BOT fills, normalizes seconds ts_ms
    from intelligence.sentiment.guards.high_stake_guard import (
        enrich_microstructure_cvd,
        parse_venue_cycle_state,
    )
    from providers.execution_audit import build_last_cycle_execution_status, extract_intent_notional

    lim_intent = [
        {
            "intent_id": "lim_new_1",
            "timestamp": 1793300000.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 78000.0,
            "reference_price": 79000.0,
            "suggested_notional_usd": 390.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "L1", "status": "NEW", "price": "78000.0", "origQty": "0.005", "executedQty": "0.0"},
        }
    ]
    ro_bot = {"binance:BTCUSDC": {"last_buy": {"price": 78900.0, "qty": 0.01, "ts_ms": 1793300005.0, "actor": "BOT"}}}
    st_unfilled = build_last_cycle_execution_status(
        lim_intent, "crypto", 1793300060.0, extract_intent_notional, recent_orders=ro_bot, open_orders=[]
    )
    assert st_unfilled["status"] == "OPEN_RESTING"
    assert st_unfilled["fill_px"] is None
    assert st_unfilled["filled_qty"] is None

    ro_ai_sec = {"binance:BTCUSDC": {"last_buy": {"price": 77950.0, "qty": 0.005, "ts_ms": 1793300010.0, "actor": "AI"}}}
    st_filled_sec = build_last_cycle_execution_status(
        lim_intent, "crypto", 1793300060.0, extract_intent_notional, recent_orders=ro_ai_sec, open_orders=[]
    )
    assert st_filled_sec["status"] == "FILLED"
    assert st_filled_sec["fill_px"] == pytest.approx(77950.0)
    assert st_filled_sec["filled_qty"] == pytest.approx(0.005)

    # 5. enrich_microstructure_cvd derives ob_mid_price/spread_bps and uses chg_15m_pct for cvd_regime
    m_sample = {
        "best_bid": 99.9,
        "best_ask": 100.1,
        "ob_imbalance": 0.60,
        "taker_buy_vol_usd": 500_000.0,
        "taker_sell_vol_usd": 200_000.0,
    }
    enrich_microstructure_cvd(m_sample, {"price": 100.0, "chg_15m_pct": 1.4})
    assert m_sample["ob_mid_price"] == pytest.approx(100.0)
    assert m_sample["spread_bps"] == pytest.approx(20.0)
    assert m_sample["cvd_regime"] == "aggressive_buying"

    # 6. parse_venue_cycle_state omits pos_peak on qty=0 and closed cycle does not overwrite live position anchors
    closed_cyc = parse_venue_cycle_state(
        {"qty": 0.0, "cost": 0.0, "entry_price": 140.0, "pos_peak": 165.0},
        "SOLUSD",
        {"SOLUSD": {"price": 150.0}},
    )
    assert "pos_peak" not in closed_cyc
    live_pos_with_closed_cyc = [{"venue": "kraken", "symbol": "SOLUSD", "price": 150.0, "free_base": 2.0, "free_quote": 500.0}]
    harvest_recent_orders_and_enrich_positions(
        str(tmp_path), {}, live_pos_with_closed_cyc, {"SOLUSD": {"price": 150.0}}, {"SOLUSD": closed_cyc}, {}, {}
    )
    assert "active_avg_buy_px" not in live_pos_with_closed_cyc[0]
    assert "trailing_peak" not in live_pos_with_closed_cyc[0]

    # 7. Iteration 17: Hyperliquid msg='resting' / 'filled <qty> @ <px>' parsing & extract_remaining_qty without status
    from providers.base import extract_order_price, extract_remaining_qty

    assert extract_remaining_qty({"origQty": "10.0"}) == pytest.approx(10.0)
    assert extract_order_price({"orderId": 124, "ok": True, "msg": "filled 32.5 @ 61.42"}) == pytest.approx(61.42)
    assert extract_executed_qty({"orderId": 124, "ok": True, "msg": "filled 32.5 @ 61.42"}) == pytest.approx(32.5)
    assert extract_executed_qty({"orderId": 123, "ok": True, "msg": "resting"}) == 0.0

    hl_rest_st = build_last_cycle_execution_status(
        [
            {
                "intent_id": "hl_lim_1",
                "timestamp": 1793300000.0,
                "decision": "BUY",
                "symbol": "HYPEUSDC",
                "venue": "hyperliquid",
                "order_type": "LIMIT",
                "limit_price": 60.5,
                "reference_price": 61.0,
                "suggested_notional_usd": 302.5,
                "status": "EXECUTED",
                "execution_payload": {"orderId": 123, "ok": True, "msg": "resting"},
            }
        ],
        "crypto",
        1793300060.0,
        extract_intent_notional,
        open_orders=[],
    )
    assert hl_rest_st["status"] == "OPEN_RESTING"
    assert hl_rest_st["fill_px"] is None
    assert hl_rest_st["filled_qty"] is None

    hl_fill_st = build_last_cycle_execution_status(
        [
            {
                "intent_id": "hl_mkt_1",
                "timestamp": 1793300000.0,
                "decision": "BUY",
                "symbol": "HYPEUSDC",
                "venue": "hyperliquid",
                "order_type": "MARKET",
                "reference_price": 61.40,
                "suggested_notional_usd": 1996.15,
                "status": "EXECUTED",
                "execution_payload": {"orderId": 124, "ok": True, "msg": "filled 32.5 @ 61.42"},
            }
        ],
        "crypto",
        1793300060.0,
        extract_intent_notional,
        open_orders=[],
    )
    assert hl_fill_st["status"] == "FILLED"
    assert hl_fill_st["fill_px"] == pytest.approx(61.42)
    assert hl_fill_st["filled_qty"] == pytest.approx(32.5)


def test_iteration18_normalize_order_partial_fills_and_pretrade_freshest_venue(tmp_path):
    from intelligence.sentiment.guards.high_stake_guard import (
        collect_pretrade_telemetry,
        extract_symbol_recent_fills,
        harvest_recent_orders_and_enrich_positions,
    )
    from providers.base import normalize_order

    # 1. normalize_order and extract_symbol_recent_fills use executed qty on partial fills and skip resting HL orders
    norm_partial = normalize_order(
        {"orderId": "501", "side": "BUY", "price": "100.0", "qty": "10.0", "executedQty": "4.0", "time": 1000}
    )
    assert norm_partial["qty"] == pytest.approx(4.0)

    hl_fills = extract_symbol_recent_fills(
        [
            {"oid": 701, "side": "B", "limitPx": "50.0", "origSz": "10.0", "sz": "10.0", "timestamp": 1000},
            {"oid": 702, "side": "B", "limitPx": "52.0", "origSz": "10.0", "sz": "6.0", "status": "OPEN", "timestamp": 2000},
        ]
    )
    assert hl_fills is not None
    assert len(hl_fills["recent_buys"]) == 1
    assert hl_fills["last_buy"]["price"] == pytest.approx(52.0)
    assert hl_fills["last_buy"]["qty"] == pytest.approx(4.0)

    # 2. harvest_recent_orders_and_enrich_positions clears stale active_avg_buy_px when cyc_flat is True
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "cache_trade_kraken.json").write_text(
        json.dumps(
            {
                "items": {
                    "SOLUSDC": [
                        {"ordertxid": "K1", "type": "buy", "price": "150.0", "vol_exec": "2.0", "time": 1000}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    positions = [{"symbol": "SOLUSDC", "venue": "kraken", "free_base": 0.0, "free_quote": 500.0, "price": 140.0}]
    ro_map = harvest_recent_orders_and_enrich_positions(
        str(cache_dir),
        {},
        positions,
        {"SOLUSDC": {"price": 140.0}},
        {"SOLUSDC": {"qty": 0.0, "resting_orders": [], "last_buy_price": 150.0}},
        {},
        {},
    )
    assert "active_avg_buy_px" not in positions[0]
    assert "lowest_buy_px" not in positions[0]
    assert "active_avg_buy_px" not in ro_map["SOLUSDC"]

    # 3. collect_pretrade_telemetry selects the freshest venue fills across all 4 order caches
    (cache_dir / "cache_order.json").write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": [
                        {"orderId": 1, "side": "BUY", "status": "FILLED", "price": "280.0", "executedQty": "1.0", "time": 1000}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "cache_trade_hyperliquid.json").write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": [
                        {"oid": 2, "side": "A", "px": "325.0", "sz": "1.5", "time": 5000}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    tel = collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cache_dir))
    assert "Last Action=SELL" in tel.get("Recent Venue Fills", "")
    assert "325.0" in tel.get("Recent Venue Fills", "")


def test_iteration19_sibling_diverged_micro_px_and_lazy_prompt_vp_liq_formatting():
    from intelligence.sentiment.guards.high_stake_guard import sync_sibling_symbol_telemetry
    from orchestratorOS.supervisor.prompt_formatters import (
        _format_market_prices_section,
        _format_microstructure_section,
    )

    # 1. sync_sibling_symbol_telemetry does not copy USD ob_mid_price/walls onto a >10% diverged EUR sibling
    prices = {
        "BTCUSDC": {"price": 82000.0, "vol_1h_pct": 1.2},
        "BTCEUR": {"price": 74000.0, "vol_1h_pct": 1.2},
    }
    micro = {
        "BTCUSDC": {
            "ob_imbalance": 0.62,
            "ob_mid_price": 82000.0,
            "best_bid": 81995.0,
            "best_ask": 82005.0,
            "bid_wall_px": 81000.0,
            "ask_wall_px": 83000.0,
            "oi_usd": 500_000_000.0,
            "whale_long_pct": 0.58,
            "funding_rate": 0.0001,
        }
    }
    sync_sibling_symbol_telemetry(prices, micro)
    assert "BTCEUR" in micro
    assert "ob_mid_price" not in micro["BTCEUR"]
    assert micro["BTCEUR"]["long_liq_px"] < 74000.0 < micro["BTCEUR"]["short_liq_px"]

    # 2. _format_market_prices_section and _format_microstructure_section lazily compute missing VP and liq cluster metrics
    lines_3a = _format_market_prices_section(
        {"SOLUSDC": {"price": 155.0, "poc_px": 150.0, "val_px": 145.0, "vah_px": 153.0}}
    )
    joined_3a = "\n".join(lines_3a)
    assert "vol_profile(POC=150.0[+3.33%], VAL=145.0[+6.90%], VAH=153.0[+1.31%], va_pos=125.0%, regime=ABOVE_VAH)" in joined_3a

    lines_3b = _format_microstructure_section(
        {"XBTUSD": {"ob_imbalance": 0.55, "oi_usd": 200_000_000.0, "whale_long_pct": 0.60, "funding_rate": 0.0001}},
        {"BTCUSDC": {"price": 80000.0, "vol_1h_pct": 1.5}},
    )
    joined_3b = "\n".join(lines_3b)
    assert "liq_clusters(long_liq=$" in joined_3b
    assert "short_liq=$" in joined_3b


def test_iteration20_apply_asset_scope_filter_positions_and_cash():
    from intelligence.sentiment.guards.high_stake_guard import apply_asset_scope_filter

    ps_eq = {
        "tradable_positions": [
            {"venue": "binance", "symbol": "BTCUSDC"},
            {"venue": "Trading212", "symbol": "NVDA_US_EQ"},
        ],
        "venue_quote_cash": {
            "binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1200.0},
            "t212:USD": {"venue": "Trading212", "asset": "USD", "free": 850.0},
        },
        "assetguardian_state": {"BTCUSDC": {"mode": "protect"}},
        "smart_stop_loss_state": {"BTCUSDC": {"active": True}},
    }
    pt_eq, _ = apply_asset_scope_filter(
        {"BTCUSDC": {"price": 80000.0}, "NVDA_US_EQ": {"price": 135.0}},
        {},
        ps_eq,
        "equities",
        {},
        {},
    )
    assert list(pt_eq.keys()) == ["NVDA_US_EQ"]
    assert [p["symbol"] for p in ps_eq["tradable_positions"]] == ["NVDA_US_EQ"]
    assert list(ps_eq["venue_quote_cash"].keys()) == ["t212:USD"]
    assert "assetguardian_state" not in ps_eq
    assert "smart_stop_loss_state" not in ps_eq


def test_iteration21_execution_audit_kraken_xbt_variant_resolution():
    from providers.execution_audit import build_last_cycle_execution_status, extract_intent_notional

    st_fill = build_last_cycle_execution_status(
        [
            {
                "intent_id": "kr_btc_1",
                "timestamp": 1793300000.0,
                "decision": "BUY",
                "symbol": "BTCUSD",
                "venue": "kraken",
                "order_type": "LIMIT",
                "limit_price": 81500.0,
                "reference_price": 81600.0,
                "suggested_notional_usd": 815.0,
                "status": "EXECUTED",
                "execution_payload": {"status": "OPEN"},
            }
        ],
        "crypto",
        1793300060.0,
        extract_intent_notional,
        recent_orders={
            "kraken:XBTUSD": {
                "last_buy": {"price": 81490.0, "qty": 0.01, "ts_ms": 1793300010000, "actor": "AI"}
            }
        },
        open_orders=[],
    )
    assert st_fill["status"] == "FILLED"
    assert st_fill["fill_px"] == pytest.approx(81490.0)
    assert st_fill["filled_qty"] == pytest.approx(0.01)


def test_iteration22_diverged_fiat_series_overlay_and_micro_strip(tmp_path):
    from intelligence.sentiment.guards.high_stake_guard import (
        build_cross_market_context,
        enrich_24h_and_microstructure,
        sync_sibling_symbol_telemetry,
    )

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    now_ts = 1793300000.0
    now_ms = int(now_ts * 1000)

    # 1. Only BTCUSDC 24h series and orderbook exist on disk; cur_p_items has both BTCEUR (74000) and BTCUSDC (82000)
    (cache_dir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        [now_ms - 3600_000, 81500.0],
                        [now_ms - 900_000, 81800.0],
                        [now_ms - 60_000, 82000.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps(
            {
                "imbalance_ratio": 0.62,
                "mid_price": 82000.0,
                "best_bid": 81995.0,
                "best_ask": 82005.0,
                "spread_bps": 1.22,
                "bid_depth_usd": 450000.0,
                "ask_depth_usd": 300000.0,
                "largest_bid_wall_price": 81000.0,
                "largest_ask_wall_price": 83000.0,
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps({"funding_rate": 0.0001, "open_interest_usd": 500_000_000.0}),
        encoding="utf-8",
    )

    prices = {
        "BTCUSDC": {"price": 82000.0, "vol_1h_pct": 1.2},
        "BTCEUR": {"price": 74000.0, "vol_1h_pct": 1.2},
    }
    micro = {}
    cur_p_items = {"BTCEUR": 74000.0, "BTCUSDC": 82000.0}

    enrich_24h_and_microstructure(
        str(cache_dir), str(tmp_path), prices, micro, cur_p_items, {}, now_ts
    )
    sync_sibling_symbol_telemetry(prices, micro)

    # BTCEUR must not suffer a fake -9.76% crash from appending 74000 EUR onto 82000 USD series
    assert prices["BTCEUR"]["price"] == pytest.approx(74000.0)
    assert prices["BTCEUR"]["chg_15m_pct"] == pytest.approx(prices["BTCUSDC"]["chg_15m_pct"])
    assert prices["BTCEUR"]["chg_15m_pct"] > 0.0
    assert "low_24h" not in prices["BTCEUR"]
    assert "poc_px" not in prices["BTCEUR"]
    assert "ob_mid_price" not in micro["BTCEUR"]
    assert micro["BTCEUR"]["long_liq_px"] < 74000.0 < micro["BTCEUR"]["short_liq_px"]

    # 2. build_cross_market_context resolves T212 cycles keyed by base ticker via _first_variant_dict
    ctx = build_cross_market_context(
        {"NVDA_US_EQ": {"price": 138.5, "chg_1h_pct": 1.2}},
        {},
        {"NVDA": {"qty": 10.0, "avg_cost_px": 130.0, "dd_pct": 6.54}},
    )
    assert ctx["equity_bellwethers"]["NVDA_US_EQ"]["t212_cycle"]["qty"] == pytest.approx(10.0)


def test_iteration23_execution_audit_cross_venue_recent_orders_isolation(tmp_path):
    from intelligence.sentiment.guards.high_stake_guard import harvest_recent_orders_and_enrich_positions
    from providers.execution_audit import build_last_cycle_execution_status, extract_intent_notional

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    # Binance has BTCUSDC @ 80000, Kraken has XBTUSD @ 81490
    (cache_dir / "cache_order.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {"orderId": 1, "side": "BUY", "status": "FILLED", "price": "80000.0", "executedQty": "0.05", "time": 1793300010000}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "cache_trade_kraken.json").write_text(
        json.dumps(
            {
                "items": {
                    "XBTUSD": [
                        {"ordertxid": "K_AI_1", "type": "buy", "price": "81490.0", "vol_exec": "0.01", "time": 1793300010.0}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    ro_map = harvest_recent_orders_and_enrich_positions(
        str(cache_dir), {}, [], {}, {}, {}, {}, known_ai_order_ids={"1", "K_AI_1"}
    )
    assert ro_map["BTCUSDC"]["venue"] == "binance"
    assert ro_map["XBTUSD"]["venue"] == "kraken"

    st_fill = build_last_cycle_execution_status(
        [
            {
                "intent_id": "kr_btc_2",
                "timestamp": 1793300000.0,
                "decision": "BUY",
                "symbol": "BTCUSD",
                "venue": "kraken",
                "order_type": "LIMIT",
                "limit_price": 81500.0,
                "reference_price": 81600.0,
                "suggested_notional_usd": 815.0,
                "status": "EXECUTED",
                "execution_payload": {"status": "OPEN"},
            }
        ],
        "crypto",
        1793300060.0,
        extract_intent_notional,
        recent_orders=ro_map,
        open_orders=[],
    )
    assert st_fill["status"] == "FILLED"
    assert st_fill["fill_px"] == pytest.approx(81490.0)
    assert st_fill["filled_qty"] == pytest.approx(0.01)


def test_iteration24_prompt_formatters_lazy_position_limits_and_partial_liq():
    from orchestratorOS.supervisor.prompt_formatters import (
        _format_microstructure_section,
        _format_tradable_position_line,
    )

    # 1. _format_tradable_position_line populates limits(...) lazily via resolve_venue_fee_and_limits
    line_pos = _format_tradable_position_line(
        {
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "base": "HYPE",
            "quote": "USDC",
            "price": 40.0,
            "free_base": 5.0,
            "value_usd": 200.0,
            "free_quote": 100.0,
        }
    )
    assert "tier=SPOT_TIER0, maker=4bps/taker=7bps" in line_pos
    assert "limits(min_notional=$10, min_qty=0.01, step=0.01, tick=0.001, max_buy=$99.5, max_sell=$200.0)" in line_pos

    # 2. _format_microstructure_section computes missing dist_pct and never emits $None when deep_liq is absent
    lines_3b = _format_microstructure_section(
        {
            "TAOUSDC": {
                "ob_imbalance": 0.58,
                "long_liq_px": 290.0,
                "long_liq_usd": 1_500_000.0,
                "short_liq_px": 310.0,
                "short_liq_usd": 2_000_000.0,
            }
        },
        {"TAOUSDC": {"price": 300.0}},
    )
    joined_3b = "\n".join(lines_3b)
    assert "long_liq=$290.0[$1.50M, -3.33%]" in joined_3b
    assert "short_liq=$310.0[$2.00M, +3.33%]" in joined_3b
    assert "deep_long=$None" not in joined_3b
    assert "deep_short=$None" not in joined_3b
    assert "magnet=BALANCED_LIQ_MAGNET" in joined_3b


def test_iteration25_cache_symbol_variants_quote_priority_and_fiat_live_price(tmp_path):
    import json
    import math
    from types import SimpleNamespace
    from cachepaths import cache_symbol_variants
    from instruments_config import resolve_instrument_live_price

    # 1. cache_symbol_variants prioritizes same-quote cross-venue equivalents and USD family before EUR
    v_xbtusd = cache_symbol_variants("XBTUSD")
    assert v_xbtusd.index("BTCUSD") < v_xbtusd.index("XBTEUR")
    assert v_xbtusd.index("BTCUSDC") < v_xbtusd.index("XBTEUR")

    v_btceur = cache_symbol_variants("BTCEUR")
    assert v_btceur.index("XBTEUR") < v_btceur.index("BTCUSDC")

    # 2. resolve_instrument_live_price isolates quote currency families (USD vs EUR)
    (tmp_path / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": 82000.0, "XBTEUR": 74000.0}}),
        encoding="utf-8",
    )
    inst_btceur = SimpleNamespace(symbol="BTCEUR", base="BTC", quote="EUR", _provider=None)
    assert resolve_instrument_live_price(inst_btceur, str(tmp_path)) == pytest.approx(74000.0)

    inst_xbtusd = SimpleNamespace(symbol="XBTUSD", base="XBT", quote="USD", _provider=None)
    assert resolve_instrument_live_price(inst_xbtusd, str(tmp_path)) == pytest.approx(82000.0)

    # 3. Non-USD fiat instrument does not fall back to USD price when only USD cache entries exist
    (tmp_path / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": 82000.0}}),
        encoding="utf-8",
    )
    (tmp_path / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[1000, 81900.0], [2000, 82000.0]]}}),
        encoding="utf-8",
    )
    (tmp_path / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps({"symbol": "BTCUSDC", "ob_mid_price": 82005.0}),
        encoding="utf-8",
    )
    assert math.isnan(resolve_instrument_live_price(inst_btceur, str(tmp_path)))

    # 4. USD instrument does not fall back to EUR price when only EUR cache entries exist
    (tmp_path / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCEUR": 74000.0}}),
        encoding="utf-8",
    )
    (tmp_path / "cache_24_price_BTCUSDC.json").unlink()
    (tmp_path / "orderbook_depth_BTCUSDC.json").unlink()
    assert math.isnan(resolve_instrument_live_price(inst_xbtusd, str(tmp_path)))


def test_iteration26_compatible_quote_suffixes_and_unpriced_fiat_isolation(tmp_path):
    import json
    from cachepaths import compatible_quote_suffixes
    from intelligence.sentiment.guards.high_stake_guard import (
        enrich_24h_and_microstructure,
        parse_venue_cycle_state,
        resolve_symbol_price,
        sync_sibling_symbol_telemetry,
    )

    # 1. compatible_quote_suffixes and resolve_symbol_price isolate USD vs EUR valuation families
    assert compatible_quote_suffixes("BTCUSDC") == ("USDC", "USDT", "USD")
    assert compatible_quote_suffixes("BTCEUR") == ("EUR",)
    assert compatible_quote_suffixes("TLV.RO", quote="RON") == ("RON",)

    prices_mixed = {"BTCUSDC": {"price": 82000.0}, "XBTEUR": {"price": 74000.0}}
    assert resolve_symbol_price("BTCEUR", prices_mixed) == pytest.approx(74000.0)
    assert resolve_symbol_price("XBTUSD", prices_mixed) == pytest.approx(82000.0)
    assert resolve_symbol_price("BTCEUR", {"BTCUSDC": {"price": 82000.0}}) == 0.0
    assert resolve_symbol_price("XBTUSD", {"BTCEUR": {"price": 74000.0}}) == 0.0

    # 2. parse_venue_cycle_state on XBTEUR does not compute a fake +10.8% profit from BTCUSDC USD price
    cyc_eur = parse_venue_cycle_state(
        {"qty": 0.1, "cost_usd": 7400.0, "avg_cost": 74000.0},
        "XBTEUR",
        {"BTCUSDC": {"price": 82000.0}},
    )
    assert cyc_eur["dd_pct"] is None

    # 3. Unpriced BTCEUR does not inherit USD price, USD range_bound_keys, or USD liquidation clusters from BTCUSDC
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    now_ts = 1793300000.0
    (cache_dir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        [int((now_ts - 3600) * 1000), 81000.0, 10.0],
                        [int((now_ts - 900) * 1000), 81500.0, 12.0],
                        [int((now_ts - 60) * 1000), 82000.0, 15.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "imbalance_ratio": 0.62,
                "best_bid": 81990.0,
                "best_ask": 82010.0,
                "ob_mid_price": 82000.0,
                "bid_depth_usd": 500000.0,
                "ask_depth_usd": 300000.0,
            }
        ),
        encoding="utf-8",
    )
    prices = {
        "BTCUSDC": {"price": 82000.0, "vol_1h_pct": 1.2},
        "BTCEUR": {"price": None, "vol_1h_pct": 1.2},
    }
    micro = {}
    enrich_24h_and_microstructure(str(cache_dir), str(tmp_path), prices, micro, {}, {}, now_ts)
    sync_sibling_symbol_telemetry(prices, micro)

    assert prices["BTCEUR"]["price"] is None
    assert prices["BTCEUR"]["chg_1h_pct"] == pytest.approx(prices["BTCUSDC"]["chg_1h_pct"])
    assert "low_24h" not in prices["BTCEUR"]
    assert "poc_px" not in prices["BTCEUR"]
    assert "ob_mid_price" not in micro["BTCEUR"]
    assert "long_liq_px" not in micro["BTCEUR"]


def test_iteration27_position_enrichment_and_audit_quote_isolation(tmp_path):
    import json
    from unittest.mock import MagicMock
    from intelligence.sentiment.guards.high_stake_guard import (
        harvest_recent_orders_and_enrich_positions,
        harvest_tradable_positions_and_orders,
    )
    from providers.execution_audit import build_last_cycle_execution_status, extract_intent_notional

    # 1. harvest_tradable_positions_and_orders does not attach USD trailing_peak to EUR position
    eur_inst = MagicMock(symbol="XBTEUR", provider_name="kraken", base="XBT", quote="EUR", params={})
    eur_inst._provider.free_balance.side_effect = lambda a: 0.1 if a == "XBT" else 1000.0
    eur_inst.open_orders.return_value = []
    positions, _, _, inst_by_sym = harvest_tradable_positions_and_orders(
        {"XBTEUR": eur_inst},
        {"XBTEUR": {"price": 74000.0}, "XBTUSD": {"price": 82000.0}},
        {},
        {"kraken": {"XBTUSD": {"peak": 83500.0}}},
    )
    assert len(positions) == 1
    assert "trailing_peak" not in positions[0]

    # 2. harvest_recent_orders_and_enrich_positions isolates USD vs EUR recent_orders and cycle states on the same venue
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "cache_trade_kraken.json").write_text(
        json.dumps(
            {
                "items": {
                    "XBTUSD": [
                        {"ordertxid": "K_USD_1", "type": "buy", "price": "82000.0", "vol_exec": "0.1", "time": 1793300010}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    ro_map = harvest_recent_orders_and_enrich_positions(
        str(cache_dir),
        inst_by_sym,
        positions,
        {"XBTEUR": {"price": 74000.0}, "XBTUSD": {"price": 82000.0}},
        {"XBTUSD": {"qty": 0.1, "avg_cost_px": 82000.0, "last_buy_price": 82000.0, "pos_peak": 83500.0}},
        {},
        {},
    )
    assert "active_avg_buy_px" not in positions[0]
    assert "last_buy_px" not in positions[0]
    assert "cycle_avg_cost_px" not in positions[0]

    # 3. _resolve_intent_execution_details does not match XBTUSD open orders or fills onto an XBTEUR intent
    st_eur = build_last_cycle_execution_status(
        [
            {
                "timestamp": 1793300000.0,
                "venue": "kraken",
                "symbol": "XBTEUR",
                "decision": "BUY",
                "order_type": "LIMIT",
                "limit_price": 73800.0,
                "reference_price": 74000.0,
                "suggested_notional_usd": 738.0,
                "status": "EXECUTED",
                "execution_payload": {"status": "OPEN"},
            }
        ],
        "crypto",
        1793300060.0,
        extract_intent_notional,
        recent_orders=ro_map,
        open_orders=[{"venue": "kraken", "symbol": "XBTUSD", "side": "BUY", "price": 81500.0, "executedQty": 0.05}],
    )
    assert st_eur["status"] == "OPEN_RESTING"
    assert st_eur["fill_px"] is None
    assert st_eur["filled_qty"] is None


def test_iteration28_pretrade_telemetry_quote_isolation_and_mock_base_quote_derivation(tmp_path):
    import json
    from unittest.mock import MagicMock
    from intelligence.sentiment.guards.high_stake_guard import (
        collect_pretrade_telemetry,
        harvest_tradable_positions_and_orders,
    )

    # 1. harvest_tradable_positions_and_orders derives clean base/quote when inst.base/inst.quote are unset MagicMocks
    mock_inst = MagicMock(symbol="SOLUSD", provider_name="kraken", params={})
    mock_inst._provider.free_balance.side_effect = lambda a: 2.0 if a == "SOL" else 500.0
    mock_inst.open_orders.return_value = []
    positions, _, cash_map, _ = harvest_tradable_positions_and_orders(
        {"SOLUSD": mock_inst},
        {"SOLUSD": {"price": 150.0}},
        {},
        {"kraken": {"SOLUSD": {"peak": 160.0}}},
    )
    assert positions[0]["base"] == "SOL"
    assert positions[0]["quote"] == "USD"
    assert "kraken:USD" in cash_map
    assert positions[0]["trailing_peak"] == pytest.approx(160.0)

    # 2. collect_pretrade_telemetry on BTCEUR anchors liquidation clusters on EUR price and skips USD recent fills/walls
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCEUR": 74000.0, "BTCUSDC": 82000.0}}),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "imbalance_ratio": 0.60,
                "best_bid": 81990.0,
                "best_ask": 82010.0,
                "ob_mid_price": 82000.0,
                "bid_depth_usd": 600000.0,
                "ask_depth_usd": 400000.0,
                "largest_ask_wall_usd": 250000.0,
                "largest_ask_wall_price": 82500.0,
                "largest_bid_wall_usd": 220000.0,
                "largest_bid_wall_price": 81500.0,
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps({"funding_rate": 0.0001, "open_interest_usd": 100000000.0}),
        encoding="utf-8",
    )
    (cache_dir / "cache_order.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {"orderId": 1, "side": "BUY", "status": "FILLED", "price": "82000.0", "executedQty": "0.1", "time": 1000}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    telem_eur = collect_pretrade_telemetry("BTCEUR", cachedb_dir=str(cache_dir))
    assert "Orderbook Ask Wall Ahead" not in telem_eur
    assert "Recent Venue Fills" not in telem_eur
    assert "Liquidation Clusters (LONG vs SHORT)" in telem_eur
    assert telem_eur["Liquidation Clusters (LONG vs SHORT)"].startswith("LONG liq=$72")


def test_iteration29_prompt_sections_3a_and_3b_quote_family_grouping():
    from orchestratorOS.supervisor.prompt_formatters import (
        _format_market_prices_section,
        _format_microstructure_section,
    )

    # 1. Section 3A groups USD siblings (BTCUSDC, XBTUSD) separately from EUR siblings (BTCEUR, XBTEUR)
    lines_3a = _format_market_prices_section(
        {
            "BTCUSDC": {"price": 82000.0, "chg_15m_pct": 0.5, "poc_px": 81800.0, "val_px": 81200.0, "vah_px": 82300.0},
            "XBTUSD": {"price": 82010.0, "chg_15m_pct": 0.5},
            "BTCEUR": {"price": 74000.0, "chg_15m_pct": 0.5},
            "XBTEUR": {"price": 74005.0, "chg_15m_pct": 0.5},
        }
    )
    joined_3a = "\n".join(lines_3a)
    assert "BTCUSDC (sibling venue: XBTUSD=$82,010.00 [+0.01%])" in joined_3a
    assert "BTCEUR (sibling venue: XBTEUR=$74,005.00 [+0.01%])" in joined_3a
    assert "[-9.76%]" not in joined_3a
    assert "[+10.81%]" not in joined_3a

    # 2. Section 3B prioritizes USD-family microstructure (XBTUSD) over stripped non-USD entry (BTCEUR)
    lines_3b = _format_microstructure_section(
        {
            "BTCEUR": {"ob_imbalance": 0.61, "bid_depth_usd": 500000.0, "ask_depth_usd": 300000.0},
            "XBTUSD": {
                "ob_imbalance": 0.61,
                "bid_depth_usd": 500000.0,
                "ask_depth_usd": 300000.0,
                "bid_wall_px": 81500.0,
                "bid_wall_usd": 200000.0,
                "ask_wall_px": 82500.0,
                "ask_wall_usd": 250000.0,
                "oi_usd": 100000000.0,
            },
        },
        {"XBTUSD": {"price": 82000.0}, "BTCEUR": {"price": 74000.0}},
    )
    joined_3b = "\n".join(lines_3b)
    assert "* XBTUSD (BTCEUR):" in joined_3b
    assert "bid_wall=81500.0 ($200000.0, dist=-0.61%)" in joined_3b
    assert "ask_wall=82500.0 ($250000.0, dist=+0.61%)" in joined_3b


def test_iteration30_submit_decisions_scope_xstock_guard_and_instrument_quote_resolution(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from instruments_config import resolve_tradable_instrument
    from orchestratorOS.lib.technical_repair_ssot import (
        build_sensor_health_summary,
        submit_supervisor_trading_decisions,
        summarize_fleet_sensor_status,
    )

    # 1. submit_supervisor_trading_decisions blocks cross-scope and read-only Kraken xStocks
    gw = MagicMock()
    gw.submit.side_effect = lambda d, **kw: {"status": "EXECUTED", "symbol": d.get("symbol"), "venue": d.get("venue")}

    eq_spec = SimpleNamespace(can_trade=True, asset_scope="equities")
    decs_eq, _, emitted_eq, _, _ = submit_supervisor_trading_decisions(
        {
            "market_trading_decisions": [
                {"decision": "BUY", "venue": "binance", "symbol": "NVDA_US_EQ", "confidence": 0.9, "suggested_notional_usd": 100.0},
                {"decision": "BUY", "venue": "t212", "symbol": "BTCUSDC", "confidence": 0.9, "suggested_notional_usd": 100.0},
                {"decision": "BUY", "venue": "t212", "symbol": "NVDA_US_EQ", "confidence": 0.9, "suggested_notional_usd": 100.0},
            ]
        },
        eq_spec,
        topic_key="auto_sharetrade_executor",
        mode="enforce",
        now_ts=1793300000.0,
        intent_gateway=gw,
        is_rth_open_fn=lambda _: True,
    )
    assert len(decs_eq) == 1
    assert decs_eq[0]["symbol"] == "NVDA_US_EQ"
    assert decs_eq[0]["venue"] == "t212"
    assert len(emitted_eq) == 1

    crypto_spec = SimpleNamespace(can_trade=True, asset_scope="crypto")
    decs_cr, _, emitted_cr, _, _ = submit_supervisor_trading_decisions(
        {
            "market_trading_decisions": [
                {"decision": "SELL", "venue": "kraken", "symbol": "SPCXx.T", "confidence": 0.95, "suggested_notional_usd": 50.0},
                {"decision": "BUY", "venue": "t212", "symbol": "NVDA", "confidence": 0.95, "suggested_notional_usd": 50.0},
                {"decision": "BUY", "venue": "kraken", "symbol": "XBTUSD", "confidence": 0.95, "suggested_notional_usd": 150.0},
            ]
        },
        crypto_spec,
        topic_key="auto_cryptotrade_executor",
        mode="enforce",
        now_ts=1793300000.0,
        intent_gateway=gw,
        is_rth_open_fn=lambda _: True,
    )
    assert len(decs_cr) == 1
    assert decs_cr[0]["symbol"] == "XBTUSD"
    assert len(emitted_cr) == 1

    # 2. resolve_tradable_instrument isolates USD vs EUR quote families and rejects read-only Kraken xStocks
    usd_inst = MagicMock(symbol="XBTUSD", provider_name="kraken", base="XBT", quote="USD", enabled=True)
    eur_inst = MagicMock(symbol="XBTEUR", provider_name="kraken", base="XBT", quote="EUR", enabled=True)
    inst_map = {"KRAKEN_XBT": usd_inst, "KRAKEN_XBTEUR": eur_inst}

    assert resolve_tradable_instrument("BTCEUR", venue="kraken", instruments_map=inst_map) is eur_inst
    assert resolve_tradable_instrument("BTCUSDC", venue="kraken", instruments_map=inst_map) is usd_inst
    assert resolve_tradable_instrument("BTCEUR", venue="kraken", instruments_map={"KRAKEN_XBT": usd_inst}) is None
    assert resolve_tradable_instrument("SPCXx.T", venue="kraken", instruments_map=inst_map) is None

    # 3. summarize_fleet_sensor_status normalizes taxonomy aliases & build_sensor_health_summary reports macro/micro counts
    sensors_status = summarize_fleet_sensor_status(
        str(tmp_path),
        read_json_fn=lambda p: (
            {"threat_level": "CRITICAL", "risk_score": 0.9, "summary": "shock"}
            if p.endswith("geopolitical_eval.json")
            else (
                {"market_bias": "BULL", "risk_level": "ELEVATED", "action": "BUY"}
                if p.endswith("macro_sentiment_state.json")
                else None
            )
        ),
    )
    assert sensors_status["geopolitical_threat"]["threat_level"] == "CRITICAL_SHOCK"
    assert sensors_status["sentiment_advisor"]["market_bias"] == "BULLISH"
    assert sensors_status["sentiment_advisor"]["risk_level"] == "HIGH"
    assert sensors_status["sentiment_advisor"]["action"] == "ACCUMULATE"

    sh = build_sensor_health_summary(
        SimpleNamespace(
            fear_greed={"value": 50},
            market_breadth=None,
            sentiment_advisor={},
            geopolitical_threat={},
            macro_benchmarks={"SPY": {}},
            prices_and_trends={"BTCUSDC": {}},
            microstructure={"BTCUSDC": {}, "TAOUSDC": {}},
            active_heartbeats={"b1": {}},
        )
    )
    assert sh["macro_benchmarks_present"] is True
    assert sh["microstructure_count"] == 2


def test_iteration31_binance_lookup_orderbook_quote_isolation_and_audit_dedup(tmp_path):
    from cachepaths import resolve_binance_lookup_symbol
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.sentiment.guards.high_stake_guard import load_symbol_microstructure
    from providers.execution_audit import (
        build_last_cycle_execution_status,
        extract_intent_notional,
        normalize_trading_decisions,
    )

    # 1. resolve_binance_lookup_symbol converts bare base assets ('BTC', 'TAO', 'HYPE') to '<BASE>USDT'
    assert resolve_binance_lookup_symbol("BTC") == "BTCUSDT"
    assert resolve_binance_lookup_symbol("TAO") == "TAOUSDT"
    assert resolve_binance_lookup_symbol("HYPE") == "HYPEUSDT"
    assert resolve_binance_lookup_symbol("USDCUSDT") == "USDCUSDT"

    # 2. OrderbookDepthCollector isolates USD vs EUR snapshots in memory & on disk; OrderbookSnapshot.from_dict accepts ob_mid_price
    ob_col = OrderbookDepthCollector(cache_ttl_sec=60.0, disk_ttl_sec=60.0, cache_dir=str(tmp_path))
    usd_snap = OrderbookSnapshot.from_dict(
        {
            "symbol": "BTCUSDC",
            "ob_mid_price": 82000.0,
            "best_bid": 81990.0,
            "best_ask": 82010.0,
            "bid_depth_usd": 500000.0,
            "ask_depth_usd": 400000.0,
            "ob_imbalance": 0.556,
            "bid_wall_usd": 200000.0,
            "bid_wall_px": 81500.0,
            "ask_wall_usd": 250000.0,
            "ask_wall_px": 82500.0,
            "ts": 1793300000.0,
        }
    )
    assert usd_snap.mid_price == 82000.0
    assert usd_snap.largest_bid_wall_price == 81500.0
    ob_col.record_snapshot(usd_snap, save_to_disk=True)

    assert ob_col.fetch("XBTUSD", allow_network=False, now=1793300010.0) is not None
    assert ob_col.fetch("BTCEUR", allow_network=False, now=1793300010.0) is None

    # 3. load_symbol_microstructure accepts pre-normalized alias keys (ob_mid_price, bid_wall_px, oi_usd, whale_long_pct)
    m_norm = load_symbol_microstructure(
        str(tmp_path),
        "BTCUSDC",
        ob={"ob_imbalance": 0.58, "ob_mid_price": 82000.0, "bid_wall_usd": 150000.0, "bid_wall_px": 81500.0},
        deriv={"funding_rate": 0.0001, "pred_funding_rate": 0.00015, "oi_usd": 95000000.0},
        whale={"whale_long_pct": 0.62, "whale_long_ratio": 1.63, "oi_1h_chg_pct": 1.2},
    )
    assert m_norm["ob_mid_price"] == 82000.0
    assert m_norm["bid_wall_usd"] == 150000.0
    assert m_norm["bid_wall_px"] == 81500.0
    assert m_norm["oi_usd"] == 95000000.0
    assert m_norm["whale_long_pct"] == 0.62

    # 4. normalize_trading_decisions preserves distinct quote-family decisions (XBTUSD vs XBTEUR) on the same venue
    decs = normalize_trading_decisions(
        {
            "market_trading_decisions": [
                {"venue": "kraken", "symbol": "XBTUSD", "decision": "BUY", "confidence": 0.90, "suggested_notional_usd": 200.0},
                {"venue": "kraken", "symbol": "XBTEUR", "decision": "SELL", "confidence": 0.88, "suggested_notional_usd": 150.0},
            ]
        }
    )
    assert [d["symbol"] for d in decs] == ["XBTUSD", "XBTEUR"]

    # 5. _resolve_intent_execution_details does not match open_orders by order_id across different venues
    st_cross_oid = build_last_cycle_execution_status(
        [
            {
                "timestamp": 1793300000.0,
                "venue": "binance",
                "symbol": "TAOUSDC",
                "decision": "BUY",
                "order_type": "LIMIT",
                "limit_price": 300.0,
                "reference_price": 302.0,
                "suggested_notional_usd": 300.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "101", "status": "FILLED", "price": "300.0", "executedQty": "1.0"},
            }
        ],
        "crypto",
        1793300060.0,
        extract_intent_notional,
        open_orders=[{"venue": "hyperliquid", "orderId": "101", "symbol": "TAOUSDC", "side": "BUY", "price": 295.0, "executedQty": 0.0}],
    )
    assert st_cross_oid["status"] == "FILLED"
    assert st_cross_oid["fill_px"] == 300.0


def test_iteration32_resolve_assets_usdt_and_prompt_formatters_max_usd_and_eur_3b():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from orchestratorOS.supervisor.prompt_formatters import (
        _format_microstructure_section,
        _format_recent_order_line,
        build_supervisor_prompt,
    )
    from providers.quantity import clamp_notional_to_wallet_balance, resolve_assets

    # 1. resolve_assets and clamp_notional_to_wallet_balance support USDT pairs and unset MagicMock base/quote
    assert resolve_assets("BTCUSDT") == ("BTC", "USDT")
    assert resolve_assets("HYPEUSDT", MagicMock(), MagicMock()) == ("HYPE", "USDT")
    mock_usdt_inst = MagicMock(symbol="BTCUSDT", provider_name="binance")
    mock_usdt_inst._provider.free_balance.side_effect = lambda a: 500.0 if a == "USDT" else 0.0
    ok, clamped_usd, _ = clamp_notional_to_wallet_balance(mock_usdt_inst, "BUY", 82000.0, 1000.0)
    assert ok is True
    assert clamped_usd == pytest.approx(500.0 * 0.998)

    # 2. _format_microstructure_section renders separate 3B entry when non-USD pair has its own EUR orderbook anchors
    lines_3b_eur = _format_microstructure_section(
        {
            "XBTUSD": {"ob_imbalance": 0.60, "ob_mid_price": 82000.0, "bid_wall_px": 81500.0, "bid_wall_usd": 200000.0},
            "XBTEUR": {"ob_imbalance": 0.55, "ob_mid_price": 74000.0, "bid_wall_px": 73500.0, "bid_wall_usd": 150000.0},
        },
        {"XBTUSD": {"price": 82000.0}, "XBTEUR": {"price": 74000.0}},
    )
    joined_3b_eur = "\n".join(lines_3b_eur)
    assert "* XBTUSD:" in joined_3b_eur
    assert "* XBTEUR:" in joined_3b_eur
    assert "bid_wall=73500.0 ($150000.0, dist=-0.68%)" in joined_3b_eur

    # 3. build_supervisor_prompt forwards max_usd to lazy resolve_venue_fee_and_limits in Section 3C
    sensors = SimpleNamespace(
        macro_and_sentiment={},
        fear_greed=None,
        market_breadth=None,
        sentiment_advisor=None,
        geopolitical_threat=None,
        macro_benchmarks=None,
        breaking_news=[],
        prices_and_trends={"BTCUSDC": {"price": 80000.0}},
        microstructure={},
        portfolio_summary={
            "tradable_positions": [
                {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "base": "BTC",
                    "quote": "USDC",
                    "free_base": 0.05,
                    "value_usd": 4000.0,
                    "free_quote": 5000.0,
                    "price": 80000.0,
                }
            ]
        },
    )
    prompt_1k = build_supervisor_prompt(
        sensors,
        SimpleNamespace(),
        max_usd=1000.0,
        topic_key="auto_cryptotrade_executor",
    )
    assert "max_buy=$1000.0, max_sell=$1000.0" in prompt_1k

    # 4. _format_recent_order_line clamps negative age_h on slightly ahead exchange timestamps
    ro_line = _format_recent_order_line(
        "BTCUSDC",
        {
            "last_action_side": "BUY",
            "last_buy": {"qty": 0.01, "price": 82000.0, "ts_ms": 1793300010000, "actor": "AI"},
        },
        now_ms=1793300000000,
    )
    assert "age=0.0h" in ro_line


def test_iteration33_cross_market_bellwether_quote_isolation_and_collector_aliases():
    from intelligence.sentiment.guards.high_stake_guard import build_cross_market_context
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
    from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot

    # 1. build_cross_market_context prefers USD-quoted bellwether over EUR-quoted even when EUR has more keys
    ctx = build_cross_market_context(
        prices_and_trends={
            "BTCUSDC": {"price": 82000.0, "rsi": 55.0},
            "XBTEUR": {"price": 74000.0, "rsi": 54.0, "adx": 28.0, "atr_pct": 1.2, "ret_1h_pct": 0.5, "poc": 73800.0},
        },
        microstructure={
            "BTCUSDC": {"funding_rate": 0.0001, "long_liq_px": 80500.0, "short_liq_px": 83500.0},
        },
    )
    btc_bw = ctx["crypto_bellwethers"]["BTC"]
    assert btc_bw["symbol"] == "BTCUSDC"
    assert btc_bw["price"] == 82000.0
    assert btc_bw.get("long_liq_px") == 80500.0

    # And when only XBTEUR is present in prices, USD liquidation clusters are NOT grafted onto EUR bellwether
    ctx_eur_only = build_cross_market_context(
        prices_and_trends={"XBTEUR": {"price": 74000.0, "rsi": 54.0}},
        microstructure={"BTCUSDC": {"funding_rate": 0.0001, "long_liq_px": 80500.0, "short_liq_px": 83500.0}},
    )
    btc_bw_eur = ctx_eur_only["crypto_bellwethers"]["BTC"]
    assert btc_bw_eur["symbol"] == "XBTEUR"
    assert btc_bw_eur["price"] == 74000.0
    assert btc_bw_eur.get("funding_rate") == 0.0001
    assert "long_liq_px" not in btc_bw_eur
    assert "short_liq_px" not in btc_bw_eur

    # 2. DerivativesTelemetry.from_dict and WhalePositioningSnapshot.from_dict accept normalized alias keys
    dt = DerivativesTelemetry.from_dict(
        {"pred_funding_rate": -0.00025, "oi_usd": 1250000000.0, "ts": 1700000000.0},
        symbol="BTCUSDC",
    )
    assert dt is not None
    assert dt.symbol == "BTCUSDC"
    assert dt.predicted_funding_rate == -0.00025
    assert dt.open_interest_usd == 1250000000.0

    wp = WhalePositioningSnapshot.from_dict(
        {
            "whale_long_ratio": 1.5,
            "whale_long_pct": 0.60,
            "oi_usd": 980000000.0,
            "oi_1h_chg_pct": 3.25,
            "ts": 1700000000.0,
        },
        symbol="ETHUSDC",
    )
    assert wp is not None
    assert wp.symbol == "ETHUSDC"
    assert wp.top_traders_long_ratio == 1.5
    assert wp.top_traders_long_pct == 0.60
    assert wp.open_interest_usd == 980000000.0
    assert wp.open_interest_1h_change_pct == 3.25


def test_iteration34_microstructure_quote_isolation_near_parity_and_audit_reconcile(tmp_path):
    import json
    from types import SimpleNamespace
    from intelligence.sentiment.guards.high_stake_guard import (
        apply_asset_scope_filter,
        enrich_cross_venue_trends,
        load_symbol_microstructure,
        sync_sibling_symbol_telemetry,
    )
    from providers.execution_audit import (
        _resolve_intent_execution_details,
        extract_intent_notional,
        reconcile_intents_from_execution_audit,
    )

    # 1. load_symbol_microstructure strips USD price anchors when loading BTCUSDC orderbook for XBTEUR
    # even when EUR/USD is 1.05 (78,000 EUR vs 82,000 USD, <10% difference)
    ob_usd = {
        "symbol": "BTCUSDC",
        "imbalance_ratio": 0.62,
        "mid_price": 82000.0,
        "best_bid": 81995.0,
        "best_ask": 82005.0,
        "bid_depth_usd": 450000.0,
        "ask_depth_usd": 300000.0,
        "largest_bid_wall_price": 81500.0,
        "largest_bid_wall_usd": 180000.0,
        "largest_ask_wall_price": 82500.0,
        "largest_ask_wall_usd": 140000.0,
    }
    m_eur = load_symbol_microstructure(str(tmp_path), "XBTEUR", ob=ob_usd)
    assert m_eur is not None
    assert m_eur["ob_imbalance"] == 0.62
    assert m_eur.get("spread_bps") is not None
    assert m_eur["bid_wall_usd"] == 180000.0
    assert "ob_mid_price" not in m_eur
    assert "best_bid" not in m_eur
    assert m_eur.get("bid_wall_px") is None
    assert m_eur.get("ask_wall_px") is None

    # 2. sync_sibling_symbol_telemetry never copies USD price anchors onto XBTEUR even at 5% EUR/USD spread
    prices = {
        "BTCUSDC": {"price": 82000.0, "vol_1h_pct": 1.2},
        "XBTEUR": {"price": 78000.0, "vol_1h_pct": 1.2},
    }
    micro = {
        "BTCUSDC": {
            "ob_imbalance": 0.60,
            "ob_mid_price": 82000.0,
            "bid_wall_px": 81500.0,
            "oi_usd": 500000000.0,
            "funding_rate": 0.0001,
            "long_liq_px": 80500.0,
            "short_liq_px": 83500.0,
        }
    }
    sync_sibling_symbol_telemetry(prices, micro)
    assert "XBTEUR" in micro
    assert micro["XBTEUR"].get("ob_mid_price") is None
    assert micro["XBTEUR"].get("bid_wall_px") is None
    # Liquidation clusters for XBTEUR are computed around EUR 78,000, not inherited from USD 82,000
    assert micro["XBTEUR"]["long_liq_px"] < 78000.0
    assert micro["XBTEUR"]["short_liq_px"] < 81000.0

    # 3. enrich_cross_venue_trends does not overwrite XBTEUR 7d range with BTCUSDC watcher_price_items at 5% EUR/USD spread
    p_trends = {"XBTEUR": {"price": 78000.0}}
    enrich_cross_venue_trends(
        p_trends,
        {},
        {},
        {},
        {},
        {},
        {"BTCUSDC": [[1000, 81000.0], [2000, 82000.0]]},
    )
    assert "low_7d" not in p_trends["XBTEUR"]

    # 4. apply_asset_scope_filter excludes non-T212 instruments from equities prices_and_trends
    p_filtered, _ = apply_asset_scope_filter(
        {"BTCUSDC": {"price": 82000.0}, "NVDA_US_EQ": {"price": 135.0}},
        {},
        {},
        "equities",
        {"BTCUSDC": SimpleNamespace(provider_name="binance"), "NVDA_US_EQ": SimpleNamespace(provider_name="t212")},
        {},
    )
    assert "BTCUSDC" not in p_filtered
    assert "NVDA_US_EQ" in p_filtered

    # 5. reconcile_intents_from_execution_audit preserves price and keeps submit_accepted LIMIT as NEW/OPEN_RESTING
    audit_file = tmp_path / "execution_audit_20261030.jsonl"
    audit_file.write_text(
        json.dumps(
            {
                "intent_id": "intent_lim_1",
                "event": "submit_accepted",
                "order_id": "oid_77",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "price": 81500.0,
                "ts": 1700000000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    intents = [
        {
            "intent_id": "intent_lim_1",
            "status": "FAILED",
            "order_type": "LIMIT",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "decision": "BUY",
            "limit_price": 81500.0,
            "timestamp": 1700000000.0,
        }
    ]
    assert reconcile_intents_from_execution_audit(intents, str(tmp_path)) is True
    assert intents[0]["execution_payload"]["status"] == "NEW"
    assert intents[0]["execution_payload"]["price"] == 81500.0
    det = _resolve_intent_execution_details(
        intents[0],
        1700000060.0,
        extract_intent_notional,
        open_orders=[{"venue": "binance", "symbol": "ETHUSDC", "orderId": "oid_77", "filled": 0.5}],
    )
    assert det["exec_status"] == "OPEN_RESTING"
    assert det["filled_qty"] is None


def test_iteration35_kraken_xstock_discovery_filter_and_external_guard_aliases(tmp_path):
    from types import SimpleNamespace
    from instruments_config import discover_kraken_cache_pairs, resolve_tradable_instrument
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard

    # 1. discover_kraken_cache_pairs excludes read-only Kraken xStocks (.T / X.T)
    kr_dir = tmp_path / "kraken"
    kr_dir.mkdir(parents=True, exist_ok=True)
    (kr_dir / "config.env").write_text(
        "KRAKEN_CACHE_PAIRS=TAOUSD,SPCXx.T/USD,NVDAX.TUSD,XBTUSD\n",
        encoding="utf-8",
    )
    pairs = discover_kraken_cache_pairs(str(tmp_path))
    assert [p[0] for p in pairs] == ["TAOUSD", "XBTUSD"]

    # 2. resolve_tradable_instrument enforces quote compatibility on exact/variant loops even for bare symbols
    inst_eur = SimpleNamespace(symbol="BTC", base="BTC", quote="EUR", provider_name="kraken", enabled=True)
    inst_usd = SimpleNamespace(symbol="XBTUSD", base="BTC", quote="USD", provider_name="kraken", enabled=True)
    inst_xstock = SimpleNamespace(symbol="SPCXX.T/USD", base="SPCXX.T", quote="USD", provider_name="kraken", enabled=True)
    imap = {"KR_EUR": inst_eur, "KR_USD": inst_usd, "KR_X": inst_xstock}
    assert resolve_tradable_instrument("BTC", "kraken", imap) is inst_usd
    assert resolve_tradable_instrument("BTCEUR", "kraken", imap) is inst_eur
    assert resolve_tradable_instrument("SPCXX.T/USD", "kraken", imap) is None

    # 3. OrderbookWallGuard, WhaleDivergenceGuard, and FundingCrowdingGuard accept normalized telemetry aliases
    ob_guard = OrderbookWallGuard(min_buy_imbalance=0.25, whale_wall_usd_limit=1_000_000.0)
    dec_ob = ob_guard.check(
        "BTCUSDC",
        "BUY",
        snapshot={"ob_imbalance": 0.40, "ask_wall_usd": 1_500_000.0, "ask_wall_px": 83000.0},
    )
    assert not dec_ob.allowed
    assert "whale_sell_wall_blocking" in dec_ob.reason

    w_guard = WhaleDivergenceGuard(severe_cascade_oi_pct=-5.0)
    dec_w = w_guard.check(
        "BTCUSDC",
        "BUY",
        snapshot={"divergence_regime": "long_liquidation", "oi_1h_chg_pct": -6.2, "taker_buy_sell_ratio": 0.70, "whale_long_pct": 62.0},
    )
    assert not dec_w.allowed
    assert "liquidation_cascade_active" in dec_w.reason

    f_guard = FundingCrowdingGuard(max_long_funding_rate=0.0005, crowding_policy="defer")
    dec_f = f_guard.check("BTCUSDC", "BUY", snapshot={"funding_rate": 0.0008})
    assert not dec_f.allowed
    assert "long_crowding_extreme" in dec_f.reason


def test_iteration37_prompt_sibling_merge_max_usd_clamp_and_bare_btc_preflight():
    from types import SimpleNamespace
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from providers.execution_audit import normalize_trading_decisions, preflight_intent_and_resolve_price

    # 1. Section 3A & 3B merge missing telemetry from collapsed sibling entries and Section 3C clamps precomputed max_buy/max_sell to max_usd
    sensors = SimpleNamespace(
        prices_and_trends={
            "BTCUSDC": {"price": 82000.0, "chg_15m_pct": 0.25},
            "XBTUSD": {"price": 82010.0, "poc_px": 81500.0, "val_px": 81000.0, "vah_px": 82200.0},
        },
        microstructure={
            "BTCUSDC": {"ob_imbalance": 0.64, "bid_wall_px": 81500.0, "bid_wall_usd": 450000.0, "ask_wall_px": 82500.0, "ask_wall_usd": 380000.0},
            "BTCUSDT": {"funding_rate": 0.00012, "oi_usd": 95000000.0, "whale_long_pct": 0.61},
        },
        portfolio_summary={
            "tradable_positions": [
                {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "base": "BTC",
                    "quote": "USDC",
                    "free_base": 0.05,
                    "value_usd": 4100.0,
                    "free_quote": 1800.0,
                    "price": 82000.0,
                    "effective_max_buy_usd": 1800.0,
                    "effective_max_sell_usd": 2000.0,
                }
            ]
        },
    )
    prompt = build_supervisor_prompt(sensors, [], max_usd=500.0)
    assert "vol_profile(POC=81500.0" in prompt
    assert "funding=0.00012" in prompt
    assert "oi_usd=95000000.0" in prompt
    assert "max_buy=$500.0, max_sell=$500.0" in prompt

    # 2. normalize_trading_decisions uppercases/strips symbol and strips boolean symbol
    decs = normalize_trading_decisions(
        {
            "market_trading_decisions": [
                {"decision": "buy", "symbol": "  hypeusdc ", "venue": "hyperliquid", "confidence": 0.91},
                {"decision": "hold", "symbol": True, "venue": "binance"},
            ]
        }
    )
    assert decs[0]["symbol"] == "HYPEUSDC"
    assert "symbol" not in decs[1]

    # 3. preflight_intent_and_resolve_price allows bare 'BTC' USD-quoted instrument while rejecting 'TAOBTC'
    inst_btc = SimpleNamespace(symbol="BTC", base="BTC", quote="USDC", provider_name="hyperliquid", enabled=True)
    inst_taobtc = SimpleNamespace(symbol="TAOBTC", base="TAO", quote="BTC", provider_name="binance", enabled=True)
    intent_ok = {"intent_id": "i_btc", "decision": "BUY", "symbol": "BTC", "venue": "hyperliquid", "confidence": 0.92, "suggested_notional_usd": 300.0}
    ok_btc, *_ = preflight_intent_and_resolve_price(
        intent_ok,
        min_confidence=0.85,
        min_notional_usd=10.0,
        max_notional_usd=2000.0,
        resolve_instrument_fn=lambda s, venue=None, instruments_map=None: inst_btc,
        get_live_price_fn=lambda inst: 82000.0,
        instruments_map=None,
        now_ts=1700000000.0,
    )
    assert ok_btc is True

    intent_bad = {"intent_id": "i_taobtc", "decision": "BUY", "symbol": "TAOBTC", "venue": "binance", "confidence": 0.92, "suggested_notional_usd": 300.0}
    ok_bad, *_ = preflight_intent_and_resolve_price(
        intent_bad,
        min_confidence=0.85,
        min_notional_usd=10.0,
        max_notional_usd=2000.0,
        resolve_instrument_fn=lambda s, venue=None, instruments_map=None: inst_taobtc,
        get_live_price_fn=lambda inst: 0.005,
        instruments_map=None,
        now_ts=1700000000.0,
    )
    assert ok_bad is False
    assert intent_bad["status"] == "REJECTED_UNKNOWN_INSTRUMENT"


def test_iteration38_whale_oi_fallback_kraken_cache_alias_and_orderbook_from_dict(tmp_path):
    from types import SimpleNamespace
    from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
    from intelligence.sentiment.guards.high_stake_guard import (
        load_symbol_microstructure,
        resolve_venue_fee_and_limits,
    )

    # 1. load_symbol_microstructure populates oi_usd and funding_rate from whale snapshot when deriv is absent
    m_info = load_symbol_microstructure(
        str(tmp_path),
        "BTCUSDC",
        whale={
            "top_traders_long_pct": 0.63,
            "top_traders_long_ratio": 1.70,
            "taker_buy_sell_ratio": 1.25,
            "taker_buy_vol_usd": 1_200_000.0,
            "taker_sell_vol_usd": 960_000.0,
            "open_interest_usd": 85_000_000.0,
            "open_interest_1h_change_pct": 1.8,
            "divergence_regime": "accumulation",
            "funding_rate": 0.00015,
        },
    )
    assert m_info is not None
    assert m_info["oi_usd"] == 85_000_000.0
    assert m_info["funding_rate"] == 0.00015

    # 2. resolve_venue_fee_and_limits resolves Kraken 'BTCUSD' / 'HYPE/USD' against 'XBTUSD' / 'HYPEUSD' in _CACHE
    fake_cli = SimpleNamespace(
        _CACHE={
            "pair:XBTUSD": (
                1700000000.0,
                {
                    "fees_maker": [[0, 0.12]],
                    "fees": [[0, 0.22]],
                    "ordermin": "0.0002",
                    "costmin": "2.0",
                    "lot_decimals": 5,
                    "pair_decimals": 1,
                },
            )
        }
    )
    fake_inst = SimpleNamespace(params={}, _provider=SimpleNamespace(_cli=fake_cli, _minqty={"XBTUSD": 0.0002}))
    fees = resolve_venue_fee_and_limits(
        "kraken", "BTCUSD", fake_inst, price=82000.0, free_quote=1500.0, gateway_max_usd=None
    )
    assert fees["maker_fee_bps"] == 12.0
    assert fees["taker_fee_bps"] == 22.0
    assert fees["min_qty"] == 0.0002
    assert fees["exchange_min_notional_usd"] == 2.0
    assert fees["step_size"] == 1e-05
    assert fees["tick_size"] == 0.1
    assert fees["effective_max_buy_usd"] == round(1500.0 * 0.995, 2)

    # 3. OrderbookSnapshot.from_dict accepts optional symbol fallback
    ob_snap = OrderbookSnapshot.from_dict({"mid_price": 82000.0, "imbalance_ratio": 0.58}, symbol="BTCUSDC")
    assert ob_snap.symbol == "BTCUSDC"


def test_iteration39_cross_quote_trend_price_isolation_resolve_assets_and_bare_base_price(tmp_path):
    from types import SimpleNamespace
    from intelligence.sentiment.guards.high_stake_guard import (
        enrich_cross_venue_trends,
        harvest_recent_orders_and_enrich_positions,
        harvest_tradable_positions_and_orders,
        resolve_symbol_price,
    )

    # 1. enrich_cross_venue_trends inherits dimensionless slopes from BTCUSDC for XBTEUR without copying USD current_price
    pt = {
        "XBTEUR": {"price": None, "trend": 0, "growth_coefficient": 0.0},
        "BTCUSD": {"price": None, "trend": 0, "growth_coefficient": 0.0},
    }
    trends_data = {
        "items": {
            "BTCUSDC": {
                "current_price": 82000.0,
                "final_trend": 1,
                "slope_full": 0.0012,
                "dynamic": {
                    "222": {"slope_full": 0.0012, "final_trend": 1},
                    "900": {"slope_full": 0.0008, "final_trend": 1},
                    "9000": {"slope_full": 0.0004, "final_trend": 1},
                },
            }
        }
    }
    enrich_cross_venue_trends(pt, trends_data, {}, {}, {}, {}, {})
    assert pt["XBTEUR"]["slope_3_7m"] == 0.0012
    assert pt["XBTEUR"]["price"] is None  # Must not leak USD price into EUR entry
    assert pt["BTCUSD"]["price"] == 82000.0  # Compatible USD-family quote updates price

    # 2. resolve_symbol_price resolves bare base key ('HYPE') for USD-family pairs ('HYPEUSDC') but not EUR pairs ('XBTEUR')
    assert resolve_symbol_price("HYPEUSDC", {"HYPE": {"price": 24.5}}) == 24.5
    assert resolve_symbol_price("XBTEUR", {"BTC": {"price": 82000.0}}) == 0.0
    assert resolve_symbol_price("BTC", {"BTCUSDC": {"price": 82000.0}}, explicit_quote="EUR") == 0.0

    # 3. harvest_tradable_positions_and_orders reuses resolve_assets (preserving T212 full ticker base)
    t212_inst = SimpleNamespace(
        symbol="NVDA_US_EQ",
        provider_name="t212",
        base=None,
        quote=None,
        params={},
        _provider=SimpleNamespace(
            get_current_price=lambda s: 120.0,
            free_balance=lambda a: 2.0 if a == "NVDA_US_EQ" else (500.0 if a == "USD" else 0.0),
        ),
        open_orders=lambda: [],
    )
    positions, _, _, _ = harvest_tradable_positions_and_orders(
        {"T212_NVDA": t212_inst},
        {},
        {},
        {},
    )
    assert len(positions) == 1
    assert positions[0]["base"] == "NVDA_US_EQ"
    assert positions[0]["quote"] == "USD"
    assert positions[0]["free_base"] == 2.0

    # 4. harvest_recent_orders_and_enrich_positions matches bare Hyperliquid fills ('HYPE') for 'HYPEUSDC' position
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "cache_trade_hyperliquid.json").write_text(
        '{"items": {"HYPE": [{"oid": 11, "side": "B", "px": "22.0", "sz": "5.0", "time": 1700000000000}]}}',
        encoding="utf-8",
    )
    hl_pos = [
        {
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "base": "HYPE",
            "quote": "USDC",
            "price": 0.0,
            "free_base": 5.0,
            "free_quote": 100.0,
        }
    ]
    harvest_recent_orders_and_enrich_positions(
        str(cache_dir),
        {},
        hl_pos,
        {"HYPE": {"price": 24.5}},
        {},
        {},
        {},
    )
    assert hl_pos[0]["price"] == 24.5
    assert hl_pos[0]["active_avg_buy_px"] == 22.0
    assert hl_pos[0]["effective_max_sell_usd"] == 122.5


def test_iteration40_monolithic_cache24_isolation_and_bare_base_series_micro_pretrade(tmp_path):
    import json
    from intelligence.sentiment.guards.high_stake_guard import (
        collect_pretrade_telemetry,
        load_symbol_microstructure,
        load_symbol_price_series,
    )

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    # Monolithic legacy cache_24_price.json containing only BTCUSDC must not leak into SOLUSDC or XBTEUR
    (cache_dir / "cache_24_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[1700000000000, 82000.0], [1700003600000, 82500.0]]}}),
        encoding="utf-8",
    )
    sol_24h, _ = load_symbol_price_series(str(cache_dir), "SOLUSDC", include_7d=False)
    assert sol_24h == []
    eur_24h, _ = load_symbol_price_series(str(cache_dir), "XBTEUR", include_7d=False)
    assert eur_24h == []

    # Bare-base Hyperliquid files ('cache_24_price_HYPE.json', 'orderbook_depth_HYPE.json', 'cache_short_trend.json', 'cache_trade_hyperliquid.json')
    # resolve cleanly for 'HYPEUSDC' while keeping EUR pairs isolated
    (cache_dir / "cache_24_price_HYPE.json").write_text(
        json.dumps({"items": {"HYPE": [[1700000000000, 24.0, 100.0], [1700003600000, 25.2, 150.0]]}}),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_HYPE.json").write_text(
        json.dumps({
            "symbol": "HYPE",
            "mid_price": 25.2,
            "best_bid": 25.19,
            "best_ask": 25.21,
            "imbalance_ratio": 0.62,
            "bid_depth_usd": 150000.0,
            "ask_depth_usd": 90000.0,
        }),
        encoding="utf-8",
    )
    (cache_dir / "cache_short_trend.json").write_text(
        json.dumps({
            "items": {
                "HYPE": {
                    "current_price": 25.2,
                    "final_trend": 1,
                    "slope_full": 0.0025,
                    "dynamic": {
                        "222": {"slope_full": 0.0025, "final_trend": 1},
                        "900": {"slope_full": 0.0018, "final_trend": 1},
                        "9000": {"slope_full": 0.0009, "final_trend": 1},
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    (cache_dir / "cache_trade_hyperliquid.json").write_text(
        json.dumps({
            "items": {
                "HYPE": [
                    {"oid": 77, "side": "B", "px": "23.5", "sz": "10.0", "time": 1700001000000}
                ]
            }
        }),
        encoding="utf-8",
    )

    hype_24h, _ = load_symbol_price_series(str(cache_dir), "HYPEUSDC", include_7d=False)
    assert len(hype_24h) == 2
    assert hype_24h[-1][1] == 25.2

    m_hype = load_symbol_microstructure(str(cache_dir), "HYPEUSDC")
    assert m_hype is not None
    assert m_hype["ob_mid_price"] == 25.2
    assert m_hype["ob_imbalance"] == 0.62

    telem = collect_pretrade_telemetry("HYPEUSDC", cachedb_dir=str(cache_dir))
    assert "Dynamic Multi-Window Slopes" in telem
    assert "+0.00250" in telem["Dynamic Multi-Window Slopes"]
    assert "Recent Venue Fills" in telem
    assert "23.5" in telem["Recent Venue Fills"]


def test_intraday_6h_bottom_structure_and_smt_divergence_formatting():
    from intelligence.sentiment.guards.high_stake_guard import compute_24h_price_metrics
    from orchestratorOS.supervisor.prompt_formatters import _format_market_prices_section

    t0 = 1_793_300_000_000
    hr = 3_600_000
    # Simulate a 2-wave overnight crash:
    # - BTC: drops from 80,500 (-5h) to 76,740 (-2h), then sweeps 76,387 (-25m) and reclaims 77,208 (now)
    btc_series = [
        [t0 - 5 * hr, 80500.0],
        [t0 - 3 * hr, 78900.0],
        [t0 - 2 * hr, 76739.78],
        [t0 - 90 * 60000, 77400.0],
        [t0 - 25 * 60000, 76387.17],
        [t0, 77208.86],
    ]
    # - TAO: drops from 250.0 (-5h) to 220.0 (-2h), then holds a higher low at 226.1 (-25m) and bounces to 231.8 (now)
    tao_series = [
        [t0 - 5 * hr, 250.0],
        [t0 - 3 * hr, 238.0],
        [t0 - 2 * hr, 220.0],
        [t0 - 90 * 60000, 229.0],
        [t0 - 25 * 60000, 226.1],
        [t0, 231.8],
    ]

    btc_m = compute_24h_price_metrics(btc_series)
    tao_m = compute_24h_price_metrics(tao_series)

    assert btc_m["bottom_structure_6h"] == "LIQUIDITY_SWEEP_RECLAIM"
    assert btc_m["prior_6h_low_px"] == 76739.78
    assert btc_m["recent_1h_low_px"] == 76387.17
    assert btc_m["reclaim_above_prior_low_pct"] == 0.61

    assert tao_m["bottom_structure_6h"] == "HIGHER_LOW_DOUBLE_BOTTOM"
    assert tao_m["prior_6h_low_px"] == 220.0
    assert tao_m["recent_1h_low_px"] == 226.1
    assert tao_m["higher_low_vs_prior_pct"] == 2.77

    lines = _format_market_prices_section({
        "BTCUSDC": {"price": 77208.86, **btc_m},
        "TAOUSDC": {"price": 231.8, **tao_m},
    })
    joined = "\n".join(lines)
    assert "bottom_6h=LIQUIDITY_SWEEP_RECLAIM(prior_low=76739.78, 1h_low=76387.17, reclaim=+0.61%)" in joined
    assert (
        "bottom_6h=HIGHER_LOW_DOUBLE_BOTTOM(prior_low=220.0, 1h_low=226.1, hl_diff=+2.77%, smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE)"
        in joined
    )


def test_all_ai_agent_modules_propagate_6h_bottom_smt_and_contrarian_rules(tmp_path):
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard, collect_pretrade_telemetry
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from state_io import save_json_state

    t0 = 1_793_300_000_000
    hr = 3_600_000
    save_json_state(
        str(tmp_path / "cache_24_price_BTCUSDC.json"),
        {
            "items": {
                "BTCUSDC": [
                    [t0 - 5 * hr, 80500.0],
                    [t0 - 3 * hr, 78900.0],
                    [t0 - 2 * hr, 76739.78],
                    [t0 - 90 * 60000, 77400.0],
                    [t0 - 25 * 60000, 76387.17],
                    [t0, 77208.86],
                ]
            }
        },
    )
    save_json_state(
        str(tmp_path / "cache_24_price_TAOUSDC.json"),
        {
            "items": {
                "TAOUSDC": [
                    [t0 - 5 * hr, 250.0],
                    [t0 - 3 * hr, 238.0],
                    [t0 - 2 * hr, 220.0],
                    [t0 - 90 * 60000, 229.0],
                    [t0 - 25 * 60000, 226.1],
                    [t0, 231.8],
                ]
            }
        },
    )

    telem = collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(tmp_path))
    assert "6h Intraday Bottom Structure" in telem
    assert "HIGHER_LOW_DOUBLE_BOTTOM" in telem["6h Intraday Bottom Structure"]
    assert "smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE" in telem["6h Intraday Bottom Structure"]

    captured_prompts = {}

    class _DummyClient:
        _custom_runner = True

        def __init__(self, key, resp):
            self.key = key
            self.resp = resp

        def query_json(self, prompt, **kwargs):
            captured_prompts[self.key] = prompt
            return self.resp

    # 1. HighStakeGuard (BUY & SELL)
    hs_guard = HighStakeGuard(
        gemini_client=_DummyClient(
            "hs_buy",
            {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Verified 6h double bottom"},
        ),
        min_notional_eur=100.0,
        cache_ttl_sec=0.0,
    )
    hs_guard.check("TAOUSDC", "BUY", 2.0, 231.8, telemetry=telem, notional_eur=463.6)
    assert "HIGHER_LOW_DOUBLE_BOTTOM" in captured_prompts["hs_buy"]
    assert "smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE" in captured_prompts["hs_buy"]
    assert "CONTRARIAN BOTTOM ACCUMULATION" in captured_prompts["hs_buy"]

    hs_sell_guard = HighStakeGuard(
        gemini_client=_DummyClient(
            "hs_sell",
            {"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Do not sell into 6h double bottom"},
        ),
        min_notional_eur=100.0,
        cache_ttl_sec=0.0,
    )
    hs_sell_guard.check("TAOUSDC", "SELL", 2.0, 231.8, telemetry=telem, notional_eur=463.6)
    assert "CAPITULATION & SQUEEZE PROTECTION (6H DOUBLE-BOTTOM & RECLAIM DEFENSE)" in captured_prompts["hs_sell"]

    # 2. PostStopReentryGuard
    ps_guard = PostStopReentryGuard(
        gemini_client=_DummyClient(
            "post_stop",
            {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Confirmed 6h higher low"},
        ),
        notify_enabled=False,
        cache_ttl_sec=0.0,
    )
    ps_guard.check(
        "TAOUSDC",
        current_price=231.8,
        last_sell_price=230.0,
        trough=226.1,
        bounce_pct=0.025,
        qty=2.0,
        telemetry=telem,
        notional_eur=463.6,
    )
    assert "GENUINE ACCUMULATION & 6H BOTTOM STRUCTURE CONFIRMATION" in captured_prompts["post_stop"]
    assert "HIGHER_LOW_DOUBLE_BOTTOM" in captured_prompts["post_stop"]

    # 3. SentimentAdvisor
    advisor = SentimentAdvisor(
        gemini_client=_DummyClient(
            "sentiment",
            {
                "market_bias": "NEUTRAL",
                "risk_level": "MODERATE",
                "confidence": 0.82,
                "summary": "Intraday liquidity sweep reclaim underway.",
                "key_risks": ["Residual volatility"],
                "recommended_action": "ACCUMULATE",
            },
        ),
        cache_file=str(tmp_path / "sentiment_eval.json"),
    )
    advisor.review(
        fear_greed={"value": 25, "classification": "Extreme Fear", "trend_7d_change": -10},
        breadth={
            "advance_ratio": 0.146,
            "median_change_pct": -3.59,
            "regime": "BEARISH_BREADTH",
        },
        whale_telemetry={"BTCUSDC": {"divergence_regime": "accumulation", "taker_buy_sell_ratio": 1.14}},
        force_refresh=True,
        now_ts=t0 / 1000.0,
    )
    assert "BTC Intraday Structure:" in captured_prompts["sentiment"]
    assert "bottom_structure_6h=LIQUIDITY_SWEEP_RECLAIM" in captured_prompts["sentiment"]
    assert "LAGGING 24H BREADTH VS INTRADAY BOTTOM ACCUMULATION" in captured_prompts["sentiment"]

    # 4. GeopoliticalThreatAnalyzer
    geo = GeopoliticalThreatAnalyzer(
        gemini_client=_DummyClient(
            "geo",
            {
                "threat_level": "ELEVATED",
                "risk_score": 0.55,
                "summary": "Ongoing priced-in headlines.",
                "recommended_brake": "DOWNSCALE_50",
            },
        ),
        state_file=str(tmp_path / "geopolitical_threat_eval.json"),
    )
    geo.assess(
        {
            "headlines": [
                {
                    "title": "Ongoing diplomatic statements on regional sanctions and conflict",
                    "source": "Reuters",
                    "published_ts": t0 / 1000.0,
                    "matched_keywords": ["sanctions", "conflict"],
                    "is_high_severity": True,
                }
            ],
            "high_severity_count": 1,
            "has_critical_shock_keywords": False,
            "ts": t0 / 1000.0,
        },
        force_refresh=True,
        now_ts=t0 / 1000.0,
    )
    assert "CRITICAL CALIBRATION — FRESH UNPRICED SHOCK VS ONGOING / PRICED-IN HEADLINES" in captured_prompts["geo"]


def test_per_topic_argon_high_vs_medium_routing_and_120s_timeouts():
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
    from orchestratorOS.agy.topics import topic_spec

    # 1. High-stake / rare topics use gemini-4-argon-high and 120.0s timeout
    for high_topic in (
        "auto_technical_repair",
        "macro_geopolitical",
        "macro_sentiment",
        "guard_high_stake",
        "guard_post_stop",
    ):
        spec = topic_spec(high_topic)
        assert spec.preferred_model == "gemini-4-argon-high"
        assert spec.default_timeout_sec == 120.0

    # 2. 10-minute executors use gemini-4-argon-medium and 90.0s timeout
    for exec_topic in ("auto_cryptotrade_executor", "auto_sharetrade_executor"):
        spec = topic_spec(exec_topic)
        assert spec.preferred_model == "gemini-4-argon-medium"
        assert spec.default_timeout_sec == 90.0

    # 3. GeminiClient fallback chain deduplicates primary model cleanly
    observed = {}

    def _runner(prompt, model, timeout):
        observed["model"] = model
        observed["timeout"] = timeout
        return '{"ok": true}'

    client = GeminiClient(custom_runner=_runner, project_id="autonommptrade")
    high_chain = client.get_model_candidates("gemini-4-argon-high")
    assert high_chain[:6] == [
        "gemini-4-argon-high",
        "gemini-4-argon-medium",
        "gemini-4-argon-low",
        "gemini-3.8-flash-medium",
        "gpt-oss-120b-medium",
        "claude-sonnet-5-5-medium",
    ]
    assert "claude-sonnet-5-5-low" in high_chain

    med_chain = client.get_model_candidates("gemini-4-argon-medium")
    assert med_chain[:5] == [
        "gemini-4-argon-medium",
        "gemini-4-argon-low",
        "gemini-3.8-flash-medium",
        "gpt-oss-120b-medium",
        "claude-sonnet-5-5-medium",
    ]

    client.query_json("test guard", topic_key="guard_high_stake", timeout_sec=120.0)
    assert observed["model"] == "gemini-4-argon-high"
    assert observed["timeout"] == 120.0

    client.query_json("test exec", topic_key="auto_cryptotrade_executor", timeout_sec=90.0)
    assert observed["model"] == "gemini-4-argon-medium"
    assert observed["timeout"] == 90.0

    assert HighStakeGuard(gemini_client=client).timeout_sec == 120.0
    assert PostStopReentryGuard(gemini_client=client, notify_enabled=False).timeout_sec == 120.0


def test_six_prompt_telemetry_and_news_lexicon_refinements(tmp_path):
    """Verify the 6 telemetry & news lexicon refinements:
    1. Sub-$1 coin (ARBUSDC) slopes, g_coeff, grad_fast/eps scientific notation when < 1e-4.
    2. cvd_15m regime correlated with chg_15m_pct and CVD sign (ask_absorbing vs spot_absorbing vs aggressive_buying vs aggressive_market_dumps).
    3. Section 3B ob_mid display and bid_wall/ask_wall dist anchored to ob_mid.
    4. Section 3C pnl_vs_avg=+X.XX% when positive vs dd=-X.XX% when negative, plus dd_from_peak on pos_peak.
    5. build_last_cycle_execution_status preserves active OPEN_RESTING limit orders from earlier cycles on other symbols.
    6. News severity keyword filter strips generic stopwords, publisher names, and substring stem false positives.
    """
    from intelligence.macro.news_feed_collector import NewsFeedCollector
    from intelligence.macro_sentiment_ssot import (
        discover_dynamic_terms_from_headlines,
        filter_valid_severity_keywords,
    )
    from intelligence.sentiment.guards.high_stake_guard import (
        enrich_microstructure_cvd,
        harvest_macro_and_news_sensors,
        parse_venue_cycle_state,
    )
    from orchestratorOS.supervisor.prompt_formatters import (
        _format_market_prices_section,
        _format_microstructure_section,
        _format_venue_cycle_line,
    )
    from providers.execution_audit import build_last_cycle_execution_status, extract_intent_notional

    # 1. Sub-$1 coin slopes & g_coeff scientific formatting (< 1e-4)
    lines_3a = _format_market_prices_section(
        {
            "ARBUSDC": {
                "price": 0.163,
                "trend": 0,
                "slope_90s": -8.04e-6,
                "slope_3_7m": 1.12e-5,
                "slope_15m": -1.50e-6,
                "slope_2_5h": 1.74e-7,
                "gradient_recent_fast": 4.20e-6,
                "epsilon": 3.50e-5,
                "growth_90s": -8.04e-6,
                "growth_3_7m": 1.12e-5,
                "growth_15m": -1.50e-6,
                "growth_2_5h": 1.74e-7,
                "kalman_velocity": 0.14083,
            }
        }
    )
    joined_3a = "\n".join(lines_3a)
    assert "slopes(90s=-8.04e-06, 3.7m=+1.12e-05, 15m=-1.50e-06, 2.5h=+1.74e-07, grad_fast=+4.20e-06/eps=3.50e-05)" in joined_3a
    assert "g_coeff(90s=-8.04e-06, 3.7m=+1.12e-05, 15m=-1.50e-06, 2.5h=+1.74e-07)" in joined_3a

    # 2. cvd_15m regime correlation with chg_15m_pct and CVD sign
    m_pos_rally = {"cvd_15m_usd": 758_000.0, "ob_imbalance": 0.62}
    enrich_microstructure_cvd(m_pos_rally, {"price": 0.25, "chg_15m_pct": 0.18})
    assert m_pos_rally["cvd_regime"] == "aggressive_buying"

    m_pos_stalled = {"cvd_15m_usd": 360_500.0, "ob_imbalance": 0.58}
    enrich_microstructure_cvd(m_pos_stalled, {"price": 0.163, "chg_15m_pct": -0.05})
    assert m_pos_stalled["cvd_regime"] == "ask_absorbing"

    m_neg_rally = {"cvd_15m_usd": -3_870_000.0, "ob_imbalance": 0.48, "div_regime": "short_covering"}
    enrich_microstructure_cvd(m_neg_rally, {"price": 93.0, "chg_15m_pct": 0.74})
    assert m_neg_rally["cvd_regime"] == "spot_absorbing"

    m_neg_dump = {"cvd_15m_usd": -1_200_000.0, "ob_imbalance": 0.42}
    enrich_microstructure_cvd(m_neg_dump, {"price": 90.0, "chg_15m_pct": -0.65})
    assert m_neg_dump["cvd_regime"] == "aggressive_market_dumps"

    # 3. Section 3B ob_mid display and bid_wall/ask_wall dist anchored to ob_mid
    lines_3b = _format_microstructure_section(
        {
            "BTCUSDC": {
                "ob_imbalance": 0.55,
                "spread_bps": 0.0013,
                "best_bid": 77746.99,
                "best_ask": 77747.00,
                "ob_mid_price": 77746.995,
                "bid_wall_px": 77746.99,
                "bid_wall_usd": 120000.0,
                "ask_wall_px": 77900.0,
                "ask_wall_usd": 150000.0,
                "cvd_15m_usd": 758000.0,
                "cvd_15m_base": 9.75,
                "cvd_regime": "spot_absorbing",
            }
        },
        {"BTCUSDC": {"price": 77660.41, "chg_15m_pct": 0.15}},
    )
    joined_3b = "\n".join(lines_3b)
    assert "spread=0.0013bps (bid=77746.99/ask=77747.0), ob_mid=77746.995" in joined_3b
    assert "bid_wall=77746.99 ($120000.0, dist=-0.00%)" in joined_3b
    assert "cvd_15m=+9.75 BTC (+$758.0K, aggressive_buying)" in joined_3b

    # 4. Section 3C pnl_vs_avg=+6.89% when positive vs dd=-17.89% when negative + dd_from_peak on pos_peak
    cyc_hl = parse_venue_cycle_state(
        {"qty": 0.90913, "cost_usd": 79.19, "avg_cost": 87.109854, "pos_peak": 95.0, "realized_pnl": 179.25},
        "HYPEUSDC",
        {"HYPEUSDC": {"price": 93.11}},
    )
    assert cyc_hl["pnl_vs_avg_pct"] == pytest.approx(6.89, abs=0.01)
    assert cyc_hl["dd_from_peak_pct"] == pytest.approx(-1.99, abs=0.01)
    line_hl = _format_venue_cycle_line("hyperliquid", "HYPEUSDC", cyc_hl)
    assert "pnl_vs_avg=+6.89%" in line_hl
    assert "pos_peak=95.0 (dd_from_peak=-1.99%)" in line_hl

    line_krk = _format_venue_cycle_line(
        "kraken",
        "TAOUSD",
        {"qty": 1.5, "cost_usd": 500.0, "avg_cost_px": 333.33, "dd_pct": -17.89},
    )
    assert "dd=-17.89%" in line_krk

    # 5. build_last_cycle_execution_status preserves resting open order on HYPEUSDC when a later cycle updates BTCUSDC
    intents_multi = [
        {
            "intent_id": "hl_hype_1",
            "timestamp": 1793300000.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 91.5,
            "reference_price": 92.0,
            "suggested_notional_usd": 50.0,
            "status": "EXECUTED",
            "execution_payload": {"status": "RESTING", "oid": "1001"},
        },
        {
            "intent_id": "hl_btc_1",
            "timestamp": 1793300010.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 76500.0,
            "reference_price": 77000.0,
            "suggested_notional_usd": 100.0,
            "status": "EXECUTED",
            "execution_payload": {"status": "RESTING", "oid": "1002"},
        },
        {
            "intent_id": "hl_btc_2",
            "timestamp": 1793300660.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 76800.0,
            "reference_price": 77100.0,
            "suggested_notional_usd": 100.0,
            "status": "EXECUTED",
            "execution_payload": {"status": "RESTING", "oid": "1003"},
        },
    ]
    open_ords = [
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "side": "BUY", "order_id": "1001", "price": 91.5},
        {"venue": "hyperliquid", "symbol": "BTCUSDC", "side": "BUY", "order_id": "1003", "price": 76800.0},
    ]
    st_carried = build_last_cycle_execution_status(
        intents_multi,
        "crypto",
        1793300700.0,
        extract_intent_notional,
        recent_orders={},
        open_orders=open_ords,
    )
    assert isinstance(st_carried, list)
    assert len(st_carried) == 2
    assert [x["symbol"] for x in st_carried] == ["HYPEUSDC", "BTCUSDC"]
    assert [x["order_id"] for x in st_carried] == ["1001", "1003"]
    assert all(x["status"] == "OPEN_RESTING" for x in st_carried)

    # 6. News severity keyword filter strips generic lifestyle stopwords & substring stem collisions
    assert list(filter_valid_severity_keywords(["more than", "was heartbroken", "leave everything", "financial times", "new york", "yahoo finance"])) == []
    assert list(filter_valid_severity_keywords(["more than", "sanctions"])) == ["sanctions"]

    collector = NewsFeedCollector(history_file=str(tmp_path / "news_hist.json"))
    coerced_lifestyle = collector._coerce_headline(
        {
            "title": "Gen Z is ditching bars for run clubs — and finding more than friends along the way",
            "source": "Yahoo",
            "is_high_severity": True,
            "matched_keywords": ["more than"],
            "first_seen_ts": 1793300000.0,
        }
    )
    assert coerced_lifestyle is not None
    assert coerced_lifestyle.is_high_severity is False
    assert list(coerced_lifestyle.matched_keywords) == []

    # Substring stem collision ('ban' in 'husband'/'abandoned') must not discover lifestyle bigrams
    discovered = discover_dynamic_terms_from_headlines(
        [
            "He was heartbroken: My husband's kids abandoned him when he was dying - Financial Times",
            "She was heartbroken after her husband abandoned the house - Yahoo Finance",
        ]
    )
    assert "was heartbroken" not in discovered
    assert "financial times" not in discovered
    assert "yahoo finance" not in discovered
    assert "more than" not in discovered

    # harvest_macro_and_news_sensors demotes cached false-alarm [HIGH-SEV] items
    import json
    (tmp_path / "news_feed_collect.json").write_text(
        json.dumps(
            [
                {
                    "title": "What killed Weil Gotshal? - Financial Times",
                    "source": "FT",
                    "is_high_severity": True,
                    "matched_keywords": ["financial times"],
                    "first_seen_ts": 1793300000.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    _, b_news = harvest_macro_and_news_sensors(str(tmp_path), 1793300060.0)
    assert len(b_news) == 1
    assert b_news[0]["is_high_severity"] is False
    assert b_news[0]["keywords"] == []

    # Corporate M&A 'strikes ... deal' idiom must not discover dynamic crisis bigrams
    corp_discovered = discover_dynamic_terms_from_headlines(
        [
            {"title": "DoorDash strikes $17 billion deal for US rival Instacart in mega grocery delivery bet - Reuters", "source": "Reuters"},
            {"title": "DoorDash strikes $17 billion deal for rival Instacart in grocery delivery - CNBC", "source": "CNBC"},
        ],
        persist=False,
    )
    assert corp_discovered == ()

    # 7. enrich_cross_venue_trends keeps adapt_reentry_pct/adapt_dca_pct aligned with fresh vol_1h_pct
    from intelligence.sentiment.guards.high_stake_guard import (
        enrich_cross_venue_trends,
        harvest_tradable_positions_and_orders,
    )
    pt_hype = {"HYPEUSDC": {"price": 91.50, "vol_1h_pct": 1.62}}
    enrich_cross_venue_trends(
        pt_hype,
        trends_data={"items": {"HYPEUSDC": {"vol_1h_pct": 1.62}}},
        kalman_data={"HYPE": {"kalman_vel": 0.08, "kalman_vel_std": 0.06, "kalman_trend": 1, "vol_1h_pct": 0.316, "adapt_reentry_pct": 0.632, "adapt_dca_pct": 0.316}},
        long_items={},
        priceanalysis_data={},
        t_trend_map={},
        watcher_price_items={},
    )
    assert pt_hype["HYPEUSDC"]["vol_1h_pct"] == 1.62
    assert pt_hype["HYPEUSDC"]["adapt_reentry_pct"] == 3.24
    assert pt_hype["HYPEUSDC"]["adapt_dca_pct"] == 1.62

    # 8. harvest_tradable_positions_and_orders falls back to cached venue_balances_state.json when API fails
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "venue_balances_state.json").write_text(
        json.dumps({"binance:BTC": 0.0085, "binance:USDC": 1450.25}),
        encoding="utf-8",
    )

    class _DownProv:
        def free_balance(self, _asset):
            return None

    class _DownInst:
        symbol = "BTCUSDC"
        provider_name = "binance"
        base = "BTC"
        quote = "USDC"
        params = {}
        _provider = _DownProv()

    pos_fb, _, cash_fb, _ = harvest_tradable_positions_and_orders(
        {"BTCUSDC": _DownInst()},
        {"BTCUSDC": {"price": 77500.0}},
        {},
        {},
        cache_dir=str(tmp_path),
    )
    assert len(pos_fb) == 1
    assert pos_fb[0]["free_base"] == 0.0085
    assert pos_fb[0]["free_quote"] == 1450.25
    assert cash_fb["binance:USDC"]["free"] == 1450.25


def test_iteration40_kraken_bot_fallback_unfilled_limit_status_dust_peak_and_retry_ttl(tmp_path):
    """Verify:
    1. KrakenProvider open_orders / order_status fall back to BOT client when auto-created SPARE client raises Permission denied or returns empty {}.
    2. _resolve_intent_execution_details treats Kraken add_order payload without explicit 'status' as OPEN_RESTING (not FILLED), and marks >10m missing limit orders as CANCELED_OR_EXPIRED.
    3. Dust balances below venue min_notional_usd ($1.08 ARB < $5.0 on Binance) do not retain stale trailing_peak / dd_from_peak_pct.
    4. order_retry does not wipe accumulated ttl_started_ts when transitioning out of a transient non-failure deferral at attempts > 0.
    5. monitororder.MODULE_ORDER_TTLS grants ai_executor a 24h default TTL instead of the 4h fallback.
    """
    from types import SimpleNamespace
    import monitororder
    import order_retry as oq
    from intelligence.sentiment.guards.high_stake_guard import (
        harvest_recent_orders_and_enrich_positions,
        harvest_tradable_positions_and_orders,
    )
    from providers.execution_audit import (
        build_last_cycle_execution_status,
        extract_intent_notional,
        summarize_recent_ai_intents,
    )
    from providers.kraken_provider import KrakenProvider

    # 1. KrakenProvider falls back to _bot_fallback_cli on Permission denied / missing order in SPARE
    kp = KrakenProvider()
    spare_cli = SimpleNamespace(
        api_key="spare_k",
        open_orders=lambda: (_ for _ in ()).throw(RuntimeError("EGeneral:Permission denied")),
        query_orders=lambda oid: {},
    )
    bot_cli = SimpleNamespace(
        api_key="bot_k",
        open_orders=lambda: {
            "OWZ3YO-EKP5O-IPKB2Y": {
                "status": "open",
                "vol": "4.87693127",
                "vol_exec": "0.00000000",
                "descr": {"pair": "HYPEUSD", "type": "sell", "ordertype": "limit", "price": "93.85"},
            }
        },
        query_orders=lambda oid: {
            oid: {
                "status": "open",
                "vol": "4.87693127",
                "vol_exec": "0.00000000",
                "cost": "0.00000",
                "fee": "0.00000",
            }
        },
    )
    kp._cli = spare_cli
    kp._auto_cli = spare_cli
    kp._bot_fallback_cli = bot_cli

    oo_list = kp.open_orders("HYPEUSD")
    assert len(oo_list) == 1
    assert oo_list[0]["orderId"] == "OWZ3YO-EKP5O-IPKB2Y"
    assert oo_list[0]["price"] == 93.85
    st_ord = kp.order_status("HYPEUSD", "OWZ3YO-EKP5O-IPKB2Y")
    assert st_ord.status == "open"
    assert st_ord.orig_qty == pytest.approx(4.87693127)

    # 2. Kraken add_order payload without 'status' field is OPEN_RESTING, and stale >10m missing limit is CANCELED_OR_EXPIRED
    krk_intent = {
        "intent_id": "krk_lim_sell",
        "timestamp": 1793360000.0,
        "decision": "SELL",
        "symbol": "HYPEUSD",
        "venue": "kraken",
        "order_type": "LIMIT",
        "limit_price": 93.85,
        "reference_price": 92.61,
        "suggested_notional_usd": 457.70,
        "status": "EXECUTED",
        "execution_payload": {
            "txid": ["OWZ3YO-EKP5O-IPKB2Y"],
            "descr": {"order": "sell 4.87693127 HYPEUSD @ limit 93.85"},
        },
    }
    st_krk = build_last_cycle_execution_status(
        [krk_intent],
        "crypto",
        1793360060.0,
        extract_intent_notional,
        recent_orders={},
        open_orders=[],
    )
    assert st_krk["status"] == "OPEN_RESTING"
    assert st_krk["fill_px"] is None
    assert st_krk["filled_qty"] is None
    assert st_krk["slippage_bps"] is None

    stale_btc_intent = {
        "intent_id": "bin_lim_stale",
        "timestamp": 1793340000.0,
        "decision": "BUY",
        "symbol": "BTCUSDC",
        "venue": "binance",
        "order_type": "LIMIT",
        "limit_price": 76450.0,
        "reference_price": 77100.0,
        "suggested_notional_usd": 449.53,
        "status": "EXECUTED",
        "execution_payload": {
            "orderId": "10685631358",
            "status": "NEW",
            "price": "76450.0",
            "origQty": "0.00588",
            "executedQty": "0.0",
        },
    }
    sum_stale = summarize_recent_ai_intents(
        [stale_btc_intent],
        "crypto",
        1793360000.0,
        extract_intent_notional,
        recent_orders={},
        open_orders=[],
    )
    assert sum_stale[0]["venue_status"] == "CANCELED_OR_EXPIRED"

    # 3. Dust balance ($1.08 ARB < $5.0 min_notional on Binance) strips trailing_peak / dd_from_peak_pct
    class _ArbProv:
        def free_balance(self, asset):
            return 6.66268 if asset == "ARB" else 1500.0

    class _ArbInst:
        symbol = "ARBUSDC"
        provider_name = "binance"
        base = "ARB"
        quote = "USDC"
        params = {}
        _provider = _ArbProv()

    pos_arb, _, _, inst_by_sym = harvest_tradable_positions_and_orders(
        {"ARBUSDC": _ArbInst()},
        {"ARBUSDC": {"price": 0.1624}},
        {"ARBUSDC": 0.1636},
        {},
    )
    assert len(pos_arb) == 1
    assert pos_arb[0]["value_usd"] == pytest.approx(1.08, abs=0.02)
    assert "trailing_peak" not in pos_arb[0]
    assert "dd_from_peak_pct" not in pos_arb[0]

    pos_arb[0]["trailing_peak"] = 0.1636
    pos_arb[0]["dd_from_peak_pct"] = -0.73
    harvest_recent_orders_and_enrich_positions(
        str(tmp_path), inst_by_sym, pos_arb, {"ARBUSDC": {"price": 0.1624}}, {}, {}, {}
    )
    assert "trailing_peak" not in pos_arb[0]
    assert "dd_from_peak_pct" not in pos_arb[0]

    # 4. order_retry complete_claim preserves prior elapsed TTL when attempts > 0 transitions out of deferral
    rec_existing = {
        "id": "r_tao_1",
        "symbol": "TAOUSDC",
        "side": "BUY",
        "qty": 1.0,
        "place_kwargs": {"client_order_id": "MT_OR_r_tao_1_0"},
        "revision": 0,
        "claim_token": "tok1",
        "claim_until": 20100.0,
        "claim_revision": 0,
        "created_ts": 1000.0,
        "ttl_started_ts": 1000.0,
        "attempts": 10,
        "last_attempt_ts": 19700.0,
        "last_failure_reason": "account_cache_not_fresh",
        "submission_state": "refused",
        "safe_to_discard": True,
    }
    oq.rewrite([rec_existing])
    oq.complete_claim(
        rec_existing,
        "failure",
        now=20000.0,
        failure_reason="qty_zero_after_policy",
        submission_state="refused",
    )
    updated_rec = oq.load_all()[0]
    assert updated_rec["attempts"] == 11
    # Only shifted by the 300s deferral window (19700 -> 20000), not reset to 20000.0
    assert updated_rec["ttl_started_ts"] == pytest.approx(1300.0)

    # 5. monitororder grants ai_executor 24h TTL
    assert monitororder.resolve_effective_order_ttl("1", "AI_intent_123_buy_btcusdc") == 86400.0

    # 6. harvest_bot_runtime_states & assetguardian._sell_campaign_tier clear dust (<$5) sell campaigns
    import assetguardian as ag
    from intelligence.sentiment.guards.high_stake_guard import harvest_bot_runtime_states

    ag_file = tmp_path / "assetguardian_state.json"
    ag_file.write_text(
        json.dumps(
            {
                "version": 3,
                "symbols": {
                    "ARBUSDC": {"sell": {"initial_qty": 6596.16, "completed_tiers": [9.0]}},
                    "BTCUSDC": {"buy": {"initial_cash": 3000.0, "completed_tiers": [7.0]}},
                },
            }
        ),
        encoding="utf-8",
    )
    bot_states = harvest_bot_runtime_states(
        str(tmp_path),
        1793365000.0,
        tradable_positions=pos_arb,
    )
    assert "ARBUSDC" not in bot_states.get("assetguardian_state", {})
    assert "BTCUSDC" in bot_states.get("assetguardian_state", {})

    ag_mem = ag.AssetGuardianState(str(ag_file), fail_closed=False)
    old_ag_state = ag.STATE
    try:
        ag.STATE = ag_mem
        tier_dust, camp_dust, _ = ag._sell_campaign_tier(
            "ARBUSDC",
            0.1625,
            {"price": 0.1479, "timestamp": 1793319542.0},
            6.66268,
            SimpleNamespace(open_orders=lambda _s: []),
        )
        assert tier_dust is None
        assert camp_dust == {}
    finally:
        ag.STATE = old_ag_state

    # 7. Filled MARKET/LIMIT intents normalize venue_status ("NEW"/"RESTING" -> "FILLED")
    t212_mkt_intent = {
        "intent_id": "t212_mkt_1",
        "timestamp": 1793367352.0,
        "decision": "BUY",
        "symbol": "SPCX_US_EQ",
        "venue": "t212",
        "order_type": "MARKET",
        "reference_price": 137.69,
        "suggested_notional_usd": 150.0,
        "status": "EXECUTED",
        "execution_payload": {"id": "59656394943", "status": "NEW"},
    }
    hl_lim_filled_intent = {
        "intent_id": "hl_lim_filled_1",
        "timestamp": 1793355669.0,
        "decision": "BUY",
        "symbol": "HYPEUSDC",
        "venue": "hyperliquid",
        "order_type": "LIMIT",
        "limit_price": 90.85,
        "reference_price": 91.74,
        "suggested_notional_usd": 250.0,
        "status": "EXECUTED",
        "execution_payload": {"orderId": "590563846044", "status": "RESTING"},
    }
    sum_eq = summarize_recent_ai_intents(
        [t212_mkt_intent],
        "equities",
        1793380000.0,
        extract_intent_notional,
        recent_orders={"t212:SPCX_US_EQ": {"venue": "t212", "last_buy": {"price": 137.69, "qty": 1.08, "ts_ms": 1793367355000, "actor": "AI"}}},
        open_orders=[],
    )
    assert sum_eq[0]["venue_status"] == "FILLED"

    sum_hl = summarize_recent_ai_intents(
        [hl_lim_filled_intent],
        "crypto",
        1793380000.0,
        extract_intent_notional,
        recent_orders={"hyperliquid:HYPEUSDC": {"venue": "hyperliquid", "last_buy": {"price": 90.85, "qty": 2.75, "ts_ms": 1793360000000, "actor": "AI"}}},
        open_orders=[],
    )
    assert sum_hl[0]["venue_status"] == "FILLED"

    # 8. _fmt_slope_val zeroes sub-5e-8 IEEE-754 polyfit noise while keeping low-price slopes >= 5e-8
    from orchestratorOS.supervisor.prompt_formatters import _fmt_slope_val
    from intelligence.sentiment.guards.high_stake_guard import _compute_kalman_from_series

    assert _fmt_slope_val(-1.22e-8) == "+0.00000"
    assert _fmt_slope_val(2.72e-7) == "+2.72e-07"

    # 9. _compute_kalman_from_series derives kalman_vel for non-Binance symbols from 24h series
    synth_series = [[(1793375000 + i * 60) * 1000, 90.0 + i * 0.05] for i in range(30)]
    k_res = _compute_kalman_from_series(synth_series)
    assert isinstance(k_res, dict)
    assert k_res["kalman_vel"] > 0

    # 10. Fast short-window impulse kinds (accel_surge_up) honor place_kwargs.hours TTL and expire from active_retry_queue
    surge_rec = {
        "id": "surge123",
        "provider_name": "Binance",
        "symbol": "TAOUSDC",
        "side": "BUY",
        "kind": "accel_surge_up",
        "qty": 1.0794,
        "requested_price": 237.1626,
        "created_ts": 1793340806.0,
        "ttl_started_ts": 1793377545.0,
        "attempts": 99,
        "last_failure_reason": "qty_zero_after_policy",
        "lifecycle": "submit_pending",
        "place_kwargs": {"hours": 0.3, "kind": "accel_surge_up", "module": "tradeall"},
    }
    assert oq.resolve_record_ttl(surge_rec) == pytest.approx(1080.0)
    assert oq.is_expired(surge_rec, 1793381000.0) is True
    (tmp_path / "order_retry_queue.jsonl").write_text(json.dumps(surge_rec) + "\n", encoding="utf-8")
    bot_states_after_exp = harvest_bot_runtime_states(str(tmp_path), 1793381000.0)
    assert "active_retry_queue" not in bot_states_after_exp

    # 11. harvest_recent_orders_and_enrich_positions reconciles stale .state_*.json cycles against newer exit+rebuy fills
    from intelligence.sentiment.guards.high_stake_guard import (
        harvest_recent_orders_and_enrich_positions,
        parse_venue_cycle_state,
    )

    (tmp_path / "cache_trade_kraken.json").write_text(
        json.dumps({
            "items": {
                "TAOUSD": [
                    {
                        "symbol": "TAOUSD",
                        "id": "T_SELL_OLD",
                        "orderId": "T_SELL_OLD",
                        "price": 226.016729,
                        "qty": 7.40379,
                        "time": 1793319677319,
                        "isBuyer": False,
                        "side": "SELL",
                    },
                    {
                        "symbol": "TAOUSD",
                        "id": "T_BUY_NEW",
                        "orderId": "T_BUY_NEW",
                        "price": 235.741094,
                        "qty": 3.70189,
                        "time": 1793331231572,
                        "isBuyer": True,
                        "side": "BUY",
                    },
                ]
            }
        }),
        encoding="utf-8",
    )
    stale_tao_cyc = parse_venue_cycle_state(
        {
            "qty": 7.38813,
            "cost": 2105.506352,
            "entry_price": 305.469,
            "last_buy_price": 256.0432,
            "last_sell_price": 291.14414,
            "dca_buys": 5,
            "entry_ts": 1791188569.638,
        },
        "TAOUSD",
        {"TAOUSDC": {"price": 233.10}},
        avg_decimals=6,
        include_meta=True,
    )
    krk_cycles = {"TAOUSD": stale_tao_cyc}
    harvest_recent_orders_and_enrich_positions(
        str(tmp_path),
        {},
        [],
        {"TAOUSDC": {"price": 233.10}},
        krk_cycles,
        {},
        {},
    )
    assert "_entry_ts_ms" not in krk_cycles["TAOUSD"]
    assert krk_cycles["TAOUSD"]["qty"] == pytest.approx(3.70189)
    assert krk_cycles["TAOUSD"]["cost_usd"] == pytest.approx(872.69)
    assert krk_cycles["TAOUSD"]["avg_cost_px"] == pytest.approx(235.741094)
    assert krk_cycles["TAOUSD"]["last_sell_price"] == pytest.approx(226.016729)
    assert krk_cycles["TAOUSD"]["dca_buys"] == 0
    assert krk_cycles["TAOUSD"]["dd_pct"] == pytest.approx(-1.12)


def test_t212_client_cross_process_shared_disk_cache(tmp_path, monkeypatch):
    """Verify T212Client shares fresh portfolio/order snapshots across processes and falls back on HTTP 429."""
    import importlib
    t212_mod = importlib.import_module("212trading.t212_client")
    T212Client = t212_mod.T212Client

    shared_file = str(tmp_path / ".t212_shared_cache_live_test.json")
    http_calls = []

    def _fake_http_get(url, headers=None):
        http_calls.append(url)
        if url.endswith("/equity/portfolio"):
            return 200, json.dumps([{"ticker": "NVDA_US_EQ", "quantity": 0.21, "currentPrice": 202.35}]).encode("utf-8")
        if url.endswith("/equity/orders"):
            return 200, json.dumps([{"id": 991, "ticker": "SPCX_US_EQ", "quantity": 1.0, "limitPrice": 135.0}]).encode("utf-8")
        return 404, b""

    monkeypatch.setattr(t212_mod, "http_get", _fake_http_get)

    # 1. Producer process (T212-bot) fetches and populates shared disk cache
    producer = T212Client("test_key", "test_sec", "live", min_gap_sec=0.0, portfolio_ttl_sec=6.0, order_min_gap_sec=0.0)
    producer._shared_cache_path = shared_file
    producer._enable_shared_disk_cache = True
    assert producer.get_portfolio() == [{"ticker": "NVDA_US_EQ", "quantity": 0.21, "currentPrice": 202.35}]
    assert producer.list_active_orders() == [{"id": 991, "ticker": "SPCX_US_EQ", "quantity": 1.0, "limitPrice": 135.0}]
    assert len(http_calls) == 2

    # 2. Consumer process (auto_sharetrade_executor) reuses fresh (<6s) shared disk cache with 0 HTTP calls
    consumer = T212Client("test_key", "test_sec", "live", min_gap_sec=0.0, portfolio_ttl_sec=6.0, order_min_gap_sec=0.0)
    consumer._shared_cache_path = shared_file
    consumer._enable_shared_disk_cache = True
    assert consumer.get_portfolio() == [{"ticker": "NVDA_US_EQ", "quantity": 0.21, "currentPrice": 202.35}]
    assert consumer.list_active_orders() == [{"id": 991, "ticker": "SPCX_US_EQ", "quantity": 1.0, "limitPrice": 135.0}]
    assert len(http_calls) == 2

    # 3. When shared disk cache is >6s old (<60s) and consumer hits HTTP 429, it falls back to the <60s shared cache
    with open(shared_file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for k in payload:
        payload[k]["ts"] = time.time() - 12.0
    with open(shared_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    monkeypatch.setattr(t212_mod, "http_get", lambda url, headers=None: (429, b"rate limit"))
    consumer2 = T212Client("test_key", "test_sec", "live", min_gap_sec=0.0, portfolio_ttl_sec=6.0, order_min_gap_sec=0.0)
    consumer2._shared_cache_path = shared_file
    consumer2._enable_shared_disk_cache = True
    assert consumer2.get_portfolio() == [{"ticker": "NVDA_US_EQ", "quantity": 0.21, "currentPrice": 202.35}]
    assert consumer2.list_active_orders() == [{"id": 991, "ticker": "SPCX_US_EQ", "quantity": 1.0, "limitPrice": 135.0}]

    # 4. Cache invalidation removes the shared disk cache file
    with producer._lock:
        producer._invalidate_order_caches_locked()
    assert not os.path.exists(shared_file)


def test_reconcile_executed_intents_from_audit_and_kraken_t212_fill_attribution(tmp_path):
    """Verify EXECUTED intents are enriched with terminal cost/filled_qty, T212 cycle realized_net_usd is parsed, and Kraken/T212 fills are attributed to AI."""
    from providers.execution_audit import (
        build_last_cycle_execution_status,
        extract_intent_notional,
        reconcile_intents_from_execution_audit,
        summarize_recent_ai_intents,
    )
    from intelligence.sentiment.guards.high_stake_guard import (
        extract_symbol_recent_fills,
        harvest_recent_orders_and_enrich_positions,
        parse_venue_cycle_state,
    )

    audit_dir = tmp_path / "logger" / "execution_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. T212 market order initially recorded as status="NEW" and Kraken market order without cost/price
    intents = [
        {
            "intent_id": "intent_1793367352_buy_spcx_us_eq",
            "timestamp": 1793367352.98,
            "decision": "BUY",
            "symbol": "SPCX_US_EQ",
            "venue": "t212",
            "order_type": "MARKET",
            "suggested_notional_usd": 150.0,
            "clamped_notional_usd": 150.0,
            "reference_price": 137.69,
            "requested_price": 137.69,
            "status": "EXECUTED",
            "execution_payload": {
                "id": 59656394943,
                "orderId": "59656394943",
                "type": "MARKET",
                "ticker": "SPCX_US_EQ",
                "quantity": 1.08,
                "filledQuantity": 0,
                "status": "NEW",
            },
            "history": [{"ts": 1793367356.99, "action": "EXECUTED", "price": 137.69, "qty": 1.0894}],
        },
        {
            "intent_id": "intent_1793365871_buy_hypeusd",
            "timestamp": 1793365871.08,
            "decision": "BUY",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "MARKET",
            "suggested_notional_usd": 450.0,
            "reference_price": 91.17,
            "status": "EXECUTED",
            "execution_payload": {
                "txid": ["OX463P-YOSTO-S5NMAZ"],
                "descr": {"order": "buy 4.93583 HYPEUSD @ market"},
            },
            "history": [{"ts": 1793365874.83, "action": "EXECUTED", "price": 91.17, "qty": 4.93583416}],
        },
    ]

    audit_file = audit_dir / "execution_audit_2026-10-30.jsonl"
    audit_file.write_text(
        "\n".join([
            json.dumps({
                "event": "order_terminal",
                "intent_id": "intent_1793367352_buy_spcx_us_eq",
                "client_order_id": "AI_intent_1793367352_buy_spcx_us_eq",
                "order_id": "59656394943",
                "symbol": "SPCX_US_EQ",
                "venue": "t212",
                "side": "buy",
                "status": "closed",
                "venue_status": "FILLED",
                "filled_qty": 1.08,
                "cost": 148.5864,
                "fee": 1.07,
                "ts": 1793367696.36,
            }),
            json.dumps({
                "event": "order_terminal",
                "intent_id": "intent_1793365871_buy_hypeusd",
                "client_order_id": "AI_intent_1793365871_buy_hypeusd",
                "order_id": "OX463P-YOSTO-S5NMAZ",
                "symbol": "HYPEUSD",
                "venue": "kraken",
                "side": "buy",
                "status": "closed",
                "venue_status": "CLOSED",
                "filled_qty": 4.93583415,
                "cost": 450.19743,
                "fee": 1.57569,
                "ts": 1793366203.03,
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    assert reconcile_intents_from_execution_audit(intents, str(audit_dir)) is True
    # Running a second time is idempotent
    assert reconcile_intents_from_execution_audit(intents, str(audit_dir)) is False

    eq_status = build_last_cycle_execution_status(intents, "equities", 1793370000.0, extract_intent_notional)
    assert eq_status["fill_px"] == pytest.approx(137.58)
    assert eq_status["ref_px"] == pytest.approx(137.69)
    assert eq_status["filled_qty"] == pytest.approx(1.08)
    assert eq_status["notional_usd"] == pytest.approx(148.59)
    assert eq_status["slippage_bps"] == pytest.approx(-7.99)

    krk_summary = summarize_recent_ai_intents(intents, "crypto", 1793370000.0, extract_intent_notional)
    assert len(krk_summary) == 1
    assert krk_summary[0]["venue_status"] == "FILLED"
    assert krk_summary[0]["fill_px"] == pytest.approx(91.21, abs=1e-4)
    assert krk_summary[0]["slippage_bps"] == pytest.approx(4.39, abs=0.05)

    # 2. Kraken Ledgers fill with trade refid TZHKJT-VEG3G-YZAXLQ is attributed to AI via ai_intents
    krk_fills = extract_symbol_recent_fills(
        [
            {
                "symbol": "HYPEUSD",
                "id": "TEUGID-NDEDW-YJAF2M",
                "orderId": "TEUGID-NDEDW-YJAF2M",
                "price": 93.30,
                "qty": 11.7548,
                "timestamp": 1793287711318,
                "side": "SELL",
            },
            {
                "symbol": "HYPEUSD",
                "id": "TZHKJT-VEG3G-YZAXLQ",
                "orderId": "TZHKJT-VEG3G-YZAXLQ",
                "price": 91.209992,
                "qty": 4.9358342,
                "timestamp": 1793365875456,
                "side": "BUY",
            },
        ],
        known_ai_order_ids={"OX463P-YOSTO-S5NMAZ"},
        ai_intents=intents,
    )
    assert krk_fills["last_action_side"] == "BUY"
    assert krk_fills["last_action_actor"] == "AI"
    assert krk_fills["last_buy"]["actor"] == "AI"
    assert krk_fills["last_sell"]["actor"] == "BOT"

    # 3. T212 cycle state parses realized_net_usd and harvest_recent_orders synthesizes T212 fills from execution_audit
    t212_cyc = parse_venue_cycle_state(
        {
            "qty": 5.922965,
            "cost_usd": 893.90,
            "entry_price": 146.8203,
            "last_buy_price": 137.5833,
            "last_sell_price": 150.0015,
            "realized_pnl_usd": -44.01,
            "realized_net_usd": -50.3231,
            "pos_peak": 166.0119,
        },
        "SPCX_US_EQ",
        {"SPCX_US_EQ": {"price": 135.97}},
        avg_decimals=4,
    )
    assert t212_cyc["realized_pnl_usd"] == pytest.approx(-50.32)

    t212_pos = [{"venue": "t212", "symbol": "SPCX_US_EQ", "base": "SPCX", "quote": "USD", "price": 135.97, "free_base": 5.922965}]
    ro_out = harvest_recent_orders_and_enrich_positions(
        str(cache_dir),
        {},
        t212_pos,
        {"SPCX_US_EQ": {"price": 135.97}},
        {},
        {},
        {"SPCX_US_EQ": t212_cyc},
        known_ai_order_ids={"59656394943"},
        ai_intents=intents,
    )
    assert "SPCX_US_EQ" in ro_out
    assert ro_out["SPCX_US_EQ"]["last_action_actor"] == "AI"
    assert ro_out["SPCX_US_EQ"]["last_buy"]["price"] == pytest.approx(137.58)
    assert t212_pos[0]["active_avg_buy_px"] == pytest.approx(150.921)











