"""Unit tests for AutonomousAIReconciler and AutonomousOvernightMonitor."""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from orchestratorOS.livecheck.autonomous_overnight_monitor import (
    AutonomousOvernightMonitor,
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
    cachedb.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()

    (logs / "bot1.log").write_text("heartbeat")
    (cachedb / "fleet1.heartbeat").write_text("heartbeat")

    (cachedb / "geopolitical_threat_eval.json").write_text(json.dumps({
        "threat_level": "ELEVATED",
        "risk_score": 0.65,
        "summary": "Active conflict attrition",
        "ts": time.time(),
    }))

    (cachedb / "auto_trade_executor_eval.json").write_text(json.dumps({
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

    status_file = mock_workspace / "cachedb" / "overnight_monitor_status.json"
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
    assert recent["last_sell"] == {"price": 265.0, "qty": 1.0, "ts_ms": 2000}
    assert recent["last_buy"] == {"price": 250.0, "qty": 1.0, "ts_ms": 1000}


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

    with open(os.path.join(cache_dir, "sentiment_advisor_eval.json"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(cache_dir, "geopolitical_threat_eval.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "threat_level": "ELEVATED",
                "risk_score": 0.62,
                "recommended_brake": "DOWNSCALE_QTY",
                "summary": "Regional energy tension",
            },
            f,
        )
    with open(os.path.join(cache_dir, "market_breadth_cache.json"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(cache_dir, "kalman_state.json"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(cache_dir, "cache_prices_multi.json"), "w", encoding="utf-8") as f:
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

    # Verify auto_technical_repair receives fleet, bot heartbeats, Linux system resources, VPN (wgpia0), and log audit
    reconciler.run_cycle(topic_key="auto_technical_repair")
    repair_prompt = captured_prompts[-1]
    assert "LOG ANOMALIES & GUARD VETOES" in repair_prompt
    assert "LINUX SYSTEM RESOURCES & VPN TUNNEL HEALTH" in repair_prompt
    assert "VPN Tunnel (wgpia0):" in repair_prompt
    assert "Starea Flotei, Boților, Sistemului Linux & VPN (wgpia0) & Loguri" in repair_prompt
    assert "diagnosed_issues" in repair_prompt


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
    (tmp_path / "cachedb").mkdir()
    (tmp_path / "logs").mkdir()
    hb = tmp_path / "cachedb" / "rtrade.heartbeat"
    lg = tmp_path / "logs" / "rtrade.log"
    hb.write_text("hb")
    lg.write_text("log")
    now = 1_800_000_000.0
    # Log is fresh (now), but heartbeat is stale (now - 300s > 180s)
    os.utime(hb, (now - 300.0, now - 300.0))
    os.utime(lg, (now, now))

    monitor = AutonomousOvernightMonitor(workspace_dir=str(tmp_path))
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
        "market_monitor/price_notifier.py|$ROOT|python market_monitor/price_notifier.py|price_notifier|cachedb/price_notifier.heartbeat,logs/price_notifier.log|180|fleet\n"
        "hyperliquid/bot.py|$ROOT/hyperliquid|python bot.py|HL-bot|cachedb/HL-bot.heartbeat,logs/HL-bot.log|600|bot\n"
    )
    with patch.object(monitor_night, "run", return_value=procs_text):
        procs = monitor_night.load_procs(MagicMock())
    assert len(procs) == 2
    assert procs[0].log == "logs/price_notifier.log"
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
    with open(os.path.join(cache_dir, "autonomous_trade_intents.json"), "w", encoding="utf-8") as f:
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

    # 4. AutonomousOvernightMonitor sanitizes sensor floats, strips quotes in order_guard.conf, and parses latest_overnight.log
    with open(os.path.join(cache_dir, "geopolitical_threat_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"threat_level": None, "risk_score": True, "summary": None}, f)
    with open(os.path.join(cache_dir, "fear_greed_cache.json"), "w", encoding="utf-8") as f:
        json.dump({"value": "bad_val"}, f)
    mon = AutonomousOvernightMonitor(workspace_dir=workspace)
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

    # Verify full multi-horizon vs_btc and alt_btc_24h on TAOUSDC
    assert "vs_btc(15m=-0.30%, 1h=-1.00%, 3h=-0.50%, 6h=+1.00%, 12h=+2.00%, 24h=+3.00%, 7d=+12.00%)" in prompt
    assert "alt_btc_24h(pos=74.5%, dd_high=-1.20%, bounce_low=+3.40%, beta=1.42, corr=0.65)" in prompt
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
    cache_dir.mkdir(parents=True)
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
    kraken_dir.mkdir(parents=True)
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

