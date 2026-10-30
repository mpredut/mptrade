"""Autonomous AI supervisor running the cron-driven LLM topic threads.

- Ingests passive telemetry, sensors, spot wallet balances, cycle states, and fills from cachedb/, kraken/, and logs/.
- Reasons on the topic's permanent Antigravity thread (orchestratorOS.agy.topics) via GeminiClient/agy-bot.
- Each topic's TopicSpec bounds its side effects:
  * auto_technical_repair: diagnostics + verified self-healing code patches (never trades);
  * auto_trade_executor: multi-coin market/portfolio evaluation + symmetric BUY/SELL trade decisions submitted to AutonomousIntentGateway (no fleet/log audit, never patches code).
- supervisor_mode ('shadow' / 'enforce') bounds every side effect, including the gateway execution mode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from botcore import (
    _parse_bool_flag,
    is_us_rth_open as _is_us_rth_open,
    load_dotenv,
    load_key_value_conf,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.high_stake_guard import (
    apply_short_trend_item as _apply_short_trend_item,  # noqa: F401 (re-export: test compat)
    compute_liquidation_clusters,  # noqa: F401 (re-export: test compat)
    compute_swing_wave_metrics as _compute_swing_wave_metrics,  # noqa: F401 (re-export: test compat)
    compute_volume_profile_metrics,  # noqa: F401 (re-export: test compat)
    parse_venue_cycle_state as _parse_venue_cycle_state,  # noqa: F401 (re-export: test compat)
    populate_market_and_portfolio_snapshot,
    resolve_venue_fee_and_limits,  # noqa: F401 (re-export: test compat)
)
from orchestratorOS.agy.topics import TOPIC_ALIASES, TOPIC_SPECS, topic_spec
from orchestratorOS.lib.technical_repair_ssot import (
    BOT_AND_ORDER_ERROR_CATEGORIES,
    LogEventSummary,
    SYSTEM_OS_ERROR_CATEGORIES,
    apply_code_patch_safely,
    apply_reconciliation_plan,
    audit_workspace_configs,
    build_sensor_health_summary,
    default_supervisor_fallback_plan,
    execute_venue_or_fleet_repair_action,
    format_supervisor_ntfy_notification,
    harvest_fleet_and_system_telemetry,
    harvest_heartbeats_and_state_files,
    harvest_log_anomalies_from_dirs,
    is_bot_or_code_error_report,
    is_error_report,
    is_system_os_error_report,
    normalize_error_signature,
    parse_supervisor_config,
    persist_supervisor_audit_artifact,
    read_json_recovering,
    recent_log_lines,  # noqa: F401 (re-export: test compat)
    send_supervisor_ntfy_http,
    submit_supervisor_trading_decisions,
)
from orchestratorOS.supervisor.prompt_formatters import (
    _format_venue_cycle_line,  # noqa: F401 (re-export: test compat)
    build_supervisor_prompt,
)
from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
from providers.execution_audit import (
    normalize_trading_decisions as _normalize_trading_decisions,  # noqa: F401 (re-export: test compat)
)

_parse_finite_float = safe_float

load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("orchestratorOS.supervisor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class TelemetrySensorSnapshot:
    """Consolidated snapshot of passive sensors and wallet state collected from bot runtime files."""
    fear_greed: Optional[Dict[str, Any]] = None
    market_breadth: Optional[Dict[str, Any]] = None
    sentiment_advisor: Optional[Dict[str, Any]] = None
    geopolitical_threat: Optional[Dict[str, Any]] = None
    macro_benchmarks: Optional[Dict[str, Any]] = None
    breaking_news: List[Dict[str, Any]] = field(default_factory=list)
    active_heartbeats: Dict[str, Any] = field(default_factory=dict)
    state_files: Dict[str, Any] = field(default_factory=dict)
    prices_and_trends: Dict[str, Any] = field(default_factory=dict)
    microstructure: Dict[str, Any] = field(default_factory=dict)
    portfolio_summary: Dict[str, Any] = field(default_factory=dict)
    fleet_processes: List[Dict[str, Any]] = field(default_factory=list)
    system_and_vpn: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisorAuditReport:
    """Consolidated audit and autonomous trade execution result produced by the supervisor."""
    ts: float
    mode: str                          # "shadow" | "enforce" | "off"
    summary: str
    diagnosed_issues: List[Dict[str, Any]] = field(default_factory=list)
    venue_actions: List[Dict[str, Any]] = field(default_factory=list)
    code_remediation: Dict[str, Any] = field(default_factory=dict)
    market_trading_decision: Dict[str, Any] = field(default_factory=dict)
    market_trading_decisions: List[Dict[str, Any]] = field(default_factory=list)
    intent_emitted: Optional[Dict[str, Any]] = None
    intents_emitted: List[Dict[str, Any]] = field(default_factory=list)
    sensor_health: Dict[str, Any] = field(default_factory=dict)
    actions_executed: bool = False
    tests_passed: Optional[bool] = None
    notification_sent: bool = False


class AutonomousAIReconciler:
    """Autonomous supervisor orchestrating decoupled sensor harvesting, log auditing, and multi-coin trading."""

    def __init__(
        self,
        mode: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        cache_dir: str = "cachedb",
        logs_dirs: Optional[List[str]] = None,
        gemini_auditor: Optional[GeminiClient] = None,
        notify_enabled: Optional[bool] = None,
        lookback_hours: float = 24.0,
        intent_gateway: Optional[Any] = None,
    ) -> None:
        self.workspace_dir = workspace_dir or os.getcwd()
        self.cache_dir = os.path.join(self.workspace_dir, cache_dir)
        lb = safe_pos_float(lookback_hours, default=None) if isinstance(lookback_hours, (int, float)) else None
        if lb is None:
            raise ValueError("lookback_hours must be a positive finite number")
        self.lookback_hours = lb
        self.logs_dirs = [os.path.join(self.workspace_dir, d) for d in (logs_dirs or ["logs", "logger"])]
        self.gemini_auditor = gemini_auditor or GeminiClient()
        self._intent_gateway = intent_gateway

        # Parse decoupled configuration strictly from config files without fallbacks
        conf = self._load_decoupled_conf()
        (
            self.mode,
            self.notify_enabled,
            self.ntfy_url,
            self.ntfy_topic_guard,
            self.ntfy_topic_server,
            self.ntfy_topic_error,
            self.ntfy_token,
        ) = parse_supervisor_config(conf, mode=mode, notify_enabled=notify_enabled)
        self.ntfy_topic = self.ntfy_topic_server

        # Per-topic artifact paths are set by run_cycle(topic_key=...).
        self.state_file = ""
        self.heartbeat_file = ""

    @property
    def intent_gateway(self) -> Any:
        """The single trade-intent choke point (owns autonomous_trading_* config, clamping, dedup, mode)."""
        if getattr(self, "_intent_gateway", None) is None:
            self._intent_gateway = AutonomousIntentGateway(workspace_dir=self.workspace_dir)
        return self._intent_gateway

    def _load_decoupled_conf(self) -> Dict[str, str]:
        """Parses key-value pairs from order_guard.conf and .env without external imports."""
        return load_key_value_conf(self.workspace_dir, _ROOT_DIR)

    def harvest_sensors_telemetry(
        self,
        *,
        include_fleet: bool = True,
        include_market_portfolio: bool = True,
        asset_scope: str = "all",
    ) -> TelemetrySensorSnapshot:
        """Reads passive telemetry, spot wallet balances, cycle states, and fills published to disk."""
        scope = (asset_scope or "all").strip().lower()
        snapshot = TelemetrySensorSnapshot()
        now_ts = time.time()
        _read_json = read_json_recovering

        snapshot.active_heartbeats, snapshot.state_files = harvest_heartbeats_and_state_files(
            self.cache_dir, self.workspace_dir, now_ts, _read_json
        )
        if include_fleet:
            snapshot.fleet_processes, snapshot.system_and_vpn = harvest_fleet_and_system_telemetry(
                self.workspace_dir
            )
        if not include_market_portfolio:
            return snapshot

        return populate_market_and_portfolio_snapshot(
            snapshot,
            cache_dir=self.cache_dir,
            workspace_dir=self.workspace_dir,
            logs_dirs=self.logs_dirs,
            intent_gateway=self.intent_gateway,
            scope=scope,
            now_ts=now_ts,
            read_json_fn=_read_json,
            log=logger,
        )

    def _audit_workspace_configs(self) -> Dict[str, Any]:
        """Audits workspace configuration files via the technical repair SSOT."""
        return audit_workspace_configs(self.workspace_dir)

    @staticmethod
    def _normalize_error_signature(line: str) -> str:
        """Strip timestamps, PIDs, and hex addresses so recurring log errors group into one signature."""
        return normalize_error_signature(line)

    def harvest_log_anomalies(self, lookback_hours: Optional[float] = None) -> LogEventSummary:
        """Parses recent log files for errors, persistent unresolved exceptions, and guard vetoes."""
        hours = safe_pos_float(lookback_hours, default=self.lookback_hours) or self.lookback_hours
        return harvest_log_anomalies_from_dirs(self.logs_dirs, hours)

    def synthesize_and_diagnose(
        self,
        sensors: TelemetrySensorSnapshot,
        logs: LogEventSummary,
        *,
        focus_symbols: Optional[List[str]] = None,
        topic_key: str,
    ) -> Dict[str, Any]:
        """Feeds topic-appropriate telemetry to Gemini; the topic spec bounds both prompt scope and side effects."""
        spec = topic_spec(topic_key)
        min_usd, max_usd, min_conf = 10.0, 2000.0, 0.85
        if spec.can_trade:
            gw = self.intent_gateway
            max_usd = safe_pos_float(getattr(gw, "max_notional_usd", None), default=2000.0) or 2000.0
            min_usd = safe_pos_float(getattr(gw, "min_notional_usd", None), default=10.0) or 10.0
            min_conf = safe_nonneg_float(getattr(gw, "min_confidence", None), default=0.85)

        prompt = build_supervisor_prompt(
            sensors,
            logs,
            spec,
            self.lookback_hours,
            focus_symbols=focus_symbols,
            min_usd=min_usd,
            max_usd=max_usd,
            min_conf=min_conf,
        )

        resp = self.gemini_auditor.query_json(
            prompt,
            timeout_sec=90.0,
            thread_title=f"{spec.title} — {self.mode.upper()}",
            topic_key=topic_key,
        )

        if not resp or not isinstance(resp, dict):
            logger.warning("Gemini supervisor query failed. Falling back to default baseline.")
            return default_supervisor_fallback_plan(spec, logs)

        return resp

    def _execute_venue_or_fleet_action(self, act: Dict[str, Any]) -> bool:
        """Executes a single repair action (bot/module/fleet/VPN/DNS restart, system reboot, or order sync/cancel) in enforce mode."""
        return execute_venue_or_fleet_repair_action(
            act,
            self.workspace_dir,
            resolve_instrument_fn=lambda sym, venue=None: self.intent_gateway.resolve_instrument(sym, venue=venue),
            log=logger,
        )

    def apply_reconciliation(
        self, plan: Dict[str, Any], *, allow_code_remediation: bool = True
    ) -> Tuple[bool, Optional[bool]]:
        """Applies or simulates venue/fleet repair actions and code remediation based on operational mode."""
        return apply_reconciliation_plan(
            plan,
            self.workspace_dir,
            is_enforce=(self.mode == "enforce"),
            allow_code_remediation=allow_code_remediation,
            execute_action_fn=self._execute_venue_or_fleet_action,
            apply_patch_fn=self._apply_code_patch_safely,
            log=logger,
        )

    def _apply_code_patch_safely(
        self,
        target_file: str,
        search_block: str,
        replace_block: str,
        rationale: str,
    ) -> bool:
        """Applies code or configuration patch, executes pytest test suite, commits on 100% green or rolls back immediately."""
        return apply_code_patch_safely(
            target_file,
            search_block,
            replace_block,
            rationale,
            self.workspace_dir,
            log=logger,
        )

    _BOT_AND_ORDER_ERROR_CATEGORIES = BOT_AND_ORDER_ERROR_CATEGORIES
    _SYSTEM_OS_ERROR_CATEGORIES = SYSTEM_OS_ERROR_CATEGORIES

    @classmethod
    def _is_bot_or_code_error_report(cls, report: SupervisorAuditReport) -> bool:
        """Return True strictly for trading bot exceptions, fleet/config anomalies, or code/config patches."""
        return is_bot_or_code_error_report(report)

    @classmethod
    def _is_system_os_error_report(cls, report: SupervisorAuditReport) -> bool:
        """Return True for Linux OS, internet/VPN, memory, disk space, or server infrastructure anomalies."""
        return is_system_os_error_report(report)

    @classmethod
    def _is_error_report(cls, report: SupervisorAuditReport) -> bool:
        """Return True if the audit report diagnoses any bot/order/code error or system/OS error."""
        return is_error_report(report)

    def dispatch_notification(
        self,
        report: SupervisorAuditReport,
        *,
        topic_key: str = "auto_technical_repair",
    ) -> bool:
        """Sends a clean, compact push alert via standard HTTP POST to ntfy."""
        if not self.notify_enabled:
            return False

        title, body, priority, target_topic, is_any_error = format_supervisor_ntfy_notification(
            report,
            topic_key,
            self.ntfy_topic_error,
            self.ntfy_topic_server,
            self.ntfy_topic,
            is_bot_error=self._is_bot_or_code_error_report(report),
            is_sys_error=self._is_system_os_error_report(report),
        )
        return send_supervisor_ntfy_http(
            title,
            body,
            priority,
            target_topic,
            is_any_error,
            ntfy_url=self.ntfy_url,
            ntfy_token=self.ntfy_token,
            urlopen_fn=urllib.request.urlopen,
            log=logger,
        )

    def write_audit_artifact(self, report: SupervisorAuditReport) -> None:
        """Persists the full audit report and heartbeat to disk in cachedb/."""
        persist_supervisor_audit_artifact(
            asdict(report),
            self.cache_dir,
            self.state_file,
            self.heartbeat_file,
            log=logger,
        )

    def run_cycle(
        self,
        *,
        topic_key: str,
        focus_symbols: Optional[List[str]] = None,
        force_rth_open: bool = False,
    ) -> SupervisorAuditReport:
        """Run one audit turn on the topic's permanent thread; its TopicSpec bounds the allowed side effects."""
        spec = topic_spec(topic_key)
        logger.info("Starting Autonomous AI Supervisor cycle (mode=%s, topic=%s)...", self.mode, topic_key)
        now = time.time()
        # Per-topic artifacts so one topic never overwrites another topic's report.
        self.state_file = os.path.join(self.cache_dir, f"{topic_key}_eval.json")
        self.heartbeat_file = os.path.join(self.cache_dir, f"{topic_key}.heartbeat")

        if self.mode == "off":
            report = SupervisorAuditReport(ts=now, mode="off", summary="Supervisor mode is off; cycle skipped.")
            self.write_audit_artifact(report)
            return report

        scope = getattr(spec, "asset_scope", "all")
        if (
            scope == "equities"
            and not _parse_bool_flag(force_rth_open, False)
            and not _parse_bool_flag(os.environ.get("FORCE_SHARE_RTH_OPEN"), False)
            and not _is_us_rth_open(now)
        ):
            msg = "US equities regular trading hours (09:30-16:00 America/New_York, Mon-Fri) closed; cycle skipped."
            logger.info("[SHARE EXECUTOR] %s", msg)
            report = SupervisorAuditReport(ts=now, mode=self.mode, summary=msg)
            self.write_audit_artifact(report)
            return report

        try:
            sensors = self.harvest_sensors_telemetry(
                include_fleet=spec.can_self_heal,
                include_market_portfolio=spec.can_trade,
                asset_scope=scope,
            )
        except TypeError:
            try:
                sensors = self.harvest_sensors_telemetry(include_fleet=spec.can_self_heal, asset_scope=scope)
            except TypeError:
                sensors = self.harvest_sensors_telemetry(include_fleet=spec.can_self_heal)
        logs = (
            self.harvest_log_anomalies()
            if spec.can_self_heal
            else LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)
        )
        plan = self.synthesize_and_diagnose(sensors, logs, focus_symbols=focus_symbols, topic_key=topic_key)
        executed, tests_passed = (
            self.apply_reconciliation(plan, allow_code_remediation=True)
            if spec.can_self_heal
            else (False, None)
        )

        (
            decisions_list,
            market_decision,
            intents_emitted,
            primary_intent,
            trade_executed,
        ) = submit_supervisor_trading_decisions(
            plan,
            spec,
            topic_key=topic_key,
            mode=self.mode,
            now_ts=now,
            intent_gateway=self.intent_gateway,
            is_rth_open_fn=_is_us_rth_open,
        )
        if trade_executed:
            executed = True

        raw_issues = plan.get("diagnosed_issues") if spec.can_self_heal else []
        raw_actions = plan.get("venue_actions") if spec.can_self_heal else []
        raw_fix = plan.get("code_remediation") if spec.can_self_heal else None
        report = SupervisorAuditReport(
            ts=now,
            mode=self.mode,
            summary=str(plan.get("summary") or "Audit complete."),
            diagnosed_issues=raw_issues if isinstance(raw_issues, list) else [],
            venue_actions=raw_actions if isinstance(raw_actions, list) else [],
            code_remediation=raw_fix if isinstance(raw_fix, dict) else {},
            market_trading_decision=market_decision,
            market_trading_decisions=decisions_list,
            intent_emitted=primary_intent,
            intents_emitted=intents_emitted,
            sensor_health=build_sensor_health_summary(sensors),
            actions_executed=executed,
            tests_passed=tests_passed,
        )
        # SRE/system/bot-error notifications (ntfy-server / ntfy-error) belong strictly to auto_technical_repair
        # (spec.can_self_heal); trade notifications are dispatched by AutonomousIntentGateway to ntfy-trade.
        if spec.can_self_heal:
            report.notification_sent = self.dispatch_notification(report, topic_key=topic_key)
        else:
            notified_statuses = ("OBSERVED_SHADOW", "EXECUTED", "BLOCKED_BY_GUARD", "FAILED")
            report.notification_sent = bool(
                getattr(self.intent_gateway, "notify_enabled", False)
                and any(
                    isinstance(it, dict) and str(it.get("status") or "").upper() in notified_statuses
                    for it in intents_emitted
                )
            )
        self.write_audit_artifact(report)
        logger.info("Autonomous AI Supervisor cycle complete: %s", report.summary)
        return report


def main() -> None:
    """CLI entry point for the cron-driven supervisor topics."""
    valid_topics = sorted(
        {k for k, v in TOPIC_SPECS.items() if v.can_trade or v.can_self_heal} | set(TOPIC_ALIASES.keys())
    )
    parser = argparse.ArgumentParser(description="Autonomous AI supervisor (technical repair / trade executor)")
    parser.add_argument("--topic-key", required=True,
                        choices=valid_topics,
                        help="Permanent LLM topic thread to run")
    parser.add_argument("--mode", choices=["shadow", "enforce"], default=None,
                        help="Override supervisor_mode from order_guard.conf (bounds trading and self-heal)")
    parser.add_argument("--focus-symbols", default="",
                        help="Optional comma-separated priority watchlist (by default all live coins are evaluated)")
    parser.add_argument("--lookback-hours", type=float, default=24.0,
                        help="Log anomaly lookback window in hours (default: 24; 0.2 for 10-minute runs)")
    parser.add_argument("--force-rth-open", action="store_true",
                        help="Allow auto_sharetrade_executor evaluation even outside US Regular Trading Hours")
    args = parser.parse_args()

    focus = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()] or None
    reconciler = AutonomousAIReconciler(mode=args.mode, lookback_hours=args.lookback_hours)
    report = reconciler.run_cycle(
        topic_key=args.topic_key,
        focus_symbols=focus,
        force_rth_open=args.force_rth_open,
    )
    print(f"[{report.mode.upper()}] {args.topic_key}: {report.summary}")


if __name__ == "__main__":
    main()
