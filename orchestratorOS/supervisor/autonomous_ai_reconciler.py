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
from datetime import datetime
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from zoneinfo import ZoneInfo

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import (
    TOPIC_ALIASES,
    TOPIC_SPECS,
    _parse_bool_flag,
    bool_env,
    canonical_topic_key,
    disable_auto_start_websockets_default,
    float_env,
    is_cache_fresh,
    is_unmocked_llm_query_allowed,
    is_us_rth_open,
    load_dotenv,
    load_key_value_conf,
    normalize_epoch_sec,
    optional_env,
    parse_pos_cli_float,
    parse_symbol_list,
    resolve_llm_timeout_sec,
    safe_nonneg_float,
    safe_pos_float,
    topic_spec,
    us_market_early_close_name,
    us_market_holiday_name,
)
from cachepaths import LLM_QUOTA_STATUS_FILENAME, resolve_cache_subdomain, resolve_cachedb_filepath
from instrument_registry import disabled_registry_symbols, observe_only_registry_symbols

disable_auto_start_websockets_default()
from intelligence.macro_sentiment_ssot import (
    TelemetrySensorSnapshot,
    resolve_derived_telemetry_intervals,
)
from intelligence.sentiment.gemini_client import GeminiClient, query_topic_json
from intelligence.sentiment.llm_quota_router import (
    derive_xhigh_model_from_high,
    resolve_configured_openai_compat_model,
    resolve_crypto_executor_model_override,
)
from intelligence.sentiment.openai_compat_rest import strip_openai_compat_prefix
from intelligence.sentiment.collectors.macro_telemetry_collector import (
    MarketAndPortfolioSnapshotOrchestrator,
)
from intelligence.sentiment.evaluators.prompt_delta_evaluator import evaluate_prompt_delta_plan
from intelligence.sentiment.evaluators.prompt_invariants_evaluator import (
    evaluate_live_prompt_invariants,
    extract_temporal_invariant_anchor,
)
from orchestratorOS.agy.topics import get_topic_thread_state, get_topic_turn_count, has_pending_ui_turn
from orchestratorOS.supervisor.crypto_executor_delta_gate import (
    evaluate_crypto_executor_delta_gate,
    is_extreme_stakes_delta_gate_wakeup,
    is_urgent_delta_gate_wakeup,
)
from providers.ai_circuit_breaker import DEFAULT_MIN_TARGET_RR
from providers.intent_state import eval_doc_was_actionable
from state_io import read_json_recovering
from orchestratorOS.lib.technical_repair_ssot import (
    apply_code_patch_safely,
    apply_reconciliation_plan,
    build_llm_config_env_remediation,
    build_sensor_health_summary,
    default_supervisor_fallback_plan,
    execute_venue_or_fleet_repair_action,
    format_supervisor_ntfy_notification,
    harvest_heartbeats_and_state_files,
    harvest_log_anomalies_from_dirs,
    is_bot_or_code_error_report,
    is_system_os_error_report,
    LogEventSummary,
    parse_supervisor_config,
    persist_llm_model_catalog_state,
    persist_supervisor_audit_artifact,
    send_supervisor_ntfy_http,
    submit_supervisor_trading_decisions,
    _resolve_repair_json_reader,
)
from orchestratorOS.livecheck.fleet_health_monitor import harvest_fleet_and_system_telemetry
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
import instrument as _instrument_preload  # noqa: F401 -- preload with core modules to prevent mid-deploy split-commit lazy import races

load_dotenv()

logger = logging.getLogger("orchestratorOS.supervisor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PROMPT_INVARIANT_ALERT_COOLDOWN_SEC: float = 14400.0


def _resolve_topic_cache_path(cache_dir: str, filename: str) -> str:
    """Resolve ``filename`` inside ``cache_dir``, honoring ``cachedb/<subdomain>/`` routing."""
    raw = os.path.join(cache_dir, filename)
    resolved = str(resolve_cachedb_filepath(raw))
    if os.path.exists(resolved) or os.path.exists(f"{resolved}.previous"):
        return resolved
    if sub := resolve_cache_subdomain(filename):
        sub_path = os.path.join(cache_dir, sub, filename)
        if os.path.exists(sub_path) or os.path.exists(f"{sub_path}.previous"):
            return sub_path
    return raw


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
    model_used: str = ""
    delta_gate_meta: Dict[str, Any] = field(default_factory=dict)
    prompt_delta_state: Dict[str, Any] = field(default_factory=dict)
    prompt_invariant_violations: List[str] = field(default_factory=list)
    prompt_invariant_alert_ts: Dict[str, float] = field(default_factory=dict)
    temporal_invariant_anchor: Dict[str, Any] = field(default_factory=dict)
    completed_ts: Optional[float] = None


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
        self._last_model_used: str = ""
        self._last_query_succeeded: bool = False
        self._last_prompt_delta_state: Dict[str, Any] = {}
        self._last_prompt_invariant_violations: List[str] = []
        self._prompt_invariant_alert_ts: Dict[str, float] = {}
        self._last_temporal_invariant_anchor: Dict[str, Any] = {}

        # Parse decoupled configuration strictly from config files without fallbacks
        conf = load_key_value_conf(self.workspace_dir, _ROOT_DIR)
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

    def harvest_sensors_telemetry(
        self,
        *,
        include_fleet: bool = True,
        include_market_portfolio: bool = True,
        asset_scope: str = "all",
        allow_network: bool = False,
        persist: bool = False,
    ) -> TelemetrySensorSnapshot:
        """Reads passive telemetry, spot wallet balances, cycle states, and fills published to disk."""
        scope = (asset_scope or "all").strip().lower()
        snapshot = TelemetrySensorSnapshot()
        now_ts = time.time()
        _read_json = _resolve_repair_json_reader(read_json_recovering)

        snapshot.active_heartbeats, snapshot.state_files = harvest_heartbeats_and_state_files(
            self.cache_dir, self.workspace_dir, now_ts, _read_json
        )
        if include_fleet:
            snapshot.fleet_processes, snapshot.system_and_vpn = harvest_fleet_and_system_telemetry(
                self.workspace_dir
            )
        if not include_market_portfolio:
            snapshot.snapshot_id = f"sensors_only:{scope}:{round(now_ts * 1000.0)}"
            snapshot.source = "cachedb_read_only" if not allow_network and not persist else "live_harvest"
            snapshot.asset_scope = scope
            snapshot.ts = now_ts
            snapshot.evaluated_at_ts = now_ts
            snapshot.age_sec = 0.0
            return snapshot

        if not allow_network and not persist:
            return MarketAndPortfolioSnapshotOrchestrator(start_sync=False).read_snapshot(
                self.cache_dir,
                "*",
                snapshot=snapshot,
                workspace_dir=self.workspace_dir,
                logs_dirs=self.logs_dirs,
                intent_gateway=self.intent_gateway,
                scope=scope,
                now_ts=0.0,
                read_json_fn=_read_json,
                log=logger,
            )

        return MarketAndPortfolioSnapshotOrchestrator(start_sync=allow_network).collect(
            self.cache_dir,
            "*",
            snapshot=snapshot,
            workspace_dir=self.workspace_dir,
            logs_dirs=self.logs_dirs,
            intent_gateway=self.intent_gateway,
            scope=scope,
            now_ts=0.0,
            read_json_fn=_read_json,
            log=logger,
            allow_network=allow_network,
            persist=persist,
        )

    def harvest_log_anomalies(self, lookback_hours: Optional[float] = None) -> LogEventSummary:
        """Parses recent log files for errors, persistent unresolved exceptions, and guard vetoes."""
        hours = safe_pos_float(lookback_hours, default=self.lookback_hours) or self.lookback_hours
        return harvest_log_anomalies_from_dirs(self.logs_dirs, hours)

    def _load_cached_quota_snapshot(self) -> Dict[str, Any]:
        """Read cached ``llm_quota_status.json`` from ``self.cache_dir`` via ``LlmQuotaStatusStore`` and refresh when stale."""
        from state_stores.llm_quota_status import LlmQuotaStatusStore

        quota_path = _resolve_topic_cache_path(self.cache_dir, LLM_QUOTA_STATUS_FILENAME)
        raw: Dict[str, Any] = LlmQuotaStatusStore.read_recovering_path(quota_path, default={})
        probe_fn = getattr(self.gemini_auditor, "probe_live_model_quota", None)
        if callable(probe_fn) and is_unmocked_llm_query_allowed():
            ttl_sec = max(30.0, float_env("LLM_QUOTA_CACHE_TTL_SEC", default=30.0) or 30.0)
            quota_ts = raw.get("checked_ts") if raw.get("checked_ts") is not None else raw.get("ts")
            if not raw or not is_cache_fresh(quota_ts, time.time(), ttl_sec=ttl_sec):
                try:
                    live = probe_fn()
                    if isinstance(live, dict) and live:
                        return live
                except Exception as exc:
                    logger.debug("Live quota refresh skipped; using cached snapshot: %s", exc)
        return raw

    def _validate_runtime_prompt_invariants(
        self,
        sensors: TelemetrySensorSnapshot,
        prompt: str,
        resume_prompt: Optional[str],
        *,
        topic_key: str,
        min_usd: float,
        prev_temporal_anchor: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Run pure ~1ms mathematical & structural validation on the rendered prompt before every LLM call."""
        try:
            non_tradable = (
                disabled_registry_symbols(self.workspace_dir)
                | observe_only_registry_symbols(self.workspace_dir)
            )
            violations = evaluate_live_prompt_invariants(
                sensors,
                prompt,
                resume_prompt_text=resume_prompt,
                non_tradable_symbols=non_tradable,
                gateway_min_usd=min_usd,
                prev_temporal_anchor=prev_temporal_anchor,
            )
        except Exception as exc:
            logger.debug("Runtime prompt invariant check skipped (%s): %s", topic_key, exc)
            return []
        if not violations:
            return []
        lines = [v.format_line() for v in violations[:20]]
        logger.error(
            "[PROMPT_INVARIANT_VIOLATION] [%s] %d violation(s): %s",
            topic_key,
            len(violations),
            "; ".join(lines[:5]),
        )
        if self.notify_enabled:
            sig = f"{topic_key}:{','.join(sorted({v.code for v in violations}))}"
            now_ts = time.time()
            last_ts = self._prompt_invariant_alert_ts.get(sig, 0.0)
            if now_ts - last_ts >= _PROMPT_INVARIANT_ALERT_COOLDOWN_SEC:
                self._prompt_invariant_alert_ts[sig] = now_ts
                target_topic = self.ntfy_topic_error or self.ntfy_topic_server or self.ntfy_topic
                if target_topic:
                    body = "\n".join(lines[:8])
                    if len(violations) > 8:
                        body += f"\n... (+{len(violations) - 8} more)"
                    send_supervisor_ntfy_http(
                        f"Prompt Invariant Violation [{topic_key}] ({len(violations)})",
                        body,
                        "default",
                        target_topic,
                        True,
                        ntfy_url=self.ntfy_url,
                        ntfy_token=self.ntfy_token,
                        urlopen_fn=urllib.request.urlopen,
                        log=logger,
                    )
        return lines

    def synthesize_and_diagnose(
        self,
        sensors: TelemetrySensorSnapshot,
        logs: LogEventSummary,
        *,
        focus_symbols: Optional[List[str]] = None,
        topic_key: str,
        escalate_high_effort: bool = False,
        urgent_escalation: Optional[bool] = None,
        extreme_escalation: Optional[bool] = None,
        prev_prompt_delta_state: Optional[Dict[str, Any]] = None,
        ui_turn_pending: bool = False,
        prev_eval_was_actionable: bool = False,
        wake_reason: Optional[str] = None,
        prev_eval_ts: Optional[float] = None,
        prev_temporal_anchor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Feeds topic-appropriate telemetry to Gemini; the topic spec bounds both prompt scope and side effects."""
        spec = topic_spec(topic_key)
        resolved_topic = canonical_topic_key(topic_key)
        min_usd, max_usd, min_conf = 10.0, 2000.0, 0.85
        if spec.can_trade:
            gw = self.intent_gateway
            max_usd = safe_pos_float(getattr(gw, "max_notional_usd", None), default=2000.0) or 2000.0
            min_usd = safe_pos_float(getattr(gw, "min_notional_usd", None), default=10.0) or 10.0
            min_conf = safe_nonneg_float(getattr(gw, "min_confidence", None), default=0.85)

        effective_escalate = _parse_bool_flag(escalate_high_effort, False) or bool(
            focus_symbols and parse_symbol_list(focus_symbols)
        )
        effective_urgent = (
            _parse_bool_flag(urgent_escalation, False)
            or bool(focus_symbols and parse_symbol_list(focus_symbols))
            if urgent_escalation is not None
            else is_urgent_delta_gate_wakeup("", focus_symbols=focus_symbols, sensors=sensors)
        )
        effective_extreme = (
            _parse_bool_flag(extreme_escalation, False)
            if extreme_escalation is not None
            else is_extreme_stakes_delta_gate_wakeup("", sensors=sensors)
        )

        eval_ts = safe_pos_float(getattr(sensors, "evaluated_at_ts", None), default=None)
        iteration_num: Optional[int] = None
        try:
            iteration_num = get_topic_turn_count(topic_key) + 1
        except Exception:
            iteration_num = None
        prev_ts_norm = normalize_epoch_sec(prev_eval_ts, default=None, allow_iso=True) if prev_eval_ts is not None else None
        eff_now_iter = eval_ts if eval_ts is not None else time.time()
        prev_eval_elapsed_sec: Optional[float] = (
            max(0.0, eff_now_iter - prev_ts_norm)
            if prev_ts_norm is not None and prev_ts_norm > 0.0 and eff_now_iter >= prev_ts_norm - 5.0
            else None
        )

        full_delta_plan = None
        warm_delta_plan = None
        if spec.can_trade and not spec.can_self_heal:
            cur_cid, cur_comp_step, cur_is_warm = "", 0, False
            try:
                cur_cid, cur_comp_step, cur_is_warm = get_topic_thread_state(topic_key)
            except Exception:
                pass
            scope_s = getattr(spec, "asset_scope", "all")
            full_delta_plan = evaluate_prompt_delta_plan(
                sensors,
                asset_scope=scope_s,
                now_ts=eval_ts,
                min_usd=min_usd,
                max_usd=max_usd,
                is_warm_resume=False,
                prev_state=prev_prompt_delta_state,
                conversation_id=cur_cid,
                compacted_at_step=cur_comp_step,
            )
            warm_delta_plan = evaluate_prompt_delta_plan(
                sensors,
                asset_scope=scope_s,
                now_ts=eval_ts,
                min_usd=min_usd,
                max_usd=max_usd,
                is_warm_resume=cur_is_warm,
                prev_state=prev_prompt_delta_state,
                conversation_id=cur_cid,
                compacted_at_step=cur_comp_step,
                escalate_high_effort=effective_escalate,
                urgent_escalation=effective_urgent,
                extreme_escalation=effective_extreme,
                ui_turn_pending=bool(ui_turn_pending),
                focus_symbols=focus_symbols,
                prev_eval_was_actionable=bool(prev_eval_was_actionable),
            )

        prompt = build_supervisor_prompt(
            sensors,
            logs,
            spec,
            self.lookback_hours,
            focus_symbols=focus_symbols,
            min_usd=min_usd,
            max_usd=max_usd,
            min_conf=min_conf,
            compact_rules=False,
            delta_plan=full_delta_plan,
            wake_reason=wake_reason,
            iteration_num=iteration_num,
            prev_eval_elapsed_sec=prev_eval_elapsed_sec,
        )
        resume_prompt: Optional[str] = None
        if spec.can_trade and not spec.can_self_heal:
            resume_prompt = build_supervisor_prompt(
                sensors,
                logs,
                spec,
                self.lookback_hours,
                focus_symbols=focus_symbols,
                min_usd=min_usd,
                max_usd=max_usd,
                min_conf=min_conf,
                compact_rules=True,
                delta_plan=warm_delta_plan,
                wake_reason=wake_reason,
                iteration_num=iteration_num,
                prev_eval_elapsed_sec=prev_eval_elapsed_sec,
            )

        self._last_prompt_invariant_violations = []
        self._last_temporal_invariant_anchor = {}
        if spec.can_trade and not spec.can_self_heal:
            self._last_prompt_invariant_violations = self._validate_runtime_prompt_invariants(
                sensors,
                prompt,
                resume_prompt,
                topic_key=resolved_topic,
                min_usd=min_usd,
                prev_temporal_anchor=prev_temporal_anchor,
            )
            self._last_temporal_invariant_anchor = extract_temporal_invariant_anchor(sensors)

        model_override: Optional[str] = None
        allow_cli_fallback = False
        topic_override_key = getattr(spec, "override_env_key", None) or ""
        if (
            resolved_topic in ("auto_cryptotrade_executor", "auto_sharetrade_executor")
            and not (topic_override_key and (optional_env(topic_override_key, "") or "").strip())
            and not getattr(self.gemini_auditor, "_explicit_model", False)
        ):
            high_model = (optional_env("LLM_HIGH_STAKE_MODEL", "") or "").strip()
            xhigh_model = derive_xhigh_model_from_high(
                high_model,
                optional_env("LLM_XHIGH_STAKE_MODEL", ""),
            )
            oai_compat_model = resolve_configured_openai_compat_model()
            q_snap = self._load_cached_quota_snapshot()
            model_override, route_reason = resolve_crypto_executor_model_override(
                q_snap,
                high_model=high_model,
                default_model=str(getattr(self.gemini_auditor, "model", "") or ""),
                fallback_models=getattr(self.gemini_auditor, "fallback_models", ()) or (),
                escalate_high_effort=effective_escalate or effective_extreme,
                urgent_escalation=effective_urgent or effective_extreme,
                xhigh_model=xhigh_model,
                escalate_xhigh=effective_extreme,
                openai_compat_model=oai_compat_model,
            )
            if (
                route_reason.startswith("soft_conserve_fallback(")
                and model_override
                and strip_openai_compat_prefix(model_override)[0]
            ):
                allow_cli_fallback = True
            if route_reason == "escalate_xhigh_effort" and model_override:
                logger.info(
                    "[%s] Escalating reasoning effort to %s for extreme-stakes catalyst.",
                    resolved_topic,
                    model_override,
                )
            elif route_reason == "escalate_high_effort" and model_override:
                logger.info(
                    "[%s] Escalating reasoning effort to %s for high-signal cycle.",
                    resolved_topic,
                    model_override,
                )
            elif route_reason.startswith("suppress_"):
                logger.info(
                    "[%s] Suppressing high-effort escalation to %s (%s).",
                    resolved_topic,
                    high_model,
                    route_reason,
                )
            elif route_reason.startswith("soft_conserve_fallback(") and model_override:
                logger.info(
                    "[%s] Quota soft-conserve routing routine cycle to %s (%s).",
                    resolved_topic,
                    model_override,
                    route_reason,
                )
            elif route_reason.startswith("latency_demoted_fallback(") and model_override:
                logger.info(
                    "[%s] Latency fallback routing cycle to faster model %s (%s).",
                    resolved_topic,
                    model_override,
                    route_reason,
                )

        call_timeout = safe_pos_float(getattr(spec, "default_timeout_sec", None), default=None) or resolve_llm_timeout_sec()
        now_ts_str = time.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        self._last_model_used = ""
        self._last_query_succeeded = False
        self._last_prompt_delta_state = {}
        query_kwargs: Dict[str, Any] = {
            "subtitle": f"{self.mode.upper()} [{now_ts_str}]",
            "timeout_sec": call_timeout,
        }
        if model_override:
            query_kwargs["model"] = model_override
        if allow_cli_fallback:
            query_kwargs["allow_cli_fallback"] = True
        if resume_prompt:
            query_kwargs["resume_prompt"] = resume_prompt
        resp = query_topic_json(
            self.gemini_auditor,
            topic_key,
            prompt,
            **query_kwargs,
        )
        raw_used = getattr(self.gemini_auditor, "last_model_used", None)
        self._last_model_used = raw_used.strip() if isinstance(raw_used, str) and raw_used.strip() else ""

        if not resp or not isinstance(resp, dict):
            logger.warning("Gemini supervisor query failed. Falling back to default baseline.")
            return default_supervisor_fallback_plan(spec, logs)

        self._last_query_succeeded = True
        if full_delta_plan is not None:
            used_warm = bool(getattr(self.gemini_auditor, "last_used_warm_resume", False))
            chosen_plan = (
                warm_delta_plan
                if (used_warm and warm_delta_plan is not None)
                else full_delta_plan
            )
            chosen_state = dict(chosen_plan.next_state)
            try:
                post_cid, post_comp_step, _ = get_topic_thread_state(topic_key)
                if post_cid:
                    chosen_state["conversation_id"] = post_cid
                    chosen_state["compacted_at_step"] = post_comp_step
            except Exception:
                pass
            self._last_prompt_delta_state = chosen_state
        return resp

    def apply_reconciliation(
        self, plan: Dict[str, Any], *, allow_code_remediation: bool = True
    ) -> Tuple[bool, Optional[bool]]:
        """Applies or simulates venue/fleet repair actions and code remediation based on operational mode."""
        return apply_reconciliation_plan(
            plan,
            self.workspace_dir,
            is_enforce=(self.mode == "enforce"),
            allow_code_remediation=allow_code_remediation,
            execute_action_fn=lambda act: execute_venue_or_fleet_repair_action(
                act,
                self.workspace_dir,
                resolve_instrument_fn=lambda sym, venue=None: self.intent_gateway.resolve_instrument(sym, venue=venue),
                log=logger,
            ),
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
            is_bot_error=is_bot_or_code_error_report(report),
            is_sys_error=is_system_os_error_report(report),
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

    def write_audit_artifact(
        self,
        report: SupervisorAuditReport,
        *,
        persist_eval: bool = True,
    ) -> SupervisorAuditReport:
        """Persists the full audit report and heartbeat to disk in cachedb/."""
        persist_supervisor_audit_artifact(
            asdict(report),
            self.cache_dir,
            self.state_file,
            self.heartbeat_file,
            log=logger,
            persist_eval=persist_eval,
        )
        return report

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
            return self.write_audit_artifact(
                SupervisorAuditReport(ts=now, mode="off", summary="Supervisor mode is off; cycle skipped."),
                persist_eval=False,
            )

        scope = getattr(spec, "asset_scope", "all")
        if (
            scope == "equities"
            and not _parse_bool_flag(force_rth_open, False)
            and not bool_env("FORCE_SHARE_RTH_OPEN", default=False)
            and not is_us_rth_open(now)
        ):
            ny_now = datetime.fromtimestamp(now, tz=ZoneInfo("America/New_York"))
            hol_name = us_market_holiday_name(ny_now.date())
            early_name = None if hol_name else us_market_early_close_name(ny_now.date())
            if hol_name:
                msg = f"US equities market holiday ({hol_name}) -- closed; cycle skipped."
            elif early_name:
                msg = f"US equities early-close day ({early_name}, 09:30-13:00 America/New_York) closed; cycle skipped."
            else:
                msg = "US equities regular trading hours (09:30-16:00 America/New_York, Mon-Fri) closed; cycle skipped."
            logger.info("[SHARE EXECUTOR] %s", msg)
            return self.write_audit_artifact(
                SupervisorAuditReport(ts=now, mode=self.mode, summary=msg),
                persist_eval=False,
            )

        sensors = self.harvest_sensors_telemetry(
            include_fleet=spec.can_self_heal,
            include_market_portfolio=spec.can_trade,
            asset_scope=scope,
        )
        logs = (
            self.harvest_log_anomalies()
            if spec.can_self_heal
            else LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)
        )
        resolved_topic = canonical_topic_key(topic_key)
        escalate_high_effort = False
        urgent_escalation = False
        extreme_escalation = False
        gate_reason = ""
        delta_gate_meta: Dict[str, Any] = {}
        ui_turn_pending = False
        prev_eval: Dict[str, Any] = {}
        prev_prompt_delta_state: Dict[str, Any] = {}
        prev_temporal_anchor: Dict[str, Any] = {}
        prev_eval_was_actionable = False
        if spec.can_trade and not spec.can_self_heal:
            _read_json = _resolve_repair_json_reader(read_json_recovering)
            eval_disk_path = _resolve_topic_cache_path(self.cache_dir, f"{topic_key}_eval.json")
            prev_eval_raw = (
                _read_json(eval_disk_path, default={})
                if (os.path.exists(eval_disk_path) or os.path.exists(f"{eval_disk_path}.previous"))
                else {}
            )
            prev_eval = prev_eval_raw if isinstance(prev_eval_raw, dict) else {}
            raw_pds = prev_eval.get("prompt_delta_state")
            if isinstance(raw_pds, dict):
                prev_prompt_delta_state = raw_pds
            raw_ta = prev_eval.get("temporal_invariant_anchor")
            if isinstance(raw_ta, dict):
                prev_temporal_anchor = raw_ta
            raw_alert_ts = prev_eval.get("prompt_invariant_alert_ts")
            if isinstance(raw_alert_ts, dict):
                self._prompt_invariant_alert_ts = {
                    str(k): ts_v
                    for k, v in raw_alert_ts.items()
                    if (ts_v := safe_pos_float(v, default=None)) is not None
                }
            prev_eval_was_actionable = eval_doc_was_actionable(prev_eval)
        if resolved_topic in ("auto_cryptotrade_executor", "auto_sharetrade_executor"):
            delta_gate_enabled = bool_env("AI_DELTA_GATE_ENABLED", True)
            derived_itvs = resolve_derived_telemetry_intervals()
            is_share_exec = resolved_topic == "auto_sharetrade_executor"
            exec_interval_sec = (
                safe_pos_float(derived_itvs.get("share_executor_interval_sec"), default=1800.0) or 1800.0
                if is_share_exec
                else (safe_pos_float(derived_itvs.get("crypto_executor_interval_sec"), default=360.0) or 360.0)
            )
            max_idle_mult = safe_pos_float(optional_env("AI_DELTA_GATE_MAX_IDLE_MULT", ""), default=3.0) or 3.0
            max_idle_sec = exec_interval_sec * max_idle_mult
            min_move_pct = safe_pos_float(optional_env("AI_DELTA_GATE_MIN_PRICE_MOVE_PCT", ""), default=0.45) or 0.45
            high_effort_move_pct = safe_pos_float(optional_env("AI_HIGH_EFFORT_PRICE_MOVE_PCT", ""), default=1.25) or 1.25
            if delta_gate_enabled:
                try:
                    ui_turn_pending = has_pending_ui_turn(topic_key)
                except Exception:
                    ui_turn_pending = False
            should_skip_llm, gate_reason, escalate_high_effort, delta_gate_meta = evaluate_crypto_executor_delta_gate(
                sensors,
                prev_eval,
                now_ts=now,
                focus_symbols=focus_symbols,
                min_price_move_pct=min_move_pct,
                max_idle_sec=max_idle_sec,
                high_effort_price_move_pct=high_effort_move_pct,
                ui_turn_pending=ui_turn_pending,
                min_target_rr=getattr(self.intent_gateway, "min_target_rr", DEFAULT_MIN_TARGET_RR),
            )
            urgent_escalation = is_urgent_delta_gate_wakeup(
                gate_reason,
                focus_symbols=focus_symbols,
                sensors=sensors,
                prev_eval=prev_eval,
            )
            extreme_escalation = is_extreme_stakes_delta_gate_wakeup(
                gate_reason,
                sensors=sensors,
                prev_eval=prev_eval,
            )
            if delta_gate_enabled and should_skip_llm:
                prev_meta = prev_eval.get("delta_gate_meta") if isinstance(prev_eval.get("delta_gate_meta"), dict) else {}
                skip_summary = f"Delta-Gate HOLD carry-forward ({gate_reason}); redundant LLM call skipped."
                logger.info("[DELTA GATE] %s", skip_summary)
                skip_report = SupervisorAuditReport(
                    ts=now,
                    mode=self.mode,
                    summary=skip_summary,
                    market_trading_decision={"decision": "HOLD", "confidence": 0.0, "thesis": gate_reason},
                    market_trading_decisions=[{"decision": "HOLD", "confidence": 0.0, "thesis": gate_reason}],
                    sensor_health=build_sensor_health_summary(sensors),
                    actions_executed=False,
                    tests_passed=None,
                    notification_sent=False,
                    model_used="delta_gate_skip",
                    delta_gate_meta=dict(prev_meta),
                    prompt_delta_state=dict(prev_prompt_delta_state),
                    temporal_invariant_anchor=dict(prev_temporal_anchor),
                )
                return self.write_audit_artifact(skip_report, persist_eval=False)
            elif delta_gate_enabled:
                logger.info(
                    "[DELTA GATE] Wake LLM (%s, high_effort=%s%s).",
                    gate_reason,
                    escalate_high_effort,
                    ", xhigh_effort=True" if extreme_escalation else "",
                )

        plan = self.synthesize_and_diagnose(
            sensors,
            logs,
            focus_symbols=focus_symbols,
            topic_key=topic_key,
            escalate_high_effort=escalate_high_effort,
            urgent_escalation=urgent_escalation,
            extreme_escalation=extreme_escalation,
            prev_prompt_delta_state=prev_prompt_delta_state,
            ui_turn_pending=ui_turn_pending,
            prev_eval_was_actionable=prev_eval_was_actionable,
            wake_reason=gate_reason or None,
            prev_eval_ts=normalize_epoch_sec(prev_eval.get("ts"), default=None, allow_iso=True) if prev_eval else None,
            prev_temporal_anchor=prev_temporal_anchor or None,
        )
        cat_audit = (
            sensors.system_and_vpn.get("llm_catalog_audit")
            if spec.can_self_heal and isinstance(getattr(sensors, "system_and_vpn", None), dict)
            else None
        )
        auto_llm_patch_attached = False
        if spec.can_self_heal and isinstance(cat_audit, dict):
            rec_updates = cat_audit.get("recommended_config_updates")
            cur_fix = plan.get("code_remediation")
            has_existing_fix = isinstance(cur_fix, dict) and _parse_bool_flag(cur_fix.get("has_fix"), False)
            if isinstance(rec_updates, dict) and rec_updates and not has_existing_fix:
                if llm_patch := build_llm_config_env_remediation(self.workspace_dir, rec_updates):
                    plan["code_remediation"] = llm_patch
                    auto_llm_patch_attached = True
        executed, tests_passed = (
            self.apply_reconciliation(plan, allow_code_remediation=True)
            if spec.can_self_heal
            else (False, None)
        )
        if auto_llm_patch_attached and executed and tests_passed is True and isinstance(cat_audit, dict):
            if cat_path := str(cat_audit.get("catalog_path") or "").strip():
                persist_llm_model_catalog_state(cat_path, cat_audit)

        model_used = str(getattr(self, "_last_model_used", "") or "").strip()
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
            is_rth_open_fn=is_us_rth_open,
            model_used=model_used,
            log=logger,
        )
        if trade_executed:
            executed = True

        raw_issues = list(plan.get("diagnosed_issues") or []) if (spec.can_self_heal and isinstance(plan.get("diagnosed_issues"), list)) else []
        if spec.can_self_heal:
            missing_llms = (
                [str(m).strip() for m in (cat_audit.get("missing_configured_models") or []) if str(m).strip()]
                if isinstance(cat_audit, dict)
                else []
            )
            if missing_llms and not any(
                isinstance(it, dict)
                and str(it.get("category") or "").strip().upper() == "CONFIG_ERROR"
                and any(m in str(it.get("description") or "") for m in missing_llms)
                for it in raw_issues
            ):
                raw_issues.insert(
                    0,
                    {
                        "severity": "HIGH",
                        "category": "CONFIG_ERROR",
                        "description": f"Configured LLM model(s) missing from agy models: {', '.join(missing_llms)}",
                        "root_cause": "Model slug configured in config.env is not advertised by local agy CLI; update LLM_*_MODEL / LLM_FALLBACK_MODELS in config.env.",
                    },
                )
        raw_actions = plan.get("venue_actions") if spec.can_self_heal else []
        raw_fix = plan.get("code_remediation") if spec.can_self_heal else None
        report = SupervisorAuditReport(
            ts=now,
            mode=self.mode,
            summary=str(plan.get("summary") or "Audit complete."),
            diagnosed_issues=raw_issues,
            venue_actions=raw_actions if isinstance(raw_actions, list) else [],
            code_remediation=raw_fix if isinstance(raw_fix, dict) else {},
            market_trading_decision=market_decision,
            market_trading_decisions=decisions_list,
            intent_emitted=primary_intent,
            intents_emitted=intents_emitted,
            sensor_health=build_sensor_health_summary(sensors),
            actions_executed=executed,
            tests_passed=tests_passed,
            model_used=model_used,
            delta_gate_meta=delta_gate_meta if self._last_query_succeeded else {},
            prompt_delta_state=(
                self._last_prompt_delta_state
                if self._last_query_succeeded
                else dict(prev_prompt_delta_state)
            ),
            prompt_invariant_violations=list(self._last_prompt_invariant_violations),
            prompt_invariant_alert_ts=dict(self._prompt_invariant_alert_ts),
            temporal_invariant_anchor=dict(self._last_temporal_invariant_anchor or prev_temporal_anchor),
            completed_ts=time.time(),
        )
        # SRE/system/bot-error notifications (ntfy-server / ntfy-error) belong strictly to auto_technical_repair
        # (spec.can_self_heal); trade notifications are dispatched by AutonomousIntentGateway to ntfy-trade.
        if spec.can_self_heal:
            report.notification_sent = self.dispatch_notification(report, topic_key=topic_key)
        else:
            notified_statuses = (
                "OBSERVED_SHADOW",
                "EXECUTED",
                "BLOCKED_BY_GUARD",
                "REJECTED_AI_CIRCUIT_BREAKER",
                "FAILED",
            )
            report.notification_sent = bool(
                getattr(self.intent_gateway, "notify_enabled", False)
                and any(
                    isinstance(it, dict)
                    and str(it.get("decision") or "").upper() not in ("CANCEL", "HOLD_CANCEL")
                    and str(it.get("status") or "").upper() in notified_statuses
                    for it in intents_emitted
                )
            )
        self.write_audit_artifact(report)
        logger.info("Autonomous AI Supervisor cycle complete: %s", report.summary)
        return report


def main() -> None:
    """CLI entry point for the cron-driven supervisor topics."""
    import log  # noqa: F401 -- ensure stdout/stderr mirroring to logger/autonomous_ai_reconciler_*.log from startup

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
    parser.add_argument("--lookback-hours", type=parse_pos_cli_float, default=24.0,
                        help="Log anomaly lookback window in hours (default: 24; 0.2 for 10-minute runs)")
    parser.add_argument("--force-rth-open", action="store_true",
                        help="Allow auto_sharetrade_executor evaluation even outside US Regular Trading Hours")
    args = parser.parse_args()

    focus = parse_symbol_list(args.focus_symbols) or None
    reconciler = AutonomousAIReconciler(mode=args.mode, lookback_hours=args.lookback_hours)
    report = reconciler.run_cycle(
        topic_key=args.topic_key,
        focus_symbols=focus,
        force_rth_open=args.force_rth_open,
    )
    print(f"[{report.mode.upper()}] {args.topic_key}: {report.summary}")


if __name__ == "__main__":
    main()
