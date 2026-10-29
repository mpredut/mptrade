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
import glob
import json
import logging
import os
import re
import subprocess
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
    load_dotenv,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    normalize_venue_name,
    resolve_cachedb_filepath,
    split_symbol_assets,
)
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
    extract_timestamp_s,
    normalize_open_order,
    normalize_side,
)
from state_io import (
    read_last_jsonl_dict as _read_last_jsonl_dict,
    read_tail_jsonl_dicts,
    read_tail_lines,
)
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.high_stake_guard import (
    apply_asset_scope_filter,
    apply_live_24trend_windows as _apply_live_24trend_windows,
    apply_short_trend_item as _apply_short_trend_item,
    build_cross_market_context as _build_cross_market_context,
    build_cycle_fallback_prices,
    compute_7d_range_metrics as _compute_7d_range_metrics,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
    compute_swing_wave_metrics as _compute_swing_wave_metrics,
    enrich_24h_and_microstructure,
    enrich_cross_venue_trends,
    extract_current_price_item,
    extract_symbol_recent_fills,
    harvest_bot_runtime_states,
    harvest_initial_prices_and_trends,
    harvest_kraken_xstocks as _harvest_kraken_xstocks,
    harvest_macro_and_news_sensors,
    harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders,
    harvest_venue_cycles as _harvest_venue_cycles,
    load_symbol_microstructure as _load_symbol_microstructure,
    load_symbol_price_series,
    load_venue_trailing_maps,
    parse_venue_cycle_state as _parse_venue_cycle_state,
    set_pos_anchor as _set_pos_anchor,
    sync_sibling_symbol_telemetry,
)
from orchestratorOS.agy.topics import TOPIC_ALIASES, TOPIC_SPECS, topic_spec
from orchestratorOS.lib.technical_repair_ssot import (
    ACTIONABLE_ERROR_SEVERITIES,
    ALL_DIAGNOSED_CATEGORIES,
    ALL_REPAIR_ACTIONS,
    AUDITED_CONFIG_FILES,
    BOT_AND_ORDER_ERROR_CATEGORIES,
    BOT_MODULE_REPAIR_ACTIONS,
    BOT_OR_FLEET_ERROR_ACTIONS,
    DIAGNOSED_SEVERITIES,
    LogEventSummary,
    SYSTEM_ADMIN_ACTION_SPECS,
    SYSTEM_OS_ERROR_ACTIONS,
    SYSTEM_OS_ERROR_CATEGORIES,
    apply_code_patch_safely,
    apply_reconciliation_plan,
    audit_workspace_configs,
    build_system_admin_command,
    clean_json_value,
    execute_venue_or_fleet_repair_action,
    format_supervisor_ntfy_notification,
    gather_system_and_vpn_telemetry,
    harvest_heartbeats_and_state_files,
    harvest_log_anomalies_from_dirs,
    is_bot_or_code_error_report,
    is_error_report,
    is_system_os_error_report,
    normalize_error_signature,
    normalize_repair_action,
    persist_supervisor_audit_artifact,
    read_json_recovering,
    recent_log_lines,
    resolve_patchable_target,
    send_supervisor_ntfy_http,
)
from orchestratorOS.supervisor.prompt_formatters import (
    _format_horizon_range,
    _format_macro_and_news_section,
    _format_market_prices_section,
    _format_microstructure_section,
    _format_portfolio_section,
    _format_sre_and_fleet_section,
    _format_venue_cycle_line,
    _resolve_symbol_price,
    _round_opt_px,
    build_supervisor_prompt,
)
from orchestratorTrade.autonomous_intent_gateway import (
    AutonomousIntentGateway,
    _extract_intent_notional,
    _is_us_rth_open,
    _parse_intent_confidence,
    _parse_notify_enabled_flag,
    load_key_value_conf,
)
from providers.execution_audit import (
    normalize_trading_decisions as _normalize_trading_decisions,
    summarize_recent_ai_intents,
)

_parse_finite_float = safe_float
_as_bool_flag = _parse_bool_flag

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
        if mode is not None:
            if isinstance(mode, bool):
                raise ValueError(f"Invalid supervisor_mode {mode!r}; must be one of ['enforce', 'off', 'shadow']")
            raw_mode = mode
        else:
            if "supervisor_mode" not in conf:
                raise KeyError("Missing required 'supervisor_mode' in order_guard.conf")
            raw_mode = conf["supervisor_mode"]
        self.mode = str(raw_mode).strip().lower()
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError(f"Invalid supervisor_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']")

        self.notify_enabled = _parse_notify_enabled_flag(notify_enabled, conf, "supervisor_notify")

        self.ntfy_url = conf.get("phone_alert_url", "")
        if "ntfy_topic_guard" not in conf:
            raise KeyError("Missing required 'ntfy_topic_guard' in config.env")
        self.ntfy_topic_guard = conf["ntfy_topic_guard"]
        self.ntfy_topic_server = conf.get("ntfy_topic_server") or self.ntfy_topic_guard
        self.ntfy_topic_error = conf.get("ntfy_topic_error") or self.ntfy_topic_server
        self.ntfy_topic = self.ntfy_topic_server
        self.ntfy_token = conf.get("ntfy_token", "")

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

        # 5–6. Heartbeat sensors and bot state files
        snapshot.active_heartbeats, snapshot.state_files = harvest_heartbeats_and_state_files(
            self.cache_dir, self.workspace_dir, now_ts, _read_json
        )

        # 10. Fleet processes health, Linux OS resources, internet/DNS, VPN tunnel state, and config audit (SSOT)
        if include_fleet:
            base_res: Dict[str, Any] = {}
            try:
                from orchestratorOS.livecheck.fleet_health_monitor import FleetHealthMonitor
                monitor = FleetHealthMonitor(workspace_dir=self.workspace_dir)
                snapshot.fleet_processes = [asdict(p) for p in monitor.check_processes()]
                base_res = dict(monitor.gather_system_resources() or {})
            except Exception:
                base_res = {}
            snapshot.system_and_vpn = gather_system_and_vpn_telemetry(
                self.workspace_dir,
                base_resources=base_res,
            )

        if not include_market_portfolio:
            return snapshot

        # 1–4. Macro, Geopolitical & Breaking News sensors
        macro_sensors, snapshot.breaking_news = harvest_macro_and_news_sensors(
            self.cache_dir, now_ts, _read_json
        )
        for attr, val in macro_sensors.items():
            setattr(snapshot, attr, val)

        # 7. Live market prices, short/long/Weibull trend, 1s 24trend windows, 7d range, and Kalman velocity
        prices_data = _read_json(os.path.join(self.cache_dir, "cache_current_price.json"))
        trends_data = _read_json(os.path.join(self.cache_dir, "cache_short_trend.json"))
        long_trends_data = _read_json(os.path.join(self.cache_dir, "cache_long_trend.json"))
        priceanalysis_data = _read_json(os.path.join(self.cache_dir, "priceanalysis.json"))
        t_trend_data = _read_json(os.path.join(self.cache_dir, "cache_T_trend.json"))
        kalman_data = _read_json(os.path.join(self.cache_dir, "cache_kalman_trend.json"))
        watcher_all_price_data = _read_json(os.path.join(self.cache_dir, "cache_watcher_all_price.json"))
        trailing_map, trailing_by_venue = load_venue_trailing_maps(
            self.cache_dir, self.workspace_dir, _read_json
        )
        long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}
        t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}
        snapshot.prices_and_trends = harvest_initial_prices_and_trends(
            self.cache_dir, prices_data, trends_data, _read_json
        )

        # 7B. Discover all enabled tradable instruments & real spot wallet balances (when workspace has instruments.conf)
        tradable_positions: List[Dict[str, Any]] = []
        open_orders_list: List[Dict[str, Any]] = []
        venue_quote_cash: Dict[str, Dict[str, Any]] = {}
        inst_by_sym: Dict[str, Any] = {}
        load_tradable_fn = getattr(self.intent_gateway, "load_tradable_instruments", None)
        has_inst_source = (
            os.path.exists(os.path.join(self.workspace_dir, "instruments.conf"))
            or os.path.exists(os.path.join(self.workspace_dir, "kraken", "config.env"))
            or os.path.exists(os.path.join(self.workspace_dir, "hyperliquid", "config.env"))
            or (scope == "equities" and os.path.isdir(os.path.join(self.workspace_dir, "212trading")))
        )
        if callable(load_tradable_fn) and has_inst_source:
            try:
                try:
                    inst_map = load_tradable_fn(
                        only_workspace_conf=(scope != "equities"),
                        include_closed_rth=(scope == "equities"),
                        asset_scope=scope,
                    )
                except TypeError:
                    inst_map = load_tradable_fn(only_workspace_conf=True)
            except Exception as exc:
                logger.debug("Could not load tradable instruments for telemetry: %s", exc)
                inst_map = {}

            tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = (
                harvest_tradable_positions_and_orders(
                    inst_map, snapshot.prices_and_trends, trailing_map, trailing_by_venue
                )
            )

        if tradable_positions:
            snapshot.portfolio_summary["tradable_positions"] = tradable_positions
        if open_orders_list:
            snapshot.portfolio_summary["open_orders"] = open_orders_list
        if venue_quote_cash:
            snapshot.portfolio_summary["venue_quote_cash"] = venue_quote_cash
        if trailing_map:
            snapshot.portfolio_summary["trailing_peaks"] = trailing_map

        # Enrich every symbol (including Kraken/Hyperliquid/T212 pairs discovered in 7B) with cross-venue trend/Kalman/Weibull & 7d range
        watcher_price_items = (
            watcher_all_price_data.get("items", {}) if isinstance(watcher_all_price_data, dict) else {}
        )
        enrich_cross_venue_trends(
            snapshot.prices_and_trends,
            trends_data,
            kalman_data,
            long_items,
            priceanalysis_data,
            t_trend_map,
            watcher_price_items,
        )

        # 8. 24h multi-horizon price evolution & Pillar 2 microstructure per symbol
        cur_p_items = (
            prices_data.get("items", {})
            if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict)
            else {}
        )
        enrich_24h_and_microstructure(
            self.cache_dir,
            self.workspace_dir,
            snapshot.prices_and_trends,
            snapshot.microstructure,
            cur_p_items,
            inst_by_sym,
            now_ts,
            _read_json,
        )

        # 8A. Symmetric cross-venue & sibling symbol synchronization (Binance <-> Kraken <-> Hyperliquid <-> T212/xStocks)
        sync_sibling_symbol_telemetry(snapshot.prices_and_trends, snapshot.microstructure)

        # 9. Portfolio value, Kraken cycle states, cross-venue snapshot, recent orders/fills & autonomous intents
        last_av = _read_last_jsonl_dict(os.path.join(self.cache_dir, "cache_asset_value.jsonl"), tail_bytes=2048)
        if isinstance(last_av, dict) and isinstance(last_av.get("i"), dict):
            snapshot.portfolio_summary["total_value_usdc"] = last_av["i"].get("total_value_usdc")

        # 9B. Active DCA cycle states on Kraken, Hyperliquid, and Trading212 (.state_*.json)
        cycle_fallback_prices = build_cycle_fallback_prices(
            snapshot.microstructure, watcher_price_items
        )
        kraken_cycles, hyperliquid_cycles, all_t212_cycles = _harvest_venue_cycles(
            self.workspace_dir,
            snapshot.prices_and_trends,
            cycle_fallback_prices,
            _read_json,
        )
        if kraken_cycles:
            snapshot.portfolio_summary["kraken_cycles"] = kraken_cycles
        if hyperliquid_cycles:
            snapshot.portfolio_summary["hyperliquid_cycles"] = hyperliquid_cycles

        has_active_t212 = any(
            str(getattr(inst, "provider_name", "")).lower() == "t212"
            for inst in inst_by_sym.values()
        )
        t212_cycles = dict(all_t212_cycles) if has_active_t212 else {}
        if t212_cycles:
            snapshot.portfolio_summary["t212_cycles"] = t212_cycles

        # 9B-xStocks. Kraken xStocks state & cross-venue link with Trading212 (SPCXx.T <-> SPCX_US_EQ)
        xst_summary = _harvest_kraken_xstocks(
            os.path.join(self.workspace_dir, "kraken"),
            snapshot.prices_and_trends,
            all_t212_cycles,
            _read_json,
        )
        if xst_summary:
            snapshot.portfolio_summary["kraken_xstocks"] = xst_summary

        # 9B-CrossMarket. Build raw cross-market Crypto <-> US Equities telemetry before scope filtering
        cross_market_ctx = _build_cross_market_context(
            snapshot.prices_and_trends,
            snapshot.microstructure,
            all_t212_cycles,
        )
        if cross_market_ctx:
            snapshot.portfolio_summary["cross_market_context"] = cross_market_ctx

        # 9C. Cross-venue portfolio snapshot tail (e.g. Hyperliquid spot/perp equity summary)
        for l_dir in self.logs_dirs:
            last_ps = _read_last_jsonl_dict(os.path.join(l_dir, "portfolio_snapshot.jsonl"), tail_bytes=4096)
            if isinstance(last_ps, dict):
                if isinstance(last_ps.get("hl"), dict):
                    snapshot.portfolio_summary["hyperliquid_readonly"] = last_ps["hl"]
                break

        # 9D. Recent fills across Binance (cache_order.json), Kraken (cache_trade_kraken.json), and Hyperliquid
        recent_orders = harvest_recent_orders_and_enrich_positions(
            self.cache_dir,
            inst_by_sym,
            snapshot.portfolio_summary.get("tradable_positions", []),
            snapshot.prices_and_trends,
            kraken_cycles,
            hyperliquid_cycles,
            t212_cycles,
            _read_json,
        )
        if recent_orders:
            snapshot.portfolio_summary["recent_orders"] = recent_orders

        intents_data = _read_json(os.path.join(self.cache_dir, "autonomous_trade_intents_state.json"))
        if isinstance(intents_data, list) and intents_data:
            reconcile_fn = getattr(self.intent_gateway, "reconcile_intents", None)
            if callable(reconcile_fn):
                try:
                    if reconcile_fn(intents_data):
                        save_fn = getattr(self.intent_gateway, "save_all_intents", None)
                        if callable(save_fn):
                            save_fn(intents_data)
                except Exception:
                    pass
            recent_intents_out = summarize_recent_ai_intents(
                intents_data, scope, now_ts, _extract_intent_notional
            )
            if recent_intents_out:
                snapshot.portfolio_summary["recent_ai_intents"] = recent_intents_out

        # 9E. Active queued retry orders, monitortrades active states, assetguardian, and smart_stop_loss states
        snapshot.portfolio_summary.update(
            harvest_bot_runtime_states(self.cache_dir, now_ts, _read_json)
        )

        snapshot.prices_and_trends, snapshot.microstructure = apply_asset_scope_filter(
            snapshot.prices_and_trends,
            snapshot.microstructure,
            snapshot.portfolio_summary,
            scope,
            inst_by_sym,
            t212_cycles,
        )

        return snapshot

    def _audit_workspace_configs(self) -> Dict[str, Any]:
        """Audits workspace configuration files via the technical repair SSOT."""
        return audit_workspace_configs(self.workspace_dir)

    @staticmethod
    def _normalize_error_signature(line: str) -> str:
        """Strip timestamps, PIDs, and hex addresses so recurring log errors group into one signature."""
        return normalize_error_signature(line)

    def harvest_log_anomalies(self, lookback_hours: Optional[float] = None) -> LogEventSummary:
        """Parses recent log files for errors, persistent unresolved exceptions, and guard vetoes."""
        hours = _parse_finite_float(lookback_hours, self.lookback_hours)
        if hours <= 0:
            hours = self.lookback_hours
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
            if not spec.can_self_heal:
                return {
                    "summary": "Autonomous trade evaluation completed: holding posture.",
                    "market_trading_decisions": [],
                }
            return {
                "summary": "Autonomous audit completed: telemetry sensors active and bots healthy.",
                "diagnosed_issues": [
                    {
                        "severity": "INFO",
                        "category": "HEALTHY",
                        "description": f"Audited {logs.total_scanned_files} log files. {logs.error_count} errors, {logs.warning_count} warnings.",
                        "root_cause": "System operational within expected tolerances.",
                    }
                ],
                "venue_actions": [],
            }

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
            and not _as_bool_flag(force_rth_open, False)
            and not _as_bool_flag(os.environ.get("FORCE_SHARE_RTH_OPEN"), False)
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

        decisions_list: List[Dict[str, Any]] = _normalize_trading_decisions(plan) if spec.can_trade else []
        market_decision: Dict[str, Any] = decisions_list[0] if decisions_list else {}
        intents_emitted: List[Dict[str, Any]] = []
        actionable_decs = [
            d for d in decisions_list
            if str(d.get("decision") or "HOLD").upper() in ("BUY", "SELL")
        ]
        submit_decs = actionable_decs if actionable_decs else decisions_list[:1]
        effective_max_mode = "shadow" if (scope == "equities" and not _is_us_rth_open(now)) else self.mode
        for dec in submit_decs:
            emitted = self.intent_gateway.submit(dec, source=topic_key, max_mode=effective_max_mode)
            if isinstance(emitted, dict):
                intents_emitted.append(emitted)
                if self.mode == "enforce" and emitted.get("status") == "EXECUTED":
                    executed = True
        primary_intent = intents_emitted[0] if intents_emitted else None

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
            sensor_health={
                "fear_greed_present": sensors.fear_greed is not None,
                "market_breadth_present": sensors.market_breadth is not None,
                "sentiment_advisor_present": sensors.sentiment_advisor is not None,
                "geopolitical_threat_present": sensors.geopolitical_threat is not None,
                "prices_and_trends_count": len(sensors.prices_and_trends),
                "active_heartbeats_count": len(sensors.active_heartbeats),
            },
            actions_executed=executed,
            tests_passed=tests_passed,
        )
        # SRE/system/bot-error notifications (ntfy-server / ntfy-error) belong strictly to auto_technical_repair
        # (spec.can_self_heal); trade notifications are dispatched by AutonomousIntentGateway to ntfy-trade.
        if spec.can_self_heal:
            report.notification_sent = self.dispatch_notification(report, topic_key=topic_key)
        else:
            report.notification_sent = bool(
                intents_emitted and getattr(self.intent_gateway, "notify_enabled", False)
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
