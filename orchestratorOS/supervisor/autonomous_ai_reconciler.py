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
    apply_live_24trend_windows as _apply_live_24trend_windows,
    apply_short_trend_item as _apply_short_trend_item,
    build_cross_market_context as _build_cross_market_context,
    compute_7d_range_metrics as _compute_7d_range_metrics,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
    compute_swing_wave_metrics as _compute_swing_wave_metrics,
    extract_current_price_item,
    extract_symbol_recent_fills,
    harvest_kraken_xstocks as _harvest_kraken_xstocks,
    harvest_venue_cycles as _harvest_venue_cycles,
    load_symbol_microstructure as _load_symbol_microstructure,
    load_symbol_price_series,
    parse_venue_cycle_state as _parse_venue_cycle_state,
    set_pos_anchor as _set_pos_anchor,
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
    audit_workspace_configs,
    build_system_admin_command,
    clean_json_value,
    gather_system_and_vpn_telemetry,
    harvest_log_anomalies_from_dirs,
    is_bot_or_code_error_report,
    is_error_report,
    is_system_os_error_report,
    normalize_error_signature,
    normalize_repair_action,
    read_json_recovering,
    recent_log_lines,
    resolve_patchable_target,
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
)
from orchestratorTrade.autonomous_intent_gateway import (
    AutonomousIntentGateway,
    _extract_intent_notional,
    _is_us_rth_open,
    _parse_intent_confidence,
    _parse_notify_enabled_flag,
    load_key_value_conf,
)

_parse_finite_float = safe_float
_as_bool_flag = _parse_bool_flag

load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("orchestratorOS.supervisor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _normalize_trading_decisions(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and deduplicate single or multi-coin trading decisions from Gemini's JSON plan."""
    if not isinstance(plan, dict):
        return []

    raw_multi = None
    for k in ("market_trading_decisions", "trading_decisions", "decisions"):
        if isinstance(plan.get(k), list):
            raw_multi = plan.get(k)
            break
    raw_single = plan.get("market_trading_decision")
    candidates: List[Dict[str, Any]] = []
    for item in (
        *(raw_multi if isinstance(raw_multi, list) else ()),
        *([raw_single] if isinstance(raw_single, dict) and raw_single else ()),
    ):
        if isinstance(item, dict):
            d = dict(item)
            raw_val = d.get("decision") or d.get("action") or d.get("side")
            if isinstance(raw_val, bool) or raw_val is None:
                raw_side = "HOLD"
            else:
                raw_side = normalize_side(raw_val) or str(raw_val).strip().upper() or "HOLD"
            d["decision"] = raw_side
            v_norm = normalize_venue_name(d.get("venue"))
            if v_norm:
                d["venue"] = v_norm
            if "confidence" in d and d.get("confidence") is not None:
                d["confidence"] = _parse_intent_confidence(d.get("confidence"), default=0.0)
            extracted_usd = _extract_intent_notional(d, default=None)
            if extracted_usd is not None:
                d["suggested_notional_usd"] = extracted_usd
            candidates.append(d)

    def _sort_key(d: Dict[str, Any]) -> Tuple[int, float]:
        side = str(d.get("decision") or "HOLD").strip().upper()
        is_action = 1 if side in ("BUY", "SELL") else 0
        conf = _parse_intent_confidence(d.get("confidence"), default=0.0)
        return (-is_action, -conf)

    candidates.sort(key=_sort_key)

    decisions: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str, str]] = set()
    for item in candidates:
        sym_raw = str(item.get("symbol") or "").strip().upper()
        sym_canon = canonical_base_asset(sym_raw) or sym_raw
        venue_norm = normalize_venue_name(item.get("venue"))
        side_norm = str(item.get("decision") or "HOLD").strip().upper()
        key = (
            (venue_norm, sym_canon, "")
            if sym_canon
            else (venue_norm, "", side_norm)
        )
        if key not in seen_keys:
            seen_keys.add(key)
            decisions.append(item)
    return decisions


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

        def _with_age_min(payload: Dict[str, Any], fpath: str) -> Dict[str, Any]:
            enriched = dict(payload)
            ref_ts = extract_timestamp_s(enriched)
            eff_path = str(resolve_cachedb_filepath(fpath))
            if ref_ts is None and os.path.exists(eff_path):
                try:
                    ref_ts = os.path.getmtime(eff_path)
                except OSError:
                    ref_ts = None
            if ref_ts is not None:
                enriched.setdefault("age_min", round(max(0.0, now_ts - ref_ts) / 60.0, 1))
            return enriched

        # 5. Heartbeat sensors
        for hb_name in (
            "macro_analyzer.heartbeat",
            "globaltelemetry_collector.heartbeat",
            "rtrade.heartbeat",
        ):
            data = _read_json(os.path.join(self.cache_dir, hb_name))
            if isinstance(data, dict) and data:
                snapshot.active_heartbeats[hb_name] = data

        # 6. Bot state files in workspace
        for s_file in (".watchdog_state.json", ".anomaly_watchdog_state.json"):
            s_path = os.path.join(self.workspace_dir, s_file)
            if os.path.exists(s_path):
                try:
                    mtime = os.path.getmtime(s_path)
                    snapshot.state_files[s_file] = {"age_hours": max(0.0, (now_ts - mtime) / 3600.0)}
                except OSError:
                    pass

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

        # 1. Fear & Greed sensor
        fg_path = os.path.join(self.cache_dir, "fear_greed_collect.json")
        data = _read_json(fg_path)
        if isinstance(data, dict) and data:
            snapshot.fear_greed = _with_age_min(data, fg_path)

        # 2. Market Breadth sensor
        mb_path = os.path.join(self.cache_dir, "market_breadth_collect.json")
        data = _read_json(mb_path)
        if isinstance(data, dict) and data:
            snapshot.market_breadth = _with_age_min(data, mb_path)

        # 3. Macro Sentiment Advisor sensor
        sa_path = os.path.join(self.cache_dir, "sentiment_eval.json")
        data = _read_json(sa_path)
        if isinstance(data, dict) and data:
            snapshot.sentiment_advisor = _with_age_min(data, sa_path)

        # 4. Geopolitical Threat Shield sensor + live 1-minute Breaking News RSS feed
        gt_path = os.path.join(self.cache_dir, "geopolitical_eval.json")
        data = _read_json(gt_path)
        if isinstance(data, dict) and data:
            snapshot.geopolitical_threat = _with_age_min(data, gt_path)

        news_data = _read_json(os.path.join(self.cache_dir, "news_feed_collect.json"))
        if isinstance(news_data, list) and news_data:
            valid_news = [item for item in news_data if isinstance(item, dict) and item.get("title")]
            valid_news.sort(
                key=lambda x: (
                    1 if x.get("is_high_severity") else 0,
                    safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
                ),
                reverse=True,
            )
            for item in valid_news[:6]:
                seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
                age_m = round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None
                snapshot.breaking_news.append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "is_high_severity": bool(item.get("is_high_severity")),
                    "keywords": item.get("matched_keywords") or [],
                    "age_min": age_m,
                })

        # 7. Live market prices, short/long/Weibull trend, 1s 24trend windows, 7d range, and Kalman velocity
        prices_data = _read_json(os.path.join(self.cache_dir, "cache_current_price.json"))
        trends_data = _read_json(os.path.join(self.cache_dir, "cache_short_trend.json"))
        long_trends_data = _read_json(os.path.join(self.cache_dir, "cache_long_trend.json"))
        priceanalysis_data = _read_json(os.path.join(self.cache_dir, "priceanalysis.json"))
        t_trend_data = _read_json(os.path.join(self.cache_dir, "cache_T_trend.json"))
        kalman_data = _read_json(os.path.join(self.cache_dir, "cache_kalman_trend.json"))
        watcher_all_price_data = _read_json(os.path.join(self.cache_dir, "cache_watcher_all_price.json"))
        trailing_map: Dict[str, Dict[str, Any]] = {}
        trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]] = {"binance": {}, "kraken": {}}
        for trail_venue, raw_trail, default_quote in (
            ("binance", _read_json(os.path.join(self.cache_dir, "trailing_state.json")), ""),
            ("kraken", _read_json(os.path.join(self.workspace_dir, "kraken", "trailing_state.json")), "USD"),
        ):
            if not isinstance(raw_trail, dict):
                continue
            for k_asset, k_st in raw_trail.items():
                k_up = str(k_asset).strip().upper()
                if not k_up:
                    continue
                k_key = k_up if (not default_quote or split_symbol_assets(k_up) is not None) else f"{k_up}{default_quote}"
                norm_st: Optional[Dict[str, Any]] = None
                if isinstance(k_st, dict) and k_st.get("peak") is not None:
                    norm_st = k_st
                else:
                    k_pk = safe_pos_float(k_st, default=0.0) or 0.0
                    if k_pk > 0:
                        norm_st = {"peak": k_pk}
                if norm_st is not None:
                    trailing_map.setdefault(k_key, norm_st)
                    trailing_by_venue[trail_venue].setdefault(k_key, norm_st)

        price_map: Dict[str, float] = {}
        if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict):
            for sym, item in prices_data["items"].items():
                px_val = extract_current_price_item(item)
                if px_val is not None:
                    price_map[sym] = px_val

        long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}
        t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}

        if isinstance(trends_data, dict):
            trends_items = trends_data.get("items") if isinstance(trends_data.get("items"), dict) else trends_data
            for sym, tinfo in trends_items.items():
                if sym in ("items", "fetchtime", "rest_sync_cursor", "metadata", "counts"):
                    continue
                if isinstance(tinfo, dict):
                    entry: Dict[str, Any] = {}
                    _apply_short_trend_item(entry, tinfo, fallback_price=price_map.get(sym))
                    snapshot.prices_and_trends[sym] = entry

        for sym, px in price_map.items():
            if sym not in snapshot.prices_and_trends:
                snapshot.prices_and_trends[sym] = {"price": px, "trend": 0, "growth_coefficient": 0.0}

        # Overlay fresh 1-second sampled dynamic windows from cachedb/24trend/live_<SYM>.json
        live_24trend_dir = str(resolve_cachedb_filepath(os.path.join(self.cache_dir, "24trend")))
        if os.path.isdir(live_24trend_dir):
            for lt_path in sorted(glob.glob(os.path.join(live_24trend_dir, "live_*.json"))):
                fname = os.path.basename(lt_path)
                sym = fname[len("live_") : -len(".json")].upper()
                if not sym:
                    continue
                lt_win = _read_json(lt_path)
                if not isinstance(lt_win, dict):
                    continue
                entry = snapshot.prices_and_trends.setdefault(sym, {})
                _apply_live_24trend_windows(entry, lt_win, fallback_price=price_map.get(sym))

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

            balance_memo: Dict[Tuple[str, str], Optional[float]] = {}

            def _query_free_bal(inst_obj: Any, asset: str) -> Optional[float]:
                if not asset:
                    return None
                v_name = str(getattr(inst_obj, "provider_name", "")).lower()
                key = (v_name, asset.upper())
                if key in balance_memo:
                    return balance_memo[key]
                prov = getattr(inst_obj, "_provider", None)
                free_fn = getattr(prov, "free_balance", None)
                val_f: Optional[float] = None
                if callable(free_fn):
                    try:
                        raw = free_fn(asset)
                        if isinstance(raw, (int, float)):
                            val_f = safe_float(raw, default=None)
                    except Exception:
                        val_f = None
                balance_memo[key] = val_f
                return val_f

            if isinstance(inst_map, dict):
                for inst in inst_map.values():
                    sym = str(getattr(inst, "symbol", "")).upper()
                    venue = str(getattr(inst, "provider_name", "")).lower()
                    base = str(getattr(inst, "base", "")).upper()
                    quote = str(getattr(inst, "quote", "")).upper()
                    if not sym or not venue:
                        continue
                    inst_by_sym[sym] = inst

                    px = snapshot.prices_and_trends.get(sym, {}).get("price")
                    px_f = safe_pos_float(px, default=0.0) or 0.0
                    if px_f <= 0:
                        try:
                            get_px_fn = getattr(inst._provider, "get_current_price", None)
                            if callable(get_px_fn):
                                live_px = safe_pos_float(get_px_fn(sym), default=0.0) or 0.0
                                if live_px > 0:
                                    px = live_px
                                    px_f = live_px
                        except Exception:
                            px = None
                            px_f = 0.0
                    if px_f <= 0:
                        px_f = _resolve_symbol_price(sym, snapshot.prices_and_trends)
                        if px_f > 0:
                            px = px_f
                    sym_entry = snapshot.prices_and_trends.setdefault(
                        sym, {"price": px if px_f > 0 else None, "trend": 0, "growth_coefficient": 0.0}
                    )
                    if px_f > 0 and (safe_pos_float(sym_entry.get("price"), default=0.0) or 0.0) <= 0:
                        sym_entry["price"] = px

                    base_free = _query_free_bal(inst, base)
                    quote_free = _query_free_bal(inst, quote)
                    quote_breakdown: Optional[Dict[str, float]] = None
                    if quote.upper() == "USD":
                        bd_fn = getattr(inst._provider, "usd_cash_breakdown", None)
                        if callable(bd_fn):
                            try:
                                raw_bd = bd_fn()
                                if isinstance(raw_bd, dict) and raw_bd:
                                    quote_breakdown = {
                                        str(k): round(float(v), 2)
                                        for k, v in raw_bd.items()
                                        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
                                    } or None
                            except Exception:
                                quote_breakdown = None
                    if quote_free is not None:
                        cash_entry: Dict[str, Any] = {
                            "venue": venue,
                            "asset": quote,
                            "free": round(quote_free, 2),
                        }
                        if quote_breakdown:
                            cash_entry["breakdown"] = quote_breakdown
                        venue_quote_cash[f"{venue}:{quote}"] = cash_entry

                    pos_val_usd = (
                        round(base_free * px_f, 2)
                        if base_free is not None and px_f > 0
                        else None
                    )
                    params = getattr(inst, "params", {}) if isinstance(getattr(inst, "params", None), dict) else {}
                    v_norm = normalize_venue_name(venue)
                    v_trail_map = trailing_by_venue.get(v_norm, {})
                    t_state = next(
                        (
                            v_trail_map[c]
                            for c in cache_symbol_variants(sym, include_original=True, include_base=True)
                            if isinstance(v_trail_map.get(c), dict)
                        ),
                        {},
                    )
                    if not t_state and v_norm == "binance" and isinstance(trailing_map.get(sym), dict):
                        t_state = trailing_map[sym]
                    peak = t_state.get("peak")
                    peak_f = _parse_finite_float(peak, 0.0)
                    if peak_f > 0 and px_f > 0 and peak_f < px_f:
                        peak_f = px_f
                    dd_from_peak_pct = (
                        round(((px_f - peak_f) / peak_f) * 100.0, 2)
                        if px_f > 0 and peak_f > 0
                        else None
                    )
                    pos_entry: Dict[str, Any] = {
                        "venue": venue,
                        "symbol": sym,
                        "base": base,
                        "quote": quote,
                        "price": _round_opt_px(px, 6) if px is not None else None,
                        "free_base": round(base_free, 6) if base_free is not None else None,
                        "value_usd": pos_val_usd,
                        "free_quote": round(quote_free, 2) if quote_free is not None else None,
                    }
                    if quote_breakdown:
                        pos_entry["free_quote_breakdown"] = quote_breakdown
                    if peak is not None and peak_f > 0:
                        pos_entry["trailing_peak"] = round(peak_f, 6)
                        pos_entry["dd_from_peak_pct"] = dd_from_peak_pct
                    if params.get("trailing.pct"):
                        pos_entry["trailing_pct_cfg"] = params.get("trailing.pct")
                    if params.get("mt.gain") or params.get("mt.lost"):
                        pos_entry["mt_gain_pct"] = params.get("mt.gain")
                        pos_entry["mt_lost_pct"] = params.get("mt.lost")
                    tradable_positions.append(pos_entry)

                    open_fn = getattr(inst, "open_orders", None)
                    if callable(open_fn):
                        try:
                            raw_open = open_fn()
                            if isinstance(raw_open, list):
                                for oo in raw_open:
                                    if isinstance(oo, dict):
                                        open_orders_list.append(
                                            normalize_open_order(oo, venue=venue, symbol=sym)
                                        )
                        except Exception:
                            pass

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
        trends_items_map = (
            trends_data.get("items")
            if isinstance(trends_data, dict) and isinstance(trends_data.get("items"), dict)
            else (trends_data if isinstance(trends_data, dict) else {})
        )
        for sym, entry in list(snapshot.prices_and_trends.items()):
            cand_keys = tuple(cache_symbol_variants(sym, include_original=True, include_base=True))

            # Inherit short/dynamic slopes from cross-venue sibling (e.g. TAOUSDC -> TAOUSD, BTCUSDC -> XBTUSD) when missing
            has_slopes = any(
                entry.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h")
            )
            if not has_slopes and isinstance(trends_items_map, dict):
                for ck in cand_keys[1:]:
                    tinfo = trends_items_map.get(ck)
                    if isinstance(tinfo, dict) and _apply_short_trend_item(entry, tinfo, require_any_slope=True):
                        has_slopes = True
                        break
            if not has_slopes:
                for ck in cand_keys[1:]:
                    sib = snapshot.prices_and_trends.get(ck)
                    if isinstance(sib, dict) and any(
                        sib.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h")
                    ):
                        for k_inh in (
                            "slope_90s",
                            "trend_90s",
                            "growth_90s",
                            "slope_3_7m",
                            "trend_3_7m",
                            "growth_3_7m",
                            "slope_15m",
                            "trend_15m",
                            "growth_15m",
                            "slope_2_5h",
                            "trend_2_5h",
                            "growth_2_5h",
                            "gradient_recent_fast",
                            "epsilon",
                            "vol_1h_pct",
                        ):
                            if k_inh in sib and sib.get(k_inh) is not None and not entry.get(k_inh):
                                entry[k_inh] = sib[k_inh]
                        break

            if isinstance(kalman_data, dict):
                k_sym = next((kalman_data[ck] for ck in cand_keys if isinstance(kalman_data.get(ck), dict)), None)
                if isinstance(k_sym, dict):
                    entry["kalman_vel"] = k_sym.get("kalman_vel")
                    entry["kalman_vel_std"] = k_sym.get("kalman_vel_std")
                    entry["kalman_trend"] = k_sym.get("kalman_trend")
                    if entry.get("vol_1h_pct") is None and k_sym.get("vol_1h_pct") is not None:
                        entry["vol_1h_pct"] = k_sym.get("vol_1h_pct")
                    if k_sym.get("adapt_reentry_pct") is not None:
                        entry["adapt_reentry_pct"] = k_sym.get("adapt_reentry_pct")
                    if k_sym.get("adapt_dca_pct") is not None:
                        entry["adapt_dca_pct"] = k_sym.get("adapt_dca_pct")
            lt_obj: Optional[Dict[str, Any]] = None
            if isinstance(long_items, dict):
                for ck in cand_keys:
                    raw_lt = long_items.get(ck)
                    if isinstance(raw_lt, list):
                        valid_lts = [x for x in raw_lt if isinstance(x, dict) and x.get("direction")]
                        if valid_lts:
                            lt_obj = valid_lts[-1]
                            break
                    elif isinstance(raw_lt, dict) and raw_lt.get("direction"):
                        lt_obj = raw_lt
                        break
            if lt_obj is None and isinstance(priceanalysis_data, dict):
                pa_map = (
                    priceanalysis_data.get("items")
                    if isinstance(priceanalysis_data.get("items"), dict)
                    else priceanalysis_data
                )
                if isinstance(pa_map, dict):
                    lt_obj = next(
                        (
                            pa_map[ck]
                            for ck in cand_keys
                            if isinstance(pa_map.get(ck), dict) and pa_map[ck].get("direction")
                        ),
                        None,
                    )
            if isinstance(lt_obj, dict):
                entry["long_trend_dir"] = lt_obj.get("direction")
                dur_s = _parse_finite_float(lt_obj.get("duration_seconds"), 0.0)
                entry["long_trend_hours"] = round(dur_s / 3600.0, 1)
                est_rem = _parse_finite_float(lt_obj.get("estimated_future_hours"), -1.0)
                if est_rem >= 0.0:
                    entry["long_trend_est_rem_hours"] = round(est_rem, 1)
            tt = next((t_trend_map[ck] for ck in cand_keys if isinstance(t_trend_map.get(ck), dict)), None)
            if isinstance(tt, dict):
                entry["weibull_median_d"] = tt.get("median_d")
                entry["weibull_p90_d"] = tt.get("p90_d")
            if isinstance(watcher_price_items, dict):
                sym_px_cur = _parse_finite_float(entry.get("price"), 0.0)
                m_series = None
                for ck in cand_keys:
                    cand_ser = watcher_price_items.get(ck)
                    if not isinstance(cand_ser, list) or len(cand_ser) < 2:
                        continue
                    last_m_px = (
                        _parse_finite_float(cand_ser[-1][1], 0.0)
                        if isinstance(cand_ser[-1], (list, tuple)) and len(cand_ser[-1]) > 1
                        else 0.0
                    )
                    if (
                        ck != sym
                        and sym_px_cur > 0
                        and last_m_px > 0
                        and abs(sym_px_cur - last_m_px) / last_m_px > 0.10
                    ):
                        continue
                    m_series = cand_ser
                    break
                if isinstance(m_series, list) and len(m_series) >= 2:
                    vals = [
                        _parse_finite_float(pt[1], 0.0)
                        for pt in m_series
                        if isinstance(pt, list) and len(pt) > 1 and _parse_finite_float(pt[1], 0.0) > 0
                    ]
                    entry.update(_compute_7d_range_metrics(vals))

        # 8. 24h multi-horizon price evolution & Pillar 2 microstructure per symbol
        cur_p_items = (
            prices_data.get("items", {})
            if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict)
            else {}
        )
        btc_series_24h, _ = load_symbol_price_series(
            self.cache_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False
        )
        for sym in list(snapshot.prices_and_trends.keys()):
            sym_px_cur = _parse_finite_float(snapshot.prices_and_trends[sym].get("price"), 0.0)
            matched_series_24h, arch_vals_7d = load_symbol_price_series(
                self.cache_dir, sym, current_price_items=cur_p_items
            )
            if sym_px_cur > 0 and matched_series_24h:
                last_24_px = _parse_finite_float(matched_series_24h[-1][1], 0.0)
                if last_24_px > 0 and abs(sym_px_cur - last_24_px) / last_24_px > 0.10:
                    matched_series_24h = []
            if sym_px_cur > 0 and arch_vals_7d:
                last_7d_px = _parse_finite_float(arch_vals_7d[-1], 0.0)
                if last_7d_px > 0 and abs(sym_px_cur - last_7d_px) / last_7d_px > 0.10:
                    arch_vals_7d = []
            if (
                _parse_finite_float(snapshot.prices_and_trends[sym].get("price"), 0.0) <= 0.0
                and matched_series_24h
            ):
                snapshot.prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
            elif (
                _parse_finite_float(snapshot.prices_and_trends[sym].get("price"), 0.0) <= 0.0
                and arch_vals_7d
            ):
                snapshot.prices_and_trends[sym]["price"] = arch_vals_7d[-1]
            m24: Dict[str, Any] = compute_24h_price_metrics(matched_series_24h) or {} if matched_series_24h else {}
            if arch_vals_7d and len(arch_vals_7d) >= 2 and "chg_7d_pct" not in snapshot.prices_and_trends[sym]:
                snapshot.prices_and_trends[sym].update(_compute_7d_range_metrics(arch_vals_7d))
            if not m24 and sym in inst_by_sym:
                # Fallback for standalone Kraken/Hyperliquid/T212 pairs without a cachedb/cache_24_price_* file
                prov = getattr(inst_by_sym[sym], "_provider", None)
                ohlc_fn = getattr(prov, "ohlc_series", None)
                if callable(ohlc_fn):
                    try:
                        s5 = ohlc_fn(sym, 5)
                        cutoff_24h = now_ts - 86400.0
                        pts_24h = [
                            [int(ts * 1000), float(px)]
                            for ts, px in zip(getattr(s5, "timestamps", ()), getattr(s5, "closes", ()))
                            if ts >= cutoff_24h and float(px) > 0
                        ]
                        m24 = compute_24h_price_metrics(pts_24h) or {}
                        if m24:
                            matched_series_24h = pts_24h
                        if "chg_7d_pct" not in snapshot.prices_and_trends[sym]:
                            s60 = ohlc_fn(sym, 60)
                            cutoff_7d = now_ts - 7 * 86400.0
                            vals_7d = [
                                float(px)
                                for ts, px in zip(getattr(s60, "timestamps", ()), getattr(s60, "closes", ()))
                                if ts >= cutoff_7d and float(px) > 0
                            ]
                            snapshot.prices_and_trends[sym].update(_compute_7d_range_metrics(vals_7d))
                    except Exception:
                        pass
                if not m24 and normalize_venue_name(getattr(inst_by_sym[sym], "provider_name", "")) == "t212":
                    try:
                        t212_dir = os.path.join(self.workspace_dir, "212trading")
                        if t212_dir not in sys.path:
                            sys.path.insert(0, t212_dir)
                        from market_data import _chart
                        yahoo_sym = sym.split("_", 1)[0]
                        chart_res = _chart(yahoo_sym, rng="5d", interval="5m")
                        pts_5d: List[List[Any]] = []
                        if isinstance(chart_res, tuple) and len(chart_res) >= 2 and isinstance(chart_res[1], list):
                            pts_5d = [
                                [int(ts_f * 1000), round(px_f, 4)]
                                for bar in chart_res[1]
                                if isinstance(bar, (list, tuple))
                                and len(bar) >= 2
                                and (ts_f := safe_pos_float(bar[0], default=None)) is not None
                                and (px_f := safe_pos_float(bar[1], default=None)) is not None
                            ]
                        elif isinstance(chart_res, dict):
                            ts_list = chart_res.get("timestamp") or []
                            closes = (((chart_res.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
                            pts_5d = [
                                [int(ts_f * 1000), round(c_f, 4)]
                                for ts, c in zip(ts_list, closes)
                                if (ts_f := safe_pos_float(ts, default=None)) is not None
                                and (c_f := safe_pos_float(c, default=None)) is not None
                            ]
                        if pts_5d:
                            cutoff_24h_ms = int((now_ts - 86400.0) * 1000)
                            pts_24h = [pt for pt in pts_5d if pt[0] >= cutoff_24h_ms]
                            if len(pts_24h) < 10:
                                pts_24h = pts_5d[-78:]
                            m24 = compute_24h_price_metrics(pts_24h) or {}
                            if m24:
                                matched_series_24h = pts_24h
                            vals_5d = [pt[1] for pt in pts_5d]
                            if len(vals_5d) >= 2 and "chg_7d_pct" not in snapshot.prices_and_trends[sym]:
                                snapshot.prices_and_trends[sym].update(
                                    _compute_7d_range_metrics(vals_5d, decimals=4)
                                )
                    except Exception:
                        pass
            if m24:
                snapshot.prices_and_trends[sym].update(m24)
                if matched_series_24h and _parse_finite_float(snapshot.prices_and_trends[sym].get("price"), 0.0) <= 0.0:
                    snapshot.prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
                if snapshot.prices_and_trends[sym].get("trend", 0) == 0 and "chg_1h_pct" in m24:
                    c1h = _parse_finite_float(m24.get("chg_1h_pct"), 0.0)
                    if c1h >= 0.25:
                        snapshot.prices_and_trends[sym]["trend"] = 1
                    elif c1h <= -0.25:
                        snapshot.prices_and_trends[sym]["trend"] = -1
                if (
                    matched_series_24h
                    and btc_series_24h
                    and not sym.endswith(("_EQ", "_US_EQ"))
                    and canonical_base_asset(sym) != "BTC"
                ):
                    cross_btc = compute_cross_btc_metrics(matched_series_24h, btc_series_24h)
                    for ck in ("alt_btc_pos_24h_pct", "alt_btc_dd_24h_high_pct", "alt_btc_bounce_24h_low_pct"):
                        if cross_btc.get(ck) is not None:
                            snapshot.prices_and_trends[sym][ck] = cross_btc[ck]

            m_info = _load_symbol_microstructure(self.cache_dir, sym, _read_json)
            if m_info:
                snapshot.microstructure[sym] = m_info

        # Also harvest reference microstructure files in cachedb/ (e.g. ADAUSD from KRAKEN_CACHE_PAIRS)
        if os.path.isdir(self.cache_dir):
            covered_micro_bases = {canonical_base_asset(k) for k in snapshot.microstructure}
            ref_micro_syms: List[str] = []
            for pfx in ("orderbook_depth_", "derivatives_telemetry_", "whale_snapshot_"):
                for search_dir in (self.cache_dir, os.path.join(self.cache_dir, "telemetry")):
                    for mpath in sorted(glob.glob(os.path.join(search_dir, f"{pfx}*.json"))):
                        mfname = os.path.basename(mpath)
                        msym = mfname[len(pfx) : -len(".json")].upper()
                        if not msym or not msym.isalnum() or msym in snapshot.microstructure or msym in ref_micro_syms:
                            continue
                        mbase = canonical_base_asset(msym)
                        if mbase not in covered_micro_bases:
                            ref_micro_syms.append(msym)
                            covered_micro_bases.add(mbase)
            for msym in ref_micro_syms:
                m_info_ref = _load_symbol_microstructure(self.cache_dir, msym, _read_json)
                if m_info_ref:
                    snapshot.microstructure[msym] = m_info_ref

        # 8A. Symmetric cross-venue & sibling symbol synchronization (Binance <-> Kraken <-> Hyperliquid <-> T212/xStocks)
        range_bound_keys = {
            "low_24h", "high_24h", "pos_24h_pct", "dd_24h_high_pct", "bounce_24h_low_pct",
            "low_7d", "high_7d", "pos_7d_pct", "dd_7d_high_pct", "bounce_7d_low_pct", "span_7d_pct",
            "swing_low_px", "swing_high_px", "major_swing_low_px",
            "rally_from_swing_low_pct", "pullback_from_swing_high_pct", "rally_from_major_low_pct",
        }
        for sym, pinfo in list(snapshot.prices_and_trends.items()):
            if not isinstance(pinfo, dict):
                continue
            sym_variants = cache_symbol_variants(sym, include_original=True, include_base=True)
            for cand in sym_variants:
                if cand == sym:
                    continue
                sib_pinfo = snapshot.prices_and_trends.get(cand)
                if isinstance(sib_pinfo, dict):
                    p_px = _parse_finite_float(pinfo.get("price"), 0.0)
                    s_px = _parse_finite_float(sib_pinfo.get("price"), 0.0)
                    if p_px <= 0.0 and s_px > 0.0:
                        pinfo["price"] = sib_pinfo["price"]
                        p_px = s_px
                    px_diverged = p_px > 0.0 and s_px > 0.0 and abs(p_px - s_px) / s_px > 0.10
                    for k, v in sib_pinfo.items():
                        if v is None:
                            continue
                        if px_diverged and k in range_bound_keys:
                            continue
                        if k not in pinfo or pinfo.get(k) is None:
                            pinfo[k] = v
                        elif k in ("trend", "growth_coefficient") and pinfo.get(k) in (0, 0.0) and v not in (0, 0.0):
                            pinfo[k] = v
                sib_micro = snapshot.microstructure.get(cand)
                if isinstance(sib_micro, dict) and sib_micro:
                    if sym not in snapshot.microstructure:
                        snapshot.microstructure[sym] = dict(sib_micro)
                    else:
                        for mk, mv in sib_micro.items():
                            if snapshot.microstructure[sym].get(mk) is None and mv is not None:
                                snapshot.microstructure[sym][mk] = mv

        # 9. Portfolio value, Kraken cycle states, cross-venue snapshot, recent orders/fills & autonomous intents
        last_av = _read_last_jsonl_dict(os.path.join(self.cache_dir, "cache_asset_value.jsonl"), tail_bytes=2048)
        if isinstance(last_av, dict) and isinstance(last_av.get("i"), dict):
            snapshot.portfolio_summary["total_value_usdc"] = last_av["i"].get("total_value_usdc")

        # 9B. Active DCA cycle states on Kraken, Hyperliquid, and Trading212 (.state_*.json)
        cycle_fallback_prices: Dict[str, float] = {}
        for m_sym, m_dict in snapshot.microstructure.items():
            if isinstance(m_dict, dict):
                ob_mid_v = safe_pos_float(m_dict.get("ob_mid_price"), default=0.0) or 0.0
                if ob_mid_v > 0:
                    cycle_fallback_prices[m_sym] = ob_mid_v
        if isinstance(watcher_price_items, dict):
            for m_sym, m_ser in watcher_price_items.items():
                if isinstance(m_ser, list) and m_ser and isinstance(m_ser[-1], (list, tuple)) and len(m_ser[-1]) > 1:
                    m_px = safe_pos_float(m_ser[-1][1], default=0.0) or 0.0
                    if m_px > 0:
                        cycle_fallback_prices.setdefault(str(m_sym).upper(), m_px)

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
        recent_orders: Dict[str, Any] = {}
        venue_recent_orders: Dict[str, Dict[str, Any]] = {
            "binance": {},
            "kraken": {},
            "hyperliquid": {},
            "t212": {},
        }
        for order_cache_name, cache_venue in (
            ("cache_order.json", "binance"),
            ("cache_trade_kraken.json", "kraken"),
        ):
            orders_data = _read_json(os.path.join(self.cache_dir, order_cache_name))
            if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
                for sym, o_list in orders_data["items"].items():
                    fills = extract_symbol_recent_fills(o_list)
                    if fills:
                        venue_recent_orders[cache_venue][sym] = fills
                        recent_orders.setdefault(sym, fills)
        for sym, inst in inst_by_sym.items():
            v_inst = normalize_venue_name(getattr(inst, "provider_name", ""))
            if v_inst != "hyperliquid" or sym in venue_recent_orders["hyperliquid"]:
                continue
            try:
                prov = getattr(inst, "_provider", None)
                if prov is not None and hasattr(prov, "get_orders"):
                    hl_orders = prov.get_orders(sym, None, 90 * 86400.0)
                    fills = extract_symbol_recent_fills(hl_orders)
                    if fills:
                        venue_recent_orders["hyperliquid"][sym] = fills
                        recent_orders[sym] = fills
            except Exception:
                pass
        if recent_orders:
            snapshot.portfolio_summary["recent_orders"] = recent_orders
        for pos in snapshot.portfolio_summary.get("tradable_positions", []):
            if not isinstance(pos, dict):
                continue
            p_sym = str(pos.get("symbol") or "").upper()
            p_venue = normalize_venue_name(pos.get("venue"))
            p_variants = cache_symbol_variants(p_sym)
            pos_px_f = _parse_finite_float(pos.get("price"), 0.0)
            if pos_px_f <= 0:
                cand_px = _resolve_symbol_price(p_sym, snapshot.prices_and_trends)
                if cand_px > 0:
                    pos_px_f = cand_px
                    pos["price"] = round(cand_px, 6)
                    fb_val = _parse_finite_float(pos.get("free_base"), -1.0)
                    if fb_val >= 0:
                        pos["value_usd"] = round(fb_val * cand_px, 2)
                    pk_val = _parse_finite_float(pos.get("trailing_peak"), 0.0)
                    if pk_val > 0:
                        if pk_val < cand_px:
                            pk_val = cand_px
                            pos["trailing_peak"] = round(pk_val, 6)
                        pos["dd_from_peak_pct"] = round(((cand_px - pk_val) / pk_val) * 100.0, 2)
            v_ro_map = venue_recent_orders.get(p_venue, {})
            ro = next(
                (v_ro_map[c] for c in p_variants if isinstance(v_ro_map.get(c), dict)),
                None,
            )
            if isinstance(ro, dict):
                lb = ro.get("last_buy")
                ls = ro.get("last_sell")
                if ro.get("last_action_side"):
                    pos["last_action_side"] = str(ro.get("last_action_side")).upper()
                _set_pos_anchor(pos, "active_avg_buy_px", "pnl_vs_avg_pct", ro.get("active_avg_buy_px"), pos_px_f)
                _set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", lb.get("price") if isinstance(lb, dict) else None, pos_px_f)
                _set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", ls.get("price") if isinstance(ls, dict) else None, pos_px_f)
                r_buys = ro.get("recent_buys") if isinstance(ro.get("recent_buys"), list) else []
                low_buy_px = min(
                    (
                        _parse_finite_float(b.get("price"), 0.0)
                        for b in r_buys
                        if isinstance(b, dict) and _parse_finite_float(b.get("price"), 0.0) > 0
                    ),
                    default=0.0,
                )
                if low_buy_px > 0:
                    _set_pos_anchor(pos, "lowest_buy_px", "pnl_vs_lowest_buy_pct", low_buy_px, pos_px_f)
            venue_cyc_maps = (
                (hyperliquid_cycles,)
                if p_venue == "hyperliquid"
                else (t212_cycles,)
                if p_venue == "t212"
                else (kraken_cycles,)
                if p_venue == "kraken"
                else ()
            )
            for cyc_map in venue_cyc_maps:
                cyc = cyc_map.get(p_sym)
                if not isinstance(cyc, dict):
                    cyc = next((cyc_map[c] for c in p_variants if isinstance(cyc_map.get(c), dict)), None)
                if not isinstance(cyc, dict):
                    continue
                if pos.get("free_base") is None and cyc.get("qty") is not None:
                    cyc_qty = safe_nonneg_float(cyc.get("qty"), default=None)
                    if cyc_qty is not None:
                        pos["free_base"] = round(cyc_qty, 6)
                        if pos_px_f > 0:
                            pos["value_usd"] = round(cyc_qty * pos_px_f, 2)
                cyc_avg = cyc.get("avg_cost_px") or cyc.get("entry_price") or cyc.get("last_buy_price")
                _set_pos_anchor(pos, "active_avg_buy_px", "pnl_vs_avg_pct", cyc_avg, pos_px_f, overwrite=False)
                _set_pos_anchor(pos, "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct", cyc.get("avg_cost_px"), pos_px_f, overwrite=False)
                _set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", cyc.get("last_buy_price"), pos_px_f, overwrite=False)
                _set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", cyc.get("last_sell_price"), pos_px_f, overwrite=False)
                _set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", cyc.get("pos_peak"), pos_px_f, overwrite=False)

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
            scoped_intents = [
                i
                for i in intents_data
                if isinstance(i, dict)
                and (
                    scope not in ("equities", "crypto")
                    or (normalize_venue_name(i.get("venue")) == "t212") == (scope == "equities")
                )
            ]
            recent_intents_out: List[Dict[str, Any]] = []
            for i in scoped_intents[-8:]:
                exec_pay = i.get("execution_payload") if isinstance(i.get("execution_payload"), dict) else {}
                hist = i.get("history") if isinstance(i.get("history"), list) else []
                last_hist = hist[-1] if hist and isinstance(hist[-1], dict) else {}
                eff_notional = (
                    safe_pos_float(exec_pay.get("cummulativeQuoteQty"), default=None)
                    or safe_pos_float(exec_pay.get("cost"), default=None)
                    or safe_pos_float(i.get("clamped_notional_usd"), default=None)
                    or safe_pos_float(last_hist.get("notional_usd"), default=None)
                    or _extract_intent_notional(i, default=None)
                )
                ts_val = _parse_finite_float(i.get("timestamp"), 0.0)
                intent_summary: Dict[str, Any] = {
                    "decision": i.get("decision"),
                    "symbol": i.get("symbol"),
                    "venue": normalize_venue_name(i.get("venue")) or i.get("venue"),
                    "status": i.get("status"),
                    "notional_usd": round(eff_notional, 2) if eff_notional is not None else i.get("suggested_notional_usd"),
                    "ts": i.get("timestamp"),
                }
                if ts_val > 0 and now_ts >= ts_val:
                    intent_summary["age_h"] = round((now_ts - ts_val) / 3600.0, 2)
                recent_intents_out.append(intent_summary)
            if recent_intents_out:
                snapshot.portfolio_summary["recent_ai_intents"] = recent_intents_out

        # 9E. Active queued retry orders (cachedb/order_retry_queue.jsonl), monitortrades active states (mt_state.json),
        # and recent exchange execution audit records (logger/execution_audit/execution_audit_*.jsonl)
        retry_q_path = str(resolve_cachedb_filepath(os.path.join(self.cache_dir, "order_retry_queue.jsonl")))
        if os.path.exists(retry_q_path):
            active_retries: List[Dict[str, Any]] = []
            for rec in read_tail_jsonl_dicts(retry_q_path, limit=50):
                lifecycle = str(rec.get("lifecycle") or "").lower()
                exp_ts = _parse_finite_float(rec.get("expires_at_ts"), 0.0)
                if lifecycle in ("submit_pending", "pending", "active") and (exp_ts == 0.0 or exp_ts > now_ts):
                    active_retries.append({
                        "venue": str(rec.get("provider_name") or "").lower(),
                        "symbol": rec.get("symbol"),
                        "side": rec.get("side"),
                        "qty": rec.get("qty"),
                        "requested_price": rec.get("requested_price"),
                        "kind": rec.get("kind"),
                        "attempts": rec.get("attempts"),
                        "last_failure_reason": rec.get("last_failure_reason"),
                    })
            if active_retries:
                snapshot.portfolio_summary["active_retry_queue"] = active_retries

        mt_state = _read_json(os.path.join(self.cache_dir, "mt_state.json"))
        if isinstance(mt_state, dict):
            active_mt = {
                k: v
                for k in ("tp_ride_state", "sl_bounce_state", "sl_episode_state", "reentry_state", "pending_stop_exit")
                for v in (mt_state.get(k),)
                if isinstance(v, dict) and v
            }
            if active_mt:
                snapshot.portfolio_summary["monitortrades_active_state"] = active_mt

        ag_state = _read_json(os.path.join(self.cache_dir, "assetguardian_state.json"))
        if isinstance(ag_state, dict) and isinstance(ag_state.get("symbols"), dict) and ag_state["symbols"]:
            snapshot.portfolio_summary["assetguardian_state"] = ag_state["symbols"]

        ssl_state = _read_json(os.path.join(self.cache_dir, "smart_stop_loss_state.json"))
        if isinstance(ssl_state, dict) and ssl_state:
            snapshot.portfolio_summary["smart_stop_loss_state"] = ssl_state

        if scope in ("equities", "crypto"):
            want_t212 = scope == "equities"
            if want_t212:
                eq_syms = set(inst_by_sym.keys()) | set(t212_cycles.keys())
                snapshot.prices_and_trends = {
                    k: v for k, v in snapshot.prices_and_trends.items() if k in eq_syms
                }
                snapshot.microstructure = {}
                for crypto_key in (
                    "total_value_usdc",
                    "kraken_cycles",
                    "hyperliquid_cycles",
                    "hyperliquid_signal",
                    "hyperliquid_readonly",
                    "recent_orders",
                    "monitortrades_active_state",
                    "trailing_peaks",
                ):
                    snapshot.portfolio_summary.pop(crypto_key, None)
            else:
                snapshot.prices_and_trends = {
                    k: v for k, v in snapshot.prices_and_trends.items()
                    if not str(k).upper().endswith(("_EQ", "_US_EQ"))
                }
                snapshot.portfolio_summary.pop("t212_cycles", None)
            for list_key in ("recent_ai_intents", "active_retry_queue"):
                if list_key in snapshot.portfolio_summary:
                    filtered_items = [
                        item for item in snapshot.portfolio_summary[list_key]
                        if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
                    ]
                    if filtered_items:
                        snapshot.portfolio_summary[list_key] = filtered_items
                    else:
                        snapshot.portfolio_summary.pop(list_key, None)

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
        is_pure_sre = bool(spec.can_self_heal and not spec.can_trade)

        context_items: List[str] = []
        if not is_pure_sre:
            context_items.extend(_format_macro_and_news_section(sensors))
        if spec.can_self_heal:
            context_items.extend(_format_sre_and_fleet_section(sensors, logs, self.lookback_hours))
        if not is_pure_sre and sensors.prices_and_trends:
            context_items.extend(_format_market_prices_section(sensors.prices_and_trends))
        if not is_pure_sre and sensors.microstructure:
            context_items.extend(_format_microstructure_section(sensors.microstructure, sensors.prices_and_trends))
        if not is_pure_sre and sensors.portfolio_summary:
            context_items.extend(
                _format_portfolio_section(sensors.portfolio_summary, getattr(spec, "asset_scope", "all"))
            )
        if not is_pure_sre and focus_symbols:
            raw_focus = [focus_symbols] if isinstance(focus_symbols, str) else focus_symbols
            clean_syms = [s.strip().upper() for s in raw_focus if isinstance(s, str) and s.strip()]
            if clean_syms:
                context_items.append(
                    f"\n### 4. TACTICAL PRIORITY SYMBOLS: {', '.join(clean_syms)}\n"
                    "- Pay heightened attention to these symbols alongside all other live tradable coins above."
                )

        context_str = "\n".join(context_items)

        tasks: List[str] = []
        schema: List[str] = []
        if spec.can_self_heal:
            sev_union = " | ".join(f'"{s}"' for s in DIAGNOSED_SEVERITIES)
            cat_union = " | ".join(f'"{c}"' for c in ALL_DIAGNOSED_CATEGORIES)
            act_union = " | ".join(f'"{a}"' for a in ALL_REPAIR_ACTIONS)
            cfg_list_str = ", ".join(AUDITED_CONFIG_FILES)
            tasks.extend([
                f"1. Audit functional system anomalies: fleet processes and modules (procs.conf), bot heartbeats, Linux OS resources (CPU load, RAM, disk), internet & VPN tunnel (wgpia0), configuration files ({cfg_list_str}), and persistent unresolved errors in logs: explain the technical root cause of any bot exceptions, fail-closed blocks, stale heartbeats, OS/network/VPN issues, or configuration errors.",
                "2. Determine whether hung/errored bots or modules require restart (RESTART_BOT / RESTART_MODULE), whether the entire fleet or VPN/DNS requires restart (RESTART_FLEET / RESTART_VPN / PIN_DNS), or whether a critical unrecoverable Linux OS / kernel / system-wide hang requires a full machine reboot (REBOOT_SYSTEM).",
                "3. If a reproducible code bug or configuration error exists in the workspace, formulate an exact search_block and replace_block in `code_remediation` to fix the code or configuration file.",
            ])
            schema.extend([
                '  "summary": "1-2 sentence executive technical assessment of functional system, Linux OS, internet/VPN, configurations, and fleet health"',
                f'  "diagnosed_issues": [{{"severity": {sev_union}, '
                f'"category": {cat_union}, '
                '"description": "Short explanation", "root_cause": "Detailed technical root cause"}]',
                f'  "venue_actions": [{{"action": {act_union}, '
                '"venue": "fleet" | "system" | "binance" | "kraken" | "hyperliquid" | "t212", "symbol": "bot_or_module_name or SYSTEM", '
                '"order_id": "optional order id", "reason": "Why this technical repair or restart/reboot action is needed"}]',
                '  "code_remediation": {"has_fix": false, "target_file": "", "rationale": "", '
                '"search_block": "", "replace_block": ""}',
            ])
        else:
            schema.append(
                '  "summary": "1-2 sentence executive assessment of multi-coin market posture and trading decisions"'
            )
        if spec.can_trade:
            gw = self.intent_gateway
            max_usd = safe_pos_float(getattr(gw, "max_notional_usd", None), default=2000.0) or 2000.0
            min_usd = safe_pos_float(getattr(gw, "min_notional_usd", None), default=10.0) or 10.0
            min_conf = safe_nonneg_float(getattr(gw, "min_confidence", None), default=0.85)
            is_equities = getattr(spec, "asset_scope", "all") == "equities"
            target_desc = (
                "ALL live tradable US equity shares on 't212' (SPCX_US_EQ, NVDA_US_EQ, RGNT_US_EQ during US Regular Trading Hours, cross-referencing Kraken read-only xStocks SPCXx.T exposure)"
                if is_equities
                else "ALL live tradable crypto spot coins (BTCUSDC, TAOUSDC, ARBUSDC on 'binance'; HYPEUSDC intensively on 'hyperliquid'; HYPEUSD on 'kraken', while TAOUSD/ADAUSD on Kraken serve as cross-venue reference telemetry)"
            )
            venue_schema = '"t212"' if is_equities else '"binance" | "kraken" | "hyperliquid"'
            ex_sym = '"NVDA_US_EQ"' if is_equities else '"BTCUSDC"'
            tasks.append(
                f"{len(tasks) + 1}. Formulate symmetric autonomous trading decisions across {target_desc} by synthesizing the "
                "entire MPTrade system telemetry as a whole:\n"
                "   - FULL ACCESS TO EVERYTHING IN `cachedb/` AS A DECISION INFORMATION SOURCE: The telemetry above is a compact "
                "synthesized dashboard (zero raw file dumps); whenever you want deeper detail, you have direct, unrestricted read access "
                "to the entire `cachedb/` (`cachedb/prices/`, `cachedb/trends/`, `cachedb/trends/24trend/`, `cachedb/telemetry/`, "
                "`cachedb/macro/`, `cachedb/orders/`, `cachedb/state/`), `logger/execution_audit/`, `logs/`, and `providers.market_api` "
                "to inspect any file or query any venue on your own.\n"
                "   - BUSINESS OBJECTIVE & FULL AUTONOMOUS JUDGMENT (ZERO PROFIT OR STRATEGY GUARDS): Your objective "
                "is to maximize net portfolio value and risk-adjusted returns (net of venue fees) across all platforms while "
                "protecting capital. ALL deterministic order_guard gates (G_PROF profit reference, GO1 daily cap/cooldown, "
                "GT_* trend deferrals, G_QLIMIT* weight caps, GI_* microstructure, and GLLM* secondary vetoes) are BYPASSED "
                "symmetrically on both BUY and SELL as binding constraints. Deterministic bots (`tradeall.py`, `monitortrades.py`, `assetguardian.py`, "
                "`rtrade.py`, `kraken/bot.py`, `hyperliquid/bot.py`, `212trading/bot.py`) and rigid code filters/configs "
                "(`order_guard.py`, `order_guard.conf`, `instruments.conf`, `intelligence/`) continue working in parallel in the "
                "system and can be consulted by you on demand as advisory indicators and reference guidance—you have full quantitative authority "
                "to validate, adapt, or override them according to live market structure, with ZERO obligation to consult or follow them.\n"
                "   - UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING: No rigid trading rules, fixed percentage thresholds, "
                "or directional biases are imposed on you—judge 100% freely on your own WHEN, WHAT, HOW MUCH, on WHICH platform "
                "(`binance`, `kraken`, `hyperliquid`, `t212`), and WHY to `BUY`, `SELL`, or `HOLD`. Do NOT restrict yourself to tiny "
                "fixed slices when conviction or profitability is clear: when a position is clearly in profit (e.g. above "
                "`active_avg_buy_px` / `last_buy_px` or after a strong upward swing), you are completely free to sell a large "
                f"quantity or the entire profitable position (`50%–100%` of `value_usd`, up to {max_usd:g} USD) at once.\n"
                f"   - Emit one entry per actionable instrument in `market_trading_decisions` with `confidence >= {min_conf:g}` "
                f"and `suggested_notional_usd` sized between {min_usd:g} and {max_usd:g} USD (the gateway automatically "
                "clamps notional strictly to available free wallet balance with zero artificial cooldowns or dedup locks). "
                "If no instrument warrants action right now, return a single 'HOLD' entry."
            )
            schema.append(
                f'  "market_trading_decisions": [{{"decision": "BUY" | "SELL" | "HOLD", "symbol": {ex_sym}, '
                f'"venue": {venue_schema}, "confidence": 0.90, '
                f'"suggested_notional_usd": 1000.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}}]'
            )

        if spec.can_self_heal:
            role_intro = (
                "You are the Lead SRE and Autonomous Technical Repair Supervisor for MPTrade.\n"
                "Your STRICT mandate is solely to detect functional system anomalies (bots, fleet modules, heartbeats), "
                "Linux OS resource issues, internet & VPN (wgpia0) connectivity failures, configuration errors, and "
                "persistent unresolved errors in logs, and to remediate them by patching code or configurations and "
                "restarting/rebooting bots, modules, the trading fleet, VPN/DNS, or the entire Linux system.\n"
                "Do NOT analyze or report on market prices, trends/slopes, macro sentiment, portfolio balances, or active trading positions.\n\n"
                f"{context_str}\n\n"
                "Perform an executive technical diagnostic and self-healing audit:\n" + "\n".join(tasks) + "\n\n"
            )
            tool_rule = (
                "- If you diagnose a reproducible code bug or configuration error and need to verify exact lines before proposing `code_remediation`, "
                "you MAY read the target file in the workspace so `search_block` matches verbatim.\n"
            )
            report_format_rule = (
                "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
                "  1) A concise, well-structured Markdown report in **Romanian** strictly covering technical & SRE health "
                "(bullet points for: Starea Flotei, Boților, Sistemului Linux & VPN (wgpia0) & Loguri, "
                "Conectivitate Internet & Audit Configurații, Erori Persistente Nerezolvate, și "
                "Acțiuni de Remediere Cod/Config & Reboot Boți/Module/Flotă/Sistem). "
                "NU analiza și NU menționa prețuri de piață, pante, portofoliu, solduri sau poziții active!\n"
            )
        else:
            role_intro = (
                "You are the Quantitative Autonomous Portfolio Manager and Trade Executor for MPTrade.\n"
                "Analyze the complete market telemetry, macro/geopolitical sensors, microstructure, spot wallet balances, and cycle states:\n\n"
                f"{context_str}\n\n"
                "Evaluate the market context and formulate autonomous BUY/SELL trading decisions:\n" + "\n".join(tasks) + "\n\n"
            )
            tool_rule = (
                "- You have full, unrestricted access to everything in `cachedb/` (`cachedb/prices/`, `cachedb/trends/`, `cachedb/trends/24trend/`, "
                "`cachedb/telemetry/`, `cachedb/macro/`, `cachedb/orders/`, `cachedb/state/`) as a direct source of information for your trading decisions, "
                "as well as full freedom to query live market APIs (`providers.market_api`, `binance_api`, `kraken`, `hyperliquid`, `212trading`) "
                "and read trade/execution logs (`logs/`, `logger/execution_audit/`) whenever you want deeper verification or to answer user questions.\n"
                "- Deterministic bots (`tradeall.py`, `monitortrades.py`, `assetguardian.py`, `rtrade.py`, venue bots) and rigid code filters/guards "
                "(`order_guard.py`, `order_guard.conf`, `instruments.conf`, `intelligence/`) run in parallel and may be inspected on demand if you wish, "
                "but consulting or following them is never required.\n"
                "- Keep this thread focused on market analysis, portfolio management, and autonomous BUY/SELL execution across venues.\n"
            )
            report_format_rule = (
                "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
                "  1) A concise, well-structured Markdown report in **Romanian** (bullet points for: "
                "Piața Live & Pante (Toate Monedele), Portofoliu, Solduri & Poziții Active, și Decizia AI de Tranzacționare / Răspuns la intervențiile din chat).\n"
            )

        prompt = (
            role_intro
            + "INTERACTIVE THREAD RULES:\n"
            "- If the user wrote any tactical instructions, price targets, overrides, or questions earlier in this "
            "conversation thread, honor them with highest priority and address them directly.\n"
            + tool_rule
            + report_format_rule
            + "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
            "```json\n{\n" + ",\n".join(schema) + "\n}\n```"
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
        if not isinstance(act, dict):
            return False
        norm = normalize_repair_action(act, self.workspace_dir)
        action_type = norm["action"]
        venue = norm["venue"]
        symbol = norm["symbol"]
        order_id = norm["order_id"]
        reason = norm["reason"]
        matched = norm["matched_process"]

        if action_type in BOT_MODULE_REPAIR_ACTIONS:
            if not symbol:
                logger.warning("[ENFORCE] %s missing bot/module name in action: %s", action_type, act)
                return False
            try:
                if matched is None:
                    logger.warning("[ENFORCE] Refusing %s for unknown process %r (not in procs.conf)", action_type, symbol)
                    return False
                pattern = matched.get("clean_pattern") or matched["pattern"].rstrip("$")
                logger.warning("[ENFORCE EXECUTING] Restarting fleet bot/module %s (pattern=%s): %s", matched["name"], pattern, reason)
                subprocess.run(["pkill", "-f", pattern], check=False)
                return True
            except Exception as e:
                logger.error("[ENFORCE] Failed restarting fleet bot/module %s: %s", symbol, e)
                return False

        if action_type in SYSTEM_ADMIN_ACTION_SPECS:
            try:
                built = build_system_admin_command(action_type)
                if built is None:
                    return False
                cmd, desc = built
                logger.warning("[ENFORCE EXECUTING] %s: %s", desc, reason)
                res = subprocess.run(cmd, check=False)
                return getattr(res, "returncode", 0) == 0
            except Exception as e:
                logger.error("[ENFORCE] Failed %s: %s", action_type, e)
                return False

        if action_type == "CANCEL_ORDER":
            if not symbol or not order_id:
                logger.warning("[ENFORCE] CANCEL_ORDER missing symbol or order_id: %s", act)
                return False
            try:
                inst = self.intent_gateway.resolve_instrument(symbol.upper(), venue=venue)
                if inst is None:
                    logger.warning("[ENFORCE] CANCEL_ORDER could not resolve instrument %s on %s", symbol, venue)
                    return False
                logger.warning("[ENFORCE EXECUTING] Canceling order %s for %s on %s: %s",
                               order_id, inst.symbol, inst.provider_name, reason)
                if getattr(inst, "_api", None) is not None and hasattr(inst._api, "cancel_order"):
                    inst._api.cancel_order(inst.symbol, order_id, provider_name=inst.provider_name)
                elif getattr(inst, "_provider", None) is not None and hasattr(inst._provider, "cancel_order"):
                    inst._provider.cancel_order(inst.symbol, order_id)
                else:
                    logger.warning("[ENFORCE] Provider for %s does not expose cancel_order", inst.symbol)
                    return False
                return True
            except Exception as e:
                logger.error("[ENFORCE] Failed CANCEL_ORDER %s on %s (%s): %s", order_id, symbol, venue, e)
                return False

        if action_type == "SYNC_POSITION":
            if not symbol:
                return False
            try:
                inst = self.intent_gateway.resolve_instrument(symbol.upper(), venue=venue)
                if inst is None:
                    return False
                logger.info("[ENFORCE EXECUTING] Syncing open orders/state for %s on %s: %s",
                            inst.symbol, inst.provider_name, reason)
                if getattr(inst, "_api", None) is not None and hasattr(inst._api, "open_orders"):
                    inst._api.open_orders(inst.symbol, provider_name=inst.provider_name)
                elif getattr(inst, "_provider", None) is not None and hasattr(inst._provider, "open_orders"):
                    inst._provider.open_orders(inst.symbol)
                return True
            except Exception as e:
                logger.error("[ENFORCE] Failed SYNC_POSITION for %s on %s: %s", symbol, venue, e)
                return False

        logger.warning("[ENFORCE] Unsupported venue/fleet action %r: %s", action_type, act)
        return False

    def apply_reconciliation(
        self, plan: Dict[str, Any], *, allow_code_remediation: bool = True
    ) -> Tuple[bool, Optional[bool]]:
        """Applies or simulates venue/fleet repair actions and code remediation based on operational mode."""
        if not isinstance(plan, dict):
            return False, None
        allow_remediation = _as_bool_flag(allow_code_remediation, True)
        is_enforce = (self.mode == "enforce")
        raw_actions = plan.get("venue_actions")
        venue_actions = raw_actions if isinstance(raw_actions, list) else []
        raw_fix = plan.get("code_remediation") if allow_remediation else None
        code_fix = raw_fix if isinstance(raw_fix, dict) else {}

        venue_executed = False
        for act in venue_actions:
            if not isinstance(act, dict):
                continue
            norm = normalize_repair_action(act, self.workspace_dir)
            action_type = norm["action"]
            if action_type == "NONE":
                continue
            if not is_enforce or not allow_remediation:
                logger.info("[SHADOW WOULD EXECUTE] Venue/fleet action %s for %s (%s): %s",
                            action_type, norm["symbol"] or act.get("symbol"), norm["venue"], norm["reason"])
            else:
                if self._execute_venue_or_fleet_action(act):
                    venue_executed = True

        tests_passed: Optional[bool] = None
        has_fix = _as_bool_flag(code_fix.get("has_fix"), False)
        raw_target_file = code_fix.get("target_file")
        raw_search_block = code_fix.get("search_block")
        if (
            has_fix
            and isinstance(raw_target_file, str)
            and raw_target_file.strip()
            and isinstance(raw_search_block, str)
            and raw_search_block
        ):
            target_file = os.path.join(self.workspace_dir, raw_target_file.strip())
            raw_rationale = code_fix.get("rationale")
            rationale = (
                str(raw_rationale).strip()
                if raw_rationale is not None and not isinstance(raw_rationale, bool) and str(raw_rationale).strip()
                else "Autonomous self-heal patch"
            )
            raw_replace = code_fix.get("replace_block", "")
            replace_block = raw_replace if isinstance(raw_replace, str) else ""
            if not is_enforce:
                logger.info("[SHADOW WOULD APPLY] Code patch on %s: %s", target_file, rationale)
            else:
                logger.info("[ENFORCE ATTEMPTING PATCH] Applying self-healing patch on %s: %s", target_file, rationale)
                tests_passed = self._apply_code_patch_safely(
                    target_file=target_file,
                    search_block=raw_search_block,
                    replace_block=replace_block,
                    rationale=rationale,
                )

        executed = is_enforce and bool(venue_executed or (has_fix and tests_passed))
        return executed, tests_passed

    def _apply_code_patch_safely(
        self,
        target_file: str,
        search_block: str,
        replace_block: str,
        rationale: str,
    ) -> bool:
        """Applies code or configuration patch, executes pytest test suite, commits on 100% green or rolls back immediately."""
        real_target, err_msg = resolve_patchable_target(target_file, self.workspace_dir)
        if real_target is None:
            logger.warning("%s", err_msg)
            return False

        if not search_block or search_block == replace_block:
            logger.warning("Refusing empty or no-op autonomous patch on %s.", target_file)
            return False

        content: Optional[str] = None
        try:
            with open(real_target, "r", encoding="utf-8") as f:
                content = f.read()

            if search_block not in content:
                logger.warning("Search block not found in %s. Skipping patch.", target_file)
                return False

            from state_io import atomic_write_text

            new_content = content.replace(search_block, replace_block, 1)
            atomic_write_text(real_target, new_content)

            pytest_bin = "pytest"
            for cand in (".venv/bin/pytest", "myenv/bin/pytest"):
                cand_abs = os.path.join(self.workspace_dir, cand)
                if os.path.exists(cand_abs):
                    pytest_bin = cand_abs
                    break
            test_cmd = [pytest_bin, "tests/"]
            res = subprocess.run(test_cmd, cwd=self.workspace_dir, capture_output=True, text=True, timeout=240)

            if res.returncode == 0:
                logger.info("Self-healing patch verified with 100%% green tests. Committing to git.")
                subprocess.run(["git", "add", real_target], cwd=self.workspace_dir, check=False)
                commit_msg = f"[AI-SUPERVISOR] Autonomous self-heal: {rationale}"
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=self.workspace_dir, check=False)
                subprocess.run(["git", "push", "origin", "main"], cwd=self.workspace_dir, check=False)
                return True
            else:
                logger.warning("Self-healing patch broke tests (exit code %d). Performing immediate rollback.", res.returncode)
                atomic_write_text(real_target, content)
                subprocess.run(["git", "checkout", "--", real_target], cwd=self.workspace_dir, check=False)
                return False
        except Exception as e:
            logger.error("Error during safe code patching: %s. Performing rollback.", e)
            if content is not None:
                try:
                    from state_io import atomic_write_text
                    atomic_write_text(real_target, content)
                except OSError:
                    pass
            subprocess.run(["git", "checkout", "--", real_target], cwd=self.workspace_dir, check=False)
            return False

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
        """Sends a clean, compact push alert via standard HTTP POST to ntfy.

        Routing policy:
        - Trading bot errors (Python exceptions/code patches, order anomalies, hung fleet bots) -> NTFY_TOPIC_ERROR (ntfy-error).
        - Linux OS / server / VPN / memory / disk issues and routine server health audits -> NTFY_TOPIC_SERVER (ntfy-server).
        """
        if not self.notify_enabled:
            return False

        issues_cnt = sum(
            1 for i in (report.diagnosed_issues or [])
            if isinstance(i, dict) and str(i.get("category") or "").strip().upper() != "HEALTHY"
        )
        actions_cnt = sum(
            1 for a in (report.venue_actions or [])
            if isinstance(a, dict) and str(a.get("action") or "NONE").strip().upper() != "NONE"
        )
        has_fix = bool(
            isinstance(report.code_remediation, dict)
            and _as_bool_flag(report.code_remediation.get("has_fix", False), False)
        )
        if has_fix:
            actions_cnt += 1
        mode_label = report.mode.upper()
        is_bot_error = self._is_bot_or_code_error_report(report)
        is_sys_error = self._is_system_os_error_report(report)
        is_any_error = is_bot_error or is_sys_error

        spec = (
            topic_spec(topic_key)
            if (topic_key in TOPIC_SPECS or topic_key in TOPIC_ALIASES)
            else topic_spec("auto_technical_repair")
        )
        if is_bot_error:
            title = f"🚨 {spec.bracket_tag} ERROR Issues: {issues_cnt} | Actions: {actions_cnt}"
            target_topic = self.ntfy_topic_error or self.ntfy_topic or "ntfy-error-1978"
            priority = "high"
        elif is_sys_error:
            title = f"🚨 {spec.bracket_tag} SYSTEM Issues: {issues_cnt} | Actions: {actions_cnt}"
            target_topic = self.ntfy_topic_server or self.ntfy_topic or "ntfy-server-1978"
            priority = "high"
        else:
            title = f"{spec.ntfy_prefix} Issues: {issues_cnt} | Actions: {actions_cnt}"
            target_topic = self.ntfy_topic_server or self.ntfy_topic or "ntfy-server-1978"
            priority = "default"

        body_lines = [
            f"Mode: {mode_label} · Status: {'Executed' if report.actions_executed else 'Simulated/Observed'}",
            f"{report.summary}",
        ]
        if report.diagnosed_issues:
            top_issue = report.diagnosed_issues[0]
            if isinstance(top_issue, dict):
                body_lines.append(f"Primary Issue: [{top_issue.get('severity', 'INFO')}] {top_issue.get('description', '')}")

        if has_fix:
            patch_file = report.code_remediation.get("target_file", "unknown")
            body_lines.append(f"Code Patch Proposed: {patch_file} (tests={report.tests_passed})")

        decisions_to_show = report.market_trading_decisions or (
            [report.market_trading_decision] if report.market_trading_decision else []
        )
        for m_dec in decisions_to_show:
            if not isinstance(m_dec, dict):
                continue
            m_decision = str(m_dec.get("decision", "HOLD")).upper()
            if m_decision in ("BUY", "SELL"):
                m_sym = m_dec.get("symbol", "")
                m_venue = m_dec.get("venue", "")
                m_conf = safe_nonneg_float(m_dec.get("confidence"), default=0.0) or 0.0
                m_usd = safe_nonneg_float(m_dec.get("suggested_notional_usd"), default=0.0) or 0.0
                venue_sfx = f" ({m_venue})" if m_venue else ""
                body_lines.append(f"Trading Intent: {m_decision} {m_sym}{venue_sfx} ${m_usd:.0f} (conf={m_conf:.2f})")

        body = "\n".join(body_lines)

        try:
            from notify_engine.alertnotifiers import pass_common_gate
            allowed, reason, _, out_payload = pass_common_gate(
                "ntfy",
                title,
                body,
                priority,
                target_topic,
                [{"type": "bot_event", "name": title, "body": body, "source": "ai_supervisor", "symbol": "SYSTEM"}],
                urgent=is_any_error,
            )
            if not allowed:
                logger.info("Supervisor ntfy gated by common policy (%s): %s", reason, title)
                return reason == "buffered_for_bulk"
            if out_payload:
                title, body, priority, target_topic = out_payload
        except Exception as e:
            logger.debug("Common gate fallback in reconciler: %s", e)

        url = self.ntfy_url or f"https://ntfy.sh/{target_topic}"
        headers = {
            "Title": title.encode("utf-8").decode("latin-1", "ignore"),
            "Priority": priority,
            "Tags": "rotating_light,robot" if is_any_error else "shield,robot",
        }
        if self.ntfy_token:
            headers["Authorization"] = f"Bearer {self.ntfy_token}"

        try:
            req = urllib.request.Request(
                url,
                data=body.encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = getattr(resp, "status", None) or getattr(resp, "code", None)
                if status is None and hasattr(resp, "getcode") and callable(resp.getcode):
                    status = resp.getcode()
                return status == 200
        except Exception as exc:
            logger.error("Failed sending standalone ntfy notification: %s", exc)
            return False

    def write_audit_artifact(self, report: SupervisorAuditReport) -> None:
        """Persists the full audit report and heartbeat to disk in cachedb/."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            from state_io import save_json_state

            report_dict = clean_json_value(asdict(report))
            save_json_state(self.state_file, report_dict, indent=2)

            hb = clean_json_value({
                "status": "alive",
                "ts": report.ts,
                "mode": report.mode,
                "summary": report.summary,
                "diagnosed_issues_count": len(report.diagnosed_issues),
                "venue_actions_count": len(report.venue_actions),
                "market_trading_decision": report.market_trading_decision,
                "market_trading_decisions": report.market_trading_decisions,
            })
            save_json_state(self.heartbeat_file, hb, indent=2)
        except Exception as e:
            logger.warning("Could not persist audit artifact: %s", e)

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
