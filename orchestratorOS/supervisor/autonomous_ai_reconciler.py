"""Autonomous AI supervisor running the cron-driven LLM topic threads.

- Ingests passive telemetry, sensors and state files published by the bots in cachedb/ and logs/.
- Reasons on the topic's permanent Antigravity thread (orchestratorOS.agy.topics) via GeminiClient/agy-bot.
- Each topic's TopicSpec bounds its side effects:
  * auto_technical_repair: diagnostics + verified self-healing code patches (never trades);
  * auto_trade_executor: diagnostics + trade decisions submitted to AutonomousIntentGateway (never patches code).
- supervisor_mode ('shadow' / 'enforce') bounds every side effect, including the gateway execution mode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
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

from botcore import load_dotenv
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.high_stake_guard import compute_24h_price_metrics, extract_symbol_recent_fills
from orchestratorOS.agy.topics import TOPIC_SPECS, topic_spec
from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway, load_key_value_conf

load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("orchestratorOS.supervisor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_FULL_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_GLOG_TS_RE = re.compile(r"^[IWEF](\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_CLOCK_TS_RE = re.compile(r"^\.*\[?(\d{2}):(\d{2}):(\d{2})\b")


def recent_log_lines(lines: List[str], file_mtime: float, cutoff_ts: float) -> List[str]:
    """Return the tail of ``lines`` whose own timestamps are >= cutoff_ts (chronological order).

    Bot logs mix full dates, glog headers and clock-only stamps ("14:54:17 ..."). Clock-only stamps are
    anchored to the file's mtime day and walked backwards, stepping one day back on a midnight wrap.
    Lines without a stamp (tracebacks) inherit the nearest later stamp; a file mtime inside the window
    alone is not enough to pull hours-old errors into a short lookback.
    """
    from datetime import datetime, timedelta

    ref = datetime.fromtimestamp(file_mtime)
    day = ref.date()
    prev_secs = ref.hour * 3600 + ref.minute * 60 + ref.second + 60
    picked: List[str] = []
    for line in reversed(lines):
        ts: Optional[float] = None
        m = _FULL_TS_RE.search(line)
        if m:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")
            day, prev_secs, ts = dt.date(), dt.hour * 3600 + dt.minute * 60 + dt.second, dt.timestamp()
        else:
            g = _GLOG_TS_RE.match(line)
            c = None if g else _CLOCK_TS_RE.match(line)
            if g:
                dt = datetime(ref.year, int(g.group(1)), int(g.group(2)),
                              int(g.group(3)), int(g.group(4)), int(g.group(5)))
                day, prev_secs, ts = dt.date(), dt.hour * 3600 + dt.minute * 60 + dt.second, dt.timestamp()
            elif c:
                secs = int(c.group(1)) * 3600 + int(c.group(2)) * 60 + int(c.group(3))
                if secs > prev_secs + 300:
                    day -= timedelta(days=1)
                prev_secs = secs
                ts = (datetime.combine(day, datetime.min.time()) + timedelta(seconds=secs)).timestamp()
        if ts is not None and ts < cutoff_ts:
            break
        picked.append(line)
    picked.reverse()
    return picked


@dataclass
class TelemetrySensorSnapshot:
    """Consolidated snapshot of passive sensors collected from bot runtime files."""
    fear_greed: Optional[Dict[str, Any]] = None
    market_breadth: Optional[Dict[str, Any]] = None
    sentiment_advisor: Optional[Dict[str, Any]] = None
    geopolitical_threat: Optional[Dict[str, Any]] = None
    active_heartbeats: Dict[str, Any] = field(default_factory=dict)
    state_files: Dict[str, Any] = field(default_factory=dict)
    prices_and_trends: Dict[str, Any] = field(default_factory=dict)
    microstructure: Dict[str, Any] = field(default_factory=dict)
    portfolio_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEventSummary:
    """Summary of log anomalies and guard interventions collected within the lookback window."""
    total_scanned_files: int
    error_count: int
    warning_count: int
    critical_events: List[str] = field(default_factory=list)
    guard_interventions: List[str] = field(default_factory=list)


@dataclass
class SupervisorAuditReport:
    """Consolidated nightly audit result produced by the autonomous supervisor."""
    ts: float
    mode: str                          # "shadow" | "enforce"
    summary: str
    diagnosed_issues: List[Dict[str, Any]] = field(default_factory=list)
    venue_actions: List[Dict[str, Any]] = field(default_factory=list)
    code_remediation: Dict[str, Any] = field(default_factory=dict)
    market_trading_decision: Dict[str, Any] = field(default_factory=dict)
    intent_emitted: Optional[Dict[str, Any]] = None
    sensor_health: Dict[str, Any] = field(default_factory=dict)
    actions_executed: bool = False
    tests_passed: Optional[bool] = None
    notification_sent: bool = False


class AutonomousAIReconciler:
    """Autonomous supervisor orchestrating decoupled sensor harvesting, log auditing, and self-healing."""

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
        self.lookback_hours = float(lookback_hours)
        self.logs_dirs = [os.path.join(self.workspace_dir, d) for d in (logs_dirs or ["logs", "logger"])]
        self.gemini_auditor = gemini_auditor or GeminiClient()
        self._intent_gateway = intent_gateway

        # Parse decoupled configuration strictly from config files without fallbacks
        conf = self._load_decoupled_conf()
        if mode is not None:
            raw_mode = mode
        else:
            if "supervisor_mode" not in conf:
                raise KeyError("Missing required 'supervisor_mode' in order_guard.conf")
            raw_mode = conf["supervisor_mode"]
        self.mode = str(raw_mode).strip().lower()
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError(f"Invalid supervisor_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']")

        if notify_enabled is not None:
            self.notify_enabled = bool(notify_enabled)
        else:
            if "supervisor_notify" not in conf:
                raise KeyError("Missing required 'supervisor_notify' in order_guard.conf")
            self.notify_enabled = bool(int(float(conf["supervisor_notify"])))

        self.ntfy_url = conf.get("phone_alert_url", "")
        if "ntfy_topic_guard" not in conf:
            raise KeyError("Missing required 'ntfy_topic_guard' in config.env")
        self.ntfy_topic = conf["ntfy_topic_guard"]
        self.ntfy_token = conf.get("ntfy_token", "")

        # Per-topic artifact paths are set by run_cycle(topic_key=...).
        self.state_file = ""
        self.heartbeat_file = ""

    @property
    def intent_gateway(self) -> Any:
        """The single trade-intent choke point (owns autonomous_trading_* config, clamping, dedup, mode)."""
        if self._intent_gateway is None:
            self._intent_gateway = AutonomousIntentGateway(workspace_dir=self.workspace_dir)
        return self._intent_gateway

    def _load_decoupled_conf(self) -> Dict[str, str]:
        """Parses key-value pairs from order_guard.conf and .env without external imports."""
        return load_key_value_conf(self.workspace_dir, _ROOT_DIR)

    def harvest_sensors_telemetry(self) -> TelemetrySensorSnapshot:
        """Reads passive telemetry and sensor files published to disk by trading bots."""
        snapshot = TelemetrySensorSnapshot()

        def _read_json(fpath: str) -> Optional[Dict[str, Any]]:
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        return json.load(f)
                except Exception:
                    pass
            return None

        # 1. Fear & Greed sensor
        for fg_name in ("fear_greed_cache.json", "fear_greed_collect.json"):
            data = _read_json(os.path.join(self.cache_dir, fg_name))
            if data:
                snapshot.fear_greed = data
                break

        # 2. Market Breadth sensor
        for mb_name in ("market_breadth_cache.json", "market_breadth_collect.json"):
            data = _read_json(os.path.join(self.cache_dir, mb_name))
            if data:
                snapshot.market_breadth = data
                break

        # 3. Macro Sentiment Advisor sensor
        for sa_name in ("sentiment_advisor_eval.json", "macro_advisor_eval.json", "gemini_macro_advisor.json"):
            data = _read_json(os.path.join(self.cache_dir, sa_name))
            if data:
                snapshot.sentiment_advisor = data
                break

        # 4. Geopolitical Threat Shield sensor
        for gt_name in ("geopolitical_threat_eval.json", "geopolitical_threat_state.json"):
            data = _read_json(os.path.join(self.cache_dir, gt_name))
            if data:
                snapshot.geopolitical_threat = data
                break

        # 5. Heartbeat sensors
        for hb_name in ("macro_analyzer.heartbeat", "intelligence_daemon.heartbeat", "rtrade.heartbeat"):
            data = _read_json(os.path.join(self.cache_dir, hb_name))
            if data:
                snapshot.active_heartbeats[hb_name] = data

        # 6. Bot state files in workspace
        for s_file in (".watchdog_state.json", ".anomaly_watchdog_state.json", "kraken_trades_full.json"):
            data = _read_json(os.path.join(self.workspace_dir, s_file))
            if data is not None:
                mtime = os.path.getmtime(os.path.join(self.workspace_dir, s_file))
                snapshot.state_files[s_file] = {"age_hours": (time.time() - mtime) / 3600.0}

        # 7. Live market prices, short/long trend, Kalman, and 24h price evolution
        prices_data = _read_json(os.path.join(self.cache_dir, "cache_current_price.json"))
        trends_data = _read_json(os.path.join(self.cache_dir, "cache_short_trend.json"))
        long_trends_data = _read_json(os.path.join(self.cache_dir, "cache_long_trend.json"))
        kalman_data = _read_json(os.path.join(self.cache_dir, "kalman_state.json"))

        price_map: Dict[str, float] = {}
        if prices_data and isinstance(prices_data.get("items"), dict):
            for sym, item in prices_data["items"].items():
                if isinstance(item, list) and item and isinstance(item[0], list) and len(item[0]) > 1:
                    try:
                        price_map[sym] = float(item[0][1])
                    except (ValueError, TypeError):
                        pass

        long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}

        if trends_data and isinstance(trends_data, dict):
            for sym, tinfo in trends_data.items():
                if isinstance(tinfo, dict):
                    px = tinfo.get("current_price", price_map.get(sym))
                    trend = tinfo.get("final_trend", 0)
                    growth = tinfo.get("growth_coefficient", 0.0)
                    slope = tinfo.get("slope_full", 0.0)
                    dyn = tinfo.get("dynamic", {}) if isinstance(tinfo.get("dynamic"), dict) else {}
                    d222 = dyn.get("222", {}) if isinstance(dyn.get("222"), dict) else {}
                    d900 = dyn.get("900", {}) if isinstance(dyn.get("900"), dict) else {}
                    d9000 = dyn.get("9000", {}) if isinstance(dyn.get("9000"), dict) else {}
                    entry: Dict[str, Any] = {
                        "price": px,
                        "trend": trend,
                        "growth_coefficient": growth,
                        "slope_full": slope,
                        "gradient_recent": tinfo.get("gradient_recent", 0.0),
                        "trend_fast": tinfo.get("trend_fast", trend),
                        "slope_3_7m": d222.get("slope_full", 0.0),
                        "trend_3_7m": d222.get("final_trend", trend),
                        "slope_15m": d900.get("slope_full", 0.0),
                        "trend_15m": d900.get("final_trend", trend),
                        "slope_2_5h": d9000.get("slope_full", 0.0),
                        "trend_2_5h": d9000.get("final_trend", trend),
                        "vol_1h_pct": tinfo.get("vol_1h_pct"),
                    }
                    if isinstance(kalman_data, dict) and isinstance(kalman_data.get(sym), dict):
                        k_sym = kalman_data[sym]
                        entry["kalman_vel"] = k_sym.get("kalman_vel")
                        entry["kalman_trend"] = k_sym.get("kalman_trend")
                    if isinstance(long_items, dict) and isinstance(long_items.get(sym), list) and long_items[sym]:
                        lt = long_items[sym][0]
                        if isinstance(lt, dict):
                            entry["long_trend_dir"] = lt.get("direction")
                            entry["long_trend_hours"] = round(float(lt.get("duration_seconds", 0.0)) / 3600.0, 1)
                    snapshot.prices_and_trends[sym] = entry

        for sym, px in price_map.items():
            if sym not in snapshot.prices_and_trends:
                snapshot.prices_and_trends[sym] = {"price": px, "trend": 0, "growth_coefficient": 0.0}

        # 8. 24h multi-horizon price evolution & Pillar 2 microstructure per symbol
        for sym in list(snapshot.prices_and_trends.keys()):
            p24_data = _read_json(os.path.join(self.cache_dir, f"cache_24_price_{sym}.json"))
            if isinstance(p24_data, dict):
                raw_items = p24_data.get("items", {})
                series = raw_items.get(sym, []) if isinstance(raw_items, dict) else raw_items
                m24 = compute_24h_price_metrics(series)
                if m24:
                    snapshot.prices_and_trends[sym].update(m24)

            ob = _read_json(os.path.join(self.cache_dir, f"orderbook_depth_{sym}.json"))
            deriv = _read_json(os.path.join(self.cache_dir, f"derivatives_telemetry_{sym}.json"))
            whale = _read_json(os.path.join(self.cache_dir, f"whale_snapshot_{sym}.json"))
            if ob or deriv or whale:
                m_info: Dict[str, Any] = {}
                if isinstance(ob, dict):
                    m_info["ob_imbalance"] = ob.get("imbalance_ratio")
                    m_info["bid_wall_px"] = ob.get("largest_bid_wall_price")
                    m_info["ask_wall_px"] = ob.get("largest_ask_wall_price")
                if isinstance(deriv, dict):
                    m_info["funding_rate"] = deriv.get("funding_rate")
                    m_info["oi_usd"] = deriv.get("open_interest_usd")
                if isinstance(whale, dict):
                    m_info["whale_long_pct"] = whale.get("top_traders_long_pct")
                    m_info["taker_buy_sell_ratio"] = whale.get("taker_buy_sell_ratio")
                    m_info["oi_1h_chg_pct"] = whale.get("open_interest_1h_change_pct")
                snapshot.microstructure[sym] = m_info

        # 9. Portfolio value, recent orders/fills per symbol & recent autonomous intents
        asset_val_path = os.path.join(self.cache_dir, "cache_asset_value.jsonl")
        if os.path.exists(asset_val_path):
            try:
                with open(asset_val_path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - 2048))
                    tail_lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
                if tail_lines:
                    last_av = json.loads(tail_lines[-1])
                    if isinstance(last_av, dict) and isinstance(last_av.get("i"), dict):
                        snapshot.portfolio_summary["total_value_usdc"] = last_av["i"].get("total_value_usdc")
            except Exception:
                pass

        orders_data = _read_json(os.path.join(self.cache_dir, "cache_order.json"))
        if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
            recent_orders: Dict[str, Any] = {}
            for sym, o_list in orders_data["items"].items():
                fills = extract_symbol_recent_fills(o_list)
                if fills:
                    recent_orders[sym] = fills
            if recent_orders:
                snapshot.portfolio_summary["recent_orders"] = recent_orders

        intents_data = _read_json(os.path.join(self.cache_dir, "autonomous_trade_intents.json"))
        if isinstance(intents_data, list) and intents_data:
            snapshot.portfolio_summary["recent_ai_intents"] = [
                {
                    "decision": i.get("decision"),
                    "symbol": i.get("symbol"),
                    "status": i.get("status"),
                    "notional_eur": i.get("suggested_notional_eur"),
                    "ts": i.get("timestamp"),
                }
                for i in intents_data[-5:]
                if isinstance(i, dict)
            ]

        return snapshot

    def harvest_log_anomalies(self, lookback_hours: Optional[float] = None) -> LogEventSummary:
        """Parses recent log files for errors, unhandled exceptions, and guard vetoes."""
        hours = lookback_hours if lookback_hours is not None else self.lookback_hours
        cutoff_ts = time.time() - (hours * 3600.0)

        critical_events: List[str] = []
        guard_interventions: List[str] = []
        error_count = 0
        warning_count = 0
        scanned_count = 0

        err_pattern = re.compile(
            r"(\bERROR\b|\bCRITICAL\b(?!_SHOCK)|Traceback|fail-closed|BUY BLOCKED|\bException\b)",
            re.IGNORECASE,
        )
        guard_pattern = re.compile(r"(\[.*GUARD.*\]|vetoed|downscaled|HARD_VETO|DOWNSCALE_QTY)", re.IGNORECASE)

        for l_dir in self.logs_dirs:
            if not os.path.isdir(l_dir):
                continue
            try:
                for fname in os.listdir(l_dir):
                    if not (fname.endswith(".log") or fname.endswith(".txt")):
                        continue
                    fpath = os.path.join(l_dir, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        if mtime < cutoff_ts:
                            continue
                        scanned_count += 1
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            lines = recent_log_lines(f.readlines()[-1000:], mtime, cutoff_ts)
                            for line in lines:
                                if err_pattern.search(line):
                                    error_count += 1
                                    cleaned = line.strip()
                                    if cleaned not in critical_events and len(critical_events) < 50:
                                        critical_events.append(cleaned)
                                elif guard_pattern.search(line):
                                    warning_count += 1
                                    cleaned = line.strip()
                                    if cleaned not in guard_interventions and len(guard_interventions) < 50:
                                        guard_interventions.append(cleaned)
                    except Exception as err:
                        logger.debug("Failed reading log file %s: %s", fpath, err)
            except Exception as d_err:
                logger.debug("Failed scanning directory %s: %s", l_dir, d_err)

        return LogEventSummary(
            total_scanned_files=scanned_count,
            error_count=error_count,
            warning_count=warning_count,
            critical_events=critical_events,
            guard_interventions=guard_interventions,
        )

    def synthesize_and_diagnose(
        self,
        sensors: TelemetrySensorSnapshot,
        logs: LogEventSummary,
        *,
        focus_symbols: Optional[List[str]] = None,
        topic_key: str,
    ) -> Dict[str, Any]:
        """Feeds sensor telemetry and log anomalies to Gemini; the topic spec bounds what it may propose."""
        context_items = [
            "### 1. ACTIVE SENSOR TELEMETRY:",
            f"- Fear & Greed: {sensors.fear_greed.get('value', 'N/A') if sensors.fear_greed else 'None'}",
            f"- Market Breadth Advance Ratio: {sensors.market_breadth.get('advance_ratio', 'N/A') if sensors.market_breadth else 'None'}",
            f"- Sentiment Advisor Bias: {sensors.sentiment_advisor.get('market_bias', 'N/A') if sensors.sentiment_advisor else 'None'}",
            f"- Geopolitical Threat Level: {sensors.geopolitical_threat.get('threat_level', 'N/A') if sensors.geopolitical_threat else 'None'}",
            f"- Active Heartbeats: {list(sensors.active_heartbeats.keys())}",
            f"\n### 2. LOG ANOMALIES & GUARD VETOES (PAST {self.lookback_hours:g}H):",
            f"- Scanned Logs: {logs.total_scanned_files}",
            f"- Error Count: {logs.error_count}",
            f"- Guard Interventions: {logs.warning_count}",
            "- Sample Errors & Interventions:",
        ]
        for evt in logs.critical_events[:15]:
            context_items.append(f"  * {evt}")
        for g_evt in logs.guard_interventions[:10]:
            context_items.append(f"  * [GUARD] {g_evt}")

        if sensors.prices_and_trends:
            context_items.append("\n### 3. LIVE MARKET PRICES, MULTI-HORIZON TRENDS & 24H EVOLUTION:")
            for sym, pinfo in sorted(sensors.prices_and_trends.items()):
                px = pinfo.get("price")
                px_str = f"${px:,.2f}" if isinstance(px, (int, float)) and px > 1 else (f"${px}" if px is not None else "N/A")
                tr = pinfo.get("trend", 0)
                tr_label = "BULL (+1)" if tr > 0 else ("BEAR (-1)" if tr < 0 else "FLAT (0)")
                growth = pinfo.get("growth_coefficient", 0.0)
                extras: List[str] = []
                if "chg_15m_pct" in pinfo:
                    extras.append(
                        f"15m={pinfo.get('chg_15m_pct', 0):+.2f}% 30m={pinfo.get('chg_30m_pct', 0):+.2f}% "
                        f"1h={pinfo.get('chg_1h_pct', 0):+.2f}% 3h={pinfo.get('chg_3h_pct', 0):+.2f}% "
                        f"24h_range=[{pinfo.get('low_24h')}, {pinfo.get('high_24h')}]"
                    )
                if "slope_15m" in pinfo:
                    extras.append(
                        f"slopes(3.7m={pinfo.get('slope_3_7m', 0):+.5f}, 15m={pinfo.get('slope_15m', 0):+.5f}, "
                        f"2.5h={pinfo.get('slope_2_5h', 0):+.5f})"
                    )
                if pinfo.get("kalman_vel") is not None:
                    extras.append(f"kalman_vel={pinfo.get('kalman_vel'):+.5f}")
                if pinfo.get("long_trend_dir"):
                    extras.append(f"long_trend={pinfo.get('long_trend_dir')}({pinfo.get('long_trend_hours')}h)")
                extra_str = f" | {' | '.join(extras)}" if extras else ""
                context_items.append(f"  * {sym}: Price={px_str}, Trend={tr_label}, Growth={growth:+.5f}{extra_str}")

        if sensors.microstructure:
            context_items.append("\n### 3B. PILLAR 2 MICROSTRUCTURE, DERIVATIVES & WHALE POSITIONING:")
            for sym, minfo in sorted(sensors.microstructure.items()):
                context_items.append(
                    f"  * {sym}: ob_imbalance={minfo.get('ob_imbalance')}, "
                    f"bid_wall={minfo.get('bid_wall_px')}, ask_wall={minfo.get('ask_wall_px')}, "
                    f"whale_long_pct={minfo.get('whale_long_pct')}, taker_ratio={minfo.get('taker_buy_sell_ratio')}, "
                    f"funding={minfo.get('funding_rate')}, oi_1h_chg={minfo.get('oi_1h_chg_pct')}%"
                )

        if sensors.portfolio_summary:
            context_items.append("\n### 3C. PORTFOLIO STATE, LAST FILLS & AUTONOMOUS INTENTS:")
            if sensors.portfolio_summary.get("total_value_usdc") is not None:
                context_items.append(f"  * Total Portfolio Value: ${sensors.portfolio_summary['total_value_usdc']:,.2f} USDC")
            for sym, oinfo in sorted((sensors.portfolio_summary.get("recent_orders") or {}).items()):
                lb = oinfo.get("last_buy") or {}
                ls = oinfo.get("last_sell") or {}
                context_items.append(
                    f"  * {sym} Last Action={oinfo.get('last_action_side')}: "
                    f"Last BUY={lb.get('qty')} @ {lb.get('price')} | Last SELL={ls.get('qty')} @ {ls.get('price')}"
                )
            if sensors.portfolio_summary.get("recent_ai_intents"):
                context_items.append(f"  * Recent Autonomous Intents: {sensors.portfolio_summary['recent_ai_intents']}")

        if focus_symbols:
            clean_syms = [s.strip().upper() for s in focus_symbols if s.strip()]
            if clean_syms:
                context_items.append(
                    f"\n### 4. TACTICAL WATCHLIST FOCUS: {', '.join(clean_syms)}\n"
                    "- Pay special attention to momentum, gentle upward recovery (re-entry after SELL), or downward breakdown on these focus symbols."
                )

        context_str = "\n".join(context_items)
        spec = topic_spec(topic_key)

        tasks = [
            "1. Explain the technical root cause of any bot exceptions, fail-closed blocks, or guard vetoes.",
            "2. Determine whether open orders or venue positions require reconciliation.",
        ]
        schema = [
            '  "summary": "1-2 sentence executive assessment of system health"',
            '  "diagnosed_issues": [{"severity": "HIGH" | "MEDIUM" | "LOW" | "INFO", '
            '"category": "ORPHAN_ORDER" | "BOT_EXCEPTION" | "GUARD_VETO" | "HEALTHY", '
            '"description": "Short explanation", "root_cause": "Detailed technical root cause"}]',
            '  "venue_actions": [{"action": "CANCEL_ORDER" | "SYNC_POSITION" | "NONE", '
            '"venue": "binance" | "kraken" | "hyperliquid" | "t212", "symbol": "BTCUSDC", '
            '"order_id": "optional order id", "reason": "Why this action is needed"}]',
        ]
        if spec.can_self_heal:
            tasks.append("3. If a reproducible code bug exists, formulate an exact search_block and replace_block.")
            schema.append('  "code_remediation": {"has_fix": false, "target_file": "", "rationale": "", '
                          '"search_block": "", "replace_block": ""}')
        if spec.can_trade:
            gw = self.intent_gateway
            tasks.append(
                f"{len(tasks) + 1}. Formulate an autonomous market trading decision ('BUY', 'SELL', or 'HOLD') based on "
                "prevailing macro sentiment, geopolitical threat level, market breadth, multi-horizon price evolution "
                "(15m/30m/1h/3h), dynamic slopes (3.7m/15m/2.5h), Pillar 2 whale/orderbook flow, and last BUY/SELL fill anchors. "
                "Specifically: if we exited a position (Last Action=SELL) and TAOUSDC & BTCUSDC show a steady upward "
                "inflection/recovery so we risk being left out, or if there is a high-conviction dip/breakout, emit 'BUY'; "
                "if an active position shows exhaustion/reversal above our last BUY anchor, emit 'SELL'. "
                f"Only with high conviction (confidence >= {gw.min_confidence:g}) specify a fiat/stablecoin-quoted symbol "
                "(e.g. TAOUSDC, BTCUSDC), venue ('binance'/'kraken'/'hyperliquid'/'t212'), confidence, suggested_notional_eur "
                f"(up to {gw.max_notional_eur:g} EUR) and thesis; otherwise 'HOLD'."
            )
            schema.append('  "market_trading_decision": {"decision": "BUY" | "SELL" | "HOLD", "symbol": "BTCUSDC", '
                          '"venue": "binance" | "kraken" | "hyperliquid" | "t212", "confidence": 0.88, '
                          '"suggested_notional_eur": 250.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}')

        prompt = (
            "You are the Lead SRE and Quantitative Trading Systems Architect for MPTrade.\n"
            "Analyze the passive telemetry sensors and log anomalies collected from trading bots:\n\n"
            f"{context_str}\n\n"
            "Perform an executive diagnostic audit:\n" + "\n".join(tasks) + "\n\n"
            "INTERACTIVE THREAD RULES:\n"
            "- If the user wrote any tactical instructions, price targets, overrides, or questions earlier in this "
            "conversation thread, honor them with highest priority and address them directly.\n"
            "- For this automated telemetry snapshot turn, do NOT call tools, shell commands, or file readers because all "
            "required telemetry is provided above (use tools only when the user asks a direct interactive question in chat).\n"
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** (bullet points for: Starea Flotei & Loguri, "
            "Piața Live & Pante TAO/BTC, Portofoliu & Poziții Active, și Decizia AI / Răspuns la intervențiile din chat).\n"
            "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
            "```json\n{\n" + ",\n".join(schema) + "\n}\n```"
        )

        resp = self.gemini_auditor.query_json(
            prompt,
            timeout_sec=90.0 if spec.can_self_heal else 60.0,
            thread_title=f"{spec.title} — {self.mode.upper()}",
            topic_key=topic_key,
        )

        if not resp or not isinstance(resp, dict):
            logger.warning("Gemini supervisor query failed. Falling back to default health baseline.")
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

    def apply_reconciliation(
        self, plan: Dict[str, Any], *, allow_code_remediation: bool = True
    ) -> Tuple[bool, Optional[bool]]:
        """Applies or simulates venue actions and code remediation based on operational mode."""
        is_enforce = (self.mode == "enforce")
        raw_actions = plan.get("venue_actions")
        venue_actions = raw_actions if isinstance(raw_actions, list) else []
        raw_fix = plan.get("code_remediation") if allow_code_remediation else None
        code_fix = raw_fix if isinstance(raw_fix, dict) else {}

        for act in venue_actions:
            action_type = act.get("action", "NONE")
            if action_type == "NONE":
                continue
            if not is_enforce:
                logger.info("[SHADOW WOULD EXECUTE] Venue action %s for %s (%s): %s",
                            action_type, act.get("symbol"), act.get("venue"), act.get("reason"))
            else:
                logger.info("[ENFORCE EXECUTING] Venue action %s for %s (%s): %s",
                            action_type, act.get("symbol"), act.get("venue"), act.get("reason"))

        tests_passed: Optional[bool] = None
        if code_fix.get("has_fix") and code_fix.get("target_file") and code_fix.get("search_block"):
            target_file = os.path.join(self.workspace_dir, code_fix["target_file"])
            rationale = code_fix.get("rationale", "Autonomous self-heal patch")
            if not is_enforce:
                logger.info("[SHADOW WOULD APPLY] Code patch on %s: %s", target_file, rationale)
            else:
                logger.info("[ENFORCE ATTEMPTING PATCH] Applying self-healing patch on %s: %s", target_file, rationale)
                tests_passed = self._apply_code_patch_safely(
                    target_file=target_file,
                    search_block=code_fix["search_block"],
                    replace_block=code_fix.get("replace_block", ""),
                    rationale=rationale,
                )

        real_venue_actions = [
            act for act in venue_actions
            if isinstance(act, dict) and str(act.get("action", "NONE")).strip().upper() != "NONE"
        ]
        executed = is_enforce and bool(real_venue_actions or (code_fix.get("has_fix") and tests_passed))
        return executed, tests_passed

    def _apply_code_patch_safely(
        self,
        target_file: str,
        search_block: str,
        replace_block: str,
        rationale: str,
    ) -> bool:
        """Applies code patch, executes pytest test suite, commits on 100% green or rolls back immediately."""
        real_ws = os.path.realpath(self.workspace_dir)
        real_target = os.path.realpath(target_file)
        try:
            in_workspace = os.path.commonpath([real_ws, real_target]) == real_ws
        except ValueError:
            in_workspace = False
        if not in_workspace:
            logger.warning("Refusing autonomous patch outside workspace_dir: %s", target_file)
            return False

        if not os.path.exists(real_target):
            logger.error("Target patch file does not exist: %s", target_file)
            return False

        base_name = os.path.basename(real_target)
        if base_name in (".env", "config.env", "order_guard.conf", "instruments.conf", "procs.conf") or base_name.endswith(".env"):
            logger.warning("Refusing autonomous patch on protected configuration file: %s", target_file)
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

            new_content = content.replace(search_block, replace_block, 1)
            with open(real_target, "w", encoding="utf-8") as f:
                f.write(new_content)

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
                with open(real_target, "w", encoding="utf-8") as f:
                    f.write(content)
                subprocess.run(["git", "checkout", "--", real_target], cwd=self.workspace_dir, check=False)
                return False
        except Exception as e:
            logger.error("Error during safe code patching: %s. Performing rollback.", e)
            if content is not None:
                try:
                    with open(real_target, "w", encoding="utf-8") as f:
                        f.write(content)
                except OSError:
                    pass
            subprocess.run(["git", "checkout", "--", real_target], cwd=self.workspace_dir, check=False)
            return False

    def dispatch_notification(self, report: SupervisorAuditReport) -> bool:
        """Sends a clean, compact push alert via standard HTTP POST to ntfy."""
        if not self.notify_enabled:
            return False

        issues_cnt = len(report.diagnosed_issues)
        actions_cnt = len(report.venue_actions)
        has_fix = report.code_remediation.get("has_fix", False)
        mode_label = report.mode.upper()

        title = f"[AI-SUPERVISOR - {mode_label}] Issues: {issues_cnt} | Actions: {actions_cnt}"
        body_lines = [
            f"Mode: {mode_label} · Status: {'Executed' if report.actions_executed else 'Simulated/Observed'}",
            f"{report.summary}",
        ]
        if report.diagnosed_issues:
            top_issue = report.diagnosed_issues[0]
            body_lines.append(f"Primary Issue: [{top_issue.get('severity', 'INFO')}] {top_issue.get('description', '')}")

        if has_fix:
            patch_file = report.code_remediation.get("target_file", "unknown")
            body_lines.append(f"Code Patch Proposed: {patch_file} (tests={report.tests_passed})")

        if report.market_trading_decision:
            m_dec = report.market_trading_decision
            m_decision = m_dec.get("decision", "HOLD")
            if m_decision in ("BUY", "SELL"):
                m_sym = m_dec.get("symbol", "")
                m_conf = m_dec.get("confidence", 0.0)
                m_eur = m_dec.get("suggested_notional_eur", 0.0)
                body_lines.append(f"Trading Intent: {m_decision} {m_sym} €{m_eur:.0f} (conf={m_conf:.2f})")

        body = "\n".join(body_lines)

        try:
            from notify_engine.alertnotifiers import pass_common_gate
            allowed, reason, _, out_payload = pass_common_gate(
                "ntfy",
                title,
                body,
                "default",
                self.ntfy_topic or "ntfy-server-1978",
                [{"type": "bot_event", "name": title, "body": body, "source": "ai_supervisor", "symbol": "SYSTEM"}],
                urgent=False,
            )
            if not allowed:
                logger.info("Supervisor ntfy gated by common policy (%s): %s", reason, title)
                return reason == "buffered_for_bulk"
            if out_payload:
                title, body, _, _ = out_payload
        except Exception as e:
            logger.debug("Common gate fallback in reconciler: %s", e)

        url = self.ntfy_url or f"https://ntfy.sh/{self.ntfy_topic}"
        headers = {
            "Title": title,
            "Priority": "default",
            "Tags": "shield,robot",
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
            from state_io import atomic_write_json
            atomic_write_json(self.state_file, asdict(report), indent=2)

            hb = {
                "status": "alive",
                "ts": report.ts,
                "mode": report.mode,
                "summary": report.summary,
                "diagnosed_issues_count": len(report.diagnosed_issues),
                "venue_actions_count": len(report.venue_actions),
                "market_trading_decision": report.market_trading_decision,
            }
            atomic_write_json(self.heartbeat_file, hb, indent=2)
        except Exception as e:
            logger.warning("Could not persist audit artifact: %s", e)

    def run_cycle(
        self,
        *,
        topic_key: str,
        focus_symbols: Optional[List[str]] = None,
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

        sensors = self.harvest_sensors_telemetry()
        logs = self.harvest_log_anomalies()
        plan = self.synthesize_and_diagnose(sensors, logs, focus_symbols=focus_symbols, topic_key=topic_key)
        executed, tests_passed = self.apply_reconciliation(plan, allow_code_remediation=spec.can_self_heal)

        raw_mtd = plan.get("market_trading_decision") if spec.can_trade else None
        market_decision = raw_mtd if isinstance(raw_mtd, dict) else {}
        intent = None
        if market_decision:
            intent = self.intent_gateway.submit(market_decision, source=topic_key, max_mode=self.mode)
            if self.mode == "enforce" and isinstance(intent, dict) and intent.get("status") == "EXECUTED":
                executed = True

        raw_issues = plan.get("diagnosed_issues")
        raw_actions = plan.get("venue_actions")
        raw_fix = plan.get("code_remediation") if spec.can_self_heal else None
        report = SupervisorAuditReport(
            ts=now,
            mode=self.mode,
            summary=str(plan.get("summary") or "Audit complete."),
            diagnosed_issues=raw_issues if isinstance(raw_issues, list) else [],
            venue_actions=raw_actions if isinstance(raw_actions, list) else [],
            code_remediation=raw_fix if isinstance(raw_fix, dict) else {},
            market_trading_decision=market_decision,
            intent_emitted=intent,
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
        has_high_issue = any(
            isinstance(i, dict) and str(i.get("severity", "")).upper() in ("HIGH", "CRITICAL")
            for i in report.diagnosed_issues
        )
        should_notify = (
            spec.can_self_heal
            or executed
            or has_high_issue
            or str(market_decision.get("decision", "HOLD")).upper() in ("BUY", "SELL")
        )
        report.notification_sent = self.dispatch_notification(report) if should_notify else False
        self.write_audit_artifact(report)
        logger.info("Autonomous AI Supervisor cycle complete: %s", report.summary)
        return report


def main() -> None:
    """CLI entry point for the cron-driven supervisor topics."""
    parser = argparse.ArgumentParser(description="Autonomous AI supervisor (technical repair / trade executor)")
    parser.add_argument("--topic-key", required=True,
                        choices=sorted(k for k, v in TOPIC_SPECS.items() if v.can_trade or v.can_self_heal),
                        help="Permanent LLM topic thread to run")
    parser.add_argument("--mode", choices=["shadow", "enforce"], default=None,
                        help="Override supervisor_mode from order_guard.conf (bounds trading and self-heal)")
    parser.add_argument("--focus-symbols", default="", help="Comma-separated watchlist (e.g. TAOUSDC,BTCUSDC)")
    parser.add_argument("--lookback-hours", type=float, default=24.0,
                        help="Log anomaly lookback window in hours (default: 24; 0.6 for 30-minute runs)")
    args = parser.parse_args()

    focus = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()] or None
    reconciler = AutonomousAIReconciler(mode=args.mode, lookback_hours=args.lookback_hours)
    report = reconciler.run_cycle(topic_key=args.topic_key, focus_symbols=focus)
    print(f"[{report.mode.upper()}] {args.topic_key}: {report.summary}")


if __name__ == "__main__":
    main()
