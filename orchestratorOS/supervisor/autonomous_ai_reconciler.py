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
import math
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

from botcore import load_dotenv
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.high_stake_guard import compute_24h_price_metrics, extract_symbol_recent_fills
from orchestratorOS.agy.topics import TOPIC_SPECS, topic_spec
from orchestratorTrade.autonomous_intent_gateway import (
    AutonomousIntentGateway,
    _parse_finite_float,
    load_key_value_conf,
)

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


def _normalize_trading_decisions(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and deduplicate single or multi-coin trading decisions from Gemini's JSON plan."""
    decisions: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str, str]] = set()

    raw_multi = plan.get("market_trading_decisions")
    if isinstance(raw_multi, list):
        for item in raw_multi:
            if isinstance(item, dict):
                key = (
                    str(item.get("venue") or "").strip().lower(),
                    str(item.get("symbol") or "").strip().upper(),
                    str(item.get("decision") or "HOLD").strip().upper(),
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    decisions.append(item)

    raw_single = plan.get("market_trading_decision")
    if isinstance(raw_single, dict) and raw_single:
        key = (
            str(raw_single.get("venue") or "").strip().lower(),
            str(raw_single.get("symbol") or "").strip().upper(),
            str(raw_single.get("decision") or "HOLD").strip().upper(),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            decisions.append(raw_single)

    # Sort actionable BUY/SELL decisions ahead of HOLD, ordered by highest confidence first.
    def _sort_key(d: Dict[str, Any]) -> Tuple[int, float]:
        side = str(d.get("decision") or "HOLD").strip().upper()
        is_action = 1 if side in ("BUY", "SELL") else 0
        try:
            conf = float(d.get("confidence") or 0.0)
            if not math.isfinite(conf):
                conf = 0.0
        except (ValueError, TypeError):
            conf = 0.0
        return (-is_action, -conf)

    decisions.sort(key=_sort_key)
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
        if self._intent_gateway is None:
            self._intent_gateway = AutonomousIntentGateway(workspace_dir=self.workspace_dir)
        return self._intent_gateway

    def _load_decoupled_conf(self) -> Dict[str, str]:
        """Parses key-value pairs from order_guard.conf and .env without external imports."""
        return load_key_value_conf(self.workspace_dir, _ROOT_DIR)

    def harvest_sensors_telemetry(self, *, include_fleet: bool = True) -> TelemetrySensorSnapshot:
        """Reads passive telemetry, spot wallet balances, cycle states, and fills published to disk."""
        snapshot = TelemetrySensorSnapshot()
        now_ts = time.time()

        def _read_json(fpath: str) -> Optional[Any]:
            for cand in (fpath, f"{fpath}.previous"):
                if os.path.exists(cand):
                    try:
                        with open(cand, "r", encoding="utf-8", errors="replace") as f:
                            return json.load(f)
                    except Exception:
                        continue
            return None

        def _with_age_min(payload: Dict[str, Any], fpath: str) -> Dict[str, Any]:
            enriched = dict(payload)
            raw_ts = enriched.get("ts")
            ref_ts: Optional[float] = None
            if isinstance(raw_ts, (int, float)) and not isinstance(raw_ts, bool) and raw_ts > 0:
                ref_ts = float(raw_ts) / 1000.0 if float(raw_ts) > 1e12 else float(raw_ts)
            elif os.path.exists(fpath):
                ref_ts = os.path.getmtime(fpath)
            if ref_ts is not None:
                enriched.setdefault("age_min", round(max(0.0, now_ts - ref_ts) / 60.0, 1))
            return enriched

        # 1. Fear & Greed sensor
        for fg_name in ("fear_greed_cache.json", "fear_greed_collect.json"):
            fg_path = os.path.join(self.cache_dir, fg_name)
            data = _read_json(fg_path)
            if isinstance(data, dict) and data:
                snapshot.fear_greed = _with_age_min(data, fg_path)
                break

        # 2. Market Breadth sensor
        for mb_name in ("market_breadth_cache.json", "market_breadth_collect.json"):
            mb_path = os.path.join(self.cache_dir, mb_name)
            data = _read_json(mb_path)
            if isinstance(data, dict) and data:
                snapshot.market_breadth = _with_age_min(data, mb_path)
                break

        # 3. Macro Sentiment Advisor sensor
        for sa_name in ("sentiment_advisor_eval.json", "macro_advisor_eval.json", "gemini_macro_advisor.json"):
            sa_path = os.path.join(self.cache_dir, sa_name)
            data = _read_json(sa_path)
            if isinstance(data, dict) and data:
                snapshot.sentiment_advisor = _with_age_min(data, sa_path)
                break

        # 4. Geopolitical Threat Shield sensor + live 1-minute Breaking News RSS feed
        for gt_name in ("geopolitical_threat_eval.json", "geopolitical_threat_state.json"):
            gt_path = os.path.join(self.cache_dir, gt_name)
            data = _read_json(gt_path)
            if isinstance(data, dict) and data:
                snapshot.geopolitical_threat = _with_age_min(data, gt_path)
                break

        news_data = _read_json(os.path.join(self.cache_dir, "news_feed_collect.json"))
        if isinstance(news_data, list) and news_data:
            valid_news = [item for item in news_data if isinstance(item, dict) and item.get("title")]
            valid_news.sort(
                key=lambda x: (
                    1 if x.get("is_high_severity") else 0,
                    float(x.get("first_seen_ts") or 0.0),
                ),
                reverse=True,
            )
            for item in valid_news[:6]:
                seen_ts = float(item.get("first_seen_ts") or 0.0)
                age_m = round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None
                snapshot.breaking_news.append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "is_high_severity": bool(item.get("is_high_severity")),
                    "keywords": item.get("matched_keywords") or [],
                    "age_min": age_m,
                })

        # 5. Heartbeat sensors
        for hb_name in (
            "macro_analyzer.heartbeat",
            "globaltelemetry_collector.heartbeat",
            "intelligence_daemon.heartbeat",
            "rtrade.heartbeat",
        ):
            data = _read_json(os.path.join(self.cache_dir, hb_name))
            if isinstance(data, dict) and data:
                snapshot.active_heartbeats[hb_name] = data

        # 6. Bot state files in workspace
        for s_file in (".watchdog_state.json", ".anomaly_watchdog_state.json", "kraken_trades_full.json"):
            s_path = os.path.join(self.workspace_dir, s_file)
            if os.path.exists(s_path):
                mtime = os.path.getmtime(s_path)
                snapshot.state_files[s_file] = {"age_hours": (now_ts - mtime) / 3600.0}

        # 7. Live market prices, short/long/Weibull trend, 1s 24trend windows, 7d range, and Kalman velocity
        prices_data = _read_json(os.path.join(self.cache_dir, "cache_current_price.json"))
        trends_data = _read_json(os.path.join(self.cache_dir, "cache_short_trend.json"))
        long_trends_data = _read_json(os.path.join(self.cache_dir, "cache_long_trend.json"))
        t_trend_data = _read_json(os.path.join(self.cache_dir, "cache_T_trend.json"))
        kalman_data = _read_json(os.path.join(self.cache_dir, "kalman_state.json"))
        prices_multi_data = _read_json(os.path.join(self.cache_dir, "cache_prices_multi.json"))
        trailing_data = _read_json(os.path.join(self.cache_dir, "trailing_state.json"))
        trailing_map = trailing_data if isinstance(trailing_data, dict) else {}

        price_map: Dict[str, float] = {}
        if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict):
            for sym, item in prices_data["items"].items():
                if isinstance(item, list) and item and isinstance(item[0], list) and len(item[0]) > 1:
                    try:
                        price_map[sym] = float(item[0][1])
                    except (ValueError, TypeError):
                        pass

        long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}
        t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}

        if isinstance(trends_data, dict):
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
                    snapshot.prices_and_trends[sym] = entry

        for sym, px in price_map.items():
            if sym not in snapshot.prices_and_trends:
                snapshot.prices_and_trends[sym] = {"price": px, "trend": 0, "growth_coefficient": 0.0}

        # Overlay fresh 1-second sampled dynamic windows from cachedb/24trend/live_<SYM>.json
        live_24trend_dir = os.path.join(self.cache_dir, "24trend")
        if os.path.isdir(live_24trend_dir):
            for lt_path in sorted(glob.glob(os.path.join(live_24trend_dir, "live_*.json"))):
                fname = os.path.basename(lt_path)
                sym = fname[len("live_") : -len(".json")].upper()
                if not sym:
                    continue
                lt_win = _read_json(lt_path)
                if not isinstance(lt_win, dict):
                    continue
                w222 = lt_win.get("222") if isinstance(lt_win.get("222"), dict) else {}
                w900 = lt_win.get("900") if isinstance(lt_win.get("900"), dict) else {}
                w9000 = lt_win.get("9000") if isinstance(lt_win.get("9000"), dict) else {}
                entry = snapshot.prices_and_trends.setdefault(
                    sym,
                    {
                        "price": w222.get("current_price") or w900.get("current_price") or price_map.get(sym),
                        "trend": w222.get("final_trend", 0),
                        "growth_coefficient": w222.get("growth_coefficient", 0.0),
                    },
                )
                if entry.get("price") is None and (w222.get("current_price") or w900.get("current_price")):
                    entry["price"] = w222.get("current_price") or w900.get("current_price")
                if w222:
                    entry["slope_3_7m"] = w222.get("slope_full", entry.get("slope_3_7m", 0.0))
                    entry["trend_3_7m"] = w222.get("final_trend", entry.get("trend_3_7m", 0))
                    entry["growth_3_7m"] = w222.get("growth_coefficient")
                if w900:
                    entry["slope_15m"] = w900.get("slope_full", entry.get("slope_15m", 0.0))
                    entry["trend_15m"] = w900.get("final_trend", entry.get("trend_15m", 0))
                    entry["growth_15m"] = w900.get("growth_coefficient")
                if w9000:
                    entry["slope_2_5h"] = w9000.get("slope_full", entry.get("slope_2_5h", 0.0))
                    entry["trend_2_5h"] = w9000.get("final_trend", entry.get("trend_2_5h", 0))
                    entry["growth_2_5h"] = w9000.get("growth_coefficient")

        # Enrich every symbol with Kalman velocity/volatility, Long Trend forecast, Weibull survival & 7d range
        multi_items = prices_multi_data.get("items", {}) if isinstance(prices_multi_data, dict) else {}
        for sym, entry in list(snapshot.prices_and_trends.items()):
            if isinstance(kalman_data, dict) and isinstance(kalman_data.get(sym), dict):
                k_sym = kalman_data[sym]
                entry["kalman_vel"] = k_sym.get("kalman_vel")
                entry["kalman_vel_std"] = k_sym.get("kalman_vel_std")
                entry["kalman_trend"] = k_sym.get("kalman_trend")
                if entry.get("vol_1h_pct") is None and k_sym.get("vol_1h_pct") is not None:
                    entry["vol_1h_pct"] = k_sym.get("vol_1h_pct")
                if k_sym.get("adapt_reentry_pct") is not None:
                    entry["adapt_reentry_pct"] = k_sym.get("adapt_reentry_pct")
                if k_sym.get("adapt_dca_pct") is not None:
                    entry["adapt_dca_pct"] = k_sym.get("adapt_dca_pct")
            if isinstance(long_items, dict) and isinstance(long_items.get(sym), list) and long_items[sym]:
                lt = long_items[sym][-1]
                if isinstance(lt, dict):
                    entry["long_trend_dir"] = lt.get("direction")
                    entry["long_trend_hours"] = round(float(lt.get("duration_seconds", 0.0)) / 3600.0, 1)
                    if isinstance(lt.get("estimated_future_hours"), (int, float)):
                        entry["long_trend_est_rem_hours"] = round(float(lt["estimated_future_hours"]), 1)
            if isinstance(t_trend_map.get(sym), dict):
                tt = t_trend_map[sym]
                entry["weibull_median_d"] = tt.get("median_d")
                entry["weibull_p90_d"] = tt.get("p90_d")
            if isinstance(multi_items, dict):
                base_prefix = sym[:-4] if sym.endswith(("USDC", "USDT")) else (sym[:-3] if sym.endswith(("USD", "EUR")) else sym)
                m_series = multi_items.get(sym) or multi_items.get(base_prefix)
                if isinstance(m_series, list) and len(m_series) >= 2:
                    try:
                        vals = [float(pt[1]) for pt in m_series if isinstance(pt, list) and len(pt) > 1 and float(pt[1]) > 0]
                        if len(vals) >= 2:
                            entry["chg_7d_pct"] = round(((vals[-1] - vals[0]) / vals[0]) * 100.0, 2)
                            entry["low_7d"] = min(vals)
                            entry["high_7d"] = max(vals)
                    except (ValueError, TypeError):
                        pass

        # 7B. Discover all enabled tradable instruments & real spot wallet balances (when workspace has instruments.conf)
        tradable_positions: List[Dict[str, Any]] = []
        venue_quote_cash: Dict[str, Dict[str, Any]] = {}
        load_tradable_fn = getattr(self.intent_gateway, "load_tradable_instruments", None)
        if callable(load_tradable_fn) and os.path.exists(os.path.join(self.workspace_dir, "instruments.conf")):
            try:
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
                        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
                            val_f = float(raw)
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

                    px = snapshot.prices_and_trends.get(sym, {}).get("price")
                    if not (isinstance(px, (int, float)) and math.isfinite(float(px)) and float(px) > 0):
                        try:
                            get_px_fn = getattr(inst._provider, "get_current_price", None)
                            if callable(get_px_fn):
                                live_px = float(get_px_fn(sym))
                                if math.isfinite(live_px) and live_px > 0:
                                    px = live_px
                                    snapshot.prices_and_trends.setdefault(
                                        sym, {"price": px, "trend": 0, "growth_coefficient": 0.0}
                                    )["price"] = px
                        except Exception:
                            px = None

                    base_free = _query_free_bal(inst, base)
                    quote_free = _query_free_bal(inst, quote)
                    if quote_free is not None:
                        venue_quote_cash[f"{venue}:{quote}"] = {
                            "venue": venue,
                            "asset": quote,
                            "free": round(quote_free, 2),
                        }

                    pos_val_usd = (
                        round(base_free * float(px), 2)
                        if base_free is not None and isinstance(px, (int, float)) and float(px) > 0
                        else None
                    )
                    params = getattr(inst, "params", {}) if isinstance(getattr(inst, "params", None), dict) else {}
                    t_state = trailing_map.get(sym, {}) if isinstance(trailing_map.get(sym), dict) else {}
                    peak = t_state.get("peak")
                    dd_from_peak_pct = (
                        round(((float(px) - float(peak)) / float(peak)) * 100.0, 2)
                        if isinstance(px, (int, float)) and isinstance(peak, (int, float)) and float(peak) > 0
                        else None
                    )
                    pos_entry: Dict[str, Any] = {
                        "venue": venue,
                        "symbol": sym,
                        "base": base,
                        "quote": quote,
                        "price": px,
                        "free_base": round(base_free, 6) if base_free is not None else None,
                        "value_usd": pos_val_usd,
                        "free_quote": round(quote_free, 2) if quote_free is not None else None,
                    }
                    if peak is not None:
                        pos_entry["trailing_peak"] = peak
                        pos_entry["dd_from_peak_pct"] = dd_from_peak_pct
                    if params.get("trailing.pct"):
                        pos_entry["trailing_pct_cfg"] = params.get("trailing.pct")
                    if params.get("mt.gain") or params.get("mt.lost"):
                        pos_entry["mt_gain_pct"] = params.get("mt.gain")
                        pos_entry["mt_lost_pct"] = params.get("mt.lost")
                    tradable_positions.append(pos_entry)

        if tradable_positions:
            snapshot.portfolio_summary["tradable_positions"] = tradable_positions
        if venue_quote_cash:
            snapshot.portfolio_summary["venue_quote_cash"] = venue_quote_cash
        if trailing_map:
            snapshot.portfolio_summary["trailing_peaks"] = trailing_map

        # 8. 24h multi-horizon price evolution & Pillar 2 microstructure per symbol
        for sym in list(snapshot.prices_and_trends.keys()):
            p24_data = _read_json(os.path.join(self.cache_dir, f"cache_24_price_{sym}.json"))
            if isinstance(p24_data, dict):
                raw_items = p24_data.get("items", {})
                series = raw_items.get(sym, []) if isinstance(raw_items, dict) else raw_items
                m24 = compute_24h_price_metrics(series)
                if m24:
                    snapshot.prices_and_trends[sym].update(m24)

            base_sym = sym[:-4] if sym.endswith(("USDC", "USDT")) else (sym[:-3] if sym.endswith(("USD", "EUR")) else sym)
            ob = _read_json(os.path.join(self.cache_dir, f"orderbook_depth_{sym}.json"))
            if ob is None and base_sym:
                ob = _read_json(os.path.join(self.cache_dir, f"orderbook_depth_{base_sym}USDT.json"))
            deriv = _read_json(os.path.join(self.cache_dir, f"derivatives_telemetry_{sym}.json"))
            if deriv is None and base_sym:
                deriv = _read_json(os.path.join(self.cache_dir, f"derivatives_telemetry_{base_sym}USDT.json"))
            whale = _read_json(os.path.join(self.cache_dir, f"whale_snapshot_{sym}.json"))
            if whale is None and base_sym:
                whale = _read_json(os.path.join(self.cache_dir, f"whale_snapshot_{base_sym}USDT.json"))
            if ob or deriv or whale:
                m_info: Dict[str, Any] = {}
                if isinstance(ob, dict):
                    m_info["ob_imbalance"] = ob.get("imbalance_ratio")
                    m_info["bid_depth_usd"] = round(float(ob["bid_depth_usd"]), 0) if isinstance(ob.get("bid_depth_usd"), (int, float)) else None
                    m_info["ask_depth_usd"] = round(float(ob["ask_depth_usd"]), 0) if isinstance(ob.get("ask_depth_usd"), (int, float)) else None
                    m_info["bid_wall_px"] = ob.get("largest_bid_wall_price")
                    m_info["bid_wall_usd"] = round(float(ob["largest_bid_wall_usd"]), 0) if isinstance(ob.get("largest_bid_wall_usd"), (int, float)) else None
                    m_info["ask_wall_px"] = ob.get("largest_ask_wall_price")
                    m_info["ask_wall_usd"] = round(float(ob["largest_ask_wall_usd"]), 0) if isinstance(ob.get("largest_ask_wall_usd"), (int, float)) else None
                if isinstance(deriv, dict):
                    m_info["funding_rate"] = deriv.get("funding_rate")
                    m_info["pred_funding_rate"] = deriv.get("predicted_funding_rate")
                    m_info["oi_usd"] = deriv.get("open_interest_usd")
                if isinstance(whale, dict):
                    m_info["whale_long_pct"] = whale.get("top_traders_long_pct")
                    m_info["whale_long_ratio"] = whale.get("top_traders_long_ratio")
                    m_info["taker_buy_sell_ratio"] = whale.get("taker_buy_sell_ratio")
                    m_info["taker_buy_vol_usd"] = whale.get("taker_buy_vol_usd")
                    m_info["taker_sell_vol_usd"] = whale.get("taker_sell_vol_usd")
                    m_info["oi_1h_chg_pct"] = whale.get("open_interest_1h_change_pct")
                    m_info["divergence_regime"] = whale.get("divergence_regime")
                snapshot.microstructure[sym] = m_info

        # 9. Portfolio value, Kraken cycle states, cross-venue snapshot, recent orders/fills & autonomous intents
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

        # 9B. Kraken active DCA cycle states (kraken/.state_<PAIR>.json)
        kraken_cycles: Dict[str, Any] = {}
        kraken_dir = os.path.join(self.workspace_dir, "kraken")
        if os.path.isdir(kraken_dir):
            for st_path in sorted(glob.glob(os.path.join(kraken_dir, ".state_*.json"))):
                fname = os.path.basename(st_path)
                pair_name = fname[len(".state_") : -len(".json")].upper()
                if not pair_name.endswith(("USD", "EUR", "USDC", "USDT")):
                    continue
                st_data = _read_json(st_path)
                if not isinstance(st_data, dict):
                    continue
                qty = float(st_data.get("qty") or 0.0)
                cost = float(st_data.get("cost") or 0.0)
                avg_cost = round(cost / qty, 6) if qty > 1e-12 and cost > 0 else None
                px = snapshot.prices_and_trends.get(pair_name, {}).get("price")
                dd_pct = (
                    round(((float(px) - avg_cost) / avg_cost) * 100.0, 2)
                    if avg_cost and isinstance(px, (int, float)) and float(px) > 0
                    else None
                )
                kraken_cycles[pair_name] = {
                    "qty": round(qty, 6),
                    "cost_usd": round(cost, 2),
                    "avg_cost_px": avg_cost,
                    "entry_price": st_data.get("entry_price"),
                    "last_buy_price": st_data.get("last_buy_price"),
                    "last_sell_price": st_data.get("last_sell_price"),
                    "last_exit_kind": st_data.get("last_exit_kind"),
                    "dca_buys": st_data.get("dca_buys"),
                    "dd_pct": dd_pct,
                }
        if kraken_cycles:
            snapshot.portfolio_summary["kraken_cycles"] = kraken_cycles

        # 9C. Cross-venue portfolio snapshot tail (e.g. Hyperliquid read-only holdings)
        for l_dir in self.logs_dirs:
            ps_path = os.path.join(l_dir, "portfolio_snapshot.jsonl")
            if os.path.exists(ps_path):
                try:
                    with open(ps_path, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        ps_lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
                    if ps_lines:
                        last_ps = json.loads(ps_lines[-1])
                        if isinstance(last_ps, dict) and isinstance(last_ps.get("hl"), dict):
                            snapshot.portfolio_summary["hyperliquid_readonly"] = last_ps["hl"]
                except Exception:
                    pass
                break

        # 9D. Recent fills across Binance (cache_order.json) and Kraken (cache_trade_kraken.json)
        recent_orders: Dict[str, Any] = {}
        for order_cache_name in ("cache_order.json", "cache_trade_kraken.json"):
            orders_data = _read_json(os.path.join(self.cache_dir, order_cache_name))
            if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
                for sym, o_list in orders_data["items"].items():
                    if sym in recent_orders:
                        continue
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
                    "venue": i.get("venue"),
                    "status": i.get("status"),
                    "notional_usd": i.get("suggested_notional_usd"),
                    "ts": i.get("timestamp"),
                }
                for i in intents_data[-8:]
                if isinstance(i, dict)
            ]

        # 9E. Active queued retry orders (cachedb/order_retry_queue.jsonl) & monitortrades active states (mt_state.json)
        retry_q_path = os.path.join(self.cache_dir, "order_retry_queue.jsonl")
        if os.path.exists(retry_q_path):
            active_retries: List[Dict[str, Any]] = []
            try:
                with open(retry_q_path, "r", encoding="utf-8", errors="replace") as f:
                    for raw_line in f.readlines()[-50:]:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            rec = json.loads(raw_line)
                        except ValueError:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        lifecycle = str(rec.get("lifecycle") or "").lower()
                        exp_ts = float(rec.get("expires_at_ts") or 0.0)
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
            except Exception:
                pass
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

        # 10. Fleet processes health from procs.conf (only needed for SRE/self-heal audits)
        if include_fleet:
            try:
                from orchestratorOS.livecheck.autonomous_overnight_monitor import AutonomousOvernightMonitor
                monitor = AutonomousOvernightMonitor(workspace_dir=self.workspace_dir)
                snapshot.fleet_processes = [asdict(p) for p in monitor.check_processes()]
            except Exception:
                pass

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
        """Feeds full sensor, market, and portfolio telemetry to Gemini; the topic spec bounds side effects."""
        fg = sensors.fear_greed or {}
        mb = sensors.market_breadth or {}
        sa = sensors.sentiment_advisor or {}
        gt = sensors.geopolitical_threat or {}

        fg_hist = fg.get("historical_values")[:7] if isinstance(fg.get("historical_values"), list) else None
        fg_line = (
            f"{fg.get('value', 'N/A')} ({fg.get('classification') or fg.get('sentiment') or 'N/A'}, "
            f"7d_chg={fg.get('trend_7d_change', 'N/A')}"
            + (f", 7d_hist={fg_hist}" if fg_hist else "")
            + (f", age={fg.get('age_min')}m" if fg.get("age_min") is not None else "")
            + ")"
            if fg else "None"
        )
        mb_gainers = mb.get("top_gainers")[:3] if isinstance(mb.get("top_gainers"), list) else None
        mb_losers = mb.get("top_losers")[:3] if isinstance(mb.get("top_losers"), list) else None
        mb_line = (
            f"advance_ratio={mb.get('advance_ratio', 'N/A')}, regime={mb.get('regime', 'N/A')}, "
            f"median_chg={mb.get('median_change_pct', 'N/A')}%, "
            f"mean_chg={mb.get('mean_change_pct', 'N/A')}%, disp_std={mb.get('dispersion_std', 'N/A')}, "
            f"advancing={mb.get('advancing', 'N/A')}/{mb.get('total_symbols', 'N/A')}"
            + (f", top_gainers={mb_gainers}" if mb_gainers else "")
            + (f", top_losers={mb_losers}" if mb_losers else "")
            if mb else "None"
        )
        sa_line = (
            f"bias={sa.get('market_bias', 'N/A')}, risk={sa.get('risk_level', 'N/A')}, "
            f"action={sa.get('recommended_action', 'N/A')}, conf={sa.get('confidence', 'N/A')}"
            + (f", age={sa.get('age_min')}m" if sa.get("age_min") is not None else "")
            + f" | summary={sa.get('summary', '')} | key_risks={sa.get('key_risks', [])}"
            if sa else "None"
        )
        gt_line = (
            f"level={gt.get('threat_level', 'N/A')}, risk_score={gt.get('risk_score', 'N/A')}, "
            f"brake={gt.get('recommended_brake', 'N/A')}"
            + (f", age={gt.get('age_min')}m" if gt.get("age_min") is not None else "")
            + f" | summary={gt.get('summary', '')}"
            if gt else "None"
        )

        spec = topic_spec(topic_key)

        context_items = [
            "### 1. ACTIVE MACRO, SENTIMENT, GEOPOLITICAL & LIVE BREAKING NEWS TELEMETRY:",
            f"- Fear & Greed: {fg_line}",
            f"- Market Breadth: {mb_line}",
            f"- Macro Sentiment Advisor: {sa_line}",
            f"- Geopolitical Threat Shield: {gt_line}",
        ]
        if sensors.breaking_news:
            context_items.append("- Live Breaking News Feed (cachedb/news_feed_collect.json, 1m refresh):")
            for hn in sensors.breaking_news:
                sev_tag = "[HIGH-SEV] " if hn.get("is_high_severity") else ""
                kw_sfx = f" kw={hn.get('keywords')}" if hn.get("keywords") else ""
                age_sfx = f" (age={hn.get('age_min')}m)" if hn.get("age_min") is not None else ""
                context_items.append(f"  * {sev_tag}{hn.get('title')}{kw_sfx}{age_sfx}")
        if spec.can_self_heal:
            context_items.extend([
                f"- Active Heartbeats: {list(sensors.active_heartbeats.keys())}",
                f"\n### 2. LOG ANOMALIES & GUARD VETOES (PAST {self.lookback_hours:g}H):",
                f"- Scanned Logs: {logs.total_scanned_files}",
                f"- Error Count: {logs.error_count}",
                f"- Guard Interventions: {logs.warning_count}",
                "- Sample Errors & Interventions:",
            ])
            for evt in logs.critical_events[:15]:
                context_items.append(f"  * {evt}")
            for g_evt in logs.guard_interventions[:10]:
                context_items.append(f"  * [GUARD] {g_evt}")

            if sensors.fleet_processes:
                running_p = [p for p in sensors.fleet_processes if p.get("is_running")]
                stopped_p = [p["name"] for p in sensors.fleet_processes if not p.get("is_running")]
                stale_p = [
                    f"{p['name']} (age={p.get('age_seconds')}s > {p.get('max_stale_seconds')}s)"
                    for p in sensors.fleet_processes
                    if p.get("is_running") and not p.get("heartbeat_fresh", True)
                ]
                context_items.append(
                    f"\n### 2B. FLEET PROCESSES (procs.conf): {len(running_p)}/{len(sensors.fleet_processes)} running"
                    + (f" | STOPPED: {stopped_p}" if stopped_p else "")
                    + (f" | STALE HEARTBEAT: {stale_p}" if stale_p else " | All heartbeats fresh")
                )

        if sensors.prices_and_trends:
            context_items.append("\n### 3. LIVE MARKET PRICES, MULTI-HORIZON TRENDS & 24H EVOLUTION (ALL COINS):")
            for sym, pinfo in sorted(sensors.prices_and_trends.items()):
                px = pinfo.get("price")
                px_str = (
                    f"${px:,.2f}"
                    if isinstance(px, (int, float)) and px >= 1.0
                    else (f"${px:.6g}" if isinstance(px, (int, float)) else (f"${px}" if px is not None else "N/A"))
                )
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
                if "chg_7d_pct" in pinfo:
                    extras.append(
                        f"7d={pinfo.get('chg_7d_pct', 0):+.2f}% 7d_range=[{pinfo.get('low_7d')}, {pinfo.get('high_7d')}]"
                    )
                if "slope_15m" in pinfo:
                    extras.append(
                        f"slopes(3.7m={pinfo.get('slope_3_7m', 0):+.5f}[T={pinfo.get('trend_3_7m', 0)}], "
                        f"15m={pinfo.get('slope_15m', 0):+.5f}[T={pinfo.get('trend_15m', 0)}], "
                        f"2.5h={pinfo.get('slope_2_5h', 0):+.5f}[T={pinfo.get('trend_2_5h', 0)}])"
                    )
                if pinfo.get("vol_1h_pct") is not None:
                    vol_part = f"vol_1h={pinfo.get('vol_1h_pct')}%"
                    if pinfo.get("adapt_reentry_pct") is not None:
                        vol_part += f"(adapt_reentry={pinfo.get('adapt_reentry_pct')}%)"
                    extras.append(vol_part)
                if pinfo.get("kalman_vel") is not None:
                    extras.append(f"kalman_vel={pinfo.get('kalman_vel'):+.5f}")
                if pinfo.get("long_trend_dir"):
                    lt_str = f"long_trend={pinfo.get('long_trend_dir')}({pinfo.get('long_trend_hours')}h"
                    if pinfo.get("long_trend_est_rem_hours") is not None:
                        lt_str += f", est_rem={pinfo.get('long_trend_est_rem_hours')}h"
                    if pinfo.get("weibull_median_d") is not None:
                        lt_str += f", med_d={pinfo.get('weibull_median_d')}d/p90={pinfo.get('weibull_p90_d')}d"
                    lt_str += ")"
                    extras.append(lt_str)
                extra_str = f" | {' | '.join(extras)}" if extras else ""
                context_items.append(f"  * {sym}: Price={px_str}, Trend={tr_label}, Growth={growth:+.5f}{extra_str}")

        if sensors.microstructure:
            context_items.append("\n### 3B. PILLAR 2 MICROSTRUCTURE, ORDERBOOK WALLS, DERIVATIVES & WHALE FLOW:")
            for sym, minfo in sorted(sensors.microstructure.items()):
                context_items.append(
                    f"  * {sym}: ob_imbalance={minfo.get('ob_imbalance')} "
                    f"(bid_depth=${minfo.get('bid_depth_usd')}, ask_depth=${minfo.get('ask_depth_usd')}), "
                    f"bid_wall={minfo.get('bid_wall_px')} (${minfo.get('bid_wall_usd')}), "
                    f"ask_wall={minfo.get('ask_wall_px')} (${minfo.get('ask_wall_usd')}), "
                    f"whale_long_pct={minfo.get('whale_long_pct')}, div_regime={minfo.get('divergence_regime')}, "
                    f"taker_ratio={minfo.get('taker_buy_sell_ratio')} "
                    f"(buy_v=${minfo.get('taker_buy_vol_usd')}/sell_v=${minfo.get('taker_sell_vol_usd')}), "
                    f"funding={minfo.get('funding_rate')} (pred={minfo.get('pred_funding_rate')}), "
                    f"oi_usd={minfo.get('oi_usd')}, oi_1h_chg={minfo.get('oi_1h_chg_pct')}%"
                )

        if sensors.portfolio_summary:
            ps = sensors.portfolio_summary
            context_items.append("\n### 3C. SPOT WALLET BALANCES, TRADABLE POSITIONS, CYCLE COST ANCHORS & LAST FILLS:")
            if ps.get("total_value_usdc") is not None:
                context_items.append(f"  * Total Binance Portfolio Value: ${ps['total_value_usdc']:,.2f} USDC")
            if ps.get("venue_quote_cash"):
                cash_parts = [
                    f"{v['venue']} {v['asset']}=${v['free']:,.2f}"
                    for v in ps["venue_quote_cash"].values()
                    if isinstance(v, dict)
                ]
                if cash_parts:
                    context_items.append(f"  * Available Free Quote Cash per Venue: {', '.join(cash_parts)}")
            if ps.get("tradable_positions"):
                context_items.append("  * Live Tradable Spot Instruments (Eligible for Autonomous BUY / SELL):")
                for pos in ps["tradable_positions"]:
                    t_extra = ""
                    if pos.get("trailing_peak") is not None:
                        t_extra += f", trail_peak={pos.get('trailing_peak')} (dd_from_peak={pos.get('dd_from_peak_pct')}%)"
                    if pos.get("mt_gain_pct") or pos.get("mt_lost_pct"):
                        t_extra += f", bot_ref_gain={pos.get('mt_gain_pct')}%/lost={pos.get('mt_lost_pct')}%"
                    context_items.append(
                        f"    - [{pos['venue']}] {pos['symbol']}: free_{pos['base']}={pos.get('free_base')} "
                        f"(${pos.get('value_usd')} USD), free_{pos['quote']}=${pos.get('free_quote')}, "
                        f"px={pos.get('price')}{t_extra}"
                    )
            if ps.get("kraken_cycles"):
                context_items.append("  * Kraken Active Cycle Cost & Drawdown Anchors:")
                for k_sym, kc in sorted(ps["kraken_cycles"].items()):
                    context_items.append(
                        f"    - [kraken] {k_sym}: qty={kc.get('qty')}, cost=${kc.get('cost_usd')}, "
                        f"avg_cost={kc.get('avg_cost_px')}, entry_px={kc.get('entry_price')}, "
                        f"last_buy={kc.get('last_buy_price')}, last_sell={kc.get('last_sell_price')} "
                        f"({kc.get('last_exit_kind')}), dca_buys={kc.get('dca_buys')}, dd={kc.get('dd_pct')}%"
                    )
            if ps.get("hyperliquid_readonly"):
                context_items.append(f"  * Hyperliquid Read-Only Holding (managed by HL-bot): {ps['hyperliquid_readonly']}")
            for sym, oinfo in sorted((ps.get("recent_orders") or {}).items()):
                lb = oinfo.get("last_buy") or {}
                ls = oinfo.get("last_sell") or {}
                now_ms = int(time.time() * 1000)
                lb_age_h = round((now_ms - int(lb["ts_ms"])) / 3_600_000.0, 1) if lb.get("ts_ms") else None
                ls_age_h = round((now_ms - int(ls["ts_ms"])) / 3_600_000.0, 1) if ls.get("ts_ms") else None
                context_items.append(
                    f"  * {sym} Last Action={oinfo.get('last_action_side')}: "
                    f"Last BUY={lb.get('qty')} @ {lb.get('price')} (age={lb_age_h}h) | "
                    f"Last SELL={ls.get('qty')} @ {ls.get('price')} (age={ls_age_h}h)"
                )
            if ps.get("active_retry_queue"):
                context_items.append(f"  * Active Queued Retry Orders (order_retry_queue.jsonl): {ps['active_retry_queue']}")
            if ps.get("monitortrades_active_state"):
                context_items.append(f"  * Active MonitorTrades Ride/Stop State (mt_state.json): {ps['monitortrades_active_state']}")
            if ps.get("recent_ai_intents"):
                context_items.append(f"  * Recent Autonomous Intents: {ps['recent_ai_intents']}")

        if focus_symbols:
            clean_syms = [s.strip().upper() for s in focus_symbols if s.strip()]
            if clean_syms:
                context_items.append(
                    f"\n### 4. TACTICAL PRIORITY SYMBOLS: {', '.join(clean_syms)}\n"
                    "- Pay heightened attention to these symbols alongside all other live tradable coins above."
                )

        context_str = "\n".join(context_items)

        tasks: List[str] = []
        schema: List[str] = []
        if spec.can_self_heal:
            tasks.extend([
                "1. Explain the technical root cause of any bot exceptions, fail-closed blocks, stale heartbeats, or guard vetoes.",
                "2. Determine whether open orders, venue positions, or hung/errored fleet bots require reconciliation or restart.",
                "3. If a reproducible code bug exists, formulate an exact search_block and replace_block.",
            ])
            schema.extend([
                '  "summary": "1-2 sentence executive assessment of system health and multi-coin market posture"',
                '  "diagnosed_issues": [{"severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO", '
                '"category": "BOT_EXCEPTION" | "ORPHAN_ORDER" | "FLEET_STALE" | "SYSTEM_ERROR" | "OS_RESOURCE" | "VPN_ERROR" | "GUARD_VETO" | "HEALTHY", '
                '"description": "Short explanation", "root_cause": "Detailed technical root cause"}]',
                '  "venue_actions": [{"action": "CANCEL_ORDER" | "SYNC_POSITION" | "RESTART_BOT" | "NONE", '
                '"venue": "binance" | "kraken" | "hyperliquid" | "t212" | "fleet", "symbol": "BTCUSDC or bot_name", '
                '"order_id": "optional order id", "reason": "Why this action is needed"}]',
                '  "code_remediation": {"has_fix": false, "target_file": "", "rationale": "", '
                '"search_block": "", "replace_block": ""}',
            ])
        else:
            schema.append(
                '  "summary": "1-2 sentence executive assessment of multi-coin market posture and trading decisions"'
            )
        if spec.can_trade:
            gw = self.intent_gateway
            max_usd = _parse_finite_float(getattr(gw, "max_notional_usd", None), 2000.0)
            min_usd = _parse_finite_float(getattr(gw, "min_notional_usd", None), 10.0)
            min_conf = _parse_finite_float(getattr(gw, "min_confidence", None), 0.85)
            tasks.append(
                f"{len(tasks) + 1}. Formulate symmetric autonomous trading decisions across ALL live tradable spot coins "
                "(e.g. BTCUSDC, TAOUSDC, ARBUSDC on 'binance'; HYPEUSD, TAOUSD, ADAUSD on 'kraken') using the complete "
                "macro sentiment, geopolitical risk, live breaking news, market breadth, multi-horizon price evolution "
                "(15m/30m/1h/3h/24h/7d), 1s-sampled dynamic slopes (3.7m/15m/2.5h), Kalman velocity/volatility, "
                "Weibull long-trend survival, Pillar 2 whale/orderbook walls/taker flow, and real spot wallet balances + "
                "cycle entry anchors + active retry queue above.\n"
                "   - FULL AI AUTHORITY (ZERO PROFIT OR STRATEGY GUARDS): ALL deterministic order_guard gates (G_PROF "
                "profit reference, GO1 daily cap/cooldown, GT_* trend deferrals, G_QLIMIT* weight caps, GI_* microstructure, "
                "and GLLM* secondary vetoes) are BYPASSED symmetrically on both BUY and SELL. You hold 100% quantitative "
                "responsibility for profit harvesting, multi-tranche scaling, and capital protection:\n"
                "     * SELL (Take-Profit OR Defensive Risk Reduction): Emit 'SELL' on any tradable coin with free base "
                f"inventory >= ${min_usd:g} USD BOTH when an advance shows rally exhaustion AND when protective/defensive "
                "de-risking is warranted EVEN BELOW entry/last-BUY price (e.g. structural breakdown, accelerating multi-horizon "
                "selloff, distribution, or macro/geopolitical shock where cutting/trimming exposure protects capital against a "
                "deeper drop). You may scale out in tranches across consecutive 10m cycles up to the per-order notional cap. "
                "Distinguish true structural breakdowns from shallow liquidity sweeps into strong bid walls.\n"
                "     * BUY (Accumulation, Re-Entry & Breakout): Emit 'BUY' on any tradable coin whose venue has free quote "
                f"cash >= ${min_usd:g} USD on high-conviction washed-out reversals, steady upward recovery/inflection after "
                "a prior exit, or confirmed momentum breakouts (scaling in across 10m cycles when warranted).\n"
                f"   - Emit one entry per actionable coin in `market_trading_decisions` with `confidence >= {min_conf:g}` "
                f"and `suggested_notional_usd` sized between {min_usd:g} and {max_usd:g} USD (the gateway automatically "
                "clamps notional strictly to available free wallet balance with zero artificial cooldowns or dedup locks). "
                "If no coin warrants action right now, return a single 'HOLD' entry."
            )
            schema.append(
                '  "market_trading_decisions": [{"decision": "BUY" | "SELL" | "HOLD", "symbol": "BTCUSDC", '
                '"venue": "binance" | "kraken", "confidence": 0.90, '
                f'"suggested_notional_usd": 500.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}}]'
            )

        if spec.can_self_heal:
            role_intro = (
                "You are the Lead SRE and Quantitative Autonomous System Supervisor for MPTrade.\n"
                "Analyze the complete telemetry sensors, spot wallet balances, cycle states, and log anomalies:\n\n"
                f"{context_str}\n\n"
                "Perform an executive diagnostic and self-healing audit:\n" + "\n".join(tasks) + "\n\n"
            )
            tool_rule = (
                "- If you diagnose a reproducible code bug and need to verify exact lines before proposing `code_remediation`, "
                "you MAY read the target file in the workspace so `search_block` matches verbatim.\n"
            )
            report_format_rule = (
                "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
                "  1) A concise, well-structured Markdown report in **Romanian** (bullet points for: Starea Flotei & Loguri, "
                "Piața Live & Pante (Toate Monedele), Portofoliu, Solduri & Poziții Active, și Decizia AI / Răspuns la intervențiile din chat).\n"
            )
        else:
            role_intro = (
                "You are the Quantitative Autonomous Portfolio Manager and Trade Executor for MPTrade.\n"
                "Analyze the complete market telemetry, macro/geopolitical sensors, microstructure, spot wallet balances, and cycle states:\n\n"
                f"{context_str}\n\n"
                "Evaluate the market context and formulate autonomous BUY/SELL trading decisions:\n" + "\n".join(tasks) + "\n\n"
            )
            tool_rule = (
                "- For this automated telemetry snapshot turn, do NOT call tools, shell commands, or file readers because all "
                "required telemetry and wallet balances are provided above (use tools only when the user asks a direct interactive question in chat).\n"
                "- Do NOT check or report on fleet process state, heartbeats, or system logs ('Starea Flotei & Loguri'); "
                "your sole responsibility on this thread is executing BUY/SELL transactions based on market and portfolio context.\n"
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
        """Executes a single repair action (CANCEL_ORDER, SYNC_POSITION, or RESTART_BOT) in enforce mode."""
        action_type = str(act.get("action") or "NONE").strip().upper()
        venue = str(act.get("venue") or "binance").strip().lower()
        symbol = str(act.get("symbol") or act.get("bot") or act.get("target") or "").strip()
        order_id = str(act.get("order_id") or "").strip()
        reason = str(act.get("reason") or "")

        if action_type == "RESTART_BOT":
            if not symbol:
                logger.warning("[ENFORCE] RESTART_BOT missing bot name/symbol in action: %s", act)
                return False
            try:
                from orchestratorOS.livecheck.autonomous_overnight_monitor import AutonomousOvernightMonitor
                monitor = AutonomousOvernightMonitor(workspace_dir=self.workspace_dir)
                target_lower = symbol.lower()
                matched = next(
                    (
                        p for p in monitor.parse_process_manifest()
                        if p["name"].lower() == target_lower
                        or p["pattern"].rstrip("$").lower() == target_lower
                        or os.path.basename(p["pattern"].rstrip("$")).lower() == target_lower
                    ),
                    None,
                )
                if not matched:
                    logger.warning("[ENFORCE] Refusing RESTART_BOT for unknown process %r (not in procs.conf)", symbol)
                    return False
                pattern = matched["pattern"].rstrip("$")
                logger.warning("[ENFORCE EXECUTING] Restarting fleet bot %s (pattern=%s): %s", matched["name"], pattern, reason)
                subprocess.run(["pkill", "-f", pattern], check=False)
                return True
            except Exception as e:
                logger.error("[ENFORCE] Failed restarting fleet bot %s: %s", symbol, e)
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
        is_enforce = (self.mode == "enforce")
        raw_actions = plan.get("venue_actions")
        venue_actions = raw_actions if isinstance(raw_actions, list) else []
        raw_fix = plan.get("code_remediation") if allow_code_remediation else None
        code_fix = raw_fix if isinstance(raw_fix, dict) else {}

        venue_executed = False
        for act in venue_actions:
            if not isinstance(act, dict):
                continue
            action_type = str(act.get("action") or "NONE").strip().upper()
            if action_type == "NONE":
                continue
            if not is_enforce or not allow_code_remediation:
                logger.info("[SHADOW WOULD EXECUTE] Venue/fleet action %s for %s (%s): %s",
                            action_type, act.get("symbol"), act.get("venue"), act.get("reason"))
            else:
                if self._execute_venue_or_fleet_action(act):
                    venue_executed = True

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

        executed = is_enforce and bool(venue_executed or (code_fix.get("has_fix") and tests_passed))
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

    _BOT_AND_ORDER_ERROR_CATEGORIES = frozenset({
        "BOT_EXCEPTION",
        "ORPHAN_ORDER",
        "FLEET_STALE",
        "CODE_BUG",
    })
    _SYSTEM_OS_ERROR_CATEGORIES = frozenset({
        "SYSTEM_ERROR",
        "SERVER_ERROR",
        "OS_RESOURCE",
        "VPN_ERROR",
    })

    @classmethod
    def _is_bot_or_code_error_report(cls, report: SupervisorAuditReport) -> bool:
        """Return True strictly for trading bot exceptions, venue order anomalies, or Python code bugs/patches."""
        for issue in report.diagnosed_issues or []:
            if not isinstance(issue, dict):
                continue
            cat = str(issue.get("category") or "").strip().upper()
            sev = str(issue.get("severity") or "").strip().upper()
            if cat in cls._BOT_AND_ORDER_ERROR_CATEGORIES and sev in ("CRITICAL", "HIGH", "MEDIUM"):
                return True
        if isinstance(report.code_remediation, dict) and report.code_remediation.get("has_fix"):
            return True
        if report.tests_passed is False:
            return True
        for act in report.venue_actions or []:
            if isinstance(act, dict) and str(act.get("action") or "NONE").strip().upper() in (
                "CANCEL_ORDER", "SYNC_POSITION", "RESTART_BOT"
            ):
                return True
        return False

    @classmethod
    def _is_system_os_error_report(cls, report: SupervisorAuditReport) -> bool:
        """Return True for Linux OS, VPN, memory, disk space, or server infrastructure anomalies."""
        for issue in report.diagnosed_issues or []:
            if not isinstance(issue, dict):
                continue
            cat = str(issue.get("category") or "").strip().upper()
            sev = str(issue.get("severity") or "").strip().upper()
            if cat in cls._SYSTEM_OS_ERROR_CATEGORIES and sev in ("CRITICAL", "HIGH", "MEDIUM"):
                return True
            if sev in ("CRITICAL", "HIGH", "MEDIUM") and cat not in cls._BOT_AND_ORDER_ERROR_CATEGORIES:
                return True
        return False

    @classmethod
    def _is_error_report(cls, report: SupervisorAuditReport) -> bool:
        """Return True if the audit report diagnoses any bot/order/code error or system/OS error."""
        return cls._is_bot_or_code_error_report(report) or cls._is_system_os_error_report(report)

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
        has_fix = bool(isinstance(report.code_remediation, dict) and report.code_remediation.get("has_fix", False))
        if has_fix:
            actions_cnt += 1
        mode_label = report.mode.upper()
        is_bot_error = self._is_bot_or_code_error_report(report)
        is_sys_error = self._is_system_os_error_report(report)
        is_any_error = is_bot_error or is_sys_error

        spec = topic_spec(topic_key) if topic_key in TOPIC_SPECS else topic_spec("auto_technical_repair")
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
                m_conf = float(m_dec.get("confidence") or 0.0)
                m_usd = float(m_dec.get("suggested_notional_usd") or 0.0)
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
                "market_trading_decisions": report.market_trading_decisions,
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
        for dec in decisions_list:
            emitted = self.intent_gateway.submit(dec, source=topic_key, max_mode=self.mode)
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
        # (spec.can_self_heal); auto_trade_executor notifications are dispatched by AutonomousIntentGateway to ntfy-trade.
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
    parser = argparse.ArgumentParser(description="Autonomous AI supervisor (technical repair / trade executor)")
    parser.add_argument("--topic-key", required=True,
                        choices=sorted(k for k, v in TOPIC_SPECS.items() if v.can_trade or v.can_self_heal),
                        help="Permanent LLM topic thread to run")
    parser.add_argument("--mode", choices=["shadow", "enforce"], default=None,
                        help="Override supervisor_mode from order_guard.conf (bounds trading and self-heal)")
    parser.add_argument("--focus-symbols", default="",
                        help="Optional comma-separated priority watchlist (by default all live coins are evaluated)")
    parser.add_argument("--lookback-hours", type=float, default=24.0,
                        help="Log anomaly lookback window in hours (default: 24; 0.2 for 10-minute runs)")
    args = parser.parse_args()

    focus = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()] or None
    reconciler = AutonomousAIReconciler(mode=args.mode, lookback_hours=args.lookback_hours)
    report = reconciler.run_cycle(topic_key=args.topic_key, focus_symbols=focus)
    print(f"[{report.mode.upper()}] {args.topic_key}: {report.summary}")


if __name__ == "__main__":
    main()
