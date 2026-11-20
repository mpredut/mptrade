"""Global Telemetry Collector Daemon.

Continuously and asynchronously ingests high-frequency market telemetry into cachedb/:
- Pillar 2: Spot orderbook depth & imbalance, perpetual derivatives funding & OI, whale positioning.
- Pillar 3: Crypto Fear & Greed Index, Binance 24h market breadth.
- Pillar 4: Google News RSS raw geopolitical & macro headlines (pure ingestion, zero LLM).

Allows trading bots and order_guard.py to execute in sub-millisecond local non-blocking mode.
Strictly decoupled from LLM inference for guaranteed latency and resilience.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import (
    _parse_bool_flag,
    bool_env,
    load_dotenv,
    normalize_epoch_sec,
    parse_symbol_list,
    pos_float_env,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_nonneg_ts,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_REALIZED_LIQUIDATIONS_FILENAME,
    CACHE_SHORT_TREND_FILENAME,
    KINEMATIC_IMPULSE_SIGNAL_FILENAME,
    canonical_base_asset,
    clean_symbol_key,
    crypto_quote_priority_rank,
    has_crypto_quote_suffix,
    resolve_cachedb_filepath,
    resolve_workspace_dir_from_cachedb,
)
from intelligence.external.collectors.base_collector import collector_used_stale_fallback
from intelligence.external.collectors.bybit_liquidations_collector import (
    BinanceFuturesLiquidationCollector,
    BybitLiquidationCollector,
    LiquidationCollectorBase,
    OkxLiquidationCollector,
)
from intelligence.macro.daemon_ssot import (
    ExternalTelemetryMixin,
    add_external_telemetry_cli_args,
    detect_active_binance_symbols,
    finalize_daemon_cycle,
    normalize_cycle_args,
    parse_and_configure_daemon_cli,
    run_daemon_cli_once_or_start,
    run_daemon_loop,
    run_scheduled_step,
    write_daemon_heartbeat,
)
from intelligence.macro.news_feed_collector import NewsFeedCollector
from intelligence.macro_sentiment_ssot import (
    resolve_derived_telemetry_intervals,
    resolve_effective_macro_benchmarks_interval_sec,
)
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
from instrument_registry import (
    canonical_display_symbol,
    has_workspace_instrument_sources,
    is_equity_symbol,
    resolve_symbol_identity,
)
from providers.base import MarketDataProvider
from providers.nav_drawdown import record_nav_history_snapshot

load_tradable_instruments = MarketDataProvider.load_tradable_instruments
from intelligence.sentiment.collectors.macro_telemetry_collector import (
    InitialPricesAndTrendsCollector,
    collect_macro_benchmarks_snapshot,
)
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
from intelligence.sentiment.collectors.portfolio_collector import (
    VenueTrailingMapsCollector,
    harvest_tradable_positions_and_orders,
)
from intelligence.sentiment.collectors.pretrade_telemetry_collector import refresh_pretrade_telemetry_caches
from intelligence.sentiment.evaluators.microstructure_evaluator import update_microstructure_cycle_deltas
from state_io import open_append_text_file, read_json_dict_recovering, save_json_state

load_dotenv()

logger = logging.getLogger("intelligence.telemetry_collector")


def evaluate_kinematic_impulse_symbols(
    prices_and_trends: Dict[str, Any],
    candidate_symbols: Optional[List[str]] = None,
) -> List[str]:
    """Identify non-equity crypto symbols experiencing an acute symmetric short-horizon price/kinematic shock."""
    if not isinstance(prices_and_trends, dict) or not prices_and_trends:
        return []
    allowed_bases: Optional[set[str]] = None
    if candidate_symbols is not None:
        allowed_bases = {
            b
            for s in parse_symbol_list(candidate_symbols)
            if s and not is_equity_symbol(s) and (b := canonical_base_asset(s))
        }
        if not allowed_bases:
            return []
    best_by_base: Dict[str, tuple[int, str]] = {}
    for raw_sym, pinfo in prices_and_trends.items():
        if not isinstance(raw_sym, str) or not isinstance(pinfo, dict):
            continue
        sym = clean_symbol_key(raw_sym)
        if not sym or is_equity_symbol(sym) or not has_crypto_quote_suffix(sym):
            continue
        base = canonical_base_asset(sym)
        if not base or (allowed_bases is not None and base not in allowed_bases):
            continue
        c3m = safe_float(pinfo.get("chg_3m_pct"), default=None)
        c5m = safe_float(pinfo.get("chg_5m_pct"), default=None)
        v90 = safe_float(pinfo.get("vel_90s_pct_min"), default=None)
        a90 = safe_float(pinfo.get("accel_90s_pct_min2"), default=None)
        v5m = safe_float(pinfo.get("vel_5m_pct_min"), default=None)
        shock = (
            (c3m is not None and abs(c3m) >= 0.55)
            or (c5m is not None and abs(c5m) >= 0.70)
            or (
                v90 is not None
                and a90 is not None
                and abs(v90) >= 0.22
                and abs(a90) >= 0.14
                and (v90 * a90) > 0.0
            )
            or (v90 is not None and abs(v90) >= 0.35)
            or (
                v5m is not None
                and v90 is not None
                and abs(v5m) >= 0.18
                and abs(v90) >= 0.20
                and (v5m * v90) > 0.0
            )
        )
        if not shock:
            continue
        q_idx = crypto_quote_priority_rank(sym)
        prev = best_by_base.get(base)
        if prev is None or q_idx < prev[0]:
            best_by_base[base] = (q_idx, sym)
    return sorted(sym for _, sym in best_by_base.values())


class GlobalTelemetryCollector(ExternalTelemetryMixin):
    """Orchestrates periodic background ingestion of external microstructure and macro telemetry."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        cache_dir: str = "cachedb",
        orderbook_interval_sec: Optional[float] = None,
        derivatives_interval_sec: Optional[float] = None,
        whale_interval_sec: Optional[float] = None,
        news_interval_sec: Optional[float] = None,
        fear_greed_interval_sec: Optional[float] = None,
        breadth_interval_sec: Optional[float] = None,
        macro_benchmarks_interval_sec: Optional[float] = None,
        venue_balances_interval_sec: Optional[float] = None,
        skip_news: bool = False,
        skip_external: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        derived = resolve_derived_telemetry_intervals()
        self.crypto_executor_interval_sec = derived["crypto_executor_interval_sec"]
        self.ai_impulse_cooldown_sec = derived["ai_impulse_cooldown_sec"]
        self.ai_impulse_post_cron_guard_sec = derived["ai_impulse_post_cron_guard_sec"]
        self.ai_impulse_pre_cron_guard_sec = derived["ai_impulse_pre_cron_guard_sec"]
        self.ai_impulse_min_eval_gap_sec = derived["ai_impulse_min_eval_gap_sec"]
        self.ai_impulse_exec_timeout_sec = derived["ai_impulse_exec_timeout_sec"]
        default_news_itv = (
            derived["news_interval_sec"]
            if derived["crypto_executor_interval_sec"] != 360.0
            else pos_float_env("MACRO_NEWS_INTERVAL_SEC", derived["news_interval_sec"])
        )
        self.orderbook_interval_sec = safe_pos_float(orderbook_interval_sec, derived["orderbook_interval_sec"])
        self.derivatives_interval_sec = safe_pos_float(derivatives_interval_sec, derived["derivatives_interval_sec"])
        self.whale_interval_sec = safe_pos_float(whale_interval_sec, derived["whale_interval_sec"])
        self.news_interval_sec = safe_pos_float(news_interval_sec, default_news_itv)
        self.fear_greed_interval_sec = safe_pos_float(fear_greed_interval_sec, derived["fear_greed_interval_sec"])
        self.breadth_interval_sec = safe_pos_float(breadth_interval_sec, derived["breadth_interval_sec"])
        self._explicit_macro_benchmarks_interval = macro_benchmarks_interval_sec is not None
        self._default_macro_benchmarks_interval_sec = derived["macro_benchmarks_interval_sec"]
        self.macro_benchmarks_interval_sec = safe_pos_float(macro_benchmarks_interval_sec, derived["macro_benchmarks_interval_sec"])
        self.pretrade_caches_interval_sec = max(60.0, self.orderbook_interval_sec * 3.0)
        self.venue_balances_interval_sec = safe_pos_float(venue_balances_interval_sec, self.pretrade_caches_interval_sec)
        self.skip_news = _parse_bool_flag(skip_news, False)
        parsed_syms = list(dict.fromkeys(canonical_display_symbol(s) or s for s in parse_symbol_list(symbols)))
        self._explicit_symbols = bool(parsed_syms)
        self.symbols = parsed_syms or detect_active_binance_symbols(self.cache_dir, include_kraken_cache=True)
        self._auto_detected_symbols = list(self.symbols)

        self._init_external_collectors(logger, skip_external=skip_external)
        self.bybit_liq_collector: Optional[BybitLiquidationCollector] = None
        self.binance_liq_collector: Optional[BinanceFuturesLiquidationCollector] = None
        self.okx_liq_collector: Optional[OkxLiquidationCollector] = None
        self._last_realized_liq_ts: float = 0.0
        self.news_collector = NewsFeedCollector(
            cache_ttl_sec=max(30.0, self.news_interval_sec * 0.5),
            history_file=os.path.join(self.cache_dir, "news_feed_collect.json"),
            start_sync=True,
        )
        self.fear_greed_collector = FearGreedCollector(
            cache_ttl_sec=max(300.0, self.fear_greed_interval_sec * 0.8),
            start_sync=True,
        )
        self.breadth_collector = MarketBreadthCollector(
            cache_ttl_sec=max(60.0, self.breadth_interval_sec * 0.5),
            start_sync=True,
        )

        self._last_news_ts: float = 0.0
        self._last_fear_greed_ts: float = 0.0
        self._last_breadth_ts: float = 0.0
        self._last_macro_benchmarks_ts: float = 0.0
        self._last_pretrade_caches_ts: float = 0.0
        self._last_venue_balances_ts: float = 0.0
        self._last_ai_impulse_ts: float = 0.0
        self._last_ai_impulse_check_ts: float = 0.0
        self._live_impulse_wakeup_enabled: bool = False

        self.heartbeat_path = os.path.join(self.cache_dir, "globaltelemetry_collector.heartbeat")

    def _resolve_step_now(self, now: Optional[float] = None) -> Optional[float]:
        if now is not None:
            return safe_pos_float(now, default=None)
        return safe_pos_float(getattr(self, "_active_cycle_ts", None), default=None)

    def _fetch_fresh_collector_snapshot(
        self, collector: Any, *, force: bool = False, now: Optional[float] = None
    ) -> Tuple[Any, Optional[float]]:
        eff_now = self._resolve_step_now(now)
        snap = self._call_collector_fetch(
            collector,
            force_refresh=_parse_bool_flag(force, False),
            now_ts=eff_now,
        )
        if not snap or collector_used_stale_fallback(collector):
            return None, eff_now
        return snap, eff_now

    def update_news(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch breaking macro RSS headlines and update local rolling pool (zero LLM)."""
        try:
            news_snap, _ = self._fetch_fresh_collector_snapshot(self.news_collector, force=force, now=now)
            if news_snap:
                logger.debug("RSS news pool updated: %d headlines in pool", news_snap.total_fetched)
                return True
        except Exception as e:
            logger.warning("Failed updating RSS news feed: %s", e)
        return False

    def update_fear_greed(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch Crypto Fear & Greed Index and persist snapshot to cachedb/."""
        try:
            snap, eff_now = self._fetch_fresh_collector_snapshot(self.fear_greed_collector, force=force, now=now)
            if snap:
                now_val = resolve_now_ts(eff_now)
                idx_ts = safe_nonneg_ts(getattr(snap, "timestamp", None), now_val) or now_val
                fetch_ts = safe_pos_float(getattr(self.fear_greed_collector, "_last_fetch_ts", None), default=None) or now_val
                save_json_state(
                    os.path.join(self.cache_dir, "fear_greed_collect.json"),
                    {
                        "value": snap.value,
                        "sentiment": snap.classification,
                        "classification": snap.classification,
                        "historical_values": list(snap.historical_values),
                        "trend_7d_change": snap.trend_7d_change,
                        "is_extreme_fear": snap.is_extreme_fear,
                        "is_extreme_greed": snap.is_extreme_greed,
                        "index_timestamp": idx_ts,
                        "ts": fetch_ts,
                    },
                    indent=2,
                )
                logger.debug("Fear & Greed index updated: %d (%s)", snap.value, snap.classification)
                return True
        except Exception as e:
            logger.warning("Failed updating Fear & Greed index: %s", e)
        return False

    def update_market_breadth(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch Binance 24h market breadth and persist snapshot to cachedb/."""
        try:
            breadth, eff_now = self._fetch_fresh_collector_snapshot(self.breadth_collector, force=force, now=now)
            if breadth:
                now_val = resolve_now_ts(eff_now)
                raw_gainers = getattr(breadth, "top_gainers", None) or ()
                raw_losers = getattr(breadth, "top_losers", None) or ()
                tot = getattr(breadth, "total_count", 0)
                b_ts = safe_nonneg_ts(getattr(breadth, "ts", None), now_val) or now_val
                mb_payload: Dict[str, Any] = {
                    "advance_ratio": breadth.advance_ratio,
                    "median_change_pct": breadth.median_change_pct,
                    "mean_change_pct": breadth.mean_change_pct,
                    "dispersion_std": breadth.dispersion_std,
                    "regime": breadth.regime,
                    "total_symbols": tot,
                    "total_count": tot,
                    "advancing": breadth.advancing_count,
                    "advancing_count": breadth.advancing_count,
                    "declining": breadth.declining_count,
                    "declining_count": breadth.declining_count,
                    "top_gainers": [list(x) for x in raw_gainers if isinstance(x, (list, tuple))],
                    "top_losers": [list(x) for x in raw_losers if isinstance(x, (list, tuple))],
                    "ts": b_ts,
                }
                for rot_k in (
                    "btc_price", "btc_chg_24h_pct", "eth_price", "eth_chg_24h_pct",
                    "eth_btc_ratio", "eth_btc_chg_24h_pct", "btc_vol_share_pct",
                    "eth_vol_share_pct", "alt_vol_share_pct", "rotation_regime",
                ):
                    if (rot_v := getattr(breadth, rot_k, None)) is not None:
                        mb_payload[rot_k] = rot_v
                save_json_state(os.path.join(self.cache_dir, "market_breadth_collect.json"), mb_payload, indent=2)
                logger.debug("Market breadth updated: %.1f%% advancing", breadth.advance_ratio * 100)
                return True
        except Exception as e:
            logger.warning("Failed updating market breadth: %s", e)
        return False

    def _effective_macro_benchmarks_interval_sec(self, now_ts: Optional[float] = None) -> float:
        """Use faster 45s macro-benchmark cadence during US cash open (09:25-10:15 NY) unless explicitly overridden."""
        base = safe_pos_float(getattr(self, "macro_benchmarks_interval_sec", None), 300.0) or 300.0
        if getattr(self, "_explicit_macro_benchmarks_interval", False):
            return base
        default_itv = safe_pos_float(getattr(self, "_default_macro_benchmarks_interval_sec", None), None)
        if default_itv is not None and abs(base - default_itv) > 1e-6:
            return base
        return resolve_effective_macro_benchmarks_interval_sec(base, now_ts=now_ts)

    def _realized_liquidation_perp_symbols(self) -> List[str]:
        perp_syms: List[str] = []
        seen: set[str] = set()
        for raw_s in (*self.symbols, "BTCUSDT", "ETHUSDT", "SOLUSDT"):
            ident = resolve_symbol_identity(raw_s)
            if not ident.has_crypto_perpetuals:
                continue
            base = canonical_base_asset(ident.wire_symbol)
            if base and base not in seen:
                seen.add(base)
                perp_syms.append(f"{base}USDT")
        return perp_syms

    def _active_liquidation_collectors(self) -> List[LiquidationCollectorBase]:
        cols: List[LiquidationCollectorBase] = []
        for attr in ("bybit_liq_collector", "binance_liq_collector", "okx_liq_collector"):
            col = getattr(self, attr, None)
            if col is not None:
                cols.append(col)
        return cols

    def update_realized_liquidations(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Persist rolling 15m multi-venue realized forced liquidations snapshot when WS stream(s) are active."""
        _ = force
        active_cols = self._active_liquidation_collectors()
        if not active_cols:
            return False
        try:
            eff_now = resolve_now_ts(self._resolve_step_now(now))
            sources = [str(getattr(c, "VENUE_SOURCE", "") or "bybit_ws") for c in active_cols]
            connected_sources = [
                str(getattr(c, "VENUE_SOURCE", "") or "bybit_ws")
                for c in active_cols
                if bool(getattr(c, "is_running", False))
            ]
            items: Dict[str, Dict[str, Any]] = {}
            for psym in self._realized_liquidation_perp_symbols():
                long_tot = short_tot = last_ts = 0.0
                events_tot = 0
                by_venue: Dict[str, Dict[str, Any]] = {}
                for col, src_lbl in zip(active_cols, sources):
                    summ = col.get_summary(psym, window_sec=900.0, now=eff_now)
                    l_u = round(summ.long_liq_usd, 0)
                    s_u = round(summ.short_liq_usd, 0)
                    ev_n = safe_int(summ.events_count, 0)
                    long_tot += l_u
                    short_tot += s_u
                    events_tot += ev_n
                    if summ.last_event_ts:
                        last_ts = max(last_ts, summ.last_event_ts)
                    if len(active_cols) > 1 and (ev_n > 0 or l_u > 0.0 or s_u > 0.0):
                        by_venue[src_lbl] = {
                            "long_15m_usd": l_u,
                            "short_15m_usd": s_u,
                            "events_15m": ev_n,
                        }
                entry: Dict[str, Any] = {
                    "long_15m_usd": round(long_tot, 0),
                    "short_15m_usd": round(short_tot, 0),
                    "net_15m_usd": round(short_tot - long_tot, 0),
                    "events_15m": events_tot,
                    "last_event_ts": round(last_ts, 3) if last_ts else 0.0,
                }
                if by_venue:
                    entry["by_venue"] = by_venue
                items[psym] = entry
            payload = {
                "ts": eff_now,
                "source": sources[0] if len(sources) == 1 else "multi_venue_ws",
                "sources": sources,
                "connected_sources": connected_sources,
                "connected": bool(connected_sources),
                "symbols": items,
            }
            out_path = str(resolve_cachedb_filepath(os.path.join(self.cache_dir, CACHE_REALIZED_LIQUIDATIONS_FILENAME)))
            save_json_state(out_path, payload, indent=2)
            return True
        except Exception as e:
            logger.debug("Failed persisting realized liquidations snapshot: %s", e)
            return False

    def update_macro_benchmarks(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch SPY, QQQ, ES=F, NQ=F, DXY, and US10Y macro benchmarks and persist to cachedb/."""
        try:
            eff_now = self._resolve_step_now(now)
            snap = collect_macro_benchmarks_snapshot(self.cache_dir, timeout_sec=3.5, persist=True, now_ts=eff_now)
            if isinstance(snap, dict) and snap.get("items"):
                logger.debug("Macro benchmarks updated: %s", snap.get("summary_line", ""))
                return True
        except Exception as e:
            logger.warning("Failed updating macro benchmarks: %s", e)
        return False

    def update_pretrade_caches(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Producer refresh of pre-trade caches (per-symbol microstructure and once-per-round T212 position prices)."""
        try:
            eff_now = resolve_now_ts(self._resolve_step_now(now))
            any_refreshed = False
            for sym in list(self.symbols):
                status = refresh_pretrade_telemetry_caches(
                    sym,
                    cachedb_dir=self.cache_dir,
                    now_ts=eff_now,
                    include_t212_portfolio=False,
                )
                if any(status.values()):
                    any_refreshed = True
            portfolio_status = refresh_pretrade_telemetry_caches(
                "",
                cachedb_dir=self.cache_dir,
                now_ts=eff_now,
                include_t212_portfolio=True,
            )
            if any(portfolio_status.values()):
                any_refreshed = True
            self._last_pretrade_caches_ts_noop = not any_refreshed
            return any_refreshed
        except Exception as e:
            logger.warning("Failed refreshing pre-trade caches: %s", e)
        return False

    def update_venue_balances(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Producer refresh of multi-venue spot balances, resting open orders (`venue_balances_state.json`), and cycle deltas."""
        try:
            eff_now = resolve_now_ts(self._resolve_step_now(now))
            workspace_dir = resolve_workspace_dir_from_cachedb(self.cache_dir)
            has_inst_source = has_workspace_instrument_sources(workspace_dir, asset_scope="all")
            inst_map: Dict[str, Any] = {}
            if has_inst_source:
                inst_map = load_tradable_instruments(
                    workspace_dir,
                    only_workspace_conf=True,
                    include_closed_rth=False,
                    now_ts=eff_now,
                    asset_scope="all",
                )
            prices_and_trends = InitialPricesAndTrendsCollector(start_sync=False).collect(self.cache_dir)
            if inst_map:
                trailing_map, trailing_by_venue = VenueTrailingMapsCollector(start_sync=False).collect(
                    self.cache_dir, "*", workspace_dir=workspace_dir
                )
                gw_max_usd = pos_float_env("AUTONOMOUS_TRADING_MAX_NOTIONAL_USD", 2000.0)
                harvest_tradable_positions_and_orders(
                    inst_map,
                    prices_and_trends,
                    trailing_map,
                    trailing_by_venue,
                    gateway_max_usd=gw_max_usd,
                    cache_dir=self.cache_dir,
                    allow_network=True,
                    persist=True,
                    force_persist=True,
                )
                try:
                    record_nav_history_snapshot(workspace_dir, now_ts=eff_now)
                except Exception as nav_exc:
                    logger.debug("Skipped NAV history snapshot update: %s", nav_exc)
            micro_map: Dict[str, Any] = {}
            enrich_24h_and_microstructure(
                self.cache_dir,
                workspace_dir,
                prices_and_trends,
                micro_map,
                {},
                {},
                eff_now,
                allow_network=False,
                persist=False,
            )
            if micro_map:
                update_microstructure_cycle_deltas(self.cache_dir, micro_map, now_ts=eff_now, persist=True)
            published = bool(inst_map or micro_map)
            self._last_venue_balances_ts_noop = not published
            return published
        except Exception as e:
            logger.warning("Failed refreshing venue balances state: %s", e)
        return False

    def maybe_trigger_kinematic_ai_wakeup(self, now: Optional[float] = None) -> Optional[List[str]]:
        """Spawn a non-blocking flock-guarded crypto AI evaluation on acute mid-cycle kinematic price shocks."""
        try:
            if not bool_env("AI_IMPULSE_TRIGGER_ENABLED", True):
                return None
            eff_now = resolve_now_ts(self._resolve_step_now(now))
            cycle_sec = safe_pos_float(getattr(self, "crypto_executor_interval_sec", None), 360.0)
            cooldown_sec = pos_float_env(
                "AI_IMPULSE_COOLDOWN_SEC",
                safe_pos_float(getattr(self, "ai_impulse_cooldown_sec", None), 150.0),
            )
            if self._last_ai_impulse_ts > 0.0 and 0.0 <= (eff_now - self._last_ai_impulse_ts) < cooldown_sec:
                return None
            post_cron_guard_sec = safe_pos_float(getattr(self, "ai_impulse_post_cron_guard_sec", None), 20.0)
            pre_cron_guard_sec = safe_pos_float(getattr(self, "ai_impulse_pre_cron_guard_sec", None), 45.0)
            cycle_offset = eff_now % cycle_sec
            if cycle_offset < post_cron_guard_sec or (cycle_sec - cycle_offset) < pre_cron_guard_sec:
                return None
            eval_path = str(resolve_cachedb_filepath(os.path.join(self.cache_dir, "auto_cryptotrade_executor_eval.json")))
            last_eval_ts = 0.0
            if os.path.exists(eval_path):
                try:
                    mtime_val = safe_nonneg_ts(os.path.getmtime(eval_path), 0.0)
                    if mtime_val <= eff_now + 5.0:
                        last_eval_ts = max(last_eval_ts, mtime_val)
                except OSError:
                    pass
                eval_doc = read_json_dict_recovering(eval_path)
                if isinstance(eval_doc, dict):
                    raw_doc_ts = eval_doc.get("ts") if eval_doc.get("ts") is not None else eval_doc.get("evaluated_at_ts")
                    doc_ts = normalize_epoch_sec(raw_doc_ts, default=0.0, allow_iso=True) or 0.0
                    if 0.0 < doc_ts <= eff_now + 5.0:
                        last_eval_ts = max(last_eval_ts, doc_ts)
            min_eval_gap_sec = safe_pos_float(getattr(self, "ai_impulse_min_eval_gap_sec", None), 65.0)
            if last_eval_ts > 0.0 and 0.0 <= (eff_now - last_eval_ts) < min_eval_gap_sec:
                return None
            st_doc = read_json_dict_recovering(
                str(resolve_cachedb_filepath(os.path.join(self.cache_dir, CACHE_SHORT_TREND_FILENAME)))
            )
            cp_doc = read_json_dict_recovering(
                str(resolve_cachedb_filepath(os.path.join(self.cache_dir, CACHE_CURRENT_PRICE_FILENAME)))
            )
            prices_and_trends = InitialPricesAndTrendsCollector(start_sync=False).collect(
                self.cache_dir,
                prices_data=cp_doc,
                trends_data=st_doc,
            )
            cand_syms = self.symbols if (self.symbols or getattr(self, "_explicit_symbols", False)) else None
            shocked = evaluate_kinematic_impulse_symbols(prices_and_trends, candidate_symbols=cand_syms)
            if not shocked:
                return None
            workspace_dir = resolve_workspace_dir_from_cachedb(self.cache_dir)
            runner_sh = os.path.join(workspace_dir, "orchestratorOS", "run_python.sh")
            flock_bin = "/usr/bin/flock"
            timeout_bin = "/usr/bin/timeout"
            if not (os.path.isfile(runner_sh) and os.path.isfile(flock_bin) and os.path.isfile(timeout_bin)):
                return None
            focus_csv = ",".join(shocked)
            save_json_state(
                str(resolve_cachedb_filepath(os.path.join(self.cache_dir, KINEMATIC_IMPULSE_SIGNAL_FILENAME))),
                {"ts": eff_now, "symbols": shocked},
                indent=2,
            )
            exec_timeout_sec = str(
                max(30, safe_pos_int(round(safe_pos_float(getattr(self, "ai_impulse_exec_timeout_sec", None), 270.0)), default=270))
            )
            log_path = os.path.join(workspace_dir, "logs", "auto_trade_executor.log")
            out_fh = open_append_text_file(log_path)
            try:
                subprocess.Popen(
                    [
                        flock_bin,
                        "-n",
                        "logs/.auto_cryptotrade_executor.lock",
                        timeout_bin,
                        exec_timeout_sec,
                        "./orchestratorOS/run_python.sh",
                        "orchestratorOS/supervisor/autonomous_ai_reconciler.py",
                        "--topic-key",
                        "auto_cryptotrade_executor",
                        "--focus-symbols",
                        focus_csv,
                    ],
                    cwd=workspace_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=out_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                out_fh.close()
            self._last_ai_impulse_ts = eff_now
            logger.info("[IMPULSE WAKEUP] Triggered auto_cryptotrade_executor for kinematic shock on %s", focus_csv)
            return shocked
        except Exception as e:
            logger.debug("Kinematic impulse wakeup check skipped: %s", e)
            return None

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, updating telemetry whose scheduled interval has elapsed."""
        force, current_ts = normalize_cycle_args(force, now)
        if not getattr(self, "_explicit_symbols", False) and self.symbols == getattr(self, "_auto_detected_symbols", None):
            if detected_syms := detect_active_binance_symbols(self.cache_dir, include_kraken_cache=True):
                self.symbols = detected_syms
                self._auto_detected_symbols = list(detected_syms)
        results = self._new_external_cycle_results(
            current_ts,
            news_updated=False,
            fear_greed_updated=False,
            breadth_updated=False,
            macro_benchmarks_updated=False,
            pretrade_caches_updated=False,
            venue_balances_updated=False,
        )

        self.write_heartbeat(results)
        self._active_cycle_ts = current_ts
        try:
            self._step_external_symbols(
                current_ts, force, results, bounded=True, on_symbol_done=lambda: self.write_heartbeat(results)
            )

            eff_mb_itv = self._effective_macro_benchmarks_interval_sec(current_ts)
            scheduled_tasks = (
                (not self.skip_news, "_last_news_ts", self.news_interval_sec, self.update_news, (force,), 15.0, "news", "news_updated", lambda: self.write_heartbeat(results)),
                (True, "_last_fear_greed_ts", self.fear_greed_interval_sec, self.update_fear_greed, (force,), 12.0, "fear_greed", "fear_greed_updated", None),
                (True, "_last_breadth_ts", self.breadth_interval_sec, self.update_market_breadth, (force,), 15.0, "breadth", "breadth_updated", None),
                (not self.skip_external, "_last_macro_benchmarks_ts", eff_mb_itv, self.update_macro_benchmarks, (force,), 15.0, "macro_benchmarks", "macro_benchmarks_updated", None),
                (not self.skip_external and bool(self._active_liquidation_collectors()), "_last_realized_liq_ts", getattr(self, "derivatives_interval_sec", 30.0), self.update_realized_liquidations, (force,), 5.0, "realized_liquidations", "realized_liquidations_updated", None),
                (not self.skip_external, "_last_pretrade_caches_ts", self.pretrade_caches_interval_sec, self.update_pretrade_caches, (force,), 24.0, "pretrade_caches", "pretrade_caches_updated", None),
                (not self.skip_external, "_last_venue_balances_ts", self.venue_balances_interval_sec, self.update_venue_balances, (force,), 18.0, "venue_balances", "venue_balances_updated", None),
            )
            for enabled, ts_attr, itv_sec, fn, fn_args, timeout_s, lbl, res_key, on_done_cb in scheduled_tasks:
                if enabled:
                    run_scheduled_step(
                        self,
                        current_ts=current_ts,
                        ts_attr=ts_attr,
                        interval_sec=itv_sec,
                        fn=fn,
                        fn_args=fn_args,
                        force=force,
                        bounded=True,
                        timeout_sec=timeout_s,
                        label=lbl,
                        results=results,
                        result_key=res_key,
                        on_done=on_done_cb,
                    )
            last_imp_chk = safe_nonneg_ts(getattr(self, "_last_ai_impulse_check_ts", 0.0), 0.0)
            if (
                not self.skip_external
                and _parse_bool_flag(getattr(self, "_live_impulse_wakeup_enabled", False), False)
                and (last_imp_chk <= 0.0 or not (0.0 < (current_ts - last_imp_chk) < 5.0))
            ):
                self._last_ai_impulse_check_ts = current_ts
                self.maybe_trigger_kinematic_ai_wakeup(now=current_ts)
        finally:
            self._active_cycle_ts = None

        return finalize_daemon_cycle(
            self,
            results,
            current_ts=current_ts,
            force=force,
            logger_inst=logger,
            hb_msg="Global telemetry cycle active: %d symbols monitored (PID=%d)",
            hb_args=(len(self.symbols), os.getpid()),
        )

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb, with fallback to direct touch/write."""
        write_daemon_heartbeat(
            self.heartbeat_path,
            status,
            symbols=self.symbols,
            intervals={
                "orderbook": self.orderbook_interval_sec,
                "derivatives": self.derivatives_interval_sec,
                "whale": self.whale_interval_sec,
                "news": self.news_interval_sec,
                "fear_greed": self.fear_greed_interval_sec,
                "breadth": self.breadth_interval_sec,
                "macro_benchmarks": getattr(self, "macro_benchmarks_interval_sec", None),
                "venue_balances": getattr(self, "venue_balances_interval_sec", None),
            },
            logger_inst=logger,
        )

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main telemetry collection loop continuously."""
        self._live_impulse_wakeup_enabled = bool_env("AI_IMPULSE_TRIGGER_ENABLED", True)
        if not self.skip_external and bool_env("REALIZED_LIQUIDATIONS_WS_ENABLED", True):
            perp_syms = self._realized_liquidation_perp_symbols()
            if perp_syms:
                for attr, env_k, cls in (
                    ("bybit_liq_collector", "BYBIT_LIQUIDATIONS_WS_ENABLED", BybitLiquidationCollector),
                    ("binance_liq_collector", "BINANCE_LIQUIDATIONS_WS_ENABLED", BinanceFuturesLiquidationCollector),
                    ("okx_liq_collector", "OKX_LIQUIDATIONS_WS_ENABLED", OkxLiquidationCollector),
                ):
                    if bool_env(env_k, True) and getattr(self, attr, None) is None:
                        try:
                            setattr(self, attr, cls(symbols=perp_syms, start_sync=False).start())
                        except Exception as exc:
                            logger.debug("Could not start %s WS: %s", cls.cls_name, exc)
        try:
            run_daemon_loop(
                self,
                poll_interval_sec=poll_interval_sec,
                default_poll_sec=1.0,
                daemon_name="GlobalTelemetryCollector",
                stop_label="collector",
                cycle_label="telemetry",
                start_msg=f"Starting GlobalTelemetryCollector for symbols: {', '.join(self.symbols)} (PID={os.getpid()})",
                logger_inst=logger,
                initial_force=True,
                initial_heartbeat={"ts": time.time(), "symbols": list(self.symbols)},
                reraise_initial_exit=True,
            )
        finally:
            for col in self._active_liquidation_collectors():
                try:
                    col.stop(timeout=3.0)
                except Exception:
                    pass


def main():
    """Command-line entrypoint for the global telemetry collector."""
    parser = argparse.ArgumentParser(description="Global Market Telemetry Collector Daemon")
    add_external_telemetry_cli_args(
        parser,
        skip_flag="--skip-news",
        skip_help="Skip RSS macro news collection.",
        extra_float_flags=(
            ("--news-interval", "Macro news RSS aggregation interval in seconds."),
        ),
    )
    args = parse_and_configure_daemon_cli(parser)

    collector = GlobalTelemetryCollector(
        symbols=parse_symbol_list(args.symbols) or None,
        orderbook_interval_sec=args.ob_interval,
        derivatives_interval_sec=args.deriv_interval,
        whale_interval_sec=args.whale_interval,
        news_interval_sec=args.news_interval,
        skip_news=args.skip_news,
        skip_external=args.skip_external,
    )
    run_daemon_cli_once_or_start(
        collector, once=args.once, force=True, logger_inst=logger
    )


if __name__ == "__main__":
    main()
