"""Macro benchmarks, breaking news sensors, snapshot orchestrator, and pre-trade LLM telemetry collector."""
from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta, timezone
import glob as _glob
import logging
import math
import os
import time as _time
from typing import Any, Callable, Dict, List, Optional

from botcore import (
    _parse_bool_flag,
    coerce_dict_maps,
    extract_current_price_item,
    extract_current_price_timestamp_sec,
    extract_timestamp_s,
    fetch_public_json,
    fetch_public_text,
    first_epoch_sec,
    first_pos_float,
    is_live_network_backfill_enabled,
    is_us_rth_open,
    load_key_value_conf,
    normalize_epoch_sec,
    resolve_now_ts,
    resolve_patched_attr,
    resolve_remaining_timeout_sec,
    resolve_us_equity_session_state,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    CACHE_ETF_FLOWS_FILENAME,
    CACHE_MACRO_BENCHMARKS_FILENAME,
    CACHE_MACRO_CALENDAR_FILENAME,
    CACHE_MARKET_BREADTH_FILENAME,
    CACHE_NEWS_FEED_FILENAME,
    CACHE_WATCHER_ALL_PRICE_FILENAME,
    cache_symbol_variants,
    canonical_base_asset,
    has_equity_suffix,
    is_equity_venue,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from providers.portfolio_state_store import resolve_venue_balance_cache_path
from providers.venue_fee_maintenance import (
    build_persisted_venue_fee_resolver,
    read_persisted_venue_fee_schedule,
)
from state_stores.cache_T_trend import CacheTTrendStore
from intelligence.internal.state import survival as _surv_mod
from intelligence.macro.calendar_ssot import resolve_macro_economic_calendar
from intelligence.macro.lexicon_ssot import (
    _strip_headline_publisher_suffix,
    filter_valid_severity_keywords,
    has_financial_or_geopolitical_context,
    has_macro_policy_context,
    has_tracked_equity_corporate_risk,
    has_tracked_equity_entity,
    has_tracked_tokenized_equity_corporate_risk,
    has_tracked_tokenized_equity_entity,
    headlines_are_semantic_duplicates,
    is_excluded_reference_news_item,
    is_off_topic_general_news_title,
)
import intelligence.macro_sentiment_ssot as _mss
from intelligence.macro_sentiment_ssot import (
    CRYPTO_ONLY_MACRO_SENSOR_KEYS,
    MACRO_SENSOR_CACHE_SPECS,
    SYNTHETIC_MACRO_SENSOR_KEYS,
    get_breadth_excluded_bases,
    get_deribit_options_currencies,
    get_earnings_bellwether_tickers,
    get_macro_calendar_countries,
    get_macro_calendar_endpoint_url,
    get_macro_calendar_medium_tier_keywords,
    get_nasdaq_earnings_calendar_url,
    get_spot_etf_flow_urls,
    get_tracked_equity_base_tickers,
    get_yahoo_chart_url,
    load_microstructure_cycle_history,
    load_venue_balances_state,
    read_first_cached_dict,
    resolve_macro_fresh_guard_sec,
    resolve_read_json_cached,
    save_microstructure_cycle_history,
    serialize_market_breadth_snapshot,
    split_headlines_digest,
    TelemetrySensorSnapshot,
    _cfg_str,
)
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
)
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
import intelligence.sentiment.collectors.microstructure_collector as _mc
from intelligence.sentiment.collectors.long_horizon_collector import load_active_operator_regime_note
from intelligence.sentiment.collectors.microstructure_collector import (
    enrich_24h_and_microstructure,
    resolve_cached_btc_24h_change_pct,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    VenueTrailingMapsCollector,
    _resolve_ai_intents_and_known_order_ids,
    harvest_bot_runtime_states,
    harvest_kraken_xstocks,
    harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders,
    harvest_venue_cycles,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    apply_asset_scope_filter,
    build_cycle_fallback_prices,
    compute_cross_asset_regime,
    compute_deribit_options_surface,
    compute_equity_macro_regime,
    compute_futures_premarket_lead,
    enrich_macro_benchmarks_live_state,
    enrich_sentiment_advisor_live_flow_divergence,
    format_etf_flows_summary_line,
    format_macro_benchmarks_summary_line,
    MACRO_BENCHMARK_SPECS,
    parse_farside_etf_flows_html,
    reconcile_live_venue_equity_summary,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    enrich_cross_venue_trends,
    sync_sibling_symbol_telemetry,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    build_cross_market_context,
    compute_enterprise_portfolio_risk_metrics,
    enrich_open_limit_orders_tca,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    apply_live_24trend_windows,
    apply_short_trend_item,
)
from instrument_registry import (
    canonical_display_symbol,
    casefold_get,
    disabled_registry_symbols,
    has_workspace_instrument_sources,
)
from cacheprice import resolve_ssot_price_from_cache
from providers.ai_circuit_breaker import parse_intent_circuit_breaker_config
from providers.ai_survival_guard import (
    build_intent_live_price_map,
    build_persisted_round_trip_fee_resolver,
    evaluate_ai_calibration_feedback,
    evaluate_ai_decision_scorecard,
)
from providers.nav_history_store import load_and_evaluate_nav_drawdown
from providers.intent_telemetry import (
    build_last_cycle_execution_status,
    summarize_active_ai_trade_plans,
    summarize_recent_ai_intents,
)
from providers.intent_state import (
    collect_known_ai_order_ids,
    extract_intent_notional,
    parse_intent_gateway_config,
    prune_reconciled_terminal_open_orders,
)
from providers.ohlc_utils import (
    extract_yahoo_chart_ohlcv_bars,
    resolve_yahoo_chart_market_state_and_price,
)
from state_io import (
    read_json_dict_recovering as _default_read_json_cached,
    read_json_recovering,
    read_last_jsonl_dict,
    save_json_state as _default_save_json_state,
)

read_json_cached = _default_read_json_cached
save_json_state = _default_save_json_state
_default_fetch_public_json = fetch_public_json
_default_fetch_public_text = fetch_public_text

logger = logging.getLogger("intelligence.sentiment.macro_telemetry_collector")


def _resolve_macro_ssot_attr(name: str, local_val: Any, default_val: Any) -> Any:
    return resolve_patched_attr(
        name,
        default_val,
        local_val=local_val,
        defaults=(default_val,),
        modules=("intelligence.macro_sentiment_ssot",),
        require_callable=True,
    )


def _resolve_read_json_cached(read_json_fn: Any = None, *, default_fn: Any = None) -> Any:
    return resolve_read_json_cached(read_json_fn, local_fn=read_json_cached, default_fn=default_fn)


def _read_first_cached_dict(
    cache_dir: str, group_key: str, default_file: str, read_json_fn: Any = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Read the first non-empty dict matching `get_cache_filenames(group_key)` (or `default_file`) in `cache_dir`."""
    eff_rj = _resolve_read_json_cached(read_json_fn)
    return read_first_cached_dict(cache_dir, group_key, default_file, read_json_fn=eff_rj)


def _harvest_initial_prices_and_trends_impl(
    cache_dir: str, prices_data: Any, trends_data: Any, read_json_fn: Any = None, *, workspace_dir: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Build initial prices_and_trends map from cache_current_price, cache_short_trend, and 24trend/live_*.json."""
    _rj = read_json_fn or read_json_cached
    eff_ws = workspace_dir or (os.path.dirname(os.path.abspath(str(cache_dir).rstrip("/\\"))) if cache_dir else "")
    disabled_syms = disabled_registry_symbols(eff_ws) if eff_ws else frozenset()

    def _is_disabled(raw_s: Any) -> bool:
        if not disabled_syms:
            return False
        s_u = str(raw_s or "").strip().upper()
        return bool(s_u and s_u in disabled_syms)

    prices_and_trends: Dict[str, Dict[str, Any]] = {}
    price_map: Dict[str, float] = {
        sym: px_val for sym, item in (prices_data.get("items", {}) if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict) else {}).items()
        if not _is_disabled(sym) and (px_val := extract_current_price_item(item)) is not None
    }
    if isinstance(trends_data, dict):
        trends_items = trends_data.get("items") if isinstance(trends_data.get("items"), dict) else trends_data
        for sym, tinfo in trends_items.items():
            if sym not in ("items", "fetchtime", "rest_sync_cursor", "metadata", "counts") and isinstance(tinfo, dict) and not _is_disabled(sym):
                entry: Dict[str, Any] = {}
                apply_short_trend_item(entry, tinfo, fallback_price=price_map.get(sym))
                prices_and_trends[sym] = entry

    for sym, px in price_map.items():
        prices_and_trends.setdefault(sym, {"price": px, "trend": 0, "growth_coefficient": 0.0})

    _mc.ensure_cacheprice_classes_bound()
    st_cls = _mc.CacheShortTrendManager
    lw_subdir = st_cls.LIVE_TREND_SUBDIR
    lw_prefix = st_cls.LIVE_FILE_PREFIX
    lw_ext = f".{st_cls.LIVE_FILE_EXT}"
    if os.path.isdir(live_24trend_dir := str(resolve_cachedb_filepath(os.path.join(cache_dir, lw_subdir)))):
        for lt_path in sorted(_glob.glob(os.path.join(live_24trend_dir, f"{lw_prefix}*{lw_ext}"))):
            raw_sym = os.path.basename(lt_path)[len(lw_prefix) : -len(lw_ext)]
            sym = canonical_display_symbol(raw_sym) or raw_sym.upper()
            if sym and not _is_disabled(sym) and not _is_disabled(raw_sym) and isinstance(lt_win := _rj(lt_path), dict):
                apply_live_24trend_windows(prices_and_trends.setdefault(sym, {}), lt_win, fallback_price=price_map.get(sym))
    return prices_and_trends


class MacroTelemetryCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for macro benchmarks, calendar/news sensors, and unified snapshot orchestrators."""

    cls_name = "MacroTelemetryCollectorBase"
    _FETCH_DEADLINE_SEC: float = 0.0

    @staticmethod
    def _resolve_fetch_fn(fetch_fn: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        """Resolve public JSON fetcher honoring module monkeypatches."""
        return fetch_fn or _resolve_macro_ssot_attr("fetch_public_json", fetch_public_json, _default_fetch_public_json)

    @staticmethod
    def _resolve_text_fetch_fn(fetch_text_fn: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        """Resolve public text/HTML/CSV fetcher honoring module monkeypatches."""
        return fetch_text_fn or _resolve_macro_ssot_attr(
            "fetch_public_text", fetch_public_text, _default_fetch_public_text
        )

    @classmethod
    def _build_fetch_deadline(cls, fetch_fn: Optional[Callable[..., Any]] = None) -> Optional[float]:
        """Return monotonic wall-clock deadline when using live network fetches and ``_FETCH_DEADLINE_SEC > 0``."""
        budget = safe_nonneg_float(getattr(cls, "_FETCH_DEADLINE_SEC", 0.0), 0.0) or 0.0
        return (_time.monotonic() + budget) if (fetch_fn is None and budget > 0.0) else None

    @staticmethod
    def _resolve_cache_path(cache_dir: str, filename: str) -> str:
        """Resolve the canonical cachedb subdomain path for ``filename`` inside ``cache_dir``."""
        return str(resolve_cachedb_filepath(os.path.join(cache_dir, filename)))

    @staticmethod
    def _persist_cache_json(path: str, payload: Any, *, indent: int = 2) -> None:
        """Persist a macro cache artifact atomically honoring ``save_json_state`` monkeypatches."""
        _save_fn = _resolve_macro_ssot_attr("save_json_state", save_json_state, _default_save_json_state)
        _save_fn(path, payload, indent=indent)

    @classmethod
    def _persist_snapshot(
        cls,
        cache_dir: str,
        filename: str,
        payload: Any,
        *,
        persist: bool = True,
        indent: int = 2,
    ) -> bool:
        """Safely persist ``payload`` to ``filename`` under ``cache_dir`` when ``persist`` is enabled."""
        if not persist or not cache_dir:
            return False
        try:
            cls._persist_cache_json(cls._resolve_cache_path(cache_dir, filename), payload, indent=indent)
            return True
        except Exception:
            return False

    @staticmethod
    def _read_cached_dict(
        cache_dir: str,
        group_key: str,
        default_file: str,
        read_json_fn: Any = None,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        """Read the first non-empty dict matching configured cache filenames for ``group_key``."""
        return _read_first_cached_dict(cache_dir, group_key, default_file, read_json_fn)

    @staticmethod
    def _is_snapshot_within_age(
        snap: Any,
        *,
        now_ts: float,
        max_age_sec: float,
        inclusive: bool = False,
    ) -> bool:
        """Return True when ``snap['ts']`` is valid and within ``max_age_sec`` of ``now_ts``."""
        limit = safe_nonneg_float(max_age_sec, 0.0) or 0.0
        if limit <= 0.0 or not isinstance(snap, dict):
            return False
        prev_ts = normalize_epoch_sec(snap.get("ts"), default=0.0) or 0.0
        if prev_ts <= 0.0:
            return False
        age = now_ts - prev_ts
        return bool(0.0 <= age <= limit) if inclusive else bool(0.0 <= age < limit)

    def _read_fresh_cached_snapshot(
        self,
        cache_dir: str,
        group_key: str,
        default_file: str,
        *,
        now_ts: float,
        skip_if_fresh_sec: float = 0.0,
        read_json_fn: Any = None,
        is_healthy_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """Read cached snapshot and indicate whether it satisfies ``skip_if_fresh_sec`` and ``is_healthy_fn``."""
        prev_snap, _ = self._read_cached_dict(cache_dir, group_key, default_file, read_json_fn=read_json_fn)
        fresh_guard_sec = safe_nonneg_float(skip_if_fresh_sec, 0.0) or 0.0
        if fresh_guard_sec <= 0.0 or not isinstance(prev_snap, dict):
            return prev_snap, False
        if callable(is_healthy_fn) and not is_healthy_fn(prev_snap):
            return prev_snap, False
        return prev_snap, self._is_snapshot_within_age(prev_snap, now_ts=now_ts, max_age_sec=fresh_guard_sec)


class MacroBenchmarksSnapshotCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) fetching SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y/IBIT/Deribit DVOL benchmarks."""

    cls_name = "MacroBenchmarksSnapshotCollector"
    COLLECTOR_ID = "macro_benchmarks_snapshot"
    # Shared wall-clock budget strictly below GlobalTelemetryCollector's 15.0s macro_benchmarks step timeout.
    _FETCH_DEADLINE_SEC = 12.5
    _POST_YAHOO_RESERVED_SEC = 4.0
    _DERIBIT_CARRYOVER_MAX_AGE_SEC = 3600.0
    _COINBASE_CARRYOVER_MAX_AGE_SEC = 300.0
    _DERIBIT_METRIC_SUFFIXES = ("dvol", "rv_pct", "vrp_pct", "pc_oi_ratio", "skew_25d_pct")

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        now_ts: Optional[float] = None,
        timeout_sec: float = 3.5,
        allow_network: bool = True,
        persist: bool = True,
        fetch_fn: Optional[Callable[..., Any]] = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        if not allow_network and fetch_fn is None:
            return self._read_cached_dict(
                cachedb_dir,
                "macro_benchmarks_files",
                CACHE_MACRO_BENCHMARKS_FILENAME,
                read_json_fn=read_json_fn,
            )[0]
        cache_dir = cachedb_dir
        explicit_now = now_ts is not None
        eff_now = resolve_now_ts(now_ts)
        _fetch = self._resolve_fetch_fn(fetch_fn)
        deadline_mono = self._build_fetch_deadline(fetch_fn)
        yahoo_deadline_mono = (
            (deadline_mono - self._POST_YAHOO_RESERVED_SEC)
            if (fetch_fn is None and deadline_mono is not None)
            else deadline_mono
        )
        items: Dict[str, Dict[str, Any]] = {}
        for label, yahoo_sym, unit_kind in MACRO_BENCHMARK_SPECS:
            req_to = resolve_remaining_timeout_sec(yahoo_deadline_mono, timeout_sec, min_timeout=0.8)
            if req_to is None:
                break
            try:
                url = get_yahoo_chart_url(yahoo_sym)
                raw = _fetch(url, timeout=req_to, headers={"User-Agent": "Mozilla/5.0"})
                meta, ohlcv_bars = extract_yahoo_chart_ohlcv_bars(raw)
                if meta is None:
                    continue
                pts = [(b[0], b[4]) for b in ohlcv_bars]
                pts_24h = ([p for p in pts if p[0] >= pts[-1][0] - 86400.0] or pts) if pts else []
                mkt_state, px, ext_px = resolve_yahoo_chart_market_state_and_price(meta, pts, now_ts=eff_now)
                if px is None:
                    continue
                fullday_chg = safe_float(meta.get("fulldayChangePercent"), default=None)
                reg_chg = safe_float(meta.get("regularMarketChangePercent"), default=None)
                reg_px = safe_pos_float(meta.get("regularMarketPrice"), default=None)
                px_is_extended = bool(
                    ext_px is not None
                    and (reg_px is None or abs(px - reg_px) / reg_px >= 0.0005)
                )
                if fullday_chg is not None and abs(fullday_chg) > 1e-9:
                    chg_24h: Optional[float] = round(fullday_chg, 2) + 0.0
                elif not px_is_extended and reg_chg is not None and abs(reg_chg) > 1e-9:
                    chg_24h = round(reg_chg, 2) + 0.0
                else:
                    chg_24h = None
                if chg_24h is not None and pts_24h:
                    lo_24_m = min(p[1] for p in pts_24h)
                    hi_24_m = max(p[1] for p in pts_24h)
                    if chg_24h <= -90.0 or not (lo_24_m * 0.90 <= (px / (1.0 + chg_24h / 100.0)) <= hi_24_m * 1.10):
                        chg_24h = None
                if chg_24h is None or (ext_px is not None and safe_pos_float(meta.get("fulldayPrice"), default=None) is None and fullday_chg is None):
                    prev_close = (
                        first_pos_float(reg_px, meta.get("previousClose"), meta.get("chartPreviousClose"))
                        if (px_is_extended and mkt_state in ("PRE", "PREPRE", "OVERNIGHT"))
                        else first_pos_float(meta.get("previousClose"), meta.get("chartPreviousClose"))
                    )
                    if prev_close is not None and pts_24h:
                        lo_24_b = min(p[1] for p in pts_24h)
                        hi_24_b = max(p[1] for p in pts_24h)
                        if not (lo_24_b * 0.90 <= prev_close <= hi_24_b * 1.10):
                            prev_close = None
                    if prev_close is not None and prev_close > 0:
                        chg_24h = round(((px - prev_close) / prev_close) * 100.0, 2) + 0.0
                    elif len(pts_24h) >= 2 and pts_24h[0][1] > 0:
                        chg_24h = round(((px - pts_24h[0][1]) / pts_24h[0][1]) * 100.0, 2) + 0.0
                globex_session_chg = None
                if (
                    len(pts_24h) >= 2
                    and (pts_24h[-1][0] - pts_24h[0][0]) >= 64800.0
                    and pts_24h[0][1] > 0
                ):
                    bars_24h_chg = round(((px - pts_24h[0][1]) / pts_24h[0][1]) * 100.0, 2) + 0.0
                    if chg_24h is None or abs(bars_24h_chg - chg_24h) >= 1.0:
                        if chg_24h is not None:
                            globex_session_chg = chg_24h
                        chg_24h = bars_24h_chg

                dec = 4 if unit_kind == "yield_pct" else 2
                item_entry: Dict[str, Any] = {
                    "symbol": label, "yahoo_symbol": yahoo_sym, "unit": unit_kind,
                    "price": round(px, dec), "chg_24h_pct": (chg_24h + 0.0) if chg_24h is not None else 0.0,
                }
                if globex_session_chg is not None:
                    item_entry["globex_session_chg_pct"] = globex_session_chg
                if (cash_lbl := {"ES=F": "SPY", "NQ=F": "QQQ"}.get(label)) and isinstance((cash_it := items.get(cash_lbl)), dict) and len(pts) >= 2:
                    if (cash_bar_ts := safe_pos_float(cash_it.get("last_bar_ts"), default=None)) is not None and pts[-1][0] > cash_bar_ts + 300.0:
                        ref_cash_pt = min(pts, key=lambda p: abs(p[0] - cash_bar_ts))
                        if abs(ref_cash_pt[0] - cash_bar_ts) <= 1800.0 and ref_cash_pt[1] > 0.0:
                            item_entry["since_cash_close_pct"] = round(((px - ref_cash_pt[1]) / ref_cash_pt[1]) * 100.0, 2) + 0.0
                if len(pts) >= 2:
                    last_bar_ts = pts[-1][0]
                    raw_quote_ts = (
                        first_pos_float(meta.get("preMarketTime"), meta.get("postMarketTime"))
                        if ext_px is not None and mkt_state in ("PRE", "PREPRE")
                        else (
                            first_pos_float(meta.get("postMarketTime"), meta.get("preMarketTime"))
                            if ext_px is not None and mkt_state in ("POST", "POSTPOST")
                            else (meta.get("regularMarketTime") if mkt_state != "CLOSED" else None)
                        )
                    )
                    quote_ts_meta = normalize_epoch_sec(raw_quote_ts, default=None) if raw_quote_ts is not None else None
                    eff_quote_ts = max(last_bar_ts, quote_ts_meta) if quote_ts_meta is not None else last_bar_ts
                    item_entry["last_bar_ts"] = math.trunc(eff_quote_ts)
                    chart_bar_age_s = max(0.0, eff_now - last_bar_ts) if eff_now > 0.0 else 0.0
                    bar_age_s = max(0.0, eff_now - eff_quote_ts) if eff_now > 0.0 else 0.0
                    if eff_now > 0.0:
                        item_entry["bar_age_min"] = round(bar_age_s / 60.0, 1)
                    if not mkt_state and eff_now > 0.0 and bar_age_s > 2700.0:
                        mkt_state = "CLOSED"
                    if mkt_state != "CLOSED" and (eff_now <= 0.0 or chart_bar_age_s <= 2700.0):
                        prior_pts = pts[:-1]
                        for target_sec, key_name in ((3600.0, "chg_1h_pct"), (900.0, "chg_15m_pct")):
                            target_ts = last_bar_ts - target_sec
                            max_tol = target_sec * 0.6 if target_sec < 1800.0 else target_sec * 0.5
                            ref_pt = min(prior_pts, key=lambda p: abs(p[0] - target_ts))
                            if abs(ref_pt[0] - target_ts) <= max_tol and ref_pt[1] > 0:
                                item_entry[key_name] = round(((px - ref_pt[1]) / ref_pt[1]) * 100.0, 2) + 0.0
                if mkt_state:
                    item_entry["market_state"] = mkt_state
                if pts_24h:
                    vals_24h = [p[1] for p in pts_24h] + [px]
                    item_entry["low_24h"], item_entry["high_24h"] = round(min(vals_24h), dec), round(max(vals_24h), dec)
                items[label] = item_entry
            except Exception:
                continue

        if not items:
            return None
        extra_macro: Dict[str, Any] = {}
        if fetch_fn is None:
            prev_snap, _ = self._read_cached_dict(
                cache_dir,
                "macro_benchmarks_files",
                CACHE_MACRO_BENCHMARKS_FILENAME,
                read_json_fn=read_json_fn,
            )
            # 1. Coinbase BTC Premium Index (Coinbase BTC-USD vs SSOT cacheManager BTCUSDC / fallback Binance HTTP)
            if (cb_to := resolve_remaining_timeout_sec(deadline_mono, 2.0, min_timeout=1.0)) is not None:
                try:
                    cb_raw = _fetch(_cfg_str("feeds", "coinbase_btc_ticker_url"), timeout=cb_to, headers={"User-Agent": "Mozilla/5.0"})
                    cb_px = safe_pos_float((cb_raw or {}).get("price") if isinstance(cb_raw, dict) else None, default=None)
                    bn_px = resolve_ssot_price_from_cache("BTCUSDC", cache_dir=cache_dir, max_age_sec=60.0, now_ts=eff_now)
                    if bn_px is None and (bn_to := resolve_remaining_timeout_sec(deadline_mono, 2.0, min_timeout=0.8)) is not None:
                        bn_raw = _fetch(_cfg_str("feeds", "binance_btc_ticker_url"), timeout=bn_to, headers={"User-Agent": "Mozilla/5.0"})
                        bn_px = safe_pos_float((bn_raw or {}).get("price") if isinstance(bn_raw, dict) else None, default=None)
                    if cb_px and bn_px and bn_px > 0:
                        cb_bps = round(((cb_px - bn_px) / bn_px) * 10000.0, 2) + 0.0
                        extra_macro["coinbase_premium_bps"] = cb_bps
                        extra_macro["coinbase_premium_regime"] = (
                            "US_INSTITUTIONAL_BUYING" if cb_bps >= 5.0 else ("US_SPOT_SELLING_PRESSURE" if cb_bps <= -5.0 else "NEUTRAL")
                        )
                        extra_macro["coinbase_premium_ts"] = eff_now
                except Exception:
                    pass
            if "coinbase_premium_bps" not in extra_macro and isinstance(prev_snap, dict) and prev_snap.get("coinbase_premium_bps") is not None:
                prev_cb_ts = normalize_epoch_sec(prev_snap.get("coinbase_premium_ts") or prev_snap.get("ts"), default=0.0) or 0.0
                if prev_cb_ts > 0.0 and 0.0 <= (eff_now - prev_cb_ts) <= self._COINBASE_CARRYOVER_MAX_AGE_SEC:
                    extra_macro["coinbase_premium_bps"] = prev_snap["coinbase_premium_bps"]
                    if prev_snap.get("coinbase_premium_regime") is not None:
                        extra_macro["coinbase_premium_regime"] = prev_snap["coinbase_premium_regime"]
                    extra_macro["coinbase_premium_ts"] = prev_cb_ts
            # 2. Deribit BTC / ETH Implied Volatility Index (DVOL) & Institutional Options Surface
            deribit_currencies = tuple(get_deribit_options_currencies())
            deribit_ccy_ts: Dict[str, float] = {}
            if resolve_remaining_timeout_sec(deadline_mono, 2.0, min_timeout=1.2) is not None:
                try:
                    end_ms = math.trunc(eff_now * 1000)
                    start_ms = end_ms - 3600 * 1000
                    dvol_tpl = _cfg_str("feeds", "deribit_dvol_url_template")
                    hvol_tpl = _cfg_str("feeds", "deribit_hist_vol_url_template")
                    obook_tpl = _cfg_str("feeds", "deribit_options_book_url_template")
                    for ccy in deribit_currencies:
                        dvol_to = resolve_remaining_timeout_sec(deadline_mono, 2.0, min_timeout=1.0)
                        if dvol_to is None:
                            break
                        d_key = f"{ccy.lower()}_dvol"
                        ccy_fresh = False
                        try:
                            d_raw = _fetch(dvol_tpl.format(currency=ccy, start_ms=start_ms, end_ms=end_ms), timeout=dvol_to, headers={"User-Agent": "Mozilla/5.0"})
                            d_data = ((d_raw or {}).get("result") or {}).get("data") if isinstance(d_raw, dict) else None
                            if isinstance(d_data, list) and d_data and isinstance(d_data[-1], list) and len(d_data[-1]) >= 5:
                                if (dv := safe_pos_float(d_data[-1][4], default=None)) is not None:
                                    extra_macro[d_key] = round(dv, 2)
                                    ccy_fresh = True
                        except Exception:
                            pass
                        rv_val: Optional[float] = None
                        if (hv_to := resolve_remaining_timeout_sec(deadline_mono, 2.0, min_timeout=0.8)) is not None:
                            try:
                                hv_raw = _fetch(hvol_tpl.format(currency=ccy), timeout=hv_to, headers={"User-Agent": "Mozilla/5.0"})
                                hv_res = (hv_raw or {}).get("result") if isinstance(hv_raw, dict) else None
                                if isinstance(hv_res, list) and hv_res and isinstance(hv_res[-1], list) and len(hv_res[-1]) >= 2:
                                    rv_val = safe_pos_float(hv_res[-1][1], default=None)
                            except Exception:
                                pass
                        opt_books_list: Any = None
                        if (ob_to := resolve_remaining_timeout_sec(deadline_mono, 2.5, min_timeout=1.0)) is not None:
                            try:
                                ob_raw = _fetch(obook_tpl.format(currency=ccy), timeout=ob_to, headers={"User-Agent": "Mozilla/5.0"})
                                opt_books_list = (ob_raw or {}).get("result") if isinstance(ob_raw, dict) else None
                            except Exception:
                                pass
                        surf = compute_deribit_options_surface(
                            opt_books_list,
                            currency=ccy,
                            dvol=extra_macro.get(d_key),
                            rv=rv_val,
                            now_ts=eff_now,
                        )
                        if surf:
                            extra_macro.update(surf)
                            ccy_fresh = True
                        if ccy_fresh:
                            deribit_ccy_ts[ccy] = eff_now
                except Exception:
                    pass
            if isinstance(prev_snap, dict):
                fallback_db_ts = normalize_epoch_sec(prev_snap.get("deribit_ts") or prev_snap.get("ts"), default=0.0) or 0.0
                for ccy in deribit_currencies:
                    if ccy in deribit_ccy_ts:
                        continue
                    ccy_ts_key = f"{ccy.lower()}_deribit_ts"
                    prev_ccy_ts = (
                        normalize_epoch_sec(prev_snap.get(ccy_ts_key), default=0.0) or 0.0
                        if prev_snap.get(ccy_ts_key) is not None
                        else fallback_db_ts
                    )
                    if not (prev_ccy_ts > 0.0 and 0.0 <= (eff_now - prev_ccy_ts) <= self._DERIBIT_CARRYOVER_MAX_AGE_SEC):
                        continue
                    ccy_keys = tuple(f"{ccy.lower()}_{sfx}" for sfx in self._DERIBIT_METRIC_SUFFIXES)
                    if not any(k in extra_macro for k in ccy_keys):
                        carried_ccy = False
                        for k in ccy_keys:
                            if prev_snap.get(k) is not None:
                                extra_macro[k] = prev_snap[k]
                                carried_ccy = True
                        if carried_ccy:
                            deribit_ccy_ts[ccy] = prev_ccy_ts
            for ccy, ccy_ts_val in deribit_ccy_ts.items():
                extra_macro[f"{ccy.lower()}_deribit_ts"] = ccy_ts_val
            if deribit_ccy_ts:
                extra_macro["deribit_ts"] = min(deribit_ccy_ts.values())
            if "btc_dvol" in extra_macro:
                bv = extra_macro["btc_dvol"]
                extra_macro["options_vol_regime"] = "HIGH_IV_EVENT_HEDGING" if bv >= 62.0 else ("COMPRESSED_IV" if bv <= 42.0 else "NORMAL_IV")

        extra_macro.update(compute_futures_premarket_lead({"items": items}))
        payload_ts = eff_now if explicit_now else resolve_now_ts(None)
        payload: Dict[str, Any] = {
            "ts": payload_ts,
            "items": items,
            "equity_regime": compute_equity_macro_regime({"items": items, **extra_macro}),
            **extra_macro,
        }
        payload["cross_asset_regime"] = compute_cross_asset_regime(
            payload,
            btc_chg_24h_pct=resolve_cached_btc_24h_change_pct(cache_dir),
        )
        payload["summary_line"] = format_macro_benchmarks_summary_line(payload)
        self._persist_snapshot(cache_dir, CACHE_MACRO_BENCHMARKS_FILENAME, payload, persist=persist)
        return payload


class MacroEconomicCalendarCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) fetching weekly USD macro releases (ForexFactory) and tracked/bellwether earnings (Nasdaq)."""

    cls_name = "MacroEconomicCalendarCollector"
    COLLECTOR_ID = "macro_economic_calendar"
    # Shared wall-clock budget strictly below GlobalTelemetryCollector's 12.0s macro_calendar step timeout.
    _FETCH_DEADLINE_SEC = 10.0

    @staticmethod
    def _target_earnings_dates(dt_ny: datetime) -> List[str]:
        cur = dt_ny.date()
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        nxt = cur + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        return [cur.isoformat(), nxt.isoformat()]

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        now_ts: Optional[float] = None,
        timeout_sec: float = 3.0,
        allow_network: bool = True,
        persist: bool = True,
        fetch_fn: Optional[Callable[..., Any]] = None,
        read_json_fn: Any = None,
        tracked_equity_tickers: Optional[List[str]] = None,
        skip_if_fresh_sec: float = 0.0,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        eff_now = resolve_now_ts(now_ts)
        prev_snap, is_fresh_cached = self._read_fresh_cached_snapshot(
            cachedb_dir,
            "macro_calendar_files",
            CACHE_MACRO_CALENDAR_FILENAME,
            now_ts=eff_now,
            skip_if_fresh_sec=skip_if_fresh_sec,
            read_json_fn=read_json_fn,
            is_healthy_fn=lambda d: _parse_bool_flag(d.get("events_ok"), False)
            and _parse_bool_flag(d.get("earnings_ok"), False),
        )
        if not allow_network and fetch_fn is None:
            return prev_snap
        if is_fresh_cached and prev_snap is not None:
            return prev_snap
        _fetch = self._resolve_fetch_fn(fetch_fn)
        deadline_mono = self._build_fetch_deadline(fetch_fn)
        dt_ny = _mss.resolve_secondary_ny_datetime(eff_now)
        tz_ny = dt_ny.tzinfo or timezone.utc
        req_headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

        # 1. Weekly economic releases (ForexFactory JSON)
        events: List[Dict[str, Any]] = []
        events_ok = False
        allowed_countries = set(get_macro_calendar_countries())
        medium_kws = get_macro_calendar_medium_tier_keywords()
        ff_url = get_macro_calendar_endpoint_url()
        if ff_url and (ff_to := resolve_remaining_timeout_sec(deadline_mono, timeout_sec, min_timeout=0.8)) is not None:
            try:
                raw_ff = _fetch(ff_url, timeout=ff_to, headers=req_headers)
                if isinstance(raw_ff, list):
                    events_ok = True
                    seen_ev: set[tuple[int, str]] = set()
                    for row in raw_ff:
                        if not isinstance(row, dict):
                            continue
                        country = str(row.get("country") or "").strip().upper()
                        if allowed_countries and country not in allowed_countries:
                            continue
                        title = str(row.get("title") or "").strip()
                        if not title:
                            continue
                        impact_u = str(row.get("impact") or "").strip().upper()
                        if impact_u == "HIGH":
                            norm_impact = "HIGH"
                        elif impact_u == "MEDIUM" and medium_kws and any(kw in title.lower() for kw in medium_kws):
                            norm_impact = "MEDIUM"
                        else:
                            continue
                        date_raw = str(row.get("date") or "").strip()
                        if not date_raw:
                            continue
                        try:
                            ev_dt = datetime.fromisoformat(date_raw)
                        except Exception:
                            continue
                        if ev_dt.tzinfo is None:
                            ev_dt = ev_dt.replace(tzinfo=tz_ny)
                        ev_ts = ev_dt.timestamp()
                        dedup_k = (round(ev_ts / 60.0), title.lower())
                        if dedup_k in seen_ev:
                            continue
                        seen_ev.add(dedup_k)
                        ev_ny = ev_dt.astimezone(tz_ny)
                        events.append(
                            {
                                "ts": round(ev_ts, 3),
                                "date_iso": ev_ny.date().isoformat(),
                                "ny_time": ev_ny.strftime("%H:%M"),
                                "title": title,
                                "country": country,
                                "impact": norm_impact,
                                "forecast": str(row.get("forecast") or "").strip(),
                                "previous": str(row.get("previous") or "").strip(),
                                "source": "forexfactory",
                            }
                        )
                    events.sort(key=lambda e: (e["ts"], 0 if e["impact"] == "HIGH" else 1, e["title"]))
            except Exception as exc:
                logger.debug("ForexFactory economic calendar fetch failed: %s", exc)

        # 2. Tracked portfolio & bellwether corporate earnings (Nasdaq API, 2 weekday requests max)
        earnings: List[Dict[str, Any]] = []
        earnings_ok = False
        target_dates = self._target_earnings_dates(dt_ny)
        tracked_src = (
            tracked_equity_tickers
            if tracked_equity_tickers is not None
            else get_tracked_equity_base_tickers(include_tokenized=True)
        )
        tracked_set = {str(t or "").strip().upper() for t in tracked_src if str(t or "").strip()}
        bellwether_set = set(get_earnings_bellwether_tickers())
        watch_tickers = tracked_set | bellwether_set
        seen_er: set[tuple[str, str]] = set()
        for d_iso in target_dates:
            nq_to = resolve_remaining_timeout_sec(deadline_mono, timeout_sec, min_timeout=0.8)
            if nq_to is None:
                break
            try:
                nq_url = get_nasdaq_earnings_calendar_url(d_iso)
                raw_nq = _fetch(nq_url, timeout=nq_to, headers=req_headers)
                if not isinstance(raw_nq, dict):
                    continue
                data_obj = raw_nq.get("data")
                if not isinstance(data_obj, dict):
                    continue
                earnings_ok = True
                rows = data_obj.get("rows")
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    sym_u = str(row.get("symbol") or "").strip().upper()
                    if not sym_u or (watch_tickers and sym_u not in watch_tickers) or (d_iso, sym_u) in seen_er:
                        continue
                    seen_er.add((d_iso, sym_u))
                    time_raw = str(row.get("time") or "").strip().lower()
                    est_ts: Optional[float] = None
                    if "pre" in time_raw or "bmo" in time_raw:
                        timing = "BMO"
                        est_hm = (8, 0)
                    elif "after" in time_raw or "amc" in time_raw:
                        timing = "AMC"
                        est_hm = (16, 5)
                    else:
                        timing = "TBD"
                        est_hm = None
                    if est_hm is not None:
                        try:
                            d_dt = datetime.fromisoformat(d_iso)
                            est_ts = datetime(d_dt.year, d_dt.month, d_dt.day, est_hm[0], est_hm[1], tzinfo=tz_ny).timestamp()
                        except Exception:
                            est_ts = None
                    earnings.append(
                        {
                            "symbol": sym_u,
                            "name": str(row.get("name") or "").strip(),
                            "date_iso": d_iso,
                            "timing": timing,
                            "est_ts": round(est_ts, 3) if est_ts is not None else None,
                            "eps_forecast": str(row.get("epsForecast") or "").strip(),
                            "fiscal_quarter": str(row.get("fiscalQuarterEnding") or "").strip(),
                            "is_portfolio_holding": sym_u in tracked_set,
                            "source": "nasdaq",
                        }
                    )
            except Exception as exc:
                logger.debug("Nasdaq earnings calendar fetch failed for %s: %s", d_iso, exc)

        def _er_sort(e: Dict[str, Any]) -> tuple[str, int, int, str]:
            return (
                e["date_iso"],
                0 if e["is_portfolio_holding"] else 1,
                0 if e["timing"] == "BMO" else (1 if e["timing"] == "AMC" else 2),
                e["symbol"],
            )

        earnings.sort(key=_er_sort)

        if not events_ok and not earnings_ok:
            return None
        done_now = resolve_now_ts(now_ts)
        done_now_r = round(done_now, 3)
        events_ts: Optional[float] = done_now_r if events_ok else None
        earnings_ts: Optional[float] = done_now_r if earnings_ok else None
        if isinstance(prev_snap, dict):
            prev_snap_ts = normalize_epoch_sec(prev_snap.get("ts"), default=0.0) or 0.0
            if not events_ok and isinstance(prev_snap.get("events"), list):
                prev_ev_ts = (
                    normalize_epoch_sec(prev_snap.get("events_ts"), default=0.0) or 0.0
                    if prev_snap.get("events_ts") is not None
                    else prev_snap_ts
                )
                if prev_ev_ts > 0.0 and 0.0 <= (done_now - prev_ev_ts) <= 86400.0:
                    events = list(prev_snap["events"])
                    events_ts = round(prev_ev_ts, 3)
            if not earnings_ok and isinstance(prev_snap.get("earnings"), list):
                prev_er_ts = (
                    normalize_epoch_sec(prev_snap.get("earnings_ts"), default=0.0) or 0.0
                    if prev_snap.get("earnings_ts") is not None
                    else prev_snap_ts
                )
                if prev_er_ts > 0.0 and 0.0 <= (done_now - prev_er_ts) <= 86400.0:
                    earnings = list(prev_snap["earnings"])
                    earnings_ts = round(prev_er_ts, 3)

        payload: Dict[str, Any] = {
            "ts": done_now_r,
            "source": "forexfactory+nasdaq",
            "events_ok": events_ok,
            "earnings_ok": earnings_ok,
            "fetched_earnings_dates": target_dates,
            "events": events,
            "earnings": earnings,
        }
        if events_ts is not None:
            payload["events_ts"] = events_ts
        if earnings_ts is not None:
            payload["earnings_ts"] = earnings_ts
        self._persist_snapshot(cachedb_dir, CACHE_MACRO_CALENDAR_FILENAME, payload, persist=persist)
        return payload


class SpotEtfFlowsCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) fetching daily US Spot BTC/ETH ETF net flows (Farside Investors HTML tables)."""

    cls_name = "SpotEtfFlowsCollector"
    COLLECTOR_ID = "spot_etf_flows"
    # Shared wall-clock budget strictly below GlobalTelemetryCollector's 12.0s etf_flows step timeout.
    _FETCH_DEADLINE_SEC = 10.0
    _REQUEST_HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        now_ts: Optional[float] = None,
        timeout_sec: float = 4.0,
        allow_network: bool = True,
        persist: bool = True,
        fetch_text_fn: Optional[Callable[..., Any]] = None,
        read_json_fn: Any = None,
        skip_if_fresh_sec: float = 0.0,
        max_days: int = 10,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        _ = (symbol, kwargs)
        eff_now = resolve_now_ts(now_ts)
        flow_urls = get_spot_etf_flow_urls()
        expected_assets = tuple(asset_u for asset_u, _ in flow_urls)
        prev_snap, is_fresh_cached = self._read_fresh_cached_snapshot(
            cachedb_dir,
            "etf_flows_files",
            CACHE_ETF_FLOWS_FILENAME,
            now_ts=eff_now,
            skip_if_fresh_sec=skip_if_fresh_sec,
            read_json_fn=read_json_fn,
            is_healthy_fn=lambda d: isinstance(d.get("assets"), dict)
            and bool(d["assets"])
            and all(
                a in d["assets"] and _parse_bool_flag(d.get(f"{a.lower()}_ok"), False)
                for a in expected_assets
            ),
        )
        if not allow_network and fetch_text_fn is None:
            return prev_snap
        if is_fresh_cached and prev_snap is not None:
            return prev_snap
        if not flow_urls:
            return None

        _fetch_text = self._resolve_text_fetch_fn(fetch_text_fn)
        deadline_mono = self._build_fetch_deadline(fetch_text_fn)
        assets_out: Dict[str, Dict[str, Any]] = {}
        ok_flags: Dict[str, bool] = {}

        for asset_u, url in flow_urls:
            req_to = resolve_remaining_timeout_sec(deadline_mono, timeout_sec, min_timeout=1.5)
            if req_to is None:
                ok_flags[asset_u] = False
                continue
            try:
                html_raw = _fetch_text(url, timeout=req_to, headers=dict(self._REQUEST_HEADERS))
                parsed = parse_farside_etf_flows_html(html_raw, asset=asset_u, max_days=max_days)
                if isinstance(parsed, dict) and parsed:
                    assets_out[asset_u] = parsed
                    ok_flags[asset_u] = True
                else:
                    ok_flags[asset_u] = False
            except Exception as exc:
                ok_flags[asset_u] = False
                logger.debug("Farside %s spot ETF flow fetch failed: %s", asset_u, exc)

        if not assets_out:
            return None

        done_now = resolve_now_ts(now_ts)
        done_now_r = round(done_now, 3)
        asset_ts_out: Dict[str, float] = {
            f"{asset_u.lower()}_ts": done_now_r
            for asset_u in expected_assets
            if asset_u in assets_out
        }
        if isinstance(prev_snap, dict):
            prev_assets = prev_snap.get("assets")
            prev_snap_ts = normalize_epoch_sec(prev_snap.get("ts"), default=0.0) or 0.0
            if isinstance(prev_assets, dict):
                for asset_u in expected_assets:
                    if asset_u not in assets_out and isinstance(prev_assets.get(asset_u), dict):
                        a_ts_key = f"{asset_u.lower()}_ts"
                        prev_a_ts = (
                            normalize_epoch_sec(prev_snap.get(a_ts_key), default=0.0) or 0.0
                            if prev_snap.get(a_ts_key) is not None
                            else prev_snap_ts
                        )
                        if prev_a_ts > 0.0 and 0.0 <= (done_now - prev_a_ts) <= 86400.0:
                            assets_out[asset_u] = dict(prev_assets[asset_u])
                            asset_ts_out[a_ts_key] = round(prev_a_ts, 3)

        payload: Dict[str, Any] = {
            "ts": done_now_r,
            "source": "farside",
            "assets": assets_out,
            **{f"{asset_u.lower()}_ok": bool(ok_flags.get(asset_u, False)) for asset_u in expected_assets},
            **asset_ts_out,
        }
        payload["summary_line"] = format_etf_flows_summary_line(payload)
        self._persist_snapshot(cachedb_dir, CACHE_ETF_FLOWS_FILENAME, payload, persist=persist)
        return payload


def _resolve_macro_collect_inputs(
    now_ts: Optional[float], read_json_fn: Optional[Callable]
) -> tuple[float, Callable, Callable]:
    eff_now = resolve_now_ts(now_ts)
    rj = _resolve_read_json_cached(read_json_fn, default_fn=read_json_recovering)
    rj_any = read_json_recovering if rj is _default_read_json_cached else rj
    return eff_now, rj, rj_any


class MacroAndNewsSensorsCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) harvesting Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, and Breaking News."""

    cls_name = "MacroAndNewsSensorsCollector"
    COLLECTOR_ID = "macro_and_news_sensors"
    _MACRO_SENSOR_CACHE_SPECS = MACRO_SENSOR_CACHE_SPECS

    @staticmethod
    def _normalize_market_breadth_movers(enriched: Dict[str, Any]) -> None:
        excl_mb_bases = set(get_breadth_excluded_bases()) | {"USDCUSDT"}
        for mover_key in ("top_gainers", "top_losers"):
            if isinstance(raw_movers := enriched.get(mover_key), list):
                dedup_movers = []
                seen_mb_bases: set[str] = set()
                for m in raw_movers:
                    if (
                        isinstance(m, (list, tuple))
                        and len(m) >= 2
                        and isinstance(m[0], str)
                        and m[0].strip().isascii()
                        and m[0].strip().isalnum()
                    ):
                        mb_base = canonical_base_asset(m[0].strip()) or m[0].strip().upper()
                        if (
                            mb_base in excl_mb_bases
                            or (len(mb_base) > 4 and mb_base.endswith(("UP", "DOWN", "BULL", "BEAR")))
                        ):
                            continue
                        if mb_base not in seen_mb_bases:
                            seen_mb_bases.add(mb_base)
                            dedup_movers.append(m)
                enriched[mover_key] = dedup_movers

    @staticmethod
    def _normalize_macro_benchmarks_items(
        enriched: Dict[str, Any], *, cache_dir: str, now_ts: float
    ) -> Dict[str, Any]:
        stripped_stale_intraday = False
        if isinstance(mb_items := enriched.get("items"), dict):
            enriched_items: Dict[str, Any] = {}
            for mb_k, mb_v in mb_items.items():
                if isinstance(mb_v, dict):
                    mb_copy = dict(mb_v)
                    mb_px = safe_pos_float(mb_copy.get("price"), default=None)
                    mb_lo = safe_pos_float(mb_copy.get("low_24h"), default=None)
                    mb_hi = safe_pos_float(mb_copy.get("high_24h"), default=None)
                    if mb_px is not None:
                        if mb_lo is not None and mb_px < mb_lo:
                            mb_copy["low_24h"] = mb_lo = mb_px
                        if mb_hi is not None and mb_px > mb_hi:
                            mb_copy["high_24h"] = mb_hi = mb_px
                        chg_24_raw = safe_float(mb_copy.get("chg_24h_pct"), default=None)
                        if (
                            chg_24_raw is not None
                            and mb_lo is not None
                            and mb_hi is not None
                            and mb_lo > 0.0
                            and mb_hi >= mb_lo
                        ):
                            span_24_pct = ((mb_hi - mb_lo) / mb_lo) * 100.0
                            if abs(chg_24_raw) > max(span_24_pct * 2.5, 5.0):
                                ref_bound = mb_hi if chg_24_raw < 0.0 else mb_lo
                                mb_copy["chg_24h_pct"] = round(((mb_px - ref_bound) / ref_bound) * 100.0, 2) + 0.0
                                stripped_stale_intraday = True
                    bar_ts_v = safe_pos_float(mb_copy.get("last_bar_ts"), default=None)
                    bar_age_s: Optional[float] = None
                    if bar_ts_v is not None and now_ts > 0.0:
                        bar_age_s = max(0.0, now_ts - bar_ts_v)
                        mb_copy["bar_age_min"] = round(bar_age_s / 60.0, 1)
                        if bar_age_s > 2700.0:
                            cur_ms = str(mb_copy.get("market_state") or "").strip().upper()
                            if cur_ms in ("", "REGULAR", "PRE", "POST", "PREPRE", "POSTPOST"):
                                mb_copy["market_state"] = "CLOSED"
                                stripped_stale_intraday = True
                            for stale_k in ("chg_1h_pct", "chg_15m_pct"):
                                if stale_k in mb_copy:
                                    mb_copy.pop(stale_k, None)
                                    stripped_stale_intraday = True
                    if mb_k in ("SPY", "QQQ", "IBIT") and now_ts > 0.0 and bar_age_s is not None and bar_age_s >= 600.0:
                        if resolve_us_equity_session_state(now_ts) == "CLOSED":
                            if str(mb_copy.get("market_state") or "").strip().upper() != "CLOSED":
                                mb_copy["market_state"] = "CLOSED"
                                stripped_stale_intraday = True
                    elif mb_k in ("DXY", "ES=F", "NQ=F") and now_ts > 0.0 and bar_age_s is not None and bar_age_s >= 600.0:
                        try:
                            dt_ny = _mss.resolve_secondary_ny_datetime(now_ts)
                            mins_ny = dt_ny.hour * 60 + dt_ny.minute
                            if (
                                (dt_ny.weekday() == 4 and mins_ny >= 1020)
                                or dt_ny.weekday() == 5
                                or (dt_ny.weekday() == 6 and mins_ny < 1020)
                            ):
                                if str(mb_copy.get("market_state") or "").strip().upper() != "CLOSED":
                                    mb_copy["market_state"] = "CLOSED"
                                    stripped_stale_intraday = True
                                for stale_k in ("chg_1h_pct", "chg_15m_pct"):
                                    if stale_k in mb_copy:
                                        mb_copy.pop(stale_k, None)
                                        stripped_stale_intraday = True
                        except Exception:
                            pass
                    if mb_k in ("SPY", "QQQ", "IBIT", "US10Y", "VIX"):
                        is_stale_us_bar = str(mb_copy.get("market_state") or "").strip().upper() == "CLOSED"
                        if bar_ts_v is None and not mb_copy.get("market_state") and now_ts > 0.0:
                            try:
                                dt_ny = _mss.resolve_secondary_ny_datetime(now_ts)
                                mins_ny = dt_ny.hour * 60 + dt_ny.minute
                                if dt_ny.weekday() >= 5 or not (570 <= mins_ny <= 1005):
                                    is_stale_us_bar = True
                            except Exception:
                                pass
                        if is_stale_us_bar:
                            for stale_k in ("chg_1h_pct", "chg_15m_pct"):
                                if stale_k in mb_copy:
                                    mb_copy.pop(stale_k, None)
                                    stripped_stale_intraday = True
                    for chg_k in ("chg_24h_pct", "chg_1h_pct", "chg_15m_pct"):
                        if (chg_v := safe_float(mb_copy.get(chg_k), default=None)) is not None:
                            mb_copy[chg_k] = round(chg_v, 2) + 0.0
                    enriched_items[mb_k] = mb_copy
                else:
                    enriched_items[mb_k] = mb_v
            enriched["items"] = enriched_items
        return (
            enrich_macro_benchmarks_live_state(
                enriched,
                btc_chg_24h_pct=_mss.load_cached_btc_24h_change_pct(cache_dir),
                force_recompute=stripped_stale_intraday,
                update_summary_line=True,
            )
            or enriched
        )

    @staticmethod
    def _backfill_stale_macro_sensors(
        macro_sensors: Dict[str, Optional[Dict[str, Any]]],
        *,
        cache_dir: str,
        now_ts: float,
        persist: bool,
        with_age_fn: Any,
    ) -> None:
        stale_age_min = _mss.resolve_derived_telemetry_intervals()["sensor_stale_age_min"]
        mb_cur = macro_sensors.get("market_breadth")
        mb_age = safe_float(mb_cur.get("age_min") if isinstance(mb_cur, dict) else None, 999.0)
        mb_missing_rot = isinstance(mb_cur, dict) and (
            mb_cur.get("eth_btc_ratio") is None
            or mb_cur.get("btc_vol_share_pct") is None
            or (
                (first_pos_float(mb_cur.get("total_symbols"), mb_cur.get("total_count"), default=0.0) or 0.0) >= 5.0
                and (len(mb_cur.get("top_gainers") or ()) < 3 or len(mb_cur.get("top_losers") or ()) < 3)
            )
        )
        if mb_cur is None or mb_age is None or mb_age > stale_age_min or mb_missing_rot:
            try:
                breadth_snap = MarketBreadthCollector(cache_ttl_sec=60.0).fetch(force_refresh=True, now_ts=now_ts)
                if breadth_snap is not None:
                    mb_payload = serialize_market_breadth_snapshot(breadth_snap, now_ts=now_ts)
                    mb_path = os.path.join(cache_dir, CACHE_MARKET_BREADTH_FILENAME)
                    MacroTelemetryCollectorBase._persist_snapshot(
                        cache_dir, CACHE_MARKET_BREADTH_FILENAME, mb_payload, persist=persist
                    )
                    macro_sensors["market_breadth"] = with_age_fn(mb_payload, mb_path)
            except Exception:
                pass
        bench_cur = macro_sensors.get("macro_benchmarks")
        bench_age = safe_float(bench_cur.get("age_min") if isinstance(bench_cur, dict) else None, 999.0)
        bench_stale_min = (
            min(stale_age_min, 1.5)
            if _mss.is_us_cash_open_volatility_window(now_ts)
            else stale_age_min
        )
        if bench_cur is None or bench_age is None or bench_age > bench_stale_min:
            if isinstance(live_mb := collect_macro_benchmarks_snapshot(cache_dir, now_ts=now_ts, timeout_sec=2.5, persist=persist), dict) and live_mb:
                macro_sensors["macro_benchmarks"] = with_age_fn(live_mb, os.path.join(cache_dir, CACHE_MACRO_BENCHMARKS_FILENAME))

    @staticmethod
    def _maybe_backfill_cached_macro_snapshot(
        cur_snap: Optional[Dict[str, Any]],
        snap_path: str,
        *,
        now_ts: float,
        max_age_sec: float,
        collect_fn: Callable[[], Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Invoke ``collect_fn`` when a slow disk-backed macro snapshot is missing or older than ``max_age_sec``."""
        snap_ts = normalize_epoch_sec((cur_snap or {}).get("ts"), default=0.0) or 0.0
        if snap_ts <= 0.0 and snap_path and os.path.exists(snap_path):
            try:
                snap_ts = os.path.getmtime(snap_path)
            except OSError:
                snap_ts = 0.0
        if not cur_snap or snap_ts <= 0.0 or (now_ts - snap_ts) > max_age_sec:
            if isinstance(live_snap := collect_fn(), dict) and live_snap:
                return live_snap
        return cur_snap

    @staticmethod
    def _collect_and_rank_breaking_news(
        cache_dir: str, *, now_ts: float, scope: str, read_json_any_fn: Any
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[str]]:
        breaking_news: List[Dict[str, Any]] = []
        valid_news: List[Dict[str, Any]] = []
        stale_valid_sev_titles: set[str] = set()
        news_raw = read_json_any_fn(os.path.join(cache_dir, CACHE_NEWS_FEED_FILENAME))
        news_data = (news_raw.get("items") or news_raw.get("headlines")) if isinstance(news_raw, dict) else news_raw
        if not (isinstance(news_data, list) and news_data):
            return breaking_news, valid_news, stale_valid_sev_titles
        for item in news_data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            if is_excluded_reference_news_item(item.get("title"), item.get("source"), item.get("url")):
                continue
            raw_kws = item.get("matched_keywords") if isinstance(item.get("matched_keywords"), list) else []
            clean_kws = list(filter_valid_severity_keywords(raw_kws, title=str(item.get("title") or ""))) if raw_kws else []
            item_seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            is_stale_sev = item_seen_ts > 0.0 and now_ts > item_seen_ts and (now_ts - item_seen_ts) > 5400.0
            item_high_sev = bool(_parse_bool_flag(item.get("is_high_severity"), False))
            was_valid_aged_sev = bool(is_stale_sev and item_high_sev and (clean_kws or not raw_kws))
            if was_valid_aged_sev:
                stale_valid_sev_titles.add(_strip_headline_publisher_suffix(str(item.get("title") or "")).strip().lower())
                stale_valid_sev_titles.add(str(item.get("title") or "").strip().lower())
            cand_entry = {
                **item,
                "matched_keywords": clean_kws,
                "is_high_severity": False if (is_stale_sev or (raw_kws and not clean_kws)) else item_high_sev,
                "_is_stale_sev": is_stale_sev,
                "_aged_high_sev_context": was_valid_aged_sev,
            }
            merged_dup = False
            for exist_entry in valid_news:
                if headlines_are_semantic_duplicates(cand_entry.get("title"), exist_entry.get("title")):
                    c_ts = safe_pos_float(cand_entry.get("first_seen_ts"), default=0.0) or 0.0
                    e_ts = safe_pos_float(exist_entry.get("first_seen_ts"), default=0.0) or 0.0
                    if c_ts > 0.0 and (e_ts <= 0.0 or c_ts < e_ts):
                        exist_entry["first_seen_ts"] = c_ts
                    c_title = str(cand_entry.get("title") or "")
                    e_title = str(exist_entry.get("title") or "")
                    if _strip_headline_publisher_suffix(c_title) == c_title.strip() and _strip_headline_publisher_suffix(e_title) != e_title.strip():
                        exist_entry["title"] = c_title
                    exist_entry["matched_keywords"] = list(
                        dict.fromkeys([*(exist_entry.get("matched_keywords") or []), *(cand_entry.get("matched_keywords") or [])])
                    )
                    e_sev = bool(_parse_bool_flag(exist_entry.get("is_high_severity"), False))
                    c_sev = bool(_parse_bool_flag(cand_entry.get("is_high_severity"), False))
                    e_aged = bool(_parse_bool_flag(exist_entry.get("_aged_high_sev_context"), False))
                    c_aged = bool(_parse_bool_flag(cand_entry.get("_aged_high_sev_context"), False))
                    had_sev = e_sev or c_sev or e_aged or c_aged
                    merged_ts = safe_pos_float(exist_entry.get("first_seen_ts"), default=0.0) or 0.0
                    merged_stale = merged_ts > 0.0 and now_ts > merged_ts and (now_ts - merged_ts) > 5400.0
                    exist_entry["is_high_severity"] = False if merged_stale else (e_sev or c_sev)
                    exist_entry["_is_stale_sev"] = merged_stale
                    exist_entry["_aged_high_sev_context"] = bool((merged_stale and had_sev) or e_aged or c_aged)
                    if merged_stale and had_sev:
                        for t_dup in (c_title, e_title):
                            if t_dup:
                                stale_valid_sev_titles.add(_strip_headline_publisher_suffix(t_dup).strip().lower())
                                stale_valid_sev_titles.add(t_dup.strip().lower())
                    merged_dup = True
                    break
            if not merged_dup:
                valid_news.append(cand_entry)
        scope_norm = str(scope or "").strip().lower()
        valid_news.sort(
            key=lambda x: (
                1 if x.get("is_high_severity") else 0,
                (
                    (
                        3
                        if (not x.get("_is_stale_sev") and has_tracked_equity_corporate_risk(x.get("title")))
                        else (2 if has_tracked_equity_entity(x.get("title")) else 1)
                    )
                    if (
                        scope_norm == "equities"
                        and (
                            has_tracked_equity_entity(x.get("title"))
                            or "yahoo! finance" in str(x.get("source") or "").lower()
                        )
                    )
                    else (
                        1
                        if (
                            scope_norm == "crypto"
                            and not x.get("_is_stale_sev")
                            and has_tracked_tokenized_equity_corporate_risk(x.get("title"))
                        )
                        else (
                            -1
                            if (
                                scope_norm == "crypto"
                                and not has_tracked_tokenized_equity_entity(x.get("title"))
                                and (
                                    has_tracked_equity_entity(x.get("title"))
                                    or "yahoo! finance" in str(x.get("source") or "").lower()
                                )
                                and not x.get("is_high_severity")
                                and not has_macro_policy_context(x.get("title"))
                            )
                            else 0
                        )
                    )
                ),
                0
                if is_off_topic_general_news_title(x.get("title"))
                else (2 if has_financial_or_geopolitical_context(x.get("title")) else 1),
                safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
            ),
            reverse=True,
        )
        top_news = list(valid_news[:6])
        if len(valid_news) > 6 and not any(
            x.get("is_high_severity") or x.get("_aged_high_sev_context") for x in top_news
        ):
            aged_ctx = sorted(
                (x for x in valid_news[6:] if x.get("_aged_high_sev_context")),
                key=lambda x: safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
                reverse=True,
            )[:2]
            if aged_ctx:
                top_news = top_news[: 6 - len(aged_ctx)] + aged_ctx
        for item in top_news:
            seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            raw_title = str(item.get("title") or "")
            clean_title = _strip_headline_publisher_suffix(raw_title) if raw_title else item.get("title")
            pub_utc = (
                datetime.fromtimestamp(seen_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if 0.0 < seen_ts <= now_ts + 300.0
                else None
            )
            resolved_title = clean_title or item.get("title")
            is_corp_risk = bool(
                not item.get("_is_stale_sev")
                and (
                    (scope_norm == "equities" and has_tracked_equity_corporate_risk(resolved_title))
                    or (scope_norm == "crypto" and has_tracked_tokenized_equity_corporate_risk(resolved_title))
                )
            )
            breaking_news.append({
                "title": resolved_title, "source": item.get("source"),
                "is_high_severity": bool(_parse_bool_flag(item.get("is_high_severity"), False)),
                "keywords": item.get("matched_keywords") or [],
                "age_min": round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None,
                **({"published_utc": pub_utc} if pub_utc else {}),
                **({"is_corporate_risk": True} if is_corp_risk else {}),
            })
        return breaking_news, valid_news, stale_valid_sev_titles

    @staticmethod
    def _reconcile_geopolitical_deescalation(
        gt_cur: Any,
        valid_news: List[Dict[str, Any]],
        breaking_news: List[Dict[str, Any]],
        stale_valid_sev_titles: set[str],
        *,
        cache_dir: str,
        now_ts: float,
        allow_network: bool,
        persist: bool,
        read_json_fn: Any,
    ) -> None:
        if not isinstance(gt_cur, dict):
            return
        if valid_news and not any(_parse_bool_flag(n.get("is_high_severity"), False) for n in valid_news):
            gt_digest = str(gt_cur.get("headlines_digest") or "").strip()
            gt_digest_titles = split_headlines_digest(gt_digest) if gt_digest else []
            valid_digest_titles = [
                t
                for t in gt_digest_titles
                if filter_valid_severity_keywords((), title=t, cache_dir=cache_dir, now_ts=now_ts)
            ]
            gt_digest_invalid = bool(gt_digest_titles and not valid_digest_titles)
            all_valid_digest_aged_90m = bool(
                valid_digest_titles
                and all(
                    t.strip().lower() in stale_valid_sev_titles
                    or _strip_headline_publisher_suffix(t).strip().lower() in stale_valid_sev_titles
                    for t in valid_digest_titles
                )
            )
            newest_news_ts = max((safe_pos_float(n.get("first_seen_ts"), default=0.0) or 0.0 for n in valid_news), default=0.0)
            has_fresh_calm_news = newest_news_ts > 0.0 and 0.0 <= (now_ts - newest_news_ts) <= 3600.0
            gt_age_min = safe_float(gt_cur.get("age_min"), 0.0) or 0.0

            def _reset_and_persist_normal_geopolitical_eval(summary_txt: str, digest_txt: str) -> None:
                gt_cur["threat_level"] = "NORMAL"
                gt_cur["risk_score"] = 0.1
                gt_cur["recommended_brake"] = "NONE"
                gt_cur["summary"] = summary_txt
                gt_cur["headlines_digest"] = digest_txt
                gt_cur["headlines_analyzed"] = len(valid_news)
                gt_cur["ts"] = now_ts
                gt_cur["age_min"] = 0.0
                if persist and allow_network and read_json_fn in (None, read_json_recovering, read_json_cached) and is_live_network_backfill_enabled():
                    try:
                        geo_save_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "geopolitical_eval.json")))
                        MacroTelemetryCollectorBase._persist_cache_json(
                            geo_save_path, {k: v for k, v in gt_cur.items() if k != "age_min"}, indent=2
                        )
                    except Exception:
                        pass

            if gt_digest_invalid or (
                str(gt_cur.get("threat_level") or "NORMAL").upper() == "NORMAL"
                and not gt_digest
                and has_fresh_calm_news
                and gt_age_min >= 60.0
            ):
                _reset_and_persist_normal_geopolitical_eval(
                    "Calm geopolitical and energy climate. No systemic shock detected.",
                    "",
                )
            elif all_valid_digest_aged_90m and (
                str(gt_cur.get("threat_level") or "").upper() in ("ELEVATED", "CRITICAL_SHOCK")
                or gt_age_min >= 90.0
            ):
                is_elev = str(gt_cur.get("threat_level") or "").upper() in ("ELEVATED", "CRITICAL_SHOCK")
                _reset_and_persist_normal_geopolitical_eval(
                    (
                        "Geopolitical threat de-escalated to NORMAL: no fresh high-severity macro headlines in the last 90 minutes."
                        if is_elev
                        else "Calm geopolitical and energy climate: no fresh high-severity macro headlines in the last 90 minutes."
                    ),
                    " | ".join(valid_digest_titles[:12]),
                )
        if (
            str(gt_cur.get("threat_level") or "").upper() in ("ELEVATED", "CRITICAL_SHOCK")
            and (safe_float(gt_cur.get("age_min"), 0.0) or 0.0) >= 90.0
            and breaking_news
            and not any(n.get("is_high_severity") for n in breaking_news)
        ):
            gt_cur["live_deescalation_note"] = (
                f"NO_FRESH_HIGH_SEV_HEADLINES_90M (shield age={gt_cur.get('age_min')}m; 0/{len(breaking_news)} live headlines high-sev)"
            )
        elif (
            "no fresh high-severity macro headlines in the last 90 minutes"
            in str(gt_cur.get("summary") or "").lower()
            and (
                fresh_hs_news := [
                    n
                    for n in breaking_news
                    if _parse_bool_flag(n.get("is_high_severity"), False)
                    and safe_nonneg_float(n.get("age_min"), default=999.0) < 90.0
                ]
            )
        ):
            gt_cur["summary"] = (
                f"Geopolitical threat level {str(gt_cur.get('threat_level') or 'NORMAL').upper()}: "
                f"monitoring {len(fresh_hs_news)} fresh high-severity macro headline(s) (<90m old); no acute systemic shock brake triggered."
            )
        gt_ts_f = safe_pos_float(gt_cur.get("ts"), default=0.0) or 0.0
        if gt_ts_f > 0.0 and valid_news:
            gt_dig_s = str(gt_cur.get("headlines_digest") or "").strip()
            gt_dig_low = gt_dig_s.lower()
            gt_dig_titles = split_headlines_digest(gt_dig_s) if gt_dig_s else []
            fresh_unassessed_hs: List[Dict[str, Any]] = []
            for n in valid_news:
                if not _parse_bool_flag(n.get("is_high_severity"), False):
                    continue
                n_ts = safe_pos_float(n.get("first_seen_ts"), default=0.0) or 0.0
                if n_ts <= gt_ts_f or (now_ts - n_ts) >= 5400.0:
                    continue
                n_title = str(n.get("title") or "").strip()
                if not n_title:
                    continue
                n_clean = _strip_headline_publisher_suffix(n_title).strip()
                if n_title.lower() in gt_dig_low or (n_clean and n_clean.lower() in gt_dig_low):
                    continue
                if gt_dig_titles and any(headlines_are_semantic_duplicates(n_title, dt) for dt in gt_dig_titles):
                    continue
                fresh_unassessed_hs.append(n)
            if fresh_unassessed_hs:
                first_unassessed = _strip_headline_publisher_suffix(
                    str(fresh_unassessed_hs[0].get("title") or "")
                ).strip()
                snippet = (
                    f"{first_unassessed[:90].rsplit(' ', 1)[0]}..."
                    if len(first_unassessed) > 90
                    else first_unassessed
                )
                gt_cur["live_unassessed_high_sev_note"] = (
                    f"UNASSESSED_FRESH_HIGH_SEV ({len(fresh_unassessed_hs)} headline(s) arrived after shield eval: {snippet})"
                )

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        now_ts: float = 0.0,
        read_json_fn: Any = None,
        scope: str = "",
        allow_network: bool = True,
        persist: bool = True,
        **kwargs: Any,
    ) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
        cache_dir = cachedb_dir
        now_ts, _rj, _rj_any = _resolve_macro_collect_inputs(now_ts, read_json_fn)

        def _with_age_min(payload: Dict[str, Any], fpath: str, *, is_fear_greed: bool = False) -> Dict[str, Any]:
            enriched = dict(payload)
            ref_ts = extract_timestamp_s(enriched)
            eff_path = str(resolve_cachedb_filepath(fpath))
            if os.path.exists(eff_path):
                try:
                    mtime = os.path.getmtime(eff_path)
                    if ref_ts is None or (
                        is_fear_greed
                        and math.trunc(ref_ts) % 86400 == 0
                        and (now_ts - ref_ts) > 3600.0
                        and ref_ts < mtime <= now_ts + 60.0
                    ):
                        ref_ts = mtime
                except OSError:
                    pass
            if ref_ts is not None:
                enriched.setdefault("age_min", round(max(0.0, now_ts - ref_ts) / 60.0, 1))
            return enriched

        macro_sensors: Dict[str, Optional[Dict[str, Any]]] = {}
        for attr, group_key, fname in self._MACRO_SENSOR_CACHE_SPECS:
            data, s_path = self._read_cached_dict(cache_dir, group_key, fname, _rj)
            if data is not None:
                enriched = _with_age_min(data, s_path, is_fear_greed=(attr == "fear_greed"))
                if attr == "market_breadth":
                    self._normalize_market_breadth_movers(enriched)
                if attr == "macro_benchmarks":
                    enriched = self._normalize_macro_benchmarks_items(enriched, cache_dir=cache_dir, now_ts=now_ts)
                macro_sensors[attr] = enriched
        if allow_network and read_json_fn in (None, read_json_recovering, read_json_cached) and is_live_network_backfill_enabled():
            self._backfill_stale_macro_sensors(
                macro_sensors, cache_dir=cache_dir, now_ts=now_ts, persist=persist, with_age_fn=_with_age_min
            )

        breaking_news, valid_news, stale_valid_sev_titles = self._collect_and_rank_breaking_news(
            cache_dir, now_ts=now_ts, scope=scope, read_json_any_fn=_rj_any
        )
        self._reconcile_geopolitical_deescalation(
            macro_sensors.get("geopolitical_threat"),
            valid_news,
            breaking_news,
            stale_valid_sev_titles,
            cache_dir=cache_dir,
            now_ts=now_ts,
            allow_network=allow_network,
            persist=persist,
            read_json_fn=read_json_fn,
        )
        cal_snap, cal_path = self._read_cached_dict(
            cache_dir, "macro_calendar_files", CACHE_MACRO_CALENDAR_FILENAME, _rj
        )
        if allow_network and read_json_fn in (None, read_json_recovering, read_json_cached) and is_live_network_backfill_enabled():
            derived_itvs = _mss.resolve_derived_telemetry_intervals()
            cal_itv_sec = derived_itvs["macro_calendar_interval_sec"]
            cal_snap = self._maybe_backfill_cached_macro_snapshot(
                cal_snap,
                cal_path,
                now_ts=now_ts,
                max_age_sec=cal_itv_sec,
                collect_fn=lambda: collect_macro_calendar_snapshot(
                    cache_dir,
                    now_ts=now_ts,
                    timeout_sec=3.0,
                    persist=persist,
                    skip_if_fresh_sec=resolve_macro_fresh_guard_sec(cal_itv_sec),
                ),
            )
            if str(scope or "").strip().lower() not in ("equities", "equity"):
                etf_itv_sec = derived_itvs["etf_flows_interval_sec"]
                etf_cur = macro_sensors.get("etf_flows")
                etf_path = self._resolve_cache_path(cache_dir, CACHE_ETF_FLOWS_FILENAME)
                live_etf = self._maybe_backfill_cached_macro_snapshot(
                    etf_cur,
                    etf_path,
                    now_ts=now_ts,
                    max_age_sec=etf_itv_sec,
                    collect_fn=lambda: collect_spot_etf_flows_snapshot(
                        cache_dir,
                        now_ts=now_ts,
                        timeout_sec=4.0,
                        persist=persist,
                        skip_if_fresh_sec=resolve_macro_fresh_guard_sec(etf_itv_sec),
                    ),
                )
                if isinstance(live_etf, dict) and live_etf and live_etf is not etf_cur:
                    macro_sensors["etf_flows"] = _with_age_min(live_etf, etf_path)
        macro_sensors["macro_calendar"] = resolve_macro_economic_calendar(
            now_ts=now_ts,
            breaking_news=valid_news or breaking_news,
            calendar_snapshot=cal_snap,
        )
        return macro_sensors, breaking_news


class InitialPricesAndTrendsCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) building initial prices_and_trends from CacheCurrentPriceManager & CacheShortTrendManager."""

    cls_name = "InitialPricesAndTrendsCollector"
    COLLECTOR_ID = "macro_initial_prices_trends"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        prices_data: Any = None,
        trends_data: Any = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        return _harvest_initial_prices_and_trends_impl(
            cachedb_dir,
            prices_data,
            trends_data,
            read_json_fn,
        )


_VENUE_FEE_RESERVE_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    ("binance", "binance_bnb_fee_reserve_usdc", "BNB", "held_bnb_usdc", "bnb_price_usdc"),
)


class PortfolioAndVenueCyclesCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) harvesting CacheAssetValueManager summary, venue cycles, Kraken xStocks, and cross-market context."""

    cls_name = "PortfolioAndVenueCyclesCollector"
    COLLECTOR_ID = "macro_portfolio_venue_cycles"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        workspace_dir: str = "",
        logs_dirs: Optional[List[str]] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
        watcher_price_items: Any = None,
        inst_by_sym: Optional[Dict[str, Any]] = None,
        read_json_fn: Any = None,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        eff_ps, eff_pt, eff_ms, eff_inst = coerce_dict_maps(
            portfolio_summary, prices_and_trends, microstructure, inst_by_sym
        )
        _rj = read_json_fn or read_json_cached
        _mc.ensure_cacheprice_classes_bound()
        av_cls = _mc.CacheAssetValueManager
        last_av = read_last_jsonl_dict(os.path.join(cachedb_dir, av_cls.DEFAULT_FILENAME), tail_bytes=2048)
        if av_summary := av_cls.extract_asset_value_summary(last_av, now_ts=now_ts):
            eff_ps.update(av_summary)
        fee_st = read_persisted_venue_fee_schedule(cachedb_dir, read_json_fn=_rj)
        if isinstance(fee_st, dict) and isinstance(fee_venues := fee_st.get("venues"), dict):
            for fee_venue, ps_key, asset_k, held_usd_k, px_k in _VENUE_FEE_RESERVE_SPECS:
                if not isinstance(v_fee_st := fee_venues.get(fee_venue), dict):
                    continue
                res_usdc = safe_pos_float(v_fee_st.get(held_usd_k), default=None)
                if (res_px := safe_pos_float(v_fee_st.get(px_k), default=None)) is not None:
                    v_bal_path = resolve_venue_balance_cache_path(cachedb_dir, fee_venue)
                    v_bal_doc = _rj(v_bal_path) if v_bal_path else None
                    v_items = (
                        v_bal_doc.get("items")
                        if isinstance(v_bal_doc, dict) and isinstance(v_bal_doc.get("items"), dict)
                        else {}
                    )
                    if isinstance(res_row := v_items.get(asset_k), dict):
                        res_qty = (safe_nonneg_float(res_row.get("free"), 0.0) or 0.0) + (
                            safe_nonneg_float(res_row.get("locked"), 0.0) or 0.0
                        )
                        if res_qty > 0.0:
                            res_usdc = res_qty * res_px
                if res_usdc is not None:
                    eff_ps[ps_key] = round(res_usdc, 2)

        kraken_cycles, hyperliquid_cycles, all_t212_cycles = harvest_venue_cycles(
            workspace_dir, eff_pt, build_cycle_fallback_prices(eff_ms, watcher_price_items), _rj,
        )
        if kraken_cycles:
            eff_ps["kraken_cycles"] = kraken_cycles
        if hyperliquid_cycles:
            eff_ps["hyperliquid_cycles"] = hyperliquid_cycles

        has_active_t212 = any(
            is_equity_venue(getattr(inst, "provider_name", "")) for inst in eff_inst.values()
        ) or any(
            isinstance(p, dict) and is_equity_venue(p.get("venue"))
            for p in (eff_ps.get("tradable_positions") or ())
        )
        t212_cycles = all_t212_cycles
        if has_active_t212 and t212_cycles:
            eff_ps["t212_cycles"] = t212_cycles
        if xst_summary := harvest_kraken_xstocks(
            os.path.join(workspace_dir, "kraken"), eff_pt, all_t212_cycles, _rj,
            kraken_cycles=kraken_cycles, cachedb_dir=cachedb_dir,
        ):
            eff_ps["kraken_xstocks"] = xst_summary
        if cross_market_ctx := build_cross_market_context(eff_pt, eff_ms, all_t212_cycles):
            eff_ps["cross_market_context"] = cross_market_ctx

        # Build Hyperliquid read-only summary directly from SSOT caches (cache_balance_hyperliquid.json + prices)
        hl_bal_path = resolve_venue_balance_cache_path(cachedb_dir, "hyperliquid")
        hl_bal_doc = _rj(hl_bal_path) if hl_bal_path else None
        hl_items = (
            hl_bal_doc.get("items")
            if isinstance(hl_bal_doc, dict) and isinstance(hl_bal_doc.get("items"), dict)
            else {}
        )
        if hl_items:
            hl_hype_row = hl_items.get("HYPE") if isinstance(hl_items.get("HYPE"), dict) else {}
            hl_usdc_row = hl_items.get("USDC") if isinstance(hl_items.get("USDC"), dict) else {}
            hl_qty = (safe_nonneg_float(hl_hype_row.get("free"), 0.0) or 0.0) + (
                safe_nonneg_float(hl_hype_row.get("locked"), 0.0) or 0.0
            )
            hl_cash_free = safe_nonneg_float(hl_usdc_row.get("free"), 0.0) or 0.0
            hl_cash_locked = safe_nonneg_float(hl_usdc_row.get("locked"), 0.0) or 0.0
            hl_cash_total = hl_cash_free + hl_cash_locked
            hl_px_entry = eff_pt.get("HYPEUSDC") or eff_pt.get("HYPEUSD") or {}
            hl_px = safe_pos_float(
                hl_px_entry.get("price") if isinstance(hl_px_entry, dict) else hl_px_entry,
                default=0.0,
            ) or 0.0
            hl_val = hl_qty * hl_px
            hl_ro: Dict[str, Any] = {
                "engine": "HL HYPE (long-hold)",
                "paper": False,
                "qty": round(hl_qty, 4),
                "value_usd": round(hl_val, 2),
                "cash_usd": round(hl_cash_total, 2),
                "total_equity_usd": round(hl_val + hl_cash_total, 2),
            }
            if hl_cash_locked > 0.0:
                hl_ro["free_cash_usd"] = round(hl_cash_free, 2)
                hl_ro["locked_quote_usd"] = round(hl_cash_locked, 2)
            hl_ts_s = first_epoch_sec(extract_timestamp_s(hl_bal_doc), allow_iso=True)
            eff_now_s = resolve_now_ts(now_ts)
            if hl_ts_s is not None and eff_now_s is not None:
                hl_ro["snapshot_age_min"] = round(max(0.0, eff_now_s - hl_ts_s) / 60.0, 1)
            eff_ps["hyperliquid_readonly"] = hl_ro

        return kraken_cycles, hyperliquid_cycles, t212_cycles


class MarketAndPortfolioSnapshotOrchestrator(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) orchestrating full market, macro, microstructure, and portfolio telemetry into TelemetrySensorSnapshot."""

    cls_name = "MarketAndPortfolioSnapshotOrchestrator"
    COLLECTOR_ID = "macro_market_portfolio_snapshot"

    @staticmethod
    def _enrich_ai_intent_and_guard_telemetry(
        snapshot: Any,
        intents_data: Any,
        *,
        scope: str,
        now_ts: float,
        workspace_dir: str,
        recent_orders: Any,
        open_orders_list: List[Dict[str, Any]],
        eff_verified_venues: Optional[set[str]],
        eff_verified_symbols: Optional[set[tuple[str, str]]] = None,
        price_bars_by_symbol: Optional[Dict[str, List[List[Any]]]] = None,
        cache_dir: str = "",
        intent_gateway: Any = None,
        persist: bool = False,
    ) -> None:
        if isinstance(intents_data, list) and intents_data:
            if recent_intents_out := summarize_recent_ai_intents(
                intents_data,
                scope,
                now_ts,
                extract_intent_notional,
                recent_orders=recent_orders,
                open_orders=open_orders_list,
                verified_open_venues=eff_verified_venues,
                verified_open_symbols=eff_verified_symbols,
            ):
                snapshot.portfolio_summary["recent_ai_intents"] = recent_intents_out
            if (last_exec_status := build_last_cycle_execution_status(
                intents_data,
                scope,
                now_ts,
                extract_intent_notional,
                recent_orders=recent_orders,
                open_orders=open_orders_list,
                verified_open_venues=eff_verified_venues,
                verified_open_symbols=eff_verified_symbols,
            )) is not None:
                snapshot.portfolio_summary["last_cycle_execution_status"] = last_exec_status
            if active_plans_out := summarize_active_ai_trade_plans(
                intents_data,
                scope,
                now_ts,
                extract_intent_notional,
                tradable_positions=snapshot.portfolio_summary.get("tradable_positions"),
                prices_and_trends=snapshot.prices_and_trends,
                recent_orders=recent_orders,
                open_orders=open_orders_list,
                verified_open_venues=eff_verified_venues,
                verified_open_symbols=eff_verified_symbols,
                price_bars_by_symbol=price_bars_by_symbol,
            ):
                snapshot.portfolio_summary["active_ai_trade_plans"] = active_plans_out
            try:
                cb_cfg = parse_intent_circuit_breaker_config(load_key_value_conf(workspace_dir))
                sc_prices = build_intent_live_price_map(
                    snapshot.prices_and_trends,
                    snapshot.portfolio_summary.get("tradable_positions"),
                )
                sc_fee_resolver = build_persisted_round_trip_fee_resolver(cache_dir or None, now_ts=now_ts)
                ai_scorecard = evaluate_ai_decision_scorecard(
                    intents_data,
                    now_ts=now_ts,
                    live_prices=sc_prices,
                    window_trades=cb_cfg["scorecard_window_trades"],
                    max_loss_streak=cb_cfg["scorecard_max_loss_streak"],
                    cooldown_hours=cb_cfg["scorecard_cooldown_hours"],
                    scope=scope,
                    fee_resolver=sc_fee_resolver,
                )
                if ai_scorecard.get("count", 0) > 0:
                    snapshot.portfolio_summary["ai_decision_scorecard"] = ai_scorecard
                newly_resolved_plans: Dict[str, str] = {}
                ai_cal = evaluate_ai_calibration_feedback(
                    intents_data,
                    now_ts=now_ts,
                    live_prices=sc_prices,
                    price_bars_by_symbol=price_bars_by_symbol,
                    lookback_days=7.0,
                    scope=scope,
                    newly_resolved_out=newly_resolved_plans,
                    fee_resolver=sc_fee_resolver,
                )
                if ai_cal.get("filled_total", 0) > 0 or ai_cal.get("venue_tca"):
                    snapshot.portfolio_summary["ai_calibration_feedback"] = ai_cal
                if (
                    newly_resolved_plans
                    and intent_gateway is not None
                    and callable(persist_plans_fn := getattr(intent_gateway, "persist_resolved_trade_plan_statuses", None))
                ):
                    persist_plans_fn(newly_resolved_plans, now_ts=now_ts)
            except Exception:
                pass

    @staticmethod
    def _apply_macro_scope_and_risk_overlays(
        snapshot: Any,
        *,
        scope: str,
        now_ts: float,
        inst_by_sym: Dict[str, Any],
        t212_cycles: Any,
        workspace_dir: str = "",
    ) -> None:
        if isinstance(mb_snap_pre := getattr(snapshot, "macro_benchmarks", None), dict) and mb_snap_pre:
            btc_chg_24h_live = next(
                (
                    c24
                    for k in ("BTCUSDC", "BTCUSDT", "BTCUSD", "XBTUSD", "XXBTZUSD", "BTC")
                    if isinstance(b_pt := casefold_get(snapshot.prices_and_trends, k), dict)
                    and (c24 := safe_float(b_pt.get("chg_24h_pct"), default=None)) is not None
                ),
                None,
            )
            if btc_chg_24h_live is not None and abs(btc_chg_24h_live) > 1e-6:
                cross_reg_live = compute_cross_asset_regime(mb_snap_pre, btc_chg_24h_pct=btc_chg_24h_live)
                if cross_reg_live != mb_snap_pre.get("cross_asset_regime"):
                    mb_snap_pre["cross_asset_regime"] = cross_reg_live
                    if summary_live := format_macro_benchmarks_summary_line(mb_snap_pre):
                        mb_snap_pre["summary_line"] = summary_live
        enrich_sentiment_advisor_live_flow_divergence(
            getattr(snapshot, "sentiment_advisor", None),
            snapshot.microstructure,
            macro_benchmarks=getattr(snapshot, "macro_benchmarks", None),
        )
        snapshot.prices_and_trends, snapshot.microstructure = apply_asset_scope_filter(
            snapshot.prices_and_trends, snapshot.microstructure, snapshot.portfolio_summary, scope, inst_by_sym, t212_cycles
        )
        reconcile_live_venue_equity_summary(snapshot.portfolio_summary)
        if isinstance(mb_snap := getattr(snapshot, "macro_benchmarks", None), dict) and mb_snap and snapshot.portfolio_summary:
            snapshot.portfolio_summary["macro_benchmarks"] = mb_snap
            if isinstance(cmc := snapshot.portfolio_summary.get("cross_market_context"), dict):
                cmc["macro_benchmarks"] = mb_snap
        if snapshot.portfolio_summary:
            snapshot.portfolio_summary.setdefault("open_limit_orders", [])
            if isinstance(snapshot.portfolio_summary.get("open_limit_orders"), list) and snapshot.portfolio_summary["open_limit_orders"]:
                ob_stale_sec_ssot = (
                    _mss.resolve_derived_telemetry_intervals()["orderbook_interval_sec"]
                    * _mss.TELEMETRY_SOURCE_STALE_INTERVAL_MULT
                )
                enrich_open_limit_orders_tca(
                    snapshot.portfolio_summary["open_limit_orders"],
                    snapshot.prices_and_trends,
                    snapshot.microstructure,
                    now_ts=now_ts,
                    ob_stale_sec=ob_stale_sec_ssot,
                    tradable_positions=snapshot.portfolio_summary.get("tradable_positions"),
                )
            if risk_eng := compute_enterprise_portfolio_risk_metrics(
                snapshot.portfolio_summary, snapshot.prices_and_trends, snapshot.microstructure
            ):
                if workspace_dir:
                    try:
                        live_nav_pt: Optional[Dict[str, Any]] = None
                        if (live_nav_u := safe_pos_float(risk_eng.get("unified_nav_usd"), default=None)) is not None:
                            live_venues = dict(risk_eng.get("venue_nav_breakdown_usd") or {})
                            if str(scope or "all").strip().lower() not in ("equities", "equity"):
                                t_pos_list = snapshot.portfolio_summary.get("tradable_positions") or ()
                                for fee_venue, ps_key, asset_k, _, _ in _VENUE_FEE_RESERVE_SPECS:
                                    res_u = safe_pos_float(snapshot.portfolio_summary.get(ps_key), default=None)
                                    if res_u is None or fee_venue not in live_venues:
                                        continue
                                    has_res_pos = any(
                                        isinstance(p, dict)
                                        and normalize_venue_name(p.get("venue")) == fee_venue
                                        and canonical_base_asset(p.get("symbol") or p.get("base") or "") == asset_k
                                        for p in t_pos_list
                                    )
                                    if not has_res_pos:
                                        live_venues[fee_venue] = round(
                                            (safe_nonneg_float(live_venues.get(fee_venue), 0.0) or 0.0) + res_u, 2
                                        )
                                        live_nav_u = round(live_nav_u + res_u, 2)
                            live_nav_pt = {
                                "ts": now_ts,
                                "nav_usd": live_nav_u,
                                "invested_usd": safe_nonneg_float(risk_eng.get("invested_exposure_usd"), default=0.0),
                                "free_cash_usd": safe_nonneg_float(risk_eng.get("global_free_cash_usd"), default=0.0),
                                "locked_quote_usd": safe_nonneg_float(risk_eng.get("locked_quote_usd"), default=0.0),
                                "venues_usd": live_venues,
                            }
                        nav_dd = load_and_evaluate_nav_drawdown(
                            workspace_dir,
                            now_ts=now_ts,
                            include_live_point=False,
                            scope=scope,
                            current_point=live_nav_pt,
                        )
                        if nav_dd.get("sample_count", 0) >= 2:
                            risk_eng["daily_nav_drawdown_pct"] = nav_dd["daily_nav_drawdown_pct"]
                            risk_eng["daily_nav_return_pct"] = nav_dd["daily_nav_return_pct"]
                    except Exception:
                        pass
                snapshot.portfolio_summary["portfolio_risk_engine"] = risk_eng

    @staticmethod
    def _stamp_snapshot_domain_timestamps(
        snapshot: Any,
        *,
        scope: str,
        allow_network: bool,
        persist: bool,
        now_ts: float,
        cache_dir: str,
        read_json_fn: Any,
        macro_sensors: Dict[str, Any],
        prices_data: Any,
        cur_p_items: Dict[str, Any],
        trends_data: Any,
        inst_by_sym: Dict[str, Any],
        t212_cycles: Any,
    ) -> None:
        scope_label = str(scope or "all").strip().lower() or "all"
        scope_norm = "equities" if scope_label in ("equities", "equity") else scope_label
        snap_source = "cachedb_read_only" if not allow_network and not persist else "live_harvest"
        t212_sym_set = {str(s).upper() for s in (t212_cycles or ())} | {
            str(s).split(":", 1)[-1].upper()
            for s, iv in (inst_by_sym or {}).items()
            if is_equity_venue(getattr(iv, "provider_name", None))
        }
        scoped_pt_syms = {
            str(k).strip().upper()
            for k in (snapshot.prices_and_trends if isinstance(snapshot.prices_and_trends, dict) else ())
            if str(k).strip()
        }
        _meta_keys_up = {"TS", "TIMESTAMP", "UPDATED_AT", "ITEMS", "FETCHTIME", "DYNAMIC"}
        eff_ws_ts = os.path.dirname(os.path.abspath(str(cache_dir).rstrip("/\\"))) if cache_dir else ""
        disabled_ts_syms = disabled_registry_symbols(eff_ws_ts) if eff_ws_ts else frozenset()

        def _is_eq_sym(sym_up: str) -> bool:
            return has_equity_suffix(sym_up) or sym_up in t212_sym_set

        def _sym_in_scope(sym_key: Any) -> bool:
            k_up = str(sym_key or "").strip().upper()
            if not k_up or k_up in _meta_keys_up:
                return False
            if k_up in disabled_ts_syms and k_up not in scoped_pt_syms:
                return False
            if scope_norm in ("equities", "crypto") and _is_eq_sym(k_up) != (scope_norm == "equities"):
                return False
            if scoped_pt_syms:
                return k_up in scoped_pt_syms or any(
                    c.upper() in scoped_pt_syms for c in cache_symbol_variants(k_up, include_base=False)
                )
            return True

        def _vb_venue_in_scope(v_raw: Any) -> bool:
            v_n = normalize_venue_name(v_raw)
            if not v_n:
                return scope_norm not in ("equities", "crypto")
            if scope_norm in ("equities", "crypto"):
                return is_equity_venue(v_n) == (scope_norm == "equities")
            return True

        def _collect_scoped_map_timestamps(
            items_map: Any,
            envelope_doc: Any,
            item_ts_fn: Callable[[str, Any], Optional[float]],
            *,
            exclude_eq: bool = False,
        ) -> tuple[List[float], Optional[float]]:
            has_in_scope = False
            has_out_of_scope = False
            item_ts: List[float] = []
            if isinstance(items_map, dict):
                for k_raw, v_raw in items_map.items():
                    k_up = str(k_raw or "").strip().upper()
                    if not k_up or k_up in _meta_keys_up or (exclude_eq and _is_eq_sym(k_up)):
                        continue
                    if _sym_in_scope(k_up):
                        has_in_scope = True
                        ts_v = item_ts_fn(k_raw, v_raw)
                        if ts_v is not None and ts_v > 0.0:
                            item_ts.append(ts_v)
                    else:
                        has_out_of_scope = True
            top_ts = (
                extract_timestamp_s(envelope_doc)
                if isinstance(envelope_doc, dict) and (has_in_scope or not has_out_of_scope)
                else None
            )
            return item_ts, top_ts

        cp_source_items = (
            cur_p_items
            if isinstance(cur_p_items, dict) and cur_p_items
            else (prices_data if isinstance(prices_data, dict) else {})
        )
        cp_ft_map = (
            prices_data.get("fetchtime")
            if isinstance(prices_data, dict) and isinstance(prices_data.get("fetchtime"), dict)
            else {}
        )

        def _cp_ts(k_cp: str, v_cp: Any) -> Optional[float]:
            cp_res = extract_current_price_timestamp_sec(v_cp, cp_ft_map.get(k_cp))
            return cp_res[0] if cp_res[1] and cp_res[0] > 0.0 else None

        cp_item_ts, prices_top_ts = _collect_scoped_map_timestamps(cp_source_items, prices_data, _cp_ts)
        if scope_norm in ("equities", "crypto") and cp_item_ts:
            snap_ts: Optional[float] = max(cp_item_ts)
        else:
            snap_ts = prices_top_ts
            if snap_ts is None and cp_item_ts:
                snap_ts = max(cp_item_ts)
        active_macro_sensors = {
            m_name: m_val
            for m_name in macro_sensors
            if m_name not in SYNTHETIC_MACRO_SENSOR_KEYS and isinstance(m_val := getattr(snapshot, m_name, None), dict)
        }
        if snap_ts is None:
            sensor_ts_candidates = [
                ts_v
                for s_val in active_macro_sensors.values()
                if (ts_v := extract_timestamp_s(s_val)) is not None
            ]
            snap_ts = max(sensor_ts_candidates) if sensor_ts_candidates else now_ts
        domain_ts: Dict[str, float] = {}
        prices_newest_ts = max(cp_item_ts) if cp_item_ts else prices_top_ts
        if prices_newest_ts is not None and prices_newest_ts > 0.0:
            domain_ts["prices_newest"] = prices_newest_ts
        us_rth_open = is_us_rth_open(now_ts)
        cp_oldest_pool = (
            (_collect_scoped_map_timestamps(cp_source_items, None, _cp_ts, exclude_eq=True)[0] or cp_item_ts)
            if (scope_norm == "all" and not us_rth_open)
            else cp_item_ts
        )
        if len(cp_oldest_pool) > 1:
            domain_ts["prices_oldest"] = min(cp_oldest_pool)
        micro_snap = getattr(snapshot, "microstructure", None)
        if isinstance(micro_snap, dict) and micro_snap:
            raw_tp = (snapshot.portfolio_summary or {}).get("tradable_positions") if isinstance(snapshot.portfolio_summary, dict) else None
            tp_ob_syms = (
                {
                    str(p_v.get("symbol") or "").strip().upper()
                    for p_v in raw_tp
                    if isinstance(p_v, dict) and p_v.get("symbol") and _vb_venue_in_scope(p_v.get("venue"))
                }
                if isinstance(raw_tp, list)
                else (
                    {
                        str(p_k).split(":", 1)[-1].strip().upper()
                        for p_k, p_v in raw_tp.items()
                        if isinstance(p_v, dict) and _vb_venue_in_scope(p_v.get("venue"))
                    }
                    if isinstance(raw_tp, dict)
                    else set()
                )
            )
            inst_ob_syms = (
                {
                    str(s_inst).split(":", 1)[-1].strip().upper()
                    for s_inst, iv_inst in (inst_by_sym or {}).items()
                    if _vb_venue_in_scope(getattr(iv_inst, "provider_name", None))
                }
                | tp_ob_syms
                | (t212_sym_set if scope_norm != "crypto" else set())
            )

            def _is_active_ob_sym(sym_up: str) -> bool:
                if not inst_ob_syms:
                    return True
                return sym_up in inst_ob_syms or any(
                    c.upper() in inst_ob_syms for c in cache_symbol_variants(sym_up, include_base=False)
                )

            def _collect_deduped_micro_ts(
                ts_keys: tuple[str, ...], *, exclude_eq: bool = False, active_only: bool = True
            ) -> List[float]:
                by_sym: Dict[str, float] = {}
                for mk, mv in micro_snap.items():
                    k_up = str(mk or "").strip().upper()
                    if not k_up or k_up in _meta_keys_up or (exclude_eq and _is_eq_sym(k_up)) or not _sym_in_scope(k_up):
                        continue
                    if active_only and not _is_active_ob_sym(k_up):
                        continue
                    if not isinstance(mv, dict):
                        continue
                    raw_t = next((mv.get(tk) for tk in ts_keys if mv.get(tk) is not None), None)
                    ts_m = normalize_epoch_sec(raw_t, default=None)
                    if ts_m is not None and ts_m > 0.0:
                        by_sym[k_up] = max(by_sym.get(k_up, 0.0), ts_m)
                return list(by_sym.values())

            ob_keys = ("_ob_raw_ts", "_ob_ts")
            ob_item_ts = _collect_deduped_micro_ts(ob_keys, active_only=True) or _collect_deduped_micro_ts(ob_keys, active_only=False)
            if ob_item_ts:
                domain_ts["orderbook_newest"] = max(ob_item_ts)
                ob_oldest_pool = (
                    (
                        _collect_deduped_micro_ts(ob_keys, exclude_eq=True, active_only=True)
                        or _collect_deduped_micro_ts(ob_keys, exclude_eq=True, active_only=False)
                        or ob_item_ts
                    )
                    if (scope_norm == "all" and not us_rth_open)
                    else ob_item_ts
                )
                if len(ob_oldest_pool) > 1:
                    domain_ts["orderbook_oldest"] = min(ob_oldest_pool)
            if scope_norm != "equities":
                deriv_ts_vals = _collect_deduped_micro_ts(("_deriv_ts",), exclude_eq=True, active_only=True)
                if deriv_ts_vals:
                    domain_ts["derivatives_oldest"] = min(deriv_ts_vals)
                whale_ts_vals = _collect_deduped_micro_ts(("_whale_ts",), exclude_eq=True, active_only=True)
                if whale_ts_vals:
                    domain_ts["whale_oldest"] = min(whale_ts_vals)
        st_items_d = trends_data.get("items") if isinstance(trends_data, dict) and isinstance(trends_data.get("items"), dict) else trends_data
        st_ft_map = (
            trends_data.get("fetchtime")
            if isinstance(trends_data, dict) and isinstance(trends_data.get("fetchtime"), dict)
            else {}
        )

        def _st_ts(k_st: str, v_st: Any) -> Optional[float]:
            item_t = extract_timestamp_s(v_st) if isinstance(v_st, dict) else None
            ft_t = normalize_epoch_sec(st_ft_map.get(k_st), default=None)
            if item_t is not None and item_t > 0.0 and ft_t is not None and ft_t > 0.0:
                return max(item_t, ft_t)
            if item_t is not None and item_t > 0.0:
                return item_t
            return ft_t if (ft_t is not None and ft_t > 0.0) else None

        st_ts_vals, st_top_ts = _collect_scoped_map_timestamps(st_items_d, trends_data, _st_ts)
        st_oldest_pool = (
            (_collect_scoped_map_timestamps(st_items_d, None, _st_ts, exclude_eq=True)[0] or st_ts_vals)
            if (scope_norm == "all" and not us_rth_open)
            else st_ts_vals
        )
        if st_oldest_pool:
            domain_ts["short_trends"] = min(st_oldest_pool)
        elif st_top_ts is not None and st_top_ts > 0.0:
            domain_ts["short_trends"] = st_top_ts
        if vb_doc := load_venue_balances_state(cache_dir, read_json_fn)[1]:
            inst_vb_syms = {
                str(s_inst).split(":", 1)[-1].strip().upper()
                for s_inst, iv_inst in (inst_by_sym or {}).items()
                if _vb_venue_in_scope(getattr(iv_inst, "provider_name", None))
            }
            verified_by_venue: Dict[str, float] = {}
            fallback_by_venue: Dict[str, List[float]] = {}
            for bk, bv in vb_doc.items():
                if not isinstance(bk, str):
                    continue
                if bk.startswith("open_orders_verified_ts:"):
                    v_part = normalize_venue_name(bk.split(":", 1)[1])
                    if v_part and _vb_venue_in_scope(v_part) and (ts_vb := normalize_epoch_sec(bv, default=None)) is not None and ts_vb > 0.0:
                        verified_by_venue[v_part] = max(verified_by_venue.get(v_part, 0.0), ts_vb)
                elif isinstance(bv, dict) and (ts_vb := extract_timestamp_s(bv)) is not None and ts_vb > 0.0:
                    bk_parts = bk.split(":")
                    v_cand = normalize_venue_name(
                        bv.get("venue")
                        or (
                            bk_parts[1]
                            if len(bk_parts) >= 3 and bk_parts[0] in ("pos", "cash", "breakdown")
                            else (bk_parts[0] if len(bk_parts) >= 2 else "")
                        )
                    )
                    sym_cand = str(
                        bv.get("symbol") or (bk_parts[2] if len(bk_parts) >= 3 and bk_parts[0] == "pos" else "")
                    ).strip().upper()
                    if v_cand:
                        if not _vb_venue_in_scope(v_cand):
                            continue
                    elif sym_cand:
                        is_eq_cand = _is_eq_sym(sym_cand)
                        if scope_norm in ("equities", "crypto") and is_eq_cand != (scope_norm == "equities"):
                            continue
                        v_cand = "t212" if is_eq_cand else "crypto"
                    elif scope_norm in ("equities", "crypto"):
                        continue
                    else:
                        v_cand = "unknown"
                    if sym_cand and inst_vb_syms and sym_cand not in inst_vb_syms:
                        if not any(c.upper() in inst_vb_syms for c in cache_symbol_variants(sym_cand, include_base=False)):
                            continue
                    fallback_by_venue.setdefault(v_cand, []).append(ts_vb)

            def _vb_pool(exclude_eq: bool = False) -> List[float]:
                return [
                    ts_v for v_k, ts_v in verified_by_venue.items() if not (exclude_eq and is_equity_venue(v_k))
                ] + [
                    max(ts_list)
                    for v_k, ts_list in fallback_by_venue.items()
                    if v_k not in verified_by_venue and ts_list and not (exclude_eq and is_equity_venue(v_k))
                ]

            vb_ts_vals = (_vb_pool(exclude_eq=True) if (scope_norm == "all" and not us_rth_open) else []) or _vb_pool()
            if vb_ts_vals:
                domain_ts["venue_balances"] = min(vb_ts_vals)
        for m_name, m_val in active_macro_sensors.items():
            m_ts = normalize_epoch_sec(m_val.get("ts"), default=None) or extract_timestamp_s(m_val)
            if m_ts is not None and m_ts > 0.0:
                domain_ts[m_name] = m_ts
        snap_age_sec = round(max(0.0, now_ts - snap_ts), 3)
        if not getattr(snapshot, "snapshot_id", ""):
            snapshot.snapshot_id = f"market_portfolio:{scope_label}:{round(snap_ts * 1000.0)}"
        snapshot.source = snap_source
        snapshot.asset_scope = scope_label
        snapshot.ts = snap_ts
        snapshot.evaluated_at_ts = now_ts
        snapshot.age_sec = snap_age_sec
        snapshot.source_timestamps = domain_ts
        snapshot.source_ages_sec = {k: round(max(0.0, now_ts - v), 1) for k, v in domain_ts.items()}

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        snapshot: Any = None,
        workspace_dir: str = "",
        logs_dirs: Optional[List[str]] = None,
        intent_gateway: Any = None,
        scope: str = "crypto",
        now_ts: float = 0.0,
        read_json_fn: Any = None,
        log: Optional[Any] = None,
        allow_network: bool = True,
        persist: bool = True,
        **kwargs: Any,
    ) -> Any:
        cache_dir = cachedb_dir
        logs_dirs = list(logs_dirs or ())
        if snapshot is None:
            snapshot = TelemetrySensorSnapshot()
        if not workspace_dir and cache_dir:
            workspace_dir = os.path.dirname(os.path.abspath(str(cache_dir).rstrip("/\\")))
        is_live_wallclock = (safe_pos_float(now_ts, default=0.0) or 0.0) <= 0.0
        now_ts, _rj, _rj_any = _resolve_macro_collect_inputs(now_ts, read_json_fn)
        start_now_ts = now_ts
        macro_sensors, snapshot.breaking_news = harvest_macro_and_news_sensors(
            cache_dir, now_ts, _rj, scope=scope, allow_network=allow_network, persist=persist
        )
        for attr, val in macro_sensors.items():
            setattr(snapshot, attr, val)
        if lh_snap := _mss.load_cached_long_horizon_snapshot(cache_dir):
            snapshot.long_horizon = lh_snap
        if op_note := load_active_operator_regime_note(workspace_dir, now_ts=now_ts):
            snapshot.operator_regime_note = op_note
        if str(scope or "").strip().lower() in ("equities", "equity"):
            for _crypto_attr in CRYPTO_ONLY_MACRO_SENSOR_KEYS:
                setattr(snapshot, _crypto_attr, None)

        _mc.ensure_cacheprice_classes_bound()
        prices_data = _rj(os.path.join(cache_dir, _mc.CacheCurrentPriceManager.DEFAULT_FILENAME))
        trends_data = _rj(os.path.join(cache_dir, _mc.CacheShortTrendManager.DEFAULT_FILENAME))
        long_trends_data = _rj(os.path.join(cache_dir, _mc.CacheLongTrendManager.DEFAULT_FILENAME))
        priceanalysis_data = _rj(os.path.join(cache_dir, _mc.CacheLongTrendManager.SOURCE_ANALYSIS_FILENAME))
        t_trend_data = CacheTTrendStore.read_from_cachedb(cache_dir, read_json_fn=_rj)
        kalman_data = _rj(os.path.join(cache_dir, _mc.CacheShortTrendManager.KALMAN_TREND_FILENAME))
        watcher_all_price_data = _rj(os.path.join(cache_dir, CACHE_WATCHER_ALL_PRICE_FILENAME))
        trailing_map, trailing_by_venue = VenueTrailingMapsCollector().collect(
            cache_dir, "*", workspace_dir=workspace_dir, read_json_fn=_rj
        )
        long_items = long_trends_data if isinstance(long_trends_data, dict) else {}
        t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}
        snapshot.prices_and_trends = _harvest_initial_prices_and_trends_impl(
            cache_dir, prices_data, trends_data, _rj, workspace_dir=workspace_dir
        )

        intents_data = None
        if callable(load_intents_fn := getattr(intent_gateway, "load_all_intents", None)):
            try:
                loaded = load_intents_fn()
                if isinstance(loaded, dict):
                    loaded = loaded.get("intents")
                intents_data = loaded if isinstance(loaded, list) else None
            except Exception:
                intents_data = None
        intents_data, raw_known_oids = _resolve_ai_intents_and_known_order_ids(
            cache_dir, ai_intents=intents_data, read_json_fn=_rj_any
        )
        known_ai_oids = raw_known_oids if raw_known_oids is not None else collect_known_ai_order_ids(intents_data)
        gw_max_usd = safe_pos_float(getattr(intent_gateway, "max_notional_usd", None), default=None)
        gw_min_usd = safe_nonneg_float(getattr(intent_gateway, "min_notional_usd", None), default=None)
        if (gw_max_usd is None or gw_min_usd is None) and workspace_dir:
            og_conf_path = os.path.join(workspace_dir, "order_guard.conf")
            if os.path.isfile(og_conf_path):
                try:
                    _, cfg_min_usd, cfg_max_usd, _, _ = parse_intent_gateway_config(
                        load_key_value_conf(workspace_dir, workspace_dir)
                    )
                    if gw_min_usd is None:
                        gw_min_usd = cfg_min_usd
                    if gw_max_usd is None:
                        gw_max_usd = cfg_max_usd
                except Exception:
                    pass
        gw_max_usd = gw_max_usd or 2000.0
        gw_min_usd = gw_min_usd or 0.0

        tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = [], [], {}, {}
        verified_open_venues: set[str] = set()
        verified_open_symbols: set[tuple[str, str]] = set()
        load_tradable_fn = getattr(intent_gateway, "load_tradable_instruments", None)
        has_inst_source = has_workspace_instrument_sources(workspace_dir, asset_scope=scope)
        should_harvest = False
        inst_map: Dict[str, Any] = {}
        allow_empty_recovery = False
        harvest_persist = persist
        if callable(load_tradable_fn) and has_inst_source:
            should_harvest = True
            try:
                inst_map = load_tradable_fn(
                    only_workspace_conf=(scope != "equities"),
                    include_closed_rth=(scope == "equities"),
                    asset_scope=scope,
                )
            except Exception as exc:
                if log:
                    log.debug("Could not load tradable instruments for telemetry: %s", exc)
                inst_map = {}
                allow_empty_recovery = True
        elif not allow_network and cache_dir:
            should_harvest = True
            allow_empty_recovery = True
            harvest_persist = False
        if should_harvest:
            tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = harvest_tradable_positions_and_orders(
                inst_map, snapshot.prices_and_trends, trailing_map, trailing_by_venue,
                known_ai_order_ids=known_ai_oids, gateway_max_usd=gw_max_usd, gateway_min_usd=gw_min_usd, cache_dir=cache_dir,
                verified_open_venues=verified_open_venues,
                verified_open_symbols=verified_open_symbols,
                allow_empty_inst_recovery=allow_empty_recovery,
                allow_network=allow_network,
                persist=harvest_persist,
                now_ts=now_ts,
            )
            if open_orders_list and isinstance(intents_data, list):
                open_orders_list = prune_reconciled_terminal_open_orders(open_orders_list, intents_data)

        if tradable_positions:
            snapshot.portfolio_summary["tradable_positions"] = tradable_positions
        if open_orders_list:
            snapshot.portfolio_summary["open_orders"] = open_orders_list
            snapshot.portfolio_summary["open_limit_orders"] = list(open_orders_list)
        if venue_quote_cash:
            snapshot.portfolio_summary["venue_quote_cash"] = venue_quote_cash
        if trailing_map:
            snapshot.portfolio_summary["trailing_peaks"] = trailing_map

        watcher_price_items = watcher_all_price_data.get("items", {}) if isinstance(watcher_all_price_data, dict) else {}

        def _run_enrich_cross_venue_trends() -> None:
            est_t_fn = (
                _surv_mod.estimate_T
                if (
                    allow_network
                    and persist
                    and (
                        is_live_network_backfill_enabled()
                        or getattr(_surv_mod, "estimate_T", None) is not getattr(_surv_mod, "_ORIG_ESTIMATE_T", None)
                        or getattr(_surv_mod, "fetch_klines", None) is not getattr(_surv_mod, "_ORIG_FETCH_KLINES", None)
                    )
                )
                else None
            )
            enrich_cross_venue_trends(
                snapshot.prices_and_trends,
                trends_data,
                kalman_data,
                long_items,
                priceanalysis_data,
                t_trend_map,
                watcher_price_items,
                allow_network=False if not allow_network else None,
                estimate_t_fn=est_t_fn,
            )

        _run_enrich_cross_venue_trends()
        cur_p_items = prices_data.get("items", {}) if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict) else {}
        price_path_bars: Dict[str, List[List[Any]]] = {}
        enrich_24h_and_microstructure(
            cache_dir,
            workspace_dir,
            snapshot.prices_and_trends,
            snapshot.microstructure,
            cur_p_items,
            inst_by_sym,
            now_ts,
            _rj,
            allow_network=allow_network,
            persist=persist,
            price_path_bars=price_path_bars,
        )
        _run_enrich_cross_venue_trends()
        cycle_history = load_microstructure_cycle_history(cache_dir)
        orig_cycle_history = dict(cycle_history)
        sync_sibling_symbol_telemetry(
            snapshot.prices_and_trends,
            snapshot.microstructure,
            fee_spec_resolver=build_persisted_venue_fee_resolver(cache_dir, now_ts=now_ts),
            now_ts=now_ts,
            persist_cycle_deltas=False,
            cycle_history=cycle_history,
        )
        if persist and allow_network and cycle_history != orig_cycle_history:
            save_microstructure_cycle_history(cache_dir, cycle_history)

        kraken_cycles, hyperliquid_cycles, t212_cycles = PortfolioAndVenueCyclesCollector(start_sync=False).collect(
            cache_dir,
            "*",
            workspace_dir=workspace_dir,
            logs_dirs=logs_dirs,
            portfolio_summary=snapshot.portfolio_summary,
            prices_and_trends=snapshot.prices_and_trends,
            microstructure=snapshot.microstructure,
            watcher_price_items=watcher_price_items,
            inst_by_sym=inst_by_sym,
            read_json_fn=_rj,
            now_ts=now_ts,
        )

        recent_orders = harvest_recent_orders_and_enrich_positions(
            cache_dir, inst_by_sym, snapshot.portfolio_summary.get("tradable_positions", []), snapshot.prices_and_trends,
            kraken_cycles, hyperliquid_cycles, t212_cycles, _rj,
            known_ai_order_ids=known_ai_oids, gateway_max_usd=gw_max_usd, gateway_min_usd=gw_min_usd,
            ai_intents=intents_data if isinstance(intents_data, list) else None,
            allow_network=allow_network,
            persist=bool(persist and allow_network),
            now_ts=now_ts,
        )
        if t212_cycles:
            if xst_summary := harvest_kraken_xstocks(
                os.path.join(workspace_dir, "kraken"),
                snapshot.prices_and_trends,
                t212_cycles,
                _rj,
                kraken_cycles=kraken_cycles,
                cachedb_dir=cache_dir,
            ):
                snapshot.portfolio_summary["kraken_xstocks"] = xst_summary
            if cross_market_ctx := build_cross_market_context(
                snapshot.prices_and_trends,
                snapshot.microstructure,
                t212_cycles,
                macro_benchmarks=getattr(snapshot, "macro_benchmarks", None),
            ):
                snapshot.portfolio_summary["cross_market_context"] = cross_market_ctx
        if recent_orders:
            snapshot.portfolio_summary["recent_orders"] = recent_orders

        if is_live_wallclock:
            now_ts = max(now_ts, _time.time())
            elapsed_wall_s = max(0.0, now_ts - start_now_ts)
            if elapsed_wall_s >= 1.0:
                for m_val in macro_sensors.values():
                    if isinstance(m_val, dict):
                        m_ts = normalize_epoch_sec(m_val.get("ts"), default=None) or extract_timestamp_s(m_val)
                        if m_ts is not None and m_ts > 0.0:
                            m_val["age_min"] = round(max(0.0, now_ts - m_ts) / 60.0, 1)
                if isinstance(snapshot.breaking_news, list):
                    for hn in snapshot.breaking_news:
                        if isinstance(hn, dict) and (old_age := safe_float(hn.get("age_min"), default=None)) is not None:
                            hn["age_min"] = round(old_age + elapsed_wall_s / 60.0, 1)
        self._enrich_ai_intent_and_guard_telemetry(
            snapshot,
            intents_data,
            scope=scope,
            now_ts=now_ts,
            workspace_dir=workspace_dir,
            recent_orders=recent_orders,
            open_orders_list=open_orders_list,
            eff_verified_venues=verified_open_venues if (has_inst_source or should_harvest) else None,
            eff_verified_symbols=verified_open_symbols if (has_inst_source or should_harvest) else None,
            price_bars_by_symbol=price_path_bars,
            cache_dir=cache_dir,
            intent_gateway=intent_gateway,
            persist=persist,
        )
        snapshot.portfolio_summary.update(
            harvest_bot_runtime_states(cache_dir, now_ts, _rj, tradable_positions=snapshot.portfolio_summary.get("tradable_positions"))
        )
        self._apply_macro_scope_and_risk_overlays(
            snapshot,
            scope=scope,
            now_ts=now_ts,
            inst_by_sym=inst_by_sym,
            t212_cycles=t212_cycles,
            workspace_dir=workspace_dir,
        )
        self._stamp_snapshot_domain_timestamps(
            snapshot,
            scope=scope,
            allow_network=allow_network,
            persist=persist,
            now_ts=now_ts,
            cache_dir=cache_dir,
            read_json_fn=_rj,
            macro_sensors=macro_sensors,
            prices_data=prices_data,
            cur_p_items=cur_p_items,
            trends_data=trends_data,
            inst_by_sym=inst_by_sym,
            t212_cycles=t212_cycles,
        )
        return snapshot

    def read_snapshot(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        snapshot: Any = None,
        workspace_dir: str = "",
        logs_dirs: Optional[List[str]] = None,
        intent_gateway: Any = None,
        scope: str = "crypto",
        now_ts: float = 0.0,
        read_json_fn: Any = None,
        log: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Read-only consumer entrypoint: populate `TelemetrySensorSnapshot` from `cachedb/` with 0 network I/O and 0 disk writes."""
        return self.collect(
            cachedb_dir,
            symbol,
            snapshot=snapshot,
            workspace_dir=workspace_dir,
            logs_dirs=logs_dirs,
            intent_gateway=intent_gateway,
            scope=scope,
            now_ts=now_ts,
            read_json_fn=read_json_fn,
            log=log,
            allow_network=False,
            persist=False,
            **kwargs,
        )


for _macro_collector_cls in (
    MacroBenchmarksSnapshotCollector,
    MacroEconomicCalendarCollector,
    SpotEtfFlowsCollector,
    MacroAndNewsSensorsCollector,
    InitialPricesAndTrendsCollector,
    PortfolioAndVenueCyclesCollector,
    MarketAndPortfolioSnapshotOrchestrator,
):
    TelemetryCollectorFactory.register(_macro_collector_cls.COLLECTOR_ID, _macro_collector_cls)


def collect_spot_etf_flows_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 4.0,
    persist: bool = True,
    fetch_text_fn: Optional[Callable[..., Any]] = None,
    skip_if_fresh_sec: float = 0.0,
    max_days: int = 10,
) -> Optional[Dict[str, Any]]:
    """Fetch daily US Spot BTC/ETH ETF net flows (Farside Investors) and persist to cachedb."""
    return TelemetryCollectorFactory.create("spot_etf_flows", start_sync=True).collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        timeout_sec=timeout_sec,
        persist=persist,
        fetch_text_fn=fetch_text_fn,
        skip_if_fresh_sec=skip_if_fresh_sec,
        max_days=max_days,
    )


def collect_macro_calendar_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.0,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
    tracked_equity_tickers: Optional[List[str]] = None,
    skip_if_fresh_sec: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Fetch weekly USD macro releases (ForexFactory) and tracked/bellwether corporate earnings (Nasdaq) and persist to cachedb."""
    return TelemetryCollectorFactory.create("macro_economic_calendar", start_sync=True).collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        timeout_sec=timeout_sec,
        persist=persist,
        fetch_fn=fetch_fn,
        tracked_equity_tickers=tracked_equity_tickers,
        skip_if_fresh_sec=skip_if_fresh_sec,
    )


def collect_macro_benchmarks_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.5,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch SPY, QQQ, ES=F, NQ=F, DXY, US10Y, IBIT, Coinbase Premium Index, and Deribit DVOL and persist to cachedb."""
    return TelemetryCollectorFactory.create("macro_benchmarks_snapshot", start_sync=True).collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        timeout_sec=timeout_sec,
        persist=persist,
        fetch_fn=fetch_fn,
    )


def load_cached_macro_benchmarks(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached macro benchmarks snapshot from ``cache_dir`` without network I/O, enriched with live futures lead and BTC 24h delta."""
    data, _ = _mss.read_first_cached_dict(cache_dir, "macro_benchmarks_files", CACHE_MACRO_BENCHMARKS_FILENAME)
    return enrich_macro_benchmarks_live_state(
        data,
        btc_chg_24h_pct=_mss.load_cached_btc_24h_change_pct(cache_dir),
    )


def harvest_macro_and_news_sensors(
    cache_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
    *,
    scope: str = "",
    allow_network: bool = True,
    persist: bool = True,
) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Harvest Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, Macro Benchmarks, and Breaking News."""
    return TelemetryCollectorFactory.create("macro_and_news_sensors", start_sync=allow_network).collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        read_json_fn=read_json_fn,
        scope=scope,
        allow_network=allow_network,
        persist=persist,
    )


__all__ = [
    "InitialPricesAndTrendsCollector",
    "MACRO_BENCHMARK_SPECS",
    "MacroAndNewsSensorsCollector",
    "MacroBenchmarksSnapshotCollector",
    "MacroEconomicCalendarCollector",
    "MacroTelemetryCollectorBase",
    "MarketAndPortfolioSnapshotOrchestrator",
    "PortfolioAndVenueCyclesCollector",
    "SpotEtfFlowsCollector",
    "apply_asset_scope_filter",
    "build_cycle_fallback_prices",
    "collect_macro_benchmarks_snapshot",
    "collect_macro_calendar_snapshot",
    "collect_spot_etf_flows_snapshot",
    "compute_cross_asset_regime",
    "compute_deribit_options_surface",
    "compute_equity_macro_regime",
    "compute_futures_premarket_lead",
    "enrich_sentiment_advisor_live_flow_divergence",
    "format_etf_flows_summary_line",
    "format_macro_benchmarks_summary_line",
    "harvest_macro_and_news_sensors",
    "load_cached_macro_benchmarks",
    "reconcile_live_venue_equity_summary",
]
