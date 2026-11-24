"""Pre-trade telemetry collector for HighStakeGuard and PostStopReentryGuard (0 ms local cache)."""
from __future__ import annotations

import math
import os
import time as _time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from botcore import (
    SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
    clean_scale,
    extract_current_price_item,
    extract_timestamp_s,
    is_live_network_backfill_enabled,
    normalize_epoch_sec,
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_pos_float,
    _prices_diverged,
)
from cachepaths import (
    CACHE_DERIVATIVES_TELEMETRY_PREFIX,
    CACHE_ETF_FLOWS_FILENAME,
    CACHE_FEAR_GREED_FILENAME,
    CACHE_GEOPOLITICAL_EVAL_FILENAME,
    CACHE_KALMAN_TREND_FILENAME,
    CACHE_MACRO_BENCHMARKS_FILENAME,
    CACHE_MACRO_CALENDAR_FILENAME,
    CACHE_ORDERBOOK_DEPTH_PREFIX,
    CACHE_SENTIMENT_EVAL_FILENAME,
    CACHE_WATCHER_ALL_PRICE_FILENAME,
    CACHE_WHALE_SNAPSHOT_PREFIX,
    cache_ohlc_24h_filename,
    canonical_base_asset,
    clean_symbol_key,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    get_cachedb_dir,
    resolve_cachedb_filepath,
)
from intelligence.internal.guards.guard_decision import fetch_guard_snapshot
from intelligence.macro.calendar_ssot import (
    format_pretrade_macro_catalyst_line,
    resolve_macro_economic_calendar,
)
from intelligence.macro_sentiment_ssot import (
    ALL_KINEMATIC_METRIC_KEYS,
    TELEMETRY_SOURCE_STALE_INTERVAL_MULT,
    format_interval_sec_label,
    load_microstructure_cycle_history,
    load_venue_balances_state,
    read_first_cached_dict,
    resolve_derived_telemetry_intervals,
    resolve_kalman_vel_z,
    resolve_read_json_cached,
    SHORT_KINEMATIC_METRIC_KEYS,
    parse_ts_px_row,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    enrich_macro_benchmarks_live_state,
    format_bottom_structure_6h_parts,
    format_equity_macro_regime_line,
    format_pretrade_etf_flows_line,
)
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
    get_derivatives_collector,
    get_orderbook_collector,
    get_whale_collector,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    ensure_fresh_symbol_microstructure,
    load_symbol_microstructure,
    load_symbol_price_series,
    resolve_cached_btc_24h_change_pct,
    upgrade_tpo_vp_from_ohlc,
    ensure_cacheprice_classes_bound,
    EQ_YAHOO_CHART_MEMO,
    load_stitched_7d_ohlc_closes,
    refresh_stale_t212_cached_positions,
    resolve_fallback_ohlc_fn,
    resolve_live_24trend_current_price,
)
import intelligence.sentiment.collectors.microstructure_collector as _mc_mod
from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    MicrostructureCycleDeltaEvaluator,
    compute_cross_venue_arb_metrics,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    format_agg_spot_cvd_part,
    format_book_window_cap_tag,
    format_flow_horizon_parts,
    format_realized_liq_sources_tag,
    format_spot_cvd_source_tag,
    format_spot_volume_rvol_parts,
    is_ob_live_and_newer,
    purge_liquidation_cluster_keys,
    reconcile_sibling_flow_and_liq_regimes,
    reconcile_spot_and_orderbook_microstructure,
    rescale_orderbook_walls_to_price,
    should_rescale_usd_sibling_walls,
    first_variant_dict,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    format_recent_fill_subcycle_parts,
    is_last_buy_lot_harvested,
)
from intelligence.sentiment.evaluators.asset_session_policy import (
    AssetSessionPolicy,
    resolve_asset_session_policy,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_24h_price_metrics,
    compute_7d_range_metrics,
    compute_cross_btc_metrics,
    compute_series_kinematics,
    format_atr_parts,
    format_kinematic_parts,
    format_multi_window_slopes_core,
    format_session_extremes_parts,
    format_session_vwap_summary,
    format_va_hold_suffix,
    is_unseeded_kalman_state,
    recalculate_live_price_geometry,
    apply_24h_series_enrichment,
    compute_kalman_from_series,
    compute_short_slopes_from_series,
    LAST_7D_4H_BARS_BY_SYM,
    LAST_7D_SPAN_MS_BY_SYM,
    load_stitched_24h_ohlc_points,
    STALE_SHORT_24H_KEYS,
)
from instrument_registry import casefold_get, disabled_registry_symbols, is_equity_symbol, is_symbol_market_open, resolve_symbol_identity, venue_for_symbol
from providers.base import MarketDataProvider
from state_io import (
    read_json_dict_recovering as _default_read_json_cached,
)

read_json_cached = _default_read_json_cached
_ORIG_COMPUTE_24H_METRICS = compute_24h_price_metrics


def _call_compute_24h(
    series: list,
    *,
    is_equity: bool = False,
    series_7d_ts_px: Optional[list] = None,
) -> Dict[str, Any]:
    if compute_24h_price_metrics is _ORIG_COMPUTE_24H_METRICS:
        return compute_24h_price_metrics(series, is_equity=is_equity, series_7d_ts_px=series_7d_ts_px) or {}
    return compute_24h_price_metrics(series) or {}

DEFAULT_MAX_PRETRADE_TELEMETRY_AGE_SEC: float = 1800.0
PRETRADE_SOURCE_STALE_INTERVAL_MULT: float = TELEMETRY_SOURCE_STALE_INTERVAL_MULT


def resolve_pretrade_source_intervals_sec(asset_scope: str = "CRYPTO") -> Dict[str, float]:
    """Return SSOT expected update intervals (seconds) per pre-trade telemetry source."""
    derived = resolve_derived_telemetry_intervals()
    ob_itv = derived["orderbook_interval_sec"]
    deriv_itv = derived["derivatives_interval_sec"]
    whale_itv = derived["whale_interval_sec"]
    is_eq = str(asset_scope or "").strip().upper() in ("EQUITY", "EQUITIES")
    return {
        "price_series_24h": max(whale_itv, ob_itv * 3.0) if is_eq else deriv_itv,
        "orderbook": ob_itv,
        "microstructure": ob_itv,
        "derivatives": deriv_itv,
        "whale": whale_itv,
        "short_slopes": SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
        "macro_benchmarks": derived["macro_benchmarks_interval_sec"],
        "macro_calendar": derived["macro_calendar_interval_sec"],
        "etf_flows": derived["etf_flows_interval_sec"],
        "fear_greed": derived["fear_greed_interval_sec"],
        "sentiment_eval": derived["default_sentiment_interval_sec"],
        "geopolitical_eval": derived["default_geopolitical_interval_sec"],
    }


def _clean_pillar_timestamps(raw_map: Any) -> Dict[str, float]:
    if not isinstance(raw_map, Mapping):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw_map.items():
        key_s = str(k or "").strip()
        ts_v = normalize_epoch_sec(v, default=0.0) or 0.0
        if key_s and ts_v > 0.0:
            out[key_s] = ts_v
    return out


@dataclass(frozen=True)
class PretradeTelemetrySnapshot(Mapping[str, Any]):
    """Immutable read-only telemetry snapshot carrying explicit identity, source, and age."""

    symbol: str
    canonical_symbol: str
    asset_scope: str
    source: str
    pillar_sources: Dict[str, str] = field(default_factory=dict)
    ts: float = 0.0
    evaluated_at_ts: float = 0.0
    age_sec: Optional[float] = None
    max_age_sec: float = DEFAULT_MAX_PRETRADE_TELEMETRY_AGE_SEC
    data: Dict[str, Any] = field(default_factory=dict)
    unavailable_reason: Optional[str] = None
    enforce_freshness: bool = False
    pillar_timestamps: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return bool(self.data)

    @property
    def is_fresh(self) -> bool:
        """True when snapshot is not marked unavailable and age is within ``max_age_sec``."""
        if self.unavailable_reason:
            return False
        if self.enforce_freshness and not self.data:
            return False
        if self.age_sec is None:
            return not self.enforce_freshness
        return 0.0 <= self.age_sec <= self.max_age_sec

    @property
    def is_stale(self) -> bool:
        """True when snapshot age exceeds ``max_age_sec`` or is explicitly marked unavailable."""
        return not self.is_fresh

    @property
    def is_available(self) -> bool:
        """True when snapshot is usable by pre-trade guards under its configured freshness policy."""
        if self.unavailable_reason:
            return False
        if self.enforce_freshness:
            return bool(self.data) and self.is_fresh
        return True

    @property
    def source_freshness(self) -> Dict[str, Dict[str, Any]]:
        """Per-source age, expected update interval, age/interval ratio, and freshness flag."""
        if not self.pillar_timestamps or self.evaluated_at_ts <= 0.0:
            return {}
        intervals = resolve_pretrade_source_intervals_sec(self.asset_scope)
        is_eq_scope = str(self.asset_scope or "").strip().upper() in ("EQUITY", "EQUITIES")
        ob_ts = normalize_epoch_sec(self.pillar_timestamps.get("orderbook"), default=0.0) or 0.0
        out: Dict[str, Dict[str, Any]] = {}
        for name, raw_ts in self.pillar_timestamps.items():
            ts_val = normalize_epoch_sec(raw_ts, default=0.0) or 0.0
            if ts_val <= 0.0:
                continue
            if name == "microstructure" and ob_ts > 0.0 and abs(ts_val - ob_ts) < 1e-3:
                continue
            if (
                name == "derivatives"
                and is_eq_scope
                and self.source != "explicit"
                and "Derivatives Open Interest" not in self.data
                and "Perpetual Funding Rate" not in self.data
            ):
                continue
            age_s = max(0.0, self.evaluated_at_ts - ts_val)
            itv_s = safe_pos_float(intervals.get(name), self.max_age_sec) or self.max_age_sec
            out[name] = {
                "ts": ts_val,
                "age_sec": round(age_s, 1),
                "interval_sec": round(itv_s, 1),
                "age_ratio": round(age_s / itv_s, 2),
                "fresh": age_s <= itv_s * PRETRADE_SOURCE_STALE_INTERVAL_MULT,
            }
        return out

    def format_source_ages(self) -> str:
        """Render compact per-source age vs update interval summary for LLM guard prompts."""
        parts = []
        for name, info in self.source_freshness.items():
            itv_s = format_interval_sec_label(info["interval_sec"])
            state = "fresh" if info["fresh"] else "stale"
            parts.append(f"{name}={info['age_sec']:.1f}s/{itv_s} ({info['age_ratio']:.1f}x, {state})")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow dict copy of the formatted telemetry lines."""
        return dict(self.data)

    @staticmethod
    def _resolve_now_and_max_age(
        now_ts: Optional[float], max_age_sec: Optional[float]
    ) -> Tuple[float, float]:
        return (
            resolve_now_ts(now_ts),
            safe_pos_float(max_age_sec, DEFAULT_MAX_PRETRADE_TELEMETRY_AGE_SEC)
            or DEFAULT_MAX_PRETRADE_TELEMETRY_AGE_SEC,
        )

    @classmethod
    def unavailable(
        cls,
        symbol: str,
        reason: str = "telemetry_unavailable",
        *,
        now_ts: Optional[float] = None,
        source: str = "unavailable",
        max_age_sec: Optional[float] = None,
    ) -> "PretradeTelemetrySnapshot":
        eff_now, eff_max_age = cls._resolve_now_and_max_age(now_ts, max_age_sec)
        sym_raw, canon_sym, scope = (
            (s := str(symbol or "").strip()),
            clean_symbol_key(s) or s.upper(),
            "EQUITY" if is_equity_symbol(s) else "CRYPTO",
        )
        return cls(
            symbol=sym_raw,
            canonical_symbol=canon_sym,
            asset_scope=scope,
            source=source,
            pillar_sources={},
            ts=0.0,
            evaluated_at_ts=eff_now,
            age_sec=None,
            max_age_sec=eff_max_age,
            data={},
            unavailable_reason=str(reason or "telemetry_unavailable"),
            enforce_freshness=True,
        )

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        symbol: str,
        now_ts: Optional[float] = None,
        source: str = "explicit",
        ts: Optional[float] = None,
        pillar_sources: Optional[Dict[str, str]] = None,
        pillar_timestamps: Optional[Dict[str, float]] = None,
        max_age_sec: Optional[float] = None,
        unavailable_reason: Optional[str] = None,
        enforce_freshness: bool = False,
    ) -> "PretradeTelemetrySnapshot":
        eff_now, eff_max_age = cls._resolve_now_and_max_age(now_ts, max_age_sec)
        if isinstance(raw, cls):
            if (
                max_age_sec is None
                and not enforce_freshness
                and unavailable_reason is None
                and pillar_timestamps is None
                and (now_ts is None or abs(eff_now - raw.evaluated_at_ts) < 1e-6)
            ):
                return raw
            snap_ts = raw.ts
            snap_age = max(0.0, eff_now - snap_ts) if snap_ts > 0.0 else raw.age_sec
            eff_unavail = unavailable_reason or raw.unavailable_reason
            eff_enforce = bool(enforce_freshness or raw.enforce_freshness or max_age_sec is not None)
            if eff_enforce and eff_unavail is None:
                if not raw.data:
                    eff_unavail = "telemetry_empty"
                elif snap_age is None or snap_age > eff_max_age:
                    eff_unavail = "telemetry_stale"
            return cls(
                symbol=raw.symbol or str(symbol or "").strip(),
                canonical_symbol=raw.canonical_symbol or clean_symbol_key(symbol) or str(symbol or "").strip().upper(),
                asset_scope=raw.asset_scope,
                source=raw.source or source,
                pillar_sources=dict(raw.pillar_sources),
                ts=snap_ts,
                evaluated_at_ts=eff_now,
                age_sec=snap_age,
                max_age_sec=eff_max_age,
                data=dict(raw.data),
                unavailable_reason=eff_unavail,
                enforce_freshness=eff_enforce,
                pillar_timestamps=_clean_pillar_timestamps(
                    pillar_timestamps if pillar_timestamps is not None else raw.pillar_timestamps
                ),
            )
        sym_raw = str(symbol or "").strip()
        canon_sym = clean_symbol_key(sym_raw) or sym_raw.upper()
        scope = "EQUITY" if is_equity_symbol(sym_raw) else "CRYPTO"
        if not isinstance(raw, Mapping):
            return cls.unavailable(
                sym_raw,
                unavailable_reason or "telemetry_unavailable",
                now_ts=eff_now,
                source=source,
                max_age_sec=eff_max_age,
            )
        raw_dict = dict(raw)
        meta_ts = normalize_epoch_sec(raw_dict.pop("_telemetry_ts", None), default=None)
        meta_src = raw_dict.pop("_telemetry_source", None)
        meta_pillars = raw_dict.pop("_telemetry_pillar_sources", None)
        meta_pillar_ts = raw_dict.pop("_telemetry_pillar_timestamps", None)
        meta_unavail = raw_dict.pop("_telemetry_unavailable_reason", None)
        eff_ts = normalize_epoch_sec(ts if ts is not None else meta_ts, default=0.0) or 0.0
        if eff_ts <= 0.0 and source == "explicit" and raw_dict:
            eff_ts = eff_now
        age_val = max(0.0, eff_now - eff_ts) if eff_ts > 0.0 else None
        eff_enforce = bool(enforce_freshness or max_age_sec is not None)
        eff_unavail = unavailable_reason or (str(meta_unavail) if meta_unavail else None)
        if eff_enforce and eff_unavail is None:
            if not raw_dict:
                eff_unavail = "telemetry_empty"
            elif age_val is not None and age_val > eff_max_age:
                eff_unavail = "telemetry_stale"
            elif age_val is None and source != "explicit":
                eff_unavail = "telemetry_timestamp_missing"
        return cls(
            symbol=sym_raw,
            canonical_symbol=canon_sym,
            asset_scope=scope,
            source=str(meta_src or source),
            pillar_sources=dict(pillar_sources if isinstance(pillar_sources, dict) else (meta_pillars if isinstance(meta_pillars, dict) else {})),
            ts=eff_ts,
            evaluated_at_ts=eff_now,
            age_sec=age_val,
            max_age_sec=eff_max_age,
            data=raw_dict,
            unavailable_reason=eff_unavail,
            enforce_freshness=eff_enforce,
            pillar_timestamps=_clean_pillar_timestamps(
                pillar_timestamps if pillar_timestamps is not None else meta_pillar_ts
            ),
        )


def _resolve_read_json_cached() -> Callable[[str], Any]:
    return resolve_read_json_cached(local_fn=read_json_cached)


def _read_first_cached_dict(cachedb_dir: str, group_key: str, fallback_fname: str) -> tuple[Optional[Dict[str, Any]], str]:
    return read_first_cached_dict(cachedb_dir, group_key, fallback_fname, read_json_fn=_resolve_read_json_cached())


def _scope_symbol_variants(
    symbol: str,
    is_eq_sym: bool,
    is_usd_sym: bool,
    sym_quotes: tuple[str, ...],
) -> Tuple[List[str], List[str]]:
    eff_sym = ensure_equity_suffix(symbol) if is_eq_sym else symbol
    ident = resolve_symbol_identity(eff_sym)
    all_quotes = ("USDC", "USDT", "USD", "FDUSD", "BUSD", "TUSD", "EUR", "BTC", "ETH")
    sym_variants = list(ident.candidate_cache_symbols(include_base=True, quotes=all_quotes))
    quote_variants = list(ident.candidate_cache_symbols(include_base=is_usd_sym, quotes=sym_quotes))
    return sym_variants, quote_variants


_ORIG_ENSURE_FRESH_SYMBOL_MICROSTRUCTURE = ensure_fresh_symbol_microstructure


def _ensure_fresh_pretrade_microstructure(
    cachedb_dir: str,
    symbol: str,
    quote_variants: List[str],
    is_usd_sym: bool,
    is_eq_sym: bool,
    _rj: Callable[[str], Any],
    *,
    allow_network: Optional[bool] = False,
) -> bool:
    is_monkeypatched = ensure_fresh_symbol_microstructure is not _ORIG_ENSURE_FRESH_SYMBOL_MICROSTRUCTURE
    can_refresh = (allow_network is True and is_live_network_backfill_enabled()) or (
        allow_network is None and is_monkeypatched
    )
    if not ((is_usd_sym or is_eq_sym) and can_refresh and _rj is _default_read_json_cached):
        return False
    refresh_cands = list(quote_variants)
    if not is_eq_sym and canonical_base_asset(symbol) != "BTC" and "BTCUSDC" not in refresh_cands:
        refresh_cands.append("BTCUSDC")
    disabled_syms = disabled_registry_symbols()
    refreshed_any = False
    for cand_m in refresh_cands:
        cand_mu = str(cand_m or "").strip().upper()
        if not cand_mu or cand_mu in disabled_syms:
            continue
        has_existing_m = any(
            os.path.exists(str(resolve_cachedb_filepath(os.path.join(cachedb_dir, f"{pfx_m}{cand_mu}.json"))))
            for pfx_m in (CACHE_ORDERBOOK_DEPTH_PREFIX, CACHE_DERIVATIVES_TELEMETRY_PREFIX, CACHE_WHALE_SNAPSHOT_PREFIX)
        )
        if has_existing_m:
            res = ensure_fresh_symbol_microstructure(
                cachedb_dir,
                cand_mu,
                max_age_ob_sec=180.0,
                max_age_whale_sec=360.0,
                include_derivatives=True,
            )
            if res is True or (is_monkeypatched and res is None):
                refreshed_any = True
    return refreshed_any


def _funding_period_label(
    fr: float,
    ann_pct: Optional[float] = None,
    *,
    symbol: str = "",
) -> str:
    """Resolve '1h' (native L1 hourly funding, e.g. Hyperliquid) vs '8h' (standard CEX perp funding)."""
    if ann_pct is not None and abs(fr) > 1e-8:
        return "1h" if abs(ann_pct / (fr * 100.0)) > 4000.0 else "8h"
    if symbol and MarketDataProvider.has_native_l1_derivatives(venue_for_symbol(symbol)):
        return "1h"
    return "8h"


def _snap_ts(obj: Any) -> float:
    raw_t = obj_attr(obj, "ts", obj_attr(obj, "timestamp", None))
    return normalize_epoch_sec(raw_t, default=0.0) or 0.0


def _fetch_pillar2_snapshot(
    cachedb_dir: str,
    symbol: str,
    getter_fn: Callable[[], Any],
    prefix: str,
    sym_variants: List[str],
    quote_variants: List[str],
    _rj: Callable[[str], Any],
    *,
    now_ts: Optional[float] = None,
) -> Any:
    if prefix in (CACHE_WHALE_SNAPSHOT_PREFIX, CACHE_DERIVATIVES_TELEMETRY_PREFIX) and resolve_asset_session_policy(symbol).is_tokenized_equity:
        return None
    snap = None
    try:
        col = getter_fn()
        col_cdir = getattr(col, "_cache_dir", None)
        if not isinstance(col_cdir, str) and hasattr(type(col), "_effective_cache_dir"):
            col_cdir = col._effective_cache_dir()
        if not isinstance(col_cdir, str):
            raw_cdir = getattr(col, "cache_dir", None)
            if isinstance(raw_cdir, str):
                col_cdir = raw_cdir
        if not isinstance(col_cdir, str) or os.path.abspath(col_cdir) == os.path.abspath(cachedb_dir):
            snap = fetch_guard_snapshot(
                col,
                symbol,
                now=now_ts if hasattr(col, "_sync_cache_dir_scope") else None,
                allow_network=False,
            )
    except Exception:
        snap = None

    variants = quote_variants if prefix == CACHE_ORDERBOOK_DEPTH_PREFIX else sym_variants
    prefer_freshest = prefix in (CACHE_WHALE_SNAPSHOT_PREFIX, CACHE_DERIVATIVES_TELEMETRY_PREFIX)
    disk_d: Optional[Dict[str, Any]] = None
    best_ts = -1.0
    for cand in variants:
        d = _rj(os.path.join(cachedb_dir, f"{prefix}{cand}.json"))
        if isinstance(d, dict):
            cand_d = d if d.get("symbol") else {**d, "symbol": cand}
            if not prefer_freshest:
                disk_d = cand_d
                break
            ts_v = _snap_ts(cand_d)
            if disk_d is None or ts_v > best_ts:
                disk_d, best_ts = cand_d, ts_v

    if snap is not None and disk_d is not None:
        if prefix == CACHE_ORDERBOOK_DEPTH_PREFIX:
            sym_u_p2 = clean_symbol_key(symbol)
            snap_sym_p2 = clean_symbol_key(obj_attr(snap, "symbol"))
            disk_sym_p2 = clean_symbol_key(disk_d.get("symbol"))
            if snap_sym_p2 and disk_sym_p2 == sym_u_p2 and snap_sym_p2 != sym_u_p2:
                return disk_d
            if (
                snap_sym_p2
                and (disk_sym_p2 == sym_u_p2) == (snap_sym_p2 == sym_u_p2)
                and _snap_ts(snap) > 0.0
                and _snap_ts(disk_d) > _snap_ts(snap) + 1.0
            ):
                return disk_d
        elif prefix in (CACHE_WHALE_SNAPSHOT_PREFIX, CACHE_DERIVATIVES_TELEMETRY_PREFIX):
            if _snap_ts(snap) > 0.0 and _snap_ts(disk_d) > _snap_ts(snap) + 1.0:
                return disk_d
    return snap if snap is not None else disk_d


def _upgrade_pretrade_tpo_vp_from_ohlc(
    cachedb_dir: str,
    sym_v: str,
    m24_d: Dict[str, Any],
    px_v: float,
    last_pt_v: Optional[Tuple[float, float]],
    ob_ts_v: float = 0.0,
    *,
    allow_network: bool = False,
) -> None:
    sym_u_pt = clean_symbol_key(sym_v) or str(sym_v or "").strip().upper()
    upgrade_tpo_vp_from_ohlc(
        sym_u_pt,
        m24_d,
        px_v,
        _time.time(),
        last_series_pt=last_pt_v,
        ob_ts_sec=ob_ts_v,
        respect_policy_network_gate=True,
        cache_dir=cachedb_dir,
        allow_network=allow_network,
    )


@dataclass
class PretradeEvaluationContext:
    """Shared normalized snapshot & state context passed across polymorphic PretradePillarCollectorBase stages."""

    symbol: str
    cachedb_dir: str
    regime_context: Optional[Any]
    dur_sec: Optional[float]
    session_policy: AssetSessionPolicy
    sym_quotes: Tuple[str, ...]
    is_usd_sym: bool
    is_eq_sym: bool
    sym_variants: List[str]
    quote_variants: List[str]
    rj: Callable[[str], Any]
    allow_network: Optional[bool] = False
    now_ts: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)
    pillar_sources: Dict[str, str] = field(default_factory=dict)
    pillar_timestamps: Dict[str, float] = field(default_factory=dict)
    source_timestamps: List[float] = field(default_factory=list)
    ob_snap: Any = None
    f_snap: Any = None
    w_snap: Any = None
    cur_p_items: Dict[str, Any] = field(default_factory=dict)
    live_cur_px: Optional[float] = None
    cp_ts_sec: float = 0.0
    vb_pos_ts: float = 0.0
    alt_series_24h: list = field(default_factory=list)
    arch_vals_7d: list = field(default_factory=list)
    m24: Dict[str, Any] = field(default_factory=dict)
    last_alt_pt: Optional[Tuple[float, float]] = None
    cur_px_est: float = 0.0
    rth_fresh_eq: bool = False
    eq_px_diverged_05pct: bool = False
    eq_chart_refreshed: bool = False
    cur_m_info: Dict[str, Any] = field(default_factory=dict)
    ob_ts_est: float = 0.0
    btc_m24: Dict[str, Any] = field(default_factory=dict)
    btc_vals_7d: list = field(default_factory=list)
    btc_live_px: Optional[float] = None
    m7d: Dict[str, Any] = field(default_factory=dict)
    is_sub24h_dup_7d: bool = False
    t_entry: Dict[str, Any] = field(default_factory=dict)
    has_st_tinfo: bool = False
    st_items_map: Dict[str, Any] = field(default_factory=dict)
    k_items_map: Dict[str, Any] = field(default_factory=dict)

    def record_pillar_source(
        self,
        pillar: str,
        source: str,
        ts: Optional[float] = None,
        *,
        include_in_snapshot_ts: bool = True,
    ) -> None:
        """Record pillar provenance and valid epoch timestamp on the evaluation context."""
        self.pillar_sources[pillar] = str(source or "cachedb")
        norm_ts = self.record_source_timestamp(pillar, ts)
        if norm_ts > 0.0 and include_in_snapshot_ts:
            self.source_timestamps.append(norm_ts)

    def record_source_timestamp(self, source_key: str, ts: Optional[float]) -> float:
        """Record a per-source timestamp without altering the snapshot-level gate timestamp."""
        norm_ts = normalize_epoch_sec(ts, default=0.0) or 0.0
        if norm_ts > 0.0:
            self.pillar_timestamps[source_key] = norm_ts
        return norm_ts


class PretradePillarCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for pre-trade telemetry pillar collectors (Open-Closed Principle)."""

    cls_name = "PretradePillarCollectorBase"

    def collect(self, cachedb_dir: str, symbol: str, *, ctx: Optional[PretradeEvaluationContext] = None, **kwargs: Any) -> Dict[str, Any]:
        if ctx is not None:
            self.evaluate(ctx)
            return ctx.telemetry
        return {}

    @abstractmethod
    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        """Enrich ``ctx.telemetry`` using the normalized snapshot in ``ctx``."""


class Pillar2AndRegimeStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Pillar 2 Orderbook/Funding/Whale + Pillar 3 Regime snapshots."""

    cls_name = "Pillar2AndRegimeStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        _ensure_fresh_pretrade_microstructure(
            ctx.cachedb_dir,
            ctx.symbol,
            ctx.quote_variants,
            ctx.is_usd_sym,
            ctx.is_eq_sym,
            ctx.rj,
            allow_network=ctx.allow_network,
        )
        cachedb_dir, symbol, regime_context, dur_sec, sym_variants, quote_variants, sym_quotes, is_usd_sym, is_eq_sym, _rj = (
            ctx.cachedb_dir,
            ctx.symbol,
            ctx.regime_context,
            ctx.dur_sec,
            ctx.sym_variants,
            ctx.quote_variants,
            ctx.sym_quotes,
            ctx.is_usd_sym,
            ctx.is_eq_sym,
            ctx.rj,
        )
        telemetry: Dict[str, Any] = {}
        ob_snap = f_snap = w_snap = None
        try:
            ob_snap = _fetch_pillar2_snapshot(
                cachedb_dir, symbol, get_orderbook_collector, CACHE_ORDERBOOK_DEPTH_PREFIX, sym_variants, quote_variants, _rj, now_ts=ctx.now_ts
            )
            if ob_snap is not None:
                raw_imb = obj_attr(ob_snap, "imbalance_ratio", obj_attr(ob_snap, "ob_imbalance", None))
                imb_f = clean_scale(raw_imb, default=None, allow_percent=True)
                if imb_f is not None:
                    bid_pct = max(0, min(100, round(imb_f * 100.0)))
                    ask_pct = 100 - bid_pct
                    telemetry["Orderbook Depth Imbalance"] = f"{bid_pct / 100.0:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
                ob_sym_raw = obj_attr(ob_snap, "symbol")
                ob_same_q = (
                    compatible_quote_suffixes(ob_sym_raw) == sym_quotes
                    if isinstance(ob_sym_raw, str) and ob_sym_raw.strip()
                    else is_usd_sym
                )
                bid_d_usd = safe_pos_float(obj_attr(ob_snap, "bid_depth_usd", 0.0), 0.0) or 0.0
                ask_d_usd = safe_pos_float(obj_attr(ob_snap, "ask_depth_usd", 0.0), 0.0) or 0.0
                tot_ob_depth = bid_d_usd + ask_d_usd
                for side_lbl, usd_attr, px_attr, side_d_usd in (
                    ("Ask", "largest_ask_wall_usd", "largest_ask_wall_price", ask_d_usd),
                    ("Bid", "largest_bid_wall_usd", "largest_bid_wall_price", bid_d_usd),
                ):
                    if not ob_same_q:
                        continue
                    eff_side_d = side_d_usd if side_d_usd > 0.0 else (tot_ob_depth * 0.5)
                    min_wall_usd = (
                        min(100_000.0, max(500.0 if is_eq_sym else 5_000.0, eff_side_d * 0.05))
                        if eff_side_d > 0.0
                        else 100_000.0
                    )
                    w_usd = safe_pos_float(obj_attr(ob_snap, usd_attr, 0.0), 0.0)
                    w_px = safe_pos_float(obj_attr(ob_snap, px_attr, 0.0), 0.0)
                    w_px_fmt = (
                        f"{w_px:.4f}".rstrip("0").rstrip(".")
                        if is_eq_sym and 1.0 <= w_px < 10.0 and abs(w_px - round(w_px, 2)) >= 1e-4
                        else (f"{w_px:,.2f}" if w_px >= 1.0 else f"{w_px:.6f}".rstrip("0").rstrip("."))
                    )
                    ob_span_pt = safe_pos_float(obj_attr(ob_snap, "book_span_pct", 0.0), 0.0) or 0.0
                    none_wall_lbl = (
                        f"None detected within +/-{ob_span_pt:g}%[level_cap]"
                        if 0.0 < ob_span_pt < 1.45
                        else "None detected within 2%"
                    )
                    telemetry[f"Orderbook {side_lbl} Wall Ahead"] = (
                        f"${w_usd:,.0f} @ ${w_px_fmt}"
                        if w_usd >= min_wall_usd and w_px > 0.0
                        else none_wall_lbl
                    )
        except Exception:
            pass

        try:
            f_snap = _fetch_pillar2_snapshot(
                cachedb_dir, symbol, get_derivatives_collector, CACHE_DERIVATIVES_TELEMETRY_PREFIX, sym_variants, quote_variants, _rj, now_ts=ctx.now_ts
            )
            if f_snap is not None:
                fr_f = safe_float(obj_attr(f_snap, "funding_rate", None), default=None)
                fr_ann_f = safe_float(obj_attr(f_snap, "funding_annualized_pct", None), default=None)
                oi_f = safe_pos_float(obj_attr(f_snap, "open_interest_usd", None), 0.0)
                if fr_f is not None and not (is_eq_sym and abs(fr_f) < 1e-12):
                    fr_period = _funding_period_label(fr_f, fr_ann_f, symbol=symbol)
                    telemetry["Perpetual Funding Rate"] = f"{fr_f * 100:+.4f}% ({fr_period})"
                if oi_f > 0.0:
                    telemetry["Derivatives Open Interest"] = (
                        f"${oi_f:,.0f} (Cboe options chain OI)" if is_eq_sym else f"${oi_f:,.0f}"
                    )
        except Exception:
            pass

        try:
            w_snap = _fetch_pillar2_snapshot(
                cachedb_dir, symbol, get_whale_collector, CACHE_WHALE_SNAPSHOT_PREFIX, sym_variants, quote_variants, _rj, now_ts=ctx.now_ts
            )
            if w_snap is not None:
                raw_lp = obj_attr(w_snap, "top_traders_long_pct", obj_attr(w_snap, "whale_long_pct", None))
                lp_f = clean_scale(raw_lp, default=None, allow_percent=True)
                lr_f = safe_float(obj_attr(w_snap, "top_traders_long_ratio", None), default=None)
                if lp_f is not None and lr_f is None and 0.0 < lp_f < 1.0:
                    lr_f = lp_f / (1.0 - lp_f)
                if lp_f is not None and lr_f is not None:
                    telemetry["Whale Positioning"] = f"{lp_f * 100.0:.1f}% Top Traders Long (L/S ratio: {lr_f:.2f})"
                tr_f = safe_float(obj_attr(w_snap, "taker_buy_sell_ratio", None), default=None)
                if tr_f is not None:
                    telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {tr_f:.2f}"
        except Exception:
            pass

        if regime_context is not None:
            try:
                trend = obj_attr(regime_context, "resolved_trend", None) or obj_attr(regime_context, "regime", None)
                dec = obj_attr(regime_context, "decision", None)
                gf = safe_float(obj_attr(dec, "gradient", None), default=None) if dec is not None else None
                ef = safe_float(obj_attr(dec, "epsilon", None), default=None) if dec is not None else None
                trend_str = f"{trend}" if trend else "unspecified"
                if gf is not None and ef is not None:
                    gf_s = f"{gf:+.2e}" if 0.0 < abs(gf) < 1e-4 else f"{gf:+.4f}"
                    ef_s = f"{ef:.2e}" if 0.0 < abs(ef) < 1e-4 else f"{ef:.4f}"
                    trend_str += f" (gradient={gf_s}, epsilon={ef_s})"
                telemetry["Technical Trend Regime"] = trend_str
            except Exception:
                pass

        df = safe_pos_float(dur_sec, 0.0)
        if df > 0.0:
            telemetry["Trend Duration"] = f"{df / 3600.0:.1f} hours"
        ctx.telemetry, ctx.ob_snap, ctx.f_snap, ctx.w_snap = telemetry, ob_snap, f_snap, w_snap
        for snap_lbl, snap_obj in (("orderbook", ctx.ob_snap), ("derivatives", ctx.f_snap), ("whale", ctx.w_snap)):
            if snap_obj is not None:
                ctx.record_pillar_source(
                    snap_lbl,
                    "memory_cache" if not isinstance(snap_obj, dict) else "cachedb",
                    _snap_ts(snap_obj),
                )


class PretradePriceSeries24hStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Live price resolution and 24h price-series normalization."""

    cls_name = "PretradePriceSeries24hStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        ensure_cacheprice_classes_bound()
        eff_allow_net = bool(ctx.allow_network)
        cachedb_dir = ctx.cachedb_dir
        symbol = ctx.symbol
        is_eq_sym = ctx.is_eq_sym
        cur_p_data = ctx.rj(os.path.join(cachedb_dir, _mc_mod.CacheCurrentPriceManager.DEFAULT_FILENAME))
        ctx.cur_p_items = (
            cur_p_data.get("items", {})
            if isinstance(cur_p_data, dict) and isinstance(cur_p_data.get("items"), dict)
            else {}
        )
        cur_p_items = ctx.cur_p_items
        live_cur_px = 0.0
        cp_ts_sec = 0.0
        for cand in ctx.quote_variants:
            raw_cp = casefold_get(cur_p_items, cand)
            if (px_v := extract_current_price_item(raw_cp)) is not None and px_v > 0.0:
                live_cur_px = px_v
                if isinstance(raw_cp, list) and raw_cp and (cp_pt := parse_ts_px_row(raw_cp[-1])) is not None:
                    cp_ts_sec = normalize_epoch_sec(cp_pt[0], default=0.0) or 0.0
                break
        vb_pos_ts = 0.0
        vb_path, vb_d, _ = (
            load_venue_balances_state(cachedb_dir, ctx.rj, require_exists=False)
            if (live_cur_px <= 0.0 or is_eq_sym)
            else (None, {}, 0.0)
        )
        if vb_path and vb_d:
            if eff_allow_net and is_eq_sym and ctx.rj is _default_read_json_cached:
                refresh_stale_t212_cached_positions(vb_d, vb_path, _time.time())
            for vb_k, vb_v in vb_d.items():
                if isinstance(vb_k, str) and vb_k.startswith("pos:") and isinstance(vb_v, dict):
                    if clean_symbol_key(vb_v.get("symbol") or vb_k.split(":", 2)[-1]) == clean_symbol_key(symbol):
                        vb_px = safe_pos_float(vb_v.get("price"), default=0.0) or 0.0
                        vb_ts = safe_pos_float(vb_v.get("ts"), default=0.0) or 0.0
                        if vb_px > 0.0:
                            if (
                                vb_ts > 0.0
                                and cp_ts_sec > 0.0
                                and abs(round(vb_px, 4) - round(live_cur_px, 4)) < 1e-4
                                and cp_ts_sec < vb_ts
                            ):
                                vb_ts = cp_ts_sec
                            vb_pos_ts = vb_ts
                            if (
                                live_cur_px <= 0.0
                                or abs(round(live_cur_px, 4) - round(vb_px, 4)) < 1e-4
                                or (cp_ts_sec > 0.0 and vb_ts > cp_ts_sec)
                            ):
                                live_cur_px = round(vb_px, 4) if is_eq_sym else vb_px
                            break
        ctx.live_cur_px, ctx.cp_ts_sec, ctx.vb_pos_ts = live_cur_px, cp_ts_sec, vb_pos_ts
        alt_series_24h, arch_vals_7d = load_symbol_price_series(
            cachedb_dir, symbol, current_price_items=cur_p_items, include_7d=True
        )
        bars_4h_pt = LAST_7D_4H_BARS_BY_SYM.get(str(symbol).strip().upper()) if arch_vals_7d else None
        eq_7d_series = bars_4h_pt
        m24 = (
            _call_compute_24h(alt_series_24h, is_equity=is_eq_sym, series_7d_ts_px=eq_7d_series)
            if alt_series_24h
            else {}
        )
        last_alt_pt = parse_ts_px_row(alt_series_24h[-1]) if alt_series_24h else None
        cur_px_est = last_alt_pt[1] if last_alt_pt is not None else 0.0
        if not cur_px_est or cur_px_est <= 0.0:
            cur_px_est = live_cur_px

        rth_fresh_eq = True
        eq_px_diverged_05pct = False
        eq_chart_refreshed = False
        sym_u_eq = str(symbol).strip().upper()
        memo_chart_eq = (
            EQ_YAHOO_CHART_MEMO.get(sym_u_eq)
            if (eff_allow_net and is_eq_sym and is_live_network_backfill_enabled())
            else None
        )
        raw_alt_last_ts_ms = last_alt_pt[0] if last_alt_pt is not None else 0.0
        now_s_eq_chk = _time.time()
        eq_rth_open_chk = bool(is_symbol_market_open(symbol, now_s_eq_chk)) if is_eq_sym else False
        memo_ttl_eq = 240.0 if (eq_rth_open_chk or memo_chart_eq is None) else 1800.0
        memo_chart_fresh_eq = bool(memo_chart_eq is not None and 0.0 <= (now_s_eq_chk - memo_chart_eq[0]) < memo_ttl_eq)
        if (
            eff_allow_net
            and is_eq_sym
            and is_live_network_backfill_enabled()
            and (
                not m24
                or "chg_24h_pct" not in m24
                or "chg_1h_pct" not in m24
                or m24.get("vp_source") == "TPO"
                or memo_chart_eq is not None
                or eq_rth_open_chk
                or raw_alt_last_ts_ms <= 0.0
                or (math.trunc(now_s_eq_chk * 1000) - raw_alt_last_ts_ms) > 5 * 60 * 1000
            )
        ):
            try:
                pts_5d_eq, _, pts_24_eq, _, _, _, _ = TelemetryCollectorFactory.get_yahoo_chart_collector().collect(
                    cachedb_dir,
                    sym_u_eq,
                    raw_series_24h=list(alt_series_24h),
                    now_ts=now_s_eq_chk,
                    eq_rth_open=eq_rth_open_chk,
                    memo_chart=memo_chart_eq,
                    memo_chart_fresh=memo_chart_fresh_eq,
                    persist=False,
                )
                if pts_5d_eq:
                    alt_series_24h = list(pts_24_eq)
                    arch_vals_7d = [pt[1] for pt in pts_5d_eq]
                    eq_7d_series = pts_5d_eq
                    eq_chart_refreshed = len(alt_series_24h) >= 5
                    m24 = _call_compute_24h(alt_series_24h, is_equity=True, series_7d_ts_px=eq_7d_series)
                    last_alt_pt = parse_ts_px_row(alt_series_24h[-1]) if alt_series_24h else None
                    if last_alt_pt is not None:
                        cur_px_est = last_alt_pt[1]
            except Exception:
                pass

        if is_eq_sym and last_alt_pt is not None:
            now_ms_eq = math.trunc(_time.time() * 1000)
            eq_rth_open = bool(is_symbol_market_open(symbol, now_ms_eq / 1000.0))
            eff_live_px_ts = max(cp_ts_sec, vb_pos_ts)
            if (
                live_cur_px > 0.0
                and last_alt_pt[0] > 1e11
                and eff_live_px_ts > 0.0
                and (last_alt_pt[0] / 1000.0) > eff_live_px_ts + 60.0
            ):
                live_cur_px = round(last_alt_pt[1], 4)
            rth_fresh_eq = (
                (now_ms_eq - last_alt_pt[0]) <= (4 * 3600 * 1000 if eq_rth_open else 20 * 60 * 1000)
                and (live_cur_px <= 0.0 or abs(live_cur_px - last_alt_pt[1]) / max(live_cur_px, last_alt_pt[1]) <= 0.005)
            )
            if (
                rth_fresh_eq
                and live_cur_px > 0.0
                and last_alt_pt[0] > 1e11
                and (vb_pos_ts <= 0.0 or (last_alt_pt[0] / 1000.0) > vb_pos_ts)
            ):
                live_cur_px = round(last_alt_pt[1], 4)
            if not rth_fresh_eq and m24:
                for k_stale_eq in STALE_SHORT_24H_KEYS:
                    m24.pop(k_stale_eq, None)
            if live_cur_px > 0.0 and not _prices_diverged(live_cur_px, last_alt_pt[1], threshold=0.35) and abs(round(live_cur_px, 4) - round(last_alt_pt[1], 4)) > 1e-4:
                cur_px_est = round(live_cur_px, 4)
                if rth_fresh_eq:
                    vb_pos_ts_ms = math.trunc(vb_pos_ts * 1000)
                    eq_pt_ts_ms = (
                        vb_pos_ts_ms
                        if vb_pos_ts > 0.0 and vb_pos_ts_ms > last_alt_pt[0]
                        else now_ms_eq
                    )
                    if (eq_pt_ts_ms - last_alt_pt[0]) > 300 * 1000:
                        eq_pt_ts_ms = math.trunc(last_alt_pt[0] + 1000)
                    alt_series_24h = list(alt_series_24h) + [[eq_pt_ts_ms, cur_px_est]]
                    m24 = _call_compute_24h(alt_series_24h, is_equity=True, series_7d_ts_px=eq_7d_series) or m24
                elif m24:
                    alt_series_24h, m24, eq_px_diverged_05pct = ctx.session_policy.apply_post_rth_price_override(
                        alt_series_24h, m24, live_cur_px
                    )
                    recalculate_live_price_geometry(m24, cur_px_est)
            if rth_fresh_eq and m24 and cur_px_est > 0.0:
                recalculate_live_price_geometry(m24, cur_px_est)
            if m24 and not any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct")):
                for k_stale_15m in ("recent_tape_15m", "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct"):
                    m24.pop(k_stale_15m, None)

        _upgrade_pretrade_tpo_vp_from_ohlc(
            cachedb_dir, symbol, m24, cur_px_est, last_alt_pt, allow_network=eff_allow_net
        )
        (
            ctx.alt_series_24h,
            ctx.arch_vals_7d,
            ctx.m24,
            ctx.last_alt_pt,
            ctx.cur_px_est,
            ctx.rth_fresh_eq,
            ctx.eq_px_diverged_05pct,
            ctx.eq_chart_refreshed,
        ) = (
            alt_series_24h,
            arch_vals_7d,
            m24,
            last_alt_pt,
            cur_px_est,
            rth_fresh_eq,
            eq_px_diverged_05pct,
            eq_chart_refreshed,
        )
        last_s24_ts = (normalize_epoch_sec(ctx.last_alt_pt[0], default=0.0) or 0.0) if ctx.last_alt_pt is not None else 0.0
        best_px_ts = max(ctx.cp_ts_sec, ctx.vb_pos_ts, last_s24_ts)
        if ctx.m24 or ctx.live_cur_px:
            ctx.record_pillar_source("price_series_24h", "cachedb", best_px_ts)


class PretradeMicrostructureReconcileStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Spot-vs-orderbook wall rescaling and microstructure reconciliation."""

    cls_name = "PretradeMicrostructureReconcileStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        cachedb_dir, symbol, telemetry, m24, cur_px_est, last_alt_pt, is_usd_sym, is_eq_sym, allow_network = (
            ctx.cachedb_dir,
            ctx.symbol,
            ctx.telemetry,
            ctx.m24,
            ctx.cur_px_est,
            ctx.last_alt_pt,
            ctx.is_usd_sym,
            ctx.is_eq_sym,
            bool(ctx.allow_network),
        )
        ob_ts_est = 0.0
        cur_m_info = load_symbol_microstructure(cachedb_dir, symbol, ctx.rj, ob=ctx.ob_snap, deriv=ctx.f_snap, whale=ctx.w_snap)
        if not cur_m_info:
            ctx.cur_m_info, ctx.cur_px_est, ctx.ob_ts_est = {}, cur_px_est, ob_ts_est
            return
        now_ts_s = _time.time()
        ob_ts_est = safe_pos_float(cur_m_info.pop("_ob_ts", None), default=0.0) or 0.0
        ob_raw_ts_m = safe_pos_float(cur_m_info.pop("_ob_raw_ts", None), default=0.0) or 0.0
        deriv_ts_m = safe_pos_float(cur_m_info.pop("_deriv_ts", None), default=0.0) or 0.0
        whale_ts_m = safe_pos_float(cur_m_info.pop("_whale_ts", None), default=0.0) or 0.0
        def _refresh_wall_telemetry(use_sub_ten_decimals: bool = False) -> None:
            for side_lbl, usd_k, px_k in (("Ask", "ask_wall_usd", "ask_wall_px"), ("Bid", "bid_wall_usd", "bid_wall_px")):
                w_px_rc = safe_pos_float(cur_m_info.get(px_k), 0.0)
                w_usd_rc = safe_pos_float(cur_m_info.get(usd_k), 0.0)
                if w_px_rc > 0.0 and w_usd_rc > 0.0 and str(telemetry.get(f"Orderbook {side_lbl} Wall Ahead") or "").startswith("$"):
                    w_px_fmt = (
                        f"{w_px_rc:.4f}".rstrip("0").rstrip(".")
                        if use_sub_ten_decimals and 1.0 <= w_px_rc < 10.0 and abs(w_px_rc - round(w_px_rc, 2)) >= 1e-4
                        else (f"{w_px_rc:,.2f}" if w_px_rc >= 1.0 else f"{w_px_rc:.6f}".rstrip("0").rstrip("."))
                    )
                    telemetry[f"Orderbook {side_lbl} Wall Ahead"] = f"${w_usd_rc:,.0f} @ ${w_px_fmt}"

        is_exact_ob_sym = clean_symbol_key(cur_m_info.get("_source_cand") or symbol) == clean_symbol_key(symbol)
        cur_px_est, adopted_est_px, rec_action = reconcile_spot_and_orderbook_microstructure(
            symbol,
            cur_px_est,
            cur_m_info,
            is_usd_sym=is_usd_sym,
            is_eq_sym=is_eq_sym,
            is_exact_ob_sym=is_exact_ob_sym,
            ob_ts_sec=ob_ts_est,
            now_ts=now_ts_s,
            last_series_pt=last_alt_pt,
            extra_ts_sec=safe_pos_float(ctx.cp_ts_sec, default=0.0) or 0.0,
        )
        if rec_action == "adopted_ob_mid" and adopted_est_px is not None:
            if m24:
                recalculate_live_price_geometry(m24, adopted_est_px)
        elif rec_action == "rescaled_fiat_walls":
            if m24:
                recalculate_live_price_geometry(m24, cur_px_est)
            _refresh_wall_telemetry()
        elif rec_action == "aligned_equity_ob":
            _refresh_wall_telemetry(use_sub_ten_decimals=True)
        if m24 and m24.get("vp_source") == "TPO":
            _upgrade_pretrade_tpo_vp_from_ohlc(
                cachedb_dir, symbol, m24, cur_px_est, last_alt_pt, ob_ts_est, allow_network=allow_network
            )
            if cur_px_est > 0.0 and m24.get("vp_source") == "REAL_VOL":
                recalculate_live_price_geometry(m24, cur_px_est)
        ctx.cur_m_info, ctx.cur_px_est, ctx.ob_ts_est = cur_m_info, cur_px_est, ob_ts_est
        if ctx.cur_m_info:
            ctx.record_pillar_source("microstructure", "cachedb", ctx.ob_ts_est)
            has_ob_m = cur_m_info.get("ob_imbalance") is not None or cur_m_info.get("bid_depth_usd") is not None
            has_deriv_m = (
                ((safe_pos_float(cur_m_info.get("oi_usd"), default=0.0) or 0.0) > 0.0)
                if is_eq_sym
                else (cur_m_info.get("funding_rate") is not None or cur_m_info.get("oi_usd") is not None)
            )
            has_whale_m = (
                cur_m_info.get("whale_long_pct") is not None or cur_m_info.get("taker_buy_sell_ratio") is not None
            )
            for src_k_m, src_ts_m, present_m in (
                ("orderbook", ob_raw_ts_m or ob_ts_est, has_ob_m),
                ("derivatives", deriv_ts_m, has_deriv_m),
                ("whale", whale_ts_m, has_whale_m),
            ):
                if present_m and src_ts_m > 0.0 and src_k_m not in ctx.pillar_timestamps:
                    ctx.record_source_timestamp(src_k_m, src_ts_m)


class Pretrade24hAndBtcFormatStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): 24h Multi-Horizon Price Evolution, Volume Profile, and BTC 24h benchmark."""

    cls_name = "Pretrade24hAndBtcFormatStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        cachedb_dir, symbol, telemetry, m24, alt_series_24h, cur_p_items, _rj = (
            ctx.cachedb_dir,
            ctx.symbol,
            ctx.telemetry,
            ctx.m24,
            ctx.alt_series_24h,
            ctx.cur_p_items,
            ctx.rj,
        )
        btc_m24: Dict[str, Any] = {}
        btc_vals_7d: list = []
        btc_live_px: float = 0.0
        if not m24:
            ctx.btc_m24, ctx.btc_vals_7d, ctx.btc_live_px = btc_m24, btc_vals_7d, btc_live_px
            return

        has_any_m24_h = any(
            safe_float(m24.get(k), default=None) is not None
            for k in ("chg_15m_pct", "chg_30m_pct", "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct")
        )
        h_parts = [
            f"{lbl}={safe_float(m24.get(k), 0.0):+.2f}%"
            for lbl, k, req in (
                ("1m", "chg_1m_pct", False), ("3m", "chg_3m_pct", False), ("5m", "chg_5m_pct", False),
                ("15m", "chg_15m_pct", not has_any_m24_h), ("30m", "chg_30m_pct", not has_any_m24_h),
                ("1h", "chg_1h_pct", not has_any_m24_h), ("3h", "chg_3h_pct", not has_any_m24_h),
                ("6h", "chg_6h_pct", False), ("12h", "chg_12h_pct", False), ("24h", "chg_24h_pct", False),
            )
            if req or safe_float(m24.get(k), default=None) is not None
        ]
        rng_str = f"24h Range=[${m24.get('low_24h')}, ${m24.get('high_24h')}]"
        geom_parts = [
            f"{g_lbl}={g_val:+.2f}%" if g_signed else f"{g_lbl}={m24[g_k]}%"
            for g_lbl, g_k, g_signed in (
                ("pos", "pos_24h_pct", False), ("dd_high", "dd_24h_high_pct", True),
                ("bounce_low", "bounce_24h_low_pct", True), ("span", "span_24h_pct", False),
            )
            if (g_val := safe_float(m24.get(g_k), default=None)) is not None
        ]
        if geom_parts:
            rng_str += f" ({', '.join(geom_parts)})"
        if isinstance(m24.get("recent_tape_15m"), (list, tuple)) and m24["recent_tape_15m"]:
            rng_str += f", tape_15m=[{', '.join(str(x) for x in m24['recent_tape_15m'])}]"
        telemetry["24h Multi-Horizon Price Evolution"] = f"{', '.join(h_parts)}, {rng_str}"

        if m24.get("poc_px") is not None:
            sec_poc_val = m24.get("sec_poc_px") if m24.get("sec_poc_px") is not None else m24.get("hvn2_px")
            sec_poc_s = f", sec_POC=${sec_poc_val}" if sec_poc_val is not None else ""
            lvn_s = f", LVN=${m24['lvn_px']}" if m24.get("lvn_px") is not None else ""
            vp_src_s = f", vp_source={m24['vp_source']}" if m24.get("vp_source") else ""
            telemetry["24h Volume Profile (POC / Value Area)"] = (
                f"POC=${m24.get('poc_px')} ({safe_float(m24.get('poc_dist_pct'), 0.0):+.2f}%){sec_poc_s}{lvn_s}, "
                f"VAL=${m24.get('val_px')} ({safe_float(m24.get('val_dist_pct'), 0.0):+.2f}%), "
                f"VAH=${m24.get('vah_px')} ({safe_float(m24.get('vah_dist_pct'), 0.0):+.2f}%), "
                f"va_pos={m24.get('va_pos_pct')}%{format_va_hold_suffix(m24)}{vp_src_s}"
            )

        if atr_parts := format_atr_parts(m24):
            telemetry["Multi-Timeframe ATR"] = ", ".join(atr_parts)
        sess_struct_parts: List[str] = []
        if sv_summary := format_session_vwap_summary(m24, cur_px=ctx.cur_px_est):
            sess_struct_parts.append(f"Session VWAP={sv_summary}")
        if sess_ex_parts := format_session_extremes_parts(m24):
            sess_struct_parts.append(f"sessions({', '.join(sess_ex_parts)})")
        if sess_struct_parts:
            telemetry["Session VWAP & Intraday Structure"] = " | ".join(sess_struct_parts)

        if canonical_base_asset(symbol) != "BTC" and not is_equity_symbol(symbol):
            btc_series_24h, btc_vals_7d = load_symbol_price_series(
                cachedb_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=True
            )
            btc_m24 = compute_24h_price_metrics(btc_series_24h) if btc_series_24h else {}
            btc_live_px = extract_current_price_item(casefold_get(cur_p_items, "BTCUSDC")) or 0.0
            if btc_m24 and btc_series_24h:
                btc_last_pt = parse_ts_px_row(btc_series_24h[-1])
                if btc_last_pt is not None and btc_last_pt[1] > 0.0:
                    btc_live_px = btc_last_pt[1]
                btc_m_info = load_symbol_microstructure(cachedb_dir, "BTCUSDC", _rj)
                btc_ob_ts = safe_pos_float((btc_m_info or {}).get("_ob_ts"), default=0.0) or 0.0
                if isinstance(btc_m_info, dict):
                    for _k_int_ts in ("_ob_ts", "_ob_raw_ts", "_deriv_ts", "_whale_ts"):
                        btc_m_info.pop(_k_int_ts, None)
                btc_ob_mid = safe_pos_float((btc_m_info or {}).get("ob_mid_price"), default=None)
                btc_ob_fresh = is_ob_live_and_newer(btc_ob_ts, _time.time(), btc_last_pt)
                if (
                    btc_ob_mid is not None
                    and btc_ob_fresh
                    and btc_last_pt is not None
                    and btc_last_pt[1] > 0.0
                    and abs(btc_ob_mid - btc_last_pt[1]) / max(btc_ob_mid, btc_last_pt[1]) <= 0.01
                ):
                    btc_live_px = btc_ob_mid
                    recalculate_live_price_geometry(btc_m24, btc_ob_mid)
                    if btc_vals_7d and not _prices_diverged(btc_ob_mid, safe_float(btc_vals_7d[-1], 0.0)):
                        btc_vals_7d = list(btc_vals_7d) + [btc_ob_mid]
            if btc_m24:
                vs_parts = [
                    f"{h_lbl}={m_v - b_v:+.2f}%"
                    for h_lbl, h_k in (
                        ("5m", "chg_5m_pct"), ("15m", "chg_15m_pct"), ("1h", "chg_1h_pct"), ("3h", "chg_3h_pct"),
                        ("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct"),
                    )
                    if (m_v := safe_float(m24.get(h_k), default=None)) is not None
                    and (b_v := safe_float(btc_m24.get(h_k), default=None)) is not None
                ]
                cross_btc = compute_cross_btc_metrics(alt_series_24h, btc_series_24h)
                cross_parts = [
                    f"{c_lbl}={c_val:+.2f}%" if c_signed else f"{c_lbl}={cross_btc[c_k]}%"
                    for c_lbl, c_k, c_signed in (
                        ("alt_btc_pos_24h", "alt_btc_pos_24h_pct", False),
                        ("alt_btc_dd_high", "alt_btc_dd_24h_high_pct", True),
                        ("alt_btc_bounce_low", "alt_btc_bounce_24h_low_pct", True),
                    )
                    if (c_val := safe_float(cross_btc.get(c_k), default=None)) is not None
                ]
                for sc_lbl, sc_k in (("beta", "beta_btc_24h"), ("corr", "corr_btc_24h")):
                    if (sc_val := safe_float(cross_btc.get(sc_k), default=None)) is not None:
                        cross_parts.append(f"{sc_lbl}={sc_val:.2f}")
                if vs_parts or cross_parts:
                    rel_str = f"vs_btc({', '.join(vs_parts)})" if vs_parts else ""
                    if cross_parts:
                        rel_str = f"{rel_str} | {', '.join(cross_parts)}" if rel_str else ", ".join(cross_parts)
                    telemetry["Relative Strength vs BTC (24h Benchmark)"] = rel_str

        if m24.get("bottom_structure_6h"):
            _, b_parts = format_bottom_structure_6h_parts(m24, prior_label="prior_6h_low", px_prefix="$")
            if b_parts:
                telemetry["6h Intraday Bottom Structure"] = ", ".join(b_parts)
        ctx.btc_m24, ctx.btc_vals_7d, ctx.btc_live_px = btc_m24, btc_vals_7d, btc_live_px


class PretradeStructural7dStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): 7d Structural Context, Swing Waves, and 7d Relative Strength vs BTC."""

    cls_name = "PretradeStructural7dStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        (
            cachedb_dir,
            symbol,
            telemetry,
            m24,
            arch_vals_7d,
            cur_px_est,
            last_alt_pt,
            ob_ts_est,
            btc_m24,
            btc_vals_7d,
            btc_live_px,
            sym_variants,
            cur_p_items,
            _rj,
            allow_network,
        ) = (
            ctx.cachedb_dir,
            ctx.symbol,
            ctx.telemetry,
            ctx.m24,
            list(ctx.arch_vals_7d),
            ctx.cur_px_est,
            ctx.last_alt_pt,
            ctx.ob_ts_est,
            ctx.btc_m24,
            list(ctx.btc_vals_7d),
            ctx.btc_live_px or 0.0,
            ctx.sym_variants,
            ctx.cur_p_items,
            ctx.rj,
            bool(ctx.allow_network),
        )
        is_eq = is_equity_symbol(symbol)
        eq_div_thr = 0.35 if is_eq else 0.10
        if cur_px_est > 0.0 and arch_vals_7d:
            if _prices_diverged(cur_px_est, safe_float(arch_vals_7d[-1], 0.0), threshold=eq_div_thr):
                arch_vals_7d = []
            elif abs(cur_px_est - safe_float(arch_vals_7d[-1], 0.0)) > 1e-9:
                arch_vals_7d = list(arch_vals_7d) + [cur_px_est]
        m7d: Dict[str, Any] = (
            compute_7d_range_metrics(arch_vals_7d, decimals=4 if is_eq else 6)
            if arch_vals_7d and len(arch_vals_7d) >= 2
            else {}
        )
        partial_w_7d_span = False
        w_data = _rj(os.path.join(cachedb_dir, CACHE_WATCHER_ALL_PRICE_FILENAME))
        if isinstance(w_data, dict) and isinstance(w_items := w_data.get("items"), dict):
            sym_quotes_7d = compatible_quote_suffixes(symbol)
            for ck in sym_variants:
                w_ser = casefold_get(w_items, ck)
                if not isinstance(w_ser, (list, tuple)) or len(w_ser) < 2:
                    continue
                vals_7d = [
                    px_w
                    for pt in w_ser
                    if isinstance(pt, (list, tuple)) and len(pt) > 1 and (px_w := safe_pos_float(pt[1], default=None)) is not None
                ]
                if len(vals_7d) < 2:
                    continue
                if clean_symbol_key(ck) != clean_symbol_key(symbol) and (
                    compatible_quote_suffixes(ck) != sym_quotes_7d or (cur_px_est > 0.0 and _prices_diverged(cur_px_est, vals_7d[-1], threshold=eq_div_thr))
                ):
                    continue
                w0, w1 = parse_ts_px_row(w_ser[0]), parse_ts_px_row(w_ser[-1])
                w_span_7d = (w1[0] - w0[0]) if (w0 is not None and w1 is not None) else -1.0
                arch_span_7d_pt = LAST_7D_SPAN_MS_BY_SYM.get(str(symbol).strip().upper(), 0.0)
                if m7d and (is_eq or arch_span_7d_pt >= 5 * 86400 * 1000 or (0.0 <= w_span_7d < 5 * 86400 * 1000)):
                    continue
                if cur_px_est > 0.0 and not _prices_diverged(cur_px_est, vals_7d[-1], threshold=eq_div_thr) and abs(cur_px_est - vals_7d[-1]) > 1e-9:
                    vals_7d = list(vals_7d) + [cur_px_est]
                if m7d_w := compute_7d_range_metrics(vals_7d, decimals=4 if is_eq else 6):
                    m7d = m7d_w
                    partial_w_7d_span = 0.0 <= w_span_7d < 5 * 86400 * 1000
                    break
        if (
            allow_network
            and not is_eq
            and is_live_network_backfill_enabled()
            and (not m7d or partial_w_7d_span or 0.0 < LAST_7D_SPAN_MS_BY_SYM.get(str(symbol).strip().upper(), 0.0) < 5 * 86400 * 1000)
        ):
            try:
                now_s_7d = _time.time()
                if (
                    last_alt_pt is None
                    or abs(now_s_7d * 1000.0 - last_alt_pt[0]) <= 30 * 60 * 1000.0
                    or (ob_ts_est > 0.0 and abs(now_s_7d - ob_ts_est) <= 30 * 60.0)
                ):
                    sym_u_7d = clean_symbol_key(symbol) or str(symbol).strip().upper()
                    ohlc_fn_7d = resolve_fallback_ohlc_fn(sym_u_7d)
                    if callable(ohlc_fn_7d):
                        vals_7d_ohlc = load_stitched_7d_ohlc_closes(ohlc_fn_7d, sym_u_7d, now_s_7d, m24, cur_px_est)
                        if len(vals_7d_ohlc) >= 2 and (not m7d or len(vals_7d_ohlc) >= 24):
                            if m7d_ohlc := compute_7d_range_metrics(vals_7d_ohlc, decimals=6):
                                m7d = m7d_ohlc
            except Exception:
                pass
        if m7d:
            if m24.get("low_24h") is not None and m24.get("high_24h") is not None:
                m7d["low_24h"], m7d["high_24h"] = m24["low_24h"], m24["high_24h"]
            low_7d_v = safe_pos_float(m7d.get("low_7d"), 0.0) or cur_px_est
            high_7d_v = safe_pos_float(m7d.get("high_7d"), 0.0) or cur_px_est
            if cur_px_est > 0.0 and (
                low_7d_v <= cur_px_est <= high_7d_v
                or not (_prices_diverged(cur_px_est, low_7d_v, threshold=eq_div_thr) and _prices_diverged(cur_px_est, high_7d_v, threshold=eq_div_thr))
            ):
                recalculate_live_price_geometry(m7d, cur_px_est)
        is_sub24h_dup_7d = (
            bool(m24)
            and "chg_24h_pct" not in m24
            and m24.get("low_24h") is not None
            and (m7d.get("low_7d"), m7d.get("high_7d")) == (m24.get("low_24h"), m24.get("high_24h"))
        )
        if m7d and not is_sub24h_dup_7d:
            if canonical_base_asset(symbol) != "BTC" and not is_eq:
                if not btc_vals_7d and not btc_m24:
                    _, btc_vals_7d = load_symbol_price_series(
                        cachedb_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=True
                    )
                    if btc_live_px <= 0.0:
                        btc_live_px = extract_current_price_item(casefold_get(cur_p_items, "BTCUSDC")) or 0.0
                eff_btc_vals_7d = list(btc_vals_7d)
                btc_arch_span_7d = LAST_7D_SPAN_MS_BY_SYM.get("BTCUSDC", 0.0)
                if (
                    (len(eff_btc_vals_7d) < 2 or btc_arch_span_7d < 5 * 86400 * 1000)
                    and isinstance(w_data, dict)
                    and isinstance(w_items_b := w_data.get("items"), dict)
                ):
                    w_btc = casefold_get(w_items_b, "BTCUSDC")
                    if isinstance(w_btc, (list, tuple)) and len(w_btc) >= 2:
                        wb0, wb1 = parse_ts_px_row(w_btc[0]), parse_ts_px_row(w_btc[-1])
                        if wb0 is not None and wb1 is not None and (wb1[0] - wb0[0]) >= 5 * 86400 * 1000:
                            w_b_vals = [
                                px_b
                                for pt in w_btc
                                if isinstance(pt, (list, tuple)) and len(pt) > 1 and (px_b := safe_pos_float(pt[1], default=None)) is not None
                            ]
                            if len(w_b_vals) >= 2:
                                eff_btc_vals_7d = w_b_vals
                if (
                    eff_btc_vals_7d
                    and btc_live_px > 0.0
                    and not _prices_diverged(btc_live_px, eff_btc_vals_7d[-1])
                    and abs(btc_live_px - eff_btc_vals_7d[-1]) > 1e-9
                ):
                    eff_btc_vals_7d = list(eff_btc_vals_7d) + [btc_live_px]
                btc_m7d = compute_7d_range_metrics(eff_btc_vals_7d, decimals=6) if len(eff_btc_vals_7d) >= 2 else {}
                if btc_m7d and btc_m24.get("low_24h") is not None and btc_m24.get("high_24h") is not None:
                    btc_m7d["low_24h"], btc_m7d["high_24h"] = btc_m24["low_24h"], btc_m24["high_24h"]
                    recalculate_live_price_geometry(
                        btc_m7d,
                        btc_live_px if btc_live_px > 0.0 else (safe_pos_float(eff_btc_vals_7d[-1], 0.0) or 0.0),
                    )
                m7_chg = safe_float(m7d.get("chg_7d_pct"), default=None)
                b7_chg = safe_float((btc_m7d or {}).get("chg_7d_pct"), default=None)
                if m7_chg is not None and b7_chg is not None:
                    cur_rel_btc = telemetry.get("Relative Strength vs BTC (24h Benchmark)")
                    if not cur_rel_btc:
                        telemetry["Relative Strength vs BTC (24h Benchmark)"] = f"vs_btc(7d={m7_chg - b7_chg:+.2f}%)"
                    elif cur_rel_btc.startswith("vs_btc("):
                        telemetry["Relative Strength vs BTC (24h Benchmark)"] = cur_rel_btc.replace(
                            ")", f", 7d={m7_chg - b7_chg:+.2f}%)", 1
                        )
                    else:
                        telemetry["Relative Strength vs BTC (24h Benchmark)"] = (
                            f"vs_btc(7d={m7_chg - b7_chg:+.2f}%) | {cur_rel_btc}"
                        )
            p7_parts = [
                f"7d={safe_float(m7d.get('chg_7d_pct'), 0.0):+.2f}%",
                (
                    f"7d Range=[${m7d.get('low_7d')}, ${m7d.get('high_7d')}] "
                    f"(pos={m7d.get('pos_7d_pct')}%, dd_high={safe_float(m7d.get('dd_7d_high_pct'), 0.0):+.2f}%, "
                    f"bounce_low={safe_float(m7d.get('bounce_7d_low_pct'), 0.0):+.2f}%, span={m7d.get('span_7d_pct')}%)"
                ),
            ]
            if m7d.get("swing_low_px") is not None and m7d.get("swing_high_px") is not None:
                maj_sfx = (
                    f", major_low=${m7d.get('major_swing_low_px')}"
                    f"(rally={safe_float(m7d.get('rally_from_major_low_pct'), 0.0):+.2f}%)"
                    if m7d.get("rally_from_major_low_pct") is not None and m7d.get("major_swing_low_px") is not None
                    else ""
                )
                hist_sfx = ""
                if isinstance(m7d.get("recent_swings_7d"), list) and m7d["recent_swings_7d"]:
                    amp_s = (
                        f", avg_amp={m7d.get('avg_swing_amp_7d_pct')}%"
                        if m7d.get("avg_swing_amp_7d_pct") is not None
                        else ""
                    )
                    hist_sfx = f", swings_7d=[{', '.join(str(x) for x in m7d['recent_swings_7d'])}]{amp_s}"
                p7_parts.append(
                    f"swing_wave(low=${m7d.get('swing_low_px')}, high=${m7d.get('swing_high_px')}, "
                    f"rally_low={safe_float(m7d.get('rally_from_swing_low_pct'), 0.0):+.2f}%, "
                    f"pullback_high={safe_float(m7d.get('pullback_from_swing_high_pct'), 0.0):+.2f}%{maj_sfx}){hist_sfx}"
                )
            telemetry["7d Price Range & Swing Structure"] = ", ".join(p7_parts)
        ctx.m7d, ctx.is_sub24h_dup_7d = m7d, is_sub24h_dup_7d
        if ctx.m7d:
            ctx.record_pillar_source("structural_7d", "cachedb")


class PretradeShortSlopesKalmanStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Short-term polynomial slopes, 24trend windows, and 1D Kalman velocity."""

    cls_name = "PretradeShortSlopesKalmanStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        (
            cachedb_dir,
            symbol,
            sym_variants,
            sym_quotes,
            alt_series_24h,
            m24,
            cur_px_est,
            last_alt_pt,
            is_eq_sym,
            rth_fresh_eq,
            eq_px_diverged_05pct,
            eq_chart_refreshed,
            _rj,
        ) = (
            ctx.cachedb_dir,
            ctx.symbol,
            ctx.sym_variants,
            ctx.sym_quotes,
            ctx.alt_series_24h,
            ctx.m24,
            ctx.cur_px_est,
            ctx.last_alt_pt,
            ctx.is_eq_sym,
            ctx.rth_fresh_eq,
            ctx.eq_px_diverged_05pct,
            ctx.eq_chart_refreshed,
            ctx.rj,
        )
        t_entry: Dict[str, Any] = {}
        has_st_tinfo = False
        st_items_map: Dict[str, Any] = {}
        k_items_map: Dict[str, Any] = {}
        st_ts_t: Optional[float] = None
        try:
            ensure_cacheprice_classes_bound()
            if isinstance(st_data := _rj(os.path.join(cachedb_dir, _mc_mod.CacheShortTrendManager.DEFAULT_FILENAME)), dict):
                st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
                if isinstance(st_items, dict):
                    st_items_map = st_items
                    for cand_st in sym_variants:
                        if isinstance(tinfo := casefold_get(st_items, cand_st), dict):
                            has_st_tinfo = True
                            apply_short_trend_item(
                                t_entry,
                                tinfo,
                                allow_price_update=(compatible_quote_suffixes(cand_st) == sym_quotes),
                            )
                            break
            for cand in sym_variants:
                if isinstance(l24 := _rj(os.path.join(cachedb_dir, _mc_mod.CacheShortTrendManager.live_window_relpath_for_symbol(cand))), dict) and (
                    l24.get("windows") or any(wk in l24 for wk in ("90", "222", "900", "9000"))
                ):
                    had_t_px = t_entry.get("price") is not None
                    apply_live_24trend_windows(t_entry, l24)
                    if not had_t_px and compatible_quote_suffixes(cand) != sym_quotes:
                        t_entry.pop("price", None)
                    if abs(safe_float(t_entry.get("slope_90s"), 0.0)) < 5e-8:
                        for cand_sib in sym_variants:
                            if cand_sib == cand:
                                continue
                            if isinstance(l24_sib := _rj(os.path.join(cachedb_dir, _mc_mod.CacheShortTrendManager.live_window_relpath_for_symbol(cand_sib))), dict) and (
                                l24_sib.get("windows") or any(wk in l24_sib for wk in ("90", "222", "900", "9000"))
                            ):
                                sib_t: Dict[str, Any] = {}
                                apply_live_24trend_windows(sib_t, l24_sib)
                                sib_px = safe_pos_float(sib_t.get("price"), default=0.0) or 0.0
                                if cur_px_est <= 0.0 or sib_px <= 0.0 or not _prices_diverged(cur_px_est, sib_px):
                                    for k90 in (
                                        "slope_90s", "growth_90s", "trend_90s",
                                        "gradient_recent", "gradient_recent_fast",
                                        *SHORT_KINEMATIC_METRIC_KEYS,
                                    ):
                                        if k90 == "trend_90s":
                                            if t_entry.get(k90) in (None, 0) and sib_t.get(k90) not in (None, 0):
                                                t_entry[k90] = sib_t[k90]
                                        elif abs(safe_float(t_entry.get(k90), 0.0)) < 5e-8 and abs(safe_float(sib_t.get(k90), 0.0)) >= 5e-8:
                                            t_entry[k90] = sib_t[k90]
                                    if abs(safe_float(t_entry.get("slope_90s"), 0.0)) >= 5e-8:
                                        break
                    break
            k_data = _rj(os.path.join(cachedb_dir, CACHE_KALMAN_TREND_FILENAME))
            if not isinstance(k_data, dict) or not k_data:
                k_data = _rj(os.path.join(cachedb_dir, "cache_kalman_price.json"))
            if isinstance(k_data, dict):
                k_items = k_data.get("items") if isinstance(k_data.get("items"), dict) else k_data
                if isinstance(k_items, dict):
                    k_items_map = k_items
                if isinstance(k_sym := first_variant_dict(k_items, sym_variants), dict):
                    if not is_unseeded_kalman_state(k_sym) or t_entry.get("kalman_vel") is None:
                        for k_kal in ("kalman_vel", "kalman_vel_std", "kalman_trend"):
                            t_entry[k_kal] = k_sym.get(k_kal)
                    if t_entry.get("vol_1h_pct") is None and (k_vol := safe_pos_float(k_sym.get("vol_1h_pct"), default=None)) is not None:
                        t_entry["vol_1h_pct"] = k_vol
            st_ts_t = safe_pos_float(t_entry.pop("_short_trend_ts", None), default=None)
            if st_ts_t is None and has_st_tinfo and isinstance(st_data, dict):
                st_ts_t = extract_timestamp_s(st_data)
            stale_short_ts = bool(
                st_ts_t is not None
                and last_alt_pt is not None
                and (safe_float(last_alt_pt[0], 0.0) / 1000.0 - st_ts_t) > 600.0
            )
            s15_val, s25_val = t_entry.get("slope_15m"), t_entry.get("slope_2_5h")
            eq_ohlc_flat_short = bool(
                is_eq_sym
                and t_entry.get("slope_3_7m") is None
                and alt_series_24h
                and len(alt_series_24h) >= 5
                and (
                    sum(1 for r in alt_series_24h if isinstance(r, (list, tuple)) and len(r) >= 3) >= 4
                    or os.path.exists(str(resolve_cachedb_filepath(os.path.join(cachedb_dir, cache_ohlc_24h_filename(symbol)))))
                )
            )
            if (eq_px_diverged_05pct or eq_chart_refreshed or stale_short_ts or eq_ohlc_flat_short) and alt_series_24h:
                for k_ov in (
                    "slope_90s", "growth_90s", "trend_90s",
                    "slope_3_7m", "growth_3_7m", "trend_3_7m",
                    "slope_15m", "growth_15m", "trend_15m",
                    "slope_2_5h", "growth_2_5h", "trend_2_5h",
                    "gradient_recent_fast", "gradient_recent",
                    *ALL_KINEMATIC_METRIC_KEYS,
                    "vol_1h_pct",
                ):
                    t_entry.pop(k_ov, None)
                t_entry.update(compute_short_slopes_from_series(alt_series_24h))
                if last_alt_pt is not None:
                    st_ts_t = normalize_epoch_sec(last_alt_pt[0], default=st_ts_t)
                if (eq_px_diverged_05pct or eq_chart_refreshed or eq_ohlc_flat_short) and isinstance(k_recomp := compute_kalman_from_series(alt_series_24h), dict):
                    t_entry.update(k_recomp)
            elif alt_series_24h and (
                t_entry.get("slope_3_7m") is None
                or all(t_entry.get(k) is None for k in ("slope_3_7m", "slope_15m", "slope_2_5h"))
                or (s15_val is not None and s25_val is not None and s15_val != 0.0 and s15_val == s25_val)
            ):
                for sk, sv in compute_short_slopes_from_series(alt_series_24h).items():
                    if t_entry.get(sk) is None or (s15_val == s25_val and sk in ("slope_3_7m", "slope_15m", "slope_2_5h")):
                        t_entry[sk] = sv
                if st_ts_t is None and last_alt_pt is not None:
                    st_ts_t = normalize_epoch_sec(last_alt_pt[0], default=None)
            ctx.session_policy.prune_stale_short_metrics(
                t_entry,
                m24,
                alt_series_24h,
                _time.time(),
                post_rth_series_replaced=eq_px_diverged_05pct,
                rth_fresh_override=rth_fresh_eq,
                include_kinematics=True,
                has_series_tail=(last_alt_pt is not None),
            )
            resolve_kalman_vel_z(t_entry, populate=True)
            if alt_series_24h and len(alt_series_24h) >= 3:
                allow_90_kin = not is_eq_sym or eq_px_diverged_05pct or any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct"))
                allow_5m_kin = not is_eq_sym or eq_px_diverged_05pct or (rth_fresh_eq and any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct")))
                for kk, kv_kin in compute_series_kinematics(alt_series_24h).items():
                    if kk in ("vel_90s_pct_min", "accel_90s_pct_min2") and not allow_90_kin:
                        continue
                    if kk in ("vel_5m_pct_min", "accel_5m_pct_min2") and not allow_5m_kin:
                        continue
                    if t_entry.get(kk) is None:
                        t_entry[kk] = kv_kin
        except Exception:
            pass
        if (not cur_px_est or cur_px_est <= 0.0) and (t_px_est := safe_pos_float(t_entry.get("price"), default=None)) is not None:
            cur_px_est = t_px_est
        ctx.t_entry, ctx.has_st_tinfo, ctx.st_items_map, ctx.k_items_map, ctx.cur_px_est = (
            t_entry,
            has_st_tinfo,
            st_items_map,
            k_items_map,
            cur_px_est,
        )
        if ctx.t_entry:
            has_emitted_slopes = bool(
                has_st_tinfo or any(t_entry.get(k) is not None for k in ("slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h"))
            )
            ctx.record_pillar_source(
                "short_slopes",
                "cachedb",
                st_ts_t if has_emitted_slopes else None,
                include_in_snapshot_ts=False,
            )


class PretradeOrderbookCvdAndLiqStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Orderbook spread/depth, CVD flow horizons, Liquidation Clusters, and Sibling Arbitrage."""

    cls_name = "PretradeOrderbookCvdAndLiqStage"

    @staticmethod
    def _format_derivatives_and_whale_lines(
        telemetry: Dict[str, Any],
        cur_m_info: Dict[str, Any],
        is_eq_sym: bool,
        is_tok_xstock_pt: bool,
        symbol: str = "",
    ) -> None:
        eq_no_oi_pt = is_eq_sym and (safe_pos_float(cur_m_info.get("oi_usd"), default=0.0) or 0.0) <= 0.0
        if (
            (wl_p := clean_scale(cur_m_info.get("whale_long_pct"), default=None, allow_percent=True)) is not None
            and 0.0 < wl_p < 1.0
            and (
                "Whale Positioning" not in telemetry
                or str(telemetry.get("Whale Positioning") or "").startswith(("0.0% ", "100.0% "))
            )
        ):
            wl_r = safe_pos_float(cur_m_info.get("whale_long_ratio"), default=None) or (wl_p / (1.0 - wl_p))
            telemetry["Whale Positioning"] = f"{wl_p * 100.0:.1f}% Top Traders Long (L/S ratio: {wl_r:.2f})"
        if (
            (tr_m := safe_pos_float(cur_m_info.get("taker_buy_sell_ratio"), default=None)) is not None
            and (
                "Whale Taker Flow" not in telemetry
                or str(telemetry.get("Whale Taker Flow") or "").endswith(" 0.00")
                or eq_no_oi_pt
            )
        ):
            telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {tr_m:.2f}"
        if is_eq_sym and abs(safe_float(cur_m_info.get("funding_rate"), default=0.0) or 0.0) < 1e-12:
            telemetry.pop("Perpetual Funding Rate", None)
        elif (fr_v := safe_float(cur_m_info.get("funding_rate"), default=None)) is not None:
            f_ann = safe_float(cur_m_info.get("funding_annualized_pct"), default=None)
            fr_period = _funding_period_label(fr_v, f_ann, symbol=symbol)
            fr_base = telemetry.get("Perpetual Funding Rate") or f"{fr_v * 100:+.4f}% ({fr_period})"
            b_bps = safe_float(cur_m_info.get("perp_spot_basis_bps"), default=None)
            b_d15 = safe_float(cur_m_info.get("basis_delta_15m_bps"), default=None)
            f_d1h = safe_float(cur_m_info.get("funding_delta_1h_bps"), default=None)
            fr_extras = [
                x
                for x in (
                    f"funding_ann={f_ann:+.1f}%"
                    if f_ann is not None and not (is_eq_sym and abs(f_ann) < 1e-9 and abs(fr_v) < 1e-12)
                    else "",
                    f"fund_d1h={f_d1h:+.2f}bps"
                    if f_d1h is not None and abs(f_d1h) >= 0.05 and not is_eq_sym
                    else "",
                    f"perp_basis={b_bps:+.1f}bps"
                    if b_bps is not None and not (is_eq_sym and abs(b_bps) < 1e-9)
                    else "",
                    f"basis_d15m={b_d15:+.1f}bps"
                    if b_d15 is not None and abs(b_d15) >= 0.1 and not is_eq_sym
                    else "",
                )
                if x and x.split("=", 1)[0] not in fr_base
            ]
            telemetry["Perpetual Funding Rate"] = f"{fr_base} ({', '.join(fr_extras)})" if fr_extras else fr_base
        if is_tok_xstock_pt:
            for _tok_pop_k in ("Whale Positioning", "Whale Taker Flow", "Perpetual Funding Rate", "Derivatives Open Interest"):
                telemetry.pop(_tok_pop_k, None)
            telemetry["Perpetual Derivatives"] = "N/A (spot tokenized equity)"
        if (oi_v := safe_pos_float(cur_m_info.get("oi_usd"), default=0.0) or 0.0) > 0.0 and not is_tok_xstock_pt:
            l1_oi = safe_pos_float(cur_m_info.get("native_l1_oi_usd"), default=0.0) or 0.0
            oi_base = f"${oi_v:,.0f} (l1_oi=${l1_oi / 1e9:.2f}B)" if (l1_oi > 0.0 and abs(l1_oi - oi_v) > 1.0) else f"${oi_v:,.0f}"
            eq_oi_tag = " (Cboe options chain OI)" if is_eq_sym else ""
            oi_chg_parts: List[str] = []
            if (oi_chg := safe_float(cur_m_info.get("oi_1h_chg_pct"), default=None)) is not None:
                oi_chg_parts.append(f"1h_chg={oi_chg:+.2f}%")
            if (oi_15m := safe_float(cur_m_info.get("oi_15m_chg_pct"), default=None)) is not None:
                oi_chg_parts.append(f"15m_chg={oi_15m:+.2f}%")
            if oi_chg_parts:
                telemetry["Derivatives Open Interest"] = f"{oi_base}{eq_oi_tag} ({', '.join(oi_chg_parts)})"
            else:
                telemetry["Derivatives Open Interest"] = f"{oi_base}{eq_oi_tag}"

    @staticmethod
    def _reconcile_sibling_arbitrage(
        ctx: PretradeEvaluationContext,
        p_est: Dict[str, Any],
        liq_c: Dict[str, Any],
    ) -> None:
        cachedb_dir, symbol, telemetry, cur_m_info, cur_px_est = (
            ctx.cachedb_dir, ctx.symbol, ctx.telemetry, ctx.cur_m_info, ctx.cur_px_est
        )
        _rj, allow_network = ctx.rj, bool(ctx.allow_network)
        cur_bd = safe_pos_float(cur_m_info.get("bid_depth_usd"), default=0.0) or 0.0
        for cand in ctx.quote_variants:
            cand_u = str(cand or "").strip().upper()
            if not cand_u or clean_symbol_key(cand_u) == clean_symbol_key(symbol):
                continue
            sib_ob_d = _rj(os.path.join(cachedb_dir, f"{CACHE_ORDERBOOK_DEPTH_PREFIX}{cand_u}.json"))
            if not isinstance(sib_ob_d, dict):
                sib_ob_d = _rj(os.path.join(cachedb_dir, "telemetry", f"{CACHE_ORDERBOOK_DEPTH_PREFIX}{cand_u}.json"))
            if not isinstance(sib_ob_d, dict):
                continue
            sib_bd = safe_pos_float(sib_ob_d.get("bid_depth_usd"), default=0.0) or 0.0
            if not (sib_bd > 0.0 and abs(sib_bd - cur_bd) > 1.0):
                continue
            if sib_m := load_symbol_microstructure(cachedb_dir, cand_u, _rj, ob=sib_ob_d):
                for _k_sib_ts in ("_ob_raw_ts", "_deriv_ts", "_whale_ts"):
                    sib_m.pop(_k_sib_ts, None)
                sib_ob_ts = safe_pos_float(sib_m.pop("_ob_ts", None), default=0.0) or normalize_epoch_sec(
                    sib_ob_d.get("ts"), default=0.0
                ) or 0.0
                sib_ob_mid = safe_pos_float(sib_m.get("ob_mid_price"), default=None)
                sib_s24, _ = load_symbol_price_series(cachedb_dir, cand_u, current_price_items=ctx.cur_p_items)
                sib_last_pt = parse_ts_px_row(sib_s24[-1]) if sib_s24 else None
                sib_live_px = extract_current_price_item(casefold_get(ctx.cur_p_items, cand_u))
                if not sib_live_px or sib_live_px <= 0.0:
                    sib_live_px = resolve_live_24trend_current_price(cachedb_dir, cand_u, _rj)
                if (not sib_live_px or sib_live_px <= 0.0) and cur_px_est > 0.0 and sib_ob_ts > 0.0 and (_time.time() - sib_ob_ts) > 120.0:
                    sib_live_px = cur_px_est
                sib_ob_fresh = is_ob_live_and_newer(sib_ob_ts, _time.time(), sib_last_pt)
                if should_rescale_usd_sibling_walls(cand_u, sib_live_px, sib_ob_mid, cur_px_est, ob_ts=sib_ob_ts, last_pt=sib_last_pt, now_ts=_time.time()):
                    rescale_orderbook_walls_to_price(sib_m, sib_live_px, sib_ob_mid)
                sib_px = (
                    (safe_pos_float(sib_m.get("ob_mid_price"), default=None) if sib_ob_fresh else None)
                    or sib_live_px
                    or safe_pos_float(sib_m.get("ob_mid_price"), default=None)
                    or cur_px_est
                )
                sib_p_est = {**p_est, "price": sib_px} if sib_px and sib_px > 0 else dict(p_est)
                if isinstance(sib_tinfo := casefold_get(ctx.st_items_map, cand_u), dict):
                    apply_short_trend_item(sib_p_est, sib_tinfo)
                ensure_cacheprice_classes_bound()
                if isinstance(l24_sib_d := _rj(os.path.join(cachedb_dir, _mc_mod.CacheShortTrendManager.live_window_relpath_for_symbol(cand_u))), dict) and (
                    l24_sib_d.get("windows") or any(wk in l24_sib_d for wk in ("90", "222", "900", "9000"))
                ):
                    apply_live_24trend_windows(sib_p_est, l24_sib_d)
                if isinstance(sib_k := casefold_get(ctx.k_items_map, cand_u), dict):
                    if not is_unseeded_kalman_state(sib_k) or sib_p_est.get("kalman_vel") is None:
                        for k_kal in ("kalman_vel", "kalman_vel_std", "kalman_trend"):
                            sib_p_est[k_kal] = sib_k.get(k_kal)
                    if (sib_k_vol := safe_pos_float(sib_k.get("vol_1h_pct"), default=None)) is not None:
                        sib_p_est["vol_1h_pct"] = sib_k_vol
                if sib_s24 and (sib_m24 := compute_24h_price_metrics(sib_s24)):
                    if not ctx.k_items_map and not isinstance(casefold_get(ctx.st_items_map, cand_u), dict):
                        for k_kal in ("kalman_vel", "kalman_vel_std", "kalman_trend"):
                            sib_p_est.pop(k_kal, None)
                    _upgrade_pretrade_tpo_vp_from_ohlc(
                        cachedb_dir, cand_u, sib_m24, sib_px or 0.0, sib_last_pt, sib_ob_ts, allow_network=allow_network
                    )
                    apply_24h_series_enrichment(cand_u, sib_p_est, sib_m24, sib_s24, None)
                    if sib_px and sib_px > 0:
                        sib_p_est["price"] = sib_px
                        recalculate_live_price_geometry(sib_p_est, sib_px)
                enrich_microstructure_cvd(sib_m, sib_p_est)
                sib_m.update(compute_liquidation_clusters(sib_p_est, sib_m, symbol=cand_u))
                micro_pair, pt_pair = {symbol: cur_m_info, cand_u: sib_m}, {symbol: p_est, cand_u: sib_p_est}
                reconcile_sibling_flow_and_liq_regimes(micro_pair, pt_pair)
                compute_cross_venue_arb_metrics(
                    symbol,
                    cur_m_info,
                    p_est,
                    micro_pair,
                    pt_pair,
                    cache_dir=cachedb_dir,
                    now_ts=_time.time(),
                )
                liq_c["liq_magnet_bias"] = cur_m_info.get("liq_magnet_bias")
                if (arb_bps := safe_float(cur_m_info.get("cross_venue_arb_bps"), default=None)) is not None and cur_m_info.get("cross_venue_peer"):
                    net_b = safe_float(cur_m_info.get("net_buy_edge_vs_peer_bps"), default=None)
                    best_b = cur_m_info.get("best_buy_venue")
                    fee_basis = str(cur_m_info.get("arb_fee_basis") or "maker").strip().lower()
                    arb_s = f"arb_vs_{cur_m_info['cross_venue_peer']}={arb_bps:+.1f}bps"
                    if net_b is not None and best_b:
                        arb_s += f", net_buy_edge={net_b:+.1f}bps[{fee_basis}] (best_buy={best_b})"
                    telemetry["Cross-Venue Relative Value (Arbitrage)"] = arb_s
            break

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        symbol, telemetry, cur_m_info, m24, m7d, alt_series_24h, cur_px_est = (
            ctx.symbol, ctx.telemetry, ctx.cur_m_info, ctx.m24, ctx.m7d, ctx.alt_series_24h, ctx.cur_px_est
        )
        if not cur_m_info:
            return
        if cur_m_info.get("spread_bps") is not None:
            telemetry["Orderbook Bid-Ask Spread"] = f"{cur_m_info['spread_bps']} bps"
        d1_b = safe_pos_float(cur_m_info.get("depth_1pct_bid_usd"), default=0.0) or 0.0
        d1_a = safe_pos_float(cur_m_info.get("depth_1pct_ask_usd"), default=0.0) or 0.0
        if d1_b > 0.0 or d1_a > 0.0:
            bw_pt_tag = format_book_window_cap_tag(cur_m_info.get("book_span_pct"))
            bw_pt_s = f", {bw_pt_tag}" if bw_pt_tag else ""
            slip_s = f", est_slip_2k={cur_m_info['est_slip_2k_bps']} bps" if cur_m_info.get("est_slip_2k_bps") is not None else ""
            telemetry["Orderbook Depth Within 1%"] = f"bid=${d1_b:,.0f} / ask=${d1_a:,.0f}{bw_pt_s}{slip_s}"
        p_est: Dict[str, Any] = {k: v for k, v in ctx.t_entry.items() if v is not None and k != "price"}
        if cur_px_est and cur_px_est > 0:
            p_est["price"] = cur_px_est
        if m7d and not ctx.is_sub24h_dup_7d:
            p_est.update(m7d)
        if m24:
            apply_24h_series_enrichment(symbol, p_est, m24, alt_series_24h, None)
            ctx.session_policy.prune_stale_short_metrics(
                p_est,
                m24,
                alt_series_24h,
                _time.time(),
                post_rth_series_replaced=ctx.eq_px_diverged_05pct,
                rth_fresh_override=(ctx.rth_fresh_eq if ctx.last_alt_pt is not None else True),
            )
        if cur_px_est and cur_px_est > 0:
            p_est["price"] = cur_px_est
            recalculate_live_price_geometry(p_est, cur_px_est)
        enrich_microstructure_cvd(cur_m_info, p_est)
        is_tok_xstock_pt = ctx.session_policy.is_tokenized_equity
        if (
            not ctx.is_eq_sym
            and not is_tok_xstock_pt
            and (cur_m_info.get("basis_delta_15m_bps") is None or cur_m_info.get("funding_delta_1h_bps") is None)
            and (cur_m_info.get("perp_spot_basis_bps") is not None or cur_m_info.get("funding_rate") is not None)
        ):
            try:
                mhist_pt = load_microstructure_cycle_history(ctx.cachedb_dir, read_json_fn=ctx.rj)
                if mhist_pt:
                    MicrostructureCycleDeltaEvaluator().evaluate(
                        {symbol: cur_m_info},
                        history=mhist_pt,
                        now_ts=safe_pos_float(getattr(ctx, "now_ts", None), default=None) or _time.time(),
                    )
            except Exception:
                pass
        self._format_derivatives_and_whale_lines(telemetry, cur_m_info, ctx.is_eq_sym, is_tok_xstock_pt, symbol=symbol)
        no_perp_liq = not ctx.session_policy.has_liquidation_clusters
        if no_perp_liq:
            liq_c: Dict[str, Any] = {}
            purge_liquidation_cluster_keys(cur_m_info)
        else:
            liq_c = compute_liquidation_clusters(p_est, cur_m_info, symbol=symbol)
            cur_m_info.update(liq_c)
        if ctx.is_usd_sym:
            self._reconcile_sibling_arbitrage(ctx, p_est, liq_c)
        sp_vol_parts = format_spot_volume_rvol_parts(cur_m_info, lambda v: f"${v:,.0f}")
        agg_sp_part = format_agg_spot_cvd_part(cur_m_info, lambda v: f"${v:+,.0f}", lambda v: f"${v:,.0f}")
        if cur_m_info.get("cvd_15m_usd") is not None:
            cvd_u = safe_float(cur_m_info.get("cvd_15m_usd"), 0.0)
            cvd_line = f"${cvd_u:+,.0f}"
            if mh_parts := format_flow_horizon_parts(
                cur_m_info, lambda v: f"${v:+,.0f}", pos_ratio_only=True
            ):
                cvd_line += f", flow_horizons({', '.join(mh_parts)})"
            if (sp_cvd := safe_float(cur_m_info.get("spot_cvd_15m_usd"), default=None)) is not None:
                cvd_line += f", spot_cvd_15m=${sp_cvd:+,.0f}{format_spot_cvd_source_tag(cur_m_info)}"
            if agg_sp_part:
                cvd_line += f", {agg_sp_part}"
            if sp_vol_parts:
                cvd_line += f", {', '.join(sp_vol_parts)}"
            telemetry["15m Cumulative Volume Delta (CVD)"] = cvd_line
        elif (
            (sp_only_cvd := safe_float(cur_m_info.get("spot_cvd_15m_usd"), default=None)) is not None
            and cur_m_info.get("spot_cvd_source") == "real_spot"
        ):
            spot_tail = [p for p in (agg_sp_part, *sp_vol_parts) if p]
            tail_sfx = f", {', '.join(spot_tail)}" if spot_tail else ""
            telemetry["15m Cumulative Volume Delta (CVD)"] = f"spot_cvd_15m=${sp_only_cvd:+,.0f}[real_spot]{tail_sfx}"
        elif agg_sp_part or sp_vol_parts:
            telemetry["15m Cumulative Volume Delta (CVD)"] = ", ".join(p for p in (agg_sp_part, *sp_vol_parts) if p)
        if not no_perp_liq and liq_c.get("long_liq_px") is not None and liq_c.get("short_liq_px") is not None:
            deep_parts = [
                f"deep_{s_lbl}=${liq_c[px_k]} (${safe_float(liq_c.get(usd_k), 0.0):,.0f}, {safe_float(liq_c.get(dist_k), 0.0):+.2f}%)"
                for s_lbl, px_k, usd_k, dist_k in (
                    ("LONG", "long_liq_deep_px", "long_liq_deep_usd", "long_liq_deep_dist_pct"),
                    ("SHORT", "short_liq_deep_px", "short_liq_deep_usd", "short_liq_deep_dist_pct"),
                )
                if liq_c.get(px_k) is not None
            ]
            deep_sfx = f", {', '.join(deep_parts)}" if deep_parts else ""
            liq_src_lbl = str(liq_c.get("liq_source") or "model_est").strip()
            telemetry["Liquidation Clusters (LONG vs SHORT)"] = (
                f"LONG liq=${liq_c['long_liq_px']} (${safe_float(liq_c.get('long_liq_usd'), 0.0):,.0f}, {safe_float(liq_c.get('long_liq_dist_pct'), 0.0):+.2f}%), "
                f"SHORT liq=${liq_c['short_liq_px']} (${safe_float(liq_c.get('short_liq_usd'), 0.0):,.0f}, {safe_float(liq_c.get('short_liq_dist_pct'), 0.0):+.2f}%){deep_sfx}, "
                f"src={liq_src_lbl}"
            )
        if not no_perp_liq and (
            (rl_l := safe_float(cur_m_info.get("realized_liq_15m_long_usd"), default=None)) is not None
            and (rl_s := safe_float(cur_m_info.get("realized_liq_15m_short_usd"), default=None)) is not None
            and (rl_l > 0.0 or rl_s > 0.0)
        ):
            rl_n = safe_int(cur_m_info.get("realized_liq_15m_events"), default=0) or 0
            rl_src_tag = format_realized_liq_sources_tag(cur_m_info, multi_only=False)
            telemetry["Realized Liquidations (15m)"] = (
                f"long=${rl_l:,.0f}, short=${rl_s:,.0f}"
                + (f", events={rl_n}" if rl_n > 0 else "")
                + (f", src={rl_src_tag}" if rl_src_tag else "")
            )


class PretradeDynamicSlopesFormatStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Format multi-window polynomial slopes (`90s`, `3.7m`, `15m`, `2.5h`)."""

    cls_name = "PretradeDynamicSlopesFormatStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        t_entry = ctx.t_entry
        if not (ctx.has_st_tinfo or any(t_entry.get(k) is not None for k in ("slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h"))):
            return
        ctx.telemetry["Dynamic Multi-Window Slopes"] = format_multi_window_slopes_core(t_entry, fallback_full=True)
        if kin_seg := format_kinematic_parts(t_entry, compact=False):
            ctx.telemetry["Kinematic Velocity & Acceleration"] = " | ".join(kin_seg)


class QuantitativeSeriesAndMicrostructureStage(PretradePillarCollectorBase):
    """Composite pillar collector (Level 3) chaining the quantitative price-series and microstructure stages."""

    cls_name = "QuantitativeSeriesAndMicrostructureStage"

    _SUB_STAGES: Tuple[PretradePillarCollectorBase, ...] = (
        PretradePriceSeries24hStage(),
        PretradeMicrostructureReconcileStage(),
        Pretrade24hAndBtcFormatStage(),
        PretradeStructural7dStage(),
        PretradeShortSlopesKalmanStage(),
        PretradeOrderbookCvdAndLiqStage(),
        PretradeDynamicSlopesFormatStage(),
    )

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        try:
            for sub_stage in self._SUB_STAGES:
                sub_stage.evaluate(ctx)
        except Exception:
            pass


class RecentFillsExecutionStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Recent BUY/SELL fills and round-trip excursion from order/trade caches."""

    cls_name = "RecentFillsExecutionStage"

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        before_cnt = len(ctx.telemetry)
        try:
            if fills := load_symbol_best_recent_fills(ctx.cachedb_dir, ctx.symbol, ctx.quote_variants, ctx.rj):
                lb, ls = fills.get("last_buy") or {}, fills.get("last_sell") or {}
                lb_px = safe_pos_float(lb.get("price"), default=None)
                open_lots_f = fills.get("open_lots") if isinstance(fills.get("open_lots"), list) else []
                is_lb_harvested = is_last_buy_lot_harvested(
                    lb_px,
                    open_lots_f,
                    last_buy_ts_ms=safe_pos_float(lb.get("ts_ms"), default=0.0) or 0.0,
                    last_sell_ts_ms=safe_pos_float(ls.get("ts_ms"), default=0.0) or 0.0,
                )
                lb_meta = [f"by={lb.get('actor_tag') or lb.get('actor')}"] if (lb.get("actor_tag") or lb.get("actor")) else []
                if is_lb_harvested:
                    lb_meta.append("lot_harvested")
                lb_by = f" ({', '.join(lb_meta)})" if lb_meta else ""
                ls_by = f" (by={ls.get('actor_tag') or ls.get('actor')})" if (ls.get("actor_tag") or ls.get("actor")) else ""
                lb_s = (
                    f"{lb.get('qty')} @ {lb.get('price')}{lb_by}"
                    if (lb.get("qty") is not None and lb.get("price") is not None)
                    else (f"{lb.get('price')}{lb_by}" if lb.get("price") is not None else "none")
                )
                ls_s = (
                    f"{ls.get('qty')} @ {ls.get('price')}{ls_by}"
                    if (ls.get("qty") is not None and ls.get("price") is not None)
                    else (f"{ls.get('price')}{ls_by}" if ls.get("price") is not None else "none")
                )
                sub_parts = []
                now_ms_f = _time.time() * 1000.0
                for side_lbl, side_d in (("Last BUY Age", lb), ("Last SELL Age", ls)):
                    if isinstance(side_d, dict) and (s_ts := safe_pos_float(side_d.get("ts_ms"), default=None)) is not None:
                        diff_ms = now_ms_f - s_ts
                        if 0.0 <= diff_ms <= 365.0 * 86400.0 * 1000.0:
                            sub_parts.append(f"{side_lbl}={round(diff_ms / 3_600_000.0, 1)}h")
                sub_parts.extend(
                    format_recent_fill_subcycle_parts(
                        fills,
                        last_buy_px=lb_px,
                        open_lots=open_lots_f,
                        round_anchors=False,
                        include_open_lots=bool(is_lb_harvested or len(open_lots_f) > 1),
                        include_markout=False,
                    )
                )
                sub_sfx = f", {', '.join(sub_parts)}" if sub_parts else ""
                ctx.telemetry["Recent Venue Fills"] = (
                    f"Last Action={fills.get('last_action_side')}, "
                    f"Last BUY={lb_s}, "
                    f"Last SELL={ls_s}{sub_sfx}"
                )
        except Exception:
            pass
        if len(ctx.telemetry) > before_cnt:
            ctx.record_pillar_source("recent_fills", "cachedb")


class MacroPillarsStage(PretradePillarCollectorBase):
    """Concrete pillar collector (Level 3): Equity/Crypto Macro Regime, Fear & Greed, Geopolitical Threat, and Macro Calendar pillars."""

    cls_name = "MacroPillarsStage"

    @staticmethod
    def _read_cached_pillar_dict(
        ctx: PretradeEvaluationContext,
        group_key: str,
        default_filename: str,
    ) -> Optional[Dict[str, Any]]:
        data = _read_first_cached_dict(ctx.cachedb_dir, group_key, default_filename)[0]
        return data if isinstance(data, dict) else None

    def evaluate(self, ctx: PretradeEvaluationContext) -> None:
        before_cnt = len(ctx.telemetry)
        try:
            if (mb_data := self._read_cached_pillar_dict(ctx, "macro_benchmarks_files", CACHE_MACRO_BENCHMARKS_FILENAME)) is not None:
                btc_chg_24h_live: Optional[float] = None
                if not ctx.is_eq_sym and canonical_base_asset(ctx.symbol) == "BTC":
                    btc_chg_24h_live = safe_float(ctx.m24.get("chg_24h_pct"), default=None)
                elif ctx.btc_m24:
                    btc_chg_24h_live = safe_float(ctx.btc_m24.get("chg_24h_pct"), default=None)
                else:
                    btc_chg_24h_live = resolve_cached_btc_24h_change_pct(
                        ctx.cachedb_dir, current_price_items=ctx.cur_p_items
                    )
                mb_data = enrich_macro_benchmarks_live_state(mb_data, btc_chg_24h_pct=btc_chg_24h_live) or mb_data
                eq_reg_line = format_equity_macro_regime_line(mb_data)
                if eq_reg_line:
                    ctx.telemetry["Equity & Macro Regime"] = eq_reg_line
                    ctx.record_source_timestamp("macro_benchmarks", extract_timestamp_s(mb_data))
        except Exception:
            pass
        if not ctx.is_eq_sym:
            try:
                if (m_data := self._read_cached_pillar_dict(ctx, "sentiment_eval_files", CACHE_SENTIMENT_EVAL_FILENAME)) is not None:
                    bias, risk, action = (
                        str(m_data.get(k) or "").strip() for k in ("market_bias", "risk_level", "recommended_action")
                    )
                    if bias or risk or action:
                        ctx.telemetry["Macro Sentiment Bias"] = (
                            f"{bias or 'UNKNOWN'} (Risk: {risk or 'UNKNOWN'}, Recommendation: {action or 'NONE'})"
                        )
                        ctx.record_source_timestamp("sentiment_eval", extract_timestamp_s(m_data))
            except Exception:
                pass
            try:
                if (fg_data := self._read_cached_pillar_dict(ctx, "fear_greed_files", CACHE_FEAR_GREED_FILENAME)) is not None:
                    val, label = fg_data.get("value", ""), fg_data.get("classification", "")
                    if val is not None and not isinstance(val, bool) and str(val).strip() != "":
                        ctx.telemetry["Fear & Greed Index"] = (
                            f"{val} ({str(label).strip() if label is not None and not isinstance(label, bool) else ''})"
                        )
                        ctx.record_source_timestamp(
                            "fear_greed",
                            normalize_epoch_sec(fg_data.get("ts"), default=None) or extract_timestamp_s(fg_data),
                        )
            except Exception:
                pass
            try:
                if (etf_data := self._read_cached_pillar_dict(ctx, "etf_flows_files", CACHE_ETF_FLOWS_FILENAME)) is not None:
                    if etf_line := format_pretrade_etf_flows_line(etf_data, symbol=ctx.symbol):
                        ctx.telemetry["US Spot ETF Net Flows"] = etf_line
                        ctx.record_source_timestamp("etf_flows", extract_timestamp_s(etf_data))
            except Exception:
                pass
        try:
            if (g_data := self._read_cached_pillar_dict(ctx, "geopolitical_eval_file", CACHE_GEOPOLITICAL_EVAL_FILENAME)) is not None:
                level = g_data.get("threat_level", "")
                if level is not None and not isinstance(level, bool) and str(level).strip() != "":
                    ctx.telemetry["Geopolitical Threat Level"] = str(level).strip()
                    ctx.record_source_timestamp("geopolitical_eval", extract_timestamp_s(g_data))
        except Exception:
            pass
        try:
            cal_snap = self._read_cached_pillar_dict(ctx, "macro_calendar_files", CACHE_MACRO_CALENDAR_FILENAME)
            if (cal_snap is not None and cal_snap) or bool(ctx.telemetry):
                eff_cal_now = resolve_now_ts(getattr(ctx, "now_ts", None))
                cal_eval = resolve_macro_economic_calendar(
                    now_ts=eff_cal_now,
                    calendar_snapshot=cal_snap,
                )
                if cat_line := format_pretrade_macro_catalyst_line(cal_eval, symbol=ctx.symbol):
                    ctx.telemetry["Upcoming Macro / Earnings Catalyst"] = cat_line
                    if cal_snap:
                        ctx.record_source_timestamp("macro_calendar", extract_timestamp_s(cal_snap))
        except Exception:
            pass
        if len(ctx.telemetry) > before_cnt:
            ctx.record_pillar_source("macro_pillars", "cachedb")


DEFAULT_PRETRADE_TELEMETRY_STAGES: Tuple[PretradePillarCollectorBase, ...] = (
    Pillar2AndRegimeStage(),
    QuantitativeSeriesAndMicrostructureStage(),
    RecentFillsExecutionStage(),
    MacroPillarsStage(),
)


class PretradeTelemetryOrchestrator(TelemetryCacheCollectorInterface):
    """Concrete orchestrator (Level 3) executing the polymorphic ``PretradePillarCollectorBase`` pipeline."""

    cls_name = "PretradeTelemetryOrchestrator"
    COLLECTOR_ID = "pretrade_telemetry_orchestrator"

    def __init__(
        self,
        stages: Optional[Sequence[PretradePillarCollectorBase]] = None,
        *,
        start_sync: bool = False,
    ) -> None:
        super().__init__(start_sync=start_sync)
        self.stages: Tuple[PretradePillarCollectorBase, ...] = tuple(stages or DEFAULT_PRETRADE_TELEMETRY_STAGES)

    def _build_evaluation_context(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        allow_network: Optional[bool] = False,
        now_ts: Optional[float] = None,
    ) -> PretradeEvaluationContext:
        _rj = _resolve_read_json_cached()
        if (
            isinstance(regime_context, (str, os.PathLike))
            and cachedb_dir == "cachedb"
            and (
                os.path.isdir(str(regime_context))
                or "/" in str(regime_context)
                or "\\" in str(regime_context)
                or str(regime_context).endswith("cachedb")
            )
        ):
            cachedb_dir = str(regime_context)
            regime_context = None
        if cachedb_dir == "cachedb":
            cachedb_dir = get_cachedb_dir()

        session_policy = self.resolve_session_policy(symbol)
        sym_quotes = compatible_quote_suffixes(symbol)
        is_usd_sym = "USD" in sym_quotes
        sym_u_clean = str(symbol or "").strip().upper()
        is_eq_sym = session_policy.is_equity or is_equity_symbol(symbol)
        if is_eq_sym and not session_policy.is_equity:
            session_policy = self.resolve_session_policy(ensure_equity_suffix(sym_u_clean))
        sym_variants, quote_variants = _scope_symbol_variants(symbol, is_eq_sym, is_usd_sym, sym_quotes)

        return PretradeEvaluationContext(
            symbol=symbol,
            cachedb_dir=cachedb_dir,
            regime_context=regime_context,
            dur_sec=dur_sec,
            session_policy=session_policy,
            sym_quotes=sym_quotes,
            is_usd_sym=is_usd_sym,
            is_eq_sym=is_eq_sym,
            sym_variants=sym_variants,
            quote_variants=quote_variants,
            rj=_rj,
            allow_network=allow_network,
            now_ts=resolve_now_ts(now_ts),
        )

    def read_snapshot(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        now_ts: Optional[float] = None,
        max_age_sec: Optional[float] = None,
        enforce_freshness: bool = False,
        stages: Optional[Sequence[PretradePillarCollectorBase]] = None,
    ) -> PretradeTelemetrySnapshot:
        """Read-only consumer entrypoint: assemble a ``PretradeTelemetrySnapshot`` from local ``cachedb/`` with zero network/writes."""
        eff_now = resolve_now_ts(now_ts)
        ctx = self._build_evaluation_context(
            cachedb_dir,
            symbol,
            regime_context=regime_context,
            dur_sec=dur_sec,
            allow_network=False,
            now_ts=eff_now,
        )
        for stage in (stages or self.stages):
            stage.evaluate(ctx)
        valid_ts = [t for t in ctx.source_timestamps if t > 0.0]
        snap_ts = max(valid_ts) if valid_ts else 0.0
        return PretradeTelemetrySnapshot.from_raw(
            ctx.telemetry,
            symbol=symbol,
            now_ts=eff_now,
            source="cachedb",
            ts=snap_ts,
            pillar_sources=ctx.pillar_sources,
            pillar_timestamps=ctx.pillar_timestamps,
            max_age_sec=max_age_sec,
            enforce_freshness=enforce_freshness,
        )

    def collect(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        stages: Optional[Sequence[PretradePillarCollectorBase]] = None,
        allow_network: Optional[bool] = None,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ctx = self._build_evaluation_context(
            cachedb_dir,
            symbol,
            regime_context=regime_context,
            dur_sec=dur_sec,
            allow_network=allow_network,
            now_ts=now_ts,
        )
        for stage in (stages or self.stages):
            stage.evaluate(ctx)
        return ctx.telemetry


TelemetryCollectorFactory.register(PretradeTelemetryOrchestrator.COLLECTOR_ID, PretradeTelemetryOrchestrator)


def read_pretrade_telemetry_snapshot(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    max_age_sec: Optional[float] = None,
    enforce_freshness: bool = False,
    stages: Optional[Sequence[PretradePillarCollectorBase]] = None,
) -> PretradeTelemetrySnapshot:
    """Read-only consumer function returning an explicit ``PretradeTelemetrySnapshot`` from local ``cachedb/``."""
    orchestrator = TelemetryCollectorFactory.get_collector(PretradeTelemetryOrchestrator, start_sync=False)
    return orchestrator.read_snapshot(
        cachedb_dir,
        symbol,
        regime_context=regime_context,
        dur_sec=dur_sec,
        now_ts=now_ts,
        max_age_sec=max_age_sec,
        enforce_freshness=enforce_freshness,
        stages=stages,
    )


def refresh_pretrade_telemetry_caches(
    symbol: str = "",
    cachedb_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    include_t212_portfolio: bool = True,
) -> Dict[str, bool]:
    """Producer-only function refreshing pre-trade caches (microstructure and T212 position prices; equity OHLCV series is owned by cacheManager)."""
    orchestrator = TelemetryCollectorFactory.get_collector(PretradeTelemetryOrchestrator, start_sync=True)
    if cachedb_dir == "cachedb":
        cachedb_dir = get_cachedb_dir()
    eff_now = resolve_now_ts(now_ts)
    status: Dict[str, bool] = {
        "microstructure_refreshed": False,
        "t212_positions_refreshed": False,
        "equity_chart_refreshed": False,
    }
    _rj = _resolve_read_json_cached()
    sym_raw = str(symbol or "").strip()
    is_eq_sym = False
    if sym_raw and sym_raw.upper() not in disabled_registry_symbols():
        session_policy = orchestrator.resolve_session_policy(sym_raw)
        sym_quotes = compatible_quote_suffixes(sym_raw)
        is_usd_sym = "USD" in sym_quotes
        is_eq_sym = session_policy.is_equity or is_equity_symbol(sym_raw)
        _, quote_variants = _scope_symbol_variants(sym_raw, is_eq_sym, is_usd_sym, sym_quotes)
        try:
            status["microstructure_refreshed"] = bool(
                _ensure_fresh_pretrade_microstructure(
                    cachedb_dir, sym_raw, quote_variants, is_usd_sym, is_eq_sym, _rj, allow_network=True
                )
            )
        except Exception:
            pass
        if (
            not is_eq_sym
            and is_live_network_backfill_enabled()
            and _rj is _default_read_json_cached
            and ensure_fresh_symbol_microstructure is _ORIG_ENSURE_FRESH_SYMBOL_MICROSTRUCTURE
        ):
            ident_pt = resolve_symbol_identity(sym_raw)
            sym_canon = (
                ident_pt.display_symbol
                if (ident_pt.is_tokenized_equity and ident_pt.display_symbol)
                else (clean_symbol_key(sym_raw) or sym_raw.upper())
            )
            ohlc_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, cache_ohlc_24h_filename(sym_canon))))
            if os.path.exists(ohlc_path) or os.path.exists(
                str(resolve_cachedb_filepath(os.path.join(cachedb_dir, f"{CACHE_ORDERBOOK_DEPTH_PREFIX}{sym_canon}.json")))
            ):
                try:
                    prev_mtime = os.path.getmtime(ohlc_path) if os.path.exists(ohlc_path) else 0.0
                    load_stitched_24h_ohlc_points(
                        resolve_fallback_ohlc_fn(sym_canon),
                        sym_canon,
                        eff_now - 86400.0,
                        require_positive_volume=True,
                        cache_dir=cachedb_dir,
                    )
                    new_mtime = os.path.getmtime(ohlc_path) if os.path.exists(ohlc_path) else 0.0
                    if new_mtime > prev_mtime:
                        status["microstructure_refreshed"] = True
                except Exception:
                    pass
    if include_t212_portfolio and (not sym_raw or is_eq_sym):
        try:
            vb_path, vb_d, _ = load_venue_balances_state(cachedb_dir, _rj)
            if vb_path and vb_d:
                status["t212_positions_refreshed"] = bool(
                    refresh_stale_t212_cached_positions(vb_d, vb_path, eff_now)
                )
        except Exception:
            pass
    return status


def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
    *,
    stages: Optional[Sequence[PretradePillarCollectorBase]] = None,
    allow_network: Optional[bool] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) via ``PretradeTelemetryOrchestrator``."""
    orchestrator = TelemetryCollectorFactory.get_collector(PretradeTelemetryOrchestrator)
    return orchestrator.collect(
        cachedb_dir,
        symbol,
        regime_context=regime_context,
        dur_sec=dur_sec,
        stages=stages,
        allow_network=allow_network,
        now_ts=now_ts,
    )


__all__ = [
    "DEFAULT_MAX_PRETRADE_TELEMETRY_AGE_SEC",
    "DEFAULT_PRETRADE_TELEMETRY_STAGES",
    "MacroPillarsStage",
    "Pillar2AndRegimeStage",
    "Pretrade24hAndBtcFormatStage",
    "PretradeDynamicSlopesFormatStage",
    "PretradeEvaluationContext",
    "PretradeMicrostructureReconcileStage",
    "PretradeOrderbookCvdAndLiqStage",
    "PretradePillarCollectorBase",
    "PretradePriceSeries24hStage",
    "PretradeShortSlopesKalmanStage",
    "PretradeStructural7dStage",
    "PretradeTelemetryOrchestrator",
    "PretradeTelemetrySnapshot",
    "QuantitativeSeriesAndMicrostructureStage",
    "RecentFillsExecutionStage",
    "collect_pretrade_telemetry",
    "PRETRADE_SOURCE_STALE_INTERVAL_MULT",
    "read_pretrade_telemetry_snapshot",
    "refresh_pretrade_telemetry_caches",
    "resolve_pretrade_source_intervals_sec",
]
