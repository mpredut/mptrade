"""Spot/perpetual microstructure, 24h/7d price series loader, and reference symbol discovery collector."""
from __future__ import annotations

import glob as _glob
import logging
import os
import time as _time
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from botcore import (
    extract_current_price_item,
    is_live_network_backfill_enabled,
    load_dotenv,
    load_local_module,
    safe_nonneg_float,
)
from cachepaths import (
    CACHE_DERIVATIVES_TELEMETRY_PREFIX,
    CACHE_ORDERBOOK_DEPTH_PREFIX,
    CACHE_WATCHER_ALL_PRICE_FILENAME,
    CACHE_WHALE_SNAPSHOT_PREFIX,
    cache_symbol_variants,
    canonical_base_asset,
    clean_symbol_key,
    compatible_quote_suffixes,
    has_equity_suffix,
    legacy_filenames_for,
    normalize_venue_name,
    resolve_cache_dir_path,
    split_symbol_assets,
)
Cache24PriceManager: Any = None
Cache365PriceManager: Any = None
CacheCurrentPriceManager: Any = None
CachePriceManagerBase: Any = None
CacheSparsePriceManager: Any = None
CacheShortTrendManager: Any = None
CacheLongTrendManager: Any = None
CacheOrderManager: Any = None
CacheTradeManager: Any = None
CacheAssetValueManager: Any = None
CacheAccountManagerBase: Any = None


def _ensure_cacheprice_classes_bound() -> None:
    global Cache24PriceManager, Cache365PriceManager, CacheCurrentPriceManager, CachePriceManagerBase, CacheSparsePriceManager
    global CacheShortTrendManager, CacheLongTrendManager, CacheOrderManager, CacheTradeManager, CacheAssetValueManager
    global CacheAccountManagerBase
    if Cache24PriceManager is not None and CacheShortTrendManager is not None and CacheOrderManager is not None and CacheAccountManagerBase is not None:
        return
    load_dotenv()
    from cacheaccount import (
        CacheAccountManagerBase as _CABase,
        CacheAssetValueManager as _CAV,
        CacheOrderManager as _COrd,
        CacheTradeManager as _CTrd,
    )
    from cacheprice import (
        Cache24PriceManager as _C24,
        Cache365PriceManager as _C365,
        CacheCurrentPriceManager as _CCP,
        CachePriceManagerBase as _CPBase,
        CacheSparsePriceManager as _CSparse,
    )
    try:
        from cachetrend import (
            CacheLongTrendManager as _CLT,
            CacheShortTrendManager as _CST,
        )
    except ImportError:
        _CLT = None
        _CST = None

    Cache24PriceManager = _C24
    Cache365PriceManager = _C365
    CacheCurrentPriceManager = _CCP
    CachePriceManagerBase = _CPBase
    CacheSparsePriceManager = _CSparse
    CacheShortTrendManager = _CST
    CacheLongTrendManager = _CLT
    CacheOrderManager = _COrd
    CacheTradeManager = _CTrd
    CacheAssetValueManager = _CAV
    CacheAccountManagerBase = _CABase
    if "CompositeSymbolPriceSeriesLoader" in globals():
        CompositeSymbolPriceSeriesLoader.price_24_cls = _C24
        CompositeSymbolPriceSeriesLoader.price_365_cls = _C365
        CompositeSymbolPriceSeriesLoader.price_sparse_cls = _CSparse
        CompositeSymbolPriceSeriesLoader.current_price_cls = _CCP
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
    round_spread_bps,
)
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    load_microstructure_cycle_history,
    lookup_symbol_cache_items,
    normalize_epoch_sec,
    obj_attr,
    resolve_now_ts,
    resolve_read_json_cached,
    safe_float,
    safe_pos_float,
    save_microstructure_cycle_history,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _MICRO_PRICE_ANCHOR_KEYS,
    _first_variant_dict,
    align_equity_orderbook_to_spot,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    is_ob_mid_fresher_than_series,
    rescale_orderbook_walls_to_price,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    resolve_symbol_price,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _LAST_7D_SPAN_MS_BY_SYM,
    _RTH_VP_ANCHOR_KEYS,
    _SHORT_SLOPE_OVERRIDE_KEYS,
    _STALE_SHORT_24H_KEYS,
    _apply_24h_series_enrichment,
    _load_stitched_24h_ohlc_points,
    _parse_ts_px_row,
    _prices_diverged,
    append_newer_24h_to_7d_vals,
    collapse_watcher_equity_series,
    compute_7d_range_metrics,
    compute_24h_price_metrics,
    compute_volume_profile_metrics,
    extract_archive_24h_and_7d_slices,
    normalize_yahoo_5d_chart_bars,
    pct_delta,
    prepend_older_non_diverged_points,
    recalculate_live_price_geometry,
    resolve_asset_session_policy,
    sort_ts_rows,
)
from instrument_registry import casefold_get, is_equity_symbol
from state_io import (
    read_json_dict_recovering as _default_read_json_cached,
    read_json_recovering,
    read_tail_jsonl_records,
    resolve_cachedb_filepath,
    save_json_state,
)

read_json_cached = _default_read_json_cached
_FALLBACK_PROVIDER_OHLC_FNS: Dict[str, Any] = {}
_FALLBACK_OHLC_SERIES_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_PRICE_SERIES_MEMO: Dict[Tuple[str, str, bool, Tuple[str, ...]], Tuple[Any, list, list, float]] = {}
_ARCH_FILE_MEMO: Dict[Tuple[str, int, int, int, Tuple[str, ...]], list] = {}
_EQ_YAHOO_CHART_MEMO: Dict[str, Tuple[float, Any]] = {}

logger = logging.getLogger("intelligence.sentiment.microstructure_collector")


def _resolve_read_json_cached(read_json_fn: Any = None) -> Any:
    return resolve_read_json_cached(read_json_fn, local_fn=read_json_cached)


def _filter_ohlc_closes_since(
    series_obj: Any,
    cutoff_ts: float,
    max_ts: Optional[float] = None,
) -> List[float]:
    ts_seq = getattr(series_obj, "timestamps", None) or ()
    cl_seq = getattr(series_obj, "closes", None) or ()
    if not isinstance(ts_seq, (list, tuple)) or not isinstance(cl_seq, (list, tuple)):
        return []
    cutoff_s = normalize_epoch_sec(cutoff_ts, default=0.0, allow_zero=True) or 0.0
    max_s = normalize_epoch_sec(max_ts, default=None, allow_zero=True) if max_ts is not None else None
    timed_out: List[Tuple[float, float]] = []
    for raw_ts, raw_px in zip(ts_seq, cl_seq):
        ts_s = normalize_epoch_sec(raw_ts, default=None)
        px_v = safe_pos_float(raw_px, default=None)
        if ts_s is not None and px_v is not None and ts_s >= cutoff_s and (max_s is None or ts_s <= max_s):
            timed_out.append((ts_s, px_v))
    return [px_v for _, px_v in sorted(timed_out, key=lambda x: x[0])]


def _wrap_cached_ohlc_fn(
    prov_key: str,
    raw_fn_or_factory: Any,
    ttl_sec: float = 60.0,
    *,
    is_factory: bool = False,
) -> Any:
    resolved_fn: Any = None if is_factory else raw_fn_or_factory

    def _get_raw_fn() -> Any:
        nonlocal resolved_fn
        if resolved_fn is None and is_factory and callable(raw_fn_or_factory):
            resolved_fn = raw_fn_or_factory()
        return resolved_fn

    def _cached_ohlc(s_arg: str, interval_min: int) -> Any:
        fn = _get_raw_fn()
        if not callable(fn):
            return None
        ck = (prov_key, id(fn), str(s_arg or "").strip().upper(), int(interval_min))
        now_s = _time.time()
        hit = _FALLBACK_OHLC_SERIES_CACHE.get(ck)
        if hit is not None and 0.0 <= (now_s - hit[0]) < ttl_sec:
            return hit[1]
        res = fn(s_arg, interval_min)
        if res is not None and getattr(res, "closes", None):
            if len(_FALLBACK_OHLC_SERIES_CACHE) > 128:
                _FALLBACK_OHLC_SERIES_CACHE.clear()
            _FALLBACK_OHLC_SERIES_CACHE[ck] = (now_s, res)
        return res

    return _cached_ohlc


def _resolve_fallback_ohlc_fn(sym: str, inst_by_sym: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    sym_u = str(sym or "").strip().upper()
    sym_clean = clean_symbol_key(sym_u)
    inst_map = inst_by_sym or {}
    inst_obj = (
        (inst_map.get(f"binance:{sym_clean}") if sym_clean != "HYPEUSDC" else None)
        or inst_map.get(sym)
        or inst_map.get(sym_u)
        or inst_map.get(sym_clean)
    )
    ohlc_fn = getattr(getattr(inst_obj, "_provider", None), "ohlc_series", None)
    if callable(ohlc_fn) or not is_live_network_backfill_enabled():
        return ohlc_fn if callable(ohlc_fn) else None
    try:
        prov_key = ""
        sym_parts = split_symbol_assets(sym_u) if sym_u and not is_equity_symbol(sym_u) else None
        sym_q = sym_parts[1] if sym_parts is not None else ""
        if sym_clean == "HYPEUSDC":
            prov_key = "hyperliquid"
        elif sym_q in ("USDC", "USDT", "FDUSD", "TUSD", "BUSD") or sym_u.endswith(("USDC", "USDT", "FDUSD", "TUSD", "BUSD")):
            prov_key = "binance"
        elif sym_q == "USD" or (sym_u.endswith("USD") and not is_equity_symbol(sym_u) and not sym_u.endswith(("FDUSD", "TUSD", "BUSD"))):
            prov_key = "kraken"
        if prov_key and prov_key not in _FALLBACK_PROVIDER_OHLC_FNS:
            _FALLBACK_PROVIDER_OHLC_FNS[prov_key] = _wrap_cached_ohlc_fn(
                prov_key,
                lambda pk=prov_key: getattr(
                    __import__("providers.market_api", fromlist=["api"]).api.provider_by_name(pk),
                    "ohlc_series",
                    None,
                ),
                is_factory=True,
            )
        if prov_key and callable(cached_fn := _FALLBACK_PROVIDER_OHLC_FNS.get(prov_key)):
            return cached_fn
    except Exception:
        pass
    return None


def _load_stitched_7d_ohlc_closes(
    ohlc_fn: Any,
    sym: str,
    now_ts: float,
    m24: Optional[Dict[str, Any]] = None,
    cur_px: float = 0.0,
) -> List[float]:
    if not callable(ohlc_fn):
        return []
    s60, cutoff_7d = ohlc_fn(sym, 60), now_ts - 7 * 86400.0
    s60_ts = sorted(
        ts_s
        for raw_ts in (getattr(s60, "timestamps", None) or ())
        if (ts_s := normalize_epoch_sec(raw_ts, default=None)) is not None
    )
    vals_7d = _filter_ohlc_closes_since(s60, cutoff_7d)
    if s60_ts and (s60_ts[-1] - s60_ts[0]) < 6.75 * 86400.0:
        try:
            s240 = ohlc_fn(sym, 240)
            first_s60 = s60_ts[0] - 1800.0
            older_7d = _filter_ohlc_closes_since(s240, cutoff_7d, max_ts=first_s60)
            if older_7d:
                vals_7d = older_7d + vals_7d
        except Exception:
            pass
    if len(vals_7d) >= 2:
        if isinstance(m24, dict):
            tail_start = max(0, len(vals_7d) - 24)
            l24_ex = safe_pos_float(m24.get("low_24h"), default=None)
            h24_ex = safe_pos_float(m24.get("high_24h"), default=None)
            if l24_ex is not None and not _prices_diverged(l24_ex, vals_7d[-1]):
                min_idx = min(range(tail_start, len(vals_7d)), key=lambda i: vals_7d[i])
                if l24_ex < vals_7d[min_idx]:
                    vals_7d.insert(min_idx, l24_ex)
            if h24_ex is not None and not _prices_diverged(h24_ex, vals_7d[-1]):
                max_idx = max(range(tail_start, len(vals_7d)), key=lambda i: vals_7d[i])
                if h24_ex > vals_7d[max_idx]:
                    vals_7d.insert(max_idx, h24_ex)
        if cur_px > 0 and not _prices_diverged(cur_px, vals_7d[-1]) and abs(cur_px - vals_7d[-1]) > 1e-9:
            vals_7d.append(cur_px)
    return vals_7d


def _extract_cached_items_series(data: Any, cands: list[str]) -> list:
    return (
        lookup_symbol_cache_items(
            data,
            cand_keys=cands,
            allow_base_fallback=False,
            single_item_quote_fallback=True,
            require_list=True,
            unwrap_items=True,
        )
        or []
    )


def _candidate_cache_paths(cachedb_dir: str, *basenames: str) -> list[str]:
    return [
        p
        for bn in basenames
        for full_p in (str(resolve_cachedb_filepath(os.path.join(cachedb_dir, bn))),)
        for p in (full_p, *legacy_filenames_for(full_p))
    ]


def _read_archive_series_file(
    arch_path: str,
    cands: list[str],
    cands_upper: set[str],
    cands_upper_tup: Tuple[str, ...],
    _rj: Any,
    is_default_rj: bool,
    max_b: int = 1024 * 1024,
) -> list:
    if not os.path.exists(arch_path):
        return []
    _ensure_cacheprice_classes_bound()
    try:
        if arch_path.endswith(".json"):
            aser = sort_ts_rows(_extract_cached_items_series(_rj(arch_path), cands))
            return list(aser) if len(aser) >= 2 else []
        arch_key: Optional[Tuple[str, int, int, int, Tuple[str, ...]]] = None
        if is_default_rj:
            try:
                ast = os.stat(arch_path)
                arch_key = (os.path.abspath(arch_path), ast.st_mtime_ns, ast.st_size, max_b, cands_upper_tup)
                if (hit_arch := _ARCH_FILE_MEMO.get(arch_key)) is not None:
                    return [list(r) for r in hit_arch]
            except OSError:
                arch_key = None
        parsed_rows = [
            pt
            for rec in read_tail_jsonl_records(arch_path, limit=None, max_bytes=max_b)
            if (pt := CachePriceManagerBase.extract_jsonl_series_point(rec, cands_upper)) is not None
        ]
        parsed_rows = sort_ts_rows(parsed_rows)
        res_arch = parsed_rows if len(parsed_rows) >= 2 else []
        if arch_key is not None and res_arch:
            if len(_ARCH_FILE_MEMO) > 128:
                _ARCH_FILE_MEMO.clear()
            _ARCH_FILE_MEMO[arch_key] = [list(r) for r in res_arch]
        return res_arch
    except Exception:
        return []


def load_symbol_price_series(
    cachedb_dir: str,
    symbol: str,
    current_price_items: Optional[Dict[str, Any]] = None,
    *,
    include_7d: bool = True,
    quotes: Optional[tuple[str, ...]] = None,
) -> Tuple[list, list]:
    """Load and merge high-density 24h & 7d price series via ``TelemetryCollectorFactory.get_price_series_loader()``."""
    return TelemetryCollectorFactory.get_price_series_loader().collect(
        cachedb_dir,
        symbol,
        current_price_items=current_price_items,
        include_7d=include_7d,
        quotes=quotes,
    )


def resolve_cached_btc_24h_change_pct(
    cache_dir: str,
    *,
    current_price_items: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Return BTC 24h percentage change from cached 24h BTCUSDC/BTCUSDT price series if available."""
    if not cache_dir:
        return None
    _ensure_cacheprice_classes_bound()
    btc_24h_files = dict.fromkeys(
        fn
        for s in ("BTCUSDC", "BTCUSDT")
        for base_fn in (Cache24PriceManager.disk_filename_for_symbol(s),)
        for fn in (*legacy_filenames_for(base_fn), base_fn)
    )
    if not any(
        os.path.exists(str(resolve_cachedb_filepath(os.path.join(cache_dir, fn))))
        for fn in btc_24h_files
    ):
        return None
    btc_pts, _ = load_symbol_price_series(
        cache_dir,
        "BTCUSDC",
        current_price_items=current_price_items,
        include_7d=False,
    )
    if not btc_pts:
        return None
    btc_m = compute_24h_price_metrics(btc_pts)
    chg = safe_float(btc_m.get("chg_24h_pct"), default=None) if isinstance(btc_m, dict) else None
    return chg if chg is not None and abs(chg) > 1e-6 else None


def load_symbol_microstructure(
    cache_dir: str,
    sym: str,
    read_json_fn: Any = None,
    *,
    ob: Any = None,
    deriv: Any = None,
    whale: Any = None,
) -> Optional[Dict[str, Any]]:
    """Load and normalize orderbook depth, derivatives, and whale snapshot across symbol variants."""
    read_json_fn = _resolve_read_json_cached(read_json_fn)
    sym_u = str(sym or "").strip().upper()
    is_eq_sym = is_equity_symbol(sym_u)
    sym_cands = [
        c
        for c in cache_symbol_variants(sym, include_base=True)
        if is_equity_symbol(c) == is_eq_sym
    ]

    def _first_telemetry_dict(prefix: str, *, prefer_freshest: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        best_d: Optional[Dict[str, Any]] = None
        best_cand: Optional[str] = None
        best_ts = -1.0
        for cand in sym_cands:
            if isinstance((d := read_json_fn(os.path.join(cache_dir, f"{prefix}{cand}.json"))), dict):
                if not prefer_freshest:
                    return d, cand
                ts_raw = d.get("ts") if d.get("ts") is not None else d.get("timestamp")
                ts_v = normalize_epoch_sec(safe_pos_float(ts_raw, default=0.0) or 0.0, default=0.0)
                if best_d is None or ts_v > best_ts:
                    best_d, best_cand, best_ts = d, cand, ts_v
        return best_d, best_cand

    if ob is None:
        ob, ob_cand = _first_telemetry_dict(CACHE_ORDERBOOK_DEPTH_PREFIX)
    else:
        raw_ob_sym = obj_attr(ob, "symbol")
        ob_cand = raw_ob_sym.strip() if isinstance(raw_ob_sym, str) and raw_ob_sym.strip() else None
    if deriv is None:
        deriv, _ = _first_telemetry_dict(CACHE_DERIVATIVES_TELEMETRY_PREFIX, prefer_freshest=True)
    if whale is None:
        whale, _ = _first_telemetry_dict(CACHE_WHALE_SNAPSHOT_PREFIX, prefer_freshest=True)
    if not (ob or deriv or whale):
        return None
    m_info: Dict[str, Any] = {"_source_cand": ob_cand} if ob_cand else {}
    if ob is not None:
        ob_same_quote = (not ob_cand) or (compatible_quote_suffixes(ob_cand) == compatible_quote_suffixes(sym))
        raw_ob_imb = obj_attr(ob, "imbalance_ratio", obj_attr(ob, "ob_imbalance"))
        m_info["ob_imbalance"] = clean_scale(raw_ob_imb, default=None, allow_percent=True) if raw_ob_imb is not None else None
        ob_mid = safe_pos_float(obj_attr(ob, "mid_price", obj_attr(ob, "ob_mid_price")), default=None)
        best_bid = safe_pos_float(obj_attr(ob, "best_bid"), default=None)
        best_ask = safe_pos_float(obj_attr(ob, "best_ask"), default=None)
        if (spread_bps := round_spread_bps(obj_attr(ob, "spread_bps"), best_bid=best_bid, best_ask=best_ask, mid_price=ob_mid)) is not None:
            m_info["spread_bps"] = spread_bps
        if ob_same_quote:
            if (ob_ts_v := normalize_epoch_sec(safe_pos_float(obj_attr(ob, "ts"), default=0.0) or 0.0, default=0.0)) > 0.0:
                m_info["_ob_ts"] = ob_ts_v
            for k_px, v_px in (("ob_mid_price", ob_mid), ("best_bid", best_bid), ("best_ask", best_ask)):
                if v_px is not None:
                    m_info[k_px] = v_px
        for k_src, k_dst in (
            ("bid_depth_usd", "bid_depth_usd"), ("ask_depth_usd", "ask_depth_usd"),
            ("depth_1pct_bid_usd", "depth_1pct_bid_usd"), ("depth_1pct_ask_usd", "depth_1pct_ask_usd"),
            ("largest_bid_wall_usd", "bid_wall_usd"), ("largest_ask_wall_usd", "ask_wall_usd"),
        ):
            val = safe_nonneg_float(obj_attr(ob, k_src, obj_attr(ob, k_dst) if k_src != k_dst else None), default=None)
            m_info[k_dst] = round(val, 0) if val is not None else None
        m_info["bid_wall_px"] = safe_pos_float(obj_attr(ob, "largest_bid_wall_price", obj_attr(ob, "bid_wall_px")), default=None) if ob_same_quote else None
        m_info["ask_wall_px"] = safe_pos_float(obj_attr(ob, "largest_ask_wall_price", obj_attr(ob, "ask_wall_px")), default=None) if ob_same_quote else None
    if deriv is not None:
        m_info["funding_rate"] = safe_float(obj_attr(deriv, "funding_rate"), default=None)
        m_info["pred_funding_rate"] = safe_float(obj_attr(deriv, "predicted_funding_rate", obj_attr(deriv, "pred_funding_rate")), default=None)
        m_info["oi_usd"] = obj_attr(deriv, "open_interest_usd", obj_attr(deriv, "oi_usd"))
        m_info["mark_price"] = safe_pos_float(obj_attr(deriv, "mark_price"), default=None)
        m_info["index_price"] = safe_pos_float(obj_attr(deriv, "index_price"), default=None)
        m_info["perp_spot_basis_bps"] = safe_float(obj_attr(deriv, "perp_spot_basis_bps"), default=None)
        m_info["funding_annualized_pct"] = safe_float(obj_attr(deriv, "funding_annualized_pct"), default=None)
    if whale is not None:
        for dst_k, src_k, alt_k in (
            ("whale_long_pct", "top_traders_long_pct", "whale_long_pct"),
            ("whale_long_ratio", "top_traders_long_ratio", "whale_long_ratio"),
            ("taker_buy_sell_ratio", "taker_buy_sell_ratio", None),
            ("taker_buy_vol_usd", "taker_buy_vol_usd", None),
            ("taker_sell_vol_usd", "taker_sell_vol_usd", None),
            ("oi_1h_chg_pct", "open_interest_1h_change_pct", "oi_1h_chg_pct"),
            ("divergence_regime", "divergence_regime", "div_regime"),
            ("taker_ratio_5m", "taker_ratio_5m", None),
            ("taker_ratio_1h", "taker_ratio_1h", None),
            ("taker_ratio_4h", "taker_ratio_4h", None),
            ("cvd_5m_usd", "cvd_5m_usd", None),
            ("cvd_15m_usd", "cvd_15m_usd", None),
            ("spot_cvd_15m_usd", "spot_cvd_15m_usd", None),
            ("cvd_1h_usd", "cvd_1h_usd", None),
            ("cvd_4h_usd", "cvd_4h_usd", None),
        ):
            val_w = obj_attr(whale, src_k, obj_attr(whale, alt_k) if alt_k else None)
            if dst_k == "whale_long_pct" and val_w is not None:
                val_w = clean_scale(val_w, default=None, allow_percent=True)
            if val_w is not None or dst_k not in m_info:
                m_info[dst_k] = val_w
        w_oi = safe_pos_float(obj_attr(whale, "open_interest_usd", obj_attr(whale, "oi_usd")), default=None)
        deriv_oi = safe_pos_float(m_info.get("oi_usd"), default=None)
        if not deriv_oi and w_oi is not None:
            m_info["oi_usd"] = w_oi
        elif (
            deriv_oi is not None
            and w_oi is not None
            and deriv_oi > w_oi * 1.35
            and (
                (safe_pos_float(m_info.get("taker_buy_vol_usd"), default=0.0) or 0.0) > 0.0
                or abs((safe_float(m_info.get("whale_long_pct"), default=0.5) or 0.5) - 0.5) > 1e-4
            )
        ):
            m_info["native_l1_oi_usd"] = round(deriv_oi, 2)
            m_info["oi_usd"] = w_oi
        wl_cur = safe_pos_float(m_info.get("whale_long_pct"), default=0.0) or 0.0
        if (safe_pos_float(m_info.get("oi_usd"), default=0.0) or 0.0) <= 0.0 and (
            wl_cur <= 0.0 or wl_cur >= 1.0
        ):
            tr_ref = safe_pos_float(
                m_info.get("taker_ratio_1h") or m_info.get("taker_buy_sell_ratio"),
                default=None,
            )
            if tr_ref is not None and tr_ref < 100.0:
                wl_norm = round(tr_ref / (1.0 + tr_ref), 4)
                m_info["whale_long_pct"] = wl_norm
                m_info["whale_long_ratio"] = round(tr_ref, 4)
                if (safe_pos_float(m_info.get("taker_buy_sell_ratio"), default=0.0) or 0.0) <= 0.0:
                    m_info["taker_buy_sell_ratio"] = round(tr_ref, 4)
                tb_cur = safe_nonneg_float(m_info.get("taker_buy_vol_usd"), default=0.0) or 0.0
                ts_cur = safe_nonneg_float(m_info.get("taker_sell_vol_usd"), default=0.0) or 0.0
                if (tb_cur <= 0.0 < ts_cur) or (ts_cur <= 0.0 < tb_cur):
                    tot_v = tb_cur + ts_cur
                    m_info["taker_buy_sell_ratio"] = round(tr_ref, 4)
                    m_info["taker_buy_vol_usd"] = round(tot_v * wl_norm, 2)
                    m_info["taker_sell_vol_usd"] = round(tot_v * (1.0 - wl_norm), 2)
                    if (
                        m_info.get("taker_ratio_5m") == 0.5
                        and (safe_float(m_info.get("cvd_5m_usd"), default=0.0) or 0.0) < 0.0
                    ):
                        m_info["taker_ratio_5m"] = round(0.15 / 0.85, 4)
                    elif (
                        m_info.get("taker_ratio_5m") == 2.0
                        and (safe_float(m_info.get("cvd_5m_usd"), default=0.0) or 0.0) > 0.0
                    ):
                        m_info["taker_ratio_5m"] = round(0.85 / 0.15, 4)
        if m_info.get("funding_rate") is None and (w_fr := safe_float(obj_attr(whale, "funding_rate"), default=None)) is not None:
            m_info["funding_rate"] = w_fr
    if is_equity_symbol(sym):
        for eq_zero_k in ("perp_spot_basis_bps", "funding_rate", "pred_funding_rate", "funding_annualized_pct"):
            if m_info.get(eq_zero_k) is not None:
                m_info[eq_zero_k] = 0.0
        tr5_eq = safe_pos_float(m_info.get("taker_ratio_5m"), default=None)
        if tr5_eq is not None and (tr5_eq < 0.176 or tr5_eq > 5.67):
            clv_old = tr5_eq / (1.0 + tr5_eq)
            u_bar = max(0.0, min(1.0, (clv_old - 0.05) / 0.90))
            clv_new = 0.15 + 0.70 * u_bar
            m_info["taker_ratio_5m"] = round(clv_new / (1.0 - clv_new), 4)
            cvd5_old = safe_float(m_info.get("cvd_5m_usd"), default=None)
            if cvd5_old is not None and abs(2.0 * clv_old - 1.0) > 1e-6:
                vol_5m = abs(cvd5_old) / abs(2.0 * clv_old - 1.0)
                m_info["cvd_5m_usd"] = round(vol_5m * (2.0 * clv_new - 1.0), 0)
    if m_info.get("funding_annualized_pct") is None and (fr_v := safe_float(m_info.get("funding_rate"), default=None)) is not None:
        m_info["funding_annualized_pct"] = round(fr_v * 3.0 * 365.0 * 100.0, 2)
    for usd_k in ("oi_usd", "taker_buy_vol_usd", "taker_sell_vol_usd", "cvd_5m_usd", "cvd_15m_usd", "spot_cvd_15m_usd", "cvd_1h_usd", "cvd_4h_usd"):
        if (uv := safe_float(m_info.get(usd_k), default=None)) is not None:
            m_info[usd_k] = round(uv, 2)
    enrich_microstructure_cvd(m_info)
    return m_info


def ensure_fresh_symbol_microstructure(
    cache_dir: str,
    symbol: str,
    *,
    now_ts: Optional[float] = None,
    max_age_ob_sec: Optional[float] = None,
    max_age_whale_sec: Optional[float] = None,
    include_derivatives: bool = False,
    allow_network: bool = True,
) -> None:
    """Trigger on-demand orderbook, derivatives, or whale collector fetches when cached files are missing or stale."""
    if not allow_network or not is_live_network_backfill_enabled() or not cache_dir:
        return
    eff_cache_dir = resolve_cache_dir_path(cache_dir)
    if not eff_cache_dir or not os.path.isdir(eff_cache_dir):
        return
    sym_u = str(symbol or "").strip().upper()
    if not sym_u:
        return
    eff_now = resolve_now_ts(now_ts)
    is_eq = is_equity_symbol(sym_u)
    if is_eq:
        try:
            from instrument_registry import is_symbol_market_open

            rth_open = is_symbol_market_open(sym_u, eff_now)
        except Exception:
            rth_open = True
    else:
        rth_open = True

    def _needs_fetch(prefix: str, max_age: Optional[float]) -> bool:
        fpath = str(resolve_cachedb_filepath(os.path.join(eff_cache_dir, f"{prefix}{sym_u}.json")))
        if not os.path.exists(fpath):
            return True
        if is_eq and not rth_open:
            if prefix == CACHE_ORDERBOOK_DEPTH_PREFIX:
                ob_disk = read_json_cached(fpath)
                if isinstance(ob_disk, dict) and (safe_pos_float(ob_disk.get("spread_bps"), default=0.0) or 0.0) > 200.0:
                    return True
            return False
        return max_age is not None and (eff_now - os.path.getmtime(fpath)) > max_age

    if _needs_fetch(CACHE_ORDERBOOK_DEPTH_PREFIX, max_age_ob_sec):
        try:
            from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector

            OrderbookDepthCollector(cache_dir=eff_cache_dir, start_sync=True).fetch(sym_u, allow_network=True, now=eff_now)
        except Exception:
            pass
    if include_derivatives and _needs_fetch(CACHE_DERIVATIVES_TELEMETRY_PREFIX, max_age_whale_sec):
        try:
            from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector

            DerivativesTelemetryCollector(cache_dir=eff_cache_dir, start_sync=True).fetch(sym_u, allow_network=True, now=eff_now)
        except Exception:
            pass
    if _needs_fetch(CACHE_WHALE_SNAPSHOT_PREFIX, max_age_whale_sec):
        try:
            from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector

            WhalePositioningCollector(cache_dir=eff_cache_dir, start_sync=True).fetch(sym_u, allow_network=True, now=eff_now)
        except Exception:
            pass


def discover_reference_microstructure_symbols(
    cache_dir: str,
    microstructure: Dict[str, Any],
    read_json_fn: Any = None,
) -> List[str]:
    """Discover additional microstructure symbols on disk not yet represented in `microstructure`."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return []
    _rj = _resolve_read_json_cached(read_json_fn)
    covered_micro_bases = {
        canonical_base_asset(k)
        for k in microstructure
        if not has_equity_suffix(k)
    }
    ref_micro_syms: List[str] = []
    for pfx in (CACHE_ORDERBOOK_DEPTH_PREFIX, CACHE_DERIVATIVES_TELEMETRY_PREFIX, CACHE_WHALE_SNAPSHOT_PREFIX):
        for search_dir in (cache_dir, os.path.join(cache_dir, "telemetry")):
            for mpath in sorted(_glob.glob(os.path.join(search_dir, f"{pfx}*.json"))):
                msym = os.path.basename(mpath)[len(pfx) : -len(".json")].upper()
                if (
                    msym
                    and msym.isalnum()
                    and not has_equity_suffix(msym)
                    and msym not in microstructure
                    and msym not in ref_micro_syms
                ):
                    mbase = canonical_base_asset(msym)
                    if mbase not in covered_micro_bases:
                        ref_micro_syms.append(msym)
                        covered_micro_bases.add(mbase)
                    elif pfx == CACHE_ORDERBOOK_DEPTH_PREFIX and isinstance(ob_d := _rj(mpath), dict):
                        cand_bd = safe_nonneg_float(ob_d.get("bid_depth_usd"), default=None)
                        existing_bds = [
                            safe_nonneg_float(mv.get("bid_depth_usd"), default=None)
                            for mk, mv in microstructure.items()
                            if not has_equity_suffix(mk)
                            and canonical_base_asset(mk) == mbase
                            and isinstance(mv, dict)
                        ]
                        if cand_bd is not None and cand_bd > 0 and all(ebd is None or abs(cand_bd - ebd) > 1.0 for ebd in existing_bds):
                            ref_micro_syms.append(msym)
    return ref_micro_syms


def _iter_t212_state_files(workspace_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(t212_dir := os.path.join(workspace_dir, "212trading")):
        return []
    out: List[Tuple[str, str]] = []
    for st_path in sorted(_glob.glob(os.path.join(t212_dir, ".state_*.json"))):
        fname = os.path.basename(st_path)
        ticker = fname[len(".state_") : -len(".json")].upper()
        if not fname.startswith("..") and ticker.replace("_", "").isalnum():
            out.append((st_path, ticker))
    return out


_LAST_T212_PORTFOLIO_POLL_TS: float = 0.0


def _refresh_stale_t212_cached_positions(
    cached_bals: Dict[str, Any],
    bcache_path: Optional[str],
    now_ts: float,
    *,
    max_age_sec: float = 180.0,
) -> bool:
    """Refresh stale (`>= max_age_sec`) `pos:t212:*` prices in `cached_bals` from Trading212 live portfolio."""
    global _LAST_T212_PORTFOLIO_POLL_TS
    if (
        not isinstance(cached_bals, dict)
        or not bcache_path
        or not os.path.exists(bcache_path)
    ):
        return False
    if not is_live_network_backfill_enabled():
        try:
            from providers import t212_provider as _t212_prov_mod

            if getattr(_t212_prov_mod.T212Provider, "__module__", "") == "providers.t212_provider":
                return False
        except Exception:
            return False
    t212_ts_vals = [
        safe_pos_float(bval.get("ts"), default=0.0) or 0.0
        for bkey, bval in cached_bals.items()
        if isinstance(bkey, str) and bkey.startswith("pos:t212:") and isinstance(bval, dict)
    ]
    if not t212_ts_vals:
        return False
    min_t212_ts = min(t212_ts_vals)
    max_t212_ts = max(t212_ts_vals)
    if max_t212_ts <= 0.0 or (now_ts - max(min_t212_ts, _LAST_T212_PORTFOLIO_POLL_TS)) < max_age_sec:
        return False
    try:
        from providers.t212_provider import T212Provider
        from state_io import save_json_state

        _LAST_T212_PORTFOLIO_POLL_TS = now_ts
        try:
            t212_rows = T212Provider()._client().get_portfolio()
        except Exception:
            t212_rows = []
        if not isinstance(t212_rows, list):
            t212_rows = []
        t212_dirty = False
        seen_portfolio_tickers: set[str] = set()
        for row in t212_rows:
            if not isinstance(row, dict):
                continue
            rticker = str(row.get("ticker") or "").strip().upper()
            if rticker:
                seen_portfolio_tickers.add(rticker)
            cur_px_r = safe_pos_float(row.get("currentPrice"), default=None)
            if rticker and cur_px_r is not None and isinstance(c_pos := cached_bals.get(f"pos:t212:{rticker}"), dict):
                prev_c_px = safe_pos_float(c_pos.get("price"), default=None)
                prev_c_ts = safe_pos_float(c_pos.get("ts"), default=None)
                new_c_px = round(cur_px_r, 4)
                if prev_c_px is None or abs(new_c_px - round(prev_c_px, 4)) >= 1e-4 or prev_c_ts is None:
                    c_free = safe_pos_float(
                        c_pos.get("free") if c_pos.get("free") is not None else c_pos.get("free_base"),
                        default=0.0,
                    ) or 0.0
                    c_pos["price"] = new_c_px
                    if c_free > 0.0:
                        c_pos["value_usd"] = round(c_free * cur_px_r, 2)
                    c_pos["ts"] = now_ts
                    t212_dirty = True
        for bkey, c_pos in list(cached_bals.items()):
            if not (isinstance(bkey, str) and bkey.startswith("pos:t212:") and isinstance(c_pos, dict)):
                continue
            rticker = str(c_pos.get("symbol") or bkey.split(":", 2)[-1]).strip().upper()
            if not rticker or rticker in seen_portfolio_tickers:
                continue
            prev_c_ts = safe_pos_float(c_pos.get("ts"), default=0.0) or 0.0
            if (now_ts - prev_c_ts) < max_age_sec:
                continue
            try:
                from providers.base import MarketDataProvider

                cur_px_fb = MarketDataProvider.resolve_venue_public_price_usd("t212", rticker)
                if cur_px_fb is not None:
                    prev_c_px = safe_pos_float(c_pos.get("price"), default=None)
                    new_c_px = round(cur_px_fb, 4)
                    if prev_c_px is None or abs(new_c_px - round(prev_c_px, 4)) >= 1e-4 or prev_c_ts <= 0.0:
                        c_free = safe_pos_float(
                            c_pos.get("free") if c_pos.get("free") is not None else c_pos.get("free_base"),
                            default=0.0,
                        ) or 0.0
                        c_pos["price"] = new_c_px
                        if c_free > 0.0:
                            c_pos["value_usd"] = round(c_free * cur_px_fb, 2)
                        c_pos["ts"] = now_ts
                        t212_dirty = True
            except Exception:
                pass
        if t212_dirty:
            save_json_state(bcache_path, cached_bals, indent=2)
        return t212_dirty
    except Exception:
        return False


def upgrade_tpo_vp_from_ohlc(
    sym: str,
    m24: Dict[str, Any],
    cur_px: float,
    now_ts: float,
    *,
    s24_is_live_clock: bool = True,
    ohlc_fn: Any = None,
    inst_by_sym: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    allow_network: bool = True,
) -> None:
    """Upgrade a TPO volume profile in `m24` to REAL_VOL using 24h OHLC bars when available and non-diverged."""
    if not m24 or m24.get("vp_source") != "TPO" or not s24_is_live_clock:
        return
    if ohlc_fn is None and allow_network:
        ohlc_fn = _resolve_fallback_ohlc_fn(sym, inst_by_sym)
    has_local_ohlc_cache = bool(cache_dir and os.path.isdir(cache_dir))
    if not (
        callable(ohlc_fn)
        or (allow_network and is_live_network_backfill_enabled())
        or os.environ.get("MPTRADE_OFFLINE_DISK_CACHE") == "1"
        or has_local_ohlc_cache
    ):
        return
    try:
        pts_vp = _load_stitched_24h_ohlc_points(
            ohlc_fn if allow_network else None,
            sym,
            now_ts - 86400.0,
            require_positive_volume=True,
            cache_dir=cache_dir,
        )
        if (len(pts_vp) < 8 or (pts_vp[-1][0] - pts_vp[0][0]) < 18 * 3600 * 1000) and cache_dir and os.path.isdir(cache_dir):
            from state_io import read_json_recovering as _rj_vp

            for c_vp in cache_symbol_variants(sym, include_original=True, include_base=False, quotes=compatible_quote_suffixes(sym)):
                p_vp = os.path.join(cache_dir, f"cache_ohlc_24h_{c_vp}.json")
                if os.path.exists(p_vp) and isinstance(d_vp := _rj_vp(p_vp), dict) and isinstance(d_vp.get("points"), list):
                    cand_vp = [
                        [int(r[0]), float(r[1]), float(r[2])]
                        for r in d_vp["points"]
                        if isinstance(r, (list, tuple)) and len(r) >= 3 and (safe_pos_float(r[2], 0.0) or 0.0) > 0.0
                    ]
                    if len(cand_vp) >= 8 and (cand_vp[-1][0] - cand_vp[0][0]) >= 18 * 3600 * 1000:
                        pts_vp = cand_vp
                        break
        if len(pts_vp) >= 8 and (pts_vp[-1][0] - pts_vp[0][0]) >= 18 * 3600 * 1000:
            vp_last_px = safe_pos_float(pts_vp[-1][1], default=0.0) or 0.0
            if cur_px <= 0.0 or (vp_last_px > 0.0 and abs(cur_px - vp_last_px) / max(cur_px, vp_last_px) <= 0.015):
                l24_b = safe_pos_float(m24.get("low_24h"), default=None)
                h24_b = safe_pos_float(m24.get("high_24h"), default=None)
                vp_pxs = [float(p[1]) for p in pts_vp if len(p) >= 2 and safe_pos_float(p[1], 0.0)]
                if (
                    l24_b is None
                    or h24_b is None
                    or (vp_pxs and min(vp_pxs) >= l24_b * 0.985 and max(vp_pxs) <= h24_b * 1.015)
                ):
                    if l24_b is not None and h24_b is not None:
                        pts_vp = [[p[0], min(h24_b, max(l24_b, float(p[1]))), *p[2:]] for p in pts_vp]
                    if cur_px > 0:
                        pts_vp.append([int(now_ts * 1000), cur_px, 0.0])
                    if (vp_real := compute_volume_profile_metrics(pts_vp)) and vp_real.get("vp_source") == "REAL_VOL":
                        for vp_clr_k in ("sec_poc_px", "hvn2_px", "lvn_px"):
                            m24.pop(vp_clr_k, None)
                        m24.update(vp_real)
    except Exception:
        pass


def _persist_refreshed_equity_chart_cache(
    cache_dir: str,
    sym: str,
    p24_eq_path: str,
    pts_24h_to_save: list,
    pts_24h_bars: list,
    pts_5d: list,
    raw_matched_series_24h: list,
    rth_bar_fresh: bool,
    now_ts: float,
) -> None:
    if not (cache_dir and os.path.isdir(cache_dir) and is_live_network_backfill_enabled()):
        return
    try:
        from state_io import read_json_dict, save_json_state

        _ensure_cacheprice_classes_bound()
        eff_p24_pts = pts_24h_to_save or pts_24h_bars
        p24_target = p24_eq_path or str(
            resolve_cachedb_filepath(os.path.join(cache_dir, Cache24PriceManager.disk_filename_for_symbol(sym)))
        )
        disk_p24_obj = read_json_dict(p24_target) if (p24_target and os.path.exists(p24_target)) else {}
        disk_p24_rows = _extract_cached_items_series(disk_p24_obj, (sym,)) if isinstance(disk_p24_obj, dict) else []
        check_rows = disk_p24_rows or raw_matched_series_24h
        check_last_pt = _parse_ts_px_row(check_rows[-1]) if check_rows else None
        eff_last_pt = _parse_ts_px_row(eff_p24_pts[-1]) if eff_p24_pts else None
        raw_ohlc_cnt = sum(1 for p in check_rows if isinstance(p, (list, tuple)) and len(p) >= 3)
        new_ohlc_cnt = sum(1 for p in eff_p24_pts if isinstance(p, (list, tuple)) and len(p) >= 3)
        if eff_p24_pts and (
            len(check_rows) < 10
            or raw_ohlc_cnt < new_ohlc_cnt
            or check_last_pt is None
            or (
                eff_last_pt is not None
                and (eff_last_pt[0] > check_last_pt[0] or abs(eff_last_pt[1] - check_last_pt[1]) > 1e-4)
            )
            or (
                not rth_bar_fresh
                and raw_ohlc_cnt == 0
                and any(isinstance(p, (list, tuple)) and len(p) < 3 for p in check_rows)
            )
        ):
            Cache24PriceManager.persist_external_ohlc_series(
                p24_target,
                sym,
                eff_p24_pts,
                now_ts,
                existing_obj=disk_p24_obj if isinstance(disk_p24_obj, dict) else None,
            )
        if pts_24h_to_save or pts_24h_bars:
            ohlc24_target = str(resolve_cachedb_filepath(os.path.join(cache_dir, f"cache_ohlc_24h_{sym}.json")))
            save_json_state(ohlc24_target, {"ts": now_ts, "symbol": sym, "points": pts_24h_to_save or pts_24h_bars})
        p365_jsonl_path = str(
            resolve_cachedb_filepath(os.path.join(cache_dir, Cache365PriceManager.disk_filename_for_symbol(sym)))
        )
        Cache365PriceManager.seed_initial_archive_if_missing(p365_jsonl_path, sym, pts_5d)
    except Exception:
        pass


class PriceSeriesCacheCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for price-series cache collectors (24h, 365d, sparse, watcher, live, Yahoo 5d)."""

    cls_name = "PriceSeriesCacheCollectorBase"

    def __init__(self, *, start_sync: bool = False) -> None:
        if type(self) is PriceSeriesCacheCollectorBase:
            raise TypeError("Cannot instantiate abstract class PriceSeriesCacheCollectorBase")
        super().__init__(start_sync=start_sync)


class EquityYahoo5dChartCollector(PriceSeriesCacheCollectorBase):
    """Concrete collector (Level 3) for Yahoo 5d/5m equity chart backfill and disk persistence."""

    cls_name = "EquityYahoo5dChartCollector"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        workspace_dir: str = "",
        raw_series_24h: Optional[list] = None,
        now_ts: Optional[float] = None,
        eq_rth_open: bool = False,
        memo_chart: Optional[Tuple[float, Any]] = None,
        memo_chart_fresh: bool = False,
        allow_network: bool = True,
        persist: bool = False,
        p24_eq_path: str = "",
        **kwargs: Any,
    ) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]], bool, bool, float, int]:
        if not allow_network and not (memo_chart_fresh and memo_chart is not None):
            return [], [], [], False, False, 0.0, 0
        eff_now = resolve_now_ts(now_ts)
        ws_dir = workspace_dir
        if not ws_dir and cachedb_dir:
            ws_dir = os.path.dirname(os.path.abspath(cachedb_dir.rstrip("/\\")))
            if not os.path.exists(os.path.join(ws_dir, "212trading", "market_data.py")):
                ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        raw_s24 = list(raw_series_24h or [])
        if memo_chart_fresh and memo_chart is not None:
            chart_res = memo_chart[1]
        else:
            _chart = load_local_module(os.path.join(ws_dir, "212trading"), "market_data")._chart
            chart_res = _chart(canonical_base_asset(symbol) or symbol, rng="5d", interval="5m")
            if is_live_network_backfill_enabled():
                _EQ_YAHOO_CHART_MEMO[symbol] = (eff_now, chart_res)
        res = normalize_yahoo_5d_chart_bars(chart_res, raw_s24, eff_now, eq_rth_open=eq_rth_open)
        if persist and (res[0] or res[2]):
            _persist_refreshed_equity_chart_cache(
                cachedb_dir,
                symbol,
                p24_eq_path,
                list(res[2]),
                list(res[1]),
                list(res[0]),
                raw_s24,
                res[3],
                eff_now,
            )
        return res


class CompositeSymbolPriceSeriesLoader(PriceSeriesCacheCollectorBase):
    """Concrete composite loader (Level 3) that reads & merges artifacts produced by the ``CachePriceManagerBase`` hierarchy:
    - ``Cache24PriceManager`` (``cache_24_price_<SYM>.json``)
    - ``Cache365PriceManager`` (``cache_365_price_<SYM>.jsonl`` — dense long-horizon archive)
    - ``CacheSparsePriceManager`` (``cache_sparse_price_<SYM>.jsonl`` — sparse long-horizon archive)
    - ``CacheCurrentPriceManager`` (``cache_current_price.json`` — live ticks)
    """

    cls_name = "CompositeSymbolPriceSeriesLoader"
    price_24_cls: Any = None
    price_365_cls: Any = None
    price_sparse_cls: Any = None
    current_price_cls: Any = None

    def __init__(self, *, start_sync: bool = False) -> None:
        super().__init__(start_sync=start_sync)
        _ensure_cacheprice_classes_bound()

    def collect(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        current_price_items: Optional[Dict[str, Any]] = None,
        include_7d: bool = True,
        quotes: Optional[Tuple[str, ...]] = None,
        **kwargs: Any,
    ) -> Tuple[list, list]:
        sym_u = str(symbol).strip().upper()
        session_policy = self.resolve_session_policy(sym_u)
        is_eq_sym_load = session_policy.is_equity
        is_eq_sym = is_eq_sym_load or is_equity_symbol(sym_u)
        eff_quotes = quotes if quotes is not None else compatible_quote_suffixes(symbol)
        cands = [
            c
            for c in cache_symbol_variants(symbol, include_original=True, include_base="USD" in eff_quotes, quotes=eff_quotes)
            if isinstance(c, str)
            and "/" not in c
            and (
                not c.upper().endswith(("USD", "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "EUR", "BTC", "ETH"))
                if is_eq_sym
                else not is_equity_symbol(c)
            )
        ]
        _ensure_cacheprice_classes_bound()
        _rj_cand = _resolve_read_json_cached()
        is_default_rj = _rj_cand is _default_read_json_cached
        _rj = read_json_recovering if is_default_rj else _rj_cand
        memo_key: Optional[Tuple[Any, ...]] = None
        memo_file_sig: Tuple[Any, ...] = ()
        memo_cur_sig: Tuple[Any, ...] = ()
        if is_default_rj:
            try:
                memo_key = (os.path.abspath(str(cachedb_dir)), sym_u, bool(include_7d), tuple(eff_quotes))
                stat_list = []
                for sp in _candidate_cache_paths(
                    cachedb_dir,
                    *(Cache24PriceManager.disk_filename_for_symbol(c) for c in cands),
                    *((f"cache_ohlc_24h_{c}.json" for c in cands) if is_eq_sym else ()),
                    *(CacheSparsePriceManager.disk_filename_for_symbol(c) for c in cands),
                    *(Cache365PriceManager.disk_filename_for_symbol(c) for c in cands),
                    *(f"cache_365_price_{c}.json" for c in cands),
                ):
                    for cand_sp in (sp, sp + ".previous") if Cache24PriceManager.FILE_PREFIX in sp else (sp,):
                        try:
                            st = os.stat(cand_sp)
                            stat_list.append((cand_sp, st.st_mtime_ns, st.st_size))
                        except OSError:
                            pass
                if not stat_list:
                    for w_sp in _candidate_cache_paths(cachedb_dir, CACHE_WATCHER_ALL_PRICE_FILENAME):
                        try:
                            st = os.stat(w_sp)
                            stat_list.append((w_sp, st.st_mtime_ns, st.st_size))
                        except OSError:
                            pass
                cur_list = []
                if isinstance(current_price_items, dict) and current_price_items:
                    for c in cands:
                        ci = casefold_get(current_price_items, c)
                        if isinstance(ci, list) and ci:
                            cur_list.append((c, len(ci), repr(ci[-1])))
                        elif ci is not None:
                            cur_list.append((c, extract_current_price_item(ci)))
                memo_file_sig, memo_cur_sig = tuple(stat_list), tuple(cur_list)
                cached_entry = _PRICE_SERIES_MEMO.get(memo_key)
                if cached_entry is not None and cached_entry[0] == memo_file_sig and cached_entry[1] == memo_cur_sig:
                    _LAST_7D_SPAN_MS_BY_SYM[sym_u] = cached_entry[4]
                    return [list(pt) for pt in cached_entry[2]], list(cached_entry[3])
            except Exception:
                memo_key = None

        series_24h: list = []
        span_24h_ms = 0.0
        for cand in cands:
            for p24_path in _candidate_cache_paths(cachedb_dir, Cache24PriceManager.disk_filename_for_symbol(cand)):
                if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                    continue
                raw_ser = sort_ts_rows(_extract_cached_items_series(_rj(p24_path), cands))
                if is_eq_sym_load and len(raw_ser) >= 2:
                    for ohlc24_path in _candidate_cache_paths(cachedb_dir, f"cache_ohlc_24h_{cand}.json"):
                        if not os.path.exists(ohlc24_path):
                            continue
                        ohlc24_obj = _rj(ohlc24_path)
                        ohlc24_pts = (
                            ohlc24_obj.get("points")
                            if isinstance(ohlc24_obj, dict) and isinstance(ohlc24_obj.get("points"), list)
                            else _extract_cached_items_series(ohlc24_obj, cands)
                        )
                        raw_ser = session_policy.merge_ohlc24_into_raw_series(raw_ser, ohlc24_pts or [])
                        break
                ser = session_policy.clean_raw_24h_series(raw_ser, symbol)
                if len(ser) >= 2:
                    series_24h = list(ser) if not series_24h else prepend_older_non_diverged_points(series_24h, ser)
                    p0, p1 = _parse_ts_px_row(series_24h[0]), _parse_ts_px_row(series_24h[-1])
                    span_24h_ms = max(0.0, p1[0] - p0[0]) if (p0 and p1) else 0.0
                    break
            if series_24h and span_24h_ms >= 3600000.0 * 18:
                break
        need_24h_supplement = span_24h_ms < 3600000.0 * 18
        archive_pts: list = []
        sparse_7d_pts: list = []
        cands_upper = {str(c).strip().upper() for c in cands if isinstance(c, str) and str(c).strip()}
        cands_upper_tup = tuple(sorted(cands_upper))
        for cand in (cands if (include_7d or need_24h_supplement or is_eq_sym_load) else ()):
            f_sparse = CacheSparsePriceManager.disk_filename_for_symbol(cand)
            f_365 = Cache365PriceManager.disk_filename_for_symbol(cand)
            f_365_legacy = f"cache_365_price_{cand}.json"
            primary_names = (
                (f_sparse, f_365, f_365_legacy)
                if not need_24h_supplement
                else (f_365, f_365_legacy, f_sparse)
            )
            cand_arch: list = []
            for arch_path in _candidate_cache_paths(cachedb_dir, *primary_names):
                if rows := _read_archive_series_file(arch_path, cands, cands_upper, cands_upper_tup, _rj, is_default_rj):
                    cand_arch = rows
                    break
            if cand_arch:
                archive_pts = cand_arch if not archive_pts else prepend_older_non_diverged_points(archive_pts, cand_arch)
                a0, a1 = _parse_ts_px_row(archive_pts[0]), _parse_ts_px_row(archive_pts[-1])
                arch_span_ms = (a1[0] - a0[0]) if (a0 and a1) else 0.0
                if include_7d and arch_span_ms < 5.0 * 86400.0 * 1000.0 and not sparse_7d_pts:
                    for sp_path in _candidate_cache_paths(
                        cachedb_dir,
                        f_sparse,
                        f_365,
                        f_365_legacy,
                    ):
                        if sp_rows := _read_archive_series_file(
                            sp_path, cands, cands_upper, cands_upper_tup, _rj, is_default_rj, max_b=512 * 1024
                        ):
                            sp0, sp1 = _parse_ts_px_row(sp_rows[0]), _parse_ts_px_row(sp_rows[-1])
                            if (
                                sp0
                                and sp1
                                and (sp1[0] - sp0[0]) > arch_span_ms
                                and (a0 is None or sp1[0] >= a0[0] - 86400.0 * 1000.0)
                                and (a1 is None or not _prices_diverged(sp1[1], a1[1]))
                            ):
                                sparse_7d_pts = sp_rows
                                break
                if arch_span_ms >= 5.0 * 86400.0 * 1000.0 or sparse_7d_pts:
                    break

        watcher_cutoff_24h: Optional[float] = None
        watcher_7d_pts: list = []
        if not archive_pts and need_24h_supplement and not series_24h:
            for w_path in _candidate_cache_paths(cachedb_dir, CACHE_WATCHER_ALL_PRICE_FILENAME):
                if not os.path.exists(w_path):
                    continue
                w_raw = lookup_symbol_cache_items(
                    _rj(w_path),
                    cand_keys=cands,
                    allow_base_fallback=False,
                    single_item_quote_fallback=False,
                    require_list=True,
                    unwrap_items=True,
                )
                collapsed_w, w_7d, w_cut_24 = collapse_watcher_equity_series(w_raw or [], is_eq_sym_load)
                if collapsed_w:
                    archive_pts = collapsed_w
                    watcher_7d_pts = w_7d
                    watcher_cutoff_24h = w_cut_24
                    break
        series_24h, vals_7d, arch_7d_end_ts, arch_7d_span_ms = extract_archive_24h_and_7d_slices(
            archive_pts, sparse_7d_pts, watcher_7d_pts, series_24h, symbol, need_24h_supplement, watcher_cutoff_24h
        )
        _LAST_7D_SPAN_MS_BY_SYM[sym_u] = arch_7d_span_ms
        if isinstance(current_price_items, dict) and current_price_items:
            last_24_pt = _parse_ts_px_row(series_24h[-1]) if series_24h else None
            for cand in cands:
                cur_item = casefold_get(current_price_items, cand)
                if isinstance(cur_item, list) and cur_item:
                    cand_pts = [
                        [pt[0], pt[1]]
                        for pt in cur_item
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2 and _parse_ts_px_row(pt) is not None
                    ]
                    cand_last = _parse_ts_px_row(cand_pts[-1]) if cand_pts else None
                    if cand_pts and cand_last is not None:
                        if last_24_pt is not None and _prices_diverged(last_24_pt[1], cand_last[1]):
                            continue
                        if (
                            is_eq_sym_load
                            and len(series_24h) >= 2
                            and last_24_pt is not None
                            and abs(cand_last[1] - last_24_pt[1]) <= 1e-6
                            and (
                                cand_last[0] <= last_24_pt[0]
                                or (cand_last[0] - last_24_pt[0]) > 20 * 60 * 1000
                            )
                        ):
                            break
                        series_24h = list(series_24h) + cand_pts
                        break
                if (px_f := extract_current_price_item(cur_item)) is not None and series_24h and last_24_pt is not None:
                    if not _prices_diverged(last_24_pt[1], px_f):
                        if is_eq_sym_load and len(series_24h) >= 2 and abs(px_f - last_24_pt[1]) <= 1e-6:
                            break
                        raw_last_t = (
                            safe_pos_float(series_24h[-1][0], default=None)
                            if isinstance(series_24h[-1], (list, tuple)) and series_24h[-1]
                            else None
                        )
                        next_raw_t = (
                            (raw_last_t + 1.0)
                            if (raw_last_t is not None and raw_last_t < 1e11 - 1.0)
                            else (last_24_pt[0] + 1000.0)
                        )
                        series_24h = list(series_24h) + [[next_raw_t, px_f]]
                        break
        series_24h = session_policy.normalize_24h_series(series_24h, symbol)
        vals_7d = append_newer_24h_to_7d_vals(vals_7d, series_24h, arch_7d_end_ts)

        if memo_key is not None:
            if len(_PRICE_SERIES_MEMO) > 128:
                _PRICE_SERIES_MEMO.clear()
            _PRICE_SERIES_MEMO[memo_key] = (
                memo_file_sig,
                memo_cur_sig,
                [list(pt) for pt in series_24h],
                list(vals_7d),
                arch_7d_span_ms,
            )
        return series_24h, vals_7d


class MicrostructureCacheCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for orderbook, derivatives, and whale positioning cache collectors."""

    cls_name = "MicrostructureCacheCollectorBase"

    def __init__(self, *, start_sync: bool = False) -> None:
        if type(self) is MicrostructureCacheCollectorBase:
            raise TypeError("Cannot instantiate abstract class MicrostructureCacheCollectorBase")
        super().__init__(start_sync=start_sync)


class SymbolMicrostructureSnapshotCollector(MicrostructureCacheCollectorBase):
    """Concrete collector (Level 3) for per-symbol orderbook depth, derivatives OI/funding, and whale snapshots."""

    cls_name = "SymbolMicrostructureSnapshotCollector"
    COLLECTOR_ID = "symbol_microstructure_snapshot"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        read_json_fn: Any = None,
        ensure_fresh: bool = False,
        ob: Optional[Dict[str, Any]] = None,
        deriv: Optional[Dict[str, Any]] = None,
        whale: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        if ensure_fresh:
            ensure_fresh_symbol_microstructure(cachedb_dir, symbol, read_json_fn=read_json_fn)
        return load_symbol_microstructure(cachedb_dir, symbol, read_json_fn=read_json_fn, ob=ob, deriv=deriv, whale=whale)


class Fleet24hAndMicrostructureEnricher(MicrostructureCacheCollectorBase):
    """Concrete collector/enricher (Level 3) that orchestrates fleet-wide 24h/7d series and microstructure enrichment."""

    cls_name = "Fleet24hAndMicrostructureEnricher"
    COLLECTOR_ID = "fleet_24h_and_microstructure"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "BTCUSDC",
        *,
        workspace_dir: str = "",
        prices_and_trends: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
        cur_p_items: Optional[Dict[str, Any]] = None,
        inst_by_sym: Optional[Dict[str, Any]] = None,
        now_ts: Optional[float] = None,
        read_json_fn: Any = None,
        allow_network: bool = True,
        persist: bool = True,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pt_map = prices_and_trends if isinstance(prices_and_trends, dict) else {}
        m_map = microstructure if isinstance(microstructure, dict) else {}
        cp_map = cur_p_items if isinstance(cur_p_items, dict) else {}
        inst_map = inst_by_sym if isinstance(inst_by_sym, dict) else {}
        eff_now = resolve_now_ts(now_ts)
        _rj = _resolve_read_json_cached(read_json_fn)
        btc_series_24h, btc_vals_7d = load_symbol_price_series(
            cachedb_dir, symbol or "BTCUSDC", current_price_items=cp_map, include_7d=True
        )
        bcache_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, "venue_balances_state.json"))) if cachedb_dir else None
        raw_cb = _rj(bcache_path) if bcache_path and os.path.exists(bcache_path) else None
        bcache_mtime = 0.0
        if bcache_path and os.path.exists(bcache_path):
            try:
                bcache_mtime = os.path.getmtime(bcache_path)
            except OSError:
                bcache_mtime = 0.0
        cached_bals: Dict[str, Any] = raw_cb if isinstance(raw_cb, dict) else {}
        t212_seeded_from_cache: set[str] = set()
        t212_pos_ts: Dict[str, float] = {}
        t212_seed_tickers = [ticker for _, ticker in _iter_t212_state_files(workspace_dir)]
        for bkey, bval in cached_bals.items():
            if isinstance(bkey, str) and bkey.startswith("pos:t212:") and isinstance(bval, dict):
                bticker = str(bval.get("symbol") or bkey.split(":", 2)[-1]).strip().upper()
                b_free = safe_pos_float(
                    bval.get("free") if bval.get("free") is not None else bval.get("free_base"),
                    default=0.0,
                ) or 0.0
                if bticker and bticker not in t212_seed_tickers and (bticker in pt_map or b_free > 0.0):
                    t212_seed_tickers.append(bticker)
        if (
            allow_network
            and t212_seed_tickers
            and not any(t in inst_map for t in t212_seed_tickers)
            and _rj in (_default_read_json_cached, read_json_recovering, read_json_cached)
        ):
            _refresh_stale_t212_cached_positions(cached_bals, bcache_path, eff_now)
        for ticker in t212_seed_tickers:
            cached_pos = cached_bals.get(f"pos:t212:{ticker}")
            cached_t212_px = safe_pos_float(cached_pos.get("price") if isinstance(cached_pos, dict) else None, default=None)
            if cached_t212_px is not None:
                cached_t212_px = round(cached_t212_px, 4)
            pos_ts_val = safe_pos_float(cached_pos.get("ts") if isinstance(cached_pos, dict) else None, default=None)
            cp_raw = cp_map.get(ticker) if isinstance(cp_map, dict) else None
            cp_pt = _parse_ts_px_row(cp_raw[-1]) if isinstance(cp_raw, list) and cp_raw else None
            if (
                pos_ts_val is not None
                and cached_t212_px is not None
                and cp_pt is not None
                and cp_pt[0] > 1e11
                and abs(cached_t212_px - round(cp_pt[1], 4)) < 1e-4
            ):
                cp_ts_sec = cp_pt[0] / 1000.0
                if 0.0 < cp_ts_sec < pos_ts_val:
                    pos_ts_val = cp_ts_sec
            if pos_ts_val is not None:
                t212_pos_ts[ticker] = pos_ts_val
            elif not is_live_network_backfill_enabled() and bcache_mtime > 0.0:
                t212_pos_ts[ticker] = bcache_mtime
            t_entry = pt_map.setdefault(ticker, {"price": cached_t212_px, "trend": 0, "growth_coefficient": 0.0})
            cur_t_px = safe_pos_float(t_entry.get("price"), default=0.0) or 0.0
            if cached_t212_px is not None and (
                ticker not in inst_map
                or cur_t_px <= 0.0
                or abs(round(cur_t_px, 4) - cached_t212_px) < 1e-4
                or (
                    cp_pt is not None
                    and abs(round(cur_t_px, 4) - round(cp_pt[1], 4)) < 1e-4
                    and (pos_ts_val or 0.0) > (cp_pt[0] / 1000.0)
                )
            ):
                t_entry["price"] = cached_t212_px
                t212_seeded_from_cache.add(ticker)
        from instrument_registry import discover_hyperliquid_spot

        for hl_sym, hl_base, hl_quote in discover_hyperliquid_spot(workspace_dir):
            ensure_fresh_symbol_microstructure(
                cachedb_dir,
                hl_sym,
                now_ts=eff_now,
                max_age_ob_sec=180.0,
                max_age_whale_sec=360.0,
                include_derivatives=True,
                allow_network=allow_network,
            )
            hl_m_seed = load_symbol_microstructure(cachedb_dir, hl_sym, _rj) if hl_sym not in pt_map else None
            if hl_sym not in pt_map and (
                any(not has_equity_suffix(k) and canonical_base_asset(k) == hl_base for k in pt_map)
                or hl_m_seed
            ):
                cached_hl_pos = cached_bals.get(f"pos:hyperliquid:{hl_sym}")
                hl_px = (
                    safe_pos_float((hl_m_seed or {}).get("ob_mid_price"), default=0.0)
                    or safe_pos_float(cached_hl_pos.get("price") if isinstance(cached_hl_pos, dict) else None, default=0.0)
                    or resolve_symbol_price(hl_sym, pt_map, explicit_quote=hl_quote)
                )
                pt_map[hl_sym] = {"price": hl_px if hl_px > 0 else None, "trend": 0, "growth_coefficient": 0.0}
        for sym in list(pt_map.keys()):
            sym_px_cur = safe_float(pt_map[sym].get("price"), 0.0)
            is_eq_sym = has_equity_suffix(sym)
            eq_div_thr = 0.35 if is_eq_sym else 0.10
            if sym == "BTCUSDC" and btc_series_24h:
                matched_series_24h, arch_vals_7d = list(btc_series_24h), list(btc_vals_7d)
            else:
                matched_series_24h, arch_vals_7d = load_symbol_price_series(cachedb_dir, sym, current_price_items=cp_map)
            raw_matched_series_24h = list(matched_series_24h)
            raw_s24_last_pt = _parse_ts_px_row(matched_series_24h[-1]) if matched_series_24h else None
            raw_s24_last_ts_ms = raw_s24_last_pt[0] if raw_s24_last_pt is not None else 0.0
            if sym_px_cur > 0 and matched_series_24h:
                last_24_pt = _parse_ts_px_row(matched_series_24h[-1])
                last_24_px = last_24_pt[1] if last_24_pt is not None else safe_float(matched_series_24h[-1][1], 0.0)
                eff_t212_ts = t212_pos_ts.get(sym, 0.0)
                if (
                    sym in t212_seeded_from_cache
                    and last_24_pt is not None
                    and last_24_px > 0.0
                    and last_24_pt[0] > 1e11
                    and (int(eff_now * 1000) - last_24_pt[0]) <= 20 * 60 * 1000
                    and not _prices_diverged(sym_px_cur, last_24_px, threshold=eq_div_thr)
                    and (
                        eff_t212_ts <= 0.0
                        or (last_24_pt[0] / 1000.0) > eff_t212_ts
                        or abs(sym_px_cur - last_24_px) / max(sym_px_cur, last_24_px) > 0.005
                    )
                ):
                    pt_map[sym]["price"] = sym_px_cur = round(last_24_px, 4)
                if _prices_diverged(sym_px_cur, last_24_px, threshold=eq_div_thr):
                    matched_series_24h = []
                elif last_24_px > 0 and abs(sym_px_cur - last_24_px) > 1e-9:
                    now_ms_24 = int(eff_now * 1000)
                    if not is_eq_sym or (
                        last_24_pt is not None
                        and (now_ms_24 - last_24_pt[0]) <= 20 * 60 * 1000
                        and abs(sym_px_cur - last_24_px) / max(sym_px_cur, last_24_px) <= 0.005
                    ):
                        eq_pt_ts_ms = (
                            int(eff_t212_ts * 1000)
                            if is_eq_sym
                            and sym in t212_seeded_from_cache
                            and eff_t212_ts > 0.0
                            and last_24_pt is not None
                            and int(eff_t212_ts * 1000) > last_24_pt[0]
                            else now_ms_24
                        )
                        if is_eq_sym and last_24_pt is not None and (eq_pt_ts_ms - last_24_pt[0]) > 300 * 1000:
                            eq_pt_ts_ms = int(last_24_pt[0] + 1000)
                        matched_series_24h = list(matched_series_24h) + [[eq_pt_ts_ms, sym_px_cur]]
            if sym_px_cur > 0 and arch_vals_7d:
                if _prices_diverged(sym_px_cur, safe_float(arch_vals_7d[-1], 0.0), threshold=eq_div_thr):
                    arch_vals_7d = []
                elif abs(sym_px_cur - safe_float(arch_vals_7d[-1], 0.0)) > 1e-9:
                    arch_vals_7d = list(arch_vals_7d) + [sym_px_cur]
            if safe_float(pt_map[sym].get("price"), 0.0) <= 0.0:
                if matched_series_24h:
                    pt_map[sym]["price"] = matched_series_24h[-1][1]
                elif arch_vals_7d:
                    pt_map[sym]["price"] = arch_vals_7d[-1]
            s24_p0 = _parse_ts_px_row(matched_series_24h[0]) if len(matched_series_24h) >= 2 else None
            s24_p1 = _parse_ts_px_row(matched_series_24h[-1]) if len(matched_series_24h) >= 2 else None
            span_24h_ms = max(0.0, s24_p1[0] - s24_p0[0]) if (s24_p0 and s24_p1) else 0.0
            m24: Dict[str, Any] = compute_24h_price_metrics(matched_series_24h) or {} if matched_series_24h else {}
            session_policy = resolve_asset_session_policy(sym)
            matched_series_24h, m24, post_rth_px_override, post_rth_series_replaced = session_policy.evaluate_post_rth_override(
                matched_series_24h,
                m24,
                sym_px_cur,
                eff_now,
                entry=pt_map[sym],
            )
            arch_span_7d_ms = _LAST_7D_SPAN_MS_BY_SYM.get(sym.upper(), 0.0)
            active_7d_vals: List[float] = []
            if arch_vals_7d and len(arch_vals_7d) >= 2 and (
                is_eq_sym
                or "chg_7d_pct" not in pt_map[sym]
                or arch_span_7d_ms >= 5 * 86400 * 1000
                or (
                    pt_map[sym].get("_partial_7d_span_ms") is not None
                    and arch_span_7d_ms > safe_float(pt_map[sym].get("_partial_7d_span_ms"), 0.0)
                )
            ):
                for k_7d_opt in ("major_swing_low_px", "rally_from_major_low_pct", "recent_swings_7d", "avg_swing_amp_7d_pct"):
                    pt_map[sym].pop(k_7d_opt, None)
                pt_map[sym].update(compute_7d_range_metrics(arch_vals_7d, decimals=4 if is_eq_sym else None))
                active_7d_vals = list(arch_vals_7d)
                if 0.0 <= arch_span_7d_ms < 5 * 86400 * 1000:
                    pt_map[sym]["_partial_7d_span_ms"] = arch_span_7d_ms
                else:
                    pt_map[sym].pop("_partial_7d_span_ms", None)
            ohlc_fn = _resolve_fallback_ohlc_fn(sym, inst_map) if allow_network else None
            s24_is_live_clock = (
                raw_s24_last_ts_ms <= 0.0
                or abs(eff_now * 1000.0 - raw_s24_last_ts_ms) <= 30 * 60 * 1000.0
                or sym in (inst_map or {})
            )
            if allow_network and callable(ohlc_fn) and s24_is_live_clock:
                if not m24 or (0.0 < span_24h_ms < 18 * 3600 * 1000):
                    try:
                        pts_24h = _load_stitched_24h_ohlc_points(ohlc_fn, sym, eff_now - 86400.0)
                        if not m24:
                            if m24 := (compute_24h_price_metrics(pts_24h) or {}):
                                matched_series_24h = pts_24h
                        elif len(pts_24h) >= 8 and (pts_24h[-1][0] - pts_24h[0][0]) >= 18 * 3600 * 1000 and s24_p0 is not None:
                            older_ohlc = [pt for pt in pts_24h if pt[0] < s24_p0[0]]
                            if older_ohlc:
                                matched_series_24h = older_ohlc + list(matched_series_24h)
                                if m24_full := (compute_24h_price_metrics(matched_series_24h) or {}):
                                    m24 = m24_full
                    except Exception:
                        pass
                if (
                    "chg_7d_pct" not in pt_map[sym]
                    or (0.0 < span_24h_ms < 18 * 3600 * 1000)
                    or (0.0 < arch_span_7d_ms < 5 * 86400 * 1000)
                    or pt_map[sym].get("_partial_7d_span_ms") is not None
                ):
                    try:
                        vals_7d = _load_stitched_7d_ohlc_closes(ohlc_fn, sym, eff_now, m24, sym_px_cur)
                        if len(vals_7d) >= 2 and ("chg_7d_pct" not in pt_map[sym] or len(vals_7d) >= 24):
                            for k_7d_opt in ("major_swing_low_px", "rally_from_major_low_pct", "recent_swings_7d", "avg_swing_amp_7d_pct"):
                                pt_map[sym].pop(k_7d_opt, None)
                            pt_map[sym].update(compute_7d_range_metrics(vals_7d))
                            pt_map[sym].pop("_partial_7d_span_ms", None)
                            active_7d_vals = list(vals_7d)
                    except Exception:
                        pass
            upgrade_tpo_vp_from_ohlc(
                sym,
                m24,
                sym_px_cur,
                eff_now,
                s24_is_live_clock=s24_is_live_clock,
                ohlc_fn=ohlc_fn,
                inst_by_sym=inst_map,
                allow_network=allow_network,
            )
            pt_map[sym].pop("_partial_7d_span_ms", None)
            is_t212_inst = sym in inst_map and normalize_venue_name(getattr(inst_map[sym], "provider_name", "")) == "t212"
            if allow_network:
                _ensure_cacheprice_classes_bound()
                p24_eq_path = (
                    str(resolve_cachedb_filepath(os.path.join(cachedb_dir, Cache24PriceManager.disk_filename_for_symbol(sym))))
                    if cachedb_dir and is_eq_sym
                    else ""
                )
                memo_chart = _EQ_YAHOO_CHART_MEMO.get(sym) if (is_eq_sym and is_live_network_backfill_enabled()) else None
                memo_ttl_s = 240.0
                eq_rth_open = False
                if is_eq_sym:
                    try:
                        from instrument_registry import is_symbol_market_open

                        eq_rth_open = bool(is_symbol_market_open(sym, eff_now))
                        if not eq_rth_open and memo_chart is not None:
                            memo_ttl_s = 1800.0
                    except Exception:
                        pass
                memo_chart_fresh = bool(memo_chart is not None and 0.0 <= (eff_now - memo_chart[0]) < memo_ttl_s)
                eq_cache_stale = bool(
                    is_eq_sym
                    and is_live_network_backfill_enabled()
                    and (
                        memo_chart is not None
                        or eq_rth_open
                        or raw_s24_last_ts_ms <= 0.0
                        or (int(eff_now * 1000) - raw_s24_last_ts_ms) > 5 * 60 * 1000
                        or "chg_24h_pct" not in m24
                        or "chg_1h_pct" not in m24
                        or m24.get("vp_source") == "TPO"
                        or not p24_eq_path
                        or not os.path.exists(p24_eq_path)
                    )
                )
                if (not m24 and (is_t212_inst or (is_eq_sym and is_live_network_backfill_enabled()))) or eq_cache_stale:
                    try:
                        pts_5d, pts_24h_bars, pts_24h, rth_bar_fresh, chart_tick_fresh, chart_px, raw_last_5d_ts = (
                            TelemetryCollectorFactory.get_yahoo_chart_collector().collect(
                                cachedb_dir,
                                sym,
                                workspace_dir=workspace_dir,
                                raw_series_24h=raw_matched_series_24h,
                                now_ts=eff_now,
                                eq_rth_open=eq_rth_open,
                                memo_chart=memo_chart,
                                memo_chart_fresh=memo_chart_fresh,
                                persist=False,
                            )
                        )
                        if pts_5d:
                            if chart_px > 0.0 and (
                                sym_px_cur <= 0.0
                                or abs(sym_px_cur - chart_px) < 1e-4
                                or (
                                    (not is_t212_inst or (is_live_network_backfill_enabled() and eq_cache_stale))
                                    and chart_tick_fresh
                                    and not _prices_diverged(sym_px_cur, chart_px, threshold=eq_div_thr)
                                    and (
                                        sym not in t212_seeded_from_cache
                                        or t212_pos_ts.get(sym, 0.0) <= 0.0
                                        or (raw_last_5d_ts / 1000.0) > t212_pos_ts.get(sym, 0.0)
                                        or abs(sym_px_cur - chart_px) / max(sym_px_cur, chart_px) > 0.005
                                    )
                                )
                            ):
                                pt_map[sym]["price"] = chart_px
                                sym_px_cur = chart_px
                            pts_24h_to_save = list(pts_24h)
                            px_diff_from_chart = bool(
                                pts_24h
                                and sym_px_cur > 0
                                and not _prices_diverged(sym_px_cur, pts_24h[-1][1], threshold=eq_div_thr)
                                and abs(round(sym_px_cur, 4) - round(pts_24h[-1][1], 4)) > 1e-4
                            )
                            px_diverged_05pct = bool(
                                pts_24h and sym_px_cur > 0 and abs(sym_px_cur - pts_24h[-1][1]) / max(sym_px_cur, pts_24h[-1][1]) > 0.005
                            )
                            if px_diff_from_chart and rth_bar_fresh and not px_diverged_05pct:
                                pts_24h = list(pts_24h) + [[int(pts_24h[-1][0] + 1000), round(sym_px_cur, 4)]]
                                pts_24h_to_save = list(pts_24h)
                                post_rth_series_replaced = False
                            elif px_diff_from_chart and px_diverged_05pct and len(pts_24h) >= 2:
                                post_rth_series_replaced = True
                                last_bar_tail = list(pts_24h[-1][2:]) if isinstance(pts_24h[-1], (list, tuple)) and len(pts_24h[-1]) >= 3 else []
                                pts_24h = list(pts_24h[:-1]) + [[pts_24h[-1][0], round(sym_px_cur, 4), *last_bar_tail]]
                            else:
                                post_rth_series_replaced = False
                            if post_rth_series_replaced or (eq_cache_stale and len(pts_24h) >= 5):
                                for k_ov in _SHORT_SLOPE_OVERRIDE_KEYS:
                                    pt_map[sym].pop(k_ov, None)
                            pts_24h_for_m24 = pts_24h
                            if px_diff_from_chart and not rth_bar_fresh and not px_diverged_05pct and len(pts_24h) >= 2:
                                last_bar_tail = list(pts_24h[-1][2:]) if isinstance(pts_24h[-1], (list, tuple)) and len(pts_24h[-1]) >= 3 else []
                                pts_24h_for_m24 = list(pts_24h[:-1]) + [[pts_24h[-1][0], round(sym_px_cur, 4), *last_bar_tail]]
                            if m24_new := (compute_24h_price_metrics(pts_24h_for_m24) or {}):
                                m24 = m24_new
                                if not rth_bar_fresh and px_diff_from_chart and (rth_vp_chart := compute_volume_profile_metrics(pts_24h_to_save)):
                                    for k_vp_clr in _RTH_VP_ANCHOR_KEYS:
                                        m24.pop(k_vp_clr, None)
                                    for k_vp in _RTH_VP_ANCHOR_KEYS:
                                        if k_vp in rth_vp_chart:
                                            m24[k_vp] = rth_vp_chart[k_vp]
                                matched_series_24h = pts_24h
                                post_rth_px_override = None
                                if not rth_bar_fresh or px_diverged_05pct:
                                    for k_stale in _STALE_SHORT_24H_KEYS:
                                        m24.pop(k_stale, None)
                                    if px_diff_from_chart:
                                        post_rth_px_override = round(sym_px_cur, 4)
                                        if "chg_24h_pct" in m24 and pts_24h[0][1] > 0:
                                            m24["chg_24h_pct"] = pct_delta(post_rth_px_override, pts_24h[0][1])
                            vals_5d = [pt[1] for pt in pts_5d]
                            if post_rth_px_override is not None:
                                vals_5d.append(post_rth_px_override)
                            elif rth_bar_fresh and px_diff_from_chart:
                                vals_5d.append(round(sym_px_cur, 4))
                            if len(vals_5d) >= 2:
                                for k_7d_opt in ("major_swing_low_px", "rally_from_major_low_pct", "recent_swings_7d", "avg_swing_amp_7d_pct"):
                                    pt_map[sym].pop(k_7d_opt, None)
                                pt_map[sym].update(compute_7d_range_metrics(vals_5d, decimals=4))
                                active_7d_vals = list(vals_5d)
                            if persist:
                                _persist_refreshed_equity_chart_cache(
                                    cachedb_dir, sym, p24_eq_path, pts_24h_to_save, pts_24h_bars, pts_5d, raw_matched_series_24h, rth_bar_fresh, eff_now
                                )
                    except Exception:
                        pass
            if m24:
                if is_eq_sym and not any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct")):
                    for k_stale_15m in ("recent_tape_15m", "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct"):
                        m24.pop(k_stale_15m, None)
                _apply_24h_series_enrichment(sym, pt_map[sym], m24, matched_series_24h, btc_series_24h)
                session_policy.prune_stale_short_metrics(
                    pt_map[sym],
                    m24,
                    matched_series_24h,
                    eff_now,
                    post_rth_series_replaced=post_rth_series_replaced,
                )
                if post_rth_px_override is not None:
                    recalculate_live_price_geometry(pt_map[sym], post_rth_px_override)

            if is_eq_sym or "USD" in compatible_quote_suffixes(sym):
                ensure_fresh_symbol_microstructure(
                    cachedb_dir,
                    sym.upper(),
                    now_ts=eff_now,
                    max_age_ob_sec=180.0,
                    max_age_whale_sec=360.0,
                    include_derivatives=True,
                    allow_network=allow_network,
                )
            if not (m_info := load_symbol_microstructure(cachedb_dir, sym, _rj)):
                continue
            is_usd_sym = "USD" in compatible_quote_suffixes(sym)
            sym_px_m = safe_float((pt_map.get(sym) or {}).get("price"), 0.0)
            ob_mid_m = safe_pos_float(m_info.get("ob_mid_price"), default=None)
            ob_ts_m = safe_pos_float(m_info.pop("_ob_ts", None), default=0.0) or 0.0
            src_cand_u = str(m_info.get("_source_cand") or sym).strip().upper()
            is_exact_ob_sym = clean_symbol_key(src_cand_u) == clean_symbol_key(sym)
            cross_venue_collision = (not is_exact_ob_sym) and (src_cand_u in inst_map) and (sym in inst_map)
            if not is_eq_sym and is_usd_sym and isinstance(pt_map.get(sym), dict) and ob_mid_m is not None and not cross_venue_collision:
                if (
                    sym_px_m <= 0.0
                    or (abs(sym_px_m - ob_mid_m) / max(sym_px_m, ob_mid_m) <= 0.01)
                    or is_ob_mid_fresher_than_series(
                        sym_px_m, ob_mid_m, ob_ts_m, eff_now, raw_s24_last_pt, is_exact_ob_sym=is_exact_ob_sym
                    )
                ):
                    pt_map[sym]["price"] = ob_mid_m
                    sym_px_m = ob_mid_m
                    if (
                        active_7d_vals
                        and len(active_7d_vals) >= 2
                        and safe_float(active_7d_vals[0], 0.0) > 0.0
                        and not _prices_diverged(ob_mid_m, safe_float(active_7d_vals[-1], 0.0))
                    ):
                        pt_map[sym]["chg_7d_pct"] = pct_delta(ob_mid_m, safe_float(active_7d_vals[0], 0.0))
                    recalculate_live_price_geometry(pt_map[sym], ob_mid_m)
                elif (
                    not _prices_diverged(sym_px_m, ob_mid_m)
                    and is_exact_ob_sym
                    and str(sym).strip().upper().endswith("USD")
                    and not str(sym).strip().upper().endswith(("FDUSD", "TUSD", "BUSD"))
                    and (
                        bool(matched_series_24h)
                        or any(
                            pk != sym
                            and has_equity_suffix(pk) == is_eq_sym
                            and canonical_base_asset(pk) == canonical_base_asset(sym)
                            and (p_sib := safe_pos_float((pv if isinstance(pv, dict) else {}).get("price"), default=None)) is not None
                            and abs(sym_px_m - p_sib) / max(sym_px_m, p_sib) <= 0.01
                            for pk, pv in pt_map.items()
                        )
                    )
                ):
                    sym_px_m = pt_map[sym]["price"] = rescale_orderbook_walls_to_price(m_info, sym_px_m, ob_mid_m)
            elif is_eq_sym and align_equity_orderbook_to_spot(m_info, sym_px_m):
                pass
            elif not is_usd_sym and (sym_px_m <= 0.0 or _prices_diverged(sym_px_m, ob_mid_m)):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    m_info.pop(k_strip, None)
            enrich_microstructure_cvd(m_info, pt_map.get(sym))
            m_info.update(compute_liquidation_clusters(pt_map.get(sym), m_info))
            if m_info.get("spread_bps") is not None and isinstance(pt_map.get(sym), dict):
                pt_map[sym]["spread_bps"] = m_info["spread_bps"]
            m_map[sym] = m_info
        for msym in discover_reference_microstructure_symbols(cachedb_dir, m_map, _rj):
            if allow_network and "USD" in compatible_quote_suffixes(msym) and _rj in (_default_read_json_cached, read_json_recovering, read_json_cached):
                ensure_fresh_symbol_microstructure(
                    cachedb_dir,
                    msym.upper(),
                    now_ts=eff_now,
                    max_age_ob_sec=180.0,
                    max_age_whale_sec=360.0,
                    include_derivatives=True,
                    allow_network=allow_network,
                )
            if not (m_info_ref := load_symbol_microstructure(cachedb_dir, msym, _rj)):
                continue
            ob_ref_ts = safe_pos_float(m_info_ref.pop("_ob_ts", None), default=0.0) or 0.0
            msym_quotes = compatible_quote_suffixes(msym)
            msym_is_eq = is_equity_symbol(msym)
            p_ref = _first_variant_dict(
                pt_map,
                [
                    c
                    for c in cache_symbol_variants(msym, include_original=True, include_base="USD" in msym_quotes, quotes=msym_quotes)
                    if is_equity_symbol(c) == msym_is_eq
                ],
            )
            if p_ref is not None and msym not in pt_map:
                ob_mid_sib = safe_pos_float(m_info_ref.get("ob_mid_price"), default=None)
                p_ref_px = safe_pos_float(p_ref.get("price"), default=0.0) or 0.0
                if msym.upper().endswith("USD") and not msym.upper().endswith(("FDUSD", "TUSD", "BUSD")) and ob_mid_sib is not None:
                    sib_live_px = extract_current_price_item(casefold_get(cp_map, msym))
                    if (not sib_live_px or sib_live_px <= 0.0) and cachedb_dir:
                        _ensure_cacheprice_classes_bound()
                        if isinstance(l24_s := _rj(os.path.join(cachedb_dir, CacheShortTrendManager.live_window_relpath_for_symbol(msym.upper()))), dict):
                            w_src = l24_s.get("windows") if isinstance(l24_s.get("windows"), dict) else l24_s
                            sib_live_px = safe_pos_float(
                                ((w_src.get("222") or w_src.get("900") or {}) if isinstance(w_src, dict) else {}).get("current_price"),
                                default=None,
                            )
                        if not sib_live_px or sib_live_px <= 0.0:
                            if ob_ref_ts <= 0.0:
                                ob_ref_d = _rj(os.path.join(cachedb_dir, f"{CACHE_ORDERBOOK_DEPTH_PREFIX}{msym.upper()}.json"))
                                if not isinstance(ob_ref_d, dict):
                                    ob_ref_d = _rj(os.path.join(cachedb_dir, "telemetry", f"{CACHE_ORDERBOOK_DEPTH_PREFIX}{msym.upper()}.json"))
                                ob_ref_ts = normalize_epoch_sec(
                                    safe_pos_float((ob_ref_d if isinstance(ob_ref_d, dict) else {}).get("ts"), default=0.0) or 0.0,
                                    default=0.0,
                                )
                            if ob_ref_ts > 0.0 and (eff_now - ob_ref_ts) > 120.0:
                                sib_live_px = p_ref_px if p_ref_px > 0.0 else None
                    if (
                        sib_live_px
                        and sib_live_px > 0.0
                        and abs(sib_live_px - ob_mid_sib) / max(sib_live_px, ob_mid_sib) > 0.01
                        and not _prices_diverged(sib_live_px, ob_mid_sib)
                        and not (
                            p_ref_px > 0.0
                            and abs(p_ref_px - ob_mid_sib) / max(p_ref_px, ob_mid_sib) <= 0.005
                            and abs(sib_live_px - p_ref_px) / max(sib_live_px, p_ref_px) > 0.01
                        )
                    ):
                        ob_mid_sib = rescale_orderbook_walls_to_price(m_info_ref, sib_live_px, ob_mid_sib)
                if ob_mid_sib is not None and not _prices_diverged(ob_mid_sib, safe_float(p_ref.get("price"), 0.0)):
                    p_sib = dict(p_ref)
                    p_sib["price"] = ob_mid_sib
                    recalculate_live_price_geometry(p_sib, ob_mid_sib)
                    if m_info_ref.get("spread_bps") is not None:
                        p_sib["spread_bps"] = m_info_ref["spread_bps"]
                    pt_map[msym] = p_sib
                    p_ref = p_sib
            elif p_ref is None:
                ref_px = safe_pos_float(m_info_ref.get("ob_mid_price"), default=None)
                ser_24h, vals_7d = load_symbol_price_series(cachedb_dir, msym, current_price_items=cp_map)
                if not ser_24h and allow_network and msym.endswith("USD") and not msym.endswith(("FDUSD", "TUSD", "BUSD")) and is_live_network_backfill_enabled():
                    try:
                        krk_ohlc_fn = _resolve_fallback_ohlc_fn(msym, inst_map)
                        if callable(krk_ohlc_fn):
                            ser_24h = _load_stitched_24h_ohlc_points(krk_ohlc_fn, msym, eff_now - 86400.0)
                            if ref_px is not None and ser_24h and abs(ref_px - float(ser_24h[-1][1])) > 1e-9 and not _prices_diverged(ref_px, float(ser_24h[-1][1])):
                                ser_24h.append([int(eff_now * 1000), ref_px])
                            if len(vals_7d) < 2:
                                vals_7d = _load_stitched_7d_ohlc_closes(krk_ohlc_fn, msym, eff_now, None, ref_px or 0.0)
                    except Exception:
                        pass
                if ref_px is None and ser_24h:
                    ref_px = safe_pos_float(ser_24h[-1][1], default=None)
                if ref_px is not None:
                    p_ref = {"price": ref_px, "trend": 0, "growth_coefficient": 0.0}
                    if m_info_ref.get("spread_bps") is not None:
                        p_ref["spread_bps"] = m_info_ref["spread_bps"]
                    if ser_24h and (m24_ref := compute_24h_price_metrics(ser_24h)):
                        _apply_24h_series_enrichment(msym, p_ref, m24_ref, ser_24h, btc_series_24h)
                    if vals_7d and len(vals_7d) >= 2:
                        if not _prices_diverged(ref_px, safe_float(vals_7d[-1], 0.0)) and abs(ref_px - safe_float(vals_7d[-1], 0.0)) > 1e-9:
                            vals_7d = list(vals_7d) + [ref_px]
                        p_ref.update(compute_7d_range_metrics(vals_7d))
                    pt_map[msym] = p_ref
            enrich_microstructure_cvd(m_info_ref, p_ref)
            m_info_ref.update(compute_liquidation_clusters(p_ref, m_info_ref))
            m_map[msym] = m_info_ref
        return pt_map, m_map


for _micro_collector_cls in (
    ("price_series_loader", CompositeSymbolPriceSeriesLoader),
    ("equity_yahoo_5d_chart", EquityYahoo5dChartCollector),
    (SymbolMicrostructureSnapshotCollector.COLLECTOR_ID, SymbolMicrostructureSnapshotCollector),
    (Fleet24hAndMicrostructureEnricher.COLLECTOR_ID, Fleet24hAndMicrostructureEnricher),
):
    TelemetryCollectorFactory.register(_micro_collector_cls[0], _micro_collector_cls[1])


def enrich_24h_and_microstructure(
    cache_dir: str,
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    cur_p_items: Dict[str, Any],
    inst_by_sym: Dict[str, Any],
    now_ts: float,
    read_json_fn: Any = None,
    *,
    allow_network: bool = True,
    persist: bool = True,
) -> None:
    """Enrich prices_and_trends with 24h/7d metrics, fallback OHLC/Yahoo chart series, and microstructure."""
    TelemetryCollectorFactory.get_fleet_enricher(start_sync=allow_network).collect(
        cache_dir,
        "BTCUSDC",
        workspace_dir=workspace_dir,
        prices_and_trends=prices_and_trends,
        microstructure=microstructure,
        cur_p_items=cur_p_items,
        inst_by_sym=inst_by_sym,
        now_ts=now_ts,
        read_json_fn=read_json_fn,
        allow_network=allow_network,
        persist=persist,
    )


__all__ = [
    "CompositeSymbolPriceSeriesLoader",
    "EquityYahoo5dChartCollector",
    "Fleet24hAndMicrostructureEnricher",
    "MicrostructureCacheCollectorBase",
    "PriceSeriesCacheCollectorBase",
    "SymbolMicrostructureSnapshotCollector",
    "_iter_t212_state_files",
    "_refresh_stale_t212_cached_positions",
    "discover_reference_microstructure_symbols",
    "enrich_24h_and_microstructure",
    "ensure_fresh_symbol_microstructure",
    "load_microstructure_cycle_history",
    "load_symbol_microstructure",
    "load_symbol_price_series",
    "resolve_cached_btc_24h_change_pct",
    "save_microstructure_cycle_history",
]
