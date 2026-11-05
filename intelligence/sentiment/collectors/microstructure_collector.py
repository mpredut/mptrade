"""Spot/perpetual microstructure, 24h/7d price series loader, and reference symbol discovery collector."""
from __future__ import annotations

import glob as _glob
import logging
import os
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from botcore import extract_current_price_item, load_local_module
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    legacy_filenames_for,
    normalize_venue_name,
    resolve_cache_dir_path,
    split_symbol_assets,
)
from intelligence.external.collectors.orderbook_depth_collector import round_spread_bps
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    lookup_symbol_cache_items,
    normalize_epoch_sec,
    obj_attr,
    resolve_now_ts,
    resolve_read_json_cached,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _MICRO_PRICE_ANCHOR_KEYS,
    _blend_structural_liq_anchor,
    _first_variant_dict,
    _infer_symbol_venue_and_fees_bps,
    _resolve_non_static_prev_metric,
    align_equity_orderbook_to_spot,
    compute_cross_venue_arb_metrics,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    reconcile_sibling_flow_and_liq_regimes,
    update_microstructure_cycle_deltas,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    _VENUE_DEFAULT_FEE_SPECS,
    extract_symbol_recent_fills,
    resolve_symbol_price,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _WINDOW_TREND_SPECS,
    _apply_24h_series_enrichment,
    _auto_px_decimals,
    _build_vp_position_dict,
    _compute_horizon_range_dict,
    _extract_ohlc_series_points,
    _load_stitched_24h_ohlc_points,
    _parse_ts_px_row,
    _prices_diverged,
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_7d_range_metrics,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
    compute_swing_wave_metrics,
    compute_volume_profile_metrics,
    pct_delta,
    range_position_pct,
    recalculate_live_price_geometry,
    round_opt_px,
)
from instrument_registry import casefold_get, is_equity_symbol
from state_io import (
    read_json_dict_recovering as _default_read_json_cached,
    read_json_recovering,
    read_tail_jsonl_records,
    resolve_cachedb_filepath,
)

read_json_cached = _default_read_json_cached
_LAST_7D_SPAN_MS_BY_SYM: Dict[str, float] = {}
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
        if "PYTEST_CURRENT_TEST" in os.environ:
            fn = _get_raw_fn()
            return fn(s_arg, interval_min) if callable(fn) else None
        ck = (prov_key, str(s_arg or "").strip().upper(), int(interval_min))
        now_s = _time.time()
        hit = _FALLBACK_OHLC_SERIES_CACHE.get(ck)
        if hit is not None and 0.0 <= (now_s - hit[0]) < ttl_sec:
            return hit[1]
        fn = _get_raw_fn()
        if not callable(fn):
            return None
        res = fn(s_arg, interval_min)
        if res is not None and getattr(res, "closes", None):
            if len(_FALLBACK_OHLC_SERIES_CACHE) > 128:
                _FALLBACK_OHLC_SERIES_CACHE.clear()
            _FALLBACK_OHLC_SERIES_CACHE[ck] = (now_s, res)
        return res

    return _cached_ohlc


def _resolve_fallback_ohlc_fn(sym: str, inst_by_sym: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    sym_u = str(sym or "").strip().upper()
    inst_map = inst_by_sym or {}
    inst_obj = (
        (inst_map.get(f"binance:{sym_u}") if sym_u != "HYPEUSDC" else None)
        or inst_map.get(sym)
        or inst_map.get(sym_u)
    )
    ohlc_fn = getattr(getattr(inst_obj, "_provider", None), "ohlc_series", None)
    if callable(ohlc_fn) or "PYTEST_CURRENT_TEST" in os.environ:
        return ohlc_fn if callable(ohlc_fn) else None
    try:
        prov_key = ""
        sym_parts = split_symbol_assets(sym_u) if sym_u and not is_equity_symbol(sym_u) else None
        sym_q = sym_parts[1] if sym_parts is not None else ""
        if sym_u == "HYPEUSDC":
            prov_key = "hyperliquid"
            if prov_key not in _FALLBACK_PROVIDER_OHLC_FNS:
                _FALLBACK_PROVIDER_OHLC_FNS[prov_key] = _wrap_cached_ohlc_fn(
                    prov_key,
                    lambda: __import__("providers.hyperliquid_provider", fromlist=["HyperliquidProvider"]).HyperliquidProvider().ohlc_series,
                    is_factory=True,
                )
        elif sym_q in ("USDC", "USDT", "FDUSD", "TUSD", "BUSD") or sym_u.endswith(("USDC", "USDT", "FDUSD", "TUSD", "BUSD")):
            prov_key = "binance"
            if prov_key not in _FALLBACK_PROVIDER_OHLC_FNS:
                _FALLBACK_PROVIDER_OHLC_FNS[prov_key] = _wrap_cached_ohlc_fn(
                    prov_key,
                    lambda: getattr(
                        __import__("providers.market_api", fromlist=["api"]).api.provider_by_name("binance"),
                        "ohlc_series",
                        None,
                    ),
                    is_factory=True,
                )
        elif sym_q == "USD" or (sym_u.endswith("USD") and not is_equity_symbol(sym_u) and not sym_u.endswith(("FDUSD", "TUSD", "BUSD"))):
            prov_key = "kraken"
            if prov_key not in _FALLBACK_PROVIDER_OHLC_FNS:
                _FALLBACK_PROVIDER_OHLC_FNS[prov_key] = _wrap_cached_ohlc_fn(
                    prov_key,
                    lambda: __import__("providers.kraken_provider", fromlist=["KrakenProvider"]).KrakenProvider().ohlc_series,
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
            for ex_px in (safe_pos_float(m24.get("low_24h"), default=None), safe_pos_float(m24.get("high_24h"), default=None)):
                if ex_px is not None and not _prices_diverged(ex_px, vals_7d[-1]):
                    vals_7d.insert(-1, ex_px)
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


def load_symbol_price_series(
    cachedb_dir: str,
    symbol: str,
    current_price_items: Optional[Dict[str, Any]] = None,
    *,
    include_7d: bool = True,
    quotes: Optional[tuple[str, ...]] = None,
) -> Tuple[list, list]:
    """Load and merge high-density 24h & 7d price series from cache_24_price, cache_365_price, cache_sparse_price, and cache_current_price."""
    sym_u = str(symbol).strip().upper()
    is_eq_sym = is_equity_symbol(sym_u)
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
    series_24h: list = []
    vals_7d: list = []

    def _cand_paths(*basenames: str) -> list[str]:
        return [
            p
            for bn in basenames
            for full_p in (str(resolve_cachedb_filepath(os.path.join(cachedb_dir, bn))),)
            for p in (full_p, *legacy_filenames_for(full_p))
        ]

    def _prepend_older_non_diverged_points(cur_ser: list, cand_ser: list) -> list:
        if not cur_ser or not cand_ser:
            return cur_ser
        cur_p0 = _parse_ts_px_row(cur_ser[0])
        sib_p0, sib_p1 = _parse_ts_px_row(cand_ser[0]), _parse_ts_px_row(cand_ser[-1])
        if cur_p0 and sib_p0 and sib_p1 and sib_p0[0] < cur_p0[0] and not _prices_diverged(sib_p1[1], cur_p0[1]):
            older_pts = [r for r in cand_ser if (pt := _parse_ts_px_row(r)) is not None and pt[0] < cur_p0[0]]
            if older_pts:
                return older_pts + cur_ser
        return cur_ser

    def _sort_ts_rows(rows: list) -> list:
        if len(rows) < 2:
            return rows
        prev_t = -1.0
        for r in rows:
            pt = _parse_ts_px_row(r)
            if pt is not None:
                if pt[0] < prev_t:
                    keyed = []
                    for r_item in rows:
                        p = _parse_ts_px_row(r_item)
                        if p is not None:
                            keyed.append((p[0], r_item))
                    keyed.sort(key=lambda x: x[0])
                    return [r_item for _, r_item in keyed]
                prev_t = pt[0]
        return rows

    # 1. Try primary 24h dense cache (cache_24_price_<SYM>.json and legacy variants)
    _rj_cand = _resolve_read_json_cached()
    _rj = read_json_recovering if _rj_cand is _default_read_json_cached else _rj_cand
    sym_u = str(symbol).strip().upper()
    _memo_key: Optional[Tuple[Any, ...]] = None
    _memo_file_sig: Tuple[Any, ...] = ()
    _memo_cur_sig: Tuple[Any, ...] = ()
    if _rj_cand is _default_read_json_cached:
        try:
            _memo_key = (os.path.abspath(str(cachedb_dir)), sym_u, bool(include_7d), tuple(eff_quotes))
            _stat_list = []
            for _sp in _cand_paths(
                *(f"cache_24_price_{c}.json" for c in cands),
                *(f"cache_sparse_price_{c}.jsonl" for c in cands),
                *(f"cache_365_price_{c}.jsonl" for c in cands),
                *(f"cache_365_price_{c}.json" for c in cands),
            ):
                for _cand_sp in (_sp, _sp + ".previous") if "cache_24_price_" in _sp else (_sp,):
                    try:
                        _st = os.stat(_cand_sp)
                        _stat_list.append((_cand_sp, _st.st_mtime_ns, _st.st_size))
                    except OSError:
                        pass
            if not _stat_list:
                for _w_sp in _cand_paths("cache_watcher_all_price.json"):
                    try:
                        _st = os.stat(_w_sp)
                        _stat_list.append((_w_sp, _st.st_mtime_ns, _st.st_size))
                    except OSError:
                        pass
            _memo_file_sig = tuple(_stat_list)
            if isinstance(current_price_items, dict) and current_price_items:
                _cur_list = []
                for _c in cands:
                    _ci = casefold_get(current_price_items, _c)
                    if isinstance(_ci, list) and _ci:
                        _cur_list.append((_c, len(_ci), repr(_ci[-1])))
                    elif _ci is not None:
                        _cur_list.append((_c, extract_current_price_item(_ci)))
                _memo_cur_sig = tuple(_cur_list)
            _cached_entry = _PRICE_SERIES_MEMO.get(_memo_key)
            if (
                _cached_entry is not None
                and _cached_entry[0] == _memo_file_sig
                and _cached_entry[1] == _memo_cur_sig
            ):
                _LAST_7D_SPAN_MS_BY_SYM[sym_u] = _cached_entry[4]
                return [list(pt) for pt in _cached_entry[2]], list(_cached_entry[3])
        except Exception:
            _memo_key = None
    p0: Optional[Tuple[float, float]] = None
    p1: Optional[Tuple[float, float]] = None
    span_24h_ms = 0.0
    is_eq_sym_load = str(symbol).strip().upper().endswith(("_EQ", "_US_EQ"))

    def _strip_off_rth_flat_poller_tail(rows: list) -> list:
        if not is_eq_sym_load or len(rows) < 3:
            return rows
        last_ohlc_idx = max(
            (i for i, r in enumerate(rows) if isinstance(r, (list, tuple)) and len(r) >= 3),
            default=-1,
        )
        if 1 <= last_ohlc_idx < len(rows) - 1:
            last_ohlc_pt = _parse_ts_px_row(rows[last_ohlc_idx])
            tail_pts = [pt for r in rows[last_ohlc_idx + 1 :] if (pt := _parse_ts_px_row(r)) is not None]
            if (
                last_ohlc_pt is not None
                and tail_pts
                and (tail_pts[-1][0] - last_ohlc_pt[0]) > 20 * 60 * 1000
                and max(p[1] for p in tail_pts) == min(p[1] for p in tail_pts)
            ):
                return rows[: last_ohlc_idx + 1]
        last_pt = _parse_ts_px_row(rows[-1])
        if last_pt is not None and isinstance(rows[-1], (list, tuple)) and len(rows[-1]) < 3:
            flat_start = len(rows) - 1
            while flat_start > 0:
                prev_r = rows[flat_start - 1]
                prev_pt = _parse_ts_px_row(prev_r)
                if (
                    prev_pt is not None
                    and isinstance(prev_r, (list, tuple))
                    and len(prev_r) < 3
                    and abs(prev_pt[1] - last_pt[1]) <= 1e-6
                ):
                    flat_start -= 1
                else:
                    break
            anchor_idx = flat_start
            if anchor_idx > 0:
                before_pt = _parse_ts_px_row(rows[anchor_idx - 1])
                if before_pt is not None and abs(before_pt[1] - last_pt[1]) <= 1e-6:
                    anchor_idx -= 1
            anchor_pt = _parse_ts_px_row(rows[anchor_idx]) if 1 <= anchor_idx < len(rows) - 1 else None
            if anchor_pt is not None and (last_pt[0] - anchor_pt[0]) > 20 * 60 * 1000:
                return rows[: anchor_idx + 1]
        return rows

    for cand in cands:
        for p24_path in _cand_paths(f"cache_24_price_{cand}.json"):
            if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                continue
            ser = _strip_off_rth_flat_poller_tail(_sort_ts_rows(_extract_cached_items_series(_rj(p24_path), cands)))
            if len(ser) >= 2:
                series_24h = list(ser) if not series_24h else _prepend_older_non_diverged_points(series_24h, ser)
                p0, p1 = _parse_ts_px_row(series_24h[0]), _parse_ts_px_row(series_24h[-1])
                span_24h_ms = max(0.0, p1[0] - p0[0]) if (p0 and p1) else 0.0
                break
        if series_24h and span_24h_ms >= 3600000.0 * 18:
            break

    # 2. Supplement 24h (when < 18h span) and extract 7d history from cache_sparse_price_<SYM> / cache_365_price_<SYM>
    need_24h_supplement = span_24h_ms < 3600000.0 * 18
    archive_pts: list = []
    sparse_7d_pts: list = []
    cands_upper = {str(c).strip().upper() for c in cands if isinstance(c, str) and str(c).strip()}
    cands_upper_tup = tuple(sorted(cands_upper))

    def _read_arch_file(arch_path: str, max_b: int = 1024 * 1024) -> list:
        if not os.path.exists(arch_path):
            return []
        try:
            if arch_path.endswith(".json"):
                aser = _sort_ts_rows(_extract_cached_items_series(_rj(arch_path), cands))
                return list(aser) if len(aser) >= 2 else []
            _arch_key: Optional[Tuple[str, int, int, int, Tuple[str, ...]]] = None
            if _rj_cand is _default_read_json_cached:
                try:
                    _ast = os.stat(arch_path)
                    _arch_key = (os.path.abspath(arch_path), _ast.st_mtime_ns, _ast.st_size, max_b, cands_upper_tup)
                    if (_hit_arch := _ARCH_FILE_MEMO.get(_arch_key)) is not None:
                        return [list(r) for r in _hit_arch]
                except OSError:
                    _arch_key = None
            parsed_rows = []
            for rec in read_tail_jsonl_records(arch_path, limit=None, max_bytes=max_b):
                if isinstance(rec, dict):
                    rec_s = rec.get("s")
                    if isinstance(rec_s, str) and rec_s.strip() and cands_upper and rec_s.strip().upper() not in cands_upper:
                        continue
                    item = rec.get("i")
                else:
                    item = rec if isinstance(rec, list) else None
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    parsed_rows.append([item[0], item[1]])
            parsed_rows = _sort_ts_rows(parsed_rows)
            res_arch = parsed_rows if len(parsed_rows) >= 2 else []
            if _arch_key is not None and res_arch:
                if len(_ARCH_FILE_MEMO) > 128:
                    _ARCH_FILE_MEMO.clear()
                _ARCH_FILE_MEMO[_arch_key] = [list(r) for r in res_arch]
            return res_arch
        except Exception:
            return []

    is_eq_sym_load = str(symbol).strip().upper().endswith(("_EQ", "_US_EQ"))
    for cand in (cands if (include_7d or need_24h_supplement or is_eq_sym_load) else ()):
        primary_names = (
            (f"cache_sparse_price_{cand}.jsonl", f"cache_365_price_{cand}.jsonl", f"cache_365_price_{cand}.json")
            if not need_24h_supplement
            else (f"cache_365_price_{cand}.jsonl", f"cache_365_price_{cand}.json", f"cache_sparse_price_{cand}.jsonl")
        )
        cand_arch: list = []
        for arch_path in _cand_paths(*primary_names):
            if rows := _read_arch_file(arch_path):
                cand_arch = rows
                break
        if cand_arch:
            archive_pts = cand_arch if not archive_pts else _prepend_older_non_diverged_points(archive_pts, cand_arch)
            a0, a1 = _parse_ts_px_row(archive_pts[0]), _parse_ts_px_row(archive_pts[-1])
            arch_span_ms = (a1[0] - a0[0]) if (a0 and a1) else 0.0
            if include_7d and arch_span_ms < 5.0 * 86400.0 * 1000.0 and not sparse_7d_pts:
                for sp_path in _cand_paths(
                    f"cache_sparse_price_{cand}.jsonl",
                    f"cache_365_price_{cand}.jsonl",
                    f"cache_365_price_{cand}.json",
                ):
                    if sp_rows := _read_arch_file(sp_path, max_b=512 * 1024):
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
        is_eq_w = str(symbol).strip().upper().endswith(("_EQ", "_US_EQ"))
        for w_path in _cand_paths("cache_watcher_all_price.json"):
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
            w_rows = _sort_ts_rows(w_raw or [])
            if len(w_rows) >= 10 and (w_last := _parse_ts_px_row(w_rows[-1])) is not None:
                w_24 = [pt for r in w_rows if (pt := _parse_ts_px_row(r)) is not None and pt[0] >= w_last[0] - 86400.0 * 1000.0]
                if len(w_24) >= 10 and (w_24[-1][0] - w_24[0][0]) >= 3600.0 * 1000.0:
                    if is_eq_w:
                        w_parsed = [[pt[0], round(pt[1], 4)] for r in w_rows if (pt := _parse_ts_px_row(r)) is not None]
                        raw_last_ts = w_parsed[-1][0]
                        watcher_7d_pts = [pt for pt in w_parsed if pt[0] >= raw_last_ts - 7.05 * 86400.0 * 1000.0]
                        runs: List[List[List[float]]] = []
                        for pt in w_parsed:
                            if not runs or abs(pt[1] - runs[-1][-1][1]) > 1e-9:
                                runs.append([pt])
                            else:
                                runs[-1].append(pt)
                        last_active_ts = runs[-1][0][0] if len(runs[-1]) >= 4 else raw_last_ts
                        watcher_cutoff_24h = (
                            raw_last_ts - 86400.0 * 1000.0
                            if (raw_last_ts - last_active_ts) <= 6.0 * 3600.0 * 1000.0
                            else last_active_ts - 86400.0 * 1000.0
                        )
                        anchor_targets = (
                            watcher_cutoff_24h,
                            last_active_ts - 86400.0 * 1000.0,
                            last_active_ts - 43200.0 * 1000.0,
                            last_active_ts - 21600.0 * 1000.0,
                        )
                        collapsed_w: list = []
                        for r_idx, r_pts in enumerate(runs):
                            if len(r_pts) < 4:
                                collapsed_w.extend(r_pts)
                            elif r_idx == len(runs) - 1:
                                collapsed_w.append(r_pts[0])
                            else:
                                keep_idx = {0, len(r_pts) - 1}
                                for t_anc in anchor_targets:
                                    if r_pts[0][0] <= t_anc <= r_pts[-1][0]:
                                        best_i = min(range(len(r_pts)), key=lambda i: abs(r_pts[i][0] - t_anc))
                                        keep_idx.add(best_i)
                                for k_i in sorted(keep_idx):
                                    collapsed_w.append(r_pts[k_i])
                        archive_pts = collapsed_w
                    else:
                        archive_pts = list(w_rows)
                    break

    arch_7d_end_ts = 0.0
    arch_7d_span_ms = 0.0
    if archive_pts:
        try:
            if (last_pt := _parse_ts_px_row(archive_pts[-1])) is not None:
                if (
                    str(symbol).strip().upper().endswith(("_EQ", "_US_EQ"))
                    and series_24h
                    and isinstance(series_24h[-1], (list, tuple))
                    and len(series_24h[-1]) >= 3
                    and isinstance(archive_pts[-1], (list, tuple))
                    and len(archive_pts[-1]) >= 3
                    and (s24_end_pt := _parse_ts_px_row(series_24h[-1])) is not None
                    and s24_end_pt[0] == last_pt[0]
                    and safe_nonneg_float(series_24h[-1][2], -1.0) == safe_nonneg_float(archive_pts[-1][2], -2.0)
                    and abs(s24_end_pt[1] - last_pt[1]) > 1e-4
                ):
                    series_24h = list(series_24h[:-1]) + [[series_24h[-1][0], last_pt[1], *series_24h[-1][2:]]]
                cutoff_24h = watcher_cutoff_24h if watcher_cutoff_24h is not None else (last_pt[0] - 86400.0 * 1000.0)
                cutoff_7d = last_pt[0] - 7.0 * 86400.0 * 1000.0
                pts_24, pts_7d_vals, pts_7d_ts = [], [], []
                parsed_arch = [
                    (
                        pt[0],
                        pt[1],
                        safe_pos_float(row[2], default=None) if isinstance(row, (list, tuple)) and len(row) >= 3 else None,
                        row[0] if isinstance(row, (list, tuple)) and row else pt[0],
                    )
                    for row in archive_pts
                    if (pt := _parse_ts_px_row(row)) is not None
                ]
                has_arch_vol = any(v is not None for _, _, v, _ in parsed_arch)
                clean_arch = []
                for idx_a, item_a in enumerate(parsed_arch):
                    if has_arch_vol and 0 < idx_a < len(parsed_arch) - 1 and item_a[2] is None:
                        prev_a = clean_arch[-1] if clean_arch else parsed_arch[idx_a - 1]
                        next_a = parsed_arch[idx_a + 1]
                        if prev_a[2] is not None or next_a[2] is not None:
                            ref_mid = 0.5 * (prev_a[1] + next_a[1])
                            if (
                                ref_mid > 0.0
                                and abs(prev_a[1] - next_a[1]) / ref_mid <= 0.025
                                and abs(item_a[1] - ref_mid) / ref_mid > 0.025
                            ):
                                continue
                    clean_arch.append(item_a)
                older_24_arch = []
                for pt_t, pt_p, pt_v, raw_t0 in clean_arch:
                    if pt_t >= cutoff_7d:
                        pts_7d_vals.append(pt_p)
                        pts_7d_ts.append(pt_t)
                    if need_24h_supplement:
                        row_24 = [raw_t0, pt_p, pt_v] if pt_v is not None else [raw_t0, pt_p]
                        if pt_t >= cutoff_24h:
                            pts_24.append(row_24)
                        elif pt_t >= last_pt[0] - 36.0 * 3600.0 * 1000.0:
                            older_24_arch.append(row_24)
                if (
                    need_24h_supplement
                    and older_24_arch
                    and (
                        not pts_24
                        or (
                            (p24_0 := _parse_ts_px_row(pts_24[0])) is not None
                            and (p24_1 := _parse_ts_px_row(pts_24[-1])) is not None
                            and (p24_1[0] - p24_0[0]) < 18.0 * 3600.0 * 1000.0
                        )
                    )
                ):
                    pts_24 = [older_24_arch[-1], *pts_24]
                if len(watcher_7d_pts) >= 2:
                    pts_7d_vals = [pt[1] for pt in watcher_7d_pts]
                    pts_7d_ts = [pt[0] for pt in watcher_7d_pts]
                arch_7d_end_ts = pts_7d_ts[-1] if pts_7d_ts else last_pt[0]
                if len(pts_7d_ts) >= 2:
                    arch_7d_span_ms = max(0.0, pts_7d_ts[-1] - pts_7d_ts[0])
                if sparse_7d_pts and (sp_last := _parse_ts_px_row(sparse_7d_pts[-1])) is not None:
                    merged_sp = (
                        _prepend_older_non_diverged_points(archive_pts, sparse_7d_pts)
                        if last_pt[0] >= sp_last[0]
                        else _prepend_older_non_diverged_points(sparse_7d_pts, archive_pts)
                    )
                    sp_cutoff_7d = max(last_pt[0], sp_last[0]) - 7.0 * 86400.0 * 1000.0
                    sp_pts = [pt for row in merged_sp if (pt := _parse_ts_px_row(row)) is not None and pt[0] >= sp_cutoff_7d]
                    if len(sp_pts) >= 2:
                        pts_7d_vals = [pt[1] for pt in sp_pts]
                        arch_7d_end_ts = sp_pts[-1][0]
                        arch_7d_span_ms = max(0.0, sp_pts[-1][0] - sp_pts[0][0])
                if len(pts_7d_vals) >= 2:
                    vals_7d = pts_7d_vals
                if need_24h_supplement and pts_24:
                    series_24h = _prepend_older_non_diverged_points(series_24h, pts_24) if series_24h else pts_24
        except Exception:
            pass
    _LAST_7D_SPAN_MS_BY_SYM[str(symbol).strip().upper()] = arch_7d_span_ms

    # 3. Overlay ultra-recent 1-second ticks from cache_current_price.json
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
                        and (cand_last[0] - last_24_pt[0]) > 20 * 60 * 1000
                        and abs(cand_last[1] - last_24_pt[1]) <= 1e-6
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

    # 4. Normalize, sort, and deduplicate series_24h so callers always receive clean [tms, px] (or [tms, px, vol]) points
    if series_24h:
        cleaned_24h = []
        prev_t = -1.0
        is_sorted = True
        for row in series_24h:
            if (pt := _parse_ts_px_row(row)) is not None:
                if pt[0] < prev_t:
                    is_sorted = False
                prev_t = pt[0]
                if len(row) >= 3 and (vol_v := safe_pos_float(row[2], default=None)) is not None:
                    cleaned_24h.append([pt[0], pt[1], vol_v])
                else:
                    cleaned_24h.append([pt[0], pt[1]])
        if not is_sorted:
            cleaned_24h.sort(key=lambda x: x[0])
        if len(cleaned_24h) >= 2:
            dedup_24h = [cleaned_24h[0]]
            for pt in cleaned_24h[1:]:
                if pt[0] > dedup_24h[-1][0]:
                    dedup_24h.append(pt)
                else:
                    dedup_24h[-1] = [pt[0], pt[1], dedup_24h[-1][2]] if (len(pt) < 3 and len(dedup_24h[-1]) >= 3) else pt
            if len(dedup_24h) >= 3 and any(len(p) >= 3 for p in dedup_24h):
                filtered_24h = [dedup_24h[0]]
                for idx_d in range(1, len(dedup_24h) - 1):
                    cur_d, next_d = dedup_24h[idx_d], dedup_24h[idx_d + 1]
                    prev_d = filtered_24h[-1]
                    if len(cur_d) < 3 and (len(prev_d) >= 3 or len(next_d) >= 3):
                        ref_mid = 0.5 * (prev_d[1] + next_d[1])
                        if (
                            ref_mid > 0.0
                            and abs(prev_d[1] - next_d[1]) / ref_mid <= 0.025
                            and abs(cur_d[1] - ref_mid) / ref_mid > 0.025
                        ):
                            continue
                    filtered_24h.append(cur_d)
                filtered_24h.append(dedup_24h[-1])
                dedup_24h = _strip_off_rth_flat_poller_tail(filtered_24h)
            series_24h = dedup_24h if len(dedup_24h) >= 2 else []
        else:
            series_24h = []

    if vals_7d and series_24h:
        appended_newer = False
        if arch_7d_end_ts > 0.0:
            for pt in series_24h:
                if pt[0] > arch_7d_end_ts and (pt_px := safe_pos_float(pt[1], default=None)) is not None:
                    if vals_7d[-1] != pt_px and not _prices_diverged(pt_px, vals_7d[-1]):
                        vals_7d.append(pt_px)
                    appended_newer = True
        if not appended_newer and (latest_px := safe_pos_float(series_24h[-1][1], default=None)) is not None:
            if vals_7d[-1] != latest_px and not _prices_diverged(latest_px, vals_7d[-1]):
                vals_7d.append(latest_px)

    if _memo_key is not None:
        if len(_PRICE_SERIES_MEMO) > 128:
            _PRICE_SERIES_MEMO.clear()
        _PRICE_SERIES_MEMO[_memo_key] = (
            _memo_file_sig,
            _memo_cur_sig,
            [list(pt) for pt in series_24h],
            list(vals_7d),
            arch_7d_span_ms,
        )

    return series_24h, vals_7d


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
    is_eq_sym = sym_u.endswith(("_EQ", "_US_EQ"))
    sym_cands = [
        c
        for c in cache_symbol_variants(sym, include_base=True)
        if c.upper().endswith(("_EQ", "_US_EQ")) == is_eq_sym
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
        ob, ob_cand = _first_telemetry_dict("orderbook_depth_")
    else:
        raw_ob_sym = obj_attr(ob, "symbol")
        ob_cand = raw_ob_sym.strip() if isinstance(raw_ob_sym, str) and raw_ob_sym.strip() else None
    if deriv is None:
        deriv, _ = _first_telemetry_dict("derivatives_telemetry_", prefer_freshest=True)
    if whale is None:
        whale, _ = _first_telemetry_dict("whale_snapshot_", prefer_freshest=True)
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
        if m_info.get("funding_rate") is None and (w_fr := safe_float(obj_attr(whale, "funding_rate"), default=None)) is not None:
            m_info["funding_rate"] = w_fr
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
) -> None:
    """Trigger on-demand orderbook, derivatives, or whale collector fetches when cached files are missing or stale."""
    if "PYTEST_CURRENT_TEST" in os.environ or not cache_dir:
        return
    eff_cache_dir = resolve_cache_dir_path(cache_dir)
    if not eff_cache_dir or not os.path.isdir(eff_cache_dir):
        return
    sym_u = str(symbol or "").strip().upper()
    if not sym_u or "PYTEST_CURRENT_TEST" in os.environ:
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
            if prefix == "orderbook_depth_":
                ob_disk = read_json_cached(fpath)
                if isinstance(ob_disk, dict) and (safe_pos_float(ob_disk.get("spread_bps"), default=0.0) or 0.0) > 200.0:
                    return True
            return False
        return max_age is not None and (eff_now - os.path.getmtime(fpath)) > max_age

    if _needs_fetch("orderbook_depth_", max_age_ob_sec):
        try:
            from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector

            OrderbookDepthCollector(cache_dir=eff_cache_dir).fetch(sym_u)
        except Exception:
            pass
    if include_derivatives and _needs_fetch("derivatives_telemetry_", max_age_whale_sec):
        try:
            from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector

            DerivativesTelemetryCollector(cache_dir=eff_cache_dir).fetch(sym_u)
        except Exception:
            pass
    if _needs_fetch("whale_snapshot_", max_age_whale_sec):
        try:
            from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector

            WhalePositioningCollector(cache_dir=eff_cache_dir).fetch(sym_u)
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
        if not str(k).strip().upper().endswith(("_EQ", "_US_EQ"))
    }
    ref_micro_syms: List[str] = []
    for pfx in ("orderbook_depth_", "derivatives_telemetry_", "whale_snapshot_"):
        for search_dir in (cache_dir, os.path.join(cache_dir, "telemetry")):
            for mpath in sorted(_glob.glob(os.path.join(search_dir, f"{pfx}*.json"))):
                msym = os.path.basename(mpath)[len(pfx) : -len(".json")].upper()
                if (
                    msym
                    and msym.isalnum()
                    and not msym.endswith(("_EQ", "_US_EQ"))
                    and msym not in microstructure
                    and msym not in ref_micro_syms
                ):
                    mbase = canonical_base_asset(msym)
                    if mbase not in covered_micro_bases:
                        ref_micro_syms.append(msym)
                        covered_micro_bases.add(mbase)
                    elif pfx == "orderbook_depth_" and isinstance(ob_d := _rj(mpath), dict):
                        cand_bd = safe_nonneg_float(ob_d.get("bid_depth_usd"), default=None)
                        existing_bds = [
                            safe_nonneg_float(mv.get("bid_depth_usd"), default=None)
                            for mk, mv in microstructure.items()
                            if not str(mk).strip().upper().endswith(("_EQ", "_US_EQ"))
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


def _refresh_stale_t212_cached_positions(
    cached_bals: Dict[str, Any],
    bcache_path: Optional[str],
    now_ts: float,
    *,
    max_age_sec: float = 180.0,
) -> bool:
    """Refresh stale (`>= max_age_sec`) `pos:t212:*` prices in `cached_bals` from Trading212 live portfolio."""
    if (
        "PYTEST_CURRENT_TEST" in os.environ
        or not isinstance(cached_bals, dict)
        or not bcache_path
        or not os.path.exists(bcache_path)
    ):
        return False
    t212_ts_vals = [
        safe_pos_float(bval.get("ts"), default=0.0) or 0.0
        for bkey, bval in cached_bals.items()
        if isinstance(bkey, str) and bkey.startswith("pos:t212:") and isinstance(bval, dict)
    ]
    if not t212_ts_vals:
        return False
    max_t212_ts = max(t212_ts_vals)
    if max_t212_ts <= 0.0 or (now_ts - max_t212_ts) < max_age_sec:
        return False
    try:
        from providers.t212_provider import T212Provider
        from state_io import save_json_state

        t212_rows = T212Provider()._client().get_portfolio()
        if not isinstance(t212_rows, list) or not t212_rows:
            return False
        t212_dirty = False
        for row in t212_rows:
            if not isinstance(row, dict):
                continue
            rticker = str(row.get("ticker") or "").strip().upper()
            cur_px_r = safe_pos_float(row.get("currentPrice"), default=None)
            if rticker and cur_px_r is not None and isinstance(c_pos := cached_bals.get(f"pos:t212:{rticker}"), dict):
                c_free = safe_pos_float(
                    c_pos.get("free") if c_pos.get("free") is not None else c_pos.get("free_base"),
                    default=0.0,
                ) or 0.0
                c_pos["price"] = round(cur_px_r, 4)
                if c_free > 0.0:
                    c_pos["value_usd"] = round(c_free * cur_px_r, 2)
                c_pos["ts"] = now_ts
                t212_dirty = True
        if t212_dirty:
            save_json_state(bcache_path, cached_bals, indent=2)
        return t212_dirty
    except Exception:
        return False


def enrich_24h_and_microstructure(
    cache_dir: str,
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    cur_p_items: Dict[str, Any],
    inst_by_sym: Dict[str, Any],
    now_ts: float,
    read_json_fn: Any = None,
) -> None:
    """Enrich prices_and_trends with 24h/7d metrics, fallback OHLC/Yahoo chart series, and microstructure."""
    now_ts = resolve_now_ts(now_ts)
    _rj = _resolve_read_json_cached(read_json_fn)
    btc_series_24h, btc_vals_7d = load_symbol_price_series(cache_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=True)
    bcache_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "venue_balances_state.json"))) if cache_dir else None
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
            if bticker and bticker not in t212_seed_tickers and (
                bticker in prices_and_trends or b_free > 0.0
            ):
                t212_seed_tickers.append(bticker)
    if (
        t212_seed_tickers
        and not any(t in inst_by_sym for t in t212_seed_tickers)
        and _rj in (_default_read_json_cached, read_json_recovering, read_json_cached)
    ):
        _refresh_stale_t212_cached_positions(cached_bals, bcache_path, now_ts)
    for ticker in t212_seed_tickers:
        cached_pos = cached_bals.get(f"pos:t212:{ticker}")
        cached_t212_px = safe_pos_float(cached_pos.get("price") if isinstance(cached_pos, dict) else None, default=None)
        if cached_t212_px is not None:
            cached_t212_px = round(cached_t212_px, 4)
        pos_ts_val = safe_pos_float(cached_pos.get("ts") if isinstance(cached_pos, dict) else None, default=None)
        if pos_ts_val is not None:
            t212_pos_ts[ticker] = pos_ts_val
        elif "PYTEST_CURRENT_TEST" in os.environ and bcache_mtime > 0.0:
            t212_pos_ts[ticker] = bcache_mtime
        t_entry = prices_and_trends.setdefault(ticker, {"price": cached_t212_px, "trend": 0, "growth_coefficient": 0.0})
        if cached_t212_px is not None and (
            ticker not in inst_by_sym or (safe_pos_float(t_entry.get("price"), default=0.0) or 0.0) <= 0.0
        ):
            t_entry["price"] = cached_t212_px
            t212_seeded_from_cache.add(ticker)
    from instrument_registry import discover_hyperliquid_spot

    for hl_sym, hl_base, hl_quote in discover_hyperliquid_spot(workspace_dir):
        ensure_fresh_symbol_microstructure(
            cache_dir, hl_sym, now_ts=now_ts, max_age_ob_sec=180.0, max_age_whale_sec=360.0, include_derivatives=True
        )
        hl_m_seed = load_symbol_microstructure(cache_dir, hl_sym, _rj) if hl_sym not in prices_and_trends else None
        if hl_sym not in prices_and_trends and (
            any(not str(k).strip().upper().endswith(("_EQ", "_US_EQ")) and canonical_base_asset(k) == hl_base for k in prices_and_trends)
            or hl_m_seed
        ):
            cached_hl_pos = cached_bals.get(f"pos:hyperliquid:{hl_sym}")
            hl_px = (
                safe_pos_float((hl_m_seed or {}).get("ob_mid_price"), default=0.0)
                or safe_pos_float(cached_hl_pos.get("price") if isinstance(cached_hl_pos, dict) else None, default=0.0)
                or resolve_symbol_price(hl_sym, prices_and_trends, explicit_quote=hl_quote)
            )
            prices_and_trends[hl_sym] = {"price": hl_px if hl_px > 0 else None, "trend": 0, "growth_coefficient": 0.0}
    for sym in list(prices_and_trends.keys()):
        sym_px_cur = safe_float(prices_and_trends[sym].get("price"), 0.0)
        is_eq_sym = sym.endswith(("_EQ", "_US_EQ"))
        if sym == "BTCUSDC" and btc_series_24h:
            matched_series_24h, arch_vals_7d = list(btc_series_24h), list(btc_vals_7d)
        else:
            matched_series_24h, arch_vals_7d = load_symbol_price_series(cache_dir, sym, current_price_items=cur_p_items)
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
                and (eff_t212_ts <= 0.0 or (last_24_pt[0] / 1000.0) > eff_t212_ts)
                and (int(now_ts * 1000) - last_24_pt[0]) <= 20 * 60 * 1000
                and not _prices_diverged(sym_px_cur, last_24_px)
                and (
                    (eff_t212_ts > 0.0 and (last_24_pt[0] / 1000.0) > eff_t212_ts)
                    or abs(sym_px_cur - last_24_px) / max(sym_px_cur, last_24_px) <= 0.005
                )
            ):
                prices_and_trends[sym]["price"] = sym_px_cur = round(last_24_px, 4)
            if _prices_diverged(sym_px_cur, last_24_px):
                matched_series_24h = []
            elif last_24_px > 0 and abs(sym_px_cur - last_24_px) > 1e-9:
                now_ms_24 = int(now_ts * 1000)
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
            if _prices_diverged(sym_px_cur, safe_float(arch_vals_7d[-1], 0.0)):
                arch_vals_7d = []
            elif abs(sym_px_cur - safe_float(arch_vals_7d[-1], 0.0)) > 1e-9:
                arch_vals_7d = list(arch_vals_7d) + [sym_px_cur]
        if safe_float(prices_and_trends[sym].get("price"), 0.0) <= 0.0:
            if matched_series_24h:
                prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
            elif arch_vals_7d:
                prices_and_trends[sym]["price"] = arch_vals_7d[-1]
        s24_p0 = _parse_ts_px_row(matched_series_24h[0]) if len(matched_series_24h) >= 2 else None
        s24_p1 = _parse_ts_px_row(matched_series_24h[-1]) if len(matched_series_24h) >= 2 else None
        span_24h_ms = max(0.0, s24_p1[0] - s24_p0[0]) if (s24_p0 and s24_p1) else 0.0
        post_rth_px_override: Optional[float] = None
        post_rth_series_replaced = False
        m24: Dict[str, Any] = compute_24h_price_metrics(matched_series_24h) or {} if matched_series_24h else {}
        if is_eq_sym and m24 and s24_p1 is not None and (
            (int(now_ts * 1000) - s24_p1[0]) > 20 * 60 * 1000
            or (sym_px_cur > 0 and abs(sym_px_cur - s24_p1[1]) / max(sym_px_cur, s24_p1[1]) > 0.005)
        ):
            if sym_px_cur > 0 and not _prices_diverged(sym_px_cur, s24_p1[1]) and abs(round(sym_px_cur, 4) - round(s24_p1[1], 4)) > 1e-4:
                post_rth_px_override = round(sym_px_cur, 4)
                if len(matched_series_24h) >= 2:
                    rth_vp_anchors = {
                        k_vp: m24[k_vp]
                        for k_vp in ("poc_px", "sec_poc_px", "hvn2_px", "lvn_px", "val_px", "vah_px", "profile_shape", "vp_source")
                        if k_vp in m24
                    }
                    last_bar_tail = (
                        list(matched_series_24h[-1][2:])
                        if isinstance(matched_series_24h[-1], (list, tuple)) and len(matched_series_24h[-1]) >= 3
                        else []
                    )
                    override_series_24h = list(matched_series_24h[:-1]) + [[s24_p1[0], post_rth_px_override, *last_bar_tail]]
                    m24 = compute_24h_price_metrics(override_series_24h) or m24
                    if rth_vp_anchors:
                        for k_vp_clr in ("poc_px", "sec_poc_px", "hvn2_px", "lvn_px", "val_px", "vah_px", "profile_shape", "vp_source"):
                            m24.pop(k_vp_clr, None)
                        m24.update(rth_vp_anchors)
                    if abs(sym_px_cur - s24_p1[1]) / max(sym_px_cur, s24_p1[1]) > 0.005:
                        post_rth_series_replaced = True
                        matched_series_24h = override_series_24h
                        for k_ov in (
                            "kalman_vel",
                            "kalman_vel_std",
                            "kalman_trend",
                            "slope_90s",
                            "growth_90s",
                            "trend_90s",
                            "slope_3_7m",
                            "growth_3_7m",
                            "trend_3_7m",
                            "slope_15m",
                            "growth_15m",
                            "trend_15m",
                            "slope_2_5h",
                            "growth_2_5h",
                            "trend_2_5h",
                            "gradient_recent_fast",
                            "gradient_recent",
                        ):
                            prices_and_trends[sym].pop(k_ov, None)
                if "chg_24h_pct" in m24 and s24_p0 is not None and s24_p0[1] > 0:
                    m24["chg_24h_pct"] = pct_delta(post_rth_px_override, s24_p0[1])
            for k_stale in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "recent_tape_15m", "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct"):
                m24.pop(k_stale, None)
        arch_span_7d_ms = _LAST_7D_SPAN_MS_BY_SYM.get(sym.upper(), 0.0)
        active_7d_vals: List[float] = []
        if arch_vals_7d and len(arch_vals_7d) >= 2 and (
            is_eq_sym
            or "chg_7d_pct" not in prices_and_trends[sym]
            or arch_span_7d_ms >= 5 * 86400 * 1000
            or (
                prices_and_trends[sym].get("_partial_7d_span_ms") is not None
                and arch_span_7d_ms > safe_float(prices_and_trends[sym].get("_partial_7d_span_ms"), 0.0)
            )
        ):
            for k_7d_opt in ("major_swing_low_px", "rally_from_major_low_pct", "recent_swings_7d", "avg_swing_amp_7d_pct"):
                prices_and_trends[sym].pop(k_7d_opt, None)
            prices_and_trends[sym].update(compute_7d_range_metrics(arch_vals_7d, decimals=4 if is_eq_sym else None))
            active_7d_vals = list(arch_vals_7d)
            if 0.0 <= arch_span_7d_ms < 5 * 86400 * 1000:
                prices_and_trends[sym]["_partial_7d_span_ms"] = arch_span_7d_ms
            else:
                prices_and_trends[sym].pop("_partial_7d_span_ms", None)
        ohlc_fn = _resolve_fallback_ohlc_fn(sym, inst_by_sym)
        s24_is_live_clock = (
            raw_s24_last_ts_ms <= 0.0
            or abs(now_ts * 1000.0 - raw_s24_last_ts_ms) <= 30 * 60 * 1000.0
            or sym in (inst_by_sym or {})
        )
        if callable(ohlc_fn) and s24_is_live_clock:
            if not m24 or (0.0 < span_24h_ms < 18 * 3600 * 1000):
                try:
                    pts_24h = _load_stitched_24h_ohlc_points(ohlc_fn, sym, now_ts - 86400.0)
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
                "chg_7d_pct" not in prices_and_trends[sym]
                or (0.0 < span_24h_ms < 18 * 3600 * 1000)
                or (0.0 < arch_span_7d_ms < 5 * 86400 * 1000)
                or prices_and_trends[sym].get("_partial_7d_span_ms") is not None
            ):
                try:
                    vals_7d = _load_stitched_7d_ohlc_closes(ohlc_fn, sym, now_ts, m24, sym_px_cur)
                    if len(vals_7d) >= 2 and ("chg_7d_pct" not in prices_and_trends[sym] or len(vals_7d) >= 24):
                        for k_7d_opt in ("major_swing_low_px", "rally_from_major_low_pct", "recent_swings_7d", "avg_swing_amp_7d_pct"):
                            prices_and_trends[sym].pop(k_7d_opt, None)
                        prices_and_trends[sym].update(compute_7d_range_metrics(vals_7d))
                        prices_and_trends[sym].pop("_partial_7d_span_ms", None)
                        active_7d_vals = list(vals_7d)
                except Exception:
                    pass
        if (
            s24_is_live_clock
            and m24.get("vp_source") == "TPO"
            and (
                callable(ohlc_fn)
                or "PYTEST_CURRENT_TEST" not in os.environ
                or os.environ.get("MPTRADE_OFFLINE_DISK_CACHE") == "1"
            )
        ):
            try:
                pts_vp = _load_stitched_24h_ohlc_points(ohlc_fn, sym, now_ts - 86400.0, require_positive_volume=True)
                if len(pts_vp) >= 8 and (pts_vp[-1][0] - pts_vp[0][0]) >= 18 * 3600 * 1000:
                    vp_last_px = safe_pos_float(pts_vp[-1][1], default=0.0) or 0.0
                    if sym_px_cur <= 0.0 or (vp_last_px > 0.0 and abs(sym_px_cur - vp_last_px) / max(sym_px_cur, vp_last_px) <= 0.015):
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
                            if sym_px_cur > 0:
                                pts_vp.append([int(now_ts * 1000), sym_px_cur, 0.0])
                            if (vp_real := compute_volume_profile_metrics(pts_vp)) and vp_real.get("vp_source") == "REAL_VOL":
                                for _vp_clr_k in ("sec_poc_px", "hvn2_px", "lvn_px"):
                                    m24.pop(_vp_clr_k, None)
                                m24.update(vp_real)
            except Exception:
                pass
        prices_and_trends[sym].pop("_partial_7d_span_ms", None)
        is_t212_inst = sym in inst_by_sym and normalize_venue_name(getattr(inst_by_sym[sym], "provider_name", "")) == "t212"
        p24_eq_path = (
            str(resolve_cachedb_filepath(os.path.join(cache_dir, f"cache_24_price_{sym}.json")))
            if cache_dir and is_eq_sym
            else ""
        )
        memo_chart = _EQ_YAHOO_CHART_MEMO.get(sym) if (is_eq_sym and "PYTEST_CURRENT_TEST" not in os.environ) else None
        memo_ttl_s = 240.0
        if is_eq_sym and memo_chart is not None:
            try:
                from instrument_registry import is_symbol_market_open

                if not is_symbol_market_open(sym, now_ts):
                    memo_ttl_s = 1800.0
            except Exception:
                pass
        memo_chart_fresh = bool(memo_chart is not None and 0.0 <= (now_ts - memo_chart[0]) < memo_ttl_s)
        eq_cache_stale = bool(
            is_eq_sym
            and "PYTEST_CURRENT_TEST" not in os.environ
            and (
                memo_chart_fresh
                or raw_s24_last_ts_ms <= 0.0
                or (int(now_ts * 1000) - raw_s24_last_ts_ms) > 5 * 60 * 1000
                or "chg_24h_pct" not in m24
                or "chg_1h_pct" not in m24
                or m24.get("vp_source") == "TPO"
                or not p24_eq_path
                or not os.path.exists(p24_eq_path)
            )
        )
        if (not m24 and (is_t212_inst or (is_eq_sym and "PYTEST_CURRENT_TEST" not in os.environ))) or eq_cache_stale:
            try:
                if memo_chart_fresh and memo_chart is not None:
                    chart_res = memo_chart[1]
                else:
                    _chart = load_local_module(os.path.join(workspace_dir, "212trading"), "market_data")._chart
                    chart_res = _chart(sym.split("_", 1)[0], rng="5d", interval="5m")
                    if "PYTEST_CURRENT_TEST" not in os.environ:
                        _EQ_YAHOO_CHART_MEMO[sym] = (now_ts, chart_res)
                pts_5d: List[List[Any]] = []
                if isinstance(chart_res, tuple) and len(chart_res) >= 2 and isinstance(chart_res[1], list):
                    pts_5d = sorted(
                        (
                            (
                                [int(ts_f if ts_f > 1e11 else ts_f * 1000), round(px_f, 4), safe_nonneg_float(bar[2], 0.0) or 0.0]
                                if len(bar) >= 3
                                else [int(ts_f if ts_f > 1e11 else ts_f * 1000), round(px_f, 4)]
                            )
                            for bar in chart_res[1]
                            if isinstance(bar, (list, tuple))
                            and len(bar) >= 2
                            and (ts_f := safe_pos_float(bar[0], default=None)) is not None
                            and (px_f := safe_pos_float(bar[1], default=None)) is not None
                        ),
                        key=lambda pt: pt[0],
                    )
                elif isinstance(chart_res, dict):
                    q_obj = ((chart_res.get("indicators") or {}).get("quote") or [{}])[0]
                    vols = q_obj.get("volume") or []
                    pts_5d = sorted(
                        (
                            [
                                int(ts_f if ts_f > 1e11 else ts_f * 1000),
                                round(c_f, 4),
                                safe_nonneg_float(vols[idx], 0.0) or 0.0 if idx < len(vols) else 0.0,
                            ]
                            for idx, (ts, c) in enumerate(zip(chart_res.get("timestamp") or [], q_obj.get("close") or []))
                            if (ts_f := safe_pos_float(ts, default=None)) is not None and (c_f := safe_pos_float(c, default=None)) is not None
                        ),
                        key=lambda pt: pt[0],
                    )
                if pts_5d:
                    chart_px = safe_pos_float(pts_5d[-1][1], default=0.0) or 0.0
                    now_ms_i = int(now_ts * 1000)
                    rth_bar_fresh = (now_ms_i - pts_5d[-1][0]) <= 20 * 60 * 1000
                    if chart_px > 0.0 and (
                        sym_px_cur <= 0.0
                        or abs(sym_px_cur - chart_px) < 1e-4
                        or (
                            (not is_t212_inst or ("PYTEST_CURRENT_TEST" not in os.environ and eq_cache_stale))
                            and rth_bar_fresh
                            and not _prices_diverged(sym_px_cur, chart_px)
                            and (
                                sym not in t212_seeded_from_cache
                                or (
                                    (t212_pos_ts.get(sym, 0.0) <= 0.0 or (pts_5d[-1][0] / 1000.0) > t212_pos_ts.get(sym, 0.0))
                                    and (
                                        (t212_pos_ts.get(sym, 0.0) > 0.0 and (pts_5d[-1][0] / 1000.0) > t212_pos_ts.get(sym, 0.0))
                                        or abs(sym_px_cur - chart_px) / max(sym_px_cur, chart_px) <= 0.005
                                    )
                                )
                            )
                        )
                    ):
                        prices_and_trends[sym]["price"] = chart_px
                        sym_px_cur = chart_px
                    anchor_24h_ms = min(int(now_ts * 1000), int(pts_5d[-1][0])) if pts_5d else int(now_ts * 1000)
                    cutoff_24h_ms = anchor_24h_ms - 86400 * 1000
                    pts_24h_bars = [pt for pt in pts_5d if pt[0] >= cutoff_24h_ms]
                    if pts_24h_bars and (pts_24h_bars[-1][0] - pts_24h_bars[0][0]) < 18 * 3600 * 1000:
                        older_bars = [pt for pt in pts_5d if pt[0] < cutoff_24h_ms]
                        if older_bars:
                            pts_24h_bars = [older_bars[-1], *pts_24h_bars]
                    if len(pts_24h_bars) < 10:
                        pts_24h_bars = list(pts_5d[-78:])
                    pts_24h = list(pts_24h_bars)
                    if rth_bar_fresh and raw_matched_series_24h and pts_24h:
                        last_bar_ts = pts_24h[-1][0]
                        recent_sub5m = [
                            pt for pt in raw_matched_series_24h
                            if isinstance(pt, (list, tuple))
                            and len(pt) >= 2
                            and last_bar_ts - 15 * 60 * 1000 <= pt[0] <= min(now_ms_i + 5000, last_bar_ts + 300 * 1000)
                            and not _prices_diverged(pt[1], pts_24h[-1][1])
                        ]
                        if recent_sub5m:
                            by_ts = {pt[0]: pt for pt in pts_24h}
                            for r_pt in recent_sub5m:
                                by_ts.setdefault(r_pt[0], r_pt)
                            pts_24h = [by_ts[k] for k in sorted(by_ts)]
                    pts_24h_to_save = list(pts_24h)
                    px_diff_from_chart = bool(
                        pts_24h and sym_px_cur > 0 and not _prices_diverged(sym_px_cur, pts_24h[-1][1]) and abs(round(sym_px_cur, 4) - round(pts_24h[-1][1], 4)) > 1e-4
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
                    if (post_rth_series_replaced or (eq_cache_stale and len(pts_24h) >= 5)):
                        for k_ov in (
                            "kalman_vel",
                            "kalman_vel_std",
                            "kalman_trend",
                            "slope_90s",
                            "growth_90s",
                            "trend_90s",
                            "slope_3_7m",
                            "growth_3_7m",
                            "trend_3_7m",
                            "slope_15m",
                            "growth_15m",
                            "trend_15m",
                            "slope_2_5h",
                            "growth_2_5h",
                            "trend_2_5h",
                            "gradient_recent_fast",
                            "gradient_recent",
                        ):
                            prices_and_trends[sym].pop(k_ov, None)
                    pts_24h_for_m24 = pts_24h
                    if px_diff_from_chart and not rth_bar_fresh and not px_diverged_05pct and len(pts_24h) >= 2:
                        last_bar_tail = list(pts_24h[-1][2:]) if isinstance(pts_24h[-1], (list, tuple)) and len(pts_24h[-1]) >= 3 else []
                        pts_24h_for_m24 = list(pts_24h[:-1]) + [[pts_24h[-1][0], round(sym_px_cur, 4), *last_bar_tail]]
                    if m24 := (compute_24h_price_metrics(pts_24h_for_m24) or {}):
                        if not rth_bar_fresh and px_diff_from_chart and (rth_vp_chart := compute_volume_profile_metrics(pts_24h_to_save)):
                            for k_vp_clr in ("poc_px", "sec_poc_px", "hvn2_px", "lvn_px", "val_px", "vah_px", "profile_shape", "vp_source"):
                                m24.pop(k_vp_clr, None)
                            for k_vp in ("poc_px", "sec_poc_px", "hvn2_px", "lvn_px", "val_px", "vah_px", "profile_shape", "vp_source"):
                                if k_vp in rth_vp_chart:
                                    m24[k_vp] = rth_vp_chart[k_vp]
                        matched_series_24h = pts_24h
                        post_rth_px_override = None
                        if not rth_bar_fresh or px_diverged_05pct:
                            for k_stale in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "recent_tape_15m", "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct"):
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
                            prices_and_trends[sym].pop(k_7d_opt, None)
                        prices_and_trends[sym].update(compute_7d_range_metrics(vals_5d, decimals=4))
                        active_7d_vals = list(vals_5d)
                    if cache_dir and os.path.isdir(cache_dir) and "PYTEST_CURRENT_TEST" not in os.environ:
                        try:
                            from state_io import save_json_state

                            eff_p24_pts = pts_24h_to_save or pts_24h_bars
                            raw_ohlc_cnt = sum(
                                1 for p in raw_matched_series_24h if isinstance(p, (list, tuple)) and len(p) >= 3
                            )
                            new_ohlc_cnt = sum(
                                1 for p in eff_p24_pts if isinstance(p, (list, tuple)) and len(p) >= 3
                            )
                            if eff_p24_pts and (
                                len(raw_matched_series_24h) < 10
                                or raw_ohlc_cnt < new_ohlc_cnt
                                or (
                                    not rth_bar_fresh
                                    and any(isinstance(p, (list, tuple)) and len(p) < 3 for p in raw_matched_series_24h)
                                )
                            ):
                                p24_target = p24_eq_path or str(resolve_cachedb_filepath(os.path.join(cache_dir, f"cache_24_price_{sym}.json")))
                                save_json_state(p24_target, {"items": {sym: eff_p24_pts}})
                            if pts_24h_to_save or pts_24h_bars:
                                ohlc24_target = str(resolve_cachedb_filepath(os.path.join(cache_dir, f"cache_ohlc_24h_{sym}.json")))
                                save_json_state(ohlc24_target, {"ts": now_ts, "symbol": sym, "points": pts_24h_to_save or pts_24h_bars})
                            p365_jsonl_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, f"cache_365_price_{sym}.jsonl")))
                            if not os.path.exists(p365_jsonl_path) and pts_5d:
                                os.makedirs(os.path.dirname(p365_jsonl_path), exist_ok=True)
                                with open(p365_jsonl_path, "w", encoding="utf-8") as _f365:
                                    for _pt in pts_5d:
                                        if isinstance(_pt, (list, tuple)) and len(_pt) >= 2:
                                            _f365.write(f'{{"i": [{int(_pt[0])}, {_pt[1]}]}}\n')
                        except Exception:
                            pass
            except Exception:
                pass
        if m24:
            if is_eq_sym and not any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct")):
                for k_stale_15m in ("recent_tape_15m", "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct"):
                    m24.pop(k_stale_15m, None)
            _apply_24h_series_enrichment(sym, prices_and_trends[sym], m24, matched_series_24h, btc_series_24h)
            if (
                is_eq_sym
                and not post_rth_series_replaced
                and (
                    not any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct"))
                    or (
                        matched_series_24h
                        and (last_eq_pt := _parse_ts_px_row(matched_series_24h[-1])) is not None
                        and (int(now_ts * 1000) - last_eq_pt[0]) > 20 * 60 * 1000
                    )
                )
            ):
                for k_stale_s in (
                    "slope_90s",
                    "growth_90s",
                    "trend_90s",
                    "slope_3_7m",
                    "growth_3_7m",
                    "trend_3_7m",
                    "gradient_recent_fast",
                ):
                    prices_and_trends[sym].pop(k_stale_s, None)
            if post_rth_px_override is not None:
                recalculate_live_price_geometry(prices_and_trends[sym], post_rth_px_override)

        if is_eq_sym or "USD" in compatible_quote_suffixes(sym):
            ensure_fresh_symbol_microstructure(
                cache_dir,
                sym.upper(),
                now_ts=now_ts,
                max_age_ob_sec=180.0,
                max_age_whale_sec=360.0,
                include_derivatives=True,
            )
        if m_info := load_symbol_microstructure(cache_dir, sym, _rj):
            is_usd_sym = "USD" in compatible_quote_suffixes(sym)
            sym_px_m = safe_float((prices_and_trends.get(sym) or {}).get("price"), 0.0)
            ob_mid_m = safe_pos_float(m_info.get("ob_mid_price"), default=None)
            src_cand_u = str(m_info.get("_source_cand") or sym).strip().upper()
            is_exact_ob_sym = src_cand_u == str(sym).strip().upper()
            cross_venue_collision = (not is_exact_ob_sym) and (src_cand_u in inst_by_sym) and (sym in inst_by_sym)
            if not is_eq_sym and is_usd_sym and isinstance(prices_and_trends.get(sym), dict) and ob_mid_m is not None and not cross_venue_collision:
                if sym_px_m <= 0.0 or (abs(sym_px_m - ob_mid_m) / max(sym_px_m, ob_mid_m) <= 0.01):
                    prices_and_trends[sym]["price"] = ob_mid_m
                    sym_px_m = ob_mid_m
                    if active_7d_vals and len(active_7d_vals) >= 2 and not _prices_diverged(ob_mid_m, safe_float(active_7d_vals[0], 0.0)):
                        prices_and_trends[sym]["chg_7d_pct"] = pct_delta(ob_mid_m, safe_float(active_7d_vals[0], 0.0))
                    recalculate_live_price_geometry(prices_and_trends[sym], ob_mid_m)
                elif (
                    not _prices_diverged(sym_px_m, ob_mid_m)
                    and is_exact_ob_sym
                    and str(sym).strip().upper().endswith("USD")
                    and not str(sym).strip().upper().endswith(("FDUSD", "TUSD", "BUSD"))
                    and (
                        bool(matched_series_24h)
                        or any(
                            pk != sym
                            and str(pk).strip().upper().endswith(("_EQ", "_US_EQ")) == is_eq_sym
                            and canonical_base_asset(pk) == canonical_base_asset(sym)
                            and (p_sib := safe_pos_float((pv if isinstance(pv, dict) else {}).get("price"), default=None)) is not None
                            and abs(sym_px_m - p_sib) / max(sym_px_m, p_sib) <= 0.01
                            for pk, pv in prices_and_trends.items()
                        )
                    )
                ):
                    scale_r = sym_px_m / ob_mid_m
                    px_dec = _auto_px_decimals(sym_px_m)
                    m_info["ob_mid_price"] = prices_and_trends[sym]["price"] = round(sym_px_m, px_dec)
                    sym_px_m = m_info["ob_mid_price"]
                    for k_sc in ("best_bid", "best_ask", "bid_wall_px", "ask_wall_px"):
                        if (v_sc := safe_pos_float(m_info.get(k_sc), default=None)) is not None:
                            m_info[k_sc] = round(v_sc * scale_r, px_dec)
            elif is_eq_sym and align_equity_orderbook_to_spot(m_info, sym_px_m):
                pass
            elif not is_usd_sym and (sym_px_m <= 0.0 or _prices_diverged(sym_px_m, ob_mid_m)):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    m_info.pop(k_strip, None)
            enrich_microstructure_cvd(m_info, prices_and_trends.get(sym))
            m_info.update(compute_liquidation_clusters(prices_and_trends.get(sym), m_info))
            if m_info.get("spread_bps") is not None and isinstance(prices_and_trends.get(sym), dict):
                prices_and_trends[sym]["spread_bps"] = m_info["spread_bps"]
            microstructure[sym] = m_info

    for msym in discover_reference_microstructure_symbols(cache_dir, microstructure, _rj):
        if "USD" in compatible_quote_suffixes(msym) and _rj in (_default_read_json_cached, read_json_recovering, read_json_cached):
            ensure_fresh_symbol_microstructure(
                cache_dir,
                msym.upper(),
                now_ts=now_ts,
                max_age_ob_sec=180.0,
                max_age_whale_sec=360.0,
                include_derivatives=True,
            )
        if m_info_ref := load_symbol_microstructure(cache_dir, msym, _rj):
            msym_quotes = compatible_quote_suffixes(msym)
            msym_is_eq = is_equity_symbol(msym)
            p_ref = _first_variant_dict(
                prices_and_trends,
                [
                    c
                    for c in cache_symbol_variants(msym, include_original=True, include_base="USD" in msym_quotes, quotes=msym_quotes)
                    if is_equity_symbol(c) == msym_is_eq
                ],
            )
            if p_ref is not None and msym not in prices_and_trends:
                ob_mid_sib = safe_pos_float(m_info_ref.get("ob_mid_price"), default=None)
                if msym.upper().endswith("USD") and not msym.upper().endswith(("FDUSD", "TUSD", "BUSD")) and ob_mid_sib is not None:
                    sib_live_px = extract_current_price_item(casefold_get(cur_p_items, msym))
                    if (not sib_live_px or sib_live_px <= 0.0) and cache_dir:
                        if isinstance(l24_s := _rj(os.path.join(cache_dir, "24trend", f"live_{msym.upper()}.json")), dict):
                            w_src = l24_s.get("windows") if isinstance(l24_s.get("windows"), dict) else l24_s
                            sib_live_px = safe_pos_float(
                                ((w_src.get("222") or w_src.get("900") or {}) if isinstance(w_src, dict) else {}).get("current_price"),
                                default=None,
                            )
                        if not sib_live_px or sib_live_px <= 0.0:
                            ob_ref_d = _rj(os.path.join(cache_dir, f"orderbook_depth_{msym.upper()}.json"))
                            if not isinstance(ob_ref_d, dict):
                                ob_ref_d = _rj(os.path.join(cache_dir, "telemetry", f"orderbook_depth_{msym.upper()}.json"))
                            ob_ref_ts = normalize_epoch_sec(
                                safe_pos_float((ob_ref_d if isinstance(ob_ref_d, dict) else {}).get("ts"), default=0.0) or 0.0,
                                default=0.0,
                            )
                            if ob_ref_ts > 0.0 and (now_ts - ob_ref_ts) > 120.0:
                                sib_live_px = safe_pos_float(p_ref.get("price"), default=None)
                    if (
                        sib_live_px
                        and sib_live_px > 0.0
                        and abs(sib_live_px - ob_mid_sib) / max(sib_live_px, ob_mid_sib) > 0.01
                        and not _prices_diverged(sib_live_px, ob_mid_sib)
                    ):
                        s_scale = sib_live_px / ob_mid_sib
                        s_dec = _auto_px_decimals(sib_live_px)
                        ob_mid_sib = m_info_ref["ob_mid_price"] = round(sib_live_px, s_dec)
                        for k_sc in ("best_bid", "best_ask", "bid_wall_px", "ask_wall_px"):
                            if (v_sc := safe_pos_float(m_info_ref.get(k_sc), default=None)) is not None:
                                m_info_ref[k_sc] = round(v_sc * s_scale, s_dec)
                if ob_mid_sib is not None and not _prices_diverged(ob_mid_sib, safe_float(p_ref.get("price"), 0.0)):
                    p_sib = dict(p_ref)
                    p_sib["price"] = ob_mid_sib
                    recalculate_live_price_geometry(p_sib, ob_mid_sib)
                    if m_info_ref.get("spread_bps") is not None:
                        p_sib["spread_bps"] = m_info_ref["spread_bps"]
                    prices_and_trends[msym] = p_sib
                    p_ref = p_sib
            elif p_ref is None:
                ref_px = safe_pos_float(m_info_ref.get("ob_mid_price"), default=None)
                ser_24h, vals_7d = load_symbol_price_series(cache_dir, msym, current_price_items=cur_p_items)
                if not ser_24h and msym.endswith("USD") and not msym.endswith(("FDUSD", "TUSD", "BUSD")) and "PYTEST_CURRENT_TEST" not in os.environ:
                    try:
                        krk_ohlc_fn = _resolve_fallback_ohlc_fn(msym, inst_by_sym)
                        if callable(krk_ohlc_fn):
                            ser_24h = _load_stitched_24h_ohlc_points(krk_ohlc_fn, msym, now_ts - 86400.0)
                            if ref_px is not None and ser_24h and abs(ref_px - float(ser_24h[-1][1])) > 1e-9 and not _prices_diverged(ref_px, float(ser_24h[-1][1])):
                                ser_24h.append([int(now_ts * 1000), ref_px])
                            if len(vals_7d) < 2:
                                vals_7d = _load_stitched_7d_ohlc_closes(krk_ohlc_fn, msym, now_ts, None, ref_px or 0.0)
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
                    prices_and_trends[msym] = p_ref
            enrich_microstructure_cvd(m_info_ref, p_ref)
            m_info_ref.update(compute_liquidation_clusters(p_ref, m_info_ref))
            microstructure[msym] = m_info_ref


__all__ = [
    "_MICRO_PRICE_ANCHOR_KEYS",
    "_VENUE_DEFAULT_FEE_SPECS",
    "_WINDOW_TREND_SPECS",
    "_apply_24h_series_enrichment",
    "_auto_px_decimals",
    "_blend_structural_liq_anchor",
    "_build_vp_position_dict",
    "_compute_horizon_range_dict",
    "_extract_ohlc_series_points",
    "_first_variant_dict",
    "_infer_symbol_venue_and_fees_bps",
    "_iter_t212_state_files",
    "_load_stitched_24h_ohlc_points",
    "_parse_ts_px_row",
    "_prices_diverged",
    "_refresh_stale_t212_cached_positions",
    "_resolve_non_static_prev_metric",
    "apply_live_24trend_windows",
    "apply_short_trend_item",
    "compute_24h_price_metrics",
    "compute_7d_range_metrics",
    "compute_cross_btc_metrics",
    "compute_cross_venue_arb_metrics",
    "compute_liquidation_clusters",
    "compute_swing_wave_metrics",
    "compute_volume_profile_metrics",
    "discover_reference_microstructure_symbols",
    "enrich_24h_and_microstructure",
    "enrich_microstructure_cvd",
    "ensure_fresh_symbol_microstructure",
    "extract_symbol_recent_fills",
    "load_symbol_microstructure",
    "load_symbol_price_series",
    "pct_delta",
    "range_position_pct",
    "reconcile_sibling_flow_and_liq_regimes",
    "round_opt_px",
    "update_microstructure_cycle_deltas",
]
