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
)
from intelligence.external.collectors.orderbook_depth_collector import round_spread_bps
from intelligence.macro_sentiment_ssot import (
    obj_attr,
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
    compute_cross_venue_arb_metrics,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
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
    round_opt_px,
)
from instrument_registry import casefold_get
from state_io import (
    read_json_dict_recovering as _default_read_json_cached,
    read_json_recovering,
    read_tail_jsonl_records,
    resolve_cachedb_filepath,
)

read_json_cached = _default_read_json_cached

logger = logging.getLogger("intelligence.sentiment.microstructure_collector")


def _resolve_read_json_cached(read_json_fn: Any = None) -> Any:
    _defaults = (_default_read_json_cached, read_json_recovering)
    if callable(read_json_fn) and read_json_fn not in _defaults:
        return read_json_fn
    if callable(read_json_cached) and read_json_cached not in _defaults:
        return read_json_cached
    import sys as _sys

    for _mod_name in (
        "intelligence.sentiment.collectors.pretrade_telemetry_collector",
        "intelligence.sentiment.guards.high_stake_guard",
        "intelligence.sentiment.collectors.macro_telemetry_collector",
        "intelligence.macro_sentiment_ssot",
    ):
        _mod = _sys.modules.get(_mod_name)
        _cand = getattr(_mod, "read_json_cached", None) if _mod is not None else None
        if callable(_cand) and _cand not in _defaults:
            return _cand
    return read_json_fn if callable(read_json_fn) else read_json_cached


def _filter_ohlc_closes_since(
    series_obj: Any,
    cutoff_ts: float,
    max_ts: Optional[float] = None,
) -> List[float]:
    ts_seq = getattr(series_obj, "timestamps", None) or ()
    cl_seq = getattr(series_obj, "closes", None) or ()
    if not isinstance(ts_seq, (list, tuple)) or not isinstance(cl_seq, (list, tuple)):
        return []
    out: List[float] = []
    for raw_ts, raw_px in zip(ts_seq, cl_seq):
        ts_v = safe_pos_float(raw_ts, default=None)
        px_v = safe_pos_float(raw_px, default=None)
        if ts_v is None or px_v is None:
            continue
        if ts_v >= cutoff_ts and (max_ts is None or ts_v <= max_ts):
            out.append(px_v)
    return out


def _extract_cached_items_series(data: Any, cands: list[str]) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    raw_items = data.get("items") if "items" in data else data
    if isinstance(raw_items, dict):
        for k_cand in cands:
            if isinstance((ser := casefold_get(raw_items, k_cand, [])), list) and ser:
                return ser
        if "items" in data and len(raw_items) == 1 and cands:
            only_k, first_val = next(iter(raw_items.items()))
            if (
                isinstance(first_val, list)
                and canonical_base_asset(only_k) == canonical_base_asset(cands[0])
                and compatible_quote_suffixes(only_k) == compatible_quote_suffixes(cands[0])
            ):
                return first_val
    elif isinstance(raw_items, list):
        return raw_items
    return []


def load_symbol_price_series(
    cachedb_dir: str,
    symbol: str,
    current_price_items: Optional[Dict[str, Any]] = None,
    *,
    include_7d: bool = True,
    quotes: Optional[tuple[str, ...]] = None,
) -> Tuple[list, list]:
    """Load and merge high-density 24h & 7d price series from cache_24_price, cache_365_price, cache_sparse_price, and cache_current_price."""
    eff_quotes = quotes if quotes is not None else compatible_quote_suffixes(symbol)
    cands = [
        c for c in cache_symbol_variants(symbol, include_original=True, include_base="USD" in eff_quotes, quotes=eff_quotes)
        if isinstance(c, str) and "/" not in c
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

    # 1. Try primary 24h dense cache (cache_24_price_<SYM>.json and legacy variants)
    _rj_cand = _resolve_read_json_cached()
    _rj = read_json_recovering if _rj_cand is _default_read_json_cached else _rj_cand
    p0: Optional[Tuple[float, float]] = None
    p1: Optional[Tuple[float, float]] = None
    span_24h_ms = 0.0
    for cand in cands:
        for p24_path in _cand_paths(f"cache_24_price_{cand}.json"):
            if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                continue
            ser = _extract_cached_items_series(_rj(p24_path), cands)
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

    def _read_arch_file(arch_path: str, max_b: int = 1024 * 1024) -> list:
        if not os.path.exists(arch_path):
            return []
        try:
            if arch_path.endswith(".json"):
                aser = _extract_cached_items_series(_rj(arch_path), cands)
                return list(aser) if len(aser) >= 2 else []
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
            return parsed_rows if len(parsed_rows) >= 2 else []
        except Exception:
            return []

    for cand in (cands if (include_7d or need_24h_supplement) else ()):
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
                for sp_path in _cand_paths(f"cache_sparse_price_{cand}.jsonl"):
                    if sp_rows := _read_arch_file(sp_path, max_b=512 * 1024):
                        sp0, sp1 = _parse_ts_px_row(sp_rows[0]), _parse_ts_px_row(sp_rows[-1])
                        if sp0 and sp1 and (sp1[0] - sp0[0]) > arch_span_ms:
                            sparse_7d_pts = sp_rows
                            break
            if arch_span_ms >= 5.0 * 86400.0 * 1000.0 or sparse_7d_pts:
                break

    if archive_pts:
        try:
            if (last_pt := _parse_ts_px_row(archive_pts[-1])) is not None:
                cutoff_24h, cutoff_7d = last_pt[0] - 86400.0 * 1000.0, last_pt[0] - 7.0 * 86400.0 * 1000.0
                pts_24, pts_7d_vals = [], []
                for row in archive_pts:
                    if (pt := _parse_ts_px_row(row)) is not None:
                        if pt[0] >= cutoff_7d:
                            pts_7d_vals.append(pt[1])
                        if need_24h_supplement and pt[0] >= cutoff_24h:
                            pts_24.append([pt[0], pt[1]])
                if sparse_7d_pts and (sp_last := _parse_ts_px_row(sparse_7d_pts[-1])) is not None:
                    sp_cutoff_7d = sp_last[0] - 7.0 * 86400.0 * 1000.0
                    sp_vals = [pt[1] for row in sparse_7d_pts if (pt := _parse_ts_px_row(row)) is not None and pt[0] >= sp_cutoff_7d]
                    if len(sp_vals) >= 2:
                        pts_7d_vals = sp_vals
                if len(pts_7d_vals) >= 2:
                    vals_7d = pts_7d_vals
                if need_24h_supplement and pts_24:
                    s0_pt = _parse_ts_px_row(series_24h[0]) if series_24h else None
                    if s0_pt is None or not _prices_diverged(pts_24[-1][1], s0_pt[1]):
                        series_24h = (pts_24 + series_24h) if series_24h else pts_24
        except Exception:
            pass

    # 3. Overlay ultra-recent 1-second ticks from cache_current_price.json
    if isinstance(current_price_items, dict) and current_price_items:
        last_24_pt = _parse_ts_px_row(series_24h[-1]) if series_24h else None
        for cand in cands:
            cur_item = casefold_get(current_price_items, cand)
            if isinstance(cur_item, list) and cur_item:
                cand_pts = [[pt[0], pt[1]] for pt in cur_item if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                cand_last = _parse_ts_px_row(cand_pts[-1]) if cand_pts else None
                if last_24_pt is not None and cand_last is not None and _prices_diverged(last_24_pt[1], cand_last[1]):
                    continue
                series_24h = list(series_24h) + cand_pts
                break
            elif (px_f := extract_current_price_item(cur_item)) is not None and series_24h and last_24_pt is not None:
                if not _prices_diverged(last_24_pt[1], px_f):
                    series_24h = list(series_24h) + [[last_24_pt[0] + 1000.0, px_f]]
                    break

    # 4. Normalize, sort, and deduplicate series_24h so callers always receive clean [tms, px] (or [tms, px, vol]) points
    if series_24h:
        cleaned_24h = sorted(
            (
                ([pt[0], pt[1], vol_v] if len(row) >= 3 and (vol_v := safe_pos_float(row[2], default=None)) is not None else [pt[0], pt[1]])
                for row in series_24h if (pt := _parse_ts_px_row(row)) is not None
            ),
            key=lambda x: x[0],
        )
        if len(cleaned_24h) >= 2:
            dedup_24h = [cleaned_24h[0]]
            for pt in cleaned_24h[1:]:
                if pt[0] > dedup_24h[-1][0]:
                    dedup_24h.append(pt)
                else:
                    dedup_24h[-1] = [pt[0], pt[1], dedup_24h[-1][2]] if (len(pt) < 3 and len(dedup_24h[-1]) >= 3) else pt
            series_24h = dedup_24h if len(dedup_24h) >= 2 else []
        else:
            series_24h = []

    if vals_7d and series_24h and (latest_px := safe_pos_float(series_24h[-1][1], default=None)) is not None:
        if vals_7d[-1] != latest_px and not _prices_diverged(latest_px, vals_7d[-1]):
            vals_7d.append(latest_px)

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
    sym_cands = cache_symbol_variants(sym, include_base=True)

    def _first_telemetry_dict(prefix: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        for cand in sym_cands:
            if isinstance((d := read_json_fn(os.path.join(cache_dir, f"{prefix}{cand}.json"))), dict):
                return d, cand
        return None, None

    if ob is None:
        ob, ob_cand = _first_telemetry_dict("orderbook_depth_")
    else:
        raw_ob_sym = obj_attr(ob, "symbol")
        ob_cand = raw_ob_sym.strip() if isinstance(raw_ob_sym, str) and raw_ob_sym.strip() else None
    if deriv is None:
        deriv, _ = _first_telemetry_dict("derivatives_telemetry_")
    if whale is None:
        whale, _ = _first_telemetry_dict("whale_snapshot_")
    if not (ob or deriv or whale):
        return None
    m_info: Dict[str, Any] = {"_source_cand": ob_cand} if ob_cand else {}
    if ob is not None:
        ob_same_quote = (not ob_cand) or (compatible_quote_suffixes(ob_cand) == compatible_quote_suffixes(sym))
        m_info["ob_imbalance"] = obj_attr(ob, "imbalance_ratio", obj_attr(ob, "ob_imbalance"))
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
        m_info["bid_wall_px"] = obj_attr(ob, "largest_bid_wall_price", obj_attr(ob, "bid_wall_px")) if ob_same_quote else None
        m_info["ask_wall_px"] = obj_attr(ob, "largest_ask_wall_price", obj_attr(ob, "ask_wall_px")) if ob_same_quote else None
    if deriv is not None:
        m_info["funding_rate"] = obj_attr(deriv, "funding_rate")
        m_info["pred_funding_rate"] = obj_attr(deriv, "predicted_funding_rate", obj_attr(deriv, "pred_funding_rate"))
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
            ("divergence_regime", "divergence_regime", None),
            ("taker_ratio_5m", "taker_ratio_5m", None),
            ("taker_ratio_1h", "taker_ratio_1h", None),
            ("taker_ratio_4h", "taker_ratio_4h", None),
            ("cvd_5m_usd", "cvd_5m_usd", None),
            ("cvd_1h_usd", "cvd_1h_usd", None),
            ("cvd_4h_usd", "cvd_4h_usd", None),
        ):
            val_w = obj_attr(whale, src_k, obj_attr(whale, alt_k) if alt_k else None)
            if val_w is not None or dst_k not in m_info:
                m_info[dst_k] = val_w
        if not safe_pos_float(m_info.get("oi_usd"), default=None) and (w_oi := safe_pos_float(obj_attr(whale, "open_interest_usd", obj_attr(whale, "oi_usd")), default=None)) is not None:
            m_info["oi_usd"] = w_oi
        if m_info.get("funding_rate") is None and (w_fr := safe_float(obj_attr(whale, "funding_rate"), default=None)) is not None:
            m_info["funding_rate"] = w_fr
    if m_info.get("funding_annualized_pct") is None and (fr_v := safe_float(m_info.get("funding_rate"), default=None)) is not None:
        m_info["funding_annualized_pct"] = round(fr_v * 3.0 * 365.0 * 100.0, 2)
    for usd_k in ("oi_usd", "taker_buy_vol_usd", "taker_sell_vol_usd", "cvd_5m_usd", "cvd_1h_usd", "cvd_4h_usd"):
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
    if "PYTEST_CURRENT_TEST" in os.environ or not cache_dir or not os.path.isdir(cache_dir):
        return
    sym_u = str(symbol or "").strip().upper()
    if not sym_u:
        return
    eff_now = safe_pos_float(now_ts, default=None) or _time.time()

    def _needs_fetch(prefix: str, max_age: Optional[float]) -> bool:
        fpath = str(resolve_cachedb_filepath(os.path.join(cache_dir, f"{prefix}{sym_u}.json")))
        if not os.path.exists(fpath):
            return True
        return max_age is not None and (eff_now - os.path.getmtime(fpath)) > max_age

    if _needs_fetch("orderbook_depth_", max_age_ob_sec):
        try:
            from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector

            OrderbookDepthCollector(cache_dir=cache_dir).fetch(sym_u)
        except Exception:
            pass
    if include_derivatives and _needs_fetch("derivatives_telemetry_", None):
        try:
            from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector

            DerivativesTelemetryCollector(cache_dir=cache_dir).fetch(sym_u)
        except Exception:
            pass
    if _needs_fetch("whale_snapshot_", max_age_whale_sec):
        try:
            from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector

            WhalePositioningCollector(cache_dir=cache_dir).fetch(sym_u)
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
    covered_micro_bases = {canonical_base_asset(k) for k in microstructure}
    ref_micro_syms: List[str] = []
    for pfx in ("orderbook_depth_", "derivatives_telemetry_", "whale_snapshot_"):
        for search_dir in (cache_dir, os.path.join(cache_dir, "telemetry")):
            for mpath in sorted(_glob.glob(os.path.join(search_dir, f"{pfx}*.json"))):
                msym = os.path.basename(mpath)[len(pfx) : -len(".json")].upper()
                if msym and msym.isalnum() and msym not in microstructure and msym not in ref_micro_syms:
                    mbase = canonical_base_asset(msym)
                    if mbase not in covered_micro_bases:
                        ref_micro_syms.append(msym)
                        covered_micro_bases.add(mbase)
                    elif pfx == "orderbook_depth_" and isinstance(ob_d := _rj(mpath), dict):
                        cand_bd = safe_nonneg_float(ob_d.get("bid_depth_usd"), default=None)
                        existing_bds = [
                            safe_nonneg_float(mv.get("bid_depth_usd"), default=None)
                            for mk, mv in microstructure.items()
                            if canonical_base_asset(mk) == mbase and isinstance(mv, dict)
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
    _rj = _resolve_read_json_cached(read_json_fn)
    btc_series_24h, _ = load_symbol_price_series(cache_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False)
    bcache_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "venue_balances_state.json"))) if cache_dir else None
    raw_cb = _rj(bcache_path) if bcache_path and os.path.exists(bcache_path) else None
    cached_bals: Dict[str, Any] = raw_cb if isinstance(raw_cb, dict) else {}
    for _, ticker in _iter_t212_state_files(workspace_dir):
        cached_pos = cached_bals.get(f"pos:t212:{ticker}")
        cached_t212_px = safe_pos_float(cached_pos.get("price") if isinstance(cached_pos, dict) else None, default=None)
        t_entry = prices_and_trends.setdefault(ticker, {"price": cached_t212_px, "trend": 0, "growth_coefficient": 0.0})
        if cached_t212_px is not None and (
            ticker not in inst_by_sym or (safe_pos_float(t_entry.get("price"), default=0.0) or 0.0) <= 0.0
        ):
            t_entry["price"] = cached_t212_px
    from instruments_config import discover_hyperliquid_spot

    for hl_sym, hl_base, hl_quote in discover_hyperliquid_spot(workspace_dir):
        ensure_fresh_symbol_microstructure(cache_dir, hl_sym, now_ts=now_ts, include_derivatives=True)
        if hl_sym not in prices_and_trends and (
            any(canonical_base_asset(k) == hl_base for k in prices_and_trends)
            or load_symbol_microstructure(cache_dir, hl_sym, _rj)
        ):
            cached_hl_pos = cached_bals.get(f"pos:hyperliquid:{hl_sym}")
            hl_px = safe_pos_float(cached_hl_pos.get("price") if isinstance(cached_hl_pos, dict) else None, default=0.0) or resolve_symbol_price(hl_sym, prices_and_trends, explicit_quote=hl_quote)
            prices_and_trends[hl_sym] = {"price": hl_px if hl_px > 0 else None, "trend": 0, "growth_coefficient": 0.0}
    for sym in list(prices_and_trends.keys()):
        sym_px_cur = safe_float(prices_and_trends[sym].get("price"), 0.0)
        matched_series_24h, arch_vals_7d = load_symbol_price_series(cache_dir, sym, current_price_items=cur_p_items)
        if sym_px_cur > 0 and matched_series_24h:
            last_24_px = safe_float(matched_series_24h[-1][1], 0.0)
            if _prices_diverged(sym_px_cur, last_24_px):
                matched_series_24h = []
            elif last_24_px > 0 and abs(sym_px_cur - last_24_px) > 1e-9:
                matched_series_24h = list(matched_series_24h) + [[int(now_ts * 1000), sym_px_cur]]
        if sym_px_cur > 0 and arch_vals_7d and _prices_diverged(sym_px_cur, safe_float(arch_vals_7d[-1], 0.0)):
            arch_vals_7d = []
        if safe_float(prices_and_trends[sym].get("price"), 0.0) <= 0.0:
            if matched_series_24h:
                prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
            elif arch_vals_7d:
                prices_and_trends[sym]["price"] = arch_vals_7d[-1]
        s24_p0 = _parse_ts_px_row(matched_series_24h[0]) if len(matched_series_24h) >= 2 else None
        s24_p1 = _parse_ts_px_row(matched_series_24h[-1]) if len(matched_series_24h) >= 2 else None
        span_24h_ms = max(0.0, s24_p1[0] - s24_p0[0]) if (s24_p0 and s24_p1) else 0.0
        m24: Dict[str, Any] = compute_24h_price_metrics(matched_series_24h) or {} if matched_series_24h else {}
        if arch_vals_7d and len(arch_vals_7d) >= 2 and "chg_7d_pct" not in prices_and_trends[sym]:
            prices_and_trends[sym].update(compute_7d_range_metrics(arch_vals_7d))
        ohlc_fn = getattr(getattr(inst_by_sym.get(sym), "_provider", None), "ohlc_series", None)
        if not callable(ohlc_fn) and "PYTEST_CURRENT_TEST" not in os.environ:
            sym_u = sym.upper()
            try:
                if sym_u in ("ADAUSD", "TAOUSD", "HYPEUSD"):
                    from providers.kraken_provider import KrakenProvider

                    ohlc_fn = KrakenProvider().ohlc_series
                elif sym_u == "HYPEUSDC":
                    from providers.hyperliquid_provider import HyperliquidProvider

                    ohlc_fn = HyperliquidProvider().ohlc_series
                elif sym_u.endswith(("USDC", "USDT")):
                    from providers.market_api import api as _mkt_api

                    ohlc_fn = getattr(_mkt_api.provider_by_name("binance"), "ohlc_series", None)
            except Exception:
                ohlc_fn = None
        if callable(ohlc_fn):
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
                    if "chg_7d_pct" not in prices_and_trends[sym] or (0.0 < span_24h_ms < 18 * 3600 * 1000):
                        s60, cutoff_7d = ohlc_fn(sym, 60), now_ts - 7 * 86400.0
                        s60_ts = [
                            ts_v for raw_ts in (getattr(s60, "timestamps", None) or ())
                            if (ts_v := safe_pos_float(raw_ts, default=None)) is not None
                        ]
                        vals_7d = _filter_ohlc_closes_since(s60, cutoff_7d)
                        if s60_ts and (s60_ts[-1] - s60_ts[0]) < 5 * 86400.0:
                            try:
                                s240 = ohlc_fn(sym, 240)
                                first_s60 = s60_ts[0] - 1800.0
                                older_7d = _filter_ohlc_closes_since(s240, cutoff_7d, max_ts=first_s60)
                                if older_7d:
                                    vals_7d = older_7d + vals_7d
                            except Exception:
                                pass
                        if len(vals_7d) >= 2 and ("chg_7d_pct" not in prices_and_trends[sym] or len(vals_7d) >= 24):
                            prices_and_trends[sym].update(compute_7d_range_metrics(vals_7d))
                except Exception:
                    pass
            if m24.get("vp_source") == "TPO":
                try:
                    pts_vp = _load_stitched_24h_ohlc_points(ohlc_fn, sym, now_ts - 86400.0, require_positive_volume=True)
                    if len(pts_vp) >= 8 and (pts_vp[-1][0] - pts_vp[0][0]) >= 18 * 3600 * 1000:
                        if sym_px_cur > 0 and not _prices_diverged(sym_px_cur, pts_vp[-1][1]):
                            pts_vp.append([int(now_ts * 1000), sym_px_cur, 0.0])
                        if (vp_real := compute_volume_profile_metrics(pts_vp)) and vp_real.get("vp_source") == "REAL_VOL":
                            m24.update(vp_real)
                except Exception:
                    pass
        is_t212_inst = sym in inst_by_sym and normalize_venue_name(getattr(inst_by_sym[sym], "provider_name", "")) == "t212"
        if not m24 and (is_t212_inst or (sym.endswith(("_EQ", "_US_EQ")) and "PYTEST_CURRENT_TEST" not in os.environ)):
            try:
                _chart = load_local_module(os.path.join(workspace_dir, "212trading"), "market_data")._chart
                chart_res = _chart(sym.split("_", 1)[0], rng="5d", interval="5m")
                pts_5d: List[List[Any]] = []
                if isinstance(chart_res, tuple) and len(chart_res) >= 2 and isinstance(chart_res[1], list):
                    pts_5d = [
                        ([int(ts_f * 1000), round(px_f, 4), safe_nonneg_float(bar[2], 0.0) or 0.0] if len(bar) >= 3 else [int(ts_f * 1000), round(px_f, 4)])
                        for bar in chart_res[1]
                        if isinstance(bar, (list, tuple)) and len(bar) >= 2 and (ts_f := safe_pos_float(bar[0], default=None)) is not None and (px_f := safe_pos_float(bar[1], default=None)) is not None
                    ]
                elif isinstance(chart_res, dict):
                    q_obj = ((chart_res.get("indicators") or {}).get("quote") or [{}])[0]
                    vols = q_obj.get("volume") or []
                    pts_5d = [
                        [int(ts_f * 1000), round(c_f, 4), safe_nonneg_float(vols[idx], 0.0) or 0.0 if idx < len(vols) else 0.0]
                        for idx, (ts, c) in enumerate(zip(chart_res.get("timestamp") or [], q_obj.get("close") or []))
                        if (ts_f := safe_pos_float(ts, default=None)) is not None and (c_f := safe_pos_float(c, default=None)) is not None
                    ]
                if pts_5d:
                    chart_px = safe_pos_float(pts_5d[-1][1], default=0.0) or 0.0
                    if chart_px > 0.0 and (not is_t212_inst or sym_px_cur <= 0.0 or abs(sym_px_cur - chart_px) < 1e-4):
                        prices_and_trends[sym]["price"] = chart_px
                        sym_px_cur = chart_px
                    cutoff_24h_ms = int((now_ts - 86400.0) * 1000)
                    pts_24h = [pt for pt in pts_5d if pt[0] >= cutoff_24h_ms]
                    if len(pts_24h) < 10:
                        pts_24h = list(pts_5d[-78:])
                    if pts_24h and sym_px_cur > 0 and not _prices_diverged(sym_px_cur, pts_24h[-1][1]) and abs(round(sym_px_cur, 4) - round(pts_24h[-1][1], 4)) > 1e-4:
                        now_ms_i = int(now_ts * 1000)
                        if (now_ms_i - pts_24h[-1][0]) <= 20 * 60 * 1000:
                            pts_24h = list(pts_24h) + [[now_ms_i, round(sym_px_cur, 4)]]
                        else:
                            pts_24h = list(pts_24h[:-1]) + [[pts_24h[-1][0], round(sym_px_cur, 4), *pts_24h[-1][2:]]]
                    if m24 := (compute_24h_price_metrics(pts_24h) or {}):
                        matched_series_24h = pts_24h
                    vals_5d = [pt[1] for pt in pts_5d]
                    if len(vals_5d) >= 2 and "chg_7d_pct" not in prices_and_trends[sym]:
                        prices_and_trends[sym].update(compute_7d_range_metrics(vals_5d, decimals=4))
            except Exception:
                pass
        if m24:
            _apply_24h_series_enrichment(sym, prices_and_trends[sym], m24, matched_series_24h, btc_series_24h)

        if sym.upper() in ("HYPEUSD", "TAOUSD", "ADAUSD"):
            ensure_fresh_symbol_microstructure(
                cache_dir, sym.upper(), now_ts=now_ts, max_age_ob_sec=180.0, max_age_whale_sec=360.0
            )
        if m_info := load_symbol_microstructure(cache_dir, sym, _rj):
            is_usd_sym = "USD" in compatible_quote_suffixes(sym)
            sym_px_m = safe_float((prices_and_trends.get(sym) or {}).get("price"), 0.0)
            ob_mid_m = safe_pos_float(m_info.get("ob_mid_price"), default=None)
            src_cand_u = str(m_info.get("_source_cand") or sym).strip().upper()
            is_exact_ob_sym = src_cand_u == str(sym).strip().upper()
            cross_venue_collision = (not is_exact_ob_sym) and (src_cand_u in inst_by_sym) and (sym in inst_by_sym)
            if is_usd_sym and isinstance(prices_and_trends.get(sym), dict) and ob_mid_m is not None and not cross_venue_collision:
                if sym_px_m <= 0.0 or (abs(sym_px_m - ob_mid_m) / max(sym_px_m, ob_mid_m) <= 0.01):
                    prices_and_trends[sym]["price"] = ob_mid_m
                    sym_px_m = ob_mid_m
            elif not is_usd_sym and (sym_px_m <= 0.0 or _prices_diverged(sym_px_m, ob_mid_m)):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    m_info.pop(k_strip, None)
            enrich_microstructure_cvd(m_info, prices_and_trends.get(sym))
            m_info.update(compute_liquidation_clusters(prices_and_trends.get(sym), m_info))
            if m_info.get("spread_bps") is not None and isinstance(prices_and_trends.get(sym), dict):
                prices_and_trends[sym]["spread_bps"] = m_info["spread_bps"]
            microstructure[sym] = m_info

    for msym in discover_reference_microstructure_symbols(cache_dir, microstructure, _rj):
        if m_info_ref := load_symbol_microstructure(cache_dir, msym, _rj):
            msym_quotes = compatible_quote_suffixes(msym)
            p_ref = _first_variant_dict(
                prices_and_trends,
                cache_symbol_variants(msym, include_original=True, include_base="USD" in msym_quotes, quotes=msym_quotes),
            )
            if p_ref is not None and msym not in prices_and_trends:
                ob_mid_sib = safe_pos_float(m_info_ref.get("ob_mid_price"), default=None)
                if ob_mid_sib is not None and not _prices_diverged(ob_mid_sib, safe_float(p_ref.get("price"), 0.0)):
                    p_sib = dict(p_ref)
                    p_sib["price"] = ob_mid_sib
                    if m_info_ref.get("spread_bps") is not None:
                        p_sib["spread_bps"] = m_info_ref["spread_bps"]
                    prices_and_trends[msym] = p_sib
                    p_ref = p_sib
            elif p_ref is None:
                ref_px = safe_pos_float(m_info_ref.get("ob_mid_price"), default=None)
                ser_24h, vals_7d = load_symbol_price_series(cache_dir, msym, current_price_items=cur_p_items)
                if not ser_24h and msym.endswith("USD") and "PYTEST_CURRENT_TEST" not in os.environ:
                    try:
                        from providers.kraken_provider import KrakenProvider

                        krk_p = KrakenProvider()
                        ser_24h = _extract_ohlc_series_points(krk_p.ohlc_series(msym, 5), min_ts=now_ts - 86400.0)
                        if ref_px is not None and ser_24h and abs(ref_px - float(ser_24h[-1][1])) > 1e-9 and not _prices_diverged(ref_px, float(ser_24h[-1][1])):
                            ser_24h.append([int(now_ts * 1000), ref_px])
                        if len(vals_7d) < 2:
                            s60, cutoff_7d = krk_p.ohlc_series(msym, 60), now_ts - 7 * 86400.0
                            vals_7d = _filter_ohlc_closes_since(s60, cutoff_7d)
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
    "round_opt_px",
    "update_microstructure_cycle_deltas",
]
