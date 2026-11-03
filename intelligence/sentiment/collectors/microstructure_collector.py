"""Spot/perpetual microstructure, 24h/7d price series loader, and reference symbol discovery collector."""
from __future__ import annotations

import glob as _glob
import logging
import os
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from botcore import extract_current_price_item
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    legacy_filenames_for,
)
from intelligence.external.collectors.orderbook_depth import round_spread_bps
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
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _WINDOW_TREND_SPECS,
    _auto_px_decimals,
    _build_vp_position_dict,
    _compute_horizon_range_dict,
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
    read_json_dict_recovering as read_json_cached,
    read_tail_jsonl_records,
    resolve_cachedb_filepath,
)

logger = logging.getLogger("intelligence.sentiment.microstructure_collector")


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
    p0: Optional[Tuple[float, float]] = None
    p1: Optional[Tuple[float, float]] = None
    span_24h_ms = 0.0
    for cand in cands:
        for p24_path in _cand_paths(f"cache_24_price_{cand}.json"):
            if not os.path.exists(p24_path) and not os.path.exists(p24_path + ".previous"):
                continue
            ser = _extract_cached_items_series(read_json_cached(p24_path), cands)
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
                aser = _extract_cached_items_series(read_json_cached(arch_path), cands)
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
    read_json_fn = read_json_cached if read_json_fn is None else read_json_fn
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
            from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

            OrderbookDepthCollector(cache_dir=cache_dir).fetch(sym_u)
        except Exception:
            pass
    if include_derivatives and _needs_fetch("derivatives_telemetry_", None):
        try:
            from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector

            DerivativesTelemetryCollector(cache_dir=cache_dir).fetch(sym_u)
        except Exception:
            pass
    if _needs_fetch("whale_snapshot_", max_age_whale_sec):
        try:
            from intelligence.external.collectors.whale_positioning import WhalePositioningCollector

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
    _rj = read_json_cached if read_json_fn is None else read_json_fn
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


__all__ = [
    "_MICRO_PRICE_ANCHOR_KEYS",
    "_VENUE_DEFAULT_FEE_SPECS",
    "_WINDOW_TREND_SPECS",
    "_auto_px_decimals",
    "_blend_structural_liq_anchor",
    "_build_vp_position_dict",
    "_compute_horizon_range_dict",
    "_first_variant_dict",
    "_infer_symbol_venue_and_fees_bps",
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
