"""Portfolio positions, venue DCA cycles, trailing peaks, recent fills, and cross-venue trend collector."""
from __future__ import annotations

import glob as _glob
import logging
import math
import os
import re
import sys as _sys
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from botcore import load_key_value_conf, load_local_module
from cachepaths import (
    cache_symbol_variants, canonical_base_asset, compatible_quote_suffixes,
    normalize_venue_name, resolve_kraken_lookup_symbol, split_symbol_assets,
)
from intelligence.external.collectors.orderbook_depth import round_spread_bps
from intelligence.internal.state.volatility import adaptive_thresholds, calculate_volatility_1h
from intelligence.internal.triggers.kalman_trigger import KalmanTrend
from intelligence.macro_sentiment_ssot import (
    clean_symbol_key, lookup_symbol_cache_items, safe_float, safe_nonneg_float,
    safe_pos_float, scan_latest_long_trend_value,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    _MICRO_PRICE_ANCHOR_KEYS, _build_vp_position_dict, _first_variant_dict,
    _infer_symbol_venue_and_fees_bps, _parse_ts_px_row, _prices_diverged,
    apply_short_trend_item, compute_24h_price_metrics, compute_7d_range_metrics,
    compute_cross_btc_metrics, compute_cross_venue_arb_metrics, compute_liquidation_clusters,
    compute_volume_profile_metrics, discover_reference_microstructure_symbols,
    enrich_microstructure_cvd, ensure_fresh_symbol_microstructure, extract_symbol_recent_fills,
    load_symbol_microstructure, load_symbol_price_series, round_opt_px,
    update_microstructure_cycle_deltas,
)
from intelligence.sentiment.collectors.portfolio_risk_tca import (
    _VENUE_DEFAULT_FEE_SPECS,
    _load_audit_terminal_orders,
    _load_open_order_timestamps,
    _load_recent_audit_origins,
    _normalize_open_orders_list,
    _populate_cycle_fallback_recent_order,
    _sync_cycle_wallet_qty_and_avg,
    compute_enterprise_portfolio_risk_metrics,
    enrich_open_limit_orders_tca,
    enrich_recent_orders_markout,
    parse_venue_cycle_state,
    reconcile_venue_cycles_with_recent_orders,
    resolve_symbol_price,
    resolve_venue_fee_and_limits,
    set_pos_anchor,
)
from intelligence.sentiment.collectors.quant_price_evaluators import (
    _apply_24h_series_enrichment,
    _auto_px_decimals,
    _compute_horizon_range_dict,
    _compute_kalman_from_series,
    _compute_short_slopes_from_series,
    _extract_ohlc_series_points,
    _load_stitched_24h_ohlc_points,
    pct_delta,
    recalculate_live_price_geometry,
)
from instrument_registry import casefold_get
from providers.base import extract_order_id, extract_order_side, normalize_open_order, normalize_timestamp_ms, resolve_order_origin
from providers.quantity import resolve_assets, venue_fee_cap_quantity
from state_io import (
    read_json_dict_recovering as read_json_cached, read_json_file, read_tail_jsonl_dicts,
    resolve_cachedb_filepath, save_json_state,
)

logger = logging.getLogger("intelligence.sentiment.portfolio_collector")


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


def harvest_venue_cycles(
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Dict[str, float],
    read_json_fn: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest normalized ``.state_*.json`` DCA cycle states for Kraken, Hyperliquid, and Trading212."""
    def _record_cyc(target: Dict[str, Any], st_path: str, sym: str, dec: int) -> None:
        if isinstance((st_data := read_json_fn(st_path)), dict):
            target[sym] = parse_venue_cycle_state(
                st_data, sym, prices_and_trends, avg_decimals=dec, fallback_prices=fallback_prices, include_meta=True
            )

    kraken_cycles: Dict[str, Any] = {}
    if os.path.isdir(kraken_dir := os.path.join(workspace_dir, "kraken")):
        for st_path in sorted(_glob.glob(os.path.join(kraken_dir, ".state_*.json"))):
            pair_name = os.path.basename(st_path)[len(".state_") : -len(".json")].upper()
            if pair_name.isalnum() and pair_name.endswith(("USD", "EUR", "USDC", "USDT")):
                _record_cyc(kraken_cycles, st_path, pair_name, 6)

    hyperliquid_cycles: Dict[str, Any] = {}
    if os.path.isdir(hl_dir := os.path.join(workspace_dir, "hyperliquid")):
        hl_state_paths = sorted(
            _glob.glob(os.path.join(hl_dir, ".state_*.json")),
            key=lambda p: (0 if os.path.basename(p).upper().endswith(("_LONG.JSON", "_SHORT.JSON")) else 1, os.path.basename(p)),
        )
        for st_path in hl_state_paths:
            raw_tok = os.path.basename(st_path)[len(".state_") : -len(".json")].upper()
            if raw_tok.startswith("DN_") or raw_tok == "DN":
                continue
            for sfx in ("_LONG", "_SHORT"):
                if raw_tok.endswith(sfx):
                    raw_tok = raw_tok[: -len(sfx)]
                    break
            if raw_tok.isalnum():
                _record_cyc(hyperliquid_cycles, st_path, raw_tok if raw_tok.endswith("USDC") else f"{raw_tok}USDC", 6)

    all_t212_cycles: Dict[str, Any] = {}
    for st_path, ticker in _iter_t212_state_files(workspace_dir):
        _record_cyc(all_t212_cycles, st_path, ticker, 4)

    return kraken_cycles, hyperliquid_cycles, all_t212_cycles


def harvest_kraken_xstocks(
    kraken_dir: str,
    prices_and_trends: Dict[str, Any],
    all_t212_cycles: Dict[str, Any],
    read_json_fn: Any,
) -> Optional[Dict[str, Any]]:
    """Harvest Kraken read-only xStocks holdings and link to Trading212 sibling cycle."""
    cand_paths = [
        p for p in (os.path.join(kraken_dir, "xstock_state.json"), os.path.join(kraken_dir, ".xstock_state.json"))
        if os.path.exists(p)
    ]
    if not cand_paths:
        return None
    xst_path = max(cand_paths, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
    xst_data = read_json_fn(xst_path)
    if not isinstance(xst_data, dict):
        return None
    xstock_cfg = load_key_value_conf(kraken_dir, root_dir=kraken_dir)
    known_assets = xst_data.get("known_assets") if isinstance(xst_data.get("known_assets"), dict) else {}
    x_regex = xstock_cfg.get("xstock_regex") or "SPCX|SPACEX"
    x_holdings: Dict[str, float] = {}
    for ak, av in known_assets.items():
        ak_str, av_f = str(ak).strip(), safe_float(av, 0.0) or 0.0
        if av_f <= 0:
            continue
        is_xstock_tok = ak_str.upper().endswith(("X.T", ".T"))
        if not is_xstock_tok:
            try:
                is_xstock_tok = bool(re.search(x_regex, ak_str, re.IGNORECASE))
            except re.error:
                is_xstock_tok = False
        if is_xstock_tok:
            x_holdings[ak_str] = round(av_f, 6)
    alloc_info = xst_data.get("allocated") if isinstance(xst_data.get("allocated"), dict) else None
    raw_pair = xst_data.get("pair")
    listed_pair = str(raw_pair).strip().upper() if isinstance(raw_pair, str) and raw_pair.strip() and raw_pair.strip().lower() not in ("none", "null", "nan") else None
    if not (x_holdings or alloc_info or listed_pair):
        return None
    underlying = (xstock_cfg.get("xstock_yahoo") or "SPCX").strip().upper()
    t212_sib = f"{underlying}_US_EQ"
    alloc_px = safe_pos_float(xstock_cfg.get("xstock_alloc_price"), default=None)
    live_px = resolve_symbol_price(listed_pair or underlying, prices_and_trends)
    xst_summary: Dict[str, Any] = {
        "underlying": underlying, "t212_sibling": t212_sib, "holdings": x_holdings,
        "allocated": alloc_info, "listed_pair": listed_pair, "alloc_price": alloc_px,
        "tp1_alert_pct": safe_pos_float(xstock_cfg.get("xstock_tp_alert_pct"), default=None),
        "tp2_alert_pct": safe_pos_float(xstock_cfg.get("xstock_tp2_alert_pct"), default=None),
        "sl_alert_pct": safe_pos_float(xstock_cfg.get("xstock_sl_alert_pct"), default=None),
    }
    if live_px > 0:
        xst_summary["live_price"] = live_px
        if alloc_px is not None:
            xst_summary["pnl_vs_alloc_pct"] = round(((live_px - alloc_px) / alloc_px) * 100.0, 2)
    if isinstance((sib_cyc := _first_variant_dict(all_t212_cycles, (t212_sib, underlying))), dict):
        xst_summary["t212_sibling_cycle"] = {k: v for k, v in sib_cyc.items() if not str(k).startswith("_")}
    return xst_summary


def build_cross_market_context(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    all_t212_cycles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build raw, objective cross-market telemetry (Crypto <-> US Equities) with zero pre-calculated correlations or bias."""
    all_t212_cycles = all_t212_cycles if isinstance(all_t212_cycles, dict) else {}
    ctx: Dict[str, Any] = {}
    def _bellwether_rank(s: str, p: Dict[str, Any]) -> Tuple[int, int, int]:
        return (1 if "USD" in compatible_quote_suffixes(s) else 0, 1 if s.endswith(("USDC", "_US_EQ", "_EQ")) else 0, len(p))

    by_base: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for sym, pinfo in sorted(prices_and_trends.items()):
        if isinstance(pinfo, dict) and (b_canon := canonical_base_asset(sym)):
            prev = by_base.get(b_canon)
            if prev is None or _bellwether_rank(sym, pinfo) > _bellwether_rank(prev[0], prev[1]):
                by_base[b_canon] = (sym, pinfo)

    raw_horizon_keys = (
        "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
        "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
        "chg_7d_pct", "pos_24h_pct", "poc_px", "vah_px", "val_px", "va_regime",
        "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel",
    )

    crypto_bellwethers: Dict[str, Any] = {}
    for b_canon, (sym, pinfo) in sorted(by_base.items()):
        if sym.endswith(("_EQ", "_US_EQ")) or (px_f := safe_float(pinfo.get("price"), 0.0) or 0.0) <= 0:
            continue
        cb_entry: Dict[str, Any] = {
            "symbol": sym, "price": px_f,
            **{k: pinfo[k] for k in raw_horizon_keys if pinfo.get(k) is not None},
        }
        sym_quotes = compatible_quote_suffixes(sym)
        m_info, m_same_quote = microstructure.get(sym), True
        if not isinstance(m_info, dict):
            m_info, m_same_quote = next(
                ((mv, compatible_quote_suffixes(mk) == sym_quotes) for mk, mv in microstructure.items() if canonical_base_asset(mk) == b_canon and isinstance(mv, dict)),
                (None, False),
            )
        if isinstance(m_info, dict):
            for mk_copy in (
                "ob_imbalance", "spread_bps", "whale_long_pct", "funding_rate", "oi_1h_chg_pct",
                "cvd_15m_usd", "cvd_regime", "depth_1pct_bid_usd", "depth_1pct_ask_usd",
                *(("long_liq_px", "long_liq_usd", "short_liq_px", "short_liq_usd", "liq_magnet_bias") if m_same_quote else ()),
            ):
                if m_info.get(mk_copy) is not None:
                    cb_entry[mk_copy] = m_info.get(mk_copy)
        crypto_bellwethers[b_canon] = cb_entry
    if crypto_bellwethers:
        ctx["crypto_bellwethers"] = crypto_bellwethers

    equity_bellwethers: Dict[str, Any] = {}
    eq_syms = sorted(
        {s for s in prices_and_trends if str(s).upper().endswith(("_EQ", "_US_EQ"))}
        | {s for s in all_t212_cycles if str(s).upper().endswith(("_EQ", "_US_EQ"))}
    )
    for eq_sym in eq_syms:
        pinfo = prices_and_trends.get(eq_sym) if isinstance(prices_and_trends.get(eq_sym), dict) else {}
        t_cyc = _first_variant_dict(all_t212_cycles, cache_symbol_variants(eq_sym, include_original=True, include_base=True), default={})
        eq_entry: Dict[str, Any] = {
            "symbol": eq_sym,
            **({"price": pinfo["price"]} if pinfo.get("price") is not None else {}),
            **{k: pinfo[k] for k in raw_horizon_keys if pinfo.get(k) is not None},
        }
        if t_cyc:
            eq_entry["t212_cycle"] = {k: t_cyc.get(k) for k in ("qty", "avg_cost_px", "entry_price", "last_buy_price", "dd_pct") if t_cyc.get(k) is not None}
        if len(eq_entry) > 1:
            equity_bellwethers[eq_sym] = eq_entry
    if equity_bellwethers:
        ctx["equity_bellwethers"] = equity_bellwethers

    return ctx


def load_venue_trailing_maps(
    cache_dir: str, workspace_dir: str, read_json_fn: Any = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Load trailing peak maps from cachedb/trailing_state.json and kraken/trailing_state.json."""
    _rj = read_json_fn or read_json_cached
    trailing_map: Dict[str, Dict[str, Any]] = {}
    trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]] = {"binance": {}, "kraken": {}}
    for trail_venue, raw_trail, default_quote in (
        ("binance", _rj(os.path.join(cache_dir, "trailing_state.json")), ""),
        ("kraken", _rj(os.path.join(workspace_dir, "kraken", "trailing_state.json")), "USD"),
    ):
        if not isinstance(raw_trail, dict):
            continue
        for k_asset, k_st in raw_trail.items():
            if not (k_up := str(k_asset).strip().upper()):
                continue
            k_key = k_up if (not default_quote or split_symbol_assets(k_up) is not None) else f"{k_up}{default_quote}"
            norm_st = k_st if isinstance(k_st, dict) and k_st.get("peak") is not None else ({"peak": k_pk} if (k_pk := safe_pos_float(k_st, default=0.0) or 0.0) > 0 else None)
            if norm_st is not None:
                trailing_map.setdefault(k_key, norm_st)
                trailing_by_venue[trail_venue].setdefault(k_key, norm_st)
    return trailing_map, trailing_by_venue


def harvest_tradable_positions_and_orders(
    inst_map: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    trailing_map: Dict[str, Dict[str, Any]],
    trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    gateway_max_usd: float = 2000.0,
    cache_dir: Optional[str] = None,
    verified_open_venues: Optional[set[str]] = None,
    allow_empty_inst_recovery: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Harvest spot wallet balances, trailing peaks, and open orders across tradable instruments."""
    tradable_positions: List[Dict[str, Any]] = []
    open_orders_list: List[Dict[str, Any]] = []
    venue_quote_cash: Dict[str, Dict[str, Any]] = {}
    inst_by_sym: Dict[str, Any] = {}
    balance_memo: Dict[tuple[str, str], Optional[float]] = {}
    bcache_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "venue_balances_state.json"))) if cache_dir else None
    raw_cb = read_json_cached(bcache_path) if bcache_path and os.path.exists(bcache_path) else None
    cached_bals: Dict[str, Any] = raw_cb if isinstance(raw_cb, dict) else {}
    updated_bals: Dict[str, Any] = {}
    live_open_ok_venues: set[str] = set()
    failed_open_venues: set[str] = set()
    now_ts_oo = _time.time()
    order_ts_map = _load_open_order_timestamps(cache_dir, cached_bals)

    def _query_free_bal(inst_obj: Any, asset: str) -> Optional[float]:
        if not asset:
            return None
        raw_v = str(getattr(inst_obj, "provider_name", "") or "").strip().lower()
        v_name = normalize_venue_name(raw_v) or raw_v
        key = (v_name, asset.upper())
        if key in balance_memo:
            return balance_memo[key]
        free_fn = getattr(getattr(inst_obj, "_provider", None), "free_balance", None)
        val_f: Optional[float] = None
        if callable(free_fn):
            try:
                val_f = safe_nonneg_float(free_fn(asset), default=None)
            except Exception:
                val_f = None
        bkey = f"{v_name}:{asset.upper()}"
        raw_bkey = f"{raw_v}:{asset.upper()}"
        if val_f is not None:
            updated_bals[bkey] = round(val_f, 8)
        elif bkey in cached_bals:
            val_f = safe_nonneg_float(cached_bals.get(bkey), default=None)
        elif raw_bkey in cached_bals:
            val_f = safe_nonneg_float(cached_bals.get(raw_bkey), default=None)
        balance_memo[key] = val_f
        return val_f

    pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]] = {}

    if isinstance(inst_map, dict) and inst_map:
        for inst in inst_map.values():
            sym = str(getattr(inst, "symbol", "") or "").strip().upper()
            venue = str(getattr(inst, "provider_name", "") or "").strip().lower()
            if not sym or not venue:
                continue
            v_norm = normalize_venue_name(venue)
            res_b, res_q = resolve_assets(sym, getattr(inst, "base", None), getattr(inst, "quote", None))
            base, quote = res_b or sym, res_q or "USD"
            inst_by_sym[sym] = inst_by_sym[f"{v_norm}:{sym}"] = inst

            px = prices_and_trends.get(sym, {}).get("price")
            px_f = safe_pos_float(px, default=0.0) or 0.0
            if px_f <= 0:
                try:
                    get_px_fn = getattr(inst._provider, "get_current_price", None)
                    if callable(get_px_fn) and (live_px := safe_pos_float(get_px_fn(sym), default=0.0) or 0.0) > 0:
                        px = px_f = live_px
                except Exception:
                    px, px_f = None, 0.0
            if px_f <= 0 and (px_f := resolve_symbol_price(sym, prices_and_trends, explicit_quote=quote)) > 0:
                px = px_f
            sym_entry = prices_and_trends.setdefault(sym, {"price": px if px_f > 0 else None, "trend": 0, "growth_coefficient": 0.0})
            if px_f > 0 and (safe_pos_float(sym_entry.get("price"), default=0.0) or 0.0) <= 0:
                sym_entry["price"] = px

            base_free, quote_free = _query_free_bal(inst, base), _query_free_bal(inst, quote)
            quote_breakdown: Optional[Dict[str, float]] = None
            bd_key = f"breakdown:{v_norm}:{quote.upper()}"
            if quote.upper() == "USD" and callable(bd_fn := getattr(inst._provider, "usd_cash_breakdown", None)):
                try:
                    if isinstance(raw_bd := bd_fn(), dict) and raw_bd:
                        quote_breakdown = {str(k): round(vf, 2) for k, v in raw_bd.items() if (vf := safe_pos_float(v, default=None)) is not None} or None
                        if quote_breakdown:
                            updated_bals[bd_key] = quote_breakdown
                except Exception:
                    quote_breakdown = None
            if quote_breakdown is None and isinstance(cached_bd := cached_bals.get(bd_key), dict) and cached_bd:
                quote_breakdown = {str(k): round(vf, 2) for k, v in cached_bd.items() if (vf := safe_pos_float(v, default=None)) is not None} or None
            if quote_free is not None:
                cash_entry: Dict[str, Any] = {"venue": venue, "asset": quote, "free": round(quote_free, 2)}
                if quote_breakdown:
                    cash_entry["breakdown"] = quote_breakdown
                venue_quote_cash[f"{venue}:{quote}"] = cash_entry
                updated_bals[f"cash:{v_norm}:{quote.upper()}"] = dict(cash_entry)

            pos_val_usd = round(base_free * px_f, 2) if base_free is not None and px_f > 0 else None
            params = getattr(inst, "params", {}) if isinstance(getattr(inst, "params", None), dict) else {}
            sym_quotes = compatible_quote_suffixes(sym, quote)
            t_state = _first_variant_dict(
                trailing_by_venue.get(v_norm, {}),
                cache_symbol_variants(sym, include_original=True, include_base="USD" in sym_quotes, quotes=sym_quotes),
                default={},
            )
            if not t_state and v_norm == "binance" and isinstance((val_b := casefold_get(trailing_map, sym)), dict):
                t_state = val_b
            peak = t_state.get("peak")
            peak_f = safe_float(peak, 0.0)
            if peak_f > 0 and px_f > 0 and peak_f < px_f:
                peak_f = px_f
            dd_from_peak_pct = round(((px_f - peak_f) / peak_f) * 100.0, 2) if px_f > 0 and peak_f > 0 else None
            pos_entry: Dict[str, Any] = {
                "venue": venue, "symbol": sym, "base": base, "quote": quote,
                "price": round_opt_px(px, 6) if px is not None else None,
                "free_base": round(base_free, 6) if base_free is not None else None,
                "value_usd": pos_val_usd,
                "free_quote": round(quote_free, 2) if quote_free is not None else None,
            }
            if quote_breakdown:
                pos_entry["free_quote_breakdown"] = quote_breakdown
            pos_entry.update(
                resolve_venue_fee_and_limits(venue, sym, inst, price=px_f, free_quote=quote_free, free_base=base_free, gateway_max_usd=gateway_max_usd)
            )
            min_notional_f = max(1.0, safe_float(pos_entry.get("exchange_min_notional_usd") or pos_entry.get("min_notional_usd"), 1.0))
            has_held_base = base_free is None or (base_free > 1e-8 and (px_f <= 0.0 or base_free * px_f >= min_notional_f))
            if peak is not None and peak_f > 0:
                if has_held_base:
                    pos_entry["trailing_peak"] = round(peak_f, 6)
                    pos_entry["dd_from_peak_pct"] = dd_from_peak_pct
                else:
                    pending_peaks_by_vs[(v_norm, sym)] = (round(peak_f, 6), dd_from_peak_pct)
            if params.get("trailing.pct"):
                pos_entry["trailing_pct_cfg"] = params.get("trailing.pct")
            if params.get("mt.gain") or params.get("mt.lost"):
                pos_entry["mt_gain_pct"], pos_entry["mt_lost_pct"] = params.get("mt.gain"), params.get("mt.lost")
            tradable_positions.append(pos_entry)
            updated_bals[f"pos:{v_norm}:{sym}"] = dict(pos_entry)

            if callable(open_fn := getattr(inst, "open_orders", None)):
                oo_key = f"open_orders:{v_norm}:{sym}"
                sym_open_ok = False
                try:
                    if isinstance(raw_open := open_fn(), list):
                        sym_open_ok = True
                        sym_norm_orders = _normalize_open_orders_list(
                            raw_open, venue, sym, known_ai_order_ids, now_ts=now_ts_oo, order_ts_map=order_ts_map
                        )
                        open_orders_list.extend(sym_norm_orders)
                        updated_bals[oo_key] = sym_norm_orders
                except Exception:
                    sym_open_ok = False
                if sym_open_ok:
                    live_open_ok_venues.add(v_norm)
                else:
                    failed_open_venues.add(v_norm)
                    open_orders_list.extend(
                        _normalize_open_orders_list(
                            cached_bals.get(oo_key), venue, sym, known_ai_order_ids, now_ts=now_ts_oo, order_ts_map=order_ts_map
                        )
                    )
    elif allow_empty_inst_recovery and cached_bals:
        gw_max_rec = safe_pos_float(gateway_max_usd, default=2000.0) or 2000.0
        for k, v in cached_bals.items():
            if k.startswith("pos:") and isinstance(v, dict):
                rec_pos = dict(v)
                for stale_k in ("locked_quote_usd", "replace_max_buy_usd", "locked_base_qty", "locked_base_usd", "replace_max_sell_usd"):
                    rec_pos.pop(stale_k, None)
                p_sym = str(rec_pos.get("symbol") or "").strip().upper()
                p_quote = str(rec_pos.get("quote") or "USD").strip().upper()
                v_rec_norm = normalize_venue_name(rec_pos.get("venue"))
                fb_val = safe_nonneg_float(rec_pos.get("free_base"), default=None)
                px_rec = 0.0
                if p_sym:
                    exact_pt_px = safe_pos_float((prices_and_trends.get(p_sym) or {}).get("price"), default=0.0) or 0.0
                    cached_pos_px = safe_pos_float(rec_pos.get("price"), default=0.0) or 0.0
                    sib_pt_px = resolve_symbol_price(p_sym, prices_and_trends, explicit_quote=p_quote) if exact_pt_px <= 0.0 else exact_pt_px
                    if exact_pt_px > 0.0:
                        px_rec = exact_pt_px
                    elif cached_pos_px > 0.0 and (sib_pt_px <= 0.0 or not _prices_diverged(cached_pos_px, sib_pt_px)):
                        px_rec = cached_pos_px
                    else:
                        px_rec = sib_pt_px
                    if px_rec > 0:
                        rec_pos["price"] = round_opt_px(px_rec, 6)
                        if fb_val is not None:
                            rec_pos["value_usd"] = round(fb_val * px_rec, 2)
                            rec_pos["effective_max_sell_usd"] = round(
                                min(gw_max_rec, venue_fee_cap_quantity(v_rec_norm, "SELL", px_rec, fb_val) * px_rec), 2
                            )
                        prices_and_trends.setdefault(p_sym, {"price": px_rec, "trend": 0, "growth_coefficient": 0.0})
                    else:
                        px_rec = cached_pos_px
                rec_peak = safe_pos_float(rec_pos.get("trailing_peak"), default=None)
                if rec_peak is None and p_sym:
                    v_trail_map = trailing_by_venue.get(v_rec_norm, {}) if isinstance(trailing_by_venue, dict) else {}
                    t_cands = cache_symbol_variants(p_sym, include_original=True, include_base=True, quotes=compatible_quote_suffixes(p_sym))
                    t_dict = _first_variant_dict(v_trail_map, t_cands) or (
                        _first_variant_dict(trailing_map, t_cands) if v_rec_norm in ("binance", "kraken") else None
                    )
                    if isinstance(t_dict, dict):
                        rec_peak = safe_pos_float(t_dict.get("peak"), default=None)
                if rec_peak is not None:
                    eff_rec_peak = max(rec_peak, px_rec) if px_rec > 0 else rec_peak
                    rec_dd = round(((px_rec - eff_rec_peak) / eff_rec_peak) * 100.0, 2) if px_rec > 0 and eff_rec_peak > 0 else rec_pos.get("dd_from_peak_pct")
                    min_not_rec = max(1.0, safe_float(rec_pos.get("exchange_min_notional_usd") or rec_pos.get("min_notional_usd"), 1.0))
                    has_held_rec = fb_val is None or (fb_val > 1e-8 and (px_rec <= 0.0 or fb_val * px_rec >= min_not_rec))
                    if has_held_rec:
                        rec_pos["trailing_peak"] = round(eff_rec_peak, 6)
                        rec_pos["dd_from_peak_pct"] = rec_dd
                    else:
                        rec_pos.pop("trailing_peak", None)
                        rec_pos.pop("dd_from_peak_pct", None)
                        if v_rec_norm and p_sym:
                            pending_peaks_by_vs[(v_rec_norm, p_sym)] = (round(eff_rec_peak, 6), rec_dd)
                tradable_positions.append(rec_pos)
            elif k.startswith("cash:") and isinstance(v, dict):
                c_venue, c_asset = str(v.get("venue") or "").strip().lower(), str(v.get("asset") or "").strip().upper()
                if c_venue and c_asset:
                    rec_cash = dict(v)
                    rec_cash.pop("locked_quote_usd", None)
                    venue_quote_cash[f"{c_venue}:{c_asset}"] = rec_cash
            elif k.startswith("open_orders:") and isinstance(v, list):
                parts = k.split(":", 2)
                open_orders_list.extend(
                    _normalize_open_orders_list(
                        v,
                        parts[1] if len(parts) > 1 else "",
                        parts[2] if len(parts) > 2 else "",
                        known_ai_order_ids,
                        now_ts=now_ts_oo,
                        order_ts_map=order_ts_map,
                    )
                )

    if isinstance(verified_open_venues, set):
        verified_open_venues.update(live_open_ok_venues - failed_open_venues)

    if open_orders_list:
        locked_quote_by_vq: Dict[tuple[str, str], float] = {}
        locked_base_by_vs: Dict[tuple[str, str], float] = {}
        for oo_item in open_orders_list:
            if not isinstance(oo_item, dict):
                continue
            oo_v = normalize_venue_name(oo_item.get("venue"))
            oo_sym = str(oo_item.get("symbol") or "").strip().upper()
            oo_side = str(oo_item.get("side") or "").strip().upper()
            oo_px = safe_pos_float(oo_item.get("price"), default=0.0) or 0.0
            oo_rem = safe_pos_float(oo_item.get("remaining_qty"), default=None) or safe_pos_float(oo_item.get("qty"), default=0.0) or 0.0
            if oo_item.get("created_ts") is None:
                oo_item["created_ts"] = round(now_ts_oo, 3)
                oo_item["age_min"] = 0.0
                oo_item["age"] = "0.0m"
            if oo_px > 0.0 and oo_sym and oo_item.get("dist_from_spot_pct") is None:
                spot_oo = safe_pos_float((prices_and_trends.get(oo_sym) or {}).get("price"), default=0.0) or resolve_symbol_price(oo_sym, prices_and_trends)
                if spot_oo > 0.0:
                    oo_item["dist_from_spot_pct"] = round(((oo_px - spot_oo) / spot_oo) * 100.0, 2)
            if not oo_v or not oo_sym or oo_rem <= 0.0:
                continue
            _, oo_quote = resolve_assets(oo_sym, None, None)
            oo_q_u = (oo_quote or "USD").strip().upper()
            if oo_side == "BUY" and oo_px > 0.0:
                locked_quote_by_vq[(oo_v, oo_q_u)] = locked_quote_by_vq.get((oo_v, oo_q_u), 0.0) + oo_rem * oo_px
            elif oo_side == "SELL":
                locked_base_by_vs[(oo_v, oo_sym)] = locked_base_by_vs.get((oo_v, oo_sym), 0.0) + oo_rem

        for cash_val in venue_quote_cash.values():
            if isinstance(cash_val, dict):
                cv_norm, cq_u = normalize_venue_name(cash_val.get("venue")), str(cash_val.get("asset") or "").strip().upper()
                if (lq_val := locked_quote_by_vq.get((cv_norm, cq_u), 0.0)) > 0.0:
                    cash_val["locked_quote_usd"] = round(lq_val, 2)
                    cash_bkey = f"cash:{cv_norm}:{cq_u}"
                    if cash_bkey in updated_bals:
                        updated_bals[cash_bkey] = dict(cash_val)

        for pos_entry in tradable_positions:
            p_v_norm = normalize_venue_name(pos_entry.get("venue"))
            p_sym_u = str(pos_entry.get("symbol") or "").strip().upper()
            p_quote_u = str(pos_entry.get("quote") or "USD").strip().upper()
            lq_pos = locked_quote_by_vq.get((p_v_norm, p_quote_u), 0.0)
            lb_pos = locked_base_by_vs.get((p_v_norm, p_sym_u), 0.0)
            if lq_pos > 0.0 or lb_pos > 0.0:
                inst_obj = inst_by_sym.get(f"{p_v_norm}:{p_sym_u}") or inst_by_sym.get(p_sym_u)
                px_pos = safe_pos_float(pos_entry.get("price"), default=0.0) or 0.0
                fb_pos = safe_nonneg_float(pos_entry.get("free_base"), default=None)
                prev_inst_limits = (
                    {k: pos_entry[k] for k in ("exchange_min_notional_usd", "min_qty", "step_size", "tick_size", "spread_bps") if pos_entry.get(k) is not None}
                    if inst_obj is None
                    else {}
                )
                pos_entry.update(
                    resolve_venue_fee_and_limits(
                        str(pos_entry.get("venue") or ""),
                        p_sym_u,
                        inst_obj,
                        price=px_pos,
                        free_quote=safe_nonneg_float(pos_entry.get("free_quote"), default=None),
                        free_base=fb_pos,
                        locked_quote_usd=lq_pos if lq_pos > 0.0 else None,
                        locked_base_qty=lb_pos if lb_pos > 0.0 else None,
                        gateway_max_usd=gateway_max_usd,
                    )
                )
                if prev_inst_limits:
                    pos_entry.update(prev_inst_limits)
                if lb_pos > 1e-8 and "trailing_peak" not in pos_entry and (pend_peak := pending_peaks_by_vs.get((p_v_norm, p_sym_u))) is not None:
                    min_not_pos = max(1.0, safe_float(pos_entry.get("exchange_min_notional_usd") or pos_entry.get("min_notional_usd"), 1.0))
                    held_total = (fb_pos or 0.0) + lb_pos
                    if px_pos <= 0.0 or held_total * px_pos >= min_not_pos:
                        pos_entry["trailing_peak"] = pend_peak[0]
                        pos_entry["dd_from_peak_pct"] = pend_peak[1]
                pos_bkey = f"pos:{p_v_norm}:{p_sym_u}"
                if pos_bkey in updated_bals:
                    updated_bals[pos_bkey] = dict(pos_entry)

    if bcache_path and updated_bals and ("PYTEST_CURRENT_TEST" not in os.environ or os.path.exists(bcache_path)):
        try:
            merged_bals = {**cached_bals, **updated_bals}
            if merged_bals != cached_bals:
                os.makedirs(os.path.dirname(os.path.abspath(bcache_path)), exist_ok=True)
                save_json_state(bcache_path, merged_bals, indent=2)
        except Exception:
            pass

    return tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym


def enrich_cross_venue_trends(
    prices_and_trends: Dict[str, Any],
    trends_data: Any,
    kalman_data: Any,
    long_items: Any,
    priceanalysis_data: Any,
    t_trend_map: Any,
    watcher_price_items: Any,
) -> None:
    """Enrich every symbol with cross-venue short slopes, Kalman velocity, long trend, Weibull, and 7d range."""
    def _unwrap_items_dict(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw["items"] if isinstance(raw.get("items"), dict) else raw
        return {}

    trends_items_map, pa_map = _unwrap_items_dict(trends_data), _unwrap_items_dict(priceanalysis_data)
    long_map = _unwrap_items_dict(long_items)
    pa_root_ts = safe_pos_float(priceanalysis_data.get("timestamp") if isinstance(priceanalysis_data, dict) else None, default=0.0) or 0.0
    lt_root_ts = safe_pos_float(long_items.get("timestamp") if isinstance(long_items, dict) else None, default=0.0) or 0.0
    pa_cf_keys = {k.casefold(): k for k in pa_map if isinstance(k, str)}
    lt_cf_keys = {k.casefold(): k for k in long_map if isinstance(k, str)}

    def _resolve_source_long_trend(
        src_map: Dict[str, Any], cf_keys: Dict[str, str], cands: Tuple[str, ...], root_ts: float
    ) -> Tuple[Optional[Dict[str, Any]], bool, float]:
        if not src_map:
            return None, False, 0.0
        for ck in cands:
            if not ck or not isinstance(ck, str):
                continue
            real_k = ck if ck in src_map else cf_keys.get(ck.casefold())
            if real_k is None:
                continue
            raw_val = src_map[real_k]
            if raw_val is None or (isinstance(raw_val, list) and raw_val and raw_val[-1] is None):
                return None, True, root_ts
            parsed = scan_latest_long_trend_value(
                raw_val,
                lambda e: e if e.get("direction") else None,
                stop_on_none=True,
            )
            if isinstance(parsed, dict):
                item_ts = safe_pos_float(parsed.get("timestamp"), default=root_ts) or root_ts
                return parsed, False, item_ts
        return None, False, root_ts

    for sym, entry in list(prices_and_trends.items()):
        sym_quotes = compatible_quote_suffixes(sym)
        cand_keys = tuple(cache_symbol_variants(sym, include_original=True, include_base=True))
        has_slopes = any(entry.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h"))

        if not has_slopes and trends_items_map:
            for ck in cand_keys[1:]:
                tinfo = casefold_get(trends_items_map, ck)
                if isinstance(tinfo, dict) and apply_short_trend_item(
                    entry, tinfo, require_any_slope=True, allow_price_update=(compatible_quote_suffixes(ck) == sym_quotes)
                ):
                    has_slopes = True
                    break
        if not has_slopes:
            for ck in cand_keys[1:]:
                sib = casefold_get(prices_and_trends, ck)
                if isinstance(sib, dict) and any(sib.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h")):
                    for k_inh in (
                        "slope_90s", "trend_90s", "growth_90s", "slope_3_7m", "trend_3_7m", "growth_3_7m",
                        "slope_15m", "trend_15m", "growth_15m", "slope_2_5h", "trend_2_5h", "growth_2_5h",
                        "gradient_recent_fast", "epsilon", "vol_1h_pct",
                    ):
                        if sib.get(k_inh) is not None and not entry.get(k_inh):
                            entry[k_inh] = sib[k_inh]
                    break

        if entry.get("vol_1h_pct") is None and trends_items_map:
            for ck in cand_keys:
                if isinstance(t_vol_info := casefold_get(trends_items_map, ck), dict) and t_vol_info.get("vol_1h_pct") is not None:
                    entry["vol_1h_pct"] = t_vol_info.get("vol_1h_pct")
                    for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
                        if entry.get(ak) is None and t_vol_info.get(ak) is not None:
                            entry[ak] = t_vol_info[ak]
                    break

        if isinstance(kalman_data, dict) and isinstance(k_sym := _first_variant_dict(kalman_data, cand_keys), dict):
            entry["kalman_vel"], entry["kalman_vel_std"], entry["kalman_trend"] = (
                k_sym.get("kalman_vel"), k_sym.get("kalman_vel_std"), k_sym.get("kalman_trend")
            )
            cur_vol, k_vol = safe_pos_float(entry.get("vol_1h_pct"), default=None), safe_pos_float(k_sym.get("vol_1h_pct"), default=None)
            vol_matches_kalman = cur_vol is None or (k_vol is not None and abs(cur_vol - k_vol) <= 0.05)
            if cur_vol is None and k_vol is not None:
                entry["vol_1h_pct"] = k_vol
            if vol_matches_kalman:
                for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
                    if entry.get(ak) is None and k_sym.get(ak) is not None:
                        entry[ak] = k_sym[ak]

        if (eff_vol := safe_pos_float(entry.get("vol_1h_pct"), default=None)) is not None and (entry.get("adapt_reentry_pct") is None or entry.get("adapt_dca_pct") is None):
            calc_re, calc_dca = adaptive_thresholds(eff_vol)
            if entry.get("adapt_reentry_pct") is None and calc_re is not None:
                entry["adapt_reentry_pct"] = calc_re
            if entry.get("adapt_dca_pct") is None and calc_dca is not None:
                entry["adapt_dca_pct"] = calc_dca

        pa_obj, pa_neutral, pa_item_ts = _resolve_source_long_trend(pa_map, pa_cf_keys, cand_keys, pa_root_ts)
        lt_val, lt_neutral, lt_item_ts = _resolve_source_long_trend(long_map, lt_cf_keys, cand_keys, lt_root_ts)
        if isinstance(pa_obj, dict) and isinstance(lt_val, dict):
            lt_obj: Optional[Dict[str, Any]] = pa_obj if pa_item_ts >= lt_item_ts else lt_val
        elif isinstance(pa_obj, dict):
            lt_obj = None if (lt_neutral and lt_item_ts > pa_item_ts > 0.0) else pa_obj
        elif isinstance(lt_val, dict):
            lt_obj = None if (pa_neutral and (pa_item_ts >= lt_item_ts or lt_item_ts <= 0.0)) else lt_val
        else:
            lt_obj = None

        if isinstance(lt_obj, dict):
            entry["long_trend_dir"] = lt_obj.get("direction")
            entry["long_trend_hours"] = round(safe_float(lt_obj.get("duration_seconds"), 0.0) / 3600.0, 1)
            entry.pop("long_trend_neutral", None)
            if (est_rem := safe_float(lt_obj.get("estimated_future_hours"), -1.0)) >= 0.0:
                entry["long_trend_est_rem_hours"] = round(est_rem, 1)
            else:
                entry.pop("long_trend_est_rem_hours", None)
        else:
            entry.pop("long_trend_dir", None)
            entry.pop("long_trend_hours", None)
            entry.pop("long_trend_est_rem_hours", None)
            any_neutral_regime = bool(pa_map or long_map) and all(
                v is None or (isinstance(v, list) and v and v[-1] is None)
                for m in (pa_map, long_map)
                for v in m.values()
            )
            if pa_neutral or lt_neutral or (any_neutral_regime and not sym.endswith(("_EQ", "_US_EQ"))):
                entry["long_trend_neutral"] = True
        if isinstance(t_trend_map, dict) and isinstance(tt := _first_variant_dict(t_trend_map, cand_keys), dict):
            entry["weibull_median_d"], entry["weibull_p90_d"] = tt.get("median_d"), tt.get("p90_d")
        elif (
            entry.get("weibull_median_d") is None
            and not sym.endswith(("_EQ", "_US_EQ"))
            and "PYTEST_CURRENT_TEST" not in os.environ
        ):
            try:
                from intelligence.internal.state.survival import estimate_T

                if isinstance(est_tt := estimate_T(sym), dict) and est_tt.get("median_d") is not None:
                    entry["weibull_median_d"], entry["weibull_p90_d"] = est_tt.get("median_d"), est_tt.get("p90_d")
                    if isinstance(t_trend_map, dict):
                        t_trend_map[sym] = est_tt
            except Exception:
                pass
        if isinstance(watcher_price_items, dict):
            sym_px_cur = safe_float(entry.get("price"), 0.0)
            for ck in cand_keys:
                cand_ser = casefold_get(watcher_price_items, ck)
                if not isinstance(cand_ser, list) or len(cand_ser) < 2:
                    continue
                last_m_px = safe_float(cand_ser[-1][1], 0.0) if isinstance(cand_ser[-1], (list, tuple)) and len(cand_ser[-1]) > 1 else 0.0
                if ck != sym and (compatible_quote_suffixes(ck) != sym_quotes or _prices_diverged(sym_px_cur, last_m_px)):
                    continue
                w0, w1 = _parse_ts_px_row(cand_ser[0]), _parse_ts_px_row(cand_ser[-1])
                if "chg_7d_pct" in entry and w0 is not None and w1 is not None and 0.0 <= (w1[0] - w0[0]) < 5 * 86400 * 1000:
                    continue
                vals = [safe_float(pt[1], 0.0) for pt in cand_ser if isinstance(pt, list) and len(pt) > 1 and safe_float(pt[1], 0.0) > 0]
                entry.update(compute_7d_range_metrics(vals))
                break


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
    _rj = read_json_fn or read_json_cached
    btc_series_24h, _ = load_symbol_price_series(cache_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False)
    bcache_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "venue_balances_state.json"))) if cache_dir else None
    raw_cb = _rj(bcache_path) if bcache_path and os.path.exists(bcache_path) else None
    cached_bals: Dict[str, Any] = raw_cb if isinstance(raw_cb, dict) else {}
    for _, ticker in _iter_t212_state_files(workspace_dir):
        cached_pos = cached_bals.get(f"pos:t212:{ticker}")
        cached_t212_px = safe_pos_float(cached_pos.get("price") if isinstance(cached_pos, dict) else None, default=None)
        t_entry = prices_and_trends.setdefault(ticker, {"price": cached_t212_px, "trend": 0, "growth_coefficient": 0.0})
        if cached_t212_px is not None and (safe_pos_float(t_entry.get("price"), default=0.0) or 0.0) <= 0.0:
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
        if sym in inst_by_sym and callable(ohlc_fn := getattr(getattr(inst_by_sym[sym], "_provider", None), "ohlc_series", None)):
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
                        s60_ts = getattr(s60, "timestamps", ()) or ()
                        vals_7d = [float(px) for ts, px in zip(s60_ts, getattr(s60, "closes", ())) if ts >= cutoff_7d and float(px) > 0]
                        if s60_ts and (float(s60_ts[-1]) - float(s60_ts[0])) < 5 * 86400.0:
                            try:
                                s240 = ohlc_fn(sym, 240)
                                first_s60 = float(s60_ts[0]) - 1800.0
                                older_7d = [float(px) for ts, px in zip(getattr(s240, "timestamps", ()), getattr(s240, "closes", ())) if cutoff_7d <= ts <= first_s60 and float(px) > 0]
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
                    cutoff_24h_ms = int((now_ts - 86400.0) * 1000)
                    pts_24h = [pt for pt in pts_5d if pt[0] >= cutoff_24h_ms]
                    if len(pts_24h) < 10:
                        pts_24h = list(pts_5d[-78:])
                    if pts_24h and sym_px_cur > 0 and not _prices_diverged(sym_px_cur, pts_24h[-1][1]) and abs(sym_px_cur - pts_24h[-1][1]) > 1e-9:
                        pts_24h = list(pts_24h) + [[int(now_ts * 1000), sym_px_cur]]
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
                            vals_7d = [float(px) for ts, px in zip(getattr(s60, "timestamps", ()), getattr(s60, "closes", ())) if ts >= cutoff_7d and float(px) > 0]
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


def sync_sibling_symbol_telemetry(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> None:
    """Symmetric cross-venue & sibling symbol synchronization across Binance, Kraken, Hyperliquid, and T212/xStocks."""
    range_bound_keys = {
        "low_24h", "high_24h", "pos_24h_pct", "dd_24h_high_pct", "bounce_24h_low_pct",
        "low_7d", "high_7d", "pos_7d_pct", "dd_7d_high_pct", "bounce_7d_low_pct", "span_7d_pct",
        "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct", "recent_tape_15m",
        "swing_low_px", "swing_high_px", "major_swing_low_px",
        "rally_from_swing_low_pct", "pullback_from_swing_high_pct", "rally_from_major_low_pct",
        "poc_px", "sec_poc_px", "vah_px", "val_px", "poc_dist_pct", "vah_dist_pct", "val_dist_pct", "va_pos_pct", "va_regime", "vp_source",
        "bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px", "higher_low_vs_prior_pct", "reclaim_above_prior_low_pct",
    }
    for sym, pinfo in list(prices_and_trends.items()):
        if not isinstance(pinfo, dict):
            continue
        sym_quotes = compatible_quote_suffixes(sym)
        for cand in cache_symbol_variants(sym, include_original=True, include_base=True):
            if cand == sym:
                continue
            cand_same_quote_family = compatible_quote_suffixes(cand) == sym_quotes
            sib_pinfo = casefold_get(prices_and_trends, cand)
            if isinstance(sib_pinfo, dict):
                p_px, s_px = safe_float(pinfo.get("price"), 0.0), safe_float(sib_pinfo.get("price"), 0.0)
                cand_has_ob = (
                    isinstance(casefold_get(microstructure, cand), dict)
                    and casefold_get(microstructure, cand).get("ob_mid_price") is not None
                )
                sym_has_ob = (
                    isinstance(casefold_get(microstructure, sym), dict)
                    and casefold_get(microstructure, sym).get("ob_mid_price") is not None
                )
                if cand_same_quote_family and s_px > 0.0 and (p_px <= 0.0 or (cand_has_ob and not sym_has_ob and not _prices_diverged(p_px, s_px))):
                    pinfo["price"] = sib_pinfo["price"]
                    p_px = s_px
                px_diverged = (not cand_same_quote_family) or _prices_diverged(p_px, s_px)
                for k, v in sib_pinfo.items():
                    if v is None or k == "price" or (px_diverged and k in range_bound_keys):
                        continue
                    if k not in pinfo or pinfo.get(k) is None:
                        pinfo[k] = v
                    elif k in ("trend", "growth_coefficient") and pinfo.get(k) in (0, 0.0) and v not in (0, 0.0):
                        pinfo[k] = v
            sib_micro = casefold_get(microstructure, cand)
            if isinstance(sib_micro, dict) and sib_micro:
                p_px_m = safe_float(pinfo.get("price"), 0.0)
                s_ob_mid = safe_float(sib_micro.get("ob_mid_price"), 0.0) or safe_float(sib_pinfo.get("price") if isinstance(sib_pinfo, dict) else None, 0.0)
                micro_px_diverged = (not cand_same_quote_family) or _prices_diverged(p_px_m, s_ob_mid)
                if sym not in microstructure:
                    microstructure[sym] = {mk: mv for mk, mv in sib_micro.items() if not (micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS)}
                else:
                    sym_has_own_ob = (
                        microstructure[sym].get("bid_depth_usd") is not None
                        or microstructure[sym].get("ask_depth_usd") is not None
                    )
                    ob_local_keys = {
                        "ob_imbalance", "spread_bps", "bid_depth_usd", "ask_depth_usd",
                        "depth_1pct_bid_usd", "depth_1pct_ask_usd", "bid_wall_usd", "ask_wall_usd",
                        "bid_wall_px", "ask_wall_px", "est_slip_2k_bps", "spot_cvd_15m_usd", "spot_perp_flow_regime",
                    }
                    synced_any_micro = False
                    for mk, mv in sib_micro.items():
                        if sym_has_own_ob and mk in ob_local_keys:
                            continue
                        if not (micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS) and microstructure[sym].get(mk) is None and mv is not None:
                            microstructure[sym][mk] = mv
                            synced_any_micro = True
                    if synced_any_micro and sym_has_own_ob:
                        enrich_microstructure_cvd(microstructure[sym], pinfo)
        recalculate_live_price_geometry(pinfo, safe_pos_float(pinfo.get("price"), default=0.0) or 0.0)

    for msym, minfo in list(microstructure.items()):
        if not isinstance(minfo, dict):
            continue
        msym_quotes = compatible_quote_suffixes(msym)
        p_match = _first_variant_dict(
            prices_and_trends,
            cache_symbol_variants(msym, include_original=True, include_base="USD" in msym_quotes, quotes=msym_quotes),
        )
        minfo.update(compute_liquidation_clusters(p_match, minfo))
        enrich_microstructure_cvd(minfo, p_match)
        if isinstance(p_match, dict) and minfo.get("spread_bps") is not None:
            p_match.setdefault("spread_bps", minfo["spread_bps"])
        compute_cross_venue_arb_metrics(msym, minfo, p_match, microstructure, prices_and_trends)

    update_microstructure_cycle_deltas(cache_dir, microstructure, now_ts=now_ts)


def harvest_recent_orders_and_enrich_positions(
    cache_dir: str,
    inst_by_sym: Dict[str, Any],
    tradable_positions: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    kraken_cycles: Dict[str, Dict[str, Any]],
    hyperliquid_cycles: Dict[str, Dict[str, Any]],
    t212_cycles: Dict[str, Dict[str, Any]],
    read_json_fn: Any = None,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    gateway_max_usd: float = 2000.0,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Load recent fills across Binance, Kraken, Hyperliquid, and T212 and enrich tradable_positions with cost-basis anchors."""
    _rj = read_json_fn or read_json_cached
    audit_origin_map = _load_recent_audit_origins(cache_dir)
    recent_orders: Dict[str, Any] = {}
    venue_recent_orders: Dict[str, Dict[str, Any]] = {"binance": {}, "kraken": {}, "hyperliquid": {}, "t212": {}}

    def _venue_intents(v_name: str) -> Optional[List[Dict[str, Any]]]:
        return [it for it in ai_intents if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == v_name] if isinstance(ai_intents, list) else None

    def _record_venue_fills(c_venue: str, sym_k: str, fills_d: Dict[str, Any]) -> None:
        fills_d.setdefault("venue", c_venue)
        venue_recent_orders[c_venue][sym_k] = fills_d
        if sym_k in recent_orders and c_venue != "binance":
            recent_orders[f"{c_venue}:{sym_k}"] = fills_d
        else:
            recent_orders.setdefault(sym_k, fills_d)

    for order_cache_name, cache_venue in (
        ("cache_order.json", "binance"), ("cache_trade_kraken.json", "kraken"),
        ("cache_trade_hyperliquid.json", "hyperliquid"), ("cache_trade_t212.json", "t212"),
    ):
        orders_data = _rj(os.path.join(cache_dir, order_cache_name))
        if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
            v_ai_intents = _venue_intents(cache_venue)
            for sym, o_list in orders_data["items"].items():
                if fills := extract_symbol_recent_fills(o_list, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=v_ai_intents, symbol=sym):
                    _record_venue_fills(cache_venue, sym, fills)

    seen_hl_syms: set[str] = set()
    hl_ai_intents = _venue_intents("hyperliquid")
    for inst in inst_by_sym.values():
        sym = str(getattr(inst, "symbol", "")).upper()
        v_inst = normalize_venue_name(getattr(inst, "provider_name", ""))
        if not sym or v_inst != "hyperliquid" or sym in seen_hl_syms or _first_variant_dict(venue_recent_orders["hyperliquid"], cache_symbol_variants(sym, include_base=True)) is not None:
            continue
        seen_hl_syms.add(sym)
        try:
            prov = getattr(inst, "_provider", None)
            if prov is not None and hasattr(prov, "get_orders"):
                if fills := extract_symbol_recent_fills(prov.get_orders(sym, None, 90 * 86400.0), known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=hl_ai_intents, symbol=sym):
                    _record_venue_fills("hyperliquid", sym, fills)
        except Exception:
            pass
    for fallback_venue in ("hyperliquid", "t212"):
        if not venue_recent_orders[fallback_venue] and cache_dir:
            fb_ai_intents = _venue_intents(fallback_venue)
            for sym_fb, o_list in _load_audit_terminal_orders(cache_dir, fallback_venue).items():
                if fills := extract_symbol_recent_fills(o_list, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=fb_ai_intents, symbol=sym_fb):
                    _record_venue_fills(fallback_venue, sym_fb, fills)
    live_held_by_venue_sym: Dict[tuple[str, str], float] = {}
    for pos_item in tradable_positions:
        if isinstance(pos_item, dict):
            pv = normalize_venue_name(pos_item.get("venue"))
            ps = str(pos_item.get("symbol") or "").strip().upper()
            fb = safe_float(pos_item.get("free_base"), -1.0)
            lb = safe_pos_float(pos_item.get("locked_base_qty"), default=0.0) or 0.0
            if pv and ps and (fb >= 0.0 or lb > 0.0):
                held_q = (max(0.0, fb) + lb) if fb >= 0.0 else lb
                for sv in (ps, *cache_symbol_variants(ps, include_base=True)):
                    live_held_by_venue_sym.setdefault((pv, sv.upper()), held_q)
    cyc_sell_ts_map = reconcile_venue_cycles_with_recent_orders(
        kraken_cycles, hyperliquid_cycles, t212_cycles, venue_recent_orders, live_held_by_venue_sym, prices_and_trends
    )
    for pos in tradable_positions:
        if not isinstance(pos, dict):
            continue
        p_sym = str(pos.get("symbol") or "").upper()
        p_venue = normalize_venue_name(pos.get("venue"))
        p_quotes = compatible_quote_suffixes(p_sym, pos.get("quote"))
        p_quote_variants = cache_symbol_variants(p_sym, include_base="USD" in p_quotes, quotes=p_quotes)
        p_variants = (*p_quote_variants, *cache_symbol_variants(p_sym))
        pos_px_f = safe_float(pos.get("price"), 0.0)
        cand_px = resolve_symbol_price(p_sym, prices_and_trends, explicit_quote=pos.get("quote"))
        if cand_px > 0 and (pos_px_f <= 0 or (abs(pos_px_f - cand_px) > 1e-9 and not _prices_diverged(pos_px_f, cand_px))):
            pos_px_f = cand_px
            pos["price"] = round(cand_px, 6)
            if (fb_val := safe_float(pos.get("free_base"), -1.0)) >= 0:
                pos["value_usd"] = round(fb_val * cand_px, 2)
            if (pk_val := safe_float(pos.get("trailing_peak"), 0.0)) > 0:
                if pk_val < cand_px:
                    pk_val = cand_px
                    pos["trailing_peak"] = round(pk_val, 6)
                pos["dd_from_peak_pct"] = round(((cand_px - pk_val) / pk_val) * 100.0, 2)
        v_ro_map = venue_recent_orders.get(p_venue, {})
        ro = _first_variant_dict(v_ro_map, p_quote_variants)
        fb_cur = safe_float(pos.get("free_base"), -1.0)
        lb_cur = safe_pos_float(pos.get("locked_base_qty"), default=0.0) or 0.0
        held_cur = (fb_cur + lb_cur) if fb_cur >= 0.0 else (lb_cur if lb_cur > 0.0 else -1.0)
        min_notional_f = max(1.0, safe_float(pos.get("exchange_min_notional_usd") or pos.get("min_notional_usd"), 1.0))
        is_flat_balance = held_cur >= 0.0 and (held_cur <= 1e-8 or (pos_px_f > 0.0 and held_cur * pos_px_f < min_notional_f))
        if is_flat_balance:
            pos.pop("trailing_peak", None)
            pos.pop("dd_from_peak_pct", None)
        last_side_ro = str(ro.get("last_action_side") or "").upper() if isinstance(ro, dict) else ""
        is_flat_after_exit = is_flat_balance and last_side_ro == "SELL"
        if isinstance(ro, dict):
            lb, ls = ro.get("last_buy"), ro.get("last_sell")
            if ro.get("last_action_side"):
                pos["last_action_side"] = str(ro.get("last_action_side")).upper()
            if is_flat_after_exit:
                ro.pop("active_avg_buy_px", None)
                ro.pop("tactical_avg_buy_px", None)
                ro.pop("swing_subcycle_avg_px", None)
                ro.pop("open_lots", None)
                pos.pop("trailing_peak", None)
                pos.pop("dd_from_peak_pct", None)
            else:
                set_pos_anchor(pos, "active_avg_buy_px", "pnl_vs_avg_pct", ro.get("active_avg_buy_px"), pos_px_f)
                set_pos_anchor(pos, "tactical_avg_buy_px", "pnl_vs_tactical_avg_pct", ro.get("tactical_avg_buy_px"), pos_px_f)
                sw_avg = safe_pos_float(ro.get("swing_subcycle_avg_px"), default=None)
                act_avg = safe_pos_float(ro.get("active_avg_buy_px"), default=None)
                if sw_avg is not None and (act_avg is None or abs(sw_avg - act_avg) / max(act_avg, 1e-9) >= 0.002):
                    set_pos_anchor(pos, "swing_subcycle_avg_px", "pnl_vs_swing_avg_pct", sw_avg, pos_px_f)
                if isinstance(ro.get("open_lots"), list) and ro["open_lots"]:
                    pos["open_lots"] = ro["open_lots"][-4:]
                if (lifo_pnl := safe_float(ro.get("lifo_realized_pnl_usd"), default=None)) is not None and abs(lifo_pnl) >= 0.01:
                    pos["lifo_realized_pnl_usd"] = round(lifo_pnl, 2)
            set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", lb.get("price") if isinstance(lb, dict) else None, pos_px_f)
            set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", ls.get("price") if isinstance(ls, dict) else None, pos_px_f)
            r_buys = ro.get("recent_buys") if isinstance(ro.get("recent_buys"), list) else []
            low_buy_px = min((safe_float(b.get("price"), 0.0) for b in r_buys if isinstance(b, dict) and safe_float(b.get("price"), 0.0) > 0), default=0.0)
            if low_buy_px > 0 and not is_flat_after_exit:
                set_pos_anchor(pos, "lowest_buy_px", "pnl_vs_lowest_buy_pct", low_buy_px, pos_px_f)
        for cyc_map in {"hyperliquid": (hyperliquid_cycles,), "t212": (t212_cycles,), "kraken": (kraken_cycles,)}.get(p_venue, ()):
            cyc = _first_variant_dict(cyc_map, (p_sym, *p_quote_variants))
            if not isinstance(cyc, dict):
                continue
            cyc_qty = safe_nonneg_float(cyc.get("qty"), default=None)
            if (
                held_cur > 1e-8
                and not is_flat_balance
                and ((cyc_qty is not None and cyc_qty > 1e-8) or pos.get("active_avg_buy_px") is not None)
            ):
                if cyc_qty is None or abs(cyc_qty - held_cur) > 1e-6:
                    cyc_qty = round(held_cur, 6)
                    cyc["qty"] = cyc_qty
                eff_avg_cost = (
                    (
                        safe_pos_float(cyc.get("avg_cost_px"), default=None)
                        or safe_pos_float(pos.get("active_avg_buy_px"), default=None)
                    )
                    if p_venue == "t212"
                    else (
                        safe_pos_float(pos.get("active_avg_buy_px"), default=None)
                        or safe_pos_float(cyc.get("avg_cost_px"), default=None)
                        or safe_pos_float(cyc.get("entry_price"), default=None)
                    )
                )
                if eff_avg_cost is not None and eff_avg_cost > 0.0:
                    cyc["avg_cost_px"] = round(eff_avg_cost, 6)
                    cyc["cost_usd"] = round(cyc_qty * eff_avg_cost, 2)
                    if pos_px_f > 0.0:
                        cyc["pnl_vs_avg_pct"] = round(((pos_px_f - eff_avg_cost) / eff_avg_cost) * 100.0, 2)
                        cyc["dd_pct"] = cyc["pnl_vs_avg_pct"]
                if pos_px_f > 0.0 and (c_peak := safe_pos_float(cyc.get("pos_peak") or pos.get("trailing_peak"), default=None)) is not None:
                    eff_peak = max(c_peak, pos_px_f)
                    cyc["pos_peak"] = round(eff_peak, 6)
                    cyc["dd_from_peak_pct"] = round(((pos_px_f - eff_peak) / eff_peak) * 100.0, 2)
            if pos.get("free_base") is None and cyc_qty is not None:
                pos["free_base"] = round(cyc_qty, 6)
                fb_cur = cyc_qty
                held_cur = fb_cur + lb_cur
                is_flat_balance = held_cur <= 1e-8 or (pos_px_f > 0.0 and held_cur * pos_px_f < min_notional_f)
                is_flat_after_exit = is_flat_balance and last_side_ro == "SELL"
                if is_flat_balance:
                    pos.pop("trailing_peak", None)
                    pos.pop("dd_from_peak_pct", None)
                if pos_px_f > 0:
                    pos["value_usd"] = round(cyc_qty * pos_px_f, 2)
            cyc_has_open_qty = (cyc_qty > 1e-8) if cyc_qty is not None else (not is_flat_balance)
            cyc_flat = is_flat_balance and not cyc_has_open_qty
            if (is_flat_after_exit or cyc_flat) and not cyc.get("resting_orders"):
                cyc.update({"qty": round(max(0.0, fb_cur), 6), "cost_usd": 0.0, "avg_cost_px": None, "dd_pct": None})
                if isinstance(ro, dict):
                    ro.pop("active_avg_buy_px", None)
                    ro.pop("tactical_avg_buy_px", None)
                    ro.pop("swing_subcycle_avg_px", None)
                    ro.pop("open_lots", None)
                for _stale_k in (
                    "active_avg_buy_px",
                    "pnl_vs_avg_pct",
                    "tactical_avg_buy_px",
                    "pnl_vs_tactical_avg_pct",
                    "swing_subcycle_avg_px",
                    "pnl_vs_swing_avg_pct",
                    "open_lots",
                    "lowest_buy_px",
                    "pnl_vs_lowest_buy_pct",
                    "trailing_peak",
                    "dd_from_peak_pct",
                ):
                    pos.pop(_stale_k, None)
                if is_flat_after_exit and pos.get("last_sell_px") is not None:
                    cyc["last_sell_price"] = pos["last_sell_px"]
            elif cyc_flat:
                pos.pop("trailing_peak", None)
                pos.pop("dd_from_peak_pct", None)
            elif not is_flat_after_exit and cyc_has_open_qty:
                cyc_avg = cyc.get("avg_cost_px") or cyc.get("entry_price") or cyc.get("last_buy_price")
                set_pos_anchor(
                    pos, "active_avg_buy_px", "pnl_vs_avg_pct", cyc_avg, pos_px_f,
                    overwrite=(p_venue == "t212" and cyc.get("avg_cost_px") is not None),
                )
                ro = _populate_cycle_fallback_recent_order(
                    p_venue, p_sym, cyc, cyc_has_open_qty, ro, recent_orders, v_ro_map, cyc_sell_ts_map
                )
                set_pos_anchor(pos, "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct", cyc.get("avg_cost_px"), pos_px_f, overwrite=False)
                set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", cyc.get("pos_peak"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", cyc.get("last_buy_price"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", cyc.get("last_sell_price"), pos_px_f, overwrite=False)
            if pos.get("last_buy_px") is not None and (cyc.get("last_buy_price") is None or p_venue in ("hyperliquid", "kraken")):
                cyc["last_buy_price"] = pos["last_buy_px"]
            if pos.get("last_sell_px") is not None and (cyc.get("last_sell_price") is None or p_venue in ("hyperliquid", "kraken")):
                cyc["last_sell_price"] = pos["last_sell_px"]
        exact_sym_sp = (prices_and_trends.get(p_sym) or {}).get("spread_bps") if isinstance(prices_and_trends.get(p_sym), dict) else None
        sym_spread = (
            exact_sym_sp
            if exact_sym_sp is not None
            else (
                pos.get("spread_bps")
                if pos.get("spread_bps") is not None
                else next(
                    (val.get("spread_bps") for c in p_variants if isinstance((val := casefold_get(prices_and_trends, c)), dict) and val.get("spread_bps") is not None),
                    None,
                )
            )
        )
        inst_obj = inst_by_sym.get(f"{p_venue}:{p_sym}") or inst_by_sym.get(p_sym)
        pos.update(
            resolve_venue_fee_and_limits(
                p_venue, p_sym, inst_obj, price=pos_px_f,
                free_quote=safe_nonneg_float(pos.get("free_quote"), default=None),
                free_base=safe_nonneg_float(pos.get("free_base"), default=None),
                locked_quote_usd=safe_pos_float(pos.get("locked_quote_usd"), default=None),
                locked_base_qty=safe_pos_float(pos.get("locked_base_qty"), default=None),
                spread_bps=safe_nonneg_float(sym_spread, default=None),
                gateway_max_usd=gateway_max_usd,
            )
        )
    return recent_orders


def harvest_bot_runtime_states(
    cache_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
    tradable_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Harvest active retry queue, monitortrades, assetguardian, and smart_stop_loss states."""
    _rj = read_json_fn or read_json_cached
    out: Dict[str, Any] = {}
    retry_q_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "order_retry_queue.jsonl")))
    if os.path.exists(retry_q_path):
        try:
            import order_retry as _oq
        except Exception:
            _oq = None
        terminal_ai_intent_ids: set[str] = set()
        intents_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "autonomous_trade_intents_state.json")))
        intents_raw = read_json_file(intents_path, default=[], recover_previous=True) if os.path.exists(intents_path) else []
        if isinstance(intents_raw, list):
            for it in intents_raw:
                if not isinstance(it, dict):
                    continue
                st_u = str(it.get("status") or "").strip().upper()
                if st_u in {
                    "EXECUTED",
                    "FAILED",
                    "CANCELED",
                    "CANCELLED",
                    "EXPIRED",
                    "REJECTED",
                    "BLOCKED_BY_GUARD",
                    "SKIPPED_DUPLICATE_OPEN_ORDER",
                    "OBSERVED_SHADOW",
                } or st_u.startswith("REJECTED_"):
                    if iid := str(it.get("intent_id") or "").strip():
                        terminal_ai_intent_ids.add(iid)
        active_retries: List[Dict[str, Any]] = []
        for rec in read_tail_jsonl_dicts(retry_q_path, limit=50):
            rec_iid = str(rec.get("intent_id") or "").strip()
            if rec_iid and rec_iid in terminal_ai_intent_ids:
                continue
            lifecycle = str(rec.get("lifecycle") or "submit_pending").lower()
            exp_ts = safe_float(rec.get("expires_at_ts"), 0.0)
            checked_oq_expiry = False
            if _oq is not None and (rec.get("created_ts") is not None or rec.get("ttl_started_ts") is not None):
                try:
                    checked_oq_expiry = True
                    if _oq.is_expired(rec, now_ts):
                        continue
                except Exception:
                    checked_oq_expiry = False
            if lifecycle in ("submit_pending", "awaiting_cancel", "pending", "active") and (
                checked_oq_expiry or lifecycle == "awaiting_cancel" or exp_ts == 0.0 or exp_ts > now_ts
            ):
                active_retries.append({
                    "venue": str(rec.get("provider_name") or "").lower(),
                    **{k: rec.get(k) for k in ("symbol", "side", "qty", "requested_price", "kind", "attempts", "last_failure_reason")},
                })
        if active_retries:
            out["active_retry_queue"] = active_retries

    if isinstance(mt_state := _rj(os.path.join(cache_dir, "mt_state.json")), dict):
        active_mt = {
            k: v for k in ("tp_ride_state", "sl_bounce_state", "sl_episode_state", "reentry_state", "pending_stop_exit")
            if isinstance(v := mt_state.get(k), dict) and v
        }
        if active_mt:
            out["monitortrades_active_state"] = active_mt

    ag_state = _rj(os.path.join(cache_dir, "assetguardian_state.json"))
    if isinstance(ag_state, dict) and isinstance(ag_state.get("symbols"), dict) and ag_state["symbols"]:
        dust_binance_syms = {
            str(pos.get("symbol") or "").upper()
            for pos in (tradable_positions or ())
            if isinstance(pos, dict) and normalize_venue_name(pos.get("venue")) == "binance"
            and 0.0 <= safe_float(pos.get("value_usd"), -1.0)
            and (safe_float(pos.get("value_usd"), 0.0) + max(0.0, safe_float(pos.get("locked_base_usd"), 0.0)))
            < max(1.0, safe_float(pos.get("exchange_min_notional_usd") or pos.get("min_notional_usd"), 5.0))
        }
        cleaned_ag_syms: Dict[str, Any] = {}
        for ag_sym, ag_val in ag_state["symbols"].items():
            if not isinstance(ag_val, dict):
                continue
            sym_copy = dict(ag_val)
            if str(ag_sym).upper() in dust_binance_syms and "sell" in sym_copy:
                if not isinstance(sell_st := sym_copy.get("sell"), dict) or not sell_st.get("pending"):
                    sym_copy.pop("sell", None)
            if sym_copy:
                cleaned_ag_syms[ag_sym] = sym_copy
        if cleaned_ag_syms:
            out["assetguardian_state"] = cleaned_ag_syms

    if isinstance(ssl_state := _rj(os.path.join(cache_dir, "smart_stop_loss_state.json")), dict) and ssl_state:
        out["smart_stop_loss_state"] = ssl_state
    return out


