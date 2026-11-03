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
    _MICRO_PRICE_ANCHOR_KEYS, _build_vp_position_dict, _first_variant_dict, _parse_ts_px_row,
    _prices_diverged, apply_short_trend_item, compute_24h_price_metrics, compute_7d_range_metrics,
    compute_cross_btc_metrics, compute_liquidation_clusters, compute_volume_profile_metrics,
    enrich_microstructure_cvd, extract_symbol_recent_fills, load_symbol_microstructure,
    load_symbol_price_series, round_opt_px,
)
from instrument_registry import casefold_get
from providers.base import extract_order_id, extract_order_side, normalize_open_order, normalize_timestamp_ms, resolve_order_origin
from providers.quantity import resolve_assets, venue_fee_cap_quantity
from state_io import (
    read_json_dict_recovering as read_json_cached, read_json_file, read_tail_jsonl_dicts,
    resolve_cachedb_filepath, save_json_state,
)

logger = logging.getLogger("intelligence.sentiment.portfolio_collector")


_VENUE_DEFAULT_FEE_SPECS: Dict[str, tuple[str, float, float, float]] = {
    "binance": ("VIP0", 10.0, 10.0, 5.0),
    "kraken": ("PRO_TIER0", 16.0, 26.0, 1.0),
    "hyperliquid": ("SPOT_TIER0", 4.0, 7.0, 10.0),
    "t212": ("INVEST_RTH", 15.0, 15.0, 1.0),
}


def resolve_venue_fee_and_limits(
    venue: str,
    symbol: str,
    inst: Any = None,
    *,
    price: float = 0.0,
    free_quote: Optional[float] = None,
    free_base: Optional[float] = None,
    locked_quote_usd: Optional[float] = None,
    locked_base_qty: Optional[float] = None,
    spread_bps: Optional[float] = None,
    gateway_max_usd: float = 2000.0,
) -> Dict[str, Any]:
    """Resolve active venue fee tier, maker vs taker bps, live spread, and exchange/account order limits (0 ms, zero network calls)."""
    v_norm = normalize_venue_name(venue)
    fee_tier, maker_bps, taker_bps, min_notional = _VENUE_DEFAULT_FEE_SPECS.get(v_norm, ("STANDARD", 15.0, 15.0, 5.0))
    px_f = safe_pos_float(price, default=0.0) or 0.0
    if v_norm == "t212":
        default_step, default_tick = 0.001, 0.01
    elif v_norm == "hyperliquid":
        default_step = 0.0001 if px_f >= 1000.0 else 0.01
        default_tick = 0.01 if px_f >= 100.0 else (0.001 if px_f >= 1.0 else 0.0001)
    else:
        default_step = 0.0001 if px_f >= 1000.0 else (0.001 if px_f >= 50.0 else 0.01)
        default_tick = 0.01 if px_f >= 10.0 else (0.001 if px_f >= 1.0 else 0.0001)

    params = getattr(inst, "params", {}) if inst is not None and isinstance(getattr(inst, "params", None), dict) else {}
    if params.get("fee_tier"):
        fee_tier = str(params["fee_tier"]).strip()
    maker_bps = float(safe_nonneg_float(params.get("maker_fee_bps"), default=maker_bps))
    taker_bps = float(safe_nonneg_float(params.get("taker_fee_bps"), default=taker_bps))
    min_notional = float(safe_pos_float(params.get("min_notional"), default=min_notional))
    min_qty = float(safe_pos_float(params.get("min_qty"), default=default_step))
    step_size = float(safe_pos_float(params.get("step_size"), default=default_step))
    tick_size = float(safe_pos_float(params.get("tick_size"), default=default_tick))

    prov = getattr(inst, "_provider", None) if inst is not None else None
    if v_norm == "kraken" and prov is not None:
        sym_u = str(symbol or "").strip().upper()
        kr_cands = list(
            dict.fromkeys(
                kc for kc in (symbol, sym_u, clean_symbol_key(sym_u), resolve_kraken_lookup_symbol(sym_u) if "USD" in compatible_quote_suffixes(sym_u) else None)
                if isinstance(kc, str) and kc
            )
        )
        if isinstance((minqty_map := getattr(prov, "_minqty", None)), dict):
            min_qty = next((v for kc in kr_cands if (v := safe_pos_float(minqty_map.get(kc), default=None)) is not None), min_qty)
        client = getattr(prov, "_cli", None) or getattr(prov, "_client", None)
        cache_dict = getattr(client, "_CACHE", None)
        if not isinstance(cache_dict, dict):
            kc_mod = _sys.modules.get("kraken_client") or _sys.modules.get("kraken.kraken_client")
            cache_dict = getattr(kc_mod, "_CACHE", None)
        if isinstance(cache_dict, dict):
            def _cache_val(ckey: Any) -> Any:
                return hit[1] if isinstance((hit := cache_dict.get(ckey)), tuple) and len(hit) >= 2 else None

            p_meta = next((m for kc in kr_cands if isinstance((m := _cache_val(f"pair:{kc}")), dict)), None)
            if p_meta is None:
                p_meta = next(
                    (v for kc in kr_cands if isinstance((ap_res := _cache_val(("AssetPairs", (("pair", kc),)))), dict) for v in ap_res.values() if isinstance(v, dict)),
                    None,
                )
            if p_meta is None and isinstance((ap_all_res := _cache_val(("AssetPairs", ()))), dict):
                p_meta = next((ap_all_res[kc] for kc in kr_cands if isinstance(ap_all_res.get(kc), dict)), None)
            if isinstance(p_meta, dict):
                for fee_key, is_maker in (("fees_maker", True), ("fees", False)):
                    tbl = p_meta.get(fee_key)
                    if isinstance(tbl, list) and tbl and isinstance(tbl[0], (list, tuple)) and len(tbl[0]) >= 2:
                        if (fv := safe_nonneg_float(tbl[0][1], default=None)) is not None:
                            if is_maker:
                                maker_bps = round(fv * 100.0, 2)
                            else:
                                taker_bps = round(fv * 100.0, 2)
                min_qty = safe_pos_float(p_meta.get("ordermin"), default=min_qty) or min_qty
                min_notional = safe_pos_float(p_meta.get("costmin"), default=min_notional) or min_notional
                if (lot_dec := safe_nonneg_float(p_meta.get("lot_decimals"), default=None)) is not None:
                    step_size = round(10.0 ** (-int(lot_dec)), 8)
                if (pair_dec := safe_nonneg_float(p_meta.get("pair_decimals"), default=None)) is not None:
                    tick_size = round(10.0 ** (-int(pair_dec)), 8)
    elif v_norm == "hyperliquid" and prov is not None:
        client = getattr(prov, "_client", None)
        base_tok = canonical_base_asset(symbol)
        meta_cache = getattr(prov, "_meta_cache", None) or getattr(client, "_meta_cache", None)
        sz_dec = safe_nonneg_float(meta_cache[base_tok].get("szDecimals"), default=None) if isinstance(meta_cache, dict) and isinstance(meta_cache.get(base_tok), dict) else None
        if sz_dec is None:
            info_obj = getattr(client, "info", None)
            c2a, a2sz, n2c = getattr(info_obj, "coin_to_asset", None), getattr(info_obj, "asset_to_sz_decimals", None), getattr(info_obj, "name_to_coin", None)
            if isinstance(c2a, dict) and isinstance(a2sz, dict):
                spot_pair = getattr(prov, "_spot_pair", None) or (n2c.get(f"{base_tok}/USDC") if isinstance(n2c, dict) else None)
                sz_dec = next((v for ck in (spot_pair, base_tok) if ck and ck in c2a and (v := safe_nonneg_float(a2sz.get(c2a[ck]), default=None)) is not None), None)
        if sz_dec is not None:
            sz_i = int(sz_dec)
            step_size = min_qty = round(10.0 ** (-sz_i), 8)
            if safe_pos_float(params.get("tick_size"), default=None) is None:
                eff_px_dec = min(max(8 - sz_i, 0), max(0, 4 - int(math.floor(math.log10(px_f)))) if px_f > 0.0 else 4)
                tick_size = round(10.0 ** (-eff_px_dec), 8)

    out: Dict[str, Any] = {
        "fee_tier": fee_tier,
        "maker_fee_bps": round(maker_bps, 2),
        "taker_fee_bps": round(taker_bps, 2),
        "maker_rebate_edge_bps": round(taker_bps - maker_bps, 2),
        "exchange_min_notional_usd": round(min_notional, 2),
        "min_qty": min_qty,
        "step_size": step_size,
        "tick_size": tick_size,
        "buy_buffer_pct": 99.5,
    }
    if (sp_rounded := round_spread_bps(spread_bps)) is not None:
        out["spread_bps"] = sp_rounded
    gw_max = safe_pos_float(gateway_max_usd, default=2000.0) or 2000.0
    fq_f = safe_nonneg_float(free_quote, default=None)
    fb_f = safe_nonneg_float(free_base, default=None)
    lq_f = safe_pos_float(locked_quote_usd, default=0.0) or 0.0
    lb_f = safe_pos_float(locked_base_qty, default=0.0) or 0.0
    if fq_f is not None:
        out["effective_max_buy_usd"] = round(min(gw_max, fq_f * 0.995), 2)
    if lq_f > 0.0:
        out["locked_quote_usd"] = round(lq_f, 2)
        if fq_f is not None:
            out["replace_max_buy_usd"] = round(min(gw_max, (fq_f + lq_f) * 0.995), 2)
    if fb_f is not None and px_f > 0.0:
        out["effective_max_sell_usd"] = round(min(gw_max, venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f) * px_f), 2)
    if lb_f > 0.0:
        out["locked_base_qty"] = round(lb_f, 6)
        if px_f > 0.0:
            out["locked_base_usd"] = round(lb_f * px_f, 2)
            if fb_f is not None:
                out["replace_max_sell_usd"] = round(min(gw_max, venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f + lb_f) * px_f), 2)
    return out


def resolve_symbol_price(
    symbol: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Optional[Dict[str, float]] = None,
    *,
    explicit_quote: Optional[str] = None,
) -> float:
    """Resolve a positive live price for ``symbol`` across quote-compatible symbol variants and optional fallback map."""
    if not symbol:
        return 0.0
    sym_quotes = compatible_quote_suffixes(symbol, explicit_quote)
    sym_variants = cache_symbol_variants(symbol, include_base="USD" in sym_quotes, quotes=sym_quotes)
    for cand in sym_variants:
        entry = casefold_get(prices_and_trends, cand)
        if (cand_px := safe_pos_float(entry.get("price") if isinstance(entry, dict) else None, default=0.0) or 0.0) > 0:
            return cand_px
    if isinstance(fallback_prices, dict):
        for cand in sym_variants:
            if (cand_px := safe_pos_float(casefold_get(fallback_prices, cand), default=0.0) or 0.0) > 0:
                return cand_px
    return 0.0


def set_pos_anchor(
    pos: Dict[str, Any], px_key: str, pct_key: str, anchor_val: Any, cur_px: float, *, overwrite: bool = True,
) -> None:
    """Attach a cost/buy/sell/peak price anchor and its percentage delta to a position dict."""
    a_f = safe_pos_float(anchor_val, default=0.0) or 0.0
    if a_f <= 0 or (not overwrite and pos.get(px_key) is not None):
        return
    if px_key == "trailing_peak" and cur_px > a_f:
        a_f = cur_px
    pos[px_key] = round(a_f, 6)
    if cur_px > 0:
        pos[pct_key] = round(((cur_px - a_f) / a_f) * 100.0, 2)


def parse_venue_cycle_state(
    st_data: Dict[str, Any],
    symbol: str,
    prices_and_trends: Dict[str, Any],
    *,
    avg_decimals: int = 6,
    fallback_prices: Optional[Dict[str, float]] = None,
    include_meta: bool = False,
) -> Dict[str, Any]:
    """Normalize a venue cycle ``.state_*.json`` dict (Kraken, Hyperliquid, Trading212) into a uniform schema."""
    qty = safe_nonneg_float(st_data.get("qty"), default=0.0) or 0.0
    raw_cost = st_data.get("cost_usd") if st_data.get("cost_usd") is not None else st_data.get("cost")
    cost = safe_nonneg_float(raw_cost, default=0.0) or 0.0
    avg_cost = (
        round(cost / qty, avg_decimals)
        if qty > 1e-12 and cost > 0
        else (round_opt_px(st_data.get("avg_cost") or st_data.get("avg_price") or st_data.get("entry_price"), avg_decimals) if qty > 1e-12 else None)
    )
    px_f = resolve_symbol_price(symbol, prices_and_trends, fallback_prices)
    dd_pct = round(((px_f - avg_cost) / avg_cost) * 100.0, 2) if avg_cost and px_f > 0 else None
    last_buy_px = st_data.get("last_buy_price") if st_data.get("last_buy_price") is not None else st_data.get("last_open_price")
    cyc_entry: Dict[str, Any] = {
        "qty": round(qty, 6),
        "cost_usd": round(cost, 2),
        "avg_cost_px": avg_cost,
        "entry_price": round_opt_px(st_data.get("entry_price"), avg_decimals),
        "last_buy_price": round_opt_px(last_buy_px, avg_decimals),
        "last_sell_price": round_opt_px(st_data.get("last_sell_price"), avg_decimals),
        "last_exit_kind": st_data.get("last_exit_kind"),
        "dca_buys": st_data.get("dca_buys"),
        "dd_pct": dd_pct,
        "pnl_vs_avg_pct": dd_pct,
    }
    pos_peak = round_opt_px(st_data.get("pos_peak") or st_data.get("trail_peak"), avg_decimals)
    if pos_peak is not None and qty > 1e-12:
        cyc_entry["pos_peak"] = pos_peak
        if px_f > 0 and pos_peak > 0:
            eff_peak = max(pos_peak, px_f)
            cyc_entry["dd_from_peak_pct"] = round(((px_f - eff_peak) / eff_peak) * 100.0, 2)
    raw_realized = next(
        (st_data[rk] for rk in ("realized_net", "realized_net_usd", "realized_pnl", "realized_pnl_usd", "realized_gross") if st_data.get(rk) is not None),
        None,
    )
    if abs(realized_pnl := safe_float(raw_realized, 0.0) or 0.0) > 1e-9:
        cyc_entry["realized_pnl_usd"] = round(realized_pnl, 2)
    ref_entry_px = safe_pos_float(avg_cost or cyc_entry.get("entry_price") or cyc_entry.get("last_buy_price"), default=0.0) or 0.0
    last_sell_px_f = safe_pos_float(cyc_entry.get("last_sell_price"), default=0.0) or 0.0
    if (
        str(cyc_entry.get("last_exit_kind") or "").strip().upper() == "TP"
        and ref_entry_px > 0.0
        and last_sell_px_f > 0.0
        and last_sell_px_f < ref_entry_px * 0.995
        and realized_pnl < 0.0
    ):
        cyc_entry["last_exit_kind"] = "REDUCE"
    tp_sold = st_data.get("tp_sold") if st_data.get("tp_sold") is not None else st_data.get("tp_sold_levels")
    if isinstance(tp_sold, list) and tp_sold:
        cyc_entry["tp_sold_levels"] = tp_sold
    if isinstance((raw_orders := st_data.get("orders")), list) and raw_orders:
        close_orders = [
            {"side": o.get("side"), "px": o.get("px") or o.get("price") or o.get("limit"), "sz": o.get("sz") or o.get("qty"), "role": o.get("role") or o.get("kind")}
            for o in raw_orders if isinstance(o, dict)
        ]
        if close_orders:
            cyc_entry["resting_orders"] = close_orders
    if include_meta:
        for src_k, dst_k in (("entry_ts", "_entry_ts_ms"), ("last_sell_ts", "_last_sell_ts_ms")):
            if (ts_ms := normalize_timestamp_ms(st_data.get(src_k), default=None)) is not None:
                cyc_entry[dst_k] = ts_ms
    return cyc_entry


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


def _normalize_open_orders_list(
    raw_orders: Any,
    venue: str,
    symbol: str,
    known_ai_order_ids: Optional[set[str]] = None,
    *,
    now_ts: Optional[float] = None,
    order_ts_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_orders, list):
        return []
    return [
        oo_norm
        for oo in raw_orders
        if isinstance(oo, dict)
        and isinstance(
            (oo_norm := normalize_open_order(
                oo,
                venue=venue,
                symbol=symbol,
                known_ai_order_ids=known_ai_order_ids,
                now_ts=now_ts,
                order_ts_map=order_ts_map,
            )),
            dict,
        )
    ]


def _load_open_order_timestamps(cache_dir: Optional[str], cached_bals: Dict[str, Any]) -> Dict[str, float]:
    """Build order_id / client_order_id -> created_ts (epoch seconds) lookup for resting open orders."""
    out: Dict[str, float] = {}
    for k, v in (cached_bals or {}).items():
        if isinstance(k, str) and k.startswith("open_orders:") and isinstance(v, list):
            for oo_prev in v:
                if isinstance(oo_prev, dict):
                    c_ts = safe_pos_float(oo_prev.get("created_ts"), default=None)
                    if c_ts is None and (ms_v := normalize_timestamp_ms(oo_prev.get("time") or oo_prev.get("timestamp"), default=None)) is not None:
                        c_ts = float(ms_v) / 1000.0
                    if c_ts is not None and c_ts > 0.0:
                        for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                            if kv := str(oo_prev.get(id_k) or "").strip():
                                out.setdefault(kv, c_ts)
    if not cache_dir:
        return out
    intents_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "autonomous_trade_intents_state.json")))
    if os.path.exists(intents_path) and isinstance(raw_int := read_json_cached(intents_path), dict):
        for it in raw_int.get("intents") or ():
            if isinstance(it, dict) and (it_ts := safe_pos_float(it.get("timestamp"), default=None)) is not None:
                exec_pay = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
                for src_d in (it, exec_pay):
                    for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId", "intent_id"):
                        if kv := str(src_d.get(id_k) or "").strip():
                            out.setdefault(kv, it_ts)
    for d in (
        os.path.join(os.path.dirname(os.path.abspath(cache_dir)), "logger", "execution_audit"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cache_dir))), "logger", "execution_audit"),
        cache_dir,
    ):
        if not os.path.isdir(d):
            continue
        try:
            files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("execution_audit") and f.endswith(".jsonl")), reverse=True)
        except Exception:
            continue
        for fpath in files[:2]:
            for rec in read_tail_jsonl_dicts(fpath, limit=None, max_bytes=512 * 1024):
                if (rec_ts := safe_pos_float(rec.get("ts"), default=None)) is not None:
                    for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := str(rec.get(id_k) or "").strip():
                            out.setdefault(kv, rec_ts)
        if out:
            break
    return out


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
                vals = [safe_float(pt[1], 0.0) for pt in cand_ser if isinstance(pt, list) and len(pt) > 1 and safe_float(pt[1], 0.0) > 0]
                entry.update(compute_7d_range_metrics(vals))
                break


def _compute_kalman_from_series(series_24h: List[Any]) -> Optional[Dict[str, Any]]:
    """Derive Kalman velocity (% per min), std, and trend from a 24h price series when cache_kalman_trend lacks the symbol."""
    pts = [(parsed[0] / 1000.0, parsed[1]) for row in series_24h if (parsed := _parse_ts_px_row(row)) is not None]
    if len(pts) < 5:
        return None
    cutoff_s = pts[-1][0] - 7200.0
    subsampled: List[Tuple[float, float]] = []
    last_ts = -1e18
    for ts_s, px_v in pts:
        if ts_s >= cutoff_s and ts_s - last_ts >= 55.0:
            subsampled.append((ts_s, px_v))
            last_ts = ts_s
    if subsampled and pts[-1][0] > subsampled[-1][0]:
        subsampled.append(pts[-1])
    if len(subsampled) < 5:
        subsampled = pts[-min(len(pts), 30) :]
    if len(subsampled) < 5:
        return None
    kf = KalmanTrend(gap_reset_sec=7200.0)
    fed: List[float] = []
    last_out: Optional[Dict[str, Any]] = None
    for ts_s, px_v in subsampled[-60:]:
        fed.append(px_v)
        if len(fed) >= 3:
            diffs = [fed[i] - fed[i - 1] for i in range(1, len(fed))]
            mean_d = sum(diffs) / len(diffs)
            var_d = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
            eps_eff = max(math.sqrt(max(0.0, var_d)), px_v * 1e-4)
        else:
            eps_eff = max(px_v * 8e-4, 1e-9)
        last_out = kf.update(ts_s, px_v, eps_eff)
    if not isinstance(last_out, dict):
        return None
    return {"kalman_vel": last_out.get("vel"), "kalman_vel_std": last_out.get("vel_std"), "kalman_trend": last_out.get("trend")}


def _compute_short_slopes_from_series(series_24h: List[Any]) -> Dict[str, Any]:
    """Derive fallback multi-horizon slopes (90s, 3.7m, 15m, 2.5h) and growth coefficients from a 24h series."""
    pts = [(parsed[0] / 1000.0, parsed[1]) for row in series_24h if (parsed := _parse_ts_px_row(row)) is not None and parsed[1] > 0]
    if len(pts) < 3:
        return {}
    t_now, p_now = pts[-1]
    out: Dict[str, Any] = {}
    for win_sec, win_min, s_key, t_key, g_key in (
        (90.0, 1.5, "slope_90s", "trend_90s", "growth_90s"),
        (222.0, 3.7, "slope_3_7m", "trend_3_7m", "growth_3_7m"),
        (900.0, 15.0, "slope_15m", "trend_15m", "growth_15m"),
        (9000.0, 150.0, "slope_2_5h", "trend_2_5h", "growth_2_5h"),
    ):
        target_t = t_now - win_sec
        ref_t, ref_p = min(pts[:-1], key=lambda item: abs(item[0] - target_t))
        dt_min = max(win_min, (t_now - ref_t) / 60.0)
        if ref_p > 0 and dt_min > 0:
            slope_v = round(((p_now - ref_p) / ref_p) / dt_min, 6) + 0.0
            g_v = round(slope_v * win_min * 1000.0, 2) + 0.0
            out[s_key] = slope_v
            out[g_key] = g_v
            out[t_key] = 1 if g_v >= 0.25 else (-1 if g_v <= -0.25 else 0)
    if "slope_3_7m" in out:
        out["gradient_recent_fast"] = out["slope_3_7m"]
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
        if "PYTEST_CURRENT_TEST" not in os.environ and cache_dir and os.path.isdir(cache_dir):
            if not os.path.exists(os.path.join(cache_dir, f"orderbook_depth_{hl_sym}.json")):
                try:
                    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

                    OrderbookDepthCollector(cache_dir=cache_dir).fetch(hl_sym)
                except Exception:
                    pass
            if not os.path.exists(os.path.join(cache_dir, f"derivatives_telemetry_{hl_sym}.json")):
                try:
                    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector

                    DerivativesTelemetryCollector(cache_dir=cache_dir).fetch(hl_sym)
                except Exception:
                    pass
            if not os.path.exists(os.path.join(cache_dir, f"whale_snapshot_{hl_sym}.json")):
                try:
                    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector

                    WhalePositioningCollector(cache_dir=cache_dir).fetch(hl_sym)
                except Exception:
                    pass
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
        m24: Dict[str, Any] = compute_24h_price_metrics(matched_series_24h) or {} if matched_series_24h else {}
        if arch_vals_7d and len(arch_vals_7d) >= 2 and "chg_7d_pct" not in prices_and_trends[sym]:
            prices_and_trends[sym].update(compute_7d_range_metrics(arch_vals_7d))
        if sym in inst_by_sym and callable(ohlc_fn := getattr(getattr(inst_by_sym[sym], "_provider", None), "ohlc_series", None)):
            if not m24:
                try:
                    s5, cutoff_24h = ohlc_fn(sym, 5), now_ts - 86400.0
                    s5_vols = getattr(s5, "volumes", ()) or ()
                    pts_24h = [
                        ([int(ts * 1000), float(px), float(s5_vols[idx])] if idx < len(s5_vols) and (safe_pos_float(s5_vols[idx], 0.0) or 0.0) > 0.0 else [int(ts * 1000), float(px)])
                        for idx, (ts, px) in enumerate(zip(getattr(s5, "timestamps", ()), getattr(s5, "closes", ())))
                        if ts >= cutoff_24h and float(px) > 0
                    ]
                    if m24 := (compute_24h_price_metrics(pts_24h) or {}):
                        matched_series_24h = pts_24h
                    if "chg_7d_pct" not in prices_and_trends[sym]:
                        s60, cutoff_7d = ohlc_fn(sym, 60), now_ts - 7 * 86400.0
                        vals_7d = [float(px) for ts, px in zip(getattr(s60, "timestamps", ()), getattr(s60, "closes", ())) if ts >= cutoff_7d and float(px) > 0]
                        prices_and_trends[sym].update(compute_7d_range_metrics(vals_7d))
                except Exception:
                    pass
            elif m24.get("vp_source") == "TPO":
                try:
                    s5_vp, cutoff_vp = ohlc_fn(sym, 5), now_ts - 86400.0
                    s5_vp_vols = getattr(s5_vp, "volumes", ())
                    pts_vp: List[List[Any]] = []
                    if isinstance(s5_vp_vols, (list, tuple)) and s5_vp_vols:
                        pts_vp = [
                            [int(ts * 1000), float(px), float(s5_vp_vols[idx])]
                            for idx, (ts, px) in enumerate(zip(getattr(s5_vp, "timestamps", ()), getattr(s5_vp, "closes", ())))
                            if ts >= cutoff_vp and float(px) > 0 and idx < len(s5_vp_vols) and (safe_pos_float(s5_vp_vols[idx], 0.0) or 0.0) > 0.0
                        ]
                    if not pts_vp or (pts_vp[-1][0] - pts_vp[0][0]) < 18 * 3600 * 1000:
                        s15_vp = ohlc_fn(sym, 15)
                        s15_vols = getattr(s15_vp, "volumes", ())
                        if isinstance(s15_vols, (list, tuple)) and s15_vols:
                            first_s5_ts = (pts_vp[0][0] / 1000.0 - 300.0) if pts_vp else float("inf")
                            pts_15 = [
                                [int(ts * 1000), float(px), float(s15_vols[idx])]
                                for idx, (ts, px) in enumerate(zip(getattr(s15_vp, "timestamps", ()), getattr(s15_vp, "closes", ())))
                                if cutoff_vp <= ts <= first_s5_ts and float(px) > 0 and idx < len(s15_vols) and (safe_pos_float(s15_vols[idx], 0.0) or 0.0) > 0.0
                            ]
                            pts_vp = pts_15 + pts_vp
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
            prices_and_trends[sym].update(m24)
            if matched_series_24h and safe_float(prices_and_trends[sym].get("price"), 0.0) <= 0.0:
                prices_and_trends[sym]["price"] = matched_series_24h[-1][1]
            if prices_and_trends[sym].get("trend", 0) == 0 and "chg_1h_pct" in m24:
                c1h = safe_float(m24.get("chg_1h_pct"), 0.0)
                if c1h >= 0.25 or c1h <= -0.25:
                    prices_and_trends[sym]["trend"] = 1 if c1h >= 0.25 else -1
            if matched_series_24h and btc_series_24h and not sym.endswith(("_EQ", "_US_EQ")) and canonical_base_asset(sym) != "BTC":
                cross_btc = compute_cross_btc_metrics(matched_series_24h, btc_series_24h)
                for ck in ("alt_btc_pos_24h_pct", "alt_btc_dd_24h_high_pct", "alt_btc_bounce_24h_low_pct", "beta_btc_24h", "corr_btc_24h"):
                    if cross_btc.get(ck) is not None:
                        prices_and_trends[sym][ck] = cross_btc[ck]
            if prices_and_trends[sym].get("kalman_vel") is None and matched_series_24h and isinstance(k_fb := _compute_kalman_from_series(matched_series_24h), dict):
                for kk, kv in k_fb.items():
                    if kv is not None and prices_and_trends[sym].get(kk) is None:
                        prices_and_trends[sym][kk] = kv
            if not sym.endswith(("_EQ", "_US_EQ")) and matched_series_24h:
                if prices_and_trends[sym].get("slope_3_7m") is None and (s_fb := _compute_short_slopes_from_series(matched_series_24h)):
                    for sk, sv in s_fb.items():
                        if sv is not None and prices_and_trends[sym].get(sk) is None:
                            prices_and_trends[sym][sk] = sv
                if prices_and_trends[sym].get("vol_1h_pct") is None and len(matched_series_24h) >= 20:
                    tail_pts_m = [pt for r in matched_series_24h[-60:] if (pt := _parse_ts_px_row(r)) is not None]
                    if len(tail_pts_m) >= 10:
                        dt_s_m = max(60.0, (float(tail_pts_m[-1][0]) - float(tail_pts_m[0][0])) / (1000.0 * max(1, len(tail_pts_m) - 1)))
                        if (v1h_m := calculate_volatility_1h([float(r[1]) for r in tail_pts_m], dt_s_m)) is not None and v1h_m > 0:
                            prices_and_trends[sym]["vol_1h_pct"] = v1h_m
                            re_th_m, dca_th_m = adaptive_thresholds(v1h_m)
                            if re_th_m is not None and prices_and_trends[sym].get("adapt_reentry_pct") is None:
                                prices_and_trends[sym]["adapt_reentry_pct"] = re_th_m
                            if dca_th_m is not None and prices_and_trends[sym].get("adapt_dca_pct") is None:
                                prices_and_trends[sym]["adapt_dca_pct"] = dca_th_m

        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            and cache_dir
            and os.path.isdir(cache_dir)
            and sym.upper() in ("HYPEUSD", "TAOUSD", "ADAUSD")
            and not os.path.exists(os.path.join(cache_dir, f"orderbook_depth_{sym.upper()}.json"))
        ):
            try:
                from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

                OrderbookDepthCollector(cache_dir=cache_dir).fetch(sym.upper())
            except Exception:
                pass
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

    if os.path.isdir(cache_dir):
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
        for msym in ref_micro_syms:
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
                            s5, cutoff_24h = krk_p.ohlc_series(msym, 5), now_ts - 86400.0
                            s5_vols = getattr(s5, "volumes", ()) or ()
                            ser_24h = [
                                ([int(ts * 1000), float(px), float(s5_vols[idx])] if idx < len(s5_vols) and (safe_pos_float(s5_vols[idx], 0.0) or 0.0) > 0.0 else [int(ts * 1000), float(px)])
                                for idx, (ts, px) in enumerate(zip(getattr(s5, "timestamps", ()), getattr(s5, "closes", ())))
                                if ts >= cutoff_24h and float(px) > 0
                            ]
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
                            p_ref.update(m24_ref)
                            if "chg_1h_pct" in m24_ref:
                                c1h_r = safe_float(m24_ref.get("chg_1h_pct"), 0.0)
                                if c1h_r >= 0.25 or c1h_r <= -0.25:
                                    p_ref["trend"] = 1 if c1h_r >= 0.25 else -1
                            if btc_series_24h and canonical_base_asset(msym) != "BTC":
                                cross_btc_r = compute_cross_btc_metrics(ser_24h, btc_series_24h)
                                for ck in ("alt_btc_pos_24h_pct", "alt_btc_dd_24h_high_pct", "alt_btc_bounce_24h_low_pct", "beta_btc_24h", "corr_btc_24h"):
                                    if cross_btc_r.get(ck) is not None:
                                        p_ref[ck] = cross_btc_r[ck]
                            if isinstance(k_fb_r := _compute_kalman_from_series(ser_24h), dict):
                                for kk, kv in k_fb_r.items():
                                    if kv is not None:
                                        p_ref[kk] = kv
                            if s_fb_r := _compute_short_slopes_from_series(ser_24h):
                                for sk, sv in s_fb_r.items():
                                    if sv is not None and p_ref.get(sk) is None:
                                        p_ref[sk] = sv
                            if len(ser_24h) >= 20:
                                tail_pts = ser_24h[-60:]
                                dt_s = max(60.0, (float(tail_pts[-1][0]) - float(tail_pts[0][0])) / (1000.0 * max(1, len(tail_pts) - 1)))
                                if (v1h := calculate_volatility_1h([float(r[1]) for r in tail_pts], dt_s)) is not None and v1h > 0:
                                    p_ref["vol_1h_pct"] = v1h
                                    re_th, dca_th = adaptive_thresholds(v1h)
                                    if re_th is not None:
                                        p_ref["adapt_reentry_pct"] = re_th
                                    if dca_th is not None:
                                        p_ref["adapt_dca_pct"] = dca_th
                        if vals_7d and len(vals_7d) >= 2:
                            p_ref.update(compute_7d_range_metrics(vals_7d))
                        prices_and_trends[msym] = p_ref
                enrich_microstructure_cvd(m_info_ref, p_ref)
                m_info_ref.update(compute_liquidation_clusters(p_ref, m_info_ref))
                microstructure[msym] = m_info_ref


def _infer_symbol_venue_and_fees_bps(sym: str, m_dict: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """Infer canonical venue name, maker/taker fee bps, and half-spread bps for cross-venue smart routing."""
    sym_u = str(sym or "").strip().upper()
    if sym_u == "HYPEUSDC":
        venue, maker_bps, taker_bps = "hyperliquid", 4.0, 7.0
    elif sym_u.endswith(("USDC", "USDT")):
        venue, maker_bps, taker_bps = "binance", 10.0, 10.0
    elif sym_u.endswith(("USD", "EUR")) or sym_u.startswith(("XX", "XZ")):
        venue, maker_bps, taker_bps = "kraken", 16.0, 26.0
    else:
        venue, maker_bps, taker_bps = "t212", 15.0, 15.0
    half_sp = 0.5 * (safe_nonneg_float((m_dict or {}).get("spread_bps"), default=0.0) or 0.0)
    return venue, maker_bps, taker_bps, half_sp


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
                    for mk, mv in sib_micro.items():
                        if not (micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS) and microstructure[sym].get(mk) is None and mv is not None:
                            microstructure[sym][mk] = mv
        p_cur = safe_pos_float(pinfo.get("price"), default=0.0) or 0.0
        poc_v, val_v, vah_v = (safe_pos_float(pinfo.get(k), default=0.0) or 0.0 for k in ("poc_px", "val_px", "vah_px"))
        if p_cur > 0.0 and poc_v > 0.0 and val_v > 0.0 and vah_v > val_v:
            pinfo.update(_build_vp_position_dict(p_cur, poc_v, val_v, vah_v))
        if p_cur > 0.0 and isinstance(tape_15m := pinfo.get("recent_tape_15m"), list) and tape_15m:
            tape_copy = list(tape_15m)
            if isinstance(tape_copy[-1], str) and tape_copy[-1].startswith("now:"):
                tape_copy[-1] = f"now:{p_cur:.6g}"
                pinfo["recent_tape_15m"] = tape_copy
            l15 = safe_pos_float(pinfo.get("low_15m"), default=None)
            h15 = safe_pos_float(pinfo.get("high_15m"), default=None)
            if l15 is not None and h15 is not None:
                l15_new, h15_new = min(l15, p_cur), max(h15, p_cur)
                pinfo["low_15m"], pinfo["high_15m"] = round(l15_new, 6), round(h15_new, 6)
                if h15_new > l15_new > 0:
                    pinfo["pos_15m_pct"] = round(min(100.0, max(0.0, ((p_cur - l15_new) / (h15_new - l15_new)) * 100.0)), 1)

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
        # Compute Cross-Venue Relative Value / Arbitrage Spread (Discrepancy 3 & Layer II.3)
        own_px = safe_pos_float(minfo.get("ob_mid_price"), default=None) or safe_pos_float((p_match or {}).get("price"), default=None)
        if own_px and own_px > 0.0 and "USD" in msym_quotes:
            mbase = canonical_base_asset(msym)
            own_v, _, own_taker, own_hsp = _infer_symbol_venue_and_fees_bps(msym, minfo)
            peer_candidates = [
                (peer_sym, peer_m)
                for peer_sym, peer_m in microstructure.items()
                if peer_sym != msym
                and isinstance(peer_m, dict)
                and canonical_base_asset(peer_sym) == mbase
                and "USD" in compatible_quote_suffixes(peer_sym)
            ]
            peer_candidates.sort(
                key=lambda kv: (
                    0 if str(kv[0]).upper().endswith(("USDC", "USD")) else 1,
                    str(kv[0]).upper(),
                )
            )
            for peer_sym, peer_m in peer_candidates:
                peer_v, _, peer_taker, peer_hsp = _infer_symbol_venue_and_fees_bps(peer_sym, peer_m)
                if peer_v == own_v:
                    continue
                peer_px = (
                    safe_pos_float(peer_m.get("ob_mid_price"), default=None)
                    or safe_pos_float((prices_and_trends.get(peer_sym) or {}).get("price"), default=None)
                )
                if peer_px and peer_px > 0.0 and not _prices_diverged(own_px, peer_px):
                    arb_bps = round(((own_px - peer_px) / peer_px) * 10000.0, 1)
                    minfo["cross_venue_peer"] = peer_sym
                    minfo["cross_venue_peer_px"] = peer_px
                    minfo["cross_venue_arb_bps"] = arb_bps
                    own_buy_cost = arb_bps + own_taker + own_hsp
                    peer_buy_cost = peer_taker + peer_hsp
                    net_buy_edge = round(peer_buy_cost - own_buy_cost, 1)
                    own_sell_net = arb_bps - own_taker - own_hsp
                    peer_sell_net = -peer_taker - peer_hsp
                    net_sell_edge = round(own_sell_net - peer_sell_net, 1)
                    minfo["net_buy_edge_vs_peer_bps"] = net_buy_edge
                    minfo["net_sell_edge_vs_peer_bps"] = net_sell_edge
                    minfo["best_buy_venue"] = f"{own_v}:{msym}" if net_buy_edge >= 0.0 else f"{peer_v}:{peer_sym}"
                    minfo["best_sell_venue"] = f"{own_v}:{msym}" if net_sell_edge >= 0.0 else f"{peer_v}:{peer_sym}"
                    break

    if cache_dir and microstructure:
        eff_now = safe_pos_float(now_ts, default=None) or _time.time()
        mhist_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "microstructure_cycle_history.json")))
        repo_cachedb = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "cachedb"))
        is_repo_cachedb = os.path.abspath(mhist_path).startswith(repo_cachedb)
        raw_mh = (
            read_json_file(mhist_path, default={})
            if os.path.exists(mhist_path) and ("PYTEST_CURRENT_TEST" not in os.environ or not is_repo_cachedb)
            else None
        )
        mhist: Dict[str, Any] = dict(raw_mh) if isinstance(raw_mh, dict) else {}
        mhist_dirty = False
        for msym, minfo in list(microstructure.items()):
            if not isinstance(minfo, dict):
                continue
            cur_ob = safe_float(minfo.get("ob_imbalance"), None)
            cur_cvd = safe_float(minfo.get("cvd_15m_usd"), None)
            cur_wl = safe_float(minfo.get("whale_long_pct"), None)
            cur_oi = safe_pos_float(minfo.get("oi_usd"), default=None)
            cur_tr = safe_pos_float(minfo.get("taker_buy_sell_ratio"), default=None)
            if all(v is None for v in (cur_ob, cur_cvd, cur_wl, cur_oi, cur_tr)):
                continue
            h_list = [x for x in (mhist.get(msym) or []) if isinstance(x, dict) and (safe_pos_float(x.get("ts"), default=0.0) or 0.0) > 0.0]
            if not h_list:
                for ck in cache_symbol_variants(msym, include_base=True):
                    if ck in mhist and isinstance(mhist[ck], list):
                        h_list = [x for x in mhist[ck] if isinstance(x, dict) and (safe_pos_float(x.get("ts"), default=0.0) or 0.0) > 0.0]
                        if h_list:
                            break
            ref_snap: Optional[Dict[str, Any]] = None
            for cand_s in reversed(h_list):
                dt_s = eff_now - float(cand_s["ts"])
                if 330.0 <= dt_s <= 7200.0:
                    ref_snap = cand_s
                    break
            if ref_snap is None:
                for cand_s in reversed(h_list):
                    dt_s = eff_now - float(cand_s["ts"])
                    if 180.0 <= dt_s <= 7200.0:
                        ref_snap = cand_s
                        break
            if ref_snap is None and h_list:
                dt_first = eff_now - float(h_list[0]["ts"])
                if 20.0 <= dt_first <= 7200.0:
                    ref_snap = h_list[0]
            if isinstance(ref_snap, dict):
                delta_d: Dict[str, Any] = {"dt_min": round((eff_now - float(ref_snap["ts"])) / 60.0, 1)}
                if cur_ob is not None and (p_ob := safe_float(ref_snap.get("ob_imbalance"), None)) is not None:
                    delta_d["ob_imbalance_delta"] = round(cur_ob - p_ob, 4)
                    delta_d["prev_ob_imbalance"] = round(p_ob, 4)
                p_cvd = safe_float(ref_snap.get("cvd_15m_usd"), None)
                if cur_cvd is not None and p_cvd is not None and abs(cur_cvd - p_cvd) < 0.5:
                    for cand_c in reversed(h_list):
                        if 180.0 <= (eff_now - float(cand_c["ts"])) <= 3600.0:
                            cand_cvd = safe_float(cand_c.get("cvd_15m_usd"), None)
                            if cand_cvd is not None and abs(cur_cvd - cand_cvd) >= 0.5:
                                p_cvd = cand_cvd
                                break
                if cur_cvd is not None and p_cvd is not None:
                    delta_d["cvd_15m_delta_usd"] = round(cur_cvd - p_cvd, 2)
                    delta_d["prev_cvd_15m_usd"] = round(p_cvd, 2)
                p_wl = safe_float(ref_snap.get("whale_long_pct"), None)
                if cur_wl is not None and p_wl is not None and abs(cur_wl - p_wl) < 1e-5:
                    for cand_w in reversed(h_list):
                        if 180.0 <= (eff_now - float(cand_w["ts"])) <= 3600.0:
                            cand_wl = safe_float(cand_w.get("whale_long_pct"), None)
                            if cand_wl is not None and abs(cur_wl - cand_wl) >= 1e-4:
                                p_wl = cand_wl
                                break
                if cur_wl is not None and p_wl is not None:
                    delta_d["whale_long_delta_pct"] = round(cur_wl - p_wl, 2)
                    delta_d["prev_whale_long_pct"] = round(p_wl, 2)
                if cur_oi is not None and (p_oi := safe_pos_float(ref_snap.get("oi_usd"), default=None)) is not None and p_oi > 0.0:
                    oi_d_pct = round(((cur_oi - p_oi) / p_oi) * 100.0, 2)
                    if abs(oi_d_pct) <= 80.0:
                        delta_d["oi_delta_pct"] = oi_d_pct
                        delta_d["prev_oi_usd"] = round(p_oi, 2)
                if cur_tr is not None and (p_tr := safe_pos_float(ref_snap.get("taker_buy_sell_ratio"), default=None)) is not None:
                    delta_d["taker_ratio_delta"] = round(cur_tr - p_tr, 4)
                    delta_d["prev_taker_ratio"] = round(p_tr, 4)
                if len(delta_d) > 1:
                    minfo["cycle_delta"] = delta_d
            if not h_list or (eff_now - float(h_list[-1]["ts"])) >= 60.0:
                new_rec: Dict[str, Any] = {"ts": round(eff_now, 2)}
                if cur_ob is not None:
                    new_rec["ob_imbalance"] = round(cur_ob, 4)
                if cur_cvd is not None:
                    new_rec["cvd_15m_usd"] = round(cur_cvd, 2)
                if cur_wl is not None:
                    new_rec["whale_long_pct"] = round(cur_wl, 2)
                if cur_oi is not None:
                    new_rec["oi_usd"] = round(cur_oi, 2)
                if cur_tr is not None:
                    new_rec["taker_buy_sell_ratio"] = round(cur_tr, 4)
                mhist[msym] = [x for x in h_list if eff_now - float(x["ts"]) <= 14400.0][-11:] + [new_rec]
                mhist_dirty = True
        if mhist_dirty and ("PYTEST_CURRENT_TEST" not in os.environ or not is_repo_cachedb):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(mhist_path)), exist_ok=True)
                save_json_state(mhist_path, mhist, indent=2)
            except Exception:
                pass


def _load_recent_audit_origins(cache_dir: str) -> Dict[str, tuple[Optional[str], Optional[str]]]:
    """Load recent orderId/clientOrderId -> (module, rule) mappings from execution_audit jsonl logs."""
    out: Dict[str, tuple[Optional[str], Optional[str]]] = {}
    for d in (
        os.path.join(os.path.dirname(os.path.abspath(cache_dir)), "logger", "execution_audit"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cache_dir))), "logger", "execution_audit"),
        cache_dir,
    ):
        if not os.path.isdir(d):
            continue
        try:
            files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("execution_audit") and f.endswith(".jsonl")), reverse=True)
        except Exception:
            continue
        for fpath in files[:2]:
            for rec in read_tail_jsonl_dicts(fpath, limit=None, max_bytes=1024 * 1024):
                _, mod, rule = resolve_order_origin(rec)
                if mod or rule:
                    for k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := str(rec.get(k) or "").strip():
                            out.setdefault(kv, (mod, rule))
        if out:
            break
    return out


def _load_audit_terminal_orders(cache_dir: str, venue: str) -> Dict[str, List[Dict[str, Any]]]:
    """Synthesize deduplicated per-symbol filled order lists from execution_audit_*.jsonl for venues lacking a disk order cache."""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    if not cache_dir:
        return by_sym
    v_target = normalize_venue_name(venue)
    audit_dirs = (
        os.path.join(os.path.dirname(os.path.abspath(cache_dir)), "logger", "execution_audit"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cache_dir))), "logger", "execution_audit"),
        cache_dir,
    )
    for audit_dir in audit_dirs:
        if not os.path.isdir(audit_dir):
            continue
        try:
            audit_files = sorted(
                (os.path.join(audit_dir, f) for f in os.listdir(audit_dir) if f.startswith("execution_audit") and f.endswith(".jsonl"))
            )[-10:]
            if not audit_files:
                continue
            submit_ts_by_oid: Dict[str, float] = {}
            idx_by_oid: Dict[tuple[str, str], int] = {}
            for af in audit_files:
                for rec in read_tail_jsonl_dicts(af, limit=None, max_bytes=1024 * 1024):
                    if normalize_venue_name(rec.get("venue")) != v_target:
                        continue
                    ev_name = str(rec.get("event") or "")
                    oid_s = str(rec.get("order_id") or "").strip()
                    ts_s = safe_pos_float(rec.get("ts"), default=0.0) or 0.0
                    if ev_name == "submit_accepted" and oid_s and ts_s > 0.0:
                        submit_ts_by_oid.setdefault(oid_s, ts_s)
                        continue
                    if ev_name not in ("order_terminal", "order_terminal_retry"):
                        continue
                    sym_t = str(rec.get("symbol") or "").strip().upper()
                    f_q = safe_pos_float(rec.get("filled_qty") if rec.get("filled_qty") is not None else rec.get("qty"), default=0.0) or 0.0
                    cost_f = safe_pos_float(rec.get("cost"), default=0.0) or 0.0
                    px_f = safe_pos_float(rec.get("price"), default=None) or (round(cost_f / f_q, 6) if cost_f > 0 and f_q > 0 else None)
                    eff_ts_s = submit_ts_by_oid.get(oid_s, ts_s) if oid_s else ts_s
                    side_t = extract_order_side(rec) or str(rec.get("side") or "").strip().upper()
                    if sym_t and f_q > 0 and px_f is not None and px_f > 0 and eff_ts_s > 0 and side_t in ("BUY", "SELL"):
                        row_d = {
                            "symbol": sym_t,
                            "orderId": rec.get("order_id"),
                            "clientOrderId": rec.get("client_order_id"),
                            "side": side_t,
                            "price": px_f,
                            "qty": f_q,
                            "executedQty": f_q,
                            "timestamp": int(round(eff_ts_s * 1000.0)),
                            "status": "FILLED",
                        }
                        dedup_k = oid_s or str(rec.get("client_order_id") or "").strip()
                        sym_list = by_sym.setdefault(sym_t, [])
                        if dedup_k and (sym_t, dedup_k) in idx_by_oid:
                            existing = sym_list[idx_by_oid[(sym_t, dedup_k)]]
                            new_q = max(float(existing.get("qty") or 0.0), f_q)
                            existing["qty"] = new_q
                            existing["executedQty"] = new_q
                            existing["price"] = px_f
                            if rec.get("client_order_id") and not existing.get("clientOrderId"):
                                existing["clientOrderId"] = rec.get("client_order_id")
                        else:
                            if dedup_k:
                                idx_by_oid[(sym_t, dedup_k)] = len(sym_list)
                            sym_list.append(row_d)
            if by_sym:
                break
        except Exception:
            pass
    return by_sym


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
    cyc_sell_ts_map: Dict[tuple[str, str], int] = {}
    for c_venue, cyc_map, avg_dec in (("kraken", kraken_cycles, 6), ("hyperliquid", hyperliquid_cycles, 6), ("t212", t212_cycles, 4)):
        if not isinstance(cyc_map, dict):
            continue
        v_ro_c = venue_recent_orders.get(c_venue, {})
        for c_sym, cyc in cyc_map.items():
            if not isinstance(cyc, dict):
                continue
            entry_ts_ms, sell_ts_ms = cyc.pop("_entry_ts_ms", None), cyc.pop("_last_sell_ts_ms", None)
            if sell_ts_ms:
                cyc_sell_ts_map[(c_venue, str(c_sym).upper())] = int(sell_ts_ms)
            if not entry_ts_ms or cyc.get("resting_orders"):
                continue
            c_quotes = compatible_quote_suffixes(c_sym)
            ro_c = _first_variant_dict(v_ro_c, (c_sym, *cache_symbol_variants(c_sym, include_base="USD" in c_quotes, quotes=c_quotes)))
            if not isinstance(ro_c, dict):
                continue
            ls_c = ro_c.get("last_sell") if isinstance(ro_c.get("last_sell"), dict) else {}
            ls_ts_ms = safe_pos_float(ls_c.get("ts_ms"), default=0.0) or 0.0
            ls_qty = safe_pos_float(ls_c.get("qty"), default=0.0) or 0.0
            ls_px = safe_pos_float(ls_c.get("price"), default=0.0) or 0.0
            cyc_qty_cur = safe_nonneg_float(cyc.get("qty"), default=0.0) or 0.0
            r_buys_c = ro_c.get("recent_buys") if isinstance(ro_c.get("recent_buys"), list) else []
            buys_after = [
                b for b in r_buys_c
                if isinstance(b, dict)
                and (safe_pos_float(b.get("ts_ms"), default=0.0) or 0.0) > ls_ts_ms
                and (safe_pos_float(b.get("qty"), default=0.0) or 0.0) > 0.0
                and (safe_pos_float(b.get("price"), default=0.0) or 0.0) > 0.0
            ]
            reconciled_external_buy = bool(
                buys_after
                and cyc_qty_cur > 1e-8
                and float(entry_ts_ms) > (safe_pos_float(buys_after[-1].get("ts_ms"), default=0.0) or 0.0) + 5000.0
                and abs(sum(float(b["qty"]) for b in buys_after) - cyc_qty_cur) <= max(1e-4, cyc_qty_cur * 0.02)
            )
            rem_q_ro = safe_pos_float(ro_c.get("active_cycle_rem_qty"), default=0.0) or 0.0
            live_held = live_held_by_venue_sym.get((c_venue, str(c_sym).upper()))
            already_deducted_partial = bool(
                not buys_after
                and cyc_qty_cur > 1e-8
                and (
                    (sell_ts_ms is not None and float(sell_ts_ms) >= ls_ts_ms - 5000.0)
                    or (rem_q_ro > 1e-8 and abs(rem_q_ro - cyc_qty_cur) <= max(1e-4, max(rem_q_ro, cyc_qty_cur) * 0.05))
                    or (live_held is not None and live_held > 1e-8 and abs(live_held - cyc_qty_cur) <= max(1e-4, max(live_held, cyc_qty_cur) * 0.05))
                )
            )

            def _sync_exit_kind(c_dict: Dict[str, Any], sell_p: float, rem_q: float) -> None:
                if sell_p <= 0.0:
                    return
                ref_avg = (
                    safe_pos_float(ro_c.get("active_avg_buy_px"), default=None)
                    or safe_pos_float(c_dict.get("avg_cost_px"), default=None)
                    or safe_pos_float(c_dict.get("entry_price"), default=None)
                    or safe_pos_float(c_dict.get("last_buy_price"), default=None)
                )
                if ref_avg is not None and ref_avg > 0.0:
                    if sell_p < ref_avg * 0.995 and str(c_dict.get("last_exit_kind") or "").strip().upper() == "TP":
                        c_dict["last_exit_kind"] = "REDUCE"
                    elif sell_p >= ref_avg * 1.001 and not c_dict.get("last_exit_kind"):
                        c_dict["last_exit_kind"] = "TP"

            if already_deducted_partial:
                if ls_px > 0:
                    cyc["last_sell_price"] = round(ls_px, avg_dec)
                    _sync_exit_kind(cyc, ls_px, cyc_qty_cur)
            elif (
                not buys_after
                and ls_ts_ms > float(entry_ts_ms)
                and cyc_qty_cur > 1e-8
                and 0.0 < ls_qty < cyc_qty_cur - 1e-8
                and (sell_ts_ms is None or float(sell_ts_ms) < ls_ts_ms - 5000.0)
                and (
                    (live_held is not None and abs(live_held - (cyc_qty_cur - ls_qty)) <= max(1e-4, max(live_held, cyc_qty_cur - ls_qty) * 0.05))
                    or (live_held is None and rem_q_ro > 1e-8 and abs(rem_q_ro - (cyc_qty_cur - ls_qty)) <= max(1e-4, max(rem_q_ro, cyc_qty_cur - ls_qty) * 0.05))
                )
            ):
                rem_after_sell = cyc_qty_cur - ls_qty
                avg_c = safe_pos_float(cyc.get("avg_cost_px"), default=None) or (
                    (safe_pos_float(cyc.get("cost_usd"), default=0.0) or 0.0) / cyc_qty_cur
                )
                cyc["qty"] = round(rem_after_sell, 6)
                if avg_c and avg_c > 0:
                    cyc["cost_usd"] = round(rem_after_sell * avg_c, 2)
                if ls_px > 0:
                    cyc["last_sell_price"] = round(ls_px, avg_dec)
                    _sync_exit_kind(cyc, ls_px, rem_after_sell)
            elif (ls_ts_ms > float(entry_ts_ms) and cyc_qty_cur > 1e-8 and ls_qty >= cyc_qty_cur * 0.95) or reconciled_external_buy:
                px_c = resolve_symbol_price(c_sym, prices_and_trends)
                eff_rem_wallet = (
                    live_held
                    if (live_held is not None and live_held > 1e-6)
                    else (rem_q_ro if (live_held is None and rem_q_ro > 1e-6) else 0.0)
                )
                has_non_dust_wallet = eff_rem_wallet > 1e-6 and (px_c <= 0.0 or eff_rem_wallet * px_c >= 5.0)
                if buys_after:
                    new_qty = sum(float(b["qty"]) for b in buys_after)
                    new_cost = sum(float(b["qty"]) * float(b["price"]) for b in buys_after)
                    new_avg = round(new_cost / new_qty, avg_dec)
                    new_dd = round(((px_c - new_avg) / new_avg) * 100.0, 2) if px_c > 0 and new_avg > 0 else None
                    cyc.update({
                        "qty": round(new_qty, 6), "cost_usd": round(new_cost, 2), "avg_cost_px": new_avg,
                        "entry_price": round(float(buys_after[0]["price"]), avg_dec),
                        "last_buy_price": round(float(buys_after[-1]["price"]), avg_dec),
                        "dca_buys": max(0, len(buys_after) - 1), "dd_pct": new_dd, "pnl_vs_avg_pct": new_dd,
                    })
                elif has_non_dust_wallet:
                    eff_avg_w = (
                        safe_pos_float(ro_c.get("active_avg_buy_px"), default=None)
                        or safe_pos_float(cyc.get("avg_cost_px"), default=None)
                        or safe_pos_float(cyc.get("entry_price"), default=None)
                        or safe_pos_float(cyc.get("last_buy_price"), default=None)
                    )
                    eff_avg_r = round(eff_avg_w, avg_dec) if eff_avg_w and eff_avg_w > 0 else None
                    new_dd_w = round(((px_c - eff_avg_r) / eff_avg_r) * 100.0, 2) if (px_c > 0 and eff_avg_r and eff_avg_r > 0) else None
                    cyc.update({
                        "qty": round(eff_rem_wallet, 6),
                        "cost_usd": round(eff_rem_wallet * eff_avg_r, 2) if eff_avg_r else 0.0,
                        "avg_cost_px": eff_avg_r,
                        "dd_pct": new_dd_w,
                        "pnl_vs_avg_pct": new_dd_w,
                    })
                    if eff_avg_r and not cyc.get("entry_price"):
                        cyc["entry_price"] = eff_avg_r
                else:
                    cyc.update({"qty": 0.0, "cost_usd": 0.0, "avg_cost_px": None, "dd_pct": None, "pnl_vs_avg_pct": None})
                if ls_px > 0:
                    cyc["last_sell_price"] = round(ls_px, avg_dec)
                    _sync_exit_kind(cyc, ls_px, safe_nonneg_float(cyc.get("qty"), default=0.0) or 0.0)
                if not has_non_dust_wallet and not buys_after:
                    cyc.pop("pos_peak", None)
                    cyc.pop("dd_from_peak_pct", None)
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
                cyc_qty is not None
                and fb_cur >= 0.0
                and lb_cur > 0.0
                and abs(cyc_qty - fb_cur) <= max(1e-4, fb_cur * 0.02)
            ):
                cyc_qty = round(fb_cur + lb_cur, 6)
                cyc["qty"] = cyc_qty
                if (c_avg_px := safe_pos_float(cyc.get("avg_cost_px"), default=None)) is not None:
                    cyc["cost_usd"] = round(cyc_qty * c_avg_px, 2)
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
                if p_venue in ("t212", "kraken", "hyperliquid"):
                    _cyc_avg = safe_pos_float(cyc.get("avg_cost_px") or cyc.get("entry_price"), default=0.0) or 0.0
                    _cyc_lb = safe_pos_float(cyc.get("last_buy_price") or cyc.get("entry_price"), default=None)
                    _cyc_ls = safe_pos_float(cyc.get("last_sell_price"), default=None)
                    _cyc_ls_ts = cyc_sell_ts_map.get((p_venue, p_sym))
                    _created_ro = False
                    if (
                        not isinstance(ro, dict)
                        and (p_venue == "t212" or p_sym not in recent_orders)
                        and (_cyc_lb is not None or _cyc_ls is not None or _cyc_avg > 0)
                    ):
                        ro = {
                            "last_action_side": "BUY" if cyc_has_open_qty else ("SELL" if _cyc_ls is not None else "BUY"),
                            "last_action_actor": "BOT", "last_action_actor_tag": "BOT",
                            "last_buy": None, "last_sell": None, "active_avg_buy_px": None, "recent_buys": [],
                        }
                        recent_orders[p_sym] = v_ro_map[p_sym] = ro
                        _created_ro = True
                    if isinstance(ro, dict) and (p_venue == "t212" or _created_ro):
                        if _cyc_avg > 0:
                            ro["active_avg_buy_px"] = round(_cyc_avg, 6)
                        if ro.get("last_buy") is None and _cyc_lb is not None:
                            lb_d: Dict[str, Any] = {"price": _cyc_lb, "actor": "BOT", "actor_tag": "BOT"}
                            if not ro.get("recent_buys"):
                                lb_d["qty"] = cyc.get("qty") if not cyc.get("dca_buys") and cyc_has_open_qty else None
                            ro["last_buy"] = lb_d
                        if ro.get("last_sell") is None and _cyc_ls is not None and (p_venue == "t212" or _cyc_ls_ts is not None or not cyc_has_open_qty):
                            ro["last_sell"] = {"price": _cyc_ls, "ts_ms": _cyc_ls_ts, "actor": "BOT", "actor_tag": "BOT"}
                set_pos_anchor(pos, "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct", cyc.get("avg_cost_px"), pos_px_f, overwrite=False)
                set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", cyc.get("pos_peak"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", cyc.get("last_buy_price"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", cyc.get("last_sell_price"), pos_px_f, overwrite=False)
            if cyc.get("last_buy_price") is None and pos.get("last_buy_px") is not None:
                cyc["last_buy_price"] = pos["last_buy_px"]
            if cyc.get("last_sell_price") is None and pos.get("last_sell_px") is not None:
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
    enrich_recent_orders_markout(recent_orders, prices_and_trends)
    return recent_orders


def enrich_recent_orders_markout(
    recent_orders: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
) -> None:
    """Enrich recent BUY and SELL fills with post-trade markout in bps vs live spot price."""
    if not isinstance(recent_orders, dict) or not isinstance(prices_and_trends, dict):
        return
    for r_sym, ro in recent_orders.items():
        if not isinstance(ro, dict):
            continue
        sym_clean = str(r_sym).split(":", 1)[-1].strip().upper()
        cur_px = resolve_symbol_price(sym_clean, prices_and_trends)
        if cur_px <= 0.0:
            continue
        if isinstance(lb := ro.get("last_buy"), dict) and (lb_px := safe_pos_float(lb.get("price"), default=None)) is not None:
            lb["markout_now_bps"] = round(((cur_px - lb_px) / lb_px) * 10000.0, 1)
        if isinstance(ls := ro.get("last_sell"), dict) and (ls_px := safe_pos_float(ls.get("price"), default=None)) is not None:
            ls["markout_now_bps"] = round(((ls_px - cur_px) / ls_px) * 10000.0, 1)
        b_num = b_den = 0.0
        for b in (ro.get("recent_buys") or []):
            if isinstance(b, dict) and (b_px := safe_pos_float(b.get("price"), default=None)) is not None:
                m_bps = round(((cur_px - b_px) / b_px) * 10000.0, 1)
                b["markout_now_bps"] = m_bps
                w = (safe_pos_float(b.get("qty"), default=1.0) or 1.0) * b_px
                b_num += m_bps * w
                b_den += w
        if b_den > 0.0:
            ro["buy_markout_bps"] = round(b_num / b_den, 1)
        elif isinstance(ro.get("last_buy"), dict) and ro["last_buy"].get("markout_now_bps") is not None:
            ro["buy_markout_bps"] = ro["last_buy"]["markout_now_bps"]
        s_num = s_den = 0.0
        for s in (ro.get("recent_sells") or []):
            if isinstance(s, dict) and (s_px := safe_pos_float(s.get("price"), default=None)) is not None:
                m_bps = round(((s_px - cur_px) / s_px) * 10000.0, 1)
                s["markout_now_bps"] = m_bps
                w = (safe_pos_float(s.get("qty"), default=1.0) or 1.0) * s_px
                s_num += m_bps * w
                s_den += w
        if s_den > 0.0:
            ro["sell_markout_bps"] = round(s_num / s_den, 1)
        elif isinstance(ro.get("last_sell"), dict) and ro["last_sell"].get("markout_now_bps") is not None:
            ro["sell_markout_bps"] = ro["last_sell"]["markout_now_bps"]


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


def enrich_open_limit_orders_tca(
    open_orders: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Enrich resting open limit orders with Orderbook Wall Queue Position, Front-Run Price, and 1h Fill Probability (Layer II.3)."""
    import math as _math

    if not isinstance(open_orders, list) or not open_orders:
        return open_orders
    for o in open_orders:
        if not isinstance(o, dict):
            continue
        sym = str(o.get("symbol") or "").strip().upper()
        side = str(o.get("side") or "").strip().upper()
        lim_px = safe_pos_float(o.get("price"), default=None)
        if not sym or lim_px is None or lim_px <= 0.0:
            continue
        p_entry = casefold_get(prices_and_trends, sym) if isinstance(prices_and_trends, dict) else None
        m_entry = casefold_get(microstructure, sym) if isinstance(microstructure, dict) else None
        if not isinstance(m_entry, dict):
            for cand in cache_symbol_variants(sym, include_base=False):
                cand_m = casefold_get(microstructure, cand) if isinstance(microstructure, dict) else None
                if isinstance(cand_m, dict) and compatible_quote_suffixes(cand) == compatible_quote_suffixes(sym):
                    m_entry = cand_m
                    break
        spot_px = (
            safe_pos_float((m_entry or {}).get("ob_mid_price"), default=None)
            or safe_pos_float((p_entry or {}).get("price"), default=None)
            or resolve_symbol_price(sym, prices_and_trends)
        )
        if spot_px and spot_px > 0.0 and o.get("dist_from_spot_pct") is None:
            o["dist_from_spot_pct"] = round(((lim_px - spot_px) / spot_px) * 100.0, 2)
        dist_pct = abs(safe_float(o.get("dist_from_spot_pct"), default=0.0) or 0.0)
        vol_1h = max(0.35, safe_pos_float((p_entry or {}).get("vol_1h_pct"), default=1.1) or 1.1)
        dec = 2 if lim_px >= 100.0 else (4 if lim_px >= 1.0 else 6)
        queue_penalty = 1.0
        if isinstance(m_entry, dict):
            sp_bps = safe_nonneg_float(m_entry.get("spread_bps"), default=2.0) or 2.0
            best_bid_px = safe_pos_float(m_entry.get("best_bid"), default=None) or (
                spot_px * (1.0 - sp_bps / 20000.0) if spot_px and spot_px > 0.0 else None
            )
            best_ask_px = safe_pos_float(m_entry.get("best_ask"), default=None) or (
                spot_px * (1.0 + sp_bps / 20000.0) if spot_px and spot_px > 0.0 else None
            )
            tick_step = 10.0 ** (-dec)
            if side == "SELL":
                aw_px = safe_pos_float(m_entry.get("ask_wall_px"), default=None)
                aw_usd = safe_pos_float(m_entry.get("ask_wall_usd"), default=0.0) or 0.0
                if aw_px is not None and aw_px > 0.0 and aw_usd >= 5000.0 and (not spot_px or aw_px >= spot_px * 1.0010):
                    step_fr = max(tick_step, min(aw_px * 0.0001, 5.0))
                    raw_fr_ask = aw_px - step_fr
                    floor_ask = max(best_ask_px or 0.0, (spot_px or 0.0) * 1.0002)
                    if floor_ask > 0.0:
                        raw_fr_ask = max(floor_ask + tick_step, raw_fr_ask)
                    fr_ask_px = round(min(aw_px, raw_fr_ask), dec)
                    if lim_px > aw_px * 1.0003:
                        o["wall_queue_pos"] = "BEHIND_ASK_WALL"
                        o["ask_wall_ahead_px"] = aw_px
                        o["ask_wall_ahead_usd"] = round(aw_usd, 0)
                        o["recommended_front_run_px"] = fr_ask_px
                        queue_penalty = 0.60
                    elif abs(lim_px - aw_px) / aw_px <= 0.0003:
                        o["wall_queue_pos"] = "AT_ASK_WALL"
                        o["recommended_front_run_px"] = fr_ask_px
                        queue_penalty = 0.78
                    else:
                        o["wall_queue_pos"] = "FRONT_OF_ASK_WALL"
            elif side == "BUY":
                bw_px = safe_pos_float(m_entry.get("bid_wall_px"), default=None)
                bw_usd = safe_pos_float(m_entry.get("bid_wall_usd"), default=0.0) or 0.0
                if bw_px is not None and bw_px > 0.0 and bw_usd >= 5000.0 and (not spot_px or bw_px <= spot_px * 0.9990):
                    step_fr = max(tick_step, min(bw_px * 0.0001, 5.0))
                    raw_fr_bid = bw_px + step_fr
                    ceil_bid = min(
                        best_bid_px if best_bid_px and best_bid_px > 0.0 else bw_px * 2.0,
                        (spot_px * 0.9998) if spot_px and spot_px > 0.0 else bw_px * 2.0,
                    )
                    if ceil_bid > 0.0:
                        raw_fr_bid = min(ceil_bid - tick_step, raw_fr_bid)
                    fr_bid_px = round(max(bw_px, raw_fr_bid), dec)
                    if lim_px < bw_px * 0.9997:
                        o["wall_queue_pos"] = "BEHIND_BID_WALL"
                        o["bid_wall_ahead_px"] = bw_px
                        o["bid_wall_ahead_usd"] = round(bw_usd, 0)
                        o["recommended_front_run_px"] = fr_bid_px
                        queue_penalty = 0.60
                    elif abs(lim_px - bw_px) / bw_px <= 0.0003:
                        o["wall_queue_pos"] = "AT_BID_WALL"
                        o["recommended_front_run_px"] = fr_bid_px
                        queue_penalty = 0.78
                    else:
                        o["wall_queue_pos"] = "FRONT_OF_BID_WALL"
        raw_prob = 100.0 * _math.exp(-0.5 * ((dist_pct / vol_1h) ** 2)) * queue_penalty
        o["fill_prob_1h_pct"] = round(min(99.0, max(1.0, raw_prob)), 1)
    return open_orders


def compute_enterprise_portfolio_risk_metrics(
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute Global Enterprise Portfolio Risk Engine metrics (Unified NAV, Cash %, BTC-Beta, 24h Correlation, 95% VaR, LIFO Alpha)."""
    import math as _math

    if not isinstance(portfolio_summary, dict):
        return {}
    t_pos = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
    vq_cash = portfolio_summary.get("venue_quote_cash") if isinstance(portfolio_summary.get("venue_quote_cash"), dict) else {}
    rec_orders = portfolio_summary.get("recent_orders") if isinstance(portfolio_summary.get("recent_orders"), dict) else {}

    venue_pos_usd: Dict[str, float] = {}
    venue_cash_usd: Dict[str, float] = {}
    venue_free_cash_usd: Dict[str, float] = {}
    venue_locked_quote_usd: Dict[str, float] = {}
    pos_exposures: List[Tuple[str, float, float, float, float]] = []  # (base, val_usd, beta_btc, corr_btc, daily_vol_pct)

    for p in t_pos:
        if not isinstance(p, dict):
            continue
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        val_u = (safe_pos_float(p.get("value_usd"), default=0.0) or 0.0) + (safe_pos_float(p.get("locked_base_usd"), default=0.0) or 0.0)
        venue_pos_usd[v_name] = venue_pos_usd.get(v_name, 0.0) + val_u
        if val_u >= 1.0:
            sym = str(p.get("symbol") or "").strip().upper()
            base = canonical_base_asset(sym)
            p_info = casefold_get(prices_and_trends, sym) if isinstance(prices_and_trends, dict) else {}
            if not isinstance(p_info, dict):
                p_info = {}
            is_eq = sym.endswith(("_EQ", "_US_EQ")) or v_name == "t212"
            beta_v = 1.0 if base == "BTC" else (safe_float(p_info.get("beta_btc_24h"), default=(0.35 if is_eq else 1.15)) or (0.35 if is_eq else 1.15))
            corr_v = 1.0 if base == "BTC" else (safe_float(p_info.get("corr_btc_24h"), default=(0.30 if is_eq else 0.78)) or (0.30 if is_eq else 0.78))
            vol_1h = safe_pos_float(p_info.get("vol_1h_pct"), default=1.0) or 1.0
            daily_vol_pct = min(25.0, max(0.8, vol_1h * _math.sqrt(24.0)))
            pos_exposures.append((base, val_u, beta_v, corr_v, daily_vol_pct))

    for c in vq_cash.values():
        if not isinstance(c, dict):
            continue
        v_name = normalize_venue_name(c.get("venue")) or "unknown"
        f_u = safe_pos_float(c.get("free"), default=0.0) or 0.0
        l_u = safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0
        venue_free_cash_usd[v_name] = venue_free_cash_usd.get(v_name, 0.0) + f_u
        venue_locked_quote_usd[v_name] = venue_locked_quote_usd.get(v_name, 0.0) + l_u
        venue_cash_usd[v_name] = venue_cash_usd.get(v_name, 0.0) + f_u + l_u

    all_venues = sorted(set(venue_pos_usd) | set(venue_cash_usd))
    venue_nav: Dict[str, float] = {}
    for v_k in all_venues:
        v_tot = round(venue_pos_usd.get(v_k, 0.0) + venue_cash_usd.get(v_k, 0.0), 2)
        if v_tot > 0.0:
            venue_nav[v_k] = v_tot

    # Reconcile Binance or Hyperliquid if snapshot total exceeds partial instrument sum
    bn_snap = safe_pos_float(portfolio_summary.get("live_binance_equity_usdc") or portfolio_summary.get("total_value_usdc"), default=0.0) or 0.0
    if bn_snap > venue_nav.get("binance", 0.0):
        venue_nav["binance"] = round(bn_snap, 2)
    hl_ro = portfolio_summary.get("hyperliquid_readonly") if isinstance(portfolio_summary.get("hyperliquid_readonly"), dict) else {}
    hl_snap = safe_pos_float(hl_ro.get("live_reconciled_spot_usd") or hl_ro.get("total_equity_usd"), default=0.0) or 0.0
    if hl_snap > venue_nav.get("hyperliquid", 0.0):
        venue_nav["hyperliquid"] = round(hl_snap, 2)

    invested_usd = round(sum(venue_pos_usd.values()), 2)
    free_cash_usd = round(sum(venue_free_cash_usd.values()), 2)
    locked_quote_usd = round(sum(venue_locked_quote_usd.values()), 2)
    total_quote_reserve_usd = round(free_cash_usd + locked_quote_usd, 2)
    unified_nav = round(max(sum(venue_nav.values()), invested_usd + total_quote_reserve_usd), 2)
    if unified_nav <= 0.0:
        return {}

    free_cash_pct = round((free_cash_usd / unified_nav) * 100.0, 1)
    locked_quote_pct = round((locked_quote_usd / unified_nav) * 100.0, 1)
    total_quote_reserve_pct = round((total_quote_reserve_usd / unified_nav) * 100.0, 1)
    beta_adj_usd = sum(val_u * beta_v for _, val_u, beta_v, _, _ in pos_exposures)
    portfolio_btc_beta = round(beta_adj_usd / unified_nav, 2)

    # Correlation & Parametric 95% Intraday VaR (accounting for pairwise correlation via BTC factor)
    corr_map: Dict[str, float] = {}
    for base, _, _, corr_v, _ in pos_exposures:
        if base != "BTC":
            corr_map[base] = round(corr_v, 2)
    non_btc_corrs = [c for b, _, _, c, _ in pos_exposures if b != "BTC"]
    avg_corr_24h = round(sum(non_btc_corrs) / len(non_btc_corrs), 2) if non_btc_corrs else (1.0 if pos_exposures else 0.0)

    var_cov_sum = 0.0
    for i, (_, val_i, _, corr_i, dvol_i) in enumerate(pos_exposures):
        sig_i = val_i * (dvol_i / 100.0)
        for j, (_, val_j, _, corr_j, dvol_j) in enumerate(pos_exposures):
            sig_j = val_j * (dvol_j / 100.0)
            rho_ij = 1.0 if i == j else max(-0.5, min(0.98, corr_i * corr_j))
            var_cov_sum += sig_i * sig_j * rho_ij
    port_sigma_usd = _math.sqrt(max(0.0, var_cov_sum))
    var_95_usd = round(1.645 * port_sigma_usd, 2)
    var_95_pct = round((var_95_usd / unified_nav) * 100.0, 2)
    cvar_975_usd = round(2.34 * port_sigma_usd, 2)
    cvar_975_pct = round((cvar_975_usd / unified_nav) * 100.0, 2)

    base_tranche = min(2000.0, max(100.0, unified_nav * 0.035))
    var_scale = min(1.4, max(0.5, 2.5 / max(0.8, var_95_pct))) if var_95_pct > 0.0 else 1.0
    beta_scale = min(1.25, max(0.6, 0.85 / max(0.4, portfolio_btc_beta))) if portfolio_btc_beta > 0.0 else 1.0
    vol_target_tranche_usd = round(min(2000.0, max(50.0, base_tranche * var_scale * beta_scale)), 0)

    concentration_regime = (
        "HIGH_CORR_CONCENTRATED_BETA"
        if len(pos_exposures) >= 2 and avg_corr_24h >= 0.80 and portfolio_btc_beta >= 0.65
        else ("DEFENSIVE_HIGH_CASH" if free_cash_pct >= 55.0 else "BALANCED_DIVERSIFIED")
    )

    enrich_recent_orders_markout(rec_orders, prices_and_trends)
    lifo_pnl_total = 0.0
    est_fee_drag_usd = 0.0
    entry_m_num = entry_m_den = exit_m_num = exit_m_den = 0.0
    wins_cnt = total_fills_cnt = 0
    for r_sym, r_info in rec_orders.items():
        if not isinstance(r_info, dict):
            continue
        lifo_pnl_total += safe_float(r_info.get("lifo_realized_pnl_usd"), default=0.0) or 0.0
        _, _, sym_taker_bps, _ = _infer_symbol_venue_and_fees_bps(
            r_sym,
            (microstructure or {}).get(r_sym) if isinstance(microstructure, dict) else {},
        )
        fee_rate = (sym_taker_bps / 10000.0) if sym_taker_bps > 0.0 else 0.0010
        for side_list_k, single_k, is_buy_side in (
            ("recent_buys", "last_buy", True),
            ("recent_sells", "last_sell", False),
        ):
            side_rows = r_info.get(side_list_k)
            if not isinstance(side_rows, list) or not side_rows:
                single_row = r_info.get(single_k)
                side_rows = [single_row] if isinstance(single_row, dict) else []
            for tr in side_rows:
                if isinstance(tr, dict):
                    q_t = safe_pos_float(tr.get("qty"), default=0.0) or 0.0
                    p_t = safe_pos_float(tr.get("price"), default=0.0) or 0.0
                    notional_t = q_t * p_t if (q_t > 0.0 and p_t > 0.0) else 0.0
                    if notional_t > 0.0:
                        est_fee_drag_usd += notional_t * fee_rate
                    if (m_bps := safe_float(tr.get("markout_now_bps"), default=None)) is not None:
                        w_t = notional_t if notional_t > 0.0 else 1.0
                        if is_buy_side:
                            entry_m_num += m_bps * w_t
                            entry_m_den += w_t
                        else:
                            exit_m_num += m_bps * w_t
                            exit_m_den += w_t
                        total_fills_cnt += 1
                        if m_bps > 0.0:
                            wins_cnt += 1

    cross_venue_routing: Dict[str, Any] = {}
    if isinstance(microstructure, dict):
        for m_sym, m_dict in sorted(microstructure.items()):
            if not isinstance(m_dict, dict) or not m_dict.get("best_buy_venue"):
                continue
            b_asset = canonical_base_asset(m_sym)
            if b_asset and b_asset not in cross_venue_routing:
                cross_venue_routing[b_asset] = {
                    "best_buy": m_dict.get("best_buy_venue"),
                    "best_sell": m_dict.get("best_sell_venue"),
                    "net_buy_edge_bps": abs(safe_float(m_dict.get("net_buy_edge_vs_peer_bps"), default=0.0) or 0.0),
                    "net_sell_edge_bps": abs(safe_float(m_dict.get("net_sell_edge_vs_peer_bps"), default=0.0) or 0.0),
                }

    out_risk: Dict[str, Any] = {
        "unified_nav_usd": unified_nav,
        "invested_exposure_usd": invested_usd,
        "global_free_cash_usd": free_cash_usd,
        "global_free_cash_pct": free_cash_pct,
        "locked_quote_usd": locked_quote_usd,
        "locked_quote_pct": locked_quote_pct,
        "total_quote_reserve_usd": total_quote_reserve_usd,
        "total_quote_reserve_pct": total_quote_reserve_pct,
        "venue_nav_breakdown_usd": venue_nav,
        "portfolio_btc_beta": portfolio_btc_beta,
        "beta_adjusted_exposure_usd": round(beta_adj_usd, 2),
        "avg_corr_vs_btc_24h": avg_corr_24h,
        "corr_vs_btc_24h": corr_map,
        "concentration_regime": concentration_regime,
        "intraday_var_95_usd": var_95_usd,
        "intraday_var_95_pct": var_95_pct,
        "intraday_cvar_975_usd": cvar_975_usd,
        "intraday_cvar_975_pct": cvar_975_pct,
        "vol_target_tranche_usd": vol_target_tranche_usd,
    }
    if abs(lifo_pnl_total) >= 0.01:
        out_risk["subcycle_lifo_realized_pnl_usd"] = round(lifo_pnl_total, 2)
    if est_fee_drag_usd >= 0.01:
        out_risk["est_recent_fee_drag_usd"] = round(est_fee_drag_usd, 2)
    if entry_m_den > 0.0:
        out_risk["avg_entry_markout_bps"] = round(entry_m_num / entry_m_den, 1)
    if exit_m_den > 0.0:
        out_risk["avg_exit_markout_bps"] = round(exit_m_num / exit_m_den, 1)
    if total_fills_cnt > 0:
        out_risk["post_trade_win_rate_pct"] = round((wins_cnt / total_fills_cnt) * 100.0, 1)
    if cross_venue_routing:
        out_risk["cross_venue_routing"] = cross_venue_routing
    return out_risk

