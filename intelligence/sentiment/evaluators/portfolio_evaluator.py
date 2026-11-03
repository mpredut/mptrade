"""Stateless order-lot accounting (FIFO/LIFO), limit-order TCA, venue fee/limit resolution, and enterprise portfolio risk evaluators."""
from __future__ import annotations

import math
import sys as _sys
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    normalize_venue_name,
    resolve_kraken_lookup_symbol,
)
from intelligence.external.collectors.orderbook_depth import round_spread_bps
from intelligence.macro_sentiment_ssot import (
    clean_symbol_key,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from intelligence.sentiment.evaluators.price_series_evaluator import _prices_diverged, pct_delta, round_opt_px
from instrument_registry import casefold_get
from providers.base import (
    _EXEC_QTY_KEYS,
    _UNFILLED_ORDER_STATUSES,
    aggregate_fills_by_order_id,
    extract_executed_qty,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
    extract_timestamp_ms,
    format_actor_tag,
    normalize_timestamp_ms,
    resolve_order_origin,
)
from providers.execution_audit import extract_intent_order_type, extract_intent_side, intent_venue_client_order_ids
from providers.quantity import venue_fee_cap_quantity


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
        pos[pct_key] = pct_delta(cur_px, a_f)


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
    dd_pct = pct_delta(px_f, avg_cost) if avg_cost and px_f > 0 else None
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
            cyc_entry["dd_from_peak_pct"] = pct_delta(px_f, eff_peak)
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


def extract_symbol_recent_fills(
    orders_list: list,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[Dict[str, tuple[Optional[str], Optional[str]]]] = None,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
    symbol: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Extract last action side/actor, active cycle avg BUY price, last BUY/SELL fills, and recent tranches."""
    if not isinstance(orders_list, list):
        orders_list = []
    if not orders_list and not ai_intents:
        return None
    _explicit_exec_keys = (*_EXEC_QTY_KEYS, "filled", "origSz", "resting", "msg")
    valid = [
        o for o in orders_list
        if isinstance(o, dict) and (
            extract_executed_qty(o, default=0.0) > 0.0
            or (str(o.get("status") or "").strip().upper() not in _UNFILLED_ORDER_STATUSES and not any(k in o for k in _explicit_exec_keys))
        )
    ]
    fallback_sym = str(symbol or "").strip().upper()
    if not fallback_sym:
        for o_item in valid:
            if s_cand := str(o_item.get("symbol") or o_item.get("pair") or "").strip().upper():
                fallback_sym = s_cand
                break
    if not valid and not (fallback_sym and ai_intents):
        return None
    ai_oids: set[str] = set(known_ai_order_ids or ())
    valid_oids: set[str] = {str(oid_x) for x in valid if (oid_x := extract_order_id(x, default=""))}
    max_valid_ts_s = max(((extract_timestamp_ms(x, default=0) or 0) / 1000.0 for x in valid), default=0.0)
    parsed_ai_intents: List[tuple[str, str, float, float, str]] = []
    for it in ai_intents or ():
        if not isinstance(it, dict) or str(it.get("status") or "").strip().upper() not in ("EXECUTED", "FILLED", "PARTIALLY_FILLED", "CLOSED"):
            continue
        exec_pay = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
        it_oid = extract_order_id(exec_pay, default="") or extract_order_id(it, default="")
        if it_oid:
            ai_oids.add(str(it_oid))
            if str(it_oid) in valid_oids:
                continue
        i_side = extract_intent_side(it) or str(it.get("decision") or "").strip().upper()
        i_ts_s = safe_pos_float(it.get("timestamp") or it.get("executed_ts"), default=0.0) or 0.0
        if i_side not in ("BUY", "SELL") or i_ts_s <= 0.0:
            continue
        i_sym = str(it.get("symbol") or exec_pay.get("symbol") or exec_pay.get("ticker") or "").strip().upper()
        hist = it.get("history") if isinstance(it.get("history"), list) else []
        last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
        exec_q = extract_executed_qty(exec_pay, default=0.0) or extract_executed_qty(it, default=0.0) or 0.0
        exec_p = extract_order_price(exec_pay, default=0.0) or extract_order_price(it, default=0.0) or safe_pos_float(it.get("executed_avg_price"), default=0.0) or 0.0
        exec_st = str(exec_pay.get("status") or it.get("exchange_status") or "").strip().upper()
        sym_matches_target = bool(
            fallback_sym
            and i_sym
            and (
                i_sym == fallback_sym
                or (
                    canonical_base_asset(i_sym) == canonical_base_asset(fallback_sym)
                    and compatible_quote_suffixes(i_sym) == compatible_quote_suffixes(fallback_sym)
                )
            )
        )
        if (
            sym_matches_target
            and exec_st in ("FILLED", "PARTIALLY_FILLED", "CLOSED")
            and exec_q > 0.0
            and exec_p > 0.0
            and (not valid or i_ts_s >= max_valid_ts_s - 60.0)
            and not any(
                extract_order_side(vo) == i_side
                and abs(((extract_timestamp_ms(vo, default=0) or 0) / 1000.0) - i_ts_s) <= 86400.0
                and abs((extract_executed_qty(vo, default=0.0) or 0.0) - exec_q) <= max(1e-5, exec_q * 0.01)
                for vo in valid
            )
        ):
            synth_oid = str(it_oid or it.get("intent_id") or "")
            valid.append({
                "symbol": i_sym or fallback_sym,
                "orderId": synth_oid or None,
                "clientOrderId": exec_pay.get("clientOrderId") or exec_pay.get("client_order_id") or it.get("intent_id"),
                "side": i_side,
                "price": exec_p,
                "qty": exec_q,
                "executedQty": exec_q,
                "timestamp": int(round(i_ts_s * 1000.0)),
                "status": "FILLED",
            })
            if synth_oid:
                ai_oids.add(synth_oid)
                valid_oids.add(synth_oid)
            continue
        i_qty = (
            exec_q
            or extract_orig_qty(exec_pay, default=0.0)
            or safe_pos_float(last_h.get("qty"), default=0.0)
            or 0.0
        )
        parsed_ai_intents.append((i_sym, i_side, i_ts_s, i_qty, extract_intent_order_type(it)))
    if not valid:
        return None

    def _matches_ai_intent(o_dict: Dict[str, Any]) -> bool:
        o_side = extract_order_side(o_dict)
        o_ts_s = (extract_timestamp_ms(o_dict, default=0) or 0) / 1000.0
        o_qty = extract_executed_qty(o_dict, default=0.0) or 0.0
        o_sym = str(o_dict.get("symbol") or o_dict.get("pair") or fallback_sym).strip().upper()
        for i_sym, i_side, i_ts_s, i_qty, i_otype in parsed_ai_intents:
            if i_side != o_side or o_ts_s <= 0.0 or o_ts_s < i_ts_s - 5.0:
                continue
            if i_sym and o_sym and (canonical_base_asset(i_sym) != canonical_base_asset(o_sym) or compatible_quote_suffixes(i_sym) != compatible_quote_suffixes(o_sym)):
                continue
            if (i_qty > 0.0 and o_qty > 0.0 and abs(o_qty - i_qty) <= max(1e-5, i_qty * 0.01) and o_ts_s <= i_ts_s + 86400.0) or (i_otype == "MARKET" and o_ts_s <= i_ts_s + 120.0):
                return True
        return False

    def _mark_ai_origin(o_dict: Dict[str, Any]) -> None:
        if oid_m := extract_order_id(o_dict, default=""):
            oid_s = str(oid_m)
            ai_oids.add(oid_s)
            prev_m = origin_by_oid.get(oid_s)
            origin_by_oid[oid_s] = ("AI", prev_m[1] if prev_m else None, prev_m[2] if prev_m else None)
        else:
            side_ts_m = (extract_order_side(o_dict), extract_timestamp_ms(o_dict, default=None))
            prev_m = origin_by_side_ts.get(side_ts_m)
            origin_by_side_ts[side_ts_m] = ("AI", prev_m[1] if prev_m else None, prev_m[2] if prev_m else None)

    origin_by_oid: Dict[str, tuple[str, Optional[str], Optional[str]]] = {}
    origin_by_side_ts: Dict[tuple[str, Any], tuple[str, Optional[str], Optional[str]]] = {}
    for o in valid:
        act_v, mod_v, rule_v = resolve_order_origin(o, ai_oids, audit_origin_map)
        if act_v == "BOT" and not (mod_v or rule_v or o.get("clientOrderId") or o.get("client_order_id")) and parsed_ai_intents:
            if _matches_ai_intent(o):
                act_v = "AI"
        if oid_v := extract_order_id(o, default=""):
            oid_str = str(oid_v)
            if act_v == "AI":
                ai_oids.add(oid_str)
            prev = origin_by_oid.get(oid_str)
            if prev is None or (not prev[1] and mod_v) or (not prev[2] and rule_v) or act_v == "AI":
                origin_by_oid[oid_str] = (
                    "AI" if (act_v == "AI" or (prev and prev[0] == "AI")) else "BOT",
                    mod_v or (prev[1] if prev else None),
                    rule_v or (prev[2] if prev else None),
                )
        else:
            origin_by_side_ts[(extract_order_side(o), extract_timestamp_ms(o, default=None))] = (act_v, mod_v, rule_v)

    if any(o.get(k) is not None for o in valid for k in ("orderId", "order_id", "ordertxid", "oid")):
        valid = aggregate_fills_by_order_id(valid) or valid

    if parsed_ai_intents:
        for o in valid:
            oid_k = str(extract_order_id(o, default="") or "")
            side_ts_k = (extract_order_side(o), extract_timestamp_ms(o, default=None))
            cur_act, cur_mod, cur_rule = origin_by_oid.get(oid_k) or origin_by_side_ts.get(side_ts_k) or ("BOT", None, None)
            if cur_act == "BOT" and not (cur_mod or cur_rule or o.get("clientOrderId") or o.get("client_order_id")) and _matches_ai_intent(o):
                _mark_ai_origin(o)
        for i_sym, i_side, i_ts_s, i_qty, _ in parsed_ai_intents:
            if i_qty <= 0.0:
                continue
            cands: List[tuple[Dict[str, Any], float]] = []
            for o in valid:
                if extract_order_side(o) != i_side:
                    continue
                o_ts_s = (extract_timestamp_ms(o, default=0) or 0) / 1000.0
                if o_ts_s <= 0.0 or o_ts_s < i_ts_s - 5.0 or o_ts_s > i_ts_s + 86400.0:
                    continue
                o_sym = str(o.get("symbol") or o.get("pair") or fallback_sym).strip().upper()
                if i_sym and o_sym and (canonical_base_asset(i_sym) != canonical_base_asset(o_sym) or compatible_quote_suffixes(i_sym) != compatible_quote_suffixes(o_sym)):
                    continue
                oid_k = str(extract_order_id(o, default="") or "")
                side_ts_k = (extract_order_side(o), extract_timestamp_ms(o, default=None))
                cur_act, cur_mod, cur_rule = origin_by_oid.get(oid_k) or origin_by_side_ts.get(side_ts_k) or ("BOT", None, None)
                if cur_act == "BOT" and not (cur_mod or cur_rule or o.get("clientOrderId") or o.get("client_order_id")):
                    if (o_q := extract_executed_qty(o, default=0.0) or 0.0) > 0.0:
                        cands.append((o, o_q))
            if len(cands) > 1 and abs(sum(q for _, q in cands) - i_qty) <= max(1e-5, i_qty * 0.01):
                for o_cand, _ in cands:
                    _mark_ai_origin(o_cand)

    def _fill_origin(o: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not isinstance(o, dict):
            return None, None, None, None
        oid_k = str(extract_order_id(o, default="") or "")
        side_ts_k = (extract_order_side(o), extract_timestamp_ms(o, default=None))
        act, mod, rule = origin_by_oid.get(oid_k) or origin_by_side_ts.get(side_ts_k) or resolve_order_origin(o, ai_oids, audit_origin_map)
        return ("AI", None, None, "AI") if act == "AI" else ("BOT", mod, rule, format_actor_tag("BOT", mod, rule))

    chron = [o for _, o in sorted(enumerate(valid), key=lambda item: (extract_timestamp_ms(item[1], default=0) or 0, item[0]))]
    last_any = chron[-1]
    latest_fill_ts_ms = max((extract_timestamp_ms(x, default=0) or 0) for x in chron) if chron else 0
    cost_qty, active_avg = 0.0, 0.0
    tactical_qty, tactical_avg = 0.0, 0.0
    buys_fmt: list[Dict[str, Any]] = []
    sells_fmt: list[Dict[str, Any]] = []
    open_lots: list[Dict[str, Any]] = []
    lifo_realized_pnl_usd = 0.0
    lot_seq = 0
    for o in chron:
        side_o = extract_order_side(o)
        p_f, q_f, ts_raw = round_opt_px(extract_order_price(o, default=None), 6), extract_executed_qty(o, default=None), extract_timestamp_ms(o, default=None)
        act_o, mod_o, rule_o, tag_o = _fill_origin(o)
        item_d: Dict[str, Any] = {"price": p_f, "qty": q_f, "ts_ms": ts_raw, "actor": act_o}
        if oid_item := extract_order_id(o, default=""):
            item_d["order_id"] = str(oid_item)
        if mod_o:
            item_d["module"] = mod_o
        if rule_o:
            item_d["rule"] = rule_o
        if tag_o and tag_o != act_o:
            item_d["actor_tag"] = tag_o
        if side_o == "BUY":
            buys_fmt.append(item_d)
            if q_f is not None and p_f is not None and q_f > 0.0 and p_f > 0.0:
                if cost_qty <= 1e-12:
                    open_lots.clear()
                    lifo_realized_pnl_usd = 0.0
                new_cost_qty = cost_qty + q_f
                active_avg = ((cost_qty * active_avg) + (q_f * p_f)) / new_cost_qty if new_cost_qty > 0 else 0.0
                cost_qty = new_cost_qty
                new_tac_qty = tactical_qty + q_f
                tactical_avg = ((tactical_qty * tactical_avg) + (q_f * p_f)) / new_tac_qty if new_tac_qty > 0 else 0.0
                tactical_qty = new_tac_qty
                lot_seq += 1
                open_lots.append({
                    "lot_id": f"L{lot_seq}",
                    "qty": round(q_f, 6),
                    "entry_px": p_f,
                    "ts_ms": ts_raw,
                    "actor": act_o,
                })
        elif side_o == "SELL":
            sells_fmt.append(item_d)
            tactical_qty, tactical_avg = 0.0, 0.0
            if q_f is not None and q_f > 0.0:
                is_full_exit = q_f >= cost_qty * 0.98 or cost_qty - q_f <= 1e-12
                cost_qty, active_avg = (0.0, 0.0) if is_full_exit else (cost_qty - q_f, active_avg)
                in_active_swing_window = (
                    latest_fill_ts_ms <= 0
                    or not ts_raw
                    or (latest_fill_ts_ms - int(ts_raw)) <= 72 * 3600 * 1000
                )
                rem_to_close = q_f
                while rem_to_close > 1e-12 and open_lots:
                    top_lot = open_lots[-1]
                    lot_q = float(top_lot.get("qty") or 0.0)
                    lot_px = float(top_lot.get("entry_px") or 0.0)
                    matched_q = min(rem_to_close, lot_q)
                    if in_active_swing_window and p_f is not None and p_f > 0.0 and lot_px > 0.0:
                        lifo_realized_pnl_usd += matched_q * (p_f - lot_px)
                    rem_to_close -= matched_q
                    if lot_q - matched_q <= 1e-9:
                        open_lots.pop()
                    else:
                        top_lot["qty"] = round(lot_q - matched_q, 6)
                if is_full_exit:
                    open_lots.clear()

    swing_subcycle_avg_px: Optional[float] = None
    swing_subcycle_qty = 0.0
    cand_swing_lots = open_lots[-4:] if open_lots else []
    if cand_swing_lots:
        newest_lot = cand_swing_lots[-1]
        newest_px = float(newest_lot.get("entry_px") or 0.0)
        newest_ts = int(newest_lot.get("ts_ms") or 0)
        active_swing_lots = []
        for lt in reversed(cand_swing_lots):
            lt_px = float(lt.get("entry_px") or 0.0)
            lt_ts = int(lt.get("ts_ms") or 0)
            time_ok = (newest_ts <= 0 or lt_ts <= 0) or ((newest_ts - lt_ts) <= 72 * 3600 * 1000)
            px_ok = newest_px <= 0.0 or (abs(lt_px - newest_px) / newest_px <= 0.075)
            if time_ok and px_ok:
                active_swing_lots.append(lt)
            else:
                break
        if not active_swing_lots:
            active_swing_lots = [newest_lot]
        sw_cost = sum(float(x.get("qty") or 0.0) * float(x.get("entry_px") or 0.0) for x in active_swing_lots)
        sw_qty = sum(float(x.get("qty") or 0.0) for x in active_swing_lots)
        if sw_qty > 1e-12 and sw_cost > 0.0:
            swing_subcycle_avg_px = round(sw_cost / sw_qty, 6)
            swing_subcycle_qty = round(sw_qty, 6)
    elif buys_fmt:
        rec_b = [b for b in buys_fmt[-4:] if (b.get("qty") or 0.0) > 0.0 and (b.get("price") or 0.0) > 0.0]
        if rec_b:
            sw_cost = sum(float(b["qty"]) * float(b["price"]) for b in rec_b)
            sw_qty = sum(float(b["qty"]) for b in rec_b)
            if sw_qty > 1e-12:
                swing_subcycle_avg_px = round(sw_cost / sw_qty, 6)
                swing_subcycle_qty = round(sw_qty, 6)

    last_act, last_mod, last_rule, last_tag = _fill_origin(last_any)
    active_avg_r = round(active_avg, 6) if active_avg > 0 else None
    tactical_avg_r = round(tactical_avg, 6) if tactical_avg > 0 else None
    out_fills: Dict[str, Any] = {
        "last_action_side": extract_order_side(last_any) or last_any.get("side"),
        "last_action_actor": last_act,
        "active_avg_buy_px": active_avg_r,
        "swing_subcycle_avg_px": swing_subcycle_avg_px,
        "swing_subcycle_qty": swing_subcycle_qty if swing_subcycle_qty > 0 else None,
        "open_lots": open_lots[-6:],
        "lifo_realized_pnl_usd": round(lifo_realized_pnl_usd, 2) if abs(lifo_realized_pnl_usd) >= 0.01 else 0.0,
        "last_buy": dict(buys_fmt[-1]) if buys_fmt else None,
        "last_sell": dict(sells_fmt[-1]) if sells_fmt else None,
        "recent_buys": buys_fmt[-4:],
        "recent_sells": sells_fmt[-4:],
    }
    if cost_qty > 1e-12:
        out_fills["active_cycle_rem_qty"] = round(cost_qty, 6)
    if tactical_avg_r is not None and active_avg_r is not None and abs(tactical_avg_r - active_avg_r) / max(active_avg_r, 1e-12) >= 0.001:
        out_fills["tactical_avg_buy_px"] = tactical_avg_r
    elif swing_subcycle_avg_px is not None and active_avg_r is not None and abs(swing_subcycle_avg_px - active_avg_r) / max(active_avg_r, 1e-12) >= 0.001:
        out_fills["tactical_avg_buy_px"] = swing_subcycle_avg_px
    if last_mod:
        out_fills["last_action_module"] = last_mod
    if last_rule:
        out_fills["last_action_rule"] = last_rule
    if last_tag and last_tag != last_act:
        out_fills["last_action_actor_tag"] = last_tag
    return out_fills


def enrich_recent_orders_markout(
    recent_orders: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
) -> None:
    """Deprecated no-op: spot-to-last-fill delta is already exposed via `pnl_vs_last_buy_pct` / `chg_vs_last_sell_pct`."""
    return


def enrich_open_limit_orders_tca(
    open_orders: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Enrich resting open limit orders with Orderbook Wall Queue Position, Front-Run Price, and 1h Fill Probability (Layer II.3)."""
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
        if spot_px and spot_px > 0.0:
            o["dist_from_spot_pct"] = pct_delta(lim_px, spot_px)
        for _stale_tca_k in (
            "wall_queue_pos",
            "recommended_front_run_px",
            "ask_wall_ahead_px",
            "ask_wall_ahead_usd",
            "bid_wall_ahead_px",
            "bid_wall_ahead_usd",
        ):
            o.pop(_stale_tca_k, None)
        dist_pct = abs(safe_float(o.get("dist_from_spot_pct"), default=0.0) or 0.0)
        vol_1h = max(0.35, safe_pos_float((p_entry or {}).get("vol_1h_pct"), default=1.1) or 1.1)
        v_norm = normalize_venue_name(o.get("venue"))
        if v_norm == "t212":
            dec = 2
        elif v_norm == "hyperliquid":
            dec = 2 if lim_px >= 100.0 else (3 if lim_px >= 1.0 else 4)
        else:
            dec = 2 if lim_px >= 10.0 else (3 if lim_px >= 1.0 else (4 if lim_px >= 0.1 else 6))
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
                if aw_px is not None and aw_px > 0.0 and aw_usd >= 5000.0 and (not spot_px or aw_px >= spot_px * 1.0002):
                    can_front_run_ask = not spot_px or aw_px >= spot_px * 1.0010
                    step_fr = max(tick_step, min(aw_px * 0.0001, 5.0))
                    raw_fr_ask = aw_px - step_fr
                    floor_ask = max(best_ask_px or 0.0, (spot_px or 0.0) * 1.0002)
                    if floor_ask > 0.0:
                        raw_fr_ask = max(floor_ask + tick_step, raw_fr_ask)
                    fr_ask_px = round(min(aw_px, raw_fr_ask), dec)
                    if lim_px > aw_px * 1.015:
                        o["wall_queue_pos"] = "FAR_ABOVE_ASK_WALL"
                        o["ask_wall_ahead_px"] = aw_px
                        o["ask_wall_ahead_usd"] = round(aw_usd, 0)
                        queue_penalty = 0.85
                    elif lim_px > aw_px * 1.0003:
                        o["wall_queue_pos"] = "BEHIND_ASK_WALL"
                        o["ask_wall_ahead_px"] = aw_px
                        o["ask_wall_ahead_usd"] = round(aw_usd, 0)
                        if can_front_run_ask and fr_ask_px < aw_px and fr_ask_px < lim_px:
                            o["recommended_front_run_px"] = fr_ask_px
                        queue_penalty = 0.60
                    elif abs(lim_px - aw_px) / aw_px <= 0.0003:
                        o["wall_queue_pos"] = "AT_ASK_WALL"
                        if can_front_run_ask and fr_ask_px < aw_px and fr_ask_px < lim_px:
                            o["recommended_front_run_px"] = fr_ask_px
                        queue_penalty = 0.78
                    else:
                        o["wall_queue_pos"] = "FRONT_OF_ASK_WALL"
            elif side == "BUY":
                bw_px = safe_pos_float(m_entry.get("bid_wall_px"), default=None)
                bw_usd = safe_pos_float(m_entry.get("bid_wall_usd"), default=0.0) or 0.0
                if bw_px is not None and bw_px > 0.0 and bw_usd >= 5000.0 and (not spot_px or bw_px <= spot_px * 0.9998):
                    can_front_run_bid = not spot_px or bw_px <= spot_px * 0.9990
                    step_fr = max(tick_step, min(bw_px * 0.0001, 5.0))
                    raw_fr_bid = bw_px + step_fr
                    ceil_bid = min(
                        best_bid_px if best_bid_px and best_bid_px > 0.0 else bw_px * 2.0,
                        (spot_px * 0.9998) if spot_px and spot_px > 0.0 else bw_px * 2.0,
                    )
                    if ceil_bid > 0.0:
                        raw_fr_bid = min(ceil_bid - tick_step, raw_fr_bid)
                    fr_bid_px = round(max(bw_px, raw_fr_bid), dec)
                    if lim_px < bw_px * 0.985:
                        o["wall_queue_pos"] = "DEEP_PASSIVE_BID"
                        o["bid_wall_ahead_px"] = bw_px
                        o["bid_wall_ahead_usd"] = round(bw_usd, 0)
                        queue_penalty = 0.85
                    elif lim_px < bw_px * 0.9997:
                        o["wall_queue_pos"] = "BEHIND_BID_WALL"
                        o["bid_wall_ahead_px"] = bw_px
                        o["bid_wall_ahead_usd"] = round(bw_usd, 0)
                        if can_front_run_bid and fr_bid_px > bw_px and fr_bid_px > lim_px:
                            o["recommended_front_run_px"] = fr_bid_px
                        queue_penalty = 0.60
                    elif abs(lim_px - bw_px) / bw_px <= 0.0003:
                        o["wall_queue_pos"] = "AT_BID_WALL"
                        if can_front_run_bid and fr_bid_px > bw_px and fr_bid_px > lim_px:
                            o["recommended_front_run_px"] = fr_bid_px
                        queue_penalty = 0.78
                    else:
                        o["wall_queue_pos"] = "FRONT_OF_BID_WALL"
        raw_prob = 100.0 * math.exp(-0.5 * ((dist_pct / vol_1h) ** 2)) * queue_penalty
        o["fill_prob_1h_pct"] = round(min(99.0, max(1.0, raw_prob)), 1)
    return open_orders


def compute_enterprise_portfolio_risk_metrics(
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute Global Enterprise Portfolio Risk Engine metrics (Unified NAV, Cash %, BTC-Beta, 24h Correlation, 95% VaR, LIFO Alpha)."""
    if not isinstance(portfolio_summary, dict):
        return {}
    t_pos = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
    vq_cash = portfolio_summary.get("venue_quote_cash") if isinstance(portfolio_summary.get("venue_quote_cash"), dict) else {}
    rec_orders = portfolio_summary.get("recent_orders") if isinstance(portfolio_summary.get("recent_orders"), dict) else {}

    venue_pos_usd: Dict[str, float] = {}
    venue_cash_usd: Dict[str, float] = {}
    venue_free_cash_usd: Dict[str, float] = {}
    venue_locked_quote_usd: Dict[str, float] = {}
    pos_exposures: List[Tuple[str, float, float, float, float]] = []

    for p in t_pos:
        if not isinstance(p, dict):
            continue
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        val_u = (safe_pos_float(p.get("value_usd"), default=0.0) or 0.0) + (safe_pos_float(p.get("locked_base_usd"), default=0.0) or 0.0)
        venue_pos_usd[v_name] = venue_pos_usd.get(v_name, 0.0) + val_u
        if val_u >= 1.0:
            sym = str(p.get("symbol") or "").strip().upper()
            base = canonical_base_asset(sym)
            p_info = casefold_get(prices_and_trends, sym) if isinstance(prices_and_trends, dict) else None
            if not isinstance(p_info, dict) and isinstance(prices_and_trends, dict):
                p_info = next(
                    (v for c in cache_symbol_variants(sym) if isinstance((v := casefold_get(prices_and_trends, c)), dict)),
                    {},
                )
            elif not isinstance(p_info, dict):
                p_info = {}
            is_eq = sym.endswith(("_EQ", "_US_EQ")) or v_name == "t212"
            beta_v = 1.0 if base == "BTC" else safe_float(p_info.get("beta_btc_24h"), default=(0.35 if is_eq else 1.15))
            corr_v = 1.0 if base == "BTC" else safe_float(p_info.get("corr_btc_24h"), default=(0.30 if is_eq else 0.78))
            vol_1h = safe_pos_float(p_info.get("vol_1h_pct"), default=1.0) or 1.0
            daily_vol_pct = min(25.0, max(0.8, vol_1h * math.sqrt(24.0)))
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

    corr_map: Dict[str, float] = {}
    for base, _, _, corr_v, _ in pos_exposures:
        if base != "BTC":
            corr_map[base] = round(corr_v, 2)
    non_btc_corrs = [c for b, _, _, c, _ in pos_exposures if b != "BTC"]
    avg_corr_24h = round(sum(non_btc_corrs) / len(non_btc_corrs), 2) if non_btc_corrs else (1.0 if pos_exposures else 0.0)

    var_cov_sum = 0.0
    for i, (base_i, val_i, _, corr_i, dvol_i) in enumerate(pos_exposures):
        sig_i = val_i * (dvol_i / 100.0)
        for j, (base_j, val_j, _, corr_j, dvol_j) in enumerate(pos_exposures):
            sig_j = val_j * (dvol_j / 100.0)
            rho_ij = 1.0 if (i == j or (base_i and base_i == base_j)) else max(-0.5, min(0.98, corr_i * corr_j))
            var_cov_sum += sig_i * sig_j * rho_ij
    port_sigma_usd = math.sqrt(max(0.0, var_cov_sum))
    var_95_usd = round(1.645 * port_sigma_usd, 2)
    var_95_pct = round((var_95_usd / unified_nav) * 100.0, 2)

    concentration_regime = (
        "HIGH_CORR_CONCENTRATED_BETA"
        if len(pos_exposures) >= 2 and avg_corr_24h >= 0.80 and portfolio_btc_beta >= 0.65
        else ("DEFENSIVE_HIGH_CASH" if free_cash_pct >= 55.0 else "BALANCED_DIVERSIFIED")
    )

    lifo_pnl_total = sum(
        safe_float(r_info.get("lifo_realized_pnl_usd"), default=0.0) or 0.0
        for r_info in rec_orders.values()
        if isinstance(r_info, dict)
    )

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
    }
    if abs(lifo_pnl_total) >= 0.01:
        out_risk["subcycle_lifo_realized_pnl_usd"] = round(lifo_pnl_total, 2)
    return out_risk


def _sync_cycle_wallet_qty_and_avg(
    cyc: Dict[str, Any],
    *,
    qty: float,
    avg_px: Optional[float],
    spot_px: float,
    avg_dec: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Update a venue cycle dict's `qty`, `cost_usd`, `avg_cost_px`, and `dd_pct`/`pnl_vs_avg_pct` in-place."""
    if qty <= 0.0:
        cyc.update({"qty": 0.0, "cost_usd": 0.0, "avg_cost_px": None, "dd_pct": None, "pnl_vs_avg_pct": None})
        return
    avg_r = round(avg_px, avg_dec) if avg_px and avg_px > 0 else None
    dd_v = round(((spot_px - avg_r) / avg_r) * 100.0, 2) if (spot_px > 0 and avg_r and avg_r > 0) else None
    cyc.update({
        "qty": round(qty, 6),
        "cost_usd": round(qty * avg_r, 2) if avg_r else 0.0,
        "avg_cost_px": avg_r,
        "dd_pct": dd_v,
        "pnl_vs_avg_pct": dd_v,
        **(extra_fields or {}),
    })


def _populate_cycle_fallback_recent_order(
    p_venue: str,
    p_sym: str,
    cyc: Dict[str, Any],
    cyc_has_open_qty: bool,
    ro: Optional[Dict[str, Any]],
    recent_orders: Dict[str, Any],
    v_ro_map: Dict[str, Any],
    cyc_sell_ts_map: Dict[tuple[str, str], int],
) -> Optional[Dict[str, Any]]:
    """Synthesize or enrich a fallback `recent_orders` entry from `.state_*.json` cycle anchors when trade cache is absent."""
    if p_venue not in ("t212", "kraken", "hyperliquid"):
        return ro
    cyc_avg = safe_pos_float(cyc.get("avg_cost_px") or cyc.get("entry_price"), default=0.0) or 0.0
    cyc_lb = safe_pos_float(cyc.get("last_buy_price") or cyc.get("entry_price"), default=None)
    cyc_ls = safe_pos_float(cyc.get("last_sell_price"), default=None)
    cyc_ls_ts = cyc_sell_ts_map.get((p_venue, p_sym))
    created_ro = False
    if (
        not isinstance(ro, dict)
        and (p_venue == "t212" or p_sym not in recent_orders)
        and (cyc_lb is not None or cyc_ls is not None or cyc_avg > 0)
    ):
        ro = {
            "last_action_side": "BUY" if cyc_has_open_qty else ("SELL" if cyc_ls is not None else "BUY"),
            "last_action_actor": "BOT",
            "last_action_actor_tag": "BOT",
            "last_buy": None,
            "last_sell": None,
            "active_avg_buy_px": None,
            "recent_buys": [],
        }
        recent_orders[p_sym] = v_ro_map[p_sym] = ro
        created_ro = True
    if isinstance(ro, dict) and (p_venue == "t212" or created_ro):
        if cyc_avg > 0:
            ro["active_avg_buy_px"] = round(cyc_avg, 6)
        if ro.get("last_buy") is None and cyc_lb is not None:
            lb_d: Dict[str, Any] = {"price": cyc_lb, "actor": "BOT", "actor_tag": "BOT"}
            if not ro.get("recent_buys"):
                lb_d["qty"] = cyc.get("qty") if not cyc.get("dca_buys") and cyc_has_open_qty else None
            ro["last_buy"] = lb_d
        if ro.get("last_sell") is None and cyc_ls is not None and (p_venue == "t212" or cyc_ls_ts is not None or not cyc_has_open_qty):
            ro["last_sell"] = {"price": cyc_ls, "ts_ms": cyc_ls_ts, "actor": "BOT", "actor_tag": "BOT"}
    return ro


def reconcile_venue_cycles_with_recent_orders(
    kraken_cycles: Dict[str, Dict[str, Any]],
    hyperliquid_cycles: Dict[str, Dict[str, Any]],
    t212_cycles: Dict[str, Dict[str, Any]],
    venue_recent_orders: Dict[str, Dict[str, Any]],
    live_held_by_venue_sym: Dict[tuple[str, str], float],
    prices_and_trends: Dict[str, Any],
) -> Dict[tuple[str, str], int]:
    """Reconcile `.state_*.json` DCA cycles against recent fills and live wallet balances; return `(venue, sym) -> sell_ts_ms`."""
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
            ro_c = next(
                (
                    val
                    for cand in (c_sym, *cache_symbol_variants(c_sym, include_base="USD" in c_quotes, quotes=c_quotes))
                    if isinstance((val := casefold_get(v_ro_c, cand)), dict)
                ),
                None,
            )
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
                    _sync_cycle_wallet_qty_and_avg(
                        cyc,
                        qty=new_qty,
                        avg_px=new_cost / new_qty,
                        spot_px=px_c,
                        avg_dec=avg_dec,
                        extra_fields={
                            "cost_usd": round(new_cost, 2),
                            "entry_price": round(float(buys_after[0]["price"]), avg_dec),
                            "last_buy_price": round(float(buys_after[-1]["price"]), avg_dec),
                            "dca_buys": max(0, len(buys_after) - 1),
                        },
                    )
                elif has_non_dust_wallet:
                    eff_avg_w = (
                        safe_pos_float(ro_c.get("active_avg_buy_px"), default=None)
                        or safe_pos_float(cyc.get("avg_cost_px"), default=None)
                        or safe_pos_float(cyc.get("entry_price"), default=None)
                        or safe_pos_float(cyc.get("last_buy_price"), default=None)
                    )
                    _sync_cycle_wallet_qty_and_avg(cyc, qty=eff_rem_wallet, avg_px=eff_avg_w, spot_px=px_c, avg_dec=avg_dec)
                    if cyc.get("avg_cost_px") and not cyc.get("entry_price"):
                        cyc["entry_price"] = cyc["avg_cost_px"]
                else:
                    _sync_cycle_wallet_qty_and_avg(cyc, qty=0.0, avg_px=None, spot_px=px_c, avg_dec=avg_dec)
                if ls_px > 0:
                    cyc["last_sell_price"] = round(ls_px, avg_dec)
                    _sync_exit_kind(cyc, ls_px, safe_nonneg_float(cyc.get("qty"), default=0.0) or 0.0)
                if not has_non_dust_wallet and not buys_after:
                    cyc.pop("pos_peak", None)
                    cyc.pop("dd_from_peak_pct", None)
    return cyc_sell_ts_map


def enrich_positions_with_orders_and_cycles(
    tradable_positions: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    inst_by_sym: Dict[str, Any],
    venue_recent_orders: Dict[str, Dict[str, Any]],
    recent_orders: Dict[str, Any],
    kraken_cycles: Dict[str, Dict[str, Any]],
    hyperliquid_cycles: Dict[str, Dict[str, Any]],
    t212_cycles: Dict[str, Dict[str, Any]],
    cyc_sell_ts_map: Dict[tuple[str, str], int],
    *,
    gateway_max_usd: float = 2000.0,
) -> None:
    """Enrich `tradable_positions` and venue cycles in-place with cost-basis anchors, LIFO PnL, trailing peaks, and venue fee/limits."""
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
        ro = next((val for cand in p_quote_variants if isinstance((val := casefold_get(v_ro_map, cand)), dict)), None)
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
            cyc = next((val for cand in (p_sym, *p_quote_variants) if isinstance((val := casefold_get(cyc_map, cand)), dict)), None)
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
                if isinstance(ro, dict) and (sw_avg_c := safe_pos_float(ro.get("swing_subcycle_avg_px"), default=None)) is not None:
                    eff_act_avg = safe_pos_float(pos.get("active_avg_buy_px"), default=None)
                    if eff_act_avg is None or abs(sw_avg_c - eff_act_avg) / max(eff_act_avg, 1e-9) >= 0.002:
                        set_pos_anchor(pos, "swing_subcycle_avg_px", "pnl_vs_swing_avg_pct", sw_avg_c, pos_px_f)
                set_pos_anchor(pos, "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct", cyc.get("avg_cost_px"), pos_px_f, overwrite=False)
                set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", cyc.get("pos_peak"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", cyc.get("last_buy_price"), pos_px_f, overwrite=False)
            set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", cyc.get("last_sell_price"), pos_px_f, overwrite=False)
            has_ro_buys = isinstance(ro, dict) and bool(ro.get("recent_buys"))
            has_ro_sells = isinstance(ro, dict) and bool(ro.get("recent_sells"))
            if pos.get("last_buy_px") is not None and (cyc.get("last_buy_price") is None or p_venue in ("hyperliquid", "kraken") or has_ro_buys):
                cyc["last_buy_price"] = pos["last_buy_px"]
            if pos.get("last_sell_px") is not None and (cyc.get("last_sell_price") is None or p_venue in ("hyperliquid", "kraken") or has_ro_sells):
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

    _round_slope_keys = {"slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel"}
    crypto_bellwethers: Dict[str, Any] = {}
    for b_canon, (sym, pinfo) in sorted(by_base.items()):
        if sym.endswith(("_EQ", "_US_EQ")) or (px_f := safe_float(pinfo.get("price"), 0.0) or 0.0) <= 0:
            continue
        cb_entry: Dict[str, Any] = {
            "symbol": sym, "price": round(px_f, 6),
            **{
                k: (round(float(pinfo[k]), 6) if (k in _round_slope_keys and isinstance(pinfo[k], (int, float))) else pinfo[k])
                for k in raw_horizon_keys
                if pinfo.get(k) is not None
            },
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
        t_cyc = next(
            (val for cand in cache_symbol_variants(eq_sym, include_original=True, include_base=True) if isinstance((val := casefold_get(all_t212_cycles, cand)), dict)),
            {},
        )
        eq_px_f = safe_pos_float(pinfo.get("price"), default=None)
        eq_entry: Dict[str, Any] = {
            "symbol": eq_sym,
            **({"price": round(eq_px_f, 6)} if eq_px_f is not None else {}),
            **{
                k: (round(float(pinfo[k]), 6) if (k in _round_slope_keys and isinstance(pinfo[k], (int, float))) else pinfo[k])
                for k in raw_horizon_keys
                if pinfo.get(k) is not None and not (k == "kalman_vel" and abs(safe_float(pinfo.get(k), 0.0) or 0.0) < 1e-12)
            },
        }
        if t_cyc:
            eq_entry["t212_cycle"] = {k: t_cyc.get(k) for k in ("qty", "avg_cost_px", "entry_price", "last_buy_price", "dd_pct") if t_cyc.get(k) is not None}
        if len(eq_entry) > 1:
            equity_bellwethers[eq_sym] = eq_entry
    if equity_bellwethers:
        ctx["equity_bellwethers"] = equity_bellwethers

    return ctx


__all__ = [
    "_VENUE_DEFAULT_FEE_SPECS",
    "_populate_cycle_fallback_recent_order",
    "_sync_cycle_wallet_qty_and_avg",
    "build_cross_market_context",
    "compute_enterprise_portfolio_risk_metrics",
    "enrich_open_limit_orders_tca",
    "enrich_positions_with_orders_and_cycles",
    "enrich_recent_orders_markout",
    "extract_symbol_recent_fills",
    "parse_venue_cycle_state",
    "reconcile_venue_cycles_with_recent_orders",
    "resolve_symbol_price",
    "resolve_venue_fee_and_limits",
    "set_pos_anchor",
]
