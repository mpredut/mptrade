"""Stateless order-lot accounting (FIFO/LIFO), venue DCA cycle state parsing, and position cost-basis reconciliation evaluators."""
from __future__ import annotations

from abc import ABC
import math
import os
import time as _time
from typing import Any, Dict, List, Optional

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    filter_asset_class_variants,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    uses_cycle_state_files,
)
from cacheprice import Cache365PriceManager, CacheSparsePriceManager
from state_io import read_tail_jsonl_records, resolve_cachedb_filepath
from botcore import safe_int, safe_nonneg_float, safe_nonneg_int, safe_pos_int
from intelligence.macro_sentiment_ssot import (
    coerce_dict_maps,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
    _prices_diverged,
    pct_delta,
    round_opt_px,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    resolve_symbol_price,
    resolve_venue_fee_and_limits,
)
from instrument_registry import casefold_get
from providers.base import (
    MarketDataProvider,
    _ts_ms_to_s,
    aggregate_fills_by_order_id,
    extract_order_id,
    extract_order_price,
    extract_timestamp_ms,
    normalize_timestamp_ms,
)
from providers.order_ohlc_utils import (
    _EXEC_QTY_KEYS,
    _UNFILLED_ORDER_STATUSES,
    _parse_epoch_or_iso_s,
    extract_executed_qty,
    extract_order_side,
    extract_orig_qty,
    format_actor_tag,
    resolve_order_origin,
)
from providers.intent_state import extract_intent_order_type, extract_intent_side


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
    if not uses_cycle_state_files(p_venue):
        return ro
    is_eq_v = is_equity_venue(p_venue)
    cyc_avg = safe_pos_float(cyc.get("avg_cost_px") or cyc.get("entry_price"), default=0.0) or 0.0
    cyc_lb = safe_pos_float(cyc.get("last_buy_price") or cyc.get("entry_price"), default=None)
    cyc_ls = safe_pos_float(cyc.get("last_sell_price"), default=None)
    cyc_ls_ts = cyc_sell_ts_map.get((p_venue, p_sym)) or normalize_timestamp_ms(cyc.get("_last_sell_ts_ms"), default=None)
    cyc_lb_ts = cyc_sell_ts_map.get((f"{p_venue}:entry", p_sym)) or normalize_timestamp_ms(cyc.get("_entry_ts_ms"), default=None)
    created_ro = False
    existing_ro = recent_orders.get(p_sym) if isinstance(recent_orders, dict) else None
    existing_venue = normalize_venue_name(existing_ro.get("venue")) if isinstance(existing_ro, dict) else ""
    can_create_ro = (
        is_eq_v
        or p_sym not in recent_orders
        or (existing_venue and existing_venue != p_venue and f"{p_venue}:{p_sym}" not in recent_orders)
    )
    if (
        not isinstance(ro, dict)
        and can_create_ro
        and (cyc_lb is not None or cyc_ls is not None or cyc_avg > 0)
    ):
        ro = {
            "venue": p_venue,
            "last_action_side": "BUY" if cyc_has_open_qty else ("SELL" if cyc_ls is not None else "BUY"),
            "last_action_actor": "BOT",
            "last_action_actor_tag": "BOT",
            "last_buy": None,
            "last_sell": None,
            "active_avg_buy_px": None,
            "recent_buys": [],
        }
        v_ro_map[p_sym] = ro
        if p_sym not in recent_orders or is_eq_v:
            recent_orders[p_sym] = ro
        else:
            recent_orders[f"{p_venue}:{p_sym}"] = ro
        created_ro = True
    if isinstance(ro, dict):
        if created_ro:
            ro.setdefault("venue", p_venue)
        if (is_eq_v or created_ro) and cyc_avg > 0 and cyc_has_open_qty:
            if (
                is_eq_v
                and not created_ro
                and ro.get("tactical_avg_buy_px") is None
                and (prev_act_avg := safe_pos_float(ro.get("active_avg_buy_px"), default=None)) is not None
                and abs(prev_act_avg - cyc_avg) / max(cyc_avg, 1e-9) >= 0.002
            ):
                ro["tactical_avg_buy_px"] = round(prev_act_avg, 6)
            ro["active_avg_buy_px"] = round(cyc_avg, 6)
        if (is_eq_v or created_ro) and ro.get("last_buy") is None and cyc_lb is not None:
            lb_d: Dict[str, Any] = {"price": cyc_lb, "actor": "BOT", "actor_tag": "BOT"}
            if cyc_lb_ts is not None:
                lb_d["ts_ms"] = cyc_lb_ts
            if not ro.get("recent_buys"):
                lb_d["qty"] = cyc.get("qty") if not cyc.get("dca_buys") and cyc_has_open_qty else None
            ro["last_buy"] = lb_d
        elif isinstance(ro.get("last_buy"), dict) and ro["last_buy"].get("ts_ms") is None and cyc_lb_ts is not None:
            ro["last_buy"]["ts_ms"] = cyc_lb_ts
        if ro.get("last_sell") is None and cyc_ls is not None:
            ro["last_sell"] = {"price": cyc_ls, "ts_ms": cyc_ls_ts, "actor": "BOT", "actor_tag": "BOT"}
    return ro


_FLAT_RO_KEYS = (
    "active_avg_buy_px",
    "tactical_avg_buy_px",
    "swing_subcycle_avg_px",
    "swing_subcycle_qty",
    "active_cycle_rem_qty",
    "open_lots",
)
_FLAT_POS_KEYS = (
    "active_avg_buy_px", "pnl_vs_avg_pct", "tactical_avg_buy_px", "pnl_vs_tactical_avg_pct",
    "swing_subcycle_avg_px", "pnl_vs_swing_avg_pct", "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct",
    "open_lots", "lowest_buy_px", "pnl_vs_lowest_buy_pct", "trailing_peak", "dd_from_peak_pct",
)


def _clear_flat_ro_and_pos_anchors(ro: Optional[Dict[str, Any]], pos: Dict[str, Any], *, full_pos_clear: bool = False) -> None:
    if isinstance(ro, dict):
        for k_ro in _FLAT_RO_KEYS:
            ro.pop(k_ro, None)
    for k_p in (_FLAT_POS_KEYS if full_pos_clear else ("trailing_peak", "dd_from_peak_pct")):
        pos.pop(k_p, None)


def select_fresh_open_lots(
    open_lots: Any,
    *,
    held_qty: float = -1.0,
    fallback_newest_ts_ms: int = 0,
) -> tuple[List[Dict[str, Any]], int]:
    """Filter up to the 4 newest open lots within the active holding window, extending to match ``held_qty``."""
    lots_seq = open_lots if isinstance(open_lots, list) else []
    newest_lot_ts = (
        max(
            (safe_pos_int(x.get("ts_ms"), default=0) or 0 for x in lots_seq if isinstance(x, dict)),
            default=0,
        )
        or (safe_pos_int(fallback_newest_ts_ms, default=0) or 0)
    )
    tail_lots = lots_seq[-4:]
    fresh_lots = [
        ({**lot, "entry_px": lot_px} if lot.get("entry_px") is None else lot)
        for lot in tail_lots
        if isinstance(lot, dict)
        and (lot_px := safe_float(lot.get("entry_px") if lot.get("entry_px") is not None else lot.get("price"), 0.0)) > 0
        and (
            newest_lot_ts <= 0
            or not (lot_ts := safe_pos_int(lot.get("ts_ms"), default=0) or 0)
            or (newest_lot_ts - lot_ts) <= 14 * 86400 * 1000
        )
    ]
    if held_qty > 1e-8 and fresh_lots and len(fresh_lots) < len(tail_lots):
        fresh_q_chk = sum(safe_pos_float(lt.get("qty"), 0.0) or 0.0 for lt in fresh_lots)
        if fresh_q_chk < held_qty * 0.90:
            ext_rev: List[Dict[str, Any]] = []
            acc_q = 0.0
            for lot in reversed(tail_lots):
                if not isinstance(lot, dict):
                    continue
                lot_px = safe_float(lot.get("entry_px") if lot.get("entry_px") is not None else lot.get("price"), 0.0)
                lq = safe_pos_float(lot.get("qty"), 0.0) or 0.0
                lot_ts = safe_pos_int(lot.get("ts_ms"), default=0) or 0
                if lot_px <= 0.0 or lq <= 0.0:
                    continue
                if newest_lot_ts > 0 and lot_ts > 0 and (newest_lot_ts - lot_ts) > 45 * 86400 * 1000:
                    continue
                if acc_q + lq <= held_qty * 1.05 or (
                    newest_lot_ts <= 0 or not lot_ts or (newest_lot_ts - lot_ts) <= 14 * 86400 * 1000
                ):
                    ext_rev.append({**lot, "entry_px": lot_px} if lot.get("entry_px") is None else lot)
                    acc_q += lq
            if len(ext_rev) > len(fresh_lots):
                fresh_lots = list(reversed(ext_rev))
    return fresh_lots, newest_lot_ts


def reconcile_fresh_lots_vwap(
    fills_or_ro: Dict[str, Any],
    open_lots: Any,
    fresh_lots: List[Dict[str, Any]],
    held_qty: float,
) -> tuple[Optional[float], float]:
    """Recompute active VWAP on ``fills_or_ro`` when pruned ``fresh_lots`` match ``held_qty`` better than stale open lots."""
    fresh_q = sum(safe_pos_float(lt.get("qty"), 0.0) or 0.0 for lt in fresh_lots)
    lots_seq = open_lots if isinstance(open_lots, list) else []
    if held_qty > 1e-8 and fresh_lots and len(fresh_lots) < len(lots_seq):
        unfiltered_q = safe_pos_float(fills_or_ro.get("active_cycle_rem_qty"), 0.0) or sum(
            safe_pos_float(lt.get("qty"), 0.0) or 0.0 for lt in lots_seq if isinstance(lt, dict)
        )
        if fresh_q >= held_qty * 0.98 and unfiltered_q > held_qty * 1.15:
            fresh_cost = sum(
                (safe_pos_float(lt.get("qty"), 0.0) or 0.0) * (safe_pos_float(lt.get("entry_px"), 0.0) or 0.0)
                for lt in fresh_lots
            )
            if fresh_cost > 0.0:
                fresh_vwap = round(fresh_cost / fresh_q, 6)
                fills_or_ro["active_avg_buy_px"] = fresh_vwap
                fills_or_ro["active_cycle_rem_qty"] = round(fresh_q, 6)
                if (tac_cur := safe_pos_float(fills_or_ro.get("tactical_avg_buy_px"), default=None)) is not None and abs(
                    tac_cur - fresh_vwap
                ) <= max(1e-6, fresh_vwap * 1e-5):
                    fills_or_ro.pop("tactical_avg_buy_px", None)
                return fresh_vwap, fresh_q
    return None, fresh_q


def _extract_oldest_cycle_lot_ts_ms(ro: Any, fallback_ts_ms: int = 0) -> int:
    if isinstance(ro, dict):
        lot_ts = [
            t
            for x in (ro.get("open_lots") or ro.get("recent_buys") or ())
            if isinstance(x, dict) and (t := safe_pos_int(x.get("ts_ms"), default=0) or 0) > 0
        ]
        if not lot_ts and isinstance(lb := ro.get("last_buy"), dict):
            if (lb_t := safe_pos_int(lb.get("ts_ms"), default=0) or 0) > 0:
                lot_ts.append(lb_t)
        if lot_ts:
            return min(lot_ts)
    return safe_pos_int(fallback_ts_ms, default=0) or 0


def _resolve_cycle_high_since(
    symbol: str,
    quote_variants: tuple[str, ...] | list[str],
    oldest_ts_ms: int,
    prices_and_trends: Dict[str, Any],
    cache_dir: str = "",
    memo: Optional[Dict[tuple[str, int], float]] = None,
) -> float:
    """Resolve the highest price reached during an active position cycle since ``oldest_ts_ms``."""
    sym_u = str(symbol or "").strip().upper()
    if memo is not None and (sym_u, oldest_ts_ms) in memo:
        return memo[(sym_u, oldest_ts_ms)]
    cands = (sym_u, *[str(c).strip().upper() for c in (quote_variants or ()) if str(c).strip()])
    pt_entry = next(
        (
            val for cand in cands
            if isinstance((val := casefold_get(prices_and_trends, cand)), dict)
            and safe_pos_float(val.get("high_24h"), default=0.0)
        ),
        casefold_get(prices_and_trends, sym_u),
    )
    best_high = safe_pos_float(pt_entry.get("high_24h") if isinstance(pt_entry, dict) else None, default=0.0) or 0.0
    now_ms = math.trunc(_time.time() * 1000)
    if cache_dir and oldest_ts_ms > 0 and (now_ms - oldest_ts_ms) > 24 * 3600 * 1000:
        for cand in dict.fromkeys(cands):
            for mgr_cls in (Cache365PriceManager, CacheSparsePriceManager):
                arch_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, mgr_cls.disk_filename_for_symbol(cand))))
                if not os.path.exists(arch_path):
                    continue
                try:
                    for rec in read_tail_jsonl_records(arch_path, limit=None, max_bytes=512 * 1024):
                        pt = mgr_cls.extract_jsonl_series_point(rec)
                        if pt is not None and len(pt) >= 2:
                            t_f = safe_pos_float(pt[0], default=0.0) or 0.0
                            p_f = safe_pos_float(pt[1], default=0.0) or 0.0
                            t_ms = math.trunc(t_f * 1000.0 if 0.0 < t_f < 1e11 else t_f)
                            if t_ms >= oldest_ts_ms and p_f > best_high:
                                best_high = p_f
                except Exception:
                    pass
    if memo is not None:
        memo[(sym_u, oldest_ts_ms)] = best_high
    return best_high


class PortfolioEvaluatorBase(TelemetryEvaluatorInterface, ABC):
    """Abstract base contract (Level 2) for stateless portfolio, order-lot, and venue cycle evaluators."""

    cls_name = "PortfolioEvaluatorBase"


class VenueCycleStateEvaluator(PortfolioEvaluatorBase):
    """Concrete evaluator (Level 3) normalizing venue `.state_*.json` DCA cycle records."""

    cls_name = "VenueCycleStateEvaluator"

    def evaluate(
        self,
        st_data: Dict[str, Any],
        symbol: str = "",
        prices_and_trends: Optional[Dict[str, Any]] = None,
        *,
        avg_decimals: int = 6,
        fallback_prices: Optional[Dict[str, float]] = None,
        include_meta: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prices_and_trends = prices_and_trends or {}
        st_data = st_data if isinstance(st_data, dict) else {}
        qty = safe_nonneg_float(st_data.get("qty"), default=0.0) or 0.0
        raw_cost = st_data.get("cost_usd") if st_data.get("cost_usd") is not None else st_data.get("cost")
        cost = safe_nonneg_float(raw_cost, default=0.0) or 0.0
        last_buy_px = st_data.get("last_buy_price") if st_data.get("last_buy_price") is not None else st_data.get("last_open_price")
        explicit_avg = safe_pos_float(
            st_data.get("avg_cost") or st_data.get("avg_cost_px") or st_data.get("avg_price"),
            default=None,
        )
        dca_buys_i = safe_nonneg_int(st_data.get("dca_buys"), default=0)
        single_lot_px = (
            safe_pos_float(st_data.get("entry_price") or last_buy_px, default=None)
            if dca_buys_i == 0
            else None
        )
        if qty > 1e-12 and cost > 0:
            if explicit_avg is not None and abs(qty * explicit_avg - cost) <= 0.015:
                avg_cost = round(explicit_avg, avg_decimals)
            elif single_lot_px is not None and abs(qty * single_lot_px - cost) <= 0.015:
                avg_cost = round(single_lot_px, avg_decimals)
            else:
                avg_cost = round(cost / qty, avg_decimals)
        elif qty > 1e-12:
            avg_cost = round_opt_px(
                explicit_avg or st_data.get("entry_price"),
                avg_decimals,
            )
        else:
            avg_cost = None
        px_f = resolve_symbol_price(symbol, prices_and_trends, fallback_prices)
        dd_pct = pct_delta(px_f, avg_cost) if avg_cost and px_f > 0 else None
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
            if px_f > 0 and pos_peak > 0:
                eff_peak = max(pos_peak, px_f)
                cyc_entry["pos_peak"] = round_opt_px(eff_peak, avg_decimals)
                cyc_entry["dd_from_peak_pct"] = pct_delta(px_f, eff_peak)
            else:
                cyc_entry["pos_peak"] = pos_peak
        raw_realized = next(
            (st_data[rk] for rk in ("realized_net", "realized_net_usd", "realized_pnl", "realized_pnl_usd", "realized_gross") if st_data.get(rk) is not None),
            None,
        )
        if abs(realized_pnl := safe_float(raw_realized, 0.0) or 0.0) > 1e-9:
            cyc_entry["realized_pnl_usd"] = round(realized_pnl, 2)
        entry_px_f = safe_pos_float(cyc_entry.get("entry_price"), default=0.0) or 0.0
        ref_entry_px = safe_pos_float(avg_cost or entry_px_f or cyc_entry.get("last_buy_price"), default=0.0) or 0.0
        if realized_pnl < 0.0 and entry_px_f > ref_entry_px:
            ref_entry_px = entry_px_f
        last_sell_px_f = safe_pos_float(cyc_entry.get("last_sell_price"), default=0.0) or 0.0
        if (
            str(cyc_entry.get("last_exit_kind") or "").strip().upper() in ("", "TP")
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


class SymbolRecentFillsEvaluator(PortfolioEvaluatorBase):
    """Concrete evaluator (Level 3) computing FIFO/LIFO lot accounting and recent fill summaries."""

    cls_name = "SymbolRecentFillsEvaluator"

    def evaluate(
        self,
        orders_list: list,
        *,
        known_ai_order_ids: Optional[set[str]] = None,
        audit_origin_map: Optional[Dict[str, tuple[Optional[str], Optional[str]]]] = None,
        ai_intents: Optional[List[Dict[str, Any]]] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
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

        def _same_asset_scope_and_quote(s1: str, s2: str) -> bool:
            u1, u2 = str(s1 or "").strip().upper(), str(s2 or "").strip().upper()
            if not u1 or not u2 or has_equity_suffix(u1) != has_equity_suffix(u2):
                return False
            return u1 == u2 or (
                canonical_base_asset(u1) == canonical_base_asset(u2)
                and compatible_quote_suffixes(u1) == compatible_quote_suffixes(u2)
            )

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
            i_ts_s = 0.0
            for _tk in ("timestamp", "executed_ts", "executed_at", "created_at", "ts"):
                if (_parsed_ts := _parse_epoch_or_iso_s(it.get(_tk))) is not None and _parsed_ts > 0.0:
                    i_ts_s = _ts_ms_to_s(_parsed_ts)
                    break
            if i_side not in ("BUY", "SELL") or i_ts_s <= 0.0:
                continue
            i_sym = str(it.get("symbol") or exec_pay.get("symbol") or exec_pay.get("ticker") or "").strip().upper()
            hist = it.get("history") if isinstance(it.get("history"), list) else []
            last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
            exec_q = extract_executed_qty(exec_pay, default=0.0) or extract_executed_qty(it, default=0.0) or 0.0
            exec_p = extract_order_price(exec_pay, default=0.0) or extract_order_price(it, default=0.0) or safe_pos_float(it.get("executed_avg_price"), default=0.0) or 0.0
            exec_st = str(exec_pay.get("status") or it.get("exchange_status") or "").strip().upper()
            has_explicit_exec_q = (
                any(k in exec_pay or k in it for k in (*_EXEC_QTY_KEYS, "filled"))
                or str(exec_pay.get("msg") or "").strip().lower().startswith("filled ")
            )
            is_confirmed_fill = exec_st in ("FILLED", "PARTIALLY_FILLED", "CLOSED") or (
                has_explicit_exec_q and exec_st not in _UNFILLED_ORDER_STATUSES
            )
            sym_matches_target = _same_asset_scope_and_quote(i_sym, fallback_sym)
            if (
                sym_matches_target
                and is_confirmed_fill
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
                    "timestamp": round(i_ts_s * 1000.0),
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
                if i_sym and o_sym and not _same_asset_scope_and_quote(i_sym, o_sym):
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
                    if i_sym and o_sym and not _same_asset_scope_and_quote(i_sym, o_sym):
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
        cycle_buys_fmt: list[Dict[str, Any]] = []
        sells_fmt: list[Dict[str, Any]] = []
        cycle_sells_fmt: list[Dict[str, Any]] = []
        open_lots: list[Dict[str, Any]] = []
        lifo_realized_pnl_usd = 0.0
        lot_seq = 0
        last_sell_ts_ms = 0
        last_exit_oversold = False
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
                        keep_oversold_swing_pnl = (
                            last_exit_oversold
                            and (
                                last_sell_ts_ms <= 0
                                or not ts_raw
                                or (safe_int(ts_raw, 0) - last_sell_ts_ms) <= 72 * 3600 * 1000
                            )
                        )
                        if not keep_oversold_swing_pnl:
                            lifo_realized_pnl_usd = 0.0
                            cycle_buys_fmt.clear()
                        elif latest_fill_ts_ms > 0:
                            cycle_buys_fmt = [
                                b
                                for b in cycle_buys_fmt
                                if (b_ts := safe_pos_int(b.get("ts_ms"), default=0) or 0) <= 0
                                or (latest_fill_ts_ms - b_ts) <= 7 * 86400 * 1000
                            ]
                        if latest_fill_ts_ms > 0:
                            ref_buy_ts = safe_pos_int(ts_raw, default=0) or latest_fill_ts_ms
                            cycle_sells_fmt = [
                                s
                                for s in cycle_sells_fmt
                                if (s_ts := safe_pos_int(s.get("ts_ms"), default=0) or 0) <= 0
                                or (
                                    (ref_buy_ts - s_ts) <= 72 * 3600 * 1000
                                    and (latest_fill_ts_ms - s_ts) <= 7 * 86400 * 1000
                                )
                            ]
                        else:
                            cycle_sells_fmt.clear()
                        last_exit_oversold = False
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
                cycle_buys_fmt.append(item_d)
            elif side_o == "SELL":
                sells_fmt.append(item_d)
                cycle_sells_fmt.append(item_d)
                tactical_qty, tactical_avg = 0.0, 0.0
                if q_f is not None and q_f > 0.0:
                    prev_cost_qty = cost_qty
                    prev_top_lot_ts = (
                        safe_pos_int(open_lots[-1].get("ts_ms"), default=0) or 0
                        if open_lots
                        else ((safe_pos_int(buys_fmt[-1].get("ts_ms"), default=0) or 0) if buys_fmt else 0)
                    )
                    ts_sell_i = safe_pos_int(ts_raw, default=0) or 0
                    is_full_exit = q_f >= cost_qty * 0.98 or cost_qty - q_f <= 1e-12
                    last_exit_oversold = bool(
                        is_full_exit
                        and (
                            (prev_cost_qty > 1e-12 and q_f > prev_cost_qty * 1.05)
                            or (prev_cost_qty <= 1e-12 and (last_exit_oversold or bool(buys_fmt)))
                        )
                        and (
                            ts_sell_i <= 0
                            or prev_top_lot_ts <= 0
                            or (ts_sell_i - prev_top_lot_ts) <= 7 * 86400 * 1000
                        )
                    )
                    if ts_sell_i > 0:
                        last_sell_ts_ms = max(last_sell_ts_ms, ts_sell_i)
                    cost_qty, active_avg = (0.0, 0.0) if is_full_exit else (cost_qty - q_f, active_avg)
                    in_active_swing_window = (
                        latest_fill_ts_ms <= 0
                        or ts_sell_i <= 0
                        or (latest_fill_ts_ms - ts_sell_i) <= 72 * 3600 * 1000
                    )
                    rem_to_close = q_f
                    while rem_to_close > 1e-12 and open_lots:
                        top_lot = open_lots[-1]
                        lot_q = safe_pos_float(top_lot.get("qty"), default=0.0) or 0.0
                        lot_px = safe_pos_float(top_lot.get("entry_px"), default=0.0) or 0.0
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
            newest_px = safe_pos_float(newest_lot.get("entry_px"), default=0.0) or 0.0
            newest_ts = safe_pos_int(newest_lot.get("ts_ms"), default=0) or 0
            sold_after_newest = last_sell_ts_ms > 0 and newest_ts > 0 and last_sell_ts_ms > newest_ts
            top_lot_stale_after_sell = (
                sold_after_newest
                and latest_fill_ts_ms > 0
                and (latest_fill_ts_ms - newest_ts) > 72 * 3600 * 1000
            )
            active_swing_lots = []
            if not top_lot_stale_after_sell:
                for lt in reversed(cand_swing_lots):
                    lt_px = safe_pos_float(lt.get("entry_px"), default=0.0) or 0.0
                    lt_ts = safe_pos_int(lt.get("ts_ms"), default=0) or 0
                    time_ok = (newest_ts <= 0 or lt_ts <= 0) or ((newest_ts - lt_ts) <= 72 * 3600 * 1000)
                    px_ok = newest_px <= 0.0 or (abs(lt_px - newest_px) / newest_px <= 0.075)
                    if time_ok and px_ok:
                        active_swing_lots.append(lt)
                    else:
                        break
                if not active_swing_lots:
                    active_swing_lots = [newest_lot]
                if sold_after_newest and len(active_swing_lots) >= len(open_lots):
                    active_swing_lots = []
            sw_cost = sum(
                (safe_pos_float(x.get("qty"), default=0.0) or 0.0)
                * (safe_pos_float(x.get("entry_px"), default=0.0) or 0.0)
                for x in active_swing_lots
            )
            sw_qty = sum(safe_pos_float(x.get("qty"), default=0.0) or 0.0 for x in active_swing_lots)
            if sw_qty > 1e-12 and sw_cost > 0.0:
                swing_subcycle_avg_px = round(sw_cost / sw_qty, 6)
                swing_subcycle_qty = round(sw_qty, 6)
        elif buys_fmt:
            rec_b = [
                b for b in buys_fmt[-4:]
                if (safe_pos_float(b.get("qty"), default=0.0) or 0.0) > 0.0
                and (safe_pos_float(b.get("price"), default=0.0) or 0.0) > 0.0
            ]
            if rec_b:
                sw_cost = sum(
                    (safe_pos_float(b.get("qty"), default=0.0) or 0.0)
                    * (safe_pos_float(b.get("price"), default=0.0) or 0.0)
                    for b in rec_b
                )
                sw_qty = sum(safe_pos_float(b.get("qty"), default=0.0) or 0.0 for b in rec_b)
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
            "open_lots": [open_lots[0], *open_lots[-5:]] if len(open_lots) > 6 else open_lots,
            "lifo_realized_pnl_usd": round(lifo_realized_pnl_usd, 2) if abs(lifo_realized_pnl_usd) >= 0.01 else 0.0,
            "last_buy": dict(buys_fmt[-1]) if buys_fmt else None,
            "last_sell": dict(sells_fmt[-1]) if sells_fmt else None,
            "recent_buys": (cycle_buys_fmt if (cost_qty > 1e-12 and cycle_buys_fmt) else buys_fmt)[-4:],
            "recent_sells": (
                (
                    [
                        s
                        for s in cycle_sells_fmt
                        if latest_fill_ts_ms <= 0
                        or (s_ts := safe_pos_int(s.get("ts_ms"), default=0) or 0) <= 0
                        or (latest_fill_ts_ms - s_ts) <= 7 * 86400 * 1000
                    ]
                    or cycle_sells_fmt
                )
                if (cost_qty > 1e-12 and cycle_sells_fmt)
                else sells_fmt
            )[-4:],
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


class VenueCycleReconciliationEvaluator(PortfolioEvaluatorBase):
    """Concrete evaluator (Level 3) reconciling venue DCA cycles against recent fills and live wallet balances."""

    cls_name = "VenueCycleReconciliationEvaluator"

    def evaluate(
        self,
        kraken_cycles: Dict[str, Dict[str, Any]],
        hyperliquid_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        t212_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        venue_recent_orders: Optional[Dict[str, Dict[str, Any]]] = None,
        live_held_by_venue_sym: Optional[Dict[tuple[str, str], float]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[tuple[str, str], int]:
        hyperliquid_cycles = hyperliquid_cycles or {}
        t212_cycles = t212_cycles or {}
        venue_recent_orders = venue_recent_orders or {}
        live_held_by_venue_sym = live_held_by_venue_sym or {}
        prices_and_trends = prices_and_trends or {}
        cyc_sell_ts_map: Dict[tuple[str, str], int] = {}
        for c_venue, cyc_map, avg_dec in (("kraken", kraken_cycles, 6), ("hyperliquid", hyperliquid_cycles, 6), ("t212", t212_cycles, 4)):
            if not isinstance(cyc_map, dict):
                continue
            v_ro_c = venue_recent_orders.get(c_venue, {})
            for c_sym, cyc in cyc_map.items():
                if not isinstance(cyc, dict):
                    continue
                entry_ts_ms, sell_ts_ms = cyc.pop("_entry_ts_ms", None), cyc.pop("_last_sell_ts_ms", None)
                c_quotes = compatible_quote_suffixes(c_sym)
                c_is_eq = is_equity_venue_or_suffix(c_sym, c_venue)
                c_cands = filter_asset_class_variants(
                    c_sym,
                    (c_sym, *cache_symbol_variants(c_sym, include_base="USD" in c_quotes, quotes=c_quotes)),
                    is_equity=c_is_eq,
                )
                if (entry_ts_i := safe_pos_int(entry_ts_ms, default=0) or 0) > 0:
                    for cand in c_cands:
                        cyc_sell_ts_map.setdefault((f"{c_venue}:entry", str(cand).upper()), entry_ts_i)
                    cyc_sell_ts_map[(f"{c_venue}:entry", str(c_sym).upper())] = entry_ts_i
                if (sell_ts_i := safe_pos_int(sell_ts_ms, default=0) or 0) > 0:
                    for cand in c_cands:
                        cyc_sell_ts_map.setdefault((c_venue, str(cand).upper()), sell_ts_i)
                    cyc_sell_ts_map[(c_venue, str(c_sym).upper())] = sell_ts_i
                ro_c = next(
                    (
                        val
                        for cand in c_cands
                        if isinstance((val := casefold_get(v_ro_c, cand)), dict)
                    ),
                    None,
                )
                cyc_qty_cur = safe_nonneg_float(cyc.get("qty"), default=0.0) or 0.0
                if isinstance(ro_c, dict) and is_equity_venue(c_venue) and cyc_qty_cur > 1e-8:
                    lb_ro, ls_ro = coerce_dict_maps(ro_c.get("last_buy"), ro_c.get("last_sell"))
                    if ro_c.get("recent_buys") and (lb_px_ro := safe_pos_float(lb_ro.get("price"), default=None)) is not None:
                        cyc["last_buy_price"] = round(lb_px_ro, avg_dec)
                    if ro_c.get("recent_sells") and (ls_px_ro := safe_pos_float(ls_ro.get("price"), default=None)) is not None:
                        cyc["last_sell_price"] = round(ls_px_ro, avg_dec)
                if not isinstance(ro_c, dict):
                    continue
                ls_c = ro_c.get("last_sell") if isinstance(ro_c.get("last_sell"), dict) else {}
                ls_ts_ms = safe_pos_float(ls_c.get("ts_ms"), default=0.0) or 0.0
                ls_qty = safe_pos_float(ls_c.get("qty"), default=0.0) or 0.0
                ls_px = safe_pos_float(ls_c.get("price"), default=0.0) or 0.0
                r_buys_c = ro_c.get("recent_buys") if isinstance(ro_c.get("recent_buys"), list) else []
                entry_ts_f = safe_pos_float(entry_ts_ms, default=None)
                sell_ts_f = safe_pos_float(sell_ts_ms, default=None)
                buys_after = [
                    b for b in r_buys_c
                    if isinstance(b, dict)
                    and (safe_pos_float(b.get("ts_ms"), default=0.0) or 0.0) > ls_ts_ms
                    and (safe_pos_float(b.get("qty"), default=0.0) or 0.0) > 0.0
                    and (safe_pos_float(b.get("price"), default=0.0) or 0.0) > 0.0
                ]
                buys_after_qty = sum(safe_pos_float(b.get("qty"), default=0.0) or 0.0 for b in buys_after)
                post_sell_new_cycle = bool(
                    entry_ts_f is not None
                    and ls_ts_ms > 0.0
                    and entry_ts_f > ls_ts_ms + 5000.0
                    and buys_after
                    and cyc_qty_cur <= buys_after_qty * 1.05
                )
                buys_before = (
                    []
                    if post_sell_new_cycle
                    else [
                        b for b in r_buys_c
                        if isinstance(b, dict)
                        and 0.0 < (safe_pos_float(b.get("ts_ms"), default=0.0) or 0.0) <= ls_ts_ms
                        and (safe_pos_float(b.get("price"), default=0.0) or 0.0) > 0.0
                    ]
                )
                reconciled_external_buy = bool(
                    entry_ts_f is not None
                    and not cyc.get("resting_orders")
                    and buys_after
                    and cyc_qty_cur > 1e-8
                    and entry_ts_f > (safe_pos_float(buys_after[-1].get("ts_ms"), default=0.0) or 0.0) + 5000.0
                    and abs(buys_after_qty - cyc_qty_cur) <= max(1e-4, cyc_qty_cur * 0.02)
                )
                rem_q_ro = safe_pos_float(ro_c.get("active_cycle_rem_qty"), default=0.0) or 0.0
                live_held = live_held_by_venue_sym.get((c_venue, str(c_sym).upper()))
                if post_sell_new_cycle and (
                    live_held is None
                    or live_held <= 1e-6
                    or abs(buys_after_qty - live_held) <= max(1e-4, live_held * 0.02)
                ):
                    ro_c["lifo_realized_pnl_usd"] = 0.0
                    ro_c["recent_buys"] = buys_after[-4:]
                    if isinstance(ro_c.get("open_lots"), list):
                        ro_c["open_lots"] = [
                            lt for lt in ro_c["open_lots"]
                            if isinstance(lt, dict) and (safe_pos_float(lt.get("ts_ms"), default=0.0) or 0.0) > ls_ts_ms
                        ]
                has_pre_sell_rem = (
                    buys_after_qty > 0.0
                    and (cyc_qty_cur - buys_after_qty) > 1e-6
                    and rem_q_ro > 1e-8
                    and abs(rem_q_ro - cyc_qty_cur) <= max(1e-4, max(rem_q_ro, cyc_qty_cur) * 0.05)
                )
                already_deducted_partial = bool(
                    not reconciled_external_buy
                    and (not buys_after or ls_qty < cyc_qty_cur * 0.95 or has_pre_sell_rem)
                    and cyc_qty_cur > 1e-8
                    and (
                        (sell_ts_f is not None and sell_ts_f >= ls_ts_ms - 5000.0 and (live_held is None or live_held > 1e-6))
                        or (live_held is None and rem_q_ro > 1e-8 and abs(rem_q_ro - cyc_qty_cur) <= max(1e-4, max(rem_q_ro, cyc_qty_cur) * 0.05))
                        or (live_held is not None and live_held > 1e-8 and abs(live_held - cyc_qty_cur) <= max(1e-4, max(live_held, cyc_qty_cur) * 0.05))
                    )
                )

                def _sync_exit_kind(c_dict: Dict[str, Any], sell_p: float, rem_q: float) -> None:
                    if sell_p <= 0.0 or (post_sell_new_cycle and str(c_dict.get("last_exit_kind") or "").strip()):
                        return
                    pre_sell_ref = (
                        (safe_pos_float(buys_before[-1].get("price"), default=None) if buys_before else None)
                        or safe_pos_float(c_dict.get("entry_price"), default=None)
                    ) if buys_after else None
                    ref_avg = (
                        pre_sell_ref
                        or safe_pos_float(ro_c.get("active_avg_buy_px"), default=None)
                        or safe_pos_float(c_dict.get("avg_cost_px"), default=None)
                        or safe_pos_float(c_dict.get("entry_price"), default=None)
                        or safe_pos_float(c_dict.get("last_buy_price"), default=None)
                    )
                    if ref_avg is not None and ref_avg > 0.0:
                        cur_kind = str(c_dict.get("last_exit_kind") or "").strip().upper()
                        is_newer_partial_sell = (
                            not post_sell_new_cycle
                            and rem_q > 1e-6
                            and ls_ts_ms > 0.0
                            and (
                                sell_ts_f is None
                                or sell_ts_f < ls_ts_ms - 5000.0
                                or (already_deducted_partial and sell_ts_f <= ls_ts_ms + 60000.0)
                            )
                        )
                        replaceable_kinds = ("", "TP", "STOP") if is_newer_partial_sell else ("", "TP")
                        lifo_pnl_ro = safe_float(ro_c.get("lifo_realized_pnl_usd"), default=0.0) or 0.0
                        is_loss_or_scratch_sell = sell_p < ref_avg * 0.995 or (
                            sell_p < ref_avg - 1e-6 and (lifo_pnl_ro < 0.0 or (is_newer_partial_sell and lifo_pnl_ro <= 0.0))
                        )
                        if is_loss_or_scratch_sell and cur_kind in replaceable_kinds:
                            c_dict["last_exit_kind"] = "REDUCE"
                        elif sell_p >= ref_avg * 1.001 and (not cur_kind or (is_newer_partial_sell and cur_kind == "STOP")):
                            c_dict["last_exit_kind"] = "TP"

                if already_deducted_partial:
                    if ls_px > 0:
                        cyc["last_sell_price"] = round(ls_px, avg_dec)
                        _sync_exit_kind(cyc, ls_px, cyc_qty_cur)
                    if buys_after and buys_before:
                        cyc["dca_buys"] = max(safe_nonneg_int(cyc.get("dca_buys"), default=0), len(buys_after))
                    elif len(buys_after) >= 2 and not cyc.get("dca_buys"):
                        cyc["dca_buys"] = len(buys_after) - 1
                    continue
                if entry_ts_f is None or cyc.get("resting_orders"):
                    continue
                if (
                    not buys_after
                    and ls_ts_ms > entry_ts_f
                    and cyc_qty_cur > 1e-8
                    and 0.0 < ls_qty < cyc_qty_cur - 1e-8
                    and (sell_ts_f is None or sell_ts_f < ls_ts_ms - 5000.0)
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
                elif (ls_ts_ms > entry_ts_f and cyc_qty_cur > 1e-8 and ls_qty >= cyc_qty_cur * 0.95) or reconciled_external_buy:
                    px_c = resolve_symbol_price(
                        ensure_equity_suffix(c_sym, is_equity=c_is_eq),
                        prices_and_trends,
                    )
                    eff_rem_wallet = (
                        live_held
                        if (live_held is not None and live_held > 1e-6)
                        else (rem_q_ro if (live_held is None and rem_q_ro > 1e-6) else 0.0)
                    )
                    has_non_dust_wallet = eff_rem_wallet > 1e-6 and (px_c <= 0.0 or eff_rem_wallet * px_c >= 5.0)
                    if buys_after:
                        new_qty = buys_after_qty
                        new_cost = sum(
                            (safe_pos_float(b.get("qty"), default=0.0) or 0.0)
                            * (safe_pos_float(b.get("price"), default=0.0) or 0.0)
                            for b in buys_after
                        )
                        first_buy_px = safe_pos_float(buys_after[0].get("price"), default=0.0) or 0.0
                        last_buy_px = safe_pos_float(buys_after[-1].get("price"), default=0.0) or 0.0
                        _sync_cycle_wallet_qty_and_avg(
                            cyc,
                            qty=new_qty,
                            avg_px=new_cost / new_qty,
                            spot_px=px_c,
                            avg_dec=avg_dec,
                            extra_fields={
                                "cost_usd": round(new_cost, 2),
                                "entry_price": round(first_buy_px, avg_dec),
                                "last_buy_price": round(last_buy_px, avg_dec),
                                "dca_buys": max(0, len(buys_after) - 1),
                            },
                        )
                        if eff_rem_wallet <= 1e-6 or abs(new_qty - eff_rem_wallet) <= max(1e-4, eff_rem_wallet * 0.02):
                            ro_c["lifo_realized_pnl_usd"] = 0.0
                            ro_c["recent_buys"] = buys_after[-4:]
                            if isinstance(ro_c.get("open_lots"), list):
                                ro_c["open_lots"] = [
                                    lt for lt in ro_c["open_lots"]
                                    if isinstance(lt, dict) and (safe_pos_float(lt.get("ts_ms"), default=0.0) or 0.0) > ls_ts_ms
                                ]
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
                elif (
                    buys_after
                    and cyc_qty_cur > 1e-8
                    and ls_ts_ms > 0.0
                    and (sell_ts_ms is None or safe_float(sell_ts_ms, 0.0) < ls_ts_ms - 5000.0)
                ):
                    if ls_px > 0:
                        cyc["last_sell_price"] = round(ls_px, avg_dec)
                        _sync_exit_kind(cyc, ls_px, max(cyc_qty_cur, live_held or 0.0, rem_q_ro))
                    if buys_before:
                        cyc["dca_buys"] = max(safe_nonneg_int(cyc.get("dca_buys"), default=0), len(buys_after))
                    elif len(buys_after) >= 2 and not cyc.get("dca_buys"):
                        cyc["dca_buys"] = len(buys_after) - 1
        return cyc_sell_ts_map


class PositionCostBasisEnrichmentEvaluator(PortfolioEvaluatorBase):
    """Concrete evaluator (Level 3) enriching tradable positions and venue cycles with cost basis, LIFO PnL, and trailing peaks."""

    cls_name = "PositionCostBasisEnrichmentEvaluator"

    def evaluate(
        self,
        tradable_positions: List[Dict[str, Any]],
        prices_and_trends: Optional[Dict[str, Any]] = None,
        inst_by_sym: Optional[Dict[str, Any]] = None,
        venue_recent_orders: Optional[Dict[str, Dict[str, Any]]] = None,
        recent_orders: Optional[Dict[str, Any]] = None,
        kraken_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        hyperliquid_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        t212_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        cyc_sell_ts_map: Optional[Dict[tuple[str, str], int]] = None,
        *,
        gateway_max_usd: float = 2000.0,
        cache_dir: str = "",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        prices_and_trends = prices_and_trends or {}
        inst_by_sym = inst_by_sym or {}
        venue_recent_orders, recent_orders, kraken_cycles, hyperliquid_cycles, t212_cycles, cyc_sell_ts_map = coerce_dict_maps(
            venue_recent_orders, recent_orders, kraken_cycles, hyperliquid_cycles, t212_cycles, cyc_sell_ts_map
        )
        seen_venue_syms: set[tuple[str, str]] = set()
        _cycle_high_memo: Dict[tuple[str, int], float] = {}
        for pos in tradable_positions:
            if not isinstance(pos, dict):
                continue
            p_sym = str(pos.get("symbol") or "").upper()
            p_venue = normalize_venue_name(pos.get("venue"))
            p_is_eq = is_equity_venue_or_suffix(p_sym, p_venue)
            p_quotes = compatible_quote_suffixes(p_sym, pos.get("quote"))
            p_quote_variants = filter_asset_class_variants(
                p_sym,
                cache_symbol_variants(p_sym, include_base="USD" in p_quotes, quotes=p_quotes),
                is_equity=p_is_eq,
            )
            if p_venue and p_sym:
                for cand_vs in (p_sym, *p_quote_variants):
                    seen_venue_syms.add((p_venue, str(cand_vs).upper()))
            p_variants = (*p_quote_variants, *filter_asset_class_variants(p_sym, cache_symbol_variants(p_sym), is_equity=p_is_eq))
            pos_px_f = safe_float(pos.get("price"), 0.0)
            p_lookup_sym = ensure_equity_suffix(p_sym, is_equity=p_is_eq)
            cand_px = resolve_symbol_price(p_lookup_sym, prices_and_trends, explicit_quote=pos.get("quote"))
            if cand_px > 0 and (pos_px_f <= 0 or (abs(pos_px_f - cand_px) > 1e-9 and not _prices_diverged(pos_px_f, cand_px, threshold=0.35 if p_is_eq else 0.10))):
                pos_px_f = round(cand_px, MarketDataProvider.resolve_venue_position_price_decimals(p_venue))
                pos["price"] = pos_px_f
                if (fb_val := safe_float(pos.get("free_base"), -1.0)) >= 0:
                    pos["value_usd"] = round(fb_val * cand_px, 2)
                if (lb_val := safe_pos_float(pos.get("locked_base_qty"), default=0.0) or 0.0) > 0:
                    pos["locked_base_usd"] = round(lb_val * cand_px, 2)
                if (pk_val := safe_float(pos.get("trailing_peak"), 0.0)) > 0:
                    if pk_val < cand_px:
                        pk_val = cand_px
                        pos["trailing_peak"] = round(pk_val, 6)
                    pos["dd_from_peak_pct"] = pct_delta(cand_px, pk_val)
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
                    _clear_flat_ro_and_pos_anchors(ro, pos, full_pos_clear=True)
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
                lb_px = safe_pos_float(lb.get("price") if isinstance(lb, dict) else None, default=None)
                set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", lb_px, pos_px_f)
                set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", ls.get("price") if isinstance(ls, dict) else None, pos_px_f)
                r_buys = ro.get("recent_buys") if isinstance(ro.get("recent_buys"), list) else []
                open_lots = ro.get("open_lots") if isinstance(ro.get("open_lots"), list) else []
                fresh_lots, newest_lot_ts = select_fresh_open_lots(
                    open_lots,
                    held_qty=held_cur,
                    fallback_newest_ts_ms=(safe_pos_int(lb.get("ts_ms"), default=0) or 0) if isinstance(lb, dict) else 0,
                )
                if not is_flat_after_exit and "open_lots" in pos:
                    if fresh_lots:
                        pos["open_lots"] = fresh_lots
                        fresh_vwap, _ = reconcile_fresh_lots_vwap(ro, open_lots, fresh_lots, held_cur)
                        if fresh_vwap is not None:
                            ro["open_lots"] = fresh_lots
                            set_pos_anchor(pos, "active_avg_buy_px", "pnl_vs_avg_pct", fresh_vwap, pos_px_f, overwrite=True)
                            if "tactical_avg_buy_px" not in ro:
                                pos.pop("tactical_avg_buy_px", None)
                                pos.pop("pnl_vs_tactical_avg_pct", None)
                            if (sw_cur := safe_pos_float(ro.get("swing_subcycle_avg_px"), default=None)) is not None and abs(
                                sw_cur - fresh_vwap
                            ) / max(fresh_vwap, 1e-9) < 0.002:
                                pos.pop("swing_subcycle_avg_px", None)
                                pos.pop("pnl_vs_swing_avg_pct", None)
                    else:
                        pos.pop("open_lots", None)
                low_buy_px = (
                    min((safe_float(lot.get("entry_px"), 0.0) for lot in fresh_lots), default=0.0)
                    if fresh_lots
                    else min((safe_float(b.get("price"), 0.0) for b in r_buys if isinstance(b, dict) and safe_float(b.get("price"), 0.0) > 0), default=0.0)
                )
                if lb_px is not None and lb_px > 0:
                    low_buy_px = min(low_buy_px, lb_px) if low_buy_px > 0 else lb_px
                if low_buy_px > 0 and not is_flat_after_exit:
                    set_pos_anchor(pos, "lowest_buy_px", "pnl_vs_lowest_buy_pct", low_buy_px, pos_px_f)
                if held_cur > 1e-8 and not is_flat_balance and not is_flat_after_exit and pos_px_f > 0.0:
                    open_lot_ts_vals = [
                        t for x in open_lots
                        if isinstance(x, dict) and (t := safe_pos_int(x.get("ts_ms"), default=0) or 0) > 0
                    ]
                    oldest_pos_ts = min(open_lot_ts_vals) if open_lot_ts_vals else newest_lot_ts
                    pos_age_ms = (math.trunc(_time.time() * 1000) - oldest_pos_ts) if oldest_pos_ts > 0 else 0
                    cur_pk = safe_pos_float(pos.get("trailing_peak"), default=None)
                    act_avg_pk = safe_pos_float(pos.get("active_avg_buy_px"), default=None)
                    pk_is_missing_or_unwarmed = cur_pk is None or (
                        act_avg_pk is not None and abs(cur_pk - act_avg_pk) <= max(1e-6, max(cur_pk, act_avg_pk) * 1e-5)
                    )
                    if pk_is_missing_or_unwarmed and 12 * 3600 * 1000 <= pos_age_ms <= 90 * 86400 * 1000:
                        hi_24h = _resolve_cycle_high_since(
                            p_sym, p_quote_variants, oldest_pos_ts, prices_and_trends, cache_dir=cache_dir, memo=_cycle_high_memo
                        )
                        ls_ts_ms = (safe_pos_int(ls.get("ts_ms"), default=0) or 0) if isinstance(ls, dict) else 0
                        ls_px_pk = (safe_pos_float(pos.get("last_sell_px"), default=0.0) or 0.0) if ls_ts_ms >= oldest_pos_ts else 0.0
                        ref_pk_anchor = max(pos_px_f, act_avg_pk or 0.0, ls_px_pk)
                        if hi_24h > 0.0 and not _prices_diverged(ref_pk_anchor, hi_24h, threshold=0.35):
                            seed_pk = max(hi_24h, ls_px_pk, pos_px_f, cur_pk or 0.0, act_avg_pk or 0.0)
                            set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", seed_pk, pos_px_f, overwrite=True)
            for cyc_map in {"hyperliquid": (hyperliquid_cycles,), "t212": (t212_cycles,), "kraken": (kraken_cycles,)}.get(p_venue, ()):
                cyc = next((val for cand in (p_sym, *p_quote_variants) if isinstance((val := casefold_get(cyc_map, cand)), dict)), None)
                if not isinstance(cyc, dict):
                    continue
                cyc_qty = safe_nonneg_float(cyc.get("qty"), default=None)
                c_pk_orig = safe_pos_float(cyc.get("pos_peak"), default=0.0) or 0.0
                p_pk_orig = safe_pos_float(pos.get("trailing_peak"), default=0.0) or 0.0
                raw_pk_orig = max(c_pk_orig, p_pk_orig) if max(c_pk_orig, p_pk_orig) > 0.0 else None
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
                        if is_equity_venue(p_venue)
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
                            cyc["pnl_vs_avg_pct"] = pct_delta(pos_px_f, eff_avg_cost)
                            cyc["dd_pct"] = cyc["pnl_vs_avg_pct"]
                    c_pk_val = safe_pos_float(cyc.get("pos_peak"), default=0.0) or 0.0
                    p_pk_val = safe_pos_float(pos.get("trailing_peak"), default=0.0) or 0.0
                    if pos_px_f > 0.0 and max(c_pk_val, p_pk_val) > 0.0:
                        eff_peak = max(c_pk_val, p_pk_val, pos_px_f)
                        cyc["pos_peak"] = round(eff_peak, 6)
                        cyc["dd_from_peak_pct"] = pct_delta(pos_px_f, eff_peak)
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
                    cyc.update({"qty": round(max(0.0, fb_cur), 6), "cost_usd": 0.0, "avg_cost_px": None, "dd_pct": None, "pnl_vs_avg_pct": None})
                    cyc.pop("pos_peak", None)
                    cyc.pop("dd_from_peak_pct", None)
                    _clear_flat_ro_and_pos_anchors(ro, pos, full_pos_clear=True)
                    if is_flat_after_exit and pos.get("last_sell_px") is not None:
                        cyc["last_sell_price"] = pos["last_sell_px"]
                elif cyc_flat:
                    pos.pop("trailing_peak", None)
                    pos.pop("dd_from_peak_pct", None)
                    cyc.pop("pos_peak", None)
                    cyc.pop("dd_from_peak_pct", None)
                elif not is_flat_after_exit and cyc_has_open_qty:
                    cyc_avg = cyc.get("avg_cost_px") or cyc.get("entry_price") or cyc.get("last_buy_price")
                    set_pos_anchor(
                        pos, "active_avg_buy_px", "pnl_vs_avg_pct", cyc_avg, pos_px_f,
                        overwrite=(is_equity_venue(p_venue) and cyc.get("avg_cost_px") is not None),
                    )
                    ro = _populate_cycle_fallback_recent_order(
                        p_venue, p_sym, cyc, cyc_has_open_qty, ro, recent_orders, v_ro_map, cyc_sell_ts_map
                    )
                    lp_ro = safe_float(
                        pos.get("lifo_realized_pnl_usd")
                        if pos.get("lifo_realized_pnl_usd") is not None
                        else (ro.get("lifo_realized_pnl_usd") if isinstance(ro, dict) else None),
                        default=None,
                    )
                    c_rpnl = safe_float(cyc.get("realized_pnl_usd"), default=None)
                    if lp_ro is not None and c_rpnl is not None and c_rpnl < 0.0 and lp_ro > c_rpnl:
                        cyc["realized_pnl_usd"] = round(lp_ro, 2)
                    if isinstance(ro, dict) and (tac_avg_c := safe_pos_float(ro.get("tactical_avg_buy_px"), default=None)) is not None:
                        set_pos_anchor(pos, "tactical_avg_buy_px", "pnl_vs_tactical_avg_pct", tac_avg_c, pos_px_f, overwrite=False)
                    if isinstance(ro, dict) and (sw_avg_c := safe_pos_float(ro.get("swing_subcycle_avg_px"), default=None)) is not None:
                        eff_act_avg = safe_pos_float(pos.get("active_avg_buy_px"), default=None)
                        if eff_act_avg is None or abs(sw_avg_c - eff_act_avg) / max(eff_act_avg, 1e-9) >= 0.002:
                            set_pos_anchor(pos, "swing_subcycle_avg_px", "pnl_vs_swing_avg_pct", sw_avg_c, pos_px_f)
                    set_pos_anchor(pos, "cycle_avg_cost_px", "pnl_vs_cycle_cost_pct", cyc.get("avg_cost_px"), pos_px_f, overwrite=False)
                    c_pk_sync = safe_pos_float(cyc.get("pos_peak"), default=0.0) or 0.0
                    p_pk_sync = safe_pos_float(pos.get("trailing_peak"), default=0.0) or 0.0
                    if c_pk_sync > 0.0:
                        set_pos_anchor(
                            pos, "trailing_peak", "dd_from_peak_pct", max(c_pk_sync, p_pk_sync), pos_px_f, overwrite=(c_pk_sync > p_pk_sync)
                        )
                    if pos_px_f > 0.0:
                        cur_pk_c = safe_pos_float(pos.get("trailing_peak"), default=None)
                        act_avg_c = safe_pos_float(pos.get("active_avg_buy_px") or cyc.get("avg_cost_px") or cyc.get("entry_price"), default=None)
                        c_entry_c = safe_pos_float(cyc.get("entry_price"), default=None)
                        if (
                            raw_pk_orig is None
                            or (act_avg_c is not None and abs(raw_pk_orig - act_avg_c) <= max(1e-6, max(raw_pk_orig, act_avg_c) * 1e-5))
                            or (c_entry_c is not None and abs(raw_pk_orig - c_entry_c) <= max(1e-6, max(raw_pk_orig, c_entry_c) * 1e-5))
                        ):
                            oldest_c_ts_p = _extract_oldest_cycle_lot_ts_ms(
                                ro, fallback_ts_ms=cyc_sell_ts_map.get((f"{p_venue}:entry", p_sym)) or 0
                            )
                            c_age_ms_p = (math.trunc(_time.time() * 1000) - oldest_c_ts_p) if oldest_c_ts_p > 0 else 0
                            if 12 * 3600 * 1000 <= c_age_ms_p <= 90 * 86400 * 1000:
                                hi_24h_p = _resolve_cycle_high_since(
                                    p_sym, p_quote_variants, oldest_c_ts_p, prices_and_trends, cache_dir=cache_dir, memo=_cycle_high_memo
                                )
                                ref_pk_anchor_p = max(pos_px_f, act_avg_c or 0.0, c_entry_c or 0.0, c_pk_sync)
                                if hi_24h_p > 0.0 and not _prices_diverged(ref_pk_anchor_p, hi_24h_p, threshold=0.35):
                                    seed_pk_p = max(hi_24h_p, pos_px_f, cur_pk_c or 0.0, act_avg_c or 0.0, c_pk_sync)
                                    set_pos_anchor(pos, "trailing_peak", "dd_from_peak_pct", seed_pk_p, pos_px_f, overwrite=True)
                                    cyc["pos_peak"] = round(seed_pk_p, 6)
                                    cyc["dd_from_peak_pct"] = pct_delta(pos_px_f, seed_pk_p)
                set_pos_anchor(pos, "last_buy_px", "pnl_vs_last_buy_pct", cyc.get("last_buy_price"), pos_px_f, overwrite=False)
                set_pos_anchor(pos, "last_sell_px", "chg_vs_last_sell_pct", cyc.get("last_sell_price"), pos_px_f, overwrite=False)
                has_ro_buys = isinstance(ro, dict) and bool(ro.get("recent_buys"))
                has_ro_sells = isinstance(ro, dict) and bool(ro.get("recent_sells"))
                is_crypto_cyc_venue = uses_cycle_state_files(p_venue) and not is_equity_venue(p_venue)
                if pos.get("last_buy_px") is not None and (cyc.get("last_buy_price") is None or is_crypto_cyc_venue or has_ro_buys):
                    cyc["last_buy_price"] = pos["last_buy_px"]
                if pos.get("last_sell_px") is not None and (cyc.get("last_sell_price") is None or is_crypto_cyc_venue or has_ro_sells):
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
            prev_inst_limits = (
                {
                    k: pos[k]
                    for k in (
                        "fee_tier",
                        "maker_fee_bps",
                        "taker_fee_bps",
                        "maker_rebate_edge_bps",
                        "exchange_min_notional_usd",
                        "min_qty",
                        "step_size",
                        "tick_size",
                    )
                    if pos.get(k) is not None
                }
                if inst_obj is None
                else {}
            )
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
            if prev_inst_limits:
                pos.update(prev_inst_limits)
        for c_venue, cyc_map in (("t212", t212_cycles), ("kraken", kraken_cycles), ("hyperliquid", hyperliquid_cycles)):
            if not isinstance(cyc_map, dict):
                continue
            v_ro_map = venue_recent_orders.setdefault(c_venue, {}) if isinstance(venue_recent_orders, dict) else {}
            for c_sym, cyc in cyc_map.items():
                if not isinstance(cyc, dict):
                    continue
                c_sym_u = str(c_sym or "").strip().upper()
                if not c_sym_u or (c_venue, c_sym_u) in seen_venue_syms:
                    continue
                cyc_qty = safe_nonneg_float(cyc.get("qty"), default=0.0) or 0.0
                if cyc_qty <= 1e-8:
                    continue
                c_quotes = compatible_quote_suffixes(c_sym_u)
                c_quote_variants = filter_asset_class_variants(
                    c_sym_u,
                    cache_symbol_variants(c_sym_u, include_base="USD" in c_quotes, quotes=c_quotes),
                    is_equity=is_equity_venue_or_suffix(c_sym_u, c_venue),
                )
                ro = next((val for cand in (c_sym_u, *c_quote_variants) if isinstance((val := casefold_get(v_ro_map, cand)), dict)), None)
                if ro is None and isinstance(recent_orders, dict):
                    ro = next(
                        (
                            val
                            for cand in (c_sym_u, *c_quote_variants)
                            if isinstance((val := casefold_get(recent_orders, cand)), dict)
                            and normalize_venue_name(val.get("venue") or c_venue) == c_venue
                        ),
                        None,
                    )
                ro = _populate_cycle_fallback_recent_order(
                    c_venue, c_sym_u, cyc, True, ro, recent_orders, v_ro_map, cyc_sell_ts_map
                )
                lp_ro_c = safe_float(ro.get("lifo_realized_pnl_usd") if isinstance(ro, dict) else None, default=None)
                c_rpnl_c = safe_float(cyc.get("realized_pnl_usd"), default=None)
                if lp_ro_c is not None and c_rpnl_c is not None and c_rpnl_c < 0.0 and lp_ro_c > c_rpnl_c:
                    cyc["realized_pnl_usd"] = round(lp_ro_c, 2)
                c_is_eq = is_equity_venue_or_suffix(c_sym_u, c_venue)
                c_px = resolve_symbol_price(
                    ensure_equity_suffix(c_sym_u, is_equity=c_is_eq),
                    prices_and_trends,
                )
                if c_px > 0.0:
                    c_avg = safe_pos_float(cyc.get("avg_cost_px") or cyc.get("entry_price"), default=None)
                    c_entry = safe_pos_float(cyc.get("entry_price"), default=None)
                    c_pk = safe_pos_float(cyc.get("pos_peak"), default=None)
                    c_pk_missing_or_unwarmed = (
                        c_pk is None
                        or (c_avg is not None and abs(c_pk - c_avg) <= max(1e-6, max(c_pk, c_avg) * 1e-5))
                        or (c_entry is not None and abs(c_pk - c_entry) <= max(1e-6, max(c_pk, c_entry) * 1e-5))
                    )
                    if c_pk_missing_or_unwarmed:
                        oldest_c_ts = _extract_oldest_cycle_lot_ts_ms(
                            ro, fallback_ts_ms=cyc_sell_ts_map.get((f"{c_venue}:entry", c_sym_u)) or 0
                        )
                        c_age_ms = (math.trunc(_time.time() * 1000) - oldest_c_ts) if oldest_c_ts > 0 else 0
                        if 12 * 3600 * 1000 <= c_age_ms <= 90 * 86400 * 1000:
                            hi_24h_c = _resolve_cycle_high_since(
                                c_sym_u, c_quote_variants, oldest_c_ts, prices_and_trends, cache_dir=cache_dir, memo=_cycle_high_memo
                            )
                            ls_c_d = ro.get("last_sell") if isinstance(ro, dict) and isinstance(ro.get("last_sell"), dict) else {}
                            ls_c_ts = safe_pos_int(ls_c_d.get("ts_ms"), default=0) or 0
                            ls_c_px = (safe_pos_float(cyc.get("last_sell_price") or ls_c_d.get("price"), default=0.0) or 0.0) if ls_c_ts >= oldest_c_ts else 0.0
                            ref_pk_anchor_c = max(c_px, c_avg or 0.0, c_entry or 0.0, ls_c_px)
                            if hi_24h_c > 0.0 and not _prices_diverged(ref_pk_anchor_c, hi_24h_c, threshold=0.35):
                                c_pk = max(
                                    hi_24h_c,
                                    ls_c_px,
                                    c_px,
                                    c_pk or 0.0,
                                    c_avg or 0.0,
                                    safe_pos_float(cyc.get("entry_price"), default=0.0) or 0.0,
                                )
                    if c_pk is not None and c_pk > 0.0:
                        eff_c_pk = max(c_pk, c_px)
                        cyc["pos_peak"] = round(eff_c_pk, 6)
                        cyc["dd_from_peak_pct"] = pct_delta(c_px, eff_c_pk)
        return tradable_positions


for _port_eval_id, _port_eval_cls in (
    ("portfolio_venue_cycle_state", VenueCycleStateEvaluator),
    ("portfolio_symbol_recent_fills", SymbolRecentFillsEvaluator),
    ("portfolio_venue_cycle_reconciliation", VenueCycleReconciliationEvaluator),
    ("portfolio_position_cost_basis", PositionCostBasisEnrichmentEvaluator),
):
    TelemetryEvaluatorFactory.register(_port_eval_id, _port_eval_cls)

_VENUE_CYCLE_STATE_EVALUATOR = TelemetryEvaluatorFactory.create("portfolio_venue_cycle_state")
_SYMBOL_RECENT_FILLS_EVALUATOR = TelemetryEvaluatorFactory.create("portfolio_symbol_recent_fills")
_VENUE_CYCLE_RECONCILIATION_EVALUATOR = TelemetryEvaluatorFactory.create("portfolio_venue_cycle_reconciliation")
_POSITION_COST_BASIS_ENRICHMENT_EVALUATOR = TelemetryEvaluatorFactory.create("portfolio_position_cost_basis")


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
    return _VENUE_CYCLE_STATE_EVALUATOR.evaluate(
        st_data,
        symbol,
        prices_and_trends,
        avg_decimals=avg_decimals,
        fallback_prices=fallback_prices,
        include_meta=include_meta,
    )


def extract_symbol_recent_fills(
    orders_list: list,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[Dict[str, tuple[Optional[str], Optional[str]]]] = None,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
    symbol: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Extract last action side/actor, active cycle avg BUY price, last BUY/SELL fills, and recent tranches."""
    return _SYMBOL_RECENT_FILLS_EVALUATOR.evaluate(
        orders_list,
        known_ai_order_ids=known_ai_order_ids,
        audit_origin_map=audit_origin_map,
        ai_intents=ai_intents,
        symbol=symbol,
    )


def reconcile_venue_cycles_with_recent_orders(
    kraken_cycles: Dict[str, Dict[str, Any]],
    hyperliquid_cycles: Dict[str, Dict[str, Any]],
    t212_cycles: Dict[str, Dict[str, Any]],
    venue_recent_orders: Dict[str, Dict[str, Any]],
    live_held_by_venue_sym: Dict[tuple[str, str], float],
    prices_and_trends: Dict[str, Any],
) -> Dict[tuple[str, str], int]:
    """Reconcile `.state_*.json` DCA cycles against recent fills and live wallet balances; return `(venue, sym) -> sell_ts_ms`."""
    return _VENUE_CYCLE_RECONCILIATION_EVALUATOR.evaluate(
        kraken_cycles,
        hyperliquid_cycles,
        t212_cycles,
        venue_recent_orders,
        live_held_by_venue_sym,
        prices_and_trends,
    )


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
    cache_dir: str = "",
) -> None:
    """Enrich `tradable_positions` and venue cycles in-place with cost-basis anchors, LIFO PnL, trailing peaks, and venue fee/limits."""
    _POSITION_COST_BASIS_ENRICHMENT_EVALUATOR.evaluate(
        tradable_positions,
        prices_and_trends,
        inst_by_sym,
        venue_recent_orders,
        recent_orders,
        kraken_cycles,
        hyperliquid_cycles,
        t212_cycles,
        cyc_sell_ts_map,
        gateway_max_usd=gateway_max_usd,
        cache_dir=cache_dir,
    )


__all__ = [
    "PortfolioEvaluatorBase",
    "PositionCostBasisEnrichmentEvaluator",
    "SymbolRecentFillsEvaluator",
    "VenueCycleReconciliationEvaluator",
    "VenueCycleStateEvaluator",
    "_populate_cycle_fallback_recent_order",
    "_sync_cycle_wallet_qty_and_avg",
    "enrich_positions_with_orders_and_cycles",
    "extract_symbol_recent_fills",
    "parse_venue_cycle_state",
    "reconcile_venue_cycles_with_recent_orders",
    "set_pos_anchor",
]
