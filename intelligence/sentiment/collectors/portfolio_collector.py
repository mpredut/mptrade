"""Portfolio positions, venue DCA cycles, trailing peaks, recent fills, and runtime states collector."""
from __future__ import annotations

from abc import ABC
import logging
import os
import re
import time as _time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from botcore import (
    _parse_bool_flag,
    extract_order_id,
    extract_order_side,
    extract_timestamp_s,
    first_epoch_sec,
    first_pos_float,
    is_live_network_backfill_enabled,
    load_key_value_conf,
    normalize_timestamp_ms,
    safe_nonneg_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    filter_asset_class_variants,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    is_primary_stream_venue,
    iter_venue_cycle_state_files,
    normalize_venue_name,
    resolve_venue_trailing_state_path,
    split_symbol_assets,
)
from intelligence.macro_sentiment_ssot import (
    coerce_dict_maps,
    normalize_epoch_sec,
    resolve_read_json_cached,
    resolve_save_json_state,
    safe_float,
    safe_pos_float,
)
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
)
import intelligence.sentiment.collectors.microstructure_collector as _mc
from intelligence.sentiment.collectors.microstructure_collector import (
    _ensure_cacheprice_classes_bound,
    _iter_t212_state_files,
    _load_venue_balances_state,
    _resolve_now_and_read_json,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _first_variant_dict,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    clip_open_lots_to_held_qty,
    enrich_positions_with_orders_and_cycles,
    extract_symbol_recent_fills,
    parse_venue_cycle_state,
    reconcile_fresh_lots_vwap,
    reconcile_venue_cycles_with_recent_orders,
    select_fresh_open_lots,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _prices_diverged,
    round_opt_px,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    resolve_symbol_price,
    resolve_venue_fee_and_limits,
)
from instrument_registry import casefold_get
import order_retry as _oq
from providers.base import MarketDataProvider
from providers.order_ohlc_utils import (
    normalize_open_order,
    resolve_order_origin,
)
from providers.intent_state import collect_known_ai_order_ids, intent_venue_client_order_ids
from providers.quantity import resolve_assets, venue_fee_cap_quantity
from state_io import (
    generation_file_lock,
    read_json_dict,
    read_json_dict_recovering as _default_read_json_cached,
    read_json_file,
    read_json_recovering,
    read_tail_jsonl_dicts,
    resolve_cachedb_filepath,
    save_json_state as _default_save_json_state,
)

read_json_cached = _default_read_json_cached
save_json_state = _default_save_json_state

logger = logging.getLogger("intelligence.sentiment.portfolio_collector")

_resolve_read_json_cached = resolve_read_json_cached


def _resolve_order_cache_files() -> Tuple[Tuple[str, str], ...]:
    _ensure_cacheprice_classes_bound()
    contracts_fn = getattr(_mc.CacheAccountManagerBase, "venue_order_and_trade_cache_contracts", None)
    if callable(contracts_fn):
        return tuple(contracts_fn())
    trd_cls = _mc.CacheTradeManager
    return (
        (getattr(_mc.CacheOrderManager, "DEFAULT_FILENAME", ""), "binance"),
        (trd_cls.disk_filename("kraken"), "kraken"),
        (trd_cls.disk_filename("hyperliquid"), "hyperliquid"),
        (trd_cls.disk_filename("t212"), "t212"),
    )


def _iter_audit_dir_file_batches(
    cache_dir: Optional[str], *, max_files: int = 2, newest_first: bool = True
) -> List[List[str]]:
    """Yield sorted batches of `execution_audit_*.jsonl` file paths from candidate logger/cache directories."""
    if not cache_dir:
        return []
    abs_cache = os.path.abspath(cache_dir)
    batches: List[List[str]] = []
    seen_dirs: set[str] = set()
    for d in (
        cache_dir,
        os.path.join(os.path.dirname(abs_cache), "logger", "execution_audit"),
        os.path.join(os.path.dirname(os.path.dirname(abs_cache)), "logger", "execution_audit"),
    ):
        abs_d = os.path.abspath(d)
        if abs_d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(abs_d)
        try:
            files = sorted(
                (os.path.join(d, f) for f in os.listdir(d) if f.startswith("execution_audit") and f.endswith(".jsonl")),
                reverse=newest_first,
            )
            batch = files[:max_files] if newest_first else files[-max_files:]
            if batch:
                batches.append(batch)
        except Exception:
            continue
    return batches


def _normalize_open_orders_list(
    raw_orders: Any,
    venue: str,
    symbol: str,
    known_ai_order_ids: Optional[set[str]] = None,
    *,
    audit_origin_map: Optional[Dict[str, tuple[Optional[str], Optional[str]]]] = None,
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
                audit_origin_map=audit_origin_map,
                now_ts=now_ts,
                order_ts_map=order_ts_map,
            )),
            dict,
        )
    ]


def _load_cached_ai_intents_list(cache_dir: Optional[str], read_json_fn: Any = None) -> List[Dict[str, Any]]:
    """Load persisted AI trade intents from `autonomous_trade_intents_state.json` across `state/` and root cache paths."""
    if not cache_dir:
        return []
    _rj = _resolve_read_json_cached(read_json_fn)
    raw_path = os.path.join(cache_dir, "autonomous_trade_intents_state.json")
    resolved_path = str(resolve_cachedb_filepath(raw_path))
    raw_int: Any = None
    if _rj not in (_default_read_json_cached, read_json_recovering):
        for cand_p in dict.fromkeys((resolved_path, raw_path)):
            try:
                raw_int = _rj(cand_p)
            except Exception:
                raw_int = None
            if raw_int is not None:
                break
    if raw_int is None:
        for cand_p in dict.fromkeys((resolved_path, raw_path)):
            if os.path.exists(cand_p):
                raw_int = read_json_file(cand_p, default=[], recover_previous=True)
                if raw_int:
                    break
    int_list = raw_int.get("intents") if isinstance(raw_int, dict) else (raw_int if isinstance(raw_int, list) else [])
    return [it for it in (int_list or ()) if isinstance(it, dict)]


def _resolve_ai_intents_and_known_order_ids(
    cache_dir: Optional[str],
    *,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
    known_ai_order_ids: Optional[set[str]] = None,
    read_json_fn: Any = None,
    need_intents: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[set[str]]]:
    """Resolve cached AI trade intents and ``known_ai_order_ids`` without redundant disk reads."""
    if ai_intents is None and (need_intents or known_ai_order_ids is None) and cache_dir:
        cached_intents = _load_cached_ai_intents_list(cache_dir, read_json_fn)
        if cached_intents:
            ai_intents = cached_intents
        if known_ai_order_ids is None:
            known_ai_order_ids = collect_known_ai_order_ids(ai_intents or cached_intents)
    if known_ai_order_ids is None and ai_intents is not None:
        known_ai_order_ids = collect_known_ai_order_ids(ai_intents)
    return ai_intents, known_ai_order_ids


def _load_open_order_timestamps(cache_dir: Optional[str], cached_bals: Dict[str, Any]) -> Dict[str, float]:
    """Build order_id / client_order_id -> created_ts (epoch seconds) lookup for resting open orders."""
    out: Dict[str, float] = {}
    for k, v in (cached_bals or {}).items():
        if isinstance(k, str) and k.startswith("open_orders:") and isinstance(v, list):
            for oo_prev in v:
                if isinstance(oo_prev, dict):
                    c_ts = first_epoch_sec(
                        oo_prev.get("created_ts"),
                        oo_prev.get("time"),
                        oo_prev.get("timestamp"),
                        extract_timestamp_s(oo_prev),
                        allow_iso=True,
                    )
                    if c_ts is not None and c_ts > 0.0:
                        for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                            if kv := str(oo_prev.get(id_k) or "").strip():
                                out.setdefault(kv, c_ts)
    if not cache_dir:
        return out
    for it in _load_cached_ai_intents_list(cache_dir):
        it_ts = first_epoch_sec(
            it.get("executed_ts"),
            it.get("executed_at"),
            it.get("timestamp"),
            it.get("created_at"),
            it.get("ts"),
            allow_iso=True,
        )
        if it_ts is not None:
            exec_pay = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
            for src_d in (it, exec_pay):
                for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId", "intent_id"):
                    if kv := str(src_d.get(id_k) or "").strip():
                        out.setdefault(kv, it_ts)
            if oid_pay := extract_order_id(exec_pay, default=""):
                out.setdefault(str(oid_pay), it_ts)
            for cid_v in intent_venue_client_order_ids(it):
                if cid_v:
                    out.setdefault(cid_v, it_ts)
    for files in _iter_audit_dir_file_batches(cache_dir, max_files=2, newest_first=False):
        found_in_batch = False
        for fpath in files:
            for rec in read_tail_jsonl_dicts(fpath, limit=None, max_bytes=512 * 1024):
                if (rec_ts := normalize_epoch_sec(rec.get("ts"), default=None, allow_iso=True)) is not None:
                    for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := str(rec.get(id_k) or "").strip():
                            out.setdefault(kv, rec_ts)
                            found_in_batch = True
        if found_in_batch:
            break
    return out


def _load_recent_audit_origins(cache_dir: str) -> Dict[str, tuple[Optional[str], Optional[str]]]:
    """Load recent orderId/clientOrderId -> (module, rule) mappings from execution_audit jsonl logs."""
    out: Dict[str, tuple[Optional[str], Optional[str]]] = {}
    for files in _iter_audit_dir_file_batches(cache_dir, max_files=2, newest_first=True):
        for fpath in files:
            for rec in read_tail_jsonl_dicts(fpath, limit=None, max_bytes=1024 * 1024):
                _, mod, rule = resolve_order_origin(rec)
                if mod or rule:
                    for k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := str(rec.get(k) or "").strip():
                            prev_mod, prev_rule = out.get(kv, (None, None))
                            out[kv] = (prev_mod or mod, prev_rule or rule)
        if out:
            break
    return out


def _load_audit_terminal_orders(cache_dir: str, venue: str) -> Dict[str, List[Dict[str, Any]]]:
    """Synthesize deduplicated per-symbol filled order lists from execution_audit_*.jsonl for venues lacking a disk order cache."""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    if not cache_dir:
        return by_sym
    v_target = normalize_venue_name(venue)
    for audit_files in _iter_audit_dir_file_batches(cache_dir, max_files=10, newest_first=False):
        try:
            submit_ts_by_oid: Dict[str, float] = {}
            idx_by_oid: Dict[tuple[str, str], int] = {}
            for af in audit_files:
                for rec in read_tail_jsonl_dicts(af, limit=None, max_bytes=1024 * 1024):
                    if normalize_venue_name(rec.get("venue")) != v_target:
                        continue
                    ev_name = str(rec.get("event") or "")
                    oid_s = str(rec.get("order_id") or "").strip()
                    ts_s = normalize_epoch_sec(rec.get("ts"), default=0.0, allow_iso=True) or 0.0
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
                            "timestamp": round(eff_ts_s * 1000.0),
                            "status": "FILLED",
                        }
                        dedup_k = oid_s or str(rec.get("client_order_id") or "").strip()
                        sym_list = by_sym.setdefault(sym_t, [])
                        if dedup_k and (sym_t, dedup_k) in idx_by_oid:
                            existing = sym_list[idx_by_oid[(sym_t, dedup_k)]]
                            new_q = max(safe_pos_float(existing.get("qty"), default=0.0) or 0.0, f_q)
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


def _attach_trailing_peak_to_position(
    pos_entry: Dict[str, Any],
    v_norm: str,
    sym: str,
    px_f: float,
    fb_val: Optional[float],
    raw_peak: Any,
    pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]],
    *,
    fallback_dd: Any = None,
) -> None:
    """Attach trailing peak and drawdown to `pos_entry` when held base exceeds min notional, or defer to `pending_peaks_by_vs`."""
    for k in ("trailing_peak", "dd_from_peak_pct"):
        pos_entry.pop(k, None)
    peak_f = safe_pos_float(raw_peak, default=0.0) or 0.0
    if raw_peak is None or peak_f <= 0.0:
        return
    eff_peak = max(peak_f, px_f) if px_f > 0.0 else peak_f
    dd_pct = (round(((px_f - eff_peak) / eff_peak) * 100.0, 2) + 0.0) if px_f > 0.0 else fallback_dd
    min_notional_f = max(1.0, first_pos_float(pos_entry.get("exchange_min_notional_usd"), pos_entry.get("min_notional_usd"), default=1.0) or 1.0)
    has_held_base = fb_val is None or (fb_val > 1e-8 and (px_f <= 0.0 or fb_val * px_f >= min_notional_f))
    if has_held_base:
        pos_entry["trailing_peak"] = round(eff_peak, 6)
        pos_entry["dd_from_peak_pct"] = dd_pct
    elif v_norm and sym:
        pending_peaks_by_vs[(v_norm, sym)] = (round(eff_peak, 6), dd_pct)


def _iter_valid_tradable_positions(
    tradable_positions: List[Dict[str, Any]],
) -> List[tuple[Dict[str, Any], str, str]]:
    out: List[tuple[Dict[str, Any], str, str]] = []
    for pos_item in tradable_positions:
        if isinstance(pos_item, dict):
            pv = normalize_venue_name(pos_item.get("venue"))
            ps = str(pos_item.get("symbol") or "").strip().upper()
            if pv and ps:
                out.append((pos_item, pv, ps))
    return out


def _apply_open_orders_locked_balances(
    open_orders_list: List[Dict[str, Any]],
    tradable_positions: List[Dict[str, Any]],
    venue_quote_cash: Dict[str, Dict[str, Any]],
    inst_by_sym: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    updated_bals: Dict[str, Any],
    pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]],
    *,
    now_ts_oo: float,
    gateway_max_usd: float,
) -> None:
    """Aggregate locked quote/base from `open_orders_list` and enrich `venue_quote_cash` and `tradable_positions`."""
    if not open_orders_list:
        return
    pos_by_vs: Dict[tuple[str, str], Dict[str, Any]] = {
        (pv, ps): pos_item for pos_item, pv, ps in _iter_valid_tradable_positions(tradable_positions)
    }

    locked_quote_by_vq: Dict[tuple[str, str], float] = {}
    locked_quote_by_vs: Dict[tuple[str, str], float] = {}
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
            oo_lookup_sym = ensure_equity_suffix(oo_sym, is_equity=is_equity_venue(oo_v))
            spot_oo = safe_pos_float((prices_and_trends.get(oo_sym) or {}).get("price"), default=0.0) or resolve_symbol_price(oo_lookup_sym, prices_and_trends)
            if spot_oo > 0.0:
                oo_item["dist_from_spot_pct"] = round(((oo_px - spot_oo) / spot_oo) * 100.0, 2)
        if not oo_v or not oo_sym or oo_rem <= 0.0:
            continue
        inst_oo = inst_by_sym.get(f"{oo_v}:{oo_sym}") or inst_by_sym.get(oo_sym)
        matched_pos = pos_by_vs.get((oo_v, oo_sym))
        oo_pos_sym = oo_sym
        if matched_pos is None:
            oo_canon = canonical_symbol_key(oo_sym)
            oo_base = canonical_base_asset(oo_sym)
            for (pv, ps), p_obj in pos_by_vs.items():
                if pv != oo_v:
                    continue
                if (oo_canon and canonical_symbol_key(ps) == oo_canon) or (
                    MarketDataProvider.uses_bare_base_order_symbols(oo_v) and oo_base and canonical_base_asset(ps) == oo_base
                ):
                    matched_pos = p_obj
                    oo_pos_sym = ps
                    if inst_oo is None:
                        inst_oo = inst_by_sym.get(f"{oo_v}:{ps}") or inst_by_sym.get(ps)
                    break
        inst_q = str(getattr(inst_oo, "quote", "") or getattr(inst_oo, "quote_asset", "") or "").strip().upper() if inst_oo is not None else ""
        pos_q = str((matched_pos or {}).get("quote") or (matched_pos or {}).get("quote_asset") or "").strip().upper()
        _, parsed_q = resolve_assets(oo_sym, None, None)
        oo_q_u = inst_q or pos_q or str(parsed_q or "").strip().upper() or MarketDataProvider.resolve_venue_default_quote(oo_v, fallback="USD")
        if oo_side == "BUY" and oo_px > 0.0:
            locked_notional = oo_rem * oo_px
            locked_quote_by_vq[(oo_v, oo_q_u)] = locked_quote_by_vq.get((oo_v, oo_q_u), 0.0) + locked_notional
            locked_quote_by_vs[(oo_v, oo_pos_sym)] = locked_quote_by_vs.get((oo_v, oo_pos_sym), 0.0) + locked_notional
        elif oo_side == "SELL":
            locked_base_by_vs[(oo_v, oo_pos_sym)] = locked_base_by_vs.get((oo_v, oo_pos_sym), 0.0) + oo_rem

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
        lq_pos = locked_quote_by_vs.get((p_v_norm, p_sym_u), 0.0)
        lb_pos = locked_base_by_vs.get((p_v_norm, p_sym_u), 0.0)
        if lq_pos > 0.0 or lb_pos > 0.0:
            inst_obj = inst_by_sym.get(f"{p_v_norm}:{p_sym_u}") or inst_by_sym.get(p_sym_u)
            px_pos = safe_pos_float(pos_entry.get("price"), default=0.0) or 0.0
            fb_pos = safe_nonneg_float(pos_entry.get("free_base"), default=None)
            prev_inst_limits = (
                {
                    k: pos_entry[k]
                    for k in (
                        "fee_tier",
                        "maker_fee_bps",
                        "taker_fee_bps",
                        "maker_rebate_edge_bps",
                        "exchange_min_notional_usd",
                        "min_qty",
                        "step_size",
                        "tick_size",
                        "spread_bps",
                    )
                    if pos_entry.get(k) is not None
                }
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
                min_not_pos = max(1.0, first_pos_float(pos_entry.get("exchange_min_notional_usd"), pos_entry.get("min_notional_usd"), default=1.0) or 1.0)
                held_total = (fb_pos or 0.0) + lb_pos
                if px_pos <= 0.0 or held_total * px_pos >= min_not_pos:
                    pos_entry["trailing_peak"] = pend_peak[0]
                    pos_entry["dd_from_peak_pct"] = pend_peak[1]
            pos_bkey = f"pos:{p_v_norm}:{p_sym_u}"
            if pos_bkey in updated_bals:
                prev_b_ts = updated_bals[pos_bkey].get("ts") if isinstance(updated_bals[pos_bkey], dict) else None
                updated_bals[pos_bkey] = {**pos_entry, "ts": prev_b_ts} if prev_b_ts is not None else dict(pos_entry)


def _merge_audit_orders(primary_list: Any, audit_list: Any) -> List[Dict[str, Any]]:
    """Merge cached venue orders with terminal execution_audit orders, deduplicating by orderId or side/qty/timestamp."""
    base_rows = [r for r in (primary_list if isinstance(primary_list, list) else ()) if isinstance(r, dict)]
    extra_rows = [r for r in (audit_list if isinstance(audit_list, list) else ()) if isinstance(r, dict)]
    if not base_rows:
        return extra_rows
    if not extra_rows:
        return base_rows
    seen_oids = {str(oid) for r in base_rows if (oid := extract_order_id(r, default=""))}
    merged = list(base_rows)
    for ar in extra_rows:
        a_oid = str(extract_order_id(ar, default="") or "")
        if a_oid and a_oid in seen_oids:
            continue
        a_side = extract_order_side(ar)
        a_ts = (
            normalize_timestamp_ms(ar.get("timestamp"), default=0)
            or normalize_timestamp_ms(ar.get("time"), default=0)
            or 0
        )
        a_qty = first_pos_float(ar.get("executedQty"), ar.get("qty"), default=0.0) or 0.0
        if a_ts > 0 and a_qty > 0.0 and any(
            extract_order_side(br) == a_side
            and abs(
                (
                    normalize_timestamp_ms(br.get("timestamp"), default=0)
                    or normalize_timestamp_ms(br.get("time"), default=0)
                    or 0
                )
                - a_ts
            ) <= 60000
            and abs((first_pos_float(br.get("executedQty"), br.get("qty"), default=0.0) or 0.0) - a_qty) <= max(1e-5, a_qty * 0.01)
            for br in base_rows
        ):
            continue
        if a_oid:
            seen_oids.add(a_oid)
        merged.append(ar)
    return merged


class PortfolioCacheCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for account/portfolio domain cache collectors.

    Modeled on ``CacheAccountManagerBase(CacheManagerInterface, ABC)``: encapsulates
    shared order/fill cache bindings (`CacheOrderManager`, `CacheTradeManager`,
    `CacheAssetValueManager`, and execution_audit fallback) and portfolio state readers.
    """

    cls_name = "PortfolioCacheCollectorBase"
    _on_collector_init = staticmethod(_ensure_cacheprice_classes_bound)


class VenueTrailingMapsCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for Binance and Kraken trailing peak state maps."""

    cls_name = "VenueTrailingMapsCollector"
    COLLECTOR_ID = "portfolio_trailing_maps"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        workspace_dir: str = "",
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
        _rj = _resolve_read_json_cached(read_json_fn)
        trailing_map: Dict[str, Dict[str, Any]] = {}
        trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]] = {"binance": {}, "kraken": {}}
        for trail_venue, raw_trail, default_quote in (
            ("binance", _rj(resolve_venue_trailing_state_path("binance", cache_dir=cachedb_dir)), ""),
            ("kraken", _rj(resolve_venue_trailing_state_path("kraken", workspace_dir=workspace_dir)), "USD"),
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


class VenueCyclesCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for Kraken, Hyperliquid, and Trading212 DCA `.state_*.json` cycles."""

    cls_name = "VenueCyclesCollector"
    COLLECTOR_ID = "portfolio_venue_cycles"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        workspace_dir: str = "",
        prices_and_trends: Optional[Dict[str, Any]] = None,
        fallback_prices: Optional[Dict[str, float]] = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        eff_workspace = workspace_dir or cachedb_dir
        eff_pt = prices_and_trends or {}
        eff_fb = fallback_prices or {}
        _rj = _resolve_read_json_cached(read_json_fn)

        def _record_cyc(target: Dict[str, Any], st_path: str, sym: str, dec: int, *, is_eq: bool = False) -> None:
            if isinstance((st_data := _rj(st_path)), dict):
                lookup_sym = ensure_equity_suffix(sym, is_equity=is_eq)
                parsed = parse_venue_cycle_state(
                    st_data, lookup_sym, eff_pt, avg_decimals=dec, fallback_prices=eff_fb, include_meta=True
                )
                prev = target.get(sym)
                if isinstance(prev, dict):
                    prev_active = (safe_nonneg_float(prev.get("qty"), default=0.0) or 0.0) > 1e-8 or bool(prev.get("resting_orders"))
                    new_active = (safe_nonneg_float(parsed.get("qty"), default=0.0) or 0.0) > 1e-8 or bool(parsed.get("resting_orders"))
                    if prev_active and not new_active:
                        return
                target[sym] = parsed

        kraken_cycles: Dict[str, Any] = {}
        for st_path in iter_venue_cycle_state_files("kraken", workspace_dir=eff_workspace):
            pair_name = os.path.basename(st_path)[len(".state_") : -len(".json")].upper()
            if pair_name.isalnum() and pair_name.endswith(("USD", "EUR", "USDC", "USDT")):
                _record_cyc(kraken_cycles, st_path, pair_name, 6)

        hyperliquid_cycles: Dict[str, Any] = {}
        for st_path in iter_venue_cycle_state_files("hyperliquid", workspace_dir=eff_workspace):
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
        for st_path, ticker in _iter_t212_state_files(eff_workspace):
            _record_cyc(all_t212_cycles, st_path, ticker, 4, is_eq=True)

        return kraken_cycles, hyperliquid_cycles, all_t212_cycles


class KrakenXstocksCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for Kraken read-only xStocks holdings and T212 sibling linkage."""

    cls_name = "KrakenXstocksCollector"
    COLLECTOR_ID = "portfolio_kraken_xstocks"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        kraken_dir: str = "",
        prices_and_trends: Optional[Dict[str, Any]] = None,
        all_t212_cycles: Optional[Dict[str, Any]] = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        eff_k_dir = kraken_dir or cachedb_dir
        eff_pt = prices_and_trends or {}
        eff_t212 = all_t212_cycles or {}
        _rj = _resolve_read_json_cached(read_json_fn)
        cand_paths = [
            p for p in (os.path.join(eff_k_dir, "xstock_state.json"), os.path.join(eff_k_dir, ".xstock_state.json"))
            if os.path.exists(p)
        ]
        if not cand_paths:
            return None
        xst_path = max(cand_paths, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
        xst_data = _rj(xst_path)
        if not isinstance(xst_data, dict):
            return None
        xstock_cfg = load_key_value_conf(eff_k_dir, root_dir=eff_k_dir)
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
        listed_pair = (
            str(raw_pair).strip().upper()
            if isinstance(raw_pair, str) and raw_pair.strip() and raw_pair.strip().lower() not in ("none", "null", "nan")
            else None
        )
        if not (x_holdings or alloc_info or listed_pair):
            return None
        underlying = (xstock_cfg.get("xstock_yahoo") or "SPCX").strip().upper()
        t212_sib = f"{underlying}_US_EQ"
        alloc_px = safe_pos_float(xstock_cfg.get("xstock_alloc_price"), default=None)
        live_px = (
            (resolve_symbol_price(listed_pair, eff_pt) if listed_pair else 0.0)
            or resolve_symbol_price(t212_sib, eff_pt)
        )
        xst_summary: Dict[str, Any] = {
            "underlying": underlying,
            "t212_sibling": t212_sib,
            "holdings": x_holdings,
            "allocated": alloc_info,
            "listed_pair": listed_pair,
            "alloc_price": alloc_px,
            "tp1_alert_pct": safe_pos_float(xstock_cfg.get("xstock_tp_alert_pct"), default=None),
            "tp2_alert_pct": safe_pos_float(xstock_cfg.get("xstock_tp2_alert_pct"), default=None),
            "sl_alert_pct": safe_pos_float(xstock_cfg.get("xstock_sl_alert_pct"), default=None),
        }
        if live_px > 0:
            xst_summary["live_price"] = round(live_px, 4)
            if alloc_px is not None:
                xst_summary["pnl_vs_alloc_pct"] = round(((live_px - alloc_px) / alloc_px) * 100.0, 2)
        if isinstance((sib_cyc := _first_variant_dict(eff_t212, (t212_sib, underlying))), dict):
            xst_summary["t212_sibling_cycle"] = {k: v for k, v in sib_cyc.items() if not str(k).startswith("_")}
        return xst_summary


class TradablePositionsAndOrdersCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for spot wallet balances, trailing peaks, and resting open orders."""

    cls_name = "TradablePositionsAndOrdersCollector"
    COLLECTOR_ID = "portfolio_tradable_positions"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        inst_map: Optional[Dict[str, Any]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        trailing_map: Optional[Dict[str, Dict[str, Any]]] = None,
        trailing_by_venue: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        known_ai_order_ids: Optional[set[str]] = None,
        gateway_max_usd: float = 2000.0,
        verified_open_venues: Optional[set[str]] = None,
        allow_empty_inst_recovery: bool = False,
        allow_network: bool = True,
        persist: bool = True,
        force_persist: bool = False,
        **kwargs: Any,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        inst_map = inst_map or {}
        prices_and_trends = prices_and_trends if isinstance(prices_and_trends, dict) else {}
        trailing_map = trailing_map or {}
        trailing_by_venue = trailing_by_venue or {}
        cache_dir = cachedb_dir or None

        tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = [], [], {}, {}
        balance_memo: Dict[tuple[str, str], Optional[float]] = {}
        _rj_bals = _resolve_read_json_cached()
        bcache_path, cached_bals, _ = _load_venue_balances_state(cache_dir, _rj_bals)
        updated_bals: Dict[str, Any] = {}
        live_open_ok_venues: set[str] = set()
        failed_open_venues: set[str] = set()
        now_ts_oo = _time.time()
        order_ts_map = _load_open_order_timestamps(cache_dir, cached_bals)
        audit_origin_map = _load_recent_audit_origins(cache_dir) if cache_dir else {}
        _, known_ai_order_ids = _resolve_ai_intents_and_known_order_ids(
            cache_dir, known_ai_order_ids=known_ai_order_ids, read_json_fn=_rj_bals
        )

        def _query_free_bal(inst_obj: Any, asset: str) -> Optional[float]:
            if not asset:
                return None
            raw_v = str(getattr(inst_obj, "provider_name", "") or "").strip().lower()
            v_name = normalize_venue_name(raw_v) or raw_v
            key = (v_name, asset.upper())
            if key in balance_memo:
                return balance_memo[key]
            free_fn = getattr(getattr(inst_obj, "_provider", None), "free_balance", None) if allow_network else None
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
            merged_params_by_vs: Dict[tuple[str, str], Dict[str, Any]] = {}
            for _inst_cand in inst_map.values():
                _c_sym = str(getattr(_inst_cand, "symbol", "") or "").strip().upper()
                _c_ven = normalize_venue_name(str(getattr(_inst_cand, "provider_name", "") or "").strip().lower())
                if _c_sym and _c_ven and isinstance(_c_p := getattr(_inst_cand, "params", None), dict) and _c_p:
                    merged_params_by_vs.setdefault((_c_ven, _c_sym), {}).update(_c_p)
            seen_vs: set[tuple[str, str]] = set()
            for inst in inst_map.values():
                sym = str(getattr(inst, "symbol", "") or "").strip().upper()
                venue = str(getattr(inst, "provider_name", "") or "").strip().lower()
                if not sym or not venue:
                    continue
                v_norm = normalize_venue_name(venue)
                if (v_norm, sym) in seen_vs:
                    continue
                seen_vs.add((v_norm, sym))
                is_eq_v = is_equity_venue(v_norm)
                pos_px_dec = MarketDataProvider.resolve_venue_position_price_decimals(v_norm)
                res_b, res_q = resolve_assets(sym, getattr(inst, "base", None), getattr(inst, "quote", None))
                base, quote = res_b or sym, res_q or "USD"
                if sym not in inst_by_sym or not MarketDataProvider.uses_bare_base_order_symbols(v_norm):
                    inst_by_sym[sym] = inst
                inst_by_sym[f"{v_norm}:{sym}"] = inst

                base_free, quote_free = _query_free_bal(inst, base), _query_free_bal(inst, quote)
                px = prices_and_trends.get(sym, {}).get("price")
                px_f = safe_pos_float(px, default=0.0) or 0.0
                pos_bkey_cur = f"pos:{v_norm}:{sym}"
                prev_cached_pos = cached_bals.get(pos_bkey_cur) if isinstance(cached_bals.get(pos_bkey_cur), dict) else {}
                if not allow_network:
                    if base_free is None and prev_cached_pos:
                        base_free = safe_nonneg_float(
                            prev_cached_pos.get("free_base")
                            if prev_cached_pos.get("free_base") is not None
                            else prev_cached_pos.get("free"),
                            default=None,
                        )
                    if quote_free is None:
                        if prev_cached_pos and prev_cached_pos.get("free_quote") is not None:
                            quote_free = safe_nonneg_float(prev_cached_pos.get("free_quote"), default=None)
                        elif isinstance(c_cash := cached_bals.get(f"cash:{v_norm}:{quote.upper()}"), dict):
                            quote_free = safe_nonneg_float(c_cash.get("free"), default=None)
                prev_pos_px = safe_pos_float(prev_cached_pos.get("price"), default=0.0) or 0.0
                prev_pos_ts = safe_pos_float(prev_cached_pos.get("ts"), default=None)
                px_from_cached_pos = False
                if allow_network and (
                    px_f <= 0
                    or (
                        is_eq_v
                        and (
                            _parse_bool_flag(force_persist, False)
                            or ((base_free or 0.0) > 0 and is_live_network_backfill_enabled())
                        )
                    )
                ):
                    try:
                        get_px_fn = getattr(inst._provider, "get_current_price", None)
                        if callable(get_px_fn) and (live_px := safe_pos_float(get_px_fn(sym), default=0.0) or 0.0) > 0:
                            px = px_f = round(live_px, pos_px_dec) if is_eq_v else live_px
                    except Exception:
                        if px_f <= 0:
                            px, px_f = None, 0.0
                elif (
                    not allow_network
                    and is_eq_v
                    and prev_pos_px > 0
                    and prev_pos_ts is not None
                    and 0.0 <= (now_ts_oo - prev_pos_ts) <= 300.0
                ):
                    px = px_f = round(prev_pos_px, pos_px_dec)
                    px_from_cached_pos = True
                lookup_sym_p = ensure_equity_suffix(sym, is_equity=is_eq_v)
                if px_f <= 0 and (px_f := resolve_symbol_price(lookup_sym_p, prices_and_trends, explicit_quote=None if is_eq_v else quote)) > 0:
                    px = px_f
                if px_f <= 0 and prev_pos_px > 0:
                    px = px_f = round(prev_pos_px, pos_px_dec) if is_eq_v else prev_pos_px
                    px_from_cached_pos = True
                sym_entry = prices_and_trends.setdefault(sym, {"price": px if px_f > 0 else None, "trend": 0, "growth_coefficient": 0.0})
                if px_f > 0 and ((safe_pos_float(sym_entry.get("price"), default=0.0) or 0.0) <= 0 or is_eq_v):
                    sym_entry["price"] = px
                quote_breakdown: Optional[Dict[str, float]] = None
                bd_key = f"breakdown:{v_norm}:{quote.upper()}"
                if allow_network and quote.upper() == "USD" and callable(bd_fn := getattr(inst._provider, "usd_cash_breakdown", None)):
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
                params = merged_params_by_vs.get((v_norm, sym)) or (
                    getattr(inst, "params", {}) if isinstance(getattr(inst, "params", None), dict) else {}
                )
                sym_quotes = compatible_quote_suffixes(sym, quote)
                t_state = _first_variant_dict(
                    trailing_by_venue.get(v_norm, {}),
                    cache_symbol_variants(sym, include_original=True, include_base="USD" in sym_quotes, quotes=sym_quotes),
                    default={},
                )
                if not t_state and is_primary_stream_venue(v_norm) and isinstance((val_b := casefold_get(trailing_map, sym)), dict):
                    t_state = val_b
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
                _attach_trailing_peak_to_position(
                    pos_entry, v_norm, sym, px_f, base_free, t_state.get("peak"), pending_peaks_by_vs
                )
                if params.get("trailing.pct"):
                    pos_entry["trailing_pct_cfg"] = params.get("trailing.pct")
                if params.get("mt.gain") or params.get("mt.lost"):
                    pos_entry["mt_gain_pct"], pos_entry["mt_lost_pct"] = params.get("mt.gain"), params.get("mt.lost")
                tradable_positions.append(pos_entry)
                eff_pos_ts = (
                    prev_pos_ts or round(now_ts_oo, 3)
                    if (
                        px_from_cached_pos
                        or (
                            is_eq_v
                            and prev_pos_ts is not None
                            and prev_pos_px > 0.0
                            and px_f > 0.0
                            and abs(round(px_f, pos_px_dec) - round(prev_pos_px, pos_px_dec)) < 1e-4
                        )
                    )
                    else round(now_ts_oo, 3)
                )
                updated_bals[pos_bkey_cur] = {**pos_entry, "ts": eff_pos_ts}

                oo_key = f"open_orders:{v_norm}:{sym}"
                if allow_network and callable(open_fn := getattr(inst, "open_orders", None)):
                    sym_open_ok = False
                    try:
                        if isinstance(raw_open := open_fn(), list):
                            sym_open_ok = True
                            sym_norm_orders = _normalize_open_orders_list(
                                raw_open,
                                venue,
                                sym,
                                known_ai_order_ids,
                                audit_origin_map=audit_origin_map,
                                now_ts=now_ts_oo,
                                order_ts_map=order_ts_map,
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
                                cached_bals.get(oo_key),
                                venue,
                                sym,
                                known_ai_order_ids,
                                audit_origin_map=audit_origin_map,
                                now_ts=now_ts_oo,
                                order_ts_map=order_ts_map,
                            )
                        )
                elif oo_key in cached_bals:
                    open_orders_list.extend(
                        _normalize_open_orders_list(
                            cached_bals.get(oo_key),
                            venue,
                            sym,
                            known_ai_order_ids,
                            audit_origin_map=audit_origin_map,
                            now_ts=now_ts_oo,
                            order_ts_map=order_ts_map,
                        )
                    )
                    ver_ts = safe_pos_float(cached_bals.get(f"open_orders_verified_ts:{v_norm}"), default=0.0) or 0.0
                    if ver_ts > 0.0 and 0.0 <= (now_ts_oo - ver_ts) <= 300.0:
                        live_open_ok_venues.add(v_norm)
        elif (allow_empty_inst_recovery or not allow_network) and cached_bals:
            gw_max_rec = safe_pos_float(gateway_max_usd, default=2000.0) or 2000.0
            for k, v in cached_bals.items():
                if k.startswith("pos:") and isinstance(v, dict):
                    rec_pos = dict(v)
                    for stale_k in ("locked_quote_usd", "replace_max_buy_usd", "locked_base_qty", "locked_base_usd", "replace_max_sell_usd", "ts"):
                        rec_pos.pop(stale_k, None)
                    p_sym = str(rec_pos.get("symbol") or "").strip().upper()
                    p_quote = str(rec_pos.get("quote") or "USD").strip().upper()
                    v_rec_norm = normalize_venue_name(rec_pos.get("venue"))
                    is_eq_rec = is_equity_venue(v_rec_norm)
                    fb_val = safe_nonneg_float(rec_pos.get("free_base"), default=None)
                    px_rec = 0.0
                    if p_sym:
                        exact_pt_px = safe_pos_float((prices_and_trends.get(p_sym) or {}).get("price"), default=0.0) or 0.0
                        cached_pos_px = safe_pos_float(rec_pos.get("price"), default=0.0) or 0.0
                        p_lookup_rec = ensure_equity_suffix(p_sym, is_equity=is_eq_rec)
                        sib_pt_px = (
                            resolve_symbol_price(p_lookup_rec, prices_and_trends, explicit_quote=None if is_eq_rec else p_quote)
                            if exact_pt_px <= 0.0
                            else exact_pt_px
                        )
                        if exact_pt_px > 0.0:
                            px_rec = exact_pt_px
                        elif cached_pos_px > 0.0 and (sib_pt_px <= 0.0 or not _prices_diverged(cached_pos_px, sib_pt_px, threshold=0.35 if is_eq_rec else 0.10)):
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
                    if p_sym:
                        v_trail_map = trailing_by_venue.get(v_rec_norm, {}) if isinstance(trailing_by_venue, dict) else {}
                        t_cands = cache_symbol_variants(p_sym, include_original=True, include_base=True, quotes=compatible_quote_suffixes(p_sym))
                        t_dict = _first_variant_dict(v_trail_map, t_cands) or (
                            _first_variant_dict(trailing_map, t_cands) if MarketDataProvider.uses_shared_trailing_map(v_rec_norm) else None
                        )
                        if isinstance(t_dict, dict) and ("peak" in t_dict or rec_peak is None):
                            rec_peak = safe_pos_float(t_dict.get("peak"), default=None)
                    _attach_trailing_peak_to_position(
                        rec_pos, v_rec_norm, p_sym, px_rec, fb_val, rec_peak, pending_peaks_by_vs,
                        fallback_dd=rec_pos.get("dd_from_peak_pct"),
                    )
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
                            audit_origin_map=audit_origin_map,
                            now_ts=now_ts_oo,
                            order_ts_map=order_ts_map,
                        )
                    )

        verified_now = live_open_ok_venues - failed_open_venues
        if isinstance(verified_open_venues, set):
            verified_open_venues.update(verified_now)
        if allow_network:
            for v_ok in verified_now:
                updated_bals[f"open_orders_verified_ts:{v_ok}"] = round(now_ts_oo, 3)

        if len(open_orders_list) > 1:
            deduped_open: List[Dict[str, Any]] = []
            seen_open_keys: set[tuple[str, str, str]] = set()
            for oo_item in open_orders_list:
                if not isinstance(oo_item, dict):
                    deduped_open.append(oo_item)
                    continue
                oo_oid = str(oo_item.get("order_id") or oo_item.get("orderId") or "").strip()
                if oo_oid:
                    oo_k = (
                        normalize_venue_name(oo_item.get("venue")),
                        str(oo_item.get("symbol") or "").strip().upper(),
                        oo_oid,
                    )
                    if oo_k in seen_open_keys:
                        continue
                    seen_open_keys.add(oo_k)
                deduped_open.append(oo_item)
            open_orders_list = deduped_open

        _apply_open_orders_locked_balances(
            open_orders_list,
            tradable_positions,
            venue_quote_cash,
            inst_by_sym,
            prices_and_trends,
            updated_bals,
            pending_peaks_by_vs,
            now_ts_oo=now_ts_oo,
            gateway_max_usd=gateway_max_usd,
        )

        if persist and allow_network and bcache_path and updated_bals and (
            _parse_bool_flag(force_persist, False) or is_live_network_backfill_enabled() or os.path.exists(bcache_path)
        ):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(bcache_path)), exist_ok=True)
                with generation_file_lock(bcache_path):
                    disk_bals = (
                        read_json_dict(bcache_path, default=None, repair_files=True)
                        if os.path.exists(bcache_path)
                        and _rj_bals in (_default_read_json_cached, read_json_recovering, read_json_cached)
                        else None
                    )
                    src_bals = disk_bals if isinstance(disk_bals, dict) else cached_bals
                    base_bals = {
                        k: v
                        for k, v in src_bals.items()
                        if not (k.startswith("open_orders_verified_ts:") and k.split(":", 1)[1] in failed_open_venues)
                    }
                    merged_bals = {**base_bals, **updated_bals}
                    if merged_bals != src_bals:
                        resolve_save_json_state(local_fn=save_json_state)(bcache_path, merged_bals, indent=2)
            except Exception:
                pass

        return tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym


class SymbolBestRecentFillsCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for single-symbol recent fills across `CacheOrderManager` and venue trade caches."""

    cls_name = "SymbolBestRecentFillsCollector"
    COLLECTOR_ID = "portfolio_symbol_recent_fills"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        quote_variants: Sequence[str] = (),
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        _rj = _resolve_read_json_cached(read_json_fn)
        loaded_intents = _load_cached_ai_intents_list(cachedb_dir, _rj)
        known_ai_oids = collect_known_ai_order_ids(loaded_intents)
        ai_intents_list = loaded_intents or None
        audit_origin_map = _load_recent_audit_origins(cachedb_dir)
        fills: Optional[Dict[str, Any]] = None
        best_venue: Optional[str] = None
        best_fill_key: Tuple[int, int, float] = (-1, -1, -1.0)
        sym_upper = str(symbol or "").strip().upper()
        is_eq_sym = has_equity_suffix(sym_upper)
        is_crypto_pair = (not is_eq_sym) and (split_symbol_assets(sym_upper) is not None)
        scoped_variants = (
            list(filter_asset_class_variants(sym_upper, quote_variants, is_equity=is_eq_sym))
            if (is_eq_sym or is_crypto_pair)
            else list(quote_variants)
        )
        sym_quotes = compatible_quote_suffixes(sym_upper)
        base_upper = canonical_base_asset(sym_upper)
        for ord_file, cache_venue in _resolve_order_cache_files():
            is_eq_cache_v = is_equity_venue(cache_venue)
            if (is_eq_sym and not is_eq_cache_v) or (is_crypto_pair and is_eq_cache_v):
                continue
            ord_data = _rj(os.path.join(cachedb_dir, ord_file))
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) and isinstance(ord_data.get("items"), dict) else {}
            matched_cand = next((cand for cand in scoped_variants if casefold_get(o_items, cand)), None)
            orders_for_sym = casefold_get(o_items, matched_cand) if matched_cand else []
            if MarketDataProvider.merges_audit_terminal_orders(cache_venue):
                audit_items = _load_audit_terminal_orders(cachedb_dir, cache_venue)
                audit_cand = next((cand for cand in scoped_variants if casefold_get(audit_items, cand)), None)
                if audit_cand:
                    orders_for_sym = _merge_audit_orders(orders_for_sym, casefold_get(audit_items, audit_cand))
                    if not matched_cand:
                        matched_cand = audit_cand
            v_ai_intents = (
                [it for it in ai_intents_list if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == cache_venue]
                if ai_intents_list is not None
                else None
            )
            if cand_fills := extract_symbol_recent_fills(
                orders_for_sym,
                known_ai_order_ids=known_ai_oids,
                audit_origin_map=audit_origin_map,
                ai_intents=v_ai_intents,
                symbol=symbol,
            ):
                cand_u = str(matched_cand or "").strip().upper()
                default_q_v = MarketDataProvider.resolve_venue_default_quote(cache_venue, fallback="USD")
                is_bare_base_cand = MarketDataProvider.uses_bare_base_order_symbols(cache_venue) and cand_u == base_upper
                cand_quotes = (
                    compatible_quote_suffixes(f"{base_upper}{default_q_v}")
                    if is_bare_base_cand
                    else compatible_quote_suffixes(cand_u)
                )
                same_quote_fam = 1 if (not sym_quotes or cand_quotes == sym_quotes) else 0
                sym_clean = clean_symbol_key(sym_upper)
                cand_clean = clean_symbol_key(cand_u)
                exact_sym = (
                    1
                    if (
                        cand_clean == sym_clean
                        or (is_bare_base_cand and sym_clean == f"{base_upper}{default_q_v}")
                    )
                    else 0
                )
                cand_ts = max(
                    safe_nonneg_float((cand_fills.get("last_buy") or {}).get("ts_ms"), 0.0),
                    safe_nonneg_float((cand_fills.get("last_sell") or {}).get("ts_ms"), 0.0),
                )
                cand_key = (same_quote_fam, exact_sym, cand_ts)
                if fills is None or cand_key > best_fill_key:
                    cand_fills.setdefault("venue", cache_venue)
                    fills, best_venue, best_fill_key = cand_fills, cache_venue, cand_key
        if fills is None or fills.get("last_buy") is None or fills.get("last_sell") is None or is_equity_venue(best_venue):
            norm_cdir = os.path.normpath(os.path.abspath(cachedb_dir)) if cachedb_dir else ""
            ws_dir = os.path.dirname(norm_cdir) if os.path.basename(norm_cdir).lower() == "cachedb" else norm_cdir
            if ws_dir and os.path.isdir(ws_dir):
                k_cycs, hl_cycs, t212_cycs = harvest_venue_cycles(ws_dir, {}, {}, read_json_fn=_rj)
                cyc_maps = [
                    vc
                    for vc in (("t212", t212_cycs), ("kraken", k_cycs), ("hyperliquid", hl_cycs))
                    if not ((is_eq_sym and not is_equity_venue(vc[0])) or (is_crypto_pair and is_equity_venue(vc[0])))
                ]
                if best_venue:
                    cyc_maps.sort(key=lambda vc: 0 if vc[0] == best_venue else 1)
                cyc_cands = list(dict.fromkeys([sym_upper, *(str(c).strip().upper() for c in scoped_variants if str(c).strip())]))
                matched_cyc: Optional[Dict[str, Any]] = None
                matched_cyc_venue: Optional[str] = None
                for cand in cyc_cands:
                    if sym_quotes and compatible_quote_suffixes(cand) != sym_quotes:
                        continue
                    for c_venue, c_map in cyc_maps:
                        if isinstance((val := casefold_get(c_map, cand)), dict):
                            matched_cyc, matched_cyc_venue = val, c_venue
                            break
                    if matched_cyc is not None:
                        break
                if isinstance(matched_cyc, dict):
                    cyc_lb = first_pos_float(matched_cyc.get("last_buy_price"), matched_cyc.get("entry_price"))
                    cyc_ls = safe_pos_float(matched_cyc.get("last_sell_price"), default=None)
                    cyc_avg = first_pos_float(matched_cyc.get("avg_cost_px"), matched_cyc.get("entry_price"))
                    cyc_qty = safe_nonneg_float(matched_cyc.get("qty"), default=0.0) or 0.0
                    cyc_ls_ts = matched_cyc.get("_last_sell_ts_ms")
                    cyc_lb_ts = matched_cyc.get("_entry_ts_ms")
                    if fills is None and (cyc_lb is not None or cyc_ls is not None or (cyc_avg is not None and cyc_avg > 0)):
                        lb_init: Optional[Dict[str, Any]] = None
                        if cyc_lb is not None:
                            lb_init = {
                                "price": cyc_lb,
                                "qty": matched_cyc.get("qty") if not matched_cyc.get("dca_buys") and cyc_qty > 1e-8 else None,
                                "actor": "BOT",
                                "actor_tag": "BOT",
                            }
                            if cyc_lb_ts is not None:
                                lb_init["ts_ms"] = cyc_lb_ts
                        fills = {
                            "venue": matched_cyc_venue,
                            "last_action_side": "BUY" if cyc_qty > 1e-8 else ("SELL" if cyc_ls is not None else "BUY"),
                            "last_action_actor": "BOT",
                            "last_action_actor_tag": "BOT",
                            "active_avg_buy_px": round(cyc_avg, 6) if (cyc_avg and cyc_qty > 1e-8) else None,
                            "last_buy": lb_init,
                            "last_sell": (
                                {"price": cyc_ls, "ts_ms": cyc_ls_ts, "actor": "BOT", "actor_tag": "BOT"}
                                if cyc_ls is not None
                                else None
                            ),
                            "recent_buys": [],
                        }
                    elif isinstance(fills, dict):
                        if is_equity_venue(matched_cyc_venue) and cyc_avg is not None and cyc_avg > 0 and cyc_qty > 1e-8:
                            if (
                                fills.get("tactical_avg_buy_px") is None
                                and (prev_act_avg := safe_pos_float(fills.get("active_avg_buy_px"), default=None)) is not None
                                and abs(prev_act_avg - cyc_avg) / max(cyc_avg, 1e-9) >= 0.002
                            ):
                                fills["tactical_avg_buy_px"] = round(prev_act_avg, 6)
                            fills["active_avg_buy_px"] = round(cyc_avg, 6)
                        elif fills.get("active_avg_buy_px") is None and cyc_avg is not None and cyc_avg > 0 and cyc_qty > 1e-8:
                            fills["active_avg_buy_px"] = round(cyc_avg, 6)
                        if fills.get("last_buy") is None and cyc_lb is not None:
                            lb_fb: Dict[str, Any] = {
                                "price": cyc_lb,
                                "qty": matched_cyc.get("qty") if not matched_cyc.get("dca_buys") and cyc_qty > 1e-8 else None,
                                "actor": "BOT",
                                "actor_tag": "BOT",
                            }
                            if cyc_lb_ts is not None:
                                lb_fb["ts_ms"] = cyc_lb_ts
                            fills["last_buy"] = lb_fb
                        elif isinstance(fills.get("last_buy"), dict) and fills["last_buy"].get("ts_ms") is None and cyc_lb_ts is not None:
                            fills["last_buy"]["ts_ms"] = cyc_lb_ts
                        if fills.get("last_sell") is None and cyc_ls is not None:
                            fills["last_sell"] = {"price": cyc_ls, "ts_ms": cyc_ls_ts, "actor": "BOT", "actor_tag": "BOT"}
        if isinstance(fills, dict) and isinstance(open_lots_raw := fills.get("open_lots"), list) and open_lots_raw:
            held_cur_f = -1.0
            if cachedb_dir and best_venue:
                vb_p_f = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, "venue_balances_state.json")))
                vb_d_f = _rj(vb_p_f) if os.path.exists(vb_p_f) else None
                if isinstance(vb_d_f, dict):
                    for cand_v in dict.fromkeys([sym_upper, *(str(c).strip().upper() for c in scoped_variants if str(c).strip())]):
                        if isinstance(p_d := vb_d_f.get(f"pos:{best_venue}:{cand_v}"), dict):
                            fb_v = safe_nonneg_float(
                                p_d.get("free_base") if p_d.get("free_base") is not None else p_d.get("free"),
                                default=None,
                            )
                            lb_v = first_pos_float(p_d.get("locked_base_qty"), p_d.get("locked"), default=0.0) or 0.0
                            if fb_v is not None:
                                held_cur_f = fb_v + lb_v
                                break
                    if held_cur_f < 0.0 and base_upper:
                        if (raw_b_v := safe_nonneg_float(vb_d_f.get(f"{best_venue}:{base_upper}"), default=None)) is not None:
                            held_cur_f = raw_b_v
            fresh_lots, _ = select_fresh_open_lots(open_lots_raw, held_qty=held_cur_f)
            if fresh_lots:
                _, fresh_q = reconcile_fresh_lots_vwap(fills, open_lots_raw, fresh_lots, held_cur_f)
                fills["open_lots"] = clip_open_lots_to_held_qty(
                    fresh_lots, held_cur_f, total_qty=fresh_q
                )
        return fills


class RecentOrdersAndPositionEnricherCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for fleet-wide recent fills and position cost-basis reconciliation."""

    cls_name = "RecentOrdersAndPositionEnricherCollector"
    COLLECTOR_ID = "portfolio_recent_orders"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        inst_by_sym: Optional[Dict[str, Any]] = None,
        tradable_positions: Optional[List[Dict[str, Any]]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        kraken_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        hyperliquid_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        t212_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        read_json_fn: Any = None,
        known_ai_order_ids: Optional[set[str]] = None,
        gateway_max_usd: float = 2000.0,
        ai_intents: Optional[List[Dict[str, Any]]] = None,
        allow_network: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cache_dir = cachedb_dir
        inst_by_sym = inst_by_sym or {}
        tradable_positions = tradable_positions if isinstance(tradable_positions, list) else []
        prices_and_trends, kraken_cycles, hyperliquid_cycles, t212_cycles = coerce_dict_maps(
            prices_and_trends, kraken_cycles, hyperliquid_cycles, t212_cycles
        )

        _rj = _resolve_read_json_cached(read_json_fn)
        ai_intents, known_ai_order_ids = _resolve_ai_intents_and_known_order_ids(
            cache_dir,
            ai_intents=ai_intents,
            known_ai_order_ids=known_ai_order_ids,
            read_json_fn=_rj,
            need_intents=True,
        )
        audit_origin_map = _load_recent_audit_origins(cache_dir)
        recent_orders: Dict[str, Any] = {}
        venue_recent_orders: Dict[str, Dict[str, Any]] = {v: {} for v in MarketDataProvider.canonical_venues()}

        def _venue_intents(v_name: str) -> Optional[List[Dict[str, Any]]]:
            return [it for it in ai_intents if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == v_name] if isinstance(ai_intents, list) else None

        def _record_venue_fills(c_venue: str, sym_k: str, fills_d: Dict[str, Any]) -> None:
            fills_d.setdefault("venue", c_venue)
            venue_recent_orders[c_venue][sym_k] = fills_d
            if sym_k in recent_orders and not is_primary_stream_venue(c_venue):
                recent_orders[f"{c_venue}:{sym_k}"] = fills_d
            else:
                recent_orders.setdefault(sym_k, fills_d)

        def _scoped_sym_variants(raw_sym: Any, *, is_eq: bool) -> list[str]:
            s_u = str(raw_sym or "").strip().upper()
            if not s_u:
                return []
            return [
                str(v_s).strip().upper()
                for v_s in filter_asset_class_variants(
                    s_u,
                    (s_u, *cache_symbol_variants(s_u, include_base=True)),
                    is_equity=is_eq,
                )
            ]

        active_t212_syms: set[str] = set()
        for s_k in (t212_cycles or {}):
            active_t212_syms.update(_scoped_sym_variants(s_k, is_eq=True))
        for s_k, iv in (inst_by_sym or {}).items():
            if is_equity_venue(getattr(iv, "provider_name", None)):
                raw_s = str(getattr(iv, "symbol", "") or str(s_k).split(":", 1)[-1]).strip().upper()
                active_t212_syms.update(_scoped_sym_variants(raw_s, is_eq=True))
        for pos_it in (tradable_positions or ()):
            if isinstance(pos_it, dict) and is_equity_venue(pos_it.get("venue")):
                raw_s = str(pos_it.get("symbol") or "").strip().upper()
                active_t212_syms.update(_scoped_sym_variants(raw_s, is_eq=True))
        allowed_t212_syms = set(active_t212_syms)
        if allowed_t212_syms:
            for eq_v in MarketDataProvider.canonical_venues():
                if is_equity_venue(eq_v):
                    for it_t212 in (_venue_intents(eq_v) or ()):
                        raw_s = str(it_t212.get("symbol") or "").strip().upper()
                        allowed_t212_syms.update(_scoped_sym_variants(raw_s, is_eq=True))

        audit_cache_by_venue: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for order_cache_name, cache_venue in _resolve_order_cache_files():
            orders_data = _rj(os.path.join(cache_dir, order_cache_name))
            if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
                v_ai_intents = _venue_intents(cache_venue)
                v_audit_map = (
                    audit_cache_by_venue.setdefault(cache_venue, _load_audit_terminal_orders(cache_dir, cache_venue))
                    if cache_dir and MarketDataProvider.merges_audit_terminal_orders(cache_venue)
                    else {}
                )
                for sym, o_list in orders_data["items"].items():
                    sym_u = str(sym or "").strip().upper()
                    if is_equity_venue(cache_venue) and allowed_t212_syms and sym_u not in allowed_t212_syms:
                        continue
                    merged_o_list = (
                        _merge_audit_orders(o_list, v_audit_map.get(sym_u) or v_audit_map.get(str(sym)) or [])
                        if v_audit_map and isinstance(o_list, list)
                        else o_list
                    )
                    if fills := extract_symbol_recent_fills(merged_o_list, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=v_ai_intents, symbol=sym):
                        _record_venue_fills(cache_venue, sym, fills)

        seen_live_order_vs: set[tuple[str, str]] = set()
        if allow_network:
            for inst in inst_by_sym.values():
                sym = str(getattr(inst, "symbol", "")).upper()
                v_inst = normalize_venue_name(getattr(inst, "provider_name", ""))
                if (
                    not sym
                    or not v_inst
                    or not MarketDataProvider.uses_bare_base_order_symbols(v_inst)
                    or (v_inst, sym) in seen_live_order_vs
                    or _first_variant_dict(venue_recent_orders.get(v_inst, {}), cache_symbol_variants(sym, include_base=True)) is not None
                ):
                    continue
                seen_live_order_vs.add((v_inst, sym))
                try:
                    prov = getattr(inst, "_provider", None)
                    if prov is not None and hasattr(prov, "get_orders"):
                        if fills := extract_symbol_recent_fills(prov.get_orders(sym, None, 90 * 86400.0), known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=_venue_intents(v_inst), symbol=sym):
                            _record_venue_fills(v_inst, sym, fills)
                except Exception:
                    pass
        for fallback_venue in MarketDataProvider.canonical_venues():
            fb_ai_intents = _venue_intents(fallback_venue)
            audit_by_sym = (
                audit_cache_by_venue.get(fallback_venue)
                if fallback_venue in audit_cache_by_venue
                else (
                    _load_audit_terminal_orders(cache_dir, fallback_venue)
                    if cache_dir and MarketDataProvider.merges_audit_terminal_orders(fallback_venue)
                    else {}
                )
            )
            intent_syms = {
                str(it.get("symbol") or "").strip().upper()
                for it in (fb_ai_intents or ())
                if isinstance(it, dict) and str(it.get("symbol") or "").strip()
            }
            for sym_fb in sorted(set(audit_by_sym.keys()) | intent_syms):
                if _first_variant_dict(venue_recent_orders[fallback_venue], cache_symbol_variants(sym_fb, include_base=True)) is not None:
                    continue
                o_list = audit_by_sym.get(sym_fb, [])
                if fills := extract_symbol_recent_fills(o_list, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=fb_ai_intents, symbol=sym_fb):
                    _record_venue_fills(fallback_venue, sym_fb, fills)
        live_held_by_venue_sym: Dict[tuple[str, str], float] = {}
        for pos_item, pv, ps in _iter_valid_tradable_positions(tradable_positions):
            fb = safe_float(pos_item.get("free_base"), -1.0)
            lb = safe_pos_float(pos_item.get("locked_base_qty"), default=0.0) or 0.0
            if fb >= 0.0 or lb > 0.0:
                held_q = (max(0.0, fb) + lb) if fb >= 0.0 else lb
                for sv in _scoped_sym_variants(ps, is_eq=is_equity_venue_or_suffix(ps, pv)):
                    sv_k = (pv, sv.upper())
                    if sv_k not in live_held_by_venue_sym or held_q > live_held_by_venue_sym[sv_k]:
                        live_held_by_venue_sym[sv_k] = held_q
        bcache_path_ro = str(resolve_cachedb_filepath(os.path.join(cache_dir, "venue_balances_state.json"))) if cache_dir else None
        raw_cb_ro = _rj(bcache_path_ro) if bcache_path_ro and os.path.exists(bcache_path_ro) else None
        cached_bals_ro: Dict[str, Any] = raw_cb_ro if isinstance(raw_cb_ro, dict) else {}
        for c_venue_ro, cyc_map_ro in (("kraken", kraken_cycles), ("hyperliquid", hyperliquid_cycles), ("t212", t212_cycles)):
            for t_sym, t_cyc in (cyc_map_ro or {}).items():
                t_sym_u = str(t_sym).strip().upper()
                if not t_sym_u or (c_venue_ro, t_sym_u) in live_held_by_venue_sym or not isinstance(t_cyc, dict):
                    continue
                t_base_u = canonical_base_asset(t_sym_u)
                ro_cands = _scoped_sym_variants(t_sym_u, is_eq=is_equity_venue_or_suffix(t_sym_u, c_venue_ro))
                c_pos = next(
                    (
                        v_pos
                        for cand_k in ro_cands
                        if isinstance((v_pos := cached_bals_ro.get(f"pos:{c_venue_ro}:{cand_k.upper()}")), dict)
                    ),
                    None,
                )
                fb_c = safe_nonneg_float(
                    c_pos.get("free_base")
                    if isinstance(c_pos, dict)
                    else (
                        cached_bals_ro.get(f"{c_venue_ro}:{t_sym_u}")
                        if cached_bals_ro.get(f"{c_venue_ro}:{t_sym_u}") is not None
                        else (cached_bals_ro.get(f"{c_venue_ro}:{t_base_u}") if t_base_u else None)
                    ),
                    default=None,
                )
                lb_c = safe_pos_float(c_pos.get("locked_base_qty") if isinstance(c_pos, dict) else None, default=0.0) or 0.0
                if fb_c is not None:
                    held_c = fb_c + lb_c
                    for sv_c in ro_cands:
                        live_held_by_venue_sym.setdefault((c_venue_ro, sv_c.upper()), held_c)
                    live_held_by_venue_sym[(c_venue_ro, t_sym_u)] = held_c
                    if is_equity_venue(c_venue_ro) and held_c > 1e-8:
                        t_cyc["qty"] = round(held_c, 6)
                        t_px = resolve_symbol_price(
                            ensure_equity_suffix(t_sym_u),
                            prices_and_trends,
                        )
                        if (t_avg := first_pos_float(t_cyc.get("avg_cost_px"), t_cyc.get("entry_price"))) is not None:
                            t_cyc["cost_usd"] = round(held_c * t_avg, 2)
                            if t_px > 0.0:
                                t_cyc["pnl_vs_avg_pct"] = t_cyc["dd_pct"] = round(((t_px - t_avg) / t_avg) * 100.0, 2) + 0.0
                        if t_px > 0.0 and (t_pk := safe_pos_float(t_cyc.get("pos_peak"), default=None)) is not None:
                            eff_t_pk = max(t_pk, t_px)
                            t_cyc["pos_peak"] = round(eff_t_pk, 6)
                            t_cyc["dd_from_peak_pct"] = round(((t_px - eff_t_pk) / eff_t_pk) * 100.0, 2) + 0.0
        cyc_sell_ts_map = reconcile_venue_cycles_with_recent_orders(
            kraken_cycles, hyperliquid_cycles, t212_cycles, venue_recent_orders, live_held_by_venue_sym, prices_and_trends
        )
        enrich_positions_with_orders_and_cycles(
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
        return recent_orders


class BotRuntimeStatesCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for order_retry_queue, monitortrades, assetguardian, and smart_stop_loss states."""

    cls_name = "BotRuntimeStatesCollector"
    COLLECTOR_ID = "portfolio_bot_runtime_states"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        now_ts: Any = None,
        read_json_fn: Any = None,
        tradable_positions: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cache_dir = cachedb_dir
        eff_now, _rj = _resolve_now_and_read_json(now_ts, read_json_fn)
        out: Dict[str, Any] = {}
        retry_q_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "order_retry_queue.jsonl")))
        if os.path.exists(retry_q_path):
            terminal_ai_intent_ids: set[str] = set()
            for it in _load_cached_ai_intents_list(cache_dir, _rj):
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
                exp_ts = normalize_epoch_sec(rec.get("expires_at_ts"), default=0.0, allow_zero=True, allow_iso=True) or 0.0
                checked_oq_expiry = False
                if _oq is not None and (rec.get("created_ts") is not None or rec.get("ttl_started_ts") is not None):
                    try:
                        checked_oq_expiry = True
                        if _oq.is_expired(rec, eff_now):
                            continue
                    except Exception:
                        checked_oq_expiry = False
                if (
                    lifecycle == "submit_pending"
                    and str(rec.get("submission_state") or "").lower() == "refused"
                    and bool(rec.get("safe_to_discard"))
                    and (safe_float(rec.get("attempts"), 0.0) or 0.0) >= 5.0
                    and str(rec.get("last_failure_reason") or "").lower()
                    in {"qty_zero_after_policy", "qty_zero_after_weight", "qty_zero_or_negative", "below_min_notional"}
                ):
                    continue
                if lifecycle in ("submit_pending", "awaiting_cancel", "pending", "active") and (
                    checked_oq_expiry or lifecycle == "awaiting_cancel" or exp_ts == 0.0 or exp_ts > eff_now
                ):
                    raw_rq_v = rec.get("provider_name") if rec.get("provider_name") is not None else rec.get("venue")
                    active_retries.append({
                        "venue": normalize_venue_name(raw_rq_v) or str(raw_rq_v or "").strip().lower(),
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
                if isinstance(pos, dict) and is_primary_stream_venue(pos.get("venue"))
                and (
                    (
                        0.0 <= safe_float(pos.get("value_usd"), -1.0)
                        and (safe_float(pos.get("value_usd"), 0.0) + max(0.0, safe_float(pos.get("locked_base_usd"), 0.0)))
                        < max(1.0, first_pos_float(pos.get("exchange_min_notional_usd"), pos.get("min_notional_usd"), default=5.0) or 5.0)
                    )
                    or (
                        0.0 <= safe_float(pos.get("free_base"), -1.0) <= 1e-8
                        and (safe_pos_float(pos.get("locked_base_qty"), default=0.0) or 0.0) <= 1e-8
                    )
                )
            }
            cleaned_ag_syms: Dict[str, Any] = {}
            for ag_sym, ag_val in ag_state["symbols"].items():
                if not isinstance(ag_val, dict):
                    continue
                sym_copy = dict(ag_val)
                if str(ag_sym).upper() in dust_binance_syms and "sell" in sym_copy:
                    if not isinstance(sell_st := sym_copy.get("sell"), dict) or not sell_st.get("pending"):
                        sym_copy.pop("sell", None)
                if isinstance(sell_d := sym_copy.get("sell"), dict):
                    sell_copy = dict(sell_d)
                    for qk in ("total_filled_qty", "initial_qty"):
                        if isinstance(sell_copy.get(qk), float):
                            sell_copy[qk] = round(sell_copy[qk], 8)
                    sym_copy["sell"] = sell_copy
                if sym_copy:
                    cleaned_ag_syms[ag_sym] = sym_copy
            if cleaned_ag_syms:
                out["assetguardian_state"] = cleaned_ag_syms

        if isinstance(ssl_state := _rj(os.path.join(cache_dir, "smart_stop_loss_state.json")), dict) and ssl_state:
            out["smart_stop_loss_state"] = ssl_state
        return out


for _port_col_cls in (
    VenueTrailingMapsCollector,
    VenueCyclesCollector,
    KrakenXstocksCollector,
    TradablePositionsAndOrdersCollector,
    SymbolBestRecentFillsCollector,
    RecentOrdersAndPositionEnricherCollector,
    BotRuntimeStatesCollector,
):
    TelemetryCollectorFactory.register(_port_col_cls.COLLECTOR_ID, _port_col_cls)


def harvest_venue_cycles(
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Dict[str, float],
    read_json_fn: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest normalized ``.state_*.json`` DCA cycle states for Kraken, Hyperliquid, and Trading212."""
    return TelemetryCollectorFactory.create("portfolio_venue_cycles").collect(
        workspace_dir,
        "*",
        workspace_dir=workspace_dir,
        prices_and_trends=prices_and_trends,
        fallback_prices=fallback_prices,
        read_json_fn=read_json_fn,
    )


def harvest_kraken_xstocks(
    kraken_dir: str,
    prices_and_trends: Dict[str, Any],
    all_t212_cycles: Dict[str, Any],
    read_json_fn: Any = None,
) -> Optional[Dict[str, Any]]:
    """Harvest Kraken read-only xStocks holdings and link to Trading212 sibling cycle."""
    return TelemetryCollectorFactory.create("portfolio_kraken_xstocks").collect(
        kraken_dir,
        "*",
        kraken_dir=kraken_dir,
        prices_and_trends=prices_and_trends,
        all_t212_cycles=all_t212_cycles,
        read_json_fn=read_json_fn,
    )


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
    allow_network: bool = True,
    persist: bool = True,
    force_persist: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Harvest spot wallet balances, trailing peaks, and open orders across tradable instruments."""
    return TelemetryCollectorFactory.create("portfolio_tradable_positions", start_sync=allow_network).collect(
        cache_dir or "",
        "*",
        inst_map=inst_map,
        prices_and_trends=prices_and_trends,
        trailing_map=trailing_map,
        trailing_by_venue=trailing_by_venue,
        known_ai_order_ids=known_ai_order_ids,
        gateway_max_usd=gateway_max_usd,
        verified_open_venues=verified_open_venues,
        allow_empty_inst_recovery=allow_empty_inst_recovery,
        allow_network=allow_network,
        persist=persist,
        force_persist=force_persist,
    )


def load_symbol_best_recent_fills(
    cachedb_dir: str,
    symbol: str,
    quote_variants: Sequence[str],
    read_json_fn: Any = None,
) -> Optional[Dict[str, Any]]:
    """Load the most recent venue fills dict for `symbol` across order caches and execution_audit fallback."""
    return TelemetryCollectorFactory.create("portfolio_symbol_recent_fills", start_sync=False).collect(
        cachedb_dir,
        symbol,
        quote_variants=quote_variants,
        read_json_fn=read_json_fn,
    )


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
    allow_network: bool = True,
) -> Dict[str, Any]:
    """Load recent fills across Binance, Kraken, Hyperliquid, and T212 and enrich tradable_positions with cost-basis anchors."""
    return TelemetryCollectorFactory.create("portfolio_recent_orders", start_sync=allow_network).collect(
        cache_dir,
        "*",
        inst_by_sym=inst_by_sym,
        tradable_positions=tradable_positions,
        prices_and_trends=prices_and_trends,
        kraken_cycles=kraken_cycles,
        hyperliquid_cycles=hyperliquid_cycles,
        t212_cycles=t212_cycles,
        read_json_fn=read_json_fn,
        known_ai_order_ids=known_ai_order_ids,
        gateway_max_usd=gateway_max_usd,
        ai_intents=ai_intents,
        allow_network=allow_network,
    )


def harvest_bot_runtime_states(
    cache_dir: str,
    now_ts: Any = None,
    read_json_fn: Any = None,
    tradable_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Harvest active retry queue, monitortrades, assetguardian, and smart_stop_loss states."""
    return TelemetryCollectorFactory.create("portfolio_bot_runtime_states").collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        read_json_fn=read_json_fn,
        tradable_positions=tradable_positions,
    )


__all__ = [
    "BotRuntimeStatesCollector",
    "KrakenXstocksCollector",
    "PortfolioCacheCollectorBase",
    "RecentOrdersAndPositionEnricherCollector",
    "SymbolBestRecentFillsCollector",
    "TradablePositionsAndOrdersCollector",
    "VenueCyclesCollector",
    "VenueTrailingMapsCollector",
    "_iter_audit_dir_file_batches",
    "_iter_t212_state_files",
    "_load_audit_terminal_orders",
    "_load_open_order_timestamps",
    "_load_recent_audit_origins",
    "_normalize_open_orders_list",
    "enrich_positions_with_orders_and_cycles",
    "extract_symbol_recent_fills",
    "harvest_bot_runtime_states",
    "harvest_kraken_xstocks",
    "harvest_recent_orders_and_enrich_positions",
    "harvest_tradable_positions_and_orders",
    "harvest_venue_cycles",
    "load_symbol_best_recent_fills",
    "parse_venue_cycle_state",
    "reconcile_venue_cycles_with_recent_orders",
]
