"""AI trade intent reconciliation against execution audit logs and cycle telemetry reporting."""
from __future__ import annotations

import glob
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from botcore import (
    _parse_bool_flag,
    _parse_epoch_or_iso_s,
    _ts_ms_to_s,
    clean_order_id,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    first_pos_float,
    normalize_timestamp_ms,
    safe_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    compatible_quote_suffixes,
    filter_candidates_by_asset_class,
    is_equity_venue_or_suffix,
    matches_asset_scope,
    normalize_venue_name,
    same_canonical_base_asset,
)
from instrument_registry import casefold_get
from .base import MarketDataProvider
from .execution_audit import (
    _AUDIT_DAY_RE,
    _OPEN_WORKING_STATUSES,
    _TERMINAL_CANCEL_STATUSES,
    _extract_audit_fill_metrics,
    _read_audit_intent_records,
)
from .intent_state import (
    _extract_intent_id,
    _intent_venue_symbol_side_key,
    _record_intent_state,
    extract_intent_order_type,
    extract_intent_side,
    intent_venue_client_order_ids,
)
from .order_id_context import (
    extract_client_order_id,
    resolve_order_actor,
)
from .order_ohlc_utils import (
    _UNFILLED_ORDER_STATUSES,
    extract_executed_qty,
    extract_orig_qty,
)


def _classify_audit_fill_state(
    ev: str,
    v_stat: str,
    l_stat: str,
) -> tuple[str, bool]:
    """Return ``(norm_venue_status, is_terminal)`` for an audit record where ``is_fill`` is True."""
    is_full_fill = v_stat in ("FILLED", "CLOSED") or l_stat in ("closed", "filled")
    is_cancel_or_expire = (
        v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")
        or l_stat in ("canceled", "cancelled", "expired", "rejected")
    )
    is_partial = (
        v_stat == "PARTIALLY_FILLED"
        or l_stat == "partially_filled"
        or (not is_full_fill and (ev in ("order_status", "order_terminal_retry") or is_cancel_or_expire))
    )
    norm_st = "PARTIALLY_FILLED" if is_partial else "FILLED"
    is_term = norm_st == "FILLED" or (ev == "order_terminal" and is_cancel_or_expire)
    return norm_st, is_term


def _is_terminal_fill_record(rec: dict[str, Any]) -> bool:
    v_stat, l_stat, *_, is_fill = _extract_audit_fill_metrics(rec)
    if not is_fill:
        return False
    _, is_term = _classify_audit_fill_state(str(rec.get("event") or ""), v_stat, l_stat)
    return is_term


def reconcile_intents_from_execution_audit(
    intents: list[dict[str, Any]],
    audit_dir: str,
) -> bool:
    """Reconcile FAILED/PENDING/PENDING_EXECUTION/SUBMIT_UNKNOWN intents and enrich incomplete EXECUTED intents from execution_audit_*.jsonl."""
    canceled_prior_oids: dict[str, float] = {}
    for item in intents:
        if isinstance(item, dict) and isinstance(item.get("canceled_prior_open_orders"), list):
            c_ts = safe_pos_float(item.get("timestamp"), default=0.0) or 0.0
            for raw_oid in item["canceled_prior_open_orders"]:
                if oid_s := clean_order_id(raw_oid):
                    canceled_prior_oids[oid_s] = max(canceled_prior_oids.get(oid_s, 0.0), c_ts)
    updated = False
    failed_by_id: dict[str, dict[str, Any]] = {}
    executed_incomplete_by_id: dict[str, dict[str, Any]] = {}
    for item in intents:
        if not isinstance(item, dict):
            continue
        clean_id = _extract_intent_id(item)
        if not clean_id:
            continue
        st_up = str(item.get("status") or "").strip().upper()
        if st_up in ("FAILED", "PENDING", "PENDING_EXECUTION", "SUBMIT_UNKNOWN"):
            failed_by_id[clean_id] = item
        elif st_up == "EXECUTED":
            exec_pay = item.get("execution_payload") if isinstance(item.get("execution_payload"), dict) else {}
            is_reconciled_term = bool(_parse_bool_flag(exec_pay.get("reconciled_terminal"), False))
            needs_ts_backfill = (
                is_reconciled_term
                and "filled_at" not in exec_pay
                and "terminal_ts" not in exec_pay
                and not _parse_bool_flag(exec_pay.get("_ts_backfill_checked"), False)
            )
            if not is_reconciled_term or needs_ts_backfill:
                cur_v_st = str(exec_pay.get("status") or "").strip().upper()
                item_oid = clean_order_id(extract_order_id(exec_pay, default=None))
                has_fill_metrics = (
                    extract_executed_qty(exec_pay, default=0.0) > 0.0
                    and extract_order_price(exec_pay, default=0.0) > 0.0
                )
                is_resting_msg = str(exec_pay.get("msg") or "").strip().lower() == "resting"
                if (
                    item_oid
                    and item_oid in canceled_prior_oids
                    and (
                        (
                            not needs_ts_backfill
                            and cur_v_st in _OPEN_WORKING_STATUSES
                            and extract_executed_qty(exec_pay, default=0.0) <= 0.0
                        )
                        or (needs_ts_backfill and cur_v_st in ("CANCELED", "CANCELLED"))
                    )
                ):
                    exec_pay = dict(exec_pay)
                    exec_pay["status"] = "CANCELED"
                    if is_resting_msg:
                        exec_pay.pop("msg", None)
                    c_ts_val = canceled_prior_oids[item_oid] or safe_pos_float(item.get("timestamp"), default=None)
                    if c_ts_val is not None:
                        exec_pay["terminal_ts"] = c_ts_val
                    exec_pay["reconciled_terminal"] = True
                    item["execution_payload"] = exec_pay
                    updated = True
                    continue
                if not needs_ts_backfill and cur_v_st in _TERMINAL_CANCEL_STATUSES:
                    continue
                if not needs_ts_backfill and has_fill_metrics and not is_resting_msg and cur_v_st in ("", "FILLED", "CLOSED"):
                    continue
                executed_incomplete_by_id[clean_id] = item
    if (not failed_by_id and not executed_incomplete_by_id) or not os.path.isdir(audit_dir):
        return updated

    # Audit files are UTC-day partitioned and never precede the intent that produced them: scan only days
    # from the oldest pending intent (minus one day of clock-skew margin), via the per-file record cache.
    pending_ts = [
        ts_i
        for it in (*failed_by_id.values(), *executed_incomplete_by_id.values())
        if (ts_i := safe_pos_float(it.get("timestamp"), default=0.0) or 0.0) > 0.0
    ]
    min_day = (
        datetime.fromtimestamp(min(pending_ts) - 86400.0, timezone.utc).strftime("%Y-%m-%d")
        if pending_ts
        else ""
    )
    matched_records: dict[str, dict[str, Any]] = {}
    matched_exec_records: dict[str, dict[str, Any]] = {}
    submitted_qty_by_id: dict[str, Any] = {}
    active_exec_oids: dict[str, Any] = {
        iid_e: extract_order_id(it_e.get("execution_payload"), default=None)
        for iid_e, it_e in executed_incomplete_by_id.items()
    }
    for audit_path in sorted(glob.glob(os.path.join(glob.escape(audit_dir), "execution_audit_*.jsonl"))):
        file_day = os.path.basename(audit_path)[len("execution_audit_"):-len(".jsonl")]
        if min_day and _AUDIT_DAY_RE.fullmatch(file_day) and file_day < min_day:
            continue
        for rec in _read_audit_intent_records(audit_path):
            iid = _extract_intent_id(rec)
            if not iid:
                continue
            ev = str(rec.get("event") or "")
            if ev in ("submit_accepted", "submit_unknown", "order_recovered") and rec.get("qty") is not None:
                submitted_qty_by_id[iid] = rec.get("qty")
            if iid in failed_by_id:
                prev = matched_records.get(iid)
                prev_is_fill = _extract_audit_fill_metrics(prev)[-1] if prev is not None else False
                if ev == "submit_unknown":
                    if prev is None or str(prev.get("event") or "") == "submit_unknown":
                        matched_records[iid] = rec
                elif ev in ("submit_accepted", "order_recovered"):
                    if not prev_is_fill:
                        matched_records[iid] = rec
                elif ev == "order_status":
                    *_, is_fill = _extract_audit_fill_metrics(rec)
                    if is_fill:
                        if prev is None or not _is_terminal_fill_record(prev):
                            matched_records[iid] = rec
                    elif prev is None or str(prev.get("event") or "") == "submit_unknown":
                        matched_records[iid] = rec
                elif ev in ("order_terminal", "order_terminal_retry"):
                    v_stat, l_stat, *_, is_fill = _extract_audit_fill_metrics(rec)
                    if is_fill:
                        if prev is None or not _is_terminal_fill_record(prev) or _is_terminal_fill_record(rec):
                            matched_records[iid] = rec
                    elif not prev_is_fill and (
                        v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")
                        or l_stat in ("canceled", "cancelled", "expired", "rejected")
                    ):
                        prev_oid = prev.get("order_id") if prev is not None else None
                        cur_oid = rec.get("order_id")
                        if not prev_oid or not cur_oid or str(prev_oid) == str(cur_oid):
                            matched_records[iid] = rec
            elif iid in executed_incomplete_by_id:
                rec_oid = extract_order_id(rec.get("order_id"), default=None)
                prev_exec = matched_exec_records.get(iid)
                prev_exec_is_fill = _extract_audit_fill_metrics(prev_exec)[-1] if prev_exec is not None else False
                if ev in ("submit_accepted", "order_recovered"):
                    if rec_oid is not None:
                        orig_oid = extract_order_id(
                            executed_incomplete_by_id[iid].get("execution_payload"), default=None
                        )
                        active_exec_oids[iid] = rec_oid
                        if not prev_exec_is_fill and (orig_oid is None or str(orig_oid) != str(rec_oid)):
                            matched_exec_records[iid] = rec
                elif ev in ("order_status", "order_terminal", "order_terminal_retry"):
                    cur_oid = active_exec_oids.get(iid)
                    if cur_oid is not None and rec_oid is not None and str(cur_oid) != str(rec_oid):
                        continue
                    v_stat, l_stat, *_, is_fill = _extract_audit_fill_metrics(rec)
                    if is_fill:
                        if prev_exec is None or not _is_terminal_fill_record(prev_exec) or _is_terminal_fill_record(rec):
                            matched_exec_records[iid] = rec
                    elif (
                        ev in ("order_terminal", "order_terminal_retry")
                        and not prev_exec_is_fill
                        and (
                            v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")
                            or l_stat in ("canceled", "cancelled", "expired", "rejected")
                        )
                    ):
                        matched_exec_records[iid] = rec

    for iid_chk, item_chk in executed_incomplete_by_id.items():
        if iid_chk not in matched_exec_records:
            ep_chk = item_chk.get("execution_payload")
            if isinstance(ep_chk, dict) and _parse_bool_flag(ep_chk.get("reconciled_terminal"), False):
                ep_chk["_ts_backfill_checked"] = True

    if not matched_records and not matched_exec_records:
        return updated

    for iid, rec in matched_records.items():
        intent = failed_by_id[iid]
        st_up = str(intent.get("status") or "").strip().upper()
        ev = str(rec.get("event") or "")
        v_stat, l_stat, filled_qty, _, raw_cost, _, f_px, is_fill = _extract_audit_fill_metrics(
            rec, fallback_ref_price=True
        )
        rec_ts_f = safe_pos_float(rec.get("ts"), default=None)
        eff_ts = rec_ts_f if rec_ts_f is not None else time.time()
        orig_q = rec.get("qty") if rec.get("qty") is not None else submitted_qty_by_id.get(iid)
        side_u = str(rec.get("side") or intent.get("decision") or "").upper()
        sym_val = rec.get("symbol") or intent.get("symbol")

        if ev == "submit_unknown":
            if st_up == "SUBMIT_UNKNOWN":
                continue
            exec_payload: dict[str, Any] = {
                "symbol": sym_val,
                "side": side_u,
                "executedQty": "",
                "cummulativeQuoteQty": "",
                "status": "UNKNOWN",
                "submission_state": "unknown",
                "reconciled_from_audit": True,
            }
            if orig_q is not None:
                exec_payload["origQty"] = str(orig_q)
            if rec.get("client_order_id"):
                exec_payload["clientOrderId"] = rec.get("client_order_id")
            if reason_unk := (rec.get("reason") or rec.get("failure_reason")):
                exec_payload["reason"] = reason_unk
            intent["execution_payload"] = exec_payload
            _record_intent_state(
                intent,
                "SUBMIT_UNKNOWN",
                eff_ts,
                action="RECONCILED_UNKNOWN",
                payload=exec_payload,
            )
            updated = True
            continue

        is_cancel_terminal = not is_fill and ev in ("order_terminal", "order_terminal_retry") and (
            v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")
            or l_stat in ("canceled", "cancelled", "expired", "rejected")
        )
        if is_cancel_terminal:
            if st_up == "FAILED":
                continue
            raw_v_status = v_stat if v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED") else l_stat.upper()
            target_state, action_lbl, reason_val = "FAILED", "RECONCILED_TERMINAL", raw_v_status
            is_term = ev == "order_terminal"
            exec_qty_s, quote_qty_s = "0", ""
        elif is_fill:
            raw_v_status, is_term = _classify_audit_fill_state(ev, v_stat, l_stat)
            target_state, reason_val = "EXECUTED", None
            action_lbl = "RECONCILED_PARTIAL" if raw_v_status == "PARTIALLY_FILLED" else "RECONCILED_EXECUTED"
            exec_qty_s = str(filled_qty) if filled_qty is not None else ""
            quote_qty_s = str(raw_cost) if raw_cost is not None else ""
        else:
            raw_v_status = v_stat if v_stat in ("NEW", "OPEN", "RESTING", "PENDING", "WORKING") else "NEW"
            target_state, action_lbl, reason_val = "EXECUTED", "RECONCILED_ACCEPTED", None
            is_term = False
            exec_qty_s = quote_qty_s = ""

        exec_payload = {
            "orderId": rec.get("order_id"),
            "symbol": sym_val,
            "side": side_u,
            "executedQty": exec_qty_s,
            "cummulativeQuoteQty": quote_qty_s,
            "status": raw_v_status,
            "reconciled_from_audit": True,
        }
        if orig_q is not None:
            exec_payload["origQty"] = str(orig_q)
        if not is_cancel_terminal and f_px is not None:
            exec_payload["price"] = f_px
        if rec_ts_f is not None:
            if not is_cancel_terminal:
                exec_payload["filled_at" if is_fill else "accepted_at"] = rec_ts_f
            if is_term:
                exec_payload["terminal_ts"] = rec_ts_f
        if is_term:
            exec_payload["reconciled_terminal"] = True

        intent["execution_payload"] = exec_payload
        _record_intent_state(
            intent,
            target_state,
            eff_ts,
            action=action_lbl,
            **({"reason": reason_val} if reason_val is not None else {}),
            order_id=rec.get("order_id"),
            payload=exec_payload,
        )
        updated = True

    for iid, rec in matched_exec_records.items():
        intent = executed_incomplete_by_id[iid]
        exec_pay = dict(intent.get("execution_payload")) if isinstance(intent.get("execution_payload"), dict) else {}
        ev = str(rec.get("event") or "")
        v_stat, l_stat, filled_qty, f_q, raw_cost, cost_f, f_px, is_fill = _extract_audit_fill_metrics(rec)
        rec_ts = safe_pos_float(rec.get("ts"), default=None)
        cur_st = str(exec_pay.get("status") or "").strip().upper()
        if is_fill:
            norm_st, is_term = _classify_audit_fill_state(ev, v_stat, l_stat)
            if rec.get("order_id") is not None:
                exec_pay["orderId"] = rec.get("order_id")
            exec_pay["status"] = norm_st
            if f_q is not None and f_q > 0:
                exec_pay["executedQty"] = str(filled_qty)
                if "filledQuantity" in exec_pay:
                    exec_pay["filledQuantity"] = f_q
            if raw_cost is not None and cost_f is not None and cost_f > 0:
                exec_pay["cummulativeQuoteQty"] = str(raw_cost)
                exec_pay["cost"] = raw_cost
            if f_px is not None and f_px > 0:
                exec_pay["price"] = f_px
            if rec_ts is not None:
                exec_pay["filled_at"] = rec_ts
                if is_term:
                    exec_pay["terminal_ts"] = rec_ts
            if is_term:
                exec_pay["reconciled_terminal"] = True
        elif ev in ("submit_accepted", "order_recovered"):
            if rec.get("order_id") is not None:
                exec_pay["orderId"] = rec.get("order_id")
            exec_pay["status"] = v_stat if v_stat in ("NEW", "OPEN", "RESTING", "PENDING", "WORKING") else "NEW"
            if rec_ts is not None:
                exec_pay["accepted_at"] = rec_ts
            exec_pay.pop("terminal_ts", None)
            exec_pay.pop("reconciled_terminal", None)
        elif cur_st in _OPEN_WORKING_STATUSES or (
            _parse_bool_flag(exec_pay.get("reconciled_terminal"), False)
            and "terminal_ts" not in exec_pay
            and rec_ts is not None
        ):
            norm_cancel = v_stat if v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED") else "CANCELED"
            exec_pay["status"] = norm_cancel
            if ev == "order_terminal":
                if rec_ts is not None:
                    exec_pay["terminal_ts"] = rec_ts
                exec_pay["reconciled_terminal"] = True
        else:
            continue
        if str(exec_pay.get("msg") or "").strip().lower() == "resting":
            exec_pay.pop("msg", None)
        if exec_pay != intent.get("execution_payload"):
            intent["execution_payload"] = exec_pay
            updated = True

    return updated


def _filter_intents_by_scope(intents_data: Any, scope: str) -> list[dict[str, Any]]:
    """Filter intent records to those matching the requested asset scope ('crypto', 'equities', or 'all')."""
    if isinstance(intents_data, dict):
        intents_data = intents_data.get("intents")
    if not isinstance(intents_data, list):
        return []
    return [
        i
        for i in intents_data
        if isinstance(i, dict) and matches_asset_scope(i.get("symbol"), i.get("venue"), scope)
    ]


def _resolve_intent_execution_details(
    i: dict[str, Any],
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Resolve fill price, filled quantity, execution status, and slippage (bps) for an intent."""
    venue = normalize_venue_name(i.get("venue")) or i.get("venue")
    symbol = str(i.get("symbol") or "").strip().upper()
    side = extract_intent_side(i) or str(i.get("decision") or "").strip().upper()
    order_type = extract_intent_order_type(i)
    exec_pay = i.get("execution_payload") if isinstance(i.get("execution_payload"), dict) else {}
    hist = i.get("history") if isinstance(i.get("history"), list) else []
    last_hist = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    exec_quote_notional = (
        safe_pos_float(exec_pay.get("cummulativeQuoteQty"), default=None)
        or safe_pos_float(exec_pay.get("cost"), default=None)
        or safe_pos_float(exec_pay.get("filledValue"), default=None)
    )
    intent_order_notional = (
        safe_pos_float(i.get("clamped_notional_usd"), default=None)
        or safe_pos_float(last_hist.get("notional_usd"), default=None)
        or extract_notional_fn(i, default=None)
    )
    eff_notional = exec_quote_notional or intent_order_notional
    ts_raw: Any = None
    ts_val = 0.0
    for _tk in ("timestamp", "executed_at", "executed_ts", "created_at", "ts"):
        _cand_raw = i.get(_tk)
        if _cand_raw is None:
            continue
        if ts_raw is None:
            ts_raw = _cand_raw
        _parsed_s = _parse_epoch_or_iso_s(_cand_raw)
        if _parsed_s is not None and _parsed_s > 0.0:
            ts_raw = _cand_raw
            ts_val = _ts_ms_to_s(_parsed_s)
            break
    order_id = extract_order_id(exec_pay, default=None) or extract_order_id(
        last_hist.get("order_id") or last_hist.get("payload"), default=None
    )

    raw_intent_st = str(i.get("status") or "").strip().upper()
    is_eq_intent = is_equity_venue_or_suffix(symbol, venue)
    matched_open = None
    expected_cids: Optional[set[str]] = None
    if isinstance(open_orders, list) and raw_intent_st in (
        "EXECUTED",
        "FAILED",
        "PENDING",
        "PENDING_EXECUTION",
        "SUBMIT_UNKNOWN",
    ):
        for oo in open_orders:
            if not isinstance(oo, dict):
                continue
            oo_venue = normalize_venue_name(oo.get("venue"))
            if oo_venue and venue and oo_venue != venue:
                continue
            oo_sym = str(oo.get("symbol") or "").strip().upper()
            oo_is_eq = is_equity_venue_or_suffix(oo_sym, oo_venue)
            if (oo_venue or oo_sym) and (venue or symbol) and oo_is_eq != is_eq_intent:
                continue
            if oo_sym and symbol and (
                not same_canonical_base_asset(oo_sym, symbol)
                or compatible_quote_suffixes(oo_sym) != compatible_quote_suffixes(symbol)
            ):
                continue
            oo_oid = extract_order_id(oo, default=None)
            if order_id is not None and oo_oid is not None and str(oo_oid) == str(order_id):
                matched_open = oo
                break
            if (
                order_id is None
                and order_type == "LIMIT"
                and oo_venue == venue
                and oo_sym
                and (extract_order_side(oo) or str(oo.get("side") or "").strip().upper()) == side
            ):
                # Response-lost intent: adopt only the order carrying this intent's own client ID
                # (venue-native form), never a BOT / monitororder-repriced order or another AI intent's order.
                if expected_cids is None:
                    expected_cids = intent_venue_client_order_ids(i)
                oo_cid = (extract_client_order_id(oo, default="") or "").lower()
                if (oo_cid in expected_cids) if oo_cid else (resolve_order_actor(oo) == "AI"):
                    matched_open = oo
                    break

    fill_px = extract_order_price(exec_pay, default=None)
    filled_qty = extract_executed_qty(exec_pay, default=0.0)
    orig_qty = extract_orig_qty(exec_pay, default=None) or safe_pos_float(last_hist.get("qty"), default=None)
    if matched_open is not None:
        open_filled = extract_executed_qty(matched_open, default=0.0)
        if open_filled > filled_qty:
            filled_qty = open_filled
        if fill_px is None:
            fill_px = safe_pos_float(matched_open.get("price"), default=None)

    venue_st = str(
        exec_pay.get("status")
        or ("RESTING" if str(exec_pay.get("msg") or "").strip().lower() == "resting" else "")
        or (
            "OPEN"
            if (
                order_type == "LIMIT"
                and filled_qty <= 0
                and isinstance(exec_pay, dict)
                and any(k in exec_pay for k in ("txid", "ordertxid", "descr", "orderId", "order_id", "oid"))
            )
            else ""
        )
    ).strip().upper()
    if matched_open is not None and not venue_st:
        venue_st = str(matched_open.get("status") or "OPEN").strip().upper()
    if (
        matched_open is None
        and raw_intent_st == "EXECUTED"
        and venue_st not in _TERMINAL_CANCEL_STATUSES
        and isinstance(recent_orders, dict)
        and symbol
    ):
        raw_cands = cache_symbol_variants(symbol, quotes=compatible_quote_suffixes(symbol))
        sym_cands = filter_candidates_by_asset_class(raw_cands, is_equity=is_eq_intent)
        ro_entry = next(
            (
                val
                for key_cand in (*(f"{venue}:{cand}" for cand in sym_cands), *sym_cands)
                if isinstance((val := casefold_get(recent_orders, key_cand)), dict)
                and (
                    ":" in key_cand
                    or not venue
                    or not val.get("venue")
                    or normalize_venue_name(val.get("venue")) == venue
                )
            ),
            None,
        )
        if isinstance(ro_entry, dict):
            side_key = "buy" if side == "BUY" else ("sell" if side == "SELL" else "")
            # Scan newest-first across last_<side> and recent_<side>s: a later BOT fill on the same side
            # (e.g. a monitororder-repriced tradeall order) must not hide this intent's own AI fill.
            fill_cands: list[dict[str, Any]] = []
            if side_key:
                last_c = ro_entry.get(f"last_{side_key}")
                recent_c = ro_entry.get(f"recent_{side_key}s")
                for cand_f in ([last_c] if isinstance(last_c, dict) else []) + (
                    list(reversed(recent_c)) if isinstance(recent_c, list) else []
                ):
                    if isinstance(cand_f, dict) and cand_f not in fill_cands:
                        fill_cands.append(cand_f)
            min_fill_ts_ms = ts_val * 1000.0 if order_type == "LIMIT" else (ts_val - 120.0) * 1000.0
            last_fill = None
            for cand_f in fill_cands:
                f_oid = clean_order_id(cand_f.get("order_id"))
                if order_id is not None and f_oid:
                    if f_oid == str(order_id):
                        last_fill = cand_f
                        break
                    continue
                f_ts_ms = normalize_timestamp_ms(cand_f.get("ts_ms"), default=0) or 0
                f_actor = str(cand_f.get("actor") or "").strip().upper()
                if f_actor != "BOT" and (ts_val <= 0 or f_ts_ms <= 0 or f_ts_ms >= min_fill_ts_ms):
                    last_fill = cand_f
                    break
            if isinstance(last_fill, dict):
                if fill_px is None or (venue_st in _UNFILLED_ORDER_STATUSES and filled_qty <= 0):
                    fill_px = safe_pos_float(last_fill.get("price"), default=fill_px)
                if filled_qty <= 0:
                    filled_qty = safe_pos_float(last_fill.get("qty"), default=0.0) or 0.0
                matched_fill_ts_ms = normalize_timestamp_ms(last_fill.get("ts_ms"), default=0) or 0
            else:
                matched_fill_ts_ms = 0
        else:
            matched_fill_ts_ms = 0
    else:
        matched_fill_ts_ms = 0
    is_unfilled_resting_limit = (
        order_type == "LIMIT"
        and (venue_st in _UNFILLED_ORDER_STATUSES or not venue_st)
        and filled_qty <= 0
    )
    is_unfilled_accepted_market = (
        order_type != "LIMIT"
        and _parse_bool_flag(exec_pay.get("reconciled_from_audit"), False)
        and (venue_st in _UNFILLED_ORDER_STATUSES or not venue_st)
        and filled_qty <= 0
    )
    venue_open_verified = (
        verified_open_venues is None
        or (bool(venue) and normalize_venue_name(venue) in verified_open_venues)
    )
    if (
        is_unfilled_resting_limit
        and matched_open is None
        and isinstance(open_orders, list)
        and venue_open_verified
        and ts_val > 0
    ):
        cancel_grace_sec = MarketDataProvider.resolve_unfilled_order_cancel_grace_sec()
        if (now_ts - ts_val) > cancel_grace_sec:
            venue_st = "CANCELED_OR_EXPIRED"
    if (
        fill_px is None
        and raw_intent_st in ("EXECUTED", "OBSERVED_SHADOW")
        and not is_unfilled_resting_limit
        and not is_unfilled_accepted_market
    ):
        fill_px = first_pos_float(
            last_hist.get("price"),
            i.get("requested_price"),
            i.get("limit_price"),
            i.get("reference_price"),
        )

    if matched_open is not None:
        exec_status = "PARTIALLY_FILLED" if filled_qty > 0 else "OPEN_RESTING"
        if exec_status == "PARTIALLY_FILLED" and venue_st in _UNFILLED_ORDER_STATUSES:
            venue_st = "PARTIALLY_FILLED"
    elif raw_intent_st == "EXECUTED":
        if venue_st in _TERMINAL_CANCEL_STATUSES:
            exec_status = venue_st
        elif is_unfilled_resting_limit:
            exec_status = "OPEN_RESTING"
        elif is_unfilled_accepted_market:
            exec_status = "ACCEPTED"
        elif venue_st == "PARTIALLY_FILLED":
            exec_status = "PARTIALLY_FILLED"
        else:
            exec_status = "FILLED"
            if venue_st in _UNFILLED_ORDER_STATUSES:
                venue_st = "FILLED"
    elif raw_intent_st == "OBSERVED_SHADOW":
        exec_status = "SHADOW_SIMULATED"
    else:
        exec_status = raw_intent_st or "UNKNOWN"

    ref_px = first_pos_float(
        i.get("reference_price"),
        i.get("limit_price"),
        i.get("requested_price"),
        last_hist.get("price"),
        fill_px,
    )
    if exec_status in ("OPEN_RESTING", "ACCEPTED", *_TERMINAL_CANCEL_STATUSES) and filled_qty <= 0:
        fill_px = None
    slippage_bps: Optional[float] = None
    if (
        exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED")
        and fill_px is not None
        and fill_px > 0
        and ref_px is not None
        and ref_px > 0
    ):
        raw_slip = (
            ((fill_px - ref_px) / ref_px) * 10_000.0
            if side == "BUY"
            else ((ref_px - fill_px) / ref_px) * 10_000.0
        )
        slippage_bps = round(raw_slip, 2)
    elif exec_status in ("FILLED", "SHADOW_SIMULATED") and fill_px is not None:
        slippage_bps = 0.0

    eff_qty = (
        filled_qty
        if filled_qty > 0
        else (orig_qty if (orig_qty is not None and exec_status in ("FILLED", "SHADOW_SIMULATED")) else None)
    )
    limit_px = (
        first_pos_float(
            i.get("limit_price"),
            matched_open.get("price") if isinstance(matched_open, dict) else None,
        )
        if order_type == "LIMIT"
        else None
    )
    fill_ts_val = ts_val
    is_fill_exec = exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED") or (
        eff_qty is not None and eff_qty > 0
    )
    if is_fill_exec and isinstance(exec_pay, dict):
        for _ftk in ("filled_at", "filled_ts", "terminal_ts", "updated_ts", "updateTime", "workingTime", "transactTime"):
            _fcand = exec_pay.get(_ftk)
            if _fcand is None:
                continue
            _fparsed = _parse_epoch_or_iso_s(_fcand)
            if _fparsed is not None and _fparsed > 0.0:
                _f_s = _ts_ms_to_s(_fparsed)
                if _f_s >= ts_val:
                    fill_ts_val = max(fill_ts_val, _f_s)
                    break
    if is_fill_exec and matched_fill_ts_ms > 0:
        _mf_s = _ts_ms_to_s(safe_pos_float(matched_fill_ts_ms, default=0.0) or 0.0)
        if _mf_s > fill_ts_val:
            fill_ts_val = _mf_s
    if is_fill_exec and fill_ts_val <= ts_val and str(last_hist.get("action") or "") in (
        "RECONCILED_EXECUTED",
        "RECONCILED_PARTIAL",
    ):
        _lh_parsed = _parse_epoch_or_iso_s(last_hist.get("ts"))
        if _lh_parsed is not None and _lh_parsed > 0.0:
            _lh_s = _ts_ms_to_s(_lh_parsed)
            if _lh_s > fill_ts_val:
                fill_ts_val = _lh_s
    fail_reason: Optional[str] = None
    if exec_status not in ("FILLED", "PARTIALLY_FILLED", "OPEN_RESTING", "ACCEPTED", "SHADOW_SIMULATED"):
        raw_fail_reason = (
            last_hist.get("reason")
            or last_hist.get("guard_reason")
            or last_hist.get("error")
            or i.get("guard_reason")
            or i.get("error")
            or (exec_pay.get("reason") or exec_pay.get("error") if isinstance(exec_pay, dict) else None)
        )
        if raw_fail_reason:
            fail_reason = " ".join(str(raw_fail_reason).split())[:200] or None
    filled_notional_usd: Optional[float] = None
    if exec_status == "PARTIALLY_FILLED":
        raw_filled_notional = exec_quote_notional or (
            (eff_qty * fill_px)
            if (eff_qty is not None and eff_qty > 0 and fill_px is not None and fill_px > 0)
            else None
        )
        if raw_filled_notional is not None and raw_filled_notional > 0:
            filled_notional_usd = round(raw_filled_notional, 2)
        orig_order_px = limit_px or fill_px or ref_px
        eff_notional = (
            intent_order_notional
            or ((orig_qty * orig_order_px) if (orig_qty is not None and orig_order_px is not None) else None)
            or exec_quote_notional
        )
    return {
        "venue": venue,
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "raw_status": i.get("status"),
        "venue_status": venue_st or None,
        "exec_status": exec_status,
        "is_open_in_venue": matched_open is not None,
        "order_id": str(order_id) if order_id is not None else None,
        "fill_px": round(fill_px, 6) if fill_px is not None else None,
        "limit_px": round(limit_px, 6) if limit_px is not None else None,
        "ref_px": round(ref_px, 6) if ref_px is not None else None,
        "filled_qty": round(eff_qty, 8) if eff_qty is not None and eff_qty > 0 else None,
        "filled_notional_usd": filled_notional_usd,
        "notional_usd": round(eff_notional, 2) if eff_notional is not None else i.get("suggested_notional_usd"),
        "slippage_bps": slippage_bps,
        "reason": fail_reason,
        "ts": ts_raw,
        "ts_val": ts_val,
        "fill_ts_val": fill_ts_val,
    }


def _is_open_resting_detail(det: dict[str, Any]) -> bool:
    return bool(
        _parse_bool_flag(det.get("is_open_in_venue"), False)
        and det.get("exec_status") in ("OPEN_RESTING", "PARTIALLY_FILLED")
    )


def _attach_intent_age_fields(
    target: dict[str, Any],
    det: dict[str, Any],
    now_ts: float,
    *,
    unit_sec: float,
    ndigits: int,
    suffix: str,
) -> None:
    ts_val = det["ts_val"]
    eff_ts = safe_pos_float(det.get("fill_ts_val"), default=None) or ts_val
    if eff_ts > 0 and now_ts >= eff_ts:
        target[f"age_{suffix}"] = round((now_ts - eff_ts) / unit_sec, ndigits)
    if ts_val > 0 and eff_ts > ts_val + 60.0 and now_ts >= ts_val:
        target[f"placed_age_{suffix}"] = round((now_ts - ts_val) / unit_sec, ndigits)


def _resolve_executed_open_resting_detail(
    intent_obj: dict[str, Any],
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> Optional[dict[str, Any]]:
    if str(intent_obj.get("status") or "").strip().upper() != "EXECUTED":
        return None
    det = _resolve_intent_execution_details(
        intent_obj,
        now_ts,
        extract_notional_fn,
        recent_orders=recent_orders,
        open_orders=open_orders,
        verified_open_venues=verified_open_venues,
    )
    return det if _is_open_resting_detail(det) else None


def summarize_recent_ai_intents(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Summarize the most recent AI trade intents filtered by asset scope."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)

    def _to_summary(i: dict[str, Any], det: dict[str, Any]) -> dict[str, Any]:
        intent_summary: dict[str, Any] = {
            "decision": i.get("decision") or det["side"],
            "symbol": i.get("symbol"),
            "venue": det["venue"],
            "order_type": det["order_type"],
            "status": det["raw_status"],
            "notional_usd": det["notional_usd"],
            "ts": det["ts"],
        }
        if i.get("limit_price") is not None:
            intent_summary["limit_price"] = i.get("limit_price")
        if det["venue_status"]:
            intent_summary["venue_status"] = det["venue_status"]
        for k in ("order_id", "fill_px", "filled_qty", "filled_notional_usd", "slippage_bps"):
            if det.get(k) is not None:
                intent_summary[k] = det[k]
        if det.get("reason"):
            intent_summary["reason"] = det["reason"]
        _attach_intent_age_fields(intent_summary, det, now_ts, unit_sec=3600.0, ndigits=2, suffix="h")
        return intent_summary

    recent_rev: list[dict[str, Any]] = []
    non_resting_count = 0
    seen_oids: set[str] = set()
    seen_sym_side: set[tuple[str, str, tuple[str, ...], str]] = set()
    recent_indices: set[int] = set()
    for idx in range(len(scoped_intents) - 1, -1, -1):
        if non_resting_count >= 8 or len(recent_rev) >= 16:
            break
        i = scoped_intents[idx]
        det = _resolve_intent_execution_details(
            i,
            now_ts,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        sym_side_key = _intent_venue_symbol_side_key(det.get("venue"), det.get("symbol"), det.get("side"))
        if (
            det.get("exec_status")
            in ("CANCELED", "CANCELLED", "EXPIRED", "CANCELED_OR_EXPIRED", "SKIPPED_DUPLICATE_OPEN_ORDER")
            and (safe_pos_float(det.get("filled_qty"), default=0.0) or 0.0) <= 0.0
            and sym_side_key in seen_sym_side
        ):
            continue
        seen_sym_side.add(sym_side_key)
        recent_indices.add(idx)
        if det["order_id"] is not None:
            seen_oids.add(str(det["order_id"]))
        sum_item = _to_summary(i, det)
        recent_rev.append(sum_item)
        if not (
            isinstance(open_orders, list)
            and open_orders
            and _is_open_resting_detail(det)
            and (sum_item.get("age_h") or 0.0) > 0.5
        ):
            non_resting_count += 1
    recent_intents_out: list[dict[str, Any]] = list(reversed(recent_rev))

    if isinstance(open_orders, list) and open_orders and len(scoped_intents) > len(recent_intents_out):
        older_resting: list[dict[str, Any]] = []
        for idx, i in enumerate(scoped_intents):
            if idx in recent_indices:
                continue
            det = _resolve_executed_open_resting_detail(
                i,
                now_ts,
                extract_notional_fn,
                recent_orders=recent_orders,
                open_orders=open_orders,
                verified_open_venues=verified_open_venues,
            )
            if det is not None and (det["order_id"] is None or str(det["order_id"]) not in seen_oids):
                if det["order_id"] is not None:
                    seen_oids.add(str(det["order_id"]))
                older_resting.append(_to_summary(i, det))
        if older_resting:
            recent_intents_out = older_resting + recent_intents_out
    return recent_intents_out


def build_last_cycle_execution_status(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> Optional[Any]:
    """Build closed-loop mathematical fill confirmation (fill_px, slippage_bps, status) for the last cycle's intent(s)."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)
    if not scoped_intents:
        return None

    def _eff_intent_ts(intent_obj: dict[str, Any]) -> float:
        base_t = safe_float(intent_obj.get("timestamp"), 0.0) or 0.0
        ep = intent_obj.get("execution_payload")
        if isinstance(ep, dict):
            ep_st = str(ep.get("status") or "").strip().upper()
            has_fill = (
                ep_st in ("FILLED", "PARTIALLY_FILLED", "CLOSED")
                or extract_executed_qty(ep, default=0.0) > 0.0
                or ep.get("filled_at") is not None
                or ep.get("filled_ts") is not None
            )
            if has_fill and ep_st not in _TERMINAL_CANCEL_STATUSES:
                for k in ("filled_at", "filled_ts", "terminal_ts", "updated_ts"):
                    ft = _parse_epoch_or_iso_s(ep.get(k))
                    if ft is not None and ft > 0:
                        base_t = max(base_t, _ts_ms_to_s(ft))
        return base_t

    last_ts = max((_eff_intent_ts(i) for i in scoped_intents), default=0.0)
    if last_ts > 0:
        cycle_intents = [
            i for i in scoped_intents if abs(_eff_intent_ts(i) - last_ts) <= 90.0
        ]
    else:
        cycle_intents = [scoped_intents[-1]]

    def _build_status_rec(det: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "venue": det["venue"],
            "symbol": det["symbol"],
            "side": det["side"],
            "order_type": det["order_type"],
            "status": det["exec_status"],
            "fill_px": det["fill_px"],
            "ref_px": det["ref_px"],
            "filled_qty": det["filled_qty"],
            "notional_usd": det["notional_usd"],
            "slippage_bps": det["slippage_bps"],
        }
        for k in ("filled_notional_usd", "limit_px", "order_id"):
            if det.get(k) is not None:
                rec[k] = det[k]
        if det.get("reason"):
            rec["reason"] = det["reason"]
        _attach_intent_age_fields(rec, det, now_ts, unit_sec=60.0, ndigits=1, suffix="min")
        return rec

    out: list[dict[str, Any]] = []
    seen_oids: set[str] = set()
    seen_sym_keys: set[tuple[str, str, tuple[str, ...], str]] = set()
    cycle_ids = {id(i) for i in cycle_intents}
    for i in cycle_intents:
        det = _resolve_intent_execution_details(
            i,
            now_ts,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        eff_ts_det = safe_pos_float(det.get("fill_ts_val"), default=None) or det["ts_val"]
        if (
            not _is_open_resting_detail(det)
            and det.get("exec_status") not in ("OPEN_RESTING", "PARTIALLY_FILLED")
            and eff_ts_det > 0
            and now_ts > eff_ts_det
            and (now_ts - eff_ts_det) > 3600.0
        ):
            continue
        if det["order_id"] is not None:
            seen_oids.add(str(det["order_id"]))
        if _is_open_resting_detail(det):
            seen_sym_keys.add(_intent_venue_symbol_side_key(i.get("venue"), i.get("symbol"), extract_intent_side(i)))
        out.append(_build_status_rec(det))

    if isinstance(open_orders, list) and open_orders:
        carried_resting: list[dict[str, Any]] = []
        for i in reversed(scoped_intents):
            if id(i) in cycle_ids:
                continue
            det = _resolve_executed_open_resting_detail(
                i,
                now_ts,
                extract_notional_fn,
                recent_orders=recent_orders,
                open_orders=open_orders,
                verified_open_venues=verified_open_venues,
            )
            if det is None:
                continue
            oid_str = str(det["order_id"]) if det["order_id"] is not None else None
            s_key = _intent_venue_symbol_side_key(i.get("venue"), i.get("symbol"), extract_intent_side(i))
            if oid_str is not None:
                if oid_str in seen_oids:
                    continue
                seen_oids.add(oid_str)
            elif s_key in seen_sym_keys:
                continue
            seen_sym_keys.add(s_key)
            carried_resting.append(_build_status_rec(det))
        if carried_resting:
            out = list(reversed(carried_resting)) + out

    if not out:
        return None
    return out[0] if len(out) == 1 else out


__all__ = [
    "_attach_intent_age_fields",
    "_filter_intents_by_scope",
    "_is_open_resting_detail",
    "_resolve_executed_open_resting_detail",
    "_resolve_intent_execution_details",
    "build_last_cycle_execution_status",
    "reconcile_intents_from_execution_audit",
    "summarize_recent_ai_intents",
]
