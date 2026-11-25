"""Durable AI trade intent state reconciliation against execution audit JSONL logs."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from botcore import (
    _parse_bool_flag,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    extract_order_id,
    extract_order_price,
    safe_pos_float,
)
from .execution_audit import (
    _extract_audit_fill_metrics,
    _read_audit_intent_records,
    iter_audit_jsonl_paths,
)
from .intent_state import (
    _clean_intent_symbol,
    _extract_intent_id,
    _record_intent_state,
    extract_intent_side,
)
from .intent_telemetry import (  # noqa: F401
    _attach_intent_age_fields,
    _filter_intents_by_scope,
    _is_open_resting_detail,
    _resolve_executed_open_resting_detail,
    _resolve_intent_execution_details,
    build_last_cycle_execution_status,
    summarize_recent_ai_intents,
)
from .order_ledger import OrderLedgerReader
from .order_ohlc_utils import extract_executed_qty
from .order_status_ssot import (
    ACCEPTED_WORKING_VENUE_STATUSES,
    CANCEL_OR_EXPIRE_VENUE_STATUSES,
    OPEN_WORKING_OR_EMPTY_VENUE_STATUSES,
    RECONCILABLE_AI_INTENT_STATUSES,
    TERMINAL_CANCEL_VENUE_STATUSES,
    is_cancel_or_expire_status,
)


def _classify_audit_fill_state(
    ev: str,
    v_stat: str,
    l_stat: str,
) -> tuple[str, bool]:
    """Return ``(norm_venue_status, is_terminal)`` for an audit record where ``is_fill`` is True."""
    is_full_fill = v_stat in ("FILLED", "CLOSED") or l_stat in ("closed", "filled")
    is_cancel_or_expire = is_cancel_or_expire_status(v_stat, l_stat)
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


def _fill_record_rank(rec: dict[str, Any]) -> tuple[int, int, float, float]:
    """Monotonic preference rank for competing fill audit records: (is_full_filled, is_terminal, filled_qty, ts)."""
    v_stat, l_stat, _, f_q, *_, is_fill = _extract_audit_fill_metrics(rec)
    if not is_fill:
        return (0, 0, 0.0, 0.0)
    norm_st, is_term = _classify_audit_fill_state(str(rec.get("event") or ""), v_stat, l_stat)
    fq_val = safe_pos_float(f_q, default=0.0) or 0.0
    ts_val = safe_pos_float(rec.get("ts"), default=0.0) or 0.0
    return (1 if norm_st == "FILLED" else 0, 1 if is_term else 0, fq_val, ts_val)


def _should_replace_fill_record(prev: dict[str, Any] | None, rec: dict[str, Any]) -> bool:
    """Return True when ``rec`` should supersede ``prev`` without downgrading terminal status or fill qty."""
    if prev is None:
        return True
    p_full, p_term, p_fq, p_ts = _fill_record_rank(prev)
    r_full, r_term, r_fq, r_ts = _fill_record_rank(rec)
    if p_full and not r_full and r_fq <= p_fq + 1e-12:
        return False
    if p_term and not r_term and r_fq <= p_fq + 1e-12:
        return False
    if p_term and r_fq < p_fq - 1e-12:
        return False
    if r_full != p_full:
        return r_full > p_full
    if r_term != p_term:
        return r_term > p_term
    if abs(r_fq - p_fq) > 1e-12:
        return r_fq > p_fq
    return r_ts >= p_ts


def _partition_intents_for_audit_reconciliation(
    intents: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool]:
    """Partition intents into failed/pending vs incomplete-executed maps, applying prior-cancel shortcuts."""
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
        raw_sym = item.get("symbol")
        if isinstance(raw_sym, str) and raw_sym.strip():
            norm_sym = _clean_intent_symbol(raw_sym)
            if norm_sym and norm_sym != raw_sym:
                item["symbol"] = norm_sym
                updated = True
        clean_id = _extract_intent_id(item)
        if not clean_id:
            continue
        st_up = clean_upper_scalar(item.get("status")) or ""
        if st_up in RECONCILABLE_AI_INTENT_STATUSES:
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
                cur_v_st = clean_upper_scalar(exec_pay.get("status")) or ""
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
                            and cur_v_st in OPEN_WORKING_OR_EMPTY_VENUE_STATUSES
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
                if not needs_ts_backfill and cur_v_st in TERMINAL_CANCEL_VENUE_STATUSES:
                    continue
                if not needs_ts_backfill and has_fill_metrics and not is_resting_msg and cur_v_st in ("", "FILLED", "CLOSED"):
                    continue
                executed_incomplete_by_id[clean_id] = item
    return failed_by_id, executed_incomplete_by_id, updated


_STALE_PENDING_EXECUTION_SEC = 600.0


def _resolve_sibling_cachedb_dir(audit_dir: str, cachedb_dir: Any = None) -> str:
    """Resolve an explicit ``cachedb_dir`` or workspace sibling ``cachedb`` when ``audit_dir`` is ``<ws>/logger/execution_audit``."""
    if isinstance(cachedb_dir, (str, os.PathLike)) and (clean_c := clean_scalar_str(os.fspath(cachedb_dir))):
        return clean_c
    if audit_dir:
        norm_a = os.path.normpath(os.fspath(audit_dir))
        if os.path.basename(norm_a) == "execution_audit" and os.path.basename(os.path.dirname(norm_a)) == "logger":
            sib = os.path.join(os.path.dirname(os.path.dirname(norm_a)), "cachedb")
            if os.path.isdir(sib):
                return sib
    return ""


def _scan_matching_audit_records(
    audit_dir: str,
    failed_by_id: dict[str, dict[str, Any]],
    executed_incomplete_by_id: dict[str, dict[str, Any]],
    *,
    cachedb_dir: Any = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], set[str], dict[str, dict[str, Any]], bool]:
    """Scan OrderLedger (primary SSoT) and pre-ledger execution audit JSONL logs to collect authoritative records per intent ID."""
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
    any_audit_seen_ids: set[str] = set()
    submit_requested_by_id: dict[str, dict[str, Any]] = {}
    has_audit_files = False
    active_exec_oids: dict[str, Any] = {
        iid_e: extract_order_id(it_e.get("execution_payload"), default=None)
        for iid_e, it_e in executed_incomplete_by_id.items()
    }

    record_batches: list[list[dict[str, Any]]] = []
    eff_cachedb = _resolve_sibling_cachedb_dir(audit_dir, cachedb_dir)
    ledger_records: list[dict[str, Any]] = []
    has_ledger = False
    if eff_cachedb:
        ledger_records, has_ledger = OrderLedgerReader(cachedb_dir=eff_cachedb).iter_audit_compat_records(min_day=min_day)
    if has_ledger:
        has_audit_files = True
        if ledger_records:
            record_batches.append(ledger_records)
    else:
        for audit_path in iter_audit_jsonl_paths(audit_dir, min_day=min_day):
            has_audit_files = True
            record_batches.append(_read_audit_intent_records(audit_path))

    for batch in record_batches:
        for rec in batch:
            iid = _extract_intent_id(rec)
            if not iid:
                continue
            any_audit_seen_ids.add(iid)
            ev = str(rec.get("event") or "")
            if ev == "submit_requested":
                submit_requested_by_id[iid] = rec
                if rec.get("qty") is not None and iid not in submitted_qty_by_id:
                    submitted_qty_by_id[iid] = rec.get("qty")
                continue
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
                        if _should_replace_fill_record(prev, rec):
                            matched_records[iid] = rec
                    elif prev is None or str(prev.get("event") or "") == "submit_unknown":
                        matched_records[iid] = rec
                elif ev in ("order_terminal", "order_terminal_retry"):
                    v_stat, l_stat, *_, is_fill = _extract_audit_fill_metrics(rec)
                    if is_fill:
                        if _should_replace_fill_record(prev, rec):
                            matched_records[iid] = rec
                    elif not prev_is_fill and is_cancel_or_expire_status(v_stat, l_stat):
                        prev_oid = clean_order_id(prev.get("order_id")) if prev is not None else None
                        cur_oid = clean_order_id(rec.get("order_id"))
                        if not prev_oid or not cur_oid or prev_oid == cur_oid:
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
                        if _should_replace_fill_record(prev_exec, rec):
                            matched_exec_records[iid] = rec
                    elif (
                        ev in ("order_terminal", "order_terminal_retry")
                        and not prev_exec_is_fill
                        and is_cancel_or_expire_status(v_stat, l_stat)
                    ):
                        matched_exec_records[iid] = rec
    return matched_records, matched_exec_records, submitted_qty_by_id, any_audit_seen_ids, submit_requested_by_id, has_audit_files


def _apply_failed_intent_audit_record(
    intent: dict[str, Any],
    rec: dict[str, Any],
    *,
    fallback_qty: Any = None,
) -> bool:
    """Apply a matched execution-audit record to a FAILED/PENDING/SUBMIT_UNKNOWN intent."""
    st_up = str(intent.get("status") or "").strip().upper()
    ev = str(rec.get("event") or "")
    v_stat, l_stat, filled_qty, _, raw_cost, _, f_px, is_fill = _extract_audit_fill_metrics(
        rec, fallback_ref_price=True
    )
    rec_ts_f = safe_pos_float(rec.get("ts"), default=None)
    eff_ts = rec_ts_f if rec_ts_f is not None else time.time()
    orig_q = rec.get("qty") if rec.get("qty") is not None else fallback_qty
    side_u = (
        extract_intent_side(rec)
        or extract_intent_side(intent)
        or clean_upper_scalar(rec.get("side"))
        or ""
    )
    sym_val = _clean_intent_symbol(rec.get("symbol")) or _clean_intent_symbol(intent.get("symbol")) or None
    clean_rec_oid = clean_order_id(rec.get("order_id"))

    if ev == "submit_unknown":
        if st_up == "SUBMIT_UNKNOWN":
            return False
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
        if cid_unk := clean_order_id(rec.get("client_order_id")):
            exec_payload["clientOrderId"] = cid_unk
        if reason_unk := (clean_scalar_str(rec.get("reason")) or clean_scalar_str(rec.get("failure_reason"))):
            exec_payload["reason"] = reason_unk
        intent["execution_payload"] = exec_payload
        _record_intent_state(
            intent,
            "SUBMIT_UNKNOWN",
            eff_ts,
            action="RECONCILED_UNKNOWN",
            payload=exec_payload,
        )
        return True

    is_cancel_terminal = (
        not is_fill
        and ev in ("order_terminal", "order_terminal_retry")
        and is_cancel_or_expire_status(v_stat, l_stat)
    )
    if is_cancel_terminal:
        if st_up == "FAILED":
            return False
        raw_v_status = v_stat if v_stat in CANCEL_OR_EXPIRE_VENUE_STATUSES else l_stat.upper()
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
        raw_v_status = v_stat if v_stat in ACCEPTED_WORKING_VENUE_STATUSES else "NEW"
        target_state, action_lbl, reason_val = "EXECUTED", "RECONCILED_ACCEPTED", None
        is_term = False
        exec_qty_s = quote_qty_s = ""

    exec_payload = {
        "orderId": clean_rec_oid,
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
        order_id=clean_rec_oid,
        payload=exec_payload,
    )
    return True


def _apply_executed_intent_audit_record(
    intent: dict[str, Any],
    rec: dict[str, Any],
) -> bool:
    """Enrich an incomplete EXECUTED intent's ``execution_payload`` from a matched audit record."""
    exec_pay = dict(intent.get("execution_payload")) if isinstance(intent.get("execution_payload"), dict) else {}
    ev = str(rec.get("event") or "")
    v_stat, l_stat, filled_qty, f_q, raw_cost, cost_f, f_px, is_fill = _extract_audit_fill_metrics(rec)
    rec_ts = safe_pos_float(rec.get("ts"), default=None)
    cur_st = clean_upper_scalar(exec_pay.get("status")) or ""
    clean_rec_oid = clean_order_id(rec.get("order_id"))
    if is_fill:
        norm_st, is_term = _classify_audit_fill_state(ev, v_stat, l_stat)
        if clean_rec_oid is not None:
            exec_pay["orderId"] = clean_rec_oid
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
        if clean_rec_oid is not None:
            exec_pay["orderId"] = clean_rec_oid
        exec_pay["status"] = v_stat if v_stat in ACCEPTED_WORKING_VENUE_STATUSES else "NEW"
        if rec_ts is not None:
            exec_pay["accepted_at"] = rec_ts
        exec_pay.pop("terminal_ts", None)
        exec_pay.pop("reconciled_terminal", None)
    elif cur_st in OPEN_WORKING_OR_EMPTY_VENUE_STATUSES or (
        _parse_bool_flag(exec_pay.get("reconciled_terminal"), False)
        and "terminal_ts" not in exec_pay
        and rec_ts is not None
    ):
        norm_cancel = v_stat if v_stat in CANCEL_OR_EXPIRE_VENUE_STATUSES else "CANCELED"
        exec_pay["status"] = norm_cancel
        if ev == "order_terminal":
            if rec_ts is not None:
                exec_pay["terminal_ts"] = rec_ts
            exec_pay["reconciled_terminal"] = True
    else:
        return False
    if str(exec_pay.get("msg") or "").strip().lower() == "resting":
        exec_pay.pop("msg", None)
    if exec_pay != intent.get("execution_payload"):
        intent["execution_payload"] = exec_pay
        return True
    return False


def reconcile_intents_from_execution_audit(
    intents: list[dict[str, Any]],
    audit_dir: str,
    *,
    cachedb_dir: Any = None,
) -> bool:
    """Reconcile FAILED/PENDING/PENDING_EXECUTION/SUBMIT_UNKNOWN intents and enrich incomplete EXECUTED intents from OrderLedger (with legacy execution_audit fallback)."""
    failed_by_id, executed_incomplete_by_id, updated = _partition_intents_for_audit_reconciliation(intents)
    eff_cachedb = _resolve_sibling_cachedb_dir(audit_dir, cachedb_dir)
    has_audit_dir = bool(audit_dir and os.path.isdir(audit_dir))
    if (not failed_by_id and not executed_incomplete_by_id) or (not has_audit_dir and not eff_cachedb):
        return updated

    matched_records, matched_exec_records, submitted_qty_by_id, any_audit_seen_ids, submit_requested_by_id, has_audit_files = (
        _scan_matching_audit_records(audit_dir, failed_by_id, executed_incomplete_by_id, cachedb_dir=eff_cachedb)
    )
    for iid_chk, item_chk in executed_incomplete_by_id.items():
        if iid_chk not in matched_exec_records:
            ep_chk = item_chk.get("execution_payload")
            if isinstance(ep_chk, dict) and _parse_bool_flag(ep_chk.get("reconciled_terminal"), False):
                ep_chk["_ts_backfill_checked"] = True

    if has_audit_files:
        now_s = time.time()
        for iid_p, item_p in failed_by_id.items():
            if (
                iid_p not in matched_records
                and clean_upper_scalar(item_p.get("status")) == "PENDING_EXECUTION"
                and not item_p.get("execution_payload")
                and (ts_p := safe_pos_float(item_p.get("timestamp"), default=0.0) or 0.0) > 0.0
                and (now_s - ts_p) > _STALE_PENDING_EXECUTION_SEC
            ):
                if iid_p in submit_requested_by_id:
                    req_rec = submit_requested_by_id[iid_p]
                    exec_payload: dict[str, Any] = {
                        "symbol": _clean_intent_symbol(req_rec.get("symbol")) or _clean_intent_symbol(item_p.get("symbol")) or None,
                        "side": extract_intent_side(req_rec) or extract_intent_side(item_p) or "",
                        "executedQty": "",
                        "cummulativeQuoteQty": "",
                        "status": "UNKNOWN",
                        "submission_state": "unknown",
                        "reconciled_from_audit": True,
                    }
                    if req_rec.get("qty") is not None:
                        exec_payload["origQty"] = str(req_rec.get("qty"))
                    if cid_req := clean_order_id(req_rec.get("client_order_id")):
                        exec_payload["clientOrderId"] = cid_req
                    item_p["execution_payload"] = exec_payload
                    _record_intent_state(
                        item_p,
                        "SUBMIT_UNKNOWN",
                        now_s,
                        action="RECONCILED_SUBMIT_REQUESTED_UNKNOWN",
                        reason="submit_requested_unconfirmed_requires_client_id_reconcile",
                        payload=exec_payload,
                    )
                    updated = True
                elif iid_p not in any_audit_seen_ids:
                    _record_intent_state(
                        item_p,
                        "FAILED",
                        now_s,
                        action="RECONCILED_ABANDONED_PRE_SUBMIT",
                        reason="abandoned_pending_execution_no_submit",
                    )
                    updated = True

    if not matched_records and not matched_exec_records:
        return updated

    for iid, rec in matched_records.items():
        if _apply_failed_intent_audit_record(
            failed_by_id[iid], rec, fallback_qty=submitted_qty_by_id.get(iid)
        ):
            updated = True
    for iid, rec in matched_exec_records.items():
        if _apply_executed_intent_audit_record(executed_incomplete_by_id[iid], rec):
            updated = True
    return updated


__all__ = [
    "_attach_intent_age_fields",
    "_classify_audit_fill_state",
    "_filter_intents_by_scope",
    "_is_open_resting_detail",
    "_is_terminal_fill_record",
    "_resolve_executed_open_resting_detail",
    "_resolve_intent_execution_details",
    "build_last_cycle_execution_status",
    "reconcile_intents_from_execution_audit",
    "summarize_recent_ai_intents",
]
