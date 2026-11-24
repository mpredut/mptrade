"""Cross-venue, file-based Order Ledger SSoT core (event journal, state machine, passive tail-overlay reader).

Strict Separation of Concerns (SoC):
- Internal operational memory of local order decisions, intents, submits, cancels, and reconciled transitions.
- Zero exchange network I/O and zero dependency on ``cacheManager``/``cacheaccount``.
- Multi-producer callers append compact JSONL lines via ``OrderLedgerWriter`` (``O_APPEND`` + ``flock``).
- Single-writer reconciliation against ``CacheOrderOpenManager`` and ``CacheOrderFilledManager`` lives in
  ``order_ledger_reconciler_worker.py`` (``OrderLedgerReconciler``), which publishes ``orders_active.json``
  and archives terminal rows into ``orders_closed_YYYY-MM.jsonl``.
- Passive readers use ``OrderLedgerReader`` (``orders_active.json`` + in-RAM tail replay after ``journal_cursor``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
import threading
from typing import Any, Optional, Sequence

from botcore import (
    StringEnum,
    _parse_bool_flag,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    extract_order_id,
    normalize_side,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    sanitize_identifier,
)
from cachepaths import (
    ORDER_LEDGER_CLOSED_PREFIX,
    ORDER_LEDGER_EVENTS_PREFIX,
    clean_symbol_key,
    get_cachedb_dir,
    normalize_venue_name,
    order_ledger_active_path,
    order_ledger_closed_path,
    order_ledger_events_path,
    resolve_cachedb_filepath,
)
from instrument_registry import canonical_display_symbol
from providers.order_id_context import (
    all_correlatable_client_order_ids,
    extract_client_order_id,
    resolve_order_owner,
)
from providers.order_status_ssot import CANCEL_OR_EXPIRE_VENUE_STATUSES, is_open_working_status
from state_io import (
    append_jsonl_record,
    read_json_dict_recovering,
    read_jsonl_dicts,
    read_jsonl_dicts_from_offset,
    save_json_state,
)

ORDER_LEDGER_SCHEMA_VERSION = 1
_EVENT_WRITE_LOCK = threading.Lock()
_GENERIC_WRAPPER_OWNERS = frozenset({
    "unspecified",
    "unknown",
    "market_api",
    "market_api_py",
    "instrument",
    "instrument_py",
    "strategy_executor",
    "strategy_executor_py",
    "order_retry_worker",
    "order_retry_worker_py",
})


class OrderLedgerState(StringEnum):
    """Canonical state-machine states for an order tracked in the global cross-venue ledger."""

    INTENT = "INTENT"
    SUBMITTED = "SUBMITTED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CANCELING = "CANCELING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    REPLACED = "REPLACED"
    FAILED = "FAILED"


ACTIVE_LEDGER_STATES: frozenset[str] = frozenset({
    OrderLedgerState.INTENT.value,
    OrderLedgerState.SUBMITTED.value,
    OrderLedgerState.SUBMIT_UNKNOWN.value,
    OrderLedgerState.OPEN.value,
    OrderLedgerState.PARTIAL.value,
    OrderLedgerState.CANCELING.value,
})
TERMINAL_LEDGER_STATES: frozenset[str] = frozenset({
    OrderLedgerState.FILLED.value,
    OrderLedgerState.CANCELED.value,
    OrderLedgerState.EXPIRED.value,
    OrderLedgerState.REJECTED.value,
    OrderLedgerState.REPLACED.value,
    OrderLedgerState.FAILED.value,
})


class OrderEventType(StringEnum):
    """Immutable transition events appended to ``order_events_YYYY-MM-DD.jsonl``."""

    INTENT = "INTENT"
    GUARD_BLOCKED = "GUARD_BLOCKED"
    SCALED = "SCALED"
    SUBMITTED = "SUBMITTED"
    ACKED = "ACKED"
    SUBMIT_REFUSED = "SUBMIT_REFUSED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    CANCEL_REQ = "CANCEL_REQ"
    CANCEL_ACK = "CANCEL_ACK"
    CANCEL_FAILED = "CANCEL_FAILED"
    STATUS = "STATUS"
    RECOVERED = "RECOVERED"
    RECONCILED_OPEN = "RECONCILED_OPEN"
    RECONCILED_PARTIAL = "RECONCILED_PARTIAL"
    RECONCILED_FILLED = "RECONCILED_FILLED"
    RECONCILED_CANCELED = "RECONCILED_CANCELED"
    RECONCILED_EXPIRED = "RECONCILED_EXPIRED"
    RECONCILED_FAILED = "RECONCILED_FAILED"
    ORPHAN_ADOPTED = "ORPHAN_ADOPTED"


_AUDIT_EVENT_TO_LEDGER_EVENT: dict[str, str] = {
    "submit_requested": OrderEventType.SUBMITTED.value,
    "submit_accepted": OrderEventType.ACKED.value,
    "submit_refused": OrderEventType.SUBMIT_REFUSED.value,
    "submit_unknown": OrderEventType.SUBMIT_UNKNOWN.value,
    "submit_scaled": OrderEventType.SCALED.value,
    "cancel_requested": OrderEventType.CANCEL_REQ.value,
    "cancel_accepted": OrderEventType.CANCEL_ACK.value,
    "cancel_rejected": OrderEventType.CANCEL_FAILED.value,
    "order_status": OrderEventType.STATUS.value,
    "order_terminal": OrderEventType.STATUS.value,
    "order_recovered": OrderEventType.RECOVERED.value,
    "order_reconciled": OrderEventType.RECOVERED.value,
}
_DIRECT_TERMINAL_EVENTS: dict[str, str] = {
    OrderEventType.SUBMIT_REFUSED.value: OrderLedgerState.REJECTED.value,
    OrderEventType.CANCEL_REQ.value: OrderLedgerState.CANCELING.value,
    OrderEventType.CANCEL_ACK.value: OrderLedgerState.CANCELED.value,
    OrderEventType.RECONCILED_FILLED.value: OrderLedgerState.FILLED.value,
    OrderEventType.RECONCILED_CANCELED.value: OrderLedgerState.CANCELED.value,
    OrderEventType.RECONCILED_EXPIRED.value: OrderLedgerState.EXPIRED.value,
    OrderEventType.RECONCILED_FAILED.value: OrderLedgerState.FAILED.value,
}
_SUBMISSION_PROOF_EVENTS: frozenset[str] = frozenset({
    OrderEventType.SUBMITTED.value,
    OrderEventType.ACKED.value,
    OrderEventType.SUBMIT_REFUSED.value,
    OrderEventType.SUBMIT_UNKNOWN.value,
    OrderEventType.CANCEL_REQ.value,
    OrderEventType.CANCEL_ACK.value,
    OrderEventType.STATUS.value,
    OrderEventType.RECOVERED.value,
    OrderEventType.RECONCILED_OPEN.value,
    OrderEventType.RECONCILED_PARTIAL.value,
    OrderEventType.RECONCILED_FILLED.value,
    OrderEventType.ORPHAN_ADOPTED.value,
})
_TRAILING_NON_NEW_EVENTS: frozenset[str] = frozenset({
    OrderEventType.ACKED.value,
    OrderEventType.STATUS.value,
    OrderEventType.CANCEL_REQ.value,
    OrderEventType.CANCEL_ACK.value,
    OrderEventType.CANCEL_FAILED.value,
    OrderEventType.RECOVERED.value,
})


_LOCAL_PRESUBMIT_REFUSAL_TOKENS: tuple[str, ...] = (
    "trend_deferred",
    "profit_guard",
    "insufficient_funds",
    "qty_zero",
    "aborted_by_caller",
    "pre_submit_refused",
    "cooldown",
    "wait_for_trend",
    "circuit_breaker",
    "trading_disabled",
    "execution_disabled",
    "guard",
    "shadow",
    "deferred",
)


def _is_local_presubmit_refusal(reason: object) -> bool:
    txt = (clean_scalar_str(reason) or "").lower()
    return bool(txt) and any(tok in txt for tok in _LOCAL_PRESUBMIT_REFUSAL_TOKENS)


def closed_order_tokens(
    venue: object,
    *,
    symbol: object = None,
    vid: object = None,
    iid: object = None,
    cid: object = None,
) -> list[str]:
    """Return normalized deduplication tokens for a closed order across ``(venue, sym, vid)``, ``iid``, and ``cid``."""
    v_norm = _norm_venue(venue)
    s_lc = _norm_symbol(symbol, venue=v_norm).lower() if symbol else ""
    tokens: list[str] = []
    if vid_c := clean_order_id(vid):
        tokens.append(f"{v_norm}:{s_lc}:vid:{vid_c.lower()}" if s_lc else f"{v_norm}:vid:{vid_c.lower()}")
    if iid_c := clean_order_id(iid):
        tokens.append(f"{v_norm}:iid:{iid_c.lower()}")
    if cid_c := clean_order_id(cid):
        for tok in _correlatable_id_set(cid_c):
            tokens.append(f"{v_norm}:cid:{tok}")
    return tokens


def utc_day_for_ts(ts: float) -> str:
    """Return UTC calendar day ``YYYY-MM-DD`` for epoch timestamp ``ts``."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def utc_month_for_ts(ts: float) -> str:
    """Return UTC calendar month ``YYYY-MM`` for epoch timestamp ``ts``."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def _norm_venue(raw_venue: object) -> str:
    if norm := normalize_venue_name(raw_venue):
        return norm
    clean = clean_scalar_str(raw_venue)
    return clean.lower() if clean else "unknown"


def _norm_symbol(raw_symbol: object, *, venue: object = None) -> str:
    if raw_symbol is None or isinstance(raw_symbol, bool):
        return ""
    return canonical_display_symbol(raw_symbol, venue=venue) or clean_symbol_key(raw_symbol)


def _norm_owner(owner: object = None, *, cid: Optional[str] = None) -> Optional[str]:
    cid_owner: Optional[str] = None
    if cid:
        inferred = resolve_order_owner(cid)
        if inferred and inferred not in ("unspecified", "unknown") and not inferred.startswith("alt:"):
            cid_owner = inferred
    clean = clean_scalar_str(owner)
    if clean:
        norm = sanitize_identifier(clean, default="", sep="_", allow_dots_hyphens=False, strip_chars="_").lower()
        if norm and not norm.startswith("alt_"):
            if norm not in _GENERIC_WRAPPER_OWNERS:
                return norm
            if cid_owner:
                return cid_owner
            return None if norm in ("unspecified", "unknown") else norm
    return cid_owner


def _norm_order_type(order_type: object = None, *, market: Any = None) -> Optional[str]:
    if _parse_bool_flag(market, False):
        return "MARKET"
    clean = clean_upper_scalar(order_type)
    if clean and clean not in {"BUY", "SELL", "B", "S"}:
        return clean
    return "LIMIT" if market is False else None


def _clean_reason(raw: object) -> Optional[str]:
    clean = clean_scalar_str(raw)
    return clean[:240] if clean else None


def build_order_event(
    ev: str | OrderEventType,
    *,
    venue: object,
    symbol: object,
    ts: Optional[float] = None,
    owner: object = None,
    side: object = None,
    order_type: object = None,
    market: Any = None,
    qty: Any = None,
    px: Any = None,
    fq: Any = None,
    avg_px: Any = None,
    fee: Any = None,
    fee_asset: object = None,
    scale: Any = None,
    iid: object = None,
    cid: object = None,
    vid: object = None,
    kind: object = None,
    vst: object = None,
    reason: object = None,
    deferred: Optional[bool] = None,
    src: object = None,
    key: object = None,
) -> dict[str, Any]:
    """Construct a compact, canonical order-ledger event dict omitting empty/None fields."""
    v_str = _norm_venue(venue)
    cid_clean = clean_order_id(cid)
    payload: dict[str, Any] = {
        "ts": round(resolve_now_ts(ts), 3),
        "ev": str(ev.value if isinstance(ev, OrderEventType) else ev).strip().upper(),
        "v": v_str,
        "sym": _norm_symbol(symbol, venue=v_str),
    }
    optional_str = (
        ("owner", _norm_owner(owner, cid=cid_clean)),
        ("side", normalize_side(side)),
        ("type", _norm_order_type(order_type, market=market)),
        ("fee_asset", clean_upper_scalar(fee_asset)),
        ("iid", clean_order_id(iid)),
        ("cid", cid_clean),
        ("vid", clean_order_id(vid)),
        ("kind", clean_scalar_str(kind)),
        ("vst", clean_upper_scalar(vst)),
        ("reason", _clean_reason(reason)),
        ("src", clean_scalar_str(src)),
        ("key", clean_scalar_str(key)),
    )
    for k, val in optional_str:
        if val:
            payload[k] = val
    for k, raw, digits in (("qty", qty, 8), ("px", px, 8), ("avg_px", avg_px, 8), ("scale", scale, 6)):
        if (val_f := safe_pos_float(raw, default=None)) is not None:
            payload[k] = round(val_f, digits)
    for k, raw in (("fq", fq), ("fee", fee)):
        if (val_f := safe_nonneg_float(raw, default=None)) is not None and val_f > 0.0:
            payload[k] = round(val_f, 8)
    if deferred is not None:
        payload["deferred"] = _parse_bool_flag(deferred, False)
    return payload


def _canonical_order_key(
    venue: str,
    *,
    symbol: Optional[str] = None,
    iid: Optional[str] = None,
    cid: Optional[str] = None,
    vid: Optional[str] = None,
    ts: float = 0.0,
) -> str:
    if iid:
        return f"{venue}:iid:{iid}"
    if cid:
        return f"{venue}:cid:{cid}"
    if vid:
        sym_part = _norm_symbol(symbol, venue=venue) if symbol else ""
        return f"{venue}:{sym_part}:vid:{vid}" if sym_part else f"{venue}:vid:{vid}"
    return f"{venue}:anon:{safe_nonneg_int(ts * 1000.0, default=0) or 0}"


def _correlatable_id_set(*cands: Any) -> set[str]:
    out: set[str] = set()
    for raw in cands:
        if clean := clean_order_id(raw):
            out.add(clean.lower())
            out.update(all_correlatable_client_order_ids(clean))
    return out


def match_active_order_key(
    orders: dict[str, dict[str, Any]],
    venue: str,
    *,
    symbol: Optional[str] = None,
    iid: Optional[str] = None,
    cid: Optional[str] = None,
    vid: Optional[str] = None,
) -> Optional[str]:
    """Find the existing active-order key on ``venue`` matching ``(symbol, vid)``, correlatable ``cid``, or ``iid``."""
    v_norm = _norm_venue(venue)
    s_norm = _norm_symbol(symbol, venue=v_norm).lower() if symbol else ""
    q_vid = vid_c.lower() if (vid_c := clean_order_id(vid)) else ""
    q_iid = iid_c.lower() if (iid_c := clean_order_id(iid)) else ""
    q_cids = _correlatable_id_set(cid, iid)
    if not (q_vid or q_iid or q_cids):
        return None

    cid_match: Optional[str] = None
    iid_match: Optional[str] = None
    for key, rec in orders.items():
        if not isinstance(rec, dict) or _norm_venue(rec.get("v")) != v_norm:
            continue
        r_sym = _norm_symbol(rec.get("sym"), venue=v_norm).lower()
        sym_ok = not s_norm or not r_sym or s_norm == r_sym
        r_vid = clean_order_id(rec.get("vid"))
        if q_vid and r_vid and q_vid == r_vid.lower() and sym_ok:
            return key
        if cid_match is None and q_cids and sym_ok and (q_cids & _correlatable_id_set(rec.get("cid"), rec.get("iid"))):
            cid_match = key
        r_iid = clean_order_id(rec.get("iid"))
        if iid_match is None and q_iid and r_iid and q_iid == r_iid.lower():
            iid_match = key
    return cid_match or iid_match


def _merge_records(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for field in ("iid", "cid", "vid", "owner", "side", "type", "kind", "sym", "fee_asset", "reason", "vst", "qty", "px", "avg_px", "fee"):
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]
    p_c, s_c = safe_pos_float(merged.get("created_ts"), default=None), safe_pos_float(secondary.get("created_ts"), default=None)
    if s_c is not None and (p_c is None or s_c < p_c):
        merged["created_ts"] = s_c
    merged["updated_ts"] = max(
        safe_pos_float(merged.get("updated_ts"), default=0.0) or 0.0,
        safe_pos_float(secondary.get("updated_ts"), default=0.0) or 0.0,
    )
    max_fq = max(safe_nonneg_float(merged.get("fq"), default=0.0) or 0.0, safe_nonneg_float(secondary.get("fq"), default=0.0) or 0.0)
    if max_fq > 0.0:
        merged["fq"] = round(max_fq, 8)
    merged["events"] = (safe_nonneg_int(merged.get("events"), default=0) or 0) + (safe_nonneg_int(secondary.get("events"), default=0) or 0)
    return merged


def _classify_venue_status_transition(
    vst: Optional[str], *, fq: float, req_qty: Optional[float], current_state: str
) -> Optional[str]:
    v_u = clean_upper_scalar(vst) or ""
    if v_u in {"FILLED", "CLOSED"}:
        return OrderLedgerState.FILLED.value
    if v_u in {"REPLACED", "EXPIRED", "REJECTED"}:
        return v_u
    if v_u in CANCEL_OR_EXPIRE_VENUE_STATUSES:
        return OrderLedgerState.CANCELED.value
    if v_u == "PARTIALLY_FILLED" or (fq > 0.0 and (req_qty is None or fq < req_qty * 0.9999)):
        return OrderLedgerState.CANCELING.value if current_state == OrderLedgerState.CANCELING.value else OrderLedgerState.PARTIAL.value
    if fq > 0.0 and req_qty is not None and fq >= req_qty * 0.9999:
        return OrderLedgerState.FILLED.value
    if v_u and is_open_working_status(v_u, allow_empty=False):
        return OrderLedgerState.CANCELING.value if current_state == OrderLedgerState.CANCELING.value else OrderLedgerState.OPEN.value
    return None


def make_closed_summary(key: str, rec: dict[str, Any], *, closed_ts: float) -> dict[str, Any]:
    """Build one compact terminal summary row for ``orders_closed_YYYY-MM.jsonl``."""
    summary: dict[str, Any] = {
        "key": key,
        "v": _norm_venue(rec.get("v")),
        "sym": _norm_symbol(rec.get("sym"), venue=rec.get("v")),
        "state": str(rec.get("state") or OrderLedgerState.FAILED.value),
        "created_ts": rec.get("created_ts") or closed_ts,
        "closed_ts": round(closed_ts, 3),
    }
    for k in ("owner", "side", "type", "kind", "iid", "cid", "vid", "qty", "px", "fq", "avg_px", "fee", "fee_asset", "vst", "reason", "events"):
        val = rec.get(k)
        if val is not None and val != "" and val != 0:
            summary[k] = val
    return summary


def apply_order_event(
    orders: dict[str, dict[str, Any]],
    event: dict[str, Any],
    *,
    closed_out: Optional[list[dict[str, Any]]] = None,
    recently_closed: Optional[dict[str, float]] = None,
) -> Optional[str]:
    """Apply one ledger event to ``orders`` in place; return affected key (or archived key)."""
    if not isinstance(event, dict) or not (ev := clean_upper_scalar(event.get("ev"))):
        return None
    venue = _norm_venue(event.get("v"))
    sym = _norm_symbol(event.get("sym"), venue=venue)
    ts = safe_pos_float(event.get("ts"), default=None) or resolve_now_ts()
    iid, cid, vid = clean_order_id(event.get("iid")), clean_order_id(event.get("cid")), clean_order_id(event.get("vid"))

    key_vid = match_active_order_key(orders, venue, symbol=sym, vid=vid) if vid else None
    key_id = match_active_order_key(orders, venue, symbol=sym, iid=iid, cid=cid) if (iid or cid) else None
    if key_vid and key_id and key_vid != key_id:
        orders[key_id] = _merge_records(orders[key_id], orders.pop(key_vid))
        key = key_id
    else:
        key = key_id or key_vid
    if key is None:
        raw_ev_key = clean_scalar_str(event.get("key"))
        if raw_ev_key and raw_ev_key in orders:
            key = raw_ev_key

    if key is None:
        # Ignore late trailing updates (e.g. ExecutionAudit order_terminal arriving after reconciler already closed the order).
        if (
            ev in _TRAILING_NON_NEW_EVENTS
            and isinstance(recently_closed, dict)
            and recently_closed
            and any(tok in recently_closed for tok in closed_order_tokens(venue, symbol=sym, vid=vid, iid=iid, cid=cid))
        ):
            return None
        key = _canonical_order_key(venue, symbol=sym, iid=iid, cid=cid, vid=vid, ts=ts)
        rec = {"v": venue, "sym": sym, "state": OrderLedgerState.INTENT.value, "created_ts": round(ts, 3), "updated_ts": round(ts, 3), "events": 0}
        orders[key] = rec
    else:
        rec = orders[key]

    rec["updated_ts"] = round(max(safe_pos_float(rec.get("updated_ts"), default=0.0) or 0.0, ts), 3)
    rec["events"] = (safe_nonneg_int(rec.get("events"), default=0) or 0) + 1
    rec["last_ev"] = ev
    if vid or ev in _SUBMISSION_PROOF_EVENTS:
        rec["submitted"] = True

    for f_k, f_v in (("sym", sym), ("iid", iid), ("cid", cid), ("side", normalize_side(event.get("side"))), ("type", clean_upper_scalar(event.get("type"))), ("kind", clean_scalar_str(event.get("kind")))):
        if f_v and not rec.get(f_k):
            rec[f_k] = f_v
    if vid:
        rec["vid"] = vid
    if (owner := _norm_owner(event.get("owner"), cid=cid or clean_order_id(rec.get("cid")))) and (
        not rec.get("owner") or rec.get("owner") in _GENERIC_WRAPPER_OWNERS or rec.get("owner") == "external"
    ):
        rec["owner"] = owner
    if (qty_f := safe_pos_float(event.get("qty"), default=None)) is not None and (ev == OrderEventType.SCALED.value or not rec.get("qty")):
        rec["qty"] = round(qty_f, 8)
    if (px_f := safe_pos_float(event.get("px"), default=None)) is not None and not rec.get("px"):
        rec["px"] = round(px_f, 8)
    if (avg_f := safe_pos_float(event.get("avg_px"), default=None)) is not None:
        rec["avg_px"] = round(avg_f, 8)
    for mono_k in ("fq", "fee"):
        if (val_f := safe_nonneg_float(event.get(mono_k), default=None)) is not None and val_f > 0.0:
            rec[mono_k] = round(max(safe_nonneg_float(rec.get(mono_k), default=0.0) or 0.0, val_f), 8)
    for ov_k, ov_v in (("fee_asset", clean_upper_scalar(event.get("fee_asset"))), ("vst", clean_upper_scalar(event.get("vst"))), ("reason", _clean_reason(event.get("reason")))):
        if ov_v:
            rec[ov_k] = ov_v
    if "deferred" in event:
        rec["deferred"] = _parse_bool_flag(event.get("deferred"), False)

    cur_state = str(rec.get("state") or OrderLedgerState.INTENT.value)
    cur_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
    req_qty = safe_pos_float(rec.get("qty"), default=None)
    live_states = {OrderLedgerState.OPEN.value, OrderLedgerState.PARTIAL.value, OrderLedgerState.CANCELING.value}

    if ev in _DIRECT_TERMINAL_EVENTS:
        new_state = _DIRECT_TERMINAL_EVENTS[ev]
    elif ev == OrderEventType.INTENT.value:
        new_state = cur_state if cur_state in live_states else OrderLedgerState.INTENT.value
    elif ev == OrderEventType.GUARD_BLOCKED.value:
        new_state = OrderLedgerState.INTENT.value if _parse_bool_flag(event.get("deferred"), False) else OrderLedgerState.REJECTED.value
    elif ev == OrderEventType.SUBMITTED.value:
        new_state = cur_state if cur_state in live_states else OrderLedgerState.SUBMITTED.value
    elif ev == OrderEventType.SUBMIT_UNKNOWN.value:
        new_state = cur_state if cur_state in {OrderLedgerState.OPEN.value, OrderLedgerState.PARTIAL.value} else OrderLedgerState.SUBMIT_UNKNOWN.value
    elif ev in {OrderEventType.ACKED.value, OrderEventType.RECOVERED.value}:
        vst_st = _classify_venue_status_transition(event.get("vst"), fq=cur_fq, req_qty=req_qty, current_state=cur_state)
        new_state = vst_st or (OrderLedgerState.PARTIAL.value if cur_fq > 0.0 else OrderLedgerState.OPEN.value)
    elif ev == OrderEventType.CANCEL_FAILED.value:
        if cur_state == OrderLedgerState.CANCELING.value:
            new_state = OrderLedgerState.PARTIAL.value if cur_fq > 0.0 else (OrderLedgerState.OPEN.value if rec.get("vid") else cur_state)
        else:
            new_state = cur_state
    elif ev == OrderEventType.STATUS.value:
        new_state = _classify_venue_status_transition(event.get("vst"), fq=cur_fq, req_qty=req_qty, current_state=cur_state) or cur_state
    elif ev in {OrderEventType.RECONCILED_OPEN.value, OrderEventType.ORPHAN_ADOPTED.value}:
        new_state = cur_state if cur_state == OrderLedgerState.CANCELING.value else (OrderLedgerState.PARTIAL.value if cur_fq > 0.0 else OrderLedgerState.OPEN.value)
    elif ev == OrderEventType.RECONCILED_PARTIAL.value:
        new_state = cur_state if cur_state == OrderLedgerState.CANCELING.value else OrderLedgerState.PARTIAL.value
    else:
        new_state = cur_state

    rec["state"] = new_state
    if new_state in TERMINAL_LEDGER_STATES:
        archived = orders.pop(key, rec)
        # Only archive real orders that crossed the venue boundary; pre-submit guard blocks,
        # local outbox guard deferrals (trend_deferred/profit_guard), and unsubmitted intents
        # stay traceable in order_events_YYYY-MM-DD.jsonl only.
        crossed_submit = bool(archived.get("submitted") or archived.get("vid"))
        is_local_refusal = (
            ev == OrderEventType.SUBMIT_REFUSED.value
            and not archived.get("vid")
            and _is_local_presubmit_refusal(archived.get("reason"))
        )
        if closed_out is not None and crossed_submit and not is_local_refusal:
            closed_out.append(make_closed_summary(key, archived, closed_ts=ts))
    return key


def update_recently_closed(
    recently_closed: dict[str, float],
    summaries: Sequence[dict[str, Any]],
    eff_now: Optional[float] = None,
    *,
    ttl_sec: float = 86400.0,
    max_entries: int = 512,
    prune: bool = True,
) -> dict[str, float]:
    """Update deduplication tokens for recently closed orders (shared by reader and reconciler)."""
    if prune and eff_now is not None and eff_now > 0.0:
        cutoff = eff_now - ttl_sec
        pruned = {k: ts_v for k, ts_v in recently_closed.items() if ts_v >= cutoff}
    else:
        pruned = dict(recently_closed)
    for row in summaries:
        if not isinstance(row, dict):
            continue
        row_ts = safe_pos_float(row.get("closed_ts"), default=None) or eff_now or resolve_now_ts()
        for tok in closed_order_tokens(
            row.get("v"),
            symbol=row.get("sym"),
            vid=row.get("vid"),
            iid=row.get("iid"),
            cid=row.get("cid"),
        ):
            pruned[tok] = round(max(pruned.get(tok, 0.0), row_ts), 3)
    if len(pruned) > max_entries:
        keep = sorted(pruned.items(), key=lambda kv: kv[1], reverse=True)[:max_entries]
        pruned = dict(keep)
    return pruned


def replay_producer_events(
    orders: dict[str, dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    recently_closed: dict[str, float],
    closed_out: Optional[list[dict[str, Any]]] = None,
    eff_now: Optional[float] = None,
    prune_recently_closed: bool = True,
) -> tuple[int, dict[str, float]]:
    """Apply producer events in order while keeping ``recently_closed`` strictly synchronized.

    Single Source of Truth used by both ``OrderLedgerReader.load_active_state`` and
    ``OrderLedgerReconciler.reconcile_once`` so identical event streams always yield identical active state.
    """
    rc = update_recently_closed(recently_closed, (), eff_now, prune=prune_recently_closed)
    applied = 0
    sink = closed_out if closed_out is not None else []
    for ev in events:
        if not isinstance(ev, dict) or clean_upper_scalar(ev.get("src")) == "RECONCILER":
            continue
        before_n = len(sink)
        if apply_order_event(orders, ev, closed_out=sink, recently_closed=rc) is not None:
            applied += 1
        if len(sink) > before_n:
            rc = update_recently_closed(rc, sink[before_n:], eff_now, prune=prune_recently_closed)
    return applied, rc


class OrderLedgerBackendInterface(ABC):
    """Level 1 pure abstract contract for cross-venue order-ledger persistence."""

    @abstractmethod
    def append_event(self, event: dict[str, Any]) -> bool: ...

    @abstractmethod
    def read_active_snapshot(self, *, repair_files: bool = False) -> dict[str, Any]: ...

    @abstractmethod
    def write_active_snapshot(self, snapshot: dict[str, Any]) -> bool: ...

    @abstractmethod
    def append_closed_record(self, summary: dict[str, Any]) -> bool: ...

    @abstractmethod
    def read_events_after_cursor(
        self, cursor: Optional[dict[str, Any]], *, max_days: int = 2
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


class FileOrderLedgerBase(OrderLedgerBackendInterface, ABC):
    """Level 2 file-based ledger base over ``cachedb/orders/`` using ``state_io`` primitives."""

    def __init__(self, cachedb_dir: Optional[str] = None):
        self._explicit_cachedb_dir = clean_scalar_str(os.fspath(cachedb_dir)) if isinstance(cachedb_dir, (str, os.PathLike)) else None

    @property
    def cachedb_dir(self) -> str:
        return self._explicit_cachedb_dir or get_cachedb_dir()

    @property
    def orders_dir(self) -> str:
        return os.path.dirname(self.active_path())

    def active_path(self) -> str:
        return order_ledger_active_path(cachedb_dir=self._explicit_cachedb_dir)

    def events_path(self, day_utc: str) -> str:
        return order_ledger_events_path(day_utc, cachedb_dir=self._explicit_cachedb_dir)

    def closed_path(self, month_utc: str) -> str:
        return order_ledger_closed_path(month_utc, cachedb_dir=self._explicit_cachedb_dir)

    def _list_prefixed_jsonl_files(self, prefix: str) -> list[str]:
        candidates: dict[str, str] = {}
        for scan_dir in dict.fromkeys((self.orders_dir, self.cachedb_dir)):
            if not scan_dir or not os.path.isdir(scan_dir):
                continue
            try:
                for fname in os.listdir(scan_dir):
                    if fname.startswith(prefix) and fname.endswith(".jsonl"):
                        candidates.setdefault(fname, os.path.join(scan_dir, fname))
            except OSError:
                continue
        return [candidates[name] for name in sorted(candidates.keys())]

    def list_event_files(self) -> list[str]:
        return self._list_prefixed_jsonl_files(ORDER_LEDGER_EVENTS_PREFIX)

    def list_closed_files(self) -> list[str]:
        return self._list_prefixed_jsonl_files(ORDER_LEDGER_CLOSED_PREFIX)

    @staticmethod
    def _empty_snapshot() -> dict[str, Any]:
        return {
            "schema_version": ORDER_LEDGER_SCHEMA_VERSION,
            "updated_ts": 0.0,
            "journal_cursor": {"file": "", "offset": 0, "offsets": {}},
            "venue_verified_ts": {},
            "symbol_verified_ts": {},
            "recently_closed": {},
            "orders": {},
        }

    @staticmethod
    def _append_jsonl(path: str, record: dict[str, Any], label: str) -> bool:
        try:
            with _EVENT_WRITE_LOCK:
                append_jsonl_record(
                    path, record, sort_keys=False, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str, flock=True
                )
            return True
        except Exception as exc:  # noqa: BLE001 -- ledger append must never block or crash trading
            print(f"[order_ledger] {label} append failed: {exc}")
            return False

    def append_event(self, event: dict[str, Any]) -> bool:
        ts = safe_pos_float(event.get("ts"), default=None) or resolve_now_ts()
        return self._append_jsonl(self.events_path(utc_day_for_ts(ts)), event, "event")

    def append_closed_record(self, summary: dict[str, Any]) -> bool:
        ts = safe_pos_float(summary.get("closed_ts"), default=None) or resolve_now_ts()
        return self._append_jsonl(self.closed_path(utc_month_for_ts(ts)), summary, "closed record")

    def read_active_snapshot(self, *, repair_files: bool = False) -> dict[str, Any]:
        path = str(resolve_cachedb_filepath(self.active_path()))
        if not os.path.isfile(path):
            return self._empty_snapshot()
        raw = read_json_dict_recovering(path, repair_files=_parse_bool_flag(repair_files, False))
        if not isinstance(raw, dict):
            return self._empty_snapshot()
        raw_orders = raw.get("orders") if isinstance(raw.get("orders"), dict) else {}
        cur = raw.get("journal_cursor") if isinstance(raw.get("journal_cursor"), dict) else {}
        raw_ver = raw.get("venue_verified_ts") if isinstance(raw.get("venue_verified_ts"), dict) else {}
        raw_sym_ver = raw.get("symbol_verified_ts") if isinstance(raw.get("symbol_verified_ts"), dict) else {}
        raw_rc = raw.get("recently_closed") if isinstance(raw.get("recently_closed"), dict) else {}
        cur_file = clean_scalar_str(cur.get("file")) or ""
        cur_off = safe_nonneg_int(cur.get("offset"), default=0) or 0
        norm_cursor: dict[str, Any] = {"file": cur_file, "offset": cur_off}
        if isinstance(cur.get("offsets"), dict):
            norm_offsets = {
                str(k): safe_nonneg_int(v, default=0) or 0
                for k, v in cur["offsets"].items()
                if clean_scalar_str(k) is not None
            }
            if cur_file and cur_file not in norm_offsets:
                norm_offsets[cur_file] = cur_off
            norm_cursor["offsets"] = norm_offsets
        return {
            "schema_version": safe_nonneg_int(raw.get("schema_version"), default=ORDER_LEDGER_SCHEMA_VERSION) or ORDER_LEDGER_SCHEMA_VERSION,
            "updated_ts": safe_nonneg_float(raw.get("updated_ts"), default=0.0) or 0.0,
            "journal_cursor": norm_cursor,
            "venue_verified_ts": {str(k): safe_nonneg_float(v, default=0.0) or 0.0 for k, v in raw_ver.items() if clean_scalar_str(k) is not None},
            "symbol_verified_ts": {str(k): safe_nonneg_float(v, default=0.0) or 0.0 for k, v in raw_sym_ver.items() if clean_scalar_str(k) is not None},
            "recently_closed": {str(k): safe_nonneg_float(v, default=0.0) or 0.0 for k, v in raw_rc.items() if clean_scalar_str(k) is not None},
            "orders": {str(k): dict(v) for k, v in raw_orders.items() if clean_scalar_str(k) is not None and isinstance(v, dict)},
        }

    def write_active_snapshot(self, snapshot: dict[str, Any]) -> bool:
        try:
            save_json_state(self.active_path(), snapshot, indent=1, sort_keys=False)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[order_ledger] active snapshot write failed: {exc}")
            return False

    def read_events_after_cursor(
        self, cursor: Optional[dict[str, Any]], *, max_days: int = 2
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Read unread journal records across all eligible UTC-day files using per-file byte offsets."""
        files = self.list_event_files()
        cur_file = clean_scalar_str(cursor.get("file")) if isinstance(cursor, dict) else None
        cur_offset = safe_nonneg_int(cursor.get("offset"), default=0) if isinstance(cursor, dict) else 0
        raw_offsets = cursor.get("offsets") if isinstance(cursor, dict) else None
        has_offsets_map = isinstance(raw_offsets, dict)
        offsets: dict[str, int] = (
            {
                str(k): safe_nonneg_int(v, default=0) or 0
                for k, v in raw_offsets.items()
                if clean_scalar_str(k) is not None
            }
            if has_offsets_map
            else ({cur_file: cur_offset} if cur_file else {})
        )
        if not files:
            return [], {"file": cur_file or "", "offset": cur_offset or 0, "offsets": offsets}

        window = max(1, safe_nonneg_int(max_days, default=2) or 2)
        recent_files = files[-window:]
        recent_names = {os.path.basename(p) for p in recent_files}
        eligible_files = [
            p
            for p in files
            if (
                (fn := os.path.basename(p)) in recent_names
                or fn in offsets
                or (cur_file and fn >= cur_file)
            )
        ]

        events: list[dict[str, Any]] = []
        updated_offsets: dict[str, int] = dict(offsets)
        last_file, last_offset = cur_file or "", cur_offset or 0
        for fpath in eligible_files:
            fname = os.path.basename(fpath)
            if fname in offsets:
                start_off = offsets[fname]
            elif (not has_offsets_map) and cur_file and fname < cur_file:
                # Upgrading from legacy single-file cursor: baseline older day file at current EOF
                # so already-processed history is not replayed, while future late writes ARE captured.
                try:
                    updated_offsets[fname] = os.path.getsize(fpath)
                except OSError:
                    updated_offsets[fname] = 0
                continue
            else:
                start_off = 0
            chunk, new_off = read_jsonl_dicts_from_offset(fpath, start_off)
            events.extend(chunk)
            updated_offsets[fname] = new_off
            if not last_file or fname >= last_file:
                last_file, last_offset = fname, new_off

        # Retain offsets for the active rollover window (recent_names) so cursor stays bounded.
        pruned_offsets = {
            fname: off
            for fname, off in sorted(updated_offsets.items())
            if fname in recent_names or (last_file and fname >= last_file)
        }
        return events, {"file": last_file, "offset": last_offset, "offsets": pruned_offsets}


class OrderLedgerWriter(FileOrderLedgerBase):
    """Level 3 multi-producer best-effort event writer for bots, gateways, and adapters."""

    def record_event(self, ev: str | OrderEventType, *, venue: object, symbol: object, **kwargs: Any) -> bool:
        """Build and append one canonical order-lifecycle event."""
        return self.append_event(build_order_event(ev, venue=venue, symbol=symbol, **kwargs))

    def record_audit_event(
        self,
        audit_event: str,
        *,
        intent_id: object,
        venue: object,
        symbol: object,
        ts: Optional[float] = None,
        fields: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Mirror a legacy ``ExecutionAudit`` event into the canonical Order Ledger journal."""
        mapped = _AUDIT_EVENT_TO_LEDGER_EVENT.get(str(audit_event or "").strip().lower())
        if mapped is None:
            return False
        kw = fields if isinstance(fields, dict) else {}
        fq_val = safe_nonneg_float(kw.get("filled_qty"), default=None)
        cost_val = safe_nonneg_float(kw.get("cost"), default=None)
        avg_px = (cost_val / fq_val) if (cost_val and fq_val and fq_val > 0.0) else safe_pos_float(kw.get("avg_price"), default=None)
        return self.record_event(
            mapped,
            venue=venue,
            symbol=symbol,
            ts=ts,
            owner=kw.get("owner"),
            side=kw.get("side"),
            order_type=kw.get("type") or kw.get("order_type"),
            market=kw.get("market"),
            qty=kw.get("qty"),
            px=kw.get("price"),
            fq=fq_val,
            avg_px=avg_px,
            fee=kw.get("fee"),
            fee_asset=kw.get("fee_asset") or kw.get("feeAsset"),
            scale=kw.get("scale"),
            iid=intent_id,
            cid=kw.get("client_order_id") or extract_client_order_id(kw, default=None),
            vid=kw.get("order_id") or extract_order_id(kw, default=None),
            kind=kw.get("kind"),
            vst=kw.get("venue_status") or kw.get("status"),
            reason=kw.get("reason") or kw.get("failure_reason") or kw.get("error"),
            src="audit_bridge",
        )


class OrderLedgerReader(FileOrderLedgerBase):
    """Level 3 passive read-only query/tracer with 0ms tail-overlay consistency."""

    def load_active_state(self, *, apply_tail: bool = True) -> dict[str, Any]:
        """Load ``orders_active.json`` passively and optionally replay unmaterialized tail events in RAM."""
        snap = self.read_active_snapshot(repair_files=False)
        orders, cursor, recently_closed, tail_count = snap["orders"], snap["journal_cursor"], dict(snap["recently_closed"]), 0
        if _parse_bool_flag(apply_tail, True):
            tail_events, cursor = self.read_events_after_cursor(cursor)
            tail_count, recently_closed = replay_producer_events(
                orders,
                tail_events,
                recently_closed=recently_closed,
                closed_out=None,
                eff_now=snap.get("updated_ts") or None,
                prune_recently_closed=False,
            )
        return {
            **snap,
            "journal_cursor": cursor,
            "recently_closed": recently_closed,
            "tail_events_applied": tail_count,
            "orders": orders,
        }

    def list_active_orders(
        self,
        *,
        venue: Optional[str] = None,
        symbol: Optional[str] = None,
        owner: Optional[str] = None,
        apply_tail: bool = True,
    ) -> list[dict[str, Any]]:
        """Return active orders filtered by optional venue, symbol, or owner."""
        state = self.load_active_state(apply_tail=apply_tail)
        v_f = _norm_venue(venue) if venue else None
        s_f = _norm_symbol(symbol).upper() if symbol else None
        o_f = _norm_owner(owner) if owner else None
        rows = [
            {"key": key, **rec}
            for key, rec in state["orders"].items()
            if (not v_f or _norm_venue(rec.get("v")) == v_f)
            and (not s_f or _norm_symbol(rec.get("sym")).upper() == s_f)
            and (not o_f or _norm_owner(rec.get("owner")) == o_f)
        ]
        rows.sort(key=lambda r: (safe_pos_float(r.get("created_ts"), default=0.0) or 0.0, str(r.get("key") or "")))
        return rows

    def trace_order(
        self, query_id: str, *, max_event_days: int = 14, max_closed_months: int = 3
    ) -> dict[str, Any]:
        """Trace an ``intent_id``, ``client_order_id``, or venue ``order_id`` across active, closed, and events."""
        clean_q = clean_order_id(query_id)
        if not clean_q:
            return {"query_id": "", "found": False, "active_order": None, "closed_orders": [], "events": [], "correlated_ids": []}
        correlated = _correlatable_id_set(clean_q)

        def _row_matches(row: dict[str, Any]) -> bool:
            cands = (row.get("iid"), row.get("cid"), row.get("vid"))
            if _correlatable_id_set(*cands) & correlated:
                correlated.update(_correlatable_id_set(*cands))
                return True
            return False

        active_match = next(
            ({"key": k, **r} for k, r in self.load_active_state(apply_tail=True)["orders"].items() if _row_matches(r)),
            None,
        )
        c_lim = max(1, safe_nonneg_int(max_closed_months, default=3) or 3)
        e_lim = max(1, safe_nonneg_int(max_event_days, default=14) or 14)
        closed_matches = [r for p in self.list_closed_files()[-c_lim:] for r in read_jsonl_dicts(p) if _row_matches(r)]
        event_matches = sorted(
            (r for p in self.list_event_files()[-e_lim:] for r in read_jsonl_dicts(p) if _row_matches(r)),
            key=lambda r: safe_float(r.get("ts"), default=0.0) or 0.0,
        )
        return {
            "query_id": clean_q,
            "found": bool(active_match or closed_matches or event_matches),
            "active_order": active_match,
            "closed_orders": closed_matches,
            "events": event_matches,
            "correlated_ids": sorted(correlated),
        }


_DEFAULT_LEDGER_WRITER = OrderLedgerWriter()


def get_default_ledger_writer() -> OrderLedgerWriter:
    """Return the shared default multi-producer ``OrderLedgerWriter``."""
    return _DEFAULT_LEDGER_WRITER


def ledger_events_tail_for_testing(
    cachedb_dir: Optional[str] = None, *, days: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Read raw event rows from ``order_events_*.jsonl`` (utility for diagnostics and tests)."""
    reader = OrderLedgerReader(cachedb_dir=cachedb_dir)
    paths = [reader.events_path(d) for d in days] if days else reader.list_event_files()
    return [row for path in paths for row in read_jsonl_dicts(path)]
