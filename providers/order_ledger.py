"""Cross-venue, file-based Order Ledger SSoT core (event journal, state machine, passive tail-overlay reader).

Strict Separation of Concerns (SoC):
- This module is the canonical internal operational memory of all local order decisions,
  intents, guard outcomes, submits, cancels, and reconciled transitions across all venues.
- It performs zero exchange network I/O and has zero dependency on ``cacheManager``/``cacheaccount``.
- Multi-producer callers (bots, ``Instrument.place``, ``AuditedStrategyExecutor``, ``MarketApi``,
  ``order_retry_worker``, one-shot scripts) append compact immutable JSONL lines via
  ``OrderLedgerWriter`` (``O_APPEND`` + ``flock``, non-blocking best-effort).
- Single-writer reconciliation against ``CacheOrderOpenManager`` and ``CacheOrderFilledManager``
  lives in ``order_ledger_reconciler.py`` (``OrderLedgerReconciler``), which atomically publishes
  ``orders_active.json`` and archives terminal rows into ``orders_closed_YYYY-MM.jsonl``.
- Passive readers use ``OrderLedgerReader`` (``orders_active.json`` + in-RAM replay of any tail
  lines after ``journal_cursor``) for immediate read-your-own-writes visibility with zero daemon SPOF.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
import threading
import time
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
from providers.order_status_ssot import (
    CANCEL_OR_EXPIRE_VENUE_STATUSES,
    TERMINAL_AUDIT_VENUE_STATUSES,
    is_open_working_status,
)
from state_io import (
    append_jsonl_record,
    read_json_dict_recovering,
    read_jsonl_dicts,
    read_jsonl_dicts_from_offset,
    save_json_state,
)

ORDER_LEDGER_SCHEMA_VERSION = 1

_EVENT_WRITE_LOCK = threading.Lock()


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
    "order_recovered": OrderEventType.RECOVERED.value,
    "order_reconciled": OrderEventType.RECOVERED.value,
}


def utc_day_for_ts(ts: float) -> str:
    """Return UTC calendar day ``YYYY-MM-DD`` for epoch timestamp ``ts``."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def utc_month_for_ts(ts: float) -> str:
    """Return UTC calendar month ``YYYY-MM`` for epoch timestamp ``ts``."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def _norm_venue(raw_venue: object) -> str:
    norm = normalize_venue_name(raw_venue)
    if norm:
        return norm
    clean = clean_scalar_str(raw_venue)
    return clean.lower() if clean else "unknown"


def _norm_symbol(raw_symbol: object, *, venue: object = None) -> str:
    if raw_symbol is None or isinstance(raw_symbol, bool):
        return ""
    return canonical_display_symbol(raw_symbol, venue=venue) or clean_symbol_key(raw_symbol)


def _norm_owner(owner: object = None, *, cid: Optional[str] = None) -> Optional[str]:
    clean = clean_scalar_str(owner)
    if clean:
        norm = sanitize_identifier(clean, default="", sep="_", allow_dots_hyphens=False, strip_chars="_").lower()
        if norm and norm not in {"unspecified", "unknown"}:
            return norm
    if cid:
        inferred = resolve_order_owner(cid)
        if inferred and inferred != "unspecified":
            return inferred
    return None


def _norm_order_type(order_type: object = None, *, market: Any = None) -> Optional[str]:
    if _parse_bool_flag(market, False):
        return "MARKET"
    clean = clean_upper_scalar(order_type)
    if clean and clean not in {"BUY", "SELL", "B", "S"}:
        return clean
    if market is False:
        return "LIMIT"
    return None


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
) -> dict[str, Any]:
    """Construct a compact, canonical order-ledger event dict omitting empty/None fields."""
    eff_ts = round(resolve_now_ts(ts), 3)
    ev_str = str(ev.value if isinstance(ev, OrderEventType) else ev).strip().upper()
    v_str = _norm_venue(venue)
    sym_str = _norm_symbol(symbol, venue=v_str)
    cid_clean = clean_order_id(cid)
    iid_clean = clean_order_id(iid)
    vid_clean = clean_order_id(vid)
    owner_clean = _norm_owner(owner, cid=cid_clean)
    side_clean = normalize_side(side)
    type_clean = _norm_order_type(order_type, market=market)

    payload: dict[str, Any] = {
        "ts": eff_ts,
        "ev": ev_str,
        "v": v_str,
        "sym": sym_str,
    }
    if owner_clean:
        payload["owner"] = owner_clean
    if side_clean:
        payload["side"] = side_clean
    if type_clean:
        payload["type"] = type_clean
    if (qty_f := safe_pos_float(qty, default=None)) is not None:
        payload["qty"] = round(qty_f, 8)
    if (px_f := safe_pos_float(px, default=None)) is not None:
        payload["px"] = round(px_f, 8)
    if (fq_f := safe_nonneg_float(fq, default=None)) is not None and fq_f > 0.0:
        payload["fq"] = round(fq_f, 8)
    if (avg_f := safe_pos_float(avg_px, default=None)) is not None:
        payload["avg_px"] = round(avg_f, 8)
    if (fee_f := safe_nonneg_float(fee, default=None)) is not None and fee_f > 0.0:
        payload["fee"] = round(fee_f, 8)
    if fee_asset_clean := clean_upper_scalar(fee_asset):
        payload["fee_asset"] = fee_asset_clean
    if (scale_f := safe_pos_float(scale, default=None)) is not None:
        payload["scale"] = round(scale_f, 6)
    if iid_clean:
        payload["iid"] = iid_clean
    if cid_clean:
        payload["cid"] = cid_clean
    if vid_clean:
        payload["vid"] = vid_clean
    if kind_clean := clean_scalar_str(kind):
        payload["kind"] = kind_clean
    if vst_clean := clean_upper_scalar(vst):
        payload["vst"] = vst_clean
    if reason_clean := _clean_reason(reason):
        payload["reason"] = reason_clean
    if deferred is not None:
        payload["deferred"] = _parse_bool_flag(deferred, False)
    if src_clean := clean_scalar_str(src):
        payload["src"] = src_clean
    return payload


def _canonical_order_key(venue: str, *, iid: Optional[str], cid: Optional[str], vid: Optional[str], ts: float) -> str:
    if iid:
        return f"{venue}:iid:{iid}"
    if cid:
        return f"{venue}:cid:{cid}"
    if vid:
        return f"{venue}:vid:{vid}"
    ts_ms = safe_nonneg_int(ts * 1000.0, default=0) or 0
    return f"{venue}:anon:{ts_ms}"


def _record_correlatable_ids(rec: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    vids: set[str] = set()
    cids: set[str] = set()
    iids: set[str] = set()
    if vid := clean_order_id(rec.get("vid")):
        vids.add(vid.lower())
    if cid := clean_order_id(rec.get("cid")):
        cids.add(cid.lower())
        cids.update(all_correlatable_client_order_ids(cid))
    if iid := clean_order_id(rec.get("iid")):
        iids.add(iid.lower())
        cids.update(all_correlatable_client_order_ids(iid))
    return vids, cids, iids


def match_active_order_key(
    orders: dict[str, dict[str, Any]],
    venue: str,
    *,
    iid: Optional[str] = None,
    cid: Optional[str] = None,
    vid: Optional[str] = None,
) -> Optional[str]:
    """Find the existing active-order key on ``venue`` matching ``vid``, correlatable ``cid``, or ``iid``."""
    v_norm = _norm_venue(venue)
    q_vid = vid.lower() if (vid := clean_order_id(vid)) else ""
    q_iid = iid.lower() if (iid := clean_order_id(iid)) else ""
    q_cids: set[str] = set()
    if cid_c := clean_order_id(cid):
        q_cids.add(cid_c.lower())
        q_cids.update(all_correlatable_client_order_ids(cid_c))
    if q_iid:
        q_cids.update(all_correlatable_client_order_ids(q_iid))

    if not (q_vid or q_iid or q_cids):
        return None

    cid_match: Optional[str] = None
    iid_match: Optional[str] = None
    for key, rec in orders.items():
        if not isinstance(rec, dict) or _norm_venue(rec.get("v")) != v_norm:
            continue
        r_vids, r_cids, r_iids = _record_correlatable_ids(rec)
        if q_vid and q_vid in r_vids:
            return key
        if cid_match is None and q_cids and (q_cids & r_cids):
            cid_match = key
        if iid_match is None and q_iid and q_iid in r_iids:
            iid_match = key
    return cid_match or iid_match


def _merge_records(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for field in ("iid", "cid", "vid", "owner", "side", "type", "kind", "sym", "fee_asset", "reason", "vst"):
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]
    p_created = safe_pos_float(merged.get("created_ts"), default=None)
    s_created = safe_pos_float(secondary.get("created_ts"), default=None)
    if s_created is not None and (p_created is None or s_created < p_created):
        merged["created_ts"] = s_created
    p_upd = safe_pos_float(merged.get("updated_ts"), default=0.0) or 0.0
    s_upd = safe_pos_float(secondary.get("updated_ts"), default=0.0) or 0.0
    merged["updated_ts"] = max(p_upd, s_upd)
    for num_k in ("qty", "px", "avg_px", "fee"):
        if not merged.get(num_k) and secondary.get(num_k):
            merged[num_k] = secondary[num_k]
    p_fq = safe_nonneg_float(merged.get("fq"), default=0.0) or 0.0
    s_fq = safe_nonneg_float(secondary.get("fq"), default=0.0) or 0.0
    if max(p_fq, s_fq) > 0.0:
        merged["fq"] = round(max(p_fq, s_fq), 8)
    merged["events"] = (safe_nonneg_int(merged.get("events"), default=0) or 0) + (
        safe_nonneg_int(secondary.get("events"), default=0) or 0
    )
    return merged


def _classify_venue_status_transition(
    vst: Optional[str],
    *,
    fq: float,
    req_qty: Optional[float],
    current_state: str,
) -> Optional[str]:
    v_u = clean_upper_scalar(vst) or ""
    if v_u in {"FILLED", "CLOSED"}:
        return OrderLedgerState.FILLED.value
    if v_u == "REPLACED":
        return OrderLedgerState.REPLACED.value
    if v_u == "EXPIRED":
        return OrderLedgerState.EXPIRED.value
    if v_u == "REJECTED":
        return OrderLedgerState.REJECTED.value
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
) -> Optional[str]:
    """Apply one ledger event to ``orders`` in place; return affected key (or archived key)."""
    if not isinstance(event, dict):
        return None
    ev = clean_upper_scalar(event.get("ev"))
    if not ev:
        return None
    venue = _norm_venue(event.get("v"))
    sym = _norm_symbol(event.get("sym"), venue=venue)
    ts = safe_pos_float(event.get("ts"), default=None) or resolve_now_ts()
    iid = clean_order_id(event.get("iid"))
    cid = clean_order_id(event.get("cid"))
    vid = clean_order_id(event.get("vid"))

    # Match existing record; if both vid and iid/cid match different records, merge them cleanly.
    key_vid = match_active_order_key(orders, venue, vid=vid) if vid else None
    key_id = match_active_order_key(orders, venue, iid=iid, cid=cid) if (iid or cid) else None
    if key_vid and key_id and key_vid != key_id:
        primary_k, secondary_k = (key_id, key_vid)
        orders[primary_k] = _merge_records(orders[primary_k], orders.pop(secondary_k))
        key = primary_k
    else:
        key = key_id or key_vid

    if key is None:
        key = _canonical_order_key(venue, iid=iid, cid=cid, vid=vid, ts=ts)
        rec: dict[str, Any] = {
            "v": venue,
            "sym": sym,
            "state": OrderLedgerState.INTENT.value,
            "created_ts": round(ts, 3),
            "updated_ts": round(ts, 3),
            "events": 0,
        }
        orders[key] = rec
    else:
        rec = orders[key]

    rec["updated_ts"] = round(max(safe_pos_float(rec.get("updated_ts"), default=0.0) or 0.0, ts), 3)
    rec["events"] = (safe_nonneg_int(rec.get("events"), default=0) or 0) + 1
    rec["last_ev"] = ev

    if sym and not rec.get("sym"):
        rec["sym"] = sym
    if iid and not rec.get("iid"):
        rec["iid"] = iid
    if cid and not rec.get("cid"):
        rec["cid"] = cid
    if vid:
        rec["vid"] = vid
    if (owner := _norm_owner(event.get("owner"), cid=cid or clean_order_id(rec.get("cid")))) and (
        not rec.get("owner") or rec.get("owner") == "external"
    ):
        rec["owner"] = owner
    if (side := normalize_side(event.get("side"))) and not rec.get("side"):
        rec["side"] = side
    if (otype := clean_upper_scalar(event.get("type"))) and not rec.get("type"):
        rec["type"] = otype
    if (kind := clean_scalar_str(event.get("kind"))) and not rec.get("kind"):
        rec["kind"] = kind
    if (qty_f := safe_pos_float(event.get("qty"), default=None)) is not None:
        if ev == OrderEventType.SCALED.value or not rec.get("qty"):
            rec["qty"] = round(qty_f, 8)
    if (px_f := safe_pos_float(event.get("px"), default=None)) is not None:
        if not rec.get("px"):
            rec["px"] = round(px_f, 8)
    if (fq_f := safe_nonneg_float(event.get("fq"), default=None)) is not None and fq_f > 0.0:
        prev_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
        rec["fq"] = round(max(prev_fq, fq_f), 8)
    if (avg_f := safe_pos_float(event.get("avg_px"), default=None)) is not None:
        rec["avg_px"] = round(avg_f, 8)
    if (fee_f := safe_nonneg_float(event.get("fee"), default=None)) is not None and fee_f > 0.0:
        prev_fee = safe_nonneg_float(rec.get("fee"), default=0.0) or 0.0
        rec["fee"] = round(max(prev_fee, fee_f), 8)
    if fee_asset := clean_upper_scalar(event.get("fee_asset")):
        rec["fee_asset"] = fee_asset
    if vst := clean_upper_scalar(event.get("vst")):
        rec["vst"] = vst
    if reason := _clean_reason(event.get("reason")):
        rec["reason"] = reason
    if "deferred" in event:
        rec["deferred"] = _parse_bool_flag(event.get("deferred"), False)

    cur_state = str(rec.get("state") or OrderLedgerState.INTENT.value)
    cur_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
    req_qty = safe_pos_float(rec.get("qty"), default=None)

    new_state = cur_state
    if ev == OrderEventType.INTENT.value:
        if cur_state not in {OrderLedgerState.OPEN.value, OrderLedgerState.PARTIAL.value, OrderLedgerState.CANCELING.value}:
            new_state = OrderLedgerState.INTENT.value
    elif ev == OrderEventType.GUARD_BLOCKED.value:
        if _parse_bool_flag(event.get("deferred"), False):
            new_state = OrderLedgerState.INTENT.value
        else:
            new_state = OrderLedgerState.REJECTED.value
    elif ev == OrderEventType.SUBMITTED.value:
        if cur_state not in {OrderLedgerState.OPEN.value, OrderLedgerState.PARTIAL.value, OrderLedgerState.CANCELING.value}:
            new_state = OrderLedgerState.SUBMITTED.value
    elif ev == OrderEventType.SUBMIT_REFUSED.value:
        new_state = OrderLedgerState.REJECTED.value
    elif ev == OrderEventType.SUBMIT_UNKNOWN.value:
        if cur_state not in {OrderLedgerState.OPEN.value, OrderLedgerState.PARTIAL.value}:
            new_state = OrderLedgerState.SUBMIT_UNKNOWN.value
    elif ev in {OrderEventType.ACKED.value, OrderEventType.RECOVERED.value}:
        vst_state = _classify_venue_status_transition(
            event.get("vst"), fq=cur_fq, req_qty=req_qty, current_state=cur_state
        )
        new_state = vst_state or (OrderLedgerState.PARTIAL.value if cur_fq > 0.0 else OrderLedgerState.OPEN.value)
    elif ev == OrderEventType.CANCEL_REQ.value:
        new_state = OrderLedgerState.CANCELING.value
    elif ev == OrderEventType.CANCEL_ACK.value:
        new_state = OrderLedgerState.CANCELED.value
    elif ev == OrderEventType.STATUS.value:
        if vst_state := _classify_venue_status_transition(
            event.get("vst"), fq=cur_fq, req_qty=req_qty, current_state=cur_state
        ):
            new_state = vst_state
    elif ev in {OrderEventType.RECONCILED_OPEN.value, OrderEventType.ORPHAN_ADOPTED.value}:
        if cur_state != OrderLedgerState.CANCELING.value:
            new_state = OrderLedgerState.PARTIAL.value if cur_fq > 0.0 else OrderLedgerState.OPEN.value
    elif ev == OrderEventType.RECONCILED_PARTIAL.value:
        if cur_state != OrderLedgerState.CANCELING.value:
            new_state = OrderLedgerState.PARTIAL.value
    elif ev == OrderEventType.RECONCILED_FILLED.value:
        new_state = OrderLedgerState.FILLED.value
    elif ev == OrderEventType.RECONCILED_CANCELED.value:
        new_state = OrderLedgerState.CANCELED.value
    elif ev == OrderEventType.RECONCILED_EXPIRED.value:
        new_state = OrderLedgerState.EXPIRED.value
    elif ev == OrderEventType.RECONCILED_FAILED.value:
        new_state = OrderLedgerState.FAILED.value

    rec["state"] = new_state
    if new_state in TERMINAL_LEDGER_STATES:
        archived = orders.pop(key, rec)
        if closed_out is not None:
            closed_out.append(make_closed_summary(key, archived, closed_ts=ts))
    return key


# =============================================================================
# 3-Tier Polymorphic Ledger Hierarchy (Interface -> File Base -> Writer/Reader)
# =============================================================================


class OrderLedgerBackendInterface(ABC):
    """Level 1 pure abstract contract for cross-venue order-ledger persistence."""

    @abstractmethod
    def append_event(self, event: dict[str, Any]) -> bool:
        """Append one immutable transition event to the durable event journal."""

    @abstractmethod
    def read_active_snapshot(self, *, repair_files: bool = False) -> dict[str, Any]:
        """Read the materialized active-orders snapshot and journal cursor."""

    @abstractmethod
    def write_active_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Atomically persist the materialized active-orders snapshot."""

    @abstractmethod
    def append_closed_record(self, summary: dict[str, Any]) -> bool:
        """Append one compact terminal order summary to the closed archive."""

    @abstractmethod
    def read_events_after_cursor(
        self,
        cursor: Optional[dict[str, Any]],
        *,
        max_days: int = 2,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Read all newly appended events strictly after ``cursor`` and return the advanced cursor."""


class FileOrderLedgerBase(OrderLedgerBackendInterface, ABC):
    """Level 2 file-based ledger base over ``cachedb/orders/`` using ``state_io`` primitives."""

    def __init__(self, cachedb_dir: Optional[str] = None):
        self._explicit_cachedb_dir = clean_scalar_str(os.fspath(cachedb_dir)) if isinstance(cachedb_dir, (str, os.PathLike)) else None

    @property
    def cachedb_dir(self) -> str:
        return self._explicit_cachedb_dir or get_cachedb_dir()

    @property
    def orders_dir(self) -> str:
        active_p = self.active_path()
        return os.path.dirname(active_p)

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

    def append_event(self, event: dict[str, Any]) -> bool:
        try:
            ts = safe_pos_float(event.get("ts"), default=None) or resolve_now_ts()
            path = self.events_path(utc_day_for_ts(ts))
            with _EVENT_WRITE_LOCK:
                append_jsonl_record(
                    path,
                    event,
                    sort_keys=False,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                    flock=True,
                )
            return True
        except Exception as exc:  # noqa: BLE001 -- ledger append must never block or crash trading
            print(f"[order_ledger] event append failed: {exc}")
            return False

    def read_active_snapshot(self, *, repair_files: bool = False) -> dict[str, Any]:
        path = str(resolve_cachedb_filepath(self.active_path()))
        if not os.path.isfile(path):
            return {
                "schema_version": ORDER_LEDGER_SCHEMA_VERSION,
                "updated_ts": 0.0,
                "journal_cursor": {"file": "", "offset": 0},
                "venue_verified_ts": {},
                "orders": {},
            }
        raw = read_json_dict_recovering(path, repair_files=_parse_bool_flag(repair_files, False))
        if not isinstance(raw, dict):
            return {
                "schema_version": ORDER_LEDGER_SCHEMA_VERSION,
                "updated_ts": 0.0,
                "journal_cursor": {"file": "", "offset": 0},
                "venue_verified_ts": {},
                "orders": {},
            }
        raw_orders = raw.get("orders") if isinstance(raw.get("orders"), dict) else {}
        orders_clean = {
            str(k): dict(v)
            for k, v in raw_orders.items()
            if clean_scalar_str(k) is not None and isinstance(v, dict)
        }
        cur = raw.get("journal_cursor") if isinstance(raw.get("journal_cursor"), dict) else {}
        cursor = {
            "file": clean_scalar_str(cur.get("file")) or "",
            "offset": safe_nonneg_int(cur.get("offset"), default=0) or 0,
        }
        raw_ver = raw.get("venue_verified_ts") if isinstance(raw.get("venue_verified_ts"), dict) else {}
        ver_map = {
            str(k): safe_nonneg_float(v, default=0.0) or 0.0
            for k, v in raw_ver.items()
            if clean_scalar_str(k) is not None
        }
        return {
            "schema_version": safe_nonneg_int(raw.get("schema_version"), default=ORDER_LEDGER_SCHEMA_VERSION)
            or ORDER_LEDGER_SCHEMA_VERSION,
            "updated_ts": safe_nonneg_float(raw.get("updated_ts"), default=0.0) or 0.0,
            "journal_cursor": cursor,
            "venue_verified_ts": ver_map,
            "orders": orders_clean,
        }

    def write_active_snapshot(self, snapshot: dict[str, Any]) -> bool:
        try:
            save_json_state(self.active_path(), snapshot, indent=1, sort_keys=False)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[order_ledger] active snapshot write failed: {exc}")
            return False

    def append_closed_record(self, summary: dict[str, Any]) -> bool:
        try:
            ts = safe_pos_float(summary.get("closed_ts"), default=None) or resolve_now_ts()
            path = self.closed_path(utc_month_for_ts(ts))
            with _EVENT_WRITE_LOCK:
                append_jsonl_record(
                    path,
                    summary,
                    sort_keys=False,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                    flock=True,
                )
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[order_ledger] closed record append failed: {exc}")
            return False

    def read_events_after_cursor(
        self,
        cursor: Optional[dict[str, Any]],
        *,
        max_days: int = 2,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        files = self.list_event_files()
        cur_file = clean_scalar_str(cursor.get("file")) if isinstance(cursor, dict) else None
        cur_offset = safe_nonneg_int(cursor.get("offset"), default=0) if isinstance(cursor, dict) else 0
        if not files:
            return [], {"file": cur_file or "", "offset": cur_offset or 0}

        if cur_file:
            target_files = [p for p in files if os.path.basename(p) >= cur_file]
        else:
            limit = max(1, safe_nonneg_int(max_days, default=2) or 2)
            target_files = files[-limit:]

        events: list[dict[str, Any]] = []
        last_file = cur_file or ""
        last_offset = cur_offset or 0
        for fpath in target_files:
            fname = os.path.basename(fpath)
            start_off = cur_offset if (cur_file and fname == cur_file) else 0
            chunk_events, new_off = read_jsonl_dicts_from_offset(fpath, start_off)
            events.extend(chunk_events)
            last_file = fname
            last_offset = new_off
        return events, {"file": last_file, "offset": last_offset}


class OrderLedgerWriter(FileOrderLedgerBase):
    """Level 3 multi-producer best-effort event writer for bots, gateways, and adapters."""

    def record_event(
        self,
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
    ) -> bool:
        """Build and append one canonical order-lifecycle event."""
        event = build_order_event(
            ev,
            venue=venue,
            symbol=symbol,
            ts=ts,
            owner=owner,
            side=side,
            order_type=order_type,
            market=market,
            qty=qty,
            px=px,
            fq=fq,
            avg_px=avg_px,
            fee=fee,
            fee_asset=fee_asset,
            scale=scale,
            iid=iid,
            cid=cid,
            vid=vid,
            kind=kind,
            vst=vst,
            reason=reason,
            deferred=deferred,
            src=src,
        )
        return self.append_event(event)

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
        orders: dict[str, dict[str, Any]] = snap["orders"]
        cursor = snap["journal_cursor"]
        tail_count = 0
        if _parse_bool_flag(apply_tail, True):
            tail_events, adv_cursor = self.read_events_after_cursor(cursor)
            for ev in tail_events:
                if clean_upper_scalar(ev.get("src")) == "RECONCILER":
                    continue
                apply_order_event(orders, ev)
                tail_count += 1
            cursor = adv_cursor
        return {
            "schema_version": snap["schema_version"],
            "updated_ts": snap["updated_ts"],
            "journal_cursor": cursor,
            "venue_verified_ts": snap["venue_verified_ts"],
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
        v_filter = _norm_venue(venue) if venue else None
        sym_filter = _norm_symbol(symbol).upper() if symbol else None
        owner_filter = _norm_owner(owner) if owner else None
        rows: list[dict[str, Any]] = []
        for key, rec in state["orders"].items():
            if v_filter and _norm_venue(rec.get("v")) != v_filter:
                continue
            if sym_filter and _norm_symbol(rec.get("sym")).upper() != sym_filter:
                continue
            if owner_filter and _norm_owner(rec.get("owner")) != owner_filter:
                continue
            rows.append({"key": key, **rec})
        rows.sort(key=lambda r: (safe_pos_float(r.get("created_ts"), default=0.0) or 0.0, str(r.get("key") or "")))
        return rows

    def trace_order(
        self,
        query_id: str,
        *,
        max_event_days: int = 14,
        max_closed_months: int = 3,
    ) -> dict[str, Any]:
        """Trace an ``intent_id``, ``client_order_id``, or venue ``order_id`` across active, closed, and events."""
        clean_q = clean_order_id(query_id)
        if not clean_q:
            return {
                "query_id": "",
                "found": False,
                "active_order": None,
                "closed_orders": [],
                "events": [],
                "correlated_ids": [],
            }
        correlated: set[str] = {clean_q.lower()} | all_correlatable_client_order_ids(clean_q)

        def _expand(*cands: Any) -> None:
            for cand in cands:
                if c := clean_order_id(cand):
                    correlated.add(c.lower())
                    correlated.update(all_correlatable_client_order_ids(c))

        def _row_matches(row: dict[str, Any]) -> bool:
            cands = (row.get("iid"), row.get("cid"), row.get("vid"))
            row_ids: set[str] = set()
            for c in cands:
                if clean_c := clean_order_id(c):
                    row_ids.add(clean_c.lower())
                    row_ids.update(all_correlatable_client_order_ids(clean_c))
            if row_ids & correlated:
                _expand(*cands)
                return True
            return False

        active_state = self.load_active_state(apply_tail=True)
        active_match: Optional[dict[str, Any]] = None
        for key, rec in active_state["orders"].items():
            if _row_matches(rec):
                active_match = {"key": key, **rec}
                break

        closed_matches: list[dict[str, Any]] = []
        closed_limit = max(1, safe_nonneg_int(max_closed_months, default=3) or 3)
        for cpath in self.list_closed_files()[-closed_limit:]:
            for rec in read_jsonl_dicts(cpath):
                if _row_matches(rec):
                    closed_matches.append(rec)

        event_matches: list[dict[str, Any]] = []
        ev_limit = max(1, safe_nonneg_int(max_event_days, default=14) or 14)
        for epath in self.list_event_files()[-ev_limit:]:
            for rec in read_jsonl_dicts(epath):
                if _row_matches(rec):
                    event_matches.append(rec)
        event_matches.sort(key=lambda r: safe_float(r.get("ts"), default=0.0) or 0.0)

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
    """Return the shared default multi-producer ``OrderLedgerWriter`` (dynamically tracks ``get_cachedb_dir()``)."""
    return _DEFAULT_LEDGER_WRITER


def ledger_events_tail_for_testing(
    cachedb_dir: Optional[str] = None, *, days: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Read raw event rows from ``order_events_*.jsonl`` (utility for diagnostics and tests)."""
    reader = OrderLedgerReader(cachedb_dir=cachedb_dir)
    paths = [reader.events_path(d) for d in days] if days else reader.list_event_files()
    out: list[dict[str, Any]] = []
    for path in paths:
        out.extend(read_jsonl_dicts(path))
    return out
