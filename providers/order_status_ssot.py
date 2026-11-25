"""Canonical SSOT taxonomy, venue status classifier, and monotonic evidence lattice for orders and intents."""
from __future__ import annotations

from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass
from typing import Optional

from botcore import (
    StringEnum,
    clean_scalar_str,
    clean_upper_scalar,
    safe_nonneg_float,
    safe_pos_float,
)


class CanonicalOrderState(StringEnum):
    """Unified cross-venue lifecycle state for any bot order or autonomous AI intent."""

    OPEN_WORKING = "OPEN_WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    REPLACED = "REPLACED"
    REFUSED_PRE_SUBMIT = "REFUSED_PRE_SUBMIT"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"


ACCEPTED_WORKING_VENUE_STATUSES: frozenset[str] = frozenset({
    "NEW",
    "OPEN",
    "RESTING",
    "PENDING",
    "WORKING",
})

OPEN_WORKING_OR_EMPTY_VENUE_STATUSES: frozenset[str] = ACCEPTED_WORKING_VENUE_STATUSES | frozenset({""})

CANCEL_OR_EXPIRE_VENUE_STATUSES: frozenset[str] = frozenset({
    "CANCELED",
    "CANCELLED",
    "EXPIRED",
    "REJECTED",
})

TERMINAL_CANCEL_VENUE_STATUSES: frozenset[str] = CANCEL_OR_EXPIRE_VENUE_STATUSES | frozenset({
    "CANCELED_OR_EXPIRED",
})

CANCEL_OR_EXPIRE_LOCAL_STATUSES: frozenset[str] = frozenset({
    "canceled",
    "cancelled",
    "expired",
    "rejected",
})

UNFILLED_ORDER_STATUSES: frozenset[str] = (
    ACCEPTED_WORKING_VENUE_STATUSES
    | CANCEL_OR_EXPIRE_VENUE_STATUSES
    | frozenset({"UNFILLED"})
)

NON_FILL_ORDER_STATUSES: frozenset[str] = UNFILLED_ORDER_STATUSES | frozenset({
    "UNKNOWN",
    "SUBMIT_UNKNOWN",
    "FAILED",
    "ERROR",
    "REFUSED",
    "CANCELED_OR_EXPIRED",
})

NON_FILL_SUBMISSION_STATES: frozenset[str] = frozenset({
    "unknown",
    "refused",
    "failed",
})

ACTIVE_OR_FILLED_LOCAL_STATUSES: frozenset[str] = frozenset({
    "closed",
    "filled",
    "open",
    "new",
    "partially_filled",
    "pending",
    "working",
    "resting",
})

TERMINAL_BROKER_HISTORY_STATUSES: frozenset[str] = frozenset({
    "FILLED",
    "CANCELLED",
    "CANCELED",
    "REPLACED",
    "REJECTED",
    "EXPIRED",
})

TERMINAL_AUDIT_VENUE_STATUSES: frozenset[str] = TERMINAL_BROKER_HISTORY_STATUSES | frozenset({"CLOSED"})

TERMINAL_AUDIT_LOCAL_STATUSES: frozenset[str] = frozenset({
    "closed",
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "replaced",
})

TERMINAL_ACTIVE_INTENT_STATUSES: frozenset[str] = frozenset({
    "closed",
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "refused",
    "refused_before_submit",
    "canceled_or_expired",
    "failed",
    "blocked_by_guard",
    "rejected_ai_circuit_breaker",
    "rejected_gateway_disabled",
    "skipped_duplicate_open_order",
    "observed_shadow",
    "ignored_hold",
})

VALID_SNAPSHOT_ORDER_STATUSES: frozenset[str] = frozenset({
    "open",
    "closed",
    "canceled",
    "expired",
})

TERMINAL_SNAPSHOT_ORDER_STATUSES: frozenset[str] = frozenset({
    "closed",
    "canceled",
    "expired",
})

RECONCILABLE_AI_INTENT_STATUSES: frozenset[str] = frozenset({
    "FAILED",
    "PENDING",
    "PENDING_EXECUTION",
    "SUBMIT_UNKNOWN",
})

TERMINAL_FILL_ROW_STATUSES: frozenset[str] = frozenset({
    "FILLED",
    "CLOSED",
    "CANCELED",
    "CANCELLED",
    "EXPIRED",
    "EXPIRED_IN_MATCH",
    "REJECTED",
    "REPLACED",
})

FILL_COMPLETE_QTY_RATIO: float = 0.9999

_MONOTONIC_STATUS_RANK: dict[str, int] = {
    "OPEN": 1,
    "NEW": 1,
    "RESTING": 1,
    "PENDING": 1,
    "PENDING_NEW": 1,
    "PENDING_CANCEL": 1,
    "WORKING": 1,
    "LOCAL": 1,
    "UNCONFIRMED": 1,
    "CONFIRMED": 1,
    "CANCELLING": 1,
    "REPLACING": 1,
    "PARTIAL": 2,
    "PARTIALLY_FILLED": 2,
    "CANCELED": 3,
    "CANCELLED": 3,
    "CANCELED_OR_EXPIRED": 3,
    "ARCHIVED_CANCELED": 3,
    "EXPIRED": 3,
    "EXPIRED_IN_MATCH": 3,
    "REJECTED": 3,
    "REPLACED": 3,
    "FILLED": 4,
    "CLOSED": 4,
}


@dataclass(frozen=True, slots=True)
class NormalizedVenueStatus:
    """Typed classification result for a raw venue order status string."""

    local_status: str
    venue_status: str
    is_terminal: bool
    is_filled: bool
    is_terminal_non_fill: bool
    is_confirmed_cancel: bool


def classify_venue_order_status(
    raw_status: object,
    *,
    status_map: Mapping[str, str],
    open_statuses: AbstractSet[str] = frozenset(),
    case_sensitive: bool = False,
    preserve_raw_venue_status: bool = False,
    fallback_non_open_status: Optional[str] = None,
) -> NormalizedVenueStatus:
    """Strictly classify a raw venue status string using a declarative venue table.

    Raises ``ValueError`` for boolean, non-scalar, empty, or unknown statuses so
    every venue adapter fails closed on unrecognized payloads.
    """
    if raw_status is None or isinstance(raw_status, (bool, dict, list, tuple, set, frozenset)):
        raise ValueError("invalid or empty venue order status")
    raw_clean = clean_scalar_str(raw_status) or ""
    if not raw_clean:
        raise ValueError("empty venue order status")
    lookup_key = raw_clean if case_sensitive else raw_clean.upper()
    norm_open = open_statuses if case_sensitive else {str(k).strip().upper() for k in open_statuses}
    norm_map = status_map if case_sensitive else {str(k).strip().upper(): v for k, v in status_map.items()}

    if lookup_key in norm_open:
        local_status = "open"
    elif lookup_key in norm_map:
        local_status = norm_map[lookup_key]
    elif fallback_non_open_status is not None:
        local_status = fallback_non_open_status
    else:
        raise ValueError(f"unknown venue order status: {raw_clean!r}")

    if local_status not in VALID_SNAPSHOT_ORDER_STATUSES:
        raise ValueError(f"mapped local status {local_status!r} is not canonical")

    venue_status = raw_clean if preserve_raw_venue_status else raw_clean.upper()
    v_u = venue_status.upper()
    is_terminal = local_status in TERMINAL_SNAPSHOT_ORDER_STATUSES
    is_filled = local_status == "closed"
    is_terminal_non_fill = local_status in {"canceled", "expired"}
    is_confirmed_cancel = bool(
        v_u in {"CANCELED", "CANCELLED", "ARCHIVED_CANCELED"}
        or (
            local_status == "canceled"
            and v_u.endswith(("CANCELED", "CANCELLED", "CANCEL"))
            and "REJECT" not in v_u
            and "PENDING" not in v_u
        )
    )
    return NormalizedVenueStatus(
        local_status=local_status,
        venue_status=venue_status,
        is_terminal=is_terminal,
        is_filled=is_filled,
        is_terminal_non_fill=is_terminal_non_fill,
        is_confirmed_cancel=is_confirmed_cancel,
    )


def _norm_upper_status(raw: object) -> str:
    if raw is None or isinstance(raw, (bool, dict, list, tuple, set, frozenset)):
        return ""
    return clean_upper_scalar(str(raw)) or ""


def _norm_lower_status(raw: object) -> str:
    if raw is None or isinstance(raw, (bool, dict, list, tuple, set, frozenset)):
        return ""
    s = clean_scalar_str(str(raw))
    return s.lower() if s is not None else ""


def is_open_working_status(status: object, *, allow_empty: bool = False) -> bool:
    """Return True when ``status`` represents an accepted, non-terminal resting/working order."""
    if status is None or isinstance(status, (bool, dict, list, tuple, set, frozenset)):
        return bool(allow_empty and status == "")
    raw_str = str(status).strip()
    if not raw_str:
        return bool(allow_empty)
    st_u = clean_upper_scalar(raw_str) or ""
    return st_u in ACCEPTED_WORKING_VENUE_STATUSES


def is_cancel_or_expire_status(
    venue_status: object,
    local_status: object = None,
    *,
    include_composite: bool = False,
) -> bool:
    """Return True when either venue or local status represents a terminal cancel, expiry, or rejection."""
    v_u = _norm_upper_status(venue_status)
    l_lc = _norm_lower_status(local_status)
    v_set = TERMINAL_CANCEL_VENUE_STATUSES if include_composite else CANCEL_OR_EXPIRE_VENUE_STATUSES
    if v_u in v_set or l_lc in CANCEL_OR_EXPIRE_LOCAL_STATUSES:
        return True
    if include_composite and l_lc == "canceled_or_expired":
        return True
    if v_u and (include_composite or v_u != "CANCELED_OR_EXPIRED"):
        return is_terminal_non_fill_venue_status(v_u)
    return False


def is_non_fill_order_status(status: object, submission_state: object = None) -> bool:
    """Return True when ``status`` or ``submission_state`` proves an order has no verified fill fallback."""
    st_u = _norm_upper_status(status)
    sub_lc = _norm_lower_status(submission_state)
    return st_u in NON_FILL_ORDER_STATUSES or sub_lc in NON_FILL_SUBMISSION_STATES


def is_active_or_filled_status(status: object, filled_qty: object = 0.0) -> bool:
    """Return True when a looked-up order status is active/working or has any executed fill."""
    st_lc = _norm_lower_status(status)
    fq = safe_pos_float(filled_qty, default=0.0) or 0.0
    return (bool(st_lc) and st_lc in ACTIVE_OR_FILLED_LOCAL_STATUSES) or fq > 0.0


def is_terminal_audit_status(venue_status: object, local_status: object = None) -> bool:
    """Return True when an execution-audit ``order_status`` record reflects a terminal state."""
    v_u = _norm_upper_status(venue_status)
    return (
        v_u in TERMINAL_AUDIT_VENUE_STATUSES
        or is_terminal_non_fill_venue_status(v_u)
        or _norm_lower_status(local_status) in TERMINAL_AUDIT_LOCAL_STATUSES
    )


def is_confirmed_cancel_order_status(status: object) -> bool:
    """Return True when an ``OrderStatus`` proves a venue cancellation (excluding rejections/pending)."""
    v_u = _norm_upper_status(getattr(status, "venue_status", None))
    if v_u in {"CANCELED", "CANCELLED", "ARCHIVED_CANCELED"}:
        return True
    s_lc = _norm_lower_status(getattr(status, "status", None))
    return bool(
        s_lc == "canceled"
        and (v_u.endswith("CANCELED") or v_u.endswith("CANCELLED") or v_u.endswith("CANCEL"))
        and "REJECT" not in v_u
        and "PENDING" not in v_u
    )


def is_terminal_non_fill_venue_status(status: object) -> bool:
    """Return True when a raw venue status string proves a terminal cancel, expiry, or rejection."""
    v_u = _norm_upper_status(status)
    if not v_u or "PENDING" in v_u:
        return False
    if v_u in TERMINAL_CANCEL_VENUE_STATUSES or v_u in {"ARCHIVED_CANCELED", "EXPIRED_IN_MATCH"}:
        return True
    return v_u.endswith(("CANCELED", "CANCELLED", "CANCEL", "EXPIRED", "REJECTED"))


def is_working_or_pending_venue_status(status: object) -> bool:
    """Return True when a venue status indicates an active, partially filled, or pending order."""
    v_u = _norm_upper_status(status)
    if not v_u:
        return False
    return v_u in {"PARTIALLY_FILLED", "PENDING_CANCEL", "PENDING_NEW"} or is_open_working_status(
        v_u, allow_empty=False
    )


def is_filled_venue_status(status: object, *, include_resolved: bool = False) -> bool:
    """Return True when a raw venue status string represents full execution."""
    v_u = _norm_upper_status(status)
    if v_u in {"FILLED", "CLOSED"}:
        return True
    return bool(include_resolved and v_u == "RESOLVED")


def status_lattice_rank(status: object) -> int:
    """Return monotonic evidence lattice rank: UNKNOWN(0) < OPEN(1) < PARTIAL(2) < TERMINAL_NON_FILL(3) < FILLED(4)."""
    v_u = _norm_upper_status(status)
    if not v_u:
        return 0
    if (rank := _MONOTONIC_STATUS_RANK.get(v_u)) is not None:
        return rank
    if is_terminal_non_fill_venue_status(v_u):
        return 3
    if is_open_working_status(v_u, allow_empty=False):
        return 1
    return 0


def merge_monotonic_fill_row_status(
    existing_st: object,
    incoming_st: object,
    *,
    incoming_ts: int = 0,
    existing_ts: int = 0,
) -> Optional[str]:
    """Merge two fill-row status strings along the monotonic lattice ``OPEN < PARTIAL < TERMINAL_NON_FILL < FILLED``."""
    ex_st = _norm_upper_status(existing_st) or None
    in_st = _norm_upper_status(incoming_st) or None
    if not in_st:
        return ex_st
    if not ex_st:
        return in_st
    ex_rank = status_lattice_rank(ex_st)
    in_rank = status_lattice_rank(in_st)
    if ex_rank >= 3 and in_rank < 3:
        return ex_st
    if in_rank >= 3 and ex_rank < 3:
        return in_st
    if ex_rank == 4 and in_rank == 3:
        return ex_st
    if in_rank == 4 and ex_rank == 3:
        return in_st
    if ex_rank == 2 and in_rank == 1:
        return ex_st
    if in_rank == 2 and ex_rank == 1:
        return in_st
    return in_st if incoming_ts >= existing_ts else ex_st


def classify_order_evidence_state(
    venue_status: object,
    *,
    filled_qty: object = 0.0,
    req_qty: object = None,
    require_qty_match_for_filled: bool = False,
    collapse_cancel_variants: bool = False,
    default_unqualified_state: Optional[str] = None,
) -> Optional[str]:
    """Single canonical evidence classifier shared by ``order_ledger`` transitions and fill-cache reconciliation.

    Returns one of ``"FILLED"``, ``"CANCELED"``, ``"EXPIRED"``, ``"REJECTED"``, ``"REPLACED"``,
    ``"PARTIAL"``, ``"OPEN"``, or ``default_unqualified_state``.
    """
    v_u = _norm_upper_status(venue_status)
    fq = safe_nonneg_float(filled_qty, default=0.0) or 0.0
    rq = safe_pos_float(req_qty, default=None)
    qty_complete = rq is not None and rq > 0.0 and fq >= rq * FILL_COMPLETE_QTY_RATIO
    qty_strictly_partial = rq is not None and rq > 0.0 and fq < rq * FILL_COMPLETE_QTY_RATIO

    if v_u in {"PARTIALLY_FILLED", "PARTIAL"}:
        return "PARTIAL"
    if v_u in {"EXPIRED", "EXPIRED_IN_MATCH"}:
        return "EXPIRED"
    if not collapse_cancel_variants and v_u in {"REPLACED", "REJECTED"}:
        return v_u
    if v_u in CANCEL_OR_EXPIRE_VENUE_STATUSES or is_terminal_non_fill_venue_status(v_u):
        return "CANCELED"
    if v_u and is_open_working_status(v_u, allow_empty=False):
        if not require_qty_match_for_filled and qty_complete:
            return "FILLED"
        return "PARTIAL" if fq > 0.0 else "OPEN"
    if qty_complete:
        return "FILLED"
    if v_u in {"FILLED", "CLOSED"}:
        if require_qty_match_for_filled and qty_strictly_partial:
            return "PARTIAL"
        return "FILLED"
    if fq > 0.0 and (rq is None or qty_strictly_partial):
        return "PARTIAL"
    return default_unqualified_state


def classify_order_lifecycle_state(
    venue_status: object = None,
    local_status: object = None,
    *,
    submission_state: object = None,
    filled_qty: object = 0.0,
    orig_qty: object = None,
    is_resting_in_book: bool = False,
) -> CanonicalOrderState:
    """Resolve the single ``CanonicalOrderState`` from venue/local/submission telemetry."""
    fq = safe_pos_float(filled_qty, default=0.0) or 0.0
    oq = safe_pos_float(orig_qty, default=None)
    v_u = _norm_upper_status(venue_status)
    l_lc = _norm_lower_status(local_status)
    sub_lc = _norm_lower_status(submission_state)

    if is_resting_in_book:
        return CanonicalOrderState.PARTIALLY_FILLED if fq > 0.0 else CanonicalOrderState.OPEN_WORKING
    if v_u in ("FILLED", "CLOSED") or l_lc in ("filled", "closed"):
        return CanonicalOrderState.FILLED
    if v_u == "PARTIALLY_FILLED" or l_lc == "partially_filled":
        return CanonicalOrderState.PARTIALLY_FILLED
    if fq > 0.0:
        if oq is not None and fq + 1e-12 < oq and not is_cancel_or_expire_status(v_u, l_lc, include_composite=True):
            return CanonicalOrderState.PARTIALLY_FILLED
        if is_cancel_or_expire_status(v_u, l_lc, include_composite=True) and (oq is None or fq + 1e-12 < oq):
            return CanonicalOrderState.PARTIALLY_FILLED
        return CanonicalOrderState.FILLED
    if v_u == "REPLACED" or l_lc == "replaced":
        return CanonicalOrderState.REPLACED
    if v_u == "EXPIRED" or l_lc == "expired":
        return CanonicalOrderState.EXPIRED
    if v_u == "REJECTED" or l_lc == "rejected":
        return CanonicalOrderState.REJECTED
    if v_u in ("CANCELED", "CANCELLED", "CANCELED_OR_EXPIRED") or l_lc in ("canceled", "cancelled", "canceled_or_expired"):
        return CanonicalOrderState.CANCELED
    if sub_lc == "refused" or v_u in ("REFUSED", "BLOCKED_BY_GUARD") or l_lc in ("refused", "refused_before_submit"):
        return CanonicalOrderState.REFUSED_PRE_SUBMIT
    if sub_lc == "unknown" or v_u in ("UNKNOWN", "SUBMIT_UNKNOWN") or l_lc in ("unknown", "submit_unknown"):
        return CanonicalOrderState.SUBMIT_UNKNOWN
    if v_u in ACCEPTED_WORKING_VENUE_STATUSES or l_lc in ("open", "new", "resting", "pending", "working", "accepted"):
        return CanonicalOrderState.OPEN_WORKING
    return CanonicalOrderState.SUBMIT_UNKNOWN


def format_prompt_intent_status(
    intent_status: object,
    venue_status: object,
    *,
    in_open_book: bool = False,
    filled_qty: object = 0.0,
    is_unfilled_resting_limit: bool = False,
    is_unfilled_accepted_market: bool = False,
    section: str = "ai_trade_intents",
) -> str:
    """Deterministically map an AI intent's state to the Supervisor prompt display token."""
    it_u = _norm_upper_status(intent_status)
    v_u = _norm_upper_status(venue_status)
    fq = safe_pos_float(filled_qty, default=0.0) or 0.0
    if section == "last_cycle_execution_status":
        if in_open_book:
            return "PARTIALLY_FILLED" if fq > 0.0 else "OPEN_RESTING"
        if it_u == "EXECUTED":
            if v_u in TERMINAL_CANCEL_VENUE_STATUSES:
                return v_u
            if is_unfilled_resting_limit:
                return "OPEN_RESTING"
            if is_unfilled_accepted_market:
                return "ACCEPTED"
            if v_u == "PARTIALLY_FILLED":
                return "PARTIALLY_FILLED"
            return "FILLED"
        if it_u == "OBSERVED_SHADOW":
            return "SHADOW_SIMULATED"
        return it_u or "UNKNOWN"
    if section == "ai_trade_intents" and intent_status == "EXECUTED":
        if v_u in ("OPEN", "NEW", "RESTING"):
            return "RESTING_IN_BOOK"
        if v_u == "PARTIALLY_FILLED" and in_open_book:
            return "PARTIALLY_FILLED_IN_BOOK"
    return str(intent_status) if intent_status is not None else ""
