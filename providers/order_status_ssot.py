"""Canonical SSOT taxonomy and pure predicates for venue order and AI/bot intent lifecycle states."""
from __future__ import annotations

from botcore import (
    StringEnum,
    clean_scalar_str,
    clean_upper_scalar,
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
    return bool(include_composite and l_lc == "canceled_or_expired")


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
    return (
        _norm_upper_status(venue_status) in TERMINAL_AUDIT_VENUE_STATUSES
        or _norm_lower_status(local_status) in TERMINAL_AUDIT_LOCAL_STATUSES
    )


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
