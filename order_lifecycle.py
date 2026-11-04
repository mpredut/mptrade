# order_lifecycle.py
"""Strategy-owned tracked order lifecycle state machine and executor adapter.

Extracted from ``order_retry.py`` to separate strategy-scoped persist-before-submit
and venue status reconciliation from the shared persistent JSONL retry outbox.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from botcore import (
    _parse_bool_flag,
    exact_pos_int,
    safe_float,
    safe_nonneg_int,
    safe_pos_float,
)
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_orig_qty,
    normalize_side,
)
from providers.strategy_executor import (
    OrderStatus,
    ProviderError,
    SubmissionOutcome,
    SubmissionRefused,
    capture_submission,
    reconciliation_capabilities_of,
)

PersistPending = Callable[[Optional[dict]], None]
SubmitIntent = Callable[[], object]


def _clean_retry_order_id(val) -> Optional[str]:
    return extract_order_id(val, default=None)


def _lifecycle_finite(raw, *, positive: bool = False):
    return safe_pos_float(raw, default=None) if positive else safe_float(raw, default=None)


def _terminal_payload(status: OrderStatus, observed_at: float) -> dict:
    payload = {
        "status": status.status,
        "venue_status": status.venue_status,
        "filled_qty": status.filled_qty,
        "cost": status.cost,
        "fee": status.fee,
        "observed_at": observed_at,
    }
    if status.orig_qty is not None:
        payload["orig_qty"] = status.orig_qty
    return payload


def _status_from_terminal_payload(payload) -> Optional[OrderStatus]:
    if not isinstance(payload, dict):
        return None
    try:
        status = OrderStatus(
            str(payload["status"]),
            payload["filled_qty"],
            payload["cost"],
            payload["fee"],
            venue_status=payload.get("venue_status", ""),
            orig_qty=payload.get("orig_qty"),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return status if status.terminal else None


_DEFINITIVE_REFUSAL_TOKENS = (
    "insufficient funds",
    "insufficient margin",
    "order minimum not met",
    "volume minimum not met",
    "cost minimum not met",
    "invalid price",
    "invalid arguments",
    "positions limit exceeded",
    "tick size check failed",
)


def _is_definitive_submit_refusal(error: object) -> bool:
    text = str(error or "").lower()
    return any(token in text for token in _DEFINITIVE_REFUSAL_TOKENS)


def propagate_submission_refusal(response, outcome_context):
    """Bridge Instrument's result channel into the strict lifecycle contract."""
    if response is None and isinstance(outcome_context, dict):
        if str(outcome_context.get("state") or "").strip().lower() == "refused":
            reason = str(outcome_context.get("reason") or "pre_submit_refused").strip()
            raise SubmissionRefused(reason)
    return response


# Backwards-compatible import name for existing providers and external scripts.
OrderSubmissionRefused = SubmissionRefused


@dataclass(frozen=True)
class TrackedOrderResult:
    """One lifecycle observation without equating submit acceptance with a fill."""

    outcome: str  # active, terminal, absent, retryable, refused
    intent: dict
    status: Optional[OrderStatus] = None

    def __post_init__(self):
        if self.outcome not in {"active", "terminal", "absent", "retryable", "refused"}:
            raise ValueError(f"invalid tracked-order outcome: {self.outcome!r}")
        if self.outcome == "terminal" and self.status is None:
            raise ValueError("terminal tracked-order result requires status")

    @property
    def order_known(self) -> bool:
        return _clean_retry_order_id(self.intent.get("order_id")) is not None


class StrategyExecutorLifecycleApi:
    """Adapt one strict executor to the lifecycle lookup contract."""

    def __init__(self, executor):
        self.executor = executor

    def reconciliation_capabilities(self):
        return reconciliation_capabilities_of(self.executor)

    def _require(self, capability: str, operation: str) -> None:
        if not getattr(self.reconciliation_capabilities(), capability):
            raise RuntimeError(f"{self.executor.__class__.__name__}: {operation} is unsupported")

    def order_by_client_id(self, symbol: str, client_order_id: str, *, provider_name=None):
        self._require("lookup_by_client_order_id", "order_by_client_id")
        method = getattr(self.executor, "order_by_client_id", None)
        if not callable(method):
            raise RuntimeError(
                f"{provider_name or type(self.executor).__name__}: order_by_client_id is unsupported"
            )
        clean_cid = _clean_retry_order_id(client_order_id)
        if not clean_cid:
            raise ProviderError("invalid client_order_id")
        return method(symbol, clean_cid)

    def order_status(self, symbol: str, order_id: str, *, provider_name=None):
        self._require("status_by_order_id", "order_status")
        clean_oid = _clean_retry_order_id(order_id)
        if not clean_oid:
            raise ProviderError("invalid order_id")
        return self.executor.order_status(symbol, clean_oid)

    def cancel_order(self, symbol: str, order_id: str, *, provider_name=None):
        self._require("cancel_by_order_id", "cancel_order")
        clean_oid = _clean_retry_order_id(order_id)
        if not clean_oid:
            raise ProviderError("invalid order_id")
        return self.executor.cancel_order(symbol, clean_oid)


_RESERVED_INTENT_KEYS = frozenset({
    "order_id", "submit_status", "submitted_qty", "submitted_price",
    "last_status", "filled_qty", "terminal_status", "cancel_attempted_at",
})


class TrackedOrderLifecycle:
    """State machine for one persisted order owned by a strategy."""

    def __init__(
        self,
        market_api,
        *,
        provider_name: str,
        venue: Optional[str] = None,
        missing_confirmations: int = 2,
        retry_on_lookup_error: bool = False,
        max_age_seconds: Optional[float] = None,
        audit=None,
        clock: Callable[[], float] = time.time,
    ):
        if isinstance(provider_name, bool) or not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("provider_name is required")
        if venue is not None and (isinstance(venue, bool) or not isinstance(venue, str) or not venue.strip()):
            raise ValueError("venue must be a non-empty string")
        mc_int = exact_pos_int(missing_confirmations, default=None)
        if mc_int is None:
            raise ValueError("missing_confirmations must be positive")
        if max_age_seconds is not None:
            parsed_age = _lifecycle_finite(max_age_seconds, positive=True)
            if parsed_age is None:
                raise ValueError("max_age_seconds must be finite and positive")
            max_age_seconds = parsed_age
        self.market_api = market_api
        self.provider_name = provider_name.strip()
        self.venue = venue.strip() if venue is not None else self.provider_name
        self.missing_confirmations = mc_int
        self.retry_on_lookup_error = _parse_bool_flag(retry_on_lookup_error, False)
        self.max_age_seconds = max_age_seconds
        self.audit = audit
        self.clock = clock
        self._last_refusal_by_intent: dict[tuple[str, str, str, str], str] = {}

    def _now(self) -> float:
        now = _lifecycle_finite(self.clock(), positive=True)
        if now is None:
            raise ValueError("tracked-order clock returned invalid time")
        return now

    def _audit(self, event: str, intent: dict, **fields) -> None:
        if self.audit is None:
            return
        base = {
            "intent_id": intent.get("intent_id"),
            "venue": self.venue,
            "symbol": intent.get("symbol"),
            "side": intent.get("side"),
            "kind": intent.get("kind"),
            "client_order_id": intent.get("client_order_id"),
            "order_id": intent.get("order_id"),
            **fields,
        }
        self.audit.record(event, **base)

    @staticmethod
    def _validate_identity(intent: dict) -> None:
        if not isinstance(intent, dict):
            raise ValueError("tracked intent must be a dict")
        for key in ("intent_id", "client_order_id", "symbol"):
            if _clean_retry_order_id(intent.get(key)) is None:
                raise ValueError(f"tracked intent missing {key}")
        side_val = intent.get("side")
        if isinstance(side_val, bool) or not str(side_val or "").strip():
            raise ValueError("tracked intent missing side")
        if str(side_val).strip().upper() not in {"BUY", "SELL"}:
            raise ValueError("tracked intent side must be BUY or SELL")
        if _lifecycle_finite(intent.get("requested_qty"), positive=True) is None:
            raise ValueError("tracked intent requested_qty must be positive")
        price = intent.get("requested_price")
        if price is not None and _lifecycle_finite(price, positive=True) is None:
            raise ValueError("tracked intent requested_price must be positive")

    def new_intent(
        self,
        *,
        intent_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        requested_qty: float,
        requested_price: Optional[float] = None,
        kind: Optional[str] = None,
        attempt: int = 1,
        metadata: Optional[Dict] = None,
    ) -> dict:
        if any(isinstance(x, bool) for x in (intent_id, client_order_id, symbol, side, kind)):
            raise ValueError("tracked intent identity fields must not be boolean")
        if isinstance(requested_qty, bool):
            raise ValueError("tracked intent requested_qty must be positive")
        if isinstance(requested_price, bool):
            raise ValueError("tracked intent requested_price must be positive")
        att_int = exact_pos_int(attempt, default=None)
        if att_int is None:
            raise ValueError("tracked intent attempt must be positive")
        clean_iid = _clean_retry_order_id(intent_id)
        clean_cid = _clean_retry_order_id(client_order_id)
        clean_sym = _clean_retry_order_id(symbol)
        if not clean_iid:
            raise ValueError("tracked intent missing intent_id")
        if not clean_cid:
            raise ValueError("tracked intent missing client_order_id")
        if not clean_sym:
            raise ValueError("tracked intent missing symbol")
        intent = {
            "intent_id": clean_iid,
            "client_order_id": clean_cid,
            "symbol": clean_sym,
            "side": normalize_side(side),
            "kind": None if kind is None else str(kind),
            "requested_qty": float(requested_qty),
            "requested_price": None if requested_price is None else float(requested_price),
            "attempt": att_int,
            "created_at": self._now(),
            "lookup_misses": 0,
        }
        reserved = set(intent) | _RESERVED_INTENT_KEYS
        for key, value in dict(metadata or {}).items():
            if key in reserved:
                raise ValueError(f"tracked intent metadata uses reserved key: {key}")
            intent[str(key)] = value
        self._validate_identity(intent)
        return intent

    @staticmethod
    def _order_fields(response) -> dict:
        if isinstance(response, SubmissionOutcome):
            fields = {"submission_outcome": response.state}
            clean_oid = _clean_retry_order_id(response.order_id)
            if clean_oid is not None:
                fields["order_id"] = clean_oid
            if response.reason:
                fields["submission_reason"] = response.reason
            if response.native is not None:
                fields.update(TrackedOrderLifecycle._order_fields(response.native))
            return fields
        if isinstance(response, dict):
            fields = {}
            order_id = _clean_retry_order_id(extract_order_id(response))
            if order_id is not None:
                fields["order_id"] = str(order_id)
            if response.get("status") is not None:
                fields["submit_status"] = str(response["status"])
            orig_qty = extract_orig_qty(response)
            submitted_qty = orig_qty if orig_qty is not None else extract_order_qty(response, default=None)
            if submitted_qty is not None:
                fields["submitted_qty"] = submitted_qty
            submitted_price = extract_order_price(response, default=None)
            if submitted_price is not None:
                fields["submitted_price"] = submitted_price
            return fields
        order_id = _clean_retry_order_id(extract_order_id(response))
        return {"order_id": str(order_id)} if order_id is not None else {}

    def submit(self, intent: dict, *, persist: PersistPending, submit: SubmitIntent) -> TrackedOrderResult:
        """Persist and submit exactly once; reconcile status from a later tick."""
        pending = dict(intent)
        self._validate_identity(pending)
        persist(dict(pending))
        refusal_key = (
            str(pending.get("intent_id") or ""),
            str(pending.get("symbol") or ""),
            str(pending.get("side") or ""),
            str(pending.get("kind") or ""),
        )
        outcome = capture_submission(submit)
        pending["submission_outcome"] = outcome.state
        if outcome.state == "refused":
            pending["refusal_reason"] = outcome.reason
            pending["submit_status"] = "refused_before_submit"
            persist(dict(pending))
            reason_str = str(outcome.reason or "")
            if self._last_refusal_by_intent.get(refusal_key) != reason_str:
                self._last_refusal_by_intent[refusal_key] = reason_str
                self._audit("submit_requested", pending, qty=pending["requested_qty"], price=pending.get("requested_price"))
                self._audit("submit_refused", pending, reason=outcome.reason)
            return TrackedOrderResult("refused", pending)
        self._last_refusal_by_intent.pop(refusal_key, None)
        self._audit("submit_requested", pending, qty=pending["requested_qty"], price=pending.get("requested_price"))
        if outcome.state == "unknown":
            pending["submit_error"] = outcome.reason
            persist(dict(pending))
            self._audit("submit_ambiguous", pending, error=outcome.reason)
            return TrackedOrderResult("active", pending)

        pending.update(self._order_fields(outcome))
        persist(dict(pending))
        if _clean_retry_order_id(pending.get("order_id")):
            self._audit(
                "submit_accepted", pending,
                qty=pending["requested_qty"],
                submitted_qty=pending.get("submitted_qty"),
                price=(
                    pending.get("submitted_price")
                    if pending.get("submitted_price") is not None
                    else pending.get("requested_price")
                ),
                status=pending.get("submit_status"),
            )
        else:
            self._audit("submit_ambiguous", pending, error="response_without_order_id")
        return TrackedOrderResult("active", pending)

    def _persist_terminal(self, pending: dict, status: OrderStatus, persist: PersistPending) -> TrackedOrderResult:
        pending = dict(pending)
        pending["last_status"] = status.status
        pending["filled_qty"] = status.filled_qty
        pending["terminal_status"] = _terminal_payload(status, self._now())
        persist(dict(pending))
        self._audit(
            "order_terminal", pending, status=status.status,
            venue_status=status.venue_status,
            filled_qty=status.filled_qty, cost=status.cost, fee=status.fee,
        )
        return TrackedOrderResult("terminal", pending, status)

    def reconcile(self, intent: dict, *, persist: PersistPending) -> TrackedOrderResult:
        """Reconcile one pending intent against venue truth; never submit here."""
        pending = dict(intent)
        self._validate_identity(pending)

        cached_terminal = _status_from_terminal_payload(pending.get("terminal_status"))
        if cached_terminal is not None:
            return TrackedOrderResult("terminal", pending, cached_terminal)

        symbol = pending["symbol"]
        client_order_id = pending["client_order_id"]
        order_id = _clean_retry_order_id(pending.get("order_id"))
        if not order_id:
            if (
                str(pending.get("submission_outcome") or "").strip().lower() == "refused"
                or _is_definitive_submit_refusal(pending.get("submit_error") or pending.get("refusal_reason"))
            ):
                persist(None)
                return TrackedOrderResult("refused", pending)
            try:
                native = self.market_api.order_by_client_id(
                    symbol, client_order_id, provider_name=self.provider_name
                )
            except Exception as exc:
                pending["lookup_error"] = f"{exc.__class__.__name__}: {exc}"
                persist(dict(pending))
                self._audit("lookup_error", pending, error_type=exc.__class__.__name__, error=str(exc))
                if self.retry_on_lookup_error:
                    persist(None)
                    self._audit("lookup_retryable", pending)
                    return TrackedOrderResult("retryable", pending)
                return TrackedOrderResult("active", pending)
            fields = self._order_fields(native)
            if not fields.get("order_id"):
                misses = (safe_nonneg_int(pending.get("lookup_misses"), default=0) or 0) + 1
                pending["lookup_misses"] = misses
                persist(dict(pending))
                self._audit(
                    "order_missing", pending, missing_confirmation=misses,
                    missing_confirmations_required=self.missing_confirmations,
                )
                if misses < self.missing_confirmations:
                    return TrackedOrderResult("active", pending)
                persist(None)
                return TrackedOrderResult("absent", pending)
            pending.update(fields)
            pending["lookup_misses"] = 0
            pending.pop("lookup_error", None)
            persist(dict(pending))
            order_id = pending["order_id"]
            self._audit("order_recovered", pending)

        try:
            status = self.market_api.order_status(symbol, str(order_id), provider_name=self.provider_name)
        except Exception as exc:
            pending["status_error"] = f"{exc.__class__.__name__}: {exc}"
            persist(dict(pending))
            self._audit("status_error", pending, error_type=exc.__class__.__name__, error=str(exc))
            return TrackedOrderResult("active", pending)

        pending.pop("status_error", None)
        if status.terminal:
            return self._persist_terminal(pending, status, persist)

        pending["last_status"] = status.status
        pending["filled_qty"] = status.filled_qty
        persist(dict(pending))
        self._audit(
            "order_status", pending, status=status.status,
            venue_status=status.venue_status,
            filled_qty=status.filled_qty, cost=status.cost, fee=status.fee,
        )

        created_at = _lifecycle_finite(pending.get("created_at"), positive=True)
        age_seconds = max(0.0, self._now() - created_at) if created_at is not None else None
        cancel_attempted = _lifecycle_finite(pending.get("cancel_attempted_at"), positive=True)
        if (
            self.max_age_seconds is None
            or age_seconds is None
            or age_seconds < self.max_age_seconds
            or cancel_attempted is not None
        ):
            return TrackedOrderResult("active", pending, status)

        pending["cancel_attempted_at"] = self._now()
        persist(dict(pending))
        self._audit("cancel_requested", pending, age_seconds=age_seconds)
        try:
            self.market_api.cancel_order(symbol, str(order_id), provider_name=self.provider_name)
        except Exception as exc:
            pending["cancel_error"] = f"{exc.__class__.__name__}: {exc}"
            persist(dict(pending))
            self._audit("cancel_ambiguous", pending, error_type=exc.__class__.__name__, error=str(exc))
            return TrackedOrderResult("active", pending, status)

        pending.pop("cancel_error", None)
        persist(dict(pending))
        self._audit("cancel_accepted", pending)
        try:
            post_cancel = self.market_api.order_status(symbol, str(order_id), provider_name=self.provider_name)
        except Exception as exc:
            pending["status_error"] = f"{exc.__class__.__name__}: {exc}"
            persist(dict(pending))
            self._audit("post_cancel_status_error", pending, error_type=exc.__class__.__name__, error=str(exc))
            return TrackedOrderResult("active", pending, status)
        if post_cancel.terminal:
            return self._persist_terminal(pending, post_cancel, persist)
        pending["last_status"] = post_cancel.status
        pending["filled_qty"] = post_cancel.filled_qty
        persist(dict(pending))
        self._audit("cancel_pending", pending, status=post_cancel.status, filled_qty=post_cancel.filled_qty)
        return TrackedOrderResult("active", pending, post_cancel)
