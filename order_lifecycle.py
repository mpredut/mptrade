# order_lifecycle.py
"""Strategy-owned tracked order lifecycle state machine and executor adapter.

Extracted from ``order_retry.py`` to separate strategy-scoped persist-before-submit
and venue status reconciliation from the shared persistent JSONL retry outbox.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from botcore import (
    INVALID_SCALAR_STRINGS,
    _parse_bool_flag,
    check_cooldown_gate,
    clean_order_id,
    exact_pos_int,
    load_root_config,
    normalize_side,
    obj_attr,
    required_float_env,
    required_int_env,
    safe_float,
    safe_nonneg_int,
    safe_pos_float,
    sanitize_pipe_field,
)
from notify_engine import alertnotifiers as alert
from providers.order_ohlc_utils import (
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_orig_qty,
    extract_remaining_qty,
)
from providers.strategy_executor import (
    OrderStatus,
    ProviderError,
    SubmissionOutcome,
    SubmissionRefused,
    capture_submission,
    reconciliation_capabilities_of,
)
from state_io import append_text_line

PersistPending = Callable[[Optional[dict]], None]
SubmitIntent = Callable[[], object]


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


def propagate_submission_refusal(response, outcome_context):
    """Bridge Instrument's result channel into the strict lifecycle contract."""
    if response is None and isinstance(outcome_context, dict):
        if str(outcome_context.get("state") or "").strip().lower() == "refused":
            reason = str(outcome_context.get("reason") or "pre_submit_refused").strip()
            raise SubmissionRefused(reason)
    return response


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
        return extract_order_id(self.intent.get("order_id")) is not None


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
        clean_cid = extract_order_id(client_order_id)
        if not clean_cid:
            raise ProviderError("invalid client_order_id")
        return method(symbol, clean_cid)

    def order_status(self, symbol: str, order_id: str, *, provider_name=None):
        self._require("status_by_order_id", "order_status")
        clean_oid = extract_order_id(order_id)
        if not clean_oid:
            raise ProviderError("invalid order_id")
        return self.executor.order_status(symbol, clean_oid)

    def cancel_order(self, symbol: str, order_id: str, *, provider_name=None):
        self._require("cancel_by_order_id", "cancel_order")
        clean_oid = extract_order_id(order_id)
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
            if extract_order_id(intent.get(key)) is None:
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
        clean_iid = extract_order_id(intent_id)
        clean_cid = extract_order_id(client_order_id)
        clean_sym = extract_order_id(symbol)
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
            clean_oid = extract_order_id(response.order_id)
            if clean_oid is not None:
                fields["order_id"] = clean_oid
            if response.reason:
                fields["submission_reason"] = response.reason
            if response.native is not None:
                fields.update(TrackedOrderLifecycle._order_fields(response.native))
            return fields
        if isinstance(response, dict):
            fields = {}
            order_id = extract_order_id(response)
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
        order_id = extract_order_id(response)
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
        if extract_order_id(pending.get("order_id")):
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
        order_id = extract_order_id(pending.get("order_id"))
        if not order_id:
            err_text = str(pending.get("submit_error") or pending.get("refusal_reason") or "").lower()
            if (
                str(pending.get("submission_outcome") or "").strip().lower() == "refused"
                or any(token in err_text for token in _DEFINITIVE_REFUSAL_TOKENS)
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


# ─── Accepted Order Claim Persistence (formerly accepted_order_persistence.py) ───

load_root_config()

ACCEPTED_TRACKING_PERSIST_ATTEMPTS = required_int_env("ACCEPTED_TRACKING_PERSIST_ATTEMPTS")
ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC = required_float_env("ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC")

if not 2 <= ACCEPTED_TRACKING_PERSIST_ATTEMPTS <= 10:
    raise ValueError("ACCEPTED_TRACKING_PERSIST_ATTEMPTS must be between 2 and 10")
if not 0 < ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC <= 5:
    raise ValueError("ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC must be finite, > 0, and <= 5")


def _claim_client_order_id(record) -> str | None:
    if not isinstance(record, dict):
        return None
    pk = record.get("place_kwargs")
    return clean_order_id(pk.get("client_order_id")) if isinstance(pk, dict) else None


def _accepted_is_durable(order_retry_module, claim, order) -> bool:
    """Recognize a first attempt that committed before reporting an I/O error."""
    try:
        if not isinstance(claim, dict):
            return False
        claim_id = clean_order_id(claim.get("id"))
        expected_order_id = extract_order_id(order)
        expected_client_id = _claim_client_order_id(claim)
        if claim_id is None or expected_order_id is None or expected_client_id is None:
            return False
        for record in (order_retry_module.load_all() or ()):
            if not isinstance(record, dict) or clean_order_id(record.get("id")) != claim_id:
                continue
            current_client_id = _claim_client_order_id(record)
            current_order_id = clean_order_id(record.get("order_id"))
            return (
                str(record.get("lifecycle") or "").strip().lower() == "accepted"
                and current_order_id == expected_order_id
                and current_client_id == expected_client_id
            )
    except Exception:  # noqa: BLE001
        return False
    return False


_ALERTED_PERSISTENCE_FAILURES: dict[tuple, float] = {}
_PERSISTENCE_ALERT_COOLDOWN_SEC = 1800.0


def complete_accepted_claim(
    order_retry_module, claim, *, order, provider_name, symbol, side
) -> bool:
    """Persist venue acceptance under the exact claim, retrying transient I/O."""
    last_error = None
    for attempt in range(ACCEPTED_TRACKING_PERSIST_ATTEMPTS):
        try:
            res = order_retry_module.complete_claim(
                claim, "accepted", order=order, provider_name=provider_name
            )
            if res is True or (not isinstance(res, (bool, str, float, dict, list, tuple, set)) and bool(res)):
                return True
            last_error = RuntimeError("exact accepted-order claim completion returned false")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if _accepted_is_durable(order_retry_module, claim, order):
            return True
        if attempt + 1 < ACCEPTED_TRACKING_PERSIST_ATTEMPTS:
            time.sleep(ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC)

    order_id = extract_order_id(order)
    claim_dict = claim if isinstance(claim, dict) else {}
    client_order_id = _claim_client_order_id(claim_dict)
    owner = ""
    infer_fn = getattr(order_retry_module, "infer_concrete_record_owner", None) or getattr(
        order_retry_module, "infer_record_owner", None
    )
    if callable(infer_fn):
        try:
            owner = str(infer_fn(claim_dict) or "").strip()
        except Exception:
            owner = ""
    owner_tag, _ = alert.format_owner_tags(owner)
    owner_part = f"owner={owner_tag.strip(' []')} " if owner_tag else ""
    qty = claim_dict.get("qty")
    px = claim_dict.get("requested_price") if claim_dict.get("requested_price") is not None else claim_dict.get("ref_price")
    qty_f = safe_float(qty, default=None)
    px_f = safe_float(px, default=None)
    if isinstance(qty, (bool, float)) and qty_f is None:
        qty = None
    if isinstance(px, (bool, float)) and px_f is None:
        px = None
    val_f = safe_float(qty_f * px_f, default=None) if (qty_f is not None and px_f is not None) else None
    val_str = f"~${val_f:.2f}" if val_f is not None else "N/A"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail = (
        f"venue={provider_name} symbol={symbol} side={side} {owner_part}"
        f"qty={qty} price={px} value={val_str} time={now_str} "
        f"order_id={order_id} client_order_id={client_order_id} "
        f"attempts={ACCEPTED_TRACKING_PERSIST_ATTEMPTS} error={last_error}"
    )
    print(f"CRITICAL: accepted order tracking could not be persisted: {detail}")
    dedup_key = (str(provider_name or ""), str(symbol or ""), str(side or ""), str(order_id or client_order_id or ""))
    if check_cooldown_gate(
        _ALERTED_PERSISTENCE_FAILURES, dedup_key, time.monotonic(), _PERSISTENCE_ALERT_COOLDOWN_SEC, 1000
    ):
        alert.notify(
            title=f"🛑 CRITICAL accepted order tracking persistence failure{owner_tag}",
            body=detail,
            source="order-persistence",
            symbol=str(symbol),
            email=True,
        )
    return False


# ─── Order Outcomes Log (formerly order_outcomes_log.py) ───

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER_OUTCOMES_LOG_DIR = os.path.join(ROOT, "logger")


def _sanitize_outcome_field(value):
    """Remove characters that would break the pipe-delimited record format."""
    if value is None or isinstance(value, (bool, dict, list, tuple, set, frozenset)):
        return ""
    if isinstance(value, float) and safe_float(value, default=None) is None:
        return ""
    text = sanitize_pipe_field(value).strip()
    return "" if text.lower() in INVALID_SCALAR_STRINGS else text


def _clean_numeric_field(value):
    if isinstance(value, (dict, list, tuple, set, frozenset)) or safe_float(value, default=None) is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    s = str(value).strip()
    return s if s.lower() not in INVALID_SCALAR_STRINGS else ""


def _clean_bool_field(value) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, frozenset)):
        return ""
    parsed = _parse_bool_flag(value, default=None)
    return str(parsed) if parsed is not None else ""


def log_order_outcome(
    symbol,
    side,
    price,
    qty,
    outcome,
    refuse_reason,
    motivation,
    caller=None,
    *,
    regime_context=None,
):
    """Append one submission-attempt record with optional market regime telemetry."""
    try:
        os.makedirs(ORDER_OUTCOMES_LOG_DIR, exist_ok=True)
        path = os.path.join(
            ORDER_OUTCOMES_LOG_DIR, f"order_outcomes_{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        cols = [
            time.time(),
            _sanitize_outcome_field(symbol),
            _sanitize_outcome_field(side),
            _clean_numeric_field(price),
            _clean_numeric_field(qty),
            _sanitize_outcome_field(outcome),
            _sanitize_outcome_field(refuse_reason),
            _sanitize_outcome_field(caller),
            _sanitize_outcome_field(motivation),
        ]
        if regime_context is not None and not isinstance(
            regime_context, (bool, int, float, str, bytes, list, tuple, set, frozenset)
        ):
            sf = safe_float(obj_attr(regime_context, "strength", None), default=None)
            strength_str = f"{sf:.4f}" if sf is not None else ""

            def _clean_regime_str(val) -> str:
                if isinstance(val, str):
                    return _sanitize_outcome_field(val)
                if val is not None and not isinstance(
                    val, (bool, int, float, bytes, dict, list, tuple, set, frozenset)
                ):
                    inner = getattr(val, "value", getattr(val, "name", None))
                    if isinstance(inner, str):
                        return _sanitize_outcome_field(inner)
                return ""

            cols.extend([
                _clean_regime_str(obj_attr(regime_context, "regime", "")),
                strength_str,
                _clean_regime_str(obj_attr(regime_context, "source", "")),
                _clean_bool_field(obj_attr(regime_context, "fresh", None)),
                _clean_bool_field(obj_attr(regime_context, "fallback_used", None)),
                _clean_regime_str(obj_attr(regime_context, "reason", "")),
            ])
        append_text_line(path, "|".join(sanitize_pipe_field(c) for c in cols))
    except Exception as e:
        print(f"[log_order_outcome] error writing outcomes log: {e}")


# ─── Atomic Order Rollback (formerly order_rollback.py) ───


def extract_order_restore_params(order: dict) -> dict[str, Any]:
    """Normalize resting order dictionaries across venues for restoration after a failed replacement/exit."""
    if not isinstance(order, dict):
        order = {}
    side = extract_order_side(order) or "SELL"
    price = extract_order_price(order, default=0.0)
    qty = safe_pos_float(extract_remaining_qty(order, default=0.0), default=0.0)
    level = safe_pos_float(order.get("level"), default=None)
    raw_kind = order.get("kind")
    kind = (
        raw_kind.strip()
        if isinstance(raw_kind, str) and raw_kind.strip() and raw_kind.strip().lower() not in INVALID_SCALAR_STRINGS
        else ("TP" if side == "SELL" else "ENTRY")
    )
    prod = qty * price if qty > 0.0 and price > 0.0 else 0.0
    amount = safe_pos_float(order.get("amount"), default=safe_pos_float(prod, default=0.0))
    return {
        "side": side,
        "qty": qty,
        "price": price,
        "limit": price,
        "kind": kind,
        "level": level,
        "amount": amount,
        "raw": order,
    }


def restore_cancelled_orders(
    cancelled_orders: list[dict],
    restore_fn: Callable[[dict[str, Any]], Any],
    log_fn: Callable[[str], None] = print,
    reason: str = "",
) -> int:
    """Best-effort restoration of resting orders if replacement/exit fails."""
    if not cancelled_orders or not isinstance(cancelled_orders, (list, tuple)):
        return 0
    clean_reason = reason.strip() if isinstance(reason, str) else ""
    tag = f" ({clean_reason})" if clean_reason else ""
    log_fn(f"  ! [ATOMIC_ROLLBACK] restoring {len(cancelled_orders)} cancelled order(s){tag}")
    restored = 0
    for raw_ord in cancelled_orders:
        if not isinstance(raw_ord, dict):
            continue
        params = extract_order_restore_params(raw_ord)
        if not (safe_pos_float(params.get("qty"), default=None) and safe_pos_float(params.get("limit"), default=None)):
            continue
        try:
            if restore_fn(params) is not False:
                restored += 1
        except Exception as exc:
            log_fn(f"  ! [ATOMIC_ROLLBACK] failed restoring order: {exc}")
    return restored

