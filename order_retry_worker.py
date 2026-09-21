from __future__ import annotations
#!/usr/bin/env python3
"""order_retry_worker.py — the SINGLE READER of the order_retry.py replacement outbox.

Model B uses multiple writers (Instrument.place -> durable pre-submit outbox) and
ONE reader: this process. It scans the queue and, for every DUE order whose
RETRY_INTERVAL_SEC has elapsed, checks oq.price_gate_ok to determine whether the current
price is more favorable than the requested price. If not, the order remains queued
without counting an attempt while it waits for price to recover. If favorable, the worker
retries through mkt.place() at CURRENT PRICE with caller_owns_retry=True to prevent an
automatic re-enqueue. Provider acceptance keeps the leased revision as ``accepted``;
the worker then polls normalized status without resubmitting open/partial orders. A
fill removes it. REJECTED/EXPIRED creates a new revision for only the unfilled
remainder; CANCELED is terminal and is never blindly resubmitted. Failure increments
attempts and releases the lease while retaining it. A deterministic venue-filter
refusal is terminal and removes the unchanged intent instead of clogging the queue.
Trend deferral releases without consuming attempts or TTL. Exceeding active TTL or the
attempt limit removes it and sends an alert for manual intervention.

``process_once`` has an injectable market facade for tests. It still mutates the
persistent queue and sends alerts. Claiming uses a durable lease, so a crash leaves the
intent recoverable after the lease expires instead of deleting it before submission.

``main`` adds the single-instance loop and supervision. ``RETRY_ENABLED=false`` is
the configured kill switch.
"""
import os
import sys
import time
import argparse
import math

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import order_retry as oq
import alertnotifiers as alert
from botcore import required_float_env, single_instance
from providers.execution_audit import ExecutionAudit


WORKER_POLL_SEC = required_float_env("RETRY_WORKER_POLL_SEC")
if not math.isfinite(WORKER_POLL_SEC) or WORKER_POLL_SEC <= 0:
    raise ValueError("RETRY_WORKER_POLL_SEC must be finite and > 0")
GIVEUP_SUMMARY_SEC = required_float_env("RETRY_GIVEUP_SUMMARY_SEC")
if not math.isfinite(GIVEUP_SUMMARY_SEC) or GIVEUP_SUMMARY_SEC <= 0:
    raise ValueError("RETRY_GIVEUP_SUMMARY_SEC must be finite and > 0")

_AUDIT = ExecutionAudit()
_RECONCILE_FOUND = "found"
_RECONCILE_NOT_FOUND = "not_found"
_RECONCILE_UNAVAILABLE = "unavailable"
_AMBIGUOUS_SUBMISSION_REASONS = frozenset({
    "",
    "submit_pending",
    "submit_ambiguous",
    "response_without_order_id",
})
_ALERTED_QUEUE_CORRUPTIONS = set()
_ALERTED_PRODUCER_QUARANTINES = set()
_ALERTED_RECONCILIATION_QUARANTINES = set()


def _accepted_order(order):
    return oq.order_id_from_response(order) is not None


def _submission_requires_reconciliation(record):
    """Return whether a persisted intent might already have reached its venue."""
    submission_state = str(
        record.get("submission_state") or "").strip().lower()
    if submission_state == "refused":
        return False
    if submission_state in {"producer_claimed", "unknown"}:
        return True
    reason = str(record.get("last_failure_reason") or "").strip().lower()
    return reason in _AMBIGUOUS_SUBMISSION_REASONS


def _reconcile_client_order(
        mkt, symbol, client_order_id, *, provider_name=None):
    """Return a tri-state client-ID lookup result and any accepted order.

    Only an explicit None response confirms absence. Missing lookup support,
    lookup errors, and malformed responses remain unavailable so callers cannot
    turn uncertainty into a duplicate submission.
    """
    lookup = getattr(mkt, "order_by_client_id", None)
    if not callable(lookup) or not client_order_id:
        return _RECONCILE_UNAVAILABLE, None
    try:
        order = lookup(
            symbol, client_order_id, provider_name=provider_name)
    except Exception as exc:  # noqa: BLE001 — unsupported/unavailable stays fail closed.
        print(f"[order_retry] reconciliation unavailable {symbol}/{client_order_id}: {exc}")
        return _RECONCILE_UNAVAILABLE, None
    if order is None:
        return _RECONCILE_NOT_FOUND, None
    if not isinstance(order, dict):
        print(
            f"[order_retry] malformed reconciliation response "
            f"{symbol}/{client_order_id}: {type(order).__name__}")
        return _RECONCILE_UNAVAILABLE, None
    if oq.order_id_from_response(order) is None:
        print(
            f"[order_retry] reconciliation response has no order ID "
            f"{symbol}/{client_order_id}")
        return _RECONCILE_UNAVAILABLE, None
    return _RECONCILE_FOUND, order


def _provider_name(mkt, record):
    configured = record.get("provider_name")
    if configured:
        return str(configured)
    resolver = getattr(mkt, "provider_name_for", None)
    if callable(resolver):
        return str(resolver(record.get("symbol")))
    return None


def _reconciliation_capabilities(mkt, symbol, provider_name):
    """Return an explicit capability declaration, or None fail closed."""
    capabilities = getattr(mkt, "reconciliation_capabilities", None)
    if not callable(capabilities):
        return None
    try:
        result = capabilities(symbol, provider_name=provider_name)
    except Exception:  # noqa: BLE001 — capability ambiguity fails closed.
        return None
    if type(getattr(result, "lookup_by_client_order_id", None)) is not bool:
        return None
    return result


def _supports_client_id_reconciliation(mkt, symbol, provider_name):
    """Return whether the provider explicitly supports client-ID lookup."""
    capabilities = _reconciliation_capabilities(
        mkt, symbol, provider_name)
    return bool(
        capabilities is not None
        and capabilities.lookup_by_client_order_id)


def _not_found_is_reliable(mkt, record, now, provider_name):
    """Return whether this venue absence is recent enough to authorize submit."""
    capabilities = _reconciliation_capabilities(
        mkt, record.get("symbol"), provider_name)
    if capabilities is None or not capabilities.lookup_by_client_order_id:
        return False
    try:
        venue_horizon = float(
            getattr(capabilities, "not_found_reliable_for_seconds", None))
        configured_horizon = float(oq.RETRY_NOT_FOUND_MAX_AGE_SEC)
        observed_at = float(now)
        created_at = float(record.get("created_ts") or 0.0)
        last_attempt_at = float(record.get("last_attempt_ts") or 0.0)
    except (TypeError, ValueError, OverflowError):
        return False
    reference_at = max(created_at, last_attempt_at)
    values = (
        venue_horizon, configured_horizon, observed_at, reference_at)
    if (
        not all(math.isfinite(value) for value in values)
        or venue_horizon <= 0
        or configured_horizon <= 0
        or reference_at <= 0
    ):
        return False
    age = observed_at - reference_at
    return 0 <= age <= min(venue_horizon, configured_horizon)


def _audit_event(record, event, **fields):
    payload = {
        "side": str(record.get("side") or "").lower() or None,
        "kind": record.get("kind"),
        "client_order_id": dict(record.get("place_kwargs") or {}).get(
            "client_order_id"),
        "order_id": record.get("order_id"),
        "revision": record.get("revision"),
    }
    payload.update(fields)
    _AUDIT.record(
        event,
        intent_id=str(record.get("intent_id") or f"outbox-{record.get('id')}"),
        venue=str(record.get("provider_name") or "routed"),
        symbol=str(record.get("symbol") or "unknown"),
        **payload,
    )


def process_once(mkt, now=None):
    """Advance each due outbox record by at most one non-blocking lifecycle step.

    Pending records are leased before placement and submitted at the current guarded
    price with ``caller_owns_retry=True``. Accepted records are status-polled once;
    open/partial orders stay tracked, fills are removed, and only explicit
    rejection/expiry can create a remainder revision. Return processing statistics.
    """
    now = now if now is not None else time.time()
    snapshot = oq.load_validated(now)

    to_retry = []       # due pending records; reconciliation precedes price gating
    to_observe = []     # accepted venue orders due for one status snapshot
    to_reconcile_cancel = []  # replacements blocked on the prior order status
    live_producer_reconciliation = []  # expired lease; lookup only, never submit
    expired = []
    skipped_price = 0   # due, but current price is still worse than requested
    quarantined = 0
    for r in snapshot:
        if oq.is_expired(r, now):
            expired.append(r)
            continue
        lifecycle = str(
            r.get("lifecycle") or "submit_pending").lower()
        symbol = r.get("symbol")
        if str(r.get("submission_state") or "").lower() == "producer_claimed":
            try:
                claim_until = float(r.get("claim_until") or 0.0)
            except (TypeError, ValueError, OverflowError):
                claim_until = 0.0
            if claim_until > float(now):
                continue
            owner_state = oq.producer_claim_owner_state(r)
            if owner_state == "alive":
                # A live producer keeps its exact claim. After lease expiry the
                # worker may only recover venue truth by deterministic lookup.
                live_producer_reconciliation.append(r)
                continue
            if owner_state not in {"dead", "mismatched"}:
                quarantined += 1
                _alert_producer_quarantine(
                    r, reason="producer_identity_unverifiable")
                continue
            provider_name = _provider_name(mkt, r)
            if not _supports_client_id_reconciliation(
                    mkt, symbol, provider_name):
                # The producer is gone, but a non-idempotent venue without
                # deterministic lookup still requires manual recovery.
                quarantined += 1
                _alert_producer_quarantine(
                    r, reason="lookup_unavailable")
                continue
        if not oq.is_due(r, now):
            continue
        if lifecycle == "accepted":
            to_observe.append(r)
            continue
        if lifecycle == "awaiting_cancel":
            to_reconcile_cancel.append(r)
            continue
        to_retry.append(r)

    # Expired work is never submitted. Retryable work remains persisted under a lease.
    expired = oq.discard_expired(expired, now)
    claimed = oq.claim(
        [r["id"] for r in to_retry]
        + [r["id"] for r in to_observe]
        + [r["id"] for r in to_reconcile_cancel],
        now)

    # Notification aggregation is independent of order lifecycle processing.
    # Empty passes also flush the final pending summary after the queue drains.
    _alert_giveups(expired, now)

    succeeded = 0
    reconciled = 0
    attempted = 0
    observed = 0
    filled = 0
    terminal_failed = 0
    terminal_retried = 0
    for r in live_producer_reconciliation:
        symbol = r.get("symbol")
        provider_name = _provider_name(mkt, r)
        client_order_id = dict(
            r.get("place_kwargs") or {}).get("client_order_id")
        reconciliation_state, existing_order = _reconcile_client_order(
            mkt, symbol, client_order_id, provider_name=provider_name
        )
        if reconciliation_state == _RECONCILE_FOUND:
            if oq.complete_claim(
                    r, "accepted", now, order=existing_order,
                    provider_name=provider_name):
                succeeded += 1
                reconciled += 1
                _audit_event(
                    r, "order_recovered",
                    order_id=oq.order_id_from_response(existing_order),
                    live_producer=True,
                )
                print(
                    f"[order_retry] RECONCILED live producer "
                    f"{r.get('side')} {symbol} "
                    f"orderId={oq.order_id_from_response(existing_order)}"
                )
            continue
        quarantine_reason = (
            "live_owner_not_found"
            if reconciliation_state == _RECONCILE_NOT_FOUND
            else "live_owner_lookup_unavailable"
        )
        quarantined += 1
        _alert_producer_quarantine(r, reason=quarantine_reason)
        _audit_event(
            r, "producer_reconciliation_deferred",
            reconciliation_state=reconciliation_state,
            quarantine_reason=quarantine_reason,
        )

    for r in claimed:
        lifecycle = str(r.get("lifecycle") or "submit_pending").lower()
        symbol = r.get("symbol")
        provider_name = _provider_name(mkt, r)
        if lifecycle == "awaiting_cancel":
            replaced_order_id = str(r.get("replaces_order_id") or "")
            try:
                status = mkt.order_status(
                    symbol, replaced_order_id, provider_name=provider_name)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[order_retry] replaced-order status unavailable "
                    f"{symbol}/{replaced_order_id}: {exc}")
                oq.complete_claim(
                    r, "status_error", now, failure_reason=str(exc))
                _audit_event(
                    r, "replacement_cancel_status_error",
                    replaced_order_id=replaced_order_id,
                    error_type=exc.__class__.__name__, error=str(exc))
                continue
            venue_status = str(status.venue_status or "").upper()
            canceled = venue_status in {"CANCELED", "CANCELLED"}
            if canceled:
                try:
                    original_qty = float(r.get("replaces_original_qty"))
                    replacement_qty = max(
                        0.0, original_qty - float(status.filled_qty or 0.0))
                except (TypeError, ValueError, OverflowError):
                    replacement_qty = -1.0
                activation = oq.activate_claimed_replacement(
                    r, replacement_qty, now)
                if activation == "activated":
                    oq.complete_claim(r, "release", now)
                    _audit_event(
                        r, "replacement_cancel_confirmed",
                        replaced_order_id=replaced_order_id,
                        replacement_qty=replacement_qty)
                elif activation == "resolved":
                    _audit_event(
                        r, "replacement_not_required",
                        replaced_order_id=replaced_order_id,
                        status=status.status, venue_status=venue_status,
                        filled_qty=status.filled_qty)
                else:
                    oq.complete_claim(
                        r, "status_error", now,
                        failure_reason="cancel_activation_failed")
                continue
            if status.terminal:
                oq.complete_claim(r, "success", now)
                _audit_event(
                    r, "replacement_not_required",
                    replaced_order_id=replaced_order_id,
                    status=status.status, venue_status=venue_status,
                    filled_qty=status.filled_qty)
                continue
            # The prior order remains live. Keep the replacement non-submittable
            # and recheck it on a later worker pass.
            oq.complete_claim(r, "release", now)
            continue
        if lifecycle == "accepted":
            observed += 1
            try:
                status = mkt.order_status(
                    symbol, str(r.get("order_id")), provider_name=provider_name)
            except Exception as exc:  # noqa: BLE001 — status ambiguity stays tracked.
                print(
                    f"[order_retry] status unavailable {symbol}/{r.get('order_id')}: {exc}")
                oq.complete_claim(
                    r, "status_error", now, failure_reason=str(exc))
                _audit_event(
                    r, "status_error", error_type=exc.__class__.__name__,
                    error=str(exc))
                continue
            transition = oq.advance_claimed_status(r, status, now)
            if transition.action == "observed":
                if transition.status_changed:
                    _audit_event(
                        r, "order_status", status=status.status,
                        venue_status=status.venue_status,
                        filled_qty=status.filled_qty, cost=status.cost,
                        fee=status.fee)
                continue
            if transition.action == "filled":
                filled += 1
                _audit_event(
                    r, "order_terminal", status=status.status,
                    venue_status=status.venue_status,
                    filled_qty=status.filled_qty, cost=status.cost,
                    fee=status.fee)
                print(
                    f"[order_retry] FILLED {r.get('side')} {symbol} "
                    f"orderId={r.get('order_id')} qty={status.filled_qty}")
                continue
            if transition.action == "retry_terminal":
                terminal_retried += 1
                _audit_event(
                    r, "order_terminal_retry", status=status.status,
                    venue_status=status.venue_status,
                    filled_qty=status.filled_qty, cost=status.cost,
                    fee=status.fee,
                    remaining_qty=transition.remaining_qty)
                print(
                    f"[order_retry] {status.venue_status or status.status} {symbol} "
                    f"orderId={r.get('order_id')} filled={status.filled_qty} "
                    f"remainder={transition.remaining_qty}")
                continue
            terminal_failed += 1
            _audit_event(
                r, "order_terminal", status=status.status,
                venue_status=status.venue_status,
                filled_qty=status.filled_qty, cost=status.cost,
                fee=status.fee, retry=False)
            _alert_terminal(r, status)
            continue

        kwargs = dict(r.get("place_kwargs") or {})
        client_order_id = kwargs.get("client_order_id")
        if _submission_requires_reconciliation(r):
            reconciliation_state, existing_order = _reconcile_client_order(
                mkt, symbol, client_order_id,
                provider_name=provider_name)
            if reconciliation_state == _RECONCILE_FOUND:
                succeeded += 1
                reconciled += 1
                oq.complete_claim(
                    r, "accepted", now, order=existing_order,
                    provider_name=provider_name)
                _audit_event(
                    r, "order_recovered",
                    order_id=oq.order_id_from_response(existing_order))
                print(f"[order_retry] RECONCILED {r.get('side')} {symbol} "
                      f"orderId={oq.order_id_from_response(existing_order)}")
                continue
            absence_reliable = (
                reconciliation_state == _RECONCILE_NOT_FOUND
                and _not_found_is_reliable(
                    mkt, r, now, provider_name))
            if not absence_reliable:
                quarantine_reason = (
                    "not_found_outside_reliable_horizon"
                    if reconciliation_state == _RECONCILE_NOT_FOUND
                    else "lookup_unavailable")
                quarantined += 1
                _alert_reconciliation_quarantine(
                    r, reason=quarantine_reason)
                oq.complete_claim(r, "release", now)
                _audit_event(
                    r, "reconciliation_deferred",
                    failure_reason=r.get("last_failure_reason"),
                    reconciliation_state=reconciliation_state,
                    quarantine_reason=quarantine_reason)
                continue
        # Only a known pre-submit refusal or confirmed client-ID absence can
        # reach the price gate. Reconciliation must not depend on current price:
        # an already accepted venue order remains real after the quote moves.
        try:
            price = mkt.get_current_price(symbol)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[order_retry] price unavailable {symbol} ({exc}); "
                "skipping, it stays queued")
            oq.complete_claim(r, "release", now)
            continue
        if price is None:
            oq.complete_claim(r, "release", now)
            continue
        if not oq.price_gate_ok(r, price):
            oq.complete_claim(r, "release", now)
            skipped_price += 1
            continue
        kwargs["caller_owns_retry"] = True
        kwargs["_retry_requested_price"] = r.get("requested_price")
        kwargs["_retry_price_tolerance"] = oq.RETRY_PRICE_TOL
        if provider_name is not None:
            kwargs["provider_name"] = provider_name
        outcome_context = {}
        kwargs["_outcome_context"] = outcome_context
        dispatch_claim = oq.begin_claimed_submit(r, now=now)
        if dispatch_claim is None:
            quarantined += 1
            _audit_event(
                r, "submit_claim_lost",
                failure_reason="exact_dispatch_claim_unavailable")
            continue
        r = dispatch_claim
        attempted += 1
        try:
            _audit_event(
                r, "submit_requested", qty=r.get("qty"), price=price,
                attempt=int(r.get("attempts") or 0) + 1)
            order = mkt.place(symbol, r.get("side"), price, r.get("qty"), **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[order_retry] retry {r.get('side')} {symbol} raised ({e}) — treating it as a failure")
            order = None
        accepted = _accepted_order(order)
        if not accepted:
            reconciliation_state, recovered = _reconcile_client_order(
                mkt, symbol, client_order_id,
                provider_name=provider_name)
            if reconciliation_state == _RECONCILE_FOUND:
                order = recovered
                accepted = True
                reconciled += 1
        if accepted:
            succeeded += 1
            oq.complete_claim(
                r, "accepted", now, order=order,
                provider_name=provider_name)
            _audit_event(
                r, "submit_accepted",
                order_id=oq.order_id_from_response(order),
                qty=r.get("qty"), price=price)
            print(f"[order_retry] ACCEPTED {r.get('side')} {symbol} @ {price} "
                  f"orderId={oq.order_id_from_response(order)}")
        else:
            refusal_reason = outcome_context.get("reason")
            submission_state = outcome_context.get("state") or "unknown"
            if not refusal_reason:
                refusal_reason = (
                    "pre_submit_refused"
                    if submission_state == "refused"
                    else "submit_ambiguous")
            if ((refusal_reason in {"execution_disabled", "trading_disabled"}
                    or oq.is_terminal_filter_refusal(refusal_reason))
                    and submission_state == "refused"):
                # A durable dry or venue-invalid intent cannot become executable
                # unchanged. Remove it under the worker's exact claim.
                oq.complete_claim(r, "success", now)
                terminal_failed += 1
            elif oq.is_non_failure_deferral(refusal_reason):
                # Pre-submit freshness and trend gates defer without consuming an
                # attempt or TTL, then retry at the normal worker cadence.
                oq.complete_claim(r, "deferred", now,
                                  failure_reason=refusal_reason,
                                  submission_state=submission_state)
            else:
                oq.complete_claim(r, "failure", now,
                                  failure_reason=refusal_reason,
                                  submission_state=submission_state)
            _audit_event(
                r, "submit_refused" if submission_state == "refused" else "submit_unknown",
                qty=r.get("qty"), price=price,
                submission_state=submission_state,
                failure_reason=refusal_reason)

    return {"attempted": attempted,
            "succeeded": succeeded,
            "reconciled": reconciled,
            "observed": observed,
            "filled": filled,
            "terminal_failed": terminal_failed,
            "terminal_retried": terminal_retried,
            "expired": len(expired),
            "skipped_price": skipped_price,
            "quarantined": quarantined,
            "remaining": len(oq.load_all(now))}


def _alert_queue_corruption(exc):
    """Alert at most once per distinct queue corruption in this process."""
    fingerprint = str(
        getattr(exc, "fingerprint", "") or f"{type(exc).__name__}:{exc}"
    )
    if fingerprint in _ALERTED_QUEUE_CORRUPTIONS:
        return False
    _ALERTED_QUEUE_CORRUPTIONS.add(fingerprint)
    try:
        alert.notify(
            title="🛑 order-retry queue corruption",
            body=str(exc),
            source="order_retry")
    except Exception as alert_exc:  # noqa: BLE001
        print(f"[order_retry] queue-corruption alert failed (ignored): {alert_exc}")
    return True


def _alert_producer_quarantine(rec, *, reason="lookup_unavailable"):
    """Alert once when an expired producer claim cannot be reconciled safely."""
    fingerprint = (
        str(rec.get("id")), int(rec.get("revision", 0)),
        str(rec.get("provider_name") or "routed"),
        str(reason),
    )
    if fingerprint in _ALERTED_PRODUCER_QUARANTINES:
        return False
    _ALERTED_PRODUCER_QUARANTINES.add(fingerprint)
    try:
        alert.notify(
            title=(f"🛑 order producer claim quarantined "
                   f"{rec.get('side')} {rec.get('symbol')}"),
            body=(
                "The producer lease expired, but its PID/start identity cannot "
                "be verified as either dead or replaced. Automatic recovery is "
                "blocked; inspect the producer and outbox record manually."
                if reason == "producer_identity_unverifiable" else
                (
                    "The producer lease expired while its exact process identity "
                    "is still alive. The venue lookup found no matching client "
                    "order, so the record remains claimed for manual review; the "
                    "worker will never resubmit it."
                    if reason == "live_owner_not_found" else
                    "The producer lease expired while its exact process identity "
                    "is still alive, but deterministic venue lookup is unavailable. "
                    "The record remains claimed for manual review; the worker will "
                    "never resubmit it."
                )
                if reason.startswith("live_owner_") else
                "The producer is no longer alive and this provider cannot "
                "reconcile the deterministic client order ID. Automatic "
                "submission is blocked; inspect the venue and outbox manually."
            ),
            source="order_retry", symbol=str(rec.get("symbol")))
    except Exception as exc:  # noqa: BLE001
        print(f"[order_retry] producer-quarantine alert failed (ignored): {exc}")
    return True


def _alert_reconciliation_quarantine(rec, *, reason="lookup_unavailable"):
    """Alert once when a possibly submitted intent cannot be reconciled."""
    fingerprint = (
        str(rec.get("id")), int(rec.get("revision", 0)),
        str(rec.get("provider_name") or "routed"),
        str(rec.get("last_failure_reason") or "unknown"),
        str(reason),
    )
    if fingerprint in _ALERTED_RECONCILIATION_QUARANTINES:
        return False
    _ALERTED_RECONCILIATION_QUARANTINES.add(fingerprint)
    try:
        alert.notify(
            title=(f"🛑 order submission quarantined "
                   f"{rec.get('side')} {rec.get('symbol')}"),
            body=(
                "The venue reports no matching client order, but this intent is "
                "older than the bounded interval in which absence is reliable. "
                "Automatic submission and expiry are blocked; inspect the venue "
                "and outbox manually."
                if reason == "not_found_outside_reliable_horizon"
                else
                "This intent may already exist at the venue, but deterministic "
                "client-ID reconciliation is unavailable. Automatic submission "
                "and expiry are blocked; inspect the venue and outbox manually."
            ),
            source="order_retry", symbol=str(rec.get("symbol")))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[order_retry] reconciliation-quarantine alert failed (ignored): {exc}")
    return True


def _alert_terminal(rec, status):
    """Alert on a canceled intent whose remainder is deliberately not replayed."""
    try:
        alert.notify(
            title=f"🛑 order terminal {rec.get('side')} {rec.get('symbol')}",
            body=(f"Order {rec.get('order_id')} became "
                  f"{status.venue_status or status.status}; filled={status.filled_qty}, "
                  "the rest is NOT resent automatically after a cancel."),
            source="order_retry", symbol=str(rec.get("symbol")))
    except Exception as exc:  # noqa: BLE001
        print(f"[order_retry] the terminal alert failed (ignored): {exc}")


def _alert_giveups(records, now):
    """Audit each expiry, but notify once per incident and summary interval."""
    try:
        events = []
        for rec in records:
            reason = ("attempt_limit" if oq.RETRY_MAX_ATTEMPTS > 0
                      and rec.get("attempts", 0) >= oq.RETRY_MAX_ATTEMPTS
                      else "active_ttl")
            _audit_event(
                rec, "retry_giveup", expiry_reason=reason, expired_ts=now,
                qty=rec.get("qty"), attempts=rec.get("attempts"),
                requested_price=rec.get("requested_price"),
                last_attempt_ts=rec.get("last_attempt_ts"),
                created_ts=rec.get("created_ts"),
                ttl_started_ts=rec.get("ttl_started_ts"),
                last_failure_reason=rec.get("last_failure_reason"))
            events.append({
                "labels": {
                    "provider": str(rec.get("provider_name") or "routed"),
                    "symbol": str(rec.get("symbol")), "side": str(rec.get("side")),
                    "kind": str(rec.get("kind") or "unspecified"),
                    "reason": str(rec.get("last_failure_reason") or "unspecified"),
                    "expiry": reason,
                },
                "quantity": rec["qty"],
                "sample_id": rec.get("intent_id") or rec["id"],
            })
        digest = IncidentDigest(
            os.path.join(os.path.dirname(oq.QUEUE_FILE), "order_retry_giveup_alerts.json"),
            GIVEUP_SUMMARY_SEC)
        for report in digest.collect(events, now=now):
            labels = report["labels"]
            suffix = " summary" if report["reported"] else ""
            try:
                alert.notify(
                    title=f"🛑 order-retry GIVE UP{suffix} {labels['side']} {labels['symbol']}",
                    body=(f"{report['count']} intent(s) expired; no order submitted for these expiries. "
                          f"Total requested qty={report['quantity']:.8g}. "
                          f"Provider={labels['provider']}; kind={labels['kind']}; "
                          f"last refusal={labels['reason']}; expiry={labels['expiry']}. "
                          f"Further expiries for this incident are summarized every "
                          f"{GIVEUP_SUMMARY_SEC / 60:g} minutes. "
                          f"Per-intent details: execution audit; sample={report['sample_id']}."),
                    source="order_retry", symbol=labels["symbol"])
            except Exception as exc:  # One transport failure must not skip other incidents.
                print(f"[order_retry] the giveup delivery failed (ignored): {exc}")
    except Exception as exc:  # Notification failure must not change order processing.
        print(f"[order_retry] the giveup digest failed (ignored): {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="a single step, then exit")
    args = ap.parse_args()

    if args.once:
        if not oq.RETRY_ENABLED:
            print("[order_retry] RETRY_ENABLED=false — nothing to do (--once).")
            return 0
        from providers.market_api import api as mkt
        try:
            print(f"[order_retry] single pass: {process_once(mkt)}")
            return 0
        except oq.RetryQueueCorruptionError as exc:
            _alert_queue_corruption(exc)
            print(f"[order_retry] queue corruption; pass aborted: {exc}")
            return 2

    single_instance("order_retry_worker")

    # When the kill switch is disabled, stay alive instead of exiting; otherwise
    # flota_start repeatedly restarts the worker, causing flapping and phone alerts.
    # Re-enabling already requires a process restart because RETRY_ENABLED is read at
    # import, so remain idle until the next fleet restart without spamming supervision.
    if not oq.RETRY_ENABLED:
        print("[order_retry] RETRY_ENABLED=false — worker ALIVE but inactive (idle).")
        while True:
            time.sleep(60)

    from providers.market_api import api as mkt

    try:
        consolidated = oq.consolidate_deferred_streams()
        if consolidated:
            print(
                f"[order_retry] startup consolidation removed {len(consolidated)} "
                "superseded trend-deferred record(s)")
    except oq.RetryQueueCorruptionError as exc:
        _alert_queue_corruption(exc)
        print(f"[order_retry] startup consolidation aborted: {exc}")

    print(f"[order_retry] start (poll={WORKER_POLL_SEC:.0f}s, interval={oq.RETRY_INTERVAL_SEC:.0f}s, "
          f"TTL={oq.RETRY_TTL_SEC:.0f}s)")
    while True:
        try:
            stats = process_once(mkt)
            if stats["attempted"] or stats["expired"]:
                print(f"[order_retry] {stats}")
        except oq.RetryQueueCorruptionError as exc:
            _alert_queue_corruption(exc)
            print(f"[order_retry] queue corruption; pass aborted: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[order_retry] pass error (continuing): {exc}")
        time.sleep(WORKER_POLL_SEC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Persistent incident aggregation, independent of venue and delivery transport.

The first batch is due immediately. Further events are summarized once per
interval, including on an empty poll. Reservations precede delivery: this limits
restart duplicates but is not a guaranteed-delivery outbox.
"""
import json
import math
import os

from lock import FileLock
from state_io import atomic_write_json, load_json_state


class IncidentDigest:
    def __init__(self, path, interval_seconds):
        self.path = os.fspath(path)
        self.interval = float(interval_seconds)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("digest interval must be finite and positive")

    def collect(self, events, *, now):
        """Return due summaries of events with labels, quantity and sample ID.

        Labels define an incident. Store counts and one sample, not an unbounded
        list of IDs; detailed records belong in the caller's existing audit.
        Invalid state raises instead of resetting cooldowns and causing a storm.
        """
        if not math.isfinite(now) or now <= 0:
            raise ValueError("digest time must be finite and positive")
        if not events and not os.path.exists(self.path):
            return []
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with FileLock(self.path + ".lock"):
            state = load_json_state(
                self.path, default_factory=lambda: {"version": 1, "groups": {}},
                fail_closed=True, label="notification digest")
            if state.get("version") != 1 or not isinstance(state.get("groups"), dict):
                raise ValueError("invalid notification digest schema")
            groups = state["groups"]
            changed = False
            for key, group in list(groups.items()):
                if (not isinstance(group["count"], int) or group["count"] < 0
                        or not math.isfinite(group["next_at"]) or group["next_at"] <= 0
                        or not math.isfinite(group["quantity"]) or group["quantity"] < 0):
                    raise ValueError("invalid notification digest group")
                if group["count"] == 0 and group["next_at"] <= now:
                    del groups[key]
                    changed = True
            for event in events:
                labels = event["labels"]
                quantity = float(event["quantity"])
                if (not isinstance(labels, dict) or not labels
                        or not all(isinstance(k, str) and isinstance(v, str)
                                   for k, v in labels.items())
                        or not math.isfinite(quantity) or quantity < 0):
                    raise ValueError("invalid notification digest event")
                key = json.dumps(labels, sort_keys=True, separators=(",", ":"))
                group = groups.setdefault(key, {
                    "labels": labels, "next_at": now, "reported": False,
                    "count": 0, "quantity": 0.0,
                })
                if group["count"] == 0:
                    group["first_ts"] = now
                    group["sample_id"] = str(event["sample_id"])
                group["last_ts"] = now
                group["count"] += 1
                group["quantity"] += quantity
                changed = True
            due = []
            for group in groups.values():
                if group["count"] and group["next_at"] <= now:
                    due.append(dict(group))
                    group.update(next_at=now + self.interval, reported=True,
                                 count=0, quantity=0.0)
                    changed = True
            if changed:
                atomic_write_json(self.path, state, separators=(",", ":"))
            return due
