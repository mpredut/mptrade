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
import json
import math
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import order_retry as oq
import alertnotifiers as alert
from botcore import _parse_bool_flag, required_float_env, single_instance
from cachepaths import split_symbol_assets
from lock import FileLock
from providers.base import normalize_side
from providers.execution_audit import ExecutionAudit
from providers.quantity import resolve_assets
from state_io import (
    atomic_write_json,
    load_json_state,
    previous_generation_path,
    save_json_state,
)


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
class _PersistentQuarantineSet(set):
    """In-memory set backed by a JSON state file next to oq.QUEUE_FILE so quarantine
    alerts are deduplicated across worker process restarts."""

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._loaded_path: str | None = None

    def _path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(oq.QUEUE_FILE)),
            "order_retry_quarantine_alerts.json",
        )

    def sync_from_disk(self) -> None:
        path = self._path()
        if self._loaded_path == path:
            return
        self._loaded_path = path
        try:
            data = load_json_state(
                path, default_factory=dict, fail_closed=False, label="quarantine alerts"
            )
            if isinstance(data, dict):
                items = data.get(self._key, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, list):
                            super().add(tuple(item))
        except Exception:
            pass

    def persist_to_disk(self) -> None:
        path = self._path()
        self._loaded_path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {}
            loaded = load_json_state(
                path, default_factory=dict, fail_closed=False, label="quarantine alerts"
            )
            if isinstance(loaded, dict):
                data = dict(loaded)
            data[self._key] = [
                list(x) if isinstance(x, tuple) else [x]
                for x in sorted(
                    self,
                    key=lambda item: tuple(str(e) for e in item) if isinstance(item, tuple) else (str(item),),
                )
            ]
            save_json_state(path, data, separators=(",", ":"))
        except Exception:
            pass

    def __contains__(self, element) -> bool:
        self.sync_from_disk()
        return super().__contains__(element)

    def add(self, element) -> None:
        self.sync_from_disk()
        super().add(element)
        self.persist_to_disk()

    def clear(self) -> None:
        super().clear()
        path = self._path()
        self._loaded_path = path
        if os.path.exists(path) or os.path.exists(previous_generation_path(path)):
            try:
                loaded = load_json_state(
                    path, default_factory=dict, fail_closed=False, label="quarantine alerts"
                )
                if isinstance(loaded, dict) and self._key in loaded:
                    loaded.pop(self._key, None)
                    save_json_state(path, loaded, separators=(",", ":"))
            except Exception:
                pass


_ALERTED_QUEUE_CORRUPTIONS = set()
_ALERTED_PRODUCER_QUARANTINES = _PersistentQuarantineSet("producer")
_ALERTED_RECONCILIATION_QUARANTINES = _PersistentQuarantineSet("reconciliation")
_ALERTED_TERMINALS = _PersistentQuarantineSet("terminal")


def _normalize_bypass_set(raw_bg) -> set[str]:
    if isinstance(raw_bg, str):
        return {s.strip() for s in raw_bg.split(",") if s.strip()}
    if isinstance(raw_bg, (list, tuple, set, frozenset)):
        return {
            str(s).strip()
            for s in raw_bg
            if s is not None and not isinstance(s, bool) and str(s).strip()
        }
    return set()


def _safe_int(val, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return default
    try:
        vf = float(val)
        return int(vf) if math.isfinite(vf) else default
    except (TypeError, ValueError, OverflowError):
        return default


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
    configured = record.get("provider_name") if isinstance(record, dict) else None
    if configured is not None and not isinstance(configured, bool) and str(configured).strip():
        return str(configured).strip()
    resolver = getattr(mkt, "provider_name_for", None)
    if callable(resolver) and isinstance(record, dict):
        try:
            resolved = resolver(record.get("symbol"))
            if resolved is not None and not isinstance(resolved, bool) and str(resolved).strip():
                return str(resolved).strip()
        except Exception:
            pass
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


def _not_found_reliability_status(mkt, record, now, provider_name):
    """Return whether venue absence is recent enough, and the specific failure reason."""
    capabilities = _reconciliation_capabilities(
        mkt, record.get("symbol"), provider_name)
    if capabilities is None or not capabilities.lookup_by_client_order_id:
        return False, "lookup_unavailable"
    raw_venue_horizon = getattr(capabilities, "not_found_reliable_for_seconds", None)
    if raw_venue_horizon is None or isinstance(raw_venue_horizon, bool):
        return False, "missing_reconciliation_horizon"
    if isinstance(now, bool) or isinstance(record.get("created_ts"), bool) or isinstance(record.get("last_attempt_ts"), bool):
        return False, "invalid_reference_timestamp"
    try:
        venue_horizon = float(raw_venue_horizon)
        configured_horizon = float(oq.RETRY_NOT_FOUND_MAX_AGE_SEC)
        observed_at = float(now)
        created_at = float(record.get("created_ts") or 0.0)
        last_attempt_at = float(record.get("last_attempt_ts") or 0.0)
    except (TypeError, ValueError, OverflowError):
        return False, "invalid_reference_timestamp"
    if not math.isfinite(venue_horizon) or venue_horizon <= 0:
        return False, "missing_reconciliation_horizon"
    if not math.isfinite(configured_horizon) or configured_horizon <= 0:
        return False, "missing_reconciliation_horizon"
    reference_at = max(created_at, last_attempt_at)
    if not math.isfinite(reference_at) or reference_at <= 0 or not math.isfinite(observed_at):
        return False, "invalid_reference_timestamp"
    age = observed_at - reference_at
    if age < 0:
        return False, "future_reference_timestamp"
    if age > min(venue_horizon, configured_horizon):
        return False, "not_found_outside_reliable_horizon"
    return True, None


def _not_found_is_reliable(mkt, record, now, provider_name):
    """Return whether this venue absence is recent enough to authorize submit."""
    reliable, _ = _not_found_reliability_status(mkt, record, now, provider_name)
    return reliable


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
    now = oq._clean_now_sec(now)
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
                raw_cu = r.get("claim_until")
                claim_until = 0.0 if isinstance(raw_cu, bool) else float(raw_cu or 0.0)
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
            if oq.order_qty_from_response(existing_order) is None and hasattr(mkt, "order_status"):
                try:
                    oid = oq.order_id_from_response(existing_order)
                    if oid:
                        st = mkt.order_status(symbol, oid, provider_name=provider_name)
                        if st and getattr(st, "orig_qty", None):
                            existing_order = dict(existing_order)
                            existing_order["orig_qty"] = st.orig_qty
                except Exception:
                    pass
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
                raw_orig_qty = r.get("replaces_original_qty")
                raw_filled_qty = getattr(status, "filled_qty", 0.0)
                if isinstance(raw_orig_qty, bool) or isinstance(raw_filled_qty, bool):
                    replacement_qty = -1.0
                else:
                    try:
                        original_qty = float(raw_orig_qty)
                        filled_qty_val = float(raw_filled_qty or 0.0)
                        if (
                            math.isfinite(original_qty)
                            and original_qty > 0
                            and math.isfinite(filled_qty_val)
                            and filled_qty_val >= 0
                        ):
                            replacement_qty = max(0.0, original_qty - filled_qty_val)
                        else:
                            replacement_qty = -1.0
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
            if not transition.status_changed:
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
                if oq.order_qty_from_response(existing_order) is None and hasattr(mkt, "order_status"):
                    try:
                        oid = oq.order_id_from_response(existing_order)
                        if oid:
                            st = mkt.order_status(symbol, oid, provider_name=provider_name)
                            if st and getattr(st, "orig_qty", None):
                                existing_order = dict(existing_order)
                                existing_order["orig_qty"] = st.orig_qty
                    except Exception:
                        pass
                oq.complete_claim(
                    r, "accepted", now, order=existing_order,
                    provider_name=provider_name)
                _audit_event(
                    r, "order_recovered",
                    order_id=oq.order_id_from_response(existing_order))
                print(f"[order_retry] RECONCILED {r.get('side')} {symbol} "
                      f"orderId={oq.order_id_from_response(existing_order)}")
                continue
            if reconciliation_state == _RECONCILE_NOT_FOUND:
                absence_reliable, not_found_reason = _not_found_reliability_status(
                    mkt, r, now, provider_name)
            else:
                absence_reliable, not_found_reason = False, "lookup_unavailable"
            if not absence_reliable:
                quarantine_reason = not_found_reason or "lookup_unavailable"
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
            if provider_name is not None and hasattr(mkt, "get_current_price"):
                try:
                    price = mkt.get_current_price(symbol, provider_name=provider_name)
                except TypeError:
                    price = mkt.get_current_price(symbol)
            else:
                price = mkt.get_current_price(symbol)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[order_retry] price unavailable {symbol} ({exc}); "
                "skipping, it stays queued")
            oq.complete_claim(r, "release", now)
            continue
        if price is None or isinstance(price, bool):
            oq.complete_claim(r, "release", now)
            continue
        try:
            price_f = float(price)
        except (TypeError, ValueError, OverflowError):
            oq.complete_claim(r, "release", now)
            continue
        if not math.isfinite(price_f) or price_f <= 0:
            oq.complete_claim(r, "release", now)
            continue
        if not oq.price_gate_ok(r, price_f):
            oq.complete_claim(r, "release", now)
            skipped_price += 1
            continue
        kwargs["caller_owns_retry"] = True
        kwargs["_retry_requested_price"] = r.get("requested_price")
        kwargs["_retry_price_tolerance"] = oq.RETRY_PRICE_TOL
        if provider_name is not None:
            kwargs["provider_name"] = provider_name

        # Age-staged retry progression across active TTL (RETRY_TTL_SEC):
        # - Stage 1 (Fresh, 0% <= progress < 50%): Standard configured profit margin.
        # - Stage 2 (Breakeven, 50% <= progress < 75%): Relax profit margin to fee buffer (0.15%).
        # - Stage 3 (Controlled Salvage, progress >= 75%): For SELL, allow math_reference bypass
        #           while strictly maintaining wait_for_trend=True and P2 whale/orderbook protections.
        try:
            raw_ttl_s = r.get("ttl_started_ts")
            if raw_ttl_s is None or isinstance(raw_ttl_s, bool) or not raw_ttl_s:
                raw_ttl_s = r.get("created_ts")
            if raw_ttl_s is None or isinstance(raw_ttl_s, bool) or not raw_ttl_s:
                raw_ttl_s = now
            ttl_started = float(raw_ttl_s)
        except (TypeError, ValueError, OverflowError):
            ttl_started = now
        if not math.isfinite(ttl_started) or ttl_started <= 0:
            ttl_started = now
        if oq.is_non_failure_deferral(r.get("last_failure_reason")):
            age_sec = 0.0
        else:
            age_sec = max(0.0, now - ttl_started)
        effective_ttl = (
            oq.resolve_record_ttl(r)
            if hasattr(oq, "resolve_record_ttl")
            else float(oq.RETRY_TTL_SEC)
        )
        total_ttl = (
            effective_ttl
            if (isinstance(effective_ttl, (int, float)) and not isinstance(effective_ttl, bool) and math.isfinite(effective_ttl) and effective_ttl > 0)
            else 302400.0
        )
        age_progress = min(1.0, max(0.0, age_sec / total_ttl))
        side_u = normalize_side(r.get("side")) or str(r.get("side") or "").upper()

        retry_stage = "standard"
        if r.get("kind") == "monitor_order_replace":
            existing_bypasses = _normalize_bypass_set(kwargs.get("bypass_guards"))
            existing_bypasses.update({"G_PROF", "math_reference"})
            kwargs["bypass_guards"] = sorted(list(existing_bypasses))
            if side_u == "SELL":
                kwargs["allow_math_reference_sell"] = True
            kwargs["wait_for_trend"] = True
        else:
            is_stop_expired_timer = (
                str(r.get("kind") or kwargs.get("kind") or "").strip().upper() == "STOP"
                and not _parse_bool_flag(kwargs.get("wait_for_trend"), True)
            )
            kwargs["wait_for_trend"] = False if is_stop_expired_timer else True
            cur_pm = kwargs.get("profit_margin")
            try:
                cur_pm_f = (
                    float(cur_pm)
                    if (cur_pm is not None and not isinstance(cur_pm, bool))
                    else None
                )
            except (TypeError, ValueError, OverflowError):
                cur_pm_f = None
            if age_progress >= 0.75:
                retry_stage = "salvage"
                if side_u == "SELL":
                    existing_bypasses = _normalize_bypass_set(kwargs.get("bypass_guards"))
                    existing_bypasses.update({"G_PROF", "math_reference"})
                    kwargs["bypass_guards"] = sorted(list(existing_bypasses))
                    kwargs["allow_math_reference_sell"] = True
                if cur_pm_f is None or not math.isfinite(cur_pm_f) or cur_pm_f > 0.15:
                    kwargs["profit_margin"] = 0.15
            elif age_progress >= 0.50:
                retry_stage = "breakeven"
                if cur_pm_f is None or not math.isfinite(cur_pm_f) or cur_pm_f > 0.15:
                    kwargs["profit_margin"] = 0.15

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
        kwargs["_retry_claim"] = dispatch_claim
        attempted += 1
        try:
            _audit_event(
                r, "submit_requested", qty=r.get("qty"), price=price,
                attempt=_safe_int(r.get("attempts"), 0) + 1,
                retry_stage=retry_stage,
                age_progress=round(age_progress, 3))
            if retry_stage != "standard":
                print(f"[order_retry] {side_u} {symbol} retrying under Stage {retry_stage.upper()} "
                      f"(age={age_sec / 3600:.1f}h, progress={age_progress * 100:.1f}%)")
            order = mkt.place(symbol, r.get("side"), price, r.get("qty"), **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[order_retry] retry {r.get('side')} {symbol} raised ({e}) — treating it as a failure")
            order = None
        submitted_qty = outcome_context.get("submitted_qty")
        if submitted_qty is not None and not isinstance(submitted_qty, bool):
            try:
                sq = float(submitted_qty)
                if math.isfinite(sq) and sq > 0:
                    r["qty"] = sq
                    r["requested_qty_total"] = sq
            except (TypeError, ValueError, OverflowError):
                pass
        accepted = _accepted_order(order)
        if not accepted and outcome_context.get("state") != "refused":
            reconciliation_state, recovered = _reconcile_client_order(
                mkt, symbol, client_order_id,
                provider_name=provider_name)
            if reconciliation_state == _RECONCILE_FOUND:
                order = recovered
                if oq.order_qty_from_response(order) is None and hasattr(mkt, "order_status"):
                    try:
                        oid = oq.order_id_from_response(order)
                        if oid:
                            st = mkt.order_status(symbol, oid, provider_name=provider_name)
                            if st and getattr(st, "orig_qty", None):
                                order = dict(order)
                                order["orig_qty"] = st.orig_qty
                    except Exception:
                        pass
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
            if ((refusal_reason in {
                    "execution_disabled", "trading_disabled",
                    "aborted_by_caller", "invalid_price"}
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


def _resolve_display_assets(symbol):
    """Return (base_asset, quote_asset) for human-readable alert formatting."""
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        return "?", "USD"
    sym = symbol.strip().upper()
    split = split_symbol_assets(symbol)
    if split is not None:
        base, quote = split
        return (base or sym), (quote or "USD")
    base, quote = resolve_assets(sym)
    return (base or sym), (quote or "USD")


def _format_ts(ts):
    """Format a POSIX timestamp in local YYYY-MM-DD HH:MM:SS, or N/A."""
    if ts is None or isinstance(ts, bool):
        return "N/A"
    try:
        val = float(ts or 0.0)
        if not math.isfinite(val) or val <= 0:
            return "N/A"
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return "N/A"


def _format_record_context(rec, *, include_lease=False):
    """Build a structured summary of order time, base asset qty/price, and quote value."""
    raw_side = rec.get("side")
    side = raw_side.strip().upper() if isinstance(raw_side, str) and raw_side.strip() else "?"
    raw_sym = rec.get("symbol")
    symbol = raw_sym.strip() if isinstance(raw_sym, str) and raw_sym.strip() else "?"
    raw_prov = rec.get("provider_name")
    provider = raw_prov.strip().upper() if isinstance(raw_prov, str) and raw_prov.strip() else "ROUTED"
    base_asset, quote_asset = _resolve_display_assets(symbol)
    try:
        raw_qty = rec.get("qty")
        qty_val = float(raw_qty) if raw_qty is not None and not isinstance(raw_qty, bool) else 0.0
    except (TypeError, ValueError, OverflowError):
        qty_val = 0.0
    try:
        raw_req_px = rec.get("requested_price")
        req_px = (
            float(raw_req_px)
            if raw_req_px is not None and not isinstance(raw_req_px, bool)
            else None
        )
        if req_px is not None and (not math.isfinite(req_px) or req_px <= 0):
            req_px = None
    except (TypeError, ValueError, OverflowError):
        req_px = None
    try:
        raw_ref_px = rec.get("ref_price")
        ref_px = (
            float(raw_ref_px)
            if raw_ref_px is not None and not isinstance(raw_ref_px, bool)
            else None
        )
        if ref_px is not None and (not math.isfinite(ref_px) or ref_px <= 0):
            ref_px = None
    except (TypeError, ValueError, OverflowError):
        ref_px = None

    effective_px = req_px if req_px is not None else ref_px
    if req_px is not None and ref_px is not None and abs(req_px - ref_px) > 1e-9:
        price_str = f"@ ${req_px:.6g} (ref ${ref_px:.6g})"
    elif effective_px is not None:
        price_str = f"@ ${effective_px:.6g}"
    else:
        price_str = "@ market"

    if effective_px is not None and math.isfinite(qty_val) and qty_val > 0:
        notional = qty_val * effective_px
        value_str = f"~${notional:.2f} {quote_asset}"
    else:
        value_str = "N/A"

    created_str = _format_ts(rec.get("created_ts"))
    last_attempt_str = _format_ts(rec.get("last_attempt_ts"))
    raw_att = rec.get("attempts")
    try:
        attempts = int(raw_att) if raw_att is not None and not isinstance(raw_att, bool) else 0
    except (TypeError, ValueError, OverflowError):
        attempts = 0
    raw_reason = rec.get("last_failure_reason")
    last_reason = (
        str(raw_reason).strip()
        if raw_reason is not None and not isinstance(raw_reason, bool) and str(raw_reason).strip()
        else "none"
    )
    time_parts = [f"Created: {created_str}"]
    if last_attempt_str != "N/A":
        time_parts.append(f"Last attempt: {last_attempt_str}")
    if include_lease:
        lease_str = _format_ts(rec.get("claim_until"))
        if lease_str != "N/A":
            time_parts.append(f"Lease expired: {lease_str}")
    time_line = " | ".join(time_parts) + f" ({attempts} attempts, reason={last_reason})"
    raw_place_kw = rec.get("place_kwargs")
    place_kw = raw_place_kw if isinstance(raw_place_kw, dict) else {}
    client_order_id = (
        place_kw.get("client_order_id")
        or rec.get("order_id")
        or rec.get("id")
        or "N/A"
    )
    return (
        f"Order: {side} {qty_val:.6g} {base_asset} ({symbol} · {provider}) {price_str} | Value: {value_str}\n"
        f"Time: {time_line}\n"
        f"ID: {client_order_id}"
    )


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
        str(rec.get("id")), _safe_int(rec.get("revision"), 0),
        str(rec.get("provider_name") or "routed"),
        str(reason),
    )
    if hasattr(_ALERTED_PRODUCER_QUARANTINES, "sync_from_disk"):
        _ALERTED_PRODUCER_QUARANTINES.sync_from_disk()
    if fingerprint in _ALERTED_PRODUCER_QUARANTINES:
        return False
    _ALERTED_PRODUCER_QUARANTINES.add(fingerprint)
    detail_msg = (
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
    )
    try:
        context = _format_record_context(rec, include_lease=True)
        alert.notify(
            title=(f"🛑 order producer claim quarantined "
                   f"{rec.get('side')} {rec.get('symbol')}"),
            body=f"{context}\n{detail_msg}",
            source="order_retry", symbol=str(rec.get("symbol")))
    except Exception as exc:  # noqa: BLE001
        print(f"[order_retry] producer-quarantine alert failed (ignored): {exc}")
    return True


def _alert_reconciliation_quarantine(rec, *, reason="lookup_unavailable"):
    """Alert once when a possibly submitted intent cannot be reconciled."""
    fingerprint = (
        str(rec.get("id")), _safe_int(rec.get("revision"), 0),
        str(rec.get("provider_name") or "routed"),
        str(rec.get("last_failure_reason") or "unknown"),
        str(reason),
    )
    if hasattr(_ALERTED_RECONCILIATION_QUARANTINES, "sync_from_disk"):
        _ALERTED_RECONCILIATION_QUARANTINES.sync_from_disk()
    if fingerprint in _ALERTED_RECONCILIATION_QUARANTINES:
        return False
    _ALERTED_RECONCILIATION_QUARANTINES.add(fingerprint)
    if reason == "not_found_outside_reliable_horizon":
        detail_msg = (
            "The venue reports no matching client order, but this intent is "
            "older than the bounded interval in which absence is reliable. "
            "Automatic submission and expiry are blocked; inspect the venue "
            "and outbox manually."
        )
    elif reason == "missing_reconciliation_horizon":
        detail_msg = (
            "The venue reports no matching client order, but the provider does "
            "not declare a reliable reconciliation interval. Automatic submission "
            "and expiry are blocked; inspect the venue and outbox manually."
        )
    elif reason in {"invalid_reference_timestamp", "future_reference_timestamp"}:
        detail_msg = (
            "The venue reports no matching client order, but intent timestamps "
            "are invalid or in the future. Automatic submission and expiry are "
            "blocked; inspect the venue and outbox manually."
        )
    else:
        detail_msg = (
            "This intent may already exist at the venue, but deterministic "
            "client-ID reconciliation is unavailable. Automatic submission "
            "and expiry are blocked; inspect the venue and outbox manually."
        )
    try:
        context = _format_record_context(rec)
        alert.notify(
            title=(f"🛑 order submission quarantined "
                   f"{rec.get('side')} {rec.get('symbol')}"),
            body=f"{context}\n{detail_msg}",
            source="order_retry", symbol=str(rec.get("symbol")))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[order_retry] reconciliation-quarantine alert failed (ignored): {exc}")
    return True


def _alert_terminal(rec, status):
    """Alert on a canceled intent whose remainder is deliberately not replayed."""
    terminal_key = (
        str(rec.get("id") or ""),
        _safe_int(rec.get("revision"), 0),
        str(rec.get("order_id") or ""),
        str(status.venue_status or status.status or ""),
    )
    if terminal_key in _ALERTED_TERMINALS:
        return
    _ALERTED_TERMINALS.add(terminal_key)
    try:
        context = _format_record_context(rec)
        alert.notify(
            title=f"🛑 order terminal {rec.get('side')} {rec.get('symbol')}",
            body=(f"{context}\n"
                  f"Order {rec.get('order_id')} became "
                  f"{status.venue_status or status.status}; filled={status.filled_qty}, "
                  "the rest is NOT resent automatically after a cancel."),
            source="order_retry", symbol=str(rec.get("symbol")))
    except Exception as exc:  # noqa: BLE001
        print(f"[order_retry] the terminal alert failed (ignored): {exc}")


class IncidentDigest:
    """Persistent incident aggregation, independent of venue and delivery transport."""

    def __init__(self, path, interval_seconds):
        self.path = os.fspath(path)
        if isinstance(interval_seconds, bool) or interval_seconds is None:
            raise ValueError("digest interval must be finite and positive")
        try:
            self.interval = float(interval_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("digest interval must be finite and positive") from exc
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("digest interval must be finite and positive")

    def collect(self, events, *, now):
        """Return due summaries of events with labels, quantity and sample ID."""
        if isinstance(now, bool) or now is None:
            raise ValueError("digest time must be finite and positive")
        try:
            now = float(now)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("digest time must be finite and positive") from exc
        if not math.isfinite(now) or now <= 0:
            raise ValueError("digest time must be finite and positive")
        if (
            not events
            and not os.path.exists(self.path)
            and not os.path.exists(previous_generation_path(self.path))
        ):
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
                if (not isinstance(group, dict)
                        or isinstance(group.get("count"), bool)
                        or not isinstance(group.get("count"), int) or group["count"] < 0
                        or isinstance(group.get("next_at"), bool)
                        or not isinstance(group.get("next_at"), (int, float))
                        or not math.isfinite(group["next_at"]) or group["next_at"] <= 0
                        or isinstance(group.get("quantity"), bool)
                        or not isinstance(group.get("quantity"), (int, float))
                        or not math.isfinite(group["quantity"]) or group["quantity"] < 0
                        or isinstance(group.get("attempts", 0), bool)
                        or not isinstance(group.get("attempts", 0), int) or group.get("attempts", 0) < 0):
                    raise ValueError("invalid notification digest group")
                if group["count"] == 0 and group["next_at"] <= now:
                    del groups[key]
                    changed = True
            for event in events:
                if (not isinstance(event, dict)
                        or "labels" not in event
                        or "quantity" not in event
                        or isinstance(event.get("quantity"), bool)
                        or isinstance(event.get("sample_id"), bool)
                        or not str(event.get("sample_id") or "").strip()):
                    raise ValueError("invalid notification digest event")
                labels = event["labels"]
                raw_attempts = event.get("attempts", 0)
                if isinstance(raw_attempts, bool):
                    raise ValueError("invalid notification digest event")
                try:
                    quantity = float(event["quantity"])
                    att_f = float(0 if raw_attempts is None else raw_attempts)
                    if not math.isfinite(att_f) or not att_f.is_integer() or int(att_f) < 0:
                        raise ValueError("invalid notification digest event")
                    ev_attempts = int(att_f)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("invalid notification digest event") from exc
                if (not isinstance(labels, dict) or not labels
                        or not all(isinstance(k, str) and isinstance(v, str)
                                   for k, v in labels.items())
                        or not math.isfinite(quantity) or quantity < 0):
                    raise ValueError("invalid notification digest event")
                key = json.dumps(labels, sort_keys=True, separators=(",", ":"))
                group = groups.setdefault(key, {
                    "labels": labels, "next_at": now, "reported": False,
                    "count": 0, "quantity": 0.0, "attempts": 0,
                })
                if group["count"] == 0:
                    group["first_ts"] = now
                    group["sample_id"] = str(event["sample_id"]).strip()
                group["last_ts"] = now
                group["count"] += 1
                group["quantity"] += quantity
                group["attempts"] = max(group.get("attempts", 0), ev_attempts)
                raw_px = event.get("price")
                if raw_px is not None and not isinstance(raw_px, bool):
                    try:
                        px_val = float(raw_px)
                        if math.isfinite(px_val) and px_val > 0:
                            group["last_price"] = px_val
                            raw_prev_notional = group.get("notional")
                            prev_notional = (
                                float(raw_prev_notional)
                                if (raw_prev_notional is not None
                                    and not isinstance(raw_prev_notional, bool)
                                    and math.isfinite(float(raw_prev_notional))
                                    and float(raw_prev_notional) >= 0)
                                else 0.0
                            )
                            group["notional"] = prev_notional + quantity * px_val
                    except (TypeError, ValueError, OverflowError):
                        pass
                raw_created = event.get("created_ts")
                if raw_created is not None and not isinstance(raw_created, bool):
                    try:
                        c_val = float(raw_created)
                        if math.isfinite(c_val) and c_val > 0:
                            prev_c = group.get("created_ts")
                            group["created_ts"] = (
                                min(float(prev_c), c_val)
                                if (prev_c is not None
                                    and not isinstance(prev_c, bool)
                                    and math.isfinite(float(prev_c))
                                    and float(prev_c) > 0)
                                else c_val
                            )
                    except (TypeError, ValueError, OverflowError):
                        pass
                raw_last_attempt = event.get("last_attempt_ts")
                if raw_last_attempt is not None and not isinstance(raw_last_attempt, bool):
                    try:
                        la_val = float(raw_last_attempt)
                        if math.isfinite(la_val) and la_val > 0:
                            prev_la = group.get("last_attempt_ts")
                            prev_la_val = (
                                float(prev_la)
                                if (prev_la is not None
                                    and not isinstance(prev_la, bool)
                                    and math.isfinite(float(prev_la))
                                    and float(prev_la) > 0)
                                else 0.0
                            )
                            group["last_attempt_ts"] = max(prev_la_val, la_val)
                    except (TypeError, ValueError, OverflowError):
                        pass
                raw_ttl = event.get("ttl_sec")
                if raw_ttl is not None and not isinstance(raw_ttl, bool):
                    try:
                        ttl_val = float(raw_ttl)
                        if math.isfinite(ttl_val) and ttl_val > 0:
                            group["ttl_sec"] = ttl_val
                    except (TypeError, ValueError, OverflowError):
                        pass
                changed = True
            due = []
            for group in groups.values():
                if group["count"] and group["next_at"] <= now:
                    due.append(dict(group))
                    group.update(next_at=now + self.interval, reported=True,
                                 count=0, quantity=0.0, attempts=0,
                                 notional=0.0, last_price=None,
                                 created_ts=None, last_attempt_ts=None,
                                 ttl_sec=None)
                    changed = True
            if changed:
                save_json_state(self.path, state, separators=(",", ":"))
            return due


def _alert_giveups(records, now):
    """Audit each expiry, but notify once per incident and summary interval."""
    try:
        events = []
        for rec in records:
            raw_att = rec.get("attempts")
            try:
                att_f = float(raw_att) if (raw_att is not None and not isinstance(raw_att, bool)) else 0.0
                attempts_val = int(att_f) if (math.isfinite(att_f) and att_f >= 0) else 0
            except (TypeError, ValueError, OverflowError):
                attempts_val = 0
            reason = ("attempt_limit" if oq.RETRY_MAX_ATTEMPTS > 0
                      and attempts_val >= oq.RETRY_MAX_ATTEMPTS
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
                "quantity": rec.get("qty", 0.0),
                "sample_id": rec.get("intent_id") or rec.get("id") or "unknown",
                "attempts": attempts_val,
                "price": (
                    rec.get("requested_price")
                    if rec.get("requested_price") is not None
                    else rec.get("ref_price")
                ),
                "created_ts": rec.get("created_ts"),
                "last_attempt_ts": rec.get("last_attempt_ts"),
                "ttl_sec": oq.resolve_record_ttl(rec),
            })
        digest = IncidentDigest(
            os.path.join(os.path.dirname(oq.QUEUE_FILE), "order_retry_giveup_alerts.json"),
            GIVEUP_SUMMARY_SEC)
        for report in digest.collect(events, now=now):
            labels = report["labels"]
            suffix = " summary" if report["reported"] else ""
            cnt = report["count"]
            tot_qty = report["quantity"]
            unit_qty = (tot_qty / cnt) if cnt > 0 else tot_qty
            attempts = int(report.get("attempts") or 0)
            attempts_str = f", {attempts} attempts" if attempts > 0 else ""
            prov = str(labels.get("provider") or "routed").upper()
            side = str(labels.get("side") or "").upper()
            sym = str(labels.get("symbol") or "")
            base_asset, quote_asset = _resolve_display_assets(sym)
            reason = str(labels.get("reason") or "unspecified")
            expiry = str(labels.get("expiry") or "active_ttl")
            kind = labels.get("kind")
            kind_str = f" [{kind}]" if kind and kind != "unspecified" else ""

            px_val = report.get("last_price")
            tot_notional = float(report.get("notional") or 0.0)
            unit_notional = (
                (tot_notional / cnt)
                if (cnt > 0 and tot_notional > 0)
                else ((unit_qty * float(px_val)) if px_val else None)
            )
            price_part = f" @ ${float(px_val):.6g}" if px_val else ""
            value_part = (
                f", Value: ~${unit_notional:.2f} {quote_asset}"
                if unit_notional is not None
                else ""
            )
            created_str = _format_ts(report.get("created_ts"))
            last_attempt_str = _format_ts(report.get("last_attempt_ts"))
            expired_str = _format_ts(report.get("last_ts") or now)
            time_parts = []
            if created_str != "N/A":
                time_parts.append(f"Created: {created_str}")
            if last_attempt_str != "N/A":
                time_parts.append(f"Last attempt: {last_attempt_str}")
            if expired_str != "N/A":
                time_parts.append(f"Expired: {expired_str}")
            time_info = ("Time: " + " | ".join(time_parts) + ". ") if time_parts else ""

            eff_ttl = float(report.get("ttl_sec") or oq.RETRY_TTL_SEC)
            ttl_h = eff_ttl / 3600.0
            if eff_ttl >= 86400 and (eff_ttl % 43200 == 0):
                ttl_str = f"{eff_ttl / 86400:g}d ({ttl_h:g}h)"
            elif eff_ttl >= 3600:
                ttl_str = f"{ttl_h:g}h"
            else:
                ttl_str = f"{eff_ttl / 60.0:g}m"
            expiry_detail = f"{expiry} (TTL: {ttl_str})" if expiry == "active_ttl" else expiry
            try:
                alert.notify(
                    title=f"🛑 order-retry GIVE UP{suffix} {side} {sym}",
                    body=(f"{side} {sym} ({prov}){kind_str}: {cnt} intent(s) expired "
                          f"(size ~{unit_qty:.6g} {base_asset}{price_part}{value_part}{attempts_str}). "
                          f"{time_info}"
                          f"Blocked by {reason} (expiry={expiry_detail}). "
                          f"No order submitted to exchange."),
                    source="order_retry", symbol=sym)
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

