from __future__ import annotations
#!/usr/bin/env python3
"""order_retry_worker.py — the SINGLE READER of the order_retry.py replacement outbox.

Model B uses multiple writers (Instrument.place -> durable pre-submit outbox) and
ONE reader: this process. It scans the queue and advances each DUE order by at most
one bounded lifecycle step per pass.
"""
import argparse
from datetime import datetime
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from notify_engine import alertnotifiers as alert
from botcore import (
    _parse_bool_flag,
    exact_nonneg_int,
    normalize_epoch_sec,
    required_float_env,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    single_instance,
    touch_module_heartbeat,
)
from cachepaths import split_symbol_assets
from lock import FileLock
import order_retry as oq
from providers.base import normalize_side
from providers.execution_audit import ExecutionAudit
from providers.quantity import resolve_assets
from state_io import (
    load_json_state,
    previous_generation_path,
    save_json_state,
)

WORKER_POLL_SEC = required_float_env("RETRY_WORKER_POLL_SEC")
if WORKER_POLL_SEC <= 0:
    raise ValueError("RETRY_WORKER_POLL_SEC must be finite and > 0")
GIVEUP_SUMMARY_SEC = required_float_env("RETRY_GIVEUP_SUMMARY_SEC")
if GIVEUP_SUMMARY_SEC <= 0:
    raise ValueError("RETRY_GIVEUP_SUMMARY_SEC must be finite and > 0")

_AUDIT = ExecutionAudit()
_RECONCILE_FOUND = "found"
_RECONCILE_NOT_FOUND = "not_found"
_RECONCILE_UNAVAILABLE = "unavailable"


class _PersistentQuarantineSet(set):
    """In-memory set backed by a JSON state file next to oq.QUEUE_FILE so quarantine
    alerts are deduplicated across worker process restarts."""

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._loaded_path: str | None = None

    def _path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(str(oq.resolve_cachedb_filepath(oq.QUEUE_FILE)))),
            "order_retry_quarantine_alerts.json",
        )

    def sync_from_disk(self) -> None:
        path = self._path()
        if self._loaded_path == path:
            return
        self._loaded_path = path
        try:
            data = load_json_state(path, default_factory=dict, fail_closed=False, label="quarantine alerts")
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
            loaded = load_json_state(path, default_factory=dict, fail_closed=False, label="quarantine alerts")
            data = dict(loaded) if isinstance(loaded, dict) else {}
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
                loaded = load_json_state(path, default_factory=dict, fail_closed=False, label="quarantine alerts")
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


def _hydrate_missing_order_orig_qty(mkt, symbol, order, *, provider_name=None):
    """Populate orig_qty from venue order_status when absent on the reconciled payload."""
    if oq.extract_orig_qty(order) is None and hasattr(mkt, "order_status"):
        try:
            oid = oq.extract_order_id(order)
            if oid:
                st = mkt.order_status(symbol, oid, provider_name=provider_name)
                if st and getattr(st, "orig_qty", None):
                    order = dict(order)
                    order["orig_qty"] = st.orig_qty
        except Exception:
            pass
    return order


def _reconcile_client_order(mkt, symbol, client_order_id, *, provider_name=None):
    """Return a tri-state client-ID lookup result and any accepted order."""
    lookup = getattr(mkt, "order_by_client_id", None)
    if not callable(lookup) or not client_order_id:
        return _RECONCILE_UNAVAILABLE, None
    try:
        order = lookup(symbol, client_order_id, provider_name=provider_name)
    except Exception as exc:
        print(f"[order_retry] reconciliation unavailable {symbol}/{client_order_id}: {exc}")
        return _RECONCILE_UNAVAILABLE, None
    if order is None:
        return _RECONCILE_NOT_FOUND, None
    if not isinstance(order, dict):
        print(f"[order_retry] malformed reconciliation response {symbol}/{client_order_id}: {type(order).__name__}")
        return _RECONCILE_UNAVAILABLE, None
    if oq.extract_order_id(order) is None:
        print(f"[order_retry] reconciliation response has no order ID {symbol}/{client_order_id}")
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
    except Exception:
        return None
    return result if type(getattr(result, "lookup_by_client_order_id", None)) is bool else None


def _not_found_reliability_status(mkt, record, now, provider_name):
    """Return whether venue absence is recent enough, and the specific failure reason."""
    capabilities = _reconciliation_capabilities(mkt, record.get("symbol"), provider_name)
    if capabilities is None or not capabilities.lookup_by_client_order_id:
        return False, "lookup_unavailable"
    venue_horizon = safe_pos_float(getattr(capabilities, "not_found_reliable_for_seconds", None), default=None)
    configured_horizon = safe_pos_float(oq.RETRY_NOT_FOUND_MAX_AGE_SEC, default=None)
    if venue_horizon is None or configured_horizon is None:
        return False, "missing_reconciliation_horizon"
    observed_at = safe_float(now, default=None)
    created_at = safe_float(record.get("created_ts") or 0.0, default=None)
    last_attempt_at = safe_float(record.get("last_attempt_ts") or 0.0, default=None)
    if observed_at is None or created_at is None or last_attempt_at is None:
        return False, "invalid_reference_timestamp"
    reference_at = max(created_at, last_attempt_at)
    if reference_at <= 0:
        return False, "invalid_reference_timestamp"
    age = observed_at - reference_at
    if age < 0:
        return False, "future_reference_timestamp"
    if age > min(venue_horizon, configured_horizon):
        return False, "not_found_outside_reliable_horizon"
    return True, None


def _audit_event(record, event, **fields):
    place_kwargs = record.get("place_kwargs") if isinstance(record.get("place_kwargs"), dict) else {}
    payload = {
        "side": str(record.get("side") or "").lower() or None,
        "kind": record.get("kind") or place_kwargs.get("kind"),
        "client_order_id": oq._record_client_order_id(record),
        "order_id": record.get("order_id"),
        "revision": record.get("revision"),
        **fields,
    }
    _AUDIT.record(
        event,
        intent_id=str(record.get("intent_id") or f"outbox-{record.get('id')}"),
        venue=str(record.get("provider_name") or "routed"),
        symbol=str(record.get("symbol") or "unknown"),
        **payload,
    )


def _complete_reconciled_order(
    mkt, r: dict, existing_order, now: float, *, provider_name=None, live_producer: bool = False
) -> bool:
    symbol = r.get("symbol")
    existing_order = _hydrate_missing_order_orig_qty(mkt, symbol, existing_order, provider_name=provider_name)
    if not oq.complete_claim(r, "accepted", now, order=existing_order, provider_name=provider_name):
        return False
    oid = oq.extract_order_id(existing_order)
    audit_kw = {"order_id": oid, "live_producer": True} if live_producer else {"order_id": oid}
    _audit_event(r, "order_recovered", **audit_kw)
    label = "RECONCILED live producer" if live_producer else "RECONCILED"
    print(f"[order_retry] {label} {r.get('side')} {symbol} orderId={oid}")
    return True


def _audit_status_event(record, event: str, status, **fields) -> None:
    _audit_event(
        record, event,
        status=status.status, venue_status=status.venue_status,
        filled_qty=status.filled_qty, cost=status.cost, fee=status.fee,
        **fields,
    )


def _process_awaiting_cancel_claim(mkt, r: dict, symbol, provider_name, now: float) -> None:
    replaced_order_id = str(r.get("replaces_order_id") or "")
    try:
        status = mkt.order_status(symbol, replaced_order_id, provider_name=provider_name)
    except Exception as exc:
        err_str = str(exc)
        prev_err = str(r.get("status_error") or "")
        print(f"[order_retry] replaced-order status unavailable {symbol}/{replaced_order_id}: {exc}")
        oq.complete_claim(r, "status_error", now, failure_reason=err_str)
        if prev_err != err_str:
            _audit_event(
                r, "replacement_cancel_status_error",
                replaced_order_id=replaced_order_id,
                error_type=exc.__class__.__name__, error=err_str,
            )
        return
    venue_status = str(status.venue_status or "").upper()
    if venue_status in {"CANCELED", "CANCELLED"}:
        orig_f = safe_pos_float(r.get("replaces_original_qty"), default=None)
        filled_f = safe_nonneg_float(getattr(status, "filled_qty", 0.0) or 0.0, default=None)
        replacement_qty = max(0.0, orig_f - filled_f) if (orig_f is not None and filled_f is not None) else -1.0
        activation = oq.activate_claimed_replacement(r, replacement_qty, now)
        if activation == "activated":
            oq.complete_claim(r, "release", now)
            _audit_event(
                r, "replacement_cancel_confirmed",
                replaced_order_id=replaced_order_id, replacement_qty=replacement_qty,
            )
        elif activation == "resolved":
            _audit_event(
                r, "replacement_not_required",
                replaced_order_id=replaced_order_id,
                status=status.status, venue_status=venue_status, filled_qty=status.filled_qty,
            )
        else:
            oq.complete_claim(r, "status_error", now, failure_reason="cancel_activation_failed")
        return
    if status.terminal:
        oq.complete_claim(r, "success", now)
        _audit_event(
            r, "replacement_not_required",
            replaced_order_id=replaced_order_id,
            status=status.status, venue_status=venue_status, filled_qty=status.filled_qty,
        )
        return
    oq.complete_claim(r, "release", now)


def _process_accepted_claim(mkt, r: dict, symbol, provider_name, now: float) -> tuple[int, int, int]:
    """Poll status for one accepted claim. Returns (filled_inc, terminal_retried_inc, terminal_failed_inc)."""
    try:
        status = mkt.order_status(symbol, str(r.get("order_id")), provider_name=provider_name)
    except Exception as exc:
        err_str = str(exc)
        prev_err = str(r.get("status_error") or "")
        print(f"[order_retry] status unavailable {symbol}/{r.get('order_id')}: {exc}")
        oq.complete_claim(r, "status_error", now, failure_reason=err_str)
        if prev_err != err_str:
            _audit_event(r, "status_error", error_type=exc.__class__.__name__, error=err_str)
        return 0, 0, 0
    transition = oq.advance_claimed_status(r, status, now)
    if transition.action == "observed":
        if transition.status_changed:
            _audit_status_event(r, "order_status", status)
        return 0, 0, 0
    if transition.action == "filled":
        _audit_status_event(r, "order_terminal", status)
        print(f"[order_retry] FILLED {r.get('side')} {symbol} orderId={r.get('order_id')} qty={status.filled_qty}")
        return 1, 0, 0
    if transition.action == "retry_terminal":
        _audit_status_event(r, "order_terminal_retry", status, remaining_qty=transition.remaining_qty)
        print(
            f"[order_retry] {status.venue_status or status.status} {symbol} "
            f"orderId={r.get('order_id')} filled={status.filled_qty} remainder={transition.remaining_qty}"
        )
        return 0, 1, 0
    if not transition.status_changed:
        return 0, 0, 0
    _audit_status_event(r, "order_terminal", status, retry=False)
    _alert_terminal(r, status)
    return 0, 0, 1


def _prepare_retry_placement_kwargs(r: dict, kwargs: dict, provider_name, now: float) -> tuple[str, float, float, str]:
    kwargs["caller_owns_retry"] = True
    kwargs["_retry_requested_price"] = r.get("requested_price")
    kwargs["_retry_price_tolerance"] = oq.RETRY_PRICE_TOL
    if provider_name is not None:
        kwargs["provider_name"] = provider_name

    ttl_started = (
        safe_pos_float(r.get("ttl_started_ts"), default=None)
        or safe_pos_float(r.get("created_ts"), default=None)
        or now
    )
    age_sec = 0.0 if oq.is_non_failure_deferral(r.get("last_failure_reason")) else max(0.0, now - ttl_started)
    effective_ttl = oq.resolve_record_ttl(r) if hasattr(oq, "resolve_record_ttl") else float(oq.RETRY_TTL_SEC)
    total_ttl = safe_pos_float(effective_ttl, default=302400.0)
    age_progress = min(1.0, max(0.0, age_sec / total_ttl))
    side_u = normalize_side(r.get("side")) or str(r.get("side") or "").upper()

    def _enable_profit_math_bypass():
        bypasses = _normalize_bypass_set(kwargs.get("bypass_guards"))
        bypasses.update({"G_PROF", "math_reference"})
        kwargs["bypass_guards"] = sorted(bypasses)
        if side_u == "SELL":
            kwargs["allow_math_reference_sell"] = True

    retry_stage = "standard"
    eff_kind = str(r.get("kind") or kwargs.get("kind") or "").strip()
    if "monitor_order_replace" in eff_kind:
        _enable_profit_math_bypass()
        kwargs["wait_for_trend"] = True
    else:
        is_stop_expired_timer = (
            eff_kind.upper() == "STOP"
            and not _parse_bool_flag(kwargs.get("wait_for_trend"), True)
        )
        kwargs["wait_for_trend"] = not is_stop_expired_timer
        cur_pm_f = safe_float(kwargs.get("profit_margin"), default=None)
        if age_progress >= 0.50:
            retry_stage = "salvage" if age_progress >= 0.75 else "breakeven"
            if retry_stage == "salvage" and side_u == "SELL":
                _enable_profit_math_bypass()
            if cur_pm_f is None or cur_pm_f > 0.15:
                kwargs["profit_margin"] = 0.15
    return side_u, age_sec, age_progress, retry_stage


def process_once(mkt, now=None):
    """Advance each due outbox record by at most one non-blocking lifecycle step."""
    now = resolve_now_ts(now)
    snapshot = oq.load_validated()

    to_retry, to_observe, to_reconcile_cancel, live_producer_reconciliation, expired = [], [], [], [], []
    skipped_price = 0
    quarantined = 0
    for r in snapshot:
        if oq.is_expired(r, now):
            expired.append(r)
            continue
        lifecycle = str(r.get("lifecycle") or "submit_pending").lower()
        symbol = r.get("symbol")
        if str(r.get("submission_state") or "").lower() == "producer_claimed":
            if safe_float(r.get("claim_until"), default=0.0) > float(now):
                continue
            owner_state = oq.producer_claim_owner_state(r)
            if owner_state == "alive":
                live_producer_reconciliation.append(r)
                continue
            if owner_state not in {"dead", "mismatched"}:
                quarantined += 1
                _alert_producer_quarantine(r, reason="producer_identity_unverifiable")
                continue
            provider_name = _provider_name(mkt, r)
            caps = _reconciliation_capabilities(mkt, symbol, provider_name)
            if caps is None or not caps.lookup_by_client_order_id:
                quarantined += 1
                _alert_producer_quarantine(r, reason="lookup_unavailable")
                continue
        if not oq.is_due(r, now):
            continue
        if lifecycle == "accepted":
            to_observe.append(r)
        elif lifecycle == "awaiting_cancel":
            to_reconcile_cancel.append(r)
        else:
            to_retry.append(r)

    expired = oq.discard_expired(expired, now)
    claimed = oq.claim(
        [r["id"] for r in to_retry] + [r["id"] for r in to_observe] + [r["id"] for r in to_reconcile_cancel],
        now,
    )
    _alert_giveups(expired, now)

    succeeded = reconciled = attempted = observed = filled = terminal_failed = terminal_retried = 0
    for r in live_producer_reconciliation:
        symbol = r.get("symbol")
        provider_name = _provider_name(mkt, r)
        reconciliation_state, existing_order = _reconcile_client_order(
            mkt, symbol, oq._record_client_order_id(r), provider_name=provider_name
        )
        if reconciliation_state == _RECONCILE_FOUND:
            if _complete_reconciled_order(
                mkt, r, existing_order, now, provider_name=provider_name, live_producer=True
            ):
                succeeded += 1
                reconciled += 1
            continue
        quarantine_reason = (
            "live_owner_not_found"
            if reconciliation_state == _RECONCILE_NOT_FOUND
            else "live_owner_lookup_unavailable"
        )
        quarantined += 1
        if _alert_producer_quarantine(r, reason=quarantine_reason):
            _audit_event(
                r, "producer_reconciliation_deferred",
                reconciliation_state=reconciliation_state, quarantine_reason=quarantine_reason,
            )

    for r in claimed:
        lifecycle = str(r.get("lifecycle") or "submit_pending").lower()
        symbol = r.get("symbol")
        provider_name = _provider_name(mkt, r)
        if lifecycle == "awaiting_cancel":
            _process_awaiting_cancel_claim(mkt, r, symbol, provider_name, now)
            continue
        if lifecycle == "accepted":
            observed += 1
            f_inc, tr_inc, tf_inc = _process_accepted_claim(mkt, r, symbol, provider_name, now)
            filled += f_inc
            terminal_retried += tr_inc
            terminal_failed += tf_inc
            continue

        kwargs = dict(r.get("place_kwargs") or {})
        client_order_id = kwargs.get("client_order_id")
        if oq.submission_requires_reconciliation(r):
            reconciliation_state, existing_order = _reconcile_client_order(
                mkt, symbol, client_order_id, provider_name=provider_name
            )
            if reconciliation_state == _RECONCILE_FOUND:
                succeeded += 1
                reconciled += 1
                _complete_reconciled_order(mkt, r, existing_order, now, provider_name=provider_name)
                continue
            if reconciliation_state == _RECONCILE_NOT_FOUND:
                absence_reliable, not_found_reason = _not_found_reliability_status(mkt, r, now, provider_name)
            else:
                absence_reliable, not_found_reason = False, "lookup_unavailable"
            if not absence_reliable:
                quarantine_reason = not_found_reason or "lookup_unavailable"
                quarantined += 1
                newly_quarantined = _alert_reconciliation_quarantine(r, reason=quarantine_reason)
                oq.complete_claim(r, "release", now)
                if newly_quarantined:
                    _audit_event(
                        r, "reconciliation_deferred",
                        failure_reason=r.get("last_failure_reason"),
                        reconciliation_state=reconciliation_state,
                        quarantine_reason=quarantine_reason,
                    )
                continue

        try:
            if provider_name is not None and hasattr(mkt, "get_current_price"):
                try:
                    price = mkt.get_current_price(symbol, provider_name=provider_name)
                except TypeError:
                    price = mkt.get_current_price(symbol)
            else:
                price = mkt.get_current_price(symbol)
        except Exception as exc:
            print(f"[order_retry] price unavailable {symbol} ({exc}); skipping, it stays queued")
            oq.complete_claim(r, "release", now)
            continue
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            oq.complete_claim(r, "release", now)
            continue
        if not oq.price_gate_ok(r, price_f):
            oq.complete_claim(r, "release", now)
            skipped_price += 1
            continue

        prev_failure_reason = str(r.get("last_failure_reason") or "").strip()
        prev_submission_state = str(r.get("submission_state") or "").strip().lower()
        side_u, age_sec, age_progress, retry_stage = _prepare_retry_placement_kwargs(r, kwargs, provider_name, now)
        outcome_context = {}
        kwargs["_outcome_context"] = outcome_context
        dispatch_claim = oq.begin_claimed_submit(r, now=now)
        if dispatch_claim is None:
            quarantined += 1
            _audit_event(r, "submit_claim_lost", failure_reason="exact_dispatch_claim_unavailable")
            continue
        r = dispatch_claim
        kwargs["_retry_claim"] = dispatch_claim
        attempted += 1
        next_attempt_num = safe_int(r.get("attempts"), 0) + 1
        req_qty_before_place = r.get("qty")
        try:
            if retry_stage != "standard":
                print(
                    f"[order_retry] {side_u} {symbol} retrying under Stage {retry_stage.upper()} "
                    f"(age={age_sec / 3600:.1f}h, progress={age_progress * 100:.1f}%)"
                )
            order = mkt.place(symbol, r.get("side"), price, r.get("qty"), **kwargs)
        except Exception as e:
            print(f"[order_retry] retry {r.get('side')} {symbol} raised ({e}) — treating it as a failure")
            order = None
        sq = safe_pos_float(outcome_context.get("submitted_qty"), default=None)
        if sq is not None:
            oq._set_record_qty(r, sq)
        accepted = oq.extract_order_id(order) is not None
        if not accepted and outcome_context.get("state") != "refused":
            reconciliation_state, recovered = _reconcile_client_order(
                mkt, symbol, client_order_id, provider_name=provider_name
            )
            if reconciliation_state == _RECONCILE_FOUND:
                order = _hydrate_missing_order_orig_qty(mkt, symbol, recovered, provider_name=provider_name)
                accepted = True
                reconciled += 1
        req_audit_kw = {
            "qty": req_qty_before_place, "price": price, "attempt": next_attempt_num,
            "retry_stage": retry_stage, "age_progress": round(age_progress, 3),
        }
        if accepted:
            succeeded += 1
            oq.complete_claim(r, "accepted", now, order=order, provider_name=provider_name)
            _audit_event(r, "submit_requested", **req_audit_kw)
            _audit_event(
                r, "submit_accepted",
                order_id=oq.extract_order_id(order), qty=r.get("qty"), price=price,
            )
            print(f"[order_retry] ACCEPTED {r.get('side')} {symbol} @ {price} orderId={oq.extract_order_id(order)}")
        else:
            submission_state = outcome_context.get("state") or "unknown"
            refusal_reason = outcome_context.get("reason") or (
                "pre_submit_refused" if submission_state == "refused" else "submit_ambiguous"
            )
            if (
                (
                    refusal_reason in {"execution_disabled", "trading_disabled", "aborted_by_caller", "invalid_price"}
                    or oq.is_terminal_filter_refusal(refusal_reason)
                )
                and submission_state == "refused"
            ):
                oq.complete_claim(r, "success", now)
                terminal_failed += 1
            elif oq.is_non_failure_deferral(refusal_reason):
                oq.complete_claim(
                    r, "deferred", now, failure_reason=refusal_reason, submission_state=submission_state
                )
            else:
                oq.complete_claim(
                    r, "failure", now, failure_reason=refusal_reason, submission_state=submission_state
                )
            repeat_pre_submit_refusal = (
                submission_state == "refused"
                and prev_submission_state == "refused"
                and str(refusal_reason) == prev_failure_reason
            )
            if not repeat_pre_submit_refusal:
                _audit_event(r, "submit_requested", **req_audit_kw)
                _audit_event(
                    r, "submit_refused" if submission_state == "refused" else "submit_unknown",
                    qty=r.get("qty"), price=price,
                    submission_state=submission_state, failure_reason=refusal_reason,
                )

    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "reconciled": reconciled,
        "observed": observed,
        "filled": filled,
        "terminal_failed": terminal_failed,
        "terminal_retried": terminal_retried,
        "expired": len(expired),
        "skipped_price": skipped_price,
        "quarantined": quarantined,
        "remaining": len(oq.load_all()),
    }


def _resolve_display_assets(symbol):
    """Return (base_asset, quote_asset) for human-readable alert formatting."""
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        return "?", "USD"
    sym = symbol.strip().upper()
    split = split_symbol_assets(symbol)
    base, quote = split if split is not None else resolve_assets(sym)
    return (base or sym), (quote or "USD")


def _format_ts(ts):
    """Format a POSIX timestamp in local YYYY-MM-DD HH:MM:SS, or N/A."""
    val = normalize_epoch_sec(ts, default=None, allow_iso=True)
    if val is None or val <= 0:
        return "N/A"
    try:
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"


def _fmt_num(val: float) -> str:
    return f"{float(val):.8f}".rstrip("0").rstrip(".") or "0"


def _concrete_record_owner(rec) -> str:
    if hasattr(oq, "infer_concrete_record_owner"):
        return str(oq.infer_concrete_record_owner(rec) or "").strip()
    if not isinstance(rec, dict):
        return ""
    owner = str(oq.infer_record_owner(rec) or "").strip()
    if not owner or owner in ("unknown", "unspecified", "order_retry") or owner.startswith("alt:"):
        return ""
    return owner


def _format_record_context(rec, *, include_lease=False):
    """Build a structured summary of order time, base asset qty/price, and quote value."""
    raw_side = rec.get("side")
    side = (
        normalize_side(raw_side)
        or (raw_side.strip().upper() if isinstance(raw_side, str) and raw_side.strip() else "?")
    )
    raw_sym = rec.get("symbol")
    symbol = raw_sym.strip() if isinstance(raw_sym, str) and raw_sym.strip() else "?"
    raw_prov = rec.get("provider_name")
    provider = raw_prov.strip().upper() if isinstance(raw_prov, str) and raw_prov.strip() else "ROUTED"
    base_asset, quote_asset = _resolve_display_assets(symbol)
    qty_val = safe_pos_float(rec.get("qty"), default=0.0)
    req_px = safe_pos_float(rec.get("requested_price"), default=None)
    ref_px = safe_pos_float(rec.get("ref_price"), default=None)

    effective_px = req_px if req_px is not None else ref_px
    if req_px is not None and ref_px is not None and abs(req_px - ref_px) > 1e-9:
        price_str = f"@ ${_fmt_num(req_px)} (ref ${_fmt_num(ref_px)})"
    elif effective_px is not None:
        price_str = f"@ ${_fmt_num(effective_px)}"
    else:
        price_str = "@ market"

    value_str = f"~${qty_val * effective_px:.2f} {quote_asset}" if (effective_px is not None and qty_val > 0) else "N/A"
    created_str = _format_ts(rec.get("created_ts"))
    last_attempt_str = _format_ts(rec.get("last_attempt_ts"))
    attempts = safe_int(rec.get("attempts"), 0)
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
    client_order_id = place_kw.get("client_order_id") or rec.get("order_id") or rec.get("id") or "N/A"
    owner = _concrete_record_owner(rec)
    owner_part = f" · {owner}" if owner else ""
    return (
        f"Order: {side} {_fmt_num(qty_val)} {base_asset} ({symbol} · {provider}{owner_part}) {price_str} | Value: {value_str}\n"
        f"Time: {time_line}\n"
        f"ID: {client_order_id}"
    )


def _alert_queue_corruption(exc):
    """Alert at most once per distinct queue corruption in this process."""
    fingerprint = str(getattr(exc, "fingerprint", "") or f"{type(exc).__name__}:{exc}")
    if fingerprint in _ALERTED_QUEUE_CORRUPTIONS:
        return False
    _ALERTED_QUEUE_CORRUPTIONS.add(fingerprint)
    try:
        alert.notify(title="🛑 order-retry queue corruption", body=str(exc), source="order_retry")
    except Exception as alert_exc:
        print(f"[order_retry] queue-corruption alert failed (ignored): {alert_exc}")
    return True


def _send_quarantine_alert(alert_set, fingerprint, rec, title_prefix, detail_msg, *, label: str, include_lease: bool = False):
    if hasattr(alert_set, "sync_from_disk"):
        alert_set.sync_from_disk()
    if fingerprint in alert_set:
        return False
    alert_set.add(fingerprint)
    try:
        context = _format_record_context(rec, include_lease=include_lease)
        owner_sfx, _ = alert.format_owner_tags(_concrete_record_owner(rec))
        alert.notify(
            title=f"{title_prefix} {rec.get('side')} {rec.get('symbol')}{owner_sfx}",
            body=f"{context}\n{detail_msg}",
            source="order_retry",
            symbol=str(rec.get("symbol")),
        )
    except Exception as exc:
        print(f"[order_retry] {label} alert failed (ignored): {exc}")
    return True


_PRODUCER_QUARANTINE_MESSAGES = {
    "producer_identity_unverifiable": (
        "The producer lease expired, but its PID/start identity cannot be verified as either dead or replaced. "
        "Automatic recovery is blocked; inspect the producer and outbox record manually."
    ),
    "live_owner_not_found": (
        "The producer lease expired while its exact process identity is still alive. The venue lookup found no "
        "matching client order, so the record remains claimed for manual review; the worker will never resubmit it."
    ),
}
_DEFAULT_LIVE_OWNER_MSG = (
    "The producer lease expired while its exact process identity is still alive, but deterministic venue lookup "
    "is unavailable. The record remains claimed for manual review; the worker will never resubmit it."
)
_DEFAULT_PRODUCER_QUARANTINE_MSG = (
    "The producer is no longer alive and this provider cannot reconcile the deterministic client order ID. "
    "Automatic submission is blocked; inspect the venue and outbox manually."
)


def _alert_producer_quarantine(rec, *, reason="lookup_unavailable"):
    """Alert once when an expired producer claim cannot be reconciled safely."""
    fingerprint = (
        str(rec.get("id")), safe_int(rec.get("revision"), 0),
        str(rec.get("provider_name") or "routed"), str(reason),
    )
    detail_msg = _PRODUCER_QUARANTINE_MESSAGES.get(reason) or (
        _DEFAULT_LIVE_OWNER_MSG if reason.startswith("live_owner_") else _DEFAULT_PRODUCER_QUARANTINE_MSG
    )
    return _send_quarantine_alert(
        _ALERTED_PRODUCER_QUARANTINES, fingerprint, rec,
        "🛑 order producer claim quarantined", detail_msg,
        label="producer-quarantine", include_lease=True,
    )


_INVALID_REF_TS_MSG = (
    "The venue reports no matching client order, but intent timestamps are invalid or in the future. "
    "Automatic submission and expiry are blocked; inspect the venue and outbox manually."
)
_RECONCILIATION_QUARANTINE_MESSAGES = {
    "not_found_outside_reliable_horizon": (
        "The venue reports no matching client order, but this intent is older than the bounded interval in which "
        "absence is reliable. Automatic submission and expiry are blocked; inspect the venue and outbox manually."
    ),
    "missing_reconciliation_horizon": (
        "The venue reports no matching client order, but the provider does not declare a reliable reconciliation "
        "interval. Automatic submission and expiry are blocked; inspect the venue and outbox manually."
    ),
    "invalid_reference_timestamp": _INVALID_REF_TS_MSG,
    "future_reference_timestamp": _INVALID_REF_TS_MSG,
}
_DEFAULT_RECONCILIATION_QUARANTINE_MSG = (
    "This intent may already exist at the venue, but deterministic client-ID reconciliation is unavailable. "
    "Automatic submission and expiry are blocked; inspect the venue and outbox manually."
)


def _alert_reconciliation_quarantine(rec, *, reason="lookup_unavailable"):
    """Alert once when a possibly submitted intent cannot be reconciled."""
    fingerprint = (
        str(rec.get("id")), safe_int(rec.get("revision"), 0),
        str(rec.get("provider_name") or "routed"),
        str(rec.get("last_failure_reason") or "unknown"), str(reason),
    )
    detail_msg = _RECONCILIATION_QUARANTINE_MESSAGES.get(reason, _DEFAULT_RECONCILIATION_QUARANTINE_MSG)
    return _send_quarantine_alert(
        _ALERTED_RECONCILIATION_QUARANTINES, fingerprint, rec,
        "🛑 order submission quarantined", detail_msg,
        label="reconciliation-quarantine",
    )


def _alert_terminal(rec, status):
    """Alert on a canceled intent whose remainder is deliberately not replayed."""
    terminal_key = (
        str(rec.get("id") or ""), safe_int(rec.get("revision"), 0),
        str(rec.get("order_id") or ""), str(status.venue_status or status.status or ""),
    )
    _send_quarantine_alert(
        _ALERTED_TERMINALS, terminal_key, rec,
        "🛑 order terminal",
        (
            f"Order {rec.get('order_id')} became "
            f"{status.venue_status or status.status}; filled={status.filled_qty}, "
            "the rest is NOT resent automatically after a cancel."
        ),
        label="the terminal",
    )


class IncidentDigest:
    """Persistent incident aggregation, independent of venue and delivery transport."""

    def __init__(self, path, interval_seconds):
        self.path = os.fspath(path)
        interval = safe_pos_float(interval_seconds, default=None)
        if interval is None:
            raise ValueError("digest interval must be finite and positive")
        self.interval = interval

    def collect(self, events, *, now):
        """Return due summaries of events with labels, quantity and sample ID."""
        now_f = safe_pos_float(now, default=None)
        if now_f is None:
            raise ValueError("digest time must be finite and positive")
        now = now_f
        if not events and not os.path.exists(self.path) and not os.path.exists(previous_generation_path(self.path)):
            return []
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with FileLock(self.path + ".lock"):
            state = load_json_state(
                self.path, default_factory=lambda: {"version": 1, "groups": {}},
                fail_closed=True, label="notification digest",
            )
            if state.get("version") != 1 or not isinstance(state.get("groups"), dict):
                raise ValueError("invalid notification digest schema")
            groups = state["groups"]
            changed = False
            for key, group in list(groups.items()):
                if not isinstance(group, dict):
                    raise ValueError("invalid notification digest group")
                count = group.get("count")
                next_at = safe_pos_float(group.get("next_at"), default=None)
                qty = safe_nonneg_float(group.get("quantity"), default=None)
                attempts = group.get("attempts", 0)
                if (
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    or next_at is None or qty is None
                    or not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0
                ):
                    raise ValueError("invalid notification digest group")
                if count == 0 and next_at <= now:
                    del groups[key]
                    changed = True
            for event in events:
                if (
                    not isinstance(event, dict)
                    or "labels" not in event
                    or "quantity" not in event
                    or isinstance(event.get("quantity"), bool)
                    or isinstance(event.get("sample_id"), bool)
                    or not str(event.get("sample_id") or "").strip()
                ):
                    raise ValueError("invalid notification digest event")
                labels = event["labels"]
                quantity = safe_nonneg_float(event.get("quantity"), default=None)
                raw_attempts = event.get("attempts", 0)
                ev_attempts = exact_nonneg_int(0 if raw_attempts is None else raw_attempts, default=None)
                if (
                    quantity is None or ev_attempts is None
                    or not isinstance(labels, dict) or not labels
                    or not all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items())
                ):
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
                px_val = safe_pos_float(event.get("price"), default=None)
                if px_val is not None:
                    group["last_price"] = px_val
                    group["notional"] = safe_nonneg_float(group.get("notional"), default=0.0) + quantity * px_val
                c_val = safe_pos_float(event.get("created_ts"), default=None)
                if c_val is not None:
                    prev_c = safe_pos_float(group.get("created_ts"), default=None)
                    group["created_ts"] = min(prev_c, c_val) if prev_c is not None else c_val
                la_val = safe_pos_float(event.get("last_attempt_ts"), default=None)
                if la_val is not None:
                    group["last_attempt_ts"] = max(safe_pos_float(group.get("last_attempt_ts"), default=0.0), la_val)
                ttl_val = safe_pos_float(event.get("ttl_sec"), default=None)
                if ttl_val is not None:
                    group["ttl_sec"] = ttl_val
                changed = True
            due = []
            for group in groups.values():
                if group["count"] and group["next_at"] <= now:
                    due.append(dict(group))
                    group.update(
                        next_at=now + self.interval, reported=True,
                        count=0, quantity=0.0, attempts=0,
                        notional=0.0, last_price=None,
                        created_ts=None, last_attempt_ts=None, ttl_sec=None,
                    )
                    changed = True
            if changed:
                save_json_state(self.path, state, separators=(",", ":"))
            return due


def _alert_giveups(records, now):
    """Audit each expiry, but notify once per incident and summary interval."""
    try:
        events = []
        for rec in records:
            attempts_val = safe_nonneg_int(rec.get("attempts"), default=0)
            reason = (
                "attempt_limit"
                if oq.RETRY_MAX_ATTEMPTS > 0 and attempts_val >= oq.RETRY_MAX_ATTEMPTS
                else "active_ttl"
            )
            _audit_event(
                rec, "retry_giveup", expiry_reason=reason, expired_ts=now,
                qty=rec.get("qty"), attempts=rec.get("attempts"),
                requested_price=rec.get("requested_price"),
                last_attempt_ts=rec.get("last_attempt_ts"),
                created_ts=rec.get("created_ts"),
                ttl_started_ts=rec.get("ttl_started_ts"),
                last_failure_reason=rec.get("last_failure_reason"),
            )
            place_kw = rec.get("place_kwargs") if isinstance(rec.get("place_kwargs"), dict) else {}
            owner_lbl = _concrete_record_owner(rec) or "unknown"
            events.append({
                "labels": {
                    "provider": str(rec.get("provider_name") or "routed"),
                    "symbol": str(rec.get("symbol")),
                    "side": str(rec.get("side")),
                    "kind": str(rec.get("kind") or place_kw.get("kind") or "unspecified"),
                    "owner": owner_lbl,
                    "reason": str(rec.get("last_failure_reason") or "unspecified"),
                    "expiry": reason,
                },
                "quantity": rec.get("qty", 0.0),
                "sample_id": rec.get("intent_id") or rec.get("id") or "unknown",
                "attempts": attempts_val,
                "price": rec.get("requested_price") if rec.get("requested_price") is not None else rec.get("ref_price"),
                "created_ts": rec.get("created_ts"),
                "last_attempt_ts": rec.get("last_attempt_ts"),
                "ttl_sec": oq.resolve_record_ttl(rec),
            })
        digest = IncidentDigest(
            os.path.join(
                os.path.dirname(os.path.abspath(str(oq.resolve_cachedb_filepath(oq.QUEUE_FILE)))),
                "order_retry_giveup_alerts.json",
            ),
            GIVEUP_SUMMARY_SEC,
        )
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
            owner = str(labels.get("owner") or "")
            owner_sfx = f" [{owner}]" if owner and owner != "unknown" else ""
            owner_body_sfx = f" · {owner}" if owner and owner != "unknown" else ""

            px_val = report.get("last_price")
            tot_notional = float(report.get("notional") or 0.0)
            unit_notional = (
                (tot_notional / cnt)
                if (cnt > 0 and tot_notional > 0)
                else ((unit_qty * float(px_val)) if px_val else None)
            )
            price_part = f" @ ${_fmt_num(float(px_val))}" if px_val else ""
            value_part = f", Value: ~${unit_notional:.2f} {quote_asset}" if unit_notional is not None else ""
            time_parts = [
                f"{lbl}: {val}"
                for lbl, val in (
                    ("Created", _format_ts(report.get("created_ts"))),
                    ("Last attempt", _format_ts(report.get("last_attempt_ts"))),
                    ("Expired", _format_ts(report.get("last_ts") or now)),
                )
                if val != "N/A"
            ]
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
                    title=f"🛑 order-retry GIVE UP{suffix} {side} {sym}{owner_sfx}",
                    body=(
                        f"{side} {sym} ({prov}{owner_body_sfx}){kind_str}: {cnt} intent(s) expired "
                        f"(size ~{_fmt_num(unit_qty)} {base_asset}{price_part}{value_part}{attempts_str}). "
                        f"{time_info}Blocked by {reason} (expiry={expiry_detail}). "
                        f"No order submitted to exchange."
                    ),
                    source="order_retry",
                    symbol=sym,
                )
            except Exception as exc:
                print(f"[order_retry] the giveup delivery failed (ignored): {exc}")
    except Exception as exc:
        print(f"[order_retry] the giveup digest failed (ignored): {exc}")


def _touch_worker_hb(status, *, poll: float = WORKER_POLL_SEC, force: bool = False) -> None:
    touch_module_heartbeat("order_retry", interval_sec=30.0, force=force, status=status, intervals={"poll": poll})


def main():
    import log  # noqa: F401 — ensure stdout timestamps and daily logger file from startup

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

    if not oq.RETRY_ENABLED:
        print("[order_retry] RETRY_ENABLED=false — worker ALIVE but inactive (idle).")
        _touch_worker_hb({"retry_enabled": False}, poll=60.0, force=True)
        while True:
            _touch_worker_hb({"retry_enabled": False}, poll=60.0)
            time.sleep(60)

    from providers.market_api import api as mkt

    try:
        consolidated = oq.consolidate_deferred_streams()
        if consolidated:
            print(
                f"[order_retry] startup consolidation removed {len(consolidated)} "
                "superseded trend-deferred record(s)"
            )
    except oq.RetryQueueCorruptionError as exc:
        _alert_queue_corruption(exc)
        print(f"[order_retry] startup consolidation aborted: {exc}")

    print(f"[order_retry] start (poll={WORKER_POLL_SEC:.0f}s, interval={oq.RETRY_INTERVAL_SEC:.0f}s, TTL={oq.RETRY_TTL_SEC:.0f}s)")
    _touch_worker_hb({"retry_enabled": True}, force=True)
    while True:
        last_stats = None
        try:
            stats = process_once(mkt)
            last_stats = stats
            if stats["attempted"] or stats["expired"]:
                print(f"[order_retry] {stats}")
        except oq.RetryQueueCorruptionError as exc:
            _alert_queue_corruption(exc)
            print(f"[order_retry] queue corruption; pass aborted: {exc}")
        except Exception as exc:
            print(f"[order_retry] pass error (continuing): {exc}")
        _touch_worker_hb(last_stats if isinstance(last_stats, dict) else {"retry_enabled": True})
        time.sleep(WORKER_POLL_SEC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
