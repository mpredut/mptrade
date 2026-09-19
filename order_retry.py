# order_retry.py
"""Provider-agnostic order lifecycle and persistent submission outbox.

This is the canonical home of reusable ``OrderLifecycle`` mechanics. Stateful
strategies may keep their financial campaign state, but use
``TrackedOrderLifecycle`` here for persist-before-submit, response-loss recovery,
status reconciliation, partial/terminal observation, and bounded cancellation.

``Instrument.place`` persists a valid intent before its provider submit and also
enqueues attempts rejected by an earlier guard;
``order_retry_worker.py`` is intended to be the single consumer. Records are stored
as JSONL under ``cachedb`` and queue mutations are serialized with a cross-process
file lock.

Each record preserves the symbol, side, quantity, placement options, requested and
reference prices, timestamps, and attempt count. A retry recalculates its placement
from the current market price and proceeds only when ``price_gate_ok`` accepts that
price. Optional legacy deduplication can retain one pending record per symbol and
side, but it is disabled operationally so independent strategy intents are preserved.

Claims are durable leases: a worker crash leaves the record unavailable only until the
lease expires. A trend deferral consumes neither attempt count nor TTL. The outbox owns
delivery through terminal venue truth: acceptance is persisted and monitored, never
called a fill. Rejected/expired orders can retry only their unfilled remainder, while
an intentional/ambiguous cancellation is never blindly resubmitted. Queue retry still
cannot reconstruct every originating strategy signal, so it relies on placement guards
and the stored price constraint. Configuration lives in ``config.env``.
"""
import hashlib
import os
import json
import math
import time
import uuid
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from lock import FileLock
from providers.strategy_executor import (
    OrderStatus, ProviderError, SubmissionOutcome, SubmissionRefused,
    capture_submission, extract_order_id, reconciliation_capabilities_of,
)

from botcore import (load_dotenv as _load_dotenv, required_bool_env,
                     required_float_env, required_int_env)
_ROOT = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(_ROOT, "config.env"))

RETRY_ENABLED = required_bool_env("RETRY_ENABLED")
RETRY_INTERVAL_SEC = required_float_env("RETRY_INTERVAL_SEC")
RETRY_TTL_SEC = required_float_env("RETRY_TTL_SEC")
RETRY_MAX_ATTEMPTS = required_int_env("RETRY_MAX_ATTEMPTS")
# Retry only at an equal or more favorable current price than the original request.
RETRY_PRICE_TOL = required_float_env("RETRY_PRICE_TOL")
# Optional legacy compaction. Disabled operationally so independent strategies or
# distinct same-side intents cannot overwrite each other in the shared outbox.
RETRY_DEDUP = required_bool_env("RETRY_DEDUP")
# Hard queue-size bound; zero disables the bound.
RETRY_MAX_QUEUE = required_int_env("RETRY_MAX_QUEUE")
RETRY_CLAIM_LEASE_SEC = required_float_env("RETRY_CLAIM_LEASE_SEC")
RETRY_NOT_FOUND_MAX_AGE_SEC = required_float_env("RETRY_NOT_FOUND_MAX_AGE_SEC")


def _validate_retry_config():
    """Reject unsafe retry policy before queue processing can start."""
    invalid = []
    if not math.isfinite(RETRY_INTERVAL_SEC) or RETRY_INTERVAL_SEC <= 0:
        invalid.append("RETRY_INTERVAL_SEC must be finite and > 0")
    if not math.isfinite(RETRY_TTL_SEC) or RETRY_TTL_SEC <= 0:
        invalid.append("RETRY_TTL_SEC must be finite and > 0")
    if (not math.isfinite(RETRY_CLAIM_LEASE_SEC)
            or RETRY_CLAIM_LEASE_SEC <= 0):
        invalid.append("RETRY_CLAIM_LEASE_SEC must be finite and > 0")
    if (not math.isfinite(RETRY_NOT_FOUND_MAX_AGE_SEC)
            or RETRY_NOT_FOUND_MAX_AGE_SEC <= 0):
        invalid.append(
            "RETRY_NOT_FOUND_MAX_AGE_SEC must be finite and > 0")
    if (not math.isfinite(RETRY_PRICE_TOL)
            or not 0 <= RETRY_PRICE_TOL < 1):
        invalid.append(
            "RETRY_PRICE_TOL must be finite, >= 0, and < 1")
    if RETRY_MAX_ATTEMPTS < 0:
        invalid.append("RETRY_MAX_ATTEMPTS must be >= 0")
    if RETRY_MAX_QUEUE < 0:
        invalid.append("RETRY_MAX_QUEUE must be >= 0")
    if invalid:
        raise ValueError("Invalid order-retry configuration: " + "; ".join(invalid))


_validate_retry_config()

NON_FAILURE_DEFERRAL_REASONS = frozenset({
    "trend_deferred",
    "account_cache_not_fresh",
    "account_cache_snapshot_changed",
    "retry_price_unfavorable",
})

TERMINAL_FILTER_REFUSAL_REASONS = frozenset({"below_min_notional"})
TERMINAL_FILTER_REFUSAL_PREFIXES = ("binance_filter_refused:",)


def is_terminal_filter_refusal(reason) -> bool:
    """Return whether unchanged order data can never pass its venue filters."""
    value = str(reason or "")
    return (
        value in TERMINAL_FILTER_REFUSAL_REASONS
        or value.startswith(TERMINAL_FILTER_REFUSAL_PREFIXES)
    )


POSSIBLY_SUBMITTED_REASONS = frozenset({
    "",
    "submit_pending",
    "submit_ambiguous",
    "response_without_order_id",
})


def is_possibly_submitted(rec):
    """Return whether an intent may already exist at its configured venue."""
    if str(rec.get("lifecycle") or "submit_pending").lower() != "submit_pending":
        return False
    submission_state = str(
        rec.get("submission_state") or "").strip().lower()
    if submission_state == "refused":
        return False
    if submission_state in {"producer_claimed", "unknown"}:
        return True
    reason = str(rec.get("last_failure_reason") or "").strip().lower()
    return reason in POSSIBLY_SUBMITTED_REASONS


def is_non_failure_deferral(reason):
    """Return whether a pre-submit refusal pauses attempts and retry TTL."""
    return reason in NON_FAILURE_DEFERRAL_REASONS


QUEUE_FILE = os.path.join(_ROOT, "cachedb", "order_retry_queue.jsonl")
LOCK_FILE = os.path.join(_ROOT, "cachedb", "order_retry_queue.lock")


def _process_start_identity(pid):
    """Return a Linux boot/process-start identity that survives PID reuse.

    Field 22 in /proc/<pid>/stat is the process start time in clock ticks.
    Pairing it with the kernel boot ID makes the value unique across both PID
    reuse and host restarts. A zombie cannot complete a producer claim and is
    therefore treated as dead.
    """
    if isinstance(pid, bool):
        raise ValueError("process PID must be a positive integer")
    pid = int(pid)
    if pid <= 0:
        raise ValueError("process PID must be a positive integer")
    with open(f"/proc/{pid}/stat", "r", encoding="ascii") as stat_file:
        raw_stat = stat_file.read().strip()
    closing_paren = raw_stat.rfind(")")
    if closing_paren < 0:
        raise ValueError("malformed process stat")
    fields = raw_stat[closing_paren + 1:].split()
    if len(fields) <= 19:
        raise ValueError("process stat has no start time")
    if fields[0] in {"Z", "X", "x"}:
        raise ProcessLookupError(f"process {pid} is not running")
    start_ticks = int(fields[19])
    if start_ticks <= 0:
        raise ValueError("process start time must be positive")
    with open(
            "/proc/sys/kernel/random/boot_id", "r",
            encoding="ascii") as boot_file:
        boot_id = boot_file.read().strip()
    if not boot_id:
        raise ValueError("kernel boot ID is unavailable")
    return f"linux:{boot_id}:{start_ticks}"


def producer_claim_owner_state(record):
    """Return alive, dead, mismatched, or unknown for a producer claim.

    Only dead and mismatched authorize worker recovery. Unknown is deliberately
    distinct from dead so missing legacy metadata, permissions, or an unreadable
    process table cannot turn uncertainty into a duplicate submit.
    """
    pid = record.get("producer_pid")
    start_identity = str(
        record.get("producer_process_start_id") or "").strip()
    if isinstance(pid, bool) or not start_identity:
        return "unknown"
    try:
        pid = int(pid)
        if pid <= 0:
            return "unknown"
        current_identity = _process_start_identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        return "dead"
    except (OSError, TypeError, ValueError, OverflowError):
        return "unknown"
    return "alive" if current_identity == start_identity else "mismatched"


class RetryQueueCorruptionError(RuntimeError):
    """Raised when the durable retry queue cannot be parsed completely."""

    def __init__(self, path, line_number, raw_line, detail):
        digest = hashlib.sha256(
            raw_line.encode("utf-8", errors="replace")
        ).hexdigest()[:20]
        self.path = os.path.abspath(path)
        self.line_number = int(line_number)
        self.fingerprint = f"{self.path}:{self.line_number}:{digest}"
        super().__init__(
            "Malformed order-retry queue JSONL at "
            f"{self.path}:{self.line_number}: {detail}. "
            "The queue was left untouched."
        )


def _ensure_dir():
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)


def _read_nolock(*, validate=False):
    """Read the complete queue or fail closed while the caller holds the lock."""
    if not os.path.exists(QUEUE_FILE):
        return []
    items = []
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise RetryQueueCorruptionError(
                    QUEUE_FILE, line_number, raw_line, str(exc)
                ) from exc
            if not isinstance(item, dict):
                raise RetryQueueCorruptionError(
                    QUEUE_FILE,
                    line_number,
                    raw_line,
                    f"expected a JSON object, got {type(item).__name__}",
                )
            if validate and not valid_record(item):
                raise RetryQueueCorruptionError(
                    QUEUE_FILE,
                    line_number,
                    raw_line,
                    "record failed semantic validation",
                )
            items.append(item)
    return items


def _client_order_id(record_id, revision):
    """Return a Binance-compatible stable ID for one queued intent revision."""
    return f"OR_{record_id[:24]}_{int(revision)}"


def order_id_from_response(response):
    """Extract one venue order ID from common Binance/Kraken/HL/T212 shapes."""
    return extract_order_id(response)


def valid_record(rec):
    """Return whether a persisted record is safe to evaluate or submit."""
    try:
        qty = float(rec.get("qty"))
        created = float(rec.get("created_ts"))
        revision = int(rec.get("revision", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    lifecycle = str(rec.get("lifecycle") or "submit_pending").lower()
    lifecycle_ok = lifecycle in {
        "awaiting_cancel", "submit_pending", "accepted"}
    if lifecycle == "accepted" and not str(rec.get("order_id") or "").strip():
        lifecycle_ok = False
    if (lifecycle == "awaiting_cancel"
            and not str(rec.get("replaces_order_id") or "").strip()):
        lifecycle_ok = False
    if lifecycle == "awaiting_cancel":
        try:
            original_qty = float(rec.get("replaces_original_qty"))
            lifecycle_ok = (
                lifecycle_ok
                and math.isfinite(original_qty)
                and original_qty > 0)
        except (TypeError, ValueError, OverflowError):
            lifecycle_ok = False
    return bool(
        rec.get("id") and rec.get("symbol") and
        str(rec.get("side") or "").upper() in {"BUY", "SELL"} and
        math.isfinite(qty) and qty > 0 and
        math.isfinite(created) and created > 0 and revision >= 0 and lifecycle_ok
    )


def _same_deferred_stream(left, right):
    """Return whether two pending trend deferrals represent one current intent.

    Missing provider names in legacy records act as routed wildcards for the same
    symbol. Signal kind remains part of the identity so independent strategies never
    overwrite each other. Quantities are deliberately not summed: the latest strategy
    decision replaces stale desired exposure instead of accumulating phantom orders.
    """
    if any(
            str(item.get("lifecycle") or "submit_pending").lower()
            != "submit_pending"
            or item.get("last_failure_reason") != "trend_deferred"
            for item in (left, right)):
        return False
    left_provider = str(left.get("provider_name") or "").strip().lower()
    right_provider = str(right.get("provider_name") or "").strip().lower()
    provider_matches = (
        not left_provider or not right_provider or left_provider == right_provider)
    return bool(
        provider_matches
        and left.get("symbol") == right.get("symbol")
        and str(left.get("side") or "").upper()
        == str(right.get("side") or "").upper()
        and (left.get("kind") or None) == (right.get("kind") or None)
    )


def _record_has_claim(rec):
    """Return whether a durable record contains any lease ownership state."""
    return any(
        key in rec for key in ("claim_token", "claim_revision", "claim_until")
    )


def _has_prior_venue_evidence(rec):
    """Return whether a record may represent an order or terminal remainder."""
    if str(rec.get("lifecycle") or "submit_pending").lower() != "submit_pending":
        return True
    if any(
        rec.get(key) not in (None, "", [], {})
        for key in (
            "order_id",
            "accepted_ts",
            "submit_status",
            "venue_status",
            "status_error",
            "order_history",
            "replaces_order_id",
            "replaces_original_qty",
        )
    ):
        return True
    reason = str(rec.get("last_failure_reason") or "").strip().lower()
    if reason.startswith("terminal_"):
        return True
    for key in ("delivered_qty", "filled_qty", "filled_cost", "filled_fee"):
        value = rec.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return True
        if not math.isfinite(numeric) or numeric != 0:
            return True
    try:
        requested = float(rec.get("requested_qty_total"))
        remaining = float(rec.get("qty"))
    except (TypeError, ValueError, OverflowError):
        return rec.get("requested_qty_total") not in (None, "")
    return (
        not math.isfinite(requested)
        or not math.isfinite(remaining)
        or abs(requested - remaining) > max(1e-12, abs(requested) * 1e-9)
    )


def _set_safe_to_discard(rec, eligible):
    """Persist explicit pre-submit discard eligibility, or fail closed."""
    if eligible and not _has_prior_venue_evidence(rec):
        rec["safe_to_discard"] = True
    else:
        rec.pop("safe_to_discard", None)


def _safe_to_evict_for_capacity(rec):
    """Return whether capacity pressure may safely discard this record.

    Eviction is deliberately allowlisted. Only known pre-submit deferrals, or a
    persisted explicit pre-submit refusal state, are disposable. Unknown states
    may represent an ambiguous submission and therefore remain protected.
    """
    if (
        str(rec.get("lifecycle") or "submit_pending").lower()
        != "submit_pending"
        or _record_has_claim(rec)
    ):
        return False
    if any(
        rec.get(key) not in (None, "")
        for key in (
            "order_id",
            "accepted_ts",
            "submit_status",
            "venue_status",
            "status_error",
        )
    ):
        return False
    return bool(
        rec.get("safe_to_discard") is True
        and not is_possibly_submitted(rec)
        and not _has_prior_venue_evidence(rec)
    )


def enqueue(symbol, side, qty, place_kwargs=None, requested_price=None, ref_price=None,
            now=None, created_ts=None, attempts=0, last_attempt_ts=0.0,
            failure_reason=None, *, provider_name=None, intent_id=None, kind=None,
            lifecycle="submit_pending", replaces_order_id=None,
            replaces_original_qty=None, _producer_lease_sec=None):
    """Persist a submission or a pre-cancel replacement under the queue lock.

    Pending submissions may use the configured symbol/side deduplication policy.
    ``awaiting_cancel`` records instead deduplicate by provider, symbol, side, and
    replaced order ID, returning the existing record without mutating it or its
    lease. Returns the new or retained record ID, or ``None`` when retries are
    disabled or capacity is exhausted by protected records.
    """
    if not RETRY_ENABLED:
        return None
    lifecycle = str(lifecycle or "").lower()
    if lifecycle not in {"submit_pending", "awaiting_cancel"}:
        return None
    if (lifecycle == "awaiting_cancel"
            and not str(replaces_order_id or "").strip()):
        return None
    if lifecycle == "awaiting_cancel":
        try:
            replaces_original_qty = float(replaces_original_qty)
        except (TypeError, ValueError, OverflowError):
            return None
        if (not math.isfinite(replaces_original_qty)
                or replaces_original_qty <= 0):
            return None
    if _producer_lease_sec is not None:
        try:
            _producer_lease_sec = float(_producer_lease_sec)
        except (TypeError, ValueError, OverflowError):
            return None
        if (lifecycle != "submit_pending"
                or not math.isfinite(_producer_lease_sec)
                or _producer_lease_sec <= 0):
            return None
        producer_pid = os.getpid()
        try:
            producer_process_start_id = _process_start_identity(producer_pid)
        except (
                OSError, ProcessLookupError, TypeError, ValueError,
                OverflowError):
            # A lease without exact process identity makes a live producer
            # indistinguishable from a crashed one. Refuse before provider I/O.
            return None
    else:
        producer_pid = None
        producer_process_start_id = None
    now = now if now is not None else time.time()
    side_u = (side or "").upper()
    try:
        qty = float(qty)
        now = float(now)
    except (TypeError, ValueError, OverflowError):
        return None
    if (not symbol or side_u not in {"BUY", "SELL"} or
            not math.isfinite(qty) or qty <= 0 or
            not math.isfinite(now) or now <= 0):
        return None
    try:
        cts = float(created_ts) if created_ts is not None else now
        attempts = int(attempts)
        last_attempt_ts = float(last_attempt_ts)
    except (TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(cts) or cts <= 0 or not math.isfinite(last_attempt_ts) or
            attempts < 0):
        return None
    record_id = uuid.uuid4().hex
    placement = dict(place_kwargs or {})
    placement.setdefault("client_order_id", _client_order_id(record_id, 0))
    rec = {
        "id": record_id,
        "intent_id": str(intent_id or f"outbox-{record_id}"),
        "provider_name": (None if provider_name is None else str(provider_name)),
        "symbol": symbol,
        "side": side_u,
        "kind": (None if kind is None else str(kind)),
        "qty": qty,
        "requested_qty_total": qty,
        "delivered_qty": 0.0,
        "place_kwargs": placement,
        "requested_price": requested_price,
        "ref_price": ref_price,
        "created_ts": cts,
        "attempts": attempts,
        "last_attempt_ts": last_attempt_ts,
        "last_failure_reason": (str(failure_reason)
                                if failure_reason is not None else None),
        "ttl_started_ts": cts,
        "revision": 0,
        "lifecycle": lifecycle,
    }
    if lifecycle == "awaiting_cancel":
        rec["replaces_order_id"] = str(replaces_order_id)
        rec["replaces_original_qty"] = replaces_original_qty
    _set_safe_to_discard(
        rec, lifecycle == "submit_pending"
        and is_non_failure_deferral(rec.get("last_failure_reason")))
    if _producer_lease_sec is not None:
        # The producer owns this exact revision before it becomes visible. A
        # worker can recover it only after the bounded lease expires following a
        # producer crash; it can never race the in-flight provider call.
        rec["claim_token"] = uuid.uuid4().hex
        rec["claim_until"] = now + _producer_lease_sec
        rec["claim_revision"] = int(rec["revision"])
        rec["submission_state"] = "producer_claimed"
        rec["producer_pid"] = producer_pid
        rec["producer_process_start_id"] = producer_process_start_id
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        if lifecycle == "awaiting_cancel":
            provider_key = str(provider_name or "").casefold()
            replaced_key = str(replaces_order_id)
            for item in existing:
                if (
                    str(item.get("lifecycle") or "").lower()
                    == "awaiting_cancel"
                    and str(item.get("provider_name") or "").casefold()
                    == provider_key
                    and item.get("symbol") == symbol
                    and str(item.get("side") or "").upper() == side_u
                    and str(item.get("replaces_order_id") or "")
                    == replaced_key
                ):
                    # Recovery re-observations are idempotent, including while
                    # another process owns the existing record's lease.
                    return item.get("id")
        if lifecycle == "submit_pending" and failure_reason == "trend_deferred":
            superseded = [
                item for item in existing
                if _safe_to_evict_for_capacity(item) and _same_deferred_stream(item, rec)
            ]
            if superseded:
                superseded_ids = {id(item) for item in superseded}
                existing = [
                    item for item in existing if id(item) not in superseded_ids
                ]
                print(
                    f"[order_retry] consolidated {len(superseded)} older "
                    f"trend-deferred {side_u} {symbol} record(s) into the latest intent")
        if (RETRY_DEDUP and lifecycle == "submit_pending"
                and _producer_lease_sec is None):
            for e in existing:
                if not _safe_to_evict_for_capacity(e):
                    continue
                if (e.get("symbol") == symbol
                        and (e.get("side") or "").upper() == side_u
                        and str(e.get("lifecycle") or "submit_pending").lower()
                        == "submit_pending"):
                    # Refresh one symbol/side target while preserving age and attempts.
                    e["requested_price"] = requested_price
                    e["ref_price"] = ref_price
                    e["qty"] = qty
                    e["requested_qty_total"] = qty
                    revision = int(e.get("revision", 0)) + 1
                    refreshed_kwargs = dict(place_kwargs or {})
                    refreshed_kwargs.setdefault(
                        "client_order_id", _client_order_id(e.get("id", record_id), revision))
                    e["place_kwargs"] = refreshed_kwargs
                    e["created_ts"] = min(float(e.get("created_ts", cts)), cts)
                    e["attempts"] = max(int(e.get("attempts", 0)), int(attempts))
                    e["last_attempt_ts"] = max(float(e.get("last_attempt_ts", 0)),
                                               float(last_attempt_ts))
                    previous_reason = e.get("last_failure_reason")
                    next_reason = (str(failure_reason)
                                   if failure_reason is not None else None)
                    if (is_non_failure_deferral(previous_reason)
                            and not is_non_failure_deferral(next_reason)):
                        e["ttl_started_ts"] = now
                    e["last_failure_reason"] = next_reason
                    e["provider_name"] = (None if provider_name is None
                                            else str(provider_name))
                    e["intent_id"] = str(intent_id or e.get("intent_id")
                                         or f"outbox-{e.get('id', record_id)}")
                    e["kind"] = None if kind is None else str(kind)
                    e["lifecycle"] = "submit_pending"
                    e.pop("order_id", None)
                    _set_safe_to_discard(
                        e, is_non_failure_deferral(
                            e.get("last_failure_reason")))
                    e["revision"] = revision
                    _write_nolock(existing)
                    return e.get("id")
        if RETRY_MAX_QUEUE > 0 and len(existing) >= RETRY_MAX_QUEUE:
            # Evict only records proven never to have reached a venue. Claims and
            # ambiguous, accepted, cancellation, and reconciliation states are
            # protected even if that means refusing the new enqueue.
            pending = [
                (index, item) for index, item in enumerate(existing)
                if str(item.get("lifecycle") or "submit_pending").lower()
                == "submit_pending" and _safe_to_evict_for_capacity(item)
            ]
            deferred = [
                pair for pair in pending
                if pair[1].get("last_failure_reason") == "trend_deferred"
            ]
            candidates = deferred or pending
            if not candidates:
                print(
                    f"[order_retry] queue full ({len(existing)}/{RETRY_MAX_QUEUE}); "
                    f"cannot enqueue {side_u} {symbol} because every record is protected")
                return None

            def _created(pair):
                try:
                    value = float(pair[1].get("created_ts", 0))
                    return value if math.isfinite(value) else 0.0
                except (TypeError, ValueError, OverflowError):
                    return 0.0

            remove_count = len(existing) - RETRY_MAX_QUEUE + 1
            victims = sorted(candidates, key=_created)[:remove_count]
            if len(victims) < remove_count:
                print(
                    f"[order_retry] queue over capacity ({len(existing)}/{RETRY_MAX_QUEUE}); "
                    f"cannot enqueue {side_u} {symbol} without evicting protected records")
                return None
            victim_ids = {id(item) for _, item in victims}
            existing = [item for item in existing if id(item) not in victim_ids]
            evicted = victims[0][1]
            print(
                f"[order_retry] queue full ({len(existing) + len(victims)}/"
                f"{RETRY_MAX_QUEUE}); replaced {len(victims)} oldest pending record(s), "
                f"starting with {evicted.get('side')} {evicted.get('symbol')} "
                f"id={evicted.get('id')}, with {side_u} {symbol}")
        _write_nolock(existing + [rec])
    return dict(rec) if _producer_lease_sec is not None else rec["id"]


def enqueue_claimed(
        symbol, side, qty, place_kwargs=None, requested_price=None,
        ref_price=None, now=None, created_ts=None, attempts=0,
        last_attempt_ts=0.0, failure_reason=None, *, provider_name=None,
        intent_id=None, kind=None, lease_sec=None):
    """Atomically persist and lease one unique producer-owned submission.

    The default lease covers the first complete retry interval plus the normal
    crash-recovery lease. This prevents the worker from starting the same intent
    while the producer is still inside bounded permit, cancellation, or provider
    I/O. The returned snapshot must be finalized with ``complete_claim``.
    """
    try:
        lease_sec = float(
            RETRY_INTERVAL_SEC + RETRY_CLAIM_LEASE_SEC
            if lease_sec is None else lease_sec)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(lease_sec) or lease_sec <= RETRY_INTERVAL_SEC:
        return None
    return enqueue(
        symbol, side, qty, place_kwargs, requested_price, ref_price,
        now=now, created_ts=created_ts, attempts=attempts,
        last_attempt_ts=last_attempt_ts, failure_reason=failure_reason,
        provider_name=provider_name, intent_id=intent_id, kind=kind,
        lifecycle="submit_pending", _producer_lease_sec=lease_sec)


def begin_claimed_submit(claim_snapshot, *, now=None, lease_sec=None):
    """Durably mark one exact claimed revision possibly submitted.

    Token, revision, and deterministic client order ID must still match under
    the queue lock. The current PID/start identity becomes the dispatch owner,
    submission state becomes producer_claimed, and the lease is renewed before
    any external provider call. None means the caller must not submit.
    """
    if not isinstance(claim_snapshot, dict):
        return None
    try:
        now = float(time.time() if now is None else now)
        lease_sec = float(
            RETRY_CLAIM_LEASE_SEC if lease_sec is None else lease_sec)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(now)
        or now <= 0
        or not math.isfinite(lease_sec)
        or lease_sec <= 0
    ):
        return None
    current_pid = os.getpid()
    try:
        current_start_identity = _process_start_identity(current_pid)
    except (
            OSError, ProcessLookupError, TypeError, ValueError,
            OverflowError):
        return None
    snapshot_client_id = dict(
        claim_snapshot.get("place_kwargs") or {}).get("client_order_id")
    if not snapshot_client_id:
        return None
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        refreshed = None
        for record in existing:
            record_client_id = dict(
                record.get("place_kwargs") or {}).get("client_order_id")
            exact_claim = (
                record.get("id") == claim_snapshot.get("id")
                and record.get("claim_token")
                == claim_snapshot.get("claim_token")
                and int(record.get("revision", 0))
                == int(claim_snapshot.get("claim_revision", -1))
                and int(record.get("claim_revision", -1))
                == int(claim_snapshot.get("claim_revision", -2))
                and record_client_id == snapshot_client_id
            )
            if not exact_claim:
                continue
            _set_safe_to_discard(record, False)
            record["producer_pid"] = current_pid
            record["producer_process_start_id"] = current_start_identity
            record["submission_state"] = "producer_claimed"
            record["claim_until"] = max(
                float(record.get("claim_until", 0) or 0),
                now + lease_sec)
            refreshed = dict(record)
            _write_nolock(existing)
            break
        return refreshed


def load_all(now=None):
    """Return all queued entries under lock, failing closed on any corruption."""
    _ensure_dir()
    with FileLock(LOCK_FILE):
        return _read_nolock()


def load_validated(now=None):
    """Return the queue only when every durable record is semantically valid."""
    _ensure_dir()
    with FileLock(LOCK_FILE):
        return _read_nolock(validate=True)


def consolidate_deferred_streams():
    """Atomically keep only the newest pending record per deferred signal stream.

    Return the removed records for operational accounting. Accepted venue orders and
    every non-trend failure remain byte-for-byte represented in the rewritten queue.
    """
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()

        def _created(item):
            try:
                value = float(item.get("created_ts", 0))
                return value if math.isfinite(value) else 0.0
            except (TypeError, ValueError, OverflowError):
                return 0.0

        kept_deferred = []
        keep_ids = set()
        removed = []
        for index, item in sorted(
                enumerate(existing), key=lambda pair: _created(pair[1]), reverse=True):
            is_deferred = (
                _safe_to_evict_for_capacity(item)
                and item.get("last_failure_reason") == "trend_deferred"
            )
            if is_deferred and any(
                    _same_deferred_stream(item, newer) for newer in kept_deferred):
                removed.append(item)
                continue
            keep_ids.add(index)
            if is_deferred:
                kept_deferred.append(item)
        if removed:
            _write_nolock([
                item for index, item in enumerate(existing) if index in keep_ids
            ])
        return removed


def get(record_id):
    """Return one current queue record by ID under lock, or ``None``."""
    if not record_id:
        return None
    _ensure_dir()
    with FileLock(LOCK_FILE):
        for rec in _read_nolock():
            if rec.get("id") == record_id:
                return dict(rec)
    return None


def mark_failure(record_id, failure_reason, *, now=None, client_order_id=None,
                 submission_state=None):
    """Annotate an already persisted pre-submit record without changing its ID.

    ``client_order_id`` protects a newer concurrent revision from being overwritten
    by the result of an older submit call.
    """
    if not record_id:
        return False
    if submission_state not in {None, "refused", "unknown"}:
        raise ValueError(f"invalid submission state: {submission_state!r}")
    now = float(time.time() if now is None else now)
    changed = False
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        for rec in existing:
            if rec.get("id") != record_id:
                continue
            current_cid = dict(rec.get("place_kwargs") or {}).get("client_order_id")
            if client_order_id is not None and current_cid != client_order_id:
                continue
            previous = rec.get("last_failure_reason")
            reason = str(failure_reason) if failure_reason is not None else None
            if (is_non_failure_deferral(previous)
                    and not is_non_failure_deferral(reason)):
                rec["ttl_started_ts"] = now
            rec["last_failure_reason"] = reason
            if submission_state is not None:
                rec["submission_state"] = submission_state
                _set_safe_to_discard(
                    rec, submission_state == "refused")
            elif not is_non_failure_deferral(reason):
                _set_safe_to_discard(rec, False)
            changed = True
            break
        if changed:
            _write_nolock(existing)
    return changed


def _append_order_history(rec, *, observed_at, status=None):
    """Retain a bounded audit trail when one logical intent has several orders."""
    order_id = rec.get("order_id")
    if not order_id:
        return
    row = {
        "order_id": str(order_id),
        "client_order_id": dict(rec.get("place_kwargs") or {}).get(
            "client_order_id"),
        "revision": int(rec.get("revision", 0)),
        "observed_at": float(observed_at),
    }
    if status is not None:
        row.update({
            "status": str(getattr(status, "status", "") or ""),
            "venue_status": str(getattr(status, "venue_status", "") or ""),
            "filled_qty": float(getattr(status, "filled_qty", 0.0) or 0.0),
            "cost": float(getattr(status, "cost", 0.0) or 0.0),
            "fee": float(getattr(status, "fee", 0.0) or 0.0),
        })
    history = list(rec.get("order_history") or [])
    if not history or history[-1].get("order_id") != row["order_id"]:
        history.append(row)
    else:
        history[-1].update(row)
    rec["order_history"] = history[-20:]


def _apply_accepted(rec, order, now, provider_name=None):
    order_id = order_id_from_response(order)
    if order_id is None:
        return False
    _set_safe_to_discard(rec, False)
    rec["order_id"] = order_id
    rec["lifecycle"] = "accepted"
    rec["accepted_ts"] = float(now)
    rec["last_status_ts"] = 0.0
    rec["last_failure_reason"] = None
    rec["submission_state"] = "accepted"
    if provider_name is not None:
        rec["provider_name"] = str(provider_name)
    if isinstance(order, dict) and order.get("status") is not None:
        rec["submit_status"] = str(order.get("status"))
    _append_order_history(rec, observed_at=now)
    return True


def mark_accepted(record_id, order, *, now=None, client_order_id=None,
                  provider_name=None):
    """Move one exact pre-submit revision to durable accepted tracking."""
    if not record_id:
        return False
    now = float(time.time() if now is None else now)
    changed = False
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        for rec in existing:
            if rec.get("id") != record_id:
                continue
            current_cid = dict(rec.get("place_kwargs") or {}).get("client_order_id")
            if client_order_id is not None and current_cid != client_order_id:
                continue
            changed = _apply_accepted(
                rec, order, now, provider_name=provider_name)
            break
        if changed:
            _write_nolock(existing)
    return changed


def resolve_record(record_id, *, client_order_id=None):
    """Legacy helper: remove one exact *unsubmitted* persisted revision.

    Once a venue ID has been accepted, only terminal reconciliation may remove the
    tracker. This prevents an old caller from turning acceptance into an implicit
    fill acknowledgement.
    """
    if not record_id:
        return False
    removed = False
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        remaining = []
        for rec in existing:
            matches = (
                rec.get("id") == record_id
                and str(rec.get("lifecycle") or "submit_pending").lower()
                == "submit_pending"
            )
            if matches and client_order_id is not None:
                matches = (dict(rec.get("place_kwargs") or {}).get(
                    "client_order_id") == client_order_id)
            if matches:
                removed = True
            else:
                remaining.append(rec)
        if removed:
            _write_nolock(remaining)
    return removed


def claim(ids, now=None, lease_sec=None):
    """Lease entries without removing them, making crash recovery deterministic."""
    ids = set(ids)
    if not ids:
        return []
    _ensure_dir()
    now = float(time.time() if now is None else now)
    lease_sec = float(RETRY_CLAIM_LEASE_SEC if lease_sec is None else lease_sec)
    if not math.isfinite(lease_sec) or lease_sec <= 0:
        raise ValueError("claim lease must be finite and positive")
    claimed = []
    token = uuid.uuid4().hex
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        for rec in existing:
            if rec.get("id") not in ids:
                continue
            if float(rec.get("claim_until", 0) or 0) > now:
                continue
            revision = int(rec.get("revision", 0))
            placement = dict(rec.get("place_kwargs") or {})
            placement.setdefault(
                "client_order_id", _client_order_id(rec.get("id", ""), revision))
            rec["place_kwargs"] = placement
            rec["revision"] = revision
            rec["claim_token"] = token
            rec["claim_until"] = now + lease_sec
            rec["claim_revision"] = revision
            claimed.append(dict(rec))
        if claimed:
            _write_nolock(existing)
    return claimed


def complete_claim(claimed, outcome, now=None, *, failure_reason=None,
                   order=None, status=None, provider_name=None,
                   submission_state=None):
    """Finalize one exact leased revision without losing accepted-order tracking.

    ``accepted`` persists the venue ID instead of deleting the record. ``observed``
    keeps an open/partial order. ``retry_terminal`` creates a new client-ID revision
    for only the unfilled remainder. ``success`` removes a filled or intentionally
    canceled terminal order.
    """
    allowed = {
        "success", "failure", "deferred", "release", "accepted",
        "observed", "status_error", "retry_terminal",
    }
    if outcome not in allowed:
        raise ValueError(f"invalid claim outcome: {outcome}")
    if (outcome == "deferred"
            and not is_non_failure_deferral(failure_reason)):
        raise ValueError(
            f"invalid non-failure deferral reason: {failure_reason!r}")
    if submission_state not in {None, "refused", "unknown"}:
        raise ValueError(f"invalid submission state: {submission_state!r}")
    now = float(time.time() if now is None else now)
    changed = False
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        remaining = []
        for rec in existing:
            matches = (rec.get("id") == claimed.get("id") and
                       rec.get("claim_token") == claimed.get("claim_token"))
            if not matches:
                remaining.append(rec)
                continue
            claimed_client_id = dict(
                claimed.get("place_kwargs") or {}).get("client_order_id")
            current_client_id = dict(
                rec.get("place_kwargs") or {}).get("client_order_id")
            same_revision = (
                int(rec.get("revision", 0))
                == int(claimed.get("claim_revision", 0))
                and claimed_client_id is not None
                and current_client_id == claimed_client_id
            )
            if not same_revision:
                # Never release or overwrite ownership of a different durable
                # revision, even if malformed legacy state reused a claim token.
                remaining.append(rec)
                continue
            changed = True
            if outcome == "success" and same_revision:
                continue
            if outcome == "accepted" and same_revision:
                _set_safe_to_discard(rec, False)
                if not _apply_accepted(
                        rec, order, now, provider_name=provider_name):
                    rec["last_failure_reason"] = "response_without_order_id"
                    rec["attempts"] = max(int(rec.get("attempts", 0)),
                                          int(claimed.get("attempts", 0))) + 1
                    rec["last_attempt_ts"] = now
            elif outcome == "observed" and same_revision:
                rec["last_status_ts"] = now
                rec["last_status"] = str(getattr(status, "status", "") or "")
                rec["venue_status"] = str(
                    getattr(status, "venue_status", "") or "")
                rec["filled_qty"] = float(
                    getattr(status, "filled_qty", 0.0) or 0.0)
                rec["filled_cost"] = float(getattr(status, "cost", 0.0) or 0.0)
                rec["filled_fee"] = float(getattr(status, "fee", 0.0) or 0.0)
                rec.pop("status_error", None)
            elif outcome == "status_error" and same_revision:
                rec["last_status_ts"] = now
                rec["status_error"] = (str(failure_reason)
                                       if failure_reason is not None else "unknown")
            elif outcome == "retry_terminal" and same_revision:
                filled_qty = max(0.0, float(
                    getattr(status, "filled_qty", 0.0) or 0.0))
                current_qty = float(rec.get("qty") or 0.0)
                remaining_qty = max(0.0, current_qty - filled_qty)
                _append_order_history(rec, observed_at=now, status=status)
                rec["delivered_qty"] = float(rec.get("delivered_qty") or 0.0) + min(
                    current_qty, filled_qty)
                if remaining_qty <= max(1e-12, current_qty * 1e-9):
                    continue
                revision = int(rec.get("revision", 0)) + 1
                rec["revision"] = revision
                rec["qty"] = remaining_qty
                rec["attempts"] = 0
                rec["last_attempt_ts"] = now
                rec["ttl_started_ts"] = now
                _set_safe_to_discard(rec, False)
                rec["last_failure_reason"] = (
                    "terminal_" + str(getattr(status, "venue_status", "")
                                      or getattr(status, "status", "unknown")).lower())
                rec["lifecycle"] = "submit_pending"
                rec["submission_state"] = "refused"
                for key in (
                        "order_id", "accepted_ts", "last_status_ts", "last_status",
                        "venue_status", "submit_status", "filled_qty", "filled_cost",
                        "filled_fee"):
                    rec.pop(key, None)
                placement = dict(rec.get("place_kwargs") or {})
                placement["client_order_id"] = _client_order_id(
                    rec.get("id", ""), revision)
                rec["place_kwargs"] = placement
            elif outcome == "failure" and same_revision:
                rec["attempts"] = max(int(rec.get("attempts", 0)),
                                      int(claimed.get("attempts", 0))) + 1
                rec["last_attempt_ts"] = now
                next_reason = (str(failure_reason)
                               if failure_reason is not None else None)
                if (is_non_failure_deferral(rec.get("last_failure_reason"))
                        and not is_non_failure_deferral(next_reason)):
                    # TTL begins only after a non-failure deferral ends. Waiting for
                    # trend or current account state cannot discard an intent.
                    rec["ttl_started_ts"] = now
                rec["last_failure_reason"] = next_reason
            elif outcome == "deferred" and same_revision:
                rec["last_attempt_ts"] = now
                rec["last_failure_reason"] = str(failure_reason)
            if same_revision and submission_state is not None:
                rec["submission_state"] = submission_state
                _set_safe_to_discard(
                    rec, submission_state == "refused")
            elif outcome == "retry_terminal":
                _set_safe_to_discard(rec, False)
            for key in ("claim_token", "claim_until", "claim_revision"):
                rec.pop(key, None)
            remaining.append(rec)
        if changed:
            _write_nolock(remaining)
    return changed


def activate_claimed_replacement(claimed, qty, now=None):
    """Atomically make a confirmed-cancel replacement eligible for submission.

    The existing lease is retained. A crash after this transition cannot submit
    concurrently; the generic worker may claim the intent only after the lease
    expires, when the replaced venue order is already confirmed canceled.
    """
    now = float(time.time() if now is None else now)
    try:
        qty = float(qty)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(qty) or qty < 0:
        return None
    result = None
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        for index, rec in enumerate(existing):
            matches = (
                rec.get("id") == claimed.get("id")
                and rec.get("claim_token") == claimed.get("claim_token")
                and int(rec.get("revision", 0))
                == int(claimed.get("claim_revision", -1))
            )
            if not matches:
                continue
            if str(rec.get("lifecycle") or "").lower() != "awaiting_cancel":
                break
            current_qty = float(rec.get("qty") or 0.0)
            if qty > current_qty + max(1e-12, current_qty * 1e-9):
                break
            if qty <= max(1e-12, current_qty * 1e-9):
                existing.pop(index)
                result = "resolved"
            else:
                rec["qty"] = qty
                rec["requested_qty_total"] = qty
                rec["lifecycle"] = "submit_pending"
                rec["last_failure_reason"] = "submit_pending"
                rec["submission_state"] = "refused"
                _set_safe_to_discard(rec, False)
                rec["ttl_started_ts"] = now
                rec["last_attempt_ts"] = 0.0
                result = "activated"
            break
        if result is not None:
            _write_nolock(existing)
    return result


@dataclass(frozen=True)
class OutboxStatusTransition:
    """Result of one accepted outbox record advancing from one venue snapshot."""

    action: str  # observed, filled, retry_terminal, terminal
    status_changed: bool = False
    remaining_qty: float = 0.0

    def __post_init__(self):
        if self.action not in {
                "observed", "filled", "retry_terminal", "terminal"}:
            raise ValueError(f"invalid outbox status action: {self.action!r}")


def _outbox_retryable_terminal(status: OrderStatus) -> bool:
    """Preserve the current mechanical outbox terminal rule.

    This is deliberately not the future financial-policy API. Rejected/expired
    orders retain the existing remainder-retry behavior; other cancellations remain
    terminal. Strategy-specific policy will be added separately.
    """
    native = str(status.venue_status or "").upper()
    return native in {"REJECTED", "EXPIRED"} or status.status == "expired"


def advance_claimed_status(claimed: dict, status: OrderStatus,
                           now=None) -> OutboxStatusTransition:
    """Persist one accepted order-status transition without venue I/O.

    ``Instrument.place`` never calls this function. The external worker obtains one
    status snapshot, then delegates the state mutation here. Therefore terminal
    monitoring remains outside the submit call and every invocation is bounded.
    """
    if not isinstance(status, OrderStatus):
        raise TypeError("status must be OrderStatus")
    if str(claimed.get("lifecycle") or "").lower() != "accepted":
        raise ValueError("status transition requires an accepted claimed record")

    if not status.terminal:
        fingerprint = (
            str(status.status), float(status.filled_qty),
            str(status.venue_status),
        )
        previous = (
            str(claimed.get("last_status") or ""),
            float(claimed.get("filled_qty") or 0.0),
            str(claimed.get("venue_status") or ""),
        )
        complete_claim(claimed, "observed", now, status=status)
        return OutboxStatusTransition(
            "observed", status_changed=fingerprint != previous)

    if status.status == "closed":
        complete_claim(claimed, "success", now, status=status)
        return OutboxStatusTransition("filled", status_changed=True)

    current_qty = float(claimed.get("qty") or 0.0)
    remaining_qty = max(0.0, current_qty - status.filled_qty)
    if _outbox_retryable_terminal(status):
        complete_claim(claimed, "retry_terminal", now, status=status)
        return OutboxStatusTransition(
            "retry_terminal", status_changed=True,
            remaining_qty=remaining_qty)

    complete_claim(claimed, "success", now, status=status)
    return OutboxStatusTransition(
        "terminal", status_changed=True, remaining_qty=remaining_qty)


def discard_expired(snapshots, now=None):
    """Remove only records that remain the exact expired durable revision.

    Rechecking identity and expiry under the queue lock prevents cleanup from
    deleting an intent that another producer refreshed or another worker claimed
    after the initial snapshot.
    """
    if not snapshots:
        return []
    try:
        now = float(time.time() if now is None else now)
    except (TypeError, ValueError, OverflowError):
        return []
    if not math.isfinite(now):
        return []
    exact = set()
    for snapshot in snapshots:
        try:
            key = (
                snapshot.get("id"),
                int(snapshot.get("revision", 0)),
                dict(snapshot.get("place_kwargs") or {}).get(
                    "client_order_id"),
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if key[0] and key[2]:
            exact.add(key)
    if not exact:
        return []
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock(validate=True)
        removed = []
        retained = []
        for record in existing:
            key = (
                record.get("id"),
                int(record.get("revision", 0)),
                dict(record.get("place_kwargs") or {}).get(
                    "client_order_id"),
            )
            if key in exact and is_expired(record, now):
                removed.append(record)
            else:
                retained.append(record)
        if removed:
            _write_nolock(retained)
        return removed


def discard(ids):
    """Atomically remove records by ID without submitting them (expiry/give-up)."""
    ids = set(ids)
    if not ids:
        return []
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        removed = [rec for rec in existing if rec.get("id") in ids]
        if removed:
            _write_nolock([rec for rec in existing if rec.get("id") not in ids])
        return removed


def resolve(symbol, side):
    """Legacy helper: remove only unaccepted pending records by symbol and side.

    Accepted trackers are venue truth and can only be removed by exact terminal
    reconciliation. New code should use ``mark_accepted``/``complete_claim``.
    """
    side_u = (side or "").upper()
    _ensure_dir()
    with FileLock(LOCK_FILE):
        existing = _read_nolock()
        remaining = [
            rec for rec in existing
            if not (rec.get("symbol") == symbol
                    and (rec.get("side") or "").upper() == side_u
                    and str(rec.get("lifecycle") or "submit_pending").lower()
                    == "submit_pending")
        ]
        removed = len(existing) - len(remaining)
        if removed:
            _write_nolock(remaining)
        return removed


def _write_nolock(items):
    """Replace the queue file atomically while the caller holds the lock.

    The temporary file and parent directory are fsynced around the atomic rename so a
    successful mutation survives an ordinary process crash and is power-loss resilient
    on filesystems that honor those primitives.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(QUEUE_FILE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in items:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, QUEUE_FILE)
        try:
            dfd = os.open(os.path.dirname(QUEUE_FILE), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def rewrite(items):
    """Replace the queue with ``items`` atomically while holding the lock."""
    _ensure_dir()
    with FileLock(LOCK_FILE):
        _write_nolock(items)


def is_expired(rec, now=None):
    """Return true after active retry TTL or hard attempt limit.

    A non-failure deferral has made no venue attempt and therefore never consumes TTL
    or the attempt budget. Once another refusal reason appears, ``ttl_started_ts``
    is reset and normal expiry resumes.
    """
    # An accepted order is real venue state, not a stale submit attempt. Never
    # discard its tracker merely because the submission TTL elapsed. Likewise,
    # ambiguous submission state requires positive venue truth or manual action.
    if str(rec.get("lifecycle") or "submit_pending").lower() in {
            "accepted", "awaiting_cancel"} or is_possibly_submitted(rec):
        return False
    try:
        now = float(time.time() if now is None else now)
        claim_until = float(rec.get("claim_until", 0) or 0)
        created = float(rec.get("created_ts", 0))
        ttl_started = float(rec.get("ttl_started_ts", created))
        attempts = int(rec.get("attempts", 0))
    except (TypeError, ValueError, OverflowError):
        return True
    if (not all(math.isfinite(v) for v in (now, claim_until, created, ttl_started))
            or created <= 0 or ttl_started <= 0):
        return True
    if claim_until > now:
        return False
    if is_non_failure_deferral(rec.get("last_failure_reason")):
        return False
    return ((now - ttl_started) > RETRY_TTL_SEC or
            (RETRY_MAX_ATTEMPTS > 0 and attempts >= RETRY_MAX_ATTEMPTS))


def is_due(rec, now=None):
    """Return true once the retry interval has elapsed since creation or last attempt."""
    try:
        now = float(time.time() if now is None else now)
        claim_until = float(rec.get("claim_until", 0) or 0)
        lifecycle = str(
            rec.get("lifecycle") or "submit_pending").lower()
        if lifecycle in {"accepted", "awaiting_cancel"}:
            base = max(float(rec.get("last_status_ts", 0) or 0),
                       float(rec.get("accepted_ts", 0) or 0),
                       float(rec.get("created_ts", 0)))
        else:
            base = max(float(rec.get("last_attempt_ts", 0)),
                       float(rec.get("created_ts", 0)))
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(v) for v in (now, claim_until, base)):
        return False
    return claim_until <= now and (now - base) >= RETRY_INTERVAL_SEC


def price_gate_ok(rec, current_price, tol=None):
    """Return whether the current price is favorable relative to the stored request.

    SELL requires ``current >= requested*(1-tol)`` and BUY requires
    ``current <= requested*(1+tol)``. Missing or invalid prices fail closed.
    """
    if current_price is None:
        return False
    req = rec.get("requested_price")
    if req is None:
        return False
    try:
        current_price = float(current_price)
        req = float(req)
        tol = float(RETRY_PRICE_TOL if tol is None else tol)
    except (TypeError, ValueError, OverflowError):
        return False
    if (not math.isfinite(current_price) or current_price <= 0 or
            not math.isfinite(req) or req <= 0 or
            not math.isfinite(tol) or tol < 0):
        return False
    side = (rec.get("side") or "").upper()
    if side == "SELL":
        return current_price >= req * (1.0 - tol)
    if side == "BUY":
        return current_price <= req * (1.0 + tol)
    return False


# ---------------------------------------------------------------------------
# Strategy-owned lifecycle facade
# ---------------------------------------------------------------------------

PersistPending = Callable[[Optional[dict]], None]
SubmitIntent = Callable[[], object]


@dataclass(frozen=True)
class TrackedOrderResult:
    """One lifecycle observation without equating submit acceptance with a fill."""

    outcome: str  # active, terminal, absent, retryable, refused
    intent: dict
    status: Optional[OrderStatus] = None

    def __post_init__(self):
        if self.outcome not in {
                "active", "terminal", "absent", "retryable", "refused"}:
            raise ValueError(f"invalid tracked-order outcome: {self.outcome!r}")
        if self.outcome == "terminal" and self.status is None:
            raise ValueError("terminal tracked-order result requires status")

    @property
    def order_known(self) -> bool:
        return bool(self.intent.get("order_id"))


class StrategyExecutorLifecycleApi:
    """Adapt one strict executor to the lifecycle lookup contract.

    Stateful strategies already own a credential-scoped executor and should not
    create a second global provider merely to reconcile an ambiguous submission.
    This adapter keeps lookup, status, and cancel calls on that same executor.
    """

    def __init__(self, executor):
        self.executor = executor

    def reconciliation_capabilities(self):
        return reconciliation_capabilities_of(self.executor)

    def _require(self, capability: str, operation: str) -> None:
        if not getattr(self.reconciliation_capabilities(), capability):
            raise RuntimeError(
                f"{self.executor.__class__.__name__}: {operation} is unsupported")

    def order_by_client_id(self, symbol: str, client_order_id: str, *,
                           provider_name=None):
        self._require("lookup_by_client_order_id", "order_by_client_id")
        method = getattr(self.executor, "order_by_client_id", None)
        if not callable(method):
            raise RuntimeError(
                f"{provider_name or type(self.executor).__name__}: "
                "order_by_client_id is unsupported")
        return method(symbol, str(client_order_id))

    def order_status(self, symbol: str, order_id: str, *, provider_name=None):
        self._require("status_by_order_id", "order_status")
        return self.executor.order_status(symbol, str(order_id))

    def cancel_order(self, symbol: str, order_id: str, *, provider_name=None):
        self._require("cancel_by_order_id", "cancel_order")
        return self.executor.cancel_order(symbol, str(order_id))


def _lifecycle_finite(raw, *, positive=False):
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or (positive and value <= 0):
        return None
    return value


def _terminal_payload(status: OrderStatus, observed_at: float) -> dict:
    return {
        "status": status.status,
        "venue_status": status.venue_status,
        "filled_qty": status.filled_qty,
        "cost": status.cost,
        "fee": status.fee,
        "observed_at": observed_at,
    }


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
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return status if status.terminal else None


def propagate_submission_refusal(response, outcome_context):
    """Bridge Instrument's result channel into the strict lifecycle contract.

    A known pre-submit refusal becomes ``SubmissionRefused``. An ambiguous result
    remains unchanged so ``capture_submission`` keeps it recoverable.
    """
    if response is None and isinstance(outcome_context, dict):
        state = str(outcome_context.get("state") or "").strip().lower()
        if state == "refused":
            reason = str(
                outcome_context.get("reason") or "pre_submit_refused").strip()
            raise SubmissionRefused(reason)
    return response


# Backwards-compatible import name for existing providers and external scripts.
OrderSubmissionRefused = SubmissionRefused


class TrackedOrderLifecycle:
    """State machine for one persisted order owned by a strategy.

    The caller supplies a persistence callback because campaign state belongs to
    the strategy. The callback must durably replace the pending intent; raising
    before submit prevents the external side effect. Terminal venue truth stays
    persisted in ``terminal_status`` until the strategy atomically applies the fill
    and removes pending from its own state.

    The class does not run as a daemon. The owning strategy calls ``submit`` once
    and ``reconcile`` from later bounded ticks. The fleet-wide daemon is still
    ``order_retry_worker.py`` and operates only on the shared outbox records.
    """

    def __init__(self, market_api, *, provider_name: str, venue: Optional[str] = None,
                 missing_confirmations: int = 2,
                 retry_on_lookup_error: bool = False,
                 max_age_seconds: Optional[float] = None,
                 audit=None, clock: Callable[[], float] = time.time):
        if not str(provider_name or "").strip():
            raise ValueError("provider_name is required")
        if int(missing_confirmations) <= 0:
            raise ValueError("missing_confirmations must be positive")
        if max_age_seconds is not None:
            parsed_age = _lifecycle_finite(max_age_seconds, positive=True)
            if parsed_age is None:
                raise ValueError("max_age_seconds must be finite and positive")
            max_age_seconds = parsed_age
        self.market_api = market_api
        self.provider_name = str(provider_name)
        self.venue = str(venue or provider_name)
        self.missing_confirmations = int(missing_confirmations)
        self.retry_on_lookup_error = bool(retry_on_lookup_error)
        self.max_age_seconds = max_age_seconds
        self.audit = audit
        self.clock = clock

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
        }
        base.update(fields)
        self.audit.record(event, **base)

    @staticmethod
    def _validate_identity(intent: dict) -> None:
        for key in ("intent_id", "client_order_id", "symbol", "side"):
            if not str(intent.get(key) or "").strip():
                raise ValueError(f"tracked intent missing {key}")
        if str(intent["side"]).upper() not in {"BUY", "SELL"}:
            raise ValueError("tracked intent side must be BUY or SELL")
        if _lifecycle_finite(intent.get("requested_qty"), positive=True) is None:
            raise ValueError("tracked intent requested_qty must be positive")
        price = intent.get("requested_price")
        if price is not None and _lifecycle_finite(price, positive=True) is None:
            raise ValueError("tracked intent requested_price must be positive")

    def new_intent(self, *, intent_id: str, client_order_id: str, symbol: str,
                   side: str, requested_qty: float,
                   requested_price: Optional[float] = None,
                   kind: Optional[str] = None, attempt: int = 1,
                   metadata: Optional[Dict] = None) -> dict:
        now = self._now()
        intent = {
            "intent_id": str(intent_id),
            "client_order_id": str(client_order_id),
            "symbol": str(symbol),
            "side": str(side).upper(),
            "kind": None if kind is None else str(kind),
            "requested_qty": float(requested_qty),
            "requested_price": (None if requested_price is None
                                else float(requested_price)),
            "attempt": int(attempt),
            "created_at": now,
            "lookup_misses": 0,
        }
        reserved = set(intent) | {
            "order_id", "submit_status", "submitted_qty", "submitted_price",
            "last_status", "filled_qty", "terminal_status",
            "cancel_attempted_at",
        }
        for key, value in dict(metadata or {}).items():
            if key in reserved:
                raise ValueError(f"tracked intent metadata uses reserved key: {key}")
            intent[str(key)] = value
        self._validate_identity(intent)
        if intent["attempt"] <= 0:
            raise ValueError("tracked intent attempt must be positive")
        return intent

    @staticmethod
    def _order_fields(response) -> dict:
        if isinstance(response, SubmissionOutcome):
            fields = {"submission_outcome": response.state}
            if response.order_id:
                fields["order_id"] = response.order_id
            if response.reason:
                fields["submission_reason"] = response.reason
            response = response.native
            if response is None:
                return fields
            native_fields = TrackedOrderLifecycle._order_fields(response)
            fields.update(native_fields)
            return fields
        if isinstance(response, dict):
            order_id = (response.get("orderId") or response.get("order_id")
                        or response.get("id") or response.get("txid"))
            fields = {}
            if order_id is not None:
                fields["order_id"] = str(order_id)
            status = response.get("status")
            if status is not None:
                fields["submit_status"] = str(status)
            submitted_qty = _lifecycle_finite(
                response.get("origQty", response.get("qty")), positive=True)
            if submitted_qty is not None:
                fields["submitted_qty"] = submitted_qty
            submitted_price = _lifecycle_finite(
                response.get("price"), positive=True)
            if submitted_price is not None:
                fields["submitted_price"] = submitted_price
            return fields
        if (not isinstance(response, bool)
                and isinstance(response, (str, int)) and str(response).strip()):
            return {"order_id": str(response)}
        return {}

    def submit(self, intent: dict, *, persist: PersistPending,
               submit: SubmitIntent) -> TrackedOrderResult:
        """Persist and submit exactly once; reconcile status from a later tick."""
        pending = dict(intent)
        self._validate_identity(pending)
        persist(dict(pending))
        self._audit(
            "submit_requested", pending,
            qty=pending["requested_qty"], price=pending.get("requested_price"),
        )
        outcome = capture_submission(submit)
        pending["submission_outcome"] = outcome.state
        if outcome.state == "refused":
            pending["refusal_reason"] = outcome.reason
            pending["submit_status"] = "refused_before_submit"
            persist(dict(pending))
            self._audit(
                "submit_refused", pending, reason=outcome.reason,
            )
            return TrackedOrderResult("refused", pending)
        if outcome.state == "unknown":
            pending["submit_error"] = outcome.reason
            persist(dict(pending))
            self._audit(
                "submit_ambiguous", pending,
                error=outcome.reason,
            )
            return TrackedOrderResult("active", pending)

        fields = self._order_fields(outcome)
        pending.update(fields)
        persist(dict(pending))
        if pending.get("order_id"):
            self._audit(
                "submit_accepted", pending,
                qty=pending["requested_qty"],
                submitted_qty=pending.get("submitted_qty"),
                status=pending.get("submit_status"),
            )
        else:
            self._audit("submit_ambiguous", pending, error="response_without_order_id")
        return TrackedOrderResult("active", pending)

    def _persist_terminal(self, pending: dict, status: OrderStatus,
                          persist: PersistPending) -> TrackedOrderResult:
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

        cached_terminal = _status_from_terminal_payload(
            pending.get("terminal_status"))
        if cached_terminal is not None:
            return TrackedOrderResult("terminal", pending, cached_terminal)

        symbol = pending["symbol"]
        client_order_id = pending["client_order_id"]
        order_id = pending.get("order_id")
        if not order_id:
            try:
                native = self.market_api.order_by_client_id(
                    symbol, client_order_id, provider_name=self.provider_name)
            except Exception as exc:
                pending["lookup_error"] = f"{exc.__class__.__name__}: {exc}"
                persist(dict(pending))
                self._audit(
                    "lookup_error", pending,
                    error_type=exc.__class__.__name__, error=str(exc),
                )
                if self.retry_on_lookup_error:
                    persist(None)
                    self._audit("lookup_retryable", pending)
                    return TrackedOrderResult("retryable", pending)
                return TrackedOrderResult("active", pending)
            fields = self._order_fields(native)
            if not fields.get("order_id"):
                misses = int(pending.get("lookup_misses") or 0) + 1
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
            status = self.market_api.order_status(
                symbol, str(order_id), provider_name=self.provider_name)
        except Exception as exc:
            pending["status_error"] = f"{exc.__class__.__name__}: {exc}"
            persist(dict(pending))
            self._audit(
                "status_error", pending,
                error_type=exc.__class__.__name__, error=str(exc),
            )
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
        age_seconds = (max(0.0, self._now() - created_at)
                       if created_at is not None else None)
        cancel_attempted = _lifecycle_finite(
            pending.get("cancel_attempted_at"), positive=True)
        if (self.max_age_seconds is None or age_seconds is None
                or age_seconds < self.max_age_seconds or cancel_attempted is not None):
            return TrackedOrderResult("active", pending, status)

        pending["cancel_attempted_at"] = self._now()
        persist(dict(pending))
        self._audit("cancel_requested", pending, age_seconds=age_seconds)
        try:
            self.market_api.cancel_order(
                symbol, str(order_id), provider_name=self.provider_name)
        except Exception as exc:
            pending["cancel_error"] = f"{exc.__class__.__name__}: {exc}"
            persist(dict(pending))
            self._audit(
                "cancel_ambiguous", pending,
                error_type=exc.__class__.__name__, error=str(exc),
            )
            return TrackedOrderResult("active", pending, status)

        pending.pop("cancel_error", None)
        persist(dict(pending))
        self._audit("cancel_accepted", pending)
        try:
            post_cancel = self.market_api.order_status(
                symbol, str(order_id), provider_name=self.provider_name)
        except Exception as exc:
            pending["status_error"] = f"{exc.__class__.__name__}: {exc}"
            persist(dict(pending))
            self._audit(
                "post_cancel_status_error", pending,
                error_type=exc.__class__.__name__, error=str(exc),
            )
            return TrackedOrderResult("active", pending, status)
        if post_cancel.terminal:
            return self._persist_terminal(pending, post_cancel, persist)
        pending["last_status"] = post_cancel.status
        pending["filled_qty"] = post_cancel.filled_qty
        persist(dict(pending))
        self._audit(
            "cancel_pending", pending, status=post_cancel.status,
            filled_qty=post_cancel.filled_qty,
        )
        return TrackedOrderResult("active", pending, post_cancel)
