# order_retry.py
"""Provider-agnostic order lifecycle and persistent submission outbox.

``Instrument.place`` persists a valid intent before its provider submit and also
enqueues attempts rejected by an earlier guard; ``order_retry_worker.py`` is the
single consumer. Records are stored as JSONL under ``cachedb`` and queue mutations
are serialized with a cross-process file lock.
"""
import hashlib
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

from botcore import (
    FileLock,
    clean_non_bool_str as _clean_str,
    exact_int,
    exact_nonneg_int,
    exact_pos_int,
    extract_order_id,
    load_dotenv,
    normalize_side,
    required_bool_env,
    required_float_env,
    required_int_env,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import canonical_symbol_key, normalize_venue_name, resolve_cachedb_filepath
from providers import order_id_context as rc
from providers.base import MarketDataProvider
from providers.execution_audit import ExecutionAudit, find_order_audit_record
from providers.order_ohlc_utils import extract_orig_qty, resolve_order_actor
from providers.strategy_executor import OrderStatus
from state_io import atomic_write_jsonl, parse_json_payload, parse_proc_stat_fields, read_text_file, read_text_lines

_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

RETRY_ENABLED = required_bool_env("RETRY_ENABLED")
RETRY_INTERVAL_SEC = required_float_env("RETRY_INTERVAL_SEC")
RETRY_TTL_SEC = required_float_env("RETRY_TTL_SEC")
RETRY_MAX_ATTEMPTS = required_int_env("RETRY_MAX_ATTEMPTS")
RETRY_PRICE_TOL = required_float_env("RETRY_PRICE_TOL")
RETRY_DEDUP = required_bool_env("RETRY_DEDUP")
RETRY_MAX_QUEUE = required_int_env("RETRY_MAX_QUEUE")
RETRY_CLAIM_LEASE_SEC = required_float_env("RETRY_CLAIM_LEASE_SEC")
RETRY_NOT_FOUND_MAX_AGE_SEC = required_float_env("RETRY_NOT_FOUND_MAX_AGE_SEC")
MAX_GLOBAL_ORDER_TTL_SEC = required_float_env("MAX_GLOBAL_ORDER_TTL_SEC")
_INVALID_RETRY_CONFIG = [
    msg for cond, msg in (
        (RETRY_INTERVAL_SEC <= 0, "RETRY_INTERVAL_SEC must be finite and > 0"),
        (RETRY_TTL_SEC <= 0, "RETRY_TTL_SEC must be finite and > 0"),
        (RETRY_CLAIM_LEASE_SEC <= 0, "RETRY_CLAIM_LEASE_SEC must be finite and > 0"),
        (RETRY_NOT_FOUND_MAX_AGE_SEC <= 0, "RETRY_NOT_FOUND_MAX_AGE_SEC must be finite and > 0"),
        (not 0 <= RETRY_PRICE_TOL < 1, "RETRY_PRICE_TOL must be finite, >= 0, and < 1"),
        (RETRY_MAX_ATTEMPTS < 0, "RETRY_MAX_ATTEMPTS must be >= 0"),
        (RETRY_MAX_QUEUE < 0, "RETRY_MAX_QUEUE must be >= 0"),
    ) if cond
]
if _INVALID_RETRY_CONFIG:
    raise ValueError("Invalid order-retry configuration: " + "; ".join(_INVALID_RETRY_CONFIG))

NON_FAILURE_DEFERRAL_REASONS = frozenset({
    "daily_limit", "trend_deferred", "medium_trend_deferred",
    "account_cache_not_fresh", "account_cache_snapshot_changed", "retry_price_unfavorable",
})
TERMINAL_FILTER_REFUSAL_REASONS = frozenset({"below_min_notional", "qty_zero_or_negative"})
TERMINAL_FILTER_REFUSAL_PREFIXES = ("binance_filter_refused:", "filter_")
TERMINAL_FILTER_REFUSAL_SUBSTRINGS = (
    "order minimum not met", "volume minimum not met", "cost minimum not met",
    "minimum value of", "invalid price", "invalid size", "invalid arguments", "tick size check failed",
)


def is_terminal_filter_refusal(reason) -> bool:
    """Return whether unchanged order data can never pass its venue filters."""
    value = str(reason or "")
    low = value.lower()
    return (
        value in TERMINAL_FILTER_REFUSAL_REASONS
        or value.startswith(TERMINAL_FILTER_REFUSAL_PREFIXES)
        or any(token in low for token in TERMINAL_FILTER_REFUSAL_SUBSTRINGS)
    )


POSSIBLY_SUBMITTED_REASONS = frozenset({"", "submit_pending", "submit_ambiguous", "response_without_order_id"})


def submission_requires_reconciliation(rec) -> bool:
    """Return whether a persisted intent might already have reached its venue."""
    submission_state = str(rec.get("submission_state") or "").strip().lower()
    if submission_state == "refused":
        return False
    if submission_state in {"producer_claimed", "unknown"}:
        return True
    return str(rec.get("last_failure_reason") or "").strip().lower() in POSSIBLY_SUBMITTED_REASONS


def is_possibly_submitted(rec):
    """Return whether an intent may already exist at its configured venue."""
    if str(rec.get("lifecycle") or "submit_pending").lower() != "submit_pending":
        return False
    return submission_requires_reconciliation(rec)


def is_non_failure_deferral(reason):
    """Return whether a pre-submit refusal pauses attempts and retry TTL."""
    return reason in NON_FAILURE_DEFERRAL_REASONS


QUEUE_FILE = os.path.join(_ROOT, "cachedb", "order_retry_queue.jsonl")
LOCK_FILE = os.path.join(_ROOT, "cachedb", "order_retry_queue.lock")


def _process_start_identity(pid):
    """Return a Linux boot/process-start identity that survives PID reuse."""
    pid_i = exact_pos_int(pid, default=None)
    if pid_i is None:
        raise ValueError("process PID must be a positive integer")
    raw_stat = read_text_file(f"/proc/{pid_i}/stat", encoding="ascii").strip()
    _, fields = parse_proc_stat_fields(raw_stat, min_fields=20)
    if fields[0] in {"Z", "X", "x"}:
        raise ProcessLookupError(f"process {pid_i} is not running")
    start_ticks = exact_pos_int(fields[19], default=None)
    if start_ticks is None:
        raise ValueError("process start time must be positive")
    boot_id = read_text_file("/proc/sys/kernel/random/boot_id", encoding="ascii").strip()
    if not boot_id:
        raise ValueError("kernel boot ID is unavailable")
    return f"linux:{boot_id}:{start_ticks}"


def producer_claim_owner_state(record):
    """Return alive, dead, mismatched, or unknown for a producer claim."""
    pid_i = safe_pos_int(record.get("producer_pid"), default=None)
    start_identity = str(record.get("producer_process_start_id") or "").strip()
    if pid_i is None or not start_identity:
        return "unknown"
    try:
        current_identity = _process_start_identity(pid_i)
    except (FileNotFoundError, ProcessLookupError):
        return "dead"
    except (OSError, TypeError, ValueError, OverflowError):
        return "unknown"
    return "alive" if current_identity == start_identity else "mismatched"


class RetryQueueCorruptionError(RuntimeError):
    """Raised when the durable retry queue cannot be parsed completely."""

    def __init__(self, path, line_number, raw_line, detail):
        digest = hashlib.sha256(raw_line.encode("utf-8", errors="replace")).hexdigest()[:20]
        self.path = os.path.abspath(path)
        self.line_number = exact_nonneg_int(line_number, default=0)
        self.fingerprint = f"{self.path}:{self.line_number}:{digest}"
        super().__init__(
            f"Malformed order-retry queue JSONL at {self.path}:{self.line_number}: {detail}. "
            "The queue was left untouched."
        )


@contextmanager
def _locked_queue(*, validate: bool = False):
    os.makedirs(os.path.dirname(str(resolve_cachedb_filepath(QUEUE_FILE))), exist_ok=True)
    with FileLock(LOCK_FILE):
        yield _read_nolock(validate=validate)


def _remove_matching_locked(predicate, *, validate: bool = False) -> list[dict]:
    with _locked_queue(validate=validate) as existing:
        removed, retained = [], []
        for rec in existing:
            if predicate(rec):
                removed.append(rec)
            else:
                retained.append(rec)
        if removed:
            _write_nolock(retained)
        return removed


def _read_nolock(*, validate=False):
    """Read the complete queue or fail closed while the caller holds the lock."""
    q_path = str(resolve_cachedb_filepath(QUEUE_FILE))
    if not os.path.exists(q_path):
        return []
    items = []
    for line_number, raw_line in enumerate(read_text_lines(q_path, keepends=True), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = parse_json_payload(line)
        except (TypeError, ValueError) as exc:
            raise RetryQueueCorruptionError(QUEUE_FILE, line_number, raw_line, str(exc)) from exc
        if not isinstance(item, dict):
            raise RetryQueueCorruptionError(
                QUEUE_FILE, line_number, raw_line, f"expected a JSON object, got {type(item).__name__}"
            )
        if validate and not valid_record(item):
            raise RetryQueueCorruptionError(
                QUEUE_FILE, line_number, raw_line, "record failed semantic validation"
            )
        items.append(item)
    return items


def _client_order_id(record_id, revision, prefix="OR"):
    """Return a Binance-compatible stable ID for one queued intent revision."""
    rev_str = str(exact_nonneg_int(revision, default=0))
    max_pfx_len = max(1, 36 - 1 - 8 - 1 - len(rev_str))
    pfx = str(prefix or "OR").strip().rstrip("_")[:max_pfx_len].rstrip("_") or "OR"
    overhead = len(pfx) + 1 + len(rev_str) + 1
    hash_len = max(8, min(24, 36 - overhead))
    return f"{pfx}_{record_id[:hash_len]}_{rev_str}"[:36]


def valid_record(rec):
    """Return whether a persisted record is safe to evaluate or submit."""
    if (
        not isinstance(rec, dict)
        or isinstance(rec.get("id"), bool)
        or isinstance(rec.get("symbol"), bool)
    ):
        return False
    qty = safe_pos_float(rec.get("qty"), default=None)
    created = safe_pos_float(rec.get("created_ts"), default=None)
    revision = exact_int(rec.get("revision", 0), default=-1)
    if qty is None or created is None or revision < 0:
        return False
    lifecycle = str(rec.get("lifecycle") or "submit_pending").lower()
    lifecycle_ok = lifecycle in {"awaiting_cancel", "submit_pending", "accepted"}
    if lifecycle == "accepted" and extract_order_id(rec.get("order_id")) is None:
        lifecycle_ok = False
    if lifecycle == "awaiting_cancel":
        if (
            extract_order_id(rec.get("replaces_order_id")) is None
            or safe_pos_float(rec.get("replaces_original_qty"), default=None) is None
        ):
            lifecycle_ok = False
    side_raw = rec.get("side")
    side_u = normalize_side(side_raw) if not isinstance(side_raw, bool) else ""
    return bool(
        str(rec.get("id") or "").strip()
        and str(rec.get("symbol") or "").strip()
        and side_u in {"BUY", "SELL"}
        and lifecycle_ok
    )


_CONSOLIDATED_DEFERRAL_REASONS = frozenset({
    "balance_unavailable", "daily_limit", "insufficient_funds", "trend_deferred",
    "medium_trend_deferred", "medium_trend_qty_zero", "profit_guard",
    "qty_zero_after_policy", "qty_zero_after_weight",
})


def _canonical_queue_symbol(sym) -> str:
    if isinstance(sym, bool):
        return ""
    raw = str(sym or "").strip()
    if not raw:
        return ""
    return canonical_symbol_key(raw) or raw.upper()


def _same_deferred_stream(left, right):
    """Return whether two pending pre-submit deferrals represent one current intent."""
    if any(
        str(item.get("lifecycle") or "submit_pending").lower() != "submit_pending"
        or item.get("last_failure_reason") not in _CONSOLIDATED_DEFERRAL_REASONS
        for item in (left, right)
    ):
        return False
    left_provider = normalize_venue_name(left.get("provider_name"))
    right_provider = normalize_venue_name(right.get("provider_name"))
    provider_matches = not left_provider or not right_provider or left_provider == right_provider
    left_sym = _canonical_queue_symbol(left.get("symbol"))
    right_sym = _canonical_queue_symbol(right.get("symbol"))
    left_pk = left.get("place_kwargs") if isinstance(left.get("place_kwargs"), dict) else {}
    right_pk = right.get("place_kwargs") if isinstance(right.get("place_kwargs"), dict) else {}
    left_kind = str(left.get("kind") or left_pk.get("kind") or "").strip() or None
    right_kind = str(right.get("kind") or right_pk.get("kind") or "").strip() or None
    return bool(
        provider_matches
        and left_sym
        and left_sym == right_sym
        and str(left.get("side") or "").strip().upper() == str(right.get("side") or "").strip().upper()
        and left_kind == right_kind
    )


_PRIOR_VENUE_EVIDENCE_KEYS = (
    "order_id", "accepted_ts", "submit_status", "venue_status",
    "status_error", "order_history", "replaces_order_id", "replaces_original_qty",
)


def _has_prior_venue_evidence(rec):
    """Return whether a record may represent an order or terminal remainder."""
    if str(rec.get("lifecycle") or "submit_pending").lower() != "submit_pending":
        return True
    if any(rec.get(key) not in (None, "", [], {}) for key in _PRIOR_VENUE_EVIDENCE_KEYS):
        return True
    if str(rec.get("last_failure_reason") or "").strip().lower().startswith("terminal_"):
        return True
    for key in ("delivered_qty", "filled_qty", "filled_cost", "filled_fee"):
        value = rec.get(key)
        if value not in (None, ""):
            num = safe_float(value, default=None)
            if num is None or num != 0:
                return True
    requested = safe_float(rec.get("requested_qty_total"), default=None)
    remaining = safe_float(rec.get("qty"), default=None)
    if requested is None or remaining is None:
        return rec.get("requested_qty_total") not in (None, "")
    return abs(requested - remaining) > max(1e-12, abs(requested) * 1e-9)


_TRADEALL_KIND_PREFIXES = (
    "kalman_primary_", "small_window_", "accel_surge_", "accel_plunge_",
    "trend_confirmed_", "consistent_or_old_", "slope<=", "slope>=",
)


def _set_safe_to_discard(rec, eligible):
    """Persist explicit pre-submit discard eligibility, or fail closed."""
    if eligible and not _has_prior_venue_evidence(rec):
        rec["safe_to_discard"] = True
    else:
        rec.pop("safe_to_discard", None)


def _safe_to_evict_for_capacity(rec):
    """Return whether capacity pressure may safely discard this record."""
    if str(rec.get("lifecycle") or "submit_pending").lower() != "submit_pending" or any(
        key in rec for key in ("claim_token", "claim_revision", "claim_until")
    ):
        return False
    if any(
        rec.get(key) not in (None, "")
        for key in ("order_id", "accepted_ts", "submit_status", "venue_status", "status_error")
    ):
        return False
    return bool(
        rec.get("safe_to_discard") is True
        and not is_possibly_submitted(rec)
        and not _has_prior_venue_evidence(rec)
    )


def _shift_ttl_after_deferral(rec: dict, previous_reason, next_reason, prev_attempts: int, prev_last_ts, now: float) -> None:
    if is_non_failure_deferral(previous_reason) and not is_non_failure_deferral(next_reason):
        if prev_attempts == 0:
            rec["ttl_started_ts"] = now
        else:
            ttl_ts = safe_pos_float(rec.get("ttl_started_ts"), default=None)
            if prev_last_ts is not None and ttl_ts is not None and now > prev_last_ts >= ttl_ts:
                rec["ttl_started_ts"] = ttl_ts + (now - prev_last_ts)


def enqueue(
    symbol, side, qty, place_kwargs=None, requested_price=None, ref_price=None,
    now=None, created_ts=None, attempts=0, last_attempt_ts=0.0,
    failure_reason=None, *, provider_name=None, intent_id=None, kind=None,
    lifecycle="submit_pending", replaces_order_id=None,
    replaces_original_qty=None, _producer_lease_sec=None, ttl_sec=None,
):
    """Persist a submission or a pre-cancel replacement under the queue lock."""
    if not RETRY_ENABLED:
        return None
    lifecycle = str(lifecycle or "").lower()
    if lifecycle not in {"submit_pending", "awaiting_cancel"}:
        return None
    clean_rep_oid = extract_order_id(replaces_order_id) if lifecycle == "awaiting_cancel" else None
    if lifecycle == "awaiting_cancel":
        replaces_original_qty = safe_pos_float(replaces_original_qty, default=None)
        if clean_rep_oid is None or replaces_original_qty is None:
            return None
    if _producer_lease_sec is not None:
        _producer_lease_sec = safe_pos_float(_producer_lease_sec, default=None)
        if _producer_lease_sec is None or lifecycle != "submit_pending":
            return None
        producer_pid = os.getpid()
        try:
            producer_process_start_id = _process_start_identity(producer_pid)
        except (OSError, ProcessLookupError, TypeError, ValueError, OverflowError):
            return None
    else:
        producer_pid = None
        producer_process_start_id = None
    if (
        not isinstance(symbol, str)
        or any(
            isinstance(x, bool)
            for x in (
                symbol, side, qty, now, created_ts, attempts,
                last_attempt_ts, intent_id, provider_name, kind, failure_reason,
            )
        )
    ):
        return None
    symbol = symbol.strip()
    now_f = safe_pos_float(time.time() if now is None else now, default=None)
    side_u = normalize_side(side)
    qty_f = safe_pos_float(qty, default=None)
    if not symbol or side_u not in {"BUY", "SELL"} or qty_f is None or now_f is None:
        return None
    qty, now = qty_f, now_f
    cts = safe_pos_float(now if created_ts is None else created_ts, default=None)
    attempts_i = exact_int(attempts, default=-1)
    last_attempt_f = safe_float(last_attempt_ts, default=None)
    if cts is None or attempts_i < 0 or last_attempt_f is None:
        return None
    attempts, last_attempt_ts = attempts_i, last_attempt_f
    record_id = uuid.uuid4().hex
    placement = {
        k: (sorted(str(x) for x in v) if isinstance(v, (set, frozenset)) else v)
        for k, v in dict(place_kwargs or {}).items()
    }
    prefix = placement.get("prefix") or (
        f"{placement.get('owner')}_OR" if placement.get("owner") else None
    )
    eff_kind = kind if kind is not None else placement.get("kind")
    if not prefix and eff_kind:
        kind_s = str(eff_kind).strip()
        if "monitor_order_replace" in kind_s:
            parent_owner = placement.get("parent_owner")
            prefix = "MO_MN_OR" if parent_owner == "manual" else ("MO_TA_OR" if parent_owner == "tradeall" else "MO_OR")
        elif kind_s in {"STOP", "TP"}:
            prefix = "MT_OR"
        elif kind_s == "trend_signal" or kind_s.startswith(_TRADEALL_KIND_PREFIXES):
            prefix = "TA_OR"
        elif kind_s.lower().startswith("assetguardian"):
            prefix = "AG_OR"
    prefix = prefix or "OR"
    placement.setdefault("client_order_id", _client_order_id(record_id, 0, prefix=prefix))
    raw_ttl = ttl_sec if ttl_sec is not None else (placement.get("ttl_sec") if placement else None)
    custom_ttl = safe_pos_float(raw_ttl, default=None)
    if custom_ttl is not None:
        custom_ttl = min(custom_ttl, MAX_GLOBAL_ORDER_TTL_SEC)
    requested_price = safe_pos_float(requested_price, default=None)
    ref_price = safe_pos_float(ref_price, default=None)
    rec = {
        "id": record_id, "prefix": prefix,
        "intent_id": str(intent_id or f"outbox-{record_id}"),
        "provider_name": None if provider_name is None else str(provider_name),
        "symbol": symbol, "side": side_u, "kind": None if kind is None else str(kind),
        "qty": qty, "requested_qty_total": qty, "delivered_qty": 0.0,
        "place_kwargs": placement, "requested_price": requested_price, "ref_price": ref_price,
        "created_ts": cts, "attempts": attempts, "last_attempt_ts": last_attempt_ts,
        "last_failure_reason": str(failure_reason) if failure_reason is not None else None,
        "ttl_started_ts": cts, "revision": 0, "lifecycle": lifecycle,
    }
    if custom_ttl is not None:
        rec["ttl_sec"] = custom_ttl
        rec["expires_at_ts"] = cts + custom_ttl
    if lifecycle == "awaiting_cancel":
        rec["replaces_order_id"] = clean_rep_oid
        rec["replaces_original_qty"] = replaces_original_qty
    _set_safe_to_discard(
        rec,
        lifecycle == "submit_pending"
        and (
            is_non_failure_deferral(rec.get("last_failure_reason"))
            or (_producer_lease_sec is None and rec.get("last_failure_reason") in _CONSOLIDATED_DEFERRAL_REASONS)
        ),
    )
    if _producer_lease_sec is not None:
        rec["claim_token"] = uuid.uuid4().hex
        rec["claim_until"] = now + _producer_lease_sec
        rec["claim_revision"] = safe_nonneg_int(rec.get("revision"), default=0)
        rec["submission_state"] = "producer_claimed"
        rec["producer_pid"] = producer_pid
        rec["producer_process_start_id"] = producer_process_start_id
    provider_key = normalize_venue_name(provider_name)
    sym_key = _canonical_queue_symbol(symbol)
    with _locked_queue() as existing:
        if lifecycle == "awaiting_cancel":
            for item in existing:
                if (
                    str(item.get("lifecycle") or "").lower() == "awaiting_cancel"
                    and normalize_venue_name(item.get("provider_name")) == provider_key
                    and _canonical_queue_symbol(item.get("symbol")) == sym_key
                    and str(item.get("side") or "").upper() == side_u
                    and (extract_order_id(item.get("replaces_order_id")) or str(item.get("replaces_order_id") or ""))
                    == clean_rep_oid
                ):
                    return item.get("id")
        if (
            lifecycle == "submit_pending"
            and (
                failure_reason in {"trend_deferred", "medium_trend_deferred"}
                or (not RETRY_DEDUP and failure_reason in _CONSOLIDATED_DEFERRAL_REASONS)
            )
        ):
            superseded = [
                item for item in existing
                if _safe_to_evict_for_capacity(item) and _same_deferred_stream(item, rec)
            ]
            if superseded:
                superseded_ids = {id(item) for item in superseded}
                existing = [item for item in existing if id(item) not in superseded_ids]
                print(
                    f"[order_retry] consolidated {len(superseded)} older "
                    f"deferred ({failure_reason}) {side_u} {symbol} record(s) into the latest intent"
                )
        if RETRY_DEDUP and lifecycle == "submit_pending" and _producer_lease_sec is None:
            for e in existing:
                if not _safe_to_evict_for_capacity(e):
                    continue
                e_provider = normalize_venue_name(e.get("provider_name"))
                if (
                    _canonical_queue_symbol(e.get("symbol")) == sym_key
                    and (e.get("side") or "").upper() == side_u
                    and str(e.get("lifecycle") or "submit_pending").lower() == "submit_pending"
                    and (not provider_key or not e_provider or provider_key == e_provider)
                ):
                    e["requested_price"] = requested_price
                    e["ref_price"] = ref_price
                    _set_record_qty(e, qty)
                    revision = safe_nonneg_int(e.get("revision"), default=0) + 1
                    refreshed_kwargs = {
                        k: (sorted(str(x) for x in v) if isinstance(v, (set, frozenset)) else v)
                        for k, v in dict(place_kwargs or {}).items()
                    }
                    pfx = e.get("prefix") or refreshed_kwargs.get("prefix") or prefix or "OR"
                    refreshed_kwargs.setdefault(
                        "client_order_id", _client_order_id(e.get("id", record_id), revision, prefix=pfx)
                    )
                    e["prefix"] = pfx
                    e["place_kwargs"] = refreshed_kwargs
                    e["created_ts"] = min(safe_pos_float(e.get("created_ts"), default=cts) or cts, cts)
                    prev_attempts = max(0, exact_int(e.get("attempts"), 0))
                    prev_last_ts = safe_pos_float(e.get("last_attempt_ts"), default=None)
                    e["attempts"] = max(prev_attempts, safe_nonneg_int(attempts, default=0))
                    e["last_attempt_ts"] = max(
                        safe_nonneg_float(e.get("last_attempt_ts"), default=0.0) or 0.0,
                        safe_nonneg_float(last_attempt_ts, default=0.0) or 0.0,
                    )
                    previous_reason = e.get("last_failure_reason")
                    next_reason = str(failure_reason) if failure_reason is not None else None
                    _shift_ttl_after_deferral(
                        e, previous_reason, next_reason, prev_attempts, prev_last_ts, now,
                    )
                    e["last_failure_reason"] = next_reason
                    e["provider_name"] = None if provider_name is None else str(provider_name)
                    e["intent_id"] = str(intent_id or e.get("intent_id") or f"outbox-{e.get('id', record_id)}")
                    e["kind"] = None if kind is None else str(kind)
                    e["lifecycle"] = "submit_pending"
                    if custom_ttl is not None:
                        e["ttl_sec"] = custom_ttl
                        e["expires_at_ts"] = cts + custom_ttl
                    else:
                        e.pop("ttl_sec", None)
                        e.pop("expires_at_ts", None)
                    e.pop("order_id", None)
                    _set_safe_to_discard(
                        e,
                        is_non_failure_deferral(e.get("last_failure_reason"))
                        or e.get("last_failure_reason") in _CONSOLIDATED_DEFERRAL_REASONS,
                    )
                    e["revision"] = revision
                    _write_nolock(existing)
                    return e.get("id")
        if RETRY_MAX_QUEUE > 0 and len(existing) >= RETRY_MAX_QUEUE:
            pending = [
                (index, item) for index, item in enumerate(existing)
                if str(item.get("lifecycle") or "submit_pending").lower() == "submit_pending"
                and _safe_to_evict_for_capacity(item)
            ]
            deferred = [
                pair for pair in pending
                if pair[1].get("last_failure_reason") in {"trend_deferred", "medium_trend_deferred"}
            ]
            candidates = deferred or pending
            remove_count = len(existing) - RETRY_MAX_QUEUE + 1
            if len(candidates) < remove_count:
                reason_msg = (
                    f"queue full ({len(existing)}/{RETRY_MAX_QUEUE}); "
                    f"cannot enqueue {side_u} {symbol} because every record is protected"
                    if not candidates
                    else f"queue over capacity ({len(existing)}/{RETRY_MAX_QUEUE}); "
                    f"cannot enqueue {side_u} {symbol} without evicting protected records"
                )
                print(f"[order_retry] {reason_msg}")
                return None
            victims = sorted(
                candidates, key=lambda pair: safe_nonneg_float(pair[1].get("created_ts"), default=0.0)
            )[:remove_count]
            victim_ids = {id(item) for _, item in victims}
            existing = [item for item in existing if id(item) not in victim_ids]
            evicted = victims[0][1]
            print(
                f"[order_retry] queue full ({len(existing) + len(victims)}/{RETRY_MAX_QUEUE}); "
                f"replaced {len(victims)} oldest pending record(s), starting with "
                f"{evicted.get('side')} {evicted.get('symbol')} id={evicted.get('id')}, with {side_u} {symbol}"
            )
        _write_nolock(existing + [rec])
    return dict(rec) if _producer_lease_sec is not None else rec["id"]


def enqueue_claimed(
    symbol, side, qty, place_kwargs=None, requested_price=None,
    ref_price=None, now=None, created_ts=None, attempts=0,
    last_attempt_ts=0.0, failure_reason=None, *, provider_name=None,
    intent_id=None, kind=None, lease_sec=None, ttl_sec=None,
):
    """Atomically persist and lease one unique producer-owned submission."""
    default_lease = RETRY_INTERVAL_SEC + RETRY_CLAIM_LEASE_SEC
    lease = safe_pos_float(default_lease if lease_sec is None else lease_sec, default=None)
    if lease is None or lease <= RETRY_INTERVAL_SEC:
        return None
    return enqueue(
        symbol, side, qty, place_kwargs, requested_price, ref_price,
        now=now, created_ts=created_ts, attempts=attempts,
        last_attempt_ts=last_attempt_ts, failure_reason=failure_reason,
        provider_name=provider_name, intent_id=intent_id, kind=kind,
        lifecycle="submit_pending", _producer_lease_sec=lease,
        ttl_sec=ttl_sec,
    )


def _record_client_order_id(rec: object):
    if not isinstance(rec, dict):
        return None
    place_kw = rec.get("place_kwargs")
    return place_kw.get("client_order_id") if isinstance(place_kw, dict) else None


def _matches_exact_claim(record: dict, claim_snapshot: dict, snapshot_client_id: str) -> bool:
    return (
        record.get("id") == claim_snapshot.get("id")
        and record.get("claim_token") == claim_snapshot.get("claim_token")
        and exact_int(record.get("revision"), 0) == exact_int(claim_snapshot.get("claim_revision"), -1)
        and exact_int(record.get("claim_revision"), -1) == exact_int(claim_snapshot.get("claim_revision"), -2)
        and _record_client_order_id(record) == snapshot_client_id
    )


def _set_record_qty(rec: dict, qty: float) -> None:
    rec["qty"] = qty
    rec["requested_qty_total"] = safe_nonneg_float(rec.get("delivered_qty"), default=0.0) + qty


def _apply_claim_qty_and_scale(record: dict, existing: list, qty=None, applied_scale=None) -> dict:
    q_val = safe_pos_float(qty, default=None)
    if q_val is not None:
        _set_record_qty(record, q_val)
    if applied_scale is not None:
        s_val = safe_pos_float(applied_scale, default=None)
        if s_val is not None and s_val < 1.0:
            pk = dict(record.get("place_kwargs") or {})
            pk["applied_scale"] = s_val
            record["place_kwargs"] = pk
    refreshed = dict(record)
    _write_nolock(existing)
    return refreshed


def update_claimed_qty(claim_snapshot, qty, *, now=None, applied_scale=None):
    """Durably update the requested quantity of an exact claimed intent before submit."""
    if not isinstance(claim_snapshot, dict):
        return None
    q_val = safe_pos_float(qty, default=None)
    snapshot_client_id = _record_client_order_id(claim_snapshot)
    if q_val is None or not snapshot_client_id:
        return None
    with _locked_queue() as existing:
        for record in existing:
            if _matches_exact_claim(record, claim_snapshot, snapshot_client_id):
                return _apply_claim_qty_and_scale(record, existing, q_val, applied_scale)
    return None


def begin_claimed_submit(claim_snapshot, *, now=None, lease_sec=None, qty=None, applied_scale=None):
    """Durably mark one exact claimed revision possibly submitted."""
    if not isinstance(claim_snapshot, dict):
        return None
    now_f = safe_pos_float(time.time() if now is None else now, default=None)
    lease_f = safe_pos_float(RETRY_CLAIM_LEASE_SEC if lease_sec is None else lease_sec, default=None)
    snapshot_client_id = _record_client_order_id(claim_snapshot)
    if now_f is None or lease_f is None or not snapshot_client_id:
        return None
    current_pid = os.getpid()
    try:
        current_start_identity = _process_start_identity(current_pid)
    except (OSError, ProcessLookupError, TypeError, ValueError, OverflowError):
        return None
    with _locked_queue() as existing:
        for record in existing:
            if _matches_exact_claim(record, claim_snapshot, snapshot_client_id):
                _set_safe_to_discard(record, False)
                record["producer_pid"] = current_pid
                record["producer_process_start_id"] = current_start_identity
                record["submission_state"] = "producer_claimed"
                record["claim_until"] = max(
                    safe_nonneg_float(record.get("claim_until"), default=0.0) or 0.0,
                    now_f + lease_f,
                )
                return _apply_claim_qty_and_scale(record, existing, qty, applied_scale)
    return None


def load_all():
    """Return all queued entries under lock, failing closed on any corruption."""
    with _locked_queue() as existing:
        return existing


def _safe_load_all() -> list:
    try:
        return load_all()
    except Exception:
        return []


def get_accepted_order_ttl_map() -> dict:
    """Return a mapping of order_id and client_order_id to custom ttl_sec for active accepted orders."""
    return {
        k: ttl_val
        for r in _safe_load_all()
        if (lc := str(r.get("lifecycle") or "").lower()) == "accepted"
        or (lc == "submit_pending" and r.get("order_id") is None)
        if (ttl_val := safe_pos_float(r.get("ttl_sec"), default=None)) is not None
        for k in (extract_order_id(r.get("order_id")), extract_order_id(_record_client_order_id(r)))
        if k
    }


def _is_concrete_owner(owner: str) -> bool:
    return bool(
        owner
        and not isinstance(owner, bool)
        and owner not in {"unknown", "unspecified", "order_retry"}
        and not str(owner).startswith("alt:")
    )


def infer_record_owner(rec: dict) -> str:
    """Infer the logical bot owner of a durable outbox or execution-audit record."""
    if not isinstance(rec, dict):
        return "unspecified"
    raw_place_kw = rec.get("place_kwargs")
    placement = dict(raw_place_kw) if isinstance(raw_place_kw, dict) else {}
    parent_owner = (_clean_str(rec.get("parent_owner")) or _clean_str(placement.get("parent_owner"))).lower()
    if _is_concrete_owner(parent_owner):
        return parent_owner
    owner = "unspecified"
    cid = (
        _clean_str(rec.get("client_order_id"))
        or _clean_str(placement.get("client_order_id"))
        or _clean_str(rec.get("prefix"))
    )
    if cid:
        try:
            owner = rc.resolve_order_owner(cid)
        except Exception:
            owner = "unspecified"
    if _is_concrete_owner(owner):
        return owner
    iid = _clean_str(rec.get("intent_id"))
    if iid:
        try:
            iid_owner = rc.resolve_order_owner(iid)
            if _is_concrete_owner(iid_owner):
                return iid_owner
        except Exception:
            pass
        if iid.startswith(("intent_", "AI_")):
            return "ai_executor"
        if iid.startswith("spot-dca:"):
            return "spot_dca"
        if iid.startswith("trailing:"):
            return "trailing_stop"
    kind = _clean_str(rec.get("kind")) or _clean_str(placement.get("kind"))
    if "monitor_order_replace" in kind:
        return "monitororder"
    if kind in {"STOP", "TP", "SL", "REENTRY"}:
        return "monitortrades"
    if kind.startswith(_TRADEALL_KIND_PREFIXES) or kind in {"trend_signal", "INITIAL", "TREND", "ADD", "DCA", "CYCLE_SELL", "CYCLE_BUY"}:
        return "tradeall"
    if kind.lower().startswith("assetguardian") or "asset_guardian" in kind.lower():
        return "assetguardian"
    if kind.upper().startswith("TRAIL"):
        return "trailing_stop"
    return owner


def infer_concrete_record_owner(rec: dict) -> str:
    """Return the inferred owner of a record only when it is a concrete bot/module owner."""
    owner = infer_record_owner(rec)
    return owner if _is_concrete_owner(owner) else ""


def get_accepted_order_owner_map() -> dict:
    """Return a mapping of order_id and client_order_id to logical bot owner for active accepted orders."""
    res = {}
    for r in _safe_load_all():
        lc = str(r.get("lifecycle") or "").lower()
        if lc != "accepted" and not (lc == "submit_pending" and r.get("order_id") is None):
            continue
        inferred = infer_concrete_record_owner(r)
        if not inferred:
            continue
        for k in (extract_order_id(r.get("order_id")), extract_order_id(_record_client_order_id(r))):
            if k:
                res[k] = inferred
    return res


def resolve_tracked_order_owner(
    order_id=None,
    client_order_id=None,
    *,
    symbol=None,
    event_ts_s: Optional[float] = None,
    accepted_owner_map: Optional[dict] = None,
    audit_dir: Optional[str] = None,
) -> str:
    """Resolve the logical bot owner for an order across clientOrderId, outbox, and execution audit."""
    oid_str = extract_order_id(order_id) or _clean_str(order_id)
    cid_str = _clean_str(client_order_id)
    owner = "unspecified"
    if cid_str:
        try:
            owner = rc.resolve_order_owner(cid_str)
        except Exception:
            owner = "unspecified"
        if _is_concrete_owner(owner):
            return owner
    owner_map = accepted_owner_map if isinstance(accepted_owner_map, dict) else get_accepted_order_owner_map()
    for key in (oid_str, cid_str):
        if key and key in owner_map:
            mapped = _clean_str(owner_map.get(key))
            if _is_concrete_owner(mapped):
                return mapped
    if oid_str or cid_str:
        try:
            audit_rec = find_order_audit_record(
                oid_str, cid_str, symbol=symbol, event_ts_s=event_ts_s, audit_dir=audit_dir
            )
            if isinstance(audit_rec, dict):
                audit_owner = infer_record_owner(audit_rec)
                if _is_concrete_owner(audit_owner):
                    return audit_owner
        except Exception:
            pass
    return owner


def get_accepted_order_ids_for_owner(symbol: str, owner: str) -> list:
    """Return active accepted order_ids for ``symbol`` belonging to logical ``owner``."""
    if isinstance(symbol, bool) or isinstance(owner, bool):
        return []
    target_sym = _canonical_queue_symbol(symbol)
    target_owner = str(owner or "").strip().lower()
    if not target_sym or not target_owner:
        return []
    out = []
    for r in _safe_load_all():
        if (
            str(r.get("lifecycle") or "").lower() == "accepted"
            and _canonical_queue_symbol(r.get("symbol")) == target_sym
        ):
            oid_str = extract_order_id(r.get("order_id"))
            if oid_str and infer_record_owner(r) == target_owner and oid_str not in out:
                out.append(oid_str)
    return out


def load_validated():
    """Return the queue only when every durable record is semantically valid."""
    with _locked_queue(validate=True) as existing:
        return existing


def consolidate_deferred_streams():
    """Atomically keep only the newest pending record per deferred signal stream."""
    with _locked_queue() as existing:
        kept_deferred = []
        keep_ids = set()
        removed = []
        for index, item in sorted(
            enumerate(existing),
            key=lambda pair: safe_nonneg_float(pair[1].get("created_ts"), default=0.0),
            reverse=True,
        ):
            is_deferred = (
                _safe_to_evict_for_capacity(item)
                and item.get("last_failure_reason") in _CONSOLIDATED_DEFERRAL_REASONS
            )
            if is_deferred and any(_same_deferred_stream(item, newer) for newer in kept_deferred):
                removed.append(item)
                continue
            keep_ids.add(index)
            if is_deferred:
                kept_deferred.append(item)
        if removed:
            _write_nolock([item for index, item in enumerate(existing) if index in keep_ids])
        return removed


def get(record_id):
    """Return one current queue record by ID under lock, or ``None``."""
    if not record_id or isinstance(record_id, bool):
        return None
    with _locked_queue() as existing:
        for rec in existing:
            if rec.get("id") == record_id:
                return dict(rec)
    return None


def _append_order_history(rec, *, observed_at, status=None):
    """Retain a bounded audit trail when one logical intent has several orders."""
    order_id = rec.get("order_id")
    if not order_id:
        return
    row = {
        "order_id": str(order_id),
        "client_order_id": _record_client_order_id(rec),
        "revision": safe_nonneg_int(rec.get("revision"), default=0),
        "observed_at": resolve_now_ts(observed_at),
    }
    if status is not None:
        row.update({
            "status": str(getattr(status, "status", "") or ""),
            "venue_status": str(getattr(status, "venue_status", "") or ""),
            "filled_qty": safe_nonneg_float(getattr(status, "filled_qty", None), default=0.0),
            "cost": safe_nonneg_float(getattr(status, "cost", None), default=0.0),
            "fee": safe_nonneg_float(getattr(status, "fee", None), default=0.0),
        })
    history = list(rec.get("order_history") or [])
    if not history or history[-1].get("order_id") != row["order_id"]:
        history.append(row)
    else:
        history[-1].update(row)
    rec["order_history"] = history[-20:]


def claim(ids, now=None, lease_sec=None):
    """Lease entries without removing them, making crash recovery deterministic."""
    if ids is None or isinstance(ids, (str, bytes, bool)) or not (ids := {i for i in ids if not isinstance(i, bool) and i}):
        return []
    now = resolve_now_ts(now)
    lease_f = safe_pos_float(RETRY_CLAIM_LEASE_SEC if lease_sec is None else lease_sec, default=None)
    if lease_f is None:
        raise ValueError("claim lease must be finite and positive")
    claimed = []
    token = uuid.uuid4().hex
    with _locked_queue() as existing:
        for rec in existing:
            if rec.get("id") not in ids or safe_float(rec.get("claim_until"), default=0.0) > now:
                continue
            revision = max(0, exact_int(rec.get("revision"), 0))
            placement = dict(rec.get("place_kwargs") or {})
            pfx = rec.get("prefix") or placement.get("prefix") or "OR"
            placement.setdefault("client_order_id", _client_order_id(rec.get("id", ""), revision, prefix=pfx))
            rec["place_kwargs"] = placement
            rec["revision"] = revision
            rec["claim_token"] = token
            rec["claim_until"] = now + lease_f
            rec["claim_revision"] = revision
            claimed.append(dict(rec))
        if claimed:
            _write_nolock(existing)
    return claimed


_TERMINAL_RESET_KEYS = (
    "order_id", "accepted_ts", "last_status_ts", "last_status",
    "venue_status", "submit_status", "filled_qty", "filled_cost", "filled_fee",
)


def complete_claim(
    claimed, outcome, now=None, *, failure_reason=None,
    order=None, status=None, provider_name=None, submission_state=None,
):
    """Finalize one exact leased revision without losing accepted-order tracking."""
    if not isinstance(claimed, dict):
        return False
    allowed = {
        "success", "failure", "deferred", "release", "accepted",
        "observed", "status_error", "retry_terminal",
    }
    if outcome not in allowed:
        raise ValueError(f"invalid claim outcome: {outcome}")
    if outcome == "deferred" and not is_non_failure_deferral(failure_reason):
        raise ValueError(f"invalid non-failure deferral reason: {failure_reason!r}")
    if submission_state not in {None, "refused", "unknown"}:
        raise ValueError(f"invalid submission state: {submission_state!r}")
    now = resolve_now_ts(now)
    changed = False
    with _locked_queue() as existing:
        remaining = []
        for rec in existing:
            claimed_client_id = _record_client_order_id(claimed)
            if (
                rec.get("id") != claimed.get("id")
                or rec.get("claim_token") != claimed.get("claim_token")
                or exact_int(rec.get("revision"), 0) != exact_int(claimed.get("claim_revision"), -1)
                or claimed_client_id is None
                or _record_client_order_id(rec) != claimed_client_id
            ):
                remaining.append(rec)
                continue
            changed = True
            if outcome == "success":
                continue
            if outcome == "accepted":
                _set_safe_to_discard(rec, False)
                venue_qty = extract_orig_qty(order) or safe_pos_float(claimed.get("qty"), default=None)
                if venue_qty is not None:
                    _set_record_qty(rec, venue_qty)
                order_id = extract_order_id(order)
                if order_id is not None:
                    rec["order_id"] = order_id
                    rec["lifecycle"] = "accepted"
                    rec["accepted_ts"] = resolve_now_ts(now)
                    rec["last_status_ts"] = 0.0
                    rec["last_failure_reason"] = None
                    rec["submission_state"] = "accepted"
                    if provider_name is not None:
                        rec["provider_name"] = str(provider_name)
                    if isinstance(order, dict) and order.get("status") is not None:
                        rec["submit_status"] = str(order.get("status"))
                    _append_order_history(rec, observed_at=now)
                else:
                    rec["last_failure_reason"] = "response_without_order_id"
                    rec["attempts"] = max(
                        max(0, exact_int(rec.get("attempts"), 0)),
                        max(0, exact_int(claimed.get("attempts"), 0)),
                    ) + 1
                    rec["last_attempt_ts"] = now
            elif outcome == "observed":
                rec["last_status_ts"] = now
                rec["last_status"] = str(getattr(status, "status", "") or "")
                rec["venue_status"] = str(getattr(status, "venue_status", "") or "")
                status_orig = safe_pos_float(getattr(status, "orig_qty", None), default=None)
                if status_orig is not None:
                    _set_record_qty(rec, status_orig)
                for rec_key, attr_name in (
                    ("filled_qty", "filled_qty"),
                    ("filled_cost", "cost"),
                    ("filled_fee", "fee"),
                ):
                    rec[rec_key] = safe_nonneg_float(getattr(status, attr_name, None), default=0.0)
                rec.pop("status_error", None)
            elif outcome == "status_error":
                rec["last_status_ts"] = now
                rec["status_error"] = str(failure_reason) if failure_reason is not None else "unknown"
            elif outcome == "retry_terminal":
                filled_qty = safe_nonneg_float(getattr(status, "filled_qty", None), default=0.0)
                current_qty = safe_nonneg_float(rec.get("qty"), default=0.0)
                status_orig = safe_pos_float(getattr(status, "orig_qty", None), default=None)
                if status_orig is not None:
                    current_qty = status_orig
                    _set_record_qty(rec, current_qty)
                remaining_qty = max(0.0, current_qty - filled_qty)
                _append_order_history(rec, observed_at=now, status=status)
                rec["delivered_qty"] = safe_nonneg_float(rec.get("delivered_qty"), default=0.0) + min(current_qty, filled_qty)
                if remaining_qty <= max(1e-12, current_qty * 1e-9):
                    continue
                revision = max(0, exact_int(rec.get("revision"), 0)) + 1
                rec["revision"] = revision
                rec["qty"] = remaining_qty
                rec["attempts"] = 0
                rec["last_attempt_ts"] = now
                rec["ttl_started_ts"] = now
                if rec.get("ttl_sec") is not None:
                    rec["expires_at_ts"] = now + resolve_record_ttl(rec)
                _set_safe_to_discard(rec, False)
                rec["last_failure_reason"] = (
                    "terminal_" + str(getattr(status, "venue_status", "") or getattr(status, "status", "unknown")).lower()
                )
                rec["lifecycle"] = "submit_pending"
                rec["submission_state"] = "refused"
                for key in _TERMINAL_RESET_KEYS:
                    rec.pop(key, None)
                placement = dict(rec.get("place_kwargs") or {})
                rec["place_kwargs"] = {
                    **placement,
                    "client_order_id": _client_order_id(
                        rec.get("id", ""), revision, prefix=rec.get("prefix") or placement.get("prefix") or "OR"
                    ),
                }
            elif outcome == "failure":
                prev_attempts = max(
                    max(0, exact_int(rec.get("attempts"), 0)),
                    max(0, exact_int(claimed.get("attempts"), 0)),
                )
                prev_last_ts = safe_pos_float(rec.get("last_attempt_ts"), default=None)
                rec["attempts"] = prev_attempts + 1
                rec["last_attempt_ts"] = now
                next_reason = str(failure_reason) if failure_reason is not None else None
                _shift_ttl_after_deferral(
                    rec, rec.get("last_failure_reason"), next_reason, prev_attempts, prev_last_ts, now
                )
                rec["last_failure_reason"] = next_reason
            elif outcome == "deferred":
                rec["last_attempt_ts"] = now
                rec["last_failure_reason"] = str(failure_reason)
            if submission_state is not None:
                rec["submission_state"] = submission_state
                _set_safe_to_discard(rec, submission_state == "refused")
            elif outcome == "retry_terminal":
                _set_safe_to_discard(rec, False)
            for key in ("claim_token", "claim_until", "claim_revision"):
                rec.pop(key, None)
            remaining.append(rec)
        if changed:
            _write_nolock(remaining)
    return changed


def activate_claimed_replacement(claimed, qty, now=None):
    """Atomically make a confirmed-cancel replacement eligible for submission."""
    if not isinstance(claimed, dict):
        return None
    now = safe_pos_float(time.time() if now is None else now, default=None)
    qty = safe_nonneg_float(qty, default=None)
    if now is None or qty is None:
        return None
    result = None
    replaced_order_id = None
    replaced_symbol = None
    with _locked_queue() as existing:
        for index, rec in enumerate(existing):
            if not (
                rec.get("id") == claimed.get("id")
                and rec.get("claim_token") == claimed.get("claim_token")
                and exact_int(rec.get("revision"), 0) == exact_int(claimed.get("claim_revision"), -1)
            ):
                continue
            if str(rec.get("lifecycle") or "").lower() != "awaiting_cancel":
                break
            current_qty = safe_nonneg_float(rec.get("qty"), default=0.0)
            if qty > current_qty + max(1e-12, current_qty * 1e-9):
                break
            replaced_order_id = extract_order_id(rec.get("replaces_order_id")) or str(rec.get("replaces_order_id") or "").strip()
            replaced_symbol = _canonical_queue_symbol(rec.get("symbol"))
            if qty <= max(1e-12, current_qty * 1e-9):
                existing.pop(index)
                result = "resolved"
            else:
                _set_record_qty(rec, qty)
                rec.update(
                    lifecycle="submit_pending",
                    last_failure_reason="submit_pending",
                    submission_state="refused",
                    ttl_started_ts=now,
                    last_attempt_ts=0.0,
                )
                _set_safe_to_discard(rec, False)
                result = "activated"
            break
        if result is not None:
            if replaced_order_id and replaced_symbol:
                existing = [
                    item for item in existing
                    if not (
                        str(item.get("lifecycle") or "").lower() == "accepted"
                        and _canonical_queue_symbol(item.get("symbol")) == replaced_symbol
                        and (extract_order_id(item.get("order_id")) or str(item.get("order_id") or "").strip()) == replaced_order_id
                    )
                ]
            _write_nolock(existing)
    return result


def resolve_accepted_order(symbol, order_id) -> bool:
    """Remove an ``accepted`` tracker after intentional external cancellation."""
    if isinstance(symbol, bool) or isinstance(order_id, bool):
        return False
    target_sym = _canonical_queue_symbol(symbol)
    target_oid = extract_order_id(order_id)
    if not target_sym or not target_oid:
        return False
    removed_records = _remove_matching_locked(
        lambda rec: (
            str(rec.get("lifecycle") or "").lower() == "accepted"
            and _canonical_queue_symbol(rec.get("symbol")) == target_sym
            and extract_order_id(rec.get("order_id")) == target_oid
        )
    )
    if removed_records:
        try:
            audit = ExecutionAudit()
            for rec in removed_records:
                rec_pk = rec.get("place_kwargs") if isinstance(rec.get("place_kwargs"), dict) else {}
                audit.record(
                    "order_terminal",
                    intent_id=str(rec.get("intent_id") or f"outbox-{rec.get('id')}"),
                    venue=str(rec.get("provider_name") or "routed"),
                    symbol=str(rec.get("symbol") or target_sym),
                    side=str(rec.get("side") or "").lower() or None,
                    kind=rec.get("kind") or rec_pk.get("kind"),
                    client_order_id=_record_client_order_id(rec),
                    order_id=rec.get("order_id") or target_oid,
                    revision=rec.get("revision"),
                    status="canceled",
                    venue_status="CANCELED",
                    filled_qty=safe_nonneg_float(rec.get("filled_qty"), default=0.0) or 0.0,
                    cost=safe_nonneg_float(rec.get("filled_cost"), default=0.0) or 0.0,
                    fee=safe_nonneg_float(rec.get("filled_fee"), default=0.0) or 0.0,
                    retry=False,
                )
        except Exception:
            pass
    return bool(removed_records)


@dataclass(frozen=True)
class OutboxStatusTransition:
    """Result of one accepted outbox record advancing from one venue snapshot."""

    action: str  # observed, filled, retry_terminal, terminal
    status_changed: bool = False
    remaining_qty: float = 0.0

    def __post_init__(self):
        if self.action not in {"observed", "filled", "retry_terminal", "terminal"}:
            raise ValueError(f"invalid outbox status action: {self.action!r}")


def advance_claimed_status(claimed: dict, status: OrderStatus, now=None) -> OutboxStatusTransition:
    """Persist one accepted order-status transition without venue I/O."""
    if not isinstance(status, OrderStatus):
        raise TypeError("status must be OrderStatus")
    if str(claimed.get("lifecycle") or "").lower() != "accepted":
        raise ValueError("status transition requires an accepted claimed record")

    if not status.terminal:
        fingerprint = (
            str(status.status),
            safe_nonneg_float(status.filled_qty, default=0.0),
            str(status.venue_status),
        )
        previous = (
            str(claimed.get("last_status") or ""),
            safe_nonneg_float(claimed.get("filled_qty"), default=0.0),
            str(claimed.get("venue_status") or ""),
        )
        complete_claim(claimed, "observed", now, status=status)
        return OutboxStatusTransition("observed", status_changed=fingerprint != previous)

    if status.status == "closed":
        complete_claim(claimed, "success", now, status=status)
        return OutboxStatusTransition("filled", status_changed=True)

    current_qty = safe_pos_float(getattr(status, "orig_qty", None), default=None) or safe_nonneg_float(claimed.get("qty"), default=0.0)
    remaining_qty = max(0.0, current_qty - status.filled_qty)
    if str(status.venue_status or "").upper() in {"REJECTED", "EXPIRED"} or status.status == "expired":
        complete_claim(claimed, "retry_terminal", now, status=status)
        return OutboxStatusTransition("retry_terminal", status_changed=True, remaining_qty=remaining_qty)

    changed = complete_claim(claimed, "success", now, status=status)
    return OutboxStatusTransition("terminal", status_changed=bool(changed), remaining_qty=remaining_qty)


def discard_expired(snapshots, now=None):
    """Remove only records that remain the exact expired durable revision."""
    if not snapshots:
        return []
    now = safe_float(time.time() if now is None else now, default=None)
    if now is None:
        return []
    exact = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        rev = exact_int(snapshot.get("revision"), -1)
        if rev >= 0:
            key = (snapshot.get("id"), rev, _record_client_order_id(snapshot))
            if key[0] and key[2]:
                exact.add(key)
    if not exact:
        return []
    return _remove_matching_locked(
        lambda record: (
            (record.get("id"), exact_int(record.get("revision"), 0), _record_client_order_id(record)) in exact
            and is_expired(record, now)
        ),
        validate=True,
    )


def discard(ids):
    """Atomically remove records by ID without submitting them (expiry/give-up)."""
    if ids is None or isinstance(ids, (str, bytes, bool)) or not (ids := {i for i in ids if not isinstance(i, bool) and i}):
        return []
    return _remove_matching_locked(lambda rec: rec.get("id") in ids)


def resolve(symbol, side):
    """Legacy helper: remove only unaccepted pending records by symbol and side."""
    if isinstance(symbol, bool) or isinstance(side, bool):
        return 0
    target_sym = _canonical_queue_symbol(symbol)
    side_u = normalize_side(side)
    return len(
        _remove_matching_locked(
            lambda rec: (
                _canonical_queue_symbol(rec.get("symbol")) == target_sym
                and str(rec.get("side") or "").strip().upper() == side_u
                and str(rec.get("lifecycle") or "submit_pending").lower() == "submit_pending"
            )
        )
    )


def _write_nolock(items):
    """Replace the queue file atomically while the caller holds the lock."""
    atomic_write_jsonl(QUEUE_FILE, items)


def rewrite(items):
    """Replace the queue with ``items`` atomically while holding the lock."""
    os.makedirs(os.path.dirname(str(resolve_cachedb_filepath(QUEUE_FILE))), exist_ok=True)
    with FileLock(LOCK_FILE):
        _write_nolock(items)


_FAST_IMPULSE_KIND_PREFIXES = (
    "small_window_",
    "accel_surge_",
    "accel_plunge_",
    "kalman_primary_",
)


def resolve_record_ttl(rec):
    """Return the effective active retry TTL for ``rec``, honoring custom ``ttl_sec``."""
    base_ttl = safe_pos_float(RETRY_TTL_SEC, default=1800.0)
    if isinstance(rec, dict):
        custom_ttl = safe_pos_float(rec.get("ttl_sec"), default=None)
        if custom_ttl is not None:
            return min(custom_ttl, MAX_GLOBAL_ORDER_TTL_SEC)
        if _clean_str(rec.get("kind")).startswith(_FAST_IMPULSE_KIND_PREFIXES):
            raw_kw = rec.get("place_kwargs")
            kw_hours = safe_pos_float(raw_kw.get("hours") if isinstance(raw_kw, dict) else None, default=None)
            fast_ttl = kw_hours * 3600.0 if kw_hours is not None else 1800.0
            return min(fast_ttl, base_ttl, MAX_GLOBAL_ORDER_TTL_SEC)
    return base_ttl


def is_expired(rec, now=None):
    """Return true after active retry TTL or hard attempt limit."""
    if not isinstance(rec, dict):
        return True
    if (
        str(rec.get("lifecycle") or "submit_pending").lower() in {"accepted", "awaiting_cancel"}
        or is_possibly_submitted(rec)
    ):
        return False
    now_f = safe_pos_float(time.time() if now is None else now, default=None)
    claim_until = safe_nonneg_float(rec.get("claim_until"), default=0.0) or 0.0
    created = safe_pos_float(rec.get("created_ts"), default=None)
    ttl_started = safe_pos_float(rec.get("ttl_started_ts"), default=created)
    raw_attempts = rec.get("attempts")
    attempts = 0 if raw_attempts is None else exact_nonneg_int(raw_attempts, default=None)
    if attempts is None or now_f is None or created is None or ttl_started is None:
        return True
    if claim_until > now_f:
        return False
    raw_expires_at = rec.get("expires_at_ts")
    if raw_expires_at is not None:
        expires_at = safe_pos_float(raw_expires_at, default=None)
        if expires_at is None or now_f > expires_at:
            return True
    effective_ttl = resolve_record_ttl(rec)
    if _clean_str(rec.get("kind")).startswith(_FAST_IMPULSE_KIND_PREFIXES) and (now_f - created) > effective_ttl:
        return True
    if is_non_failure_deferral(rec.get("last_failure_reason")):
        return False
    return (now_f - ttl_started) > effective_ttl or (RETRY_MAX_ATTEMPTS > 0 and attempts >= RETRY_MAX_ATTEMPTS)


def is_due(rec, now=None):
    """Return true once the retry interval has elapsed since creation or last attempt."""
    if not isinstance(rec, dict):
        return False
    now_f = safe_float(time.time() if now is None else now, default=None)
    claim_until = safe_float(rec.get("claim_until", 0) or 0, default=None)
    created = safe_float(rec.get("created_ts", 0) or 0, default=None)
    if now_f is None or claim_until is None or created is None:
        return False
    lifecycle = str(rec.get("lifecycle") or "submit_pending").lower()
    if lifecycle in {"accepted", "awaiting_cancel"}:
        last_st = safe_float(rec.get("last_status_ts", 0) or 0, default=None)
        acc_ts = safe_float(rec.get("accepted_ts", 0) or 0, default=None)
        if last_st is None or acc_ts is None:
            return False
        base = max(last_st, acc_ts, created)
    else:
        last_att = safe_float(rec.get("last_attempt_ts", 0) or 0, default=None)
        if last_att is None:
            return False
        base = max(last_att, created)
    return claim_until <= now_f and (now_f - base) >= RETRY_INTERVAL_SEC


def is_retry_price_favorable(side, current_price, requested_price, tol=None) -> bool:
    """Return whether ``current_price`` is within ``tol`` of ``requested_price`` for ``side``."""
    current_price = safe_pos_float(current_price, default=None)
    req = safe_pos_float(requested_price, default=None)
    tol = safe_nonneg_float(RETRY_PRICE_TOL if tol is None else tol, default=None)
    if current_price is None or req is None or tol is None:
        return False
    side_u = normalize_side(side)
    if side_u == "SELL":
        return current_price >= req * (1.0 - tol)
    if side_u == "BUY":
        return current_price <= req * (1.0 + tol)
    return False


def price_gate_ok(rec, current_price, tol=None):
    """Return whether the current price is favorable relative to the stored request."""
    if not isinstance(rec, dict):
        return False
    return is_retry_price_favorable(rec.get("side"), current_price, rec.get("requested_price"), tol=tol)


def discard_refused_intent_records(intent_id: str) -> list[str]:
    """Discard queued records matching ``intent_id`` whose ``submission_state`` is ``refused``."""
    iid_str = "" if isinstance(intent_id, bool) else str(intent_id or "").strip()
    if not iid_str:
        return []
    rm_ids = [
        str(r["id"])
        for r in load_all()
        if isinstance(r, dict)
        and str(r.get("intent_id") or "").strip() == iid_str
        and str(r.get("submission_state") or "").lower() == "refused"
        and r.get("id") is not None
    ]
    if rm_ids:
        discard(rm_ids)
    return rm_ids


def discard_superseded_symbol_records(
    symbol: str,
    want_side: Optional[str],
    *,
    canceled_ids: Iterable[str] = (),
    ai_order_ids: Optional[set[str]] = None,
    include_bot_orders: bool = False,
    any_side: bool = False,
    cancel_fn_is_mock: bool = False,
) -> list[str]:
    """Discard superseded ``submit_pending`` / ``awaiting_cancel`` retry records for ``symbol``."""
    sym_str = "" if isinstance(symbol, bool) else str(symbol or "").strip().upper()
    if not sym_str:
        return []
    default_q = os.path.abspath(str(resolve_cachedb_filepath("order_retry_queue.jsonl")))
    active_q = os.path.abspath(str(QUEUE_FILE))
    if active_q == default_q and cancel_fn_is_mock:
        return []
    canceled_set = {str(x) for x in (canceled_ids or ()) if x}
    ai_oids = set(ai_order_ids or ())
    to_discard: list[str] = []
    for rec in load_all():
        if not isinstance(rec, dict) or str(rec.get("symbol") or "").strip().upper() != sym_str:
            continue
        rec_id = str(rec.get("id") or "").strip()
        if not rec_id:
            continue
        rec_oid = str(rec.get("order_id") or "").strip()
        if rec_oid and rec_oid in canceled_set:
            to_discard.append(rec_id)
            continue
        lc = str(rec.get("lifecycle") or "submit_pending").strip().lower()
        if lc not in ("submit_pending", "awaiting_cancel"):
            continue
        rec_side = normalize_side(rec.get("side"))
        if not any_side and rec_side != want_side:
            continue
        rec_kw = rec.get("place_kwargs") if isinstance(rec.get("place_kwargs"), dict) else {}
        rec_probe = {
            "orderId": rec_oid,
            "clientOrderId": rec_kw.get("client_order_id") or rec.get("prefix") or "",
            "intent_id": rec.get("intent_id") or "",
        }
        is_ai_rec = resolve_order_actor(rec_probe, ai_oids) == "AI"
        if not is_ai_rec and not include_bot_orders:
            continue
        to_discard.append(rec_id)
    if to_discard:
        discard(to_discard)
    return to_discard


MarketDataProvider.bind_order_retry_hooks(
    discard_refused_intent_fn=discard_refused_intent_records,
    discard_superseded_symbol_fn=discard_superseded_symbol_records,
    accepted_owner_map_fn=get_accepted_order_owner_map,
    resolve_accepted_order_fn=resolve_accepted_order,
)
