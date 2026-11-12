"""Observational lifecycle audit for ``StrategyExecutor`` orders.

This layer neither authorizes an order nor changes the submitted financial values.
Writes are best-effort, so an audit I/O failure does not change the executor result.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

from botcore import (
    _parse_bool_flag,
    bind_lazy_module_exports,
    clean_order_id,
    clean_order_id_str,
    float_env,
    lru_dict_put,
    optional_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
    sanitize_identifier,
)
from cachepaths import (
    canonical_symbol_key,
    clean_symbol_key,
    normalize_venue_name,
)
from state_io import append_jsonl_record, read_jsonl_dicts
from .base import MarketDataProvider
from .intent_state import (
    _extract_intent_id,
    _intent_order_ids,
    _intent_venue_symbol_side_key,
    _record_intent_state,
    collect_known_ai_order_ids,
    durable_intents_lock,
    extract_intent_limit_price,
    extract_intent_notional,
    extract_intent_order_type,
    extract_intent_side,
    intent_venue_client_order_ids,
    load_durable_intents_state,
    normalize_trading_decisions,
    save_durable_intents_state,
    upsert_durable_intent,
    validate_and_build_intent_dict,
)
from .ohlc_utils import ClosedPriceSeries, require_pos_ohlc_interval
from .order_ohlc_utils import format_venue_intent_client_order_id
from .strategy_executor import (
    OrderStatus,
    ProviderError,
    SubmissionOutcome,
    SubmissionRefused,
    capture_submission,
    reconciliation_capabilities_of,
)


_WRITE_LOCK = threading.Lock()
_UUID_HEX_SUFFIX = re.compile(r"([0-9a-fA-F]{32})$")
_TERMINAL_CANCEL_STATUSES = ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "CANCELED_OR_EXPIRED")
_OPEN_WORKING_STATUSES = ("", "NEW", "OPEN", "RESTING", "PENDING", "WORKING")


def _parse_cache_max() -> int:
    return max(100, safe_pos_int(float_env("EXECUTION_AUDIT_CACHE_MAX"), 10000))


_CACHE_MAX = _parse_cache_max()


def _slug(value: object) -> str:
    return sanitize_identifier(value, sep="-", allow_dots_hyphens=True, strip_chars="-")


def _json_safe_value(val: Any) -> Any:
    if isinstance(val, (float, Decimal)):
        return safe_float(val, default=None)
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            cleaned = _json_safe_value(v)
            if cleaned is not None:
                out[str(k)] = cleaned
        return out
    if isinstance(val, (list, tuple, set, frozenset)):
        return [_json_safe_value(v) for v in val]
    return val


def new_intent_id(venue: str, symbol: str, kind: Optional[str] = None) -> str:
    """Create a unique intent ID before submission; callers persist it as needed."""
    v_slug = _slug(venue).lower()
    v_norm = normalize_venue_name(v_slug) or v_slug
    prefix = "-".join((v_norm, _slug(symbol), _slug(kind or "order").lower()))
    return f"{prefix}-{uuid.uuid4().hex}"


def intent_client_order_id(venue: str, intent_id: str) -> Optional[str]:
    """Encode an intent in the venue's client-ID format without additional state.

    Preserve the complete 128-bit UUID. Legacy/nonstandard intents use a
    deterministic hash so the same intent produces the same identifier after restart.
    """
    if isinstance(intent_id, (bool, list, tuple, dict, set, frozenset)):
        return None
    raw = clean_order_id(intent_id)
    if not raw:
        return None
    match = _UUID_HEX_SUFFIX.search(raw)
    token = (match.group(1).lower() if match else
             hashlib.blake2s(raw.encode("utf-8"), digest_size=16).hexdigest())
    v_slug = _slug(venue).lower()
    return format_venue_intent_client_order_id(normalize_venue_name(v_slug) or v_slug, token)


_DEFAULT_AUDIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logger", "execution_audit"
)


class ExecutionAudit:
    """Best-effort JSONL writer shared by Kraken, T212, HL, and Binance."""

    def __init__(self, directory: Optional[str] = None,
                 clock: Callable[[], float] = time.time):
        self.directory = directory or optional_env("EXECUTION_AUDIT_DIR") or _DEFAULT_AUDIT_DIR
        self._clock = clock

    def record(self, event: str, *, intent_id: str, venue: str, symbol: str,
               **fields) -> bool:
        """Append one serialized event; return ``False`` when audit is unavailable.

        The process lock and Linux ``flock`` prevent interleaved writers, but the
        append is not fsynced and therefore is not a durability boundary.
        """
        try:
            ts_val = safe_pos_float(self._clock(), None)
            ts = ts_val if (ts_val is not None and ts_val < 32503680000) else time.time()
            payload = {
                "ts": ts,
                "ts_utc": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                "event": str(event),
                "intent_id": str(intent_id),
                "venue": str(venue),
                "symbol": str(symbol),
            }
            for key, value in fields.items():
                cleaned = _json_safe_value(value)
                if cleaned is not None:
                    payload[key] = cleaned
            day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
            path = os.path.join(self.directory, f"execution_audit_{day}.jsonl")
            with _WRITE_LOCK:
                append_jsonl_record(
                    path,
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                    flock=True,
                )
            return True
        except Exception as exc:  # noqa: BLE001 - auditing must never stop trading
            print(f"[execution_audit] the write failed: {exc}")
            return False


class AuditedStrategyExecutor:
    """Execution decorator that records lifecycle calls without changing decisions."""

    def __init__(self, executor, audit: Optional[ExecutionAudit] = None,
                 venue: Optional[str] = None):
        self._executor = executor
        self.audit = audit or ExecutionAudit()
        self.name = venue or str(getattr(executor, "name", executor.__class__.__name__))
        self._intent_by_order = OrderedDict()
        self._last_status = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _norm_symbol_key(symbol: object) -> str:
        if symbol is None or isinstance(symbol, bool):
            return ""
        return canonical_symbol_key(symbol) or clean_symbol_key(symbol)

    def _remember(self, symbol: str, order_id: str, intent_id: str) -> None:
        oid_key = clean_order_id_str(order_id)
        if not oid_key:
            return
        sym_key = self._norm_symbol_key(symbol)
        with self._lock:
            lru_dict_put(self._intent_by_order, (sym_key, oid_key), intent_id, _CACHE_MAX)

    def _intent(self, symbol: str, order_id: str, explicit: Optional[str] = None) -> str:
        clean_exp = clean_order_id_str(explicit)
        if clean_exp:
            self._remember(symbol, order_id, clean_exp)
            return clean_exp
        oid_key = clean_order_id_str(order_id)
        if oid_key:
            sym_key = self._norm_symbol_key(symbol)
            with self._lock:
                known = self._intent_by_order.get((sym_key, oid_key))
            if known:
                return known
        return f"recovered-{_slug(self.name).lower()}-{_slug(symbol)}-{_slug(oid_key or order_id)}"

    def submit_order_with_intent(self, intent_id: str, symbol: str, side: str, qty: float,
                                 price: Optional[float] = None, *, market: bool = False,
                                 kind: Optional[str] = None,
                                 reference_price: Optional[float] = None,
                                 client_order_id: Optional[str] = None,
                                 cache_permit=None,
                                 post_only: Optional[bool] = None) -> str:
        outcome = self.submit_outcome_with_intent(
            intent_id, symbol, side, qty, price, market=market, kind=kind,
            reference_price=reference_price, client_order_id=client_order_id,
            cache_permit=cache_permit, post_only=post_only,
        )
        if outcome.state == "refused":
            raise SubmissionRefused(outcome.reason)
        if outcome.state == "unknown":
            raise ProviderError(outcome.reason or "submission outcome is unknown")
        return str(outcome.order_id)

    def submit_outcome_with_intent(
            self, intent_id: str, symbol: str, side: str, qty: float,
            price: Optional[float] = None, *, market: bool = False,
            kind: Optional[str] = None, reference_price: Optional[float] = None,
            client_order_id: Optional[str] = None,
            cache_permit=None,
            post_only: Optional[bool] = None,
            _requested_recorded: bool = False) -> SubmissionOutcome:
        """Submit once and retain refusal versus response-loss ambiguity."""
        clean_cid = clean_order_id_str(client_order_id) or None
        client_order_id = clean_cid or intent_client_order_id(self.name, intent_id)
        is_market = _parse_bool_flag(market, False)
        fields = {
            "side": str(side).lower(), "qty": qty, "price": price,
            "market": is_market, "kind": kind,
            "reference_price": reference_price,
            "client_order_id": client_order_id,
        }
        if not _requested_recorded:
            self.audit.record("submit_requested", intent_id=intent_id, venue=self.name,
                              symbol=symbol, **fields)
        submit_kwargs = {"market": is_market, "kind": kind}
        if client_order_id is not None:
            submit_kwargs["client_order_id"] = client_order_id
        if cache_permit is not None:
            submit_kwargs["cache_permit"] = cache_permit
        if post_only is not None:
            submit_kwargs["post_only"] = _parse_bool_flag(post_only, True)
        outcome = capture_submission(
            lambda: self._executor.submit_order(
                symbol, side, qty, price, **submit_kwargs))
        if outcome.state == "accepted":
            self._remember(symbol, str(outcome.order_id), intent_id)
            event = "submit_accepted"
        elif outcome.state == "refused":
            event = "submit_refused"
        else:
            event = "submit_unknown"
        self.audit.record(event, intent_id=intent_id, venue=self.name,
                          symbol=symbol, order_id=outcome.order_id,
                          submission_state=outcome.state,
                          reason=outcome.reason, **fields)
        return outcome

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None,
                     cache_permit=None,
                     post_only: Optional[bool] = None) -> str:
        return self.submit_order_with_intent(
            new_intent_id(self.name, symbol, kind), symbol, side, qty, price,
            market=market, kind=kind, client_order_id=client_order_id,
            cache_permit=cache_permit, post_only=post_only,
        )

    def _call_executor_or_record(
        self, fn, error_event: str, intent_id: str, symbol: str, order_id: str,
    ):
        try:
            return fn(symbol, order_id)
        except Exception as exc:
            self.audit.record(
                error_event, intent_id=intent_id, venue=self.name, symbol=symbol,
                order_id=str(order_id), error_type=exc.__class__.__name__, error=str(exc),
            )
            raise

    def order_status_with_intent(self, intent_id: str, symbol: str,
                                 order_id: str) -> OrderStatus:
        intent_id = self._intent(symbol, order_id, intent_id)
        status = self._call_executor_or_record(
            self._executor.order_status, "status_error", intent_id, symbol, order_id,
        )
        fingerprint = (
            status.status, status.filled_qty, status.cost, status.fee,
            status.venue_status,
        )
        oid_key = clean_order_id_str(order_id)
        key = (self._norm_symbol_key(symbol), oid_key)
        with self._lock:
            changed = self._last_status.get(key) != fingerprint
            lru_dict_put(self._last_status, key, fingerprint, _CACHE_MAX)
        if changed:
            self.audit.record(
                "order_status", intent_id=intent_id, venue=self.name, symbol=symbol,
                order_id=oid_key or str(order_id), status=status.status,
                venue_status=status.venue_status,
                filled_qty=status.filled_qty, cost=status.cost, fee=status.fee,
            )
        return status

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        return self.order_status_with_intent(
            self._intent(symbol, order_id), symbol, order_id,
        )

    def cancel_order_with_intent(self, intent_id: str, symbol: str, order_id: str) -> None:
        intent_id = self._intent(symbol, order_id, intent_id)
        self.audit.record(
            "cancel_requested", intent_id=intent_id, venue=self.name,
            symbol=symbol, order_id=str(order_id),
        )
        self._call_executor_or_record(
            self._executor.cancel_order, "cancel_rejected", intent_id, symbol, order_id,
        )
        self.audit.record(
            "cancel_accepted", intent_id=intent_id, venue=self.name,
            symbol=symbol, order_id=str(order_id),
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        return self.cancel_order_with_intent(
            self._intent(symbol, order_id), symbol, order_id,
        )

    def order_by_client_id(self, symbol: str, client_order_id: str):
        """Delegate deterministic recovery without inventing audit acceptance."""
        clean_cid = clean_order_id(client_order_id)
        if not clean_cid:
            raise ProviderError("client_order_id is required")
        lookup = getattr(self._executor, "order_by_client_id", None)
        if not callable(lookup):
            raise RuntimeError(f"{self.name}: order_by_client_id is unsupported")
        return lookup(symbol, clean_cid)

    def reconciliation_capabilities(self):
        """Preserve the wrapped adapter's venue facts through decoration."""
        return reconciliation_capabilities_of(self._executor)

    def execution_enabled(self) -> bool:
        """Preserve the wrapped adapter's live-order gate through decoration."""
        method = getattr(self._executor, "execution_enabled", None)
        return True if not callable(method) else _parse_bool_flag(method(), False)

    def get_current_price(self, symbol: str):
        return self._executor.get_current_price(symbol)

    def pair_precision(self, symbol: str):
        return self._executor.pair_precision(symbol)

    def free_balance(self, asset: str):
        return self._executor.free_balance(asset)

    def preflight_order(self, symbol: str, side: str, qty: float, price=None, *, market: bool = False, kind: Optional[str] = None) -> Any:
        """Run the adapter's optional preflight before a submit is audited."""
        return self._call_opt("preflight_order", symbol, side, qty, price, market=_parse_bool_flag(market, False), kind=kind)

    def ohlc_closes(self, symbol: str, interval_min: int):
        return self._executor.ohlc_closes(symbol, interval_min)

    @property
    def _inner(self):
        return self._executor

    def _call_opt(self, attr: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        reader = getattr(self._executor, attr, None)
        return reader(*args, **kwargs) if callable(reader) else default

    def open_orders(self, symbol: str):
        return self._call_opt("open_orders", symbol, default=[])

    def get_orders(self, symbol: str, side: Optional[str] = None, lookback: Optional[float] = None):
        return self._call_opt("get_orders", symbol, side, lookback, default=[])

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        return self._call_opt("position_cost_basis", symbol, held_qty, since_s)

    def _pair_precision_attr(self, symbol: str, attr: str, default: Any = None) -> Any:
        return MarketDataProvider._pair_precision_attr(self, symbol, attr, default)

    def round_quantity(self, symbol: str, qty: float) -> float:
        fn = getattr(self._executor, "round_quantity", None)
        return fn(symbol, qty) if callable(fn) else MarketDataProvider.round_quantity(self, symbol, qty)

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float) -> float:
        fn = getattr(self._executor, "round_price", None)
        return fn(symbol, price) if callable(fn) else MarketDataProvider.round_price(self, symbol, price)

    def price_tick_size(self, symbol: str, reference_price: Optional[float] = None) -> float:
        fn = getattr(self._executor, "price_tick_size", None)
        if callable(fn):
            return safe_nonneg_float(fn(symbol, reference_price=reference_price), default=0.0) or 0.0
        return MarketDataProvider.price_tick_size(self, symbol, reference_price=reference_price)

    def min_order_qty(self, symbol: str) -> float:
        fn = getattr(self._executor, "min_order_qty", None)
        if callable(fn):
            return safe_nonneg_float(fn(symbol), default=0.0) or 0.0
        return MarketDataProvider.min_order_qty(self, symbol)

    def ohlc_series(self, symbol: str, interval_min: int):
        """Preserve optional candle timestamps without expanding the protocol."""
        reader = getattr(self._executor, "ohlc_series", None)
        if callable(reader):
            return reader(symbol, interval_min)
        interval_i = require_pos_ohlc_interval(symbol, interval_min)
        return ClosedPriceSeries(
            tuple(self._executor.ohlc_closes(symbol, interval_i) or ()),
            interval_i,
        )


def _extract_audit_fill_metrics(
    rec: dict[str, Any],
    *,
    fallback_ref_price: bool = False,
) -> tuple[str, str, Any, Optional[float], Any, Optional[float], Optional[float], bool]:
    """Extract normalized venue/local status, filled qty, cost, price, and fill flag from an audit record."""
    ev = str(rec.get("event") or "").strip().lower()
    v_stat = str(rec.get("venue_status") or "").strip().upper()
    l_stat = str(rec.get("status") or "").strip().lower()
    term_filled = v_stat in ("FILLED", "CLOSED") or l_stat in ("closed", "filled")
    filled_qty = (
        rec.get("filled_qty")
        if rec.get("filled_qty") is not None
        else (rec.get("qty") if (term_filled and ev not in ("submit_accepted", "submit_unknown", "order_recovered")) else None)
    )
    f_q = safe_pos_float(filled_qty, default=None)
    raw_cost = rec.get("cost") if (f_q is not None or term_filled or v_stat == "PARTIALLY_FILLED" or l_stat == "partially_filled") else None
    cost_f = safe_pos_float(raw_cost, default=None)
    f_px = safe_pos_float(rec.get("price"), default=None)
    if f_px is None and cost_f is not None and f_q is not None and f_q > 0:
        f_px = round(cost_f / f_q, 8)
    if f_px is None and fallback_ref_price:
        f_px = safe_pos_float(rec.get("reference_price"), default=None)
    if raw_cost is None and f_q is not None and f_px is not None:
        raw_cost = round(f_q * f_px, 8)
        cost_f = safe_pos_float(raw_cost, default=None)
    is_fill = (
        term_filled
        or v_stat == "PARTIALLY_FILLED"
        or l_stat in ("partially_filled",)
        or (f_q is not None and f_q > 0.0)
    )
    return v_stat, l_stat, filled_qty, f_q, raw_cost, cost_f, f_px, is_fill


_AUDIT_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_AUDIT_OWNER_EVENTS = frozenset({"submit_accepted", "order_terminal", "order_terminal_retry"})
_AUDIT_RECONCILE_EVENTS = frozenset({
    "submit_accepted",
    "submit_unknown",
    "order_recovered",
    "order_status",
    "order_terminal",
    "order_terminal_retry",
})
_AUDIT_FILE_RECORDS_CACHE: dict[str, tuple[tuple[float, int], list[dict[str, Any]]]] = {}


def _read_audit_intent_records(audit_path: str) -> list[dict[str, Any]]:
    """Return reconciliation-relevant audit records of one file, cached by ``(mtime, size)`` (append-only files)."""
    try:
        st = os.stat(audit_path)
        sig = (st.st_mtime, st.st_size)
    except OSError:
        return []
    cached = _AUDIT_FILE_RECORDS_CACHE.get(audit_path)
    if cached is not None and cached[0] == sig:
        return cached[1]
    recs = [
        rec for rec in read_jsonl_dicts(audit_path)
        if str(rec.get("event") or "") in _AUDIT_RECONCILE_EVENTS and _extract_intent_id(rec)
    ]
    _AUDIT_FILE_RECORDS_CACHE[audit_path] = (sig, recs)
    return recs


def find_order_audit_record(
    order_id: Any = None,
    client_order_id: Any = None,
    *,
    symbol: Any = None,
    event_ts_s: Optional[float] = None,
    audit_dir: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Return the most recent execution-audit record matching ``order_id`` or ``client_order_id``."""
    target_oid = clean_order_id(order_id)
    target_cid = clean_order_id(client_order_id)
    if not target_oid and not target_cid:
        return None
    eff_dir = audit_dir or optional_env("EXECUTION_AUDIT_DIR") or _DEFAULT_AUDIT_DIR
    if not os.path.isdir(eff_dir):
        return None
    target_sym = canonical_symbol_key(symbol) if symbol and not isinstance(symbol, bool) else ""
    ref_ts = safe_pos_float(event_ts_s, default=None)
    min_day = max_day = ""
    if ref_ts is not None and ref_ts > 0.0:
        min_day = datetime.fromtimestamp(max(0.0, ref_ts - 5 * 86400.0), timezone.utc).strftime("%Y-%m-%d")
        max_day = datetime.fromtimestamp(ref_ts + 86400.0, timezone.utc).strftime("%Y-%m-%d")
    all_paths = sorted(
        glob.glob(os.path.join(glob.escape(eff_dir), "execution_audit_*.jsonl")),
        reverse=True,
    )
    if not min_day:
        all_paths = all_paths[:5]
    for audit_path in all_paths:
        file_day = os.path.basename(audit_path)[len("execution_audit_"):-len(".jsonl")]
        if min_day and _AUDIT_DAY_RE.fullmatch(file_day) and (file_day < min_day or file_day > max_day):
            continue
        for rec in reversed(_read_audit_intent_records(audit_path)):
            if str(rec.get("event") or "") not in _AUDIT_OWNER_EVENTS:
                continue
            rec_oid = clean_order_id(rec.get("order_id"))
            rec_cid = clean_order_id(rec.get("client_order_id"))
            if not ((target_oid and rec_oid == target_oid) or (target_cid and rec_cid == target_cid)):
                continue
            if target_sym:
                rec_sym = canonical_symbol_key(rec.get("symbol"))
                if rec_sym and rec_sym != target_sym:
                    continue
            if ref_ts is not None and ref_ts > 0.0:
                rec_ts = safe_pos_float(rec.get("ts"), default=None)
                if rec_ts is not None and abs(rec_ts - ref_ts) > 6 * 86400.0:
                    continue
            if rec.get("client_order_id") or rec.get("kind") or rec.get("intent_id"):
                return rec
    return None


_LAZY_COMPAT_EXPORTS: dict[str, str] = {
    "cancel_existing_ai_open_orders": "providers.intent_execution",
    "execute_enforce_intent_order": "providers.intent_execution",
    "find_duplicate_ai_open_limit_order": "providers.intent_execution",
    "preflight_intent_and_resolve_price": "providers.intent_execution",
    "reconcile_open_intent_by_client_id": "providers.intent_execution",
    "record_shadow_intent_observation": "providers.intent_execution",
    "resolve_intent_execution_price": "providers.intent_execution",
    "resolve_intent_topic_tag": "providers.intent_execution",
    "_attach_intent_age_fields": "providers.intent_reconciliation",
    "_filter_intents_by_scope": "providers.intent_reconciliation",
    "_is_open_resting_detail": "providers.intent_reconciliation",
    "_resolve_executed_open_resting_detail": "providers.intent_reconciliation",
    "_resolve_intent_execution_details": "providers.intent_reconciliation",
    "build_last_cycle_execution_status": "providers.intent_reconciliation",
    "reconcile_intents_from_execution_audit": "providers.intent_reconciliation",
    "summarize_recent_ai_intents": "providers.intent_reconciliation",
}

__all__ = [
    "AuditedStrategyExecutor",
    "ExecutionAudit",
    "find_order_audit_record",
    "intent_client_order_id",
    "new_intent_id",
    "_extract_intent_id",
    "_intent_order_ids",
    "_intent_venue_symbol_side_key",
    "_record_intent_state",
    "collect_known_ai_order_ids",
    "durable_intents_lock",
    "extract_intent_limit_price",
    "extract_intent_notional",
    "extract_intent_order_type",
    "extract_intent_side",
    "intent_venue_client_order_ids",
    "load_durable_intents_state",
    "normalize_trading_decisions",
    "save_durable_intents_state",
    "upsert_durable_intent",
    "validate_and_build_intent_dict",
    *_LAZY_COMPAT_EXPORTS,
]

bind_lazy_module_exports(globals(), _LAZY_COMPAT_EXPORTS)
