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
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

from botcore import (
    _parse_bool_flag,
    clean_order_id,
    clean_order_id_str,
    exact_pos_int,
    float_env,
    lru_dict_put,
    normalize_side,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
    sanitize_identifier,
)
from cachepaths import (
    _is_eq_suffix,
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
    normalize_venue_name,
    split_symbol_assets,
)
from instrument_registry import casefold_get
from market_regime import ClosedPriceSeries
from state_io import append_jsonl_record, read_jsonl_dicts
from .base import MarketDataProvider
from .intent_execution import (
    _fmt_intent_px,
    _fmt_intent_qty,
    cancel_existing_ai_open_orders,
    execute_enforce_intent_order,
    find_duplicate_ai_open_limit_order,
    preflight_intent_and_resolve_price,
    record_shadow_intent_observation,
    resolve_intent_execution_price,
    resolve_intent_topic_tag,
)
from .intent_state import (
    _CURRENCY_STRIP_RE,
    _INTENTS_LOCK_LOCAL,
    _NOTIONAL_KEYS,
    _extract_intent_id,
    _intent_order_ids,
    _intent_venue_symbol_side_key,
    _parse_currency_pos_float,
    _record_intent_state,
    _resolve_clamped_intent_notional,
    ai_intent_client_order_id,
    append_intent_history,
    collect_known_ai_order_ids,
    durable_intents_lock,
    extract_intent_limit_price,
    extract_intent_notional,
    extract_intent_order_type,
    extract_intent_side,
    generate_unique_intent_id,
    intent_venue_client_order_ids,
    load_durable_intents_state,
    normalize_intent_fields,
    normalize_trading_decisions,
    parse_intent_confidence,
    parse_intent_gateway_config,
    save_durable_intents_state,
    upsert_durable_intent,
    validate_and_build_intent_dict,
    venue_native_client_order_ids,
)
from .order_ohlc_utils import (
    _UNFILLED_ORDER_STATUSES,
    _parse_epoch_or_iso_s,
    _ts_ms_to_s,
    extract_client_order_id,
    extract_executed_qty,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
    normalize_timestamp_ms,
    resolve_order_actor,
)
from .strategy_executor import (
    OrderStatus, ProviderError, SubmissionOutcome, SubmissionRefused,
    capture_submission, reconciliation_capabilities_of,
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
    normalized_venue = normalize_venue_name(v_slug) or v_slug
    if normalized_venue == "kraken":
        return token                         # cl_ord_id: UUID without hyphens
    if normalized_venue == "binance":
        return f"SD_{token}"                 # newClientOrderId: 35 <= 36 characters
    if normalized_venue == "hyperliquid":
        return f"0x{token}"                  # cloid: uint128 hexadecimal
    return None                              # T212/unknown venue: local correlation


class ExecutionAudit:
    """Best-effort JSONL writer shared by Kraken, T212, HL, and Binance."""

    def __init__(self, directory: Optional[str] = None,
                 clock: Callable[[], float] = time.time):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.directory = directory or os.environ.get(
            "EXECUTION_AUDIT_DIR", os.path.join(root, "logger", "execution_audit")
        )
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

    @staticmethod
    def _lru_put(cache: OrderedDict, key: Any, value: Any) -> None:
        lru_dict_put(cache, key, value, _CACHE_MAX)

    def _remember(self, symbol: str, order_id: str, intent_id: str) -> None:
        oid_key = clean_order_id_str(order_id)
        if not oid_key:
            return
        sym_key = self._norm_symbol_key(symbol)
        with self._lock:
            self._lru_put(self._intent_by_order, (sym_key, oid_key), intent_id)

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
                                 cache_permit=None) -> str:
        outcome = self.submit_outcome_with_intent(
            intent_id, symbol, side, qty, price, market=market, kind=kind,
            reference_price=reference_price, client_order_id=client_order_id,
            cache_permit=cache_permit,
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
                     cache_permit=None) -> str:
        return self.submit_order_with_intent(
            new_intent_id(self.name, symbol, kind), symbol, side, qty, price,
            market=market, kind=kind, client_order_id=client_order_id,
            cache_permit=cache_permit,
        )

    def order_status_with_intent(self, intent_id: str, symbol: str,
                                 order_id: str) -> OrderStatus:
        intent_id = self._intent(symbol, order_id, intent_id)
        try:
            status = self._executor.order_status(symbol, order_id)
        except Exception as exc:
            self.audit.record(
                "status_error", intent_id=intent_id, venue=self.name, symbol=symbol,
                order_id=str(order_id), error_type=exc.__class__.__name__, error=str(exc),
            )
            raise
        fingerprint = (
            status.status, status.filled_qty, status.cost, status.fee,
            status.venue_status,
        )
        oid_key = clean_order_id_str(order_id)
        key = (self._norm_symbol_key(symbol), oid_key)
        with self._lock:
            changed = self._last_status.get(key) != fingerprint
            self._lru_put(self._last_status, key, fingerprint)
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
        try:
            self._executor.cancel_order(symbol, order_id)
        except Exception as exc:
            self.audit.record(
                "cancel_rejected", intent_id=intent_id, venue=self.name, symbol=symbol,
                order_id=str(order_id), error_type=exc.__class__.__name__, error=str(exc),
            )
            raise
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
        preflight = getattr(self._executor, "preflight_order", None)
        return preflight(symbol, side, qty, price, market=_parse_bool_flag(market, False), kind=kind) if callable(preflight) else None

    def ohlc_closes(self, symbol: str, interval_min: int):
        return self._executor.ohlc_closes(symbol, interval_min)

    @property
    def _inner(self):
        return self._executor

    def _call_opt(self, attr: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        reader = getattr(self._executor, attr, None)
        return reader(*args, **kwargs) if callable(reader) else default

    def _call_trend_opt(self, attr: str, symbol: str, max_age_sec: float) -> tuple[bool, Any]:
        reader = getattr(self._executor, attr, None)
        if not callable(reader):
            return False, None
        try:
            return True, reader(symbol, max_age_sec=max_age_sec)
        except TypeError:
            return True, reader(symbol)

    def open_orders(self, symbol: str):
        return self._call_opt("open_orders", symbol, default=[])

    def get_orders(self, symbol: str, side: Optional[str] = None, lookback: Optional[float] = None):
        return self._call_opt("get_orders", symbol, side, lookback, default=[])

    def trend_snapshot(self, symbol: str, max_age_sec: float = 15.0):
        return self._call_trend_opt("trend_snapshot", symbol, max_age_sec)[1]

    def trend(self, symbol: str, max_age_sec: float = 15.0) -> float:
        ok, raw = self._call_trend_opt("trend", symbol, max_age_sec)
        return safe_float(raw, default=0.0) if ok else 0.0

    def trend_direction(self, symbol: str, max_age_sec: float = 15.0) -> Optional[str]:
        ok, raw = self._call_trend_opt("trend_direction", symbol, max_age_sec)
        return raw if ok else MarketDataProvider.trend_direction(self, symbol, max_age_sec=max_age_sec)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        return self._call_opt("position_cost_basis", symbol, held_qty, since_s)

    def get_ssot_cached_price(self, symbol: str, max_age_sec: float = 12.0) -> Optional[float]:
        return self._call_opt("get_ssot_cached_price", symbol, max_age_sec=max_age_sec)

    def get_ssot_price_history(self, symbol: str, lookback_hours: float, max_latest_age_sec: float = 120.0) -> Optional[list[dict]]:
        return self._call_opt("get_ssot_price_history", symbol, lookback_hours, max_latest_age_sec=max_latest_age_sec)

    def _pair_precision_attr(self, symbol: str, attr: str, default: Any = None) -> Any:
        return MarketDataProvider._pair_precision_attr(self, symbol, attr, default)

    def round_quantity(self, symbol: str, qty: float) -> float:
        fn = getattr(self._executor, "round_quantity", None)
        return fn(symbol, qty) if callable(fn) else MarketDataProvider.round_quantity(self, symbol, qty)

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float) -> float:
        fn = getattr(self._executor, "round_price", None)
        return fn(symbol, price) if callable(fn) else MarketDataProvider.round_price(self, symbol, price)

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
        interval_i = exact_pos_int(interval_min, default=None)
        if interval_i is None:
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
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
    v_stat = str(rec.get("venue_status") or "").upper()
    l_stat = str(rec.get("status") or "").lower()
    filled_qty = rec.get("filled_qty") if rec.get("filled_qty") is not None else rec.get("qty")
    f_q = safe_pos_float(filled_qty, default=None)
    raw_cost = rec.get("cost")
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
        v_stat in ("FILLED", "CLOSED", "PARTIALLY_FILLED")
        or l_stat in ("closed", "filled", "partially_filled")
        or (f_q is not None and f_q > 0.0)
    )
    return v_stat, l_stat, filled_qty, f_q, raw_cost, cost_f, f_px, is_fill


_AUDIT_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_AUDIT_RECONCILE_EVENTS = frozenset({"submit_accepted", "order_terminal", "order_terminal_retry"})
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
    eff_dir = (
        audit_dir
        or os.environ.get("EXECUTION_AUDIT_DIR")
        or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logger", "execution_audit")
    )
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


def reconcile_intents_from_execution_audit(
    intents: list[dict[str, Any]],
    audit_dir: str,
) -> bool:
    """Reconcile FAILED/PENDING_EXECUTION intents and enrich incomplete EXECUTED intents from execution_audit_*.jsonl."""
    canceled_prior_oids: dict[str, float] = {}
    for item in intents:
        if isinstance(item, dict) and isinstance(item.get("canceled_prior_open_orders"), list):
            c_ts = safe_pos_float(item.get("timestamp"), default=0.0) or 0.0
            for raw_oid in item["canceled_prior_open_orders"]:
                if oid_s := clean_order_id(raw_oid):
                    canceled_prior_oids[oid_s] = max(canceled_prior_oids.get(oid_s, 0.0), c_ts)
    updated = False
    failed_by_id: dict[str, dict[str, Any]] = {}
    executed_incomplete_by_id: dict[str, dict[str, Any]] = {}
    for item in intents:
        if not isinstance(item, dict):
            continue
        clean_id = _extract_intent_id(item)
        if not clean_id:
            continue
        st_up = str(item.get("status") or "").strip().upper()
        if st_up in ("FAILED", "PENDING_EXECUTION"):
            failed_by_id[clean_id] = item
        elif st_up == "EXECUTED":
            exec_pay = item.get("execution_payload") if isinstance(item.get("execution_payload"), dict) else {}
            needs_ts_backfill = (
                bool(exec_pay.get("reconciled_terminal"))
                and "filled_at" not in exec_pay
                and "terminal_ts" not in exec_pay
                and not exec_pay.get("_ts_backfill_checked")
            )
            if not exec_pay.get("reconciled_terminal") or needs_ts_backfill:
                cur_v_st = str(exec_pay.get("status") or "").strip().upper()
                item_oid = clean_order_id(extract_order_id(exec_pay, default=None))
                has_fill_metrics = (
                    extract_executed_qty(exec_pay, default=0.0) > 0.0
                    and extract_order_price(exec_pay, default=0.0) > 0.0
                )
                is_resting_msg = str(exec_pay.get("msg") or "").strip().lower() == "resting"
                if (
                    item_oid
                    and item_oid in canceled_prior_oids
                    and (
                        (
                            not needs_ts_backfill
                            and cur_v_st in _OPEN_WORKING_STATUSES
                            and extract_executed_qty(exec_pay, default=0.0) <= 0.0
                        )
                        or (needs_ts_backfill and cur_v_st in ("CANCELED", "CANCELLED"))
                    )
                ):
                    exec_pay = dict(exec_pay)
                    exec_pay["status"] = "CANCELED"
                    if is_resting_msg:
                        exec_pay.pop("msg", None)
                    c_ts_val = canceled_prior_oids[item_oid] or safe_pos_float(item.get("timestamp"), default=None)
                    if c_ts_val is not None:
                        exec_pay["terminal_ts"] = c_ts_val
                    exec_pay["reconciled_terminal"] = True
                    item["execution_payload"] = exec_pay
                    updated = True
                    continue
                if not needs_ts_backfill and cur_v_st in _TERMINAL_CANCEL_STATUSES:
                    continue
                if not needs_ts_backfill and has_fill_metrics and not is_resting_msg and cur_v_st in ("", "FILLED", "CLOSED"):
                    continue
                executed_incomplete_by_id[clean_id] = item
    if (not failed_by_id and not executed_incomplete_by_id) or not os.path.isdir(audit_dir):
        return updated

    # Audit files are UTC-day partitioned and never precede the intent that produced them: scan only days
    # from the oldest pending intent (minus one day of clock-skew margin), via the per-file record cache.
    pending_ts = [
        ts_i
        for it in (*failed_by_id.values(), *executed_incomplete_by_id.values())
        if (ts_i := safe_pos_float(it.get("timestamp"), default=0.0) or 0.0) > 0.0
    ]
    min_day = (
        datetime.fromtimestamp(min(pending_ts) - 86400.0, timezone.utc).strftime("%Y-%m-%d")
        if pending_ts
        else ""
    )
    matched_records: dict[str, dict[str, Any]] = {}
    matched_exec_records: dict[str, dict[str, Any]] = {}
    for audit_path in sorted(glob.glob(os.path.join(glob.escape(audit_dir), "execution_audit_*.jsonl"))):
        file_day = os.path.basename(audit_path)[len("execution_audit_"):-len(".jsonl")]
        if min_day and _AUDIT_DAY_RE.fullmatch(file_day) and file_day < min_day:
            continue
        for rec in _read_audit_intent_records(audit_path):
            iid = _extract_intent_id(rec)
            if not iid:
                continue
            ev = str(rec.get("event") or "")
            if iid in failed_by_id:
                if ev == "submit_accepted":
                    matched_records.setdefault(iid, rec)
                elif ev in ("order_terminal", "order_terminal_retry"):
                    *_, is_fill = _extract_audit_fill_metrics(rec)
                    if is_fill:
                        matched_records[iid] = rec
                    elif matched_records.get(iid, {}).get("event") == "submit_accepted":
                        prev_oid = matched_records[iid].get("order_id")
                        cur_oid = rec.get("order_id")
                        if not prev_oid or not cur_oid or str(prev_oid) == str(cur_oid):
                            matched_records.pop(iid, None)
            elif iid in executed_incomplete_by_id and ev in ("order_terminal", "order_terminal_retry"):
                intent_obj = executed_incomplete_by_id[iid]
                intent_oid = extract_order_id(intent_obj.get("execution_payload"), default=None)
                rec_oid = extract_order_id(rec.get("order_id"), default=None)
                if intent_oid is not None and rec_oid is not None and str(intent_oid) != str(rec_oid):
                    continue
                v_stat, l_stat, *_, is_fill = _extract_audit_fill_metrics(rec)
                if is_fill:
                    matched_exec_records[iid] = rec
                elif (
                    iid not in matched_exec_records
                    or not _extract_audit_fill_metrics(matched_exec_records[iid])[-1]
                ) and (
                    v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")
                    or l_stat in ("canceled", "cancelled", "expired", "rejected")
                ):
                    matched_exec_records[iid] = rec

    for iid_chk, item_chk in executed_incomplete_by_id.items():
        if iid_chk not in matched_exec_records:
            ep_chk = item_chk.get("execution_payload")
            if isinstance(ep_chk, dict) and ep_chk.get("reconciled_terminal"):
                ep_chk["_ts_backfill_checked"] = True

    if not matched_records and not matched_exec_records:
        return updated

    for iid, rec in matched_records.items():
        intent = failed_by_id[iid]
        v_stat, _, filled_qty, _, raw_cost, _, f_px, _ = _extract_audit_fill_metrics(
            rec, fallback_ref_price=True
        )
        default_v_status = (
            "NEW"
            if (
                str(rec.get("event") or "") == "submit_accepted"
                and extract_intent_order_type(intent) == "LIMIT"
                and (safe_pos_float(rec.get("filled_qty"), default=0.0) or 0.0) <= 0.0
            )
            else "FILLED"
        )
        raw_v_status = v_stat or default_v_status
        if raw_v_status == "CLOSED":
            raw_v_status = "FILLED"
        exec_payload: dict[str, Any] = {
            "orderId": rec.get("order_id"),
            "symbol": rec.get("symbol") or intent.get("symbol"),
            "side": str(rec.get("side") or intent.get("decision") or "").upper(),
            "executedQty": str(filled_qty) if filled_qty is not None else "",
            "cummulativeQuoteQty": str(raw_cost) if raw_cost is not None else "",
            "status": raw_v_status,
            "reconciled_from_audit": True,
        }
        if f_px is not None:
            exec_payload["price"] = f_px
        rec_ts_f = safe_pos_float(rec.get("ts"), default=None)
        if rec_ts_f is not None and raw_v_status == "FILLED":
            exec_payload["filled_at"] = rec_ts_f
            exec_payload["terminal_ts"] = rec_ts_f
        intent["execution_payload"] = exec_payload
        _record_intent_state(
            intent,
            "EXECUTED",
            rec_ts_f if rec_ts_f is not None else time.time(),
            action="RECONCILED_EXECUTED",
            order_id=rec.get("order_id"),
            payload=exec_payload,
        )
        updated = True

    for iid, rec in matched_exec_records.items():
        intent = executed_incomplete_by_id[iid]
        exec_pay = dict(intent.get("execution_payload")) if isinstance(intent.get("execution_payload"), dict) else {}
        v_stat, l_stat, filled_qty, f_q, raw_cost, cost_f, f_px, is_fill = _extract_audit_fill_metrics(rec)
        rec_ts = safe_pos_float(rec.get("ts"), default=None)
        if is_fill:
            norm_st = "PARTIALLY_FILLED" if (v_stat == "PARTIALLY_FILLED" or l_stat == "partially_filled") else "FILLED"
            if rec.get("order_id") is not None and not extract_order_id(exec_pay, default=None):
                exec_pay["orderId"] = rec.get("order_id")
            exec_pay["status"] = norm_st
            if f_q is not None and f_q > 0:
                exec_pay["executedQty"] = str(filled_qty)
                if "filledQuantity" in exec_pay:
                    exec_pay["filledQuantity"] = f_q
            if raw_cost is not None and cost_f is not None and cost_f > 0:
                exec_pay["cummulativeQuoteQty"] = str(raw_cost)
                exec_pay["cost"] = raw_cost
            if f_px is not None and f_px > 0:
                exec_pay["price"] = f_px
            if rec_ts is not None:
                exec_pay["filled_at"] = rec_ts
                exec_pay["terminal_ts"] = rec_ts
            if str(exec_pay.get("msg") or "").strip().lower() == "resting":
                exec_pay.pop("msg", None)
            if norm_st != "PARTIALLY_FILLED":
                exec_pay["reconciled_terminal"] = True
            if exec_pay != intent.get("execution_payload"):
                intent["execution_payload"] = exec_pay
                updated = True
        else:
            cur_st = str(exec_pay.get("status") or "").strip().upper()
            if cur_st in _OPEN_WORKING_STATUSES or (
                exec_pay.get("reconciled_terminal") and "terminal_ts" not in exec_pay and rec_ts is not None
            ):
                norm_cancel = v_stat if v_stat in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED") else "CANCELED"
                exec_pay["status"] = norm_cancel
                if rec_ts is not None:
                    exec_pay["terminal_ts"] = rec_ts
                if str(exec_pay.get("msg") or "").strip().lower() == "resting":
                    exec_pay.pop("msg", None)
                exec_pay["reconciled_terminal"] = True
                if exec_pay != intent.get("execution_payload"):
                    intent["execution_payload"] = exec_pay
                    updated = True

    return updated


def _filter_intents_by_scope(intents_data: Any, scope: str) -> list[dict[str, Any]]:
    """Filter intent records to those matching the requested asset scope ('crypto', 'equities', or 'all')."""
    if isinstance(intents_data, dict):
        intents_data = intents_data.get("intents")
    if not isinstance(intents_data, list):
        return []
    scope_norm = str(scope or "all").strip().lower()
    out: list[dict[str, Any]] = []
    for i in intents_data:
        if not isinstance(i, dict):
            continue
        if scope_norm in ("equities", "crypto"):
            v_norm = normalize_venue_name(i.get("venue"))
            is_eq = v_norm == "t212" if v_norm else _is_eq_suffix(i.get("symbol"))
            if is_eq != (scope_norm == "equities"):
                continue
        out.append(i)
    return out


def _resolve_intent_execution_details(
    i: dict[str, Any],
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Resolve fill price, filled quantity, execution status, and slippage (bps) for an intent."""
    venue = normalize_venue_name(i.get("venue")) or i.get("venue")
    symbol = str(i.get("symbol") or "").strip().upper()
    side = extract_intent_side(i) or str(i.get("decision") or "").strip().upper()
    order_type = extract_intent_order_type(i)
    exec_pay = i.get("execution_payload") if isinstance(i.get("execution_payload"), dict) else {}
    hist = i.get("history") if isinstance(i.get("history"), list) else []
    last_hist = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    exec_quote_notional = (
        safe_pos_float(exec_pay.get("cummulativeQuoteQty"), default=None)
        or safe_pos_float(exec_pay.get("cost"), default=None)
        or safe_pos_float(exec_pay.get("filledValue"), default=None)
    )
    intent_order_notional = (
        safe_pos_float(i.get("clamped_notional_usd"), default=None)
        or safe_pos_float(last_hist.get("notional_usd"), default=None)
        or extract_notional_fn(i, default=None)
    )
    eff_notional = exec_quote_notional or intent_order_notional
    ts_raw: Any = None
    ts_val = 0.0
    for _tk in ("timestamp", "executed_at", "executed_ts", "created_at", "ts"):
        _cand_raw = i.get(_tk)
        if _cand_raw is None:
            continue
        if ts_raw is None:
            ts_raw = _cand_raw
        _parsed_s = _parse_epoch_or_iso_s(_cand_raw)
        if _parsed_s is not None and _parsed_s > 0.0:
            ts_raw = _cand_raw
            ts_val = _ts_ms_to_s(_parsed_s)
            break
    order_id = extract_order_id(exec_pay, default=None) or extract_order_id(
        last_hist.get("order_id") or last_hist.get("payload"), default=None
    )

    raw_intent_st = str(i.get("status") or "").strip().upper()
    is_eq_intent = (venue == "t212") or _is_eq_suffix(symbol)
    matched_open = None
    expected_cids: Optional[set[str]] = None
    if isinstance(open_orders, list) and raw_intent_st in ("EXECUTED", "FAILED", "PENDING_EXECUTION"):
        for oo in open_orders:
            if not isinstance(oo, dict):
                continue
            oo_venue = normalize_venue_name(oo.get("venue"))
            if oo_venue and venue and oo_venue != venue:
                continue
            oo_sym = str(oo.get("symbol") or "").strip().upper()
            oo_is_eq = (oo_venue == "t212") or _is_eq_suffix(oo_sym)
            if (oo_venue or oo_sym) and (venue or symbol) and oo_is_eq != is_eq_intent:
                continue
            if oo_sym and symbol and (
                canonical_base_asset(oo_sym) != canonical_base_asset(symbol)
                or compatible_quote_suffixes(oo_sym) != compatible_quote_suffixes(symbol)
            ):
                continue
            oo_oid = extract_order_id(oo, default=None)
            if order_id is not None and oo_oid is not None and str(oo_oid) == str(order_id):
                matched_open = oo
                break
            if (
                order_id is None
                and order_type == "LIMIT"
                and oo_venue == venue
                and oo_sym
                and (extract_order_side(oo) or str(oo.get("side") or "").strip().upper()) == side
            ):
                # Response-lost intent: adopt only the order carrying this intent's own client ID
                # (venue-native form), never a BOT / monitororder-repriced order or another AI intent's order.
                if expected_cids is None:
                    expected_cids = intent_venue_client_order_ids(i)
                oo_cid = (extract_client_order_id(oo, default="") or "").lower()
                if (oo_cid in expected_cids) if oo_cid else (resolve_order_actor(oo) == "AI"):
                    matched_open = oo
                    break

    fill_px = extract_order_price(exec_pay, default=None)
    filled_qty = extract_executed_qty(exec_pay, default=0.0)
    orig_qty = extract_orig_qty(exec_pay, default=None) or safe_pos_float(last_hist.get("qty"), default=None)
    if matched_open is not None:
        open_filled = extract_executed_qty(matched_open, default=0.0)
        if open_filled > filled_qty:
            filled_qty = open_filled
        if fill_px is None:
            fill_px = safe_pos_float(matched_open.get("price"), default=None)

    venue_st = str(
        exec_pay.get("status")
        or ("RESTING" if str(exec_pay.get("msg") or "").strip().lower() == "resting" else "")
        or (
            "OPEN"
            if (
                order_type == "LIMIT"
                and filled_qty <= 0
                and isinstance(exec_pay, dict)
                and any(k in exec_pay for k in ("txid", "ordertxid", "descr", "orderId", "order_id", "oid"))
            )
            else ""
        )
    ).strip().upper()
    if matched_open is not None and not venue_st:
        venue_st = str(matched_open.get("status") or "OPEN").strip().upper()
    if (
        matched_open is None
        and raw_intent_st == "EXECUTED"
        and venue_st not in _TERMINAL_CANCEL_STATUSES
        and isinstance(recent_orders, dict)
        and symbol
    ):
        raw_cands = cache_symbol_variants(symbol, quotes=compatible_quote_suffixes(symbol))
        sym_cands = (
            [c for c in raw_cands if _is_eq_suffix(c) or split_symbol_assets(c) is None]
            if is_eq_intent
            else [c for c in raw_cands if not _is_eq_suffix(c)]
        )
        ro_entry = next(
            (
                val
                for key_cand in (*(f"{venue}:{cand}" for cand in sym_cands), *sym_cands)
                if isinstance((val := casefold_get(recent_orders, key_cand)), dict)
                and (
                    ":" in key_cand
                    or not venue
                    or not val.get("venue")
                    or normalize_venue_name(val.get("venue")) == venue
                )
            ),
            None,
        )
        if isinstance(ro_entry, dict):
            side_key = "buy" if side == "BUY" else ("sell" if side == "SELL" else "")
            # Scan newest-first across last_<side> and recent_<side>s: a later BOT fill on the same side
            # (e.g. a monitororder-repriced tradeall order) must not hide this intent's own AI fill.
            fill_cands: list[dict[str, Any]] = []
            if side_key:
                last_c = ro_entry.get(f"last_{side_key}")
                recent_c = ro_entry.get(f"recent_{side_key}s")
                for cand_f in ([last_c] if isinstance(last_c, dict) else []) + (
                    list(reversed(recent_c)) if isinstance(recent_c, list) else []
                ):
                    if isinstance(cand_f, dict) and cand_f not in fill_cands:
                        fill_cands.append(cand_f)
            min_fill_ts_ms = ts_val * 1000.0 if order_type == "LIMIT" else (ts_val - 120.0) * 1000.0
            last_fill = None
            for cand_f in fill_cands:
                f_oid = clean_order_id(cand_f.get("order_id"))
                if order_id is not None and f_oid:
                    if f_oid == str(order_id):
                        last_fill = cand_f
                        break
                    continue
                f_ts_ms = normalize_timestamp_ms(cand_f.get("ts_ms"), default=0) or 0
                f_actor = str(cand_f.get("actor") or "").strip().upper()
                if f_actor != "BOT" and (ts_val <= 0 or f_ts_ms <= 0 or f_ts_ms >= min_fill_ts_ms):
                    last_fill = cand_f
                    break
            if isinstance(last_fill, dict):
                if fill_px is None or (venue_st in _UNFILLED_ORDER_STATUSES and filled_qty <= 0):
                    fill_px = safe_pos_float(last_fill.get("price"), default=fill_px)
                if filled_qty <= 0:
                    filled_qty = safe_pos_float(last_fill.get("qty"), default=0.0) or 0.0
                matched_fill_ts_ms = normalize_timestamp_ms(last_fill.get("ts_ms"), default=0) or 0
            else:
                matched_fill_ts_ms = 0
        else:
            matched_fill_ts_ms = 0
    else:
        matched_fill_ts_ms = 0
    is_unfilled_resting_limit = (
        order_type == "LIMIT"
        and (venue_st in _UNFILLED_ORDER_STATUSES or not venue_st)
        and filled_qty <= 0
    )
    venue_open_verified = (
        verified_open_venues is None
        or (bool(venue) and normalize_venue_name(venue) in verified_open_venues)
    )
    if (
        is_unfilled_resting_limit
        and matched_open is None
        and isinstance(open_orders, list)
        and venue_open_verified
        and ts_val > 0
    ):
        from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals

        cancel_grace_sec = resolve_derived_telemetry_intervals()["unfilled_order_cancel_grace_sec"]
        if (now_ts - ts_val) > cancel_grace_sec:
            venue_st = "CANCELED_OR_EXPIRED"
    if fill_px is None and raw_intent_st in ("EXECUTED", "OBSERVED_SHADOW") and not is_unfilled_resting_limit:
        fill_px = safe_pos_float(
            last_hist.get("price")
            or i.get("requested_price")
            or i.get("limit_price")
            or i.get("reference_price"),
            default=None,
        )

    if matched_open is not None:
        exec_status = "PARTIALLY_FILLED" if filled_qty > 0 else "OPEN_RESTING"
        if exec_status == "PARTIALLY_FILLED" and venue_st in _UNFILLED_ORDER_STATUSES:
            venue_st = "PARTIALLY_FILLED"
    elif raw_intent_st == "EXECUTED":
        if venue_st in _TERMINAL_CANCEL_STATUSES:
            exec_status = venue_st
        elif is_unfilled_resting_limit:
            exec_status = "OPEN_RESTING"
        elif venue_st == "PARTIALLY_FILLED":
            exec_status = "PARTIALLY_FILLED"
        else:
            exec_status = "FILLED"
            if venue_st in _UNFILLED_ORDER_STATUSES:
                venue_st = "FILLED"
    elif raw_intent_st == "OBSERVED_SHADOW":
        exec_status = "SHADOW_SIMULATED"
    else:
        exec_status = raw_intent_st or "UNKNOWN"

    ref_px = safe_pos_float(
        i.get("reference_price")
        or i.get("limit_price")
        or i.get("requested_price")
        or last_hist.get("price")
        or fill_px,
        default=None,
    )
    if exec_status in ("OPEN_RESTING", *_TERMINAL_CANCEL_STATUSES) and filled_qty <= 0:
        fill_px = None
    slippage_bps: Optional[float] = None
    if (
        exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED")
        and fill_px is not None
        and fill_px > 0
        and ref_px is not None
        and ref_px > 0
    ):
        raw_slip = (
            ((fill_px - ref_px) / ref_px) * 10_000.0
            if side == "BUY"
            else ((ref_px - fill_px) / ref_px) * 10_000.0
        )
        slippage_bps = round(raw_slip, 2)
    elif exec_status in ("FILLED", "SHADOW_SIMULATED") and fill_px is not None:
        slippage_bps = 0.0

    eff_qty = (
        filled_qty
        if filled_qty > 0
        else (orig_qty if (orig_qty is not None and exec_status in ("FILLED", "SHADOW_SIMULATED")) else None)
    )
    limit_px = (
        safe_pos_float(
            i.get("limit_price")
            or (matched_open.get("price") if isinstance(matched_open, dict) else None),
            default=None,
        )
        if order_type == "LIMIT"
        else None
    )
    fill_ts_val = ts_val
    is_fill_exec = exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED") or (
        eff_qty is not None and eff_qty > 0
    )
    if is_fill_exec and isinstance(exec_pay, dict):
        for _ftk in ("filled_at", "filled_ts", "terminal_ts", "updated_ts", "updateTime", "workingTime", "transactTime"):
            _fcand = exec_pay.get(_ftk)
            if _fcand is None:
                continue
            _fparsed = _parse_epoch_or_iso_s(_fcand)
            if _fparsed is not None and _fparsed > 0.0:
                _f_s = _ts_ms_to_s(_fparsed)
                if _f_s >= ts_val:
                    fill_ts_val = max(fill_ts_val, _f_s)
                    break
    if is_fill_exec and matched_fill_ts_ms > 0:
        _mf_s = _ts_ms_to_s(float(matched_fill_ts_ms))
        if _mf_s > fill_ts_val:
            fill_ts_val = _mf_s
    if is_fill_exec and fill_ts_val <= ts_val and str(last_hist.get("action") or "") == "RECONCILED_EXECUTED":
        _lh_parsed = _parse_epoch_or_iso_s(last_hist.get("ts"))
        if _lh_parsed is not None and _lh_parsed > 0.0:
            _lh_s = _ts_ms_to_s(_lh_parsed)
            if _lh_s > fill_ts_val:
                fill_ts_val = _lh_s
    fail_reason: Optional[str] = None
    if exec_status not in ("FILLED", "PARTIALLY_FILLED", "OPEN_RESTING", "SHADOW_SIMULATED"):
        raw_fail_reason = (
            last_hist.get("reason")
            or last_hist.get("guard_reason")
            or last_hist.get("error")
            or i.get("guard_reason")
            or i.get("error")
            or (exec_pay.get("reason") or exec_pay.get("error") if isinstance(exec_pay, dict) else None)
        )
        if raw_fail_reason:
            fail_reason = " ".join(str(raw_fail_reason).split())[:200] or None
    filled_notional_usd: Optional[float] = None
    if exec_status == "PARTIALLY_FILLED":
        raw_filled_notional = exec_quote_notional or (
            (eff_qty * fill_px)
            if (eff_qty is not None and eff_qty > 0 and fill_px is not None and fill_px > 0)
            else None
        )
        if raw_filled_notional is not None and raw_filled_notional > 0:
            filled_notional_usd = round(raw_filled_notional, 2)
        orig_order_px = limit_px or fill_px or ref_px
        eff_notional = (
            intent_order_notional
            or ((orig_qty * orig_order_px) if (orig_qty is not None and orig_order_px is not None) else None)
            or exec_quote_notional
        )
    return {
        "venue": venue,
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "raw_status": i.get("status"),
        "venue_status": venue_st or None,
        "exec_status": exec_status,
        "is_open_in_venue": matched_open is not None,
        "order_id": str(order_id) if order_id is not None else None,
        "fill_px": round(fill_px, 6) if fill_px is not None else None,
        "limit_px": round(limit_px, 6) if limit_px is not None else None,
        "ref_px": round(ref_px, 6) if ref_px is not None else None,
        "filled_qty": round(eff_qty, 8) if eff_qty is not None and eff_qty > 0 else None,
        "filled_notional_usd": filled_notional_usd,
        "notional_usd": round(eff_notional, 2) if eff_notional is not None else i.get("suggested_notional_usd"),
        "slippage_bps": slippage_bps,
        "reason": fail_reason,
        "ts": ts_raw,
        "ts_val": ts_val,
        "fill_ts_val": fill_ts_val,
    }


def _is_open_resting_detail(det: dict[str, Any]) -> bool:
    return bool(det.get("is_open_in_venue") and det.get("exec_status") in ("OPEN_RESTING", "PARTIALLY_FILLED"))


def summarize_recent_ai_intents(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Summarize the most recent AI trade intents filtered by asset scope."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)

    def _to_summary(i: dict[str, Any], det: dict[str, Any]) -> dict[str, Any]:
        intent_summary: dict[str, Any] = {
            "decision": i.get("decision") or det["side"],
            "symbol": i.get("symbol"),
            "venue": det["venue"],
            "order_type": det["order_type"],
            "status": det["raw_status"],
            "notional_usd": det["notional_usd"],
            "ts": det["ts"],
        }
        if i.get("limit_price") is not None:
            intent_summary["limit_price"] = i.get("limit_price")
        if det["venue_status"]:
            intent_summary["venue_status"] = det["venue_status"]
        for k in ("order_id", "fill_px", "filled_qty", "filled_notional_usd", "slippage_bps"):
            if det.get(k) is not None:
                intent_summary[k] = det[k]
        if det.get("reason"):
            intent_summary["reason"] = det["reason"]
        eff_ts_val = safe_pos_float(det.get("fill_ts_val"), default=None) or det["ts_val"]
        if eff_ts_val > 0 and now_ts >= eff_ts_val:
            intent_summary["age_h"] = round((now_ts - eff_ts_val) / 3600.0, 2)
        if det["ts_val"] > 0 and eff_ts_val > det["ts_val"] + 60.0 and now_ts >= det["ts_val"]:
            intent_summary["placed_age_h"] = round((now_ts - det["ts_val"]) / 3600.0, 2)
        return intent_summary

    recent_rev: list[dict[str, Any]] = []
    non_resting_count = 0
    seen_oids: set[str] = set()
    seen_sym_side: set[tuple[str, str, tuple[str, ...], str]] = set()
    recent_indices: set[int] = set()
    for idx in range(len(scoped_intents) - 1, -1, -1):
        if non_resting_count >= 8 or len(recent_rev) >= 16:
            break
        i = scoped_intents[idx]
        det = _resolve_intent_execution_details(
            i,
            now_ts,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        sym_side_key = _intent_venue_symbol_side_key(det.get("venue"), det.get("symbol"), det.get("side"))
        if (
            det.get("exec_status")
            in ("CANCELED", "CANCELLED", "EXPIRED", "CANCELED_OR_EXPIRED", "SKIPPED_DUPLICATE_OPEN_ORDER")
            and not det.get("filled_qty")
            and sym_side_key in seen_sym_side
        ):
            continue
        seen_sym_side.add(sym_side_key)
        recent_indices.add(idx)
        if det["order_id"] is not None:
            seen_oids.add(str(det["order_id"]))
        sum_item = _to_summary(i, det)
        recent_rev.append(sum_item)
        if not (
            isinstance(open_orders, list)
            and open_orders
            and _is_open_resting_detail(det)
            and (sum_item.get("age_h") or 0.0) > 0.5
        ):
            non_resting_count += 1
    recent_intents_out: list[dict[str, Any]] = list(reversed(recent_rev))

    if isinstance(open_orders, list) and open_orders and len(scoped_intents) > len(recent_intents_out):
        older_resting: list[dict[str, Any]] = []
        for idx, i in enumerate(scoped_intents):
            if idx in recent_indices:
                continue
            if str(i.get("status") or "").strip().upper() != "EXECUTED":
                continue
            det = _resolve_intent_execution_details(
                i,
                now_ts,
                extract_notional_fn,
                recent_orders=recent_orders,
                open_orders=open_orders,
                verified_open_venues=verified_open_venues,
            )
            if (
                _is_open_resting_detail(det)
                and (det["order_id"] is None or str(det["order_id"]) not in seen_oids)
            ):
                if det["order_id"] is not None:
                    seen_oids.add(str(det["order_id"]))
                older_resting.append(_to_summary(i, det))
        if older_resting:
            recent_intents_out = older_resting + recent_intents_out
    return recent_intents_out


def build_last_cycle_execution_status(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> Optional[Any]:
    """Build closed-loop mathematical fill confirmation (fill_px, slippage_bps, status) for the last cycle's intent(s)."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)
    if not scoped_intents:
        return None
    def _eff_intent_ts(intent_obj: dict[str, Any]) -> float:
        base_t = safe_float(intent_obj.get("timestamp"), 0.0) or 0.0
        ep = intent_obj.get("execution_payload")
        if isinstance(ep, dict):
            ep_st = str(ep.get("status") or "").strip().upper()
            has_fill = (
                ep_st in ("FILLED", "PARTIALLY_FILLED", "CLOSED")
                or extract_executed_qty(ep, default=0.0) > 0.0
                or ep.get("filled_at") is not None
                or ep.get("filled_ts") is not None
            )
            if has_fill and ep_st not in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "CANCELED_OR_EXPIRED"):
                for k in ("filled_at", "filled_ts", "terminal_ts", "updated_ts"):
                    ft = _parse_epoch_or_iso_s(ep.get(k))
                    if ft is not None and ft > 0:
                        base_t = max(base_t, _ts_ms_to_s(ft))
        return base_t

    last_ts = max((_eff_intent_ts(i) for i in scoped_intents), default=0.0)
    if last_ts > 0:
        cycle_intents = [
            i for i in scoped_intents if abs(_eff_intent_ts(i) - last_ts) <= 90.0
        ]
    else:
        cycle_intents = [scoped_intents[-1]]

    def _intent_sym_key(intent_obj: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
        return _intent_venue_symbol_side_key(
            intent_obj.get("venue"), intent_obj.get("symbol"), extract_intent_side(intent_obj)
        )

    def _build_status_rec(det: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "venue": det["venue"],
            "symbol": det["symbol"],
            "side": det["side"],
            "order_type": det["order_type"],
            "status": det["exec_status"],
            "fill_px": det["fill_px"],
            "ref_px": det["ref_px"],
            "filled_qty": det["filled_qty"],
            "notional_usd": det["notional_usd"],
            "slippage_bps": det["slippage_bps"],
        }
        for k in ("filled_notional_usd", "limit_px", "order_id"):
            if det.get(k) is not None:
                rec[k] = det[k]
        if det.get("reason"):
            rec["reason"] = det["reason"]
        eff_ts = safe_pos_float(det.get("fill_ts_val"), default=None) or det["ts_val"]
        if eff_ts > 0 and now_ts >= eff_ts:
            rec["age_min"] = round((now_ts - eff_ts) / 60.0, 1)
        if det["ts_val"] > 0 and eff_ts > det["ts_val"] + 60.0 and now_ts >= det["ts_val"]:
            rec["placed_age_min"] = round((now_ts - det["ts_val"]) / 60.0, 1)
        return rec

    out: list[dict[str, Any]] = []
    seen_oids: set[str] = set()
    seen_sym_keys: set[tuple[str, str, tuple[str, ...], str]] = set()
    cycle_ids = {id(i) for i in cycle_intents}
    for i in cycle_intents:
        det = _resolve_intent_execution_details(
            i,
            now_ts,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        eff_ts_det = safe_pos_float(det.get("fill_ts_val"), default=None) or det["ts_val"]
        if (
            not _is_open_resting_detail(det)
            and det.get("exec_status") not in ("OPEN_RESTING", "PARTIALLY_FILLED")
            and eff_ts_det > 0
            and now_ts > eff_ts_det
            and (now_ts - eff_ts_det) > 3600.0
        ):
            continue
        if det["order_id"] is not None:
            seen_oids.add(str(det["order_id"]))
        if _is_open_resting_detail(det):
            seen_sym_keys.add(_intent_sym_key(i))
        out.append(_build_status_rec(det))

    if isinstance(open_orders, list) and open_orders:
        carried_resting: list[dict[str, Any]] = []
        for i in reversed(scoped_intents):
            if id(i) in cycle_ids:
                continue
            if str(i.get("status") or "").strip().upper() != "EXECUTED":
                continue
            det = _resolve_intent_execution_details(
                i,
                now_ts,
                extract_notional_fn,
                recent_orders=recent_orders,
                open_orders=open_orders,
                verified_open_venues=verified_open_venues,
            )
            if not _is_open_resting_detail(det):
                continue
            oid_str = str(det["order_id"]) if det["order_id"] is not None else None
            s_key = _intent_sym_key(i)
            if oid_str is not None:
                if oid_str in seen_oids:
                    continue
                seen_oids.add(oid_str)
            elif s_key in seen_sym_keys:
                continue
            seen_sym_keys.add(s_key)
            carried_resting.append(_build_status_rec(det))
        if carried_resting:
            out = list(reversed(carried_resting)) + out

    if not out:
        return None
    return out[0] if len(out) == 1 else out


