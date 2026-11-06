"""Observational lifecycle audit for ``StrategyExecutor`` orders.

This layer neither authorizes an order nor changes the submitted financial values.
Writes are best-effort, so an audit I/O failure does not change the executor result.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import sys
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
    clean_scale,
    conf_or_override,
    exact_pos_int,
    float_env,
    lru_dict_put,
    normalize_side,
    parse_notify_enabled_flag,
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
    resolve_cachedb_filepath,
    split_symbol_assets,
)
from instrument_registry import casefold_get
from market_regime import ClosedPriceSeries
from state_io import append_jsonl_record, generation_lock_path, load_json_state, read_jsonl_dicts, save_json_state
from .base import MarketDataProvider
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


def ai_intent_client_order_id(intent_id: Any) -> Optional[str]:
    """Durable ``AI_`` client ID of an autonomous intent (<= 36 chars, Binance newClientOrderId compatible)."""
    raw = "" if isinstance(intent_id, (bool, list, tuple, dict, set, frozenset)) else str(intent_id or "").strip()
    if not raw:
        return None
    cid_body = (
        f"{raw[:18]}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}_{raw[-7:]}"
        if len(raw) > 33
        else raw
    )
    return f"AI_{cid_body}"


def venue_native_client_order_ids(client_order_id: Any) -> set[str]:
    """Return the lowercase venue-native forms of one durable client ID (Binance raw, Kraken UUID hex, HL cloid)."""
    raw = clean_order_id(client_order_id)
    if not raw:
        return set()
    out = {raw.lower()}
    try:
        from .kraken_provider import _kraken_client_order_id
        out.add(_kraken_client_order_id(raw).lower())
    except Exception:
        pass
    try:
        from .hyperliquid_provider import _cloid_for_client_order_id
        out.add(_cloid_for_client_order_id(raw).lower())
    except Exception:
        pass
    return out


def intent_venue_client_order_ids(intent: Any) -> set[str]:
    """Return all lowercase venue-native client IDs an autonomous intent's order may carry on any venue."""
    if not isinstance(intent, dict):
        return set()
    return venue_native_client_order_ids(ai_intent_client_order_id(_extract_intent_id(intent)))


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

    _norm_order_id_key = staticmethod(clean_order_id_str)

    @staticmethod
    def _lru_put(cache: OrderedDict, key: Any, value: Any) -> None:
        lru_dict_put(cache, key, value, _CACHE_MAX)

    def _remember(self, symbol: str, order_id: str, intent_id: str) -> None:
        oid_key = self._norm_order_id_key(order_id)
        if not oid_key:
            return
        sym_key = self._norm_symbol_key(symbol)
        with self._lock:
            self._lru_put(self._intent_by_order, (sym_key, oid_key), intent_id)

    def _intent(self, symbol: str, order_id: str, explicit: Optional[str] = None) -> str:
        clean_exp = self._norm_order_id_key(explicit)
        if clean_exp:
            self._remember(symbol, order_id, clean_exp)
            return clean_exp
        oid_key = self._norm_order_id_key(order_id)
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
        clean_cid = self._norm_order_id_key(client_order_id) or None
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
        oid_key = self._norm_order_id_key(order_id)
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

    def get_orderbook(self, symbol: str, limit: int = 200):
        return self._call_opt("get_orderbook", symbol, limit)

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

    def strategy_dry_run(self, force_paper: bool, strategy_execute: bool) -> bool:
        fn = getattr(self._executor, "strategy_dry_run", None)
        if callable(fn):
            return bool(fn(force_paper, strategy_execute))
        return MarketDataProvider.resolve_strategy_dry_run(force_paper, strategy_execute, self.execution_enabled())

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


def _extract_intent_id(record: Any, default: str = "") -> str:
    """Return a clean non-boolean intent_id (falling back to ``id``) from ``record``."""
    if not isinstance(record, dict):
        return default
    for key in ("intent_id", "id"):
        raw = record.get(key)
        if raw is not None and not isinstance(raw, bool):
            clean = str(raw).strip()
            if clean:
                return clean
    return default


def append_intent_history(intent: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append an entry to ``intent['history']``, coercing null or non-list history fields."""
    hist = intent.get("history")
    if not isinstance(hist, list):
        hist = []
        intent["history"] = hist
    hist.append(entry)


def _record_intent_state(
    intent: dict[str, Any],
    status: str,
    now_ts: float,
    *,
    action: Optional[str] = None,
    **fields: Any,
) -> None:
    """Set ``intent['status'] = status`` and append a timestamped history entry."""
    intent["status"] = status
    append_intent_history(intent, {"ts": now_ts, "action": action or status, **fields})


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
            is_eq = (
                v_norm == "t212"
                if v_norm
                else str(i.get("symbol") or "").strip().upper().endswith(("_EQ", "_US_EQ"))
            )
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
    eff_notional = (
        safe_pos_float(exec_pay.get("cummulativeQuoteQty"), default=None)
        or safe_pos_float(exec_pay.get("cost"), default=None)
        or safe_pos_float(exec_pay.get("filledValue"), default=None)
        or safe_pos_float(i.get("clamped_notional_usd"), default=None)
        or safe_pos_float(last_hist.get("notional_usd"), default=None)
        or extract_notional_fn(i, default=None)
    )
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
    filled_notional_usd = (
        round(eff_qty * fill_px, 2)
        if exec_status == "PARTIALLY_FILLED" and eff_qty is not None and eff_qty > 0 and fill_px is not None and fill_px > 0
        else None
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


def _intent_venue_symbol_side_key(
    venue: Any, symbol: Any, side: Any
) -> tuple[str, str, tuple[str, ...], str]:
    s_raw = str(symbol or "").strip().upper()
    v_norm = normalize_venue_name(venue)
    base = canonical_base_asset(s_raw) or s_raw
    if base and (_is_eq_suffix(s_raw) or v_norm == "t212"):
        base = f"EQ:{base}"
    return (
        v_norm,
        base,
        compatible_quote_suffixes(s_raw),
        str(side or "").strip().upper(),
    )


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


_CURRENCY_STRIP_RE = re.compile(r"[\$,\s]|(?:USD[CT]?)\b", re.IGNORECASE)


def _parse_currency_pos_float(raw: Any, default: Any = None) -> Any:
    if isinstance(raw, str):
        raw = _CURRENCY_STRIP_RE.sub("", raw).strip()
    return safe_pos_float(raw, default)


def extract_intent_side(data: dict[str, Any]) -> str:
    """Extract normalized BUY/SELL/HOLD side from an intent dictionary."""
    if not isinstance(data, dict):
        return ""
    for k in ("decision", "action", "side"):
        if data.get(k) is not None:
            norm = normalize_side(data.get(k))
            if norm:
                return norm
    return ""


_NOTIONAL_KEYS: tuple[str, ...] = (
    "suggested_notional_usd",
    "notional_usd",
    "amount_usd",
    "size_usd",
    "usd_notional",
    "notional",
    "value_usd",
)


def extract_intent_notional(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive suggested notional USD across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    for nk in _NOTIONAL_KEYS:
        if data.get(nk) is not None:
            val = _parse_currency_pos_float(data.get(nk), None)
            if val is not None:
                return val
    return default


def _resolve_clamped_intent_notional(
    data: dict[str, Any],
    min_notional_usd: float,
    max_notional_usd: float,
) -> float:
    """Resolve and clamp suggested notional USD; explicit non-positive numeric values clamp to min_notional_usd."""
    pos = extract_intent_notional(data, None)
    if pos is not None:
        return max(min_notional_usd, min(float(pos), max_notional_usd))
    if isinstance(data, dict):
        for nk in _NOTIONAL_KEYS:
            raw = data.get(nk)
            if raw is None or isinstance(raw, bool):
                continue
            if isinstance(raw, str):
                raw = _CURRENCY_STRIP_RE.sub("", raw).strip()
            num = safe_float(raw, None)
            if num is not None and num <= 0.0:
                return float(min_notional_usd)
    return max(min_notional_usd, float(max_notional_usd))


def extract_intent_limit_price(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive passive limit price across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    keys: tuple[str, ...] = ("limit_price", "target_price", "order_price", "limit_px")
    if str(data.get("order_type") or data.get("type") or "").strip().upper() == "LIMIT":
        keys = (*keys, "price")
    for pk in keys:
        if data.get(pk) is not None:
            val = _parse_currency_pos_float(data.get(pk), None)
            if val is not None:
                return val
    return default


def extract_intent_order_type(data: dict[str, Any]) -> str:
    """Resolve normalized order execution type ('MARKET' or 'LIMIT') for an intent."""
    if not isinstance(data, dict):
        return "MARKET"
    raw_type = data.get("order_type") or data.get("type")
    norm_type = "" if isinstance(raw_type, bool) or raw_type is None else str(raw_type).strip().upper()
    limit_px = extract_intent_limit_price(data, default=None)
    if norm_type == "MARKET":
        return "MARKET"
    if norm_type == "LIMIT" or limit_px is not None:
        return "LIMIT"
    return "MARKET"


def parse_intent_confidence(raw_conf: Any, default: float = 0.0) -> float:
    """Parse confidence into [0.0, 1.0], normalizing percentage strings or 1..100 values."""
    return clean_scale(raw_conf, default, allow_percent=True)


def normalize_trading_decisions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and deduplicate single or multi-coin trading decisions from Gemini's JSON plan."""
    if not isinstance(plan, dict):
        return []
    raw_multi = None
    for k in ("market_trading_decisions", "trading_decisions", "decisions"):
        if isinstance(plan.get(k), list):
            raw_multi = plan.get(k)
            break
    raw_single = plan.get("market_trading_decision")
    candidates: list[dict[str, Any]] = []
    for item in (
        *(raw_multi if isinstance(raw_multi, list) else ()),
        *([raw_single] if isinstance(raw_single, dict) and raw_single else ()),
    ):
        if isinstance(item, dict):
            d = dict(item)
            raw_val = d.get("decision") or d.get("action") or d.get("side")
            if isinstance(raw_val, bool) or raw_val is None:
                raw_side = "HOLD"
            else:
                raw_side = normalize_side(raw_val) or str(raw_val).strip().upper() or "HOLD"
            d["decision"] = raw_side
            raw_sym = d.get("symbol")
            if isinstance(raw_sym, str) and raw_sym.strip():
                d["symbol"] = raw_sym.strip().upper()
            elif isinstance(raw_sym, bool):
                d.pop("symbol", None)
            v_norm = normalize_venue_name(d.get("venue"))
            if v_norm:
                d["venue"] = v_norm
            if "confidence" in d and d.get("confidence") is not None:
                d["confidence"] = parse_intent_confidence(d.get("confidence"), default=0.0)
            extracted_usd = extract_intent_notional(d, default=None)
            if extracted_usd is not None:
                d["suggested_notional_usd"] = extracted_usd
            limit_px = extract_intent_limit_price(d, default=None)
            order_type = extract_intent_order_type(d)
            d["order_type"] = order_type
            if limit_px is not None and order_type == "LIMIT":
                d["limit_price"] = limit_px
            if "replace_open_orders" in d:
                d["replace_open_orders"] = _parse_bool_flag(d.get("replace_open_orders"), False)
            if "cancel_bot_open_orders" in d:
                d["cancel_bot_open_orders"] = _parse_bool_flag(d.get("cancel_bot_open_orders"), False)
            candidates.append(d)

    def _sort_key(d: dict[str, Any]) -> tuple[int, float]:
        side = str(d.get("decision") or "HOLD").strip().upper()
        is_action = 1 if side in ("BUY", "SELL") else 0
        conf = parse_intent_confidence(d.get("confidence"), default=0.0)
        return (-is_action, -conf)

    candidates.sort(key=_sort_key)

    decisions: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for item in candidates:
        sym_raw = str(item.get("symbol") or "").strip().upper()
        sym_canon = canonical_base_asset(sym_raw) or sym_raw
        venue_norm = normalize_venue_name(item.get("venue"))
        if sym_canon and (_is_eq_suffix(sym_raw) or venue_norm == "t212"):
            sym_canon = f"EQ:{sym_canon}"
        side_norm = str(item.get("decision") or "HOLD").strip().upper()
        key = (
            (venue_norm, sym_canon, compatible_quote_suffixes(sym_raw), "")
            if sym_canon
            else (venue_norm, "", (), side_norm)
        )
        if key not in seen_keys:
            seen_keys.add(key)
            decisions.append(item)
    return decisions


def resolve_intent_topic_tag(intent: dict[str, Any], venue: str) -> str:
    """Resolve canonical topic bracket tag ('[auto_cryptotrade_executor]' or '[auto_sharetrade_executor]') for an intent."""
    from orchestratorOS.agy.topics import TOPIC_ALIASES, TOPIC_SPECS, topic_spec

    raw_src = intent.get("created_by")
    src_topic = "" if raw_src is None or isinstance(raw_src, bool) else str(raw_src).strip()
    if src_topic in TOPIC_SPECS or src_topic in TOPIC_ALIASES:
        return topic_spec(src_topic).bracket_tag
    if venue == "t212":
        return topic_spec("auto_sharetrade_executor").bracket_tag
    return topic_spec("auto_cryptotrade_executor").bracket_tag


def generate_unique_intent_id(
    now_ts: float,
    side: str,
    symbol: str,
    existing_intents: list[dict[str, Any]],
) -> str:
    """Generate a unique intent_id for the durable intent queue."""
    base_id = f"intent_{int(now_ts)}_{side.lower()}_{symbol.lower()}"
    existing_ids = {_extract_intent_id(i) for i in existing_intents if isinstance(i, dict)} - {""}
    intent_id = base_id
    suffix = len(existing_intents) + 1
    while intent_id in existing_ids:
        intent_id = f"{base_id}_{suffix}"
        suffix += 1
    return intent_id


def _fmt_intent_qty(val: Any) -> str:
    fval = safe_pos_float(val, default=0.0) or 0.0
    return f"{fval:.8f}".rstrip("0").rstrip(".") or "0"


def _fmt_intent_px(val: Any, *, comma: bool = False) -> str:
    fval = safe_pos_float(val, default=0.0) or 0.0
    if fval >= 1.0 and round(fval, 2) == round(fval, 6):
        return f"{fval:,.2f}" if comma else f"{fval:.2f}"
    raw = (f"{fval:,.8f}" if comma else f"{fval:.8f}").rstrip("0")
    if raw.endswith("."):
        return f"{raw}00"
    if len(raw.rsplit(".", 1)[-1]) == 1:
        return f"{raw}0"
    return raw


def record_shadow_intent_observation(
    inst: Any,
    intent: dict[str, Any],
    *,
    decision: str,
    target_qty: float,
    price: float,
    effective_usd: float,
    confidence: float,
    guard_allowed: bool,
    guard_reason: str,
    thesis: str,
    trade_tag: str,
    now_ts: float,
    is_limit: bool = False,
    dispatch_notify_fn: Optional[Callable[..., Any]] = None,
    log: Optional[Any] = None,
) -> dict[str, Any]:
    """Record a shadow-mode intent observation and dispatch notification."""
    ord_lbl = f"{decision} LIMIT" if is_limit else decision
    qty_s = _fmt_intent_qty(target_qty)
    px_log_s = _fmt_intent_px(price, comma=False)
    px_body_s = _fmt_intent_px(price, comma=True)
    if log:
        log.info(
            "[LLM TRADER · SHADOW] Would execute %s %s (qty=%s, price=$%s, notional=$%.2f) on %s. Thesis: %s",
            ord_lbl, inst.symbol, qty_s, px_log_s, effective_usd, inst.provider_name, thesis,
        )
    title = f"🟢 [{trade_tag.strip('[]')} · SHADOW] {ord_lbl} {inst.symbol} ${effective_usd:.0f}"
    body = (
        f"Venue: {inst.provider_name} · Qty: {qty_s} @ ${px_body_s} ({'LIMIT' if is_limit else 'MARKET'})\n"
        f"Confidence: {confidence:.2f} · Notional: ${effective_usd:.2f}\n"
        f"Operational Preflight: {'PASSED' if guard_allowed else guard_reason}\n"
        f"Thesis: {thesis}"
    )
    if callable(dispatch_notify_fn):
        dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

    extra_limit = {"order_type": "LIMIT", "limit_price": price} if is_limit else {}
    _record_intent_state(
        intent,
        "OBSERVED_SHADOW",
        now_ts,
        guard_allowed=guard_allowed,
        guard_reason=guard_reason,
        price=price,
        qty=target_qty,
        notional_usd=effective_usd,
        **extra_limit,
    )
    return intent


def execute_enforce_intent_order(
    inst: Any,
    intent: dict[str, Any],
    *,
    intent_id: str,
    decision: str,
    target_qty: float,
    price: float,
    effective_usd: float,
    guard_allowed: bool,
    guard_reason: str,
    thesis: str,
    trade_tag: str,
    now_ts: float,
    bypass_guards: Any,
    is_limit: bool = False,
    dispatch_notify_fn: Optional[Callable[..., Any]] = None,
    log: Optional[Any] = None,
) -> dict[str, Any]:
    """Execute an enforce-mode trade intent via Instrument.place with 1 retry on stale account cache."""
    ord_lbl = f"{decision} LIMIT" if is_limit else decision
    px_body_s = _fmt_intent_px(price, comma=True)
    if not guard_allowed:
        if log:
            log.warning(
                "[LLM TRADER · ENFORCE] Operational guard blocked %s %s: %s",
                ord_lbl, inst.symbol, guard_reason,
            )
        title = f"🛡 {trade_tag} BLOCKED {ord_lbl} {inst.symbol}"
        body = (
            f"Venue: {inst.provider_name} · Price: ${px_body_s} · Notional: ${effective_usd:.0f}\n"
            f"Reason: {guard_reason}\n"
            f"Thesis: {thesis}"
        )
        if callable(dispatch_notify_fn):
            dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

        _record_intent_state(
            intent,
            "BLOCKED_BY_GUARD",
            now_ts,
            guard_reason=guard_reason,
            price=price,
            qty=target_qty,
            notional_usd=effective_usd,
        )
        return intent

    if log:
        log.warning(
            "[LLM TRADER · ENFORCE] Executing %s %s (qty=%s, price=$%s, notional=$%.2f) on %s...",
            ord_lbl, inst.symbol, _fmt_intent_qty(target_qty), _fmt_intent_px(price, comma=False), effective_usd, inst.provider_name,
        )

    try:
        intent_cid = ai_intent_client_order_id(intent_id)
        outcome_ctx: dict[str, Any] = {}
        order_res = None

        def _cleanup_refused_retry_records() -> None:
            if not intent_id:
                return
            try:
                import order_retry

                rm_ids = [
                    str(r["id"])
                    for r in order_retry.load_all()
                    if isinstance(r, dict)
                    and str(r.get("intent_id") or "") == str(intent_id)
                    and str(r.get("submission_state") or "").lower() == "refused"
                    and r.get("id") is not None
                ]
                if rm_ids:
                    order_retry.discard(rm_ids)
            except Exception:
                pass

        stale_cache_retries = 0
        retried_insufficient_free = False
        for _attempt in range(5):
            outcome_ctx.clear()
            place_kwargs: dict[str, Any] = {
                "side": decision,
                "qty": target_qty,
                "price": price,
                "market": not is_limit,
                "enforce_business_minimum": False,
                "wait_for_trend": False,
                "intent_id": intent_id,
                "client_order_id": intent_cid,
                "regime_context": None,
                "bypass_guards": bypass_guards,
                "_outcome_context": outcome_ctx,
            }
            if is_limit:
                place_kwargs["smart"] = False
            order_res = inst.place(**place_kwargs)
            if order_res:
                break
            refused_state = outcome_ctx.get("state") == "refused" or outcome_ctx.get("category") == "refused_pre_submit"
            reason_str = str(outcome_ctx.get("reason") or "")
            if (
                stale_cache_retries < 3
                and refused_state
                and reason_str in ("account_cache_not_fresh", "account_cache_snapshot_changed")
            ):
                stale_cache_retries += 1
                _cleanup_refused_retry_records()
                time.sleep(0.25 if ("PYTEST_CURRENT_TEST" in os.environ or "unittest" in sys.modules) else 1.5)
                continue
            if (
                not retried_insufficient_free
                and refused_state
                and decision == "BUY"
                and "insufficient-free-for-stocks-buy" in reason_str.lower()
            ):
                retried_insufficient_free = True
                from .quantity import _floor_2dec, round_provider_qty

                scaled_qty = _floor_2dec(
                    round_provider_qty(getattr(inst, "_provider", None), str(getattr(inst, "symbol", "") or ""), target_qty * 0.94)
                )
                if 0.0 < scaled_qty < target_qty and (price <= 0.0 or scaled_qty * price >= 1.0):
                    target_qty = scaled_qty
                    if price > 0.0:
                        effective_usd = round(target_qty * price, 2)
                    _cleanup_refused_retry_records()
                    continue
            break
        if order_res:
            _cleanup_refused_retry_records()
            if log:
                log.info(
                    "[LLM TRADER · ENFORCE] Successfully placed %s %s: %s",
                    ord_lbl, inst.symbol, order_res,
                )
            act_verb = "PLACED LIMIT" if is_limit else "EXECUTED"
            icon = "📝" if is_limit else "🎉"
            title = f"{icon} {trade_tag} {act_verb} {decision} {inst.symbol} ${effective_usd:.0f}"
            order_id_str = extract_order_id(order_res, default="N/A")
            body = (
                f"Venue: {inst.provider_name} · Qty: {_fmt_intent_qty(target_qty)} @ ${_fmt_intent_px(price, comma=True)} "
                f"({'LIMIT' if is_limit else 'MARKET'} · ${effective_usd:,.0f})\n"
                f"Order ID: {order_id_str}\n"
                f"Thesis: {thesis}"
            )
            intent["execution_payload"] = order_res
            extra_limit: dict[str, Any] = {"order_type": "LIMIT", "limit_price": price} if is_limit else {}
            if order_id_str and order_id_str != "N/A":
                extra_limit["order_id"] = order_id_str
            _record_intent_state(
                intent,
                "EXECUTED",
                now_ts,
                price=price,
                qty=round(target_qty, 8),
                notional_usd=effective_usd,
                payload=order_res,
                **extra_limit,
            )

            if callable(dispatch_notify_fn):
                try:
                    from notify_engine.alertnotifiers import notify_order_executed, notify_order_placed

                    notify_kwargs: dict[str, Any] = {
                        "venue": inst.provider_name,
                        "owner": "ai_executor",
                        "source": None,
                        "title": title,
                        "body": body,
                        "notify_fn": dispatch_notify_fn,
                        "include_price": False,
                    }
                    if is_limit:
                        notify_order_placed(
                            inst.symbol,
                            decision,
                            order_id_str,
                            price,
                            target_qty,
                            order_type="LIMIT",
                            allow_ai_executor=True,
                            **notify_kwargs,
                        )
                    else:
                        notify_order_executed(
                            inst.symbol,
                            decision,
                            order_id_str,
                            price,
                            target_qty,
                            **notify_kwargs,
                        )
                except Exception as notify_exc:
                    if log:
                        log.debug("Notification failed after order execution: %s", notify_exc)
        else:
            refusal_reason = str(outcome_ctx.get("reason") or "Instrument.place returned None")
            refusal_cat = str(outcome_ctx.get("category") or "")
            if log:
                log.warning(
                    "[LLM TRADER · ENFORCE] Order placement refused or failed for %s %s: %s",
                    decision, inst.symbol, refusal_reason,
                )
            title = f"⚠️ {trade_tag} REFUSED {decision} {inst.symbol}"
            body = f"Venue {inst.provider_name} refused placement ({refusal_reason}).\nThesis: {thesis}"
            if callable(dispatch_notify_fn):
                dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

            extra_cat = {"category": refusal_cat} if refusal_cat else {}
            _record_intent_state(intent, "FAILED", now_ts, reason=refusal_reason, **extra_cat)
            if outcome_ctx.get("state") == "refused":
                _cleanup_refused_retry_records()
    except Exception as e:
        if log:
            log.error("[LLM TRADER · ENFORCE] Exception executing %s %s: %s", decision, inst.symbol, e)
        title = f"🚨 {trade_tag} ERROR {decision} {inst.symbol}"
        body = f"Execution exception: {e}\nThesis: {thesis}"
        if callable(dispatch_notify_fn):
            dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

        _record_intent_state(intent, "FAILED", now_ts, action="ERROR", error=str(e))

    return intent


def normalize_intent_fields(
    intent: dict[str, Any],
    now_ts: float,
) -> tuple[str, str, str, str]:
    """Extract and normalize (intent_id, decision, symbol, venue) from an intent dict."""
    intent_id = _extract_intent_id(intent, default=f"intent_{int(now_ts)}")
    decision = extract_intent_side(intent)
    raw_sym = intent.get("symbol", "")
    symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip().upper()
    raw_venue = intent.get("venue", "binance")
    if isinstance(raw_venue, bool):
        venue = "__invalid__"
    elif raw_venue is None or not str(raw_venue).strip():
        venue = "binance"
    else:
        venue = normalize_venue_name(str(raw_venue)) or str(raw_venue).strip().lower()
    return intent_id, decision, symbol, venue


def validate_and_build_intent_dict(
    decision: dict[str, Any],
    *,
    source: str,
    min_confidence: float,
    min_notional_usd: float,
    max_notional_usd: float,
    existing_intents: list[dict[str, Any]],
    now_ts: float,
    log: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Validate an LLM trading decision and build a canonical TradeIntentRecord dict, or return None if non-actionable."""
    if not isinstance(decision, dict):
        return None
    side = extract_intent_side(decision)
    if side not in ("BUY", "SELL"):
        raw_dec = (
            str(decision.get("decision") or decision.get("action") or decision.get("side") or "HOLD").strip().upper()
            or "HOLD"
        )
        if log:
            log.info("[INTENT GATEWAY] %s decision is %s (no intent).", source, raw_dec)
        return None
    raw_sym = decision.get("symbol", "")
    symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip().upper()
    raw_venue = decision.get("venue", "")
    venue = "" if isinstance(raw_venue, bool) or raw_venue is None else normalize_venue_name(str(raw_venue))
    if not symbol or not venue:
        if log:
            log.warning("[INTENT GATEWAY] %s %s decision missing symbol/venue (no intent).", source, side)
        return None
    confidence = parse_intent_confidence(decision.get("confidence"), 0.0)
    if confidence < min_confidence:
        if log:
            log.info(
                "[INTENT GATEWAY] %s %s %s confidence %.2f < %.2f (no intent).",
                source, side, symbol, confidence, min_confidence,
            )
        return None
    notional = _resolve_clamped_intent_notional(decision, min_notional_usd, max_notional_usd)
    intent_id = generate_unique_intent_id(now_ts, side, symbol, existing_intents)
    raw_urg = decision.get("urgency", "NORMAL")
    urgency = (
        "NORMAL"
        if isinstance(raw_urg, bool) or raw_urg is None or not str(raw_urg).strip()
        else str(raw_urg).strip().upper()
    )
    raw_thesis = decision.get("thesis", "")
    thesis = "" if isinstance(raw_thesis, bool) or raw_thesis is None else str(raw_thesis).strip()
    order_type = extract_intent_order_type(decision)
    limit_px = extract_intent_limit_price(decision, default=None) if order_type == "LIMIT" else None
    replace_open = _parse_bool_flag(decision.get("replace_open_orders"), default=False)
    cancel_bot_open = _parse_bool_flag(decision.get("cancel_bot_open_orders"), default=False)
    rec = {
        "intent_id": intent_id,
        "timestamp": now_ts,
        "decision": side,
        "symbol": symbol,
        "venue": venue,
        "confidence": confidence,
        "suggested_notional_usd": notional,
        "order_type": order_type,
        "limit_price": limit_px,
        "replace_open_orders": replace_open,
        "urgency": urgency,
        "thesis": thesis,
        "status": "PENDING",
        "execution_payload": None,
        "created_by": str(source or "").strip(),
        "history": [],
    }
    if cancel_bot_open:
        rec["cancel_bot_open_orders"] = True
    return rec


def _intent_order_ids(item: dict[str, Any]) -> set[str]:
    ids: set[str] = set(intent_venue_client_order_ids(item))
    oid_rec = extract_order_id(item.get("execution_payload"), default="")
    if oid_rec:
        ids.add(str(oid_rec))
    for h_entry in (item.get("history") if isinstance(item.get("history"), list) else ()):
        if isinstance(h_entry, dict):
            h_oid = extract_order_id(h_entry.get("order_id"), default="") or extract_order_id(h_entry.get("payload"), default="")
            if h_oid:
                ids.add(str(h_oid))
    return ids


def collect_known_ai_order_ids(
    intents: Any,
    initial_ids: Optional[set[str]] = None,
) -> set[str]:
    """Collect venue order IDs (``execution_payload`` / ``history``) and venue-native AI client IDs across intents."""
    ai_oids: set[str] = set(initial_ids or ())
    if isinstance(intents, dict):
        intents = intents.get("intents")
    if not isinstance(intents, list):
        return ai_oids
    for item in intents:
        if isinstance(item, dict):
            ai_oids.update(_intent_order_ids(item))
    return ai_oids


def cancel_existing_ai_open_orders(
    inst: Any,
    side: str,
    log: Optional[Any] = None,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    intents: Optional[list[dict[str, Any]]] = None,
    include_bot_orders: bool = False,
    any_side: bool = False,
    exclude_order_ids: Optional[set[str]] = None,
) -> list[str]:
    """Cancel prior unfilled AI-placed (or bot-placed when ``include_bot_orders=True``) open orders on the symbol."""
    canceled_ids: list[str] = []
    open_fn = getattr(inst, "open_orders", None)
    cancel_fn = getattr(inst, "cancel_order", None)
    want_side = normalize_side(side)
    if not callable(open_fn) or not callable(cancel_fn) or (not any_side and not want_side):
        return canceled_ids
    excluded = {str(x) for x in (exclude_order_ids or ()) if x}
    try:
        raw_open = open_fn()
        if not isinstance(raw_open, list):
            return canceled_ids
        ai_oids = collect_known_ai_order_ids(intents, known_ai_order_ids)
        for oo in raw_open:
            if not isinstance(oo, dict):
                continue
            oo_side = extract_order_side(oo) or normalize_side(oo.get("side") or oo.get("type"))
            if not any_side and oo_side != want_side:
                continue
            oid = extract_order_id(oo, default="")
            if oid and str(oid) in excluded:
                continue
            is_ai_order = resolve_order_actor(oo, ai_oids) == "AI"
            if not is_ai_order and not include_bot_orders:
                continue
            if oid:
                try:
                    if cancel_fn(str(oid)) is False:
                        if log:
                            log.warning(
                                "[INTENT GATEWAY] Venue refused cancel of prior open order %s on %s; it stays active.",
                                oid, getattr(inst, "symbol", ""),
                            )
                        continue
                    canceled_ids.append(str(oid))
                    if log:
                        log.info(
                            "[INTENT GATEWAY] Canceled prior %s open %s order %s on %s before replacement.",
                            "AI" if is_ai_order else "BOT",
                            oo_side or side,
                            oid,
                            getattr(inst, "symbol", ""),
                        )
                except Exception as ce:
                    if log:
                        log.debug("Could not cancel open order %s: %s", oid, ce)
    except Exception as e:
        if log:
            log.debug("Open order scan failed before replacement: %s", e)
    return canceled_ids


def find_duplicate_ai_open_limit_order(
    inst: Any,
    side: str,
    limit_price: float,
    notional_usd: float,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    intents: Optional[list[dict[str, Any]]] = None,
    require_notional_match: bool = True,
) -> Optional[str]:
    """Return the venue order ID of an unfilled AI open limit order at the same side, price (<= 1 bps), and (when required) notional (<= 5%)."""
    open_fn = getattr(inst, "open_orders", None)
    want_side = normalize_side(side)
    if not callable(open_fn) or not want_side or not (limit_price and limit_price > 0) or not (notional_usd and notional_usd > 0):
        return None
    try:
        raw_open = open_fn()
        if not isinstance(raw_open, list):
            return None
        ai_oids = collect_known_ai_order_ids(intents, known_ai_order_ids)
        if isinstance(intents, dict):
            intents = intents.get("intents")
        intent_notional_by_id: dict[str, tuple[float, float]] = {}
        if require_notional_match and isinstance(intents, list):
            for item in intents:
                if not isinstance(item, dict):
                    continue
                p_sug = safe_pos_float(item.get("suggested_notional_usd"), default=None)
                p_clamp = safe_pos_float(item.get("clamped_notional_usd"), default=None)
                if p_clamp is None and isinstance(item.get("history"), list):
                    for h_entry in reversed(item["history"]):
                        if isinstance(h_entry, dict) and str(h_entry.get("action") or "").upper() in ("EXECUTED", "PLACED"):
                            p_clamp = safe_pos_float(h_entry.get("notional_usd"), default=None)
                            if p_clamp is not None:
                                break
                if p_sug is None or p_clamp is None:
                    continue
                for k in _intent_order_ids(item):
                    if k:
                        intent_notional_by_id[k] = (p_sug, p_clamp)
                        intent_notional_by_id[k.lower()] = (p_sug, p_clamp)
        for oo in raw_open:
            if not isinstance(oo, dict):
                continue
            oo_side = extract_order_side(oo) or normalize_side(oo.get("side") or oo.get("type"))
            if oo_side != want_side:
                continue
            if resolve_order_actor(oo, ai_oids) != "AI":
                continue
            if extract_executed_qty(oo) > 0:
                continue
            oo_px = extract_order_price(oo)
            if oo_px <= 0 or abs(oo_px - limit_price) / limit_price > 1e-4:
                continue
            oo_qty = extract_orig_qty(oo)
            oo_notional = oo_px * oo_qty
            if oo_notional <= 0:
                continue
            oid = extract_order_id(oo, default="")
            if require_notional_match:
                notional_ok = abs(oo_notional - notional_usd) <= max(1.0, 0.05 * max(notional_usd, oo_notional))
                if not notional_ok and intent_notional_by_id:
                    cid = extract_client_order_id(oo, default="")
                    prior_pair = (
                        intent_notional_by_id.get(str(oid))
                        or intent_notional_by_id.get(str(oid).lower())
                    ) if oid else None
                    if prior_pair is None and cid:
                        prior_pair = (
                            intent_notional_by_id.get(str(cid))
                            or intent_notional_by_id.get(str(cid).lower())
                        )
                    if prior_pair is not None:
                        p_sug, p_clamp = prior_pair
                        if (
                            abs(p_sug - notional_usd) <= max(1.0, 0.05 * max(notional_usd, p_sug))
                            and abs(oo_notional - p_clamp) <= max(1.0, 0.05 * max(p_clamp, oo_notional))
                        ):
                            notional_ok = True
                if not notional_ok:
                    continue
            if oid:
                return str(oid)
    except Exception:
        return None
    return None


def parse_intent_gateway_config(
    conf: dict[str, str],
    *,
    mode: Optional[str] = None,
    max_notional_usd: Optional[float] = None,
    min_confidence: Optional[float] = None,
    notify_enabled: Optional[bool] = None,
) -> tuple[str, float, float, float, bool]:
    """Validate and parse (mode, min_notional_usd, max_notional_usd, min_confidence, notify_enabled) from config."""
    raw_mode = conf_or_override(conf, "autonomous_trading_mode", mode)
    norm_mode = "" if isinstance(raw_mode, bool) else str(raw_mode).strip().lower()
    if norm_mode not in ("off", "shadow", "enforce"):
        raise ValueError(
            f"Invalid autonomous_trading_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']"
        )

    raw_min = conf_or_override(conf, "autonomous_trading_min_notional_usd")
    min_usd = safe_pos_float(raw_min, default=None)
    if min_usd is None:
        raise ValueError(f"Invalid autonomous_trading_min_notional_usd {raw_min!r}")

    raw_max = conf_or_override(conf, "autonomous_trading_max_notional_usd", max_notional_usd)
    max_usd = safe_pos_float(raw_max, default=None)
    if max_usd is None or max_usd < min_usd:
        raise ValueError(
            f"Invalid autonomous_trading_max_notional_usd {raw_max!r}; "
            f"must be >= autonomous_trading_min_notional_usd ({min_usd})"
        )

    raw_conf = conf_or_override(conf, "autonomous_trading_min_confidence", min_confidence)
    conf_val = safe_nonneg_float(raw_conf, default=None)
    if conf_val is None or conf_val > 1.0:
        raise ValueError(
            f"Invalid autonomous_trading_min_confidence {raw_conf!r}; must be in [0.0, 1.0]"
        )

    notify_flag = parse_notify_enabled_flag(notify_enabled, conf, "autonomous_trading_notify")
    return norm_mode, min_usd, max_usd, conf_val, notify_flag


def resolve_intent_execution_price(
    intent: dict[str, Any],
    *,
    intent_id: str,
    decision: str,
    symbol: str,
    live_price: float,
    now_ts: float,
    log: Optional[Any] = None,
) -> tuple[bool, float, bool]:
    """Validate and resolve MARKET or LIMIT execution price against live_price.

    Returns ``(ok, effective_price, is_limit)``. When ``order_type == 'LIMIT'`` and
    ``limit_price`` is missing or outside ``[0.5x, 2.0x]`` of ``live_price``, clamps to a
    safe passive offset (``0.995 * live_price`` for BUY, ``1.005 * live_price`` for SELL)
    so valid trading decisions are never left unsent.
    """
    order_type = extract_intent_order_type(intent)
    limit_px = extract_intent_limit_price(intent, default=None)
    if order_type == "LIMIT":
        if limit_px is None or limit_px <= 0 or limit_px < live_price * 0.50 or limit_px > live_price * 2.00:
            fallback_mult = 0.995 if decision == "BUY" else 1.005
            price = round(live_price * fallback_mult, 8)
            if log:
                log.warning(
                    "[INTENT GATEWAY] Intent %s limit_price %r missing or out of [0.5x, 2.0x] bounds vs live $%.6g; "
                    "using passive fallback $%.6g for %s",
                    intent_id, limit_px, live_price, price, symbol,
                )
            append_intent_history(intent, {
                "ts": now_ts,
                "action": "FALLBACK_PASSIVE_LIMIT_PRICE",
                "requested_limit_price": limit_px,
                "fallback_limit_price": price,
                "live_price": live_price,
            })
        elif decision == "BUY" and limit_px > live_price:
            price = live_price
        elif decision == "SELL" and limit_px < live_price:
            price = live_price
        else:
            price = limit_px
        intent["order_type"] = "LIMIT"
        intent["limit_price"] = price
        return True, price, True
    intent["order_type"] = "MARKET"
    return True, live_price, False


def load_durable_intents_state(
    intents_file: str,
    reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
) -> list[dict[str, Any]]:
    """Load all trade intents from the durable JSON queue with `.previous` crash recovery."""
    with durable_intents_lock(intents_file):
        data = load_json_state(
            intents_file,
            default_factory=list,
            fail_closed=False,
            label="autonomous_trade_intents",
            root_type=list,
        )
        items = [item for item in data if isinstance(item, dict)]
        if callable(reconcile_fn):
            reconcile_fn(items)
        return items


def save_durable_intents_state(
    intents_file: str,
    intents: list[dict[str, Any]],
    log: Optional[Any] = None,
) -> bool:
    """Transactionally persist all intents to disk using `.previous` staging and atomic replace."""
    from orchestratorOS.lib.technical_repair_ssot import clean_json_value

    try:
        cleaned = clean_json_value(intents if isinstance(intents, list) else [])
        with durable_intents_lock(intents_file):
            save_json_state(intents_file, cleaned, indent=2)
        return True
    except Exception as e:
        if log:
            log.error("Failed saving intents to %s: %s", intents_file, e)
        return False


_INTENTS_LOCK_LOCAL = threading.local()


@contextmanager
def durable_intents_lock(intents_file: str):
    """Reentrant (per thread) cross-process lock serializing read-modify-write of the durable intents file.

    The crypto and share executors run as separate cron processes whose cycles can overlap; without this lock
    one process's load -> append -> save silently drops the other's intents (lost update).
    """
    from lock.cooldown import FileLock

    key = generation_lock_path(str(resolve_cachedb_filepath(os.fspath(intents_file))))
    depths = getattr(_INTENTS_LOCK_LOCAL, "depths", None)
    if depths is None:
        depths = _INTENTS_LOCK_LOCAL.depths = {}
    if depths.get(key):
        depths[key] += 1
        try:
            yield
        finally:
            depths[key] -= 1
        return
    with FileLock(key):
        depths[key] = 1
        try:
            yield
        finally:
            depths.pop(key, None)


def upsert_durable_intent(
    intents_file: str,
    intent: dict[str, Any],
    *,
    reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
    log: Optional[Any] = None,
) -> bool:
    """Atomically merge one intent (by ``intent_id``) into the freshly reloaded durable intents file."""
    if not isinstance(intent, dict):
        return False
    iid = _extract_intent_id(intent)
    with durable_intents_lock(intents_file):
        intents = load_durable_intents_state(intents_file, reconcile_fn=reconcile_fn)
        idx = next(
            (k for k in range(len(intents) - 1, -1, -1) if iid and _extract_intent_id(intents[k]) == iid),
            None,
        )
        if idx is None:
            intents.append(intent)
        else:
            intents[idx] = intent
        return save_durable_intents_state(intents_file, intents, log=log)


def preflight_intent_and_resolve_price(
    intent: dict[str, Any],
    *,
    min_confidence: float,
    min_notional_usd: float,
    max_notional_usd: float,
    resolve_instrument_fn: Callable[..., Any],
    get_live_price_fn: Callable[[Any], Any],
    instruments_map: Optional[dict[str, Any]],
    now_ts: float,
    log: Optional[Any] = None,
) -> tuple[bool, str, str, str, str, float, Any, float, float, bool]:
    """Validate intent fields, confidence, instrument, and live/limit price before order execution."""
    intent_id, decision, symbol, venue = normalize_intent_fields(intent, now_ts)
    if decision not in ("BUY", "SELL"):
        _record_intent_state(intent, "REJECTED_INVALID_DECISION", now_ts, decision=decision)
        return False, intent_id, decision, symbol, venue, 0.0, None, 0.0, 0.0, False

    confidence = parse_intent_confidence(intent.get("confidence"), 0.0)
    intent["confidence"] = confidence
    if confidence < min_confidence:
        if log:
            log.info(
                "[INTENT GATEWAY] Intent %s rejected: confidence %.2f < %.2f",
                intent_id, confidence, min_confidence,
            )
        _record_intent_state(
            intent,
            "REJECTED_LOW_CONFIDENCE",
            now_ts,
            confidence=confidence,
            min_confidence=min_confidence,
        )
        return False, intent_id, decision, symbol, venue, confidence, None, 0.0, 0.0, False

    inst = resolve_instrument_fn(symbol, venue=venue, instruments_map=instruments_map)
    inst_sym_u = str(getattr(inst, "symbol", "") or "").strip().upper()
    inst_quote_u = str(getattr(inst, "quote", "") or "").strip().upper()
    if not inst or (
        normalize_venue_name(str(getattr(inst, "provider_name", ""))) != "t212"
        and ((len(inst_sym_u) > 3 and inst_sym_u.endswith("BTC")) or inst_quote_u == "BTC")
    ):
        if log:
            log.warning(
                "[INTENT GATEWAY] Intent %s rejected: unknown or non-fiat instrument %s on %s",
                intent_id, symbol, venue,
            )
        _record_intent_state(intent, "REJECTED_UNKNOWN_INSTRUMENT", now_ts, symbol=symbol, venue=venue)
        return False, intent_id, decision, symbol, venue, confidence, None, 0.0, 0.0, False

    effective_usd = _resolve_clamped_intent_notional(intent, min_notional_usd, max_notional_usd)

    live_price = safe_pos_float(get_live_price_fn(inst), default=None)
    if live_price is None:
        if log:
            log.warning("[INTENT GATEWAY] Intent %s rejected: invalid price for %s", intent_id, symbol)
        _record_intent_state(intent, "REJECTED_PRICE_UNAVAILABLE", now_ts, symbol=symbol)
        return False, intent_id, decision, symbol, venue, confidence, inst, effective_usd, 0.0, False

    intent["reference_price"] = round(live_price, 8)
    px_ok, price, is_limit = resolve_intent_execution_price(
        intent,
        intent_id=intent_id,
        decision=decision,
        symbol=symbol,
        live_price=live_price,
        now_ts=now_ts,
        log=log,
    )
    if not px_ok:
        return False, intent_id, decision, symbol, venue, confidence, inst, effective_usd, price, is_limit

    intent["requested_price"] = round(price, 8)
    return True, intent_id, decision, symbol, venue, confidence, inst, effective_usd, price, is_limit

