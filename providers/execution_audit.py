"""Observational lifecycle audit for ``StrategyExecutor`` orders.

This layer neither authorizes an order nor changes the submitted financial values.
Writes are best-effort, so an audit I/O failure does not change the executor result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .strategy_executor import (
    OrderStatus, ProviderError, SubmissionOutcome, SubmissionRefused,
    capture_submission, reconciliation_capabilities_of,
)

try:  # Linux live: cross-process serialization; safe per-process fallback.
    import fcntl
except ImportError:  # pragma: no cover - Windows/import tooling
    fcntl = None


_WRITE_LOCK = threading.Lock()
_UUID_HEX_SUFFIX = re.compile(r"([0-9a-fA-F]{32})$")


def _parse_cache_max() -> int:
    raw = str(os.environ.get("EXECUTION_AUDIT_CACHE_MAX", "10000") or "").split("#", 1)[0].strip()
    try:
        f = float(raw)
        if math.isfinite(f):
            return max(100, int(f))
    except (ValueError, OverflowError):
        pass
    return 10000


_CACHE_MAX = _parse_cache_max()


def _slug(value: object) -> str:
    if isinstance(value, bool):
        value = ""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return text.strip("-") or "unknown"


def _json_safe_value(val: Any) -> Any:
    if isinstance(val, float):
        return val if math.isfinite(val) else None
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
    prefix = "-".join((_slug(venue).lower(), _slug(symbol), _slug(kind or "order").lower()))
    return f"{prefix}-{uuid.uuid4().hex}"


def intent_client_order_id(venue: str, intent_id: str) -> Optional[str]:
    """Encode an intent in the venue's client-ID format without additional state.

    Preserve the complete 128-bit UUID. Legacy/nonstandard intents use a
    deterministic hash so the same intent produces the same identifier after restart.
    """
    if intent_id is None or isinstance(intent_id, bool):
        return None
    raw = str(intent_id).strip()
    if not raw:
        return None
    match = _UUID_HEX_SUFFIX.search(raw)
    token = (match.group(1).lower() if match else
             hashlib.blake2s(raw.encode("utf-8"), digest_size=16).hexdigest())
    normalized_venue = _slug(venue).lower()
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
            raw_ts = self._clock()
            if isinstance(raw_ts, bool) or raw_ts is None:
                ts = time.time()
            else:
                try:
                    ts = float(raw_ts)
                except (TypeError, ValueError, OverflowError):
                    ts = time.time()
                if not math.isfinite(ts) or ts <= 0 or ts >= 32503680000:
                    ts = time.time()
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
            line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)
            os.makedirs(self.directory, exist_ok=True)
            day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
            path = os.path.join(self.directory, f"execution_audit_{day}.jsonl")
            with _WRITE_LOCK:
                with open(path, "a", encoding="utf-8") as handle:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        handle.write(line + "\n")
                        handle.flush()
                    finally:
                        if fcntl is not None:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
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
        return str(symbol).strip().upper()

    @staticmethod
    def _norm_order_id_key(order_id: object) -> str:
        from .base import _clean_single_order_id
        cleaned = _clean_single_order_id(order_id)
        if cleaned is not None:
            return cleaned
        if order_id is None or isinstance(order_id, bool):
            return ""
        return str(order_id).strip()

    def _remember(self, symbol: str, order_id: str, intent_id: str) -> None:
        oid_key = self._norm_order_id_key(order_id)
        if not oid_key:
            return
        sym_key = self._norm_symbol_key(symbol)
        with self._lock:
            key = (sym_key, oid_key)
            self._intent_by_order[key] = intent_id
            self._intent_by_order.move_to_end(key)
            while len(self._intent_by_order) > _CACHE_MAX:
                self._intent_by_order.popitem(last=False)

    def _intent(self, symbol: str, order_id: str, explicit: Optional[str] = None) -> str:
        if explicit:
            self._remember(symbol, order_id, explicit)
            return explicit
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
        client_order_id = client_order_id or intent_client_order_id(self.name, intent_id)
        fields = {
            "side": str(side).lower(), "qty": qty, "price": price,
            "market": bool(market), "kind": kind,
            "reference_price": reference_price,
            "client_order_id": client_order_id,
        }
        self.audit.record("submit_requested", intent_id=intent_id, venue=self.name,
                          symbol=symbol, **fields)
        outcome = self.submit_outcome_with_intent(
            intent_id, symbol, side, qty, price, market=market, kind=kind,
            reference_price=reference_price, client_order_id=client_order_id,
            cache_permit=cache_permit, _requested_recorded=True,
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
        client_order_id = client_order_id or intent_client_order_id(self.name, intent_id)
        fields = {
            "side": str(side).lower(), "qty": qty, "price": price,
            "market": bool(market), "kind": kind,
            "reference_price": reference_price,
            "client_order_id": client_order_id,
        }
        if not _requested_recorded:
            self.audit.record("submit_requested", intent_id=intent_id, venue=self.name,
                              symbol=symbol, **fields)
        submit_kwargs = {"market": market, "kind": kind}
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
            self._last_status[key] = fingerprint
            self._last_status.move_to_end(key)
            while len(self._last_status) > _CACHE_MAX:
                self._last_status.popitem(last=False)
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
        from .base import _clean_single_order_id
        clean_cid = _clean_single_order_id(client_order_id)
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
        return True if not callable(method) else bool(method())

    def get_current_price(self, symbol: str):
        return self._executor.get_current_price(symbol)

    def pair_precision(self, symbol: str):
        return self._executor.pair_precision(symbol)

    def free_balance(self, asset: str):
        return self._executor.free_balance(asset)

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Run the adapter's optional preflight before a submit is audited."""
        preflight = getattr(self._executor, "preflight_order", None)
        if callable(preflight):
            return preflight(
                symbol, side, qty, price, market=market, kind=kind,
            )
        return None

    def ohlc_closes(self, symbol: str, interval_min: int):
        return self._executor.ohlc_closes(symbol, interval_min)

    def ohlc_series(self, symbol: str, interval_min: int):
        """Preserve optional candle timestamps without expanding the protocol."""
        reader = getattr(self._executor, "ohlc_series", None)
        if callable(reader):
            return reader(symbol, interval_min)
        if isinstance(interval_min, bool) or interval_min is None:
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
        try:
            itv_f = float(interval_min)
            if not math.isfinite(itv_f) or not itv_f.is_integer():
                raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
            interval_i = int(itv_f)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}") from exc
        if interval_i <= 0:
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
        from market_regime import ClosedPriceSeries
        return ClosedPriceSeries(
            tuple(self._executor.ohlc_closes(symbol, interval_i) or ()),
            interval_i,
        )
