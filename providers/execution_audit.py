"""Observational lifecycle audit for ``StrategyExecutor`` orders.

This layer neither authorizes an order nor changes the submitted financial values.
Writes are best-effort, so an audit I/O failure does not change the executor result.
"""
from __future__ import annotations

import hashlib
import json
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
    exact_pos_int,
    float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import canonical_symbol_key

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
    return max(100, safe_pos_int(float_env("EXECUTION_AUDIT_CACHE_MAX"), 10000))


_CACHE_MAX = _parse_cache_max()


def _slug(value: object) -> str:
    if isinstance(value, bool):
        value = ""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return text.strip("-") or "unknown"


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
    from cachepaths import normalize_venue_name
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
    from cachepaths import normalize_venue_name
    from .base import _clean_single_order_id
    raw = _clean_single_order_id(intent_id)
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
        return canonical_symbol_key(symbol) or str(symbol).strip().upper().replace("/", "").replace("-", "").replace("_", "")

    @staticmethod
    def _norm_order_id_key(order_id: object) -> str:
        from .base import _clean_single_order_id
        return _clean_single_order_id(order_id) or ""

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
        return True if not callable(method) else _parse_bool_flag(method(), False)

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
                symbol, side, qty, price, market=_parse_bool_flag(market, False), kind=kind,
            )
        return None

    def ohlc_closes(self, symbol: str, interval_min: int):
        return self._executor.ohlc_closes(symbol, interval_min)

    def ohlc_series(self, symbol: str, interval_min: int):
        """Preserve optional candle timestamps without expanding the protocol."""
        reader = getattr(self._executor, "ohlc_series", None)
        if callable(reader):
            return reader(symbol, interval_min)
        interval_i = exact_pos_int(interval_min, default=None)
        if interval_i is None:
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
        from market_regime import ClosedPriceSeries
        return ClosedPriceSeries(
            tuple(self._executor.ohlc_closes(symbol, interval_i) or ()),
            interval_i,
        )


def append_intent_history(intent: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append an entry to ``intent['history']``, coercing null or non-list history fields."""
    hist = intent.get("history")
    if not isinstance(hist, list):
        hist = []
        intent["history"] = hist
    hist.append(entry)


def reconcile_intents_from_execution_audit(
    intents: list[dict[str, Any]],
    audit_dir: str,
) -> bool:
    """Reconcile FAILED or PENDING_EXECUTION intents against execution_audit_*.jsonl if later filled."""
    import glob

    failed_by_id: dict[str, dict[str, Any]] = {}
    for item in intents:
        if not isinstance(item, dict) or str(item.get("status") or "").strip().upper() not in ("FAILED", "PENDING_EXECUTION"):
            continue
        raw_id = item.get("intent_id") if item.get("intent_id") is not None else item.get("id")
        if raw_id is None or isinstance(raw_id, bool):
            continue
        clean_id = str(raw_id).strip()
        if clean_id:
            failed_by_id[clean_id] = item
    if not failed_by_id or not os.path.isdir(audit_dir):
        return False

    from state_io import read_jsonl_dicts

    matched_records: dict[str, dict[str, Any]] = {}
    for audit_path in sorted(glob.glob(os.path.join(glob.escape(audit_dir), "execution_audit_*.jsonl"))):
        for rec in read_jsonl_dicts(audit_path):
            raw_iid = rec.get("intent_id") if rec.get("intent_id") is not None else rec.get("id")
            if raw_iid is None or isinstance(raw_iid, bool):
                continue
            iid = str(raw_iid).strip()
            if iid not in failed_by_id:
                continue
            ev = str(rec.get("event") or "")
            if ev == "submit_accepted":
                matched_records.setdefault(iid, rec)
            elif ev in ("order_terminal", "order_terminal_retry"):
                v_stat = str(rec.get("venue_status") or "").upper()
                l_stat = str(rec.get("status") or "").lower()
                f_qty = safe_pos_float(
                    rec.get("filled_qty") if rec.get("filled_qty") is not None else rec.get("qty"),
                    0.0,
                )
                if (
                    v_stat in ("FILLED", "PARTIALLY_FILLED")
                    or l_stat in ("closed", "filled", "partially_filled")
                    or f_qty > 0.0
                ):
                    matched_records[iid] = rec
                elif matched_records.get(iid, {}).get("event") == "submit_accepted":
                    prev_oid = matched_records[iid].get("order_id")
                    cur_oid = rec.get("order_id")
                    if not prev_oid or not cur_oid or str(prev_oid) == str(cur_oid):
                        matched_records.pop(iid, None)

    if not matched_records:
        return False

    updated = False
    for iid, rec in matched_records.items():
        intent = failed_by_id[iid]
        filled_qty = rec.get("filled_qty") if rec.get("filled_qty") is not None else rec.get("qty")
        raw_cost = rec.get("cost")
        if raw_cost is None:
            f_q = safe_pos_float(filled_qty, default=None)
            f_px = safe_pos_float(
                rec.get("price") if rec.get("price") is not None else rec.get("reference_price"),
                default=None,
            )
            if f_q is not None and f_px is not None:
                raw_cost = round(f_q * f_px, 8)
        exec_payload: dict[str, Any] = {
            "orderId": rec.get("order_id"),
            "symbol": rec.get("symbol") or intent.get("symbol"),
            "side": str(rec.get("side") or intent.get("decision") or "").upper(),
            "executedQty": str(filled_qty) if filled_qty is not None else "",
            "cummulativeQuoteQty": str(raw_cost) if raw_cost is not None else "",
            "status": str(rec.get("venue_status") or "FILLED").upper(),
            "reconciled_from_audit": True,
        }
        intent["status"] = "EXECUTED"
        intent["execution_payload"] = exec_payload
        rec_ts = safe_pos_float(rec.get("ts"), time.time())
        append_intent_history(intent, {
            "ts": rec_ts,
            "action": "RECONCILED_EXECUTED",
            "order_id": rec.get("order_id"),
            "payload": exec_payload,
        })
        updated = True
    return updated


def _filter_intents_by_scope(intents_data: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    """Filter intent records to those matching the requested asset scope ('crypto', 'equities', or 'all')."""
    from cachepaths import normalize_venue_name

    if not isinstance(intents_data, list):
        return []
    return [
        i
        for i in intents_data
        if isinstance(i, dict)
        and (
            scope not in ("equities", "crypto")
            or (normalize_venue_name(i.get("venue")) == "t212") == (scope == "equities")
        )
    ]


def _resolve_intent_execution_details(
    i: dict[str, Any],
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Resolve fill price, filled quantity, execution status, and slippage (bps) for an intent."""
    from cachepaths import normalize_venue_name
    from .base import (
        extract_executed_qty,
        extract_order_id,
        extract_order_price,
        extract_order_side,
        extract_orig_qty,
    )

    venue = normalize_venue_name(i.get("venue")) or i.get("venue")
    symbol = str(i.get("symbol") or "").strip().upper()
    side = extract_intent_side(i) or str(i.get("decision") or "").strip().upper()
    order_type = str(i.get("order_type") or "MARKET").strip().upper()
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
    ts_val = safe_float(i.get("timestamp"), 0.0)
    order_id = extract_order_id(exec_pay, default=None) or extract_order_id(
        last_hist.get("order_id") or last_hist.get("payload"), default=None
    )

    matched_open = None
    if isinstance(open_orders, list):
        for oo in open_orders:
            if not isinstance(oo, dict):
                continue
            oo_oid = extract_order_id(oo, default=None)
            if order_id is not None and oo_oid is not None and str(oo_oid) == str(order_id):
                matched_open = oo
                break
            if (
                order_id is None
                and order_type == "LIMIT"
                and normalize_venue_name(oo.get("venue")) == venue
                and str(oo.get("symbol") or "").strip().upper() == symbol
                and (extract_order_side(oo) or str(oo.get("side") or "").strip().upper()) == side
            ):
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

    raw_intent_st = str(i.get("status") or "").strip().upper()
    venue_st = str(exec_pay.get("status") or "").strip().upper()
    if (
        matched_open is None
        and raw_intent_st == "EXECUTED"
        and venue_st not in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")
        and isinstance(recent_orders, dict)
        and symbol
    ):
        ro_entry = recent_orders.get(f"{venue}:{symbol}") or recent_orders.get(symbol)
        if isinstance(ro_entry, dict):
            side_key = "last_buy" if side == "BUY" else ("last_sell" if side == "SELL" else "")
            last_fill = ro_entry.get(side_key) if side_key else None
            if isinstance(last_fill, dict):
                f_ts_ms = safe_float(last_fill.get("ts_ms"), 0.0)
                if ts_val <= 0 or f_ts_ms <= 0 or f_ts_ms >= (ts_val - 120.0) * 1000.0:
                    if fill_px is None:
                        fill_px = safe_pos_float(last_fill.get("price"), default=None)
                    if filled_qty <= 0:
                        filled_qty = safe_pos_float(last_fill.get("qty"), default=0.0) or 0.0
    if fill_px is None and raw_intent_st in ("EXECUTED", "OBSERVED_SHADOW"):
        fill_px = safe_pos_float(
            last_hist.get("price")
            or i.get("requested_price")
            or i.get("limit_price")
            or i.get("reference_price"),
            default=None,
        )

    if matched_open is not None:
        exec_status = "PARTIALLY_FILLED" if filled_qty > 0 else "OPEN_RESTING"
    elif raw_intent_st == "EXECUTED":
        if venue_st in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED"):
            exec_status = venue_st
        elif venue_st == "PARTIALLY_FILLED":
            exec_status = "PARTIALLY_FILLED"
        else:
            exec_status = "FILLED"
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
    if exec_status in ("OPEN_RESTING", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED") and filled_qty <= 0:
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
    return {
        "venue": venue,
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "raw_status": i.get("status"),
        "venue_status": venue_st or None,
        "exec_status": exec_status,
        "order_id": str(order_id) if order_id is not None else None,
        "fill_px": round(fill_px, 6) if fill_px is not None else None,
        "ref_px": round(ref_px, 6) if ref_px is not None else None,
        "filled_qty": round(eff_qty, 8) if eff_qty is not None and eff_qty > 0 else None,
        "notional_usd": round(eff_notional, 2) if eff_notional is not None else i.get("suggested_notional_usd"),
        "slippage_bps": slippage_bps,
        "ts": i.get("timestamp"),
        "ts_val": ts_val,
    }


def summarize_recent_ai_intents(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Summarize the most recent AI trade intents filtered by asset scope."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)
    recent_intents_out: list[dict[str, Any]] = []
    for i in scoped_intents[-8:]:
        det = _resolve_intent_execution_details(
            i, now_ts, extract_notional_fn, recent_orders=recent_orders, open_orders=open_orders
        )
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
        if det["order_id"] is not None:
            intent_summary["order_id"] = det["order_id"]
        if det["fill_px"] is not None:
            intent_summary["fill_px"] = det["fill_px"]
        if det["filled_qty"] is not None:
            intent_summary["filled_qty"] = det["filled_qty"]
        if det["slippage_bps"] is not None:
            intent_summary["slippage_bps"] = det["slippage_bps"]
        if det["ts_val"] > 0 and now_ts >= det["ts_val"]:
            intent_summary["age_h"] = round((now_ts - det["ts_val"]) / 3600.0, 2)
        recent_intents_out.append(intent_summary)
    return recent_intents_out


def build_last_cycle_execution_status(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
) -> Optional[Any]:
    """Build closed-loop mathematical fill confirmation (fill_px, slippage_bps, status) for the last cycle's intent(s)."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)
    if not scoped_intents:
        return None
    last_ts = safe_float(scoped_intents[-1].get("timestamp"), 0.0)
    if last_ts > 0:
        cycle_intents = [
            i for i in scoped_intents if abs(safe_float(i.get("timestamp"), 0.0) - last_ts) <= 90.0
        ]
    else:
        cycle_intents = [scoped_intents[-1]]
    out: list[dict[str, Any]] = []
    for i in cycle_intents:
        det = _resolve_intent_execution_details(
            i, now_ts, extract_notional_fn, recent_orders=recent_orders, open_orders=open_orders
        )
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
        if det["order_id"] is not None:
            rec["order_id"] = det["order_id"]
        if det["ts_val"] > 0 and now_ts >= det["ts_val"]:
            rec["age_min"] = round((now_ts - det["ts_val"]) / 60.0, 1)
        out.append(rec)
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
    from .base import normalize_side

    for k in ("decision", "action", "side"):
        if data.get(k) is not None:
            return normalize_side(data.get(k))
    return ""


def extract_intent_notional(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive suggested notional USD across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    raw_notional = data.get("suggested_notional_usd")
    if raw_notional is None:
        for nk in ("notional_usd", "amount_usd", "size_usd", "usd_notional", "notional", "value_usd"):
            if data.get(nk) is not None:
                raw_notional = data.get(nk)
                break
    return _parse_currency_pos_float(raw_notional, default)


def extract_intent_limit_price(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive passive limit price across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    raw_px = data.get("limit_price")
    if raw_px is None:
        for pk in ("target_price", "order_price", "limit_px"):
            if data.get(pk) is not None:
                raw_px = data.get(pk)
                break
        if raw_px is None and str(data.get("order_type") or data.get("type") or "").strip().upper() == "LIMIT":
            raw_px = data.get("price")
    return _parse_currency_pos_float(raw_px, default)


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
    from intelligence.macro_sentiment_ssot import clean_scale

    return clean_scale(raw_conf, default, allow_percent=True)


def normalize_trading_decisions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and deduplicate single or multi-coin trading decisions from Gemini's JSON plan."""
    if not isinstance(plan, dict):
        return []
    from cachepaths import canonical_base_asset, normalize_venue_name
    from .base import normalize_side

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
    seen_keys: set[tuple[str, str, str]] = set()
    for item in candidates:
        sym_raw = str(item.get("symbol") or "").strip().upper()
        sym_canon = canonical_base_asset(sym_raw) or sym_raw
        venue_norm = normalize_venue_name(item.get("venue"))
        side_norm = str(item.get("decision") or "HOLD").strip().upper()
        key = (
            (venue_norm, sym_canon, "")
            if sym_canon
            else (venue_norm, "", side_norm)
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
    existing_ids = {str(i.get("intent_id") or i.get("id")) for i in existing_intents if isinstance(i, dict)}
    intent_id = base_id
    suffix = len(existing_intents) + 1
    while intent_id in existing_ids:
        intent_id = f"{base_id}_{suffix}"
        suffix += 1
    return intent_id


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
    if log:
        log.info(
            "[LLM TRADER · SHADOW] Would execute %s %s (qty=%g, price=$%.2f, notional=$%.2f) on %s. Thesis: %s",
            ord_lbl, inst.symbol, target_qty, price, effective_usd, inst.provider_name, thesis,
        )
    title = f"🟢 [{trade_tag.strip('[]')} · SHADOW] {ord_lbl} {inst.symbol} ${effective_usd:.0f}"
    body = (
        f"Venue: {inst.provider_name} · Qty: {target_qty:g} @ ${price:,.2f} ({'LIMIT' if is_limit else 'MARKET'})\n"
        f"Confidence: {confidence:.2f} · Notional: ${effective_usd:.2f}\n"
        f"Operational Preflight: {'PASSED' if guard_allowed else guard_reason}\n"
        f"Thesis: {thesis}"
    )
    if callable(dispatch_notify_fn):
        dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

    intent["status"] = "OBSERVED_SHADOW"
    hist_entry: dict[str, Any] = {
        "ts": now_ts,
        "action": "OBSERVED_SHADOW",
        "guard_allowed": guard_allowed,
        "guard_reason": guard_reason,
        "price": price,
        "qty": target_qty,
        "notional_usd": effective_usd,
    }
    if is_limit:
        hist_entry["order_type"] = "LIMIT"
        hist_entry["limit_price"] = price
    append_intent_history(intent, hist_entry)
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
    if not guard_allowed:
        if log:
            log.warning(
                "[LLM TRADER · ENFORCE] Operational guard blocked %s %s: %s",
                ord_lbl, inst.symbol, guard_reason,
            )
        title = f"🛡 {trade_tag} BLOCKED {ord_lbl} {inst.symbol}"
        body = (
            f"Venue: {inst.provider_name} · Price: ${price:,.2f} · Notional: ${effective_usd:.0f}\n"
            f"Reason: {guard_reason}\n"
            f"Thesis: {thesis}"
        )
        if callable(dispatch_notify_fn):
            dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

        intent["status"] = "BLOCKED_BY_GUARD"
        append_intent_history(intent, {
            "ts": now_ts,
            "action": "BLOCKED_BY_GUARD",
            "guard_reason": guard_reason,
            "price": price,
            "qty": target_qty,
            "notional_usd": effective_usd,
        })
        return intent

    if log:
        log.warning(
            "[LLM TRADER · ENFORCE] Executing %s %s (qty=%g, price=$%.2f, notional=$%.2f) on %s...",
            ord_lbl, inst.symbol, target_qty, price, effective_usd, inst.provider_name,
        )

    try:
        intent_cid = f"AI_{intent_id[:28]}" if intent_id else None
        outcome_ctx: dict[str, Any] = {}
        order_res = None
        for attempt in range(2):
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
            if (
                order_res
                or attempt > 0
                or (
                    outcome_ctx.get("state") != "refused"
                    and outcome_ctx.get("category") != "refused_pre_submit"
                )
                or outcome_ctx.get("reason") not in ("account_cache_not_fresh", "account_cache_snapshot_changed")
            ):
                break
            time.sleep(0.25)
        if order_res:
            from providers.base import extract_order_id, normalize_side

            if log:
                log.info(
                    "[LLM TRADER · ENFORCE] Successfully placed %s %s: %s",
                    ord_lbl, inst.symbol, order_res,
                )
            act_verb = "PLACED LIMIT" if is_limit else "EXECUTED"
            title = f"🚀 {trade_tag} {act_verb} {decision} {inst.symbol} ${effective_usd:.0f}"
            order_id_str = extract_order_id(order_res, default="N/A")
            body = (
                f"Venue: {inst.provider_name} · Qty: {target_qty:g} @ ${price:,.2f} "
                f"({'LIMIT' if is_limit else 'MARKET'} · ${effective_usd:,.0f})\n"
                f"Order ID: {order_id_str}\n"
                f"Thesis: {thesis}"
            )
            if callable(dispatch_notify_fn):
                dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

            intent["status"] = "EXECUTED"
            intent["execution_payload"] = order_res
            hist_exec: dict[str, Any] = {
                "ts": now_ts,
                "action": "EXECUTED",
                "price": price,
                "qty": round(target_qty, 8),
                "notional_usd": effective_usd,
                "payload": order_res,
            }
            if is_limit:
                hist_exec["order_type"] = "LIMIT"
                hist_exec["limit_price"] = price
            append_intent_history(intent, hist_exec)
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

            intent["status"] = "FAILED"
            hist_entry: dict[str, Any] = {
                "ts": now_ts,
                "action": "FAILED",
                "reason": refusal_reason,
            }
            if refusal_cat:
                hist_entry["category"] = refusal_cat
            append_intent_history(intent, hist_entry)
    except Exception as e:
        if log:
            log.error("[LLM TRADER · ENFORCE] Exception executing %s %s: %s", decision, inst.symbol, e)
        title = f"🚨 {trade_tag} ERROR {decision} {inst.symbol}"
        body = f"Execution exception: {e}\nThesis: {thesis}"
        if callable(dispatch_notify_fn):
            dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

        intent["status"] = "FAILED"
        append_intent_history(intent, {
            "ts": now_ts,
            "action": "ERROR",
            "error": str(e),
        })

    return intent


def normalize_intent_fields(
    intent: dict[str, Any],
    now_ts: float,
) -> tuple[str, str, str, str]:
    """Extract and normalize (intent_id, decision, symbol, venue) from an intent dict."""
    from cachepaths import normalize_venue_name

    raw_iid = intent.get("intent_id")
    if raw_iid is None or isinstance(raw_iid, bool) or not str(raw_iid).strip():
        raw_iid = intent.get("id")
    if raw_iid is None or isinstance(raw_iid, bool) or not str(raw_iid).strip():
        raw_iid = f"intent_{int(now_ts)}"
    intent_id = str(raw_iid).strip()
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
    from cachepaths import normalize_venue_name

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
    notional = extract_intent_notional(decision, max_notional_usd)
    notional = max(min_notional_usd, min(notional, max_notional_usd))
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


def collect_known_ai_order_ids(
    intents: Any,
    initial_ids: Optional[set[str]] = None,
) -> set[str]:
    """Collect all venue order IDs recorded in ``execution_payload`` and ``history`` across intent records."""
    from .base import extract_order_id

    ai_oids: set[str] = set(initial_ids or ())
    if not isinstance(intents, list):
        return ai_oids
    for item in intents:
        if not isinstance(item, dict):
            continue
        oid_rec = extract_order_id(item.get("execution_payload"), default="")
        if oid_rec:
            ai_oids.add(str(oid_rec))
        for h_entry in (item.get("history") if isinstance(item.get("history"), list) else ()):
            if isinstance(h_entry, dict):
                h_oid = extract_order_id(h_entry.get("order_id") or h_entry.get("payload"), default="")
                if h_oid:
                    ai_oids.add(str(h_oid))
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
) -> list[str]:
    """Cancel prior unfilled AI-placed (or bot-placed when ``include_bot_orders=True``) open orders on the symbol."""
    canceled_ids: list[str] = []
    open_fn = getattr(inst, "open_orders", None)
    cancel_fn = getattr(inst, "cancel_order", None)
    if not callable(open_fn) or not callable(cancel_fn):
        return canceled_ids
    try:
        raw_open = open_fn()
        if not isinstance(raw_open, list):
            return canceled_ids
        from providers.base import extract_order_id, extract_order_side, normalize_side, resolve_order_actor

        ai_oids = collect_known_ai_order_ids(intents, known_ai_order_ids)
        for oo in raw_open:
            if not isinstance(oo, dict):
                continue
            oo_side = extract_order_side(oo) or normalize_side(oo.get("side") or oo.get("type"))
            if not any_side and oo_side != side:
                continue
            oid = extract_order_id(oo, default="")
            is_ai_order = resolve_order_actor(oo, ai_oids) == "AI"
            if not is_ai_order and not include_bot_orders:
                continue
            if oid:
                try:
                    cancel_fn(str(oid))
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


def parse_intent_gateway_config(
    conf: dict[str, str],
    *,
    mode: Optional[str] = None,
    max_notional_usd: Optional[float] = None,
    min_confidence: Optional[float] = None,
    notify_enabled: Optional[bool] = None,
) -> tuple[str, float, float, float, bool]:
    """Validate and parse (mode, min_notional_usd, max_notional_usd, min_confidence, notify_enabled) from config."""
    from botcore import conf_or_override, parse_notify_enabled_flag

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
                    "[INTENT GATEWAY] Intent %s limit_price %r missing or out of [0.5x, 2.0x] bounds vs live $%.4g; "
                    "using passive fallback $%.4g for %s",
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
    from state_io import load_json_state

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
    from state_io import save_json_state

    try:
        cleaned = clean_json_value(intents if isinstance(intents, list) else [])
        save_json_state(intents_file, cleaned, indent=2)
        return True
    except Exception as e:
        if log:
            log.error("Failed saving intents to %s: %s", intents_file, e)
        return False


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
    from cachepaths import normalize_venue_name

    intent_id, decision, symbol, venue = normalize_intent_fields(intent, now_ts)
    if decision not in ("BUY", "SELL"):
        intent["status"] = "REJECTED_INVALID_DECISION"
        append_intent_history(intent, {
            "ts": now_ts, "action": "REJECTED_INVALID_DECISION", "decision": decision,
        })
        return False, intent_id, decision, symbol, venue, 0.0, None, 0.0, 0.0, False

    confidence = parse_intent_confidence(intent.get("confidence"), 0.0)
    intent["confidence"] = confidence
    if confidence < min_confidence:
        if log:
            log.info(
                "[INTENT GATEWAY] Intent %s rejected: confidence %.2f < %.2f",
                intent_id, confidence, min_confidence,
            )
        intent["status"] = "REJECTED_LOW_CONFIDENCE"
        append_intent_history(intent, {
            "ts": now_ts,
            "action": "REJECTED_LOW_CONFIDENCE",
            "confidence": confidence,
            "min_confidence": min_confidence,
        })
        return False, intent_id, decision, symbol, venue, confidence, None, 0.0, 0.0, False

    inst = resolve_instrument_fn(symbol, venue=venue, instruments_map=instruments_map)
    if not inst or (
        normalize_venue_name(str(getattr(inst, "provider_name", ""))) != "t212"
        and str(getattr(inst, "symbol", "")).upper().endswith("BTC")
    ):
        if log:
            log.warning(
                "[INTENT GATEWAY] Intent %s rejected: unknown or non-fiat instrument %s on %s",
                intent_id, symbol, venue,
            )
        intent["status"] = "REJECTED_UNKNOWN_INSTRUMENT"
        append_intent_history(intent, {
            "ts": now_ts, "action": "REJECTED_UNKNOWN_INSTRUMENT", "symbol": symbol, "venue": venue,
        })
        return False, intent_id, decision, symbol, venue, confidence, None, 0.0, 0.0, False

    suggested_usd = extract_intent_notional(intent, max_notional_usd)
    effective_usd = max(min_notional_usd, min(suggested_usd, max_notional_usd))

    live_price = safe_pos_float(get_live_price_fn(inst), default=None)
    if live_price is None:
        if log:
            log.warning("[INTENT GATEWAY] Intent %s rejected: invalid price for %s", intent_id, symbol)
        intent["status"] = "REJECTED_PRICE_UNAVAILABLE"
        append_intent_history(intent, {
            "ts": now_ts, "action": "REJECTED_PRICE_UNAVAILABLE", "symbol": symbol,
        })
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






