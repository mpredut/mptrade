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
    float_env,
    safe_float,
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
        if isinstance(order_id, (bool, list, tuple, dict, set, frozenset)):
            return ""
        from .base import _clean_single_order_id
        cleaned = _clean_single_order_id(order_id)
        return cleaned if cleaned is not None else ""

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
        from .base import _clean_single_order_id
        clean_exp = (
            _clean_single_order_id(explicit)
            if not isinstance(explicit, (bool, list, tuple, dict, set, frozenset))
            else None
        )
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
        from .base import _clean_single_order_id
        clean_cid = (
            _clean_single_order_id(client_order_id)
            if not isinstance(client_order_id, (bool, list, tuple, dict, set, frozenset))
            else None
        )
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
        itv_f = safe_pos_float(interval_min, default=None)
        if itv_f is None or not itv_f.is_integer():
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
        interval_i = int(itv_f)
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


def summarize_recent_ai_intents(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
) -> list[dict[str, Any]]:
    """Summarize the most recent AI trade intents filtered by asset scope."""
    from cachepaths import normalize_venue_name

    scoped_intents = [
        i
        for i in intents_data
        if isinstance(i, dict)
        and (
            scope not in ("equities", "crypto")
            or (normalize_venue_name(i.get("venue")) == "t212") == (scope == "equities")
        )
    ]
    recent_intents_out: list[dict[str, Any]] = []
    for i in scoped_intents[-8:]:
        exec_pay = i.get("execution_payload") if isinstance(i.get("execution_payload"), dict) else {}
        hist = i.get("history") if isinstance(i.get("history"), list) else []
        last_hist = hist[-1] if hist and isinstance(hist[-1], dict) else {}
        eff_notional = (
            safe_pos_float(exec_pay.get("cummulativeQuoteQty"), default=None)
            or safe_pos_float(exec_pay.get("cost"), default=None)
            or safe_pos_float(i.get("clamped_notional_usd"), default=None)
            or safe_pos_float(last_hist.get("notional_usd"), default=None)
            or extract_notional_fn(i, default=None)
        )
        ts_val = safe_float(i.get("timestamp"), 0.0)
        intent_summary: dict[str, Any] = {
            "decision": i.get("decision"),
            "symbol": i.get("symbol"),
            "venue": normalize_venue_name(i.get("venue")) or i.get("venue"),
            "status": i.get("status"),
            "notional_usd": round(eff_notional, 2) if eff_notional is not None else i.get("suggested_notional_usd"),
            "ts": i.get("timestamp"),
        }
        if ts_val > 0 and now_ts >= ts_val:
            intent_summary["age_h"] = round((now_ts - ts_val) / 3600.0, 2)
        recent_intents_out.append(intent_summary)
    return recent_intents_out


_CURRENCY_STRIP_RE = re.compile(r"[\$,\s]|(?:USD[CT]?)\b", re.IGNORECASE)


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
    if isinstance(raw_notional, str):
        raw_notional = _CURRENCY_STRIP_RE.sub("", raw_notional).strip()
    return safe_pos_float(raw_notional, default)


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


