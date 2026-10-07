"""Provider-neutral execution contract used by financially tracked strategies.

``strategies.spot_engine`` consumes this interface. Venue adapters normalize native
responses into the explicit types below so the strategy engine can operate on one
order lifecycle model.

``submit_order`` is intentionally distinct from ``MarketDataProvider.place_order``.
The former returns a venue order ID or raises ``ProviderError``; the latter is the
mechanical/legacy facade entry point and may return a native payload or ``None``.
Each adapter still applies its configured live-order gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Protocol, runtime_checkable


class ProviderError(Exception):
    """Venue-neutral error raised by strict strategy-execution adapters."""


class SubmissionRefused(ProviderError, RuntimeError):
    """The order was definitively refused before or during synchronous submit."""

    def __init__(self, reason: str):
        self.reason = str(reason or "").strip()
        if not self.reason:
            raise ValueError("submission refusal reason is required")
        super().__init__(self.reason)


@dataclass(frozen=True)
class SubmissionOutcome:
    """Typed result of one submit attempt, without implying that acceptance is a fill.

    ``unknown`` means the caller cannot prove whether the venue accepted the order.
    It must be reconciled, never blindly retried. ``refused`` proves that no accepted
    order exists. Existing adapters may still return a native ID; ``capture_submission``
    provides the compatibility boundary while they are migrated incrementally.
    """

    state: str
    order_id: Optional[str] = None
    reason: str = ""
    native: object = None

    def __post_init__(self):
        if self.state not in {"accepted", "refused", "unknown"}:
            raise ValueError(f"invalid submission outcome: {self.state!r}")
        order_id = None if self.order_id is None else str(self.order_id).strip()
        if self.state == "accepted" and not order_id:
            raise ValueError("accepted submission requires order_id")
        if self.state != "accepted" and order_id:
            raise ValueError(f"{self.state} submission cannot carry order_id")
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "reason", str(self.reason or "").strip())

    @property
    def accepted(self) -> bool:
        return self.state == "accepted"


def capture_submission(submit: Callable[[], object]) -> SubmissionOutcome:
    """Normalize the current strict-adapter contract into a typed outcome."""
    try:
        native = submit()
    except SubmissionRefused as exc:
        return SubmissionOutcome("refused", reason=exc.reason)
    except Exception as exc:  # transport/provider ambiguity stays recoverable
        return SubmissionOutcome(
            "unknown", reason=f"{exc.__class__.__name__}: {exc}")
    if isinstance(native, SubmissionOutcome):
        return native
    order_id = extract_order_id(native)
    if order_id is None:
        return SubmissionOutcome(
            "unknown", reason="response_without_order_id", native=native)
    return SubmissionOutcome("accepted", order_id=order_id, native=native)


def extract_order_id(native) -> Optional[str]:
    """Extract a venue order ID from normalized and common native response shapes."""
    if isinstance(native, SubmissionOutcome):
        return native.order_id
    order_id = None
    if isinstance(native, dict):
        order_id = native.get("orderId", native.get("order_id", native.get("id")))
        if order_id is None:
            order_id = native.get("txid")
    elif not isinstance(native, bool):
        order_id = native
    if isinstance(order_id, (list, tuple)):
        order_id = next((item for item in order_id if str(item).strip()), None)
    if order_id is None or not str(order_id).strip():
        return None
    return str(order_id)


def extract_order_qty(native) -> Optional[float]:
    """Extract the venue-accepted original order quantity from common response shapes."""
    if native is None:
        return None
    if isinstance(native, SubmissionOutcome):
        native = native.native
        if native is None:
            return None
    val = None
    if isinstance(native, dict):
        for key in (
            "origQty", "orig_qty", "origSz", "vol", "qty", "amount",
            "original_qty", "order_qty", "orderedQuantity", "quantity", "sz",
        ):
            if key in native and native[key] is not None:
                val = native[key]
                break
    else:
        for attr in (
            "orig_qty", "origQty", "origSz", "vol", "qty", "amount",
            "original_qty", "order_qty", "orderedQuantity", "quantity", "sz",
        ):
            if hasattr(native, attr):
                val = getattr(native, attr)
                if val is not None:
                    break
    if val is None:
        return None
    try:
        f = abs(float(val))
        return f if math.isfinite(f) and f > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class OrderStatus:
    """Normalized order result: ``open``, ``closed``, ``canceled``, or ``expired``."""
    status: str
    filled_qty: float          # Executed quantity (Kraken ``vol_exec``).
    cost: float                # Executed notional; average price is cost / filled_qty.
    fee: float                 # Actual fee reported by the venue.
    venue_status: str = ""     # Native terminal reason (REJECTED vs CANCELED, etc.).
    orig_qty: Optional[float] = None  # Original quantity accepted by the venue.

    def __post_init__(self):
        if self.status not in {"open", "closed", "canceled", "expired"}:
            raise ValueError(f"unnormalized order status: {self.status!r}")
        for name, raw in (("filled_qty", self.filled_qty),
                          ("cost", self.cost), ("fee", self.fee)):
            value = float(raw)
            # A negative fee is valid when the venue pays a maker rebate.
            if not math.isfinite(value) or (name != "fee" and value < 0):
                raise ValueError(f"invalid {name} in OrderStatus: {raw!r}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "venue_status", str(self.venue_status or "").upper())
        if self.orig_qty is not None:
            raw_orig = self.orig_qty
            try:
                orig_val = float(raw_orig)
                if not math.isfinite(orig_val) or orig_val < 0:
                    raise ValueError(f"invalid orig_qty in OrderStatus: {raw_orig!r}")
                object.__setattr__(self, "orig_qty", orig_val)
            except (TypeError, OverflowError):
                raise ValueError(f"invalid orig_qty in OrderStatus: {raw_orig!r}")

    @property
    def terminal(self) -> bool:
        return self.status in {"closed", "canceled", "expired"}

    @property
    def fully_filled(self) -> bool:
        """The venue confirmed a complete fill, not merely order acceptance."""
        return self.status == "closed"

    @property
    def has_fill(self) -> bool:
        return self.filled_qty > 0

    @property
    def partially_filled(self) -> bool:
        return self.has_fill and not self.fully_filled


@dataclass(frozen=True)
class PairPrecision:
    """Normalized price, volume, and minimum-quantity metadata for a pair."""
    price_decimals: int
    volume_decimals: int
    order_min: float           # Minimum quantity (Kraken ``ordermin``).
    base_asset: str = ""       # Pair base asset, used when adopting an existing position.

    def __post_init__(self):
        for name in ("price_decimals", "volume_decimals"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or int(raw) != raw or int(raw) < 0:
                raise ValueError(f"invalid {name} in PairPrecision: {raw!r}")
            object.__setattr__(self, name, int(raw))
        minimum = float(self.order_min)
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError(f"invalid order_min in PairPrecision: {self.order_min!r}")
        object.__setattr__(self, "order_min", minimum)
        object.__setattr__(self, "base_asset", str(self.base_asset or "").strip())


_CANDLE_INTERVALS = {
    1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h", 1440: "1d",
}


def candle_interval(interval_min: int) -> str:
    """Return the canonical venue interval or reject an unsupported horizon."""
    try:
        minutes = int(interval_min)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderError(f"invalid candle interval: {interval_min!r}") from exc
    if minutes != interval_min or minutes not in _CANDLE_INTERVALS:
        raise ProviderError(f"unsupported candle interval: {interval_min!r} minutes")
    return _CANDLE_INTERVALS[minutes]


@dataclass(frozen=True)
class OrderReconciliationCapabilities:
    """Venue facts needed by the common order-lifecycle machinery.

    A capability means the adapter exposes a strict, normalized operation. It does
    not promise that the venue is currently reachable. Missing declarations fail
    closed through the conservative all-false default on ``MarketDataProvider``.
    """

    lookup_by_client_order_id: bool = False
    status_by_order_id: bool = False
    cancel_by_order_id: bool = False
    list_open_orders: bool = False
    not_found_reliable_for_seconds: Optional[float] = None

    def __post_init__(self):
        for field_name in (
                "lookup_by_client_order_id", "status_by_order_id",
                "cancel_by_order_id", "list_open_orders"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} capability must be bool")
        horizon = self.not_found_reliable_for_seconds
        if horizon is None:
            return
        if isinstance(horizon, bool):
            raise TypeError(
                "not_found_reliable_for_seconds must be a positive number")
        try:
            horizon = float(horizon)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "not_found_reliable_for_seconds must be a positive number"
            ) from exc
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError(
                "not_found_reliable_for_seconds must be finite and > 0")
        if not self.lookup_by_client_order_id:
            raise ValueError(
                "a NOT_FOUND reliability horizon requires client-ID lookup")
        object.__setattr__(self, "not_found_reliable_for_seconds", horizon)


def reconciliation_capabilities_of(provider) -> OrderReconciliationCapabilities:
    """Read and validate an adapter's explicit reconciliation declaration."""
    name = str(getattr(provider, "name", type(provider).__name__))
    method = getattr(provider, "reconciliation_capabilities", None)
    if not callable(method):
        raise ProviderError(f"{name}: reconciliation capabilities are undeclared")
    capabilities = method()
    if not isinstance(capabilities, OrderReconciliationCapabilities):
        raise ProviderError(f"{name}: invalid reconciliation capabilities")
    return capabilities


@runtime_checkable
class StrategyExecutor(Protocol):
    """Minimum venue-neutral interface required by the spot-DCA engine."""

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return the current last/mid price, or ``None`` when unavailable."""
        ...

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        """Submit an order and return its venue ID.

        ``price=None`` or ``market=True`` requests a market order.
        ``client_order_id`` carries the persisted intent when the venue supports it.
        Raise ``ProviderError`` when submission cannot be confirmed.
        """
        ...

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        """Return normalized status and cumulative fills, or raise ``ProviderError``."""
        ...

    def cancel_order(self, symbol: str, order_id: str) -> None:
        """Request cancellation by venue ID or raise ``ProviderError``.

        Adapters are not uniformly idempotent for already-terminal or unknown orders;
        callers must reconcile status when cancellation is ambiguous.
        """
        ...

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        """Return pair precision and minimum quantity, or ``None`` if unavailable."""
        ...

    def free_balance(self, asset: str) -> Optional[float]:
        """Return free quantity: ``0.0`` is real zero; ``None`` means unavailable."""
        ...

    def ohlc_closes(self, symbol: str, interval_min: int) -> list[float]:
        """Return completed-bar closes for ``interval_min``; empty means unavailable."""
        ...
