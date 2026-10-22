"""Provider primitives with no registry or venue-module dependencies."""

import math
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional


def env_value(folder: str, key: str) -> Optional[str]:
    """Read one key from ``.env``/``config.env`` without mutating ``os.environ``."""
    for fname in (".env", "config.env"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    if name.strip() == key:
                        return value.strip().strip('"').strip("'") or None
        except OSError:
            continue
    return None


def bool_env(key: str, folder: Optional[str] = None, default: bool = False) -> bool:
    """Resolve a boolean flag from os.environ or optional provider .env/config.env."""
    v = os.environ.get(key)
    if v is None and folder is not None:
        v = env_value(folder, key)
    if v is None:
        return default
    return v.strip().lower() == "true"


_TIMESTAMP_KEYS = ("timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime")
_ORDER_ID_KEYS = ("orderId", "order_id", "id", "txid", "oid", "i")
_PRICE_KEYS = ("price", "px", "limitPx", "fillPrice", "limit", "limitPrice", "averagePrice", "avgPrice", "rate", "p")
_QTY_KEYS = (
    "qty", "quantity", "executedQty", "filledQuantity", "filledQty", "vol_exec",
    "sz", "vol", "origQty", "origSz", "orig_qty", "orderedQuantity", "amount_shares", "q",
)
_ORIG_QTY_KEYS = (
    "origQty", "orig_qty", "origSz", "vol", "orderedQuantity", "original_qty",
    "order_qty", "qty", "amount", "quantity", "sz", "amount_shares", "q",
)
_EXEC_QTY_KEYS = ("executedQty", "filledQuantity", "filledQty", "vol_exec", "filled_qty")
_FEE_KEYS = ("commission", "fee", "filled_fee", "n")
_FEE_ASSET_KEYS = ("commissionAsset", "feeToken", "fee_asset", "feeAsset", "N")


def normalize_timestamp_ms(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Normalize a raw timestamp (in seconds or milliseconds) to integer milliseconds.

    Epoch values in ``[1_000_000_000, 100_000_000_000)`` (all Unix seconds between
    2001 and 2286) and non-integer float seconds are converted to milliseconds.
    Millisecond values (``>= 100_000_000_000``) and small synthetic test integers
    (``< 1_000_000_000``) are preserved as-is.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric):
        return default
    if numeric == 0.0:
        return 0
    if 1_000_000_000.0 <= numeric < 100_000_000_000.0:
        return int(round(numeric * 1000.0))
    if 0.0 < numeric < 1_000_000_000.0 and isinstance(value, float) and not numeric.is_integer():
        return int(round(numeric * 1000.0))
    return int(round(numeric))


def extract_timestamp_ms(order: Any, default: Optional[int] = None) -> Optional[int]:
    """Extract and normalize an order/fill timestamp in milliseconds from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _TIMESTAMP_KEYS:
        if key in order and order[key] is not None:
            normalized = normalize_timestamp_ms(order[key], default=None)
            if normalized is not None:
                return normalized
    return default


def extract_timestamp_s(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract and normalize an order/fill timestamp in seconds from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _TIMESTAMP_KEYS:
        if key in order and order[key] is not None:
            raw = order[key]
            if isinstance(raw, bool):
                continue
            try:
                numeric = float(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(numeric):
                continue
            if numeric >= 100_000_000_000.0:
                return numeric / 1000.0
            return numeric
    return default


def extract_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize a venue order ID from any response shape.

    Handles ``SubmissionOutcome`` objects, venue dicts (``orderId``, ``order_id``,
    ``id``, ``txid``, ``oid``, ``i``), Kraken ``txid`` lists/tuples, and scalar IDs.
    """
    if order is None or isinstance(order, bool):
        return default
    if hasattr(order, "order_id") and hasattr(order, "state"):
        oid = getattr(order, "order_id", None)
        if oid is not None and not isinstance(oid, bool) and str(oid).strip():
            return str(oid).strip()
        return default
    order_id = None
    if isinstance(order, dict):
        for key in _ORDER_ID_KEYS:
            if key in order and order[key] is not None and not isinstance(order[key], bool):
                order_id = order[key]
                break
    else:
        order_id = order
    if isinstance(order_id, (list, tuple)):
        order_id = next(
            (
                item for item in order_id
                if item is not None and not isinstance(item, bool) and str(item).strip()
            ),
            None,
        )
    if order_id is None or isinstance(order_id, bool) or not str(order_id).strip():
        return default
    return str(order_id).strip()


def extract_order_side(order: Any) -> str:
    """Extract and normalize order side (``'BUY'`` or ``'SELL'``) from any venue shape."""
    if not isinstance(order, dict):
        return ""
    raw_side = order.get("side")
    if raw_side is not None and str(raw_side).strip():
        s = str(raw_side).strip().upper()
        if s == "B":
            return "BUY"
        if s in ("A", "S"):
            return "SELL"
        return s
    raw_type = order.get("type")
    if raw_type is not None and str(raw_type).strip().upper() in ("BUY", "SELL"):
        return str(raw_type).strip().upper()
    if "isBuyer" in order and order.get("isBuyer") is not None:
        return "BUY" if bool(order["isBuyer"]) else "SELL"
    signed_qty = order.get("signed_qty")
    if signed_qty is not None and not isinstance(signed_qty, bool):
        try:
            sq = float(signed_qty)
            if math.isfinite(sq) and sq != 0.0:
                return "BUY" if sq > 0 else "SELL"
        except (TypeError, ValueError, OverflowError):
            pass
    return ""


def extract_order_price(order: Any, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _PRICE_KEYS:
        if key in order and order[key] is not None:
            try:
                val = float(order[key] or 0.0)
                if math.isfinite(val) and val > 0:
                    return val
            except (TypeError, ValueError, OverflowError):
                continue
    for cost_key, exec_key in (
        ("cummulativeQuoteQty", "executedQty"),
        ("quoteQty", "qty"),
        ("quoteQty", "quantity"),
        ("cost", "vol_exec"),
        ("filledValue", "filledQuantity"),
    ):
        if cost_key in order and exec_key in order:
            try:
                cost_val = abs(float(order[cost_key] or 0.0))
                exec_val = abs(float(order[exec_key] or 0.0))
                if math.isfinite(cost_val) and math.isfinite(exec_val) and cost_val > 0 and exec_val > 0:
                    return cost_val / exec_val
            except (TypeError, ValueError, OverflowError):
                continue
    fills = order.get("fills")
    if isinstance(fills, list) and fills:
        total_quote = 0.0
        total_qty = 0.0
        for f in fills:
            if not isinstance(f, dict):
                continue
            try:
                f_px = float(f.get("price", f.get("px", 0.0)) or 0.0)
                f_qty = abs(float(f.get("qty", f.get("quantity", f.get("sz", 0.0))) or 0.0))
                if math.isfinite(f_px) and math.isfinite(f_qty) and f_px > 0 and f_qty > 0:
                    total_quote += f_px * f_qty
                    total_qty += f_qty
            except (TypeError, ValueError, OverflowError):
                continue
        if total_qty > 0:
            return total_quote / total_qty
    return default


def extract_order_qty(order: Any, default: float = 0.0) -> float:
    """Extract order/fill quantity from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _QTY_KEYS:
        if key in order and order[key] is not None:
            try:
                val = abs(float(order[key] or 0.0))
                if math.isfinite(val) and val > 0:
                    return val
            except (TypeError, ValueError, OverflowError):
                continue
    return default


def extract_orig_qty(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract the venue-accepted original order quantity from any response shape.

    Prioritizes original-quantity fields (``origQty``, ``origSz``, ``vol``,
    ``orderedQuantity``, etc.) over executed-quantity fields so recovering a
    ``PARTIALLY_FILLED`` order never overwrites the total order quantity with
    the partial fill quantity.
    """
    if order is None or isinstance(order, bool):
        return default
    if hasattr(order, "native") and hasattr(order, "state"):
        order = getattr(order, "native", None)
        if order is None or isinstance(order, bool):
            return default
    if isinstance(order, dict):
        for key in (*_ORIG_QTY_KEYS, *_EXEC_QTY_KEYS):
            if key in order and order[key] is not None and not isinstance(order[key], bool):
                try:
                    val = abs(float(order[key]))
                    if math.isfinite(val) and val > 0:
                        return val
                except (TypeError, ValueError, OverflowError):
                    continue
    else:
        for attr in (*_ORIG_QTY_KEYS, *_EXEC_QTY_KEYS):
            if hasattr(order, attr):
                raw = getattr(order, attr)
                if raw is not None and not isinstance(raw, bool):
                    try:
                        val = abs(float(raw))
                        if math.isfinite(val) and val > 0:
                            return val
                    except (TypeError, ValueError, OverflowError):
                        continue
    return default


def extract_remaining_qty(order: Any, default: float = 0.0) -> float:
    """Extract remaining unfilled quantity for an open/cancelled order across any venue shape."""
    if not isinstance(order, dict):
        return default
    rem = order.get("remainingQty")
    if rem is not None and not isinstance(rem, bool):
        try:
            rem_f = abs(float(rem))
            if math.isfinite(rem_f) and rem_f > 0:
                return rem_f
        except (TypeError, ValueError, OverflowError):
            pass
    orig = 0.0
    for key in _ORIG_QTY_KEYS:
        if key in order and order[key] is not None:
            try:
                val = abs(float(order[key] or 0.0))
                if math.isfinite(val) and val > 0:
                    orig = val
                    break
            except (TypeError, ValueError, OverflowError):
                continue
    exec_q = 0.0
    for key in _EXEC_QTY_KEYS:
        if key in order and order[key] is not None:
            try:
                val = abs(float(order[key] or 0.0))
                if math.isfinite(val) and val > 0:
                    exec_q = val
                    break
            except (TypeError, ValueError, OverflowError):
                continue
    if orig > 0:
        return max(0.0, orig - exec_q) if exec_q > 0 else orig
    return default


def extract_order_fee(
    order: Any,
    base_asset: str = "",
    quote_asset: str = "",
    fill_price: Optional[float] = None,
    require_explicit_asset: bool = False,
) -> Optional[dict]:
    """Extract and normalize order/fill fee metadata across any venue shape.

    Returns ``None`` when no fee field is present or when fee metadata is invalid,
    otherwise returns ``{"fee": float, "fee_asset": str, "base_fee": float,
    "quote_fee": float, "fee_in_quote": Optional[float]}``.
    """
    if not isinstance(order, dict):
        return None
    raw_fee = None
    has_fee_key = False
    for key in _FEE_KEYS:
        if key in order and order[key] is not None:
            raw_fee = order[key]
            has_fee_key = True
            break
    if not has_fee_key or isinstance(raw_fee, bool):
        return None
    try:
        fee = float(raw_fee)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(fee) or fee < 0.0:
        return None

    fee_asset = ""
    has_asset_key = False
    for key in _FEE_ASSET_KEYS:
        if key in order:
            has_asset_key = True
            val = order[key]
            if val is not None and not isinstance(val, bool):
                fee_asset = str(val).strip().upper()
            break

    if require_explicit_asset and (not has_asset_key or (fee > 0.0 and not fee_asset)):
        return None

    base_u = str(base_asset or "").strip().upper()
    quote_u = str(quote_asset or "").strip().upper()
    if not fee_asset and not require_explicit_asset and quote_u:
        fee_asset = quote_u

    px = (
        float(fill_price)
        if (fill_price is not None and math.isfinite(float(fill_price)) and float(fill_price) > 0.0)
        else extract_order_price(order, default=0.0)
    )
    base_fee = 0.0
    quote_fee = 0.0
    fee_in_quote: Optional[float] = 0.0 if fee == 0.0 else None

    if fee > 0.0:
        if base_u and fee_asset == base_u:
            base_fee = fee
            if px > 0.0:
                fee_in_quote = fee * px
        elif (quote_u and fee_asset == quote_u) or (not fee_asset and not require_explicit_asset):
            quote_fee = fee
            fee_in_quote = fee

    return {
        "fee": fee,
        "fee_asset": fee_asset,
        "base_fee": base_fee,
        "quote_fee": quote_fee,
        "fee_in_quote": fee_in_quote,
    }


def normalize_order(order: dict) -> dict:
    """Normalize a native order/fill to canonical ``side``, ``price``, ``qty``, and ``timestamp``."""
    return {
        "side": extract_order_side(order),
        "price": extract_order_price(order),
        "qty": extract_order_qty(order),
        "timestamp": extract_timestamp_ms(order),
    }


# Compatibility alias retained for existing internal and external imports.
_normalize_order = normalize_order

from .strategy_executor import (  # noqa: E402
    OrderReconciliationCapabilities,
    PairPrecision,
    ProviderError,
)


class MarketDataProvider(ABC):
    """Shared market-data, account, order, and policy-hook interface."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        ...

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        ...

    def validate_symbol(self, symbol: str) -> None:
        """Reject a symbol incompatible with this configured provider instance."""
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def free_balance(self, asset: str) -> Optional[float]:
        """Return free balance: ``0.0`` is real zero; ``None`` means unavailable."""
        return None

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        return []

    def get_trades(self, symbol: str, since_s: float) -> List[dict]:
        return self.get_orders(symbol, None, since_s)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost, or unknown when unsupported."""
        return None

    def open_orders(self, symbol: str) -> List[dict]:
        return []

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        """Return explicitly supported strict lifecycle operations.

        The default is deliberately conservative. An inherited empty
        ``open_orders`` implementation must never be mistaken for a confirmed
        empty venue snapshot.
        """
        return OrderReconciliationCapabilities()

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Validate venue state and optionally return opaque submit authorization."""
        return None

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now.

        Providers with an explicit dry/live switch override this hook. The shared
        placement pipeline checks it before creating durable retry state so a dry
        validation cannot become a live order after a later configuration change.
        """
        return True

    def prepare_order_state(self) -> Any:
        """Synchronize provider state used by shared pre-submit policy checks."""
        return None

    def validate_order_state(self, expected_state: Any) -> Any:
        """Require that policy checks still describe the prepared provider state."""
        return expected_state

    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        return None

    def cancel_order(self, symbol: str, order_id: str) -> None:
        return None

    def guards_internally(self) -> bool:
        return False

    def requires_cache_permit(self) -> bool:
        """Return whether this provider enforces account-cache submit permits."""
        return str(getattr(self, "name", "") or "").casefold() == "binance"

    def defers_smart_cancellation(self) -> bool:
        """Return whether opposing order cancellation is deferred until dispatch."""
        return str(getattr(self, "name", "") or "").casefold() == "binance"

    def supports_business_minimum(self) -> bool:
        """Return whether this provider enforces venue business minimum rules."""
        return str(getattr(self, "name", "") or "").casefold() == "binance"

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        """Return normalized pair precision and limits, or None if unavailable."""
        return None

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round or floor quantity to venue volume precision."""
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None and getattr(pp, "volume_decimals", None) is not None:
                dec = int(pp.volume_decimals)
                if dec >= 0:
                    factor = 10 ** dec
                    return math.floor(float(qty) * factor + 1e-12) / factor
        except Exception:
            pass
        return float(qty)

    def round_amount(self, symbol: str, qty: float) -> float:
        """Backward-compatible alias for round_quantity."""
        return self.round_quantity(symbol, qty)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None and getattr(pp, "price_decimals", None) is not None:
                dec = int(pp.price_decimals)
                if dec >= 0:
                    return round(float(price), dec)
        except Exception:
            pass
        return float(price)

    def min_order_qty(self, symbol: str) -> float:
        """Return venue minimum order quantity; 0.0 means no known minimum."""
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None:
                return float(getattr(pp, "order_min", 0.0) or 0.0)
        except Exception:
            pass
        return 0.0

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Return a provider-specific pre-submit refusal, if one is known."""
        try:
            qty_f = float(qty)
            if not math.isfinite(qty_f) or qty_f <= 0:
                return "qty_zero_or_negative"
            min_qty = self.min_order_qty(symbol)
            if min_qty > 0 and qty_f < min_qty:
                return f"filter_lot_size_min:{qty_f}<{min_qty}"
        except Exception:
            pass
        return None

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs) -> float:
        import order_guard
        return order_guard.weight_limit(
            self, symbol, side, price, qty,
            available_qty=available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        return available_qty

    def quantity_decision(self, symbol: str, side: str, price: float,
                          qty: float, **kwargs):
        from providers.quantity import decide_quantity
        return decide_quantity(self, symbol, side, price, qty, **kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float,
                           cancel_opposite: bool = True) -> float:
        return price

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        """Cancel adverse opposing orders when the venue implements this policy."""
        return None

    def profit_guard_window_ref(self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None):
        import order_guard
        window_s = order_guard.get_window_cfg(
            self.name,
            symbol=symbol,
            order_type=side,
            regime_context=regime_context,
            module=module,
            reference_mode=reference_mode,
        )
        if window_s is None or window_s <= 0:
            window_s = float(safeback_sec) if safeback_sec else 0.0
        wr_kwargs = {}
        if reference_mode is not None:
            wr_kwargs["reference_mode"] = reference_mode
        if module is not None:
            wr_kwargs["module"] = module
        return order_guard.window_reference(self, symbol, side, window_s, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        opposite = "SELL" if order_type.upper() == "BUY" else "BUY"
        fills = self.get_orders(symbol, opposite, since_s)
        if fills is None:
            raise ProviderError(f"last_opposite_fill({symbol}): failed to read order history")
        if not fills:
            return None
        latest = max(fills, key=lambda order: extract_timestamp_ms(order, default=0) or 0)
        price = extract_order_price(latest)
        return price or None

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        """Return completed-bar closing prices from closed interval series."""
        if hasattr(self, "ohlc_series"):
            return list(self.ohlc_series(symbol, interval_min).closes)
        return []
