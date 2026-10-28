"""Provider primitives with no registry or venue-module dependencies."""

import math
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from botcore import _parse_bool_flag, parse_dotenv


def env_value(folder: str, key: str) -> Optional[str]:
    """Read one key from ``.env``/``config.env``/``runtime.env`` without mutating ``os.environ``."""
    if not isinstance(folder, str) or not folder.strip() or not isinstance(key, str) or not key.strip():
        return None
    for fname in (".env", "config.env", "runtime.env"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            parsed = parse_dotenv(path)
            if key in parsed:
                val = str(parsed[key]).strip()
                return val or None
        except Exception:
            continue
    return None


def bool_env(key: str, folder: Optional[str] = None, default: bool = False) -> bool:
    """Resolve a boolean flag from os.environ or optional provider .env/config.env."""
    def_flag = _parse_bool_flag(default, False)
    if not isinstance(key, str) or not key.strip():
        return def_flag
    v = os.environ.get(key)
    if v is None and folder is not None:
        v = env_value(folder, key)
    if v is None:
        return def_flag
    return _parse_bool_flag(v, def_flag)


_TIMESTAMP_KEYS = ("timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime", "T", "E", "ts")
_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "txid", "oid", "i", "id")
_PRICE_KEYS = ("price", "px", "limitPx", "fillPrice", "limit", "limitPrice", "averagePrice", "avgPrice", "rate", "L", "p")
_QTY_KEYS = (
    "qty", "quantity", "executedQty", "filledQuantity", "filledQty", "vol_exec",
    "sz", "vol", "origQty", "origSz", "orig_qty", "orderedQuantity", "amount_shares", "l", "z", "q",
)
_ORIG_QTY_KEYS = (
    "origQty", "orig_qty", "origSz", "vol", "orderedQuantity", "original_qty",
    "order_qty", "qty", "amount", "quantity", "sz", "amount_shares", "q",
)
_EXEC_QTY_KEYS = ("executedQty", "filledQuantity", "filledQty", "vol_exec", "filled_qty", "z")
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
    if not math.isfinite(numeric) or numeric < 0.0:
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
            if not math.isfinite(numeric) or numeric < 0.0:
                continue
            if numeric >= 100_000_000_000.0:
                return numeric / 1000.0
            return numeric
    return default


def _clean_single_order_id(val: Any) -> Optional[str]:
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer() or val <= 0:
            return None
        val = int(val)
    if isinstance(val, int):
        if val <= 0:
            return None
    if not isinstance(val, (str, int)):
        return None
    text = str(val).strip()
    if (
        not text
        or text.startswith("-")
        or text.lower() in {"0", "0.0", "-0", "+0", "+0.0", "-0.0", "nan", "inf", "-inf", "+inf", "none", "null"}
    ):
        return None
    if text.endswith(".0") and text[:-2].lstrip("+").isdigit():
        text = text[:-2]
    if text in {"0", "+0"}:
        return None
    if text.startswith("+") and text[1:].isdigit():
        text = text[1:]
    try:
        f = float(text)
        if not math.isfinite(f) or f <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    return text


def extract_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize a venue order ID from any response shape.

    Handles ``SubmissionOutcome`` objects, venue dicts (``orderId``, ``order_id``,
    ``id``, ``txid``, ``oid``, ``i``), Kraken ``txid`` lists/tuples, and scalar IDs.
    """
    if order is None or isinstance(order, bool):
        return default
    if isinstance(order, (list, tuple)):
        for item in order:
            cleaned = extract_order_id(item)
            if cleaned is not None:
                return cleaned
        return default
    if hasattr(order, "order_id") and hasattr(order, "state"):
        cleaned = _clean_single_order_id(getattr(order, "order_id", None))
        return cleaned if cleaned is not None else default
    if isinstance(order, dict):
        for key in _ORDER_ID_KEYS:
            if key in order:
                cleaned = extract_order_id(order[key])
                if cleaned is not None:
                    return cleaned
        return default
    cleaned = _clean_single_order_id(order)
    return cleaned if cleaned is not None else default


def normalize_side(side: Any) -> str:
    """Normalize a scalar order side token to ``'BUY'`` or ``'SELL'``."""
    if side is None or isinstance(side, bool):
        return ""
    s = str(side).strip().upper()
    if s in ("B", "BUY", "BID", "LONG", "OPEN LONG", "CLOSE SHORT"):
        return "BUY"
    if s in ("A", "S", "SELL", "ASK", "SHORT", "CLOSE LONG", "OPEN SHORT"):
        return "SELL"
    return ""


def extract_order_side(order: Any) -> str:
    """Extract and normalize order side (``'BUY'`` or ``'SELL'``) from any venue shape."""
    if not isinstance(order, dict):
        return ""
    raw_side = next(
        (
            order[k]
            for k in ("side", "S", "dir")
            if k in order
            and order[k] is not None
            and not isinstance(order[k], bool)
            and str(order[k]).strip()
        ),
        None,
    )
    if raw_side is not None and str(raw_side).strip():
        return normalize_side(raw_side)
    raw_type = order.get("type")
    if raw_type is None and isinstance(order.get("descr"), dict):
        raw_type = order["descr"].get("type")
    t_norm = normalize_side(raw_type)
    if t_norm in ("BUY", "SELL"):
        return t_norm
    if "isBuyer" in order and order.get("isBuyer") is not None:
        raw_buyer = order["isBuyer"]
        parsed_b = _parse_bool_flag(raw_buyer, None)
        if parsed_b is not None:
            return "BUY" if parsed_b else "SELL"
        if isinstance(raw_buyer, str):
            b_norm = raw_buyer.strip().lower()
            if b_norm in ("b", "buy"):
                return "BUY"
            if b_norm in ("s", "sell"):
                return "SELL"
    for sq_key in ("signed_qty", "orderedQuantity", "quantity"):
        signed_qty = order.get(sq_key)
        if signed_qty is not None and not isinstance(signed_qty, bool):
            try:
                sq = float(signed_qty)
                if math.isfinite(sq) and sq != 0.0:
                    if sq < 0.0:
                        return "SELL"
                    if sq_key in ("signed_qty", "orderedQuantity") or "ticker" in order or "limitPrice" in order:
                        return "BUY"
            except (TypeError, ValueError, OverflowError):
                pass
    return ""


def extract_order_price(order: Any, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _PRICE_KEYS:
        if key in order and order[key] is not None and not isinstance(order[key], bool):
            try:
                val = float(order[key] or 0.0)
                if math.isfinite(val) and val > 0:
                    return val
            except (TypeError, ValueError, OverflowError):
                continue
    descr = order.get("descr")
    if (
        isinstance(descr, dict)
        and descr.get("price") is not None
        and not isinstance(descr.get("price"), bool)
    ):
        try:
            val = float(descr["price"] or 0.0)
            if math.isfinite(val) and val > 0:
                return val
        except (TypeError, ValueError, OverflowError):
            pass
    for cost_key, exec_key in (
        ("cummulativeQuoteQty", "executedQty"),
        ("cummulativeQuoteQty", "qty"),
        ("cummulativeQuoteQty", "quantity"),
        ("quoteQty", "qty"),
        ("quoteQty", "quantity"),
        ("cost", "vol_exec"),
        ("cost", "vol"),
        ("filledValue", "filledQuantity"),
        ("Y", "l"),
        ("Z", "z"),
    ):
        if (
            cost_key in order
            and exec_key in order
            and not isinstance(order.get(cost_key), bool)
            and not isinstance(order.get(exec_key), bool)
        ):
            try:
                cost_val = abs(float(order[cost_key] or 0.0))
                exec_val = abs(float(order[exec_key] or 0.0))
                if math.isfinite(cost_val) and math.isfinite(exec_val) and cost_val > 0 and exec_val > 0:
                    res = cost_val / exec_val
                    if math.isfinite(res) and res > 0:
                        return res
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
                raw_px = f.get("price", f.get("px", 0.0))
                raw_qty = f.get("qty", f.get("quantity", f.get("sz", 0.0)))
                if isinstance(raw_px, bool) or isinstance(raw_qty, bool):
                    continue
                f_px = float(raw_px or 0.0)
                f_qty = abs(float(raw_qty or 0.0))
                if math.isfinite(f_px) and math.isfinite(f_qty) and f_px > 0 and f_qty > 0:
                    total_quote += f_px * f_qty
                    total_qty += f_qty
            except (TypeError, ValueError, OverflowError):
                continue
        if total_qty > 0:
            res = total_quote / total_qty
            if math.isfinite(res) and res > 0:
                return res
    return default


def extract_order_qty(order: Any, default: float = 0.0) -> float:
    """Extract order/fill quantity from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _QTY_KEYS:
        if key in order and order[key] is not None and not isinstance(order[key], bool):
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
            rem_f = float(rem)
            if math.isfinite(rem_f) and rem_f >= 0.0:
                return rem_f
        except (TypeError, ValueError, OverflowError):
            pass
    if "origSz" in order and "sz" in order and not any(k in order for k in _EXEC_QTY_KEYS):
        raw_sz = order.get("sz")
        if raw_sz is not None and not isinstance(raw_sz, bool):
            try:
                sz_f = abs(float(raw_sz))
                if math.isfinite(sz_f) and sz_f >= 0.0:
                    return sz_f
            except (TypeError, ValueError, OverflowError):
                pass
    orig = 0.0
    for key in _ORIG_QTY_KEYS:
        if key in order and order[key] is not None and not isinstance(order[key], bool):
            try:
                val = abs(float(order[key] or 0.0))
                if math.isfinite(val) and val > 0:
                    orig = val
                    break
            except (TypeError, ValueError, OverflowError):
                continue
    exec_q = 0.0
    for key in _EXEC_QTY_KEYS:
        if key in order and order[key] is not None and not isinstance(order[key], bool):
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
    fee: Optional[float] = None
    for key in _FEE_KEYS:
        if key in order and order[key] is not None:
            cand_raw = order[key]
            if isinstance(cand_raw, bool):
                continue
            try:
                cand_f = float(cand_raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(cand_f) and cand_f >= 0.0:
                fee = cand_f
                break
    if fee is None:
        return None

    fee_asset = ""
    has_asset_key = False
    for key in _FEE_ASSET_KEYS:
        if key in order:
            has_asset_key = True
            val = order[key]
            if isinstance(val, str) and val.strip():
                fee_asset = val.strip().upper()
                break

    req_explicit = _parse_bool_flag(require_explicit_asset, False)
    if req_explicit and (not has_asset_key or (fee > 0.0 and not fee_asset)):
        return None

    base_u = base_asset.strip().upper() if isinstance(base_asset, str) else ""
    quote_u = quote_asset.strip().upper() if isinstance(quote_asset, str) else ""
    if not fee_asset and not req_explicit and quote_u:
        fee_asset = quote_u

    px = 0.0
    if fill_price is not None and not isinstance(fill_price, bool):
        try:
            fp_val = float(fill_price)
            if math.isfinite(fp_val) and fp_val > 0.0:
                px = fp_val
        except (TypeError, ValueError, OverflowError):
            px = 0.0
    if px <= 0.0:
        px = extract_order_price(order, default=0.0)
    base_fee = 0.0
    quote_fee = 0.0
    fee_in_quote: Optional[float] = 0.0 if fee == 0.0 else None

    if fee > 0.0:
        if base_u and fee_asset == base_u:
            base_fee = fee
            if px > 0.0:
                fee_in_quote = fee * px
        elif (quote_u and fee_asset == quote_u) or (not fee_asset and not req_explicit):
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


_PARENT_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "oid")


def aggregate_fills_by_order_id(
    rows: list,
    *,
    cutoff_ms: Optional[float] = None,
    side: Optional[str] = None,
) -> List[dict]:
    """Normalize and aggregate fills sharing the same parent order ID and side.

    Multi-fill executions sharing the same parent order ID (``orderId``, ``order_id``,
    ``ordertxid``, ``oid``) and side are combined into a single order record with
    volume-weighted average ``price``, summed ``qty``, and ``max(timestamp)``,
    matching ``BinanceProvider.get_orders()``.
    """
    if rows is None or isinstance(rows, (str, bytes, bool)) or not hasattr(rows, "__iter__"):
        return []
    want = None
    if side is not None:
        if isinstance(side, bool):
            return []
        want = normalize_side(side)
        if not want:
            return []
    out: List[dict] = []
    grouped: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        parent_oid = None
        for k in _PARENT_ORDER_ID_KEYS:
            cleaned_oid = _clean_single_order_id(r.get(k))
            if cleaned_oid is not None and cleaned_oid != "0":
                parent_oid = cleaned_oid
                break
        norm = normalize_order(r)
        ts_ms = norm["timestamp"]
        if cutoff_ms is not None:
            if isinstance(cutoff_ms, bool):
                continue
            try:
                cutoff_f = float(cutoff_ms)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(cutoff_f) or ts_ms is None or float(ts_ms) < cutoff_f:
                continue
        if want and norm["side"] != want:
            continue
        if parent_oid:
            key = (parent_oid, norm["side"])
            px = float(norm["price"] or 0.0)
            qty = float(norm["qty"] or 0.0)
            priced_qty = qty if (px > 0.0 and qty > 0.0) else 0.0
            ts_val = ts_ms if ts_ms is not None else 0
            if key not in grouped:
                grouped[key] = {
                    "orderId": parent_oid,
                    "side": norm["side"],
                    "cost": px * priced_qty,
                    "cost_qty": priced_qty,
                    "qty": qty,
                    "fallback_price": px,
                    "timestamp": ts_val,
                }
                out.append(grouped[key])
            else:
                g = grouped[key]
                g["cost"] += px * priced_qty
                g["cost_qty"] += priced_qty
                g["qty"] += qty
                if px > 0:
                    g["fallback_price"] = px
                if ts_val > (g["timestamp"] or 0):
                    g["timestamp"] = ts_val
        else:
            out.append(norm)
    result: List[dict] = []
    for item in out:
        if "cost" in item:
            total_qty = item["qty"]
            cost_qty = item.get("cost_qty", total_qty)
            vwap = (item["cost"] / cost_qty) if cost_qty > 0 else item["fallback_price"]
            result.append({
                "orderId": item["orderId"],
                "side": item["side"],
                "price": vwap,
                "qty": total_qty,
                "timestamp": item["timestamp"],
            })
        else:
            result.append(item)
    return result


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

    def place_order(self, symbol: str, side: str, price: float, qty: float, force: bool = False, **kwargs):
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
        if qty is None or isinstance(qty, bool):
            return 0.0
        try:
            qty_f = float(qty)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(qty_f) or qty_f <= 0.0:
            return 0.0
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            dec_raw = getattr(pp, "volume_decimals", None) if pp is not None else None
            if dec_raw is not None and not isinstance(dec_raw, bool):
                dec_f = float(dec_raw)
                if math.isfinite(dec_f) and dec_f.is_integer():
                    dec = int(dec_f)
                    if dec >= 0:
                        dec = min(dec, 18)
                        factor = 10 ** dec
                        return math.floor(qty_f * factor + 1e-12) / factor
        except Exception:
            pass
        return qty_f

    def round_amount(self, symbol: str, qty: float) -> float:
        """Backward-compatible alias for round_quantity."""
        return self.round_quantity(symbol, qty)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        if price is None or isinstance(price, bool):
            return 0.0
        try:
            price_f = float(price)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(price_f) or price_f <= 0.0:
            return 0.0
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            dec_raw = getattr(pp, "price_decimals", None) if pp is not None else None
            if dec_raw is not None and not isinstance(dec_raw, bool):
                dec_f = float(dec_raw)
                if math.isfinite(dec_f) and dec_f.is_integer():
                    dec = int(dec_f)
                    if dec >= 0:
                        return round(price_f, min(dec, 18))
        except Exception:
            pass
        return price_f

    def min_order_qty(self, symbol: str) -> float:
        """Return venue minimum order quantity; 0.0 means no known minimum."""
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None:
                raw_min = getattr(pp, "order_min", 0.0)
                if raw_min is not None and not isinstance(raw_min, bool):
                    val = float(raw_min)
                    if math.isfinite(val) and val > 0.0:
                        return val
        except Exception:
            pass
        return 0.0

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Return a provider-specific pre-submit refusal, if one is known."""
        if qty is None or isinstance(qty, bool):
            return "qty_zero_or_negative"
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
        win_valid = False
        if window_s is not None and not isinstance(window_s, bool):
            try:
                win_f = float(window_s)
                if math.isfinite(win_f) and win_f > 0:
                    window_s = win_f
                    win_valid = True
            except (TypeError, ValueError, OverflowError):
                win_valid = False
        if not win_valid:
            sb_val = 0.0
            if safeback_sec is not None and not isinstance(safeback_sec, bool):
                try:
                    cand_sb = float(safeback_sec)
                    if math.isfinite(cand_sb) and cand_sb > 0.0:
                        sb_val = cand_sb
                except (TypeError, ValueError, OverflowError):
                    sb_val = 0.0
            window_s = sb_val
        wr_kwargs = {}
        if reference_mode is not None:
            wr_kwargs["reference_mode"] = reference_mode
        if module is not None:
            wr_kwargs["module"] = module
        return order_guard.window_reference(self, symbol, side, window_s, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        if isinstance(order_type, bool):
            return None
        ot_norm = str(order_type or "").strip().upper()
        if ot_norm in ("BUY", "B", "BID", "LONG"):
            opposite = "SELL"
        elif ot_norm in ("SELL", "S", "A", "ASK", "SHORT"):
            opposite = "BUY"
        else:
            return None
        eff_since_s = 90.0 * 24.0 * 3600.0
        if since_s is not None and not isinstance(since_s, bool):
            try:
                cand_s = float(since_s)
                if math.isfinite(cand_s) and cand_s >= 0.0:
                    eff_since_s = cand_s
            except (TypeError, ValueError, OverflowError):
                pass
        fills = self.get_orders(symbol, opposite, eff_since_s)
        if fills is None:
            raise ProviderError(f"last_opposite_fill({symbol}): failed to read order history")
        valid_fills = [
            order for order in fills
            if isinstance(order, dict) and extract_order_price(order, default=0.0) > 0.0
        ]
        if not valid_fills:
            return None
        latest = max(valid_fills, key=lambda order: extract_timestamp_ms(order, default=0) or 0)
        price = extract_order_price(latest)
        return price or None

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        """Return completed-bar closing prices from closed interval series."""
        if hasattr(self, "ohlc_series"):
            series = self.ohlc_series(symbol, interval_min)
            closes = getattr(series, "closes", None) if series is not None else None
            return list(closes) if closes is not None else []
        return []
