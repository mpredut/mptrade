"""Pure normalization of Binance WebSocket executionReport events into Order/Trade rows."""
from __future__ import annotations

import builtins
import time
from typing import Any

from botcore import exact_pos_int, safe_float, safe_nonneg_float, safe_pos_float, safe_pos_int
from providers.base import (
    clean_order_id,
    extract_order_price,
    extract_order_side,
    extract_timestamp_ms,
)


def _is_positive_finite(val: Any) -> bool:
    return safe_pos_float(val, default=None) is not None


def _clean_now_ms(now_ms: Any) -> int:
    return safe_pos_int(now_ms, default=None) or int(time.time() * 1000)


def _strip_int_str(val: Any) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
        s = s[:-2]
    if s.startswith("+") and s[1:].isdigit():
        s = s[1:]
    return s


def _normalize_ws_order_id(raw_id: Any, *, strict_float: bool = False) -> int | str | None:
    if isinstance(raw_id, bool):
        if strict_float:
            raise ValueError("boolean Binance order ID")
        return None
    if isinstance(raw_id, (int, float)):
        iv = exact_pos_int(raw_id, default=None)
        if iv is None and strict_float and isinstance(raw_id, float):
            f = safe_float(raw_id, default=None)
            if f is None or not f.is_integer():
                raise ValueError("non-integer Binance order ID")
        return iv
    return clean_order_id(raw_id) if isinstance(raw_id, str) else None


def _derive_ws_quote_fill_price(event: dict[str, Any], last_quantity: float) -> float:
    if last_quantity <= 0:
        return 0.0
    last_quote = safe_pos_float(event.get("Y"), default=0.0)
    if last_quote > 0:
        return last_quote / last_quantity
    cum_quote = safe_pos_float(event.get("Z"), default=0.0)
    cum_qty = safe_pos_float(event.get("z"), default=last_quantity)
    if cum_quote > 0 and cum_qty > 0:
        return cum_quote / cum_qty
    return 0.0


def parse_order_execution_report(
    event: dict[str, Any],
    *,
    now_ms: int | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Normalize a Binance ``executionReport`` fill into ``(symbol, order_item)``.

    Returns ``None`` when the report is not a fill (e.g. ``NEW`` or ``CANCELED``
    without execution) or when required fields are malformed.
    """
    if not isinstance(event, dict):
        return None
    raw_symbol = event.get("s")
    if not raw_symbol or not isinstance(raw_symbol, str):
        return None
    symbol = raw_symbol.strip().upper()
    if not symbol:
        return None

    # Cache only executed orders, matching get_filled_orders. NEW or terminal orders
    # without fills are not transactions and would pollute the profit guard with zero prices.
    if (
        event.get("x") != "TRADE"
        or event.get("X") not in ("FILLED", "PARTIALLY_FILLED")
    ):
        return None

    try:
        raw_t = event.get("t")
        raw_i = event.get("i")
        raw_s = event.get("S")
        if isinstance(raw_t, bool) or isinstance(raw_i, bool) or isinstance(raw_s, bool):
            raise ValueError("boolean Binance trade/order ID or side")
        if isinstance(raw_t, float):
            f = safe_float(raw_t, default=None)
            if f is None or not f.is_integer():
                raise ValueError("non-integer Binance trade ID")
        elif isinstance(raw_t, str):
            raw_t = _strip_int_str(raw_t) or ""
        raw_i = _normalize_ws_order_id(raw_i, strict_float=True)
        if raw_i is None:
            raise ValueError("missing or invalid Binance order ID")
        fill_id = int(raw_t)
        if fill_id < 0:
            raise ValueError("missing Binance trade ID")
        side = extract_order_side(event)
        last_quantity = safe_pos_float(event.get("l"), default=0.0)
        if last_quantity <= 0 and event.get("X") == "FILLED":
            last_quantity = (
                safe_pos_float(event.get("z"), default=None)
                or safe_pos_float(event.get("q"), default=0.0)
            )
        last_price = safe_pos_float(extract_order_price(event, default=0.0), default=0.0)
        if last_price <= 0 and last_quantity > 0:
            last_price = _derive_ws_quote_fill_price(event, last_quantity)
        fallback_ms = _clean_now_ms(now_ms)
        ts_ms = extract_timestamp_ms(event, default=None)
        if ts_ms is None or ts_ms <= 0:
            ts_ms = fallback_ms
        order_item = {
            "orderId": raw_i,
            # Store one immutable fill. A cumulative Z/z aggregate cannot name all
            # constituent fill IDs after reconnect, so REST backfill could count the
            # older fills twice. L/l plus t is exactly deduplicable across both paths.
            "price": last_price,
            "quantity": last_quantity,
            "timestamp": int(ts_ms),
            "side": side,
            "status": event.get("X"),
            "symbol": symbol,
            "eventType": event.get("x"),
            "_fillId": str(fill_id),
        }
        return symbol, order_item
    except (TypeError, ValueError, OverflowError) as exc:
        builtins.print(
            f"[cacheManager][WS] rejected malformed order report: {exc}"
        )
        return None


def parse_trade_execution_report(
    event: dict[str, Any],
    *,
    now_ms: int | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Normalize a Binance ``executionReport`` fill into ``(symbol, trade_item)``.

    Returns ``None`` when the report is not a trade fill or when required fields
    are invalid.
    """
    if not isinstance(event, dict) or event.get("x") != "TRADE":
        return None
    raw_symbol = event.get("s")
    if not raw_symbol or not isinstance(raw_symbol, str):
        return None
    symbol = raw_symbol.strip().upper()
    if not symbol:
        return None

    side = extract_order_side(event)
    if side not in ("BUY", "SELL"):
        builtins.print("[cacheManager][WS] rejected Trade report with invalid side")
        return None
    raw_trade_id = event.get("t")
    raw_order_id = event.get("i")
    if isinstance(raw_trade_id, bool) or isinstance(raw_order_id, bool):
        builtins.print("[cacheManager][WS] rejected Trade report with boolean ID")
        return None
    if isinstance(raw_trade_id, float):
        f = safe_float(raw_trade_id, default=None)
        if f is None or not f.is_integer():
            builtins.print("[cacheManager][WS] rejected Trade report with non-integer float trade ID")
            return None
        raw_trade_id = int(f)
    elif isinstance(raw_trade_id, str):
        raw_trade_id = _strip_int_str(raw_trade_id)
        if raw_trade_id is not None and (
            raw_trade_id.startswith("-")
            or raw_trade_id.lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}
        ):
            builtins.print("[cacheManager][WS] rejected Trade report with missing trade ID")
            return None
    elif raw_trade_id is not None and not isinstance(raw_trade_id, int):
        builtins.print("[cacheManager][WS] rejected Trade report with non-scalar trade ID")
        return None
    raw_order_id = _normalize_ws_order_id(raw_order_id, strict_float=False)
    try:
        if raw_trade_id is not None and int(raw_trade_id) < 0:
            builtins.print("[cacheManager][WS] rejected Trade report with missing trade ID")
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    if raw_trade_id is None and raw_order_id is None:
        builtins.print("[cacheManager][WS] rejected Trade report with missing trade and order ID")
        return None
    ts_ms = extract_timestamp_ms(event)
    if ts_ms is None or ts_ms <= 0:
        ts_ms = _clean_now_ms(now_ms)
    trade_id = str(
        raw_trade_id if raw_trade_id is not None
        else f"{raw_order_id}-{int(ts_ms)}"
    )
    raw_price = event.get("L")
    if not _is_positive_finite(raw_price):
        raw_price = event.get("p")
    raw_qty = event.get("l")
    if not _is_positive_finite(raw_qty) and event.get("X") in (None, "FILLED"):
        raw_z = event.get("z")
        raw_qty = raw_z if _is_positive_finite(raw_z) else event.get("q")
    if not _is_positive_finite(raw_price):
        extracted_px = extract_order_price(event, default=0.0)
        if _is_positive_finite(extracted_px):
            raw_price = str(extracted_px)
        elif _is_positive_finite(raw_qty):
            derived_px = _derive_ws_quote_fill_price(event, float(raw_qty))
            if derived_px > 0:
                raw_price = str(derived_px)
    trade_item: dict[str, Any] = {
        "symbol": symbol,
        "id": trade_id,
        "orderId": raw_order_id,
        "price": raw_price,
        "qty": raw_qty,
        "time": int(ts_ms),
        "isBuyer": side == "BUY",
    }
    raw_n = event.get("n")
    if safe_nonneg_float(raw_n, default=None) is not None:
        trade_item["commission"] = str(raw_n).strip()
    raw_N = event.get("N")
    if isinstance(raw_N, str) and raw_N.strip():
        trade_item["commissionAsset"] = raw_N.strip()
    return symbol, trade_item
