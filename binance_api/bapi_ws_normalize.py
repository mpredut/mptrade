"""Pure normalization of Binance WebSocket executionReport events into Order/Trade rows."""
from __future__ import annotations

import builtins
import math
import time
from typing import Any

from providers.base import (
    extract_order_price,
    extract_order_side,
    extract_timestamp_ms,
)


def _is_positive_finite(val: Any) -> bool:
    if isinstance(val, bool) or val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f) and f > 0
    except (TypeError, ValueError, OverflowError):
        return False


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
    symbol = event.get("s")
    if not symbol or not isinstance(symbol, str):
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
        if isinstance(raw_t, bool) or isinstance(event.get("i"), bool):
            raise ValueError("boolean Binance trade/order ID")
        fill_id = int(raw_t)
        if fill_id < 0:
            raise ValueError("missing Binance trade ID")
        raw_l = event.get("l")
        last_quantity = float(raw_l) if _is_positive_finite(raw_l) else 0.0
        if last_quantity <= 0 and event.get("X") == "FILLED":
            raw_z = event.get("z")
            raw_q = event.get("q")
            if _is_positive_finite(raw_z):
                last_quantity = float(raw_z)
            elif _is_positive_finite(raw_q):
                last_quantity = float(raw_q)
        raw_px = extract_order_price(event, default=0.0)
        last_price = float(raw_px) if _is_positive_finite(raw_px) else 0.0
        if last_price <= 0 and last_quantity > 0:
            raw_y = event.get("Y")
            last_quote = float(raw_y) if _is_positive_finite(raw_y) else 0.0
            if last_quote > 0:
                last_price = last_quote / last_quantity
            else:
                raw_cum_quote = event.get("Z")
                raw_cum_qty = event.get("z")
                cum_quote = float(raw_cum_quote) if _is_positive_finite(raw_cum_quote) else 0.0
                cum_qty = float(raw_cum_qty) if _is_positive_finite(raw_cum_qty) else last_quantity
                if cum_quote > 0 and cum_qty > 0:
                    last_price = cum_quote / cum_qty
        fallback_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        order_item = {
            "orderId": event.get("i"),
            # Store one immutable fill. A cumulative Z/z aggregate cannot name all
            # constituent fill IDs after reconnect, so REST backfill could count the
            # older fills twice. L/l plus t is exactly deduplicable across both paths.
            "price": last_price,
            "quantity": last_quantity,
            "timestamp": int(
                extract_timestamp_ms(event, default=None) or fallback_ms
            ),
            "side": extract_order_side(event) or event.get("S"),
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
    symbol = event.get("s")
    if not symbol or not isinstance(symbol, str):
        return None

    side = extract_order_side(event)
    if side not in ("BUY", "SELL"):
        builtins.print("[cacheManager][WS] rejected Trade report with invalid side")
        return None
    raw_trade_id = event.get("t")
    if isinstance(raw_trade_id, bool) or isinstance(event.get("i"), bool):
        builtins.print("[cacheManager][WS] rejected Trade report with boolean ID")
        return None
    try:
        if raw_trade_id is not None and int(raw_trade_id) < 0:
            builtins.print("[cacheManager][WS] rejected Trade report with missing trade ID")
            return None
    except (TypeError, ValueError):
        pass
    trade_id = str(
        raw_trade_id if raw_trade_id is not None
        else f"{event.get('i')}-{event.get('T')}"
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
    ts_ms = extract_timestamp_ms(event)
    if ts_ms is None or ts_ms <= 0:
        ts_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    trade_item: dict[str, Any] = {
        "symbol": symbol,
        "id": trade_id,
        "orderId": event.get("i"),
        "price": raw_price,
        "qty": raw_qty,
        "time": int(ts_ms),
        "isBuyer": side == "BUY",
    }
    if event.get("n") is not None:
        trade_item["commission"] = str(event.get("n"))
    if event.get("N") is not None:
        trade_item["commissionAsset"] = str(event.get("N"))
    return symbol, trade_item
