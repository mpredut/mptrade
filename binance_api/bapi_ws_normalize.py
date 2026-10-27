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


def _clean_now_ms(now_ms: Any) -> int:
    if now_ms is not None and not isinstance(now_ms, bool):
        try:
            f = float(now_ms)
            if math.isfinite(f) and f > 0:
                return int(f)
        except (TypeError, ValueError, OverflowError):
            pass
    return int(time.time() * 1000)


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
        if isinstance(raw_t, float) and (not math.isfinite(raw_t) or not raw_t.is_integer()):
            raise ValueError("non-integer Binance trade ID")
        elif isinstance(raw_t, str):
            raw_t = raw_t.strip()
            if raw_t.endswith(".0") and raw_t[:-2].lstrip("+-").isdigit():
                raw_t = raw_t[:-2]
            if raw_t.startswith("+") and raw_t[1:].isdigit():
                raw_t = raw_t[1:]
        if isinstance(raw_i, float):
            if not math.isfinite(raw_i) or not raw_i.is_integer():
                raise ValueError("non-integer Binance order ID")
            raw_i = int(raw_i)
        elif isinstance(raw_i, str):
            raw_i = raw_i.strip() or None
            if raw_i is not None and raw_i.endswith(".0") and raw_i[:-2].lstrip("+-").isdigit():
                raw_i = raw_i[:-2]
            if raw_i is not None and raw_i.startswith("+") and raw_i[1:].isdigit():
                raw_i = raw_i[1:]
            if raw_i is not None and (
                raw_i in {"0", "+0", "-0"}
                or raw_i.lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}
            ):
                raw_i = None
        elif not isinstance(raw_i, int):
            raw_i = None
        if raw_i is None or (isinstance(raw_i, int) and raw_i <= 0) or (isinstance(raw_i, str) and raw_i.startswith("-")):
            raise ValueError("missing or invalid Binance order ID")
        fill_id = int(raw_t)
        if fill_id < 0:
            raise ValueError("missing Binance trade ID")
        side = extract_order_side(event)
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
        if not math.isfinite(raw_trade_id) or not raw_trade_id.is_integer():
            builtins.print("[cacheManager][WS] rejected Trade report with non-integer float trade ID")
            return None
        raw_trade_id = int(raw_trade_id)
    elif isinstance(raw_trade_id, str):
        raw_trade_id = raw_trade_id.strip() or None
        if raw_trade_id is not None and raw_trade_id.endswith(".0") and raw_trade_id[:-2].lstrip("+-").isdigit():
            raw_trade_id = raw_trade_id[:-2]
        if raw_trade_id is not None and raw_trade_id.startswith("+") and raw_trade_id[1:].isdigit():
            raw_trade_id = raw_trade_id[1:]
        if raw_trade_id is not None and (
            raw_trade_id.startswith("-")
            or raw_trade_id.lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}
        ):
            builtins.print("[cacheManager][WS] rejected Trade report with missing trade ID")
            return None
    elif raw_trade_id is not None and not isinstance(raw_trade_id, int):
        builtins.print("[cacheManager][WS] rejected Trade report with non-scalar trade ID")
        return None
    if isinstance(raw_order_id, float):
        if not math.isfinite(raw_order_id) or not raw_order_id.is_integer():
            raw_order_id = None
        else:
            raw_order_id = int(raw_order_id)
    elif isinstance(raw_order_id, str):
        raw_order_id = raw_order_id.strip() or None
        if raw_order_id is not None and raw_order_id.endswith(".0") and raw_order_id[:-2].lstrip("+-").isdigit():
            raw_order_id = raw_order_id[:-2]
        if raw_order_id is not None and raw_order_id.startswith("+") and raw_order_id[1:].isdigit():
            raw_order_id = raw_order_id[1:]
        if raw_order_id is not None and (
            raw_order_id.startswith("-")
            or raw_order_id in {"0", "+0", "-0"}
            or raw_order_id.lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}
        ):
            raw_order_id = None
    elif not isinstance(raw_order_id, int):
        raw_order_id = None
    if isinstance(raw_order_id, int) and raw_order_id <= 0:
        raw_order_id = None
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
            qty_f = float(raw_qty)
            raw_y = event.get("Y")
            if _is_positive_finite(raw_y):
                raw_price = str(float(raw_y) / qty_f)
            else:
                raw_cum_quote = event.get("Z")
                raw_cum_qty = event.get("z")
                cum_quote = float(raw_cum_quote) if _is_positive_finite(raw_cum_quote) else 0.0
                cum_qty = float(raw_cum_qty) if _is_positive_finite(raw_cum_qty) else qty_f
                if cum_quote > 0 and cum_qty > 0:
                    raw_price = str(cum_quote / cum_qty)
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
    if raw_n is not None and not isinstance(raw_n, bool):
        try:
            nf = float(raw_n)
            if math.isfinite(nf) and nf >= 0:
                trade_item["commission"] = str(raw_n).strip()
        except (TypeError, ValueError, OverflowError):
            pass
    raw_N = event.get("N")
    if isinstance(raw_N, str) and raw_N.strip():
        trade_item["commissionAsset"] = raw_N.strip()
    return symbol, trade_item
