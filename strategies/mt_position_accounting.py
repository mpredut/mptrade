"""Stateless spot position accounting, order summarization, and trade fill tracking helpers
for ``monitortrades.py``."""
from __future__ import annotations

import builtins
from typing import Any

from botcore import safe_pos_int
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_timestamp_ms,
)
import utils as u


class _FifoTradeIdTracker:
    """Memory-bounded trade ID tracker preserving FIFO insertion order."""

    def __init__(self, maxsize: int = 10000):
        self._data: dict[str, None] = {}
        ms = safe_pos_int(maxsize, default=10000) or 10000
        self._maxsize = ms if ms >= 2 else 10000

    def add(self, item: str) -> None:
        key = str(item)
        self._data.pop(key, None)
        self._data[key] = None
        if len(self._data) > self._maxsize:
            excess = len(self._data) - max(1, self._maxsize // 2)
            for _ in range(excess):
                self._data.pop(next(iter(self._data)), None)

    def __contains__(self, item: object) -> bool:
        return str(item) in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


def _extract_trade_uid(trade: Any) -> str:
    if not isinstance(trade, dict):
        return ""
    for key in ("id", "tradeId", "a"):
        s = extract_order_id(trade.get(key), default="")
        if s:
            return s
    return ""


def _order_timestamp_ms(order: dict) -> float:
    """Extract normalized millisecond timestamp from an order or fill record."""
    return float(extract_timestamp_ms(order, default=0) or 0)


def get_relevant_trade(trade_orders, trade_type, threshold_s=0, symbol=None, now_fn=None):
    """Return latest execution price and timestamp for trade_type in normalized format."""
    if not trade_orders:
        if builtins.print.__name__ != "_dummy_print":
            print(f"Warning: No {trade_type} transactions for that currency!!!")
        return None, 0, True

    if len(trade_orders) > 1:
        if isinstance(trade_orders, list):
            trade_orders.sort(key=_order_timestamp_ms, reverse=True)
        else:
            trade_orders = sorted(trade_orders, key=_order_timestamp_ms, reverse=True)
    first = trade_orders[0]
    trade_price = extract_order_price(first)
    trade_time = _order_timestamp_ms(first) / 1000.0
    if builtins.print.__name__ != "_dummy_print":
        print(f"{trade_type.capitalize()} price for {symbol}: {trade_price} at {u.timeToHMS(trade_time)}")
    return trade_price, trade_time, True


def _summarize_orders(orders):
    if not orders:
        return 0.0, 0.0, (0, 0.0, 0.0, 0, 0)
    total_qty = 0.0
    total_val = 0.0
    latest_ts = 0
    for o in orders:
        q = extract_order_qty(o)
        p = extract_order_price(o)
        ts = extract_timestamp_ms(o, default=0) or 0
        if q > 0.0 and p > 0.0:
            total_qty += q
            total_val += p * q
        if ts > latest_ts:
            latest_ts = ts
    first = orders[0]
    first_ts = extract_timestamp_ms(first, default=0) or 0
    first_id = (first.get("id") or first.get("orderId") or first_ts) if isinstance(first, dict) else first_ts
    sig = (len(orders), round(total_qty, 8), round(total_val, 4), latest_ts, first_id)
    return total_qty, total_val, sig


def _make_order_signature(orders):
    return _summarize_orders(orders)[2]


def _api_cache_identity(api) -> tuple[str, int]:
    name = getattr(api, "name", None) or getattr(api, "provider_label", None) or type(api).__name__
    return (str(name), id(api))


def _compute_spot_position(buy_orders, sell_orders) -> tuple[float, float, float]:
    """Compute remaining spot-attributed inventory and active cycle cost basis.

    Returns ``(owned_qty, active_avg_buy_price, last_cycle_buy_price)``.
    """
    if not buy_orders:
        return 0.0, 0.0, 0.0
    if not sell_orders:
        tot_q = 0.0
        tot_v = 0.0
        for o in buy_orders:
            q, p = extract_order_qty(o), extract_order_price(o)
            if q > 0.0 and p > 0.0:
                tot_q += q
                tot_v += q * p
        avg_px = (tot_v / tot_q) if tot_q > 0 else 0.0
        return tot_q, avg_px, avg_px
    events = [
        (_order_timestamp_ms(o), 0, q, p)
        for o in buy_orders
        if (q := extract_order_qty(o)) > 0.0 and (p := extract_order_price(o)) > 0.0
    ]
    events.extend(
        (_order_timestamp_ms(o), 1, q, extract_order_price(o))
        for o in sell_orders
        if (q := extract_order_qty(o)) > 0.0
    )
    events.sort(key=lambda x: (x[0], x[1]))
    owned = cost_qty = active_avg = last_cycle_avg = 0.0
    for _, is_sell, q, p in events:
        if q <= 0:
            continue
        if is_sell == 0:
            owned += q
            new_cost_qty = cost_qty + q
            active_avg = ((cost_qty * active_avg) + (q * p)) / new_cost_qty if new_cost_qty > 0 else 0.0
            cost_qty = new_cost_qty
            last_cycle_avg = active_avg
        else:
            if cost_qty > 0 and active_avg > 0:
                last_cycle_avg = active_avg
            owned = max(0.0, owned - q)
            if owned <= 1e-12:
                owned = cost_qty = active_avg = 0.0
            elif q >= cost_qty * 0.98 or cost_qty - q <= 1e-12:
                cost_qty = active_avg = 0.0
            else:
                cost_qty -= q
    return owned, active_avg, last_cycle_avg


def _compute_spot_owned_qty(buy_orders, sell_orders) -> float:
    """Compute remaining spot-attributed inventory in the lookback window."""
    return _compute_spot_position(buy_orders, sell_orders)[0]


def _coerce_order_list(orders) -> list:
    if isinstance(orders, list):
        return orders
    return [] if orders is None or isinstance(orders, bool) else list(orders)
