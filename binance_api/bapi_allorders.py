
from __future__ import annotations

import time
from typing import Any

# BINANCE
from binance.exceptions import BinanceAPIException

# MYLIB
import instrument_registry as sym
from botcore import (
    clean_order_id,
    exact_pos_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import cache_symbol_variants
from instrument_registry import casefold_get
from providers.order_ohlc_utils import (
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)


from .bapi import (
    _normalize_side,
    _parse_bool_flag,
)


def _build_trade_order_record(order_id: Any, price: float, qty: float, timestamp_ms: int, side: str) -> dict[str, Any]:
    return {
        'orderId': order_id,
        'price': price,
        'quantity': qty,
        'qty': qty,
        'timestamp': timestamp_ms,
        'time': timestamp_ms,
        'side': side,
        'isBuyer': side == 'BUY',
    }


def paginate_my_trades(api_client, symbol, start_time_ms, limit=1000, *,
                       strict=False):
    """Fetch every fill from ``start_time_ms`` through now using pagination.

    The first page uses ``startTime`` and later pages use the previous final fill ID
    plus one until a short page arrives.
    """
    strict = _parse_bool_flag(strict, False)
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        if strict:
            raise ValueError("symbol must be a non-empty string")
        return []
    symbol = symbol.strip().upper()
    start_i = safe_nonneg_int(start_time_ms, default=None)
    if start_i is None:
        if strict:
            raise ValueError("start_time_ms must be a finite non-negative timestamp")
        return []
    start_time_ms = start_i
    limit = min(1000, max(1, safe_pos_int(limit, default=1000)))
    out = []
    from_id = None
    while True:
        try:
            if from_id is None:
                batch = api_client.get_my_trades(
                    symbol=symbol, startTime=start_time_ms, limit=limit)
            else:
                batch = api_client.get_my_trades(
                    symbol=symbol, fromId=from_id, limit=limit)
            if batch is None:
                if strict:
                    raise RuntimeError(
                        f"Binance returned no trade page for {symbol}")
                batch = []
        except BinanceAPIException as api_err:
            err_msg = str(api_err)
            if getattr(api_err, "code", None) == -1003 or "too much request weight" in err_msg.lower():
                print(f"[paginate_my_trades] rate limit → backoff 60s: {err_msg}")
                time.sleep(60)
                continue
            print(f"[paginate_my_trades] {symbol}: {api_err}")
            if strict:
                raise
            break
        if not batch:
            break
        out.extend(batch)
        if len(batch) < limit:          # Final page.
            break
        tail = batch[-1]
        tid_int = exact_pos_int(clean_order_id(tail.get("id") if isinstance(tail, dict) else None), default=None)
        if tid_int is None:
            if strict:
                raise RuntimeError(f"Binance trade page missing valid tail id for {symbol}")
            break
        next_from_id = tid_int + 1   # Continue from the next fill ID.
        if from_id is not None and next_from_id <= from_id:
            break
        from_id = next_from_id
        time.sleep(0.2)                 # Polite towards the rate limit.
    return out


def snapshot_cache_symbol_rows(manager, symbol: str) -> list[dict]:
    sync_reader = getattr(manager, "sync_account_reader_if_needed", None)
    if callable(sync_reader):
        sync_reader()

    # A WebSocket can update a partially filled order in place. Copy each row
    # under the manager lock so financial totals never observe a torn mutation.
    with manager.lock:
        raw_bucket = next(
            (
                b
                for cand in cache_symbol_variants(symbol)
                if isinstance(b := casefold_get(manager.cache, cand), (list, tuple)) and b
            ),
            [],
        )
        return [
            dict(row) for row in raw_bucket
            if isinstance(row, dict)
        ]


def filter_recent_cached_orders(manager, order_type, symbol, max_age_seconds):
    """Validate inputs, snapshot symbol rows from ``manager``, and return recent ``(row, side, ts_ms, px, qty)``."""
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    orders_for_symbol = snapshot_cache_symbol_rows(manager, symbol)
    max_age_val = safe_nonneg_float(max_age_seconds, default=None) if orders_for_symbol else None
    if max_age_val is None:
        return []
    current_time_ms = int(time.time() * 1000)
    max_age_ms = max_age_val * 1000.0
    filtered = []
    for order in orders_for_symbol:
        side = extract_order_side(order)
        ts_ms = extract_timestamp_ms(order, default=0) or 0
        age_ms = current_time_ms - ts_ms
        if ts_ms > 0 and (order_type is None or side == order_type) and -5_000 <= age_ms <= max_age_ms:
            px = extract_order_price(order)
            qty = extract_order_qty(order)
            if px > 0 and qty > 0:
                filtered.append((order, side, ts_ms, px, qty))
    return filtered


def get_trade_orders(order_type, symbol, max_age_seconds):
    import cacheManager as cm
    # Consumer processes read the cache written by the dedicated cacheManager process.
    # Starting a second polling loop here duplicates API traffic and mixes cache logs
    # into callers such as order_retry_worker.
    cache_order_manager = cm.get_cache_manager("Order", start_sync=False)
    return [
        _build_trade_order_record(order.get("orderId"), px, qty, ts_ms, side)
        for order, side, ts_ms, px, qty in filter_recent_cached_orders(
            cache_order_manager, order_type, symbol, max_age_seconds
        )
    ]

# The default aggregation window is the previous 24 hours.
def get_total_traded_stats(symbol, period_seconds=86400):
    period_val = safe_pos_float(period_seconds, default=86400.0) or 86400.0
    trades = get_trade_orders(order_type=None, symbol=symbol, max_age_seconds=period_val)

    stats = {
        'BUY': {'total_quantity': 0, 'total_value': 0, 'trade_count': 0},
        'SELL': {'total_quantity': 0, 'total_value': 0, 'trade_count': 0}
    }

    for t in trades or []:
        if not isinstance(t, dict):
            continue
        side = _normalize_side(t.get('side'))
        if side in stats:
            px_f = safe_pos_float(t.get('price'), default=None)
            raw_qty = t.get('quantity') if t.get('quantity') is not None else t.get('qty')
            qty_f = safe_pos_float(raw_qty, default=None)
            if px_f is None or qty_f is None:
                continue
            stats[side]['total_quantity'] += qty_f
            stats[side]['total_value'] += px_f * qty_f
            stats[side]['trade_count'] += 1

    # Round the accumulated values once, after all rows are included.
    for side in stats:
        stats[side]['total_quantity'] = round(stats[side]['total_quantity'], 8)
        stats[side]['total_value'] = round(stats[side]['total_value'], 8)

    return stats
