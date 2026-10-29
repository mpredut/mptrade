
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Optional

# BINANCE
from binance.exceptions import BinanceAPIException

# MYLIB
from .bapi_client import client
import symbols as sym
from botcore import safe_int, safe_pos_int, safe_nonneg_float, safe_pos_float
from providers.base import (
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)


from .bapi import (
    _INVALID_BAPI_ID_STRINGS,
    _clean_bapi_oid_str,
    _normalize_side,
    _parse_bool_flag,
)


def _parse_tail_trade_id(raw_tid: Any) -> Optional[int]:
    if raw_tid is None or isinstance(raw_tid, bool):
        return None
    if isinstance(raw_tid, float):
        if not math.isfinite(raw_tid) or not raw_tid.is_integer():
            return None
        raw_tid = int(raw_tid)
    elif isinstance(raw_tid, str):
        s = raw_tid.strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.startswith("+"):
            s = s[1:]
        if not s or not s.isdigit():
            return None
        raw_tid = s
    return safe_pos_int(raw_tid, default=None)


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
    start_i = safe_int(start_time_ms, default=None)
    if start_i is None:
        if strict:
            raise ValueError("start_time_ms must be a valid timestamp")
        return []
    if start_i < 0:
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
        tid_int = _parse_tail_trade_id(tail.get("id") if isinstance(tail, dict) else None)
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


def get_filled_orders(order_type, symbol, startTime, limit=1000, *, strict=False):
    strict = _parse_bool_flag(strict, False)
    try:
        order_type = _normalize_side(order_type)
        sym.validate_symbols(symbol)
        sym.validate_ordertype(order_type)
        start_i = safe_int(startTime, default=None)
        if start_i is None or start_i < 0:
            raise ValueError("startTime must be a valid timestamp")
        startTime = start_i

        end_time = int(time.time() * 1000)  # ms

        # Pagination prevents truncation above 1,000 fills.
        trades = paginate_my_trades(
            client, symbol, startTime, limit, strict=strict)

        filtered = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            ts_ms = extract_timestamp_ms(trade, default=0) or 0
            side = extract_order_side(trade)
            px = extract_order_price(trade)
            qty = extract_order_qty(trade)
            if (
                px > 0
                and qty > 0
                and startTime <= ts_ms <= end_time + 5_000
                and (order_type is None or side == order_type.upper())
            ):
                filtered.append({
                    'orderId': trade.get('orderId'),
                    'price': px,
                    'quantity': qty,
                    'timestamp': ts_ms,
                    'side': side,
                })

        #print(f"Filtered filled trades ({'ALL' if order_type is None else order_type}): {len(filtered_trades)} from {len(trades)} trades")
        for t in filtered[:5]:
            print(t)

        if not filtered:
            return []
        
        grouped = defaultdict(lambda: {'price_qty': 0, 'quantity': 0, 'timestamp': 0, 'side': ''})

        for idx, t in enumerate(filtered):
            raw_oid = t['orderId']
            if isinstance(raw_oid, float) and math.isfinite(raw_oid) and raw_oid.is_integer():
                raw_oid = int(raw_oid)
            elif isinstance(raw_oid, str):
                raw_oid = _clean_bapi_oid_str(raw_oid)
            valid_oid = (
                raw_oid is not None
                and not isinstance(raw_oid, (bool, float, dict, list, tuple, set, frozenset))
                and (not isinstance(raw_oid, int) or raw_oid > 0)
            )
            key = raw_oid if valid_oid else f"_unkeyed_{idx}"
            grouped[key]['price_qty'] += t['price'] * t['quantity']
            grouped[key]['quantity'] += t['quantity']
            grouped[key]['timestamp'] = max(grouped[key]['timestamp'], t['timestamp'])
            grouped[key]['side'] = t['side']

        result = []
        for oid, data in grouped.items():
            raw_qty = data['quantity']
            rounded_qty = round(raw_qty, 8) if raw_qty >= 1e-8 else round(raw_qty, 12)
            qty_out = rounded_qty if rounded_qty > 0 else raw_qty
            if raw_qty > 0:
                raw_px = data['price_qty'] / raw_qty
                rounded_px = round(raw_px, 8) if raw_px >= 1e-8 else round(raw_px, 12)
                px_out = rounded_px if rounded_px > 0 else raw_px
            else:
                px_out = 0
            result.append({
                'orderId': oid,
                'price': px_out,
                'quantity': qty_out,
                'qty': qty_out,
                'timestamp': data['timestamp'],
                'time': data['timestamp'],
                'side': data['side'],
                'isBuyer': data['side'] == 'BUY',
            })

        for r in result[:5]:
            print(r)

        return result


    except Exception as e:
        print(f"Unexpected error in get_filled_orders: {e}")
        if strict:
            raise
        return []



def get_recent_filled_orders(order_type, symbol, max_age_seconds):
    max_age_val = safe_nonneg_float(max_age_seconds, default=None)
    if max_age_val is None:
        return []
    # Convert duration to an absolute millisecond start time. Passing duration directly
    # would query from approximately 1970 and return incorrect data.
    start_time_ms = int(time.time() * 1000) - int(max_age_val * 1000)
    all_filled_orders = get_filled_orders(order_type, symbol, start_time_ms)
    recent_filled_orders = []
    current_time = time.time()
    if(len(all_filled_orders) < 1) :
        return []
    
    print(f"have {len(all_filled_orders)} orders. ignore oldest.")
    for order in all_filled_orders:
        ts_ms = extract_timestamp_ms(order, default=0) or 0
        age_sec = current_time - ts_ms / 1000.0
        if ts_ms > 0 and -5.0 <= age_sec <= max_age_val:
            recent_filled_orders.append(order)

    # Sort the recent_filled_orders by price in ascending order
    recent_filled_orders.sort(key=lambda x: x['price'])

    return recent_filled_orders



def get_trade_orders(order_type, symbol, max_age_seconds):
    import cacheManager as cm
    # Consumer processes read the cache written by the dedicated cacheManager process.
    # Starting a second polling loop here duplicates API traffic and mixes cache logs
    # into callers such as order_retry_worker.
    cache_order_manager = cm.get_cache_manager("Order", start_sync=False)

    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)

    sync_reader = getattr(
        cache_order_manager, "sync_account_reader_if_needed", None
    )
    if callable(sync_reader):
        sync_reader()

    # A WebSocket can update a partially filled order in place. Copy each row
    # under the manager lock so financial totals never observe a torn mutation.
    sym_u = str(symbol or "").strip().upper()
    sym_clean = sym_u.replace("/", "")
    with cache_order_manager.lock:
        raw_bucket = (
            cache_order_manager.cache.get(symbol)
            or cache_order_manager.cache.get(sym_u)
            or cache_order_manager.cache.get(sym_clean)
            or []
        )
        if not isinstance(raw_bucket, (list, tuple)):
            raw_bucket = []
        orders_for_symbol = [
            dict(order) for order in raw_bucket
            if isinstance(order, dict)
        ]
    if not orders_for_symbol:
        return []

    #print(f" orders_for_symbol {orders_for_symbol}")
    max_age_val = safe_nonneg_float(max_age_seconds, default=None)
    if max_age_val is None:
        return []
    current_time_ms = int(time.time() * 1000)
    max_age_ms = max_age_val * 1000.0  # convert to ms

    filtered_orders = []
    for order in orders_for_symbol:
        side = extract_order_side(order)
        ts_ms = extract_timestamp_ms(order, default=0) or 0
        age_ms = current_time_ms - ts_ms
        if ts_ms > 0 and (order_type is None or side == order_type) and -5_000 <= age_ms <= max_age_ms:
            px = extract_order_price(order)
            qty = extract_order_qty(order)
            if px <= 0 or qty <= 0:
                continue
            filtered_orders.append({
                'orderId': order.get('orderId'),
                'price': px,
                'quantity': qty,
                'qty': qty,
                'timestamp': ts_ms,
                'time': ts_ms,
                'side': side,
                'isBuyer': side == 'BUY',
            })

    #print(f" filtered_orders {filtered_orders} , current_time_ms {current_time_ms} timestamp  max_age+ms {max_age_ms}")
    return filtered_orders

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
            qty_f = safe_pos_float(t.get('quantity', t.get('qty')), default=None)
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
