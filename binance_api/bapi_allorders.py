
import math
import time
from collections import defaultdict

# BINANCE
from binance.exceptions import BinanceAPIException

# MYLIB
from .bapi_client import client
import symbols as sym
from providers.base import (
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)


def paginate_my_trades(api_client, symbol, start_time_ms, limit=1000, *,
                       strict=False):
    """Fetch every fill from ``start_time_ms`` through now using pagination.

    The first page uses ``startTime`` and later pages use the previous final fill ID
    plus one until a short page arrives.
    """
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
        try:
            next_from_id = int(batch[-1]["id"]) + 1   # Continue from the next fill ID.
        except (KeyError, TypeError, ValueError):
            if strict:
                raise RuntimeError(f"Binance trade page missing valid tail id for {symbol}")
            break
        if from_id is not None and next_from_id <= from_id:
            break
        from_id = next_from_id
        time.sleep(0.2)                 # Polite towards the rate limit.
    return out


def get_filled_orders(order_type, symbol, startTime, limit=1000, *, strict=False):
    try:
        if order_type is not None:
            order_type = order_type.upper()
        sym.validate_symbols(symbol)
        sym.validate_ordertype(order_type)

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

        for t in filtered:
            key = t['orderId']
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
    if max_age_seconds is None or isinstance(max_age_seconds, bool):
        return []
    try:
        max_age_val = float(max_age_seconds)
    except (TypeError, ValueError, OverflowError):
        return []
    if not math.isfinite(max_age_val) or max_age_val < 0:
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

    if order_type is not None:
        order_type = order_type.upper()
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
        orders_for_symbol = [
            dict(order) for order in raw_bucket
            if isinstance(order, dict)
        ]
    if not orders_for_symbol:
        return []

    #print(f" orders_for_symbol {orders_for_symbol}")
    try:
        if max_age_seconds is None or isinstance(max_age_seconds, bool):
            return []
        max_age_val = float(max_age_seconds)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(max_age_val) or max_age_val < 0:
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
    
    trades = get_trade_orders(order_type=None, symbol=symbol, max_age_seconds=period_seconds)

    stats = {
        'BUY': {'total_quantity': 0, 'total_value': 0, 'trade_count': 0},
        'SELL': {'total_quantity': 0, 'total_value': 0, 'trade_count': 0}
    }

    for t in trades:
        side = str(t.get('side') or '').upper()
        if side in stats:
            stats[side]['total_quantity'] += t['quantity']
            stats[side]['total_value'] += t['price'] * t['quantity']
            stats[side]['trade_count'] += 1

    # Round the accumulated values once, after all rows are included.
    for side in stats:
        stats[side]['total_quantity'] = round(stats[side]['total_quantity'], 8)
        stats[side]['total_value'] = round(stats[side]['total_value'], 8)

    return stats
