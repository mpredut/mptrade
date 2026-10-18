
import time
from collections import defaultdict

# BINANCE
from binance.exceptions import BinanceAPIException

# MYLIB
from .bapi_client import client
import symbols as sym


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
        from_id = batch[-1]["id"] + 1   # Continue from the next fill ID.
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

        filtered = [
            {
                'orderId': trade.get('orderId'),
                'price': float(trade.get('price', 0)),
                'quantity': float(trade.get('qty', 0)),
                'timestamp': trade.get('time'),  # ms
                'side': 'BUY' if trade.get('isBuyer') else 'SELL'
            }
            for trade in trades
            if startTime <= trade.get('time', 0) <= end_time
            and (order_type is None or 
                 (order_type.upper() == "BUY" and trade.get('isBuyer')) or
                 (order_type.upper() == "SELL" and not trade.get('isBuyer')))
        ]

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

        result = [
            {
                'orderId': oid,
                'price': round(data['price_qty'] / data['quantity'], 8) if data['quantity'] else 0,
                'quantity': round(data['quantity'], 8),
                'timestamp': data['timestamp'],
                'side': data['side']
            }
            for oid, data in grouped.items()
        ]

        for r in result[:5]:
            print(r)

        return result


    except Exception as e:
        print(f"Unexpected error in get_filled_orders: {e}")
        if strict:
            raise
        return []



def get_recent_filled_orders(order_type, symbol, max_age_seconds):
    # Convert duration to an absolute millisecond start time. Passing duration directly
    # would query from approximately 1970 and return incorrect data.
    start_time_ms = int(time.time() * 1000) - int(max_age_seconds * 1000)
    all_filled_orders = get_filled_orders(order_type, symbol, start_time_ms)
    recent_filled_orders = []
    current_time = time.time()
    if(len(all_filled_orders) < 1) :
        return []
    
    print(f"have {len(all_filled_orders)} orders. ignore oldest.")
    for order in all_filled_orders:
        if current_time - order['timestamp']/1000 <= max_age_seconds:
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
    with cache_order_manager.lock:
        orders_for_symbol = [
            dict(order) for order in cache_order_manager.cache.get(symbol, [])
            if isinstance(order, dict)
        ]
    if not orders_for_symbol:
        return []

    #print(f" orders_for_symbol {orders_for_symbol}")
    current_time_ms = int(time.time() * 1000)
    max_age_ms = max_age_seconds * 1000  # convert to ms

    filtered_orders = [
        {
            'orderId': order.get('orderId'),
            'price': float(order.get('price', 0)),
            'quantity': float(order.get('quantity', 0)),  # This cache uses ``quantity``, not ``origQty``.
            'qty': float(order.get('quantity', 0)),       # Fill-compatible consumer alias.
            'timestamp': order.get('timestamp'),          # Already milliseconds in cache.
            'side': order.get('side', '').upper()
        }
        for order in orders_for_symbol
        if (order_type is None or order.get('side', '').upper() == order_type)
        and (current_time_ms - order.get('timestamp', 0)) <= max_age_ms
    ]

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
        side = t['side'].upper()
        if side in stats:
            stats[side]['total_quantity'] += t['quantity']
            stats[side]['total_value'] += t['price'] * t['quantity']
            stats[side]['trade_count'] += 1

    # Round the accumulated values once, after all rows are included.
    for side in stats:
        stats[side]['total_quantity'] = round(stats[side]['total_quantity'], 8)
        stats[side]['total_value'] = round(stats[side]['total_value'], 8)

    return stats
