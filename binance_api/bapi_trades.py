
import os
import time
import math
import json
import threading
from collections import defaultdict
from state_io import atomic_write_json

# Repository imports.
import utils as u
import symbols as sym
from . import bapi as api
from providers.base import (
    extract_order_fee,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)

# 
# Process-wide trade cache.
#
cache_trade_manager = None
_cache_trade_manager_lock = threading.Lock()

def init_cache_trade_manager():
    """Build the cache on the first call that actually consumes trades.

    Backtests, configuration tools, and tests import this module, so importing it must
    not start Binance polling merely because the functions were loaded.
    """
    global cache_trade_manager
    if cache_trade_manager is not None:
        return cache_trade_manager
    with _cache_trade_manager_lock:
        if cache_trade_manager is None:
            import cacheManager as cm
            # Trading processes are read-only consumers. Only the central
            # cacheManager process may poll Binance and publish cache versions.
            cache_trade_manager = cm.get_cache_manager("Trade", start_sync=False)
    return cache_trade_manager

trade_cache = []

#######
#######      get_my_trades     #######
#######

def aggregate_trades(trades):
    aggregated_trades = defaultdict(lambda: {
        'symbol': '', 'price': '', 'qty': 0.0, 'quoteQty': 0.0, 'weightedCost': 0.0,
        'count': 0, 'commission': 0.0, 'commissionAsset': '', 'time': 0,
        'isBuyer': None, 'isMaker': None, 'isBestMatch': None, 'id': 0,
    })

    # Group fills by order ID.
    for trade in trades or ():
        if not isinstance(trade, dict):
            continue
        orderId = trade['orderId'] if trade.get('orderId') is not None else extract_order_id(trade)
        if orderId is None:
            continue
        qty = extract_order_qty(trade)
        px = extract_order_price(trade)
        if qty <= 0 or px <= 0:
            continue
        try:
            raw_qq = float(trade.get('quoteQty', 0.0) or 0.0)
            quote_qty = raw_qq if math.isfinite(raw_qq) and raw_qq > 0 else (px * qty)
        except (TypeError, ValueError):
            quote_qty = px * qty
        ts_ms = extract_timestamp_ms(trade, default=0) or 0
        side = extract_order_side(trade)
        is_buyer = trade.get('isBuyer') if 'isBuyer' in trade else (side == 'BUY')

        # Aggregate fields belonging to the same order.
        if aggregated_trades[orderId]['count'] == 0:
            aggregated_trades[orderId]['id'] = trade.get('id', 0)
        aggregated_trades[orderId]['count'] += 1
        aggregated_trades[orderId]['orderId'] = orderId
        aggregated_trades[orderId]['symbol'] = trade.get('symbol', '')
        raw_px = trade.get('price')
        try:
            has_pos_raw_px = raw_px is not None and not isinstance(raw_px, bool) and float(raw_px) > 0
        except (TypeError, ValueError):
            has_pos_raw_px = False
        aggregated_trades[orderId]['price'] = (
            raw_px
            if has_pos_raw_px
            else (
                (
                    (f"{px:.8f}" if px >= 1e-8 else f"{px:.12f}")
                    .rstrip("0")
                    .rstrip(".")
                    or str(px)
                )
                if px > 0
                else (raw_px if raw_px is not None else px)
            )
        )
        aggregated_trades[orderId]['qty'] += qty
        aggregated_trades[orderId]['quoteQty'] += quote_qty
        aggregated_trades[orderId]['weightedCost'] += px * qty
        fee_info = extract_order_fee(trade)
        try:
            comm_val = float(trade.get('commission', 0.0) or 0.0)
            if not math.isfinite(comm_val) or comm_val < 0:
                comm_val = 0.0
        except (TypeError, ValueError):
            comm_val = 0.0
        if comm_val == 0.0 and fee_info is not None:
            comm_val = fee_info['fee']
        comm_asset = trade.get('commissionAsset') or (fee_info['fee_asset'] if fee_info is not None else '')
        aggregated_trades[orderId]['commission'] += comm_val
        aggregated_trades[orderId]['commissionAsset'] = comm_asset
        aggregated_trades[orderId]['time'] = max(aggregated_trades[orderId]['time'], ts_ms)
        aggregated_trades[orderId]['isBuyer'] = is_buyer
        aggregated_trades[orderId]['isMaker'] = trade.get('isMaker')
        aggregated_trades[orderId]['isBestMatch'] = trade.get('isBestMatch')

    # Build the aggregate list.
    aggregated_list = []
    for aggregated in aggregated_trades.values():
        total_qty = aggregated['qty']
        if aggregated['count'] > 1 and total_qty > 0:
            vwap = aggregated['weightedCost'] / total_qty
            price_val = (
                (
                    (f"{vwap:.8f}" if vwap >= 1e-8 else f"{vwap:.12f}")
                    .rstrip("0")
                    .rstrip(".")
                    or str(vwap)
                )
                if vwap > 0
                else "0"
            )
        else:
            price_val = aggregated['price']
        qty_str = (
            f"{total_qty:.8f}"
            if total_qty <= 0 or total_qty >= 1e-8
            else (f"{total_qty:.12f}".rstrip("0").rstrip(".") or str(total_qty))
        )
        side = "BUY" if aggregated['isBuyer'] else "SELL"
        aggregated_list.append({
            'id': aggregated['id'],
            'orderId': aggregated['orderId'],
            'symbol': aggregated['symbol'],
            'orderListId': -1,
            'price': price_val,
            'qty': qty_str,  # Preserve the eight-decimal format (or sub-satoshi precision).
            'quantity': qty_str,
            'quoteQty': f"{aggregated['quoteQty']:.8f}",
            'commission': f"{aggregated['commission']:.8f}",
            'commissionAsset': aggregated['commissionAsset'],
            'time': aggregated['time'],
            'timestamp': aggregated['time'],
            'isBuyer': aggregated['isBuyer'],
            'side': side,
            'isMaker': aggregated['isMaker'],
            'isBestMatch': aggregated['isBestMatch']
        })

    return aggregated_list
    
def get_my_trades_24(order_type, symbol, days_ago=0, limit=1000):
    
    if order_type is not None:
        order_type = order_type.upper()
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    
    
    all_trades = []
    try:
        current_time = int(time.time() * 1000)
        
        # Calculate the requested historical day's start and end times.
        end_time = current_time - days_ago * 24 * 60 * 60 * 1000
        start_time = end_time - 24 * 60 * 60 * 1000  # Twenty-four hours before end_time.

        # Binance returns at most ``limit`` fills per request. Advance start_time to
        # the last fill time plus one millisecond until a short page arrives. A rare
        # boundary case remains: multiple fills with the exact same millisecond at a
        # page boundary can be skipped. ``paginate_my_trades`` uses ``fromId`` and is
        # not vulnerable to that timestamp collision.
        while start_time < end_time:

            trades = api.client.get_my_trades(symbol=symbol, limit=limit, startTime=start_time, endTime=end_time)
            time.sleep(2)
                        
            if not trades:
                break

            if order_type in ("BUY", "SELL"):
                filtered_trades = [trade for trade in trades if extract_order_side(trade) == order_type]
            else:
                filtered_trades = trades

            all_trades.extend(filtered_trades)
            
            if len(trades) < limit:
                break

            # Continue one millisecond after the newest fill to avoid duplicates.
            next_start = (extract_timestamp_ms(trades[-1], default=0) or 0) + 1
            if next_start <= start_time:
                break
            start_time = next_start
        
        
        # aggregated_trades = {}

        # for trade in all_trades:
            # oid = trade['orderId']
            # if oid not in aggregated_trades:
                # aggregated_trades[oid] = {
                    # 'symbol': trade['symbol'],
                    # 'id': oid,
                    # 'orderId': oid,
                    # 'isBuyer': trade['isBuyer'],
                    # 'side': 'BUY' if trade['isBuyer'] else 'SELL',
                    # 'qty': 0.0,
                    # 'price': 0.0,
                    # 'time': trade['time'],
                    # 'trades': []
                # }
            # # Add the fill to the internal list.
            # aggregated_trades[oid]['trades'].append(trade)
            # # Update total quantity and cost.
            # qty = float(trade['qty'])
            # price = float(trade['price'])
            # aggregated_trades[oid]['qty'] += qty
            # aggregated_trades[oid]['price'] += qty * price

        # An average price could also be calculated here.
        # for agg in aggregated_trades.values():
        #    agg['avg_price'] = agg['price'] / agg['qty'] if agg['qty'] else 0
            
        ##########
        latest_trades = {}
        for trade in all_trades:
            fill_key = trade.get('id', trade.get('orderId'))
            ts_ms = extract_timestamp_ms(trade, default=0) or 0

            # Deduplicate by individual fill ID (falling back to orderId when id is absent).
            if (
                fill_key not in latest_trades
                or ts_ms >= (extract_timestamp_ms(latest_trades[fill_key], default=0) or 0)
            ):
                latest_trades[fill_key] = trade

        return list(latest_trades.values())  # Return a list, not a mapping.
        

    except Exception as e:
        print(f"get_my_trades_24: An error occurred: {e}")
        return []


def get_my_trades(order_type, symbol, backdays: int = 3, limit=1000):
    
    if order_type is not None:
        order_type = order_type.upper()
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    
    all_trades = []
    
    #try:
    for days_ago in range(backdays + 1):
        print(f"[{symbol}] get_my_trades: Fetching trades for day {days_ago:03d}... ", end=" ")
        trades = get_my_trades_24(order_type, symbol, days_ago=days_ago, limit=limit)
        
        if not trades:
            continue
            # retry from cache .....
            #trades = get_trade_orders_for_day_24(order_type, symbol, days_ago)
            #if not trades:
                #print(f"No trades found for day {days_ago:03d}.")
                #continue
        
        print(f" [{len(trades)}] found ")
        if order_type in ("BUY", "SELL"):
            filtered_trades = [trade for trade in trades if extract_order_side(trade) == order_type]
        else:
            filtered_trades = trades
            
        all_trades.extend(filtered_trades)

    print(f"")
    return all_trades

    #except Exception as e:
    #    print(f"get_my_trades: An error occurred: {e}") #3600 * 24 * 7
    #    return get_trade_orders(order_type, symbol, (backdays + 1) * 24 * 3600 )
        
        
        
def get_my_trades_simple(order_type, symbol, backdays=3, limit=1000):

    if order_type is not None:
        order_type = order_type.upper()
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)

    try:
        from . import bapi_allorders as apiorders
        # Paginate the entire window with ``fromId`` so days containing more than 1,000
        # fills are not truncated.
        start_time = int(time.time() * 1000) - (backdays + 1) * 24 * 60 * 60 * 1000
        all_trades = apiorders.paginate_my_trades(api.client, symbol, start_time, limit)

        if order_type in ("BUY", "SELL"):
            all_trades = [t for t in all_trades if extract_order_side(t) == order_type]

        # Deduplicate by individual fill ID (falling back to orderId only when id is absent)
        # so multi-fill orders do not lose earlier partial fills.
        latest_trades = {}
        for trade in all_trades:
            fill_key = trade.get('id', trade.get('orderId'))
            ts_ms = extract_timestamp_ms(trade, default=0) or 0
            if (
                fill_key not in latest_trades
                or ts_ms >= (extract_timestamp_ms(latest_trades[fill_key], default=0) or 0)
            ):
                latest_trades[fill_key] = trade

        return list(latest_trades.values())

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Append newly fetched fills to an existing file.
def save_trades_to_file(order_type, symbol, filename, limit=1000, years_to_keep=2):
    if order_type is not None:
        order_type = order_type.upper()
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)

    print(f"save_trades_to_file")
    print(f"{symbol} symbol and order_type {order_type} ")
     
    # Initialize the main collection as a list.
    all_trades_list = []

    # Load existing data when the file exists.
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                all_trades_list = json.load(f)  # The on-disk structure is a list.
                print(f"Loaded {len(all_trades_list)} total trades from {filename}.")
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON from {filename}. Starting with an empty list.")
                all_trades_list = []

    # Select existing fills for the current symbol.
    existing_trades = [trade for trade in all_trades_list if isinstance(trade, dict) and trade.get('symbol') == symbol]
    print(f"Already saved in file {len(existing_trades)} trades for {symbol}")

    # Calculate the retention cutoff.
    current_time_ms = int(time.time() * 1000)
    cutoff_time_ms = current_time_ms - (years_to_keep * 365 * 24 * 60 * 60 * 1000)

    # Remove fills older than the requested retention period.
    filtered_existing_trades = [
        trade for trade in existing_trades
        if (extract_timestamp_ms(trade, default=0) or 0) > cutoff_time_ms
    ]
    print(f"Preserve ony {len(filtered_existing_trades)} trades for {symbol} , "
        f"rest are too old , less than {years_to_keep * 365} days")

    # Find the newest retained fill for this symbol.
    most_recent_trade_time = max(
        ((extract_timestamp_ms(trade, default=0) or 0) for trade in filtered_existing_trades),
        default=0,
    )
    print(f"Most recent saved trade time is {u.timestampToTime(most_recent_trade_time)} for {symbol}")
    
    # Calculate how many days must be fetched.
    if most_recent_trade_time == 0:
        backdays = years_to_keep * 365
    else:
        time_diff_ms = current_time_ms - most_recent_trade_time
        backdays = time_diff_ms // (24 * 60 * 60 * 1000) + 1

    print(f"Fetching trades from the last {backdays} days for {symbol}, order type {order_type}.")

    # Fetch new fills.
    new_trades = get_my_trades_simple(order_type, symbol, backdays=math.ceil(backdays), limit=limit)

    # Remove duplicates by fill ID.
    # new_trades = [trade for trade in new_trades if trade['time'] > most_recent_trade_time]    #
    existing_trade_ids = {
        trade.get('id', trade.get('orderId'))
        for trade in filtered_existing_trades
        if trade.get('id', trade.get('orderId')) is not None
    }
    unique_new_trades = [
        trade for trade in new_trades
        if trade.get('id', trade.get('orderId')) not in existing_trade_ids
    ]

    if unique_new_trades:
        print(f"Found {len(unique_new_trades)} new trades for {symbol}.")
        
        # Merge retained and new fills in chronological order.
        updated_trades = filtered_existing_trades + unique_new_trades
        updated_trades.sort(key=lambda x: extract_timestamp_ms(x, default=0) or 0)

        # Replace this symbol's entries in the main list.
        all_trades_list = [trade for trade in all_trades_list if isinstance(trade, dict) and trade.get('symbol') != symbol]
        all_trades_list.extend(updated_trades)

        # Persist the updated list.
        atomic_write_json(filename, all_trades_list)

        print(f"Updated file with {len(updated_trades)} new trades for {symbol}.")
    else:
        print(f"No new trades found to save for {symbol}.")
    print(f"Total trades in file: {len(all_trades_list)}.")



def load_trades_from_file(filename):

    global trade_cache

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                trade_cache = json.load(f)
                print(f"Cache loaded with {len(trade_cache)} trades.")
            except json.JSONDecodeError:
                print("Error reading file.")
                trade_cache = []
    else:
        print(f"File {filename} not found.")
        trade_cache = []
        
    print(set(trade.get('symbol') for trade in trade_cache if isinstance(trade, dict) and trade.get('symbol')))


# Return recent BUY or SELL fills from the manager cache.
def get_trade_orders(order_type, symbol, max_age_seconds):
   
    if order_type is not None:
        order_type = order_type.upper()
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    
    manager = init_cache_trade_manager()
    sync_reader = getattr(manager, "sync_account_reader_if_needed", None)
    if callable(sync_reader):
        sync_reader()

    # Copy one coherent snapshot while the manager may reload a new generation.
    sym_u = str(symbol or "").strip().upper()
    sym_clean = sym_u.replace("/", "")
    with manager.lock:
        raw_bucket = (
            manager.cache.get(symbol)
            or manager.cache.get(sym_u)
            or manager.cache.get(sym_clean)
            or []
        )
        trades = [
            dict(trade) for trade in raw_bucket
            if isinstance(trade, dict)
        ]
    if not trades:
        return []
        
    try:
        if max_age_seconds is None or isinstance(max_age_seconds, bool):
            return []
        max_age_val = float(max_age_seconds)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(max_age_val) or max_age_val < 0:
        return []
    current_time_ms = int(time.time() * 1000)
    max_age_ms = max_age_val * 1000.0  # Convert to milliseconds.
    
    filtered_trades = []
    for trade in trades:
        side = extract_order_side(trade)
        ts_ms = extract_timestamp_ms(trade, default=0) or 0
        age_ms = current_time_ms - ts_ms
        if ts_ms > 0 and (order_type is None or side == order_type) and -5_000 <= age_ms <= max_age_ms:
            px = extract_order_price(trade)
            qty = extract_order_qty(trade)
            if px <= 0 or qty <= 0:
                continue
            is_buyer = side == "BUY"
            filtered_trades.append({
                'symbol': trade.get('symbol', symbol),
                'id': trade.get('id'),
                'orderId': trade.get('orderId'),
                'orderListId': trade.get('orderId'),
                'price': px,
                'qty': qty,
                'quantity': qty,
                'time': ts_ms,
                'timestamp': ts_ms,
                'isBuyer': is_buyer,
                'side': side,
            })

    #  filtered_trades.sort(key=lambda x: x['price'])

    return filtered_trades
