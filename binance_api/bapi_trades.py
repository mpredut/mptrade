
import time
import math
import json
import threading
from state_io import atomic_write_json

# Repository imports.
import utils as u
import symbols as sym
from . import bapi as api

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
        'symbol': '', 'price': '', 'qty': 0, 'quoteQty': 0, 'commission': 0, 'commissionAsset': '', 'time': 0, 'isBuyer': None, 'isMaker': None, 'isBestMatch': None, 'id': 0
    })
    
    # Group fills by order ID.
    for trade in trades:
        orderId = trade['orderId']
        
        # Aggregate fields belonging to the same order.
        aggregated_trades[orderId]['id'] = trade['id']  # Retain the first fill ID as a reference.
        aggregated_trades[orderId]['orderId'] = orderId
        aggregated_trades[orderId]['symbol'] = trade['symbol']
        aggregated_trades[orderId]['price'] = trade['price']
        aggregated_trades[orderId]['qty'] += float(trade['qty'])
        aggregated_trades[orderId]['quoteQty'] += float(trade['quoteQty'])
        aggregated_trades[orderId]['commission'] += float(trade['commission'])
        aggregated_trades[orderId]['commissionAsset'] = trade['commissionAsset']
        aggregated_trades[orderId]['time'] = max(aggregated_trades[orderId]['time'], trade['time'])  # Use the latest fill time.
        aggregated_trades[orderId]['isBuyer'] = trade['isBuyer']
        aggregated_trades[orderId]['isMaker'] = trade['isMaker']
        aggregated_trades[orderId]['isBestMatch'] = trade['isBestMatch']

    # Build the aggregate list.
    aggregated_list = []
    for aggregated in aggregated_trades.values():
        aggregated_list.append({
            'id': aggregated['id'],
            'orderId': aggregated['orderId'],
            'symbol': aggregated['symbol'],
            'orderListId': -1,
            'price': aggregated['price'],
            'qty': f"{aggregated['qty']:.8f}",  # Preserve the eight-decimal format.
            'quoteQty': f"{aggregated['quoteQty']:.8f}",
            'commission': f"{aggregated['commission']:.8f}",
            'commissionAsset': aggregated['commissionAsset'],
            'time': aggregated['time'],
            'isBuyer': aggregated['isBuyer'],
            'isMaker': aggregated['isMaker'],
            'isBestMatch': aggregated['isBestMatch']
        })

    return aggregated_list
    
def get_my_trades_24(order_type, symbol, days_ago=0, limit=1000):
    
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

            if order_type == "BUY":
                filtered_trades = [trade for trade in trades if trade['isBuyer']]
            elif order_type == "SELL":
                filtered_trades = [trade for trade in trades if not trade['isBuyer']]
            else:
                filtered_trades = trades

            all_trades.extend(filtered_trades)
            
            if len(trades) < limit:
                break

            # Continue one millisecond after the newest fill to avoid duplicates.
            start_time = trades[-1]['time'] + 1
        
        
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
            order_id = trade['orderId']
            
            # Retain the newest fill for each order ID.
            if order_id not in latest_trades or trade['time'] > latest_trades[order_id]['time']:
               latest_trades[order_id] = trade

        return list(latest_trades.values())  # Return a list, not a mapping.
        
        return list(aggregated_trades.values())
        

    except Exception as e:
        print(f"get_my_trades_24: An error occurred: {e}")
        return []


def get_my_trades(order_type, symbol, backdays: int = 3, limit=1000):
    
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
        #filtered_trades = [trade for trade in trades if trade['isBuyer'] == (order_type == "BUY")]
        if order_type == "BUY":
            filtered_trades = [trade for trade in trades if trade['isBuyer']]
        elif order_type == "SELL":
            filtered_trades = [trade for trade in trades if not trade['isBuyer']]
        else:
            filtered_trades = trades
            
        all_trades.extend(filtered_trades)

    print(f"")
    return all_trades

    #except Exception as e:
    #    print(f"get_my_trades: An error occurred: {e}") #3600 * 24 * 7
    #    return get_trade_orders(order_type, symbol, (backdays + 1) * 24 * 3600 )
        
        
        
def get_my_trades_simple(order_type, symbol, backdays=3, limit=1000):

    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)

    try:
        from . import bapi_allorders as apiorders
        # Paginate the entire window with ``fromId`` so days containing more than 1,000
        # fills are not truncated.
        start_time = int(time.time() * 1000) - (backdays + 1) * 24 * 60 * 60 * 1000
        all_trades = apiorders.paginate_my_trades(api.client, symbol, start_time, limit)

        if order_type == "BUY":
            all_trades = [t for t in all_trades if t['isBuyer']]
        elif order_type == "SELL":
            all_trades = [t for t in all_trades if not t['isBuyer']]

        # Deduplicate by order ID while retaining the newest fill.
        latest_trades = {}
        for trade in all_trades:
            order_id = trade['orderId']
            if order_id not in latest_trades or trade['time'] > latest_trades[order_id]['time']:
                latest_trades[order_id] = trade

        return list(latest_trades.values())

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Append newly fetched fills to an existing file.
def save_trades_to_file(order_type, symbol, filename, limit=1000, years_to_keep=2):
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
    existing_trades = [trade for trade in all_trades_list if trade['symbol'] == symbol]
    print(f"Already saved in file {len(existing_trades)} trades for {symbol}")

    # Calculate the retention cutoff.
    current_time_ms = int(time.time() * 1000)
    cutoff_time_ms = current_time_ms - (years_to_keep * 365 * 24 * 60 * 60 * 1000)

    # Remove fills older than the requested retention period.
    filtered_existing_trades = [trade for trade in existing_trades if trade['time'] > cutoff_time_ms]
    print(f"Preserve ony {len(filtered_existing_trades)} trades for {symbol} , "
        f"rest are too old , less than {u.secondsToDays(cutoff_time_ms)} days")

    # Find the newest retained fill for this symbol.
    most_recent_trade_time = max((trade['time'] for trade in filtered_existing_trades), default=0)
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
    existing_trade_ids = {trade['id'] for trade in filtered_existing_trades}
    unique_new_trades = [trade for trade in new_trades if trade['id'] not in existing_trade_ids]

    if unique_new_trades:
        print(f"Found {len(unique_new_trades)} new trades for {symbol}.")
        
        # Merge retained and new fills in chronological order.
        updated_trades = filtered_existing_trades + unique_new_trades
        updated_trades.sort(key=lambda x: x['time'])

        # Replace this symbol's entries in the main list.
        all_trades_list = [trade for trade in all_trades_list if trade['symbol'] != symbol]
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
        
    print(set(trade['symbol'] for trade in trade_cache))


# Return recent BUY or SELL fills from the manager cache.
def get_trade_orders(order_type, symbol, max_age_seconds):
   
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    
    manager = init_cache_trade_manager()

    # Copy one coherent snapshot while the manager may reload a new generation.
    with manager.lock:
        trades = [dict(trade) for trade in manager.cache.get(symbol, [])]
    if not trades:
        return []
        
    current_time_ms = int(time.time() * 1000)
    max_age_ms = max_age_seconds * 1000  # Convert to milliseconds.
    
    filtered_trades = [
        {
            'symbol': trade['symbol'],
            'id': trade['id'],
            'orderId': trade['orderId'],
            'orderListId': trade['orderId'],
            'price': float(trade['price']),
            'qty': float(trade['qty']),
            #'quoteQty': float(trade['quoteQty']),
            #'commission': float(trade['commission']),
            #'commissionAsset': trade['commissionAsset'],
            'time': trade['time'],
            'isBuyer': trade['isBuyer'],
            #'isMaker': trade['isMaker'],
            #'isBestMatch': trade['isBestMatch']
        }
        for trade in trades
        #if trade['symbol'] == symbol
        if (order_type is None or trade['isBuyer'] == (order_type == "BUY"))  # Apply side filtering only when requested.
        and (current_time_ms - trade['time']) <= max_age_ms
    ]

    #  filtered_trades.sort(key=lambda x: x['price'])
    
    return filtered_trades
