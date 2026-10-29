
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
from botcore import safe_int, safe_nonneg_float, safe_pos_float, safe_nonneg_int
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

_normalize_side = api._normalize_side


def _valid_fill_id(val):
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer() or val < 0:
            return None
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.startswith("-") or s.lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}:
            return None
        if s.endswith(".0") and s[:-2].lstrip("+").isdigit():
            s = s[:-2]
        if s.startswith("+") and s[1:].isdigit():
            s = s[1:]
        return s
    if isinstance(val, int):
        return val if val >= 0 else None
    return None


def _dedup_trades_by_fill_id(all_trades):
    """Deduplicate trade fills by individual fill ID (falling back to orderId when absent)."""
    latest_trades = {}
    for idx, trade in enumerate(all_trades or ()):
        if not isinstance(trade, dict):
            continue
        raw_key = _valid_fill_id(trade.get('id'))
        if raw_key is None:
            raw_key = _valid_fill_id(trade.get('orderId'))
        fill_key = raw_key if raw_key is not None else f"_unkeyed_{idx}"
        ts_ms = extract_timestamp_ms(trade, default=0) or 0
        if (
            fill_key not in latest_trades
            or ts_ms >= (extract_timestamp_ms(latest_trades[fill_key], default=0) or 0)
        ):
            latest_trades[fill_key] = trade
    return list(latest_trades.values())


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
        raw_oid = _valid_fill_id(trade.get('orderId'))
        if raw_oid in (0, "0", "0.0", "-0", "+0"):
            raw_oid = None
        orderId = raw_oid if raw_oid is not None else extract_order_id(trade)
        if orderId is None or isinstance(orderId, bool) or not str(orderId).strip():
            continue
        qty = extract_order_qty(trade)
        px = extract_order_price(trade)
        if qty <= 0 or px <= 0:
            continue
        raw_qq = safe_pos_float(trade.get('quoteQty'), default=None)
        quote_qty = raw_qq if raw_qq is not None else (px * qty)
        ts_ms = extract_timestamp_ms(trade, default=0) or 0
        side = extract_order_side(trade)
        raw_is_buyer = trade.get('isBuyer')
        is_buyer = raw_is_buyer if isinstance(raw_is_buyer, bool) else (side == 'BUY')

        # Aggregate fields belonging to the same order.
        raw_tid = _valid_fill_id(trade.get('id'))
        if aggregated_trades[orderId]['count'] == 0 or not aggregated_trades[orderId]['id']:
            aggregated_trades[orderId]['id'] = 0 if raw_tid is None else raw_tid
        aggregated_trades[orderId]['count'] += 1
        aggregated_trades[orderId]['orderId'] = orderId
        raw_sym = trade.get('symbol', '')
        clean_sym = raw_sym.strip() if isinstance(raw_sym, str) else ''
        if clean_sym or not aggregated_trades[orderId]['symbol']:
            aggregated_trades[orderId]['symbol'] = clean_sym
        raw_px = trade.get('price')
        has_pos_raw_px = safe_pos_float(raw_px, default=None) is not None
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
        comm_val = safe_nonneg_float(trade.get('commission'), default=0.0) or 0.0
        if comm_val == 0.0 and fee_info is not None:
            comm_val = fee_info['fee']
        raw_comm_asset = trade.get('commissionAsset')
        comm_asset = (
            raw_comm_asset.strip()
            if (isinstance(raw_comm_asset, str) and raw_comm_asset.strip())
            else (fee_info['fee_asset'] if fee_info is not None else '')
        )
        aggregated_trades[orderId]['commission'] += comm_val
        if comm_asset or not aggregated_trades[orderId]['commissionAsset']:
            aggregated_trades[orderId]['commissionAsset'] = comm_asset
        aggregated_trades[orderId]['time'] = max(aggregated_trades[orderId]['time'], ts_ms)
        aggregated_trades[orderId]['isBuyer'] = is_buyer
        raw_maker = trade.get('isMaker')
        raw_best = trade.get('isBestMatch')
        aggregated_trades[orderId]['isMaker'] = raw_maker if isinstance(raw_maker, bool) else None
        aggregated_trades[orderId]['isBestMatch'] = raw_best if isinstance(raw_best, bool) else None

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
    
def _clean_non_neg_int(val, default: int = 0) -> int:
    return safe_nonneg_int(val, default=default) or default


def get_my_trades_24(order_type, symbol, days_ago=0, limit=1000):
    
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    days_ago = _clean_non_neg_int(days_ago, default=0)
    limit = min(1000, _clean_non_neg_int(limit, default=1000) or 1000)
    
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

        return _dedup_trades_by_fill_id(all_trades)
        

    except Exception as e:
        print(f"get_my_trades_24: An error occurred: {e}")
        return []


def get_my_trades(order_type, symbol, backdays: int = 3, limit=1000):
    
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    backdays = _clean_non_neg_int(backdays, default=3)
    limit = min(1000, _clean_non_neg_int(limit, default=1000) or 1000)
    
    all_trades = []
    
    for days_ago in range(backdays + 1):
        print(f"[{symbol}] get_my_trades: Fetching trades for day {days_ago:03d}... ", end=" ")
        trades = get_my_trades_24(order_type, symbol, days_ago=days_ago, limit=limit)
        
        if not trades:
            continue
        
        print(f" [{len(trades)}] found ")
        if order_type in ("BUY", "SELL"):
            filtered_trades = [trade for trade in trades if extract_order_side(trade) == order_type]
        else:
            filtered_trades = trades
            
        all_trades.extend(filtered_trades)

    print(f"")
    return all_trades
        
        
        
def get_my_trades_simple(order_type, symbol, backdays=3, limit=1000):

    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    backdays = _clean_non_neg_int(backdays, default=3)
    limit = min(1000, _clean_non_neg_int(limit, default=1000) or 1000)

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
        return _dedup_trades_by_fill_id(all_trades)

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Append newly fetched fills to an existing file.
def save_trades_to_file(order_type, symbol, filename, limit=1000, years_to_keep=2):
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    symbol = str(symbol).strip().upper().replace("/", "")
    limit = min(1000, _clean_non_neg_int(limit, default=1000) or 1000)
    years_to_keep = _clean_non_neg_int(years_to_keep, default=2) or 2

    print(f"save_trades_to_file")
    print(f"{symbol} symbol and order_type {order_type} ")
     
    # Initialize the main collection as a list.
    all_trades_list = []

    # Load existing data when the file exists.
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded = json.load(f)  # The on-disk structure is a list.
                all_trades_list = loaded if isinstance(loaded, list) else []
                print(f"Loaded {len(all_trades_list)} total trades from {filename}.")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            print(f"Warning: Failed to decode JSON from {filename}. Starting with an empty list.")
            all_trades_list = []

    # Select existing fills for the current symbol.
    existing_trades = [
        trade for trade in all_trades_list
        if isinstance(trade, dict) and str(trade.get('symbol') or '').strip().upper().replace('/', '') == symbol
    ]
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
        time_diff_ms = max(0, current_time_ms - most_recent_trade_time)
        backdays = max(1, time_diff_ms // (24 * 60 * 60 * 1000) + 1)

    print(f"Fetching trades from the last {backdays} days for {symbol}, order type {order_type}.")

    # Fetch new fills.
    new_trades = get_my_trades_simple(order_type, symbol, backdays=math.ceil(backdays), limit=limit)

    # Remove duplicates by fill ID.
    # new_trades = [trade for trade in new_trades if trade['time'] > most_recent_trade_time]    #
    def _trade_fill_key(trade_obj):
        if not isinstance(trade_obj, dict):
            return None
        rk = _valid_fill_id(trade_obj.get('id'))
        if rk is None:
            rk = _valid_fill_id(trade_obj.get('orderId'))
        return rk

    existing_trade_ids = {
        k for k in (_trade_fill_key(trade) for trade in filtered_existing_trades)
        if k is not None
    }
    unique_new_trades = [
        trade for trade in new_trades
        if isinstance(trade, dict) and (
            _trade_fill_key(trade) is None or _trade_fill_key(trade) not in existing_trade_ids
        )
    ]

    if unique_new_trades:
        print(f"Found {len(unique_new_trades)} new trades for {symbol}.")
        
        # Merge retained and new fills in chronological order.
        updated_trades = filtered_existing_trades + unique_new_trades
        updated_trades.sort(key=lambda x: extract_timestamp_ms(x, default=0) or 0)

        # Replace this symbol's entries in the main list.
        all_trades_list = [
            trade for trade in all_trades_list
            if isinstance(trade, dict) and str(trade.get('symbol') or '').strip().upper().replace('/', '') != symbol
        ]
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
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                trade_cache = loaded if isinstance(loaded, list) else []
                print(f"Cache loaded with {len(trade_cache)} trades.")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            print("Error reading file.")
            trade_cache = []
    else:
        print(f"File {filename} not found.")
        trade_cache = []
        
    print(set(trade.get('symbol') for trade in trade_cache if isinstance(trade, dict) and trade.get('symbol')))


# Return recent BUY or SELL fills from the manager cache.
def get_trade_orders(order_type, symbol, max_age_seconds):
   
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    
    manager = init_cache_trade_manager()
    sync_reader = getattr(manager, "sync_account_reader_if_needed", None)
    if callable(sync_reader):
        sync_reader()

    # Copy one coherent snapshot while the manager may reload a new generation.
    sym_u = str(symbol or "").strip().upper()
    sym_clean = sym_u.replace("/", "").replace("-", "").replace("_", "")
    with manager.lock:
        raw_bucket = (
            manager.cache.get(symbol)
            or manager.cache.get(sym_u)
            or manager.cache.get(sym_clean)
            or []
        )
        if not isinstance(raw_bucket, (list, tuple)):
            raw_bucket = []
        trades = [
            dict(trade) for trade in raw_bucket
            if isinstance(trade, dict)
        ]
    if not trades:
        return []
        
    max_age_val = safe_nonneg_float(max_age_seconds, default=None)
    if max_age_val is None:
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
            raw_sym = trade.get('symbol')
            sym_out = (
                raw_sym.strip().upper()
                if isinstance(raw_sym, str) and raw_sym.strip()
                else symbol
            )
            clean_tid = _valid_fill_id(trade.get('id'))
            clean_oid = _valid_fill_id(trade.get('orderId'))
            if clean_oid in (0, "0", "0.0", "-0", "+0"):
                clean_oid = None
            if clean_oid is None:
                clean_oid = extract_order_id(trade)
            filtered_trades.append({
                'symbol': sym_out,
                'id': clean_tid,
                'orderId': clean_oid,
                'orderListId': clean_oid,
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
