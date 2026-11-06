
import os
import time
import threading
from collections import defaultdict
from state_io import atomic_write_json, load_json_state

# Repository imports.
import utils as u
import instrument_registry as sym
from botcore import exact_nonneg_int, safe_nonneg_float, safe_pos_float, safe_nonneg_int
from . import bapi as api
from . import bapi_allorders as apiorders
from providers.base import (
    clean_order_id,
    extract_client_order_id,
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
    if isinstance(val, (int, float)):
        return exact_nonneg_int(val, default=None)
    if isinstance(val, str) and val.strip() in {"0", "0.0", "+0", "+0.0"}:
        return "0"
    return clean_order_id(val) if isinstance(val, str) else None


def _trade_fill_key(trade_obj):
    if not isinstance(trade_obj, dict):
        return None
    raw_key = _valid_fill_id(trade_obj.get('id'))
    return raw_key if raw_key is not None else _valid_fill_id(trade_obj.get('orderId'))


def _resolve_trade_order_id(trade_obj):
    raw_oid = _valid_fill_id(trade_obj.get('orderId'))
    if raw_oid in (None, 0, "0"):
        return extract_order_id(trade_obj)
    return raw_oid


def _dedup_trades_by_fill_id(all_trades):
    """Deduplicate trade fills by individual fill ID (falling back to orderId when absent)."""
    latest_trades = {}
    for idx, trade in enumerate(all_trades or ()):
        if not isinstance(trade, dict):
            continue
        raw_key = _trade_fill_key(trade)
        fill_key = raw_key if raw_key is not None else f"_unkeyed_{idx}"
        ts_ms = extract_timestamp_ms(trade, default=0) or 0
        if (
            fill_key not in latest_trades
            or ts_ms >= (extract_timestamp_ms(latest_trades[fill_key], default=0) or 0)
        ):
            latest_trades[fill_key] = trade
    return list(latest_trades.values())


def _fmt_trade_num(val: float) -> str:
    return (f"{val:.8f}" if val >= 1e-8 else f"{val:.12f}").rstrip("0").rstrip(".") or str(val)


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
        orderId = _resolve_trade_order_id(trade)
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
            else (_fmt_trade_num(px) if px > 0 else (raw_px if raw_px is not None else px))
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
            price_val = _fmt_trade_num(vwap) if vwap > 0 else "0"
        else:
            price_val = aggregated['price']
        qty_str = (
            f"{total_qty:.8f}"
            if total_qty <= 0 or total_qty >= 1e-8
            else _fmt_trade_num(total_qty)
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


_clean_non_neg_int = safe_nonneg_int


def _validate_trade_query_args(order_type, symbol, days, *, default_days: int, limit=1000):
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    days_val = safe_nonneg_int(days, default=default_days)
    limit_val = min(1000, safe_nonneg_int(limit, default=1000) or 1000)
    return order_type, days_val, limit_val


def _filter_trades_by_side(trades, order_type):
    if order_type in ("BUY", "SELL"):
        return [trade for trade in (trades or ()) if extract_order_side(trade) == order_type]
    return list(trades or ())


def get_my_trades_24(order_type, symbol, days_ago=0, limit=1000):
    order_type, days_ago, limit = _validate_trade_query_args(
        order_type, symbol, days_ago, default_days=0, limit=limit
    )
    try:
        end_time = int(time.time() * 1000) - days_ago * 24 * 60 * 60 * 1000
        start_time = end_time - 24 * 60 * 60 * 1000
        trades = [
            t for t in apiorders.paginate_my_trades(api.client, symbol, start_time, limit)
            if (extract_timestamp_ms(t, default=0) or 0) <= end_time
        ]
        return _dedup_trades_by_fill_id(_filter_trades_by_side(trades, order_type))
    except Exception as e:
        print(f"get_my_trades_24: An error occurred: {e}")
        return []


def get_my_trades_simple(order_type, symbol, backdays=3, limit=1000):
    order_type, backdays, limit = _validate_trade_query_args(
        order_type, symbol, backdays, default_days=3, limit=limit
    )

    try:
        # Paginate the entire window with ``fromId`` so days containing more than 1,000
        # fills are not truncated.
        start_time = int(time.time() * 1000) - (backdays + 1) * 24 * 60 * 60 * 1000
        all_trades = _filter_trades_by_side(
            apiorders.paginate_my_trades(api.client, symbol, start_time, limit),
            order_type,
        )

        # Deduplicate by individual fill ID (falling back to orderId only when id is absent)
        # so multi-fill orders do not lose earlier partial fills.
        return _dedup_trades_by_fill_id(all_trades)

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def load_trades_from_file(filename):
    global trade_cache
    try:
        trade_cache = load_json_state(
            filename, default_factory=list, fail_closed=False, label="bapi_trades", root_type=list
        )
        if trade_cache:
            print(f"Cache loaded with {len(trade_cache)} trades.")
        elif not os.path.exists(filename):
            print(f"File {filename} not found.")
    except Exception:
        print("Error reading file.")
        trade_cache = []
    print(set(trade.get('symbol') for trade in trade_cache if isinstance(trade, dict) and trade.get('symbol')))


# Return recent BUY or SELL fills from the manager cache.
def get_trade_orders(order_type, symbol, max_age_seconds):
    order_type = _normalize_side(order_type)
    sym.validate_ordertype(order_type)
    sym.validate_symbols(symbol)
    filtered_trades = []
    for trade, side, ts_ms, px, qty in apiorders.filter_recent_cached_orders(
        init_cache_trade_manager(), order_type, symbol, max_age_seconds
    ):
        raw_sym = trade.get('symbol')
        sym_out = (
            raw_sym.strip().upper()
            if isinstance(raw_sym, str) and raw_sym.strip()
            else symbol
        )
        clean_oid = _resolve_trade_order_id(trade)
        row = {
            'symbol': sym_out,
            'id': _valid_fill_id(trade.get('id')),
            'orderId': clean_oid,
            'orderListId': clean_oid,
            'price': px,
            'qty': qty,
            'quantity': qty,
            'time': ts_ms,
            'timestamp': ts_ms,
            'isBuyer': side == "BUY",
            'side': side,
        }
        cid = extract_client_order_id(trade, default="")
        if cid:
            row['clientOrderId'] = cid
        filtered_trades.append(row)
    return filtered_trades
