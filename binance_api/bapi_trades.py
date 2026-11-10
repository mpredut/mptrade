import time
import threading

# Repository imports.
import instrument_registry as sym
from botcore import clean_order_id, exact_nonneg_int, safe_nonneg_int
from . import bapi as api
from .bapi import _normalize_side
from . import bapi_allorders as apiorders
from providers.order_ohlc_utils import (
    extract_client_order_id,
    extract_order_id,
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

#######
#######      get_my_trades     #######
#######


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
