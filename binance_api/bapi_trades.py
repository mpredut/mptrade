import math
import time

# Repository imports.
from botcore import clean_order_id, clean_upper_token, exact_nonneg_int, safe_nonneg_int
from . import bapi as api, bapi_allorders as apiorders
from providers.base import MarketDataProvider
from providers.order_ohlc_utils import (
    extract_client_order_id,
    extract_order_id,
    extract_order_side,
    extract_timestamp_ms,
)


def init_cache_trade_manager():
    """Return the process-wide BinanceTrade cache reader from CacheFactory."""
    return MarketDataProvider.get_ssot_cache_manager("BinanceTrade")

#######
#######      get_my_trades     #######
#######


def _valid_fill_id(val):
    if isinstance(val, (int, float)):
        return exact_nonneg_int(val, default=None)
    if isinstance(val, str) and val.strip() in {"0", "0.0", "+0", "+0.0"}:
        return "0"
    return clean_order_id(val) if isinstance(val, str) else None


def _resolve_trade_order_id(trade_obj):
    raw_oid = _valid_fill_id(trade_obj.get('orderId'))
    if raw_oid in (None, 0, "0"):
        return extract_order_id(trade_obj)
    return raw_oid


def get_my_trades_24(order_type, symbol, days_ago=0, limit=1000):
    order_type = apiorders.validate_order_query_args(order_type, symbol)
    days_ago = safe_nonneg_int(days_ago, default=0)
    limit = min(1000, safe_nonneg_int(limit, default=1000) or 1000)
    try:
        end_time = math.trunc(time.time() * 1000) - days_ago * 24 * 60 * 60 * 1000
        start_time = end_time - 24 * 60 * 60 * 1000
        latest_trades = {}
        for idx, trade in enumerate(apiorders.paginate_my_trades(api.client, symbol, start_time, limit) or ()):
            if not isinstance(trade, dict):
                continue
            ts_ms = extract_timestamp_ms(trade, default=0) or 0
            if ts_ms > end_time or (order_type in ("BUY", "SELL") and extract_order_side(trade) != order_type):
                continue
            raw_id = _valid_fill_id(trade.get('id'))
            raw_key = raw_id if raw_id is not None else _valid_fill_id(trade.get('orderId'))
            fill_key = raw_key if raw_key is not None else f"_unkeyed_{idx}"
            if (
                fill_key not in latest_trades
                or ts_ms >= (extract_timestamp_ms(latest_trades[fill_key], default=0) or 0)
            ):
                latest_trades[fill_key] = trade
        return list(latest_trades.values())
    except Exception as e:
        print(f"get_my_trades_24: An error occurred: {e}")
        return []


# Return recent BUY or SELL fills from the manager cache.
def get_trade_orders(order_type, symbol, max_age_seconds):
    order_type = apiorders.validate_order_query_args(order_type, symbol)
    filtered_trades = []
    for trade, side, ts_ms, px, qty in apiorders.filter_recent_cached_orders(
        init_cache_trade_manager(), order_type, symbol, max_age_seconds, _validated=True
    ):
        sym_out = clean_upper_token(trade.get('symbol')) or symbol
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
