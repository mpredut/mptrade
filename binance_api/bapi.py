
import time
import sys
import threading
import signal

#### MYLIB
from botcore import (
    _parse_bool_flag,
    clean_upper_token,
    load_dotenv,
    normalize_scalar_order_id,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
import instrument_registry as sym

from .bapi_client import client
from . import bapi_ws
from providers import order_id_context as rc
from providers.base import MarketDataProvider
from providers.order_ohlc_utils import (
    filter_and_normalize_open_orders,
    iter_expired_open_order_ids,
)

load_dotenv()
BINANCE_REST_PRICE_CACHE_TTL_SEC = required_float_env(
    "CM_BINANCE_REST_PRICE_CACHE_TTL_SEC"
)
if BINANCE_REST_PRICE_CACHE_TTL_SEC <= 0:
    raise ValueError(
        "CM_BINANCE_REST_PRICE_CACHE_TTL_SEC must be finite and positive")

stop = False

# Function to handle Ctrl+C and shut down the WebSocket properly
def signal_handler(sig, frame):
    global stop
    stop = True
    print("Shutting down...")
    bapi_ws.bapi_ws_manager.stop()
    sys.exit(0)


try:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)
except (ValueError, AttributeError):
    pass

cprice = {}
cprice_time = {}

_normalize_side = sym._normalize_ordertype


def update_price(symbol):
    """Refresh one REST quote without making an older value look fresh."""
    if (symbol := clean_upper_token(symbol)) is None:
        return None
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        raw_px = ticker["price"] if isinstance(ticker, dict) else None
        price = safe_pos_float(raw_px, default=None)
        if price is None:
            raise ValueError(f"invalid ticker price: {raw_px!r}")
    except Exception as e:
        print("update_price: An unexpected error occurred:", e)
        return None

    # Publish value and success time together only after complete validation.
    cprice[symbol] = price
    cprice_time[symbol] = time.time()
    return price

def get_current_price(symbol):
    """Return a bounded-age REST quote, failing closed after refresh failure."""
    if (symbol := clean_upper_token(symbol)) is None:
        return None
    now = time.time()
    cached_at_f = safe_float(cprice_time.get(symbol), default=None)
    cached_f = safe_pos_float(cprice.get(symbol), default=None)
    if cached_at_f is not None and cached_f is not None:
        age = safe_nonneg_float(now - cached_at_f, default=None)
        if age is not None and age < BINANCE_REST_PRICE_CACHE_TTL_SEC:
            return cached_f

    # A failed refresh returns None. It deliberately does not extend the timestamp
    # of an older quote, so repeated outages cannot keep stale data apparently fresh.
    return update_price(symbol)


def get_free_balance(asset: str):
    try:
        asset_u = clean_upper_token(asset)
        if asset_u is None:
            return None
        # Return the free Binance balance for an asset.
        asset_info = client.get_asset_balance(asset=asset_u)
        raw_free = (asset_info or {}).get("free", 0)
        if isinstance(raw_free, bool):
            raise TypeError(f"invalid boolean free balance: {raw_free!r}")
        free_val = safe_nonneg_float(raw_free, default=None)
        if free_val is None:
            raise ValueError(f"invalid free balance: {raw_free!r}")
        return free_val
    except Exception as e:
        print(f"get_free_balance: Error for {asset}: {e}")
        return None


def get_account_assets_balances(raise_errors=False):
    raise_errors = _parse_bool_flag(raise_errors, False)
    try:
        account = client.get_account()
        balances = (account.get("balances") if isinstance(account, dict) else None) or []
        result = []
        for balance in balances:
            if not isinstance(balance, dict):
                raise TypeError(f"invalid balance entry: {balance!r}")
            raw_free = balance.get("free", 0.0)
            raw_locked = balance.get("locked", 0.0)
            if isinstance(raw_free, bool) or isinstance(raw_locked, bool):
                raise TypeError(f"invalid boolean balance fields: {balance!r}")
            free_qty = safe_nonneg_float(raw_free, default=None)
            locked_qty = safe_nonneg_float(raw_locked, default=None)
            if free_qty is None or locked_qty is None:
                raise ValueError(f"invalid non-finite or negative balance: {balance!r}")
            total_qty = free_qty + locked_qty
            raw_asset = balance.get("asset")
            asset_name = raw_asset.strip() if isinstance(raw_asset, str) else ""
            if not asset_name:
                continue
            if total_qty <= 0:
                if asset_name in sym.symbols:
                    print(f"get_account_assets_balances: Skip {asset_name} because total_qty is 0")
                    continue
            result.append(
                {
                    "asset": asset_name,
                    "free": free_qty,
                    "locked": locked_qty,
                    "total": total_qty,
                }
            )
        return result
    except Exception as e:
        print(f"get_account_assets_balances: Error reading balances: {e}")
        if raise_errors:
            raise
        return []




def _validate_side(order_type, symbol) -> str:
    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol)
    return order_type


def get_open_orders(order_type, symbol, *, strict=False, allowed_owners=None):
    strict = _parse_bool_flag(strict, False)
    order_type = _validate_side(order_type, symbol)
    symbol = symbol.strip().upper()
        
    try:
        open_orders = client.get_open_orders(symbol=symbol)
        return filter_and_normalize_open_orders(
            open_orders,
            order_type.upper(),
            allowed_owners=allowed_owners,
            rc_mod=rc,
            owner_map_fn=MarketDataProvider.get_accepted_retry_order_owner_map,
            fallback_extract_oid=True,
        )
    except Exception as e:
        print(f"Error getting open {order_type} orders: {e}")
        if strict:
            raise
        return {}


def _cancel_and_resolve(symbol, order_id) -> bool:
    if cancel_order(symbol, order_id):
        try:
            MarketDataProvider.resolve_accepted_retry_order(symbol, order_id)
        except Exception:
            pass
        return True
    return False


def cancel_order(symbol, order_id):
    try:
        if (symbol := clean_upper_token(symbol)) is None:
            return False
        oid_arg = normalize_scalar_order_id(order_id)
        if oid_arg is None:
            return False
        client.cancel_order(symbol=symbol, orderId=oid_arg)
        print(f"The order with ID {order_id} was cancelled.")
        return True
    except Exception as e:
        print(f"Error canceling order {order_id}: {e}")
        return False

def cancel_open_orders(order_type, symbol, allowed_owners=None):
    order_type = _validate_side(order_type, symbol)
    try:
        open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)
        for order_id in open_orders:
            print(f"Cancelling order {order_id} for {symbol}")
            _cancel_and_resolve(symbol, order_id)
    except Exception as e:
        print(f"Error cancelling orders for {symbol}: {e}")


def cancel_expired_orders(order_type, symbol, expire_time, allowed_owners=None):
    order_type = _validate_side(order_type, symbol)
    if (expire_val := safe_pos_float(expire_time, default=None)) is None:
        return
    if not (open_orders := get_open_orders(order_type, symbol, allowed_owners=allowed_owners)):
        return
    current_time = time.time()
    print(f"Available open orders {len(open_orders)}. Try cancel {order_type} orders type ... ")
    count = 0
    for order_id in iter_expired_open_order_ids(open_orders, expire_val, current_time_sec=current_time):
        if _cancel_and_resolve(symbol, order_id):
            print(f"Cancelled {order_type} order with ID: {order_id} due to expiration.")
            count += 1
        else:
            print("Needs cancel because expiration!")
    print(f"Cancelled {count} orders")


# ---------------- Portfolio value query API ----------------

ASSET_VALUE_CACHE_TTL_SECONDS = 120

_asset_value_cache = {"value": None, "timestamp": 0.0}
_asset_value_cache_lock = threading.Lock()


def _get_symbol_price_safe(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        raw_px = (ticker or {}).get("price") if isinstance(ticker, dict) else None
        return safe_pos_float(raw_px, default=None)
    except Exception:
        return None


def _convert_to_usdc(asset, amount):
    if (asset_clean := clean_upper_token(asset)) is None:
        return 0.0
    amt_f = safe_pos_float(amount, default=0.0)
    if amt_f <= 0:
        return 0.0
    if asset_clean in {"USDC", "USD"}:
        return amt_f

    price = _get_symbol_price_safe(f"{asset_clean}USDC")
    if price:
        return amt_f * price

    inv_price = _get_symbol_price_safe(f"USDC{asset_clean}")
    if inv_price and inv_price > 0:
        return amt_f / inv_price

    if asset_clean in {"USDT", "FDUSD"}:
        return amt_f

    return 0.0


def get_total_assets_value_usdc(use_cache=True, cache_ttl_seconds=ASSET_VALUE_CACHE_TTL_SECONDS, raise_errors=False):
    use_cache = _parse_bool_flag(use_cache, True)
    raise_errors = _parse_bool_flag(raise_errors, False)
    now = time.time()
    ttl_f = safe_nonneg_float(
        cache_ttl_seconds,
        default=safe_nonneg_float(ASSET_VALUE_CACHE_TTL_SECONDS, default=0.0),
    )
    if use_cache:
        with _asset_value_cache_lock:
            if (
                _asset_value_cache["value"] is not None
                and 0 <= (now - _asset_value_cache["timestamp"]) < ttl_f
            ):
                return _asset_value_cache["value"]

    total_value = 0.0
    try:
        for balance in get_account_assets_balances(raise_errors=True):
            total_value += _convert_to_usdc(balance["asset"], balance["total"])
    except Exception as e:
        print(f"Error: get_total_assets_value_usdc: Error calculating portfolio value: {e}")
        if raise_errors:
            raise
        return None

    with _asset_value_cache_lock:
        if total_value > 0:
            _asset_value_cache["value"] = total_value
            _asset_value_cache["timestamp"] = now
        else:
            print("Error: get_total_assets_value_usdc: Total value can't be calculated")
            return None

    return _asset_value_cache["value"]
