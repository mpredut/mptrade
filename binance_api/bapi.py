
import os
import time
import sys
import threading
import signal

#### MYLIB
from botcore import (
    _parse_bool_flag,
    load_dotenv as _load_dotenv,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
import symbols as sym

from .bapi_client import client
from . import bapi_ws
from . import order_id_context as rc
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
    extract_remaining_qty,
    extract_timestamp_ms,
    normalize_side,
)

resolve_order_owner = rc.resolve_order_owner
is_order_repriceable_by_monitororder = rc.is_order_repriceable_by_monitororder
order_matches_owner = rc.order_matches_owner
create_client_order_id = rc.create_client_order_id

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_load_dotenv(os.path.join(_ROOT, "config.env"))
BINANCE_REST_PRICE_CACHE_TTL_SEC = required_float_env(
    "CM_BINANCE_REST_PRICE_CACHE_TTL_SEC"
)
if BINANCE_REST_PRICE_CACHE_TTL_SEC <= 0:
    raise ValueError(
        "CM_BINANCE_REST_PRICE_CACHE_TTL_SEC must be finite and positive")

stop = False

# Function to handle Ctrl+C and shut down the WebSocket properly
def signal_handler(sig, frame):
    global websocket_thread, stop
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

_INVALID_BAPI_ID_STRINGS = {"", "0", "0.0", "-0", "+0", "nan", "inf", "-inf", "+inf", "none", "null"}


def _normalize_side(order_type):
    if isinstance(order_type, str):
        return normalize_side(order_type) or order_type.strip().upper()
    return order_type


def update_price(symbol):
    """Refresh one REST quote without making an older value look fresh."""
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        return None
    symbol = symbol.strip().upper()
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        raw_px = ticker["price"] if isinstance(ticker, dict) else None
        if isinstance(raw_px, bool):
            raise TypeError(f"invalid boolean ticker price: {raw_px!r}")
        price = safe_pos_float(raw_px, default=None)
        if price is None:
            raise ValueError(f"invalid ticker price: {raw_px!r}")
    except Exception as e:
        print(f"update_price: An unexpected error occurred: {e}")
        return None

    # Publish value and success time together only after complete validation.
    cprice[symbol] = price
    cprice_time[symbol] = time.time()
    return price

def get_current_price(symbol):
    """Return a bounded-age REST quote, failing closed after refresh failure."""
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        return None
    symbol = symbol.strip().upper()
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

currenttime = time.time()       
def get_current_time():
        global currenttime
        currenttime = time.time()
        return currenttime


def split_symbol(symbol: str):
    # Split the symbol into base and quote; operational Binance pairs use USDC.
   from providers.quantity import resolve_assets
   base, quote = resolve_assets(symbol)
   if not quote:
       raise ValueError(f"Unknown symbol: {symbol}")
   return base, quote


def get_free_balance(asset: str):
    try:
        if isinstance(asset, bool) or not isinstance(asset, str) or not asset.strip():
            return None
        # Return the free Binance balance for an asset.
        asset_info = client.get_asset_balance(asset=asset.strip().upper())
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




def _clean_bapi_oid_str(val):
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or s.lower() in _INVALID_BAPI_ID_STRINGS or s.startswith("-"):
        return None
    if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
        s = s[:-2]
    if s.startswith("+") and s[1:].isdigit():
        s = s[1:]
    if not s or s.lower() in _INVALID_BAPI_ID_STRINGS or s.startswith("-"):
        return None
    return s


def get_open_orders(order_type, symbol, *, strict=False, allowed_owners=None):
    strict = _parse_bool_flag(strict, False)
    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol)
    symbol = symbol.strip().upper()
        
    try:
        open_orders = client.get_open_orders(symbol=symbol)
        #print(open_orders)
        filtered_orders = {}
        accepted_owner_map = None
        for order in (open_orders or ()):
            if not isinstance(order, dict):
                continue
            if extract_order_side(order) != order_type.upper():
                continue
            raw_cid = order.get('clientOrderId')
            cid = extract_order_id(raw_cid, default='') if raw_cid is not None and not isinstance(raw_cid, (bool, dict, list, tuple, set, frozenset)) else ''
            if cid and cid.strip().lower() in _INVALID_BAPI_ID_STRINGS:
                cid = ''
            if allowed_owners is not None and not rc.order_matches_owner(cid, allowed_owners):
                if cid.startswith("OR_"):
                    if accepted_owner_map is None:
                        try:
                            import order_retry
                            accepted_owner_map = order_retry.get_accepted_order_owner_map()
                        except Exception:
                            accepted_owner_map = {}
                    inferred = (
                        accepted_owner_map.get(str(extract_order_id(order) or ''))
                        or accepted_owner_map.get(cid)
                    )
                    allowed_set = rc._normalize_allowed_owners(allowed_owners)
                    if not inferred or str(inferred).strip().lower() not in allowed_set:
                        continue
                else:
                    continue
            original_qty = float(extract_orig_qty(order, default=0.0) or 0.0)
            raw_exec_qty = order.get('executedQty', order.get('z', order.get('vol_exec', 0.0)))
            executed_qty = safe_nonneg_float(raw_exec_qty, default=0.0)
            remaining_qty = extract_remaining_qty(order, default=max(0.0, original_qty - executed_qty))
            ts_ms = extract_timestamp_ms(order, default=0) or 0
            ts_sec = ts_ms / 1000.0 if ts_ms >= 100_000_000_000 else float(ts_ms)
            raw_time = safe_nonneg_float(order.get('time'), default=None)
            if raw_time is not None:
                ts_sec = raw_time / 1000.0 if (raw_time < 1_000_000_000.0 or raw_time >= 100_000_000_000.0) else raw_time
            raw_oid = order.get('orderId')
            if isinstance(raw_oid, float):
                oid_f = safe_pos_float(raw_oid, default=None)
                raw_oid = int(oid_f) if oid_f is not None and oid_f.is_integer() else None
            elif isinstance(raw_oid, str):
                raw_oid = _clean_bapi_oid_str(raw_oid)
            elif not isinstance(raw_oid, int) or isinstance(raw_oid, bool) or raw_oid <= 0:
                raw_oid = None
            if raw_oid is None:
                raw_oid = extract_order_id(order)
                if isinstance(raw_oid, str):
                    raw_oid = _clean_bapi_oid_str(raw_oid)
            if raw_oid is None or isinstance(raw_oid, bool):
                continue
            filtered_orders[raw_oid] = {
                'price': extract_order_price(order),
                # Preserve the legacy field while exposing the financially correct
                # unfilled quantity to cancel-and-replace consumers.
                'quantity': original_qty,
                'origQty': original_qty,
                'qty': remaining_qty,
                'executedQty': executed_qty,
                'remainingQty': remaining_qty,
                'timestamp': ts_sec,
                'time': ts_ms,
                'clientOrderId': cid,
            }
        
        return filtered_orders
    except Exception as e:
        print(f"Error getting open {order_type} orders: {e}")
        if strict:
            raise
        return {}
 
          
def _resolve_valid_bapi_oid(order_id):
    cleaned_oid = (
        extract_order_id(order_id, default=None)
        if not isinstance(order_id, (dict, list, tuple, set, frozenset))
        else None
    )
    if isinstance(cleaned_oid, str):
        cleaned_oid = _clean_bapi_oid_str(cleaned_oid)
    if cleaned_oid is None or str(cleaned_oid).strip().lower() in _INVALID_BAPI_ID_STRINGS:
        return None
    return cleaned_oid


def _cancel_and_resolve(symbol, order_id) -> bool:
    if cancel_order(symbol, order_id):
        try:
            import order_retry
            order_retry.resolve_accepted_order(symbol, order_id)
        except Exception:
            pass
        return True
    return False


def cancel_order(symbol, order_id):
    try:
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return False
        symbol = symbol.strip().upper()
        cleaned_oid = _resolve_valid_bapi_oid(order_id)
        if cleaned_oid is None:
            return False
        if isinstance(order_id, int) and not isinstance(order_id, bool):
            if order_id <= 0:
                return False
            oid_arg = order_id
        elif isinstance(order_id, float):
            oid_f = safe_pos_float(order_id, default=None)
            if oid_f is None or not oid_f.is_integer():
                return False
            oid_arg = int(oid_f)
        else:
            oid_arg = cleaned_oid
        client.cancel_order(symbol=symbol, orderId=oid_arg)
        print(f"The order with ID {order_id} was cancelled.")
        return True
    except Exception as e:
        print(f"Error canceling order {order_id}: {e}")
        return False

def cancel_open_orders(order_type, symbol, allowed_owners=None):
    
    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol) 
    
    try:
        open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)
        for order_id, order_details in open_orders.items():
            print(f"Cancelling order {order_id} for {symbol}")
            _cancel_and_resolve(symbol, order_id)
    except Exception as e:
        print(f"Error cancelling orders for {symbol}: {e}")
        

def cancel_expired_orders(order_type, symbol, expire_time, allowed_owners=None):
    
    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol)
    expire_val = safe_pos_float(expire_time, default=None)
    if expire_val is None:
        return
    
    open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)

    #current_time = int(time.time() * 1000)  # Convert current time to milliseconds
    current_time = float(time.time())
  
    if len(open_orders) < 1:
        return
    print(f"Available open orders {len(open_orders)}. Try cancel {order_type} orders type ... ")
      
    count = 0   
    for order_id, order_details in open_orders.items():
        order_time = (extract_timestamp_ms(order_details, default=0) or 0) / 1000.0

        if order_time > 0 and current_time - order_time > expire_val:
            if _cancel_and_resolve(symbol, order_id):
                print(f"Cancelled {order_type} order with ID: {order_id} due to expiration.")
            else:
                 print(f"Needs cancel because expiration!")
            count +=1
    print(f"Cancelled {count} orders")
        

def cancel_recent_orders(order_type, symbol, max_age_seconds, allowed_owners=None):

    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol)
    max_age_val = safe_nonneg_float(max_age_seconds, default=None)
    if max_age_val is None:
        return
    
    open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)
    current_time = float(time.time())  # Current time in seconds

    if len(open_orders) < 1:
        return
    print(f"Available open orders {len(open_orders)}. Checking for recent {order_type} orders to cancel... ")
   
    count = 0
    for order_id, order_details in open_orders.items():
        order_time = (extract_timestamp_ms(order_details, default=0) or 0) / 1000.0
        age_sec = current_time - order_time
        if order_time > 0 and -5.0 <= age_sec <= max_age_val:  # Order is recent
            if _cancel_and_resolve(symbol, order_id):
                print(f"Cancelled {order_type} order with ID: {order_id} (recent order).")
                count += 1
            else:
                print(f"Failed to cancel {order_type} order with ID: {order_id}. Needs cancel because recent order.")
    
    print(f"Cancelled {count} recent orders.")


def check_order_filled(order_id, symbol):
    """Legacy adapter; new code uses market_api.order_status()."""
    try:
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return False
        symbol = symbol.strip().upper()
        cleaned_oid = _resolve_valid_bapi_oid(order_id)
        if cleaned_oid is None:
            return False
        from providers.market_api import api as market_api
        return market_api.order_status(symbol, cleaned_oid).fully_filled
    except Exception as e:
        print(f"Error checking order status: {e}")
        return False



def check_order_filled_by_time(order_type, symbol, time_back_in_seconds, pret_min=None, pret_max=None):
    """Legacy adapter that delegates fill discovery to the shared facade."""
    from providers.market_api import api as market_api
    return market_api.latest_fill_price(
        symbol, order_type, time_back_in_seconds,
        min_notional=pret_min, max_notional=pret_max)


# ---------------- Portfolio value query API ----------------
import threading

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
    if isinstance(asset, bool) or not isinstance(asset, str) or not asset.strip():
        return 0.0
    amt_f = safe_pos_float(amount, default=0.0)
    if amt_f <= 0:
        return 0.0
    asset_clean = asset.strip().upper()
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
    ttl_f = safe_nonneg_float(cache_ttl_seconds, default=float(ASSET_VALUE_CACHE_TTL_SECONDS))
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
            print(f"Error: get_total_assets_value_usdc: Total value can't be calculated")
            return None

    return _asset_value_cache["value"]
