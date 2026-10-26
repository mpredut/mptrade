
import os
import time
import math
import sys
import threading
import signal

#### MYLIB
from botcore import load_dotenv as _load_dotenv, required_float_env
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
if (
    not math.isfinite(BINANCE_REST_PRICE_CACHE_TTL_SEC)
    or BINANCE_REST_PRICE_CACHE_TTL_SEC <= 0
):
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

def update_price(symbol):
    """Refresh one REST quote without making an older value look fresh."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        raw_px = ticker["price"]
        if isinstance(raw_px, bool):
            raise TypeError(f"invalid boolean ticker price: {raw_px!r}")
        price = float(raw_px)
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"invalid ticker price: {price!r}")
    except Exception as e:
        print(f"update_price: An unexpected error occurred: {e}")
        return None

    # Publish value and success time together only after complete validation.
    cprice[symbol] = price
    cprice_time[symbol] = time.time()
    return price

def get_current_price(symbol):
    """Return a bounded-age REST quote, failing closed after refresh failure."""
    now = time.time()
    cached_at = cprice_time.get(symbol)
    cached = cprice.get(symbol)
    if cached_at is not None and cached is not None:
        age = now - cached_at
        if 0 <= age < BINANCE_REST_PRICE_CACHE_TTL_SEC:
            return cached

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
        # Return the free Binance balance for an asset.
        asset_info = client.get_asset_balance(asset=asset)
        raw_free = (asset_info or {}).get("free", 0)
        if isinstance(raw_free, bool):
            raise TypeError(f"invalid boolean free balance: {raw_free!r}")
        free_val = float(raw_free)
        if not math.isfinite(free_val) or free_val < 0:
            raise ValueError(f"invalid free balance: {free_val!r}")
        return free_val
    except Exception as e:
        print(f"get_free_balance: Error for {asset}: {e}")
        return None


def get_account_assets_balances(raise_errors=False):
    try:
        account = client.get_account()
        balances = account.get("balances", [])
        result = []
        for balance in balances:
            if not isinstance(balance, dict):
                raise TypeError(f"invalid balance entry: {balance!r}")
            raw_free = balance.get("free", 0.0)
            raw_locked = balance.get("locked", 0.0)
            if isinstance(raw_free, bool) or isinstance(raw_locked, bool):
                raise TypeError(f"invalid boolean balance fields: {balance!r}")
            free_qty = float(raw_free)
            locked_qty = float(raw_locked)
            if (
                not math.isfinite(free_qty)
                or free_qty < 0
                or not math.isfinite(locked_qty)
                or locked_qty < 0
            ):
                raise ValueError(f"invalid non-finite or negative balance: {balance!r}")
            total_qty = free_qty + locked_qty
            if total_qty <= 0:
                if balance.get('asset') in sym.symbols:
                    print(f"get_account_assets_balances: Skip {balance.get('asset')} because total_qty is 0")
                    continue
            result.append(
                {
                    "asset": balance.get("asset"),
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




def get_open_orders(order_type, symbol, *, strict=False, allowed_owners=None):

    order_type = order_type.upper()
    sym.validate_params(order_type, symbol)
        
    try:
        open_orders = client.get_open_orders(symbol=symbol)
        #print(open_orders)
        filtered_orders = {}
        accepted_owner_map = None
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            if extract_order_side(order) != order_type.upper():
                continue
            cid = str(order.get('clientOrderId') or '')
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
                    allowed_set = (
                        {allowed_owners.lower()}
                        if isinstance(allowed_owners, str)
                        else {str(o).lower() for o in allowed_owners}
                    )
                    if not inferred or inferred.lower() not in allowed_set:
                        continue
                else:
                    continue
            original_qty = float(extract_orig_qty(order, default=0.0) or 0.0)
            raw_exec_qty = order.get('executedQty', order.get('z', order.get('vol_exec', 0.0)))
            try:
                exec_f = float(raw_exec_qty) if (raw_exec_qty is not None and not isinstance(raw_exec_qty, bool)) else 0.0
                executed_qty = exec_f if (math.isfinite(exec_f) and exec_f >= 0.0) else 0.0
            except (TypeError, ValueError, OverflowError):
                executed_qty = 0.0
            remaining_qty = extract_remaining_qty(order, default=max(0.0, original_qty - executed_qty))
            ts_ms = extract_timestamp_ms(order, default=0) or 0
            if 'time' in order and order.get('time') is not None and not isinstance(order.get('time'), bool):
                try:
                    raw_time = float(order['time'])
                    ts_sec = raw_time / 1000.0 if (raw_time < 1_000_000_000.0 or raw_time >= 100_000_000_000.0) else raw_time
                except (TypeError, ValueError):
                    ts_sec = ts_ms / 1000.0 if ts_ms >= 100_000_000_000 else float(ts_ms)
            else:
                ts_sec = ts_ms / 1000.0 if ts_ms >= 100_000_000_000 else float(ts_ms)
            raw_oid = order['orderId'] if order.get('orderId') is not None else extract_order_id(order)
            if raw_oid is None:
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
 
          
def cancel_order(symbol, order_id):
    try:
        if not order_id:
            return False
        client.cancel_order(symbol=symbol, orderId=order_id)
        print(f"The order with ID {order_id} was cancelled.")
        return True
    except Exception as e:
        print(f"Error canceling order {order_id}: {e}")
        return False

def cancel_open_orders(order_type, symbol, allowed_owners=None):
    
    order_type = order_type.upper()
    sym.validate_params(order_type, symbol) 
    
    try:
        open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)
        for order_id, order_details in open_orders.items():
            print(f"Cancelling order {order_id} for {symbol}")
            if cancel_order(symbol, order_id):
                try:
                    import order_retry
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass
    except Exception as e:
        print(f"Error cancelling orders for {symbol}: {e}")
        

def cancel_expired_orders(order_type, symbol, expire_time, allowed_owners=None):
    
    order_type = order_type.upper()
    sym.validate_params(order_type, symbol)
    if expire_time is None or isinstance(expire_time, bool):
        return
    try:
        expire_val = float(expire_time)
    except (TypeError, ValueError, OverflowError):
        return
    if not math.isfinite(expire_val) or expire_val <= 0:
        return
    
    open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)

    #current_time = int(time.time() * 1000)  # Convert current time to milliseconds
    current_time = int(time.time())
  
    if len(open_orders) < 1:
        return
    print(f"Available open orders {len(open_orders)}. Try cancel {order_type} orders type ... ")
      
    count = 0   
    for order_id, order_details in open_orders.items():
        order_time = (extract_timestamp_ms(order_details, default=0) or 0) / 1000.0

        if order_time > 0 and current_time - order_time > expire_val:
            cancel = cancel_order(symbol, order_id)
            if cancel:
                try:
                    import order_retry
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass
                print(f"Cancelled {order_type} order with ID: {order_id} due to expiration.")
            else:
                 print(f"Needs cancel because expiration!")
            count +=1
    print(f"Cancelled {count} orders")
        

def cancel_recent_orders(order_type, symbol, max_age_seconds, allowed_owners=None):

    order_type = order_type.upper()
    sym.validate_params(order_type, symbol)
    if max_age_seconds is None or isinstance(max_age_seconds, bool):
        return
    try:
        max_age_val = float(max_age_seconds)
    except (TypeError, ValueError, OverflowError):
        return
    if not math.isfinite(max_age_val) or max_age_val < 0:
        return
    
    open_orders = get_open_orders(order_type, symbol, allowed_owners=allowed_owners)
    current_time = int(time.time())  # Current time in seconds

    if len(open_orders) < 1:
        return
    print(f"Available open orders {len(open_orders)}. Checking for recent {order_type} orders to cancel... ")
   
    count = 0
    for order_id, order_details in open_orders.items():
        order_time = (extract_timestamp_ms(order_details, default=0) or 0) / 1000.0
        age_sec = current_time - order_time
        if order_time > 0 and -5.0 <= age_sec <= max_age_val:  # Order is recent
            cancel = cancel_order(symbol, order_id)
            if cancel:
                try:
                    import order_retry
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass
                print(f"Cancelled {order_type} order with ID: {order_id} (recent order).")
                count += 1
            else:
                print(f"Failed to cancel {order_type} order with ID: {order_id}. Needs cancel because recent order.")
    
    print(f"Cancelled {count} recent orders.")


def check_order_filled(order_id, symbol):
    """Legacy adapter; new code uses market_api.order_status()."""
    try:
        if not order_id:
            return False
        from providers.market_api import api as market_api
        return market_api.order_status(symbol, str(order_id)).fully_filled
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
        if raw_px is None or isinstance(raw_px, bool):
            return None
        price = float(raw_px)
        return price if math.isfinite(price) and price > 0 else None
    except Exception:
        return None


def _convert_to_usdc(asset, amount):
    if amount <= 0:
        return 0.0
    if asset == "USDC":
        return amount

    price = _get_symbol_price_safe(f"{asset}USDC")
    if price:
        return amount * price

    return 0.0


def get_total_assets_value_usdc(use_cache=True, cache_ttl_seconds=ASSET_VALUE_CACHE_TTL_SECONDS, raise_errors=False):
    now = time.time()
    if use_cache:
        with _asset_value_cache_lock:
            if (
                _asset_value_cache["value"] is not None
                and 0 <= (now - _asset_value_cache["timestamp"]) < cache_ttl_seconds
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
