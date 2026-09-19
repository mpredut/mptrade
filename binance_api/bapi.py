
import os
import time
import datetime
import math
import sys
from datetime import datetime, timedelta

import signal

import json
#from twisted.internet import reactor

####Binance
#from binance.streams import BinanceSocketManager
#from binance.streams import BinanceSocketManager
#print(dir(BinanceSocketManager))


####MYLIB
from botcore import load_dotenv as _load_dotenv, required_float_env
import utils as u
import symbols as sym

from .bapi_client import client

import binance
print(binance.__version__)

from . import bapi_ws

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

# Function to handle Ctrl+C and shut down the WebSocket properly
def signal_handler(sig, frame):
    global websocket_thread, stop
    print("Shutting down...")
    bapi_ws.bapi_ws_manager.stop()
    
    # Invoke the default SIGINT handler.
    #signal.default_int_handler(sig, frame)
    sys.exit(0)

    
signal.signal(signal.SIGINT, signal_handler)




def normalize_quantity(symbol, quantity):
    min_qty, max_qty, step_size = get_symbol_limits(symbol)
    if quantity < min_qty:
        print(f"Quantity {quantity} is below the minimum limit. Setting to minimum: {min_qty}")
        quantity = min_qty
    elif quantity > max_qty:
        print(f"Quantity {quantity} is above the maximum limit. Setting to maximum: {max_qty}")
        quantity = max_qty
    
    adjusted_quantity = round(quantity // step_size * step_size, 5)
    
    if adjusted_quantity < min_qty:
        adjusted_quantity = min_qty
    elif adjusted_quantity > max_qty:
        adjusted_quantity = max_qty
    
    return adjusted_quantity


def get_symbol_limits(symbol):
    info = client.get_symbol_info(symbol)
    if info:
        filters = info['filters']
        for f in filters:
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                max_qty = float(f['maxQty'])
                step_size = float(f['stepSize'])
                print(f"Min quantity: {min_qty}, Max quantity: {max_qty}, Step size: {step_size}")
                return min_qty, max_qty, step_size
    return None, None, None

cprice = {}
cprice_time = {}

def update_price(symbol):
    """Refresh one REST quote without making an older value look fresh."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
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
        return float((asset_info or {}).get("free", 0))
    except Exception as e:
        print(f"get_free_balance: Error for {asset}: {e}")
        return None


def get_account_assets_balances():
    try:
        account = client.get_account()
        balances = account.get("balances", [])
        result = []
        for balance in balances:
            free_qty = float(balance.get("free", 0.0))
            locked_qty = float(balance.get("locked", 0.0))
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
        return []


def cancel_orders_old_or_outlier(order_type, symbol, required_quantity, hours=5, price_difference_percentage=0.1):

    order_type = order_type.upper()
    sym.validate_params(order_type, symbol, 1, required_quantity)
    
    open_orders = get_open_orders(order_type, symbol)
    available_qty = 0  # Initially no quantity is available.
    current_price = get_current_price(symbol)
    if open_orders:
        # Sort SELL orders descending and BUY orders ascending.
        sorted_orders = sorted(
            open_orders.items(),
            key=lambda x: (x[1]['price'] if order_type == 'BUY' else -x[1]['price'])
        )

        # Cutoff time for recent orders.
        cutoff_time = datetime.now().timestamp() - timedelta(hours=hours).total_seconds()

        for order_id, order_info in sorted_orders:
            cancel = False
            if order_info['timestamp'] <= cutoff_time:
                cancel = True
            else:
                price_diff_percentage = abs(float(order_info['price']) - current_price) / current_price * 100
                if price_diff_percentage >= price_difference_percentage * 100:  # Convert 0.1 to 10%.
                    cancel = True

            if cancel:
                cancel_order(symbol, order_id)
                available_qty += float(order_info['quantity'])
                print(f"New available quantity: {available_qty:.8f}")

            if available_qty >= required_quantity:
                break

    return available_qty




def get_open_orders(order_type, symbol, *, strict=False):

    order_type = order_type.upper()
    sym.validate_params(order_type, symbol)
        
    try:
        open_orders = client.get_open_orders(symbol=symbol)
        #print(open_orders)
        filtered_orders = {}
        for order in open_orders:
            if order['side'] != order_type.upper():
                continue
            original_qty = float(order['origQty'])
            executed_qty = float(order.get('executedQty') or 0.0)
            filtered_orders[order['orderId']] = {
                'price': float(order['price']),
                # Preserve the legacy field while exposing the financially correct
                # unfilled quantity to cancel-and-replace consumers.
                'quantity': original_qty,
                'executedQty': executed_qty,
                'remainingQty': max(0.0, original_qty - executed_qty),
                'timestamp': order['time'] / 1000
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

def cancel_open_orders(order_type, symbol):
    
    order_type = order_type.upper()
    sym.validate_params(order_type, symbol) 
    
    try:
        open_orders = get_open_orders(order_type, symbol)
        for order_id, order_details in open_orders.items():
            print(f"Cancelling order {order_id} for {symbol}")
            cancel_order(symbol, order_id)
    except Exception as e:
        print(f"Error cancelling orders for {symbol}: {e}")
        

def cancel_expired_orders(order_type, symbol, expire_time):
    
    order_type = order_type.upper()
    sym.validate_params(order_type, symbol)
    
    open_orders = get_open_orders(order_type, symbol)

    #current_time = int(time.time() * 1000)  # Convert current time to milliseconds
    current_time = int(time.time())
  
    if len(open_orders) < 1:
        return
    print(f"Available open orders {len(open_orders)}. Try cancel {order_type} orders type ... ")
      
    count = 0   
    for order_id, order_details in open_orders.items():
        order_time = order_details.get('timestamp')

        if current_time - order_time > expire_time:
            cancel = cancel_order(symbol, order_id)
            if cancel:
                print(f"Cancelled {order_type} order with ID: {order_id} due to expiration.")
            else:
                 print(f"Needs cancel because expiration!")
            count +=1
    print(f"Cancelled {count} orders")
        

def cancel_recent_orders(order_type, symbol, max_age_seconds):

    order_type = order_type.upper()
    sym.validate_params(order_type, symbol)
    
    open_orders = get_open_orders(order_type, symbol)
    current_time = int(time.time())  # Current time in seconds

    if len(open_orders) < 1:
        return
    print(f"Available open orders {len(open_orders)}. Checking for recent {order_type} orders to cancel... ")
   
    count = 0
    for order_id, order_details in open_orders.items():
        order_time = order_details.get('timestamp')  # Assuming timestamp is in seconds
        if current_time - order_time <= max_age_seconds:  # Order is recent
            cancel = cancel_order(symbol, order_id)
            if cancel:
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
        return float(ticker["price"])
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


def get_total_assets_value_usdc(use_cache=True, cache_ttl_seconds=ASSET_VALUE_CACHE_TTL_SECONDS):
    now = time.time()
    if use_cache:
        with _asset_value_cache_lock:
            if (
                _asset_value_cache["value"] is not None
                and (now - _asset_value_cache["timestamp"]) < cache_ttl_seconds
            ):
                return _asset_value_cache["value"]

    total_value = 0.0
    try:
        for balance in get_account_assets_balances():
            total_value += _convert_to_usdc(balance["asset"], balance["total"])
    except Exception as e:
        print(f"Error: get_total_assets_value_usdc: Error calculating portfolio value: {e}")
        return None

    with _asset_value_cache_lock:
        if total_value > 0:
            _asset_value_cache["value"] = total_value
            _asset_value_cache["timestamp"] = now
        else:
            print(f"Error: get_total_assets_value_usdc: Total value can't be calculated")
            return None

    return _asset_value_cache["value"]
