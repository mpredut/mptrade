
import math


####MYLIB
from binance_api.bapi_client import client
from instrument_registry import load_registry, symbols_for


# Historical labels remain available to diagnostics; membership is registry-driven.
_registry = load_registry()
btcsymbol = _registry["BINANCE_BTC"].symbol
taosymbol = _registry["BINANCE_TAO"].symbol
hypesymbol = _registry["HYPERLIQUID_HYPE"].symbol
symbols = symbols_for("binance")
forcesellsymbol = symbols_for("binance", "force_sell")
def validate_ordertype(order_type):
    if order_type not in [None, 'BUY', 'SELL']:
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL' or None.")
   
   
def validate_symbols(symbol):
    if symbol not in symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {symbols}.")


def validate_params(order_type, symbol, price = 1, qty = 1):
    if order_type not in ['BUY', 'SELL']:
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL'.")
    
    if not isinstance(price, (int, float)) or price <= 0:
        raise ValueError(f"Invalid price '{price}'. Price must be a positive number.")
    
    if not isinstance(qty, (int, float)) or qty <= 0:
        raise ValueError(f"Invalid quantity '{qty}'. Quantity must be a positive number.")
    
    if symbol not in symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {symbols}.")
        
      

def get_binance_symbols(keysearch):
    try:
        exchange_info = client.get_exchange_info()
        print(f"Number of symbols on Binance: {len(exchange_info['symbols'])}")

        symbols = [s['symbol'] for s in exchange_info['symbols']]  # Extract symbols only.
        if keysearch:
            matching_symbols = [symbol for symbol in symbols if keysearch.upper() in symbol]
            print(f"Symbols containing '{keysearch}': {matching_symbols}")
        else:
            print(f"All symbols: {symbols}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
   
def get_quantity_precision(symbol):
    try:
        info = client.get_symbol_info(symbol)
        for filter in info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = filter['stepSize']
                precision = -int(round(-math.log10(float(step_size)), 0))
                return precision
    except Exception as e:
        print(f"Error getting quantity precision: {e}")
    return 8  # Default value.


def validate_binance_api_keys():
    try:
        client.get_account()
        print("The API keys are valid!")
        return True
    except Exception as e:
        print(f"Error validating API keys: {e}")
        return False


if __name__ == "__main__":
    if validate_binance_api_keys():
        precision = get_quantity_precision(btcsymbol)
        print(f"Precision is '{precision}'")
