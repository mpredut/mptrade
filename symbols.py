
import math


from instrument_registry import load_registry, symbols_for


# Historical labels remain available to diagnostics; membership is registry-driven.
_registry = load_registry()
btcsymbol = _registry["BINANCE_BTC"].symbol if "BINANCE_BTC" in _registry else "BTCUSDC"
taosymbol = _registry["BINANCE_TAO"].symbol if "BINANCE_TAO" in _registry else "TAOUSDC"
arbsymbol = _registry["BINANCE_ARB"].symbol if "BINANCE_ARB" in _registry else "ARBUSDC"
hypesymbol = _registry["HYPERLIQUID_HYPE"].symbol if "HYPERLIQUID_HYPE" in _registry else "HYPEUSDC"
symbols = symbols_for("binance")
forcesellsymbol = symbols_for("binance", "force_sell")


def __getattr__(name):
    if name.endswith("symbol"):
        base = name[:-len("symbol")].upper()
        for spec in _registry.values():
            if spec.base.upper() == base:
                return spec.symbol
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _get_client():
    from binance_api.bapi_client import client
    return client
def _normalize_ordertype(order_type):
    if isinstance(order_type, str):
        s = order_type.strip().upper()
        return {"B": "BUY", "S": "SELL"}.get(s, s)
    return order_type


def _normalize_symbol(symbol):
    if isinstance(symbol, str):
        s = symbol.strip().upper()
        if s in symbols:
            return s
        s_clean = s.replace("/", "")
        if s_clean in symbols:
            return s_clean
    return symbol


def validate_ordertype(order_type):
    norm_type = _normalize_ordertype(order_type)
    if norm_type not in [None, 'BUY', 'SELL']:
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL' or None.")


def validate_symbols(symbol):
    norm_sym = _normalize_symbol(symbol)
    if norm_sym not in symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {symbols}.")


def validate_params(order_type, symbol, price = 1, qty = 1):
    norm_type = _normalize_ordertype(order_type)
    if norm_type not in ['BUY', 'SELL']:
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL'.")

    price_ok = False
    if price is not None and not isinstance(price, bool):
        try:
            pf = float(price)
            price_ok = math.isfinite(pf) and pf > 0
        except (TypeError, ValueError, OverflowError):
            price_ok = False
    if not price_ok:
        raise ValueError(f"Invalid price '{price}'. Price must be a positive number.")

    qty_ok = False
    if qty is not None and not isinstance(qty, bool):
        try:
            qf = float(qty)
            qty_ok = math.isfinite(qf) and qf > 0
        except (TypeError, ValueError, OverflowError):
            qty_ok = False
    if not qty_ok:
        raise ValueError(f"Invalid quantity '{qty}'. Quantity must be a positive number.")

    norm_sym = _normalize_symbol(symbol)
    if norm_sym not in symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {symbols}.")
        
      

def get_binance_symbols(keysearch):
    try:
        exchange_info = _get_client().get_exchange_info()
        if not isinstance(exchange_info, dict) or not isinstance(exchange_info.get("symbols"), list):
            print("Number of symbols on Binance: 0")
            return
        raw_symbols = exchange_info["symbols"]
        print(f"Number of symbols on Binance: {len(raw_symbols)}")

        all_symbols = [
            str(s["symbol"])
            for s in raw_symbols
            if isinstance(s, dict) and s.get("symbol")
        ]  # Extract symbols only.
        if keysearch:
            key_up = str(keysearch).upper()
            matching_symbols = [symbol for symbol in all_symbols if key_up in symbol]
            print(f"Symbols containing '{keysearch}': {matching_symbols}")
        else:
            print(f"All symbols: {all_symbols}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
   
def get_quantity_precision(symbol):
    if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        return 8
    try:
        info = _get_client().get_symbol_info(symbol.strip().upper())
        if not isinstance(info, dict) or not isinstance(info.get("filters"), list):
            return 8
        for f in info["filters"]:
            if isinstance(f, dict) and f.get("filterType") == "LOT_SIZE":
                raw_step = f.get("stepSize")
                if raw_step is None or isinstance(raw_step, bool):
                    continue
                step_size = float(raw_step or 0)
                if step_size <= 0 or not math.isfinite(step_size):
                    break
                precision = max(0, int(round(-math.log10(step_size), 0)))
                return precision
    except Exception as e:
        print(f"Error getting quantity precision: {e}")
    return 8  # Default value.


def validate_binance_api_keys():
    try:
        _get_client().get_account()
        print("The API keys are valid!")
        return True
    except Exception as e:
        print(f"Error validating API keys: {e}")
        return False


if __name__ == "__main__":
    if validate_binance_api_keys():
        precision = get_quantity_precision(btcsymbol)
        print(f"Precision is '{precision}'")
