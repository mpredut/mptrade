# order_guard.py
"""Platform-agnostic profit guard.

Rule: do not BUY above the last SELL or SELL below the last BUY, subject to a minimum
margin. Decoupled from Binance, it accepts any `provider` implementing
`last_opposite_fill(symbol, order_type)` and an optional caller-computed `window_ref`
(for example, min(sell)/max(buy) from the Binance order cache). The SAME profit logic
therefore runs on every venue instead of being embedded only in bapi_placeorder.

Imports only utils, with no providers/cacheManager, avoiding circular imports. Reference
read failures from provider.last_opposite_fill propagate so the caller can fail closed.
Returns True when placement is allowed and False when blocked.

The profit threshold is configured per venue in versioned, non-sensitive
`order_guard.conf`. A missing file or invalid value falls back to fail-safe 1.15%."""
import os
import time
import math
import utils as u

_MARGINS = None   # cache: {provider_lower: percentage, "default": 1.15}


def _load_margins():
    """Read and cache `key = value` lines from order_guard.conf once.
    The configuration file is the SINGLE source of truth. The dictionary below is only
    a safety net for missing entries, such as a truncated file, and centralizes fallbacks
    so hard-coded shadow defaults are not scattered across functions. Change operational
    values in order_guard.conf, not here."""
    global _MARGINS
    if _MARGINS is not None:
        return _MARGINS
    m = {
        "default": 1.15,                       # fallback profit threshold (%)
        "default_window_h": 0.0,               # profit-guard window (hours); 0 uses only last_opposite_fill
        "default_max_daily_trades": 25,        # daily trade cap
        "default_safeback_sec": 14 * 24 * 3600 + 60,  # own-trade search window (seconds): 14 days
        "default_recent_transaction_sec": 180,   # anti-spam window (seconds)
        "default_buy_reference": 1.0,          # 1 = BUY must beat the historical sell reference
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_guard.conf")
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip().lower(); v = v.strip()
                try:                       # thresholds/hours/weights are floats; proxies are strings
                    m[k] = float(v)
                except ValueError:
                    m[k] = v
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[order_guard] conf invalid ({e}) — folosesc default 1.15")
    _MARGINS = m
    return m


def _provider_name(provider):
    if isinstance(provider, str):
        return provider
    return getattr(provider, "name", "") or ""


def buy_reference_enabled(provider_name):
    """Return whether a BUY must sit below the venue's historical sell reference.

    Configuration key `<venue>_buy_reference` (1 on, 0 off), falling back to
    `default_buy_reference`. Off removes the static re-entry anchor ("never buy above the
    lowest recent sell"); the SELL reference and the quantity/weight guards stay active."""
    m = _load_margins()
    name = _provider_name(provider_name)
    key = name.lower() + "_buy_reference"
    try:
        val = m.get(key, m.get("default_buy_reference", 1.0))
        return float(val) != 0.0
    except (TypeError, ValueError, KeyError):
        return True     # an unreadable value keeps the conservative guard


def margin_for(provider_name):
    """Return the venue's configured minimum profit percentage, defaulting to 1.15."""
    m = _load_margins()
    return m.get((provider_name or "").lower(), m["default"])


def window_for(provider_name):
    """Return the per-venue min/max reference window in seconds.
    Configuration key `<venue>_window_h`; 0 disables the windowed tier."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_window_h"
    hours = m.get(key, m["default_window_h"])
    return float(hours) * 3600.0


def weight_proxy_for(provider_name):
    """Return the trend/Gaussian-weight proxy symbol used when the current symbol lacks
    its own long trend (e.g. HYPE -> BTC until data exists). Uses `<venue>_weight_proxy`
    or `default_weight_proxy`; None falls back to weight 0.03."""
    m = _load_margins()
    return m.get((provider_name or "").lower() + "_weight_proxy", m.get("default_weight_proxy"))


def window_reference(provider, symbol, order_type, window_s):
    """Return a time-windowed reference from opposite-side orders/fills: minimum SELL
    for a BUY or maximum BUY for a SELL. Return None for an empty or disabled window.
    Read errors propagate so the caller fails closed. Non-positive prices are ignored.
    This is the platform-agnostic equivalent of Binance tier 1."""
    if not window_s or window_s <= 0:
        return None
    opp = "SELL" if order_type.upper() == "BUY" else "BUY"
    recent = provider.get_orders(symbol, opp, window_s) or []
    prices = [float(o.get("price") or 0) for o in recent if float(o.get("price") or 0) > 0]
    if not prices:
        return None
    return min(prices) if order_type.upper() == "BUY" else max(prices)


def weight_limit(provider, symbol, order_type, price, required_qty, *, available_qty):
    """Cap per-order quantity using the Gaussian curve, the platform-agnostic equivalent
    of bapi.apply_weight_limit. Allocate tradable value by trend position so the whole
    amount is not bought or sold at once. priceAnalysis supplies the Gaussian weight;
    provider.get_orders supplies 24-hour traded value. The shared quantity decision must
    supply the side-aware balance, which this guard does not re-read. Return the smaller
    of requested and permitted quantity. Errors propagate so the caller fails closed."""
    import math
    def _ok(w):
        return w is not None and not (isinstance(w, float) and math.isnan(w)) and w > 0

    def _gauss(sym):
        try:
            import priceAnalysis as pa
            return pa.get_weight_for_cash_permission_at_quant_time(sym, order_type)
        except Exception as e:
            print(f"[WEIGHT] {sym}: cannot compute the gauss value ({e})")
            return None

    weight = _gauss(symbol)                                 # own Gaussian weight when a long trend exists
    if not _ok(weight):                                     # no own trend: try a proxy such as BTC for HYPE
        proxy = weight_proxy_for(getattr(provider, "name", ""))
        if proxy and proxy != symbol:
            pw = _gauss(proxy)
            if _ok(pw):
                weight = pw
                print(f"[WEIGHT] {symbol}: no trend of its own -> proxy {proxy} (weight={weight})")
    if not _ok(weight):                                     # no valid proxy: use conservative default
        weight = 0.03
    recent = provider.get_orders(symbol, order_type, 86400) or []      # same side over the last 24 hours
    traded_value = sum(float(o.get("price", 0)) * float(o.get("qty", o.get("quantity", 0))) for o in recent)
    # available is in BASE and already side-aware from providers.quantity:
    #   SELL: BASE balance available to sell;
    #   BUY: BASE purchasable with QUOTE balance (free_balance maps USD to ZUSD on Kraken).
    available = float(available_qty)
    total_ref = traded_value + available * price                       # total potentially tradable quote value
    max_trade_value = total_ref * weight                               # Gaussian-weighted cap
    remaining_value = max(0.0, max_trade_value - traded_value)         # remaining value allowed today
    remaining_qty = remaining_value / price if price else 0.0
    adjusted = min(required_qty, remaining_qty)
    print(f"[WEIGHT] {order_type} {symbol}: weight={weight} traded24h={traded_value:.2f} "
          f"avail={available:.6f} max={max_trade_value:.2f} remaining={remaining_value:.2f} "
          f"cerut={required_qty:.6f} -> {adjusted:.6f}")
    return adjusted


def max_daily_trades_for(provider_name):
    """Return the venue's daily trade cap from order_guard.conf or the seed fallback."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_max_daily_trades"
    return int(m.get(key, m["default_max_daily_trades"]))


def safeback_sec_for(provider_name):
    """Return the per-venue trade-search window in seconds for the daily cap."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_safeback_sec"
    return float(m.get(key, m["default_safeback_sec"]))


def recent_transaction_sec_for(provider_name):
    """Return the anti-spam window that rejects a recent same-symbol, same-side order."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_recent_transaction_sec"
    return float(m.get(key, m["default_recent_transaction_sec"]))


def daily_limit_guard(provider, symbol, order_type, max_daily_trades=None,
                      safeback_sec=None, recent_transaction_sec=None):
    """Enforce the daily cap and anti-spam policy as the SINGLE implementation.
    Since July 30, Binance bapi_placeorder.if_place_safe_order delegates here instead of
    maintaining a second inline version; Kraken and Hyperliquid use it through
    Instrument.place(). provider.get_orders supplies same-side orders from safeback_sec.
    Exceeding max_daily_trades per day returns "daily_limit"; a record inside
    recent_transaction_sec returns "recent_transaction".

    backdays = math.ceil(safeback_sec/86400), rounded UP to preserve the historical
    Binance formula and its effective threshold rather than using simple division.

    Return (True, None) or (False, reason). Read errors propagate so the caller can fail
    closed, consistently with profit_guard and weight_limit."""
    name = _provider_name(provider)
    max_daily_trades = max_daily_trades if max_daily_trades is not None else max_daily_trades_for(name)
    safeback_sec = float(safeback_sec if safeback_sec is not None else safeback_sec_for(name))
    recent_transaction_sec = float(recent_transaction_sec if recent_transaction_sec is not None
                                   else recent_transaction_sec_for(name))
    order_type = order_type.upper()
    trades = provider.get_orders(symbol, order_type, safeback_sec) or []
    backdays = max(math.ceil(safeback_sec / 86400.0), 1)
    if len(trades) / backdays > max_daily_trades:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: {len(trades)} trades in "
              f"{safeback_sec/3600:.1f}h, a cap of {max_daily_trades}/day -> BLOCKED")
        return False, "daily_limit"
    cutoff_ms = time.time() * 1000 - recent_transaction_sec * 1000
    for t in trades:
        ts = t.get("timestamp")
        if ts is not None and float(ts) >= cutoff_ms:
            print(f"[DAILY-LIMIT] {order_type} {symbol}: recent trade "
                  f"(<{recent_transaction_sec:.0f}s) -> BLOCKED")
            return False, "recent_transaction"
    return True, None


def profit_guard(provider, symbol, order_type, price, profit_percentage, window_ref=None):
    """Return whether the order is profitable relative to its reference.
    Reference cascade: caller-provided window_ref first, otherwise
    provider.last_opposite_fill(symbol, order_type). A missing or non-positive reference
    allows placement because there is no prior transaction to compare."""
    order_type = order_type.upper()
    provider_name = _provider_name(provider)
    if order_type == "BUY" and not buy_reference_enabled(provider_name):
        print(f"[GUARD] BUY {symbol}: the historical sell reference is off for this venue "
              f"(order_guard.conf); price {price} is not compared with past sells")
        return True
    ref = window_ref if window_ref is not None else (
        provider.last_opposite_fill(symbol, order_type) if hasattr(provider, "last_opposite_fill") else None
    )
    if ref is None or ref <= 0:
        return True
    if order_type == "BUY":
        diff = u.value_diff_to_percent(ref, price)   # (ref_SELL - BUY_price) / ref_SELL
    else:
        diff = u.value_diff_to_percent(price, ref)   # (SELL_price - ref_BUY) / SELL_price
    src = "the window" if window_ref is not None else "the provider"
    print(f"[GUARD] {order_type} {symbol}: ref {ref} ({src}), price {price}, "
          f"diff {diff:.2f}%, prag {profit_percentage}%")
    if diff < profit_percentage:
        print(f"Diferenta procentuala ({diff:.2f}%) sub prag {profit_percentage}%. "
              f"The {order_type} order is BLOCKED.")
        return False
    return True
