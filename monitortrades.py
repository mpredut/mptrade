import os
import sys
import time
import datetime
import json
import math

import pandas as pd

import threading
from threading import Thread,Timer


#my imports
import symbols as sym
from binance_api import bapi as api
from binance_api import bapi_placeorder as po
from binance_api import bapi_trades as apitrades
from binance_api import bapi_allorders as apiorders
# Read account state through the generic facade instead of bapi/apiorders so this module
# is portable to HYPE. The facade still routes Binance symbols identically to bapi.
# Placement and WebSocket paths remain Binance-specific.
from providers.market_api import api as mkt

import utils as u
import log

# Load versioned, non-secret global tuning parameters before reading the environment.
# ``botcore.load_dotenv`` fills missing values without overwriting the real environment.
# Per-instrument gain/loss/age/hard-TP configuration remains in the instrument configs.
from botcore import load_dotenv as _load_dotenv, required_float_env
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config.env"))

# Trading parameters are mandatory; missing configuration aborts startup.
MT_ARE_CLOSE_TOLERANCE_PCT = required_float_env("MT_ARE_CLOSE_TOLERANCE_PCT")
MT_RECENT_TRADE_BLOCK_SEC = required_float_env("MT_RECENT_TRADE_BLOCK_HOURS") * 3600
MT_ALL_TRADES_BLOCK_SEC = required_float_env("MT_ALL_TRADES_BLOCK_HOURS") * 3600
MT_MAIN_LOOP_SLEEP_SEC = required_float_env("MT_MAIN_LOOP_SLEEP_SEC")
MT_BUY_PRICE_OFFSET = required_float_env("MT_BUY_PRICE_OFFSET")
MT_SELL_SAFEBACK_HOURS = required_float_env("MT_SELL_SAFEBACK_HOURS")
MT_BUY_SAFEBACK_HOURS = required_float_env("MT_BUY_SAFEBACK_HOURS")
MT_GUARD_WINDOW_DAYS = required_float_env("MT_GUARD_WINDOW_DAYS")


# Legacy gradual-sale and monitoring code moved to archive/monitortrades_legacy.py.
# Retain the empty ``trades`` value only for commented main-path calls that may be restored.
trades = []


def print_number_of_trades(maxage_trade_s):
    print(f"TRADE COUNT")
    for symbol in sym.symbols:
        print(f"For {symbol}")
        close_buy_orders = apitrades.get_trade_orders("BUY", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        close_sell_orders = apitrades.get_trade_orders("SELL", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
        print(f"get_trade_orders:           Total found {len(orders)} orders in the last {u.secondsToDays(maxage_trade_s)} days.")


def print_number_of_orders(maxage_trade_s):
    print(f"ORDER COUNT")
    for symbol in sym.symbols:
        print(f"For {symbol}")
        close_buy_orders = apiorders.get_trade_orders("BUY", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        close_sell_orders = apiorders.get_trade_orders("SELL", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        orders = apiorders.get_trade_orders(None, symbol, maxage_trade_s)
        print(f"get_trade_orders:           Total found {len(orders)} orders in the last {u.secondsToDays(maxage_trade_s)} days.")





# Retained for historical reference: StateTracker algorithm and state progression logic.
class StateTracker:
    def __init__(self):
        self.running = True
        self.states = {}  # To hold states for each symbol

    def update_state(self, symbol, slope, tick=0, min_val=0.0, max_val=0.0):
        """Historical state transition tracking based on slope momentum."""
        if symbol not in self.states:
            self.states[symbol] = []

        last_state = self.states[symbol][-1] if self.states[symbol] else None
        self.process_state(symbol, slope, tick, min_val, max_val, last_state)

    def process_state(self, symbol, slope, tick, min_val, max_val, last_state):
        MAX_STATES = 1000
        # If there is no previous state, create a new one
        if last_state is None:
            new_state = {
                'slope': slope,
                'tick': tick,
                'min': min_val,
                'max': max_val
            }
            self.states[symbol].append(new_state)
            if len(self.states[symbol]) > MAX_STATES:
                self.states[symbol].pop(0)
            return

        # If slope is the same as the last state, update the current state's tick and min/max
        if slope * last_state['slope'] > 0 or (abs(slope - last_state['slope']) < 1e-9):  # Same sign.
            last_state['tick'] = tick
            last_state['min'] = min(last_state['min'], min_val)
            last_state['max'] = max(last_state['max'], max_val)
        else:
            # If slope has changed, create a new state
            new_state = {
                'slope': slope,
                'tick': tick,
                'min': min_val,
                'max': max_val
            }
            self.states[symbol].append(new_state)
            if len(self.states[symbol]) > MAX_STATES:
                self.states[symbol].pop(0)

    def display_states(self):
        print("Current states:")
        for symbol, states_list in self.states.items():
            print(f"Symbol: {symbol}")
            for i, state in enumerate(states_list):
                print(f"  State {i + 1}:")
                for key, value in state.items():
                    print(f"    {key}: {value}")
            print()


state_tracker = StateTracker()


# Simplified upward-trend check.
def is_trend_up(symbol):
    """Read instant trend directly from a fresh shared cache snapshot.

    Recent gradient supplies fast momentum. The shared freshness guard prevents stale
    files from influencing the unchanged upward-trend formula.
    """
    try:
        import cacheManager as cm
        snap = cm.get_short_trend_manager().fresh_snapshot(symbol)
        if snap:
            slope = float(snap.get('gradient_recent', snap.get('slope_small', 0.0)) or 0.0)
            gradient = float(snap.get('final_trend', 0.0) or 0.0)
            return slope > 0 or (slope == 0 and gradient > 0)
    except Exception as e:
        print(f"is_trend_up: snapshot direct failed ({e}) — treating it as neutral")
    return False   # Missing or stale data is neutral and does not block a profitable sale.


def get_relevant_trade(trade_orders, trade_type, threshold_s, symbol, now_fn=None):
    """Use injectable ``now_fn`` for replay while preserving wall-clock behavior by default."""
    now_fn = now_fn or time.time
    if not trade_orders:
        print(f"Warning: No {trade_type} transactions for that currency!!!")
        return None, 0, True

    current_time_s = int(now_fn())
     
    trade_orders.sort(key=lambda x: x['timestamp'], reverse=True)
    trade_price = float(trade_orders[0]['price'])
    trade_time = float(trade_orders[0]['timestamp']) / 1000  # Seconds.
    print(f"{trade_type.capitalize()} price for {symbol}: {trade_price} at {u.timeToHMS(trade_time)}")
    
    can_trade = True
    if current_time_s - trade_time < threshold_s:
        print(f"Tranzactii de {trade_type.upper()} prea recente."
            f"Only {u.secondsToHours(current_time_s - trade_time):.2f} h have passed. Waiting for {u.secondsToHours(threshold_s)} h.")
        can_trade = False

    return trade_price, trade_time, can_trade


def get_position_stats(symbol, maxage_trade_s, api=None, buy_orders=None, sell_orders=None):

    api = api or mkt
    if buy_orders is None:
        buy_orders = api.get_orders(symbol, "BUY", maxage_trade_s)
    if sell_orders is None:
        sell_orders = api.get_orders(symbol, "SELL", maxage_trade_s)

    total_buy_qty = sum(float(o['qty']) for o in buy_orders)
    total_sell_qty = sum(float(o['qty']) for o in sell_orders)

    total_buy_value = sum(float(o['price']) * float(o['qty']) for o in buy_orders)
    total_sell_value = sum(float(o['price']) * float(o['qty']) for o in sell_orders)

    average_buy_price = (
        total_buy_value / total_buy_qty
        if total_buy_qty > 0 else 0
    )

    average_sell_price = (
        total_sell_value / total_sell_qty
        if total_sell_qty > 0 else 0
    )

    net_qty = total_buy_qty - total_sell_qty

    return {
        "buy_qty": total_buy_qty,
        "sell_qty": total_sell_qty,
        "net_qty": net_qty,
        "average_buy_price": average_buy_price,
        "average_sell_price": average_sell_price,
        "buy_count": len(buy_orders),
        "sell_count": len(sell_orders),
    }

# Hard TP coexists with trend logic. It sells a position fraction on a large gain
# regardless of trend, catching peaks the trend gate may miss. Cooldown prevents a cascade.
# Defaults below may be overridden by monitortrades.conf.
HARD_TP_ENABLED    = True
HARD_TP_PCT        = 0.17       # Gain fraction that triggers a proportional hard TP.
HARD_TP_FRACTION   = 0.5        # Fraction of free balance to sell.
HARD_TP_COOLDOWN_S = 6 * 3600
TP_REFERENCE       = "last"     # "last" (the last buy) | "average" (the average over maxage days)
_hard_tp_last = {}


def _load_mt_conf(path=None):
    """Override global hard-TP and reference settings from optional configuration.

    Per-symbol gain, loss, and age settings come only from the ``mt`` namespace in
    instruments.conf. Code defaults apply when this optional file is absent or invalid.
    """
    global HARD_TP_ENABLED, HARD_TP_PCT, HARD_TP_FRACTION, HARD_TP_COOLDOWN_S, TP_REFERENCE
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitortrades.conf")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, _, v = line.partition("="); k, v = k.strip(), v.strip()
                if k == "hard_tp_enabled":      HARD_TP_ENABLED = v.lower() in ("yes", "true", "1", "on", "da")
                elif k == "hard_tp_pct":        HARD_TP_PCT = float(v) / 100.0
                elif k == "hard_tp_fraction":   HARD_TP_FRACTION = float(v)
                elif k == "hard_tp_cooldown_h": HARD_TP_COOLDOWN_S = float(v) * 3600
                elif k == "tp_reference":       TP_REFERENCE = v.lower()
    except (OSError, ValueError) as e:
        print(f"monitortrades.conf: {e} — using the values from the code")


_load_mt_conf()


from instrument import Instrument as _Instrument
from instruments_config import load_for


def _as_instrument(x):
    """Accept an Instrument or adapt a symbol string through the provider facade."""
    if isinstance(x, _Instrument):
        return x
    sym = str(x)
    base = u.base_asset(sym)
    return _Instrument(name=sym, symbol=sym, provider=mkt.provider_name_for(sym), base=base)


def get_available_qty(symbol, api=None):
    """Return free base quantity, preserving ``None`` when balance is unavailable."""
    api = api or mkt
    try:
        if api is mkt:
            value = _as_instrument(symbol).free()
            return None if value is None else float(value)
        base = u.base_asset(symbol)
        value = api.free_balance(base)
        return None if value is None else float(value)
    except Exception as e:
        print(f"get_available_qty {symbol}: {e}")
    return None


#//todo: review 0.5
def _place_guarded(inst, side, price, qty, min_qty, **kwargs):
    """Place only positive quantities meeting the venue minimum; return whether submitted."""
    if qty is None or not math.isfinite(float(qty)) or qty <= 0:
        print(f"[{inst.symbol}] {side} skip: qty={qty}")
        return False
    if price is None or not math.isfinite(float(price)) or price <= 0:
        print(f"[{inst.symbol}] {side} skip: price={price}")
        return False
    if min_qty and qty < min_qty:
        print(f"[{inst.symbol}] {side} skip: qty {qty} < volum minim venue {min_qty}")
        return False
    result = inst.place(side, price, qty, **kwargs)
    return result is not None


def monitor_price_and_trade(inst, sbs, maxage_trade_s=None, gain_threshold=None, lost_threshold=None,
                            now_fn=None):
    """Use injectable ``now_fn`` so replay time can follow its market-data provider."""
    now_fn = now_fn or time.time
    inst = _as_instrument(inst)
    symbol = inst.symbol
    # Per-instrument parameters fall back to arguments and then code defaults.
    if gain_threshold is None:
        _g = inst.param("mt", "gain", None, float)
        gain_threshold = _g / 100.0 if _g is not None else 0.07
    if lost_threshold is None:
        _l = inst.param("mt", "lost", None, float)
        lost_threshold = _l / 100.0 if _l is not None else 0.033
    if maxage_trade_s is None:
        _md = inst.param("mt", "maxage_days", None, float)
        maxage_trade_s = int(_md * 24 * 3600) if _md is not None else 4 * 24 * 3600
    hard_tp_pct = inst.param("mt", "hardtp", HARD_TP_PCT * 100, float) / 100.0
    hard_tp_frac = inst.param("mt", "hardtp_fraction", HARD_TP_FRACTION, float)
    hard_tp_cd = inst.param("mt", "hardtp_cooldown_h", HARD_TP_COOLDOWN_S / 3600.0, float) * 3600
    tp_ref = inst.param("mt", "ref", TP_REFERENCE)
    buy_budget = inst.param("mt", "buy_budget", None, float)   # Convert per-buy USD budget to quantity.
    buy_qty_cfg = inst.param("mt", "buy_qty", None, float)     # Alternative fixed quantity.
    max_budget = inst.param("mt", "max_budget", None, float)   # Total USD exposure cap.
    #try:
    
    qty = 1 #qty = calculate_position_size(...)
    threshold_s = MT_RECENT_TRADE_BLOCK_SEC
    current_time_s = int(now_fn())

    # 1. Fetch recent BUY and SELL orders through the facade in normalized form.
    trade_orders_buy = inst.orders("BUY", maxage_trade_s)
    trade_orders_sell = inst.orders("SELL", maxage_trade_s)
    if not (trade_orders_buy or trade_orders_sell):
        print(f"No trade orders found for {symbol} in the last {maxage_trade_s} seconds.")
        return
    buy_price, buy_time, can_buy = get_relevant_trade(trade_orders_buy, "BUY", threshold_s, symbol, now_fn=now_fn)
    sell_price, sell_time, can_sell = get_relevant_trade(trade_orders_sell, "SELL", threshold_s, symbol, now_fn=now_fn)

    position = get_position_stats(
        symbol,
        maxage_trade_s,
        api=inst.provider,
        buy_orders=trade_orders_buy,
        sell_orders=trade_orders_sell,
    )
    # Profit reference is configurable: latest BUY by default or the lookback average.
    if tp_ref == "average" and position["average_buy_price"] > 0:
        buy_price = position["average_buy_price"]
        print(f"POSITION (referinta=AVG {maxage_trade_s/86400:.0f}z) for {symbol} : {position}")
    else:
        print(f"POSITION (reference=the last buy {buy_price}) for {symbol} : {position}")
    if position["average_sell_price"] > 0:
        sell_price = position["average_sell_price"]
        print(f"POSITION for {symbol} : {position}")

    threshold_all_s = MT_ALL_TRADES_BLOCK_SEC
    if current_time_s - max(buy_time, sell_time)  < threshold_all_s:
        print(f"Trades too ... recente."
            f"Pass only {u.secondsToHours(current_time_s -  max(buy_time, sell_time)):.2f} h. Wait to pass {u.secondsToHours(threshold_all_s)} h.")
        can_buy = False
        can_sell = False
        
    
    # 2. Fetch current price through the facade for the instrument's provider.
    current_price = inst.price()
    if current_price is None or not math.isfinite(float(current_price)) or current_price <= 0:
        print(f"No current price for {symbol} (piata inchisa / indisponibil) — skip")
        return
    print(f"Current price for {symbol}: {current_price}")
    avail_qty = inst.free()
    if avail_qty is None:
        print(f"No available balance snapshot for {symbol} — skip")
        return
    avail_qty = float(avail_qty)
    if not math.isfinite(avail_qty) or avail_qty < 0:
        print(f"Invalid available balance for {symbol}: {avail_qty} — skip")
        return
    min_qty = inst.min_qty() or 0.0  # Venue minimum-volume rejection guard.
    isolation = inst.isolation
    owned_qty = max(float(position["net_qty"]), 0.0)
    sellable_qty = min(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty

    # 3. Evaluate BUY history.
    if trade_orders_buy:
        if not buy_price:
            print(f"No buy_price !!!!!")
            return
        price_increase = (current_price - buy_price) / buy_price
        price_decrease = (buy_price - current_price) / buy_price

        print(f"(increase: {price_increase * 100}%, decrease: {price_decrease * 100}%)")
        # 3.0. Hard TP sells a fraction on a large gain regardless of trend. Tolerance
        # prevents permanently missing a peak that fell just short on one tick.
        hard_tp_hit = price_increase >= hard_tp_pct or u.are_close(
            price_increase, hard_tp_pct, target_tolerance_percent=MT_ARE_CLOSE_TOLERANCE_PCT)
        hard_tp_config_valid = 0 < hard_tp_frac <= 1 and hard_tp_pct > 0 and hard_tp_cd >= 0
        if HARD_TP_ENABLED and hard_tp_hit and sellable_qty > 0 and hard_tp_config_valid:
            if current_time_s - _hard_tp_last.get(symbol, 0) >= hard_tp_cd:
                hard_qty = round(sellable_qty * hard_tp_frac, 4)
                print(f"[HARD-TP] {symbol} +{price_increase*100:.1f}% >= {hard_tp_pct*100:.0f}% "
                      f"-> vand {hard_tp_frac*100:.0f}% ({hard_qty}) INDIFERENT de trend")
                if _place_guarded(inst, "SELL", current_price, hard_qty, min_qty,
                                  safeback_seconds=sbs, force=True, cancelorders=True,
                                  hours=MT_SELL_SAFEBACK_HOURS):
                    _hard_tp_last[symbol] = current_time_s
                    return   # Already sold this tick; do not use stale balance below.
            else:
                print(f"[HARD-TP] {symbol} +{price_increase*100:.1f}% but in cooldown (the last one "
                      f"{u.secondsToHours(current_time_s - _hard_tp_last.get(symbol, 0)):.1f}h)")
        # 3.1. Evaluate the existing SELL placement rules.
        if price_increase > gain_threshold or u.are_close(price_increase, gain_threshold, target_tolerance_percent=MT_ARE_CLOSE_TOLERANCE_PCT):
            if not is_trend_up(symbol):
                print(f"Price increased with {price_increase * 100}% by more than {gain_threshold * 100}% versus buy price and not trend up!")
                if can_sell and sellable_qty > 0:
                    _place_guarded(inst, "SELL", current_price, sellable_qty, min_qty,
                        safeback_seconds=sbs, force=False, cancelorders=True,
                        hours=MT_SELL_SAFEBACK_HOURS)
                else:
                    print(f"No can sell (can_sell={can_sell}, sellable_qty={sellable_qty})")
            else :
                print(f"No action taken, because trend is up!")
        elif price_decrease > lost_threshold or u.are_close(price_decrease, lost_threshold, target_tolerance_percent=MT_ARE_CLOSE_TOLERANCE_PCT):
            if not is_trend_up(symbol):
                print(f"Price decreased with {price_decrease * 100}% by more than {lost_threshold * 100}% versus buy price and not trend up!")
                if can_sell and sellable_qty > 0:
                    _place_guarded(inst, "SELL", current_price, sellable_qty, min_qty,
                        safeback_seconds=sbs, force=False, cancelorders=True,
                        hours=MT_SELL_SAFEBACK_HOURS, bypass_profit_guard=True,
                    )
                else:
                    print(f"No can sell (can_sell={can_sell}, sellable_qty={sellable_qty})")
            else:
                print(f"No action taken, because trend is up!")
        else:
            print(f"Nothing interesting")

    # 4. Evaluate SELL history.
    if trade_orders_sell:     
        if not sell_price:
            print(f"No sell_price !!!!!")
            return
        price_decrease_versus_sell = (sell_price - current_price) / sell_price
        print(f"(price_decrease_versus_sell: {price_decrease_versus_sell * 100}%)")
        if price_decrease_versus_sell > gain_threshold or u.are_close(price_decrease_versus_sell, gain_threshold, target_tolerance_percent=MT_ARE_CLOSE_TOLERANCE_PCT):
            if is_trend_up(symbol):
                print(f"Price decreased with {price_decrease_versus_sell * 100}% by more than {gain_threshold * 100}% versus sell price: Placing buy order")
                if can_buy:
                    # Derive quantity from budget, configured fixed quantity, or default.
                    _buy_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
                    # BUY exposure uses the account's actual free base balance.  Own-ledger
                    # attribution restricts SELLs, but must not hide existing holdings from
                    # the post-trade budget cap.
                    _pos_value = avail_qty * current_price
                    _projected_value = _pos_value + (_buy_qty * current_price)
                    if max_budget and _projected_value > max_budget:
                        print(f"[{symbol}] budget cap depasit post-trade "
                              f"({_projected_value:.0f} > {max_budget} USD) — not buying")
                    else:
                        _place_guarded(inst, "BUY", current_price + MT_BUY_PRICE_OFFSET, _buy_qty, min_qty,
                            safeback_seconds=sbs, cancelorders=True,
                            hours=MT_BUY_SAFEBACK_HOURS)
                else:
                   print("No can buy")
            else :
                print(f"No action taken, because trend is down!")

    return

    #except Exception as e:
    #    print(f"An error occurred while monitoring the price: {e}")

def main():
    instruments = load_for("mt")
    if not instruments:
        raise RuntimeError("No enabled monitortrades instruments are configured")

    # Explicit user-data bridge lets each process update Order/Trade memory through
    # its own WebSocket and polling without rereading files.
    import cacheManager as cm
    cm.enable_real_ws_event_sync()

    filename = "trades.json"
    
    maxage_trade_s =  4 * 24 * 3600  # Maximum age for considering filled orders recent.
    interval = 60 * 4 #4 minute


    close_sell_orders = apiorders.get_trade_orders("SELL", sym.taosymbol, maxage_trade_s)
    print(f"get_trade_orders:           Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")
    close_buy_orders = apiorders.get_trade_orders("BUY", sym.taosymbol, maxage_trade_s)
    print(f"get_trade_orders:           Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")
    print(f"close_buy_orders {close_buy_orders}")
    print(f"close_sell_orders {close_sell_orders}")

    d = MT_GUARD_WINDOW_DAYS
    while True:

        print_number_of_orders(maxage_trade_s)
        print_number_of_trades(maxage_trade_s)
        
        # Reload the authoritative registry so deliberate configuration changes take
        # effect without restart. Validation errors propagate and stop the service
        # rather than leaving a trading loop alive with an unknown configuration.
        # Iterate enabled instruments from the ``mt`` namespace and route each explicitly
        # to its provider. Non-Binance orders remain dry until their live gates are enabled.
        instruments = load_for("mt")
        if not instruments:
            raise RuntimeError("No enabled monitortrades instruments are configured")
        for _inst in instruments.values():
            print(f"-----{_inst.name} ({_inst.symbol}@{_inst.provider_label})------")
            try:
                monitor_price_and_trade(_inst, sbs=d*24*3600+60)
            except Exception as _e:
                print(f"[{_inst.name}] monitoring error: {_e}")
            print("--------------")
  
        # Removed a recommendation block consumed only by commented legacy functions.
        # Active trend data now comes directly from cacheManager.
        time.sleep(MT_MAIN_LOOP_SLEEP_SEC)  # Default is 48 seconds.
        
        
if __name__ == "__main__":

     main()


# Reference ideas for unimplemented improvements: is_trend_up may lag and crypto can
# produce false reversals. Possible additions include confidence scoring, multiple
# indicators, and multi-timeframe agreement.
# ──────────────────────────────────────────────────────────────────────────────
