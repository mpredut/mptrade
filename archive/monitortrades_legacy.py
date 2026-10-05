# Legacy code moved out of monitortrades.py on 16 Jun 2026 and archived here later.
# It is NOT imported or used by the active path. The live bot lives entirely in
# monitortrades.py. Singura referinta externa este tests/testdistributor.py.
# The imports below exist only so the file stays parseable and runnable standalone as a reference.

import sys
import time
import threading
from threading import Thread, Timer

import symbols as sym
from binance_api import bapi as api
from providers.market_api import api as _mkt   # proxy unic guardat (30 iul)

def _legacy_place(side, symbol, price, qty, **kw):
    """Shim compat pt codul LEGACY (nefolosit in flota — vezi monitortrades.py):
    keeps the (side, symbol, ...) order used by the calls below
    (thread targets included), and routes through the single guarded proxy mkt.place()."""
    return _mkt.place(symbol, side, price, qty, smart=False, **kw)
from binance_api import bapi_trades as apitrades
from binance_api import bapi_allorders as apiorders
import utils as u


def adjust_monitor_interval(initial_interval, min_interval, total_duration, elapsed_time):
    if elapsed_time >= total_duration:
        return min_interval
    
    interval_range = initial_interval - min_interval
    time_fraction = elapsed_time / total_duration
    current_interval = initial_interval - (interval_range * time_fraction)
    
    return max(current_interval, min_interval)
    
def calculate_target_price(filled_price, current_price, procent_defined, time_fraction):
    # Calculul procentului ajustat initial
    procent_adjusted = (procent_defined * (1 - time_fraction)) - (1 - current_price / filled_price)

    # Calculul pretului tinta initial
    target_price = filled_price * (1 + procent_adjusted)
    
    # If target_price has fallen below current_price, adjust it.
    if target_price < current_price:
        # Define a dynamic percentage that decreases gradually over time.
        dynamic_percent = 0.01 * (1 - time_fraction) + 1  # It starts at 1.01 and decreases towards 1.
        target_price = current_price * dynamic_percent
    
    return target_price
 
procent_defined = 0.10  # Procentul initial (10%)
def sell_order_gradually(order, start_time, end_time):

    symbol = sym.btcsymbol
    filled_quantity = order['quantity']
    filled_price = order['price']
    close_order_id = order.get('orderId')
    order_id = None

    initial_interval = 20  # Interval initial de monitorizare (in secunde)
    min_interval = 5       # Interval minim de monitorizare (in secunde)
    total_duration = end_time - start_time  # Durata totala a procesului
    current_time = start_time
    
    #while time.time() < end_time:
    while current_time < end_time:
        #elapsed_time = time.time() - start_time
        elapsed_time = current_time - start_time
        monitor_interval = adjust_monitor_interval(initial_interval, min_interval, total_duration, elapsed_time)
         
        print(f"Monitor interval: {monitor_interval:.2f} seconds")

        current_price = get_current_price(symbol)

        if current_price is None:
            print("Failed to obtain the price. Retrying in a few seconds.")
            time.sleep(monitor_interval)
            continue

        
        time_fraction = elapsed_time / total_duration
        target_price = calculate_target_price(filled_price, current_price, procent_defined, time_fraction)

        # Compute the proposed price.
        #if current_price > filled_price:
        #    target_price = max(filled_price * 1.01, current_price * 1.01)  # A price 1% higher.
        #else:
        #    target_price = filled_price * (1 + time_fraction * (current_price / filled_price - 1))

        print(f"Gradual sell: target_price={target_price:.2f}, current_price={current_price:.2f}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds, Target Price: {target_price:.2f} USD")

        # Cancel the previous order before placing a new one.
        if order_id:
            if api.check_order_filled(order_id, symbol) :
                return; #order filled!
            api.cancel_order(symbol, order_id)
            print(f"Cancelled the previous order with ID: {order_id}")

        # Place the sell order.
        new_order = _legacy_place("SELL", symbol, target_price, filled_quantity)
        if new_order:
            order_id = new_order['orderId']
            print(f"Placed a sell order at the price {target_price:.2f}. New Order ID: {order_id}")
        else:
            print("Failed to place the sell order.")
            order_id = None  # Reset the order ID if the placement fails.
        
        # Wait an adjusted interval before the next adjustment.
        time.sleep(monitor_interval)
        current_time += monitor_interval



def monitor_filled_buy_orders_old():
    if threading.active_count() > 1:  # If threads are already active (besides the main one).
        print("Active threads detected, leaving the function so no new threads are started.")
        return
 
    maxage_trade_s =  3 * 24 * 3600  # Cata vechime maxima au ordinele considerate „recente"
    # get_recent_filled_orders expects a DURATION (seconds), not an absolute timestamp.
    filled_buy_orders = apiorders.get_recent_filled_orders("BUY", sym.btcsymbol, maxage_trade_s)

    for order in filled_buy_orders:
        current_time = time.time()
        end_time = current_time + 2 * 3600  # Procesul dureaza doua ore
        print("marius")
        print(order)
        # Start a new thread for each recently filled buy order.
        #thread = threading.Thread(target=sell_order_gradually, args=(order, current_time, end_time))
        #thread = threading.Thread(target=sell_order_gradually, args=(order, current_time, end_time, filled_price, current_price, procent_defined))      
        #thread.start()


def get_close_buy_orders_without_sell(api, maxage_trade_s, profit_percentage):
    symbol = sym.btcsymbol
    #close_buy_orders = apitrades.get_trade_orders("BUY", symbol, maxage_trade_s)
    #close_sell_orders = apitrades.get_trade_orders("SELL", symbol, maxage_trade_s)
    close_buy_orders = apiorders.get_trade_orders("BUY", symbol, maxage_trade_s)
    close_sell_orders = apiorders.get_trade_orders("SELL", symbol, maxage_trade_s)
    
    # The list of "BUY" orders that have no "SELL" associated at the wanted profit.
    buy_orders_without_sell = []

    for buy_order in close_buy_orders:
        filled_price = buy_order['filled_price']
        symbol = buy_order['symbol']
        buy_quantity = buy_order['quantity']  # Cantitatea cumparata
        
        # Filter the "SELL" orders associated with this "BUY" (the same symbol and the wanted price).
        related_sell_orders = [
            order for order in close_sell_orders 
            if order['symbol'] == symbol and order['filled_price'] >= filled_price * (1 + profit_percentage / 100)
        ]
        
        # Sum the quantity sold across the "SELL" orders found.
        total_sell_quantity = sum(order['quantity'] for order in related_sell_orders)
        
        # If the total quantity sold is smaller than the quantity bought.
        if total_sell_quantity < buy_quantity:
            # Add buy_order to the list of orders that are not fully sold yet.
            buy_orders_without_sell.append(buy_order)

    return buy_orders_without_sell
    

def monitor_close_orders_by_age1(maxage_trade_s):
    if threading.active_count() > 2:  # If threads are already active (besides the main one).
        print("Active threads detected, leaving the function so no new threads are started.")
        return
 
    symbol = sym.btcsymbol
    #close_buy_orders = apitrades.get_trade_orders("BUY",  symbol, maxage_trade_s)
    close_buy_orders = apiorders.get_trade_orders("BUY",  symbol, maxage_trade_s)

    print(f"BUY ORDERS, {len(close_buy_orders)}")
    current_price = api.get_current_price(symbol)
    for order in close_buy_orders:
        current_time = time.time()
        end_time = current_time + 2 * 3600  # Procesul dureaza doua ore
        filled_price = order['price']
        quantity = float(order['qty']) #quantity

        if current_price >= filled_price * 1.04 or u.are_close(current_price, filled_price * 1.04):  # If the current price is 7% higher.
            print(f"The current price ({current_price}) is 4% higher than the buy price ({filled_price}). Starting the sell. quantity{quantity}")
            
            # Start a new thread to sell the BTC.
            thread = threading.Thread(target=_legacy_place,
                name="sell_monitor_close_orders_by_age1",
                args=("SELL", symbol, current_price + 200, quantity))
            #sell_order_gradually, args=(order, current_time, end_time))
            thread.start()
            #return
        else:
            print(f"The current price ({current_price}) has not reached the 4% threshold above the buy price ({filled_price}) yet.")
            #return
            
    #close_sell_orders = apitrades.get_trade_orders("SELL",  symbol, maxage_trade_s)
    close_sell_orders = apiorders.get_trade_orders("SELL",  symbol, maxage_trade_s)
    sorted_sell_orders = sorted(close_sell_orders, key=lambda x: x['price'])
    close_sell_orders = sorted_sell_orders
    print(f"SELL ORDERS, {len(close_sell_orders)}")
    for order in close_sell_orders:
        current_time = time.time()
        end_time = current_time + 2 * 3600  # Procesul dureaza doua ore
        filled_price = order['price']
        quantity = float(order['qty']) #quantity

        if current_price <= filled_price * 0.94 or u.are_close(current_price, filled_price * 0.94):  # If the current price is 7% higher.
            print(f"The current price ({current_price}) is 4% lower than the sell price ({filled_price}). Starting the buy. quantity{quantity}.")
            
            # Start a new thread to sell the BTC.
            thread = threading.Thread(target=_legacy_place,
                name="buy_monitor_close_orders_by_age1",
                args=("BUY", symbol, current_price - 200, quantity))
            #sell_order_gradually, args=(order, current_time, end_time))
            thread.start()
            #return
        else:
            print(f"The current price ({current_price}) has not reached the 4% threshold below the sell price ({filled_price}) yet.")
            #return        



# A global variable holding the time at which monitoring started.
start_time_global = None

def monitor_close_orders_by_age2(maxage_trade_s):
    global start_time_global
    
    symbol = sym.btcsymbol
    if threading.active_count() > 2:  # If threads are already active (besides the main one).
        print("Active threads detected, leaving the function so no new threads are started.")
        return
    
    # Initialise the global time on the first execution.
    if start_time_global is None:
        start_time_global = time.time()

    # Compute the total time elapsed since the function first ran.
    current_time = time.time()
    elapsed_time = current_time - start_time_global
    interval_duration = 2 * 3600  # Durata maxima (2 ore)

    # Calculam procentul in functie de timpul scurs (de la 4% pana la 0%)
    drop_percent = max(0, 4 - (4 * (elapsed_time / interval_duration)))
    
    print(f"Current percentage: {drop_percent:.2f}%")

    # Obtain the buy orders.
    #close_buy_orders = apitrades.get_trade_orders("BUY", symbol, maxage_trade_s)
    close_buy_orders = apiorders.get_trade_orders("BUY", symbol, maxage_trade_s)
    print(f"BUY ORDERS, {len(close_buy_orders)}")
    
    current_price = api.get_current_price(symbol)

    for order in close_buy_orders:
        filled_price = order['price']
        quantity = float(order['qty'])  # Cantitatea

        # Check whether the current price has risen by the dynamic percentage.
        if current_price >= filled_price * (1 + drop_percent / 100) or u.are_close(current_price, filled_price * (1 + drop_percent / 100)):
            print(f"The current price ({current_price}) is {drop_percent:.2f}% higher than the buy price ({filled_price}). Starting the sell. Quantity: {quantity}")
            
            # Start a new thread to sell the BTC.
            thread = threading.Thread(target=_legacy_place,
                name="monitor_close_orders_by_age2",
                args=("SELL", symbol, current_price + 200, quantity))
            thread.start()
            
            # Reset the global time so the process restarts.
            start_time_global = time.time()
            return  # Leave the function after the first trade.
        else:
            print(f"The current price ({current_price}) has not reached the {drop_percent:.2f}% threshold above the buy price ({filled_price}).")
    
    # Obtain the sell orders.
    #close_sell_orders = apitrades.get_trade_orders("SELL", symbol, maxage_trade_s)
    close_sell_orders = apiorders.get_trade_orders("SELL", symbol, maxage_trade_s)
    sorted_sell_orders = sorted(close_sell_orders, key=lambda x: x['price'])
    close_sell_orders = sorted_sell_orders
    print(f"SELL ORDERS, {len(close_sell_orders)}")
    
    for order in close_sell_orders:
        filled_price = order['price']
        quantity = float(order['qty'])  # Cantitatea

        # Check whether the current price has fallen by the dynamic percentage.
        if current_price <= filled_price * (1 - drop_percent / 100) or u.are_close(current_price, filled_price * (1 - drop_percent / 100)):
            print(f"The current price ({current_price}) is {drop_percent:.2f}% lower than the sell price ({filled_price}). Starting the buy. Quantity: {quantity}")
            
            # Start a new thread to buy the BTC.
            thread = threading.Thread(target=_legacy_place, 
            name="monitor_close_orders_by_age2",
            args=("BUY", symbol, current_price - 200, quantity))
            thread.start()

            # Reset the global time so the process restarts.
            start_time_global = time.time()
            return  # Leave the function after the first trade.
        else:
            print(f"The current price ({current_price}) has not reached the {drop_percent:.2f}% threshold below the sell price ({filled_price}).")



import time
trades = []

class ProcentDistributor:

    def __init__(self, start_time, expired_duration, max_procent, min_procent=0.008, unitate_timp=60, momentum_weight=0.5):
        if min_procent < 0 or max_procent < min_procent:
            raise ValueError("Invalid procent values")

        self.start_time = start_time
        self.expired_duration = max(1, expired_duration)
        self.unitate_timp = max(1, unitate_timp)

        self.initial_max_procent = max_procent
        self.max_procent = max_procent
        self.min_procent = min_procent
        self.momentum_weight = momentum_weight

        self.total_units = max(1, self.expired_duration / self.unitate_timp)
        self.update_decay()

    def update_decay(self):
        self.procent_per_unit = (self.max_procent - self.min_procent) / self.total_units

    def get_time_based_procent(self, current_time):

        if current_time <= self.start_time:
            return self.max_procent

        elapsed = current_time - self.start_time

        if elapsed >= self.expired_duration:
            return self.min_procent

        units_passed = elapsed / self.unitate_timp
        decayed = self.max_procent - units_passed * self.procent_per_unit

        return max(decayed, self.min_procent)

    def get_market_adjustment(self, current_price, buy_price):

        if buy_price <= 0:
            return 0

        price_change = (current_price - buy_price) / buy_price
        return -price_change * self.momentum_weight

    def get_final_procent(self, current_time, current_price, buy_price):

        base = self.get_time_based_procent(current_time)
        adjustment = self.get_market_adjustment(current_price, buy_price)

        return max(base + adjustment, self.min_procent)

    def update_tick(self, passed=0, half_life_duration=24*60*60):

        if passed <= 0:
            return

        decay_factor = 0.5 ** (passed * self.expired_duration / half_life_duration)

        self.max_procent = max(
            self.initial_max_procent * decay_factor,
            self.min_procent
        )

        self.update_decay()


class BuyTransaction:

    def __init__(self, trade_id, qty, buy_price, procent_desired_profit, min_procent, expired_duration, time_trade):

        self.trade_id = trade_id
        self.qty = qty
        self.buy_price = buy_price
        self.time_trade = time_trade
        self.expired_duration = expired_duration
        self.sell_order_id = None

        self.distributor = ProcentDistributor(
            start_time=time_trade,
            expired_duration=expired_duration,
            max_procent=procent_desired_profit,
            min_procent=min_procent,
        )

    def get_passed_cycles(self, current_time):
        return int((current_time - self.time_trade) // self.expired_duration)

    def get_reference_price(self, current_price, current_time, days=7):

        elapsed = current_time - self.time_trade
        passed_cycles = self.get_passed_cycles(current_time)

        if passed_cycles == 0:
            return max(self.buy_price, current_price)

        if elapsed < days * 24 * 60 * 60:
            return self.buy_price

        return current_price

    def get_proposed_sell_price(self, current_price, current_time, days=7):

        passed_cycles = self.get_passed_cycles(current_time)

        self.distributor.update_tick(
            passed=passed_cycles,
            half_life_duration=24*60*60
        )

        reference_price = self.get_reference_price(current_price, current_time, days)

        procent = self.distributor.get_final_procent(
            current_time,
            current_price,
            self.buy_price
        )

        return max(
            reference_price * (1 + procent),
            current_price * 1.001
        )


def update_trades(trades, symbol, maxage_trade_s, procent_desired_profit, expired_duration, min_procent):
    #new_trades = apitrades.get_trade_orders("BUY", symbol, maxage_trade_s)
    new_trades = apiorders.get_trade_orders("BUY", symbol, maxage_trade_s)
    #TODO filter out trades that are too recent, under 2 hours
    for trade in new_trades:
        if not any(t.trade_id == trade['id'] for t in trades):
            trades.append(BuyTransaction(
                trade_id=trade['id'],
                qty=trade['qty'],
                buy_price=trade['price'],
                procent_desired_profit=procent_desired_profit,  # Procentul initial
                min_procent=min_procent,
                expired_duration=expired_duration,  # Durata de 2.7 ore * (3600 secunde)
                time_trade=trade['time'] / 1000  # Convert the time from milliseconds to seconds.
            ))
    new_trade_ids = {trade['id'] for trade in new_trades}
    trades[:] = [t for t in trades if t.trade_id in new_trade_ids]
    #trades.sort(key=lambda t: t.buy_price)
    trades.sort(key=lambda t: t.buy_price, reverse=True)


def apply_sell_orders(trades, days, force_sell):
    symbol = sym.btcsymbol

    placed_order_count = 0
    total_weighted_price = 0
    total_quantity = 0

      
    current_time = time.time()    
    current_price = api.get_current_price(symbol)

    count = 0
    for trade in trades:
        
        print(f"\nTrade {count} ({trade.trade_id})") 
        count+=1
        if trade.sell_order_id and api.check_order_filled(trade.sell_order_id['orderId'], symbol):
            print(f"check_order_filled {trade.sell_order_id}")
            trade.sell_order_id = 0  # Mark it as executed.
        if trade.sell_order_id == 0:
            continue  # Skip the trades marked as executed.

        sell_price = trade.get_proposed_sell_price(current_price, current_time, days=days)
        if force_sell: #disperare!!!
            print("\nDESPERATION\n Selling at the current price!")
            sell_price = min(sell_price, current_price * 1.001)

        if trade.sell_order_id:
            #print(f"cancel {trade.sell_order_id}")
            api.cancel_order(symbol, trade.sell_order_id['orderId'])
            trade.sell_order_id = None

        # Check whether the number of orders has exceeded 8.
        if placed_order_count < 6:
            new_sell_order_id = _legacy_place("SELL", symbol, sell_price, trade.qty)
            trade.sell_order_id = new_sell_order_id
            placed_order_count += 1
        else:
            #print(f"Placing a single sell order: Quantity {trade.qty}, Price {sell_price}")
            # Adaugam tranzactia in calculul mediei ponderate
            total_weighted_price += sell_price * trade.qty
            total_quantity += trade.qty
            trade.sell_order_id = None  # We do not place the order immediately, but mark it as in progress.


    print("\n")
    # If there were extra orders, compute the weighted average and place a single order.
    if total_quantity > 0:
        average_sell_price = total_weighted_price / total_quantity
        print(f"Total: Quantity {total_quantity}, Price {average_sell_price}")
        #quantity = min(api.get_asset_info("SELL", symbol), total_quantity)
        new_sell_order_id = _legacy_place("SELL", symbol, average_sell_price, total_quantity)
        #trade.sell_order_id = new_sell_order_id
        



# The main function that periodically runs the updates and the cache.
def monitor_trades(filename, interval=3600, limit=1000, years_to_keep=2):
    order_type = None
    while True:
        for symbol in sym.symbols:
            apitrades.save_trades_to_file(order_type, symbol, filename, limit=limit, years_to_keep=years_to_keep)
        
        # Reincarcam tranzactiile in cache
        apitrades.load_trades_from_file(filename)   
        time.sleep(interval)

# The function that starts periodic monitoring in a separate thread.
def start_monitoring(filename, interval=3600, limit=1000, years_to_keep=2):
    monitoring_thread = Thread(
        target=monitor_trades,
        name="monitor_trades",
        args=(filename, interval, limit, years_to_keep),
        daemon=True
    )
    monitoring_thread.start()


def test() :
    filename="trades.json"
    limit=1000
    years_to_keep=0.09
    order_type=None
    #for symbol in sym.symbols:
    #    apitrades.save_trades_to_file(order_type, symbol, filename, limit=limit, years_to_keep=years_to_keep)
    
    apitrades.save_trades_to_file(order_type, sym.taosymbol, filename, limit=limit, years_to_keep=years_to_keep)
    apitrades.load_trades_from_file(filename)
    #trade_orders_buy = apitrades.get_trade_orders(None, sym.taosymbol, 24 * 60 * 60 * 11)
    #trade_orders_buy = apiorders.get_trade_orders(None, sym.taosymbol, 24 * 60 * 60 * 11)
    trade_orders_buy = apiorders.get_trade_orders(None, sym.taosymbol, 24 * 60 * 60 * 11)
    trade_orders_buy = apiorders.get_trade_orders(None, sym.taosymbol, 24 * 60 * 60 * 11)
    
    print(f"{len(trade_orders_buy)}, {trade_orders_buy}")
    sys.exit(1)


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

