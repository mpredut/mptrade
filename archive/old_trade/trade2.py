import time
import datetime
import sys
import math
from binance.client import Client
from binance.exceptions import BinanceAPIException
from collections import deque

from keys.apikeys import api_key, api_secret

# my imports

#my imports
import log
import alertnotifiers as alert
import utils as u
import symbols as sym
import bapi as api
import bapi_placeorder as po
import bapi_trades as apitrades
import bapi_allorders as apiorders


class PriceWindow:
    def __init__(self, window_size, max_index=10000, epsilon=1e-2):
        self.window_size = window_size
        self.prices = deque()  # Store all prices in the window
        self.min_deque = deque()  # Manage the minimums
        self.max_deque = deque()  # Manage the maximums
        self.current_index = 0  # Internal counter to keep track of index
        self.max_index = max_index  # Threshold for normalization
        self.epsilon = 10  # Tolerance for approximately equal minimums

    def process_price(self, price):
        self.prices.append(price)

        if len(self.prices) > self.window_size:
            removed_price = self.prices.popleft()
            #print(f"Price removed from window: {removed_price}")

        self._manage_minimum(price)
        self._manage_maximum(price)

        self.current_index += 1
        print(f"BTC: {round(price, 0):.0f} Current index in window: {self.current_index}")
       
        #if self.current_index > self.max_index:
        #    print(f"Start normalize indexes")
        #    self._normalize_indices()

    def _normalize_indices(self):
        old_index = self.current_index
        self.current_index = 0  # Reset current index

        self.min_deque = deque([(i - old_index, price) for price, i in self.min_deque if i >= old_index])
        self.max_deque = deque([(i - old_index, price) for price, i in self.max_deque if i >= old_index])

        print(f"Indices normalized. Old index: {old_index}, New index: {self.current_index}")

    def _manage_minimum(self, price):
        if self.min_deque and self.min_deque[0][1] <= self.current_index - self.window_size:
            removed_min = self.min_deque.popleft()
            print(f"Minimum removed as it is outside the window: {removed_min}")

        #do this also for max
        #for existing_price, index in self.min_deque:
        #    if abs(existing_price - price) <= self.epsilon: 
        #        return  # Don't add the current price if an equivalent exists

        while self.min_deque and self.min_deque[-1][0] > price:
            removed_min = self.min_deque.pop()
            #print(f"Minimums removed from back: {removed_min}")

        self.min_deque.append((price, self.current_index))

    def _manage_maximum(self, price):
        if self.max_deque and self.max_deque[0][1] <= self.current_index - self.window_size:
            removed_max = self.max_deque.popleft()
            print(f"Maximum removed as it is outside the window: {removed_max}")

        while self.max_deque and self.max_deque[-1][0] <= price:
            removed_max = self.max_deque.pop()
            #print(f"Maximums removed from back: {removed_max}")

        self.max_deque.append((price, self.current_index))

    def get_min(self):
        if not self.min_deque:
            return None
        return self.min_deque[0][0]  # Return the minimum price

    def get_max(self):
        if not self.max_deque:
            return None
        return self.max_deque[0][0]  # Return the maximum price

    def get_min_and_index(self):
        if not self.min_deque:
            return None
        return self.min_deque[0][0], self.min_deque[0][1]  # Return min price and its index

    def get_max_and_index(self):
        if not self.max_deque:
            return None
        return self.max_deque[0][0], self.max_deque[0][1]  # Return max price and its index

    def calculate_slope(self):
        min_price = self.get_min()
        max_price = self.get_max()

        if min_price is None or max_price is None:
            return 0

        min_index = self.min_deque[0][1]
        max_index = self.max_deque[0][1]

        if max_index == min_index:
            return 0

        slope = (max_price - min_price) / (max_index - min_index)
        return slope


    def calculate_proximities(self, current_price):
        min_price, _ = self.get_min_and_index()
        max_price, _ = self.get_max_and_index()

        if max_price != min_price:
            min_proximity = (current_price - min_price) / (max_price - min_price)
            max_proximity = (max_price - current_price) / (max_price - min_price)

            # Check for negative values and raise an exception.
            if min_proximity < 0 or max_proximity < 0:
                print(f"Negative proximity detected! min_proximity: {min_proximity}, max_proximity: {max_proximity}")
                min_proximity = max_proximity = 0
                sys.exit(1)
        else:
            min_proximity = max_proximity = 0 # Almost exactly at the min and the max.
        #min_proximity + max_proximity = 1 
        #If min_proximity -> 0 it means the price is closer to the min.
        #If max_proximity -> 0 it means the price is closer to the max.

        return min_proximity, max_proximity


    def calculate_positions(self):
        min_index = self.min_deque[0][1] if self.min_deque else 0
        max_index = self.max_deque[0][1] if self.max_deque else 0
        min_position = (min_index % self.window_size) / self.window_size
        max_position = (max_index % self.window_size) / self.window_size
        return min_position, max_position

        
    def evaluate_buy_sell_opportunity(self, current_price, threshold_percent=1, decrease_percent=3.7):
        slope = self.calculate_slope()
     
        min_price, min_index = self.get_min_and_index()
        max_price, max_index = self.get_max_and_index()
        
        print(
            f"Min price: {min_price} at index: {min_index} "
            f"Max price: {max_price} at index: {max_index}"
        )

        min_position, max_position = self.calculate_positions()
        min_proximity, max_proximity = self.calculate_proximities(current_price)
 
        print(
            f"Min position: {min_position:.2f}, Max position: {max_position:.2f} "
            f"Min proximity: {min_proximity:.2f}, Max proximity: {max_proximity:.2f}"
        )
        
        price_change_percent = (max_price - min_price) / min_price * 100 if min_price and max_price else 0
        print(
            f"Price change percent: {price_change_percent:.2f} "
            f"slope: {slope:.2f} "
            f"Market trending: {'upwards' if slope > 0 else 'downwards'}"
        )

        if price_change_percent < threshold_percent and not u.are_close(price_change_percent, threshold_percent):
            action = 'HOLD'
            print(f"Action: {action}")
            return action, current_price, price_change_percent, slope
            
        alert.check_alert(True, f"Price changed {price_change_percent:.2f}%. Current price {current_price}")
        action = 'BUY'
        remaining_decrease_percent = max(0, decrease_percent - price_change_percent)
        print(f"Remaining decrease percent: {remaining_decrease_percent:.2f}")
        proposed_price = current_price * (1 - remaining_decrease_percent / 100)
        print(f"Proposed price: {proposed_price:.2f} Action: {action}")
        
        if slope > 0:
            print("Market trending upwards")
            if min_proximity <= 0.2 or u.are_close(min_proximity, 0.2, target_tolerance_percent=1.0):
                if min_position >= 0.8 or u.are_close(min_position, 0.8, target_tolerance_percent=1.0):
                    action = 'BUY'
                    print(f"Near recent low. Action: {action}")
                    proposed_price = current_price * 0.995
                    print(f"Proposed price updated  to {proposed_price} to be close to current price {current_price}")
        else:
            print("Market trending downwards")
            if max_proximity <= 0.2 or u.are_close(max_proximity, 0.2, target_tolerance_percent=1.0):
                if max_position >= 0.8 or u.are_close(max_position, 0.8, target_tolerance_percent=1.0):
                    action = 'SELL'
                    print(f"Near recent high. Action: {action}")
                    proposed_price = current_price * 1.005
                    print(f"Proposed price updated  to {proposed_price} to be close to current price {current_price}")

        return action, proposed_price, price_change_percent, slope

    def check_price_change(self, threshold):
        if len(self.prices) < 2:
            return None
        oldest_price = self.prices[0]
        newest_price = self.prices[-1]
        price_diff = newest_price - oldest_price
        if abs(price_diff) >= threshold or u.are_close(price_diff, threshold) :
            return price_diff
        else:
            return None
            
    def current_window_size(self):
        return len(self.prices)


TIME_SLEEP_GET_PRICE = 2  # seconds to sleep for price collection
EXP_TIME_BUY_ORDER = (2.6 * 60) * 60 # After 1.6 hours.
EXP_TIME_SELL_ORDER = EXP_TIME_BUY_ORDER
TIME_SLEEP_EVALUATE = TIME_SLEEP_GET_PRICE + 60  # seconds to sleep for buy/sell evaluation
# We are allowed 6 orders per expiry period, which is 2.6 hours; that is why it is divided by 6.
TIME_SLEEP_PLACE_ORDER = TIME_SLEEP_EVALUATE + EXP_TIME_SELL_ORDER/ 6 + 4*79  # seconds to sleep for order placement
WINDOWS_SIZE_MIN = TIME_SLEEP_GET_PRICE + 4  # minutes
window_size = WINDOWS_SIZE_MIN * 60 / TIME_SLEEP_GET_PRICE

window_size2 = 2 * 60 * 60 / TIME_SLEEP_GET_PRICE
SELL_BUY_THRESHOLD = 3  # Threshold for the number of consecutive signals


def track_and_place_order(action, proposed_price, current_price, slope, quantity=0.017/2, order_placed=False, order_id=None):
    
    if action == 'HOLD':
        return order_placed, order_id
    
    api.cancel_expired_orders(action, sym.btcsymbol, EXP_TIME_BUY_ORDER if action == 'BUY' else EXP_TIME_SELL_ORDER)
        
    # Determine the number of orders and their spacing based on price trend
    if slope is not None and slope > 0:
        # Price is rising, place fewer, larger orders
        num_orders = 3
        price_step = 0.2  # Increase the spacing between orders as procents
        print(f"Placing fewer, {num_orders} larger orders due to rising price.")
    else:
        # Price is falling, place more, smaller orders
        num_orders = 3
        price_step = 0.08  # Reduce the spacing between orders as procents
        print(f"Placing more, {num_orders} smaller orders due to falling price.")

    if action == 'BUY':
        buy_price = min(proposed_price, current_price * 0.998)
        print(f"BUY price: {buy_price:.2f} USDT")

        alert.check_alert(True, f"BUY order {buy_price:.2f}")
       
        # Place the custom buy orders
        for i in range(num_orders):
            adjusted_buy_price = buy_price * (1 - i * price_step / 100)
            order_quantity = quantity / num_orders  # Divide quantity among orders
            print(f"Placing buy order at price: {adjusted_buy_price:.2f} USDT for {order_quantity:.6f} BTC")
            order = po.place_order_smart("BUY", sym.btcsymbol, adjusted_buy_price, order_quantity)
            if order:
                print(f"Buy order placed successfully with ID: {order['orderId']}")
                order_placed = True
                order_id = order['orderId']

    elif action == 'SELL':
        sell_price = max(proposed_price, current_price * 1.002)
        print(f"SELL price: {sell_price:.2f} USDT")
        
        alert.check_alert(True, f"SELL order {sell_price:.2f}")

        # Place the custom sell orders
        for i in range(num_orders):
            adjusted_sell_price = sell_price * (1 + i * price_step / 100)
            order_quantity = quantity / num_orders  # Divide quantity among orders
            print(f"Placing sell order at price: {adjusted_sell_price:.2f} USDT for {order_quantity:.6f} BTC")
            order = po.place_order_smart("SELL", sym.btcsymbol, adjusted_sell_price, order_quantity)
            if order:
                print(f"Sell order placed successfully with ID: {order['orderId']}")
                order_placed = True
                order_id = order['orderId']

    return order_placed, order_id  # Return the updated order state


import time

class TrendState:
    def __init__(self, max_duration_seconds, expiration_threshold):
        self.state = 'HOLD'  # Initial, starea este 'HOLD'
        self.old_state = self.state 
        self.start_time = None  # Timpul de inceput al trendului
        self.end_time = None  # Timpul de sfarsit al trendului
        self.last_confirmation_time = None  # The last confirmation time of the trend.
        self.max_duration_seconds = max_duration_seconds  # The maximum duration allowed for a trend.
        self.confirm_count = 0  # The confirmation counter for the trend.
        self.expiration_threshold = expiration_threshold  # The time threshold between confirmations (in seconds).

    def start_trend(self, new_state):
        
        self.end_trend()  # Marcheaza sfarsitul trendului anterior
        
        self.state = new_state
        self.start_time = time.time()
        self.last_confirmation_time = self.start_time
        self.confirm_count = 1  # The first confirmation.
        self.end_time = None  # Resetam timpul de sfarsit
        print(f"Trend started: {self.state} at {time.ctime(self.start_time)}")
        return self.old_state

    def confirm_trend(self):
        self.last_confirmation_time = time.time()
        self.confirm_count += 1
        print(f"Trend confirmed: {self.state} at {time.ctime(self.last_confirmation_time)}")

    def check_trend_expiration(self):
        """Check whether the trend has expired for lack of confirmations within the allowed interval."""
        if self.last_confirmation_time:
            time_since_last_confirmation = time.time() - self.last_confirmation_time
            if time_since_last_confirmation > self.expiration_threshold:
                print(f"Trend expired: {self.state}. Time since last confirmation: {time_since_last_confirmation} seconds")
                self.end_trend()
                return True
        return False

    def end_trend(self):
        self.old_state = self.state
        self.end_time = self.last_confirmation_time  # The trend's end time is the last confirmation time.
        self.confirm_count = 0
        print(f"Trend ended: {self.state} at {time.ctime(self.end_time)} after {self.confirm_count} confirmations.")
  
    def is_trend_up(self):
        if not self.check_trend_expiration() and self.state == 'UP':
            return self.confirm_count
        return 0

    def is_trend_down(self):
        if not self.check_trend_expiration() and self.state == 'DOWN':
            return self.confirm_count
        return 0

    def is_hold(self):
        if self.check_trend_expiration() or self.state == 'HOLD':
            return self.confirm_count
        return 0
        
trend_state = TrendState(max_duration_seconds= 2 * 60 * 60, expiration_threshold=10 * 60)  # Expira in 10 minute
trend_state2 = TrendState(max_duration_seconds= 2 * 60 * 60, expiration_threshold=10 * 60)  # Expira in 10 minute

#
#       MAIN 
#

alert.check_alert(True, f"SELL order ")
  

price_window = PriceWindow(window_size)
price_window2 = PriceWindow(window_size2)

order_placed = False
order_id = None
last_order_time = time.time()
last_evaluate_time = time.time()

# Counters for BUY and SELL evaluations
buy_count = 0
sell_count = 0

PRICE_CHANGE_THRESHOLD_EUR = 260

while True:
    try:
        time.sleep(TIME_SLEEP_GET_PRICE)

        current_time = time.time()
        current_price = api.get_current_price(sym.btcsymbol)
        if current_price is None:
            time.sleep(TIME_SLEEP_GET_PRICE)
            continue

        price_window.process_price(current_price)
        price_window2.process_price(current_price)
           
        # Confirmarea trendului folosind `evaluate_buy_sell_opportunity`
        action, proposed_price, price_change_percent, slope = price_window.evaluate_buy_sell_opportunity(
            current_price, threshold_percent=0.8, decrease_percent=7
        )

        if action == 'BUY':
            if trend_state.is_trend_up():
                trend_state.confirm_trend()  # Confirmam trendul de crestere
               
        elif action == 'SELL':
            if trend_state.is_trend_down():
                trend_state.confirm_trend()  # Confirmam trendul de scadere


        # Periodically check whether the current trend has expired.
        if trend_state.check_trend_expiration():
            expired_trend = trend_state.state  # Remember the trend that expired.
            trend_state.end_trend()  # Marcheaza sfarsitul trendului
            # Place orders at the end of a trend.
            if expired_trend == 'UP':
                proposed_price = proposed_price + 112  # A sell price.
                print(f"End of UP trend. SELL order at {proposed_price:.2f} EUR")
                #order_placed, order_id = track_and_place_order('SELL', proposed_price, current_price, slope=None, order_placed=order_placed, order_id=order_id)
            elif expired_trend == 'DOWN':
                #proposed_price = proposed_price - 242  # A buy price.
                print(f"End of DOWN trend. BUY order at {proposed_price:.2f} EUR")
                #order_placed, order_id = track_and_place_order('BUY', proposed_price - 242, current_price, slope=None, order_placed=order_placed, order_id=order_id)
            #last_order_time = current_time

        # Check the price changes and manage the trends.
        price_change = price_window.check_price_change(PRICE_CHANGE_THRESHOLD_EUR)
        
        if price_change is not None and price_change > 0:
            # Confirmam un trend de crestere
            print("DIFERENTA MARE!")
            if trend_state.is_trend_up():
                trend_state.confirm_trend()  # Confirmam ca trendul de crestere continua
            else:
                expired_trend = trend_state.start_trend('UP')  # Incepem un trend nou de crestere
                order_placed, order_id = track_and_place_order('BUY', proposed_price, current_price, slope=None, order_placed=order_placed, order_id=order_id)
                # If the previous trend was DOWN, we buy at the start of the UP trend.
                # if expired_trend == 'DOWN':
                proposed_price = proposed_price - 142
                print(f"Start of UP trend. BUY order at {proposed_price:.2f} EUR")
                
                order_placed, order_id = track_and_place_order('BUY', proposed_price, current_price, slope=None, order_placed=order_placed, order_id=order_id)
                #last_order_time = current_time

        elif price_change is not None and price_change < 0:
            # Confirmam un trend de scadere
            if trend_state.is_trend_down():
                trend_state.confirm_trend()  # Confirmam ca trendul de scadere continua
            else:
                expired_trend = trend_state.start_trend('DOWN')  # Incepem un trend nou de scadere

                # If the previous trend was UP, we sell at the start of the DOWN trend.
                # if expired_trend == 'UP':
                proposed_price = proposed_price + 142
                print(f"Start of DOWN trend. SELL order at {proposed_price:.2f} EUR")
                #order_placed, order_id = track_and_place_order('SELL', proposed_price, current_price, slope=None, order_placed=order_placed, order_id=order_id)
                #last_order_time = current_time


   
 
        # Reset the price window after the action.
        #price_window = PriceWindow(window_size)           


        ########

        # Evaluate buy/sell opportunity more frequently
        if current_time - last_evaluate_time >= TIME_SLEEP_EVALUATE:
        
            api.cancel_expired_orders("BUY", sym.btcsymbol, EXP_TIME_BUY_ORDER)
            api.cancel_expired_orders("SELL", sym.btcsymbol, EXP_TIME_SELL_ORDER)
            
            action, proposed_price, price_change_percent, slope = price_window2.evaluate_buy_sell_opportunity(current_price, threshold_percent=0.8, decrease_percent=4)
            last_evaluate_time = current_time

            # Count consecutive BUY/SELL actions
            if action == 'BUY':
                buy_count += 1
                sell_count -= 1  # Reset sell count if a BUY signal is received
            elif action == 'SELL':
                sell_count += 1
                buy_count -= 1  # Reset buy count if a SELL signal is received
            else:
                buy_count -= 1
                sell_count -= 1
            
            if buy_count < 0:
                buy_count = 0
            if sell_count < 0:
                sell_count = 0
            
            
        diff_count = abs(buy_count - sell_count)
        # Place orders based on the threshold
        if current_time - last_order_time >= TIME_SLEEP_PLACE_ORDER and \
        (u.are_close(diff_count, SELL_BUY_THRESHOLD, 1) or diff_count >= SELL_BUY_THRESHOLD)  :
            if abs(last_evaluate_time - time.time()) > 2:
                action, proposed_price, price_change_percent, slope = price_window2.evaluate_buy_sell_opportunity(current_price, threshold_percent=0.8, decrease_percent=4)
            if action == 'HOLD':
                continue
            last_evaluate_time = time.time()
            if buy_count >= sell_count :
                print(f"Multi confirmed. BUY order at {proposed_price:.2f} USDT")
                order_placed, order_id = track_and_place_order('BUY', proposed_price, current_price, slope, order_placed=order_placed, order_id=order_id)
                #buy_count = 0  # Reset buy count after placing the order
            else:
                print(f"Multi confirmed. SELL order at {proposed_price:.2f} USDT")
                order_placed, order_id = track_and_place_order('SELL', proposed_price, current_price, slope, order_placed=order_placed, order_id=order_id)
                #sell_count = 0  # Reset sell count after placing the order
            last_order_time = current_time

    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
        time.sleep(TIME_SLEEP_GET_PRICE)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(TIME_SLEEP_GET_PRICE)

