import os
import time
import datetime
import math
from collections import deque

####Binance
#from binance.exceptions import BinanceAPIException

# my imports
import log
import alertnotifiers as alert
import utils as u
import symbols as sym
import bapi as api
import bapi_placeorder as po

#import priceprediction as pp


# The time interval between attempts to cancel and recreate the order (in seconds).
WAIT_FOR_ORDER = 22

class TradingBot:
    def __init__(self, symbol, qty, DEFAULT_ADJUSTMENT_PERCENT):
        self.symbol = symbol
        self.qty = qty
        self.transaction_state = "COMPLETED"  # Starea initiala
        current_price = api.get_current_price(symbol)
        self.filled_buy_price = round(current_price * (1 - 0.1), 4)
        self.filled_sell_price = round(current_price * (1 + 0.1), 4)
        self.buy_filled = False
        self.sell_filled = False
        self.DEFAULT_ADJUSTMENT_PERCENT = DEFAULT_ADJUSTMENT_PERCENT
        
    def mark_buy_filled(self) :
        self.buy_filled = True
        self.sell_filled = False
        
    def mark_sell_filled(self) :
        self.buy_filled = False
        self.sell_filled = True
        
    def repetitive_buy(self, current_price, filled_sell_price):
        adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT
        failure_count = 0  # A counter for the number of failures.
        max_failures = 5  # Definim numarul maxim de esecuri acceptabile

        while True:
            target_buy_price = round(current_price * (1 - adjustment_percent), 4)
            print(f"[{self.symbol}] Order BUY initiated at {target_buy_price:.2f} procent {adjustment_percent}%")
            
            if self.buy_filled:
                print(f"[{self.symbol}] Ignore BUY order. It was previously filled at {self.filled_buy_price:.2f}")
                return self.filled_buy_price
            
            if adjustment_percent > 0:
                buy_order = po.place_safe_order("BUY", self.symbol, target_buy_price, self.qty)
            else:
                buy_order = po.place_safe_order("BUY", self.symbol, target_buy_price, self.qty)

            if buy_order is None:
                print(f"[{self.symbol}] Order BUY failed, retrying...")
                api.cancel_recent_orders("BUY", self.symbol, WAIT_FOR_ORDER)
                time.sleep(WAIT_FOR_ORDER)
                failure_count += 1
                if failure_count >= max_failures:
                    print(f"[{self.symbol}] Order BUY failed {failure_count} times. Exiting.")
                    self.mark_buy_filled()
                    return round(api.get_current_price(self.symbol) * (1 - 0.01), 4)
                continue

            time.sleep(WAIT_FOR_ORDER)
            order_id = buy_order['orderId']
            self.filled_buy_price = round(float(buy_order['price']), 4)
            
            if api.check_order_filled(order_id, self.symbol):
                print(f"[{self.symbol}] BUY order filled at {self.filled_buy_price:.2f}")
                print(f"[{self.symbol}] SELL disperat tot....")
                po.place_order_smart("SELL", self.symbol, api.get_current_price(self.symbol) * (1 + 0.01), 0.2, 
                    force=True, cancelorders=True, hours=1)
                self.mark_buy_filled()
                return self.filled_buy_price
                
            filled_buy_price = api.check_order_filled_by_time("BUY", self.symbol, time_back_in_seconds=WAIT_FOR_ORDER)
            if filled_buy_price is not None:
                print(f"[{self.symbol}] BUY order may have been filled :-) at {filled_buy_price:.2f}")
                print(f"[{self.symbol}] SELL disperat tot....")
                po.place_order_smart("SELL", self.symbol, api.get_current_price(self.symbol) * (1 + 0.01), 0.2, 
                    force=True, cancelorders=True, hours=1)
                self.mark_buy_filled()
                self.filled_buy_price = filled_buy_price
                return self.filled_buy_price
                
            current_price = api.get_current_price(self.symbol)
            if current_price > filled_sell_price:
                print(f"[{self.symbol}] Bed day :-(. Trying BUY at current price - x2 {current_price:.2f}")
                adjustment_percent = 2 * self.DEFAULT_ADJUSTMENT_PERCENT
            else:
                adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT

            # if arrived here it means 
            # current order was not filled , so try cancel and retry in the loop
            if not api.cancel_order(self.symbol, order_id):
                if api.check_order_filled(order_id, self.symbol):
                    print(f"[{self.symbol}] Cancel BUY order failed. Maybe it was filled :-)? Moving to SELL ...")
                    print(f"[{self.symbol}] SELL disperat tot....")
                    po.place_order_smart("SELL", self.symbol, api.get_current_price(self.symbol) * (1 + 0.01), 0.2, 
                    force=True, cancelorders=True, hours=1)
                    self.mark_buy_filled()
                    return self.filled_buy_price
                else:
                    print(f"[{self.symbol}] Cancel BUY order failed. Someone canceled it. Continuing BUY...")


    def repetitive_sell(self, current_price, filled_buy_price):
        adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT
        failure_count = 0  # A counter for the number of failures.
        max_failures = 5  # Definim numarul maxim de esecuri acceptabile

        while True:
            target_sell_price = round(current_price * (1 + adjustment_percent), 4)
            print(f"[{self.symbol}] Order SELL initiated at {target_sell_price:.2f} procent {adjustment_percent}%")

            if self.sell_filled:
                print(f"[{self.symbol}] Ignore SELL order. It was previously filled at {self.filled_sell_price:.2f}")
                return self.filled_sell_price

            if adjustment_percent > 0:
                sell_order = po.place_safe_order("SELL", self.symbol, target_sell_price, self.qty)
            else:
                sell_order = po.place_safe_order("SELL", self.symbol, target_sell_price, self.qty)

            if sell_order is None:
                print(f"[{self.symbol}] Order SELL failed, retrying...")
                api.cancel_recent_orders("SELL", self.symbol, WAIT_FOR_ORDER)
                time.sleep(WAIT_FOR_ORDER)
                failure_count += 1  # Incrementam contorul de esecuri
                if failure_count >= max_failures:
                    print(f"[{self.symbol}] Order SELL failed {failure_count} times. Exiting.")
                    self.mark_sell_filled()
                    return round(api.get_current_price(self.symbol) * (1 + 0.1), 4)
                continue

            time.sleep(WAIT_FOR_ORDER)
            order_id = sell_order['orderId']
            self.filled_sell_price = round(float(sell_order['price']), 4)

            if api.check_order_filled(order_id, self.symbol):
                print(f"[{self.symbol}] SELL order filled at {self.filled_sell_price:.2f}")
                print(f"[{self.symbol}] BUY disperat tot....")
                po.place_order_smart("BUY", self.symbol, api.get_current_price(self.symbol) * (1 - 0.01), 0.2, 
                    force=True, cancelorders=True, hours=1)
                self.mark_sell_filled()
                return self.filled_sell_price

            filled_sell_price = api.check_order_filled_by_time("SELL", self.symbol, time_back_in_seconds=WAIT_FOR_ORDER)
            if filled_sell_price is not None:
                print(f"[{self.symbol}] SELL order may have been filled :-) at {filled_sell_price:.2f}")
                print(f"[{self.symbol}] BUY disperat tot....")
                po.place_order_smart("BUY", self.symbol, api.get_current_price(self.symbol) * (1 - 0.01), 0.2, 
                    force=True, cancelorders=True, hours=1)
                self.mark_sell_filled()
                self.filled_sell_price = filled_sell_price
                return self.filled_sell_price

            current_price = api.get_current_price(self.symbol)
            if current_price < filled_buy_price:
                print(f"[{self.symbol}] Bed day :-(. Trying SELL at current price + x2 {current_price:.2f}")
                adjustment_percent = 2 * self.DEFAULT_ADJUSTMENT_PERCENT
            else:
                adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT
           
            # if arrived here it means 
            # current order was not filled , so try cancel and retry in the loop
            if not api.cancel_order(self.symbol, order_id):
                if api.check_order_filled(order_id, self.symbol):
                    print(f"[{self.symbol}] Cancel SELL order failed. Maybe it was filled :-)? Moving to BUY ...")
                    print(f"[{self.symbol}] BUY disperat tot....")
                    po.place_order_smart("BUY", self.symbol, api.get_current_price(self.symbol) * (1 - 0.01), 0.2, 
                        force=True, cancelorders=True, hours=1)                     
                    self.buy_filled = False
                    self.sell_filled = True                   
                    return self.filled_sell_price
                else:
                    print(f"[{self.symbol}] Cancel SELL order failed. Someone canceled it. Continuing sell...")


    def run(self):
        while True:
            try:
                current_price = api.get_current_price(self.symbol)
                print(f"[{self.symbol}] Current price: {current_price:.2f}")

                filled_sell_price = self.repetitive_sell(current_price, self.filled_buy_price)
                current_price = api.get_current_price(self.symbol)
                filled_buy_price = self.repetitive_buy(current_price, filled_sell_price) + 0.0001 ## avoid zero

                print(f"[{self.symbol}] Transaction complete: Bought at {filled_buy_price:.2f}, Sold at {filled_sell_price:.2f}")
                if filled_buy_price < filled_sell_price:
                    print(f"[{self.symbol}] PROFIT: Profit ratio {filled_sell_price / filled_buy_price:.2f}")
                else:
                    print(f"[{self.symbol}] LOSS: Loss ratio {filled_sell_price / filled_buy_price:.2f}")

                time.sleep(1)

                if self.buy_filled == self.sell_filled:
                    self.buy_filled = not self.sell_filled
            except Exception as e:
                print(f"[{self.symbol}] Unexpected error: {e}")
                if self.buy_filled == self.sell_filled:
                    self.buy_filled = not self.sell_filled
                api.cancel_recent_orders("SELL", self.symbol, WAIT_FOR_ORDER)
                api.cancel_recent_orders("BUY", self.symbol, WAIT_FOR_ORDER)
                time.sleep(1)


DEFAULT_ADJUSTMENT_PERCENT = round(u.calculate_difference_percent(60000, 60000 - 380) / 100, 4)
print(f"[INFO] DEFAULT_ADJUSTMENT_PERCENT = {DEFAULT_ADJUSTMENT_PERCENT}")

bot = TradingBot(sym.taosymbol, 0.017, DEFAULT_ADJUSTMENT_PERCENT=DEFAULT_ADJUSTMENT_PERCENT)
bot.run()

    
