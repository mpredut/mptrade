# Dead module with no importers, replaced by ``cacheManager.CacheTradeManager``.
# Retained as a reference; do not run it directly.
import json
import os
import time
from collections import defaultdict
import threading

#my imports
import log
import utils as u
import symbols as sym
from binance_api import bapi as api

SYNC_INTERVAL_SEC = 600/10*5  # 10 minute

class TradeCacheManager:
    def __init__(self, filename="cache.json"):
        self.filename = filename
        self.trade_cache = []
        self.last_fetch_time_per_symbol = {}

        self.load_state()

    def load_state(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    data = json.load(f)
                    self.trade_cache = data.get("trades", [])
                    self.last_fetch_time_per_symbol = data.get("last_fetch", {})
            except Exception as e:
                print(f"[Error] While reading the cache file: {e}")

        if not self.last_fetch_time_per_symbol:
            self.last_fetch_time_per_symbol = self._rebuild_last_fetch_times()

    def get_all_symbols_from_cache(self):
        return list(set(t.get("symbol") for t in self.trade_cache if "symbol" in t))

    def _rebuild_last_fetch_times(self):
        # Derive each symbol's most recent query time from cache.
        last_times = defaultdict(int)
        for trade in self.trade_cache:
            symbol = trade.get("symbol")
            time_ = trade.get("time", 0)
            if time_ > last_times[symbol]:
                last_times[symbol] = time_

        # Sixty-second safety offset.
        for symbol in last_times:
            last_times[symbol] = max(0, last_times[symbol] - 60_000)

        if not last_times:
            # Fall back to the file timestamp.
            if os.path.exists(self.filename):
                fallback_time = int(os.path.getmtime(self.filename) * 1000) - 60_000
            else:
                fallback_time = 0
            return {symbol: fallback_time for symbol in self.get_all_symbols_from_cache()}
        
        return dict(last_times)


    def save_state(self):
        try:
            from state_io import atomic_write_json
            atomic_write_json(self.filename, {
                "trades": self.trade_cache,
                "last_fetch": self.last_fetch_time_per_symbol
            })
        except Exception as e:
            print(f"[Error] While saving the cache file: {e}")

    def _is_valid_trade(self, trade):
        required_keys = ['symbol', 'id', 'orderId', 'price', 'qty', 'time', 'isBuyer']
        return all(k in trade for k in required_keys)
    
    
    def _get_my_trades(self, symbol, startTime):
        try:
            # ``startTime`` is the parameter; the former undefined name raised NameError.
            new_trades = api.client.get_my_trades(symbol=symbol, startTime=startTime)
        except Exception as e:
            print(f"[Error] Binance API for {symbol}: {e}")
            return []

        # Deduplicate by ID when present, otherwise by time and symbol.
        existing_keys = set(
            (t.get("id"), t["symbol"]) for t in self.trade_cache if "id" in t
        )

        unique_new_trades = []
        for t in new_trades:
            if not self._is_valid_trade(t):
                print(f"BED DAY???")
                continue 
            key = (t.get("id"), t["symbol"]) if "id" in t else (t["symbol"], t["time"])
            if key not in existing_keys:
                unique_new_trades.append(t)
                existing_keys.add(key)
                
        return unique_new_trades
                
    def update_symbol_from_binance(self, symbol):
        # Use current time as the endTime reference.
        current_time = int(time.time() * 1000)
        start_time = self.last_fetch_time_per_symbol.get(symbol, 0)

        unique_new_trades = self._get_my_trades(symbol=symbol, startTime=start_time)
        self.trade_cache.extend(unique_new_trades)
        self.last_fetch_time_per_symbol[symbol] = current_time

        print(f"[Info] {symbol}: added {len(unique_new_trades)} new trades.")

    def update_all(self, symbols):
        for symbol in symbols:
            self.update_symbol_from_binance(symbol)
        self.save_state()


def periodic_sync():
    print(f"\n--- Sync started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    cache_manager.update_all(sym.symbols)
    print(f"--- Sync completed ---")
    
    # Schedule the next run.
    t = threading.Timer(SYNC_INTERVAL_SEC, periodic_sync, args=())
    t.daemon = True  # Do not let this thread block process shutdown.
    t.start()


if __name__ == "__main__":
    cache_manager = TradeCacheManager("cache.json")
    periodic_sync()
