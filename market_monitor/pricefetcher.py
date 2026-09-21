# pricefetcher.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import threading
import requests
import os
from state_io import atomic_write_json
from datetime import datetime
from typing import Dict, Optional, List, Set, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

# Import your existing modules
import log
import utils as u
import symbols as sym
from binance_api import bapi as api

# Import the base classes from cacheManager
from cacheManager import CacheManagerInterface, CacheFactory, _should_poll_for_manager
# Market-data facade. market_api imports binance_api.bapi without importing pricefetcher,
# so this direction is safe. BinancePricePlatform uses the facade as its price client.
import providers.market_api as _market_api
from providers.quantity import resolve_assets


# ============================================
# Global configuration.
# ============================================

PRICE_HISTORY_RETENTION_DAYS = 7
MAX_PRICE_HISTORY_PER_SYMBOL = 2000
MAX_MONITORED_SYMBOLS = 20
QUOTE_CURRENCY = "USDC"
FALLBACK_QUOTE = "USDC"
#DEFAULT_SYMBOLS = ["BTC", "ETH", "HYPE", "SOL", "BNB", "ADA", "DOGE", "XRP"]
DEFAULT_SYMBOLS = ["BTC", "TAO", "HYPE"]


# ============================================
# Specialized price platforms.
# ============================================

class PricePlatformInterface(ABC):
    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        pass
    
    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> Set[str]:
        pass
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        pass
    
    def refresh_symbols(self):
        pass


class BinancePricePlatform(PricePlatformInterface):
    def __init__(self, api_client=None):
# The price client is the market-data facade. Binance pairs route to bapi with
# identical behavior. ``self.api_client`` remains an object exposing
        # .get_current_price(symbol=...), just like bapi.
        self.api_client = api_client or _market_api.api
        self._supported_symbols: Set[str] = set()
        self._usdc_pairs: Set[str] = set()
        self._symbol_mapping: Dict[str, str] = {}
        self._last_refresh = 0
        self._refresh_interval = 3600
        self._load_symbols()
    
    @property
    def platform_name(self) -> str:
        return "Binance"
    
    def _load_symbols(self):
        try:
            response = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            self._usdc_pairs.clear()
            self._supported_symbols.clear()
            self._symbol_mapping.clear()
            
            for symbol_info in data.get("symbols", []):
                symbol = symbol_info.get("symbol")
                base_asset = symbol_info.get("baseAsset")
                quote_asset = symbol_info.get("quoteAsset")
                status = symbol_info.get("status")
                
                if status != "TRADING":
                    continue
                
                self._supported_symbols.add(symbol)
                
                if quote_asset == "USDC":
                    self._usdc_pairs.add(symbol)
                    if base_asset not in self._symbol_mapping:
                        self._symbol_mapping[base_asset] = symbol
                    self._symbol_mapping[symbol] = symbol
                
            print(f"[BinancePlatform] USDC: {len(self._usdc_pairs)} pairs")
            self._last_refresh = time.time()

        except Exception as e:
            print(f"[BinancePlatform] Error loading symbols: {e}")
            self._fallback_symbols()
            self._last_refresh = time.time()
    
    def _fallback_symbols(self):
        self._usdc_pairs = {"BTCUSDC", "ETHUSDC", "BNBUSDC"}
        self._supported_symbols = self._usdc_pairs
        self._symbol_mapping = {
            "BTC": "BTCUSDC", "ETH": "ETHUSDC", "BNB": "BNBUSDC",
            "SOL": "SOLUSDC", "ADA": "ADAUSDC", "DOGE": "DOGEUSDC", "XRP": "XRPUSDC"
        }
    
    def refresh_symbols(self):
        if time.time() - self._last_refresh > self._refresh_interval:
            self._load_symbols()
    
    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return self._supported_symbols.copy()
    
    def supports_symbol(self, symbol: str) -> bool:
        self.refresh_symbols()
        if symbol in self._supported_symbols:
            return True
        if symbol in self._symbol_mapping:
            return True
        pair = f"{symbol}USDC"
        if pair in self._supported_symbols:
            return True
        return False
    
    def get_price(self, symbol: str) -> Optional[float]:
        try:
            trading_pair = None
            
            if symbol in self._symbol_mapping:
                trading_pair = self._symbol_mapping[symbol]
            elif f"{symbol}USDC" in self._supported_symbols:
                trading_pair = f"{symbol}USDC"
            elif symbol in self._supported_symbols:
                trading_pair = symbol
            
            if not trading_pair:
                print(f"[BinancePlatform] No trading pair found for {symbol}")
                return None

            price = self.api_client.get_current_price(symbol=trading_pair)
            print(f"[BinancePlatform] {symbol} -> {trading_pair} = ${price}")
            return float(price)

        except Exception as e:
            print(f"[BinancePlatform] Error for {symbol}: {e}")
            return None


class HyperliquidPricePlatform(PricePlatformInterface):
    def __init__(self):
        self._url = "https://api.hyperliquid.xyz/info"
        self._supported_symbols: Set[str] = set()
        self._all_mids: Dict[str, float] = {}
        self._last_refresh = 0
        self._refresh_interval = 300
        self._load_symbols()
    
    @property
    def platform_name(self) -> str:
        return "Hyperliquid"
    
    def _load_symbols(self):
        try:
            payload = {"type": "allMids"}
            headers = {"Content-Type": "application/json"}
            response = requests.post(self._url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            self._all_mids = response.json()
            self._supported_symbols = set(self._all_mids.keys())
            print(f"[HyperliquidPlatform] Loaded {len(self._supported_symbols)} symbols")
            self._last_refresh = time.time()
        except Exception as e:
            print(f"[HyperliquidPlatform] Error loading symbols: {e}")
            self._supported_symbols = {"HYPE", "PURR", "BTC", "ETH", "SOL", "USDC"}
            self._last_refresh = time.time()
    
    def refresh_symbols(self):
        if time.time() - self._last_refresh > self._refresh_interval:
            self._load_symbols()
    
    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return self._supported_symbols.copy()
    
    def supports_symbol(self, symbol: str) -> bool:
        self.refresh_symbols()
        return symbol in self._supported_symbols
    
    def get_price(self, symbol: str) -> Optional[float]:
        try:
            self.refresh_symbols()
            if symbol not in self._all_mids:
                print(f"[HyperliquidPlatform] Symbol {symbol} not found")
                return None
            return float(self._all_mids[symbol])
        except Exception as e:
            print(f"[HyperliquidPlatform] Error for {symbol}: {e}")
            return None


class CoinMarketCapPricePlatform(PricePlatformInterface):
    def __init__(self, api_key: Optional[str] = None):
        # Define ``api_key`` before use.
        self.api_key = api_key or os.environ.get('CMC_API_KEY')
        self._base_url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
        self._supported_symbols: Set[str] = set()
        self._all_listings: Dict[str, Dict] = {}
        self._last_refresh = 0
        self._refresh_interval = 3600
        
        if self.api_key:
            self._load_symbols()
        else:
            print("[CMCPlatform] No API key - platform disabled")

    @property
    def platform_name(self) -> str:
        return "CoinMarketCap"
    
    def _load_symbols(self):
        if not self.api_key:
            return
        try:
            headers = {'X-CMC_PRO_API_KEY': self.api_key, 'Accept': 'application/json'}
            params = {'limit': 5000, 'convert': 'USD'}
            response = requests.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                headers=headers, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()
            self._supported_symbols.clear()
            for crypto in data.get('data', []):
                symbol = crypto.get('symbol')
                if symbol:
                    self._supported_symbols.add(symbol)
                    self._all_listings[symbol] = {
                        'id': crypto.get('id'),
                        'name': crypto.get('name'),
                        'slug': crypto.get('slug')
                    }
            print(f"[CMCPlatform] Loaded {len(self._supported_symbols)} symbols")
            self._last_refresh = time.time()
        except Exception as e:
            print(f"[CMCPlatform] Error loading symbols: {e}")
            self._last_refresh = time.time()  # Do not retest immediately; honour _refresh_interval (3600s).

    def refresh_symbols(self):
        if self.api_key and time.time() - self._last_refresh > self._refresh_interval:
            self._load_symbols()
    
    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return self._supported_symbols.copy()
    
    def supports_symbol(self, symbol: str) -> bool:
        if not self.api_key:
            return False
        self.refresh_symbols()
        return symbol in self._supported_symbols

    def _extract_usd_price(self, symbol: str, data: Dict) -> Optional[float]:
        coin_data = data.get('data', {}).get(symbol)
        if isinstance(coin_data, list):
            coin_data = coin_data[0] if coin_data else None
        if isinstance(coin_data, dict):
            price = coin_data.get('quote', {}).get('USD', {}).get('price')
            if price is not None:
                return float(price)
        return None
    
    def get_price_old(self, symbol: str) -> Optional[float]:
        if not self.api_key:
            print(f"[CMCPlatform] Missing API key")
            return None
        try:
            headers = {'X-CMC_PRO_API_KEY': self.api_key, 'Accept': 'application/json'}
            params = {'symbol': symbol, 'convert': 'USD'}
            time.sleep(0.2)
            response = requests.get(self._base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            price = self._extract_usd_price(symbol, data)
            if price is not None:
                print(f"[CMCPlatform] {symbol} = ${price}")
                return price
            else:
                print(f"[CMCPlatform] {symbol} not found")
                return None
        except Exception as e:
            print(f"[CMCPlatform] Error for {symbol}: {e}")
            return None
        
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the price from the internal cache (updated from listings/latest).
        This is the fastest and safest way to get prices for new coins.
        """
        if not self.api_key:
            print(f"[CMCPlatform] Missing API key")
            return None

        # Refresh if enough time has passed
        self.refresh_symbols()

        # Search the symbol in the internal cache (_all_listings only contains metadata,
        # but we also need the price. We will use a separate dictionary for prices.
        # A simpler solution is to use CoinMarketCapSource directly. If there is no dedicated cache,
        # we will make a request to listings/latest for the specific symbol.
        try:
            headers = {'X-CMC_PRO_API_KEY': self.api_key, 'Accept': 'application/json'}
            params = {'symbol': symbol, 'convert': 'USD'}
            response = requests.get(
                self._base_url,
                headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            price = self._extract_usd_price(symbol, data)
            if price is not None:
                print(f"[CMCPlatform] {symbol} = ${price}")
                return price
            print(f"[CMCPlatform] {symbol} not found")
            return None
        except Exception as e:
            print(f"[CMCPlatform] Error for {symbol}: {e}")
            return None


class StockYahooPricePlatform(PricePlatformInterface):
    """Fetch configured tokenized stocks from Yahoo Finance as a last resort.

    Binance, Hyperliquid, and CMC claim crypto first, so only equity tickers reach this
    platform. It shares the T212/price-alert data source and prefers the latest intraday
    bar because metadata can be stale. Tickers come from the required
    ``STOCK_TICKERS`` configuration unless explicitly injected by the caller.
    """
    _YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    _UA = {"User-Agent": "Mozilla/5.0 (pricefetcher)"}

    def __init__(self, tickers: Optional[List[str]] = None):
        if tickers is None:
            configured = os.environ.get("STOCK_TICKERS", "").strip()
            if not configured:
                raise ValueError("STOCK_TICKERS is required when tickers are not injected")
            src = configured.split(",")
        else:
            src = tickers
        self._symbols: Set[str] = {t.strip().upper() for t in src if t.strip()}

    @property
    def platform_name(self) -> str:
        return "Yahoo"

    def get_available_symbols(self) -> Set[str]:
        return self._symbols.copy()

    def supports_symbol(self, symbol: str) -> bool:
        # Accept only configured tickers; do not divert crypto or make a network call here.
        return symbol.upper() in self._symbols

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            sym_y = symbol.split("_")[0]                      # NVDA_US_EQ -> NVDA
            url = self._YAHOO.format(sym=sym_y) + "?range=1d&interval=5m"
            resp = requests.get(url, headers=self._UA, timeout=10)
            resp.raise_for_status()
            result = (resp.json().get("chart", {}).get("result") or [None])[0]
            if not result:
                return None
            q = (result.get("indicators", {}).get("quote") or [{}])[0]
            closes = [c for c in (q.get("close") or []) if c is not None]
            if closes:
                return float(closes[-1])                      # freshest intraday bar
            price = (result.get("meta", {}) or {}).get("regularMarketPrice")  # metadata fallback
            return float(price) if price else None
        except Exception as e:
            print(f"[YahooPlatform] Error for {symbol}: {e}")
            return None


# ============================================
# Price Platform Factory
# ============================================

class PricePlatformFactory:
    def __init__(self, cmc_api_key: Optional[str] = None):
        self._platforms: List[PricePlatformInterface] = [
            BinancePricePlatform(),
            HyperliquidPricePlatform(),
        ]
        if cmc_api_key:
            self._platforms.append(CoinMarketCapPricePlatform(cmc_api_key))
        self._platforms.append(StockYahooPricePlatform())   # LAST: tokenised stocks (SPCX...).
        self._symbol_platform_cache: Dict[str, str] = {}
        self._discover_all_symbols()
    
    def _discover_all_symbols(self):
        print("[PriceFactory] 🔍 Discovering available symbols...")
        all_symbols = {}
        for platform in self._platforms:
            try:
                symbols = platform.get_available_symbols()
                all_symbols[platform.platform_name] = {
                    "count": len(symbols),
                    "sample": list(symbols)[:10]
                }
                print(f"[PriceFactory]   {platform.platform_name}: {len(symbols)} symbols")
            except Exception as e:
                print(f"[PriceFactory]   {platform.platform_name}: error - {e}")
        self._capabilities = all_symbols

    def get_price(self, symbol: str) -> Dict:
        cached_platform = None
        if symbol in self._symbol_platform_cache:
            platform_name = self._symbol_platform_cache[symbol]
            for platform in self._platforms:
                if platform.platform_name == platform_name:
                    cached_platform = platform
                    price = platform.get_price(symbol)
                    if price is not None:
                        return {
                            "symbol": symbol, "price": price,
                            "platform": platform.platform_name, "timestamp": int(time.time())
                        }
                    self._symbol_platform_cache.pop(symbol, None)
                    break
        for platform in self._platforms:
            if platform is cached_platform:
                continue
            if platform.supports_symbol(symbol):
                price = platform.get_price(symbol)
                if price is not None:
                    self._symbol_platform_cache[symbol] = platform.platform_name
                    return {
                        "symbol": symbol, "price": price,
                        "platform": platform.platform_name, "timestamp": int(time.time())
                    }
        raise Exception(f"Symbol '{symbol}' is not supported by any platform")

    def check_symbol_support(self, symbol: str) -> Dict:
        support = {}
        for platform in self._platforms:
            support[platform.platform_name] = platform.supports_symbol(symbol)
        return support


# ============================================
# CacheAllPriceFetcherManager
# ============================================

class CacheAllPriceFetcherManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api, cmc_api_key: Optional[str] = None):
        self.price_factory = PricePlatformFactory(cmc_api_key=cmc_api_key)
        self.max_monitored_symbols = MAX_MONITORED_SYMBOLS
        initial_symbols = []
        seen_bases = set()
        for symbol in symbols:
            #print(f"{symbol}")
            base_symbol = get_base_symbol(symbol)
            if base_symbol not in seen_bases and is_valid_symbol_for_monitoring(symbol):
                seen_bases.add(base_symbol)
                initial_symbols.append(symbol)
            if len(initial_symbols) >= MAX_MONITORED_SYMBOLS:
                break
        self.original_symbols = list(initial_symbols)
        super().__init__(sync_ts, initial_symbols, filename, append_mode=True, api_client=api_client)
        self.active_symbols = set(initial_symbols)
        self.symbol_added_time: Dict[str, float] = {}
        self.symbol_preferred_source: Dict[str, str] = {}  # Required source preference state.
        self._load_symbol_metadata()
        self._log_symbol_support()
    
    def _load_symbol_metadata(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    data = json.load(f)
                    if "symbol_metadata" in data:
                        self.symbol_added_time = data["symbol_metadata"].get("added_time", {})
                        seen_bases = {get_base_symbol(symbol) for symbol in self.active_symbols}
                        active_symbols = data["symbol_metadata"].get("active_symbols", [])
                        recovered_symbols = [
                            symbol for symbol, added_time in self.symbol_added_time.items()
                            if symbol in data.get("items", {}) and time.time() - added_time <= PRICE_HISTORY_RETENTION_DAYS * 24 * 3600
                        ]
                        for symbol in list(dict.fromkeys(active_symbols + recovered_symbols)):
                            if len(self.active_symbols) >= MAX_MONITORED_SYMBOLS:
                                print(f"[Pricefetcher] Limit reached while loading: maximum {MAX_MONITORED_SYMBOLS} monitored coins, the rest will be ignored")
                                break
                            base_symbol = get_base_symbol(symbol)
                            if (
                                symbol not in self.active_symbols
                                and base_symbol not in seen_bases
                                and is_valid_symbol_for_monitoring(symbol)
                                and any(self.price_factory.check_symbol_support(symbol).values())
                            ):
                                seen_bases.add(base_symbol)
                                self.active_symbols.add(symbol)
                                self.symbols.append(symbol)
                                self.original_symbols.append(symbol)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                print(f"[Pricefetcher] Metadata cache ignored: {exc}")
    
    def _log_symbol_support(self):
        print(f"[Pricefetcher] Checking symbol support:")
        for symbol in self.original_symbols:
            support = self.price_factory.check_symbol_support(symbol)
            supported_platforms = [p for p, s in support.items() if s]
            if supported_platforms:
                print(f"  ✅ {symbol} -> {', '.join(supported_platforms)}")
            else:
                print(f"  ❌ {symbol} -> NO PLATFORM SUPPORT!")

    def rebuild_fetchtime_times(self):
        if not self.cache:
            return {}
        last_times = {}
        for symbol in self.symbols:
            entries = self.cache.get(symbol, [])
            if entries:
                last_times[symbol] = max(entry[0] for entry in entries)
        return last_times
    
    def get_remote_items(self, symbol, startTime):
        try:
            # Ensure the attribute exists
            if not hasattr(self, 'symbol_preferred_source'):
                self.symbol_preferred_source = {}

            preferred_source = self.symbol_preferred_source.get(symbol)
            if preferred_source:
                for platform in self.price_factory._platforms:
                    if platform.platform_name == preferred_source:
                        price = platform.get_price(symbol)
                        if price is not None:
                            timestamp = int(time.time())
                            timestamp_ms = timestamp * 1000
                            print(f"[Pricefetcher][{symbol}] ${price:.4f} (source: {preferred_source} - preferred)")
                            return [[timestamp_ms, price]]
                        print(f"[Pricefetcher][{symbol}] Preferred source {preferred_source} failed; trying fallback providers")
                        break
            result = self.price_factory.get_price(symbol)
            price = result["price"]
            platform_used = result["platform"]
            timestamp = int(time.time())
            timestamp_ms = timestamp * 1000
            print(f"[Pricefetcher][{symbol}] ${price:.4f} (source: {platform_used})")
            return [[timestamp_ms, price]]
        except Exception as e:
            print(f"[Pricefetcher][Error] {symbol}: {e}")
            return []
    
    def add_symbol(self, symbol: str, preferred_source: Optional[str] = None):
        with self.lock:
            base_symbol = get_base_symbol(symbol)
            if symbol in self.active_symbols:
                print(f"[Pricefetcher] {symbol} is already in the watchlist")
                return False
            if any(get_base_symbol(active_symbol) == base_symbol for active_symbol in self.active_symbols):
                print(f"[Pricefetcher] {symbol} ignored: asset {base_symbol} is already being monitored")
                return False
            if len(self.active_symbols) >= MAX_MONITORED_SYMBOLS:
                print(f"[Pricefetcher] Limit reached: maximum {MAX_MONITORED_SYMBOLS} monitored coins")
                return False
            self.symbols.append(symbol)
            self.original_symbols.append(symbol)
            self.active_symbols.add(symbol)
            if preferred_source:
                self.symbol_preferred_source[symbol] = preferred_source
                print(f"[Pricefetcher] {symbol} - preferred source: {preferred_source}")
            self.symbol_added_time[symbol] = time.time()
            if symbol not in self.cache:
                self.cache[symbol] = []
            if symbol not in self.fetchtime_time_per_symbol:
                self.fetchtime_time_per_symbol[symbol] = self.fallback_time_default
            print(f"[Pricefetcher] ✅ Symbol added: {symbol}")
        # Fetch and store the new symbol's initial price outside this lock;
        # ``update_cache_per_symbol`` acquires its own lock.
        items = self.get_remote_items(
            symbol, self.fetchtime_time_per_symbol.get(symbol, self.fallback_time_default))
        if items:
            self.update_cache_per_symbol(symbol, items)
        return True
    
    def remove_symbol(self, symbol: str, reason: str = ""):
        with self.lock:
            if symbol not in self.active_symbols:
                return False
            if symbol in self.symbols:
                self.symbols.remove(symbol)
            if symbol in self.original_symbols:
                self.original_symbols.remove(symbol)
            self.active_symbols.discard(symbol)
            # Clear auxiliary state too; otherwise ``cleanup_old_symbols`` repeatedly
            # revisits the same symbol through ``symbol_added_time``.
            self.symbol_added_time.pop(symbol, None)
            if hasattr(self, "symbol_preferred_source"):
                self.symbol_preferred_source.pop(symbol, None)
            print(f"[Pricefetcher] ❌ Symbol removed: {symbol} {reason}")
            return True

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest saved price for a symbol."""
        with self.lock:
            entries = self.cache.get(symbol, [])
            if entries:
                return entries[-1][1]
        return None

    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return []
        # Copy only references under the lock. Convert and format afterward so datetime
        # work and allocations do not block price synchronization.
        with self.lock:
            entries = self.cache.get(symbol, ())[-limit:]

        return [
            {
                "timestamp": entry[0],
                "timestamp_readable": datetime.fromtimestamp(entry[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "price": entry[1],
            }
            for entry in entries
        ]
        
    def cleanup_old_prices(self, retention_days: int = PRICE_HISTORY_RETENTION_DAYS):
        cutoff_timestamp = (time.time() - retention_days * 24 * 3600) * 1000
        with self.lock:
            removed_count = 0
            trimmed_count = 0
            for symbol in list(self.cache.keys()):
                original_count = len(self.cache[symbol])
                self.cache[symbol] = [entry for entry in self.cache[symbol] if entry[0] >= cutoff_timestamp]
                removed = original_count - len(self.cache[symbol])
                if removed > 0:
                    removed_count += removed
                    print(f"[Cleanup] {symbol}: removed {removed} old entries")
                if len(self.cache[symbol]) > MAX_PRICE_HISTORY_PER_SYMBOL:
                    over_limit = len(self.cache[symbol]) - MAX_PRICE_HISTORY_PER_SYMBOL
                    self.cache[symbol] = self.cache[symbol][-MAX_PRICE_HISTORY_PER_SYMBOL:]
                    trimmed_count += over_limit
                    print(f"[Cleanup] {symbol}: trimmed to the latest {MAX_PRICE_HISTORY_PER_SYMBOL} entries")
                if not self.cache[symbol] and symbol not in self.active_symbols:
                    del self.cache[symbol]
                    print(f"[Cleanup] {symbol}: fully removed")
            if removed_count > 0:
                print(f"[Cleanup] Total: {removed_count} old entries removed")
            if trimmed_count > 0:
                print(f"[Cleanup] Total: {trimmed_count} entries trimmed beyond the per-symbol limit")

    def cleanup_old_symbols(self, max_age_days: int = 7):
        cutoff_time = time.time() - max_age_days * 24 * 3600
        removed_symbols = []
        with self.lock:
            for symbol, added_time in list(self.symbol_added_time.items()):
                if added_time < cutoff_time:
                    removed_symbols.append(symbol)
        for symbol in removed_symbols:
            self.remove_symbol(symbol, reason=f"(older than {max_age_days} days)")
        if removed_symbols:
            print(f"[Cleanup] Removed {len(removed_symbols)} old symbols: {removed_symbols}")
        return removed_symbols

    def save_state_to_file_if_enabled(self):
        if not self.save_state:
            return
        self.cleanup_old_prices()
        self.cleanup_old_symbols(max_age_days=PRICE_HISTORY_RETENTION_DAYS)
        try:
            with self.lock:
                atomic_write_json(self.filename, {
                    "items": self.cache,
                    "fetchtime": self.fetchtime_time_per_symbol,
                    "metadata": {
                        "last_cleanup": time.time(),
                        "retention_days": PRICE_HISTORY_RETENTION_DAYS,
                        "symbols_count": len(self.cache),
                        "total_entries": sum(len(v) for v in self.cache.values())
                    },
                    "symbol_metadata": {
                        "added_time": self.symbol_added_time,
                        "active_symbols": list(self.active_symbols)
                    }
                }, indent=1)
                print(f"[{self.cls_name}][info] Save cache to file {self.filename}")
        except Exception as e:
            print(f"[{self.cls_name}][Error] Saving cache file {self.filename}: {e}")


# ============================================
# Main functions
# ============================================

PRICE_MULTI_SYNC_INTERVAL_SEC = 5 * 60

def register_enhanced_cachePriceAll(cmc_api_key: Optional[str] = None):
    if not hasattr(CacheFactory, '_CONFIG'):
        CacheFactory._CONFIG = {}
    CacheFactory._CONFIG["PriceMulti"] = {
        "class": CacheAllPriceFetcherManager,
        "filename": "cache_prices_multi.json",
        "sync_ts": lambda: PRICE_MULTI_SYNC_INTERVAL_SEC,
        "cmc_api_key": cmc_api_key
    }
    print("[Pricefetcher] Manager registered in CacheFactory as 'PriceMulti'")


def is_valid_symbol_for_monitoring(symbol: str) -> bool:
    if not symbol:
        return False
    if len(symbol) < 1 or len(symbol) > 10:
        return False
    if not symbol.isalnum():
        return False
    if symbol.isdigit():
        return False
    if len(symbol) > 8 and symbol.startswith('0'):
        return False
    return True


def get_base_symbol(symbol: str) -> str:
    return resolve_assets(symbol)[0]


def create_cachePriceAll(cmc_api_key: Optional[str] = None, symbols=None, max_symbols=None):
    # Use symbols/max_symbols from market_alerts.conf when supplied; otherwise retain legacy defaults.
    all_symbols = []
    seen_bases = set()
    
    if hasattr(sym, 'symbols'):
        for s in sym.symbols:
            if is_valid_symbol_for_monitoring(s):
                seen_bases.add(get_base_symbol(s))
                all_symbols.append(s)
    
    for sym_default in (symbols or DEFAULT_SYMBOLS):
        base_symbol = get_base_symbol(sym_default)
        if (
            base_symbol not in seen_bases
            and sym_default not in all_symbols
            and is_valid_symbol_for_monitoring(sym_default)
        ):
            seen_bases.add(base_symbol)
            all_symbols.append(sym_default)
    
    all_symbols = list(dict.fromkeys(all_symbols))
    all_symbols = [s for s in all_symbols if is_valid_symbol_for_monitoring(s)]
    all_symbols = all_symbols[:(max_symbols or MAX_MONITORED_SYMBOLS)]

    print(f"[pricefetcher] Valid symbols to monitor: {len(all_symbols)}")
    print(f"[pricefetcher] List: {all_symbols}")

    register_enhanced_cachePriceAll(cmc_api_key)
    cachePriceAll = CacheFactory.get("PriceMulti", symbols=all_symbols)
    thread = cachePriceAll.periodic_sync(sync_ts=PRICE_MULTI_SYNC_INTERVAL_SEC, save_state=True)

    return cachePriceAll


if __name__ == "__main__":
    CMC_API_KEY = os.environ.get('CMC_API_KEY', None)
    print("=" * 60)
    print("🚀 Starting crypto price monitor (multi-platform)")
    print(f"💰 Base quote currency: {QUOTE_CURRENCY}")
    print("=" * 60)

    monitor = create_cachePriceAll(cmc_api_key=CMC_API_KEY)
    print("\n⏳ Waiting for the first price collection...")
    time.sleep(35)

    print("\n📊 Current prices:")
    for symbol in monitor.original_symbols:
        price = monitor.get_latest_price(symbol)
        if price:
            print(f"  {symbol}: ${price:.4f}")
        else:
            print(f"  {symbol}: waiting...")

    print("\n✅ Monitor active. Running in background (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
