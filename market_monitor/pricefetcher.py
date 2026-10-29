# pricefetcher.py
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import time
import requests
from state_io import atomic_write_json, read_json_dict, read_recovering_previous, save_json_state
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import quote
from abc import ABC, abstractmethod

import symbols as sym
from binance_api import bapi as api
from botcore import exact_pos_int, required_env, safe_pos_float, safe_pos_int
from cachepaths import cache_path, normalize_venue_name

# Import the base classes from cacheManager
from cacheManager import CacheFactory, CacheManagerInterface
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

    def _supports_in_set(self, symbol: str, symbols_set: Set[str]) -> bool:
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        sym_up = symbol.strip().upper()
        self.refresh_symbols()
        return sym_up in symbols_set or get_base_symbol(sym_up) in symbols_set


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
            if not isinstance(data, dict):
                raise ValueError("Binance exchangeInfo response must be a dict")
            raw_symbols = data.get("symbols")
            if not isinstance(raw_symbols, list):
                raise ValueError("Binance exchangeInfo symbols must be a list")

            usdc_pairs = set()
            supported_symbols = set()
            symbol_mapping = {}

            for symbol_info in raw_symbols:
                if not isinstance(symbol_info, dict):
                    continue
                raw_sym = symbol_info.get("symbol")
                raw_base = symbol_info.get("baseAsset")
                raw_quote = symbol_info.get("quoteAsset")
                raw_status = symbol_info.get("status")
                symbol = raw_sym.strip().upper() if isinstance(raw_sym, str) else ""
                base_asset = raw_base.strip().upper() if isinstance(raw_base, str) else ""
                quote_asset = raw_quote.strip().upper() if isinstance(raw_quote, str) else ""
                status = raw_status.strip().upper() if isinstance(raw_status, str) else ""

                if not symbol or status != "TRADING":
                    continue

                supported_symbols.add(symbol)

                if quote_asset == "USDC":
                    usdc_pairs.add(symbol)
                    if base_asset and base_asset not in symbol_mapping:
                        symbol_mapping[base_asset] = symbol
                    symbol_mapping[symbol] = symbol
            
            if usdc_pairs:
                self._usdc_pairs = usdc_pairs
                self._supported_symbols = supported_symbols
                self._symbol_mapping = symbol_mapping
                print(f"[BinancePlatform] USDC: {len(self._usdc_pairs)} pairs")
            self._last_refresh = time.time()

        except Exception as e:
            print(f"[BinancePlatform] Warning loading symbols: {e}")
            if not self._supported_symbols:
                self._fallback_symbols()
            self._last_refresh = time.time()
    
    def _fallback_symbols(self):
        self._usdc_pairs = {
            "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "ADAUSDC",
            "DOGEUSDC", "XRPUSDC", "TAOUSDC", "ARBUSDC"
        }
        self._supported_symbols = set(self._usdc_pairs)
        self._symbol_mapping = {
            "BTC": "BTCUSDC", "ETH": "ETHUSDC", "BNB": "BNBUSDC",
            "SOL": "SOLUSDC", "ADA": "ADAUSDC", "DOGE": "DOGEUSDC",
            "XRP": "XRPUSDC", "TAO": "TAOUSDC", "ARB": "ARBUSDC",
            "BTCUSDC": "BTCUSDC", "ETHUSDC": "ETHUSDC", "BNBUSDC": "BNBUSDC",
            "SOLUSDC": "SOLUSDC", "ADAUSDC": "ADAUSDC", "DOGEUSDC": "DOGEUSDC",
            "XRPUSDC": "XRPUSDC", "TAOUSDC": "TAOUSDC", "ARBUSDC": "ARBUSDC"
        }
    
    def refresh_symbols(self):
        if time.time() - self._last_refresh > self._refresh_interval:
            self._load_symbols()
    
    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return self._supported_symbols.copy()
    
    def _resolve_trading_pair(self, symbol: str) -> Optional[str]:
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        sym_up = symbol.strip().upper()
        if sym_up in self._symbol_mapping:
            return self._symbol_mapping[sym_up]
        pair = f"{sym_up}USDC"
        if pair in self._supported_symbols or pair in self._usdc_pairs:
            return pair
        if sym_up in self._supported_symbols or sym_up in self._usdc_pairs:
            return sym_up
        base = get_base_symbol(sym_up)
        if base in self._symbol_mapping:
            return self._symbol_mapping[base]
        base_pair = f"{base}USDC"
        if base_pair in self._supported_symbols or base_pair in self._usdc_pairs:
            return base_pair
        return None

    def supports_symbol(self, symbol: str) -> bool:
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        self.refresh_symbols()
        return self._resolve_trading_pair(symbol) is not None
    
    def get_price(self, symbol: str) -> Optional[float]:
        try:
            trading_pair = self._resolve_trading_pair(symbol)
            if not trading_pair:
                if isinstance(symbol, str) and symbol.strip():
                    print(f"[BinancePlatform] No trading pair found for {symbol.strip().upper()}")
                return None

            price = self.api_client.get_current_price(symbol=trading_pair)
            price_val = safe_pos_float(price, default=None)
            if price_val is None:
                if price is None or isinstance(price, bool):
                    print(f"[BinancePlatform] Could not get price for {trading_pair}")
                return None
            print(f"[BinancePlatform] {symbol.strip().upper()} -> {trading_pair} = ${price_val}")
            return price_val

        except Exception as e:
            print(f"[BinancePlatform] Warning for {symbol}: {e}")
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
            mids = response.json()
            if not isinstance(mids, dict) or not mids:
                raise ValueError("Hyperliquid allMids response must be a non-empty dict")
            self._all_mids = mids
            self._supported_symbols = set(self._all_mids.keys())
            print(f"[HyperliquidPlatform] Loaded {len(self._supported_symbols)} symbols")
            self._last_refresh = time.time()
        except Exception as e:
            print(f"[HyperliquidPlatform] Warning loading symbols: {e}")
            if not self._supported_symbols:
                self._supported_symbols = {"HYPE", "PURR", "BTC", "ETH", "SOL", "USDC", "TAO", "ARB", "DOGE"}
            self._last_refresh = time.time()
    
    def refresh_symbols(self):
        if time.time() - self._last_refresh > self._refresh_interval:
            self._load_symbols()
    
    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return self._supported_symbols.copy()
    
    def supports_symbol(self, symbol: str) -> bool:
        return self._supports_in_set(symbol, self._supported_symbols)
    
    def get_price(self, symbol: str) -> Optional[float]:
        try:
            if not isinstance(symbol, str) or not symbol.strip():
                return None
            symbol = symbol.strip().upper()
            self.refresh_symbols()
            lookup_key = symbol if symbol in self._all_mids else get_base_symbol(symbol)
            if lookup_key not in self._all_mids:
                print(f"[HyperliquidPlatform] Symbol {symbol} not found")
                return None
            return safe_pos_float(self._all_mids[lookup_key], default=None)
        except Exception as e:
            print(f"[HyperliquidPlatform] Warning for {symbol}: {e}")
            return None


_CMC_KEY_UNSET = object()


class CoinMarketCapPricePlatform(PricePlatformInterface):
    def __init__(self, api_key: Optional[str] = _CMC_KEY_UNSET):  # type: ignore[assignment]
        # Respect explicit api_key=None (offline/disabled) without falling back to os.environ.
        if api_key is _CMC_KEY_UNSET:
            raw_key = os.environ.get('CMC_API_KEY')
        else:
            raw_key = api_key
        self.api_key = raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None
        self._base_url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
        self._supported_symbols: Set[str] = set()
        self._all_listings: Dict[str, Dict] = {}
        self._last_refresh = 0
        self._refresh_interval = 86400
        self._cache_file = cache_path("cmc_listings_cache.json")
        
        if self.api_key:
            self._load_symbols()
        else:
            print("[CMCPlatform] No API key - platform disabled")

    @property
    def platform_name(self) -> str:
        return "CoinMarketCap"
    
    def _load_symbols_from_disk_cache(self, max_age_s: Optional[float] = None) -> bool:
        try:
            prev_file = f"{self._cache_file}.previous"
            if not os.path.exists(self._cache_file) and not os.path.exists(prev_file):
                return False
            stat_path = self._cache_file if os.path.exists(self._cache_file) else prev_file
            mtime = os.path.getmtime(stat_path)
            if max_age_s is not None and (time.time() - mtime) > max_age_s:
                return False

            def _read_cache(path: str):
                payload = read_json_dict(path)
                if not payload:
                    raise ValueError("CMC cache must be a non-empty dict")
                return payload

            cached = read_recovering_previous(self._cache_file, _read_cache)
            if not isinstance(cached, dict) or not cached:
                return False
            cleaned = {
                k.strip().upper(): v
                for k, v in cached.items()
                if isinstance(k, str) and k.strip() and isinstance(v, dict)
            }
            if not cleaned:
                return False
            self._all_listings = cleaned
            self._supported_symbols = set(cleaned.keys())
            self._last_refresh = mtime
            return True
        except Exception:
            return False

    def _load_symbols(self):
        if not self.api_key:
            return
        if self._load_symbols_from_disk_cache(max_age_s=self._refresh_interval):
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
            if not isinstance(data, dict) or not isinstance(data.get('data'), list):
                raise ValueError("CoinMarketCap listings response must contain a data list")
            self._supported_symbols.clear()
            for crypto in data['data']:
                if not isinstance(crypto, dict):
                    continue
                raw_symbol = crypto.get('symbol')
                symbol = raw_symbol.strip().upper() if isinstance(raw_symbol, str) else ""
                if symbol:
                    self._supported_symbols.add(symbol)
                    self._all_listings[symbol] = {
                        'id': crypto.get('id'),
                        'name': crypto.get('name'),
                        'slug': crypto.get('slug')
                    }
            print(f"[CMCPlatform] Loaded {len(self._supported_symbols)} symbols")
            self._last_refresh = time.time()
            try:
                os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
                save_json_state(self._cache_file, self._all_listings)
            except Exception:
                pass
        except Exception as e:
            if not self._supported_symbols:
                self._load_symbols_from_disk_cache(max_age_s=None)
            print(f"[CMCPlatform] Warning loading symbols: {e}")
            self._last_refresh = time.time()  # Do not retest immediately; honour _refresh_interval.

    def refresh_symbols(self):
        if self.api_key and time.time() - self._last_refresh > self._refresh_interval:
            self._load_symbols()
    
    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return self._supported_symbols.copy()
    
    def supports_symbol(self, symbol: str) -> bool:
        if not self.api_key:
            return False
        return self._supports_in_set(symbol, self._supported_symbols)

    def _extract_usd_price(self, symbol: str, data: Dict) -> Optional[float]:
        if not isinstance(data, dict):
            return None
        data_map = data.get('data')
        coin_data = data_map.get(symbol) if isinstance(data_map, dict) else None
        if isinstance(coin_data, list):
            coin_data = coin_data[0] if coin_data else None
        if isinstance(coin_data, dict):
            quote = coin_data.get('quote')
            usd = quote.get('USD') if isinstance(quote, dict) else None
            price = usd.get('price') if isinstance(usd, dict) else None
            return safe_pos_float(price, default=None)
        return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the price from the internal cache (updated from listings/latest).
        This is the fastest and safest way to get prices for new coins.
        """
        if not self.api_key or not isinstance(symbol, str) or not symbol.strip():
            if not self.api_key:
                print(f"[CMCPlatform] Missing API key")
            return None
        symbol = symbol.strip().upper()

        # Refresh if enough time has passed
        self.refresh_symbols()

        # Search the symbol in the internal cache (_all_listings only contains metadata,
        # but we also need the price. We will use a separate dictionary for prices.
        # A simpler solution is to use CoinMarketCapSource directly. If there is no dedicated cache,
        # we will make a request to listings/latest for the specific symbol.
        try:
            lookup_symbol = symbol if symbol in self._supported_symbols else get_base_symbol(symbol)
            headers = {'X-CMC_PRO_API_KEY': self.api_key, 'Accept': 'application/json'}
            params = {'symbol': lookup_symbol, 'convert': 'USD'}
            response = requests.get(
                self._base_url,
                headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            price = self._extract_usd_price(lookup_symbol, data)
            if price is not None:
                print(f"[CMCPlatform] {symbol} = ${price}")
                return price
            print(f"[CMCPlatform] {symbol} not found")
            return None
        except Exception as e:
            print(f"[CMCPlatform] Warning for {symbol}: {e}")
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
            configured = required_env("STOCK_TICKERS")
            src = configured.split(",")
        elif isinstance(tickers, str):
            src = tickers.split(",")
        elif isinstance(tickers, (list, tuple, set, frozenset)):
            src = tickers
        else:
            src = []
        self._symbols: Set[str] = {
            t.strip().upper()
            for t in src
            if isinstance(t, str) and t.strip()
        }

    @property
    def platform_name(self) -> str:
        return "Yahoo"

    def get_available_symbols(self) -> Set[str]:
        return self._symbols.copy()

    def supports_symbol(self, symbol: str) -> bool:
        # Accept only configured tickers; do not divert crypto or make a network call here.
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            return False
        return symbol.strip().upper() in self._symbols

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            if not isinstance(symbol, str) or not symbol.strip():
                return None
            sym_y = symbol.strip().upper().split("_")[0]
            url = self._YAHOO.format(sym=quote(sym_y, safe="")) + "?range=1d&interval=5m"
            resp = requests.get(url, headers=self._UA, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                return None
            chart = payload.get("chart")
            if not isinstance(chart, dict):
                return None
            result = (chart.get("result") or [None])[0]
            if not isinstance(result, dict):
                return None
            indicators = result.get("indicators")
            quote_list = indicators.get("quote") if isinstance(indicators, dict) else None
            q = (quote_list or [{}])[0] if isinstance(quote_list, list) else {}
            if not isinstance(q, dict):
                q = {}
            closes = [
                cf for c in (q.get("close") or [])
                if (cf := safe_pos_float(c, default=None)) is not None
            ]
            if closes:
                return closes[-1]                      # freshest intraday bar
            meta = result.get("meta")
            price = meta.get("regularMarketPrice") if isinstance(meta, dict) else None  # metadata fallback
            return safe_pos_float(price, default=None)
        except Exception as e:
            print(f"[YahooPlatform] Warning for {symbol}: {e}")
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
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Symbol '{symbol}' is not supported by any platform")
        sym_norm = symbol.strip().upper()
        cached_platform = None
        supported = False
        platform_name = self._symbol_platform_cache.get(sym_norm) or self._symbol_platform_cache.get(symbol)
        if platform_name:
            for platform in self._platforms:
                if platform.platform_name == platform_name:
                    cached_platform = platform
                    supported = True
                    price = platform.get_price(symbol)
                    if price is not None:
                        self._symbol_platform_cache[sym_norm] = platform.platform_name
                        return {
                            "symbol": symbol, "price": price,
                            "platform": platform.platform_name, "timestamp": int(time.time())
                        }
                    self._symbol_platform_cache.pop(sym_norm, None)
                    self._symbol_platform_cache.pop(symbol, None)
                    break
        for platform in self._platforms:
            if platform is cached_platform:
                continue
            if platform.supports_symbol(symbol):
                supported = True
                price = platform.get_price(symbol)
                if price is not None:
                    self._symbol_platform_cache[sym_norm] = platform.platform_name
                    return {
                        "symbol": symbol, "price": price,
                        "platform": platform.platform_name, "timestamp": int(time.time())
                    }
        if not supported:
            raise ValueError(f"Symbol '{symbol}' is not supported by any platform")
        raise RuntimeError(f"Unable to fetch price for '{symbol}' from any supported platform")

    def check_symbol_support(self, symbol: str) -> Dict:
        support = {}
        for platform in self._platforms:
            support[platform.platform_name] = platform.supports_symbol(symbol)
        return support


# ============================================
# CacheAllPriceFetcherManager
# ============================================

class CacheAllPriceFetcherManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api, cmc_api_key: Optional[str] = None,
                 read_only=False, max_symbols: Optional[int] = None):
        self.price_factory = PricePlatformFactory(cmc_api_key=cmc_api_key)
        self.max_monitored_symbols = _safe_max_symbols(max_symbols, MAX_MONITORED_SYMBOLS)
        initial_symbols = []
        seen_bases = set()
        for symbol in (symbols or ()):
            #print(f"{symbol}")
            base_symbol = get_base_symbol(symbol)
            if base_symbol not in seen_bases and is_valid_symbol_for_monitoring(symbol):
                seen_bases.add(base_symbol)
                initial_symbols.append(symbol)
            if len(initial_symbols) >= self.max_monitored_symbols:
                break
        self.original_symbols = list(initial_symbols)
        self.active_symbols = set(initial_symbols)
        self.symbol_added_time: Dict[str, float] = {}
        self.symbol_preferred_source: Dict[str, str] = {}  # Required source preference state.
        super().__init__(sync_ts, initial_symbols, filename, append_mode=True,
                         api_client=api_client, read_only=read_only)
        self._load_symbol_metadata()
        self._log_symbol_support()
    
    def _load_symbol_metadata(self):
        data = read_json_dict(self.filename, default=None, recover_previous=True)
        sym_meta = data.get("symbol_metadata") if isinstance(data, dict) else None
        if not isinstance(sym_meta, dict):
            return
        try:
            raw_added = sym_meta.get("added_time")
            self.symbol_added_time = (
                {
                    str(k): vf
                    for k, v in raw_added.items()
                    if isinstance(k, str)
                    and isinstance(v, (int, float))
                    and (vf := safe_pos_float(v, default=None)) is not None
                }
                if isinstance(raw_added, dict)
                else {}
            )
            preferred = sym_meta.get("preferred_source")
            if isinstance(preferred, dict):
                self.symbol_preferred_source = {
                    str(k): str(v)
                    for k, v in preferred.items()
                    if isinstance(k, str) and isinstance(v, str) and v.strip()
                }
            seen_bases = {get_base_symbol(symbol) for symbol in self.active_symbols}
            raw_active = sym_meta.get("active_symbols")
            active_symbols = (
                [s for s in raw_active if isinstance(s, str) and s.strip()]
                if isinstance(raw_active, list)
                else []
            )
            raw_items = data.get("items")
            items_map = raw_items if isinstance(raw_items, dict) else {}
            recovered_symbols = [
                symbol for symbol, added_time in self.symbol_added_time.items()
                if (
                    symbol in items_map
                    and (at_val := safe_pos_float(added_time, default=None)) is not None
                    and time.time() - at_val <= PRICE_HISTORY_RETENTION_DAYS * 24 * 3600
                )
            ]
            for symbol in list(dict.fromkeys(active_symbols + recovered_symbols)):
                if len(self.active_symbols) >= self.max_monitored_symbols:
                    print(f"[Pricefetcher] Limit reached while loading: maximum {self.max_monitored_symbols} monitored coins, the rest will be ignored")
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

    def _reload_from_disk(self):
        super()._reload_from_disk()
        with self.lock:
            self._load_symbol_metadata()
    
    def _log_symbol_support(self):
        print(f"[Pricefetcher] Checking symbol support:")
        for symbol in self.original_symbols:
            support = self.price_factory.check_symbol_support(symbol)
            supported_platforms = [p for p, s in support.items() if s]
            if supported_platforms:
                print(f"  ✅ {symbol} -> {', '.join(supported_platforms)}")
            else:
                print(f"  ❌ {symbol} -> NO PLATFORM SUPPORT!")

    def rebuild_fetchtimes(self):
        if not self.cache:
            return {}
        last_times = {}
        for symbol in self.symbols:
            entries = self.cache.get(symbol, [])
            if isinstance(entries, list) and entries:
                max_ts = max((self._entry_timestamp_ms(entry) for entry in entries), default=0)
                if max_ts > 0:
                    last_times[symbol] = max_ts
        return last_times
    
    def get_remote_items(self, symbol, startTime):
        try:
            # Ensure the attribute exists
            if not hasattr(self, 'symbol_preferred_source'):
                self.symbol_preferred_source = {}

            preferred_source = self.symbol_preferred_source.get(symbol)
            if preferred_source and isinstance(preferred_source, str) and preferred_source.strip():
                pref_norm = normalize_venue_name(preferred_source) or preferred_source.strip().lower()
                for platform in self.price_factory._platforms:
                    plat_raw = str(platform.platform_name).strip().lower()
                    plat_norm = normalize_venue_name(plat_raw) or plat_raw
                    if plat_norm == pref_norm or plat_raw == preferred_source.strip().lower():
                        price_val = safe_pos_float(platform.get_price(symbol), default=None)
                        if price_val is not None:
                            timestamp_ms = int(time.time() * 1000)
                            print(f"[Pricefetcher][{symbol}] ${price_val:.4f} (source: {preferred_source} - preferred)")
                            return [[timestamp_ms, price_val]]
                        print(f"[Pricefetcher][{symbol}] Preferred source {preferred_source} failed; trying fallback providers")
                        break
            result = self.price_factory.get_price(symbol)
            price_val = safe_pos_float(result.get("price") if isinstance(result, dict) else None, default=None)
            if price_val is None:
                return []
            platform_used = result["platform"]
            timestamp_ms = int(time.time() * 1000)
            print(f"[Pricefetcher][{symbol}] ${price_val:.4f} (source: {platform_used})")
            return [[timestamp_ms, price_val]]
        except Exception as e:
            print(f"[Pricefetcher][Warn] {symbol}: {e}")
            return []
    
    def add_symbol(self, symbol: str, preferred_source: Optional[str] = None):
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        symbol = symbol.strip()
        with self.lock:
            base_symbol = get_base_symbol(symbol)
            if symbol in self.active_symbols:
                print(f"[Pricefetcher] {symbol} is already in the watchlist")
                return False
            if any(get_base_symbol(active_symbol) == base_symbol for active_symbol in self.active_symbols):
                print(f"[Pricefetcher] {symbol} ignored: asset {base_symbol} is already being monitored")
                return False
            if len(self.active_symbols) >= self.max_monitored_symbols:
                print(f"[Pricefetcher] Limit reached: maximum {self.max_monitored_symbols} monitored coins")
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
            if symbol not in self.fetchtime_per_symbol:
                self.fetchtime_per_symbol[symbol] = self.fallback_time_default
            print(f"[Pricefetcher] ✅ Symbol added: {symbol}")
        # Fetch and store the new symbol's initial price outside this lock;
        # ``update_cache_per_symbol`` acquires its own lock.
        items = self.get_remote_items(
            symbol, self.fetchtime_per_symbol.get(symbol, self.fallback_time_default))
        if items:
            self.update_cache_per_symbol(symbol, items)
        return True
    
    def remove_symbol(self, symbol: str, reason: str = ""):
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        symbol = symbol.strip()
        with self.lock:
            was_active = symbol in self.active_symbols
            had_meta = (
                symbol in self.symbol_added_time
                or symbol in getattr(self, "symbol_preferred_source", {})
            )
            if not was_active and not had_meta:
                return False
            while symbol in self.symbols:
                self.symbols.remove(symbol)
            while symbol in self.original_symbols:
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
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        sym_clean = symbol.strip()
        with self.lock:
            entries = self.cache.get(symbol)
            if entries is None:
                entries = self.cache.get(sym_clean, [])
            if isinstance(entries, (list, tuple)) and entries:
                for entry in reversed(entries):
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        ts_val = safe_pos_float(entry[0], default=None)
                        px = safe_pos_float(entry[1], default=None)
                        if ts_val is not None and px is not None:
                            return px
        return None

    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        limit_i = exact_pos_int(limit, default=None)
        if limit_i is None or not isinstance(symbol, str) or not symbol.strip():
            return []
        sym_clean = symbol.strip()
        # Copy only references under the lock. Convert and format afterward so datetime
        # work and allocations do not block price synchronization.
        with self.lock:
            raw_entries = self.cache.get(symbol)
            if raw_entries is None:
                raw_entries = self.cache.get(sym_clean, ())
            entries = raw_entries[-limit_i:] if isinstance(raw_entries, (list, tuple)) else ()

        history: List[Dict] = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            ts_val = safe_pos_float(entry[0], default=None)
            px_val = safe_pos_float(entry[1], default=None)
            if ts_val is None or px_val is None:
                continue
            try:
                ts_sec = ts_val / 1000.0 if ts_val >= 10_000_000_000 else ts_val
                readable = datetime.fromtimestamp(ts_sec).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            history.append(
                {
                    "timestamp": int(ts_val) if ts_val.is_integer() else ts_val,
                    "timestamp_readable": readable,
                    "price": px_val,
                }
            )
        return history
        
    def cleanup_old_prices(self, retention_days: int = PRICE_HISTORY_RETENTION_DAYS):
        if not isinstance(retention_days, (int, float)) or safe_pos_float(retention_days, default=None) is None:
            return
        cutoff_timestamp = (time.time() - retention_days * 24 * 3600) * 1000
        with self.lock:
            removed_count = 0
            trimmed_count = 0
            for symbol in list(self.cache.keys()):
                raw_bucket = self.cache[symbol]
                if not isinstance(raw_bucket, list):
                    self.cache[symbol] = []
                    continue
                original_count = len(raw_bucket)
                self.cache[symbol] = [
                    entry
                    for entry in raw_bucket
                    if isinstance(entry, (list, tuple))
                    and len(entry) >= 2
                    and self._entry_timestamp_ms(entry) >= cutoff_timestamp
                ]
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
                    self.fetchtime_per_symbol.pop(symbol, None)
                    self._rest_sync_cursor_per_symbol.pop(symbol, None)
                    print(f"[Cleanup] {symbol}: fully removed")
            if removed_count > 0:
                print(f"[Cleanup] Total: {removed_count} old entries removed")
            if trimmed_count > 0:
                print(f"[Cleanup] Total: {trimmed_count} entries trimmed beyond the per-symbol limit")

    def cleanup_old_symbols(self, max_age_days: int = 7):
        if not isinstance(max_age_days, (int, float)) or safe_pos_float(max_age_days, default=None) is None:
            return []
        cutoff_time = time.time() - max_age_days * 24 * 3600
        removed_symbols = []
        with self.lock:
            for symbol, added_time in list(self.symbol_added_time.items()):
                at_val = safe_pos_float(added_time, default=None) if isinstance(added_time, (int, float)) else None
                if at_val is None or at_val < cutoff_time:
                    removed_symbols.append(symbol)
        for symbol in removed_symbols:
            self.remove_symbol(symbol, reason=f"(older than {max_age_days} days)")
        if removed_symbols:
            print(f"[Cleanup] Removed {len(removed_symbols)} old symbols: {removed_symbols}")
        return removed_symbols

    def save_state_to_file(self):
        self.cleanup_old_prices()
        self.cleanup_old_symbols(max_age_days=PRICE_HISTORY_RETENTION_DAYS)
        try:
            with self.lock:
                if self._persisted_max_ts() > self._mem_max_ts():
                    print(
                        f"[{self.cls_name}][resync] file newer than memory -> "
                        f"refusing to overwrite with stale data ({self.filename})"
                    )
                    return False
                counts = {s: len(v) for s, v in self.cache.items()}
                payload = {
                    "items": self.cache,
                    "fetchtime": self.fetchtime_per_symbol,
                    "rest_sync_cursor": self._rest_sync_cursor_per_symbol,
                    "metadata": {
                        "last_cleanup": time.time(),
                        "retention_days": PRICE_HISTORY_RETENTION_DAYS,
                        "symbols_count": len(self.cache),
                        "total_entries": sum(len(v) for v in self.cache.values())
                    },
                    "symbol_metadata": {
                        "added_time": self.symbol_added_time,
                        "active_symbols": list(self.active_symbols),
                        "preferred_source": self.symbol_preferred_source,
                    }
                }
                meta_payload = {
                    "counts": dict(counts),
                    "fetchtime": dict(self.fetchtime_per_symbol),
                    "rest_sync_cursor": dict(self._rest_sync_cursor_per_symbol),
                }
                if not self._commit_file_and_meta_locked(
                    lambda: atomic_write_json(self.filename, payload, indent=1),
                    meta_payload,
                ):
                    return False
                self._persisted_counts = counts
                print(f"[{self.cls_name}][info] Save cache to file {self.filename}")
                return True
        except Exception as e:
            print(f"[{self.cls_name}][Warn] Saving cache file {self.filename}: {e}")
            return False

    def save_state_to_file_if_enabled(self):
        if not self.save_state:
            return False
        return self.save_state_to_file()


CachePriceMulti = CacheAllPriceFetcherManager


# ============================================
# Main functions
# ============================================

PRICE_MULTI_SYNC_INTERVAL_SEC = 5 * 60

def _safe_max_symbols(val, default: int = MAX_MONITORED_SYMBOLS) -> int:
    return safe_pos_int(val, default=default)


def register_enhanced_cachePriceAll(cmc_api_key: Optional[str] = None, max_symbols: Optional[int] = None):
    if not hasattr(CacheFactory, '_CONFIG'):
        CacheFactory._CONFIG = {}
    cfg = {
        "class": CacheAllPriceFetcherManager,
        "filename": "cache_watcher_all_price.json",
        "sync_ts": lambda: PRICE_MULTI_SYNC_INTERVAL_SEC,
        "cmc_api_key": cmc_api_key,
    }
    if max_symbols is not None and not isinstance(max_symbols, bool):
        cfg["max_symbols"] = _safe_max_symbols(max_symbols, MAX_MONITORED_SYMBOLS)
    CacheFactory._CONFIG["PriceMulti"] = cfg
    print("[Pricefetcher] Manager registered in CacheFactory as 'PriceMulti'")


def is_valid_symbol_for_monitoring(symbol: str) -> bool:
    if not isinstance(symbol, str) or not (1 <= len(symbol) <= 10):
        return False
    if not symbol.isalnum() or symbol.isdigit():
        return False
    return not (len(symbol) > 8 and symbol.startswith("0"))


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
        if not is_valid_symbol_for_monitoring(sym_default):
            continue
        base_symbol = get_base_symbol(sym_default)
        if (
            base_symbol not in seen_bases
            and sym_default not in all_symbols
        ):
            seen_bases.add(base_symbol)
            all_symbols.append(sym_default)
    
    eff_max = _safe_max_symbols(max_symbols, MAX_MONITORED_SYMBOLS)
    all_symbols = list(dict.fromkeys(all_symbols))
    all_symbols = [s for s in all_symbols if is_valid_symbol_for_monitoring(s)]
    all_symbols = all_symbols[:eff_max]

    print(f"[pricefetcher] Valid symbols to monitor: {len(all_symbols)}")
    print(f"[pricefetcher] List: {all_symbols}")

    register_enhanced_cachePriceAll(cmc_api_key, max_symbols=eff_max)
    cachePriceAll = CacheFactory.get("PriceMulti", symbols=all_symbols)
    cachePriceAll.max_monitored_symbols = eff_max
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
