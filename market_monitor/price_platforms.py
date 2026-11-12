"""Multi-venue price platform transport adapters and declarative PricePlatformFactory.

Follows a 3-tier polymorphic hierarchy:
- Level 1: ``PricePlatformInterface`` (abstract base contract + periodic symbol refresh)
- Level 2: ``CryptoPricePlatformBase`` / ``EquityPricePlatformBase`` (asset-class domain bases)
- Level 3: ``BinancePricePlatform``, ``HyperliquidPricePlatform``, ``CoinMarketCapPricePlatform``,
  and ``StockYahooPricePlatform`` registered declaratively via ``@PricePlatformFactory.register_platform``.
"""

from __future__ import annotations

import math
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from urllib.parse import quote

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import instrument_registry as sym
from botcore import (
    fetch_requests_json_get,
    fetch_requests_json_post,
    optional_env,
    parse_symbol_list,
    required_env,
    resolve_patched_attr,
    safe_pos_float,
)
from cachepaths import (
    CMC_LISTINGS_CACHE_FILENAME,
    cache_path,
    canonical_base_asset,
    normalize_venue_name,
    strip_equity_suffix,
)
from instrument_registry import casefold_get
import providers.market_api as _market_api
from providers.ohlc_utils import (
    extract_trading_quote_pairs,
    extract_yahoo_chart_latest_price,
)
from providers.quantity import resolve_base_asset
from state_io import read_json_dict, save_json_state


get_base_symbol = resolve_base_asset


def is_valid_symbol_for_monitoring(symbol: str) -> bool:
    if not isinstance(symbol, str) or not (1 <= len(symbol) <= 10):
        return False
    if not symbol.isalnum() or symbol.isdigit():
        return False
    return not (len(symbol) > 8 and symbol.startswith("0"))


def _active_requests_mod() -> object:
    return resolve_patched_attr(
        "requests",
        requests,
        local_val=requests,
        modules=("market_monitor.pricefetcher", "market_monitor.price_platforms"),
    )


class PricePlatformInterface(ABC):
    """Level 1 abstract contract for venue/aggregator price discovery and spot lookup."""

    PLATFORM_KEY: str = ""
    IS_EQUITY_PLATFORM: bool = False
    REQUIRES_CMC_KEY: bool = False
    PRIORITY_ORDER: int = 100

    _supported_symbols: Set[str]
    _last_refresh: float = 0.0
    _refresh_interval: float = 0.0

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def platform_name(self) -> str:
        pass

    def _init_platform_refresh(self, refresh_interval: float, *, load_now: bool = True) -> None:
        self._supported_symbols = set()
        self._last_refresh = 0
        self._refresh_interval = refresh_interval
        if load_now:
            self._load_symbols()

    def _load_symbols(self):
        pass

    def refresh_symbols(self):
        if (
            self._refresh_interval > 0
            and bool(getattr(self, "api_key", True))
            and time.time() - self._last_refresh > self._refresh_interval
        ):
            self._load_symbols()

    def get_available_symbols(self) -> Set[str]:
        self.refresh_symbols()
        return getattr(self, "_supported_symbols", set()).copy()

    def supports_symbol(self, symbol: str) -> bool:
        if (
            self._refresh_interval <= 0
            or not bool(getattr(self, "api_key", True))
            or not isinstance(symbol, str)
            or not symbol.strip()
        ):
            return False
        self.refresh_symbols()
        symbols_set = getattr(self, "_supported_symbols", set())
        sym_clean = symbol.strip()
        sym_up = sym_clean.upper()
        base_up = get_base_symbol(sym_up)
        if sym_clean in symbols_set or sym_up in symbols_set or base_up in symbols_set:
            return True
        upper_set = {s.upper() for s in symbols_set if isinstance(s, str)}
        return sym_up in upper_set or base_up in upper_set

    @staticmethod
    def _log_platform_price_warning(tag: str, symbol: str, exc: Exception) -> None:
        print(f"[{tag}] Warning for {symbol}: {exc}")
        return None

    @staticmethod
    def _fetch_json_get(url: str, *, timeout: float = 10, **kwargs):
        return fetch_requests_json_get(
            url, timeout=timeout, requests_mod=_active_requests_mod(), **kwargs
        )

    @staticmethod
    def _fetch_json_post(url: str, payload: object, *, headers: Optional[Dict[str, str]] = None, timeout: float = 10):
        return fetch_requests_json_post(
            url, payload, headers=headers, timeout=timeout, requests_mod=_active_requests_mod()
        )


class CryptoPricePlatformBase(PricePlatformInterface):
    """Level 2 abstract domain base for 24/7 crypto spot and aggregator platforms."""

    IS_EQUITY_PLATFORM: bool = False


class EquityPricePlatformBase(PricePlatformInterface):
    """Level 2 abstract domain base for US RTH equity price platforms."""

    IS_EQUITY_PLATFORM: bool = True


class PricePlatformFactory:
    """Declarative registry and ordered fallback router across price platforms."""

    _PLATFORM_REGISTRY: Dict[str, type[PricePlatformInterface]] = {}

    @classmethod
    def register_platform(cls, platform_cls: type[PricePlatformInterface]) -> type[PricePlatformInterface]:
        key = str(getattr(platform_cls, "PLATFORM_KEY", "") or getattr(platform_cls, "platform_name", "")).strip().lower()
        if not key:
            raise ValueError(f"Platform class {platform_cls.__name__} must declare PLATFORM_KEY")
        cls._PLATFORM_REGISTRY[key] = platform_cls
        return platform_cls

    def __init__(self, cmc_api_key: Optional[str] = None):
        ordered_classes = sorted(
            self._PLATFORM_REGISTRY.values(),
            key=lambda c: getattr(c, "PRIORITY_ORDER", 100),
        )
        self._platforms: List[PricePlatformInterface] = []
        for plat_cls in ordered_classes:
            eff_cls = resolve_patched_attr(
                plat_cls.__name__,
                plat_cls,
                modules=("market_monitor.pricefetcher", "market_monitor.price_platforms"),
                require_callable=True,
            )
            if getattr(plat_cls, "REQUIRES_CMC_KEY", False):
                if cmc_api_key:
                    self._platforms.append(eff_cls(cmc_api_key))
            else:
                self._platforms.append(eff_cls())
        self._symbol_platform_cache: Dict[str, str] = {}
        self._discover_all_symbols()

    def _discover_all_symbols(self):
        print("[PriceFactory] Discovering available symbols...")
        all_symbols = {}
        for platform in self._platforms:
            try:
                symbols = platform.get_available_symbols()
                all_symbols[platform.platform_name] = {
                    "count": len(symbols),
                    "sample": list(symbols)[:10],
                }
                print(f"[PriceFactory]   {platform.platform_name}: {len(symbols)} symbols")
            except Exception as e:
                print(f"[PriceFactory]   {platform.platform_name}: error - {e}")
        self._capabilities = all_symbols

    def resolve_preferred_platform(self, preferred_source: object) -> Optional[PricePlatformInterface]:
        if not isinstance(preferred_source, str) or not preferred_source.strip():
            return None
        pref_raw = preferred_source.strip().lower()
        pref_norm = normalize_venue_name(pref_raw) or pref_raw
        for platform in getattr(self, "_platforms", ()) or ():
            plat_raw = str(platform.platform_name).strip().lower()
            plat_norm = normalize_venue_name(plat_raw) or plat_raw
            if plat_norm == pref_norm or plat_raw == pref_raw:
                return platform
        return None

    def get_price(self, symbol: str) -> Dict:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Symbol '{symbol}' is not supported by any platform")
        sym_norm = symbol.strip().upper()
        cached_platform = None
        supported = False
        platform_name = casefold_get(self._symbol_platform_cache, sym_norm)

        def _try_platform(platform) -> Optional[Dict]:
            nonlocal supported
            supported = True
            price = platform.get_price(symbol)
            if price is not None:
                self._symbol_platform_cache[sym_norm] = platform.platform_name
                return {
                    "symbol": symbol,
                    "price": price,
                    "platform": platform.platform_name,
                    "timestamp": math.trunc(time.time()),
                }
            return None

        if platform_name:
            for platform in self._platforms:
                if platform.platform_name == platform_name:
                    cached_platform = platform
                    if (res := _try_platform(platform)) is not None:
                        return res
                    self._symbol_platform_cache.pop(sym_norm, None)
                    break
        for platform in self._platforms:
            if platform is not cached_platform and platform.supports_symbol(symbol):
                if (res := _try_platform(platform)) is not None:
                    return res
        if not supported:
            raise ValueError(f"Symbol '{symbol}' is not supported by any platform")
        raise RuntimeError(f"Unable to fetch price for '{symbol}' from any supported platform")

    def check_symbol_support(self, symbol: str) -> Dict:
        return {platform.platform_name: platform.supports_symbol(symbol) for platform in self._platforms}


@PricePlatformFactory.register_platform
class BinancePricePlatform(CryptoPricePlatformBase):
    PLATFORM_KEY = "binance"
    PRIORITY_ORDER = 10
    platform_name = "Binance"

    def __init__(self, api_client=None):
        self.api_client = api_client or _market_api.api
        self._usdc_pairs: Set[str] = set()
        self._symbol_mapping: Dict[str, str] = {}
        self._init_platform_refresh(3600)

    def _load_symbols(self):
        try:
            data = self._fetch_json_get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
            supported_symbols, usdc_pairs, pair_to_base = extract_trading_quote_pairs(
                data, quote="USDC", strict=True
            )
            symbol_mapping = {}
            for pair in sorted(usdc_pairs):
                base_asset = pair_to_base.get(pair, "")
                if base_asset and base_asset not in symbol_mapping:
                    symbol_mapping[base_asset] = pair
                symbol_mapping[pair] = pair

            if usdc_pairs:
                self._usdc_pairs = usdc_pairs
                self._supported_symbols = supported_symbols
                self._symbol_mapping = symbol_mapping
                print(f"[BinancePlatform] USDC: {len(self._usdc_pairs)} pairs")
        except Exception as e:
            print(f"[BinancePlatform] Warning loading symbols: {e}")
            if not self._supported_symbols:
                self._usdc_pairs = {
                    "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "ADAUSDC",
                    "DOGEUSDC", "XRPUSDC", "TAOUSDC", "ARBUSDC",
                }
                self._supported_symbols = set(self._usdc_pairs)
                self._symbol_mapping = {
                    "BTC": "BTCUSDC", "ETH": "ETHUSDC", "BNB": "BNBUSDC",
                    "SOL": "SOLUSDC", "ADA": "ADAUSDC", "DOGE": "DOGEUSDC",
                    "XRP": "XRPUSDC", "TAO": "TAOUSDC", "ARB": "ARBUSDC",
                    **{pair: pair for pair in self._usdc_pairs},
                }
        finally:
            self._last_refresh = time.time()

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
            return self._log_platform_price_warning("BinancePlatform", symbol, e)


@PricePlatformFactory.register_platform
class HyperliquidPricePlatform(CryptoPricePlatformBase):
    PLATFORM_KEY = "hyperliquid"
    PRIORITY_ORDER = 20
    platform_name = "Hyperliquid"

    def __init__(self):
        self._url = "https://api.hyperliquid.xyz/info"
        self._all_mids: Dict[str, float] = {}
        self._init_platform_refresh(300)

    def _load_symbols(self):
        try:
            mids = self._fetch_json_post(
                self._url,
                {"type": "allMids"},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if not isinstance(mids, dict) or not mids:
                raise ValueError("Hyperliquid allMids response must be a non-empty dict")
            self._all_mids = mids
            self._supported_symbols = set(self._all_mids.keys())
            print(f"[HyperliquidPlatform] Loaded {len(self._supported_symbols)} symbols")
        except Exception as e:
            print(f"[HyperliquidPlatform] Warning loading symbols: {e}")
            if not self._supported_symbols:
                self._supported_symbols = {"HYPE", "PURR", "BTC", "ETH", "SOL", "USDC", "TAO", "ARB", "DOGE"}
        finally:
            self._last_refresh = time.time()

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            if not isinstance(symbol, str) or not symbol.strip():
                return None
            sym_clean = symbol.strip()
            sym_up = sym_clean.upper()
            base_up = get_base_symbol(sym_up)
            self.refresh_symbols()
            raw_val = casefold_get(self._all_mids, sym_clean)
            if raw_val is None and base_up:
                raw_val = casefold_get(self._all_mids, base_up)
            if raw_val is None:
                print(f"[HyperliquidPlatform] Symbol {sym_up} not found")
                return None
            return safe_pos_float(raw_val, default=None)
        except Exception as e:
            return self._log_platform_price_warning("HyperliquidPlatform", symbol, e)


_CMC_KEY_UNSET = object()


@PricePlatformFactory.register_platform
class CoinMarketCapPricePlatform(CryptoPricePlatformBase):
    PLATFORM_KEY = "coinmarketcap"
    REQUIRES_CMC_KEY = True
    PRIORITY_ORDER = 30
    platform_name = "CoinMarketCap"

    def __init__(self, api_key: Optional[str] = _CMC_KEY_UNSET):  # type: ignore[assignment]
        # Respect explicit api_key=None (offline/disabled) without falling back to os.environ.
        if api_key is _CMC_KEY_UNSET:
            raw_key = optional_env("CMC_API_KEY", "") or None
        else:
            raw_key = api_key
        self.api_key = raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None
        self._base_url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
        self._all_listings: Dict[str, Dict] = {}
        self._cache_file = cache_path(CMC_LISTINGS_CACHE_FILENAME)
        self._init_platform_refresh(86400, load_now=bool(self.api_key))
        if not self.api_key:
            print("[CMCPlatform] No API key - platform disabled")

    def _load_symbols_from_disk_cache(self, max_age_s: Optional[float] = None) -> bool:
        try:
            prev_file = f"{self._cache_file}.previous"
            if not os.path.exists(self._cache_file) and not os.path.exists(prev_file):
                return False
            stat_path = self._cache_file if os.path.exists(self._cache_file) else prev_file
            mtime = os.path.getmtime(stat_path)
            if max_age_s is not None and (time.time() - mtime) > max_age_s:
                return False

            cached = read_json_dict(self._cache_file, default=None, recover_previous=True)
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
            headers = {"X-CMC_PRO_API_KEY": self.api_key, "Accept": "application/json"}
            params = {"limit": 5000, "convert": "USD"}
            data = self._fetch_json_get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                headers=headers,
                params=params,
                timeout=15,
            )
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError("CoinMarketCap listings response must contain a data list")
            self._supported_symbols.clear()
            for crypto in data["data"]:
                if not isinstance(crypto, dict):
                    continue
                raw_symbol = crypto.get("symbol")
                symbol = raw_symbol.strip().upper() if isinstance(raw_symbol, str) else ""
                if symbol:
                    self._supported_symbols.add(symbol)
                    self._all_listings[symbol] = {
                        "id": crypto.get("id"),
                        "name": crypto.get("name"),
                        "slug": crypto.get("slug"),
                    }
            print(f"[CMCPlatform] Loaded {len(self._supported_symbols)} symbols")
            try:
                os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
                save_json_state(self._cache_file, self._all_listings)
            except Exception:
                pass
        except Exception as e:
            if not self._supported_symbols:
                self._load_symbols_from_disk_cache(max_age_s=None)
            print(f"[CMCPlatform] Warning loading symbols: {e}")
        finally:
            self._last_refresh = time.time()

    def _extract_usd_price(self, symbol: str, data: Dict) -> Optional[float]:
        if not isinstance(data, dict):
            return None
        data_map = data.get("data")
        coin_data = casefold_get(data_map, symbol) if isinstance(data_map, dict) else None
        if isinstance(coin_data, list):
            coin_data = coin_data[0] if coin_data else None
        if isinstance(coin_data, dict):
            quote_map = coin_data.get("quote")
            usd = quote_map.get("USD") if isinstance(quote_map, dict) else None
            price = usd.get("price") if isinstance(usd, dict) else None
            return safe_pos_float(price, default=None)
        return None

    def get_price(self, symbol: str) -> Optional[float]:
        if not self.api_key or not isinstance(symbol, str) or not symbol.strip():
            if not self.api_key:
                print("[CMCPlatform] Missing API key")
            return None
        symbol = symbol.strip().upper()
        self.refresh_symbols()
        try:
            lookup_symbol = symbol if symbol in self._supported_symbols else get_base_symbol(symbol)
            headers = {"X-CMC_PRO_API_KEY": self.api_key, "Accept": "application/json"}
            params = {"symbol": lookup_symbol, "convert": "USD"}
            data = self._fetch_json_get(
                self._base_url, headers=headers, params=params, timeout=10
            )
            price = self._extract_usd_price(lookup_symbol, data)
            if price is not None:
                print(f"[CMCPlatform] {symbol} = ${price}")
                return price
            print(f"[CMCPlatform] {symbol} not found")
            return None
        except Exception as e:
            return self._log_platform_price_warning("CMCPlatform", symbol, e)


@PricePlatformFactory.register_platform
class StockYahooPricePlatform(EquityPricePlatformBase):
    """Fetch configured tokenized/US stocks from Yahoo Finance as the RTH equity leaf platform."""

    PLATFORM_KEY = "yahoo"
    PRIORITY_ORDER = 90
    _YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    _UA = {"User-Agent": "Mozilla/5.0 (pricefetcher)"}
    platform_name = "Yahoo"

    def __init__(self, tickers: Optional[List[str]] = None):
        raw_tickers = required_env("STOCK_TICKERS") if tickers is None else tickers
        self._symbols: Set[str] = set(parse_symbol_list(raw_tickers))
        self._supported_symbols = self._symbols

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            return False
        return symbol.strip().upper() in self._symbols

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            if not isinstance(symbol, str) or not symbol.strip():
                return None
            sym_y = strip_equity_suffix(symbol, base_only=True)
            url = self._YAHOO.format(sym=quote(sym_y, safe="")) + "?range=1d&interval=5m"
            return extract_yahoo_chart_latest_price(
                self._fetch_json_get(url, headers=self._UA, timeout=10), decimals=4
            )
        except Exception as e:
            return self._log_platform_price_warning("YahooPlatform", symbol, e)


def is_rth_equity_symbol(symbol: object, price_factory: Optional[object] = None) -> bool:
    """Return True when ``symbol`` is a US RTH equity (by suffix, equity-only platform, or instruments.conf)."""
    if not isinstance(symbol, str) or isinstance(symbol, bool) or not symbol.strip():
        return False
    sym_u = symbol.strip().upper()
    if sym.is_equity_symbol(sym_u):
        return True
    if price_factory is not None and hasattr(price_factory, "_platforms"):
        eq_sup = False
        other_sup = False
        for plat in getattr(price_factory, "_platforms", ()) or ():
            is_eq_plat = bool(getattr(plat, "IS_EQUITY_PLATFORM", False)) or getattr(plat, "platform_name", "") == "Yahoo"
            if is_eq_plat:
                eq_sup = eq_sup or sym_u in getattr(plat, "_symbols", getattr(plat, "_supported_symbols", set()))
            elif sym_u in getattr(plat, "_supported_symbols", set()):
                other_sup = True
        if eq_sup and not other_sup:
            return True
    try:
        reg = sym.load_registry()

        def _matches_sym(s) -> bool:
            return (
                s.symbol.upper() == sym_u
                or s.base.upper() == sym_u
                or canonical_base_asset(s.symbol).upper() == sym_u
            )

        has_eq = any(
            (sym.is_equity_venue(s.provider) or s.market_hours == "rth") and _matches_sym(s)
            for s in reg.values()
        )
        has_crypto = any(
            not sym.is_equity_venue(s.provider) and s.market_hours != "rth" and _matches_sym(s)
            for s in reg.values()
        )
        return has_eq and not has_crypto
    except Exception:
        return False
