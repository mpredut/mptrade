# new_coins_discovery.py
import time
import threading
import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set, Callable
from abc import ABC, abstractmethod

from botcore import _parse_bool_flag, safe_float, safe_pos_float, safe_pos_int
from instrument_registry import casefold_get

# ============================================
# Global configuration
# ============================================

# Sourced from environment (CMC_API_KEY), passed to CoinMarketCapSource unless callers override it.
CMC_API_KEY = os.environ.get("CMC_API_KEY", "").split("#", 1)[0].strip()

# Discovery configuration
NEW_COINS_AGE_DAYS = 3
MAX_NEW_COINS_TO_TRACK = 15
REFRESH_INTERVAL_SECONDS = 3600
# Historical formula: twice the maximum tracked-symbol count, interpreted as days.
NEW_COIN_CLEANUP_MAX_AGE_DAYS = MAX_NEW_COINS_TO_TRACK * 2


# Symbols excluded automatically
EXCLUDED_SYMBOLS = {
    "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "FDUSD",
    "WBTC", "WETH", "stETH", "rETH", "LIDO", "STETH",
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "MATIC", "DOT"
}

EXCLUDED_ASSET_NAME_KEYWORDS = {
    "derivative", "derivatives", "xstock", "stock", "etf", "futures"
}


_safe_num = safe_float


def _safe_days_back(days_back, default: float = NEW_COINS_AGE_DAYS) -> float:
    d = safe_pos_float(days_back, default=None)
    return min(d, 3650.0) if d is not None else float(default)


def _parse_coin_added_at(raw_added, *, utc: bool = False) -> Optional[datetime]:
    """Normalize heterogeneous source timestamps (epoch s/ms, ISO string, datetime) into a comparable datetime."""
    if raw_added is None or isinstance(raw_added, bool):
        return None
    try:
        if pd.isna(raw_added):
            return None
    except Exception:
        pass
    val_f = safe_pos_float(raw_added, default=None) if isinstance(raw_added, (int, float, str)) else None
    if val_f is not None:
        sec = val_f / 1000.0 if val_f > 1e11 else val_f
        try:
            return datetime.fromtimestamp(sec, tz=timezone.utc) if utc else datetime.fromtimestamp(sec)
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(raw_added, str) or not hasattr(raw_added, "replace"):
        try:
            parsed = pd.to_datetime(raw_added)
            if parsed is None or pd.isna(parsed):
                return None
            raw_added = parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed
        except Exception:
            return None
    if not isinstance(raw_added, datetime) and not hasattr(raw_added, "replace"):
        return None
    if utc:
        return raw_added.replace(tzinfo=timezone.utc) if getattr(raw_added, "tzinfo", None) is None else raw_added
    return raw_added.replace(tzinfo=None) if getattr(raw_added, "tzinfo", None) is not None else raw_added


def _filter_recent_coins(items: list, days_back: object, *, utc: bool = False) -> List[Dict]:
    if not items:
        return []
    eff_days = _safe_days_back(days_back)
    cutoff_date = (datetime.now(timezone.utc) if utc else datetime.now()) - timedelta(days=eff_days)
    return [
        item for item in items
        if isinstance(item, dict)
        and (added_at := _parse_coin_added_at(item.get("added_at"), utc=utc)) is not None
        and added_at >= cutoff_date
    ]


def is_trackable_symbol(symbol: str, name: str = "", slug: str = "") -> bool:
    if not symbol or not isinstance(symbol, str):
        return False
    sym_clean = symbol.strip().upper()
    if not sym_clean or not sym_clean.isalnum():
        return False
    if sym_clean in EXCLUDED_SYMBOLS or symbol.strip() in EXCLUDED_SYMBOLS:
        return False
    searchable_name = f"{name or ''} {slug or ''}".lower()
    return not any(keyword in searchable_name for keyword in EXCLUDED_ASSET_NAME_KEYWORDS)

# ============================================
# Abstract interface for new coin sources
# ============================================

class NewCoinsSource(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_new_coins(self, days_back: int = 30) -> List[Dict]:
        pass

    @abstractmethod
    def get_supported_symbols(self) -> Set[str]:
        pass

    def refresh(self):
        pass

    def requires_api_key(self) -> bool:
        return False

    def is_available(self) -> bool:
        return True

# ============================================
# Source 1: CoinMarketCap
# ============================================

class CoinMarketCapSource(NewCoinsSource):
    def __init__(self, api_key: str = CMC_API_KEY):
        self.api_key = api_key
        self.headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': self.api_key}
        self._supported_symbols: Set[str] = set()
        self._last_refresh = 0
        self._cache: List[Dict] = []
        if self.api_key:
            self.refresh()
    
    def get_name(self) -> str:
        return "CoinMarketCap"
    
    def requires_api_key(self) -> bool:
        return True
    
    def is_available(self) -> bool:
        return isinstance(self.api_key, str) and len(self.api_key.strip()) > 0
    
    def get_supported_symbols(self) -> Set[str]:
        return self._supported_symbols.copy()
    
    def refresh(self):
        if not self.is_available():
            print("[CMC Source] Missing API key!")
            return
        try:
            params = {'start': '1', 'limit': '2000', 'convert': 'USD', 'sort': 'date_added', 'sort_dir': 'desc'}
            response = requests.get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest", headers=self.headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and isinstance(data.get('data'), list):
                self._cache = []
                self._supported_symbols.clear()
                for coin in data['data']:
                    if not isinstance(coin, dict):
                        continue
                    symbol = str(coin.get('symbol') or '').strip().upper()
                    name = coin.get('name') or symbol
                    slug = coin.get('slug') or ''
                    if is_trackable_symbol(symbol, name=name, slug=slug):
                        self._supported_symbols.add(symbol)
                        quote_map = coin.get('quote') if isinstance(coin.get('quote'), dict) else {}
                        usd_quote = quote_map.get('USD') if isinstance(quote_map.get('USD'), dict) else {}
                        raw_date = coin.get('date_added')
                        try:
                            added_dt = pd.to_datetime(raw_date) if raw_date else None
                        except Exception:
                            added_dt = None
                        self._cache.append({
                            "symbol": symbol,
                            "name": name,
                            "added_at": added_dt,
                            "source": self.get_name(),
                            "price": usd_quote.get('price') if usd_quote.get('price') is not None else 0,
                            "volume_24h": usd_quote.get('volume_24h') if usd_quote.get('volume_24h') is not None else 0,
                            "market_cap": usd_quote.get('market_cap') if usd_quote.get('market_cap') is not None else 0,
                            "change_24h": usd_quote.get('percent_change_24h') if usd_quote.get('percent_change_24h') is not None else 0,
                            "change_7d": usd_quote.get('percent_change_7d') if usd_quote.get('percent_change_7d') is not None else 0,
                            "slug": slug,
                            "url": f"https://coinmarketcap.com/currencies/{slug}/" if slug else ""
                        })
                print(f"[CMC Source] Loaded {len(self._cache)} coins")
                self._last_refresh = time.time()
        except Exception as e:
            print(f"[CMC Source] Error: {e}")

    def get_new_coins(self, days_back: int = NEW_COINS_AGE_DAYS) -> List[Dict]:
        if not self._cache:
            self.refresh()
        return _filter_recent_coins(self._cache, days_back, utc=True)

# ============================================
# Source 2: CoinGecko
# ============================================

class CoinGeckoSource(NewCoinsSource):
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self._supported_symbols: Set[str] = set()
        self._cache: List[Dict] = []
        self._last_refresh = 0
        self.refresh()
    
    def get_name(self) -> str:
        return "CoinGecko"
    
    def requires_api_key(self) -> bool:
        return False
    
    def is_available(self) -> bool:
        return True
    
    def get_supported_symbols(self) -> Set[str]:
        return self._supported_symbols.copy()
    
    def refresh(self):
        try:
            time.sleep(1.2)
            response = requests.get(f"{self.base_url}/coins/list", timeout=15)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                return
            self._cache = []
            self._supported_symbols.clear()
            for coin in data:
                if not isinstance(coin, dict):
                    continue
                symbol = str(coin.get('symbol') or '').upper()
                if symbol and symbol not in EXCLUDED_SYMBOLS:
                    self._supported_symbols.add(symbol)
                    self._cache.append({
                        "symbol": symbol,
                        "name": coin.get('name') or symbol,
                        "added_at": None,
                        "source": self.get_name(),
                        "price": None, "volume_24h": None, "market_cap": None,
                        "change_24h": None, "change_7d": None,
                        "url": f"https://www.coingecko.com/en/coins/{coin.get('id', '')}"
                    })
            print(f"[CoinGecko Source] Loaded {len(self._cache)} coins")
            self._last_refresh = time.time()
        except Exception as e:
            print(f"[CoinGecko Source] Error: {e}")

    def get_new_coins(self, days_back: int = NEW_COINS_AGE_DAYS) -> List[Dict]:
        if not self._cache:
            self.refresh()
        return []

# ============================================
# Source 3: Binance
# ============================================

class BinanceNewListingsSource(NewCoinsSource):
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self._supported_symbols: Set[str] = set()
        self._new_listings: List[Dict] = []
        self._last_refresh = 0
        self._historical_symbols: Set[str] = set()
        self.refresh()
    
    def get_name(self) -> str:
        return "Binance"
    
    def requires_api_key(self) -> bool:
        return False
    
    def is_available(self) -> bool:
        return True
    
    def get_supported_symbols(self) -> Set[str]:
        return self._supported_symbols.copy()
    
    def refresh(self):
        try:
            response = requests.get(f"{self.base_url}/exchangeInfo", timeout=15)
            response.raise_for_status()
            data = response.json()
            current_symbols = set()
            symbol_details = {}
            raw_symbols = data.get('symbols', []) if isinstance(data, dict) else []
            for symbol_info in (raw_symbols if isinstance(raw_symbols, list) else []):
                if not isinstance(symbol_info, dict):
                    continue
                symbol = str(symbol_info.get('symbol') or '').strip().upper()
                base_asset = str(symbol_info.get('baseAsset') or '').strip().upper()
                quote_asset = str(symbol_info.get('quoteAsset') or '').strip().upper()
                status = str(symbol_info.get('status') or '').strip().upper()
                if symbol and base_asset and quote_asset == 'USDC' and status == 'TRADING':
                    current_symbols.add(symbol)
                    symbol_details[symbol] = {"symbol": base_asset, "full_symbol": symbol, "quote": quote_asset}
            if self._historical_symbols:
                new_symbols = current_symbols - self._historical_symbols
                if new_symbols:
                    print(f"[Binance Source] 🆕 Newly listed coins: {new_symbols}")
                    for sym in new_symbols:
                        if sym in symbol_details:
                            self._new_listings.append({
                                "symbol": symbol_details[sym]['symbol'],
                                "full_symbol": sym,
                                "name": symbol_details[sym]['symbol'],
                                "added_at": datetime.now(),
                                "source": self.get_name(),
                                "price": None, "volume_24h": None, "market_cap": None,
                                "change_24h": None, "change_7d": None,
                                "url": f"https://www.binance.com/en/trade/{sym}"
                            })
            self._historical_symbols = current_symbols.copy()
            self._supported_symbols = current_symbols
            self._last_refresh = time.time()
            print(f"[Binance Source] {len(current_symbols)} active pairs")
        except Exception as e:
            print(f"[Binance Source] Error: {e}")

    def get_new_coins(self, days_back: int = NEW_COINS_AGE_DAYS) -> List[Dict]:
        return _filter_recent_coins(self._new_listings, days_back, utc=False)

# ============================================
# Source 4: DexScreener
# ============================================

class DexScreenerSource(NewCoinsSource):
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"
        self._cache: List[Dict] = []
        self._supported_symbols: Set[str] = set()
        self._last_refresh = 0
        self.refresh()
    
    def get_name(self) -> str:
        return "DexScreener"
    
    def requires_api_key(self) -> bool:
        return False
    
    def is_available(self) -> bool:
        return True
    
    def get_supported_symbols(self) -> Set[str]:
        return self._supported_symbols.copy()
    
    def refresh(self):
        try:
            response = requests.get(f"{self.base_url}/token-profiles", timeout=15)
            if response.status_code == 404:
                response = requests.get("https://api.dexscreener.com/token-profiles/latest", timeout=15)
            if response.status_code == 200:
                data = response.json()
                self._cache = []
                self._supported_symbols.clear()
                profiles = (
                    data.get('profiles', [])
                    if isinstance(data, dict)
                    else (data if isinstance(data, list) else [])
                )
                for profile in profiles[:100]:
                    if not isinstance(profile, dict):
                        continue
                    symbol = str(profile.get('symbol') or '').upper()
                    name = profile.get('name') or symbol
                    chain = profile.get('chainId') or 'unknown'
                    token_address = profile.get('tokenAddress') or ''
                    listed_at = profile.get('listedAt', None)
                    if symbol and len(symbol) <= 10 and symbol.isalnum() and symbol not in EXCLUDED_SYMBOLS:
                        self._supported_symbols.add(symbol)
                        added_time = (
                            _parse_coin_added_at(listed_at, utc=False)
                            if safe_pos_float(listed_at, default=None) is not None
                            else None
                        )
                        self._cache.append({
                            "symbol": symbol, "name": name, "added_at": added_time,
                            "source": self.get_name(), "chain": chain, "token_address": token_address,
                            "price": None, "volume_24h": None, "market_cap": None,
                            "change_24h": None, "change_7d": None,
                            "url": f"https://dexscreener.com/{chain}/{token_address}" if token_address else None
                        })
                print(f"[DexScreener Source] Found {len(self._cache)} tokens")
            else:
                self._cache = []
            self._last_refresh = time.time()
        except Exception as e:
            print(f"[DexScreener Source] Error: {e}")
            self._cache = []
    
    def get_new_coins(self, days_back: int = NEW_COINS_AGE_DAYS) -> List[Dict]:
        return _filter_recent_coins(self._cache, days_back, utc=False)[:50]

# ============================================
# Factory for new-coin data sources.
# ============================================

class NewCoinsFactory:
    def __init__(self, enabled_sources: Optional[List[str]] = None, cmc_api_key: str = CMC_API_KEY):
        builders: Dict[str, Callable[[], NewCoinsSource]] = {
            "coinmarketcap": lambda: CoinMarketCapSource(cmc_api_key),
            "coingecko": CoinGeckoSource,
            "binance": BinanceNewListingsSource,
            "dexscreener": DexScreenerSource,
        }
        if enabled_sources is not None:
            raw_sources = (
                enabled_sources.split(",")
                if isinstance(enabled_sources, str)
                else (enabled_sources if isinstance(enabled_sources, (list, tuple, set, frozenset)) else [])
            )
            selected_names = []
            for name in raw_sources:
                if name is None or isinstance(name, bool):
                    continue
                norm_n = str(name).strip().lower()
                if norm_n in builders and norm_n not in selected_names:
                    selected_names.append(norm_n)
            if not selected_names and not raw_sources:
                selected_names = list(builders.keys())
        else:
            selected_names = list(builders.keys())
        self._all_sources: Dict[str, NewCoinsSource] = {
            name: builders[name]() for name in selected_names
        }
        self.sources = {
            name: src for name, src in self._all_sources.items() if src.is_available()
        }
        print(f"[NewCoinsFactory] 🔧 Enabled sources: {list(self.sources.keys())}")

    def get_all_new_coins(self, days_back: int = NEW_COINS_AGE_DAYS) -> Dict[str, List[Dict]]:
        results = {}
        for name, source in self.sources.items():
            try:
                results[name] = source.get_new_coins(days_back)
            except Exception as e:
                print(f"[NewCoinsFactory] Error for {name}: {e}")
                results[name] = []
        return results
    
    def get_all_new_symbols(self, days_back: int = NEW_COINS_AGE_DAYS) -> Set[str]:
        all_symbols = set()
        for name, source in self.sources.items():
            try:
                for coin in source.get_new_coins(days_back):
                    if isinstance(coin, dict) and isinstance(coin.get('symbol'), str) and coin['symbol'].strip():
                        all_symbols.add(coin['symbol'].strip().upper())
            except Exception:
                pass
        return all_symbols
    
    def get_available_sources(self) -> List[str]:
        return list(self.sources.keys())
    
    def refresh_all(self):
        for source in self.sources.values():
            try:
                source.refresh()
            except Exception as e:
                print(f"[NewCoinsFactory] Error refreshing {source.get_name()}: {e}")

# ============================================
# Main monitor
# ============================================

class NewCoinsMonitor:
    def __init__(self, price_monitor=None, factory: Optional[NewCoinsFactory] = None):
        self.price_monitor = price_monitor
        self.factory = factory or NewCoinsFactory()
        self.all_new_coins: Dict[str, List[Dict]] = {}
        self.all_symbols: Set[str] = set()
        self.alert_callbacks: List[Callable] = []
        self._running = False
        self._thread = None
        self.refresh()

    def is_valid_symbol(self, symbol: str, name: str = "", slug: str = "") -> bool:
        return is_trackable_symbol(symbol, name=name, slug=slug)

    def refresh(self, refresh_sources: bool = False):
        if _parse_bool_flag(refresh_sources, False):
            self.factory.refresh_all()
        self.all_new_coins = self.factory.get_all_new_coins(NEW_COINS_AGE_DAYS)
        self.all_symbols = set()
        print(f"[NewCoinsMonitor] Refreshed new coins from sources: {', '.join(self.all_new_coins.keys())}")
        for source_name, coins in self.all_new_coins.items():
            if not isinstance(coins, (list, tuple)):
                continue
            for coin in coins:
                if not isinstance(coin, dict):
                    continue
                raw_symbol = coin.get('symbol')
                symbol = str(raw_symbol).strip().upper() if isinstance(raw_symbol, str) else ""
                if symbol and self.is_valid_symbol(symbol, str(coin.get('name') or ''), str(coin.get('slug') or '')):
                    self.all_symbols.add(symbol)
                else:
                    print(f"[NewCoinsMonitor] Ignored invalid symbol: {raw_symbol} (from {source_name})")
                    continue

    def register_alerts_callback(self, callback: Callable):
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, new_coins: List[Dict], source_name: str, auto_add: bool = True):
        if not new_coins:
            return
        auto_add = _parse_bool_flag(auto_add, True)

        source_has_price = str(source_name or "").strip().lower() == "coinmarketcap"
        alerts = []

        for coin in new_coins:
            if not isinstance(coin, dict) or not coin.get("symbol"):
                continue
            sym = str(coin["symbol"]).strip()
            if not sym:
                continue
            if auto_add and source_has_price:
                auto_added_status = self.add_new_coin_to_watchlist(coin)
            else:
                auto_added_status = False

                if not source_has_price:
                    print(
                        f"[NewCoinsMonitor] ℹ️ {sym} discovered from "
                        f"{source_name} - informational only (no price in the system)"
                    )

            alerts.append({
                "type": "new_coin_discovered",
                "source": source_name,
                "symbol": sym,
                "name": coin.get("name", sym),
                "added_at": coin.get("added_at"),
                "price": coin.get("price", 0),
                "auto_added": auto_added_status,
                "has_price": source_has_price,
                "url": coin.get("url", "")
            })

        if not alerts:
            return

        # Send all alerts through a single callback invocation.
        for callback in self.alert_callbacks:
            try:
                callback(alerts)
            except Exception as e:
                print(f"[NewCoinsMonitor] Callback error: {e}")

                
    def start_monitoring(self, interval_seconds: int = REFRESH_INTERVAL_SECONDS):
        if not isinstance(interval_seconds, (int, float)) or safe_pos_float(interval_seconds, default=None) is None:
            raise ValueError("interval_seconds must be a finite positive number")
        if self._running:
            return
        self._running = True
        def run():
            print(f"[NewCoinsMonitor] 🔄 Monitoring started (refresh every {interval_seconds}s)")
            print(f"[NewCoinsMonitor] 📡 Active sources: {self.factory.get_available_sources()}")
            print("[NewCoinsMonitor] ⚠️ Only CoinMarketCap coins will be auto-added (they include price data)")
            while self._running:
                try:
                    old_symbols = self.all_symbols.copy()
                    self.refresh(refresh_sources=True)
                    new_symbols = self.all_symbols - old_symbols
                    if new_symbols:
                        print(f"[NewCoinsMonitor] 🆕 New symbols detected: {new_symbols}")
                        for source_name, coins in self.all_new_coins.items():
                            if not isinstance(coins, (list, tuple)):
                                continue
                            new_from_source = [
                                c for c in coins
                                if isinstance(c, dict)
                                and isinstance(c.get('symbol'), str)
                                and c['symbol'].strip().upper() in new_symbols
                            ]
                            if new_from_source:
                                auto_add = (source_name.lower() == "coinmarketcap")
                                self._trigger_alerts(new_from_source, source_name, auto_add=auto_add)
                except Exception as e:
                    print(f"[NewCoinsMonitor] Error: {e}")
                itv_f = safe_pos_float(interval_seconds, default=60.0)
                print(f"[NewCoinsMonitor] ⏳ Waiting {itv_f} seconds until the next check...")
                steps = max(1, int(itv_f) + (1 if itv_f > int(itv_f) else 0))
                for _ in range(steps):
                    if not self._running:
                        break
                    time.sleep(min(1.0, itv_f))
        self._thread = threading.Thread(target=run, name="NewCoinsMonitor", daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        print("[NewCoinsMonitor] Monitoring stopped")

    def get_report(self) -> str:
        report = "\n" + "=" * 80 + "\n"
        report += f"🆕 NEW COINS REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
        report += "=" * 80 + "\n"
        for source_name, coins in self.all_new_coins.items():
            if coins:
                report += f"\n📡 SOURCE: {source_name.upper()} ({len(coins)} coins)\n"
                report += "-" * 40 + "\n"
                for coin in coins[:15]:
                    if not isinstance(coin, dict):
                        continue
                    sym_str = str(coin.get('symbol') or 'N/A')
                    raw_added = coin.get('added_at')
                    added_str = raw_added.strftime('%Y-%m-%d') if (raw_added is not None and not pd.isna(raw_added) and hasattr(raw_added, 'strftime')) else 'N/A'
                    px_val = safe_float(coin.get('price'), 0.0)
                    price_str = f"${px_val:.6f}" if px_val > 0 else 'N/A'
                    report += f"   🆕 {sym_str} - {coin.get('name', sym_str)}\n"
                    report += f"      📅 Added: {added_str} | 💰 Price: {price_str}\n"
                    if coin.get('url'):
                        report += f"      🔗 {coin['url']}\n"
        report += "\n" + "=" * 80 + "\n"
        return report

    def get_summary(self) -> Dict:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "sources": {},
            "total_new_coins": len(self.all_symbols),
            "all_symbols": list(self.all_symbols)[:50]
        }
        for source_name, coins in self.all_new_coins.items():
            summary["sources"][source_name] = {
                "count": len(coins),
                "symbols": [c.get('symbol') for c in coins if isinstance(c, dict) and c.get('symbol')][:20]
            }
        return summary

    def add_new_coin_to_watchlist(self, coin_info: Dict, auto_add_thresholds: Optional[Dict] = None):
        """
        Add a newly discovered coin to the price watchlist.
        The filters are informational only - the coin is added regardless.
        """
        if not self.price_monitor:
            print("[NewCoinsMonitor] No price_monitor available")
            return False
        if not isinstance(coin_info, dict) or not coin_info.get('symbol') or isinstance(coin_info.get('symbol'), bool):
            return False

        symbol = str(coin_info['symbol']).strip().upper()
        if not symbol:
            return False
        source = coin_info.get('source', 'unknown')

        if not self.is_valid_symbol(symbol, coin_info.get('name', ''), coin_info.get('slug', '')):
            print(f"[NewCoinsMonitor] ❌ Invalid symbol: {symbol}")
            return False

        if hasattr(self.price_monitor, 'original_symbols') and symbol in self.price_monitor.original_symbols:
            print(f"[NewCoinsMonitor] {symbol} is already in the watchlist")
            return False
        raw_max_wl = getattr(self.price_monitor, 'max_monitored_symbols', MAX_NEW_COINS_TO_TRACK)
        max_watchlist_symbols = safe_pos_int(raw_max_wl, default=MAX_NEW_COINS_TO_TRACK) or MAX_NEW_COINS_TO_TRACK
        if hasattr(self.price_monitor, 'active_symbols') and len(self.price_monitor.active_symbols) >= max_watchlist_symbols:
            print(f"[NewCoinsMonitor] Watchlist limit reached: maximum {max_watchlist_symbols} coins")
            return False

        # Informational thresholds only; they do not block the add operation
        default_thresholds = {
            "min_volume_24h": 100000.0,      # Minimum volume $100k
            "min_price_usd": 0.000001,       # Minimum price
            "max_price_usd": 100000.0,       # Maximum price
            "min_change_24h": -90.0,         # Minimum accepted decline
            "max_change_24h": 1000.0,        # Maximum accepted increase
        }
        if isinstance(auto_add_thresholds, dict):
            merged_thresholds = {
                k: safe_float(auto_add_thresholds.get(k, v), v)
                for k, v in default_thresholds.items()
            }
        else:
            merged_thresholds = default_thresholds
        auto_add_thresholds = merged_thresholds

        # Check thresholds for informational display only
        volume = safe_float(coin_info.get('volume_24h', 0), 0.0)
        price = safe_float(coin_info.get('price', 0), 0.0)
        change = safe_float(coin_info.get('change_24h', 0), 0.0)

        issues = []

        if volume and volume < auto_add_thresholds["min_volume_24h"]:
            issues.append(f"⚠️ volume below threshold (${volume:,.0f} < ${auto_add_thresholds['min_volume_24h']:,.0f})")

        if price and price < auto_add_thresholds["min_price_usd"]:
            issues.append(f"⚠️ price below threshold (${price:.8f} < ${auto_add_thresholds['min_price_usd']})")

        if price and price > auto_add_thresholds["max_price_usd"]:
            issues.append(f"⚠️ price above threshold (${price:.2f} > ${auto_add_thresholds['max_price_usd']})")

        if change and change < auto_add_thresholds["min_change_24h"]:
            issues.append(f"⚠️ decline below threshold ({change:.1f}% < {auto_add_thresholds['min_change_24h']}%)")

        if change and change > auto_add_thresholds["max_change_24h"]:
            issues.append(f"⚠️ increase above threshold ({change:.1f}% > {auto_add_thresholds['max_change_24h']}%)")

        # Display warnings, but do not block
        if issues:
            print(f"[NewCoinsMonitor] 📊 {symbol}: {', '.join(issues)}")
        else:
            print(f"[NewCoinsMonitor] 📊 {symbol} - ALL THRESHOLDS MET ✅")

        # Add the symbol to the watchlist regardless of filters
        try:
            if hasattr(self.price_monitor, 'add_symbol'):
                added = self.price_monitor.add_symbol(symbol, preferred_source=source)
                if added is False:
                    return False
                print(f"[NewCoinsMonitor] ✅ {symbol} added to the watchlist (price from {source})")
                return True
            else:
                print("[NewCoinsMonitor] ⚠️ price_monitor does not have add_symbol()")
                return False
        except Exception as e:
            print(f"[NewCoinsMonitor] Error while adding {symbol}: {e}")
            return False

    def should_keep_monitoring(self, symbol: str) -> bool:
        added_map = getattr(self.price_monitor, 'symbol_added_time', None)
        if not isinstance(added_map, dict):
            return True
        added_time = casefold_get(added_map, symbol)
        ts_val = safe_float(added_time, 0.0)
        if ts_val <= 0.0:
            return True
        age_days = (time.time() - ts_val) / (24 * 3600)
        return age_days <= NEW_COIN_CLEANUP_MAX_AGE_DAYS

    def cleanup_old_new_coins(self):
        if not self.price_monitor:
            return
        removed = []
        for symbol in list(self.all_symbols):
            if not self.should_keep_monitoring(symbol):
                if hasattr(self.price_monitor, 'remove_symbol'):
                    self.price_monitor.remove_symbol(symbol, reason="old new coin")
                self.all_symbols.discard(symbol)
                removed.append(symbol)
        if removed:
            print(f"[NewCoinsMonitor] Cleaned up {len(removed)} old coins from the watchlist")

# ============================================
# Convenience functions
# ============================================

def create_new_coins_checker(price_monitor=None, enabled_sources=None, cmc_api_key=CMC_API_KEY):
    factory = NewCoinsFactory(enabled_sources=enabled_sources, cmc_api_key=cmc_api_key)
    monitor = NewCoinsMonitor(price_monitor, factory)
    print("[NewCoins] ✅ New coin monitor initialized")
    print(f"[NewCoins] 📡 Active sources: {factory.get_available_sources()}")
    return monitor, factory

# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("=" * 80)
    print("🆕 NEW COIN DISCOVERY MODULE TEST")
    print("=" * 80)
    monitor, factory = create_new_coins_checker()
    print(f"\n📡 Active sources: {factory.get_available_sources()}")
    print(monitor.get_report())
    summary = monitor.get_summary()
    print(f"\n📊 SUMMARY: Total new symbols: {summary['total_new_coins']}")
    for source, data in summary['sources'].items():
        print(f"   {source}: {data['count']} coins")
    print("\n✅ Test complete")
