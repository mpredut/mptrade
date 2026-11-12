# pricefetcher.py
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import time
import requests  # noqa: F401 -- retained as monkeypatch seam for market_monitor.pricefetcher.requests
from datetime import datetime
from typing import Dict, List, Optional

import instrument_registry as sym
from botcore import (
    clean_nonempty_str,
    exact_pos_int,
    is_rth_gate_enabled,
    is_us_rth_open,
    normalize_epoch_sec,
    optional_env,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_WATCHER_ALL_PRICE_FILENAME,
    normalize_venue_name,
)
from cacheManager import CacheFactory
from cacheprice import CachePriceManagerBase
from instrument_registry import casefold_get
from market_monitor.price_platforms import (
    BinancePricePlatform,
    CoinMarketCapPricePlatform,
    CryptoPricePlatformBase,
    EquityPricePlatformBase,
    HyperliquidPricePlatform,
    PricePlatformFactory,
    PricePlatformInterface,
    StockYahooPricePlatform,
    get_base_symbol,
    is_rth_equity_symbol,
    is_valid_symbol_for_monitoring,
)
from providers.market_api import api
from state_io import atomic_write_json, read_json_dict


PRICE_HISTORY_RETENTION_DAYS = 7
MAX_PRICE_HISTORY_PER_SYMBOL = 2000
MAX_MONITORED_SYMBOLS = 20
QUOTE_CURRENCY = "USDC"
DEFAULT_SYMBOLS = ["BTC", "TAO", "HYPE"]
PRICE_MULTI_SYNC_INTERVAL_SEC = 5 * 60


class CacheAllPriceFetcherManager(CachePriceManagerBase):
    DEFAULT_FILENAME = CACHE_WATCHER_ALL_PRICE_FILENAME

    def __init__(
        self,
        sync_ts,
        symbols,
        filename=CACHE_WATCHER_ALL_PRICE_FILENAME,
        api_client=api,
        cmc_api_key: Optional[str] = None,
        read_only=False,
        max_symbols: Optional[int] = None,
        *,
        data_sync_mode: Optional[str] = None,
    ):
        self.price_factory = PricePlatformFactory(cmc_api_key=cmc_api_key)
        self.max_monitored_symbols = safe_pos_int(max_symbols, default=MAX_MONITORED_SYMBOLS)
        initial_symbols = []
        seen_bases = set()
        for symbol in (symbols or ()):
            base_symbol = get_base_symbol(symbol)
            if base_symbol not in seen_bases and is_valid_symbol_for_monitoring(symbol):
                seen_bases.add(base_symbol)
                initial_symbols.append(symbol)
            if len(initial_symbols) >= self.max_monitored_symbols:
                break
        self.original_symbols = list(initial_symbols)
        self.active_symbols = set(initial_symbols)
        self.symbol_added_time: Dict[str, float] = {}
        self.symbol_preferred_source: Dict[str, str] = {}
        super().__init__(
            sync_ts,
            initial_symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
        )
        self._load_symbol_metadata()
        print("[Pricefetcher] Checking symbol support:")
        for symbol in self.original_symbols:
            support = self.price_factory.check_symbol_support(symbol)
            supported_platforms = [p for p, s in support.items() if s]
            if supported_platforms:
                print(f"  [OK] {symbol} -> {', '.join(supported_platforms)}")
            else:
                print(f"  [MISSING] {symbol} -> NO PLATFORM SUPPORT!")

    def _load_symbol_metadata(self):
        data = read_json_dict(
            self.filename,
            default=None,
            recover_previous=True,
            repair_files=not getattr(self, "read_only", False),
        )
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
                symbol
                for symbol, added_time in self.symbol_added_time.items()
                if (
                    casefold_get(items_map, symbol) is not None
                    and (at_val := safe_pos_float(added_time, default=None)) is not None
                    and time.time() - at_val <= PRICE_HISTORY_RETENTION_DAYS * 24 * 3600
                )
            ]
            for symbol in list(dict.fromkeys(active_symbols + recovered_symbols)):
                if len(self.active_symbols) >= self.max_monitored_symbols:
                    print(
                        f"[Pricefetcher] Limit reached while loading: maximum "
                        f"{self.max_monitored_symbols} monitored coins, the rest will be ignored"
                    )
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
                    for sym_list in (self.symbols, self.original_symbols):
                        sym_list.append(symbol)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"[Pricefetcher] Metadata cache ignored: {exc}")

    def _reload_from_disk(self):
        super()._reload_from_disk()
        with self.lock:
            self._load_symbol_metadata()

    def rebuild_fetchtimes(self):
        return self._rebuild_symbol_entry_fetchtimes(entry_lookup_fn=casefold_get)

    def _resolve_preferred_platform(self, preferred_source: str):
        resolver = getattr(self.price_factory, "resolve_preferred_platform", None)
        if callable(resolver) and type(self.price_factory) is PricePlatformFactory:
            return resolver(preferred_source)
        pref_raw = preferred_source.strip().lower()
        pref_norm = normalize_venue_name(pref_raw) or pref_raw
        for platform in getattr(self.price_factory, "_platforms", ()) or ():
            plat_raw = str(platform.platform_name).strip().lower()
            plat_norm = normalize_venue_name(plat_raw) or plat_raw
            if plat_norm == pref_norm or plat_raw == pref_raw:
                return platform
        return None

    def get_remote_items(self, symbol, startTime):
        if (
            is_rth_equity_symbol(symbol, getattr(self, "price_factory", None))
            and is_rth_gate_enabled(getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open)
            and not is_us_rth_open()
        ):
            with self.lock:
                if self._cached_entries_locked(symbol):
                    return []
        try:
            if not hasattr(self, "symbol_preferred_source"):
                self.symbol_preferred_source = {}

            preferred_source = casefold_get(self.symbol_preferred_source, symbol)
            if preferred_source and isinstance(preferred_source, str) and preferred_source.strip():
                platform = self._resolve_preferred_platform(preferred_source)
                if platform is not None:
                    price_val = safe_pos_float(platform.get_price(symbol), default=None)
                    if price_val is not None:
                        timestamp_ms = math.trunc(time.time() * 1000)
                        print(f"[Pricefetcher][{symbol}] ${price_val:.4f} (source: {preferred_source} - preferred)")
                        return [[timestamp_ms, price_val]]
                    print(
                        f"[Pricefetcher][{symbol}] Preferred source {preferred_source} failed; "
                        f"trying fallback providers"
                    )
            result = self.price_factory.get_price(symbol)
            price_val = safe_pos_float(result.get("price") if isinstance(result, dict) else None, default=None)
            if price_val is None:
                return []
            platform_used = result["platform"]
            timestamp_ms = math.trunc(time.time() * 1000)
            print(f"[Pricefetcher][{symbol}] ${price_val:.4f} (source: {platform_used})")
            return [[timestamp_ms, price_val]]
        except Exception as e:
            print(f"[{self.cls_name}][Warn] {symbol}: {e}")
            return []

    def add_symbol(self, symbol: str, preferred_source: Optional[str] = None):
        if not (symbol := clean_nonempty_str(symbol)):
            return False
        with self.lock:
            base_symbol = get_base_symbol(symbol)
            if symbol in self.active_symbols:
                return self._log_tagged_false("Pricefetcher", f"{symbol} is already in the watchlist")
            if any(get_base_symbol(active_symbol) == base_symbol for active_symbol in self.active_symbols):
                return self._log_tagged_false(
                    "Pricefetcher", f"{symbol} ignored: asset {base_symbol} is already being monitored"
                )
            if len(self.active_symbols) >= self.max_monitored_symbols:
                return self._log_tagged_false(
                    "Pricefetcher", f"Limit reached: maximum {self.max_monitored_symbols} monitored coins"
                )
            self.active_symbols.add(symbol)
            self.symbols.append(symbol)
            self.original_symbols.append(symbol)
            if preferred_source:
                self.symbol_preferred_source[symbol] = preferred_source
                print(f"[Pricefetcher] {symbol} - preferred source: {preferred_source}")
            self.symbol_added_time[symbol] = time.time()
            if symbol not in self.cache:
                self.cache[symbol] = []
            if symbol not in self.fetchtime_per_symbol:
                self.fetchtime_per_symbol[symbol] = self.fallback_time_default
            print(f"[Pricefetcher] Symbol added: {symbol}")
        items = self.get_remote_items(
            symbol, self.fetchtime_per_symbol.get(symbol, self.fallback_time_default)
        )
        if items:
            self.update_cache_per_symbol(symbol, items)
        return True

    def remove_symbol(self, symbol: str, reason: str = ""):
        if not (symbol := clean_nonempty_str(symbol)):
            return False
        sym_up = symbol.upper()
        with self.lock:
            pref_map = getattr(self, "symbol_preferred_source", {})
            matching_keys = {
                s
                for s in (
                    set(self.active_symbols)
                    | set(self.symbol_added_time.keys())
                    | (set(pref_map.keys()) if isinstance(pref_map, dict) else set())
                )
                if isinstance(s, str) and (s == symbol or s.strip().upper() == sym_up)
            }
            if not matching_keys:
                return False
            self.symbols = [
                s
                for s in self.symbols
                if not (isinstance(s, str) and (s in matching_keys or s.strip().upper() == sym_up))
            ]
            self.original_symbols = [
                s
                for s in self.original_symbols
                if not (isinstance(s, str) and (s in matching_keys or s.strip().upper() == sym_up))
            ]
            for k in matching_keys:
                self.active_symbols.discard(k)
                self.symbol_added_time.pop(k, None)
                if isinstance(pref_map, dict):
                    pref_map.pop(k, None)
            print(f"[Pricefetcher] Symbol removed: {symbol} {reason}")
            return True

    def _cached_entries_locked(self, symbol: str):
        entries = self.cache.get(symbol)
        if entries is None:
            entries = casefold_get(self.cache, symbol.strip())
        return entries if isinstance(entries, (list, tuple)) else ()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest saved price for a symbol."""
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        with self.lock:
            for entry in reversed(self._cached_entries_locked(symbol)):
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
        with self.lock:
            entries = self._cached_entries_locked(symbol)[-limit_i:]

        history: List[Dict] = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            ts_val = safe_pos_float(entry[0], default=None)
            px_val = safe_pos_float(entry[1], default=None)
            if ts_val is None or px_val is None:
                continue
            try:
                ts_sec = normalize_epoch_sec(ts_val)
                readable = datetime.fromtimestamp(ts_sec).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            history.append(
                {
                    "timestamp": math.trunc(ts_val) if ts_val.is_integer() else ts_val,
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
                        "total_entries": sum(len(v) for v in self.cache.values()),
                    },
                    "symbol_metadata": {
                        "added_time": self.symbol_added_time,
                        "active_symbols": list(self.active_symbols),
                        "preferred_source": self.symbol_preferred_source,
                    },
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
                return self._log_cache_saved_ok()
        except Exception as e:
            print(f"[{self.cls_name}][Warn] Saving cache file {self.filename}:", e)
            return False

    def save_state_to_file_if_enabled(self):
        if not self.save_state:
            return False
        return self.save_state_to_file()


def create_cachePriceAll(cmc_api_key: Optional[str] = None, symbols=None, max_symbols=None):
    all_symbols = []
    seen_bases = set()

    if hasattr(sym, "symbols"):
        for s in sym.symbols:
            if is_valid_symbol_for_monitoring(s):
                seen_bases.add(get_base_symbol(s))
                all_symbols.append(s)

    for sym_default in (symbols or DEFAULT_SYMBOLS):
        if not is_valid_symbol_for_monitoring(sym_default):
            continue
        base_symbol = get_base_symbol(sym_default)
        if base_symbol not in seen_bases and sym_default not in all_symbols:
            seen_bases.add(base_symbol)
            all_symbols.append(sym_default)

    eff_max = safe_pos_int(max_symbols, default=MAX_MONITORED_SYMBOLS)
    all_symbols = list(dict.fromkeys(all_symbols))
    all_symbols = [s for s in all_symbols if is_valid_symbol_for_monitoring(s)]
    all_symbols = all_symbols[:eff_max]

    print(f"[pricefetcher] Valid symbols to monitor: {len(all_symbols)}")
    print(f"[pricefetcher] List: {all_symbols}")

    CacheFactory.register(
        "PriceMulti",
        CacheAllPriceFetcherManager,
        filename=CacheAllPriceFetcherManager.DEFAULT_FILENAME,
        sync_ts=lambda: PRICE_MULTI_SYNC_INTERVAL_SEC,
        cmc_api_key=cmc_api_key,
        max_symbols=eff_max,
    )
    print("[Pricefetcher] Manager registered in CacheFactory as 'PriceMulti'")
    cachePriceAll = CacheFactory.get(
        "PriceMulti",
        symbols=all_symbols,
        data_sync_mode=CachePriceManagerBase.MODE_PRODUCER,
    )
    cachePriceAll.max_monitored_symbols = eff_max
    cachePriceAll.periodic_sync(sync_ts=PRICE_MULTI_SYNC_INTERVAL_SEC, save_state=True)

    return cachePriceAll


__all__ = [
    "BinancePricePlatform",
    "CacheAllPriceFetcherManager",
    "CoinMarketCapPricePlatform",
    "CryptoPricePlatformBase",
    "DEFAULT_SYMBOLS",
    "EquityPricePlatformBase",
    "HyperliquidPricePlatform",
    "MAX_MONITORED_SYMBOLS",
    "MAX_PRICE_HISTORY_PER_SYMBOL",
    "PRICE_HISTORY_RETENTION_DAYS",
    "PRICE_MULTI_SYNC_INTERVAL_SEC",
    "PricePlatformFactory",
    "PricePlatformInterface",
    "QUOTE_CURRENCY",
    "StockYahooPricePlatform",
    "create_cachePriceAll",
    "get_base_symbol",
    "is_rth_equity_symbol",
    "is_valid_symbol_for_monitoring",
]


if __name__ == "__main__":
    CMC_API_KEY = optional_env("CMC_API_KEY", "") or None
    print("=" * 60)
    print("Starting crypto price monitor (multi-platform)")
    print(f"Base quote currency: {QUOTE_CURRENCY}")
    print("=" * 60)

    monitor = create_cachePriceAll(cmc_api_key=CMC_API_KEY)
    print("\nWaiting for the first price collection...")
    time.sleep(35)

    print("\nCurrent prices:")
    for symbol in monitor.original_symbols:
        price = monitor.get_latest_price(symbol)
        if price:
            print(f"  {symbol}: ${price:.4f}")
        else:
            print(f"  {symbol}: waiting...")

    print("\nMonitor active. Running in background (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
