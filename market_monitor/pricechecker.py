# pricechecker.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
import math
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from operator import itemgetter
from urllib.parse import quote

# Import your existing modules
import log
from botcore import safe_float, safe_nonneg_float, safe_pos_float
from cachepaths import canonical_base_asset
from market_monitor.pricefetcher import get_base_symbol

# Canonical CoinMarketCap slugs for major symbols. _all_listings is keyed by
# symbol, so when several coins share one (for example, real Bitcoin and a scam
# "Bitcoin AI" token both using BTC), the last listing overwrites the correct
# slug. This map forces canonical slugs for watchlist and major coins.
_CANONICAL_CMC_SLUG = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano",
    "XRP": "xrp", "DOGE": "dogecoin", "TAO": "bittensor", "HYPE": "hyperliquid",
    "PEPE": "pepe", "PURR": "purr-2", "WIF": "dogwifhat", "FLR": "flare-networks",
}

# Threshold configuration (can be adjusted at any time)
PRICE_ALERT_CONFIG = {
    "default": {
        "up_percent": 4.1,    # Trigger an alert when the price rises by 5% from the 24h low
        "down_percent": 7.5,  # Trigger an alert when the price drops by 7.5% from the 24h high
    },
    "dynamic": {
        "up_percent": 12.0,    # Stricter threshold for dynamically added coins
        "down_percent": 25.0,  # Stricter threshold for dynamically added coins
    },
    "lookback_hours": 24,    # Analysis interval (24 hours)
    "cooldown_minutes": 30,  # Do not send the same alert more often than every 15 minutes
}


class PriceAlert:

    def __init__(self, symbol: str, alert_type: str, current_price: float,
                 reference_price: float, percent_change: float, threshold: float,
                 url: Optional[str] = None, reference_time: Optional[str] = None):
        self.symbol = "N/A" if isinstance(symbol, bool) or symbol is None else (str(symbol).strip() or "N/A")
        raw_atype = "down" if isinstance(alert_type, bool) or alert_type is None else (str(alert_type).strip().lower() or "down")
        self.alert_type = "up" if raw_atype in ("up", "rise", "▲") else ("down" if raw_atype in ("down", "drop", "▼") else raw_atype)
        self.current_price = safe_float(current_price, default=0.0)
        self.reference_price = safe_float(reference_price, default=0.0)
        self.percent_change = safe_float(percent_change, default=0.0)
        self.threshold = safe_float(threshold, default=0.0)
        self.timestamp = time.time()
        self.url = "" if isinstance(url, bool) or url is None else str(url)
        self.reference_time = None if isinstance(reference_time, bool) else reference_time

    def __str__(self) -> str:
        direction = "🚀 RISE" if self.alert_type == "up" else "📉 DROP"
        emoji = "🟢" if self.alert_type == "up" else "🔴"
        if self.reference_time:
            reference_time = self.reference_time
        else:
            try:
                reference_time = datetime.fromtimestamp(float(self.timestamp)).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError, OSError, OverflowError):
                reference_time = str(self.timestamp)
        url_line = f"🔗 CoinMarketCap: {self.url}\n" if self.url else ""
        return (
            f"\n{emoji} {direction} {emoji}\n"
            f"📊 Coin: {self.symbol}\n"
            f"💰 Current price: ${self.current_price:.4f}\n"
            f"📈 Reference: ${self.reference_price:.4f} ({'24h low' if self.alert_type == 'up' else '24h high'}, at {reference_time})\n"
            f"{url_line}"
            f"📊 Change: {self.percent_change:+.2f}% (threshold: {self.threshold}%)"
        )

    def to_dict(self) -> dict:
        try:
            ts_readable = datetime.fromtimestamp(float(self.timestamp)).isoformat()
        except (TypeError, ValueError, OSError, OverflowError):
            ts_readable = str(self.timestamp)
        return {
            "symbol": self.symbol,
            "alert_type": self.alert_type,
            "current_price": self.current_price,
            "reference_price": self.reference_price,
            "percent_change": self.percent_change,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "timestamp_readable": ts_readable,
            "reference_time": self.reference_time,
            "url": self.url,
        }


class PriceChecker:
    """
    Analyze cached prices and generate alerts when thresholds are exceeded.
    Runs in a separate thread.
    """

    def __init__(self, cachePriceAll, alert_callback: Optional[Callable] = None,
                 config: Optional[dict] = None):
        self.cachePriceAll = cachePriceAll
        self.alert_callback = alert_callback or self._default_alert_handler
        # Use market_alerts.conf configuration, including per_coin, when supplied;
        # otherwise retain the legacy hardcoded defaults.
        self.config = copy.deepcopy(PRICE_ALERT_CONFIG)
        if isinstance(config, dict):
            for key, value in config.items():
                if key in ("default", "dynamic"):
                    if isinstance(value, dict):
                        self.config[key].update(copy.deepcopy(value))
                else:
                    self.config[key] = copy.deepcopy(value)

        # Prevent spam: remember the last alert per symbol and alert type
        self._last_alert_time = defaultdict(float)

        # Thread for continuous analysis
        self._thread = None
        self._running = False

    def _default_alert_handler(self, alert):
        """Default handler that prints the alert(s) to the console."""
        items = alert if isinstance(alert, (list, tuple)) else [alert]
        for item in items:
            print("\n" + "=" * 60)
            print(str(item))
            print("=" * 60)

    def _build_cmc_url(self, symbol: str) -> str:
        try:
            base = get_base_symbol(symbol) or symbol
            # Canonical major-coin slugs avoid symbol collisions in _all_listings.
            canonical = _CANONICAL_CMC_SLUG.get((base or "").upper())
            if canonical:
                return f"https://coinmarketcap.com/currencies/{canonical}/"
            candidate_symbols = [symbol, base]
            for candidate in candidate_symbols:
                if not candidate:
                    continue
                for platform in getattr(getattr(self.cachePriceAll, "price_factory", None), "_platforms", []):
                    if getattr(platform, "platform_name", "") != "CoinMarketCap":
                        continue
                    listings = getattr(platform, "_all_listings", {})
                    metadata = listings.get(candidate) or listings.get(candidate.upper())
                    if metadata and metadata.get("slug"):
                        return f"https://coinmarketcap.com/currencies/{metadata['slug']}/"
            return f"https://coinmarketcap.com/search/?q={quote(symbol)}"
        except Exception:
            return f"https://coinmarketcap.com/search/?q={quote(symbol)}"

    def _get_price_history_last_hours(self, symbol: str, hours: float):
        """Return compact validated (timestamp_ms, price, readable_time) tuples."""
        hours_f = safe_pos_float(hours, default=None)
        if hours_f is None:
            return []
        eff_hours = min(hours_f, 24.0 * 365.0)
        history_limit = max(1000, min(100_000, int(math.ceil(eff_hours * 60)) + 2))
        history = self.cachePriceAll.get_price_history(symbol, limit=history_limit)
        cutoff_time = (time.time() - eff_hours * 3600) * 1000

        recent_history = []
        for entry in history or ():
            if not isinstance(entry, dict):
                continue
            timestamp = safe_float(entry.get("timestamp"), default=None)
            price = safe_pos_float(entry.get("price"), default=None)
            if timestamp is None or price is None:
                continue
            if timestamp < 10_000_000_000:
                timestamp *= 1000
            if timestamp >= cutoff_time:
                recent_history.append((timestamp, price, entry.get("timestamp_readable")))
        return recent_history


    @staticmethod
    def _readable_time(entry) -> str:
        timestamp, _, readable = entry
        if readable:
            return str(readable)
        try:
            return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, OSError, OverflowError):
            return str(timestamp)

    def _calculate_24h_stats(self, symbol: str) -> Dict:
        """Calculate price extremes and time bounds in one O(n) pass."""
        current_price = safe_pos_float(self.cachePriceAll.get_latest_price(symbol), default=None)
        if current_price is None:
            return {"has_data": False, "error": "No current price available"}

        lookback_hours = self.config.get("lookback_hours", 24)
        history = self._get_price_history_last_hours(symbol, lookback_hours)
        count = len(history)

        if count < 2:
            return {
                "has_data": False,
                "error": f"Insufficient data: only {count} records in the last {lookback_hours}h",
            }

        by_price = itemgetter(1)
        by_time = itemgetter(0)
        min_entry = min(history, key=by_price)
        max_entry = max(history, key=by_price)
        oldest_entry = min(history, key=by_time)
        newest_entry = max(history, key=by_time)
        min_price = min_entry[1]
        max_price = max_entry[1]
        return {
            "has_data": True,
            "current_price": current_price,
            "min_price": min_price,
            "min_price_timestamp_readable": self._readable_time(min_entry),
            "max_price": max_price,
            "max_price_timestamp_readable": self._readable_time(max_entry),
            "up_from_min": ((current_price - min_price) / min_price) * 100,
            "down_from_max": ((current_price - max_price) / max_price) * 100,
            "history_count": count,
            "oldest_time": self._readable_time(oldest_entry),
            "newest_time": self._readable_time(newest_entry),
        }

    @staticmethod
    def _alert_key(symbol: str, alert_type: str) -> str:
        return f"{str(symbol or '').strip().upper()}_{str(alert_type or '').strip().lower()}"

    def _monitored_symbols(self) -> List:
        raw_symbols = getattr(self.cachePriceAll, "original_symbols", None)
        if raw_symbols is None:
            raw_symbols = getattr(self.cachePriceAll, "symbols", None)
        return list(raw_symbols) if isinstance(raw_symbols, (list, tuple, set, frozenset)) else []

    def _should_send_alert(self, symbol: str, alert_type: str) -> bool:
        """Check whether we can send a new alert (spam prevention)."""
        key = self._alert_key(symbol, alert_type)
        now = time.time()
        lf = safe_nonneg_float(self._last_alert_time.get(key, 0.0), default=None)
        if lf is not None and lf <= now:
            last_time = lf
        else:
            last_time = 0.0
            self._last_alert_time[key] = 0.0
        cd_min = safe_nonneg_float(self.config.get("cooldown_minutes", 30), default=30.0)
        cooldown_seconds = cd_min * 60.0

        return (now - last_time) >= cooldown_seconds

    def _is_dynamic_symbol(self, symbol: str) -> bool:
        symbol_added_time = getattr(self.cachePriceAll, "symbol_added_time", {})
        if not isinstance(symbol_added_time, dict) or isinstance(symbol, bool) or symbol is None:
            return False
        sym_clean = str(symbol).strip()
        sym_up = sym_clean.upper()
        raw_ts = (
            symbol_added_time.get(symbol)
            if symbol in symbol_added_time
            else (
                symbol_added_time.get(sym_clean)
                if sym_clean in symbol_added_time
                else symbol_added_time.get(sym_up)
            )
        )
        if raw_ts is None:
            for k, v in symbol_added_time.items():
                if not isinstance(k, bool) and k is not None and str(k).strip().upper() == sym_up:
                    raw_ts = v
                    break
        return safe_pos_float(raw_ts, default=None) is not None

    def _get_thresholds_for_symbol(self, symbol: str) -> Dict:
        fallback = self.config.get("dynamic") if self._is_dynamic_symbol(symbol) else self.config.get("default")
        if not isinstance(fallback, dict):
            fallback = PRICE_ALERT_CONFIG["default"]
        per_coin = self.config.get("per_coin")
        chosen = None
        if isinstance(per_coin, dict) and not isinstance(symbol, bool) and symbol is not None:
            sym_str = str(symbol).strip()
            sym_up = sym_str.upper()
            base = get_base_symbol(sym_up) or get_base_symbol(sym_str) or ""
            base_up = str(base).strip().upper() if base else ""
            canon = canonical_base_asset(sym_up)
            for cand in (sym_str, sym_up, base, base_up, canon):
                if cand and isinstance(per_coin.get(cand), dict):
                    chosen = per_coin[cand]
                    break
            if chosen is None:
                target_keys = {k for k in (sym_up, base_up, canon) if k}
                for k, v in per_coin.items():
                    if isinstance(v, dict) and not isinstance(k, bool) and k is not None:
                        if str(k).strip().upper() in target_keys:
                            chosen = v
                            break
        raw = chosen if isinstance(chosen, dict) else fallback

        def_up = safe_pos_float(fallback.get("up_percent"), default=PRICE_ALERT_CONFIG["default"]["up_percent"])
        def_down = safe_pos_float(fallback.get("down_percent"), default=PRICE_ALERT_CONFIG["default"]["down_percent"])
        return {
            "up_percent": safe_pos_float(raw.get("up_percent"), default=def_up),
            "down_percent": safe_pos_float(raw.get("down_percent"), default=def_down),
        }

    def _record_alert_sent(self, symbol: str, alert_type: str):
        """Record that an alert was sent."""
        self._last_alert_time[self._alert_key(symbol, alert_type)] = time.time()

    def check_symbol(self, symbol: str) -> List[PriceAlert]:
        """Check a symbol and return a list of alerts (0, 1, or 2)."""
        alerts = []
        stats = self._calculate_24h_stats(symbol)

        if not stats.get("has_data", False):
            print(f"[Checker][{symbol}] {stats.get('error', 'Unknown error')}")
            return alerts

        current_price = stats["current_price"]
        up_percent = stats["up_from_min"]
        down_percent = stats["down_from_max"]
        thresholds = self._get_thresholds_for_symbol(symbol)
        up_threshold = thresholds["up_percent"]
        down_threshold = thresholds["down_percent"]

        # Check for price increase (upper threshold)
        if up_percent >= up_threshold:
            if self._should_send_alert(symbol, "up"):
                alert = PriceAlert(
                    symbol=symbol,
                    alert_type="up",
                    current_price=current_price,
                    reference_price=stats["min_price"],
                    reference_time=stats.get("min_price_timestamp_readable"),
                    percent_change=up_percent,
                    threshold=up_threshold,
                    url=self._build_cmc_url(symbol)
                )
                alerts.append(alert)
                self._record_alert_sent(symbol, "up")

        # Check for price decrease (lower threshold)
        if down_percent <= -down_threshold:
            if self._should_send_alert(symbol, "down"):
                alert = PriceAlert(
                    symbol=symbol,
                    alert_type="down",
                    current_price=current_price,
                    reference_price=stats["max_price"],
                    reference_time=stats.get("max_price_timestamp_readable"),
                    percent_change=down_percent,
                    threshold=down_threshold,
                    url=self._build_cmc_url(symbol)
                )
                alerts.append(alert)
                self._record_alert_sent(symbol, "down")

        print(
            f"[Checker][{symbol}] Price: ${current_price:.4f} | "
            f"↑ {up_percent:+.2f}% (threshold +{up_threshold}%) | "
            f"↓ {down_percent:+.2f}% (threshold -{down_threshold}%) | "
            f"Min: ${stats['min_price']:.4f} | Max: ${stats['max_price']:.4f}"
        )

        return alerts

    def check_all_symbols(self) -> List[PriceAlert]:
        """Check all symbols in the watchlist."""
        all_alerts = []
        for symbol in dict.fromkeys(self._monitored_symbols()):
            try:
                alerts = self.check_symbol(symbol)
                all_alerts.extend(alerts)
            except Exception as e:
                print(f"[Checker][{symbol}] Error: {e}")

        return all_alerts

    def start_monitoring(self, interval_seconds: int = 60):
        """
        Start continuous monitoring in a separate thread.

        Args:
            interval_seconds: How often to check (for example 60 seconds).
        """
        iv_f = safe_pos_float(interval_seconds, default=None)
        if iv_f is None:
            raise ValueError("interval_seconds must be greater than zero")
        if self._running:
            print("[Checker] Already running!")
            return

        self._running = True

        def run():
            print(f"[Checker] price checker started - checking every {interval_seconds}s")
            print(
                f"[Checker] Thresholds: default ↑ +{self.config['default']['up_percent']}% | ↓ -{self.config['default']['down_percent']}% | "
                f"dynamic ↑ +{self.config['dynamic']['up_percent']}% | ↓ -{self.config['dynamic']['down_percent']}%"
            )

            while self._running:
                try:
                    alerts = self.check_all_symbols()

                    if alerts:
                        self.alert_callback(alerts)

                except Exception as e:
                    print(f"[Checker] Error in main loop: {e}")

                print(f"[Checker] Waiting {interval_seconds} seconds until next check...")
                for _ in range(max(1, int(math.ceil(iv_f)))):
                    if not self._running:
                        break
                    time.sleep(1)

        self._thread = threading.Thread(target=run, name="PriceChecker", daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        print("[Checker] Monitoring stopped")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "config": self.config,
            "symbols_count": len(self._monitored_symbols()),
            "last_alerts": dict(self._last_alert_time)
        }


def start_price_alert_checker(cachePriceAll, alert_callback=None, check_interval_seconds=60, config=None):
    checker = PriceChecker(cachePriceAll, alert_callback=alert_callback, config=config)
    checker.start_monitoring(check_interval_seconds)
    return checker
