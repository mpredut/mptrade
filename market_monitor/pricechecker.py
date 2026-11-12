# pricechecker.py
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import copy
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from operator import itemgetter
from urllib.parse import quote

from botcore import (
    _parse_bool_flag,
    is_rth_gate_enabled,
    is_us_rth_open,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    sleep_monitoring_interval,
    start_monitoring_daemon_thread,
    stop_monitoring_thread,
)
from cachepaths import cache_symbol_variants
from instrument_registry import casefold_get
from market_monitor.alerts_config import resolve as resolve_alert_threshold
from market_monitor.pricefetcher import get_base_symbol, is_rth_equity_symbol
from state_io import load_json_state, save_json_state

# Canonical CoinMarketCap slugs for major symbols. _all_listings is keyed by
# symbol, so when several coins share one (for example, real Bitcoin and a scam
# "Bitcoin AI" token both using BTC), the last listing overwrites the correct
# slug. This map forces canonical slugs for watchlist and major coins.
_CANONICAL_CMC_SLUG = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano",
    "XRP": "xrp", "DOGE": "dogecoin", "TAO": "bittensor", "HYPE": "hyperliquid",
    "PEPE": "pepe", "PURR": "purr-2", "WIF": "dogwifhat", "FLR": "flare-networks",
    "ARB": "arbitrum", "ENA": "ethena", "FET": "artificial-superintelligence-alliance",
    "WLD": "worldcoin-org", "RENDER": "render", "SUI": "sui", "AVAX": "avalanche",
    "LINK": "chainlink", "NEAR": "near-protocol", "TRUMP": "official-trump",
    "XMR": "monero", "ZEC": "zcash",
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
        reference_time = self.reference_time
        if not reference_time:
            reference_time = str(self.timestamp)
            if (ts_f := safe_float(self.timestamp, default=None)) is not None:
                try:
                    reference_time = datetime.fromtimestamp(ts_f).strftime("%Y-%m-%d %H:%M:%S")
                except (OSError, OverflowError, ValueError):
                    pass
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
        ts_readable = str(self.timestamp)
        if (ts_f := safe_float(self.timestamp, default=None)) is not None:
            try:
                ts_readable = datetime.fromtimestamp(ts_f).isoformat()
            except (OSError, OverflowError, ValueError):
                pass
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
                 config: Optional[dict] = None, state_file: Optional[str] = None):
        self.cachePriceAll = cachePriceAll
        self.alert_callback = alert_callback or (
            lambda alert: [
                print(f"\n{'=' * 60}\n{item}\n{'=' * 60}")
                for item in (alert if isinstance(alert, (list, tuple)) else [alert])
            ]
        )
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

        raw_sf = state_file if state_file is not None else (self.config.get("state_file") if isinstance(self.config, dict) else None)
        self.state_file = (
            str(raw_sf).strip()
            if isinstance(raw_sf, (str, os.PathLike)) and not isinstance(raw_sf, bool) and str(raw_sf).strip()
            else None
        )

        # Prevent spam: remember the last alert per symbol and alert type
        self._last_alert_time = defaultdict(float)
        if self.state_file and os.path.exists(self.state_file):
            try:
                raw = load_json_state(self.state_file, default_factory=dict, fail_closed=False, label="PriceChecker cooldowns")
                if isinstance(raw, dict):
                    now = time.time()
                    for k, v in raw.items():
                        if isinstance(k, str) and k.strip():
                            ts = safe_nonneg_float(v, default=None)
                            if ts is not None and 0.0 < ts <= now + 60.0:
                                self._last_alert_time[k.strip()] = min(ts, now)
            except Exception:
                pass

        # Thread for continuous analysis
        self._thread = None
        self._running = False

    def _get_price_history_last_hours(self, symbol: str, hours: float):
        """Return compact validated (timestamp_ms, price, readable_time) tuples."""
        hours_f = safe_pos_float(hours, default=None)
        if hours_f is None:
            return []
        eff_hours = min(hours_f, 24.0 * 365.0)
        mins = eff_hours * 60.0
        history_limit = max(1000, min(100_000, math.ceil(mins) + 2))
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
        if (
            recent_history
            and is_rth_gate_enabled(getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open)
            and is_rth_equity_symbol(symbol, getattr(self.cachePriceAll, "price_factory", None))
        ):
            rth_history = [item for item in recent_history if is_us_rth_open(item[0] / 1000.0)]
            if len(rth_history) >= 2:
                return rth_history
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

    def _should_send_alert(self, symbol: str, alert_type: str) -> bool:
        """Check whether we can send a new alert (spam prevention)."""
        key = f"{str(symbol or '').strip().upper()}_{str(alert_type or '').strip().lower()}"
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
        raw_ts = next(
            (v for c in cache_symbol_variants(symbol) if (v := casefold_get(symbol_added_time, c)) is not None),
            None,
        )
        return safe_pos_float(raw_ts, default=None) is not None

    def _get_thresholds_for_symbol(self, symbol: str) -> Dict:
        is_dyn = self._is_dynamic_symbol(symbol)
        fallback = self.config.get("dynamic") if is_dyn else self.config.get("default")
        if not isinstance(fallback, dict):
            fallback = PRICE_ALERT_CONFIG["default"]
        raw = resolve_alert_threshold(self.config if isinstance(self.config, dict) else {}, symbol, is_dyn)
        def_up = safe_pos_float(fallback.get("up_percent"), default=PRICE_ALERT_CONFIG["default"]["up_percent"])
        def_down = safe_pos_float(fallback.get("down_percent"), default=PRICE_ALERT_CONFIG["default"]["down_percent"])
        return {
            "up_percent": safe_pos_float(raw.get("up_percent"), default=def_up),
            "down_percent": safe_pos_float(raw.get("down_percent"), default=def_down),
        }

    def check_symbol(self, symbol: str) -> List[PriceAlert]:
        """Check a symbol and return a list of alerts (0, 1, or 2)."""
        alerts = []
        stats = self._calculate_24h_stats(symbol)

        if not _parse_bool_flag(stats.get("has_data"), False):
            print(f"[Checker][{symbol}] {stats.get('error', 'Unknown error')}")
            return alerts

        current_price = stats["current_price"]
        up_percent = stats["up_from_min"]
        down_percent = stats["down_from_max"]
        thresholds = self._get_thresholds_for_symbol(symbol)
        up_threshold = thresholds["up_percent"]
        down_threshold = thresholds["down_percent"]
        rth_closed_eq = (
            is_rth_gate_enabled(getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open)
            and is_rth_equity_symbol(symbol, getattr(self.cachePriceAll, "price_factory", None))
            and not is_us_rth_open()
        )

        # Check for price increase (upper threshold)
        if not rth_closed_eq and up_percent >= up_threshold:
            self._maybe_append_price_alert(
                alerts,
                symbol,
                "up",
                current_price,
                stats["min_price"],
                stats.get("min_price_timestamp_readable"),
                up_percent,
                up_threshold,
            )

        # Check for price decrease (lower threshold)
        if not rth_closed_eq and down_percent <= -down_threshold:
            self._maybe_append_price_alert(
                alerts,
                symbol,
                "down",
                current_price,
                stats["max_price"],
                stats.get("max_price_timestamp_readable"),
                down_percent,
                down_threshold,
            )

        print(
            f"[Checker][{symbol}] Price: ${current_price:.4f} | "
            f"↑ {up_percent:+.2f}% (threshold +{up_threshold}%) | "
            f"↓ {down_percent:+.2f}% (threshold -{down_threshold}%) | "
            f"Min: ${stats['min_price']:.4f} | Max: ${stats['max_price']:.4f}"
        )

        return alerts

    def _maybe_append_price_alert(
        self,
        alerts: List[PriceAlert],
        symbol: str,
        alert_type: str,
        current_price: float,
        reference_price: float,
        reference_time: object,
        percent_change: float,
        threshold: float,
    ) -> None:
        if self._should_send_alert(symbol, alert_type):
            try:
                cmc_url = f"https://coinmarketcap.com/search/?q={quote(symbol)}"
                base = get_base_symbol(symbol) or symbol
                canonical = _CANONICAL_CMC_SLUG.get((base or "").upper())
                if canonical:
                    cmc_url = f"https://coinmarketcap.com/currencies/{canonical}/"
                else:
                    for candidate in (symbol, base):
                        if not candidate:
                            continue
                        found_slug = False
                        for platform in getattr(getattr(self.cachePriceAll, "price_factory", None), "_platforms", []):
                            if getattr(platform, "platform_name", "") != "CoinMarketCap":
                                continue
                            listings = getattr(platform, "_all_listings", {})
                            metadata = casefold_get(listings, candidate) if isinstance(listings, dict) else None
                            if metadata and metadata.get("slug"):
                                cmc_url = f"https://coinmarketcap.com/currencies/{metadata['slug']}/"
                                found_slug = True
                                break
                        if found_slug:
                            break
            except Exception:
                cmc_url = f"https://coinmarketcap.com/search/?q={quote(symbol)}"
            alerts.append(
                PriceAlert(
                    symbol=symbol,
                    alert_type=alert_type,
                    current_price=current_price,
                    reference_price=reference_price,
                    reference_time=reference_time,
                    percent_change=percent_change,
                    threshold=threshold,
                    url=cmc_url,
                )
            )
            self._last_alert_time[f"{str(symbol or '').strip().upper()}_{str(alert_type or '').strip().lower()}"] = time.time()
            if self.state_file:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
                    save_json_state(self.state_file, dict(self._last_alert_time), indent=2)
                except Exception:
                    pass

    def check_all_symbols(self) -> List[PriceAlert]:
        """Check all symbols in the watchlist."""
        all_alerts = []
        raw_symbols = getattr(self.cachePriceAll, "original_symbols", None)
        if raw_symbols is None:
            raw_symbols = getattr(self.cachePriceAll, "symbols", None)
        monitored = list(raw_symbols) if isinstance(raw_symbols, (list, tuple, set, frozenset)) else []
        for symbol in dict.fromkeys(monitored):
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
                sleep_monitoring_interval(self, iv_f)

        start_monitoring_daemon_thread(self, run, "PriceChecker")

    _MONITOR_LABEL = "Checker"
    stop_monitoring = stop_monitoring_thread
