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
    safe_pos_int,
    sleep_monitoring_interval,
    start_monitoring_daemon_thread,
    stop_monitoring_thread,
)
from cachepaths import cache_symbol_variants
from instrument_registry import casefold_get
from market_monitor.alertsconfig import resolve as resolve_alert_threshold
from market_monitor.priceplatforms import get_base_symbol, is_rth_equity_symbol
from state_stores.price_watcher_state import PriceWatcherStateStore

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
        "up_percent": 4.1,    # Trigger an alert when the price rises by 4.1% from the 24h low
        "down_percent": 7.5,  # Trigger an alert when the price drops by 7.5% from the 24h high
    },
    "dynamic": {
        "up_percent": 12.0,    # Stricter threshold for dynamically added coins
        "down_percent": 25.0,  # Stricter threshold for dynamically added coins
    },
    "lookback_hours": 24,       # Analysis interval (24 hours)
    "cooldown_minutes": 30,     # Notification rate-limit (verification always runs every scan)
    "escalation_step_ratio": 0.35,  # Re-notify during cooldown when move worsens by threshold * ratio
}


class PriceAlert:

    def __init__(self, symbol: str, alert_type: str, current_price: float,
                 reference_price: float, percent_change: float, threshold: float,
                 url: Optional[str] = None, reference_time: Optional[str] = None,
                 escalation_tier: int = 1, previous_percent_change: Optional[float] = None):
        self.symbol = "N/A" if isinstance(symbol, bool) or symbol is None else (str(symbol).strip() or "N/A")
        raw_atype = "down" if isinstance(alert_type, bool) or alert_type is None else (str(alert_type).strip().lower() or "down")
        self.alert_type = "up" if raw_atype in ("up", "rise", "\u25b2") else ("down" if raw_atype in ("down", "drop", "\u25bc") else raw_atype)
        self.current_price = safe_float(current_price, default=0.0)
        self.reference_price = safe_float(reference_price, default=0.0)
        self.percent_change = safe_float(percent_change, default=0.0)
        self.threshold = safe_float(threshold, default=0.0)
        self.timestamp = time.time()
        self.url = "" if isinstance(url, bool) or url is None else str(url)
        self.reference_time = None if isinstance(reference_time, bool) else reference_time
        self.escalation_tier = safe_pos_int(escalation_tier, default=1)
        self.previous_percent_change = safe_float(previous_percent_change, default=None)

    def __str__(self) -> str:
        direction = "\U0001f680 RISE" if self.alert_type == "up" else "\U0001f4c9 DROP"
        emoji = "\U0001f7e2" if self.alert_type == "up" else "\U0001f534"
        reference_time = self.reference_time
        if not reference_time:
            reference_time = str(self.timestamp)
            if (ts_f := safe_float(self.timestamp, default=None)) is not None:
                try:
                    reference_time = datetime.fromtimestamp(ts_f).strftime("%Y-%m-%d %H:%M:%S")
                except (OSError, OverflowError, ValueError):
                    pass
        url_line = f"\U0001f517 CoinMarketCap: {self.url}\n" if self.url else ""
        esc_suffix = ""
        if self.escalation_tier > 1:
            prev_part = (
                f", prev {self.previous_percent_change:+.2f}%"
                if self.previous_percent_change is not None
                else ""
            )
            esc_suffix = f" [ESCALATION #{self.escalation_tier}{prev_part}]"
        return (
            f"\n{emoji} {direction}{esc_suffix} {emoji}\n"
            f"\U0001f4ca Coin: {self.symbol}\n"
            f"\U0001f4b0 Current price: ${self.current_price:.4f}\n"
            f"\U0001f4c8 Reference: ${self.reference_price:.4f} ({'24h low' if self.alert_type == 'up' else '24h high'}, at {reference_time})\n"
            f"{url_line}"
            f"\U0001f4ca Change: {self.percent_change:+.2f}% (threshold: {self.threshold}%)"
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
            "escalation_tier": self.escalation_tier,
            "previous_percent_change": self.previous_percent_change,
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
        # Use pricewatcher.conf configuration, including per_coin, when supplied;
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

        # Notification gate state (strictly for outgoing notifications, never blocks verification).
        # _last_alert_time keeps float timestamps keyed by "<SYMBOL>_<dir>" for backwards compatibility.
        self._last_alert_time = defaultdict(float)
        self._last_alert_magnitude: Dict[str, float] = {}
        self._last_alert_tier: Dict[str, int] = {}
        self._last_logged_snapshot: Dict[str, tuple] = {}
        self._load_notification_state()

        # Thread for continuous analysis
        self._thread = None
        self._running = False

    @staticmethod
    def _alert_key(symbol: str, alert_type: str) -> str:
        return f"{str(symbol or '').strip().upper()}_{str(alert_type or '').strip().lower()}"

    def _load_notification_state(self) -> None:
        if not self.state_file:
            return
        try:
            raw = PriceWatcherStateStore.load_state(self.state_file)
            if not isinstance(raw, dict) or not raw:
                return
            now = time.time()
            for k, v in raw.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                key = k.strip()
                if isinstance(v, dict):
                    ts = safe_nonneg_float(v.get("ts"), default=None)
                    if ts is not None and 0.0 < ts <= now + 60.0:
                        self._last_alert_time[key] = min(ts, now)
                        mag = safe_pos_float(v.get("last_pct"), default=None)
                        if mag is not None:
                            self._last_alert_magnitude[key] = mag
                        self._last_alert_tier[key] = safe_pos_int(v.get("tier"), default=1)
                else:
                    ts = safe_nonneg_float(v, default=None)
                    if ts is not None and 0.0 < ts <= now + 60.0:
                        self._last_alert_time[key] = min(ts, now)
                        self._last_alert_tier[key] = 1
        except Exception:
            pass

    def _save_notification_state(self) -> None:
        if not self.state_file:
            return
        payload = {}
        for key, raw_ts in self._last_alert_time.items():
            ts = safe_pos_float(raw_ts, default=None)
            if ts is None:
                continue
            entry: Dict[str, object] = {
                "ts": ts,
                "tier": safe_pos_int(self._last_alert_tier.get(key), default=1),
            }
            mag = safe_pos_float(self._last_alert_magnitude.get(key), default=None)
            if mag is not None:
                entry["last_pct"] = round(mag, 4)
            payload[key] = entry
        PriceWatcherStateStore.save_state(self.state_file, payload)

    def _effective_sample_interval_sec(self) -> float:
        cache_sync = safe_pos_float(getattr(self.cachePriceAll, "sync_interval_sec", None), default=None)
        cfg_scan = safe_pos_float(self.config.get("price_scan_seconds") if isinstance(self.config, dict) else None, default=None)
        candidates = [v for v in (cache_sync, cfg_scan) if v is not None]
        return max(1.0, min(candidates)) if candidates else 60.0

    def _get_price_history_last_hours(self, symbol: str, hours: float):
        """Return compact validated (timestamp_ms, price, readable_time) tuples."""
        hours_f = safe_pos_float(hours, default=None)
        if hours_f is None:
            return []
        eff_hours = min(hours_f, 24.0 * 365.0)
        sample_sec = self._effective_sample_interval_sec()
        history_limit = max(1000, min(100_000, math.ceil((eff_hours * 3600.0) / sample_sec) + 120))
        history = self.cachePriceAll.get_price_history(symbol, limit=history_limit)
        now_ms = time.time() * 1000.0
        cutoff_time = now_ms - eff_hours * 3600.0 * 1000.0
        max_valid_ms = now_ms + 60_000.0

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
            if cutoff_time <= timestamp <= max_valid_ms:
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

    def _evaluate_notification_gate(
        self,
        symbol: str,
        alert_type: str,
        *,
        percent_change: Optional[float] = None,
        threshold: Optional[float] = None,
        now: Optional[float] = None,
    ) -> tuple[bool, str, int, Optional[float], Optional[float]]:
        """Hybrid notification gate: enforce cooldown_minutes, with worsening escalation bypass.

        Returns ``(allowed, reason, tier, prev_signed_pct, next_escalation_signed_pct)``.
        Strictly gates outgoing notifications; threshold verification always runs freely.
        """
        atype = str(alert_type or "").strip().lower()
        key = self._alert_key(symbol, atype)
        now_ts = time.time() if now is None else safe_float(now, default=time.time())
        lf = safe_nonneg_float(self._last_alert_time.get(key, 0.0), default=None)
        if lf is not None and lf <= now_ts:
            last_time = lf
        else:
            last_time = 0.0
            self._last_alert_time[key] = 0.0
            self._last_alert_magnitude.pop(key, None)
            self._last_alert_tier.pop(key, None)

        cd_min = safe_nonneg_float(self.config.get("cooldown_minutes", 30), default=30.0)
        cooldown_seconds = cd_min * 60.0
        if last_time <= 0.0 or (now_ts - last_time) >= cooldown_seconds:
            return True, "initial", 1, None, None

        ratio = safe_nonneg_float(
            self.config.get("escalation_step_ratio", PRICE_ALERT_CONFIG["escalation_step_ratio"]),
            default=0.0,
        )
        pct_f = safe_float(percent_change, default=None)
        mag = abs(pct_f) if pct_f is not None else None
        thr = safe_pos_float(threshold, default=None)
        cur_tier = safe_pos_int(self._last_alert_tier.get(key), default=1)
        last_mag = safe_pos_float(self._last_alert_magnitude.get(key), default=None)
        baseline_mag = last_mag if last_mag is not None else thr
        sign = -1.0 if atype == "down" else 1.0
        prev_signed = (sign * baseline_mag) if baseline_mag is not None else None
        next_target_mag = (
            (baseline_mag + (thr * ratio))
            if (ratio > 0.0 and thr is not None and baseline_mag is not None)
            else None
        )
        next_signed = (sign * next_target_mag) if next_target_mag is not None else None

        if mag is not None and next_target_mag is not None and mag >= next_target_mag - 1e-9:
            return True, "escalation", cur_tier + 1, prev_signed, next_signed
        return False, "cooldown", cur_tier, prev_signed, next_signed

    def _should_send_alert(
        self,
        symbol: str,
        alert_type: str,
        *,
        percent_change: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check whether an outgoing notification is allowed under the hybrid cooldown policy."""
        allowed, _, _, _, _ = self._evaluate_notification_gate(
            symbol,
            alert_type,
            percent_change=percent_change,
            threshold=threshold,
        )
        return allowed

    def _record_notification_sent(
        self,
        symbol: str,
        alert_type: str,
        percent_change: float,
        tier: int,
        *,
        now: Optional[float] = None,
    ) -> None:
        key = self._alert_key(symbol, alert_type)
        now_ts = time.time() if now is None else safe_float(now, default=time.time())
        self._last_alert_time[key] = now_ts
        pct_f = safe_float(percent_change, default=None)
        if pct_f is not None:
            self._last_alert_magnitude[key] = abs(pct_f)
        self._last_alert_tier[key] = safe_pos_int(tier, default=1)

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

    def evaluate_symbol(self, symbol: str) -> tuple[Dict, List[Dict]]:
        """Pure threshold verification for ``symbol`` (independent of notification cooldown)."""
        stats = self._calculate_24h_stats(symbol)
        if not _parse_bool_flag(stats.get("has_data"), False):
            return stats, []

        current_price = stats["current_price"]
        up_percent = stats["up_from_min"]
        down_percent = stats["down_from_max"]
        thresholds = self._get_thresholds_for_symbol(symbol)
        up_threshold = thresholds["up_percent"]
        down_threshold = thresholds["down_percent"]
        stats["up_threshold"] = up_threshold
        stats["down_threshold"] = down_threshold

        rth_closed_eq = (
            is_rth_gate_enabled(getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open)
            and is_rth_equity_symbol(symbol, getattr(self.cachePriceAll, "price_factory", None))
            and not is_us_rth_open()
        )
        breaches: List[Dict] = []
        if not rth_closed_eq and up_percent >= up_threshold:
            breaches.append(
                {
                    "alert_type": "up",
                    "current_price": current_price,
                    "reference_price": stats["min_price"],
                    "reference_time": stats.get("min_price_timestamp_readable"),
                    "percent_change": up_percent,
                    "threshold": up_threshold,
                }
            )
        if not rth_closed_eq and down_percent <= -down_threshold:
            breaches.append(
                {
                    "alert_type": "down",
                    "current_price": current_price,
                    "reference_price": stats["max_price"],
                    "reference_time": stats.get("max_price_timestamp_readable"),
                    "percent_change": down_percent,
                    "threshold": down_threshold,
                }
            )
        return stats, breaches

    def _resolve_cmc_url(self, symbol: str) -> str:
        try:
            cmc_url = f"https://coinmarketcap.com/search/?q={quote(symbol)}"
            base = get_base_symbol(symbol) or symbol
            canonical = _CANONICAL_CMC_SLUG.get((base or "").upper())
            if canonical:
                return f"https://coinmarketcap.com/currencies/{canonical}/"
            for candidate in (symbol, base):
                if not candidate:
                    continue
                for platform in getattr(getattr(self.cachePriceAll, "price_factory", None), "_platforms", []):
                    if getattr(platform, "platform_name", "") != "CoinMarketCap":
                        continue
                    listings = getattr(platform, "_all_listings", {})
                    metadata = casefold_get(listings, candidate) if isinstance(listings, dict) else None
                    if metadata and metadata.get("slug"):
                        return f"https://coinmarketcap.com/currencies/{metadata['slug']}/"
            return cmc_url
        except Exception:
            return f"https://coinmarketcap.com/search/?q={quote(symbol)}"

    def check_symbol(self, symbol: str) -> List[PriceAlert]:
        """Verify ``symbol`` thresholds and emit only notifications permitted by the hybrid gate."""
        alerts: List[PriceAlert] = []
        stats, breaches = self.evaluate_symbol(symbol)

        logged_map = getattr(self, "_last_logged_snapshot", None)
        if not _parse_bool_flag(stats.get("has_data"), False):
            err_msg = stats.get("error", "Unknown error")
            err_sig = ("no_data", err_msg)
            if logged_map is None or logged_map.get(symbol) != err_sig:
                if logged_map is not None:
                    logged_map[symbol] = err_sig
                print(f"[Checker][{symbol}] {err_msg}")
            return alerts

        suppressed_notes: List[str] = []
        state_dirty = False
        for breach in breaches:
            if self._maybe_append_price_alert(
                alerts,
                symbol,
                breach["alert_type"],
                breach["current_price"],
                breach["reference_price"],
                breach.get("reference_time"),
                breach["percent_change"],
                breach["threshold"],
                suppressed_notes=suppressed_notes,
            ):
                state_dirty = True

        if state_dirty:
            self._save_notification_state()

        current_price = stats["current_price"]
        up_percent = stats["up_from_min"]
        down_percent = stats["down_from_max"]
        up_threshold = stats["up_threshold"]
        down_threshold = stats["down_threshold"]
        snap_sig = (
            round(current_price, 8),
            round(stats["min_price"], 8),
            round(stats["max_price"], 8),
            tuple(suppressed_notes),
        )
        if alerts or logged_map is None or logged_map.get(symbol) != snap_sig:
            if logged_map is not None:
                logged_map[symbol] = snap_sig
            note_suffix = f" | {'; '.join(suppressed_notes)}" if suppressed_notes else ""
            print(
                f"[Checker][{symbol}] Price: ${current_price:.4f} | "
                f"\u2191 {up_percent:+.2f}% (threshold +{up_threshold}%) | "
                f"\u2193 {down_percent:+.2f}% (threshold -{down_threshold}%) | "
                f"Min: ${stats['min_price']:.4f} | Max: ${stats['max_price']:.4f}{note_suffix}"
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
        *,
        suppressed_notes: Optional[List[str]] = None,
    ) -> bool:
        allowed, reason, tier, prev_signed, next_signed = self._evaluate_notification_gate(
            symbol,
            alert_type,
            percent_change=percent_change,
            threshold=threshold,
        )
        if not allowed:
            if suppressed_notes is not None:
                esc_hint = f", next esc {next_signed:+.2f}%" if next_signed is not None else ""
                suppressed_notes.append(f"{alert_type.upper()} breach muted ({reason}{esc_hint})")
            return False

        alerts.append(
            PriceAlert(
                symbol=symbol,
                alert_type=alert_type,
                current_price=current_price,
                reference_price=reference_price,
                reference_time=reference_time,
                percent_change=percent_change,
                threshold=threshold,
                url=self._resolve_cmc_url(symbol),
                escalation_tier=tier,
                previous_percent_change=prev_signed,
            )
        )
        self._record_notification_sent(symbol, alert_type, percent_change, tier)
        return True

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
            print(f"[{self._MONITOR_LABEL}] Already running!")
            return

        self._running = True

        def run():
            print(f"[Checker] price checker started - checking every {interval_seconds}s")
            print(
                f"[Checker] Thresholds: default \u2191 +{self.config['default']['up_percent']}% | \u2193 -{self.config['default']['down_percent']}% | "
                f"dynamic \u2191 +{self.config['dynamic']['up_percent']}% | \u2193 -{self.config['dynamic']['down_percent']}%"
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
