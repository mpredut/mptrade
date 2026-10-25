"""Realized volatility estimation from series of prices."""
from __future__ import annotations

from collections import deque
import json
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np


def _f_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default
    except ValueError:
        return default


K_REENTRY_DEFAULT = _f_env("SHADOW_K_REENTRY", 2.0)
K_DCA_DEFAULT = _f_env("SHADOW_K_DCA", 1.0)

_vol_memo_cache: dict = {}
_VOL_MEMO_MAX_ENTRIES = 128


def calculate_volatility_1h(prices: Sequence[float], sample_rate_sec: float) -> Optional[float]:
    """Estimate one-sigma hourly volatility percentage from scaled log returns."""
    n = len(prices)
    if n < 20 or sample_rate_sec <= 0:
        return None

    p0 = float(prices[0])
    pn = float(prices[-1])
    tbuf = getattr(prices, "_trend_buf", None)
    if tbuf is not None:
        blen = len(tbuf.prices)
        if (
            blen >= n
            and tbuf.prices[blen - n] == p0
            and tbuf.prices[-1] == pn
        ):
            std = tbuf.log_return_std(blen - n, blen - 1)
            if std is not None:
                if std == 0.0:
                    return 0.0
                return round(std * math.sqrt(3600.0 / sample_rate_sec) * 100.0, 4)

    cache_key = (n, p0, pn, round(float(sample_rate_sec), 3))
    cached = _vol_memo_cache.get(cache_key)
    if cached is not None:
        return cached

    p = np.asarray(prices, dtype=float)
    p = p[p > 0]
    if len(p) < 20:
        return None

    # Single-ratio logarithm: ln(p[i+1] / p[i]) instead of 2 * log calls
    rets = np.log(p[1:] / p[:-1])
    std = float(np.std(rets))
    if std == 0.0:
        res = 0.0
    else:
        res = round(std * math.sqrt(3600.0 / sample_rate_sec) * 100.0, 4)

    if len(_vol_memo_cache) > _VOL_MEMO_MAX_ENTRIES:
        _vol_memo_cache.clear()
    _vol_memo_cache[cache_key] = res
    return res


class RollingVolatilityTracker:
    """O(1) incremental realized volatility tracker using Welford's algorithm over a sliding window."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._prices: deque[float] = deque(maxlen=window_size + 1)
        self._rets: deque[float] = deque(maxlen=window_size)
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add_price(self, price: float, sample_rate_sec: float) -> Optional[float]:
        """Add a price tick and return updated 1h volatility in O(1)."""
        if price <= 0:
            return None
        if len(self._prices) > 0:
            prev = self._prices[-1]
            if prev > 0:
                ret = math.log(price / prev)
                if len(self._rets) == self.window_size:
                    old_ret = self._rets[0]
                    self._count -= 1
                    delta = old_ret - self._mean
                    self._mean -= delta / self._count if self._count > 0 else 0.0
                    delta2 = old_ret - self._mean
                    self._m2 -= delta * delta2
                    self._m2 = max(0.0, self._m2)

                self._rets.append(ret)
                self._count += 1
                delta = ret - self._mean
                self._mean += delta / self._count
                delta2 = ret - self._mean
                self._m2 += delta * delta2

        self._prices.append(price)

        if self._count < 20 or sample_rate_sec <= 0:
            return None

        variance = self._m2 / self._count
        std = math.sqrt(variance)
        if std == 0.0:
            return 0.0
        return round(std * math.sqrt(3600.0 / sample_rate_sec) * 100.0, 4)


def adaptive_thresholds(
    vol1h: Optional[float],
    k_reentry: float = K_REENTRY_DEFAULT,
    k_dca: float = K_DCA_DEFAULT,
) -> Tuple[Optional[float], Optional[float]]:
    """Return adaptive reentry and DCA percentages as multipliers of hourly volatility."""
    if vol1h is None:
        return None, None
    return round(k_reentry * vol1h, 3), round(k_dca * vol1h, 3)


# Direct backwards-compatible alias
vol_1h_pct = calculate_volatility_1h


def vol_pct_to_ratio(vol_pct: float) -> float:
    """Convert an hourly volatility percentage to decimal ratio (e.g. 2.5% -> 0.025, 0.05% -> 0.0005)."""
    return float(vol_pct) / 100.0


def vol_ratio_to_pct(vol_ratio: float) -> float:
    """Convert an hourly volatility decimal ratio to percentage (e.g. 0.025 -> 2.5%, 0.0005 -> 0.05%)."""
    return float(vol_ratio) * 100.0


def get_kalman_state_path() -> str:
    """Return the canonical path to kalman_state.json honoring MPTRADE_CACHEDB_DIR."""
    try:
        from cachepaths import cache_path
        return cache_path("kalman_state.json")
    except Exception:
        return os.path.join(
            os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb"), "kalman_state.json"
        )


def _vol_symbol_variants(symbol: str) -> Tuple[str, ...]:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ()
    clean = raw.replace("/", "").replace("-", "").replace("_", "")
    if clean == "USDCUSDT":
        return ("USDCUSDT",)
    base = clean
    for suffix in ("USDC", "USDT", "FDUSD", "EUR", "RON", "GBP", "USD"):
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)]
            break
    variants = [raw]
    for cand in (clean, f"{base}USDC", f"{base}USDT", f"{base}USD", f"{base}/USDC", base):
        if cand and cand not in variants:
            variants.append(cand)
    return tuple(variants)


def _extract_vol_ratio_from_file(path: str, symbol: str) -> Optional[float]:
    if not path:
        return None
    try:
        from state_io import read_recovering_previous
        data = read_recovering_previous(path, json.loads, default=None)
        if isinstance(data, dict):
            search_maps = [data]
            items_map = data.get("items")
            if isinstance(items_map, dict):
                search_maps.append(items_map)
            entry = None
            for mapping in search_maps:
                for cand in _vol_symbol_variants(symbol):
                    if cand in mapping and isinstance(mapping[cand], dict):
                        entry = mapping[cand]
                        break
                if entry is not None:
                    break
            if entry and isinstance(entry, dict):
                v = entry.get("vol_1h_pct")
                if v is not None and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0:
                    return vol_pct_to_ratio(float(v))
    except Exception:
        pass
    return None


def resolve_cached_volatility_1h(
    symbol: str, state_path: Optional[str] = None
) -> Optional[float]:
    """Retrieve 1h realized volatility for symbol from shared Kalman or trend cache as a decimal ratio.

    Producers (e.g. shadow_signals / cacheManager / calculate_volatility_1h) output percentages
    (e.g. 2.5 for 2.5%, 0.05 for 0.05%). This function acts as the Single Source of Truth (SSOT)
    to explicitly convert the percentage to a decimal ratio by dividing by 100.0.
    Returns None if cache is missing, symbol is not found, or value is non-positive/non-finite.
    """
    if state_path is not None:
        return _extract_vol_ratio_from_file(state_path, symbol)

    kalman_path = get_kalman_state_path()
    res = _extract_vol_ratio_from_file(kalman_path, symbol)
    if res is not None:
        return res

    trend_path = os.path.join(os.path.dirname(kalman_path), "cache_short_trend.json")
    return _extract_vol_ratio_from_file(trend_path, symbol)


def resolve_cached_volatility_1h_pct(
    symbol: str, state_path: Optional[str] = None
) -> Optional[float]:
    """Retrieve 1h realized volatility for symbol from shared shadow cache as a percentage (e.g. 2.5 for 2.5%)."""
    ratio = resolve_cached_volatility_1h(symbol, state_path=state_path)
    if ratio is not None:
        return vol_ratio_to_pct(ratio)
    return None

