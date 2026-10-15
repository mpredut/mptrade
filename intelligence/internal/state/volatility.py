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


def resolve_cached_volatility_1h(
    symbol: str, state_path: Optional[str] = None
) -> Optional[float]:
    """Retrieve 1h realized volatility for symbol from shared shadow cache as a decimal ratio.

    Producers (e.g. shadow_signals / calculate_volatility_1h) output percentages (e.g. 2.5 for 2.5%,
    0.05 for 0.05%). This function acts as the Single Source of Truth (SSOT) to explicitly convert
    the percentage to a decimal ratio by dividing by 100.0.
    Returns None if cache is missing, symbol is not found, or value is non-positive/non-finite.
    """
    if not state_path:
        try:
            from cachepaths import cache_path
            state_path = cache_path("shadow_state.json")
        except Exception:
            state_path = os.path.join(
                os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb"), "shadow_state.json"
            )

    if not os.path.exists(state_path):
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            sym_key = str(symbol).upper()
            entry = (
                data.get(sym_key)
                or data.get(sym_key.replace("USDC", "USDT"))
                or data.get(sym_key.replace("USDT", "USDC"))
            )
            if entry and isinstance(entry, dict):
                v = entry.get("vol_1h_pct")
                if v is not None and math.isfinite(float(v)) and float(v) > 0:
                    return vol_pct_to_ratio(float(v))
    except Exception:
        pass
    return None

