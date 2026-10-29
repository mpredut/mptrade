"""Realized volatility estimation from series of prices."""
from __future__ import annotations

from collections import deque
import json
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from botcore import float_env, safe_float, safe_nonneg_float, safe_pos_float


def _f_env(name: str, default: float) -> float:
    return safe_float(float_env(name), default)


K_REENTRY_DEFAULT = _f_env("SHADOW_K_REENTRY", 2.0)
K_DCA_DEFAULT = _f_env("SHADOW_K_DCA", 1.0)

_vol_memo_cache: dict = {}
_VOL_MEMO_MAX_ENTRIES = 128


def calculate_volatility_1h(prices: Sequence[float], sample_rate_sec: float) -> Optional[float]:
    """Estimate one-sigma hourly volatility percentage from scaled log returns."""
    if prices is None or isinstance(prices, (bool, str, bytes)):
        return None
    sr_f = safe_pos_float(sample_rate_sec, default=None)
    if sr_f is None:
        return None
    try:
        n = len(prices)
    except TypeError:
        return None
    if n < 20:
        return None
    sample_rate_sec = sr_f

    p0 = None
    pn = None
    if not isinstance(prices[0], (bool, np.bool_)) and not isinstance(prices[-1], (bool, np.bool_)):
        cand0 = safe_pos_float(prices[0], default=None)
        candn = safe_pos_float(prices[-1], default=None)
        if cand0 is not None and candn is not None:
            p0 = cand0
            pn = candn

    tbuf = getattr(prices, "_trend_buf", None)
    if tbuf is not None and p0 is not None and pn is not None:
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

    cache_key = (
        (n, p0, pn, round(float(sample_rate_sec), 3))
        if (p0 is not None and pn is not None)
        else None
    )
    if cache_key is not None:
        cached = _vol_memo_cache.get(cache_key)
        if cached is not None:
            return cached

    if isinstance(prices, np.ndarray):
        if prices.dtype == bool or prices.ndim != 1:
            return None
        try:
            p = prices.astype(float, copy=False)
        except (TypeError, ValueError):
            return None
    else:
        try:
            if any(isinstance(x, (bool, np.bool_)) for x in prices):
                raise TypeError("bool in prices")
            p = np.asarray(prices, dtype=float)
        except (TypeError, ValueError):
            clean_p = [
                xf for x in prices
                if not isinstance(x, (bool, np.bool_))
                and (xf := safe_pos_float(x, default=None)) is not None
            ]
            p = np.asarray(clean_p, dtype=float)
            cache_key = None
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 20:
        return None
    if len(p) != n:
        cache_key = None

    # Single-ratio logarithm: ln(p[i+1] / p[i]) instead of 2 * log calls
    rets = np.log(p[1:] / p[:-1])
    std = float(np.std(rets))
    if std == 0.0:
        res = 0.0
    else:
        res = round(std * math.sqrt(3600.0 / sample_rate_sec) * 100.0, 4)

    if cache_key is not None:
        if len(_vol_memo_cache) > _VOL_MEMO_MAX_ENTRIES:
            _vol_memo_cache.clear()
        _vol_memo_cache[cache_key] = res
    return res


class RollingVolatilityTracker:
    """O(1) incremental realized volatility tracker using Welford's algorithm over a sliding window."""

    def __init__(self, window_size: int = 100) -> None:
        wf = safe_pos_float(window_size, default=None)
        ws = max(20, int(wf)) if wf is not None else 100
        self.window_size = ws
        self._prices: deque[float] = deque(maxlen=ws + 1)
        self._rets: deque[float] = deque(maxlen=ws)
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add_price(self, price: float, sample_rate_sec: float) -> Optional[float]:
        """Add a price tick and return updated 1h volatility in O(1)."""
        price = safe_pos_float(price, default=None)
        if price is None:
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

        sr_f = safe_pos_float(sample_rate_sec, default=None)
        if self._count < 20 or sr_f is None:
            return None

        variance = self._m2 / self._count
        std = math.sqrt(variance)
        if std == 0.0:
            return 0.0
        return round(std * math.sqrt(3600.0 / sr_f) * 100.0, 4)


def adaptive_thresholds(
    vol1h: Optional[float],
    k_reentry: float = K_REENTRY_DEFAULT,
    k_dca: float = K_DCA_DEFAULT,
) -> Tuple[Optional[float], Optional[float]]:
    """Return adaptive reentry and DCA percentages as multipliers of hourly volatility."""
    vol_f = safe_nonneg_float(vol1h, default=None)
    kr_f = safe_nonneg_float(k_reentry, default=None)
    kd_f = safe_nonneg_float(k_dca, default=None)
    if vol_f is None or kr_f is None or kd_f is None:
        return None, None
    return round(kr_f * vol_f, 3), round(kd_f * vol_f, 3)


# Direct backwards-compatible alias
vol_1h_pct = calculate_volatility_1h


def vol_pct_to_ratio(vol_pct: float) -> float:
    """Convert an hourly volatility percentage to decimal ratio (e.g. 2.5% -> 0.025, 0.05% -> 0.0005)."""
    vf = safe_pos_float(vol_pct, default=None)
    return (vf / 100.0) if vf is not None else 0.0


def vol_ratio_to_pct(vol_ratio: float) -> float:
    """Convert an hourly volatility decimal ratio to percentage (e.g. 0.025 -> 2.5%, 0.0005 -> 0.05%)."""
    vf = safe_pos_float(vol_ratio, default=None)
    return (vf * 100.0) if vf is not None else 0.0


def get_kalman_state_path() -> str:
    """Return the canonical path to cache_kalman_trend.json honoring MPTRADE_CACHEDB_DIR."""
    from cachepaths import cache_path
    return cache_path("cache_kalman_trend.json")


def _vol_symbol_variants(symbol: str) -> Tuple[str, ...]:
    if not isinstance(symbol, str):
        return ()
    raw = symbol.strip().upper()
    if not raw:
        return ()
    clean = raw.replace("/", "").replace("-", "").replace("_", "")
    if clean == "USDCUSDT":
        return ("USDCUSDT",)
    from cachepaths import cache_symbol_variants, normalize_cache_base_asset
    base = normalize_cache_base_asset(raw)
    variants = [raw]
    for cand in (*cache_symbol_variants(raw, include_base=True), f"{base}/USDC" if base else ""):
        if cand and cand not in variants:
            variants.append(cand)
    return tuple(variants)


_VOL_FILE_CACHE: dict = {}


def _extract_vol_ratio_from_file(path: str, symbol: str) -> Optional[float]:
    if not path or not isinstance(path, str) or not isinstance(symbol, str):
        return None
    try:
        from state_io import read_recovering_previous
        data = None
        stat_sig = None
        try:
            st = os.stat(path)
            stat_sig = (st.st_mtime_ns, st.st_size, id(read_recovering_previous))
        except OSError:
            stat_sig = None
        if stat_sig is not None:
            cached_entry = _VOL_FILE_CACHE.get(path)
            if cached_entry is not None and cached_entry[0] == stat_sig:
                data = cached_entry[1]
        if data is None:
            data = read_recovering_previous(path, json.loads, default=None)
            if stat_sig is not None and isinstance(data, dict):
                if len(_VOL_FILE_CACHE) > 32:
                    _VOL_FILE_CACHE.clear()
                _VOL_FILE_CACHE[path] = (stat_sig, data)
        if isinstance(data, dict):
            search_maps = [data]
            items_map = data.get("items")
            if isinstance(items_map, dict):
                search_maps.append(items_map)
            variants = _vol_symbol_variants(symbol)
            for mapping in search_maps:
                for cand in variants:
                    entry = mapping.get(cand)
                    if not isinstance(entry, dict):
                        continue
                    vf = safe_pos_float(entry.get("vol_1h_pct"), default=None)
                    if vf is not None:
                        return vol_pct_to_ratio(vf)
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

