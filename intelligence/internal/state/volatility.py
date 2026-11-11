"""Realized volatility estimation and cached volatility resolution."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from botcore import safe_pos_float
from cachepaths import (
    CACHE_KALMAN_TREND_FILENAME,
    CACHE_SHORT_TREND_FILENAME,
    cache_path,
    cache_symbol_variants,
    canonical_symbol_key,
    resolve_cachedb_filepath,
)
from instrument_registry import casefold_get
from pricewindow import (
    K_DCA_DEFAULT,
    K_REENTRY_DEFAULT,
    _vol_memo_cache,
    adaptive_thresholds,
    calculate_volatility_1h,
    vol_pct_to_ratio,
    vol_ratio_to_pct,
)
from state_io import read_json_dict_cached


def get_kalman_state_path() -> str:
    """Return the canonical path to cache_kalman_trend.json honoring MPTRADE_CACHEDB_DIR."""
    return cache_path(CACHE_KALMAN_TREND_FILENAME)


def _vol_symbol_variants(symbol: str) -> Tuple[str, ...]:
    if not isinstance(symbol, str) or not symbol.strip():
        return ()
    if canonical_symbol_key(symbol) == "USDCUSDT":
        return ("USDCUSDT",)
    return tuple(cache_symbol_variants(symbol, include_base=True))


_VOL_FILE_CACHE: dict = {}


def _extract_vol_ratio_from_file(path: str, symbol: str) -> Optional[float]:
    if not path or not isinstance(path, str) or not isinstance(symbol, str):
        return None
    try:
        data = read_json_dict_cached(path, _VOL_FILE_CACHE, max_entries=32)
        if isinstance(data, dict):
            search_maps = [data]
            items_map = data.get("items")
            if isinstance(items_map, dict):
                search_maps.append(items_map)
            variants = _vol_symbol_variants(symbol)
            for mapping in search_maps:
                for cand in variants:
                    entry = casefold_get(mapping, cand)
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

    trend_path = str(resolve_cachedb_filepath(os.path.join(os.path.dirname(kalman_path), CACHE_SHORT_TREND_FILENAME)))
    return _extract_vol_ratio_from_file(trend_path, symbol)


def resolve_cached_volatility_1h_pct(
    symbol: str, state_path: Optional[str] = None
) -> Optional[float]:
    """Retrieve 1h realized volatility for symbol from shared shadow cache as a percentage (e.g. 2.5 for 2.5%)."""
    ratio = resolve_cached_volatility_1h(symbol, state_path=state_path)
    if ratio is not None:
        return vol_ratio_to_pct(ratio)
    return None

