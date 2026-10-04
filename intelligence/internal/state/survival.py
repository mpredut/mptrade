"""Bridge to trend survival models and empirical distribution metrics."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_T_CACHE_FILE = os.path.join(_ROOT, "cachedb", "cache_T_trend.json")


def get_trend_survival_metrics(symbol: str, trend_duration_seconds: float = 0.0) -> Dict[str, Any]:
    """Retrieve or compute empirical trend survival metrics for a symbol.

    Returns dictionary with:
      - median_days: Median expected trend length
      - p90_days: 90th percentile of trend length (empirical lifetime boundary)
      - duration_days: Current active duration
      - duration_ratio_to_p90: Current duration / P90
      - is_exhausted: True if duration > P90
    """
    duration_days = max(0.0, float(trend_duration_seconds) / 86400.0)
    median_d = 3.0
    p90_d = 7.0
    T_val = 8.0

    # First attempt reading from local cache
    cached_entry: Optional[dict] = None
    if os.path.exists(_T_CACHE_FILE):
        try:
            with open(_T_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                cached_entry = data.get(symbol)
        except Exception:
            cached_entry = None

    if cached_entry:
        median_d = float(cached_entry.get("median_d") or median_d)
        p90_d = float(cached_entry.get("p90_d") or p90_d)
        T_val = float(cached_entry.get("T") or T_val)
    else:
        # Fallback to estimate_T if available
        try:
            from forecast.trend_survival import estimate_T
            est = estimate_T(symbol)
            if est:
                median_d = float(est.get("median_d") or median_d)
                p90_d = float(est.get("p90_d") or p90_d)
                T_val = float(est.get("T") or T_val)
        except Exception:
            pass

    ratio = duration_days / p90_d if p90_d > 0 else 0.0
    is_exhausted = duration_days > p90_d if duration_days > 0 else False

    return {
        "symbol": symbol,
        "duration_days": round(duration_days, 2),
        "median_days": round(median_d, 2),
        "p90_days": round(p90_d, 2),
        "T": round(T_val, 1),
        "duration_ratio_to_p90": round(ratio, 2),
        "is_exhausted": is_exhausted,
    }
