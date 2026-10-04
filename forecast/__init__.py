"""Backwards-compatibility facade for trend statistics and empirical survival analysis.

Canonical implementations have moved to:
- intelligence.internal.state.persistence (Mann-Kendall, Hurst)
- intelligence.internal.state.survival (Weibull survival distributions, estimate_T, hybrid_T)
Experimental ML forecast scripts are located in:
- offline/research/ml_forecast/ (priceprediction, forecast, vol_chronos)
"""
from __future__ import annotations

from intelligence.internal.state.persistence import (
    calculate_mann_kendall as mann_kendall,
    calculate_hurst_exponent as hurst_rs,
)
from intelligence.internal.state.survival import (
    estimate_T,
    hybrid_T,
    fetch_klines,
    get_trend_survival_metrics,
)

__all__ = [
    "mann_kendall",
    "hurst_rs",
    "estimate_T",
    "hybrid_T",
    "fetch_klines",
    "get_trend_survival_metrics",
]
