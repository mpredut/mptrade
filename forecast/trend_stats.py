#!/usr/bin/env python3
"""Compatibility facade for trend statistics.

Canonical implementation has moved to intelligence.internal.state.persistence.
This facade preserves backwards compatibility for legacy callers (priceAnalysis.py, forecast.py, tests).
"""

from __future__ import annotations

from intelligence.internal.state.persistence import (
    calculate_mann_kendall as mann_kendall,
    calculate_hurst_exponent as hurst_rs,
    classify_hurst_regime,
)


def hurst_regime(h: float | None, lo: float = 0.45, hi: float = 0.55) -> str:
    """Classify Hurst exponent into legacy regime strings."""
    regime = classify_hurst_regime(h, lo=lo, hi=hi)
    if regime == "unknown":
        return "necunoscut"
    if regime == "mean_reverting":
        return "mean-reverting"
    if regime == "random_walk":
        return "random-walk"
    return regime


__all__ = ["mann_kendall", "hurst_rs", "hurst_regime"]
