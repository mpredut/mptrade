#!/usr/bin/env python3
"""Compatibility facade for trend survival.

Canonical implementation has moved to intelligence.internal.state.survival.
This facade preserves backwards compatibility for existing callers.
"""

from __future__ import annotations

import sys
from intelligence.internal.state.survival import (
    fetch_klines,
    block_slopes,
    episodes,
    survival_report,
    verdict,
    hybrid_T,
    estimate_T,
    get_trend_survival_metrics,
    main,
    T_CACHE_FILE,
)

__all__ = [
    "fetch_klines",
    "block_slopes",
    "episodes",
    "survival_report",
    "verdict",
    "hybrid_T",
    "estimate_T",
    "get_trend_survival_metrics",
    "T_CACHE_FILE",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
