"""Realized volatility estimation from series of prices."""
from __future__ import annotations

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


def calculate_volatility_1h(prices: Sequence[float], sample_rate_sec: float) -> Optional[float]:
    """Estimate one-sigma hourly volatility percentage from scaled log returns."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 20 or sample_rate_sec <= 0:
        return None
    p = p[p > 0]
    if len(p) < 20:
        return None
    rets = np.diff(np.log(p))
    std = float(np.std(rets))
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
