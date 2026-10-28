"""Market persistence and statistical trend significance analysis.

Extracts and adapts non-parametric trend verification:
- Mann-Kendall test: determines whether price motion is a statistically significant trend (p < alpha) vs Brownian noise.
- Hurst Exponent (H): classifies whether market dynamics are persistent/trending (H > 0.55),
  mean-reverting (H < 0.45), or random-walk (H ~ 0.5).
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def _extract_finite_prices(prices: Sequence[float]) -> np.ndarray:
    if prices is None or isinstance(prices, (str, bytes, bool)):
        return np.empty(0, dtype=float)
    if isinstance(prices, np.ndarray):
        if prices.dtype == np.bool_:
            return np.empty(0, dtype=float)
        if prices.dtype.kind in ("f", "i", "u"):
            arr = prices.astype(float, copy=False).ravel()
            return arr[np.isfinite(arr)]
    vals = []
    try:
        for p in prices:
            if p is not None and not isinstance(p, bool):
                try:
                    pf = float(p)
                    if math.isfinite(pf):
                        vals.append(pf)
                except (TypeError, ValueError, OverflowError):
                    pass
    except TypeError:
        pass
    return np.asarray(vals, dtype=float)


def calculate_mann_kendall(prices: Sequence[float]) -> tuple[int, float, float]:
    """Calculate Mann-Kendall test statistic S, normalized Z score, and two-sided p-value.
    
    A small p-value (e.g. p < 0.05) rejects the null hypothesis of no trend (pure noise).
    Returns (S, Z, p-value). Sequences with fewer than 8 observations return (0, 0.0, 1.0).
    """
    y = _extract_finite_prices(prices)
    n = len(y)
    if n < 8:
        return 0, 0.0, 1.0
        
    s = 0.0
    for k in range(n - 1):
        s += np.sign(y[k + 1:] - y[k]).sum()
        
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - (counts * (counts - 1) * (2 * counts + 5)).sum()) / 18.0
    if var <= 0:
        return int(s), 0.0, 1.0
        
    z = (s - np.sign(s)) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return int(s), float(z), float(p)


def calculate_hurst_exponent(prices: Sequence[float]) -> float | None:
    """Estimate Hurst exponent H using aggregated log-returns variance.
    
    Var(sum of k returns) ~ k^(2H). H is half the log-log slope.
    - H > 0.55: Persistent / trend-following regime.
    - H < 0.45: Anti-persistent / mean-reverting regime.
    - 0.45 <= H <= 0.55: Geometric Brownian motion (random walk).
    
    Returns None if series has fewer than 65 points or non-positive values.
    """
    y = _extract_finite_prices(prices)
    if len(y) < 65 or np.any(y <= 0):
        return None
        
    r = np.diff(np.log(y))
    n = len(r)
    ks, vs = [], []
    k = 1
    while k <= n // 8:
        m = (n // k) * k
        agg = r[:m].reshape(-1, k).sum(axis=1)
        if len(agg) >= 8:
            v = float(np.var(agg))
            if v > 0 and math.isfinite(v):
                ks.append(k)
                vs.append(v)
        k *= 2
        
    if len(ks) < 3:
        return None

    lx = np.log(ks)
    ly = np.log(vs)
    dev_x = lx - float(np.mean(lx))
    var_x = float(np.dot(dev_x, dev_x))
    if var_x <= 0.0:
        return None
    slope = float(np.dot(dev_x, ly) / var_x)
    h_val = float(slope / 2.0)
    return h_val if math.isfinite(h_val) else None


def classify_hurst_regime(h: float | None, lo: float = 0.45, hi: float = 0.55) -> str:
    """Categorize Hurst exponent into descriptive market regimes."""
    if h is None or isinstance(h, bool):
        return "unknown"
    try:
        h_f = float(h)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(h_f):
        return "unknown"
    try:
        lo_f = float(lo) if lo is not None and not isinstance(lo, bool) else 0.45
        if not math.isfinite(lo_f):
            lo_f = 0.45
    except (TypeError, ValueError):
        lo_f = 0.45
    try:
        hi_f = float(hi) if hi is not None and not isinstance(hi, bool) else 0.55
        if not math.isfinite(hi_f):
            hi_f = 0.55
    except (TypeError, ValueError):
        hi_f = 0.55
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    if h_f > hi_f:
        return "persistent"
    if h_f < lo_f:
        return "mean_reverting"
    return "random_walk"
