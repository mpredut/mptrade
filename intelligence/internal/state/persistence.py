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

from botcore import lru_dict_put, require_int, safe_float, safe_nonneg_float, safe_pos_float


def _extract_finite_prices(prices: Sequence[float]) -> np.ndarray:
    if prices is None or isinstance(prices, (str, bytes, bytearray, dict, set, frozenset, bool)):
        return np.empty(0, dtype=float)
    if isinstance(prices, np.ndarray):
        if prices.dtype == np.bool_:
            return np.empty(0, dtype=float)
        if prices.dtype.kind in ("f", "i", "u"):
            if prices.ndim == 2 and prices.shape[1] >= 2:
                arr2 = prices[:, :2].astype(float, copy=False)
                ts_col = np.where(arr2[:, 0] > 1e11, arr2[:, 0] / 1000.0, arr2[:, 0])
                if np.all(np.isfinite(ts_col) & (ts_col > 0)) and len(ts_col) > 1 and np.any(np.diff(ts_col) < 0):
                    arr2 = arr2[np.argsort(ts_col, kind="mergesort")]
                arr = arr2[:, 1].ravel()
            else:
                arr = prices.astype(float, copy=False).ravel()
            return arr[np.isfinite(arr)]
    vals = []
    ts_vals = []
    all_have_ts = True
    try:
        for p in prices:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                tf = safe_pos_float(p[0], default=None)
                pf = safe_float(p[1], default=None)
                if pf is not None:
                    vals.append(pf)
                    if tf is not None:
                        ts_vals.append((tf / 1000.0 if tf > 1e11 else tf, pf))
                    else:
                        all_have_ts = False
                continue
            all_have_ts = False
            raw_p = p[0] if isinstance(p, (list, tuple)) and len(p) == 1 else p
            pf = safe_float(raw_p, default=None)
            if pf is not None:
                vals.append(pf)
    except TypeError:
        pass
    if all_have_ts and len(ts_vals) == len(vals) and len(ts_vals) > 1:
        if any(ts_vals[i][0] > ts_vals[i + 1][0] for i in range(len(ts_vals) - 1)):
            ts_vals.sort(key=lambda x: x[0])
        vals = [px for _, px in ts_vals]
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
            v = safe_pos_float(np.var(agg), default=None)
            if v is not None:
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
    return safe_float(slope / 2.0, default=None)


def classify_hurst_regime(h: float | None, lo: float = 0.45, hi: float = 0.55) -> str:
    """Categorize Hurst exponent into descriptive market regimes."""
    h_f = safe_float(h, default=None)
    if h_f is None:
        return "unknown"
    lo_f = safe_float(lo, default=0.45)
    hi_f = safe_float(hi, default=0.55)
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    if h_f > hi_f:
        return "persistent"
    if h_f < lo_f:
        return "mean_reverting"
    return "random_walk"


def _fast_linear_regression_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute OLS linear slope and intercept in O(N) without LAPACK/SVD matrix overhead."""
    dx = x - x[0]
    mean_dx = float(np.mean(dx))
    mean_y = float(np.mean(y))
    dev_x = dx - mean_dx
    var_x = float(np.dot(dev_x, dev_x))
    if var_x == 0.0:
        return 0.0, mean_y
    slope = float(np.dot(dev_x, y) / var_x)
    intercept = mean_y - slope * mean_dx
    return slope, intercept


# Minimum points required for a credible window slope; fewer points constitute a gap.
MIN_POINTS_PER_WINDOW = 4

_trend_memo_cache: dict = {}
_TREND_MEMO_MAX_ENTRIES = 128
_TREND_MEMO_NONE = object()


def _clone_trend_result(value):
    if value is None:
        return None
    if isinstance(value, dict):
        out = dict(value)
        if "blocks" in out and isinstance(out["blocks"], list):
            out["blocks"] = list(out["blocks"])
        return out
    return value


def _store_trend_memo(cache_key: tuple, value):
    lru_dict_put(_trend_memo_cache, cache_key, _TREND_MEMO_NONE if value is None else value, _TREND_MEMO_MAX_ENTRIES)
    return _clone_trend_result(value)


def detect_long_term_trend(
    timestamps,
    prices,
    window_hours=24,
    step_hours=8,
    min_consecutive_blocks=3,
    noise_tolerance=2,
    min_points_per_window=MIN_POINTS_PER_WINDOW,
    detection_lag_hours=0.0,
    mk_alpha=None,
):
    """Detect trends in time-defined windows rather than point-count windows.

    ``timestamps`` are ascending seconds. Each window covers ``[t, t+window_hours]``
    and advances by real ``step_hours``. A window below ``min_points_per_window`` is a
    gap and terminates the trend rather than inventing observations across the gap.

    Return ``{direction, start_timestamp, duration_seconds,
    estimated_future_hours, current_slope_h, blocks(index pairs for plotting)}``
    or ``None``.
    """
    if (
        timestamps is None
        or prices is None
        or isinstance(timestamps, (bool, str, bytes))
        or isinstance(prices, (bool, str, bytes))
    ):
        return None
    if any(
        isinstance(v, bool)
        for v in (
            window_hours,
            step_hours,
            min_consecutive_blocks,
            noise_tolerance,
            min_points_per_window,
            detection_lag_hours,
        )
    ):
        return None
    if mk_alpha is True:
        return None
    if mk_alpha is False or (isinstance(mk_alpha, str) and mk_alpha.strip().lower() in {"", "none", "false", "0", "off", "no"}):
        mk_alpha = None
    elif mk_alpha is not None:
        mk_f = safe_nonneg_float(mk_alpha, default=None)
        if mk_f is None or mk_f > 1.0:
            return None
        mk_alpha = None if mk_f == 0.0 else mk_f
    try:
        wh_f = safe_pos_float(window_hours, default=None)
        sh_f = safe_pos_float(step_hours, default=None)
        mcb_i = require_int(min_consecutive_blocks, error_msg="not integral")
        nt_i = require_int(noise_tolerance, error_msg="not integral")
        mpw_i = require_int(min_points_per_window, error_msg="not integral")
        lag_f = safe_nonneg_float(detection_lag_hours, default=None)
    except (TypeError, ValueError, OverflowError):
        return None
    if wh_f is None or sh_f is None or lag_f is None or mcb_i < 1 or nt_i < 0 or mpw_i < 2:
        return None
    window_hours = wh_f
    step_hours = sh_f
    min_consecutive_blocks = mcb_i
    noise_tolerance = nt_i
    min_points_per_window = mpw_i
    detection_lag_hours = lag_f

    if isinstance(timestamps, np.ndarray) and timestamps.dtype == bool:
        return None
    if isinstance(prices, np.ndarray) and prices.dtype == bool:
        return None
    try:
        if not isinstance(timestamps, np.ndarray) and any(isinstance(x, bool) for x in timestamps):
            raise TypeError("bool element")
        if not isinstance(prices, np.ndarray) and any(isinstance(x, bool) for x in prices):
            raise TypeError("bool element")
        timestamps = np.asarray(timestamps, dtype=float)
        prices = np.asarray(prices, dtype=float)
    except (TypeError, ValueError, OverflowError):
        try:
            raw_ts = list(timestamps)
            raw_px = list(prices)
        except TypeError:
            return None
        if len(raw_ts) != len(raw_px) or len(raw_ts) < 2:
            return None
        clean_ts: list[float] = []
        clean_px: list[float] = []
        for t_item, p_item in zip(raw_ts, raw_px):
            tf = safe_pos_float(t_item, default=None)
            pf = safe_pos_float(p_item, default=None)
            if tf is not None and pf is not None:
                clean_ts.append(tf)
                clean_px.append(pf)
        timestamps = np.asarray(clean_ts, dtype=float)
        prices = np.asarray(clean_px, dtype=float)
    if timestamps.ndim != 1 or prices.ndim != 1 or len(prices) != len(timestamps):
        return None
    finite_mask = np.isfinite(timestamps) & (timestamps > 0) & np.isfinite(prices) & (prices > 0)
    if not np.all(finite_mask):
        timestamps = timestamps[finite_mask]
        prices = prices[finite_mask]
    n_pts = len(timestamps)
    if n_pts < 2:
        return None
    if np.any(timestamps >= 1e11):
        timestamps = np.where(timestamps >= 1e11, timestamps / 1000.0, timestamps)
    if np.any(np.diff(timestamps) < 0):
        order = np.argsort(timestamps, kind="mergesort")
        timestamps = timestamps[order]
        prices = prices[order]

    t_end, t_first = float(timestamps[-1]), float(timestamps[0])
    p_first = float(prices[0])
    p_mid = float(prices[n_pts // 2])
    p_last = float(prices[-1])
    p_penult = float(prices[-2])
    p_sum = round(float(np.sum(prices)), 6)
    t_sum = round(float(np.sum(timestamps)), 3)
    pt_dot = round(float(np.dot(prices, timestamps - t_first)), 3)
    cache_key = (
        n_pts,
        t_first,
        t_end,
        p_first,
        p_mid,
        p_penult,
        p_last,
        p_sum,
        t_sum,
        pt_dot,
        window_hours,
        step_hours,
        min_consecutive_blocks,
        noise_tolerance,
        min_points_per_window,
        detection_lag_hours,
        mk_alpha,
    )
    cached = _trend_memo_cache.get(cache_key)
    if cached is _TREND_MEMO_NONE:
        return None
    if cached is not None:
        return _clone_trend_result(cached)

    window_sec = window_hours * 3600.0
    step_sec = step_hours * 3600.0

    def slope_h(t_lo, t_hi):
        """Return hourly slope and index bounds for ``[t_lo,t_hi)``, or None for a gap."""
        lo = int(np.searchsorted(timestamps, t_lo, "left"))
        hi = int(np.searchsorted(timestamps, t_hi, "left"))
        if hi - lo < min_points_per_window:
            return None, (lo, hi)
        x, y = timestamps[lo:hi], prices[lo:hi]
        s, _ = _fast_linear_regression_1d(x, y)
        if safe_float(s, default=None) is None:
            return None, (lo, hi)
        return s * 3600.0, (lo, hi)

    cur, cur_idx = slope_h(t_end - window_sec, t_end + 1.0)
    if cur is None or cur == 0.0:
        return _store_trend_memo(cache_key, None)
    if mk_alpha:
        _, _, p_mk = calculate_mann_kendall(prices[cur_idx[0]:cur_idx[1]])
        if p_mk > mk_alpha:
            return _store_trend_memo(cache_key, None)
    current_sign = float(np.sign(cur))

    blocks = [cur_idx]
    consecutive, noise = 1, 0
    confirm_lo = cur_idx[0]
    confirm_pos = 0
    t_ws = t_end - window_sec - step_sec
    while t_ws >= t_first:
        s, idx = slope_h(t_ws, t_ws + window_sec)
        if s is None:
            break
        if np.sign(s) == current_sign:
            if noise > 0 and (float(prices[cur_idx[1] - 1]) - float(prices[idx[0]])) * current_sign <= 0.0:
                break
            blocks.append(idx)
            consecutive += 1
            noise = 0
            confirm_lo = idx[0]
            confirm_pos = len(blocks) - 1
        elif noise < noise_tolerance:
            noise += 1
            blocks.append(idx)
        else:
            break
        t_ws -= step_sec

    if consecutive < min_consecutive_blocks:
        return _store_trend_memo(cache_key, None)

    blocks = blocks[:confirm_pos + 1]
    confirmed_start = float(timestamps[confirm_lo])
    duration_seconds = (t_end - confirmed_start) + detection_lag_hours * 3600.0
    duration_seconds = min(duration_seconds, t_end - t_first)
    trend_start_ts = t_end - duration_seconds
    if duration_seconds <= 0:
        return _store_trend_memo(cache_key, None)
    res = {
        "direction": "up" if current_sign > 0 else "down",
        "start_timestamp": float(trend_start_ts),
        "duration_seconds": float(duration_seconds),
        "estimated_future_hours": float(duration_seconds / 3600.0 * 0.5),
        "current_slope_h": float(cur),
        "blocks": list(blocks),
    }
    return _store_trend_memo(cache_key, res)

