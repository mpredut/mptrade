"""
Price-window primitives and trend analysis.

Extracted from ``tradeall`` for reuse without trading decisions.

Contains ``PriceTrendAnalyzer`` regression/EMA/gradient calculations, a lean sliding
``PriceWindow`` with volatility-informed noise epsilon, and composed trading metrics.

Short-trend SSOT (``growth_coefficient`` / ``final_trend``) is computed by the
estimator selected in ``SHORT_TREND_ESTIMATOR`` (default: adaptive causal median
filter followed by Step-WOLS cubic; see the constants block below).
"""
import builtins
import math
import os
import threading
from collections import deque
from bisect import insort, bisect_left

import numpy as np
from scipy.stats import linregress

from botcore import float_env
import utils as u

# ── Window constants ──────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE_SEC = 0.8     # Nominal fallback sampling rate.
RECENT_GRADIENT_SECONDS = 60.0    # Recent-momentum window in seconds (for primary 222s window).
WINDOW_SECONDS_SMALL = 3.7 * 60          # 3.7 minutes (222.0s)
WINDOW_SECONDS_BIG   = 2.5 * 60 * 60     # 2.5 hours (9000.0s)

# ── Short-trend Estimator Registry (SSOT for growth_coefficient / final_trend) ──
# Unified pipeline: [raw price] -> [adaptive causal median filter ~5s, bypassed for
# sparse data] -> [estimator slope calculation].
# Default: Step-WOLS Quadratic (p=2, med=5.0s) — empirical #1 champion on 11.26M 1.0s ticks
# (+8.04 net vs old baseline, +$1.75 over cubic p=3, zero regressions).
# Configurable via MPTRADE_STEP_WOLS_POWER (default 2.0; accepts any p >= 0).
# Configurable via MPTRADE_MEDIAN_FILTER_SECONDS (default 5.0; 0 = bypass).
#
# Extensibility for future short-trend models:
#   1. Add model name constant to SHORT_TREND_ESTIMATOR_*
#   2. Register in _SHORT_TREND_ESTIMATORS
#   3. Add dispatch logic in compute_short_trend_slope() and IncrementalTrendBuffer
SHORT_TREND_ESTIMATOR_STEP_WOLS = "step_wols"
SHORT_TREND_ESTIMATOR_STEP_WOLS_QUADRATIC = "step_wols_quadratic"
SHORT_TREND_ESTIMATOR_STEP_WOLS_CUBIC = "step_wols_cubic"
SHORT_TREND_ESTIMATOR_STEP_WOLS_LINEAR = "step_wols_linear"
_SHORT_TREND_ESTIMATORS = (
    SHORT_TREND_ESTIMATOR_STEP_WOLS,
    SHORT_TREND_ESTIMATOR_STEP_WOLS_QUADRATIC,
    SHORT_TREND_ESTIMATOR_STEP_WOLS_CUBIC,
    SHORT_TREND_ESTIMATOR_STEP_WOLS_LINEAR,
)

DEFAULT_STEP_WOLS_POWER = 2.0         # Quadratic: empirical champion on 11.26M real 1s ticks (+8.04 net)
DEFAULT_MEDIAN_FILTER_SECONDS = 5.0   # Adaptive causal median filter window in seconds (k=5 at 1s)


def _resolve_short_trend_estimator() -> str:
    raw = str(os.environ.get("MPTRADE_SHORT_TREND_ESTIMATOR", "") or "").strip().lower()
    if raw in _SHORT_TREND_ESTIMATORS:
        return raw
    return SHORT_TREND_ESTIMATOR_STEP_WOLS


SHORT_TREND_ESTIMATOR = _resolve_short_trend_estimator()


def _env_float(name: str, default: float) -> float:
    val = float_env(name)
    return default if val is None else val


def _resolve_step_wols_power(estimator_name: str = None) -> float:
    if "MPTRADE_STEP_WOLS_POWER" in os.environ:
        return max(0.0, _env_float("MPTRADE_STEP_WOLS_POWER", DEFAULT_STEP_WOLS_POWER))
    est = (estimator_name or SHORT_TREND_ESTIMATOR or "").lower()
    if est == SHORT_TREND_ESTIMATOR_STEP_WOLS_CUBIC:
        return 3.0
    if est == SHORT_TREND_ESTIMATOR_STEP_WOLS_LINEAR:
        return 1.0
    return DEFAULT_STEP_WOLS_POWER


MEDIAN_FILTER_SECONDS = max(0.0, _env_float("MPTRADE_MEDIAN_FILTER_SECONDS", DEFAULT_MEDIAN_FILTER_SECONDS))
STEP_WOLS_POWER = _resolve_step_wols_power(SHORT_TREND_ESTIMATOR)
DEFAULT_TREND_NOISE_GATE_K = 0.0      # Multiplier k for adaptive noise gate (0.0 = bypassed, e.g. 0.05-0.10)
TREND_NOISE_GATE_K = max(0.0, _env_float("MPTRADE_TREND_NOISE_GATE_K", DEFAULT_TREND_NOISE_GATE_K))
MEDIAN_FILTER_BYPASS_SAMPLE_RATE_SEC = 2.0  # Sparser data: no 1s bid-ask bounce to remove.
_MEDIAN_KERNELS = (1, 3, 5)
_STEP_WOLS_DIRECT_MAX_STEPS = 1024          # Exact direct dot product up to this many steps.


def _active_estimator() -> str:
    est = SHORT_TREND_ESTIMATOR
    if est in _SHORT_TREND_ESTIMATORS:
        return est
    return SHORT_TREND_ESTIMATOR_STEP_WOLS


def median_filter_kernel(sample_rate_sec, median_seconds: float = None) -> int:
    """Return the causal median kernel length (1, 3 or 5 samples) for a sampling rate.

    The filter is defined in seconds (~5s), not samples: 5 samples at the live
    0.8-1.0s cadence, 3 samples up to 2s, and bypassed (1) for sparser data where a
    multi-sample median would only add lag.
    """
    sr = DEFAULT_SAMPLE_RATE_SEC
    if sample_rate_sec is not None and not isinstance(sample_rate_sec, bool):
        try:
            cand = float(sample_rate_sec)
            if math.isfinite(cand) and cand > 0:
                sr = cand
        except (TypeError, ValueError, OverflowError):
            pass
    med_sec = MEDIAN_FILTER_SECONDS if median_seconds is None else max(0.0, float(median_seconds))
    if med_sec <= 0.0 or sr > MEDIAN_FILTER_BYPASS_SAMPLE_RATE_SEC:
        return 1
    ratio = med_sec / sr
    if not math.isfinite(ratio) or ratio >= 5.0:
        return 5
    if ratio < 1.5:
        return 1
    k = int(round(ratio))
    if k % 2 == 0:
        k -= 1
    return max(3, min(5, k))


def _small_median(vals) -> float:
    if len(vals) == 0:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def causal_median_filter(prices, kernel: int) -> np.ndarray:
    """Slice-local causal median: out[i] = median(prices[max(0, i-k+1) .. i])."""
    arr = np.asarray(prices, dtype=np.float64)
    if kernel <= 1 or len(arr) == 0:
        return arr.copy()
    out = np.empty_like(arr)
    head = min(kernel - 1, len(arr))
    for i in range(head):
        out[i] = _small_median(arr[: i + 1])
    if len(arr) >= kernel:
        from numpy.lib.stride_tricks import sliding_window_view
        out[kernel - 1:] = np.median(sliding_window_view(arr, kernel), axis=-1)
    return out


def step_wols_slope(prices, sample_rate_sec=None, power=None, median_seconds=None) -> float:
    """Generalized Step-WOLS slope: sum(k^p * dy_k) / sum(k^p) on median-filtered prices.

    Args:
        prices: Iterable of float prices.
        sample_rate_sec: Effective sampling interval in seconds (default 0.8s).
        power: Exponent p (w_k = k^p). Defaults to STEP_WOLS_POWER (2.0 = quadratic).
        median_seconds: Duration of causal median filter. Defaults to MEDIAN_FILTER_SECONDS (5.0s).
    """
    pwr = float(STEP_WOLS_POWER if power is None else power)
    med_sec = float(MEDIAN_FILTER_SECONDS if median_seconds is None else median_seconds)
    filtered = causal_median_filter(prices, median_filter_kernel(sample_rate_sec, median_seconds=med_sec))
    m = len(filtered) - 1
    if m < 1:
        return 0.0
    k = np.arange(1, m + 1, dtype=np.float64)
    w = k ** pwr if pwr != 1.0 else k
    val = float(np.dot(w, np.diff(filtered)) / np.sum(w))
    tol = 1e-10 * max(1.0, abs(float(filtered[-1])))
    return 0.0 if (not math.isfinite(val) or abs(val) <= tol) else val


def step_wols_cubic_slope(prices, sample_rate_sec=None) -> float:
    """Backward-compatible alias for step_wols_slope."""
    return step_wols_slope(prices, sample_rate_sec=sample_rate_sec)


def effective_recent_gradient_seconds(window_seconds=None) -> float:
    """Return the recent-gradient sub-window duration in seconds.

    For the primary window (<= 222s) or unspecified windows, returns
    ``RECENT_GRADIENT_SECONDS`` (60s). For larger windows (e.g. 900s, 9000s),
    scales proportionally by ``RECENT_GRADIENT_SECONDS / WINDOW_SECONDS_SMALL``
    (60 / 222 ≈ 27.03% of the window) so multi-minute/hour windows do not
    collapse to a 60-second tail.
    """
    if window_seconds is not None and not isinstance(window_seconds, bool):
        try:
            win_f = float(window_seconds)
            if math.isfinite(win_f) and win_f > WINDOW_SECONDS_SMALL:
                return win_f * (RECENT_GRADIENT_SECONDS / WINDOW_SECONDS_SMALL)
        except (TypeError, ValueError, OverflowError):
            pass
    return RECENT_GRADIENT_SECONDS


class PriceTrendAnalyzer:
    def __init__(self, prices):
        clean = []
        if prices is not None and not isinstance(prices, (bool, str, bytes)):
            try:
                for p in prices:
                    if p is None or isinstance(p, bool):
                        continue
                    try:
                        pf = float(p)
                        if math.isfinite(pf):
                            clean.append(pf)
                    except (TypeError, ValueError, OverflowError):
                        continue
            except TypeError:
                pass
        self.prices = clean

    def linear_regression_trend(self):
        n = len(self.prices)
        if n < 2:
            print("Linear regression: not enough data to compute the trend.")
            return None, None, None

        x = np.arange(n)
        y = np.asarray(self.prices, dtype=float)

        mean_y = float(np.mean(y))
        dev_y = y - mean_y
        ss_y = float(np.dot(dev_y, dev_y))
        if ss_y == 0.0:
            print("Linear regression: prices are constant, the trend cannot be determined.")
            return None, None, None

        mean_x = (n - 1) / 2.0
        var_x = n * (n * n - 1) / 12.0
        dev_x = x - mean_x
        cov_xy = float(np.dot(dev_x, y))
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x
        r_value = cov_xy / np.sqrt(var_x * ss_y) if (var_x * ss_y) > 0 else 0.0
        trend_line = slope * x + intercept
        return trend_line, slope, r_value

    def calculate_gradient(self):
        if len(self.prices) < 2:
            print("Gradient: not enough data to compute the gradient.")
            return [], 0
        y = np.array(self.prices)
        gradient = np.gradient(y)
        avg_gradient = np.mean(gradient)
        return gradient, avg_gradient


def _safe_recent_n(recent_sec: float, sample_rate_sec) -> int:
    if sample_rate_sec is None or isinstance(sample_rate_sec, bool):
        return 2
    try:
        sr = float(sample_rate_sec)
        if not math.isfinite(sr) or sr <= 0:
            return 2
        ratio = float(recent_sec) / sr
        if not math.isfinite(ratio):
            return 2
        return max(2, int(ratio))
    except (TypeError, ValueError, OverflowError):
        return 2


def instant_trend_from_prices(prices, sample_rate_sec, window_seconds=None):
    """Apply the exact ``PriceWindow.get_short_trend`` formula directly to a
    price list and sampling rate without constructing a complete window. This avoids
    incremental deque bookkeeping for a one-shot raw slice and is the single source of
    the formula extracted from ``PriceWindow.get_short_trend`` and
    offline/research/monitortrades_backtest/replay_trend_source._instant_trend_from_slice)
    All ad-hoc dynamic-cache and replay windows delegate here.

    Return ``(final_trend, growth_coefficient, slope_full, gradient_recent)``."""
    analyzer = PriceTrendAnalyzer(prices)
    if len(analyzer.prices) < 2:
        return 0, 0.0, 0.0, 0.0
    _, slope_full, _ = analyzer.linear_regression_trend()
    if slope_full is None:
        slope_full = 0.0
    gradient_lst, _ = analyzer.calculate_gradient()
    recent_sec = effective_recent_gradient_seconds(window_seconds)
    recent_n = _safe_recent_n(recent_sec, sample_rate_sec)
    if len(gradient_lst) >= recent_n:
        gradient_recent = float(np.mean(gradient_lst[-recent_n:]))
    else:
        gradient_recent = float(np.mean(gradient_lst)) if len(gradient_lst) else 0.0
    if abs(gradient_recent) <= 1e-12:
        gradient_recent = 0.0
    growth_coefficient = step_wols_slope(analyzer.prices, sample_rate_sec)
    noise_gate_k = max(0.0, _env_float("MPTRADE_TREND_NOISE_GATE_K", TREND_NOISE_GATE_K))
    noise_thr = 0.0
    if noise_gate_k > 0.0 and len(analyzer.prices) >= 3:
        grad_arr = np.array(gradient_lst) if len(gradient_lst) else np.gradient(np.array(analyzer.prices))
        noise_thr = noise_gate_k * float(np.std(grad_arr))
    if noise_thr > 0.0 and abs(growth_coefficient) <= noise_thr:
        final_trend = 0
    else:
        final_trend = 1 if growth_coefficient > noise_thr else (-1 if growth_coefficient < -noise_thr else 0)
    return final_trend, growth_coefficient, slope_full, gradient_recent


class IncrementalTrendBuffer:
    """Maintain O(1) cumulative prefix statistics over [ts_ms, price] series.

    Supports O(1) per-tick append, O(1) amortized front-trimming, and O(1) exact
    evaluation of linear regression slope, recent telescoped gradient, growth
    coefficient, and gradient noise epsilon on any sub-slice [a, b].
    """

    def __init__(self):
        self.timestamps = []
        self.prices = []
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.pg2 = []
        self.plr2 = []
        self.nonpos = []
        self.offset = 0
        self.base_p0 = 0.0
        self.base_p1 = 0.0
        self.base_p2 = 0.0
        self.y_ref = None
        # Step-WOLS cubic support: causal medians (kernels 3/5) and, per kernel
        # (1 = raw), prefix sums of z, J*z, J^2*z with z = filtered - y_ref, J global index.
        self.med = {3: [], 5: []}
        self.fq = {k: ([], [], []) for k in _MEDIAN_KERNELS}
        self.fq_base = {k: [0.0, 0.0, 0.0] for k in _MEDIAN_KERNELS}

    def clear(self):
        self.timestamps.clear()
        self.prices.clear()
        self.p0.clear()
        self.p1.clear()
        self.p2.clear()
        self.pg2.clear()
        self.plr2.clear()
        self.nonpos.clear()
        self.offset = 0
        self.base_p0 = self.base_p1 = self.base_p2 = 0.0
        self.y_ref = None
        for lst in self.med.values():
            lst.clear()
        for k in _MEDIAN_KERNELS:
            for lst in self.fq[k]:
                lst.clear()
            self.fq_base[k] = [0.0, 0.0, 0.0]

    def append(self, price: float, ts_ms: int = 0) -> None:
        if price is None or isinstance(price, bool) or isinstance(ts_ms, bool):
            return
        try:
            y = float(price)
            ts_int = int(ts_ms)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(y):
            return
        if self.y_ref is None:
            self.y_ref = y
        i = len(self.prices)
        k = self.offset + i
        dy = y - self.y_ref
        self.timestamps.append(ts_int)
        self.prices.append(y)
        prev0 = self.p0[-1] if i > 0 else self.base_p0
        prev1 = self.p1[-1] if i > 0 else self.base_p1
        prev2 = self.p2[-1] if i > 0 else self.base_p2
        self.p0.append(prev0 + dy)
        self.p1.append(prev1 + k * dy)
        self.p2.append(prev2 + dy * dy)
        if i >= 2:
            g_prev = 0.5 * (self.prices[i] - self.prices[i - 2])
            prev_pg2 = self.pg2[i - 2] + g_prev * g_prev
            self.pg2[i - 1] = prev_pg2
            self.pg2.append(prev_pg2)
        else:
            self.pg2.append(self.pg2[-1] if i > 0 else 0.0)
        if i >= 1 and self.prices[i - 1] > 0.0 and y > 0.0:
            lr = math.log(y / self.prices[i - 1])
            self.plr2.append(self.plr2[-1] + lr * lr)
        else:
            self.plr2.append(self.plr2[-1] if i > 0 else 0.0)
        prev_np = self.nonpos[-1] if i > 0 else 0
        self.nonpos.append(prev_np + (1 if y <= 0.0 else 0))
        for kern in _MEDIAN_KERNELS:
            if kern == 1:
                fval = y
            else:
                fval = _small_median(self.prices[max(0, i - kern + 1): i + 1])
                self.med[kern].append(fval)
            z = fval - self.y_ref
            q0, q1, q2 = self.fq[kern]
            if i > 0:
                b0, b1, b2 = q0[-1], q1[-1], q2[-1]
            else:
                b0, b1, b2 = self.fq_base[kern]
            q0.append(b0 + z)
            q1.append(b1 + k * z)
            q2.append(b2 + float(k) * float(k) * z)

    def _filtered(self, kern: int, j: int) -> float:
        return self.prices[j] if kern == 1 else self.med[kern][j]

    def _fq_range(self, kern: int, lo: int, hi: int):
        """Return (sum z, sum J*z, sum J^2*z) over buffer indices [lo, hi]."""
        q0, q1, q2 = self.fq[kern]
        if lo > 0:
            b0, b1, b2 = q0[lo - 1], q1[lo - 1], q2[lo - 1]
        else:
            b0, b1, b2 = self.fq_base[kern]
        return q0[hi] - b0, q1[hi] - b1, q2[hi] - b2

    def step_wols_slope(self, a: int, b: int, sample_rate_sec=None, power=None) -> float:
        """Step-WOLS slope on [a, b] over the slice-local causal median with configurable power.

        Equals ``step_wols_slope(prices[a:b+1], sample_rate_sec, power)``.
        Slices up to ``_STEP_WOLS_DIRECT_MAX_STEPS`` steps (1024) use exact direct
        vectorized dot product on the stored medians; longer slices use O(1) Abel
        summation on prefix sums for p=2 (Quadratic) or p=3 (Cubic) with bounded
        floating-point precision error.
        """
        m = b - a
        if m < 1:
            return 0.0
        kern = median_filter_kernel(sample_rate_sec)
        p = self.prices
        n_local = min(kern - 1, m + 1) if kern > 1 else 0
        local_head = [_small_median(p[a: a + r + 1]) for r in range(n_local)]
        pwr = float(STEP_WOLS_POWER if power is None else power)

        if m <= _STEP_WOLS_DIRECT_MAX_STEPS or pwr not in (2.0, 3.0):
            src = p if kern == 1 else self.med[kern]
            f = np.asarray(src[a: b + 1], dtype=np.float64)
            if n_local:
                f[:n_local] = local_head
            w = np.arange(1, m + 1, dtype=np.float64) ** pwr if pwr != 1.0 else np.arange(1, m + 1, dtype=np.float64)
            val = float(np.dot(w, np.diff(f)) / np.sum(w))
            ref_last = float(f[-1])
        elif pwr == 2.0:
            # O(1) Quadratic Abel summation:
            # sum_{k=1..m} k^2 (f_{a+k} - f_{a+k-1}) = m^2 f_b - sum_{r=0..m-1} (2r + 1) f_{a+r}
            # with sum_{r=0..m-1} (2r + 1) z_{a+r} = 2 a1 - (2c - 1) a0
            y_ref = self.y_ref if self.y_ref is not None else 0.0
            c = float(self.offset + a)
            a0, a1, _ = self._fq_range(kern, a, b - 1)
            sum_2r1 = 2.0 * a1 - (2.0 * c - 1.0) * a0
            m2 = float(m) ** 2
            total = m2 * (self._filtered(kern, b) - y_ref) - sum_2r1
            for r in range(n_local):
                delta = local_head[r] - self._filtered(kern, a + r)
                if delta:
                    total -= (2.0 * r + 1.0) * delta
            s2 = (float(m) * float(m + 1) * float(2 * m + 1)) / 6.0
            val = float(total / s2)
            ref_last = self._filtered(kern, b)
        else:
            # O(1) Cubic Abel summation (p=3):
            y_ref = self.y_ref if self.y_ref is not None else 0.0
            c = float(self.offset + a)
            a0, a1, a2 = self._fq_range(kern, a, b - 1)
            sum_r2 = a2 - 2.0 * c * a1 + c * c * a0
            sum_r1 = a1 - c * a0
            m3 = float(m) ** 3
            total = m3 * (self._filtered(kern, b) - y_ref) - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
            for r in range(n_local):
                delta = local_head[r] - self._filtered(kern, a + r)
                if delta:
                    total -= (3.0 * r * r + 3.0 * r + 1.0) * delta
            s3 = (0.5 * m * (m + 1)) ** 2
            val = float(total / s3)
            ref_last = self._filtered(kern, b)

        tol = 1e-10 * max(1.0, abs(ref_last))
        return 0.0 if (not math.isfinite(val) or abs(val) <= tol) else val

    def step_wols_cubic(self, a: int, b: int, sample_rate_sec=None) -> float:
        """Backward-compatible alias for step_wols_slope on [a, b]."""
        return self.step_wols_slope(a, b, sample_rate_sec=sample_rate_sec)

    def _rebase(self) -> None:
        if not self.prices:
            self.clear()
            return
        saved_px = list(self.prices)
        saved_ts = list(self.timestamps)
        self.clear()
        for px, ts in zip(saved_px, saved_ts):
            self.append(px, ts)

    def trim_front(self, count: int) -> None:
        if count is None or isinstance(count, bool):
            return
        try:
            cf = float(count)
            if not math.isfinite(cf) or cf <= 0:
                return
            count_int = int(cf)
        except (TypeError, ValueError, OverflowError):
            return
        if count_int <= 0:
            return
        idx = min(count_int, len(self.prices))
        if idx <= 0:
            return
        self.base_p0 = self.p0[idx - 1]
        self.base_p1 = self.p1[idx - 1]
        self.base_p2 = self.p2[idx - 1]
        for kern in _MEDIAN_KERNELS:
            q0, q1, q2 = self.fq[kern]
            self.fq_base[kern] = [q0[idx - 1], q1[idx - 1], q2[idx - 1]]
            del q0[:idx]
            del q1[:idx]
            del q2[:idx]
        for lst in self.med.values():
            del lst[:idx]
        del self.timestamps[:idx]
        del self.prices[:idx]
        del self.p0[:idx]
        del self.p1[:idx]
        del self.p2[:idx]
        del self.pg2[:idx]
        del self.plr2[:idx]
        del self.nonpos[:idx]
        self.offset += idx
        if not self.prices:
            self.clear()
        elif self.offset >= max(65536, 16 * len(self.prices)):
            self._rebase()

    def sync_entries(self, entries) -> None:
        """Synchronize incrementally with an ordered list of [ts_ms, price] entries."""
        if not entries:
            self.clear()
            return
        if not self.timestamps:
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.append(e[1], e[0])
            return
        try:
            first_ts, first_p = int(entries[0][0]), float(entries[0][1])
        except (TypeError, ValueError, IndexError, OverflowError):
            self.clear()
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.append(e[1], e[0])
            return
        if self.timestamps[0] < first_ts:
            trim_idx = bisect_left(self.timestamps, first_ts)
            if trim_idx > 0:
                self.trim_front(trim_idx)
        if (
            not self.timestamps
            or self.timestamps[0] != first_ts
            or self.prices[0] != first_p
            or len(entries) < len(self.timestamps)
        ):
            self.clear()
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.append(e[1], e[0])
            return
        cur_len = len(self.timestamps)
        try:
            last_ts, last_p = int(entries[cur_len - 1][0]), float(entries[cur_len - 1][1])
        except (TypeError, ValueError, IndexError, OverflowError):
            self.clear()
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.append(e[1], e[0])
            return
        if self.timestamps[-1] != last_ts or self.prices[-1] != last_p:
            self.clear()
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.append(e[1], e[0])
            return
        for idx in range(cur_len, len(entries)):
            e = entries[idx]
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                self.append(e[1], e[0])

    def eval_slice(self, a: int, b: int, sample_rate_sec: float = DEFAULT_SAMPLE_RATE_SEC,
                   eps_k: float = 1.0, window_seconds: float | None = None):
        """Evaluate (final_trend, growth_coefficient, slope_full, gradient_recent, epsilon)
        on inclusive index range [a, b] in O(1) using cumulative prefix sums."""
        if isinstance(a, bool) or isinstance(b, bool):
            return 0, 0.0, 0.0, 0.0, 0.0
        try:
            a = int(a)
            b = int(b)
        except (TypeError, ValueError, OverflowError):
            return 0, 0.0, 0.0, 0.0, 0.0
        n = b - a + 1
        if a < 0 or b >= len(self.prices) or n < 2:
            return 0, 0.0, 0.0, 0.0, 0.0
        k_val = 1.0
        if eps_k is not None and not isinstance(eps_k, bool):
            try:
                kf = float(eps_k)
                if math.isfinite(kf) and kf > 0.0:
                    k_val = kf
            except (TypeError, ValueError, OverflowError):
                k_val = 1.0
        p = self.prices
        s0 = self.p0[b] - (self.p0[a - 1] if a > 0 else self.base_p0)
        s1_raw = self.p1[b] - (self.p1[a - 1] if a > 0 else self.base_p1)
        s2 = self.p2[b] - (self.p2[a - 1] if a > 0 else self.base_p2)
        ss_y = s2 - (s0 * s0) / n
        if ss_y <= 1e-18 or (ss_y <= 1e-12 and p[a] == p[b] and p[(a + b) // 2] == p[a]):
            slope_full = 0.0
        else:
            cov_xy = s1_raw - (self.offset + a + 0.5 * (n - 1)) * s0
            slope_full = float((12.0 * cov_xy) / (n * (n * n - 1)))

        recent_sec = effective_recent_gradient_seconds(window_seconds)
        recent_n = _safe_recent_n(recent_sec, sample_rate_sec)
        sum_g_all = (p[b] - p[a]) + 0.5 * ((p[b] - p[b - 1]) + (p[a + 1] - p[a]))
        if n > recent_n:
            r = recent_n
            gradient_recent = float(
                ((p[b] - p[b - 1]) + (p[b] - p[b - r + 1]) + (p[b] - p[b - r])) / (2.0 * r)
            )
        else:
            gradient_recent = float(sum_g_all / n)
        if abs(gradient_recent) <= 1e-12:
            gradient_recent = 0.0

        growth_coefficient = self.step_wols_slope(a, b, sample_rate_sec)

        if n < 3:
            epsilon = 0.0
        else:
            g0 = p[a + 1] - p[a]
            gn = p[b] - p[b - 1]
            sum_g2 = g0 * g0 + (self.pg2[b - 1] - self.pg2[a]) + gn * gn
            mean_g = sum_g_all / n
            var_g = max(0.0, sum_g2 / n - mean_g * mean_g)
            epsilon = float(k_val * math.sqrt(var_g))

        noise_gate_k = max(0.0, _env_float("MPTRADE_TREND_NOISE_GATE_K", TREND_NOISE_GATE_K))
        noise_thr = noise_gate_k * epsilon if noise_gate_k > 0.0 else 0.0
        if noise_thr > 0.0 and abs(growth_coefficient) <= noise_thr:
            final_trend = 0
        else:
            final_trend = 1 if growth_coefficient > noise_thr else (-1 if growth_coefficient < -noise_thr else 0)

        return final_trend, growth_coefficient, slope_full, gradient_recent, epsilon

    def log_return_std(self, a: int, b: int) -> float | None:
        """Return O(1) population std of log returns on slice [a, b] if all prices > 0."""
        if isinstance(a, bool) or isinstance(b, bool):
            return None
        try:
            a = int(a)
            b = int(b)
        except (TypeError, ValueError, OverflowError):
            return None
        n = b - a + 1
        if a < 0 or b >= len(self.prices) or n < 20:
            return None
        np_before = self.nonpos[a - 1] if a > 0 else (self.nonpos[0] - (1 if self.prices[0] <= 0.0 else 0))
        if (self.nonpos[b] - np_before) > 0:
            return None
        m = n - 1
        mean_lr = math.log(self.prices[b] / self.prices[a]) / m
        sum_lr2 = self.plr2[b] - self.plr2[a]
        var_lr = max(0.0, sum_lr2 / m - mean_lr * mean_lr)
        return math.sqrt(var_lr)


class _PriceDeque(deque):
    __slots__ = ("_trend_buf", "_maxlen")

    def __init__(self, iterable=(), maxlen=None):
        super().__init__(iterable)
        self._maxlen = maxlen
        if maxlen is not None and len(self) > maxlen:
            for _ in range(len(self) - maxlen):
                self.popleft()

    @property
    def maxlen(self):
        return self._maxlen


class PriceWindow:
    def __init__(self, symbol, window_size, sample_rate_sec=None, initial_prices=None,
                 window_seconds=None):
        self.symbol = symbol
        ws_int = 10
        if window_size is not None and not isinstance(window_size, bool):
            try:
                wf = float(window_size)
                if math.isfinite(wf):
                    ws_int = max(1, int(wf))
            except (TypeError, ValueError, OverflowError):
                pass
        self.window_size = ws_int
        if sample_rate_sec is not None and not isinstance(sample_rate_sec, bool):
            try:
                _sr = float(sample_rate_sec)
                self.sample_rate_sec = _sr if math.isfinite(_sr) and _sr > 0 else DEFAULT_SAMPLE_RATE_SEC
            except (TypeError, ValueError, OverflowError):
                self.sample_rate_sec = DEFAULT_SAMPLE_RATE_SEC
        else:
            self.sample_rate_sec = DEFAULT_SAMPLE_RATE_SEC
        # When set, target duration resizes with the observed sampling rate so the
        # window continues to cover the same elapsed time.
        win_sec = None
        if window_seconds is not None and not isinstance(window_seconds, bool):
            try:
                wsec_f = float(window_seconds)
                if math.isfinite(wsec_f) and wsec_f > 0:
                    win_sec = wsec_f
            except (TypeError, ValueError, OverflowError):
                win_sec = None
        self.window_seconds = win_sec
        self._subscribed_to_cache24 = False
        # WS/Cache24 updates on a different thread from evaluation. Protect mutation
        # and iteration of both price collections.
        self._lock = threading.RLock()
        self._trend_buf = IncrementalTrendBuffer()
        self.prices = _PriceDeque(maxlen=self.window_size)
        self.prices._trend_buf = self._trend_buf
        self._sorted_prices_cache = []
        self._sorted_prices_dirty = False
        self._cached_eval_key = None
        self._cached_eval_val = None

        if initial_prices:
            for price in initial_prices[-self.window_size:]:
                self.process_price(price)

    @property
    def sorted_prices(self):
        """Lazily materialize sorted prices only when min/max queries require them."""
        with self._lock:
            if (
                self._sorted_prices_dirty
                or self._sorted_prices_cache is None
                or len(self._sorted_prices_cache) != len(self.prices)
            ):
                self._sorted_prices_cache = sorted(self.prices)
                self._sorted_prices_dirty = False
            return self._sorted_prices_cache

    @sorted_prices.setter
    def sorted_prices(self, value):
        with self._lock:
            self._sorted_prices_cache = list(value) if value is not None else []
            self._sorted_prices_dirty = False

    @property
    def recent_n(self) -> int:
        """Return the sample count corresponding to the effective recent-gradient window."""
        return _safe_recent_n(
            effective_recent_gradient_seconds(self.window_seconds),
            self.sample_rate_sec,
        )

    def set_sample_rate(self, rate):
        """Update observed sampling rate and resize a duration-based window."""
        if rate is None or isinstance(rate, bool):
            return
        try:
            rate_val = float(rate)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(rate_val) or rate_val <= 0:
            return
        self.sample_rate_sec = rate_val
        if (
            self.window_seconds is None
            or isinstance(self.window_seconds, bool)
            or not math.isfinite(float(self.window_seconds))
            or float(self.window_seconds) <= 0
        ):
            return
        new_size = max(10, int(float(self.window_seconds) / rate_val))
        if new_size == self.window_size:
            return
        with self._lock:
            if new_size != self.window_size:
                self.window_size = new_size
                self.prices._maxlen = new_size
                if len(self.prices) > new_size:
                    drop = len(self.prices) - new_size
                    for _ in range(drop):
                        self.prices.popleft()
                    self._sorted_prices_dirty = True
                    if len(self._trend_buf.prices) - new_size >= max(512, new_size):
                        self._trend_buf.trim_front(len(self._trend_buf.prices) - new_size)

    def on_price_update(self, symbol: str, ts_ms: int, price: float) -> None:
        """Update the window from a ``Cache24PriceManager`` callback."""
        if symbol != self.symbol:
            return
        self.process_price(price)

    @classmethod
    def from_existing_window(cls, existing_prices, window_size, symbol=""):
        return cls(symbol, window_size, initial_prices=existing_prices)

    @staticmethod
    def _sample_rate_from_entries(entries) -> float:
        """Return the median sampling gap in seconds for [ts_ms, price] entries."""
        if not entries or isinstance(entries, (bool, str, bytes)):
            return DEFAULT_SAMPLE_RATE_SEC
        timestamps_sec = []
        try:
            for e in entries:
                if not isinstance(e, (list, tuple)) or len(e) < 1:
                    continue
                ts_raw = e[0]
                if ts_raw is None or isinstance(ts_raw, bool):
                    continue
                try:
                    ts_f = float(ts_raw) / 1000.0
                    if math.isfinite(ts_f):
                        timestamps_sec.append(ts_f)
                except (TypeError, ValueError, OverflowError):
                    continue
        except TypeError:
            return DEFAULT_SAMPLE_RATE_SEC
        if len(timestamps_sec) < 2:
            return DEFAULT_SAMPLE_RATE_SEC
        gaps = [timestamps_sec[i+1] - timestamps_sec[i]
                for i in range(len(timestamps_sec) - 1)
                if timestamps_sec[i+1] > timestamps_sec[i]]
        if not gaps:
            return DEFAULT_SAMPLE_RATE_SEC
        med = float(np.median(gaps))
        return med if math.isfinite(med) and med > 0 else DEFAULT_SAMPLE_RATE_SEC

    @classmethod
    def from_cache24(cls, symbol: str, window_seconds: float, cache24) -> "PriceWindow":
        """Build a window from recent Cache24 data and subscribe to future updates."""
        win_sec = WINDOW_SECONDS_SMALL
        if window_seconds is not None and not isinstance(window_seconds, bool):
            try:
                wf = float(window_seconds)
                if math.isfinite(wf) and wf > 0:
                    win_sec = wf
            except (TypeError, ValueError, OverflowError):
                win_sec = WINDOW_SECONDS_SMALL
        entries = cache24.get_recent_entries(symbol, last_seconds=win_sec)

        sample_rate = cls._sample_rate_from_entries(entries)
        window_size = max(10, int(win_sec / sample_rate))

        prices = [e[1] for e in entries if isinstance(e, (list, tuple)) and len(e) >= 2]
        pw = cls(symbol, window_size, sample_rate_sec=sample_rate, window_seconds=win_sec)
        for p in prices:
            pw.process_price(p)

        pw.subscribe_to_cache24(cache24)
        return pw

    def subscribe_to_cache24(self, cache24) -> None:
        cache24.subscribe_price(self)
        self._subscribed_to_cache24 = True

    def unsubscribe_from_cache24(self, cache24) -> None:
        cache24.unsubscribe_price(self)
        self._subscribed_to_cache24 = False

    def process_price(self, price):
        #print(f"{self.symbol}: {price}")
        if price is None or isinstance(price, bool):
            return
        try:
            price_f = float(price)
            if not math.isfinite(price_f):
                return
        except (TypeError, ValueError, OverflowError):
            return
        if not isinstance(price, (int, float)):
            price = price_f
        with self._lock:
            while len(self.prices) >= self.window_size:
                self.prices.popleft()

            self.prices.append(price)
            self._sorted_prices_dirty = True
            self._trend_buf.append(price)
            if len(self._trend_buf.prices) - len(self.prices) >= max(512, self.window_size):
                self._trend_buf.trim_front(len(self._trend_buf.prices) - len(self.prices))

    def get_newest_index(self):
        return len(self.prices) - 1 if self.prices else None

    def get_min(self):
        with self._lock:
            sp = self.sorted_prices
            if not sp:
                return None
            min_price = sp[0]
            if len(sp) == 1:
                return min_price
            if min_price > 0.0:
                p1 = sp[1]
                if p1 != min_price and (p1 - min_price) > (p1 + min_price) * 0.00005:
                    return min_price
                total = min_price + p1
                count = 2
                for price in sp[2:]:
                    if price == min_price or (price - min_price) <= (price + min_price) * 0.00005:
                        total += price
                        count += 1
                    else:
                        break
                return total / count
            if not u.are_close(sp[1], min_price, 0.01):
                return min_price
            total = min_price
            count = 1
            for price in sp[1:]:
                if u.are_close(price, min_price, 0.01):
                    total += price
                    count += 1
                else:
                    break
            return total / count

    def get_max(self):
        with self._lock:
            sp = self.sorted_prices
            if not sp:
                return None
            max_price = sp[-1]
            if len(sp) == 1:
                return max_price
            p_prev = sp[-2]
            if p_prev > 0.0:
                if p_prev != max_price and (max_price - p_prev) > (max_price + p_prev) * 0.00005:
                    return max_price
                total = max_price + p_prev
                count = 2
                for i in range(len(sp) - 3, -1, -1):
                    price = sp[i]
                    if price == max_price or (
                        price > 0.0 and (max_price - price) <= (max_price + price) * 0.00005
                    ):
                        total += price
                        count += 1
                    else:
                        break
                return total / count
            if not u.are_close(p_prev, max_price, 0.01):
                return max_price
            total = max_price
            count = 1
            for i in range(len(sp) - 2, -1, -1):
                price = sp[i]
                if u.are_close(price, max_price, 0.01):
                    total += price
                    count += 1
                else:
                    break
            return total / count

    def get_min_and_index(self):
        with self._lock:
            if not self.sorted_prices:
                return None, None
            min_price = self.get_min()
            if min_price is not None and min_price > 0:
                lo = min_price * (200.0 - 0.01) / (200.0 + 0.01) - 1e-12
                hi = min_price * (200.0 + 0.01) / (200.0 - 0.01) + 1e-12
                min_indices = [
                    i for i, price in enumerate(self.prices)
                    if lo <= price <= hi and u.are_close(price, min_price, 0.01)
                ]
            else:
                min_indices = [i for i, price in enumerate(self.prices) if u.are_close(price, min_price, 0.01)]
            centroid_index = sum(min_indices) / len(min_indices) if min_indices else None
            return min_price, centroid_index

    def get_max_and_index(self):
        with self._lock:
            if not self.sorted_prices:
                return None, None
            max_price = self.get_max()
            if max_price is not None and max_price > 0:
                lo = max_price * (200.0 - 0.01) / (200.0 + 0.01) - 1e-12
                hi = max_price * (200.0 + 0.01) / (200.0 - 0.01) + 1e-12
                max_indices = [
                    i for i, price in enumerate(self.prices)
                    if lo <= price <= hi and u.are_close(price, max_price, 0.01)
                ]
            else:
                max_indices = [i for i, price in enumerate(self.prices) if u.are_close(price, max_price, 0.01)]
            centroid_index = sum(max_indices) / len(max_indices) if max_indices else None
            return max_price, centroid_index

    def current_window_size(self):
        with self._lock:
            return len(self.prices)

    def get_recent_gradient(self) -> float:
        """Return recent momentum as mean gradient over the latest samples.

        This inexpensive silent path supplies the fast per-tick buy/sell gate.
        """
        with self._lock:
            prices = list(self.prices)
        if len(prices) < 2:
            return 0.0
        grad = np.gradient(np.array(prices))
        n = self.recent_n
        if len(grad) >= n:
            return float(np.mean(grad[-n:]))
        return float(np.mean(grad))

    def get_noise_epsilon(self, k: float = 1.0) -> float:
        """Noise threshold INFORMED by the volatility of the window:
        epsilon = k * stddev(np.gradient(prices)).
        It adapts by symbol and over time to distinguish real movement from noise."""
        k_val = 1.0
        if k is not None and not isinstance(k, bool):
            try:
                kf = float(k)
                if math.isfinite(kf) and kf > 0.0:
                    k_val = kf
            except (TypeError, ValueError, OverflowError):
                k_val = 1.0
        with self._lock:
            n = len(self.prices)
            if n < 3:
                return 0.0
            blen = len(self._trend_buf.prices)
            if (
                blen >= n
                and self._trend_buf.prices[blen - n] == self.prices[0]
                and self._trend_buf.prices[-1] == self.prices[-1]
            ):
                key = (blen, self._trend_buf.offset, n, self.sample_rate_sec, self.window_seconds,
                       _active_estimator(), STEP_WOLS_POWER, MEDIAN_FILTER_SECONDS, TREND_NOISE_GATE_K)
                if self._cached_eval_key == key and self._cached_eval_val is not None:
                    return float(k_val * self._cached_eval_val[4])
                res = self._trend_buf.eval_slice(
                    blen - n,
                    blen - 1,
                    sample_rate_sec=self.sample_rate_sec,
                    eps_k=1.0,
                    window_seconds=self.window_seconds,
                )
                self._cached_eval_key = key
                self._cached_eval_val = res
                return float(k_val * res[4])
            prices = list(self.prices)
        grad = np.gradient(np.array(prices))
        return float(k_val * np.std(grad))

    def get_short_trend(self):
        """Return ``(final_trend, growth_coefficient, slope_full, gradient_recent)``.

        Evaluate in O(1) via ``self._trend_buf`` when synchronized, falling back to
        ``instant_trend_from_prices`` if ``self.prices`` was mutated directly.
        """
        with self._lock:
            n = len(self.prices)
            blen = len(self._trend_buf.prices)
            if (
                blen >= n
                and (n == 0 or (self._trend_buf.prices[blen - n] == self.prices[0] and self._trend_buf.prices[-1] == self.prices[-1]))
            ):
                key = (blen, self._trend_buf.offset, n, self.sample_rate_sec, self.window_seconds,
                       _active_estimator(), STEP_WOLS_POWER, MEDIAN_FILTER_SECONDS, TREND_NOISE_GATE_K)
                if self._cached_eval_key == key and self._cached_eval_val is not None:
                    final_trend, growth_coefficient, slope_full, gradient_recent, _ = self._cached_eval_val
                else:
                    res = self._trend_buf.eval_slice(
                        blen - n,
                        blen - 1,
                        sample_rate_sec=self.sample_rate_sec,
                        eps_k=1.0,
                        window_seconds=self.window_seconds,
                    )
                    self._cached_eval_key = key
                    self._cached_eval_val = res
                    final_trend, growth_coefficient, slope_full, gradient_recent, _ = res
            else:
                prices_snapshot = list(self.prices)
                final_trend, growth_coefficient, slope_full, gradient_recent = instant_trend_from_prices(
                    prices_snapshot, self.sample_rate_sec, window_seconds=self.window_seconds)

        if getattr(builtins.print, "__name__", "") != "_dummy_print":
            raw_rate = self.sample_rate_sec
            rate_val = 1.0
            if raw_rate is not None and not isinstance(raw_rate, bool):
                try:
                    rf = float(raw_rate)
                    if math.isfinite(rf) and rf > 0:
                        rate_val = rf
                except (TypeError, ValueError, OverflowError):
                    rate_val = 1.0
            print(
                f"[{self.symbol}] slope_full={slope_full:.4f} "
                f"gradient_recent={gradient_recent:.4f} (recent_n={self.recent_n}, rate={rate_val:.2f}s)"
            )

        return final_trend, growth_coefficient, slope_full, gradient_recent

    def calculate_acceleration(self, window_sec: float = 90.0) -> tuple[float, float]:
        """Calculate recent price velocity (%/min) and acceleration (%/min^2).

        Fits a quadratic polynomial y(t) = a*t^2 + b*t + c over the latest samples,
        where t is in minutes and y is percentage move relative to window start.
        Returns (velocity, acceleration):
          velocity: dy/dt at the end of the window (in % / min)
          acceleration: d^2y/dt^2 = 2*a (in % / min^2)
        """
        raw_rate = self.sample_rate_sec
        if isinstance(raw_rate, bool) or raw_rate is None:
            rate = 1.0
        else:
            try:
                rf = float(raw_rate)
                rate = max(rf, 0.1) if math.isfinite(rf) and rf > 0 else 1.0
            except (TypeError, ValueError):
                rate = 1.0
        n_samples = None
        if window_sec is not None and not isinstance(window_sec, bool):
            try:
                wf = float(window_sec)
                if math.isfinite(wf) and wf > 0:
                    n_samples = max(int(round(wf / rate)), 3)
            except (TypeError, ValueError, OverflowError):
                n_samples = None
        with self._lock:
            n_win = len(self.prices)
            blen = len(self._trend_buf.prices)
            if (
                n_samples is not None
                and blen >= n_win
                and n_win > 0
                and self._trend_buf.prices[blen - n_win] == self.prices[0]
                and self._trend_buf.prices[-1] == self.prices[-1]
            ):
                take = min(n_samples, n_win)
                prices = self._trend_buf.prices[-take:]
            else:
                prices = list(self.prices)
                if n_samples is not None:
                    prices = prices[-n_samples:]
        n = len(prices)
        if n < 3:
            return 0.0, 0.0

        p0 = prices[0]
        if p0 <= 0:
            return 0.0, 0.0

        t_minutes = np.arange(n) * (rate / 60.0)
        y_pct = ((np.array(prices) - p0) / p0) * 100.0

        try:
            a, b, _ = np.polyfit(t_minutes, y_pct, 2)
            t_end = t_minutes[-1]
            velocity = float(2.0 * a * t_end + b)
            acceleration = float(2.0 * a)
            if not (math.isfinite(velocity) and math.isfinite(acceleration)):
                return 0.0, 0.0
            return velocity, acceleration
        except Exception:
            return 0.0, 0.0


class WindowAnalyzer:
    """Trading metrics composed from a ``PriceWindow``."""
    def __init__(self, window: "PriceWindow"):
        self.window = window

    def calculate_acceleration(self, window_sec: float = 90.0) -> tuple[float, float]:
        return self.window.calculate_acceleration(window_sec)

    def check_price_change(self, threshold):
        if threshold is None or isinstance(threshold, bool):
            return 0, 0
        try:
            threshold = float(threshold)
        except (TypeError, ValueError, OverflowError):
            return 0, 0
        if not math.isfinite(threshold):
            return 0, 0
        w = self.window
        if len(w.prices) < 2:
            return 0, 1
        min_price = w.get_min()
        max_price = w.get_max()
        newest_price = w.prices[-1]

        price_diff_min = u.calculate_difference_percent(min_price, newest_price)
        price_diff_max = u.calculate_difference_percent(max_price, newest_price)
        price_diff_newest = max(price_diff_min, price_diff_max)

        if (
            abs(price_diff_newest) >= threshold
            or ((threshold <= 0 or price_diff_newest >= threshold * 0.98) and u.are_close(price_diff_newest, threshold))
        ):
            if getattr(builtins.print, "__name__", "") != "_dummy_print":
                _, min_index = w.get_min_and_index()
                _, max_index = w.get_max_and_index()
                print(f'price_diff_minmax_versus_newest(slope)={price_diff_newest}(threshold={threshold}) '
                      f'are_close={u.are_close(price_diff_newest, threshold)}')
                print(f'min price ={min_price}, max_price = {max_price}, newest_price={newest_price}, '
                      f'min_index={min_index}, max_index={max_index}')
            return -price_diff_newest if price_diff_max > price_diff_min else price_diff_newest, 0
        return 0, 0

    # ══════════════════════════════════════════════════════════════════════════════
    # [HISTORICAL EXPERIMENTS & DRAFT METRICS — PRESERVE FOR REFERENCE — DO NOT DELETE]
    # Păstrate pentru analiză și potențială reutilizare/recalibrare în strategii.
    # Notă: evaluate_buy_sell_opportunity a fost ștearsă conform cerinței (era stub gol).
    # ══════════════════════════════════════════════════════════════════════════════
    # def calculate_slope_max_min(self):
    #     """Calculează panta directă între extremele ferestrei (min și max)."""
    #     w = self.window
    #     if len(w.sorted_prices) < 2:
    #         return 0
    #     min_price, min_index = w.get_min_and_index()
    #     max_price, max_index = w.get_max_and_index()
    #     if min_price is None or max_price is None or max_index == min_index:
    #         return 0
    #     return (max_price - min_price) / (max_index - min_index)
    #
    # def calculate_proximities(self, current_price):
    #     """Proximitatea prețului curent față de min (0..1) și max (0..1) din fereastră."""
    #     if current_price is None or isinstance(current_price, bool):
    #         return 0, 0
    #     try:
    #         cp = float(current_price)
    #     except (TypeError, ValueError, OverflowError):
    #         return 0, 0
    #     if not math.isfinite(cp):
    #         return 0, 0
    #     w = self.window
    #     min_price, _ = w.get_min_and_index()
    #     max_price, _ = w.get_max_and_index()
    #     if min_price is None or max_price is None or max_price == min_price:
    #         return 0, 0
    #     min_proximity = (cp - min_price) / (max_price - min_price)
    #     max_proximity = (max_price - cp) / (max_price - min_price)
    #     if not (math.isfinite(min_proximity) and math.isfinite(max_proximity)):
    #         return 0, 0
    #     return max(min_proximity, 0), max(max_proximity, 0)
    #
    # def calculate_positions(self):
    #     """Poziția temporală normalizată (index / window_size) a minimului și maximului."""
    #     w = self.window
    #     min_price, min_index = w.get_min_and_index()
    #     max_price, max_index = w.get_max_and_index()
    #     win_size = getattr(w, "window_size", 0)
    #     if isinstance(win_size, bool) or not isinstance(win_size, (int, float)) or not math.isfinite(win_size) or win_size <= 0:
    #         return None, None
    #     min_position = min_index / win_size if min_index is not None else None
    #     max_position = max_index / win_size if max_index is not None else None
    #     return min_position, max_position
    #
    # def _analyze_price_movement(self, min_price, min_index, max_price, max_index,
    #                             newest_price, newest_index, price_diff):
    #     """Analiză asimetrică a pantei dintre noul preț și cele două extreme (min/max)."""
    #     price_diff_min = u.calculate_difference_percent(min_price, newest_price)
    #     price_diff_max = u.calculate_difference_percent(max_price, newest_price)
    #     grow = price_diff_max < price_diff_min
    #     slope_min = u.slope(min_price, min_index, newest_price, newest_index)
    #     slope_max = u.slope(max_price, max_index, newest_price, newest_index)
    #     slope_max_min = slope_max if abs(slope_max) > abs(slope_min) else slope_min
    #     return slope_max_min, price_diff
