"""
Price-window primitives and trend analysis.

Extracted from ``tradeall`` for reuse without trading decisions.

Contains ``PriceTrendAnalyzer`` regression/EMA/gradient calculations, a lean sliding
``PriceWindow`` with volatility-informed noise epsilon, and composed trading metrics.

Short-trend SSOT (``growth_coefficient`` / ``final_trend``) is computed by the
Step-WOLS estimator directly on raw price increments (no median filter lag),
with per-instrument power ``trend.step_wols_power`` from ``instruments.conf``
(Cubic ``p=3.0`` for BTC, Quadratic ``p=2.0`` for altcoins/equities).
"""
import builtins
import math
import os
import threading
from collections import deque
from bisect import bisect_left

import numpy as np

from botcore import (
    float_env,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
import utils as u

# ── Window constants ──────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE_SEC = 0.8     # Nominal fallback sampling rate.
RECENT_GRADIENT_SECONDS = 60.0    # Recent-momentum window in seconds (for primary 222s window).
WINDOW_SECONDS_SMALL = 3.7 * 60          # 3.7 minutes (222.0s)
WINDOW_SECONDS_BIG   = 2.5 * 60 * 60     # 2.5 hours (9000.0s)

# ── Short-trend Estimator Registry (SSOT for growth_coefficient / final_trend) ──
# Direct Step-WOLS estimator on raw price increments:
#   growth_coefficient = sum(k^p * dy_k) / sum(k^p)
# Differentiated per instrument via trend.step_wols_power in instruments.conf:
#   - BTC (BTCUSDC): Cubic Step-WOLS (p=3.0)
#   - Altcoins / Equities (TAO, HYPE, ARB, ...): Quadratic Step-WOLS (p=2.0)
# Configurable globally via MPTRADE_STEP_WOLS_POWER (default 2.0; accepts any p >= 0).
# Causal median pre-filtering (med3/med5) was permanently removed after 11.58M-tick
# 1s OOS and 329-day portfolio backtests proved it destroys directional edge across all coins.
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

DEFAULT_STEP_WOLS_POWER = 2.0         # Default exponent p when symbol is not in instruments.conf


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


STEP_WOLS_POWER = _resolve_step_wols_power(SHORT_TREND_ESTIMATOR)
DEFAULT_TREND_NOISE_GATE_K = 0.0      # Multiplier k for adaptive noise gate (0.0 = bypassed, e.g. 0.05-0.10)
TREND_NOISE_GATE_K = max(0.0, _env_float("MPTRADE_TREND_NOISE_GATE_K", DEFAULT_TREND_NOISE_GATE_K))
_STEP_WOLS_DIRECT_MAX_STEPS = 1024          # Exact direct dot product up to this many steps.
_SYMBOL_POWER_CACHE = {"mtime_ns": None, "map": {}}
_SYMBOL_FAST_WINDOW_CACHE = {"mtime_ns": None, "map": {}}


def _register_symbol_param(target_map: dict[str, float], sym: str, base: str, c_base: str, val: float) -> None:
    if sym:
        target_map[sym] = val
    for b in (base, c_base):
        if b:
            target_map.setdefault(b, val)
            for q in ("USDC", "USDT", "USD", "EUR"):
                target_map.setdefault(f"{b}{q}", val)


def _load_symbol_power_map() -> dict[str, float]:
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruments.conf")
    try:
        mtime_ns = os.stat(conf_path).st_mtime_ns
    except OSError:
        return {}
    if _SYMBOL_POWER_CACHE["mtime_ns"] == mtime_ns and _SYMBOL_POWER_CACHE["map"]:
        return _SYMBOL_POWER_CACHE["map"]
    try:
        from instrument_registry import load_registry
        registry = load_registry(conf_path)
    except Exception:
        return {}
    try:
        from cachepaths import canonical_base_asset
    except Exception:
        canonical_base_asset = lambda x: ""  # noqa: E731
    mapping: dict[str, float] = {}
    fast_win_map: dict[str, float] = {}
    for spec in registry.values():
        sym = str(spec.symbol or "").strip().upper()
        base = str(spec.base or "").strip().upper()
        c_base = canonical_base_asset(base or sym)
        pval = safe_nonneg_float(spec.params.get("trend.step_wols_power"), default=None)
        if pval is not None:
            _register_symbol_param(mapping, sym, base, c_base, pval)
        fw_val = safe_pos_float(spec.params.get("trend.fast_window_sec"), default=None)
        if fw_val is not None:
            _register_symbol_param(fast_win_map, sym, base, c_base, fw_val)
    _SYMBOL_POWER_CACHE["mtime_ns"] = mtime_ns
    _SYMBOL_POWER_CACHE["map"] = mapping
    _SYMBOL_FAST_WINDOW_CACHE["mtime_ns"] = mtime_ns
    _SYMBOL_FAST_WINDOW_CACHE["map"] = fast_win_map
    return mapping


def _load_symbol_fast_window_map() -> dict[str, float]:
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruments.conf")
    try:
        mtime_ns = os.stat(conf_path).st_mtime_ns
    except OSError:
        return {}
    if _SYMBOL_FAST_WINDOW_CACHE["mtime_ns"] != mtime_ns:
        _load_symbol_power_map()
    return _SYMBOL_FAST_WINDOW_CACHE["map"]


def _lookup_symbol_param(symbol: str | None, param_map: dict[str, float]) -> float | None:
    if not (isinstance(symbol, str) and symbol.strip() and param_map):
        return None
    s_up = symbol.strip().upper().replace("/", "").replace("-", "")
    if s_up in param_map:
        return param_map[s_up]
    s_clean = s_up.replace("_", "")
    if s_clean in param_map:
        return param_map[s_clean]
    try:
        from cachepaths import canonical_base_asset, canonical_symbol_key
        ck = canonical_symbol_key(symbol)
        if ck and ck in param_map:
            return param_map[ck]
        c_base = canonical_base_asset(symbol)
        if c_base and c_base in param_map:
            return param_map[c_base]
    except Exception:
        pass
    return None


def fast_trend_window_for_symbol(symbol: str | None = None) -> float | None:
    """Resolve optional per-symbol fast short-trend window (in seconds) from ``instruments.conf``.

    Returns ``None`` when the symbol uses the default 222s primary window, or a positive
    float (e.g. ``90.0`` for ``TAOUSDC``) when ``trend.fast_window_sec`` is configured.
    """
    if "MPTRADE_FAST_TREND_WINDOW_SEC" in os.environ:
        val = _env_float("MPTRADE_FAST_TREND_WINDOW_SEC", 0.0)
        return val if val > 0.0 else None
    return _lookup_symbol_param(symbol, _load_symbol_fast_window_map())


def step_wols_power_for_symbol(symbol: str | None = None) -> float:
    """Resolve Step-WOLS exponent p for ``symbol``.

    Explicit env/module overrides (``MPTRADE_STEP_WOLS_POWER``,
    ``MPTRADE_SHORT_TREND_ESTIMATOR=step_wols_cubic|step_wols_linear``, or
    patched ``STEP_WOLS_POWER``) take precedence; otherwise looks up
    ``trend.step_wols_power`` in ``instruments.conf`` (BTC -> 3.0, altcoins -> 2.0),
    falling back to ``DEFAULT_STEP_WOLS_POWER`` (2.0).
    """
    if "MPTRADE_STEP_WOLS_POWER" in os.environ:
        return max(0.0, _env_float("MPTRADE_STEP_WOLS_POWER", DEFAULT_STEP_WOLS_POWER))
    raw_est = str(os.environ.get("MPTRADE_SHORT_TREND_ESTIMATOR", "") or "").strip().lower()
    if raw_est == SHORT_TREND_ESTIMATOR_STEP_WOLS_CUBIC:
        return 3.0
    if raw_est == SHORT_TREND_ESTIMATOR_STEP_WOLS_LINEAR:
        return 1.0
    if STEP_WOLS_POWER != DEFAULT_STEP_WOLS_POWER:
        return max(0.0, float(STEP_WOLS_POWER))
    found = _lookup_symbol_param(symbol, _load_symbol_power_map())
    if found is not None:
        return found
    return float(STEP_WOLS_POWER)


def _active_estimator() -> str:
    est = SHORT_TREND_ESTIMATOR
    if est in _SHORT_TREND_ESTIMATORS:
        return est
    return SHORT_TREND_ESTIMATOR_STEP_WOLS


def step_wols_slope(prices, sample_rate_sec=None, power=None, symbol: str | None = None) -> float:
    """Generalized Step-WOLS slope: sum(k^p * dy_k) / sum(k^p) directly on raw prices.

    Args:
        prices: Iterable of float prices.
        sample_rate_sec: Effective sampling interval in seconds (retained for signature compatibility).
        power: Exponent p (w_k = k^p). Defaults to step_wols_power_for_symbol(symbol).
        symbol: Optional symbol name for per-coin power lookup from instruments.conf.
    """
    if prices is None or isinstance(prices, (bool, str, bytes)):
        return 0.0
    pwr = safe_nonneg_float(power, default=None)
    if pwr is None:
        pwr = float(step_wols_power_for_symbol(symbol))
    try:
        arr = np.asarray(prices, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0
    if arr.ndim != 1 or len(arr) < 2 or not np.all(np.isfinite(arr)):
        return 0.0
    m = len(arr) - 1
    k = np.arange(1, m + 1, dtype=np.float64)
    w = k ** pwr if pwr != 1.0 else k
    val = safe_float(np.dot(w, np.diff(arr)) / np.sum(w), default=None)
    tol = 1e-10 * max(1.0, abs(float(arr[-1])))
    return 0.0 if (val is None or abs(val) <= tol) else val


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
    win_f = safe_float(window_seconds, default=None)
    if win_f is not None and win_f > WINDOW_SECONDS_SMALL:
        return win_f * (RECENT_GRADIENT_SECONDS / WINDOW_SECONDS_SMALL)
    return RECENT_GRADIENT_SECONDS


class PriceTrendAnalyzer:
    def __init__(self, prices):
        clean = []
        if prices is not None and not isinstance(prices, (bool, str, bytes)):
            try:
                for p in prices:
                    pf = safe_float(p, default=None)
                    if pf is not None:
                        clean.append(pf)
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
    sr = safe_pos_float(sample_rate_sec, default=None)
    rs = safe_float(recent_sec, default=None)
    if sr is None or rs is None:
        return 2
    return max(2, safe_int(rs / sr, default=2))


def instant_trend_from_prices(
    prices,
    sample_rate_sec,
    window_seconds=None,
    symbol: str | None = None,
    power: float | None = None,
):
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
    growth_coefficient = step_wols_slope(
        analyzer.prices, sample_rate_sec, power=power, symbol=symbol
    )
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
        self.pk2 = []
        self.pg2 = []
        self.plr2 = []
        self.nonpos = []
        self.offset = 0
        self.base_p0 = 0.0
        self.base_p1 = 0.0
        self.base_p2 = 0.0
        self.base_pk2 = 0.0
        self.y_ref = None

    def clear(self):
        self.timestamps.clear()
        self.prices.clear()
        self.p0.clear()
        self.p1.clear()
        self.p2.clear()
        self.pk2.clear()
        self.pg2.clear()
        self.plr2.clear()
        self.nonpos.clear()
        self.offset = 0
        self.base_p0 = self.base_p1 = self.base_p2 = self.base_pk2 = 0.0
        self.y_ref = None

    def append(self, price: float, ts_ms: int = 0) -> None:
        y = safe_float(price, default=None)
        ts_int = safe_int(ts_ms, default=None)
        if y is None or ts_int is None:
            return
        if self.y_ref is None:
            self.y_ref = y
        i = len(self.prices)
        k = self.offset + i
        kf = float(k)
        dy = y - self.y_ref
        self.timestamps.append(ts_int)
        self.prices.append(y)
        prev0 = self.p0[-1] if i > 0 else self.base_p0
        prev1 = self.p1[-1] if i > 0 else self.base_p1
        prev2 = self.p2[-1] if i > 0 else self.base_p2
        prevk2 = self.pk2[-1] if i > 0 else self.base_pk2
        self.p0.append(prev0 + dy)
        self.p1.append(prev1 + kf * dy)
        self.p2.append(prev2 + dy * dy)
        self.pk2.append(prevk2 + kf * kf * dy)
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

    def _raw_moment_range(self, lo: int, hi: int):
        """Return (sum dy, sum J*dy, sum J^2*dy) over buffer indices [lo, hi]."""
        if lo > 0:
            b0, b1, bk2 = self.p0[lo - 1], self.p1[lo - 1], self.pk2[lo - 1]
        else:
            b0, b1, bk2 = self.base_p0, self.base_p1, self.base_pk2
        return self.p0[hi] - b0, self.p1[hi] - b1, self.pk2[hi] - bk2

    def step_wols_slope(
        self, a: int, b: int, sample_rate_sec=None, power=None, symbol: str | None = None
    ) -> float:
        """Step-WOLS slope on [a, b] directly over raw prices with configurable power.

        Equals ``step_wols_slope(prices[a:b+1], sample_rate_sec, power, symbol)``.
        Slices up to ``_STEP_WOLS_DIRECT_MAX_STEPS`` steps (1024) use exact direct
        vectorized dot product; longer slices use O(1) Abel summation on prefix sums
        for p=1 (Linear), p=2 (Quadratic), or p=3 (Cubic).
        """
        m = b - a
        if m < 1:
            return 0.0
        p = self.prices
        pwr = safe_nonneg_float(power, default=None)
        if pwr is None:
            pwr = float(step_wols_power_for_symbol(symbol))
        ref_last = float(p[b])

        if m <= _STEP_WOLS_DIRECT_MAX_STEPS or pwr not in (1.0, 2.0, 3.0):
            f = np.asarray(p[a: b + 1], dtype=np.float64)
            w = np.arange(1, m + 1, dtype=np.float64) ** pwr if pwr != 1.0 else np.arange(1, m + 1, dtype=np.float64)
            val = float(np.dot(w, np.diff(f)) / np.sum(w))
        elif pwr == 1.0:
            # O(1) Linear Abel summation:
            # sum_{k=1..m} k (y_{a+k} - y_{a+k-1}) = m * dy_b - sum_{r=0..m-1} dy_{a+r}
            y_ref = self.y_ref if self.y_ref is not None else 0.0
            a0, _, _ = self._raw_moment_range(a, b - 1)
            total = float(m) * (ref_last - y_ref) - a0
            s1 = 0.5 * float(m) * float(m + 1)
            val = float(total / s1)
        elif pwr == 2.0:
            # O(1) Quadratic Abel summation:
            # sum_{k=1..m} k^2 (y_{a+k} - y_{a+k-1}) = m^2 dy_b - sum_{r=0..m-1} (2r + 1) dy_{a+r}
            y_ref = self.y_ref if self.y_ref is not None else 0.0
            c = float(self.offset + a)
            a0, a1, _ = self._raw_moment_range(a, b - 1)
            sum_2r1 = 2.0 * a1 - (2.0 * c - 1.0) * a0
            m2 = float(m) ** 2
            total = m2 * (ref_last - y_ref) - sum_2r1
            s2 = (float(m) * float(m + 1) * float(2 * m + 1)) / 6.0
            val = float(total / s2)
        else:
            # O(1) Cubic Abel summation (p=3):
            y_ref = self.y_ref if self.y_ref is not None else 0.0
            c = float(self.offset + a)
            a0, a1, a2 = self._raw_moment_range(a, b - 1)
            sum_r2 = a2 - 2.0 * c * a1 + c * c * a0
            sum_r1 = a1 - c * a0
            m3 = float(m) ** 3
            total = m3 * (ref_last - y_ref) - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
            s3 = (0.5 * float(m) * float(m + 1)) ** 2
            val = safe_float(total / s3, default=None)

        tol = 1e-10 * max(1.0, abs(ref_last))
        return 0.0 if (val is None or abs(val) <= tol) else val

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
        count_int = safe_pos_int(count, default=None)
        if count_int is None:
            return
        idx = min(count_int, len(self.prices))
        if idx <= 0:
            return
        self.base_p0 = self.p0[idx - 1]
        self.base_p1 = self.p1[idx - 1]
        self.base_p2 = self.p2[idx - 1]
        self.base_pk2 = self.pk2[idx - 1]
        del self.timestamps[:idx]
        del self.prices[:idx]
        del self.p0[:idx]
        del self.p1[:idx]
        del self.p2[:idx]
        del self.pk2[:idx]
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

        def _rebuild() -> None:
            self.clear()
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.append(e[1], e[0])

        if not self.timestamps:
            _rebuild()
            return
        try:
            first_ts, first_p = int(entries[0][0]), float(entries[0][1])
        except (TypeError, ValueError, IndexError, OverflowError):
            _rebuild()
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
            _rebuild()
            return
        cur_len = len(self.timestamps)
        try:
            last_ts, last_p = int(entries[cur_len - 1][0]), float(entries[cur_len - 1][1])
        except (TypeError, ValueError, IndexError, OverflowError):
            _rebuild()
            return
        if self.timestamps[-1] != last_ts or self.prices[-1] != last_p:
            _rebuild()
            return
        for idx in range(cur_len, len(entries)):
            e = entries[idx]
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                self.append(e[1], e[0])

    def eval_slice(
        self,
        a: int,
        b: int,
        sample_rate_sec: float = DEFAULT_SAMPLE_RATE_SEC,
        eps_k: float = 1.0,
        window_seconds: float | None = None,
        power: float | None = None,
        symbol: str | None = None,
    ):
        """Evaluate (final_trend, growth_coefficient, slope_full, gradient_recent, epsilon)
        on inclusive index range [a, b] in O(1) using cumulative prefix sums."""
        a_i = safe_nonneg_int(a, default=None)
        b_i = safe_nonneg_int(b, default=None)
        if a_i is None or b_i is None:
            return 0, 0.0, 0.0, 0.0, 0.0
        a, b = a_i, b_i
        n = b - a + 1
        if a < 0 or b >= len(self.prices) or n < 2:
            return 0, 0.0, 0.0, 0.0, 0.0
        k_val = safe_pos_float(eps_k, default=1.0) or 1.0
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

        growth_coefficient = self.step_wols_slope(
            a, b, sample_rate_sec, power=power, symbol=symbol
        )

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
        a = safe_nonneg_int(a, default=None)
        b = safe_nonneg_int(b, default=None)
        if a is None or b is None:
            return None
        n = b - a + 1
        if b >= len(self.prices) or n < 20:
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
        ws_i = safe_int(window_size, default=None)
        self.window_size = max(1, ws_i) if ws_i is not None else 10
        self.sample_rate_sec = safe_pos_float(sample_rate_sec, default=DEFAULT_SAMPLE_RATE_SEC) or DEFAULT_SAMPLE_RATE_SEC
        self.window_seconds = safe_pos_float(window_seconds, default=None)
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
        rate_val = safe_pos_float(rate, default=None)
        if rate_val is None:
            return
        self.sample_rate_sec = rate_val
        wsec = safe_pos_float(self.window_seconds, default=None)
        if wsec is None:
            return
        new_size = max(10, int(wsec / rate_val))
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
                ts_f = safe_float(e[0], default=None)
                if ts_f is not None:
                    timestamps_sec.append(ts_f / 1000.0)
        except TypeError:
            return DEFAULT_SAMPLE_RATE_SEC
        if len(timestamps_sec) < 2:
            return DEFAULT_SAMPLE_RATE_SEC
        gaps = [timestamps_sec[i+1] - timestamps_sec[i]
                for i in range(len(timestamps_sec) - 1)
                if timestamps_sec[i+1] > timestamps_sec[i]]
        if not gaps:
            return DEFAULT_SAMPLE_RATE_SEC
        return safe_pos_float(np.median(gaps), default=DEFAULT_SAMPLE_RATE_SEC)

    @classmethod
    def from_cache24(cls, symbol: str, window_seconds: float, cache24) -> "PriceWindow":
        """Build a window from recent Cache24 data and subscribe to future updates."""
        win_sec = safe_pos_float(window_seconds, default=WINDOW_SECONDS_SMALL) or WINDOW_SECONDS_SMALL
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
        price_f = safe_float(price, default=None)
        if price_f is None:
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

    def _centroid_index_for(self, target_price):
        if target_price is not None and target_price > 0:
            lo = target_price * (200.0 - 0.01) / (200.0 + 0.01) - 1e-12
            hi = target_price * (200.0 + 0.01) / (200.0 - 0.01) + 1e-12
            indices = [
                i for i, price in enumerate(self.prices)
                if lo <= price <= hi and u.are_close(price, target_price, 0.01)
            ]
        else:
            indices = [i for i, price in enumerate(self.prices) if u.are_close(price, target_price, 0.01)]
        return sum(indices) / len(indices) if indices else None

    def get_min_and_index(self):
        with self._lock:
            if not self.sorted_prices:
                return None, None
            min_price = self.get_min()
            return min_price, self._centroid_index_for(min_price)

    def get_max_and_index(self):
        with self._lock:
            if not self.sorted_prices:
                return None, None
            max_price = self.get_max()
            return max_price, self._centroid_index_for(max_price)

    def current_window_size(self):
        with self._lock:
            return len(self.prices)

    def _is_trend_buf_synced(self) -> bool:
        n = len(self.prices)
        blen = len(self._trend_buf.prices)
        return (
            blen >= n
            and (n == 0 or (self._trend_buf.prices[blen - n] == self.prices[0] and self._trend_buf.prices[-1] == self.prices[-1]))
        )

    def _eval_synced_trend_buf(self, eff_power: float | None = None):
        n = len(self.prices)
        blen = len(self._trend_buf.prices)
        if eff_power is None:
            eff_power = step_wols_power_for_symbol(self.symbol)
        noise_gate_k = max(0.0, _env_float("MPTRADE_TREND_NOISE_GATE_K", TREND_NOISE_GATE_K))
        key = (blen, self._trend_buf.offset, n, self.sample_rate_sec, self.window_seconds,
               _active_estimator(), eff_power, noise_gate_k)
        if self._cached_eval_key != key or self._cached_eval_val is None:
            self._cached_eval_val = self._trend_buf.eval_slice(
                blen - n,
                blen - 1,
                sample_rate_sec=self.sample_rate_sec,
                eps_k=1.0,
                window_seconds=self.window_seconds,
                power=eff_power,
                symbol=self.symbol,
            )
            self._cached_eval_key = key
        return self._cached_eval_val

    def get_recent_gradient(self) -> float:
        """Return recent momentum as mean gradient over the latest samples in O(1).

        This inexpensive silent path supplies the fast per-tick buy/sell gate.
        """
        r = max(2, int(self.recent_n))
        with self._lock:
            n = len(self.prices)
            if n < 2:
                return 0.0
            if self._is_trend_buf_synced():
                blen = len(self._trend_buf.prices)
                p = self._trend_buf.prices
                a = blen - n
                b = blen - 1
            else:
                p = self.prices
                a = 0
                b = n - 1
            if n > r:
                val = ((p[b] - p[b - 1]) + (p[b] - p[b - r + 1]) + (p[b] - p[b - r])) / (2.0 * r)
            else:
                sum_g_all = (p[b] - p[a]) + 0.5 * ((p[b] - p[b - 1]) + (p[a + 1] - p[a]))
                val = sum_g_all / n
        return float(val)

    def get_noise_epsilon(self, k: float = 1.0) -> float:
        """Noise threshold INFORMED by the volatility of the window:
        epsilon = k * stddev(np.gradient(prices)).
        It adapts by symbol and over time to distinguish real movement from noise."""
        k_val = safe_pos_float(k, default=1.0) or 1.0
        with self._lock:
            n = len(self.prices)
            if n < 3:
                return 0.0
            if self._is_trend_buf_synced():
                res = self._eval_synced_trend_buf()
                return float(k_val * res[4])
            prices = list(self.prices)
        return float(k_val * self._std_of_gradient(prices))

    @staticmethod
    def _std_of_gradient(prices) -> float:
        """Return std(np.gradient(prices)) in a single centered pass using the exact telescoping mean."""
        n = len(prices)
        if n < 2:
            return 0.0
        if n == 2:
            return 0.0
        mean_g = (1.5 * (prices[n - 1] - prices[0]) - 0.5 * (prices[n - 2] - prices[1])) / n
        d0 = (prices[1] - prices[0]) - mean_g
        dn = (prices[n - 1] - prices[n - 2]) - mean_g
        ssq = d0 * d0 + dn * dn
        for i in range(1, n - 1):
            di = 0.5 * (prices[i + 1] - prices[i - 1]) - mean_g
            ssq += di * di
        return math.sqrt(ssq / n)

    def get_short_trend(self):
        """Return ``(final_trend, growth_coefficient, slope_full, gradient_recent)``.

        Evaluate in O(1) via ``self._trend_buf`` when synchronized, falling back to
        ``instant_trend_from_prices`` if ``self.prices`` was mutated directly.
        """
        with self._lock:
            eff_power = step_wols_power_for_symbol(self.symbol)
            if self._is_trend_buf_synced():
                final_trend, growth_coefficient, slope_full, gradient_recent, _ = self._eval_synced_trend_buf(eff_power)
            else:
                prices_snapshot = list(self.prices)
                final_trend, growth_coefficient, slope_full, gradient_recent = instant_trend_from_prices(
                    prices_snapshot,
                    self.sample_rate_sec,
                    window_seconds=self.window_seconds,
                    symbol=self.symbol,
                    power=eff_power,
                )

        if getattr(builtins.print, "__name__", "") != "_dummy_print":
            rate_val = safe_pos_float(self.sample_rate_sec, default=1.0) or 1.0
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
        rate = max(safe_pos_float(self.sample_rate_sec, default=1.0) or 1.0, 0.1)
        wf = safe_pos_float(window_sec, default=None)
        n_samples = max(int(round(wf / rate)), 3) if wf is not None else None
        with self._lock:
            n_win = len(self.prices)
            if n_win > 0 and self._is_trend_buf_synced():
                blen = len(self._trend_buf.prices)
                take = min(n_samples, n_win) if n_samples is not None else n_win
                prices = self._trend_buf.prices[blen - take : blen]
            else:
                prices = list(self.prices)
                if n_samples is not None:
                    prices = prices[-n_samples:]
        n = len(prices)
        if n < 3:
            return 0.0, 0.0

        p0 = safe_pos_float(prices[0], default=None)
        if p0 is None:
            return 0.0, 0.0

        try:
            arr = np.asarray(prices, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                return 0.0, 0.0
            dp = arr - p0
            n_f = float(n)
            mu = 0.5 * (n_f - 1.0)
            u = np.arange(n, dtype=np.float64) - mu
            v1 = (n_f * n_f - 1.0) / 12.0
            d1 = n_f * v1
            d2 = (n_f * (n_f * n_f - 1.0) * (n_f * n_f - 4.0)) / 180.0
            scale_pct = 100.0 / p0
            c1 = scale_pct * float(np.dot(u, dp)) / d1
            c2 = scale_pct * float(np.dot(u * u - v1, dp)) / d2
            dt_min = rate / 60.0
            velocity = safe_float((c1 + c2 * (n_f - 1.0)) / dt_min, default=None)
            acceleration = safe_float((2.0 * c2) / (dt_min * dt_min), default=None)
            if velocity is None or acceleration is None:
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
        threshold_f = safe_float(threshold, default=None)
        if threshold_f is None:
            return 0, 0
        threshold = threshold_f
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
    # Preserved for analysis and potential reuse/recalibration in strategies.
    # Note: evaluate_buy_sell_opportunity was removed per requirement (empty stub).
    # ══════════════════════════════════════════════════════════════════════════════
    # def calculate_slope_max_min(self):
    #     """Calculate direct slope between window extremes (min and max)."""
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
    #     """Current price proximity to min (0..1) and max (0..1) within window."""
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
    #     """Normalized temporal position (index / window_size) of min and max."""
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
    #     """Asymmetric slope analysis between latest price and both extremes (min/max)."""
    #     price_diff_min = u.calculate_difference_percent(min_price, newest_price)
    #     price_diff_max = u.calculate_difference_percent(max_price, newest_price)
    #     grow = price_diff_max < price_diff_min
    #     slope_min = u.slope(min_price, min_index, newest_price, newest_index)
    #     slope_max = u.slope(max_price, max_index, newest_price, newest_index)
    #     slope_max_min = slope_max if abs(slope_max) > abs(slope_min) else slope_min
    #     return slope_max_min, price_diff
