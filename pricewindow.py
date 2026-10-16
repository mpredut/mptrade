"""
Price-window primitives and trend analysis.

Extracted from ``tradeall`` for reuse without trading decisions.

Contains ``PriceTrendAnalyzer`` regression/EMA/gradient calculations, a lean sliding
``PriceWindow`` with volatility-informed noise epsilon, and composed trading metrics.
"""
import threading
from collections import deque
from bisect import insort, bisect_left

import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import utils as u

# ── Constante window ─────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE_SEC = 0.8     # Nominal fallback sampling rate.
RECENT_GRADIENT_SECONDS = 60.0    # Recent-momentum window in seconds.
WINDOW_SECONDS_SMALL = 3.7 * 60          # 3.7 minute
WINDOW_SECONDS_BIG   = 2.5 * 60 * 60     # 2.5 ore


class PriceTrendAnalyzer:
    def __init__(self, prices):
        self.prices = prices

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

    def polynomial_regression_trend(self, degree=2):
        x = np.arange(len(self.prices)).reshape(-1, 1)
        y = np.array(self.prices)
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x)

        model = LinearRegression().fit(x_poly, y)
        trend_poly = model.predict(x_poly)
        return trend_poly, model.coef_

    def exponential_moving_average(self, span=5):
        prices_list = list(self.prices)
        ema = [prices_list[0]]
        alpha = 2 / (span + 1)
        for price in prices_list[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        return ema

    def calculate_gradient(self):
        if len(self.prices) < 2:
            print("Gradient: not enough data to compute the gradient.")
            return [], 0
        y = np.array(self.prices)
        gradient = np.gradient(y)
        avg_gradient = np.mean(gradient)
        return gradient, avg_gradient


def instant_trend_from_prices(prices, sample_rate_sec):
    """Apply the exact ``PriceWindow.get_instant_trend`` formula directly to a
    price list and sampling rate without constructing a complete window. This avoids
    incremental deque bookkeeping for a one-shot raw slice and is the single source of
    the formula extracted from ``PriceWindow.get_instant_trend`` and
    offline/research/monitortrades_backtest/replay_trend_source._instant_trend_from_slice)
    All ad-hoc dynamic-cache and replay windows delegate here.

    Return ``(final_trend, growth_coefficient, slope_full, gradient_recent)``."""
    if len(prices) < 2:
        return 0, 0.0, 0.0, 0.0
    analyzer = PriceTrendAnalyzer(list(prices))
    _, slope_full, _ = analyzer.linear_regression_trend()
    if slope_full is None:
        slope_full = 0.0
    gradient_lst, _ = analyzer.calculate_gradient()
    recent_n = max(2, int(RECENT_GRADIENT_SECONDS / sample_rate_sec)) if sample_rate_sec > 0 else 2
    if len(gradient_lst) >= recent_n:
        gradient_recent = float(np.mean(gradient_lst[-recent_n:]))
    else:
        gradient_recent = float(np.mean(gradient_lst)) if len(gradient_lst) else 0.0
    growth_coefficient = (slope_full + gradient_recent) / 2.0
    final_trend = 1 if growth_coefficient > 0 else (-1 if growth_coefficient < 0 else 0)
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

    def append(self, price: float, ts_ms: int = 0) -> None:
        import math
        y = float(price)
        if self.y_ref is None:
            self.y_ref = y
        i = len(self.prices)
        k = self.offset + i
        dy = y - self.y_ref
        self.timestamps.append(int(ts_ms))
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
        if count <= 0:
            return
        idx = min(int(count), len(self.prices))
        if idx <= 0:
            return
        self.base_p0 = self.p0[idx - 1]
        self.base_p1 = self.p1[idx - 1]
        self.base_p2 = self.p2[idx - 1]
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
        elif self.offset >= max(512, 2 * len(self.prices)):
            self._rebase()

    def sync_entries(self, entries) -> None:
        """Synchronize incrementally with an ordered list of [ts_ms, price] entries."""
        if not entries:
            self.clear()
            return
        if not self.timestamps:
            for e in entries:
                self.append(e[1], e[0])
            return
        first_ts, first_p = int(entries[0][0]), float(entries[0][1])
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
                self.append(e[1], e[0])
            return
        cur_len = len(self.timestamps)
        last_ts, last_p = int(entries[cur_len - 1][0]), float(entries[cur_len - 1][1])
        if self.timestamps[-1] != last_ts or self.prices[-1] != last_p:
            self.clear()
            for e in entries:
                self.append(e[1], e[0])
            return
        for idx in range(cur_len, len(entries)):
            e = entries[idx]
            self.append(e[1], e[0])

    def eval_slice(self, a: int, b: int, sample_rate_sec: float = DEFAULT_SAMPLE_RATE_SEC,
                   eps_k: float = 1.0):
        """Evaluate (final_trend, growth_coefficient, slope_full, gradient_recent, epsilon)
        on inclusive index range [a, b] in O(1) using cumulative prefix sums."""
        n = b - a + 1
        if a < 0 or b >= len(self.prices) or n < 2:
            return 0, 0.0, 0.0, 0.0, 0.0
        p = self.prices
        s0 = self.p0[b] - (self.p0[a - 1] if a > 0 else self.base_p0)
        s1_raw = self.p1[b] - (self.p1[a - 1] if a > 0 else self.base_p1)
        s2 = self.p2[b] - (self.p2[a - 1] if a > 0 else self.base_p2)
        ss_y = s2 - (s0 * s0) / n
        if ss_y <= 1e-18 or all(p[i] == p[a] for i in (a, (a + b) // 2, b)) and ss_y <= 1e-12:
            slope_full = 0.0
        else:
            cov_xy = s1_raw - (self.offset + a + 0.5 * (n - 1)) * s0
            slope_full = float((12.0 * cov_xy) / (n * (n * n - 1)))

        recent_n = (
            max(2, int(RECENT_GRADIENT_SECONDS / sample_rate_sec))
            if sample_rate_sec and sample_rate_sec > 0
            else 2
        )
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

        growth_coefficient = float((slope_full + gradient_recent) / 2.0)
        final_trend = 1 if growth_coefficient > 0 else (-1 if growth_coefficient < 0 else 0)

        if n < 3:
            epsilon = 0.0
        else:
            g0 = p[a + 1] - p[a]
            gn = p[b] - p[b - 1]
            sum_g2 = g0 * g0 + (self.pg2[b - 1] - self.pg2[a]) + gn * gn
            mean_g = sum_g_all / n
            var_g = max(0.0, sum_g2 / n - mean_g * mean_g)
            epsilon = float(eps_k * np.sqrt(var_g))

        return final_trend, growth_coefficient, slope_full, gradient_recent, epsilon

    def log_return_std(self, a: int, b: int) -> float | None:
        """Return O(1) population std of log returns on slice [a, b] if all prices > 0."""
        import math
        n = b - a + 1
        if n < 20:
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
    __slots__ = ("_trend_buf",)


class PriceWindow:
    def __init__(self, symbol, window_size, sample_rate_sec=None, initial_prices=None,
                 window_seconds=None):
        self.symbol = symbol
        self.window_size = int(window_size)
        self.sample_rate_sec = sample_rate_sec if sample_rate_sec is not None else DEFAULT_SAMPLE_RATE_SEC
        # When set, target duration resizes with the observed sampling rate so the
        # window continues to cover the same elapsed time.
        self.window_seconds = window_seconds
        self._subscribed_to_cache24 = False
        # WS/Cache24 updates on a different thread from evaluation. Protect mutation
        # and iteration of both price collections.
        self._lock = threading.RLock()
        self._trend_buf = IncrementalTrendBuffer()
        self.prices = _PriceDeque(maxlen=self.window_size)
        self.prices._trend_buf = self._trend_buf
        self.sorted_prices = []

        if initial_prices:
            for price in initial_prices[-self.window_size:]:
                self.process_price(price)

    @property
    def recent_n(self) -> int:
        """Return the sample count corresponding to ``RECENT_GRADIENT_SECONDS``."""
        return max(2, int(RECENT_GRADIENT_SECONDS / self.sample_rate_sec))

    def set_sample_rate(self, rate):
        """Update observed sampling rate and resize a duration-based window."""
        if rate is None or rate <= 0:
            return
        self.sample_rate_sec = rate
        if self.window_seconds is None:
            return
        new_size = max(10, int(self.window_seconds / rate))
        with self._lock:
            if new_size != self.window_size:
                self.window_size = new_size
                if len(self.prices) > new_size:
                    drop = len(self.prices) - new_size
                    if drop <= 64 and len(self.sorted_prices) == len(self.prices):
                        for _ in range(drop):
                            oldest_price = self.prices.popleft()
                            idx = bisect_left(self.sorted_prices, oldest_price)
                            if idx < len(self.sorted_prices) and self.sorted_prices[idx] == oldest_price:
                                del self.sorted_prices[idx]
                        self.prices = _PriceDeque(self.prices, maxlen=new_size)
                        self.prices._trend_buf = self._trend_buf
                    else:
                        kept = list(self.prices)[-new_size:]
                        self.prices = _PriceDeque(kept, maxlen=new_size)
                        self.prices._trend_buf = self._trend_buf
                        self.sorted_prices = sorted(self.prices)
                    if len(self._trend_buf.prices) > new_size:
                        self._trend_buf.trim_front(len(self._trend_buf.prices) - new_size)
                else:
                    self.prices = _PriceDeque(self.prices, maxlen=new_size)
                    self.prices._trend_buf = self._trend_buf

    def on_price_update(self, symbol: str, ts_ms: int, price: float) -> None:
        """Update the window from a ``Cache24PriceManager`` callback."""
        if symbol != self.symbol:
            return
        self.process_price(price)

    @classmethod
    def from_existing_window(cls, existing_prices, window_size):
        return cls(window_size, initial_prices=existing_prices)

    @staticmethod
    def _sample_rate_from_entries(entries) -> float:
        """Return the median sampling gap in seconds for [ts_ms, price] entries."""
        if len(entries) < 2:
            return DEFAULT_SAMPLE_RATE_SEC
        timestamps_sec = [e[0] / 1000.0 for e in entries]
        gaps = [timestamps_sec[i+1] - timestamps_sec[i]
                for i in range(len(timestamps_sec) - 1)
                if timestamps_sec[i+1] > timestamps_sec[i]]
        if not gaps:
            return DEFAULT_SAMPLE_RATE_SEC
        return float(np.median(gaps))

    @classmethod
    def from_cache24(cls, symbol: str, window_seconds: float, cache24) -> "PriceWindow":
        """Build a window from recent Cache24 data and subscribe to future updates."""
        entries = cache24.get_recent_entries(symbol, last_seconds=window_seconds)

        sample_rate = cls._sample_rate_from_entries(entries)
        window_size = max(10, int(window_seconds / sample_rate))

        prices = [e[1] for e in entries]
        pw = cls(symbol, window_size, sample_rate_sec=sample_rate, window_seconds=window_seconds)
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
        with self._lock:
            if len(self.prices) == self.window_size:
                oldest_price = self.prices.popleft()
                index = bisect_left(self.sorted_prices, oldest_price)
                if index < len(self.sorted_prices) and self.sorted_prices[index] == oldest_price:
                    del self.sorted_prices[index]
                else:
                    print("HAHAHAHAA")
            if len(self._trend_buf.prices) >= self.window_size:
                self._trend_buf.trim_front(len(self._trend_buf.prices) - self.window_size + 1)

            self.prices.append(price)
            insort(self.sorted_prices, price)
            self._trend_buf.append(price)

            if len(self.sorted_prices) != len(self.prices):
                print("XXXXXXXXXXXXXXXXXXX")

    def get_newest_index(self):
        return len(self.prices) - 1 if self.prices else None

    def get_min(self):
        with self._lock:
            if not self.sorted_prices:
                return None
            min_price = self.sorted_prices[0]
            total = 0.0
            count = 0
            for price in self.sorted_prices:
                if u.are_close(price, min_price, 0.01):
                    total += price
                    count += 1
                else:
                    break
            return total / count if count else min_price

    def get_max(self):
        with self._lock:
            if not self.sorted_prices:
                return None
            max_price = self.sorted_prices[-1]
            total = 0.0
            count = 0
            for price in reversed(self.sorted_prices):
                if u.are_close(price, max_price, 0.01):
                    total += price
                    count += 1
                else:
                    break
            return total / count if count else max_price

    def get_min_and_index(self):
        with self._lock:
            if not self.sorted_prices:
                print("BED1")
                return None, None
            min_price = self.get_min()
            min_indices = [i for i, price in enumerate(self.prices) if u.are_close(price, min_price, 0.01)]
            centroid_index = sum(min_indices) / len(min_indices) if min_indices else None
            return min_price, centroid_index

    def get_max_and_index(self):
        with self._lock:
            if not self.sorted_prices:
                print("BED2")
                return None, None
            max_price = self.get_max()
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
        with self._lock:
            n = len(self.prices)
            if n < 3:
                return 0.0
            if (
                len(self._trend_buf.prices) == n
                and self._trend_buf.prices[0] == self.prices[0]
                and self._trend_buf.prices[-1] == self.prices[-1]
            ):
                _, _, _, _, epsilon = self._trend_buf.eval_slice(
                    0, n - 1, sample_rate_sec=self.sample_rate_sec, eps_k=k
                )
                return epsilon
            prices = list(self.prices)
        grad = np.gradient(np.array(prices))
        return float(k * np.std(grad))

    def get_instant_trend(self):
        """Return ``(final_trend, growth_coefficient, slope_full, gradient_recent)``.

        Evaluate in O(1) via ``self._trend_buf`` when synchronized, falling back to
        ``instant_trend_from_prices`` if ``self.prices`` was mutated directly.
        """
        with self._lock:
            n = len(self.prices)
            if (
                len(self._trend_buf.prices) == n
                and (n == 0 or (self._trend_buf.prices[0] == self.prices[0] and self._trend_buf.prices[-1] == self.prices[-1]))
            ):
                final_trend, growth_coefficient, slope_full, gradient_recent, _ = (
                    self._trend_buf.eval_slice(0, n - 1, sample_rate_sec=self.sample_rate_sec)
                )
            else:
                prices_snapshot = list(self.prices)
                final_trend, growth_coefficient, slope_full, gradient_recent = instant_trend_from_prices(
                    prices_snapshot, self.sample_rate_sec)

        print(
            f"[{self.symbol}] slope_full={slope_full:.4f} "
            f"gradient_recent={gradient_recent:.4f} (recent_n={self.recent_n}, rate={self.sample_rate_sec:.2f}s)"
        )

        return final_trend, growth_coefficient, slope_full, gradient_recent

    # Alias retrocompatibil (vechiul nume).
    def get_trend(self):
        return self.get_instant_trend()

    def calculate_acceleration(self, window_sec: float = 90.0) -> tuple[float, float]:
        """Calculate recent price velocity (%/min) and acceleration (%/min^2).

        Fits a quadratic polynomial y(t) = a*t^2 + b*t + c over the latest samples,
        where t is in minutes and y is percentage move relative to window start.
        Returns (velocity, acceleration):
          velocity: dy/dt at the end of the window (in % / min)
          acceleration: d^2y/dt^2 = 2*a (in % / min^2)
        """
        with self._lock:
            prices = list(self.prices)
        if len(prices) < 3:
            return 0.0, 0.0

        rate = max(float(self.sample_rate_sec or 1.0), 0.1)
        if window_sec is not None and window_sec > 0:
            n_samples = max(int(round(window_sec / rate)), 3)
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
            return velocity, acceleration
        except Exception:
            return 0.0, 0.0


class WindowAnalyzer:
    """Trading metrics composed from a ``PriceWindow``."""
    def __init__(self, window: "PriceWindow"):
        self.window = window

    def calculate_acceleration(self, window_sec: float = 90.0) -> tuple[float, float]:
        return self.window.calculate_acceleration(window_sec)

    def calculate_slope_max_min(self):
        w = self.window
        if len(w.sorted_prices) < 2:
            return 0
        min_price, min_index = w.get_min_and_index()
        max_price, max_index = w.get_max_and_index()
        if min_price is None or max_price is None or max_index == min_index:
            return 0
        return (max_price - min_price) / (max_index - min_index)

    def calculate_proximities(self, current_price):
        w = self.window
        min_price, _ = w.get_min_and_index()
        max_price, _ = w.get_max_and_index()
        if min_price is None or max_price is None or max_price == min_price:
            return 0, 0
        min_proximity = (current_price - min_price) / (max_price - min_price)
        max_proximity = (max_price - current_price) / (max_price - min_price)
        return max(min_proximity, 0), max(max_proximity, 0)

    def calculate_positions(self):
        w = self.window
        min_price, min_index = w.get_min_and_index()
        max_price, max_index = w.get_max_and_index()
        min_position = min_index / w.window_size if min_index is not None else None
        max_position = max_index / w.window_size if max_index is not None else None
        return min_position, max_position

    def check_price_change(self, threshold):
        w = self.window
        if len(w.prices) < 2:
            return 0, 1
        min_price = w.get_min()
        max_price = w.get_max()
        newest_price = w.prices[-1]

        price_diff_min = u.calculate_difference_percent(min_price, newest_price)
        price_diff_max = u.calculate_difference_percent(max_price, newest_price)
        price_diff_newest = max(price_diff_min, price_diff_max)

        if abs(price_diff_newest) >= threshold or u.are_close(price_diff_newest, threshold):
            _, min_index = w.get_min_and_index()
            _, max_index = w.get_max_and_index()
            print(f'price_diff_minmax_versus_newest(slope)={price_diff_newest}(threshold={threshold}) '
                  f'are_close={u.are_close(price_diff_newest, threshold)}')
            print(f'min price ={min_price}, max_price = {max_price}, newest_price={newest_price}, '
                  f'min_index={min_index}, max_index={max_index}')
            return -price_diff_newest if price_diff_max > price_diff_min else price_diff_newest, 0
        return 0, 0

    def _analyze_price_movement(self, min_price, min_index, max_price, max_index,
                                newest_price, newest_index, price_diff):
        # Retain the earlier complex logic for possible future development.
        price_diff_min = u.calculate_difference_percent(min_price, newest_price)
        price_diff_max = u.calculate_difference_percent(max_price, newest_price)
        grow = price_diff_max < price_diff_min

        slope_min = u.slope(min_price, min_index, newest_price, newest_index)
        slope_max = u.slope(max_price, max_index, newest_price, newest_index)
        slope_max_min = slope_max if abs(slope_max) > abs(slope_min) else slope_min
        print(f"retun1 {slope_max_min}, {price_diff}")
        return slope_max_min, price_diff

        # Intentionally preserved legacy code below.
        diff_min_max_close = u.are_close(price_diff_max, price_diff_min, 1.0)
        if diff_min_max_close:
            if min_index < max_index:
                grow = 1
                print(f"retun2 {-slope_max}, {price_diff}")
                return -slope_max, price_diff
            else:
                grow = 0
                print(f"retun3 {slope_min}, {price_diff}")
                return slope_min, price_diff

        min_position, max_position = self.calculate_positions()
        min_loc = 1
        if min_position < 0.3 or u.are_close(min_position, 0.3):
            min_loc = 0
        if min_position > 0.7 or u.are_close(min_position, 0.7):
            min_loc = 2

        max_loc = 1
        if max_position > 0.7 or u.are_close(max_position, 0.7):
            max_loc = 2
        if max_position < 0.3 or u.are_close(max_position, 0.3):
            max_loc = 0

        if grow:
            if min_loc == 0:
                return slope_min, price_diff
            if min_loc == 1 and max_loc == 2:
                return slope_max_min, price_diff
            else:
                print("OUTLIER!! but can indicate something will come!!!")
        else:
            if min_loc == 2:
                return slope_max, price_diff
            if min_loc == 1 and max_loc == 0:
                return slope_max_min, price_diff
            else:
                print("OUTLIER!! but can indicate something will come!!!")

        return 0, 1

    def evaluate_buy_sell_opportunity(self, current_price, threshold_percent=1, decrease_percent=3.7):
        w = self.window
        slope = self.calculate_slope_max_min()
        min_price, min_index = w.get_min_and_index()
        max_price, max_index = w.get_max_and_index()
        print(f"Min price: {min_price} at index: {min_index} Max price: {max_price} at index: {max_index}")

        price_change_percent = (max_price - min_price) / min_price * 100 if min_price and max_price else 0
        print(f"Price change percent: {price_change_percent:.2f} slope: {slope:.4f} "
              f"Market trending: {'upwards' if slope > 0 else 'downwards'}")

        if price_change_percent < threshold_percent and not u.are_close(price_change_percent, threshold_percent):
            return 'HOLD', current_price, price_change_percent, slope

        min_position, max_position = self.calculate_positions()
        if slope > 0:
            if max_position > 0.8 or u.are_close(max_position, 0.8, target_tolerance_percent=1.0):
                proposed_price = current_price * 0.995
                return 'BUY', proposed_price, price_change_percent, slope
        else:
            if min_position < 0.2 or u.are_close(min_position, 0.2, target_tolerance_percent=1.0):
                proposed_price = current_price * 1.005
                return 'SELL', proposed_price, price_change_percent, slope

        remaining_decrease_percent = max(0, decrease_percent - price_change_percent)
        proposed_price = current_price * (1 - remaining_decrease_percent / 100)
        return 'BUY', proposed_price, price_change_percent, slope
