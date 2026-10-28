"""Step-WOLS short-trend estimator test suite (pricewindow SSOT, Oct 28).

Pipeline: raw price -> adaptive causal median (~5s, bypassed above 2s sampling)
-> growth_coefficient = sum(k^p * dy_k) / sum(k^p) over the window's steps.
Default power p=2 (Quadratic: empirical #1 champion on 11.26M ticks, +$8.04 vs legacy).
Supports p=1 (Linear), p=2 (Quadratic), p=3 (Cubic), and arbitrary powers p >= 0.
"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pricewindow as pw


def _naive_step_wols(prices, kernel, power=None):
    pwr = float(pw.STEP_WOLS_POWER if power is None else power)
    arr = np.asarray(prices, dtype=float)
    filt = np.array([np.median(arr[max(0, i - kernel + 1): i + 1]) for i in range(len(arr))])
    m = len(filt) - 1
    if m < 1:
        return 0.0
    k = np.arange(1, m + 1, dtype=float) ** pwr if pwr != 1.0 else np.arange(1, m + 1, dtype=float)
    return float(np.dot(k, np.diff(filt)) / k.sum())


class TestMedianKernel(unittest.TestCase):
    def test_kernel_is_defined_in_seconds_and_bypassed_for_sparse_data(self):
        self.assertEqual(pw.median_filter_kernel(0.8), 5)
        self.assertEqual(pw.median_filter_kernel(1.0), 5)
        self.assertEqual(pw.median_filter_kernel(1.5), 3)
        self.assertEqual(pw.median_filter_kernel(2.0), 3)
        self.assertEqual(pw.median_filter_kernel(2.5), 1)
        self.assertEqual(pw.median_filter_kernel(17.0), 1)
        for bad in (None, True, float("nan"), -1.0, 0.0, "x"):
            self.assertEqual(pw.median_filter_kernel(bad), 5)  # DEFAULT_SAMPLE_RATE_SEC=0.8
        self.assertEqual(pw.median_filter_kernel(1e-309), 5)   # subnormal rate must not overflow

    def test_explicit_median_seconds_override(self):
        self.assertEqual(pw.median_filter_kernel(1.0, median_seconds=3.0), 3)
        self.assertEqual(pw.median_filter_kernel(1.0, median_seconds=0.0), 1)
        self.assertEqual(pw.median_filter_kernel(1.0, median_seconds=5.0), 5)


class TestStepWolsGeneralization(unittest.TestCase):
    def test_reference_matches_naive_definition_across_powers(self):
        rng = np.random.default_rng(1)
        prices = 60000 + np.cumsum(rng.normal(0, 5, 300))
        for pwr in (1.0, 2.0, 3.0):
            for sr, kern in ((1.0, 5), (1.5, 3), (17.0, 1)):
                got = pw.step_wols_slope(prices, sr, power=pwr)
                expected = _naive_step_wols(prices, kern, power=pwr)
                self.assertAlmostEqual(got, expected, places=9)

    def test_buffer_matches_reference_on_subslices_after_trim_and_long_windows(self):
        rng = np.random.default_rng(2)
        buf = pw.IncrementalTrendBuffer()
        hist = 300.0 + np.cumsum(rng.normal(0, 0.05, 3000))
        for px in hist:
            buf.append(float(px))
        buf.trim_front(700)
        n = len(buf.prices)
        for pwr in (2.0, 3.0):
            with patch.object(pw, "STEP_WOLS_POWER", pwr):
                for a, b, sr in ((0, n - 1, 0.8), (n - 222, n - 1, 1.0), (n - 1500, n - 1, 1.0),
                                 (5, 9, 1.0), (100, 101, 1.5), (40, 1300, 17.0)):
                    got = buf.step_wols_slope(a, b, sr, power=pwr)
                    ref = pw.step_wols_slope(buf.prices[a:b + 1], sr, power=pwr)
                    self.assertAlmostEqual(got, ref, delta=1e-9 * max(1.0, abs(ref)))
                    res = buf.eval_slice(a, b, sample_rate_sec=sr)
                    self.assertAlmostEqual(res[1], got, places=12)
                    self.assertEqual(res[0], 1 if got > 0 else (-1 if got < 0 else 0))

    def test_backward_compatible_aliases(self):
        rng = np.random.default_rng(123)
        prices = 50000 + np.cumsum(rng.normal(0, 2, 100))
        # step_wols_cubic_slope delegates to step_wols_slope
        self.assertEqual(pw.step_wols_cubic_slope(prices, 1.0), pw.step_wols_slope(prices, 1.0))
        buf = pw.IncrementalTrendBuffer()
        for p in prices:
            buf.append(float(p))
        self.assertEqual(buf.step_wols_cubic(0, len(prices) - 1, 1.0), buf.step_wols_slope(0, len(prices) - 1, 1.0))

    def test_eval_slice_matches_instant_trend_from_prices(self):
        rng = np.random.default_rng(3)
        prices = list(111000 + np.cumsum(rng.normal(0, 20, 600)))
        buf = pw.IncrementalTrendBuffer()
        for px in prices:
            buf.append(px)
        for a, b in ((0, 599), (378, 599), (590, 599)):
            ft, gc, sf, gr, _ = buf.eval_slice(a, b, sample_rate_sec=1.0)
            eft, egc, esf, egr = pw.instant_trend_from_prices(prices[a:b + 1], 1.0)
            self.assertEqual(ft, eft)
            self.assertAlmostEqual(gc, egc, delta=1e-9 * max(1.0, abs(egc)))
            self.assertAlmostEqual(sf, esf, places=8)
            self.assertAlmostEqual(gr, egr, places=8)

    def test_single_tick_spike_does_not_flip_a_downtrend(self):
        prices = [60000.0 - 2.0 * i for i in range(222)]
        prices[-1] = prices[-2] + 300.0  # isolated wick / bid-ask bounce on the newest tick
        ft, gc, _, _ = pw.instant_trend_from_prices(prices, 1.0)
        self.assertEqual(ft, -1)         # median + Step-WOLS keeps the real direction
        self.assertLess(gc, 0.0)

    def test_constant_and_short_series_are_neutral(self):
        self.assertEqual(pw.step_wols_slope([100.0] * 50, 1.0), 0.0)
        self.assertEqual(pw.step_wols_slope([100.0], 1.0), 0.0)
        buf = pw.IncrementalTrendBuffer()
        for _ in range(50):
            buf.append(100.0)
        self.assertEqual(buf.eval_slice(0, 49, sample_rate_sec=1.0)[:2], (0, 0.0))

    def test_price_window_cache_respects_estimator_switch(self):
        win = pw.PriceWindow("BTCUSDC", 300, sample_rate_sec=1.0, window_seconds=222.0)
        rng = np.random.default_rng(4)
        for px in 60000 + np.cumsum(rng.normal(0, 5, 222)):
            win.process_price(float(px))
        _, gc_quad, _, _ = win.get_short_trend()
        self.assertAlmostEqual(gc_quad, pw.step_wols_slope(list(win.prices), 1.0, power=2.0), places=9)
        with patch.object(pw, "STEP_WOLS_POWER", 3.0):
            win._short_trend_cache = None
            _, gc_cubic, _, _ = win.get_short_trend()
            self.assertAlmostEqual(gc_cubic, pw.step_wols_slope(list(win.prices), 1.0, power=3.0), places=9)

    def test_env_resolution_defaults_to_step_wols_and_accepts_aliases(self):
        with patch.dict(os.environ, {"MPTRADE_SHORT_TREND_ESTIMATOR": "step_wols_cubic"}):
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols_cubic")
            self.assertEqual(pw._resolve_step_wols_power("step_wols_cubic"), 3.0)
        with patch.dict(os.environ, {"MPTRADE_SHORT_TREND_ESTIMATOR": "step_wols_linear"}):
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols_linear")
            self.assertEqual(pw._resolve_step_wols_power("step_wols_linear"), 1.0)
        with patch.dict(os.environ, {"MPTRADE_SHORT_TREND_ESTIMATOR": "step_wols"}):
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols")
            self.assertEqual(pw._resolve_step_wols_power("step_wols"), 2.0)
        with patch.dict(os.environ, {"MPTRADE_SHORT_TREND_ESTIMATOR": "bogus"}):
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MPTRADE_SHORT_TREND_ESTIMATOR", None)
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols")
        with patch.dict(os.environ, {"MPTRADE_STEP_WOLS_POWER": "3.5"}):
            self.assertEqual(pw._resolve_step_wols_power(), 3.5)


if __name__ == "__main__":
    unittest.main()
