"""Step-WOLS cubic short-trend estimator (pricewindow SSOT, Oct 27).

Pipeline: raw price -> adaptive causal median (~5s, bypassed above 2s sampling)
-> growth_coefficient = sum(k^3 * dy_k) / sum(k^3) over the window's steps.
"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pricewindow as pw


def _naive_step_wols(prices, kernel):
    arr = np.asarray(prices, dtype=float)
    filt = np.array([np.median(arr[max(0, i - kernel + 1): i + 1]) for i in range(len(arr))])
    m = len(filt) - 1
    k = np.arange(1, m + 1, dtype=float) ** 3
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


class TestStepWolsCubic(unittest.TestCase):
    def test_reference_matches_naive_definition(self):
        rng = np.random.default_rng(1)
        prices = 60000 + np.cumsum(rng.normal(0, 5, 300))
        for sr, kern in ((1.0, 5), (1.5, 3), (17.0, 1)):
            self.assertAlmostEqual(pw.step_wols_cubic_slope(prices, sr), _naive_step_wols(prices, kern), places=9)

    def test_buffer_matches_reference_on_subslices_after_trim_and_long_windows(self):
        rng = np.random.default_rng(2)
        buf = pw.IncrementalTrendBuffer()
        hist = 300.0 + np.cumsum(rng.normal(0, 0.05, 3000))
        for px in hist:
            buf.append(float(px))
        buf.trim_front(700)
        n = len(buf.prices)
        for a, b, sr in ((0, n - 1, 0.8), (n - 222, n - 1, 1.0), (n - 1500, n - 1, 1.0),
                         (5, 9, 1.0), (100, 101, 1.5), (40, 1300, 17.0)):
            got = buf.step_wols_cubic(a, b, sr)
            ref = pw.step_wols_cubic_slope(buf.prices[a:b + 1], sr)
            self.assertAlmostEqual(got, ref, delta=1e-9 * max(1.0, abs(ref)))
            res = buf.eval_slice(a, b, sample_rate_sec=sr)
            self.assertAlmostEqual(res[1], got, places=12)
            self.assertEqual(res[0], 1 if got > 0 else (-1 if got < 0 else 0))

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
        with patch.object(pw, "SHORT_TREND_ESTIMATOR", pw.SHORT_TREND_ESTIMATOR_LEGACY):
            legacy_ft, legacy_gc, _, _ = pw.instant_trend_from_prices(prices, 1.0)
        ft, gc, _, _ = pw.instant_trend_from_prices(prices, 1.0)
        self.assertEqual(legacy_ft, 1)   # legacy flips to UP on one bad tick
        self.assertEqual(ft, -1)         # median + Step-WOLS keeps the real direction
        self.assertLess(gc, 0.0)

    def test_constant_and_short_series_are_neutral(self):
        self.assertEqual(pw.step_wols_cubic_slope([100.0] * 50, 1.0), 0.0)
        self.assertEqual(pw.step_wols_cubic_slope([100.0], 1.0), 0.0)
        buf = pw.IncrementalTrendBuffer()
        for _ in range(50):
            buf.append(100.0)
        self.assertEqual(buf.eval_slice(0, 49, sample_rate_sec=1.0)[:2], (0, 0.0))

    def test_price_window_cache_respects_estimator_switch(self):
        win = pw.PriceWindow("BTCUSDC", 300, sample_rate_sec=1.0, window_seconds=222.0)
        rng = np.random.default_rng(4)
        for px in 60000 + np.cumsum(rng.normal(0, 5, 222)):
            win.process_price(float(px))
        _, gc_new, sf, gr = win.get_short_trend()
        with patch.object(pw, "SHORT_TREND_ESTIMATOR", pw.SHORT_TREND_ESTIMATOR_LEGACY):
            _, gc_legacy, _, _ = win.get_short_trend()
        self.assertAlmostEqual(gc_legacy, (sf + gr) / 2.0, places=10)
        self.assertAlmostEqual(gc_new, pw.step_wols_cubic_slope(list(win.prices), 1.0), places=9)

    def test_env_resolution_defaults_to_step_wols_and_accepts_legacy(self):
        with patch.dict(os.environ, {"MPTRADE_SHORT_TREND_ESTIMATOR": "legacy"}):
            self.assertEqual(pw._resolve_short_trend_estimator(), "legacy")
        with patch.dict(os.environ, {"MPTRADE_SHORT_TREND_ESTIMATOR": "bogus"}):
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols_cubic")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MPTRADE_SHORT_TREND_ESTIMATOR", None)
            self.assertEqual(pw._resolve_short_trend_estimator(), "step_wols_cubic")


if __name__ == "__main__":
    unittest.main()
