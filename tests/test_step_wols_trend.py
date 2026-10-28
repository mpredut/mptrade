"""Step-WOLS short-trend estimator test suite (pricewindow SSOT).

Pipeline: raw price -> growth_coefficient = sum(k^p * dy_k) / sum(k^p) over the window's steps.
Per-symbol power from instruments.conf (trend.step_wols_power):
  - BTC (BTCUSDC): p=3.0 (Cubic Step-WOLS)
  - Altcoins / Equities (TAOUSDC, HYPEUSD, ARBUSDC, ...): p=2.0 (Quadratic Step-WOLS)
Supports p=1 (Linear), p=2 (Quadratic), p=3 (Cubic), and arbitrary powers p >= 0.
"""
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pricewindow as pw


def _naive_step_wols(prices, power=None, symbol=None):
    pwr = float(pw.step_wols_power_for_symbol(symbol) if power is None else power)
    arr = np.asarray(prices, dtype=float)
    m = len(arr) - 1
    if m < 1:
        return 0.0
    k = np.arange(1, m + 1, dtype=float) ** pwr if pwr != 1.0 else np.arange(1, m + 1, dtype=float)
    return float(np.dot(k, np.diff(arr)) / k.sum())


class TestSymbolPowerResolution(unittest.TestCase):
    def test_per_symbol_power_from_instruments_conf(self):
        self.assertEqual(pw.step_wols_power_for_symbol("BTCUSDC"), 3.0)
        self.assertEqual(pw.step_wols_power_for_symbol("BTCUSDT"), 3.0)
        self.assertEqual(pw.step_wols_power_for_symbol("BTC"), 3.0)
        self.assertEqual(pw.step_wols_power_for_symbol("TAOUSDC"), 2.0)
        self.assertEqual(pw.step_wols_power_for_symbol("HYPEUSD"), 2.0)
        self.assertEqual(pw.step_wols_power_for_symbol("HYPEUSDC"), 2.0)
        self.assertEqual(pw.step_wols_power_for_symbol("ARBUSDC"), 2.0)
        self.assertEqual(pw.step_wols_power_for_symbol(None), pw.DEFAULT_STEP_WOLS_POWER)
        self.assertEqual(pw.step_wols_power_for_symbol("UNKNOWN_COIN"), pw.DEFAULT_STEP_WOLS_POWER)

    def test_env_and_patch_overrides_take_precedence(self):
        with patch.dict(os.environ, {"MPTRADE_STEP_WOLS_POWER": "1.5"}):
            self.assertEqual(pw.step_wols_power_for_symbol("BTCUSDC"), 1.5)
            self.assertEqual(pw.step_wols_power_for_symbol("TAOUSDC"), 1.5)
        with patch.object(pw, "STEP_WOLS_POWER", 3.0):
            self.assertEqual(pw.step_wols_power_for_symbol("TAOUSDC"), 3.0)


class TestStepWolsGeneralization(unittest.TestCase):
    def test_reference_matches_naive_definition_across_powers(self):
        rng = np.random.default_rng(1)
        prices = 60000 + np.cumsum(rng.normal(0, 5, 300))
        for pwr in (1.0, 2.0, 3.0):
            for sr in (1.0, 1.5, 17.0):
                got = pw.step_wols_slope(prices, sr, power=pwr)
                expected = _naive_step_wols(prices, power=pwr)
                self.assertAlmostEqual(got, expected, places=9)

    def test_buffer_matches_reference_on_subslices_after_trim_and_long_windows(self):
        rng = np.random.default_rng(2)
        buf = pw.IncrementalTrendBuffer()
        hist = 300.0 + np.cumsum(rng.normal(0, 0.05, 3000))
        for px in hist:
            buf.append(float(px))
        buf.trim_front(700)
        n = len(buf.prices)
        for pwr in (1.0, 2.0, 3.0):
            with patch.object(pw, "STEP_WOLS_POWER", pwr):
                for a, b, sr in ((0, n - 1, 0.8), (n - 222, n - 1, 1.0), (n - 1500, n - 1, 1.0),
                                 (5, 9, 1.0), (100, 101, 1.5), (40, 1300, 17.0)):
                    got = buf.step_wols_slope(a, b, sr, power=pwr)
                    ref = pw.step_wols_slope(buf.prices[a:b + 1], sr, power=pwr)
                    self.assertAlmostEqual(got, ref, delta=1e-9 * max(1.0, abs(ref)))
                    res = buf.eval_slice(a, b, sample_rate_sec=sr, power=pwr)
                    self.assertAlmostEqual(res[1], got, places=12)
                    self.assertEqual(res[0], 1 if got > 0 else (-1 if got < 0 else 0))

    def test_backward_compatible_aliases(self):
        rng = np.random.default_rng(123)
        prices = 50000 + np.cumsum(rng.normal(0, 2, 100))
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
        for sym in (None, "BTCUSDC", "TAOUSDC"):
            for a, b in ((0, 599), (378, 599), (590, 599)):
                ft, gc, sf, gr, _ = buf.eval_slice(a, b, sample_rate_sec=1.0, symbol=sym)
                eft, egc, esf, egr = pw.instant_trend_from_prices(prices[a:b + 1], 1.0, symbol=sym)
                self.assertEqual(ft, eft)
                self.assertAlmostEqual(gc, egc, delta=1e-9 * max(1.0, abs(egc)))
                self.assertAlmostEqual(sf, esf, places=8)
                self.assertAlmostEqual(gr, egr, places=8)

    def test_single_tick_spike_does_not_flip_a_downtrend(self):
        prices = [60000.0 - 2.0 * i for i in range(222)]
        prices[-1] = prices[-2] + 30.0   # isolated 15x step bounce on the newest tick
        ft, gc, _, _ = pw.instant_trend_from_prices(prices, 1.0)
        self.assertEqual(ft, -1)
        self.assertLess(gc, 0.0)

    def test_constant_and_short_series_are_neutral(self):
        self.assertEqual(pw.step_wols_slope([100.0] * 50, 1.0), 0.0)
        self.assertEqual(pw.step_wols_slope([100.0], 1.0), 0.0)
        buf = pw.IncrementalTrendBuffer()
        for _ in range(50):
            buf.append(100.0)
        self.assertEqual(buf.eval_slice(0, 49, sample_rate_sec=1.0)[:2], (0, 0.0))

    def test_price_window_uses_symbol_power_and_respects_patch(self):
        rng = np.random.default_rng(4)
        sample_px = [float(x) for x in (60000 + np.cumsum(rng.normal(0, 5, 222)))]

        win_btc = pw.PriceWindow("BTCUSDC", 300, sample_rate_sec=1.0, window_seconds=222.0)
        win_tao = pw.PriceWindow("TAOUSDC", 300, sample_rate_sec=1.0, window_seconds=222.0)
        for px in sample_px:
            win_btc.process_price(px)
            win_tao.process_price(px)

        _, gc_btc, _, _ = win_btc.get_short_trend()
        _, gc_tao, _, _ = win_tao.get_short_trend()
        self.assertAlmostEqual(gc_btc, pw.step_wols_slope(sample_px, 1.0, power=3.0), places=9)
        self.assertAlmostEqual(gc_tao, pw.step_wols_slope(sample_px, 1.0, power=2.0), places=9)

        with patch.object(pw, "STEP_WOLS_POWER", 1.0):
            _, gc_lin, _, _ = win_tao.get_short_trend()
            self.assertAlmostEqual(gc_lin, pw.step_wols_slope(sample_px, 1.0, power=1.0), places=9)

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

    def test_noise_gating_neutralizes_sub_epsilon_slopes(self):
        prices = [100.0 + 0.001 * (i % 2) for i in range(222)]
        ft_raw, gc, _, _ = pw.instant_trend_from_prices(prices, 1.0)
        with patch.dict(os.environ, {"MPTRADE_TREND_NOISE_GATE_K": "0.5"}):
            ft_gated, _, _, _ = pw.instant_trend_from_prices(prices, 1.0)
            self.assertEqual(ft_gated, 0)

    def test_scale_invariant_upward_formula(self):
        from cachetrend import CacheShortTrendManager
        # Sparse window (n_samples=11 -> m=10): threshold is 0.25 * eps = 0.25
        self.assertFalse(CacheShortTrendManager._eval_upward_formula(
            {"growth_coefficient": 0.10, "epsilon": 1.0, "n_samples": 11}
        ))
        self.assertTrue(CacheShortTrendManager._eval_upward_formula(
            {"growth_coefficient": 0.30, "epsilon": 1.0, "n_samples": 11}
        ))
        # Dense 1-second window (n_samples=222 -> m=221): threshold scales by sqrt(10/221) ≈ 0.2127 -> ~0.0532
        self.assertTrue(CacheShortTrendManager._eval_upward_formula(
            {"growth_coefficient": 0.10, "epsilon": 1.0, "n_samples": 222}
        ))
        self.assertFalse(CacheShortTrendManager._eval_upward_formula(
            {"growth_coefficient": 0.04, "epsilon": 1.0, "n_samples": 222}
        ))

    def test_get_recent_gradient_o1_telescoped_parity(self):
        rng = np.random.default_rng(7)
        for n_pts in (2, 15, 60, 222):
            sample_px = [float(x) for x in (300.0 + np.cumsum(rng.normal(0, 0.4, n_pts)))]
            win = pw.PriceWindow("TAOUSDC", 300, sample_rate_sec=1.0, window_seconds=222.0)
            for px in sample_px:
                win.process_price(px)
            grad_ref = np.gradient(np.array(sample_px))
            r = win.recent_n
            expected = float(np.mean(grad_ref[-r:])) if len(grad_ref) >= r else float(np.mean(grad_ref))
            self.assertAlmostEqual(win.get_recent_gradient(), expected, places=11)
            # Also verify fallback when prices deque is replaced directly
            win.prices.append(sample_px[-1] + 1.25)
            grad_ref2 = np.gradient(np.array(list(win.prices)))
            expected2 = float(np.mean(grad_ref2[-r:])) if len(grad_ref2) >= r else float(np.mean(grad_ref2))
            self.assertAlmostEqual(win.get_recent_gradient(), expected2, places=11)

    def test_volatility_1h_uses_o1_trend_buf_via_price_deque(self):
        from intelligence.internal.state.volatility import calculate_volatility_1h
        rng = np.random.default_rng(11)
        sample_px = [float(x) for x in (60000.0 + np.cumsum(rng.normal(0, 15.0, 400)))]
        win = pw.PriceWindow("BTCUSDC", 500, sample_rate_sec=1.0, window_seconds=9000.0)
        for px in sample_px:
            win.process_price(px)
        self.assertIs(getattr(win.prices, "_trend_buf", None), win._trend_buf)
        v_fast = calculate_volatility_1h(win.prices, win.sample_rate_sec)
        v_list = calculate_volatility_1h(list(win.prices), win.sample_rate_sec)
        self.assertIsNotNone(v_fast)
        self.assertAlmostEqual(v_fast, v_list, places=4)

    def test_rtrade_effective_trend_filter_k_scale_invariance(self):
        import rtrade
        with patch.object(rtrade, "RTRADE_TREND_FILTER_K", 2.0):
            self.assertEqual(rtrade._effective_trend_filter_k(None), 2.0)
            self.assertEqual(rtrade._effective_trend_filter_k({}), 2.0)
            self.assertEqual(rtrade._effective_trend_filter_k({"n_samples": 11}), 2.0)
            self.assertAlmostEqual(
                rtrade._effective_trend_filter_k({"n_samples": 901}),
                2.0 * math.sqrt(10.0 / 900.0),
                places=9,
            )


if __name__ == "__main__":
    unittest.main()
