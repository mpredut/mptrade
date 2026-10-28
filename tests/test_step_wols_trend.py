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

    def test_per_symbol_fast_trend_window_tao_90s(self):
        import cachetrend
        import monitortrades as mt
        self.assertEqual(pw.fast_trend_window_for_symbol("TAOUSDC"), 90.0)
        self.assertEqual(pw.fast_trend_window_for_symbol("TAO"), 90.0)
        self.assertIsNone(pw.fast_trend_window_for_symbol("BTCUSDC"))
        self.assertIsNone(pw.fast_trend_window_for_symbol("HYPEUSD"))
        self.assertIn(90.0, cachetrend.CM_PUBLISHED_TREND_WINDOWS_SEC)

        queried = []

        class _FakeMgr:
            @staticmethod
            def get_short_trend_for_window(sym, win_sec, now=None):
                queried.append((sym, win_sec))
                return {"growth_coefficient": 0.5, "epsilon": 0.1, "n_samples": 11}

            @staticmethod
            def is_trend_up(sym, window_seconds=None, now=None):
                queried.append(("fallback", sym, window_seconds))
                return False

        import cacheManager as cm
        with patch.object(cm, "get_short_trend_manager", return_value=_FakeMgr()):
            self.assertTrue(mt.is_trend_up("TAOUSDC"))
            self.assertEqual(queried, [("TAOUSDC", 90.0)])
            queried.clear()
            self.assertFalse(mt.is_trend_up("BTCUSDC"))
            self.assertEqual(queried, [("fallback", "BTCUSDC", None)])

    def test_replay_trend_source_fast_window_and_dynamic_noise_gate(self):
        from offline.research.monitortrades_backtest.replay_trend_source import (
            DynamicReplayTrendSource,
            ReplayTrendSource,
        )
        rts = ReplayTrendSource(["TAOUSDC", "BTCUSDC"])
        self.assertEqual(rts._windows["TAOUSDC"].window_seconds, 90.0)
        self.assertAlmostEqual(rts._windows["BTCUSDC"].window_seconds, 222.0, places=6)

        drts = DynamicReplayTrendSource(["TAOUSDC"])
        # Oscillating noisy prices with tiny positive final step: gc is tiny (< 0.25 * eps) -> False
        for i, p in enumerate([100.0, 102.0, 98.0, 102.0, 98.0, 100.01]):
            drts.advance("TAOUSDC", 1000.0 + i * 10.0, p)
        self.assertFalse(drts.is_trend_up_at("TAOUSDC", 60.0))
        self.assertTrue(drts.should_wait("TAOUSDC", "B", 60.0, use_noise_gate=True))

    def test_calculate_acceleration_o1_orthogonal_polynomial_parity(self):
        rng = np.random.default_rng(19)
        sample_px = [float(x) for x in (280.0 + np.cumsum(rng.normal(0.05, 0.2, 200)))]
        win = pw.PriceWindow("TAOUSDC", 150, sample_rate_sec=1.0, window_seconds=150.0)
        for px in sample_px:
            win.process_price(px)
        for w_sec in (10.0, 90.0, 180.0, None):
            vel_fast, acc_fast = win.calculate_acceleration(window_sec=w_sec)
            prices = list(win.prices)
            if w_sec is not None:
                n_s = max(int(round(float(w_sec) / 1.0)), 3)
                prices = prices[-n_s:]
            t_min = np.arange(len(prices)) * (1.0 / 60.0)
            y_pct = ((np.array(prices) - prices[0]) / prices[0]) * 100.0
            a, b, _ = np.polyfit(t_min, y_pct, 2)
            vel_ref = float(2.0 * a * t_min[-1] + b)
            acc_ref = float(2.0 * a)
            self.assertAlmostEqual(vel_fast, vel_ref, places=11)
            self.assertAlmostEqual(acc_fast, acc_ref, places=11)

    def test_trend_gate_deferred_uses_fast_window_for_taousdc(self):
        import cacheManager as cm
        import monitortrades as mt
        mgr = cm.CacheShortTrendManager(["TAOUSDC"])
        queried = []
        orig_get_for_win = mgr.get_short_trend_for_window
        orig_should_wait = mgr.should_wait

        def fake_get_for_win(symbol, window_sec=None):
            if symbol == "TAOUSDC" and window_sec == 90.0:
                return {"growth_coefficient": -0.05, "epsilon": 0.01}
            return None

        def fake_should_wait(side, symbol, window_seconds=None, use_noise_gate=False):
            queried.append((side, symbol, window_seconds, use_noise_gate))
            return False

        with unittest.mock.patch.object(cm, "get_short_trend_manager", return_value=mgr), \
             unittest.mock.patch.object(mgr, "get_short_trend_for_window", side_effect=fake_get_for_win), \
             unittest.mock.patch.object(mgr, "should_wait", side_effect=fake_should_wait):
            self.assertFalse(mt._is_trend_gate_deferred("TAOUSDC", side="SELL"))
            self.assertEqual(queried, [("SELL", "TAOUSDC", 90.0, False)])

    def test_market_api_regime_service_for_reuses_and_caches_services(self):
        from providers.market_api import MarketApi

        class DummyProvider:
            name = "dummy"
            def supports_symbol(self, s):
                return True

        api_inst = MarketApi([DummyProvider()])
        s_default = api_inst._regime_service_for(None)
        s_same = api_inst._regime_service_for(s_default.evaluator.strength_threshold)
        self.assertIs(s_same, s_default)

        s_custom1 = api_inst._regime_service_for(1.5)
        s_custom2 = api_inst._regime_service_for(1.5)
        self.assertIsNot(s_custom1, s_default)
        self.assertIs(s_custom1, s_custom2)

    def test_survival_flat_window_slopes_and_episodes(self):
        from intelligence.internal.state.survival import block_slopes, episodes
        ts = np.arange(0.0, 200.0 * 3600.0, 3600.0)
        px_flat = np.full_like(ts, 100.0)
        bt, bs = block_slopes(ts, px_flat, window_h=24, step_h=8)
        self.assertTrue(len(bs) > 0)
        self.assertTrue(np.all(bs == 0.0))
        eps = episodes(bt, bs, window_h=24)
        self.assertEqual(eps, [])

    def test_hurst_and_forecast_linear_closed_form_ols_parity(self):
        from intelligence.internal.state.persistence import calculate_hurst_exponent
        from visuals.drawprices import forecast_linear
        rng = np.random.default_rng(42)
        prices = [float(x) for x in (1000.0 + np.cumsum(rng.normal(0.1, 1.5, 500)))]
        h = calculate_hurst_exponent(prices)
        self.assertIsNotNone(h)
        self.assertTrue(0.0 <= h <= 1.0)

        t_in = np.linspace(1000.0, 2000.0, 50)
        p_in = 50.0 + 0.025 * (t_in - t_in[0]) + rng.normal(0.0, 0.2, 50)
        fp = forecast_linear(t_in, p_in, horizon=10)
        x_in = np.arange(len(p_in))
        m_ref, b_ref = np.polyfit(x_in, p_in, 1)
        x_future = np.arange(len(p_in), len(p_in) + 10)
        np.testing.assert_allclose(fp, m_ref * x_future + b_ref, rtol=1e-11, atol=1e-11)

    def test_kalman_state_tracker_current_trend_normalized_lookup(self):
        from shadow_signals import KalmanStateTracker
        tracker = KalmanStateTracker(["TAOUSDC"])
        tracker._now = lambda: 1010.0
        tracker._state["TAOUSDC"] = {"ts": 1000.0, "kalman_trend": 1.0}
        self.assertEqual(tracker.current_trend(" taousdc "), (1, 10.0))
        self.assertEqual(tracker.current_trend("TAOUSDC"), (1, 10.0))

    def test_assetguardian_trend_defer_uses_fast_window_and_mem_published_window(self):
        import time as _time
        import assetguardian as ag
        import cacheManager as cm

        now_ts = _time.time()
        mgr = cm.CacheShortTrendManager(["TAOUSDC"], writer=False)
        mgr._set_mem(
            "TAOUSDC",
            growth_coefficient=0.05,
            gradient_recent=0.05,
            epsilon=0.01,
            current_price=350.0,
            ts=now_ts,
            dynamic={
                "90": {
                    "growth_coefficient": -0.08,
                    "gradient_recent": -0.08,
                    "epsilon": 0.01,
                    "n_samples": 90,
                    "window_seconds": 90.0,
                    "current_price": 350.0,
                    "ts": now_ts,
                }
            },
        )
        # In-memory _mem dynamic["90"] is negative -> BUY should defer on 90s window even though 222s top-level is positive
        with unittest.mock.patch.object(cm, "get_short_trend_manager", return_value=mgr), \
             unittest.mock.patch.object(ag, "_save_symbol_campaign"):
            ready, new_state = ag._trend_defer_ready("TAOUSDC", "buy", "BUY", 5.0, {})
            self.assertFalse(ready)
            self.assertIn("trend_defer", new_state)


if __name__ == "__main__":
    unittest.main()
