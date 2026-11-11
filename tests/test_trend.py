#!/usr/bin/env python3
"""
Tests for detect_long_term_trend (priceAnalysis) — the cases that produced
incorrect reports (e.g. TAO: 4 days of drop reported as trend UP).

  python -m pytest tests/test_trend.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intelligence.internal.state.persistence import (  # noqa: E402
    calculate_hurst_exponent,
    calculate_mann_kendall,
    classify_hurst_regime,
)
from priceAnalysis import detect_long_term_trend, get_trade_weight

H = 3600.0


def series(*segments):
    """segments = lists of hourly prices; it returns (timestamps, prices)."""
    px = [p for seg in segments for p in seg]
    ts = [i * H for i in range(len(px))]
    return np.array(ts), np.array(px, float)


def run(ts, px, **kw):
    base = dict(window_hours=24, step_hours=8, min_consecutive_blocks=3,
                noise_tolerance=2, min_points_per_window=5)
    base.update(kw)
    return detect_long_term_trend(ts, px, **base)


def down(h, start=100.0, rate=0.3):
    return [start - rate * i for i in range(h)]


def up(h, start=100.0, rate=0.3):
    return [start + rate * i for i in range(h)]


class TestTAOCase(unittest.TestCase):
    """The reported case: a falling slope over 4 days but a 'current trend' of UP."""

    def test_a_one_day_bounce_against_the_fall_is_not_an_up_trend(self):
        d = down(96)                                  # 4 days fall
        b = up(24, start=d[-1], rate=0.25)            # 24h bounce
        r = run(*series(d, b))
        self.assertIsNone(r, "the 24h bounce is NOT a coherent UP trend — it must be None")

    def test_a_sustained_2_day_bounce_becomes_an_up_trend_with_the_right_duration(self):
        d = down(96)
        b = up(48, start=d[-1], rate=0.25)            # sustained 2-day bounce
        r = run(*series(d, b))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "up")
        dur_days = r["duration_seconds"] / 86400
        self.assertLess(dur_days, 2.6, "the duration must be ~2 days (the bounce), not 4-6")


class TestDuration(unittest.TestCase):
    def test_the_duration_does_not_exceed_the_data_span(self):
        ts, px = series(down(96))                     # 4 days of data
        r = run(ts, px)
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "down")
        self.assertLessEqual(r["duration_seconds"], ts[-1] - ts[0] + 1,
                             "the reported duration was longer than ALL the data (the old bug: +72h invented)")

    def test_pure_drop_gives_duration_close_to_span(self):
        ts, px = series(down(96))
        r = run(ts, px)
        self.assertGreater(r["duration_seconds"] / 86400, 3.0, "the real trend is ~4 days")


class TestNoise(unittest.TestCase):
    def test_noise_within_the_tolerance_does_not_break_the_trend(self):
        # a 2-day fall plus a noise window (a 16h rise) plus another 2-day fall
        a = down(48)
        z = up(16, start=a[-1], rate=0.1)
        b = down(48, start=z[-1])
        r = run(*series(a, z, b))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "down")
        self.assertGreater(r["duration_seconds"] / 86400, 3.5,
                           "tolerated noise must not cut the duration of the real trend")

    def test_clean_up_trend(self):
        r = run(*series(up(96)))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "up")


class TestDetectionLag(unittest.TestCase):
    """detection_lag_hours: trend starts before confirmation (heuristic ~2 days),
    explicitly and CAPPED at the data span (the duration cannot exceed the history)."""

    def test_the_lag_is_added_but_does_not_exceed_the_span(self):
        ts, px = series(down(96))                     # span = 95h
        r = run(ts, px, detection_lag_hours=48)
        self.assertIsNotNone(r)
        self.assertLessEqual(r["duration_seconds"], ts[-1] - ts[0] + 1,
                             "the lag must not push the duration beyond the existing data")
        self.assertGreater(r["duration_seconds"] / 3600, 90, "capped at the span, not cut below it")

    def test_the_lag_is_added_when_it_fits_in_the_span(self):
        d = down(96)
        u_ = up(72, start=d[-1], rate=0.25)           # confirmed up trend ~3 days
        r = run(*series(d, u_), detection_lag_hours=48)
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "up")
        dur_h = r["duration_seconds"] / 3600
        self.assertGreater(dur_h, 95, "the duration must include the lag (+48h)")
        self.assertLess(dur_h, 135)

    def test_without_the_lag_the_duration_is_strictly_confirmed(self):
        d = down(96)
        u_ = up(72, start=d[-1], rate=0.25)
        r0 = run(*series(d, u_), detection_lag_hours=0)
        r48 = run(*series(d, u_), detection_lag_hours=48)
        self.assertAlmostEqual(r48["duration_seconds"] - r0["duration_seconds"],
                               48 * 3600, delta=1.0)


class TestGaps(unittest.TestCase):
    def test_a_gap_in_the_data_stops_the_confirmation(self):
        # 2 days down, a 2-day GAP (no points), then 2 recent days down
        old = down(48, start=130)
        recent = down(48, start=115)
        ts_old = [i * H for i in range(48)]
        ts_recent = [(i + 96) * H for i in range(48)]     # gap between 48h and 96h
        ts = np.array(ts_old + ts_recent)
        px = np.array(old + recent, float)
        r = run(ts, px)
        if r is not None:
            self.assertLessEqual(r["duration_seconds"] / 3600, 50,
                                 "the trend must not stretch across the gap")

    def test_insufficient_data_returns_none(self):
        ts, px = series(down(10))
        self.assertIsNone(run(ts, px))


class TestMannKendall(unittest.TestCase):
    def test_monotonic_trends_are_significant(self):
        for label, delta, expected_sign in (("up", 0.5, 1), ("down", -0.5, -1)):
            with self.subTest(direction=label):
                _, z, p = calculate_mann_kendall([100 + delta * i for i in range(24)])
                self.assertGreater(z * expected_sign, 0)
                self.assertLess(p, 0.01)

    def test_white_noise_not_significant(self):
        rng = np.random.default_rng(42)
        s, z, p = calculate_mann_kendall(100 + rng.normal(0, 1, 24))
        self.assertGreater(p, 0.1, "pure noise must not look like a trend")

    def test_short_series_not_significant(self):
        _, _, p = calculate_mann_kendall([1, 2, 3])
        self.assertEqual(p, 1.0)


class TestHurst(unittest.TestCase):
    @staticmethod
    def _series(phi: float, n: int = 2000, seed: int = 7):
        rng = np.random.default_rng(seed)
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = phi * r[i - 1] + rng.normal(0, 0.01)
        return 100 * np.exp(np.cumsum(r))

    def test_persistent_and_mean_reverting_regimes(self):
        cases = (("persistent", 0.6, lambda h: h > 0.55),
                 ("mean-reverting", -0.6, lambda h: h < 0.45))
        for label, phi, predicate in cases:
            with self.subTest(regime=label):
                self.assertTrue(predicate(calculate_hurst_exponent(self._series(phi))))

    def test_short_series_returns_none(self):
        self.assertIsNone(calculate_hurst_exponent([100, 101, 102]))

    def test_regimes(self):
        self.assertEqual(classify_hurst_regime(0.65), "persistent")
        self.assertEqual(classify_hurst_regime(0.30), "mean_reverting")
        self.assertEqual(classify_hurst_regime(0.50), "random_walk")
        self.assertEqual(classify_hurst_regime(None), "unknown")


class TestMKFilterInDetector(unittest.TestCase):
    H = 3600.0

    def _mk_series(self, vals):
        return np.array([i * self.H for i in range(len(vals))]), np.array(vals, float)

    def test_clean_trend_passes_filter(self):
        ts, px = self._mk_series([100 - 0.3 * i for i in range(96)])
        r = detect_long_term_trend(ts, px, 24, 8, 3, 2, 5, mk_alpha=0.05)
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "down")

    def test_noise_is_filtered(self):
        rng = np.random.default_rng(3)
        base = [100 - 0.3 * i for i in range(72)]
        noise = list(base[-1] + rng.normal(0, 0.8, 24))
        ts, px = self._mk_series(base + noise)
        r = detect_long_term_trend(ts, px, 24, 8, 3, 2, 5, mk_alpha=0.05)
        self.assertIsNone(r, "a current window with no significant trend -> None")

    def test_without_the_filter_the_old_behaviour(self):
        ts, px = self._mk_series([100 - 0.3 * i for i in range(96)])
        r = detect_long_term_trend(ts, px, 24, 8, 3, 2, 5, mk_alpha=None)
        self.assertIsNotNone(r)


def build_series(climb_days=5, climb_from=233.0, peak=287.0, decline_to=265.0, decline_hours=36):
    ts, px, t = [], [], 0.0
    n_up = climb_days * 24
    for h in range(n_up):
        ts.append(t); px.append(climb_from + (peak - climb_from) * h / (n_up - 1)); t += H
    for h in range(1, decline_hours + 1):
        ts.append(t); px.append(peak - (peak - decline_to) * h / decline_hours); t += H
    return np.array(ts), np.array(px)


def direction_at(ts, px, i, **kw):
    tr = detect_long_term_trend(ts[:i + 1], px[:i + 1], window_hours=24, step_hours=8,
                                detection_lag_hours=48.0, **kw)
    return (tr or {}).get("direction")


class TestTrendLagOnDecline(unittest.TestCase):
    def setUp(self):
        self.ts, self.px = build_series()
        self.peak_i = int(np.argmax(self.px))

    def test_shows_the_lag(self):
        flip_h = None
        for i in range(self.peak_i, len(self.ts)):
            d = direction_at(self.ts, self.px, i)
            if d == "down" and flip_h is None:
                flip_h = i - self.peak_i
        self.assertIsNotNone(flip_h)

    def test_right_at_peak_is_up(self):
        self.assertEqual(direction_at(self.ts, self.px, self.peak_i), "up")

    def test_after_enough_decline_it_becomes_down(self):
        self.assertEqual(direction_at(self.ts, self.px, len(self.ts) - 1), "down")

    def test_lag_exists_on_decline(self):
        i = self.peak_i + 6
        self.assertLess(self.px[i], self.px[self.peak_i], "the price really did fall")
        self.assertEqual(direction_at(self.ts, self.px, i), "up",
                         "the bug demonstrated: it is falling but still reports 'up'")


import json
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import priceAnalysis as pa


def _uniform_series(days=20, step_sec=300.0, slope_per_day=1.0, base=100.0, end=None):
    end = end if end is not None else time.time()
    ts = np.arange(end - days * 86400, end, step_sec)
    pr = base + (ts - ts[0]) / 86400.0 * slope_per_day
    return ts, pr


class TestTimeBasedTrend(unittest.TestCase):
    def test_trend_detection_scenarios(self):
        with self.subTest(msg="uniform_uptrend"):
            ts, pr = _uniform_series(days=20, slope_per_day=1.0)
            r = pa.detect_long_term_trend(ts, pr, window_hours=16, step_hours=8)
            self.assertIsNotNone(r)
            self.assertEqual(r["direction"], "up")

        with self.subTest(msg="uniform_downtrend"):
            ts, pr = _uniform_series(days=20, slope_per_day=-1.0)
            r = pa.detect_long_term_trend(ts, pr, window_hours=16, step_hours=8)
            self.assertEqual(r["direction"], "down")

        with self.subTest(msg="window_is_time_based_not_point_count"):
            ts_a, pr_a = _uniform_series(days=15, step_sec=60.0, slope_per_day=2.0)
            ts_b, pr_b = _uniform_series(days=15, step_sec=600.0, slope_per_day=2.0)
            ra = pa.detect_long_term_trend(ts_a, pr_a, window_hours=12, step_hours=6)
            rb = pa.detect_long_term_trend(ts_b, pr_b, window_hours=12, step_hours=6)
            self.assertEqual(ra["direction"], rb["direction"])
            self.assertAlmostEqual(ra["duration_seconds"] / 86400, rb["duration_seconds"] / 86400, delta=1.0)

        with self.subTest(msg="gap_stops_trend"):
            ts, pr = _uniform_series(days=20, slope_per_day=1.0)
            keep = ~((ts > ts[0] + 5 * 86400) & (ts < ts[0] + 15 * 86400))
            r = pa.detect_long_term_trend(ts[keep], pr[keep], window_hours=16, step_hours=8, noise_tolerance=2)
            self.assertIsNotNone(r)
            self.assertGreaterEqual(r["start_timestamp"], ts[0] + 15 * 86400 - (2 + 1) * 16 * 3600 - 1)

        with self.subTest(msg="insufficient_recent_data_returns_none"):
            ts = np.array([time.time() - 100, time.time() - 50])
            pr = np.array([100.0, 101.0])
            self.assertIsNone(pa.detect_long_term_trend(ts, pr, window_hours=16))

        with self.subTest(msg="blocks_are_index_pairs"):
            ts, pr = _uniform_series(days=20, slope_per_day=1.0)
            r = pa.detect_long_term_trend(ts, pr, window_hours=16, step_hours=8)
            for lo, hi in r["blocks"]:
                self.assertTrue(0 <= lo < hi <= len(ts))

        with self.subTest(msg="memo_does_not_collide_on_identical_length_and_endpoint"):
            end_ts = 1_700_000_000.0
            ts_up, pr_up = _uniform_series(days=10, slope_per_day=1.0, base=90.0, end=end_ts)
            ts_dn, pr_dn = _uniform_series(days=10, slope_per_day=-1.0, base=110.0, end=end_ts)
            pr_dn = pr_dn + (pr_up[-1] - pr_dn[-1])
            self.assertEqual(len(ts_up), len(ts_dn))
            self.assertEqual(float(ts_up[-1]), float(ts_dn[-1]))
            self.assertAlmostEqual(float(pr_up[-1]), float(pr_dn[-1]))
            r_up = pa.detect_long_term_trend(ts_up, pr_up, window_hours=16, step_hours=8)
            r_dn = pa.detect_long_term_trend(ts_dn, pr_dn, window_hours=16, step_hours=8)
            self.assertEqual(r_up["direction"], "up")
            self.assertEqual(r_dn["direction"], "down")


class TestTrendPersistence(unittest.TestCase):
    def test_persistence_behaviors(self):
        with self.subTest(msg="write_all_trends_uses_the_shared_atomic_json_writer"):
            payload = {
                "BTCUSDC": {
                    "direction": "up",
                    "start_timestamp": 1,
                    "duration_seconds": 3600,
                    "estimated_future_hours": 2,
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(directory, "trends.json")
                with patch.object(pa, "atomic_write_json") as writer:
                    self.assertEqual(pa.write_all_trends(payload, filename), payload)
            writer.assert_called_once_with(filename, payload, indent=2)

        with self.subTest(msg="failed_atomic_write_preserves_the_previous_generation"):
            previous = {"BTCUSDC": {"direction": "down"}}
            payload = {
                "BTCUSDC": {
                    "direction": "up",
                    "start_timestamp": 1,
                    "duration_seconds": 3600,
                    "estimated_future_hours": 2,
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(directory, "trends.json")
                with open(filename, "w", encoding="utf-8") as handle:
                    json.dump(previous, handle)
                with patch.object(pa, "atomic_write_json", side_effect=OSError("disk full")):
                    self.assertEqual(pa.write_all_trends(payload, filename), payload)
                with open(filename, encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), previous)


class TestBoundedPlotIndices(unittest.TestCase):
    def test_bounded_indices_scenarios(self):
        with self.subTest(msg="short_series_is_unchanged"):
            np.testing.assert_array_equal(pa._bounded_plot_indices(4, max_points=5), np.array([0, 1, 2, 3]))

        with self.subTest(msg="long_series_is_bounded_and_keeps_endpoints"):
            indices = pa._bounded_plot_indices(43_200, max_points=5_000)
            self.assertEqual(len(indices), 5_000)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], 43_199)
            self.assertTrue(np.all(np.diff(indices) > 0))

        with self.subTest(msg="empty_series"):
            self.assertEqual(pa._bounded_plot_indices(0, max_points=5_000).size, 0)

    def test_wma_and_legacy_trend_edge_cases(self):
        with self.subTest(msg="getTrendLongTerm_fixed_handles_post_cutoff_single_point_and_coarse_delta"):
            now_s = time.time()
            sparse = [[now_s - 40 * 86400, 100.0], [now_s - 60, 105.0]]
            with patch.object(pa, "priceLstFor", return_value=sparse):
                self.assertIsNone(pa.getTrendLongTerm_fixed("TESTUSDC", lookback_days=5, draw=False))

            coarse = [[now_s - (40 - i) * 1800, 100.0 + i * 1.5] for i in range(40)]
            with patch.object(pa, "priceLstFor", return_value=coarse):
                res_lt = pa.getTrendLongTerm_fixed(
                    "TESTUSDC", window_hours=6, step_hours=2, lookback_days=5,
                    draw=False, detection_lag_hours=0.0, mk_alpha=None,
                )
                self.assertIsNotNone(res_lt)
                self.assertEqual(res_lt["direction"], "up")
                self.assertGreater(res_lt["duration_seconds"], 6 * 3600.0)

        with self.subTest(msg="detect_long_term_trend_memo_invalidates_on_interior_change"):
            ts = [1_700_000_000.0 + i * 3600.0 for i in range(60)]
            px_up = [100.0 + i * 1.0 for i in range(60)]
            r1 = pa.detect_long_term_trend(ts, px_up, window_hours=12, step_hours=4)
            self.assertEqual(r1["direction"], "up")
            px_inverted = list(px_up)
            for j in range(48, 59):
                px_inverted[j] = 300.0 - (j - 48) * 15.0
            r2 = pa.detect_long_term_trend(ts, px_inverted, window_hours=12, step_hours=4)
            self.assertNotEqual(r2, r1)

            px_zero_sum = list(px_up)
            for j in range(48, 53):
                delta = (5 - (j - 48)) * 40.0
                px_zero_sum[j] += delta
                px_zero_sum[57 - (j - 48)] -= delta
            self.assertAlmostEqual(sum(px_zero_sum), sum(px_up), places=6)
            r3 = pa.detect_long_term_trend(ts, px_zero_sum, window_hours=12, step_hours=4)
            self.assertNotEqual(r3, r1)


_WEIGHT_T = 14


def _w(trend_len, trend="up", order_type="BUY", **kw):
    _, ws = get_trade_weight(T=_WEIGHT_T, trend_len=trend_len, trend=trend, order_type=order_type, **kw)
    return float(ws[0])


class TestAligned(unittest.TestCase):
    def test_aligned_behavior(self):
        with self.subTest(msg="middle gives maximum weight"):
            self.assertAlmostEqual(_w(7), 0.95, delta=0.02)

        with self.subTest(msg="young end gives small weight"):
            self.assertLess(_w(0.5), 0.25)

        with self.subTest(msg="after the peak it caps at the middle (the lindy hypothesis validated)"):
            self.assertAlmostEqual(_w(10), 0.95, delta=0.02)
            self.assertAlmostEqual(_w(13), 0.95, delta=0.02)

        with self.subTest(msg="increases toward the middle"):
            self.assertLess(_w(0.5), _w(3))
            self.assertLess(_w(3), _w(7))

        with self.subTest(msg="the scale is coherent with zone 2"):
            self.assertGreater(_w(13.5), 0.1)
            self.assertLess(_w(14.1) / max(_w(13.5), 1e-9), 7)

        with self.subTest(msg="sell on down is aligned"):
            self.assertAlmostEqual(_w(7, trend="down", order_type="SELL"), 0.95, delta=0.02)


class TestCounterTrend(unittest.TestCase):
    def test_counter_trend_behavior(self):
        with self.subTest(msg="the middle blocks"):
            self.assertAlmostEqual(_w(7, order_type="SELL"), 0.02, delta=0.01)

        with self.subTest(msg="the young end allows a little the old end stays blocked"):
            self.assertGreater(_w(0.5, order_type="SELL"), 0.1)
            self.assertAlmostEqual(_w(13, order_type="SELL"), 0.02, delta=0.01)

        with self.subTest(msg="without the plateau the curve is symmetric"):
            young = _w(0.5, order_type="SELL", lindy_plateau=False)
            old = _w(13, order_type="SELL", lindy_plateau=False)
            self.assertAlmostEqual(young, old, delta=0.03)
            self.assertGreater(old, 0.1)
            self.assertLess(_w(13, lindy_plateau=False), 0.25)

        with self.subTest(msg="never above the counter trend cap"):
            for tl in (0.5, 3, 7, 10, 13, 15, 21):
                self.assertLessEqual(_w(tl, order_type="SELL"), 0.15 + 1e-9)


class TestZone(unittest.TestCase):
    def test_zone_behaviors(self):
        with self.subTest(msg="the seam at exactly T does not fall back"):
            self.assertGreater(_w(14.0), 0.1)

        with self.subTest(msg="zone 2 aligned"):
            self.assertAlmostEqual(_w(15), 0.86)

        with self.subTest(msg="zone 2 counter"):
            self.assertAlmostEqual(_w(15, order_type="SELL"), 0.15)

        with self.subTest(msg="zone 3 respects alignment"):
            self.assertAlmostEqual(_w(21), 0.22)
            self.assertAlmostEqual(_w(21, order_type="SELL"), 0.15)


class TestEstimateT(unittest.TestCase):
    def test_hybrid_t_estimation(self):
        from intelligence.internal.state.survival import hybrid_T

        with self.subTest(msg="many episodes dominate empirical"):
            durs = [72.0] * 100 + [160.0] * 20
            r = hybrid_T(durs, prior_T=14.0)
            self.assertLessEqual(r["T"], 9)
            self.assertGreaterEqual(r["w"], 0.75)

        with self.subTest(msg="few episodes stay at prior"):
            r = hybrid_T([72.0] * 5, prior_T=14.0)
            self.assertGreaterEqual(r["T"], 11)

        with self.subTest(msg="without episodes the prior is clean"):
            r = hybrid_T([], prior_T=14.0)
            self.assertEqual(r["T"], 14)
            self.assertEqual(r["n"], 0)

        with self.subTest(msg="safety bounds"):
            self.assertGreaterEqual(hybrid_T([10.0] * 500)["T"], 4)
            self.assertLessEqual(hybrid_T([2000.0] * 500)["T"], 30)


class TestWeightForCashPermission(unittest.TestCase):
    def test_requires_order_type_and_returns_none_or_float(self):
        weight = pa.get_weight_for_cash_permission_at_quant_time("BTCUSDC", order_type="BUY")
        self.assertTrue(weight is None or isinstance(weight, float))


class TestUnifiedWeightResolution(unittest.TestCase):
    def test_get_chop_weight_cfg_venue(self):
        import order_guard
        self.assertEqual(order_guard.get_chop_weight_cfg("binance"), 0.03)
        self.assertEqual(order_guard.get_chop_weight_cfg("kraken"), 0.03)
        self.assertEqual(order_guard.get_chop_weight_cfg(None), 0.03)

    def test_compute_weight_capped_qty(self):
        import order_guard
        adj = order_guard.compute_weight_capped_qty(
            weight=0.10, price=100.0, required_qty=1.0, available_qty=10.0, traded_24h_value=50.0)
        self.assertAlmostEqual(adj, 0.55, places=4)

    def test_resolve_trade_weight_fallback(self):
        import order_guard
        with patch("priceAnalysis.get_weight_for_cash_permission_at_quant_time", return_value=None):
            self.assertEqual(order_guard.resolve_trade_weight("UNKNOWNCOIN", "BUY", "binance"), 0.03)

    def test_backfill_ohlc_series_resamples_to_3min_for_16h_windows(self):
        pa._OHLC_BACKFILL_MEMO.clear()
        fake_ts = np.arange(1000.0, 1000.0 + 120 * 3600.0, 3600.0)
        fake_px = 1.0 + np.arange(len(fake_ts), dtype=float) * 0.01

        def _fake_fetch_klines(sym, days, interval="1h"):
            if interval == "15m":
                return np.array([]), np.array([])
            return fake_ts, fake_px

        with patch("priceAnalysis.fetch_klines", side_effect=_fake_fetch_klines):
            pts = pa._backfill_ohlc_series("ADAUSD", lookback_days=5.0)
        self.assertGreater(len(pts), 120 * 15)
        ts_arr = np.array([p[0] for p in pts])
        px_arr = np.array([p[1] for p in pts])
        res = pa.detect_long_term_trend(ts_arr, px_arr, window_hours=16, step_hours=8, min_points_per_window=20)
        self.assertIsNotNone(res)
        self.assertEqual(res["direction"], "up")

    def test_equity_fetch_klines_and_rth_gap_bridging(self):
        import importlib
        from intelligence.internal.state import survival as surv

        md = importlib.import_module("212trading.market_data")
        # Simulate 5 trading days of RTH 1h bars (7 bars/day with 17.5h overnight gaps and a 65h weekend gap)
        raw_bars = []
        t0 = 1_700_000_000.0
        px = 100.0
        for day in range(8):
            day_start = t0 + day * 86400.0 + (86400.0 * 2 if day >= 4 else 0.0)
            for h in range(7):
                px += 0.5
                raw_bars.append({"timestamp": int(day_start + h * 3600.0), "close": px})

        with (
            patch.object(md, "fetch_yahoo_candles", return_value=raw_bars) as mock_yahoo,
            patch("binance_api.market_data.fetch_ohlc_candles") as mock_binance,
        ):
            ts_k, px_k = surv.fetch_klines("NVDA_US_EQ", 33, "1h")
            self.assertEqual(len(ts_k), len(raw_bars))
            mock_yahoo.assert_called_once()
            mock_binance.assert_not_called()

            pa._OHLC_BACKFILL_MEMO.clear()
            pts = pa._backfill_ohlc_series("NVDA_US_EQ", lookback_days=10.0)
            self.assertGreater(len(pts), 1000)
            ts_arr = np.array([p[0] for p in pts])
            px_arr = np.array([p[1] for p in pts])
            res = pa.detect_long_term_trend(ts_arr, px_arr, window_hours=16, step_hours=8, min_points_per_window=20)
            self.assertIsNotNone(res)
            self.assertEqual(res["direction"], "up")

    def test_detect_long_term_trend_stops_at_major_reversal_peak(self):
        # 200h series at 180s resolution:
        # 0..32h: mild dip 82 -> 79 (lower than final price 84)
        # 32..56h: major bull impulse 79 -> 96 (within noise_tolerance=2 windows)
        # 56..200h: steady downtrend 96 -> 84
        dt = 180.0
        ts1 = np.arange(0.0, 32 * 3600.0, dt)
        px1 = np.linspace(82.0, 79.0, len(ts1))
        ts2 = np.arange(32 * 3600.0, 56 * 3600.0, dt)
        px2 = np.linspace(79.0, 96.0, len(ts2))
        ts3 = np.arange(56 * 3600.0, 200 * 3600.0, dt)
        px3 = np.linspace(96.0, 84.0, len(ts3))
        ts = np.concatenate([ts1, ts2, ts3]) + 1_700_000_000.0
        px = np.concatenate([px1, px2, px3])
        res = pa.detect_long_term_trend(
            ts, px, window_hours=16, step_hours=8, min_points_per_window=20, noise_tolerance=2, detection_lag_hours=0.0
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["direction"], "down")
        confirm_lo = res["blocks"][-1][0]
        self.assertGreater(px[confirm_lo], px[-1])
        self.assertLessEqual(res["duration_seconds"] / 3600.0, 152.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
