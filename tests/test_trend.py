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
from forecast.trend_stats import mann_kendall, hurst_rs, hurst_regime  # noqa: E402
from priceAnalysis import detect_long_term_trend  # noqa: E402

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

    def test_date_insuficiente_da_none(self):
        ts, px = series(down(10))
        self.assertIsNone(run(ts, px))


class TestMannKendall(unittest.TestCase):
    def test_monotonic_trends_are_significant(self):
        for label, delta, expected_sign in (("up", 0.5, 1), ("down", -0.5, -1)):
            with self.subTest(direction=label):
                _, z, p = mann_kendall([100 + delta * i for i in range(24)])
                self.assertGreater(z * expected_sign, 0)
                self.assertLess(p, 0.01)

    def test_zgomot_alb_nesemnificativ(self):
        rng = np.random.default_rng(42)
        s, z, p = mann_kendall(100 + rng.normal(0, 1, 24))
        self.assertGreater(p, 0.1, "pure noise must not look like a trend")

    def test_serie_scurta_nesemnificativa(self):
        _, _, p = mann_kendall([1, 2, 3])
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
                self.assertTrue(predicate(hurst_rs(self._series(phi))))

    def test_serie_scurta_da_none(self):
        self.assertIsNone(hurst_rs([100, 101, 102]))

    def test_regimuri(self):
        self.assertEqual(hurst_regime(0.65), "persistent")
        self.assertEqual(hurst_regime(0.30), "mean-reverting")
        self.assertEqual(hurst_regime(0.50), "random-walk")
        self.assertEqual(hurst_regime(None), "necunoscut")


class TestFiltruMKInDetector(unittest.TestCase):
    H = 3600.0

    def _mk_series(self, vals):
        return np.array([i * self.H for i in range(len(vals))]), np.array(vals, float)

    def test_trend_curat_trece_de_filtru(self):
        ts, px = self._mk_series([100 - 0.3 * i for i in range(96)])
        r = detect_long_term_trend(ts, px, 24, 8, 3, 2, 5, mk_alpha=0.05)
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "down")

    def test_zgomotul_e_filtrat(self):
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


class TestTrendLagPeDeclin(unittest.TestCase):
    def setUp(self):
        self.ts, self.px = build_series()
        self.peak_i = int(np.argmax(self.px))

    def test_arata_lagul(self):
        flip_h = None
        for i in range(self.peak_i, len(self.ts)):
            d = direction_at(self.ts, self.px, i)
            if d == "down" and flip_h is None:
                flip_h = i - self.peak_i
        self.assertIsNotNone(flip_h)

    def test_chiar_la_varf_e_up(self):
        self.assertEqual(direction_at(self.ts, self.px, self.peak_i), "up")

    def test_after_enough_decline_it_becomes_down(self):
        self.assertEqual(direction_at(self.ts, self.px, len(self.ts) - 1), "down")

    def test_exista_lag_pe_declin(self):
        i = self.peak_i + 6
        self.assertLess(self.px[i], self.px[self.peak_i], "the price really did fall")
        self.assertEqual(direction_at(self.ts, self.px, i), "up",
                         "the bug demonstrated: it is falling but still reports 'up'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
