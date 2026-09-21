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


class TestCazulTAO(unittest.TestCase):
    """The reported case: a falling slope over 4 days but a 'current trend' of UP."""

    def test_a_one_day_bounce_against_the_fall_is_not_an_up_trend(self):
        d = down(96)                                  # 4 zile scadere
        b = up(24, start=d[-1], rate=0.25)            # 24h bounce
        r = run(*series(d, b))
        self.assertIsNone(r, "the 24h bounce is NOT a coherent UP trend — it must be None")

    def test_a_sustained_2_day_bounce_becomes_an_up_trend_with_the_right_duration(self):
        d = down(96)
        b = up(48, start=d[-1], rate=0.25)            # bounce sustinut 2 zile
        r = run(*series(d, b))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "up")
        dur_days = r["duration_seconds"] / 86400
        self.assertLess(dur_days, 2.6, "the duration must be ~2 days (the bounce), not 4-6")


class TestDurata(unittest.TestCase):
    def test_the_duration_does_not_exceed_the_data_span(self):
        ts, px = series(down(96))                     # 4 zile date
        r = run(ts, px)
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "down")
        self.assertLessEqual(r["duration_seconds"], ts[-1] - ts[0] + 1,
                             "the reported duration was longer than ALL the data (the old bug: +72h invented)")

    def test_scadere_pura_da_durata_aproape_de_span(self):
        ts, px = series(down(96))
        r = run(ts, px)
        self.assertGreater(r["duration_seconds"] / 86400, 3.0, "trendul real e ~4 zile")


class TestZgomot(unittest.TestCase):
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

    def test_trend_up_curat(self):
        r = run(*series(up(96)))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "up")


class TestLagDetectie(unittest.TestCase):
    """detection_lag_hours: trendul incepe inaintea confirmarii (euristica ~2 zile),
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
        u_ = up(72, start=d[-1], rate=0.25)           # trend up confirmat ~3 zile
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


class TestGapuri(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
