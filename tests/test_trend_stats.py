#!/usr/bin/env python3
"""
Tests for trend_stats (Mann-Kendall plus Hurst) and the integration of the MK filter in the detector.

  python -m pytest tests/test_trend_stats.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.trend_stats import mann_kendall, hurst_rs, hurst_regime  # noqa: E402
from priceAnalysis import detect_long_term_trend  # noqa: E402


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
        """prices with AR(1) log returns: phi>0 persistent, phi<0 anti-persistent"""
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
        base = [100 - 0.3 * i for i in range(72)]            # a history with a trend
        noise = list(base[-1] + rng.normal(0, 0.8, 24))      # the current window = pure noise
        ts, px = self._mk_series(base + noise)
        r = detect_long_term_trend(ts, px, 24, 8, 3, 2, 5, mk_alpha=0.05)
        self.assertIsNone(r, "a current window with no significant trend -> None")

    def test_without_the_filter_the_old_behaviour(self):
        ts, px = self._mk_series([100 - 0.3 * i for i in range(96)])
        r = detect_long_term_trend(ts, px, 24, 8, 3, 2, 5, mk_alpha=None)
        self.assertIsNotNone(r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
