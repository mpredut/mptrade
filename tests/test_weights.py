#!/usr/bin/env python3
"""
Tests for get_trade_weight (Gaussian cash-permission weights) — bug fixes:
Zone 1 vs Zone 2/3 scaling, counter-trend reversal at the old end,
the seam at exactly T, and Zone 3 which ignores alignment.

  python -m pytest tests/test_weights.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from priceAnalysis import get_trade_weight  # noqa: E402

T = 14


def w(trend_len, trend="up", order_type="BUY", **kw):
    _, ws = get_trade_weight(T=T, trend_len=trend_len, trend=trend,
                             order_type=order_type, **kw)
    return float(ws[0])


class TestAliniat(unittest.TestCase):
    """BUY+up / SELL+down: gaussiana scalata la varf (mijloc ~0.95)."""

    def test_aliniat_behavior(self):
        with self.subTest(msg="mijlocul da ponderea maxima"):
            self.assertAlmostEqual(w(7), 0.95, delta=0.02,
                                   msg="the curve peak must be ~0.95, not ~0.11 (the scaling bug)")

        with self.subTest(msg="capatul tanar da pondere mica"):
            self.assertLess(w(0.5), 0.25, "trend tanar = posibil zgomot -> prudent")

        with self.subTest(msg="after the peak it caps at the middle (the lindy hypothesis validated)"):
            # validat empiric (trend_survival.py): trendul batran continua la fel de
            # probably as at the middle -> we behave as at the middle, we do not descend the gaussian
            self.assertAlmostEqual(w(10), 0.95, delta=0.02)
            self.assertAlmostEqual(w(13), 0.95, delta=0.02)

        with self.subTest(msg="creste spre mijloc"):
            self.assertLess(w(0.5), w(3))
            self.assertLess(w(3), w(7))

        with self.subTest(msg="the scale is coherent with zone 2"):
            # before the fix: 13.9 days -> 0.021 and 14.1 days -> 0.86 (a 40x jump)
            self.assertGreater(w(13.5), 0.1)
            self.assertLess(w(14.1) / max(w(13.5), 1e-9), 7,
                            "the Zone1->Zone2 jump must be reasonable, not 40x")

        with self.subTest(msg="sell pe down e aliniat"):
            self.assertAlmostEqual(w(7, trend="down", order_type="SELL"), 0.95, delta=0.02)


class TestContraTrend(unittest.TestCase):
    """SELL+up / BUY+down: the middle -> ~0.02 (do not trade), the ends -> ~0.13-0.15."""

    def test_contra_trend_behavior(self):
        with self.subTest(msg="the middle blocks"):
            self.assertAlmostEqual(w(7, order_type="SELL"), 0.02, delta=0.01)

        with self.subTest(msg="the young end allows a little the old end stays blocked"):
            # with the validated Lindy cap: an old trend behaves as at the middle,
            # so the counter-trade stays blocked even in old age
            self.assertGreater(w(0.5, order_type="SELL"), 0.1)
            self.assertAlmostEqual(w(13, order_type="SELL"), 0.02, delta=0.01)

        with self.subTest(msg="without the plateau the curve is symmetric"):
            # comportamentul clasic (gaussiana pura) ramane disponibil
            young = w(0.5, order_type="SELL", lindy_plateau=False)
            old = w(13, order_type="SELL", lindy_plateau=False)
            self.assertAlmostEqual(young, old, delta=0.03, msg="without the cap, the global curve is symmetric")
            self.assertGreater(old, 0.1)
            self.assertLess(w(13, lindy_plateau=False), 0.25, "aligned, without the cap, it descends at the end")

        with self.subTest(msg="never above the counter trend cap"):
            for tl in (0.5, 3, 7, 10, 13, 15, 21):
                self.assertLessEqual(w(tl, order_type="SELL"), 0.15 + 1e-9)


class TestZone(unittest.TestCase):
    def test_zone_behaviors(self):
        with self.subTest(msg="the seam at exactly T does not fall back"):
            self.assertGreater(w(14.0), 0.1, "at exactly T the slice is no longer empty (the 0.05 fallback)")

        with self.subTest(msg="zona 2 aliniat"):
            self.assertAlmostEqual(w(15), 0.86)

        with self.subTest(msg="zona 2 contra"):
            self.assertAlmostEqual(w(15, order_type="SELL"), 0.15)

        with self.subTest(msg="zona 3 respecta alinierea"):
            self.assertAlmostEqual(w(21), 0.22)
            self.assertAlmostEqual(w(21, order_type="SELL"), 0.15,
                                   msg="a counter-trend on an old trend does not exceed max_against_trend")


class TestEstimareT(unittest.TestCase):
    """hybrid_T: the empirical value is favoured when we have data, the prior when we do not (no network)."""

    def test_hybrid_t_estimation(self):
        from forecast.trend_survival import hybrid_T

        with self.subTest(msg="multe episoade domina empiricul"):
            durs = [72.0] * 100 + [160.0] * 20            # mediana 3z, P90 ~6.7z
            r = hybrid_T(durs, prior_T=14.0)
            self.assertLessEqual(r["T"], 9, "with n=120 episodes, T must be close to the empirical value (~7), not 14")
            self.assertGreaterEqual(r["w"], 0.75)

        with self.subTest(msg="putine episoade raman la prior"):
            r = hybrid_T([72.0] * 5, prior_T=14.0)
            self.assertGreaterEqual(r["T"], 11, "with 5 episodes, the prior must dominate")

        with self.subTest(msg="without episodes the prior is clean"):
            r = hybrid_T([], prior_T=14.0)
            self.assertEqual(r["T"], 14)
            self.assertEqual(r["n"], 0)

        with self.subTest(msg="limitele de siguranta"):
            self.assertGreaterEqual(hybrid_T([10.0] * 500)["T"], 4, "T does not fall below 4 days")
            self.assertLessEqual(hybrid_T([2000.0] * 500)["T"], 30, "T does not rise above 30 days")


class TestWeightForCashPermission(unittest.TestCase):
    """Fost test_weight.py (unificat 28 iul) — get_weight_for_cash_permission_at_
    quant_time (another weight function in priceAnalysis) returns None or a float."""

    def test_requires_order_type_and_returns_none_or_float(self):
        import priceAnalysis as pa
        weight = pa.get_weight_for_cash_permission_at_quant_time("BTCUSDC", order_type="BUY")
        self.assertTrue(weight is None or isinstance(weight, float))


if __name__ == "__main__":
    unittest.main(verbosity=2)
