#!/usr/bin/env python3
"""
test_tao_trend_lag.py — it reproduces the question: after TAO started FALLING from its
peak, why did the trend stay "up" (blocking the TP)?

The detector (detect_long_term_trend) takes the direction from the slope of the LAST window_hours (24h).
After the peak, the 24h window still contains the climb -> positive slope -> "up", even though
the price is ALREADY falling. Only when the window fills with the fall does the direction become "down".
The test measures that LAG: how many hours after the peak it stays "up" while the price falls.

Run it on the server (numpy):  ~/mptrade/myenv/bin/python test_tao_trend_lag.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from priceAnalysis import detect_long_term_trend  # noqa: E402

H = 3600.0


def build_series(climb_days=5, climb_from=233.0, peak=287.0, decline_to=265.0, decline_hours=36):
    """An hourly series: a smooth climb over climb_days days to the peak, then a fall over decline_hours hours."""
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
        print(f"\nPEAK at +{self.peak_i}h, price {self.px[self.peak_i]:.1f}")
        print("hours_after_peak | price | reported_trend")
        flip_h = None
        for i in range(self.peak_i, len(self.ts)):
            d = direction_at(self.ts, self.px, i)
            after = i - self.peak_i
            if after % 4 == 0 or (d == "down" and flip_h is None):
                print(f"   +{after:>2}h       | {self.px[i]:.1f} | {d}")
            if d == "down" and flip_h is None:
                flip_h = after
        print(f"\n=> The trend stayed 'up' for another ~{flip_h}h AFTER the peak, while the price was falling "
              f"de la {self.px[self.peak_i]:.1f} la {self.px[self.peak_i + (flip_h or 0)]:.1f}.")
        print("   Throughout that interval the TP was blocked by 'not is_trend_up'.")

    def test_chiar_la_varf_e_up(self):
        # Right at the peak it is still rising over 24h -> 'up' (correct, but this is where the problem starts).
        self.assertEqual(direction_at(self.ts, self.px, self.peak_i), "up")

    def test_after_enough_decline_it_becomes_down(self):
        # At the end of the fall (a 24h window full of the decline) it must be 'down'.
        self.assertEqual(direction_at(self.ts, self.px, len(self.ts) - 1), "down")

    def test_exista_lag_pe_declin(self):
        # THE KEY: right after the peak (a few hours of a real fall) it still reports 'up'.
        i = self.peak_i + 6  # 6h after the peak, the price has already fallen.
        self.assertLess(self.px[i], self.px[self.peak_i], "the price really did fall")
        self.assertEqual(direction_at(self.ts, self.px, i), "up",
                         "the bug demonstrated: it is falling but still reports 'up'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
