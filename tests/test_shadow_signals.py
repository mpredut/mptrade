"""
Tests for shadow_signals.py — in particular the behaviour on a data GAP
in KalmanTrend.update(), fixed 21 Jul: a long gap (network down, process
stopped) no longer propagates the old velocity (capped at DT_MAX) but resets
the filter (as at warm-up) — the trend comes out FLAT after a gap, not "confident" in a
stale direction.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shadow_signals as ss


class TestKalmanTrendContinuous(unittest.TestCase):
    """Baseline behaviour (unchanged) on a series without gaps: a clear UP trend."""

    def test_continuous_series_direction(self):
        for label, delta, expected in (("up", 0.05, 1), ("flat", 0.0, 0)):
            with self.subTest(case=label):
                kf = ss.KalmanTrend()
                ts = 1_000_000.0
                price = 100.0
                out = None
                for _ in range(30):
                    ts += 1.0
                    price += delta
                    out = kf.update(ts, price, epsilon=0.01)
                self.assertEqual(out["trend"], expected)


class TestKalmanTrendGapReset(unittest.TestCase):
    """21 Jul: a gap > GAP_RESET_SEC (300s) => reset, not merely a capped dt."""

    def _warm_up_uptrend(self, kf, ts, price, steps=30):
        for _ in range(steps):
            ts += 1.0
            price += 0.05
            out = kf.update(ts, price, epsilon=0.01)
        return ts, price, out

    def test_long_gap_resets_velocity_to_zero(self):
        kf = ss.KalmanTrend()
        ts, price, out = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        self.assertEqual(out["trend"], 1, "precondition: an UP trend established before the gap")

        # a 1h gap (> GAP_RESET_SEC=300s), the price returns unchanged from the last known one
        ts_after_gap = ts + 3600.0
        out_after = kf.update(ts_after_gap, price, epsilon=0.01)

        self.assertEqual(out_after["vel"], 0.0, "the old velocity must NOT be propagated across a long gap")
        self.assertEqual(out_after["trend"], 0, "after the reset the trend comes out FLAT, not a stale UP")

    def test_short_gap_under_threshold_is_not_reset(self):
        """A 30s gap (under GAP_RESET_SEC) must be handled normally (real dt), not as a reset."""
        kf = ss.KalmanTrend()
        ts, price, out = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        self.assertEqual(out["trend"], 1)

        ts_after_gap = ts + 30.0
        price += 0.05 * 30       # the UP trend continues past the gap
        out_after = kf.update(ts_after_gap, price, epsilon=0.01)

        self.assertNotEqual(out_after["vel"], 0.0, "a short gap must not reset the velocity to 0")
        self.assertEqual(out_after["trend"], 1, "the UP trend must survive a short gap")

    def test_gap_reset_does_not_crash_on_first_ever_update(self):
        """The first observation (self.x is None) must stay unaffected by the new gap branch."""
        kf = ss.KalmanTrend()
        out = kf.update(1_000_000.0, 100.0, epsilon=0.01)
        self.assertEqual(out["trend"], 0)
        self.assertEqual(out["vel"], 0.0)

    def test_gap_boundary_exactly_at_threshold_not_reset(self):
        """At exactly GAP_RESET_SEC the reset branch (strict >) must not fire."""
        kf = ss.KalmanTrend()
        ts, price, _ = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        out_after = kf.update(ts + ss.GAP_RESET_SEC, price + 0.05, epsilon=0.01)
        # real dt = exactly GAP_RESET_SEC => not strictly greater => normal path (no reset to 0)
        self.assertNotEqual(out_after["vel"], 0.0)


class TestVolAndThresholds(unittest.TestCase):
    """Sanity minim pe restul modulului — neschimbate de fix-ul de gap."""

    def test_warmup_outputs_are_unavailable(self):
        self.assertIsNone(ss.vol_1h_pct([100.0] * 5, sample_rate_sec=1.0))
        reentry, dca = ss.adaptive_thresholds(None)
        self.assertIsNone(reentry)
        self.assertIsNone(dca)




class TestCurrentTrendClock(unittest.TestCase):
    """current_trend() ages the state on the injected clock (live: wall time)."""

    def test_age_uses_injected_clock_and_defaults_to_wall_time(self):
        journal = ss.ShadowJournal(fixed_path=os.devnull)
        simulated = ss.ShadowSet(journal=journal, now_fn=lambda: 1_000_060.0)
        simulated._state["BTCUSDC"] = {"kalman_trend": -1, "ts": 1_000_000.0}
        self.assertEqual(simulated.current_trend("BTCUSDC"), (-1, 60.0))

        wall = ss.ShadowSet(journal=journal)
        wall._state["BTCUSDC"] = {"kalman_trend": 1, "ts": 1_000_000.0}
        trend, age = wall.current_trend("BTCUSDC")
        self.assertEqual(trend, 1)
        self.assertGreater(age, 1e8)


if __name__ == "__main__":
    unittest.main()
