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

    def test_long_gap_resets_trend_even_when_conf_exit_is_zero(self):
        """Even if CONF_EXIT is 0.0, a gap > GAP_RESET_SEC must reset trend to 0."""
        from unittest.mock import patch
        import intelligence.internal.triggers.kalman_trigger as kt
        kf = ss.KalmanTrend()
        ts, price, out = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        self.assertEqual(out["trend"], 1)
        with patch.object(kt, "CONF_EXIT", 0.0):
            out_after = kf.update(ts + 3600.0, price, epsilon=0.01)
        self.assertEqual(out_after["trend"], 0)
        self.assertEqual(out_after["old_trend"], 1)


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


class TestShadowRuntime(unittest.TestCase):
    """Runtime setup idempotency and interval validation."""

    def test_prepare_shadow_runtime_is_idempotent(self):
        from shadow_runtime import prepare_shadow_runtime
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_val = os.environ.get("BINANCE_AUTO_START_WEBSOCKETS")
            try:
                if "BINANCE_AUTO_START_WEBSOCKETS" in os.environ:
                    del os.environ["BINANCE_AUTO_START_WEBSOCKETS"]
                prepare_shadow_runtime(tmp_dir)
                prepare_shadow_runtime(tmp_dir)
                self.assertEqual(os.environ["BINANCE_AUTO_START_WEBSOCKETS"], "0")
                self.assertEqual(sys.path.count(tmp_dir), 1)
            finally:
                if old_val is not None:
                    os.environ["BINANCE_AUTO_START_WEBSOCKETS"] = old_val
                elif "BINANCE_AUTO_START_WEBSOCKETS" in os.environ:
                    del os.environ["BINANCE_AUTO_START_WEBSOCKETS"]
                if tmp_dir in sys.path:
                    sys.path.remove(tmp_dir)

    def test_shadow_interval_validation(self):
        from shadow_runtime import require_shadow_interval
        require_shadow_interval(240, 240, "HL shadow")
        with self.assertRaisesRegex(ValueError, "240m"):
            require_shadow_interval(60, 240, "HL shadow")

    def test_kalman_60s_step_noise_transitions_after_60_samples(self):
        """At 60s sampling with adaptive step-noise epsilon, KalmanTrend must not lock at 0 after 60 samples."""
        import numpy as np
        kf = ss.KalmanTrend()
        fed = []
        ts = 1_000_000.0
        price = 74000.0
        # 60 minutes of mild chop
        for i in range(60):
            ts += 60.0
            price += (15.0 if i % 2 == 0 else -15.0)
            fed.append(price)
            eps = float(np.std(np.gradient(np.asarray(fed)))) if len(fed) >= 5 else 60.0
            out = kf.update(ts, price, eps)
        self.assertEqual(out["trend"], 0)

        # Breakout (+75 USD/min ~ +0.10%/min for 6 minutes)
        for _ in range(6):
            ts += 60.0
            price += 75.0
            fed.append(price)
            if len(fed) > 60:
                fed.pop(0)
            eps = float(np.std(np.gradient(np.asarray(fed))))
            out = kf.update(ts, price, eps)
        self.assertEqual(out["trend"], 1)

    def test_tradeall_logic_small_acceleration_surge(self):
        """logic_small triggers accel_surge_up when short-term velocity and acceleration are strongly positive."""
        from unittest.mock import patch
        import tradeall as ta
        from pricewindow import PriceWindow

        pw = PriceWindow("BTCUSDC", 220, sample_rate_sec=1.0)
        # Parabolic upward curve over 180s
        px = 74000.0
        for i in range(180):
            t_min = i / 60.0
            px = 74000.0 * (1.0 + 0.0012 * t_min + 0.0018 * (t_min ** 2))
            pw.process_price(px)

        trend_state = ta.TrendState("BTCUSDC", 3600.0, 222.0)
        with patch.object(ta, "track_and_place_order", return_value={"orderId": "1"}) as mock_place:
            ta.logic_small(
                "SMALL", True, "BTCUSDC",
                gradient=1, slope=ta.PRICE_CHANGE_THRESHOLD_EUR + 0.05,
                trend_state=trend_state, current_price=px,
                price_window=pw,
            )
            self.assertEqual(mock_place.call_count, 1)
            self.assertEqual(mock_place.call_args.kwargs.get("reason"), "accel_surge_up")


if __name__ == "__main__":
    unittest.main()

