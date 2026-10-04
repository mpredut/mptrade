"""Comprehensive unit tests for the intelligence package (triggers, guards, and state)."""
import unittest

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.internal.triggers.kalman_trigger import KalmanTrend, KalmanTrendTrigger
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.state.volatility import calculate_volatility_1h, adaptive_thresholds
from intelligence.internal.state.survival import get_trend_survival_metrics
from intelligence.composite import CompositeMarketIntelligence


class TestIntelligenceTriggers(unittest.TestCase):
    def test_trigger_event_executable(self):
        ev_buy = TriggerEvent(
            action=TriggerAction.ENTRY,
            side=TriggerSide.BUY,
            source="test",
            symbol="BTCUSDC",
            price=85000.0,
            strength=2.5,
        )
        self.assertTrue(ev_buy.is_executable())

        ev_hold = TriggerEvent(
            action=TriggerAction.NEUTRAL,
            side=TriggerSide.HOLD,
            source="test",
            symbol="BTCUSDC",
            price=85000.0,
            strength=0.0,
        )
        self.assertFalse(ev_hold.is_executable())

    def test_kalman_trigger_transitions(self):
        trigger = KalmanTrendTrigger(qr=0.001)
        symbol = "BTCUSDC"

        # Initialize flat
        trigger.evaluate(symbol, 100.0, 80000.0, 10.0)
        trigger.evaluate(symbol, 160.0, 80005.0, 10.0)

        # Sharp upward price movement should trigger transition to UP (1)
        event = None
        for i in range(1, 10):
            t = 160.0 + i * 60.0
            p = 80005.0 + i * 500.0
            out, ev = trigger.evaluate(symbol, t, p, 10.0)
            if ev:
                event = ev
                break

        self.assertIsNotNone(event)
        self.assertEqual(event.action, TriggerAction.ENTRY)
        self.assertEqual(event.side, TriggerSide.BUY)
        self.assertEqual(event.reason, "kalman_transition_up")

    def test_gradient_trigger(self):
        trigger = LinearGradientTrigger(strength_threshold=2.0)
        symbol = "BTCUSDC"

        # Within noise floor: |gradient| <= 2 * eps
        regime, ev = trigger.evaluate(symbol, 80000.0, gradient=1.5, epsilon=1.0)
        self.assertEqual(regime, "sideways")
        self.assertIsNone(ev)

        # Breakout UP: gradient = 3.0, eps = 1.0 (strength = 3.0 > 2.0)
        regime, ev = trigger.evaluate(symbol, 80500.0, gradient=3.0, epsilon=1.0)
        self.assertEqual(regime, "bull")
        self.assertIsNotNone(ev)
        self.assertEqual(ev.side, TriggerSide.BUY)
        self.assertEqual(ev.action, TriggerAction.ENTRY)


class TestIntelligenceGuards(unittest.TestCase):
    def test_noise_floor_guard(self):
        guard = NoiseFloorGuard(min_strength_ratio=1.0)

        # In noise: |gradient| = 0.5 <= eps = 1.0
        dec = guard.check("BTCUSDC", "BUY", gradient=0.5, epsilon=1.0)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)

        # Above noise: |gradient| = 1.5 > eps = 1.0
        dec_ok = guard.check("BTCUSDC", "BUY", gradient=1.5, epsilon=1.0)
        self.assertTrue(dec_ok.allowed)
        self.assertEqual(dec_ok.brake_action, BrakeAction.NONE)

    def test_parabolic_surge_guard(self):
        guard = ParabolicSurgeGuard(surge_threshold_pct=5.0, pullback_required_pct=2.0)
        symbol = "BTCUSDC"

        # Simulated price surge from 80k to 86k (+7.5% > 5%)
        history = [(1000.0, 80000.0), (1060.0, 83000.0), (1120.0, 86000.0)]

        # Buying at peak (86k) without pullback should be deferred
        dec = guard.check(symbol, "BUY", 86000.0, price_history=history, now=1120.0)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("parabolic_surge_active", dec.reason)

        # Non-BUY (e.g. SELL) is always allowed even during surge
        dec_sell = guard.check(symbol, "SELL", 86000.0, price_history=history, now=1120.0)
        self.assertTrue(dec_sell.allowed)

        # After a pullback from 86k to 84k (pullback = (86-84)/86 = 2.32% > 2.0%), allowed
        dec_pullback = guard.check(symbol, "BUY", 84000.0, price_history=history, now=1200.0)
        self.assertTrue(dec_pullback.allowed)

    def test_weibull_exhaustion_guard(self):
        guard = WeibullExhaustionGuard(default_p90_days=7.0, policy="downscale", exhausted_scale=0.25)
        symbol = "BTCUSDC"

        # Trend duration = 3 days <= P90 (7.0 days)
        dec_young = guard.check(symbol, "BUY", trend_duration_seconds=3.0 * 86400, p90_days=7.0)
        self.assertTrue(dec_young.allowed)
        self.assertEqual(dec_young.suggested_scale, 1.0)

        # Trend duration = 10 days > P90 (7.0 days) -> Downscale brake!
        dec_old = guard.check(symbol, "BUY", trend_duration_seconds=10.0 * 86400, p90_days=7.0)
        self.assertTrue(dec_old.allowed)
        self.assertEqual(dec_old.brake_action, BrakeAction.DOWNSCALE_QTY)
        self.assertEqual(dec_old.suggested_scale, 0.25)
        self.assertIn("trend_exhausted", dec_old.reason)

        # Veto policy blocks completely
        guard_veto = WeibullExhaustionGuard(default_p90_days=7.0, policy="veto")
        dec_veto = guard_veto.check(symbol, "BUY", trend_duration_seconds=10.0 * 86400, p90_days=7.0)
        self.assertFalse(dec_veto.allowed)
        self.assertEqual(dec_veto.brake_action, BrakeAction.HARD_VETO)


class TestIntelligenceState(unittest.TestCase):
    def test_volatility_1h(self):
        prices = [100.0 * (1.0 + 0.001 * (i % 5)) for i in range(30)]
        vol = calculate_volatility_1h(prices, sample_rate_sec=60.0)
        self.assertIsNotNone(vol)
        self.assertGreater(vol, 0.0)

        reentry, dca = adaptive_thresholds(vol, k_reentry=2.0, k_dca=1.0)
        self.assertIsNotNone(reentry)
        self.assertIsNotNone(dca)
        self.assertAlmostEqual(reentry, round(2.0 * vol, 3))

    def test_survival_metrics(self):
        metrics = get_trend_survival_metrics("BTCUSDC", trend_duration_seconds=10.0 * 86400)
        self.assertEqual(metrics["symbol"], "BTCUSDC")
        self.assertEqual(metrics["duration_days"], 10.0)
        self.assertGreater(metrics["p90_days"], 0.0)
        self.assertTrue(metrics["is_exhausted"])


class TestCompositeMarketIntelligence(unittest.TestCase):
    def test_composite_orchestration_blocks_fomo(self):
        intel = CompositeMarketIntelligence(parabolic_surge_pct=5.0)
        symbol = "BTCUSDC"

        # Simulate surge history
        history = [(1000.0, 80000.0), (1060.0, 83000.0), (1120.0, 86000.0)]

        # Direct guard evaluation at peak
        guard_dec = intel.evaluate_guards(
            symbol,
            "BUY",
            86000.0,
            gradient=10.0,
            epsilon=1.0,
            price_history=history,
            now=1120.0,
        )
        self.assertFalse(guard_dec.allowed)
        self.assertEqual(guard_dec.brake_action, BrakeAction.DEFER_WAIT)



class TestMeanReversionTrigger(unittest.TestCase):
    def test_oversold_triggers_buy_entry(self):
        trigger = MeanReversionTrigger(mode="rsi", rsi_period=14, rsi_low=30.0)
        symbol = "BTCUSDC"

        # Feed dropping prices to reach oversold
        ev = None
        price = 100.0
        for i in range(30):
            price -= 2.0
            ev = trigger.update(symbol, timestamp=1000.0 + i * 60, price=price)

        self.assertIsNotNone(ev)
        self.assertEqual(ev.side, TriggerSide.BUY)
        self.assertEqual(ev.action, TriggerAction.ENTRY)
        self.assertLess(ev.metadata["rsi"], 30.0)

    def test_overbought_triggers_sell_exit(self):
        trigger = MeanReversionTrigger(mode="rsi", rsi_period=14, rsi_high=70.0)
        symbol = "BTCUSDC"

        ev = None
        price = 100.0
        for i in range(30):
            price += 2.0
            ev = trigger.update(symbol, timestamp=1000.0 + i * 60, price=price)

        self.assertIsNotNone(ev)
        self.assertEqual(ev.side, TriggerSide.SELL)
        self.assertEqual(ev.action, TriggerAction.EXIT)
        self.assertGreater(ev.metadata["rsi"], 70.0)


class TestTrendSignificanceGuard(unittest.TestCase):
    def test_significant_trend_allowed(self):
        from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
        guard = TrendSignificanceGuard(max_p_value=0.05)
        # Clear upward trend
        prices = [100.0 + 1.0 * i for i in range(24)]
        ev = TriggerEvent(
            action=TriggerAction.ENTRY,
            side=TriggerSide.BUY,
            source="test",
            symbol="BTCUSDC",
            price=prices[-1],
            strength=1.0,
        )
        dec = guard.evaluate(ev, prices)
        self.assertTrue(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.NONE)

    def test_noisy_trend_deferred(self):
        from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
        import numpy as np
        guard = TrendSignificanceGuard(max_p_value=0.05)
        # Random noise around 100
        rng = np.random.default_rng(42)
        prices = list(100.0 + rng.normal(0, 1.0, 24))
        ev = TriggerEvent(
            action=TriggerAction.ENTRY,
            side=TriggerSide.BUY,
            source="test",
            symbol="BTCUSDC",
            price=prices[-1],
            strength=1.0,
        )
        dec = guard.evaluate(ev, prices)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("trend_slope_not_statistically_significant", dec.reason)

    def test_inverted_direction_vetoed(self):
        from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
        guard = TrendSignificanceGuard(max_p_value=0.05)
        # Downward series
        prices = [100.0 - 1.0 * i for i in range(24)]
        # But trying to BUY
        ev = TriggerEvent(
            action=TriggerAction.ENTRY,
            side=TriggerSide.BUY,
            source="test",
            symbol="BTCUSDC",
            price=prices[-1],
            strength=1.0,
        )
        dec = guard.evaluate(ev, prices)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.HARD_VETO)
        self.assertIn("trend_direction_inverted", dec.reason)


class TestPersistenceState(unittest.TestCase):
    def test_mann_kendall_and_hurst(self):
        from intelligence.internal.state.persistence import (
            calculate_mann_kendall,
            calculate_hurst_exponent,
            classify_hurst_regime,
        )
        import numpy as np
        # Linear slope
        y = [100.0 + 0.5 * i for i in range(24)]
        s, z, p = calculate_mann_kendall(y)
        self.assertGreater(z, 0)
        self.assertLess(p, 0.01)

        # Regimes classification
        self.assertEqual(classify_hurst_regime(0.65), "persistent")
        self.assertEqual(classify_hurst_regime(0.35), "mean_reverting")
        self.assertEqual(classify_hurst_regime(0.50), "random_walk")
        self.assertEqual(classify_hurst_regime(None), "unknown")


if __name__ == "__main__":
    unittest.main()

