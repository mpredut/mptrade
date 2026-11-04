"""Comprehensive unit tests for the intelligence package (triggers, guards, state, daemon, and backtest)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intelligence.cli import get_daemon_status, inspect_symbol_intelligence
from intelligence.composite import CompositeMarketIntelligence
from intelligence.daemon import IntelligenceTelemetryDaemon
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.guard_decision import BrakeAction
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
from intelligence.internal.state.persistence import (
    calculate_mann_kendall,
    classify_hurst_regime,
)
from intelligence.internal.state.survival import get_trend_survival_metrics
from intelligence.internal.state.volatility import (
    adaptive_thresholds,
    calculate_volatility_1h,
    get_kalman_state_path,
    resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct,
    vol_pct_to_ratio,
    vol_ratio_to_pct,
)
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
from offline.research.intelligence_backtest import BacktestStats, run_intelligence_backtest


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

        trigger.evaluate(symbol, 100.0, 80000.0, 10.0)
        trigger.evaluate(symbol, 160.0, 80005.0, 10.0)

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

        regime, ev = trigger.evaluate(symbol, 80000.0, gradient=1.5, epsilon=1.0)
        self.assertEqual(regime, "sideways")
        self.assertIsNone(ev)

        regime, ev = trigger.evaluate(symbol, 80500.0, gradient=3.0, epsilon=1.0)
        self.assertEqual(regime, "bull")
        self.assertIsNotNone(ev)
        self.assertEqual(ev.side, TriggerSide.BUY)
        self.assertEqual(ev.action, TriggerAction.ENTRY)


class TestIntelligenceGuards(unittest.TestCase):
    def test_noise_floor_guard(self):
        guard = NoiseFloorGuard(min_strength_ratio=1.0)

        dec = guard.check("BTCUSDC", "BUY", gradient=0.5, epsilon=1.0)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)

        dec_ok = guard.check("BTCUSDC", "BUY", gradient=1.5, epsilon=1.0)
        self.assertTrue(dec_ok.allowed)
        self.assertEqual(dec_ok.brake_action, BrakeAction.NONE)

    def test_parabolic_surge_guard(self):
        guard = ParabolicSurgeGuard(surge_threshold_pct=5.0, pullback_required_pct=2.0)
        symbol = "BTCUSDC"

        history = [(1000.0, 80000.0), (1060.0, 83000.0), (1120.0, 86000.0)]

        dec = guard.check(symbol, "BUY", 86000.0, price_history=history, now=1120.0)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("parabolic_surge_active", dec.reason)

        dec_sell = guard.check(symbol, "SELL", 86000.0, price_history=history, now=1120.0)
        self.assertTrue(dec_sell.allowed)

        dec_pullback = guard.check(symbol, "BUY", 84000.0, price_history=history, now=1200.0)
        self.assertTrue(dec_pullback.allowed)

    def test_parabolic_surge_guard_v_reversal_does_not_self_disarm(self):
        guard = ParabolicSurgeGuard(surge_threshold_pct=4.0, pullback_required_pct=1.5)
        symbol = "SOLUSDC"
        # First check at pre-dip price 110.0 (no surge)
        dec1 = guard.check(symbol, "BUY", 110.0, price_history=[(900.0, 110.0), (1000.0, 110.0)], now=1000.0)
        self.assertTrue(dec1.allowed)
        # V-reversal: dips to 100.0 then surges +4.5% to 104.50; must NOT self-disarm against 110.0
        v_hist = [(1000.0, 110.0), (1060.0, 100.0), (1120.0, 104.5)]
        dec2 = guard.check(symbol, "BUY", 104.5, price_history=v_hist, now=1120.0)
        self.assertFalse(dec2.allowed)
        self.assertEqual(dec2.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("parabolic_surge_active", dec2.reason)

    def test_parabolic_surge_guard_second_surge_after_pullback(self):
        guard = ParabolicSurgeGuard(surge_threshold_pct=4.0, pullback_required_pct=1.5)
        symbol = "TAOUSDC"
        # First surge: 100 -> 106 (+6.0%), then crash to 95 (-10.38% pullback >= 1.5%),
        # then a second +4.5% vertical surge from 95 -> 99.275.
        hist = [(1000.0, 100.0), (1200.0, 106.0), (1500.0, 95.0), (1800.0, 99.275)]
        dec = guard.check(symbol, "BUY", 99.275, price_history=hist, now=1800.0)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)
        self.assertAlmostEqual(dec.metadata["peak"], 99.275, places=3)

    def test_weibull_exhaustion_guard(self):
        guard = WeibullExhaustionGuard(default_p90_days=7.0, policy="downscale", exhausted_scale=0.25)
        symbol = "BTCUSDC"

        dec_young = guard.check(symbol, "BUY", trend_duration_seconds=3.0 * 86400, p90_days=7.0)
        self.assertTrue(dec_young.allowed)
        self.assertEqual(dec_young.suggested_scale, 1.0)

        dec_old = guard.check(symbol, "BUY", trend_duration_seconds=10.0 * 86400, p90_days=7.0)
        self.assertTrue(dec_old.allowed)
        self.assertEqual(dec_old.brake_action, BrakeAction.DOWNSCALE_QTY)
        self.assertEqual(dec_old.suggested_scale, 0.25)
        self.assertIn("trend_exhausted", dec_old.reason)

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

    def test_vol_pct_and_ratio_conversions(self):
        self.assertAlmostEqual(vol_pct_to_ratio(2.5), 0.025)
        self.assertAlmostEqual(vol_pct_to_ratio(0.05), 0.0005)
        self.assertAlmostEqual(vol_ratio_to_pct(0.025), 2.5)
        self.assertAlmostEqual(vol_ratio_to_pct(0.0005), 0.05)

    def test_resolve_cached_volatility_1h(self):
        data = {
            "BTCUSDC": {"vol_1h_pct": 2.5},
            "ETHUSDT": {"vol_1h_pct": 0.05},
            "BADSYM": {"vol_1h_pct": -0.5},
            "ZEROSYM": {"vol_1h_pct": 0.0},
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            self.assertAlmostEqual(resolve_cached_volatility_1h("BTCUSDC", state_path=temp_path), 0.025)
            self.assertAlmostEqual(resolve_cached_volatility_1h_pct("BTCUSDC", state_path=temp_path), 2.5)
            self.assertAlmostEqual(resolve_cached_volatility_1h("BTCUSDT", state_path=temp_path), 0.025)
            self.assertAlmostEqual(resolve_cached_volatility_1h_pct("BTCUSDT", state_path=temp_path), 2.5)
            self.assertAlmostEqual(resolve_cached_volatility_1h("ETHUSDC", state_path=temp_path), 0.0005)
            self.assertAlmostEqual(resolve_cached_volatility_1h_pct("ETHUSDC", state_path=temp_path), 0.05)
            self.assertIsNone(resolve_cached_volatility_1h("BADSYM", state_path=temp_path))
            self.assertIsNone(resolve_cached_volatility_1h_pct("BADSYM", state_path=temp_path))
            self.assertIsNone(resolve_cached_volatility_1h("ZEROSYM", state_path=temp_path))
            self.assertIsNone(resolve_cached_volatility_1h_pct("ZEROSYM", state_path=temp_path))
            self.assertIsNone(resolve_cached_volatility_1h("NONEXISTENT", state_path=temp_path))
            self.assertIsNone(resolve_cached_volatility_1h_pct("NONEXISTENT", state_path=temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        self.assertIsNone(resolve_cached_volatility_1h("BTCUSDC", state_path="/non/existent/path.json"))
        self.assertIsNone(resolve_cached_volatility_1h_pct("BTCUSDC", state_path="/non/existent/path.json"))

    def test_get_kalman_state_path(self):
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": "/custom/cache"}):
            path = get_kalman_state_path()
            self.assertTrue(path.endswith("cache_kalman_trend.json"))
            self.assertIn("/custom/cache", path.replace("\\", "/"))

    def test_resolve_cached_volatility_1h_from_instant_trend_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trend_path = os.path.join(tmp_dir, "cache_short_trend.json")
            with open(trend_path, "w", encoding="utf-8") as f:
                json.dump({"HYPEUSD": {"vol_1h_pct": 1.85}}, f)
            with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": tmp_dir}):
                self.assertAlmostEqual(resolve_cached_volatility_1h("HYPEUSD"), 0.0185)
                self.assertAlmostEqual(resolve_cached_volatility_1h_pct("HYPEUSD"), 1.85)


class TestCompositeMarketIntelligence(unittest.TestCase):
    def test_composite_orchestration_blocks_fomo(self):
        intel = CompositeMarketIntelligence(parabolic_surge_pct=5.0)
        symbol = "BTCUSDC"

        history = [(1000.0, 80000.0), (1060.0, 83000.0), (1120.0, 86000.0)]

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
        guard = TrendSignificanceGuard(max_p_value=0.05)
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
        guard = TrendSignificanceGuard(max_p_value=0.05)
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
        guard = TrendSignificanceGuard(max_p_value=0.05)
        prices = [100.0 - 1.0 * i for i in range(24)]
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
        y = [100.0 + 0.5 * i for i in range(24)]
        _, z, p = calculate_mann_kendall(y)
        self.assertGreater(z, 0)
        self.assertLess(p, 0.01)

        self.assertEqual(classify_hurst_regime(0.65), "persistent")
        self.assertEqual(classify_hurst_regime(0.35), "mean_reverting")
        self.assertEqual(classify_hurst_regime(0.50), "random_walk")
        self.assertEqual(classify_hurst_regime(None), "unknown")


class TestIntelligenceBacktest(unittest.TestCase):
    """Verifies backtest simulation engine and guard statistics across all 4 pillars."""

    def test_run_intelligence_backtest_synthetic_series(self):
        synthetic_series = []
        base_t = 1_700_000_000.0
        p = 100.0
        for i in range(60):
            if i < 30:
                p += 0.5
            elif i < 45:
                p -= 0.8
            else:
                p += 0.3
            synthetic_series.append((base_t + (i * 300), p))

        with patch("offline.research.intelligence_backtest.load_price_series", return_value=synthetic_series):
            res = run_intelligence_backtest(symbol="TESTPAIR", cache_path="/dummy/path")

        self.assertIn("BASELINE_UNGUARDED", res)
        self.assertIn("PILLAR1_GUARDED", res)
        self.assertIn("PILLAR1_PLUS_PILLAR2", res)
        self.assertIn("FULL_COMPOSITE", res)

        for _, stats in res.items():
            self.assertIsInstance(stats, BacktestStats)
            self.assertEqual(stats.symbol, "TESTPAIR")
            self.assertGreaterEqual(stats.initial_balance, 1000.0)
            self.assertGreaterEqual(stats.final_balance, 0.0)
            self.assertGreaterEqual(stats.win_rate_pct, 0.0)
            self.assertLessEqual(stats.win_rate_pct, 100.0)
            self.assertGreaterEqual(stats.max_drawdown_pct, 0.0)
            self.assertLessEqual(stats.max_drawdown_pct, 100.0)

    def test_backtest_stats_fields_complete(self):
        stats = BacktestStats(
            symbol="BTCUSDC",
            mode="FULL_COMPOSITE",
            initial_balance=10000.0,
            final_balance=10500.0,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate_pct=70.0,
            net_profit_usd=500.0,
            net_return_pct=5.0,
            max_drawdown_pct=2.5,
            profit_factor=2.1,
            parabolic_vetoes=1,
            weibull_downscales=2,
            noise_vetoes=1,
            whale_vetoes=1,
            orderbook_vetoes=1,
            cascade_vetoes=1,
            funding_downscales=1,
            geopolitical_vetoes=1,
            sentiment_downscales=1,
        )
        self.assertEqual(stats.symbol, "BTCUSDC")
        self.assertEqual(stats.mode, "FULL_COMPOSITE")
        self.assertEqual(stats.geopolitical_vetoes, 1)
        self.assertEqual(stats.funding_downscales, 1)


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def test_daemon_initialization(temp_cache_dir):
    daemon = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC", "TAOUSDC"],
        cache_dir=temp_cache_dir,
        orderbook_interval_sec=10.0,
        derivatives_interval_sec=20.0,
        whale_interval_sec=30.0,
        macro_interval_sec=60.0,
    )
    assert daemon.symbols == ["BTCUSDC", "TAOUSDC"]
    assert daemon.orderbook_interval_sec == 10.0
    assert daemon.derivatives_interval_sec == 20.0
    assert daemon.whale_interval_sec == 30.0
    assert daemon.macro_interval_sec == 60.0


def test_daemon_run_cycle_mocked(temp_cache_dir):
    daemon = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC"],
        cache_dir=temp_cache_dir,
    )

    mock_ob_snap = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=100000.0,
        ask_depth_usd=80000.0,
        imbalance_ratio=0.55,
        largest_bid_wall_usd=20000.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=15000.0,
        largest_ask_wall_price=60100.0,
        ts=time.time(),
    )
    mock_deriv = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.0001,
        predicted_funding_rate=0.0001,
        open_interest=5000.0,
        open_interest_usd=300000000.0,
        ts=time.time(),
    )
    mock_whale = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.5,
        top_traders_long_pct=0.6,
        taker_buy_sell_ratio=1.2,
        taker_buy_vol_usd=500000.0,
        taker_sell_vol_usd=400000.0,
        open_interest_usd=300000000.0,
        open_interest_1h_change_pct=1.5,
        divergence_regime="accumulation",
        ts=time.time(),
    )
    mock_geo = GeopoliticalThreatAssessment(
        threat_level="NORMAL",
        risk_score=0.1,
        summary="Calm",
        recommended_brake="NONE",
        headlines_analyzed=5,
        ts=time.time(),
    )

    with patch.object(daemon.orderbook_collector, "fetch", return_value=mock_ob_snap), \
         patch.object(daemon.derivatives_collector, "fetch", return_value=mock_deriv), \
         patch.object(daemon.whale_collector, "fetch", return_value=mock_whale), \
         patch.object(daemon.news_collector, "fetch", return_value=MagicMock()), \
         patch.object(daemon.geopolitical_analyzer, "assess", return_value=mock_geo):

        res = daemon.run_cycle(force=True)

        assert res["orderbook_updated"] == 1
        assert res["derivatives_updated"] == 1
        assert res["whale_updated"] == 1
        assert res["macro_updated"] is True

        hb_path = os.path.join(temp_cache_dir, "intelligence_daemon.heartbeat")
        assert os.path.exists(hb_path)
        with open(hb_path, "r", encoding="utf-8") as f:
            hb = json.load(f)
        assert hb["status"] == "alive"
        assert hb["symbols"] == ["BTCUSDC"]
        assert hb["last_run"]["orderbook_updated"] == 1


def test_daemon_heartbeat_inspection(temp_cache_dir):
    hb_path = os.path.join(temp_cache_dir, "intelligence_daemon.heartbeat")
    with open(hb_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "status": "alive",
                "ts": time.time(),
                "symbols": ["BTCUSDC"],
                "intervals": {"orderbook": 15.0},
            },
            f,
        )

    status = get_daemon_status(temp_cache_dir)
    assert status["running"] is True
    assert status["pid"] == os.getpid()
    assert status["status"] == "alive"
    assert status["symbols"] == ["BTCUSDC"]


def test_inspect_symbol_intelligence(temp_cache_dir):
    ob_path = os.path.join(temp_cache_dir, "orderbook_depth_BTCUSDC.json")
    with open(ob_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": "BTCUSDC",
                "mid_price": 65000.0,
                "bid_depth_usd": 500000.0,
                "ask_depth_usd": 400000.0,
                "imbalance_ratio": 0.556,
                "largest_bid_wall_usd": 50000.0,
                "largest_bid_wall_price": 64900.0,
                "largest_ask_wall_usd": 30000.0,
                "largest_ask_wall_price": 65100.0,
                "ts": time.time(),
            },
            f,
        )

    with patch("order_guard.check_intelligence_guards", return_value=(True, "ok", 1.0)):
        report = inspect_symbol_intelligence("BTCUSDC", cache_dir=temp_cache_dir)
    assert report["symbol"] == "BTCUSDC"
    assert report["pillar2_external"]["orderbook"]["mid_price"] == 65000.0
    assert report["pillar2_external"]["orderbook"]["imbalance_ratio"] == 0.556
    assert "buy_standard" in report["guard_preview"]


def test_parabolic_surge_guard_ignores_pre_dip_high_and_disarms_cleanly():
    guard = ParabolicSurgeGuard(surge_threshold_pct=5.0, pullback_required_pct=2.0, window_seconds=7200.0)
    # High (100.0) occurred BEFORE trough (94.0); recovery to 98.5 is +4.79% (< 5.0% threshold).
    history_dip_recovery = [(1000.0, 100.0), (1500.0, 94.0), (2000.0, 98.5)]
    dec = guard.check("ETHUSDC", "BUY", 98.5, price_history=history_dip_recovery, now=2000.0)
    assert dec.allowed is True

    # Real upward surge (80000 -> 86000 = +7.5%), followed by pullback to 84000 (-2.33%), then rebound to 85000 (+1.19%)
    history_surge = [(1000.0, 80000.0), (1060.0, 83000.0), (1120.0, 86000.0)]
    dec1 = guard.check("BTCUSDC", "BUY", 86000.0, price_history=history_surge, now=1120.0)
    assert dec1.allowed is False

    history_pullback = history_surge + [(1200.0, 84000.0)]
    dec2 = guard.check("BTCUSDC", "BUY", 84000.0, price_history=history_pullback, now=1200.0)
    assert dec2.allowed is True

    # Subsequent modest bounce to 85000 should not re-trigger on the already-disarmed 1120.0 peak
    history_rebound = history_pullback + [(1300.0, 85000.0)]
    dec3 = guard.check("BTCUSDC", "BUY", 85000.0, price_history=history_rebound, now=1300.0)
    assert dec3.allowed is True


def test_global_telemetry_and_macro_analyzer_preserve_breadth_and_fg_fields(temp_cache_dir):
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

    collector = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=temp_cache_dir,
        skip_news=True,
        skip_external=True,
    )
    fg_in = FearGreedSnapshot(
        value=18,
        classification="Extreme Fear",
        timestamp=1700000000.0,
        historical_values=(18, 29, 35),
        trend_7d_change=-17,
        is_extreme_fear=True,
        is_extreme_greed=False,
    )
    mb_in = MarketBreadthSnapshot(
        advance_ratio=0.12,
        advancing_count=12,
        declining_count=88,
        total_count=100,
        median_change_pct=-5.4,
        mean_change_pct=-6.1,
        dispersion_std=3.8,
        regime="CAPITULATION",
        top_gainers=(("BTCUSDC", 1.2),),
        top_losers=(("ALTUSDC", -14.5),),
        ts=1700000000.0,
    )
    with patch.object(collector.fear_greed_collector, "fetch", return_value=fg_in), \
         patch.object(collector.breadth_collector, "fetch", return_value=mb_in):
        assert collector.update_fear_greed(force=True) is True
        assert collector.update_market_breadth(force=True) is True

    analyzer = MacroAnalyzer(
        cache_dir=temp_cache_dir,
        macro_llm_interval_sec=3600.0,
        advisor_interval_sec=3600.0,
        symbols=["BTCUSDC"],
    )
    captured = {}

    def _fake_review(*, fear_greed=None, breadth=None, whale_telemetry=None, force_refresh=False, now_ts=None, **_kwargs):
        captured["fg"] = fear_greed
        captured["mb"] = breadth
        return None

    with patch.object(analyzer.gemini_advisor, "review", side_effect=_fake_review):
        analyzer.assess_market_advisor(force=True)

    assert captured["fg"].historical_values == (18, 29, 35)
    assert captured["mb"].regime == "CAPITULATION"
    assert captured["mb"].mean_change_pct == pytest.approx(-6.1)
    assert captured["mb"].dispersion_std == pytest.approx(3.8)
    assert captured["mb"].top_gainers == (("BTCUSDC", 1.2),)
    assert captured["mb"].top_losers == (("ALTUSDC", -14.5),)


def test_high_stake_guard_sell_uses_sell_label():
    from intelligence.sentiment.guards import high_stake_guard as hsg_mod
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard

    hsg_mod._HIGH_STAKE_DECISION_CACHE.clear()
    mock_client = MagicMock()
    mock_client.query_json.return_value = {
        "decision": "REJECTED",
        "suggested_scale": 0.0,
        "reason": "dumping into whale support",
    }
    guard = HighStakeGuard(min_notional_eur=1000.0, gemini_client=mock_client, cache_ttl_sec=0.0)
    dec = guard.check("BTCUSDC", "SELL", 85000.0, 0.02, notional_eur=1700.0, telemetry={"flow": 1.2})
    assert dec.allowed is False
    assert "Gemini vetoed high-stake SELL: dumping into whale support" in dec.reason
    call_kwargs = mock_client.query_json.call_args.kwargs
    assert "SELL BTCUSDC" in call_kwargs["thread_title"]


def test_composite_evaluate_guards_does_not_skip_veto_after_downscale_and_unpacks_liquidation():
    from intelligence.external.collectors.bybit_liquidations import LiquidationSummary

    intel = CompositeMarketIntelligence(exhaustion_policy="downscale", exhaustion_scale=0.25)
    shock = GeopoliticalThreatAssessment(
        threat_level="CRITICAL_SHOCK",
        risk_score=0.95,
        summary="Acute crisis",
        recommended_brake="HARD_VETO_NEW_BUYS",
        headlines_analyzed=5,
        ts=1000.0,
    )
    # Even when Weibull exhaustion downscales (trend_duration_seconds = 10 days > P90),
    # a subsequent CRITICAL_SHOCK geopolitical veto must NOT be skipped!
    dec = intel.evaluate_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        trend_duration_seconds=10.0 * 86400.0,
        geopolitical_assessment=shock,
        now=1000.0,
    )
    assert dec.allowed is False
    assert dec.brake_action == BrakeAction.HARD_VETO

    # Also verify CompositeMarketIntelligence.evaluate handles liquidation_summary without unpacking TypeError
    liq_summary = LiquidationSummary(
        symbol="BTCUSDC",
        window_sec=60.0,
        long_liq_usd=150_000.0,
        short_liq_usd=10_000.0,
        net_usd=-140_000.0,
        capitulation_ratio=150_000.0 / 160_000.0,
        squeeze_ratio=10_000.0 / 160_000.0,
        events_count=5,
        last_event_ts=1000.0,
    )
    eval_res = intel.evaluate("BTCUSDC", 85000.0, ts=1000.0, liquidation_summary=liq_summary)
    assert eval_res.active_trigger is not None
    assert eval_res.active_trigger.source == "LiquidationCapitulationTrigger"


def test_survival_estimate_t_writes_cache_without_name_error(temp_cache_dir):
    from intelligence.internal.state import survival

    fake_ts = np.arange(0, 200 * 3600, 3600, dtype=float)
    fake_px = 100.0 + np.sin(np.linspace(0, 8 * np.pi, len(fake_ts))) * 10.0
    temp_t_cache = os.path.join(temp_cache_dir, "cache_T_trend.json")

    with patch.object(survival, "T_CACHE_FILE", temp_t_cache), \
         patch.object(survival, "fetch_klines", return_value=(fake_ts, fake_px)):
        res = survival.estimate_T("TESTCOIN", days=10)
        assert "T" in res
        assert os.path.exists(temp_t_cache)


def test_whale_collector_short_building_and_ignore_ttl_prev_regime(temp_cache_dir):
    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector

    reg = WhalePositioningCollector.classify_regime(
        price_1h_change_pct=-0.5,
        oi_change_pct=0.6,
        taker_ratio=0.90,
    )
    assert reg == "short_building"

    collector = WhalePositioningCollector(cache_ttl_sec=10.0, disk_ttl_sec=10.0, cache_dir=temp_cache_dir)
    old_snap = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.2,
        top_traders_long_pct=0.55,
        taker_buy_sell_ratio=0.75,
        taker_buy_vol_usd=100.0,
        taker_sell_vol_usd=133.0,
        open_interest_usd=1000.0,
        open_interest_1h_change_pct=1.5,
        divergence_regime="aggressive_shorting",
        ts=100.0,
    )
    collector._save_to_disk(old_snap)
    assert collector._load_from_disk("BTCUSDC", now=500.0, ignore_ttl=False) is None
    loaded_expired = collector._load_from_disk("BTCUSDC", now=500.0, ignore_ttl=True)
    assert loaded_expired is not None
    assert loaded_expired.divergence_regime == "aggressive_shorting"


def test_survival_metrics_returns_gaussian_horizon_days_and_respects_allow_network_false(temp_cache_dir):
    from intelligence.internal.state import survival

    temp_t_cache = os.path.join(temp_cache_dir, "cache_T_trend.json")
    with open(temp_t_cache, "w", encoding="utf-8") as f:
        json.dump({"TAOUSDC": {"T": 6.0, "p90_d": 4.2, "median_d": 2.0, "ts": time.time()}}, f)

    with patch.object(survival, "T_CACHE_FILE", temp_t_cache), \
         patch.object(survival, "estimate_T", side_effect=AssertionError("must not call estimate_T")) as mock_est:
        m_hit = survival.get_trend_survival_metrics("TAOUSD", trend_duration_seconds=5.0 * 86400.0)
        assert m_hit["T"] == 6.0
        assert m_hit["gaussian_horizon_days"] == pytest.approx(8.4)
        m_miss = survival.get_trend_survival_metrics("UNLISTEDUSD", trend_duration_seconds=5.0 * 86400.0)
        assert m_miss["gaussian_horizon_days"] is None
        mock_est.assert_not_called()


def test_position_exit_manager_evaluate_hard_tp_cooldown_does_not_double_initiate_ride():
    from strategies.position_exit_manager import (
        ExitAction,
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
    )

    cfg = PositionExitConfig(
        hard_tp_enabled=True,
        hard_tp_pct=0.20,
        hard_tp_frac=0.50,
        hard_tp_cooldown_sec=3600.0,
        gain_threshold=0.07,
        tp_trend_ride_enabled=True,
    )
    mgr = PositionExitManager(config=cfg)
    mgr.record_hard_tp("TAOUSDC", timestamp=1000.0)
    snap = PositionSnapshot(
        symbol="TAOUSDC",
        current_price=125.0,
        avg_buy_price=100.0,
        current_time_s=1100.0,
        trend_up=True,
    )
    dec = mgr.evaluate(snap)
    assert dec.action == ExitAction.HOLD
    assert dec.metadata.get("state") == "initiated"


def test_external_collectors_and_ssot_prompt_modularization():
    from intelligence.external.collectors import base_collector as bc
    from intelligence.external.collectors import derivatives_telemetry as dt
    from intelligence.external.collectors import orderbook_depth as ob
    from intelligence.external.collectors import whale_positioning as wp
    from intelligence.macro import calendar_ssot as cal_ssot
    from intelligence.macro import lexicon_ssot as lex_ssot
    import intelligence.macro_sentiment_ssot as ssot
    from orchestratorOS.supervisor import prompt_formatters as pf
    from orchestratorOS.supervisor import prompt_sections as ps

    for cls in (ob.OrderbookDepthCollector, dt.DerivativesTelemetryCollector, wp.WhalePositioningCollector):
        assert issubclass(cls, bc._DiskBackedSymbolCollector)
        assert hasattr(cls, "_fetch_hyperliquid_asset_ctx")

    assert ob._DiskBackedSymbolCollector is bc._DiskBackedSymbolCollector
    assert ssot.match_high_severity_keywords is lex_ssot.match_high_severity_keywords
    assert ssot.resolve_macro_economic_calendar is cal_ssot.resolve_macro_economic_calendar
    assert pf._format_macro_and_news_section is ps._format_macro_and_news_section
    assert pf._format_microstructure_section is ps._format_microstructure_section
    assert pf._format_portfolio_section is ps._format_portfolio_section


def test_intelligence_architecture_alignment_collectors_evaluators_and_daemon_ssot(tmp_path):
    """Verify Step 1-4 modularization: *_collector.py naming, microstructure/portfolio boundary, tca_risk_evaluator, and daemon_ssot."""
    import json
    import time as _time
    import intelligence.external.collectors as ext_cols
    from intelligence.external.collectors import (
        bybit_liquidations,
        bybit_liquidations_collector,
        derivatives_telemetry,
        derivatives_telemetry_collector,
        orderbook_depth,
        orderbook_depth_collector,
        whale_positioning,
        whale_positioning_collector,
    )
    from intelligence.macro import daemon_ssot
    import intelligence.macro_sentiment_ssot as ssot
    from intelligence.sentiment.collectors import (
        macro_telemetry_collector,
        microstructure_collector,
        portfolio_collector,
        pretrade_telemetry_collector,
    )
    from intelligence.sentiment.evaluators import portfolio_evaluator, tca_risk_evaluator

    # 1. External collectors follow *_collector.py naming with identical module aliases for backwards compatibility
    assert bybit_liquidations is bybit_liquidations_collector
    assert derivatives_telemetry is derivatives_telemetry_collector
    assert orderbook_depth is orderbook_depth_collector
    assert whale_positioning is whale_positioning_collector
    assert ext_cols.OrderbookDepthCollector is orderbook_depth_collector.OrderbookDepthCollector

    # 2. Microstructure, pretrade, and portfolio collectors have clean boundaries and re-exports
    assert microstructure_collector.enrich_24h_and_microstructure is portfolio_collector.enrich_24h_and_microstructure
    assert macro_telemetry_collector.collect_pretrade_telemetry is pretrade_telemetry_collector.collect_pretrade_telemetry
    assert callable(portfolio_collector.load_symbol_best_recent_fills)
    assert callable(portfolio_collector._attach_trailing_peak_to_position)
    assert callable(portfolio_collector._apply_open_orders_locked_balances)
    assert ssot.configure_daemon_cli_logging is daemon_ssot.configure_daemon_cli_logging

    orders_cache = {
        "items": {
            "BTCUSDC": [
                {"orderId": 101, "symbol": "BTCUSDC", "side": "BUY", "status": "FILLED", "price": 75000.0, "executedQty": 0.1, "time": int(_time.time() * 1000)},
            ]
        }
    }
    (tmp_path / "cache_order.json").write_text(json.dumps(orders_cache), encoding="utf-8")
    fills = portfolio_collector.load_symbol_best_recent_fills(
        str(tmp_path),
        "BTCUSDC",
        ["BTCUSDC"],
    )
    assert isinstance(fills, dict) and (fills.get("last_buy") or {}).get("price") == 75000.0

    # 3. TCA & risk evaluator owns TCA/risk/cross-market evaluators and is re-exported cleanly
    assert portfolio_evaluator.compute_enterprise_portfolio_risk_metrics is tca_risk_evaluator.compute_enterprise_portfolio_risk_metrics
    assert portfolio_evaluator.enrich_open_limit_orders_tca is tca_risk_evaluator.enrich_open_limit_orders_tca
    assert portfolio_evaluator.enrich_recent_orders_markout is tca_risk_evaluator.enrich_recent_orders_markout
    assert portfolio_evaluator.build_cross_market_context is tca_risk_evaluator.build_cross_market_context

    # 4. Daemon SSOT owns daemon runtime primitives & base collector/agent classes and is re-exported in macro_sentiment_ssot
    assert ssot.CachedHttpJsonCollector is daemon_ssot.CachedHttpJsonCollector
    assert ssot.CachedTopicAssessmentAgent is daemon_ssot.CachedTopicAssessmentAgent
    assert ssot.ExternalTelemetryMixin is daemon_ssot.ExternalTelemetryMixin
    assert ssot.run_daemon_loop is daemon_ssot.run_daemon_loop
    assert ssot.write_daemon_heartbeat is daemon_ssot.write_daemon_heartbeat


def test_iter44_intents_dict_unwrap_read_json_cached_and_orderbook_wall_and_bucket_dedup(tmp_path, monkeypatch):
    """Verify dict-wrapped intents unwrapping, read_json_cached list-root fallback, whale wall preservation, and nSigFigs bucket dedup."""
    import json
    from types import SimpleNamespace
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        harvest_macro_and_news_sensors,
        populate_market_and_portfolio_snapshot,
    )
    from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
    from providers.execution_audit import (
        _filter_intents_by_scope,
        collect_known_ai_order_ids,
        extract_intent_notional,
        summarize_recent_ai_intents,
    )
    from state_io import read_json_dict_recovering as read_json_cached

    now_ts = 1_780_000_000.0
    intents_payload = {
        "intents": [
            {
                "intent_id": "ai-hl-1",
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "decision": "BUY",
                "order_type": "LIMIT",
                "status": "EXECUTED",
                "created_at": now_ts - 120.0,
                "executed_at": now_ts - 60.0,
                "notional_usd": 300.0,
                "execution_payload": {"orderId": "998877", "status": "FILLED", "avgPrice": "85.50", "executedQty": "3.5"},
            }
        ]
    }
    assert "998877" in collect_known_ai_order_ids(intents_payload)
    assert len(_filter_intents_by_scope(intents_payload, "crypto")) == 1
    assert len(summarize_recent_ai_intents(intents_payload, "crypto", now_ts, extract_intent_notional)) == 1

    (tmp_path / "autonomous_trade_intents_state.json").write_text(json.dumps(intents_payload), encoding="utf-8")
    (tmp_path / "news_feed_collect.json").write_text(
        json.dumps([{"title": "Federal Reserve signals steady policy rate path", "source": "Reuters", "first_seen_ts": now_ts - 60.0}]),
        encoding="utf-8",
    )

    fills = load_symbol_best_recent_fills(str(tmp_path), "HYPEUSDC", ["HYPEUSDC", "HYPE"])
    assert isinstance(fills, dict)
    assert (fills.get("last_buy") or {}).get("actor") == "AI"
    assert (fills.get("last_buy") or {}).get("price") == 85.5

    _, breaking = harvest_macro_and_news_sensors(str(tmp_path), now_ts, read_json_fn=read_json_cached)
    assert len(breaking) == 1
    assert "Federal Reserve" in breaking[0]["title"]

    snap_obj = SimpleNamespace(
        breaking_news=[],
        prices_and_trends={},
        microstructure={},
        portfolio_summary={},
    )
    gw_stub = SimpleNamespace(load_all_intents=lambda: intents_payload, max_notional_usd=2000.0)
    populate_market_and_portfolio_snapshot(
        snap_obj,
        cache_dir=str(tmp_path),
        workspace_dir=str(tmp_path),
        logs_dirs=[str(tmp_path)],
        intent_gateway=gw_stub,
        scope="crypto",
        now_ts=now_ts,
        read_json_fn=read_json_cached,
    )
    assert len(snap_obj.breaking_news) == 1
    assert len(snap_obj.portfolio_summary.get("recent_ai_intents") or []) == 1

    # OrderbookDepthCollector: preserve >= whale_wall_usd wall inside 10 bps & avoid nSigFigs boundary bucket double-count
    ob_col = OrderbookDepthCollector(whale_wall_usd=500_000.0, cache_ttl_sec=60.0, cache_dir=None)

    def _fake_hl_book(url, label, lookup_sym, **kwargs):
        if "nSigFigs=4" in url:
            return {
                "levels": [
                    # 85.74 coarse bucket overlaps fine 85.742 bid; 85.20 is clean deeper level
                    [{"px": "85.74", "sz": "900.0"}, {"px": "85.20", "sz": "200.0"}],
                    # 85.77 coarse bucket overlaps fine 85.763 ask; 86.20 has only $862 dust
                    [{"px": "85.77", "sz": "900.0"}, {"px": "86.20", "sz": "10.0"}],
                ]
            }
        return {
            "levels": [
                # Massive $600k whale wall at 85.742 (<10 bps from mid ~85.751)
                [{"px": "85.750", "sz": "100.0"}, {"px": "85.742", "sz": "7000.0"}],
                # $50k ask wall at 85.763 (<10 bps from mid) vs $862 dust at 86.20
                [{"px": "85.752", "sz": "100.0"}, {"px": "85.763", "sz": "600.0"}],
            ]
        }

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_hl_book)
    ob_snap = ob_col.fetch("HYPEUSDC", allow_network=True, now=now_ts)
    assert ob_snap is not None
    # Whale wall ($600k >= $500k threshold) at 85.742 is preserved over $17,040 order at 85.20
    assert ob_snap.largest_bid_wall_price == 85.742
    assert ob_snap.largest_bid_wall_usd > 500_000.0
    # Ask wall ($51.4k >= $5k) at 85.763 is preserved over $862 (< $5k) dust at 86.20
    assert ob_snap.largest_ask_wall_price == 85.763
    # Overlapping boundary buckets 85.74 (bid) and 85.77 (ask) were excluded from depth_usd
    expected_bid_depth = 85.750 * 100.0 + 85.742 * 7000.0 + 85.20 * 200.0
    assert abs(ob_snap.bid_depth_usd - expected_bid_depth) < 1.0


def test_iter45_hl_spot_fallbacks_portfolio_per_symbol_fills_and_interval_plus_sign(tmp_path):
    """Verify HLClient spot_meta empty-dict fallback & suffix stripping, per-symbol audit/intent fills in harvest_recent_orders_and_enrich_positions, chronological open-order audit timestamps, and '+60' OHLC interval parsing."""
    from types import SimpleNamespace
    from providers.hyperliquid_provider import _hl_client_module
    from providers.base import normalize_open_order, parse_ohlc_interval_minutes
    from intelligence.sentiment.collectors.portfolio_collector import (
        _load_open_order_timestamps,
        harvest_recent_orders_and_enrich_positions,
    )

    HLClient = _hl_client_module().HLClient
    # 1. HLClient.resolve_spot_pair & sz_decimals handle empty spot_meta() and /USDC or USDC suffixes
    hl = HLClient.__new__(HLClient)
    hl._meta_cache = {}
    hl.info = SimpleNamespace(
        spot_meta=lambda: {},
        name_to_coin={"HYPE/USDC": "@107"},
        coin_to_asset={"@107": 10107},
        asset_to_sz_decimals={10107: 2},
    )
    assert hl.resolve_spot_pair("HYPE") == "@107"
    assert hl.resolve_spot_pair("HYPE/USDC") == "@107"
    assert hl.resolve_spot_pair("HYPEUSDC") == "@107"
    assert hl.sz_decimals("HYPE/USDC") == 2
    assert hl.sz_decimals("HYPEUSDC") == 2

    # 2. parse_ohlc_interval_minutes("+60") and normalize_open_order fallback symbol
    assert parse_ohlc_interval_minutes("+60") == 60
    norm_oo = normalize_open_order({"orderId": 42, "symbol": "btcusdc", "side": "BUY", "price": 90000.0, "origQty": 0.1}, venue="binance")
    assert norm_oo is not None and norm_oo["symbol"] == "BTCUSDC"

    # 3. _load_open_order_timestamps does not short-circuit when cached_bals is non-empty and preserves earliest audit ts
    (tmp_path / "execution_audit_001.jsonl").write_text(
        json.dumps({"ts": 1700001000.0, "event": "submit_accepted", "order_id": "oid_span_1"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "execution_audit_002.jsonl").write_text(
        json.dumps({"ts": 1700002000.0, "event": "order_status", "order_id": "oid_span_1"}) + "\n",
        encoding="utf-8",
    )
    ts_map = _load_open_order_timestamps(
        str(tmp_path),
        {"open_orders:binance:BTCUSDC": [{"order_id": "preexisting_oid", "created_ts": 1700000500.0}]},
    )
    assert ts_map.get("preexisting_oid") == 1700000500.0
    assert ts_map.get("oid_span_1") == 1700001000.0

    # 4. harvest_recent_orders_and_enrich_positions populates per-symbol audit/AI fills even when another symbol is already in venue_recent_orders
    (tmp_path / "cache_trade_t212.json").write_text(
        json.dumps({
            "items": {
                "AAPL": [{"orderId": "t212_aapl_1", "side": "BUY", "price": 220.0, "qty": 1.0, "timestamp": 1700001000000, "status": "FILLED"}],
            }
        }),
        encoding="utf-8",
    )
    ai_intents = [
        {
            "intent_id": "ai_nvda_1",
            "venue": "t212",
            "symbol": "NVDA",
            "action": "BUY",
            "status": "EXECUTED",
            "executed_at": 1700001500.0,
            "notional_usd": 540.0,
            "execution_payload": {"orderId": "t212_nvda_1", "price": 135.0, "executedQty": 4.0},
        },
        {
            "intent_id": "ai_sol_1",
            "venue": "binance",
            "symbol": "SOLUSDC",
            "action": "BUY",
            "status": "EXECUTED",
            "executed_at": 1700001600.0,
            "notional_usd": 360.0,
            "execution_payload": {"orderId": "bin_sol_1", "price": 180.0, "executedQty": 2.0},
        },
    ]
    recent_orders = harvest_recent_orders_and_enrich_positions(
        str(tmp_path),
        {},
        [],
        {},
        {},
        {},
        {},
        known_ai_order_ids={"t212_nvda_1", "bin_sol_1"},
        ai_intents=ai_intents,
    )
    assert "AAPL" in recent_orders
    assert "NVDA" in recent_orders
    assert (recent_orders["NVDA"].get("last_buy") or {}).get("price") == 135.0
    assert "SOLUSDC" in recent_orders
    assert (recent_orders["SOLUSDC"].get("last_buy") or {}).get("price") == 180.0


def test_iter46_audited_executor_precision_daemon_ssot_monkeypatch_and_tca_hl_casefold(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import intelligence.macro.daemon_ssot as daemon_ssot
    from intelligence.sentiment.evaluators.tca_risk_evaluator import resolve_venue_fee_and_limits
    from providers.base import PairPrecision
    from providers.execution_audit import AuditedStrategyExecutor, ExecutionAudit

    # 1. AuditedStrategyExecutor fallback to pair_precision for round_quantity/round_price/min_order_qty and trend_snapshot(max_age_sec)
    class _MinimalExec:
        name = "test_exec"

        def pair_precision(self, symbol: str):
            return PairPrecision(price_decimals=2, volume_decimals=3, order_min=0.05)

        def trend_snapshot(self, symbol: str, max_age_sec: float = 15.0):
            return {
                "symbol": symbol,
                "max_age_sec": max_age_sec,
                "growth_coefficient": 120.0,
                "gradient_recent": 120.0,
                "slope_small": 0.05,
                "final_trend": 120.0,
            }

        def trend(self, symbol: str, max_age_sec: float = 15.0) -> float:
            return float(max_age_sec)

    audited = AuditedStrategyExecutor(_MinimalExec(), ExecutionAudit(str(tmp_path)))
    assert audited.round_quantity("BTCUSDC", 1.23456) == 1.234
    assert audited.round_price("BTCUSDC", 99.987) == 99.99
    assert audited.min_order_qty("BTCUSDC") == 0.05
    assert audited.trend_snapshot("BTCUSDC", max_age_sec=42.0)["max_age_sec"] == 42.0
    assert audited.trend("BTCUSDC", max_age_sec=42.0) == 42.0
    assert audited.trend_direction("BTCUSDC", max_age_sec=42.0) == "up"

    # 2. CachedHttpJsonCollector respects monkeypatching intelligence.macro.daemon_ssot.fetch_public_json
    class _DummyCollector(daemon_ssot.CachedHttpJsonCollector):
        def __init__(self):
            super().__init__(
                cache_ttl_sec=60.0,
                default_ttl_sec=60.0,
                endpoint_url="https://example.test/api",
                default_endpoint_fn=lambda: "https://example.test/api",
            )

        def _parse_fetched_payload(self, data, now):
            return {"ok": True, "val": data.get("val"), "ts": now}

    monkeypatch.setattr(daemon_ssot, "fetch_public_json", lambda url, **kw: {"val": 777})
    coll = _DummyCollector()
    snap = coll.fetch(force_refresh=True, now_ts=1700000000.0)
    assert snap == {"ok": True, "val": 777, "ts": 1700000000.0}

    # 3. resolve_venue_fee_and_limits handles case-insensitive Hyperliquid caches and _sz_decimals_cache
    fake_hl_prov = SimpleNamespace(
        _spot_pair=None,
        _meta_cache={"hype": {"szDecimals": 2}},
        _client=SimpleNamespace(_sz_decimals_cache={"btc": 5}, info=None),
    )
    fake_inst = SimpleNamespace(_provider=fake_hl_prov, params={})
    hl_lim = resolve_venue_fee_and_limits("hyperliquid", "HYPEUSDC", fake_inst, price=25.0)
    assert hl_lim["step_size"] == 0.01
    assert hl_lim["min_qty"] == 0.01

    hl_btc_lim = resolve_venue_fee_and_limits("hyperliquid", "BTCUSDC", fake_inst, price=90000.0)
    assert hl_btc_lim["step_size"] == 0.00001
    assert hl_btc_lim["min_qty"] == 0.00001


def test_iter47_lazy_intelligence_imports_monkeypatch_forwarding_and_strict_int_fields(tmp_path, monkeypatch):
    import subprocess
    import sys

    import cacheManager
    import cacheintegrity
    import intelligence
    import intelligence.sentiment.collectors.macro_telemetry_collector as mtc
    import intelligence.sentiment.collectors.pretrade_telemetry_collector as ptc
    import intelligence.sentiment.guards.post_stop_reentry_guard as psrg

    # 1. Importing macro_sentiment_ssot and GuardDecision works even when numpy is unavailable
    code = (
        "import sys; sys.modules['numpy'] = None; "
        "import intelligence; "
        "from intelligence.macro_sentiment_ssot import clean_symbol_key; "
        "from intelligence.internal.guards.guard_decision import GuardDecision; "
        "assert intelligence.GuardDecision is GuardDecision; "
        "assert clean_symbol_key('tao/usdc') == 'TAOUSDC'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "KalmanTrendTrigger" in dir(intelligence)

    # 2. PostStopReentryGuard respects monkeypatching on post_stop_reentry_guard module globals
    psrg.clear_post_stop_reentry_cache()
    seen_prompts = []
    monkeypatch.setattr(psrg, "collect_pretrade_telemetry", lambda sym, **kw: {"CustomMetric": f"patched-{sym}"})
    monkeypatch.setattr(
        psrg,
        "query_topic_json",
        lambda client, topic, prompt, **kw: (
            seen_prompts.append((topic, prompt)) or {"decision": "DOWNSCALE", "suggested_scale": 0.4, "reason": "cautious"}
        ),
    )
    guard = psrg.PostStopReentryGuard(gemini_client=object(), notify_enabled=False, cache_ttl_sec=0.0)
    dec = guard.check("TAOUSDC", current_price=410.0, last_sell_price=400.0, trough=380.0, bounce_pct=0.05)
    assert len(seen_prompts) == 1
    assert seen_prompts[0][0] == "guard_post_stop"
    assert "CustomMetric: patched-TAOUSDC" in seen_prompts[0][1]
    assert dec.suggested_scale == 0.4

    # 3. collect_pretrade_telemetry respects monkeypatching read_json_cached on macro_telemetry_collector
    monkeypatch.setattr(
        mtc,
        "read_json_cached",
        lambda path: {"threat_level": "CRITICAL_PATCHED"} if "geopolitical_eval" in str(path) else None,
    )
    telem = ptc.collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(tmp_path))
    assert telem.get("Geopolitical Threat Level") == "CRITICAL_PATCHED"

    # 4. _strict_int_field rejects bytes and non-ASCII digits and validate_account_cache_state handles '42.0'
    assert cacheManager._strict_int_field(b"123", min_value=1) is None
    assert cacheManager._strict_int_field("१२३", min_value=1) is None
    assert cacheManager._strict_int_field("123", min_value=1) == 123
    with pytest.raises(ValueError):
        cacheintegrity._strict_int_field("१२३")
    assert cacheintegrity._strict_int_field("42.0") == 42
    cacheintegrity.validate_account_cache_state(
        cls_name="CacheOrderManager",
        is_trade_cache=False,
        is_order_cache=True,
        cache={"BTCUSDC": [{"symbol": "BTCUSDC", "orderId": "42.0", "_fillIds": ["7.0"]}]},
        fetchtime_per_symbol={"BTCUSDC": 1700000000000},
        rest_sync_cursor_per_symbol={},
        is_valid_item_fn=lambda item: True,
    )


def test_iter48_portfolio_locked_balances_ws_enqueue_threadsafe_and_bool_env_pathlike(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import botcore
    from binance_api.bapi_ws import BinancePriceStream, PublicPriceStreamBase
    from intelligence.sentiment.collectors.portfolio_collector import (
        _apply_open_orders_locked_balances,
        _attach_trailing_peak_to_position,
    )
    import providers.market_api as market_api
    from providers.base import MarketDataProvider

    # 1. _apply_open_orders_locked_balances resolves Hyperliquid USDC quote and matches HYPEUSDC order to HYPE position
    open_orders = [
        {"venue": "hyperliquid", "symbol": "HYPE", "side": "BUY", "price": 25.0, "qty": 4.0},
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "side": "SELL", "price": 30.0, "qty": 2.0},
    ]
    venue_quote_cash = {"hyperliquid:USDC": {"venue": "hyperliquid", "asset": "USDC", "free_quote_usd": 500.0}}
    tradable_positions = [
        {
            "venue": "hyperliquid",
            "symbol": "HYPE",
            "base": "HYPE",
            "quote": "USDC",
            "price": 28.0,
            "free_base": 8.0,
            "free_quote": 500.0,
        }
    ]
    _apply_open_orders_locked_balances(
        open_orders,
        tradable_positions,
        venue_quote_cash,
        {},
        {"HYPE": {"price": 28.0}},
        {},
        {},
        now_ts_oo=1700000000.0,
        gateway_max_usd=2000.0,
    )
    assert venue_quote_cash["hyperliquid:USDC"].get("locked_quote_usd") == 100.0
    assert tradable_positions[0].get("locked_base_qty") == 2.0
    assert tradable_positions[0].get("locked_quote_usd") == 100.0

    # 2. _attach_trailing_peak_to_position clears stale trailing_peak and dd_from_peak_pct when peak is None/0
    pos_entry = {"trailing_peak": 50.0, "dd_from_peak_pct": -10.0}
    _attach_trailing_peak_to_position(pos_entry, "hyperliquid", "HYPE", 45.0, 10.0, None, {})
    assert "trailing_peak" not in pos_entry
    assert "dd_from_peak_pct" not in pos_entry

    # 3. PublicPriceStreamBase handles non-iterable symbols=True and BinancePriceStream._enqueue_cmd uses call_soon_threadsafe
    base_stream = PublicPriceStreamBase(symbols=True)
    assert base_stream._symbols == ()

    b_stream = BinancePriceStream(symbols=["BTCUSDC"])
    scheduled = []
    b_stream._cmd_queue = SimpleNamespace(put_nowait=lambda x: None)
    b_stream._loop = SimpleNamespace(
        is_running=lambda: True,
        call_soon_threadsafe=lambda fn, *args: scheduled.append((fn, args)),
    )
    b_stream._enqueue_cmd("SUBSCRIBE", "ETHUSDC")
    assert len(scheduled) == 1
    assert scheduled[0][1] == ({"method": "SUBSCRIBE", "stream": "ethusdc@ticker"},)

    # 4. botcore.bool_env & env_values_map accept pathlib.Path and fall back when os.environ value is blank
    (tmp_path / ".env").write_text("TEST_ITER48_FLAG=1\nTEST_ITER48_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("TEST_ITER48_FLAG", "   ")
    assert botcore.bool_env("TEST_ITER48_FLAG", tmp_path, False) is True
    assert botcore.bool_env(tmp_path, "TEST_ITER48_FLAG", False) is True
    assert botcore.env_values_map(tmp_path, ["TEST_ITER48_KEY"]).get("TEST_ITER48_KEY") == "from_file"

    # 5. MarketDataProvider.resolve_venue_instrument_spec prioritizes exact match over canonical alias & respects monkeypatched registry
    alias_item = SimpleNamespace(venue="kraken", symbol="TAOUSD", base="TAO", quote="USD")
    exact_item = SimpleNamespace(venue="kraken", symbol="TAOUSDC", base="TAO", quote="USDC")
    fake_ir = SimpleNamespace(select_instruments=lambda v: {"TAOUSD": alias_item, "TAOUSDC": exact_item})
    monkeypatch.setattr(market_api, "instrument_registry", fake_ir, raising=False)
    resolved = MarketDataProvider.resolve_venue_instrument_spec("kraken", "TAOUSDC")
    assert resolved is exact_item
    assert resolved.quote == "USDC"


def test_iter49_lazy_sentiment_subpackages_gemini_quota_monkeypatch_and_evaluator_hardening(tmp_path, monkeypatch):
    import subprocess
    import sys
    import intelligence.sentiment.gemini_client as gc_mod
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        _populate_cycle_fallback_recent_order,
        reconcile_venue_cycles_with_recent_orders,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import compute_enterprise_portfolio_risk_metrics
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_cross_venue_trends

    # 1. Importing collectors/evaluators subpackages when numpy is blocked succeeds via PEP 562 lazy loading
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['numpy'] = None; "
                "import intelligence.sentiment.collectors.portfolio_collector; "
                "import intelligence.sentiment.evaluators.tca_risk_evaluator; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "OK"

    import intelligence.internal.state.volatility as vol_mod
    prices_sample = [100.0 + (i % 3) * 0.5 for i in range(25)]
    vol_with_np = vol_mod.calculate_volatility_1h(prices_sample, 60.0)
    vol_mod._vol_memo_cache.clear()
    monkeypatch.setattr(vol_mod, "np", None)
    vol_without_np = vol_mod.calculate_volatility_1h(prices_sample, 60.0)
    assert vol_with_np is not None and vol_without_np == vol_with_np

    # 2. GeminiClient.probe_live_model_quota and query_text (REST fallback) respect monkeypatches on gemini_client
    cli_dir = tmp_path / "cli_base"
    cli_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "llm_quota_status.json"
    monkeypatch.setattr(gc_mod, "cache_path", lambda name: str(state_file))
    monkeypatch.setattr(gc_mod, "resolve_bot_cli_base", lambda: str(cli_dir))
    monkeypatch.setattr(gc_mod, "resolve_ui_cli_base", lambda: str(cli_dir))

    def _fake_read_json(path, default=None):
        if str(path).endswith("antigravity-oauth-token"):
            return {"access_token": "tok_test_iter49", "expiry_timestamp": 9999999999.0}
        return default if default is not None else {}

    saved_payloads = []
    monkeypatch.setattr(gc_mod, "read_json_dict", _fake_read_json)
    monkeypatch.setattr(gc_mod, "save_json_state", lambda path, data: saved_payloads.append((path, dict(data))))

    def _fake_fetch_post(url, *args, **kwargs):
        if "retrieveUserQuota" in str(url):
            return {"buckets": [{"modelId": "gemini-3.8-flash-tiered", "remainingFraction": 0.75}]}
        if "loadCodeAssist" in str(url):
            return {"paidTier": {"id": "pro", "availableCredits": [{"creditAmount": 42.0}]}}
        if "generativelanguage.googleapis.com" in str(url):
            return {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        return {}

    monkeypatch.setattr(gc_mod, "fetch_json_post", _fake_fetch_post)
    monkeypatch.setattr(gc_mod.GeminiClient, "_QUOTA_MEM_CACHE", {})
    snap = gc_mod.GeminiClient.probe_live_model_quota(bot_cli_base=str(cli_dir), force_refresh=True)
    assert isinstance(snap, dict)
    assert snap["available_credits"] == 42.0
    assert snap["buckets"].get("gemini-4-argon-medium") == 0.75
    assert saved_payloads and saved_payloads[-1][0] == str(state_file)

    client = gc_mod.GeminiClient(api_key="test-rest-key", cli_path="/nonexistent/agy")
    assert client.query_json("ping") == {"ok": True}

    # 3. reconcile_venue_cycles_with_recent_orders propagates sell_ts_ms across symbol variants (e.g. HYPE -> HYPEUSDC)
    hl_cycles = {"HYPE": {"qty": 0.0, "last_sell_price": 27.5, "_last_sell_ts_ms": 1700001234000}}
    sell_ts_map = reconcile_venue_cycles_with_recent_orders({}, hl_cycles, {}, {}, {}, {})
    assert sell_ts_map.get(("hyperliquid", "HYPE")) == 1700001234000
    assert sell_ts_map.get(("hyperliquid", "HYPEUSDC")) == 1700001234000
    recent_orders: dict = {}
    v_ro_map: dict = {}
    ro_out = _populate_cycle_fallback_recent_order(
        "hyperliquid", "HYPEUSDC", hl_cycles["HYPE"], False, None, recent_orders, v_ro_map, sell_ts_map
    )
    assert isinstance(ro_out, dict)
    assert ro_out.get("last_sell", {}).get("ts_ms") == 1700001234000

    # 4. compute_enterprise_portfolio_risk_metrics falls back to free_quote_usd when free is absent
    risk_metrics = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": [{"venue": "hyperliquid", "symbol": "HYPEUSDC", "base": "HYPE", "value_usd": 200.0}],
            "venue_quote_cash": {
                "hyperliquid:USDC": {"venue": "hyperliquid", "asset": "USDC", "free_quote_usd": 300.0, "locked_quote_usd": 50.0}
            },
        },
        {"HYPEUSDC": {"price": 25.0, "vol_1h_pct": 1.2}},
        {},
    )
    assert risk_metrics.get("global_free_cash_usd") == 300.0
    assert risk_metrics.get("locked_quote_usd") == 50.0
    assert risk_metrics.get("unified_nav_usd") == 550.0

    # 5. enrich_cross_venue_trends accepts tuple rows in watcher_price_items
    pt_map = {"SOLUSDC": {"price": 150.0}}
    watcher_items = {"SOLUSDC": ((1700000000000, 120.0), (1700600000000, 160.0), (1700605000000, 150.0))}
    enrich_cross_venue_trends(pt_map, {}, {}, {}, {}, {}, watcher_items)
    assert pt_map["SOLUSDC"].get("low_7d") == 120.0
    assert pt_map["SOLUSDC"].get("high_7d") == 160.0


def test_iter50_t212_order_status_quantity_imports_cachetrend_monkeypatch_and_collector_forwarding(monkeypatch, tmp_path):
    import pytest
    import cacheManager as cm
    import cachetrend
    import intelligence.macro_sentiment_ssot as ssot_mod
    from intelligence.sentiment.collectors import (
        macro_telemetry_collector as mtc_mod,
        microstructure_collector as msc_mod,
        pretrade_telemetry_collector as ptc_mod,
    )
    from providers.strategy_executor import ProviderError
    from providers.t212_provider import T212Provider

    # 1. T212Provider.order_status rejects empty-string fill fields and falls back from fee=None to commission
    class _FakeT212Client:
        def __init__(self, payload):
            self._payload = payload

        def get_order_status(self, oid):
            return self._payload

    prov = T212Provider(client=_FakeT212Client({
        "status": "FILLED",
        "ticker": "NVDA_US_EQ",
        "filledQuantity": "",
        "filledValue": 100.0,
    }))
    with pytest.raises(ProviderError, match="invalid fill values"):
        prov.order_status("NVDA_US_EQ", "101")

    prov_comm = T212Provider(client=_FakeT212Client({
        "status": "FILLED",
        "ticker": "NVDA_US_EQ",
        "filledQuantity": 2.0,
        "filledValue": 250.0,
        "fee": None,
        "commission": 0.75,
    }))
    st = prov_comm.order_status("NVDA_US_EQ", "102")
    assert st.status == "closed"
    assert st.filled_qty == 2.0
    assert st.cost == 250.0
    assert st.fee == 0.75

    # 2. cachetrend._cm_attr respects monkeypatches on cachetrend even when cacheManager is loaded
    assert hasattr(cm, "CM_LONG_ARCHIVE_MONTHS")
    assert hasattr(cm, "CM_LONG_ARCHIVE_SAMPLE_SEC")
    assert hasattr(cm, "CM_LONG_ARCHIVE_FLUSH_SEC")
    monkeypatch.setattr(cachetrend, "CM_DYNAMIC_WINDOW_MIN_SEC", 42.5)
    assert cachetrend._cm_attr("CM_DYNAMIC_WINDOW_MIN_SEC", 14.0) == 42.5

    # 3. _filter_ohlc_closes_since handles None closes/timestamps and non-numeric values safely
    class _MalformedSeries:
        timestamps = [None, "bad", 1700001000.0, 1700002000.0]
        closes = [10.0, 20.0, "nan", 125.5]

    class _NoneClosesSeries:
        timestamps = [1700001000.0]
        closes = None

    assert msc_mod._filter_ohlc_closes_since(_MalformedSeries(), 1700000000.0) == [125.5]
    assert msc_mod._filter_ohlc_closes_since(_NoneClosesSeries(), 1700000000.0) == []

    # 4. Pretrade/macro/microstructure collectors forward monkeypatches from intelligence.macro_sentiment_ssot
    seen_paths = []

    def _custom_read_json(path):
        seen_paths.append(str(path))
        if str(path).endswith("orderbook_depth_BTCUSDC.json"):
            return {"symbol": "BTCUSDC", "bid_depth_usd": 50000.0, "ask_depth_usd": 40000.0, "mid_price": 65000.0}
        return None

    monkeypatch.setattr(ssot_mod, "read_json_cached", _custom_read_json)
    assert ptc_mod._resolve_read_json_cached() is _custom_read_json
    assert msc_mod._resolve_read_json_cached() is _custom_read_json
    m_loaded = msc_mod.load_symbol_microstructure(str(tmp_path), "BTCUSDC")
    assert isinstance(m_loaded, dict) and m_loaded.get("ob_mid_price") == 65000.0

    saved_macro = []
    monkeypatch.setattr(
        ssot_mod,
        "fetch_public_json",
        lambda url, **kw: {
            "chart": {
                "result": [{
                    "meta": {"regularMarketPrice": 500.0, "regularMarketChangePercent": 1.25},
                    "timestamp": [1700000000, 1700003600],
                    "indicators": {"quote": [{"close": [495.0, 500.0]}]},
                }]
            }
        } if "finance.yahoo.com" in str(url) else {},
    )
    monkeypatch.setattr(ssot_mod, "save_json_state", lambda path, data, **kw: saved_macro.append((path, data)))
    snap_macro = mtc_mod.collect_macro_benchmarks_snapshot(str(tmp_path), now_ts=1700003600.0, persist=True)
    assert isinstance(snap_macro, dict) and snap_macro["items"]["SPY"]["price"] == 500.0
    assert len(saved_macro) == 1


def test_iter51_hl_client_strict_empty_str_portfolio_ssot_forwarding_and_daemon_ssot_monkeypatch(monkeypatch, tmp_path):
    import importlib.util
    import os
    import intelligence.macro.daemon_ssot as daemon_ssot
    import intelligence.macro_sentiment_ssot as ssot_mod
    import intelligence.sentiment.collectors.portfolio_collector as pc_mod

    root_dir = os.path.dirname(os.path.dirname(__file__))
    hl_spec = importlib.util.spec_from_file_location(
        "hl_client_iter51",
        os.path.join(root_dir, "hyperliquid", "hl_client.py"),
    )
    hl_mod = importlib.util.module_from_spec(hl_spec)
    hl_spec.loader.exec_module(hl_mod)
    HLClient = hl_mod.HLClient
    HLError = hl_mod.HLError

    # 1. HLClient.position_strict and spot_balance_strict reject empty-string numeric fields
    #    and support spot quote suffixes (/USDC, USDC) across max_leverage, coin_listed, etc.
    class _FakeHLInfo:
        coin_to_asset = {"HYPE": 10001}

        def meta(self):
            return {"universe": [{"name": "HYPE", "szDecimals": 2, "maxLeverage": 5}]}

        def user_state(self, address):
            return {"assetPositions": [{"position": {"coin": "HYPE", "szi": self._szi, "entryPx": self._entry}}]}

        def spot_user_state(self, address):
            return {"balances": [{"coin": "HYPE", "total": self._spot_tot}]}

        def open_orders(self, address):
            return [{"coin": "HYPE", "oid": 777}]

        def meta_and_asset_ctxs(self):
            return {"universe": [{"name": "HYPE"}]}, [{"funding": "0.00012"}]

    info = _FakeHLInfo()
    info._szi = ""
    info._entry = "25.0"
    info._spot_tot = ""

    hl = HLClient.__new__(HLClient)
    hl.info = info
    hl.address = "0x1111111111111111111111111111111111111111"
    hl._meta_cache = None

    with pytest.raises(HLError, match="Invalid numeric position state"):
        hl.position_strict("HYPE")
    with pytest.raises(HLError, match="Invalid numeric spot balance"):
        hl.spot_balance_strict("HYPE")

    info._szi = None
    info._entry = None
    info._spot_tot = None
    assert hl.position_strict("HYPE/USDC") == (0.0, 0.0)
    assert hl.spot_balance_strict("HYPE/USDC") == 0.0
    assert hl.max_leverage("HYPE/USDC") == 5
    assert hl.coin_listed("HYPE/USDC") is True
    assert len(hl.open_orders_strict("HYPE/USDC")) == 1
    assert hl.funding_rate("HYPE/USDC") == pytest.approx(0.00012)

    # 2. portfolio_collector forwards monkeypatches on intelligence.macro_sentiment_ssot
    monkeypatch.setattr(
        ssot_mod,
        "read_json_cached",
        lambda p: {"BTCUSDC": {"peak": 71234.5}} if str(p).endswith("trailing_state.json") and "kraken" not in str(p) else None,
    )
    t_map, _ = pc_mod.load_venue_trailing_maps(str(tmp_path), str(tmp_path))
    assert t_map.get("BTCUSDC", {}).get("peak") == 71234.5

    # 3. daemon_ssot honours monkeypatches on intelligence.macro_sentiment_ssot
    monkeypatch.setattr(
        ssot_mod,
        "read_json_dict_recovering",
        lambda p: {"pid": os.getpid(), "ts": 9999999999.0, "intervals": {"fast": 10}},
    )
    hb = daemon_ssot.inspect_daemon_heartbeat(str(tmp_path), ("hb.json",))
    assert hb["running"] is True
    assert hb["status"] == "alive"

    saved_hb = []
    monkeypatch.setattr(ssot_mod, "save_json_state", lambda p, d, **kw: saved_hb.append((p, d)))
    daemon_ssot.write_daemon_heartbeat_files([str(tmp_path / "hb.json")], {"ok": True}, use_save_json_state=True)
    assert len(saved_hb) == 1 and saved_hb[0][1] == {"ok": True}

    # 4. 212trading/market_data.py parse_interval_minutes rejects non-ASCII or signed prefixes
    spec = importlib.util.spec_from_file_location(
        "t212_market_data_iter51",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "212trading", "market_data.py"),
    )
    md_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md_mod)
    assert md_mod.parse_interval_minutes("5m") == 5
    for bad_iv in ("+5m", "5 m", "٥m"):
        with pytest.raises(ValueError, match="invalid Yahoo interval"):
            md_mod.parse_interval_minutes(bad_iv)


def test_iter52_repair_ssot_monkeypatch_portfolio_scope_and_trailing_pct_lookup(monkeypatch, tmp_path):
    from binance_api import trailing_stop as bin_ts
    from kraken import trailing_stop as krk_ts
    from orchestratorOS.lib import technical_repair_ssot as tr_ssot
    from orchestratorOS.supervisor import autonomous_ai_reconciler as ar_mod
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from providers.execution_audit import (
        extract_intent_limit_price,
        extract_intent_notional,
        extract_intent_side,
        validate_and_build_intent_dict,
    )
    from trailing_core import TrailingAdapterBase, TrailingCore

    # 1. Monkeypatching autonomous_ai_reconciler.read_json_recovering / save_json_state
    #    is honoured by technical_repair_ssot helpers even when called without explicit fn
    monkeypatch.setattr(
        ar_mod,
        "read_json_recovering",
        lambda p: {"status": "alive", "ts": 1700000000.0} if str(p).endswith("rtrade.heartbeat") else None,
    )
    hbs, _ = tr_ssot.harvest_heartbeats_and_state_files(str(tmp_path), str(tmp_path), 1700009999.0)
    assert "rtrade.heartbeat" in hbs and hbs["rtrade.heartbeat"]["status"] == "alive"

    saved_calls = []
    monkeypatch.setattr(ar_mod, "save_json_state", lambda p, d, **kw: saved_calls.append((p, d)))
    tr_ssot.persist_supervisor_audit_artifact(
        {"ts": 1700000000.0, "mode": "shadow", "summary": "ok"},
        str(tmp_path),
        str(tmp_path / "eval.json"),
        str(tmp_path / "eval.heartbeat"),
    )
    assert len(saved_calls) == 2 and saved_calls[0][1]["summary"] == "ok"

    # 2. _format_portfolio_section normalizes asset_scope case-insensitively and defaults to "all"
    ps_sample = {
        "cross_market_context": {
            "crypto_bellwethers": {"BTC": 80000.0},
            "equity_bellwethers": {"NVDA": 140.0},
        }
    }
    eq_lines = "\n".join(_format_portfolio_section(ps_sample, " EQUITIES "))
    assert "Cross-Market Live Crypto Telemetry" in eq_lines
    assert "Cross-Market US Equities Telemetry" not in eq_lines
    default_lines = "\n".join(_format_portfolio_section(ps_sample))
    assert "Cross-Market US Equities Telemetry" in default_lines

    # 3. TrailingCore._asset_for_pair is case-insensitive and trail_pct_for validates positive floats
    class _DummyAdapter:
        def assets(self):
            return [("HYPE", "HYPE", "HYPEUSD", 2.0)]

    core = TrailingCore.__new__(TrailingCore)
    core.a = _DummyAdapter()
    assert core._asset_for_pair("hypeusd") == "HYPE"
    assert core._asset_for_pair("  HypeUsd ") == "HYPE"
    assert TrailingAdapterBase._lookup_symbol_entry({"BTCUSDC": 1.5}, "   ") is None
    assert TrailingAdapterBase._lookup_symbol_entry({"BTCUSDC": 1.5}, 123) is None

    bts = bin_ts.TrailingStop.__new__(bin_ts.TrailingStop)
    monkeypatch.setattr(bin_ts, "TRAIL_PCT", {"BTCUSDC": 1.75, "BAD": -1.0})
    assert bts.trail_pct_for("btcusdc") == pytest.approx(1.75)
    for bad_sym in ("", "   ", "BAD", True):
        with pytest.raises(KeyError):
            bts.trail_pct_for(bad_sym)

    kts = krk_ts.KrakenTrailing.__new__(krk_ts.KrakenTrailing)
    monkeypatch.setattr(krk_ts, "TRAIL_PCT", {"HYPE": 2.25, "BAD": 0.0})
    monkeypatch.setattr(krk_ts, "DEFAULT_TRAIL_PCT", 3.5)
    assert kts.trail_pct_for("hype") == pytest.approx(2.25)
    assert kts.trail_pct_for("BAD") == pytest.approx(3.5)

    # 4. Intent field extractors fall through empty/non-positive primary keys to valid aliases,
    #    and validate_and_build_intent_dict clamps explicit <= 0 notional to min_notional_usd
    assert extract_intent_side({"decision": "", "side": "buy"}) == "BUY"
    assert extract_intent_notional({"suggested_notional_usd": "", "notional_usd": "$350"}) == pytest.approx(350.0)
    assert extract_intent_limit_price({"order_type": "LIMIT", "limit_price": "", "price": "79,500"}) == pytest.approx(79500.0)

    rec_zero = validate_and_build_intent_dict(
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.90, "suggested_notional_usd": 0},
        source="test",
        min_confidence=0.85,
        min_notional_usd=10.0,
        max_notional_usd=2000.0,
        existing_intents=[],
        now_ts=1700000000.0,
    )
    assert rec_zero is not None and rec_zero["suggested_notional_usd"] == pytest.approx(10.0)


def test_iter53_runtime_states_callable_arg_collector_monkeypatch_and_evaluator_guards(monkeypatch, tmp_path):
    import json
    import time as _time
    import intelligence.macro_sentiment_ssot as ssot
    import intelligence.sentiment.collectors.macro_telemetry_collector as mtc
    import intelligence.sentiment.collectors.microstructure_collector as mc
    import intelligence.sentiment.collectors.portfolio_collector as pc
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        apply_asset_scope_filter,
        build_cycle_fallback_prices,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    from state_io import read_json_recovering

    # 1. harvest_bot_runtime_states supports 2nd-positional callable read_json_fn even when order_retry_queue.jsonl exists
    now = _time.time()
    retry_q = tmp_path / "order_retry_queue.jsonl"
    retry_q.write_text(
        json.dumps(
            {
                "provider_name": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "lifecycle": "submit_pending",
                "expires_at_ts": now + 300.0,
                "intent_id": "term_1",
            }
        )
        + "\n"
        + json.dumps(
            {
                "provider_name": "binance",
                "symbol": "ETHUSDC",
                "side": "BUY",
                "qty": 0.1,
                "lifecycle": "submit_pending",
                "expires_at_ts": now + 300.0,
                "intent_id": "active_2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _mock_reader(path: str):
        if path.endswith("autonomous_trade_intents_state.json"):
            return {"intents": [{"intent_id": "term_1", "status": "EXECUTED"}]}
        if path.endswith("smart_stop_loss_state.json"):
            return {"BTCUSDC": {"armed": True}}
        return {}

    rt_out = pc.harvest_bot_runtime_states(str(tmp_path), _mock_reader)
    assert "active_retry_queue" in rt_out
    assert [r["symbol"] for r in rt_out["active_retry_queue"]] == ["ETHUSDC"]
    assert rt_out.get("smart_stop_loss_state") == {"BTCUSDC": {"armed": True}}

    # 2. _resolve_read_json_cached honors monkeypatched read_json_cached even when default read_json_recovering is passed
    def _patched_cached(path: str, *a, **kw):
        if path.endswith("fear_greed_collect.json"):
            return {"value": 28, "classification": "Fear", "ts": now}
        return {"patched": True}

    monkeypatch.setattr(ssot, "read_json_cached", _patched_cached)
    assert pc._resolve_read_json_cached(read_json_recovering) is _patched_cached
    assert mc._resolve_read_json_cached(read_json_recovering) is _patched_cached
    sensors, _ = mtc.harvest_macro_and_news_sensors(str(tmp_path), now, read_json_fn=read_json_recovering)
    assert sensors.get("fear_greed", {}).get("value") == 28

    # 3. build_cross_market_context guards against None inputs and non-finite/bool slope values
    assert build_cross_market_context(None, None, None) == {}
    cmc = build_cross_market_context(
        {
            "BTCUSDC": {
                "price": 85000.0,
                "slope_15m": True,
                "slope_2_5h": float("nan"),
                "slope_3_7m": 0.1234567,
            },
            "NVDA_US_EQ": {
                "price": 125.0,
                "slope_15m": False,
                "kalman_vel": float("inf"),
                "slope_3_7m": -0.0543219,
            },
        },
        None,
        None,
    )
    btc_bw = cmc["crypto_bellwethers"]["BTC"]
    assert "slope_15m" not in btc_bw
    assert "slope_2_5h" not in btc_bw
    assert btc_bw["slope_3_7m"] == pytest.approx(0.123457)
    nvda_bw = cmc["equity_bellwethers"]["NVDA_US_EQ"]
    assert "slope_15m" not in nvda_bw
    assert "kalman_vel" not in nvda_bw
    assert nvda_bw["slope_3_7m"] == pytest.approx(-0.054322)

    # 4. apply_asset_scope_filter normalizes scope case/whitespace and guards None dicts; build_cycle_fallback_prices guards None
    pt_filtered, ms_filtered = apply_asset_scope_filter(
        {"BTCUSDC": {"price": 85000.0}, "NVDA_US_EQ": {"price": 125.0}},
        {"BTCUSDC": {"ob_imbalance": 0.2}},
        {"total_value_usdc": 1000.0},
        " EQUITIES ",
        None,
        None,
    )
    assert list(pt_filtered.keys()) == ["NVDA_US_EQ"]
    assert ms_filtered == {}
    assert build_cycle_fallback_prices(None, None) == {}


def test_iter54_breadth_vol_share_intent_timestamp_iso_ms_and_evaluator_sanitization():
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        compute_cross_venue_arb_metrics,
        enrich_cross_venue_trends,
        sync_sibling_symbol_telemetry,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        extract_symbol_recent_fills,
        parse_venue_cycle_state,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        apply_live_24trend_windows,
        apply_short_trend_item,
        compute_7d_range_metrics,
    )
    from providers.execution_audit import extract_intent_notional, summarize_recent_ai_intents

    # 1. MarketBreadthCollector.parse_tickers caps BTC/ETH volume share <= 100% even when sub-threshold BTC/ETH pairs exist
    tickers = [
        {"symbol": "BTCUSDC", "lastPrice": "85000", "priceChangePercent": "2.0", "quoteVolume": "10000000"},
        {"symbol": "BTCUSDT", "lastPrice": "85000", "priceChangePercent": "2.0", "quoteVolume": "4500000"},  # < 5M min_vol
        {"symbol": "ETHUSDC", "lastPrice": "2500", "priceChangePercent": "1.5", "quoteVolume": "6000000"},
        {"symbol": "ETHUSDT", "lastPrice": "2500", "priceChangePercent": "1.5", "quoteVolume": "4800000"},  # < 5M min_vol
        {"symbol": "SOLUSDC", "lastPrice": "180", "priceChangePercent": "3.0", "quoteVolume": "5000000"},
        {"symbol": "XRPUSDC", "lastPrice": "2.2", "priceChangePercent": "1.0", "quoteVolume": "5000000"},
        {"symbol": "ADAUSDC", "lastPrice": "0.8", "priceChangePercent": "-0.5", "quoteVolume": "5000000"},
    ]
    mb = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=5_000_000.0, now_ts=1700000000.0)
    assert mb is not None
    assert 0.0 <= mb.btc_vol_share_pct <= 100.0
    assert 0.0 <= mb.eth_vol_share_pct <= 100.0
    assert mb.btc_vol_share_pct + mb.eth_vol_share_pct <= 100.0

    # 2. summarize_recent_ai_intents and extract_symbol_recent_fills handle empty timestamp, ISO-8601 strings, and ms epochs
    now_s = 1700003600.0
    intents = [
        {
            "intent_id": "iso_int_1",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "decision": "BUY",
            "order_type": "MARKET",
            "status": "EXECUTED",
            "timestamp": "",
            "executed_at": "2023-11-14T22:13:20Z",  # 1700000000.0
            "suggested_notional_usd": 200.0,
            "execution_payload": {"status": "FILLED", "executedQty": "0.0025", "price": "80000"},
        },
        {
            "intent_id": "ms_int_2",
            "venue": "binance",
            "symbol": "ETHUSDC",
            "decision": "SELL",
            "order_type": "MARKET",
            "status": "EXECUTED",
            "timestamp": 1700001800000,  # ms epoch -> 1700001800.0 (age 0.5h)
            "suggested_notional_usd": 250.0,
            "execution_payload": {"status": "FILLED", "executedQty": "0.1", "price": "2500"},
        },
    ]
    summaries = summarize_recent_ai_intents(intents, "crypto", now_s, extract_intent_notional)
    by_sym = {s["symbol"]: s for s in summaries}
    assert by_sym["BTCUSDC"]["age_h"] == pytest.approx(1.0)
    assert by_sym["ETHUSDC"]["age_h"] == pytest.approx(0.5)

    fills_iso = extract_symbol_recent_fills([], ai_intents=[intents[0]], symbol="BTCUSDC")
    assert fills_iso is not None
    assert fills_iso["last_buy"]["ts_ms"] == 1700000000000

    # 3. parse_venue_cycle_state guards None and preserves avg_cost_px when re-parsing
    assert parse_venue_cycle_state(None, "BTCUSDC", {})["qty"] == 0.0
    reparsed = parse_venue_cycle_state({"qty": 2.0, "avg_cost_px": 150.5}, "SOLUSDC", {"SOLUSDC": {"price": 165.55}})
    assert reparsed["avg_cost_px"] == pytest.approx(150.5)
    assert reparsed["dd_pct"] == pytest.approx(10.0)

    # 4. compute_7d_range_metrics sanitizes [ts, px] pairs, None, bool, and non-positive values; evaluators guard None
    r7 = compute_7d_range_metrics([[1000, 100.0], None, False, -5.0, [2000, 120.0], 110.0])
    assert r7["low_7d"] == pytest.approx(100.0)
    assert r7["high_7d"] == pytest.approx(120.0)
    assert r7["chg_7d_pct"] == pytest.approx(10.0)
    assert apply_short_trend_item(None, None) is False
    apply_live_24trend_windows(None, None)
    enrich_cross_venue_trends({"BTCUSDC": None}, None, None, None, None, None, None)
    sync_sibling_symbol_telemetry({"BTCUSDC": {"price": 85000.0}, "BTCUSD": {"price": 85010.0}}, {"BTCUSDC": None, "BTCUSD": {"ob_imbalance": 0.6}})
    m_arb = {"ob_mid_price": 85000.0}
    compute_cross_venue_arb_metrics("BTCUSDC", m_arb, None, {"BTCUSDC": m_arb, "BTCUSD": {}}, {"btcusd": {"price": 84900.0}})
    assert m_arb.get("cross_venue_peer") == "BTCUSD"


def test_iter55_geopolitical_fallback_lexicon_dir_guard_cache_notional_and_news_history_dict(tmp_path, monkeypatch):
    import json
    import intelligence.macro.geopolitical_analyzer as geo_mod
    import intelligence.sentiment.sentiment_advisor as sa_mod
    import intelligence.sentiment.guards.high_stake_guard as hsg_mod
    from intelligence.macro.lexicon_ssot import update_dynamic_lexicon
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsFeedSnapshot, NewsHeadline
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard

    # 1. GeopoliticalThreatAnalyzer.assess passes state_cache_dir and now_ts to has_acute_shock_keywords on LLM fallback
    now_ts = 1710000000.0
    update_dynamic_lexicon(
        [],
        new_acute_terms=["quasarcollapse"],
        cache_dir=str(tmp_path),
        now_ts=now_ts - 60.0,
    )
    monkeypatch.setattr(geo_mod, "query_topic_json", lambda *a, **kw: None, raising=False)
    analyzer = geo_mod.GeopoliticalThreatAnalyzer(state_file=str(tmp_path / "geo_state.json"))
    snap = NewsFeedSnapshot(
        headlines=(
            NewsHeadline(
                title="Breaking: sudden quasarcollapse rattles global liquidity",
                source="reuters",
                pub_date="Wed, 13 Mar 2024 12:00:00 GMT",
                url="https://example.com/quasar",
                matched_keywords=("quasarcollapse",),
                is_high_severity=True,
            ),
        ),
        total_fetched=1,
        high_severity_count=1,
        has_critical_shock_keywords=True,
        ts=now_ts - 30.0,
    )
    assessment = analyzer.assess(snap, now_ts=now_ts)
    assert assessment.threat_level == "CRITICAL_SHOCK"
    assert assessment.risk_score == pytest.approx(0.9)
    assert assessment.is_fallback is True

    # 2. SentimentAdvisor.review honors monkeypatched query_topic_json & record_dynamic_crisis_terms_from_assessment,
    # and formats BTC Intraday Structure cleanly when series < 18h (chg_24h_pct absent) and trailing row is malformed
    captured_prompts = []
    recorded_lexicon = []

    def fake_sa_query(client, topic, prompt, **kwargs):
        captured_prompts.append(prompt)
        return {
            "market_bias": "BEARISH",
            "risk_level": "HIGH",
            "confidence": 0.85,
            "summary": "Elevated macro volatility after liquidity sweep",
            "key_risks": ["funding stress"],
            "recommended_action": "DEFENSIVE",
        }

    monkeypatch.setattr(sa_mod, "query_topic_json", fake_sa_query, raising=False)
    monkeypatch.setattr(
        sa_mod,
        "record_dynamic_crisis_terms_from_assessment",
        lambda *a, **kw: recorded_lexicon.append((a, kw)) or (),
        raising=False,
    )
    price_file = tmp_path / "cache_24_price_BTCUSDC.json"
    price_file.write_text(
        json.dumps(
            [
                [now_ts - 3600, 84000.0],
                [now_ts - 1800, 85000.0],
                ["bad_ts", None],
            ]
        ),
        encoding="utf-8",
    )
    advisor = sa_mod.SentimentAdvisor(cache_file=str(tmp_path / "sent_state.json"))
    view = advisor.review(now_ts=now_ts)
    assert view is not None
    assert view.risk_level == "HIGH"
    assert len(captured_prompts) == 1
    assert "- BTC Intraday Structure: price=$85000.00" in captured_prompts[0]
    assert len(recorded_lexicon) == 1

    # 3. HighStakeGuard & PostStopReentryGuard update metadata["notional_eur"] on enforce-mode cache hit
    hsg_mod._HIGH_STAKE_DECISION_CACHE.clear()
    from intelligence.sentiment.guards.post_stop_reentry_guard import clear_post_stop_reentry_cache
    clear_post_stop_reentry_cache()
    llm_calls = []

    def fake_guard_query(client, topic, prompt, **kwargs):
        llm_calls.append(prompt)
        return {"decision": "APPROVED", "confidence": 0.9, "reason": "ok", "suggested_scale": 1.0}

    monkeypatch.setattr(hsg_mod, "query_topic_json", fake_guard_query, raising=False)
    hsg = hsg_mod.HighStakeGuard(
        gemini_client=object(),
        min_notional_eur=500.0,
        mode="enforce",
        notify_enabled=False,
        cache_ttl_sec=300,
    )
    d1 = hsg.check("BTCUSDC", "BUY", 85000.0, 0.01, notional_eur=510.0, telemetry={}, now=now_ts)
    d2 = hsg.check("BTCUSDC", "BUY", 85000.0, 0.01, notional_eur=545.0, telemetry={}, now=now_ts)  # same $100 bucket (510//100 == 545//100 == 5)
    assert len(llm_calls) == 1
    assert d1.metadata["notional_eur"] == pytest.approx(510.0)
    assert d2.metadata["notional_eur"] == pytest.approx(545.0)

    psg = PostStopReentryGuard(
        gemini_client=object(),
        mode="enforce",
        notify_enabled=False,
        cache_ttl_sec=300,
    )
    p1 = psg.check("ETHUSDC", 2420.0, 2400.0, 2350.0, 0.02, notional_eur=300.0, last_sell_time_s=now_ts - 1800.0, telemetry={}, now=now_ts)
    p2 = psg.check("ETHUSDC", 2420.0, 2400.0, 2350.0, 0.02, notional_eur=750.0, last_sell_time_s=now_ts - 1800.0, telemetry={}, now=now_ts)
    assert len(llm_calls) == 2
    assert p1.metadata["notional_eur"] == pytest.approx(300.0)
    assert p2.metadata["notional_eur"] == pytest.approx(750.0)

    # 4. NewsFeedCollector._load_history unwraps dict-shaped {"items": [...]} payloads
    import time as _time
    recent_ts = _time.time() - 60.0
    hist_file = tmp_path / "news_feed_collect.json"
    hist_file.write_text(
        json.dumps(
            {
                "updated_at": recent_ts,
                "items": [
                    {
                        "title": "ECB holds rates steady",
                        "source": "ecb",
                        "pub_date": "",
                        "url": "https://example.com/ecb",
                        "matched_keywords": [],
                        "is_high_severity": False,
                        "first_seen_ts": recent_ts,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    collector = NewsFeedCollector(history_file=str(hist_file))
    assert len(collector._history) == 1
    loaded_h = next(iter(collector._history.values()))[1]
    assert loaded_h.title == "ECB holds rates steady"


def test_iter56_pretrade_null_safety_hl_list_wrapping_and_external_guard_aliases(tmp_path, monkeypatch):
    import json
    import intelligence.external.collectors.derivatives_telemetry_collector as dt_mod
    import intelligence.external.collectors.whale_positioning_collector as wp_mod
    import intelligence.external.guards.smart_stop_loss_guard as ssl_mod
    import intelligence.sentiment.collectors.pretrade_telemetry_collector as pt_mod
    from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector

    # 1. collect_pretrade_telemetry handles None values in m24/btc_m24/liq_c and forwards monkeypatched read_json_cached
    now_ts = 1710000000.0
    virtual_files = {
        str(tmp_path / "cache_current_price.json"): {"items": {"ETHUSDC": 2500.0, "BTCUSDC": 85000.0}},
        str(tmp_path / "orderbook_depth_ETHUSDC.json"): {
            "symbol": "ETHUSDC",
            "imbalance_ratio": 0.65,
            "spread_bps": 1.2,
            "depth_1pct_bid_usd": 400000.0,
            "depth_1pct_ask_usd": 200000.0,
            "mid_price": 2500.0,
        },
        str(tmp_path / "cache_order.json"): {
            "items": {
                "ETHUSDC": [
                    {
                        "orderId": "o_eth_1",
                        "symbol": "ETHUSDC",
                        "side": "BUY",
                        "status": "FILLED",
                        "price": 2480.0,
                        "executedQty": 0.5,
                        "time": int((now_ts - 600) * 1000),
                    }
                ]
            }
        },
    }
    monkeypatch.setattr(pt_mod, "read_json_cached", lambda path, **kw: virtual_files.get(str(path)), raising=False)
    monkeypatch.setattr(
        pt_mod,
        "load_symbol_price_series",
        lambda cdir, sym, **kw: ([[now_ts * 1000 - 60000, 2490.0], [now_ts * 1000, 2500.0]], []),
    )
    monkeypatch.setattr(
        pt_mod,
        "compute_24h_price_metrics",
        lambda ser: {
            "chg_15m_pct": None,
            "chg_30m_pct": 0.5,
            "chg_1h_pct": None,
            "chg_3h_pct": 1.2,
            "low_24h": 2400.0,
            "high_24h": 2550.0,
            "pos_24h_pct": None,
            "poc_px": 2495.0,
            "poc_dist_pct": None,
            "val_px": 2450.0,
            "val_dist_pct": None,
            "vah_px": 2520.0,
            "vah_dist_pct": None,
        },
    )
    monkeypatch.setattr(
        pt_mod,
        "compute_liquidation_clusters",
        lambda p_est, m_info: {
            "long_liq_px": 2350.0,
            "short_liq_px": 2650.0,
            "long_liq_usd": None,
            "long_liq_dist_pct": None,
            "short_liq_usd": 500000.0,
            "short_liq_dist_pct": 6.0,
            "liq_magnet_bias": "SHORT",
        },
    )
    telem = pt_mod.collect_pretrade_telemetry("ETHUSDC", cachedb_dir=str(tmp_path))
    assert "24h Multi-Horizon Price Evolution" in telem
    assert "24h Volume Profile (POC / Value Area)" in telem
    assert "Orderbook Bid-Ask Spread" in telem
    assert "Liquidation Clusters (LONG vs SHORT)" in telem
    assert "Recent Venue Fills" in telem

    # 2. _DiskBackedSymbolCollector._fetch_json_dict wraps list responses for Hyperliquid info URLs when fetch_public_json is monkeypatched
    monkeypatch.setattr(
        dt_mod,
        "fetch_public_json",
        lambda url, **kw: (
            [{"universe": [{"name": "HYPE"}]}, [{"funding": "0.0002", "openInterest": "1000", "markPx": "25.0", "oraclePx": "25.0"}]]
            if "metaAndAssetCtxs" in url
            else {}
        ),
    )
    dt_col = dt_mod.DerivativesTelemetryCollector(cache_dir=str(tmp_path))
    hl_tel = dt_col._fetch_hyperliquid_derivatives("HYPEUSDC", now=now_ts)
    assert hl_tel is not None
    assert hl_tel.open_interest_usd == pytest.approx(25000.0)

    # WhalePositioningCollector._resolve_cached_funding_rate honors monkeypatched read_json_dict
    monkeypatch.setattr(
        wp_mod,
        "read_json_dict",
        lambda path, **kw: {"ts": now_ts - 10.0, "funding_rate": -0.0003},
    )
    wp_col = wp_mod.WhalePositioningCollector(cache_dir=str(tmp_path))
    assert wp_col._resolve_cached_funding_rate("BTCUSDC", now=now_ts) == pytest.approx(-0.0003)

    # 3. BybitLiquidationCollector sanitizes malformed rows and supports explicit `now` in get_signal / capitulation_ratio
    bybit_col = BybitLiquidationCollector()
    bybit_col._events["BTCUSDT"] = [  # type: ignore[assignment]
        (now_ts - 30.0, "LONG", 120000.0),
        (now_ts - 20.0, "INVALID_SIDE", 999999.0),
        (now_ts - 10.0, "SHORT", None),  # type: ignore[list-item]
        (now_ts - 600.0, "LONG", 60000.0),
    ]
    sig = bybit_col.get_signal("BTCUSDT", window_sec=120.0, now=now_ts)
    assert sig["long_liq_usd"] == pytest.approx(120000.0)
    assert sig["short_liq_usd"] == pytest.approx(0.0)
    cap_r = bybit_col.capitulation_ratio("BTCUSDT", window_sec=120.0, baseline_sec=3600.0, now=now_ts)
    assert cap_r == pytest.approx((120000.0 / 120.0) / (180000.0 / 3600.0))

    # 4. FearGreedCollector.parse_payload honors now_ts when timestamp is absent
    fg_snap = FearGreedCollector.parse_payload({"data": [{"value": "20", "value_classification": "Extreme Fear"}]}, now_ts=now_ts)
    assert fg_snap is not None
    assert fg_snap.timestamp == pytest.approx(now_ts)

    # 5. SmartStopLossGuard._load_grace_state honors now_ts and microstructure alias keys; WhaleAccumulationTrigger honors alias keys
    ssl_file = tmp_path / "smart_sl_iter56.json"
    ssl_file.write_text(
        json.dumps({"SOLUSDC": {"start_ts": now_ts - 40.0, "initial_loss_pct": 3.2, "lost_threshold_pct": 3.0, "provider": "binance"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ssl_mod, "_SMART_SL_STATE_FILE", str(ssl_file))
    ssl_mod._GRACE_TRACKER.clear()
    ssl_mod._LOADED_STATE_FILE_KEY = None
    ssl_guard = ssl_mod.SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    should_exec, reason, meta = ssl_guard.evaluate(
        "SOLUSDC",
        0.031,
        0.030,
        now=now_ts,
        provider="binance",
        whale_snapshot={"whale_long_pct": 65.0, "taker_buy_sell_ratio": 0.95},
        orderbook_snapshot={"ob_imbalance": 0.75, "bid_wall_usd": 600000.0},
    )
    assert should_exec is False
    assert "grace_active" in reason
    assert meta["grace_elapsed_sec"] == pytest.approx(40.0)

    wat = WhaleAccumulationTrigger()
    ev = wat.evaluate(
        "SOLUSDC",
        150.0,
        snapshot={"whale_long_pct": 68.0, "taker_buy_sell_ratio": 1.35, "oi_1h_chg_pct": 1.5, "divergence_regime": "accumulation"},
        now=now_ts,
    )
    assert ev is not None
    assert ev.reason == "whale_aggressive_accumulation"


def test_iter57_hierarchical_shadow_mode_collector_guards_and_vol_memo_collision(tmp_path, monkeypatch):
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.external.collectors.bybit_liquidations_collector import LiquidationSummary
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetry
    from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.internal.guards.guard_decision import BrakeAction, resolve_hierarchical_guard_mode
    from intelligence.internal.state import survival as surv_mod
    from intelligence.internal.state import volatility as vol_mod
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

    # 1. resolve_hierarchical_guard_mode cascades "shadow" from intelligence_guards_mode
    assert (
        resolve_hierarchical_guard_mode(
            {"intelligence_guards_mode": "shadow", "pillar2_external_guards_mode": "enforce", "whale_divergence_guard_mode": "enforce"},
            "whale_divergence_guard_mode",
            "pillar2_external_guards_mode",
        )
        == "shadow"
    )

    # 2. CompositeMarketIntelligence invokes collector-backed cascade_guard, funding_guard, and liquidation_trigger when explicit snapshots are None
    liq_snap = LiquidationSummary(
        symbol="BTCUSDC",
        window_sec=60.0,
        long_liq_usd=5_000_000.0,
        short_liq_usd=100_000.0,
        net_usd=-4_900_000.0,
        capitulation_ratio=5_000_000.0 / 5_100_000.0,
        squeeze_ratio=100_000.0 / 5_100_000.0,
        events_count=50,
        last_event_ts=1700000000.0,
    )
    deriv_snap = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.0025,
        predicted_funding_rate=0.0025,
        open_interest=1000.0,
        open_interest_usd=95_000_000.0,
        ts=1700000000.0,
    )

    class _DummyLiqCol:
        def get_summary(self, symbol, **kwargs):
            return liq_snap

    class _DummyDerivCol:
        def fetch(self, symbol, **kwargs):
            return deriv_snap

    comp = CompositeMarketIntelligence(max_active_cascade_usd=1_000_000.0, max_long_funding_rate=0.0005)
    comp.cascade_guard.collector = _DummyLiqCol()
    comp.funding_guard.collector = _DummyDerivCol()
    comp.liquidation_trigger.collector = _DummyLiqCol()

    dec_casc = comp.evaluate_guards("BTCUSDC", "BUY", 95000.0, now=1700000000.0)
    assert dec_casc.allowed is False
    assert dec_casc.brake_action == BrakeAction.DEFER_WAIT
    assert dec_casc.guard_name == "LiquidationCascadeGuard"

    dec_fund = comp.evaluate_guards("BTCUSDC", "BUY", 95000.0, bypass_guards=["cascade_guard"], now=1700000000.0)
    assert dec_fund.guard_name == "FundingCrowdingGuard"
    assert dec_fund.suggested_scale < 1.0

    eval_res = comp.evaluate("BTCUSDC", 95000.0, ts=1700000000.0, bypass_guards=["cascade_guard", "funding_guard"])
    assert any(ev.source in ("liquidation_trigger", "LiquidationCapitulationTrigger") for ev in eval_res.triggers)

    # 3. GlobalTelemetryCollector propagates now_ts to update_news, update_fear_greed, and update_market_breadth
    gtc = GlobalTelemetryCollector(cache_dir=str(tmp_path), symbols=["BTCUSDC"], skip_external=True)
    seen_now = {}

    class _MockNews:
        def fetch(self, force_refresh=False, now_ts=None):
            seen_now["news"] = now_ts
            return SimpleNamespace(total_fetched=3)

    class _MockFG:
        def fetch(self, force_refresh=False, now_ts=None):
            seen_now["fg"] = now_ts
            return FearGreedSnapshot(
                value=22,
                classification="Extreme Fear",
                timestamp=0.0,
                historical_values=(22, 25),
                trend_7d_change=-3,
                is_extreme_fear=True,
                is_extreme_greed=False,
            )

    class _MockBreadth:
        def fetch(self, force_refresh=False, now_ts=None):
            seen_now["breadth"] = now_ts
            return MarketBreadthSnapshot(
                total_count=10,
                advancing_count=7,
                declining_count=3,
                advance_ratio=0.7,
                median_change_pct=1.2,
                mean_change_pct=1.5,
                dispersion_std=2.0,
                top_gainers=(("BTCUSDC", 3.0),),
                top_losers=(("ETHUSDC", -1.0),),
                regime="broad_advance",
                ts=0.0,
            )

    gtc.news_collector = _MockNews()
    gtc.fear_greed_collector = _MockFG()
    gtc.breadth_collector = _MockBreadth()
    fixed_now = 1711111111.0
    gtc.run_cycle(force=True, now=fixed_now)
    assert seen_now == {"news": fixed_now, "fg": fixed_now, "breadth": fixed_now}
    fg_saved = json.loads((tmp_path / "fear_greed_collect.json").read_text(encoding="utf-8"))
    mb_saved = json.loads((tmp_path / "market_breadth_collect.json").read_text(encoding="utf-8"))
    assert fg_saved["ts"] == pytest.approx(fixed_now)
    assert fg_saved["index_timestamp"] == pytest.approx(fixed_now)
    assert mb_saved["ts"] == pytest.approx(fixed_now)

    # 4. calculate_volatility_1h does not collide in _vol_memo_cache when interior prices differ
    vol_mod._vol_memo_cache.clear()
    flat_series = [100.0] * 30
    spike_series = [100.0] + [110.0, 90.0] * 14 + [100.0]
    assert len(flat_series) == len(spike_series)
    assert flat_series[0] == spike_series[0] and flat_series[-1] == spike_series[-1]
    v_flat = vol_mod.calculate_volatility_1h(flat_series, 60.0)
    v_spike = vol_mod.calculate_volatility_1h(spike_series, 60.0)
    assert v_flat == 0.0
    assert v_spike is not None and v_spike > 50.0

    # 5. survival respects monkeypatches on survival module and ignores n=0 cached entries for gaussian_horizon_days
    t_cache_file = tmp_path / "cache_T_iter57.json"
    t_cache_file.write_text(
        json.dumps({"NEWCOINUSDC": {"T": 14, "n": 0, "w": 0.0, "median_d": None, "p90_d": None, "ts": fixed_now}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(surv_mod, "T_CACHE_FILE", str(t_cache_file))
    surv_mod._T_CACHE_FILE_MEMO.clear()
    m_zero_n = surv_mod.get_trend_survival_metrics("NEWCOINUSDC", trend_duration_seconds=86400.0)
    assert m_zero_n["gaussian_horizon_days"] is None
    assert m_zero_n["T"] == pytest.approx(8.0)

    monkeypatch.setattr(surv_mod, "estimate_T", lambda sym: {"T": 9.5, "n": 12, "median_d": 3.5, "p90_d": 8.0})
    m_net = surv_mod.get_trend_survival_metrics("UNLISTEDCOIN", trend_duration_seconds=86400.0, allow_network=True)
    assert m_net["T"] == pytest.approx(9.5)
    assert m_net["gaussian_horizon_days"] == pytest.approx(round(9.5 * 1.4, 4))


def test_iter58_5m_series_vol_breadth_greed_percent_scales_and_tca_variants(tmp_path, monkeypatch):
    from intelligence.cli import inspect_symbol_intelligence
    from intelligence.internal.state import survival as surv_mod
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        enrich_sentiment_advisor_live_flow_divergence,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        _infer_symbol_venue_and_fees_bps,
        _resolve_non_static_prev_metric,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _apply_24h_series_enrichment,
        compute_24h_price_metrics,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
    )
    from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
    from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
    from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
    from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger

    # 1. _apply_24h_series_enrichment computes vol_1h_pct on 5-minute OHLC series (12-13 pts in last 1h, >=20 pts total)
    t0_ms = 1_710_000_000_000
    series_5m = [
        [t0_ms + i * 300_000, 100.0 + ((-1) ** i) * (0.4 + (i % 3) * 0.1)]
        for i in range(60)
    ]
    m24 = compute_24h_price_metrics(series_5m)
    entry_5m = {"price": series_5m[-1][1]}
    _apply_24h_series_enrichment("TAOUSD", entry_5m, m24, series_5m, None)
    assert entry_5m.get("vol_1h_pct") is not None and entry_5m["vol_1h_pct"] > 0.0
    assert entry_5m.get("adapt_reentry_pct") is not None and entry_5m["adapt_reentry_pct"] > 0.0

    # 2. MarketBreadthTrigger, ExtremeGreedGuard, PanicWashoutGuard accept percentage thresholds/scales & "sentiment" key
    mbt = MarketBreadthTrigger(washout_advance_threshold=15, blowoff_advance_threshold=88)
    assert mbt.washout_advance_threshold == pytest.approx(0.15)
    assert mbt.blowoff_advance_threshold == pytest.approx(0.88)
    sig_b, ev_b = mbt.evaluate("BTCUSDC", 90000.0, {"advance_ratio": 12.0, "median_change_pct": -5.5})
    assert sig_b == "BREADTH_WASHOUT_BUY"
    assert ev_b is not None

    egg = ExtremeGreedGuard(downscale_threshold=80, hard_veto_threshold=90, downscale_factor=35)
    assert egg.downscale_factor == pytest.approx(0.35)
    dec_egg = egg.check("BTCUSDC", "BUY", {"value": 84, "sentiment": "Extreme Greed"})
    assert dec_egg.suggested_scale == pytest.approx(0.35)
    assert dec_egg.metadata["classification"] == "Extreme Greed"

    sct = SentimentContrarianTrigger()
    sig_s, ev_s = sct.evaluate("BTCUSDC", 90000.0, {"value": 18, "sentiment": "Extreme Fear"})
    assert sig_s == "CONTRARIAN_BUY"
    assert ev_s is not None and ev_s.metadata["fear_greed_classification"] == "Extreme Fear"

    pwg = PanicWashoutGuard(panic_advance_threshold=10, downscale_factor=50)
    assert pwg.panic_advance_threshold == pytest.approx(0.10)
    assert pwg.downscale_factor == pytest.approx(0.50)
    dec_pwg = pwg.check("BTCUSDC", "BUY", breadth_snapshot={"advance_ratio": 8.0, "median_change_pct": -6.0})
    assert dec_pwg.suggested_scale == pytest.approx(0.50)

    # 3. enrich_open_limit_orders_tca and compute_enterprise_portfolio_risk_metrics resolve metrics across symbol variants
    prices_map = {
        "TAOUSD": {"price": 400.0},
        "TAOUSDC": {"price": 400.0, "vol_1h_pct": 2.5, "beta_btc_24h": 1.6, "corr_btc_24h": 0.92},
    }
    open_orders = [{"symbol": "TAOUSD", "side": "BUY", "price": 390.0, "venue": "kraken"}]
    enrich_open_limit_orders_tca(open_orders, prices_map, {})
    assert open_orders[0]["fill_prob_1h_pct"] > 50.0

    risk = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": [{"venue": "kraken", "symbol": "TAOUSD", "value_usd": 1000.0}],
            "venue_quote_cash": {"kraken": {"venue": "kraken", "free": 0.0}},
        },
        prices_map,
        {},
    )
    assert risk["portfolio_btc_beta"] == pytest.approx(1.6)
    assert risk["corr_vs_btc_24h"]["TAO"] == pytest.approx(0.92)

    # 4. enrich_sentiment_advisor_live_flow_divergence skips empty BTCUSDC dict and resolves lowercase/Kraken BTC keys
    sa_dict = {"age_min": 16.0, "recommended_action": "HALT_BUYS", "directional_bias": "BEARISH"}
    enrich_sentiment_advisor_live_flow_divergence(
        sa_dict,
        {"BTCUSDC": {}, "btcusd": {"taker_buy_sell_ratio": 1.45, "cvd_15m_usd": 6_000_000.0}},
    )
    assert "LIVE_BUY_REVERSAL" in sa_dict.get("live_flow_divergence", "")

    # 5. _infer_symbol_venue_and_fees_bps & _resolve_non_static_prev_metric & cli.inspect_symbol_intelligence
    assert _infer_symbol_venue_and_fees_bps("SOLUSDC", {"venue": "hyperliquid"})[0] == "hyperliquid"
    assert _infer_symbol_venue_and_fees_bps("HYPE/USDC", {})[0] == "hyperliquid"
    assert _resolve_non_static_prev_metric(
        [{"ts": None, "cvd_15m_usd": 10.0}, {"ts": 1000.0, "cvd_15m_usd": 50.0}],
        1300.0,
        "cvd_15m_usd",
        100.0,
        100.0,
        static_tol=1.0,
        min_diff=5.0,
    ) == pytest.approx(50.0)

    (tmp_path / "cache_T_trend.json").write_text(
        json.dumps({"BTCUSDC": {"T": 14, "n": 5, "median_d": 2.0, "p90_d": 5.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(surv_mod, "estimate_T", lambda sym: {"T": 19, "n": 33, "median_d": 4.0, "p90_d": 11.0, "T_emp": 20.0})
    rep = inspect_symbol_intelligence("BTCUSDC", cache_dir=str(tmp_path))
    assert rep["pillar1_internal"]["trend_survival"]["T"] == 19
    assert rep["pillar1_internal"]["trend_survival"]["samples"] == 33


def test_iter59_parabolic_ms_timestamps_persistence_pairs_and_percent_scales(tmp_path):
    import numpy as np
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookSnapshot
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.internal.state.persistence import calculate_hurst_exponent, calculate_mann_kendall
    from intelligence.macro.calendar_ssot import (
        compute_dynamic_geopolitical_interval,
        compute_dynamic_sentiment_interval,
    )
    from intelligence.macro.news_feed_collector import NewsFeedSnapshot, NewsHeadline
    from intelligence.macro_sentiment_ssot import update_dynamic_lexicon

    # 1. ParabolicSurgeGuard normalizes millisecond timestamps in price_history and now
    psg = ParabolicSurgeGuard(surge_threshold_pct=5.0, window_seconds=3600.0)
    ms_hist = [
        (1700000000000, 100.0),
        (1700001800000, 103.0),
        (1700003500000, 106.5),
    ]
    dec_psg = psg.check("BTCUSDC", "BUY", 106.5, price_history=ms_hist, now=1700003500.0)
    assert not dec_psg.allowed
    assert "parabolic_surge_active" in dec_psg.reason

    # 2. persistence.py & TrendSignificanceGuard support (ts, price) tuple/list sequences and 2D arrays
    ts_price_pairs = [(1700000000.0 + i * 60.0, 100.0 + i * 0.5) for i in range(70)]
    s_stat, z_score, p_val = calculate_mann_kendall(ts_price_pairs)
    assert s_stat > 0 and z_score > 3.0 and p_val < 0.001
    h_exp = calculate_hurst_exponent(ts_price_pairs)
    assert h_exp is not None and h_exp > 0.5
    s_2d, _, p_2d = calculate_mann_kendall(np.asarray(ts_price_pairs, dtype=float))
    assert s_2d == s_stat and p_2d == pytest.approx(p_val)

    tsg = TrendSignificanceGuard(max_p_value=0.05, min_hurst=0.50)
    dec_tsg = tsg.evaluate({"action": "ENTRY", "side": "BUY"}, ts_price_pairs)
    assert dec_tsg.allowed and "trend_statistically_significant" in dec_tsg.reason

    # 3. OrderbookSnapshot and external/internal guards accept percentage strings
    ob_snap = OrderbookSnapshot.from_dict({"symbol": "BTCUSDC", "mid_price": 90000.0, "imbalance_ratio": "65%"})
    assert ob_snap.imbalance_ratio == pytest.approx(0.65)
    owg = OrderbookWallGuard(min_buy_imbalance="30%", max_sell_imbalance="70%")
    assert owg.min_buy_imbalance == pytest.approx(0.30) and owg.max_sell_imbalance == pytest.approx(0.70)
    wdg = WhaleDivergenceGuard(taker_downscaled_scale="40%")
    assert wdg.taker_downscaled_scale == pytest.approx(0.40)
    fcg = FundingCrowdingGuard(crowded_scale="60%")
    assert fcg.crowded_scale == pytest.approx(0.60)
    weg = WeibullExhaustionGuard(exhausted_scale="35%")
    assert weg.exhausted_scale == pytest.approx(0.35)

    # 4. compute_dynamic_sentiment_interval normalizes percentage advance_ratio / prev_advance_ratio
    itv_s, rsn_s = compute_dynamic_sentiment_interval(
        base_interval_sec=1800.0,
        now_ts=1700010000.0,
        breadth={"advance_ratio": 55.0, "median_change_pct": 1.0, "regime": "NEUTRAL"},
        prev_advance_ratio=30.0,
    )
    assert "breadth_shift_+0.25" in rsn_s

    # 5. compute_dynamic_geopolitical_interval forwards cache_dir to get_active_acute_shock_phrases
    update_dynamic_lexicon(
        [],
        ["custom grid blackout"],
        cache_dir=str(tmp_path),
        now_ts=1700010000.0,
    )
    snap_news = NewsFeedSnapshot(
        headlines=(
            NewsHeadline(
                title="Breaking: custom grid blackout reported across exchange hub",
                source="Reuters",
                pub_date="2026-11-04",
                url="https://example.com/1",
                is_high_severity=True,
                matched_keywords=("blackout",),
            ),
        ),
        total_fetched=1,
        high_severity_count=1,
        has_critical_shock_keywords=True,
        ts=1700010000.0,
    )
    itv_g, rsn_g = compute_dynamic_geopolitical_interval(
        base_interval_sec=1800.0,
        last_eval_ts=1700009000.0,
        now_ts=1700010000.0,
        cached_assessment={"threat_level": "NORMAL", "is_fallback": False},
        news_snapshot=snap_news,
        headlines_changed=True,
        last_headlines_digest="older unrelated headline",
        cache_dir=str(tmp_path),
    )
    assert itv_g == 0.0 and rsn_g == "acute_shock_keyword_alert"

    # 6. ExternalTelemetryMixin._fetch_and_log forwards active cycle timestamp now to symbol collectors
    gtc = GlobalTelemetryCollector(cache_dir=str(tmp_path), symbols=["BTCUSDC"], skip_news=True)
    seen_ob_now = []
    gtc.orderbook_collector.fetch = lambda sym, allow_network=True, now=None: (seen_ob_now.append(now), ob_snap)[1]
    gtc.derivatives_collector.fetch = lambda sym, allow_network=True, now=None: None
    gtc.whale_collector.fetch = lambda sym, allow_network=True, now=None: None
    gtc.fear_greed_collector.fetch = lambda force_refresh=False, now_ts=None: None
    gtc.breadth_collector.fetch = lambda force_refresh=False, now_ts=None: None
    gtc.run_cycle(force=True, now=1700012345.0)
    assert seen_ob_now == [1700012345.0]


def test_iter60_kalman_ms_timestamps_bybit_lookup_and_whale_percent_scales(tmp_path):
    from types import SimpleNamespace
    import shadow_signals as ss
    from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
    from intelligence.external.collectors.whale_positioning_collector import (
        WhalePositioningCollector,
        WhalePositioningSnapshot,
    )
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard
    from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger

    # 1. KalmanTrend / KalmanTrendTrigger normalize millisecond timestamps so 60s steps do not trigger gap reset
    ktt = KalmanTrendTrigger()
    t0_ms = 1700000000000.0
    px = 100.0
    fired_ev = None
    for i in range(30):
        px += 0.25
        out_k, ev_k = ktt.evaluate("BTCUSDC", t0_ms + i * 60_000.0, px, epsilon=0.05)
        if ev_k is not None:
            fired_ev = ev_k
    assert out_k["trend"] == 1
    assert fired_ev is not None and fired_ev.ts < 1e11

    # 2. KalmanStateTracker supports (ts, price) pairs in big_prices and millisecond timestamps
    journal = ss.KalmanJournal(fixed_path=os.devnull)
    tracker = ss.KalmanStateTracker(journal=journal, now_fn=lambda: 1700000060.0, compute_vol=False)
    pair_hist = [(1700000000000.0 - (20 - i) * 60_000.0, 100.0 + i * 0.1) for i in range(20)]
    tracker.update("BTCUSDC", 1700000000000.0, 102.5, 0.05, pair_hist, 60.0)
    tr_val, age_sec = tracker.current_trend("btcusdc")
    assert tr_val in (-1, 0, 1) and age_sec == pytest.approx(60.0)

    # 3. BybitLiquidationCollector unifies casefold/variant symbol lookup and normalizes ms timestamps
    blc = BybitLiquidationCollector()
    blc.record_event("BTC/USDT", "Buy", 150_000.0, ts=1700000000000.0)
    blc.record_event("BTC/USDT", "Sell", 30_000.0, ts=1700000010000.0)
    sum_b = blc.get_summary("btcusdc", window_sec=300.0, now=1700000020.0)
    assert sum_b.long_liq_usd == pytest.approx(150_000.0)
    assert sum_b.short_liq_usd == pytest.approx(30_000.0)
    sig_b = blc.get_signal("BTC/USDC", window_sec=300.0, now=1700000020000.0)
    assert sig_b["long_liq_usd"] == pytest.approx(150_000.0)

    # 4. LiquidationCapitulationTrigger, WhaleAccumulationTrigger, SmartStopLossGuard, and WhalePositioningSnapshot percent scales
    lct = LiquidationCapitulationTrigger(ratio_threshold=80.0, min_notional_usd=100_000.0)
    assert lct.ratio_threshold == pytest.approx(0.80)
    ev_liq = lct.evaluate("BTCUSDC", 90000.0, summary=SimpleNamespace(long_liq_usd=160_000.0, short_liq_usd=20_000.0), now=1700000020.0)
    assert ev_liq is not None and ev_liq.reason == "liquidation_capitulation_burst"

    wat = WhaleAccumulationTrigger(min_top_traders_long_pct=65.0, min_top_traders_short_pct="70%")
    assert wat.min_top_traders_long_pct == pytest.approx(0.65)
    assert wat.min_top_traders_short_pct == pytest.approx(0.70)
    ev_wat = wat.evaluate(
        "BTCUSDC",
        90000.0,
        snapshot={"whale_long_pct": "68%", "taker_buy_sell_ratio": 1.35, "oi_1h_chg_pct": 1.5},
        now=1700000020.0,
    )
    assert ev_wat is not None and ev_wat.reason == "whale_aggressive_accumulation"

    wps = WhalePositioningSnapshot.from_dict({"symbol": "BTCUSDC", "whale_long_pct": 35.0})
    assert wps.top_traders_long_pct == pytest.approx(0.35)
    assert WhalePositioningCollector.classify_regime(1.0, 1.0, 1.25, top_long_pct=40.0) == "short_squeeze_primed"

    slg = SmartStopLossGuard(min_whale_long_pct=65.0)
    assert slg.min_whale_long_pct == pytest.approx(0.65)


def test_iter61_whale_spread_flow_sentiment_advisor_and_price_series_ordering(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import external_order_guard as eog
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from intelligence.sentiment.collectors.microstructure_collector import _filter_ohlc_closes_since
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _compute_short_slopes_from_series,
        _extract_ohlc_series_points,
        compute_7d_range_metrics,
        compute_cross_btc_metrics,
        compute_swing_wave_metrics,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        enrich_microstructure_cvd,
        update_microstructure_cycle_deltas,
    )

    # 1. evaluate_whale_spread_flow & resolve_whale_spread_adjustment support side aliases, dict snapshots, and no-kwarg collectors
    class _DictCollector:
        def fetch(self, symbol):
            return {"taker_buy_sell_ratio": 0.95, "oi_1h_chg_pct": 2.4, "div_regime": "aggressive_shorting"}

    monkeypatch.setattr(eog, "get_whale_collector", lambda: _DictCollector())
    blocked, reason = eog.evaluate_whale_spread_flow("BTCUSDC", "bid", enforced=True)
    assert blocked is True and "aggressive_shorting_divergence (OI=+2.4%)" in reason

    class _PressureCollector:
        def fetch(self, symbol):
            return {"taker_buy_sell_ratio": 0.85}

    monkeypatch.setattr(eog, "get_whale_collector", lambda: _PressureCollector())
    assert eog.resolve_whale_spread_adjustment("BTCUSDC", "0.002", enforced=True) == pytest.approx(0.003)

    # 2. SentimentAdvisor & GeopoliticalThreatAnalyzer percent string confidence/risk_score and breadth advance_ratio
    sa_built = SentimentAdvisor._build_assessment({"confidence": "80%", "market_bias": "BULLISH", "risk_level": "LOW"}, ts_override=100.0)
    assert sa_built.confidence == pytest.approx(0.80)
    ga = GeopoliticalThreatAnalyzer(state_file=str(tmp_path / "geo_iter61.json"))
    ga_built = ga._build_assessment({"risk_score": "75%", "threat_level": "ELEVATED"}, ts=100.0)
    assert ga_built.risk_score == pytest.approx(0.75)

    captured_prompts = []
    sa = SentimentAdvisor(cache_file=str(tmp_path / "sent_iter61.json"))
    monkeypatch.setattr(sa, "_query_topic", lambda prompt, **kw: captured_prompts.append(prompt) or {"confidence": "75%", "market_bias": "BULLISH"})
    sa.review(breadth={"advance_ratio": "55%", "median_change_pct": 1.2, "regime": "BROAD_ADVANCE"}, force_refresh=True, now_ts=200.0)
    assert any("55.0% advancing" in p for p in captured_prompts)

    # 3. Price series evaluators sort unsorted (ts, price) rows and normalize ms OHLC timestamps
    unsorted_7d = [[2000.0, 110.0], [1000.0, 100.0], [1500.0, 105.0]]
    m7d = compute_7d_range_metrics(unsorted_7d)
    assert m7d["chg_7d_pct"] == pytest.approx(10.0)
    sw = compute_swing_wave_metrics([[2000.0, 110.0], [1000.0, 100.0], [1500.0, 95.0]])
    assert sw["rally_from_swing_low_pct"] == pytest.approx(((110.0 / 95.0) - 1.0) * 100.0, abs=0.05)

    slopes = _compute_short_slopes_from_series([[1700000180000.0, 103.0], [1700000000000.0, 100.0], [1700000090000.0, 101.5]])
    assert slopes.get("slope_90s", 0.0) > 0.0

    ohlc_ms = SimpleNamespace(timestamps=[1700000120000.0, 1700000000000.0, 1700000060000.0], closes=[102.0, 100.0, 101.0], volumes=[10.0, 10.0, 10.0])
    pts_ohlc = _extract_ohlc_series_points(ohlc_ms, min_ts=1700000000.0, max_ts=1700000120.0)
    assert [p[0] for p in pts_ohlc] == [1700000000000, 1700000060000, 1700000120000]
    assert _filter_ohlc_closes_since(ohlc_ms, cutoff_ts=1700000000.0, max_ts=1700000120.0) == [100.0, 101.0, 102.0]

    alt_unsorted = [[1700000010000.0, 200.0], [1700000000000.0, 100.0], [1700000300000.0, 205.0], [1700000600000.0, 210.0], [1700000900000.0, 215.0]]
    btc_sorted = [[1700000000000.0, 50000.0], [1700000300000.0, 50000.0], [1700000600000.0, 50000.0], [1700000900000.0, 50000.0]]
    c_btc = compute_cross_btc_metrics(alt_unsorted, btc_sorted)
    assert c_btc.get("alt_btc_pos_24h_pct") == pytest.approx(100.0)

    # 4. Microstructure evaluator normalizes percent ob_imbalance / whale_long_pct and ms timestamps in cycle deltas
    minfo = {"ob_imbalance": "65%", "whale_long_pct": "68%", "taker_buy_vol_usd": 1000.0, "taker_sell_vol_usd": 1500.0}
    enrich_microstructure_cvd(minfo)
    assert minfo["cvd_regime"] == "spot_absorbing"

    hist_file = tmp_path / "microstructure_cycle_history.json"
    hist_file.write_text(
        '{"BTCUSDC": [{"ts": 1700000000000.0, "ob_imbalance": "55%", "whale_long_pct": "60%"}]}',
        encoding="utf-8",
    )
    micro_map = {"BTCUSDC": {"ob_imbalance": "65%", "whale_long_pct": "68%"}}
    update_microstructure_cycle_deltas(str(tmp_path), micro_map, now_ts=1700000600000.0)
    cd = micro_map["BTCUSDC"].get("cycle_delta")
    assert isinstance(cd, dict)
    assert cd["ob_imbalance_delta"] == pytest.approx(0.10)
    assert cd["whale_long_delta_pct"] == pytest.approx(0.08)


def test_iter62_tca_risk_pretrade_and_macro_telemetry_normalization(monkeypatch, tmp_path):
    import external_order_guard as eog
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        collect_macro_benchmarks_snapshot,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
        collect_pretrade_telemetry,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        apply_asset_scope_filter,
        build_cycle_fallback_prices,
        compute_deribit_options_surface,
        reconcile_live_venue_equity_summary,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
    )

    # 1. enrich_open_limit_orders_tca supports pair/ticker, side aliases (bid/ask), and px/limit_price keys
    open_orders = [
        {"pair": "BTCUSDC", "side": "bid", "px": 98.0},
    ]
    enrich_open_limit_orders_tca(
        open_orders,
        {"BTCUSDC": {"price": 100.0, "vol_1h_pct": 1.0}},
        {"BTCUSDC": {"ob_mid_price": 100.0, "bid_wall_usd": 250000.0, "bid_wall_px": 99.0}},
    )
    assert open_orders[0]["dist_from_spot_pct"] == pytest.approx(-2.0)
    assert open_orders[0]["wall_queue_pos"] == "BEHIND_BID_WALL"

    # 2. compute_enterprise_portfolio_risk_metrics, reconcile_live_venue_equity_summary, and apply_asset_scope_filter
    # infer venue from venue_quote_cash key ("binance:USDC" / "t212:USD") when "venue" field is omitted
    ps = {
        "tradable_positions": [
            {"symbol": "BTCUSDC", "venue": "binance", "value_usd": 1000.0},
        ],
        "venue_quote_cash": {
            "binance:USDC": {"free": 500.0},
        },
        "live_binance_equity_usdc": 1500.0,
    }
    risk = compute_enterprise_portfolio_risk_metrics(ps, {"BTCUSDC": {"price": 100.0}}, {})
    assert risk["unified_nav_usd"] == pytest.approx(1500.0)
    assert risk["venue_nav_breakdown_usd"].get("binance") == pytest.approx(1500.0)
    assert "unknown" not in risk["venue_nav_breakdown_usd"]

    ps_reconcile = {
        "tradable_positions": [{"symbol": "BTCUSDC", "venue": "binance", "value_usd": 1000.0}],
        "venue_quote_cash": {"binance:USDC": {"free_usd": 500.0}},
    }
    reconcile_live_venue_equity_summary(ps_reconcile)
    assert ps_reconcile.get("live_binance_equity_usdc") == pytest.approx(1500.0)

    ps_scope = {
        "venue_quote_cash": {
            "binance:USDC": {"free": 500.0},
            "t212:USD": {"free": 200.0},
        }
    }
    apply_asset_scope_filter({}, {}, ps_scope, "equities", {}, {})
    assert list(ps_scope["venue_quote_cash"].keys()) == ["t212:USD"]

    # 3. collect_pretrade_telemetry handles collectors without allow_network kwarg and percent/alias snapshot dicts
    class _NoKwObCollector:
        def fetch(self, symbol):
            return {"symbol": symbol, "ob_imbalance": "64%", "largest_ask_wall_usd": 200000.0, "largest_ask_wall_price": 105.0}

    class _NoKwWhaleCollector:
        def fetch(self, symbol):
            return {"symbol": symbol, "whale_long_pct": "60%", "taker_buy_sell_ratio": 1.35, "div_regime": "bullish_divergence"}

    monkeypatch.setattr(eog, "get_orderbook_collector", lambda: _NoKwObCollector())
    monkeypatch.setattr(eog, "get_whale_collector", lambda: _NoKwWhaleCollector())
    telem = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(tmp_path))
    assert telem.get("Orderbook Depth Imbalance") == "0.64 (64% bids vs 36% asks)"
    assert telem.get("Whale Positioning") == "60.0% Top Traders Long (L/S ratio: 1.50)"
    assert "bullish_divergence" in telem.get("Whale Taker Flow", "")

    # 4. collect_macro_benchmarks_snapshot sorts unsorted ms timestamps and compute_deribit_options_surface normalizes decimal mark_iv
    def _mock_yahoo_fetch(url, **kw):
        if "SPY" in url:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {"regularMarketPrice": 510.0, "previousClose": 500.0},
                            "timestamp": [1700003600000, 1700000000000, 1700002700000],
                            "indicators": {"quote": [{"close": [510.0, 500.0, 505.0]}]},
                        }
                    ]
                }
            }
        return {}

    mb_snap = collect_macro_benchmarks_snapshot(str(tmp_path), now_ts=1700003600000.0, persist=False, fetch_fn=_mock_yahoo_fetch)
    assert mb_snap is not None
    spy_item = mb_snap["items"]["SPY"]
    assert spy_item["chg_1h_pct"] == pytest.approx(2.0)
    assert spy_item["chg_15m_pct"] == pytest.approx(((510.0 - 505.0) / 505.0) * 100.0, abs=0.05)

    opt_surf = compute_deribit_options_surface(
        [
            {"instrument_name": "BTC-29DEC26-105000-C", "underlying_price": 100000.0, "open_interest": 100.0, "mark_iv": 0.50},
            {"instrument_name": "BTC-29DEC26-95000-P", "underlying_price": 100000.0, "open_interest": 120.0, "mark_iv": 0.58},
        ]
    )
    assert opt_surf.get("btc_skew_25d_pct") == pytest.approx(8.0)

    fb_px = build_cycle_fallback_prices({}, {"BTCUSDC": [[1700000120000.0, 102.0], [1700000000000.0, 99.0]]})
    assert fb_px["BTCUSDC"] == pytest.approx(102.0)


def test_iter63_microstructure_archive_sorting_post_stop_ms_and_news_iso_dates(monkeypatch, tmp_path, capsys):
    import json
    from intelligence.cli import _stamp_age, print_intelligence_report
    from intelligence.macro.calendar_ssot import did_cross_session_checkpoint
    from intelligence.macro.daemon_ssot import inspect_daemon_heartbeat
    from intelligence.macro.news_feed_collector import NewsFeedCollector
    from intelligence.sentiment.collectors.microstructure_collector import (
        load_symbol_microstructure,
        load_symbol_price_series,
    )
    import intelligence.sentiment.guards.post_stop_reentry_guard as psrg

    # 1. load_symbol_price_series sorts unsorted archive_pts before extracting vals_7d
    arch_file = tmp_path / "cache_365_price_BTCUSDC.json"
    arch_file.write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        [1700500000000, 110.0],
                        [1700000000000, 100.0],
                        [1700250000000, 105.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    s24, v7d = load_symbol_price_series(str(tmp_path), "BTCUSDC", include_7d=True)
    assert v7d == [100.0, 105.0, 110.0]

    # 2. load_symbol_microstructure normalizes percent ob_imbalance / whale_long_pct, string funding_rate, and div_regime alias
    m_info = load_symbol_microstructure(
        str(tmp_path),
        "BTCUSDC",
        ob={"symbol": "BTCUSDC", "ob_imbalance": "62%", "mid_price": 100.0, "bid_wall_px": "98.5"},
        deriv={"funding_rate": "0.0004", "pred_funding_rate": "0.0005"},
        whale={"whale_long_pct": "65%", "div_regime": "bullish_divergence"},
    )
    assert m_info is not None
    assert m_info["ob_imbalance"] == pytest.approx(0.62)
    assert m_info["whale_long_pct"] == pytest.approx(0.65)
    assert m_info["funding_rate"] == pytest.approx(0.0004)
    assert m_info["pred_funding_rate"] == pytest.approx(0.0005)
    assert m_info["bid_wall_px"] == pytest.approx(98.5)
    assert m_info["divergence_regime"] == "bullish_divergence"

    # 3. PostStopReentryGuard.check normalizes millisecond last_sell_time_s and percentage string bounce_pct
    captured_prompts = []
    monkeypatch.setattr(psrg, "collect_pretrade_telemetry", lambda *a, **kw: {})
    monkeypatch.setattr(psrg, "query_topic_json", lambda client, topic, prompt, **kw: captured_prompts.append(prompt) or {"decision": "APPROVED"})
    ps_guard = psrg.PostStopReentryGuard(gemini_client=object(), notify_enabled=False, cache_ttl_sec=600.0)
    dec1 = ps_guard.check(
        "BTCUSDC",
        current_price=102.5,
        last_sell_price=101.0,
        trough=100.0,
        bounce_pct="2.5%",
        last_sell_time_s=1700000000000.0,
        now=1700000600.0,
    )
    assert dec1.allowed
    assert any("Elapsed Time Since Stop-Loss: 10.0 minutes" in p and "+2.50% bounce from trough" in p for p in captured_prompts)
    # Cache hit when called with seconds timestamp for the same sell event
    dec2 = ps_guard.check(
        "BTCUSDC",
        current_price=102.5,
        last_sell_price=101.0,
        trough=100.0,
        bounce_pct=0.025,
        last_sell_time_s=1700000000.0,
        now=1700000610.0,
    )
    assert dec2.allowed and len(captured_prompts) == 1

    # 4. NewsFeedCollector._parse_recent_pub_date_ts supports ISO-8601 dates and ms timestamps in history
    iso_ts = NewsFeedCollector._parse_recent_pub_date_ts("2023-11-14T22:13:20Z", 1700000100.0)
    assert iso_ts == pytest.approx(1700000000.0)

    # 5. calendar_ssot, daemon_ssot, and cli normalize millisecond timestamps and percent scales
    cp = did_cross_session_checkpoint(1700020000000.0, 1700038000000.0, min_cooldown_sec=60.0)
    assert cp is None or isinstance(cp, str)

    hb_path = tmp_path / "test_daemon.heartbeat"
    hb_path.write_text(json.dumps({"pid": 99999999, "ts": 1700000000000.0}), encoding="utf-8")
    hb_res = inspect_daemon_heartbeat(str(tmp_path), ("test_daemon.heartbeat",))
    assert hb_res["age_seconds"] is not None and hb_res["age_seconds"] > 1000.0

    stamped = _stamp_age({"ts": 1700000000000.0}, 1700000030.0)
    assert stamped["age_seconds"] == pytest.approx(30.0)

    print_intelligence_report(
        {
            "symbol": "BTCUSDC",
            "pillar2_external": {
                "orderbook": {"ob_imbalance": "62%", "mid_price": 100.0},
                "whale": {"whale_long_pct": "65%", "div_regime": "bullish_divergence"},
            },
            "pillar4_macro": {"threat_level": "ELEVATED", "risk_score": "75%"},
            "guard_preview": {"buy_standard": {"allowed": True, "suggested_scale": "50%", "reason": "test"}},
        }
    )
    out_str = capsys.readouterr().out
    assert "0.62" in out_str and "65.0% longs" in out_str and "Risk Score: 0.75" in out_str and "Scale: 50%" in out_str


def test_iter64_breadth_rotation_fields_taxonomy_spaces_and_ms_timestamps(tmp_path):
    import json
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
    from intelligence.macro_sentiment_ssot import (
        load_cached_fear_greed_snapshot,
        load_cached_geopolitical_eval,
        load_cached_market_breadth_snapshot,
        load_cached_sentiment_advisor_eval,
        normalize_market_bias,
        normalize_recommended_action,
        normalize_recommended_brake,
        normalize_threat_level,
    )
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor

    # 1. Multi-word taxonomy tokens with spaces or hyphens normalize properly
    assert normalize_threat_level("CRITICAL SHOCK") == "CRITICAL_SHOCK"
    assert normalize_threat_level("critical-shock") == "CRITICAL_SHOCK"
    assert normalize_recommended_brake("HARD VETO NEW BUYS") == "HARD_VETO_NEW_BUYS"
    assert normalize_recommended_brake("downscale-50") == "DOWNSCALE_50"
    assert normalize_recommended_action("TRIM PROFITS") == "TRIM_PROFITS"
    assert normalize_recommended_action("take-profit") == "TRIM_PROFITS"
    assert normalize_market_bias("risk-off") == "BEARISH"

    # 2. load_cached_market_breadth_snapshot preserves all 10 rotation fields and normalizes percent advance_ratio / ms ts
    mb_path = tmp_path / "market_breadth_collect.json"
    mb_path.write_text(
        json.dumps(
            {
                "advance_ratio": "64%",
                "advancing_count": 32,
                "declining_count": 18,
                "total_count": 50,
                "median_change_pct": 1.8,
                "mean_change_pct": 2.1,
                "dispersion_std": 3.2,
                "regime": "BULLISH_BREADTH",
                "top_gainers": [["SOLUSDC", 8.5]],
                "top_losers": [["ADAUSDC", -2.1]],
                "ts": 1700000000000.0,
                "btc_price": 85000.0,
                "btc_chg_24h_pct": 2.5,
                "eth_price": 2550.0,
                "eth_chg_24h_pct": 4.2,
                "eth_btc_ratio": 0.03,
                "eth_btc_chg_24h_pct": 1.66,
                "btc_vol_share_pct": 42.0,
                "eth_vol_share_pct": 18.0,
                "alt_vol_share_pct": 40.0,
                "rotation_regime": "ETH_OUTPERFORMING",
            }
        ),
        encoding="utf-8",
    )
    mb_snap = load_cached_market_breadth_snapshot(str(tmp_path), now_ts=1700000100.0)
    assert mb_snap is not None
    assert mb_snap.advance_ratio == pytest.approx(0.64)
    assert mb_snap.ts == pytest.approx(1700000000.0)
    assert mb_snap.btc_price == pytest.approx(85000.0)
    assert mb_snap.eth_price == pytest.approx(2550.0)
    assert mb_snap.eth_btc_ratio == pytest.approx(0.03)
    assert mb_snap.alt_vol_share_pct == pytest.approx(40.0)
    assert mb_snap.rotation_regime == "ETH_OUTPERFORMING"

    # 3. load_cached_fear_greed_snapshot normalizes millisecond ts and falls back to index_timestamp
    fg_path = tmp_path / "fear_greed_collect.json"
    fg_path.write_text(
        json.dumps({"value": 72, "classification": "Greed", "index_timestamp": 1700000000000.0}),
        encoding="utf-8",
    )
    fg_snap = load_cached_fear_greed_snapshot(str(tmp_path), now_ts=1700000100.0)
    assert fg_snap is not None
    assert fg_snap.timestamp == pytest.approx(1700000000.0)

    # 4. GeopoliticalShockGuard and cached eval loaders normalize percentage risk_score/confidence and ms timestamps
    guard = GeopoliticalShockGuard(downscale_factor=0.5)
    g_dec = guard.check("BTCUSDC", "BUY", {"threat_level": "CRITICAL SHOCK", "risk_score": "85%", "summary": "Test shock"})
    assert not g_dec.allowed
    assert g_dec.metadata["risk_score"] == pytest.approx(0.85)

    (tmp_path / "geopolitical_eval.json").write_text(
        json.dumps({"threat_level": "CRITICAL SHOCK", "recommended_brake": "hard-veto", "risk_score": 85, "ts": 1700000000000}),
        encoding="utf-8",
    )
    geo_eval = load_cached_geopolitical_eval(str(tmp_path))
    assert geo_eval is not None
    assert geo_eval["threat_level"] == "CRITICAL_SHOCK"
    assert geo_eval["recommended_brake"] == "HARD_VETO_NEW_BUYS"
    assert geo_eval["risk_score"] == pytest.approx(0.85)
    assert geo_eval["ts"] == pytest.approx(1700000000.0)

    (tmp_path / "sentiment_eval.json").write_text(
        json.dumps({"market_bias": "risk-on", "risk_level": "elevated", "recommended_action": "trim profits", "confidence": 80, "ts": 1700000000000}),
        encoding="utf-8",
    )
    sent_eval = load_cached_sentiment_advisor_eval(str(tmp_path))
    assert sent_eval is not None
    assert sent_eval["market_bias"] == "BULLISH"
    assert sent_eval["risk_level"] == "HIGH"
    assert sent_eval["recommended_action"] == "TRIM_PROFITS"
    assert sent_eval["confidence"] == pytest.approx(0.80)
    assert sent_eval["ts"] == pytest.approx(1700000000.0)

    # 5. GeopoliticalThreatAnalyzer and SentimentAdvisor load ms timestamps and integer percent scores from disk
    ga = GeopoliticalThreatAnalyzer(gemini_client=object(), state_file=str(tmp_path / "geopolitical_eval.json"))
    assert ga._last_eval_ts == pytest.approx(1700000000.0)
    assert ga._cached_assessment is not None and ga._cached_assessment.risk_score == pytest.approx(0.85)

    sa = SentimentAdvisor(gemini_client=object(), cache_file=str(tmp_path / "sentiment_eval.json"))
    assert sa._last_eval_ts == pytest.approx(1700000000.0)
    assert sa._cached_assessment is not None and sa._cached_assessment.confidence == pytest.approx(0.80)


def test_iter65_internal_state_2d_series_external_snapshots_and_lexicon_ms_timestamps(monkeypatch, tmp_path):
    import json
    import numpy as np
    from intelligence.internal.state.volatility import calculate_volatility_1h
    from intelligence.internal.state.persistence import calculate_mann_kendall
    from intelligence.internal.state.survival import block_slopes, episodes
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
    from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
    from intelligence.internal.triggers.trigger_event import TriggerEvent
    from intelligence.external.collectors.base_collector import _DiskBackedSymbolCollector
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookSnapshot
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetry
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningSnapshot
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.macro.lexicon_ssot import (
        demote_dynamic_acute_terms,
        has_acute_shock_keywords,
        load_dynamic_lexicon,
        update_dynamic_lexicon,
    )

    # 1. calculate_volatility_1h handles 2D (ts, price) arrays and unsorted (ts, price) tuples identically to 1D prices
    px_1d = [100.0 + 0.5 * i + ((-1) ** i) * 0.2 for i in range(20)]
    pairs_asc = [(1700000000.0 + i * 60.0, p) for i, p in enumerate(px_1d)]
    pairs_desc = list(reversed(pairs_asc))
    v_1d = calculate_volatility_1h(px_1d, 60.0)
    v_2d = calculate_volatility_1h(np.asarray(pairs_asc, dtype=float), 60.0)
    v_desc = calculate_volatility_1h(pairs_desc)
    assert v_1d > 0.0
    assert v_2d == pytest.approx(v_1d)
    assert v_desc == pytest.approx(v_1d)

    # 2. calculate_mann_kendall sorts descending (ts, price) sequences chronologically so Z sign is preserved
    s_asc, z_asc, p_asc = calculate_mann_kendall(pairs_asc)
    s_desc, z_desc, p_desc = calculate_mann_kendall(pairs_desc)
    assert s_asc == s_desc > 0
    assert z_asc > 0.0
    assert z_desc == pytest.approx(z_asc)
    assert p_desc == pytest.approx(p_asc)

    # 3. survival.block_slopes and episodes normalize millisecond timestamps without hanging
    t_sec = np.array([1700000000.0 + i * 3600.0 for i in range(30)], dtype=float)
    t_ms = t_sec * 1000.0
    px_surv = np.array([100.0 + i * 0.5 for i in range(30)], dtype=float)
    bt_s, bs_s = block_slopes(t_sec, px_surv, window_h=6.0, step_h=2.0)
    bt_m, bs_m = block_slopes(t_ms, px_surv, window_h=6.0, step_h=2.0)
    assert len(bt_s) == len(bt_m) > 0
    assert np.allclose(bt_s, bt_m)
    assert np.allclose(bs_s, bs_m)
    ep_asc = episodes(bt_s * 1000.0, bs_s, 0.0001)
    ep_desc = episodes(bt_s[::-1] * 1000.0, bs_s[::-1], 0.0001)
    assert ep_asc == ep_desc

    # 4. Internal triggers and WeibullExhaustionGuard normalize ms timestamps and percentage scales
    eg = WeibullExhaustionGuard(exhausted_scale=25)
    assert eg.exhausted_scale == pytest.approx(0.25)
    gt = LinearGradientTrigger(strength_threshold=2.0)
    _, ev_gt = gt.evaluate("BTCUSDC", price=105.0, gradient=0.005, epsilon=0.001, ts=1700000000000.0)
    assert ev_gt is not None and ev_gt.ts == pytest.approx(1700000000.0)
    mrt = MeanReversionTrigger(mode="pb", bb_period=3, bb_k=1.0)
    mrt.update("BTCUSDC", 1700000000000.0, 100.0)
    mrt.update("BTCUSDC", 1700000000000.0, 101.0)
    ev_mr = mrt.update("BTCUSDC", 1700000000000.0, 80.0)
    assert ev_mr is not None and ev_mr.ts == pytest.approx(1700000000.0)
    ev_dir = TriggerEvent.directional(True, source="test", symbol="BTCUSDC", price=100.0, strength=1.0, ts=1700000000000.0, reason_bull="bull", reason_bear="bear")
    assert ev_dir.ts == pytest.approx(1700000000.0)

    # 5. _DiskBackedSymbolCollector._fetch_latest_json_row sorts descending rows by timestamp
    monkeypatch.setattr(
        "intelligence.external.collectors.base_collector.fetch_public_json",
        lambda url, **kw: [
            {"timestamp": 1700000300000, "val": "newest"},
            {"timestamp": 1700000000000, "val": "oldest"},
        ],
    )
    latest_row, sorted_rows = _DiskBackedSymbolCollector._fetch_latest_json_row("https://example.com", "test", "BTCUSDT")
    assert latest_row["val"] == "newest"
    assert [r["val"] for r in sorted_rows] == ["oldest", "newest"]

    # 6. External snapshots and guards normalize millisecond timestamps, numeric percentages, and div_regime alias
    ob_s = OrderbookSnapshot.from_dict({"symbol": "BTCUSDC", "imbalance_ratio": 65, "timestamp": 1700000000000})
    assert ob_s.imbalance_ratio == pytest.approx(0.65)
    assert ob_s.ts == pytest.approx(1700000000.0)

    dt_s = DerivativesTelemetry.from_dict({"symbol": "BTCUSDC", "funding_rate": 0.0001, "timestamp": 1700000000000})
    assert dt_s.ts == pytest.approx(1700000000.0)

    wp_s = WhalePositioningSnapshot.from_dict({"symbol": "BTCUSDC", "div_regime": "short_covering", "whale_long_pct": 68, "timestamp": 1700000000000})
    assert wp_s.divergence_regime == "short_covering"
    assert wp_s.top_traders_long_pct == pytest.approx(0.68)
    assert wp_s.ts == pytest.approx(1700000000.0)

    wdg = WhaleDivergenceGuard(taker_downscaled_scale=50)
    assert wdg.taker_downscaled_scale == pytest.approx(0.50)
    assert not wdg.check("BTCUSDC", "BUY", snapshot={"div_regime": "short_covering", "oi_1h_chg_pct": -1.5}).allowed

    owg = OrderbookWallGuard(min_buy_imbalance=25, max_sell_imbalance=75)
    assert owg.min_buy_imbalance == pytest.approx(0.25)
    assert owg.max_sell_imbalance == pytest.approx(0.75)

    fcg = FundingCrowdingGuard(crowded_scale=50)
    assert fcg.crowded_scale == pytest.approx(0.50)

    wat = WhaleAccumulationTrigger()
    ev_wa = wat.evaluate(
        "BTCUSDC",
        100.0,
        snapshot={"whale_long_pct": 65, "taker_buy_sell_ratio": 1.4, "oi_1h_chg_pct": 1.2, "div_regime": "accumulation"},
        now=1700000000000.0,
    )
    assert ev_wa is not None and ev_wa.metadata["regime"] == "accumulation"

    # 7. lexicon_ssot normalizes ms timestamps and supports single string / NewsHeadline objects in has_acute_shock_keywords
    lex_path = tmp_path / "macro_lexicon_collect.json"
    lex_path.write_text(
        json.dumps({"entries": {"missile strike": {"ts": 1700000000000, "tier": "acute"}}}),
        encoding="utf-8",
    )
    loaded_lex = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=1700000100.0)
    assert "missile strike" in loaded_lex["acute_shock_phrases"]
    assert has_acute_shock_keywords("Breaking: missile strike reported", cache_dir=str(tmp_path), now_ts=1700000100.0)
    assert has_acute_shock_keywords([{"title": "Breaking: missile strike reported"}], cache_dir=str(tmp_path), now_ts=1700000100.0)
    assert demote_dynamic_acute_terms(cache_dir=str(tmp_path), now_ts=1700000100.0) is True
    loaded_after = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=1700000100.0)
    assert "missile strike" in loaded_after["high_severity_terms"]
    assert "missile strike" not in loaded_after["acute_shock_phrases"]
    update_dynamic_lexicon(["sanctions package"], cache_dir=str(tmp_path), now_ts=1700000200000.0)
    assert "sanctions package" in load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=1700000250.0)["high_severity_terms"]


def test_iter66_guard_bypass_revocation_market_regime_ms_now_and_price_history_order(tmp_path, monkeypatch):
    import json
    from guard_hierarchy import (
        GUARD_REGISTRY,
        make_intel_cache_key,
        normalize_bypassed_guards,
        revoke_guard_bypass,
    )
    from market_regime import MarketRegimeContext, MarketRegimeEvidence, MarketRegimeService
    import order_guard
    from external_order_guard import evaluate_whale_spread_flow, is_gi_whale_enforced
    from llm_order_guard import check_geopolitical_order_guards, evaluate_high_stake_order_guard

    def is_guard_bypassed(code: str, bypassed: set[str]) -> bool:
        return GUARD_REGISTRY[code].is_bypassed(bypassed)

    # 1. revoke_guard_bypass: GLLM1 alias "llm" must not accidentally bypass GLLM3 when revoking GLLM2
    b_llm1 = normalize_bypassed_guards("GLLM1")
    revoke_guard_bypass(b_llm1, "GLLM2")
    assert is_guard_bypassed("GLLM1", b_llm1)
    assert not is_guard_bypassed("GLLM2", b_llm1)
    assert not is_guard_bypassed("GLLM3", b_llm1)

    # 2. revoke_guard_bypass on "all" / "*" and "profit_guard" and mixed-case set
    b_all = normalize_bypassed_guards("all")
    revoke_guard_bypass(b_all, "G_QLIMIT1")
    assert not is_guard_bypassed("G_QLIMIT1", b_all)
    assert is_guard_bypassed("G_PROF", b_all)
    assert is_guard_bypassed("GI_WHALE", b_all)

    b_prof = normalize_bypassed_guards("profit_guard")
    revoke_guard_bypass(b_prof, "G_QLIMIT1")
    assert not is_guard_bypassed("G_QLIMIT1", b_prof)
    assert is_guard_bypassed("G_PROF", b_prof)

    b_case = {"G_QLIMIT1"}
    revoke_guard_bypass(b_case, "g_qlimit1")
    assert not is_guard_bypassed("G_QLIMIT1", b_case)

    assert is_guard_bypassed("G_PROF", normalize_bypassed_guards("G_PROF"))
    assert make_intel_cache_key("buy", set()) == ("BUY", frozenset())

    # 3. MarketRegimeContext / MarketRegimeEvidence / MarketRegimeService with millisecond `now`
    ctx = MarketRegimeContext.from_decision(
        decision={"regime": "bull", "fresh": True, "source": "snapshot"},
        evaluated_at=1700000000000.0,
        symbol="BTCUSDC",
    )
    assert ctx.evaluated_at == pytest.approx(1700000000.0)
    assert ctx.is_valid_for("BTCUSDC", max_age_seconds=60.0, now=1700000030000.0)
    assert not ctx.is_valid_for("BTCUSDC", max_age_seconds=60.0, now=1700000090000.0)

    ev = MarketRegimeEvidence(
        role="asset",
        family="price_trend",
        classifier="snapshot",
        provider="binance",
        symbol="BTCUSDC",
        decision=ctx.decision,
        evaluated_at=1700000015000.0,
        observed_at=1700000000000.0,
    )
    assert ev.age_seconds == pytest.approx(15.0)

    svc = MarketRegimeService()
    snap = {
        "growth_coefficient": 0.005,
        "epsilon": 0.001,
        "n_samples": 30,
        "window_seconds": 900.0,
        "ts": 1700000000.0,
    }
    res = svc.resolve_with_evidence(
        provider="binance",
        symbol="BTCUSDC",
        snapshot=snap,
        now=1700000010000.0,
        allow_fallback=False,
    )
    assert res.decision.regime == "bull"
    assert res.decision.fresh is True
    assert res.primary is not None and res.primary.age_seconds == pytest.approx(10.0)
    assert svc.resolve(provider="binance", symbol="BTCUSDC", snapshot=snap, now=1700000010000.0, allow_fallback=False).regime == "bull"

    # 4. order_guard._read_cached_price_history with millisecond `now` and >512 descending ticks
    base_ts = 1700000000.0
    # 600 ticks in descending order (newest first: base_ts + 599 down to base_ts)
    ticks_desc = [
        {"ts": base_ts + i, "price": 50000.0 + i * 10.0}
        for i in range(599, -1, -1)
    ]
    price_file = tmp_path / "cache_24_price_BTCUSDC.json"
    price_file.write_text(json.dumps({"items": ticks_desc}), encoding="utf-8")
    monkeypatch.setattr(order_guard, "_resolve_cachedb_dir", lambda: str(tmp_path))

    hist = order_guard._read_cached_price_history(
        "BTCUSDC",
        now=(base_ts + 600.0) * 1000.0,  # millisecond `now`
        window_seconds=120.0,
    )
    assert len(hist) >= 100
    assert hist[0][0] < hist[-1][0]
    assert hist[-1][0] == pytest.approx(base_ts + 599.0)

    # 5. evaluate_whale_spread_flow uppercase / short_building & percentage downscale in llm_order_guard
    import external_order_guard
    monkeypatch.setattr(external_order_guard, "get_whale_collector", lambda: object())
    monkeypatch.setattr(
        external_order_guard,
        "fetch_guard_snapshot",
        lambda collector, sym, **kw: {"div_regime": "SHORT_BUILDING", "oi_1h_chg_pct": 1.5, "whale_long_pct": 0.35},
    )
    ws_conflict, ws_reason = evaluate_whale_spread_flow(
        "BTCUSDC",
        "BUY",
        enforced=True,
        now=1700000010000.0,
    )
    assert ws_conflict is True
    assert "aggressive_shorting_divergence" in ws_reason
    assert not is_gi_whale_enforced(bypass_guards={"GI_WHALE"})

    ctx_hs = MarketRegimeContext.from_decision(
        decision={"regime": "bull", "fresh": True, "source": "snapshot"},
        evaluated_at=1700000000.0,
        symbol="BTCUSDC",
    )
    object.__setattr__(ctx_hs, "_gemini_evaluated_notional", 1000.0)
    object.__setattr__(ctx_hs, "_gemini_decision", (True, "downscaled", 50))
    hs_ok, hs_reason, hs_scale = evaluate_high_stake_order_guard(
        symbol="BTCUSDC",
        side="BUY",
        price=50000.0,
        qty=0.02,
        computed_notional=1000.0,
        regime_context=ctx_hs,
        valid_context=True,
        cache_key=("BUY", frozenset()),
        margins={"llm_guard_mode": "enforce", "high_stake_min_notional_usd": 500.0},
        bypassed=set(),
        now=1700000010.0,
        notify_on_eval=False,
    )
    assert hs_ok is True and hs_scale == pytest.approx(0.50)

    geo_ok, geo_reason, geo_scale = check_geopolitical_order_guards("BTCUSDC", "BUY", bypass_guards={"GLLM3"})
    assert geo_ok is True and geo_scale == pytest.approx(1.0)


def test_iter67_pricewindow_sample_rate_cachetrend_ms_and_composite_granular_qlimit2(tmp_path, monkeypatch):
    from pricewindow import IncrementalTrendBuffer, PriceWindow
    from cachetrend import CacheShortTrendManager
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
    import qlimit_order_guard

    # 1. PriceWindow._sample_rate_from_entries handles both seconds and milliseconds timestamps identically
    entries_sec = [[1700000000.0 + i * 10.0, 100.0 + i] for i in range(10)]
    entries_ms = [[(1700000000.0 + i * 10.0) * 1000.0, 100.0 + i] for i in range(10)]
    assert PriceWindow._sample_rate_from_entries(entries_sec) == pytest.approx(10.0)
    assert PriceWindow._sample_rate_from_entries(entries_ms) == pytest.approx(10.0)

    # 2. IncrementalTrendBuffer.step_wols_slope handles overflow power gracefully without returning NaN
    buf = IncrementalTrendBuffer()
    for i in range(10):
        buf.append(100.0 + i, 1700000000.0 + i * 10.0)
    slope_overflow = buf.step_wols_slope(0, 9, power=1e6)
    assert slope_overflow == 0.0

    # 3. CacheShortTrendManager._snap_ts_ms, _snapshot_is_usable, and _normalize_dynamic_window_request normalize ms timestamps
    assert CacheShortTrendManager._snap_ts_ms({"ts": 1700000000.0}) == 1700000000000
    assert CacheShortTrendManager._snap_ts_ms({"ts": 1700000000000.0}) == 1700000000000
    snap_sec = {"ts": 1700000000.0, "current_price": 50000.0}
    snap_ms = {"ts": 1700000000000.0, "current_price": 50000.0}
    assert CacheShortTrendManager._snapshot_is_usable(snap_sec, 1700000010000.0, 60.0) is True
    assert CacheShortTrendManager._snapshot_is_usable(snap_ms, 1700000010.0, 60.0) is True
    mgr = CacheShortTrendManager.__new__(CacheShortTrendManager)
    norm_req = mgr._normalize_dynamic_window_request("BTCUSDC", 900.0, now=1700000010000.0)
    assert norm_req is not None and norm_req[2] == pytest.approx(1700000010.0)

    # 4. CompositeMarketIntelligence.evaluate_guards normalizes ms `now` and supports granular parabolic_surge vs weibull_exhaustion bypass
    intel = CompositeMarketIntelligence(parabolic_surge_pct=4.0, exhaustion_policy="downscale", exhaustion_scale=0.25)
    liq_col = BybitLiquidationCollector()
    liq_col.record_event("BTCUSDC", "LONG", 600_000.0, ts=1700000000.0)
    intel.cascade_guard.collector = liq_col
    dec_casc = intel.evaluate_guards("BTCUSDC", "BUY", 80000.0, now=1700000010000.0)
    assert not dec_casc.allowed
    assert "active_long_liquidation_cascade" in dec_casc.reason
    intel.cascade_guard.collector = None

    monkeypatch.setattr(
        "intelligence.composite.get_trend_survival_metrics",
        lambda sym, dur: {"p90_days": 1.0, "median_days": 0.5, "gaussian_horizon_days": 0.7},
    )
    base_ts = 1700000000.0
    surge_hist = [(base_ts, 80000.0), (base_ts + 60.0, 83000.0), (base_ts + 120.0, 86000.0)]
    now_ms = (base_ts + 120.0) * 1000.0
    # Bypassing only parabolic_surge lets WeibullExhaustionGuard downscale an exhausted 3-day trend
    dec_para_only = intel.evaluate_guards(
        "ETHUSDC",
        "BUY",
        86000.0,
        price_history=surge_hist,
        trend_duration_seconds=3.0 * 86400.0,
        now=now_ms,
        bypass_guards={"parabolic_surge"},
    )
    assert dec_para_only.allowed
    assert dec_para_only.suggested_scale == pytest.approx(0.25)

    # Bypassing only weibull_exhaustion still enforces ParabolicSurgeGuard
    dec_weib_only = intel.evaluate_guards(
        "ETHUSDC",
        "BUY",
        86000.0,
        price_history=surge_hist,
        trend_duration_seconds=3.0 * 86400.0,
        now=now_ms,
        bypass_guards={"weibull_exhaustion"},
    )
    assert not dec_weib_only.allowed

    # Bypassing full G_QLIMIT2 bypasses both
    dec_both = intel.evaluate_guards(
        "ETHUSDC",
        "BUY",
        86000.0,
        price_history=surge_hist,
        trend_duration_seconds=3.0 * 86400.0,
        now=now_ms,
        bypass_guards={"G_QLIMIT2"},
    )
    assert dec_both.allowed and dec_both.suggested_scale == pytest.approx(1.0)

    # Same granular behavior in qlimit_order_guard.check_qlimit2_order_guards
    margins = {
        "intelligence_guards_mode": "enforce",
        "internal_guard_mode": "enforce",
        "parabolic_surge_pct": 4.0,
        "parabolic_pullback_pct": 1.5,
        "weibull_exhaustion_policy": "downscale",
        "weibull_exhausted_scale": 0.25,
        "weibull_p90_days": 1.0,
    }
    q_ok, q_reason, q_scale = qlimit_order_guard.check_qlimit2_order_guards(
        "ETHUSDC",
        "BUY",
        86000.0,
        price_history=surge_hist,
        trend_duration_seconds=3.0 * 86400.0,
        margins=margins,
        now=now_ms,
        bypass_guards={"parabolic_surge"},
    )
    assert q_ok is True and q_scale == pytest.approx(0.25)


def test_iter68_accepted_ttl_map_spot_rules_ms_and_exit_manager_ms(monkeypatch):
    import order_retry
    from strategies import spot_rules
    from strategies.position_exit_manager import (
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
        ReentryAction,
        ReentrySnapshot,
    )

    # 1. order_retry.get_accepted_order_ttl_map only includes lifecycle == "accepted"
    fake_records = [
        {"lifecycle": "submit_pending", "order_id": "101", "ttl_sec": 300.0, "place_kwargs": {"client_order_id": "cid-101"}},
        {"lifecycle": "awaiting_cancel", "order_id": "102", "ttl_sec": 450.0, "place_kwargs": {"client_order_id": "cid-102"}},
        {"lifecycle": "accepted", "order_id": "103", "ttl_sec": 600.0, "place_kwargs": {"client_order_id": "cid-103"}},
    ]
    monkeypatch.setattr(order_retry, "_safe_load_all", lambda: fake_records)
    ttl_map = order_retry.get_accepted_order_ttl_map()
    assert "101" not in ttl_map and "cid-101" not in ttl_map
    assert "102" not in ttl_map and "cid-102" not in ttl_map
    assert ttl_map["103"] == pytest.approx(600.0) and ttl_map["cid-103"] == pytest.approx(600.0)

    # 2. spot_rules: check_fast_profit_reversal and check_slow_grind_exhaustion normalize ms (>1e11) and descending shadows
    base_s = 1700000000.0
    shadows_ms_desc = [
        ((base_s + 50.0) * 1000.0, 111.0),
        ((base_s + 30.0) * 1000.0, 115.0),
        ((base_s + 10.0) * 1000.0, 108.0),
    ]
    trig_fast, gain_pct, drop_pct = spot_rules.check_fast_profit_reversal(
        current_price=111.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=shadows_ms_desc,
        window_sec=60.0,
        drop_pct=2.0,
        current_time=base_s + 55.0,
    )
    assert trig_fast is True
    assert gain_pct == pytest.approx(11.0)
    assert drop_pct == pytest.approx((115.0 - 111.0) / 115.0 * 100.0)

    # check_slow_grind_exhaustion should NOT falsely trigger when entry_ts and current_ts are 900s apart in ms
    trig_grind_short, _ = spot_rules.check_slow_grind_exhaustion(
        current_price=116.0,
        avg_cost=100.0,
        entry_ts=base_s * 1000.0,
        current_ts=(base_s + 900.0) * 1000.0,
        min_days=7.0,
        min_gain_pct=15.0,
        recent_peak=120.0,
        shadow_prices=shadows_ms_desc,
        flash_window_sec=900.0,
        flash_drop_pct=1.5,
        structural_drop_pct=2.5,
    )
    assert trig_grind_short is False

    # And when genuinely >= 7 days in ms, it triggers properly
    trig_grind_long, reason_grind = spot_rules.check_slow_grind_exhaustion(
        current_price=116.0,
        avg_cost=100.0,
        entry_ts=base_s * 1000.0,
        current_ts=(base_s + 8.0 * 86400.0) * 1000.0,
        min_days=7.0,
        min_gain_pct=15.0,
        recent_peak=120.0,
        shadow_prices=[],
        flash_window_sec=900.0,
        flash_drop_pct=1.5,
        structural_drop_pct=2.5,
    )
    assert trig_grind_long is True and "slow_grind_structural_drop" in reason_grind

    # 3. PositionSnapshot and ReentrySnapshot normalize ms timestamps (>1e11) to seconds
    ps = PositionSnapshot(symbol="BTCUSDC", net_qty=1.0, sellable_qty=1.0, avg_buy_price=100.0, current_price=105.0, current_time_s=base_s * 1000.0)
    assert ps.current_time_s == pytest.approx(base_s)

    rs = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=99.5,
        last_sell_price=100.0,
        last_exit_kind="STOP",
        last_sell_time_s=base_s * 1000.0,
        current_time_s=(base_s + 30.0) * 1000.0,
    )
    assert rs.last_sell_time_s == pytest.approx(base_s)
    assert rs.current_time_s == pytest.approx(base_s + 30.0)

    mgr = PositionExitManager(PositionExitConfig(reentry_post_stop_cooldown_sec=60.0, reentry_sl_bounce_pct=0.01))
    dec = mgr.evaluate_reentry(rs)
    assert dec.action == ReentryAction.WAIT and "post_stop_stabilization_cooldown" in dec.reason and "elapsed=30s < 60s" in dec.reason


def test_iter69_spot_engine_sma_zero_ms_timestamps_and_monitororder_initial_prices(monkeypatch):
    from types import SimpleNamespace
    import monitororder
    from strategies.mt_position_accounting import _update_mt_reentry_state
    from strategies.spot_engine import Strategy, StratParams

    # 1. Strategy._sma_from_closes safely handles trend_sma_n <= 0 and non-finite elements without ZeroDivisionError
    assert Strategy._sma_from_closes([100.0, 102.0, 104.0], 0) is None
    assert Strategy._sma_from_closes([100.0, 102.0, 104.0], -2) is None
    assert Strategy._sma_from_closes([100.0, float("nan"), 104.0], 3) is None
    assert Strategy._sma_from_closes([100.0, 102.0, 104.0], 3) == pytest.approx(102.0)

    # 2. Strategy._surge_window_move_pct handles None/NaN past_price and set_simulated_time normalizes ms (> 1e11)
    class _DummyExec:
        def pair_precision(self, pair):
            return SimpleNamespace(price_decimals=2, volume_decimals=4, order_min=0.001, base_asset="BTC")

    params = StratParams.from_dict({"trend_sma_n": 0, "surge_guard": True, "surge_window_hours": 24.0, "trend_interval": 240})
    strat = Strategy(_DummyExec(), "BTCUSDC", params, dry_run=True, replay_mode=True, initial_state={"entry_ts": 1700000000000.0})
    assert strat.s["entry_ts"] == pytest.approx(1700000000.0)
    strat.set_simulated_time(1700000600000.0)
    assert strat._now() == pytest.approx(1700000600.0)
    assert strat._surge_window_move_pct(110.0, closes=[None, float("nan"), 100.0, 105.0, 108.0, 109.0, 110.0, 110.0]) is None

    # 3. _update_mt_reentry_state normalizes millisecond timestamps (> 1e11)
    rmap = {}
    st = _update_mt_reentry_state(
        "BTCUSDC",
        sell_price=100.0,
        current_price=98.0,
        current_time_s=1700000100000,
        sell_time=1700000000000.0,
        reentry_state_map=rmap,
    )
    assert st["sell_time"] == pytest.approx(1700000000.0)
    assert st["started_at"] == pytest.approx(1700000100.0)
    # Subsequent call with same sell_time in seconds does not reset trough/peak
    st2 = _update_mt_reentry_state(
        "BTCUSDC",
        sell_price=100.0,
        current_price=99.0,
        current_time_s=1700000120,
        sell_time=1700000000.0,
        reentry_state_map=rmap,
    )
    assert st2["trough"] == pytest.approx(98.0)
    assert st2["peak_bounce"] == pytest.approx(99.0)

    # 4. monitororder preserves initial_prices across str vs int order_id keys
    monitororder.reset_monitororder_state_for_test()
    monitororder.initial_sell_prices["555"] = 100.0
    monkeypatch.setattr(
        monitororder.api,
        "get_open_orders",
        lambda side, sym: {555: {"orderId": 555, "price": 100.1, "origQty": 1.0, "executedQty": 0.0, "clientOrderId": "TA_1"}},
    )
    monkeypatch.setattr(monitororder.api, "get_current_price", lambda sym: 105.0)
    monkeypatch.setattr(monitororder.order_retry, "get_accepted_order_ttl_map", lambda: {})
    monkeypatch.setattr(monitororder.order_retry, "get_accepted_order_owner_map", lambda: {})
    monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")
    assert monitororder.initial_sell_prices[555] == pytest.approx(100.0)


def test_iter70_t212_state_ms_shadow_vol_tradeall_and_rtrade_recovery():
    import importlib.util
    import os
    from types import SimpleNamespace
    import utils as u
    import shadow_signals
    import tradeall
    from intelligence.internal.state.volatility import calculate_volatility_1h
    from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairCoordinator, PairPolicy
    from strategies.spot_engine import Strategy as SpotStrategy, StratParams as SpotParams

    # 1. utils.timeToHMS normalizes millisecond timestamps (> 1e11) instead of returning N/A
    assert u.timeToHMS(1700000000000.0) == u.timeToHMS(1700000000.0)
    assert u.timeToHMS(1700000000000.0) != "N/A"

    # 2. tradeall.TrendState normalizes millisecond clock values (> 1e11) for freshness/expiration/cooldown
    now_ms = [1700000000000.0]
    ts = tradeall.TrendState(
        max_duration_seconds=3600,
        expiration_trend_time=120,
        fresh_trend_time=200,
        now_fn=lambda: now_ms[0],
    )
    ts.start_trend("UP")
    assert ts.start_time == pytest.approx(1700000000.0)
    now_ms[0] = (1700000000.0 + 30.0) * 1000.0
    ts.confirm_trend()
    assert ts.is_trend_fresh() is True
    assert ts.check_trend_expiration() is False
    assert ts.get_confirmed_trend_duration() == pytest.approx(30.0)

    # 3. shadow_signals.KalmanStateTracker.update normalizes symbol case and migrates existing _state entry
    tracker = shadow_signals.KalmanStateTracker(state_path=None, compute_vol=False)
    tracker._state["btcusdc"] = {"kalman_trend": 1, "ts": 1700000000.0}
    tracker.update("btcusdc", 1700000060.0, 100.0, 0.1, [100.0, 100.5], 1.0)
    assert "BTCUSDC" in tracker._state
    assert "btcusdc" not in tracker._state

    # 4. BaseSpotMixin._shadow_vol_1h normalizes ms timestamps (> 1e11), descending order, and ignores inf prices
    class _DummyExec:
        def pair_precision(self, pair):
            return SimpleNamespace(price_decimals=2, volume_decimals=4, order_min=0.001, base_asset="BTC")

    strat = SpotStrategy(_DummyExec(), "BTCUSDC", SpotParams.from_dict({}), dry_run=True, replay_mode=True)
    pts_sec = [(1700000000.0 + i * 60.0, 100.0 + (i % 3) * 0.5) for i in range(25)]
    strat._shadow_prices = list(pts_sec)
    vol_sec = strat._shadow_vol_1h()
    assert vol_sec is not None and vol_sec > 0.0

    # Millisecond descending shadow prices with a corrupted +inf row produce the exact same volatility
    pts_ms_desc = [((t * 1000.0), p) for t, p in reversed(pts_sec)] + [(1700009999000.0, float("inf"))]
    strat._shadow_prices = pts_ms_desc
    vol_ms = strat._shadow_vol_1h()
    assert vol_ms == pytest.approx(vol_sec)

    # calculate_volatility_1h handles leading None before (ts, price) tuples
    assert calculate_volatility_1h([None] + pts_ms_desc[:-1], 60.0) == pytest.approx(
        calculate_volatility_1h(pts_sec, 60.0)
    )

    # 5. 212trading _sanitize_t212_state normalizes ms timestamps on orders, pending_submit, and backoff/fail_until keys
    import sys
    t212_spec = importlib.util.spec_from_file_location(
        "t212_strategy_iter70",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "212trading", "strategy.py"),
    )
    t212_mod = importlib.util.module_from_spec(t212_spec)
    sys.modules["t212_strategy_iter70"] = t212_mod
    try:
        t212_spec.loader.exec_module(t212_mod)
    finally:
        sys.modules.pop("t212_strategy_iter70", None)
    sanitized = t212_mod._sanitize_t212_state(
        {
            "buy_backoff_until": 1700000500000.0,
            "locked_zero_until": 1700000600000.0,
            "tr_fail_until": 1700000700000.0,
            "sl_fail_until": 1700000800000.0,
            "tp_fail_until": {"5.0": 1700000900000.0},
            "pending_submit": {
                "side": "BUY",
                "qty": 1.0,
                "limit": 100.0,
                "intent_id": "intent-1",
                "ts": 1700000100000.0,
            },
            "orders": [
                {"id": "ord-1", "side": "BUY", "qty": 1.0, "limit": 100.0, "ts": 1700000200000.0}
            ],
        }
    )
    assert sanitized["buy_backoff_until"] == pytest.approx(1700000500.0)
    assert sanitized["locked_zero_until"] == pytest.approx(1700000600.0)
    assert sanitized["tr_fail_until"] == pytest.approx(1700000700.0)
    assert sanitized["sl_fail_until"] == pytest.approx(1700000800.0)
    assert sanitized["tp_fail_until"]["5.0"] == pytest.approx(1700000900.0)
    assert sanitized["pending_submit"]["ts"] == pytest.approx(1700000100.0)
    assert sanitized["orders"][0]["ts"] == pytest.approx(1700000200.0)

    # 6. PairCoordinator.from_state normalizes phase case, rejects unknown phases, and validates finite clock
    base_state = {
        "pair_id": "p1",
        "qty": 1.0,
        "start_side": "BUY",
        "phase": "STOPPING",
        "tickets": [vars(OrderTicket("o1", "SELL", 95.0, 1.0))],
        "snapshots": {"o1": vars(OrderSnapshot(status="open", filled_qty=0.0, cost=0.0, fee=0.0))},
    }
    pol = PairPolicy(adjustment_fraction=0.01)
    pc = PairCoordinator.from_state(object(), pol, base_state, clock=lambda: 10.0)
    assert pc.phase == "stopping"
    assert pc.stop_ticket is not None and pc.stop_ticket.order_id == "o1"

    with pytest.raises(ValueError, match="invalid rtrade checkpoint phase"):
        PairCoordinator.from_state(object(), pol, dict(base_state, phase="bogus_phase"), clock=lambda: 10.0)

    with pytest.raises(ValueError, match="clock must be finite"):
        PairCoordinator.from_state(object(), pol, base_state, clock=lambda: float("nan"))


def test_iter71_dotenv_strict_ini_cache_memo_and_sibling_sync(tmp_path):
    import json
    import subprocess
    import botcore
    import config as cfg_mod
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        reconcile_sibling_flow_and_liq_regimes,
        sync_sibling_symbol_telemetry,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _extract_sorted_ts_px,
        compute_24h_price_metrics,
    )

    # 1. parse_dotenv_line preserves quoted '#' in strict=True and rejects unquoted '#' before '=' in both modes
    assert botcore.parse_dotenv_line('KEY="val#with#hash" # inline comment', strict=True) == ("KEY", "val#with#hash")
    assert botcore.parse_dotenv_line("export\tTAB_KEY = 'abc#123' # comment", strict=True) == ("TAB_KEY", "abc#123")
    assert botcore.parse_dotenv_line("unassigned_token # comment with a=b", strict=False) is None
    with pytest.raises(ValueError, match="Malformed configuration line"):
        botcore.parse_dotenv_line("unassigned_token # comment with a=b", strict=True)
    with pytest.raises(ValueError, match="Empty configuration key"):
        botcore.parse_dotenv_line("export = 123", strict=True)

    # 2. required_int_env preserves >53-bit integer precision and _clean_env_str preserves quoted '#'
    large_id = (1 << 53) + 7
    assert botcore.required_int_env("LARGE_ID", {"LARGE_ID": str(large_id)}) == large_id
    assert botcore.required_int_env("LARGE_ID", {"LARGE_ID": large_id}) == large_id
    assert botcore.required_env("QUOTED_SECRET", {"QUOTED_SECRET": "'sec#ret' # comment"}) == "sec#ret"

    # 3. load_ini_config cache differentiates inline_comments=True vs inline_comments=False
    ini_file = tmp_path / "sample.conf"
    ini_file.write_text("[sec]\nurl = https://example.com/a;b # comment\n", encoding="utf-8")
    cp_inline = botcore.load_ini_config(ini_file, inline_comments=True)
    cp_raw = botcore.load_ini_config(ini_file, inline_comments=False)
    assert cp_inline.get("sec", "url") == "https://example.com/a;b"
    assert cp_raw.get("sec", "url") == "https://example.com/a;b # comment"

    # 4. _extract_sorted_ts_px and compute_24h_price_metrics handle unhashable/malformed boundary elements safely
    series_with_malformed = [
        {"malformed": 1},
        [1700000000000.0, 100.0],
        [1700000900000.0, 105.0],
        [999],
    ]
    pts = _extract_sorted_ts_px(series_with_malformed)
    assert pts == [(1700000000000.0, 100.0), (1700000900000.0, 105.0)]
    m24 = compute_24h_price_metrics(series_with_malformed)
    assert m24.get("chg_15m_pct") == pytest.approx(5.0)

    # 5. sync_sibling_symbol_telemetry does not leak USD hvn2_px/lvn_px into EUR siblings,
    #    and reconcile_sibling_flow_and_liq_regimes does not treat 0.0 and negative CVD as same-sign
    prices_and_trends = {
        "BTCUSDC": {
            "price": 90000.0,
            "poc_px": 89500.0,
            "sec_poc_px": 90500.0,
            "hvn2_px": 90500.0,
            "lvn_px": 89900.0,
            "profile_shape": "BIMODAL",
            "vah_px": 90800.0,
            "val_px": 89200.0,
        },
        "BTCEUR": {
            "price": 83000.0,
        },
    }
    micro = {
        "BTCUSDC": {
            "ob_mid_price": 90000.0,
            "bid_depth_usd": 500000.0,
            "ask_depth_usd": 500000.0,
            "cvd_15m_usd": 0.0,
            "cvd_regime": "balanced_flow",
            "spot_perp_flow_regime": "BALANCED_CROSS_VENUE_FLOW",
        },
        "BTCUSDT": {
            "ob_mid_price": 90010.0,
            "bid_depth_usd": 100000.0,
            "ask_depth_usd": 100000.0,
            "cvd_15m_usd": -250000.0,
            "cvd_regime": "aggressive_market_dumps",
            "spot_perp_flow_regime": "SYNCHRONIZED_SPOT_PERP_SELLING",
        },
    }
    sync_sibling_symbol_telemetry(prices_and_trends, {})
    assert "hvn2_px" not in prices_and_trends["BTCEUR"]
    assert "lvn_px" not in prices_and_trends["BTCEUR"]

    reconcile_sibling_flow_and_liq_regimes(micro, prices_and_trends)
    assert micro["BTCUSDT"]["cvd_regime"] == "aggressive_market_dumps"

    # 6. load_symbol_price_series overlays scalar list current_price_items without dropping tick
    p24_file = tmp_path / "cache_24_price_BTCUSDC.json"
    p24_file.write_text(
        json.dumps({"items": {"BTCUSDC": [[1700000000000.0, 100.0], [1700000060000.0, 101.0]]}}),
        encoding="utf-8",
    )
    s24, _ = load_symbol_price_series(str(tmp_path), "BTCUSDC", current_price_items={"BTCUSDC": [101.5, 102.0]})
    assert s24[-1][1] == pytest.approx(102.0)

    # 7. env_common.sh env_get parses 'export KEY = val' with spaces
    env_dir = tmp_path / "env_sh_test"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "config.env").write_text('export TEST_SPACED_KEY = "hello_world" # inline\n', encoding="utf-8")
    sh_Script = f'REPO_ROOT="{env_dir}"; ROOT="{env_dir}"; source orchestratorOS/lib/env_common.sh && env_get TEST_SPACED_KEY'
    res = subprocess.run(["bash", "-c", sh_Script], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "hello_world"


def test_iter72_series_24h_double_ms_order_ts_iso_and_portfolio_enrichment(tmp_path):
    import json
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series
    from intelligence.sentiment.collectors.portfolio_collector import (
        _load_audit_terminal_orders,
        _load_open_order_timestamps,
        harvest_bot_runtime_states,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context

    # 1. load_symbol_price_series does not double-multiply synthetic (<1e8) timestamps from archive_pts or scalar overlay
    (tmp_path / "cache_24_price_SYNTHUSDC.json").write_text(
        json.dumps({"items": {"SYNTHUSDC": [[150.0, 100.0], [160.0, 101.0]]}}),
        encoding="utf-8",
    )
    (tmp_path / "cache_365_price_SYNTHUSDC.json").write_text(
        json.dumps({"items": {"SYNTHUSDC": [[100.0, 99.0], [140.0, 99.5]]}}),
        encoding="utf-8",
    )
    s24, _ = load_symbol_price_series(str(tmp_path), "SYNTHUSDC", current_price_items={"SYNTHUSDC": 102.0})
    assert [pt[0] for pt in s24] == [100000.0, 140000.0, 150000.0, 160000.0, 161000.0]
    assert [pt[1] for pt in s24] == [99.0, 99.5, 100.0, 101.0, 102.0]

    # 2. _load_open_order_timestamps & _load_audit_terminal_orders & harvest_bot_runtime_states normalize ISO and ms timestamps
    (tmp_path / "autonomous_trade_intents_state.json").write_text(
        json.dumps(
            {
                "intents": [
                    {
                        "intent_id": "int-iso",
                        "order_id": "oid-iso",
                        "executed_at": "2023-11-14T22:13:20+00:00",
                        "timestamp": 1700000000000,
                    },
                    {
                        "intent_id": "int-ms",
                        "order_id": "oid-ms",
                        "timestamp": 1700000050000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_file = tmp_path / "execution_audit_20231114.jsonl"
    audit_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "submit_accepted",
                        "venue": "binance",
                        "symbol": "BTCUSDC",
                        "order_id": "oid-audit-ms",
                        "ts": 1700000100000,
                    }
                ),
                json.dumps(
                    {
                        "event": "order_terminal",
                        "venue": "binance",
                        "symbol": "BTCUSDC",
                        "order_id": "oid-audit-ms",
                        "side": "BUY",
                        "filled_qty": 0.1,
                        "price": 50000.0,
                        "ts": "2023-11-14T22:15:05+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ts_map = _load_open_order_timestamps(
        str(tmp_path),
        {"open_orders:binance": [{"order_id": "oid-prev-ms", "created_ts": 1700000020000}]},
    )
    assert ts_map["oid-iso"] == pytest.approx(1700000000.0)
    assert ts_map["oid-ms"] == pytest.approx(1700000050.0)
    assert ts_map["oid-prev-ms"] == pytest.approx(1700000020.0)
    assert ts_map["oid-audit-ms"] == pytest.approx(1700000100.0)

    term_orders = _load_audit_terminal_orders(str(tmp_path), "binance")
    assert term_orders["BTCUSDC"][0]["timestamp"] == 1700000100000

    (tmp_path / "order_retry_queue.jsonl").write_text(
        json.dumps(
            {
                "provider_name": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "requested_price": 49000.0,
                "lifecycle": "submit_pending",
                "expires_at_ts": 1700000050000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rt_expired = harvest_bot_runtime_states(str(tmp_path), now_ts=1700000100000)
    assert "active_retry_queue" not in rt_expired
    rt_active = harvest_bot_runtime_states(str(tmp_path), now_ts=1700000010000)
    assert len(rt_active.get("active_retry_queue", [])) == 1

    # 3. enrich_positions_with_orders_and_cycles preserves custom fee/limit fields when inst_obj is None
    #    and clears pos_peak/dd_from_peak_pct/pnl_vs_avg_pct on flat cycles
    pos_list = [
        {
            "venue": "kraken",
            "symbol": "BTCUSD",
            "price": 50000.0,
            "free_base": 0.0,
            "free_quote": 1000.0,
            "value_usd": 0.0,
            "fee_tier": "VIP_CUSTOM",
            "maker_fee_bps": 0.0,
            "taker_fee_bps": 5.0,
            "maker_rebate_edge_bps": 5.0,
            "exchange_min_notional_usd": 11.0,
            "min_qty": 0.0005,
            "step_size": 0.0005,
            "tick_size": 0.1,
        }
    ]
    kraken_cycles = {
        "BTCUSD": {
            "qty": 0.0,
            "cost_usd": 500.0,
            "avg_cost_px": 48000.0,
            "pos_peak": 52000.0,
            "dd_from_peak_pct": -3.85,
        }
    }
    enrich_positions_with_orders_and_cycles(
        pos_list,
        prices_and_trends={"BTCUSD": {"price": 50000.0, "spread_bps": 1.2}},
        inst_by_sym={},
        venue_recent_orders={},
        recent_orders={},
        kraken_cycles=kraken_cycles,
        hyperliquid_cycles={},
        t212_cycles={},
        cyc_sell_ts_map={},
    )
    assert pos_list[0]["fee_tier"] == "VIP_CUSTOM"
    assert pos_list[0]["maker_fee_bps"] == pytest.approx(0.0)
    assert pos_list[0]["taker_fee_bps"] == pytest.approx(5.0)
    assert pos_list[0]["exchange_min_notional_usd"] == pytest.approx(11.0)
    assert pos_list[0]["min_qty"] == pytest.approx(0.0005)
    assert kraken_cycles["BTCUSD"]["avg_cost_px"] is None
    assert kraken_cycles["BTCUSD"]["pnl_vs_avg_pct"] is None
    assert "pos_peak" not in kraken_cycles["BTCUSD"]
    assert "dd_from_peak_pct" not in kraken_cycles["BTCUSD"]

    # 4. build_cross_market_context excludes lowercase equity symbols from crypto_bellwethers
    #    and normalizes casefolded equity_bellwethers without non-BIMODAL profile_shape
    ctx = build_cross_market_context(
        prices_and_trends={
            "nvda_us_eq": {
                "price": 125.0,
                "chg_1h_pct": 1.5,
                "profile_shape": "NORMAL",
                "low_24h": 120.0,
                "high_24h": 126.0,
                "low_7d": 120.0,
                "high_7d": 126.0,
                "chg_7d_pct": 2.0,
            },
            "BTCUSDC": {"price": 90000.0, "chg_1h_pct": 0.8},
        },
        microstructure={},
        all_t212_cycles={"NVDA_US_EQ": {"qty": 2.0, "avg_cost_px": 120.0}},
    )
    assert "NVDA" not in ctx.get("crypto_bellwethers", {})
    assert list(ctx.get("equity_bellwethers", {}).keys()) == ["NVDA_US_EQ"]
    eq_nvda = ctx["equity_bellwethers"]["NVDA_US_EQ"]
    assert eq_nvda["price"] == pytest.approx(125.0)
    assert "profile_shape" not in eq_nvda
    assert "chg_7d_pct" not in eq_nvda


def test_iter73_warmup_shadow_prices_cost_basis_fill_id_and_ohlc_window():
    from collections import deque
    from intelligence.external.collectors.base_collector import _clean_snapshot_ts
    from providers.order_ohlc_utils import build_price_history_points, resolve_ohlc_time_window_ms
    from providers.quantity import compute_fills_cost_basis
    from strategies.base_spot_mixin import BaseSpotMixin
    from strategies.spot_rules import _iter_valid_shadow_prices

    # 1. _iter_valid_shadow_prices and BaseSpotMixin._shadow_vol_1h preserve negative warmup timestamps
    warmup_pts = [(-float((20 - i) * 60), 100.0 + (i % 3) * 0.5) for i in range(20)]
    valid_pts = _iter_valid_shadow_prices(warmup_pts)
    assert len(valid_pts) == 20
    assert valid_pts[0][0] == pytest.approx(-1200.0)
    assert valid_pts[-1][0] == pytest.approx(-60.0)

    class _DummySpot(BaseSpotMixin):
        replay_mode = True

    dummy_replay = _DummySpot()
    dummy_replay._shadow_prices = deque(warmup_pts, maxlen=120)
    vol_replay = dummy_replay._shadow_vol_1h()
    assert vol_replay is not None and vol_replay > 0.0

    dummy_live = _DummySpot()
    dummy_live.replay_mode = False
    dummy_live._shadow_prices = deque(warmup_pts, maxlen=120)
    vol_live = dummy_live._shadow_vol_1h()
    assert vol_live == pytest.approx(vol_replay)

    # 2. _clean_snapshot_ts supports ISO-8601 strings and millisecond timestamps
    assert _clean_snapshot_ts({"ts": "2023-11-14T22:13:20Z"}) == pytest.approx(1700000000.0)
    assert _clean_snapshot_ts({"timestamp": 1700000000000}) == pytest.approx(1700000000.0)

    # 3. compute_fills_cost_basis handles rows without explicit ID (idx=0 fallback >= 1) and rows with {"id": None, "tid": ...}
    cost_no_id = compute_fills_cost_basis(
        [{"time": 1700000000, "price": 100.0, "qty": 2.0, "side": "BUY"}],
        held_qty=2.0,
        since_s=1e9,
    )
    assert cost_no_id == pytest.approx(100.0)

    cost_null_id_with_tid = compute_fills_cost_basis(
        [{"id": None, "tid": "T-1", "time": 1700000000, "price": 50.0, "qty": 3.0, "side": "BUY"}],
        held_qty=3.0,
        since_s=1e9,
    )
    assert cost_null_id_with_tid == pytest.approx(50.0)

    # 4. build_price_history_points rounds seconds*1000 and resolve_ohlc_time_window_ms supports fractional lookback_days
    pts = build_price_history_points(
        [{"ts": 1700000000.123, "px": 42.0}],
        lambda r: r["ts"],
        lambda r: r["px"],
        ts_in_seconds=True,
    )
    assert pts == [{"timestamp": 1700000000123, "price": 42.0}]

    _, start_ms, end_ms, _ = resolve_ohlc_time_window_ms(end_ms=1700086400000, lookback_days=0.5)
    assert end_ms - start_ms == 43200 * 1000


def test_iter74_resolve_patched_attr_qlimit2_sub_bypasses_and_ssot_fills_freshest(tmp_path):
    import json
    import sys
    import time
    import types
    import botcore
    from guard_hierarchy import resolve_qlimit2_sub_bypasses
    from intelligence.macro_sentiment_ssot import has_new_acute_shock_headline
    from providers.kraken_provider import KrakenProvider

    # 1. resolve_patched_attr treats local_val / module attr as patched when defaults=(orig_fn,) is passed
    #    even if default_val is also set to the monkeypatched local_val
    orig_fn = lambda: "orig"
    patched_local = lambda: "patched_local"
    other_mod_fn = lambda: "other_mod"
    mod_name = "_dummy_iter74_mod"
    dummy_mod = types.ModuleType(mod_name)
    dummy_mod.fn = other_mod_fn
    sys.modules[mod_name] = dummy_mod
    try:
        resolved = botcore.resolve_patched_attr(
            "fn",
            patched_local,
            local_val=patched_local,
            defaults=(orig_fn,),
            modules=(mod_name,),
            require_callable=True,
        )
        assert resolved is patched_local
    finally:
        sys.modules.pop(mod_name, None)

    # 2. resolve_qlimit2_sub_bypasses recognizes legacy sub-tokens and mixed parabolic+weibull bypass sets
    _, p_only, w_only = resolve_qlimit2_sub_bypasses({"parabolic_guard"})
    assert p_only is True and w_only is False
    _, p_only_w, w_only_w = resolve_qlimit2_sub_bypasses({"exhaustion_guard"})
    assert p_only_w is False and w_only_w is True
    _, p_both, w_both = resolve_qlimit2_sub_bypasses({"parabolic_surge", "exhaustion_guard"})
    assert p_both is False and w_both is False

    # 3. has_new_acute_shock_headline with empty last_digest requires an actual acute phrase match
    assert has_new_acute_shock_headline([{"title": "Routine market update"}], "", ("strait closure",)) is False
    assert has_new_acute_shock_headline([{"title": "Sudden strait closure reported"}], "", ("strait closure",)) is True

    # 4. read_ssot_fills_cache selects the freshest matching alias key and invalidate_ssot_fills_cache clears case-variant fetchtime
    kp = KrakenProvider(client=types.SimpleNamespace())
    fills_path = tmp_path / "kraken_fills.json"
    now_ms = int(time.time() * 1000)
    fills_path.write_text(
        json.dumps(
            {
                "items": {
                    "XBTUSD": [{"id": "old-1", "price": 90000.0}],
                    "BTCUSD": [{"id": "new-2", "price": 91000.0}],
                },
                "fetchtime": {
                    "XBTUSD": now_ms - 30_000,
                    "btcusd": now_ms - 1_000,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = kp.read_ssot_fills_cache("BTCUSD", str(fills_path), max_stale_sec=90.0)
    assert rows == [{"id": "new-2", "price": 91000.0}]

    kp.invalidate_ssot_fills_cache("BTCUSD", str(fills_path))
    assert kp.read_ssot_fills_cache("BTCUSD", str(fills_path), max_stale_sec=90.0) is None


def test_iter75_order_guard_qlimit2_sub_bypasses_context_scale_and_cachetrend_items(tmp_path):
    import os
    import time
    import types
    import cachetrend
    import order_guard
    from guard_hierarchy import make_intel_cache_key
    from state_io import atomic_snapshot_file

    # 1. make_intel_cache_key distinguishes parabolic-only, weibull-only, and full G_QLIMIT2 bypasses
    k_para = make_intel_cache_key("BUY", {"parabolic_surge"})
    k_weib = make_intel_cache_key("BUY", {"weibull_exhaustion"})
    k_both = make_intel_cache_key("BUY", {"G_QLIMIT2"})
    assert k_para != k_weib
    assert k_para != k_both
    assert k_weib != k_both

    # 2. _normalize_intel_bypass_with_qlimit2_subs retains check_qlimit2_order_guards when only one sub-guard is bypassed
    norm, p_only, w_only = order_guard._normalize_intel_bypass_with_qlimit2_subs({"parabolic_surge"})
    assert p_only is True and w_only is False and "parabolic_surge" in norm and "weibull_exhaustion" not in norm
    norm_all, p_all, w_all = order_guard._normalize_intel_bypass_with_qlimit2_subs({"G_QLIMIT2"})
    assert p_all is False and w_all is False and "g_qlimit2" in norm_all

    # 3. _record_context_suggested_scale handles string / percent / NaN existing scales without TypeError
    ctx = types.SimpleNamespace(
        suggested_scale="80%",
        _suggested_scale_by_side={"BUY": float("nan")},
    )
    order_guard._record_context_suggested_scale(ctx, "BUY", 0.5, "test_scale", only_if_smaller=True)
    assert ctx.suggested_scale == 0.5
    assert ctx._suggested_scale_by_side["BUY"] == 0.5

    # 4. cachetrend._find_by_symbol_variant searches wrapped {"items": {...}} dictionaries
    found = cachetrend._find_by_symbol_variant({"items": {"BTCUSDT": [1, 2, 3]}}, "BTCUSDT")
    assert found == ("BTCUSDT", [1, 2, 3])

    # 5. atomic_snapshot_file only cleans up stale (>60s) leftover staging directories, preserving active ones
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    dest = tmp_path / "snap.json"
    active_dir = tmp_path / ".snap.json.999999.active"
    stale_dir = tmp_path / ".snap.json.888888.stale"
    active_dir.mkdir()
    stale_dir.mkdir()
    old_ts = time.time() - 120.0
    os.utime(stale_dir, (old_ts, old_ts))

    atomic_snapshot_file(str(src), str(dest))
    assert dest.exists()
    assert active_dir.exists()
    assert not stale_dir.exists()


def test_iter76_execution_audit_intent_ids_scope_filter_and_self_healing_loop_shutdown():
    import threading
    import types
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        apply_asset_scope_filter,
        enrich_sentiment_advisor_live_flow_divergence,
        reconcile_live_venue_equity_summary,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        compute_enterprise_portfolio_risk_metrics,
    )
    from providers.execution_audit import (
        _intent_order_ids,
        find_duplicate_ai_open_limit_order,
    )
    from strategies.base_spot_mixin import BaseSpotMixin

    # 1. _intent_order_ids falls back to payload orderId when history["order_id"] is sentinel "0"
    ids = _intent_order_ids(
        {
            "id": "intent_1",
            "history": [{"order_id": "0", "payload": {"orderId": "987654"}}],
        }
    )
    assert "987654" in ids

    # 2. find_duplicate_ai_open_limit_order unwraps {"intents": [...]} for notional matching
    inst = types.SimpleNamespace(
        open_orders=lambda: [
            {
                "orderId": "987654",
                "side": "BUY",
                "price": "100.0",
                "origQty": "0.5",
                "executedQty": "0.0",
            }
        ]
    )
    dup_id = find_duplicate_ai_open_limit_order(
        inst,
        "BUY",
        100.0,
        100.0,
        intents={
            "intents": [
                {
                    "id": "intent_1",
                    "suggested_notional_usd": 100.0,
                    "clamped_notional_usd": 50.0,
                    "execution_payload": {"orderId": "987654"},
                }
            ]
        },
        require_notional_match=True,
    )
    assert dup_id == "987654"

    # 3. apply_asset_scope_filter("equities") removes live_binance_equity_usdc so equities risk metrics exclude Binance
    portfolio_summary = {
        "tradable_positions": [{"venue": "t212", "symbol": "NVDA_US_EQ", "value_usd": 500.0}],
        "venue_quote_cash": {"t212:USD": {"venue": "t212", "free": "invalid", "free_usd": 200.0}},
        "live_binance_equity_usdc": 3000.0,
        "live_hl_spot_equity_usd": 400.0,
    }
    apply_asset_scope_filter({}, {}, portfolio_summary, "equities", {}, {"NVDA_US_EQ": {}})
    assert "live_binance_equity_usdc" not in portfolio_summary
    assert "live_hl_spot_equity_usd" not in portfolio_summary
    risk = compute_enterprise_portfolio_risk_metrics(
        portfolio_summary,
        {"NVDA_US_EQ": {"corr_btc_24h": 45.0}},
        {},
    )
    assert "binance" not in risk.get("venue_nav_breakdown_usd", {})
    assert risk["corr_vs_btc_24h"]["NVDA"] == 0.45

    # 4. reconcile_live_venue_equity_summary falls back to free_usd when free is non-numeric
    pf_bn = {
        "tradable_positions": [],
        "venue_quote_cash": {"binance:USDC": {"venue": "binance", "free": "n/a", "free_usd": 125.5}},
    }
    reconcile_live_venue_equity_summary(pf_bn)
    assert pf_bn["live_binance_equity_usdc"] == 125.5

    # 5. enrich_sentiment_advisor_live_flow_divergence clears stale divergence and falls back to 1h taker ratio
    sa = {
        "age_min": 20.0,
        "recommended_action": "HOLD",
        "directional_bias": "NEUTRAL",
        "summary": "BTC taker_ratio=0.70 shows selling pressure",
        "live_flow_divergence": "STALE_DIV",
    }
    enrich_sentiment_advisor_live_flow_divergence(sa, {"BTCUSDC": {"taker_ratio_1h": 1.05}})
    assert "LIVE_FLOW_RECOVERY" in sa.get("live_flow_divergence", "")
    sa["age_min"] = 2.0
    enrich_sentiment_advisor_live_flow_divergence(sa, {"BTCUSDC": {"taker_ratio_1h": 1.05}})
    assert "live_flow_divergence" not in sa

    # 6. BaseSpotMixin._run_self_healing_loop flushes dirty state on stop when _state_save_failed is True
    saved_calls = []

    class _DummySpot(BaseSpotMixin):
        def __init__(self):
            self.p = types.SimpleNamespace(check_minutes=0.01)
            self._state_save_failed = True

        def _mod_global(self, name, default):
            return default

        def _touch_strategy_heartbeat(self, *args, **kwargs):
            pass

        def _save(self):
            saved_calls.append(True)
            self._state_save_failed = False

    stop_ev = threading.Event()
    stop_ev.set()
    _DummySpot()._run_strategy_loop(price_fn=lambda: 100.0, status_log_fn=lambda _: None, stop=stop_ev)
    assert saved_calls == [True]


def test_iter77_trailing_flat_skip_worker_ts_and_monitortrades_pending_exit(tmp_path, monkeypatch):
    import monitororder
    import monitortrades
    import order_retry_worker as orw
    from trailing_core import TrailingCore

    # 1. order_retry_worker._format_ts normalizes millisecond and ISO-8601 timestamps
    assert orw._format_ts(1700000000000) != "N/A"
    assert orw._format_ts("2025-01-15T12:00:00Z") != "N/A"

    # 2. _PersistentQuarantineSet._path uses resolve_cachedb_filepath(QUEUE_FILE)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
    qset = orw._PersistentQuarantineSet("test_key")
    assert os.path.dirname(qset._path()) == str(tmp_path)

    # 3. TrailingCore does not spam trend deferral or skip finalizing exit_sold_qty when flat at price <= stop_at
    skip_calls = []

    class _FlatAdapter:
        def price(self, pair):
            return 90.0

        def free_qty(self, asset):
            return 0.0

        def total_qty(self, asset):
            return 0.0

        def trend(self, pair):
            return 1.0

        def log_skip_sell_trend(self, *args):
            skip_calls.append(args)

    tc = TrailingCore(
        _FlatAdapter(),
        log=lambda _: None,
        enabled=True,
        state_file=str(tmp_path / "trail_state.json"),
        min_notional=10.0,
        rebuy_enabled=True,
        rebuy_bounce_pct=2.0,
        rebuy_skip_if_trend_down=False,
        sell_skip_if_trend_up=True,
    )
    state = {
        "BTC": {
            "peak": 100.0,
            "is_flat": False,
            "exit_sold_qty": 0.5,
            "exit_sold_cost": 48.0,
        }
    }
    tc._process("BTC", "BTC", "BTCUSDC", 5.0, state)
    assert skip_calls == []
    assert state["BTC"]["is_flat"] is True
    assert state["BTC"]["rebuy"]["qty"] == 0.5
    assert state["BTC"]["rebuy"]["sell_price"] == 96.0
    assert "exit_sold_qty" not in state["BTC"]

    # 4. monitororder.reset_monitororder_state_for_test clears initial_prices
    monitororder.initial_prices["123"] = 100.0
    monitororder.reset_monitororder_state_for_test()
    assert len(monitororder.initial_prices) == 0

    # 5. monitortrades._restore_pending_exit_state restores _hard_tp_last when a pending Hard-TP order is canceled
    monitortrades._hard_tp_last["BTCUSDC"] = 1000.0
    monitortrades._record_pending_exit(
        "BTCUSDC",
        None,
        "TP",
        2000.0,
        0.5,
        110.0,
        {"reason": "timeout"},
        default_intent_id="mt_htp_BTCUSDC_2000",
        prev_hard_tp_last=1000.0,
        restore_hard_tp_last=True,
    )
    monitortrades._hard_tp_last["BTCUSDC"] = 2000.0
    monitortrades._restore_pending_exit_state("BTCUSDC", monitortrades._PENDING_STOP_EXIT["BTCUSDC"])
    assert monitortrades._hard_tp_last["BTCUSDC"] == 1000.0
    monitortrades._PENDING_STOP_EXIT.pop("BTCUSDC", None)
    monitortrades._hard_tp_last.pop("BTCUSDC", None)


def test_iter78_tradeall_small_big_conflict_rtrade_orphan_cancel_and_quantity_scale(monkeypatch):
    import types
    from providers.quantity import decide_quantity
    import rtrade
    import tradeall as ta

    # 1. tradeall.handle_symbol disables BIG firing when logic_small already placed an order on the same tick
    logic_calls = []
    monkeypatch.setattr(ta, "logic_small", lambda *args, **kwargs: [1001])
    monkeypatch.setattr(ta, "logic", lambda win, enable, *args, **kwargs: logic_calls.append((win, enable)))
    dummy_pw = types.SimpleNamespace(
        set_sample_rate=lambda _: None,
        get_short_trend=lambda: (1.0, 0.5, 1.2, 0.8),
        get_noise_epsilon=lambda: 0.1,
        prices=[100.0, 101.0],
        window_seconds=222.0,
        sample_rate_sec=1.0,
    )
    dummy_an = types.SimpleNamespace(check_price_change=lambda _: (4.0, 1))
    ts1 = ta.TrendState(3600, 300, 300)
    ts2 = ta.TrendState(3600, 300, 300)
    ta.handle_symbol("BTCUSDC", 101.0, dummy_pw, dummy_pw, dummy_an, dummy_an, ts1, ts2, regime_ctx="dummy")
    assert logic_calls == [("BIG", False)]

    # 2. tradeall.track_and_place_order retains order_id when mkt.cancel_order returns False
    resolved_ids = []
    monkeypatch.setattr(ta.mkt, "cancel_order", lambda sym, oid: False)
    monkeypatch.setattr(ta.order_retry, "resolve_accepted_order", lambda sym, oid: resolved_ids.append(oid))
    monkeypatch.setattr(ta.api, "cancel_expired_orders", lambda *args, **kwargs: None)
    monkeypatch.setattr(ta, "_fire_order", lambda *args, **kwargs: None)
    tracked = ["777"]
    res_ids = ta.track_and_place_order("BUY", "BTCUSDC", 1, 100.0, 100.0, order_ids=tracked)
    assert resolved_ids == []
    assert res_ids == ["777"]

    # 3. rtrade.TradingBot._cleanup_orphan_rt_orders continues canceling remaining orphans if one cancel raises
    bot = object.__new__(rtrade.TradingBot)
    bot.symbol = "TAOUSDC"
    canceled_rt = []

    def _mock_cancel(sym, oid):
        canceled_rt.append(oid)
        if oid == "91":
            raise RuntimeError("already filled -2011")

    open_seq = [
        [
            {"orderId": "91", "clientOrderId": "RT_orphan_1"},
            {"orderId": "92", "clientOrderId": "RT_orphan_2"},
        ],
        [],
    ]
    venue = types.SimpleNamespace(
        executor=types.SimpleNamespace(
            open_orders=lambda _sym: open_seq.pop(0),
            cancel_order=_mock_cancel,
        )
    )
    pair_store = types.SimpleNamespace(active=lambda _sym: [])
    blocked = bot._cleanup_orphan_rt_orders(venue, pair_store)
    assert blocked is False
    assert canceled_rt == ["91", "92"]

    # 4. decide_quantity supports percentage string scale and _suggested_scale_by_side["BUY"], and T212 partial sell min_notional
    class _DummyT212Prov:
        provider_name = "t212"

        def free_balance(self, asset):
            return 1000.0 if asset == "USD" else 0.10

        def policy_cap_quantity(self, symbol, side, price, requested, balance_cap, **kw):
            return requested

        def fee_cap_quantity(self, symbol, side, price, balance_cap):
            return balance_cap

        def round_amount(self, symbol, qty):
            return round(qty, 2)

        def order_filter_refusal(self, *args, **kwargs):
            return None

    prov = _DummyT212Prov()
    dec_buy = decide_quantity(
        prov,
        "NVDA_US_EQ",
        "BUY",
        100.0,
        2.0,
        regime_context={"_suggested_scale_by_side": {"BUY": "50%"}},
    )
    assert dec_buy.scale_applied is True
    assert dec_buy.applied_scale == 0.5
    assert dec_buy.final_qty == 1.0

    # Selling 0.06 out of 0.10 shares at $100 leaves $4 (<$6) and max_part ($4) < $6 min_notional -> sells full 0.10
    dec_sell = decide_quantity(prov, "NVDA_US_EQ", "SELL", 100.0, 0.06, enforce_business_minimum=True)
    assert dec_sell.final_qty == 0.10


def test_iter79_rtrade_pair_recovery_and_assetguardian_cancel(tmp_path, monkeypatch):
    import assetguardian as ag
    from rtrade_pair_store import RTradePairStore
    from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairCoordinator, PairPolicy

    # 1. PairCoordinator startup_recovery completes round when both BUY and SELL legs filled
    class _StubVenue:
        def __init__(self):
            self.statuses = {
                "L1": OrderSnapshot(status="closed", filled_qty=1.0, cost=99.36, fee=0.01),
                "L2": OrderSnapshot(status="closed", filled_qty=1.0, cost=100.64, fee=0.01),
            }
            self.recovery_blocked = False

        def current_price(self):
            return 100.0

        def order_status(self, oid):
            return self.statuses[oid]

        def cancel(self, oid):
            return True

        def recover_pair_intents(self, _pair_id):
            return [
                (OrderTicket(order_id="L2", side="SELL", price=100.64, qty=1.0), self.statuses["L2"], "limit"),
            ]

    venue = _StubVenue()
    policy = PairPolicy(adjustment_fraction=0.0064)
    coord = PairCoordinator(
        venue,
        qty=1.0,
        policy=policy,
        start_side="BUY",
        clock=lambda: 0.0,
        sleeper=lambda _s: None,
        pair_id_factory=lambda: "pair-79",
    )
    coord.pair_id = "pair-79"
    coord.phase = "startup_recovery"
    coord.reason = "sell_recovery_pending"
    coord.tickets = [OrderTicket(order_id="L1", side="BUY", price=99.36, qty=1.0, active=False, pair_id="pair-79")]
    coord.snapshots = {"L1": venue.statuses["L1"]}
    out = coord.step(now=2.0)
    assert out.phase == "complete"
    assert out.terminal is True
    assert out.reason is None
    assert out.first_fill_side == "BOTH"

    # 2. PairCoordinator._recover_persisted_intents does not prematurely set phase="stopping" while blocked
    venue2 = _StubVenue()
    venue2.statuses["S0"] = OrderSnapshot(status="canceled", filled_qty=0.4, cost=38.0, fee=0.01)
    venue2.recovery_blocked = True
    venue2.recover_pair_intents = lambda _pid: [
        (OrderTicket(order_id="S0", side="SELL", price=95.0, qty=1.0), venue2.statuses["S0"], "hard_stop"),
    ]
    coord2 = PairCoordinator(
        venue2,
        qty=1.0,
        policy=policy,
        start_side="BUY",
        clock=lambda: 0.0,
        sleeper=lambda _s: None,
        pair_id_factory=lambda: "pair-79b",
    )
    coord2.pair_id = "pair-79b"
    coord2.phase = "hard_stop_recovery"
    coord2.reason = "hard_stop_recovery_pending"
    coord2.tickets = [
        OrderTicket(order_id="L1", side="BUY", price=99.36, qty=1.0, active=False, pair_id="pair-79b"),
        OrderTicket(order_id="S0", side="SELL", price=95.0, qty=1.0, active=False, pair_id="pair-79b"),
    ]
    coord2.snapshots = {"L1": venue2.statuses["L1"], "S0": venue2.statuses["S0"]}
    coord2.stop_ticket = coord2.tickets[-1]
    out2 = coord2.step(now=3.0)
    assert out2.phase == "hard_stop_recovery"
    assert out2.reason == "hard_stop_recovery_pending"

    # 3. PairCoordinator.from_state does not fall back to limit ticket when explicit stop_order_id is missing from tickets
    state_stop = coord2.export_state()
    state_stop["phase"] = "stopping"
    state_stop["stop_order_id"] = "999999"
    state_stop["tickets"] = [vars(coord2.tickets[0]).copy()]
    restored = PairCoordinator.from_state(venue2, policy, state_stop)
    assert restored.stop_ticket is None

    # 4. RTradePairStore._resolved_lock_path uses resolve_cachedb_filepath
    (tmp_path / "state").mkdir(exist_ok=True)
    store = RTradePairStore(str(tmp_path / "rtrade_pairs_state.json"))
    assert store._resolved_lock_path() == str(tmp_path / "state" / "rtrade_pairs_state.json.lock")
    store.begin("TAOUSDC", "pair-store-79", "BUY", 1.0)
    assert len(store.active("TAOUSDC")) == 1

    # 5. assetguardian._resolve_blocking_sell_orders retains order when cancel returns False and resolves accepted order on True
    ag._rtrade_sell_block_since["TAOUSDC"] = 100.0
    monkeypatch.setattr(ag.time, "time", lambda: 100.0 + ag.RTRADE_BLOCK_CANCEL_SECONDS + 5.0)
    resolved_oids = []
    monkeypatch.setattr(ag, "resolve_accepted_order", lambda sym, oid: resolved_oids.append((sym, oid)))
    rtrade_sell = {"orderId": "777", "clientOrderId": "RT_777", "side": "SELL", "status": "NEW"}
    monkeypatch.setattr(ag.mkt, "cancel_order", lambda *_a, **_kw: False)
    rem_false = ag._resolve_blocking_sell_orders("TAOUSDC", [rtrade_sell])
    assert len(rem_false) == 1
    assert resolved_oids == []

    monkeypatch.setattr(ag.mkt, "cancel_order", lambda *_a, **_kw: True)
    rem_true = ag._resolve_blocking_sell_orders("TAOUSDC", [rtrade_sell])
    assert rem_true == []
    assert resolved_oids == [("TAOUSDC", "777")]



















