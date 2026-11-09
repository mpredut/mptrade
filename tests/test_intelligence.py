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
    assert len(collector._aggregated_pool) == 1
    loaded_h = next(iter(collector._aggregated_pool.values()))[1]
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
    from types import SimpleNamespace

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
    assert telem.get("Whale Taker Flow") == "Buyer/Seller ratio: 1.35"

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
    monkeypatch.setattr(order_guard, "get_cachedb_dir", lambda: str(tmp_path))

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

    # 5. atomic_snapshot_file only cleans up stale (>STALE_STAGING_MAX_AGE_SEC) leftover staging directories, preserving active ones
    from cachepaths import STALE_STAGING_MAX_AGE_SEC
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    dest = tmp_path / "snap.json"
    active_dir = tmp_path / ".snap.json.999999.active"
    stale_dir = tmp_path / ".snap.json.888888.stale"
    active_dir.mkdir()
    stale_dir.mkdir()
    old_ts = time.time() - (STALE_STAGING_MAX_AGE_SEC + 60.0)
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

    # 6. BaseSpotMixin._run_strategy_loop flushes dirty state on stop when _state_write_failed is True
    saved_calls = []

    class _DummySpot(BaseSpotMixin):
        def __init__(self):
            self.p = types.SimpleNamespace(check_minutes=0.01)
            self._state_write_failed = True

        def _mod_global(self, name, default):
            return default

        def _touch_strategy_heartbeat(self, *args, **kwargs):
            pass

        def _save(self):
            saved_calls.append(True)
            self._state_write_failed = False

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

    # 4. monitororder.reset_monitororder_state_for_test clears initial_buy_prices and initial_sell_prices
    monitororder.initial_buy_prices["123"] = 100.0
    monitororder.initial_sell_prices["456"] = 200.0
    monitororder.reset_monitororder_state_for_test()
    assert len(monitororder.initial_buy_prices) == 0
    assert len(monitororder.initial_sell_prices) == 0

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
    ts1 = ta.TrendState(300, 300)
    ts2 = ta.TrendState(300, 300)
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
    monkeypatch.setitem(ag._rtrade_sell_block_since, "TAOUSDC", 100.0)
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


def test_iter80_pricefetcher_refresh_and_kraken_trailing_unlock(monkeypatch, tmp_path):
    import threading
    from market_monitor.pricefetcher import PricePlatformInterface, CacheAllPriceFetcherManager
    from kraken.trailing_stop import KrakenTrailing
    import kraken.trailing_stop as kts_mod

    # 1. PricePlatformInterface.supports_symbol refreshes before reading _supported_symbols when _load_symbols rebinds the set
    class _RebindingPlatform(PricePlatformInterface):
        def __init__(self):
            self._supported_symbols = {"BTC"}
            self._last_refresh = 0.0
            self._refresh_interval = 10.0

        @property
        def platform_name(self) -> str:
            return "RebindTest"

        def _load_symbols(self):
            self._supported_symbols = {"BTC", "NEWTOKEN"}
            self._last_refresh = 9999999999.0

        def get_price(self, symbol: str):
            return 1.0

    plat = _RebindingPlatform()
    assert plat.supports_symbol("NEWTOKENUSDC") is True

    # 2. CacheAllPriceFetcherManager.remove_symbol removes symbols case-insensitively across active sets and metadata
    mgr = CacheAllPriceFetcherManager.__new__(CacheAllPriceFetcherManager)
    mgr.lock = threading.Lock()
    mgr.active_symbols = {"SPCX"}
    mgr.symbols = ["SPCX"]
    mgr.original_symbols = ["SPCX"]
    mgr.symbol_added_time = {"SPCX": 100.0}
    mgr.symbol_preferred_source = {"SPCX": "YahooFinance"}
    assert mgr.remove_symbol("spcx", reason="test") is True
    assert mgr.active_symbols == set()
    assert mgr.symbols == []
    assert mgr.original_symbols == []
    assert mgr.symbol_added_time == {}
    assert mgr.symbol_preferred_source == {}

    # 3. KrakenTrailing.unlock_qty continues cancelling remaining open SELL orders when one cancel raises or returns count=0
    class _MultiSellClient:
        def __init__(self):
            self.cancelled = []
            self.held = 10.0

        def _private(self, method, data=None):
            if method == "BalanceEx":
                return {"HYPE": {"balance": "10.0", "hold_trade": str(self.held)}}
            return {}

        def open_orders(self):
            return {
                "TX-FAIL": {"descr": {"type": "sell", "pair": "HYPE/USD"}},
                "TX-ZERO": {"descr": {"type": "sell", "pair": "HYPEUSD"}},
                "TX-OK": {"descr": {"type": "sell", "pair": "HYPE/USD"}},
            }

        def cancel_order(self, txid):
            self.cancelled.append(txid)
            if txid == "TX-FAIL":
                raise RuntimeError("transient cancel failure")
            if txid == "TX-ZERO":
                return {"count": 0}
            self.held = 0.0
            return {"count": 1}

    cli = _MultiSellClient()
    kt = KrakenTrailing(cli, log=lambda *_a: None, enabled=True, state_file=str(tmp_path / "kt.json"))
    assert kt.begin_tick() is True
    assert kt.free_qty("HYPE") == 0.0
    assert kt.unlock_qty("HYPEUSD") is True
    assert cli.cancelled == ["TX-FAIL", "TX-ZERO", "TX-OK"]
    assert kt.free_qty("HYPE") == pytest.approx(10.0)

    # 4. TrailingAdapterBase.trail_pct_for raises KeyError instead of ValueError or returning NaN on invalid DEFAULT_TRAIL_PCT
    monkeypatch.setattr(kts_mod, "DEFAULT_TRAIL_PCT", "invalid_float")
    with pytest.raises(KeyError):
        kt.trail_pct_for("UNKNOWN_SYMBOL_XYZ")


def test_iter81_order_retry_requested_qty_total_and_provider_null_fallbacks(tmp_path, monkeypatch):
    import order_lifecycle
    import order_retry
    from providers.base import MarketDataProvider
    from providers.strategy_executor import OrderStatus
    from providers.t212_provider import T212Provider

    # 1. order_lifecycle._terminal_payload & _status_from_terminal_payload preserve orig_qty
    st_term = OrderStatus("expired", filled_qty=4.0, cost=40.0, fee=0.04, venue_status="EXPIRED", orig_qty=10.0)
    payload = order_lifecycle._terminal_payload(st_term, 1700000000.0)
    assert payload["orig_qty"] == pytest.approx(10.0)
    restored = order_lifecycle._status_from_terminal_payload(payload)
    assert restored is not None
    assert restored.orig_qty == pytest.approx(10.0)

    # 2. order_retry preserves cumulative requested_qty_total across partial-fill remainder revisions
    #    and matches symbols canonically across queue helpers
    q_file = tmp_path / "order_retry_queue.jsonl"
    lock_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(order_retry, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(order_retry, "LOCK_FILE", str(lock_file))

    rec_id = order_retry.enqueue(
        "TAO/USDC",
        "BUY",
        qty=10.0,
        requested_price=10.0,
        ref_price=10.0,
        now=1000.0,
        place_kwargs={"kind": "assetguardian_tier"},
    )
    assert rec_id is not None
    assert order_retry.mark_accepted(rec_id, {"orderId": "101", "origQty": "10.0"}, now=1000.0) is True
    assert order_retry.get_accepted_order_ids_for_owner("taousdc", "assetguardian") == ["101"]

    claimed_r0 = order_retry.claim([rec_id], now=1020.0)[0]
    trans = order_retry.advance_claimed_status(claimed_r0, st_term, now=1020.0)
    assert trans.action == "retry_terminal"
    assert trans.remaining_qty == pytest.approx(6.0)

    rec_after_r0 = order_retry.get(rec_id)
    assert rec_after_r0["delivered_qty"] == pytest.approx(4.0)
    assert rec_after_r0["qty"] == pytest.approx(6.0)
    assert rec_after_r0["requested_qty_total"] == pytest.approx(10.0)

    # Revision 1 submitted and accepted for remaining 6.0 -> requested_qty_total must remain 4.0 + 6.0 = 10.0
    claimed_r1 = order_retry.claim([rec_id], now=1050.0)[0]
    assert order_retry.complete_claim(
        claimed_r1,
        "accepted",
        now=1050.0,
        order={"orderId": "102", "origQty": "6.0"},
    )
    rec_after_r1_accept = order_retry.get(rec_id)
    assert rec_after_r1_accept["qty"] == pytest.approx(6.0)
    assert rec_after_r1_accept["requested_qty_total"] == pytest.approx(10.0)

    # Observed status on revision 1 with orig_qty=6.0 -> requested_qty_total must remain 10.0
    claimed_r1_obs = order_retry.claim([rec_id], now=1070.0)[0]
    st_open_r1 = OrderStatus("open", filled_qty=1.0, cost=10.0, fee=0.01, venue_status="PARTIALLY_FILLED", orig_qty=6.0)
    order_retry.advance_claimed_status(claimed_r1_obs, st_open_r1, now=1070.0)
    rec_after_r1_obs = order_retry.get(rec_id)
    assert rec_after_r1_obs["qty"] == pytest.approx(6.0)
    assert rec_after_r1_obs["requested_qty_total"] == pytest.approx(10.0)

    # Canonical symbol matching on resolve_accepted_order
    assert order_retry.resolve_accepted_order("TAOUSDC", "102") is True
    assert order_retry.get(rec_id) is None

    # 3. T212Provider.open_orders and order_status fall back past explicit None fields (e.g. limitPrice=None, stopPrice=125.5)
    class _StubT212Client:
        def list_active_orders(self):
            return [
                {
                    "id": "555",
                    "ticker": "NVDA_US_EQ",
                    "status": "NEW",
                    "type": "STOP",
                    "orderedQuantity": None,
                    "quantity": -2.5,
                    "limitPrice": None,
                    "stopPrice": 125.5,
                    "filledQuantity": None,
                    "executedQty": 0.5,
                }
            ]

        def get_order_status(self, oid):
            return {
                "id": oid,
                "ticker": "NVDA_US_EQ",
                "status": "FILLED",
                "filledQuantity": 2.0,
                "filledValue": 0.0,
                "fillPrice": None,
                "averagePrice": 130.0,
                "orderedQuantity": None,
                "quantity": 2.0,
                "fee": 0.0,
            }

    t212 = T212Provider(client=_StubT212Client(), live_enabled=False)
    open_list = t212.open_orders("NVDA_US_EQ")
    assert len(open_list) == 1
    assert open_list[0]["side"] == "SELL"
    assert open_list[0]["price"] == pytest.approx(125.5)
    assert open_list[0]["origQty"] == pytest.approx(2.5)
    assert open_list[0]["executedQty"] == pytest.approx(0.5)

    st_t212 = t212.order_status("NVDA_US_EQ", "555")
    assert st_t212.status == "closed"
    assert st_t212.cost == pytest.approx(260.0)
    assert st_t212.orig_qty == pytest.approx(2.0)

    # 4. MarketDataProvider.normalize_orderbook_payload & _parse_ssot_price_point fall back past explicit None keys
    ob = MarketDataProvider.normalize_orderbook_payload(
        {
            "bids": [{"px": None, "price": 100.0, "sz": None, "qty": 1.5}],
            "asks": [{"px": None, "price": 101.0, "sz": None, "volume": 2.0}],
        }
    )
    assert ob == {"bids": [(100.0, 1.5)], "asks": [(101.0, 2.0)]}
    pt = MarketDataProvider._parse_ssot_price_point({"timestamp": 1700000000, "price": None, "px": 42.5})
    assert pt == (1700000000000, 42.5)


def test_iter82_late_intelligence_scale_retry_worker_kind_and_guard_hierarchy(monkeypatch):
    import instrument
    import order_retry_worker
    from guard_hierarchy import is_category_bypassed, normalize_bypassed_guards, revoke_guard_bypass
    import binance_api.bapi_allorders as bapi_allorders

    # 1. parse_scale_factor and _apply_late_intelligence_scale support percentage strings and _suggested_scale_by_side
    assert instrument.parse_scale_factor("25%", default=0.5) == pytest.approx(0.25)
    assert instrument.parse_scale_factor(0.4, default=0.5) == pytest.approx(0.4)

    class _DummyProvider:
        name = "binance"
        def round_quantity(self, symbol, qty):
            return round(float(qty), 4)

    inst = object.__new__(instrument.Instrument)
    inst.symbol = "BTCUSDC"
    inst._provider = _DummyProvider()
    monkeypatch.setattr(instrument, "check_order_filter_refusal", lambda *a, **kw: None)

    ctx = {"suggested_scale": None, "_suggested_scale_by_side": {"BUY": "25%"}}
    scaled_qty, new_scale, refusal = inst._apply_late_intelligence_scale(
        qty=2.0,
        price=50000.0,
        is_market=True,
        enforce_business_minimum=False,
        regime_context=ctx,
        applied_scale=1.0,
        side_u="BUY",
    )
    assert refusal is None
    assert scaled_qty == pytest.approx(0.5)
    assert new_scale == pytest.approx(0.25)

    # 2. order_retry_worker._prepare_retry_placement_kwargs honors monitor_order_replace in place_kwargs / compound kind
    rec = {
        "id": "rec-82",
        "symbol": "BTCUSDC",
        "side": "SELL",
        "kind": None,
        "requested_price": 51000.0,
        "created_ts": 1700000000.0,
        "place_kwargs": {"kind": "monitor_order_replace_sell"},
    }
    kw = {"kind": "monitor_order_replace_sell"}
    side_u, _age, _prog, stage = order_retry_worker._prepare_retry_placement_kwargs(rec, kw, "binance", 1700000100.0)
    assert side_u == "SELL"
    assert stage == "standard"
    assert kw.get("wait_for_trend") is True
    assert kw.get("allow_math_reference_sell") is True
    assert "G_PROF" in kw.get("bypass_guards", [])
    assert "math_reference" in kw.get("bypass_guards", [])

    recorded_audit = []
    monkeypatch.setattr(
        order_retry_worker._AUDIT,
        "record",
        lambda event, **fields: recorded_audit.append((event, fields)),
    )
    order_retry_worker._audit_event(rec, "test_event")
    assert len(recorded_audit) == 1
    assert recorded_audit[0][1]["kind"] == "monitor_order_replace_sell"

    # 3. guard_hierarchy.is_category_bypassed supports case-insensitivity and category shortcuts,
    #    and revoke_guard_bypass preserves canonical/alias tokens for remaining siblings
    b_op = normalize_bypassed_guards("go")
    assert is_category_bypassed("operational", b_op)
    assert is_category_bypassed("OPERATIONAL", b_op)
    assert is_category_bypassed("go", b_op)

    b_prof = {"profit_guard"}
    revoke_guard_bypass(b_prof, "G_QLIMIT1")
    assert "g_prof" in b_prof
    assert "math_reference" in b_prof
    assert "math_profit" in b_prof
    assert "g_qlimit1" not in b_prof

    # 4. bapi_allorders.get_total_traded_stats falls back to 'qty' when 'quantity' is explicitly None
    monkeypatch.setattr(
        bapi_allorders,
        "get_trade_orders",
        lambda **kw: [
            {"side": "BUY", "price": 100.0, "quantity": None, "qty": 1.5},
            {"side": "SELL", "price": 110.0, "quantity": 2.0, "qty": None},
        ],
    )
    stats = bapi_allorders.get_total_traded_stats("BTCUSDC", period_seconds=3600)
    assert stats["BUY"]["total_quantity"] == pytest.approx(1.5)
    assert stats["BUY"]["total_value"] == pytest.approx(150.0)
    assert stats["BUY"]["trade_count"] == 1
    assert stats["SELL"]["total_quantity"] == pytest.approx(2.0)
    assert stats["SELL"]["total_value"] == pytest.approx(220.0)
    assert stats["SELL"]["trade_count"] == 1


def test_iter83_mt_exit_kind_stop_variants_state_store_and_price_analysis_ms():
    from strategies.mt_position_accounting import _infer_mt_exit_kind, _update_mt_reentry_state
    from strategies.state_store import update_post_sell_anchors
    from strategies.position_exit_manager import PositionExitConfig, PositionExitManager, ReentryAction, ReentrySnapshot
    import order_retry
    import priceAnalysis

    # 1. _infer_mt_exit_kind handles AI_STOP_LOSS / SL / STOP_LOSS and empty trade_orders_sell safely
    exit_map = {}
    assert _infer_mt_exit_kind(
        "TAOUSDC",
        [{"kind": "AI_STOP_LOSS", "actor": "AI"}],
        position_closed_by_sell=True,
        buy_price=300.0,
        sell_price=280.0,
        buy_time=1000.0,
        sell_time=2000.0,
        position={},
        tp_ref="last",
        lost_threshold=0.05,
        last_exit_kind_map=exit_map,
    ) == "STOP"
    assert exit_map.get("TAOUSDC") == "STOP"

    exit_map2 = {}
    assert _infer_mt_exit_kind(
        "TAOUSDC",
        [],
        position_closed_by_sell=True,
        buy_price=300.0,
        sell_price=280.0,
        buy_time=1000.0,
        sell_time=2000.0,
        position={"last_cycle_buy_price": float("nan"), "active_average_buy_price": 300.0},
        tp_ref="average",
        lost_threshold=0.05,
        last_exit_kind_map=exit_map2,
    ) == "STOP"

    re_map = {}
    st_re = _update_mt_reentry_state("TAOUSDC", 280.0, float("nan"), 2000, 1990.0, re_map)
    assert st_re["trough"] == pytest.approx(280.0)
    assert st_re["peak"] == pytest.approx(280.0)

    # 2. update_post_sell_anchors ignores non-positive/NaN price and tracks sl_low for SL/AI_STOP_LOSS
    st_store = {"last_sell_price": 100.0, "last_exit_kind": "ai_stop_loss"}
    assert update_post_sell_anchors(st_store, 92.0, track_stop_sl_low=True) == pytest.approx(100.0)
    assert st_store["sl_low"] == pytest.approx(92.0)
    assert st_store["post_sell_trough"] == pytest.approx(92.0)
    # Invalid price (None / 0 / NaN) must not corrupt anchors
    assert update_post_sell_anchors(st_store, 0.0, track_stop_sl_low=True) == pytest.approx(100.0)
    assert update_post_sell_anchors(st_store, float("nan"), track_stop_sl_low=True) == pytest.approx(100.0)
    assert st_store["sl_low"] == pytest.approx(92.0)
    assert st_store["post_sell_trough"] == pytest.approx(92.0)

    # 3. PositionExitManager.evaluate_reentry treats AI_STOP_LOSS / SL as post-STOP bounce gate
    mgr = PositionExitManager(config=PositionExitConfig(reentry_sl_bounce_pct=0.05, reentry_drop_pct=0.02))
    snap_sl = ReentrySnapshot(
        symbol="TAOUSDC",
        current_price=93.0,
        last_sell_price=100.0,
        last_exit_kind="AI_STOP_LOSS",
        post_sell_trough=90.0,
        current_time_s=2000.0,
        last_sell_time_s=1900.0,
    )
    dec_sl = mgr.evaluate_reentry(snap_sl)
    assert dec_sl.action == ReentryAction.WAIT
    assert "post_stop_bounce_waiting" in dec_sl.reason

    # 4. order_retry._same_deferred_stream distinguishes different place_kwargs["kind"] when top-level kind is None
    left_rec = {
        "symbol": "TAOUSDC",
        "side": "SELL",
        "kind": None,
        "lifecycle": "submit_pending",
        "last_failure_reason": "trend_deferred",
        "place_kwargs": {"kind": "TP"},
    }
    right_rec = {
        "symbol": "TAOUSDC",
        "side": "SELL",
        "kind": None,
        "lifecycle": "submit_pending",
        "last_failure_reason": "trend_deferred",
        "place_kwargs": {"kind": "STOP"},
    }
    same_rec = {
        "symbol": "TAO/USDC",
        "side": "SELL",
        "kind": "TP",
        "lifecycle": "submit_pending",
        "last_failure_reason": "trend_deferred",
        "place_kwargs": {},
    }
    assert order_retry._same_deferred_stream(left_rec, right_rec) is False
    assert order_retry._same_deferred_stream(left_rec, same_rec) is True

    # 5. priceAnalysis.detect_long_term_trend and format_timestamp handle millisecond timestamps
    base_sec = 1700000000.0
    ts_ms = [(base_sec + i * 3600.0) * 1000.0 for i in range(60)]
    px_up = [100.0 + i * 1.5 for i in range(60)]
    trend_res = priceAnalysis.detect_long_term_trend(
        ts_ms, px_up, window_hours=24, step_hours=8, min_consecutive_blocks=3, mk_alpha=None
    )
    assert trend_res is not None
    assert trend_res["direction"] == "up"
    assert trend_res["duration_seconds"] >= 36 * 3600.0
    fmt_ms = priceAnalysis.format_timestamp(1700000000000.0)
    fmt_sec = priceAnalysis.format_timestamp(1700000000.0)
    assert fmt_ms == fmt_sec
    assert not fmt_ms.startswith("1970-")


def test_iter84_merged_module_alias_finder_signals_reload_and_pretrade_kalman(tmp_path, monkeypatch):
    import importlib
    import json
    import sys
    from types import SimpleNamespace
    import botcore
    from strategies.base_spot_mixin import BaseSpotMixin, SpotParamsMixin
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as ptc

    # 1. _MergedModuleAliasFinder resolves merged Package E module aliases even after sys.modules.pop
    for alias in (
        "accepted_order_persistence",
        "order_outcomes_log",
        "order_rollback",
        "shadow_runtime",
        "liquidation_signal",
        "keys.apikeys",
        "hyperliquid.price_analysis",
        "price_analysis",
    ):
        sys.modules.pop(alias, None)
    aop = importlib.import_module("accepted_order_persistence")
    assert hasattr(aop, "complete_accepted_claim")
    assert aop.__name__ == "order_lifecycle"

    ool = importlib.import_module("order_outcomes_log")
    assert hasattr(ool, "log_order_outcome")
    assert ool.__name__ == "order_lifecycle"

    sr = importlib.import_module("shadow_runtime")
    assert hasattr(sr, "require_shadow_interval")
    assert sr.__name__ == "botcore"

    # 2. credentials reload refreshes keys.apikeys shim while preserving keys._is_credentials_shim
    sys.modules.pop("keys.apikeys", None)
    import credentials
    importlib.reload(credentials)
    assert sys.modules.get("keys.apikeys") is credentials
    assert getattr(sys.modules.get("keys"), "_is_credentials_shim", False) is True

    # 3. hyperliquid.signals reload updates price_analysis alias and get_signal honors monkeypatched signals.signal
    import hyperliquid.signals as hl_sig
    hl_sig_reloaded = importlib.reload(hl_sig)
    assert sys.modules.get("price_analysis") is hl_sig_reloaded
    assert sys.modules.get("hyperliquid.price_analysis") is hl_sig_reloaded
    monkeypatch.setenv("SIGNAL_SOURCE", "analysis")
    monkeypatch.setattr(hl_sig_reloaded, "signal", lambda client, coin: {"trend": "down", "confidence": 0.9, "source": "patched_after_reload"})
    sig_res = hl_sig_reloaded.get_signal(object(), "ETH")
    assert sig_res["trend"] == "down"
    assert sig_res["source"] == "patched_after_reload"

    # 4. SpotParamsMixin handles bool/None/string fields safely and BaseSpotMixin works without dca_vol_scale_k
    class _DummyParams(SpotParamsMixin):
        def __init__(self):
            self.total_budget = True  # boolean must NOT be treated as positive float budget
            self.alloc_pct = "50.0"
            self.entry_pct = "10.0"
            self.dca_pct = "20.0"
            self.entry_amount = "25.0"
            self.dca_amount = None
            self.max_budget = None

    dp = _DummyParams()
    assert dp.pct_sizing_on() is False
    assert dp.allocated_budget() == pytest.approx(0.0)
    assert dp.effective_entry_amount() == pytest.approx(25.0)
    assert dp.effective_dca_amount_base() == pytest.approx(0.0)

    class _DummyStrat(BaseSpotMixin):
        def __init__(self):
            self.p = SimpleNamespace(
                total_budget=0.0,
                amount_usd=40.0,
                dca_amount_usd=0.0,
                dca_multiplier=None,
                effective_dca_amount_base=lambda: 40.0,
            )
            self.client = SimpleNamespace()  # no ohlc_closes method (like T212 client)

    ds = _DummyStrat()
    assert ds._dca_vol_1h() is None
    assert ds._trail_vol_1h() is None
    assert ds._effective_dca_amount() == pytest.approx(40.0)

    # 5. pretrade_telemetry_collector reads canonical cache_kalman_trend.json when cache_kalman_price.json is absent
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_kalman_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": {
                        "kalman_vel": 1.25,
                        "kalman_vel_std": 0.35,
                        "kalman_trend": "UP",
                        "vol_1h_pct": 2.1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "orderbook_depth_TAOUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSDC",
                "mid_price": 400.0,
                "best_bid": 399.9,
                "best_ask": 400.1,
                "spread_bps": 5.0,
                "bid_depth_usd": 100000.0,
                "ask_depth_usd": 100000.0,
                "imbalance_ratio": 0.56,
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "derivatives_telemetry_TAOUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSDC",
                "funding_rate": 0.0,
                "open_interest_usd": 10000000.0,
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "whale_snapshot_TAOUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSDC",
                "top_traders_long_pct": 0.50,
                "taker_buy_sell_ratio": 1.04,
            }
        ),
        encoding="utf-8",
    )
    telem = ptc.collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cache_dir))
    assert "SHORT liq=$" in telem.get("Liquidation Clusters (LONG vs SHORT)", "")


def test_iter85_package_h_alias_reload_inspect_prod_fleet_and_pretrade_eq_slopes(tmp_path, monkeypatch, capsys):
    import importlib
    import json
    import sys
    from types import SimpleNamespace
    import binance_api
    import instrument_registry
    import lock
    import lock.cooldown as lock_cooldown
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as ptc
    from verify_tools import inspect_prod_fleet

    # 1. Reloading host modules refreshes Package E/H aliases in sys.modules and parent packages
    ir_reloaded = importlib.reload(instrument_registry)
    assert sys.modules.get("instruments_config") is ir_reloaded

    lc_reloaded = importlib.reload(lock_cooldown)
    assert sys.modules.get("lock.file_lock") is lc_reloaded
    assert sys.modules.get("lock.trade_cooldown") is lc_reloaded
    assert lock.file_lock is lc_reloaded
    assert lock.trade_cooldown is lc_reloaded

    # Lazy __getattr__ on binance_api and lock resolves merged submodule aliases even after delattr/pop
    sys.modules.pop("binance_api.bapi_ws_health", None)
    binance_api.__dict__.pop("bapi_ws_health", None)
    ws_h = binance_api.bapi_ws_health
    assert ws_h.__name__ == "binance_api.bapi_ws_bridge"
    assert sys.modules.get("binance_api.bapi_ws_health") is ws_h

    # Monkeypatching instrument.Instrument is respected by _resolve_instrument_cls / bind_instruments
    import instrument as inst_mod
    spec = ir_reloaded.InstrumentSpec(
        name="BINANCE_BTC",
        provider="binance",
        symbol="BTCUSDC",
        base="BTC",
        quote="USDC",
        enabled=True,
        isolation="own_ledger",
        market_hours="24x7",
        params={},
    )
    fake_api = SimpleNamespace(provider_by_name=lambda _n: SimpleNamespace(name="binance", validate_symbol=lambda _s: None))
    bound = inst_mod.Instrument.from_spec(spec, api=fake_api)
    assert bound.spec is spec and bound.symbol == "BTCUSDC"
    sentinel_cls = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setattr(inst_mod, "Instrument", sentinel_cls, raising=False)
    assert inst_mod._resolve_instrument_cls() is sentinel_cls

    # 2. inspect_prod_fleet translates local workspace cachedb paths to /home/predut/mptrade on SSH fallback
    #    and tolerates non-dict / None fields without AttributeError
    local_Fake_path = str(
        tmp_path / "nonexistent_dir" / "cachedb" / "state" / "venue_balances_state.json"
    )
    assert (
        inspect_prod_fleet._to_prod_remote_path(local_Fake_path)
        == "/home/predut/mptrade/cachedb/state/venue_balances_state.json"
    )
    captured_cmds = []

    def _fake_run(cmd, capture_output=True, text=True):
        captured_cmds.append(cmd)
        if "venue_balances_state.json" in cmd[-1]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "cash:binance:USDC": {"free": "250.5", "locked_quote_usd": None},
                        "pos:t212:NVDA_US_EQ": {"value_usd": "100.0", "locked_base_usd": "24.5"},
                    }
                ),
            )
        if "llm_topic_threads.json" in cmd[-1]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"auto_cryptotrade_executor": {"rolling_history": ["not-a-dict"]}}),
            )
        return SimpleNamespace(returncode=0, stdout="non-json-string")

    monkeypatch.setattr(inspect_prod_fleet.subprocess, "run", _fake_run)
    monkeypatch.setattr(inspect_prod_fleet.os.path, "isfile", lambda p: False)
    inspect_prod_fleet.inspect_fleet(json_output=True)
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["total_free_cash_usd"] == pytest.approx(250.5)
    assert out_json["total_free_assets_usd"] == pytest.approx(100.0)
    assert out_json["total_locked_assets_usd"] == pytest.approx(24.5)
    assert out_json["unified_nav_usd"] == pytest.approx(375.0)
    assert out_json["topic_state"]["latest_summary"] == ""
    assert all("/home/predut/mptrade/cachedb/" in c[-1] for c in captured_cmds)

    # 3. collect_pretrade_telemetry prunes stale 3.7m slope on post-RTH equities without price override,
    #    and recomputes negative slopes/kalman when post-RTH price diverged by >0.5%
    eq_dir = tmp_path / "cachedb_eq_pt"
    eq_dir.mkdir(parents=True, exist_ok=True)
    old_base_ms = 1_700_000_000_000
    nvda_bars = [[old_base_ms - (30 - i) * 300_000, round(190.0 + i * 0.15, 4), 1000.0] for i in range(31)]
    (eq_dir / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": nvda_bars}}),
        encoding="utf-8",
    )
    telem_nvda = ptc.collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(eq_dir))
    assert "3.7m=" not in telem_nvda.get("Dynamic Multi-Window Slopes", "")
    assert "15m=" in telem_nvda.get("Dynamic Multi-Window Slopes", "")
    assert "2.5h=" in telem_nvda.get("Dynamic Multi-Window Slopes", "")

    # Now add a >0.5% lower post-RTH price in venue_balances_state.json + stale positive cache_short_trend.json
    (eq_dir / "cache_short_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "NVDA_US_EQ": {
                        "slope_3_7m": 0.0025,
                        "slope_15m": 0.0020,
                        "slope_2_5h": 0.0010,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (eq_dir / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:t212:NVDA_US_EQ": {
                    "symbol": "NVDA_US_EQ",
                    "free": 1.0,
                    "price": 185.0,
                    "ts": (old_base_ms / 1000.0) + 600.0,
                }
            }
        ),
        encoding="utf-8",
    )
    telem_nvda_div = ptc.collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(eq_dir))
    slopes_str = telem_nvda_div.get("Dynamic Multi-Window Slopes", "")
    assert "3.7m=-" in slopes_str
    assert "15m=-" in slopes_str


def test_iter11_cited_taker_buying_ratio_and_first_person_plural_advice_column_exclusion():
    from intelligence.macro.calendar_ssot import parse_cited_taker_ratio
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import enrich_sentiment_advisor_live_flow_divergence

    summary_ax = (
        "Risk assets and crypto are staging a strong intraday relief bounce with BTC reclaiming $77,600 (+2.58%) "
        "and altcoins outperforming as aggressive Asian session taker buying accelerates (2.3903 ratio, 5m ratio 4.2228, "
        "$124.2M buy vs $52.0M sell), while heavy Binance top-trader long positioning (66.05% long, 1.95 ratio) caps upside."
    )
    assert parse_cited_taker_ratio(summary_ax) == 2.3903
    sa_ax = {
        "recommended_action": "HOLD",
        "directional_bias": "NEUTRAL",
        "age_min": 24.8,
        "summary": summary_ax,
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_ax,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.0922, "cvd_15m_usd": 4_700_000.0}},
    )
    assert "LIVE_FLOW_FADE" in str(sa_ax.get("live_flow_divergence", ""))
    assert "2.39" in str(sa_ax.get("live_flow_divergence", ""))

    headline_ay = (
        "‘It seems bizarre’: We’re in our 80s with $950,000 in traditional IRAs. "
        "Why is our adviser pushing Roth conversions? - MarketWatch"
    )
    assert is_excluded_reference_news_item(headline_ay, "MarketWatch", "https://news.google.com/rss/articles/123") is True
    assert is_excluded_reference_news_item(
        "This year’s worst-performing S&P 500 stock is finally seeing some major relief",
        "MarketWatch.com - Top Stories",
        "https://www.marketwatch.com/story/this-years-worst-performing-s-p-500-stock-is-finally-seeing-some-major-relief-fff3fa85",
    ) is True
    assert is_excluded_reference_news_item(
        "A massive trade just happened in a leading mortgage lender",
        "US Top News and Analysis",
        "https://www.cnbc.com/2026/11/04/a-massive-trade-just-happened-in-a-leading-mortgage-lender-watch-options-action-live.html",
    ) is True
    assert is_excluded_reference_news_item(
        "Cramer: What must happen for Home Depot to keep its 25% monthlong rally going",
        "US Top News and Analysis",
        "https://www.cnbc.com/2026/11/04/cramer-home-depot.html",
    ) is True


def test_iter86_symbols_facade_config_reload_inspect_fleet_and_kalman_warmstart(
    tmp_path, monkeypatch
):
    import importlib
    import sys
    import botcore
    import credentials
    import instrument_registry
    import symbols
    import config

    # 1. symbols facade attributes (`forcesellsymbol`, `<base>symbol`), monkeypatch undo dynamic resolution,
    #    empty `symbols = []` override, and `_normalize_ordertype` extended tokens
    assert isinstance(symbols.forcesellsymbol, list) and "BTCUSDC" in symbols.forcesellsymbol
    assert symbols.btcsymbol == "BTCUSDC"
    assert symbols.taosymbol == "TAOUSDC"
    assert symbols._normalize_ordertype("A") == "SELL"
    assert symbols._normalize_ordertype("SOLD") == "SELL"
    assert symbols._normalize_ordertype("open_long") == "BUY"
    assert symbols._normalize_ordertype("close-long") == "SELL"

    with monkeypatch.context() as mp:
        mp.setattr(symbols, "symbols", [])
        with pytest.raises(ValueError):
            symbols.validate_symbols("BTCUSDC")
    # After monkeypatch undo, `symbols` is NOT frozen in instrument_registry.__dict__
    assert "symbols" not in instrument_registry.__dict__
    custom_conf = tmp_path / "instruments.conf"
    custom_conf.write_text(
        "[eth]\nprovider = binance\nsymbol = ETHUSDC\nbase = ETH\nquote = USDC\nenabled = true\n"
        "isolation = dedicated\nmarket_hours = 24x7\nrole.mt = false\nrole.tradeall_fire = false\n"
        "role.kalman_primary = false\nrole.trailing = false\nrole.assetguardian = false\n"
        "role.archive = false\nrole.rtrade = false\nrole.force_sell = true\n",
        encoding="utf-8",
    )
    with monkeypatch.context() as mp:
        mp.setattr(instrument_registry, "DEFAULT_PATH", custom_conf)
        assert symbols.symbols == ["ETHUSDC"]
        assert symbols.forcesellsymbol == ["ETHUSDC"]
        assert symbols.ethsymbol == "ETHUSDC"

    # 2. config.ROOT_DIR compatibility, botcore watcher preservation on reload, and importlib.reload(keys)
    assert hasattr(config, "ROOT_DIR") and os.path.isdir(config.ROOT_DIR)
    keys_mod = importlib.import_module("keys")
    reloaded_keys = importlib.reload(keys_mod)
    assert reloaded_keys.apikeys is sys.modules["credentials"]

    # 3. inspect_prod_fleet merges non-Binance open_orders:* with Binance cache_order.json fallback
    from verify_tools import inspect_prod_fleet

    fake_balances = {
        "cash:binance:USDC": {"free": 500.0, "locked_quote_usd": 100.0},
        "open_orders:kraken": [],
        "open_orders:hyperliquid:HYPE": [
            {"symbol": "HYPE", "side": "S", "limitPx": "25.5", "sz": "2.0", "oid": "hl-1"}
        ],
    }
    fake_orders = {
        "items": {
            "BTCUSDC": [
                {
                    "orderId": 999,
                    "symbol": "BTCUSDC",
                    "side": "B",
                    "price": "65000.0",
                    "origQty": "0.002",
                    "status": "NEW",
                }
            ]
        }
    }
    monkeypatch.setattr(
        inspect_prod_fleet,
        "ssh_cat",
        lambda p: (
            fake_balances
            if "venue_balances_state.json" in str(p)
            else (fake_orders if "cache_order.json" in str(p) else {})
        ),
    )
    fleet_res = inspect_prod_fleet.inspect_fleet(json_output=True)
    assert len(fleet_res["active_orders"]) == 2
    by_venue = {o["venue"]: o for o in fleet_res["active_orders"]}
    assert by_venue["hyperliquid"]["side"] == "SELL"
    assert by_venue["hyperliquid"]["price"] == pytest.approx(25.5)
    assert by_venue["hyperliquid"]["orig_qty"] == pytest.approx(2.0)
    assert by_venue["binance"]["symbol"] == "BTCUSDC"
    assert by_venue["binance"]["side"] == "BUY"
    assert by_venue["binance"]["price"] == pytest.approx(65000.0)

    # 4. KalmanStateTracker warm-start guards negative synthetic timestamps and short buffers
    from shadow_signals import KalmanStateTracker

    tracker = KalmanStateTracker(state_path=str(tmp_path / "kstate.json"), compute_vol=False)
    res_zero_ts = tracker.update(
        "BTCUSDC",
        price=105.0,
        ts=0.0,
        epsilon=0.01,
        big_prices=[100.0 + 0.1 * i for i in range(50)],
        big_sample_rate=1.0,
    )
    assert len(tracker._fed_prices["BTCUSDC"]) == 1
    assert res_zero_ts["kalman_trend"] == 0

    tracker2 = KalmanStateTracker(state_path=str(tmp_path / "kstate2.json"), compute_vol=False)
    res_short = tracker2.update(
        "ETHUSDC",
        price=2020.0,
        ts=1000.0,
        epsilon=0.1,
        big_prices=[2000.0, 2005.0, 2010.0, 2015.0],
        big_sample_rate=1.0,
    )
    assert len(tracker2._fed_prices["ETHUSDC"]) == 2
    assert res_short["kalman_vel"] > 0.0


def test_iter87_kalman_multiprocess_merge_execution_audit_cancel_ts_and_archive_subpkg(
    tmp_path, monkeypatch
):
    import importlib
    import json
    from intelligence.internal.state.survival import estimate_remaining_trend_hours
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        enrich_microstructure_cvd,
        reconcile_sibling_flow_and_liq_regimes,
    )
    from providers.execution_audit import (
        build_last_cycle_execution_status,
        extract_intent_notional,
        reconcile_intents_from_execution_audit,
    )
    from shadow_signals import KalmanStateTracker

    # 1. Multi-process KalmanStateTracker._write_state preserves newer disk_ts entries from another process
    #    and preserves seeded Kalman fields over unseeded cold-tick fields
    state_file = tmp_path / "cache_kalman_trend.json"
    state_file.write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": {
                        "price": 75000.0,
                        "kalman_price": 75000.0,
                        "kalman_vel": 2.5,
                        "kalman_vel_std": 0.4,
                        "kalman_trend": 1,
                        "ts": 1000.0,
                    },
                    "SOLUSDC": {
                        "price": 150.0,
                        "kalman_price": 150.0,
                        "kalman_vel": -0.5,
                        "kalman_vel_std": 0.2,
                        "kalman_trend": -1,
                        "ts": 1000.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    proc_a = KalmanStateTracker(state_path=str(state_file), compute_vol=False)
    proc_b = KalmanStateTracker(state_path=str(state_file), compute_vol=False)

    # proc_a updates BTCUSDC at ts=2000 with warm-up
    proc_a.update(
        "BTCUSDC",
        price=76000.0,
        ts=2000.0,
        epsilon=1.0,
        big_prices=[75000.0 + 20.0 * i for i in range(50)],
        big_sample_rate=1.0,
    )
    # proc_b (which loaded BTCUSDC at ts=1000) updates SOLUSDC at ts=2005 on a cold tick
    res_sol_cold = proc_b.update("SOLUSDC", price=151.0, ts=2005.0, epsilon=0.1)
    assert res_sol_cold["kalman_vel_std"] == pytest.approx(0.2)
    assert res_sol_cold["kalman_trend"] == -1

    raw_saved = json.loads(state_file.read_text(encoding="utf-8"))
    saved_items = raw_saved.get("items", raw_saved)
    assert saved_items["BTCUSDC"]["ts"] == pytest.approx(2000.0)
    assert saved_items["BTCUSDC"]["price"] == pytest.approx(76000.0)
    assert saved_items["SOLUSDC"]["ts"] == pytest.approx(2005.0)

    # 2. reconcile_intents_from_execution_audit sets terminal_ts on canceled_prior_oids and _ts_backfill_checked
    #    on unmatched reconciled_terminal intents, and build_last_cycle_execution_status does not let a
    #    newly-canceled old order hijack last_ts
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "execution_audit_2099-01-01.jsonl").write_text(
        json.dumps(
            {
                "event": "order_terminal",
                "intent_id": "intent_other",
                "order_id": "oid-other",
                "venue_status": "FILLED",
                "ts": 5000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    intents = [
        {
            "intent_id": "intent_old_btc",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "decision": "BUY",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "timestamp": 1000.0,
            "execution_payload": {"orderId": "oid-old", "status": "NEW"},
        },
        {
            "intent_id": "intent_unmatched_eth",
            "symbol": "ETHUSDC",
            "venue": "binance",
            "decision": "BUY",
            "order_type": "MARKET",
            "status": "EXECUTED",
            "timestamp": 4980.0,
            "execution_payload": {
                "orderId": "oid-unmatched",
                "status": "FILLED",
                "executedQty": "0.1",
                "price": "2500.0",
                "reconciled_terminal": True,
            },
        },
        {
            "intent_id": "intent_new_sol",
            "symbol": "SOLUSDC",
            "venue": "binance",
            "decision": "BUY",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "timestamp": 5000.0,
            "canceled_prior_open_orders": ["oid-old"],
            "execution_payload": {
                "orderId": "oid-new",
                "status": "FILLED",
                "executedQty": "1.0",
                "price": "150.0",
                "reconciled_terminal": True,
                "terminal_ts": 5000.0,
            },
        },
    ]
    changed = reconcile_intents_from_execution_audit(intents, str(audit_dir))
    assert changed is True
    assert intents[0]["execution_payload"]["status"] == "CANCELED"
    assert intents[0]["execution_payload"]["terminal_ts"] == pytest.approx(5000.0)
    assert intents[1]["execution_payload"].get("_ts_backfill_checked") is True

    # A second call should NOT need to re-scan audit files because terminal_ts / _ts_backfill_checked are set
    monkeypatch.setattr(
        "providers.execution_audit.glob.glob",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("glob should not be called on second pass")),
    )
    assert reconcile_intents_from_execution_audit(intents, str(audit_dir)) is False

    status_out = build_last_cycle_execution_status(intents, "crypto", 5005.0, extract_intent_notional)
    status_list = status_out if isinstance(status_out, list) else ([status_out] if status_out else [])
    status_syms = {s["symbol"] for s in status_list}
    assert "SOLUSDC" in status_syms
    assert "ETHUSDC" in status_syms
    assert "BTCUSDC" not in status_syms

    # 3. enrich_microstructure_cvd preserves explicit spot_cvd_15m_usd when pinfo is None,
    #    and reconcile_sibling_flow_and_liq_regimes updates sibling spot_perp_flow_regime
    minfo_explicit = {
        "cvd_15m_usd": 200_000.0,
        "spot_cvd_15m_usd": 150_000.0,
        "bid_depth_usd": 100_000.0,
        "ask_depth_usd": 100_000.0,
    }
    enrich_microstructure_cvd(minfo_explicit, None)
    assert minfo_explicit["spot_cvd_15m_usd"] == pytest.approx(150_000.0)
    assert "SYNCHRONIZED_SPOT_PERP_BUYING" in str(minfo_explicit.get("spot_perp_flow_regime", ""))

    micro_map = {
        "BTCUSDC": {
            "cvd_15m_usd": 500_000.0,
            "spot_cvd_15m_usd": 140_000.0,
            "bid_depth_usd": 800_000.0,
            "ask_depth_usd": 200_000.0,
            "_synthetic_spot_cvd": True,
        },
        "BTCUSDT": {
            "cvd_15m_usd": 500_000.0,
            "spot_cvd_15m_usd": -250_000.0,
            "bid_depth_usd": 200_000.0,
            "ask_depth_usd": 100_000.0,
            "_synthetic_spot_cvd": True,
        },
    }
    price_map = {
        "BTCUSDC": {"price": 75000.0, "kalman_trend": "UP", "slope_15m": 0.001, "vol_1h_pct": 1.2, "chg_15m_pct": 0.5},
        "BTCUSDT": {"price": 75000.0, "kalman_trend": "UP", "slope_15m": 0.001, "vol_1h_pct": 1.2, "chg_15m_pct": 0.5},
    }
    reconcile_sibling_flow_and_liq_regimes(micro_map, price_map)
    assert micro_map["BTCUSDT"]["spot_cvd_15m_usd"] > 0.0
    assert "SYNCHRONIZED_SPOT_PERP_BUYING" in str(micro_map["BTCUSDT"].get("spot_perp_flow_regime", ""))

    # 4. archive.<subpkg>.<mod> resolves subpackages inside legacy_archive.tar.gz
    shm_mod = importlib.import_module("archive.shmem.shmutils")
    assert hasattr(shm_mod, "SharedMemory") or shm_mod.__name__ == "archive.shmem.shmutils"

    # 5. estimate_remaining_trend_hours handles extreme duration_hours without OverflowError
    rem = estimate_remaining_trend_hours(duration_hours=1e160, p_cont=0.6)
    assert isinstance(rem, float) and rem > 0.0


def test_iter88_cachetrend_adapt_thresholds_compact_repo_rules_whale_delta_4dec_and_sidecar_migration(
    tmp_path, monkeypatch
):
    import json
    import subprocess
    from types import SimpleNamespace
    import cachetrend
    from cachepaths import resolve_cachedb_filepath
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        update_microstructure_cycle_deltas,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _apply_24h_series_enrichment,
    )
    from intelligence.sentiment.gemini_client import GeminiClient
    from orchestratorOS.agy.topics import TopicSession

    # 1. CacheShortTrendManager.evaluate_full populates adapt_reentry_pct and adapt_dca_pct
    #    from KalmanStateTracker without raising NameError on undefined adapt_reentry/adapt_dca
    pw = cachetrend.pw

    st_file = tmp_path / "cache_short_trend.json"
    mgr = cachetrend.CacheShortTrendManager(
        sync_ts=1.0,
        symbols=["BTCUSDC"],
        filename=str(st_file),
        read_only=True,
    )
    mgr.writer = True
    mgr._full_eval_thread = object()
    mgr._enable_kalman_tracker = True
    mgr.current_price_mgr = SimpleNamespace(
        get_cached_price_observation=lambda sym, stale: (1_700_003_600.0, 70600.0),
    )
    mgr.windows["BTCUSDC"] = {}
    mgr.analyzers["BTCUSDC"] = {}
    for sec in mgr.window_seconds:
        w = pw.PriceWindow("BTCUSDC", window_size=60, sample_rate_sec=60.0)
        mgr.windows["BTCUSDC"][sec] = w
        mgr.analyzers["BTCUSDC"][sec] = pw.WindowAnalyzer(w)
    for i in range(60):
        px = 70000.0 + (i % 5) * 40.0 + i * 10.0
        for w in mgr.windows["BTCUSDC"].values():
            w.prices.append(px)
            w.sample_rate_sec = 60.0
    mgr.evaluate_full("BTCUSDC")
    btc_snap = mgr.get_snapshot("BTCUSDC")
    assert "kalman_vel" in btc_snap
    assert "adapt_reentry_pct" in btc_snap and btc_snap["adapt_reentry_pct"] is not None
    assert "adapt_dca_pct" in btc_snap and btc_snap["adapt_dca_pct"] is not None

    # 2. GeminiClient._finalize_topic_turn preserves include_repo_rules=True during /compact
    #    for can_self_heal=True topics (e.g. technical_repair)
    fake_bin = tmp_path / "agy-bot"
    fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    bot_cli_base = tmp_path / "bot_cli"
    bot_cli_base.mkdir()
    seen_repo_rules = []

    client = GeminiClient(cli_path=str(fake_bin), project_id="autonommptrade")

    def _fake_exec_cli(resume, text, call_timeout, cli_model, cli_base, run_env, is_mocked, *, include_repo_rules=None):
        eff_rules = (
            bool(getattr(client._JSON_MODE_LOCAL, "include_repo_rules", False))
            if include_repo_rules is None
            else bool(include_repo_rules)
        )
        seen_repo_rules.append((text, eff_rules))
        return subprocess.CompletedProcess(
            ["agy"], 0, stdout=json.dumps({"response": '{"status": "ok"}', "conversation_id": "cid-rep"}), stderr=""
        )

    monkeypatch.setattr(client, "_execute_cli", _fake_exec_cli)
    monkeypatch.setattr(
        "intelligence.sentiment.gemini_client.prepare_topic_session",
        lambda *_a, **_kw: TopicSession("cid-rep", "", True),
    )
    monkeypatch.setattr("intelligence.sentiment.gemini_client.record_topic_turn", lambda **_kw: None)
    out_rep = client.query_text("repair system", topic_key="auto_technical_repair")
    assert out_rep == '{"status": "ok"}'
    assert len(seen_repo_rules) == 2
    assert seen_repo_rules[0][1] is True
    assert "/compact" in seen_repo_rules[1][0] and seen_repo_rules[1][1] is True

    # 3. update_microstructure_cycle_deltas preserves 4-decimal whale_long_pct deltas and
    #    bidirectional native_l1_oi_usd scale matching
    cdir = tmp_path / "cachedb_m"
    cdir.mkdir()
    mhist_f = cdir / "microstructure_cycle_history.json"
    mhist_f.write_text(
        json.dumps(
            {
                "HYPEUSDC": [
                    {
                        "ts": 1000.0,
                        "whale_long_pct": 0.6605,
                        "oi_usd": 60_000_000.0,
                        "native_l1_oi_usd": 1_200_000_000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    micro_hype = {
        "HYPEUSDC": {
            "whale_long_pct": 0.6645,
            "oi_usd": 1_260_000_000.0,
        }
    }
    update_microstructure_cycle_deltas(str(cdir), micro_hype, now_ts=1600.0)
    cd = micro_hype["HYPEUSDC"]["cycle_delta"]
    assert cd["whale_long_delta_pct"] == pytest.approx(0.004)
    assert cd["prev_whale_long_pct"] == pytest.approx(0.6605)
    assert cd["oi_delta_pct"] == pytest.approx(5.0)
    assert cd["prev_oi_usd"] == pytest.approx(1_200_000_000.0)

    # 4. _apply_24h_series_enrichment backfills missing kalman_vel_std without clobbering seeded kalman_vel
    entry_k = {"price": 105.0, "kalman_vel": 0.123, "kalman_trend": 1}
    ser_24 = [[1_700_000_000_000 + i * 60_000, 100.0 + i * 0.2] for i in range(30)]
    _apply_24h_series_enrichment("SOLUSDC", entry_k, {"chg_1h_pct": 1.0}, ser_24, None)
    assert entry_k["kalman_vel"] == pytest.approx(0.123)
    assert entry_k.get("kalman_vel_std") is not None and entry_k["kalman_vel_std"] < 5.99

    # 5. resolve_cachedb_filepath migrates .previous and .meta sidecars along with the main cache file
    sub_cdir = tmp_path / "cachedb_sub"
    (sub_cdir / "trends").mkdir(parents=True)
    flat_f = sub_cdir / "cache_short_trend.json"
    flat_f.write_text("{}", encoding="utf-8")
    (sub_cdir / "cache_short_trend.json.previous").write_text('{"prev": 1}', encoding="utf-8")
    (sub_cdir / "cache_short_trend.json.meta").write_text('{"meta": 1}', encoding="utf-8")
    resolved_p = resolve_cachedb_filepath(str(flat_f))
    assert str(resolved_p).endswith("trends/cache_short_trend.json")
    assert (sub_cdir / "trends" / "cache_short_trend.json.previous").is_file()
    assert (sub_cdir / "trends" / "cache_short_trend.json.meta").is_file()


def test_iter13_equity_post_rth_24h_cutoff_and_archive_7d_fallback(tmp_path, monkeypatch):
    import json
    import types
    from intelligence.sentiment.collectors import microstructure_collector as mc_mod
    from intelligence.sentiment.collectors.microstructure_collector import (
        enrich_24h_and_microstructure,
        load_symbol_price_series,
    )
    from state_io import save_json_state

    # 1. Discrepancy BB: when cache_sparse_price has < 5d span (and need_24h_supplement is False),
    #    load_symbol_price_series falls back to cache_365_price with longer (>5d) span
    now_ms = 1_793_836_000_000
    p24 = tmp_path / "prices" / "cache_24_price_ADAUSD.json"
    p24.parent.mkdir(parents=True, exist_ok=True)
    save_json_state(
        str(p24),
        {"items": {"ADAUSD": [[now_ms - 24 * 3600 * 1000, 0.19], [now_ms, 0.20]]}},
    )
    psparse = tmp_path / "prices" / "cache_sparse_price_ADAUSD.jsonl"
    psparse.write_text(
        "\n".join(
            [
                json.dumps({"s": "ADAUSD", "i": [now_ms - 26 * 3600 * 1000, 0.191]}),
                json.dumps({"s": "ADAUSD", "i": [now_ms, 0.20]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p365 = tmp_path / "prices" / "cache_365_price_ADAUSD.json"
    save_json_state(
        str(p365),
        {
            "items": {
                "ADAUSD": [
                    [now_ms - 7 * 86400 * 1000, 0.21],
                    [now_ms - 3 * 86400 * 1000, 0.195],
                    [now_ms, 0.20],
                ]
            }
        },
    )
    s24, vals_7d = load_symbol_price_series(str(tmp_path), "ADAUSD", include_7d=True)
    assert len(s24) >= 2
    assert vals_7d[0] == pytest.approx(0.21)

    # 2. Discrepancy BA & BC: enrich_24h_and_microstructure seeds T212 position via free_base from venue_balances_state.json
    #    and anchors 24h chart cutoff to min(now_ms, pts_5d[-1][0]) - 86400_000 so post-RTH equities preserve chg_24h_pct
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    save_json_state(
        str(state_dir / "venue_balances_state.json"),
        {
            "pos:t212:RGNT_US_EQ": {
                "venue": "t212",
                "symbol": "RGNT_US_EQ",
                "price": 1.32,
                "free_base": 122.51,
                "ts": (now_ms / 1000.0) - 60.0,
            }
        },
    )
    # Last RTH bar was 2.5 hours ago (e.g. 23:00 EET while now_ts is 01:30 EET)
    rth_close_ms = now_ms - int(2.5 * 3600 * 1000)
    prev_rth_close_ms = rth_close_ms - 86400 * 1000
    fake_5d_bars = [[prev_rth_close_ms, 1.37, 1500.0]] + [
        [rth_close_ms - (24 - i) * 15 * 60 * 1000, round(1.35 - i * 0.00125, 4), 800.0]
        for i in range(25)
    ]
    fake_md = types.SimpleNamespace(_chart=lambda sym, rng="5d", interval="5m": ({}, fake_5d_bars))
    monkeypatch.setattr(mc_mod, "load_local_module", lambda *_a, **_kw: fake_md)

    pt_map: dict = {}
    micro_map: dict = {}
    enrich_24h_and_microstructure(
        str(tmp_path),
        str(tmp_path),
        pt_map,
        micro_map,
        {},
        {"RGNT_US_EQ": types.SimpleNamespace(provider_name="t212")},
        now_ms / 1000.0,
    )
    assert "RGNT_US_EQ" in pt_map
    assert pt_map["RGNT_US_EQ"].get("chg_24h_pct") == pytest.approx(round(((1.32 - 1.37) / 1.37) * 100.0, 2))

    # 3. Discrepancy BD: load_symbol_price_series supplements <18h cache_24_price from cache_365_price
    #    including the last prior-session bar (< cutoff_24h) across an overnight session gap
    p24_spcx = tmp_path / "prices" / "cache_24_price_SPCX_US_EQ.json"
    save_json_state(
        str(p24_spcx),
        {
            "items": {
                "SPCX_US_EQ": [
                    [now_ms - 15 * 3600 * 1000, 134.74],
                    [now_ms - 8 * 3600 * 1000, 135.20],
                    [now_ms, 134.50],
                ]
            }
        },
    )
    p365_spcx = tmp_path / "prices" / "cache_365_price_SPCX_US_EQ.json"
    save_json_state(
        str(p365_spcx),
        {
            "items": {
                "SPCX_US_EQ": [
                    [now_ms - 5 * 86400 * 1000, 143.40],
                    [now_ms - int(24.25 * 3600 * 1000), 140.94],
                    [now_ms - 15 * 3600 * 1000, 134.74],
                    [now_ms, 134.50],
                ]
            }
        },
    )
    s24_spcx, v7_spcx = load_symbol_price_series(str(tmp_path), "SPCX_US_EQ", include_7d=True)
    assert s24_spcx[0][1] == pytest.approx(140.94)
    assert (s24_spcx[-1][0] - s24_spcx[0][0]) >= 24 * 3600 * 1000
    assert v7_spcx[0] == pytest.approx(143.40)


def test_iter89_geopolitical_subset_deescalation_news_first_seen_post_rth_slopes_and_sibling_geometry(
    tmp_path, monkeypatch
):
    import json
    from intelligence.macro.geopolitical_analyzer import (
        GeopoliticalThreatAnalyzer,
        GeopoliticalThreatAssessment,
    )
    from intelligence.macro.news_feed_collector import (
        NewsFeedCollector,
        NewsFeedSnapshot,
        NewsHeadline,
    )
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as ptc
    from orchestratorOS.supervisor.prompt_sections import _format_market_prices_section

    # 1. NewsFeedCollector._aggregate preserves headline.first_seen_ts when pub_date is empty
    hist_f = tmp_path / "trends" / "news_feed_collect.json"
    hist_f.parent.mkdir(parents=True, exist_ok=True)
    nfc = NewsFeedCollector(history_file=str(hist_f))
    assert nfc._cache_dir() == str(tmp_path)
    now_t = 1_700_010_000.0
    h_aged = NewsHeadline(
        title="Emergency tariff shock announced by G7",
        source="Wire",
        pub_date="",
        url="https://example.com/g7-tariff",
        matched_keywords=("tariff",),
        is_high_severity=True,
        first_seen_ts=now_t - 7200.0,
    )
    snap_agg = nfc._aggregate(
        NewsFeedSnapshot(
            headlines=(h_aged,),
            total_fetched=1,
            high_severity_count=1,
            has_critical_shock_keywords=False,
            ts=now_t,
        ),
        now=now_t,
    )
    assert len(snap_agg.headlines) == 1
    assert snap_agg.headlines[0].first_seen_ts == pytest.approx(now_t - 7200.0)

    # 2. GeopoliticalThreatAnalyzer.assess de-escalates >90m old headlines to NORMAL even when an
    #    older headline expired from the 6h window (strict subset of _last_headlines_digest)
    state_f = tmp_path / "geopolitical_eval.json"
    analyzer = GeopoliticalThreatAnalyzer(state_file=str(state_f))
    analyzer._cached_assessment = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.65,
        summary="Prior elevated state",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=2,
        ts=now_t - 2400.0,
    )
    analyzer._last_eval_ts = now_t - 2400.0
    analyzer._last_headlines_digest = (
        "Emergency tariff shock announced by G7|Older sanctions headline that aged out of 6h pool"
    )
    deesc = analyzer.assess(snap_agg, now_ts=now_t)
    assert deesc.threat_level == "NORMAL"
    assert deesc.recommended_brake == "NONE"

    # 3. collect_pretrade_telemetry prunes stale 3.7m slope on post-RTH equities even when
    #    venue_balances_state.json has a <=0.5% price difference (eq_post_rth_override=True, eq_px_diverged_05pct=False)
    eq_dir = tmp_path / "cachedb_eq89"
    eq_dir.mkdir(parents=True, exist_ok=True)
    old_base_ms = 1_700_000_000_000
    nvda_bars = [[old_base_ms - (30 - i) * 300_000, round(190.0 + i * 0.15, 4), 1000.0] for i in range(31)]
    (eq_dir / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": nvda_bars}}),
        encoding="utf-8",
    )
    (eq_dir / "cache_short_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "NVDA_US_EQ": {
                        "slope_3_7m": 0.0025,
                        "slope_15m": 0.0020,
                        "slope_2_5h": 0.0010,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # Last bar is 194.50; set T212 position price to 194.30 (~0.10% diff <= 0.5%)
    (eq_dir / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:t212:NVDA_US_EQ": {
                    "symbol": "NVDA_US_EQ",
                    "free": 1.0,
                    "price": 194.30,
                    "ts": (old_base_ms / 1000.0) + 600.0,
                }
            }
        ),
        encoding="utf-8",
    )
    telem_small_diff = ptc.collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(eq_dir))
    assert "3.7m=" not in telem_small_diff.get("Dynamic Multi-Window Slopes", "")

    # 4. _format_market_prices_section recalculates VP/range geometry for primary symbol price
    #    when poc_px/val_px/vah_px are backfilled from a sibling symbol
    lines_3a = _format_market_prices_section(
        {
            "TAOUSDC": {"price": 410.0, "chg_15m_pct": 0.5, "chg_24h_pct": 2.0},
            "TAOUSD": {
                "price": 400.0,
                "poc_px": 400.0,
                "val_px": 390.0,
                "vah_px": 405.0,
                "poc_dist_pct": 0.0,
                "vah_dist_pct": -1.23,
                "val_dist_pct": 2.56,
                "va_pos_pct": 66.7,
                "va_regime": "INSIDE_VA_UPPER",
            },
        }
    )
    joined_3a = "\n".join(lines_3a)
    assert "POC=400.0[+2.50%]" in joined_3a
    assert "va_pos=133.3%" in joined_3a


def test_iter14_non_tradable_t212_cycle_recent_order_avg_sync():
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
    )

    ro_spcx = {
        "venue": "t212",
        "last_action_side": "BUY",
        "active_avg_buy_px": 140.448972,
        "tactical_avg_buy_px": 140.96253,
        "last_buy": {"price": 140.51, "qty": 0.71, "actor": "AI"},
    }
    ro_rgnt = {
        "venue": "t212",
        "last_action_side": "BUY",
        "active_avg_buy_px": 1.43,
        "last_buy": {"price": 1.43, "qty": 72.14, "actor": "AI"},
    }
    recent_orders = {"SPCX_US_EQ": ro_spcx, "RGNT_US_EQ": ro_rgnt}
    venue_recent_orders = {"t212": {"SPCX_US_EQ": ro_spcx, "RGNT_US_EQ": ro_rgnt}}
    t212_cycles = {
        "SPCX_US_EQ": {"qty": 7.042965, "avg_cost_px": 144.9872, "entry_price": 146.8203},
        "RGNT_US_EQ": {"qty": 122.51, "avg_cost_px": 1.4395, "entry_price": 1.50},
    }
    enrich_positions_with_orders_and_cycles(
        [],
        {"SPCX_US_EQ": {"price": 134.76}, "RGNT_US_EQ": {"price": 1.32}},
        {},
        venue_recent_orders,
        recent_orders,
        {},
        {},
        t212_cycles,
        {},
    )
    assert recent_orders["SPCX_US_EQ"]["active_avg_buy_px"] == pytest.approx(144.9872)
    assert recent_orders["SPCX_US_EQ"]["tactical_avg_buy_px"] == pytest.approx(140.96253)
    assert recent_orders["RGNT_US_EQ"]["active_avg_buy_px"] == pytest.approx(1.4395)
    assert recent_orders["RGNT_US_EQ"]["tactical_avg_buy_px"] == pytest.approx(1.43)

    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        reconcile_venue_cycles_with_recent_orders,
    )

    ro_tao = {
        "venue": "kraken",
        "last_action_side": "BUY",
        "active_avg_buy_px": 235.741094,
        "lifo_realized_pnl_usd": -116.73,
        "last_buy": {"price": 235.741094, "qty": 3.70189, "ts_ms": 1_793_328_000_000},
        "last_sell": {"price": 226.016729, "qty": 7.40379, "ts_ms": 1_793_316_000_000},
        "recent_buys": [{"price": 235.741094, "qty": 3.70189, "ts_ms": 1_793_328_000_000}],
    }
    kraken_cycles = {
        "TAOUSD": {
            "qty": 7.38813,
            "cost_usd": 2105.51,
            "avg_cost_px": 284.98,
            "entry_price": 305.469,
            "_entry_ts_ms": 1_791_188_569_000,
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        kraken_cycles,
        {},
        {},
        {"kraken": {"TAOUSD": ro_tao}},
        {("kraken", "TAOUSD"): 3.706878},
        {"TAOUSD": {"price": 236.4}},
    )
    assert ro_tao["lifo_realized_pnl_usd"] == 0.0



def test_iter90_cross_cycle_dca_buys_multi_fill_oversold_pnl_taker_horizon_clause_and_non_tradable_ro_sync(
    tmp_path,
):
    import json
    from intelligence.macro.calendar_ssot import (
        compute_dynamic_geopolitical_interval,
        parse_cited_taker_ratio,
    )
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
    from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
        extract_symbol_recent_fills,
        reconcile_venue_cycles_with_recent_orders,
    )

    # 1. reconcile_venue_cycles_with_recent_orders does not inflate dca_buys or flip TP->REDUCE
    #    when buys_before and last_sell occurred BEFORE the current cycle's _entry_ts_ms
    kr_new_cyc = {
        "SOLUSD": {
            "qty": 5.0,
            "cost_usd": 525.0,
            "avg_cost_px": 105.0,
            "entry_price": 105.0,
            "last_buy_price": 105.0,
            "last_sell_price": 110.0,
            "last_exit_kind": "TP",
            "dca_buys": 0,
            "_entry_ts_ms": 1_793_800_000_000,
            "_last_sell_ts_ms": 1_793_700_000_000,
        }
    }
    v_ro_new = {
        "kraken": {
            "SOLUSD": {
                "last_sell": {"price": 102.0, "qty": 4.0, "ts_ms": 1_793_750_000_000, "actor": "BOT"},
                "recent_buys": [
                    {"price": 108.0, "qty": 4.0, "ts_ms": 1_793_700_000_000, "actor": "BOT"},
                    {"price": 105.0, "qty": 5.0, "ts_ms": 1_793_800_000_000, "actor": "BOT"},
                ],
                "active_avg_buy_px": 105.0,
                "active_cycle_rem_qty": 5.0,
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        kr_new_cyc,
        {},
        {},
        v_ro_new,
        {("kraken", "SOLUSD"): 5.0},
        {"SOLUSD": {"price": 106.0}},
    )
    assert kr_new_cyc["SOLUSD"]["dca_buys"] == 0
    assert kr_new_cyc["SOLUSD"]["last_exit_kind"] == "TP"

    # 2. extract_symbol_recent_fills preserves lifo_realized_pnl_usd when SELL #1 exhausts tracked cost_qty
    #    to 0.0 and SELL #2 sells additional core inventory while prev_cost_qty == 0.0
    t0 = 1_793_800_000_000
    fills_multi_sell = extract_symbol_recent_fills(
        [
            {"side": "BUY", "price": 10.0, "qty": 50.0, "ts_ms": t0},
            {"side": "SELL", "price": 9.5, "qty": 50.0, "ts_ms": t0 + 1_000_000},
            {"side": "SELL", "price": 9.5, "qty": 30.0, "ts_ms": t0 + 1_005_000},
            {"side": "BUY", "price": 9.2, "qty": 20.0, "ts_ms": t0 + 2_000_000},
        ]
    )
    assert fills_multi_sell is not None
    assert fills_multi_sell["lifo_realized_pnl_usd"] == pytest.approx(-25.0)

    # 3. parse_cited_taker_ratio ranks 15m/overall above 1h/4h above 5m even when adjectives
    #    appear between the horizon token and 'taker'
    txt_adj = (
        "Despite 5m aggressive taker buy resurgence (ratio 1.64, CVD +$9.25M), "
        "1h aggregate taker flow remains net negative (ratio 0.60, 1h CVD -$25.46M)."
    )
    assert parse_cited_taker_ratio(txt_adj) == pytest.approx(0.60)

    # 4. compute_dynamic_geopolitical_interval honors dict-based age_min>=90m and does not suppress
    #    genuinely unseen high-severity headlines
    now_s = 1_793_800_000.0
    cached_elev = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.5,
        summary="Prior elevated state",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=1,
        ts=now_s - 3600.0,
    )
    itv_age_min, reason_age_min = compute_dynamic_geopolitical_interval(
        base_interval_sec=16200.0,
        last_eval_ts=now_s - 3600.0,
        now_ts=now_s,
        cached_assessment=cached_elev,
        news_snapshot={
            "high_severity_count": 1,
            "has_critical_shock_keywords": False,
            "headlines": [
                {"title": "Emergency tariff shock announced by G7", "is_high_severity": True, "age_min": 120.0}
            ],
        },
        headlines_changed=False,
        last_headlines_digest="Emergency tariff shock announced by G7",
    )
    assert reason_age_min == "threat_deescalation_zero_high_sev_recheck"

    # 5. enrich_positions_with_orders_and_cycles deduplicates quote variants in seen_venue_syms
    #    and load_symbol_best_recent_fills preserves tactical_avg_buy_px on T212 cycle sync
    pos_hl = [{"venue": "hyperliquid", "symbol": "HYPEUSD", "quote": "USDC", "free_base": 10.0, "price": 25.0}]
    ro_hl = {"HYPEUSD": {"venue": "hyperliquid", "last_action_side": "BUY", "active_avg_buy_px": 24.0}}
    v_ro_hl = {"hyperliquid": {"HYPEUSD": ro_hl["HYPEUSD"]}}
    hl_cycs = {"HYPEUSDC": {"qty": 10.0, "avg_cost_px": 24.0, "entry_price": 24.0}}
    enrich_positions_with_orders_and_cycles(
        pos_hl,
        {"HYPEUSD": {"price": 25.0}},
        {},
        v_ro_hl,
        ro_hl,
        {},
        hl_cycs,
        {},
        {},
    )
    assert "HYPEUSDC" not in ro_hl

    ws_t212 = tmp_path / "ws_t212"
    cdir_t212 = ws_t212 / "cachedb"
    t212_dir = ws_t212 / "212trading"
    cdir_t212.mkdir(parents=True, exist_ok=True)
    t212_dir.mkdir(parents=True, exist_ok=True)
    (t212_dir / ".state_RGNT_US_EQ.json").write_text(
        json.dumps({"qty": 122.51, "avg_cost_px": 1.4395, "entry_price": 1.50}),
        encoding="utf-8",
    )
    (cdir_t212 / "cache_trade_t212.json").write_text(
        json.dumps(
            {
                "items": {
                    "RGNT_US_EQ": [
                        {
                            "side": "BUY",
                            "price": 1.43,
                            "qty": 72.14,
                            "executedQty": 72.14,
                            "timestamp": 1_793_800_000_000,
                            "status": "FILLED",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    best_rgnt = load_symbol_best_recent_fills(str(cdir_t212), "RGNT_US_EQ", ["RGNT_US_EQ", "RGNT"])
    assert best_rgnt is not None
    assert best_rgnt["active_avg_buy_px"] == pytest.approx(1.4395)
    assert best_rgnt["tactical_avg_buy_px"] == pytest.approx(1.43)


def test_iter16_post_close_equity_pos_ts_clamp_and_partial_15m_tape_prune(tmp_path):
    import json
    from intelligence.sentiment.collectors.microstructure_collector import (
        enrich_24h_and_microstructure,
    )
    from state_io import save_json_state

    # 1. When T212 balance harvester updates pos:t212:NVDA_US_EQ['ts'] 12.5m after the last 5m chart bar
    #    (while still within the 20m rth_bar_fresh window), eq_pt_ts_ms is clamped to last_bar_ts + 1000ms
    #    so no artificial 12.5m gap breaks chg_5m_pct or advances last_eq_pt[0] past the real bar close
    close_ms = 1_793_840_397_000
    pts_nvda = [
        [close_ms - (20 - i) * 300_000, round(201.0 - i * 0.06, 4), 1000.0]
        for i in range(21)
    ]
    p24_f = tmp_path / "prices" / "cache_24_price_NVDA_US_EQ.json"
    p24_f.parent.mkdir(parents=True, exist_ok=True)
    save_json_state(str(p24_f), {"items": {"NVDA_US_EQ": pts_nvda}})
    vb_f = tmp_path / "state" / "venue_balances_state.json"
    vb_f.parent.mkdir(parents=True, exist_ok=True)
    save_json_state(
        str(vb_f),
        {
            "pos:t212:NVDA_US_EQ": {
                "venue": "t212",
                "symbol": "NVDA_US_EQ",
                "price": 199.05,
                "free_base": 0.41,
                "ts": (close_ms / 1000.0) + 747.0,
            }
        },
    )
    pt_14m: dict = {}
    enrich_24h_and_microstructure(
        str(tmp_path),
        str(tmp_path),
        pt_14m,
        {},
        {},
        {},
        now_ts=(close_ms / 1000.0) + 867.0,
    )
    assert pt_14m["NVDA_US_EQ"].get("chg_5m_pct") is not None
    assert any(
        str(x).startswith("-5m:")
        for x in (pt_14m["NVDA_US_EQ"].get("recent_tape_15m") or [])
    )

    # 2. If an equity series already has a >10.2m tail gap (no chg_1m/3m/5m_pct) and <=0.5% price diff,
    #    partial recent_tape_15m and stale slope_3_7m are pruned
    pts_gapped = pts_nvda + [[close_ms + 747_000, 199.05]]
    save_json_state(str(p24_f), {"items": {"NVDA_US_EQ": pts_gapped}})
    pt_gapped: dict = {"NVDA_US_EQ": {"price": 199.05, "slope_3_7m": -0.00029, "slope_15m": -0.00021}}
    enrich_24h_and_microstructure(
        str(tmp_path),
        str(tmp_path),
        pt_gapped,
        {},
        {},
        {},
        now_ts=(close_ms / 1000.0) + 867.0,
    )
    assert "chg_5m_pct" not in pt_gapped["NVDA_US_EQ"]
    assert "recent_tape_15m" not in pt_gapped["NVDA_US_EQ"]
    assert "slope_3_7m" not in pt_gapped["NVDA_US_EQ"]
    assert "slope_15m" in pt_gapped["NVDA_US_EQ"]


def test_iter91_multi_fill_stale_7d_pnl_post_sell_new_cycle_lifo_reset_t212_retry_and_manifest_shutdown_order(tmp_path):
    import json
    from unittest.mock import MagicMock
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        extract_symbol_recent_fills,
        reconcile_venue_cycles_with_recent_orders,
    )
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs
    from orchestratorTrade.orchestrator import BotManager

    # 1. Multi-fill core SELL after >7d lot exhaustion followed by new BUY resets lifo_realized_pnl_usd == 0.0
    base_ts = 1_790_000_000_000
    nine_days_ms = 9 * 86400 * 1000
    items_stale = {
        "STALEUSDC": [
            {
                "side": "BUY",
                "price": 100.0,
                "qty": 5.0,
                "executedQty": 5.0,
                "timestamp": base_ts,
                "status": "FILLED",
            },
            {
                "side": "SELL",
                "price": 110.0,
                "qty": 5.0,
                "executedQty": 5.0,
                "timestamp": base_ts + nine_days_ms,
                "status": "FILLED",
            },
            {
                "side": "SELL",
                "price": 112.0,
                "qty": 5.0,
                "executedQty": 5.0,
                "timestamp": base_ts + nine_days_ms + 1000,
                "status": "FILLED",
            },
            {
                "side": "BUY",
                "price": 108.0,
                "qty": 5.0,
                "executedQty": 5.0,
                "timestamp": base_ts + nine_days_ms + 60_000,
                "status": "FILLED",
            },
        ]
    }
    res_stale = extract_symbol_recent_fills(items_stale["STALEUSDC"])
    assert res_stale is not None
    assert res_stale["lifo_realized_pnl_usd"] == pytest.approx(0.0)

    # 2. reconcile_venue_cycles_with_recent_orders resets lifo_realized_pnl_usd when post_sell_new_cycle is True
    kr_new_cyc = {
        "SUIUSD": {
            "qty": 20.0,
            "cost_usd": 19.2,
            "avg_cost_px": 0.96,
            "entry_price": 0.96,
            "last_buy_price": 0.96,
            "last_sell_price": 0.98,
            "last_exit_kind": "TP",
            "dca_buys": 0,
            "_entry_ts_ms": 1_793_800_000_000,
            "_last_sell_ts_ms": 1_793_750_000_000,
        }
    }
    v_ro_new = {
        "kraken": {
            "SUIUSD": {
                "last_sell": {"price": 0.98, "qty": 25.0, "ts_ms": 1_793_750_000_000, "actor": "BOT"},
                "recent_buys": [
                    {"price": 0.90, "qty": 25.0, "ts_ms": 1_793_700_000_000, "actor": "BOT"},
                    {"price": 0.96, "qty": 20.0, "ts_ms": 1_793_800_001_000, "actor": "BOT"},
                ],
                "active_avg_buy_px": 0.96,
                "active_cycle_rem_qty": 20.0,
                "lifo_realized_pnl_usd": 2.0,
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        kr_new_cyc,
        {},
        {},
        v_ro_new,
        {("kraken", "SUIUSD"): 20.0},
        {"SUIUSD": {"price": 0.97}},
    )
    assert v_ro_new["kraken"]["SUIUSD"]["lifo_realized_pnl_usd"] == pytest.approx(0.0)

    # 3. harvest_log_anomalies_from_dirs only suppresses insufficient-free-for-stocks-buy when same ticker succeeds nearby
    log_dir = tmp_path / "logs_iter91"
    log_dir.mkdir(parents=True)
    (log_dir / "t212_exec.log").write_text(
        "\n".join(
            [
                "2026-11-05 03:00:01 ERROR Place order failed for NVDA_US_EQ: {\"code\": \"insufficient-free-for-stocks-buy\"}",
                "2026-11-05 03:00:02 INFO Successfully placed BUY for AAPL_US_EQ",
            ]
        ),
        encoding="utf-8",
    )
    anomalies_diff_sym = harvest_log_anomalies_from_dirs([str(log_dir)], lookback_hours=999999)
    assert anomalies_diff_sym.error_count == 1
    assert any("insufficient-free-for-stocks-buy" in ev for ev in anomalies_diff_sym.critical_events)

    (log_dir / "t212_exec.log").write_text(
        "\n".join(
            [
                "2026-11-05 03:00:01 ERROR Place order failed for NVDA_US_EQ: {\"code\": \"insufficient-free-for-stocks-buy\"}",
                "2026-11-05 03:00:02 INFO Successfully placed BUY for NVDA_US_EQ",
            ]
        ),
        encoding="utf-8",
    )
    anomalies_same_sym = harvest_log_anomalies_from_dirs([str(log_dir)], lookback_hours=999999)
    assert anomalies_same_sym.error_count == 0
    assert not any("insufficient-free-for-stocks-buy" in ev for ev in anomalies_same_sym.critical_events)

    # 4. BotManager._ordered_process_names_reversed honors procs.conf manifest order even after dict re-insertion
    mgr = BotManager.__new__(BotManager)
    mgr.bots = [
        {"name": "gateway", "cmd": "gw.py"},
        {"name": "monitor", "cmd": "mon.py"},
        {"name": "worker", "cmd": "wrk.py"},
    ]
    # Simulate gateway restarted later (moved to end of dict insertion order)
    mgr.processes = {
        "monitor": MagicMock(),
        "worker": MagicMock(),
        "gateway": MagicMock(),
    }
    assert mgr._ordered_process_names_reversed() == ["worker", "monitor", "gateway"]


def test_iter92_neutral_hold_cited_flow_stale_advisor_retry_delta_whale_freshest_and_watchdog_recovery(tmp_path):
    import json
    import time
    from types import SimpleNamespace
    from intelligence.macro.calendar_ssot import compute_dynamic_sentiment_interval
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.macro_sentiment_ssot import load_cached_whale_telemetry
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        enrich_sentiment_advisor_live_flow_divergence,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        reconcile_venue_cycles_with_recent_orders,
    )
    from orchestratorOS.lib.technical_repair_ssot import gather_system_and_vpn_telemetry

    # 1. CAUTION/HOLD posture with already-cited low/high taker ratio does not trigger false LIVE_SELL_REVERSAL / LIVE_BUY_REVERSAL
    sa_caution_sell = {
        "market_bias": "CAUTION",
        "recommended_action": "HOLD",
        "age_min": 16.0,
        "summary": "Market remains cautious on HOLD amid heavy taker sell volume (taker ratio 0.53, 5m CVD -$31.3M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_caution_sell,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.55,
                "cvd_15m_usd": -12_000_000.0,
            }
        },
    )
    assert "live_flow_divergence" not in sa_caution_sell

    sa_caution_buy = {
        "market_bias": "CAUTION",
        "recommended_action": "HOLD",
        "age_min": 16.0,
        "summary": "Market awaits confirmation on HOLD despite strong taker buying (taker ratio 1.38, 15m CVD +$9.2M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_caution_buy,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 1.40,
                "cvd_15m_usd": 8_500_000.0,
            }
        },
    )
    assert "live_flow_divergence" not in sa_caution_buy

    # 2. compute_dynamic_sentiment_interval uses cited_tr when already in the same extreme regime as cur_tr
    #    even if prev_btc_taker_ratio was 0.82 (15m lagging 5m at prior evaluation)
    cached_caution = SimpleNamespace(
        market_bias="CAUTION",
        risk_level="MODERATE",
        recommended_action="HOLD",
        summary="Intraday derivatives shifted into aggressive_shorting (taker ratio 0.53, 5m CVD -$31.3M).",
        key_risks=[],
        is_fallback=False,
        ts=1_793_800_000.0,
    )
    itv_same_sell, reason_same_sell = compute_dynamic_sentiment_interval(
        base_interval_sec=14400.0,
        last_eval_ts=1_793_800_000.0,
        now_ts=1_793_800_360.0,
        cached_assessment=cached_caution,
        btc_taker_ratio=0.58,
        prev_btc_taker_ratio=0.82,
    )
    assert itv_same_sell > 300.0
    assert "btc_whale_flow_regime_flip" not in reason_same_sell
    assert "btc_seller_reversal" not in reason_same_sell

    # 3. MacroAnalyzer.assess_market_advisor does not overwrite _last_fg_value / _last_btc_taker_ratio when review()
    #    fails and returns the stale _cached_assessment
    cdir_ma = tmp_path / "cdir_ma"
    cdir_ma.mkdir(parents=True, exist_ok=True)
    (cdir_ma / "fear_greed_cache.json").write_text(
        json.dumps({"value": 72, "classification": "Greed", "ts": 1_793_800_500.0}),
        encoding="utf-8",
    )
    (cdir_ma / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps({"taker_buy_sell_ratio": 1.45, "ts": 1_793_800_500.0}),
        encoding="utf-8",
    )
    ma = MacroAnalyzer(cache_dir=str(cdir_ma), symbols=["BTCUSDC"])
    ma.shadow_notify = False
    ma._last_fg_value = 50
    ma._last_btc_taker_ratio = 0.85
    stale_adv = SimpleNamespace(
        market_bias="NEUTRAL",
        risk_level="MODERATE",
        recommended_action="HOLD",
        summary="Prior cached summary",
        key_risks=[],
        is_fallback=False,
        ts=1_793_790_000.0,
    )
    ma.gemini_advisor._cached_assessment = stale_adv
    ma.gemini_advisor._last_eval_ts = 1_793_790_000.0
    ma.gemini_advisor.review = lambda **kwargs: stale_adv
    ma.assess_market_advisor(force=False, now=1_793_800_500.0)
    assert ma._last_fg_value == 50
    assert ma._last_btc_taker_ratio == pytest.approx(0.85)

    # 4. load_cached_whale_telemetry picks the freshest snapshot across symbol variants
    cdir_w = tmp_path / "cdir_w"
    cdir_w.mkdir(parents=True, exist_ok=True)
    (cdir_w / "whale_snapshot_BTCUSD.json").write_text(
        json.dumps({"taker_buy_sell_ratio": 0.90, "ts": 1_793_700_000.0}),
        encoding="utf-8",
    )
    (cdir_w / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps({"taker_buy_sell_ratio": 1.32, "ts": 1_793_800_000.0}),
        encoding="utf-8",
    )
    w_loaded = load_cached_whale_telemetry(str(cdir_w), symbols=["BTCUSD"])
    assert w_loaded is not None
    assert w_loaded["taker_buy_sell_ratio"] == pytest.approx(1.32)

    # 5. reconcile_venue_cycles_with_recent_orders trims recent_buys to buys_after when post_sell_new_cycle is True
    kr_cyc = {
        "SUIUSD": {
            "qty": 20.0,
            "cost_usd": 19.2,
            "avg_cost_px": 0.96,
            "entry_price": 0.96,
            "last_buy_price": 0.96,
            "last_sell_price": 0.98,
            "last_exit_kind": "TP",
            "dca_buys": 0,
            "_entry_ts_ms": 1_793_800_000_000,
            "_last_sell_ts_ms": 1_793_750_000_000,
        }
    }
    ro_sui = {
        "last_sell": {"price": 0.98, "qty": 25.0, "ts_ms": 1_793_750_000_000, "actor": "BOT"},
        "recent_buys": [
            {"price": 0.90, "qty": 25.0, "ts_ms": 1_793_700_000_000, "actor": "BOT"},
            {"price": 0.96, "qty": 20.0, "ts_ms": 1_793_800_001_000, "actor": "BOT"},
        ],
        "active_avg_buy_px": 0.96,
        "active_cycle_rem_qty": 20.0,
        "lifo_realized_pnl_usd": -3.5,
    }
    reconcile_venue_cycles_with_recent_orders(
        kr_cyc,
        {},
        {},
        {"kraken": {"SUIUSD": ro_sui}},
        {("kraken", "SUIUSD"): 20.0},
        {"SUIUSD": {"price": 0.97}},
    )
    assert [b["price"] for b in ro_sui["recent_buys"]] == [0.96]

    # 6. gather_system_and_vpn_telemetry excludes processes that already recovered to high == 0 even if alerted is still True
    ws_rec = tmp_path / "ws_rec"
    ws_rec.mkdir(parents=True, exist_ok=True)
    (ws_rec / ".resource_watchdog_state.json").write_text(
        json.dumps(
            {
                "last_run": time.time() - 30.0,
                "processes": {"101:1": {"pid": 101, "command": "python macro_analyzer.py"}},
                "tracked": {"101:1": {"high": 0, "normal": 1, "alerted": True, "last_reasons": ["CPU 65.0%"]}},
            }
        ),
        encoding="utf-8",
    )
    telem_rec = gather_system_and_vpn_telemetry(str(ws_rec), base_resources={})
    assert "resource_watchdog_offenders" not in telem_rec


def test_iter17_post_rth_equity_cache_preservation_horizon_deltas_and_guard_pattern(tmp_path):
    import json
    import time
    from intelligence.sentiment.collectors.microstructure_collector import (
        enrich_24h_and_microstructure,
        load_symbol_price_series,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    cdir = tmp_path / "cachedb"
    pdir = cdir / "prices"
    pdir.mkdir(parents=True, exist_ok=True)
    now_s = time.time()
    close_ms = int((now_s - 3600.0) * 1000)
    bars_5d = [
        [close_ms - 86400_000, 195.0, 1000.0],
        [close_ms - 3600_000, 202.0, 1200.0],
        [close_ms - 1800_000, 201.0, 1100.0],
        [close_ms - 900_000, 200.5, 900.0],
        [close_ms, 199.7731, 0.0],
    ]
    corrupted_24h = [*bars_5d[:-1], [close_ms, 198.55, 0.0]]
    (pdir / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": corrupted_24h}}),
        encoding="utf-8",
    )
    (pdir / "cache_365_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": bars_5d}}),
        encoding="utf-8",
    )

    s24_healed, _ = load_symbol_price_series(str(cdir), "NVDA_US_EQ")
    assert s24_healed[-1][1] == pytest.approx(199.7731)

    # Post-RTH price move within <=0.5% (199.19 vs 199.7731 = 0.29%) updates 15m/30m/1h/24h deltas while preserving RTH slopes
    pt = {
        "NVDA_US_EQ": {
            "price": 199.19,
            "slope_15m": -0.05,
            "kalman_vel": -0.01,
        }
    }
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt, {}, {}, {}, now_ts=now_s)
    nvda_info = pt["NVDA_US_EQ"]
    assert nvda_info["chg_15m_pct"] == pytest.approx(((199.19 - 200.5) / 200.5) * 100.0, abs=0.02)
    assert nvda_info["chg_30m_pct"] == pytest.approx(((199.19 - 201.0) / 201.0) * 100.0, abs=0.02)
    assert nvda_info["chg_1h_pct"] == pytest.approx(((199.19 - 202.0) / 202.0) * 100.0, abs=0.02)
    assert nvda_info["chg_24h_pct"] == pytest.approx(((199.19 - 195.0) / 195.0) * 100.0, abs=0.02)
    assert nvda_info["slope_15m"] == pytest.approx(-0.05)

    (cdir / "venue_balances_state.json").write_text(
        json.dumps({"pos:t212:NVDA_US_EQ": {"price": 199.19, "ts": now_s - 30.0}}),
        encoding="utf-8",
    )
    pre_nvda = collect_pretrade_telemetry("NVDA_US_EQ", str(cdir))
    s24_str = pre_nvda["24h Multi-Horizon Price Evolution"]
    assert f"15m={nvda_info['chg_15m_pct']:+.2f}%" in s24_str
    assert f"1h={nvda_info['chg_1h_pct']:+.2f}%" in s24_str
    assert f"24h={nvda_info['chg_24h_pct']:+.2f}%" in s24_str

    # Guard regex does not match across line from [WARNING] to 'risk guards.']'
    ldir = tmp_path / "logs"
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "auto_technical_repair.log").write_text(
        "2026-11-05 03:37:48,420 [WARNING] In-place /compact for topic 'auto_technical_repair' failed: "
        "Command '['/home/predut/.local/bin/agy-bot', '-p', '/compact Summarize prior turns, preserving active risk guards.']' timed out\n"
        "2026-11-05 03:38:00,000 [INFO] [ORDER_GUARD] HARD_VETO symbol=TAOUSDC reason=max_exposure\n",
        encoding="utf-8",
    )
    anom = harvest_log_anomalies_from_dirs([str(ldir)], 999999)
    assert anom.error_count == 0
    assert anom.warning_count == 1
    assert len(anom.guard_interventions) == 1
    assert "ORDER_GUARD" in anom.guard_interventions[0]


def test_iter93_7d_archive_over_watcher_post_rth_kalman_preservation_ref_ob_scaling_and_staged_restore(tmp_path):
    """Iter 93:
    1. enrich_24h_and_microstructure upgrades >=5d arch_vals_7d (including live ob_mid_m chg_7d_pct) even when
       enrich_cross_venue_trends pre-populated chg_7d_pct from raw watcher_price_items, and a second
       enrich_cross_venue_trends pass does not overwrite it or clobber post-RTH recomputed kalman_vel.
    2. collect_pretrade_telemetry preserves >=5d arch_vals_7d (with live cur_px_est appended) over raw
       cache_watcher_all_price.json without cur_px_est.
    3. enrich_24h_and_microstructure scales stale >1% USD reference microstructure orderbooks (e.g. HYPEUSD)
       against live cur_p_items and recalculates p_sib geometry.
    4. GeminiClient._restore_newly_dirtied_tracked_files falls back per-file (checkout HEAD / reset HEAD)
       when batch git checkout HEAD fails on a staged unborn file.
    """
    import subprocess
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_cross_venue_trends
    from intelligence.sentiment.gemini_client import GeminiClient

    cdir = tmp_path / "cachedb"
    pdir = cdir / "prices"
    pdir.mkdir(parents=True, exist_ok=True)
    now_s = 1_793_800_000.0
    now_ms = int(now_s * 1000)

    # 1 & 2. 6-day watcher series ending at 105.0, while live ob_mid_price / cur_p_items is 110.0
    watcher_6d = [
        [now_ms - 6 * 86400 * 1000, 100.0],
        [now_ms - 3 * 86400 * 1000, 102.0],
        [now_ms - 120 * 1000, 105.0],
    ]
    (cdir / "cache_watcher_all_price.json").write_text(
        json.dumps({"items": {"SOLUSDC": watcher_6d}}),
        encoding="utf-8",
    )
    (pdir / "cache_365_price_SOLUSDC.json").write_text(
        json.dumps({"items": {"SOLUSDC": watcher_6d}}),
        encoding="utf-8",
    )
    (pdir / "cache_24_price_SOLUSDC.json").write_text(
        json.dumps({"items": {"SOLUSDC": [[now_ms - 3600 * 1000, 104.0], [now_ms - 60 * 1000, 105.0]]}}),
        encoding="utf-8",
    )
    (cdir / "cache_current_price.json").write_text(
        json.dumps({"items": {"SOLUSDC": {"price": 110.0, "ts": now_s}, "HYPEUSD": {"price": 40.50, "ts": now_s}}}),
        encoding="utf-8",
    )
    (cdir / "orderbook_depth_SOLUSDC.json").write_text(
        json.dumps({"symbol": "SOLUSDC", "mid_price": 110.0, "best_bid": 109.95, "best_ask": 110.05, "spread_bps": 9.1, "bid_depth_usd": 50000.0, "ask_depth_usd": 50000.0, "ts": now_s}),
        encoding="utf-8",
    )
    # Hyperliquid orderbook for HYPEUSDC + Stale >1% Kraken reference orderbook for HYPEUSD (38.80 vs live 40.50)
    (cdir / "orderbook_depth_HYPEUSDC.json").write_text(
        json.dumps({"symbol": "HYPEUSDC", "mid_price": 40.52, "best_bid": 40.50, "best_ask": 40.54, "spread_bps": 9.9, "bid_depth_usd": 150000.0, "ask_depth_usd": 150000.0, "ts": now_s}),
        encoding="utf-8",
    )
    (cdir / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps({"symbol": "HYPEUSD", "mid_price": 38.80, "best_bid": 38.75, "best_ask": 38.85, "bid_wall_px": 38.00, "ask_wall_px": 39.50, "spread_bps": 25.8, "bid_depth_usd": 25000.0, "ask_depth_usd": 25000.0, "ts": now_s - 600.0}),
        encoding="utf-8",
    )

    pt = {
        "SOLUSDC": {"price": 110.0},
        "HYPEUSDC": {"price": 40.52, "low_24h": 39.00, "high_24h": 41.00, "pos_24h_pct": 76.0},
    }
    micro = {}
    enrich_cross_venue_trends(pt, {}, {}, {}, {}, {}, {"SOLUSDC": watcher_6d})
    assert pt["SOLUSDC"]["chg_7d_pct"] == pytest.approx(5.0)

    cur_p_items = {"SOLUSDC": {"price": 110.0, "ts": now_s}, "HYPEUSD": {"price": 40.50, "ts": now_s}}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt, micro, cur_p_items, {}, now_ts=now_s)
    # Upgraded to include live 110.0 (+10.0% over 100.0)
    assert pt["SOLUSDC"]["chg_7d_pct"] == pytest.approx(10.0)
    # Reference symbol HYPEUSD scaled from 38.80 to live 40.50 and geometry recalculated
    assert micro["HYPEUSD"]["ob_mid_price"] == pytest.approx(40.50)
    assert pt["HYPEUSD"]["price"] == pytest.approx(40.50)
    assert pt["HYPEUSD"]["pos_24h_pct"] == pytest.approx(75.0)

    # Second enrich_cross_venue_trends pass preservesSOLUSDC +10.0% and seeded post-RTH kalman_vel
    pt["NVDA_US_EQ"] = {"price": 202.0, "kalman_vel": 0.145, "kalman_vel_std": 0.04, "kalman_trend": 1}
    enrich_cross_venue_trends(
        pt,
        {},
        {"NVDA_US_EQ": {"kalman_vel": -0.02, "kalman_vel_std": 0.05, "kalman_trend": -1}},
        {},
        {},
        {},
        {"SOLUSDC": watcher_6d},
    )
    assert pt["SOLUSDC"]["chg_7d_pct"] == pytest.approx(10.0)
    assert pt["NVDA_US_EQ"]["kalman_vel"] == pytest.approx(0.145)
    assert pt["NVDA_US_EQ"]["kalman_trend"] == 1

    pre_sol = collect_pretrade_telemetry("SOLUSDC", str(cdir))
    assert "7d=+10.00%" in pre_sol["7d Price Range & Swing Structure"]

    # 4. GeminiClient._restore_newly_dirtied_tracked_files handles staged unborn file + modified tracked file
    repo_dir = tmp_path / "git_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo_dir), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True, capture_output=True)
    tracked = repo_dir / "tracked.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "tracked.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

    pre_dirty = GeminiClient._snapshot_dirty_tracked_files(str(repo_dir))
    assert pre_dirty == set()
    tracked.write_text("x = 999\n", encoding="utf-8")
    unborn = repo_dir / "unborn_staged.py"
    unborn.write_text("y = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "unborn_staged.py"], check=True, capture_output=True)

    GeminiClient._restore_newly_dirtied_tracked_files(str(repo_dir), pre_dirty)
    assert tracked.read_text(encoding="utf-8") == "x = 1\n"
    assert GeminiClient._snapshot_dirty_tracked_files(str(repo_dir)) == set()


def test_iter18_pretrade_btc_7d_archive_precedence_and_post_rth_vp_invariance(tmp_path):
    """Iter 18:
    1. collect_pretrade_telemetry preserves >=5d BTC 7d archive series (btc_vals_7d) over a shorter 6.5d
       cache_watcher_all_price.json entry when computing vs_btc(..., 7d=...), and accepts positional cachedb_dir.
    2. _load_stitched_7d_ohlc_closes stitches 240m bars when 60m bars span 5.0d..6.75d (e.g. 6.25d).
    3. Post-RTH equity overrides preserve RTH Session Volume Profile anchors (poc_px, val_px, vah_px)
       instead of shifting closing-bar volume to the overnight price.
    """
    from types import SimpleNamespace
    from intelligence.sentiment.collectors.microstructure_collector import (
        _load_stitched_7d_ohlc_closes,
        enrich_24h_and_microstructure,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry

    cdir = tmp_path / "cachedb"
    pdir = cdir / "prices"
    pdir.mkdir(parents=True, exist_ok=True)
    now_s = 1_793_800_000.0
    now_ms = int(now_s * 1000)

    # 1. BTC 7.0d archive: 79000 -> 80580 (+2.00%), while 6.5d watcher starts at 80180 (+0.50%)
    btc_7d_arch = [
        [now_ms - 7 * 86400 * 1000, 79000.0],
        [now_ms - 6 * 86400 * 1000, 80180.0],
        [now_ms - 60 * 1000, 80580.0],
    ]
    btc_w_65d = [
        [now_ms - int(6.5 * 86400 * 1000), 80180.0],
        [now_ms - 60 * 1000, 80580.0],
    ]
    alt_7d_arch = [
        [now_ms - 7 * 86400 * 1000, 100.0],
        [now_ms - 60 * 1000, 105.0],
    ]
    btc_24h = [[now_ms - i * 3600 * 1000, 80000.0 + i * 10.0] for i in range(24, -1, -1)]
    btc_24h[-1][1] = 80580.0
    alt_24h = [[now_ms - i * 3600 * 1000, 100.0 + i * 0.1] for i in range(24, -1, -1)]
    alt_24h[-1][1] = 105.0

    (pdir / "cache_365_price_BTCUSDC.json").write_text(json.dumps({"items": {"BTCUSDC": btc_7d_arch}}), encoding="utf-8")
    (pdir / "cache_24_price_BTCUSDC.json").write_text(json.dumps({"items": {"BTCUSDC": btc_24h}}), encoding="utf-8")
    (pdir / "cache_365_price_TAOUSDC.json").write_text(json.dumps({"items": {"TAOUSDC": alt_7d_arch}}), encoding="utf-8")
    (pdir / "cache_24_price_TAOUSDC.json").write_text(json.dumps({"items": {"TAOUSDC": alt_24h}}), encoding="utf-8")
    (cdir / "cache_watcher_all_price.json").write_text(json.dumps({"items": {"BTCUSDC": btc_w_65d}}), encoding="utf-8")
    (cdir / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": {"price": 80580.0, "ts": now_s}, "TAOUSDC": {"price": 105.0, "ts": now_s}}}),
        encoding="utf-8",
    )

    pre_tao = collect_pretrade_telemetry("TAOUSDC", str(cdir))
    # TAO 7d = +5.00%, BTC 7d (from 7.0d archive) = +2.00% -> vs_btc 7d = +3.00% (not +4.50% from 6.5d watcher)
    assert "7d=+3.00%" in pre_tao["Relative Strength vs BTC (24h Benchmark)"]

    # 2. _load_stitched_7d_ohlc_closes stitches 240m bars when 60m bars span 6.25d (<6.75d)
    def _mock_ohlc(sym: str, interval: int):
        if interval == 60:
            ts_60 = [now_s - i * 3600.0 for i in range(150, -1, -1)]  # 6.25d span, starts at 90.0
            cl_60 = [90.0 + (150 - i) * 0.01 for i in range(150, -1, -1)]
            return SimpleNamespace(timestamps=ts_60, closes=cl_60)
        ts_240 = [now_s - 7 * 86400.0, now_s - 6.5 * 86400.0]
        cl_240 = [78.0, 82.0]
        return SimpleNamespace(timestamps=ts_240, closes=cl_240)

    stitched = _load_stitched_7d_ohlc_closes(_mock_ohlc, "HYPEUSDC", now_s)
    assert stitched[0] == pytest.approx(78.0)

    # 3. Post-RTH equity override (>0.5% divergence) preserves RTH POC/VAL/VAH anchors
    rth_close_ms = now_ms - 4 * 3600 * 1000
    eq_bars = []
    for i in range(78):
        t_b = rth_close_ms - (77 - i) * 300 * 1000
        px_b = 202.50 if i < 77 else 200.50
        vol_b = 100_000.0 if i < 77 else 50_000_000.0  # huge closing auction volume at 200.50
        eq_bars.append([t_b, px_b, vol_b])
    (pdir / "cache_24_price_NVDA_US_EQ.json").write_text(json.dumps({"items": {"NVDA_US_EQ": eq_bars}}), encoding="utf-8")
    (pdir / "cache_365_price_NVDA_US_EQ.json").write_text(json.dumps({"items": {"NVDA_US_EQ": eq_bars}}), encoding="utf-8")
    (cdir / "venue_balances_state.json").write_text(
        json.dumps({"pos:t212:NVDA_US_EQ": {"price": 198.90, "ts": now_s - 10.0}}),
        encoding="utf-8",
    )

    pt_eq1 = {"NVDA_US_EQ": {"price": 198.90}}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_eq1, {}, {}, {}, now_ts=now_s)
    pt_eq2 = {"NVDA_US_EQ": {"price": 199.40}}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_eq2, {}, {}, {}, now_ts=now_s)
    assert pt_eq1["NVDA_US_EQ"]["poc_px"] == pytest.approx(pt_eq2["NVDA_US_EQ"]["poc_px"])
    assert pt_eq1["NVDA_US_EQ"]["val_px"] == pytest.approx(pt_eq2["NVDA_US_EQ"]["val_px"])
    assert pt_eq1["NVDA_US_EQ"]["vah_px"] == pytest.approx(pt_eq2["NVDA_US_EQ"]["vah_px"])

    pre_eq = collect_pretrade_telemetry("NVDA_US_EQ", str(cdir))
    assert f"POC=${pt_eq1['NVDA_US_EQ']['poc_px']}" in pre_eq["24h Volume Profile (POC / Value Area)"]


def test_iter94_live_flow_contradictory_posture_guard_multi_venue_spot_isolation_and_short_trend_remove(tmp_path, monkeypatch):
    """Verify:
    1. enrich_sentiment_advisor_live_flow_divergence does not misclassify >=15m-old BULLISH/ACCUMULATE with cited_tr<=0.65 as is_bear_posture (emitting LIVE_FLOW_RECOVERY instead of contradictory LIVE_BUY_REVERSAL vs ACCUMULATE), nor BEARISH/DEFENSIVE with cited_tr>=1.35 as is_bull_posture (emitting LIVE_FLOW_FADE instead of contradictory LIVE_SELL_REVERSAL vs DEFENSIVE).
    2. OrderbookDepthCollector.fetch does not fall back to fapi.binance.com perpetual futures when api.binance.com spot fails transiently on a multi-venue symbol like TAOUSDC.
    3. CacheFactory.remove('ShortTrend') only clears _short_trend_instance when shutdown succeeds.
    """
    from types import SimpleNamespace
    import cacheManager as cm
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        enrich_sentiment_advisor_live_flow_divergence,
    )

    # 1a. >=15m-old BULLISH/ACCUMULATE with key_risks citing overall 0.60 and strong live buying -> LIVE_FLOW_RECOVERY (not LIVE_BUY_REVERSAL vs ACCUMULATE)
    sa_bull = {
        "market_bias": "BULLISH",
        "recommended_action": "ACCUMULATE",
        "age_min": 18.0,
        "summary": "BTC orderbook depth is bid-dominated (1h taker ratio 1.03).",
        "key_risks": ["Sub-1.0 short-term taker ratios (overall 0.60, 5m 0.67) show selling into bids."],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_bull,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.32, "cvd_15m_usd": 6_200_000.0}},
    )
    assert sa_bull["live_flow_divergence"].startswith("LIVE_FLOW_RECOVERY")
    assert "LIVE_BUY_REVERSAL" not in sa_bull["live_flow_divergence"]

    # 1b. >=15m-old BEARISH/DEFENSIVE with key_risks citing overall 1.40 and strong live selling -> LIVE_FLOW_FADE (not LIVE_SELL_REVERSAL vs DEFENSIVE)
    sa_bear = {
        "market_bias": "BEARISH",
        "recommended_action": "DEFENSIVE",
        "age_min": 18.0,
        "summary": "Macro pressure remains elevated.",
        "key_risks": ["Short-term bounce taker ratio 1.40 may fade rapidly."],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_bear,
        {"BTCUSDC": {"taker_buy_sell_ratio": 0.68, "cvd_15m_usd": -6_500_000.0}},
    )
    assert sa_bear["live_flow_divergence"].startswith("LIVE_FLOW_FADE")
    assert "LIVE_SELL_REVERSAL" not in sa_bear["live_flow_divergence"]

    # 2. OrderbookDepthCollector.fetch on multi-venue TAOUSDC does not query fapi.binance.com when api.binance.com fails
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), cache_ttl_sec=10.0, disk_ttl_sec=60.0)
    ob_col.record_snapshot(
        OrderbookSnapshot(
            symbol="TAOUSDC",
            mid_price=235.75,
            bid_depth_usd=169765.0,
            ask_depth_usd=101824.0,
            imbalance_ratio=0.625,
            largest_bid_wall_usd=30224.0,
            largest_bid_wall_price=235.3,
            largest_ask_wall_usd=18009.0,
            largest_ask_wall_price=236.5,
            ts=1000.0,
            best_bid=235.7,
            best_ask=235.8,
            spread_bps=4.24,
        ),
        save_to_disk=True,
    )
    called_urls = []

    def _fake_spot_down_fapi_up(url, *args, **kwargs):
        called_urls.append(url)
        if "api.binance.com" in url:
            raise RuntimeError("spot timeout")
        return {"bids": [["235.0", "10000.0"]], "asks": [["236.0", "10000.0"]]}

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_spot_down_fapi_up)
    snap_tao = ob_col.fetch("TAOUSDC", allow_network=True, now=1100.0)
    assert snap_tao is not None
    assert snap_tao.bid_depth_usd == 169765.0
    assert all("fapi.binance.com" not in u for u in called_urls)
    assert "TAOUSDC" not in ob_col._preferred_url

    # 3. CacheFactory.remove('ShortTrend') preserves _short_trend_instance if shutdown returns False, and clears when True
    prev_st = cm._short_trend_instance
    try:
        state = {"ok": False}
        fake_st = SimpleNamespace(shutdown=lambda timeout=5.0: state["ok"])
        cm._short_trend_instance = fake_st
        cm.CacheFactory._sync_started.add("ShortTrend")
        assert cm.CacheFactory.remove("ShortTrend", timeout=0.01) is False
        assert cm._short_trend_instance is fake_st
        assert "ShortTrend" not in cm.CacheFactory._sync_started

        state["ok"] = True
        assert cm.CacheFactory.remove("ShortTrend", timeout=0.01) is True
        assert cm._short_trend_instance is None
    finally:
        cm._short_trend_instance = prev_st

    # 4. canonical_base_asset strips TUSD/BUSD suffixes while preserving bare stablecoin tickers
    from cachepaths import canonical_base_asset, normalize_cache_base_asset
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry

    assert canonical_base_asset("SOLTUSD") == "SOL"
    assert canonical_base_asset("SOLBUSD") == "SOL"
    assert canonical_base_asset("MSTRBTUSD") == "MSTRB"
    assert canonical_base_asset("PYUSDUSDT") == "PYUSD"
    assert canonical_base_asset("XBTUSD") == "BTC"
    assert canonical_base_asset("ARBUSD") == "ARB"
    assert canonical_base_asset("DOTUSD") == "DOT"
    assert canonical_base_asset("BATUSD") == "BAT"
    assert canonical_base_asset("HOTUSD") == "HOT"
    assert canonical_base_asset("NOTUSD") == "NOT"
    assert canonical_base_asset("ACTUSD") == "ACT"
    assert canonical_base_asset("CRBUSD") == "CRB"
    assert canonical_base_asset("SUBUSD") == "SUB"
    assert canonical_base_asset("USDTUSD") == "USDT"
    assert canonical_base_asset("XAUTUSD") == "XAUT"
    from cachepaths import split_symbol_assets
    assert split_symbol_assets("XBTUSD") == ("XBT", "USD")
    assert split_symbol_assets("ARBUSD") == ("ARB", "USD")
    assert split_symbol_assets("XAUTUSD") == ("XAUT", "USD")
    assert split_symbol_assets("SPCXx.TUSD") == ("SPCX", "USD")
    assert split_symbol_assets("SOLTUSD") == ("SOL", "TUSD")
    assert split_symbol_assets("SOLBUSD") == ("SOL", "BUSD")
    assert split_symbol_assets("DOTTUSD") == ("DOT", "TUSD")
    assert split_symbol_assets("ARBBUSD") == ("ARB", "BUSD")
    for bare_st in ("PYUSD", "RLUSD", "TUSD", "BUSD", "FDUSD", "USD1"):
        assert normalize_cache_base_asset(bare_st) == bare_st
        assert canonical_base_asset(bare_st) == bare_st

    mb_col = MarketBreadthCollector()
    mb_snap = mb_col.parse_tickers(
        [
            {"symbol": "SOLUSDT", "priceChangePercent": "12.0", "quoteVolume": "25000000"},
            {"symbol": "SOLTUSD", "priceChangePercent": "11.8", "quoteVolume": "20000000"},
            {"symbol": "ETHUSDT", "priceChangePercent": "4.0", "quoteVolume": "19000000"},
            {"symbol": "BTCUSDT", "priceChangePercent": "2.0", "quoteVolume": "35000000"},
            {"symbol": "ADAUSDT", "priceChangePercent": "-1.0", "quoteVolume": "13000000"},
            {"symbol": "XRPUSDT", "priceChangePercent": "-2.0", "quoteVolume": "14000000"},
        ]
    )
    assert mb_snap is not None
    gainer_bases = [canonical_base_asset(sym) for sym, _ in mb_snap.top_gainers]
    assert gainer_bases.count("SOL") == 1

    # 5. collect_pretrade_telemetry appends live BTC price when eff_btc_vals_7d falls back to cache_watcher_all_price.json
    pt_dir = tmp_path / "pt_btc7d"
    pt_dir.mkdir(parents=True, exist_ok=True)
    now_ms = int(1776000000 * 1000)
    (pt_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"SOLUSDC": [now_ms, 110.0], "BTCUSDC": [now_ms, 88000.0]}}),
        encoding="utf-8",
    )
    # 24h series for SOL (100 -> 110, +10%) and BTC (80000 -> 88000, +10%)
    sol_24h = [[now_ms - (24 - i) * 3600 * 1000, 100.0 + (10.0 * i / 24.0)] for i in range(25)]
    btc_24h = [[now_ms - (24 - i) * 3600 * 1000, 80000.0 + (8000.0 * i / 24.0)] for i in range(25)]
    (pt_dir / "cache_24_price_SOLUSDC.json").write_text(json.dumps({"items": {"SOLUSDC": sol_24h}}), encoding="utf-8")
    (pt_dir / "cache_24_price_BTCUSDC.json").write_text(json.dumps({"items": {"BTCUSDC": btc_24h}}), encoding="utf-8")
    # 7d watcher series where BTC watcher last point is stale at 80000 (0% 7d change without live BTC append, +10% with 88000)
    w_sol = [[now_ms - 6 * 86400 * 1000, 100.0], [now_ms - 3600 * 1000, 105.0]]
    w_btc = [[now_ms - 6 * 86400 * 1000, 80000.0], [now_ms - 3600 * 1000, 80000.0]]
    (pt_dir / "cache_watcher_all_price.json").write_text(
        json.dumps({"items": {"SOLUSDC": w_sol, "BTCUSDC": w_btc}}),
        encoding="utf-8",
    )
    pt_out = collect_pretrade_telemetry("SOLUSDC", cachedb_dir=str(pt_dir))
    assert "7d=+0.00%)" in pt_out["Relative Strength vs BTC (24h Benchmark)"]

    # 6. enrich_24h_and_microstructure rescales stale (>120s) reference USD orderbook via sibling p_ref price
    ms_dir = tmp_path / "ms_ref_scale"
    ms_dir.mkdir(parents=True, exist_ok=True)
    import time as _t_mod

    now_s = _t_mod.time()
    (ms_dir / "orderbook_depth_HYPEUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "HYPEUSDC",
                "mid_price": 42.0,
                "best_bid": 41.99,
                "best_ask": 42.01,
                "bid_depth_usd": 80000.0,
                "ask_depth_usd": 75000.0,
                "imbalance_ratio": 0.516,
                "largest_bid_wall_usd": 20000.0,
                "largest_bid_wall_price": 41.80,
                "largest_ask_wall_usd": 18000.0,
                "largest_ask_wall_price": 42.20,
                "spread_bps": 4.76,
                "ts": now_s - 5.0,
            }
        ),
        encoding="utf-8",
    )
    (ms_dir / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps(
            {
                "symbol": "HYPEUSD",
                "mid_price": 40.0,
                "best_bid": 39.98,
                "best_ask": 40.02,
                "bid_depth_usd": 50000.0,
                "ask_depth_usd": 45000.0,
                "imbalance_ratio": 0.526,
                "largest_bid_wall_usd": 12000.0,
                "largest_bid_wall_price": 39.80,
                "largest_ask_wall_usd": 11000.0,
                "largest_ask_wall_price": 40.20,
                "spread_bps": 10.0,
                "ts": now_s - 300.0,
            }
        ),
        encoding="utf-8",
    )
    p_trends = {"HYPEUSDC": {"price": 42.0, "low_24h": 38.0, "high_24h": 43.0}}
    micro_out = {}
    enrich_24h_and_microstructure(str(ms_dir), str(tmp_path), p_trends, micro_out, {}, {}, now_s)
    assert micro_out["HYPEUSD"]["ob_mid_price"] == pytest.approx(42.0)
    assert p_trends["HYPEUSD"]["price"] == pytest.approx(42.0)

    # 7. _refresh_stale_t212_cached_positions updates stale pos:t212:* prices and collect_pretrade_telemetry refreshes microstructure
    from intelligence.sentiment.collectors import microstructure_collector as ms_col_mod
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as pt_col_mod
    from providers import t212_provider as t212_prov_mod

    vb_file = ms_dir / "venue_balances_state.json"
    vb_data = {
        "pos:t212:NVDA_US_EQ": {
            "symbol": "NVDA_US_EQ",
            "free": 3.0,
            "price": 199.33,
            "value_usd": 597.99,
            "ts": now_s - 600.0,
        }
    }
    vb_file.write_text(json.dumps(vb_data), encoding="utf-8")

    class _MockT212Client:
        def get_portfolio(self):
            return [{"ticker": "NVDA_US_EQ", "currentPrice": 199.06}]

    class _MockT212Provider:
        def _client(self):
            return _MockT212Client()

    monkeypatch.setattr(t212_prov_mod, "T212Provider", _MockT212Provider)
    refreshed = ms_col_mod._refresh_stale_t212_cached_positions(vb_data, str(vb_file), now_s)
    assert refreshed is True
    assert vb_data["pos:t212:NVDA_US_EQ"]["price"] == pytest.approx(199.06)
    assert vb_data["pos:t212:NVDA_US_EQ"]["value_usd"] == pytest.approx(597.18)
    assert vb_data["pos:t212:NVDA_US_EQ"]["ts"] == pytest.approx(now_s)

    refreshed_syms = []
    monkeypatch.setattr(
        pt_col_mod,
        "ensure_fresh_symbol_microstructure",
        lambda cdir, sym, **kw: refreshed_syms.append(sym),
    )
    pt_col_mod.collect_pretrade_telemetry("HYPEUSDC", cachedb_dir=str(ms_dir))
    assert "HYPEUSDC" in refreshed_syms
    assert "HYPEUSD" in refreshed_syms


def test_iter96_known_t_b_bases_fdusd_venue_fees_retry_queue_venue_and_crypto_scope_microstructure(tmp_path):
    from cachepaths import (
        cache_symbol_variants,
        compatible_quote_suffixes,
        normalize_cache_base_asset,
        split_symbol_assets,
    )
    from intelligence.sentiment.collectors.portfolio_collector import harvest_bot_runtime_states
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import apply_asset_scope_filter
    from intelligence.sentiment.evaluators.microstructure_evaluator import _infer_symbol_venue_and_fees_bps

    # 1. Known 4+ letter bases ending in T or B paired with USD are not truncated by TUSD/BUSD
    assert normalize_cache_base_asset("SHIBUSD") == "SHIB"
    assert split_symbol_assets("SHIBUSD") == ("SHIB", "USD")
    assert normalize_cache_base_asset("SHIBBUSD") == "SHIB"
    assert split_symbol_assets("SHIBBUSD") == ("SHIB", "BUSD")
    assert normalize_cache_base_asset("PNUTUSD") == "PNUT"
    assert split_symbol_assets("PNUTUSD") == ("PNUT", "USD")
    assert normalize_cache_base_asset("BRETTUSD") == "BRETT"
    assert split_symbol_assets("BRETTUSD") == ("BRETT", "USD")
    assert normalize_cache_base_asset("BRETTTUSD") == "BRETT"
    assert split_symbol_assets("BRETTTUSD") == ("BRETT", "TUSD")
    assert normalize_cache_base_asset("ZENTUSD") == "ZENT"
    assert split_symbol_assets("ZENTUSD") == ("ZENT", "USD")
    assert normalize_cache_base_asset("MSFTUSD") == "MSFT"
    assert split_symbol_assets("MSFTUSD") == ("MSFT", "USD")
    assert compatible_quote_suffixes("SOLTUSD") == ("USDC", "USDT", "USD")
    assert compatible_quote_suffixes("SOLBUSD") == ("USDC", "USDT", "USD")
    assert "SOLUSDC" in cache_symbol_variants("SOLTUSD")

    # 2. _infer_symbol_venue_and_fees_bps maps FDUSD/TUSD/BUSD to binance and _EQ/_US_EQ to t212
    assert _infer_symbol_venue_and_fees_bps("BTCFDUSD", None)[0] == "binance"
    assert _infer_symbol_venue_and_fees_bps("SOLTUSD", None)[0] == "binance"
    assert _infer_symbol_venue_and_fees_bps("SOLBUSD", None)[0] == "binance"
    assert _infer_symbol_venue_and_fees_bps("SHIBUSD", None)[0] == "kraken"
    assert _infer_symbol_venue_and_fees_bps("NVDA_US_EQ", None)[0] == "t212"

    # 3. harvest_bot_runtime_states normalizes retry_queue venue when provider_name is absent
    import time as _t_mod

    now_s = _t_mod.time()
    rq_dir = tmp_path / "rq_test"
    rq_dir.mkdir(parents=True, exist_ok=True)
    (rq_dir / "order_retry_queue.jsonl").write_text(
        json.dumps(
            {
                "symbol": "NVDA_US_EQ",
                "venue": "trading212",
                "side": "BUY",
                "qty": 1.0,
                "lifecycle": "submit_pending",
                "expires_at_ts": now_s + 300.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bot_states = harvest_bot_runtime_states(str(rq_dir), now_s)
    assert len(bot_states["active_retry_queue"]) == 1
    assert bot_states["active_retry_queue"][0]["venue"] == "t212"

    # 4. apply_asset_scope_filter removes _US_EQ from microstructure when scope='crypto'
    price_trends = {"BTCUSDC": {"price": 75000.0}, "NVDA_US_EQ": {"price": 199.06}}
    micro = {"BTCUSDC": {"ob_mid_price": 75000.0}, "NVDA_US_EQ": {"ob_mid_price": 199.06}}
    portfolio = {"positions_by_venue": {"t212": [{"symbol": "NVDA_US_EQ"}]}}
    price_trends, micro = apply_asset_scope_filter(price_trends, micro, portfolio, "crypto", {}, {})
    assert "NVDA_US_EQ" not in micro
    assert "BTCUSDC" in micro

    # 5. collect_pretrade_telemetry prepends vs_btc(7d=...) when 24h Relative Strength vs BTC only has cross_parts
    pt_dir = tmp_path / "pt_rel7d"
    pt_dir.mkdir(parents=True, exist_ok=True)
    sol_24h = [
        [int((now_s - m * 60 - k) * 1000), 150.0 + idx * 0.2]
        for idx, m in enumerate((55, 50, 45, 40))
        for k in (0, 1, 2)
    ] + [[int(now_s * 1000), 152.0]]
    btc_24h = [
        [int((now_s - m * 60 - k) * 1000), 75000.0 + idx * 20.0]
        for idx, m in enumerate((55, 50, 45, 40))
        for k in (0, 1, 2)
    ] + [[int(now_s * 1000), 75100.0]]
    (pt_dir / "cache_24_price_SOLUSDC.json").write_text(json.dumps(sol_24h), encoding="utf-8")
    (pt_dir / "cache_24_price_BTCUSDC.json").write_text(json.dumps(btc_24h), encoding="utf-8")
    (pt_dir / "cache_watcher_all_price.json").write_text(
        json.dumps(
            {
                "items": {
                    "SOLUSDC": [
                        [int((now_s - 6.5 * 86400) * 1000), 140.0],
                        [int((now_s - 25 * 3600) * 1000), 148.0],
                    ],
                    "BTCUSDC": [
                        [int((now_s - 6.5 * 86400) * 1000), 70000.0],
                        [int((now_s - 25 * 3600) * 1000), 74000.0],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    res_pt = collect_pretrade_telemetry("SOLUSDC", cachedb_dir=str(pt_dir))
    rel_line = res_pt.get("Relative Strength vs BTC (24h Benchmark)", "")
    assert rel_line.startswith("vs_btc(7d=")
    assert " | alt_btc_pos_24h=" in rel_line


def test_iter97_prompt_sections_btc_merge_24h_horizons_micro_wall_sync_and_pretrade_sib_rescale(tmp_path):
    import time as _t_mod

    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.microstructure_evaluator import sync_sibling_symbol_telemetry
    from orchestratorOS.lib.technical_repair_ssot import _is_decision_in_scope
    from orchestratorOS.supervisor.prompt_sections import (
        _format_market_prices_section,
        _format_microstructure_section,
    )

    now_s = _t_mod.time()

    # 1. _format_market_prices_section merges BTCUSDC into btc_entry even when partial BTCUSD precedes it,
    #    and emits 24h horizons / 24h_range even when chg_15m_pct is absent
    prices_3a = {
        "BTCUSD": {"price": 75000.0},
        "BTCUSDC": {"price": 75000.0, "chg_1h_pct": 1.0, "chg_24h_pct": 3.0, "chg_7d_pct": 8.0},
        "SOLUSDC": {
            "price": 150.0,
            "chg_1h_pct": 2.5,
            "chg_24h_pct": 5.0,
            "chg_7d_pct": 12.0,
            "low_24h": 142.0,
            "high_24h": 152.0,
        },
    }
    lines_3a = "\n".join(_format_market_prices_section(prices_3a))
    assert "1h=+2.50% 24h=+5.00% 24h_range=[142.0, 152.0]" in lines_3a
    assert "vs_btc(1h=+1.50%, 24h=+2.00%, 7d=+4.00%)" in lines_3a

    # 2. _format_microstructure_section and sync_sibling_symbol_telemetry do not pollute own orderbook
    #    and backfill bid_wall_usd > 0 over 0.0 when no own orderbook exists
    pt_3b = {"SOLUSDC": {"price": 150.0}, "SOLUSD": {"price": 150.0}}
    micro_3b = {
        "SOLUSDC": {"funding_rate": 0.0001, "oi_usd": 5000000.0, "bid_wall_usd": 0.0, "ask_wall_usd": 0.0},
        "SOLUSD": {
            "ob_mid_price": 150.0,
            "best_bid": 149.98,
            "best_ask": 150.02,
            "ob_imbalance": 0.55,
            "bid_depth_usd": 60000.0,
            "ask_depth_usd": 50000.0,
            "bid_wall_px": 148.5,
            "bid_wall_usd": 18000.0,
            "ask_wall_px": 151.5,
            "ask_wall_usd": 16000.0,
        },
    }
    lines_3b = "\n".join(_format_microstructure_section(micro_3b, pt_3b))
    assert "bid_wall=148.5 ($18000.0" in lines_3b
    assert "ask_wall=151.5 ($16000.0" in lines_3b
    sync_sibling_symbol_telemetry(pt_3b, micro_3b)
    assert micro_3b["SOLUSDC"]["bid_wall_usd"] == pytest.approx(18000.0)
    assert micro_3b["SOLUSDC"]["ask_wall_usd"] == pytest.approx(16000.0)

    # 3. collect_pretrade_telemetry rescales stale sibling USD orderbook (>120s old, >1% off) using cur_px_est
    #    and emits standalone vs_btc(7d=...) when 24h vs_btc/cross_parts are absent
    pt_dir = tmp_path / "pt_iter97"
    pt_dir.mkdir(parents=True, exist_ok=True)
    (pt_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"HYPEUSDC": 42.0, "BTCUSDC": 75000.0}}),
        encoding="utf-8",
    )
    (pt_dir / "orderbook_depth_HYPEUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "HYPEUSDC",
                "mid_price": 42.0,
                "best_bid": 41.99,
                "best_ask": 42.01,
                "bid_depth_usd": 90000.0,
                "ask_depth_usd": 85000.0,
                "imbalance_ratio": 0.514,
                "spread_bps": 4.76,
                "ts": now_s - 5.0,
            }
        ),
        encoding="utf-8",
    )
    (pt_dir / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps(
            {
                "symbol": "HYPEUSD",
                "mid_price": 40.0,
                "best_bid": 39.98,
                "best_ask": 40.02,
                "bid_depth_usd": 45000.0,
                "ask_depth_usd": 40000.0,
                "imbalance_ratio": 0.529,
                "spread_bps": 10.0,
                "ts": now_s - 300.0,
            }
        ),
        encoding="utf-8",
    )
    (pt_dir / "cache_watcher_all_price.json").write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSDC": [
                        [int((now_s - 6.5 * 86400) * 1000), 38.0],
                        [int((now_s - 25 * 3600) * 1000), 41.0],
                    ],
                    "BTCUSDC": [
                        [int((now_s - 6.5 * 86400) * 1000), 70000.0],
                        [int((now_s - 25 * 3600) * 1000), 74000.0],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    telem = collect_pretrade_telemetry("HYPEUSDC", cachedb_dir=str(pt_dir))
    assert "Cross-Venue Relative Value (Arbitrage)" in telem
    assert "arb_vs_HYPEUSD=+0.0bps" in telem["Cross-Venue Relative Value (Arbitrage)"]
    assert telem.get("Relative Strength vs BTC (24h Benchmark)", "").startswith("vs_btc(7d=")

    # 4. _is_decision_in_scope rejects lowercase-x xStock symbols and TUSD in equities scope
    assert _is_decision_in_scope({"decision": "BUY", "symbol": "SPCXx", "venue": "kraken"}, "crypto") is False
    assert _is_decision_in_scope({"decision": "BUY", "symbol": "SPCXx/USD", "venue": "kraken"}, "crypto") is False
    assert _is_decision_in_scope({"decision": "BUY", "symbol": "SOLTUSD", "venue": "t212"}, "equities") is False
    assert _is_decision_in_scope({"decision": "BUY", "symbol": "NVDA_US_EQ", "venue": "t212"}, "equities") is True


def test_iter20_price_series_memo_and_fallback_ohlc_provider_cache(tmp_path, monkeypatch):
    from intelligence.sentiment.collectors import microstructure_collector as mc

    cdir = tmp_path / "memo_cdir"
    cdir.mkdir(parents=True, exist_ok=True)
    p24 = cdir / "cache_24_price_BTCUSDC.json"
    p24.write_text(json.dumps([[1710000000000, 75000.0], [1710003600000, 75500.0]]), encoding="utf-8")

    s24_a, v7d_a = mc.load_symbol_price_series(str(cdir), "BTCUSDC")
    assert len(s24_a) == 2 and s24_a[-1][1] == 75500.0
    # Mutating returned list does not corrupt the memoized copy
    s24_a.append([1710007200000, 99999.0])
    s24_b, _ = mc.load_symbol_price_series(str(cdir), "BTCUSDC")
    assert len(s24_b) == 2 and s24_b[-1][1] == 75500.0

    # Updating file on disk automatically invalidates the stat-signature memo
    p24.write_text(
        json.dumps([[1710000000000, 75000.0], [1710003600000, 75500.0], [1710007200000, 76200.0]]),
        encoding="utf-8",
    )
    s24_c, _ = mc.load_symbol_price_series(str(cdir), "BTCUSDC")
    assert len(s24_c) == 3 and s24_c[-1][1] == 76200.0

    # _resolve_fallback_ohlc_fn caches provider OHLC callable across calls and _wrap_cached_ohlc_fn caches OHLC responses
    monkeypatch.setenv("MPTRADE_ALLOW_LIVE_BACKFILL", "1")
    fn1 = mc._resolve_fallback_ohlc_fn("HYPEUSDC")
    fn2 = mc._resolve_fallback_ohlc_fn("HYPEUSDC")
    monkeypatch.delenv("MPTRADE_ALLOW_LIVE_BACKFILL", raising=False)
    assert callable(fn1) and fn1 is fn2
    calls = []

    class _DummyOHLC:
        closes = [100.0, 101.0]

    wrapped = mc._wrap_cached_ohlc_fn("dummy_prov", lambda s, iv: (calls.append((s, iv)) or _DummyOHLC()), ttl_sec=60.0)
    r1 = wrapped("TAOUSD", 5)
    r2 = wrapped("TAOUSD", 5)
    assert r1 is r2 and len(calls) == 1


def test_iter21_ohlc_24h_disk_cache_news_macro_relevance_and_dated_log_dedup(tmp_path, monkeypatch):
    from intelligence.macro.lexicon_ssot import (
        has_financial_or_geopolitical_context,
        is_off_topic_general_news_title,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        harvest_macro_and_news_sensors,
    )
    from intelligence.sentiment.evaluators import price_series_evaluator as pse
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # 1. Discrepancy BR: _load_stitched_24h_ohlc_points persists and serves a 240s on-disk 24h OHLC cache
    cdir_ohlc = tmp_path / "ohlc_cache_dir"
    cdir_ohlc.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pse, "cache_path", lambda name: cdir_ohlc / name)
    now_s = time.time()
    base_ts = int(now_s - 20 * 3600)
    ohlc_calls = []

    class _MockOHLC24:
        timestamps = [base_ts + i * 1800 for i in range(41)]
        closes = [100.0 + i * 0.25 for i in range(41)]
        volumes = [15.0 + i for i in range(41)]

    def _mock_ohlc_fn(sym, iv):
        ohlc_calls.append((sym, iv))
        return _MockOHLC24()

    pts1 = pse._load_stitched_24h_ohlc_points(_mock_ohlc_fn, "TAOUSD", now_s - 86400.0, require_positive_volume=True)
    assert len(pts1) == 41 and len(ohlc_calls) == 1
    assert (cdir_ohlc / "cache_ohlc_24h_TAOUSD.json").exists()
    # Second call (even with a non-callable ohlc_fn or in a fresh process) hits the disk cache without calling ohlc_fn
    pse._OHLC_24H_DISK_MEMO.clear()
    pts2 = pse._load_stitched_24h_ohlc_points(None, "TAOUSD", now_s - 86400.0, require_positive_volume=True)
    assert len(pts2) == 41 and len(ohlc_calls) == 1

    # 2. Discrepancy BS: Breaking News Feed ranks financial/macro/geopolitical context above non-macro/tabloid filler
    assert is_off_topic_general_news_title("Mamdani: ‘it’s morning again in America’ after blue wave") is True
    assert is_off_topic_general_news_title("The New York Times Adds 420,000 Digital Subscribers") is True
    assert has_financial_or_geopolitical_context("Asian Stocks to Rise as Oil Slides After US Vote: Markets Wrap") is True
    assert has_financial_or_geopolitical_context("White House panel delays hearing on Fed's Lisa Cook to Dec. 14") is True

    news_cdir = tmp_path / "news_cdir"
    news_cdir.mkdir(parents=True, exist_ok=True)
    (news_cdir / "news_feed_collect.json").write_text(
        json.dumps([
            {
                "title": "Mamdani: ‘it’s morning again in America’ after blue wave",
                "source": "theguardian.com",
                "first_seen_ts": now_s - 10,
                "is_high_severity": False,
                "matched_keywords": [],
            },
            {
                "title": "The New York Times Adds 420,000 Digital Subscribers",
                "source": "nytimes.com",
                "first_seen_ts": now_s - 20,
                "is_high_severity": False,
                "matched_keywords": [],
            },
            {
                "title": "White House panel delays hearing on Fed's Lisa Cook to Dec. 14",
                "source": "cnbc.com",
                "first_seen_ts": now_s - 120,
                "is_high_severity": False,
                "matched_keywords": [],
            },
        ]),
        encoding="utf-8",
    )
    _, b_news = harvest_macro_and_news_sensors(str(news_cdir), now_s)
    assert b_news[0]["title"].startswith("White House panel delays hearing on Fed's Lisa Cook")

    # 3. Discrepancy BT: harvest_log_anomalies_from_dirs deduplicates <stem>.log vs <stem>_YYYY-MM-DD.log and sample errors
    dir_logs = tmp_path / "logs"
    dir_logger = tmp_path / "logger"
    dir_logs.mkdir(parents=True, exist_ok=True)
    dir_logger.mkdir(parents=True, exist_ok=True)
    ts1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s - 300))
    ts2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s - 295))
    ts_ok = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s - 60))
    log_body = "\n".join([
        f"{ts1} ERROR [monitororder.py:420] Error getting open SELL orders: APIError(code=-2015)",
        f"{ts2} ERROR [monitororder.py:420] Error getting open SELL orders: APIError(code=-2015)",
        *(f"{ts_ok} INFO heartbeat ok {i}" for i in range(12)),
    ])
    (dir_logs / "monitororder.log").write_text(log_body, encoding="utf-8")
    (dir_logger / "monitororder_2026-11-05.log").write_text(log_body, encoding="utf-8")
    summary_bt = harvest_log_anomalies_from_dirs([str(dir_logs), str(dir_logger)], lookback_hours=1.0)
    assert len(summary_bt.persistent_errors) == 1
    assert summary_bt.persistent_errors[0]["file"] == "monitororder.log"
    assert summary_bt.persistent_errors[0]["count"] == 2
    assert summary_bt.error_count == 2
    assert len(summary_bt.critical_events) == 1



def test_iter98_tca_front_run_wall_pos_asset_scope_and_trailing_peak_variants():
    import time as _time
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
        parse_venue_cycle_state,
    )

    # 1. enrich_open_limit_orders_tca: order placed at recommended_front_run_px is FRONT_OF_ASK_WALL / FRONT_OF_BID_WALL,
    #    while order placed at wall price is AT_ASK_WALL / AT_BID_WALL with wall_ahead_px / wall_ahead_usd populated.
    pt = {
        "BTCUSDC": {"price": 75000.0, "vol_1h_pct": 1.0},
        "COIN_US_EQ": {"price": 250.0, "vol_1h_pct": 0.4},
    }
    ms = {
        "BTCUSDC": {
            "ob_mid_price": 75000.0,
            "spread_bps": 2.0,
            "best_bid": 74992.5,
            "best_ask": 75007.5,
            "ask_wall_px": 75500.0,
            "ask_wall_usd": 50000.0,
            "bid_wall_px": 74500.0,
            "bid_wall_usd": 40000.0,
        }
    }
    orders = [
        {"symbol": "BTCUSDC", "venue": "binance", "side": "SELL", "price": 75495.0},  # at recommended_front_run_px (front of 75500 wall)
        {"symbol": "BTCUSDC", "venue": "binance", "side": "SELL", "price": 75500.0},  # right at ask wall
        {"symbol": "BTCUSDC", "venue": "binance", "side": "BUY", "price": 74505.0},   # at recommended_front_run_px (front of 74500 wall)
        {"symbol": "BTCUSDC", "venue": "binance", "side": "BUY", "price": 74500.0},   # right at bid wall
    ]
    enrich_open_limit_orders_tca(orders, pt, ms)
    assert orders[0]["wall_queue_pos"] == "FRONT_OF_ASK_WALL"
    assert "recommended_front_run_px" not in orders[0]
    assert orders[1]["wall_queue_pos"] == "AT_ASK_WALL"
    assert orders[1]["ask_wall_ahead_px"] == 75500.0
    assert orders[1]["ask_wall_ahead_usd"] == 50000.0
    assert orders[1]["recommended_front_run_px"] == 75495.0
    assert orders[1]["fill_prob_1h_pct"] < orders[0]["fill_prob_1h_pct"]

    assert orders[2]["wall_queue_pos"] == "FRONT_OF_BID_WALL"
    assert "recommended_front_run_px" not in orders[2]
    assert orders[3]["wall_queue_pos"] == "AT_BID_WALL"
    assert orders[3]["bid_wall_ahead_px"] == 74500.0
    assert orders[3]["bid_wall_ahead_usd"] == 40000.0
    assert orders[3]["recommended_front_run_px"] == 74505.0
    assert orders[3]["fill_prob_1h_pct"] < orders[2]["fill_prob_1h_pct"]

    # 2. compute_enterprise_portfolio_risk_metrics: _resolve_p_metric does not cross-contaminate crypto vs equity variants
    pf_crypto = {
        "tradable_positions": [
            {"venue": "binance", "symbol": "COINUSDC", "value_usd": 1000.0}
        ],
        "venue_quote_cash": {},
        "recent_orders": {},
    }
    pt_mixed = {
        "COIN_US_EQ": {"price": 250.0, "beta_btc_24h": 0.25, "corr_btc_24h": 0.15, "vol_1h_pct": 0.4},
    }
    risk_c = compute_enterprise_portfolio_risk_metrics(pf_crypto, pt_mixed, {})
    # Crypto position should fall back to crypto defaults (beta 1.15, corr 0.78) instead of COIN_US_EQ (beta 0.25, corr 0.15)
    assert risk_c["portfolio_btc_beta"] == pytest.approx(1.15, abs=0.01)

    # 3. parse_venue_cycle_state updates pos_peak when live price exceeds stored pos_peak
    cyc_st = parse_venue_cycle_state(
        {"qty": 10.0, "cost_usd": 350.0, "avg_cost_px": 35.0, "pos_peak": 40.0},
        "HYPE",
        {"HYPEUSDC": {"price": 42.5}},
        avg_decimals=4,
    )
    assert cyc_st["pos_peak"] == 42.5
    assert cyc_st["dd_from_peak_pct"] == 0.0

    # 4. enrich_positions_with_orders_and_cycles reconciles unwarmed trailing_peak using high_24h from quote variant (HYPE -> HYPEUSDC)
    now_ms = int(_time.time() * 1000)
    entry_24h_ago_ms = now_ms - 24 * 3600 * 1000
    t_pos = [
        {
            "venue": "hyperliquid",
            "symbol": "HYPE",
            "quote": "USDC",
            "free_base": 10.0,
            "price": 40.0,
            "value_usd": 400.0,
        }
    ]
    v_ro = {
        "hyperliquid": {
            "HYPE": {
                "last_action_side": "BUY",
                "active_avg_buy_px": 38.0,
                "last_buy": {"price": 38.0, "qty": 10.0, "ts_ms": entry_24h_ago_ms},
                "open_lots": [{"entry_px": 38.0, "qty": 10.0, "ts_ms": entry_24h_ago_ms}],
            }
        }
    }
    hl_cycles = {
        "HYPE": {
            "qty": 10.0,
            "cost_usd": 380.0,
            "avg_cost_px": 38.0,
            "pos_peak": 38.0,
        }
    }
    enrich_positions_with_orders_and_cycles(
        t_pos,
        {"HYPEUSDC": {"price": 40.0, "high_24h": 44.0}},
        {},
        v_ro,
        {},
        {},
        hl_cycles,
        {},
        {},
    )
    assert t_pos[0]["trailing_peak"] == 44.0
    assert hl_cycles["HYPE"]["pos_peak"] == 44.0


def test_iter99_cycle_fallback_entry_ts_peak_hl_flat_snapshot_and_bellwether_micro():
    import time as _time
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
        reconcile_venue_cycles_with_recent_orders,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        build_cross_market_context,
        compute_enterprise_portfolio_risk_metrics,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        reconcile_live_venue_equity_summary,
    )

    # 1. reconcile_venue_cycles_with_recent_orders preserves _entry_ts_ms in cyc_sell_ts_map
    #    so _populate_cycle_fallback_recent_order and enrich_positions_with_orders_and_cycles
    #    can reconcile unwarmed pos_peak / trailing_peak from high_24h even when trade cache is absent.
    now_ms = int(_time.time() * 1000)
    entry_24h_ago_ms = now_ms - 24 * 3600 * 1000
    hl_cycles = {
        "HYPE": {
            "qty": 10.0,
            "cost_usd": 380.0,
            "avg_cost_px": 38.0,
            "entry_price": 38.0,
            "last_buy_price": 38.0,
            "pos_peak": 38.0,
            "_entry_ts_ms": entry_24h_ago_ms,
        }
    }
    kr_cycles = {
        "TAOUSD": {
            "qty": 2.0,
            "cost_usd": 800.0,
            "avg_cost_px": 400.0,
            "entry_price": 400.0,
            "last_buy_price": 400.0,
            "pos_peak": 400.0,
            "_entry_ts_ms": entry_24h_ago_ms,
        }
    }
    pt = {
        "HYPEUSDC": {"price": 40.0, "high_24h": 45.0},
        "TAOUSD": {"price": 410.0, "high_24h": 440.0},
    }
    cyc_map = reconcile_venue_cycles_with_recent_orders(
        kr_cycles, hl_cycles, {}, {"kraken": {}, "hyperliquid": {}}, {}, pt
    )
    assert cyc_map.get(("hyperliquid:entry", "HYPE")) == entry_24h_ago_ms
    assert cyc_map.get(("kraken:entry", "TAOUSD")) == entry_24h_ago_ms

    t_pos = [
        {
            "venue": "hyperliquid",
            "symbol": "HYPE",
            "quote": "USDC",
            "free_base": 10.0,
            "price": 40.0,
            "value_usd": 400.0,
        }
    ]
    rec_orders = {"HYPE": {"venue": "binance", "last_action_side": "BUY", "lifo_realized_pnl_usd": 12.5}}
    v_ro = {"kraken": {}, "hyperliquid": {}}
    enrich_positions_with_orders_and_cycles(
        t_pos,
        pt,
        {},
        v_ro,
        rec_orders,
        kr_cycles,
        hl_cycles,
        {},
        cyc_map,
    )
    assert t_pos[0]["trailing_peak"] == 45.0
    assert hl_cycles["HYPE"]["pos_peak"] == 45.0
    assert "hyperliquid:HYPE" in rec_orders
    assert rec_orders["hyperliquid:HYPE"]["last_buy"]["ts_ms"] == entry_24h_ago_ms
    # TAOUSD (not in tradable_positions) also gets ro captured and pos_peak reconciled from high_24h
    assert kr_cycles["TAOUSD"]["pos_peak"] == 440.0
    assert rec_orders["TAOUSD"]["last_buy"]["ts_ms"] == entry_24h_ago_ms

    # 2. compute_enterprise_portfolio_risk_metrics deduplicates aliased recent_orders dicts by id()
    shared_ro = {"venue": "binance", "lifo_realized_pnl_usd": 15.0}
    pf_dup = {
        "tradable_positions": [{"venue": "binance", "symbol": "BTCUSDC", "value_usd": 1000.0}],
        "venue_quote_cash": {},
        "recent_orders": {"BTCUSDC": shared_ro, "binance:BTCUSDC": shared_ro},
    }
    risk_dup = compute_enterprise_portfolio_risk_metrics(pf_dup, {"BTCUSDC": {"price": 75000.0}}, {})
    assert risk_dup["subcycle_lifo_realized_pnl_usd"] == 15.0

    # 3. build_cross_market_context prefers compatible quote microstructure and skips _EQ/_US_EQ
    cmc = build_cross_market_context(
        {"COINUSDC": {"price": 250.0, "chg_24h_pct": 3.5}},
        {
            "COIN_US_EQ": {"ob_imbalance": -0.9, "spread_bps": 50.0},
            "COINEUR": {"ob_imbalance": -0.5, "spread_bps": 25.0, "long_liq_px": 200.0},
            "COINUSDT": {"ob_imbalance": 0.42, "spread_bps": 4.0, "long_liq_px": 235.0},
        },
    )
    cb_coin = cmc["crypto_bellwethers"]["COIN"]
    assert cb_coin["ob_imbalance"] == 0.42
    assert cb_coin["long_liq_px"] == 235.0

    # 4. reconcile_live_venue_equity_summary zeroes stale Hyperliquid spot qty/value when position is flat (only cash remains)
    pf_hl_flat = {
        "tradable_positions": [],
        "venue_quote_cash": {"hyperliquid:USDC": {"venue": "hyperliquid", "free_usd": 520.0}},
        "hyperliquid_readonly": {
            "qty": 12.5,
            "value_usd": 500.0,
            "cash_usd": 20.0,
            "spot_positions": [{"coin": "HYPE", "qty": 12.5, "value_usd": 500.0, "cash_usd": 20.0, "snapshot_age_min": 45.0}],
        },
    }
    reconcile_live_venue_equity_summary(pf_hl_flat)
    hl_ro_after = pf_hl_flat["hyperliquid_readonly"]
    assert hl_ro_after["live_reconciled_spot_usd"] == 520.0
    assert hl_ro_after["qty"] == 0.0
    assert hl_ro_after["value_usd"] == 0.0
    assert hl_ro_after["cash_usd"] == 520.0
    assert hl_ro_after["spot_positions"][0]["qty"] == 0.0
    assert hl_ro_after["spot_positions"][0]["value_usd"] == 0.0
    assert hl_ro_after["spot_positions"][0]["cash_usd"] == 520.0
    assert hl_ro_after["spot_positions"][0]["snapshot_age_min"] == 0.0


def test_iter100_portfolio_collector_audit_origins_hl_active_cycle_best_fills_and_ag_dust(tmp_path):
    import json
    from intelligence.sentiment.collectors.portfolio_collector import (
        _load_recent_audit_origins,
        harvest_bot_runtime_states,
        harvest_venue_cycles,
        load_symbol_best_recent_fills,
    )

    ws_dir = tmp_path / "ws"
    cdir = ws_dir / "cachedb"
    hl_dir = ws_dir / "hyperliquid"
    cdir.mkdir(parents=True, exist_ok=True)
    hl_dir.mkdir(parents=True, exist_ok=True)

    # 1. _load_recent_audit_origins merges module and rule across multiple events for the same order_id
    audit_file = cdir / "execution_audit_20261105.jsonl"
    audit_file.write_text(
        "\n".join(
            [
                json.dumps({"order_id": "oid-100", "origin_module": "monitortrades"}),
                json.dumps({"order_id": "oid-100", "decision_rule": "tp_ride_step1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    origins = _load_recent_audit_origins(str(cdir))
    assert origins.get("oid-100") == ("monitortrades", "tp_ride_step1")

    # 2. harvest_venue_cycles does not let a flat .state_HYPE.json overwrite an active .state_HYPE_LONG.json
    (hl_dir / ".state_HYPE_LONG.json").write_text(
        json.dumps(
            {
                "qty": 15.0,
                "cost_usd": 600.0,
                "avg_cost_px": 40.0,
                "entry_price": 40.0,
                "last_buy_price": 40.0,
                "entry_ts": 1710000000.0,
                "last_sell_price": 44.0,
                "last_sell_ts": 1710003600.0,
            }
        ),
        encoding="utf-8",
    )
    (hl_dir / ".state_HYPE.json").write_text(
        json.dumps({"qty": 0.0, "cost_usd": 0.0}),
        encoding="utf-8",
    )
    _, hl_cycs, _ = harvest_venue_cycles(str(ws_dir), {"HYPEUSDC": {"price": 42.0}}, {})
    assert hl_cycs["HYPEUSDC"]["qty"] == 15.0
    assert hl_cycs["HYPEUSDC"]["_entry_ts_ms"] == 1710000000000

    # 3. load_symbol_best_recent_fills populates last_buy["ts_ms"] from _entry_ts_ms on cycle fallback
    best_fills = load_symbol_best_recent_fills(str(cdir), "HYPEUSDC", ["HYPEUSDC", "HYPE"])
    assert best_fills is not None
    assert best_fills["venue"] == "hyperliquid"
    assert best_fills["last_buy"]["price"] == 40.0
    assert best_fills["last_buy"]["ts_ms"] == 1710000000000
    assert best_fills["last_sell"]["ts_ms"] == 1710003600000

    # 4. harvest_bot_runtime_states strips non-pending assetguardian sell state when free_base == 0.0 and value_usd is None
    (cdir / "assetguardian_state.json").write_text(
        json.dumps(
            {
                "symbols": {
                    "BTCUSDC": {
                        "buy": {"active": True},
                        "sell": {"pending": False, "total_filled_qty": 0.123456789},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rt_states = harvest_bot_runtime_states(
        str(cdir),
        1710005000.0,
        tradable_positions=[{"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.0, "value_usd": None}],
    )
    assert "sell" not in rt_states["assetguardian_state"]["BTCUSDC"]
    assert rt_states["assetguardian_state"]["BTCUSDC"]["buy"] == {"active": True}


def test_iter101_macro_benchmarks_24h_ext_px_cached_bals_ro_and_prompt_asset_scope(tmp_path):
    """Iter #101:
    1. collect_macro_benchmarks_snapshot selects extended-hours price (postMarketPrice / post-close bar)
       when marketState is POST/PRE and fulldayPrice is absent, and uses 24h boundary pts_24h[0] (not 48h pts[0])
       or previousClose (before chartPreviousClose) for chg_24h_pct fallback.
    2. harvest_recent_orders_and_enrich_positions populates live_held_by_venue_sym from venue_balances_state.json
       for kraken and hyperliquid cycles (not only t212) when tradable_positions is empty.
    3. _format_market_prices_section and _format_microstructure_section isolate _EQ/_US_EQ equities from crypto
       sibling groups, and _format_portfolio_section deduplicates resting orders by client_order_id as well as order_id.
    """
    from intelligence.sentiment.collectors.macro_telemetry_collector import collect_macro_benchmarks_snapshot
    from intelligence.sentiment.collectors.portfolio_collector import harvest_recent_orders_and_enrich_positions
    from orchestratorOS.supervisor.prompt_sections import (
        _format_market_prices_section,
        _format_microstructure_section,
        _format_portfolio_section,
    )

    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    now_s = 1710200000.0

    def _fetch_yahoo(url, **kw):
        if "SPY" in url:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "marketState": "POST",
                                "regularMarketPrice": 500.0,
                                "regularMarketTime": int(now_s - 3600),
                                "regularMarketChangePercent": 0.50,
                                "previousClose": 497.51,
                                "chartPreviousClose": 480.0,
                            },
                            "timestamp": [
                                int(now_s - 170000),  # ~47h ago (480.0)
                                int(now_s - 86000),   # ~24h ago (497.51)
                                int(now_s - 3600),    # regular close (500.0)
                                int(now_s - 60),      # post-market bar (505.0)
                            ],
                            "indicators": {"quote": [{"close": [480.0, 497.51, 500.0, 505.0]}]},
                        }
                    ]
                }
            }
        if "QQQ" in url:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {"regularMarketPrice": 440.0},
                            "timestamp": [
                                int(now_s - 170000),  # ~47h ago (400.0)
                                int(now_s - 86000),   # ~24h ago (430.0)
                                int(now_s - 60),      # latest (440.0)
                            ],
                            "indicators": {"quote": [{"close": [400.0, 430.0, 440.0]}]},
                        }
                    ]
                }
            }
        if "IBIT" in url:
            assert "includePrePost=true" in url
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "fulldayPrice": 45.58,
                                "fulldayChangePercent": 3.47,
                                "regularMarketPrice": 45.33,
                                "regularMarketTime": int(now_s - 32400),
                            },
                            "timestamp": [
                                int(now_s - 36000),   # 10h ago (45.20)
                                int(now_s - 33300),   # 9h15m ago (45.30)
                                int(now_s - 32400),   # 9h ago regular close (45.33)
                            ],
                            "indicators": {"quote": [{"close": [45.20, 45.30, 45.33]}]},
                        }
                    ]
                }
            }
        return {}

    mb = collect_macro_benchmarks_snapshot(str(cdir), now_ts=now_s, persist=False, fetch_fn=_fetch_yahoo)
    assert mb is not None
    # SPY uses post-market bar 505.0 (newer than regularMarketTime) and previousClose 497.51 (+1.51%), not 480.0
    assert mb["items"]["SPY"]["price"] == 505.0
    assert mb["items"]["SPY"]["chg_24h_pct"] == pytest.approx(((505.0 - 497.51) / 497.51) * 100.0, abs=0.02)
    # QQQ fallback without previousClose uses 24h bar 430.0 (+2.33%), not 48h bar 400.0 (+10.0%)
    assert mb["items"]["QQQ"]["chg_24h_pct"] == pytest.approx(((440.0 - 430.0) / 430.0) * 100.0, abs=0.02)
    # IBIT with 9h-old RTH bars omits bogus cross-session 15m/1h deltas against overnight fulldayPrice
    assert mb["items"]["IBIT"]["price"] == 45.58
    assert mb["items"]["IBIT"]["chg_24h_pct"] == 3.47
    assert "chg_15m_pct" not in mb["items"]["IBIT"]
    assert "chg_1h_pct" not in mb["items"]["IBIT"]

    # 2. harvest_recent_orders_and_enrich_positions reads cached flat balance for kraken cycle when tradable_positions is empty
    (cdir / "venue_balances_state.json").write_text(
        json.dumps({"pos:kraken:HYPEUSD": {"venue": "kraken", "symbol": "HYPEUSD", "free_base": 0.0, "locked_base_qty": 0.0}}),
        encoding="utf-8",
    )
    (cdir / "cache_trade_kraken.json").write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSD": [
                        {"side": "BUY", "status": "FILLED", "price": 40.0, "executedQty": 10.0, "time": 1710000000000},
                        {"side": "SELL", "status": "FILLED", "price": 44.0, "executedQty": 10.0, "time": 1710005000000},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    kr_cycles = {
        "HYPEUSD": {
            "qty": 10.0,
            "cost_usd": 400.0,
            "avg_cost_px": 40.0,
            "entry_price": 40.0,
            "pos_peak": 45.0,
            "_entry_ts_ms": 1710000000000,
        }
    }
    harvest_recent_orders_and_enrich_positions(
        str(cdir),
        {},
        [],
        {"HYPEUSD": {"price": 44.0}},
        kr_cycles,
        {},
        {},
    )
    assert kr_cycles["HYPEUSD"]["qty"] == 0.0
    assert "pos_peak" not in kr_cycles["HYPEUSD"]

    # 3. Prompt section equity vs crypto isolation + client_order_id resting deduplication
    sec_3a = "\n".join(
        _format_market_prices_section(
            {
                "COIN_US_EQ": {"price": 250.0, "chg_24h_pct": 1.5},
                "COINUSD": {"price": 1.25, "chg_24h_pct": -2.0},
            }
        )
    )
    assert "* COIN_US_EQ: Price=$250.00" in sec_3a
    assert "* COINUSD: Price=$1.25" in sec_3a

    sec_3b = "\n".join(
        _format_microstructure_section(
            {
                "HYPEUSD": {"ob_imbalance": 0.62, "bid_depth_usd": 50000.0, "ask_depth_usd": 30000.0, "cvd_15m_usd": 12000.0},
            },
            {
                "HYPEUSDC": {"price": 40.0, "chg_15m_pct": -1.5, "vol_24h_usd": 2_400_000.0},
            },
        )
    )
    assert "(+$12.0K)" in sec_3b
    assert "liq_clusters(long_liq=$" in sec_3b

    sec_3c = "\n".join(
        _format_portfolio_section(
            {
                "open_limit_orders": [
                    {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "price": 300.0, "qty": 1.0, "client_order_id": "ai-cid-77"}
                ],
                "last_cycle_execution_status": [
                    {"symbol": "TAOUSDC", "status": "OPEN_RESTING", "age_min": 45.0, "client_order_id": "ai-cid-77"}
                ],
                "recent_ai_intents": [
                    {"symbol": "TAOUSDC", "status": "EXECUTED", "venue_status": "OPEN", "age_h": 1.2, "client_order_id": "ai-cid-77"},
                    {"symbol": "BTCUSDC", "status": "EXECUTED", "venue_status": "FILLED", "age_h": 0.2, "order_id": "oid-88"},
                ],
            },
            include_macro_benchmarks=False,
        )
    )
    assert "last_cycle_execution_status" not in sec_3c
    assert "ai-cid-77" not in sec_3c.split("Recent Autonomous Intents:")[-1]


def test_iter102_pretrade_telemetry_quote_price_scope_and_micro_arb_isolation(tmp_path):
    """Iter #102:
    1. collect_pretrade_telemetry isolates _EQ/_US_EQ equities from crypto variants (so COINUSD never adopts COIN_US_EQ
       price/trends/fills and COIN_US_EQ never adopts COINUSDC price/trends/fills), and does not leak foreign-quote
       current_price from cache_short_trend.json / live_<SYM>.json into p_est["price"] on BTCEUR.
    2. compute_cross_venue_arb_metrics, update_microstructure_cycle_deltas, and reconcile_sibling_flow_and_liq_regimes
       isolate _EQ/_US_EQ symbols from crypto symbols.
    """
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        compute_cross_venue_arb_metrics,
        reconcile_sibling_flow_and_liq_regimes,
        update_microstructure_cycle_deltas,
    )

    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "cache_current_price.json").write_text(
        json.dumps({"items": {"COIN_US_EQ": 250.0}}),
        encoding="utf-8",
    )
    (cdir / "cache_short_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "COIN_US_EQ": {"current_price": 250.0, "slope_full": 0.005, "dynamic": {"900": {"slope_full": 0.005}}},
                    "BTCUSDC": {"current_price": 85000.0, "slope_full": 0.002, "dynamic": {"900": {"slope_full": 0.002}}},
                }
            }
        ),
        encoding="utf-8",
    )
    (cdir / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps({"funding_rate": 0.0001, "open_interest_usd": 100_000_000.0}),
        encoding="utf-8",
    )

    # COINUSD (crypto) must not inherit COIN_US_EQ's $250 price or slopes
    tel_coin_crypto = collect_pretrade_telemetry("COINUSD", cachedb_dir=str(cdir))
    assert "Dynamic Multi-Window Slopes" not in tel_coin_crypto

    # BTCEUR without a EUR price on disk must inherit dimensionless slopes from BTCUSDC without leaking $85,000 USD into liquidation clusters
    tel_btc_eur = collect_pretrade_telemetry("BTCEUR", cachedb_dir=str(cdir))
    assert "Dynamic Multi-Window Slopes" in tel_btc_eur
    assert "Liquidation Clusters (LONG vs SHORT)" not in tel_btc_eur

    # 2. compute_cross_venue_arb_metrics and reconcile_sibling_flow_and_liq_regimes isolate COIN_US_EQ from COINUSD
    micro_map = {
        "COIN_US_EQ": {"ob_mid_price": 250.0, "bid_depth_usd": 50000.0, "ask_depth_usd": 50000.0, "liq_magnet_bias": "SHORT_SQUEEZE_MAGNET_UP"},
        "COINUSD": {"ob_mid_price": 248.0, "bid_depth_usd": 10000.0, "ask_depth_usd": 10000.0, "liq_magnet_bias": "LONG_SWEEP_MAGNET_DOWN"},
    }
    pt_map = {"COIN_US_EQ": {"price": 250.0}, "COINUSD": {"price": 248.0}}
    compute_cross_venue_arb_metrics("COINUSD", micro_map["COINUSD"], pt_map["COINUSD"], micro_map, pt_map)
    assert "cross_venue_arb_bps" not in micro_map["COINUSD"]

    reconcile_sibling_flow_and_liq_regimes(micro_map, pt_map)
    assert micro_map["COINUSD"]["liq_magnet_bias"] == "LONG_SWEEP_MAGNET_DOWN"

    # 3. update_microstructure_cycle_deltas does not copy COINUSD crypto history onto COIN_US_EQ
    (cdir / "microstructure_cycle_history.json").write_text(
        json.dumps({"COINUSD": [{"ts": 1710000000.0, "ob_imbalance": 0.40, "cvd_15m_usd": 1000.0}]}),
        encoding="utf-8",
    )
    eq_micro = {"COIN_US_EQ": {"ob_imbalance": 0.60, "cvd_15m_usd": 5000.0}}
    update_microstructure_cycle_deltas(str(cdir), eq_micro, now_ts=1710000600.0)
    assert "cycle_delta" not in eq_micro["COIN_US_EQ"]


def test_iter103_price_series_microstructure_tca_and_bellwether_equity_crypto_isolation(tmp_path):
    import json
    from intelligence.sentiment.collectors.microstructure_collector import (
        discover_reference_microstructure_symbols,
        enrich_24h_and_microstructure,
        load_symbol_microstructure,
        load_symbol_price_series,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        build_cross_market_context,
        enrich_open_limit_orders_tca,
        resolve_symbol_price,
    )

    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    now_ms = 1_710_000_000_000

    # 1. load_symbol_price_series isolates COIN_US_EQ and COINUSD
    (cdir / "cache_24_price_COIN_US_EQ.json").write_text(
        json.dumps({"items": {"COIN_US_EQ": [[now_ms - 3600_000, 240.0], [now_ms, 250.0]]}}),
        encoding="utf-8",
    )
    s24_crypto, v7d_crypto = load_symbol_price_series(
        str(cdir), "COINUSD", current_price_items={"COIN_US_EQ": 252.0}
    )
    assert s24_crypto == []
    assert v7d_crypto == []

    (cdir / "cache_24_price_COINUSD.json").write_text(
        json.dumps({"items": {"COINUSD": [[now_ms - 3600_000, 1.20], [now_ms, 1.30]]}}),
        encoding="utf-8",
    )
    cdir_only_crypto = tmp_path / "cachedb_only_crypto"
    cdir_only_crypto.mkdir(parents=True, exist_ok=True)
    (cdir_only_crypto / "cache_24_price_COINUSD.json").write_text(
        json.dumps({"items": {"COINUSD": [[now_ms - 3600_000, 1.20], [now_ms, 1.30]]}}),
        encoding="utf-8",
    )
    s24_eq, v7d_eq = load_symbol_price_series(
        str(cdir_only_crypto), "COIN_US_EQ", current_price_items={"COINUSD": 1.35}
    )
    assert s24_eq == []
    assert v7d_eq == []

    # 2. load_symbol_microstructure, discover_reference_microstructure_symbols, and enrich_24h_and_microstructure
    (cdir / "orderbook_depth_COINUSD.json").write_text(
        json.dumps({"mid_price": 1.30, "best_bid": 1.299, "best_ask": 1.301, "imbalance_ratio": 0.62, "bid_depth_usd": 25000.0, "ask_depth_usd": 15000.0}),
        encoding="utf-8",
    )
    assert load_symbol_microstructure(str(cdir), "COIN_US_EQ") is None
    assert "COINUSD" in discover_reference_microstructure_symbols(str(cdir), {"COIN_US_EQ": {"bid_depth_usd": 25000.0}})

    pt_mix = {"COIN_US_EQ": {"price": 250.0, "trend": 0, "growth_coefficient": 0.0}}
    micro_mix = {}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_mix, micro_mix, {}, {}, now_ms / 1000.0)
    assert "COIN_US_EQ" not in micro_mix
    assert "COINUSD" in micro_mix
    assert pt_mix["COIN_US_EQ"]["price"] == pytest.approx(250.0)
    assert pt_mix["COINUSD"]["price"] == pytest.approx(1.30)

    # 3. resolve_symbol_price, enrich_open_limit_orders_tca, and build_cross_market_context
    assert resolve_symbol_price("COINUSD", {"COIN_US_EQ": {"price": 250.0}}) == 0.0
    assert resolve_symbol_price("COIN_US_EQ", {"COINUSD": {"price": 1.30}}) == 0.0
    assert resolve_symbol_price("SPCX", {"SPCX_US_EQ": {"price": 170.0}}) == pytest.approx(170.0)

    open_eq = [{"venue": "t212", "symbol": "COIN_US_EQ", "side": "SELL", "price": 255.0}]
    enrich_open_limit_orders_tca(
        open_eq,
        {"COINUSDC": {"price": 1.30, "vol_1h_pct": 4.0}},
        {"COINUSDC": {"ob_mid_price": 1.30, "ask_wall_px": 1.31, "ask_wall_usd": 50000.0}},
    )
    assert "dist_from_spot_pct" not in open_eq[0]
    assert "wall_queue_pos" not in open_eq[0]

    cmc = build_cross_market_context(
        {
            "COINUSD": {"price": 1.30, "chg_24h_pct": 8.33},
            "COIN_US_EQ": {"price": 250.0, "chg_24h_pct": 4.17, "low_24h": 240.0, "high_24h": 252.0},
        },
        micro_mix,
        {},
    )
    assert "COIN" in cmc.get("crypto_bellwethers", {})
    assert cmc["crypto_bellwethers"]["COIN"]["symbol"] == "COINUSD"
    assert cmc["crypto_bellwethers"]["COIN"]["price"] == pytest.approx(1.30)
    assert "COIN_US_EQ" in cmc.get("equity_bellwethers", {})


def test_iter104_portfolio_flat_ro_pos_clear_equity_scope_and_7d_swing_sync(tmp_path):
    import json
    from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from intelligence.sentiment.evaluators.price_series_evaluator import _apply_24h_series_enrichment

    # 1. Flat Binance position after SELL clears stale cost-basis anchors on pos and swing_subcycle_qty/active_cycle_rem_qty on ro
    pos_bn_flat = {
        "venue": "binance",
        "symbol": "SOLUSDC",
        "quote": "USDC",
        "price": 150.0,
        "free_base": 0.0,
        "free_quote": 500.0,
        "active_avg_buy_px": 140.0,
        "pnl_vs_avg_pct": 7.14,
        "tactical_avg_buy_px": 142.0,
        "pnl_vs_tactical_avg_pct": 5.63,
        "swing_subcycle_avg_px": 141.0,
        "pnl_vs_swing_avg_pct": 6.38,
        "open_lots": [{"lot_id": "L1", "qty": 1.0, "entry_px": 140.0}],
        "lowest_buy_px": 140.0,
        "pnl_vs_lowest_buy_pct": 7.14,
        "trailing_peak": 155.0,
        "dd_from_peak_pct": -3.23,
    }
    ro_sol = {
        "venue": "binance",
        "last_action_side": "SELL",
        "active_avg_buy_px": 140.0,
        "tactical_avg_buy_px": 142.0,
        "swing_subcycle_avg_px": 141.0,
        "swing_subcycle_qty": 1.0,
        "active_cycle_rem_qty": 1.0,
        "open_lots": [{"lot_id": "L1", "qty": 1.0, "entry_px": 140.0}],
        "last_buy": {"price": 140.0, "qty": 1.0, "ts_ms": 1_710_000_000_000},
        "last_sell": {"price": 150.0, "qty": 1.0, "ts_ms": 1_710_003_600_000},
        "recent_buys": [{"price": 140.0, "qty": 1.0, "ts_ms": 1_710_000_000_000}],
        "recent_sells": [{"price": 150.0, "qty": 1.0, "ts_ms": 1_710_003_600_000}],
    }
    pos_eq = {
        "venue": "t212",
        "symbol": "COIN_US_EQ",
        "quote": "USD",
        "price": 250.0,
        "free_base": 1.0,
        "free_quote": 100.0,
    }
    pt = {
        "SOLUSDC": {"price": 150.0, "spread_bps": 1.5},
        "COIN_US_EQ": {"price": 250.0},
        "COINUSDC": {"price": 1.30, "spread_bps": 18.5, "high_24h": 1.45},
    }
    enrich_positions_with_orders_and_cycles(
        [pos_bn_flat, pos_eq],
        pt,
        {},
        {"binance": {"SOLUSDC": ro_sol}, "t212": {}},
        {"SOLUSDC": ro_sol},
        {},
        {},
        {},
        {},
    )
    for stale_k in (
        "active_avg_buy_px",
        "pnl_vs_avg_pct",
        "tactical_avg_buy_px",
        "pnl_vs_tactical_avg_pct",
        "swing_subcycle_avg_px",
        "pnl_vs_swing_avg_pct",
        "open_lots",
        "lowest_buy_px",
        "pnl_vs_lowest_buy_pct",
        "trailing_peak",
        "dd_from_peak_pct",
    ):
        assert stale_k not in pos_bn_flat
    assert "swing_subcycle_qty" not in ro_sol
    assert "active_cycle_rem_qty" not in ro_sol
    assert pos_bn_flat.get("last_sell_px") == pytest.approx(150.0)
    # T212 COIN_US_EQ must not inherit COINUSDC's 18.5 bps crypto spread
    assert "spread_bps" not in pos_eq

    # 2. load_symbol_best_recent_fills isolates t212 vs crypto venues
    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "cache_trade_kraken.json").write_text(
        json.dumps(
            {
                "items": {
                    "COIN": [
                        {"symbol": "COINUSD", "side": "BUY", "price": 1.25, "qty": 100.0, "executedQty": 100.0, "timestamp": 1_710_000_000_000, "status": "FILLED"}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_symbol_best_recent_fills(str(cdir), "COIN_US_EQ", ("COIN_US_EQ", "COIN")) is None

    # 3. _apply_24h_series_enrichment syncs swing_low_px, swing_high_px, and chg_7d_pct when 24h range widens 7d range
    p_entry = {
        "price": 105.0,
        "low_7d": 100.0,
        "high_7d": 110.0,
        "chg_7d_pct": 12.0,
        "swing_low_px": 100.0,
        "swing_high_px": 110.0,
        "rally_from_swing_low_pct": 5.0,
        "pullback_from_swing_high_pct": -4.55,
    }
    m24 = {"low_24h": 98.1234, "high_24h": 112.5678, "chg_1h_pct": 0.5}
    _apply_24h_series_enrichment("SOLUSDC", p_entry, m24, [[1_710_000_000_000, 100.0], [1_710_003_600_000, 105.0]], None)
    assert p_entry["low_7d"] == pytest.approx(98.1234)
    assert p_entry["high_7d"] == pytest.approx(112.5678)
    assert p_entry["swing_low_px"] == pytest.approx(98.1234)
    assert p_entry["swing_high_px"] == pytest.approx(112.5678)
    assert p_entry["chg_7d_pct"] == pytest.approx(p_entry["bounce_7d_low_pct"])


def test_iter105_same_venue_wallet_dedup_xstocks_px_and_cycle_equity_scope(tmp_path):
    import json
    from intelligence.sentiment.collectors.portfolio_collector import harvest_kraken_xstocks
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import reconcile_live_venue_equity_summary
    from intelligence.sentiment.evaluators.portfolio_evaluator import reconcile_venue_cycles_with_recent_orders
    from intelligence.sentiment.evaluators.tca_risk_evaluator import compute_enterprise_portfolio_risk_metrics

    # 1. reconcile_live_venue_equity_summary & compute_enterprise_portfolio_risk_metrics deduplicate same-venue same-base spot wallets
    ps = {
        "tradable_positions": [
            {"venue": "binance", "symbol": "BTCUSDC", "base": "BTC", "quote": "USDC", "value_usd": 1000.0, "locked_base_usd": 100.0},
            {"venue": "binance", "symbol": "BTCUSDT", "base": "BTC", "quote": "USDT", "value_usd": 1000.0, "locked_base_usd": 50.0},
            {"venue": "hyperliquid", "symbol": "HYPEUSDC", "base": "HYPE", "quote": "USDC", "free_base": 0.0, "value_usd": 0.0},
            {"venue": "hyperliquid", "symbol": "BTCUSDC", "base": "BTC", "quote": "USDC", "free_base": 0.01, "value_usd": 850.0},
        ],
        "venue_quote_cash": {
            "binance:USDC": {"venue": "binance", "asset": "USDC", "free": 500.0},
            "BINANCE:USDC": {"venue": "BINANCE", "asset": "USDC", "free": 500.0},
            "hyperliquid:USDC": {"venue": "hyperliquid", "asset": "USDC", "free": 150.0},
        },
        "hyperliquid_readonly": {
            "qty": 0.005,
            "value_usd": 400.0,
            "perp_equity_usd": 50.0,
            "spot_positions": [{"coin": "BTC", "qty": 0.005, "value_usd": 400.0}],
        },
    }
    reconcile_live_venue_equity_summary(ps)
    assert ps["live_binance_equity_usdc"] == pytest.approx(1650.0)
    assert ps["hyperliquid_readonly"]["live_reconciled_spot_usd"] == pytest.approx(1000.0)
    assert ps["hyperliquid_readonly"]["qty"] == pytest.approx(0.01)

    risk = compute_enterprise_portfolio_risk_metrics(ps, {"BTCUSDC": {"price": 85000.0}}, {})
    assert risk["invested_exposure_usd"] == pytest.approx(2000.0)
    assert risk["global_free_cash_usd"] == pytest.approx(650.0)
    assert risk["unified_nav_usd"] == pytest.approx(2650.0)

    # 2. harvest_kraken_xstocks resolves live_price from SPCX_US_EQ even when listed_pair is SPCXUSD
    kdir = tmp_path / "kraken"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "xstock_state.json").write_text(
        json.dumps({"known_assets": {"SPCXx.T": 2.5}, "pair": "SPCXUSD"}),
        encoding="utf-8",
    )
    xst = harvest_kraken_xstocks(str(kdir), {"SPCX_US_EQ": {"price": 112.5}}, {})
    assert isinstance(xst, dict)
    assert xst.get("live_price") == pytest.approx(112.5)

    # 3. reconcile_venue_cycles_with_recent_orders isolates crypto vs equity variants in c_cands
    t212_cycs = {"COIN_US_EQ": {"qty": 2.0, "avg_cost_px": 200.0, "cost_usd": 400.0, "_entry_ts_ms": 1_700_000_000_000}}
    sell_map = reconcile_venue_cycles_with_recent_orders(
        {},
        {},
        t212_cycs,
        {"t212": {"COINUSDC": {"last_sell": {"price": 1.25, "qty": 2.0, "ts_ms": 1_710_000_000_000}}}},
        {},
        {"COIN_US_EQ": {"price": 210.0}, "COINUSDC": {"price": 1.25}},
    )
    assert ("t212:entry", "COINUSDC") not in sell_map
    assert t212_cycs["COIN_US_EQ"]["qty"] == pytest.approx(2.0)


def test_iter22_near_balanced_cvd_spot_absorbing_single_stock_shares_and_calm_geo_refresh(tmp_path):
    import json
    from intelligence.macro.lexicon_ssot import (
        has_financial_or_geopolitical_context,
        is_off_topic_general_news_title,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import harvest_macro_and_news_sensors
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_microstructure_cvd

    # 1. Discrepancy BU: enrich_microstructure_cvd classifies mild negative 15m CVD with taker_ratio >= 0.90,
    #    flat/positive 15m price change (>= -0.02%), and positive 5m momentum as spot_absorbing rather than aggressive_market_dumps
    btc_m = {
        "ob_imbalance": 0.451,
        "divergence_regime": "neutral",
        "taker_buy_sell_ratio": 0.9765,
        "taker_buy_vol_usd": 54_209_373.39,
        "taker_sell_vol_usd": 55_515_401.53,
        "cvd_15m_usd": -1_310_000.0,
        "cvd_5m_usd": 209_000.0,
    }
    enrich_microstructure_cvd(btc_m, {"price": 80354.70, "chg_15m_pct": 0.0, "chg_5m_pct": 0.02})
    assert btc_m["cvd_regime"] == "spot_absorbing"

    # 2. Discrepancy BV: single-stock headline matching only "shares" does not outrank broad macro/market wraps
    assert has_financial_or_geopolitical_context("NextDC shares soar nearly 51% after Goodman, Digital Realty stake purchases") is False
    assert has_financial_or_geopolitical_context("European shares climb ahead of inflation print") is True
    assert has_financial_or_geopolitical_context("Asian shares edge higher on chip demand") is True

    # 3. Discrepancy BW: harvest_macro_and_news_sensors refreshes NORMAL geopolitical_threat timestamp when >=60m old and 0 high-sev headlines
    now_s = 1_793_865_000.0
    cdir = tmp_path / "cachedb_iter22"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "geopolitical_eval.json").write_text(
        json.dumps(
            {
                "threat_level": "NORMAL",
                "risk_score": 0.1,
                "summary": "Calm geopolitical and energy climate. No systemic shock detected.",
                "recommended_brake": "NONE",
                "headlines_analyzed": 1,
                "ts": now_s - 242 * 60.0,
                "is_fallback": False,
                "headlines_digest": "",
            }
        ),
        encoding="utf-8",
    )
    (cdir / "news_feed_collect.json").write_text(
        json.dumps(
            [
                {
                    "title": "NextDC shares soar nearly 51% after Goodman, Digital Realty stake purchases",
                    "source": "CNBC",
                    "first_seen_ts": now_s - 300.0,
                    "is_high_severity": False,
                    "matched_keywords": [],
                },
                {
                    "title": "Asian Stocks to Rise as Oil Slides After US Vote: Markets Wrap",
                    "source": "Bloomberg",
                    "first_seen_ts": now_s - 900.0,
                    "is_high_severity": False,
                    "matched_keywords": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    sensors, b_news = harvest_macro_and_news_sensors(str(cdir), now_s)
    assert b_news[0]["title"].startswith("Asian Stocks to Rise as Oil Slides")
    assert sensors["geopolitical_threat"]["age_min"] == 0.0
    assert sensors["geopolitical_threat"]["headlines_analyzed"] == 2


def test_iter106_equity_crypto_scope_isolation_in_trends_sibling_sync_fills_and_risk():
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        enrich_cross_venue_trends,
        sync_sibling_symbol_telemetry,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import extract_symbol_recent_fills
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
    )

    # 1. enrich_cross_venue_trends & sync_sibling_symbol_telemetry do not cross-contaminate COIN_US_EQ and COINUSDC
    pt = {
        "COINUSDC": {"price": 1.25, "chg_15m_pct": 8.5, "slope_15m": 0.012, "vol_1h_pct": 4.2},
        "COIN_US_EQ": {"price": 240.0},
    }
    micro = {
        "COINUSDC": {
            "ob_mid_price": 1.25,
            "best_bid": 1.24,
            "best_ask": 1.26,
            "ob_imbalance": 0.72,
            "whale_long_pct": 0.68,
            "funding_rate": 0.0005,
        }
    }
    enrich_cross_venue_trends(
        pt,
        trends_data={"items": {"COINUSDC": {"slope_15m": 0.012, "vol_1h_pct": 4.2}}},
        kalman_data={"COINUSDC": {"kalman_vel": 0.055, "kalman_vel_std": 0.01, "kalman_trend": 1}},
        long_items={"COINUSDC": [{"direction": "UP", "duration_seconds": 7200, "timestamp": 1700000000.0}]},
        priceanalysis_data={},
        t_trend_map={"COINUSDC": {"median_d": 3.5, "p90_d": 7.0}},
        watcher_price_items={},
    )
    assert pt["COIN_US_EQ"].get("slope_15m") is None
    assert pt["COIN_US_EQ"].get("kalman_vel") is None
    assert pt["COIN_US_EQ"].get("long_trend_dir") is None
    assert pt["COIN_US_EQ"].get("weibull_median_d") is None

    sync_sibling_symbol_telemetry(pt, micro)
    assert pt["COIN_US_EQ"].get("chg_15m_pct") is None
    assert "COIN_US_EQ" not in micro

    # 2. extract_symbol_recent_fills isolates COIN_US_EQ vs COINUSDC AI intents and bot fills
    fills_eq = extract_symbol_recent_fills(
        [{"symbol": "COIN_US_EQ", "orderId": "bot-eq-1", "side": "BUY", "price": 235.0, "executedQty": 2.0, "timestamp": 1_700_000_010_000, "status": "FILLED"}],
        ai_intents=[
            {
                "intent_id": "ai-crypto-1",
                "symbol": "COINUSDC",
                "decision": "BUY",
                "order_type": "MARKET",
                "status": "EXECUTED",
                "timestamp": 1_700_000_000.0,
                "execution_payload": {"orderId": "ai-ord-1", "symbol": "COINUSDC", "side": "BUY", "price": 1.25, "executedQty": 2.0, "status": "FILLED"},
            }
        ],
        symbol="COIN_US_EQ",
    )
    assert isinstance(fills_eq, dict)
    assert fills_eq["last_action_actor"] == "BOT"
    assert fills_eq["last_buy"]["price"] == pytest.approx(235.0)
    assert len(fills_eq["recent_buys"]) == 1

    # 3. enrich_open_limit_orders_tca & compute_enterprise_portfolio_risk_metrics scope t212 bare ticker and correlation matrix
    oo = [{"venue": "t212", "symbol": "COIN", "side": "BUY", "price": 238.0}]
    enrich_open_limit_orders_tca(oo, {"COINUSDC": {"price": 1.25}, "COIN_US_EQ": {"price": 240.0}}, {})
    assert oo[0]["dist_from_spot_pct"] == pytest.approx(-0.83)

    ps_mixed = {
        "tradable_positions": [
            {"venue": "binance", "symbol": "COINUSDC", "base": "COIN", "quote": "USDC", "value_usd": 1000.0},
            {"venue": "t212", "symbol": "COIN_US_EQ", "base": "COIN", "quote": "USD", "value_usd": 1000.0},
        ],
        "venue_quote_cash": {"binance:USDC": {"venue": "binance", "asset": "USDC", "free": 500.0}},
    }
    risk_mixed = compute_enterprise_portfolio_risk_metrics(
        ps_mixed,
        {
            "COINUSDC": {"price": 1.25, "corr_btc_24h": 0.85, "vol_1h_pct": 1.0},
            "COIN_US_EQ": {"price": 240.0, "corr_btc_24h": 0.30, "vol_1h_pct": 1.0},
        },
        {},
    )
    assert risk_mixed["corr_vs_btc_24h"]["COIN"] == pytest.approx(0.85)
    assert risk_mixed["corr_vs_btc_24h"]["COIN_US_EQ"] == pytest.approx(0.30)


def test_equity_telemetry_and_rth_market_gating(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from instrument_registry import is_symbol_market_open
    from intelligence.external.collectors.base_collector import _DiskBackedSymbolCollector, _resolve_configured_venue
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector

    assert _resolve_configured_venue("NVDA_US_EQ") == "t212"
    assert _resolve_configured_venue("SPCX_US_EQ") == "t212"
    assert _resolve_configured_venue("RGNT_US_EQ") == "t212"

    # RTH check: Wednesday 15:00 UTC (10:00 ET) is open; Sunday 15:00 UTC is closed for equities, open for crypto
    wed_rth = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc).timestamp()
    sun_closed = datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc).timestamp()
    assert is_symbol_market_open("NVDA_US_EQ", wed_rth) is True
    assert is_symbol_market_open("NVDA_US_EQ", sun_closed) is False
    assert is_symbol_market_open("BTCUSDC", sun_closed) is True

    _DiskBackedSymbolCollector._EQUITY_CTX_MEMO.clear()
    _DiskBackedSymbolCollector._EQUITY_OPTIONS_MEMO.clear()
    _DiskBackedSymbolCollector._EQUITY_NO_OPTIONS_TS.clear()
    called_urls = []

    def _fake_fetch_json(cls_or_self, url, label, lookup_sym, **kwargs):
        called_urls.append(url)
        assert "binance.com" not in url
        if "SPCX" in url:
            return {}
        if "RGNT" in url and "cdn.cboe.com" in url:
            assert "_RGNT.json" not in url
            return {}
        if "api.nasdaq.com" in url:
            return {
                "data": {
                    "primaryData": {
                        "lastSalePrice": "$198.50",
                        "bidPrice": "$198.48",
                        "askPrice": "$198.52",
                        "bidSize": "250 x 100",
                        "askSize": "200 x 100",
                        "volume": "45,000,000",
                    }
                }
            }
        if "query1.finance.yahoo.com" in url:
            ts_list = [int(wed_rth - (20 - i) * 300) for i in range(20)]
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {"regularMarketPrice": 198.50},
                            "timestamp": ts_list,
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [197.0 + i * 0.08 for i in range(20)],
                                        "high": [197.2 + i * 0.08 for i in range(20)],
                                        "low": [196.9 + i * 0.08 for i in range(20)],
                                        "close": [197.15 + i * 0.08 for i in range(20)],
                                        "volume": [100_000 for _ in range(20)],
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        if "cdn.cboe.com" in url:
            return {
                "data": {
                    "options": [
                        {"option": "NVDA261120C00200000", "open_interest": 12000, "volume": 3500},
                        {"option": "NVDA261120P00195000", "open_interest": 8000, "volume": 2100},
                    ]
                }
            }
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(_DiskBackedSymbolCollector, "_fetch_json_dict", classmethod(_fake_fetch_json))

    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), cache_ttl_sec=5.0, disk_ttl_sec=60.0)
    dt_col = DerivativesTelemetryCollector(cache_dir=str(tmp_path), cache_ttl_sec=60.0, disk_ttl_sec=120.0)
    wh_col = WhalePositioningCollector(cache_dir=str(tmp_path), cache_ttl_sec=60.0, disk_ttl_sec=180.0)

    ob_snap = ob_col.fetch("NVDA_US_EQ", allow_network=True, now=wed_rth)
    dt_snap = dt_col.fetch("NVDA_US_EQ", allow_network=True, now=wed_rth)
    wh_snap = wh_col.fetch("NVDA_US_EQ", allow_network=True, now=wed_rth)

    assert ob_snap is not None and ob_snap.symbol == "NVDA_US_EQ"
    assert ob_snap.best_bid == pytest.approx(198.48)
    assert ob_snap.best_ask == pytest.approx(198.52)
    assert ob_snap.bid_depth_usd > 0.0 and ob_snap.ask_depth_usd > 0.0

    assert dt_snap is not None and dt_snap.symbol == "NVDA_US_EQ"
    assert dt_snap.open_interest == pytest.approx((12000 + 8000) * 100.0)
    assert dt_snap.open_interest_usd > 0.0

    assert wh_snap is not None and wh_snap.symbol == "NVDA_US_EQ"
    assert wh_snap.top_traders_long_pct == pytest.approx(0.6)
    assert wh_snap.top_traders_long_ratio == pytest.approx(1.5)
    assert wh_snap.taker_buy_sell_ratio > 1.0

    # Pre-IPO / unpriced ticker (SPCX_US_EQ) memoizes negative context for 60s across all 3 collectors
    called_urls.clear()
    assert ob_col.fetch("SPCX_US_EQ", allow_network=True, now=wed_rth) is None
    assert dt_col.fetch("SPCX_US_EQ", allow_network=True, now=wed_rth) is None
    assert wh_col.fetch("SPCX_US_EQ", allow_network=True, now=wed_rth) is None
    assert len(called_urls) == 2

    # Stock without listed options (RGNT_US_EQ) only queries RGNT.json once and backs off Cboe for 1800s
    called_urls.clear()
    assert dt_col.fetch("RGNT_US_EQ", allow_network=True, now=wed_rth) is not None
    cboe_calls_1 = [u for u in called_urls if "cdn.cboe.com" in u]
    assert len(cboe_calls_1) == 1 and "RGNT.json" in cboe_calls_1[0]
    called_urls.clear()
    dt_col._cache.clear()
    assert dt_col.fetch("RGNT_US_EQ", allow_network=True, now=wed_rth + 300.0) is not None
    assert [u for u in called_urls if "cdn.cboe.com" in u] == []

    # Outside RTH with MPTRADE_FORCE_RTH_GATE=1, collectors reuse the disk snapshot (even 48h old over weekend) without network calls
    sat_closed = wed_rth + 3 * 86400.0
    ob_col.record_snapshot(ob_snap.__class__(**{**ob_snap.to_dict(), "ts": sat_closed - 48.0 * 3600.0}), save_to_disk=True)
    ob_col._cache.clear()
    called_urls.clear()
    monkeypatch.setenv("MPTRADE_FORCE_RTH_GATE", "1")
    ob_reused = ob_col.fetch("NVDA_US_EQ", allow_network=True, now=sat_closed)
    assert ob_reused is not None
    assert called_urls == []


def test_iter107_equity_crypto_scope_isolation_in_instrument_registry_and_execution_audit(tmp_path):
    import json
    import math
    from types import SimpleNamespace
    from instrument import resolve_instrument_live_price, resolve_tradable_instrument
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
    )
    from providers.execution_audit import (
        normalize_trading_decisions,
        summarize_recent_ai_intents,
    )

    # 1. resolve_tradable_instrument never resolves COIN_US_EQ to crypto COINUSDC or COINUSDC to t212 COIN_US_EQ
    coin_crypto = SimpleNamespace(
        symbol="COINUSDC", provider_name="binance", base="COIN", quote="USDC", enabled=True, _provider=SimpleNamespace()
    )
    coin_eq = SimpleNamespace(
        symbol="COIN_US_EQ", provider_name="t212", base="COIN", quote="USD", enabled=True, _provider=SimpleNamespace()
    )
    assert resolve_tradable_instrument("COIN_US_EQ", venue=None, instruments_map={"BINANCE_COIN": coin_crypto}) is None
    assert resolve_tradable_instrument("COIN_US_EQ", venue="binance", instruments_map={"BINANCE_COIN": coin_crypto}) is None
    assert resolve_tradable_instrument("COINUSDC", venue=None, instruments_map={"T212_COIN": coin_eq}) is None
    assert resolve_tradable_instrument("COIN_US_EQ", venue=None, instruments_map={"BINANCE_COIN": coin_crypto, "T212_COIN": coin_eq}) is coin_eq
    assert resolve_tradable_instrument("COINUSDC", venue=None, instruments_map={"BINANCE_COIN": coin_crypto, "T212_COIN": coin_eq}) is coin_crypto

    # 2. resolve_instrument_live_price isolates equity vs crypto keys in cache_current_price.json
    (tmp_path / "cache_current_price.json").write_text(
        json.dumps({"items": {"COINUSDC": {"price": 1.25}}}),
        encoding="utf-8",
    )
    assert math.isnan(resolve_instrument_live_price(coin_eq, cache_dir=str(tmp_path)))
    assert resolve_instrument_live_price(coin_crypto, cache_dir=str(tmp_path)) == pytest.approx(1.25)

    (tmp_path / "cache_current_price.json").write_text(
        json.dumps({"items": {"COIN_US_EQ": {"price": 240.0}}}),
        encoding="utf-8",
    )
    assert math.isnan(resolve_instrument_live_price(coin_crypto, cache_dir=str(tmp_path)))
    assert resolve_instrument_live_price(coin_eq, cache_dir=str(tmp_path)) == pytest.approx(240.0)

    # 3. summarize_recent_ai_intents does not cross-match recent_orders or open_orders across equity vs crypto
    intents = [
        {
            "intent_id": "eq_1",
            "venue": "t212",
            "symbol": "COIN_US_EQ",
            "decision": "BUY",
            "order_type": "MARKET",
            "status": "EXECUTED",
            "notional_usd": 240.0,
            "timestamp": 1_700_000_000.0,
            "execution_payload": {},
        }
    ]
    recent_orders_crypto_only = {
        "COINUSDC": {
            "last_buy": {"price": 1.25, "qty": 100.0, "ts_ms": 1_700_000_001_000, "actor": "AI"},
        }
    }
    summaries = summarize_recent_ai_intents(
        intents,
        "equities",
        1_700_000_010.0,
        lambda d, default=None: d.get("notional_usd", default),
        recent_orders=recent_orders_crypto_only,
        open_orders=[{"symbol": "COINUSDC", "side": "BUY", "price": 1.25, "origQty": 100.0, "executedQty": 50.0}],
    )
    assert len(summaries) == 1
    assert summaries[0].get("fill_px") is None
    assert summaries[0].get("filled_qty") is None

    # 4. normalize_trading_decisions keeps both COIN_US_EQ and COINUSDC when venue is omitted
    decs = normalize_trading_decisions(
        {
            "decisions": [
                {"symbol": "COIN_US_EQ", "decision": "BUY", "confidence": 0.85, "rationale": "eq"},
                {"symbol": "COINUSDC", "decision": "BUY", "confidence": 0.80, "rationale": "crypto"},
            ]
        }
    )
    assert len(decs) == 2
    assert {d["symbol"] for d in decs} == {"COIN_US_EQ", "COINUSDC"}

    # 5. enrich_positions_with_orders_and_cycles does not price bare t212 COIN from crypto COINUSDC
    pos_list = [{"venue": "t212", "symbol": "COIN", "free_base": 2.0, "price": 0.0}]
    t212_cyc = {"COIN": {"qty": 2.0, "avg_cost_px": 220.0, "entry_price": 220.0, "pos_peak": 220.0}}
    enrich_positions_with_orders_and_cycles(
        pos_list,
        {"COINUSDC": {"price": 1.25}},
        {},
        {},
        {},
        {},
        {},
        t212_cyc,
        {},
    )
    assert pos_list[0]["price"] == 0.0
    assert t212_cyc["COIN"]["pos_peak"] == pytest.approx(220.0)


def test_iter23_cycle_realized_pnl_sync_legacy_whale_refresh_and_post_rth_zero_vol_hole():
    """Verify Iteration #23 fixes:
    1. BX: enrich_positions_with_orders_and_cycles and _format_portfolio_section reconcile
       negative cycle realized_pnl_usd with active sub-cycle lifo_realized_pnl_usd across
       both tradable positions (Loop 1), non-tradable cycles (Loop 2), and t212_sibling_cycle.
    2. BY: _DiskBackedSymbolCollector._prepare_fetch forces a one-shot post-RTH refresh
       when the cached equity whale snapshot was taken while RTH was still open (e.g. 15:59:47 EST)
       or contains legacy un-normalized CLV fields (taker_ratio_5m < 0.176 or top_traders_long_pct <= 0.0).
    3. BZ: microstructure_collector imports safe_pos_int and truncates trailing zero-volume
       post-RTH bars separated by a >20m gap so 15m/30m deltas do not vanish on SPCX_US_EQ.
    """
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from intelligence.sentiment.collectors.microstructure_collector import safe_pos_int

    # 1. BX: Cycle realized_pnl_usd synchronization in Loop 1 and Loop 2
    assert safe_pos_int(42, 0) == 42

    pos_list_eq = [
        {"venue": "t212", "symbol": "SPCX_US_EQ", "free_base": 54.0, "price": 56.11}
    ]
    t212_cyc_eq = {
        "SPCX_US_EQ": {
            "qty": 54.0,
            "avg_cost_px": 56.418889,
            "entry_price": 56.418889,
            "pos_peak": 58.65,
            "realized_pnl_usd": -179.69,
        }
    }
    ro_eq = {
        "t212:SPCX_US_EQ": {
            "lifo_realized_pnl_usd": -22.62,
            "lifo_avg_cost_px": 56.418889,
        },
        "SPCX_US_EQ": {
            "lifo_realized_pnl_usd": -22.62,
            "lifo_avg_cost_px": 56.418889,
        },
    }
    vro_eq = {
        "t212": {
            "SPCX_US_EQ": {
                "lifo_realized_pnl_usd": -22.62,
                "lifo_avg_cost_px": 56.418889,
            }
        }
    }
    enrich_positions_with_orders_and_cycles(
        pos_list_eq,
        {"SPCX_US_EQ": {"price": 56.11}},
        {},
        vro_eq,
        ro_eq,
        {},
        {},
        t212_cyc_eq,
        {},
    )
    assert t212_cyc_eq["SPCX_US_EQ"]["realized_pnl_usd"] == pytest.approx(-22.62)

    # Also test Loop 2 (when scope="crypto", SPCX_US_EQ is not in tradable_positions)
    t212_cyc_crypto_scope = {
        "SPCX_US_EQ": {
            "qty": 54.0,
            "avg_cost_px": 56.418889,
            "entry_price": 56.418889,
            "pos_peak": 58.65,
            "realized_pnl_usd": -179.69,
        }
    }
    enrich_positions_with_orders_and_cycles(
        [],
        {"SPCX_US_EQ": {"price": 56.11}},
        {},
        vro_eq,
        ro_eq,
        {},
        {},
        t212_cyc_crypto_scope,
        {},
    )
    assert t212_cyc_crypto_scope["SPCX_US_EQ"]["realized_pnl_usd"] == pytest.approx(-22.62)

    # Verify _format_portfolio_section reconciles t212_sibling_cycle realized_pnl_usd as well
    rendered_portfolio = "\n".join(
        _format_portfolio_section(
            {
                "scope": "equities",
                "tradable_positions": [
                    {
                        "venue": "t212",
                        "symbol": "SPCX_US_EQ",
                        "free_base": 54.0,
                        "price": 56.11,
                        "lifo_realized_pnl_usd": -22.62,
                    }
                ],
                "t212_cycles": {
                    "SPCX_US_EQ": {
                        "qty": 54.0,
                        "cost_usd": 3046.62,
                        "avg_cost_px": 56.418889,
                        "realized_pnl_usd": -179.69,
                    }
                },
                "kraken_xstocks": {
                    "symbol": "SPACEXUSD",
                    "qty": 0.9038,
                    "price": 122.99,
                    "value_usd": 111.16,
                    "t212_sibling": "SPCX_US_EQ",
                    "t212_sibling_price": 56.11,
                    "t212_sibling_cycle": {
                        "qty": 54.0,
                        "avg_cost_px": 56.418889,
                        "realized_pnl_usd": -179.69,
                    },
                },
            }
        )
    )
    assert "-179.69" not in rendered_portfolio
    assert "'realized_pnl_usd': -22.62" in rendered_portfolio
    assert "realized_pnl=$-22.62" in rendered_portfolio


def test_iter108_portfolio_collector_and_scope_filter_equity_crypto_isolation(tmp_path):
    import json
    from types import SimpleNamespace
    from cachepaths import cache_symbol_variants
    from intelligence.sentiment.collectors.portfolio_collector import (
        harvest_recent_orders_and_enrich_positions,
        harvest_tradable_positions_and_orders,
        harvest_venue_cycles,
        load_symbol_best_recent_fills,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        apply_asset_scope_filter,
    )

    # 1. harvest_venue_cycles and harvest_tradable_positions_and_orders isolate bare t212 COIN from crypto COINUSDC
    t212_dir = tmp_path / "212trading"
    t212_dir.mkdir(parents=True, exist_ok=True)
    (t212_dir / ".state_COIN_US_EQ.json").write_text(
        json.dumps({"qty": 2.0, "cost_usd": 480.0, "avg_cost": 240.0, "entry_price": 240.0}),
        encoding="utf-8",
    )
    pt_crypto_only = {"COINUSDC": {"price": 1.25}}
    _, _, t212_cycs = harvest_venue_cycles(str(tmp_path), pt_crypto_only, {})
    assert t212_cycs["COIN_US_EQ"]["dd_pct"] is None

    bare_t212_inst = SimpleNamespace(
        symbol="COIN",
        provider_name="t212",
        base="COIN",
        quote="USD",
        enabled=True,
        _provider=SimpleNamespace(free_balance=lambda a: 2.0 if a == "COIN" else 100.0),
    )
    pos_out, _, _, _ = harvest_tradable_positions_and_orders(
        {"T212_COIN": bare_t212_inst},
        {"COINUSDC": {"price": 1.25}},
        {},
        {},
    )
    assert pos_out[0]["price"] is None

    # Allow-able match when COIN_US_EQ is present in prices_and_trends even with explicit_quote="USD"
    pos_out_eq, _, _, _ = harvest_tradable_positions_and_orders(
        {"T212_COIN": bare_t212_inst},
        {"COIN_US_EQ": {"price": 245.0}, "COINUSDC": {"price": 1.25}},
        {},
        {},
    )
    assert pos_out_eq[0]["price"] == pytest.approx(245.0)

    # 2. load_symbol_best_recent_fills filters unscoped quote_variants so COINUSDC never reads COIN_US_EQ or vice versa
    cdir_iso = tmp_path / "iso" / "cachedb"
    cdir_iso.mkdir(parents=True, exist_ok=True)
    (cdir_iso / "cache_trade_t212.json").write_text(
        json.dumps(
            {
                "items": {
                    "COINUSDC": [
                        {"symbol": "COINUSDC", "side": "BUY", "price": 1.25, "qty": 50.0, "executedQty": 50.0, "timestamp": 1_700_000_000_000}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_symbol_best_recent_fills(
        str(cdir_iso),
        "COIN_US_EQ",
        cache_symbol_variants("COIN_US_EQ", include_base=True),
    ) is None

    # 3. harvest_recent_orders_and_enrich_positions does not price bare t212 cycle from COINUSDC when recovering from venue_balances_state
    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "venue_balances_state.json").write_text(
        json.dumps({"pos:t212:COIN": {"venue": "t212", "symbol": "COIN", "free_base": 3.0}}),
        encoding="utf-8",
    )
    t212_bare_cyc = {"COIN": {"qty": 1.0, "avg_cost_px": 240.0, "entry_price": 240.0}}
    harvest_recent_orders_and_enrich_positions(
        str(cdir),
        {},
        [],
        {"COINUSDC": {"price": 1.25}},
        {},
        {},
        t212_bare_cyc,
    )
    assert t212_bare_cyc["COIN"]["qty"] == pytest.approx(3.0)
    assert t212_bare_cyc["COIN"].get("dd_pct") is None

    # 4. apply_asset_scope_filter classifies venue-less items by _EQ/_US_EQ symbol suffix
    ps_crypto = {
        "active_retry_queue": [
            {"symbol": "COIN_US_EQ", "side": "BUY", "qty": 1.0},
            {"symbol": "COINUSDC", "side": "BUY", "qty": 10.0},
        ],
        "last_cycle_execution_status": {"symbol": "COIN_US_EQ", "status": "EXECUTED"},
    }
    apply_asset_scope_filter({}, {}, ps_crypto, "crypto", {}, {})
    assert [x["symbol"] for x in ps_crypto["active_retry_queue"]] == ["COINUSDC"]
    assert "last_cycle_execution_status" not in ps_crypto

    ps_eq = {
        "active_retry_queue": [
            {"symbol": "COIN_US_EQ", "side": "BUY", "qty": 1.0},
            {"symbol": "COINUSDC", "side": "BUY", "qty": 10.0},
        ],
        "last_cycle_execution_status": {"symbol": "COIN_US_EQ", "status": "EXECUTED"},
    }
    apply_asset_scope_filter({}, {}, ps_eq, "equities", {}, {})
    assert [x["symbol"] for x in ps_eq["active_retry_queue"]] == ["COIN_US_EQ"]
    assert ps_eq.get("last_cycle_execution_status", {}).get("symbol") == "COIN_US_EQ"


def test_iter24_btc_phantom_lot_avg_reconciliation_eq_bid_ask_precision_and_daemon_timeout_recovery(tmp_path):
    import json
    from cachepaths import cache_symbol_variants
    from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
    from intelligence.sentiment.evaluators.microstructure_evaluator import align_equity_orderbook_to_spot
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # 1. Reconcile active_avg_buy_px when fresh open_lots cover >= 98% of wallet balance and unfiltered > 115%
    cdir = tmp_path / "cachedb"
    (cdir / "orders").mkdir(parents=True, exist_ok=True)
    now_ms = 1_794_190_000_000
    old_dust_ts = now_ms - 30 * 86400 * 1000
    fresh_buy_ts = now_ms - 8 * 3600 * 1000
    (cdir / "orders" / "cache_order.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {"orderId": 1, "symbol": "BTCUSDC", "side": "BUY", "status": "FILLED", "price": "77595.77", "executedQty": "0.02023", "time": old_dust_ts, "updateTime": old_dust_ts},
                        {"orderId": 2, "symbol": "BTCUSDC", "side": "BUY", "status": "FILLED", "price": "74356.88", "executedQty": "0.02367", "time": fresh_buy_ts, "updateTime": fresh_buy_ts},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (cdir / "venue_balances_state.json").write_text(
        json.dumps({"pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.023677, "locked_base": 0.0}}),
        encoding="utf-8",
    )
    btc_fills = load_symbol_best_recent_fills(
        str(cdir),
        "BTCUSDC",
        cache_symbol_variants("BTCUSDC", include_base=False),
    )
    assert btc_fills is not None
    assert btc_fills["active_avg_buy_px"] == pytest.approx(74356.88)
    assert len(btc_fills["open_lots"]) == 1
    assert btc_fills["open_lots"][0]["qty"] == pytest.approx(0.02367)

    # Also verify enrich_positions_with_orders_and_cycles preserves open_lots within 14d of newest_open_lot_ts even if last_buy was harvested later
    recent_orders = {
        "TAOUSDC": {
            "active_avg_buy_px": 250.365224,
            "active_cycle_rem_qty": 8.3306,
            "last_buy": {"price": 213.2, "qty": 1.7715, "ts_ms": now_ms - 3600_000},
            "open_lots": [
                {"lot_id": "L527", "qty": 1.4268, "price": 237.1, "ts_ms": now_ms - 6 * 86400 * 1000},
                {"lot_id": "L517", "qty": 0.5441, "price": 254.7, "ts_ms": now_ms - (14 * 86400 * 1000 + 600_000) - 3600_000},
            ],
        }
    }
    # Shift L517 so it is 7.6d older than L527, even though >14d older than harvested last_buy
    recent_orders["TAOUSDC"]["open_lots"][1]["ts_ms"] = recent_orders["TAOUSDC"]["open_lots"][0]["ts_ms"] - int(7.6 * 86400 * 1000)
    tao_pos = [{"venue": "binance", "symbol": "TAOUSDC", "free_base": 8.331, "locked_base": 0.0, "price": 215.2}]
    enrich_positions_with_orders_and_cycles(tao_pos, {}, {}, {"binance": recent_orders}, recent_orders, {}, {}, {}, {})
    assert len(recent_orders["TAOUSDC"]["open_lots"]) == 2
    assert len(tao_pos[0]["open_lots"]) == 2

    # 2. align_equity_orderbook_to_spot preserves 4-decimal precision when 2-decimal rounding collapses tight spread
    ob_nvda = {"best_bid": 186.76, "best_ask": 186.77, "ob_mid_price": 186.765, "spread_bps": 0.54}
    align_equity_orderbook_to_spot(ob_nvda, 175.96)
    assert ob_nvda["best_bid"] < ob_nvda["best_ask"]
    assert ob_nvda["best_bid"] == pytest.approx(175.9553, abs=1e-4)
    assert ob_nvda["best_ask"] == pytest.approx(175.9647, abs=1e-4)

    # 3. harvest_log_anomalies_from_dirs suppresses self-recovered antigravity-cli-daemon start timeout
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "manage_agy.log").write_text(
        "\n".join(
            [
                "Error: failed to start service: exit status 1 (output: Job for antigravity-cli-daemon.service failed because a timeout was exceeded.",
                "See \"systemctl status --user antigravity-cli-daemon.service\" for details.)",
                "[manage_agy] Remote-control daemon started.",
                "Daemon status: active",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    anoms = harvest_log_anomalies_from_dirs([str(log_dir)], 24.0)
    assert anoms.error_count == 0


