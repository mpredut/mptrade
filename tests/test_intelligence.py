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


if __name__ == "__main__":
    unittest.main()



