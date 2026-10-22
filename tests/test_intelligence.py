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
            self.assertTrue(path.endswith("kalman_state.json"))
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

    def _fake_review(*, fear_greed=None, breadth=None, whale_telemetry=None, force_refresh=False):
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


if __name__ == "__main__":
    unittest.main()

