"""Comprehensive unit tests for external intelligence (Pillar 2: liquidations, guards, telemetry)."""

import unittest
import time

from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector, LiquidationSummary
from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetry, DerivativesTelemetryCollector
from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.internal.guards.guard_decision import BrakeAction
from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerSide
from cachepaths import cache_symbol_variants, normalize_cache_base_asset
from intelligence.composite import CompositeMarketIntelligence
from intelligence.daemon import IntelligenceTelemetryDaemon
from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector, WhalePositioningSnapshot
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
from intelligence.macro_analyzer import MacroAnalyzer
from types import SimpleNamespace
import external_order_guard
import json
import order_guard
import os
import pytest
import tempfile
from intelligence.cli import inspect_symbol_intelligence
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger
from tests.conftest import MockHttpResponse
from unittest.mock import MagicMock, patch


class TestExternalCollectors(unittest.TestCase):
    def test_bybit_collector_record_and_summary(self):
        collector = BybitLiquidationCollector(symbols=["BTCUSDT"])
        t0 = time.time()

        # Record $200k in long liquidations and $50k in short liquidations
        collector.record_event("BTCUSDT", "LONG", 200_000.0, ts=t0)
        collector.record_event("BTCUSDT", "SHORT", 50_000.0, ts=t0)

        summary = collector.get_summary("BTCUSDT", window_sec=300.0)
        self.assertEqual(summary.symbol, "BTCUSDT")
        self.assertEqual(summary.long_liq_usd, 200_000.0)
        self.assertEqual(summary.short_liq_usd, 50_000.0)
        self.assertEqual(summary.events_count, 2)
        self.assertAlmostEqual(summary.capitulation_ratio, 200_000.0 / 250_000.0)

    def test_derivatives_telemetry_cache(self):
        collector = DerivativesTelemetryCollector(cache_ttl_sec=60.0)
        t0 = time.time()
        telem = DerivativesTelemetry(
            symbol="BTCUSDC",
            funding_rate=0.0001,
            predicted_funding_rate=0.00012,
            open_interest=50000.0,
            open_interest_usd=4_250_000_000.0,
            ts=t0,
        )
        collector.record_snapshot(telem)

        fetched = collector.fetch("BTCUSDC")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.funding_rate, 0.0001)
        self.assertEqual(fetched.open_interest_usd, 4_250_000_000.0)


class TestExternalTriggers(unittest.TestCase):
    def test_capitulation_burst_triggers_entry_buy(self):
        trigger = LiquidationCapitulationTrigger(min_notional_usd=100_000.0, ratio_threshold=0.75)
        # Panic capitulation: $300k long liquidations vs $20k short
        summary = LiquidationSummary(
            symbol="BTCUSDC",
            window_sec=300.0,
            long_liq_usd=300_000.0,
            short_liq_usd=20_000.0,
            net_usd=-280_000.0,
            capitulation_ratio=300_000.0 / 320_000.0,
            squeeze_ratio=20_000.0 / 320_000.0,
            events_count=10,
            last_event_ts=time.time(),
        )

        ev = trigger.evaluate("BTCUSDC", price=84000.0, summary=summary)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.action, TriggerAction.ENTRY)
        self.assertEqual(ev.side, TriggerSide.BUY)
        self.assertEqual(ev.reason, "liquidation_capitulation_burst")

    def test_short_squeeze_burst_triggers_exit_sell(self):
        trigger = LiquidationCapitulationTrigger(min_notional_usd=100_000.0, ratio_threshold=0.75)
        # Short squeeze: $20k long liquidations vs $350k short liquidations
        summary = LiquidationSummary(
            symbol="BTCUSDC",
            window_sec=300.0,
            long_liq_usd=20_000.0,
            short_liq_usd=350_000.0,
            net_usd=330_000.0,
            capitulation_ratio=20_000.0 / 370_000.0,
            squeeze_ratio=350_000.0 / 370_000.0,
            events_count=12,
            last_event_ts=time.time(),
        )

        ev = trigger.evaluate("BTCUSDC", price=86000.0, summary=summary)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.action, TriggerAction.EXIT)
        self.assertEqual(ev.side, TriggerSide.SELL)
        self.assertEqual(ev.reason, "liquidation_short_squeeze_burst")


class TestExternalGuards(unittest.TestCase):
    def test_cascade_guard_defers_during_active_falling_knife(self):
        guard = LiquidationCascadeGuard(max_active_cascade_usd=500_000.0, active_window_sec=60.0)

        # Active liquidation waterfall: $800k in 60s
        summary_storm = LiquidationSummary(
            symbol="BTCUSDC",
            window_sec=60.0,
            long_liq_usd=800_000.0,
            short_liq_usd=10_000.0,
            net_usd=-790_000.0,
            capitulation_ratio=0.98,
            squeeze_ratio=0.02,
            events_count=25,
            last_event_ts=time.time(),
        )

        dec = guard.check("BTCUSDC", "BUY", summary=summary_storm)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("active_long_liquidation_cascade", dec.reason)

        # Quiescent market: only $50k in 60s -> allowed
        summary_calm = LiquidationSummary(
            symbol="BTCUSDC",
            window_sec=60.0,
            long_liq_usd=50_000.0,
            short_liq_usd=10_000.0,
            net_usd=-40_000.0,
            capitulation_ratio=0.83,
            squeeze_ratio=0.17,
            events_count=3,
            last_event_ts=time.time(),
        )
        dec_calm = guard.check("BTCUSDC", "BUY", summary=summary_calm)
        self.assertTrue(dec_calm.allowed)
        self.assertEqual(dec_calm.brake_action, BrakeAction.NONE)

    def test_funding_crowding_guard_downscales(self):
        guard = FundingCrowdingGuard(
            max_long_funding_rate=0.0005,
            crowding_policy="downscale",
            crowded_scale=0.50,
        )

        # Extreme positive funding: +0.08% / 8h (hyper-crowded longs)
        telemetry_crowded = DerivativesTelemetry(
            symbol="BTCUSDC",
            funding_rate=0.0008,
            predicted_funding_rate=0.0008,
            open_interest=50000.0,
            open_interest_usd=4_250_000_000.0,
            ts=time.time(),
        )

        dec = guard.check("BTCUSDC", "BUY", telemetry=telemetry_crowded)
        self.assertTrue(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DOWNSCALE_QTY)
        self.assertEqual(dec.suggested_scale, 0.50)
        self.assertIn("long_crowding_extreme", dec.reason)

    def test_whale_divergence_guard_defers_short_covering(self):

        guard = WhaleDivergenceGuard()
        # Price rose, but OI dropped (-3.5%): short-covering fakeout!
        snapshot_fakeout = WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=1.2,
            top_traders_long_pct=0.54,
            taker_buy_sell_ratio=0.95,
            taker_buy_vol_usd=500_000.0,
            taker_sell_vol_usd=520_000.0,
            open_interest_usd=8_000_000_000.0,
            open_interest_1h_change_pct=-3.5,
            divergence_regime="short_covering",
            ts=time.time(),
        )

        dec = guard.check("BTCUSDC", "BUY", snapshot=snapshot_fakeout)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("short_covering_fakeout", dec.reason)

    def test_orderbook_wall_guard_defers_when_blocked(self):

        guard = OrderbookWallGuard(whale_wall_usd_limit=1_000_000.0)
        # Massive $2.5M ask wall sitting right above mid price
        snapshot_wall = OrderbookSnapshot(
            symbol="BTCUSDC",
            mid_price=85000.0,
            bid_depth_usd=500_000.0,
            ask_depth_usd=3_000_000.0,
            imbalance_ratio=0.14,
            largest_bid_wall_usd=200_000.0,
            largest_bid_wall_price=84900.0,
            largest_ask_wall_usd=2_500_000.0,
            largest_ask_wall_price=85100.0,
            ts=time.time(),
        )

        dec = guard.check("BTCUSDC", "BUY", snapshot=snapshot_wall)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.brake_action, BrakeAction.DEFER_WAIT)

    def test_disk_serialization_round_trip(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Whale collector disk round-trip
            w_collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=tmpdir)
            t0 = time.time()
            w_snap = WhalePositioningSnapshot(
                symbol="BTCUSDT",
                top_traders_long_ratio=1.75,
                top_traders_long_pct=0.63,
                taker_buy_sell_ratio=1.30,
                taker_buy_vol_usd=1_000_000.0,
                taker_sell_vol_usd=750_000.0,
                open_interest_usd=5_000_000_000.0,
                open_interest_1h_change_pct=2.1,
                divergence_regime="accumulation",
                ts=t0,
            )
            w_collector.record_snapshot(w_snap, save_to_disk=True)

            # Read back with a brand new collector instance
            w_collector2 = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=tmpdir)
            loaded_w = w_collector2.fetch("BTCUSDT")
            self.assertIsNotNone(loaded_w)
            self.assertEqual(loaded_w.symbol, "BTCUSDT")
            self.assertEqual(loaded_w.divergence_regime, "accumulation")

            # 2. Orderbook collector disk round-trip
            ob_collector = OrderbookDepthCollector(cache_ttl_sec=30.0, cache_dir=tmpdir)
            ob_snap = OrderbookSnapshot(
                symbol="ETHUSDT",
                mid_price=3500.0,
                bid_depth_usd=2_000_000.0,
                ask_depth_usd=1_500_000.0,
                imbalance_ratio=0.571,
                largest_bid_wall_usd=300_000.0,
                largest_bid_wall_price=3490.0,
                largest_ask_wall_usd=250_000.0,
                largest_ask_wall_price=3510.0,
                ts=t0,
            )
            ob_collector.record_snapshot(ob_snap, save_to_disk=True)

            ob_collector2 = OrderbookDepthCollector(cache_ttl_sec=30.0, cache_dir=tmpdir)
            loaded_ob = ob_collector2.fetch("ETHUSDT")
            self.assertIsNotNone(loaded_ob)
            self.assertEqual(loaded_ob.symbol, "ETHUSDT")
            self.assertAlmostEqual(loaded_ob.imbalance_ratio, 0.571)

            # 3. Derivatives telemetry disk round-trip
            d_collector = DerivativesTelemetryCollector(cache_ttl_sec=30.0, cache_dir=tmpdir)
            d_snap = DerivativesTelemetry(
                symbol="SOLUSDT",
                funding_rate=0.0002,
                predicted_funding_rate=0.00025,
                open_interest=100_000.0,
                open_interest_usd=15_000_000.0,
                ts=t0,
            )
            d_collector.record_snapshot(d_snap, save_to_disk=True)

            d_collector2 = DerivativesTelemetryCollector(cache_ttl_sec=30.0, cache_dir=tmpdir)
            loaded_d = d_collector2.fetch("SOLUSDT")
            self.assertIsNotNone(loaded_d)
            self.assertEqual(loaded_d.symbol, "SOLUSDT")
            self.assertEqual(loaded_d.funding_rate, 0.0002)


class TestOrderGuardPillar2Integration(unittest.TestCase):
    """Verify Pillar 2 guards wire cleanly into order_guard.check_intelligence_guards."""

    def test_whale_guard_shadow_allows_while_enforce_blocks(self):

        fakeout_snap = WhalePositioningSnapshot(
            symbol="BTCUSDT",
            top_traders_long_ratio=1.0,
            top_traders_long_pct=0.5,
            taker_buy_sell_ratio=0.8,
            taker_buy_vol_usd=500_000.0,
            taker_sell_vol_usd=600_000.0,
            open_interest_usd=5_000_000_000.0,
            open_interest_1h_change_pct=-4.0,
            divergence_regime="short_covering",
            ts=time.time(),
        )

        collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
        collector.record_snapshot(fakeout_snap)

        orig_collector = external_order_guard._whale_collector
        orig_margins = order_guard._MARGINS
        base_margins = dict(orig_margins or order_guard._load_margins())
        try:
            external_order_guard._whale_collector = collector

            # Test SHADOW mode: logs warning but allows trade
            order_guard._MARGINS = {
                **base_margins,
                "intelligence_guards_mode": "shadow",
                "whale_guard_mode": "shadow",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "off",
                "llm_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
            allowed_shadow, reason_shadow, scale_shadow = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertTrue(allowed_shadow)
            self.assertEqual(scale_shadow, 1.0)

            # Test ENFORCE mode: hard veto
            order_guard._MARGINS = {
                **base_margins,
                "intelligence_guards_mode": "enforce",
                "whale_guard_mode": "enforce",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "off",
                "llm_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
            allowed_enforce, reason_enforce, scale_enforce = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertFalse(allowed_enforce)
            self.assertIn("short_covering_fakeout", reason_enforce)
            self.assertEqual(scale_enforce, 0.0)
        finally:
            external_order_guard._whale_collector = orig_collector
            order_guard._MARGINS = orig_margins

    def test_orderbook_wall_guard_shadow_and_enforce(self):

        wall_snap = OrderbookSnapshot(
            symbol="BTCUSDT",
            mid_price=85000.0,
            bid_depth_usd=200_000.0,
            ask_depth_usd=2_500_000.0,
            imbalance_ratio=0.074,
            largest_bid_wall_usd=50_000.0,
            largest_bid_wall_price=84900.0,
            largest_ask_wall_usd=2_000_000.0,
            largest_ask_wall_price=85100.0,
            ts=time.time(),
        )

        collector = OrderbookDepthCollector(cache_ttl_sec=60.0, cache_dir=None)
        collector.record_snapshot(wall_snap)

        orig_collector = external_order_guard._orderbook_collector
        orig_margins = order_guard._MARGINS
        base_margins = dict(orig_margins or order_guard._load_margins())
        try:
            external_order_guard._orderbook_collector = collector

            # Test SHADOW mode
            order_guard._MARGINS = {
                **base_margins,
                "intelligence_guards_mode": "shadow",
                "whale_guard_mode": "off",
                "orderbook_wall_guard_mode": "shadow",
                "whale_wall_usd_limit": 1_000_000.0,
                "funding_guard_mode": "off",
                "llm_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
            allowed_shadow, _, scale_shadow = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertTrue(allowed_shadow)
            self.assertEqual(scale_shadow, 1.0)

            # Test ENFORCE mode
            order_guard._MARGINS["orderbook_wall_guard_mode"] = "enforce"
            allowed_enforce, reason_enforce, scale_enforce = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertFalse(allowed_enforce)
            self.assertIn("orderbook_heavily_ask_dominated", reason_enforce)
            self.assertEqual(scale_enforce, 0.0)
        finally:
            external_order_guard._orderbook_collector = orig_collector
            order_guard._MARGINS = orig_margins

    def test_funding_crowding_guard_downscales_in_enforce(self):

        crowded_snap = DerivativesTelemetry(
            symbol="BTCUSDT",
            funding_rate=0.0008,
            predicted_funding_rate=0.0008,
            open_interest=50000.0,
            open_interest_usd=4_250_000_000.0,
            ts=time.time(),
        )

        collector = DerivativesTelemetryCollector(cache_ttl_sec=60.0, cache_dir=None)
        collector.record_snapshot(crowded_snap)

        orig_collector = external_order_guard._derivatives_collector
        orig_margins = order_guard._MARGINS
        base_margins = dict(orig_margins or order_guard._load_margins())
        try:
            external_order_guard._derivatives_collector = collector

            # Test SHADOW mode
            order_guard._MARGINS = {
                **base_margins,
                "intelligence_guards_mode": "shadow",
                "whale_guard_mode": "off",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "shadow",
                "funding_max_long_rate": 0.0005,
                "funding_crowded_scale": 0.50,
                "llm_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
            allowed_shadow, _, scale_shadow = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertTrue(allowed_shadow)
            self.assertEqual(scale_shadow, 1.0)

            # Test ENFORCE mode: allows but downscales to 0.50
            order_guard._MARGINS["funding_guard_mode"] = "enforce"
            allowed_enforce, reason_enforce, scale_enforce = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertTrue(allowed_enforce)
            self.assertIn("long_crowding_extreme", reason_enforce)
            self.assertEqual(scale_enforce, 0.50)
        finally:
            external_order_guard._derivatives_collector = orig_collector
            order_guard._MARGINS = orig_margins

    def test_whale_guard_severe_and_moderate_taker_selling(self):

        collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
        orig_collector = external_order_guard._whale_collector
        orig_margins = order_guard._MARGINS
        base_margins = dict(orig_margins or order_guard._load_margins())
        try:
            external_order_guard._whale_collector = collector
            order_guard._MARGINS = {
                **base_margins,
                "intelligence_guards_mode": "enforce",
                "whale_guard_mode": "enforce",
                "min_taker_ratio": 0.75,
                "taker_downscale_threshold": 0.90,
                "taker_downscaled_scale": 0.50,
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "off",
                "llm_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }

            # 1. Severe taker selling panic (< 0.75): DEFER_WAIT / VETO
            severe_snap = WhalePositioningSnapshot(
                symbol="BTCUSDT",
                top_traders_long_ratio=1.5,
                top_traders_long_pct=0.60,
                taker_buy_sell_ratio=0.72,
                taker_buy_vol_usd=400_000.0,
                taker_sell_vol_usd=555_000.0,
                open_interest_usd=5_000_000_000.0,
                open_interest_1h_change_pct=0.1,
                divergence_regime="neutral",
                ts=time.time(),
            )
            collector.record_snapshot(severe_snap)
            ok, reason, scale = order_guard.check_intelligence_guards(None, "BTCUSDT", "BUY", 85000.0)
            self.assertFalse(ok)
            self.assertIn("severe_taker_selling_dominated", reason)
            self.assertEqual(scale, 0.0)

            # 2. Moderate taker selling (0.75 <= ratio < 0.90): DOWNSCALE to 0.50
            moderate_snap = WhalePositioningSnapshot(
                symbol="BTCUSDT",
                top_traders_long_ratio=1.5,
                top_traders_long_pct=0.60,
                taker_buy_sell_ratio=0.82,
                taker_buy_vol_usd=450_000.0,
                taker_sell_vol_usd=550_000.0,
                open_interest_usd=5_000_000_000.0,
                open_interest_1h_change_pct=0.1,
                divergence_regime="neutral",
                ts=time.time(),
            )
            collector.record_snapshot(moderate_snap)
            ok, reason, scale = order_guard.check_intelligence_guards(None, "BTCUSDT", "BUY", 85000.0)
            self.assertTrue(ok)
            self.assertIn("moderate_taker_selling_pressure", reason)
            self.assertEqual(scale, 0.50)

            # 3. Healthy taker volume (>= 0.90): ALLOW 1.0
            healthy_snap = WhalePositioningSnapshot(
                symbol="BTCUSDT",
                top_traders_long_ratio=1.5,
                top_traders_long_pct=0.60,
                taker_buy_sell_ratio=1.05,
                taker_buy_vol_usd=525_000.0,
                taker_sell_vol_usd=500_000.0,
                open_interest_usd=5_000_000_000.0,
                open_interest_1h_change_pct=0.1,
                divergence_regime="neutral",
                ts=time.time(),
            )
            collector.record_snapshot(healthy_snap)
            ok, reason, scale = order_guard.check_intelligence_guards(None, "BTCUSDT", "BUY", 85000.0)
            self.assertTrue(ok)
            self.assertEqual(scale, 1.0)
        finally:
            external_order_guard._whale_collector = orig_collector
            order_guard._MARGINS = orig_margins

    def test_rtrade_whale_flow_checks(self):
        from rtrade import _whale_flow_blocks_side, _effective_adjustment_fraction

        collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
        orig_collector = external_order_guard._whale_collector
        try:
            external_order_guard._whale_collector = collector

            # Taker selling (< 0.80) blocks BUY
            dumping_snap = WhalePositioningSnapshot(
                symbol="TAOUSDC",
                top_traders_long_ratio=2.0,
                top_traders_long_pct=0.66,
                taker_buy_sell_ratio=0.74,
                taker_buy_vol_usd=100_000.0,
                taker_sell_vol_usd=135_000.0,
                open_interest_usd=50_000_000.0,
                open_interest_1h_change_pct=0.0,
                divergence_regime="neutral",
                ts=time.time(),
            )
            collector.record_snapshot(dumping_snap)
            blocked, b_reason = _whale_flow_blocks_side("TAOUSDC", "BUY")
            self.assertTrue(blocked)
            self.assertIn("taker_sell_dominance", b_reason)

            # SELL is unaffected by taker selling
            blocked_sell, _ = _whale_flow_blocks_side("TAOUSDC", "SELL")
            self.assertFalse(blocked_sell)

            # Symmetric SELL check: Taker buying (> 1.25) blocks SELL while BUY is unaffected
            buying_snap = WhalePositioningSnapshot(
                symbol="TAOUSDC",
                top_traders_long_ratio=2.0,
                top_traders_long_pct=0.66,
                taker_buy_sell_ratio=1.35,
                taker_buy_vol_usd=135_000.0,
                taker_sell_vol_usd=100_000.0,
                open_interest_usd=50_000_000.0,
                open_interest_1h_change_pct=0.0,
                divergence_regime="neutral",
                ts=time.time(),
            )
            collector.record_snapshot(buying_snap)
            blocked_sell_squeeze, s_reason = _whale_flow_blocks_side("TAOUSDC", "SELL")
            self.assertTrue(blocked_sell_squeeze)
            self.assertIn("taker_buy_dominance", s_reason)
            blocked_buy_on_squeeze, _ = _whale_flow_blocks_side("TAOUSDC", "BUY")
            self.assertFalse(blocked_buy_on_squeeze)

            # Moderate taker ratio widens adjustment by 1.5x
            moderate_snap = WhalePositioningSnapshot(
                symbol="TAOUSDC",
                top_traders_long_ratio=2.0,
                top_traders_long_pct=0.66,
                taker_buy_sell_ratio=0.85,
                taker_buy_vol_usd=100_000.0,
                taker_sell_vol_usd=117_000.0,
                open_interest_usd=50_000_000.0,
                open_interest_1h_change_pct=0.0,
                divergence_regime="neutral",
                ts=time.time(),
            )
            collector.record_snapshot(moderate_snap)
            adj = _effective_adjustment_fraction("TAOUSDC", 0.004)
            self.assertAlmostEqual(adj, 0.006)
        finally:
            external_order_guard._whale_collector = orig_collector

    def test_whale_positioning_regime_classification_short_squeeze_and_unwinding(self):
        """Verify regime classification handles short squeeze and distinguishes liquidations from unwinding."""

        # 1. Price rising + OI falling with aggressive taker buys = short_squeeze
        reg_squeeze = WhalePositioningCollector.classify_regime(
            price_1h_change_pct=1.2,
            oi_change_pct=-3.5,
            taker_ratio=1.45,
        )
        self.assertEqual(reg_squeeze, "short_squeeze")

        # 2. Price rising + OI falling with passive taker flow = short_covering (fakeout)
        reg_fakeout = WhalePositioningCollector.classify_regime(
            price_1h_change_pct=1.2,
            oi_change_pct=-3.5,
            taker_ratio=0.95,
        )
        self.assertEqual(reg_fakeout, "short_covering")

        # 3. Price rising + OI rising with negative funding rate = short_squeeze_primed
        reg_trapped = WhalePositioningCollector.classify_regime(
            price_1h_change_pct=0.8,
            oi_change_pct=1.5,
            taker_ratio=1.25,
            funding_rate=-0.0002,
        )
        self.assertEqual(reg_trapped, "short_squeeze_primed")

        # 4. Price rising + OI rising with positive funding rate = accumulation
        reg_accum = WhalePositioningCollector.classify_regime(
            price_1h_change_pct=0.8,
            oi_change_pct=1.5,
            taker_ratio=1.25,
            funding_rate=0.0001,
        )
        self.assertEqual(reg_accum, "accumulation")

        # 5. Price falling + OI falling with severe destruction = long_liquidation
        reg_liq = WhalePositioningCollector.classify_regime(
            price_1h_change_pct=-2.5,
            oi_change_pct=-6.0,
            taker_ratio=0.65,
        )
        self.assertEqual(reg_liq, "long_liquidation")

        # 6. Price falling + OI falling gently with healthy taker ratio = long_unwinding (orderly closure)
        reg_unwind = WhalePositioningCollector.classify_regime(
            price_1h_change_pct=-0.5,
            oi_change_pct=-1.2,
            taker_ratio=0.92,
        )
        self.assertEqual(reg_unwind, "long_unwinding")

    def test_whale_divergence_guard_short_squeeze_and_unwinding(self):
        """Verify WhaleDivergenceGuard allows short squeeze buys, defers premature sells, and permits long unwinding."""

        guard = WhaleDivergenceGuard()

        # Active short squeeze: BUY is ALLOWED (not blocked as fakeout)
        snap_squeeze = WhalePositioningSnapshot(
            symbol="BTCUSDT",
            top_traders_long_ratio=1.5,
            top_traders_long_pct=0.60,
            taker_buy_sell_ratio=1.40,
            taker_buy_vol_usd=2_000_000.0,
            taker_sell_vol_usd=1_420_000.0,
            open_interest_usd=5_000_000_000.0,
            open_interest_1h_change_pct=-4.0,
            divergence_regime="short_squeeze",
            ts=time.time(),
        )
        dec_buy = guard.check("BTCUSDT", "BUY", snapshot=snap_squeeze)
        self.assertTrue(dec_buy.allowed)
        self.assertEqual(dec_buy.brake_action, BrakeAction.NONE)

        # Active short squeeze: SELL is DEFERRED to let squeeze run
        dec_sell = guard.check("BTCUSDT", "SELL", snapshot=snap_squeeze)
        self.assertFalse(dec_sell.allowed)
        self.assertEqual(dec_sell.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("whale_squeeze_or_accumulation_active", dec_sell.reason)

        # Orderly long unwinding: BUY is allowed with clean scaling
        snap_unwind = WhalePositioningSnapshot(
            symbol="BTCUSDT",
            top_traders_long_ratio=1.2,
            top_traders_long_pct=0.55,
            taker_buy_sell_ratio=0.95,
            taker_buy_vol_usd=1_000_000.0,
            taker_sell_vol_usd=1_050_000.0,
            open_interest_usd=5_000_000_000.0,
            open_interest_1h_change_pct=-1.5,
            divergence_regime="long_unwinding",
            ts=time.time(),
        )
        dec_unwind = guard.check("BTCUSDT", "BUY", snapshot=snap_unwind)
        self.assertTrue(dec_unwind.allowed)
        self.assertEqual(dec_unwind.brake_action, BrakeAction.NONE)

    def test_bybit_collector_symbol_variants_and_cascade_guard_sell_symmetry(self):
        collector = BybitLiquidationCollector(symbols=["BTCUSDT"])
        t0 = time.time()
        collector.record_event("BTCUSDT", "SHORT", 650_000.0, ts=t0)
        collector.record_event("BTCUSDT", "LONG", 50_000.0, ts=t0)

        for variant in ("BTCUSDC", "BTCUSD", "BTC/USDC", "BTC"):
            summary = collector.get_summary(variant, window_sec=60.0)
            self.assertEqual(summary.short_liq_usd, 650_000.0)
            self.assertEqual(summary.long_liq_usd, 50_000.0)

        guard = LiquidationCascadeGuard(collector=collector, max_active_cascade_usd=500_000.0, active_window_sec=60.0)
        dec_sell = guard.check("BTCUSDC", "SELL")
        self.assertFalse(dec_sell.allowed)
        self.assertEqual(dec_sell.brake_action, BrakeAction.DEFER_WAIT)
        self.assertIn("active_short_liquidation_cascade", dec_sell.reason)

        dec_buy = guard.check("BTCUSDC", "BUY")
        self.assertTrue(dec_buy.allowed)

    def test_volatility_cache_symbol_variants_and_composite_high_stake_default(self):
        from intelligence.internal.state.volatility import resolve_cached_volatility_1h
        from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard

        with tempfile.TemporaryDirectory() as tmpdir:
            kalman_path = os.path.join(tmpdir, "cache_kalman_trend.json")
            with open(kalman_path, "w", encoding="utf-8") as f:
                json.dump({"TAOUSDC": {"vol_1h_pct": 3.2}, "HYPEUSDC": {"vol_1h_pct": 4.5}}, f)
            self.assertAlmostEqual(resolve_cached_volatility_1h("TAOUSD", state_path=kalman_path), 0.032)
            self.assertAlmostEqual(resolve_cached_volatility_1h("TAO/USD", state_path=kalman_path), 0.032)
            self.assertAlmostEqual(resolve_cached_volatility_1h("HYPE/USDC", state_path=kalman_path), 0.045)
            self.assertAlmostEqual(resolve_cached_volatility_1h("HYPE", state_path=kalman_path), 0.045)

        comp = CompositeMarketIntelligence()
        self.assertEqual(comp.gemini_guard.min_notional_eur, 700.0)

        g1 = get_post_stop_reentry_guard(margins={
            "post_stop_reentry_mode": "enforce",
            "post_stop_reentry_timeout_sec": 20.0,
            "post_stop_reentry_fallback": "allow",
            "post_stop_reentry_notify": "false",
            "post_stop_reentry_cache_ttl_sec": 300.0,
        })
        self.assertEqual(g1.mode, "enforce")
        g2 = get_post_stop_reentry_guard(margins={
            "post_stop_reentry_mode": "shadow",
            "post_stop_reentry_timeout_sec": 12.0,
            "post_stop_reentry_fallback": "veto",
            "post_stop_reentry_notify": "true",
            "post_stop_reentry_cache_ttl_sec": 120.0,
        })
        self.assertIs(g1, g2)
        self.assertEqual(g2.mode, "shadow")
        self.assertEqual(g2.timeout_sec, 12.0)
        self.assertEqual(g2.fallback_action, "veto")

    def test_monitortrades_stale_sl_bounce_buy_price_does_not_bypass_gt_s_timer(self):
        import monitortrades as mt

        sym = "TESTSLUSDC"
        now = 1_700_000_000.0
        mt._SL_BOUNCE_STATE[sym] = {"buy_price": 120.0, "started_at": now - 600.0}
        mt._SL_EPISODE_STATE[sym] = {"buy_price": 100.0, "started_at": now - 50.0, "disaster_floor": 0.065}
        try:
            snap = mt.PositionSnapshot(
                symbol=sym,
                current_price=92.0,
                avg_buy_price=100.0,
                net_qty=2.0,
                sellable_qty=2.0,
                current_time_s=now,
                trend_up=False,
            )
            dec = mt.ExitDecision(
                action=mt.ExitAction.SELL,
                should_exit=True,
                reason="hard_disaster_floor_breached",
                forced=True,
                metadata={"priority": 1},
            )
            inst = MagicMock()
            inst.symbol = sym
            inst.isolation = "shared"
            inst.free.return_value = 2.0
            with patch.object(mt, "_is_trend_gate_deferred", return_value=True), \
                 patch.object(mt, "_save_mt_state"):
                placed = mt._execute_stop_loss(
                    inst, snap, dec, 3600.0, 0.001, 0.05,
                    dict(mt._SL_BOUNCE_STATE[sym]),
                )
            self.assertFalse(placed)
            # Stale buy_price=120.0 must be rejected and replaced by current episode started_at (now - 50.0)
            self.assertEqual(mt._SL_BOUNCE_STATE[sym]["buy_price"], 100.0)
            self.assertEqual(mt._SL_BOUNCE_STATE[sym]["started_at"], now - 50.0)
        finally:
            mt._SL_BOUNCE_STATE.pop(sym, None)
            mt._SL_EPISODE_STATE.pop(sym, None)

    def test_zero_imbalance_and_zero_taker_ratio_not_coerced_to_neutral_defaults(self):
        from botcore import safe_nonneg_int

        ob_guard = OrderbookWallGuard(min_buy_imbalance=0.25)
        ob_dec = ob_guard.check(
            "BTCUSDC",
            "BUY",
            snapshot={"imbalance_ratio": 0.0, "largest_ask_wall_usd": 0.0},
        )
        self.assertFalse(ob_dec.allowed)
        self.assertIn("orderbook_heavily_ask_dominated", ob_dec.reason)

        whale_guard = WhaleDivergenceGuard(min_taker_ratio=0.75)
        w_dec = whale_guard.check(
            "BTCUSDC",
            "BUY",
            snapshot={"divergence_regime": "neutral", "taker_buy_sell_ratio": 0.0, "top_traders_long_pct": 0.0},
        )
        self.assertFalse(w_dec.allowed)
        self.assertIn("severe_taker_selling_dominated", w_dec.reason)

        self.assertEqual(safe_nonneg_int(0, default=3), 0)
        self.assertEqual(safe_nonneg_int(-1, default=3), 3)

    def test_collector_exact_venue_cache_isolation_and_stale_first_skip(self):
        from intelligence.sentiment.evaluators.microstructure_evaluator import (
            compute_liquidation_clusters,
            enrich_microstructure_cvd,
        )
        from intelligence.sentiment.evaluators.tca_risk_evaluator import compute_enterprise_portfolio_risk_metrics
        import cacheprice as cp

        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmpdir:
            ob_col = OrderbookDepthCollector(cache_ttl_sec=30.0, disk_ttl_sec=60.0, cache_dir=tmpdir)
            # 1. Stale exact entry in memory must not shadow a fresh compatible variant when allow_network=False
            ob_col.record_snapshot(
                OrderbookSnapshot(
                    symbol="TAOUSD",
                    mid_price=400.0,
                    bid_depth_usd=50_000.0,
                    ask_depth_usd=50_000.0,
                    imbalance_ratio=0.5,
                    largest_bid_wall_usd=10_000.0,
                    largest_bid_wall_price=399.0,
                    largest_ask_wall_usd=10_000.0,
                    largest_ask_wall_price=401.0,
                    ts=now - 120.0,
                )
            )
            ob_col.record_snapshot(
                OrderbookSnapshot(
                    symbol="TAOUSDC",
                    mid_price=455.0,
                    bid_depth_usd=80_000.0,
                    ask_depth_usd=80_000.0,
                    imbalance_ratio=0.5,
                    largest_bid_wall_usd=15_000.0,
                    largest_bid_wall_price=454.0,
                    largest_ask_wall_usd=15_000.0,
                    largest_ask_wall_price=456.0,
                    ts=now - 5.0,
                ),
                save_to_disk=True,
            )
            hit_ro = ob_col.fetch("TAOUSD", allow_network=False, now=now)
            self.assertIsNotNone(hit_ro)
            self.assertAlmostEqual(hit_ro.mid_price, 455.0)

            # 2. When allow_network=True on a multi-venue symbol (TAOUSD on Kraken), fresh disk TAOUSDC
            # must NOT pollute _cache['TAOUSD'] or be returned on Kraken network error.
            ob_col2 = OrderbookDepthCollector(cache_ttl_sec=30.0, disk_ttl_sec=60.0, cache_dir=tmpdir)
            with patch.object(ob_col2, "_fetch_json_dict", side_effect=RuntimeError("network down")):
                res_fail = ob_col2.fetch("TAOUSD", allow_network=True, now=now)
            self.assertIsNone(res_fail)
            self.assertNotIn("TAOUSD", ob_col2._cache)

            # 3. Hyperliquid whale fallback preserves 1h OI change on sub-15m unchanged OI and handles 0 buy volume
            w_col = WhalePositioningCollector(cache_ttl_sec=60.0, disk_ttl_sec=180.0, cache_dir=tmpdir)
            w_col.record_snapshot(
                WhalePositioningSnapshot(
                    symbol="HYPEUSDC",
                    top_traders_long_ratio=1.0,
                    top_traders_long_pct=0.5,
                    taker_buy_sell_ratio=1.1,
                    taker_buy_vol_usd=10_000.0,
                    taker_sell_vol_usd=9_000.0,
                    open_interest_usd=100_000_000.0,
                    open_interest_1h_change_pct=4.5,
                    divergence_regime="accumulation",
                    ts=now - 60.0,
                ),
                save_to_disk=True,
            )
            with patch.object(
                w_col,
                "_fetch_json_dict",
                return_value={"result_list": [{"px": "25.0", "sz": "100.0", "side": "A"}]},
            ), patch.object(
                w_col,
                "_fetch_hyperliquid_asset_ctx",
                return_value={"openInterest": "4000000.0", "markPx": "25.0", "funding": "0.0001"},
            ):
                hl_w = w_col._fetch_hyperliquid_whale_fallback("HYPEUSDC", -1.2, now, None)
            self.assertIsNotNone(hl_w)
            self.assertAlmostEqual(hl_w.taker_buy_sell_ratio, 0.0)
            self.assertAlmostEqual(hl_w.open_interest_1h_change_pct, 4.5)
            self.assertEqual(hl_w.divergence_regime, "aggressive_shorting")
            self.assertAlmostEqual(hl_w.cvd_5m_usd, -2500.0)

        # 4. Zero ob_imbalance and zero taker_buy_sell_ratio are preserved in microstructure CVD & liq clusters
        minfo = {"ob_imbalance": 0.0, "taker_buy_vol_usd": 1000.0, "taker_sell_vol_usd": 500.0}
        enrich_microstructure_cvd(minfo, pinfo={})
        self.assertEqual(minfo["cvd_regime"], "ask_absorbing")

        # 5. Enterprise portfolio VaR treats same-base holdings across venues as rho=1.0 and preserves 0.0 beta/corr
        risk_single = compute_enterprise_portfolio_risk_metrics(
            {
                "tradable_positions": [
                    {"venue": "binance", "symbol": "TAOUSDC", "value_usd": 1000.0},
                ],
                "venue_quote_cash": {"binance": {"venue": "binance", "free": 1000.0}},
            },
            {"TAOUSDC": {"beta_btc_24h": 0.0, "corr_btc_24h": 0.0, "vol_1h_pct": 2.0}},
            {},
        )
        risk_split = compute_enterprise_portfolio_risk_metrics(
            {
                "tradable_positions": [
                    {"venue": "binance", "symbol": "TAOUSDC", "value_usd": 600.0},
                    {"venue": "kraken", "symbol": "TAOUSD", "value_usd": 400.0},
                ],
                "venue_quote_cash": {"binance": {"venue": "binance", "free": 1000.0}},
            },
            {"TAOUSDC": {"beta_btc_24h": 0.0, "corr_btc_24h": 0.0, "vol_1h_pct": 2.0}},
            {},
        )
        self.assertEqual(risk_single["portfolio_btc_beta"], 0.0)
        self.assertEqual(risk_single["avg_corr_vs_btc_24h"], 0.0)
        self.assertAlmostEqual(risk_single["intraday_var_95_usd"], risk_split["intraday_var_95_usd"], places=2)

        # 6. HyperliquidPriceStream scopes provider _pair() to supported symbol only
        fake_prov = MagicMock()
        fake_prov.supports_symbol.side_effect = lambda s: s == "HYPEUSDC"
        fake_prov._pair.return_value = "@107"
        fake_api = MagicMock()
        fake_api.provider_by_name.return_value = fake_prov
        hl_stream = cp.HyperliquidPriceStream(["HYPEUSDC", "PURRUSDC"], market_api=fake_api)
        self.assertIn("@107", hl_stream._candidate_mids_keys("HYPEUSDC"))
        self.assertNotIn("@107", hl_stream._candidate_mids_keys("PURRUSDC"))


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_whalepositioningcollector__resolve_price_1h_change_pct_supports_seconds_timestamps_iter6_s3(tmp_path):
    """[ITER6 #3] WhalePositioningCollector._resolve_price_1h_change_pct supports seconds timestamps."""
    # 3. WhalePositioningCollector._resolve_price_1h_change_pct supports seconds timestamps
    c24_dir = tmp_path / "c24"
    c24_dir.mkdir()
    now_s = 1_700_010_000.0
    (c24_dir / "cache_24_price_TAOUSDC.json").write_text(
        json.dumps({
            "items": {
                "TAOUSDC": [
                    [now_s - 7200.0, 200.0],  # 2h ago: 200
                    [now_s - 3600.0, 100.0],  # 1h ago: 100
                    [now_s, 110.0],           # now: 110 (+10% over 1h, NOT -45% vs 2h ago!)
                ]
            }
        }),
        encoding="utf-8",
    )
    w_col = WhalePositioningCollector(cache_dir=str(c24_dir))
    assert w_col._resolve_price_1h_change_pct("TAOUSD") == pytest.approx(10.0)

def test_orderbookdepthcollector_ttl_on_direct_orderbooksnapshot_underscore_symbol_variants_iter9_s2(tmp_path):
    """[ITER9 #2] OrderbookDepthCollector TTL on direct OrderbookSnapshot & underscore symbol variants."""
    # 2. OrderbookDepthCollector TTL on direct OrderbookSnapshot & underscore symbol variants
    assert normalize_cache_base_asset("TAO_USD") == "TAO"
    assert normalize_cache_base_asset("BTCFDUSD") == "BTC"
    assert "TAOUSDC" in cache_symbol_variants("TAO_USD")
    ob_col = OrderbookDepthCollector(cache_ttl_sec=5.0, cache_dir=str(tmp_path))
    ob_col._cache["BTCUSDC"] = OrderbookSnapshot(
        symbol="BTCUSDC", mid_price=85000.0, bid_depth_usd=1000.0, ask_depth_usd=1000.0,
        imbalance_ratio=0.5, largest_bid_wall_usd=100.0, largest_bid_wall_price=84900.0,
        largest_ask_wall_usd=100.0, largest_ask_wall_price=85100.0, ts=1.0,
    )
    assert ob_col.fetch("BTCUSDC", allow_network=False) is None

def test_whalepositioningcollector_recovers_previous_on_corrupted_primary_json_and_iter11_s3(tmp_path):
    """[ITER11 #3] WhalePositioningCollector recovers .previous on corrupted primary JSON and sanitizes symbol in _resolve_cached_funding_rate."""
    # 3. WhalePositioningCollector recovers .previous on corrupted primary JSON and sanitizes symbol in _resolve_cached_funding_rate
    wc = WhalePositioningCollector(cache_dir=str(tmp_path))
    corrupt_p24 = tmp_path / "cache_24_price_BTCUSDC.json"
    prev_p24 = tmp_path / "cache_24_price_BTCUSDC.json.previous"
    corrupt_p24.write_text("{corrupt", encoding="utf-8")
    prev_p24.write_text(
        json.dumps({"items": {"BTCUSDC": [[1700000000000, 100.0], [1700003600000, 105.0]]}}),
        encoding="utf-8",
    )
    assert wc._resolve_price_1h_change_pct("BTC/USDC") == pytest.approx(5.0)

    deriv_p = tmp_path / "derivatives_telemetry_BTCUSDC.json"
    deriv_p.write_text(json.dumps({"ts": 1700003600.0, "funding_rate": 0.00042}), encoding="utf-8")
    assert wc._resolve_cached_funding_rate("BTC/USDC", now=1700003610.0) == pytest.approx(0.00042)

def test_external_order_guard_preserves_active_reason_of_the_stricter_upstream_downscale_iter11_s4(monkeypatch):
    """[ITER11 #4] external_order_guard preserves active_reason of the stricter upstream downscale."""
    # 4. external_order_guard preserves active_reason of the stricter upstream downscale
    monkeypatch.setattr(
        external_order_guard,
        "get_whale_collector",
        lambda: SimpleNamespace(
            fetch=lambda *_a, **_k: SimpleNamespace(
                divergence_regime="neutral",
                open_interest_1h_change_pct=0.0,
                taker_buy_sell_ratio=0.80,
                top_traders_long_pct=0.50,
                top_traders_long_ratio=1.0,
            )
        ),
    )
    monkeypatch.setattr(
        external_order_guard,
        "get_orderbook_collector",
        lambda: SimpleNamespace(fetch=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(
        external_order_guard,
        "get_derivatives_collector",
        lambda: SimpleNamespace(
            fetch=lambda *_a, **_k: DerivativesTelemetry(
                symbol="BTCUSDC",
                funding_rate=0.0008,
                predicted_funding_rate=0.0008,
                open_interest=1000.0,
                open_interest_usd=100000.0,
                ts=1700000000.0,
            )
        ),
    )
    m_cfg = dict(order_guard._load_margins())
    m_cfg.update(
        {
            "external_guard_mode": "enforce",
            "whale_guard_mode": "enforce",
            "orderbook_wall_guard_mode": "off",
            "funding_guard_mode": "enforce",
            "min_taker_ratio": 0.70,
            "taker_downscale_threshold": 0.90,
            "taker_downscaled_scale": 0.40,
            "funding_max_long_rate": 0.0005,
            "funding_crowding_policy": "downscale",
            "funding_crowded_scale": 0.75,
        }
    )
    allowed, reason, eff_scale = external_order_guard.check_external_order_guards(
        "BTCUSDC", "BUY", 100.0, margins=m_cfg, now=1700000000.0
    )
    assert allowed is True
    assert eff_scale == pytest.approx(0.40)
    assert "moderate_taker_selling_pressure" in reason

def test_external_collectors_usdcusdt_preserved_in__extract_base_asset_clock_skew_iter16_s3(tmp_path):
    """[ITER16 #3] External collectors: USDCUSDT preserved in _extract_base_asset, clock skew rejected, .previous written."""
    # 3. External collectors: USDCUSDT preserved in _extract_base_asset, clock skew rejected, .previous written

    assert normalize_cache_base_asset("USDCUSDT") == "USDCUSDT"

    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), cache_ttl_sec=60, disk_ttl_sec=60)
    ob_snap_valid = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=1000.0,
        ask_depth_usd=1000.0,
        imbalance_ratio=0.5,
        largest_bid_wall_usd=500.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=500.0,
        largest_ask_wall_price=60100.0,
        ts=1000.0,
    )
    ob_col._save_to_disk(ob_snap_valid)
    ob_snap_future = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=1000.0,
        ask_depth_usd=1000.0,
        imbalance_ratio=0.5,
        largest_bid_wall_usd=500.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=500.0,
        largest_ask_wall_price=60100.0,
        ts=2000.0,  # future timestamp relative to now=1005.0 -> clock skew
    )
    ob_col._save_to_disk(ob_snap_future)
    assert (tmp_path / "orderbook_depth_BTCUSDC.json").exists()
    # Future-dated snapshot is rejected on load at now=1005.0
    assert ob_col._load_from_disk("BTCUSDC", now=1005.0) is None

    whale_col = WhalePositioningCollector(cache_dir=str(tmp_path), cache_ttl_sec=60, disk_ttl_sec=60)
    whale_snap_future = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.0,
        top_traders_long_pct=0.5,
        taker_buy_sell_ratio=1.0,
        taker_buy_vol_usd=100.0,
        taker_sell_vol_usd=100.0,
        open_interest_usd=1000.0,
        open_interest_1h_change_pct=0.0,
        divergence_regime="neutral",
        ts=2000.0,
    )
    whale_col._save_to_disk(whale_snap_future)
    assert whale_col._load_from_disk("BTCUSDC", now=1005.0) is None

    deriv_col = DerivativesTelemetryCollector(cache_dir=str(tmp_path), cache_ttl_sec=60, disk_ttl_sec=60)
    deriv_snap_future = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.001,
        predicted_funding_rate=0.001,
        open_interest=100.0,
        open_interest_usd=6000000.0,
        ts=2000.0,
    )
    deriv_col._save_to_disk(deriv_snap_future)
    assert deriv_col._load_from_disk("BTCUSDC", now=1005.0) is None
    # WhalePositioningCollector._resolve_cached_funding_rate also rejects future-dated derivatives telemetry
    assert whale_col._resolve_cached_funding_rate("BTCUSDC", now=1005.0) is None

def test_bybitliquidationcollector_get_summary_and_liquidationcascadeguard_check_honor_now_iter17_s2():
    """[ITER17 #2] BybitLiquidationCollector.get_summary and LiquidationCascadeGuard.check honor `now`."""
    # 2. BybitLiquidationCollector.get_summary and LiquidationCascadeGuard.check honor `now`
    liq_col = BybitLiquidationCollector(symbols=["BTCUSDT"])
    liq_col.record_event("BTCUSDT", "LONG", 600_000.0, ts=5000.0)
    liq_col.record_event("BTCUSDT", "LONG", 900_000.0, ts=5100.0)  # future event relative to now=5010.0
    c_guard = LiquidationCascadeGuard(collector=liq_col, max_active_cascade_usd=500_000.0, active_window_sec=60.0)
    c_dec = c_guard.check("BTCUSDC", "BUY", now=5010.0)
    assert c_dec.allowed is False
    assert c_dec.metadata.get("long_liq_usd") == pytest.approx(600_000.0)

def test_macroanalyzer_string_shadow_notify_previous_fallback_recovery_clock_skew_iter18_s1(tmp_path, monkeypatch):
    """[ITER18 #1] MacroAnalyzer string shadow_notify + .previous / fallback recovery + clock-skew run_cycle & GlobalTelemetryCollector and IntelligenceTelemetryDaemon backward clock-skew resilience & External collectors + check_external_order_guards `now` propagation & NaN filtering."""
    # 1. MacroAnalyzer string shadow_notify + .previous / fallback recovery + clock-skew run_cycle
    orig_margins = order_guard._load_margins()
    custom_margins = dict(orig_margins)
    custom_margins["shadow_notify"] = "false"
    monkeypatch.setattr(order_guard, "_load_margins", lambda: custom_margins)

    analyzer = MacroAnalyzer(
        cache_dir=str(tmp_path),
        macro_llm_interval_sec=600.0,
        advisor_interval_sec=600.0,
        symbols=["BTCUSDC"],
    )
    assert analyzer.shadow_notify is False

    # Corrupt primary fear_greed_collect.json, valid .previous
    (tmp_path / "fear_greed_collect.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "fear_greed_collect.json.previous").write_text(
        json.dumps(
            {
                "value": 18,
                "classification": "Extreme Fear",
                "historical_values": [18, 22],
                "trend_7d_change": -4,
                "is_extreme_fear": True,
                "is_extreme_greed": False,
                "ts": 5000.0,
            }
        ),
        encoding="utf-8",
    )
    # Corrupt primary market_breadth_collect.json, valid .previous
    (tmp_path / "market_breadth_collect.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "market_breadth_collect.json.previous").write_text(
        json.dumps(
            {
                "advance_ratio": 0.25,
                "advancing_count": 25,
                "declining_count": 75,
                "total_count": 100,
                "median_change_pct": -2.5,
                "mean_change_pct": -2.8,
                "dispersion_std": 1.2,
                "regime": "BROAD_SELLOFF",
                "top_gainers": [["BTCUSDC", 1.2]],
                "top_losers": [["TAOUSDC", -5.4]],
                "ts": 5000.0,
            }
        ),
        encoding="utf-8",
    )
    # Corrupt primary whale_snapshot_BTCUSDC.json, valid .previous
    (tmp_path / "whale_snapshot_BTCUSDC.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "whale_snapshot_BTCUSDC.json.previous").write_text(
        json.dumps({"symbol": "BTCUSDC", "taker_buy_sell_ratio": 0.72, "ts": 5000.0}),
        encoding="utf-8",
    )

    captured_review = {}

    class _DummyAssessment:
        market_bias = "DEFENSIVE"
        risk_level = "HIGH"
        recommended_action = "HOLD"
        summary = "test"
        ts = 5005.0

    def _fake_review(fear_greed=None, breadth=None, whale_telemetry=None, force_refresh=False, now_ts=None):
        captured_review["fg"] = fear_greed
        captured_review["mb"] = breadth
        captured_review["whale"] = whale_telemetry
        captured_review["now_ts"] = now_ts
        return _DummyAssessment()

    monkeypatch.setattr(analyzer.gemini_advisor, "review", _fake_review)
    monkeypatch.setattr(analyzer, "assess_geopolitical_threat", lambda force=False, now=None: True)
    monkeypatch.setattr("intelligence.macro_analyzer.prune_automated_cli_threads", lambda *a, **k: 0)

    # Simulate future _last_advisor_ts / _last_macro_llm_ts (backward clock skew)
    analyzer._last_macro_llm_ts = 99999.0
    analyzer._last_advisor_ts = 99999.0
    cycle_res = analyzer.run_cycle(force=False, now=5005.0)
    assert cycle_res["geopolitical_updated"] is True
    assert cycle_res["advisor_updated"] is True
    assert captured_review["fg"] is not None and captured_review["fg"].value == 18
    assert captured_review["mb"] is not None and captured_review["mb"].regime == "BROAD_SELLOFF"
    assert captured_review["whale"] is not None and captured_review["whale"]["symbol"] == "BTCUSDC"
    assert captured_review["now_ts"] == pytest.approx(5005.0)


    # 2. GlobalTelemetryCollector and IntelligenceTelemetryDaemon backward clock-skew resilience
    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_news=True, skip_external=True)
    gtc._last_fear_greed_ts = 99999.0
    gtc._last_breadth_ts = 99999.0
    monkeypatch.setattr(gtc, "update_fear_greed", lambda force=False: True)
    monkeypatch.setattr(gtc, "update_market_breadth", lambda force=False: True)
    gtc_res = gtc.run_cycle(force=False, now=5000.0)
    assert gtc_res["fear_greed_updated"] is True
    assert gtc_res["breadth_updated"] is True

    itd = IntelligenceTelemetryDaemon(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_macro=False, skip_external=True)
    itd._last_macro_ts = 99999.0
    monkeypatch.setattr(itd, "update_macro", lambda force=False: True)
    itd_res = itd.run_cycle(force=False, now=5000.0)
    assert itd_res["macro_updated"] is True


    # 3. External collectors + check_external_order_guards `now` propagation & NaN filtering
    price_cache = tmp_path / "cache_24_price_BTCUSDC.json"
    price_cache.write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        [1400.0, 100.0],
                        [2500.0, float("nan")],
                        [5000.0, 105.0],
                    ]
                }
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    w_col = WhalePositioningCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    assert w_col._resolve_price_1h_change_pct("BTCUSDC") == pytest.approx(5.0)

    w_snap = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.2,
        top_traders_long_pct=0.55,
        taker_buy_sell_ratio=0.60,  # severe taker selling (< 0.75) -> defer BUY
        taker_buy_vol_usd=100_000.0,
        taker_sell_vol_usd=200_000.0,
        open_interest_usd=10_000_000.0,
        open_interest_1h_change_pct=0.1,
        divergence_regime="neutral",
        ts=1000.0,
    )
    w_col.record_snapshot(w_snap, save_to_disk=True)

    ob_col = OrderbookDepthCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    ob_col.record_snapshot(
        OrderbookSnapshot(
            symbol="BTCUSDC",
            mid_price=100.0,
            bid_depth_usd=500_000.0,
            ask_depth_usd=500_000.0,
            imbalance_ratio=0.5,
            largest_bid_wall_usd=50_000.0,
            largest_bid_wall_price=99.5,
            largest_ask_wall_usd=50_000.0,
            largest_ask_wall_price=100.5,
            ts=1000.0,
        ),
        save_to_disk=True,
    )
    d_col = DerivativesTelemetryCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    d_col.record_snapshot(
        DerivativesTelemetry(
            symbol="BTCUSDC",
            funding_rate=0.0001,
            predicted_funding_rate=0.0001,
            open_interest=100.0,
            open_interest_usd=10_000.0,
            ts=1000.0,
        ),
        save_to_disk=True,
    )

    monkeypatch.setattr(external_order_guard, "get_whale_collector", lambda: w_col)
    monkeypatch.setattr(external_order_guard, "get_orderbook_collector", lambda: ob_col)
    monkeypatch.setattr(external_order_guard, "get_derivatives_collector", lambda: d_col)

    ext_margins = dict(custom_margins)
    ext_margins["external_guard_mode"] = "enforce"
    ext_margins["whale_guard_mode"] = "enforce"
    ext_margins["orderbook_wall_guard_mode"] = "enforce"
    ext_margins["funding_guard_mode"] = "enforce"

    # At simulated now=1050.0 (within 120s TTL of ts=1000.0), w_snap is fresh and blocks BUY
    allowed_fresh, reason_fresh, _ = external_order_guard.check_external_order_guards(
        "BTCUSDC", "BUY", 100.0, margins=ext_margins, now=1050.0
    )
    assert allowed_fresh is False
    assert "severe_taker_selling_dominated" in reason_fresh

    # At simulated now=2000.0 (expired > 300s disk TTL), w_snap is expired -> fail-open allow
    allowed_stale, reason_stale, _ = external_order_guard.check_external_order_guards(
        "BTCUSDC", "BUY", 100.0, margins=ext_margins, now=2000.0
    )
    assert allowed_stale is True
    assert reason_stale == "ok"

def test_liquidationcapitulationtrigger_now_propagation_iter19_s4():
    """[ITER19 #4] LiquidationCapitulationTrigger `now` propagation."""
    # 4. LiquidationCapitulationTrigger `now` propagation
    liq_col = BybitLiquidationCollector(retention_sec=3600.0)
    liq_col.record_event("BTCUSDT", "LONG", 250_000.0, ts=3000.0)
    liq_trig = LiquidationCapitulationTrigger(collector=liq_col, min_notional_usd=100_000.0, window_sec=300.0)
    ev_liq = liq_trig.evaluate("BTCUSDC", 100.0, now=3050.0)
    assert ev_liq is not None and ev_liq.reason == "liquidation_capitulation_burst"

def test_intelligence_cli_previous_recovery_iter19_s5(tmp_path):
    """[ITER19 #5] intelligence.cli .previous recovery."""
    from intelligence.cli import get_daemon_status
    WhalePositioningCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0).record_snapshot(
        WhalePositioningSnapshot(
            symbol='BTCUSDC', top_traders_long_ratio=2.0, top_traders_long_pct=0.68,
            taker_buy_sell_ratio=1.10, taker_buy_vol_usd=500_000.0, taker_sell_vol_usd=450_000.0,
            open_interest_usd=20_000_000.0, open_interest_1h_change_pct=1.5,
            divergence_regime='accumulation', ts=2000.0,
        ),
        save_to_disk=True,
    )
    # 5. intelligence.cli .previous recovery
    (tmp_path / "globaltelemetry_collector.heartbeat").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "globaltelemetry_collector.heartbeat.previous").write_text(
        json.dumps({"pid": os.getpid(), "ts": time.time(), "symbols": ["BTCUSDC"]}),
        encoding="utf-8",
    )
    (tmp_path / "macro_analyzer.heartbeat").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "macro_analyzer.heartbeat.previous").write_text(
        json.dumps({"pid": os.getpid(), "ts": time.time(), "intervals": {"geopolitical_llm": 600}}),
        encoding="utf-8",
    )
    assert get_daemon_status(str(tmp_path))["running"] is True
    rep = inspect_symbol_intelligence("BTCUSDC", cache_dir=str(tmp_path))
    assert rep["pillar2_external"]["whale"] is not None

def test_snapshot_from_dict_tolerates_none_bool_nan_fields_iter23_s3():
    """[ITER23 #3] Snapshot.from_dict tolerates None / bool / NaN fields."""
    # 3. Snapshot.from_dict tolerates None / bool / NaN fields
    ob_snap = OrderbookSnapshot.from_dict({"symbol": "BTCUSDC", "imbalance_ratio": None, "largest_ask_wall_usd": "NaN"})
    assert ob_snap.imbalance_ratio == pytest.approx(0.5)
    assert ob_snap.largest_ask_wall_usd == pytest.approx(0.0)

    w_snap = WhalePositioningSnapshot.from_dict({"symbol": "BTCUSDC", "taker_buy_sell_ratio": None, "funding_rate": "NaN"})
    assert w_snap.taker_buy_sell_ratio == pytest.approx(1.0)
    assert w_snap.funding_rate is None

    d_snap = DerivativesTelemetry.from_dict({"symbol": "BTCUSDC", "funding_rate": None, "open_interest_usd": True})
    assert d_snap.funding_rate == pytest.approx(0.0)
    assert d_snap.open_interest_usd == pytest.approx(0.0)

def test_survival_get_trend_survival_metrics_estimate_t_safety_compositemarketintelligence_evaluate_guards_honors_bypass_guards_iter24_s1(tmp_path, monkeypatch):
    """[ITER24 #1] survival.get_trend_survival_metrics & estimate_T safety & CompositeMarketIntelligence.evaluate_guards honors bypass_guards & GeopoliticalShockGuard tolerates None fields & External guards tolerate None/NaN attributes on synthetic snapshots & MarketRegimeContext.is_valid_for matches TAO/USD and TAOUSD on same venue."""
    import intelligence.internal.state.survival as survival
    from market_regime import MarketRegimeContext, MarketRegimeDecision
    # 1. survival.get_trend_survival_metrics & estimate_T safety
    t_cache = tmp_path / "cache_T_trend.json"
    now_curr = time.time()
    t_cache.write_text(
        json.dumps({
            "BTCUSDC": {
                "T": -10.0,
                "median_d": float("nan"),
                "p90_d": None,
                "ts": now_curr + 86400.0,  # clock-skewed future ts
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(survival, "T_CACHE_FILE", str(t_cache))
    metrics = survival.get_trend_survival_metrics("BTCUSDC", trend_duration_seconds=None)
    assert metrics["duration_days"] == 0.0
    assert metrics["median_days"] == 3.0
    assert metrics["p90_days"] == 7.0
    assert metrics["T"] == 8.0
    assert metrics["gaussian_horizon_days"] is None

    # Future ts in cache does not short-circuit estimate_T before fetch_klines
    fetch_called = []
    monkeypatch.setattr(
        survival,
        "fetch_klines",
        lambda sym, days: (fetch_called.append(sym) or ([], [])),
    )
    est = survival.estimate_T("tao/usd")
    assert len(fetch_called) >= 1
    assert est is not None


    # 2. CompositeMarketIntelligence.evaluate_guards honors bypass_guards & GeopoliticalShockGuard tolerates None fields
    geo_guard = GeopoliticalShockGuard()
    dec_null_geo = geo_guard.check(
        "BTCUSDC",
        "BUY",
        SimpleNamespace(threat_level=None, recommended_brake=None, summary="none", risk_score=0.0),
    )
    assert dec_null_geo.allowed is True

    cmi = CompositeMarketIntelligence()
    shock_assessment = SimpleNamespace(
        threat_level="CRITICAL_SHOCK",
        recommended_brake="HARD_VETO_NEW_BUYS",
        summary="crisis",
        risk_score=0.95,
    )
    dec_blocked = cmi.evaluate_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        geopolitical_assessment=shock_assessment,
    )
    assert dec_blocked.allowed is False
    dec_bypassed = cmi.evaluate_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        geopolitical_assessment=shock_assessment,
        bypass_guards={"GLLM3"},
    )
    assert dec_bypassed.allowed is True


    # 3. External guards tolerate None/NaN attributes on synthetic snapshots
    ob_guard = OrderbookWallGuard()
    ob_dec = ob_guard.check(
        "BTCUSDC",
        "BUY",
        snapshot=SimpleNamespace(
            imbalance_ratio=None,
            largest_ask_wall_usd=float("nan"),
            largest_ask_wall_price=None,
            largest_bid_wall_usd=None,
            largest_bid_wall_price=None,
        ),
    )
    assert ob_dec.allowed is True

    whale_guard = WhaleDivergenceGuard()
    whale_dec = whale_guard.check(
        "BTCUSDC",
        "SELL",
        snapshot=SimpleNamespace(
            divergence_regime=None,
            open_interest_1h_change_pct=float("nan"),
            taker_buy_sell_ratio=None,
            top_traders_long_pct=float("nan"),
        ),
    )
    assert whale_dec.allowed is True

    fund_guard = FundingCrowdingGuard()
    fund_dec = fund_guard.check(
        "BTCUSDC",
        "BUY",
        telemetry=SimpleNamespace(funding_rate=float("nan")),
    )
    assert fund_dec.allowed is True


    # 4. MarketRegimeContext.is_valid_for matches TAO/USD and TAOUSD on same venue
    dec = MarketRegimeDecision(
        regime="bull",
        gradient=0.01,
        epsilon=0.002,
        strength=5.0,
        fresh=True,
        reason="directional_signal",
    )
    ctx = MarketRegimeContext.from_decision(
        dec,
        evaluated_at=now_curr,
        symbol="TAO/USD",
        provider="kraken",
        trend_duration_seconds=300.0,
    )
    assert ctx.is_valid_for(symbol="TAOUSD", provider="kraken", now=now_curr) is True
    assert ctx.is_valid_for(symbol="TAO/USD", provider="kraken", now=now_curr) is True
    assert ctx.is_valid_for(symbol="ETHUSD", provider="kraken", now=now_curr) is False

def test_whalepositioningcollector__resolve_cached_funding_rate_classify_regime_iter29_s3(tmp_path):
    """[ITER29 #3] WhalePositioningCollector._resolve_cached_funding_rate & classify_regime."""
    # 3. WhalePositioningCollector._resolve_cached_funding_rate & classify_regime
    w_col = WhalePositioningCollector(cache_ttl_sec=60.0, disk_ttl_sec=120.0, cache_dir=str(tmp_path))
    f_path = tmp_path / "derivatives_telemetry_BTCUSDT.json"

    # Missing / zero ts must be rejected
    f_path.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": -0.001, "ts": 0.0}), encoding="utf-8")
    assert w_col._resolve_cached_funding_rate("BTCUSDC", now=1000.0) is None

    # Bool funding_rate must be rejected
    f_path.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": True, "ts": 990.0}), encoding="utf-8")
    assert w_col._resolve_cached_funding_rate("BTCUSDC", now=1000.0) is None

    # Valid fresh funding_rate is accepted
    f_path.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": -0.0005, "ts": 990.0}), encoding="utf-8")
    assert w_col._resolve_cached_funding_rate("BTCUSDC", now=1000.0) == pytest.approx(-0.0005)

    # classify_regime with NaN price_1h_change_pct and NaN funding_rate
    assert WhalePositioningCollector.classify_regime(
        price_1h_change_pct=float("nan"),
        oi_change_pct=1.2,
        taker_ratio=1.25,
        funding_rate=float("nan"),
    ) == "accumulation"

def test_collectors_fetch_with_nan_bool_now_iter29_s4(tmp_path):
    """[ITER29 #4] Collectors fetch() with NaN/bool `now`."""
    # 4. Collectors fetch() with NaN/bool `now`
    ob_col = OrderbookDepthCollector(cache_ttl_sec=60.0, cache_dir=str(tmp_path))
    now_s = time.time()
    ob_snap = OrderbookSnapshot(
        symbol="BTCUSDC", mid_price=85000.0, bid_depth_usd=1000.0, ask_depth_usd=1000.0,
        imbalance_ratio=0.5, largest_bid_wall_usd=100.0, largest_bid_wall_price=84900.0,
        largest_ask_wall_usd=100.0, largest_ask_wall_price=85100.0, ts=now_s,
    )
    ob_col.record_snapshot(ob_snap)
    assert ob_col.fetch("BTCUSDC", allow_network=False, now=float("nan")) is ob_snap
    assert ob_col.fetch("BTCUSDC", allow_network=False, now=True) is ob_snap

    dt_col = DerivativesTelemetryCollector(cache_ttl_sec=60.0, cache_dir=str(tmp_path))
    dt_snap = DerivativesTelemetry(
        symbol="BTCUSDC", funding_rate=0.0001, predicted_funding_rate=0.0001,
        open_interest=100.0, open_interest_usd=8_500_000.0, ts=now_s,
    )
    dt_col.record_snapshot(dt_snap)
    assert dt_col.fetch("BTCUSDC", allow_network=False, now=float("nan")) is dt_snap

def test_bybitliquidationcollector_hardening_marketbreadthcollector_parse_tickers_with_nan_min_volume_usd_lowercase_iter30_s1(tmp_path):
    """[ITER30 #1] BybitLiquidationCollector hardening & MarketBreadthCollector.parse_tickers with NaN min_volume_usd & lowercase symbols & SentimentAdvisor & GeopoliticalThreatAnalyzer non-dict LLM response & enum sanitization & NewsFeedCollector.get_pool_snapshot filters future-skewed / NaN timestamps & Triggers & CompositeMarketIntelligence hardening."""
    import math
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsFeedSnapshot, NewsHeadline
    # 1. BybitLiquidationCollector hardening
    liq_col = BybitLiquidationCollector(symbols=["BTCUSDT"])
    now_s = time.time()
    liq_col._ingest("null")
    liq_col._ingest("[1, 2, 3]")
    liq_col._ingest(json.dumps({
        "topic": "allLiquidation.BTCUSDT",
        "data": [
            "not-a-dict",
            {"s": "BTCUSDT", "S": "Buy", "v": True, "p": 85000.0},
            {"s": "BTCUSDT", "S": "UNKNOWN", "v": 1.0, "p": 85000.0},
            {"s": "btc/usdt", "S": "BUY", "v": 2.0, "p": 50000.0},
        ],
    }))
    liq_col.record_event("BTC/USDT", "BUY", True, ts=now_s)
    liq_col.record_event("BTC/USDT", "BUY", 50000.0, ts=float("nan"))
    liq_col.record_event("BTC/USDT", "INVALID_SIDE", 50000.0, ts=now_s)
    # Record an expired event 2000s ago
    liq_col.record_event("BTC/USDT", "SELL", 250000.0, ts=now_s - 2000.0)

    # Calling get_summary with NaN `now` or NaN `window_sec` must NOT include the 2000s-old event
    summary = liq_col.get_summary("btc/usdt", window_sec=float("nan"), now=float("nan"))
    assert summary.symbol == "BTCUSDT"
    assert summary.events_count == 1
    assert summary.long_liq_usd == 100000.0
    assert summary.short_liq_usd == 0.0


    # 2. MarketBreadthCollector.parse_tickers with NaN min_volume_usd & lowercase symbols
    tickers = [
        {"symbol": f"coin{i}usdt", "quoteVolume": "10000000", "priceChangePercent": "-5.5"}
        for i in range(6)
    ]
    b_snap = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=float("nan"), now_ts=float("nan"))
    assert b_snap is not None
    assert b_snap.total_count == 6
    assert b_snap.regime == "PANIC_WASHOUT"
    assert math.isfinite(b_snap.ts) and b_snap.ts > 0.0


    # 3. SentimentAdvisor & GeopoliticalThreatAnalyzer non-dict LLM response & enum sanitization
    fake_gemini_list = SimpleNamespace(query_json=lambda *a, **kw: ["unexpected", "list"])
    advisor = SentimentAdvisor(
        gemini_client=fake_gemini_list,
        cache_file=str(tmp_path / "sentiment_eval.json"),
    )
    assert advisor.review(force_refresh=True, now_ts=float("nan")) is None

    fake_gemini_bad_enums = SimpleNamespace(
        query_json=lambda *a, **kw: {
            "threat_level": "APOCALYPSE",
            "risk_score": 0.8,
            "summary": "Test",
            "recommended_brake": "FULL_HALT",
        }
    )
    geo = GeopoliticalThreatAnalyzer(
        gemini_client=fake_gemini_bad_enums,
        cache_ttl_sec=60.0,
        fallback_retry_ttl_sec=10.0,
        state_file=str(tmp_path / "geo_eval.json"),
    )
    news_snap = NewsFeedSnapshot(
        headlines=(
            NewsHeadline(
                title="Military strike reported near oil pipeline",
                source="Test",
                pub_date="",
                url="",
                matched_keywords=("military strike",),
                is_high_severity=True,
            ),
        ),
        total_fetched=1,
        high_severity_count=1,
        has_critical_shock_keywords=False,
        ts=now_s,
    )
    geo_res = geo.assess(news_snap, force_refresh=True, now_ts=float("nan"))
    assert geo_res.threat_level == "NORMAL"
    assert geo_res.recommended_brake == "NONE"
    assert math.isfinite(geo_res.ts) and geo_res.ts > 0.0


    # 4. NewsFeedCollector.get_pool_snapshot filters future-skewed / NaN timestamps
    nf_col = NewsFeedCollector(history_file=str(tmp_path / "news_collect.json"))
    nf_col._aggregated_pool["future_hl"] = (now_s + 3600.0, news_snap.headlines[0])
    nf_col._aggregated_pool["nan_hl"] = (float("nan"), news_snap.headlines[0])
    nf_col._aggregated_pool["valid_hl"] = (now_s - 10.0, news_snap.headlines[0])
    pool_snap = nf_col.get_pool_snapshot(now_ts=float("nan"))
    assert pool_snap.total_fetched == 1
    assert math.isfinite(pool_snap.ts)


    # 5. Triggers & CompositeMarketIntelligence hardening
    liq_trig = LiquidationCapitulationTrigger(min_notional_usd=100_000.0)
    bad_summary = LiquidationSummary(
        symbol="BTCUSDT", window_sec=300.0, long_liq_usd=float("nan"), short_liq_usd=0.0,
        net_usd=0.0, capitulation_ratio=0.9, squeeze_ratio=0.1, events_count=1, last_event_ts=now_s,
    )
    assert liq_trig.evaluate("BTCUSDT", 85000.0, summary=bad_summary, now=float("nan")) is None
    assert liq_trig.evaluate("BTCUSDT", True, summary=summary, now=now_s) is None

    sc_trig = SentimentContrarianTrigger()
    fg_snap = FearGreedSnapshot(
        value=15, classification="Extreme Fear", timestamp=now_s,
        historical_values=(15, 16), trend_7d_change=-5, is_extreme_fear=True, is_extreme_greed=False,
    )
    assert sc_trig.evaluate("BTCUSDT", False, fg_snap) == (None, None)

    mb_trig = MarketBreadthTrigger()
    sig, ev = mb_trig.evaluate("BTCUSDT", 85000.0, b_snap, asset_24h_change_pct=False, ts=float("nan"))
    assert sig == "BREADTH_WASHOUT_BUY"
    assert ev is not None and ev.metadata["relative_strength"] is False
    assert math.isfinite(ev.ts) and ev.ts > 0.0

    comp = CompositeMarketIntelligence()
    assert comp._get_kalman("btc/usdc") is comp._get_kalman("BTCUSDC")
    assert comp._get_gradient("btc/usdc") is comp._get_gradient("BTCUSDC")
    # Passing bool notional_eur or bool qty must not crash or trigger GLLM1 falsely
    g_dec = comp.evaluate_guards("BTC/USDC", "BUY", 85000.0, qty=True, notional_eur="bad")
    assert g_dec.allowed is True

def test_external_guards_with_none_side_iter31_s2():
    """[ITER31 #2] External guards with None side."""
    # 2. External guards with None side
    assert FundingCrowdingGuard().check("BTCUSDC", None).allowed is True
    assert OrderbookWallGuard().check("BTCUSDC", None).allowed is True
    assert WhaleDivergenceGuard().check("BTCUSDC", None).allowed is True

def test_external_order_guard_fails_fast_on_empty_margins_allows_partial_iter37_s1(monkeypatch):
    """[ITER37 #1] external_order_guard fails fast on empty margins={}, allows partial margins with guard modes,."""
    # 1. external_order_guard fails fast on empty margins={}, allows partial margins with guard modes,
    #    allows now=0.0, and sanitizes NaN/bool suggested_scale
    with pytest.raises(KeyError):
        external_order_guard.check_external_order_guards("BTCUSDC", "BUY", 60000.0, margins={})

    ok_part, r_part, scale_part = external_order_guard.check_external_order_guards(
        "BTCUSDC",
        "BUY",
        60000.0,
        margins={"whale_guard_mode": "off", "orderbook_wall_guard_mode": "off", "funding_guard_mode": "off"},
        now=0.0,
    )
    assert ok_part is True and r_part == "ok"
    assert scale_part == 1.0

    import intelligence.external.guards.whale_divergence_guard as wdg
    import intelligence.external.guards.orderbook_wall_guard as owg
    import intelligence.external.guards.funding_crowding_guard as fcg

    monkeypatch.setattr(external_order_guard, "get_whale_collector", lambda: SimpleNamespace(fetch=lambda *a, **kw: {}))
    monkeypatch.setattr(external_order_guard, "get_orderbook_collector", lambda: SimpleNamespace(fetch=lambda *a, **kw: {}))
    monkeypatch.setattr(external_order_guard, "get_derivatives_collector", lambda: SimpleNamespace(fetch=lambda *a, **kw: {}))
    monkeypatch.setattr(
        wdg, "WhaleDivergenceGuard",
        lambda **kw: SimpleNamespace(check=lambda *a, **k: SimpleNamespace(allowed=True, brake_action=BrakeAction.DOWNSCALE_QTY, suggested_scale=float("nan"), reason="w_nan")),
    )
    monkeypatch.setattr(
        owg, "OrderbookWallGuard",
        lambda **kw: SimpleNamespace(check=lambda *a, **k: SimpleNamespace(allowed=True, brake_action=BrakeAction.DOWNSCALE_QTY, suggested_scale=True, reason="ob_bool")),
    )
    monkeypatch.setattr(
        fcg, "FundingCrowdingGuard",
        lambda **kw: SimpleNamespace(check=lambda *a, **k: SimpleNamespace(allowed=True, brake_action=BrakeAction.DOWNSCALE_QTY, suggested_scale=0.75, reason="f_down")),
    )
    ok_scale, r_scale, scale_val = external_order_guard.check_external_order_guards(
        "BTCUSDC",
        "BUY",
        60000.0,
        margins={
            "external_guards_mode": "enforce",
            "whale_guard_mode": "enforce",
            "orderbook_wall_guard_mode": "enforce",
            "funding_guard_mode": "enforce",
            "severe_cascade_oi_pct": 5.0,
            "min_taker_ratio": 0.5,
            "taker_downscale_threshold": 0.8,
            "taker_downscaled_scale": 0.5,
            "min_buy_imbalance": 0.2,
            "max_sell_imbalance": 0.8,
            "whale_wall_usd_limit": 500000.0,
            "funding_max_long_rate": 0.001,
            "funding_min_short_rate": -0.001,
            "funding_crowding_policy": "downscale",
            "funding_crowded_scale": 0.75,
        },
        now=0.0,
    )
    assert ok_scale is True and r_scale == "f_down"
    assert scale_val == pytest.approx(0.75)

def test_intelligence_collectors_macroanalyzer_sanitize_bool_nan_inf_ttl_iter53_s2(tmp_path):
    """[ITER53 #2] Intelligence collectors & MacroAnalyzer sanitize bool/NaN/Inf TTL & retention parameters."""
    # 2. Intelligence collectors & MacroAnalyzer sanitize bool/NaN/Inf TTL & retention parameters
    blc = BybitLiquidationCollector(retention_sec=True)
    assert blc._retention == 3600.0
    blc_nan = BybitLiquidationCollector(retention_sec=float("nan"))
    assert blc_nan._retention == 3600.0

    dtc = DerivativesTelemetryCollector(cache_ttl_sec=True, disk_ttl_sec=float("nan"), cache_dir=str(tmp_path))
    assert dtc.cache_ttl_sec == 60.0
    assert dtc.disk_ttl_sec == 120.0

    obc = OrderbookDepthCollector(
        cache_ttl_sec=True, depth_band_pct=float("nan"), whale_wall_usd=-1.0, disk_ttl_sec=float("inf"), cache_dir=str(tmp_path)
    )
    assert obc.cache_ttl_sec == 5.0
    assert obc.depth_band_pct == 1.5
    assert obc.whale_wall_usd == 500_000.0
    assert obc.disk_ttl_sec == 60.0

    wpc = WhalePositioningCollector(cache_ttl_sec=True, disk_ttl_sec=float("nan"), cache_dir=str(tmp_path))
    assert wpc.cache_ttl_sec == 120.0
    assert wpc.disk_ttl_sec == 300.0

    ma = MacroAnalyzer(
        cache_dir=str(tmp_path),
        macro_llm_interval_sec=True,
        advisor_interval_sec=float("nan"),
        thread_prune_interval_sec=-5.0,
    )
    assert ma.macro_llm_interval_sec > 1.0
    assert ma.advisor_interval_sec > 1.0
    assert ma.thread_prune_interval_sec == 10800.0

def test_intelligence_cli_daemon_globaltelemetrycollector_iter54_s3(tmp_path):
    """[ITER54 #3] Intelligence CLI & Daemon / GlobalTelemetryCollector."""
    from intelligence.cli import print_intelligence_report
    # 3. Intelligence CLI & Daemon / GlobalTelemetryCollector
    ob_file = tmp_path / "orderbook_depth_BTCUSDC.json"
    ob_file.write_text(json.dumps({"mid_price": None, "imbalance_ratio": float("nan"), "ts": 0.0}).replace("NaN", "null"))
    rep = inspect_symbol_intelligence("BTCUSDC", cache_dir=str(tmp_path))
    assert rep["pillar2_external"]["orderbook"]["age_seconds"] is None
    assert rep["pillar2_external"]["orderbook"]["age_formatted"] == "unknown"
    print_intelligence_report(rep)

    itd = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC"],
        cache_dir=str(tmp_path),
        orderbook_interval_sec=True,
        derivatives_interval_sec=float("nan"),
        whale_interval_sec=-5.0,
        macro_interval_sec=float("inf"),
    )
    assert itd.orderbook_interval_sec == 15.0
    assert itd.derivatives_interval_sec == 30.0
    assert itd.whale_interval_sec == 60.0
    assert itd.macro_interval_sec > 0.0

    gtc = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=str(tmp_path),
        orderbook_interval_sec=True,
        derivatives_interval_sec=float("nan"),
        whale_interval_sec=-5.0,
        news_interval_sec=float("inf"),
        fear_greed_interval_sec=True,
        breadth_interval_sec=float("nan"),
    )
    assert gtc.orderbook_interval_sec == 15.0
    assert gtc.derivatives_interval_sec == 30.0
    assert gtc.whale_interval_sec == 60.0
    assert gtc.fear_greed_interval_sec == 1800.0
    assert gtc.breadth_interval_sec == 300.0

def test_liquidation_signal_safe_pos_float__ingest_handle_overflowerror_and_start_has_iter63_s1():
    """[ITER63 #1] liquidation_signal safe_pos_float & _ingest handle OverflowError and start() has _start_lock."""
    from botcore import safe_pos_float
    # 1. liquidation_signal safe_pos_float & _ingest handle OverflowError and start() has _start_lock
    assert safe_pos_float(10**400, 42.0) == 42.0
    sig = BybitLiquidationCollector(symbols=["BTCUSDT"])
    assert hasattr(sig, "_start_lock")
    sig._ingest(json.dumps({
        "topic": "allLiquidation.BTCUSDT",
        "data": [
            {"s": "BTCUSDT", "S": "Buy", "v": 10**400, "p": "70000"},
            {"s": "BTCUSDT", "S": "Buy", "v": "0.1", "p": "70000"},
        ],
    }))
    out = sig.get_signal("BTCUSDT", window_sec=60.0)
    assert out["long_count"] == 1
    assert out["long_liq_usd"] == pytest.approx(7000.0)

def test_macroanalyzer_globaltelemetrycollector_and_intelligencetelemetrydaemon_sanitize_symbols_lists_iter83_s2(tmp_path):
    """[ITER83 #2] MacroAnalyzer, GlobalTelemetryCollector, and IntelligenceTelemetryDaemon sanitize symbols lists."""
    # 2. MacroAnalyzer, GlobalTelemetryCollector, and IntelligenceTelemetryDaemon sanitize symbols lists
    ma = MacroAnalyzer(cache_dir=str(tmp_path), symbols=["  btcusdc  ", "", True, None])
    assert ma.symbols == ["BTCUSDC"]
    gtc = GlobalTelemetryCollector(cache_dir=str(tmp_path), symbols=["   ", False])
    assert len(gtc.symbols) >= 1 and all(isinstance(s, str) and s.strip() == s and s.isupper() for s in gtc.symbols)
    itd = IntelligenceTelemetryDaemon(cache_dir=str(tmp_path), symbols=[" taousdc ", ""])
    assert itd.symbols == ["TAOUSDC"]

def test_parabolic_funding_orderbook_cascade_whale_highstake_extremegreed_panicwashout_iter101_s2():
    """[ITER101 #2] Parabolic, Funding, Orderbook, Cascade, Whale, HighStake, ExtremeGreed, PanicWashout, Geopolitical side safety."""
    from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
    from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
    # 2. Parabolic, Funding, Orderbook, Cascade, Whale, HighStake, ExtremeGreed, PanicWashout, Geopolitical side safety
    pg = ParabolicSurgeGuard()
    assert pg.check("BTCUSDC", True, 100.0).reason == "unsupported_side_skipped"
    assert pg.check("BTCUSDC", " b ", 100.0).reason == "normal_market_structure"

    fg = FundingCrowdingGuard(crowding_policy=True)
    assert fg.crowding_policy == "downscale"
    assert fg.check("BTCUSDC", True).reason == "unsupported_side_allowed"
    assert fg.check("BTCUSDC", " b ").reason == "no_telemetry_data"

    obg = OrderbookWallGuard()
    assert obg.check("BTCUSDC", True).reason == "unsupported_side_allowed"
    assert obg.check("BTCUSDC", " s ").reason == "no_orderbook_snapshot"

    cg = LiquidationCascadeGuard()
    assert cg.check("BTCUSDC", True).reason == "unsupported_side_allowed"
    assert cg.check("BTCUSDC", " b ").reason == "no_liquidation_data"

    wdg = WhaleDivergenceGuard()
    assert wdg.check("BTCUSDC", True).reason == "unsupported_side_allowed"
    assert wdg.check("BTCUSDC", " s ").reason == "no_whale_snapshot_data"

    hsg = HighStakeGuard(min_notional_usd=700.0)
    assert hsg.check("BTCUSDC", True, 100.0, 10.0).reason == "unsupported_side_exempt"
    assert "below_high_stake_threshold" in hsg.check("BTCUSDC", " b ", 100.0, 1.0).reason

    egg = ExtremeGreedGuard()
    assert egg.check("BTCUSDC", True, snapshot=type("S", (), {"value": 95, "classification": "Extreme Greed"})()).allowed is True
    assert egg.check("BTCUSDC", " b ", snapshot=type("S", (), {"value": 95, "classification": "Extreme Greed"})()).allowed is False

    pwg = PanicWashoutGuard()
    assert pwg.check("BTCUSDC", " b ").reason == "macro_conditions_stable"

    gsg = GeopoliticalShockGuard()
    assert gsg.check("BTCUSDC", " b ", assessment=type("A", (), {"threat_level": "CRITICAL_SHOCK", "recommended_brake": "HARD_VETO_NEW_BUYS", "summary": "test", "risk_score": 0.9})()).allowed is False

def test_orderbook_depth_helpers_collector_iter103_s1():
    """[ITER103 #1] orderbook_depth helpers & collector."""
    from cachepaths import resolve_binance_lookup_symbol
    # 1. orderbook_depth helpers & collector
    assert normalize_cache_base_asset(True) == ""
    assert resolve_binance_lookup_symbol(True) == ""
    assert cache_symbol_variants(True) == []
    obs = OrderbookSnapshot.from_dict({"symbol": True, "mid_price": -10, "imbalance_ratio": 2.5})
    assert obs.symbol == "" and obs.mid_price == 0.0 and obs.imbalance_ratio == 1.0
    obc = OrderbookDepthCollector(cache_dir=None)
    assert obc.fetch(True, allow_network=False) is None

def test_derivatives_telemetry_collector_from_dict_iter103_s2():
    """[ITER103 #2] derivatives_telemetry collector & from_dict."""
    # 2. derivatives_telemetry collector & from_dict
    dt = DerivativesTelemetry.from_dict({"symbol": True, "open_interest": -5, "open_interest_usd": -100, "ts": -1})
    assert dt.symbol == "" and dt.open_interest == 0.0 and dt.open_interest_usd == 0.0 and dt.ts == 0.0
    dtc = DerivativesTelemetryCollector(cache_dir=None)
    assert dtc.fetch(True, allow_network=False) is None

def test_whale_positioning_collector_from_dict_iter103_s3():
    """[ITER103 #3] whale_positioning collector & from_dict."""
    # 3. whale_positioning collector & from_dict
    wps = WhalePositioningSnapshot.from_dict({"symbol": True, "divergence_regime": True, "top_traders_long_pct": 1.8})
    assert wps.symbol == "" and wps.divergence_regime == "neutral" and wps.top_traders_long_pct == 1.0
    wpc = WhalePositioningCollector(cache_dir="cachedb")
    assert wpc.fetch(True, allow_network=False) is None
    assert wpc._resolve_price_1h_change_pct(True) == 0.0
    assert wpc._resolve_cached_funding_rate(True) is None
    assert wpc._resolve_cached_funding_rate("BTCUSDC", now="bad_now") is None or isinstance(wpc._resolve_cached_funding_rate("BTCUSDC", now="bad_now"), float)

def test_bybit_liquidations_record_event_get_summary_iter103_s4():
    """[ITER103 #4] bybit_liquidations record_event & get_summary."""
    # 4. bybit_liquidations record_event & get_summary
    blc = BybitLiquidationCollector()
    blc.record_event(True, "LONG", 50000.0)
    blc.record_event("BTCUSDT", True, 50000.0)
    blc.record_event("BTCUSDT", " b ", 25000.0, ts=1000.0)
    blc.record_event("BTCUSDT", " s ", 15000.0, ts=1000.0)
    s_ok = blc.get_summary("BTC/USDT", window_sec=300.0, now=1050.0)
    assert s_ok.long_liq_usd == pytest.approx(25000.0)
    assert s_ok.short_liq_usd == pytest.approx(15000.0)
    s_bool = blc.get_summary(True, window_sec=300.0, now=1050.0)
    assert s_bool.symbol == "" and s_bool.total_usd == 0.0

def test_bybitliquidationcollector_stop_self_join_and_invalid_timeout_safety_iter98_s1():
    """[ITER98 #1] BybitLiquidationCollector.stop: self-join and invalid timeout safety."""
    import threading
    # 1. BybitLiquidationCollector.stop: self-join and invalid timeout safety
    bybit = BybitLiquidationCollector(["BTCUSDT"])
    bybit._thread = threading.current_thread()
    bybit.stop(timeout=float("nan"))
    assert bybit._stop.is_set()

def test_orderbookdepthcollector_fetch_unsorted_bids_asks_and_crossed_book_iter98_s2(tmp_path, monkeypatch):
    """[ITER98 #2] OrderbookDepthCollector.fetch: unsorted bids/asks and crossed book rejection."""
    # 2. OrderbookDepthCollector.fetch: unsorted bids/asks and crossed book rejection
    ob = OrderbookDepthCollector(cache_dir=str(tmp_path), depth_band_pct=2.0, cache_ttl_sec=0.0)
    _DummyResp = MockHttpResponse

    # Crossed book (best bid 105 > best ask 100) -> rejected (returns None)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=4: _DummyResp({"bids": [["105.0", "1.0"]], "asks": [["100.0", "1.0"]]}),
    )
    assert ob.fetch("BTCUSDC", allow_network=True) is None

    # Unsorted levels -> sorted before mid and depth band calculation
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=4: _DummyResp({
            "bids": [["95.0", "10.0"], ["99.0", "1.0"], ["100.0", "2.0"]],
            "asks": [["105.0", "10.0"], ["101.0", "1.0"], ["100.0", "2.0"]],
        }),
    )
    snap = ob.fetch("BTCUSDC", allow_network=True)
    assert snap is not None
    assert snap.mid_price == 100.0
    # Within 2% of 100.0 ([98.0, 102.0]), 99.0*1 + 100.0*2 = 299.0 for bids, 100.0*2 + 101.0*1 = 301.0 for asks
    assert snap.bid_depth_usd == 299.0
    assert snap.ask_depth_usd == 301.0

def test_comma_separated_string_symbols_are_split_cleanly_instead_iter121_s1(tmp_path):
    """[ITER121 #1] Comma-separated string symbols are split cleanly instead of iterated character-by-character."""
    # 1. Comma-separated string symbols are split cleanly instead of iterated character-by-character
    itd = IntelligenceTelemetryDaemon(cache_dir=str(tmp_path), symbols=" btcusdc , taousdc ")
    assert itd.symbols == ["BTCUSDC", "TAOUSDC"]
    gtc = GlobalTelemetryCollector(cache_dir=str(tmp_path), symbols="solusdc,ethusdc")
    assert gtc.symbols == ["SOLUSDC", "ETHUSDC"]
    ma = MacroAnalyzer(cache_dir=str(tmp_path), symbols="btcusdc, xrpUsdc")
    assert ma.symbols == ["BTCUSDC", "XRPUSDC"]

def test_globaltelemetrycollector__run_step_bounded_sanitizes_invalid_timeout_sec_iter121_s2():
    """[ITER121 #2] GlobalTelemetryCollector._run_step_bounded sanitizes invalid timeout_sec."""
    # 2. GlobalTelemetryCollector._run_step_bounded sanitizes invalid timeout_sec
    assert GlobalTelemetryCollector._run_step_bounded(lambda: True, timeout_sec=None, label="test_none") is True
    assert GlobalTelemetryCollector._run_step_bounded(lambda: True, timeout_sec=False, label="test_bool") is True
    assert GlobalTelemetryCollector._run_step_bounded(lambda: True, timeout_sec=float("nan"), label="test_nan") is True

def test_intelligence_cli_treats_permissionerror_from_os_kill_pid_iter121_s3(monkeypatch, tmp_path):
    """[ITER121 #3] intelligence.cli treats PermissionError from os.kill(pid, 0) as alive and rejects boolean symbol."""
    from intelligence import cli as intel_cli
    # 3. intelligence.cli treats PermissionError from os.kill(pid, 0) as alive and rejects boolean symbol
    hb_file = tmp_path / "globaltelemetry_collector.heartbeat"
    hb_file.write_text(json.dumps({"pid": 99999, "ts": time.time(), "symbols": ["BTCUSDC"]}), encoding="utf-8")
    monkeypatch.setattr("os.kill", lambda _pid, _sig: (_ for _ in ()).throw(PermissionError("not owner")))
    st = intel_cli.get_daemon_status(cache_dir=str(tmp_path))
    assert st["running"] is True and st["status"] == "alive"

    inspected = intel_cli.inspect_symbol_intelligence(True, cache_dir=str(tmp_path))
    assert inspected["symbol"] == ""

def test_bybitliquidationcollector_splits_comma_separated_string_symbols_iter123_s1():
    """[ITER123 #1] BybitLiquidationCollector splits comma-separated string symbols."""
    # 1. BybitLiquidationCollector splits comma-separated string symbols
    blc = BybitLiquidationCollector(symbols=" btcusdt , solusdt ")
    assert blc._symbols == ["BTCUSDT", "SOLUSDT"]

def test_parabolicsurgeguard_whaledivergenceguard_orderbookwallguard_fundingcrowdingguard_highstakeguard_accept_bid_ask_iter174_s3():
    """[ITER174 #3] ParabolicSurgeGuard, WhaleDivergenceGuard, OrderbookWallGuard, FundingCrowdingGuard, HighStakeGuard accept BID/ASK/LONG/SHORT."""
    # 3. ParabolicSurgeGuard, WhaleDivergenceGuard, OrderbookWallGuard, FundingCrowdingGuard, HighStakeGuard accept BID/ASK/LONG/SHORT
    pg = ParabolicSurgeGuard(surge_threshold_pct=5.0)
    assert pg.check("BTCUSDC", "BID", current_price=100.0, price_history=[(1000.0, 90.0), (1100.0, 100.0)], now=1100.0).allowed is False
    assert pg.check("BTCUSDC", "ASK", current_price=90.0, price_history=[(1000.0, 100.0), (1100.0, 90.0)], now=1100.0).allowed is False

    assert WhaleDivergenceGuard().check("BTCUSDC", "LONG", snapshot=None).reason == "no_whale_snapshot_data"
    assert OrderbookWallGuard().check("BTCUSDC", "SHORT", snapshot=None).reason == "no_orderbook_snapshot"
    assert FundingCrowdingGuard().check("BTCUSDC", "A", telemetry=None).reason == "no_telemetry_data"
    assert HighStakeGuard().check("BTCUSDC", "BID", price=10.0, qty=1.0).reason != "unsupported_side_exempt"

def test_whaledivergenceguard_orderbookwallguard_fundingcrowdingguard_accept_dict_snapshots_iter175_s1():
    """[ITER175 #1] WhaleDivergenceGuard, OrderbookWallGuard, FundingCrowdingGuard accept dict snapshots."""
    # 1. WhaleDivergenceGuard, OrderbookWallGuard, FundingCrowdingGuard accept dict snapshots
    w_dec = WhaleDivergenceGuard().check(
        "BTCUSDC",
        "BUY",
        snapshot={"divergence_regime": "short_covering", "open_interest_1h_change_pct": -4.5},
    )
    assert w_dec.allowed is False and "short_covering_fakeout" in w_dec.reason

    ob_dec = OrderbookWallGuard(min_buy_imbalance=0.30).check(
        "BTCUSDC",
        "BUY",
        snapshot={"imbalance_ratio": 0.15, "largest_ask_wall_usd": 0.0},
    )
    assert ob_dec.allowed is False and "orderbook_heavily_ask_dominated" in ob_dec.reason

    f_dec = FundingCrowdingGuard(max_long_funding_rate=0.001, crowding_policy="defer").check(
        "BTCUSDC",
        "BUY",
        telemetry={"funding_rate": 0.005},
    )
    assert f_dec.allowed is False and "long_crowding_extreme" in f_dec.reason

def test_liquidationcascadeguard_accepts_bid_ask_long_short_and_dict_iter176_s2():
    """[ITER176 #2] LiquidationCascadeGuard accepts BID/ASK/LONG/SHORT and dict summary."""
    # 2. LiquidationCascadeGuard accepts BID/ASK/LONG/SHORT and dict summary
    lc_buy = LiquidationCascadeGuard(max_active_cascade_usd=100_000.0).check(
        "BTCUSDC",
        "BID",
        summary={"long_liq_usd": 250_000.0, "short_liq_usd": 10_000.0},
    )
    assert lc_buy.allowed is False and "active_long_liquidation_cascade" in lc_buy.reason

    lc_sell = LiquidationCascadeGuard(max_active_cascade_usd=100_000.0).check(
        "BTCUSDC",
        "SHORT",
        summary={"long_liq_usd": 10_000.0, "short_liq_usd": 250_000.0},
    )
    assert lc_sell.allowed is False and "active_short_liquidation_cascade" in lc_sell.reason

def test_sentimentcontrariantrigger_marketbreadthtrigger_liquidationcapitulationtrigger_accept_dict_snapshots_iter176_s3():
    """[ITER176 #3] SentimentContrarianTrigger, MarketBreadthTrigger, LiquidationCapitulationTrigger accept dict snapshots."""
    # 3. SentimentContrarianTrigger, MarketBreadthTrigger, LiquidationCapitulationTrigger accept dict snapshots
    sig_s, ev_s = SentimentContrarianTrigger().evaluate("BTCUSDC", 100.0, {"value": 15, "classification": "Extreme Fear"})
    assert sig_s == "CONTRARIAN_BUY" and ev_s is not None

    sig_b, ev_b = MarketBreadthTrigger().evaluate(
        "BTCUSDC",
        100.0,
        {"advance_ratio": 0.10, "median_change_pct": -6.0, "regime": "WASHOUT"},
    )
    assert sig_b == "BREADTH_WASHOUT_BUY" and ev_b is not None

    ev_l = LiquidationCapitulationTrigger(min_notional_usd=100_000.0, ratio_threshold=0.75).evaluate(
        "BTCUSDC",
        100.0,
        summary={"long_liq_usd": 180_000.0, "short_liq_usd": 20_000.0},
    )
    assert ev_l is not None and ev_l.reason == "liquidation_capitulation_burst"

def test_bybitliquidationcollector_record_event_accepts_bid_and_ask_side_aliases_iter177_s5():
    """[ITER177 #5] BybitLiquidationCollector.record_event accepts BID and ASK side aliases."""
    # 5. BybitLiquidationCollector.record_event accepts BID and ASK side aliases
    col = BybitLiquidationCollector(symbols=["BTCUSDT"])
    col.record_event("BTCUSDT", "BID", 150_000.0, ts=1000.0)
    col.record_event("BTCUSDT", "ASK", 50_000.0, ts=1000.0)
    s_liq = col.get_summary("BTCUSDT", window_sec=60.0, now=1005.0)
    assert s_liq.long_liq_usd == pytest.approx(150_000.0)
    assert s_liq.short_liq_usd == pytest.approx(50_000.0)

def test_orderbookdepthcollector_record_snapshot_accepts_dict_and_fetch_tolerates_non_iter178_s1():
    """[ITER178 #1] OrderbookDepthCollector.record_snapshot accepts dict and fetch() tolerates non-finite/None cache ts."""
    # 1. OrderbookDepthCollector.record_snapshot accepts dict and fetch() tolerates non-finite/None cache ts
    ob_col = OrderbookDepthCollector(cache_ttl_sec=60.0, cache_dir=None)
    ob_col.record_snapshot({"symbol": "BTCUSDC", "mid_price": 95000.0, "imbalance_ratio": 0.65, "ts": 1000.0})
    ob_snap = ob_col.fetch("BTCUSDC", allow_network=False, now=1010.0)
    assert ob_snap is not None and ob_snap.mid_price == pytest.approx(95000.0)
    ob_col._cache["BTCUSDC"] = (None, ob_snap)
    assert ob_col.fetch("BTCUSDC", allow_network=False, now=1010.0) is None

def test_derivativestelemetrycollector_record_snapshot_accepts_dict_and_fetch_tolerates_none_iter178_s2():
    """[ITER178 #2] DerivativesTelemetryCollector.record_snapshot accepts dict and fetch() tolerates None cache ts."""
    # 2. DerivativesTelemetryCollector.record_snapshot accepts dict and fetch() tolerates None cache ts
    dt_col = DerivativesTelemetryCollector(cache_ttl_sec=60.0, cache_dir=None)
    dt_col.record_snapshot({"symbol": "BTCUSDC", "funding_rate": 0.0004, "open_interest_usd": 5_000_000.0, "ts": 1000.0})
    dt_snap = dt_col.fetch("BTCUSDC", allow_network=False, now=1010.0)
    assert dt_snap is not None and dt_snap.funding_rate == pytest.approx(0.0004)
    dt_col._cache["BTCUSDC"] = (True, dt_snap)
    assert dt_col.fetch("BTCUSDC", allow_network=False, now=1010.0) is None

def test_whalepositioningcollector_record_snapshot_accepts_dict_and_fetch_tolerates_none_iter178_s3():
    """[ITER178 #3] WhalePositioningCollector.record_snapshot accepts dict and fetch() tolerates None cache ts."""
    # 3. WhalePositioningCollector.record_snapshot accepts dict and fetch() tolerates None cache ts
    wh_col = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
    wh_col.record_snapshot({"symbol": "BTCUSDC", "taker_buy_sell_ratio": 1.35, "divergence_regime": "accumulation", "ts": 1000.0})
    wh_snap = wh_col.fetch("BTCUSDC", allow_network=False, now=1010.0)
    assert wh_snap is not None and wh_snap.divergence_regime == "accumulation"
    wh_col._cache["BTCUSDC"] = (float("nan"), wh_snap)
    assert wh_col.fetch("BTCUSDC", allow_network=False, now=1010.0) is None

def test__diskbackedsymbolcollector__fetch_json_dict_and__fetch_latest_json_row_validate_payload_types_iter183_s4(monkeypatch):
    """[ITER183 #4] _DiskBackedSymbolCollector._fetch_json_dict and _fetch_latest_json_row validate payload types."""
    from intelligence.external.collectors.orderbook_depth_collector import _DiskBackedSymbolCollector
    # 4. _DiskBackedSymbolCollector._fetch_json_dict and _fetch_latest_json_row validate payload types
    import io
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=5: io.BytesIO(b'{"markPrice": "70000"}'))
    assert _DiskBackedSymbolCollector._fetch_json_dict("https://example.com", "premiumIndex", "BTCUSDT") == {"markPrice": "70000"}
    with pytest.raises(ValueError, match="Empty topLongShortPositionRatio"):
        _DiskBackedSymbolCollector._fetch_latest_json_row("https://example.com", "topLongShortPositionRatio", "BTCUSDT")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=5: io.BytesIO(b'[{"longShortRatio": "1.2"}, {"longShortRatio": "1.5"}]'),
    )
    latest, rows = _DiskBackedSymbolCollector._fetch_latest_json_row("https://example.com", "topLongShortPositionRatio", "BTCUSDT")
    assert latest == {"longShortRatio": "1.5"} and len(rows) == 2


def test_intelligence_and_market_monitor_zero_venue_sdk_imports_ast():
    """Enforce zero direct venue SDK imports across all intelligence/ and market_monitor/ modules."""
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    forbidden_venue_sdks = {"binance_api", "kraken", "hyperliquid", "212trading"}

    for pkg_name in ("intelligence", "market_monitor"):
        pkg_dir = repo_root / pkg_name
        for py_file in sorted(pkg_dir.rglob("*.py")):
            rel = str(py_file.relative_to(repo_root))
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    imported.add(node.args[0].value.split(".")[0])
            leak = imported & forbidden_venue_sdks
            assert not leak, f"{rel} imports venue SDK directly instead of providers/: {leak}"

