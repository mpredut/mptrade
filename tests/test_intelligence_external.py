"""Comprehensive unit tests for external intelligence (Pillar 2: liquidations, guards, telemetry)."""

import unittest
import time

from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector, LiquidationSummary
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry, DerivativesTelemetryCollector
from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.internal.guards.guard_decision import BrakeAction
from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerSide


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
        collector.record_telemetry(telem)

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
        from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
        from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard

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
        from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
        from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard

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

    def test_whale_accumulation_trigger_fires(self):
        from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
        from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger

        trigger = WhaleAccumulationTrigger()
        # Whales are 68% long, taker buys 1.6x sells, and OI is expanding (+4.2%)
        snapshot_acc = WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=2.12,
            top_traders_long_pct=0.68,
            taker_buy_sell_ratio=1.60,
            taker_buy_vol_usd=2_000_000.0,
            taker_sell_vol_usd=1_250_000.0,
            open_interest_usd=8_500_000_000.0,
            open_interest_1h_change_pct=+4.2,
            divergence_regime="accumulation",
            ts=time.time(),
        )

        ev = trigger.evaluate("BTCUSDC", price=85000.0, snapshot=snapshot_acc)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.action, TriggerAction.ENTRY)
        self.assertEqual(ev.side, TriggerSide.BUY)
        self.assertEqual(ev.reason, "whale_aggressive_accumulation")

    def test_disk_serialization_round_trip(self):
        import tempfile
        import os
        from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot, WhalePositioningCollector
        from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot, OrderbookDepthCollector
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry, DerivativesTelemetryCollector

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
            d_collector.record_telemetry(d_snap, save_to_disk=True)

            d_collector2 = DerivativesTelemetryCollector(cache_ttl_sec=30.0, cache_dir=tmpdir)
            loaded_d = d_collector2.fetch("SOLUSDT")
            self.assertIsNotNone(loaded_d)
            self.assertEqual(loaded_d.symbol, "SOLUSDT")
            self.assertEqual(loaded_d.funding_rate, 0.0002)


class TestOrderGuardPillar2Integration(unittest.TestCase):
    """Verify Pillar 2 guards wire cleanly into order_guard.check_intelligence_guards."""

    def test_whale_guard_shadow_allows_while_enforce_blocks(self):
        import order_guard
        from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot, WhalePositioningCollector

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

        orig_collector = order_guard._whale_collector
        orig_margins = order_guard._MARGINS
        try:
            order_guard._whale_collector = collector

            # Test SHADOW mode: logs warning but allows trade
            order_guard._MARGINS = {
                "intelligence_guards_mode": "shadow",
                "whale_guard_mode": "shadow",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "off",
                "gemini_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
            allowed_shadow, reason_shadow, scale_shadow = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertTrue(allowed_shadow)
            self.assertEqual(scale_shadow, 1.0)

            # Test ENFORCE mode: hard veto
            order_guard._MARGINS = {
                "intelligence_guards_mode": "enforce",
                "whale_guard_mode": "enforce",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "off",
                "gemini_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
            allowed_enforce, reason_enforce, scale_enforce = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 85000.0
            )
            self.assertFalse(allowed_enforce)
            self.assertIn("short_covering_fakeout", reason_enforce)
            self.assertEqual(scale_enforce, 0.0)
        finally:
            order_guard._whale_collector = orig_collector
            order_guard._MARGINS = orig_margins

    def test_orderbook_wall_guard_shadow_and_enforce(self):
        import order_guard
        from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot, OrderbookDepthCollector

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

        orig_collector = order_guard._orderbook_collector
        orig_margins = order_guard._MARGINS
        try:
            order_guard._orderbook_collector = collector

            # Test SHADOW mode
            order_guard._MARGINS = {
                "intelligence_guards_mode": "shadow",
                "whale_guard_mode": "off",
                "orderbook_wall_guard_mode": "shadow",
                "whale_wall_usd_limit": 1_000_000.0,
                "funding_guard_mode": "off",
                "gemini_guard_mode": "off",
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
            order_guard._orderbook_collector = orig_collector
            order_guard._MARGINS = orig_margins

    def test_funding_crowding_guard_downscales_in_enforce(self):
        import order_guard
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry, DerivativesTelemetryCollector

        crowded_snap = DerivativesTelemetry(
            symbol="BTCUSDT",
            funding_rate=0.0008,
            predicted_funding_rate=0.0008,
            open_interest=50000.0,
            open_interest_usd=4_250_000_000.0,
            ts=time.time(),
        )

        collector = DerivativesTelemetryCollector(cache_ttl_sec=60.0, cache_dir=None)
        collector.record_telemetry(crowded_snap)

        orig_collector = order_guard._derivatives_collector
        orig_margins = order_guard._MARGINS
        try:
            order_guard._derivatives_collector = collector

            # Test SHADOW mode
            order_guard._MARGINS = {
                "intelligence_guards_mode": "shadow",
                "whale_guard_mode": "off",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "shadow",
                "funding_max_long_rate": 0.0005,
                "funding_crowded_scale": 0.50,
                "gemini_guard_mode": "off",
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
            order_guard._derivatives_collector = orig_collector
            order_guard._MARGINS = orig_margins


if __name__ == "__main__":
    unittest.main()

