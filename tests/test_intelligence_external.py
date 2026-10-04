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


if __name__ == "__main__":
    unittest.main()
