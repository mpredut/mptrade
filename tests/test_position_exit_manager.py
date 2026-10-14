"""Unit tests for Centralized PositionExitManager (MHDSL Stop-Loss & MDHTP Take-Profit Mirror)."""

import time
import unittest
from unittest.mock import MagicMock

from strategies.position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentryAction,
    ReentryDecision,
    ReentrySnapshot,
)


class TestPositionExitManager(unittest.TestCase):
    def setUp(self):
        self.config = PositionExitConfig(
            lost_threshold=0.05,
            gain_threshold=0.07,
            sl_trend_defer_max_sec=220.0,
            tp_stagnation_max_sec=1080.0,
            sl_bounce_min_recovery_pct=0.004,
            sl_bounce_pullback_pct=0.003,
            sl_bounce_ratchet_enabled=True,
            sl_bounce_ratchet_k=0.50,
            sl_bounce_min_pullback_pct=0.0015,
            sl_structural_break_enabled=True,
            smart_sl_enabled=False,  # Isolated unit testing without external daemon
            tp_trend_ride_enabled=True,
            tp_profit_floor_ratio=0.50,
            tp_profit_floor_min_pct=0.008,
            tp_pullback_pct=0.006,
        )
        self.mgr = PositionExitManager(config=self.config)
        self.now = 1700000000.0

    # -------------------------------------------------------------------------
    # 1. Disaster Buffer Volatility Scaling
    # -------------------------------------------------------------------------
    def test_disaster_buffer_variant_a_wins_when_vol_is_low(self):
        """When 1h volatility is low, Variant A (lost_threshold * ratio) defines the buffer."""
        buf, meta = self.mgr.compute_disaster_buffer(
            symbol="BTCUSDT",
            lost_threshold=0.05,
            vol_1h=0.01,  # 1% vol * 0.85 = 0.0085 < Variant A (0.05 * 0.30 = 0.015)
        )
        self.assertAlmostEqual(buf, 0.015, places=5)
        self.assertEqual(meta["source"], "variant_a_proportional")
        self.assertAlmostEqual(meta["variant_a_buffer"], 0.015, places=5)

    def test_disaster_buffer_variant_b_wins_when_vol_is_high(self):
        """When 1h volatility is high, Variant B (vol_1h * multiplier) widens the buffer."""
        buf, meta = self.mgr.compute_disaster_buffer(
            symbol="BTCUSDT",
            lost_threshold=0.05,
            vol_1h=0.03,  # 3% vol * 0.85 = 0.0255 > Variant A (0.015)
        )
        self.assertAlmostEqual(buf, 0.0255, places=5)
        self.assertEqual(meta["source"], "variant_b_vol1h")
        self.assertAlmostEqual(meta["variant_b_buffer"], 0.0255, places=5)

    def test_disaster_buffer_floor_min_wins_for_tight_stops(self):
        """For very tight stops and zero volatility, minimum floor applies."""
        buf, meta = self.mgr.compute_disaster_buffer(
            symbol="BTCUSDT",
            lost_threshold=0.01,  # 1% * 0.30 = 0.003 < 0.008 min floor
            vol_1h=0.0,
        )
        self.assertAlmostEqual(buf, 0.008, places=5)
        self.assertEqual(meta["source"], "variant_a_proportional")

    # -------------------------------------------------------------------------
    # 2. Stop-Loss (MHDSL) Hierarchy & Symmetry
    # -------------------------------------------------------------------------
    def test_sl_neutral_corridor_holds(self):
        """Loss below threshold is held with no exit."""
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=98.0,
            avg_buy_price=100.0,  # 2.0% loss < 5.0% threshold
            current_time_s=self.now,
            trend_up=False,
        )
        dec = self.mgr.evaluate_stop_loss(snap)
        self.assertEqual(dec.action, ExitAction.HOLD)
        self.assertFalse(dec.should_exit)

    def test_sl_p1_hard_disaster_floor_bypasses_trend_up(self):
        """Priority 1: Hard disaster floor breach exits immediately, even if trend_up is True."""
        # Threshold 5.0% + buffer 1.5% = floor 6.5%.
        # Price 93.0 -> loss 7.0% >= 6.5%.
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=93.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
            vol_1h=0.01,
        )
        dec = self.mgr.evaluate_stop_loss(snap)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("hard_disaster_floor_breached", dec.reason)
        self.assertEqual(dec.metadata["priority"], 1)
        self.assertNotIn("BTCUSDT", self.mgr.sl_bounce_state)

    def test_sl_p2_bounce_watch_initiated_on_trend_up(self):
        """Priority 2: When in loss corridor and trend_up is True, bounce watch is armed and exit deferred."""
        # Price 94.5 -> loss 5.5% (between threshold 5.0% and floor 6.5%).
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        dec = self.mgr.evaluate_stop_loss(snap)
        self.assertEqual(dec.action, ExitAction.HOLD)
        self.assertFalse(dec.should_exit)
        self.assertIn("BTCUSDT", self.mgr.sl_bounce_state)
        state = self.mgr.sl_bounce_state["BTCUSDT"]
        self.assertEqual(state["lowest_price"], 94.5)
        self.assertEqual(state["peak_bounce_price"], 94.5)

    def test_sl_p2_bounce_timeout_forces_exit(self):
        """Priority 2: When bounce watch exceeds sl_trend_defer_max_sec (220s), exit is forced."""
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap1)

        # 225s later, price still at 94.7 (loss 5.3% > 5%)
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.7,
            avg_buy_price=100.0,
            current_time_s=self.now + 225.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_stop_loss(snap2)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mhdsl_bounce_timeout", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.sl_bounce_state)

    def test_sl_p2_bounce_rollover_forces_exit(self):
        """Priority 2: Attempted bounce rolls over by >= pullback_pct -> exit forced."""
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap1)

        # Bounce up to 94.6 (+0.638% recovery >= 0.40% min recovery)
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.6,
            avg_buy_price=100.0,
            current_time_s=self.now + 10.0,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap2)

        # Pullback to 94.25 (pullback (94.6 - 94.25)/94.6 = 0.37% >= 0.30% base threshold)
        snap3 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.25,
            avg_buy_price=100.0,
            current_time_s=self.now + 20.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_stop_loss(snap3)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mhdsl_bounce_rollover", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.sl_bounce_state)

    def test_sl_p2_structural_break_variant_proposed(self):
        """Priority 2: When sl_structural_break_enabled is True, price breaking below lowest_price exits immediately."""
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap1)

        # Bounce to 94.9 (+0.42% >= 0.40%)
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.9,
            avg_buy_price=100.0,
            current_time_s=self.now + 10.0,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap2)

        # Breakdown to 94.4 (below lowest_price 94.5)
        snap3 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.4,
            avg_buy_price=100.0,
            current_time_s=self.now + 20.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_stop_loss(snap3)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mhdsl_structural_break", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.sl_bounce_state)

    def test_sl_p2_structural_break_variant_baseline_resets_trough(self):
        """Priority 2: When sl_structural_break_enabled is False, price breaking below lowest_price resets trough and defers."""
        self.mgr.config.sl_structural_break_enabled = False
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap1)

        # Bounce to 94.9
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.9,
            avg_buy_price=100.0,
            current_time_s=self.now + 10.0,
            trend_up=True,
        )
        self.mgr.evaluate_stop_loss(snap2)

        # Drop to 94.4 with structural break disabled
        snap3 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.4,
            avg_buy_price=100.0,
            current_time_s=self.now + 20.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_stop_loss(snap3)
        self.assertEqual(dec.action, ExitAction.HOLD)
        self.assertFalse(dec.should_exit)
        self.assertIn("BTCUSDT", self.mgr.sl_bounce_state)
        self.assertEqual(self.mgr.sl_bounce_state["BTCUSDT"]["lowest_price"], 94.4)

    def test_sl_p3_trend_down_triggers_exit(self):
        """Priority 3: Loss crosses threshold with trend_up=False exits."""
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=False,
        )
        dec = self.mgr.evaluate_stop_loss(snap)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertIn("stop_loss_threshold_breached", dec.reason)

    # -------------------------------------------------------------------------
    # 3. Take-Profit (MDHTP) Mirror Symmetry
    # -------------------------------------------------------------------------
    def test_tp_p0_hard_tp_spike_executes_partial_sell(self):
        """Priority 0: Massive spike (gain >= 20%) triggers partial Hard TP regardless of trend."""
        self.mgr.config.hard_tp_enabled = True
        self.mgr.config.hard_tp_pct = 0.20
        self.mgr.config.hard_tp_frac = 0.50

        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=122.0,
            avg_buy_price=100.0,  # +22.0% gain
            current_time_s=self.now,
            trend_up=True,
        )
        dec = self.mgr.evaluate_take_profit(snap)
        self.assertEqual(dec.action, ExitAction.PARTIAL_SELL)
        self.assertTrue(dec.should_exit)
        self.assertEqual(dec.sell_fraction, 0.50)
        self.assertIn("hard_tp_spike", dec.reason)

    def test_tp_p1_trend_ride_initiated_on_trend_up(self):
        """Priority 1: Gain reaches threshold (7%) with trend_up=True initiates trend ride."""
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,  # +8.0% gain
            current_time_s=self.now,
            trend_up=True,
        )
        dec = self.mgr.evaluate_take_profit(snap)
        self.assertEqual(dec.action, ExitAction.HOLD)
        self.assertFalse(dec.should_exit)
        self.assertIn("BTCUSDT", self.mgr.tp_ride_state)
        state = self.mgr.tp_ride_state["BTCUSDT"]
        self.assertEqual(state["peak_price"], 108.0)
        self.assertEqual(state["last_peak_time"], self.now)

    def test_tp_p1_trend_not_up_exits_immediately(self):
        """Priority 1: Gain reaches threshold (7%) with trend_up=False exits without deferral."""
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=False,
        )
        dec = self.mgr.evaluate_take_profit(snap)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertIn("gain_threshold_exceeded_and_trend_not_up", dec.reason)

    def test_tp_p2_profit_floor_breached_forces_exit(self):
        """Priority 2: Price slides below guaranteed profit floor (3.5%) forces TP."""
        # Initiate at 108.0
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_take_profit(snap1)

        # Price slides to 103.0 (+3.0% gain < floor 3.5%)
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=103.0,
            avg_buy_price=100.0,
            current_time_s=self.now + 30.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_take_profit(snap2)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mdhtp_profit_floor_breached", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.tp_ride_state)

    def test_tp_p2_profit_floor_breached_on_gap_down_below_cost(self):
        """Priority 2: Price gaps down below entry cost while in tp_ride_state forces TP exit."""
        # Initiate at 108.0 (entry 100.0)
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_take_profit(snap1)
        self.assertIn("BTCUSDT", self.mgr.tp_ride_state)

        # Price violently drops to 99.0 (-1.0% below cost)
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=99.0,
            avg_buy_price=100.0,
            current_time_s=self.now + 10.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_take_profit(snap2)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mdhtp_profit_floor_breached", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.tp_ride_state)

    def test_tp_p3_peak_pullback_forces_exit(self):
        """Priority 3: Price pulls back >= 0.60% from peak forces TP."""
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=110.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_take_profit(snap1)

        # Pullback to 109.2 ((110 - 109.2)/110 = 0.727% >= 0.60%)
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=109.2,
            avg_buy_price=100.0,
            current_time_s=self.now + 20.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_take_profit(snap2)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mdhtp_peak_pullback", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.tp_ride_state)

    def test_tp_p4_stagnation_timeout_forces_exit(self):
        """Priority 4: No new peak for >= 1080s forces TP."""
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=110.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_take_profit(snap1)

        # Price at 109.8 (slight dip < 0.60% pullback) at t = now + 1100s
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=109.8,
            avg_buy_price=100.0,
            current_time_s=self.now + 1100.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate_take_profit(snap2)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mdhtp_stagnation_timeout", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.tp_ride_state)

    def test_tp_p5_trend_reversal_forces_exit(self):
        """Priority 5: Active ride when trend_up flips to False forces immediate exit."""
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=109.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate_take_profit(snap1)

        # Next tick trend turns down
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=109.0,
            avg_buy_price=100.0,
            current_time_s=self.now + 15.0,
            trend_up=False,
        )
        dec = self.mgr.evaluate_take_profit(snap2)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mdhtp_trend_reversal", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.tp_ride_state)

    # -------------------------------------------------------------------------
    # 4. Unified evaluate() Entrypoint
    # -------------------------------------------------------------------------
    def test_unified_evaluate_clears_sl_when_moving_to_tp(self):
        """Transitioning from SL corridor to TP corridor clears any lingering SL state."""
        # Arm SL state
        snap_sl = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate(snap_sl)
        self.assertIn("BTCUSDT", self.mgr.sl_bounce_state)

        # Huge pump into TP territory (+8%)
        snap_tp = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now + 60.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate(snap_tp)
        self.assertEqual(dec.action, ExitAction.HOLD)
        self.assertNotIn("BTCUSDT", self.mgr.sl_bounce_state)
        self.assertIn("BTCUSDT", self.mgr.tp_ride_state)


class TestPositionReentryManager(unittest.TestCase):
    def setUp(self):
        self.config = PositionExitConfig(
            reentry_enabled=True,
            reentry_sl_bounce_pct=0.015,
            reentry_drop_pct=0.020,
            reentry_pullback_pct=0.015,
            reentry_bear_bounce_pct=0.010,
            reentry_ttl_hours=2.0,
            reentry_tolerance_pct=0.05,
            reentry_ratchet_enabled=True,
            reentry_bear_gate_enabled=True,
            reentry_hybrid_enabled=False,
        )
        self.mgr = PositionExitManager(config=self.config)
        self.now = 1_000_000.0

    def test_reentry_without_prior_sale_barrier(self):
        snap = ReentrySnapshot(symbol="BTCUSDT", current_price=100.0, last_sell_price=None)
        dec = self.mgr.evaluate_reentry(snap)
        self.assertTrue(dec.should_enter)
        self.assertEqual(dec.action, ReentryAction.ENTER)

    def test_reentry_ttl_expired_allows_entry(self):
        snap = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=105.0,
            last_sell_price=100.0,
            last_sell_time_s=self.now - 3.0 * 3600,  # 3h > 2h TTL
            current_time_s=self.now,
        )
        dec = self.mgr.evaluate_reentry(snap)
        self.assertTrue(dec.should_enter)
        self.assertIn("reentry_ttl_expired", dec.reason)

    def test_reentry_post_stop_bounce_watch_and_entry(self):
        # Sold at 100 on STOP
        # Price drops to 90 -> establishes trough at 90, bounce is 0% -> wait
        snap1 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=90.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            current_time_s=self.now,
            trend_up=True,
        )
        dec1 = self.mgr.evaluate_reentry(snap1)
        self.assertFalse(dec1.should_enter)
        self.assertEqual(dec1.action, ReentryAction.WAIT)
        self.assertIn("post_stop_bounce_waiting", dec1.reason)

        # Price bounces to 92 (2.22% >= 1.5% required) -> enter
        snap2 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=92.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            current_time_s=self.now + 10.0,
            trend_up=True,
        )
        dec2 = self.mgr.evaluate_reentry(snap2)
        self.assertTrue(dec2.should_enter)
        self.assertEqual(dec2.action, ReentryAction.ENTER)
        self.assertIn("post_stop_bounce_confirmed", dec2.reason)

    def test_reentry_post_tp_discount_drop_range(self):
        # Sold at 100 on TP, drop_pct=2% (target=98)
        # Price at 99 -> above 98 -> wait
        snap1 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=99.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now,
            trend_up=False,
        )
        dec1 = self.mgr.evaluate_reentry(snap1)
        self.assertFalse(dec1.should_enter)
        self.assertIn("range_reentry_drop_waiting", dec1.reason)

        # Price drops to 96.0 <= 98 -> establishes trough at 96.0, waits for knife-catching bounce
        snap2 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=96.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now + 10.0,
            trend_up=False,
        )
        dec2 = self.mgr.evaluate_reentry(snap2)
        self.assertFalse(dec2.should_enter)
        self.assertIn("bear_reentry_trough_bounce_waiting", dec2.reason)

        # Price bounces to 97.5 (1.56% >= 1.0% bear bounce and <= 98.0 discount target) -> enter
        snap3 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=97.5,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now + 20.0,
            trend_up=False,
        )
        dec3 = self.mgr.evaluate_reentry(snap3)
        self.assertTrue(dec3.should_enter)
        self.assertIn("range_reentry_confirmed", dec3.reason)

    def test_reentry_hybrid_pullback_in_bull_trend(self):
        self.mgr.config.reentry_hybrid_enabled = True
        # Sold at 100 on TP, bull trend, pullback_pct=1.5%
        # Price rallies to 110 (new peak)
        snap1 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=110.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now,
            trend_up=True,
        )
        dec1 = self.mgr.evaluate_reentry(snap1)
        self.assertFalse(dec1.should_enter)
        self.assertIn("bull_reentry_pullback_waiting", dec1.reason)

        # Price pulls back to 108 (drop of 1.82% from 110 >= 1.5%) -> enter
        snap2 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now + 20.0,
            trend_up=True,
        )
        dec2 = self.mgr.evaluate_reentry(snap2)
        self.assertTrue(dec2.should_enter)
        self.assertIn("bull_reentry_pullback_confirmed", dec2.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
