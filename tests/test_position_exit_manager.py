"""Unit tests for Centralized PositionExitManager (MHDSL Stop-Loss & MDHTP Take-Profit Mirror)."""

import unittest

from strategies.position_exit_manager import (
    ExitAction,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentryAction,
    ReentrySnapshot,
)
import math
import monitortrades as mt
import pytest


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
            allow_external_cache=False,
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

    def test_disaster_buffer_capped_by_max_pct(self):
        """Extreme volatility spikes cannot expand the disaster buffer beyond sl_disaster_buffer_max_pct."""
        self.mgr.config.sl_disaster_buffer_max_pct = 0.10  # 10% maximum buffer cap
        buf, meta = self.mgr.compute_disaster_buffer(
            symbol="BTCUSDT",
            lost_threshold=0.05,
            vol_1h=0.25,  # 25% vol * 0.85 = 0.2125 > 0.10 cap
        )
        self.assertAlmostEqual(buf, 0.10, places=5)
        self.assertTrue(meta["capped"])
        self.assertEqual(meta["source"], "capped_max")
        self.assertEqual(meta["buffer_max"], 0.10)

    def test_sl_disaster_floor_cannot_loosen_when_volatility_surges(self):
        """Once armed in SL territory, disaster floor preserves protection and cannot loosen when volatility spikes."""
        # 1. Enter SL territory at 5.5% loss with vol_1h = 0.01 (buffer = 1.5%, floor = 6.5%)
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
            vol_1h=0.01,
        )
        dec1 = self.mgr.evaluate_stop_loss(snap1)
        self.assertEqual(dec1.action, ExitAction.HOLD)
        self.assertIn("mhdsl_bounce_watch_initiated", dec1.reason)
        armed_floor = dec1.metadata["disaster_floor"]
        self.assertAlmostEqual(armed_floor, 0.065, places=4)

        # 2. Next tick: volatility explodes to 20% (raw buffer would be 17%, raw floor 22%)
        # Price drops to 93.0 (loss 7.0% >= armed floor 6.5%).
        # Protection MUST NOT retreat to 22%; it must trigger Priority 1 immediate disaster exit.
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=93.0,
            avg_buy_price=100.0,
            current_time_s=self.now + 10.0,
            trend_up=True,
            vol_1h=0.20,
        )
        dec2 = self.mgr.evaluate_stop_loss(snap2)
        self.assertEqual(dec2.action, ExitAction.SELL)
        self.assertTrue(dec2.should_exit)
        self.assertTrue(dec2.forced)
        self.assertEqual(dec2.metadata["priority"], 1)
        self.assertIn("hard_disaster_floor_breached", dec2.reason)
        # Floor must remain at or below the armed 6.5% floor
        self.assertAlmostEqual(dec2.metadata["disaster_floor"], 0.065, places=4)

    def _smart_sl_manager(self):
        """Manager with a Smart-SL stub that mirrors the guard: defer unless its own disaster floor is breached."""
        calls = []

        def smart_sl_stub(**kw):
            calls.append(kw)
            floor = kw["lost_threshold"] + kw["disaster_buffer_pct"] / 100.0
            if kw["price_decrease"] >= floor:
                return True, "hard_disaster_floor_breached", {}
            return False, "whale_supported_wick", {}

        self.config.smart_sl_enabled = True
        return PositionExitManager(config=self.config, smart_sl_evaluator=smart_sl_stub), calls

    def _snap(self, price, t, trend_up, vol):
        return PositionSnapshot(
            symbol="BTCUSDT", current_price=price, avg_buy_price=100.0,
            current_time_s=self.now + t, trend_up=trend_up, vol_1h=vol,
        )

    def test_sl_episode_floor_survives_trend_to_smart_sl_handoff(self):
        """Floor armed under trend-up bounce protection must persist after Smart-SL takes over deferral."""
        mgr, calls = self._smart_sl_manager()

        # 1. Trend up, 5.5% loss, low vol -> bounce watch armed with floor 6.5%
        dec1 = mgr.evaluate_stop_loss(self._snap(94.5, 0, True, 0.01))
        self.assertIn("mhdsl_bounce_watch_initiated", dec1.reason)
        self.assertAlmostEqual(dec1.metadata["disaster_floor"], 0.065, places=4)

        # 2. Trend flips, 6.0% loss -> bounce state cleared, Smart-SL defers; episode lock persists
        dec2 = mgr.evaluate_stop_loss(self._snap(94.0, 10, False, 0.01))
        self.assertEqual(dec2.action, ExitAction.HOLD)
        self.assertIn("smart_stop_loss_deferred", dec2.reason)
        self.assertNotIn("BTCUSDT", mgr.sl_bounce_state)
        self.assertAlmostEqual(mgr.sl_episode_state["BTCUSDT"]["disaster_floor"], 0.065, places=4)

        # 3. Volatility spikes (raw floor would be 20%), loss 6.2% -> Smart-SL receives the locked 1.5% buffer
        dec3 = mgr.evaluate_stop_loss(self._snap(93.8, 20, False, 0.20))
        self.assertEqual(dec3.action, ExitAction.HOLD)
        self.assertAlmostEqual(calls[-1]["disaster_buffer_pct"], 1.5, places=4)
        self.assertAlmostEqual(dec3.metadata["disaster_floor"], 0.065, places=4)

        # 4. Loss 7.0% >= locked 6.5% -> immediate forced disaster exit (no further waiting)
        dec4 = mgr.evaluate_stop_loss(self._snap(93.0, 30, False, 0.20))
        self.assertEqual(dec4.action, ExitAction.SELL)
        self.assertTrue(dec4.forced)
        self.assertEqual(dec4.metadata["priority"], 1)
        self.assertAlmostEqual(dec4.metadata["disaster_floor"], 0.065, places=4)
        self.assertEqual(dec4.metadata["disaster_meta"]["source"], "episode_locked")

    def test_sl_episode_floor_resets_when_episode_ends(self):
        """Leaving SL territory or changing the entry price starts a fresh episode with a recomputed floor."""
        mgr, _ = self._smart_sl_manager()
        mgr.evaluate_stop_loss(self._snap(94.5, 0, False, 0.01))
        self.assertAlmostEqual(mgr.sl_episode_state["BTCUSDT"]["disaster_floor"], 0.065, places=4)

        # Recovery out of SL territory ends the episode
        dec = mgr.evaluate_stop_loss(self._snap(99.0, 10, False, 0.01))
        self.assertEqual(dec.reason, "not_in_sl_territory")
        self.assertNotIn("BTCUSDT", mgr.sl_episode_state)

        # New episode under high vol arms its own (wider, capped) floor
        dec = mgr.evaluate_stop_loss(self._snap(94.0, 20, False, 0.20))
        self.assertAlmostEqual(dec.metadata["disaster_floor"], 0.20, places=4)

        # Entry price change (DCA / new position) also starts a fresh episode
        mgr.evaluate_stop_loss(self._snap(94.0, 30, False, 0.01))  # same episode: stays tightened to 6.5%
        self.assertAlmostEqual(mgr.sl_episode_state["BTCUSDT"]["disaster_floor"], 0.065, places=4)
        snap = PositionSnapshot(
            symbol="BTCUSDT", current_price=90.0, avg_buy_price=95.0,
            current_time_s=self.now + 40, trend_up=False, vol_1h=0.20,
        )
        dec = mgr.evaluate_stop_loss(snap)
        self.assertAlmostEqual(dec.metadata["disaster_floor"], 0.20, places=4)
        self.assertEqual(mgr.sl_episode_state["BTCUSDT"]["buy_price"], 95.0)

    def test_unified_evaluate_resets_sl_episode_on_neutral_recovery(self):
        mgr, _ = self._smart_sl_manager()
        dec1 = mgr.evaluate(self._snap(94.5, 0, False, 0.01))
        self.assertEqual(dec1.action, ExitAction.HOLD)
        self.assertIn("BTCUSDT", mgr.sl_episode_state)

        dec2 = mgr.evaluate(self._snap(99.0, 10, False, 0.01))
        self.assertEqual(dec2.reason, "neutral_corridor")
        self.assertNotIn("BTCUSDT", mgr.sl_episode_state)

    def test_bind_state_wires_all_persisted_exit_dicts(self):
        """bind_state must expose every durable lifecycle dict (incl. SL episode) to the external store."""
        store = {}
        self.mgr.bind_state(store)
        self.mgr.evaluate_stop_loss(self._snap(94.5, 0, True, 0.01))
        self.assertIn("BTCUSDT", store["sl_episode_state"])
        self.assertIn("BTCUSDT", store["sl_bounce_state"])
        self.mgr.clear_state("BTCUSDT")
        self.assertNotIn("BTCUSDT", store["sl_episode_state"])

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

    def test_unified_evaluate_tp_ride_gap_down_into_sl_forces_immediate_exit(self):
        """A position actively riding TP that gaps down into SL territory must fire mdhtp_profit_floor_breached immediately."""
        snap_tp = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        self.mgr.evaluate(snap_tp)
        self.assertIn("BTCUSDT", self.mgr.tp_ride_state)

        # Gap down to 94.5 (-5.5% loss, inside SL bounce corridor with trend_up=True)
        snap_gap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.5,
            avg_buy_price=100.0,
            current_time_s=self.now + 10.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate(snap_gap)
        self.assertEqual(dec.action, ExitAction.SELL)
        self.assertTrue(dec.should_exit)
        self.assertTrue(dec.forced)
        self.assertIn("mdhtp_profit_floor_breached", dec.reason)
        self.assertNotIn("BTCUSDT", self.mgr.tp_ride_state)


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

    def test_reentry_hybrid_disabled_enforces_discount_drop_even_in_bull_trend(self):
        # Baseline mode: hybrid reentry is explicitly False
        self.mgr.config.reentry_hybrid_enabled = False
        # Sold at 100 on TP, bull trend, price rallies to 110 and pulls back to 108
        snap1 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now,
            trend_up=True,
        )
        dec1 = self.mgr.evaluate_reentry(snap1)
        # Must NOT buy at 108 (which is above last sell of 100) because hybrid pullback is disabled
        self.assertFalse(dec1.should_enter)
        self.assertIn("range_reentry_drop_waiting", dec1.reason)

        # Only enters when price drops below the 2.0% discount target (100 * 0.98 = 98.0)
        snap2 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=97.5,
            last_sell_price=100.0,
            last_exit_kind="TP",
            current_time_s=self.now + 30.0,
            trend_up=True,
        )
        dec2 = self.mgr.evaluate_reentry(snap2)
        self.assertTrue(dec2.should_enter)
        self.assertIn("range_reentry_confirmed", dec2.reason)

    def test_reentry_post_stop_stabilization_cooldown_waits_and_tracks_trough(self):
        self.mgr.config.reentry_post_stop_cooldown_sec = 300.0
        # Tick 1 at t=now+30s (< 300s cooldown): price drops to 90.0 -> tracks trough=90.0, waits
        snap1 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=90.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            last_sell_time_s=self.now,
            current_time_s=self.now + 30.0,
            trend_up=True,
        )
        dec1 = self.mgr.evaluate_reentry(snap1)
        self.assertFalse(dec1.should_enter)
        self.assertIn("post_stop_stabilization_cooldown", dec1.reason)

        # Tick 2 at t=now+120s (< 300s cooldown): premature bounce to 92.0 (+2.2%) -> still in cooldown!
        snap2 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=92.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            last_sell_time_s=self.now,
            current_time_s=self.now + 120.0,
            trend_up=True,
        )
        dec2 = self.mgr.evaluate_reentry(snap2)
        self.assertFalse(dec2.should_enter)
        self.assertIn("post_stop_stabilization_cooldown", dec2.reason)

        # Tick 3 at t=now+180s (< 300s cooldown): deeper leg down to 88.0 -> updates trough to 88.0
        snap3 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=88.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            last_sell_time_s=self.now,
            current_time_s=self.now + 180.0,
            trend_up=True,
        )
        dec3 = self.mgr.evaluate_reentry(snap3)
        self.assertFalse(dec3.should_enter)
        self.assertEqual(self.mgr.reentry_state["BTCUSDT"]["trough"], 88.0)

        # Tick 4 at t=now+310s (>= 300s cooldown): bounce to 89.5 (+1.70% from 88.0 >= 1.5%) -> enters!
        snap4 = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=89.5,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            last_sell_time_s=self.now,
            current_time_s=self.now + 310.0,
            trend_up=True,
        )
        dec4 = self.mgr.evaluate_reentry(snap4)
        self.assertTrue(dec4.should_enter)
        self.assertIn("post_stop_bounce_confirmed", dec4.reason)

    def test_hard_tp_triggers_on_first_hit_when_now_below_cooldown_and_within_fuzzy_tolerance(self):
        self.mgr.config.hard_tp_enabled = True
        self.mgr.config.hard_tp_pct = 0.20
        self.mgr.config.hard_tp_frac = 0.50
        self.mgr.config.hard_tp_cooldown_sec = 21600.0  # 6 hours
        self.mgr.config.are_close_tolerance_pct = 0.10  # 0.10% fuzzy boundary
        self.mgr.hard_tp_last.clear()

        # Price is 119.99 vs 100.0 (+19.99% gain, within 0.10% tolerance of 20.0% hard_tp_pct),
        # and current_time_s=100.0 (< 21600.0s cooldown) with no prior Hard-TP (last_hard == 0.0).
        snap = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=119.99,
            avg_buy_price=100.0,
            sellable_qty=2.0,
            current_time_s=100.0,
            trend_up=True,
        )
        dec = self.mgr.evaluate(snap)
        self.assertTrue(dec.should_exit)
        self.assertEqual(dec.action, ExitAction.PARTIAL_SELL)
        self.assertEqual(dec.sell_fraction, 0.50)
        self.assertIn("hard_tp_spike", dec.reason)

    def test_tp_anomaly_pullback_ratchet_tightens_on_out_of_pattern_spike_without_selling_on_climb(self):
        # Normal TP threshold is 7% (0.07), vol_1h is 2% (0.02).
        # Anomaly threshold = max(0.07 * 1.5 = 10.5%, 0.07 + 0.02 * 2.5 = 12.0%) = 12.0%.
        # Tick 1: Price climbs to 108.0 (+8.0% >= 7.0%) -> initiates trend ride (HOLD)
        snap1 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
            vol_1h=0.02,
        )
        dec1 = self.mgr.evaluate_take_profit(snap1)
        self.assertEqual(dec1.action, ExitAction.HOLD)

        # Tick 2: Out-of-pattern spike to 140.0 (+40.0% >> 12.0% anomaly threshold).
        # While still climbing on uptrend (0% pullback), MUST NOT sell against the trend!
        snap2 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=140.0,
            avg_buy_price=100.0,
            current_time_s=self.now + 30.0,
            trend_up=True,
            vol_1h=0.02,
        )
        dec2 = self.mgr.evaluate_take_profit(snap2)
        self.assertEqual(dec2.action, ExitAction.HOLD)
        self.assertFalse(dec2.should_exit)

        # Tick 3: Spike stalls and pulls back 0.35% from 140.0 to 139.51 (< normal 0.60%,
        # but >= ratcheted 0.32% trail for a +40% anomaly spike) -> exits at peak!
        snap3 = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=139.51,
            avg_buy_price=100.0,
            current_time_s=self.now + 45.0,
            trend_up=True,
            vol_1h=0.02,
        )
        dec3 = self.mgr.evaluate_take_profit(snap3)
        self.assertEqual(dec3.action, ExitAction.SELL)
        self.assertTrue(dec3.should_exit)
        self.assertIn("mdhtp_peak_pullback", dec3.reason)

    def test_corrupted_state_and_strategy_helpers_safety(self):
        from strategies.rtrade_pair import OrderSnapshot, PairPolicy
        from strategies.spot_engine import Strategy, StratParams, _new_state
        from strategies.spot_replay import _ReplayClient

        # 1. PositionExitManager corrupted state entries
        self.mgr.sl_bounce_state["BTCUSDT"] = "corrupt"
        self.mgr.sl_episode_state["BTCUSDT"] = "corrupt"
        self.mgr.tp_ride_state["BTCUSDT"] = "corrupt"
        self.mgr.reentry_state["BTCUSDT"] = "corrupt"

        snap_sl = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=94.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        dec_sl = self.mgr.evaluate_stop_loss(snap_sl)
        self.assertEqual(dec_sl.action, ExitAction.HOLD)
        self.assertIsInstance(self.mgr.sl_bounce_state.get("BTCUSDT"), dict)

        snap_tp = PositionSnapshot(
            symbol="BTCUSDT",
            current_price=108.0,
            avg_buy_price=100.0,
            current_time_s=self.now,
            trend_up=True,
        )
        dec_tp = self.mgr.evaluate_take_profit(snap_tp)
        self.assertEqual(dec_tp.action, ExitAction.HOLD)
        self.assertIsInstance(self.mgr.tp_ride_state.get("BTCUSDT"), dict)

        snap_re = ReentrySnapshot(
            symbol="BTCUSDT",
            current_price=90.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            current_time_s=self.now,
            trend_up=True,
        )
        dec_re = self.mgr.evaluate_reentry(snap_re)
        self.assertEqual(dec_re.action, ReentryAction.WAIT)
        self.assertIsInstance(self.mgr.reentry_state.get("BTCUSDT"), dict)

        # 2. Strategy helpers with corrupted/non-finite inputs
        client = _ReplayClient(1000.0)
        params = StratParams(
            currency="EUR",
            entry_amount=100.0,
            entry_discount_pct=0.5,
            dca_amount=100.0,
            dca_drop_pct=5.0,
            check_minutes=1.0,
            takeprofit_pct=5.0,
            max_budget=500.0,
            max_dca_buys=4,
            enable_takeprofit=True,
            order_ttl_min=60.0,
        )
        strat = Strategy(
            client,
            "REPLAY",
            params,
            dry_run=True,
            initial_state=_new_state(),
            replay_mode=True,
        )
        strat.s["placement_backoffs"] = "corrupt"
        self.assertEqual(strat._placement_backoff_remaining("buy", "ENTRY"), 0.0)
        strat._record_placement_backoff("buy", "ENTRY", "insufficient funds")
        self.assertGreater(strat._placement_backoff_remaining("buy", "ENTRY"), 0.0)
        strat._clear_placement_backoff("buy", "ENTRY")
        self.assertEqual(strat._placement_backoff_remaining("buy", "ENTRY"), 0.0)

        order = strat._order_from_pending({
            "order_id": "ORD-1",
            "side": "BUY",
            "requested_qty": "nan",
            "submitted_qty": "1.5",
            "requested_price": "100.0",
            "metadata": {"amount": "150.0", "market": 1},
            "created_at": "bad_ts",
        })
        self.assertEqual(order["vol"], 1.5)
        self.assertEqual(order["price"], 100.0)
        self.assertEqual(order["amount"], 150.0)
        self.assertTrue(order["market"])

        # _hourly_vol_from_closes with inf/nan/bool closes
        closes_bad = [100.0 + i * 0.1 for i in range(25)] + [float("inf"), float("nan"), True]
        vol_h = Strategy._hourly_vol_from_closes(closes_bad, 60.0)
        self.assertIsNotNone(vol_h)
        self.assertIsNone(Strategy._hourly_vol_from_closes(closes_bad, float("nan")))

        # 3. rtrade_pair OrderSnapshot and PairPolicy normalization
        osnap = OrderSnapshot(status="OPEN", filled_qty="1.25", cost="250.0", fee="0.5")
        self.assertEqual(osnap.status, "open")
        self.assertEqual(osnap.filled_qty, 1.25)
        self.assertEqual(osnap.cost, 250.0)
        self.assertEqual(osnap.fee, 0.5)
        policy = PairPolicy(adjustment_fraction="0.02", quote_ttl_sec="30.0")
        self.assertEqual(policy.adjustment_fraction, 0.02)
        self.assertEqual(policy.quote_ttl_sec, 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_positionexitmanager_preserves_episode_started_at_across_trend_up_flips_prevents_smart_iter8_s2():
    """[ITER8 #2] PositionExitManager preserves episode_started_at across trend_up flips & prevents Smart-SL re-arm."""
    # 2. PositionExitManager preserves episode_started_at across trend_up flips & prevents Smart-SL re-arm
    smart_calls = []

    def fake_smart_sl(**kw):
        smart_calls.append(kw["now"])
        if len(smart_calls) == 1:
            return False, "grace_active", {}
        return True, "grace_period_expired (120s elapsed without bounce)", {}

    mgr = PositionExitManager(
        config=PositionExitConfig(lost_threshold=0.05, sl_trend_defer_max_sec=300.0, smart_sl_enabled=True),
        smart_sl_evaluator=fake_smart_sl,
    )
    # t=1000: trend_up=True -> initiates bounce watch at started_at=1000
    d1 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1000.0, trend_up=True)
    )
    assert d1.should_exit is False
    assert mgr.sl_bounce_state["TAOUSDC"]["started_at"] == 1000.0

    # t=1100: trend_up=False -> Smart-SL defers (grace_active), sl_bounce_state popped
    d2 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1100.0, trend_up=False)
    )
    assert d2.should_exit is False
    assert "TAOUSDC" not in mgr.sl_bounce_state

    # t=1150: trend_up=True again -> restores started_at=1000.0 from sl_episode_state (NOT 1150.0!)
    d3 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1150.0, trend_up=True)
    )
    assert d3.should_exit is False
    assert mgr.sl_bounce_state["TAOUSDC"]["started_at"] == 1000.0

    # t=1220: trend_up=False -> Smart-SL expires (grace_period_expired) -> SELL & marks smart_sl_done
    d4 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1220.0, trend_up=False)
    )
    assert d4.should_exit is True
    assert mgr.sl_episode_state["TAOUSDC"].get("smart_sl_done") is True

    # t=1225: if order was deferred by GT_S and re-evaluated with trend_up=False, Smart-SL is NOT called again
    d5 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1225.0, trend_up=False)
    )
    assert d5.should_exit is True
    assert len(smart_calls) == 2

    # t=1310 (>= 1000 + 300s): trend_up=True -> immediately triggers mhdsl_bounce_timeout!
    d6 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1310.0, trend_up=True)
    )
    assert d6.should_exit is True
    assert d6.metadata.get("sub_reason") == "timeout"

def test_get_post_stop_reentry_guard_honors_llm_guard_mode_and_intelligence_guards_mode_monitortrades__apply_post_stop_ai_guard__execute_reentry_buy_iter15_s1(monkeypatch, tmp_path):
    """[ITER15 #1] get_post_stop_reentry_guard honors llm_guard_mode and intelligence_guards_mode & monitortrades._apply_post_stop_ai_guard & _execute_reentry_buy preserve HighStakeGuard when PostStopReentryGuard is off/shadow & smart_stop_loss_guard._load_grace_state recovers from .previous and evaluate_smart_stop_loss honors external_guard_mode & external_order_guard caps whale_guard_mode='enforce' to 'shadow' when external_guard_mode='shadow' & AutonomousIntentGateway._get_live_price recovers from .previous and skips 0/NaN price entries."""
    import json
    import time
    import external_order_guard
    import order_guard
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector, WhalePositioningSnapshot
    from intelligence.external.guards import smart_stop_loss_guard as ssl_mod
    from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    # 1. get_post_stop_reentry_guard honors llm_guard_mode and intelligence_guards_mode
    base_m = dict(order_guard._load_margins())
    m_llm_off = {**base_m, "post_stop_reentry_mode": "enforce", "llm_guard_mode": "off"}
    g_off = get_post_stop_reentry_guard(margins=m_llm_off, gemini_client=object())
    assert g_off.mode == "off"

    m_llm_shadow = {**base_m, "post_stop_reentry_mode": "enforce", "llm_guard_mode": "shadow"}
    g_shadow = get_post_stop_reentry_guard(margins=m_llm_shadow, gemini_client=object())
    assert g_shadow.mode == "shadow"


    # 2. monitortrades._apply_post_stop_ai_guard & _execute_reentry_buy preserve HighStakeGuard when PostStopReentryGuard is off/shadow
    monkeypatch.setattr(mt, "_is_simulated_env_or_inst", lambda inst=None: False)
    monkeypatch.setattr(order_guard, "_load_margins", lambda: m_llm_shadow)
    snap = ReentrySnapshot(
        symbol="TAOUSDC",
        current_price=210.0,
        last_sell_price=210.0,
        last_exit_kind="STOP",
        post_sell_peak=210.0,
        post_sell_trough=195.0,
        current_time_s=2000.0,
        last_sell_time_s=1000.0,
        trend_up=True,
        trend_down=False,
    )
    allowed, _, _, _ = mt._apply_post_stop_ai_guard(
        "TAOUSDC", 210.0, 210.0, 1000.0, snap, 1000.0, None, 5.0, inst=None
    )
    assert allowed is True
    assert "TAOUSDC" in mt._POST_STOP_AI_NOT_ENFORCED

    captured_bypasses = []
    monkeypatch.setattr(
        mt,
        "_place_guarded",
        lambda inst, side, px, q, mq, **kw: (captured_bypasses.append(kw.get("bypass_guards")) or True),
    )

    class _DummyInst:
        symbol = "TAOUSDC"

        def param(self, ns, key, default=None, cast=None):
            return default

    mt._execute_reentry_buy(_DummyInst(), "TAOUSDC", 200.0, 5.0, 0.01, 3600.0, "STOP", {"sell_time": 1000.0})
    assert captured_bypasses == [{"G_PROF"}]


    # 3. smart_stop_loss_guard._load_grace_state recovers from .previous and evaluate_smart_stop_loss honors external_guard_mode
    state_file = tmp_path / "smart_stop_loss_state.json"
    prev_file = tmp_path / "smart_stop_loss_state.json.previous"
    state_file.write_text("{corrupt_json", encoding="utf-8")
    now_ts = time.time()
    prev_file.write_text(
        json.dumps({"TAOUSDC": {"start_ts": now_ts - 10.0, "initial_loss_pct": 3.5}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ssl_mod, "_effective_smart_sl_state_file", lambda: str(state_file))
    ssl_mod._GRACE_TRACKER.clear()
    ssl_mod._LOADED_STATE_FILE_KEY = None
    tracker = ssl_mod._load_grace_state(is_sim=False)
    assert "TAOUSDC" in tracker

    m_ext_off = {**base_m, "smart_sl_mode": "enforce", "external_guard_mode": "off"}
    monkeypatch.setattr(order_guard, "_load_margins", lambda: m_ext_off)
    sl_exec, sl_reason, _ = ssl_mod.evaluate_smart_stop_loss("TAOUSDC", 0.04, 0.033, provider="sim")
    assert sl_exec is True
    assert sl_reason == "smart_sl_disabled"


    # 4. external_order_guard caps whale_guard_mode='enforce' to 'shadow' when external_guard_mode='shadow'
    collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
    collector.record_snapshot(
        WhalePositioningSnapshot(
            symbol="TAOUSDC",
            top_traders_long_ratio=2.0,
            top_traders_long_pct=0.66,
            taker_buy_sell_ratio=0.60,
            taker_buy_vol_usd=60_000.0,
            taker_sell_vol_usd=100_000.0,
            open_interest_usd=50_000_000.0,
            open_interest_1h_change_pct=-5.0,
            divergence_regime="long_liquidation_risk",
            ts=now_ts,
        )
    )
    monkeypatch.setattr(external_order_guard, "_whale_collector", collector)
    m_ext_shadow = {
        **base_m,
        "external_guard_mode": "shadow",
        "whale_guard_mode": "enforce",
        "orderbook_wall_guard_mode": "off",
        "funding_guard_mode": "off",
    }
    ext_ok, ext_reason, ext_scale = external_order_guard.check_external_order_guards(
        "TAOUSDC", "BUY", 200.0, margins=m_ext_shadow, now=now_ts
    )
    assert ext_ok is True
    assert ext_scale == 1.0


    # 5. AutonomousIntentGateway._get_live_price recovers from .previous and skips 0/NaN price entries
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    cp_file = cache_dir / "cache_current_price.json"
    cp_prev = cache_dir / "cache_current_price.json.previous"
    cp_file.write_text("{corrupt", encoding="utf-8")
    cp_prev.write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSD": [[now_ts, 0.0]],
                    "TAOUSDC": [[now_ts, 245.5]],
                }
            }
        ),
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path), mode="shadow")

    class _FakeInst:
        symbol = "TAOUSD"
        base = "TAO"
        _provider = None

    assert gw._get_live_price(_FakeInst()) == pytest.approx(245.5)

def test_positionexitmanager__evaluate_stop_loss_core_sets_sl_timer_expired_true_and_prioritizes_timeout_iter16_s2():
    """[ITER16 #2] PositionExitManager._evaluate_stop_loss_core sets sl_timer_expired=True and prioritizes timeout over structural_break."""
    # 2. PositionExitManager._evaluate_stop_loss_core sets sl_timer_expired=True and prioritizes timeout over structural_break
    cfg = PositionExitConfig(
        lost_threshold=0.05,
        sl_trend_defer_max_sec=900.0,
        sl_structural_break_enabled=True,
        smart_sl_enabled=False,
    )
    mgr = PositionExitManager(cfg)
    # Sample 1 at t=2000 initiates bounce watch (5.5% loss >= 5.0% threshold, below 6.5% disaster floor)
    d1 = mgr.evaluate_stop_loss(
        PositionSnapshot(
            symbol="TAOUSDC",
            current_price=94.5,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=2000.0,
            trend_up=True,
        )
    )
    assert d1.should_exit is False
    assert d1.metadata.get("state") == "initiated"

    # Sample 2 at t=2100 records a bounce up to 94.95 (+0.48% above 94.5 trough, still >= 5.0% loss threshold)
    d2 = mgr.evaluate_stop_loss(
        PositionSnapshot(
            symbol="TAOUSDC",
            current_price=94.95,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=2100.0,
            trend_up=True,
        )
    )
    assert d2.should_exit is False

    # Sample 3 at t=2950 (950s >= 900s max defer) with price 94.2 (< lowest_px 94.5, which would also be structural_break):
    # timeout must take precedence over structural_break and set sl_timer_expired=True
    d3 = mgr.evaluate_stop_loss(
        PositionSnapshot(
            symbol="TAOUSDC",
            current_price=94.2,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=2950.0,
            trend_up=True,
        )
    )
    assert d3.should_exit is True
    assert d3.metadata.get("sl_timer_expired") is True
    assert d3.metadata.get("sub_reason") == "timeout"

def test_positionexitmanager_is_valid_price_rejects_inf_nan_and_re_anchors_iter20_s3():
    """[ITER20 #3] PositionExitManager is_valid_price rejects Inf/NaN and re-anchors future SL/TP/Re-entry timestamps."""
    # 3. PositionExitManager is_valid_price rejects Inf/NaN and re-anchors future SL/TP/Re-entry timestamps
    snap_inf = PositionSnapshot(symbol="BTCUSDC", current_price=float("inf"), avg_buy_price=100.0)
    assert snap_inf.is_valid_price is False

    mgr = PositionExitManager(
        config=PositionExitConfig(
            lost_threshold=0.05,
            sl_trend_defer_max_sec=200.0,
            gain_threshold=0.05,
            tp_stagnation_max_sec=100.0,
            hard_tp_enabled=True,
            hard_tp_pct=0.10,
            hard_tp_frac=0.5,
            hard_tp_cooldown_sec=300.0,
            reentry_sl_bounce_pct=0.02,
            reentry_post_stop_cooldown_sec=60.0,
        )
    )
    # Seed future started_at (clock skew) on SL episode & bounce state
    mgr.sl_episode_state["BTCUSDC"] = {"started_at": 5000.0, "buy_price": 100.0, "disaster_floor": 0.08}
    mgr.sl_bounce_state["BTCUSDC"] = {
        "started_at": 5000.0,
        "entry_loss": 0.06,
        "lowest_price": 94.0,
        "peak_bounce_price": 94.0,
        "min_loss": 0.06,
        "buy_price": 100.0,
    }
    d_sl1 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1000.0, trend_up=True)
    )
    assert d_sl1.should_exit is False
    # Re-anchored to 1000.0 instead of staying at 5000.0
    assert mgr.sl_episode_state["BTCUSDC"]["started_at"] == 1000.0
    assert mgr.sl_bounce_state["BTCUSDC"]["started_at"] == 1000.0
    # 200s later (t=1200.0), SL bounce timeout fires on schedule!
    d_sl2 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1200.0, trend_up=True)
    )
    assert d_sl2.should_exit is True
    assert d_sl2.action == ExitAction.SELL
    assert d_sl2.metadata.get("sl_timer_expired") is True

    # Future hard_tp_last is cleared on clock step-back so Hard-TP spike is not blocked for hours
    mgr.hard_tp_last["BTCUSDC"] = 9000.0
    d_tp_hard = mgr.evaluate_take_profit(
        PositionSnapshot(symbol="BTCUSDC", current_price=112.0, avg_buy_price=100.0, current_time_s=1000.0, trend_up=True)
    )
    assert d_tp_hard.should_exit is True
    assert d_tp_hard.action == ExitAction.PARTIAL_SELL

    # Future last_sell_time_s on STOP re-entry does not permanently lock post_stop_stabilization_cooldown
    d_re = mgr.evaluate_reentry(
        ReentrySnapshot(
            symbol="BTCUSDC",
            current_price=98.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            post_sell_trough=94.0,
            current_time_s=1000.0,
            last_sell_time_s=9000.0,
            trend_up=True,
        )
    )
    assert d_re.should_enter is True
    assert d_re.action == ReentryAction.ENTER

def test_positionsnapshot_reentrysnapshot_reject_bool_none_prices_positionexitmanager_disaster_iter32_s2():
    """[ITER32 #2] PositionSnapshot & ReentrySnapshot reject bool/None prices & PositionExitManager disaster buffer fallback on NaN vol_1h & reentry with NaN trough."""
    # 2. PositionSnapshot & ReentrySnapshot reject bool/None prices
    snap_bool = PositionSnapshot(symbol="BTCUSDC", current_price=True, avg_buy_price=85000.0)
    assert snap_bool.is_valid_price is False
    snap_none = PositionSnapshot(symbol="BTCUSDC", current_price=None, avg_buy_price=85000.0)
    assert snap_none.is_valid_price is False
    re_bool = ReentrySnapshot(symbol="BTCUSDC", current_price=True, last_sell_price=85000.0)
    assert re_bool.is_valid_price is False


    # 3. PositionExitManager disaster buffer fallback on NaN vol_1h & reentry with NaN trough
    mgr = PositionExitManager(
        config=PositionExitConfig(
            lost_threshold=0.05,
            sl_disaster_buffer_ratio=0.30,
            sl_disaster_vol_multiplier=1.0,
            sl_disaster_buffer_min_pct=0.008,
            reentry_sl_bounce_pct=0.02,
            allow_external_cache=False,
        ),
        vol_resolver=lambda sym: (0.04, "custom_resolver"),
    )
    buf, meta = mgr.compute_disaster_buffer("BTCUSDC", 0.05, vol_1h=float("nan"))
    assert math.isclose(buf, 0.04) and meta["vol_source"] == "custom_resolver"

    # evaluate_stop_loss with current_price=True must HOLD ("invalid_prices"), not SELL
    sl_dec = mgr.evaluate_stop_loss(snap_bool)
    assert sl_dec.action == ExitAction.HOLD and sl_dec.should_exit is False

    # evaluate_reentry with post_sell_trough=NaN and last_sell_time_s=NaN must not poison trough with NaN
    re_snap = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=103.0,
        last_sell_price=105.0,
        last_exit_kind="STOP",
        post_sell_trough=float("nan"),
        post_sell_peak=float("nan"),
        current_time_s=float("nan"),
        last_sell_time_s=float("nan"),
    )
    # First tick at 100.0 sets trough=100.0 even when post_sell_trough=NaN
    re_snap_dip = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=100.0,
        last_sell_price=105.0,
        last_exit_kind="STOP",
        post_sell_trough=float("nan"),
        post_sell_peak=float("nan"),
        current_time_s=float("nan"),
        last_sell_time_s=float("nan"),
    )
    d_wait = mgr.evaluate_reentry(re_snap_dip)
    assert d_wait.action == ReentryAction.WAIT
    assert math.isfinite(mgr.reentry_state["BTCUSDC"]["trough"])
    # Second tick at 103.0 (+3% bounce >= 2% required) confirms re-entry without NaN sell_time resetting state
    d_enter = mgr.evaluate_reentry(re_snap)
    assert d_enter.action == ReentryAction.ENTER and d_enter.should_enter is True

def test_positionexitmanager_preserves_started_at_across_subsequent_ticks_without_calling_iter33_s2(monkeypatch):
    """[ITER33 #2] PositionExitManager preserves started_at=0.0 across subsequent ticks without calling time.time()."""
    # 2. PositionExitManager preserves started_at=0.0 across subsequent ticks without calling time.time()
    cfg = PositionExitConfig(
        lost_threshold=0.05,
        gain_threshold=0.05,
        sl_trend_defer_max_sec=60.0,
        tp_stagnation_max_sec=60.0,
        allow_external_cache=False,
    )
    mgr = PositionExitManager(config=cfg)
    snap_t0 = PositionSnapshot(
        symbol="BTCUSDC",
        current_price=94.0,
        avg_buy_price=100.0,
        net_qty=1.0,
        sellable_qty=1.0,
        current_time_s=0.0,
        trend_up=True,
        vol_1h=0.02,
        provider="test",
    )
    d0 = mgr.evaluate_stop_loss(snap_t0)
    assert d0.should_exit is False
    assert mgr.sl_bounce_state["BTCUSDC"]["started_at"] == 0.0

    snap_t65 = PositionSnapshot(
        symbol="BTCUSDC",
        current_price=94.0,
        avg_buy_price=100.0,
        net_qty=1.0,
        sellable_qty=1.0,
        current_time_s=65.0,
        trend_up=True,
        vol_1h=0.02,
        provider="test",
    )
    d65 = mgr.evaluate_stop_loss(snap_t65)
    assert d65.should_exit is True and "mhdsl_bounce_timeout" in d65.reason

    # Restore time.time for subsequent checks
    monkeypatch.undo()

def test_monitortrades__apply_post_stop_ai_guard_sanitizes_post_sell_trough_nan_iter33_s3(monkeypatch):
    """[ITER33 #3] monitortrades._apply_post_stop_ai_guard sanitizes post_sell_trough=NaN."""
    from types import SimpleNamespace
    # 3. monitortrades._apply_post_stop_ai_guard sanitizes post_sell_trough=NaN
    captured_trough = []

    class _DummyPSGuard:
        mode = "enforce"

        def check(self, **kwargs):
            captured_trough.append(kwargs.get("trough"))
            return SimpleNamespace(allowed=True, scale=1.0, reason="ok")

    import intelligence.sentiment.guards.post_stop_reentry_guard as ps_mod
    import intelligence.sentiment.guards.high_stake_guard as hs_mod
    monkeypatch.setattr(ps_mod, "get_post_stop_reentry_guard", lambda: _DummyPSGuard())
    monkeypatch.setattr(hs_mod, "is_llm_guard_exempt", lambda: False)
    monkeypatch.setattr(mt, "_is_simulated_env_or_inst", lambda inst: False)

    re_snap = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=105.0,
        last_sell_price=105.0,
        last_exit_kind="STOP",
        post_sell_trough=float("nan"),
        post_sell_peak=float("nan"),
    )
    allowed, _, _, _ = mt._apply_post_stop_ai_guard(
        symbol="BTCUSDC",
        current_price=105.0,
        sell_price=105.0,
        sell_time=1700000000.0,
        reentry_snap=re_snap,
        buy_budget=100.0,
        buy_qty_cfg=None,
        qty=1.0,
        inst=SimpleNamespace(provider_name="binance", params={"bypass_guards": []}),
    )
    assert allowed is True
    assert len(captured_trough) == 1 and math.isfinite(captured_trough[0]) and captured_trough[0] == 105.0

def test_positionexitmanager_evaluate_reentry_honors_lower_external_post_sell_trough_and_higher_iter121_s1():
    """[ITER121 #1] PositionExitManager.evaluate_reentry honors lower external post_sell_trough and higher post_sell_peak across ticks, plus lowercase 'stop'."""
    # 1. PositionExitManager.evaluate_reentry honors lower external post_sell_trough and higher post_sell_peak across ticks, plus lowercase "stop"
    pem = PositionExitManager(
        PositionExitConfig(
            reentry_sl_bounce_pct=0.02,
            reentry_pullback_pct=0.02,
            reentry_hybrid_enabled=True,
            reentry_post_stop_cooldown_sec=0.0,
            reentry_tolerance_pct=0.0,
        )
    )
    # Tick 1 (post-STOP with lowercase " stop "): trough=98.0 -> needs 99.96
    d1 = pem.evaluate_reentry(
        ReentrySnapshot(
            symbol="SOLUSDC",
            current_price=98.0,
            last_sell_price=100.0,
            last_exit_kind=" stop ",
            post_sell_trough=98.0,
            current_time_s=1000.0,
            last_sell_time_s=900.0,
        )
    )
    assert not d1.should_enter and "post_stop_bounce_waiting" in d1.reason
    # Tick 2: external post_sell_trough dropped to 94.0 while current_price recovered to 96.0 (+2.13% >= 2.0%)
    d2 = pem.evaluate_reentry(
        ReentrySnapshot(
            symbol="SOLUSDC",
            current_price=96.0,
            last_sell_price=100.0,
            last_exit_kind=" stop ",
            post_sell_trough=94.0,
            current_time_s=1060.0,
            last_sell_time_s=900.0,
        )
    )
    assert d2.should_enter and "post_stop_bounce_confirmed" in d2.reason

    # Post-TP bull pullback: Tick 1 peak=105.0, Tick 2 external post_sell_peak=130.0 with current_price=127.0 (-2.31% >= 2.0%)
    d3 = pem.evaluate_reentry(
        ReentrySnapshot(
            symbol="ETHUSDC",
            current_price=105.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            post_sell_peak=105.0,
            trend_up=True,
            current_time_s=1000.0,
            last_sell_time_s=900.0,
        )
    )
    assert not d3.should_enter and "bull_reentry_pullback_waiting" in d3.reason
    d4 = pem.evaluate_reentry(
        ReentrySnapshot(
            symbol="ETHUSDC",
            current_price=127.0,
            last_sell_price=100.0,
            last_exit_kind="TP",
            post_sell_peak=130.0,
            trend_up=True,
            current_time_s=1060.0,
            last_sell_time_s=900.0,
        )
    )
    assert d4.should_enter and "bull_reentry_pullback_confirmed" in d4.reason

def test_all_str_enum_classes_return_raw_value_on_iter125_s1():
    """[ITER125 #1] All (str, Enum) classes return raw value on str() in Python 3.12+."""
    from intelligence.internal.guards.guard_decision import BrakeAction
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerSide
    from market_regime import MarketRegimeHorizon
    # 1. All (str, Enum) classes return raw value on str() in Python 3.12+
    assert str(TriggerAction.ENTRY) == "ENTRY"
    assert str(TriggerSide.BUY) == "BUY"
    assert str(BrakeAction.HARD_VETO) == "HARD_VETO"
    assert str(ExitAction.SELL) == "SELL"
    assert str(ReentryAction.ENTER) == "ENTER"
    assert str(MarketRegimeHorizon.SHORT) == "short"

def test_positionsnapshot_reentrysnapshot_normalize_string_nan_boolean_flags_and_iter146_s1():
    """[ITER146 #1] PositionSnapshot & ReentrySnapshot normalize string/NaN boolean flags and guard tol_pct/threshold."""
    # 1. PositionSnapshot & ReentrySnapshot normalize string/NaN boolean flags and guard tol_pct/threshold
    ps = PositionSnapshot(symbol="BTCUSDC", current_price="95.0", avg_buy_price="100.0", trend_up="false")  # type: ignore[arg-type]
    assert ps.trend_up is False
    assert ps.in_sl_territory(0.05, tol_pct=float("nan")) is True
    assert ps.in_sl_territory(float("nan"), tol_pct=1.0) is False
    assert ps.in_tp_territory(True, tol_pct=1.0) is False  # type: ignore[arg-type]

    rs = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price="98.0",
        last_sell_price="100.0",
        trend_up="0",  # type: ignore[arg-type]
        trend_down=float("nan"),  # type: ignore[arg-type]
    )
    assert rs.trend_up is False
    assert rs.trend_down is False
    assert rs.has_dropped(100.0, 0.02, tol_pct=True) is True  # type: ignore[arg-type]

    mgr = PositionExitManager(PositionExitConfig(smart_sl_enabled=True, allow_external_cache=False))
    mgr.sl_episode_state["BTCUSDC"] = {"started_at": 1000.0, "buy_price": 100.0, "disaster_floor": 0.08, "smart_sl_done": "false"}
    called_smart = []
    mgr.smart_sl_evaluator = lambda **kw: (called_smart.append(True) or False, "deferred_test", {})
    dec = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1010.0, trend_up="false")  # type: ignore[arg-type]
    )
    assert called_smart == [True]
    assert dec.should_exit is False

def test__parse_bool_flag_preserves_default_true_on_empty_null_strings_iter173_s1():
    """[ITER173 #1] _parse_bool_flag preserves default=True on empty/null strings across modules."""
    from strategies.position_exit_manager import _parse_bool_flag as pem_bool
    from strategies.spot_engine import _parse_bool_flag as se_bool
    # 1. _parse_bool_flag preserves default=True on empty/null strings across modules
    assert pem_bool("", True) is True
    assert pem_bool("none", True) is True
    assert pem_bool("false", True) is False
    assert se_bool("", True) is True
    assert se_bool("null", True) is True
    assert se_bool("0", True) is False
    assert mt._parse_bool_flag("", True) is True
    assert mt._parse_bool_flag("y", False) is True
    assert mt._parse_bool_flag("n", True) is False

def test_positionexitconfig_preserves_default_true_on_empty_string_and_iter173_s2():
    """[ITER173 #2] PositionExitConfig preserves default=True on empty string, and PositionSnapshot/ReentrySnapshot accept Decimal."""
    from decimal import Decimal
    # 2. PositionExitConfig preserves default=True on empty string, and PositionSnapshot/ReentrySnapshot accept Decimal
    cfg = PositionExitConfig(smart_sl_enabled="", sl_bounce_ratchet_enabled="")
    assert cfg.smart_sl_enabled is True
    assert cfg.sl_bounce_ratchet_enabled is True

    ps = PositionSnapshot(symbol=None, current_price=Decimal("105.5"), avg_buy_price=Decimal("100.0"))
    assert ps.symbol == ""
    assert isinstance(ps.current_price, float) and ps.current_price == pytest.approx(105.5)
    assert isinstance(ps.avg_buy_price, float) and ps.avg_buy_price == pytest.approx(100.0)

    rs = ReentrySnapshot(
        symbol=True,
        current_price=Decimal("95.0"),
        last_sell_price=Decimal("100.0"),
        post_sell_peak=Decimal("101.0"),
        post_sell_trough=Decimal("94.0"),
    )
    assert rs.symbol == ""
    assert isinstance(rs.last_sell_price, float) and rs.last_sell_price == pytest.approx(100.0)

def test_positionexitmanager_symbol_whitespace_normalization_in_restore_reentry_state_record_hard_tp_iter173_s3():
    """[ITER173 #3] PositionExitManager symbol whitespace normalization in restore_reentry_state & record_hard_tp."""
    # 3. PositionExitManager symbol whitespace normalization in restore_reentry_state & record_hard_tp
    mgr = PositionExitManager()
    mgr.reentry_state["BTCUSDC"] = {"trough": 90.0}
    mgr._pop_reentry_state(" BTCUSDC ")
    assert "BTCUSDC" not in mgr.reentry_state
    mgr.restore_reentry_state(" BTCUSDC ")
    assert mgr.reentry_state.get("BTCUSDC") == {"trough": 90.0}
    mgr.record_hard_tp(" BTCUSDC ", 1234.0)
    assert mgr.hard_tp_last.get("BTCUSDC") == pytest.approx(1234.0)
