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


if __name__ == "__main__":
    unittest.main(verbosity=2)


