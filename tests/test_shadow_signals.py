"""
Tests for shadow_signals.py — in particular the behaviour on a data GAP
in KalmanTrend.update(), fixed 21 Jul: a long gap (network down, process
stopped) no longer propagates the old velocity (capped at DT_MAX) but resets
the filter (as at warm-up) — the trend comes out FLAT after a gap, not "confident" in a
stale direction.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shadow_signals as ss


class TestKalmanTrendContinuous(unittest.TestCase):
    """Baseline behaviour (unchanged) on a series without gaps: a clear UP trend."""

    def test_continuous_series_direction(self):
        for label, delta, expected in (("up", 0.05, 1), ("flat", 0.0, 0)):
            with self.subTest(case=label):
                kf = ss.KalmanTrend()
                ts = 1_000_000.0
                price = 100.0
                out = None
                for _ in range(30):
                    ts += 1.0
                    price += delta
                    out = kf.update(ts, price, epsilon=0.01)
                self.assertEqual(out["trend"], expected)


class TestKalmanTrendGapReset(unittest.TestCase):
    """21 Jul: a gap > GAP_RESET_SEC (300s) => reset, not merely a capped dt."""

    def _warm_up_uptrend(self, kf, ts, price, steps=30):
        for _ in range(steps):
            ts += 1.0
            price += 0.05
            out = kf.update(ts, price, epsilon=0.01)
        return ts, price, out

    def test_long_gap_resets_velocity_to_zero(self):
        kf = ss.KalmanTrend()
        ts, price, out = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        self.assertEqual(out["trend"], 1, "precondition: an UP trend established before the gap")

        # a 1h gap (> GAP_RESET_SEC=300s), the price returns unchanged from the last known one
        ts_after_gap = ts + 3600.0
        out_after = kf.update(ts_after_gap, price, epsilon=0.01)

        self.assertEqual(out_after["vel"], 0.0, "the old velocity must NOT be propagated across a long gap")
        self.assertEqual(out_after["trend"], 0, "after the reset the trend comes out FLAT, not a stale UP")

    def test_short_gap_under_threshold_is_not_reset(self):
        """A 30s gap (under GAP_RESET_SEC) must be handled normally (real dt), not as a reset."""
        kf = ss.KalmanTrend()
        ts, price, out = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        self.assertEqual(out["trend"], 1)

        ts_after_gap = ts + 30.0
        price += 0.05 * 30       # the UP trend continues past the gap
        out_after = kf.update(ts_after_gap, price, epsilon=0.01)

        self.assertNotEqual(out_after["vel"], 0.0, "a short gap must not reset the velocity to 0")
        self.assertEqual(out_after["trend"], 1, "the UP trend must survive a short gap")

    def test_gap_reset_does_not_crash_on_first_ever_update(self):
        """The first observation (self.x is None) must stay unaffected by the new gap branch."""
        kf = ss.KalmanTrend()
        out = kf.update(1_000_000.0, 100.0, epsilon=0.01)
        self.assertEqual(out["trend"], 0)
        self.assertEqual(out["vel"], 0.0)

    def test_gap_boundary_exactly_at_threshold_not_reset(self):
        """At exactly GAP_RESET_SEC the reset branch (strict >) must not fire."""
        kf = ss.KalmanTrend()
        ts, price, _ = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        out_after = kf.update(ts + ss.GAP_RESET_SEC, price + 0.05, epsilon=0.01)
        # real dt = exactly GAP_RESET_SEC => not strictly greater => normal path (no reset to 0)
        self.assertNotEqual(out_after["vel"], 0.0)

    def test_long_gap_resets_trend_even_when_conf_exit_is_zero(self):
        """Even if CONF_EXIT is 0.0, a gap > GAP_RESET_SEC must reset trend to 0."""
        from unittest.mock import patch
        import intelligence.internal.triggers.kalman_trigger as kt
        kf = ss.KalmanTrend()
        ts, price, out = self._warm_up_uptrend(kf, 1_000_000.0, 100.0)
        self.assertEqual(out["trend"], 1)
        with patch.object(kt, "CONF_EXIT", 0.0):
            out_after = kf.update(ts + 3600.0, price, epsilon=0.01)
        self.assertEqual(out_after["trend"], 0)
        self.assertEqual(out_after["old_trend"], 1)


class TestVolAndThresholds(unittest.TestCase):
    """Minimal sanity check on the rest of the module — unchanged by the gap fix."""

    def test_warmup_outputs_are_unavailable(self):
        self.assertIsNone(ss.vol_1h_pct([100.0] * 5, sample_rate_sec=1.0))
        reentry, dca = ss.adaptive_thresholds(None)
        self.assertIsNone(reentry)
        self.assertIsNone(dca)




class TestCurrentTrendClock(unittest.TestCase):
    """current_trend() ages the state on the injected clock (live: wall time)."""

    def test_age_uses_injected_clock_and_defaults_to_wall_time(self):
        journal = ss.KalmanJournal(fixed_path=os.devnull)
        simulated = ss.KalmanStateTracker(journal=journal, now_fn=lambda: 1_000_060.0)
        simulated._state["BTCUSDC"] = {"kalman_trend": -1, "ts": 1_000_000.0}
        self.assertEqual(simulated.current_trend("BTCUSDC"), (-1, 60.0))

        wall = ss.KalmanStateTracker(journal=journal)
        wall._state["BTCUSDC"] = {"kalman_trend": 1, "ts": 1_000_000.0}
        trend, age = wall.current_trend("BTCUSDC")
        self.assertEqual(trend, 1)
        self.assertGreater(age, 1e8)


class TestShadowRuntime(unittest.TestCase):
    """Runtime setup idempotency and interval validation."""

    def test_prepare_shadow_runtime_is_idempotent(self):
        from shadow_runtime import prepare_shadow_runtime
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_val = os.environ.get("BINANCE_AUTO_START_WEBSOCKETS")
            try:
                if "BINANCE_AUTO_START_WEBSOCKETS" in os.environ:
                    del os.environ["BINANCE_AUTO_START_WEBSOCKETS"]
                prepare_shadow_runtime(tmp_dir)
                prepare_shadow_runtime(tmp_dir)
                self.assertEqual(os.environ["BINANCE_AUTO_START_WEBSOCKETS"], "0")
                self.assertEqual(sys.path.count(tmp_dir), 1)
            finally:
                if old_val is not None:
                    os.environ["BINANCE_AUTO_START_WEBSOCKETS"] = old_val
                elif "BINANCE_AUTO_START_WEBSOCKETS" in os.environ:
                    del os.environ["BINANCE_AUTO_START_WEBSOCKETS"]
                if tmp_dir in sys.path:
                    sys.path.remove(tmp_dir)

    def test_shadow_interval_validation(self):
        from shadow_runtime import require_shadow_interval
        require_shadow_interval(240, 240, "HL shadow")
        with self.assertRaisesRegex(ValueError, "240m"):
            require_shadow_interval(60, 240, "HL shadow")

    def test_kalman_60s_step_noise_transitions_after_60_samples(self):
        """At 60s sampling with adaptive step-noise epsilon, KalmanTrend must not lock at 0 after 60 samples."""
        import numpy as np
        kf = ss.KalmanTrend()
        fed = []
        ts = 1_000_000.0
        price = 74000.0
        # 60 minutes of mild chop
        for i in range(60):
            ts += 60.0
            price += (15.0 if i % 2 == 0 else -15.0)
            fed.append(price)
            eps = float(np.std(np.gradient(np.asarray(fed)))) if len(fed) >= 5 else 60.0
            out = kf.update(ts, price, eps)
        self.assertEqual(out["trend"], 0)

        # Breakout (+75 USD/min ~ +0.10%/min for 6 minutes)
        for _ in range(6):
            ts += 60.0
            price += 75.0
            fed.append(price)
            if len(fed) > 60:
                fed.pop(0)
            eps = float(np.std(np.gradient(np.asarray(fed))))
            out = kf.update(ts, price, eps)
        self.assertEqual(out["trend"], 1)

    def test_tradeall_logic_small_acceleration_surge(self):
        """logic_small triggers accel_surge_up when short-term velocity and acceleration are strongly positive."""
        from unittest.mock import patch
        import tradeall as ta
        from pricewindow import PriceWindow

        pw = PriceWindow("BTCUSDC", 220, sample_rate_sec=1.0)
        # Parabolic upward curve over 180s
        px = 74000.0
        for i in range(180):
            t_min = i / 60.0
            px = 74000.0 * (1.0 + 0.0012 * t_min + 0.0018 * (t_min ** 2))
            pw.process_price(px)

        trend_state = ta.TrendState("BTCUSDC", 3600.0, 222.0)
        with patch.object(ta, "track_and_place_order", return_value={"orderId": "1"}) as mock_place:
            ta.logic_small(
                "SMALL", True, "BTCUSDC",
                gradient=1, slope=ta.PRICE_CHANGE_THRESHOLD_EUR + 0.05,
                trend_state=trend_state, current_price=px,
                price_window=pw,
            )
            self.assertEqual(mock_place.call_count, 1)
            self.assertEqual(mock_place.call_args.kwargs.get("reason"), "accel_surge_up")

    def test_tradeall_logic_small_acceleration_plunge(self):
        """logic_small triggers accel_plunge_down when short-term velocity and acceleration are strongly negative."""
        from unittest.mock import patch
        import tradeall as ta
        from pricewindow import PriceWindow

        pw = PriceWindow("TAOUSDC", 220, sample_rate_sec=1.0)
        # Parabolic downward curve over 180s (~280 -> ~275)
        px = 280.0
        for i in range(180):
            t_min = i / 60.0
            px = 280.0 * (1.0 - 0.0012 * t_min - 0.0018 * (t_min ** 2))
            pw.process_price(px)

        trend_state = ta.TrendState("TAOUSDC", 3600.0, 222.0)
        with patch.object(ta, "track_and_place_order", return_value={"orderId": "2"}) as mock_place:
            ta.logic_small(
                "SMALL", True, "TAOUSDC",
                gradient=-1, slope=-(ta.PRICE_CHANGE_THRESHOLD_EUR + 0.05),
                trend_state=trend_state, current_price=px,
                price_window=pw,
            )
            self.assertEqual(mock_place.call_count, 1)
            self.assertEqual(mock_place.call_args.args[0], "SELL")
            self.assertEqual(mock_place.call_args.kwargs.get("reason"), "accel_plunge_down")

    def test_kalman_state_tracker_ports_legacy_shadow_state_to_kalman_state(self):
        """KalmanStateTracker ports legacy shadow_state.json to cache_kalman_trend.json on init and loads state."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = os.path.join(tmp_dir, "shadow_state.json")
            new_path = os.path.join(tmp_dir, "cache_kalman_trend.json")
            payload = {"TAOUSDC": {"kalman_trend": -1, "vol_1h_pct": 1.45, "ts": 1000.0, "price": 275.0}}
            with open(legacy_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            journal = ss.KalmanJournal(fixed_path=os.devnull)
            tracker = ss.KalmanStateTracker(journal=journal, state_path=new_path, now_fn=lambda: 1010.0)
            self.assertFalse(os.path.exists(legacy_path))
            self.assertTrue(os.path.exists(new_path))
            self.assertEqual(tracker.current_trend("TAOUSDC"), (-1, 10.0))

    def test_tape_15m_range_bounds_and_non_crisis_wildlife_attack_filter(self):
        """15m_range includes -15m tape anchor, 3m covers up to 4.999m, and wildlife attack headlines are excluded."""
        from intelligence.macro.lexicon_ssot import (
            filter_valid_severity_keywords,
            match_high_severity_keywords,
        )
        from intelligence.sentiment.evaluators.price_series_evaluator import (
            compute_24h_price_metrics,
        )

        headline = "Mother and daughter both killed in grizzly bear attack, Canadian authorities confirm - The Guardian"
        self.assertEqual(match_high_severity_keywords(headline), ())
        self.assertEqual(filter_valid_severity_keywords(["attack"], title=headline), ())

        now_ms = 1_700_000_000_000
        tao_15m_series = [
            [now_ms - int(15.05 * 60_000), 232.8],
            [now_ms - int(10.0 * 60_000), 233.25],
            [now_ms - int(5.0 * 60_000), 233.4],
            [now_ms - int(3.0 * 60_000), 233.7],
            [now_ms - int(1.0 * 60_000), 233.8],
            [now_ms, 233.95],
        ]
        m24 = compute_24h_price_metrics(tao_15m_series)
        self.assertIn("-15m:232.8", m24.get("recent_tape_15m", []))
        self.assertEqual(m24.get("low_15m"), 232.8)
        self.assertEqual(m24.get("high_15m"), 233.95)

    def test_wave11_sparse_horizon_bounds_7d_chg_clamp_and_5m_bull_flow_reversal(self):
        """1h does not steal a 3.7h bar on sparse series, recalculate_live_price_geometry clamps chg_7d_pct >= dd_7d_high_pct, and 5m selloff triggers LIVE_SELL_REVERSAL."""
        from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
            enrich_sentiment_advisor_live_flow_divergence,
        )
        from intelligence.sentiment.evaluators.price_series_evaluator import (
            compute_24h_price_metrics,
            recalculate_live_price_geometry,
        )

        now_ms = 1_793_824_700_000
        rgnt_series = [
            [now_ms - int((1400 - i * 60) * 60_000), 1.40] for i in range(18)
        ] + [
            [now_ms - int(223.0 * 60_000), 1.37],
            [now_ms - int(51.0 * 60_000), 1.375],
            [now_ms - int(46.0 * 60_000), 1.37],
            [now_ms - int(26.0 * 60_000), 1.36],
            [now_ms - int(16.0 * 60_000), 1.2635],
            [now_ms - int(11.0 * 60_000), 1.355],
            [now_ms, 1.35],
        ]
        m24_rgnt = compute_24h_price_metrics(rgnt_series)
        self.assertIn("chg_30m_pct", m24_rgnt)
        self.assertNotIn("chg_1h_pct", m24_rgnt)
        self.assertIn("chg_3h_pct", m24_rgnt)
        self.assertEqual(m24_rgnt.get("recent_tape_15m"), ["-15m:1.2635", "-10m:1.355", "now:1.35"])
        self.assertEqual(m24_rgnt.get("low_15m"), 1.2635)
        self.assertEqual(m24_rgnt.get("high_15m"), 1.355)

        pinfo_7d = {"low_7d": 129.79, "high_7d": 145.23, "chg_7d_pct": -7.70}
        recalculate_live_price_geometry(pinfo_7d, 134.09)
        self.assertEqual(pinfo_7d["dd_7d_high_pct"], -7.67)
        self.assertEqual(pinfo_7d["chg_7d_pct"], -7.67)

        sa_bull = {
            "directional_bias": "BULLISH",
            "recommended_action": "ACCUMULATE",
            "age_min": 18.7,
            "summary": "BTC holding $80,000 with taker buyers regaining control (1.064 taker buy/sell ratio, positive 5m CVD of +$0.82M).",
        }
        enrich_sentiment_advisor_live_flow_divergence(
            sa_bull,
            {
                "BTCUSDC": {
                    "taker_buy_sell_ratio": 0.9027,
                    "cvd_15m_usd": -2_940_000.0,
                    "taker_ratio_5m": 0.40,
                    "cvd_5m_usd": -6_610_000.0,
                    "taker_ratio_1h": 1.03,
                    "cvd_1h_usd": 8_630_000.0,
                }
            },
        )
        self.assertIn("LIVE_SELL_REVERSAL", sa_bull.get("live_flow_divergence", ""))


import contextlib
import dataclasses
import importlib.util
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

_KRAKEN_DIR = Path(__file__).resolve().parents[1] / "kraken"
if str(_KRAKEN_DIR) not in sys.path:
    sys.path.insert(0, str(_KRAKEN_DIR))
_SHADOW_SPEC = importlib.util.spec_from_file_location("kraken_shadow_live_under_test", _KRAKEN_DIR / "shadow_live.py")
shadow = importlib.util.module_from_spec(_SHADOW_SPEC)
_SHADOW_SPEC.loader.exec_module(shadow)


class ShadowLiveTest(unittest.TestCase):
    def test_decision_distance_counts_changed_order_events(self):
        current = [
            {"bar": 1, "side": "buy", "kind": "ENTRY"},
            {"bar": 4, "side": "buy", "kind": "DCA"},
        ]
        candidate = [
            {"bar": 1, "side": "buy", "kind": "ENTRY"},
            {"bar": 5, "side": "buy", "kind": "DCA"},
            {"bar": 9, "side": "sell", "kind": "TP"},
        ]
        self.assertEqual(shadow._decision_distance(current, current), 0)
        self.assertEqual(shadow._decision_distance(current, candidate), 2)

    @staticmethod
    def _live_params():
        @dataclasses.dataclass(frozen=True)
        class Params:
            takeprofit_pct: float = 4.0
            dca_drop_pct: float = 1.25
            dca_spacing_growth_pct: float = 0.25
            reentry_drop_pct: float = 2.2
            stop_loss_pct: float = 18.0
            tp_trail_profit_floor_pct: float = 1.0
            dca_vol_scale_k: float = 0.0
            dca_vol_ref: float = 2.0
            dca_vol_interval: int = 240
            dca_trend_brake: bool = False
            dca_brake_min_pct: float = 1.5
            tp_trend_hold: bool = True
            tp_trail_adaptive: bool = True
            tp_trail_vol_interval: int = 240
            trend_interval: int = 240
            trend_overlay: bool = False
            trend_topup: float = 2000.0
            trend_trail_pct: float = 5.0
            trend_exit_break: bool = False
            tp_regime_gate: bool = True
            surge_guard: bool = True
            surge_gain_pct: float = 18.0
            surge_exit_pullback_pct: float = 3.5
            surge_move_pct: float = 18.0
            surge_dynamic: bool = False

        return Params()

    def test_variants_isolate_each_promoted_change_against_live(self):
        live = self._live_params()
        with patch("strategies.spot_engine.StratParams.from_env", return_value=live), patch.object(shadow, "_load_runtime_config"):
            variants = shadow._variants(240)

        self.assertEqual(
            list(variants),
            [
                "current", "pre0923", "rev_tp5", "rev_spacing0", "rev_trail_fixed",
                "rev_gate_off", "rev_floor0", "rev_sl125", "dca15", "reentry4",
                "dca_vol_m1", "overlay650t8_regime_v2", "B_dcabrake_regime_v2",
                "overlay_safe_combo", "fixed_surge20_pb3",
            ],
        )
        self.assertIs(variants["current"], live)
        for name, params in variants.items():
            if name != "current":
                self.assertNotEqual(params, live, name)
        reverted = {
            "rev_tp5": ("takeprofit_pct", 5.0),
            "rev_spacing0": ("dca_spacing_growth_pct", 0.0),
            "rev_trail_fixed": ("tp_trail_adaptive", False),
            "rev_gate_off": ("tp_regime_gate", False),
            "rev_floor0": ("tp_trail_profit_floor_pct", 0.0),
            "rev_sl125": ("stop_loss_pct", 12.5),
        }
        for name, (field, value) in reverted.items():
            params = variants[name]
            self.assertEqual(getattr(params, field), value, name)
            changed = [f.name for f in dataclasses.fields(params) if getattr(params, f.name) != getattr(live, f.name)]
            self.assertEqual(changed, [field], name)
        pre = variants["pre0923"]
        for field, value in reverted.values():
            self.assertEqual(getattr(pre, field), value, field)
        self.assertEqual(variants["reentry4"].reentry_drop_pct, 4.0)
        vol_scaled = variants["dca_vol_m1"]
        self.assertEqual(vol_scaled.dca_vol_scale_k, -1.0)
        self.assertEqual(vol_scaled.dca_vol_ref, 2.0)
        candidate = variants["overlay650t8_regime_v2"]
        self.assertTrue(candidate.trend_overlay)
        self.assertEqual(candidate.trend_topup, 650.0)
        self.assertEqual(candidate.trend_trail_pct, 8.0)
        self.assertTrue(variants["B_dcabrake_regime_v2"].dca_trend_brake)
        safe_overlay = variants["overlay_safe_combo"]
        self.assertTrue(safe_overlay.trend_overlay)
        self.assertEqual(safe_overlay.trend_topup, 350.0)
        self.assertEqual(safe_overlay.trend_trail_pct, 6.0)
        fixed20 = variants["fixed_surge20_pb3"]
        self.assertEqual(fixed20.surge_gain_pct, 20.0)
        self.assertEqual(fixed20.surge_exit_pullback_pct, 3.0)

    def test_interval_the_live_config_cannot_replay_is_skipped(self):
        with patch("strategies.spot_engine.StratParams.from_env", return_value=self._live_params()), patch.object(shadow, "_load_runtime_config"):
            self.assertIsNone(shadow._replay_interval_error(240))
            self.assertIn("240", shadow._replay_interval_error(60))
            with patch.dict(os.environ, {"KRAKEN_PAIR": "HYPEUSD"}), \
                    patch.object(shadow, "snapshot") as snapshot, \
                    patch.object(sys, "argv", ["shadow_live.py", "--interval", "60"]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(shadow.main(), 0)
            snapshot.assert_not_called()
            self.assertIn("skipped", out.getvalue())

    def test_runtime_config_matches_live_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            config_path = Path(tmp) / "config.env"
            env_path.write_text("KRAKEN_PAIR=FROM_ENV\nSTRAT_TAKEPROFIT_PCT=4.5\n", encoding="utf-8")
            config_path.write_text(
                "KRAKEN_PAIR=FROM_CONFIG\nSTRAT_TAKEPROFIT_PCT=5.0\nSTRAT_DCA_DROP_PCT=1.25  # versioned\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                shadow._load_runtime_config(str(env_path), str(config_path))
                self.assertEqual(os.environ["KRAKEN_PAIR"], "FROM_ENV")
                self.assertEqual(os.environ["STRAT_TAKEPROFIT_PCT"], "4.5")
                self.assertEqual(os.environ["STRAT_DCA_DROP_PCT"], "1.25")

    def test_forward_history_keeps_anchor_after_api_window_moves(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shadow, "LOG_DIR", tmp):
            first = [(100, 1.0, 2.0, 0.5, 1.5), (160, 1.5, 2.5, 1.0, 2.0)]
            later = [(160, 1.5, 2.5, 1.0, 2.0), (220, 2.0, 3.0, 1.5, 2.5)]
            shadow._merge_forward_history("PAIR", 1, 100, first)
            merged = shadow._merge_forward_history("PAIR", 1, 100, later)
            self.assertEqual([bar[0] for bar in merged], [100, 160, 220])

    def test_forward_history_fails_closed_when_a_gap_loses_bars(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shadow, "LOG_DIR", tmp):
            shadow._merge_forward_history("PAIR", 1, 100, [(100, 1.0, 2.0, 0.5, 1.5), (160, 1.5, 2.5, 1.0, 2.0)])
            with self.assertRaisesRegex(RuntimeError, "gap"):
                shadow._merge_forward_history("PAIR", 1, 100, [(400, 4.0, 5.0, 3.5, 4.5)])

    def test_display_compares_total_pnl_including_open_position(self):
        block = {
            "bars": 10,
            "buyhold_pct": 2.0,
            "configs": {
                "current": {"net_pct": 0.0, "total_pct": 1.0, "maxdd_pct": 2.0, "cycles": 0},
                "tp4": {"net_pct": 0.0, "total_pct": 2.0, "maxdd_pct": 2.0, "cycles": 0},
                "dca15": {"net_pct": 0.0, "total_pct": 0.0, "maxdd_pct": 2.0, "cycles": 0},
            },
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            shadow._print_block("TEST", block)
        rendered = out.getvalue()
        self.assertIn("+1.00pp", rendered)
        self.assertIn("-1.00pp", rendered)

    def test_single_shot_failure_returns_nonzero(self):
        with patch.object(shadow, "_load_runtime_config"), \
                patch.object(shadow, "_replay_interval_error", return_value=None), \
                patch.object(shadow, "snapshot", side_effect=RuntimeError("fetch failed")), \
                patch.dict(os.environ, {"KRAKEN_PAIR": "HYPEUSD"}), \
                patch.object(sys, "argv", ["shadow_live.py", "--quiet"]), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(shadow.main(), 1)

    def test_get_anchor_replaces_non_positive_persisted_anchor(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(shadow, "LOG_DIR", tmp):
            anchor_file = Path(tmp) / "HYPEUSD_240m.anchor"
            anchor_file.write_text("0\n", encoding="utf-8")
            self.assertEqual(shadow._get_anchor("HYPEUSD", 240, 1700000000), 1700000000)
            self.assertEqual(anchor_file.read_text(encoding="utf-8").strip(), "1700000000")


if __name__ == "__main__":
    unittest.main()

