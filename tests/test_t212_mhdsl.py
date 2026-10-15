"""Unified tests for Trading 212 environment loading, MHDSL, MDHTP, and Hybrid Re-entry."""

import importlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
T212_DIR = ROOT / "212trading"
for _p in (str(T212_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ipo_common = importlib.import_module("ipo_common")

_COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
_PRELOADED = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
try:
    spec = importlib.util.spec_from_file_location(
        "t212_strategy_mhdsl_test", os.path.join(T212_DIR, "strategy.py"),
    )
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    spec.loader.exec_module(strategy)
finally:
    for name in _COLLIDING:
        sys.modules.pop(name, None)
    sys.modules.update(_PRELOADED)


def _params(**overrides):
    config = {
        "STRAT_CURRENCY": "USD",
        "YAHOO_SYMBOL": "TEST",
        "STRATEGY_MODE": "avg_tp",
        "STRAT_ENTRY_PCT": "20",
        "STRAT_DCA_PCT": "10",
        "STRAT_ENTRY_DISCOUNT_PCT": "0.0",
        "STRAT_DCA_DROP_PCT": "2",
        "STRAT_TAKEPROFIT_PCT": "3",
        "STRAT_MAX_DCA_BUYS": "3",
        "STRAT_MAX_BUDGET": "500",
        "STRAT_FX_FEE_PCT": "0.15",
        "STRAT_STOP_LOSS_PCT": "10",
        "STRAT_CHECK_MINUTES": "5",
        "STRAT_ORDER_TTL_MIN": "10",
        "STRAT_REENTRY_DROP_PCT": "2.0",
        "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
        "STRAT_LOSS_ALERT_STEP": "1",
        "STRAT_LADDER_MIN_FREE": "6",
        "STRAT_SL_REBUY_ENABLED": "false",
        "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
        "STRAT_DCA_TREND_GATE_PCT": "0",
        "STRAT_TRAIL_PCT": "0",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
        "STRAT_REENTRY_HYBRID_ENABLED": "true",
        "STRAT_REENTRY_PULLBACK_PCT": "1.5",
        "STRAT_REENTRY_TTL_HOURS": "24.0",
        "STRAT_SL_MHDSL_ENABLED": "true",
        "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
        "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
        "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
        "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
    }
    config.update(overrides)
    return strategy.StratParams.from_env(config)


# ─── Environment Precedence & Loader ────────────────────────────────────────

def test_t212_environment_uses_one_documented_layer_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ipo_common, "load_dotenv", calls.append)
    env_file = tmp_path / ".env"
    profile_file = tmp_path / "config.asset.env"

    ipo_common.load_t212_environment(str(env_file), profile_file=str(profile_file))

    assert calls == [
        os.path.abspath(env_file),
        str(ROOT / ".env"),
        os.path.abspath(profile_file),
        str(T212_DIR / "runtime.env"),
    ]


def test_t212_launchers_use_the_shared_environment_loader():
    for relative in ("ipo.py", "t212_bot.py", "t212_status.py", "price_alert.py"):
        text = (T212_DIR / relative).read_text(encoding="utf-8")
        assert "load_t212_environment(" in text


def test_t212_environment_precedence_is_effective(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    t212 = root / "212trading"
    t212.mkdir(parents=True)
    (root / ".env").write_text("LAYER=shared\nSHARED_ONLY=yes\n", encoding="utf-8")
    secrets = t212 / ".env"
    secrets.write_text("LAYER=t212\nT212_ONLY=yes\n", encoding="utf-8")
    profile = t212 / "config.asset.env"
    profile.write_text("LAYER=profile\nPROFILE_ONLY=yes\n", encoding="utf-8")
    (t212 / "runtime.env").write_text("LAYER=runtime\nRUNTIME_ONLY=yes\n", encoding="utf-8")

    with monkeypatch.context() as context:
        context.setattr(ipo_common, "__file__", str(t212 / "ipo_common.py"))
        context.setenv("LAYER", "process")
        for key in ("SHARED_ONLY", "T212_ONLY", "PROFILE_ONLY", "RUNTIME_ONLY"):
            context.delenv(key, raising=False)
        ipo_common.load_t212_environment(str(secrets), profile_file=str(profile))
        assert os.environ["LAYER"] == "process"
        assert os.environ["T212_ONLY"] == "yes"
        assert os.environ["SHARED_ONLY"] == "yes"
        assert os.environ["PROFILE_ONLY"] == "yes"
        assert os.environ["RUNTIME_ONLY"] == "yes"


# ─── Base Helper for T212 Strategy Tests ────────────────────────────────────

class _T212BaseTest(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.account_cash.return_value = {"free": 1000.0}

    def _make_strat(self, *, clock=1000.0, slope=None, dry_run=True, positioned=False, **overrides):
        params = _params(**overrides)
        clock_fn = clock if callable(clock) else (lambda: clock)
        slope_fn = (slope if callable(slope) else (lambda _sym: slope)) if slope is not None else None
        strat = strategy.Strategy(
            client=self.mock_client,
            ticker="TEST_US_EQ",
            params=params,
            dry_run=dry_run,
            clock=clock_fn,
            trend_slope_provider=slope_fn,
        )
        if positioned:
            strat.s.update(
                qty=10.0, cost_usd=1000.0, spent_cash=1000.0,
                entry_price=100.0, last_buy_price=100.0,
            )
        return strat


# ─── Hybrid Re-Entry Tests ──────────────────────────────────────────────────

class T212HybridReentryTest(_T212BaseTest):
    def test_state_b_trend_confirmed_bypasses_sale_barrier_and_waits_pullback(self):
        strat = self._make_strat(
            clock=10000.0, slope=0.25,
            STRAT_SL_MHDSL_ENABLED="false", STRAT_STOP_LOSS_PCT="20",
            STRAT_REENTRY_HYBRID_ENABLED="true", STRAT_REENTRY_PULLBACK_PCT="1.5",
        )
        strat.s.update(last_sell_price=100.0, post_sell_peak=100.0, last_sell_ts=10000.0)

        strat.step(130.0)
        self.assertEqual(strat.s["post_sell_peak"], 130.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        strat.step(129.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        strat.step(127.5)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")
        self.assertEqual(strat.s["orders"][0]["limit"], 127.5)

    def test_state_c_no_confirmed_trend_enforces_sale_drop(self):
        strat = self._make_strat(
            clock=10000.0, slope=-0.15,
            STRAT_SL_MHDSL_ENABLED="false", STRAT_STOP_LOSS_PCT="20",
            STRAT_REENTRY_HYBRID_ENABLED="true", STRAT_REENTRY_DROP_PCT="2.0",
            STRAT_REENTRY_PULLBACK_PCT="1.5",
        )
        strat.s.update(last_sell_price=100.0, post_sell_peak=100.0, last_sell_ts=10000.0)

        strat.step(105.0)
        self.assertEqual(len(strat.s["orders"]), 0)
        strat.step(99.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        strat.step(97.5)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_ttl_expiry_unblocks_stale_barrier(self):
        current_time = 10000.0
        strat = self._make_strat(
            clock=lambda: current_time, slope=-0.10,
            STRAT_SL_MHDSL_ENABLED="false", STRAT_STOP_LOSS_PCT="20",
            STRAT_REENTRY_HYBRID_ENABLED="true", STRAT_REENTRY_DROP_PCT="2.0",
            STRAT_REENTRY_TTL_HOURS="24.0",
        )
        strat.s.update(last_sell_price=100.0, post_sell_peak=100.0, last_sell_ts=current_time)

        current_time += 3600.0
        strat.step(105.0)
        self.assertEqual(len(strat.s["orders"]), 0)

        current_time += 24.0 * 3600.0
        strat.step(105.0)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_adaptive_pullback_scaling(self):
        strat = self._make_strat(
            slope=0.15,
            STRAT_SL_MHDSL_ENABLED="false", STRAT_STOP_LOSS_PCT="20",
            STRAT_REENTRY_HYBRID_ENABLED="true", STRAT_REENTRY_PULLBACK_ADAPTIVE="true",
            STRAT_REENTRY_PULLBACK_K="1.0", STRAT_REENTRY_PULLBACK_MIN="1.0",
            STRAT_REENTRY_PULLBACK_MAX="4.0",
        )
        strat.s.update(last_sell_price=90.0, post_sell_peak=100.0)
        strat._shadow_vol_1h = lambda: 2.5

        val, desc = strat._effective_reentry_pullback_pct()
        self.assertAlmostEqual(val, 2.5)
        self.assertIn("adaptive", desc)

        strat.step(98.5)
        self.assertEqual(len(strat.s["orders"]), 0)
        strat.step(97.0)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")


# ─── MDHTP Tests ────────────────────────────────────────────────────────────

class T212MDHTPTest(_T212BaseTest):
    def _mdhtp_strat(self, *, clock=1000.0, slope=0.25, **overrides):
        defaults = dict(
            STRAT_DCA_DROP_PCT="5",
            STRAT_STOP_LOSS_PCT="20",
            STRAT_TRAIL_PCT="1.5",
            STRAT_TRAIL_MIN_PROFIT_PCT="0.0",
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_TP_STAGNATION_MAX_SEC="1080.0",
        )
        defaults.update(overrides)
        return self._make_strat(clock=clock, slope=slope, positioned=True, **defaults)

    def test_mdhtp_disabled_places_legacy_resting_limit(self):
        strat = self._mdhtp_strat(
            slope=0.5, STRAT_TP_MDHTP_ENABLED="false", STRAT_TAKEPROFIT_PCT="5",
        )
        self.assertIsNone(strat.exit_manager)
        with patch.object(strategy, "notify"):
            strat.step(100.0)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual((o["side"], o["kind"], o["limit"], o.get("market", False)), ("SELL", "TP", 105.0, False))

    def test_mdhtp_below_tp_keeps_resting_sell_clear(self):
        strat = self._mdhtp_strat(slope=0.5, STRAT_TAKEPROFIT_PCT="5")
        self.assertIsNotNone(strat.exit_manager)
        with patch.object(strategy, "notify"):
            strat.step(102.0)
            self.assertEqual(len(strat.s["orders"]), 0)
            self.assertNotIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)

    def test_mdhtp_initiates_trend_riding_in_uptrend(self):
        strat = self._mdhtp_strat()
        with patch.object(strategy, "notify"):
            strat.step(104.0)
            self.assertEqual(len(strat.s["orders"]), 0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)
            self.assertEqual(strat.s["trail_peak"], 104.0)
            self.assertEqual(strat.exit_manager.tp_ride_state["TEST_US_EQ"]["peak_price"], 104.0)

    def test_mdhtp_peak_pullback_triggers_market_exit(self):
        current_time = 1000.0
        strat = self._mdhtp_strat(clock=lambda: current_time, STRAT_TRAIL_PCT="1.5")
        with patch.object(strategy, "notify"):
            strat.step(105.0)
            self.assertEqual(strat.s["trail_peak"], 105.0)
            current_time = 1050.0
            strat.step(110.0)
            self.assertEqual(strat.s["trail_peak"], 110.0)
            current_time = 1100.0
            strat.step(108.0)
            self.assertEqual(len(strat.s["orders"]), 1)
            o = strat.s["orders"][0]
            self.assertEqual((o["side"], o["kind"], o.get("market", False)), ("SELL", "TP", True))

    def test_mdhtp_guaranteed_profit_floor_triggers_exit(self):
        current_time = 1000.0
        strat = self._mdhtp_strat(
            clock=lambda: current_time, STRAT_TRAIL_PCT="5.0", STRAT_TRAIL_MIN_PROFIT_PCT="4.0",
        )
        with patch.object(strategy, "notify"):
            strat.step(106.0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)
            current_time = 1050.0
            strat.step(103.5)
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual((strat.s["orders"][0]["kind"], strat.s["orders"][0].get("market", False)), ("TP", True))

    def test_mdhtp_stagnation_timeout_triggers_exit(self):
        current_time = 1000.0
        strat = self._mdhtp_strat(
            clock=lambda: current_time, STRAT_TRAIL_PCT="5.0", STRAT_TP_STAGNATION_MAX_SEC="600.0",
        )
        with patch.object(strategy, "notify"):
            strat.step(105.0)
            self.assertEqual(len(strat.s["orders"]), 0)
            current_time = 1601.0
            strat.step(105.0)
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual((strat.s["orders"][0]["kind"], strat.s["orders"][0].get("market", False)), ("TP", True))

    def test_mdhtp_trend_reversal_triggers_exit(self):
        current_time = 1000.0
        trend_slope = 0.25
        strat = self._mdhtp_strat(
            clock=lambda: current_time, slope=lambda _sym: trend_slope, STRAT_TRAIL_PCT="5.0",
        )
        with patch.object(strategy, "notify"):
            strat.step(105.0)
            self.assertEqual(len(strat.s["orders"]), 0)
            trend_slope = -0.15
            current_time = 1020.0
            strat.step(104.8)
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual((strat.s["orders"][0]["kind"], strat.s["orders"][0].get("market", False)), ("TP", True))

    def test_mdhtp_takes_precedence_over_tp_ladder(self):
        strat = self._mdhtp_strat(
            slope=0.2, STRAT_TRAIL_PCT="5.0", STRAT_TP_LADDER="105:20,110:30,115:50",
        )
        with patch.object(strategy, "notify"):
            strat.step(105.0)
            self.assertEqual(len(strat.s["orders"]), 0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.tp_ride_state)

    def test_mdhtp_order_refusal_intent_persistence_and_retry(self):
        current_time = 1000.0
        strat = self._mdhtp_strat(clock=lambda: current_time, slope=-0.1, STRAT_TRAIL_PCT="5.0")
        with patch.object(strategy, "notify"), patch.object(strat, "_place_sell", return_value=False):
            strat.step(105.0)
            self.assertIsNotNone(strat.s.get("tp_pending_exit"))
            self.assertEqual(strat.s["tp_pending_exit"]["reason"], "gain_threshold_exceeded_and_trend_not_up")

        with patch.object(strategy, "notify"), patch.object(strat, "_place_sell", return_value=True) as mock_place_sell:
            current_time = 1005.0
            strat.step(104.9)
            mock_place_sell.assert_called_once()
            self.assertIsNone(strat.s.get("tp_pending_exit"))

    def test_mdhtp_defers_and_cancels_all_orders_including_dca_and_resizes_on_late_fill(self):
        current_time = 1000.0
        strat = self._mdhtp_strat(clock=lambda: current_time, slope=-0.1, STRAT_TRAIL_PCT="5.0")
        strat.s["orders"] = [{"id": "DCA-BUY-1", "side": "BUY", "qty": 2.0, "limit": 95.0, "kind": "DCA", "market": False}]

        cancelled_orders = []
        with patch.object(strategy, "notify"), \
             patch.object(strat, "_cancel_all_orders", side_effect=lambda: (cancelled_orders.extend(strat.s["orders"]), True)[1]), \
             patch.object(strat, "_place_sell", return_value=True) as mock_place_sell:
            strat.step(105.0)
            self.assertEqual(len(cancelled_orders), 1)
            self.assertEqual(cancelled_orders[0]["id"], "DCA-BUY-1")
            self.assertIsNotNone(strat.s.get("tp_pending_exit"))
            mock_place_sell.assert_not_called()

            strat.s["orders"] = []
            strat.s["qty"] = 12.0
            current_time = 1005.0
            strat.step(104.9)

            mock_place_sell.assert_called_once()
            self.assertEqual(mock_place_sell.call_args[0][0], 12.0)
            self.assertIsNone(strat.s.get("tp_pending_exit"))

    def test_mdhtp_restart_with_accepted_market_exit_order_does_not_cancel_it(self):
        strat = self._mdhtp_strat(slope=None)
        strat.s.update(
            tp_pending_exit={"action": "SELL", "sell_fraction": 1.0, "reason": "mdhtp_exit", "time": 990.0},
            orders=[
                {"id": "MKT-EXIT-1", "side": "SELL", "market": True, "kind": "TP", "qty": 10.0},
                {"id": "DCA-RESTING-2", "side": "BUY", "market": False, "kind": "DCA", "qty": 2.0},
            ],
        )
        cancelled = []
        with patch.object(strategy, "notify"), \
             patch.object(strat, "_cancel_specific", side_effect=lambda o: (cancelled.append(o), True)[1]), \
             patch.object(strat, "_place_sell") as mock_place_sell:
            strat.step(105.0)
            self.assertEqual(len(cancelled), 1)
            self.assertEqual(cancelled[0]["id"], "DCA-RESTING-2")
            mock_place_sell.assert_not_called()
            self.assertIsNotNone(strat.s.get("tp_pending_exit"))

            strat.s["qty"] = 0.0
            strat.s["orders"] = []
            strat.step(105.0)
            self.assertIsNone(strat.s.get("tp_pending_exit"))


# ─── MHDSL Tests ────────────────────────────────────────────────────────────

class T212MHDSLTest(_T212BaseTest):
    def test_mhdsl_disabled_preserves_legacy_stop(self):
        strat = self._make_strat(
            slope=0.5, positioned=True,
            STRAT_SL_MHDSL_ENABLED="false", STRAT_STOP_LOSS_PCT="10",
        )
        self.assertIsNone(strat.exit_manager)
        with patch.object(strategy, "notify"):
            self.assertTrue(strat._check_stop_loss(89.0))
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual(strat.s["orders"][0]["kind"], "SL")

    def test_mhdsl_trend_deferral_holds_in_uptrend(self):
        strat = self._make_strat(
            slope=0.25, positioned=True,
            STRAT_SL_MHDSL_ENABLED="true", STRAT_STOP_LOSS_PCT="10",
            STRAT_SL_DISASTER_BUFFER_RATIO="0.30", STRAT_SL_TREND_DEFER_MAX_SEC="220.0",
        )
        self.assertIsNotNone(strat.exit_manager)
        with patch.object(strategy, "notify"):
            self.assertFalse(strat._check_stop_loss(89.0))
            self.assertEqual(len(strat.s["orders"]), 0)

    def test_mhdsl_hard_disaster_circuit_breaker(self):
        strat = self._make_strat(
            slope=0.5, positioned=True,
            STRAT_SL_MHDSL_ENABLED="true", STRAT_STOP_LOSS_PCT="10",
            STRAT_SL_DISASTER_BUFFER_RATIO="0.30",
        )
        with patch.object(strategy, "notify"):
            self.assertTrue(strat._check_stop_loss(86.0))
            self.assertEqual(len(strat.s["orders"]), 1)
            self.assertEqual(strat.s["orders"][0]["kind"], "SL")

    def test_mhdsl_reentry_mirror_requires_bounce_after_stop(self):
        strat = self._make_strat(
            slope=-0.2,
            STRAT_SL_MHDSL_ENABLED="true", STRAT_REENTRY_DROP_PCT="2.0",
            STRAT_REENTRY_PULLBACK_PCT="1.5",
        )
        strat.s.update(
            qty=0.0, last_sell_price=90.0, post_sell_peak=90.0,
            post_sell_trough=90.0, last_exit_kind="STOP", last_sell_ts=1000.0,
        )
        strat.step(85.0)
        self.assertEqual(len(strat.s["orders"]), 0)
        self.assertEqual(strat.s["post_sell_trough"], 85.0)

        strat.step(85.5)
        self.assertEqual(len(strat.s["orders"]), 0)

        strat.step(86.4)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_cycle_closure_clears_mhdsl_state(self):
        self.mock_client.get_portfolio.return_value = []
        self.mock_client.list_active_orders.return_value = []
        strat = self._make_strat(dry_run=False, STRAT_SL_MHDSL_ENABLED="true")
        strat.exit_manager.sl_bounce_state["TEST_US_EQ"] = {"state": "watching"}
        strat.s.update(
            orders=[], qty=10.0, realized_pnl_usd=15.0,
            realized_net_usd=14.5, fees_usd=0.5, last_sell_price=105.0,
        )
        strat.reconcile(105.0)
        self.assertEqual(strat.s["cycle"], 2)
        self.assertNotIn("TEST_US_EQ", strat.exit_manager.sl_bounce_state)
        self.assertEqual(strat.s["sl_bounce_state"], {})

    def test_mhdsl_blocks_dca_during_bounce_deferral(self):
        strat = self._make_strat(
            slope=0.25, positioned=True,
            STRAT_SL_MHDSL_ENABLED="true", STRAT_STOP_LOSS_PCT="10",
            STRAT_DCA_DROP_PCT="2", STRAT_MAX_DCA_BUYS="3",
        )
        with patch.object(strategy, "notify"):
            strat.step(89.0)
            self.assertIn("TEST_US_EQ", strat.exit_manager.sl_bounce_state)
            dca_buys = [o for o in strat.s["orders"] if o.get("kind") == "DCA" or o.get("side") == "buy"]
            self.assertEqual(len(dca_buys), 0)

    def test_trailing_warmup_protects_dca_and_mhdsl_layers(self):
        overrides = dict(
            STRAT_TRAIL_PCT="8.0", STRAT_TRAIL_MIN_PROFIT_PCT="5.0",
            STRAT_STOP_LOSS_PCT="30.0", STRAT_DCA_DROP_PCT="2.0",
        )
        strat = self._make_strat(positioned=True, **overrides)
        with patch.object(strategy, "notify"):
            strat.step(91.0)
            self.assertFalse(any(o.get("kind") == "TR" for o in strat.s["orders"]))
            self.assertFalse(strat.s.get("tr_armed", False))

        strat.s["orders"] = []
        with patch.object(strategy, "notify"):
            strat.step(68.0)
            self.assertTrue(any(o.get("kind") == "SL" for o in strat.s["orders"]))

        strat2 = self._make_strat(positioned=True, **overrides)
        with patch.object(strategy, "notify"):
            strat2.step(106.0)
            self.assertTrue(strat2.s.get("tr_armed"))
            self.assertEqual(strat2.s.get("pos_peak"), 106.0)
            strat2.step(97.0)
            self.assertEqual(len(strat2.s["orders"]), 1)
            self.assertEqual(strat2.s["orders"][0]["kind"], "TR")

    def test_t212_vol_1h_percentage_converted_to_ratio_for_exit_manager(self):
        strat = self._make_strat(positioned=True, STRAT_SL_MHDSL_ENABLED="true", STRAT_STOP_LOSS_PCT="10.0")
        strat._shadow_vol_1h = lambda: 2.5

        captured_snapshots = []
        orig_eval = strat.exit_manager.evaluate_stop_loss
        strat.exit_manager.evaluate_stop_loss = lambda snap, custom_threshold=None: (
            captured_snapshots.append(snap),
            orig_eval(snap, custom_threshold=custom_threshold),
        )[1]

        strat._check_stop_loss(89.0)
        self.assertEqual(len(captured_snapshots), 1)
        self.assertAlmostEqual(captured_snapshots[0].vol_1h, 0.025, places=5)
        self.assertNotAlmostEqual(captured_snapshots[0].vol_1h, 2.5, places=2)

    def test_t212_sl_disaster_buffer_max_pct_propagates_to_exit_manager(self):
        strat = self._make_strat(
            STRAT_SL_MHDSL_ENABLED="true", STRAT_SL_DISASTER_BUFFER_MAX_PCT="12.5",
        )
        self.assertEqual(strat.p.sl_disaster_buffer_max_pct, 12.5)
        self.assertIsNotNone(strat.exit_manager)
        self.assertAlmostEqual(strat.exit_manager.config.sl_disaster_buffer_max_pct, 0.125, places=5)


if __name__ == "__main__":
    unittest.main()
