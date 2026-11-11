"""Unified tests for Trading 212 environment loading, MHDSL, MDHTP, and Hybrid Re-entry."""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
T212_DIR = ROOT / "212trading"
for _p in (str(T212_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from botcore import float_env, load_local_module, parse_dotenv  # noqa: E402

ipo_common = importlib.import_module("ipo_common")

_COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
_PRELOADED = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
try:
    spec = importlib.util.spec_from_file_location(
        "t212_strategy_mhdsl_test", os.path.join(T212_DIR, "strategy.py"),
    )
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    sys.modules["strategy"] = strategy
    spec.loader.exec_module(strategy)
    market_data = sys.modules.get("market_data") or importlib.import_module("market_data")
    t212_bot = load_local_module(str(T212_DIR), "t212_bot", module_key="t212_bot_mhdsl_test")
    listing_watcher = load_local_module(str(T212_DIR), "listing_watcher", module_key="t212_listing_watcher_mhdsl_test")
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
        "STRAT_REENTRY_PEAK_RELATIVE": "false",
        "STRAT_REENTRY_PULLBACK_PCT": "1.5",
        "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false",
        "STRAT_REENTRY_PULLBACK_K": "1.0",
        "STRAT_REENTRY_PULLBACK_MIN": "0.8",
        "STRAT_REENTRY_PULLBACK_MAX": "4.0",
        "STRAT_REENTRY_TTL_HOURS": "24.0",
        "STRAT_SL_MHDSL_ENABLED": "true",
        "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
        "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0",
        "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
        "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
        "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
        "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true",
        "STRAT_TP_MDHTP_ENABLED": "false",
        "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
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
    t212.mkdir(parents=True, exist_ok=True)
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
            initial_state=strategy._new_state(),
            clock=clock_fn,
            trend_slope_provider=slope_fn,
        )
        strat._save = lambda: None
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
             patch.object(strat, "_cancel_order", side_effect=lambda o: (cancelled.append(o), True)[1]), \
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

    def test_post_stop_does_not_require_bounce_when_sl_rebuy_and_mhdsl_disabled(self):
        strat = self._make_strat(
            slope=-0.2,
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_SL_REBUY_ENABLED="false",
            STRAT_TP_MDHTP_ENABLED="true",
            STRAT_REENTRY_HYBRID_ENABLED="false",
            STRAT_REENTRY_DROP_PCT="2.0",
        )
        self.assertIsNotNone(strat.exit_manager)
        self.assertEqual(strat.exit_manager.config.reentry_sl_bounce_pct, 0.0)
        strat.s.update(
            qty=0.0, last_sell_price=100.0, post_sell_peak=100.0,
            post_sell_trough=95.0, last_exit_kind="STOP", last_sell_ts=1000.0,
        )
        # At 95.0 (trough, 0% bounce, 5% below last_sell_price=100.0), entry is allowed immediately
        strat.step(95.0)
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_sl_rebuy_ttl_expiry_and_tolerance_cleanup(self):
        current_time = 1000.0
        strat = self._make_strat(
            clock=lambda: current_time,
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_SL_REBUY_ENABLED="true",
            STRAT_SL_REBUY_BOUNCE_PCT="1.2",
            STRAT_REENTRY_TTL_HOURS="24.0",
            STRAT_REENTRY_DROP_PCT="5.0",
        )
        strat.s.update(
            qty=0.0,
            last_sell_price=100.0,
            last_sell_ts=current_time,
            last_exit_kind="STOP",
            sl_rebuy={"sl_price": 100.0, "low_since": 90.0, "qty": 5.0, "cost_usd": 500.0, "dca_buys": 1},
        )
        # Advance beyond 24h TTL: sl_rebuy should expire and allow normal entry
        current_time += 25.0 * 3600.0
        strat.step(98.0)
        self.assertIsNone(strat.s.get("sl_rebuy"))
        self.assertEqual(len(strat.s["orders"]), 1)
        self.assertEqual(strat.s["orders"][0]["kind"], "ENTRY")

    def test_reconcile_real_returns_false_when_portfolio_unavailable(self):
        self.mock_client.get_portfolio.return_value = None
        strat = self._make_strat(dry_run=False)
        self.assertFalse(strat.reconcile(100.0))

    def test_sl_rebuy_preserved_when_place_buy_deferred(self):
        strat = self._make_strat(
            dry_run=False,
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_SL_REBUY_ENABLED="true",
            STRAT_SL_REBUY_BOUNCE_PCT="1.2",
        )
        strat.s.update(
            qty=0.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            sl_rebuy={"low": 90.0, "sell_price": 100.0},
            pending_submit={"intent_id": "in-flight"},
        )
        strat.step(90.0 * 1.015)
        self.assertIsNotNone(strat.s.get("sl_rebuy"))
        self.assertEqual(strat.s.get("last_exit_kind"), "STOP")

    def test_mhdsl_sl_pending_exit_persists_across_order_cancellation_when_trend_up(self):
        current_time = 1000.0
        strat = self._make_strat(
            dry_run=False,
            clock=lambda: current_time,
            slope=0.35,
            positioned=True,
            STRAT_SL_MHDSL_ENABLED="true",
            STRAT_STOP_LOSS_PCT="10",
            STRAT_SL_DISASTER_BUFFER_RATIO="0.30",
            STRAT_SL_TREND_DEFER_MAX_SEC="220.0",
        )
        strat.s["orders"] = [{"id": "TP-RESTING-1", "side": "SELL", "qty": 10.0, "limit": 103.0, "kind": "TP", "market": False}]

        # Tick 1 (t=1000): enters MHDSL bounce watch while trend is up
        with patch.object(strategy, "notify"):
            strat.step(89.0)
        self.assertIn("TEST_US_EQ", strat.exit_manager.sl_bounce_state)
        self.assertIsNone(strat.s.get("sl_pending_exit"))

        # Tick 2 (t=1230, > 220s): mhdsl_bounce_timeout fires; open TP order is requested to cancel
        current_time = 1230.0
        with patch.object(strategy, "notify"), \
             patch.object(strat.executor, "cancel_order_with_intent", return_value=None), \
             patch.object(strat, "_place_sell", return_value=True) as mock_place_sell:
            strat.step(89.0)
            mock_place_sell.assert_not_called()
            self.assertIsNotNone(strat.s.get("sl_pending_exit"))
            self.assertIn("mhdsl_bounce_timeout", strat.s["sl_pending_exit"]["reason"])
            self.assertTrue(strat.s["orders"][0].get("cancel_requested"))

            # Tick 3 (t=1245): TP order cancellation reconciled; SL must execute immediately even though slope > 0
            strat.s["orders"] = []
            current_time = 1245.0
            strat.step(89.0)
            mock_place_sell.assert_called_once()
            self.assertEqual(mock_place_sell.call_args.kwargs.get("kind"), "SL")
            self.assertTrue(mock_place_sell.call_args.kwargs.get("market"))
            self.assertIsNone(strat.s.get("sl_pending_exit"))
            self.assertNotIn("TEST_US_EQ", strat.exit_manager.sl_bounce_state)

    def test_restore_cancelled_buy_converts_usd_notional_to_account_ccy(self):
        strat = self._make_strat(STRAT_CURRENCY="RON")
        strat.fx_to_usd = 0.22
        captured = []
        strat._place_buy = lambda amount, limit, kind="ENTRY": (captured.append((amount, limit, kind)) or True)
        # Adopted/synced BUY order has qty & limit in USD, without 'amount'
        strat._restore_cancelled_orders([{"side": "BUY", "qty": 2.0, "limit": 110.0, "kind": "DCA"}], reason="test")
        self.assertEqual(len(captured), 1)
        self.assertAlmostEqual(captured[0][0], 220.0 / 0.22, places=4)
        self.assertEqual(captured[0][1], 110.0)
        self.assertEqual(captured[0][2], "DCA")


# ─── Per-Asset Config Isolation & Discovery (from test_t212_bot.py) ─────────

_NVDA_ENV = """
T212_TICKER=NVDA_US_EQ
YAHOO_SYMBOL=NVDA
STRATEGY_MODE=avg_tp
STRAT_ENTRY_PCT=20
STRAT_DCA_PCT=10
STRAT_TAKEPROFIT_PCT=1.5
STRAT_CURRENCY=EUR
STRAT_REENTRY_DROP_PCT=1.5
STRAT_CHECK_MINUTES=0.4  # comentariu inline
STRAT_ENTRY_DISCOUNT_PCT=0.2
STRAT_DCA_DROP_PCT=2
STRAT_MAX_DCA_BUYS=3
STRAT_MAX_BUDGET=500
STRAT_FX_FEE_PCT=0.15
STRAT_STOP_LOSS_PCT=20
STRAT_ORDER_TTL_MIN=10
STRAT_REENTRY_TOLERANCE_PCT=0
STRAT_LOSS_ALERT_STEP=1
STRAT_LADDER_MIN_FREE=6
STRAT_SL_REBUY_ENABLED=false
STRAT_SL_REBUY_BOUNCE_PCT=1.2
STRAT_DCA_TREND_GATE_PCT=0
STRAT_TRAIL_PCT=0
STRAT_TRAIL_MIN_PROFIT_PCT=5
STRAT_REENTRY_HYBRID_ENABLED=false
STRAT_REENTRY_PEAK_RELATIVE=false
STRAT_REENTRY_PULLBACK_PCT=1.5
STRAT_REENTRY_PULLBACK_ADAPTIVE=false
STRAT_REENTRY_PULLBACK_K=1.0
STRAT_REENTRY_PULLBACK_MIN=0.8
STRAT_REENTRY_PULLBACK_MAX=4.0
STRAT_REENTRY_TTL_HOURS=0.0
STRAT_SL_MHDSL_ENABLED=false
STRAT_SL_DISASTER_BUFFER_RATIO=0.30
STRAT_SL_DISASTER_BUFFER_MAX_PCT=15.0
STRAT_SL_TREND_DEFER_MAX_SEC=220.0
STRAT_SL_BOUNCE_MIN_RECOVERY_PCT=0.40
STRAT_SL_BOUNCE_PULLBACK_PCT=0.30
STRAT_SL_STRUCTURAL_BREAK_ENABLED=true
STRAT_TP_MDHTP_ENABLED=false
STRAT_TP_STAGNATION_MAX_SEC=1080.0
"""

_SPCX_ENV = """
T212_TICKER=SPCX_US_EQ
YAHOO_SYMBOL=SPCX
STRATEGY_MODE=avg_tp
STRAT_ENTRY_PCT=30
STRAT_DCA_PCT=10
STRAT_TAKEPROFIT_PCT=5.0
STRAT_CURRENCY=EUR
STRAT_REENTRY_DROP_PCT=2
STRAT_CHECK_MINUTES=1
STRAT_ENTRY_DISCOUNT_PCT=0.2
STRAT_DCA_DROP_PCT=2
STRAT_MAX_DCA_BUYS=3
STRAT_MAX_BUDGET=400
STRAT_FX_FEE_PCT=0.15
STRAT_STOP_LOSS_PCT=20
STRAT_ORDER_TTL_MIN=10
STRAT_REENTRY_TOLERANCE_PCT=0
STRAT_LOSS_ALERT_STEP=1
STRAT_LADDER_MIN_FREE=6
STRAT_SL_REBUY_ENABLED=false
STRAT_SL_REBUY_BOUNCE_PCT=1.2
STRAT_DCA_TREND_GATE_PCT=0
STRAT_TRAIL_PCT=0
STRAT_TRAIL_MIN_PROFIT_PCT=5
STRAT_REENTRY_HYBRID_ENABLED=false
STRAT_REENTRY_PEAK_RELATIVE=false
STRAT_REENTRY_PULLBACK_PCT=1.5
STRAT_REENTRY_PULLBACK_ADAPTIVE=false
STRAT_REENTRY_PULLBACK_K=1.0
STRAT_REENTRY_PULLBACK_MIN=0.8
STRAT_REENTRY_PULLBACK_MAX=4.0
STRAT_REENTRY_TTL_HOURS=0.0
STRAT_SL_MHDSL_ENABLED=false
STRAT_SL_DISASTER_BUFFER_RATIO=0.30
STRAT_SL_DISASTER_BUFFER_MAX_PCT=15.0
STRAT_SL_TREND_DEFER_MAX_SEC=220.0
STRAT_SL_BOUNCE_MIN_RECOVERY_PCT=0.40
STRAT_SL_BOUNCE_PULLBACK_PCT=0.30
STRAT_SL_STRUCTURAL_BREAK_ENABLED=true
STRAT_TP_MDHTP_ENABLED=false
STRAT_TP_STAGNATION_MAX_SEC=1080.0
"""


def _tmp_env_file(content: str) -> str:
    import atexit
    f = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    f.write(content)
    f.close()
    atexit.register(lambda p=f.name: os.unlink(p) if os.path.exists(p) else None)
    return f.name


class TestParseDotenv(unittest.TestCase):
    def test_returns_dict_without_environ(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("FOO_UNIQ_123=bar\n")
            path = f.name
        try:
            cfg = parse_dotenv(path)
            self.assertEqual(cfg["FOO_UNIQ_123"], "bar")
            self.assertNotIn("FOO_UNIQ_123", os.environ, "parse_dotenv must NOT touch os.environ")
        finally:
            os.unlink(path)

    def test_it_strips_inline_comments(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("X=0.4  # check minutes\nY=\"quoted\"\n")
            path = f.name
        try:
            cfg = parse_dotenv(path)
            self.assertEqual(cfg["X"], "0.4")
            self.assertEqual(cfg["Y"], "quoted")
        finally:
            os.unlink(path)


class TestFloatEnvDict(unittest.TestCase):
    def test_from_a_dict(self):
        self.assertEqual(float_env("A", {"A": "12.5"}), 12.5)
        self.assertIsNone(float_env("MISSING", {"A": "1"}))

    def test_environ_by_default(self):
        os.environ["TST_FLT_9"] = "3.3"
        try:
            self.assertEqual(float_env("TST_FLT_9"), 3.3)
        finally:
            del os.environ["TST_FLT_9"]


class TestIzolareParams(unittest.TestCase):
    def test_two_assets_with_different_parameters(self):
        nv = parse_dotenv(_tmp_env_file(_NVDA_ENV))
        sp = parse_dotenv(_tmp_env_file(_SPCX_ENV))
        pn = strategy.StratParams.from_env(nv)
        ps = strategy.StratParams.from_env(sp)
        self.assertEqual(pn.entry_amount, 100.0)
        self.assertEqual(ps.entry_amount, 120.0)
        self.assertEqual(pn.takeprofit_pct, 1.5)
        self.assertEqual(ps.takeprofit_pct, 5.0)
        self.assertEqual(pn.yahoo_sym, "NVDA")
        self.assertEqual(ps.yahoo_sym, "SPCX")
        self.assertEqual(pn.reentry_drop_pct, 1.5)
        self.assertEqual(ps.reentry_drop_pct, 2.0)

    def test_inline_comment_in_check_minutes(self):
        pn = strategy.StratParams.from_env(parse_dotenv(_tmp_env_file(_NVDA_ENV)))
        self.assertEqual(pn.check_minutes, 0.4)

    def test_missing_environment_fails_instead_of_using_defaults(self):
        with self.assertRaisesRegex(ValueError, "mandatory setting"):
            strategy.StratParams.from_env({})


class TestDiscover(unittest.TestCase):
    def test_it_finds_the_active_configs(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("nvda", "spcx", "rgnt"):
                open(os.path.join(d, f"config.{n}.env"), "w").close()
            open(os.path.join(d, ".env"), "w").close()
            names = [n for n, _ in t212_bot.discover_assets(d)]
            self.assertEqual(names, ["nvda", "rgnt", "spcx"])


# ─── Launch Detection & ISIN Verification (from test_launch_detect.py) ──────

def _chart_payload(price=None, meta_vol=0, meta_age_s=None, state="REGULAR", series=None):
    meta = {"marketState": state, "currency": "USD"}
    if price is not None:
        meta["regularMarketPrice"] = price
    meta["regularMarketVolume"] = meta_vol
    if meta_age_s is not None:
        meta["regularMarketTime"] = time.time() - meta_age_s
    ts, closes, vols = [], [], []
    for age_s, c, v in (series or []):
        ts.append(int(time.time() - age_s))
        closes.append(c)
        vols.append(v)
    return json.dumps({"chart": {"result": [{
        "meta": meta, "timestamp": ts,
        "indicators": {"quote": [{"close": closes, "volume": vols}]}
    }]}})


class _MarketDataBase(unittest.TestCase):
    def setUp(self):
        self._orig = market_data.http_get

    def tearDown(self):
        market_data.http_get = self._orig

    def feed(self, body, status=200):
        market_data.http_get = lambda url, headers=None: (status, body)


class TestLansare(_MarketDataBase):
    def test_a_pre_ipo_placeholder_is_NOT_launched(self):
        self.feed(_chart_payload(price=135.0, meta_vol=0, meta_age_s=55 * 3600, series=[]))
        m = market_data.check_market("SPCX")
        self.assertFalse(m["launched"])

    def test_a_flat_series_placeholder_is_NOT_launched(self):
        self.feed(_chart_payload(price=135.0, meta_vol=0, meta_age_s=55 * 3600,
                                 series=[(600, 135.0, 0), (60, 135.0, 0)]))
        self.assertFalse(market_data.check_market("SPCX")["launched"])

    def test_spcx_case_meta_static_but_series_live(self):
        self.feed(_chart_payload(price=135.0, meta_vol=0, meta_age_s=55 * 3600,
                                 series=[(600, 164.0, 100), (300, 165.0, 200), (60, 166.0, 150)]))
        m = market_data.check_market("SPCX")
        self.assertTrue(m["launched"], "a live series must declare the launch")
        self.assertEqual(m["price"], 166.0, "the price is the last live bar, not the 135 from meta")

    def test_a_price_move_without_volume_counts_as_launched(self):
        self.feed(_chart_payload(price=135.0, meta_vol=0, meta_age_s=55 * 3600,
                                 series=[(300, 164.0, None), (60, 167.0, None)]))
        self.assertTrue(market_data.check_market("SPCX")["launched"])

    def test_a_normally_listed_share_NVDA(self):
        self.feed(_chart_payload(price=204.0, meta_vol=1_000_000, meta_age_s=300,
                                 series=[(300, 204.0, 5000)]))
        self.assertTrue(market_data.check_market("NVDA")["launched"])

    def test_an_OLD_series_does_not_trigger(self):
        self.feed(_chart_payload(price=135.0, meta_vol=0, meta_age_s=55 * 3600,
                                 series=[(3600, 164.0, 100), (1800, 165.0, 200)]))
        self.assertFalse(market_data.check_market("SPCX")["launched"])

    def test_without_data_it_returns_none(self):
        self.feed(json.dumps({"chart": {"result": []}}))
        self.assertIsNone(market_data.check_market("SPCX"))


class TestPretProaspat(_MarketDataBase):
    def test_get_price_prefers_the_series(self):
        self.feed(_chart_payload(price=135.0, meta_vol=0, meta_age_s=55 * 3600,
                                 series=[(120, 164.0, 100), (30, 168.0, 200)]))
        self.assertEqual(market_data.get_price_usd("SPCX"), 168.0)

    def test_get_price_falls_back_to_meta_without_a_series(self):
        self.feed(_chart_payload(price=204.0, meta_vol=1000, meta_age_s=60, series=[]))
        self.assertEqual(market_data.get_price_usd("NVDA"), 204.0)

    def test_non_list_volume_does_not_drop_series_bars(self):
        now = int(time.time())
        body = json.dumps({
            "chart": {
                "result": [{
                    "meta": {"marketState": "REGULAR", "currency": "USD", "regularMarketPrice": 135.0},
                    "timestamp": [now - 120, now - 30],
                    "indicators": {"quote": [{"close": [164.0, 168.0], "volume": {"bad": 1}}]},
                }],
            },
        })
        self.feed(body)
        self.assertEqual(market_data.get_price_usd("SPCX"), 168.0)


class TestVerifyInstrumentIsin(unittest.TestCase):
    def test_verify_instrument_isin_handles_match_and_mismatch(self):
        notified = []
        client = SimpleNamespace(
            list_instruments=lambda: [{"ticker": "NVDA_US_EQ", "isin": "US67066G1040"}]
        )
        self.assertTrue(
            listing_watcher.verify_instrument_isin(
                client, "NVDA_US_EQ", "US67066G1040", notify_fn=lambda **kw: notified.append(kw)
            )
        )
        self.assertEqual(notified, [])
        self.assertFalse(
            listing_watcher.verify_instrument_isin(
                client, "NVDA_US_EQ", "US0000000000", notify_fn=lambda **kw: notified.append(kw)
            )
        )
        self.assertEqual(len(notified), 1)


# ─── Dynamic TP Ladder Tests (from test_dynamic_ladder.py) ──────────────────

class TestDynamicTPLadder(unittest.TestCase):
    def test_generate_tp_ladder_linear_10_steps(self):
        ladder = strategy.generate_tp_ladder(9.0, 40.0, 10, curve="linear")
        self.assertEqual(len(ladder), 10)
        self.assertEqual(ladder[0][0], 9.0)
        self.assertEqual(ladder[-1][0], 40.0)
        levels = [lvl for lvl, _ in ladder]
        self.assertEqual(levels, sorted(levels))
        total_frac = sum(frac for _, frac in ladder)
        self.assertAlmostEqual(total_frac, 1.0, places=3)
        for _, frac in ladder:
            self.assertEqual(frac, 0.1)

    def test_generate_tp_ladder_curves(self):
        front = strategy.generate_tp_ladder(9.0, 40.0, 5, curve="front_heavy")
        self.assertEqual(len(front), 5)
        fracs_front = [frac for _, frac in front]
        self.assertEqual(fracs_front, sorted(fracs_front, reverse=True))

        back = strategy.generate_tp_ladder(9.0, 40.0, 5, curve="back_heavy")
        self.assertEqual(len(back), 5)
        fracs_back = [frac for _, frac in back]
        self.assertEqual(fracs_back, sorted(fracs_back))

        geo = strategy.generate_tp_ladder(9.0, 40.0, 5, curve="geometric")
        self.assertEqual(len(geo), 5)
        self.assertEqual(geo[0][0], 9.0)
        self.assertEqual(geo[-1][0], 40.0)

        bell = strategy.generate_tp_ladder(9.0, 40.0, 5, curve="bell")
        self.assertEqual(len(bell), 5)
        fracs_bell = [frac for _, frac in bell]
        self.assertEqual(fracs_bell[2], max(fracs_bell))
        self.assertLess(fracs_bell[0], fracs_bell[1])
        self.assertLess(fracs_bell[4], fracs_bell[3])
        self.assertAlmostEqual(sum(fracs_bell), 1.0, places=3)

    def test_adapt_ladder_rungs(self):
        full = strategy.generate_tp_ladder(9.0, 40.0, 10)
        adapted_full = strategy.adapt_ladder_rungs(full, held=4.57)
        self.assertEqual(len(adapted_full), 10)

        adapted_5 = strategy.adapt_ladder_rungs(full, held=0.05, min_tranche_qty=0.01)
        self.assertEqual(len(adapted_5), 5)
        self.assertEqual(adapted_5[0][0], 9.0)
        self.assertEqual(adapted_5[-1][0], 40.0)

        adapted_2 = strategy.adapt_ladder_rungs(full, held=0.02, min_tranche_qty=0.01)
        self.assertEqual(len(adapted_2), 2)
        self.assertEqual(adapted_2[0][0], 9.0)
        self.assertEqual(adapted_2[-1][0], 40.0)

        self.assertEqual(strategy.adapt_ladder_rungs(full, held=0.0), [])

    def test_from_env_range_parsing(self):
        p1 = _params(STRAT_TP_LADDER="range:9:40:10")
        self.assertEqual(len(p1.tp_ladder), 10)
        self.assertEqual(p1.tp_ladder[0][0], 9.0)
        self.assertEqual(p1.tp_ladder[-1][0], 40.0)

        p2 = _params(STRAT_TP_LADDER="9:40:10")
        self.assertEqual(len(p2.tp_ladder), 10)
        self.assertEqual(p2.tp_ladder[0][0], 9.0)

        p3 = _params(STRAT_TP_LADDER="", STRAT_TP_LADDER_MIN="9.0", STRAT_TP_LADDER_MAX="40.0", STRAT_TP_LADDER_STEPS="10")
        self.assertEqual(len(p3.tp_ladder), 10)
        self.assertEqual(p3.tp_ladder[0][0], 9.0)
        self.assertEqual(p3.tp_ladder[-1][0], 40.0)

        p4 = _params(STRAT_TP_LADDER="11:33,20:33,30:34")
        self.assertEqual(len(p4.tp_ladder), 3)
        self.assertEqual(p4.tp_ladder[0][0], 11.0)
        self.assertEqual(p4.tp_ladder[1][0], 20.0)
        self.assertEqual(p4.tp_ladder[2][0], 30.0)


# ─── Cycle Completion & Dust Handling (from test_cycle_completion.py) ────────

_CYCLE_TICK = "CYCLE_TEST_US_EQ"
_TRAIL_TICK = "TRAILTEST_US_EQ"
_SLREBUY_TICK = "SLREBUYTEST_US_EQ"


class _MockT212Client:
    def __init__(self, ticker=_CYCLE_TICK, initial_qty=2.9229, initial_avg=150.0):
        self.portfolio = [{"ticker": ticker, "quantity": initial_qty, "averagePrice": initial_avg}]
        self.active = []
        self.placed = []
        self._id = 0

    def get_portfolio(self):
        return self.portfolio

    def list_active_orders(self):
        return self.active

    def cancel_order(self, oid):
        self.active = [o for o in self.active if o.get("id") != oid]
        return True

    def place_limit_order(self, ticker, qty, limit, validity):
        self._id += 1
        oid = f"LIM-{self._id}"
        o = {"id": oid, "ticker": ticker, "quantity": qty, "limit": limit}
        self.placed.append(o)
        self.active.append(o)
        return 201, {"id": oid}

    def place_market_order(self, ticker, qty, extended_hours=False):
        self._id += 1
        oid = f"MKT-{self._id}"
        o = {"id": oid, "ticker": ticker, "quantity": qty, "market": True}
        self.placed.append(o)
        return 201, {"id": oid}


class TestCycleCompletionAndDust(unittest.TestCase):
    def setUp(self):
        self._orig_notify = strategy.notify
        strategy.notify = lambda **kw: None
        self.sf = strategy.state_path_for(_CYCLE_TICK)
        if os.path.exists(self.sf):
            os.remove(self.sf)
        self.now_val = 10000.0
        self.params = _params(
            YAHOO_SYMBOL="CYCLETEST",
            STRAT_MAX_BUDGET="1000",
            STRAT_ENTRY_PCT="20",
            STRAT_DCA_PCT="10",
            STRAT_MAX_DCA_BUYS="auto",
            STRAT_ENTRY_DISCOUNT_PCT="0.2",
            STRAT_DCA_DROP_PCT="2",
            STRAT_CHECK_MINUTES="0.4",
            STRAT_TAKEPROFIT_PCT="11.0",
            STRAT_TP_LADDER="11:33,20:33,30:34",
            STRAT_STOP_LOSS_PCT="30",
            STRAT_REENTRY_DROP_PCT="1.0",
            STRAT_LADDER_MIN_FREE="6.0",
            STRAT_SL_REBUY_ENABLED="true",
            STRAT_TRAIL_PCT="0",
            STRAT_TRAIL_MIN_PROFIT_PCT="0",
            STRAT_REENTRY_HYBRID_ENABLED="false",
            STRAT_REENTRY_TTL_HOURS="0.0",
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_TP_MDHTP_ENABLED="false",
        )
        self.client = _MockT212Client(ticker=_CYCLE_TICK, initial_qty=2.9229, initial_avg=150.0)
        self.strat = strategy.Strategy(self.client, _CYCLE_TICK, self.params, dry_run=False,
                                       clock=lambda: self.now_val)

    def tearDown(self):
        strategy.notify = self._orig_notify
        if os.path.exists(self.sf):
            os.remove(self.sf)

    def test_is_dust(self):
        self.assertTrue(self.strat._is_dust(0.0))
        self.assertTrue(self.strat._is_dust(1e-10))
        self.assertTrue(self.strat._is_dust(0.0429, ref_price=150.0))
        self.assertFalse(self.strat._is_dust(2.0, ref_price=150.0))
        self.strat.p.ladder_min_free = 0.0
        self.assertFalse(self.strat._is_dust(0.0429, ref_price=150.0))

    def test_sellable_qty_dust_returns_zero(self):
        qty = self.strat._sellable_qty(0.0429, ref_price=150.0)
        self.assertEqual(qty, 0.0)

    def test_live_tp_ladder_cycle_closure_with_dust(self):
        self.strat.s["qty"] = 2.9229
        self.strat.s["cost_usd"] = 438.435
        self.strat.s["spent_cash"] = 438.44
        self.strat.s["entry_price"] = 150.0
        self.strat.s["tp_sold_levels"] = [11.0, 20.0]
        order_30 = {
            "id": "ORD-30", "side": "SELL", "qty": 2.88,
            "limit": 195.0, "level": 30.0, "kind": "TP",
        }
        self.strat.s["orders"] = [order_30]
        self.client.portfolio = [{"ticker": _CYCLE_TICK, "quantity": 0.0429, "averagePrice": 150.0}]
        self.client.active = []

        mock_status = MagicMock()
        mock_status.status = "closed"
        mock_status.filled_qty = 2.88
        mock_status.cost = 2.88 * 195.0
        mock_status.fee = 0.5
        self.strat._tracked_order_status = lambda o, c: mock_status

        self.strat._reconcile_real(195.0)
        self.assertEqual(self.strat.s["cycle"], 2)
        self.assertEqual(self.strat.s["qty"], 0.0)
        self.assertAlmostEqual(self.strat.s["last_sell_price"], 195.0, places=2)
        self.assertGreater(self.strat.s["realized_pnl_usd"], 0.0)

        self.strat._reconcile_real(195.0)
        self.assertEqual(self.strat.s["qty"], 0.0)
        self.assertEqual(len(self.strat.s["orders"]), 0)

        placed_buys = []
        self.strat._place_buy = lambda amt, lim, kind: placed_buys.append((amt, lim, kind))
        self.strat.step(190.0)
        self.assertEqual(len(placed_buys), 1)
        self.assertEqual(placed_buys[0][2], "ENTRY")

    def test_real_position_adoption_not_blocked(self):
        self.strat.s["qty"] = 0.0
        self.strat.s["orders"] = []
        self.client.portfolio = [{"ticker": _CYCLE_TICK, "quantity": 5.0, "averagePrice": 150.0}]
        self.strat._tracked_order_status = lambda o, c: None
        self.strat._reconcile_real(150.0)
        self.assertEqual(self.strat.s["qty"], 5.0)
        self.assertEqual(self.strat.s["entry_price"], 150.0)

    def test_paper_mode_cycle_closure_with_ladder_dust(self):
        paper_strat = strategy.Strategy(self.client, _CYCLE_TICK, self.params, dry_run=True,
                                        clock=lambda: self.now_val)
        paper_strat.s["qty"] = 2.9229
        paper_strat.s["cost_usd"] = 438.435
        paper_strat.s["tp_sold_levels"] = [11.0, 20.0, 30.0]
        paper_order = {"side": "SELL", "qty": 2.88, "kind": "TP", "level": 30.0}
        paper_strat._apply_fill(paper_order, 2.88, 195.0)
        self.assertEqual(paper_strat.s["cycle"], 2)
        self.assertEqual(paper_strat.s["qty"], 0.0)
        self.assertEqual(paper_strat.s["last_sell_price"], 195.0)

    def test_absent_order_released_after_confirmation_passes(self):
        order = {"id": "GHOST-1", "side": "BUY", "qty": 0.03, "limit": 230.0, "ts": 1000.0}
        self.strat.s["orders"] = [order]
        self.client.active = []
        self.strat._tracked_order_status = lambda o, c: "not_found"

        self.strat._reconcile_real(230.0)
        self.assertEqual(len(self.strat.s["orders"]), 1)
        self.assertEqual(order.get("lookup_misses"), 1)

        self.strat._reconcile_real(230.0)
        self.assertEqual(len(self.strat.s["orders"]), 1)
        self.assertEqual(order.get("lookup_misses"), 2)

        self.strat._reconcile_real(230.0)
        self.assertEqual(len(self.strat.s["orders"]), 0)

    def test_transient_status_error_keeps_order_tracked(self):
        order = {"id": "TRANSIENT-1", "side": "BUY", "qty": 0.03, "limit": 230.0, "ts": 1000.0}
        self.strat.s["orders"] = [order]
        self.client.active = []
        self.strat._tracked_order_status = lambda o, c: None

        for _ in range(5):
            self.strat._reconcile_real(230.0)
            self.assertEqual(len(self.strat.s["orders"]), 1)
            self.assertEqual(order.get("lookup_misses", 0), 0)

    def test_absent_cancel_requested_released_after_two_passes(self):
        order = {"id": "CANCEL-1", "side": "SELL", "qty": 0.08, "limit": 250.0,
                 "cancel_requested": True, "cancel_ts": 1000.0, "ts": 900.0}
        self.strat.s["orders"] = [order]
        self.client.active = []
        self.strat._tracked_order_status = lambda o, c: "not_found"

        self.strat._reconcile_real(250.0)
        self.assertEqual(len(self.strat.s["orders"]), 1)

        self.strat._reconcile_real(250.0)
        self.assertEqual(len(self.strat.s["orders"]), 0)

    def test_strategy_run_graceful_stop(self):
        import threading
        stop_event = threading.Event()
        stop_event.set()
        self.strat.run(stop=stop_event)

    def test_cycle_closure_cancels_leftover_open_dca_order(self):
        self.strat.s["qty"] = 2.0
        self.strat.s["cost_usd"] = 300.0
        self.strat.s["spent_cash"] = 300.0
        self.strat.s["entry_price"] = 150.0
        self.strat.s["tp_sold_levels"] = [11.0, 20.0]

        dca_buy = {
            "id": "DCA-BUY-1", "intent_id": "intent-dca-1",
            "side": "BUY", "qty": 1.0, "limit": 140.0, "kind": "DCA",
        }
        order_30 = {
            "id": "ORD-30", "side": "SELL", "qty": 2.0,
            "limit": 195.0, "level": 30.0, "kind": "TP",
        }
        self.strat.s["orders"] = [dca_buy, order_30]
        self.client.portfolio = []
        self.client.active = [{"id": "DCA-BUY-1", "ticker": _CYCLE_TICK, "quantity": 1.0, "limit": 140.0}]

        mock_status = MagicMock()
        mock_status.status = "closed"
        mock_status.filled_qty = 2.0
        mock_status.cost = 2.0 * 195.0
        mock_status.fee = 0.0
        self.strat._tracked_order_status = lambda o, c: mock_status if o["id"] == "ORD-30" else None

        cancelled_ids = []
        self.strat.executor.cancel_order_with_intent = lambda iid, tkr, oid: cancelled_ids.append(oid)

        self.strat._reconcile_real(195.0)
        self.assertEqual(self.strat.s["cycle"], 2)
        self.assertIn("DCA-BUY-1", cancelled_ids)
        self.assertEqual(len(self.strat.s["orders"]), 1)
        self.assertTrue(self.strat.s["orders"][0].get("cancel_requested"))

    def test_not_owned_zero_closes_cycle_and_resets_stale_state(self):
        self.strat.s.update({
            "qty": 2.0, "cost_usd": 300.0, "spent_cash": 300.0,
            "entry_price": 150.0, "last_buy_price": 140.0, "dca_buys": 2,
            "tp_sold_levels": [11.0, 20.0], "loss_band": 3,
            "pos_peak": 180.0, "tr_armed": True,
        })
        self.client.place_limit_order = lambda tkr, q, lim, validity=None: (
            400,
            {"code": "selling-equity-not-owned", "message": "selling-equity-not-owned: owned=0"},
        )

        placed = self.strat._place_sell(2.0, 165.0, level=30.0, kind="TP")
        self.assertFalse(placed)
        self.assertEqual(self.strat.s["cycle"], 2)
        self.assertEqual(self.strat.s["qty"], 0.0)
        self.assertEqual(self.strat.s["dca_buys"], 0)
        self.assertIsNone(self.strat.s["entry_price"])
        self.assertIsNone(self.strat.s["last_buy_price"])
        self.assertEqual(self.strat.s["tp_sold_levels"], [])
        self.assertEqual(self.strat.s["loss_band"], 0)
        self.assertEqual(self.strat.s["last_sell_price"], 165.0)
        self.assertEqual(self.strat.s["post_sell_peak"], 165.0)
        self.assertEqual(self.strat.s["post_sell_trough"], 165.0)
        self.assertEqual(self.strat.s["locked_zero_until"], self.now_val + 300)

    def test_reconcile_real_rejects_bool_filled_qty_and_deduplicates_tp_sold_levels(self):
        self.strat.s["qty"] = 2.0
        self.strat.s["cost_usd"] = 300.0
        self.strat.s["tp_sold_levels"] = [11.0]
        self.strat.s["orders"] = [
            {"id": "BOOL-1", "side": "SELL", "qty": 1.0, "limit": 166.5, "level": 20.0, "kind": "TP"},
            {"id": "DUP-1", "side": "SELL", "qty": 1.0, "limit": 166.5, "level": 11.0, "kind": "TP"},
        ]
        self.client.portfolio = [{"ticker": _CYCLE_TICK, "quantity": 2.0, "averagePrice": 150.0}]
        self.client.active = []

        def _status_for(o, cache):
            st = MagicMock()
            st.status = "canceled"
            st.filled_qty = True if o["id"] == "BOOL-1" else 0.5
            st.cost = 0.0
            st.fee = 0.0
            return st

        self.strat._tracked_order_status = _status_for
        self.strat._reconcile_real(150.0)
        self.assertEqual(self.strat.s["tp_sold_levels"], [11.0])


# ─── Trailing Stop & Sellable Qty (from test_t212_trailing_stop.py) ──────────

class TestTrailingStopAndSellableQty(unittest.TestCase):
    def setUp(self):
        self._orig_notify = strategy.notify
        strategy.notify = lambda **kw: None
        self.sf = strategy.state_path_for(_TRAIL_TICK)
        if os.path.exists(self.sf):
            os.remove(self.sf)
        self.now_val = 10000.0
        self.params = _params(
            YAHOO_SYMBOL="TRAILTEST",
            STRAT_MAX_BUDGET="1000",
            STRAT_ENTRY_PCT="20",
            STRAT_DCA_PCT="10",
            STRAT_MAX_DCA_BUYS="auto",
            STRAT_ENTRY_DISCOUNT_PCT="0.2",
            STRAT_DCA_DROP_PCT="2",
            STRAT_CHECK_MINUTES="0.4",
            STRAT_TAKEPROFIT_PCT="5.0",
            STRAT_STOP_LOSS_PCT="30",
            STRAT_REENTRY_DROP_PCT="1.0",
            STRAT_LADDER_MIN_FREE="6.0",
            STRAT_SL_REBUY_ENABLED="true",
            STRAT_TRAIL_PCT="8",
            STRAT_TRAIL_MIN_PROFIT_PCT="0",
            STRAT_REENTRY_HYBRID_ENABLED="false",
            STRAT_REENTRY_TTL_HOURS="0.0",
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_TP_MDHTP_ENABLED="false",
        )
        self.client = _MockT212Client(ticker=_TRAIL_TICK, initial_qty=2.92, initial_avg=146.0)
        self.strat = strategy.Strategy(self.client, _TRAIL_TICK, self.params, dry_run=False,
                                       clock=lambda: self.now_val)
        self.strat.s["qty"] = 2.9229
        self.strat.s["cost_usd"] = 426.74
        self.strat.s["pos_peak"] = 175.0
        self.strat.s["tr_armed"] = True

    def tearDown(self):
        strategy.notify = self._orig_notify
        if os.path.exists(self.sf):
            os.remove(self.sf)

    def test_sellable_qty_with_buffer(self):
        qty = self.strat._sellable_qty(2.9229, ref_price=146.0)
        self.assertEqual(qty, 2.88)
        self.assertLess(qty, 2.9229)

    def test_sellable_qty_no_buffer(self):
        self.strat.p.ladder_min_free = 0.0
        qty = self.strat._sellable_qty(2.9229, ref_price=146.0)
        self.assertEqual(qty, 2.92)

    def test_trailing_drawdown_triggers_sell(self):
        self.assertTrue(self.strat._check_trailing(160.0))
        self.assertEqual(len(self.client.placed), 1)
        self.assertEqual(self.client.placed[0]["quantity"], -2.89)
        self.assertTrue(self.strat.s["sl_pending"])

    def test_trailing_failure_sets_backoff(self):
        self.strat._place_sell = lambda *a, **kw: False
        res = self.strat._check_trailing(160.0)
        self.assertTrue(res)
        self.assertEqual(self.strat.s.get("tr_fail_until"), self.now_val + 1800)
        self.assertFalse(self.strat._check_trailing(160.0))
        self.strat._check_trailing(180.0)
        self.assertNotIn("tr_fail_until", self.strat.s)
        self.assertEqual(self.strat.s["pos_peak"], 180.0)

    def test_stop_loss_failure_sets_backoff(self):
        self.strat._place_sell = lambda *a, **kw: False
        res = self.strat._check_stop_loss(100.0)
        self.assertTrue(res)
        self.assertEqual(self.strat.s.get("sl_fail_until"), self.now_val + 1800)
        self.assertFalse(self.strat._check_stop_loss(100.0))
        self.strat._check_stop_loss(140.0)
        self.assertNotIn("sl_fail_until", self.strat.s)

    def test_trailing_waits_for_existing_sell_order_cancellation_to_reconcile(self):
        tp_order = {"id": "TP-1", "side": "SELL", "qty": 2.88, "limit": 190.0, "kind": "TP", "level": 30.0}
        self.strat.s["orders"] = [tp_order]
        self.strat._cancel_order = lambda o: o.update({"cancel_requested": True}) or True

        res = self.strat._check_trailing(160.0)
        self.assertTrue(res)
        self.assertTrue(tp_order.get("cancel_requested"))
        self.assertEqual(len(self.client.placed), 0)
        self.assertNotIn("tr_fail_until", self.strat.s)

        self.strat.s["orders"] = []
        res2 = self.strat._check_trailing(160.0)
        self.assertTrue(res2)
        self.assertEqual(len(self.client.placed), 1)
        self.assertTrue(self.strat.s.get("sl_pending"))

    def test_stop_loss_waits_for_existing_sell_order_cancellation_to_reconcile(self):
        tp_order = {"id": "TP-1", "side": "SELL", "qty": 2.88, "limit": 190.0, "kind": "TP", "level": 30.0}
        self.strat.s["orders"] = [tp_order]
        self.strat._cancel_order = lambda o: o.update({"cancel_requested": True}) or True

        res = self.strat._check_stop_loss(100.0)
        self.assertTrue(res)
        self.assertTrue(tp_order.get("cancel_requested"))
        self.assertEqual(len(self.client.placed), 0)
        self.assertNotIn("sl_fail_until", self.strat.s)

        self.strat.s["orders"] = []
        res2 = self.strat._check_stop_loss(100.0)
        self.assertTrue(res2)
        self.assertEqual(len(self.client.placed), 1)
        self.assertTrue(self.strat.s.get("sl_pending"))

    def test_stop_loss_waits_for_existing_buy_order_cancellation_to_reconcile(self):
        buy_order = {"id": "DCA-1", "side": "BUY", "qty": 1.0, "limit": 90.0, "kind": "DCA"}
        self.strat.s["orders"] = [buy_order]
        self.strat._cancel_order = lambda o: o.update({"cancel_requested": True}) or True

        res = self.strat._check_stop_loss(100.0)
        self.assertTrue(res)
        self.assertTrue(buy_order.get("cancel_requested"))
        self.assertEqual(len(self.client.placed), 0)

        self.strat.s["orders"] = []
        res2 = self.strat._check_stop_loss(100.0)
        self.assertTrue(res2)
        self.assertEqual(len(self.client.placed), 1)
        self.assertTrue(self.strat.s.get("sl_pending"))

    def test_dca_and_tp_inhibited_during_stop_loss_pending_or_backoff(self):
        self.strat.s["last_buy_price"] = 150.0
        self.strat.s["entry_price"] = 150.0
        self.strat.s["dca_buys"] = 0
        self.strat.s["spent_cash"] = 400.0
        price = 100.0
        self.strat.s["sl_pending"] = True
        self.strat.s["orders"] = [{"id": "SL-1", "side": "SELL", "kind": "SL", "market": True, "qty": 2.86}]
        self.strat.step(price)
        self.assertEqual(len(self.client.placed), 0)

        self.strat.s["sl_pending"] = False
        self.strat.s["sl_fail_until"] = self.now_val + 1800
        self.strat.step(price)
        self.assertFalse(any(o.get("kind") == "DCA" for o in self.client.placed))

    def test_tp_ladder_does_not_lock_subsequent_levels_when_submit_is_pending(self):
        self.strat.p.tp_ladder = ((11.0, 0.5), (20.0, 0.5))
        self.strat.s["orders"] = []
        self.strat.s["pending_submit"] = {"id": "SUB-1", "side": "SELL"}
        self.strat._manage_tp_ladder(2.92, 146.0)
        self.assertNotIn("11.0", self.strat.s.get("tp_fail_until", {}))
        self.assertNotIn("20.0", self.strat.s.get("tp_fail_until", {}))

    def test_trailing_failure_restores_cancelled_tp_orders(self):
        tp_order = {"id": "TP-1", "side": "SELL", "kind": "TP", "level": 11.0, "qty": 1.45, "limit": 162.0}
        self.strat.s["orders"] = [tp_order]

        def mock_place_sell(qty, limit, level=None, kind="TP", market=False):
            if kind == "TR":
                return False
            self.strat.s["orders"].append({
                "id": f"RESTORED-{len(self.strat.s['orders'])}",
                "side": "SELL", "qty": qty, "limit": limit, "level": level, "kind": kind,
            })
            return True

        self.strat._place_sell = mock_place_sell
        self.strat._cancel_order = lambda o: True

        res = self.strat._check_trailing(160.0)
        self.assertTrue(res)
        restored = [o for o in self.strat.s["orders"] if o.get("kind") == "TP"]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["qty"], 1.45)
        self.assertEqual(restored[0]["limit"], 162.0)

    def test_trailing_warmup_prevents_loss_exit_on_initial_dip(self):
        self.strat.p.trail_min_profit_pct = 5.0
        self.strat.s["tr_armed"] = False
        self.strat.s["pos_peak"] = 146.0
        self.strat.s["qty"] = 2.9229
        self.strat.s["cost_usd"] = 146.0 * 2.9229

        res = self.strat._check_trailing(134.0)
        self.assertFalse(res)
        self.assertFalse(self.strat.s.get("tr_armed"))
        self.assertEqual(len(self.client.placed), 0)

        res = self.strat._check_trailing(154.0)
        self.assertFalse(res)
        self.assertTrue(self.strat.s.get("tr_armed"))
        self.assertEqual(self.strat.s.get("pos_peak"), 154.0)

        res = self.strat._check_trailing(141.0)
        self.assertTrue(res)
        self.assertEqual(len(self.client.placed), 1)
        self.assertEqual(self.client.placed[0]["quantity"], -2.88)
        self.assertTrue(self.strat.s.get("sl_pending"))


# ─── Stop-Loss Rebuy End-to-End Execution Path (from test_sl_rebuy.py) ───────

class TestSlRebuyEndToEnd(unittest.TestCase):
    def setUp(self):
        self._orig_notify = strategy.notify
        strategy.notify = lambda **kw: None
        self.sf = strategy.state_path_for(_SLREBUY_TICK)
        if os.path.exists(self.sf):
            os.remove(self.sf)

    def tearDown(self):
        strategy.notify = self._orig_notify
        if os.path.exists(self.sf):
            os.remove(self.sf)

    def test_sl_rebuy_workflow(self):
        params = _params(
            YAHOO_SYMBOL="SLREBUY",
            STRAT_ENTRY_PCT="10",
            STRAT_DCA_PCT="10",
            STRAT_DCA_DROP_PCT="2",
            STRAT_TAKEPROFIT_PCT="3",
            STRAT_MAX_DCA_BUYS="3",
            STRAT_MAX_BUDGET="1000",
            STRAT_ENTRY_DISCOUNT_PCT="0.2",
            STRAT_STOP_LOSS_PCT="30",
            STRAT_SL_REBUY_ENABLED="true",
            STRAT_SL_REBUY_BOUNCE_PCT="1.2",
            STRAT_REENTRY_DROP_PCT="1.0",
            STRAT_REENTRY_HYBRID_ENABLED="false",
            STRAT_REENTRY_TTL_HOURS="0.0",
            STRAT_SL_MHDSL_ENABLED="false",
            STRAT_TP_MDHTP_ENABLED="false",
        )
        client = _MockT212Client(ticker=_SLREBUY_TICK, initial_qty=1.0, initial_avg=100.0)
        s = strategy.Strategy(client, _SLREBUY_TICK, params, dry_run=False)

        s.s["qty"] = 1.0
        s.s["cost_usd"] = 100.0
        s.s["entry_price"] = 100.0
        s.s["last_buy_price"] = 100.0
        s.s["spent_cash"] = 100.0
        s.s["sl_pending"] = True

        # 1+2. SL executed -> portfolio is 0; reconciliation sees catastrophe close
        client.portfolio = [{"ticker": _SLREBUY_TICK, "quantity": 0.0, "averagePrice": 0.0}]
        client.active = []
        s._reconcile_real(94.0)
        rb = s.s.get("sl_rebuy")
        self.assertIsNotNone(rb)
        self.assertAlmostEqual(s.s.get("last_sell_price", 0), 94.0)
        self.assertLessEqual(s.s["qty"], 1e-9)
        n_after_arm = len(client.placed)

        # 3. Deeper fall tracks low and does not buy
        s.step(92.0)
        self.assertIsNotNone(s.s.get("sl_rebuy"))
        self.assertAlmostEqual(s.s["sl_rebuy"]["low"], 92.0)
        self.assertEqual(len(client.placed), n_after_arm)

        # 4. Small pullback (+0.5% from 92) < 1.2% threshold -> no buy yet
        s.step(92.0 * 1.005)
        self.assertEqual(len(client.placed), n_after_arm)
        self.assertIsNotNone(s.s.get("sl_rebuy"))

        # 5. Pullback >= 1.2% from low of 92 -> places BUY ENTRY
        s.step(92.0 * 1.013)
        self.assertEqual(len(client.placed), n_after_arm + 1)
        self.assertGreater(client.placed[-1]["quantity"], 0)

        # 6. Consumed arm prevents another buy on next bounce
        s.step(92.0 * 1.02)
        self.assertIsNone(s.s.get("sl_rebuy"))
        self.assertEqual(len(client.placed), n_after_arm + 1)


if __name__ == "__main__":
    unittest.main()

