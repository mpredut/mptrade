"""Unified SpotEngine exit lifecycle, profit-floor guard, surge protection, MDHTP, and MHDSL tests."""

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest
import strategies.spot_engine as se
from strategies import spot_rules
from strategies.position_exit_manager import (
    ExitAction,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentryAction,
    ReentrySnapshot,
)
from strategies.spot_engine import (
    StratParams,
    Strategy,
    _PLACEMENT_BACKOFF_CAP_SEC,
    _parse_bool_flag,
    _parse_tranches,
    clean_order_id,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import binance_api.bapi as bapi_mod
import order_retry as oq
import strategies.spot_replay as srep
from market_regime import MarketRegimeDecision
from offline.backtests.execution import ExecutionModel
from providers.execution_audit import _resolve_intent_execution_details, cancel_existing_ai_open_orders
from providers.order_ohlc_utils import extract_remaining_qty
from providers.strategy_executor import OrderStatus, PairPrecision, ProviderError
from providers.t212_provider import T212Provider
from strategies.spot_replay import run_replay
from test_kraken_strategy_provider_live import FakeExecutor, _strategy
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def silence_notifications(monkeypatch):
    monkeypatch.setattr(se, "notify", lambda **kwargs: None)


def order(order_id, side, *, qty=1.0, price=100.0, kind="DCA", market=False):
    return {
        "txid": order_id, "side": side, "vol": qty, "price": price,
        "amount": qty * price if side == "buy" else 0.0,
        "kind": kind, "market": market, "ts": 9999999999.0,
    }


def positioned(**params):
    executor = FakeExecutor()
    engine = _strategy(executor, **params)
    engine.s.update(
        qty=1.0, cost=100.0, spent=100.0,
        entry_price=100.0, last_buy_price=100.0,
    )
    return engine, executor


def submits(executor):
    return [call for call in executor.calls if call[0] == "submit_order"]


def _make_guard_strategy(**overrides):
    client = MagicMock()
    client.pair_precision.return_value = PairPrecision(2, 6, 0.01, "TEST")
    defaults = dict(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2, dca_amount=50.0,
        dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=1.9, max_budget=1000.0,
        max_dca_buys=10, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=0.0,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2, reentry_tolerance_pct=0.05,
        reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )
    defaults.update(overrides)
    params = se.StratParams(**defaults)
    s = se.Strategy(
        client, "TESTPAIR_GUARD", params, dry_run=True,
        initial_state=se._new_state(),
    )
    s.s["cost"] = 1000.0
    s.s["qty"] = 10.0
    return s


# ─── Adversarial Exit Lifecycle ─────────────────────────────────────────────

def test_market_exit_waits_for_terminal_cancels_and_uses_racing_fill():
    for mode in ["stop", "trailing", "overlay"]:
        engine, executor = positioned(trend_overlay=(mode == "overlay"))
        engine.s["orders"] = [
            order("DCA-1", "buy", price=95.0),
            order("TP-1", "sell", price=110.0, kind="TP"),
        ]
        engine.s.update(
            trail_peak=110.0, trail_stop=106.7,
            trend_mode=(mode == "overlay"), trend_peak=110.0,
        )
        price = 80.0 if mode == "stop" else 103.0
        engine.step(price)
        assert not submits(executor)
        assert all(o.get("cancel_requested") for o in engine.s["orders"])
        executor.order_status = lambda _symbol, oid: (
            OrderStatus("canceled", 0.25, 23.75, 0.02) if oid == "DCA-1"
            else OrderStatus("canceled", 0.0, 0.0, 0.0)
        )
        engine.reconcile(price)
        engine.step(price)
        s = submits(executor)
        assert len(s) == 1 and s[0][2] == "sell" and s[0][5]
        assert s[0][3] == engine._dust_safe_qty(1.25)


def test_triggered_stop_survives_restart_and_price_recovery_without_new_buy():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine.step(80.0)
    assert not submits(executor)
    saved = copy.deepcopy(engine.s)
    restarted = _strategy(executor)
    restarted.s = saved
    executor.next_status = OrderStatus("canceled", 0.0, 0.0, 0.0)
    restarted.reconcile(100.0)
    restarted.step(100.0)
    s = submits(executor)
    assert len(s) == 1 and s[0][2] == "sell" and s[0][6] == "STOP"
    assert not restarted._has_open("buy")


def test_terminal_sell_does_not_forget_unresolved_buy_or_close_cycle_twice():
    engine, executor = positioned()
    engine.s["orders"] = [
        order("DCA-1", "buy", price=95.0),
        order("TP-1", "sell", price=110.0, kind="TP"),
    ]
    statuses = {
        "DCA-1": OrderStatus("open", 0, 0, 0),
        "TP-1": OrderStatus("closed", 1, 110, 0.1),
    }
    executor.order_status = lambda _symbol, oid: statuses[oid]
    engine.reconcile(110.0)
    assert engine.s["cycle"] == 1
    assert engine._has_open("buy")
    engine.step(110.0)
    assert not submits(executor)
    statuses["DCA-1"] = OrderStatus("canceled", 0, 0, 0)
    engine.reconcile(110.0)
    engine.step(110.0)
    assert engine.s["cycle"] == 2
    engine.reconcile(110.0)
    engine.step(110.0)
    assert engine.s["cycle"] == 2
    assert engine.s["last_sell_price"] == 110.0
    assert engine.s["orders"] == []


def test_terminal_sell_keeps_late_buy_fill_and_exits_only_that_remainder():
    engine, executor = positioned()
    engine.s["orders"] = [
        order("DCA-1", "buy", price=95.0),
        order("TP-1", "sell", price=110.0, kind="TP"),
    ]
    statuses = {
        "DCA-1": OrderStatus("open", 0, 0, 0),
        "TP-1": OrderStatus("closed", 1, 110, 0.1),
    }
    executor.order_status = lambda _symbol, oid: statuses[oid]
    engine.reconcile(110.0)
    assert engine._has_open("buy")
    statuses["DCA-1"] = OrderStatus("canceled", 0.2, 19.0, 0.01)
    engine.reconcile(109.0)
    assert engine.s["cycle"] == 1
    assert engine.s["qty"] == pytest.approx(0.2)
    engine.step(109.0)
    s = submits(executor)
    assert len(s) == 1 and s[0][2] == "sell"
    assert s[0][3] == engine._dust_safe_qty(0.2)


def test_exit_intent_must_be_saved_before_cancel_or_submit():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine._save = lambda: (_ for _ in ()).throw(RuntimeError("disk unavailable"))
    with pytest.raises(RuntimeError, match="disk unavailable"):
        engine.step(80.0)
    with pytest.raises(RuntimeError, match="disk unavailable"):
        engine.step(100.0)
    assert not any(c[0] in {"cancel_order", "submit_order"} for c in executor.calls)


def test_pending_soft_floor_is_rechecked_but_stop_can_upgrade_the_exit():
    engine, executor = positioned(tp_trail_profit_floor_pct=1.0)
    engine.s.update(trail_peak=110.0, trail_stop=106.7)
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine.step(103.0)
    executor.next_status = OrderStatus("canceled", 0, 0, 0)
    engine.reconcile(100.0)
    engine.step(100.0)
    assert not submits(executor)
    engine.step(80.0)
    s = submits(executor)
    assert len(s) == 1 and s[0][6] == "STOP"


def test_unavailable_cancel_status_keeps_exit_and_all_trackers():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    executor.order_status = lambda *_args: (_ for _ in ()).throw(ProviderError("status unavailable"))
    engine.step(80.0)
    engine.reconcile(80.0)
    engine.step(80.0)
    assert engine._has_open("buy") and engine.s["pending_exit"]["kind"] == "STOP"
    assert not submits(executor)
    assert len([c for c in executor.calls if c[0] == "cancel_order"]) == 1


def test_accepted_market_exit_is_never_cancelled_or_duplicated():
    engine, executor = positioned()
    engine.step(80.0)
    engine.step(75.0)
    engine.step(105.0)
    assert len(submits(executor)) == 1
    assert not any(c[0] == "cancel_order" for c in executor.calls)


def test_late_buy_never_cancels_an_already_accepted_market_exit():
    engine, executor = positioned()
    engine.s["orders"] = [
        order("DCA-1", "buy", price=95.0),
        order("STOP-1", "sell", price=80.0, kind="STOP", market=True),
    ]
    executor.order_status = lambda _symbol, oid: (
        OrderStatus("canceled", 0.2, 19.0, 0.01) if oid == "DCA-1"
        else OrderStatus("open", 0, 0, 0)
    )
    engine.reconcile(80.0)
    assert engine._has_pending_market_exit()
    assert engine.s["qty"] == pytest.approx(1.2)
    assert not any(c[0] == "cancel_order" and c[2] == "STOP-1" for c in executor.calls)
    executor.order_status = lambda *_args: OrderStatus("closed", 1.0, 80.0, 0.1)
    engine.reconcile(100.0)
    engine.step(100.0)
    s = submits(executor)
    assert len(s) == 1 and s[0][6] == "STOP"
    assert s[0][3] == engine._dust_safe_qty(0.2)


def test_exit_request_round_trips_through_real_state_store(tmp_path):
    engine, executor = positioned()
    engine.state_file = str(tmp_path / "state.json")
    engine._save = se.Strategy._save.__get__(engine)
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine.step(80.0)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["pending_exit"]["kind"] == "STOP"
    assert saved["orders"][0]["cancel_requested"]
    with patch.object(se, "state_path_for", return_value=engine.state_file):
        restarted = se.Strategy(executor, engine.pair, engine.p, dry_run=False)
    executor.next_status = OrderStatus("canceled", 0, 0, 0)
    restarted.reconcile(100.0)
    restarted.step(100.0)
    s = submits(executor)
    assert len(s) == 1 and s[0][6] == "STOP"


# ─── Surge Profit Floor Protection ──────────────────────────────────────────

def surge_position(overlay=False):
    engine, executor = positioned(
        trend_overlay=overlay, trend_trail_pct=50.0, trend_exit_break=False,
        tp_trail_pct=50.0, stop_loss_pct=0.0, surge_guard=True,
        surge_gain_pct=18.0, surge_move_pct=18.0, surge_window_hours=72.0,
        surge_exit_pullback_pct=3.5, surge_dynamic=False,
        fast_profit_guard=False, slow_grind_guard=False,
    )
    engine.s.update(trend_mode=overlay, surge_peak=120.0)
    engine._regime_context = lambda: (
        MarketRegimeDecision("sideways", 0.0, 0.01, 0.0, True, "sideways"),
        [80.0] * 30,
    )
    return engine, executor


def test_surge_exit_rechecks_profit_after_cancel_and_survives_restart(tmp_path):
    for overlay in (False, True):
        engine, executor = surge_position(overlay)
        state_file = str(tmp_path / f"state_{overlay}.json")
        engine.state_file = state_file
        engine._save = se.Strategy._save.__get__(engine)
        engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
        engine.step(115.8)
        assert submits(executor) == []
        saved = json.loads((tmp_path / f"state_{overlay}.json").read_text())
        assert saved["pending_exit"]["min_profit_pct"] == pytest.approx(12.75)
        assert saved["orders"][0]["cancel_requested"]

        engine.p.surge_gain_pct = 1.0
        with patch.object(se, "state_path_for", return_value=engine.state_file):
            restarted = se.Strategy(executor, engine.pair, engine.p, dry_run=False)
        executor.next_status = OrderStatus("canceled", 0, 0, 0)
        restarted.reconcile(96.5)
        restarted.step(96.5)
        assert submits(executor) == []
        assert restarted.s["pending_exit"]["min_profit_pct"] == pytest.approx(12.75)
        assert not restarted._has_open("buy")

        restarted.step(112.75)
        assert submits(executor) == []
        restarted.step(115.8)
        assert len(submits(executor)) == 1
        assert submits(executor)[0][2] == "sell"
        assert submits(executor)[0][5:7] == (True, "TP")


def test_surge_exit_uses_cost_basis_after_a_racing_buy_fill():
    for overlay in (False, True):
        engine, executor = surge_position(overlay)
        engine.s["orders"] = [order("BUY-1", "buy", price=120.0)]
        engine.step(115.8)
        executor.next_status = OrderStatus("canceled", 0.5, 60.0, 0.02)
        engine.reconcile(115.8)
        engine.step(115.8)
        assert engine._avg() == pytest.approx(160.0 / 1.5)
        assert submits(executor) == []
        engine.step(121.0)
        assert len(submits(executor)) == 1
        assert submits(executor)[0][3] == engine._dust_safe_qty(1.5)


def test_partial_surge_sell_remainder_keeps_floor_and_does_not_duplicate():
    engine, executor = surge_position()
    engine.step(115.8)
    assert len(submits(executor)) == 1
    engine.step(96.5)
    assert len(submits(executor)) == 1
    assert not any(call[0] == "cancel_order" for call in executor.calls)

    executor.next_status = OrderStatus("canceled", 0.4, 46.32, 0.02)
    engine.reconcile(96.5)
    engine.step(96.5)
    assert engine.s["qty"] == pytest.approx(0.6)
    assert engine.s["pending_exit"]["min_profit_pct"] == pytest.approx(12.75)
    assert len(submits(executor)) == 1
    engine.step(115.8)
    assert len(submits(executor)) == 2
    assert submits(executor)[1][3] == engine._dust_safe_qty(0.6)


def test_stop_loss_can_replace_a_pending_surge_profit_exit():
    engine, executor = surge_position()
    engine.p.stop_loss_pct = 12.5
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine.step(115.8)
    executor.next_status = OrderStatus("canceled", 0, 0, 0)
    engine.reconcile(80.0)
    engine.step(80.0)
    assert len(submits(executor)) == 1
    assert submits(executor)[0][6] == "STOP"
    assert "min_profit_pct" not in engine.s["pending_exit"]


def test_window_rally_or_old_armed_state_cannot_trigger_surge_at_a_loss():
    for overlay in (False, True):
        for already_armed in (False, True):
            engine, executor = surge_position(overlay)
            engine.s.update(
                surge_peak=100.0, surge_active=already_armed,
                trail_peak=100.0 if already_armed else None,
            )
            engine.step(100.0)
            assert not engine.s["surge_active"]
            engine.step(96.5)
            assert not engine.s["pending_exit"]
            assert not any(call[2] == "sell" and call[5] for call in submits(executor))


def test_invalid_persisted_profit_floor_blocks_submission():
    for stored_floor in [None, "invalid", -1.0, float("nan"), float("inf")]:
        engine, executor = surge_position()
        engine.s["pending_exit"] = {"kind": "TP", "min_profit_pct": stored_floor}
        engine.step(115.8)
        assert submits(executor) == []


def test_unavailable_cost_basis_blocks_a_pending_profit_exit():
    for cost in [0.0, float("nan"), float("inf")]:
        engine, executor = surge_position()
        engine.s.update(cost=cost, pending_exit={"kind": "TP", "min_profit_pct": 12.75})
        engine.step(115.8)
        assert submits(executor) == []


# ─── Intent-Aware Profit-Floor Guard & DCA Partial Funding ──────────────────

class ProfitFloorGuardTest(unittest.TestCase):
    def test_profit_floor_guard_behaviors(self):
        with self.subTest(msg="equal_cost_is_allowed_without_a_fee_profit_guarantee"):
            self.assertTrue(_make_guard_strategy()._place("sell", 1.0, 100.0, kind="TP"))

        with self.subTest(msg="unknown_cost_is_not_a_proven_profitable_exit"):
            s = _make_guard_strategy()
            s.s["qty"] = 0
            self.assertIsNone(s._avg())
            self.assertTrue(s._place("sell", 1.0, 95.0, kind="TP"))

        with self.subTest(msg="stop_exemption_is_by_intent_even_for_limit_orders"):
            self.assertTrue(_make_guard_strategy()._place("sell", 1.0, 90.0, kind="STOP"))

        with self.subTest(msg="rounding_can_turn_a_nominally_profitable_tp_into_equal_cost"):
            s = _make_guard_strategy()
            self.assertTrue(s._place("sell", 1.0, 100.004, kind="TP"))
            self.assertEqual(s.s["orders"][-1]["price"], 100.0)

        with self.subTest(msg="avg_is_the_reference"):
            self.assertEqual(_make_guard_strategy()._avg(), 100.0)

        with self.subTest(msg="limit_sell_below_avg_is_refused"):
            self.assertFalse(_make_guard_strategy()._place("sell", 1.0, 95.0, kind="TP"))

        with self.subTest(msg="limit_sell_above_avg_is_allowed"):
            self.assertTrue(_make_guard_strategy()._place("sell", 1.0, 105.0, kind="TP"))

        with self.subTest(msg="stop_below_avg_is_exempt"):
            self.assertTrue(_make_guard_strategy()._place("sell", 1.0, 90.0, kind="STOP", market=True))

        with self.subTest(msg="market_trailing_exit_below_avg_is_exempt"):
            self.assertTrue(_make_guard_strategy()._place("sell", 1.0, 90.0, kind="TP", market=True))

        with self.subTest(msg="buy_below_avg_is_unaffected"):
            self.assertTrue(_make_guard_strategy()._place("buy", 1.0, 90.0, kind="DCA"))


class DcaPartialFillTest(unittest.TestCase):
    def _dca_ready(self, free_quote):
        s = _make_guard_strategy(
            dca_amount=50.0, dca_drop_pct=2.0, stop_loss_pct=0.0,
            takeprofit_pct=1.9, max_budget=1000.0, max_dca_buys=10,
        )
        s.s.update(qty=10.0, cost=1000.0, last_buy_price=100.0, dca_buys=0, spent=0.0)
        s.client.free_balance.return_value = free_quote
        s.client.ohlc_closes.return_value = []
        return s

    @staticmethod
    def _dcas(s):
        return [o for o in s.s["orders"] if o.get("kind") == "DCA"]

    def test_dca_budget_and_balance_caps(self):
        with self.subTest(msg="full_dca_when_funded"):
            s = self._dca_ready(500.0)
            s.step(97.0, timestamp=0.0)
            self.assertEqual(len(self._dcas(s)), 1)
            self.assertAlmostEqual(self._dcas(s)[-1]["amount"], 50.0)

        with self.subTest(msg="dca_capped_to_available_when_short"):
            s = self._dca_ready(20.0)
            s.step(97.0, timestamp=0.0)
            self.assertEqual(len(self._dcas(s)), 1)
            self.assertAlmostEqual(self._dcas(s)[-1]["amount"], 20.0)

        with self.subTest(msg="dca_skipped_only_when_available_is_effectively_zero"):
            s = self._dca_ready(0.0)
            s.step(97.0, timestamp=0.0)
            self.assertEqual(len(self._dcas(s)), 0)

        with self.subTest(msg="unknown_or_invalid_balance_does_not_authorize_a_buy"):
            for balance in (None, MagicMock(), True, -1, float("nan"), float("inf")):
                with self.subTest(balance=balance):
                    s = self._dca_ready(balance)
                    s.step(97.0, timestamp=0.0)
                    self.assertEqual(self._dcas(s), [])

        with self.subTest(msg="balance_outage_does_not_submit_or_crash"):
            s = self._dca_ready(50.0)
            s.client.free_balance.side_effect = RuntimeError("offline")
            s.step(97.0, timestamp=0.0)
            self.assertEqual(self._dcas(s), [])

        with self.subTest(msg="remaining_cycle_budget_can_fund_a_smaller_dca"):
            s = self._dca_ready(500.0)
            s.s["spent"] = 985.0
            s.step(97.0, timestamp=0.0)
            o = self._dcas(s)[0]
            self.assertEqual(o["amount"], 15.0)
            self.assertLessEqual(o["vol"] * o["price"], 15.0)

        with self.subTest(msg="rounding_does_not_exceed_capped_balance"):
            s = self._dca_ready(20.0)
            s.vol_dec = 2
            s.step(97.0, timestamp=0.0)
            o = self._dcas(s)[0]
            self.assertLessEqual(o["vol"] * o["price"], 20.0)

        with self.subTest(msg="exact_nominal_balance_is_still_binding_after_price_rounding"):
            s = self._dca_ready(50.0)
            s.step(97.0, timestamp=0.0)
            o = self._dcas(s)[0]
            self.assertLessEqual(o["vol"] * o["price"], 50.0)


# ─── MDHTP Integration in SpotEngine ────────────────────────────────────────

def make_mdhtp_engine(*, mdhtp_enabled: bool = True, takeprofit_pct: float = 5.0, **params):
    defaults = dict(
        takeprofit_pct=takeprofit_pct,
        tp_trend_hold=True,
        tp_trail_pct=3.0,
        tp_trail_profit_floor_pct=1.0,
        tp_mdhtp_enabled=mdhtp_enabled,
        tp_stagnation_max_sec=1080.0,
        fast_profit_guard=True,
    )
    defaults.update(params)
    return positioned(**defaults)


def test_mdhtp_trend_ride_initiation_and_hold():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    assert engine.exit_manager is not None
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(se.time, "time", return_value=now):
        engine.step(105.5)

    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    state = engine.exit_manager.tp_ride_state.get(engine.pair)
    assert state is not None and state["peak_price"] == 105.5 and state["started_at"] == now


def test_mdhtp_peak_pullback_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(se.time, "time", return_value=now):
        engine.step(110.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    with patch.object(se.time, "time", return_value=now + 60.0):
        engine.step(106.15)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    assert engine.pair not in engine.exit_manager.tp_ride_state


def test_mdhtp_profit_floor_breach_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_trail_profit_floor_pct=1.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(se.time, "time", return_value=now):
        engine.step(105.5)
    with patch.object(se.time, "time", return_value=now + 30.0):
        engine.step(100.5)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_stagnation_timeout_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_stagnation_max_sec=1080.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(se.time, "time", return_value=now):
        engine.step(106.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    with patch.object(se.time, "time", return_value=now + 1070.0):
        engine.step(105.9)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    with patch.object(se.time, "time", return_value=now + 1085.0):
        engine.step(105.9)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_trend_reversal_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(se.time, "time", return_value=now):
        engine.step(107.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    engine.s["trend_mode"] = False
    with patch.object(se.time, "time", return_value=now + 60.0):
        engine.step(106.8)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_cycle_finalization_resets_tp_ride_state():
    engine, _ = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    engine.step(106.0)
    assert engine.pair in engine.exit_manager.tp_ride_state

    engine.s["qty"] = 0.0
    engine.s["orders"] = []
    engine._finalize_cycle_if_flat(o={"kind": "TP"}, price=106.0)
    assert engine.pair not in engine.exit_manager.tp_ride_state
    assert engine.exit_manager.tp_ride_state == engine.s["tp_ride_state"]


def test_mdhtp_intrabar_high_arms_and_exits_on_same_bar_pullback():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_trail_pct=3.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(se.time, "time", return_value=now):
        # Intrabar high=106.0 (+6% >= 5% TP) arms MDHTP at peak 106.0,
        # and close=102.0 (-3.77% pullback from 106.0 >= 3% trail) triggers exit in the same bar.
        engine.step(102.0, high=106.0)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    assert engine.pair not in engine.exit_manager.tp_ride_state



# ─── MHDSL Integration in SpotEngine ────────────────────────────────────────

def make_mhdsl_engine(*, mhdsl_enabled: bool = True, stop_loss_pct: float = 10.0, **params):
    defaults = dict(
        stop_loss_pct=stop_loss_pct,
        sl_mhdsl_enabled=mhdsl_enabled,
        sl_disaster_buffer_ratio=0.30,
        sl_disaster_vol_multiplier=0.85,
        sl_disaster_buffer_min_pct=0.80,
        sl_trend_defer_max_sec=220.0,
        sl_bounce_min_recovery_pct=0.40,
        sl_bounce_pullback_pct=0.30,
        sl_bounce_ratchet_enabled=True,
        sl_bounce_ratchet_k=0.50,
        sl_bounce_min_pullback_pct=0.15,
        sl_structural_break_enabled=True,
    )
    defaults.update(params)
    return positioned(**defaults)


def test_mhdsl_disabled_preserves_legacy_stop():
    engine, executor = make_mhdsl_engine(mhdsl_enabled=False, stop_loss_pct=10.0)
    assert engine.exit_manager is None
    engine.s["trend_mode"] = True
    assert engine._check_stop_loss(89.0) is True
    assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_hard_disaster_circuit_breaker():
    engine, executor = make_mhdsl_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    assert engine.exit_manager is not None
    engine.s["trend_mode"] = True
    assert engine._check_stop_loss(86.0) is True
    assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)
    assert engine.pair not in engine.s.get("sl_bounce_state", {})


def test_mhdsl_bounce_watch_initiated_on_uptrend():
    engine, executor = make_mhdsl_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    executor.trend = lambda sym: "up"
    assert engine._check_stop_loss(89.5) is False
    assert not submits(executor)
    bounce_state = engine.s["sl_bounce_state"].get(engine.pair)
    assert bounce_state is not None
    assert bounce_state["lowest_price"] == 89.5 and bounce_state["peak_bounce_price"] == 89.5


def test_mhdsl_bounce_structural_break_and_ratchet():
    engine, executor = make_mhdsl_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    executor.trend = lambda sym: "up"
    now = time.time()

    engine._shadow_prices.append((now, 89.0))
    assert engine._check_stop_loss(89.0) is False
    assert engine.s["sl_bounce_state"][engine.pair]["lowest_price"] == 89.0

    engine._shadow_prices.append((now + 10, 90.0))
    assert engine._check_stop_loss(90.0) is False
    assert engine.s["sl_bounce_state"][engine.pair]["peak_bounce_price"] == 90.0

    engine._shadow_prices.append((now + 20, 89.6))
    assert engine._check_stop_loss(89.6) is True
    assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_bounce_timeout():
    engine, executor = make_mhdsl_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0, sl_trend_defer_max_sec=220.0,
    )
    engine.replay_mode = True
    executor.trend = lambda sym: "up"
    start_time = 1000.0

    engine._shadow_prices.append((start_time, 89.5))
    assert engine._check_stop_loss(89.5) is False

    engine._shadow_prices.append((start_time + 200.0, 89.5))
    assert engine._check_stop_loss(89.5) is False

    engine._shadow_prices.append((start_time + 225.0, 89.5))
    assert engine._check_stop_loss(89.5) is True
    assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_full_recovery_clears_bounce_state():
    engine, executor = make_mhdsl_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    executor.trend = lambda sym: "up"
    assert engine._check_stop_loss(89.5) is False
    assert engine.pair in engine.s["sl_bounce_state"]

    assert engine._check_stop_loss(92.0) is False
    assert engine.pair not in engine.s["sl_bounce_state"]


def test_mhdsl_volatility_scaling_variant_b():
    engine, executor = make_mhdsl_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0,
        sl_disaster_buffer_ratio=0.30, sl_disaster_vol_multiplier=0.85,
    )
    executor.trend = lambda sym: "up"
    engine._vol_1h_ratio = lambda: (0.05, "mock_test_vol")

    assert engine._check_stop_loss(86.5) is False
    assert engine._check_stop_loss(85.5) is True
    assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_cycle_finalization_resets_state():
    engine, _ = make_mhdsl_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    engine.s["sl_bounce_state"][engine.pair] = {"mock": "data"}
    engine.s["qty"] = 0.0
    engine._finalize_cycle_if_flat({"kind": "STOP", "side": "sell", "vol": 1.0, "price": 85.0}, 85.0)
    assert engine.s["sl_bounce_state"] == {}
    assert engine.exit_manager.sl_bounce_state == {}


def test_mhdsl_reentry_mirror_post_stop_bounce_and_ratchet():
    engine, executor = make_mhdsl_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0, reentry_sl_bounce_pct=1.5,
    )
    engine.s.update(qty=0.0, last_sell_price=90.0, last_exit_kind="STOP", last_sell_ts=time.time())
    engine.step(88.0)
    assert not any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)

    engine.step(89.4)
    assert any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)


def test_mhdsl_reentry_mirror_bull_trend_pullback():
    engine, executor = make_mhdsl_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0,
        reentry_pullback_pct=2.0, reentry_hybrid_enabled=True,
    )
    engine.s.update(qty=0.0, last_sell_price=100.0, last_exit_kind="TP", last_sell_ts=time.time())
    executor.trend = lambda sym: "up"

    engine.step(105.0)
    assert not any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)

    engine.step(102.8)
    assert any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)


def test_mhdsl_blocks_dca_during_bounce_deferral():
    engine, executor = make_mhdsl_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0, dca_drop_pct=5.0, max_dca_buys=3,
    )
    engine.s.update(qty=1.0, entry_price=100.0, last_buy_price=100.0, dca_buys=0, trend_mode=True)
    executor.trend = lambda sym: "up"

    engine.step(89.0)
    assert engine.pair in engine.exit_manager.sl_bounce_state
    assert not [c for c in executor.calls if c[0] == "submit_order" and c[2] == "buy"]


def test_spot_engine_vol_1h_ratio_explicit_conversion():
    engine, _ = make_mhdsl_engine()

    engine._trail_vol_1h = lambda: 0.05
    ratio, src = engine._vol_1h_ratio()
    assert abs(ratio - 0.0005) < 1e-7 and src == "trail_ohlc"

    engine._trail_vol_1h = lambda: None
    engine._shadow_vol_1h = lambda: 0.08
    ratio, src = engine._vol_1h_ratio()
    assert abs(ratio - 0.0008) < 1e-7 and src == "shadow_ticks"

    engine._shadow_vol_1h = lambda: None
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "cache_kalman_trend.json"), "w", encoding="utf-8") as f:
            json.dump({engine.pair.upper(): {"vol_1h_pct": 0.04}}, f)
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": tmp_dir}):
            ratio, src = engine._vol_1h_ratio()
            assert abs(ratio - 0.0004) < 1e-7 and src == "cache"


def test_replay_mode_ignores_external_cached_volatility():
    engine, _ = make_mhdsl_engine()
    engine.replay_mode = True
    engine._trail_vol_1h = lambda: None
    engine._shadow_vol_1h = lambda: None

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "cache_kalman_trend.json"), "w", encoding="utf-8") as f:
            json.dump({engine.pair.upper(): {"vol_1h_pct": 5.0}}, f)
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": tmp_dir}):
            ratio, src = engine._vol_1h_ratio()
            assert ratio == 0.0 and src == "default"


def test_mdhtp_not_in_tp_territory_clears_trail_peak():
    engine, _ = make_mhdsl_engine(
        mhdsl_enabled=True, tp_mdhtp_enabled=True, tp_trend_hold=True, takeprofit_pct=5.0,
    )
    engine.s.update(qty=1.0, cost=100.0, spent=100.0, trail_peak=106.0, trail_stop=104.0)
    engine.step(101.0)
    assert engine.s["trail_peak"] is None
    assert engine.s["trail_stop"] is None


def test_ledger_reconcile_flat_resets_trend_mode_and_allows_reentry():
    engine, executor = positioned(trend_overlay=True)
    engine.s.update(
        qty=1.0, cost=100.0, spent=100.0, dca_buys=2,
        trend_mode=True, trend_peak=115.0, trail_peak=115.0,
        surge_active=True, surge_peak=120.0,
        ledger_reconcile_required=True,
    )
    executor.free_balance = lambda _asset: 0.0
    engine.step(100.0)
    assert engine.s["qty"] == 0.0
    assert engine.s["trend_mode"] is False
    assert engine.s["trend_peak"] is None
    assert engine.s["dca_buys"] == 0
    assert engine.s["trail_peak"] is None
    assert engine.s["surge_active"] is False


def test_mdhtp_soft_floor_block_does_not_deadlock_subsequent_profit_floor_breach():
    engine, executor = make_mdhtp_engine(
        mdhtp_enabled=True, tp_dynamic_flat=False, takeprofit_pct=1.2,
        tp_trail_pct=0.5, tp_trail_profit_floor_pct=1.0,
    )
    engine.s["trend_mode"] = True
    now = 1000.0
    # Tick 1 at 101.55 (+1.55% >= 1.2% TP): arms MDHTP ride at peak 101.55
    with patch.object(se.time, "time", return_value=now):
        engine.step(101.55)
    assert not submits(executor)

    # Tick 2 at 101.04: pullback from 101.55 is 0.502% >= 0.5% trail, gain +1.04% >= 1.0% floor,
    # but exit_px = round(101.04 * 0.999, 2) = 100.94 < profit_floor (101.00).
    # Soft trailing is blocked without locking pending_exit or losing tp_ride_state.
    with patch.object(se.time, "time", return_value=now + 10.0):
        engine.step(101.04)
    assert not submits(executor)
    assert engine.s.get("pending_exit") is None
    assert engine.pair in engine.exit_manager.tp_ride_state

    # Tick 3 at 100.50 (+0.50% < 1.0% floor): mdhtp_profit_floor_breached must execute immediately!
    with patch.object(se.time, "time", return_value=now + 20.0):
        engine.step(100.50)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_hold_evaluates_advanced_profit_guards():
    engine, executor = make_mdhtp_engine(
        mdhtp_enabled=True, takeprofit_pct=5.0, tp_trail_pct=10.0,
        surge_guard=True, surge_gain_pct=15.0, surge_move_pct=15.0,
        surge_window_hours=72.0, surge_exit_pullback_pct=2.0, surge_dynamic=False,
    )
    engine.s.update(trend_mode=True, surge_peak=125.0, surge_active=True)
    now = 1000.0
    # At 120.0 (+20% >= 5% TP), MDHTP pullback from 120.0 is 0% (< 10% trail -> HOLD),
    # but Surge Guard pullback from surge_peak 125.0 is 4.0% (>= 2.0% -> SELL)!
    with patch.object(se.time, "time", return_value=now):
        engine.step(120.0)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_post_stop_reentry_tracks_peak_bounce_during_cooldown_and_blocks_rollover():
    mgr = PositionExitManager(
        config=PositionExitConfig(
            reentry_enabled=True,
            reentry_sl_bounce_pct=0.04,
            reentry_post_stop_cooldown_sec=600.0,
            reentry_ratchet_enabled=True,
        )
    )
    # T1 (100s after STOP at 100): price hits trough 90.0 during cooldown
    d1 = mgr.evaluate_reentry(ReentrySnapshot(
        symbol="SOLUSDC", current_price=90.0, last_sell_price=100.0,
        last_exit_kind="STOP", current_time_s=1100.0, last_sell_time_s=1000.0,
    ))
    assert d1.action == ReentryAction.WAIT
    assert "post_stop_stabilization_cooldown" in d1.reason

    # T2 (300s after STOP, still in cooldown): price spikes to 100.0 (+11.1% from trough 90)
    d2 = mgr.evaluate_reentry(ReentrySnapshot(
        symbol="SOLUSDC", current_price=100.0, last_sell_price=100.0,
        last_exit_kind="STOP", current_time_s=1300.0, last_sell_time_s=1000.0,
    ))
    assert d2.action == ReentryAction.WAIT
    assert mgr.reentry_state["SOLUSDC"]["peak_bounce"] == 100.0

    # T3 (700s after STOP, cooldown expired): price has rolled over from 100.0 to 94.0
    # (+4.44% >= +4% bounce from 90.0, but -6.0% rollover > 2.0% allowed_rollover)
    d3 = mgr.evaluate_reentry(ReentrySnapshot(
        symbol="SOLUSDC", current_price=94.0, last_sell_price=100.0,
        last_exit_kind="STOP", current_time_s=1700.0, last_sell_time_s=1000.0,
    ))
    assert d3.action == ReentryAction.WAIT
    assert "post_stop_bounce_rollover" in d3.reason


def test_spot_engine_reentry_gate_resolves_bear_regime_and_restores_state_on_blocked_place():
    engine, executor = positioned(trend_overlay=False, reentry_hybrid_enabled=False)
    engine.p.reentry_sl_bounce_pct = 2.0
    engine.p.reentry_bear_bounce_pct = 0.0
    engine.s.update(
        qty=0.0, cost=0.0, spent=0.0,
        last_sell_price=100.0, last_exit_kind="STOP",
        post_sell_trough=90.0, post_sell_peak=100.0, sl_low=90.0,
        last_sell_ts=1000.0,
    )
    bear_decision = MarketRegimeDecision(
        "bear", -0.5, 0.02, 3.5, True, "bear",
        n_samples=24, horizon="long",
    )
    engine._regime_context = lambda: (bear_decision, [100.0] * 24)

    # Even with trend_overlay=False and reentry_hybrid_enabled=False, Bear Gate blocks re-entry at 93.0 (+3.3% > 2%)
    engine.step(93.0, timestamp=2000.0)
    assert not submits(executor)
    assert engine.pair in engine.exit_manager.reentry_state

    # Now switch regime to sideways, but exhaust budget so step() refuses to place ENTRY after gate approval
    sideways_decision = MarketRegimeDecision(
        "sideways", 0.0, 0.01, 0.0, True, "sideways",
        n_samples=24, horizon="long",
    )
    engine._regime_context = lambda: (sideways_decision, [100.0] * 24)
    engine.s["spent"] = engine.p.allocated_budget()
    engine.step(93.0, timestamp=2060.0)
    assert not submits(executor)
    # Re-entry state must be restored because no ENTRY order was placed!
    assert engine.pair in engine.exit_manager.reentry_state
    assert engine.exit_manager.reentry_state[engine.pair]["trough"] == 90.0


def test_spot_rules_non_positive_drop_thresholds_do_not_false_trigger():
    trig, _, _ = spot_rules.check_fast_profit_reversal(
        current_price=120.0, avg_cost=100.0, base_tp_pct=5.0, mult=2.0,
        shadow_prices=[(990.0, 120.0)], window_sec=300.0, drop_pct=0.0, current_time=1000.0,
    )
    assert trig is False

    s_trig, _ = spot_rules.check_parabolic_surge_exhaustion(
        current_price=125.0, avg_cost=100.0, surge_peak=125.0,
        surge_gain_pct=18.0, exit_pullback_pct=0.0, surge_active=True,
    )
    assert s_trig is False

    g_trig, _ = spot_rules.check_slow_grind_exhaustion(
        current_price=120.0, avg_cost=100.0, entry_ts=0.0, current_ts=10 * 86400.0,
        min_days=7.0, min_gain_pct=15.0, recent_peak=120.0,
        shadow_prices=[(10 * 86400.0 - 60.0, 120.0)], flash_window_sec=900.0,
        flash_drop_pct=0.0, structural_drop_pct=0.0, sma_value=None,
    )
    assert g_trig is False


def test_position_exit_manager_resets_tp_and_sl_state_when_buy_price_changes():
    mgr = PositionExitManager(config=PositionExitConfig(gain_threshold=0.03, lost_threshold=0.05))
    # Initiate TP trend ride at avg_buy_price=100.0, current_price=104.0
    d1 = mgr.evaluate_take_profit(PositionSnapshot(
        symbol="ETHUSDC", current_price=104.0, avg_buy_price=100.0,
        net_qty=1.0, sellable_qty=1.0, current_time_s=1000.0, trend_up=True,
    ))
    assert d1.action == ExitAction.HOLD
    assert "ETHUSDC" in mgr.tp_ride_state

    # Now a new buy raises avg_buy_price to 105.0 while current_price is 104.0 (below new cost, above SL).
    # Must NOT panic-sell via mdhtp_profit_floor_breached!
    d2 = mgr.evaluate_take_profit(PositionSnapshot(
        symbol="ETHUSDC", current_price=104.0, avg_buy_price=105.0,
        net_qty=2.0, sellable_qty=2.0, current_time_s=1010.0, trend_up=True,
    ))
    assert d2.action == ExitAction.HOLD
    assert d2.reason == "not_in_tp_territory"
    assert "ETHUSDC" not in mgr.tp_ride_state


def test_spot_engine_mdhtp_intrabar_high_with_trend_not_up_does_not_false_sell_below_tp():
    engine, executor = make_mdhtp_engine(
        mdhtp_enabled=True, tp_dynamic_flat=False, takeprofit_pct=3.0,
        tp_trail_pct=0.5, tp_trail_profit_floor_pct=1.0,
    )
    engine.s["trend_mode"] = False
    executor.trend = lambda _sym: "down"
    # Bar wicks to high=104.0 (>= 3% TP) while trend is NOT up, but closes at price=99.5 (< avg=100.0).
    # Must NOT sell at 99.5 on a false TP decision!
    engine.step(99.5, high=104.0)
    assert not submits(executor)


def test_spot_engine_pending_stop_exit_does_not_reinitiate_mhdsl_bounce_watch():
    engine, executor = make_mhdsl_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0, sl_trend_defer_max_sec=220.0,
    )
    engine.dry_run = False
    executor.trend = lambda _sym: "up"
    engine.s["orders"] = [order("TP-1", "sell", price=110.0, kind="TP")]
    engine.s["pending_exit"] = {"kind": "STOP", "soft_floor": False}

    # On tick where TP-1 is reconciled as canceled, step(89.0) must execute the pending STOP market exit
    # without re-running evaluate_stop_loss or re-populating sl_bounce_state!
    executor.order_status = lambda _sym, _oid: OrderStatus("canceled", 0.0, 0.0, 0.0)
    engine.reconcile(89.0)
    engine.step(89.0)
    s = submits(executor)
    assert len(s) == 1 and s[0][6] == "STOP"
    assert engine.pair not in engine.exit_manager.sl_bounce_state


def test_reconcile_venue_ledger_uses_remaining_qty_on_partial_open_sell_and_latest_buy_fill():
    assert extract_remaining_qty({"origQty": 1.0, "executedQty": 0.4}) == pytest.approx(0.6)
    assert extract_remaining_qty({"remaining_qty": 0.35, "origQty": 1.0}) == pytest.approx(0.35)

    engine, executor = positioned()
    engine.observe_only = True
    engine.s["qty"] = 0.6
    engine.s["cost"] = 60.0
    engine.s["spent"] = 60.0
    engine.s["ledger_reconcile_required"] = True

    # 1) Partial open SELL (origQty=1.0, executedQty=0.4 -> remaining=0.6) adds 0.6 (not 1.0) to free_balance=0.6
    raw_exec = SimpleNamespace(
        open_orders=lambda _pair: [{"side": "SELL", "origQty": 1.0, "executedQty": 0.4}],
        get_orders=lambda _pair, _side, _lookback: [
            {"price": 112.0, "timestamp": 1700000200000},
            {"price": 95.0, "timestamp": 1700000100000},
        ],
    )
    executor._executor = raw_exec
    executor.free_balance = lambda _base: 0.6

    assert engine._reconcile_venue_ledger(110.0, 1700000250.0, True) is True
    assert engine.s["qty"] == pytest.approx(1.2)

    # 2) External BUY when free_balance increases to 1.5 (+0.6 open SELL = 2.1 total vs held=1.2)
    # picks the newest BUY fill (112.0 @ t=1700000200000) even when get_orders returns descending order
    executor.free_balance = lambda _base: 1.5
    assert engine._reconcile_venue_ledger(110.0, 1700000260.0, True) is True
    assert engine.s["qty"] == pytest.approx(2.1)
    assert engine.s["last_buy_price"] == pytest.approx(112.0)
    assert engine.s["cost"] == pytest.approx(120.0 + 0.9 * 112.0)


def test_ledger_reconcile_exit_kind_and_partial_fill_audit_notional(tmp_path, monkeypatch):
    # 1) B10: _reconcile_venue_ledger infers "SL" when external close is below avg_cost, "TP" otherwise
    eng_loss, exec_loss = positioned()
    eng_loss.s.update(qty=1.0, cost=100.0, spent=100.0, ledger_reconcile_required=True)
    exec_loss.free_balance = lambda _base: 0.0
    assert eng_loss._reconcile_venue_ledger(92.0, 1700000300.0, True) is True
    assert eng_loss.s["last_exit_kind"] == "SL"

    eng_gain, exec_gain = positioned()
    eng_gain.s.update(qty=1.0, cost=100.0, spent=100.0, ledger_reconcile_required=True)
    exec_gain.free_balance = lambda _base: 0.0
    assert eng_gain._reconcile_venue_ledger(108.0, 1700000300.0, True) is True
    assert eng_gain.s["last_exit_kind"] == "TP"

    # 2) B6: _resolve_intent_execution_details preserves full order notional on PARTIALLY_FILLED
    intent = {
        "venue": "binance",
        "symbol": "BTCUSDC",
        "decision": "BUY",
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "clamped_notional_usd": 100.0,
        "limit_price": 80000.0,
        "execution_payload": {
            "orderId": "555",
            "status": "PARTIALLY_FILLED",
            "price": "80000.0",
            "origQty": "0.00125",
            "executedQty": "0.00025",
            "cummulativeQuoteQty": "20.0",
        },
    }
    details = _resolve_intent_execution_details(intent, 1700000300.0, lambda d, default=None: default)
    assert details["exec_status"] == "PARTIALLY_FILLED"
    assert details["filled_notional_usd"] == 20.0
    assert details["notional_usd"] == 100.0

    # 3) B6(b): cancel_existing_ai_open_orders cleans order_retry even if open_orders() returns None
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "LOCK_FILE", str(l_file))
    rid = oq.enqueue("BTCUSDC", "BUY", 0.001, {"owner": "ai_gateway"}, requested_price=80000.0, now=1000.0)
    assert rid is not None
    fake_inst = SimpleNamespace(symbol="BTCUSDC", open_orders=lambda: None, cancel_order=lambda _oid: True)
    cancel_existing_ai_open_orders(fake_inst, "BUY", [])
    assert oq.load_all() == []

    # 4) B7: T212Provider declares list_open_orders=True
    assert T212Provider().reconciliation_capabilities().list_open_orders is True

    # 5) B8: order_retry.enqueue RETRY_DEDUP shifts ttl_started_ts when prev_attempts > 0
    monkeypatch.setattr(oq, "RETRY_DEDUP", True)
    r_dedup = oq.enqueue(
        "ETHUSDC", "BUY", 0.1, {}, requested_price=3000.0,
        now=1000.0, created_ts=1000.0, attempts=1, last_attempt_ts=1010.0,
        failure_reason="trend_deferred",
    )
    recs = oq.load_all()
    recs[0]["safe_to_discard"] = True
    oq.rewrite(recs)
    oq.enqueue(
        "ETHUSDC", "BUY", 0.1, {}, requested_price=3005.0,
        now=1050.0, created_ts=1000.0, attempts=1, last_attempt_ts=1050.0,
        failure_reason="network_timeout",
    )
    updated_rec = oq.get(r_dedup)
    assert updated_rec["ttl_started_ts"] == pytest.approx(1000.0 + (1050.0 - 1010.0))

    # 6) B11: bapi.cancel_expired_orders increments count only when _cancel_and_resolve succeeds
    monkeypatch.setattr(bapi_mod.sym, "validate_params", lambda _t, _s: None)
    monkeypatch.setattr(
        bapi_mod,
        "get_open_orders",
        lambda _t, _s, allowed_owners=None: {
            "101": {"time": 1000},
            "102": {"time": 1000},
        },
    )
    monkeypatch.setattr(bapi_mod, "_cancel_and_resolve", lambda _s, oid: oid == "101")
    out_buf = []
    monkeypatch.setattr("builtins.print", lambda *a, **_k: out_buf.append(" ".join(str(x) for x in a)))
    bapi_mod.cancel_expired_orders("BUY", "BTCUSDC", 10.0)
    assert any("Cancelled 1 orders" in line for line in out_buf)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_strategy__order_from_pending_honors_submitted_qty_and_scales_amount_proportionally_iter10_s4():
    """[ITER10 #4] Strategy._order_from_pending honors submitted_qty and scales amount proportionally."""
    # 4. Strategy._order_from_pending honors submitted_qty and scales amount proportionally
    dummy_strat = SimpleNamespace(_now=lambda: 1700000000.0)
    order_rec = Strategy._order_from_pending(
        dummy_strat,
        {
            "order_id": "tx-123",
            "side": "BUY",
            "requested_qty": 2.0,
            "submitted_qty": 1.0,
            "requested_price": 100.0,
            "amount": 200.0,
            "kind": "B0",
            "market": False,
            "intent_id": "id-1",
            "created_at": 1700000000.0,
        },
    )
    assert order_rec["vol"] == pytest.approx(1.0)
    assert order_rec["amount"] == pytest.approx(100.0)

def test_spot_rules_nan_hardening_iter26_s1():
    """[ITER26 #1] spot_rules NaN hardening."""
    # 1. spot_rules NaN hardening
    trig_sg, reason_sg = spot_rules.check_slow_grind_exhaustion(
        current_price=120.0,
        avg_cost=100.0,
        entry_ts=float("nan"),
        current_ts=1_700_000_000.0,
        min_days=7.0,
        min_gain_pct=15.0,
        recent_peak=130.0,
        shadow_prices=[(1_700_000_000.0, 130.0)],
        flash_window_sec=900.0,
        flash_drop_pct=1.5,
        structural_drop_pct=2.5,
    )
    assert trig_sg is False and reason_sg == ""

    trig_ps, reason_ps = spot_rules.check_parabolic_surge_exhaustion(
        current_price=95.0,
        avg_cost=100.0,
        surge_peak=125.0,
        surge_gain_pct=18.0,
        exit_pullback_pct=5.0,
        surge_active=True,
        min_profit_pct=float("nan"),
    )
    assert trig_ps is False and reason_ps == ""

    trig_fp, gain_fp, drop_fp = spot_rules.check_fast_profit_reversal(
        current_price=115.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(1000.0, float("nan")), (1005.0, 120.0)],
        window_sec=300.0,
        drop_pct=3.0,
        current_time=1010.0,
    )
    assert trig_fp is True
    assert drop_fp > 3.0

def test_strategy__placement_backoff_remaining_clamping_nan_safety_iter26_s2():
    """[ITER26 #2] Strategy._placement_backoff_remaining clamping & NaN safety."""
    # 2. Strategy._placement_backoff_remaining clamping & NaN safety
    dummy_strat = SimpleNamespace(
        s={
            "placement_backoffs": {
                "buy:DCA": {"until": 1_700_000_000.0 + 999999.0},
                "sell:TP": {"until": float("nan")},
            }
        },
        _now=lambda: 1_700_000_000.0,
        _placement_backoff_key=Strategy._placement_backoff_key,
    )
    rem_dca = Strategy._placement_backoff_remaining(dummy_strat, "BUY", "DCA")
    assert rem_dca == _PLACEMENT_BACKOFF_CAP_SEC
    rem_tp = Strategy._placement_backoff_remaining(dummy_strat, "SELL", "TP")
    assert rem_tp == 0.0

def test_spot_rules_hit_stop_and_dca_price_hit_reject_false_true_nan_iter45_s1():
    """[ITER45 #1] spot_rules: hit_stop and dca_price_hit reject 0.0 / False / True / NaN prices."""
    # 1. spot_rules: hit_stop and dca_price_hit reject 0.0 / False / True / NaN prices
    assert spot_rules.hit_stop(100.0, 0.0, 5.0) is False
    assert spot_rules.hit_stop(100.0, False, 5.0) is False
    assert spot_rules.hit_stop(100.0, True, 5.0) is False
    assert spot_rules.hit_stop(100.0, float("nan"), 5.0) is False
    assert spot_rules.hit_stop(100.0, 94.0, 5.0) is True

    assert spot_rules.dca_price_hit(0.0, 100.0, 5.0, 0.05) is False
    assert spot_rules.dca_price_hit(False, 100.0, 5.0, 0.05) is False
    assert spot_rules.dca_price_hit(True, 100.0, 5.0, 0.05) is False
    assert spot_rules.dca_price_hit(float("nan"), 100.0, 5.0, 0.05) is False
    assert spot_rules.dca_price_hit(95.0, 100.0, 5.0, 0.05) is True

    assert spot_rules.are_close(1.0, True, 0.05) is False
    assert spot_rules.are_close(100.0, 100.0, float("nan")) is False
    assert spot_rules.progressive_dca_drop_pct(float("nan"), True, float("nan")) == 0.0
    assert spot_rules.parabolic_surge_profit_floor_pct(float("nan"), True, float("nan")) == pytest.approx(20.2)

def test_spot_rules_handles_malformed_shadow_prices_entries_and_string_numbers_iter64_s3():
    """[ITER64 #3] spot_rules handles malformed shadow_prices entries and string numbers in check_parabolic_surge_exhaustion."""
    # 3. spot_rules handles malformed shadow_prices entries and string numbers in check_parabolic_surge_exhaustion
    trig, gain, drop = spot_rules.check_fast_profit_reversal(
        current_price=110.0,
        avg_cost=100.0,
        base_tp_pct=4.0,
        mult=2.0,
        shadow_prices=[("bad", 115.0), (1000.0, "115.0"), ("not-a-tuple",)],
        window_sec=60.0,
        drop_pct=2.0,
        current_time=1010.0,
    )
    assert trig is True and gain == pytest.approx(10.0)
    surge_trig, _ = spot_rules.check_parabolic_surge_exhaustion(
        current_price="125.0",
        avg_cost="100.0",
        surge_peak="135.0",
        surge_gain_pct="24.0",
        exit_pullback_pct="3.2",
        surge_move_pct=None,
    )
    assert surge_trig is True

def test_spot_rules_entry_price_tp_price_check_parabolic_surge_exhaustion_and_check_fast_profit_reversal_iter84_s1():
    """[ITER84 #1] spot_rules entry_price, tp_price, check_parabolic_surge_exhaustion, and check_fast_profit_reversal."""
    # 1. spot_rules entry_price, tp_price, check_parabolic_surge_exhaustion, and check_fast_profit_reversal
    assert spot_rules.entry_price("100.0", "5.0") == 95.0
    assert spot_rules.entry_price(float("nan"), 5.0) == 0.0
    assert spot_rules.tp_price("100.0", "5.0") == 105.0
    assert spot_rules.tp_price(False, 5.0) == 0.0

    # Disabled surge_gain_pct <= 0 must not trigger via window_move_pct
    triggered, _ = spot_rules.check_parabolic_surge_exhaustion(
        current_price=105.0,
        avg_cost=100.0,
        surge_peak=110.0,
        surge_gain_pct=0.0,
        exit_pullback_pct=3.0,
        window_move_pct=30.0,
        surge_move_pct=25.0,
    )
    assert triggered is False

    # Future timestamps in shadow_prices must not inflate local_peak in check_fast_profit_reversal
    rev_trig, _, _ = spot_rules.check_fast_profit_reversal(
        current_price=110.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(990.0, 110.5), (9999.0, 200.0)],
        window_sec=60.0,
        drop_pct=5.0,
        current_time=1000.0,
    )
    assert rev_trig is False

def test_strategies_spot_replay_ohlc_validation_params_isolation_when_dca_drop_series_iter93_s1():
    """[ITER93 #1] strategies.spot_replay: OHLC validation & params isolation when dca_drop_series / reentry_series is used."""
    # 1. strategies.spot_replay: OHLC validation & params isolation when dca_drop_series / reentry_series is used
    orig_params = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.2,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        check_minutes=1.0,
        takeprofit_pct=2.0,
        max_budget=1000.0,
        max_dca_buys=5,
        enable_takeprofit=True,
        order_ttl_min=10.0,
        reentry_drop_pct=1.0,
    )
    ohlc = [
        (100.0, 101.0, 99.0, 100.0, 1000),
        (100.0, 102.0, 95.0, 98.0, 1060),
    ]
    run_replay(
        ohlc,
        orig_params,
        bar_minutes=float(orig_params.trend_interval),
        execution=ExecutionModel(intrabar_policy="worst_case"),
        dca_drop_series=[3.5, 4.5],
        reentry_series=[2.5, 3.5],
    )
    # Caller's params must not be mutated in-place by dynamic series!
    assert orig_params.dca_drop_pct == pytest.approx(2.0)
    assert orig_params.reentry_drop_pct == pytest.approx(1.0)

    with pytest.raises(ValueError, match="non-finite or invalid prices"):
        run_replay(
            [(100.0, 98.0, 99.0, 100.0, 1000)],
            orig_params,
            bar_minutes=float(orig_params.trend_interval),
        )

def test_strategies_spot_engine__parse_tranches_rejects_booleans_iter93_s2():
    """[ITER93 #2] strategies.spot_engine._parse_tranches rejects booleans."""
    # 2. strategies.spot_engine._parse_tranches rejects booleans
    with pytest.raises(ValueError, match="Invalid STRAT_TP_TRANCHES"):
        _parse_tranches(True)

def test_spot_rules_string_false_boolean_flag_handling_iter112_s5():
    """[ITER112 #5] spot_rules string 'false' boolean flag handling."""
    # 5. spot_rules string 'false' boolean flag handling
    assert spot_rules.dynamic_trend_trail_pct(5.0, base_trail_pct=8.0, min_trail_pct=3.0, surge_guard_active="false") == 8.0

def test_spot_replay_align_spot_interval_params_with_sub_minute_bar_minutes_and_run_replay_iter112_s6():
    """[ITER112 #6] spot_replay: align_spot_interval_params with sub-minute bar_minutes < 1.0 and run_replay bad initial_cash."""
    # 6. spot_replay: align_spot_interval_params with sub-minute bar_minutes < 1.0 and run_replay bad initial_cash
    dummy_params = SimpleNamespace(trend_interval=60, tp_trail_vol_interval=60, dca_vol_interval=60, regime_min_samples=20)
    assert srep.align_spot_interval_params(dummy_params, 0.5) is dummy_params

def test_spot_engine_state_path_for_handles_bool_punctuation_only_and_path_iter119_s1(tmp_path):
    """[ITER119 #1] spot_engine.state_path_for handles bool, punctuation-only, and Path state_dir."""
    # 1. spot_engine.state_path_for handles bool, punctuation-only, and Path state_dir
    assert se.state_path_for(True).endswith(".state_unknown.json")
    assert se.state_path_for("---", tmp_path) == os.path.join(str(tmp_path), ".state_unknown.json")

def test_spot_engine__apply_fill_handles_order_with_amount_none_by_iter119_s2():
    """[ITER119 #2] spot_engine._apply_fill handles order with amount=None by falling back to vol * price."""
    # 2. spot_engine._apply_fill handles order with amount=None by falling back to vol * price
    dummy_exec = SimpleNamespace(pair_precision=lambda _p: PairPrecision(2, 4, 0.01, "HYPE"))
    params = se.StratParams(
        currency="USD", entry_amount=100.0, entry_discount_pct=1.0, dca_amount=50.0,
        dca_drop_pct=2.0, check_minutes=1.0, takeprofit_pct=2.0, max_budget=500.0,
        max_dca_buys=5, enable_takeprofit=True, order_ttl_min=60.0,
    )
    strat = se.Strategy(dummy_exec, "HYPEUSD", params, dry_run=True, replay_mode=True, initial_state=se._new_state())
    strat._apply_fill({"txid": "T1", "side": "buy", "amount": None, "kind": "ENTRY"}, 2.0, 25.0, fee=0.0)
    assert strat.s["spent"] == 50.0

def test_spot_rules__parse_bool_flag_rejects_nan_and_containers_iter144_s1():
    """[ITER144 #1] spot_rules._parse_bool_flag rejects NaN and containers."""
    # 1. spot_rules._parse_bool_flag rejects NaN and containers
    assert spot_rules._parse_bool_flag(float("nan")) is False
    assert spot_rules._parse_bool_flag(["true"]) is False
    assert spot_rules._parse_bool_flag("0") is False
    assert spot_rules._parse_bool_flag("true") is True
    assert spot_rules._parse_bool_flag(1) is True

def test_spot_engine__parse_bool_flag_clean_order_id_and__sanitize_state_dict_iter144_s2():
    """[ITER144 #2] spot_engine._parse_bool_flag, clean_order_id, and _sanitize_state_dict."""
    # 2. spot_engine._parse_bool_flag, clean_order_id, and _sanitize_state_dict
    assert _parse_bool_flag("false") is False
    assert _parse_bool_flag("0") is False
    assert _parse_bool_flag(float("nan")) is False
    assert clean_order_id(12345.0) == "12345"
    assert clean_order_id("nan") is None
    assert clean_order_id("0") is None
    assert clean_order_id(["1"]) is None

    fake_client = SimpleNamespace(
        pair_precision=lambda pair: SimpleNamespace(price_decimals=2, volume_decimals=4, order_min=0.01)
    )
    params = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.5,
        dca_amount=100.0,
        dca_drop_pct=2.0,
        check_minutes=5,
        takeprofit_pct=3.0,
        max_budget=500.0,
        max_dca_buys=4,
        enable_takeprofit=True,
        order_ttl_min=60,
    )
    dirty_state = {
        "qty": 2.0,
        "cost": "nan",
        "trend_mode": "false",
        "ledger_reconcile_required": "0",
        "surge_active": "no",
        "slow_grind_active": float("nan"),
        "orders": [
            {
                "txid": 98765.0,
                "side": "buy",
                "vol": 1.0,
                "price": 100.0,
                "market": "false",
                "cancel_requested": "0",
                "intent_id": "nan",
            }
        ],
        "pending_intent": {
            "order_id": 54321.0,
            "side": "BUY",
            "requested_qty": 1.0,
            "requested_price": 100.0,
            "market": "false",
            "intent_id": "null",
        },
        "pending_exit": {"kind": "TP", "soft_floor": "false"},
    }
    strat = Strategy(fake_client, "HYPEUSD", params, dry_run=True, initial_state=dirty_state, replay_mode=True)
    assert strat.s["trend_mode"] is False
    assert strat.s["ledger_reconcile_required"] is False
    assert strat.s["surge_active"] is False
    assert strat.s["slow_grind_active"] is False
    assert strat.s["orders"][0]["txid"] == "98765"
    assert strat.s["orders"][0]["market"] is False
    assert strat.s["orders"][0]["cancel_requested"] is False
    assert strat.s["orders"][0]["intent_id"] is None
    assert strat.s["pending_exit"]["soft_floor"] is False
    assert strat._avg() is None

    adopted = strat._order_from_pending(strat.s["pending_intent"])
    assert adopted["txid"] == "54321"
    assert adopted["market"] is False
    assert adopted["intent_id"] is None

def test_stratparams___post_init___normalizes_string_boolean_flags_iter173_s4():
    """[ITER173 #4] StratParams.__post_init__ normalizes string boolean flags."""
    # 4. StratParams.__post_init__ normalizes string boolean flags
    sp = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=1.0,
        dca_amount=100.0,
        dca_drop_pct=2.0,
        check_minutes=1.0,
        takeprofit_pct=5.0,
        max_budget=1000.0,
        max_dca_buys=5,
        enable_takeprofit="",
        order_ttl_min=60.0,
        sl_mhdsl_enabled="false",
        surge_guard="0",
        tp_mdhtp_enabled="true",
    )
    assert sp.enable_takeprofit is True
    assert sp.sl_mhdsl_enabled is False
    assert sp.surge_guard is False
    assert sp.tp_mdhtp_enabled is True
