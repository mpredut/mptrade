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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from market_regime import MarketRegimeDecision
from providers.strategy_executor import OrderStatus, PairPrecision, ProviderError
from strategies import spot_engine as strat
from test_kraken_strategy_provider_live import FakeExecutor, _strategy


@pytest.fixture(autouse=True)
def silence_notifications(monkeypatch):
    monkeypatch.setattr(strat, "notify", lambda **kwargs: None)


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
    params = strat.StratParams(**defaults)
    s = strat.Strategy(
        client, "TESTPAIR_GUARD", params, dry_run=True,
        initial_state=strat._new_state(),
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
    engine._save = strat.Strategy._save.__get__(engine)
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine.step(80.0)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["pending_exit"]["kind"] == "STOP"
    assert saved["orders"][0]["cancel_requested"]
    with patch.object(strat, "state_path_for", return_value=engine.state_file):
        restarted = strat.Strategy(executor, engine.pair, engine.p, dry_run=False)
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
        engine._save = strat.Strategy._save.__get__(engine)
        engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
        engine.step(115.8)
        assert submits(executor) == []
        saved = json.loads((tmp_path / f"state_{overlay}.json").read_text())
        assert saved["pending_exit"]["min_profit_pct"] == pytest.approx(12.75)
        assert saved["orders"][0]["cancel_requested"]

        engine.p.surge_gain_pct = 1.0
        with patch.object(strat, "state_path_for", return_value=engine.state_file):
            restarted = strat.Strategy(executor, engine.pair, engine.p, dry_run=False)
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
    with patch.object(strat.time, "time", return_value=now):
        engine.step(105.5)

    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    state = engine.exit_manager.tp_ride_state.get(engine.pair)
    assert state is not None and state["peak_price"] == 105.5 and state["started_at"] == now


def test_mdhtp_peak_pullback_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(strat.time, "time", return_value=now):
        engine.step(110.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    with patch.object(strat.time, "time", return_value=now + 60.0):
        engine.step(106.15)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    assert engine.pair not in engine.exit_manager.tp_ride_state


def test_mdhtp_profit_floor_breach_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_trail_profit_floor_pct=1.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(strat.time, "time", return_value=now):
        engine.step(105.5)
    with patch.object(strat.time, "time", return_value=now + 30.0):
        engine.step(100.5)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_stagnation_timeout_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_stagnation_max_sec=1080.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(strat.time, "time", return_value=now):
        engine.step(106.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    with patch.object(strat.time, "time", return_value=now + 1070.0):
        engine.step(105.9)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    with patch.object(strat.time, "time", return_value=now + 1085.0):
        engine.step(105.9)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_trend_reversal_exit():
    engine, executor = make_mdhtp_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(strat.time, "time", return_value=now):
        engine.step(107.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    engine.s["trend_mode"] = False
    with patch.object(strat.time, "time", return_value=now + 60.0):
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
    with patch.object(strat.time, "time", return_value=now):
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
        with open(os.path.join(tmp_dir, "kalman_state.json"), "w", encoding="utf-8") as f:
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
        with open(os.path.join(tmp_dir, "kalman_state.json"), "w", encoding="utf-8") as f:
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
    with patch.object(strat.time, "time", return_value=now):
        engine.step(101.55)
    assert not submits(executor)

    # Tick 2 at 101.04: pullback from 101.55 is 0.502% >= 0.5% trail, gain +1.04% >= 1.0% floor,
    # but exit_px = round(101.04 * 0.999, 2) = 100.94 < profit_floor (101.00).
    # Soft trailing is blocked without locking pending_exit or losing tp_ride_state.
    with patch.object(strat.time, "time", return_value=now + 10.0):
        engine.step(101.04)
    assert not submits(executor)
    assert engine.s.get("pending_exit") is None
    assert engine.pair in engine.exit_manager.tp_ride_state

    # Tick 3 at 100.50 (+0.50% < 1.0% floor): mdhtp_profit_floor_breached must execute immediately!
    with patch.object(strat.time, "time", return_value=now + 20.0):
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
    with patch.object(strat.time, "time", return_value=now):
        engine.step(120.0)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_post_stop_reentry_tracks_peak_bounce_during_cooldown_and_blocks_rollover():
    from strategies.position_exit_manager import (
        PositionExitConfig, PositionExitManager, ReentryAction, ReentrySnapshot,
    )
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
    engine.s["spent"] = engine._effective_max_budget()
    engine.step(93.0, timestamp=2060.0)
    assert not submits(executor)
    # Re-entry state must be restored because no ENTRY order was placed!
    assert engine.pair in engine.exit_manager.reentry_state
    assert engine.exit_manager.reentry_state[engine.pair]["trough"] == 90.0


def test_spot_rules_non_positive_drop_thresholds_do_not_false_trigger():
    import strategies.spot_rules as sr
    trig, _, _ = sr.check_fast_profit_reversal(
        current_price=120.0, avg_cost=100.0, base_tp_pct=5.0, mult=2.0,
        shadow_prices=[(990.0, 120.0)], window_sec=300.0, drop_pct=0.0, current_time=1000.0,
    )
    assert trig is False

    s_trig, _ = sr.check_parabolic_surge_exhaustion(
        current_price=125.0, avg_cost=100.0, surge_peak=125.0,
        surge_gain_pct=18.0, exit_pullback_pct=0.0, surge_active=True,
    )
    assert s_trig is False

    g_trig, _ = sr.check_slow_grind_exhaustion(
        current_price=120.0, avg_cost=100.0, entry_ts=0.0, current_ts=10 * 86400.0,
        min_days=7.0, min_gain_pct=15.0, recent_peak=120.0,
        shadow_prices=[(10 * 86400.0 - 60.0, 120.0)], flash_window_sec=900.0,
        flash_drop_pct=0.0, structural_drop_pct=0.0, sma_value=None,
    )
    assert g_trig is False


def test_position_exit_manager_resets_tp_and_sl_state_when_buy_price_changes():
    from strategies.position_exit_manager import (
        ExitAction, PositionExitConfig, PositionExitManager, PositionSnapshot,
    )
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

