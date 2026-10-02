"""Adversarial exit tests shared by Kraken and Hyperliquid spot; no network."""
import copy
import json
from unittest.mock import patch

import pytest

from strategies import spot_engine as strat
from providers.strategy_executor import OrderStatus, ProviderError
from test_kraken_strategy_provider_live import FakeExecutor, _strategy


def order(order_id, side, *, qty=1.0, price=100.0, kind="DCA", market=False):
    return {"txid": order_id, "side": side, "vol": qty, "price": price,
            "amount": qty * price if side == "buy" else 0.0,
            "kind": kind, "market": market, "ts": 9999999999.0}


def positioned(**params):
    executor = FakeExecutor()
    engine = _strategy(executor, **params)
    engine.s.update(qty=1.0, cost=100.0, spent=100.0,
                    entry_price=100.0, last_buy_price=100.0)
    return engine, executor


def test_market_exit_waits_for_terminal_cancels_and_uses_racing_fill():
    for mode in ["stop", "trailing", "overlay"]:
        engine, executor = positioned(trend_overlay=(mode == "overlay"))
        engine.s["orders"] = [order("DCA-1", "buy", price=95.0),
                              order("TP-1", "sell", price=110.0, kind="TP")]
        engine.s.update(trail_peak=110.0, trail_stop=106.7,
                        trend_mode=(mode == "overlay"), trend_peak=110.0)
        price = 80.0 if mode == "stop" else 103.0
        with patch.object(strat, "notify"):
            engine.step(price)
            assert not any(c[0] == "submit_order" for c in executor.calls)
            assert all(o.get("cancel_requested") for o in engine.s["orders"])
            # Cancellation acceptance did not prevent a partial DCA fill on the venue.
            executor.order_status = lambda _symbol, oid: (
                OrderStatus("canceled", 0.25, 23.75, 0.02) if oid == "DCA-1"
                else OrderStatus("canceled", 0.0, 0.0, 0.0))
            engine.reconcile(price)
            engine.step(price)
        submits = [c for c in executor.calls if c[0] == "submit_order"]
        assert len(submits) == 1 and submits[0][2] == "sell" and submits[0][5]
        assert submits[0][3] == engine._dust_safe_qty(1.25)


def test_triggered_stop_survives_restart_and_price_recovery_without_new_buy():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    with patch.object(strat, "notify"):
        engine.step(80.0)
        assert not any(c[0] == "submit_order" for c in executor.calls)
        saved = copy.deepcopy(engine.s)
        restarted = _strategy(executor)
        restarted.s = saved
        executor.next_status = OrderStatus("canceled", 0.0, 0.0, 0.0)
        restarted.reconcile(100.0)
        restarted.step(100.0)
    submits = [c for c in executor.calls if c[0] == "submit_order"]
    assert len(submits) == 1 and submits[0][2] == "sell" and submits[0][6] == "STOP"
    assert not restarted._has_open("buy")


def test_terminal_sell_does_not_forget_unresolved_buy_or_close_cycle_twice():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0),
                          order("TP-1", "sell", price=110.0, kind="TP")]
    statuses = {"DCA-1": OrderStatus("open", 0, 0, 0),
                "TP-1": OrderStatus("closed", 1, 110, 0.1)}
    executor.order_status = lambda _symbol, oid: statuses[oid]
    with patch.object(strat, "notify"):
        engine.reconcile(110.0)
        assert engine.s["cycle"] == 1
        assert engine._has_open("buy")
        engine.step(110.0)
        assert not any(c[0] == "submit_order" for c in executor.calls)
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
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0),
                          order("TP-1", "sell", price=110.0, kind="TP")]
    statuses = {"DCA-1": OrderStatus("open", 0, 0, 0),
                "TP-1": OrderStatus("closed", 1, 110, 0.1)}
    executor.order_status = lambda _symbol, oid: statuses[oid]
    with patch.object(strat, "notify"):
        engine.reconcile(110.0)
        assert engine._has_open("buy")
        statuses["DCA-1"] = OrderStatus("canceled", 0.2, 19.0, 0.01)
        engine.reconcile(109.0)
        assert engine.s["cycle"] == 1
        assert engine.s["qty"] == pytest.approx(0.2)
        engine.step(109.0)
    submits = [c for c in executor.calls if c[0] == "submit_order"]
    assert len(submits) == 1 and submits[0][2] == "sell"
    assert submits[0][3] == engine._dust_safe_qty(0.2)


def test_exit_intent_must_be_saved_before_cancel_or_submit():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    def fail_save():
        raise RuntimeError("disk unavailable")
    engine._save = fail_save
    with pytest.raises(RuntimeError, match="disk unavailable"):
        engine.step(80.0)
    with pytest.raises(RuntimeError, match="disk unavailable"):
        engine.step(100.0)  # The resumed request must retry its durability boundary.
    assert not any(c[0] in {"cancel_order", "submit_order"} for c in executor.calls)


def test_pending_soft_floor_is_rechecked_but_stop_can_upgrade_the_exit():
    engine, executor = positioned(tp_trail_profit_floor_pct=1.0)
    engine.s.update(trail_peak=110.0, trail_stop=106.7)
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    with patch.object(strat, "notify"):
        engine.step(103.0)
        executor.next_status = OrderStatus("canceled", 0, 0, 0)
        engine.reconcile(100.0)
        engine.step(100.0)
        assert not any(c[0] == "submit_order" for c in executor.calls)
        engine.step(80.0)
    submits = [c for c in executor.calls if c[0] == "submit_order"]
    assert len(submits) == 1 and submits[0][6] == "STOP"


def test_unavailable_cancel_status_keeps_exit_and_all_trackers():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    def unavailable(*_args):
        raise ProviderError("status unavailable")
    executor.order_status = unavailable
    with patch.object(strat, "notify"):
        engine.step(80.0)
        engine.reconcile(80.0)
        engine.step(80.0)
    assert engine._has_open("buy") and engine.s["pending_exit"]["kind"] == "STOP"
    assert not any(c[0] == "submit_order" for c in executor.calls)
    assert len([c for c in executor.calls if c[0] == "cancel_order"]) == 1


def test_accepted_market_exit_is_never_cancelled_or_duplicated():
    engine, executor = positioned()
    with patch.object(strat, "notify"):
        engine.step(80.0)
        engine.step(75.0)
        engine.step(105.0)
    assert len([c for c in executor.calls if c[0] == "submit_order"]) == 1
    assert not any(c[0] == "cancel_order" for c in executor.calls)


def test_late_buy_never_cancels_an_already_accepted_market_exit():
    engine, executor = positioned()
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0),
                          order("STOP-1", "sell", price=80.0, kind="STOP", market=True)]
    executor.order_status = lambda _symbol, oid: (
        OrderStatus("canceled", 0.2, 19.0, 0.01) if oid == "DCA-1"
        else OrderStatus("open", 0, 0, 0))
    with patch.object(strat, "notify"):
        engine.reconcile(80.0)
    assert engine._has_pending_market_exit()
    assert engine.s["qty"] == pytest.approx(1.2)
    assert not any(c[0] == "cancel_order" and c[2] == "STOP-1" for c in executor.calls)
    executor.order_status = lambda *_args: OrderStatus("closed", 1.0, 80.0, 0.1)
    with patch.object(strat, "notify"):
        engine.reconcile(100.0)
        engine.step(100.0)  # Recovery above the stop must not leave the late BUY unowned.
    submits = [c for c in executor.calls if c[0] == "submit_order"]
    assert len(submits) == 1 and submits[0][6] == "STOP"
    assert submits[0][3] == engine._dust_safe_qty(0.2)


def test_exit_request_round_trips_through_real_state_store(tmp_path):
    engine, executor = positioned()
    engine.state_file = str(tmp_path / "state.json")
    engine._save = strat.Strategy._save.__get__(engine)
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    with patch.object(strat, "notify"):
        engine.step(80.0)
        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["pending_exit"]["kind"] == "STOP"
        assert saved["orders"][0]["cancel_requested"]
        with patch.object(strat, "state_path_for", return_value=engine.state_file):
            restarted = strat.Strategy(executor, engine.pair, engine.p, dry_run=False)
        executor.next_status = OrderStatus("canceled", 0, 0, 0)
        restarted.reconcile(100.0)
        restarted.step(100.0)
    submits = [c for c in executor.calls if c[0] == "submit_order"]
    assert len(submits) == 1 and submits[0][6] == "STOP"
