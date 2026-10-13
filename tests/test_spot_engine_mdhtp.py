"""Tests for Centralized PositionExitManager (MDHTP) integration into spot_engine.py."""

import time
from unittest.mock import patch

import pytest

from providers.strategy_executor import OrderStatus
from strategies import spot_engine as strat
from test_kraken_strategy_provider_live import FakeExecutor, _strategy


def make_engine(*, mdhtp_enabled: bool = True, takeprofit_pct: float = 5.0, **params):
    executor = FakeExecutor()
    p = dict(
        takeprofit_pct=takeprofit_pct,
        tp_trend_hold=True,
        tp_trail_pct=3.0,
        tp_trail_profit_floor_pct=1.0,
        tp_mdhtp_enabled=mdhtp_enabled,
        tp_stagnation_max_sec=1080.0,
        fast_profit_guard=True,
    )
    p.update(params)
    engine = _strategy(executor, **p)
    engine.s.update(
        qty=1.0,
        cost=100.0,
        spent=100.0,
        entry_price=100.0,
        last_buy_price=100.0,
    )
    return engine, executor


def test_mdhtp_trend_ride_initiation_and_hold():
    engine, executor = make_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    assert engine.exit_manager is not None

    # Trend up + crossing 5% TP (105.5) -> initiates trend ride, HOLD
    engine.s["trend_mode"] = True
    now = 1000.0
    with patch.object(strat.time, "time", return_value=now):
        engine.step(105.5)

    # Position is held, not sold
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    assert engine.exit_manager.tp_ride_state.get(engine.pair) is not None
    state = engine.exit_manager.tp_ride_state[engine.pair]
    assert state["peak_price"] == 105.5
    assert state["started_at"] == now


def test_mdhtp_peak_pullback_exit():
    engine, executor = make_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    now = 1000.0

    # Step 1: Rise to 110.0 (10% profit)
    with patch.object(strat.time, "time", return_value=now):
        engine.step(110.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    # Step 2: Pull back by 3.5% (110 * 0.965 = 106.15) >= 3.0% trail pullback
    with patch.object(strat.time, "time", return_value=now + 60.0):
        engine.step(106.15)

    # Must trigger market exit on pullback
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)
    assert engine.pair not in engine.exit_manager.tp_ride_state


def test_mdhtp_profit_floor_breach_exit():
    # Floor is 1.0% (101.0).
    engine, executor = make_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_trail_profit_floor_pct=1.0)
    engine.s["trend_mode"] = True
    now = 1000.0

    # Step 1: Reach 105.5
    with patch.object(strat.time, "time", return_value=now):
        engine.step(105.5)

    # Step 2: Sharp drop to 100.5 (< 1.0% floor)
    with patch.object(strat.time, "time", return_value=now + 30.0):
        engine.step(100.5)

    # Guaranteed floor breached -> immediate exit
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_stagnation_timeout_exit():
    engine, executor = make_engine(mdhtp_enabled=True, takeprofit_pct=5.0, tp_stagnation_max_sec=1080.0)
    engine.s["trend_mode"] = True
    now = 1000.0

    # Step 1: Reach 106.0 at t=1000
    with patch.object(strat.time, "time", return_value=now):
        engine.step(106.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    # Step 2: At t = 1000 + 1070s (< 1080s), still holding
    with patch.object(strat.time, "time", return_value=now + 1070.0):
        engine.step(105.9)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    # Step 3: At t = 1000 + 1085s (>= 1080s stagnation), exits due to timeout
    with patch.object(strat.time, "time", return_value=now + 1085.0):
        engine.step(105.9)
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_trend_reversal_exit():
    engine, executor = make_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True
    now = 1000.0

    # Step 1: Reach 107.0 with uptrend
    with patch.object(strat.time, "time", return_value=now):
        engine.step(107.0)
    assert not any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)

    # Step 2: Trend turns False (trend reversal)
    engine.s["trend_mode"] = False
    with patch.object(strat.time, "time", return_value=now + 60.0):
        engine.step(106.8)

    # Trend reversal exit triggered
    assert any(c[0] == "submit_order" and c[6] == "TP" for c in executor.calls)


def test_mdhtp_cycle_finalization_resets_tp_ride_state():
    engine, executor = make_engine(mdhtp_enabled=True, takeprofit_pct=5.0)
    engine.s["trend_mode"] = True

    # Initiate ride
    engine.step(106.0)
    assert engine.pair in engine.exit_manager.tp_ride_state

    # Finalize cycle
    engine.s["qty"] = 0.0
    engine.s["orders"] = []
    engine._finalize_cycle_if_flat(o={"kind": "TP"}, price=106.0)

    # State cleanly cleared
    assert engine.pair not in engine.exit_manager.tp_ride_state
    assert engine.exit_manager.tp_ride_state == engine.s["tp_ride_state"]
