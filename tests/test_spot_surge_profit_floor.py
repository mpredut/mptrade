"""Surge profit protection across cancellation, restart, and partial fills."""
import json
from unittest.mock import patch

import pytest

from market_regime import MarketRegimeDecision
from strategies import spot_engine as strat
from providers.strategy_executor import OrderStatus
from test_spot_exit_lifecycle import order, positioned


@pytest.fixture(autouse=True)
def silence_notifications(monkeypatch):
    monkeypatch.setattr(strat, "notify", lambda **kwargs: None)


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


def submits(executor):
    return [call for call in executor.calls if call[0] == "submit_order"]


@pytest.mark.parametrize("overlay", [False, True])
def test_surge_exit_rechecks_profit_after_cancel_and_survives_restart(tmp_path, overlay):
    engine, executor = surge_position(overlay)
    engine.state_file = str(tmp_path / "state.json")
    engine._save = strat.Strategy._save.__get__(engine)
    engine.s["orders"] = [order("DCA-1", "buy", price=95.0)]
    engine.step(115.8)
    assert submits(executor) == []
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["pending_exit"]["min_profit_pct"] == pytest.approx(12.75)
    assert saved["orders"][0]["cancel_requested"]

    # Even lower surge thresholds after restart must not weaken this pending exit.
    engine.p.surge_gain_pct = 1.0
    with patch.object(strat, "state_path_for", return_value=engine.state_file):
        restarted = strat.Strategy(executor, engine.pair, engine.p, dry_run=False)
    executor.next_status = OrderStatus("canceled", 0, 0, 0)
    restarted.reconcile(96.5)
    restarted.step(96.5)
    assert submits(executor) == []
    assert restarted.s["pending_exit"]["min_profit_pct"] == pytest.approx(12.75)
    assert not restarted._has_open("buy")

    # The current market reference, after rounding and its buffer, must clear the floor.
    restarted.step(112.75)
    assert submits(executor) == []
    restarted.step(115.8)
    assert len(submits(executor)) == 1
    assert submits(executor)[0][2] == "sell"
    assert submits(executor)[0][5:7] == (True, "TP")


@pytest.mark.parametrize("overlay", [False, True])
def test_surge_exit_uses_cost_basis_after_a_racing_buy_fill(overlay):
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


@pytest.mark.parametrize("overlay", [False, True])
@pytest.mark.parametrize("already_armed", [False, True])
def test_window_rally_or_old_armed_state_cannot_trigger_surge_at_a_loss(overlay, already_armed):
    engine, executor = surge_position(overlay)
    engine.s.update(surge_peak=100.0, surge_active=already_armed,
                    trail_peak=100.0 if already_armed else None)
    engine.step(100.0)
    assert not engine.s["surge_active"]
    engine.step(96.5)
    assert not engine.s["pending_exit"]
    assert not any(call[2] == "sell" and call[5] for call in submits(executor))


@pytest.mark.parametrize("stored_floor", [None, "invalid", -1.0, float("nan"), float("inf")])
def test_invalid_persisted_profit_floor_blocks_submission(stored_floor):
    engine, executor = surge_position()
    engine.s["pending_exit"] = {"kind": "TP", "min_profit_pct": stored_floor}
    engine.step(115.8)
    assert submits(executor) == []


@pytest.mark.parametrize("cost", [0.0, float("nan"), float("inf")])
def test_unavailable_cost_basis_blocks_a_pending_profit_exit(cost):
    engine, executor = surge_position()
    engine.s.update(cost=cost, pending_exit={"kind": "TP", "min_profit_pct": 12.75})
    engine.step(115.8)
    assert submits(executor) == []
