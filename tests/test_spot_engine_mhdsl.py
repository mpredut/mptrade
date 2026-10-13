"""Tests for Centralized PositionExitManager (MHDSL) integration into spot_engine.py."""

import time
from unittest.mock import patch

import pytest

from providers.strategy_executor import OrderStatus
from strategies import spot_engine as strat
from test_kraken_strategy_provider_live import FakeExecutor, _strategy


def make_engine(*, mhdsl_enabled: bool = True, stop_loss_pct: float = 10.0, **params):
    executor = FakeExecutor()
    p = dict(
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


def test_mhdsl_disabled_preserves_legacy_stop():
    engine, executor = make_engine(mhdsl_enabled=False, stop_loss_pct=10.0)
    assert engine.exit_manager is None

    # Trend up should NOT defer legacy stop loss
    engine.s["trend_mode"] = True
    with patch.object(strat, "notify"):
        # 11% loss >= 10% threshold
        triggered = engine._check_stop_loss(89.0)
        assert triggered is True
        assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_hard_disaster_circuit_breaker():
    # 10% SL + (10% * 0.30 = 3% buffer) = 13% disaster floor
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    assert engine.exit_manager is not None

    # Even with confirmed uptrend, breaching disaster floor must force immediate market exit
    engine.s["trend_mode"] = True
    with patch.object(strat, "notify"):
        # 14% loss (price 86.0) >= 13% disaster floor
        triggered = engine._check_stop_loss(86.0)
        assert triggered is True
        assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)
        # sl_bounce_state should be cleared after disaster exit
        assert engine.pair not in engine.s.get("sl_bounce_state", {})


def test_mhdsl_bounce_watch_initiated_on_uptrend():
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    # Enable uptrend via client.trend
    executor.trend = lambda sym: "up"

    with patch.object(strat, "notify"):
        # 10.5% loss (price 89.5): in SL territory, but deferred by Priority 2 bounce watch
        triggered = engine._check_stop_loss(89.5)
        assert triggered is False
        assert not any(c[0] == "submit_order" for c in executor.calls)

        bounce_state = engine.s["sl_bounce_state"].get(engine.pair)
        assert bounce_state is not None
        assert bounce_state["lowest_price"] == 89.5
        assert bounce_state["peak_bounce_price"] == 89.5


def test_mhdsl_bounce_structural_break_and_ratchet():
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    executor.trend = lambda sym: "up"
    now = time.time()

    with patch.object(strat, "notify"):
        # Step 1: Initiate bounce watch at 89.0
        engine._shadow_prices.append((now, 89.0))
        assert engine._check_stop_loss(89.0) is False
        state = engine.s["sl_bounce_state"][engine.pair]
        assert state["lowest_price"] == 89.0

        # Step 2: Price bounces up to 90.0 (recovery: (90-89)/89 = 1.12% >= 0.40% min recovery)
        engine._shadow_prices.append((now + 10, 90.0))
        assert engine._check_stop_loss(90.0) is False
        state = engine.s["sl_bounce_state"][engine.pair]
        assert state["peak_bounce_price"] == 90.0

        # Step 3: Pullback from peak down to 89.6 (pullback: (90-89.6)/90 = 0.44% >= 0.30% pullback threshold)
        engine._shadow_prices.append((now + 20, 89.6))
        triggered = engine._check_stop_loss(89.6)
        assert triggered is True
        assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_bounce_timeout():
    engine, executor = make_engine(
        mhdsl_enabled=True, stop_loss_pct=10.0, sl_trend_defer_max_sec=220.0
    )
    executor.trend = lambda sym: "up"
    start_time = 1000.0

    with patch.object(strat, "notify"):
        # Initiate at t = 1000.0
        engine._shadow_prices.append((start_time, 89.5))
        assert engine._check_stop_loss(89.5) is False

        # Advance time to t = 1200.0 (200s elapsed < 220s timeout): still deferred
        engine._shadow_prices.append((start_time + 200.0, 89.5))
        assert engine._check_stop_loss(89.5) is False

        # Advance time to t = 1225.0 (225s elapsed >= 220s timeout): timeout exit!
        engine._shadow_prices.append((start_time + 225.0, 89.5))
        triggered = engine._check_stop_loss(89.5)
        assert triggered is True
        assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_full_recovery_clears_bounce_state():
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    executor.trend = lambda sym: "up"

    with patch.object(strat, "notify"):
        # Enter SL territory
        assert engine._check_stop_loss(89.5) is False
        assert engine.pair in engine.s["sl_bounce_state"]

        # Price recovers above SL territory (loss 8% < 10%)
        assert engine._check_stop_loss(92.0) is False
        assert engine.pair not in engine.s["sl_bounce_state"]


def test_mhdsl_volatility_scaling_variant_b():
    # 10% SL, vol_1h = 5% (0.05 ratio), multiplier = 0.85 -> Variant B buffer = 4.25%
    # Variant A buffer = 10% * 0.30 = 3.0%
    # Effective buffer = max(3.0%, 4.25%, 0.8%) = 4.25% -> Disaster floor = 14.25%
    engine, executor = make_engine(
        mhdsl_enabled=True,
        stop_loss_pct=10.0,
        sl_disaster_buffer_ratio=0.30,
        sl_disaster_vol_multiplier=0.85,
    )
    executor.trend = lambda sym: "up"
    # Inject 5% 1h volatility
    engine._vol_1h_ratio = lambda: (0.05, "mock_test_vol")

    with patch.object(strat, "notify"):
        # 13.5% loss (price 86.5): breaches Variant A floor (13%), but SAFE under Variant B floor (14.25%)!
        # Should be deferred on uptrend:
        assert engine._check_stop_loss(86.5) is False

        # 14.5% loss (price 85.5): breaches Variant B floor (14.25%) -> Priority 1 Hard Exit!
        triggered = engine._check_stop_loss(85.5)
        assert triggered is True
        assert any(c[0] == "submit_order" and c[6] == "STOP" for c in executor.calls)


def test_mhdsl_cycle_finalization_resets_state():
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0)
    engine.s["sl_bounce_state"][engine.pair] = {"mock": "data"}

    # Simulate flat exit fill finalization
    sell_order = {"kind": "STOP", "side": "sell", "vol": 1.0, "price": 85.0}
    engine.s["qty"] = 0.0
    engine._finalize_cycle_if_flat(sell_order, 85.0)

    # State must be cleanly reset
    assert engine.s["sl_bounce_state"] == {}
    assert engine.exit_manager.sl_bounce_state == {}

def test_mhdsl_reentry_mirror_post_stop_bounce_and_ratchet():
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0, reentry_sl_bounce_pct=1.5)
    engine.s["qty"] = 0.0  # Flat position after STOP
    engine.s["last_sell_price"] = 90.0
    engine.s["last_exit_kind"] = "STOP"
    engine.s["last_sell_ts"] = time.time()

    with patch.object(strat, "notify"):
        # Step 1: Price drops to 88.0 (forming trough) - no bounce yet -> re-entry blocked
        engine.step(88.0)
        assert not any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)

        # Step 2: Price bounces to 89.4 (bounce: (89.4 - 88.0)/88.0 = 1.59% >= 1.5%) -> re-entry approved
        engine.step(89.4)
        assert any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)


def test_mhdsl_reentry_mirror_bull_trend_pullback():
    engine, executor = make_engine(mhdsl_enabled=True, stop_loss_pct=10.0, reentry_pullback_pct=2.0)
    engine.s["qty"] = 0.0
    engine.s["last_sell_price"] = 100.0
    engine.s["last_exit_kind"] = "TP"
    engine.s["last_sell_ts"] = time.time()
    executor.trend = lambda sym: "up"  # Confirmed bull trend

    with patch.object(strat, "notify"):
        # Step 1: Price rallies to 105.0 -> waiting for pullback
        engine.step(105.0)
        assert not any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)

        # Step 2: Price pulls back to 102.8 (pullback: (105.0 - 102.8)/105.0 = 2.095% >= 2.0%) -> approved
        engine.step(102.8)
        assert any(c[0] == "submit_order" and c[2] == "buy" for c in executor.calls)
