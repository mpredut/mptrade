"""Unit tests for Pillar 2 Smart Liquidity-Aware Stop Loss Controller.

Verifies:
1. Wide stops (e.g. 20-30% trailing stops) bypass immediately.
2. Hard disaster floor circuit breaker executes unconditionally.
3. Active crash cascades with aggressive short momentum execute immediately.
4. Suspected stop-hunt wicks with whale support grant bounded 120s grace period.
5. In-flight grace period defers consecutive ticks until 120s expires.
6. Price recovery within grace window clears state and saves position.
7. Shadow mode logs without delaying execution.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from intelligence.external.guards.smart_stop_loss_guard import (
    SmartStopLossGuard,
    _GRACE_TRACKER,
    clear_smart_stop_loss_if_recovered,
    evaluate_smart_stop_loss,
)
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot


@pytest.fixture(autouse=True)
def clean_grace_tracker():
    """Ensure in-memory grace tracker is clean before and after each test."""
    _GRACE_TRACKER.clear()
    yield
    _GRACE_TRACKER.clear()


def _mock_whale(top_long=0.62, taker_ratio=1.05, regime="neutral"):
    return WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=top_long / (1.0 - top_long),
        top_traders_long_pct=top_long,
        taker_buy_sell_ratio=taker_ratio,
        taker_buy_vol_usd=10000.0,
        taker_sell_vol_usd=10000.0 / taker_ratio,
        open_interest_usd=8_000_000_000.0,
        open_interest_1h_change_pct=0.1,
        divergence_regime=regime,
        ts=time.time(),
    )


def _mock_ob(imbalance=0.80, bid_wall=600_000.0):
    return OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=83000.0,
        bid_depth_usd=1_500_000.0,
        ask_depth_usd=300_000.0,
        imbalance_ratio=imbalance,
        largest_bid_wall_usd=bid_wall,
        largest_bid_wall_price=82900.0,
        largest_ask_wall_usd=50_000.0,
        largest_ask_wall_price=83100.0,
        ts=time.time(),
    )


def test_smart_sl_disabled():
    guard = SmartStopLossGuard(mode="off")
    should_exec, reason, meta = guard.evaluate("BTCUSDC", 0.030, 0.02925)
    assert should_exec is True
    assert reason == "smart_sl_disabled"


def test_smart_sl_wide_stop_exempt():
    guard = SmartStopLossGuard(mode="enforce", max_threshold_pct=6.0)
    # 20% trailing stop
    should_exec, reason, meta = guard.evaluate("BTCUSDC", 0.205, 0.200)
    assert should_exec is True
    assert reason == "wide_stop_exempt"


def test_smart_sl_hard_disaster_floor_breached():
    guard = SmartStopLossGuard(
        mode="enforce",
        disaster_buffer_pct=0.85, # floor = 2.925% + 0.85% = 3.775%
    )
    # Price decrease is 4.0% (>= 3.775%)
    should_exec, reason, meta = guard.evaluate(
        "BTCUSDC",
        price_decrease=0.040,
        lost_threshold=0.02925,
        whale_snapshot=_mock_whale(),
        orderbook_snapshot=_mock_ob(),
    )
    assert should_exec is True
    assert "hard_disaster_floor_breached" in reason


def test_smart_sl_active_crash_cascade():
    guard = SmartStopLossGuard(mode="enforce")
    # Dumping with low taker buy/sell (0.55) and expanding short open interest
    crash_whale = _mock_whale(top_long=0.55, taker_ratio=0.55, regime="aggressive_shorting")
    should_exec, reason, meta = guard.evaluate(
        "BTCUSDC",
        price_decrease=0.030,
        lost_threshold=0.02925,
        whale_snapshot=crash_whale,
        orderbook_snapshot=_mock_ob(),
    )
    assert should_exec is True
    assert "active_crash_cascade" in reason


def test_smart_sl_whale_supported_wick_hunt_lifecycle():
    guard = SmartStopLossGuard(
        mode="enforce",
        grace_seconds=120.0,
        min_whale_long_pct=0.60,
        min_taker_ratio=0.85,
    )
    whale = _mock_whale(top_long=0.614, taker_ratio=0.98, regime="neutral")
    ob = _mock_ob(imbalance=0.75, bid_wall=550_000.0)

    t0 = 1000.0
    # Tick 1 at t0: Loss is 2.93% > SL 2.925% -> Grants Grace Period
    should_exec1, reason1, meta1 = guard.evaluate(
        "BTCUSDC", 0.0293, 0.02925, now=t0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec1 is False
    assert "suspected_wick_hunt" in reason1

    # Tick 2 at t0 + 48s: Still in grace -> Defers execution
    should_exec2, reason2, meta2 = guard.evaluate(
        "BTCUSDC", 0.0310, 0.02925, now=t0 + 48.0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec2 is False
    assert "grace_active" in reason2
    assert meta2["grace_elapsed_sec"] == 48.0

    # Tick 3 at t0 + 96s: Still in grace (< 120s) -> Defers execution
    should_exec3, reason3, meta3 = guard.evaluate(
        "BTCUSDC", 0.0305, 0.02925, now=t0 + 96.0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec3 is False
    assert "grace_active" in reason3

    # Tick 4 at t0 + 125s: Grace expired without recovery -> EXECUTES Stop-Loss!
    should_exec4, reason4, meta4 = guard.evaluate(
        "BTCUSDC", 0.0305, 0.02925, now=t0 + 125.0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec4 is True
    assert "grace_period_expired" in reason4


def test_smart_sl_recovery_saves_position():
    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.62, taker_ratio=1.02)
    ob = _mock_ob()

    t0 = 2000.0
    # Grace begins
    should_exec, reason, _ = guard.evaluate(
        "BTCUSDC", 0.0294, 0.02925, now=t0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec is False

    # 40s later, price bounces back (loss drops to 2.2% <= 2.925%)
    recovered = guard.on_recovered(
        "BTCUSDC", price_decrease=0.022, lost_threshold=0.02925, now=t0 + 40.0
    )
    assert recovered is True
    assert "BTCUSDC" not in _GRACE_TRACKER


def test_smart_sl_shadow_mode():
    guard = SmartStopLossGuard(mode="shadow")
    whale = _mock_whale(top_long=0.62, taker_ratio=1.05)
    should_exec, reason, meta = guard.evaluate(
        "BTCUSDC", 0.030, 0.02925, now=3000.0,
        whale_snapshot=whale, orderbook_snapshot=_mock_ob(),
    )
    # In shadow mode, logs deferral but returns True so trading behavior is unaffected
    assert should_exec is True
    assert "shadow_defer" in reason
