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
)
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot


@pytest.fixture(autouse=True)
def clean_grace_tracker(tmp_path, monkeypatch):
    """Ensure in-memory grace tracker and state file are isolated for each test."""
    test_state_file = str(tmp_path / "smart_stop_loss_state.json")
    monkeypatch.setattr("intelligence.external.guards.smart_stop_loss_guard._SMART_SL_STATE_FILE", test_state_file)
    from intelligence.external.guards.smart_stop_loss_guard import _PENDING_RECOVERIES, _LAST_RECOVERY_ALERT_TS
    _GRACE_TRACKER.clear()
    _PENDING_RECOVERIES.clear()
    _LAST_RECOVERY_ALERT_TS.clear()
    yield
    _GRACE_TRACKER.clear()
    _PENDING_RECOVERIES.clear()
    _LAST_RECOVERY_ALERT_TS.clear()


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


def test_smart_sl_notifications_detail_and_debounce():
    import intelligence.external.guards.smart_stop_loss_guard as ssl_guard
    ssl_guard._GRACE_TRACKER.clear()
    ssl_guard._LAST_RECOVERY_ALERT_TS.clear()
    ssl_guard._PENDING_RECOVERIES.clear()

    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    alerts_sent = []
    with patch("intelligence.external.guards.smart_stop_loss_guard._send_ntfy_alert", side_effect=lambda title, body, symbol: alerts_sent.append({"title": title, "body": body, "symbol": symbol})):
        t0 = 5000.0
        # 1. Grace period entered for BTCUSDC on Binance
        should_exec, _, _ = guard.evaluate(
            "BTCUSDC", 0.0294, 0.02925, now=t0,
            provider="binance", current_price=64200.0, buy_price=66150.0,
            whale_snapshot=whale, orderbook_snapshot=ob,
        )
        assert should_exec is False
        assert len(alerts_sent) == 1
        defer_alert = alerts_sent[-1]
        assert "BTCUSDC" in defer_alert["title"]
        assert "BINANCE" in defer_alert["title"]
        # Ensure redundant 'Vendor:' and 'Coin:' labels are NOT in the body
        assert "Vendor:" not in defer_alert["body"]
        assert "Coin:" not in defer_alert["body"]
        assert "BTCUSDC (BINANCE)" in defer_alert["body"]
        assert "$64,200.00" in defer_alert["body"]
        assert "(Last Buy Price: $66,150.00)" in defer_alert["body"]
        assert "Dip: 2.94%" in defer_alert["body"]

        # 2. Standalone recovery 40s later (batch=False)
        recovered = guard.on_recovered(
            "BTCUSDC", price_decrease=0.022, lost_threshold=0.02925,
            now=t0 + 40.0, provider="binance", current_price=64800.0, buy_price=66150.0,
            batch=False,
        )
        assert recovered is True
        assert len(alerts_sent) == 2
        recovery_alert = alerts_sent[-1]
        # 'SAVED' removed from title as requested
        assert "SAVED" not in recovery_alert["title"]
        assert "BTCUSDC (BINANCE)" in recovery_alert["title"]
        assert "Vendor:" not in recovery_alert["body"]
        assert "Coin:" not in recovery_alert["body"]
        assert "Dip: 2.20% <= 2.93%" in recovery_alert["body"]
        assert "$64,800.00" in recovery_alert["body"]
        assert "(Last Buy Price: $66,150.00)" in recovery_alert["body"]
        assert "Bounce in 40s!" in recovery_alert["body"]

        # 3. 30-minute cooldown (1800s): if same symbol recovers again within 1800s, no alert sent
        guard.evaluate(
            "BTCUSDC", 0.0294, 0.02925, now=t0 + 100.0,
            provider="binance", whale_snapshot=whale, orderbook_snapshot=ob,
        )
        alerts_before = len(alerts_sent)
        recovered_within_cd = guard.on_recovered(
            "BTCUSDC", price_decrease=0.022, lost_threshold=0.02925,
            now=t0 + 140.0, batch=False,
        )
        assert recovered_within_cd is True
        # Suppressed by 30-minute cooldown
        assert len(alerts_sent) == alerts_before

        # After 1801s (> 30m), alert is permitted again
        guard.evaluate(
            "BTCUSDC", 0.0294, 0.02925, now=t0 + 2000.0,
            provider="binance", whale_snapshot=whale, orderbook_snapshot=ob,
        )
        alerts_before_recovery = len(alerts_sent)
        recovered_after_cd = guard.on_recovered(
            "BTCUSDC", price_decrease=0.022, lost_threshold=0.02925,
            now=t0 + 2040.0, batch=False,
        )
        assert recovered_after_cd is True
        assert len(alerts_sent) == alerts_before_recovery + 1


def test_smart_sl_batch_unification_multiple_coins():
    from intelligence.external.guards.smart_stop_loss_guard import (
        _LAST_RECOVERY_ALERT_TS,
        _PENDING_RECOVERIES,
        flush_smart_stop_loss_alerts,
    )
    _LAST_RECOVERY_ALERT_TS.clear()
    _PENDING_RECOVERIES.clear()

    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    alerts_sent = []
    with patch("intelligence.external.guards.smart_stop_loss_guard._send_ntfy_alert", side_effect=lambda title, body, symbol: alerts_sent.append({"title": title, "body": body, "symbol": symbol})):
        t0 = 10000.0
        # Both BTCUSDC and TAOUSDC enter grace on Binance
        guard.evaluate("BTCUSDC", 0.0293, 0.02925, now=t0, provider="binance", current_price=64200.0, whale_snapshot=whale, orderbook_snapshot=ob)
        guard.evaluate("TAOUSDC", 0.0522, 0.0525, now=t0 + 4.0, provider="binance", current_price=580.0, whale_snapshot=whale, orderbook_snapshot=ob)

        # Clear DEFER alerts from test log
        alerts_sent.clear()

        # Both recover during the same monitoring cycle in batch mode (batch=True default)
        guard.on_recovered("BTCUSDC", price_decrease=0.0289, lost_threshold=0.02925, now=t0 + 39.0, provider="binance", current_price=64500.0)
        guard.on_recovered("TAOUSDC", price_decrease=0.0498, lost_threshold=0.0525, now=t0 + 43.0, provider="binance", current_price=595.0)

        # Alerts should NOT have been sent immediately (buffered for unification)
        assert len(alerts_sent) == 0

        # Loop ends -> flush_smart_stop_loss_alerts() called
        flushed_count = flush_smart_stop_loss_alerts()
        assert flushed_count == 2
        assert len(alerts_sent) == 1

        unified = alerts_sent[0]
        # Title has both coins and no 'SAVED'
        assert "SAVED" not in unified["title"]
        assert "BTCUSDC, TAOUSDC (BINANCE)" in unified["title"]
        # Body contains both positions as bullet points
        assert "• BTCUSDC (BINANCE): bounce in 39s" in unified["body"]
        assert "dip 2.89% <= 2.93%" in unified["body"]
        assert "• TAOUSDC (BINANCE): bounce in 39s" in unified["body"]
        assert "dip 4.98% <= 5.25%" in unified["body"]
        assert "Positions saved from wick hunt (2)!" in unified["body"]


def test_smart_sl_replay_provider_never_emits_notifications(tmp_path):
    import os
    import intelligence.external.guards.smart_stop_loss_guard as ssl_guard

    test_state_file = str(tmp_path / "smart_stop_loss_state_replay.json")
    ssl_guard._SMART_SL_STATE_FILE = test_state_file
    ssl_guard._GRACE_TRACKER.clear()
    ssl_guard._LAST_RECOVERY_ALERT_TS.clear()
    ssl_guard._PENDING_RECOVERIES.clear()

    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    alerts_sent = []
    with patch(
        "intelligence.external.guards.smart_stop_loss_guard._send_ntfy_alert",
        side_effect=lambda title, body, symbol: alerts_sent.append({"title": title, "body": body, "symbol": symbol}),
    ):
        t0 = 20000.0
        # 1. Grace period entered for BTCUSDC on replay provider
        should_exec, reason, meta = guard.evaluate(
            "BTCUSDC", 0.0294, 0.02925, now=t0,
            provider="replay", current_price=68248.66, buy_price=70850.0,
            whale_snapshot=whale, orderbook_snapshot=ob,
        )
        assert should_exec is False
        assert meta["is_simulated"] is True
        assert len(alerts_sent) == 0
        assert not os.path.exists(test_state_file)

        # 2. Timeout after 120s in replay
        should_exec, reason, meta = guard.evaluate(
            "BTCUSDC", 0.0368, 0.02925, now=t0 + 121.0,
            provider="replay", current_price=68248.66, buy_price=70850.0,
            whale_snapshot=whale, orderbook_snapshot=ob,
        )
        assert should_exec is True
        assert "grace_period_expired" in reason
        assert len(alerts_sent) == 0
        assert not os.path.exists(test_state_file)

        # 3. Recovery in replay also never emits
        # Re-enter grace
        guard.evaluate(
            "BTCUSDC", 0.0294, 0.02925, now=t0 + 200.0,
            provider="replay", current_price=68248.66,
            whale_snapshot=whale, orderbook_snapshot=ob,
        )
        recovered = guard.on_recovered(
            "BTCUSDC", price_decrease=0.022, lost_threshold=0.02925,
            now=t0 + 240.0, provider="replay", current_price=69500.0,
        )
        assert recovered is True
        assert len(alerts_sent) == 0
        assert len(ssl_guard._PENDING_RECOVERIES) == 0


def test_smart_sl_buy_price_change_resets_stale_grace_and_updates_current_price():
    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    t0 = 30000.0
    # Position 1 enters grace at t0 with buy_price=70000.0
    should_exec1, reason1, _ = guard.evaluate(
        "BTCUSDC", 0.0295, 0.02925, now=t0,
        provider="replay", current_price=67935.0, buy_price=70000.0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec1 is False
    assert "suspected_wick_hunt" in reason1

    # Tick 2 at t0 + 30s updates current_price in tracker
    should_exec2, reason2, _ = guard.evaluate(
        "BTCUSDC", 0.0300, 0.02925, now=t0 + 30.0,
        provider="replay", current_price=67900.0, buy_price=70000.0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec2 is False
    assert _GRACE_TRACKER["BTCUSDC"]["current_price"] == 67900.0

    # At t0 + 150s (> 120s), a new position (or DCA-adjusted position) has buy_price=68000.0.
    # Stale grace from buy_price=70000.0 must be reset so the new position gets a fresh 120s grace!
    should_exec3, reason3, _ = guard.evaluate(
        "BTCUSDC", 0.0295, 0.02925, now=t0 + 150.0,
        provider="replay", current_price=65994.0, buy_price=68000.0,
        whale_snapshot=whale, orderbook_snapshot=ob,
    )
    assert should_exec3 is False
    assert "suspected_wick_hunt" in reason3
    assert _GRACE_TRACKER["BTCUSDC"]["start_ts"] == t0 + 150.0
    assert _GRACE_TRACKER["BTCUSDC"]["buy_price"] == 68000.0


def test_smart_sl_on_recovered_ignores_stale_grace_when_buy_price_changed():
    import intelligence.external.guards.smart_stop_loss_guard as ssl_guard

    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    alerts_sent = []
    with patch(
        "intelligence.external.guards.smart_stop_loss_guard._send_ntfy_alert",
        side_effect=lambda title, body, symbol: alerts_sent.append({"title": title, "body": body, "symbol": symbol}),
    ):
        t0 = 40000.0
        guard.evaluate(
            "BTCUSDC", 0.0295, 0.02925, now=t0,
            provider="binance", current_price=67935.0, buy_price=70000.0,
            whale_snapshot=whale, orderbook_snapshot=ob,
        )
        alerts_sent.clear()

        # DCA or new position lowers avg buy price to 68000.0 without price bouncing
        recovered = guard.on_recovered(
            "BTCUSDC", price_decrease=0.001, lost_threshold=0.02925,
            now=t0 + 25.0, provider="binance", current_price=67935.0, buy_price=68000.0,
            batch=False,
        )
        assert recovered is False
        assert "BTCUSDC" not in _GRACE_TRACKER
        assert len(alerts_sent) == 0
        assert "BTCUSDC" not in ssl_guard._LAST_RECOVERY_ALERT_TS

