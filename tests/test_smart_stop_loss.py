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

import contextlib
import json
import math
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import external_order_guard
import intelligence.external.guards.smart_stop_loss_guard as ssl_guard
from intelligence.external.collectors.orderbook_depth_collector import OrderbookSnapshot
from intelligence.external.collectors.whale_positioning_collector import (
    WhalePositioningCollector,
    WhalePositioningSnapshot,
)
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.smart_stop_loss_guard import (
    SmartStopLossGuard,
    _GRACE_TRACKER,
    _LAST_DEFER_ALERT_TS,
    _LAST_RECOVERY_ALERT_TS,
    _LAST_TIMEOUT_ALERT_TS,
    _PENDING_RECOVERIES,
    _format_price_part,
    _is_buy_price_changed,
    _is_simulated_provider,
    flush_smart_stop_loss_alerts,
)
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
from intelligence.sentiment.guards.post_stop_reentry_guard import (
    PostStopReentryGuard,
    clear_post_stop_reentry_cache,
)
from intelligence.sentiment.sentiment_advisor import SentimentAdvisor


def _clear_all_trackers():
    for tracker in (
        _GRACE_TRACKER,
        _PENDING_RECOVERIES,
        _LAST_RECOVERY_ALERT_TS,
        _LAST_DEFER_ALERT_TS,
        _LAST_TIMEOUT_ALERT_TS,
    ):
        tracker.clear()


@contextlib.contextmanager
def _capture_ntfy_alerts():
    alerts = []
    with patch(
        "intelligence.external.guards.smart_stop_loss_guard._send_ntfy_alert",
        side_effect=lambda title, body, symbol: alerts.append(
            {"title": title, "body": body, "symbol": symbol}
        ),
    ):
        yield alerts


@pytest.fixture(autouse=True)
def clean_grace_tracker(tmp_path, monkeypatch):
    """Ensure in-memory grace tracker and state file are isolated for each test."""
    test_state_file = str(tmp_path / "smart_stop_loss_state.json")
    monkeypatch.setattr("intelligence.external.guards.smart_stop_loss_guard._SMART_SL_STATE_FILE", test_state_file)
    _clear_all_trackers()
    yield
    _clear_all_trackers()


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
    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    with _capture_ntfy_alerts() as alerts_sent:
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
    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    with _capture_ntfy_alerts() as alerts_sent:
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
    test_state_file = str(tmp_path / "smart_stop_loss_state_replay.json")
    ssl_guard._SMART_SL_STATE_FILE = test_state_file

    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    with _capture_ntfy_alerts() as alerts_sent:
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
    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    whale = _mock_whale(top_long=0.65, taker_ratio=1.1)
    ob = _mock_ob()

    with _capture_ntfy_alerts() as alerts_sent:
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


def test_smart_sl_save_grace_state_filters_simulated_entries_and_uses_exact_memo_key(tmp_path, monkeypatch):

    state_file = tmp_path / "smart_stop_loss_state.json"
    monkeypatch.setattr(ssl_guard, "_SMART_SL_STATE_FILE", str(state_file))

    now_ts = time.time()
    ssl_guard._GRACE_TRACKER.clear()
    ssl_guard._GRACE_TRACKER["SIMBTC"] = {
        "start_ts": now_ts - 5.0,
        "provider": "replay",
        "buy_price": 70000.0,
    }
    ssl_guard._GRACE_TRACKER["ETHUSDC"] = {
        "start_ts": now_ts - 5.0,
        "provider": "binance",
        "buy_price": 2500.0,
    }
    ssl_guard._save_grace_state()

    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert "SIMBTC" not in saved
    assert "ETHUSDC" in saved

    # Populate _GRACE_FILE_MEMO with an unrelated first entry and verify _load_grace_state uses state_file's own signature
    ssl_guard._GRACE_TRACKER.clear()
    ssl_guard._GRACE_FILE_MEMO.clear()
    ssl_guard._GRACE_FILE_MEMO["/tmp/unrelated_first_entry.json"] = (("unrelated_sig",), {})
    ssl_guard._LOADED_STATE_FILE_KEY = None

    loaded = ssl_guard._load_grace_state(is_sim=False)
    assert "ETHUSDC" in loaded
    assert ssl_guard._LOADED_STATE_FILE_KEY is not None
    assert ssl_guard._LOADED_STATE_FILE_KEY[1] != ("unrelated_sig",)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_smartstoplossguard_now_propagation_clock_skewed_grace_reset_iter19_s2(tmp_path, monkeypatch):
    """[ITER19 #2] SmartStopLossGuard `now` propagation & clock-skewed grace reset."""
    # 2. SmartStopLossGuard `now` propagation & clock-skewed grace reset
    w_col = WhalePositioningCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    w_col.record_snapshot(
        WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=2.0,
            top_traders_long_pct=0.68,
            taker_buy_sell_ratio=1.10,
            taker_buy_vol_usd=500_000.0,
            taker_sell_vol_usd=450_000.0,
            open_interest_usd=20_000_000.0,
            open_interest_1h_change_pct=1.5,
            divergence_regime="accumulation",
            ts=2000.0,
        ),
        save_to_disk=True,
    )
    monkeypatch.setattr(external_order_guard, "get_whale_collector", lambda: w_col)
    sl_guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    _GRACE_TRACKER["BTCUSDC"] = {
        "start_ts": 99999.0,  # future clock-skewed grace start_ts
        "initial_loss_pct": 3.1,
        "lost_threshold_pct": 3.0,
        "provider": "binance",
        "current_price": 96.9,
        "buy_price": 100.0,
    }
    should_exec, sl_reason, _ = sl_guard.evaluate(
        "BTCUSDC",
        price_decrease=0.031,
        lost_threshold=0.030,
        now=2050.0,
        provider="binance",
        current_price=96.9,
        buy_price=100.0,
    )
    assert should_exec is False
    assert "suspected_wick_hunt" in sl_reason
    assert _GRACE_TRACKER["BTCUSDC"]["start_ts"] == pytest.approx(2050.0)
    _GRACE_TRACKER.pop("BTCUSDC", None)

def test_smartstoplossguard__load_grace_state_symbol_normalization_none_snapshot_attributes_poststopreentryguard_iter25_s1(tmp_path, monkeypatch):
    """[ITER25 #1] SmartStopLossGuard._load_grace_state & symbol normalization & None snapshot attributes & PostStopReentryGuard.check handles NaN last_sell_time_s and matches TAO/USD with TAOUSD in cache & Internal, Sentiment, and Cascade guards tolerate None / bool / NaN / Inf inputs & GeopoliticalThreatAnalyzer & SentimentAdvisor _load_from_disk tolerate null JSON fields."""
    # 1. SmartStopLossGuard._load_grace_state & symbol normalization & None snapshot attributes
    ssl_file = tmp_path / "smart_stop_loss_state.json"
    now_curr = time.time()
    ssl_file.write_text(
        json.dumps({
            "BAD_NULL": {"start_ts": None},
            "FUTURE_SKEW": {"start_ts": now_curr + 600.0},
            "TAO/USD": {"start_ts": now_curr - 30.0, "initial_loss_pct": 3.1, "lost_threshold_pct": 3.0},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(ssl_guard, "_SMART_SL_STATE_FILE", str(ssl_file))
    ssl_guard._GRACE_TRACKER.clear()
    ssl_guard._LOADED_STATE_FILE_KEY = None
    tracker = ssl_guard._load_grace_state(is_sim=False)
    assert "BAD_NULL" not in tracker
    assert "FUTURE_SKEW" not in tracker
    assert "TAOUSD" in tracker

    guard = SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    should_exec, reason, meta = guard.evaluate(
        "TAO/USD",
        0.032,
        0.030,
        now=now_curr,
        provider="replay",
        whale_snapshot=SimpleNamespace(
            top_traders_long_pct=None,
            taker_buy_sell_ratio=float("nan"),
            divergence_regime=None,
        ),
        orderbook_snapshot=SimpleNamespace(
            imbalance_ratio=None,
            largest_bid_wall_usd=float("nan"),
        ),
    )
    assert should_exec is True
    assert meta["symbol"] == "TAOUSD"


    # 2. PostStopReentryGuard.check handles NaN last_sell_time_s and matches TAO/USD with TAOUSD in cache
    clear_post_stop_reentry_cache()
    fake_gemini = MagicMock()
    fake_gemini.query_json.return_value = {
        "decision": "APPROVED",
        "suggested_scale": 1.0,
        "reason": "clean accumulation",
    }
    ps_guard = PostStopReentryGuard(
        gemini_client=fake_gemini,
        mode="enforce",
        notify_enabled=False,
        cache_ttl_sec=600.0,
    )
    monkeypatch.setattr(
        "intelligence.sentiment.guards.post_stop_reentry_guard.is_llm_guard_exempt",
        lambda _explicit: False,
    )
    dec1 = ps_guard.check(
        "TAO/USD",
        current_price=405.0,
        last_sell_price=float("nan"),
        trough=395.0,
        bounce_pct=0.025,
        last_sell_time_s=float("nan"),
        telemetry={},
        now=now_curr,
    )
    assert dec1.allowed is True
    assert fake_gemini.query_json.call_count == 1
    dec2 = ps_guard.check(
        "TAOUSD",
        current_price=405.0,
        last_sell_price=0.0,
        trough=395.0,
        bounce_pct=0.025,
        last_sell_time_s=0,
        telemetry={},
        now=now_curr + 1.0,
    )
    assert dec2.allowed is True
    assert fake_gemini.query_json.call_count == 1  # served from cache


    # 3. Internal, Sentiment, and Cascade guards tolerate None / bool / NaN / Inf inputs
    exh = WeibullExhaustionGuard()
    assert exh.check(
        "BTCUSDC",
        "BUY",
        trend_duration_seconds=None,
        p90_days=float("nan"),
        median_days=True,
        gaussian_horizon_days=float("inf"),
    ).allowed is True
    assert exh.check(
        "BTCUSDC",
        "BUY",
        trend_duration_seconds=3600.0,
        p90_days=float("nan"),
        median_days=float("nan"),
        gaussian_horizon_days=float("inf"),
    ).allowed is True

    para = ParabolicSurgeGuard()
    assert para.check("BTCUSDC", "BUY", current_price=None, now=float("nan")).allowed is True

    noise = NoiseFloorGuard()
    assert noise.check("BTCUSDC", "BUY", gradient=None, epsilon=True).allowed is True

    greed = ExtremeGreedGuard()
    assert greed.check("BTCUSDC", None, SimpleNamespace(value=None, classification=None)).allowed is True
    assert greed.check("BTCUSDC", "BUY", SimpleNamespace(value=float("nan"), classification=None)).allowed is True

    panic = PanicWashoutGuard()
    assert panic.check(
        "BTCUSDC",
        "BUY",
        breadth_snapshot=SimpleNamespace(advance_ratio=None, median_change_pct=float("nan")),
        fear_greed_snapshot=SimpleNamespace(value=10, historical_values=[10, None]),
        asset_24h_change_pct=float("nan"),
    ).allowed is True

    casc = LiquidationCascadeGuard()
    assert casc.check(
        "BTCUSDC",
        "BUY",
        summary=SimpleNamespace(long_liq_usd=None, short_liq_usd=float("nan")),
    ).allowed is True


    # 4. GeopoliticalThreatAnalyzer & SentimentAdvisor _load_from_disk tolerate null JSON fields
    geo_file = tmp_path / "geopolitical_eval.json"
    geo_file.write_text(
        json.dumps({
            "threat_level": None,
            "risk_score": None,
            "summary": None,
            "recommended_brake": None,
            "headlines_analyzed": None,
            "ts": now_curr,
        }),
        encoding="utf-8",
    )
    geo_analyzer = GeopoliticalThreatAnalyzer(state_file=str(geo_file), cache_ttl_sec=3600.0, fallback_retry_ttl_sec=120.0)
    assert geo_analyzer._cached_assessment is not None
    assert geo_analyzer._cached_assessment.threat_level == "NORMAL"
    assert geo_analyzer._cached_assessment.recommended_brake == "NONE"
    assert geo_analyzer._cached_assessment.headlines_analyzed == 0

    adv_file = tmp_path / "sentiment_eval.json"
    adv_file.write_text(
        json.dumps({
            "market_bias": None,
            "risk_level": None,
            "confidence": None,
            "summary": None,
            "key_risks": None,
            "recommended_action": None,
            "ts": now_curr,
        }),
        encoding="utf-8",
    )
    adv = SentimentAdvisor(cache_file=str(adv_file), cache_ttl_sec=3600.0)
    assert adv._cached_assessment is not None
    assert adv._cached_assessment.market_bias == "NEUTRAL"
    assert adv._cached_assessment.recommended_action == "HOLD"
    assert adv._cached_assessment.key_risks == []

def test_smartstoplossguard_preserves_hard_disaster_floor_when_disaster_buffer_pct_nan_iter31_s5():
    """[ITER31 #5] SmartStopLossGuard preserves Hard Disaster Floor when disaster_buffer_pct=NaN."""
    # 5. SmartStopLossGuard preserves Hard Disaster Floor when disaster_buffer_pct=NaN
    ssl = SmartStopLossGuard(mode="enforce", max_threshold_pct=6.0, disaster_buffer_pct=0.85)
    should_exec, reason, meta = ssl.evaluate(
        "BTCUSDC",
        price_decrease=0.04,       # 4.0% loss
        lost_threshold=0.02925,    # 2.925% SL + 0.85% buf = 3.775% disaster floor
        disaster_buffer_pct=float("nan"),
        provider="sim",
    )
    assert should_exec is True
    assert "hard_disaster_floor_breached" in reason
    assert math.isfinite(meta["disaster_floor_pct"])

def test_smartstoplossguard_evaluate_and_on_recovered_with_now_must_not_iter33_s1(monkeypatch):
    """[ITER33 #1] SmartStopLossGuard.evaluate and on_recovered with now=0.0 must not invoke time.time()."""
    def _forbidden_time():
        raise AssertionError("time.time() must not be invoked when an explicit finite timestamp is provided!")
    # 1. SmartStopLossGuard.evaluate and on_recovered with now=0.0 must not invoke time.time()
    ssl_guard = SmartStopLossGuard(mode="off")
    monkeypatch.setattr(time, "time", _forbidden_time)
    should_exec, reason, meta = ssl_guard.evaluate(
        "BTCUSDC",
        0.04,
        0.03,
        now=0.0,
        provider="test",
        current_price=96.0,
        buy_price=100.0,
    )
    assert should_exec is True and meta["now"] == 0.0

def test_smartstoplossguard_mode_false_handling_iter101_s3():
    """[ITER101 #3] SmartStopLossGuard mode=False handling."""
    # 3. SmartStopLossGuard mode=False handling
    sslg = SmartStopLossGuard(mode=False)
    ok_sl, reason_sl, _ = sslg.evaluate("BTCUSDC", 0.04, 0.03)
    assert ok_sl is True and reason_sl == "smart_sl_disabled"

def test_smartstoplossguard__is_simulated_provider_handles_uppercase_replay_mode_backtest_mode_and_mode_iter125_s2(monkeypatch):
    """[ITER125 #2] SmartStopLossGuard._is_simulated_provider handles uppercase REPLAY_MODE/BACKTEST_MODE and mode='true'."""
    # 2. SmartStopLossGuard._is_simulated_provider handles uppercase REPLAY_MODE/BACKTEST_MODE and mode="true"
    monkeypatch.setenv("REPLAY_MODE", "TRUE")
    assert _is_simulated_provider("binance") is True
    monkeypatch.delenv("REPLAY_MODE", raising=False)
    sl_guard = SmartStopLossGuard(mode="true")
    assert sl_guard.mode == "enforce"

def test_smartstoplossguard_helper_deduplications_iter182_s4():
    """[ITER182 #4] SmartStopLossGuard helper deduplications."""
    # 4. SmartStopLossGuard helper deduplications
    assert _format_price_part(105.0, 100.0) == " | $105.00 (Last Buy Price: $100.00)"
    assert _format_price_part(105.0, None) == " @ $105.00"
    assert _format_price_part(None, None) == ""
    assert _is_buy_price_changed(100.0, 100.0) is False
    assert _is_buy_price_changed(100.0, 101.0) is True
