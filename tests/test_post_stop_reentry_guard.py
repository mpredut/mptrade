"""Unit tests for Pillar 3 Post-Stop Re-entry AI Validator (Gemini LLM).

Verifies:
1. Disabled mode ("off") allows immediately without invoking LLM.
2. Gemini APPROVED allows re-entry with full position size (scale 1.0).
3. Gemini REJECTED blocks re-entry (veto), preventing bull trap / dead-cat bounce re-entries.
4. Gemini DOWNSCALE reduces position size (e.g. 50%).
5. Shadow mode evaluates and notifies without blocking execution.
6. Network timeout / LLM failure properly falls back (allow / downscale / veto).
7. Prompt correctly incorporates whale taker flow, drop from SL, and bounce metrics.
8. End-to-end integration with monitortrades.py post-stop bounce flow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import monitortrades as mt
from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard


class FakeGeminiClient(GeminiClient):
    """Mock GeminiClient returning deterministic JSON responses."""

    def __init__(self, response_dict=None, raises_timeout=False):
        super().__init__()
        self.response_dict = response_dict
        self.raises_timeout = raises_timeout
        self.last_prompt = None

    def query_json(self, prompt: str, timeout_sec: float = 25.0, thread_title=None):
        self.last_prompt = prompt
        if self.raises_timeout:
            return None
        return self.response_dict


def test_disabled_mode_allows():
    client = FakeGeminiClient(response_dict={"decision": "REJECTED", "reason": "Trap"})
    guard = PostStopReentryGuard(gemini_client=client, mode="off", notify_enabled=False)
    dec = guard.check(
        symbol="TAOUSDC",
        current_price=264.0,
        last_sell_price=260.0,
        trough=258.0,
        bounce_pct=0.023,
    )
    assert dec.allowed is True
    assert dec.reason == "post_stop_reentry_guard_disabled"
    assert client.last_prompt is None


def test_gemini_approved_allows_full_reentry():
    client = FakeGeminiClient(response_dict={
        "decision": "APPROVED",
        "suggested_scale": 1.0,
        "reason": "Strong whale accumulation with rising Open Interest",
    })
    guard = PostStopReentryGuard(gemini_client=client, mode="enforce", notify_enabled=False)
    dec = guard.check(
        symbol="TAOUSDC",
        current_price=264.0,
        last_sell_price=260.0,
        trough=258.0,
        bounce_pct=0.023,
        qty=5.0,
    )
    assert dec.allowed is True
    assert dec.brake_action == BrakeAction.NONE
    assert dec.suggested_scale == 1.0
    assert "Gemini approved post-stop re-entry" in dec.reason
    assert "Strong whale accumulation" in dec.reason
    assert client.last_prompt is not None
    assert "WHALE TAKER FLOW SUPREMACY" in client.last_prompt


def test_gemini_rejected_vetoes_reentry():
    client = FakeGeminiClient(response_dict={
        "decision": "REJECTED",
        "suggested_scale": 0.0,
        "reason": "Whale taker flow is 0.62 net selling; bounce is a short-covering fakeout",
    })
    guard = PostStopReentryGuard(gemini_client=client, mode="enforce", notify_enabled=False)
    dec = guard.check(
        symbol="TAOUSDC",
        current_price=264.0,
        last_sell_price=260.0,
        trough=258.0,
        bounce_pct=0.023,
        qty=5.0,
    )
    assert dec.allowed is False
    assert dec.brake_action == BrakeAction.HARD_VETO
    assert dec.suggested_scale == 0.0
    assert "Gemini vetoed post-stop re-entry" in dec.reason
    assert "short-covering fakeout" in dec.reason


def test_gemini_downscaled_reduces_scale():
    client = FakeGeminiClient(response_dict={
        "decision": "DOWNSCALE",
        "suggested_scale": 0.5,
        "reason": "Uncertain bounce momentum, whale flow neutral",
    })
    guard = PostStopReentryGuard(gemini_client=client, mode="enforce", notify_enabled=False)
    dec = guard.check(
        symbol="TAOUSDC",
        current_price=264.0,
        last_sell_price=260.0,
        trough=258.0,
        bounce_pct=0.023,
        qty=5.0,
    )
    assert dec.allowed is True
    assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
    assert dec.suggested_scale == 0.5
    assert "Gemini downscaled post-stop re-entry" in dec.reason


def test_shadow_mode_logs_without_blocking():
    client = FakeGeminiClient(response_dict={
        "decision": "REJECTED",
        "suggested_scale": 0.0,
        "reason": "Suspected dead-cat bounce",
    })
    guard = PostStopReentryGuard(gemini_client=client, mode="shadow", notify_enabled=False)
    dec = guard.check(
        symbol="TAOUSDC",
        current_price=264.0,
        last_sell_price=260.0,
        trough=258.0,
        bounce_pct=0.023,
    )
    assert dec.allowed is True
    assert "shadow_mode" in dec.reason
    assert "Suspected dead-cat bounce" in dec.reason


def test_fallback_handling_on_timeout():
    # Fallback allow
    client = FakeGeminiClient(raises_timeout=True)
    guard_allow = PostStopReentryGuard(gemini_client=client, fallback_action="allow", notify_enabled=False)
    dec_allow = guard_allow.check(symbol="TAOUSDC", current_price=264.0, last_sell_price=260.0, trough=258.0, bounce_pct=0.023)
    assert dec_allow.allowed is True
    assert dec_allow.reason == "llm_query_timeout_fallback_allowed"

    # Fallback veto
    guard_veto = PostStopReentryGuard(gemini_client=client, fallback_action="veto", notify_enabled=False)
    dec_veto = guard_veto.check(symbol="TAOUSDC", current_price=264.0, last_sell_price=260.0, trough=258.0, bounce_pct=0.023)
    assert dec_veto.allowed is False
    assert dec_veto.reason == "llm_query_timeout_fail_closed"

    # Fallback downscale
    guard_down = PostStopReentryGuard(gemini_client=client, fallback_action="downscale", notify_enabled=False)
    dec_down = guard_down.check(symbol="TAOUSDC", current_price=264.0, last_sell_price=260.0, trough=258.0, bounce_pct=0.023)
    assert dec_down.allowed is True
    assert dec_down.brake_action == BrakeAction.DOWNSCALE_QTY
    assert dec_down.suggested_scale == 0.5


def test_telemetry_prompt_context():
    client = FakeGeminiClient(response_dict={"decision": "APPROVED", "reason": "ok"})
    guard = PostStopReentryGuard(gemini_client=client, notify_enabled=False)
    custom_telemetry = {
        "Whale Taker Flow": "Buyer/Seller ratio: 1.45 (regime: accumulation)",
        "Whale Positioning": "72.5% Top Traders Long",
    }
    guard.check(
        symbol="TAOUSDC",
        current_price=265.0,
        last_sell_price=260.0,
        trough=250.0,
        bounce_pct=0.06,
        last_sell_time_s=1700000000.0,
        telemetry=custom_telemetry,
        now=1700000600.0,  # 10 minutes later
    )
    prompt = client.last_prompt
    assert "Symbol: TAOUSDC" in prompt
    assert "Previous Stop-Loss Exit Price: $260.0000" in prompt
    assert "Lowest Post-Stop Trough: $250.0000 (-3.85% drop from SL)" in prompt
    assert "Current Recovery Price: $265.0000 (+6.00% bounce from trough)" in prompt
    assert "Elapsed Time Since Stop-Loss: 10.0 minutes" in prompt
    assert "Whale Taker Flow: Buyer/Seller ratio: 1.45 (regime: accumulation)" in prompt
    assert "Whale Positioning: 72.5% Top Traders Long" in prompt


class MockInstrument(mt._Instrument):
    def __init__(self, symbol="TAOUSDC", price=103.0, free=0.0, min_qty=0.1, buy_qty=None, orders_dict=None):
        self.symbol = symbol
        self.name = symbol
        self.isolation = "fleet"
        self._provider = "binance"
        self.enabled = True
        self._price = price
        self._free = free
        self._min_qty = min_qty
        self._buy_qty = buy_qty
        self._orders_dict = orders_dict or {}

    def price(self):
        return self._price

    def free(self):
        return self._free

    def min_qty(self):
        return self._min_qty

    def param(self, domain, key, default=None, *args):
        if key == "buy_qty" and self._buy_qty is not None:
            return self._buy_qty
        return default

    def orders(self, side, since_s=None):
        return self._orders_dict.get(side, [])


def _run_monitortrades_reentry_tick(decision, *, buy_qty=None, place_return=True):
    fake_guard = MagicMock()
    fake_guard.check.return_value = decision

    mt._LAST_EXIT_KIND["TAOUSDC"] = "STOP"
    mt._REENTRY_STATE["TAOUSDC"] = {
        "trough": 100.0,
        "peak": 105.0,
        "peak_bounce": 102.0,
        "started_at": 1000.0,
    }
    inst = MockInstrument(
        symbol="TAOUSDC",
        price=103.0,
        buy_qty=buy_qty,
        orders_dict={"SELL": [{"price": 101.0, "time": 1000.0, "timestamp": 1000000, "qty": buy_qty or 1.0, "kind": "STOP"}]},
    )
    exit_mgr = MagicMock()
    exit_mgr.evaluate_reentry.return_value = MagicMock(should_enter=True, reason="bounce confirmed")

    with patch("intelligence.sentiment.guards.post_stop_reentry_guard.get_post_stop_reentry_guard", return_value=fake_guard), \
         patch.object(mt, "_place_guarded", return_value=place_return) as mock_place, \
         patch.object(mt, "get_position_stats", return_value={"average_buy_price": 0.0, "average_sell_price": 0.0, "net_qty": 0.0}), \
         patch.object(mt, "get_exit_manager", return_value=exit_mgr), \
         patch.object(mt, "is_trend_up", return_value=True), \
         patch.object(mt, "get_symbol_volatility_1h", return_value=0.01):
        mt.monitor_price_and_trade(inst, sbs=10, now_fn=lambda: 1500.0)

    return fake_guard, mock_place


def test_monitortrades_integration_veto():
    """Verify monitortrades aborts buy order when PostStopReentryGuard vetoes."""
    fake_guard, mock_place = _run_monitortrades_reentry_tick(
        GuardDecision.veto("PostStopReentryGuard", "Gemini vetoed post-stop re-entry: trap detected"),
    )
    fake_guard.check.assert_called_once()
    mock_place.assert_not_called()
    assert "TAOUSDC" in mt._REENTRY_STATE


def test_monitortrades_integration_downscale():
    """Verify monitortrades scales down quantity when PostStopReentryGuard downscales."""
    fake_guard, mock_place = _run_monitortrades_reentry_tick(
        GuardDecision.downscale("PostStopReentryGuard", scale=0.5, reason="moderate whale momentum"),
        buy_qty=2.0,
    )
    fake_guard.check.assert_called_once()
    mock_place.assert_called_once()
    args, kwargs = mock_place.call_args
    assert args[3] == 1.0
    assert kwargs.get("bypass_guards") == {"GF1", "GI3"}


def test_decision_caching_and_notify_cooldown():
    """Verify repeated checks for the same post-stop exit episode reuse the cached LLM decision
    and suppress duplicate ntfy alerts within cache_ttl_sec."""
    from intelligence.sentiment.guards.post_stop_reentry_guard import clear_post_stop_reentry_cache

    clear_post_stop_reentry_cache()
    call_count = 0

    class CountingGeminiClient(FakeGeminiClient):
        def query_json(self, prompt: str, timeout_sec: float = 25.0, thread_title=None):
            nonlocal call_count
            call_count += 1
            return super().query_json(prompt, timeout_sec=timeout_sec, thread_title=thread_title)

    client = CountingGeminiClient(response_dict={
        "decision": "APPROVED",
        "suggested_scale": 1.0,
        "reason": "Whale accumulation confirmed",
    })
    guard = PostStopReentryGuard(
        gemini_client=client,
        mode="enforce",
        notify_enabled=True,
        cache_ttl_sec=600.0,
    )
    with patch("alertnotifiers.notify") as mock_notify:
        dec1 = guard.check(
            symbol="ARBUSDC",
            current_price=0.1930,
            last_sell_price=0.1884,
            trough=0.1880,
            bounce_pct=0.026,
            last_sell_time_s=1700000000.0,
            now=1700000100.0,
        )
        # Second tick 48 seconds later (same sell episode, slightly different current_price)
        dec2 = guard.check(
            symbol="ARBUSDC",
            current_price=0.1935,
            last_sell_price=0.1884,
            trough=0.1880,
            bounce_pct=0.029,
            last_sell_time_s=1700000000.0,
            now=1700000148.0,
        )
        assert dec1.allowed is True
        assert dec2.allowed is True
        assert call_count == 1, "Gemini LLM must only be queried once within cache_ttl_sec"
        assert mock_notify.call_count == 1, "Notification must only be sent once within cache_ttl_sec"

        # After cache_ttl_sec expires (e.g. +605s), a fresh query is allowed
        dec3 = guard.check(
            symbol="ARBUSDC",
            current_price=0.1940,
            last_sell_price=0.1884,
            trough=0.1880,
            bounce_pct=0.032,
            last_sell_time_s=1700000000.0,
            now=1700000705.0,
        )
        assert dec3.allowed is True
        assert call_count == 2
        assert mock_notify.call_count == 2
    clear_post_stop_reentry_cache()


def test_monitortrades_skips_ai_guard_when_weight_policy_blocks_buy():
    """Verify monitortrades does not invoke PostStopReentryGuard (or send APPROVED alerts)
    when the provider's 24h weight policy refuses the BUY with qty_zero_after_policy."""
    from providers.quantity import QuantityDecision

    class PolicyBlockedProvider:
        name = "binance"

        def free_balance(self, asset):
            return 20338.80

        def quantity_decision(self, symbol, side, price, qty, **kwargs):
            return QuantityDecision(
                requested_qty=qty,
                balance_cap=20338.80 / price,
                policy_cap=0.0,
                fee_cap=0.0,
                final_qty=0.0,
                refuse_reason="qty_zero_after_policy",
                balance_asset="USDC",
            )

    fake_guard = MagicMock()
    mt._LAST_EXIT_KIND["ARBUSDC"] = "STOP"
    mt._REENTRY_STATE["ARBUSDC"] = {
        "trough": 0.1884,
        "peak": 0.1950,
        "peak_bounce": 0.1896,
        "started_at": 1000.0,
        "sell_time": 1000.0,
    }
    inst = MockInstrument(
        symbol="ARBUSDC",
        price=0.1896,
        buy_qty=1318.56,
        orders_dict={
            "SELL": [{
                "price": 0.1884,
                "time": 1000.0,
                "timestamp": 1000000,
                "qty": 1318.56,
                "kind": "STOP",
            }]
        },
    )
    inst._provider = PolicyBlockedProvider()
    exit_mgr = MagicMock()
    exit_mgr.evaluate_reentry.return_value = MagicMock(
        should_enter=True, reason="post_stop_bounce_confirmed"
    )

    with patch("intelligence.sentiment.guards.post_stop_reentry_guard.get_post_stop_reentry_guard", return_value=fake_guard), \
         patch.object(mt, "_place_guarded", return_value=True) as mock_place, \
         patch.object(mt, "get_position_stats", return_value={"average_buy_price": 0.0, "average_sell_price": 0.0, "net_qty": 0.0}), \
         patch.object(mt, "get_exit_manager", return_value=exit_mgr), \
         patch.object(mt, "is_trend_up", return_value=True), \
         patch.object(mt, "get_symbol_volatility_1h", return_value=0.01):
        mt.monitor_price_and_trade(inst, sbs=10, now_fn=lambda: 1500.0)

    fake_guard.check.assert_not_called()
    mock_place.assert_not_called()
    assert "ARBUSDC" in mt._REENTRY_STATE


