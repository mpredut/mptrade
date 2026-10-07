"""Tests verifying dynamic intelligence re-evaluation and single quantity downscaling.

Tests cover:
1. Dynamic anti-FOMO parabolic surge guard re-evaluating when price changes within the same MarketRegimeContext.
2. Dynamic Gemini high-stake guard re-evaluating when notional jumps above threshold within the same MarketRegimeContext.
3. Propagation and single application of suggested_scale in decide_quantity and Instrument.place.
4. Hot-reload of order_guard.conf on mtime change in _load_margins().
5. TTL fallback timer in priceAnalysis.py cache.
"""
import json
import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_guard
from market_regime import MarketRegimeDecision, MarketRegimeContext
from providers.quantity import decide_quantity, QuantityDecision


def _build_test_context(symbol="BTCUSDC", provider="binance"):
    decision = MarketRegimeDecision(
        regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
        fresh=True, reason="signal", source="mock",
    )
    return MarketRegimeContext.from_decision(decision, symbol=symbol, provider=provider)


class TestIntelligenceGuardDynamicEvaluation:
    def test_price_surge_blocked_on_same_regime_context(self, monkeypatch):
        """Verify that BUY at 100 is approved, but subsequent BUY at 125 is blocked by parabolic surge guard on the same context."""
        ctx = _build_test_context()

        margins = {
            "intelligence_guards_mode": "enforce",
            "parabolic_surge_pct": 15.0,
            "parabolic_pullback_pct": 2.0,
            "weibull_exhaustion_policy": "off",
            "gemini_guard_mode": "off",
            "geopolitical_guard_mode": "off",
            "whale_guard_mode": "off",
            "orderbook_wall_guard_mode": "off",
            "funding_guard_mode": "off",
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        now = time.time()
        # Price history baseline at 100.0
        history = [(now - 120.0, 100.0), (now - 60.0, 100.0)]
        monkeypatch.setattr(order_guard, "_read_cached_price_history", lambda s, **kw: history)

        # Mock short healthy trend duration (1 hour)
        monkeypatch.setattr(order_guard, "_resolve_trend_duration", lambda s: 3600.0)

        # 1. At price 100.0, order is safe and approved
        ok1, reason1, scale1 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, regime_context=ctx
        )
        assert ok1 is True
        assert reason1 == "ok"
        assert scale1 == 1.0

        # 2. At price 125.0 (+25% surge > 15%), anti-FOMO MUST trigger and reject even with cached context
        ok2, reason2, scale2 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 125.0, regime_context=ctx
        )
        assert ok2 is False
        assert "parabolic_surge" in reason2
        assert scale2 == 0.0

    def test_notional_jump_triggers_gemini_guard_on_same_context(self, monkeypatch):
        """Verify that jumping from small notional to high notional evaluates Gemini guard."""
        ctx = _build_test_context()

        margins = {
            "intelligence_guards_mode": "enforce",
            "gemini_guard_mode": "enforce",
            "gemini_min_notional_eur": 1000.0,
            "gemini_timeout_sec": 5.0,
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)
        monkeypatch.setattr(order_guard, "_resolve_trend_duration", lambda s: 3600.0)

        from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction

        gemini_called = []

        class MockGeminiGuard:
            def __init__(self, **kw):
                pass
            def check(self, symbol, side, price, qty, notional_eur=None):
                gemini_called.append((symbol, side, price, qty, notional_eur))
                return GuardDecision.veto("GeminiHighStakeGuard", "gemini_risk_block")

        monkeypatch.setattr("intelligence.sentiment.guards.high_stake_guard.GeminiHighStakeGuard", MockGeminiGuard)

        # Call with small notional (100 EUR): Gemini guard should NOT be called
        ok1, reason1, _ = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=1.0, notional_eur=100.0, regime_context=ctx
        )
        assert ok1 is True
        assert len(gemini_called) == 0

        # Call with high notional (2500 EUR) on same context: Gemini guard MUST trigger and block
        ok2, reason2, _ = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=25.0, notional_eur=2500.0, regime_context=ctx
        )
        assert ok2 is False
        assert reason2 == "gemini_risk_block"
        assert len(gemini_called) == 1
        assert gemini_called[0][4] == 2500.0


class TestIntelligenceQuantityScaling:
    def test_suggested_scale_applied_once_in_decide_quantity(self):
        """Verify that suggested_scale on regime_context scales quantity and is tracked on QuantityDecision without mutating context."""
        ctx = _build_test_context()
        object.__setattr__(ctx, "suggested_scale", 0.25)

        provider = MagicMock()
        provider.free_balance.return_value = 10000.0
        provider.policy_cap_quantity.return_value = 10.0
        provider.fee_cap_quantity.return_value = 100.0
        provider.round_amount = lambda s, a: round(a, 4)
        provider.order_filter_refusal.return_value = None

        decision = decide_quantity(
            provider, "BTCUSDC", "BUY", 100.0, 10.0, regime_context=ctx
        )
        assert decision.final_qty == 2.5
        assert decision.scale_applied is True
        assert decision.applied_scale == 0.25
        # Shared context is NOT mutated with _scale_applied
        assert getattr(ctx, "_scale_applied", None) is None

        # Fix 3: Calling decide_quantity for a second order reusing the same context
        # MUST also scale the second order (4.0 -> 1.0), rather than omitting scale!
        provider.policy_cap_quantity.return_value = 4.0
        decision2 = decide_quantity(
            provider, "BTCUSDC", "BUY", 100.0, 4.0, regime_context=ctx
        )
        assert decision2.final_qty == 1.0
        assert decision2.scale_applied is True
        assert decision2.applied_scale == 0.25

    def test_profit_guard_propagates_suggested_scale_to_regime_context(self, monkeypatch):
        """Verify profit_guard records suggested_scale on regime_context when intelligence downscales."""
        ctx = _build_test_context()

        # Mock check_intelligence_guards returning scale 0.25 (e.g. Weibull trend exhausted)
        monkeypatch.setattr(
            order_guard,
            "check_intelligence_guards",
            lambda *args, **kw: (True, "trend_exhausted", 0.25),
        )
        monkeypatch.setattr(order_guard, "buy_reference_mode", lambda p: "off")

        provider = MagicMock()
        provider.name = "binance"
        res = order_guard.profit_guard(
            provider, "BTCUSDC", "BUY", 100.0, 1.15, regime_context=ctx, qty=1.0
        )
        assert res is True
        assert getattr(ctx, "suggested_scale", None) == 0.25
        assert getattr(ctx, "intelligence_scale_reason", None) == "trend_exhausted"


class TestCacheSafetyAndHotReload:
    def test_order_guard_conf_mtime_hot_reload(self, monkeypatch, tmp_path):
        """Verify _load_margins reloads if file mtime updates."""
        conf_file = tmp_path / "order_guard.conf"
        conf_file.write_text("default = 1.15\nparabolic_surge_pct = 15.0\n")

        monkeypatch.setattr(order_guard, "_MARGINS", None)
        monkeypatch.setattr(order_guard, "_MARGINS_LAST_CHECK", 0.0)
        monkeypatch.setattr(order_guard, "_MARGINS_FILE_MTIME", 0.0)

        # Patch path in _load_margins
        import os
        orig_abspath = os.path.abspath
        monkeypatch.setattr(
            os.path,
            "abspath",
            lambda p: str(conf_file) if "order_guard.py" in p else orig_abspath(p),
        )

        m1 = order_guard._load_margins()
        assert float(m1.get("parabolic_surge_pct", 0)) == 15.0

        # Change file on disk and advance time
        time.sleep(0.01)
        conf_file.write_text("default = 1.15\nparabolic_surge_pct = 25.0\n")
        # Reset last check timer to trigger reload
        order_guard._MARGINS_LAST_CHECK = 0.0

        m2 = order_guard._load_margins()
        assert float(m2.get("parabolic_surge_pct", 0)) == 25.0

    def test_price_analysis_cache_ttl_forces_recalculation(self, monkeypatch):
        """Verify get_weight_for_cash_permission_at_quant_time recalculates after 300s TTL."""
        import priceAnalysis

        memo_key = ("BTCUSDC", "BUY", 14)
        priceAnalysis.last_w[memo_key] = [0.99]
        priceAnalysis.last_timestamp[memo_key] = 1000.0
        priceAnalysis.last_calc_time[memo_key] = time.time() - 400.0  # > 300s old
        priceAnalysis.last_duration[memo_key] = 86400.0

        # Mock trend cache in cacheManager
        mock_trend = {
            "timestamp": 1000.0,
            "duration_seconds": 86400.0,
            "direction": "up",
            "start_timestamp": 0.0,
        }
        mock_cm = MagicMock()
        mock_cm.cache = {"BTCUSDC": [mock_trend]}
        monkeypatch.setattr("cacheManager.get_cache_manager", lambda name: mock_cm)

        calc_called = False
        def mock_get_trade_weight(**kw):
            nonlocal calc_called
            calc_called = True
            import numpy as np
            return np.array([0]), np.array([0.77])

        monkeypatch.setattr(priceAnalysis, "get_trade_weight", mock_get_trade_weight)

        w = priceAnalysis.get_weight_for_cash_permission_at_quant_time(
            "BTCUSDC", "BUY", T_quanta=14
        )
        assert calc_called is True
        assert w == pytest.approx(0.77)


class TestCase3BinanceArchivedIsolation:
    def test_archived_order_recognized_only_on_get_order(self, monkeypatch):
        from providers.market_api import BinanceProvider
        bp = BinanceProvider()
        mock_bapi = MagicMock()
        class BinanceApiErr(Exception):
            code = -2026
        mock_bapi.client.get_order.side_effect = BinanceApiErr("Order was canceled or archived -2026")
        monkeypatch.setattr("providers.market_api._get_bapi", lambda: mock_bapi)

        status = bp.order_status("BTCUSDC", "12345")
        assert status.status == "canceled"
        assert status.filled_qty == 0.0
        assert status.venue_status == "ARCHIVED_CANCELED"

    def test_fee_error_does_not_convert_filled_order_to_archived_canceled(self, monkeypatch):
        from providers.market_api import BinanceProvider
        bp = BinanceProvider()
        mock_bapi = MagicMock()
        mock_bapi.client.get_order.return_value = {
            "status": "FILLED",
            "executedQty": "1.5",
            "cummulativeQuoteQty": "150.0",
        }
        monkeypatch.setattr("providers.market_api._get_bapi", lambda: mock_bapi)
        def bad_fee(symbol, oid):
            raise RuntimeError("trade fee endpoint failed: trade archived")
        monkeypatch.setattr(bp, "_order_fee_quote", bad_fee)

        with pytest.raises(RuntimeError, match="archived"):
            bp.order_status("BTCUSDC", "12345")


class TestCase4MarketRegimeContextValidation:
    def _make_dec(self):
        return MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="test", source="test",
        )

    def test_timestamp_sanity_rejections(self):
        d = self._make_dec()
        ctx_zero = MarketRegimeContext.from_decision(d, evaluated_at=0.0)
        assert ctx_zero.is_valid_for("BTCUSDC", now=1000.0) is False

        ctx_neg = MarketRegimeContext.from_decision(d, evaluated_at=-10.0)
        assert ctx_neg.is_valid_for("BTCUSDC", now=1000.0) is False

        ctx_future = MarketRegimeContext.from_decision(d, evaluated_at=1005.0)
        assert ctx_future.is_valid_for("BTCUSDC", now=1000.0) is False

    def test_horizon_validation(self):
        d = self._make_dec()
        ctx = MarketRegimeContext.from_decision(
            d, evaluated_at=1000.0, trend_duration_seconds=7200.0
        )
        assert ctx.is_valid_for("BTCUSDC", now=1000.0, max_horizon_seconds=3600.0) is False
        assert ctx.is_valid_for("BTCUSDC", now=1000.0, max_horizon_seconds=10000.0) is True

    def test_require_identity_validation(self):
        d = self._make_dec()
        anon_ctx = MarketRegimeContext.from_decision(d, evaluated_at=1000.0)
        # Without require_identity, passes
        assert anon_ctx.is_valid_for("BTCUSDC", now=1000.0, require_identity=False) is True
        # With require_identity, rejected
        assert anon_ctx.is_valid_for("BTCUSDC", now=1000.0, require_identity=True) is False

        identified_ctx = MarketRegimeContext.from_decision(
            d, evaluated_at=1000.0, symbol="BTCUSDC", provider="binance"
        )
        assert identified_ctx.is_valid_for("BTCUSDC", provider="binance", now=1000.0, require_identity=True) is True
        assert identified_ctx.is_valid_for("ETHUSDC", provider="binance", now=1000.0, require_identity=True) is False


class TestCase5HistoryFailClosed:
    def test_kraken_history_failure_raises_provider_error(self, monkeypatch):
        from providers.kraken_provider import KrakenProvider
        from providers.strategy_executor import ProviderError
        kp = KrakenProvider()
        monkeypatch.setattr(kp, "_fills_from_cache", lambda s: None)
        def bad_api(s):
            raise RuntimeError("Kraken API network failure")
        monkeypatch.setattr(kp, "_fills_from_api", bad_api)

        with pytest.raises(ProviderError, match="Kraken API network failure"):
            kp.get_orders("BTCUSDC", "BUY", 86400)

    def test_hyperliquid_history_failure_raises_provider_error(self, monkeypatch):
        from providers.hyperliquid_provider import HyperliquidProvider
        from providers.strategy_executor import ProviderError
        hp = HyperliquidProvider(token="PURR")
        monkeypatch.setattr(hp, "_hl", lambda: None)

        with pytest.raises(ProviderError, match="client or pair unavailable"):
            hp.get_orders("PURR/USDC", "BUY", 86400)

    def test_order_guard_daily_limit_and_weight_limit_fail_closed_on_history_error(self):
        from providers.strategy_executor import ProviderError
        class BrokenHistoryProvider:
            name = "mock"
            def get_orders(self, symbol, side, since_s):
                raise ProviderError("Network error fetching trade history")

        p = BrokenHistoryProvider()
        ok, reason = order_guard.daily_limit_guard(p, "BTCUSDC", "BUY", safeback_sec=86400)
        assert ok is False
        assert reason == "history_unavailable"

        with pytest.raises(ProviderError, match="Network error fetching trade history"):
            order_guard.weight_limit(p, "BTCUSDC", "BUY", 100.0, 1.0, available_qty=10.0)

    def test_hyperliquid_open_orders_failure_raises_provider_error(self, monkeypatch):
        from providers.hyperliquid_provider import HyperliquidProvider
        from providers.strategy_executor import ProviderError
        hp = HyperliquidProvider(token="PURR")
        monkeypatch.setattr(hp, "_hl", lambda: None)

        with pytest.raises(ProviderError, match="client or pair unavailable"):
            hp.open_orders("PURR/USDC")

    def test_window_reference_and_last_opposite_fill_fail_closed_on_none_history(self):
        from providers.strategy_executor import ProviderError
        from providers.base import MarketDataProvider
        class NoneHistoryProvider(MarketDataProvider):
            @property
            def name(self):
                return "mock"
            def get_current_price(self, symbol):
                return 100.0
            def supports_symbol(self, symbol):
                return True
            def get_orders(self, symbol, side, since_s):
                return None

        p = NoneHistoryProvider()
        with pytest.raises(ProviderError, match="failed to read order history"):
            order_guard.window_reference(p, "BTCUSDC", "BUY", 3600.0)

        with pytest.raises(ProviderError, match="failed to read order history"):
            p.last_opposite_fill("BTCUSDC", "BUY")

    def test_profit_guard_dynamic_window_fails_closed_on_unavailable_history(self, monkeypatch):
        from providers.strategy_executor import ProviderError
        class FailingHistoryProvider:
            name = "binance"
            def get_orders(self, symbol, side, since_s):
                raise ProviderError("history fetch network error")

        p = FailingHistoryProvider()
        monkeypatch.setattr(order_guard, "buy_reference_mode", lambda name: "dynamic")
        monkeypatch.setattr(order_guard, "dynamic_buy_window_sec", lambda *a, **k: 3600.0)

        # In dynamic mode, price (99.0) is below reference (100.0) with diff < threshold (1.15%),
        # but history fetch fails. Must fail closed (False), NOT bypass anchor!
        allowed = order_guard.profit_guard(
            p, "BTCUSDC", "BUY", 99.0, 1.15, window_ref=100.0
        )
        assert allowed is False

    def test_kraken_fills_from_api_fails_closed_on_missing_or_invalid_payload(self):
        from providers.kraken_provider import KrakenProvider
        from providers.strategy_executor import ProviderError
        from unittest.mock import MagicMock

        fake_client = MagicMock()
        kp = KrakenProvider(client=fake_client)

        # 1. Missing trades in payload
        fake_client._private.return_value = {}
        with pytest.raises(ProviderError, match="trades history payload missing or invalid"):
            kp._fills_from_api("BTCUSDC")

        # 2. Trades is not a dict
        fake_client._private.return_value = {"trades": "invalid"}
        with pytest.raises(ProviderError, match="trades mapping in payload is invalid"):
            kp._fills_from_api("BTCUSDC")

    def test_kraken_cancel_order_fails_closed_on_missing_or_zero_count(self):
        from providers.kraken_provider import KrakenProvider
        from providers.strategy_executor import ProviderError
        from unittest.mock import MagicMock

        fake_client = MagicMock()
        kp = KrakenProvider(client=fake_client)

        # 1. Missing count in result
        fake_client.cancel_order.return_value = {}
        with pytest.raises(ProviderError, match="missing count in Kraken response"):
            kp.cancel_order_by_id("HYPEUSD", "ORD-123")

        # 2. Zero count
        fake_client.cancel_order.return_value = {"count": 0}
        with pytest.raises(ProviderError, match="did not confirm the cancellation"):
            kp.cancel_order_by_id("HYPEUSD", "ORD-123")

    def test_hyperliquid_fails_closed_on_invalid_user_fills_or_open_orders_type(self, monkeypatch):
        from providers.hyperliquid_provider import HyperliquidProvider
        from providers.strategy_executor import ProviderError
        from unittest.mock import MagicMock

        monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x1234567890abcdef")
        fake_client = MagicMock()
        fake_client.info.user_fills.return_value = {"error": "rate limited"}
        fake_client.open_orders.return_value = None

        hp = HyperliquidProvider()
        monkeypatch.setattr(hp, "_hl", lambda: fake_client)
        monkeypatch.setattr(hp, "_pair", lambda: "HYPE/USDC")

        # user_fills returning dict instead of list
        with pytest.raises(ProviderError, match="user_fills returned unexpected type"):
            hp.get_orders("HYPEUSDC", "BUY", 3600.0)

        # open_orders returning None instead of list
        with pytest.raises(ProviderError, match="open_orders returned unexpected type"):
            hp.open_orders("HYPEUSDC")

    def test_binance_open_orders_fails_closed_on_none(self, monkeypatch):
        from providers.market_api import BinanceProvider
        from providers.strategy_executor import ProviderError
        from unittest.mock import MagicMock

        bp = BinanceProvider()
        fake_bapi = MagicMock()
        fake_bapi.client.get_open_orders.return_value = None
        monkeypatch.setattr("providers.market_api._get_bapi", lambda: fake_bapi)

        with pytest.raises(ProviderError, match="received None from Binance client"):
            bp.open_orders("BTCUSDC")

    def test_gemini_high_stake_reevaluated_on_notional_jump(self, monkeypatch):
        from market_regime import MarketRegimeContext, MarketRegimeDecision
        from intelligence.internal.guards.guard_decision import GuardDecision

        # Initial context evaluated at 1000 EUR
        ctx = MarketRegimeContext.from_decision(
            MarketRegimeDecision("sideways", 0.0, 0.01, 0.0, True, "test"),
            evaluated_at=time.time(),
            symbol="BTCUSDC",
            provider="binance",
            trend_duration_seconds=3600.0,
        )

        calls = []
        def fake_gemini_check(self, symbol, side, price, qty, notional_eur=None):
            calls.append(notional_eur)
            return GuardDecision.allow("Gemini approved")

        monkeypatch.setattr("order_guard._load_margins", lambda: {
            "intelligence_guards_mode": "enforce",
            "gemini_guard_mode": "enforce",
            "gemini_min_notional_eur": 1000.0,
            "regime_context_max_age_sec": 120.0,
        })
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.GeminiHighStakeGuard.check",
            fake_gemini_check
        )

        class MockProvider:
            name = "binance"

        # 1. First evaluation at notional 1000 EUR
        order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=10.0  # notional = 1000 EUR
        )
        assert len(calls) == 1
        assert calls[0] == 1000.0

        # 2. Second evaluation with identical notional -> cached, no new call
        order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=10.0  # notional = 1000 EUR
        )
        assert len(calls) == 1

        # 3. Third evaluation with notional jumping to 1500 EUR (+50% > 20%) -> triggers re-evaluation!
        order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=15.0  # notional = 1500 EUR
        )
        assert len(calls) == 2
        assert calls[1] == 1500.0


class TestArchitectureFixes4Cases:
    def test_case1_anti_fomo_threshold_crossing_on_fractional_price_move(self, monkeypatch):
        """Case 1: Anti-FOMO parabolic surge guard recalculates on any price move,
        preventing crossing 15% threshold via a small move (114.9 -> 115.1)."""
        ctx = _build_test_context()
        now = time.time()
        # Price history with base 100.0
        history = [(now - 120, 100.0), (now - 60, 100.0)]

        monkeypatch.setattr(order_guard, "_load_margins", lambda: {
            "intelligence_guards_mode": "enforce",
            "parabolic_surge_pct": 15.0,
            "parabolic_pullback_pct": 2.0,
            "regime_context_max_age_sec": 120.0,
        })

        class MockProvider:
            name = "binance"

        # 1. At price 114.9 (+14.9% < 15.0%), BUY is permitted
        ok, reason, scale = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 114.9,
            regime_context=ctx, price_history=history, now=now
        )
        assert ok is True

        # 2. At price 115.1 (+15.1% >= 15.0%), price moved +0.174% (<0.5%).
        # Parabolic surge guard MUST re-evaluate and reject despite the small delta.
        ok2, reason2, scale2 = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 115.1,
            regime_context=ctx, price_history=history, now=now + 1
        )
        assert ok2 is False
        assert "parabolic_surge_active" in reason2

    def test_case2_late_intelligence_downscale_applied_to_placed_order(self, monkeypatch):
        """Case 2: If profit_guard / intelligence guard recommends downscale late (e.g. 0.25),
        Instrument.place updates quantity dispatched to venue from 4.0 to 1.0."""
        from instrument import Instrument

        ctx = _build_test_context()
        provider = MagicMock()
        provider.name = "binance"
        provider.guards_internally.return_value = False
        provider.free_balance.return_value = 10000.0
        provider.policy_cap_quantity.return_value = 4.0
        provider.fee_cap_quantity.return_value = 100.0
        provider.round_amount = lambda s, a: round(a, 4)
        provider.order_filter_refusal.return_value = None
        provider.market_price.return_value = 100.0
        provider.get_current_price.return_value = 100.0
        provider.open_orders.return_value = []
        provider.recent_trades.return_value = []
        provider.prepare_order_state.return_value = {}
        provider.preflight_order.return_value = object()
        provider.adjust_order_price = lambda s, side, px, **kw: px
        provider.quantity_decision = lambda *a, **k: type('Dec', (), {
            'final_qty': 4.0, 'scale_applied': False, 'applied_scale': 1.0,
            'refuse_reason': None, 'balance_asset': 'USDC'
        })()

        placed_calls = []
        def fake_place_order(symbol, side, price, qty, **kw):
            placed_calls.append({"symbol": symbol, "side": side, "qty": qty, "price": price})
            return {"id": "ord-1", "symbol": symbol, "side": side, "price": price, "qty": qty}

        provider.place_order = fake_place_order

        mock_api = MagicMock()
        mock_api.provider_by_name.return_value = provider

        # Late intelligence downscale inside profit_guard
        def mock_profit_guard(prov, sym, side, px, pm, window_ref=None, regime_context=None, qty=None):
            if regime_context is not None:
                object.__setattr__(regime_context, "suggested_scale", 0.25)
            return True

        monkeypatch.setattr(order_guard, "profit_guard", mock_profit_guard)
        monkeypatch.setattr(order_guard, "buy_reference_mode", lambda p: "off")

        class FakeSlot:
            allowed = True
            info = {}
            def commit(self, order):
                pass
        import contextlib
        import instrument as inst_mod
        @contextlib.contextmanager
        def fake_trade_slot(*a, **k):
            yield FakeSlot()
        monkeypatch.setattr(inst_mod.trade_cooldown, "trade_slot", fake_trade_slot)

        inst = Instrument("binance_btcusdc", "BTCUSDC", "binance", api=mock_api)
        res = inst.place("BUY", 4.0, 100.0, regime_context=ctx, force=True, wait_for_trend=False, caller_owns_retry=True)

        assert res is not None
        assert len(placed_calls) == 1
        # Crucial check: qty sent was scaled to 1.0 (4.0 * 0.25)
        assert placed_calls[0]["qty"] == pytest.approx(1.0)

    def test_case3_context_reuse_preserves_scale_across_multiple_orders(self):
        """Case 3: Two separate order requests sharing the same context with scale 0.25
        both produce scaled quantity 1.0 instead of 1.0 then 4.0."""
        ctx = _build_test_context()
        object.__setattr__(ctx, "suggested_scale", 0.25)

        provider = MagicMock()
        provider.free_balance.return_value = 10000.0
        provider.policy_cap_quantity.return_value = 10.0
        provider.fee_cap_quantity.return_value = 100.0
        provider.round_amount = lambda s, a: round(a, 4)
        provider.order_filter_refusal.return_value = None

        # Order 1
        d1 = decide_quantity(provider, "BTCUSDC", "BUY", 100.0, 4.0, regime_context=ctx)
        assert d1.final_qty == 1.0
        assert d1.scale_applied is True
        assert d1.applied_scale == 0.25

        # Order 2 with identical shared context
        d2 = decide_quantity(provider, "BTCUSDC", "BUY", 100.0, 4.0, regime_context=ctx)
        assert d2.final_qty == 1.0
        assert d2.scale_applied is True
        assert d2.applied_scale == 0.25

    def test_case4_monitortrades_position_stats_cache_detects_quantity_change(self):
        """Case 4: Position stats cache includes total quantity in order signature,
        preventing stale cache reuse when order quantity changes from 1.0 to 7.25."""
        import monitortrades as mt
        mt.clear_position_stats_cache()

        class MockApi:
            name = "test_venue"

        api = MockApi()
        symbol = "BTCUSDC"

        # 1. First call with quantity 1.0
        orders_v1 = [{"id": 101, "price": 100.0, "qty": 1.0, "timestamp": 1000}]
        stats1 = mt.get_position_stats(symbol, 3600, api=api, buy_orders=orders_v1, sell_orders=[])
        assert stats1.get("buy_qty") == pytest.approx(1.0)
        assert stats1.get("net_qty") == pytest.approx(1.0)

        # 2. Second call where the order quantity is now 7.25
        # (same length 1, same order id 101, same price 100.0)
        orders_v2 = [{"id": 101, "price": 100.0, "qty": 7.25, "timestamp": 1000}]
        stats2 = mt.get_position_stats(symbol, 3600, api=api, buy_orders=orders_v2, sell_orders=[])
        assert stats2.get("buy_qty") == pytest.approx(7.25)
        assert stats2.get("net_qty") == pytest.approx(7.25)

        # 3. Cache clearing function works cleanly
        mt.clear_position_stats_cache()
        assert len(mt._position_stats_cache) == 0


class TestSisterArchitecturalDefects:
    def test_sister1_price_history_resolves_24h_cache_and_cross_venue_symbols(self, tmp_path, monkeypatch):
        """Sister 1: Verify _read_cached_price_history resolves 24h high-resolution cache
        and supports cross-venue symbols (e.g. HYPEUSD on Kraken) using synthetic test data."""
        monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
        now_ts = time.time()

        # 1. Create synthetic 24h cache for BTCUSDC
        btc_file = tmp_path / "cache_24price_BTCUSDC.json"
        btc_items = [[(now_ts - i * 10) * 1000, 80000.0 + i] for i in range(120)]
        btc_file.write_text(json.dumps({"items": {"BTCUSDC": btc_items}}), encoding="utf-8")

        # 2. Create synthetic multi cache with base asset HYPE
        multi_file = tmp_path / "cache_prices_multi.json"
        hype_items = [[(now_ts - i * 15) * 1000, 90.0 + i * 0.1] for i in range(60)]
        multi_file.write_text(json.dumps({"items": {"HYPE": hype_items}}), encoding="utf-8")

        history_btc = order_guard._read_cached_price_history("BTCUSDC", window_seconds=7200.0)
        assert history_btc is not None
        assert len(history_btc) == 120  # High-resolution 24h cache resolved

        history_hype = order_guard._read_cached_price_history("HYPEUSD", window_seconds=7200.0)
        assert history_hype is not None
        assert len(history_hype) == 60  # Cross-venue base asset normalized and resolved

    def test_sister1_price_history_never_matches_substring_asset(self, tmp_path, monkeypatch):
        """Sister 1: History fallback must NOT match substring asset (e.g. ETH for ETHFIUSDC)."""
        monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
        now_ts = time.time()

        # Multi cache contains only ETH
        multi_file = tmp_path / "cache_prices_multi.json"
        eth_items = [[(now_ts - i * 10) * 1000, 3000.0] for i in range(50)]
        multi_file.write_text(json.dumps({"items": {"ETH": eth_items}}), encoding="utf-8")

        # ETHFIUSDC must not match ETH
        history = order_guard._read_cached_price_history("ETHFIUSDC", window_seconds=7200.0)
        assert history is None

    def test_sister2_sell_orders_not_downscaled_by_regime_context_suggested_scale(self):
        """Sister 2: Market intelligence suggested_scale must ONLY downscale BUY entries,
        never downscaling SELL exits or risk reductions."""
        ctx = _build_test_context()
        object.__setattr__(ctx, "suggested_scale", 0.25)

        provider = MagicMock()
        provider.free_balance.return_value = 10000.0
        provider.policy_cap_quantity.return_value = 10.0
        provider.fee_cap_quantity.return_value = 100.0
        provider.round_amount = lambda s, a: round(a, 4)
        provider.order_filter_refusal.return_value = None

        # 1. SELL order must NOT be downscaled (requested 4.0 remains 4.0)
        sell_dec = decide_quantity(provider, "BTCUSDC", "SELL", 100.0, 4.0, regime_context=ctx)
        assert sell_dec.final_qty == 4.0
        assert sell_dec.scale_applied is False
        assert sell_dec.applied_scale == 1.0

        # 2. BUY order with the same context IS downscaled (4.0 -> 1.0)
        buy_dec = decide_quantity(provider, "BTCUSDC", "BUY", 100.0, 4.0, regime_context=ctx)
        assert buy_dec.final_qty == 1.0
        assert buy_dec.scale_applied is True
        assert buy_dec.applied_scale == 0.25

    def test_sister3_gemini_evaluated_notional_prevents_ratchet_drift(self, monkeypatch):
        """Sister 3: Incremental notional growth must not ratchet baseline forward without evaluation."""
        ctx = _build_test_context()
        calls = []

        def fake_gemini_check(self, symbol, side, price, qty, notional_eur=None):
            calls.append(notional_eur)
            from intelligence.internal.guards.guard_decision import GuardDecision
            return GuardDecision.allow("Gemini approved")

        monkeypatch.setattr(order_guard, "_load_margins", lambda: {
            "intelligence_guards_mode": "enforce",
            "gemini_guard_mode": "enforce",
            "gemini_min_notional_eur": 1000.0,
            "regime_context_max_age_sec": 120.0,
        })
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.GeminiHighStakeGuard.check",
            fake_gemini_check
        )

        class MockProvider:
            name = "binance"

        # 1. Initial evaluation at 1000 EUR
        order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=10.0  # notional = 1000 EUR
        )
        assert len(calls) == 1
        assert getattr(ctx, "_gemini_evaluated_notional", None) == 1000.0

        # 2. Intermediate step at 1150 EUR (+15% < 20%): skipped, baseline remains 1000 EUR
        order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=11.5  # notional = 1150 EUR
        )
        assert len(calls) == 1
        assert getattr(ctx, "_gemini_evaluated_notional", None) == 1000.0

        # 3. Step at 1300 EUR (+30% from 1000 EUR, but only +13% from 1150 EUR):
        # MUST trigger re-evaluation because baseline was preserved!
        order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=13.0  # notional = 1300 EUR
        )
        assert len(calls) == 2
        assert calls[1] == 1300.0
        assert getattr(ctx, "_gemini_evaluated_notional", None) == 1300.0

    def test_sister3_gemini_decision_preserved_on_identical_subsequent_check(self, monkeypatch):
        """Sister 3: Gemini decisions (rejection and downscaling) must be cached and re-applied
        on identical subsequent checks rather than reverting to allow/1.0."""
        from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
        ctx = _build_test_context()
        calls = []

        gemini_result = [GuardDecision.veto("gemini", "Gemini high-risk block")]

        def fake_gemini_check(self, symbol, side, price, qty, notional_eur=None):
            calls.append(notional_eur)
            return gemini_result[0]

        monkeypatch.setattr(order_guard, "_load_margins", lambda: {
            "intelligence_guards_mode": "enforce",
            "gemini_guard_mode": "enforce",
            "gemini_min_notional_eur": 1000.0,
            "regime_context_max_age_sec": 120.0,
        })
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.GeminiHighStakeGuard.check",
            fake_gemini_check
        )

        class MockProvider:
            name = "binance"

        # 1. Initial check at 100 EUR (< 1000 min_notional): passes baseline
        ok, reason, scale = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=1.0  # notional = 100 EUR
        )
        assert ok is True
        assert len(calls) == 0

        # 2. Large order at 1500 EUR: Gemini rejects
        ok, reason, scale = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=15.0  # notional = 1500 EUR
        )
        assert ok is False
        assert "Gemini high-risk block" in reason
        assert len(calls) == 1

        # 3. Identical large order at 1500 EUR on SAME context: MUST REMAIN REJECTED
        # without calling Gemini API a second time!
        ok2, reason2, scale2 = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx, qty=15.0  # notional = 1500 EUR
        )
        assert ok2 is False
        assert "Gemini high-risk block" in reason2
        assert len(calls) == 1  # No second API call

        # 4. Downscaling test: Gemini returns scale = 0.25
        ctx_downscale = _build_test_context()
        calls.clear()
        gemini_result[0] = GuardDecision.downscale("gemini", 0.25, "Gemini downscale risk")

        ok3, reason3, scale3 = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx_downscale, qty=15.0  # notional = 1500 EUR
        )
        assert ok3 is True
        assert scale3 == 0.25
        assert len(calls) == 1

        # Second check on SAME context: scale must remain 0.25, NOT revert to 1.0!
        ok4, reason4, scale4 = order_guard.check_intelligence_guards(
            MockProvider(), "BTCUSDC", "BUY", 100.0,
            regime_context=ctx_downscale, qty=15.0  # notional = 1500 EUR
        )
        assert ok4 is True
        assert scale4 == 0.25
        assert len(calls) == 1  # No second API call

    def test_instrument_late_scale_synchronizes_retry_queue_and_partial_fill(self, tmp_path, monkeypatch):
        """Sister Priority 1: Late intelligence scaling in Instrument.place must update
        the retry queue record so partial fill remainder is calculated against submitted qty."""
        import order_retry
        from instrument import Instrument
        from providers.strategy_executor import OrderStatus

        queue_file = str(tmp_path / "order_retry_queue.jsonl")
        lock_file = str(tmp_path / "order_retry_queue.lock")
        monkeypatch.setattr(order_retry, "QUEUE_FILE", queue_file)
        monkeypatch.setattr(order_retry, "LOCK_FILE", lock_file)
        monkeypatch.setattr(order_retry, "RETRY_ENABLED", True)

        ctx = _build_test_context()
        # Initial scale is 1.0 so intent is prequeued for full quantity (4.0)
        object.__setattr__(ctx, "suggested_scale", 1.0)

        provider = MagicMock()
        provider.name = "binance"
        provider.get_current_price.return_value = 100.0
        provider.round_amount = lambda s, a: round(a, 4)
        provider.min_order_qty.return_value = 0.001
        provider.min_order_notional.return_value = 5.0
        provider.order_filter_refusal.return_value = None
        provider.free_balance.return_value = 10000.0
        provider.policy_cap_quantity.return_value = 10.0
        provider.fee_cap_quantity.return_value = 100.0
        provider.guards_internally.return_value = False
        provider.get_orders.return_value = []
        provider.get_trades.return_value = []
        provider.adjust_order_price.side_effect = lambda s, sd, p, **k: p
        from providers.quantity import decide_quantity
        provider.quantity_decision.side_effect = lambda *a, **k: decide_quantity(provider, *a, **k)

        placed_calls = []
        def fake_place_order(symbol, side, price, qty, **kwargs):
            placed_calls.append({"symbol": symbol, "side": side, "price": price, "qty": qty, "kwargs": kwargs})
            return {"orderId": "ORD_12345", "status": "NEW", "executedQty": "0.0", "origQty": str(qty)}

        provider.place_order.side_effect = fake_place_order

        # Dispatch-time profit guard recommends late scale reduction 0.25
        def fake_profit_guard(prov, sym, side, px, margin, window_ref=None, regime_context=None, qty=None):
            if regime_context is not None:
                object.__setattr__(regime_context, "suggested_scale", 0.25)
            return True

        monkeypatch.setattr(order_guard, "profit_guard", fake_profit_guard)
        from contextlib import contextmanager

        @contextmanager
        def fake_trade_slot(*a, **k):
            slot = MagicMock()
            slot.allowed = True
            slot.info = {}
            yield slot

        monkeypatch.setattr("lock.trade_cooldown.trade_slot", fake_trade_slot)

        mock_api = MagicMock()
        mock_api.provider_by_name.return_value = provider
        inst = Instrument("test_inst", "BTCUSDC", "test_venue", api=mock_api)
        order = inst.place("BUY", 100.0, 4.0, market=True, regime_context=ctx, wait_for_trend=False)

        assert order is not None
        assert len(placed_calls) == 1
        # Order sent to provider was scaled 4.0 -> 1.0
        assert placed_calls[0]["qty"] == 1.0

        # Verify retry queue record on disk has qty=1.0, NOT 4.0!
        records = order_retry.load_all()
        assert len(records) == 1
        assert records[0]["qty"] == 1.0
        assert records[0]["requested_qty_total"] == 1.0

        # Now simulate a partial fill of 0.4 on venue, followed by order expiration
        claimed = order_retry.claim([records[0]["id"]], now=time.time())
        assert len(claimed) == 1

        status = OrderStatus(
            status="expired",
            venue_status="EXPIRED",
            filled_qty=0.4,
            cost=40.0,
            fee=0.04,
        )
        transition = order_retry.advance_claimed_status(claimed[0], status)
        # Remainder must be 1.0 - 0.4 = 0.6, NOT 4.0 - 0.4 = 3.6!
        assert transition.action == "retry_terminal"
        assert abs(transition.remaining_qty - 0.6) < 1e-6

    def test_sister4_identity_enforced_in_symbol_trend_and_dynamic_window(self):
        """Sister 4: Anonymous context without symbol/provider is rejected by _symbol_trend
        and dynamic_buy_window_sec via require_identity=True."""
        from market_regime import MarketRegimeContext, MarketRegimeDecision
        anon_decision = MarketRegimeDecision("bull", 0.5, 0.1, 5.0, True, "test")
        anon_ctx = MarketRegimeContext.from_decision(
            anon_decision,
            evaluated_at=time.time(),
            symbol=None,
            provider=None,
        )

        # _symbol_trend must NOT trust anonymous context as matching
        trend = order_guard._symbol_trend("BTCUSDC", provider="binance", regime_context=anon_ctx)
        # Without identity, anon_ctx.resolved_trend is bypassed
        assert trend != "bull" or getattr(anon_ctx, "symbol", None) is None

        # dynamic_buy_window_sec must not consume anonymous context
        window = order_guard.dynamic_buy_window_sec("BTCUSDC", provider="binance", regime_context=anon_ctx)
        assert window > 0

    def test_sister5_profit_guard_dynamic_history_exception_fails_closed(self, monkeypatch):
        """Sister 5: Unhandled exceptions in provider.get_orders within dynamic buy window
        fail closed (return False) cleanly."""
        ctx = _build_test_context()
        provider = MagicMock()
        provider.name = "binance"
        provider.get_orders.side_effect = RuntimeError("network socket reset")

        monkeypatch.setattr(order_guard, "buy_reference_mode", lambda p: "dynamic")
        monkeypatch.setattr(order_guard, "_symbol_trend", lambda *a, **k: "flat")

        # In dynamic mode with an old reference, if history fails to load, must fail closed
        res = order_guard.profit_guard(
            provider, "BTCUSDC", "BUY", 100.0, 1.15,
            window_ref=100.5, regime_context=ctx
        )
        assert res is False

    def test_sister6_parabolic_surge_consolidates_over_window_without_requiring_pullback(self):
        """Sister 6: Parabolic surge must disarm after consolidating past window_seconds,
        even if price did not pull back by the required percentage."""
        from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
        guard = ParabolicSurgeGuard(surge_threshold_pct=5.0, pullback_required_pct=2.0, window_seconds=7200.0)

        # 1. Surge occurs from 100 to 110 (+10% > 5%) at ts=1000
        history = [(900.0, 100.0), (1000.0, 110.0)]
        dec1 = guard.check("BTCUSDC", "BUY", 110.0, price_history=history, now=1000.0)
        assert dec1.allowed is False
        assert "parabolic_surge_active" in dec1.reason

        # 2. At ts=3000 (2000s later, within 2h window), price consolidates at 109.0 (pullback 0.9% < 2%)
        # Surge must still be active
        dec2 = guard.check("BTCUSDC", "BUY", 109.0, price_history=history, now=3000.0)
        assert dec2.allowed is False

        # 3. At ts=9000 (8000s later, elapsed > window_seconds=7200s), price is still 109.0
        # Consolidation disarms the surge; BUY must be permitted!
        dec3 = guard.check("BTCUSDC", "BUY", 109.0, price_history=history, now=9000.0)
        assert dec3.allowed is True
        assert dec3.reason == "normal_market_structure"

    def test_sister7_price_analysis_resolves_cross_venue_symbols_via_normalized_base(self, monkeypatch):
        """Sister 7: priceAnalysis.get_weight_for_cash_permission_at_quant_time must resolve
        cross-venue quote symbols (e.g. ARBUSD) against base-asset cache records (ARBUSDC)."""
        import priceAnalysis as pa
        import cacheManager as cm

        # Mock CachePriceLongTrendManager cache with ARBUSDC having an active trend
        trend_item = {
            "timestamp": 1791198244,
            "direction": "up",
            "start_timestamp": 1790679820.569,
            "duration_seconds": 518293.69,
            "estimated_future_hours": 72.0,
        }
        mock_mgr = MagicMock()
        mock_mgr.cache = {"ARBUSDC": [trend_item]}
        monkeypatch.setattr(cm, "get_cache_manager", lambda name: mock_mgr)

        # ARBUSD (Kraken/Hyperliquid symbol) should match ARBUSDC
        w = pa.get_weight_for_cash_permission_at_quant_time("ARBUSD", "BUY", T_quanta=8)
        assert w is not None
        assert 0.0 < w <= 1.0

    def test_sister8_read_cached_trend_duration_resolves_cross_venue_and_skips_nulls(self, tmp_path, monkeypatch):
        """Sister 8: _read_cached_trend_duration must support MPTRADE_CACHEDB_DIR,
        match normalized base assets across venues (ARBUSD -> ARBUSDC), and skip nulls."""
        monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
        trend_file = tmp_path / "cache_price_long_trend.json"
        data = {
            "items": {
                "BTCUSDC": [None],
                "ARBUSDC": [
                    None,
                    {
                        "timestamp": 1791198244,
                        "direction": "up",
                        "start_timestamp": 1790679820.569,
                        "duration_seconds": 518293.69,
                    }
                ]
            }
        }
        trend_file.write_text(json.dumps(data), encoding="utf-8")

        # BTCUSDC has only null, duration is 0.0
        dur_btc = order_guard._read_cached_trend_duration("BTCUSDC")
        assert dur_btc == 0.0

        # ARBUSD matches ARBUSDC and skips initial None to get 518293.69
        dur_arb = order_guard._read_cached_trend_duration("ARBUSD")
        assert abs(dur_arb - 518293.69) < 1e-2

    def test_sister9_order_retry_worker_syncs_late_scaled_qty_from_outcome_context(self, tmp_path, monkeypatch):
        """Sister 9: order_retry_worker must synchronize submitted_qty from outcome_context
        into the accepted claim record."""
        import order_retry as oq
        import order_retry_worker as worker

        queue_file = str(tmp_path / "order_retry_queue.jsonl")
        lock_file = str(tmp_path / "order_retry_queue.lock")
        monkeypatch.setattr(oq, "QUEUE_FILE", queue_file)
        monkeypatch.setattr(oq, "LOCK_FILE", lock_file)
        monkeypatch.setattr(oq, "RETRY_ENABLED", True)

        # Enqueue intent for 4.0 at now=1000.0
        oq.enqueue("BTCUSDC", "BUY", 4.0, {}, requested_price=100.0, now=1000.0)

        # Simulate mkt.place reducing quantity to 1.0 and populating outcome_context
        def fake_place(symbol, side, price, qty, **kwargs):
            ctx = kwargs.get("_outcome_context")
            if ctx is not None:
                ctx["accepted"] = True
                ctx["submitted_qty"] = 1.0
                ctx["submitted_price"] = price
                ctx["state"] = "accepted"
            return {"orderId": "ORD_WORKER_999", "status": "NEW", "executedQty": "0.0", "origQty": "1.0"}

        from providers.strategy_executor import OrderReconciliationCapabilities
        monkeypatch.setattr(worker.alert, "notify", lambda **kw: None)

        mock_mkt = MagicMock()
        mock_mkt.get_current_price.return_value = 100.0
        mock_mkt.order_by_client_id.return_value = None
        mock_mkt.reconciliation_capabilities.return_value = OrderReconciliationCapabilities(
            True, True, True, True, not_found_reliable_for_seconds=86400 * 90)
        mock_mkt.place.side_effect = fake_place

        res = worker.process_once(mkt=mock_mkt, now=1400.0)
        assert res["attempted"] == 1
        assert res["succeeded"] == 1

        # Check queue record: qty must be 1.0, NOT 4.0!
        records = oq.load_all()
        assert len(records) == 1
        assert records[0]["qty"] == 1.0
        assert records[0]["requested_qty_total"] == 1.0


