"""Tests verifying dynamic intelligence re-evaluation and single quantity downscaling.

Tests cover:
1. Dynamic anti-FOMO parabolic surge guard re-evaluating when price changes within the same MarketRegimeContext.
2. Dynamic Gemini high-stake guard re-evaluating when notional jumps above threshold within the same MarketRegimeContext.
3. Propagation and single application of suggested_scale in decide_quantity and Instrument.place.
4. Hot-reload of order_guard.conf on mtime change in _load_margins().
5. TTL fallback timer in priceAnalysis.py cache.
"""
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

        monkeypatch.setattr("intelligence.sentiment.guards.gemini_high_stake_guard.GeminiHighStakeGuard", MockGeminiGuard)

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
        """Verify that suggested_scale on regime_context scales quantity and is marked applied."""
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
        assert getattr(ctx, "_scale_applied", False) is True

        # Calling decide_quantity again on the same context must NOT re-scale 2.5 to 0.625
        provider.policy_cap_quantity.return_value = 2.5
        decision2 = decide_quantity(
            provider, "BTCUSDC", "BUY", 100.0, 2.5, regime_context=ctx
        )
        assert decision2.final_qty == 2.5

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
            "intelligence.sentiment.guards.gemini_high_stake_guard.GeminiHighStakeGuard.check",
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

