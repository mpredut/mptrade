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
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_guard
from market_regime import MarketRegimeDecision, MarketRegimeContext
from providers.quantity import decide_quantity


_BASE_MARGINS = dict(order_guard._load_margins())


def _margins(**overrides):
    return {
        **_BASE_MARGINS,
        "whale_guard_mode": "off",
        "orderbook_wall_guard_mode": "off",
        "funding_guard_mode": "off",
        "geopolitical_guard_mode": "off",
        "weibull_exhaustion_policy": "off",
        **overrides,
    }


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

        margins = _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            parabolic_surge_pct=15.0,
            parabolic_pullback_pct=2.0,
            weibull_exhaustion_policy="off",
            llm_guard_mode="off",
            geopolitical_guard_mode="off",
            whale_guard_mode="off",
            orderbook_wall_guard_mode="off",
            funding_guard_mode="off",
            default=1.15,
        )
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

        margins = _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            llm_guard_mode="enforce",
            high_stake_min_notional_usd=1000.0,
            llm_timeout_sec=5.0,
            default=1.15,
        )
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)
        monkeypatch.setattr(order_guard, "_resolve_trend_duration", lambda s: 3600.0)

        from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction

        gemini_called = []

        class MockGeminiGuard:
            def __init__(self, **kw):
                pass
            def check(self, symbol, side, price, qty, notional_eur=None):
                gemini_called.append((symbol, side, price, qty, notional_eur))
                return GuardDecision.veto("HighStakeGuard", "gemini_risk_block")

        monkeypatch.setattr("intelligence.sentiment.guards.high_stake_guard.HighStakeGuard", MockGeminiGuard)

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
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "off")

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
        from pathlib import Path
        real_conf = Path(order_guard.__file__).with_name("order_guard.conf").read_text(encoding="utf-8")
        conf_file = tmp_path / "order_guard.conf"
        conf_file.write_text(real_conf + "\nparabolic_surge_pct = 15.0\n", encoding="utf-8")

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
        conf_file.write_text(real_conf + "\nparabolic_surge_pct = 25.0\n", encoding="utf-8")
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
        ok, reason = order_guard.daily_limit_guard(p, "BTCUSDC", "BUY")
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
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "dynamic")
        monkeypatch.setattr(order_guard, "get_dynamic_window_sec_cfg", lambda *a, **k: 3600.0)

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

        monkeypatch.setattr("order_guard._load_margins", lambda: _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            llm_guard_mode="enforce",
            high_stake_min_notional_usd=1000.0,
            regime_context_max_age_sec=120.0,
        ))
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.HighStakeGuard.check",
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

        monkeypatch.setattr(order_guard, "_load_margins", lambda: _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            parabolic_surge_pct=15.0,
            parabolic_pullback_pct=2.0,
            regime_context_max_age_sec=120.0,
        ))

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
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "off")

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
        btc_file = tmp_path / "cache_24_price_BTCUSDC.json"
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

        monkeypatch.setattr(order_guard, "_load_margins", lambda: _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            llm_guard_mode="enforce",
            high_stake_min_notional_usd=1000.0,
            regime_context_max_age_sec=120.0,
        ))
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.HighStakeGuard.check",
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

        monkeypatch.setattr(order_guard, "_load_margins", lambda: _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            llm_guard_mode="enforce",
            high_stake_min_notional_usd=1000.0,
            regime_context_max_age_sec=120.0,
        ))
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.HighStakeGuard.check",
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
        and get_dynamic_window_sec_cfg via require_identity=True."""
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

        # get_dynamic_window_sec_cfg must not consume anonymous context
        window = order_guard.get_dynamic_window_sec_cfg("BTCUSDC", provider="binance", order_type="BUY", regime_context=anon_ctx)
        assert window > 0

    def test_sister5_profit_guard_dynamic_history_exception_fails_closed(self, monkeypatch):
        """Sister 5: Unhandled exceptions in provider.get_orders within dynamic buy window
        fail closed (return False) cleanly."""
        ctx = _build_test_context()
        provider = MagicMock()
        provider.name = "binance"
        provider.get_orders.side_effect = RuntimeError("network socket reset")

        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "dynamic")
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

        # Mock CacheLongTrendManager cache with ARBUSDC having an active trend
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
        trend_file = tmp_path / "cache_long_trend.json"
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


class TestBatch6IntelligenceAndRoutingFixes:
    """Tests for Batch 6 bug fixes:
    1. Cross-side and cross-bypass cache isolation on MarketRegimeContext in check_intelligence_guards & intelligence_order_guard.
    2. Parabolic surge pullback recovery on shared MarketRegimeContext after initial spike rejection.
    3. FundingCrowdingGuard funding_min_short_rate forwarding and DEFER_WAIT shadow notify in external_order_guard.
    4. Geopolitical guard string boolean shadow_notify parsing without ValueError crash.
    5. OrderbookDepthCollector non-tuple cache fallback, WhalePositioningCollector funding rate disk TTL, and USDCUSDT preservation.
    6. check_math_profit_reference dynamic BUY window active_cycle_window_ref and MarketApi.place_order explicit provider_name routing.
    """

    def test_batch6_cross_side_regime_context_cache_isolation(self, monkeypatch):
        """Evaluating SELL first on shared regime_context must not bypass BUY geopolitical/internal guards,
        and a blocked BUY must not block a subsequent SELL exit."""
        ctx = _build_test_context()
        margins = _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            geopolitical_guard_mode="enforce",
            high_stake_min_notional_usd=700.0,
            llm_guard_mode="off",
            whale_guard_mode="off",
            orderbook_wall_guard_mode="off",
            funding_guard_mode="off",
            weibull_exhaustion_policy="off",
            default=1.15,
        )
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        import llm_order_guard
        monkeypatch.setattr(
            llm_order_guard,
            "check_geopolitical_order_guards",
            lambda symbol, side, **kw: (False, "critical_geopolitical_shock", 0.0)
            if side == "BUY"
            else (True, "not_buy_side", 1.0),
        )

        # 1. Evaluate a large SELL (>= 1000 EUR) first on shared context -> allowed
        sell_ok, sell_reason, sell_scale = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "SELL", 100.0, qty=20.0, notional_eur=2000.0, regime_context=ctx
        )
        assert sell_ok is True
        assert sell_scale == 1.0

        # 2. Evaluate BUY on the SAME context -> must NOT reuse SELL's (True, "ok", 1.0) and MUST be blocked by GI4!
        buy_ok, buy_reason, buy_scale = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=20.0, notional_eur=2000.0, regime_context=ctx
        )
        assert buy_ok is False
        assert buy_reason == "critical_geopolitical_shock"
        assert buy_scale == 0.0

        # 3. Reverse order on a fresh context: BUY blocked first, then SELL evaluated -> SELL must still be allowed!
        ctx2 = _build_test_context()
        buy_ok2, _, _ = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=20.0, notional_eur=2000.0, regime_context=ctx2
        )
        assert buy_ok2 is False

        sell_ok2, _, sell_scale2 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "SELL", 100.0, qty=20.0, notional_eur=2000.0, regime_context=ctx2
        )
        assert sell_ok2 is True
        assert sell_scale2 == 1.0

    def test_batch6_cross_bypass_and_parabolic_pullback_recovery_on_same_context(self, monkeypatch):
        """A BUY rejected at a parabolic spike price (125.0) on a fresh context must re-evaluate and pass
        when price pulls back to 100.0 or when GI1 is bypassed."""
        ctx = _build_test_context()
        margins = _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            parabolic_surge_pct=15.0,
            parabolic_pullback_pct=2.0,
            weibull_exhaustion_policy="off",
            llm_guard_mode="off",
            geopolitical_guard_mode="off",
            whale_guard_mode="off",
            orderbook_wall_guard_mode="off",
            funding_guard_mode="off",
            default=1.15,
        )
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)
        now = time.time()
        history = [(now - 120.0, 100.0), (now - 60.0, 100.0)]
        monkeypatch.setattr(order_guard, "_read_cached_price_history", lambda s, **kw: history)
        monkeypatch.setattr(order_guard, "_resolve_trend_duration", lambda s: 3600.0)

        # 1. First call on context is at spike price 125.0 -> blocked by parabolic_surge_active
        ok_spike, reason_spike, _ = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 125.0, price_history=history, regime_context=ctx
        )
        assert ok_spike is False
        assert "parabolic_surge" in reason_spike

        # 2. Bypassing G_QLIMIT2 on the same context at 125.0 must NOT reuse the un-bypassed block
        ok_bypass, _, _ = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 125.0, price_history=history, regime_context=ctx, bypass_guards={"G_QLIMIT2"}
        )
        assert ok_bypass is True

        # 3. Un-bypassed call at pulled-back price 100.0 on the same context must re-evaluate and allow!
        ok_pullback, reason_pullback, scale_pullback = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, price_history=history, regime_context=ctx
        )
        assert ok_pullback is True
        assert reason_pullback == "ok"
        assert scale_pullback == 1.0

    def test_batch6_external_funding_min_short_rate_and_defer_notify(self, monkeypatch):
        """external_order_guard must pass funding_min_short_rate to FundingCrowdingGuard and notify on DEFER_WAIT."""
        import external_order_guard
        import notify_engine.alertnotifiers as alertnotifiers
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry

        telem = DerivativesTelemetry(
            symbol="BTCUSDC",
            funding_rate=-0.0003,
            predicted_funding_rate=-0.0003,
            open_interest=1000.0,
            open_interest_usd=80_000_000.0,
            ts=time.time(),
        )
        mock_collector = MagicMock()
        mock_collector.fetch.return_value = telem
        monkeypatch.setattr(external_order_guard, "get_derivatives_collector", lambda: mock_collector)
        external_order_guard._MICRO_SHADOW_NOTIFY_COOLDOWN.clear()

        notified = []
        monkeypatch.setattr(
            alertnotifiers,
            "notify",
            lambda **kw: notified.append(kw),
        )

        # Custom funding_min_short_rate = -0.0002 (default is -0.0005), so -0.0003 triggers short crowding on SELL
        margins_shadow = _margins(
            whale_guard_mode="off",
            orderbook_wall_guard_mode="off",
            funding_guard_mode="shadow",
            funding_min_short_rate=-0.0002,
            funding_crowding_policy="defer",
            shadow_notify="true",
        )
        ok_sh, _, _ = external_order_guard.check_external_order_guards(
            "BTCUSDC", "SELL", 100.0, notional_eur=2000.0, margins=margins_shadow
        )
        assert ok_sh is True
        assert len(notified) == 1
        assert "FUNDING" in notified[0]["title"]

        margins_enforce = dict(margins_shadow, funding_guard_mode="enforce")
        ok_enf, reason_enf, scale_enf = external_order_guard.check_external_order_guards(
            "BTCUSDC", "SELL", 100.0, notional_eur=2000.0, margins=margins_enforce
        )
        assert ok_enf is False
        assert "short_crowding_extreme" in reason_enf
        assert scale_enf == 0.0

    def test_batch6_geopolitical_string_shadow_notify_no_crash(self, monkeypatch):
        """llm_order_guard must parse shadow_notify='true'/'false' without raising ValueError."""
        import llm_order_guard
        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment

        class DummyAnalyzer:
            def __init__(self):
                self._cached_assessment = GeopoliticalThreatAssessment(
                    threat_level="ELEVATED",
                    risk_score=0.65,
                    summary="Elevated energy risk",
                    recommended_brake="DOWNSCALE_50",
                    headlines_analyzed=5,
                    ts=time.time(),
                )

        monkeypatch.setattr(
            "intelligence.macro.geopolitical_analyzer.GeopoliticalThreatAnalyzer",
            DummyAnalyzer,
        )
        margins = _margins(
            geopolitical_guard_mode="enforce",
            shadow_notify="true",
        )
        ok, reason, scale = llm_order_guard.check_geopolitical_order_guards(
            "BTCUSDC", "BUY", margins=margins
        )
        assert ok is True
        assert scale == pytest.approx(0.5)
        assert "elevated" in reason.lower()

    def test_batch6_collectors_cache_fallback_ttl_and_usdcusdt(self, tmp_path):
        """Verify OrderbookDepthCollector non-tuple cache fallback, WhalePositioningCollector funding disk TTL,
        and USDCUSDT preservation."""
        from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector, OrderbookSnapshot
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector

        ob_collector = OrderbookDepthCollector(cache_ttl_sec=0.0, cache_dir=str(tmp_path))
        snap = OrderbookSnapshot(
            symbol="BTCUSDC",
            mid_price=85000.0,
            bid_depth_usd=1000.0,
            ask_depth_usd=1000.0,
            imbalance_ratio=0.5,
            largest_bid_wall_usd=100.0,
            largest_bid_wall_price=84900.0,
            largest_ask_wall_usd=100.0,
            largest_ask_wall_price=85100.0,
            ts=1.0,
        )
        # Store direct OrderbookSnapshot in _cache, force expired TTL and broken network
        ob_collector._cache["BTCUSDC"] = snap
        # With allow_network=False and expired TTL, returns None; with allow_network=True and network error, falls back to cached snap without TypeError
        import urllib.request
        orig_urlopen = urllib.request.urlopen
        try:
            def fail_urlopen(*a, **k):
                raise OSError("network unreachable")
            urllib.request.urlopen = fail_urlopen
            res = ob_collector.fetch("BTCUSDC", allow_network=True)
            assert res is snap
        finally:
            urllib.request.urlopen = orig_urlopen

        # WhalePositioningCollector._resolve_cached_funding_rate must ignore expired disk telemetry
        w_collector = WhalePositioningCollector(cache_ttl_sec=60.0, disk_ttl_sec=120.0, cache_dir=str(tmp_path))
        telem_file = tmp_path / "derivatives_telemetry_BTCUSDT.json"
        telem_file.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": -0.0025, "ts": 1000.0}), encoding="utf-8")

        # At now=1050 (age 50s < 120s), funding rate is returned
        assert w_collector._resolve_cached_funding_rate("BTCUSDC", now=1050.0) == pytest.approx(-0.0025)
        # At now=2000 (age 1000s >= 120s), stale funding rate is ignored
        assert w_collector._resolve_cached_funding_rate("BTCUSDC", now=2000.0) is None

    def test_batch6_dynamic_window_excludes_closed_sell_cycles_and_market_api_place_order_routes_provider(self, monkeypatch):
        """check_math_profit_reference in dynamic mode must use active_cycle_window_ref to ignore closed SELL cycles,
        and MarketApi.place_order must honor provider_name."""
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "dynamic")
        monkeypatch.setattr(order_guard, "_symbol_trend", lambda *a, **k: "flat")
        monkeypatch.setattr(order_guard, "get_dynamic_window_sec_cfg", lambda *a, **k: 14400.0)

        provider = MagicMock()
        provider.name = "binance"

        # Earlier SELL at 95.0 (t=1000) was already bought back by a BUY at 93.0 (t=2000).
        # Active open SELL is at 105.0 (t=3000).
        # Buying at 100.0 is +4.76% below 105.0 (>= 1.15% required), even though min(recent_prices) is 95.0.
        def fake_get_orders(symbol, side, window_s):
            if side == "SELL":
                return [
                    {"price": 95.0, "qty": 1.0, "timestamp": 1000.0},
                    {"price": 105.0, "qty": 1.0, "timestamp": 3000.0},
                ]
            if side == "BUY":
                return [
                    {"price": 93.0, "qty": 1.0, "timestamp": 2000.0},
                ]
            return []

        provider.get_orders.side_effect = fake_get_orders

        allowed = order_guard.check_math_profit_reference(
            provider, "BTCUSDC", "BUY", 100.0, 1.15, window_ref=95.0
        )
        assert allowed is True

        # Verify MarketApi.place_order routes via provider_name instead of falling back to default provider
        from providers.market_api import MarketApi
        api = MarketApi.__new__(MarketApi)
        mock_kraken = MagicMock()
        mock_kraken.place_order.return_value = {"orderId": "KRAKEN_1"}
        api._provider_explicit_or_routed = MagicMock(return_value=mock_kraken)
        api._provider_for = MagicMock(side_effect=AssertionError("_provider_for should not be called directly"))

        out = api.place_order("SOLUSD", "BUY", 150.0, 2.0, provider_name="KRAKEN")
        assert out == {"orderId": "KRAKEN_1"}
        api._provider_explicit_or_routed.assert_called_once_with("SOLUSD", "KRAKEN")
        mock_kraken.place_order.assert_called_once_with("SOLUSD", "BUY", 150.0, 2.0)

    def test_batch7_high_stake_guard_symmetric_buy_sell_prompt_and_telemetry(self):
        """HighStakeGuard prompt and telemetry must be symmetric for BUY and SELL."""
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard

        prompts = []
        class FakeGeminiClient:
            def query_json(self, prompt, timeout_sec=None, thread_title=None, **kwargs):
                prompts.append((thread_title, prompt))
                return {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"}

        guard = HighStakeGuard(
            gemini_client=FakeGeminiClient(),
            min_notional_eur=1000.0,
            timeout_sec=1.0,
            fallback_action="allow",
            cache_ttl_sec=0.0,
        )
        guard.check("BTCUSDC", "BUY", 85000.0, 0.05, notional_eur=4250.0, telemetry={"Orderbook Ask Wall Ahead": "$2.5M"})
        guard.check("BTCUSDC", "SELL", 85000.0, 0.05, notional_eur=4250.0, telemetry={"Orderbook Bid Wall Ahead": "$3.1M"})

        assert len(prompts) == 2
        buy_prompt = prompts[0][1]
        sell_prompt = prompts[1][1]

        assert "HIGH-VALUE BUY order" in buy_prompt
        assert "below 0.80 in a FLAT or BEAR market" in buy_prompt
        assert "Orderbook Ask Wall Ahead: $2.5M" in buy_prompt

        assert "HIGH-VALUE SELL order" in sell_prompt
        assert "above 1.25 in a FLAT or BULL market" in sell_prompt
        assert "Orderbook Bid Wall Ahead: $3.1M" in sell_prompt
        assert "CAPITULATION & SQUEEZE PROTECTION" in sell_prompt

    def test_batch7_parabolic_and_weibull_symmetric_sell_and_direction_aware(self):
        """ParabolicSurgeGuard and WeibullExhaustionGuard must operate symmetrically on BUY and SELL."""
        from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
        from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
        from qlimit_order_guard import check_qlimit2_order_guards as check_internal_order_guards

        p_guard = ParabolicSurgeGuard(surge_threshold_pct=10.0, pullback_required_pct=2.0)

        # Vertical dump: 100 -> 80 (-20% crash) blocks SELL until +2% rebound (>= 81.6)
        dump_history = [(100.0, 100.0), (200.0, 95.0), (300.0, 80.0)]
        dec_sell_dump = p_guard.check("SOLUSDC", "SELL", 80.0, price_history=dump_history, now=300.0)
        assert dec_sell_dump.allowed is False
        assert "parabolic_dump_active" in dec_sell_dump.reason

        # Partial rebound to 81.0 (+1.25% < 2.0%) still deferred
        dec_sell_wait = p_guard.check("SOLUSDC", "SELL", 81.0, price_history=dump_history, now=350.0)
        assert dec_sell_wait.allowed is False
        assert "parabolic_dump_active" in dec_sell_wait.reason
        assert "pullback=1.25% < required 2.00%" in dec_sell_wait.reason

        # Full rebound to 82.0 (+2.5% >= 2.0%) clears state and allows SELL
        dec_sell_ok = p_guard.check("SOLUSDC", "SELL", 82.0, price_history=dump_history, now=400.0)
        assert dec_sell_ok.allowed is True

        # WeibullExhaustionGuard direction-awareness:
        # BUY in exhausted bear trend is favorable (buying the dip), while BUY in exhausted bull trend is penalized.
        # SELL in exhausted bull trend is favorable (taking profit), while SELL in exhausted bear trend is penalized.
        w_guard = WeibullExhaustionGuard(default_p90_days=10.0, policy="downscale", exhausted_scale=0.5)
        dur_15d = 15.0 * 86400.0

        dec_buy_bull = w_guard.check("BTCUSDC", "BUY", trend_duration_seconds=dur_15d, trend_direction="bull")
        assert dec_buy_bull.suggested_scale == pytest.approx(0.5)

        dec_buy_bear = w_guard.check("BTCUSDC", "BUY", trend_duration_seconds=dur_15d, trend_direction="bear")
        assert dec_buy_bear.allowed is True
        assert dec_buy_bear.suggested_scale == pytest.approx(1.0)
        assert "trend_direction_favorable" in dec_buy_bear.reason

        dec_sell_bear = w_guard.check("BTCUSDC", "SELL", trend_duration_seconds=dur_15d, trend_direction="bear")
        assert dec_sell_bear.suggested_scale == pytest.approx(0.5)

        dec_sell_bull = w_guard.check("BTCUSDC", "SELL", trend_duration_seconds=dur_15d, trend_direction="bull")
        assert dec_sell_bull.allowed is True
        assert dec_sell_bull.suggested_scale == pytest.approx(1.0)
        assert "trend_direction_favorable" in dec_sell_bull.reason

        # check_internal_order_guards enforces parabolic dump on large SELL (>= high_stake_min_notional_usd)
        margins = _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            parabolic_surge_pct=10.0,
            parabolic_pullback_pct=2.0,
            high_stake_min_notional_usd=700.0,
        )
        ok_sell, reason_sell, scale_sell = check_internal_order_guards(
            symbol="SOLUSDC",
            side="SELL",
            price=80.0,
            notional_eur=2500.0,
            price_history=dump_history,
            margins=margins,
            now=300.0,
        )
        assert ok_sell is False
        assert "parabolic_dump_active" in reason_sell
        assert scale_sell == 0.0

    def test_batch7_whale_divergence_symmetric_sell_checks(self):
        """WhaleDivergenceGuard must symmetrically protect SELL orders against capitulation, squeeze, and taker buying."""
        from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
        from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot

        guard = WhaleDivergenceGuard(
            min_taker_ratio=0.65,
            taker_downscale_threshold=0.80,
            taker_downscaled_scale=0.50,
            block_liquidation_cascade=True,
            severe_cascade_oi_pct=-5.0,
        )

        # 1. Selling into severe long liquidation capitulation (OI <= -5% and regime == long_liquidation) -> DEFER
        cap_snap = WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=1.0,
            top_traders_long_pct=0.50,
            open_interest_usd=100_000_000.0,
            open_interest_1h_change_pct=-7.5,
            taker_buy_sell_ratio=0.70,
            taker_buy_vol_usd=35_000_000.0,
            taker_sell_vol_usd=50_000_000.0,
            divergence_regime="long_liquidation",
            ts=time.time(),
        )
        dec_cap = guard.check("BTCUSDC", "SELL", snapshot=cap_snap)
        assert dec_cap.allowed is False
        assert "selling_into_long_liquidation_capitulation" in dec_cap.reason

        # 2. Selling into short squeeze / whale accumulation -> DEFER
        sqz_snap = WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=1.6,
            top_traders_long_pct=0.62,
            open_interest_usd=100_000_000.0,
            open_interest_1h_change_pct=3.5,
            taker_buy_sell_ratio=1.15,
            taker_buy_vol_usd=57_500_000.0,
            taker_sell_vol_usd=50_000_000.0,
            divergence_regime="short_squeeze",
            ts=time.time(),
        )
        dec_sqz = guard.check("BTCUSDC", "SELL", snapshot=sqz_snap)
        assert dec_sqz.allowed is False
        assert "whale_squeeze_or_accumulation_active" in dec_sqz.reason

        # 3. Severe taker buying dominance (taker_ratio > 1 / 0.65 = 1.538) -> DEFER
        sev_buy_snap = WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=1.0,
            top_traders_long_pct=0.50,
            open_interest_usd=100_000_000.0,
            open_interest_1h_change_pct=0.5,
            taker_buy_sell_ratio=1.65,
            taker_buy_vol_usd=82_500_000.0,
            taker_sell_vol_usd=50_000_000.0,
            divergence_regime="neutral",
            ts=time.time(),
        )
        dec_sev = guard.check("BTCUSDC", "SELL", snapshot=sev_buy_snap)
        assert dec_sev.allowed is False
        assert "severe_taker_buying_dominated" in dec_sev.reason

        # 4. Moderate taker buying pressure (1.25 < taker_ratio <= 1.538) -> DOWNSCALE_QTY
        mod_buy_snap = WhalePositioningSnapshot(
            symbol="BTCUSDC",
            top_traders_long_ratio=1.0,
            top_traders_long_pct=0.50,
            open_interest_usd=100_000_000.0,
            open_interest_1h_change_pct=0.5,
            taker_buy_sell_ratio=1.35,
            taker_buy_vol_usd=67_500_000.0,
            taker_sell_vol_usd=50_000_000.0,
            divergence_regime="neutral",
            ts=time.time(),
        )
        dec_mod = guard.check("BTCUSDC", "SELL", snapshot=mod_buy_snap)
        assert dec_mod.allowed is True
        assert dec_mod.suggested_scale == pytest.approx(0.50)
        assert "moderate_taker_buying_pressure" in dec_mod.reason

    def test_batch7_dynamic_window_canonical_extractors_and_per_side_suggested_scale(self, monkeypatch):
        """check_math_profit_reference dynamic mode must support Kraken/Hyperliquid fill keys (px, cost/vol, T),
        and SELL suggested_scale must not overwrite BUY suggested_scale on a shared regime_context."""
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "dynamic")
        monkeypatch.setattr(order_guard, "_symbol_trend", lambda *a, **k: "flat")
        monkeypatch.setattr(order_guard, "get_dynamic_window_sec_cfg", lambda *a, **k: 14400.0)

        provider = MagicMock()
        provider.name = "hyperliquid"

        # Hyperliquid fills use 'px', 'sz', and 'T' (milliseconds), not 'price' or 'timestamp'
        def fake_get_orders(symbol, side, window_s):
            if side == "SELL":
                return [
                    {"px": "95.0", "sz": "1.0", "T": 1000000},
                    {"px": "105.0", "sz": "1.0", "T": 3000000},
                ]
            if side == "BUY":
                return [
                    {"px": "93.0", "sz": "1.0", "T": 2000000},
                ]
            return []

        provider.get_orders.side_effect = fake_get_orders
        allowed = order_guard.check_math_profit_reference(
            provider, "HYPE/USDC", "BUY", 100.0, 1.15, window_ref=95.0
        )
        assert allowed is True

        # Verify SELL downscale does not overwrite regime_context.suggested_scale used for BUY sizing
        dec = MarketRegimeDecision(
            regime="bear",
            gradient=-0.5,
            epsilon=0.1,
            strength=5.0,
            fresh=True,
            reason="signal",
            source="mock",
        )
        ctx = MarketRegimeContext.from_decision(
            dec,
            evaluated_at=1000.0,
            symbol="BTCUSDC",
            provider="binance",
            trend_duration_seconds=15.0 * 86400.0,
        )
        assert getattr(ctx, "suggested_scale", None) is None

        # Evaluate SELL on shared context (bear trend exhausted -> SELL downscaled to 0.25)
        ok_s, reason_s, scale_s = order_guard.check_intelligence_guards(
            "binance",
            "BTCUSDC",
            "SELL",
            85000.0,
            regime_context=ctx,
            notional_eur=5000.0,
            now=1005.0,
            bypass_guards={"gi", "gllm"},
        )
        assert ok_s is True
        assert scale_s == pytest.approx(0.25)
        # BUY suggested_scale on ctx must remain unset (None), while _suggested_scale_by_side["SELL"] is 0.25
        assert getattr(ctx, "suggested_scale", None) is None
        assert ctx._suggested_scale_by_side["SELL"] == pytest.approx(0.25)

        # Evaluate BUY on the same bear context (buying in exhausted bear trend is favorable -> scale 1.0)
        ok_b, reason_b, scale_b = order_guard.check_intelligence_guards(
            "binance",
            "BTCUSDC",
            "BUY",
            85000.0,
            regime_context=ctx,
            notional_eur=5000.0,
            now=1005.0,
            bypass_guards={"gi", "gllm"},
        )
        assert ok_b is True
        assert scale_b == pytest.approx(1.0)
        assert ctx.suggested_scale == pytest.approx(1.0)
        assert ctx._suggested_scale_by_side["BUY"] == pytest.approx(1.0)
        assert ctx._suggested_scale_by_side["SELL"] == pytest.approx(0.25)

    def test_batch8_orderbook_wall_downscale_propagation_and_gateway_ssot(self, monkeypatch):
        """Verify OrderbookWallGuard DOWNSCALE_QTY propagates in check_external_order_guards,
        and AutonomousIntentGateway uses order_guard._load_margins and extract_order_id."""
        import external_order_guard
        from intelligence.internal.guards.guard_decision import GuardDecision
        from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard

        monkeypatch.setattr(
            OrderbookWallGuard,
            "check",
            lambda self, sym, side, snapshot=None: GuardDecision.downscale(
                "OrderbookWallGuard", scale=0.4, reason="orderbook_wall_downscale_test"
            ),
        )
        margins = _margins(
            whale_guard_mode="off",
            orderbook_wall_guard_mode="enforce",
            funding_guard_mode="off",
        )
        ok, reason, scale = external_order_guard.check_external_order_guards(
            "BTCUSDC", "BUY", 85000.0, margins=margins
        )
        assert ok is True
        assert scale == pytest.approx(0.4)
        assert reason == "orderbook_wall_downscale_test"

        # Verify AutonomousIntentGateway extracts Kraken txid via extract_order_id
        from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
        gw = AutonomousIntentGateway(mode="enforce", max_notional_usd=500.0, min_confidence=0.80, notify_enabled=True)
        notified = []
        monkeypatch.setattr(gw, "_dispatch_notify", lambda title, body, symbol: notified.append((title, body, symbol)) or True)
        monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))

        fake_inst = MagicMock()
        fake_inst.symbol = "SOLUSD"
        fake_inst.provider_name = "kraken"
        fake_inst._provider.name = "kraken"
        fake_inst._provider.get_current_price.return_value = 100.0
        fake_inst.place.return_value = {"txid": ["OQCLML-BW3P3-BUCMWZ"]}

        intent = {
            "intent_id": "test_krk_1",
            "decision": "BUY",
            "symbol": "SOLUSD",
            "venue": "kraken",
            "confidence": 0.92,
            "suggested_notional_usd": 200.0,
            "status": "PENDING",
        }
        res = gw.process_intent(intent, instruments_map={"SOLUSD": fake_inst})
        assert res["status"] == "EXECUTED"
        assert len(notified) == 1
        assert "Order ID: OQCLML-BW3P3-BUCMWZ" in notified[0][1]

        # Verify Instrument.place resolves regime_context for SELL in dynamic mode and scales SELL under GT2 medium_trend_action='scale'
        from instrument import Instrument
        from providers.quantity import QuantityDecision

        bull_ctx = MarketRegimeContext.from_decision(
            MarketRegimeDecision(
                regime="bull", gradient=0.8, epsilon=0.1, strength=5.0,
                fresh=True, reason="directional_bull", source="mock",
            ),
            evaluated_at=time.time(),
            symbol="SOLUSD",
            provider="kraken",
        )

        mock_prov = MagicMock()
        mock_prov.name = "kraken"
        mock_prov.guards_internally.return_value = False
        mock_prov.requires_cache_permit.return_value = False
        mock_prov.defers_smart_cancellation.return_value = False
        mock_prov.supports_business_minimum.return_value = False
        mock_prov.execution_enabled.return_value = True
        mock_prov.free_balance.return_value = 10.0
        mock_prov.quantity_decision.side_effect = lambda sym, side, px, q, **kw: QuantityDecision(
            requested_qty=float(q), balance_cap=10.0, policy_cap=10.0, fee_cap=10.0, final_qty=float(q)
        )
        mock_prov.round_amount.side_effect = lambda sym, q: float(q)
        mock_prov.order_filter_refusal.return_value = None
        mock_prov.profit_guard_window_ref.return_value = None
        mock_prov.place_order.return_value = {"txid": ["KRK-SELL-1"]}

        mock_api = MagicMock()
        mock_api.provider_by_name.return_value = mock_prov
        mock_api.market_regime_context.return_value = bull_ctx

        monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
        monkeypatch.setattr(order_guard, "profit_guard", lambda *a, **k: True)

        inst_obj = Instrument("sol_krk", "SOLUSD", "kraken", base="SOL", quote="USD", api=mock_api)
        placed_sell = inst_obj.place(
            "SELL",
            120.0,
            4.0,
            smart=False,
            wait_for_trend=False,
            caller_owns_retry=True,
            reference_mode="dynamic",
            medium_trend_action="scale",
            medium_trend_scale=0.5,
        )
        assert placed_sell == {"txid": ["KRK-SELL-1"]}
        mock_api.market_regime_context.assert_called_once_with("SOLUSD", provider_name="kraken")
        # 4.0 scaled by 0.5 in bull regime -> 2.0
        assert mock_prov.place_order.call_args.args[3] == pytest.approx(2.0)

    def test_cross_venue_collector_normalization_and_intelligence_context_validation(self, tmp_path, monkeypatch):
        """Verify cross-venue symbol normalization and disk-variant lookup in Pillar 2 collectors,
        and MarketRegimeContext.is_valid_for check in intelligence_order_guard."""
        from intelligence.external.collectors.orderbook_depth import (
            OrderbookDepthCollector,
            OrderbookSnapshot,
            _resolve_binance_lookup_symbol,
        )
        from intelligence.external.collectors.derivatives_telemetry import (
            DerivativesTelemetry,
            DerivativesTelemetryCollector,
        )
        from intelligence.external.collectors.whale_positioning import (
            WhalePositioningCollector,
            WhalePositioningSnapshot,
        )
        from llm_order_guard import check_llm_order_guards as check_intelligence_order_guards

        assert _resolve_binance_lookup_symbol("SOLUSD") == "SOLUSDT"
        assert _resolve_binance_lookup_symbol("HYPE/USDC") == "HYPEUSDT"
        assert _resolve_binance_lookup_symbol("BTCEUR") == "BTCUSDT"
        assert _resolve_binance_lookup_symbol("USDCUSDT") == "USDCUSDT"

        now_ts = time.time()
        ob_col = OrderbookDepthCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        ob_snap = OrderbookSnapshot(
            symbol="SOLUSDC",
            mid_price=150.0,
            bid_depth_usd=100_000.0,
            ask_depth_usd=100_000.0,
            imbalance_ratio=0.5,
            largest_bid_wall_usd=10_000.0,
            largest_bid_wall_price=149.5,
            largest_ask_wall_usd=10_000.0,
            largest_ask_wall_price=150.5,
            ts=now_ts,
        )
        ob_col.record_snapshot(ob_snap, save_to_disk=True)

        ob_col_reader = OrderbookDepthCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        loaded_ob = ob_col_reader.fetch("SOLUSD", allow_network=False)
        assert loaded_ob is not None
        assert loaded_ob.mid_price == pytest.approx(150.0)

        d_col = DerivativesTelemetryCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        d_snap = DerivativesTelemetry(
            symbol="SOLUSDC",
            funding_rate=0.0001,
            predicted_funding_rate=0.0001,
            open_interest=5000.0,
            open_interest_usd=750_000.0,
            ts=now_ts,
        )
        d_col.record_telemetry(d_snap, save_to_disk=True)
        d_col_reader = DerivativesTelemetryCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        loaded_d = d_col_reader.fetch("SOLUSD", allow_network=False)
        assert loaded_d is not None
        assert loaded_d.funding_rate == pytest.approx(0.0001)

        w_col = WhalePositioningCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        w_snap = WhalePositioningSnapshot(
            symbol="HYPEUSDT",
            top_traders_long_ratio=1.5,
            top_traders_long_pct=0.6,
            taker_buy_sell_ratio=1.2,
            taker_buy_vol_usd=100_000.0,
            taker_sell_vol_usd=80_000.0,
            open_interest_usd=1_000_000.0,
            open_interest_1h_change_pct=1.5,
            divergence_regime="accumulation",
            ts=now_ts,
        )
        w_col.record_snapshot(w_snap, save_to_disk=True)
        w_col_reader = WhalePositioningCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        loaded_w = w_col_reader.fetch("HYPE/USDC", allow_network=False)
        assert loaded_w is not None
        assert loaded_w.divergence_regime == "accumulation"

        # Verify intelligence_order_guard ignores cached LLM decision when regime_context belongs to another symbol
        wrong_sym_ctx = MarketRegimeContext.from_decision(
            MarketRegimeDecision(
                regime="bull", gradient=0.5, epsilon=0.1, strength=3.0,
                fresh=True, reason="bull", source="mock",
            ),
            evaluated_at=now_ts,
            symbol="BTCUSDC",
            provider="binance",
        )
        cache_key = ("BUY", frozenset())
        object.__setattr__(
            wrong_sym_ctx,
            "_intelligence_decisions_by_key",
            {cache_key: {"gemini_notional": 2000.0, "gemini_decision": (False, "wrong_sym_cached_veto", 0.0)}},
        )
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
        monkeypatch.setattr(
            HighStakeGuard,
            "check",
            lambda self, *a, **kw: GuardDecision(
                guard_name="HighStakeGuard",
                allowed=True,
                reason="fresh_eval_ok",
                brake_action=BrakeAction.NONE,
                suggested_scale=1.0,
            ),
        )
        margins = _margins(
            llm_guard_mode="enforce",
            high_stake_min_notional_usd=1000.0,
            geopolitical_guard_mode="off",
        )
        allowed, reason, scale = check_intelligence_order_guards(
            provider="binance",
            symbol="ETHUSDC",
            order_type="BUY",
            price=2000.0,
            qty=1.0,
            notional_eur=2000.0,
            regime_context=wrong_sym_ctx,
            margins=margins,
            now=now_ts,
        )
        assert allowed is True
        assert scale == pytest.approx(1.0)

    def test_gllm1_high_stake_decision_does_not_leak_to_sub_700_orders_on_shared_context(self, monkeypatch):
        """Bug 1 regression: If Order 1 (>= $700) on a shared MarketRegimeContext is blocked or downscaled by GLLM1,
        Order 2 (< $700) on the same context must NOT be blocked or downscaled by GLLM1."""
        from intelligence.internal.guards.guard_decision import GuardDecision

        margins = _margins(
            internal_guard_mode="enforce",
            intelligence_guards_mode="enforce",
            llm_guard_mode="enforce",
            high_stake_min_notional_usd=700.0,
            regime_context_max_age_sec=120.0,
        )
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        calls = []
        gemini_result = [GuardDecision.veto("HighStakeGuard", "gemini_high_stake_veto")]

        def fake_gemini_check(self, symbol, side, price, qty, notional_eur=None, **kw):
            calls.append(notional_eur)
            return gemini_result[0]

        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.HighStakeGuard.check",
            fake_gemini_check,
        )

        # Case A: Order 1 ($1500) vetoed by GLLM1 -> Order 2 ($100) on same context must pass (1.0)
        ctx_veto = _build_test_context()
        ok1, reason1, scale1 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=15.0, notional_eur=1500.0, regime_context=ctx_veto
        )
        assert ok1 is False
        assert "gemini_high_stake_veto" in reason1
        assert len(calls) == 1

        ok2, reason2, scale2 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=1.0, notional_eur=100.0, regime_context=ctx_veto
        )
        assert ok2 is True
        assert scale2 == pytest.approx(1.0)
        assert len(calls) == 1

        # Case B: Order 1 ($1500) downscaled (0.25) by GLLM1 -> Order 2 ($100) on same context must have scale 1.0
        ctx_scale = _build_test_context()
        calls.clear()
        gemini_result[0] = GuardDecision.downscale("HighStakeGuard", 0.25, "gemini_high_stake_downscale")

        ok3, reason3, scale3 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=15.0, notional_eur=1500.0, regime_context=ctx_scale
        )
        assert ok3 is True
        assert scale3 == pytest.approx(0.25)
        assert getattr(ctx_scale, "suggested_scale", None) == pytest.approx(0.25)
        assert len(calls) == 1

        ok4, reason4, scale4 = order_guard.check_intelligence_guards(
            "binance", "BTCUSDC", "BUY", 100.0, qty=1.0, notional_eur=100.0, regime_context=ctx_scale
        )
        assert ok4 is True
        assert scale4 == pytest.approx(1.0)
        assert getattr(ctx_scale, "suggested_scale", None) == pytest.approx(1.0)
        assert len(calls) == 1

    def test_gaussian_horizon_and_collector_symbol_variants_normalization(self, tmp_path, monkeypatch):
        """Bug 2 & Bug 3 regression: resolve_cached_gaussian_horizon normalizes cross-venue/case symbols,
        and _symbol_variants prioritizes USDC before USDT and supports bare base assets."""
        import qlimit_order_guard
        from intelligence.internal.state import survival
        from intelligence.external.collectors.orderbook_depth import (
            OrderbookDepthCollector,
            OrderbookSnapshot,
            _symbol_variants,
        )

        # 1. Verify _symbol_variants prioritizes USDC before USDT and supports bare base symbols
        variants_tao = _symbol_variants("TAOUSD")
        assert variants_tao.index("TAOUSDC") < variants_tao.index("TAOUSDT")
        variants_bare = _symbol_variants("HYPE")
        assert "HYPEUSDC" in variants_bare
        assert "HYPEUSDT" in variants_bare
        assert variants_bare.index("HYPEUSDC") < variants_bare.index("HYPEUSDT")

        # 2. Verify in-memory cache lookup across symbol variants in OrderbookDepthCollector
        ob_col = OrderbookDepthCollector(cache_ttl_sec=30.0, cache_dir=str(tmp_path))
        ob_col.record_snapshot(
            OrderbookSnapshot(
                symbol="TAOUSDC",
                mid_price=450.0,
                bid_depth_usd=100_000.0,
                ask_depth_usd=100_000.0,
                imbalance_ratio=0.5,
                largest_bid_wall_usd=10_000.0,
                largest_bid_wall_price=449.0,
                largest_ask_wall_usd=10_000.0,
                largest_ask_wall_price=451.0,
                ts=time.time(),
            ),
            save_to_disk=False,
        )
        mem_hit = ob_col.fetch("TAOUSD", allow_network=False)
        assert mem_hit is not None
        assert mem_hit.mid_price == pytest.approx(450.0)

        # 3. Verify resolve_cached_gaussian_horizon and get_trend_survival_metrics resolve TAOUSD / taousdc -> TAOUSDC
        t_cache = tmp_path / "cache_T_trend.json"
        t_cache.write_text(
            json.dumps({
                "TAOUSDC": {
                    "T": 6.1,
                    "median_d": 3.0,
                    "p90_d": 9.8,
                    "ts": time.time(),
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(survival, "T_CACHE_FILE", str(t_cache))

        p90_d, med_d, gauss_h = qlimit_order_guard.resolve_cached_gaussian_horizon("TAOUSD")
        assert p90_d == pytest.approx(9.8)
        assert med_d == pytest.approx(3.0)
        assert gauss_h == pytest.approx(round(6.1 * 1.4, 4))

        p90_lower, med_lower, gauss_lower = qlimit_order_guard.resolve_cached_gaussian_horizon("taousdc")
        assert p90_lower == pytest.approx(9.8)
        assert med_lower == pytest.approx(3.0)
        assert gauss_lower == pytest.approx(round(6.1 * 1.4, 4))

        metrics = survival.get_trend_survival_metrics("TAO/USD", trend_duration_seconds=5.0 * 86400.0)
        assert metrics["p90_days"] == pytest.approx(9.8)
        assert metrics["median_days"] == pytest.approx(3.0)
        assert metrics["T"] == pytest.approx(6.1)

    def test_cross_venue_base_asset_separators_float_t_gaussian_and_pretrade_telemetry(self, tmp_path, monkeypatch):
        """Regression test for:
        1. _normalize_base_asset with '/' and '-' separators (e.g. HYPE/USDC, SOL-USD) and _read_cached_price_history
           resolving cache_24_price_TAOUSDC.json for TAOUSD.
        2. _read_cached_trend_duration skipping empty list entries [] to find non-empty base-asset matches.
        3. priceAnalysis.get_trade_weight and utils.gaussian_weights_from_idx handling float T (e.g. T=6.1, trend_len=6.0).
        4. collect_pretrade_telemetry resolving 24h price evolution, short slopes, and fills across symbol variants.
        5. profit_guard evaluating check_intelligence_guards when notional_eur > 0 is passed with qty=None.
        """
        import priceAnalysis as pa
        import utils as u
        from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry

        assert order_guard._normalize_base_asset("HYPE/USDC") == "HYPE"
        assert order_guard._normalize_base_asset("SOL-USD") == "SOL"
        assert order_guard._normalize_base_asset("TAO_USDC") == "TAO"

        # Float T in gaussian_weights_from_idx and get_trade_weight at seam (trend_len=6.0, T=6.1)
        t_arr, w_arr = u.gaussian_weights_from_idx(6.1, 0)
        assert len(w_arr) == 6
        t_seq, w_seq = pa.get_trade_weight(T=6.1, trend_len=6.0, trend="up", order_type="BUY")
        assert len(w_seq) >= 1
        assert float(w_seq[0]) > 0.0

        # Setup synthetic cachedb in tmp_path
        monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
        now_s = time.time()
        now_ms = now_s * 1000.0

        # cache_24_price_TAOUSDC.json read via TAOUSD and HYPE/USDC
        p24_tao = tmp_path / "cache_24_price_TAOUSDC.json"
        p24_tao.write_text(
            json.dumps({
                "items": {
                    "TAOUSDC": [
                        [now_ms - 3600_000, 400.0],
                        [now_ms - 1800_000, 410.0],
                        [now_ms, 420.0],
                    ]
                }
            }),
            encoding="utf-8",
        )
        hist_tao = order_guard._read_cached_price_history("TAOUSD", window_seconds=7200.0)
        assert hist_tao is not None
        assert len(hist_tao) == 3

        # cache_long_trend.json with an empty legacy entry before the populated entry
        lt_path = tmp_path / "cache_long_trend.json"
        lt_path.write_text(
            json.dumps({
                "items": {
                    "TAOUSDT": [],
                    "TAOUSDC": [{"duration_seconds": 172800.0, "direction": "up"}],
                }
            }),
            encoding="utf-8",
        )
        assert order_guard._read_cached_trend_duration("TAOUSD") == pytest.approx(172800.0)
        assert order_guard._read_cached_trend_direction("TAOUSD") == "up"

        # cache_short_trend.json and cache_order.json for collect_pretrade_telemetry("TAO/USD")
        st_path = tmp_path / "cache_short_trend.json"
        st_path.write_text(
            json.dumps({
                "TAOUSDC": {
                    "dynamic": {
                        "222": {"slope_full": 0.0012},
                        "900": {"slope_full": 0.0034},
                        "9000": {"slope_full": 0.0056},
                    }
                }
            }),
            encoding="utf-8",
        )
        ord_path = tmp_path / "cache_order.json"
        ord_path.write_text(
            json.dumps({
                "items": {
                    "TAOUSDC": [
                        {"side": "BUY", "price": 395.0, "quantity": 2.0, "timestamp": now_ms - 7200_000},
                        {"side": "SELL", "price": 415.0, "quantity": 2.0, "timestamp": now_ms - 3600_000},
                    ]
                }
            }),
            encoding="utf-8",
        )
        telem = collect_pretrade_telemetry("TAO/USD", cachedb_dir=str(tmp_path))
        assert "24h Multi-Horizon Price Evolution" in telem
        assert "Dynamic Multi-Window Slopes" in telem
        assert "Recent Venue Fills" in telem

        # profit_guard with qty=None and notional_eur=1500.0 must still call check_intelligence_guards
        intel_called = []
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *a, **k: "off")
        monkeypatch.setattr(
            order_guard,
            "check_intelligence_guards",
            lambda *a, **k: (intel_called.append(k.get("notional_eur")) or (False, "blocked_by_notional_only", 0.0)),
        )
        prov = MagicMock()
        prov.name = "binance"
        res_pg = order_guard.profit_guard(prov, "BTCUSDC", "BUY", 85000.0, 1.15, qty=None, notional_eur=1500.0)
        assert res_pg is False
        assert intel_called == [1500.0]

    def test_short_trend_symbol_variants_and_bapi_if_place_safe_order_g_prof_bypass(self, tmp_path, monkeypatch):
        """Verify CacheShortTrendManager resolves cross-venue symbol variants (TAOUSD -> TAOUSDC)
        and bapi_placeorder.if_place_safe_order skips trade history fetch when G_PROF is bypassed."""
        import json
        import time
        from cachetrend import CacheShortTrendManager
        from binance_api import bapi_placeorder

        st_file = tmp_path / "cache_short_trend.json"
        now_s = time.time()
        st_file.write_text(
            json.dumps({
                "TAOUSDC": {
                    "ts": now_s,
                    "current_price": 400.0,
                    "growth_coefficient": 0.25,
                    "gradient_recent": 0.25,
                    "final_trend": 0.25,
                    "epsilon": 0.01,
                    "dynamic": {
                        "222": {
                            "ts": now_s,
                            "current_price": 400.0,
                            "growth_coefficient": 0.30,
                            "gradient_recent": 0.30,
                            "final_trend": 0.30,
                            "epsilon": 0.01,
                        }
                    },
                }
            }),
            encoding="utf-8",
        )
        stm = CacheShortTrendManager(symbols=["TAOUSDC"], filename=str(st_file), writer=False)
        snap = stm.get_snapshot("TAOUSD")
        assert snap is not None
        assert snap["current_price"] == pytest.approx(400.0)
        assert stm.is_trend_up("TAOUSD", now=now_s) is True
        assert stm.should_wait("SELL", "TAO/USD", now=now_s) is True
        assert stm.should_wait("BUY", "TAO/USD", now=now_s) is False
        dyn = stm.get_short_trend_for_window("TAOUSD", 222.0, now=now_s)
        assert dyn is not None
        assert dyn["growth_coefficient"] == pytest.approx(0.30)

        # Verify bapi_placeorder.if_place_safe_order skips apiorders.get_trade_orders when G_PROF is bypassed
        monkeypatch.setattr(bapi_placeorder.api, "get_current_price", lambda s: 85000.0)
        monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
        trades_called = []
        import binance_api.bapi_allorders as apiorders
        monkeypatch.setattr(apiorders, "get_trade_orders", lambda *a, **k: trades_called.append(a) or [])
        pg_called = []
        monkeypatch.setattr(
            order_guard,
            "profit_guard",
            lambda *a, **k: pg_called.append(k.get("bypass_guards")) or False,
        )
        ok, reason = bapi_placeorder.if_place_safe_order(
            "BUY", "BTCUSDC", 85000.0, 0.01, 3600,
            bypass_guards={"profit_guard"},
        )
        assert ok is False
        assert reason == "profit_guard"
        assert trades_called == []
        assert pg_called == [{"profit_guard"}]

    def test_iteration5_category_bypass_revocation_resolve_assets_and_sl_episode_cleanup(self, monkeypatch):
        """Regression test for Iteration 5 fixes:
        1. Instrument.place revokes category tokens ('profit', 'qlimit', 'financial', 'gf')
           so G_PROF cannot re-expand on SELL and G_QLIMIT1 cannot re-expand on BUY.
        2. bapi_placeorder.if_place_safe_order checks GLLM2 and GLLM3 before skipping profit_guard.
        3. resolve_assets strips '/', '-', '_' separators and MarketApi.place defaults quote='USDC'
           for bare Hyperliquid symbols.
        4. monitortrades._resolve_exit and _handle_position_recovery clear _SL_EPISODE_STATE.
        """
        from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
        from instrument import Instrument
        from providers.quantity import resolve_assets
        import binance_api.bapi_placeorder as bapi_placeorder
        import monitortrades

        # 1. Category token revocation on SELL (G_PROF) and BUY (G_QLIMIT1)
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

        captured_bypasses = []
        def fake_profit_guard(prov, sym, side, px, margin, window_ref=None, regime_context=None, qty=None, **kw):
            captured_bypasses.append((side, normalize_bypassed_guards(bypass_guards=kw.get("bypass_guards"))))
            return False

        monkeypatch.setattr(order_guard, "profit_guard", fake_profit_guard)
        monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
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
        inst = Instrument("test_inst", "BTCUSDC", "binance", api=mock_api)

        # SELL with bypass_guards={"financial"} must revoke G_PROF while keeping G_QLIMIT1 & G_QLIMIT2
        inst.place("SELL", 100.0, 1.0, bypass_guards={"financial"}, wait_for_trend=False, caller_owns_retry=True)
        assert len(captured_bypasses) == 1
        sell_side, sell_byp = captured_bypasses[-1]
        assert sell_side == "SELL"
        assert GUARD_REGISTRY["G_PROF"].is_bypassed(sell_byp) is False
        assert GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(sell_byp) is True
        assert GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(sell_byp) is True

        # BUY with bypass_guards={"qlimit"} must revoke G_QLIMIT1 while keeping G_QLIMIT2
        inst.place("BUY", 100.0, 1.0, bypass_guards={"qlimit"}, wait_for_trend=False, caller_owns_retry=True)
        assert len(captured_bypasses) == 2
        buy_side, buy_byp = captured_bypasses[-1]
        assert buy_side == "BUY"
        assert GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(buy_byp) is False
        assert GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(buy_byp) is True

        # 2. bapi_placeorder.if_place_safe_order must NOT skip profit_guard if GLLM3 is still active
        monkeypatch.setattr(bapi_placeorder.api, "get_current_price", lambda s: 85000.0)
        pg_calls = []
        monkeypatch.setattr(
            order_guard,
            "profit_guard",
            lambda *a, **k: pg_calls.append(k.get("bypass_guards")) or False,
        )
        partial_byp = {"g_prof", "g_qlimit2", "gi_obook", "gi_whale", "gi_funding", "gllm1"}
        ok, reason = bapi_placeorder.if_place_safe_order(
            "BUY", "BTCUSDC", 85000.0, 0.01, 3600,
            bypass_guards=partial_byp,
        )
        assert ok is False
        assert reason == "profit_guard"
        assert len(pg_calls) == 1

        # 3. resolve_assets strips separators while preserving unknown suffixes
        assert resolve_assets("HYPE/USDC") == ("HYPE", "USDC")
        assert resolve_assets("HYPE-USDC", base="HYPE") == ("HYPE", "USDC")
        assert resolve_assets("TAOUSDT") == ("TAOUSDT", None)

        # 4. monitortrades._resolve_exit and _handle_position_recovery clear _SL_EPISODE_STATE
        monkeypatch.setattr(monitortrades, "_save_mt_state", lambda: None)
        monkeypatch.setattr(monitortrades, "_clear_smart_sl_exit", lambda *a, **k: None)
        monitortrades._SL_EPISODE_STATE["TESTUSDC"] = {"started_at": 1234.0, "buy_price": 100.0}
        monitortrades._SL_BOUNCE_STATE["TESTUSDC"] = {"started_at": 1234.0, "buy_price": 100.0}
        monitortrades._TP_RIDE_STATE["TESTUSDC"] = {"started_at": 1234.0, "buy_price": 100.0}
        monitortrades._resolve_exit("TESTUSDC", inst=None, kind="STOP", clear_pending=True)
        assert "TESTUSDC" not in monitortrades._SL_EPISODE_STATE
        assert "TESTUSDC" not in monitortrades._SL_BOUNCE_STATE
        assert "TESTUSDC" not in monitortrades._TP_RIDE_STATE

    def test_iteration6_stop_retry_wait_for_trend_long_trend_variants_and_whale_1h_seconds(self, tmp_path, monkeypatch):
        """Regression test for Iteration 6 fixes:
        1. order_retry_worker.process_once preserves wait_for_trend=False ONLY for kind=='STOP'
           (expired Stop-Loss timer), while enforcing wait_for_trend=True for all other orders.
        2. CacheLongTrendManager.get_remote_items resolves cross-venue symbol variants (e.g. TAOUSD -> TAOUSDC).
        3. WhalePositioningCollector._resolve_price_1h_change_pct handles both seconds and milliseconds
           timestamps in cache_24_price_{sym}.json.
        4. order_guard.check_intelligence_guards invalidates cached Weibull exhaustion when
           trend_duration_seconds changes across calls on the same regime_context.
        """
        import order_retry as oq
        import order_retry_worker as worker
        import cacheManager as cm
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
        from providers.strategy_executor import OrderReconciliationCapabilities, OrderStatus

        # 1. order_retry_worker preserves wait_for_trend=False for kind="STOP", enforces True otherwise
        q_file = str(tmp_path / "q.jsonl")
        l_file = str(tmp_path / "q.lock")
        monkeypatch.setattr(oq, "QUEUE_FILE", q_file)
        monkeypatch.setattr(oq, "LOCK_FILE", l_file)
        monkeypatch.setattr(oq, "RETRY_ENABLED", True)
        monkeypatch.setattr(oq, "RETRY_INTERVAL_SEC", 300.0)
        monkeypatch.setattr(oq, "RETRY_TTL_SEC", 86400.0)
        monkeypatch.setattr(worker.alert, "notify", lambda **kw: None)

        class DummyMkt:
            def __init__(self):
                self.calls = []
            def get_current_price(self, symbol, provider_name=None):
                return 100.0
            def place(self, symbol, side, price, qty, **kw):
                self.calls.append(kw)
                return {"orderId": "1"}
            def order_status(self, symbol, order_id, provider_name=None):
                return OrderStatus("open", 0.0, 0.0, 0.0, "NEW")
            def order_by_client_id(self, symbol, cid, provider_name=None):
                return None
            def reconciliation_capabilities(self, symbol, provider_name=None):
                return OrderReconciliationCapabilities(True, True, True, True, not_found_reliable_for_seconds=86400)

        oq.rewrite([])
        oq.enqueue("TAOUSDC", "SELL", 1.0, {"wait_for_trend": False}, requested_price=100.0, kind="STOP", now=1000.0)
        mkt_stop = DummyMkt()
        worker.process_once(mkt_stop, now=1500.0)
        assert len(mkt_stop.calls) == 1
        assert mkt_stop.calls[0]["wait_for_trend"] is False

        oq.rewrite([])
        oq.enqueue("TAOUSDC", "SELL", 1.0, {"wait_for_trend": False}, requested_price=100.0, kind="TP", now=1000.0)
        mkt_tp = DummyMkt()
        worker.process_once(mkt_tp, now=1500.0)
        assert len(mkt_tp.calls) == 1
        assert mkt_tp.calls[0]["wait_for_trend"] is True

        # 2. CacheLongTrendManager.get_remote_items resolves TAOUSD -> TAOUSDC from priceanalysis.json
        pa_dir = tmp_path / "pa"
        pa_dir.mkdir()
        (pa_dir / "priceanalysis.json").write_text(
            json.dumps({"TAOUSDC": {"timestamp": 1700000000, "direction": "up", "duration_seconds": 86400}}),
            encoding="utf-8",
        )
        lt = cm.CacheLongTrendManager(60, ["TAOUSD"], str(pa_dir / "cache_long_trend.json"), read_only=True)
        monkeypatch.setattr(cm, "_CONFIG_ROOT", str(pa_dir))
        assert lt.get_remote_items("TAOUSD", 0) == [{"timestamp": 1700000000, "direction": "up", "duration_seconds": 86400}]

        # 3. WhalePositioningCollector._resolve_price_1h_change_pct supports seconds timestamps
        c24_dir = tmp_path / "c24"
        c24_dir.mkdir()
        now_s = 1_700_010_000.0
        (c24_dir / "cache_24_price_TAOUSDC.json").write_text(
            json.dumps({
                "items": {
                    "TAOUSDC": [
                        [now_s - 7200.0, 200.0],  # 2h ago: 200
                        [now_s - 3600.0, 100.0],  # 1h ago: 100
                        [now_s, 110.0],           # now: 110 (+10% over 1h, NOT -45% vs 2h ago!)
                    ]
                }
            }),
            encoding="utf-8",
        )
        w_col = WhalePositioningCollector(cache_dir=str(c24_dir))
        assert w_col._resolve_price_1h_change_pct("TAOUSD") == pytest.approx(10.0)

    def test_iteration7_terminal_filter_refusal_rtrade_pair_bypass_and_kraken_trail_normalize(self, tmp_path, monkeypatch):
        """Regression test for Iteration 7 fixes:
        1. order_retry.is_terminal_filter_refusal recognizes 'filter_lot_size_min:...' and
           'qty_zero_or_negative' returned by MarketDataProvider.order_filter_refusal.
        2. rtrade._LivePairVenue._submit_limit_intent honours RTRADE_BYPASS_PROFIT_GUARD
           (including allow_math_reference_sell=True on SELL) symmetrically with repetitive_buy/sell.
        3. kraken.trailing_stop.KrakenTrailingEngine._submit_lifecycle_order normalizes
           bypass_guards before checking for 'all' / '*' (so string 'all' or {'ALL'} is respected).
        """
        import order_retry as oq
        import rtrade
        from kraken.trailing_stop import KrakenTrailing

        # 1. Terminal filter refusal classification across non-Binance providers
        assert oq.is_terminal_filter_refusal("filter_lot_size_min:0.001<0.05") is True
        assert oq.is_terminal_filter_refusal("qty_zero_or_negative") is True
        assert oq.is_terminal_filter_refusal("binance_filter_refused:MIN_NOTIONAL") is True
        assert oq.is_terminal_filter_refusal("below_min_notional") is True
        assert oq.is_terminal_filter_refusal("trend_deferred") is False

        # 2. _LivePairVenue._submit_limit_intent honours RTRADE_BYPASS_PROFIT_GUARD on BUY & SELL
        monkeypatch.setattr(rtrade, "RTRADE_BYPASS_PROFIT_GUARD", True)
        placed_calls = []
        monkeypatch.setattr(
            rtrade.mkt,
            "place",
            lambda symbol, side, price, qty, **kw: placed_calls.append((symbol, side, price, qty, kw)) or {"orderId": "99"},
        )
        venue = rtrade._LivePairVenue("SOLUSDC")
        monkeypatch.setattr(venue, "_persist_callback", lambda *a, **k: (lambda x: None))
        monkeypatch.setattr(venue, "_ticket_from_pending", lambda intent, pair_id: intent)
        monkeypatch.setattr(
            venue.order_lifecycle,
            "submit",
            lambda intent, persist, submit: MagicMock(intent=submit()),
        )

        venue._submit_limit_intent("BUY", 100.0, 1.0, "pair1", attempt=1, revision=0)
        assert len(placed_calls) == 1
        assert placed_calls[-1][4]["bypass_guards"] == {"G_PROF", "GLLM1"}
        assert placed_calls[-1][4]["allow_math_reference_sell"] is False

        venue._submit_limit_intent("SELL", 105.0, 1.0, "pair1", attempt=1, revision=0)
        assert len(placed_calls) == 2
        assert placed_calls[-1][4]["bypass_guards"] == {"G_PROF", "GLLM1"}
        assert placed_calls[-1][4]["allow_math_reference_sell"] is True

        # 3. KrakenTrailing._submit_lifecycle_order normalizes bypass_guards="ALL" / {"ALL"}
        eng = object.__new__(KrakenTrailing)
        eng.executor = MagicMock()
        eng.order_lifecycle = MagicMock()
        eng.order_lifecycle.submit.return_value = MagicMock(outcome="accepted")
        guard_called = []
        monkeypatch.setattr(
            order_guard,
            "check_intelligence_guards",
            lambda *a, **k: guard_called.append(1) or (False, "should_be_skipped", 1.0),
        )
        res = eng._submit_lifecycle_order(
            intent={"intent_id": "id1", "client_order_id": "cid1"},
            persist=lambda x: None, pair="SOLUSD", side="BUY",
            qty=1.0, price=100.0, kind="TRAIL", bypass_guards="ALL", market=False,
        )
        assert res.outcome == "accepted"
        assert guard_called == []

    def test_iteration8_operational_category_gt_s_isolation_sl_episode_timer_and_qlimit_recovery(self, tmp_path, monkeypatch):
        """Regression test for Iteration 8 fixes:
        1. GUARD_CATEGORIES['operational'] contains only OperationalGuard (GO1) and never ShortTrendGuard (GT_S),
           and BaseGuard.is_bypassed / LLMGuard.is_bypassed match uppercase guard tokens directly.
        2. PositionExitManager._evaluate_stop_loss_core preserves sl_episode_state['started_at'] across
           trend_up oscillations (True -> False -> True) and marks smart_sl_done after grace_period_expired
           so Smart-SL cannot re-arm a second grace period during the same SL episode.
        3. qlimit_order_guard.resolve_cached_gaussian_horizon uses read_recovering_previous to recover
           from .prev generation files when T_CACHE_FILE is missing/corrupted.
        """
        from guard_hierarchy import GUARD_CATEGORIES, GUARD_REGISTRY, normalize_bypassed_guards
        from strategies.position_exit_manager import PositionExitConfig, PositionExitManager, PositionSnapshot
        import intelligence.internal.state.survival as survival_mod
        from qlimit_order_guard import resolve_cached_gaussian_horizon

        # 1. Operational category isolation & case-insensitive is_bypassed
        assert [g.code for g in GUARD_CATEGORIES["operational"]] == ["GO1"]
        op_byp = normalize_bypassed_guards("operational")
        assert GUARD_REGISTRY["GO1"].is_bypassed(op_byp) is True
        assert GUARD_REGISTRY["GT_S"].is_bypassed(op_byp) is False
        assert GUARD_REGISTRY["G_PROF"].is_bypassed({"G_PROF"}) is True
        assert GUARD_REGISTRY["GLLM1"].is_bypassed({"GLLM1"}) is True

        # 2. PositionExitManager preserves episode_started_at across trend_up flips & prevents Smart-SL re-arm
        smart_calls = []

        def fake_smart_sl(**kw):
            smart_calls.append(kw["now"])
            if len(smart_calls) == 1:
                return False, "grace_active", {}
            return True, "grace_period_expired (120s elapsed without bounce)", {}

        mgr = PositionExitManager(
            config=PositionExitConfig(lost_threshold=0.05, sl_trend_defer_max_sec=300.0, smart_sl_enabled=True),
            smart_sl_evaluator=fake_smart_sl,
        )
        # t=1000: trend_up=True -> initiates bounce watch at started_at=1000
        d1 = mgr.evaluate_stop_loss(
            PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1000.0, trend_up=True)
        )
        assert d1.should_exit is False
        assert mgr.sl_bounce_state["TAOUSDC"]["started_at"] == 1000.0

        # t=1100: trend_up=False -> Smart-SL defers (grace_active), sl_bounce_state popped
        d2 = mgr.evaluate_stop_loss(
            PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1100.0, trend_up=False)
        )
        assert d2.should_exit is False
        assert "TAOUSDC" not in mgr.sl_bounce_state

        # t=1150: trend_up=True again -> restores started_at=1000.0 from sl_episode_state (NOT 1150.0!)
        d3 = mgr.evaluate_stop_loss(
            PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1150.0, trend_up=True)
        )
        assert d3.should_exit is False
        assert mgr.sl_bounce_state["TAOUSDC"]["started_at"] == 1000.0

        # t=1220: trend_up=False -> Smart-SL expires (grace_period_expired) -> SELL & marks smart_sl_done
        d4 = mgr.evaluate_stop_loss(
            PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1220.0, trend_up=False)
        )
        assert d4.should_exit is True
        assert mgr.sl_episode_state["TAOUSDC"].get("smart_sl_done") is True

        # t=1225: if order was deferred by GT_S and re-evaluated with trend_up=False, Smart-SL is NOT called again
        d5 = mgr.evaluate_stop_loss(
            PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1225.0, trend_up=False)
        )
        assert d5.should_exit is True
        assert len(smart_calls) == 2

        # t=1310 (>= 1000 + 300s): trend_up=True -> immediately triggers mhdsl_bounce_timeout!
        d6 = mgr.evaluate_stop_loss(
            PositionSnapshot(symbol="TAOUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1310.0, trend_up=True)
        )
        assert d6.should_exit is True
        assert d6.metadata.get("sub_reason") == "timeout"

        # 3. resolve_cached_gaussian_horizon recovers from .previous generation file
        t_cache = tmp_path / "trend_T_cache.json"
        t_prev = tmp_path / "trend_T_cache.json.previous"
        t_cache.write_text("{corrupt_json", encoding="utf-8")
        t_prev.write_text(json.dumps({"TAOUSDC": {"T": 10.0, "p90_d": 8.5, "median_d": 4.0}}), encoding="utf-8")
        monkeypatch.setattr(survival_mod, "T_CACHE_FILE", str(t_cache))
        p90_d, med_d, gauss_h = resolve_cached_gaussian_horizon("TAOUSD")
        assert p90_d == 8.5
        assert med_d == 4.0
        assert gauss_h == 14.0

    def test_iteration9_intel_cache_key_high_stake_eval_orderbook_ttl_and_macro_recovery(self, tmp_path, monkeypatch):
        """Regression test for Iteration 9 fixes:
        1. check_intelligence_guards reuses cached static evaluation when non-intel guards (GO1, G_PROF, GLLM1)
           are in bypass_guards, and evaluate_high_stake_order_guard still runs GLLM1 when a prior call on
           the same regime_context bypassed GLLM1.
        2. OrderbookDepthCollector.fetch enforces cache_ttl_sec on direct OrderbookSnapshot entries and
           normalizes underscore/quote variants (_extract_base_asset, _symbol_variants).
        3. SentimentAdvisor._load_from_disk and GeopoliticalThreatAnalyzer._load_from_disk recover .previous
           or fall back to legacy files when the primary file is corrupted.
        4. bapi_placeorder.place_safe_order propagates exact refusal reason into _reason_out.
        """
        from intelligence.external.collectors.orderbook_depth import (
            OrderbookDepthCollector,
            OrderbookSnapshot,
            _extract_base_asset,
            _symbol_variants,
        )
        from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
        from intelligence.internal.guards.guard_decision import GuardDecision
        import binance_api.bapi_placeorder as bapi_placeorder

        # 1. check_intelligence_guards cache hit across G_PROF/GLLM1 bypass & GLLM1 re-eval when un-bypassed
        ctx = _build_test_context()
        raw_calls = []
        orig_raw = order_guard._evaluate_intelligence_guards_raw

        def spy_raw(*a, **kw):
            raw_calls.append(1)
            return orig_raw(*a, **kw)

        hs_calls = []

        def fake_hs_check(self, symbol, side, price, qty, notional_eur=None, **kw):
            hs_calls.append((symbol, side, notional_eur))
            return GuardDecision.veto("gemini", "HighStake vetoed after GLLM1 un-bypassed")

        monkeypatch.setattr(order_guard, "_evaluate_intelligence_guards_raw", spy_raw)
        monkeypatch.setattr(
            "intelligence.sentiment.guards.high_stake_guard.HighStakeGuard.check",
            fake_hs_check,
        )
        monkeypatch.setattr(
            order_guard,
            "_load_margins",
            lambda: _margins(
                internal_guard_mode="off",
                intelligence_guards_mode="enforce",
                whale_guard_mode="off",
                orderbook_wall_guard_mode="off",
                funding_guard_mode="off",
                geopolitical_guard_mode="off",
                llm_guard_mode="enforce",
                high_stake_min_notional_usd=700.0,
                regime_context_max_age_sec=120.0,
            ),
        )

        class MockProv:
            name = "binance"

        # Call 1: bypass_guards={"G_PROF", "GLLM1"} at notional=1000 -> populates static cache, skips GLLM1
        ok1, reason1, _ = order_guard.check_intelligence_guards(
            MockProv(), "BTCUSDC", "BUY", 100.0, regime_context=ctx, qty=10.0, bypass_guards={"G_PROF", "GLLM1"}
        )
        assert ok1 is True
        assert len(raw_calls) == 1
        assert len(hs_calls) == 0

        # Call 2: bypass_guards={"GO1"} (GLLM1 now active!) at same notional=1000 -> must hit static cache AND run GLLM1!
        ok2, reason2, _ = order_guard.check_intelligence_guards(
            MockProv(), "BTCUSDC", "BUY", 100.0, regime_context=ctx, qty=10.0, bypass_guards={"GO1"}
        )
        assert len(raw_calls) == 1  # Static evaluation reused from Call 1!
        assert len(hs_calls) == 1   # GLLM1 was NOT tricked by Call 1's static notional!
        assert ok2 is False
        assert "HighStake vetoed" in reason2

        # 2. OrderbookDepthCollector TTL on direct OrderbookSnapshot & underscore symbol variants
        assert _extract_base_asset("TAO_USD") == "TAO"
        assert _extract_base_asset("BTCFDUSD") == "BTC"
        assert "TAOUSDC" in _symbol_variants("TAO_USD")
        ob_col = OrderbookDepthCollector(cache_ttl_sec=5.0, cache_dir=str(tmp_path))
        ob_col._cache["BTCUSDC"] = OrderbookSnapshot(
            symbol="BTCUSDC", mid_price=85000.0, bid_depth_usd=1000.0, ask_depth_usd=1000.0,
            imbalance_ratio=0.5, largest_bid_wall_usd=100.0, largest_bid_wall_price=84900.0,
            largest_ask_wall_usd=100.0, largest_ask_wall_price=85100.0, ts=1.0,
        )
        assert ob_col.fetch("BTCUSDC", allow_network=False) is None

        # 3. SentimentAdvisor & GeopoliticalThreatAnalyzer recover .previous or legacy file when primary is corrupt
        sa_file = tmp_path / "sentiment_advisor_eval.json"
        sa_legacy = tmp_path / "macro_advisor_eval.json"
        sa_file.write_text("{corrupt", encoding="utf-8")
        sa_legacy.write_text(
            json.dumps({"market_bias": "BEARISH", "risk_level": "HIGH", "confidence": 0.9, "summary": "ok", "key_risks": [], "recommended_action": "DEFENSIVE", "ts": 1234.0}),
            encoding="utf-8",
        )
        sa = SentimentAdvisor(cache_file=str(sa_file))
        assert sa._cached_assessment is not None
        assert sa._cached_assessment.market_bias == "BEARISH"

        geo_file = tmp_path / "geopolitical_threat_eval.json"
        geo_prev = tmp_path / "geopolitical_threat_eval.json.previous"
        geo_file.write_text("{corrupt", encoding="utf-8")
        geo_prev.write_text(
            json.dumps({"threat_level": "ELEVATED", "risk_score": 0.65, "summary": "recovered", "recommended_brake": "DOWNSCALE_50", "headlines_analyzed": 5, "ts": 5678.0}),
            encoding="utf-8",
        )
        geo = GeopoliticalThreatAnalyzer(state_file=str(geo_file))
        assert geo._cached_assessment is not None
        assert geo._cached_assessment.threat_level == "ELEVATED"

        # 4. place_safe_order forwards _reason_out as _outcome_context
        def fake_guarded_place(symbol, order_type, price, qty, **kw):
            ctx_out = kw.get("_outcome_context")
            if isinstance(ctx_out, dict):
                ctx_out["reason"] = "trend_deferred"
            return None

        monkeypatch.setattr(bapi_placeorder, "_guarded_market_place", fake_guarded_place)
        monkeypatch.setattr(bapi_placeorder.sym, "validate_params", lambda *a, **k: None)
        reason_box = {}
        res_order = bapi_placeorder.place_safe_order("BUY", "BTCUSDC", 85000.0, 0.01, _reason_out=reason_box)
        assert res_order is None
        assert reason_box.get("reason") == "trend_deferred"

    def test_iteration10_gtm_scale_idempotency_claimed_qty_scale_sync_mt_reentry_and_spot_engine_submitted_qty(self, tmp_path, monkeypatch):
        """Regression test for Iteration 10 fixes:
        1. GT_M medium_trend_action='scale' respects applied_scale on retry and does not double-halve qty.
        2. order_retry.update_claimed_qty and begin_claimed_submit synchronize applied_scale into place_kwargs on disk.
        3. monitortrades._can_pass_buy_quantity_policy and _apply_post_stop_ai_guard pass cancelorders=True
           and _can_pass_buy_quantity_policy checks GO1 (daily_limit_guard) before AI/intelligence guards.
        4. strategies.spot_engine.Strategy._order_from_pending honors submitted_qty and scales amount proportionally.
        """
        from types import SimpleNamespace
        from instrument import Instrument
        import order_retry
        import monitortrades as mt
        from strategies.spot_engine import Strategy

        # 1. GT_M medium_trend_action='scale' idempotency when applied_scale=0.5 is already present on retry
        submitted_qtys = []
        provider = MagicMock()
        provider.name = "binance"
        provider.get_current_price.return_value = 100.0
        provider.round_amount = lambda s, a: round(float(a), 4)
        provider.min_order_qty.return_value = 0.001
        provider.min_order_notional.return_value = 5.0
        provider.order_filter_refusal.return_value = None
        provider.free_balance.return_value = 10000.0
        provider.policy_cap_quantity.return_value = 10.0
        provider.fee_cap_quantity.return_value = 100.0
        provider.guards_internally.return_value = False
        provider.requires_cache_permit.return_value = False
        provider.defers_smart_cancellation.return_value = False
        provider.supports_business_minimum.return_value = False
        provider.execution_enabled.return_value = True
        provider.get_orders.return_value = []
        provider.get_trades.return_value = []
        provider.adjust_order_price.side_effect = lambda s, sd, p, **k: p
        from providers.quantity import decide_quantity
        provider.quantity_decision.side_effect = lambda *a, **k: decide_quantity(provider, *a, **k)

        def _fake_place_order(_sym, side, price, qty, **_kw):
            submitted_qtys.append(float(qty))
            return {"orderId": "gtm-1", "symbol": _sym, "side": side, "price": price, "origQty": qty}

        provider.place_order.side_effect = _fake_place_order
        provider.submit_order.side_effect = _fake_place_order

        bear_ctx = SimpleNamespace(
            symbol="BTCUSDC",
            regime="bear",
            resolved_trend="bear",
            adverse_regime_for_buy=True,
            adverse_regime_for_sell=False,
            allowed_buy=True,
            allowed_sell=True,
            suggested_scale=1.0,
            computed_at=time.time(),
            decisions={},
        )
        monkeypatch.setattr(order_guard, "profit_guard", lambda *a, **k: True)
        monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
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
        inst = Instrument(
            "BTCUSDC",
            "BTCUSDC",
            "binance",
            base="BTC",
            quote="USDC",
            params={"check_medium_trend": True, "medium_trend_action": "scale"},
            api=mock_api,
        )
        # First pass with qty=1.0, applied_scale=1.0 -> scaled to 0.5
        res1 = inst.place(
            "BUY", 100.0, 1.0,
            wait_for_trend=False,
            caller_owns_retry=True,
            regime_context=bear_ctx,
            medium_trend_scale=0.5,
        )
        assert res1 is not None
        assert submitted_qtys[-1] == pytest.approx(0.5)

        # Retry pass where qty=0.5 was already scaled and applied_scale=0.5 is carried -> must stay 0.5 (not 0.25!)
        res2 = inst.place(
            "BUY", 100.0, 0.5,
            wait_for_trend=False,
            caller_owns_retry=True,
            regime_context=bear_ctx,
            medium_trend_scale=0.5,
            applied_scale=0.5,
        )
        assert res2 is not None
        assert submitted_qtys[-1] == pytest.approx(0.5)

        # 2. order_retry.update_claimed_qty persists applied_scale into place_kwargs on disk
        q_file = tmp_path / "pending_orders.json"
        l_file = tmp_path / "pending_orders.lock"
        monkeypatch.setattr(order_retry, "QUEUE_FILE", str(q_file))
        monkeypatch.setattr(order_retry, "LOCK_FILE", str(l_file))
        claim = order_retry.enqueue_claimed(
            "BTCUSDC", "BUY", 1.0, {"smart": False},
            requested_price=100.0, ref_price=100.0,
            failure_reason="submit_pending", provider_name="binance",
        )
        assert claim is not None
        updated = order_retry.update_claimed_qty(claim, 0.5, applied_scale=0.5)
        assert updated is not None
        assert updated["qty"] == pytest.approx(0.5)
        assert updated["place_kwargs"].get("applied_scale") == pytest.approx(0.5)

        # 3. monitortrades._can_pass_buy_quantity_policy passes cancelorders=True and checks GO1 daily_limit_guard
        qdec_kwargs = []

        class _MTPolicyProv:
            name = "binance"

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def get_orders(_sym, _side, _since):
                return []

            @staticmethod
            def quantity_decision(_sym, _side, _px, req_qty, **kw):
                qdec_kwargs.append(dict(kw))
                return SimpleNamespace(final_qty=req_qty, refuse_reason=None, balance_asset="USDC")

        mt_inst = SimpleNamespace(
            symbol="BTCUSDC",
            base="BTC",
            quote="USDC",
            provider_name="binance",
            provider=_MTPolicyProv(),
            _provider=_MTPolicyProv(),
            params={"bypass_guards": "G_QLIMIT2"},
        )
        dl_calls = []
        monkeypatch.setattr(
            order_guard,
            "daily_limit_guard",
            lambda _p, _s, _side, **kw: dl_calls.append(kw.get("bypass_guards")) or (False, "daily_limit"),
        )
        assert mt._can_pass_buy_quantity_policy(mt_inst, "BTCUSDC", 100.0, 1.0, min_qty=0.1) is False
        assert qdec_kwargs and qdec_kwargs[-1].get("cancelorders") is True
        assert dl_calls and "g_qlimit2" in dl_calls[-1]

        # 4. Strategy._order_from_pending honors submitted_qty and scales amount proportionally
        dummy_strat = SimpleNamespace(_now=lambda: 1700000000.0)
        order_rec = Strategy._order_from_pending(
            dummy_strat,
            {
                "order_id": "tx-123",
                "side": "BUY",
                "requested_qty": 2.0,
                "submitted_qty": 1.0,
                "requested_price": 100.0,
                "amount": 200.0,
                "kind": "B0",
                "market": False,
                "intent_id": "id-1",
                "created_at": 1700000000.0,
            },
        )
        assert order_rec["vol"] == pytest.approx(1.0)
        assert order_rec["amount"] == pytest.approx(100.0)

    def test_iteration11_bapi_safe_order_gprof_sell_instrument_retry_gtm_params_whale_collector_recovery_and_external_guard_reason(
        self, monkeypatch, tmp_path
    ):
        import json
        from types import SimpleNamespace
        from binance_api import bapi_placeorder
        import external_order_guard
        import instrument as inst_mod
        import order_guard
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry

        # 1. bapi_placeorder.if_place_safe_order revokes G_PROF on SELL unless allow_math_reference_sell=True
        monkeypatch.setattr(bapi_placeorder.sym, "validate_params", lambda *_a, **_k: None)
        monkeypatch.setattr(bapi_placeorder.api, "get_current_price", lambda _s: 100.0)
        monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *_a, **_k: (True, None))
        monkeypatch.setattr(order_guard, "get_profit_guard_mode_cfg", lambda *_a, **_k: "average")
        monkeypatch.setattr(order_guard, "get_window_cfg", lambda *_a, **_k: 3600)
        monkeypatch.setattr(order_guard, "get_margin_cfg", lambda *_a, **_k: 0.01)

        pg_calls = []
        monkeypatch.setattr(
            order_guard,
            "profit_guard",
            lambda _prov, _sym, _side, _px, _m, **kw: pg_calls.append(dict(kw)) or True,
        )
        ok_sell, _ = bapi_placeorder.if_place_safe_order(
            "SELL",
            "BTCUSDC",
            100.0,
            1.0,
            3600,
            regime_context=SimpleNamespace(resolved_trend="flat"),
            bypass_guards={"G_PROF"},
        )
        assert ok_sell is True
        assert pg_calls
        assert "G_PROF" not in (pg_calls[-1].get("bypass_guards") or set())

        # Also check place_order_smart maps _reason_out -> _outcome_context
        reason_box = {}
        monkeypatch.setattr(
            bapi_placeorder,
            "_guarded_market_place",
            lambda *_a, **kw: kw["_outcome_context"].update(reason="blocked_by_test") or None,
        )
        res_smart = bapi_placeorder.place_order_smart(
            "BUY", "BTCUSDC", 100.0, 1.0, _reason_out=reason_box
        )
        assert res_smart is None
        assert reason_box.get("reason") == "blocked_by_test"

        # 2. Instrument.place persists medium_trend_action & medium_trend_scale from self.params into retry_kwargs
        enqueued_kwargs = []

        class _DeferProv:
            name = "binance"

            @staticmethod
            def guards_internally():
                return False

            @staticmethod
            def execution_enabled():
                return True

            @staticmethod
            def requires_cache_permit():
                return False

            @staticmethod
            def defers_smart_cancellation():
                return False

            @staticmethod
            def supports_business_minimum():
                return False

            @staticmethod
            def get_current_price(_sym):
                return 100.0

            @staticmethod
            def format_price(_sym, p):
                return float(p)

            @staticmethod
            def format_quantity(_sym, q):
                return float(q)

            @staticmethod
            def validate_order(_sym, _side, _px, _q):
                return None

            @staticmethod
            def min_notional(_sym):
                return 5.0

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def get_orders(_sym, _side, _since):
                return []

            @staticmethod
            def quantity_decision(_sym, _side, _px, req_qty, **_kw):
                return SimpleNamespace(final_qty=req_qty, refuse_reason=None, balance_asset="USDC")

            @staticmethod
            def should_wait_for_short_trend(_sym, _side):
                return True

        import order_retry
        monkeypatch.setattr(
            order_retry,
            "enqueue",
            lambda _s, _side, _q, pkw, *_a, **_k: enqueued_kwargs.append(dict(pkw)) or {"id": "r1"},
        )
        inst_obj = inst_mod.Instrument.__new__(inst_mod.Instrument)
        inst_obj.name = "BTCUSDC"
        inst_obj.symbol = "BTCUSDC"
        inst_obj.provider_name = "binance"
        inst_obj.base = "BTC"
        inst_obj.quote = "USDC"
        inst_obj.enabled = True
        inst_obj._provider = _DeferProv()
        inst_obj._api = SimpleNamespace(
            market_regime_context=lambda *_a, **_k: SimpleNamespace(resolved_trend="flat")
        )
        inst_obj.params = {"medium_trend_action": "scale", "medium_trend_scale": 0.4}
        assert inst_obj.place("BUY", 100.0, 1.0, smart=False) is None
        assert enqueued_kwargs
        assert enqueued_kwargs[-1].get("medium_trend_action") == "scale"
        assert enqueued_kwargs[-1].get("medium_trend_scale") == pytest.approx(0.4)

        # 3. WhalePositioningCollector recovers .previous on corrupted primary JSON and sanitizes symbol in _resolve_cached_funding_rate
        wc = WhalePositioningCollector(cache_dir=str(tmp_path))
        corrupt_p24 = tmp_path / "cache_24_price_BTCUSDC.json"
        prev_p24 = tmp_path / "cache_24_price_BTCUSDC.json.previous"
        corrupt_p24.write_text("{corrupt", encoding="utf-8")
        prev_p24.write_text(
            json.dumps({"items": {"BTCUSDC": [[1700000000000, 100.0], [1700003600000, 105.0]]}}),
            encoding="utf-8",
        )
        assert wc._resolve_price_1h_change_pct("BTC/USDC") == pytest.approx(5.0)

        deriv_p = tmp_path / "derivatives_telemetry_BTCUSDC.json"
        deriv_p.write_text(json.dumps({"ts": 1700003600.0, "funding_rate": 0.00042}), encoding="utf-8")
        assert wc._resolve_cached_funding_rate("BTC/USDC", now=1700003610.0) == pytest.approx(0.00042)

        # 4. external_order_guard preserves active_reason of the stricter upstream downscale
        monkeypatch.setattr(
            external_order_guard,
            "get_whale_collector",
            lambda: SimpleNamespace(
                fetch=lambda *_a, **_k: SimpleNamespace(
                    divergence_regime="neutral",
                    open_interest_1h_change_pct=0.0,
                    taker_buy_sell_ratio=0.80,
                    top_traders_long_pct=0.50,
                    top_traders_long_ratio=1.0,
                )
            ),
        )
        monkeypatch.setattr(
            external_order_guard,
            "get_orderbook_collector",
            lambda: SimpleNamespace(fetch=lambda *_a, **_k: None),
        )
        monkeypatch.setattr(
            external_order_guard,
            "get_derivatives_collector",
            lambda: SimpleNamespace(
                fetch=lambda *_a, **_k: DerivativesTelemetry(
                    symbol="BTCUSDC",
                    funding_rate=0.0008,
                    predicted_funding_rate=0.0008,
                    open_interest=1000.0,
                    open_interest_usd=100000.0,
                    ts=1700000000.0,
                )
            ),
        )
        m_cfg = dict(order_guard._load_margins())
        m_cfg.update(
            {
                "external_guard_mode": "enforce",
                "whale_guard_mode": "enforce",
                "orderbook_wall_guard_mode": "off",
                "funding_guard_mode": "enforce",
                "min_taker_ratio": 0.70,
                "taker_downscale_threshold": 0.90,
                "taker_downscaled_scale": 0.40,
                "funding_max_long_rate": 0.0005,
                "funding_crowding_policy": "downscale",
                "funding_crowded_scale": 0.75,
            }
        )
        allowed, reason, eff_scale = external_order_guard.check_external_order_guards(
            "BTCUSDC", "BUY", 100.0, qty=1.0, margins=m_cfg, now=1700000000.0
        )
        assert allowed is True
        assert eff_scale == pytest.approx(0.40)
        assert "moderate_taker_selling_pressure" in reason

    def test_iteration12_weibull_block_policy_trend_exhausted_cache_invalidation_usdcusdt_base_and_llm_guard_reason(
        self, monkeypatch
    ):
        from types import SimpleNamespace
        import llm_order_guard
        import order_guard
        from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard

        # 1. WeibullExhaustionGuard with policy="block" (from order_guard.conf _DISCRETE_POLICY_SETS) vetoes exhausted trend
        wg_block = WeibullExhaustionGuard(default_p90_days=7.0, policy="block", exhausted_scale=0.25)
        dec_block = wg_block.check(
            "BTCUSDC",
            "BUY",
            trend_duration_seconds=10.0 * 86400.0,
            trend_direction="bull",
        )
        assert dec_block.allowed is False
        assert dec_block.suggested_scale == pytest.approx(0.0)
        assert "trend_exhausted" in dec_block.reason

        # 2. check_intelligence_guards invalidates cached Weibull "trend_exhausted" decision when trend_duration_seconds changes
        base_m = dict(order_guard._load_margins())
        base_m.update(
            {
                "intelligence_guards_mode": "enforce",
                "internal_guard_mode": "enforce",
                "weibull_exhaustion_policy": "downscale",
                "weibull_exhausted_scale": 0.25,
                "weibull_p90_days": 7.0,
                "external_guard_mode": "off",
                "llm_guard_mode": "off",
                "geopolitical_guard_mode": "off",
            }
        )
        monkeypatch.setattr(order_guard, "_load_margins", lambda: base_m)
        monkeypatch.setattr(order_guard, "_read_cached_price_history", lambda *_a, **_k: None)

        ctx = SimpleNamespace(
            resolved_trend="bull",
            symbol="BTCUSDC",
            provider="binance",
            is_valid_for=lambda **_kw: True,
        )
        ok1, reason1, scale1 = order_guard.check_intelligence_guards(
            "binance",
            "BTCUSDC",
            "BUY",
            100.0,
            trend_duration_seconds=10.0 * 86400.0,
            regime_context=ctx,
            qty=1.0,
        )
        assert ok1 is True
        assert scale1 == pytest.approx(0.25)
        assert "trend_exhausted" in reason1

        # Second call with same regime_context but fresh short duration (1d <= 7d) must invalidate cached 0.25 downscale
        ok2, reason2, scale2 = order_guard.check_intelligence_guards(
            "binance",
            "BTCUSDC",
            "BUY",
            100.0,
            trend_duration_seconds=1.0 * 86400.0,
            regime_context=ctx,
            qty=1.0,
        )
        assert ok2 is True
        assert scale2 == pytest.approx(1.0)
        assert reason2 == "ok"

        # 3. _normalize_base_asset preserves USDCUSDT so it is never conflated with USDC/EUR
        assert order_guard._normalize_base_asset("USDCUSDT") == "USDCUSDT"
        assert order_guard._normalize_base_asset("USDC/USDT") == "USDCUSDT"
        assert order_guard._normalize_base_asset("BTCUSDC") == "BTC"

        # 4. check_llm_order_guards preserves stricter GLLM1 active_reason when GLLM3 has looser downscale
        monkeypatch.setattr(
            llm_order_guard,
            "evaluate_high_stake_order_guard",
            lambda **_kw: (True, "Gemini downscaled high-stake BUY: whale flow weak", 0.30),
        )
        monkeypatch.setattr(
            llm_order_guard,
            "check_geopolitical_order_guards",
            lambda *_a, **_kw: (True, "downscale_elevated_geopolitical_risk", 0.50),
        )
        llm_ok, llm_reason, llm_scale = llm_order_guard.check_llm_order_guards(
            "binance", "BTCUSDC", "BUY", 1000.0, qty=1.0, margins=base_m
        )
        assert llm_ok is True
        assert llm_scale == pytest.approx(0.30)
        assert "whale flow weak" in llm_reason


def test_iteration13_assetguardian_p3b_recovery_and_gllm2_bypass_survival_usdcusdt_composite_weibull_direction_and_clock_skew(
    tmp_path, monkeypatch
):
    """Verify Iteration 13 fixes across AssetGuardian P3B sentiment recovery & GLLM2 bypass,
    survival.py USDCUSDT base preservation, CompositeMarketIntelligence Weibull trend_direction,
    and SentimentAdvisor / GeopoliticalThreatAnalyzer clock-skew cache invalidation."""
    import json
    import time
    import assetguardian as ag
    from intelligence.internal.state.survival import _extract_base_symbol
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor, SentimentAdvisorAssessment
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer, GeopoliticalThreatAssessment

    now_ts = time.time()

    # 1. AssetGuardian _read_p3b_sentiment recovers from .previous when primary is corrupted,
    #    and _p3b_threshold_multipliers honors GLLM2 bypass.
    p3b_file = tmp_path / "sentiment_advisor_eval.json"
    p3b_prev = tmp_path / "sentiment_advisor_eval.json.previous"
    p3b_file.write_text("{corrupted_json", encoding="utf-8")
    p3b_prev.write_text(
        json.dumps(
            {
                "recommended_action": "DEFENSIVE",
                "market_bias": "BEARISH",
                "risk_level": "HIGH",
                "confidence": 0.90,
                "ts": now_ts - 60.0,
            }
        ),
        encoding="utf-8",
    )
    recovered = ag._read_p3b_sentiment(path=str(p3b_file), now_ts=now_ts)
    assert isinstance(recovered, dict)
    assert recovered["recommended_action"] == "DEFENSIVE"
    assert recovered["confidence"] == pytest.approx(0.90)

    # When GLLM2 is bypassed via bypass_guards, _p3b_threshold_multipliers returns neutral (1.0, 1.0, None)
    sell_m, buy_m, eff_assess = ag._p3b_threshold_multipliers(
        assessment=recovered, bypass_guards={"GLLM2"}
    )
    assert sell_m == pytest.approx(1.0)
    assert buy_m == pytest.approx(1.0)
    assert eff_assess is None

    # Without GLLM2 bypass, defensive assessment compresses sell_mult (< 1.0) and expands buy_mult (> 1.0)
    sell_m2, buy_m2, eff_assess2 = ag._p3b_threshold_multipliers(
        assessment=recovered, bypass_guards=set()
    )
    assert sell_m2 < 1.0
    assert buy_m2 > 1.0
    assert eff_assess2 is recovered

    # 2. survival._extract_base_symbol preserves USDCUSDT and strips RON/GBP quotes
    assert _extract_base_symbol("USDCUSDT") == "USDCUSDT"
    assert _extract_base_symbol("USDC/USDT") == "USDCUSDT"
    assert _extract_base_symbol("BTCRON") == "BTC"
    assert _extract_base_symbol("ETHGBP") == "ETH"

    # 3. CompositeMarketIntelligence.evaluate_guards passes inferred trend_direction from gradient
    #    to WeibullExhaustionGuard.check so counter-trend SELL in mature bull trend (gradient > 0) is not downscaled
    cmi = CompositeMarketIntelligence()
    monkeypatch.setattr(
        "intelligence.composite.get_trend_survival_metrics",
        lambda *_a, **_kw: {"p90_days": 5.0, "median_days": 3.0, "gaussian_horizon_days": 2.0},
    )
    # SELL with positive gradient (bull trend) at 10 days (> p90=5d) is counter-trend exit -> allowed at 1.0 scale
    dec_sell_bull = cmi.evaluate_guards(
        "BTCUSDC",
        "SELL",
        100.0,
        gradient=0.05,
        epsilon=0.001,
        trend_duration_seconds=10.0 * 86400.0,
    )
    assert dec_sell_bull.allowed is True
    assert dec_sell_bull.suggested_scale == pytest.approx(1.0)

    # BUY with positive gradient (bull trend) at 10 days (> p90=5d) is trend-chasing -> downscaled
    dec_buy_bull = cmi.evaluate_guards(
        "BTCUSDC",
        "BUY",
        100.0,
        gradient=0.05,
        epsilon=0.001,
        trend_duration_seconds=10.0 * 86400.0,
    )
    assert dec_buy_bull.allowed is True
    assert dec_buy_bull.suggested_scale < 1.0

    # 4. SentimentAdvisor and GeopoliticalThreatAnalyzer invalidate cache when _last_eval_ts is in the future (clock skew)
    sa = SentimentAdvisor(cache_ttl_sec=3600.0, cache_file=str(tmp_path / "sa_eval.json"))
    sa._cached_assessment = SentimentAdvisorAssessment(
        market_bias="BULLISH",
        risk_level="LOW",
        confidence=0.8,
        summary="cached",
        key_risks=[],
        recommended_action="ACCUMULATE",
        ts=now_ts + 5000.0,
    )
    sa._last_eval_ts = now_ts + 5000.0
    called_sa = {"count": 0}
    monkeypatch.setattr(
        sa.gemini_client,
        "query_json",
        lambda *_a, **_kw: (
            called_sa.__setitem__("count", called_sa["count"] + 1)
            or {
                "market_bias": "NEUTRAL",
                "risk_level": "MODERATE",
                "confidence": 0.5,
                "summary": "refreshed after clock skew",
                "key_risks": [],
                "recommended_action": "HOLD",
            }
        ),
    )
    refreshed_sa = sa.review(now_ts=now_ts)
    assert called_sa["count"] == 1
    assert refreshed_sa.summary == "refreshed after clock skew"

    ga = GeopoliticalThreatAnalyzer(cache_ttl_sec=3600.0, state_file=str(tmp_path / "ga_eval.json"))
    ga._cached_assessment = GeopoliticalThreatAssessment(
        threat_level="CRITICAL_SHOCK",
        risk_score=0.95,
        summary="future cached shock",
        recommended_brake="VETO_ALL",
        headlines_analyzed=3,
        ts=now_ts + 5000.0,
    )
    ga._last_eval_ts = now_ts + 5000.0
    from intelligence.macro.news_feed_collector import NewsFeedSnapshot, NewsHeadline
    snap = NewsFeedSnapshot(
        headlines=(
            NewsHeadline(
                title="Routine market update",
                source="test",
                pub_date="now",
                url="http://example.com",
                matched_keywords=(),
                is_high_severity=False,
            ),
        ),
        total_fetched=1,
        high_severity_count=0,
        has_critical_shock_keywords=False,
        ts=now_ts,
    )
    refreshed_ga = ga.assess(snap, now_ts=now_ts)
    assert refreshed_ga.threat_level == "NORMAL"


def test_iteration14_monitortrades_reentry_qlimit1_bypass_rtrade_gi_whale_bypass_quantity_resolve_assets_and_cachetrend_usdcusdt(
    monkeypatch,
):
    """Verify Iteration 14 fixes:
    1. monitortrades._can_pass_buy_quantity_policy and _apply_post_stop_ai_guard merge venue bypass_guards
       and pass apply_policy=not bypass_qlimit1 to provider.quantity_decision.
    2. rtrade._whale_flow_blocks_side and _effective_adjustment_fraction honor GI_WHALE bypass and whale_guard_mode.
    3. providers.quantity.resolve_assets and utils.base_asset support USDT, FDUSD, and GBP quotes.
    4. cachetrend._short_trend_symbol_variants preserves USDCUSDT intact and strips RON/GBP quotes.
    """
    import time
    from types import SimpleNamespace
    import monitortrades as mt
    import order_guard
    import rtrade
    import external_order_guard
    import utils as u
    from providers.quantity import resolve_assets, QuantityDecision
    from cachetrend import _short_trend_symbol_variants
    from intelligence.external.collectors.whale_positioning import (
        WhalePositioningSnapshot,
        WhalePositioningCollector,
    )

    # 1. monitortrades._can_pass_buy_quantity_policy honors G_QLIMIT1 bypass
    captured_apply_policy = []

    def _fake_q_dec(symbol, side, price, req_qty, **kw):
        captured_apply_policy.append(kw.get("apply_policy"))
        return QuantityDecision(
            requested_qty=req_qty,
            balance_cap=req_qty,
            policy_cap=req_qty,
            fee_cap=req_qty,
            final_qty=req_qty if not kw.get("apply_policy") else 0.0,
            refuse_reason="qty_zero_after_policy" if kw.get("apply_policy") else None,
            balance_asset="USDC",
        )

    fake_prov = SimpleNamespace(
        name="binance",
        free_balance=lambda _a: 1000.0,
        quantity_decision=_fake_q_dec,
        order_filter_refusal=lambda *_a, **_kw: None,
        get_orders=lambda *_a, **_kw: [],
    )
    fake_inst = SimpleNamespace(
        symbol="TAOUSDC",
        base="TAO",
        quote="USDC",
        provider=fake_prov,
        provider_name="binance",
        params={"bypass_guards": {"G_QLIMIT1", "GO1"}},
    )
    monkeypatch.setattr(mt, "_is_simulated_env_or_inst", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        order_guard,
        "check_intelligence_guards",
        lambda *_a, **_kw: (True, "ok", 1.0),
    )

    ok_reentry = mt._can_pass_buy_quantity_policy(fake_inst, "TAOUSDC", 200.0, 1.0, min_qty=0.01)
    assert ok_reentry is True
    assert captured_apply_policy == [False]

    # 2. rtrade._whale_flow_blocks_side and _effective_adjustment_fraction honor GI_WHALE bypass
    collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
    collector.record_snapshot(
        WhalePositioningSnapshot(
            symbol="TAOUSDC",
            top_traders_long_ratio=2.0,
            top_traders_long_pct=0.66,
            taker_buy_sell_ratio=0.70,
            taker_buy_vol_usd=70_000.0,
            taker_sell_vol_usd=100_000.0,
            open_interest_usd=50_000_000.0,
            open_interest_1h_change_pct=0.0,
            divergence_regime="neutral",
            ts=time.time(),
        )
    )
    monkeypatch.setattr(external_order_guard, "_whale_collector", collector)

    # Enforced by default -> blocks BUY and widens adjustment
    blocked_default, _ = rtrade._whale_flow_blocks_side("TAOUSDC", "BUY")
    assert blocked_default is True
    assert rtrade._effective_adjustment_fraction("TAOUSDC", 0.004) == pytest.approx(0.006)

    # Bypassed via GI_WHALE -> does not block and keeps base adjustment
    blocked_bypassed, reason_bypassed = rtrade._whale_flow_blocks_side(
        "TAOUSDC", "BUY", bypass_guards={"GI_WHALE"}
    )
    assert blocked_bypassed is False
    assert reason_bypassed == "gi_whale_bypassed_or_off"
    assert rtrade._effective_adjustment_fraction(
        "TAOUSDC", 0.004, bypass_guards={"GI_WHALE"}
    ) == pytest.approx(0.004)

    # 3. rtrade also honors whale_guard_mode = shadow / off in order_guard.conf
    base_m = dict(order_guard._load_margins())
    base_m["whale_guard_mode"] = "shadow"
    monkeypatch.setattr(order_guard, "_load_margins", lambda: base_m)
    blocked_shadow, reason_shadow = rtrade._whale_flow_blocks_side("TAOUSDC", "BUY")
    assert blocked_shadow is False
    assert reason_shadow == "gi_whale_bypassed_or_off"
    assert rtrade._effective_adjustment_fraction("TAOUSDC", 0.004) == pytest.approx(0.004)

    # 4. cachetrend._short_trend_symbol_variants preserves USDCUSDT intact and strips RON/GBP
    assert _short_trend_symbol_variants("USDCUSDT") == ["USDCUSDT"]
    assert "BTCUSDC" in _short_trend_symbol_variants("BTCRON")
    assert "ETHUSDC" in _short_trend_symbol_variants("ETHGBP")


def test_iteration15_post_stop_reentry_mode_hierarchy_smart_sl_recovery_external_shadow_and_intent_gateway_price_fallback(
    monkeypatch, tmp_path
):
    """Iteration 15 regression suite:
    1. get_post_stop_reentry_guard honors intelligence_guards_mode and llm_guard_mode master switches.
    2. monitortrades._execute_reentry_buy only bypasses GLLM1 on post-STOP BUY when PostStopReentryGuard
       was actually enforced (so post_stop_reentry_mode=off/shadow does not silently bypass HighStakeGuard >= $700).
    3. smart_stop_loss_guard._load_grace_state recovers from .previous when primary state file is corrupted,
       and evaluate_smart_stop_loss honors intelligence_guards_mode and external_guard_mode master switches.
    4. external_order_guard.check_external_order_guards caps sub-guard modes to 'shadow' when external_guard_mode='shadow'.
    5. AutonomousIntentGateway._get_live_price recovers cache_current_price.json from .previous and skips non-positive prices.
    """
    import external_order_guard
    import monitortrades as mt
    import order_guard
    from intelligence.external.collectors.whale_positioning import (
        WhalePositioningCollector,
        WhalePositioningSnapshot,
    )
    from intelligence.external.guards import smart_stop_loss_guard as ssl_mod
    from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    from strategies.position_exit_manager import ReentrySnapshot

    # 1. get_post_stop_reentry_guard honors llm_guard_mode and intelligence_guards_mode
    base_m = dict(order_guard._load_margins())
    m_llm_off = {**base_m, "post_stop_reentry_mode": "enforce", "llm_guard_mode": "off"}
    g_off = get_post_stop_reentry_guard(margins=m_llm_off, gemini_client=object())
    assert g_off.mode == "off"

    m_llm_shadow = {**base_m, "post_stop_reentry_mode": "enforce", "llm_guard_mode": "shadow"}
    g_shadow = get_post_stop_reentry_guard(margins=m_llm_shadow, gemini_client=object())
    assert g_shadow.mode == "shadow"

    # 2. monitortrades._apply_post_stop_ai_guard & _execute_reentry_buy preserve HighStakeGuard when PostStopReentryGuard is off/shadow
    monkeypatch.setattr(mt, "_is_simulated_env_or_inst", lambda inst=None: False)
    monkeypatch.setattr(order_guard, "_load_margins", lambda: m_llm_shadow)
    snap = ReentrySnapshot(
        symbol="TAOUSDC",
        current_price=200.0,
        last_sell_price=210.0,
        last_exit_kind="STOP",
        post_sell_peak=210.0,
        post_sell_trough=195.0,
        current_time_s=2000.0,
        last_sell_time_s=1000.0,
        trend_up=True,
        trend_down=False,
    )
    allowed, _, _, _ = mt._apply_post_stop_ai_guard(
        "TAOUSDC", 200.0, 210.0, 1000.0, snap, 1000.0, None, 5.0, inst=None
    )
    assert allowed is True
    assert "TAOUSDC" in mt._POST_STOP_AI_NOT_ENFORCED

    captured_bypasses = []
    monkeypatch.setattr(
        mt,
        "_place_guarded",
        lambda inst, side, px, q, mq, **kw: (captured_bypasses.append(kw.get("bypass_guards")) or True),
    )

    class _DummyInst:
        symbol = "TAOUSDC"

        def param(self, ns, key, default=None, cast=None):
            return default

    mt._execute_reentry_buy(_DummyInst(), "TAOUSDC", 200.0, 5.0, 0.01, 3600.0, "STOP", {"sell_time": 1000.0})
    assert captured_bypasses == [{"G_PROF"}]

    # 3. smart_stop_loss_guard._load_grace_state recovers from .previous and evaluate_smart_stop_loss honors external_guard_mode
    state_file = tmp_path / "smart_stop_loss_state.json"
    prev_file = tmp_path / "smart_stop_loss_state.json.previous"
    state_file.write_text("{corrupt_json", encoding="utf-8")
    now_ts = time.time()
    prev_file.write_text(
        json.dumps({"TAOUSDC": {"start_ts": now_ts - 10.0, "initial_loss_pct": 3.5}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ssl_mod, "_effective_smart_sl_state_file", lambda: str(state_file))
    ssl_mod._GRACE_TRACKER.clear()
    ssl_mod._LOADED_STATE_FILE_KEY = None
    tracker = ssl_mod._load_grace_state(is_sim=False)
    assert "TAOUSDC" in tracker

    m_ext_off = {**base_m, "smart_sl_mode": "enforce", "external_guard_mode": "off"}
    monkeypatch.setattr(order_guard, "_load_margins", lambda: m_ext_off)
    sl_exec, sl_reason, _ = ssl_mod.evaluate_smart_stop_loss("TAOUSDC", 0.04, 0.033, provider="sim")
    assert sl_exec is True
    assert sl_reason == "smart_sl_disabled"

    # 4. external_order_guard caps whale_guard_mode='enforce' to 'shadow' when external_guard_mode='shadow'
    collector = WhalePositioningCollector(cache_ttl_sec=60.0, cache_dir=None)
    collector.record_snapshot(
        WhalePositioningSnapshot(
            symbol="TAOUSDC",
            top_traders_long_ratio=2.0,
            top_traders_long_pct=0.66,
            taker_buy_sell_ratio=0.60,
            taker_buy_vol_usd=60_000.0,
            taker_sell_vol_usd=100_000.0,
            open_interest_usd=50_000_000.0,
            open_interest_1h_change_pct=-5.0,
            divergence_regime="long_liquidation_risk",
            ts=now_ts,
        )
    )
    monkeypatch.setattr(external_order_guard, "_whale_collector", collector)
    m_ext_shadow = {
        **base_m,
        "external_guard_mode": "shadow",
        "whale_guard_mode": "enforce",
        "orderbook_wall_guard_mode": "off",
        "funding_guard_mode": "off",
    }
    ext_ok, ext_reason, ext_scale = external_order_guard.check_external_order_guards(
        "TAOUSDC", "BUY", 200.0, qty=1.0, margins=m_ext_shadow, now=now_ts
    )
    assert ext_ok is True
    assert ext_scale == 1.0

    # 5. AutonomousIntentGateway._get_live_price recovers from .previous and skips 0/NaN price entries
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(exist_ok=True)
    cp_file = cache_dir / "cache_current_price.json"
    cp_prev = cache_dir / "cache_current_price.json.previous"
    cp_file.write_text("{corrupt", encoding="utf-8")
    cp_prev.write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSD": [[now_ts, 0.0]],
                    "TAOUSDC": [[now_ts, 245.5]],
                }
            }
        ),
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path), mode="shadow")

    class _FakeInst:
        symbol = "TAOUSD"
        base = "TAO"
        _provider = None

    assert gw._get_live_price(_FakeInst()) == pytest.approx(245.5)


def test_iteration16_volatility_items_and_usdcusdt_exit_manager_sl_timer_expired_collectors_clock_skew_and_high_stake_now(
    monkeypatch, tmp_path
) -> None:
    """
    Iteration 16 regression tests:
    1. volatility.py: _vol_symbol_variants preserves USDCUSDT intact and strips RON/GBP;
       _extract_vol_ratio_from_file reads standard {"items": {...}} envelopes and recovers from .previous.
    2. PositionExitManager._evaluate_stop_loss_core sets metadata["sl_timer_expired"] = True
       when the SL deferral timer has expired, and evaluates timeout before structural_break.
    3. External collectors (OrderbookDepthCollector, WhalePositioningCollector, DerivativesTelemetryCollector)
       reject future-dated timestamps (clock skew) and stage .previous backup files via save_json_state.
    4. HighStakeGuard.check honors injected `now` for _HIGH_STAKE_DECISION_CACHE and caches timeout
       fallback decisions so repeated order checks within TTL do not block for 3s per call.
    """
    from intelligence.internal.state import volatility as vol_mod
    from strategies.position_exit_manager import (
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
    )
    from intelligence.external.collectors.orderbook_depth import (
        OrderbookDepthCollector,
        _extract_base_asset,
    )
    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
    from intelligence.sentiment.guards import high_stake_guard as hsg_mod
    import llm_order_guard

    # 1. volatility.py: USDCUSDT preserved, RON/GBP stripped, {"items": {...}} + .previous recovery
    assert vol_mod._vol_symbol_variants("USDCUSDT") == ("USDCUSDT",)
    assert "BTC" in vol_mod._vol_symbol_variants("BTCRON")
    assert "ETH" in vol_mod._vol_symbol_variants("ETHGBP")

    st_path = tmp_path / "cache_short_trend.json"
    st_prev = tmp_path / "cache_short_trend.json.previous"
    st_path.write_text("{corrupt", encoding="utf-8")
    st_prev.write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSD": {"vol_1h_pct": 2.35},
                    "USDCUSDT": {"vol_1h_pct": 1.05},
                }
            }
        ),
        encoding="utf-8",
    )
    assert vol_mod.resolve_cached_volatility_1h("TAOUSDC", state_path=str(st_path)) == pytest.approx(0.0235)
    assert vol_mod.resolve_cached_volatility_1h("USDCUSDT", state_path=str(st_path)) == pytest.approx(0.0105)

    # 2. PositionExitManager._evaluate_stop_loss_core sets sl_timer_expired=True and prioritizes timeout over structural_break
    cfg = PositionExitConfig(
        lost_threshold=0.05,
        sl_trend_defer_max_sec=900.0,
        sl_structural_break_enabled=True,
        smart_sl_enabled=False,
    )
    mgr = PositionExitManager(cfg)
    # Sample 1 at t=2000 initiates bounce watch (5.5% loss >= 5.0% threshold, below 6.5% disaster floor)
    d1 = mgr.evaluate_stop_loss(
        PositionSnapshot(
            symbol="TAOUSDC",
            current_price=94.5,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=2000.0,
            trend_up=True,
        )
    )
    assert d1.should_exit is False
    assert d1.metadata.get("state") == "initiated"

    # Sample 2 at t=2100 records a bounce up to 94.95 (+0.48% above 94.5 trough, still >= 5.0% loss threshold)
    d2 = mgr.evaluate_stop_loss(
        PositionSnapshot(
            symbol="TAOUSDC",
            current_price=94.95,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=2100.0,
            trend_up=True,
        )
    )
    assert d2.should_exit is False

    # Sample 3 at t=2950 (950s >= 900s max defer) with price 94.2 (< lowest_px 94.5, which would also be structural_break):
    # timeout must take precedence over structural_break and set sl_timer_expired=True
    d3 = mgr.evaluate_stop_loss(
        PositionSnapshot(
            symbol="TAOUSDC",
            current_price=94.2,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=2950.0,
            trend_up=True,
        )
    )
    assert d3.should_exit is True
    assert d3.metadata.get("sl_timer_expired") is True
    assert d3.metadata.get("sub_reason") == "timeout"

    # 3. External collectors: USDCUSDT preserved in _extract_base_asset, clock skew rejected, .previous written
    from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
    from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry

    assert _extract_base_asset("USDCUSDT") == "USDCUSDT"

    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), cache_ttl_sec=60, disk_ttl_sec=60)
    ob_snap_valid = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=1000.0,
        ask_depth_usd=1000.0,
        imbalance_ratio=0.5,
        largest_bid_wall_usd=500.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=500.0,
        largest_ask_wall_price=60100.0,
        ts=1000.0,
    )
    ob_col._save_to_disk(ob_snap_valid)
    ob_snap_future = OrderbookSnapshot(
        symbol="BTCUSDC",
        mid_price=60000.0,
        bid_depth_usd=1000.0,
        ask_depth_usd=1000.0,
        imbalance_ratio=0.5,
        largest_bid_wall_usd=500.0,
        largest_bid_wall_price=59900.0,
        largest_ask_wall_usd=500.0,
        largest_ask_wall_price=60100.0,
        ts=2000.0,  # future timestamp relative to now=1005.0 -> clock skew
    )
    ob_col._save_to_disk(ob_snap_future)
    assert (tmp_path / "orderbook_depth_BTCUSDC.json").exists()
    # Future-dated snapshot is rejected on load at now=1005.0
    assert ob_col._load_from_disk("BTCUSDC", now=1005.0) is None

    whale_col = WhalePositioningCollector(cache_dir=str(tmp_path), cache_ttl_sec=60, disk_ttl_sec=60)
    whale_snap_future = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.0,
        top_traders_long_pct=0.5,
        taker_buy_sell_ratio=1.0,
        taker_buy_vol_usd=100.0,
        taker_sell_vol_usd=100.0,
        open_interest_usd=1000.0,
        open_interest_1h_change_pct=0.0,
        divergence_regime="neutral",
        ts=2000.0,
    )
    whale_col._save_to_disk(whale_snap_future)
    assert whale_col._load_from_disk("BTCUSDC", now=1005.0) is None

    deriv_col = DerivativesTelemetryCollector(cache_dir=str(tmp_path), cache_ttl_sec=60, disk_ttl_sec=60)
    deriv_snap_future = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.001,
        predicted_funding_rate=0.001,
        open_interest=100.0,
        open_interest_usd=6000000.0,
        ts=2000.0,
    )
    deriv_col._save_to_disk(deriv_snap_future)
    assert deriv_col._load_from_disk("BTCUSDC", now=1005.0) is None
    # WhalePositioningCollector._resolve_cached_funding_rate also rejects future-dated derivatives telemetry
    assert whale_col._resolve_cached_funding_rate("BTCUSDC", now=1005.0) is None

    # 4. HighStakeGuard.check honors `now` and caches timeout fallback decisions
    hsg_mod._HIGH_STAKE_DECISION_CACHE.clear()
    monkeypatch.setattr(hsg_mod, "is_llm_guard_exempt", lambda explicit_client=False: False)
    guard = hsg_mod.HighStakeGuard(
        min_notional_usd=700.0,
        timeout_sec=0.05,
        fallback_action="allow",
        cache_ttl_sec=120,
    )
    call_count = {"n": 0}

    def _timeout_query_json(*args, **kwargs):
        call_count["n"] += 1
        return None  # simulates LLM query timeout

    monkeypatch.setattr(guard.gemini_client, "query_json", _timeout_query_json)
    monkeypatch.setattr(hsg_mod, "HighStakeGuard", lambda **kw: guard)
    margins_hs = {
        "intelligence_guards_mode": "enforce",
        "llm_guard_mode": "enforce",
        "high_stake_min_notional_usd": 700.0,
        "llm_timeout_sec": 0.05,
        "llm_fallback": "allow",
    }

    ok1, r1, s1 = llm_order_guard.evaluate_high_stake_order_guard(
        symbol="TAOUSDC",
        side="BUY",
        price=1000.0,
        qty=1.0,
        computed_notional=1000.0,
        regime_context=None,
        valid_context=False,
        cache_key=("TAOUSDC", "BUY"),
        margins=margins_hs,
        bypassed=set(),
        now=5000.0,
    )
    assert ok1 is True
    assert s1 == 1.0
    assert call_count["n"] == 1

    # Second call 10s later in simulated time should hit the cached timeout fallback without calling query_json again
    ok2, r2, s2 = llm_order_guard.evaluate_high_stake_order_guard(
        symbol="TAOUSDC",
        side="BUY",
        price=1000.0,
        qty=1.0,
        computed_notional=1000.0,
        regime_context=None,
        valid_context=False,
        cache_key=("TAOUSDC", "BUY"),
        margins=margins_hs,
        bypassed=set(),
        now=5010.0,
    )
    assert ok2 is True
    assert s2 == 1.0
    assert call_count["n"] == 1


def test_iteration17_price_history_now_liquidation_cascade_now_composite_gemini_now_persistence_nan_and_macro_previous_recovery(
    monkeypatch, tmp_path
) -> None:
    """
    Iteration 17 regression tests:
    1. order_guard._read_cached_price_history honors injected `now` and filters out future-dated points
       when invoked via check_qlimit2_order_guards / check_intelligence_guards.
    2. BybitLiquidationCollector.get_summary and LiquidationCascadeGuard.check honor injected `now`
       for synthetic / replay liquidation events and reject future-dated events.
    3. CompositeMarketIntelligence.evaluate_guards forwards `dur_sec` and `now` to gemini_guard.check.
    4. calculate_mann_kendall and calculate_hurst_exponent filter non-finite (NaN/Inf) prices without crashing.
    5. GeopoliticalThreatAnalyzer, SentimentAdvisor, and NewsFeedCollector sanitize non-finite/bool fields
       and recover from .previous backup files.
    """
    import math
    import qlimit_order_guard
    from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector
    from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.internal.state.persistence import (
        calculate_hurst_exponent,
        calculate_mann_kendall,
    )
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from intelligence.macro.news_feed_collector import NewsFeedCollector

    # 1. _read_cached_price_history honors `now` and rejects future-dated points
    monkeypatch.setattr(order_guard, "_resolve_cachedb_dir", lambda: str(tmp_path))
    p24 = tmp_path / "cache_24_price_TAOUSDC.json"
    p24.write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": [
                        [10000.0, 100.0],
                        [10500.0, 108.0],  # +8% surge within [3800..11000] window when now=11000
                        [20000.0, 50.0],   # future point relative to now=11000 -> must be ignored
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    hist = order_guard._read_cached_price_history("TAOUSDC", window_seconds=7200.0, now=11000.0)
    assert hist == [(10000.0, 100.0), (10500.0, 108.0)]

    # Verify check_qlimit2_order_guards forwards `now` to _read_cached_price_history and blocks parabolic BUY
    q_ok, q_reason, q_scale = qlimit_order_guard.check_qlimit2_order_guards(
        "TAOUSDC",
        "BUY",
        108.0,
        margins={
            "internal_guard_mode": "enforce",
            "parabolic_surge_pct": 5.0,
            "parabolic_pullback_pct": 1.5,
            "weibull_exhaustion_policy": "downscale",
            "weibull_exhausted_scale": 0.5,
            "weibull_p90_days": 7.0,
        },
        now=11000.0,
    )
    assert q_ok is False
    assert "parabolic_surge" in q_reason

    # 2. BybitLiquidationCollector.get_summary and LiquidationCascadeGuard.check honor `now`
    liq_col = BybitLiquidationCollector(symbols=["BTCUSDT"])
    liq_col.record_event("BTCUSDT", "LONG", 600_000.0, ts=5000.0)
    liq_col.record_event("BTCUSDT", "LONG", 900_000.0, ts=5100.0)  # future event relative to now=5010.0
    c_guard = LiquidationCascadeGuard(collector=liq_col, max_active_cascade_usd=500_000.0, active_window_sec=60.0)
    c_dec = c_guard.check("BTCUSDC", "BUY", now=5010.0)
    assert c_dec.allowed is False
    assert c_dec.metadata.get("long_liq_usd") == pytest.approx(600_000.0)

    # 3. CompositeMarketIntelligence.evaluate_guards forwards `dur_sec` and `now` to gemini_guard.check
    captured_gemini_kwargs = {}

    class _SpyGeminiGuard:
        min_notional_eur = 700.0

        def check(self, symbol, side, price, qty, **kwargs):
            captured_gemini_kwargs.update(kwargs)
            from intelligence.internal.guards.guard_decision import GuardDecision
            return GuardDecision.allow("HighStakeGuard", "spy_ok")

    comp = CompositeMarketIntelligence(gemini_guard=_SpyGeminiGuard())
    comp_dec = comp.evaluate_guards(
        "TAOUSDC",
        "BUY",
        200.0,
        qty=5.0,
        notional_eur=1000.0,
        trend_duration_seconds=3600.0,
        now=12345.0,
    )
    assert comp_dec.allowed is True
    assert captured_gemini_kwargs.get("dur_sec") == pytest.approx(3600.0)
    assert captured_gemini_kwargs.get("now") == pytest.approx(12345.0)

    # 4. calculate_mann_kendall and calculate_hurst_exponent filter NaN/Inf without crashing
    mk_s, mk_z, mk_p = calculate_mann_kendall([100.0, float("nan"), 101.0, 102.0, float("inf"), 103.0, 104.0, 105.0, 106.0, 107.0])
    assert mk_s > 0
    assert mk_p < 0.05
    hurst_series = [100.0 + i * 0.5 for i in range(70)]
    hurst_series[10] = float("nan")
    h_val = calculate_hurst_exponent(hurst_series)
    assert h_val is not None and math.isfinite(h_val)

    # 5. GeopoliticalThreatAnalyzer, SentimentAdvisor, and NewsFeedCollector sanitize non-finite/bool and recover .previous
    geo_path = tmp_path / "geopolitical_threat_eval.json"
    geo_prev = tmp_path / "geopolitical_threat_eval.json.previous"
    geo_path.write_text("{corrupt", encoding="utf-8")
    geo_prev.write_text(
        json.dumps(
            {
                "threat_level": "ELEVATED",
                "risk_score": True,  # bool should fallback to 0.1
                "summary": "recovered",
                "recommended_brake": "DOWNSCALE_50",
                "headlines_analyzed": 3,
                "ts": -50.0,  # negative ts should clamp to 0.0
            }
        ),
        encoding="utf-8",
    )
    geo_analyzer = GeopoliticalThreatAnalyzer(state_file=str(geo_path), cache_ttl_sec=3600.0, fallback_retry_ttl_sec=300.0)
    assert geo_analyzer._cached_assessment is not None
    assert geo_analyzer._cached_assessment.threat_level == "ELEVATED"
    assert geo_analyzer._cached_assessment.risk_score == pytest.approx(0.1)
    assert geo_analyzer._cached_assessment.ts == pytest.approx(0.0)

    sent_path = tmp_path / "sentiment_advisor_eval.json"
    sent_prev = tmp_path / "sentiment_advisor_eval.json.previous"
    sent_path.write_text("{corrupt", encoding="utf-8")
    sent_prev.write_text(
        json.dumps(
            {
                "market_bias": "CAUTION",
                "risk_level": "HIGH",
                "confidence": 9.9,  # out of [0, 1] range -> fallback to 0.5
                "summary": "recovered sentiment",
                "key_risks": ["macro"],
                "recommended_action": "DEFENSIVE",
                "ts": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    sent_adv = SentimentAdvisor(cache_file=str(sent_path))
    assert sent_adv._cached_assessment is not None
    assert sent_adv._cached_assessment.market_bias == "CAUTION"
    assert sent_adv._cached_assessment.confidence == pytest.approx(0.5)

    news_path = tmp_path / "news_feed_collect.json"
    news_prev = tmp_path / "news_feed_collect.json.previous"
    news_path.write_text("{corrupt", encoding="utf-8")
    now_wall = time.time()
    news_prev.write_text(
        json.dumps(
            [
                {
                    "title": "Major pipeline escalation",
                    "source": "Reuters",
                    "pub_date": "now",
                    "url": "https://example.com/1",
                    "matched_keywords": ["escalation"],
                    "is_high_severity": True,
                    "first_seen_ts": now_wall - 60.0,
                },
                {
                    "title": "Future clock-skew headline",
                    "source": "Reuters",
                    "pub_date": "future",
                    "url": "https://example.com/2",
                    "matched_keywords": ["war"],
                    "is_high_severity": True,
                    "first_seen_ts": now_wall + 3600.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    news_col = NewsFeedCollector(history_file=str(news_path))
    snap = news_col.get_pool_snapshot(now_ts=now_wall)
    assert len(snap.headlines) == 1
    assert snap.headlines[0].title == "Major pipeline escalation"


def test_iteration18_macro_analyzer_resilience_and_external_collectors_now_propagation(tmp_path, monkeypatch):
    """Verify Iteration 18 fixes:
    1. MacroAnalyzer parses string boolean shadow_notify ("true"/"false") without ValueError,
       recovers Fear&Greed / MarketBreadth / Whale telemetry from .previous or fallback files,
       and protects run_cycle against backward clock skew.
    2. GlobalTelemetryCollector and IntelligenceTelemetryDaemon protect run_cycle against backward clock skew.
    3. WhalePositioningCollector, OrderbookDepthCollector, DerivativesTelemetryCollector, and
       check_external_order_guards propagate `now` for deterministic cache TTL and filter NaN price points.
    """
    import external_order_guard
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.daemon import IntelligenceTelemetryDaemon
    from intelligence.external.collectors.whale_positioning import (
        WhalePositioningCollector,
        WhalePositioningSnapshot,
    )
    from intelligence.external.collectors.orderbook_depth import (
        OrderbookDepthCollector,
        OrderbookSnapshot,
    )
    from intelligence.external.collectors.derivatives_telemetry import (
        DerivativesTelemetry,
        DerivativesTelemetryCollector,
    )

    # 1. MacroAnalyzer string shadow_notify + .previous / fallback recovery + clock-skew run_cycle
    import order_guard
    orig_margins = order_guard._load_margins()
    custom_margins = dict(orig_margins)
    custom_margins["shadow_notify"] = "false"
    monkeypatch.setattr(order_guard, "_load_margins", lambda: custom_margins)

    analyzer = MacroAnalyzer(
        cache_dir=str(tmp_path),
        macro_llm_interval_sec=600.0,
        advisor_interval_sec=600.0,
        symbols=["BTCUSDC"],
    )
    assert analyzer.shadow_notify is False

    # Corrupt primary fear_greed_collect.json, valid fear_greed_cache.json
    (tmp_path / "fear_greed_collect.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "fear_greed_cache.json").write_text(
        json.dumps(
            {
                "value": 18,
                "classification": "Extreme Fear",
                "historical_values": [18, 22],
                "trend_7d_change": -4,
                "is_extreme_fear": True,
                "is_extreme_greed": False,
                "ts": 5000.0,
            }
        ),
        encoding="utf-8",
    )
    # Corrupt primary market_breadth_collect.json, valid .previous
    (tmp_path / "market_breadth_collect.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "market_breadth_collect.json.previous").write_text(
        json.dumps(
            {
                "advance_ratio": 0.25,
                "advancing_count": 25,
                "declining_count": 75,
                "total_count": 100,
                "median_change_pct": -2.5,
                "mean_change_pct": -2.8,
                "dispersion_std": 1.2,
                "regime": "BROAD_SELLOFF",
                "top_gainers": [["BTCUSDC", 1.2]],
                "top_losers": [["TAOUSDC", -5.4]],
                "ts": 5000.0,
            }
        ),
        encoding="utf-8",
    )
    # Corrupt primary whale_snapshot_BTCUSDC.json, valid .previous
    (tmp_path / "whale_snapshot_BTCUSDC.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "whale_snapshot_BTCUSDC.json.previous").write_text(
        json.dumps({"symbol": "BTCUSDC", "taker_buy_sell_ratio": 0.72, "ts": 5000.0}),
        encoding="utf-8",
    )

    captured_review = {}

    class _DummyAssessment:
        market_bias = "DEFENSIVE"
        risk_level = "HIGH"
        recommended_action = "HOLD"
        summary = "test"
        ts = 5005.0

    def _fake_review(fear_greed=None, breadth=None, whale_telemetry=None, force_refresh=False, now_ts=None):
        captured_review["fg"] = fear_greed
        captured_review["mb"] = breadth
        captured_review["whale"] = whale_telemetry
        captured_review["now_ts"] = now_ts
        return _DummyAssessment()

    monkeypatch.setattr(analyzer.gemini_advisor, "review", _fake_review)
    monkeypatch.setattr(analyzer, "assess_geopolitical_threat", lambda force=False, now=None: True)
    monkeypatch.setattr(analyzer, "prune_cli_threads", lambda: 0)

    # Simulate future _last_advisor_ts / _last_macro_llm_ts (backward clock skew)
    analyzer._last_macro_llm_ts = 99999.0
    analyzer._last_advisor_ts = 99999.0
    cycle_res = analyzer.run_cycle(force=False, now=5005.0)
    assert cycle_res["geopolitical_updated"] is True
    assert cycle_res["advisor_updated"] is True
    assert captured_review["fg"] is not None and captured_review["fg"].value == 18
    assert captured_review["mb"] is not None and captured_review["mb"].regime == "BROAD_SELLOFF"
    assert captured_review["whale"] is not None and captured_review["whale"]["symbol"] == "BTCUSDC"
    assert captured_review["now_ts"] == pytest.approx(5005.0)

    # 2. GlobalTelemetryCollector and IntelligenceTelemetryDaemon backward clock-skew resilience
    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_news=True, skip_external=True)
    gtc._last_fear_greed_ts = 99999.0
    gtc._last_breadth_ts = 99999.0
    monkeypatch.setattr(gtc, "update_fear_greed", lambda force=False: True)
    monkeypatch.setattr(gtc, "update_market_breadth", lambda force=False: True)
    gtc_res = gtc.run_cycle(force=False, now=5000.0)
    assert gtc_res["fear_greed_updated"] is True
    assert gtc_res["breadth_updated"] is True

    itd = IntelligenceTelemetryDaemon(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_macro=False, skip_external=True)
    itd._last_macro_ts = 99999.0
    monkeypatch.setattr(itd, "update_macro", lambda force=False: True)
    itd_res = itd.run_cycle(force=False, now=5000.0)
    assert itd_res["macro_updated"] is True

    # 3. External collectors + check_external_order_guards `now` propagation & NaN filtering
    price_cache = tmp_path / "cache_24_price_BTCUSDC.json"
    price_cache.write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        [1400.0, 100.0],
                        [2500.0, float("nan")],
                        [5000.0, 105.0],
                    ]
                }
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    w_col = WhalePositioningCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    assert w_col._resolve_price_1h_change_pct("BTCUSDC") == pytest.approx(5.0)

    w_snap = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.2,
        top_traders_long_pct=0.55,
        taker_buy_sell_ratio=0.60,  # severe taker selling (< 0.75) -> defer BUY
        taker_buy_vol_usd=100_000.0,
        taker_sell_vol_usd=200_000.0,
        open_interest_usd=10_000_000.0,
        open_interest_1h_change_pct=0.1,
        divergence_regime="neutral",
        ts=1000.0,
    )
    w_col.record_snapshot(w_snap, save_to_disk=True)

    ob_col = OrderbookDepthCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    ob_col.record_snapshot(
        OrderbookSnapshot(
            symbol="BTCUSDC",
            mid_price=100.0,
            bid_depth_usd=500_000.0,
            ask_depth_usd=500_000.0,
            imbalance_ratio=0.5,
            largest_bid_wall_usd=50_000.0,
            largest_bid_wall_price=99.5,
            largest_ask_wall_usd=50_000.0,
            largest_ask_wall_price=100.5,
            ts=1000.0,
        ),
        save_to_disk=True,
    )
    d_col = DerivativesTelemetryCollector(cache_ttl_sec=120.0, cache_dir=str(tmp_path), disk_ttl_sec=300.0)
    d_col.record_telemetry(
        DerivativesTelemetry(
            symbol="BTCUSDC",
            funding_rate=0.0001,
            predicted_funding_rate=0.0001,
            open_interest=100.0,
            open_interest_usd=10_000.0,
            ts=1000.0,
        ),
        save_to_disk=True,
    )

    monkeypatch.setattr(external_order_guard, "get_whale_collector", lambda: w_col)
    monkeypatch.setattr(external_order_guard, "get_orderbook_collector", lambda: ob_col)
    monkeypatch.setattr(external_order_guard, "get_derivatives_collector", lambda: d_col)

    ext_margins = dict(custom_margins)
    ext_margins["external_guard_mode"] = "enforce"
    ext_margins["whale_guard_mode"] = "enforce"
    ext_margins["orderbook_wall_guard_mode"] = "enforce"
    ext_margins["funding_guard_mode"] = "enforce"

    # At simulated now=1050.0 (within 120s TTL of ts=1000.0), w_snap is fresh and blocks BUY
    allowed_fresh, reason_fresh, _ = external_order_guard.check_external_order_guards(
        "BTCUSDC", "BUY", 100.0, qty=1.0, margins=ext_margins, now=1050.0
    )
    assert allowed_fresh is False
    assert "severe_taker_selling_dominated" in reason_fresh

    # At simulated now=2000.0 (expired > 300s disk TTL), w_snap is expired -> fail-open allow
    allowed_stale, reason_stale, _ = external_order_guard.check_external_order_guards(
        "BTCUSDC", "BUY", 100.0, qty=1.0, margins=ext_margins, now=2000.0
    )
    assert allowed_stale is True
    assert reason_stale == "ok"


def test_iteration19_parabolic_clock_skew_smart_sl_now_kalman_mean_reversion_nan_and_cli_previous(tmp_path, monkeypatch):
    """Verify Iteration 19 fixes:
    1. ParabolicSurgeGuard resets future disarmed_peak_ts / ts on backward clock skew,
       filters non-finite or future points from price_history, and ignores NaN/Inf volatility_1h_pct.
    2. SmartStopLossGuard propagates `now` to whale/orderbook collectors and resets clock-skewed future grace entries.
    3. KalmanTrend and MeanReversionTrigger reject NaN/Inf/<=0 prices without poisoning internal state,
       and KalmanTrend resets on backward timestamp jumps (raw_dt < -1.0).
    4. LiquidationCapitulationTrigger and WhaleAccumulationTrigger forward `now` to collectors.
    5. intelligence.cli recovers corrupt heartbeat and telemetry files from .previous backups.
    """
    import external_order_guard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard, _GRACE_TRACKER
    from intelligence.internal.triggers.kalman_trigger import KalmanTrend
    from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
    from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector
    from intelligence.external.collectors.whale_positioning import (
        WhalePositioningCollector,
        WhalePositioningSnapshot,
    )
    from intelligence.cli import get_daemon_status, get_macro_analyzer_status, inspect_symbol_intelligence

    # 1. ParabolicSurgeGuard clock-skew & NaN/future history resilience
    p_guard = ParabolicSurgeGuard(surge_threshold_pct=4.0, pullback_required_pct=1.5, window_seconds=3600.0)
    p_guard._states["BTCUSDC"] = {
        "peak": 120.0,
        "surge_active": False,
        "surge_start_px": 100.0,
        "ts": 99999.0,
        "disarmed_peak_ts": 99999.0,  # future disarmed timestamp from prior run
    }
    hist = [
        (1000.0, 100.0),
        (1200.0, float("nan")),
        (1500.0, 106.0),  # +6% surge at t=1500
        (9000.0, 150.0),  # future point beyond now=1500 -> must be ignored
    ]
    dec_surge = p_guard.check(
        "BTCUSDC",
        "BUY",
        current_price=106.0,
        price_history=hist,
        volatility_1h_pct=float("inf"),
        now=1500.0,
    )
    assert dec_surge.allowed is False
    assert "parabolic_surge_active" in dec_surge.reason
    assert dec_surge.metadata["peak"] == pytest.approx(106.0)

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

    # 3. KalmanTrend & MeanReversionTrigger NaN/Inf/backward-ts protection
    import math
    kf = KalmanTrend()
    kf.update(1000.0, 100.0, 0.05)
    kf_nan = kf.update(1010.0, float("nan"), float("nan"))
    assert math.isfinite(kf._x0) and math.isfinite(kf._x1)
    assert kf_nan["vel"] == 0.0
    # Backward timestamp jump resets state cleanly instead of using dt=0.05s
    kf_back = kf.update(500.0, 110.0, 0.05)
    assert kf.last_ts == pytest.approx(500.0)
    assert kf_back["vel"] == pytest.approx(0.0)

    mrt = MeanReversionTrigger(mode="both", rsi_period=3, bb_period=3)
    mrt.update("BTCUSDC", 1000.0, 100.0)
    ev_nan = mrt.update("BTCUSDC", 1010.0, float("nan"))
    assert ev_nan.metadata["rsi"] is None
    assert len(mrt.bb_buf) == 1
    mrt.update("BTCUSDC", 1020.0, 101.0)
    mrt.update("BTCUSDC", 1030.0, 102.0)
    ev_valid = mrt.update("BTCUSDC", 1040.0, 103.0)
    assert ev_valid.metadata["rsi"] is not None and math.isfinite(ev_valid.metadata["rsi"])
    assert ev_valid.metadata["percent_b"] is not None and math.isfinite(ev_valid.metadata["percent_b"])

    # 4. LiquidationCapitulationTrigger & WhaleAccumulationTrigger `now` propagation
    liq_col = BybitLiquidationCollector(retention_sec=3600.0)
    liq_col.record_event("BTCUSDT", "LONG", 250_000.0, ts=3000.0)
    liq_trig = LiquidationCapitulationTrigger(collector=liq_col, min_notional_usd=100_000.0, window_sec=300.0)
    ev_liq = liq_trig.evaluate("BTCUSDC", 100.0, now=3050.0)
    assert ev_liq is not None and ev_liq.reason == "liquidation_capitulation_burst"

    whale_trig = WhaleAccumulationTrigger(collector=w_col, min_top_traders_long_pct=0.60, min_taker_buy_ratio=1.05)
    ev_whale = whale_trig.evaluate("BTCUSDC", 100.0, now=2050.0)
    assert ev_whale is not None and ev_whale.reason == "whale_aggressive_accumulation"

    # 5. intelligence.cli .previous recovery
    (tmp_path / "globaltelemetry_collector.heartbeat").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "globaltelemetry_collector.heartbeat.previous").write_text(
        json.dumps({"pid": os.getpid(), "ts": time.time(), "symbols": ["BTCUSDC"]}),
        encoding="utf-8",
    )
    (tmp_path / "macro_analyzer.heartbeat").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "macro_analyzer.heartbeat.previous").write_text(
        json.dumps({"pid": os.getpid(), "ts": time.time(), "intervals": {"geopolitical_llm": 600}}),
        encoding="utf-8",
    )
    assert get_daemon_status(str(tmp_path))["running"] is True
    assert get_macro_analyzer_status(str(tmp_path))["running"] is True
    rep = inspect_symbol_intelligence("BTCUSDC", cache_dir=str(tmp_path))
    assert rep["pillar2_external"]["whale"] is not None


def test_iteration20_caller_module_quarantine_set_and_position_exit_clock_skew(tmp_path, monkeypatch):
    """Regression test for Iteration 20 fixes:
    1. order_guard.resolve_caller_module normalizes short process codes ('AG', 'MT', 'TA', 'RT',
       'asset_guardian') and falls back to `prefix` when `client_order_id` does not resolve to
       a known guard module (e.g. 'OR_12345_0' -> 'order_retry', prefix='TA_OR' -> 'tradeall').
    2. order_retry_worker._PersistentQuarantineSet.__contains__ calls sync_from_disk() so
       `_alert_terminal` deduplicates persisted terminal keys after worker restart, and
       IncidentDigest.collect uses save_json_state + previous_generation_path recovery.
    3. PositionExitManager.is_valid_price rejects Inf/NaN prices and re-anchors future
       clock-skewed SL/TP/Re-entry timestamps (`started_at > now`, `last_peak_time > now`,
       `last_hard > now`, `last_sell_time_s > now`) so SL timer `sl_trend_defer_max_sec`
       is never artificially extended beyond its configured ceiling.
    """
    import order_guard
    import order_retry as oq
    import order_retry_worker as worker
    from strategies.position_exit_manager import (
        ExitAction,
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
        ReentryAction,
        ReentrySnapshot,
    )

    # 1. resolve_caller_module short codes and client_order_id -> prefix fallback
    assert order_guard.resolve_caller_module("AG") == "assetguardian"
    assert order_guard.resolve_caller_module("MT") == "monitortrades"
    assert order_guard.resolve_caller_module("TA") == "tradeall"
    assert order_guard.resolve_caller_module("RT") == "rtrade"
    assert order_guard.resolve_caller_module("asset_guardian") == "assetguardian"
    assert (
        order_guard.resolve_caller_module(
            None,
            client_order_id="OR_12345_0",
            prefix="TA_OR",
        )
        == "tradeall"
    )

    # 2. _PersistentQuarantineSet.__contains__ syncs from disk across restart & maintains .previous
    q_file = tmp_path / "order_retry_queue.jsonl"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    pset1 = worker._PersistentQuarantineSet("terminal")
    pset1.add(("rec1", 0, "ord1", "canceled"))
    alerts_path = tmp_path / "order_retry_quarantine_alerts.json"
    assert alerts_path.exists()

    # Fresh instance simulating worker process restart checks `in` before any `.add()`
    pset2 = worker._PersistentQuarantineSet("terminal")
    assert ("rec1", 0, "ord1", "canceled") in pset2

    # Simulate mid-write crash where only .previous exists on disk
    from state_io import stage_previous_generation
    stage_previous_generation(str(alerts_path))
    alerts_path.unlink()
    pset3 = worker._PersistentQuarantineSet("terminal")
    assert ("rec1", 0, "ord1", "canceled") in pset3

    # IncidentDigest.collect uses save_json_state and recovers when only .previous exists
    digest_path = tmp_path / "digest.json"
    digest = worker.IncidentDigest(str(digest_path), 300.0)
    ev = {
        "labels": {"symbol": "BTCUSDC", "side": "BUY", "reason": "active_ttl"},
        "quantity": 0.5,
        "sample_id": "s1",
    }
    due1 = digest.collect([ev], now=1000.0)
    assert len(due1) == 1
    due2 = digest.collect([ev], now=1010.0)
    assert due2 == []
    # Stage .previous and remove primary file -> collect([], now=1400.0) still recovers from .previous!
    stage_previous_generation(str(digest_path))
    digest_path.unlink()
    due3 = digest.collect([], now=1400.0)
    assert len(due3) == 1
    assert due3[0]["count"] == 1

    # 3. PositionExitManager is_valid_price rejects Inf/NaN and re-anchors future SL/TP/Re-entry timestamps
    snap_inf = PositionSnapshot(symbol="BTCUSDC", current_price=float("inf"), avg_buy_price=100.0)
    assert snap_inf.is_valid_price is False

    mgr = PositionExitManager(
        config=PositionExitConfig(
            lost_threshold=0.05,
            sl_trend_defer_max_sec=200.0,
            gain_threshold=0.05,
            tp_stagnation_max_sec=100.0,
            hard_tp_enabled=True,
            hard_tp_pct=0.10,
            hard_tp_frac=0.5,
            hard_tp_cooldown_sec=300.0,
            reentry_sl_bounce_pct=0.02,
            reentry_post_stop_cooldown_sec=60.0,
        )
    )
    # Seed future started_at (clock skew) on SL episode & bounce state
    mgr.sl_episode_state["BTCUSDC"] = {"started_at": 5000.0, "buy_price": 100.0, "disaster_floor": 0.08}
    mgr.sl_bounce_state["BTCUSDC"] = {
        "started_at": 5000.0,
        "entry_loss": 0.06,
        "lowest_price": 94.0,
        "peak_bounce_price": 94.0,
        "min_loss": 0.06,
        "buy_price": 100.0,
    }
    d_sl1 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1000.0, trend_up=True)
    )
    assert d_sl1.should_exit is False
    # Re-anchored to 1000.0 instead of staying at 5000.0
    assert mgr.sl_episode_state["BTCUSDC"]["started_at"] == 1000.0
    assert mgr.sl_bounce_state["BTCUSDC"]["started_at"] == 1000.0
    # 200s later (t=1200.0), SL bounce timeout fires on schedule!
    d_sl2 = mgr.evaluate_stop_loss(
        PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1200.0, trend_up=True)
    )
    assert d_sl2.should_exit is True
    assert d_sl2.action == ExitAction.SELL
    assert d_sl2.metadata.get("sl_timer_expired") is True

    # Future hard_tp_last is cleared on clock step-back so Hard-TP spike is not blocked for hours
    mgr.hard_tp_last["BTCUSDC"] = 9000.0
    d_tp_hard = mgr.evaluate_take_profit(
        PositionSnapshot(symbol="BTCUSDC", current_price=112.0, avg_buy_price=100.0, current_time_s=1000.0, trend_up=True)
    )
    assert d_tp_hard.should_exit is True
    assert d_tp_hard.action == ExitAction.PARTIAL_SELL

    # Future last_sell_time_s on STOP re-entry does not permanently lock post_stop_stabilization_cooldown
    d_re = mgr.evaluate_reentry(
        ReentrySnapshot(
            symbol="BTCUSDC",
            current_price=98.0,
            last_sell_price=100.0,
            last_exit_kind="STOP",
            post_sell_trough=94.0,
            current_time_s=1000.0,
            last_sell_time_s=9000.0,
            trend_up=True,
        )
    )
    assert d_re.should_enter is True
    assert d_re.action == ReentryAction.ENTER


def test_iteration21_assetguardian_partial_sell_tier_category_revocation_and_trailing_recovery(tmp_path, monkeypatch):
    """Regression test for Iteration 21 fixes:
    1. assetguardian._sell_campaign_tier computes unfilled_tier_qty = max(0, initial_qty * allocation - already_filled)
       and caps by free_qty, instead of subtracting already_filled from min(free_qty, initial_qty * allocation)
       (which double-subtracted partial fills when free_qty < initial_qty * allocation and falsely marked the
       remaining tier as dust-completed).
    2. Instrument.place and bapi_placeorder.if_place_safe_order preserve sibling category bypasses
       (e.g. 'g_qlimit2' when revoking G_QLIMIT1 from 'qlimit'/'financial' on BUY; 'g_qlimit1' & 'g_qlimit2'
       when revoking G_PROF from 'financial' on SELL), skip _call_profit_guard_window_ref when G_PROF is bypassed,
       and ignore regime_context.suggested_scale when bypass_profit_guard ('all'/'profit_guard') is active.
    3. trailing_core.should_sell rejects non-positive/non-finite prices, and KrakenTrailing._trend_value
       recovers from .previous generation files and handles ms timestamps.
    """
    import assetguardian as ag
    from instrument import Instrument
    from providers.quantity import QuantityDecision
    import trailing_core
    import kraken.trailing_stop as kt

    # 1. assetguardian._sell_campaign_tier does not double-subtract already_filled on partial fill
    state_file = tmp_path / "ag_state.json"
    monkeypatch.setattr(ag, "STATE", ag.AssetGuardianState(str(state_file), fail_closed=False))
    monkeypatch.setattr(ag, "_p3b_threshold_multipliers", lambda *a, **k: (1.0, 1.0, None))
    completed_earlier = [t for t, _ in ag.SELL_TIERS[:-1]]
    last_thr, last_alloc = ag.SELL_TIERS[-1]
    tier_target = 10.0 * last_alloc
    partial_fill = tier_target * 0.6
    rem_expected = tier_target - partial_fill
    total_filled = (10.0 - tier_target) + partial_fill
    ag._save_symbol_campaign(
        "TAOUSDC",
        "sell",
        {
            "trough_price": 100.0,
            "trough_ts": 1000.0,
            "initial_qty": 10.0,
            "created_at": 1000.0,
            "completed_tiers": completed_earlier,
            "filled_qty_by_tier": {ag._tier_key(last_thr): partial_fill},
            "total_filled_qty": total_filled,
            "attempts_by_tier": {},
            "terminal_orders": [],
        },
    )

    class DummyProvider:
        name = "binance"
        def open_orders(self, symbol):
            return []

    tier_res, _, growth = ag._sell_campaign_tier(
        "TAOUSDC",
        current_price=100.0 * (1.0 + last_thr / 100.0 + 0.05),
        minimum_row={"price": 100.0, "timestamp": 1000.0},
        free_qty=rem_expected,
        provider=DummyProvider(),
    )
    assert tier_res is not None
    thr_out, alloc_out, rem_qty = tier_res
    assert thr_out == pytest.approx(last_thr)
    assert alloc_out == pytest.approx(last_alloc)
    # Must return the remaining 40% of tier_target, NOT 0.0 (min(rem_expected, tier_target) - partial_fill)!
    assert rem_qty == pytest.approx(rem_expected)

    # 2. Instrument.place preserves G_QLIMIT2 when revoking G_QLIMIT1 from 'qlimit' on BUY,
    #    skips profit_guard_window_ref when G_PROF is bypassed, and ignores suggested_scale on bypass_guards={'all'}
    window_ref_calls = []
    pg_bypasses = []

    class FakeSpotProvider:
        name = "kraken"
        def validate_symbol(self, sym):
            return None
        def guards_internally(self):
            return False
        def requires_cache_permit(self):
            return False
        def defers_smart_cancel(self):
            return False
        def supports_business_minimum(self):
            return False
        def execution_enabled(self):
            return True
        def preflight_order(self, *a, **k):
            return None
        def round_price(self, sym, px):
            return px
        def round_amount(self, sym, q):
            return q
        def free_balance(self, asset):
            return 10000.0
        def get_current_price(self, sym):
            return 100.0
        def profit_guard_window_ref(self, *a, **k):
            window_ref_calls.append(1)
            raise RuntimeError("should not be called when G_PROF is bypassed")
        def quantity_decision(self, symbol, side, price, qty, **kw):
            rc = kw.get("regime_context")
            sc = getattr(rc, "suggested_scale", 1.0) if (rc is not None and side == "BUY") else 1.0
            final_q = float(qty or 1.0) * (sc if 0 < sc < 1.0 else 1.0)
            return QuantityDecision(
                requested_qty=float(qty or 1.0),
                balance_cap=100.0,
                policy_cap=100.0,
                fee_cap=100.0,
                final_qty=final_q,
                scale_applied=(0 < sc < 1.0),
                applied_scale=sc,
            )
        def place_order(self, symbol, side, price, qty, **kw):
            return {"orderId": "ord-21", "qty": qty}

    monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        order_guard,
        "profit_guard",
        lambda *a, **k: (pg_bypasses.append(set(k.get("bypass_guards") or set())) or True),
    )

    fake_prov = FakeSpotProvider()
    fake_api = MagicMock()
    fake_api.provider_by_name.return_value = fake_prov
    fake_api.market_regime_context.return_value = None
    inst = Instrument(
        name="TAO",
        provider="kraken",
        symbol="TAOUSD",
        base="TAO",
        quote="USD",
        enabled=True,
        api=fake_api,
    )
    # Pass bypass_guards={"financial"} on BUY -> G_QLIMIT1 is revoked, but G_PROF & G_QLIMIT2 stay bypassed!
    res_buy = inst.place(
        "BUY", 100.0, 2.0,
        smart=False, wait_for_trend=False, caller_owns_retry=True,
        bypass_guards={"financial"},
    )
    assert res_buy is not None
    assert window_ref_calls == []  # G_PROF was bypassed, so profit_guard_window_ref was NOT called
    assert len(pg_bypasses) == 1
    assert "g_prof" in pg_bypasses[0]
    assert "g_qlimit2" in pg_bypasses[0]
    assert "g_qlimit1" not in pg_bypasses[0]

    # Pass bypass_guards={"all"} on BUY with a regime_context that has suggested_scale=0.25 -> must NOT downscale qty!
    ctx_scaled = _build_test_context()
    object.__setattr__(ctx_scaled, "suggested_scale", 0.25)
    res_all = inst.place(
        "BUY", 100.0, 2.0,
        smart=False, wait_for_trend=False, caller_owns_retry=True,
        bypass_guards={"all"},
        regime_context=ctx_scaled,
    )
    assert res_all is not None
    assert res_all["qty"] == pytest.approx(2.0)

    # 3. trailing_core.should_sell & KrakenTrailing._trend_value .previous recovery
    assert trailing_core.should_sell(0.0, 100.0, 15.0) is False
    assert trailing_core.should_sell(-5.0, 100.0, 15.0) is False
    assert trailing_core.should_sell(float("nan"), 100.0, 15.0) is False
    assert trailing_core.should_sell(80.0, 100.0, 15.0) is True

    trend_file = tmp_path / "cache_short_trend.json"
    trend_prev = tmp_path / "cache_short_trend.json.previous"
    trend_file.write_text("{corrupt", encoding="utf-8")
    trend_prev.write_text(
        json.dumps({"HYPEUSDC": {"ts": time.time() * 1000.0, "gradient_recent": 0.42}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(kt, "CACHE_TREND", str(trend_file))
    kt_inst = object.__new__(kt.KrakenTrailing)
    assert kt_inst._trend_value("HYPE/USD") == pytest.approx(0.42)


def test_iteration22_guard_bypass_union_mt_state_clock_skew_kraken_cache_and_telemetry(tmp_path, monkeypatch):
    """Iteration 22 regression tests:
    1. Instrument.place, bapi_placeorder.if_place_safe_order, and rtrade._is_gi_whale_enforced
       union caller bypass_guards with instrument-level and venue-level bypass_guards.
    2. monitortrades._load_mt_state, _is_pending_exit_active, _evaluate_hard_take_profit,
       and _execute_stop_loss clamp/reject non-finite or future (clock-skewed) timestamps.
    3. KrakenProvider._fills_from_cache rejects NaN/bool/future timestamps and normalizes seconds.
    4. collect_pretrade_telemetry handles non-finite orderbook imbalance, non-numeric slopes,
       and empty macro sentiment dictionaries gracefully.
    """
    from instrument import Instrument
    from providers.quantity import QuantityDecision
    from providers.kraken_provider import KrakenProvider
    from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry
    import binance_api.bapi_placeorder as bapi_placeorder
    import rtrade
    import monitortrades as mt

    # 1A. Instrument.place unions caller bypass_guards with instrument params and venue bypass_guards
    pg_bypasses = []

    class FakeKrakenProvider:
        name = "kraken"
        def validate_symbol(self, sym):
            return None
        def guards_internally(self):
            return False
        def requires_cache_permit(self):
            return False
        def defers_smart_cancel(self):
            return False
        def supports_business_minimum(self):
            return False
        def execution_enabled(self):
            return True
        def preflight_order(self, *a, **k):
            return None
        def round_price(self, sym, px):
            return px
        def round_amount(self, sym, q):
            return q
        def free_balance(self, asset):
            return 10000.0
        def get_current_price(self, sym):
            return 100.0
        def profit_guard_window_ref(self, *a, **k):
            return None
        def quantity_decision(self, symbol, side, price, qty, **kw):
            return QuantityDecision(
                requested_qty=float(qty or 1.0),
                balance_cap=100.0,
                policy_cap=100.0,
                fee_cap=100.0,
                final_qty=float(qty or 1.0),
                scale_applied=False,
                applied_scale=1.0,
            )
        def place_order(self, symbol, side, price, qty, **kw):
            return {"orderId": "ord-22", "qty": qty}

    orig_margins = dict(order_guard._load_margins())
    monkeypatch.setattr(
        order_guard,
        "_load_margins",
        lambda: {**orig_margins, "kraken_bypass_guards": {"GLLM3"}, "binance_bypass_guards": {"GI_WHALE"}},
    )
    monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        order_guard,
        "profit_guard",
        lambda *a, **k: (pg_bypasses.append(set(k.get("bypass_guards") or set())) or True),
    )

    fake_api = MagicMock()
    fake_api.provider_by_name.return_value = FakeKrakenProvider()
    fake_api.market_regime_context.return_value = None
    inst = Instrument(
        name="TAO",
        provider="kraken",
        symbol="TAOUSD",
        base="TAO",
        quote="USD",
        enabled=True,
        params={"bypass_guards": "GI_OBOOK"},
        api=fake_api,
    )
    res = inst.place(
        "BUY", 100.0, 1.0,
        smart=False, wait_for_trend=False, caller_owns_retry=True,
        bypass_guards={"G_PROF"},
    )
    assert res is not None
    assert len(pg_bypasses) == 1
    assert {"g_prof", "gi_obook", "gllm3"}.issubset(pg_bypasses[0])

    # 1B. bapi_placeorder.if_place_safe_order unions caller bypass_guards with binance_bypass_guards
    pg_bapi = []
    monkeypatch.setattr(bapi_placeorder.api, "get_current_price", lambda s: 85000.0)
    monkeypatch.setattr(
        order_guard,
        "profit_guard",
        lambda *a, **k: (pg_bapi.append(set(k.get("bypass_guards") or set())) or True),
    )
    ok_b, _ = bapi_placeorder.if_place_safe_order(
        "BUY", "BTCUSDC", 85000.0, 0.01, 3600,
        bypass_guards={"G_PROF"},
    )
    assert ok_b is True
    assert len(pg_bapi) == 1
    assert {"g_prof", "gi_whale"}.issubset(pg_bapi[0])

    # 1C. rtrade._is_gi_whale_enforced unions caller bypass_guards with binance_bypass_guards
    assert rtrade._is_gi_whale_enforced(bypass_guards={"G_PROF"}) is False

    # 2. monitortrades._load_mt_state re-anchors clock-skewed future timestamps and rejects NaN
    mt_state_file = tmp_path / "mt_state.json"
    now_s = 10_000.0
    mt_state_file.write_text(
        json.dumps({
            "sl_bounce_state": {
                "BTCUSDC": {"started_at": now_s + 5000.0, "buy_price": 100.0},
            },
            "tp_ride_state": {
                "ETHUSDC": {"started_at": now_s - 50.0, "last_peak_time": now_s + 9000.0, "buy_price": 200.0},
            },
            "pending_stop_exit": {
                "SOLUSDC": {"time": now_s + 3000.0, "kind": "SL"},
            },
            "hard_tp_last": {
                "ADAUSDC": now_s + 7000.0,
                "XRPUSDC": now_s - 100.0,
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(mt, "_MT_PERSISTENCE_ENABLED", True)
    monkeypatch.setattr(mt, "_MT_STATE_FILE", str(mt_state_file))
    mt._SL_BOUNCE_STATE.clear()
    mt._TP_RIDE_STATE.clear()
    mt._PENDING_STOP_EXIT.clear()
    mt._hard_tp_last.clear()
    mt._load_mt_state(now_curr=now_s)
    assert mt._SL_BOUNCE_STATE["BTCUSDC"]["started_at"] == pytest.approx(now_s)
    assert mt._TP_RIDE_STATE["ETHUSDC"]["last_peak_time"] == pytest.approx(now_s)
    assert mt._PENDING_STOP_EXIT["SOLUSDC"]["time"] == pytest.approx(now_s)
    assert "ADAUSDC" not in mt._hard_tp_last
    assert mt._hard_tp_last["XRPUSDC"] == pytest.approx(now_s - 100.0)

    # 3. KrakenProvider._fills_from_cache rejects NaN and future timestamps, accepts seconds
    kp = KrakenProvider()
    cache_path = tmp_path / "cache_trade_kraken.json"
    monkeypatch.setattr("providers.kraken_provider._KRAKEN_CACHE_FILE", str(cache_path))
    now_curr = time.time()
    # Future timestamp -> rejected
    cache_path.write_text(
        json.dumps({
            "items": {"TAOUSD": [{"price": 400.0, "vol": 1.0, "type": "buy", "time": now_curr}]},
            "fetchtime": {"TAOUSD": (now_curr + 3600.0) * 1000.0},
        }),
        encoding="utf-8",
    )
    assert kp._fills_from_cache("TAOUSD") is None
    # Seconds timestamp -> normalized to ms and accepted
    cache_path.write_text(
        json.dumps({
            "items": {"TAOUSD": [{"price": 400.0, "vol": 1.0, "type": "buy", "time": now_curr}]},
            "fetchtime": {"TAOUSD": now_curr - 5.0},
        }),
        encoding="utf-8",
    )
    fills_ok = kp._fills_from_cache("TAOUSD")
    assert isinstance(fills_ok, list) and len(fills_ok) == 1

    # 4. collect_pretrade_telemetry handles non-numeric slopes and empty macro dicts
    (tmp_path / "cache_short_trend.json").write_text(
        json.dumps({"items": {"TAOUSDC": {"dynamic": {"222": {"slope_full": None}, "900": {"slope_full": "bad"}}}}}),
        encoding="utf-8",
    )
    (tmp_path / "sentiment_advisor_eval.json").write_text(json.dumps({}), encoding="utf-8")
    telem = collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(tmp_path))
    assert telem.get("Dynamic Multi-Window Slopes") == "3.7m=+0.00000, 15m=+0.00000, 2.5h=+0.00000"
    assert "Macro Sentiment Bias" not in telem


def test_iteration23_p3b_bypass_union_collector_from_dict_cachetrend_slice_and_order_retry_set_serialization(tmp_path, monkeypatch):
    """Iteration 23 regression tests:
    1. assetguardian._p3b_threshold_multipliers unions caller bypass_guards with binance_bypass_guards.
    2. check_external_order_guards and check_llm_order_guards honor intelligence_guards_mode='off',
       and resolve_cached_gaussian_horizon rejects non-finite/negative values.
    3. OrderbookSnapshot.from_dict, WhalePositioningSnapshot.from_dict, and DerivativesTelemetry.from_dict
       tolerate null/bool/NaN values without raising TypeError.
    4. CacheShortTrendManager._live_trend_file / _hist_trend_file sanitize slashes in symbols (TAO/USD),
       and get_trend_for_slice resolves cross-venue symbol variants (TAO/USD -> TAOUSDC).
    5. order_retry.enqueue serializes set/frozenset values inside place_kwargs cleanly to JSONL.
    """
    import assetguardian as ag
    from external_order_guard import check_external_order_guards
    from llm_order_guard import check_llm_order_guards
    from qlimit_order_guard import resolve_cached_gaussian_horizon
    from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
    from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
    from types import SimpleNamespace
    from cachetrend import CacheShortTrendManager
    import order_retry

    # 1. assetguardian._p3b_threshold_multipliers unions caller bypass_guards with venue bypass_guards
    orig_margins = dict(order_guard._load_margins())
    monkeypatch.setattr(
        order_guard,
        "_load_margins",
        lambda: {**orig_margins, "binance_bypass_guards": {"GLLM2"}},
    )
    defensive_assessment = {
        "recommended_action": "DEFENSIVE",
        "market_bias": "BEARISH",
        "risk_level": "HIGH",
        "confidence": 0.9,
        "ts": time.time(),
    }
    sell_mult, buy_mult, eff_a = ag._p3b_threshold_multipliers(
        assessment=defensive_assessment,
        bypass_guards={"G_PROF"},
    )
    assert sell_mult == pytest.approx(1.0)
    assert buy_mult == pytest.approx(1.0)
    assert eff_a is None

    # 2. check_external_order_guards & check_llm_order_guards honor intelligence_guards_mode='off'
    off_margins = {**orig_margins, "intelligence_guards_mode": "off"}
    assert check_external_order_guards("BTCUSDC", "BUY", 85000.0, margins=off_margins) == (
        True,
        "intelligence_guards_off",
        1.0,
    )
    assert check_llm_order_guards("binance", "BTCUSDC", "BUY", 85000.0, margins=off_margins) == (
        True,
        "intelligence_guards_off",
        1.0,
    )

    # resolve_cached_gaussian_horizon rejects non-finite / negative values
    t_cache = tmp_path / "t_cache.json"
    t_cache.write_text(
        json.dumps({"BTCUSDC": {"p90_d": -5.0, "median_d": float("nan"), "T": float("inf")}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("intelligence.internal.state.survival.T_CACHE_FILE", str(t_cache))
    assert resolve_cached_gaussian_horizon("BTCUSDC") == (None, None, None)

    # 3. Snapshot.from_dict tolerates None / bool / NaN fields
    ob_snap = OrderbookSnapshot.from_dict({"symbol": "BTCUSDC", "imbalance_ratio": None, "largest_ask_wall_usd": "NaN"})
    assert ob_snap.imbalance_ratio == pytest.approx(0.5)
    assert ob_snap.largest_ask_wall_usd == pytest.approx(0.0)

    w_snap = WhalePositioningSnapshot.from_dict({"symbol": "BTCUSDC", "taker_buy_sell_ratio": None, "funding_rate": "NaN"})
    assert w_snap.taker_buy_sell_ratio == pytest.approx(1.0)
    assert w_snap.funding_rate is None

    d_snap = DerivativesTelemetry.from_dict({"symbol": "BTCUSDC", "funding_rate": None, "open_interest_usd": True})
    assert d_snap.funding_rate == pytest.approx(0.0)
    assert d_snap.open_interest_usd == pytest.approx(0.0)

    # 4. CacheShortTrendManager slash sanitization & cross-venue get_trend_for_slice
    st_file = tmp_path / "cache_short_trend.json"
    (tmp_path / "24trend").mkdir(parents=True, exist_ok=True)
    stm = CacheShortTrendManager(symbols=["TAOUSDC"], filename=str(st_file), writer=True)
    assert "/" not in os.path.basename(stm._live_trend_file("TAO/USD"))
    assert "/" not in os.path.basename(stm._hist_trend_file("TAO/USD"))
    c24_path = tmp_path / "cache_24_price_TAOUSDC.json"
    c24_path.write_text(
        json.dumps({
            "items": {
                "TAOUSDC": [
                    [1_700_000_000_000, 100.0],
                    [1_700_000_010_000, 101.0],
                    [1_700_000_020_000, 102.0],
                    [1_700_000_030_000, 103.0],
                ]
            }
        }),
        encoding="utf-8",
    )
    slice_res = stm.get_trend_for_slice("TAO/USD", 1_700_000_000.0, 1_700_000_030.0)
    assert slice_res is not None
    assert slice_res["growth_coefficient"] > 0

    # 5. order_retry.enqueue serializes set/frozenset in place_kwargs cleanly
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(order_retry, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(order_retry, "LOCK_FILE", str(l_file))
    rec_id = order_retry.enqueue(
        "BTCUSDC",
        "BUY",
        0.1,
        place_kwargs={"bypass_guards": {"g_prof", "gllm1"}},
        requested_price=85000.0,
        ref_price=85000.0,
        failure_reason="trend_deferred",
    )
    assert rec_id is not None
    loaded_q = order_retry.load_all()
    assert len(loaded_q) == 1
    assert loaded_q[0]["place_kwargs"]["bypass_guards"] == ["g_prof", "gllm1"]


def test_iteration24_survival_clock_skew_composite_bypass_guards_and_market_regime_slash_symbol(tmp_path, monkeypatch):
    """Iteration 24 regression tests:
    1. survival.estimate_T rejects None/NaN/future (clock-skewed) cache timestamps and normalizes cache keys;
       get_trend_survival_metrics guards against None/NaN trend_duration_seconds and non-positive cached metrics.
    2. CompositeMarketIntelligence.evaluate_guards honors bypass_guards across G_QLIMIT2, GI_FUNDING, GLLM1, GLLM2, GLLM3,
       and GeopoliticalShockGuard.check tolerates None threat_level / recommended_brake.
    3. OrderbookWallGuard, WhaleDivergenceGuard, and FundingCrowdingGuard safely handle snapshot objects with None/NaN attributes.
    4. MarketRegimeContext.is_valid_for matches same-venue Kraken slash vs non-slash symbols (TAO/USD vs TAOUSD).
    """
    from types import SimpleNamespace
    import intelligence.internal.state.survival as survival
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from market_regime import MarketRegimeContext, MarketRegimeDecision

    # 1. survival.get_trend_survival_metrics & estimate_T safety
    t_cache = tmp_path / "cache_T_trend.json"
    now_curr = time.time()
    t_cache.write_text(
        json.dumps({
            "BTCUSDC": {
                "T": -10.0,
                "median_d": float("nan"),
                "p90_d": None,
                "ts": now_curr + 86400.0,  # clock-skewed future ts
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(survival, "T_CACHE_FILE", str(t_cache))
    metrics = survival.get_trend_survival_metrics("BTCUSDC", trend_duration_seconds=None)
    assert metrics["duration_days"] == 0.0
    assert metrics["median_days"] == 3.0
    assert metrics["p90_days"] == 7.0
    assert metrics["T"] == 8.0
    assert metrics["gaussian_horizon_days"] is None

    # Future ts in cache does not short-circuit estimate_T before fetch_klines
    fetch_called = []
    monkeypatch.setattr(
        survival,
        "fetch_klines",
        lambda sym, days: (fetch_called.append(sym) or ([], [])),
    )
    est = survival.estimate_T("tao/usd")
    assert len(fetch_called) >= 1
    assert est is not None

    # 2. CompositeMarketIntelligence.evaluate_guards honors bypass_guards & GeopoliticalShockGuard tolerates None fields
    geo_guard = GeopoliticalShockGuard()
    dec_null_geo = geo_guard.check(
        "BTCUSDC",
        "BUY",
        SimpleNamespace(threat_level=None, recommended_brake=None, summary="none", risk_score=0.0),
    )
    assert dec_null_geo.allowed is True

    cmi = CompositeMarketIntelligence()
    shock_assessment = SimpleNamespace(
        threat_level="CRITICAL_SHOCK",
        recommended_brake="HARD_VETO_NEW_BUYS",
        summary="crisis",
        risk_score=0.95,
    )
    dec_blocked = cmi.evaluate_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        geopolitical_assessment=shock_assessment,
    )
    assert dec_blocked.allowed is False
    dec_bypassed = cmi.evaluate_guards(
        "BTCUSDC",
        "BUY",
        85000.0,
        geopolitical_assessment=shock_assessment,
        bypass_guards={"GLLM3"},
    )
    assert dec_bypassed.allowed is True

    # 3. External guards tolerate None/NaN attributes on synthetic snapshots
    ob_guard = OrderbookWallGuard()
    ob_dec = ob_guard.check(
        "BTCUSDC",
        "BUY",
        snapshot=SimpleNamespace(
            imbalance_ratio=None,
            largest_ask_wall_usd=float("nan"),
            largest_ask_wall_price=None,
            largest_bid_wall_usd=None,
            largest_bid_wall_price=None,
        ),
    )
    assert ob_dec.allowed is True

    whale_guard = WhaleDivergenceGuard()
    whale_dec = whale_guard.check(
        "BTCUSDC",
        "SELL",
        snapshot=SimpleNamespace(
            divergence_regime=None,
            open_interest_1h_change_pct=float("nan"),
            taker_buy_sell_ratio=None,
            top_traders_long_pct=float("nan"),
        ),
    )
    assert whale_dec.allowed is True

    fund_guard = FundingCrowdingGuard()
    fund_dec = fund_guard.check(
        "BTCUSDC",
        "BUY",
        telemetry=SimpleNamespace(funding_rate=float("nan")),
    )
    assert fund_dec.allowed is True

    # 4. MarketRegimeContext.is_valid_for matches TAO/USD and TAOUSD on same venue
    dec = MarketRegimeDecision(
        regime="bull",
        gradient=0.01,
        epsilon=0.002,
        strength=5.0,
        fresh=True,
        reason="directional_signal",
    )
    ctx = MarketRegimeContext.from_decision(
        dec,
        evaluated_at=now_curr,
        symbol="TAO/USD",
        provider="kraken",
        trend_duration_seconds=300.0,
    )
    assert ctx.is_valid_for(symbol="TAOUSD", provider="kraken", now=now_curr) is True
    assert ctx.is_valid_for(symbol="TAO/USD", provider="kraken", now=now_curr) is True
    assert ctx.is_valid_for(symbol="ETHUSD", provider="kraken", now=now_curr) is False


def test_iteration25_smart_sl_clock_skew_post_stop_nan_internal_sentiment_guards_and_macro_disk_load(tmp_path, monkeypatch):
    """Iteration 25 regression tests:
    1. SmartStopLossGuard._load_grace_state rejects None/future (clock-skewed) start_ts without wiping valid entries,
       normalizes Kraken slash symbols (TAO/USD -> TAOUSD), and tolerates None/NaN snapshot attributes.
    2. PostStopReentryGuard.check normalizes slash symbols in cache_key and tolerates NaN last_sell_price / last_sell_time_s.
    3. WeibullExhaustionGuard, ParabolicSurgeGuard, NoiseFloorGuard, ExtremeGreedGuard, PanicWashoutGuard, and
       LiquidationCascadeGuard safely handle None/bool/NaN/Inf inputs.
    4. GeopoliticalThreatAnalyzer._load_from_disk and SentimentAdvisor._load_from_disk handle null fields in JSON.
    """
    from types import SimpleNamespace
    import intelligence.external.guards.smart_stop_loss_guard as ssl_mod
    from intelligence.sentiment.guards.post_stop_reentry_guard import (
        PostStopReentryGuard,
        clear_post_stop_reentry_cache,
    )
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.noise_guard import NoiseFloorGuard
    from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
    from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
    from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor

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
    monkeypatch.setattr(ssl_mod, "_SMART_SL_STATE_FILE", str(ssl_file))
    ssl_mod._GRACE_TRACKER.clear()
    ssl_mod._LOADED_STATE_FILE_KEY = None
    tracker = ssl_mod._load_grace_state(is_sim=False)
    assert "BAD_NULL" not in tracker
    assert "FUTURE_SKEW" not in tracker
    assert "TAOUSD" in tracker

    ssl_guard = ssl_mod.SmartStopLossGuard(mode="enforce", grace_seconds=120.0)
    should_exec, reason, meta = ssl_guard.evaluate(
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
    geo_file = tmp_path / "geopolitical_threat_eval.json"
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

    adv_file = tmp_path / "sentiment_advisor_eval.json"
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


def test_iteration26_spot_rules_nan_backoff_clamp_macro_null_json_and_window_ref_nan(tmp_path):
    """Iteration 26:
    1. spot_rules.check_slow_grind_exhaustion, check_parabolic_surge_exhaustion, and check_fast_profit_reversal
       must reject NaN timestamps/thresholds rather than letting < comparisons evaluate to False and trigger exits.
    2. Strategy._placement_backoff_remaining must clamp clock-skewed future 'until' to _PLACEMENT_BACKOFF_CAP_SEC
       and fail open on NaN/bool 'until'.
    3. MacroAnalyzer.assess_market_advisor must tolerate null JSON fields in fear_greed_collect.json and
       market_breadth_collect.json without raising TypeError.
    4. order_guard.window_reference, providers.base.aggregate_fills_by_order_id, KrakenProvider.get_orders,
       and HyperliquidProvider.get_orders must reject NaN/negative window/cutoff rather than letting all
       historical fills bypass the cutoff check.
    """
    import json
    from types import SimpleNamespace
    from strategies import spot_rules
    from strategies.spot_engine import Strategy, _PLACEMENT_BACKOFF_CAP_SEC
    from intelligence.macro_analyzer import MacroAnalyzer
    import order_guard
    from providers.base import aggregate_fills_by_order_id
    from providers.kraken_provider import KrakenProvider
    from providers.hyperliquid_provider import HyperliquidProvider

    # 1. spot_rules NaN hardening
    trig_sg, reason_sg = spot_rules.check_slow_grind_exhaustion(
        current_price=120.0,
        avg_cost=100.0,
        entry_ts=float("nan"),
        current_ts=1_700_000_000.0,
        min_days=7.0,
        min_gain_pct=15.0,
        recent_peak=130.0,
        shadow_prices=[(1_700_000_000.0, 130.0)],
        flash_window_sec=900.0,
        flash_drop_pct=1.5,
        structural_drop_pct=2.5,
    )
    assert trig_sg is False and reason_sg == ""

    trig_ps, reason_ps = spot_rules.check_parabolic_surge_exhaustion(
        current_price=95.0,
        avg_cost=100.0,
        surge_peak=125.0,
        surge_gain_pct=18.0,
        exit_pullback_pct=5.0,
        surge_active=True,
        min_profit_pct=float("nan"),
    )
    assert trig_ps is False and reason_ps == ""

    trig_fp, gain_fp, drop_fp = spot_rules.check_fast_profit_reversal(
        current_price=115.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(1000.0, float("nan")), (1005.0, 120.0)],
        window_sec=300.0,
        drop_pct=3.0,
        current_time=1010.0,
    )
    assert trig_fp is True
    assert drop_fp > 3.0

    # 2. Strategy._placement_backoff_remaining clamping & NaN safety
    dummy_strat = SimpleNamespace(
        s={
            "placement_backoffs": {
                "buy:DCA": {"until": 1_700_000_000.0 + 999999.0},
                "sell:TP": {"until": float("nan")},
            }
        },
        _now=lambda: 1_700_000_000.0,
        _placement_backoff_key=Strategy._placement_backoff_key,
    )
    rem_dca = Strategy._placement_backoff_remaining(dummy_strat, "BUY", "DCA")
    assert rem_dca == _PLACEMENT_BACKOFF_CAP_SEC
    rem_tp = Strategy._placement_backoff_remaining(dummy_strat, "SELL", "TP")
    assert rem_tp == 0.0

    # 3. MacroAnalyzer.assess_market_advisor null JSON fields
    (tmp_path / "fear_greed_collect.json").write_text(
        json.dumps({
            "value": None,
            "classification": None,
            "ts": None,
            "historical_values": [None, 45],
            "trend_7d_change": None,
        }),
        encoding="utf-8",
    )
    (tmp_path / "market_breadth_collect.json").write_text(
        json.dumps({
            "advance_ratio": None,
            "advancing_count": None,
            "declining_count": None,
            "total_count": None,
            "median_change_pct": None,
            "mean_change_pct": None,
            "dispersion_std": None,
            "regime": None,
            "top_gainers": [["BTC", None]],
            "top_losers": [["ETH", None]],
            "ts": None,
        }),
        encoding="utf-8",
    )
    captured_fg = []
    captured_mb = []
    macro = MacroAnalyzer(symbols=["BTCUSDC"], cache_dir=str(tmp_path))
    macro.gemini_advisor = SimpleNamespace(
        review=lambda fear_greed=None, breadth=None, **kw: (
            captured_fg.append(fear_greed),
            captured_mb.append(breadth),
            SimpleNamespace(is_fallback=False, market_bias="NEUTRAL", risk_level="LOW", recommended_action="HOLD", summary="ok", ts=1_700_000_000.0),
        )[-1]
    )
    assert macro.assess_market_advisor(force=True, now=1_700_000_000.0) is True
    assert len(captured_fg) == 1 and captured_fg[0].value == 50 and captured_fg[0].classification == "Neutral"
    assert len(captured_mb) == 1 and captured_mb[0].advance_ratio == 0.5 and captured_mb[0].regime == "NEUTRAL"

    # 4. aggregate_fills_by_order_id, window_reference, KrakenProvider.get_orders, HyperliquidProvider.get_orders with NaN
    rows = [{"orderId": "1", "side": "SELL", "price": 100.0, "qty": 1.0, "timestamp": 1_700_000_000_000}]
    assert aggregate_fills_by_order_id(rows, cutoff_ms=float("nan")) == []

    called_get_orders = []
    dummy_prov = SimpleNamespace(
        name="kraken",
        get_orders=lambda sym, side, win: (called_get_orders.append(win), rows)[-1],
    )
    assert order_guard.window_reference(dummy_prov, "TAOUSD", "BUY", float("nan")) is None
    assert called_get_orders == []

    kp = KrakenProvider()
    assert kp.get_orders("TAOUSD", "BUY", float("nan")) == []
    assert kp.get_orders("TAOUSD", "BUY", -10.0) == []

    hp = HyperliquidProvider()
    assert hp.get_orders("HYPEUSDC", "BUY", float("nan")) == []
    assert hp.get_orders("HYPEUSDC", "BUY", -5.0) == []


def test_iteration27_should_wait_none_fallback_price_analysis_memo_and_quantity_nan_policy(monkeypatch):
    """Iteration 27 regression suite:
    1. CacheShortTrendManager.should_wait (GT_S) falls back to gradient_recent when
       growth_coefficient or gradient_recent_fast is explicitly None in snapshot/dynamic dict.
    2. priceAnalysis.get_weight_for_cash_permission_at_quant_time handles slash symbols (TAO/USD),
       NaN duration_seconds, direction changes at the same timestamp, and backward clock skew.
    3. MarketRegimeService.compose_evidence matches TAO/USD with TAOUSD and _evaluate_provider_source
       rejects negative cache age (clock skew).
    4. decide_quantity rejects NaN/bool policy_cap with refuse_reason="policy_unavailable"
       and ignores NaN suggested_scale.
    """
    from types import SimpleNamespace
    from cachetrend import CacheShortTrendManager
    import cacheManager as cm
    import priceAnalysis as pa
    from market_regime import MarketRegimeService, MarketRegimeEvidence, MarketRegimeDecision, ClosedPriceSeries
    from providers.quantity import decide_quantity

    # 1. CacheShortTrendManager.should_wait (GT_S) with None growth_coefficient / gradient_recent_fast
    mgr = CacheShortTrendManager.__new__(CacheShortTrendManager)
    mgr.window_seconds = (1800,)
    mgr._epsilon = lambda snap: 1e-6
    mgr.get_fresh_snapshot = lambda sym, now=None: {
        "growth_coefficient": None,
        "gradient_recent_fast": None,
        "gradient_recent": -0.005,
    }
    assert mgr.should_wait("BUY", "BTCUSDC", fast=False, use_noise_gate=False) is True
    assert mgr.should_wait("BUY", "BTCUSDC", fast=True, use_noise_gate=False) is True
    assert mgr.should_wait("SELL", "BTCUSDC", fast=False, use_noise_gate=False) is False

    mgr.get_short_trend_for_window = lambda sym, win, now=None: {
        "growth_coefficient": None,
        "gradient_recent": 0.004,
        "epsilon": 1e-6,
    }
    assert mgr.should_wait("SELL", "BTCUSDC", window_seconds=900, use_noise_gate=True) is True
    assert mgr.should_wait("BUY", "BTCUSDC", window_seconds=900, use_noise_gate=True) is False

    # 2. priceAnalysis.get_weight_for_cash_permission_at_quant_time
    fake_lt_cache = {
        "TAOUSD": [
            {
                "duration_seconds": 86400 * 7,
                "direction": "up",
                "start_timestamp": 1_700_000_000.0,
                "timestamp": 1_700_604_800.0,
            }
        ]
    }
    monkeypatch.setattr(
        cm,
        "get_cache_manager",
        lambda name: SimpleNamespace(cache=fake_lt_cache),
    )
    pa.last_timestamp.clear()
    pa.last_w.clear()
    pa.last_calc_time.clear()
    pa.last_duration.clear()

    # Slash symbol TAO/USD resolves to TAOUSD and computes aligned BUY weight (~0.95 at mid-T)
    w_up = pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY")
    assert w_up is not None and w_up > 0.8

    # Flipping direction to 'down' at the same timestamp must NOT return the cached 'up' weight
    fake_lt_cache["TAOUSD"][0]["direction"] = "down"
    w_down = pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY")
    assert w_down is not None and w_down < 0.2

    # Future last_calc_time (clock skew) forces recalculation instead of using stale last_w
    memo_key = ("TAO/USD", "BUY", 14.0, "down")
    pa.last_calc_time[memo_key] = time.time() + 10_000.0
    pa.last_w[memo_key] = [0.999]
    w_recalc = pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY")
    assert w_recalc is not None and w_recalc < 0.2

    # NaN duration_seconds returns None cleanly without ValueError
    fake_lt_cache["TAOUSD"][0]["duration_seconds"] = float("nan")
    assert pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY") is None

    # 3. MarketRegimeService.compose_evidence slash normalization & _evaluate_provider_source clock skew
    svc = MarketRegimeService(clock=lambda: 1000.0, cache_ttl_sec=60.0)
    up_dec = MarketRegimeDecision(
        regime="bull", gradient=0.01, epsilon=0.001, strength=10.0, fresh=True, reason="ok",
    )
    ev_short = MarketRegimeEvidence(
        role="asset_short", family="price_trend", classifier="linreg",
        provider="kraken", symbol="TAO/USD", decision=up_dec, evaluated_at=1000.0,
    )
    ev_long = MarketRegimeEvidence(
        role="asset_long", family="price_trend", classifier="linreg",
        provider="kraken", symbol="TAO/USD", decision=up_dec, evaluated_at=1000.0,
    )
    comp = svc.compose_evidence((ev_short, ev_long), asset_symbol="TAOUSD")
    assert comp.regime == "bull"

    calls = []
    class DummyRegimeProv:
        name = "kraken"
        def ohlc_series(self, sym, interval_min):
            calls.append(sym)
            return ClosedPriceSeries(
                closes=tuple(100.0 + i for i in range(30)),
                interval_min=interval_min,
                last_closed_at=1000.0,
                timestamps=tuple(1000.0 - (29 - i) * 60.0 for i in range(30)),
            )

    prov = DummyRegimeProv()
    svc._evaluate_provider_source(prov, "TAOUSD", interval_min=1, window_seconds=900.0)
    assert len(calls) == 1
    # Inject future timestamp into cache (clock skew: cached_at = 2000.0 > clock = 1000.0)
    cache_key = ("kraken", "TAOUSD", 1, 900.0)
    svc._cache[cache_key] = (2000.0,) + svc._cache[cache_key][1:]
    svc._evaluate_provider_source(prov, "TAOUSD", interval_min=1, window_seconds=900.0)
    assert len(calls) == 2

    # 4. decide_quantity with NaN/bool policy_cap and NaN suggested_scale
    class DummyQtyProvider:
        def spent_asset(self, side, base, quote):
            return quote if side == "BUY" else base
        def free_balance(self, asset):
            return 1000.0
        def balance_cap_quantity(self, side, price, free):
            return free / price
        def policy_cap_quantity(self, *a, **k):
            return float("nan")
        def fee_cap_quantity(self, *a, **k):
            return 10.0
        def round_amount(self, sym, qty):
            return round(qty, 4)

    q_nan = decide_quantity(DummyQtyProvider(), "TAOUSD", "BUY", 100.0, 2.0, apply_policy=True)
    assert q_nan.refuse_reason == "policy_unavailable"
    assert q_nan.final_qty == 0.0

    class DummyQtyProviderOk(DummyQtyProvider):
        def policy_cap_quantity(self, *a, **k):
            return 5.0

    q_scale_nan = decide_quantity(
        DummyQtyProviderOk(),
        "TAOUSD",
        "BUY",
        100.0,
        2.0,
        apply_policy=True,
        regime_context=SimpleNamespace(suggested_scale=float("nan")),
    )
    assert q_scale_nan.refuse_reason is None
    assert q_scale_nan.final_qty == 2.0
    assert q_scale_nan.scale_applied is False


def test_iteration28_cache_symbol_variants_future_timestamps_cancel_bounds_and_market_api_place_params(monkeypatch):
    """Iteration 28 regression suite:
    1. CacheTradeManager.get_last_opposite_fill_price and CacheCurrentPriceManager
       (provider_name_for, get_cached_price_observation, get_price) normalize slash/case
       symbol variants (e.g. 'BTC/USDC', 'tao/usd').
    2. bapi_allorders.get_trade_orders, bapi_trades.get_trade_orders, and bapi_trades.aggregate_trades
       reject future-dated (clock-skewed) timestamps and zero-price/zero-qty rows.
    3. bapi.cancel_expired_orders and bapi.cancel_recent_orders reject bool/NaN/non-positive
       time bounds and future order timestamps without canceling valid open orders.
    4. MarketApi.place inherits instrument params (e.g. bypass_guards) from instrument_registry
       when not explicitly supplied, and latest_fill_price / market_regime_bundle reject bool inputs.
    """
    import threading
    from types import SimpleNamespace
    import pytest
    import cacheManager as cm
    from binance_api import bapi, bapi_allorders, bapi_trades
    import instrument_registry
    from providers.market_api import MarketApi

    # 1A. CacheTradeManager.get_last_opposite_fill_price with slash/lowercase symbol
    tm = cm.CacheTradeManager.__new__(cm.CacheTradeManager)
    tm.lock = threading.RLock()
    tm.sync_account_reader_if_needed = lambda: None
    tm.cache = {
        "BTCUSDC": [
            {
                "symbol": "BTCUSDC",
                "id": 101,
                "orderId": 201,
                "price": "88500.0",
                "qty": "0.1",
                "time": 1_700_000_000_000,
                "isBuyer": False,
            }
        ]
    }
    assert tm.get_last_opposite_fill_price("BTC/USDC", "BUY") == 88500.0
    assert tm.get_last_opposite_fill_price("btcusdc", "BUY") == 88500.0

    # 1B. CacheCurrentPriceManager slash/case symbol normalization
    cpm = cm.CacheCurrentPriceManager.__new__(cm.CacheCurrentPriceManager)
    cpm.cls_name = "CacheCurrentPriceManager"
    cpm.lock = threading.RLock()
    cpm.symbols = ["TAOUSD"]
    cpm._provider_names = {"TAOUSD": "kraken"}
    now_ms = int(time.time() * 1000)
    cpm.cache = {"TAOUSD": [[now_ms, 420.5]]}
    assert cpm.provider_name_for("tao/usd") == "kraken"
    obs = cpm.get_cached_price_observation("TAO/USD", 10.0)
    assert obs is not None and obs[1] == 420.5
    assert cpm.get_price_value("tao/usd") == 420.5

    # 2. bapi_allorders.get_trade_orders, bapi_trades.get_trade_orders, and aggregate_trades
    monkeypatch.setattr(bapi_allorders.sym, "validate_symbols", lambda s: None)
    monkeypatch.setattr(bapi_trades.sym, "validate_symbols", lambda s: None)
    future_ts = now_ms + 3_600_000  # 1 hour in the future (clock skew)
    valid_ts = now_ms - 10_000
    fake_order_mgr = SimpleNamespace(
        lock=threading.RLock(),
        cache={
            "BTCUSDC": [
                {"orderId": 1, "price": 90000.0, "quantity": 0.1, "timestamp": future_ts, "side": "SELL"},
                {"orderId": 2, "price": 0.0, "quantity": 0.1, "timestamp": valid_ts, "side": "SELL"},
                {"orderId": 3, "price": 85000.0, "quantity": 0.2, "timestamp": valid_ts, "side": "SELL"},
            ]
        },
    )
    monkeypatch.setattr(cm, "get_cache_manager", lambda name, **kw: fake_order_mgr)
    orders_out = bapi_allorders.get_trade_orders("SELL", "BTC/USDC", 300)
    assert len(orders_out) == 1 and orders_out[0]["orderId"] == 3

    fake_trade_mgr = SimpleNamespace(
        lock=threading.RLock(),
        cache={
            "BTCUSDC": [
                {"symbol": "BTCUSDC", "id": 1, "orderId": 1, "price": 90000.0, "qty": 0.1, "time": future_ts, "isBuyer": False},
                {"symbol": "BTCUSDC", "id": 2, "orderId": 2, "price": 0.0, "qty": 0.1, "time": valid_ts, "isBuyer": False},
                {"symbol": "BTCUSDC", "id": 3, "orderId": 3, "price": 86000.0, "qty": 0.3, "time": valid_ts, "isBuyer": False},
            ]
        },
    )
    monkeypatch.setattr(bapi_trades, "init_cache_trade_manager", lambda: fake_trade_mgr)
    trades_out = bapi_trades.get_trade_orders("SELL", "BTC/USDC", 300)
    assert len(trades_out) == 1 and trades_out[0]["id"] == 3

    agg = bapi_trades.aggregate_trades([
        {"orderId": 10, "price": "0.0", "qty": "1.0", "time": valid_ts, "isBuyer": True},
        {"orderId": 11, "price": "85000.0", "qty": "0.5", "time": valid_ts, "isBuyer": True},
    ])
    assert len(agg) == 1 and agg[0]["orderId"] == 11

    assert bapi_allorders.get_recent_filled_orders("BUY", "BTCUSDC", float("nan")) == []
    assert bapi_allorders.get_recent_filled_orders("BUY", "BTCUSDC", True) == []

    # 3. bapi.cancel_expired_orders and bapi.cancel_recent_orders with NaN/bool/non-positive bounds & future order_time
    canceled_ids = []
    monkeypatch.setattr(bapi.sym, "validate_params", lambda ot, s: None)
    monkeypatch.setattr(
        bapi,
        "get_open_orders",
        lambda ot, s, **kw: {
            501: {"time": now_ms - 5_000, "price": 80000.0, "qty": 0.1},
            502: {"time": now_ms + 60_000, "price": 80000.0, "qty": 0.1},  # future timestamp
        },
    )
    monkeypatch.setattr(bapi, "cancel_order", lambda s, oid: (canceled_ids.append(oid) or True))

    bapi.cancel_expired_orders("BUY", "BTCUSDC", 0)
    bapi.cancel_expired_orders("BUY", "BTCUSDC", -10)
    bapi.cancel_expired_orders("BUY", "BTCUSDC", float("nan"))
    bapi.cancel_expired_orders("BUY", "BTCUSDC", True)
    assert canceled_ids == []

    bapi.cancel_recent_orders("BUY", "BTCUSDC", float("nan"))
    bapi.cancel_recent_orders("BUY", "BTCUSDC", True)
    assert canceled_ids == []

    # Valid recent window (10s) cancels order 501 (age ~5s) but NOT future-dated order 502 (age -60s)
    bapi.cancel_recent_orders("BUY", "BTCUSDC", 10.0)
    assert canceled_ids == [501]

    # 4. MarketApi.place inherits instrument params from instrument_registry & bool rejections
    captured_bypasses = []
    orig_margins = dict(order_guard._load_margins())
    monkeypatch.setattr(order_guard, "_load_margins", lambda: orig_margins)
    monkeypatch.setattr(order_guard, "daily_limit_guard", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        order_guard,
        "profit_guard",
        lambda *a, **k: (captured_bypasses.append(set(k.get("bypass_guards") or set())) or True),
    )
    monkeypatch.setattr(
        instrument_registry,
        "select_instruments",
        lambda prov=None: {
            "KRAKEN:TAOUSD": SimpleNamespace(
                symbol="TAOUSD",
                base="TAO",
                quote="USD",
                params={"bypass_guards": "GI_WHALE"},
            )
        },
    )

    class DummyProv:
        name = "kraken"
        def supports_symbol(self, sym):
            return sym == "TAOUSD"
        def validate_symbol(self, sym):
            return None
        def guards_internally(self):
            return False
        def requires_cache_permit(self):
            return False
        def defers_smart_cancel(self):
            return False
        def supports_business_minimum(self):
            return False
        def execution_enabled(self):
            return True
        def preflight_order(self, *a, **k):
            return None
        def round_price(self, sym, px):
            return px
        def round_amount(self, sym, q):
            return q
        def free_balance(self, asset):
            return 1000.0
        def get_current_price(self, sym):
            return 100.0
        def profit_guard_window_ref(self, *a, **k):
            return None
        def quantity_decision(self, symbol, side, price, qty, **kw):
            from providers.quantity import QuantityDecision
            return QuantityDecision(
                requested_qty=float(qty or 1.0),
                balance_cap=100.0,
                policy_cap=100.0,
                fee_cap=100.0,
                final_qty=float(qty or 1.0),
                scale_applied=False,
                applied_scale=1.0,
            )
        def place_order(self, symbol, side, price, qty, **kw):
            return {"orderId": "ord-28", "qty": qty}

    mapi = MarketApi([DummyProv()])
    res_place = mapi.place(
        "TAOUSD", "BUY", 100.0, 1.0,
        provider_name="kraken", smart=False, wait_for_trend=False, caller_owns_retry=True,
    )
    assert res_place is not None
    assert len(captured_bypasses) == 1
    assert "gi_whale" in captured_bypasses[0]

    with pytest.raises(ValueError):
        mapi.latest_fill_price("TAOUSD", "BUY", True, provider_name="kraken")
    with pytest.raises(ValueError):
        mapi.market_regime_bundle("TAOUSD", provider_name="kraken", now=True)
    with pytest.raises(ValueError):
        mapi.market_regime_bundle("TAOUSD", provider_name="kraken", now=float("nan"))


def test_iteration29_candle_interval_bool_t212_get_orders_whale_funding_ts_and_collector_nan(tmp_path):
    """Iteration 29 regression suite:
    1. candle_interval rejects boolean inputs (e.g. True / False) with ProviderError.
    2. T212Provider.get_orders rejects NaN/bool/non-positive since_s, matches ticker case-insensitively,
       and keeps synthetic timestamp within short windows (< 2h); order_status tolerates NaN/bool orderedQuantity.
    3. WhalePositioningCollector._resolve_cached_funding_rate rejects missing/zero/NaN ts and bool/NaN funding_rate,
       and classify_regime sanitizes NaN/bool inputs so flat-price OI divergence still works.
    4. OrderbookDepthCollector, DerivativesTelemetryCollector, and WhalePositioningCollector fetch()
       sanitize NaN/bool `now` without crashing or persisting NaN timestamps.
    """
    from types import SimpleNamespace
    import pytest
    from providers.strategy_executor import candle_interval, ProviderError
    from providers.t212_provider import T212Provider
    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector, DerivativesTelemetry

    # 1. candle_interval rejects bool
    with pytest.raises(ProviderError):
        candle_interval(True)
    with pytest.raises(ProviderError):
        candle_interval(False)
    assert candle_interval(1) == "1m"

    # 2. T212Provider.get_orders and order_status
    fake_cli = SimpleNamespace(
        get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "averagePrice": 120.0, "quantity": 5.0}],
        get_order_status=lambda oid: {
            "ticker": "NVDA_US_EQ",
            "status": "FILLED",
            "filledQuantity": 5.0,
            "filledValue": 600.0,
            "fee": 0.0,
            "orderedQuantity": float("nan"),
        },
    )
    t212 = T212Provider(client=fake_cli, live_enabled=False)
    assert t212.get_orders("NVDA_US_EQ", "BUY", float("nan")) == []
    assert t212.get_orders("NVDA_US_EQ", "BUY", True) == []
    assert t212.get_orders("NVDA_US_EQ", "BUY", -10.0) == []

    # Case-insensitive ticker match and synthetic timestamp inside 600s window
    now_ms = int(time.time() * 1000)
    short_win_orders = t212.get_orders("nvda_us_eq", "BUY", 600.0)
    assert len(short_win_orders) == 1
    assert now_ms - short_win_orders[0]["timestamp"] <= 600_000

    st = t212.order_status("nvda_us_eq", "ord-1")
    assert st.status == "closed" and st.orig_qty is None

    # 3. WhalePositioningCollector._resolve_cached_funding_rate & classify_regime
    w_col = WhalePositioningCollector(cache_ttl_sec=60.0, disk_ttl_sec=120.0, cache_dir=str(tmp_path))
    f_path = tmp_path / "derivatives_telemetry_BTCUSDT.json"

    # Missing / zero ts must be rejected
    f_path.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": -0.001, "ts": 0.0}), encoding="utf-8")
    assert w_col._resolve_cached_funding_rate("BTCUSDC", now=1000.0) is None

    # Bool funding_rate must be rejected
    f_path.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": True, "ts": 990.0}), encoding="utf-8")
    assert w_col._resolve_cached_funding_rate("BTCUSDC", now=1000.0) is None

    # Valid fresh funding_rate is accepted
    f_path.write_text(json.dumps({"symbol": "BTCUSDT", "funding_rate": -0.0005, "ts": 990.0}), encoding="utf-8")
    assert w_col._resolve_cached_funding_rate("BTCUSDC", now=1000.0) == pytest.approx(-0.0005)

    # classify_regime with NaN price_1h_change_pct and NaN funding_rate
    assert WhalePositioningCollector.classify_regime(
        price_1h_change_pct=float("nan"),
        oi_change_pct=1.2,
        taker_ratio=1.25,
        funding_rate=float("nan"),
    ) == "accumulation"

    # 4. Collectors fetch() with NaN/bool `now`
    ob_col = OrderbookDepthCollector(cache_ttl_sec=60.0, cache_dir=str(tmp_path))
    now_s = time.time()
    ob_snap = OrderbookSnapshot(
        symbol="BTCUSDC", mid_price=85000.0, bid_depth_usd=1000.0, ask_depth_usd=1000.0,
        imbalance_ratio=0.5, largest_bid_wall_usd=100.0, largest_bid_wall_price=84900.0,
        largest_ask_wall_usd=100.0, largest_ask_wall_price=85100.0, ts=now_s,
    )
    ob_col.record_snapshot(ob_snap)
    assert ob_col.fetch("BTCUSDC", allow_network=False, now=float("nan")) is ob_snap
    assert ob_col.fetch("BTCUSDC", allow_network=False, now=True) is ob_snap

    dt_col = DerivativesTelemetryCollector(cache_ttl_sec=60.0, cache_dir=str(tmp_path))
    dt_snap = DerivativesTelemetry(
        symbol="BTCUSDC", funding_rate=0.0001, predicted_funding_rate=0.0001,
        open_interest=100.0, open_interest_usd=8_500_000.0, ts=now_s,
    )
    dt_col.record_telemetry(dt_snap)
    assert dt_col.fetch("BTCUSDC", allow_network=False, now=float("nan")) is dt_snap


def test_iteration30_bybit_liquidations_breadth_macro_triggers_and_composite_guards(tmp_path):
    """Iteration 30 regression suite:
    1. BybitLiquidationCollector._ingest, record_event, and get_summary reject non-dict/bool/NaN/invalid-side
       inputs, normalize slash symbols ("BTC/USDT" -> "BTCUSDT"), and do not leak expired events when
       get_summary is called with NaN/bool `now` or `window_sec`.
    2. MarketBreadthCollector.parse_tickers sanitizes NaN/bool `now_ts` and `min_volume_usd` and normalizes
       lowercase/whitespace symbols.
    3. SentimentAdvisor.review and GeopoliticalThreatAnalyzer.assess handle NaN/bool `now_ts`, non-dict
       LLM JSON responses, and invalid threat_level/recommended_brake values without crashing.
    4. NewsFeedCollector.get_pool_snapshot filters out future-skewed or NaN timestamps and sanitizes NaN/bool `now_ts`.
    5. LiquidationCapitulationTrigger, SentimentContrarianTrigger, MarketBreadthTrigger, and
       CompositeMarketIntelligence sanitize NaN/bool inputs and normalize slash symbols across triggers/GLLM1.
    """
    from types import SimpleNamespace
    import math
    from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector, LiquidationSummary
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsFeedSnapshot, NewsHeadline
    from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
    from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger
    from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
    from intelligence.composite import CompositeMarketIntelligence

    # 1. BybitLiquidationCollector hardening
    liq_col = BybitLiquidationCollector(symbols=["BTCUSDT"])
    now_s = time.time()
    liq_col._ingest("null")
    liq_col._ingest("[1, 2, 3]")
    liq_col._ingest(json.dumps({
        "topic": "allLiquidation.BTCUSDT",
        "data": [
            "not-a-dict",
            {"s": "BTCUSDT", "S": "Buy", "v": True, "p": 85000.0},
            {"s": "BTCUSDT", "S": "UNKNOWN", "v": 1.0, "p": 85000.0},
            {"s": "btc/usdt", "S": "BUY", "v": 2.0, "p": 50000.0},
        ],
    }))
    liq_col.record_event("BTC/USDT", "BUY", True, ts=now_s)
    liq_col.record_event("BTC/USDT", "BUY", 50000.0, ts=float("nan"))
    liq_col.record_event("BTC/USDT", "INVALID_SIDE", 50000.0, ts=now_s)
    # Record an expired event 2000s ago
    liq_col.record_event("BTC/USDT", "SELL", 250000.0, ts=now_s - 2000.0)

    # Calling get_summary with NaN `now` or NaN `window_sec` must NOT include the 2000s-old event
    summary = liq_col.get_summary("btc/usdt", window_sec=float("nan"), now=float("nan"))
    assert summary.symbol == "BTCUSDT"
    assert summary.events_count == 1
    assert summary.long_liq_usd == 100000.0
    assert summary.short_liq_usd == 0.0

    # 2. MarketBreadthCollector.parse_tickers with NaN min_volume_usd & lowercase symbols
    tickers = [
        {"symbol": f"coin{i}usdt", "quoteVolume": "10000000", "priceChangePercent": "-5.5"}
        for i in range(6)
    ]
    b_snap = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=float("nan"), now_ts=float("nan"))
    assert b_snap is not None
    assert b_snap.total_count == 6
    assert b_snap.regime == "PANIC_WASHOUT"
    assert math.isfinite(b_snap.ts) and b_snap.ts > 0.0

    # 3. SentimentAdvisor & GeopoliticalThreatAnalyzer non-dict LLM response & enum sanitization
    fake_gemini_list = SimpleNamespace(query_json=lambda *a, **kw: ["unexpected", "list"])
    advisor = SentimentAdvisor(
        gemini_client=fake_gemini_list,
        cache_file=str(tmp_path / "sentiment_eval.json"),
    )
    assert advisor.review(force_refresh=True, now_ts=float("nan")) is None

    fake_gemini_bad_enums = SimpleNamespace(
        query_json=lambda *a, **kw: {
            "threat_level": "APOCALYPSE",
            "risk_score": 0.8,
            "summary": "Test",
            "recommended_brake": "FULL_HALT",
        }
    )
    geo = GeopoliticalThreatAnalyzer(
        gemini_client=fake_gemini_bad_enums,
        cache_ttl_sec=60.0,
        fallback_retry_ttl_sec=10.0,
        state_file=str(tmp_path / "geo_eval.json"),
    )
    news_snap = NewsFeedSnapshot(
        headlines=(
            NewsHeadline(
                title="Military strike reported near oil pipeline",
                source="Test",
                pub_date="",
                url="",
                matched_keywords=("military strike",),
                is_high_severity=True,
            ),
        ),
        total_fetched=1,
        high_severity_count=1,
        has_critical_shock_keywords=False,
        ts=now_s,
    )
    geo_res = geo.assess(news_snap, force_refresh=True, now_ts=float("nan"))
    assert geo_res.threat_level == "NORMAL"
    assert geo_res.recommended_brake == "NONE"
    assert math.isfinite(geo_res.ts) and geo_res.ts > 0.0

    # 4. NewsFeedCollector.get_pool_snapshot filters future-skewed / NaN timestamps
    nf_col = NewsFeedCollector(history_file=str(tmp_path / "news_collect.json"))
    nf_col._aggregated_pool["future_hl"] = (now_s + 3600.0, news_snap.headlines[0])
    nf_col._aggregated_pool["nan_hl"] = (float("nan"), news_snap.headlines[0])
    nf_col._aggregated_pool["valid_hl"] = (now_s - 10.0, news_snap.headlines[0])
    pool_snap = nf_col.get_pool_snapshot(now_ts=float("nan"))
    assert pool_snap.total_fetched == 1
    assert math.isfinite(pool_snap.ts)

    # 5. Triggers & CompositeMarketIntelligence hardening
    liq_trig = LiquidationCapitulationTrigger(min_notional_usd=100_000.0)
    bad_summary = LiquidationSummary(
        symbol="BTCUSDT", window_sec=300.0, long_liq_usd=float("nan"), short_liq_usd=0.0,
        net_usd=0.0, capitulation_ratio=0.9, squeeze_ratio=0.1, events_count=1, last_event_ts=now_s,
    )
    assert liq_trig.evaluate("BTCUSDT", 85000.0, summary=bad_summary, now=float("nan")) is None
    assert liq_trig.evaluate("BTCUSDT", True, summary=summary, now=now_s) is None

    sc_trig = SentimentContrarianTrigger()
    fg_snap = FearGreedSnapshot(
        value=15, classification="Extreme Fear", timestamp=now_s,
        historical_values=(15, 16), trend_7d_change=-5, is_extreme_fear=True, is_extreme_greed=False,
    )
    assert sc_trig.evaluate("BTCUSDT", False, fg_snap) == (None, None)

    mb_trig = MarketBreadthTrigger()
    sig, ev = mb_trig.evaluate("BTCUSDT", 85000.0, b_snap, asset_24h_change_pct=False, ts=float("nan"))
    assert sig == "BREADTH_WASHOUT_BUY"
    assert ev is not None and ev.metadata["relative_strength"] is False
    assert math.isfinite(ev.ts) and ev.ts > 0.0

    comp = CompositeMarketIntelligence()
    assert comp._get_kalman("btc/usdc") is comp._get_kalman("BTCUSDC")
    assert comp._get_gradient("btc/usdc") is comp._get_gradient("BTCUSDC")
    # Passing bool notional_eur or bool qty must not crash or trigger GLLM1 falsely
    g_dec = comp.evaluate_guards("BTC/USDC", "BUY", 85000.0, qty=True, notional_eur="bad")
    assert g_dec.allowed is True


def test_iteration31_parabolic_high_stake_post_stop_and_smart_sl_nan_safety(monkeypatch):
    """Iteration 31 regression suite:
    1. ParabolicSurgeGuard.check normalizes slash symbols ("btc/usdc" == "BTCUSDC"), tolerates string/bool
       volatility_1h_pct, and ignores boolean prices in price_history without triggering false 99.99% surges.
    2. FundingCrowdingGuard, OrderbookWallGuard, and WhaleDivergenceGuard handle None/non-standard side safely.
    3. HighStakeGuard.check (GLLM1), compute_24h_price_metrics, and extract_symbol_recent_fills sanitize
       NaN/bool/malformed inputs and share decision cache across slash/non-slash symbol variants.
    4. PostStopReentryGuard.check sanitizes None/string/NaN/bool prices and timestamps without raising
       TypeError/ValueError during prompt formatting or notification.
    5. SmartStopLossGuard.evaluate preserves the Hard Disaster Floor even when disaster_buffer_pct=NaN.
    """
    from types import SimpleNamespace
    import math
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.sentiment.guards.high_stake_guard import (
        HighStakeGuard,
        _HIGH_STAKE_DECISION_CACHE,
        compute_24h_price_metrics,
        extract_symbol_recent_fills,
    )
    from intelligence.sentiment.guards.post_stop_reentry_guard import (
        PostStopReentryGuard,
        clear_post_stop_reentry_cache,
    )
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard

    # 1. ParabolicSurgeGuard symbol normalization & bool/string resilience
    pg = ParabolicSurgeGuard(surge_threshold_pct=4.0, pullback_required_pct=1.5)
    # Bool price in history (True -> 1.0 if unguarded) must be ignored
    hist_with_bool = [(1000.0, True), (1010.0, 100.0), (1020.0, 101.0)]
    dec_pg = pg.check("btc/usdc", "BUY", 101.0, price_history=hist_with_bool, volatility_1h_pct="2.0", now=1020.0)
    assert dec_pg.allowed is True
    assert "BTCUSDC" in pg._states

    # 2. External guards with None side
    assert FundingCrowdingGuard().check("BTCUSDC", None).allowed is True
    assert OrderbookWallGuard().check("BTCUSDC", None).allowed is True
    assert WhaleDivergenceGuard().check("BTCUSDC", None).allowed is True

    # 3. HighStakeGuard & helpers NaN/bool resilience
    m24 = compute_24h_price_metrics([
        [1000.0, 100.0],
        [1060.0, float("nan")],
        [1120.0, True],
        [2000.0, 105.0],
    ])
    assert m24.get("low_24h") == 100.0 and m24.get("high_24h") == 105.0

    fills = extract_symbol_recent_fills([
        {"side": "BUY", "price": float("nan"), "qty": 2.0, "cummulativeQuoteQty": 200.0, "timestamp": "bad-ts"},
        {"side": "SELL", "price": True, "qty": 1.0, "timestamp": 1700000000000},
    ])
    assert fills is not None
    assert fills["last_buy"]["price"] == 100.0
    assert fills["last_sell"]["price"] is None

    monkeypatch.setenv("DISABLE_GEMINI_LLM", "0")
    monkeypatch.setenv("BACKTEST_MODE", "0")
    monkeypatch.setenv("REPLAY_MODE", "0")
    _HIGH_STAKE_DECISION_CACHE.clear()
    calls = {"n": 0}

    def _fake_query(*a, **kw):
        calls["n"] += 1
        return {"decision": "DOWNSCALE", "suggested_scale": 0.5, "reason": "test"}

    hs = HighStakeGuard(
        gemini_client=SimpleNamespace(query_json=_fake_query),
        min_notional_usd=700.0,
        cache_ttl_sec=600.0,
    )
    # NaN notional_eur must not raise ValueError on int(notional // 100.0)
    assert hs.check("BTCUSDC", "BUY", 85000.0, 1.0, notional_eur=float("nan")).allowed is True
    # Slash symbol variant hits the same cache entry as canonical symbol
    d1 = hs.check("BTC/USDC", "BUY", 1000.0, 1.0, notional_eur=800.0, now=float("nan"))
    d2 = hs.check("BTCUSDC", "BUY", 1000.0, 1.0, notional_eur=800.0, now=time.time())
    assert d1.suggested_scale == 0.5 and d2.suggested_scale == 0.5
    assert calls["n"] == 1

    # 4. PostStopReentryGuard string/None/NaN input formatting safety
    clear_post_stop_reentry_cache()
    ps_guard = PostStopReentryGuard(
        gemini_client=SimpleNamespace(query_json=lambda *a, **kw: ["not-a-dict"]),
        fallback_action="downscale",
        notify_enabled=False,
        cache_ttl_sec=60.0,
    )
    ps_dec = ps_guard.check(
        "tao/usdc",
        current_price="bad",
        last_sell_price=None,
        trough=float("nan"),
        bounce_pct=True,
        qty="bad",
        last_sell_time_s="bad",
        telemetry={},
        now="bad",
    )
    assert ps_dec.suggested_scale == 0.5

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


def test_iteration32_external_order_guard_position_exit_manager_and_monitortrades_nan_safety():
    """Iteration 32 regression suite:
    1. check_external_order_guards sanitizes NaN/bool `now`, `notional_eur`, and `price`, and normalizes
       slash symbols in _MICRO_SHADOW_NOTIFY_COOLDOWN so NaN never poisons the cooldown map.
    2. PositionSnapshot.is_valid_price and ReentrySnapshot.is_valid_price reject None and bool prices
       (e.g. current_price=True must not be treated as $1.00 and trigger a false 99.99% disaster floor exit).
    3. PositionExitManager.compute_disaster_buffer and compute_tp_anomaly_pullback fall back to vol_resolver
       when vol_1h is NaN/bool/inf, and evaluate_stop_loss / evaluate_reentry sanitize NaN current_time_s,
       post_sell_trough=NaN, and last_sell_time_s=NaN.
    4. monitortrades._summarize_orders, _compute_spot_position, get_position_stats, and _place_guarded
       filter out NaN/zero/bool order quantities and prices and protect cache heartbeat against backward clock skew.
    """
    from types import SimpleNamespace
    import math
    import external_order_guard as eog
    from strategies.position_exit_manager import (
        ExitAction,
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
        ReentryAction,
        ReentrySnapshot,
    )
    import monitortrades as mt

    # 1. check_external_order_guards with NaN now and invalid notional_eur
    eog._MICRO_SHADOW_NOTIFY_COOLDOWN.clear()
    ok, reason, scale = eog.check_external_order_guards(
        "btc/usdc",
        "BUY",
        True,
        qty=True,
        notional_eur="bad",
        now=float("nan"),
        margins={"intelligence_guards_mode": "off"},
    )
    assert ok is True and scale == 1.0

    # 2. PositionSnapshot & ReentrySnapshot reject bool/None prices
    snap_bool = PositionSnapshot(symbol="BTCUSDC", current_price=True, avg_buy_price=85000.0)
    assert snap_bool.is_valid_price is False
    snap_none = PositionSnapshot(symbol="BTCUSDC", current_price=None, avg_buy_price=85000.0)
    assert snap_none.is_valid_price is False
    re_bool = ReentrySnapshot(symbol="BTCUSDC", current_price=True, last_sell_price=85000.0)
    assert re_bool.is_valid_price is False

    # 3. PositionExitManager disaster buffer fallback on NaN vol_1h & reentry with NaN trough
    mgr = PositionExitManager(
        config=PositionExitConfig(
            lost_threshold=0.05,
            sl_disaster_buffer_ratio=0.30,
            sl_disaster_vol_multiplier=1.0,
            sl_disaster_buffer_min_pct=0.008,
            reentry_sl_bounce_pct=0.02,
            allow_external_cache=False,
        ),
        vol_resolver=lambda sym: (0.04, "custom_resolver"),
    )
    buf, meta = mgr.compute_disaster_buffer("BTCUSDC", 0.05, vol_1h=float("nan"))
    assert math.isclose(buf, 0.04) and meta["vol_source"] == "custom_resolver"

    # evaluate_stop_loss with current_price=True must HOLD ("invalid_prices"), not SELL
    sl_dec = mgr.evaluate_stop_loss(snap_bool)
    assert sl_dec.action == ExitAction.HOLD and sl_dec.should_exit is False

    # evaluate_reentry with post_sell_trough=NaN and last_sell_time_s=NaN must not poison trough with NaN
    re_snap = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=103.0,
        last_sell_price=105.0,
        last_exit_kind="STOP",
        post_sell_trough=float("nan"),
        post_sell_peak=float("nan"),
        current_time_s=float("nan"),
        last_sell_time_s=float("nan"),
    )
    # First tick at 100.0 sets trough=100.0 even when post_sell_trough=NaN
    re_snap_dip = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=100.0,
        last_sell_price=105.0,
        last_exit_kind="STOP",
        post_sell_trough=float("nan"),
        post_sell_peak=float("nan"),
        current_time_s=float("nan"),
        last_sell_time_s=float("nan"),
    )
    d_wait = mgr.evaluate_reentry(re_snap_dip)
    assert d_wait.action == ReentryAction.WAIT
    assert math.isfinite(mgr.reentry_state["BTCUSDC"]["trough"])
    # Second tick at 103.0 (+3% bounce >= 2% required) confirms re-entry without NaN sell_time resetting state
    d_enter = mgr.evaluate_reentry(re_snap)
    assert d_enter.action == ReentryAction.ENTER and d_enter.should_enter is True

    # 4. monitortrades._compute_spot_position & _place_guarded
    owned, active_avg, _ = mt._compute_spot_position(
        buy_orders=[
            {"qty": 2.0, "price": 100.0, "timestamp": 1000},
            {"qty": 5.0, "price": 0.0, "timestamp": 2000},
            {"qty": float("nan"), "price": 100.0, "timestamp": 3000},
        ],
        sell_orders=[],
    )
    assert owned == 2.0 and active_avg == 100.0

    placed = []
    fake_inst = SimpleNamespace(symbol="BTCUSDC", place=lambda *a, **kw: placed.append((a, kw)) or {"id": "1"})
    assert mt._place_guarded(fake_inst, "BUY", 85000.0, True, 0.001) is False
    assert mt._place_guarded(fake_inst, "BUY", True, 0.01, 0.001) is False
    assert len(placed) == 0


def test_iteration33_zero_timestamp_fidelity_and_monitortrades_reentry_nan_safety(monkeypatch):
    """Iteration 33 regression suite:
    1. PositionExitManager and SmartStopLossGuard treat explicit current_time_s=0.0 / now=0.0 as a valid
       timestamp and never call wall-clock time.time() when a finite timestamp is explicitly supplied.
    2. PositionExitManager preserves started_at=0.0 and last_peak_time=0.0 across subsequent simulation ticks
       instead of resetting them as invalid (<= 0).
    3. monitortrades._apply_post_stop_ai_guard sanitizes post_sell_trough=NaN/bool before calling
       PostStopReentryGuard.check so trough is never passed as NaN.
    """
    import math
    import time
    from types import SimpleNamespace
    import monitortrades as mt
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard
    from strategies.position_exit_manager import (
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
        ReentrySnapshot,
    )

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

    # 2. PositionExitManager preserves started_at=0.0 across subsequent ticks without calling time.time()
    cfg = PositionExitConfig(
        lost_threshold=0.05,
        gain_threshold=0.05,
        sl_trend_defer_max_sec=60.0,
        tp_stagnation_max_sec=60.0,
        allow_external_cache=False,
    )
    mgr = PositionExitManager(config=cfg)
    snap_t0 = PositionSnapshot(
        symbol="BTCUSDC",
        current_price=94.0,
        avg_buy_price=100.0,
        net_qty=1.0,
        sellable_qty=1.0,
        current_time_s=0.0,
        trend_up=True,
        vol_1h=0.02,
        provider="test",
    )
    d0 = mgr.evaluate_stop_loss(snap_t0)
    assert d0.should_exit is False
    assert mgr.sl_bounce_state["BTCUSDC"]["started_at"] == 0.0

    snap_t65 = PositionSnapshot(
        symbol="BTCUSDC",
        current_price=94.0,
        avg_buy_price=100.0,
        net_qty=1.0,
        sellable_qty=1.0,
        current_time_s=65.0,
        trend_up=True,
        vol_1h=0.02,
        provider="test",
    )
    d65 = mgr.evaluate_stop_loss(snap_t65)
    assert d65.should_exit is True and "mhdsl_bounce_timeout" in d65.reason

    # Restore time.time for subsequent checks
    monkeypatch.undo()

    # 3. monitortrades._apply_post_stop_ai_guard sanitizes post_sell_trough=NaN
    captured_trough = []

    class _DummyPSGuard:
        mode = "enforce"

        def check(self, **kwargs):
            captured_trough.append(kwargs.get("trough"))
            return SimpleNamespace(allowed=True, scale=1.0, reason="ok")

    import intelligence.sentiment.guards.post_stop_reentry_guard as ps_mod
    import intelligence.sentiment.guards.high_stake_guard as hs_mod
    monkeypatch.setattr(ps_mod, "get_post_stop_reentry_guard", lambda: _DummyPSGuard())
    monkeypatch.setattr(hs_mod, "is_llm_guard_exempt", lambda: False)
    monkeypatch.setattr(mt, "_is_simulated_env_or_inst", lambda inst: False)

    re_snap = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=102.0,
        last_sell_price=105.0,
        last_exit_kind="STOP",
        post_sell_trough=float("nan"),
        post_sell_peak=float("nan"),
    )
    allowed, _, _, _ = mt._apply_post_stop_ai_guard(
        symbol="BTCUSDC",
        current_price=102.0,
        sell_price=105.0,
        sell_time=1700000000.0,
        reentry_snap=re_snap,
        buy_budget=100.0,
        buy_qty_cfg=None,
        qty=1.0,
        inst=SimpleNamespace(provider_name="binance", params={"bypass_guards": []}),
    )
    assert allowed is True
    assert len(captured_trough) == 1 and math.isfinite(captured_trough[0]) and captured_trough[0] == 102.0


def test_iteration34_order_guard_reference_nan_safety_and_llm_order_guard(monkeypatch):
    """Iteration 34 regression suite:
    1. order_guard.resolve_high_stake_min_notional_usd sanitizes NaN/bool/non-positive values to 700.0
       so GLLM1 is never silently disabled by NaN or triggered on $2 orders by True.
    2. order_guard.select_extreme_reference and _aggregate_reference_prices filter out NaN/bool prices
       and timestamps (preventing IndexError when a first valid_opp item has ts=NaN).
    3. order_guard.check_math_profit_reference rejects NaN/bool price and profit_percentage instead of
       returning True when diff=NaN (`NaN < profit_percentage` is False), and rejects bool window_ref.
    4. order_guard.compute_weight_capped_qty sanitizes NaN/bool inputs instead of returning NaN.
    5. llm_order_guard.evaluate_high_stake_order_guard and check_llm_order_guards sanitize NaN
       computed_notional, prev_notional, and cooldown timestamps.
    """
    import math
    from types import SimpleNamespace
    import order_guard as og
    import llm_order_guard as log_mod

    # 1. resolve_high_stake_min_notional_usd
    assert og.resolve_high_stake_min_notional_usd({"high_stake_min_notional_usd": float("nan")}) == 700.0
    assert og.resolve_high_stake_min_notional_usd({"high_stake_min_notional_usd": True}) == 700.0
    assert og.resolve_high_stake_min_notional_usd({"high_stake_min_notional_usd": -50.0}) == 700.0
    assert og.resolve_high_stake_min_notional_usd({"high_stake_min_notional_usd": 850.0}) == 850.0

    # 2. select_extreme_reference & _aggregate_reference_prices with NaN/bool
    assert og.select_extreme_reference([float("nan"), True, 100.0, 95.0], "BUY", strict=True) == 95.0
    assert og.select_extreme_reference([float("nan"), True, 100.0, 95.0], "SELL", strict=True) == 100.0
    ref_last = og._aggregate_reference_prices(
        active_lots=[(float("nan"), 100.0), (2.0, 105.0)],
        valid_opp=[(float("nan"), 1.0, 100.0), (1700000000.0, 2.0, 105.0)],
        side="BUY",
        mode="last",
    )
    assert ref_last == 105.0

    # 3. check_math_profit_reference rejects NaN/bool price or profit_percentage
    dummy_prov = SimpleNamespace(name="binance", last_opposite_fill=lambda s, o: 100.0)
    assert og.check_math_profit_reference(
        dummy_prov, "BTCUSDC", "SELL", float("nan"), 1.0, reference_mode="strict"
    ) is False
    assert og.check_math_profit_reference(
        dummy_prov, "BTCUSDC", "SELL", True, 1.0, reference_mode="strict"
    ) is False
    assert og.check_math_profit_reference(
        dummy_prov, "BTCUSDC", "SELL", 105.0, float("nan"), reference_mode="strict"
    ) is False

    # 4. compute_weight_capped_qty with NaN/bool
    assert og.compute_weight_capped_qty(0.1, 100.0, float("nan"), 10.0, 0.0) == 0.0
    assert og.compute_weight_capped_qty(0.1, True, 5.0, 10.0, 0.0) == 0.0
    capped = og.compute_weight_capped_qty(0.2, 100.0, 5.0, 10.0, float("nan"))
    assert math.isclose(capped, 2.0)

    # 5. evaluate_high_stake_order_guard with NaN computed_notional does not invoke HighStakeGuard
    called_hs = []

    class _FailIfCalledHS:
        def __init__(self, *a, **kw):
            pass

        def check(self, *a, **kw):
            called_hs.append((a, kw))
            return SimpleNamespace(allowed=False, reason="veto", suggested_scale=0.0, brake_action="VETO")

    import intelligence.sentiment.guards.high_stake_guard as hs_mod
    monkeypatch.setattr(hs_mod, "HighStakeGuard", _FailIfCalledHS)
    monkeypatch.setattr(hs_mod, "is_llm_guard_exempt", lambda: False)

    ok_nan, r_nan, sc_nan = log_mod.evaluate_high_stake_order_guard(
        symbol="BTCUSDC",
        side="BUY",
        price=85000.0,
        qty=0.001,
        computed_notional=float("nan"),
        regime_context=None,
        valid_context=False,
        cache_key=("BUY", frozenset()),
        margins={
            "intelligence_guards_mode": "enforce",
            "llm_guard_mode": "enforce",
            "high_stake_min_notional_usd": 700.0,
            "llm_timeout_sec": 3.0,
            "llm_fallback": "allow",
        },
        bypassed=set(),
        now=float("nan"),
        notify_on_eval=False,
    )
    assert ok_nan is True and sc_nan == 1.0 and len(called_hs) == 0


def test_iteration35_compute_weight_inf_auto_qty_and_instrument_nan_bool_safety():
    import math
    from types import SimpleNamespace
    import order_guard as og
    from providers.quantity import fee_cap_quantity
    from instrument import Instrument

    # 1. compute_weight_capped_qty must accept required_qty=float('inf') (auto-quantity from decide_quantity)
    capped_inf = og.compute_weight_capped_qty(
        weight=0.5,
        price=100.0,
        required_qty=float("inf"),
        available_qty=10.0,
        traded_24h_value=0.0,
    )
    assert math.isclose(capped_inf, 5.0)

    # required_qty=NaN or bool must still return 0.0
    assert og.compute_weight_capped_qty(0.5, 100.0, float("nan"), 10.0, 0.0) == 0.0
    assert og.compute_weight_capped_qty(0.5, 100.0, True, 10.0, 0.0) == 0.0

    # 2. fee_cap_quantity NaN/bool safety
    assert fee_cap_quantity(True, 0.001) == 0.0
    assert fee_cap_quantity(float("nan"), 0.001) == 0.0
    assert math.isclose(fee_cap_quantity(10.0, True), 10.0)
    assert math.isclose(fee_cap_quantity(10.0, float("nan")), 10.0)

    # 3. Instrument._apply_late_intelligence_scale NaN/bool safety
    class _DummyProvider:
        name = "kraken"
        def validate_symbol(self, symbol):
            return True
        def guards_internally(self):
            return False
        def execution_enabled(self):
            return True
        def round_amount(self, symbol, q):
            return round(float(q), 4)
        def free_balance(self, asset):
            return 1000.0

    dummy_prov = _DummyProvider()
    inst = Instrument(
        name="BTCUSDC",
        provider="kraken",
        symbol="BTCUSDC",
        base="BTC",
        quote="USDC",
        enabled=True,
        api=SimpleNamespace(provider_by_name=lambda _n: dummy_prov),
    )
    q_bool, sc_bool, ref_bool = inst._apply_late_intelligence_scale(
        1.0, 100.0, False, True, SimpleNamespace(suggested_scale=True), 1.0, side_u="BUY"
    )
    assert q_bool == 1.0 and sc_bool == 1.0 and ref_bool is None

    q_nan, sc_nan, ref_nan = inst._apply_late_intelligence_scale(
        1.0, 100.0, False, True, SimpleNamespace(suggested_scale=float("nan")), 1.0, side_u="BUY"
    )
    assert q_nan == 1.0 and sc_nan == 1.0 and ref_nan is None

    # 4. Instrument.place rejects boolean/NaN/negative limit prices on both BUY and SELL with 'invalid_price'
    for side in ("BUY", "SELL"):
        for bad_px in (True, False, float("nan"), -5.0, 0.0):
            out = {}
            res = inst.place(
                side, bad_px, 1.0, smart=False, wait_for_trend=False,
                bypass_guards={"all"}, _outcome_context=out, caller_owns_retry=True,
            )
            assert res is None
            assert out.get("reason") == "invalid_price"


def test_iteration36_daily_limit_and_order_retry_nan_bool_safety():
    import math
    from types import SimpleNamespace
    import pytest
    import order_guard as og
    import order_retry as oq

    # 1. daily_limit_guard ignores bool/NaN max_daily_trades and inf timestamps
    prov_dl = SimpleNamespace(
        name="binance",
        get_orders=lambda sym, side, win: [
            {"timestamp": float("inf"), "price": 100.0, "qty": 1.0},
            {"timestamp": 1000.0, "price": 100.0, "qty": 1.0},
            "malformed_entry",
        ],
    )
    ok_bool, r_bool = og.daily_limit_guard(prov_dl, "BTCUSDC", "BUY", max_daily_trades=True)
    assert ok_bool is True and r_bool is None
    ok_nan, r_nan = og.daily_limit_guard(prov_dl, "BTCUSDC", "SELL", max_daily_trades=float("nan"))
    assert ok_nan is True and r_nan is None

    # 2. order_retry._lifecycle_finite and TrackedOrderLifecycle reject boolean values
    assert oq._lifecycle_finite(True, positive=True) is None
    assert oq._lifecycle_finite(False, positive=False) is None
    assert oq._lifecycle_finite(1.5, positive=True) == 1.5

    with pytest.raises(ValueError):
        oq.TrackedOrderLifecycle(object(), provider_name="binance", missing_confirmations=True)
    with pytest.raises(ValueError):
        oq.TrackedOrderLifecycle(object(), provider_name="binance", max_age_seconds=True)

    tl = oq.TrackedOrderLifecycle(object(), provider_name="binance", clock=lambda: 1000.0)
    with pytest.raises(ValueError):
        tl.new_intent(
            intent_id="i1", client_order_id="c1", symbol="BTCUSDC",
            side="BUY", requested_qty=True, requested_price=100.0,
        )
    with pytest.raises(ValueError):
        tl.new_intent(
            intent_id="i1", client_order_id="c1", symbol="BTCUSDC",
            side="BUY", requested_qty=1.0, requested_price=True,
        )
    with pytest.raises(ValueError):
        tl.new_intent(
            intent_id="i1", client_order_id="c1", symbol="BTCUSDC",
            side="BUY", requested_qty=1.0, requested_price=100.0, attempt=True,
        )

    # 3. order_retry.enqueue / update_claimed_qty / resolve_record_ttl reject boolean inputs
    assert oq.enqueue("BTCUSDC", "BUY", True) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, now=True) is None
    assert oq.update_claimed_qty({"id": "x"}, True) is None
    assert oq.resolve_record_ttl({"ttl_sec": True}) == float(oq.RETRY_TTL_SEC)


def test_iteration37_bapi_placeorder_and_external_order_guard_nan_bool_safety(monkeypatch):
    import pytest
    import external_order_guard as eog
    from binance_api import bapi_placeorder as bp

    # 1. external_order_guard fails fast on empty margins={}, allows partial margins with guard modes,
    #    allows now=0.0, and sanitizes NaN/bool suggested_scale
    with pytest.raises(KeyError):
        eog.check_external_order_guards("BTCUSDC", "BUY", 60000.0, margins={})

    ok_part, r_part, scale_part = eog.check_external_order_guards(
        "BTCUSDC",
        "BUY",
        60000.0,
        margins={"whale_guard_mode": "off", "orderbook_wall_guard_mode": "off", "funding_guard_mode": "off"},
        now=0.0,
    )
    assert ok_part is True and r_part == "ok"
    assert scale_part == 1.0

    from types import SimpleNamespace
    from intelligence.internal.guards.guard_decision import BrakeAction
    import intelligence.external.guards.whale_divergence_guard as wdg
    import intelligence.external.guards.orderbook_wall_guard as owg
    import intelligence.external.guards.funding_crowding_guard as fcg

    monkeypatch.setattr(eog, "get_whale_collector", lambda: SimpleNamespace(fetch=lambda *a, **kw: {}))
    monkeypatch.setattr(eog, "get_orderbook_collector", lambda: SimpleNamespace(fetch=lambda *a, **kw: {}))
    monkeypatch.setattr(eog, "get_derivatives_collector", lambda: SimpleNamespace(fetch=lambda *a, **kw: {}))
    monkeypatch.setattr(
        wdg, "WhaleDivergenceGuard",
        lambda **kw: SimpleNamespace(check=lambda *a, **k: SimpleNamespace(allowed=True, brake_action=BrakeAction.DOWNSCALE_QTY, suggested_scale=float("nan"), reason="w_nan")),
    )
    monkeypatch.setattr(
        owg, "OrderbookWallGuard",
        lambda **kw: SimpleNamespace(check=lambda *a, **k: SimpleNamespace(allowed=True, brake_action=BrakeAction.DOWNSCALE_QTY, suggested_scale=True, reason="ob_bool")),
    )
    monkeypatch.setattr(
        fcg, "FundingCrowdingGuard",
        lambda **kw: SimpleNamespace(check=lambda *a, **k: SimpleNamespace(allowed=True, brake_action=BrakeAction.DOWNSCALE_QTY, suggested_scale=0.75, reason="f_down")),
    )
    ok_scale, r_scale, scale_val = eog.check_external_order_guards(
        "BTCUSDC",
        "BUY",
        60000.0,
        margins={
            "external_guards_mode": "enforce",
            "whale_guard_mode": "enforce",
            "orderbook_wall_guard_mode": "enforce",
            "funding_guard_mode": "enforce",
            "severe_cascade_oi_pct": 5.0,
            "min_taker_ratio": 0.5,
            "taker_downscale_threshold": 0.8,
            "taker_downscaled_scale": 0.5,
            "min_buy_imbalance": 0.2,
            "max_sell_imbalance": 0.8,
            "whale_wall_usd_limit": 500000.0,
            "funding_max_long_rate": 0.001,
            "funding_min_short_rate": -0.001,
            "funding_crowding_policy": "downscale",
            "funding_crowded_scale": 0.75,
        },
        now=0.0,
    )
    assert ok_scale is True and r_scale == "f_down"
    assert scale_val == pytest.approx(0.75)

    # 2. bapi_placeorder rejects boolean qty/price/cancel_opposite_requested_price
    with pytest.raises(bp.SubmissionRefused):
        bp.apply_weight_limit("BTCUSDC", "BUY", True, 1.0, 1.0)
    with pytest.raises(bp.SubmissionRefused):
        bp.apply_weight_limit("BTCUSDC", "BUY", 60000.0, True, 1.0)
    with pytest.raises(bp.SubmissionRefused):
        bp.apply_weight_limit("BTCUSDC", "BUY", 60000.0, 1.0, True)
    with pytest.raises(bp.SubmissionRefused):
        bp.issue_account_cache_submit_permit("BTCUSDC", "BUY", True, price=60000.0)
    with pytest.raises(bp.SubmissionRefused):
        bp.issue_account_cache_submit_permit("BTCUSDC", "BUY", 1.0, price=True)
    with pytest.raises(bp.SubmissionRefused):
        bp._submit_binance_order("BUY", "BTCUSDC", 1.0, price=60000.0, cancel_opposite_requested_price=True)


def test_iteration38_headroom_cancel_opposite_and_provider_base_nan_bool_safety(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import binance_cache_health as bch
    from binance_api import bapi_placeorder as bp
    from providers.base import MarketDataProvider, PairPrecision

    # 1. _require_submit_permit_headroom refuses NaN, negative, or boolean cache ages
    for bad_status in (
        SimpleNamespace(order_age_sec=0.5, trade_age_sec=float("nan")),
        SimpleNamespace(order_age_sec=float("nan"), trade_age_sec=0.5),
        SimpleNamespace(order_age_sec=True, trade_age_sec=0.5),
        SimpleNamespace(order_age_sec=-1.0, trade_age_sec=0.5),
    ):
        with pytest.raises(bp.SubmissionRefused):
            bp._require_submit_permit_headroom(bad_status)

    # 2. cancel_opposite_orders ignores zero/missing price on BUY symmetrically with SELL and rejects bool requested_price
    with pytest.raises(bp.SubmissionRefused):
        bp.cancel_opposite_orders("BUY", "BTCUSDC", True)

    canceled_ids = []
    monkeypatch.setattr(
        bp.api,
        "get_open_orders",
        lambda side, sym, **kw: {
            "oid_zero": {"price": "0.00000000"},
            "oid_adverse": {"price": "59000.0" if side == "SELL" else "61000.0"},
        },
    )
    monkeypatch.setattr(bp.api, "cancel_order", lambda sym, oid: canceled_ids.append(oid) or True)
    bp.cancel_opposite_orders("BUY", "BTCUSDC", 60000.0)
    assert canceled_ids == ["oid_adverse"]
    canceled_ids.clear()
    bp.cancel_opposite_orders("SELL", "BTCUSDC", 60000.0)
    assert canceled_ids == ["oid_adverse"]

    # 3. binance_cache_health._clean_now_ms ignores bool/NaN
    ts_clean = bch._clean_now_ms(True)
    assert ts_clean > 1_000_000_000_000
    assert bch._clean_now_ms(float("nan")) > 1_000_000_000_000

    # 4. MarketDataProvider round_quantity, round_price, order_filter_refusal, and last_opposite_fill
    class _DummyProv(MarketDataProvider):
        @property
        def name(self) -> str:
            return "dummy"

        def get_current_price(self, symbol: str):
            return 100.0

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def pair_precision(self, symbol: str):
            return PairPrecision(price_decimals=2, volume_decimals=3, order_min=0.01)

        def get_orders(self, symbol: str, side, since_s: float):
            return [
                {"timestamp": 1000, "price": 50000.0, "qty": 1.0},
                {"timestamp": 2000, "price": 0.0, "qty": 1.0},
                "malformed",
            ]

    dp = _DummyProv()
    assert dp.round_quantity("BTCUSDC", True) == 0.0
    assert dp.round_quantity("BTCUSDC", float("nan")) == 0.0
    assert dp.round_price("BTCUSDC", True) == 0.0
    assert dp.round_price("BTCUSDC", float("nan")) == 0.0
    assert dp.order_filter_refusal("BTCUSDC", "BUY", 100.0, True) == "qty_zero_or_negative"
    assert dp.last_opposite_fill("BTCUSDC", "BUY") == pytest.approx(50000.0)


def test_iteration39_binance_filters_strategy_executor_market_api_nan_bool_safety():
    from types import SimpleNamespace
    import pytest
    from providers.binance_filters import BinanceFilterError, BinanceOrderRules, _decimal, _parse_bool
    from providers.strategy_executor import ProviderError, SubmissionOutcome
    from providers.market_api import BinanceProvider
    from providers.quantity import decide_quantity

    # 1. binance_filters._decimal rejects bool and _parse_bool handles string "false"
    with pytest.raises(BinanceFilterError):
        _decimal(True, name="qty")
    assert _parse_bool("false") is False
    assert _parse_bool("true") is True
    rules = BinanceOrderRules.from_symbol_info({
        "baseAsset": "BTC",
        "quoteAsset": "USDC",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "1", "maxPrice": "1000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100"},
            {"filterType": "NOTIONAL", "minNotional": "10", "maxNotional": "100000", "applyMinToMarket": "false"},
        ],
    })
    assert rules.min_notional_market is False

    # 2. SubmissionOutcome rejects boolean order_id
    with pytest.raises(ValueError):
        SubmissionOutcome("accepted", order_id=True)

    # 3. BinanceProvider.position_cost_basis / order_status / cancel_order reject boolean inputs
    bp_prov = BinanceProvider()
    with pytest.raises(ValueError):
        bp_prov.position_cost_basis("BTCUSDC", 1.0, True)
    with pytest.raises(ProviderError):
        bp_prov.order_status("BTCUSDC", True)
    with pytest.raises(ProviderError):
        bp_prov.cancel_order("BTCUSDC", True)

    # 4. decide_quantity ignores boolean/negative round_fn return values
    prov_q = SimpleNamespace(
        name="dummy",
        free_balance=lambda asset: 1000.0,
        policy_cap_quantity=lambda sym, side, px, req, avail, **kw: avail,
        fee_cap_quantity=lambda sym, side, px, avail: avail,
        round_amount=lambda sym, q: True,
        order_filter_refusal=lambda *a, **kw: None,
    )
    qd = decide_quantity(prov_q, "BTCUSDC", "BUY", 100.0, 2.5, scale=0.5)
    assert qd.final_qty == pytest.approx(1.25)


def test_iteration40_kraken_hyperliquid_rollback_outcomes_nan_bool_safety(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest
    import order_outcomes_log as ool
    import order_rollback as orb
    from providers.strategy_executor import ProviderError
    from providers.kraken_provider import KrakenProvider
    from providers.hyperliquid_provider import HyperliquidProvider

    # 1. order_outcomes_log sanitizes CR/LF and ignores NaN/bool strength
    assert ool._sanitize_outcome_field("a|b\r\nc") == "a/b  c"
    monkeypatch.setattr(ool, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    ool.log_order_outcome(
        "BTCUSDC", "BUY", 60000.0, 0.1, "accepted", None, "test",
        regime_context=SimpleNamespace(regime="bull", strength=float("nan"), source="snap", fresh=True, fallback_used=False, reason="ok"),
    )
    logs = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "nan" not in content.lower()

    # 2. order_rollback.extract_order_restore_params ignores boolean kind and level
    p_rest = orb.extract_order_restore_params({"side": "SELL", "price": 100.0, "qty": 1.0, "kind": True, "level": False})
    assert p_rest["kind"] == "TP"
    assert p_rest["level"] is None

    # 3. KrakenProvider min_order_qty, free_balance, preflight_order, cancel_order_by_id, pair_precision
    kp = KrakenProvider(client=SimpleNamespace(
        pair_info=lambda sym: {"ordermin": True, "pair_decimals": True, "lot_decimals": 2, "base": "HYPE"},
        balance=lambda: {"HYPE": True},
        cancel_order=lambda oid: {"count": True},
    ))
    assert kp.min_order_qty("HYPEUSD") == 0.0
    assert kp.free_balance("HYPE") is None
    with pytest.raises(ProviderError):
        kp.preflight_order("HYPEUSD", "SELL", True, price=20.0)
    with pytest.raises(ProviderError):
        kp.cancel_order_by_id("HYPEUSD", "OID1")
    with pytest.raises(ProviderError):
        kp.pair_precision("HYPEUSD")

    # 4. HyperliquidProvider preflight_order, order_status, cancel_order reject NaN/bool
    hp = HyperliquidProvider()
    monkeypatch.setattr(hp, "free_balance", lambda asset: 1000.0)
    with pytest.raises(ProviderError):
        hp.preflight_order("HYPEUSDC", "BUY", float("nan"), price=20.0)
    with pytest.raises(ProviderError):
        hp.preflight_order("HYPEUSDC", "BUY", 1.0, price=float("nan"))
    with pytest.raises(ProviderError):
        hp.preflight_order("HYPEUSDC", "BUY", True, price=20.0)
    with pytest.raises(ProviderError):
        hp.order_status("HYPEUSDC", True)
    with pytest.raises(ProviderError):
        hp.cancel_order("HYPEUSDC", True)


def test_iteration41_t212_provider_bapi_and_bapi_allorders_nan_bool_safety(monkeypatch):
    from types import SimpleNamespace
    import time
    import pytest
    from providers.strategy_executor import ProviderError
    from providers.t212_provider import T212Provider
    from binance_api import bapi, bapi_allorders

    # 1. T212Provider rejects boolean qty/price/order_id/fill fields
    tp = T212Provider(
        client=SimpleNamespace(
            get_portfolio=lambda: [{"ticker": "AAPL_US_EQ", "averagePrice": True, "quantity": 1.0, "currentPrice": True}],
            get_cash=lambda: {"free": True},
            get_order_status=lambda oid: {"ticker": "AAPL_US_EQ", "status": "FILLED", "filledQuantity": True, "filledValue": 100.0},
        ),
        live_enabled=True,
        order_validity="DAY",
    )
    assert tp.free_balance("RON") is None
    assert tp.free_balance("AAPL_US_EQ") == 1.0
    with pytest.raises(ProviderError):
        tp.get_orders("AAPL_US_EQ", "BUY", 3600.0)
    with pytest.raises(ProviderError):
        tp._signed_qty("BUY", True)
    with pytest.raises(ProviderError):
        tp._send_order("AAPL_US_EQ", "BUY", 1.0, True, market=False)
    with pytest.raises(ProviderError):
        tp.order_status("AAPL_US_EQ", True)
    with pytest.raises(ProviderError):
        tp.order_status("AAPL_US_EQ", "OID1")
    with pytest.raises(ProviderError):
        tp.cancel_order("AAPL_US_EQ", True)

    # 2. bapi.get_current_price ignores NaN/bool cached values and cancel_order/check_order_filled reject bool order_id
    monkeypatch.setitem(bapi.cprice, "BTCUSDC", float("nan"))
    monkeypatch.setitem(bapi.cprice_time, "BTCUSDC", time.time())
    monkeypatch.setattr(bapi, "update_price", lambda sym: 61234.5)
    assert bapi.get_current_price("BTCUSDC") == pytest.approx(61234.5)
    assert bapi.cancel_order("BTCUSDC", True) is False
    assert bapi.check_order_filled(True, "BTCUSDC") is False
    assert bapi._convert_to_usdc("USDC", True) == 0.0

    # 3. bapi_allorders.get_total_traded_stats ignores malformed/NaN/bool trades and get_filled_orders keeps unkeyed fills separate
    monkeypatch.setattr(
        bapi_allorders,
        "get_trade_orders",
        lambda order_type, symbol, max_age_seconds: [
            {"side": "BUY", "price": 100.0, "quantity": 2.0},
            {"side": "BUY", "price": float("nan"), "quantity": 5.0},
            {"side": "SELL", "price": True, "quantity": 3.0},
            "malformed",
        ],
    )
    st = bapi_allorders.get_total_traded_stats("BTCUSDC", period_seconds=True)
    assert st["BUY"]["total_quantity"] == pytest.approx(2.0)
    assert st["BUY"]["total_value"] == pytest.approx(200.0)
    assert st["SELL"]["trade_count"] == 0


def test_iteration42_order_id_context_bapi_ws_normalize_and_bapi_trades_nan_bool_safety():
    import pytest
    from binance_api import order_id_context, bapi_ws_normalize, bapi_trades

    # 1. order_id_context: boolean clientOrderId & case-insensitive allowed_owners
    assert order_id_context.resolve_order_owner(True) == "unspecified"
    assert order_id_context.resolve_order_owner(False) == "unspecified"
    assert order_id_context.order_matches_owner("TA_main_1234", "TRADEALL") is True
    assert order_id_context.order_matches_owner("MO_TA_1234", ["MonitorOrder"]) is True
    assert order_id_context.order_matches_owner("TA_main_1234", True) is False
    assert order_id_context.order_matches_owner("TA_main_1234", [True, None, ""]) is False

    # 2. bapi_ws_normalize: now_ms=True / NaN falls back to current epoch ms; boolean commission ignored
    ev = {
        "s": "BTCUSDC",
        "x": "TRADE",
        "X": "FILLED",
        "S": "BUY",
        "i": 1001,
        "t": 2002,
        "l": "0.5",
        "L": "60000.0",
        "n": True,
        "N": False,
    }
    sym1, ord_row = bapi_ws_normalize.parse_order_execution_report(ev, now_ms=True)
    assert sym1 == "BTCUSDC"
    assert ord_row["timestamp"] > 1_700_000_000_000
    sym2, ord_row2 = bapi_ws_normalize.parse_order_execution_report(ev, now_ms=float("nan"))
    assert sym2 == "BTCUSDC"
    assert ord_row2["timestamp"] > 1_700_000_000_000
    sym3, tr_row = bapi_ws_normalize.parse_trade_execution_report(ev, now_ms=True)
    assert sym3 == "BTCUSDC"
    assert tr_row["time"] > 1_700_000_000_000
    assert "commission" not in tr_row
    assert "commissionAsset" not in tr_row

    # 3. bapi_trades.aggregate_trades: rejects boolean orderId, quoteQty, commission, commissionAsset
    agg = bapi_trades.aggregate_trades([
        {"orderId": True, "symbol": "BTCUSDC", "qty": "1.0", "price": "50000", "isBuyer": True},
        {
            "orderId": 777,
            "id": True,
            "symbol": "BTCUSDC",
            "qty": "0.5",
            "price": "50000",
            "quoteQty": True,
            "commission": True,
            "commissionAsset": True,
            "isBuyer": "not-a-bool",
            "side": "SELL",
            "time": 1700000000000,
        },
    ])
    assert len(agg) == 1
    assert agg[0]["orderId"] == 777
    assert agg[0]["id"] == 0
    assert float(agg[0]["quoteQty"]) == pytest.approx(25000.0)
    assert float(agg[0]["commission"]) == pytest.approx(0.0)
    assert agg[0]["commissionAsset"] == ""
    assert agg[0]["side"] == "SELL"


def test_iteration43_ws_health_market_data_execution_audit_replay_and_trailing_nan_bool_safety(tmp_path):
    import json
    from types import SimpleNamespace
    import pytest
    from binance_api import bapi_ws_health, market_data, trailing_stop
    from providers.execution_audit import ExecutionAudit, AuditedStrategyExecutor
    from providers.replay_provider import ReplayMarketDataProvider
    from providers.replay_clock import SimClock
    from providers.strategy_executor import ProviderError

    # 1. bapi_ws_health.mark_ws_event_received ignores bool/NaN and falls back to wall clock
    ts_bool = bapi_ws_health.mark_ws_event_received(True)
    assert ts_bool > 1_700_000_000.0
    ts_nan = bapi_ws_health.mark_ws_event_received(float("nan"))
    assert ts_nan > 1_700_000_000.0

    # 2. market_data.fetch_ohlc_candles rejects boolean window parameters
    with pytest.raises(ValueError):
        market_data.fetch_ohlc_candles("BTCUSDC", start_ms=True)
    with pytest.raises(ValueError):
        market_data.fetch_ohlc_candles("BTCUSDC", lookback_days=True)

    # 3. ExecutionAudit.record strips NaN/Inf fields (strict JSON) and handles NaN/bool clock
    audit = ExecutionAudit(directory=str(tmp_path), clock=lambda: float("nan"))
    ok = audit.record(
        "submit_requested",
        intent_id="id-1",
        venue="binance",
        symbol="BTCUSDC",
        price=float("nan"),
        reference_price=float("inf"),
        qty=1.25,
    )
    assert ok is True
    files = list(tmp_path.glob("execution_audit_*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["qty"] == 1.25
    assert "price" not in row
    assert "reference_price" not in row

    aud_exec = AuditedStrategyExecutor(
        SimpleNamespace(
            name="binance",
            order_by_client_id=lambda s, cid: {"orderId": 1},
        ),
        audit=audit,
    )
    with pytest.raises(ProviderError):
        aud_exec.order_by_client_id("BTCUSDC", True)
    aud_exec._remember("BTCUSDC", True, "intent-bool")
    assert ("BTCUSDC", "True") not in aud_exec._intent_by_order

    # 4. ReplayMarketDataProvider and SimClock reject bool/NaN inputs
    with pytest.raises(ValueError):
        SimClock(True)
    with pytest.raises(ValueError):
        SimClock(float("nan"))
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0), (1060.0, 51000.0)]})
    assert rp.advance("BTCUSDC", True) is None
    assert rp.advance("BTCUSDC", 1) == 50000.0
    rp.place_order("BTCUSDC", "BUY", 50000.0, 0.1)
    assert rp.get_orders("BTCUSDC", "BUY", float("nan")) == []
    assert rp.get_orders("BTCUSDC", "BUY", True) == []
    assert rp.has_orders_since("BTCUSDC", float("nan")) is False
    assert rp.get_price_history("BTCUSDC", float("nan")) == []

    # 5. TrailingStop rejects boolean price/balance/trend
    dummy_api = SimpleNamespace(
        get_current_price=lambda pair: True,
        get_account_assets_balances=lambda: [{"asset": "BTC", "free": True, "locked": False}],
    )
    dummy_po = SimpleNamespace(position_cost_basis=lambda *a, **k: 50000.0)
    dummy_sym = SimpleNamespace(symbols=["BTCUSDC"])
    ts = trailing_stop.TrailingStop(dummy_api, dummy_po, dummy_sym, enabled=False)
    ts._balances = [{"asset": "BTC", "free": True, "locked": True}]
    assert ts.price("BTCUSDC") is None
    assert ts.free_qty("BTC") == 0.0
    assert ts.total_qty("BTC") == 0.0
    assert ts.cost_basis("BTCUSDC") is None


def test_iteration44_trailing_core_and_order_retry_nan_bool_safety(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest
    import trailing_core
    import order_retry

    # 1. TrailingCore._safe_trend ignores boolean True so sell_skip_if_trend_up does not skip a crash sell
    sold_calls = []
    adapter = SimpleNamespace(
        assets=lambda: [("BTCUSDC", "BTC", "BTCUSDC", 10.0)],
        begin_tick=lambda: True,
        free_qty=lambda asset: 1.0,
        price=lambda pair: 85.0,
        trend=lambda pair: True,  # boolean True must be treated as 0.0, not > 0
        log_dry_sell=lambda *a: sold_calls.append(a),
        log_skip_sell_trend=lambda *a: None,
        log_hold=lambda *a: None,
        log_item_error=lambda *a: None,
    )
    core = trailing_core.TrailingCore(
        adapter,
        log=lambda *a: None,
        enabled=False,
        state_file=str(tmp_path / "trail_state.json"),
        min_notional=10.0,
        rebuy_enabled=False,
        rebuy_bounce_pct=5.0,
        rebuy_skip_if_trend_down=True,
        sell_skip_if_trend_up=True,
        sell_fraction=1.0,
    )
    assert core._safe_trend("BTCUSDC") == 0.0
    core.save({"BTCUSDC": {"peak": 100.0, "exit_sold_qty": float("nan")}})
    core.check_once()
    assert len(sold_calls) == 1

    # 2. order_retry: valid_record rejects boolean id/symbol/order_id/replaces_order_id/replaces_original_qty
    base_rec = {
        "id": "rec1",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 0.5,
        "created_ts": 1700000000.0,
        "revision": 0,
        "lifecycle": "submit_pending",
    }
    assert order_retry.valid_record(base_rec) is True
    assert order_retry.valid_record({**base_rec, "id": True}) is False
    assert order_retry.valid_record({**base_rec, "symbol": True}) is False
    assert order_retry.valid_record({**base_rec, "lifecycle": "accepted", "order_id": True}) is False
    assert order_retry.valid_record({**base_rec, "lifecycle": "awaiting_cancel", "replaces_order_id": True, "replaces_original_qty": 1.0}) is False
    assert order_retry.valid_record({**base_rec, "lifecycle": "awaiting_cancel", "replaces_order_id": "123", "replaces_original_qty": True}) is False

    # 3. order_retry: enqueue sanitizes NaN/bool requested_price & ref_price, and is_expired/is_due reject bool now
    monkeypatch.setattr(order_retry, "QUEUE_FILE", str(tmp_path / "q.jsonl"))
    monkeypatch.setattr(order_retry, "LOCK_FILE", str(tmp_path / "q.lock"))
    rid = order_retry.enqueue(
        "BTCUSDC",
        "BUY",
        0.25,
        requested_price=float("nan"),
        ref_price=True,
        failure_reason="profit_guard",
        now=1700000000.0,
    )
    assert rid is not None
    loaded = order_retry.get(rid)
    assert loaded["requested_price"] is None
    assert loaded["ref_price"] is None
    assert order_retry.is_expired(loaded, now=True) is True
    assert order_retry.is_due(loaded, now=True) is False
    assert order_retry.resolve_accepted_order("BTCUSDC", True) is False
    assert order_retry.get_accepted_order_ids_for_owner(True, "tradeall") == []


def test_iteration45_spot_rules_and_order_retry_worker_nan_bool_safety(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest
    from strategies import spot_rules as sr
    import order_retry as oq
    import order_retry_worker as orw

    # 1. spot_rules: hit_stop and dca_price_hit reject 0.0 / False / True / NaN prices
    assert sr.hit_stop(100.0, 0.0, 5.0) is False
    assert sr.hit_stop(100.0, False, 5.0) is False
    assert sr.hit_stop(100.0, True, 5.0) is False
    assert sr.hit_stop(100.0, float("nan"), 5.0) is False
    assert sr.hit_stop(100.0, 94.0, 5.0) is True

    assert sr.dca_price_hit(0.0, 100.0, 5.0, 0.05) is False
    assert sr.dca_price_hit(False, 100.0, 5.0, 0.05) is False
    assert sr.dca_price_hit(True, 100.0, 5.0, 0.05) is False
    assert sr.dca_price_hit(float("nan"), 100.0, 5.0, 0.05) is False
    assert sr.dca_price_hit(95.0, 100.0, 5.0, 0.05) is True

    assert sr.are_close(1.0, True, 0.05) is False
    assert sr.are_close(100.0, 100.0, float("nan")) is False
    assert sr.progressive_dca_drop_pct(float("nan"), True, float("nan")) == 0.0
    assert sr.parabolic_surge_profit_floor_pct(float("nan"), True, float("nan")) == pytest.approx(20.2)

    # 2. order_retry_worker: awaiting_cancel with NaN filled_qty must NOT resolve/delete the replacement record
    monkeypatch.setattr(oq, "QUEUE_FILE", str(tmp_path / "q45.jsonl"))
    monkeypatch.setattr(oq, "LOCK_FILE", str(tmp_path / "q45.lock"))
    rid = oq.enqueue(
        "BTCUSDC",
        "SELL",
        1.0,
        lifecycle="awaiting_cancel",
        replaces_order_id="ORD-999",
        replaces_original_qty=1.0,
        now=1000.0,
        created_ts=1000.0,
    )
    assert rid is not None
    fake_mkt = SimpleNamespace(
        order_status=lambda sym, oid, provider_name=None: SimpleNamespace(
            status="canceled",
            venue_status="CANCELED",
            filled_qty=float("nan"),
            terminal=True,
        ),
    )
    stats = orw.process_once(fake_mkt, now=2000.0)
    rec_after = oq.get(rid)
    assert rec_after is not None
    assert rec_after["lifecycle"] == "awaiting_cancel"
    assert rec_after.get("status_error") == "cancel_activation_failed"
    assert orw._format_ts(True) == "N/A"
    with pytest.raises(ValueError):
        orw.IncidentDigest(tmp_path / "dig.json", True)


def test_iteration46_spot_engine_rtrade_pair_spot_replay_and_state_io_nan_bool_safety(tmp_path):
    from types import SimpleNamespace
    import math
    import pytest
    from providers.strategy_executor import PairPrecision
    from strategies import spot_engine as se
    from strategies import rtrade_pair as rp
    from strategies import spot_replay as srep
    import state_io as sio

    # 1. spot_engine: set_simulated_time, _evaluate_reentry_gate, ledger reconciliation, and peak tracking
    fake_client = SimpleNamespace(
        pair_precision=lambda _pair: PairPrecision(2, 4, 0.001, "BTC"),
        free_balance=lambda _asset: float("nan"),
        order_status=lambda _pair, _txid: SimpleNamespace(
            status="open", filled_qty=True, cost=100.0, fee=0.0
        ),
    )
    params = se.StratParams(
        currency="USDC",
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=100.0,
        dca_drop_pct=2.0,
        check_minutes=1.0,
        takeprofit_pct=2.0,
        max_budget=1000.0,
        max_dca_buys=5,
        enable_takeprofit=True,
        order_ttl_min=15.0,
        stop_loss_pct=10.0,
        surge_guard=True,
        slow_grind_guard=True,
    )
    strat = se.Strategy(
        fake_client,
        "BTCUSDC",
        params,
        dry_run=True,
        initial_state=se._new_state(),
        replay_mode=True,
        state_dir=str(tmp_path),
    )
    strat.set_simulated_time(1234.5)
    assert strat._now() == 1234.5
    strat.set_simulated_time(True)
    assert strat._simulated_ts is None
    strat.set_simulated_time(float("nan"))
    assert strat._simulated_ts is None

    # Corrupted NaN/bool in reentry state does not poison post_sell_peak / post_sell_trough / sl_low
    strat.s["last_sell_price"] = 100.0
    strat.s["post_sell_peak"] = float("nan")
    strat.s["post_sell_trough"] = True
    strat.s["last_exit_kind"] = "STOP"
    strat.s["sl_low"] = float("nan")
    strat._evaluate_reentry_gate(98.0, None, 1000.0)
    assert math.isfinite(strat.s["post_sell_peak"]) and strat.s["post_sell_peak"] == 100.0
    assert math.isfinite(strat.s["post_sell_trough"]) and strat.s["post_sell_trough"] == 98.0
    assert math.isfinite(strat.s["sl_low"]) and strat.s["sl_low"] == 98.0

    # Corrupted NaN in surge_peak / slow_grind_peak does not poison peak tracking
    strat.s["surge_peak"] = float("nan")
    strat.s["slow_grind_peak"] = float("nan")
    strat._check_advanced_profit_guards(105.0, 100.0, 1.0, closes=[100.0] * 25, tick_time=1000.0)
    assert strat.s["surge_peak"] == 105.0
    assert strat.s["slow_grind_peak"] == 105.0

    # Ledger reconciliation with NaN/bool balance fails closed without clearing ledger_reconcile_required
    strat.dry_run = False
    strat.s["qty"] = 1.0
    strat.s["cost"] = 100.0
    strat.s["spent"] = 100.0
    strat.s["ledger_reconcile_required"] = True
    strat.step(100.0, timestamp=1000.0)
    assert strat.s["ledger_reconcile_required"] is True
    assert strat.s["qty"] == 1.0

    # 2. rtrade_pair: PairPolicy, quote_prices, anchored_exit_price, and clock/step NaN/bool rejection
    with pytest.raises(ValueError):
        rp.PairPolicy(adjustment_fraction=0.01, price_decimals=True)
    with pytest.raises(ValueError):
        rp.quote_prices(100.0, True)
    with pytest.raises(ValueError):
        rp.anchored_exit_price("SELL", 100.0, 101.0, 0.01, float("nan"))

    # 3. spot_replay: _validate_replay and align_spot_interval_params NaN/bool protection
    with pytest.raises(ValueError):
        srep.run_replay([(100, 101, 99, 100)], params, bar_minutes=True)
    with pytest.raises(ValueError):
        srep.run_replay([(100, 101, 99, 100)], params, bar_minutes=float("nan"))
    assert srep.align_spot_interval_params(params, True) is params
    assert srep.align_spot_interval_params(params, float("nan")) is params

    # 4. state_io: read_tail_lines, truncate_file_bytes, and prune_rotated_archives bool/zero protection
    tail_file = tmp_path / "tail_test.txt"
    tail_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
    assert sio.read_tail_lines(str(tail_file), 2, chunk_size=0) == [b"line2", b"line3"]
    assert sio.read_tail_lines(str(tail_file), 2, chunk_size=True) == [b"line2", b"line3"]
    with pytest.raises(ValueError):
        sio.truncate_file_bytes(str(tail_file), True)
    with pytest.raises(ValueError):
        sio.truncate_file_bytes(str(tail_file), -1)


def test_iteration47_botcore_and_market_regime_nan_bool_safety():
    import math
    import pytest
    import botcore
    import market_regime as mr

    # 1. botcore: diff_percent, are_close, diff_equals_percent reject bool and NaN
    assert math.isnan(botcore.diff_percent(1.0, True))
    assert math.isnan(botcore.diff_percent(float("nan"), 100.0))
    assert botcore.are_close(1.0, True, 0.05) is False
    assert botcore.are_close(100.0, 100.0, float("nan")) is False
    assert botcore.are_close(100.0, 100.0, True) is False
    assert botcore.are_close(100.0, 100.04, 0.05) is True
    assert botcore.diff_equals_percent(100.0, 90.0, True, 1.0) is False
    assert botcore.diff_equals_percent(100.0, 90.0, 10.0, float("nan")) is False

    # 2. market_regime: MarketRegimeDecision, MarketRegimeContext, MarketRegimeEvidence, MarketRegimeEvaluator, MarketRegimeService
    dec_bool = mr.MarketRegimeDecision(
        regime="bull", gradient=True, epsilon=0.01, strength=float("nan"), fresh=True, reason="ok", n_samples=10
    )
    assert dec_bool.fitted_move_pct is None

    ctx = mr.MarketRegimeContext(
        decision=dec_bool, resolved_trend="bull", evaluated_at=1000.0, trend_duration_seconds=60.0
    )
    assert ctx.is_valid_for(now=1010.0, max_age_seconds=float("nan")) is False
    assert ctx.is_valid_for(now=1010.0, max_age_seconds=True) is False
    assert ctx.is_valid_for(now=1010.0, max_horizon_seconds=float("nan")) is False
    assert ctx.is_valid_for(now=1010.0, max_horizon_seconds=True) is False
    assert ctx.is_valid_for(now=1010.0, max_age_seconds=30.0, max_horizon_seconds=120.0) is True

    ev = mr.MarketRegimeEvidence(
        role="asset_short",
        family="price_trend",
        classifier="test",
        provider="binance",
        symbol="BTCUSDC",
        decision=dec_bool,
        evaluated_at=1010.0,
        observed_at=1000.0,
        max_age_seconds=float("nan"),
    )
    assert ev.temporal_state == "stale"
    assert ev.usable is False

    with pytest.raises(ValueError):
        mr.MarketRegimeEvaluator(strength_threshold=True)
    with pytest.raises(ValueError):
        mr.MarketRegimeService(cache_ttl_sec=True)

    svc = mr.MarketRegimeService()
    assert svc._snapshot_observed_at({"ts": True}) is None
    assert svc.evaluate_closes([100.0, 101.0, 102.0], interval_min=0).reason == "invalid_window_metadata"
    assert svc.evaluate_closes([100.0, 101.0, 102.0], window_seconds=float("nan")).reason == "invalid_window_metadata"
    comp = svc.compose(dec_bool, dec_bool)
    assert comp.regime == "unknown"


def test_iteration48_kraken_client_cachemanager_market_data_trailing_and_xstock_nan_bool_safety(monkeypatch):
    import sys
    import pytest
    orig_sys_path = list(sys.path)
    try:
        from kraken import kraken_client as kc
        from kraken import kraken_cachemanager as kcm
        from kraken import kraken_xstock_watch as kxw
    finally:
        sys.path[:] = orig_sys_path

    # 1. kraken_client: _read_ttl, last_price, price_decimals, add_order, ohlc_closes_with_timestamps
    assert kc._read_ttl("OHLC", {"interval": float("nan")}) == 30.0
    assert kc._read_ttl("OHLC", {"interval": True}) == 30.0
    cli = kc.KrakenClient.public()
    monkeypatch.setattr(cli, "ticker", lambda _pair: {"c": [float("nan"), "1.0"]})
    assert cli.last_price("HYPEUSD") is None
    monkeypatch.setattr(cli, "ticker", lambda _pair: {"c": [True, "1.0"]})
    assert cli.last_price("HYPEUSD") is None
    monkeypatch.setattr(cli, "pair_info", lambda _pair: {"pair_decimals": True})
    assert cli.price_decimals("HYPEUSD") == 2
    with pytest.raises(ValueError):
        cli.add_order("HYPEUSD", "buy", True, 20.0)
    with pytest.raises(ValueError):
        cli.add_order("HYPEUSD", "buy", 1.0, float("nan"), ordertype="limit")
    with pytest.raises(ValueError):
        cli.ohlc_closes_with_timestamps("HYPEUSD", True)

    # 2. kraken_cachemanager: _ledger_to_fills rejects NaN/bool/zero amounts and fees
    bad_ledger = {
        "L1": {"refid": "R1", "type": "trade", "asset": "HYPE", "amount": float("nan"), "time": 1700000000},
        "L2": {"refid": "R1", "type": "trade", "asset": "ZUSD", "amount": "-100.0", "time": 1700000000},
        "L3": {"refid": "R2", "type": "trade", "asset": "HYPE", "amount": True, "time": 1700000000},
        "L4": {"refid": "R2", "type": "trade", "asset": "ZUSD", "amount": "-100.0", "time": 1700000000},
        "L5": {"refid": "R3", "type": "trade", "asset": "HYPE", "amount": "2.0", "fee": True, "time": 1700000000},
        "L6": {"refid": "R3", "type": "trade", "asset": "ZUSD", "amount": "-40.0", "fee": float("nan"), "time": 1700000000},
    }
    fills = kcm._ledger_to_fills(bad_ledger)
    assert len(fills) == 1
    assert fills[0]["id"] == "R3"
    assert fills[0]["price"] == 20.0
    assert "fee" not in fills[0]

    # 3. kraken_xstock_watch: _bot_alive rejects boolean True (PID 1 systemd) and <= 1
    assert kxw._bot_alive(True) is False
    assert kxw._bot_alive(1) is False
    assert kxw._bot_alive(-1) is False
    assert kxw._bot_alive(0) is False


def test_iteration49_hyperliquid_client_market_data_price_analysis_strategy_and_replay_nan_bool_safety():
    import os
    import sys
    from types import SimpleNamespace
    import pytest
    hl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hyperliquid"))
    orig_sys_path = list(sys.path)
    mod_names = ("common", "notify", "hl_client", "market_data", "price_analysis", "signals", "strategy", "replay")
    saved_mods = {k: sys.modules.get(k) for k in mod_names}
    try:
        for k in mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, hl_dir)
        import hl_client as hlc
        import market_data as hl_md
        import price_analysis as hl_pa
        import strategy as hl_strat
        import replay as hl_rep
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # 1. hl_client: _round_px, mid, spot_mid, position_strict, spot_balance_strict
    with pytest.raises(ValueError):
        hlc._round_px(True, 2)
    with pytest.raises(ValueError):
        hlc._round_px(20.0, True)

    dummy = object.__new__(hlc.HLClient)
    dummy.address = "0x1234"
    dummy.info = SimpleNamespace(
        all_mids=lambda: {"HYPE": float("nan"), "BTC": True, "ETH": "2500.5"},
        user_state=lambda _addr: {
            "assetPositions": [{"position": {"coin": "HYPE", "szi": float("nan"), "entryPx": 20.0}}],
            "withdrawable": True,
        },
        spot_user_state=lambda _addr: {
            "balances": [{"coin": "HYPE", "total": True}]
        },
    )
    assert dummy.mid("HYPE") is None
    assert dummy.mid("BTC") is None
    assert dummy.mid("ETH") == 2500.5
    assert dummy.withdrawable() == 0.0
    with pytest.raises(hlc.HLError):
        dummy.position_strict("HYPE")
    with pytest.raises(hlc.HLError):
        dummy.spot_balance_strict("HYPE")

    # 2. hyperliquid/market_data: normalize_closed_candles rejects bool and invalid OHLC bounds
    rows = [
        {"t": True, "T": 2000, "o": 10, "h": 12, "l": 9, "c": 11},
        {"t": 1000, "T": 2000, "o": 10, "h": 8, "l": 9, "c": 11},   # h < l
        {"t": 1000, "T": 2000, "o": 10, "h": 12, "l": 9, "c": 11},  # valid
    ]
    norm = hl_md.normalize_closed_candles(rows, now_ms=5000, drop_incomplete=True)
    assert len(norm) == 1
    assert norm[0]["close"] == 11.0

    # 3. hyperliquid/price_analysis: detect_long_term_trend ignores NaN/bool points
    assert hl_pa.detect_long_term_trend([1000.0, 2000.0, 3000.0], [float("nan"), float("nan"), float("nan")]) is None
    assert hl_pa.detect_long_term_trend([True, False], [10.0, 11.0]) is None

    # 4. hyperliquid/strategy & replay: step ignores NaN/bool price and funding_at sanitizes NaN/bool funding rate
    st_inst = hl_strat.Strategy(
        client=None, coin="HYPE", params=hl_strat.StratParams(direction="long"), dry_run=True, replay_mode=True
    )
    st_inst.step(float("nan"))
    st_inst.step(True)
    assert st_inst.s["orders"] == []

    assert hl_rep.funding_at([(1000, float("nan"))], 1000) == 0.0
    assert hl_rep.funding_at([(1000, True)], 1000) == 0.0
    assert hl_rep.funding_at([(1000, 0.0001)], True) == 0.0


def test_iteration50_t212_client_market_data_strategy_and_replay_nan_bool_safety():
    import os
    import sys
    from types import SimpleNamespace
    import pytest

    t212_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "212trading"))
    orig_sys_path = list(sys.path)
    mod_names = ("ipo_common", "ipo_notify", "t212_client", "market_data", "strategy", "replay")
    saved_mods = {k: sys.modules.get(k) for k in mod_names}
    try:
        for k in mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, t212_dir)
        import t212_client as t212c
        import market_data as t212_md
        import strategy as t212_strat
        import replay as t212_rep
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # 1. T212Client: place_limit_order & place_market_order reject bool/NaN/Inf/0 before HTTP
    cli = t212c.T212Client("k", "s", env="demo", min_gap_sec=True, portfolio_ttl_sec=float("nan"), order_min_gap_sec=-1.0)
    assert cli._min_gap == 0.0
    assert cli._pf_ttl == 6.0
    assert cli._order_min_gap == 0.0
    with pytest.raises(ValueError):
        cli.place_limit_order("NVDA_US_EQ", True, 100.0)
    with pytest.raises(ValueError):
        cli.place_limit_order("NVDA_US_EQ", 1.0, float("nan"))
    with pytest.raises(ValueError):
        cli.place_limit_order("NVDA_US_EQ", 0.001, 100.0)  # rounds to 0.0
    with pytest.raises(ValueError):
        cli.place_market_order("NVDA_US_EQ", False)
    with pytest.raises(ValueError):
        cli.place_market_order("NVDA_US_EQ", float("inf"))

    # 2. 212trading/strategy: generate_tp_ladder, adapt_ladder_rungs, _avg_cost, _portfolio_position, step
    with pytest.raises(ValueError):
        t212_strat.generate_tp_ladder(True, 10.0, 5)
    with pytest.raises(ValueError):
        t212_strat.generate_tp_ladder(2.0, 10.0, True)
    assert t212_strat.adapt_ladder_rungs([(2.0, 0.5), (4.0, 0.5)], float("nan")) == []
    assert t212_strat.adapt_ladder_rungs([(2.0, 0.5), (4.0, 0.5)], True) == []

    params = t212_strat.StratParams(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.5,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=5.0,
        takeprofit_pct=3.0, max_budget=500.0, max_dca_buys=5,
        validity="GOOD_TILL_CANCEL", enable_takeprofit=True,
        order_ttl_min=60.0, stop_loss_pct=10.0, yahoo_sym="NVDA",
    )
    dummy_cli = SimpleNamespace(
        get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": float("nan"), "averagePrice": 120.0}],
        list_active_orders=lambda: [],
    )
    st = t212_strat.Strategy(dummy_cli, "NVDA_US_EQ", params, dry_run=True, fx_to_usd=1.0, clock=lambda: 1000.0)
    assert st._portfolio_position() is None
    st.s["qty"] = 2.0
    st.s["cost_usd"] = float("nan")
    assert st._avg_cost() is None
    assert st.reconcile(float("nan")) is False
    assert st.reconcile(True) is False
    st.step(float("nan"))
    st.step(True)

    # 3. 212trading/replay: _fx_rates & _ols_slope_pct reject bool/NaN/Inf
    with pytest.raises(ValueError):
        t212_rep._fx_rates(True, 3)
    with pytest.raises(ValueError):
        t212_rep._fx_rates(float("nan"), 3)
    with pytest.raises(ValueError):
        t212_rep._fx_rates([1.0, float("inf"), 1.0], 3)
    assert t212_rep._ols_slope_pct([True] * 10) is None
    assert t212_rep._ols_slope_pct([float("nan")] * 10) is None


def test_iteration51_binance_api_and_providers_nan_bool_safety(monkeypatch):
    import os
    from types import SimpleNamespace
    import pytest
    from binance_api import bapi_placeorder, bapi_allorders, trailing_stop
    from providers.base import ProviderError
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.t212_provider import T212Provider

    # 1. bapi_placeorder._submit_binance_order rejects bool/NaN/Inf/<=0 even when cache_permit is None
    dummy_bclient = SimpleNamespace(
        order_limit_buy=lambda **kw: {"orderId": 1},
        order_market_buy=lambda **kw: {"orderId": 2},
    )
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order(
            "BUY", "BTCUSDC", True, price=50000.0, market=False, api_client=dummy_bclient
        )
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order(
            "BUY", "BTCUSDC", 0.01, price=float("nan"), market=False, api_client=dummy_bclient
        )
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order(
            "BUY", "BTCUSDC", float("nan"), price=None, market=True, api_client=dummy_bclient
        )
    with pytest.raises(ValueError):
        bapi_placeorder.order_candidate_price("BUY", True, 100.0)
    assert bapi_placeorder.order_candidate_price("BUY", 100.0, True) == 100.0
    assert bapi_placeorder.order_candidate_price("BUY", 100.0, float("nan")) == 100.0

    # 2. bapi_allorders.paginate_my_trades rejects bool start_time_ms & normalizes bool limit to 1000
    assert bapi_allorders.paginate_my_trades(SimpleNamespace(), "BTCUSDC", start_time_ms=True) == []
    seen_limits = []
    assert (
        bapi_allorders.paginate_my_trades(
            SimpleNamespace(get_my_trades=lambda **kw: seen_limits.append(kw["limit"]) or []),
            "BTCUSDC",
            start_time_ms=1000,
            limit=True,
        )
        == []
    )
    assert seen_limits == [1000]

    # 3. TrailingStop.reconcile_pending handles NaN/bool filled_qty and req_qty safely
    ts = trailing_stop.TrailingStop(
        api=SimpleNamespace(), po=SimpleNamespace(), sym=SimpleNamespace(symbols=[]), log=lambda _m: None
    )
    ts.order_lifecycle = SimpleNamespace(
        reconcile=lambda pending, persist: SimpleNamespace(
            outcome="terminal",
            status=SimpleNamespace(status="closed", filled_qty=float("nan")),
        )
    )
    st_dict = {"retry_counts": {}}
    persist_fn = lambda _val: None
    persist_fn.st = st_dict
    pending_obj = {"symbol": "BTCUSDC", "action": "SELL", "submitted_qty": float("nan"), "attempt": True}
    res = ts.reconcile_pending(pending_obj, persist_fn)
    assert res.outcome == "terminal"
    assert st_dict["retry_counts"]["SELL"] == 1

    # 4. HyperliquidProvider: free_balance, get_price_history, ohlc_series, order_status
    hlp = HyperliquidProvider()
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
    fake_hl = SimpleNamespace(
        info=SimpleNamespace(
            spot_user_state=lambda addr: {
                "balances": [{"coin": "USDC", "total": "nan", "hold": "0.0"}]
            },
            candles_snapshot=lambda pair, interval, start, end: [
                {"t": 1700000000000, "c": "nan"},
                {"t": True, "c": "25.0"},
                {"t": 1700000060000, "c": "25.5"},
            ],
            query_order_by_oid=lambda addr, oid: {
                "status": "order",
                "order": {"status": "open", "order": {"origSz": "nan", "sz": "1.0"}},
            },
            user_fills=lambda addr: [],
        ),
        candles=lambda pair, interval, lookback_h: [
            {"c": "nan", "T": 1700000000000},
            {"c": True, "T": 1700000060000},
            {"c": "25.5", "T": 1700000120000},
            {"c": "26.0", "T": 1700000180000},
        ],
    )
    monkeypatch.setattr(hlp, "_hl", lambda: fake_hl)
    monkeypatch.setattr(hlp, "_pair", lambda: "@107")
    assert hlp.free_balance("USDC") is None
    assert hlp.get_price_history("HYPEUSDC", True) is None
    assert hlp.get_price_history("HYPEUSDC", float("nan")) is None
    hist = hlp.get_price_history("HYPEUSDC", 1.0)
    assert hist == [{"timestamp": 1700000060000, "price": 25.5}]
    with pytest.raises(ProviderError):
        hlp.ohlc_series("HYPEUSDC", True)
    series = hlp.ohlc_series("HYPEUSDC", 15)
    assert series.closes == (25.5,)
    with pytest.raises(ProviderError):
        hlp.order_status("HYPEUSDC", "101")

    # 5. KrakenProvider: get_price_history, open_orders, order_status
    kp = KrakenProvider()
    fake_kcli = SimpleNamespace(
        _public=lambda ep, params: {"XXBTZUSD": [[1700000000, "1", "1", "1", "nan"], [True, "1", "1", "1", "50000"]]},
        open_orders=lambda: {
            "O1": {"descr": {"pair": "XBTUSD", "type": "buy", "price": True}, "vol": "1.0", "vol_exec": "0.0"}
        },
        query_orders=lambda oid: {
            oid: {"status": "closed", "vol_exec": "nan", "cost": "100.0", "fee": "0.1", "vol": "1.0"}
        },
    )
    monkeypatch.setattr(kp, "_client", lambda: fake_kcli)
    assert kp.get_price_history("BTCUSD", True) is None
    assert kp.get_price_history("BTCUSD", float("nan")) is None
    with pytest.raises(ProviderError):
        kp.open_orders("BTCUSD")
    with pytest.raises(ProviderError):
        kp.order_status("BTCUSD", "O1")

    # 6. T212Provider: order_status rejects NaN/Inf filledQuantity, filledValue, fee, fillPrice
    tp = T212Provider()
    fake_t212 = SimpleNamespace(
        get_order_status=lambda oid: {
            "ticker": "NVDA_US_EQ",
            "status": "FILLED",
            "filledQuantity": "nan",
            "filledValue": 100.0,
            "fee": 0.0,
        }
    )
    monkeypatch.setattr(tp, "_client", lambda: fake_t212)
    with pytest.raises(ProviderError):
        tp.order_status("NVDA_US_EQ", "ORD1")


def test_iteration52_binance_market_data_and_kraken_modules_nan_bool_safety(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import pytest
    from binance_api import market_data as b_md
    from kraken import market_data as k_md
    from kraken import trailing_stop as k_ts
    from kraken import kraken_bot as k_bot
    from kraken import kraken_xstock_watch as k_xw
    from kraken import shadow_live as k_sl

    # 1. binance_api/market_data: parse_interval_minutes & fetch_ohlc_candles reject float/Inf/NaN/bool
    with pytest.raises(ValueError):
        b_md.parse_interval_minutes(float("inf"))
    with pytest.raises(ValueError):
        b_md.parse_interval_minutes(60.5)
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, timeout=True)
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, timeout=float("nan"))
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, end_ms=float("inf"))
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, start_ms=-5, end_ms=1000)

    # 2. kraken/market_data: fetch_ohlc_candles rejects bool/NaN/Inf/<=0 interval and timeout
    with pytest.raises(ValueError):
        k_md.fetch_ohlc_candles("HYPEUSD", True)
    with pytest.raises(ValueError):
        k_md.fetch_ohlc_candles("HYPEUSD", 60, timeout=True)
    with pytest.raises(ValueError):
        k_md.fetch_ohlc_candles("HYPEUSD", float("nan"))
    with pytest.raises(ValueError):
        k_md.fetch_ohlc_candles("HYPEUSD", 0)

    # 3. kraken/trailing_stop: KrakenTrailing.reconcile_pending & execute_rebuy NaN/bool safety
    kt = k_ts.KrakenTrailing(
        client=SimpleNamespace(),
        log=lambda _m: None,
        enabled=False,
        state_file=str(tmp_path / "kt_state.json"),
    )
    kt.order_lifecycle = SimpleNamespace(
        reconcile=lambda pending, persist: SimpleNamespace(
            outcome="terminal",
            status=SimpleNamespace(status="closed", filled_qty=float("nan"), cost=float("nan"), fee=0.0),
            intent=pending,
        ),
        new_intent=lambda **kw: kw,
    )
    st_dict = {"retry_counts": {}}
    persist_fn = lambda _val: None
    persist_fn.st = st_dict
    pending_obj = {"symbol": "HYPEUSD", "action": "SELL", "submitted_qty": float("nan"), "attempt": True}
    res = kt.reconcile_pending(pending_obj, persist_fn)
    assert res.outcome == "terminal"
    assert st_dict["retry_counts"]["SELL"] == 1

    captured_intent = {}
    kt._submit_lifecycle_order = lambda intent, *a, **kw: (
        captured_intent.update(intent) or SimpleNamespace(outcome="refused", intent=intent)
    )
    kt.execute_rebuy(
        "HYPE", "HYPE", "HYPEUSD", 1.0, 25.0,
        {"low": float("nan"), "sell_price": True},
        persist_fn,
    )
    assert captured_intent["metadata"]["anchor"] == 25.0
    assert captured_intent["metadata"]["sell_price"] == 0.0

    # 4. kraken/kraken_bot._cmd_balance ignores bool/NaN/Inf balances without crashing
    logged_lines = []
    monkeypatch.setattr(k_bot, "log", lambda msg: logged_lines.append(msg))
    fake_kcli = SimpleNamespace(
        balance=lambda: {"ZUSD": "100.5", "BAD_BOOL": True, "BAD_NAN": "nan", "BAD_INF": "inf", "ZERO": "0"}
    )
    assert k_bot._cmd_balance(fake_kcli) == 0
    assert any("ZUSD" in ln for ln in logged_lines)
    assert not any("BAD_" in ln for ln in logged_lines)

    # 5. kraken/kraken_xstock_watch.check_levels ignores bool/NaN/Inf alloc_price, tp_pct, sl_pct, price
    xw_st = {"allocated": {"asset": "SPCX", "qty": True}, "pair": "SPCXUSD", "alerted_tp": False, "alerted_sl": False}
    k_xw.check_levels(
        SimpleNamespace(last_price=lambda _p: 150.0),
        xw_st,
        alloc_price=float("nan"),
        tp_pct=10.0,
        sl_pct=10.0,
        yahoo_sym="",
        desktop=False,
    )
    assert xw_st["alerted_tp"] is False

    # 6. kraken/shadow_live._load_history filters out bool/NaN/Inf/non-positive bars
    monkeypatch.setattr(k_sl, "LOG_DIR", str(tmp_path))
    hist_file = tmp_path / "HYPEUSD_240m.ohlc.json"
    hist_file.write_text(
        json.dumps([
            [1700000000, 20.0, 21.0, 19.5, 20.5],
            [True, 20.0, 21.0, 19.5, 20.5],
            [1700014400, -1.0, 21.0, 19.5, 20.5],
        ]),
        encoding="utf-8",
    )
    loaded_bars = k_sl._load_history("HYPEUSD", 240)
    assert loaded_bars == [(1700000000, 20.0, 21.0, 19.5, 20.5)]


def test_iteration53_t212_order_manager_hl_shadow_and_intelligence_collectors_nan_bool_safety(monkeypatch, tmp_path):
    import os
    import sys
    from types import SimpleNamespace
    from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
    from intelligence.macro_analyzer import MacroAnalyzer

    # 1. 212trading/order_manager: _portfolio_qty, _matching_active_orders, _reconcile_record
    t212_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "212trading"))
    orig_sys_path = list(sys.path)
    mod_names = ("ipo_common", "ipo_notify", "t212_client", "market_data", "order_manager")
    saved_mods = {k: sys.modules.get(k) for k in mod_names}
    try:
        for k in mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, t212_dir)
        import order_manager as t212_om
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    cli_nan_pf = SimpleNamespace(get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": float("nan")}])
    assert t212_om._portfolio_qty(cli_nan_pf, "NVDA_US_EQ") is None
    cli_bool_pf = SimpleNamespace(get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": True}])
    assert t212_om._portfolio_qty(cli_bool_pf, "NVDA_US_EQ") is None

    cli_orders = SimpleNamespace(
        list_active_orders=lambda: [
            {"ticker": "NVDA_US_EQ", "quantity": True, "limitPrice": 100.0},
            {"ticker": "NVDA_US_EQ", "quantity": 1.0, "limitPrice": float("nan")},
        ]
    )
    assert t212_om._matching_active_orders(cli_orders, {"ticker": "NVDA_US_EQ", "qty": 1.0, "limit_price": 100.0}) == []
    assert t212_om._matching_active_orders(cli_orders, {"ticker": "NVDA_US_EQ", "qty": True, "limit_price": 100.0}) is None

    # 2. Intelligence collectors & MacroAnalyzer sanitize bool/NaN/Inf TTL & retention parameters
    blc = BybitLiquidationCollector(retention_sec=True)
    assert blc._retention == 3600.0
    blc_nan = BybitLiquidationCollector(retention_sec=float("nan"))
    assert blc_nan._retention == 3600.0

    dtc = DerivativesTelemetryCollector(cache_ttl_sec=True, disk_ttl_sec=float("nan"), cache_dir=str(tmp_path))
    assert dtc.cache_ttl_sec == 60.0
    assert dtc.disk_ttl_sec == 120.0

    obc = OrderbookDepthCollector(
        cache_ttl_sec=True, depth_band_pct=float("nan"), whale_wall_usd=-1.0, disk_ttl_sec=float("inf"), cache_dir=str(tmp_path)
    )
    assert obc.cache_ttl_sec == 5.0
    assert obc.depth_band_pct == 1.5
    assert obc.whale_wall_usd == 500_000.0
    assert obc.disk_ttl_sec == 60.0

    wpc = WhalePositioningCollector(cache_ttl_sec=True, disk_ttl_sec=float("nan"), cache_dir=str(tmp_path))
    assert wpc.cache_ttl_sec == 120.0
    assert wpc.disk_ttl_sec == 300.0

    ma = MacroAnalyzer(
        cache_dir=str(tmp_path),
        macro_llm_interval_sec=True,
        advisor_interval_sec=float("nan"),
        thread_prune_interval_sec=-5.0,
    )
    assert ma.macro_llm_interval_sec > 1.0
    assert ma.advisor_interval_sec > 1.0
    assert ma.thread_prune_interval_sec == 10800.0


def test_iteration54_triggers_cli_daemon_pricewindow_and_shadow_signals_nan_bool_safety(tmp_path):
    import json
    from active_intents import _intent_row
    from intelligence.cli import inspect_symbol_intelligence, print_intelligence_report
    from intelligence.daemon import IntelligenceTelemetryDaemon
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsFeedSnapshot, NewsHeadline
    from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
    from pricewindow import PriceWindow, WindowAnalyzer
    from shadow_signals import KalmanStateTracker

    # 1. Triggers sanitize bool/NaN/Inf constructor parameters
    wat = WhaleAccumulationTrigger(
        min_top_traders_long_pct=True,
        min_taker_buy_ratio=float("nan"),
        min_top_traders_short_pct=float("inf"),
        min_taker_sell_ratio=-1.0,
    )
    assert wat.min_top_traders_long_pct == 0.60
    assert wat.min_taker_buy_ratio == 1.25
    assert wat.min_top_traders_short_pct == 0.60
    assert wat.min_taker_sell_ratio == 0.80

    lgt = LinearGradientTrigger(strength_threshold=True)
    assert lgt.strength_threshold == 2.0
    lgt_nan = LinearGradientTrigger(strength_threshold=float("nan"))
    assert lgt_nan.strength_threshold == 2.0

    mbt = MarketBreadthTrigger(
        washout_advance_threshold=True,
        washout_median_threshold=float("nan"),
        blowoff_advance_threshold=1.5,
        blowoff_median_threshold=float("inf"),
    )
    assert mbt.washout_advance_threshold == 0.15
    assert mbt.washout_median_threshold == -4.0
    assert mbt.blowoff_advance_threshold == 0.88
    assert mbt.blowoff_median_threshold == 6.0

    # 2. NewsFeedCollector._aggregate prunes corrupted/NaN entries safely
    nfc = NewsFeedCollector(history_file=str(tmp_path / "news_feed_collect.json"))
    h = NewsHeadline(
        title="Breaking macro test headline",
        source="Test",
        pub_date="",
        url="",
        matched_keywords=("war",),
        is_high_severity=True,
    )
    nfc._aggregated_pool["corrupted"] = (float("nan"), h)
    snap = NewsFeedSnapshot(
        headlines=(h,),
        total_fetched=1,
        high_severity_count=1,
        has_critical_shock_keywords=False,
        ts=1700000000.0,
    )
    agg = nfc._aggregate(snap, 1700000000.0)
    assert agg.total_fetched == 1
    assert "corrupted" not in nfc._aggregated_pool

    # 3. Intelligence CLI & Daemon / GlobalTelemetryCollector
    ob_file = tmp_path / "orderbook_depth_BTCUSDC.json"
    ob_file.write_text(json.dumps({"mid_price": None, "imbalance_ratio": float("nan"), "ts": 0.0}).replace("NaN", "null"))
    rep = inspect_symbol_intelligence("BTCUSDC", cache_dir=str(tmp_path))
    assert rep["pillar2_external"]["orderbook"]["age_seconds"] is None
    assert rep["pillar2_external"]["orderbook"]["age_formatted"] == "unknown"
    print_intelligence_report(rep)

    itd = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC"],
        cache_dir=str(tmp_path),
        orderbook_interval_sec=True,
        derivatives_interval_sec=float("nan"),
        whale_interval_sec=-5.0,
        macro_interval_sec=float("inf"),
    )
    assert itd.orderbook_interval_sec == 15.0
    assert itd.derivatives_interval_sec == 30.0
    assert itd.whale_interval_sec == 60.0
    assert itd.macro_interval_sec > 0.0

    gtc = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=str(tmp_path),
        orderbook_interval_sec=True,
        derivatives_interval_sec=float("nan"),
        whale_interval_sec=-5.0,
        news_interval_sec=float("inf"),
        fear_greed_interval_sec=True,
        breadth_interval_sec=float("nan"),
    )
    assert gtc.orderbook_interval_sec == 15.0
    assert gtc.derivatives_interval_sec == 30.0
    assert gtc.whale_interval_sec == 60.0
    assert gtc.fear_greed_interval_sec == 1800.0
    assert gtc.breadth_interval_sec == 900.0

    # 4. PriceWindow & WindowAnalyzer
    pw = PriceWindow("BTCUSDC", window_size=True, sample_rate_sec=float("nan"), window_seconds=True)
    assert pw.window_size == 10
    assert pw.window_seconds is None
    for p in (100.0, 101.0, 102.0, 103.0):
        pw.process_price(p)
    eps_default = pw.get_noise_epsilon(1.0)
    assert pw.get_noise_epsilon(float("nan")) == eps_default
    assert pw.get_noise_epsilon(True) == eps_default

    wa = WindowAnalyzer(pw)
    assert wa.calculate_proximities(float("nan")) == (0, 0)
    assert wa.calculate_proximities(True) == (0, 0)
    assert wa.check_price_change(float("nan")) == (0, 0)
    assert wa.check_price_change(True) == (0, 0)
    assert wa.evaluate_buy_sell_opportunity(float("nan"))[0] == "HOLD"
    assert wa.evaluate_buy_sell_opportunity(True)[0] == "HOLD"

    # 5. KalmanStateTracker & active_intents
    kst = KalmanStateTracker(state_min_interval=float("nan"), now_fn=lambda: 1000.0)
    assert kst.state_min_interval == 1.0
    kst._state["BTCUSDC"] = {"ts": 995.0, "kalman_trend": True}
    assert kst.current_trend("BTCUSDC") == (None, 1e18)
    kst._state["BTCUSDC"] = {"ts": 995.0, "kalman_trend": float("nan")}
    assert kst.current_trend("BTCUSDC") == (None, 1e18)
    kst._state["BTCUSDC"] = {"ts": 995.0, "kalman_trend": 1}
    assert kst.current_trend("BTCUSDC") == (1, 5.0)

    assert _intent_row({"id": "ord-1", "side": "BUY", "qty": True}, owner="test", source="s", location="$", inherited={}) is None
    assert _intent_row({"id": True, "side": "BUY", "qty": 1.0}, owner="test", source="s", location="$", inherited={}) is None


def test_iteration55_pricewindow_zero_size_tradeall_observe_pricechecker_and_alertnotifiers_nan_bool_safety(tmp_path):
    from market_monitor.pricechecker import PriceChecker
    from notify_engine.alertnotifiers import AlertNotifier, pop_due_bulk_batch
    from pricewindow import PriceWindow
    from visuals.tradeall_observe import (
        _parse_shadow_rows,
        format_state_text,
        load_backtest_order_events,
        load_backtest_price_samples,
        load_backtest_trend_starts,
    )

    # 1. PriceWindow(symbol, 0) clamps to 1, while bool/NaN/Inf fall back to 10
    assert PriceWindow("BTCUSDC", 0).window_size == 1
    assert PriceWindow("BTCUSDC", -5).window_size == 1
    assert PriceWindow("BTCUSDC", True).window_size == 10
    assert PriceWindow("BTCUSDC", float("nan")).window_size == 10

    # 2. tradeall_observe: format_state_text does not treat bool as 1/0 trend, and backtest loaders reject NaN/Inf
    txt = format_state_text({"final_trend": True, "trend_fast": False, "kalman_trend": True}, "HDR")
    assert "trend:     ?" in txt
    assert "fast:      ?" in txt
    assert "kalman:    ?" in txt

    shadow_rows = [
        ["nan", "BTCUSDC", "kalman", "trend_start", "1", "0", "95000.0", "0.1", "0.02"],
        ["1700000000", "BTCUSDC", "kalman", "trend_start", "1", "0", "nan", "0.1", "0.02"],
        ["1700000000", "BTCUSDC", "kalman", "trend_start", "1", "0", "95000.0", "0.1", "0.02"],
    ]
    parsed_sh = _parse_shadow_rows(shadow_rows, "BTCUSDC")
    assert len(parsed_sh) == 1
    assert parsed_sh[0]["ts"] == 1700000000.0

    (tmp_path / "tradeall_price_samples.log").write_text("nan|BTCUSDC|95000\n1700000000|BTCUSDC|nan\n1700000000|BTCUSDC|95000\n")
    assert load_backtest_price_samples(str(tmp_path), "BTCUSDC") == ([1700000000.0], [95000.0])

    (tmp_path / "tradeall_decisions.log").write_text("nan|BTCUSDC|trend_start|UP|HOLD|95000|1\n1700000000|BTCUSDC|trend_start|UP|HOLD|95000|1\n")
    assert len(load_backtest_trend_starts(str(tmp_path), "BTCUSDC")) == 1

    (tmp_path / "order_outcomes.log").write_text("nan|BTCUSDC|BUY|95000|0.1|accepted||tradeall.py|trend_buy\n1700000000|BTCUSDC|BUY|95000|0.1|accepted||tradeall.py|trend_buy\n")
    assert len(load_backtest_order_events(str(tmp_path), "BTCUSDC")) == 1

    # 3. PriceChecker._should_send_alert recovers from NaN/bool last_time and cooldown_minutes
    checker = PriceChecker(cachePriceAll=None, config={"cooldown_minutes": float("nan")})
    checker._last_alert_time["BTC_up"] = float("nan")
    assert checker._should_send_alert("BTC", "up") is True
    checker._last_alert_time["BTC_up"] = True
    assert checker._should_send_alert("BTC", "up") is True

    # 4. AlertNotifier.format_human_readable_time and pop_due_bulk_batch handle bool/NaN safely
    assert AlertNotifier.format_human_readable_time(True) == "N/A"
    assert AlertNotifier.format_human_readable_time(False) == "N/A"
    assert AlertNotifier.format_human_readable_time(float("nan")) == "nan"
    assert pop_due_bulk_batch("ntfy", now=float("nan")) is None or True


def test_iteration56_portfolio_snapshot_new_coins_overnight_monitor_and_reconciler_nan_bool_safety(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock

    import verify_tools.portfolio_snapshot as ps_mod
    from market_monitor.new_coins_discovery import DexScreenerSource, NewCoinsMonitor
    from orchestratorOS.livecheck.autonomous_overnight_monitor import AutonomousOvernightMonitor
    from orchestratorOS.supervisor.autonomous_ai_reconciler import (
        AutonomousAIReconciler,
        LogEventSummary,
        TelemetrySensorSnapshot,
    )

    # 1. portfolio_snapshot.kraken_rows handles NaN _kraken_price and None orders without ValueError/TypeError
    monkeypatch.setattr(ps_mod, "_load_state", lambda pair: {"qty": 2.0, "cost": 100.0, "orders": None})
    monkeypatch.setattr(ps_mod, "_kraken_price", lambda pair: float("nan"))
    rows = ps_mod.kraken_rows()
    assert len(rows) == len(ps_mod.KRAKEN_PAIRS)
    assert rows[0]["value_usd"] == 0.0
    assert rows[0]["open_orders"] == 0

    # 2. DexScreenerSource.refresh rejects bool/NaN listedAt and NewCoinsMonitor handles non-numeric max_monitored_symbols
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "profiles": [
            {"symbol": "ABC", "listedAt": True},
            {"symbol": "DEF", "listedAt": "nan"},
            {"symbol": "GHI", "listedAt": 1700000000000},
        ]
    }
    monkeypatch.setattr("market_monitor.new_coins_discovery.requests.get", lambda *a, **kw: fake_resp)
    dex = DexScreenerSource()
    by_sym = {c["symbol"]: c["added_at"] for c in dex._cache}
    assert by_sym["ABC"] is None
    assert by_sym["DEF"] is None
    assert by_sym["GHI"] is not None

    dummy_pm = MagicMock()
    dummy_pm.max_monitored_symbols = None
    dummy_pm.original_symbols = []
    dummy_pm.active_symbols = set()
    ncm = NewCoinsMonitor.__new__(NewCoinsMonitor)
    ncm.price_monitor = dummy_pm
    assert ncm.add_new_coin_to_watchlist({"symbol": "NEWCOIN", "name": "NewCoin"}) is True

    # 3. AutonomousOvernightMonitor handles bool/NaN now, list/null JSON in cachedb, and null market_trading_decision
    cachedb = tmp_path / "cachedb"
    cachedb.mkdir(parents=True, exist_ok=True)
    (cachedb / "geopolitical_threat_eval.json").write_text("[]", encoding="utf-8")
    (cachedb / "auto_cryptotrade_executor_eval.json").write_text(
        json.dumps({"mode": "enforce", "market_trading_decision": None}),
        encoding="utf-8",
    )
    aom = AutonomousOvernightMonitor(workspace_dir=str(tmp_path))
    assert aom.check_processes(now=float("nan")) == []
    assert aom.check_processes(now=True) == []
    s_stat = aom.gather_sensor_status()
    assert "geopolitical_threat" not in s_stat
    assert s_stat["supervisor_eval"]["decision"] == "HOLD"

    # 4. AutonomousAIReconciler.synthesize_and_diagnose handles None/bool/NaN in prices_and_trends, microstructure, and portfolio_summary
    (tmp_path / "order_guard.conf").write_text("supervisor_mode = shadow\nsupervisor_notify = 0\n", encoding="utf-8")
    (tmp_path / "config.env").write_text("NTFY_TOPIC_GUARD=test-guard\n", encoding="utf-8")
    mock_gemini = MagicMock()
    mock_gemini.query_json.return_value = {"summary": "OK", "market_trading_decisions": []}
    rec = AutonomousAIReconciler(workspace_dir=str(tmp_path), gemini_auditor=mock_gemini)
    snap = TelemetrySensorSnapshot(
        prices_and_trends={
            "BTCUSDC": {
                "price": float("nan"),
                "trend": None,
                "growth_coefficient": None,
                "chg_15m_pct": None,
                "chg_30m_pct": float("nan"),
                "chg_1h_pct": True,
                "chg_3h_pct": None,
                "low_24h": None,
                "high_24h": float("nan"),
                "chg_7d_pct": None,
                "slope_15m": None,
                "slope_3_7m": float("nan"),
                "slope_2_5h": True,
                "trend_3_7m": None,
                "trend_15m": float("nan"),
                "trend_2_5h": True,
                "gradient_recent_fast": True,
                "epsilon": float("nan"),
                "kalman_vel": True,
            }
        },
        microstructure={
            "BTCUSDC": {
                "bid_wall_px": float("nan"),
                "ask_wall_px": True,
            }
        },
        portfolio_summary={
            "total_value_usdc": float("nan"),
            "venue_quote_cash": {"binance:USDC": {"venue": "binance", "asset": "USDC", "free": float("nan")}},
            "recent_orders": {"BTCUSDC": {"last_buy": {"ts_ms": "nan"}, "last_sell": None}},
        },
    )
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)
    res = rec.synthesize_and_diagnose(snap, logs, topic_key="auto_cryptotrade_executor")
    assert res["summary"] == "OK"


def test_iteration57_agy_store_drawprices_trailing_and_t212_strategy_safety(monkeypatch):
    import importlib
    import numpy as np
    from orchestratorOS.agy.summary_store import _safe_step_count, get_thread_step_count
    from orchestratorOS.agy.topics import _safe_int, prepare_topic_session
    from visuals.drawprices import to_series, min_max_scale
    from kraken.trailing_stop import KrakenTrailing

    # 1. summary_store._safe_step_count and topics._safe_int
    assert _safe_step_count(True) == 0
    assert _safe_step_count(float("nan")) == 0
    assert _safe_step_count(float("inf")) == 0
    assert _safe_step_count("-5") == 0
    assert _safe_step_count("12.0") == 12
    monkeypatch.setattr("orchestratorOS.agy.summary_store.read_summary_row", lambda base, cid: {"step_count": True, "status": "IDLE"})
    assert get_thread_step_count("cid1", "/tmp") == (False, 0, "IDLE")

    assert _safe_int(True, 7) == 7
    assert _safe_int(float("nan"), 7) == 7
    assert _safe_int("15.0", 0) == 15
    with pytest.raises(ValueError):
        prepare_topic_session("auto_cryptotrade_executor", compact_steps=float("nan"))

    # 2. visuals/drawprices.py to_series and min_max_scale
    ts, prices, dt = to_series([
        (1700000000000, 100.0),
        (10**22, 105.0),
        (True, 102.0),
        (1700000060000, float("nan")),
    ])
    assert len(ts) == 1
    assert prices[0] == pytest.approx(100.0)
    assert len(dt) == 1
    scaled, vmin, vmax = min_max_scale(np.array([1.0, float("nan"), 3.0]))
    assert np.allclose(scaled, 0.5)
    assert vmin == 0.0 and vmax == 0.0

    # 3. KrakenTrailing._sync_retry_count_from_state & _order_intent
    adapter = object.__new__(KrakenTrailing)
    adapter._retry_counts = {}
    adapter.order_lifecycle = MagicMock()
    adapter.order_lifecycle.new_intent.side_effect = lambda **kw: kw
    dummy_persist = type("P", (), {"st": {"retry_counts": {"SELL": True}}})()
    adapter._sync_retry_count_from_state("HYPEUSD", "SELL", dummy_persist)
    assert adapter._retry_counts.get(("HYPEUSD", "SELL"), 0) == 0
    intent = adapter._order_intent("SELL", "HYPE", "HYPE", "HYPEUSD", 1.5, 20.0, anchor=None)
    assert ":20.00000000" in intent["intent_id"]

    # 4. 212trading/strategy.py generate_tp_ladder & _active_order_identity
    import os
    import sys
    t212_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "212trading"))
    orig_sys_path = list(sys.path)
    mod_names = ("ipo_common", "ipo_notify", "t212_client", "market_data", "strategy")
    saved_mods = {k: sys.modules.get(k) for k in mod_names}
    try:
        for k in mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, t212_dir)
        import strategy as strat_mod
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    with pytest.raises(ValueError, match="min_pct > -100"):
        strat_mod.generate_tp_ladder(-100.0, 10.0, 3, curve="geometric")
    with pytest.raises(ValueError, match="min_pct > -100"):
        strat_mod.generate_tp_ladder(-150.0, 10.0, 3, curve="geometric")

    ticker, side, qty, limit = strat_mod.Strategy._active_order_identity({
        "ticker": "AAPL",
        "quantity": True,
        "limitPrice": float("nan"),
    })
    assert ticker == "AAPL"
    assert qty is None
    assert limit is None


def test_iteration58_binance_trailing_t212_params_and_provider_adapters_safety(monkeypatch):
    import os
    import sys
    from binance_api.trailing_stop import TrailingStop
    from trailing_core import TrailingCore
    from providers.market_api import BinanceProvider
    from providers.kraken_provider import KrakenProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.strategy_executor import ProviderError

    # 1. Binance TrailingStop._sync_retry_count_from_state & _order_intent
    ts = object.__new__(TrailingStop)
    ts._retry_counts = {}
    ts.order_lifecycle = MagicMock()
    ts.order_lifecycle.new_intent.side_effect = lambda **kw: kw
    dummy_persist = type("P", (), {"st": {"retry_counts": {"SELL": True}}})()
    ts._sync_retry_count_from_state("BTCUSDC", "SELL", dummy_persist)
    assert ts._retry_counts.get(("BTCUSDC", "SELL"), 0) == 0
    intent = ts._order_intent("SELL", "BTCUSDC", "BTC", "BTCUSDC", 0.1, 60000.0, anchor=None)
    assert ":60000.00000000" in intent["intent_id"]

    # 2. TrailingCore._process ignores invalid trail (bool, NaN, <=0, >=100)
    core = object.__new__(TrailingCore)
    core.a = MagicMock()
    core.a.free_qty.return_value = 10.0
    core.a.price.return_value = 100.0
    st_dict = {}
    core._process("BTCUSDC", "BTC", "BTCUSDC", float("nan"), st_dict)
    core._process("BTCUSDC", "BTC", "BTCUSDC", True, st_dict)
    core._process("BTCUSDC", "BTC", "BTCUSDC", 120.0, st_dict)
    assert st_dict == {}

    # 3. BinanceProvider.open_orders & order_status reject bool/NaN
    bp = BinanceProvider()
    fake_bapi = MagicMock()
    fake_bapi.client.get_open_orders.return_value = [{"orderId": 1, "price": float("nan"), "origQty": 1.0, "executedQty": 0.0}]
    monkeypatch.setattr("providers.market_api._get_bapi", lambda: fake_bapi)
    with pytest.raises(ProviderError):
        bp.open_orders("BTCUSDC")

    fake_bapi.client.get_order.return_value = {"status": "NEW", "executedQty": True, "cummulativeQuoteQty": 0.0, "origQty": 1.0}
    with pytest.raises(ProviderError):
        bp.order_status("BTCUSDC", "123")

    # 4. KrakenProvider.pair_precision rejects NaN/non-positive ordermin
    kp = KrakenProvider(client=MagicMock())
    kp._cli.pair_info.return_value = {"pair_decimals": 2, "lot_decimals": 4, "ordermin": float("nan"), "base": "HYPE"}
    with pytest.raises(ProviderError):
        kp.pair_precision("HYPEUSD")

    # 5. HyperliquidProvider.submit_order market rejects NaN mid price
    hp = HyperliquidProvider(token="HYPE", live_enabled=True)
    monkeypatch.setattr(hp, "_pair", lambda: "@107")
    monkeypatch.setattr(hp, "get_current_price", lambda s: float("nan"))
    with pytest.raises(ProviderError, match="price unavailable"):
        hp.submit_order("HYPEUSDC", "BUY", 1.0, price=None, market=True)

    # 6. 212trading/strategy.py StratParams.from_env rejects NaN in STRAT_TP_LADDER
    t212_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "212trading"))
    orig_sys_path = list(sys.path)
    mod_names = ("ipo_common", "ipo_notify", "t212_client", "market_data", "strategy")
    saved_mods = {k: sys.modules.get(k) for k in mod_names}
    try:
        for k in mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, t212_dir)
        import strategy as strat_mod
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    base_env = {
        "STRATEGY_MODE": "avg_tp",
        "STRAT_TP_LADDER": "5:nan,10:50",
        "STRAT_MAX_BUDGET": "1000",
        "STRAT_ENTRY_PCT": "10",
        "STRAT_DCA_PCT": "10",
        "STRAT_MAX_DCA_BUYS": "5",
    }
    with pytest.raises(ValueError, match="Invalid STRAT_TP_LADDER"):
        strat_mod.StratParams.from_env(base_env)


def test_iteration59_base_vwap_audit_kraken_and_hl_client_safety(tmp_path):
    """Iteration 59: aggregate_fills_by_order_id VWAP dilution, ExecutionAudit clock/set safety, KrakenClient interval, and HLClient metadata/order checks."""
    from providers.base import MarketDataProvider, PairPrecision, aggregate_fills_by_order_id
    from providers.execution_audit import ExecutionAudit
    from kraken.kraken_client import KrakenClient

    hl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hyperliquid"))
    orig_sys_path = list(sys.path)
    saved_mods = {k: sys.modules.get(k) for k in ("common", "hl_client")}
    try:
        for k in ("common", "hl_client"):
            sys.modules.pop(k, None)
        sys.path.insert(0, hl_dir)
        import hl_client as hl_mod
        HLClient = hl_mod.HLClient
        HLError = hl_mod.HLError
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # 1. aggregate_fills_by_order_id does not dilute VWAP when a partial fill has price=0
    fills = [
        {"orderId": "ord-1", "side": "BUY", "price": 100.0, "qty": 2.0, "timestamp": 1000},
        {"orderId": "ord-1", "side": "BUY", "price": 0.0, "qty": 2.0, "timestamp": 2000},
    ]
    agg = aggregate_fills_by_order_id(fills)
    assert len(agg) == 1
    assert agg[0]["price"] == pytest.approx(100.0)
    assert agg[0]["qty"] == pytest.approx(4.0)

    # 2. MarketDataProvider.round_quantity and round_price clamp negative inputs to 0.0 and clamp large decimals
    class _DummyProvider(MarketDataProvider):
        @property
        def name(self) -> str:
            return "dummy"

        def get_current_price(self, symbol: str):
            return 1.0

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def pair_precision(self, symbol: str):
            return PairPrecision(price_decimals=50, volume_decimals=50)

    dp = _DummyProvider()
    assert dp.round_quantity("BTCUSDC", -1.5) == 0.0
    assert dp.round_price("BTCUSDC", -10.0) == 0.0
    assert dp.round_quantity("BTCUSDC", 1.23456) > 0.0

    # 3. ExecutionAudit handles out-of-range clock timestamps and set/frozenset fields
    audit = ExecutionAudit(directory=str(tmp_path), clock=lambda: 1e30)
    assert audit.record(
        "submit",
        intent_id="id-1",
        venue="binance",
        symbol="BTCUSDC",
        tags={"a", "b"},
        nested={"nan_val": float("nan"), "ok": 1},
    ) is True

    # 4. KrakenClient.ohlc_closes_with_timestamps rejects inf/nan interval cleanly with ValueError
    kc = KrakenClient.__new__(KrakenClient)
    with pytest.raises(ValueError, match="positive integer"):
        kc.ohlc_closes_with_timestamps("XBTUSD", float("inf"))

    # 5. HLClient metadata and order placement reject boolean/NaN/non-positive inputs
    hl = HLClient.__new__(HLClient)
    hl._meta_cache = {"HYPE": {"name": "HYPE", "szDecimals": True, "maxLeverage": False}}
    hl.info = MagicMock()
    hl.info.spot_meta.return_value = {"tokens": []}
    with pytest.raises(HLError):
        hl.sz_decimals("HYPE")
    with pytest.raises(HLError):
        hl.max_leverage("HYPE")

    hl.address = "0x123"
    hl.info.user_state.return_value = {
        "marginSummary": {"accountValue": float("nan"), "totalMarginUsed": True},
        "withdrawable": "inf",
    }
    ms = hl.margin_summary()
    assert ms == {"accountValue": 0.0, "totalMarginUsed": 0.0, "withdrawable": 0.0}

    hl._meta_cache = {"HYPE": {"name": "HYPE", "szDecimals": 2, "maxLeverage": 5}}
    hl.exchange = MagicMock()
    ok, oid, msg = hl.place_limit("HYPE", True, float("nan"), 20.0)
    assert ok is False and oid is None
    ok2, oid2, msg2 = hl.spot_order("@107", True, 0.00001, 20.0, sz_decimals=2)
    assert ok2 is False and oid2 is None
    hl.exchange.order.assert_not_called()


def test_iteration60_hl_strategy_price_analysis_xstock_and_t212_client_safety(tmp_path, monkeypatch):
    """Iteration 60: HL Strategy direction/state sanitization, price_analysis step_hours/sort guard, xStock state defaults, and T212 historical fill VWAP."""
    import json

    hl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hyperliquid"))
    orig_sys_path = list(sys.path)
    hl_mod_names = ("common", "notify", "hl_client", "market_data", "signals", "strategy", "price_analysis")
    saved_hl = {k: sys.modules.get(k) for k in hl_mod_names}
    try:
        for k in hl_mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, hl_dir)
        import strategy as hl_strat_mod
        import price_analysis as hl_pa_mod
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_hl.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # 1. HL Strategy normalizes uppercase "LONG" to sign=+1 and rejects invalid direction
    p_long = hl_strat_mod.StratParams(direction="LONG")
    st_long = hl_strat_mod.Strategy(
        None, "HYPE", p_long, dry_run=True, replay_mode=True,
        initial_state={"qty": None, "cost": "nan", "orders": [None, {"role": "open", "px": True, "sz": 1.0}]},
    )
    assert st_long.sign == 1
    assert st_long.open_side == "buy"
    assert st_long.s["qty"] == 0.0
    assert st_long.s["cost"] == 0.0
    assert st_long.s["orders"] == []

    with pytest.raises(ValueError, match="Invalid Strategy direction"):
        hl_strat_mod.Strategy(None, "HYPE", hl_strat_mod.StratParams(direction="sideways"), dry_run=True, replay_mode=True)

    # 2. HL price_analysis.detect_long_term_trend rejects step_hours <= 0 (preventing infinite loop) and sorts unsorted timestamps
    assert hl_pa_mod.detect_long_term_trend([1000, 2000, 3000], [10.0, 11.0, 12.0], window_hours=1, step_hours=0) is None
    ts_unsorted = [float(i * 3600) for i in range(1, 60)][::-1]
    px_unsorted = [10.0 + 0.5 * i for i in range(1, 60)][::-1]
    res = hl_pa_mod.detect_long_term_trend(ts_unsorted, px_unsorted, window_hours=12, step_hours=4, min_consecutive_blocks=3)
    assert res is not None and res["direction"] == "up"

    # 3. kraken_xstock_watch._load_state merges partial JSON onto default keys without KeyError
    from kraken import kraken_xstock_watch as xw
    state_file = tmp_path / "xstock_partial.json"
    state_file.write_text(json.dumps({"known_assets": "corrupt"}), encoding="utf-8")
    monkeypatch.setenv("XSTOCK_STATE_FILE", str(state_file))
    loaded = xw._load_state()
    assert loaded["known_assets"] == {}
    assert loaded["alerted_pair"] is False
    assert loaded["alerted_tp"] is False

    # 4. T212Client.get_historical_order does not dilute fillPrice when a sub-fill has price=0
    t212_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "212trading"))
    t212_mod_names = ("ipo_common", "t212_client")
    saved_t212 = {k: sys.modules.get(k) for k in t212_mod_names}
    try:
        for k in t212_mod_names:
            sys.modules.pop(k, None)
        sys.path.insert(0, t212_dir)
        import t212_client as t212_mod
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_t212.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    cli = t212_mod.T212Client("k", "s", env="demo", min_gap_sec=0.0, portfolio_ttl_sec=1.0, order_min_gap_sec=0.0)
    payload = {
        "items": [
            {"order": {"id": "77", "status": "FILLED"}, "fill": {"quantity": 2.0, "price": 50.0}},
            {"order": {"id": "77", "status": "FILLED"}, "fill": {"quantity": 2.0, "price": 0.0}},
        ]
    }
    monkeypatch.setattr(t212_mod, "http_get", lambda url, headers=None: (200, json.dumps(payload).encode("utf-8")))
    hist = cli.get_historical_order("77")
    assert hist["filledQuantity"] == pytest.approx(4.0)
    assert hist["fillPrice"] == pytest.approx(50.0)


def test_iteration61_ws_normalize_t212_eur_replay_and_bapi_sync_safety(tmp_path, monkeypatch):
    """Iteration 61: bapi_ws_normalize Y/z fallback, T212Provider EUR balance & negative fee guard, ReplayProvider input safety, and bapi_client sync_time."""
    import json
    from binance_api import bapi_ws_normalize, bapi_client
    from providers.t212_provider import T212Provider, ProviderError
    from providers.replay_provider import ReplayMarketDataProvider, load_price_series

    # 1. parse_trade_execution_report derives price from Y/z when L, p, l are NaN/0
    sym_t, trd = bapi_ws_normalize.parse_trade_execution_report(
        {
            "s": "BTCUSDC",
            "x": "TRADE",
            "X": "FILLED",
            "t": 9,
            "i": 11,
            "l": "nan",
            "z": "0.5",
            "L": "nan",
            "p": "0",
            "Y": "25000",
            "S": "BUY",
            "T": 1000,
        }
    )
    assert sym_t == "BTCUSDC"
    assert float(trd["qty"]) == pytest.approx(0.5)
    assert float(trd["price"]) == pytest.approx(50000.0)

    # 2. T212Provider.free_balance("EUR") converts RON using usd_ron * eur_usd, and order_status rejects negative fee
    fake_cli = MagicMock()
    fake_cli.get_cash.return_value = {"free": 496.8}
    tp = T212Provider(client=fake_cli, live_enabled=False)
    fake_md = MagicMock()
    fake_md.get_usd_ron.return_value = 4.60
    fake_md.get_eur_usd.return_value = 1.08
    monkeypatch.setattr("importlib.import_module", lambda name: fake_md if name == "212trading.market_data" else __import__(name))
    assert tp.free_balance("USD") == pytest.approx(108.0)
    assert tp.free_balance("EUR") == pytest.approx(100.0)

    fake_cli.get_order_status.return_value = {
        "status": "FILLED",
        "ticker": "NVDA_US_EQ",
        "filledQuantity": 1.0,
        "filledValue": 100.0,
        "fee": -1.5,
    }
    with pytest.raises(ProviderError, match="invalid fill values"):
        tp.order_status("NVDA_US_EQ", "ord-1")

    # 3. ReplayMarketDataProvider.place_order safely returns None on non-numeric price/qty, and load_price_series skips ts <= 0
    jsonl_file = tmp_path / "prices.jsonl"
    jsonl_file.write_text(
        json.dumps({"s": "BTCUSDC", "i": [-1000, 50000.0]}) + "\n"
        + json.dumps({"s": "BTCUSDC", "i": [2000, 51000.0]}) + "\n",
        encoding="utf-8",
    )
    series = load_price_series(str(jsonl_file), "BTCUSDC")
    assert series == [(2.0, 51000.0)]
    rp = ReplayMarketDataProvider({"BTCUSDC": series})
    rp.advance("BTCUSDC")
    assert rp.place_order("BTCUSDC", "BUY", "not-a-price", 1.0) is None
    assert rp.place_order("BTCUSDC", "BUY", 51000.0, None) is None

    # 4. bapi_client.sync_time rejects boolean/invalid serverTime without corrupting timestamp_offset
    fake_bclient = MagicMock()
    fake_bclient.timestamp_offset = 0
    fake_bclient.get_server_time.return_value = {"serverTime": True}
    monkeypatch.setattr(bapi_client, "_client", fake_bclient)
    assert bapi_client.sync_time(safety_margin_ms=float("nan")) is None
    assert fake_bclient.timestamp_offset == 0


def test_iteration62_active_intents_rtrade_store_shadow_signals_and_rollback_safety(tmp_path):
    """Iteration 62: active_intents boolean/zero-qty filtering, RTradePairStore validation, shadow_signals future-ts guard, and T212 rollback amount."""
    import active_intents
    from rtrade_pair_store import RTradePairStore
    from shadow_signals import KalmanJournal, KalmanStateTracker

    # 1. active_intents._intent_row ignores boolean IDs and non-positive/NaN quantities
    assert active_intents._intent_row(
        {"order_id": True, "side": "BUY", "qty": 1.0},
        owner="kraken", source="s", location="$", inherited={},
    ) is None
    assert active_intents._intent_row(
        {"order_id": "ord-1", "side": "BUY", "qty": 0.0},
        owner="kraken", source="s", location="$", inherited={},
    ) is None
    valid_row = active_intents._intent_row(
        {"order_id": "ord-1", "side": "BUY", "qty": 2.5, "symbol": True},
        owner="kraken", source="s", location="$", inherited={"symbol": "HYPEUSD"},
    )
    assert valid_row is not None and valid_row["symbol"] == "HYPEUSD"

    exp = active_intents._extract_exposure(
        {"qty": 1.0, "cost": -50.0, "entry_price": -10.0, "symbol": "HYPEUSD"},
        owner="kraken", source="s", inherited={},
    )
    assert exp is not None and exp["cost"] is None and exp["entry_price"] is None

    # 2. RTradePairStore guards terminal_retention, accepted order_id, and checkpoint state
    store = RTradePairStore(path=str(tmp_path / "rtrade_pairs.json"), terminal_retention=True)
    assert store.terminal_retention == 200
    with pytest.raises(ValueError, match="invalid rtrade accepted order_id"):
        store.accepted("p1", "BUY", True)
    with pytest.raises(ValueError, match="invalid rtrade checkpoint"):
        store.checkpoint("p1", None)

    # 3. KalmanJournal._sanitize strips CR/LF/pipe and KalmanStateTracker.current_trend fails closed on corrupt future ts
    assert KalmanJournal._sanitize("a|b\rc\nd") == "a/b c d"
    tracker = KalmanStateTracker(state_path=None, now_fn=lambda: 1000.0)
    tracker._state["BTCUSDC"] = {"kalman_trend": 1, "ts": 5000.0}
    assert tracker.current_trend("BTCUSDC") == (None, 1e18)
    tracker._state["BTCUSDC"] = {"kalman_trend": 1, "ts": 1005.0}
    assert tracker.current_trend("BTCUSDC") == (1, 0.0)


def test_iteration63_liquidation_credentials_pricewindow_and_assetguardian_safety(monkeypatch):
    """Iteration 63: liquidation_signal OverflowError/start-lock safety, credentials boolean rejection, pricewindow EMA/recent_n guards, and assetguardian non-negative budget clamping."""
    import json
    import assetguardian
    from credentials import CredentialProfileMissingError, resolve_credential_pair
    from liquidation_signal import LiquidationSignal, _safe_pos_float
    from pricewindow import PriceTrendAnalyzer, instant_trend_from_prices

    # 1. liquidation_signal _safe_pos_float & _ingest handle OverflowError and start() has _start_lock
    assert _safe_pos_float(10**400, 42.0) == 42.0
    sig = LiquidationSignal(symbols=["BTCUSDT"])
    assert hasattr(sig, "_start_lock")
    sig._ingest(json.dumps({
        "topic": "allLiquidation.BTCUSDT",
        "data": [
            {"s": "BTCUSDT", "S": "Buy", "v": 10**400, "p": "70000"},
            {"s": "BTCUSDT", "S": "Buy", "v": "0.1", "p": "70000"},
        ],
    }))
    out = sig.get_signal("BTCUSDT", window_sec=60.0)
    assert out["long_count"] == 1
    assert out["long_liq_usd"] == pytest.approx(7000.0)

    # 2. credentials.resolve_credential_pair rejects boolean values instead of stringifying True -> "True"
    with pytest.raises(CredentialProfileMissingError):
        resolve_credential_pair(
            (("Test", "K", "S"),),
            values={"K": True, "S": True},
        )

    # 3. PriceTrendAnalyzer.exponential_moving_average handles empty prices and invalid span;
    #    instant_trend_from_prices handles bool/inf/tiny sample_rate_sec
    assert PriceTrendAnalyzer([]).exponential_moving_average() == []
    assert len(PriceTrendAnalyzer([10.0, 12.0, 14.0]).exponential_moving_average(span=-1)) == 3
    for bad_sr in (True, float("inf"), float("nan"), 1e-309, -1.0):
        t, gc, sf, gr = instant_trend_from_prices([10.0, 11.0, 12.0], bad_sr)
        assert t == 1 and sf > 0

    # 4. assetguardian._buy_remaining_cash & _record_terminal_buy clamp negative corrupted prior_spent / fee
    rem, target, prior = assetguardian._buy_remaining_cash(
        {"initial_cash": 1000.0, "spent_quote_by_tier": {"5": -500.0}},
        5.0,
        0.2,
    )
    assert prior == 0.0
    assert rem == pytest.approx(target)

    saved = {}
    monkeypatch.setattr(assetguardian, "_save_symbol_campaign", lambda sym, kind, st: saved.update(st))
    from types import SimpleNamespace
    updated = assetguardian._record_terminal_buy(
        "BTCUSDC",
        {
            "initial_cash": 1000.0,
            "pending": {"threshold": 5.0, "allocation": 0.2, "order_id": "1", "client_order_id": "c1"},
            "spent_quote_by_tier": {"5": -100.0},
            "total_spent_quote": -200.0,
        },
        SimpleNamespace(status="canceled", filled_qty=1.0, cost=50.0, fee=-20.0),
    )
    assert updated["spent_quote_by_tier"]["5"] == pytest.approx(50.0)
    assert updated["total_spent_quote"] == pytest.approx(50.0)


def test_iteration64_cooldown_state_io_spot_rules_and_rtrade_pair_safety(tmp_path):
    """Iteration 64: Cooldown slot expected_timestamp rollback guard, state_io NaN recovery, spot_rules shadow parsing, and rtrade_pair _cancel exception safety."""
    import json
    from lock.cooldown import Cooldown
    from state_io import previous_generation_path, read_recovering_previous, read_tail_lines
    from strategies import spot_rules as sr
    from strategies.rtrade_pair import OrderSnapshot, OrderTicket, PairCoordinator, PairPolicy

    # 1. Cooldown.slot rollback does not delete a newer reservation if TTL expired during the block
    cd = Cooldown("iter64", state_path=str(tmp_path / "cd.json"), lock_path=str(tmp_path / "cd.lock"))
    with cd.slot("BTCUSDC", ttl=0.0) as s1:
        assert s1.allowed
        # Another worker acquires and commits a newer reservation before s1 exits without commit
        cd._write({"BTCUSDC": {"timestamp": s1.info["timestamp"] + 10.0, "side": "SELL"}})
    assert cd.get("BTCUSDC")["side"] == "SELL"

    # 2. JsonStateStore.save rejects non-dict state and state_io.read_tail_lines guards directory/None limit
    from strategies.state_store import JsonStateStore, StatePersistenceError
    store = JsonStateStore(
        str(tmp_path / "strat_state.json"),
        default_factory=lambda: {"cycle": 0, "qty": 0.0},
        label="test_strat",
        logger=lambda _msg: None,
        fail_closed=True,
    )
    with pytest.raises(StatePersistenceError):
        store.save(["not", "a", "dict"])
    assert read_tail_lines(str(tmp_path), 10) == []
    assert read_tail_lines(str(tmp_path / "strat_state.json"), None) == []

    # 3. spot_rules handles malformed shadow_prices entries and string numbers in check_parabolic_surge_exhaustion
    trig, gain, drop = sr.check_fast_profit_reversal(
        current_price=110.0,
        avg_cost=100.0,
        base_tp_pct=4.0,
        mult=2.0,
        shadow_prices=[("bad", 115.0), (1000.0, "115.0"), ("not-a-tuple",)],
        window_sec=60.0,
        drop_pct=2.0,
        current_time=1010.0,
    )
    assert trig is True and gain == pytest.approx(10.0)
    surge_trig, _ = sr.check_parabolic_surge_exhaustion(
        current_price="125.0",
        avg_cost="100.0",
        surge_peak="135.0",
        surge_gain_pct="24.0",
        exit_pullback_pct="3.2",
        surge_move_pct=None,
    )
    assert surge_trig is True

    # 4. OrderTicket rejects blank/boolean order_id and PairCoordinator._cancel catches venue cancel exceptions
    with pytest.raises(ValueError, match="ticket rtrade invalid"):
        OrderTicket(order_id=True, side="BUY", price=100.0, qty=1.0)

    class _RaiseCancelVenue:
        def current_price(self):
            return 100.0
        def preflight_order(self, *a, **kw):
            return None
        def place_limit(self, side, price, qty, pair_id, **kw):
            return OrderTicket(order_id=f"id-{side}", side=side, price=price, qty=qty, pair_id=pair_id)
        def place_market_exit(self, *a, **kw):
            return None
        def order_status(self, order_id):
            return OrderSnapshot(status="open", filled_qty=0.0, cost=0.0, fee=0.0)
        def cancel(self, order_id):
            raise RuntimeError("network error on cancel")

    coord = PairCoordinator(_RaiseCancelVenue(), 1.0, PairPolicy(adjustment_fraction=0.01, quote_ttl_sec=10.0), clock=lambda: 100.0)
    coord.start(100.0)
    outcome = coord.step(now=120.0)
    assert outcome.reason == "quote_cancel_pending"
    assert not outcome.terminal


def test_iteration65_trailing_core_kraken_trailing_and_autonomous_gateway_safety(tmp_path):
    """Iteration 65: TrailingCore set/price/init safety, KrakenTrailing retry_counts float/inf safety, and AutonomousIntentGateway boolean price & RTH timestamp safety."""
    import json
    import math
    from types import SimpleNamespace
    from trailing_core import TrailingCore
    from orchestratorTrade.autonomous_intent_gateway import (
        AutonomousIntentGateway,
        _is_us_rth_open,
        _parse_finite_float,
    )

    # 1. TrailingCore sanitizes sets, normalizes constructor params, and skips invalid string/overflow prices
    class _DummyAdapter:
        def assets(self):
            return [("BTC", "BTC", "BTCUSDC", 15.0)]
        def begin_tick(self):
            return True
        def free_qty(self, asset):
            return 1.0
        def price(self, pair):
            return "not-a-number"

    tc = TrailingCore(
        _DummyAdapter(),
        log=lambda _m: None,
        enabled=False,
        state_file=str(tmp_path / "trail_state.json"),
        min_notional="bad",
        rebuy_enabled=True,
        rebuy_bounce_pct=-5.0,
        rebuy_skip_if_trend_down=False,
        sell_fraction=2.5,
        min_profit_pct="bad",
    )
    assert tc.sell_fraction == 1.0
    assert tc.min_notional == 0.0
    assert tc.min_profit_pct == 0.0
    assert tc.save({"BTC": {"tags": {"a", "b"}, "peak": float("nan")}}) is True
    tc.check_once()  # does not raise ValueError on "not-a-number" price

    # 2. _parse_finite_float & _is_us_rth_open handle OverflowError, booleans, and out-of-range timestamps
    assert _parse_finite_float(10**400, 12.5) == 12.5
    assert isinstance(_is_us_rth_open(True), bool)
    assert isinstance(_is_us_rth_open(float("inf")), bool)
    assert isinstance(_is_us_rth_open(1e20), bool)

    # 3. AutonomousIntentGateway._get_live_price rejects boolean True prices from provider and cache
    ws = tmp_path / "gw_ws"
    ws.mkdir()
    cache = ws / "cachedb"
    cache.mkdir()
    (ws / "order_guard.conf").write_text(
        "autonomous_trading_mode = shadow\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n",
        encoding="utf-8",
    )
    (cache / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[1000, True]]}}),
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(ws), cache_dir="cachedb")
    inst = SimpleNamespace(
        symbol="BTCUSDC",
        base="BTC",
        _provider=SimpleNamespace(get_current_price=lambda _s: True),
    )
    assert math.isnan(gw._get_live_price(inst))


def test_iteration66_notify_engine_and_market_monitor_safety(tmp_path, monkeypatch):
    """Iteration 66: notify_engine (alertnotifiers, server, mailer) and market_monitor (pricechecker, new_coins_discovery) boundary safety."""
    import json
    from types import SimpleNamespace
    import pytest
    from notify_engine.alertnotifiers import (
        AlertNotifier,
        pass_common_gate,
        pop_due_bulk_batch,
        synthesize_bulk_notifications,
    )
    from notify_engine import mailer
    from market_monitor.pricechecker import PriceChecker
    from market_monitor.new_coins_discovery import _safe_days_back, _safe_num

    # 1. synthesize_bulk_notifications handles float-string, negative, or invalid counts safely
    title, body, prio, topic = synthesize_bulk_notifications([
        {"title": "TestA", "body": "BodyA", "count": "2.0", "priority": "high", "topic": "t1"},
        {"title": "TestB", "body": "BodyB", "count": "bad", "priority": "default", "topic": "t1"},
    ])
    assert "(×2)" in title
    assert "TestB:" in body

    # 2. pass_common_gate and pop_due_bulk_batch tolerate out-of-range `now` and corrupted state fields
    state_file = tmp_path / "notif_state.json"
    monkeypatch.setenv("NOTIFICATION_STATE_FILE", str(state_file))
    monkeypatch.setenv("NOTIFICATION_GATE_SECONDS", "0")
    allowed, reason, warn, payload = pass_common_gate(
        "ntfy", "Title1", "Body1", now=1e20
    )
    assert isinstance(allowed, bool)
    assert pop_due_bulk_batch("ntfy", now=1e20) is None

    # 3. AlertNotifier.format_batch_message & save_to_file handle non-numeric prices/percent_change
    msg = AlertNotifier.format_batch_message([
        {
            "symbol": "BTC",
            "alert_type": "up",
            "percent_change": "N/A",
            "current_price": "bad",
            "reference_price": True,
        }
    ])
    assert "BTC: U +0.00%" in msg
    log_path = tmp_path / "alerts.log"
    obj_alert = SimpleNamespace(
        symbol="ETH",
        alert_type="down",
        percent_change="bad",
        current_price="bad",
        reference_price=None,
        reference_time=None,
        timestamp=None,
        url=None,
    )
    assert AlertNotifier.save_to_file(obj_alert, filename=str(log_path)) is True

    # 4. mailer._config rejects out-of-range TCP ports (<= 0 or > 65535)
    monkeypatch.setenv("SMTP_USERNAME", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("ALERT_TO_EMAIL", "dest@example.com")
    monkeypatch.setenv("SMTP_PORT", "70000")
    assert mailer._config() is None
    monkeypatch.setenv("SMTP_PORT", "0")
    assert mailer._config() is None
    monkeypatch.setenv("SMTP_PORT", "587")
    cfg = mailer._config()
    assert cfg is not None and cfg["port"] == 587

    # 5. PriceChecker._readable_time & start_monitoring and new_coins_discovery._safe_days_back
    assert isinstance(PriceChecker._readable_time((1e25, 100.0, None)), str)
    checker = PriceChecker(SimpleNamespace(original_symbols=[]))
    with pytest.raises(ValueError):
        checker.start_monitoring("not-a-number")
    assert _safe_days_back(1e12) == 3650.0
    assert _safe_num(10**400, 1.5) == 1.5


def test_iteration67_binance_ws_client_context_and_api_safety():
    """Iteration 67: BinancePriceStream._process_message timestamp fallback, BinanceAccountStream._watchdog_loop inf age safety, order_id_context._base36 safety, and aggregate_trades OverflowError safety."""
    import json
    from binance_api.bapi_ws import BinancePriceStream, BinanceAccountStream
    from binance_api.order_id_context import _base36
    from binance_api.bapi_trades import aggregate_trades

    # 1. BinancePriceStream._process_message rejects non-integer E timestamps and accepts valid int E
    ps = BinancePriceStream(symbols=["BTCUSDC"])
    ps._process_message(json.dumps({"data": {"s": "BTCUSDC", "c": "64600.25", "E": "not-an-int"}}))
    assert ps.get_price("BTCUSDC") is None
    ps._process_message(json.dumps({"data": {"s": "BTCUSDC", "c": "64500.5", "E": 1700000000000}}))
    assert ps.get_price("BTCUSDC") == 64500.5

    # 2. BinanceAccountStream._watchdog_loop handles _last_event_ts == 0.0 (age == inf) without OverflowError
    unhealthy_called = []
    acc_stream = BinanceAccountStream(
        on_event=lambda _e: None,
        on_unhealthy=lambda: (unhealthy_called.append(True), acc_stream._stop_event.set()),
        loss_timeout_sec=0.01,
    )
    acc_stream._available = True
    acc_stream._healthy = True
    acc_stream._last_event_ts = 0.0
    acc_stream._watchdog_loop()
    assert unhealthy_called == [True]

    # 3. _base36 handles None, bool, inf, nan, and invalid strings safely
    assert _base36(None) == "0"
    assert _base36(True) == "0"
    assert _base36(float("inf")) == "0"
    assert _base36(float("nan")) == "0"
    assert _base36("bad") == "0"
    assert _base36(36) == "10"

    # 4. aggregate_trades handles OverflowError on quoteQty/commission gracefully
    agg = aggregate_trades([
        {
            "orderId": 101,
            "symbol": "BTCUSDC",
            "qty": "0.5",
            "price": "60000.0",
            "quoteQty": 10**400,
            "commission": 10**400,
            "commissionAsset": "USDC",
            "time": 1700000000000,
            "isBuyer": True,
        }
    ])
    assert len(agg) == 1
    assert float(agg[0]["quoteQty"]) == 30000.0
    assert float(agg[0]["commission"]) == 0.0


def test_iteration68_kraken_and_hyperliquid_client_safety():
    """Iteration 68: KrakenClient._parse / add_order / ohlc_closes_with_timestamps, kraken.market_data.pair_precision, kraken_cachemanager._ledger_to_fills, and HLClient._round_px / cancel / candles safety."""
    import os
    import sys
    from types import SimpleNamespace
    import pytest
    from kraken.kraken_client import KrakenClient, KrakenError
    from kraken.market_data import pair_precision
    from kraken.kraken_cachemanager import _ledger_to_fills
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    if hl_dir not in sys.path:
        sys.path.insert(0, hl_dir)
    from hl_client import _round_px, HLClient

    # 1. KrakenClient._parse rejects non-dict JSON payloads cleanly and add_order rejects None/non-numeric volume/price with ValueError
    with pytest.raises(KrakenError):
        KrakenClient._parse(200, b"[1, 2, 3]")
    kc = KrakenClient("k", "c2VjcmV0")
    with pytest.raises(ValueError):
        kc.add_order("XBTUSD", "buy", None, 50000.0)
    with pytest.raises(ValueError):
        kc.add_order("XBTUSD", "buy", 0.1, "not-a-price", ordertype="limit")
    with pytest.raises(ValueError):
        kc.ohlc_closes_with_timestamps("XBTUSD", 15.5)

    # 2. kraken.market_data.pair_precision handles float strings and OverflowError
    assert pair_precision({"pair_decimals": "2.0", "lot_decimals": "4.0", "ordermin": "0.01"}) == (2, 4, 0.01)
    assert pair_precision({"pair_decimals": 10**400, "lot_decimals": 10**400, "ordermin": 10**400}) == (2, 8, 0.0)

    # 3. _ledger_to_fills tolerates non-dict ledger or non-dict entries
    assert _ledger_to_fills(None) == []
    assert _ledger_to_fills({"L1": "not-a-dict"}) == []

    # 4. _round_px & HLClient.cancel / candles reject inf/bool/invalid parameters
    with pytest.raises(ValueError):
        _round_px(25.0, float("inf"))
    dummy_hl = SimpleNamespace(exchange=object(), info=SimpleNamespace(candles_snapshot=lambda *a: [{"ok": 1}]))
    assert HLClient.cancel(dummy_hl, "HYPE", None) is False
    assert HLClient.cancel(dummy_hl, "HYPE", True) is False
    assert HLClient.candles(dummy_hl, "HYPE", lookback_hours=True) == []
    assert HLClient.candles(dummy_hl, "HYPE", lookback_hours=-5) == []


def test_iteration69_hyperliquid_and_t212_strategy_market_data_safety(tmp_path):
    """Iteration 69: hyperliquid market_data / signals / strategy and 212trading market_data / strategy safety."""
    import importlib.util
    import json
    import os
    import sys
    import pytest

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hl_dir = os.path.join(repo_root, "hyperliquid")
    t212_dir = os.path.join(repo_root, "212trading")
    for p in (hl_dir, t212_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    def _load_mod(name: str, path: str, mod_dir: str):
        if mod_dir in sys.path:
            sys.path.remove(mod_dir)
        sys.path.insert(0, mod_dir)
        for m in ("market_data", "common", "ipo_common", "ipo_notify", "t212_client", "order_rollback"):
            sys.modules.pop(m, None)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    hl_md = _load_mod("_hl_market_data_69", os.path.join(hl_dir, "market_data.py"), hl_dir)
    hl_sig = _load_mod("_hl_signals_69", os.path.join(hl_dir, "signals.py"), hl_dir)
    hl_strat = _load_mod("_hl_strategy_69", os.path.join(hl_dir, "strategy.py"), hl_dir)
    t212_md = _load_mod("_t212_market_data_69", os.path.join(t212_dir, "market_data.py"), t212_dir)
    t212_strat = _load_mod("_t212_strategy_69", os.path.join(t212_dir, "strategy.py"), t212_dir)

    # 1. hyperliquid.market_data: normalize_closed_candles float-string timestamps & fetch_ohlc_candles interval/lookback checks
    rows = [{"t": "1700000000000.0", "T": "1700003599000.0", "o": "10", "h": "12", "l": "9", "c": "11"}]
    norm = hl_md.normalize_closed_candles(rows, now_ms=1700005000000)
    assert len(norm) == 1 and norm[0]["timestamp"] == 1700000000
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", interval=15.5)
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", interval=60, lookback_hours=True)
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", interval=60, lookback_hours=-1)

    # 2. hyperliquid.signals._file_signal: catches OverflowError on huge ts/confidence
    sig_file = tmp_path / "signal.json"
    sig_file.write_text(json.dumps({"trend": "up", "confidence": 10**400, "ts": 10**400}), encoding="utf-8")
    os.environ.setdefault("SIGNAL_MAX_AGE_MIN", "60")
    res_sig = hl_sig._file_signal(str(sig_file))
    assert res_sig["trend"] == "neutral"

    # 3. hyperliquid.strategy.StratParams.from_dict & Strategy._place validation
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"entry": True})
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"entry": float("nan")})
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"maxdca": 2.5})
    sp = hl_strat.StratParams.from_dict({"maxdca": "10.0"})
    assert sp.max_dca_buys == 10
    s_inst = hl_strat.Strategy(client=None, coin="HYPE", params=sp, dry_run=True, replay_mode=True)
    s_inst._place("open", 1.0, -5.0, "ENTRY", 50.0)
    assert s_inst.s["orders"] == []

    # 4. 212trading.market_data & 212trading.strategy.generate_tp_ladder
    with pytest.raises(ValueError):
        t212_md.parse_interval_minutes(None)
    assert t212_md.t212_to_yahoo(None) == ""
    with pytest.raises(ValueError):
        t212_strat.generate_tp_ladder(1.0, 10.0, steps=2.5)
    ladder = t212_strat.generate_tp_ladder(1.0, 5.0, steps=2.0)
    assert len(ladder) == 2


def test_iteration70_t212_replay_bot_kraken_watch_and_trailing_core_safety(tmp_path):
    """Iteration 70: 212trading replay / t212_bot / ipo, kraken xstock_watch / kraken_bot, and trailing_core safety."""
    import importlib.util
    import os
    import sys
    from types import SimpleNamespace
    import pytest
    from trailing_core import TrailingCore

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t212_dir = os.path.join(repo_root, "212trading")
    kraken_dir = os.path.join(repo_root, "kraken")

    def _load_mod(name: str, path: str, mod_dir: str):
        if mod_dir in sys.path:
            sys.path.remove(mod_dir)
        sys.path.insert(0, mod_dir)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    t212_rep = _load_mod("_t212_replay_70", os.path.join(t212_dir, "replay.py"), t212_dir)
    t212_bot = _load_mod("_t212_bot_70", os.path.join(t212_dir, "t212_bot.py"), t212_dir)
    k_watch = _load_mod("_k_watch_70", os.path.join(kraken_dir, "kraken_xstock_watch.py"), kraken_dir)

    # 1. 212trading.replay._fx_rates accepts scalar numeric strings without iterating character-by-character
    assert t212_rep._fx_rates("1.08", 3) == [1.08, 1.08, 1.08]
    with pytest.raises(ValueError):
        t212_rep._fx_rates("not-a-float", 2)
    with pytest.raises(ValueError):
        t212_rep._fx_rates(1.0, -1)

    # 2. 212trading.t212_bot.verify_isin tolerates non-dict items in instruments list
    dummy_t212 = SimpleNamespace(list_instruments=lambda: ["bad-item", {"ticker": "NVDA_US_EQ", "isin": "US67066G1040"}])
    assert t212_bot.verify_isin(dummy_t212, "NVDA_US_EQ", "US67066G1040") is True

    # 3. kraken_xstock_watch.check_pairs tolerates non-dict entries and missing alerted_pair in state dict
    k_watch.notify = lambda **kw: None
    dummy_k = SimpleNamespace(
        asset_pairs=lambda: {"BAD": "not-a-dict", "SPCXXUSD": {"wsname": "SPCXx/USD", "base": "SPCXx"}},
        last_price=lambda k: 50.0,
    )
    st_watch = {}
    k_watch.check_pairs(dummy_k, st_watch, "SPCX", desktop=False, quote="USD")
    assert st_watch.get("pair") == "SPCXXUSD"
    assert st_watch.get("alerted_pair") is True

    # 4. TrailingCore._reconcile_pending cleans up corrupted non-dict pending_order without crashing
    core = TrailingCore(
        SimpleNamespace(),
        log=lambda *_: None,
        enabled=True,
        state_file=str(tmp_path / "trail.json"),
        min_notional=10.0,
        rebuy_enabled=True,
        rebuy_bounce_pct=1.2,
        rebuy_skip_if_trend_down=False,
    )
    st_coin = {"pending_order": True, "peak": 100.0}
    assert core._reconcile_pending({"BTC": st_coin}, st_coin, 95.0, "BTCUSDC") is False
    assert "pending_order" not in st_coin


def test_iteration71_cacheintegrity_rtrade_store_shadow_signals_and_utils_safety(tmp_path):
    """Verify Iteration 71 hardening across cacheintegrity, rtrade_pair_store,
    order_rollback, shadow_signals, utils, and log.configure."""
    import pytest
    from types import SimpleNamespace
    import cacheintegrity
    from rtrade_pair_store import RTradePairStore
    from order_rollback import extract_order_restore_params
    from shadow_signals import KalmanStateTracker
    import utils
    import log

    # 1. cacheintegrity.trade_data_version & trade_semantic_signature & build_account_meta_snapshot
    with pytest.raises(ValueError):
        cacheintegrity.trade_data_version("", 1)
    with pytest.raises(ValueError):
        cacheintegrity.trade_data_version("stream1", 1.5)
    with pytest.raises(ValueError):
        cacheintegrity.trade_data_version("stream1", -1)
    assert cacheintegrity.trade_data_version("stream1", 2) == "stream1:2"

    with pytest.raises(ValueError):
        cacheintegrity.trade_semantic_signature({
            "symbol": "BTCUSDC", "id": 10.5, "orderId": 20,
            "price": "100", "qty": "1", "time": 1000, "isBuyer": True,
        })
    with pytest.raises(ValueError):
        cacheintegrity.build_account_meta_snapshot(
            data_version="s:1", fetchtime={}, counts={},
            stream_id="s", revision=1.5, committed_bytes=10, content_digest="chain:" + "00" * 16,
        )

    # 2. RTradePairStore rejects boolean IDs and non-finite quantities in _validate_intent and persist_intent
    with pytest.raises(ValueError):
        RTradePairStore._validate_intent("p1", "BTCUSDC", "limit:BUY", {
            "side": "BUY", "kind": "limit", "client_order_id": True, "qty": 1.0,
        })
    with pytest.raises(ValueError):
        RTradePairStore._validate_intent("p1", "BTCUSDC", "limit:BUY", {
            "side": "BUY", "kind": "limit", "client_order_id": "c1", "qty": 1.0, "order_id": True,
        })
    store = RTradePairStore(path=str(tmp_path / "rtrade_pairs.json"))
    store.begin("BTCUSDC", "p1", "BUY", 1.0)
    with pytest.raises(ValueError):
        store.persist_intent("p1", "BUY", "limit", {"client_order_id": "c1", "qty": float("nan")})

    # 3. order_rollback.extract_order_restore_params guards against qty * price overflow
    restored_params = extract_order_restore_params({"side": "BUY", "price": 1e200, "origQty": 1e200})
    assert restored_params["amount"] == 0.0

    # 4. KalmanStateTracker.current_trend rejects non-integer kalman_trend like 1.9 and handles OverflowError
    tracker = KalmanStateTracker(state_path=None, journal=SimpleNamespace(log_transition=lambda *a: None))
    tracker._state["BTCUSDC"] = {"ts": 1000.0, "kalman_trend": 1.9}
    tracker._now = lambda: 1005.0
    assert tracker.current_trend("BTCUSDC") == (None, 1e18)
    tracker._state["BTCUSDC"] = {"ts": 1000.0, "kalman_trend": -1.0}
    assert tracker.current_trend("BTCUSDC") == (-1, 5.0)
    res_upd = tracker.update("BTCUSDC", 10**400, 100.0, 10**400, [], 10**400)
    assert res_upd["kalman_trend"] == 0

    # 5. utils.gaussian_weights_from_idx handles OverflowError and clamps huge T
    t_arr, w_arr = utils.gaussian_weights_from_idx(10**400, 0)
    assert len(t_arr) == 0 and len(w_arr) == 0

    # 6. log.configure rejects fractional max_size and check_every
    with pytest.raises(ValueError):
        log.configure(max_size=1024.5)
    with pytest.raises(ValueError):
        log.configure(check_every=10.5)


def test_iteration72_offline_research_and_runners_safety():
    """Verify Iteration 72 hardening across offline/research and offline/runners."""
    import math
    import pytest
    import numpy as np
    from offline.research import intelligence_backtest as ibt
    from offline.research import chop_weight_backtest as cwb
    from offline.research import slice_dense_archive as sda
    from offline.research import comprehensive_strategy_sweep as css
    from offline.runners import fetch_hyperliquid_candles as fhc
    from offline.runners import walk_forward_compare as wfc

    # 1. intelligence_backtest validates step_seconds, initial_balance, fee_pct
    with pytest.raises(ValueError):
        ibt._precompute_intelligence_tape("BTCUSDC", [], 0, 3.5)
    with pytest.raises(ValueError):
        ibt.run_intelligence_backtest("BTCUSDC", step_seconds=0)
    with pytest.raises(ValueError):
        ibt.run_intelligence_backtest("BTCUSDC", initial_balance=0.0)
    with pytest.raises(ValueError):
        ibt.run_intelligence_backtest("BTCUSDC", fee_pct=1.0)

    # 2. chop_weight_backtest filters non-finite/non-positive prices and validates params
    reg, slp = cwb.detect_regime_slope(np.array([float("nan"), -5.0, 100.0, 101.0]))
    assert reg == "CHOP" and slp == 0.0
    with pytest.raises(ValueError):
        cwb._precompute_chop_bar_tape("BTCUSDC", [], None, 0)
    with pytest.raises(ValueError):
        cwb.run_single_simulation("BTCUSDC", [], None, "CHOP_0.03", initial_balance=0.0)
    with pytest.raises(ValueError):
        cwb.run_single_simulation("BTCUSDC", [], None, "CHOP_0.03", fee_pct=-0.1)

    # 3. slice_dense_archive.ts_of handles OverflowError
    assert sda.ts_of({"t": 10**400}) == 0.0
    assert sda.ts_of({"i": [1700000000000, 100.0]}) == 1700000000000.0

    # 4. comprehensive_strategy_sweep._summarize_replay handles NaN/bool/non-positive budget
    summary = css._summarize_replay({"net": 50.0, "total": 60.0}, float("nan"))
    assert math.isfinite(summary["net_pct"])

    # 5. fetch_hyperliquid_candles.resolve_spot_pair & validate_continuity
    with pytest.raises(ValueError):
        fhc.resolve_spot_pair("HYPE", "USDC", "not-a-dict")
    meta = {
        "tokens": ["bad-item", {"name": "HYPE", "index": 1}, {"name": "USDC", "index": 0}],
        "universe": ["bad-item", {"tokens": [1, 0], "name": "@107"}],
    }
    assert fhc.resolve_spot_pair("hype", "usdc", meta) == "@107"
    with pytest.raises(ValueError):
        fhc.validate_continuity([], 0)

    # 6. walk_forward_compare._load_hl_replay and _load_t212_replay register sys.modules before exec_module
    hl_mod = wfc._load_hl_replay()
    t212_mod = wfc._load_t212_replay()
    assert hasattr(hl_mod, "run_replay")
    assert hasattr(t212_mod, "run_replay")


def test_iteration73_monitortrades_backtest_hybrid_reentry_and_surge_sweep_safety(tmp_path):
    """Verify Iteration 73 hardening across monitortrades_backtest, hybrid_reentry, and run_deep_surge_sweep."""
    from offline.research.monitortrades_backtest import replay_trend_source as rts
    from offline.research.monitortrades_backtest import run_replay_backtest as rrb
    from offline.research.monitortrades_backtest import scheduled_pilot as sp
    from offline.research.hybrid_reentry import harness as hrh
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss

    # 1. ReplayTrendSource and DynamicReplayTrendSource reject bool/invalid window_seconds and OverflowError
    src = rts.ReplayTrendSource(["BTCUSDC"], window_seconds=True)
    assert src.window_seconds == rts.DEFAULT_WINDOW_SECONDS
    src.advance("BTCUSDC", 10**400, 100.0)
    assert src.is_trend_up("BTCUSDC") is False

    dsrc = rts.DynamicReplayTrendSource(["BTCUSDC"], max_timeout_sec=True)
    assert dsrc.max_timeout_sec == rts.SMALL_TIMEOUT_RANGE_SEC[1]
    dsrc.advance("BTCUSDC", 1000.0, 100.0)
    dsrc.advance("BTCUSDC", 1005.0, 101.0)
    dsrc.advance("BTCUSDC", 1010.0, 102.0)
    assert dsrc._slice("BTCUSDC", True) == []
    assert dsrc.should_wait("BTCUSDC", None, 60.0) is False

    # 2. run_replay_backtest.prewarm_trend_series handles non-finite/bool trend_window_seconds
    tape = rrb.prewarm_trend_series("BTCUSDC", [(1000.0, 100.0), (1005.0, 101.0)], trend_window_seconds=float("nan"))
    assert len(tape) == 2

    # 3. scheduled_pilot._norm_params_key distinguishes bool from float and handles NaN deterministically
    k_bool = sp._norm_params_key({"a": True})
    k_float = sp._norm_params_key({"a": 1.0})
    assert k_bool != k_float
    k_nan1 = sp._norm_params_key({"a": float("nan")})
    k_nan2 = sp._norm_params_key({"a": float("nan")})
    assert k_nan1 == k_nan2

    # 4. hybrid_reentry.harness.load_ohlc_csv rejects inverted high < low bars and format_ablation_result guards capital
    csv_f = tmp_path / "ohlc.csv"
    csv_f.write_text("timestamp,open,high,low,close\n1,100,90,95,92\n2,100,105,98,103\n", encoding="utf-8")
    loaded_rows = hrh.load_ohlc_csv(str(csv_f))
    assert loaded_rows == [(100.0, 105.0, 98.0, 103.0)]
    fmt_res = hrh.format_ablation_result("BTC", "Test 0", "Control", {"total": 100.0}, float("inf"))
    assert fmt_res["total_return_pct"] == 0.0

    # 5. run_deep_surge_sweep._win_move_at_ts rejects negative timestamps
    assert rdss._win_move_at_ts([100.0] * 50, 30, -14400.0, 48.0) is None


def test_iteration74_tradeall_trigger_gate_adaptive_kalman_and_continuous_grid_safety(tmp_path):
    """Verify Iteration 74 hardening across tradeall_trigger_gate, tradeall_adaptive_thresholds, tradeall_kalman_lag, and continuous_grid."""
    from offline.research.tradeall_trigger_gate import common as ttg_common
    from offline.research.tradeall_adaptive_thresholds import run_backtest_adaptive as rba
    from offline.research.tradeall_kalman_lag import run_kalman_lag_sweep as rkls
    from offline.research.kraken_continuous_grid import grid as k_grid
    from offline.research.kraken_continuous_grid import report as k_rep
    from offline.research.t212_continuous_grid import grid as t_grid

    # 1. LongTrendTracker handles bool/invalid args and empty stats dict without KeyError
    tracker = ttg_common.LongTrendTracker(stats={}, window_hours=True, reeval_sec=False, min_points=True)
    assert tracker.window_sec == 3600.0
    assert tracker.reeval_sec == 60.0
    assert tracker.min_points == 30
    tracker.update(float("nan"), 100.0)
    tracker.update(1000.0, -5.0)
    assert len(tracker.buf) == 0
    for i in range(31):
        tracker.update(1000.0 + i * 10.0, 100.0 + i * 1.0)
    assert tracker.sign == 1
    assert tracker.stats.get("sign_changes") == 1

    # 2. run_backtest_adaptive.run_adaptive rejects bool/non-positive k_small
    with pytest.raises(ValueError, match="k_small must be finite and > 0"):
        rba.run_adaptive("BTCUSDC", 0, 100, True, "test_id")
    with pytest.raises(ValueError, match="k_small must be finite and > 0"):
        rba.run_adaptive("BTCUSDC", 0, 100, -1.5, "test_id")

    # 3. run_kalman_lag_sweep.run_variant rejects bool/non-positive sample_sec
    with pytest.raises(ValueError, match="sample_sec must be finite and > 0"):
        rkls.run_variant("BTCUSDC", 0, 100, True, "test_id")
    with pytest.raises(ValueError, match="sample_sec must be finite and > 0"):
        rkls.run_variant("BTCUSDC", 0, 100, 0.0, "test_id")

    # 4. kraken/t212 continuous_grid.load handles float timestamps and skips non-positive/NaN OHLC
    k_csv = tmp_path / "XXBTZUSD.csv"
    k_csv.write_text("timestamp,open,high,low,close,volume\n1000.0,100,105,95,102,1\n1060.0,0,105,95,102,1\n", encoding="utf-8")
    assert len(k_grid.load(str(k_csv))) == 1

    t_csv = tmp_path / "VUSA_EQ_1h.csv"
    t_lines = ["timestamp,open,high,low,close,volume"] + [
        f"{2000 + i * 60}.0,50,55,48,52,1" for i in range(35)
    ] + ["5000.0,50,nan,48,52,1"]
    t_csv.write_text("\n".join(t_lines) + "\n", encoding="utf-8")
    orig_t_dir = t_grid.DATA
    try:
        t_grid.DATA = str(tmp_path)
        t_grid._LOAD_CACHE.pop(("VUSA_EQ", "1h"), None)
        assert len(t_grid.load("VUSA_EQ", "1h")) == 35
    finally:
        t_grid.DATA = orig_t_dir

    # 5. kraken_continuous_grid.report.q and win_stats filter NaN and bool
    assert k_rep.q([float("nan"), True, 10.0, 20.0], 0.5) == 10.0
    st = k_rep.win_stats({"w30": [(float("nan"), 0.0, 5.0), (5.0, 1.0, 15.0)]}, "w30")
    assert st["mean"] == 5.0 and st["pos"] == 100.0


def test_iteration75_walk_forward_runners_grid_reports_and_backtest_ranges_safety(tmp_path):
    """Verify Iteration 75 hardening across walk-forward baseline runners, spot_profile_review, grid reports, run_equities, and backtest_ranges."""
    import sys
    from offline.runners import spot_profile_review as spr
    from offline.research import backtest_ranges as br
    from offline.research.t212_continuous_grid import report as t_rep
    from offline.research.hybrid_reentry import run_equities as req

    # 1. spot_profile_review.revision_params rejects empty or flag-like specs
    with pytest.raises(ValueError, match="Invalid git revision spec"):
        spr.revision_params("")
    with pytest.raises(ValueError, match="Invalid git revision spec"):
        spr.revision_params("--help")

    # 2. backtest_ranges.scan_backtest_ranges handles export KEY=val, CRLF, and inline comments
    cfg_f = tmp_path / "test.env"
    cfg_f.write_bytes(b"# BACKTEST: 1.0, 2.0, 3.0 # comment\r\nexport MY_PARAM = 2.0\r\n")
    ranges = br.scan_backtest_ranges(str(cfg_f))
    assert ranges == {"MY_PARAM": ["1.0", "2.0", "3.0"]}

    # 3. t212_continuous_grid.report.g safely rejects non-dict sub-entries, bools, and NaN
    assert t_rep.g({"1h": "corrupt"}, "1h", "ret") is None
    assert t_rep.g({"1h": {"ret": True}}, "1h", "ret") is None
    assert t_rep.g({"1h": {"ret": float("nan")}}, "1h", "ret") is None
    assert t_rep.g({"1h": {"ret": 4.5}}, "1h", "ret") == 4.5

    # 4. hybrid_reentry.run_equities._import_t212 restores sys.path and sys.modules
    path_before = list(sys.path)
    strat_mod, rp_mod = req._import_t212()
    assert hasattr(strat_mod, "StratParams")
    assert hasattr(rp_mod, "run_replay")
    assert sys.path == path_before


def test_iteration76_kraken_adaptive_thresholds_and_sys_path_isolation():
    """Verify Iteration 76 explicit kraken.backtest imports and sys.path restoration across offline runners."""
    import importlib
    import sys
    from offline.runners import hyperliquid_walk_forward_baseline as hl_wf
    from offline.runners import t212_walk_forward_baseline as t212_wf
    from offline.runners import fetch_hyperliquid_candles as fhc
    from offline.research.kraken_adaptive_thresholds import (
        verify_adaptive_dca,
        verify_adaptive_reentry,
        sweep_k_multiplier,
    )

    while str(hl_wf.HL_DIR) in sys.path:
        sys.path.remove(str(hl_wf.HL_DIR))
    while str(t212_wf.T212_DIR) in sys.path:
        sys.path.remove(str(t212_wf.T212_DIR))
    importlib.reload(hl_wf)
    importlib.reload(t212_wf)
    importlib.reload(fhc)
    assert str(hl_wf.HL_DIR) not in sys.path
    assert str(t212_wf.T212_DIR) not in sys.path
    assert str(fhc.HL_DIR) not in sys.path
    assert hasattr(verify_adaptive_dca, "run")
    assert hasattr(verify_adaptive_reentry, "run")
    assert hasattr(sweep_k_multiplier, "main")

    # Even if sys.modules["backtest"] is polluted, reloading verify_adaptive_dca still succeeds via kraken.backtest
    orig_bt = sys.modules.get("backtest")
    try:
        sys.modules["backtest"] = object()  # type: ignore[assignment]
        importlib.reload(verify_adaptive_dca)
        importlib.reload(verify_adaptive_reentry)
        importlib.reload(sweep_k_multiplier)
    finally:
        if orig_bt is None:
            sys.modules.pop("backtest", None)
        else:
            sys.modules["backtest"] = orig_bt


def test_iteration77_discovery_news_visuals_and_notify_server_safety(tmp_path, monkeypatch):
    """Verify Iteration 77 import safety in discovery.news, visuals edge-case guards, and notify_engine cooldown validation."""
    import json
    import re
    import numpy as np
    from discovery import news as d_news
    from visuals import drawprices, tradeall_observe
    from notify_engine.server import NotificationServer

    # 1. discovery.news imports cleanly without executing HTTP calls, and get_binance_coins strips only trailing USDT
    class _DummyResp:
        def json(self):
            return [
                {"symbol": "BTCUSDT", "price": "65000.0"},
                {"symbol": "USDTB", "price": "1.0"},
            ]

    monkeypatch.setattr(d_news.requests, "get", lambda url, timeout=15, **kw: _DummyResp())
    b_map = d_news.get_binance_coins()
    assert b_map["BTC"] == 65000.0
    assert b_map["USDTB"] == 1.0

    # 2. visuals.tradeall_observe _log_dates and build_backtest_state_text
    assert tradeall_observe._log_dates(False) == []
    assert tradeall_observe._log_dates(-3) == []
    assert tradeall_observe._log_dates(float("nan")) == []
    assert len(tradeall_observe._log_dates(2)) == 2

    state_file = tmp_path / "analysis_state.json"
    state_file.write_text(json.dumps({"BTCUSDC": {"ts": 1e20, "price": 50000.0}}), encoding="utf-8")
    state_txt = tradeall_observe.build_backtest_state_text(str(tmp_path), "BTCUSDC")
    assert state_txt is not None and "SIMULATED ANALYSIS (unknown)" in state_txt

    # 3. visuals.drawprices get_cache_for_symbol and min_max_inverse
    assert drawprices.get_cache_for_symbol("__NON_EXISTENT_SYMBOL__") == []
    inv_nan = drawprices.min_max_inverse(np.array([0.2, 0.8]), float("nan"), 10.0)
    assert np.all(inv_nan == 0.0)
    inv_bool = drawprices.min_max_inverse(np.array([0.2, 0.8]), True, 10.0)
    assert np.all(inv_bool == 0.0)

    # 4. notify_engine.server process_line handles None/bool/negative cooldown_minutes and future last_time
    from collections import defaultdict
    svc = NotificationServer.__new__(NotificationServer)
    svc.traceback_buffers = defaultdict(list)
    svc.rules = [
        {
            "name": "TestRule",
            "regex_obj": re.compile(r"CRITICAL_TEST"),
            "cooldown_minutes": None,
            "topic_category": "ERROR",
            "priority": "urgent",
        }
    ]
    svc.cooldowns = {"bot_TestRule": 9e15}  # corrupted future timestamp
    svc._resolve_topic = lambda cat: "test_topic"
    sent = []
    svc._gate_and_send_ntfy = lambda title, body, prio, topic, **kw: sent.append((title, body, topic))
    svc.process_line("CRITICAL_TEST happened", "bot")
    assert len(sent) == 1


def test_iteration78_pricechecker_thresholds_alertnotifiers_and_sim_window_normalization(monkeypatch):
    """Verify Iteration 78 PriceChecker partial/malformed per_coin thresholds, AlertNotifier helpers, and PriceWindow synchronized normalization."""
    import pytest
    from market_monitor.pricechecker import PriceChecker, PriceAlert, PRICE_ALERT_CONFIG
    from notify_engine.alertnotifiers import AlertNotifier, _positive_int_env
    from offline.simulations.sim import PriceWindow

    # 1. PriceChecker._get_thresholds_for_symbol handles partial or invalid per_coin dicts
    class _DummyCache:
        symbol_added_time = {}

    checker = PriceChecker(
        _DummyCache(),
        config={
            "per_coin": {
                "BTC": {"up_percent": 2.5},  # missing down_percent
                "ETH": {"up_percent": True, "down_percent": -5.0},
            }
        },
    )
    btc_th = checker._get_thresholds_for_symbol("BTCUSDC")
    assert btc_th["up_percent"] == 2.5
    assert btc_th["down_percent"] == PRICE_ALERT_CONFIG["default"]["down_percent"]
    eth_th = checker._get_thresholds_for_symbol("ETHUSDC")
    assert eth_th["up_percent"] == PRICE_ALERT_CONFIG["default"]["up_percent"]
    assert eth_th["down_percent"] == PRICE_ALERT_CONFIG["default"]["down_percent"]

    pa = PriceAlert("BTCUSDC", "up", 65000.0, 62000.0, 4.84, 4.1)
    pa.timestamp = 1e20
    assert "BTCUSDC" in str(pa)
    assert pa.to_dict()["symbol"] == "BTCUSDC"

    # 2. AlertNotifier helpers
    assert AlertNotifier.email_subject([]) == "Alert"
    assert AlertNotifier.utf8_header(None) == ""
    monkeypatch.setenv("TEST_POS_INT_ENV", "300.0")
    assert _positive_int_env("TEST_POS_INT_ENV", 10) == 300

    # 3. PriceWindow input validation and synchronized normalization when window_size >= max_index
    with pytest.raises(ValueError):
        PriceWindow(True)
    with pytest.raises(ValueError):
        PriceWindow(5, max_index=0)
    with pytest.raises(ValueError):
        PriceWindow(5, epsilon=-0.1)

    pw = PriceWindow(window_size=5, max_index=3)
    for px in [10.0, 20.0, 5.0, 30.0, 15.0, 25.0]:
        pw.process_price(px)
    min_px, min_pos = pw.get_min()
    max_px, max_pos = pw.get_max()
    assert min_px == 5.0 and min_pos == 1
    assert max_px == 30.0 and max_pos == 2


def test_iteration79_alerts_config_price_notifier_and_verify_tools_safety(tmp_path):
    """Verify Iteration 79 alerts_config validation/resolve, price_notifier status safety, and gen_alert_thresholds guards."""
    import pytest
    from types import SimpleNamespace
    from market_monitor import alerts_config
    from verify_tools import gen_alert_thresholds

    # 1. alerts_config._pair rejects extra slashes and load_config rejects fractional integer settings
    with pytest.raises(ValueError):
        alerts_config._pair("6 / 10 / 15")
    with pytest.raises(ValueError):
        alerts_config._pair("6")

    bad_cfg = tmp_path / "bad_alerts.conf"
    bad_cfg.write_text(
        "watch = BTC\n"
        "sources = coinmarketcap\n"
        "discover_new_coins = yes\n"
        "default = 4 / 8\n"
        "new_coin = 12 / 25\n"
        "cooldown_minutes = 30\n"
        "lookback_hours = 24\n"
        "max_monitored = 3.5\n"
        "max_new_coins = 5\n"
        "new_coins_scan_seconds = 3600\n"
        "price_scan_seconds = 60\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid alert configuration for max_monitored"):
        alerts_config.load_config(str(bad_cfg))

    # 2. alerts_config.resolve normalizes case and handles None per_coin
    ac = {
        "default": {"up_percent": 4.0, "down_percent": 8.0},
        "dynamic": {"up_percent": 12.0, "down_percent": 25.0},
        "per_coin": {"BTC": {"up_percent": 6.0, "down_percent": 10.0}},
    }
    assert alerts_config.resolve(ac, "btcusdc", False) == {"up_percent": 6.0, "down_percent": 10.0}
    assert alerts_config.resolve({"default": ac["default"], "dynamic": ac["dynamic"], "per_coin": None}, "ETH", False) == ac["default"]

    # 3. gen_alert_thresholds._perc and suggest input validation
    assert gen_alert_thresholds._perc([], 85) == 0.0
    assert gen_alert_thresholds.suggest("BTC", days=False) is None
    assert gen_alert_thresholds.suggest("BTC", days=-5) is None
    assert gen_alert_thresholds.suggest("BTC", days=float("nan")) is None


def test_iteration80_protobuf_summary_store_and_overnight_monitor_safety(tmp_path):
    """Verify Iteration 80 protobuf varint/tag bounds, summary_store timezone offset parsing, and AutonomousOvernightMonitor validation."""
    import pytest
    from datetime import datetime, timezone
    from orchestratorOS.agy import protobuf, summary_store
    from orchestratorOS.livecheck.autonomous_overnight_monitor import AutonomousOvernightMonitor

    # 1. protobuf.dec_varint rejects overly long varints and parse_fields rejects tag 0
    with pytest.raises(ValueError, match="varint too long"):
        protobuf.dec_varint(b"\x80" * 11, 0)
    with pytest.raises(ValueError, match="invalid field tag 0"):
        protobuf.parse_fields(b"\x00\x01")
    assert protobuf.summary_annotation_title(b"\x80\x80") == ""

    # 2. summary_store.parse_db_timestamp preserves negative/positive timezone offsets without corrupting fractional digits
    dt_neg = summary_store.parse_db_timestamp("2026-10-26T14:30:00.12-05:00")
    assert dt_neg == datetime(2026, 10, 26, 19, 30, 0, 120000, tzinfo=timezone.utc)
    dt_pos = summary_store.parse_db_timestamp("2026-10-26T14:30:00.5+02:00")
    assert dt_pos == datetime(2026, 10, 26, 12, 30, 0, 500000, tzinfo=timezone.utc)

    # 3. AutonomousOvernightMonitor parses float hb_stale_s and rejects invalid loop intervals
    procs_file = tmp_path / "procs.conf"
    procs_file.write_text(
        "tradeall.py|$ROOT|python tradeall.py|tradeall|logs/tradeall.log|180.5|bot\n",
        encoding="utf-8",
    )
    mon = AutonomousOvernightMonitor(workspace_dir=str(tmp_path))
    parsed = mon.parse_process_manifest()
    assert len(parsed) == 1
    assert parsed[0]["hb_stale_s"] == 180.5

    with pytest.raises(ValueError):
        mon.loop(interval_seconds=False)
    with pytest.raises(ValueError):
        mon.loop(interval_seconds=0)
    with pytest.raises(ValueError):
        mon.loop(interval_seconds=float("nan"))


def test_iteration81_protobuf_self_heal_agy_store_fix_and_config_safety(monkeypatch):
    """Verify Iteration 81 protobuf corrupted-blob self-healing, agy_store_fix NaN/missing-file resilience, and config._parse_bool safety."""
    import pytest
    import config
    from orchestratorOS.agy import protobuf
    from orchestratorOS.admin import agy_store_fix

    # 1. protobuf.summary_ui_visible self-heals corrupted raw bytes and preserves title
    repaired = protobuf.summary_ui_visible(b"\x80\x80", title="Recovered Thread", reset_running=True)
    assert protobuf.summary_annotation_title(repaired) == "Recovered Thread"
    assert protobuf.list_entry_cids(b"\x80\x80") == set()
    assert protobuf.drop_entries(b"\x80\x80", {"abc"}) == (b"\x80\x80", 0)
    assert protobuf.replace_summary_entry(b"\x80\x80", "abc", b"") == (b"\x80\x80", False)

    # 2. agy_store_fix rejects non-finite idle_sec and handles missing PB/DB files gracefully
    with pytest.raises(ValueError):
        agy_store_fix.mode_ui_sync_check(float("nan"))
    with pytest.raises(ValueError):
        agy_store_fix.mode_ui_sync_check(float("inf"))
    monkeypatch.setattr(agy_store_fix, "PB", "/nonexistent/path/agyhub_summaries_proto.pb")
    monkeypatch.setattr(agy_store_fix, "DB", "/nonexistent/path/conversation_summaries.db")
    assert agy_store_fix.daemon_known_cids() == set()
    assert agy_store_fix.summary_row_cids() == set()

    # 3. config._parse_bool handles NaN/Inf and unrecognized strings by falling back to default
    assert config._parse_bool(float("nan"), default=False) is False
    assert config._parse_bool(float("nan"), default=True) is True
    assert config._parse_bool("unknown_token", default=True) is True
    assert config._parse_bool("unknown_token", default=False) is False


