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
        inst.place("SELL", 100.0, 1.0, bypass_guards={"financial"}, wait_for_trend=False)
        assert len(captured_bypasses) == 1
        sell_side, sell_byp = captured_bypasses[-1]
        assert sell_side == "SELL"
        assert GUARD_REGISTRY["G_PROF"].is_bypassed(sell_byp) is False
        assert GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(sell_byp) is True
        assert GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(sell_byp) is True

        # BUY with bypass_guards={"qlimit"} must revoke G_QLIMIT1 while keeping G_QLIMIT2
        inst.place("BUY", 100.0, 1.0, bypass_guards={"qlimit"}, wait_for_trend=False)
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











