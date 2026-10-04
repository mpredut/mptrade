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
