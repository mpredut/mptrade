"""Unit tests for Pillar 3: Sentiment Intelligence (Fear & Greed, Market Breadth, Contrarian Triggers, Euphoria/Panic Guards)."""
import json
import time
import pytest

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerSide
from intelligence.internal.guards.guard_decision import BrakeAction
from intelligence.sentiment.collectors.fear_greed_collector import (
    FearGreedCollector,
    FearGreedSnapshot,
)
from intelligence.sentiment.collectors.market_breadth_collector import (
    MarketBreadthCollector,
    MarketBreadthSnapshot,
)
from intelligence.sentiment.triggers.sentiment_contrarian_trigger import (
    SentimentContrarianTrigger,
)
from intelligence.sentiment.triggers.market_breadth_trigger import (
    MarketBreadthTrigger,
)
from intelligence.sentiment.guards.extreme_greed_guard import (
    ExtremeGreedGuard,
)
from intelligence.sentiment.guards.panic_washout_guard import (
    PanicWashoutGuard,
)
from intelligence.composite import CompositeMarketIntelligence


class TestFearGreedCollector:
    """Tests for FearGreedCollector parsing and caching logic."""

    def test_parse_valid_payload(self):
        payload = {
            "name": "Fear and Greed Index",
            "data": [
                {"value": "18", "value_classification": "Extreme Fear", "timestamp": "1710000000"},
                {"value": "19", "value_classification": "Extreme Fear", "timestamp": "1709913600"},
                {"value": "22", "value_classification": "Extreme Fear", "timestamp": "1709827200"},
                {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1709740800"},
                {"value": "28", "value_classification": "Fear", "timestamp": "1709654400"},
                {"value": "30", "value_classification": "Fear", "timestamp": "1709568000"},
                {"value": "32", "value_classification": "Fear", "timestamp": "1709481600"},
            ],
        }
        snapshot = FearGreedCollector.parse_payload(payload)
        assert snapshot is not None
        assert snapshot.value == 18
        assert snapshot.classification == "Extreme Fear"
        assert snapshot.is_extreme_fear is True
        assert snapshot.is_extreme_greed is False
        assert len(snapshot.historical_values) == 7
        # 18 - 32 = -14 change over 7 days
        assert snapshot.trend_7d_change == -14

    def test_parse_invalid_payload(self):
        assert FearGreedCollector.parse_payload({}) is None
        assert FearGreedCollector.parse_payload({"data": []}) is None
        assert FearGreedCollector.parse_payload({"data": "corrupt"}) is None

    def test_caching(self):
        collector = FearGreedCollector(cache_ttl_sec=300.0)
        snapshot = FearGreedSnapshot(
            value=88,
            classification="Extreme Greed",
            timestamp=time.time(),
            historical_values=(88, 85, 82),
            trend_7d_change=6,
            is_extreme_fear=False,
            is_extreme_greed=True,
        )
        collector._cached_snapshot = snapshot
        collector._last_fetch_ts = time.time()

        res = collector.fetch(force_refresh=False)
        assert res == snapshot


class TestMarketBreadthCollector:
    """Tests for MarketBreadthCollector cross-asset dispersion."""

    def test_parse_tickers_bullish_and_euphoric(self):
        tickers = [
            {"symbol": "BTCUSDT", "priceChangePercent": "6.5", "quoteVolume": "50000000"},
            {"symbol": "ETHUSDT", "priceChangePercent": "7.2", "quoteVolume": "40000000"},
            {"symbol": "SOLUSDT", "priceChangePercent": "12.0", "quoteVolume": "30000000"},
            {"symbol": "BNBUSDT", "priceChangePercent": "5.1", "quoteVolume": "20000000"},
            {"symbol": "ADAUSDT", "priceChangePercent": "8.0", "quoteVolume": "15000000"},
            {"symbol": "DOGEUSDT", "priceChangePercent": "9.5", "quoteVolume": "25000000"},
            {"symbol": "USDCUSDT", "priceChangePercent": "0.01", "quoteVolume": "999999999"}, # Should be skipped
        ]
        snapshot = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=10_000_000.0)
        assert snapshot is not None
        assert snapshot.total_count == 6
        assert snapshot.advancing_count == 6
        assert snapshot.declining_count == 0
        assert snapshot.advance_ratio == 1.0
        assert snapshot.regime == "EUPHORIC_BLOWOFF"
        assert snapshot.median_change_pct > 5.0
        assert len(snapshot.top_gainers) == 3

    def test_parse_tickers_panic_washout(self):
        tickers = [
            {"symbol": "BTCUSDT", "priceChangePercent": "-6.5", "quoteVolume": "50000000"},
            {"symbol": "ETHUSDT", "priceChangePercent": "-8.2", "quoteVolume": "40000000"},
            {"symbol": "SOLUSDT", "priceChangePercent": "-14.0", "quoteVolume": "30000000"},
            {"symbol": "BNBUSDT", "priceChangePercent": "-5.5", "quoteVolume": "20000000"},
            {"symbol": "ADAUSDT", "priceChangePercent": "-11.0", "quoteVolume": "15000000"},
            {"symbol": "DOGEUSDT", "priceChangePercent": "1.0", "quoteVolume": "25000000"},
        ]
        snapshot = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=10_000_000.0)
        assert snapshot is not None
        assert snapshot.total_count == 6
        assert snapshot.advancing_count == 1
        assert snapshot.declining_count == 5
        assert snapshot.advance_ratio < 0.20
        assert snapshot.regime == "PANIC_WASHOUT"


class TestSentimentContrarianTrigger:
    """Tests for extreme sentiment contrarian entry and exit triggers."""

    def test_extreme_fear_contrarian_buy(self):
        trigger = SentimentContrarianTrigger(extreme_fear_entry=22, extreme_greed_exit=85)
        snapshot = FearGreedSnapshot(
            value=16,
            classification="Extreme Fear",
            timestamp=time.time(),
            historical_values=(16, 17, 18),  # Stabilizing, no plunge
            trend_7d_change=-5,
            is_extreme_fear=True,
            is_extreme_greed=False,
        )

        sig_type, event = trigger.evaluate("BTCUSDT", 60000.0, snapshot)
        assert sig_type == "CONTRARIAN_BUY"
        assert event is not None
        assert event.action == TriggerAction.ENTRY
        assert event.side == TriggerSide.BUY
        assert event.strength >= 0.80
        assert "extreme_fear" in event.reason

    def test_free_falling_fear_delays_buy(self):
        trigger = SentimentContrarianTrigger(extreme_fear_entry=22)
        # Yesterday was 30, today plummeted to 18 (drop of 12 points in 1 day)
        snapshot = FearGreedSnapshot(
            value=18,
            classification="Extreme Fear",
            timestamp=time.time(),
            historical_values=(18, 30, 35),
            trend_7d_change=-17,
            is_extreme_fear=True,
            is_extreme_greed=False,
        )
        sig_type, event = trigger.evaluate("BTCUSDT", 60000.0, snapshot)
        assert sig_type is None
        assert event is None

    def test_extreme_greed_contrarian_sell(self):
        trigger = SentimentContrarianTrigger(extreme_fear_entry=22, extreme_greed_exit=85)
        snapshot = FearGreedSnapshot(
            value=88,
            classification="Extreme Greed",
            timestamp=time.time(),
            historical_values=(88, 86, 84),
            trend_7d_change=10,
            is_extreme_fear=False,
            is_extreme_greed=True,
        )

        sig_type, event = trigger.evaluate("BTCUSDT", 75000.0, snapshot)
        assert sig_type == "CONTRARIAN_SELL"
        assert event is not None
        assert event.action == TriggerAction.EXIT
        assert event.side == TriggerSide.SELL
        assert event.strength == 0.88

    def test_neutral_sentiment_no_trigger(self):
        trigger = SentimentContrarianTrigger()
        snapshot = FearGreedSnapshot(
            value=52,
            classification="Neutral",
            timestamp=time.time(),
            historical_values=(52, 50, 48),
            trend_7d_change=4,
            is_extreme_fear=False,
            is_extreme_greed=False,
        )
        sig_type, event = trigger.evaluate("BTCUSDT", 65000.0, snapshot)
        assert sig_type is None
        assert event is None


class TestMarketBreadthTrigger:
    """Tests for MarketBreadthTrigger washout and blowoff signals."""

    def test_washout_bounce_entry(self):
        trigger = MarketBreadthTrigger()
        snapshot = MarketBreadthSnapshot(
            advance_ratio=0.10,
            advancing_count=2,
            declining_count=18,
            total_count=20,
            median_change_pct=-6.5,
            mean_change_pct=-7.0,
            dispersion_std=2.5,
            regime="PANIC_WASHOUT",
            top_gainers=(("DOGEUSDT", 1.0),),
            top_losers=(("SOLUSDT", -15.0),),
            ts=time.time(),
        )

        sig_type, event = trigger.evaluate("BTCUSDT", 62000.0, snapshot, asset_24h_change_pct=-3.0)
        assert sig_type == "BREADTH_WASHOUT_BUY"
        assert event is not None
        assert event.action == TriggerAction.ENTRY
        assert event.side == TriggerSide.BUY
        assert event.metadata["relative_strength"] is True

    def test_blowoff_exhaustion_exit(self):
        trigger = MarketBreadthTrigger()
        snapshot = MarketBreadthSnapshot(
            advance_ratio=0.95,
            advancing_count=19,
            declining_count=1,
            total_count=20,
            median_change_pct=8.5,
            mean_change_pct=9.0,
            dispersion_std=3.0,
            regime="EUPHORIC_BLOWOFF",
            top_gainers=(("SOLUSDT", 18.0),),
            top_losers=(("DOGEUSDT", -1.0),),
            ts=time.time(),
        )

        sig_type, event = trigger.evaluate("BTCUSDT", 74000.0, snapshot)
        assert sig_type == "BREADTH_BLOWOFF_SELL"
        assert event is not None
        assert event.action == TriggerAction.EXIT
        assert event.side == TriggerSide.SELL


class TestExtremeGreedGuard:
    """Tests for ExtremeGreedGuard anti-FOMO brake."""

    def test_extreme_greed_hard_veto(self):
        guard = ExtremeGreedGuard(downscale_threshold=80, hard_veto_threshold=90)
        snapshot = FearGreedSnapshot(
            value=92,
            classification="Extreme Greed",
            timestamp=time.time(),
            historical_values=(92, 90),
            trend_7d_change=8,
            is_extreme_fear=False,
            is_extreme_greed=True,
        )

        decision = guard.check("BTCUSDT", "BUY", snapshot)
        assert decision.allowed is False
        assert decision.brake_action == BrakeAction.HARD_VETO
        assert "veto_extreme_market_euphoria" in decision.reason

    def test_greed_downscale(self):
        guard = ExtremeGreedGuard(downscale_threshold=80, hard_veto_threshold=90, downscale_factor=0.35)
        snapshot = FearGreedSnapshot(
            value=82,
            classification="Extreme Greed",
            timestamp=time.time(),
            historical_values=(82, 80),
            trend_7d_change=4,
            is_extreme_fear=False,
            is_extreme_greed=True,
        )

        decision = guard.check("BTCUSDT", "BUY", snapshot)
        assert decision.allowed is True
        assert decision.brake_action == BrakeAction.DOWNSCALE_QTY
        assert decision.suggested_scale == 0.35

    def test_allows_sell_in_greed(self):
        guard = ExtremeGreedGuard()
        snapshot = FearGreedSnapshot(
            value=95,
            classification="Extreme Greed",
            timestamp=time.time(),
            historical_values=(95,),
            trend_7d_change=5,
            is_extreme_fear=False,
            is_extreme_greed=True,
        )

        decision = guard.check("BTCUSDT", "SELL", snapshot)
        assert decision.allowed is True
        assert decision.brake_action == BrakeAction.NONE


class TestPanicWashoutGuard:
    """Tests for PanicWashoutGuard falling knife protection."""

    def test_panicking_washout_defers_dumping_asset(self):
        guard = PanicWashoutGuard(panic_advance_threshold=0.10)
        breadth = MarketBreadthSnapshot(
            advance_ratio=0.08,
            advancing_count=2,
            declining_count=22,
            total_count=24,
            median_change_pct=-7.0,
            mean_change_pct=-7.5,
            dispersion_std=2.0,
            regime="PANIC_WASHOUT",
            top_gainers=(),
            top_losers=(),
            ts=time.time(),
        )

        # Asset down -9% during market panic -> defer BUY
        decision = guard.check("SOLUSDT", "BUY", breadth_snapshot=breadth, asset_24h_change_pct=-9.0)
        assert decision.allowed is False
        assert decision.brake_action == BrakeAction.DEFER_WAIT

    def test_allows_sell_in_panic(self):
        guard = PanicWashoutGuard()
        breadth = MarketBreadthSnapshot(
            advance_ratio=0.05,
            advancing_count=1,
            declining_count=19,
            total_count=20,
            median_change_pct=-8.0,
            mean_change_pct=-8.5,
            dispersion_std=2.0,
            regime="PANIC_WASHOUT",
            top_gainers=(),
            top_losers=(),
            ts=time.time(),
        )
        decision = guard.check("BTCUSDT", "SELL", breadth_snapshot=breadth)
        assert decision.allowed is True
        assert decision.brake_action == BrakeAction.NONE


class TestCompositeWithSentiment:
    """Integration tests verifying CompositeMarketIntelligence coordinates Pillar 3."""

    def test_contrarian_fear_buy_evaluated(self):
        composite = CompositeMarketIntelligence()
        fear_snapshot = FearGreedSnapshot(
            value=18,
            classification="Extreme Fear",
            timestamp=time.time(),
            historical_values=(18, 19, 20),
            trend_7d_change=-5,
            is_extreme_fear=True,
            is_extreme_greed=False,
        )

        evaluation = composite.evaluate(
            symbol="BTCUSDT",
            price=61000.0,
            fear_greed_snapshot=fear_snapshot,
        )

        assert evaluation.active_trigger is not None
        assert evaluation.active_trigger.side == TriggerSide.BUY
        assert evaluation.active_trigger.source == "sentiment_contrarian"
        assert evaluation.can_execute is True
        assert evaluation.guard_decision.allowed is True
        assert evaluation.metrics["fear_greed"] == 18

    def test_composite_greed_guard_veto(self):
        composite = CompositeMarketIntelligence(greed_hard_veto_threshold=90)
        greed_snapshot = FearGreedSnapshot(
            value=94,
            classification="Extreme Greed",
            timestamp=time.time(),
            historical_values=(94, 91),
            trend_7d_change=8,
            is_extreme_fear=False,
            is_extreme_greed=True,
        )

        # Gradient signals BUY, but ExtremeGreedGuard vetos it
        evaluation = composite.evaluate(
            symbol="BTCUSDT",
            price=68000.0,
            gradient=0.05,
            epsilon=0.01,
            fear_greed_snapshot=greed_snapshot,
        )

        assert evaluation.active_trigger is not None
        assert evaluation.active_trigger.side == TriggerSide.BUY
        assert evaluation.guard_decision.allowed is False
        assert evaluation.guard_decision.brake_action == BrakeAction.HARD_VETO
        assert evaluation.can_execute is False


class TestGeminiClient:
    """Tests for GeminiClient wrapper with mocked runner and JSON decoding."""

    def test_query_json_clean(self):
        from intelligence.sentiment.gemini_client import GeminiClient

        def mock_runner(prompt, model, timeout):
            return '{"decision": "APPROVED", "suggested_scale": 1.0, "reason": "All metrics green"}'

        client = GeminiClient(custom_runner=mock_runner)
        res = client.query_json("Test prompt")
        assert res is not None
        assert res["decision"] == "APPROVED"
        assert res["suggested_scale"] == 1.0

    def test_query_json_markdown_wrapped(self):
        from intelligence.sentiment.gemini_client import GeminiClient

        def mock_runner(prompt, model, timeout):
            return "```json\n{\n  \"decision\": \"DOWNSCALE\",\n  \"suggested_scale\": 0.5,\n  \"reason\": \"Overleveraged\"\n}\n```"

        client = GeminiClient(custom_runner=mock_runner)
        res = client.query_json("Test prompt")
        assert res is not None
        assert res["decision"] == "DOWNSCALE"
        assert res["suggested_scale"] == 0.5


class TestGeminiMarketAdvisor:
    """Tests for periodic GeminiMarketAdvisor macro evaluations."""

    def test_advisor_review_and_caching(self, tmp_path):
        from intelligence.sentiment.gemini_client import GeminiClient
        from intelligence.sentiment.collectors.gemini_advisor import GeminiMarketAdvisor

        def mock_runner(prompt, model, timeout):
            return json.dumps({
                "market_bias": "BULLISH",
                "risk_level": "MODERATE",
                "confidence": 0.85,
                "summary": "Macro indicators suggest continued institutional accumulation.",
                "key_risks": ["Upcoming CPI release", "Derivatives OI peak"],
                "recommended_action": "ACCUMULATE",
            })

        client = GeminiClient(custom_runner=mock_runner)
        cache_file = str(tmp_path / "gemini_advisor.json")
        advisor = GeminiMarketAdvisor(gemini_client=client, cache_file=cache_file, cache_ttl_sec=300.0)

        assessment = advisor.review(force_refresh=True)
        assert assessment is not None
        assert assessment.market_bias == "BULLISH"
        assert assessment.recommended_action == "ACCUMULATE"
        assert assessment.confidence == 0.85

        # Check caching without re-evaluating
        cached = advisor.review(force_refresh=False)
        assert cached == assessment


class TestGeminiHighStakeGuard:
    """Tests for GeminiHighStakeGuard order vetting for >= 1000 EUR."""

    def test_sub_threshold_bypasses_llm(self):
        from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
        # Client runner that raises if called
        def failing_runner(prompt, model, timeout):
            raise AssertionError("Should not be called for orders < 1000 EUR")

        from intelligence.sentiment.gemini_client import GeminiClient
        client = GeminiClient(custom_runner=failing_runner)
        guard = GeminiHighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        # 500 EUR order -> instant pass
        dec = guard.check("BTCUSDT", "BUY", price=50000.0, qty=0.01, notional_eur=500.0)
        assert dec.allowed is True
        assert "below_high_stake_threshold" in dec.reason

    def test_high_stake_approved(self):
        from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def approve_runner(prompt, model, timeout):
            assert "1,500.00 EUR" in prompt
            return json.dumps({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Healthy trend and low funding"})

        client = GeminiClient(custom_runner=approve_runner)
        guard = GeminiHighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.025, notional_eur=1500.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.NONE
        assert "Gemini approved" in dec.reason

    def test_high_stake_vetoed(self):
        from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def veto_runner(prompt, model, timeout):
            return json.dumps({"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Massive whale sell wall and euphoric top"})

        client = GeminiClient(custom_runner=veto_runner)
        guard = GeminiHighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=70000.0, qty=0.03, notional_eur=2100.0)
        assert dec.allowed is False
        assert dec.brake_action == BrakeAction.HARD_VETO
        assert "Gemini vetoed" in dec.reason

    def test_high_stake_downscaled(self):
        from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def downscale_runner(prompt, model, timeout):
            return json.dumps({"decision": "DOWNSCALE", "suggested_scale": 0.35, "reason": "Caution: funding is elevated"})

        client = GeminiClient(custom_runner=downscale_runner)
        guard = GeminiHighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.02, notional_eur=1200.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
        assert dec.suggested_scale == 0.35


class TestOrderGuardWithGemini:
    """Integration test for order_guard.check_intelligence_guards with Gemini mode."""

    def test_check_intelligence_guards_with_gemini_shadow(self, monkeypatch):
        import order_guard

        margins = {
            "intelligence_guards_mode": "shadow",
            "gemini_guard_mode": "shadow",
            "gemini_min_notional_eur": 1000.0,
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def mock_veto_runner(prompt, model, timeout):
            return json.dumps({"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Risky market"})

        mock_guard = GeminiHighStakeGuard(
            gemini_client=GeminiClient(custom_runner=mock_veto_runner),
            min_notional_eur=1000.0,
        )

        with monkeypatch.context() as m:
            m.setattr("intelligence.sentiment.guards.gemini_high_stake_guard.GeminiHighStakeGuard", lambda **kw: mock_guard)
            # In shadow mode, order is allowed despite LLM rejection (only logged)
            allowed, reason, scale = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 65000.0, notional_eur=1500.0
            )
            assert allowed is True

    def test_check_intelligence_guards_memoized_on_regime_context(self, monkeypatch):
        import order_guard
        from market_regime import MarketRegimeContext, MarketRegimeDecision

        mock_decision = MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="directional_signal", source="mock",
        )
        ctx = MarketRegimeContext.from_decision(mock_decision)

        eval_count = 0
        original_eval = order_guard._evaluate_intelligence_guards_raw

        def counting_eval(*args, **kwargs):
            nonlocal eval_count
            eval_count += 1
            return original_eval(*args, **kwargs)

        monkeypatch.setattr(order_guard, "_evaluate_intelligence_guards_raw", counting_eval)

        # Call multiple times with the same regime_context (simulating multi-step MARKET placement)
        res1 = order_guard.check_intelligence_guards(None, "BTCUSDC", "BUY", 65000.0, regime_context=ctx, notional_eur=100.0)
        res2 = order_guard.check_intelligence_guards(None, "BTCUSDC", "BUY", 65100.0, regime_context=ctx, notional_eur=100.0)
        res3 = order_guard.check_intelligence_guards(None, "BTCUSDC", "BUY", 65200.0, regime_context=ctx, notional_eur=100.0)

        assert eval_count == 1
        assert res1 == res2 == res3

    def test_profit_guard_passes_qty_and_computes_notional_for_gemini(self, monkeypatch):
        import order_guard

        margins = {
            "intelligence_guards_mode": "enforce",
            "gemini_guard_mode": "enforce",
            "gemini_min_notional_eur": 1000.0,
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        called_notionals = []

        def tracking_runner(prompt, model, timeout):
            return json.dumps({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Good trade"})

        class MockGuard(GeminiHighStakeGuard):
            def check(self, symbol, side, price, qty, notional_eur=None, **kw):
                called_notionals.append(notional_eur)
                return super().check(symbol, side, price, qty, notional_eur=notional_eur)

        mock_guard = MockGuard(
            gemini_client=GeminiClient(custom_runner=tracking_runner),
            min_notional_eur=1000.0,
        )
        monkeypatch.setattr("intelligence.sentiment.guards.gemini_high_stake_guard.GeminiHighStakeGuard", lambda **kw: mock_guard)

        # Provider mockup
        class _P:
            name = "binance"
            def last_opposite_fill(self, *a, **k): return 60000.0

        # BUY 0.05 BTC @ 65,000 = 3,250 EUR > 1000 EUR min
        allowed = order_guard.profit_guard(_P(), "BTCUSDC", "BUY", 65000.0, 1.15, qty=0.05)
        assert allowed is True
        assert len(called_notionals) == 1
        assert called_notionals[0] == 65000.0 * 0.05  # 3,250

    def test_symbol_regime_fallback_when_allow_fallback_true(self, monkeypatch):
        import order_guard

        snapshot_called = False
        def mock_snapshot(symbol, now=None):
            nonlocal snapshot_called
            snapshot_called = True
            return {"gradient_recent": 0.8, "epsilon": 0.1, "ts": time.time()}

        class MockShortTrendManager:
            def fresh_snapshot(self, symbol, now=None):
                return mock_snapshot(symbol, now)

        monkeypatch.setattr("cacheManager.get_short_trend_manager", lambda: MockShortTrendManager())

        # Calling symbol_regime directly without snapshot_resolver falls back gracefully
        decision = order_guard.symbol_regime("BTCUSDT", allow_fallback=True)
        assert snapshot_called is True
        assert decision.regime == "bull"

    def test_macro_shadow_guard_notification_title_and_cooldown(self, monkeypatch):
        """Verify macro shadow guard sends clear VETO titles and deduplicates consecutive calls."""
        import order_guard
        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment

        margins = {
            "intelligence_guards_mode": "shadow",
            "geopolitical_guard_mode": "shadow",
            "shadow_notify": 1.0,
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)
        order_guard._SHADOW_NOTIFY_COOLDOWN.clear()

        shock_geo = GeopoliticalThreatAssessment(
            threat_level="CRITICAL_SHOCK",
            risk_score=0.95,
            summary="Kinetic strikes on energy hubs",
            recommended_brake="HALT_ALL",
            headlines_analyzed=10,
            ts=time.time(),
        )

        class MockAnalyzer:
            _cached_assessment = shock_geo

        monkeypatch.setattr("intelligence.macro.geopolitical_analyzer.GeopoliticalThreatAnalyzer", lambda: MockAnalyzer())

        dispatched_alerts = []
        monkeypatch.setattr("notify_engine.alertnotifiers.notify", lambda **kw: dispatched_alerts.append(kw))

        # First evaluation: must notify with clear "Would Block" title
        allowed, reason, scale = order_guard.check_intelligence_guards(None, "TAOUSDC", "BUY", 250.0)
        assert allowed is True
        assert len(dispatched_alerts) == 1
        alert = dispatched_alerts[0]
        assert alert["title"] == "🛡 [MACRO SHADOW VETO] Would Block BUY TAOUSDC"
        assert alert["symbol"] == "TAOUSDC"
        assert "veto_critical_geopolitical_shock" in alert["body"]

        # Immediate second evaluation: must be suppressed by in-memory cooldown
        order_guard.check_intelligence_guards(None, "TAOUSDC", "BUY", 250.0)
        assert len(dispatched_alerts) == 1, "Immediate repeat should be throttled by in-memory cooldown"


