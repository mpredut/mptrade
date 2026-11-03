"""Unit tests for Pillar 3: Sentiment Intelligence (Fear & Greed, Market Breadth, Contrarian Triggers, Euphoria/Panic Guards)."""
import json
import time

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


class TestSentimentAdvisor:
    """Tests for periodic SentimentAdvisor macro evaluations."""

    def test_advisor_review_and_caching(self, tmp_path):
        from intelligence.sentiment.gemini_client import GeminiClient
        from intelligence.sentiment.sentiment_advisor import SentimentAdvisor

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
        cache_file = str(tmp_path / "sentiment_advisor.json")
        advisor = SentimentAdvisor(gemini_client=client, cache_file=cache_file, cache_ttl_sec=300.0)

        assessment = advisor.review(force_refresh=True)
        assert assessment is not None
        assert assessment.market_bias == "BULLISH"
        assert assessment.recommended_action == "ACCUMULATE"
        assert assessment.confidence == 0.85

        # Check caching without re-evaluating
        cached = advisor.review(force_refresh=False)
        assert cached == assessment


class TestHighStakeGuard:
    """Tests for HighStakeGuard order vetting for >= 1000 EUR."""

    def test_sub_threshold_bypasses_llm(self):
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        # Client runner that raises if called
        def failing_runner(prompt, model, timeout):
            raise AssertionError("Should not be called for orders < 1000 EUR")

        from intelligence.sentiment.gemini_client import GeminiClient
        client = GeminiClient(custom_runner=failing_runner)
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        # 500 EUR order -> instant pass
        dec = guard.check("BTCUSDT", "BUY", price=50000.0, qty=0.01, notional_eur=500.0)
        assert dec.allowed is True
        assert "below_high_stake_threshold" in dec.reason

    def test_high_stake_approved(self):
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def approve_runner(prompt, model, timeout):
            assert "1,500.00 EUR" in prompt
            return json.dumps({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Healthy trend and low funding"})

        client = GeminiClient(custom_runner=approve_runner)
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.025, notional_eur=1500.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.NONE
        assert "Gemini approved" in dec.reason

    def test_high_stake_vetoed(self):
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def veto_runner(prompt, model, timeout):
            return json.dumps({"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Massive whale sell wall and euphoric top"})

        client = GeminiClient(custom_runner=veto_runner)
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=70000.0, qty=0.03, notional_eur=2100.0)
        assert dec.allowed is False
        assert dec.brake_action == BrakeAction.HARD_VETO
        assert "Gemini vetoed" in dec.reason

    def test_high_stake_downscaled(self):
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def downscale_runner(prompt, model, timeout):
            return json.dumps({"decision": "DOWNSCALE", "suggested_scale": 0.35, "reason": "Caution: funding is elevated"})

        client = GeminiClient(custom_runner=downscale_runner)
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.02, notional_eur=1200.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
        assert dec.suggested_scale == 0.35

    def test_high_stake_downscaled_percentage_scale(self):
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def downscale_pct_runner(prompt, model, timeout):
            return json.dumps({"decision": "DOWNSCALE", "suggested_scale": 25, "reason": "Scale to 25%"})

        client = GeminiClient(custom_runner=downscale_pct_runner)
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.02, notional_eur=1200.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
        assert dec.suggested_scale == 0.25


class TestOrderGuardWithGemini:
    """Integration test for order_guard.check_intelligence_guards with Gemini mode."""

    def test_check_intelligence_guards_with_gemini_shadow(self, monkeypatch):
        import order_guard

        margins = {
            **order_guard._load_margins(),
            "intelligence_guards_mode": "shadow",
            "llm_guard_mode": "shadow",
            "whale_guard_mode": "off",
            "orderbook_wall_guard_mode": "off",
            "funding_guard_mode": "off",
            "geopolitical_guard_mode": "off",
            "weibull_exhaustion_policy": "off",
            "high_stake_min_notional_usd": 1000.0,
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        def mock_veto_runner(prompt, model, timeout):
            return json.dumps({"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Risky market"})

        mock_guard = HighStakeGuard(
            gemini_client=GeminiClient(custom_runner=mock_veto_runner),
            min_notional_eur=1000.0,
        )

        with monkeypatch.context() as m:
            m.setattr("intelligence.sentiment.guards.high_stake_guard.HighStakeGuard", lambda **kw: mock_guard)
            # In shadow mode, order is allowed despite LLM rejection (only logged)
            allowed, reason, scale = order_guard.check_intelligence_guards(
                None, "BTCUSDT", "BUY", 65000.0, notional_eur=1500.0
            )
            assert allowed is True

    def test_check_intelligence_guards_memoized_on_regime_context(self, monkeypatch):
        import order_guard
        from market_regime import MarketRegimeContext, MarketRegimeDecision

        margins = {
            **order_guard._load_margins(),
            "whale_guard_mode": "off",
            "orderbook_wall_guard_mode": "off",
            "funding_guard_mode": "off",
            "geopolitical_guard_mode": "off",
            "weibull_exhaustion_policy": "off",
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        mock_decision = MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
            fresh=True, reason="directional_signal", source="mock",
        )
        ctx = MarketRegimeContext.from_decision(
            mock_decision, symbol="BTCUSDC", provider="binance", evaluated_at=time.time()
        )

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
            **order_guard._load_margins(),
            "intelligence_guards_mode": "enforce",
            "llm_guard_mode": "enforce",
            "whale_guard_mode": "off",
            "orderbook_wall_guard_mode": "off",
            "funding_guard_mode": "off",
            "geopolitical_guard_mode": "off",
            "weibull_exhaustion_policy": "off",
            "high_stake_min_notional_usd": 1000.0,
            "default": 1.15,
            "default_buy_reference": "strict",
            "binance_buy_reference": "strict",
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)

        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        called_notionals = []

        def tracking_runner(prompt, model, timeout):
            return json.dumps({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Good trade"})

        class MockGuard(HighStakeGuard):
            def check(self, symbol, side, price, qty, notional_eur=None, **kw):
                called_notionals.append(notional_eur)
                return super().check(symbol, side, price, qty, notional_eur=notional_eur)

        mock_guard = MockGuard(
            gemini_client=GeminiClient(custom_runner=tracking_runner),
            min_notional_eur=1000.0,
        )
        monkeypatch.setattr("intelligence.sentiment.guards.high_stake_guard.HighStakeGuard", lambda **kw: mock_guard)

        # Provider mockup
        class _P:
            name = "binance"
            def last_opposite_fill(self, *a, **k): return 70000.0

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
            def get_fresh_snapshot(self, symbol, now=None):
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
            **order_guard._load_margins(),
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

        # First evaluation: must notify with clear [auto_macro_geopolitical] block title
        allowed, reason, scale = order_guard.check_intelligence_guards(None, "TAOUSDC", "BUY", 250.0)
        assert allowed is True
        assert len(dispatched_alerts) == 1
        alert = dispatched_alerts[0]
        assert alert["title"] == "🛑 [auto_macro_geopolitical] BLOCK BUY TAOUSDC"
        assert alert["symbol"] == "TAOUSDC"
        assert "CRITICAL_SHOCK" in alert["body"]

        # Immediate second evaluation: must be suppressed by in-memory cooldown
        order_guard.check_intelligence_guards(None, "TAOUSDC", "BUY", 250.0)
        assert len(dispatched_alerts) == 1, "Immediate repeat should be throttled by in-memory cooldown"


class TestEnrichedPretradeTelemetry:
    def test_collect_pretrade_telemetry(self, tmp_path, monkeypatch):
        from intelligence.sentiment.guards.high_stake_guard import collect_pretrade_telemetry
        from market_regime import MarketRegimeContext, MarketRegimeDecision

        # Create dummy cache files
        cachedb = tmp_path / "cachedb"
        cachedb.mkdir(exist_ok=True)
        (cachedb / "sentiment_eval.json").write_text(json.dumps({
            "market_bias": "CAUTION",
            "risk_level": "HIGH",
            "recommended_action": "TRIM_PROFITS",
        }))
        (cachedb / "fear_greed_collect.json").write_text(json.dumps({
            "value": 72,
            "classification": "Greed",
        }))
        (cachedb / "geopolitical_eval.json").write_text(json.dumps({
            "threat_level": "NORMAL",
        }))

        regime_ctx = MarketRegimeContext(
            decision=MarketRegimeDecision(
                regime="bull",
                gradient=0.0015,
                epsilon=0.0003,
                strength=5.0,
                fresh=True,
                reason="trend positive",
            ),
            resolved_trend="bull",
        )

        telemetry = collect_pretrade_telemetry("BTCUSDC", regime_context=regime_ctx, dur_sec=7200.0, cachedb_dir=str(cachedb))

        assert "Macro Sentiment Bias" in telemetry
        assert "CAUTION" in telemetry["Macro Sentiment Bias"]
        assert "Fear & Greed Index" in telemetry
        assert "72 (Greed)" in telemetry["Fear & Greed Index"]
        assert "Geopolitical Threat Level" in telemetry
        assert telemetry["Geopolitical Threat Level"] == "NORMAL"
        assert "Technical Trend Regime" in telemetry
        assert "bull" in telemetry["Technical Trend Regime"]
        assert "Trend Duration" in telemetry
        assert "2.0 hours" in telemetry["Trend Duration"]

    def test_high_stake_prompt_includes_multi_pillar_telemetry(self, monkeypatch):
        from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
        from intelligence.sentiment.gemini_client import GeminiClient

        captured_prompt = []
        def capturing_runner(prompt, model, timeout):
            captured_prompt.append(prompt)
            return json.dumps({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Telemetry looks solid"})

        client = GeminiClient(custom_runner=capturing_runner)
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        explicit_telemetry = {
            "Orderbook Depth Imbalance": "0.85 (85% bids vs 15% asks)",
            "Perpetual Funding Rate": "+0.0001% (8h)",
            "Whale Positioning": "65.0% Top Traders Long",
            "Technical Trend Regime": "bull (gradient=+0.0012, epsilon=0.0004)",
        }

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.025, notional_eur=1500.0, telemetry=explicit_telemetry)
        assert dec.allowed is True
        assert len(captured_prompt) == 1
        prompt = captured_prompt[0]

        assert "Orderbook Depth Imbalance: 0.85 (85% bids vs 15% asks)" in prompt
        assert "Perpetual Funding Rate: +0.0001% (8h)" in prompt
        assert "Whale Positioning: 65.0% Top Traders Long" in prompt
        assert "Technical Trend Regime: bull (gradient=+0.0012, epsilon=0.0004)" in prompt
        assert "€1,500.00 EUR" in prompt

    def test_gemini_client_fallback_chain_and_non_gemini_rest(self):
        from intelligence.sentiment.gemini_client import GeminiClient
        import subprocess

        # 1. Test fallback models initialization and inclusion of gpt-oss
        client = GeminiClient(
            model="primary-failing-model",
            fallback_models=["gemini-3.8-flash-low", "gpt-oss-120b-medium"],
            api_key=None,
        )
        assert client.fallback_models == ("gemini-3.8-flash-low", "gpt-oss-120b-medium")

        # 2. Test REST fallback cleans non-Gemini model names safely
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}'
        mock_resp.__enter__.return_value = mock_resp

        mock_open = MagicMock(return_value=mock_resp)

        import urllib.request
        orig_urlopen = urllib.request.urlopen
        try:
            urllib.request.urlopen = mock_open
            rest_client = GeminiClient(
                model="gpt-oss-120b-medium",
                api_key="test-api-key",
                cli_path="/nonexistent/path/to/cli",
            )
            # When CLI fails, REST fallback uses canonical gemini endpoint
            res = rest_client.query_text("test prompt")
            assert res == "ok"
            assert mock_open.call_count == 1
            call_url = mock_open.call_args[0][0].full_url
            assert "gemini-2.5-flash" in call_url
        finally:
            urllib.request.urlopen = orig_urlopen

        # 3. Test CLI fallback sequence tries secondary model when primary fails
        attempted_models = []
        def mock_subproc_run(cmd, **kwargs):
            m_idx = cmd.index("--model") + 1
            used_m = cmd[m_idx]
            attempted_models.append(used_m)
            if used_m == "failing-model":
                return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="Quota exceeded")
            import json
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps({"response": f"success from {used_m}"}))

        mock_run = MagicMock(side_effect=mock_subproc_run)
        orig_run = subprocess.run
        try:
            subprocess.run = mock_run
            cli_client = GeminiClient(
                model="failing-model",
                fallback_models=["gemini-3.8-flash-low", "gpt-oss-120b-medium"],
                cli_path=__file__,
                api_key=None,
            )
            # First test: primary fails, secondary succeeds
            res = cli_client.query_text("sample prompt")
            assert res == "success from gemini-3.8-flash-low"
            assert attempted_models == ["failing-model", "gemini-3.8-flash-low"]

            # Second test: both fail, third (gpt-oss-120b-medium) succeeds
            attempted_models.clear()
            def mock_subproc_run_gpt(cmd, **kwargs):
                m_idx = cmd.index("--model") + 1
                used_m = cmd[m_idx]
                attempted_models.append(used_m)
                if used_m in ("failing-model", "gemini-3.8-flash-low"):
                    return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="Rate limit")
                import json
                return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps({"response": f"success from {used_m}"}))

            subprocess.run = MagicMock(side_effect=mock_subproc_run_gpt)
            res2 = cli_client.query_text("sample prompt 2")
            assert res2 == "success from gpt-oss-120b-medium"
            assert attempted_models == ["failing-model", "gemini-3.8-flash-low", "gpt-oss-120b-medium"]

            # 4. Topic session with credit/quota exhaustion skips redundant one-off retry on the same model
            from unittest.mock import patch
            from orchestratorOS.agy.topics import TopicSession
            import intelligence.sentiment.gemini_client as gc_mod

            cli_invocations = []
            def mock_subproc_credit(cmd, **kwargs):
                m_idx = cmd.index("--model") + 1
                used_m = cmd[m_idx]
                has_conv = "--conversation" in cmd
                cli_invocations.append((used_m, has_conv))
                if used_m in ("failing-model", "gemini-3.8-flash-low"):
                    return subprocess.CompletedProcess(
                        cmd,
                        returncode=3,
                        stdout="",
                        stderr='error: Your AI credits balance is too low to continue.\nAGY_ERROR: {"status":"RESOURCE_EXHAUSTED"}',
                    )
                import json
                return subprocess.CompletedProcess(
                    cmd,
                    returncode=0,
                    stdout=json.dumps({"response": f"success from {used_m}", "conversation_id": "cid-topic-1"}),
                )

            subprocess.run = MagicMock(side_effect=mock_subproc_credit)
            with patch.object(gc_mod, "prepare_topic_session", return_value=TopicSession("cid-topic-1", "", False)), \
                 patch.object(gc_mod, "record_topic_turn"):
                res3 = cli_client.query_text("sample prompt 3", topic_key="auto_cryptotrade_executor")
            assert res3 == "success from gpt-oss-120b-medium"
            assert cli_invocations == [
                ("failing-model", True),
                ("gemini-3.8-flash-low", True),
                ("gpt-oss-120b-medium", True),
            ]
        finally:
            subprocess.run = orig_run


def test_order_actor_resolution_and_recent_fills_tagging():
    """Verify AI vs BOT actor resolution on orders, open orders, and recent fills extraction."""
    from providers.base import (
        extract_client_order_id,
        format_actor_tag,
        normalize_open_order,
        resolve_order_actor,
        resolve_order_origin,
    )
    from intelligence.sentiment.guards.high_stake_guard import extract_symbol_recent_fills

    # 1. extract_client_order_id tests across venue shapes and edge cases
    assert extract_client_order_id({"clientOrderId": "AI_12345"}) == "AI_12345"
    assert extract_client_order_id({"client_order_id": "RT_order_1"}) == "RT_order_1"
    assert extract_client_order_id({"cloid": "0x1234567890abcdef"}) == "0x1234567890abcdef"
    assert extract_client_order_id({"cl_ord_id": "kraken_cloid_1"}) == "kraken_cloid_1"
    assert extract_client_order_id({"userref": 998877}) == "998877"
    assert extract_client_order_id({"userref": "998877.0"}) == "998877"
    assert extract_client_order_id({"clientOrderId": "nan", "cloid": "valid_id"}) == "valid_id"
    assert extract_client_order_id({"clientOrderId": "0", "cloid": "0.0"}) is None
    assert extract_client_order_id({"clientOrderId": False}) is None
    assert extract_client_order_id(None) is None
    assert extract_client_order_id({}) is None

    # 2. resolve_order_actor tests
    assert resolve_order_actor({"actor": "AI"}) == "AI"
    assert resolve_order_actor({"actor": "bot"}) == "BOT"
    assert resolve_order_actor({"clientOrderId": "AI_12345"}) == "AI"
    # SD_<32hex> is the generic intent cid of deterministic bots (Binance trailing_stop / spot_engine), never AI
    assert resolve_order_actor({"cloid": "SD_a1b2c3d4e5f60718293a4b5c6d7e8f90"}) == "BOT"
    assert resolve_order_origin({"clientOrderId": "SD_a1b2c3d4e5f60718293a4b5c6d7e8f90"})[:2] == ("BOT", "trailing")
    # monitororder replacements of manual orders and venue-native manual orders keep manual attribution
    assert resolve_order_origin({"clientOrderId": "MO_MN_OR_MainThread_1A2B"})[:2] == ("BOT", "manual")
    assert resolve_order_origin({"clientOrderId": "web_8f2c1d9e"})[:2] == ("BOT", "manual")
    assert resolve_order_actor({"cloid": "sd_manual_1"}) == "BOT"
    assert resolve_order_actor({"userref": "AI_KRAKEN"}) == "AI"
    assert resolve_order_actor({"orderId": 999}, known_ai_order_ids={"999"}) == "AI"
    assert resolve_order_actor({"orderId": 111}, known_ai_order_ids={"999"}) == "BOT"
    assert resolve_order_actor({}) == "BOT"

    # 3. resolve_order_origin and format_actor_tag tests
    assert resolve_order_origin({"actor": "AI"}) == ("AI", None, None)
    assert resolve_order_origin({"clientOrderId": "AGS_panic_stop"}) == ("BOT", "assetguardian", "SL_PROTECT")
    assert resolve_order_origin({"clientOrderId": "AGB_level_1"}) == ("BOT", "assetguardian", "BUY_TIER")
    assert resolve_order_origin({"clientOrderId": "MO_TA_kalman_up_1"}) == ("BOT", "tradeall", "KALMAN_UP")
    assert resolve_order_origin({"clientOrderId": "RT_OR_dca_step_1"}) == ("BOT", "rtrade", "DCA_STEP")
    assert resolve_order_origin(
        {"orderId": "123"},
        audit_origin_map={"123": ("tradeall", "kalman_shadow")},
    ) == ("BOT", "tradeall", "kalman_shadow")

    assert format_actor_tag("AI") == "AI"
    assert format_actor_tag("BOT") == "BOT"
    assert format_actor_tag("BOT", module="assetguardian", rule="SL_PANIC") == "BOT(module=assetguardian, rule=SL_PANIC)"
    assert format_actor_tag("BOT", module="tradeall") == "BOT(module=tradeall)"

    # 4. normalize_open_order actor and clientOrderId fields
    norm_ai = normalize_open_order({"orderId": "abc", "clientOrderId": "AI_99", "price": 100, "origQty": 1}, venue="BINANCE", symbol="BTCUSDC")
    assert norm_ai["actor"] == "AI"
    assert norm_ai["clientOrderId"] == "AI_99"
    assert norm_ai["client_order_id"] == "AI_99"
    norm_bot = normalize_open_order({"orderId": "xyz", "clientOrderId": "web_1", "price": 100, "origQty": 1}, venue="BINANCE", symbol="BTCUSDC")
    assert norm_bot["actor"] == "BOT"
    assert norm_bot["clientOrderId"] == "web_1"

    # 5. extract_symbol_recent_fills tagging
    orders = [
        {"orderId": "1", "clientOrderId": "AI_buy_1", "side": "BUY", "price": "100.0", "executedQty": "1.0", "status": "FILLED", "time": 1000},
        {"orderId": "2", "clientOrderId": "bot_sell_1", "side": "SELL", "price": "110.0", "executedQty": "1.0", "status": "FILLED", "time": 2000},
    ]
    res = extract_symbol_recent_fills(orders)
    assert res is not None
    assert res["last_action_actor"] == "BOT"
    assert res["last_action_side"] == "SELL"
    assert res["last_buy"]["actor"] == "AI"
    assert res["last_sell"]["actor"] == "BOT"


def test_conversation_state_snapshot_and_rollback_and_cache_empty_dict_skip(tmp_path):
    """Verify failed model attempts and failed /compact steps are rolled back from <cid>.db and empty {} cache files are skipped."""
    import sqlite3
    from orchestratorOS.agy.summary_store import (
        conversation_db,
        read_summary_row,
        restore_conversation_state,
        snapshot_conversation_state,
        summaries_db,
    )
    from intelligence.macro_sentiment_ssot import (
        load_cached_fear_greed_snapshot,
        load_cached_market_breadth_snapshot,
        load_cached_whale_telemetry,
    )

    cli_base = str(tmp_path / "bot_cli")
    conv_dir = tmp_path / "bot_cli" / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    cid = "cid-rollback-test"
    c_path = conversation_db(cli_base, cid)

    conn = sqlite3.connect(c_path)
    conn.execute("CREATE TABLE `steps` (`idx` integer PRIMARY KEY, `step_type` integer, `step_payload` blob)")
    conn.execute("CREATE TABLE `gen_metadata` (`idx` integer PRIMARY KEY, `data` blob, `size` integer)")
    conn.execute("CREATE TABLE `trajectory_metadata_blob` (`id` text PRIMARY KEY, `data` blob)")
    conn.execute("INSERT INTO `steps` VALUES (0, 14, ?), (1, 15, ?)", (b"prompt-0", b"reply-0"))
    conn.execute("INSERT INTO `gen_metadata` VALUES (0, ?, 10)", (b"meta-0",))
    conn.execute("INSERT INTO `trajectory_metadata_blob` VALUES ('main', ?)", (b"blob-v1",))
    conn.commit()
    conn.close()

    s_conn = sqlite3.connect(summaries_db(cli_base))
    s_conn.execute(
        "CREATE TABLE conversation_summaries (conversation_id text primary key, title text, step_count integer, raw_summary blob)"
    )
    s_conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)", (cid, "Topic Title", 2, b"raw-v1"))
    s_conn.commit()
    s_conn.close()

    snap = snapshot_conversation_state(cli_base, cid)
    assert snap is not None

    # Simulate failed model attempt polluting <cid>.db and conversation_summaries.db
    conn = sqlite3.connect(c_path)
    conn.execute(
        "INSERT INTO `steps` VALUES (2, 14, ?), (3, 101, ?), (4, 17, ?)",
        (b"duplicate-80kb-prompt", b"header", b"RESOURCE_EXHAUSTED"),
    )
    conn.execute("INSERT INTO `gen_metadata` VALUES (1, ?, 99)", (b"meta-polluted",))
    conn.execute("UPDATE `trajectory_metadata_blob` SET data=? WHERE id='main'", (b"blob-polluted",))
    conn.commit()
    conn.close()

    s_conn = sqlite3.connect(summaries_db(cli_base))
    s_conn.execute("UPDATE conversation_summaries SET step_count=5, raw_summary=? WHERE conversation_id=?", (b"raw-polluted", cid))
    s_conn.commit()
    s_conn.close()

    assert restore_conversation_state(cli_base, cid, snap) is True

    conn = sqlite3.connect(c_path)
    steps_after = conn.execute("SELECT idx, step_type, step_payload FROM steps ORDER BY idx").fetchall()
    gen_after = conn.execute("SELECT idx, data FROM gen_metadata ORDER BY idx").fetchall()
    blob_after = conn.execute("SELECT data FROM trajectory_metadata_blob WHERE id='main'").fetchone()[0]
    conn.close()

    assert steps_after == [(0, 14, b"prompt-0"), (1, 15, b"reply-0")]
    assert gen_after == [(0, b"meta-0")]
    assert blob_after == b"blob-v1"
    s_row = read_summary_row(cli_base, cid)
    assert s_row is not None and s_row["step_count"] == 2 and s_row["raw_summary"] == b"raw-v1"

    # Verify empty {} JSON files return None instead of fabricating synthetic 50 / 0.5 snapshots, and skip to next symbol
    cache_dir = tmp_path / "cache_test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "fear_greed_collect.json").write_text("{}")
    assert load_cached_fear_greed_snapshot(str(cache_dir)) is None

    (cache_dir / "market_breadth_collect.json").write_text("{}")
    assert load_cached_market_breadth_snapshot(str(cache_dir)) is None

    (cache_dir / "whale_snapshot_BTCUSDC.json").write_text("{}")
    (cache_dir / "whale_snapshot_BTCUSDT.json").write_text(json.dumps({"long_ratio": 0.64}))
    w_snap = load_cached_whale_telemetry(str(cache_dir), symbols=["BTCUSDC", "BTCUSDT"])
    assert w_snap == {"long_ratio": 0.64}


def test_gemini_client_pre_call_quota_filter_and_dynamic_topup_restoration(tmp_path, monkeypatch):
    """Verify low-latency pre-call quota check skips 0-credit models and dynamically restores primary on top-up."""
    from intelligence.sentiment.gemini_client import GeminiClient

    cli_bin = tmp_path / "agy-bot"
    cli_bin.write_text("#!/bin/sh\n")
    bot_cli_base = tmp_path / "bot_cli"
    bot_cli_base.mkdir(parents=True, exist_ok=True)
    quota_state_file = tmp_path / "llm_quota_status.json"

    (bot_cli_base / "antigravity-oauth-token").write_text(
        json.dumps(
            {
                "token": {"access_token": "tok-live-123", "expires_in": 3599},
                "auth_method": "oauth",
            }
        )
    )
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", staticmethod(lambda: str(quota_state_file)))
    GeminiClient._QUOTA_MEM_CACHE = {}

    # Phase 1: Argon models exhausted (absent from buckets, 0 availableCredits), Claude & GPT-OSS have 100% quota
    api_state = {
        "buckets": [
            {"modelId": "claude-sonnet-5-5-medium", "remainingFraction": 1.0},
            {"modelId": "claude-sonnet-5-5-low", "remainingFraction": 1.0},
            {"modelId": "gpt-oss-120b-medium", "remainingFraction": 1.0},
        ],
        "availableCredits": [],
    }

    class _FakeResp:
        def __init__(self, payload: dict):
            self._raw = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=2.5):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        assert req.headers.get("User-agent") == "antigravity/1.4.12"
        if "retrieveUserQuota" in url:
            return _FakeResp({"buckets": api_state["buckets"]})
        if "loadCodeAssist" in url:
            return _FakeResp({"paidTier": {"id": "g1-ultra-tier", "availableCredits": api_state["availableCredits"]}})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    snap1 = GeminiClient.probe_live_model_quota(bot_cli_base=str(bot_cli_base), force_refresh=True)
    assert snap1 is not None
    assert snap1["available_credits"] == 0.0

    client = GeminiClient(cli_path=str(cli_bin))
    med_candidates = client.get_model_candidates("gemini-4-argon-medium")
    filtered_med = GeminiClient.filter_models_with_confirmed_credits(med_candidates, snap1)
    # Argon models (0% quota) are skipped immediately; paid claude-sonnet-5-5-medium is first!
    assert filtered_med == [
        "claude-sonnet-5-5-medium",
        "claude-sonnet-5-5-low",
        "gpt-oss-120b-medium",
    ]

    low_candidates = client.get_model_candidates("gemini-4-argon-low")
    filtered_low = GeminiClient.filter_models_with_confirmed_credits(low_candidates, snap1)
    assert filtered_low[0] == "claude-sonnet-5-5-low"

    # Phase 2: User tops up paid credits (or 5h window resets) -> Argon dynamically restored on next probe!
    api_state["availableCredits"] = [{"creditAmount": 25.0}]
    snap2 = GeminiClient.probe_live_model_quota(bot_cli_base=str(bot_cli_base), force_refresh=True)
    assert snap2 is not None
    assert snap2["available_credits"] == 25.0
    restored_med = GeminiClient.filter_models_with_confirmed_credits(med_candidates, snap2)
    assert restored_med[0] == "gemini-4-argon-medium"


def test_gemini_client_low_quota_error_ntfy_alert_and_cooldown(tmp_path, monkeypatch):
    """Verify NTFY_TOPIC_ERROR alert fires when Argon quota is near zero (<=10%) and resets on top-up."""
    from intelligence.sentiment.gemini_client import GeminiClient

    quota_state_file = tmp_path / "llm_quota_alert.json"
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", staticmethod(lambda: str(quota_state_file)))
    GeminiClient._QUOTA_MEM_CACHE = {}

    alerts = []

    def _mock_ntfy(title, message):
        alerts.append({"title": title, "message": message})
        return True

    _mock_ntfy._mock_name = "send_ntfy"
    monkeypatch.setattr(
        "orchestratorOS.lib.watchdog_common.send_ntfy",
        _mock_ntfy,
    )
    monkeypatch.setenv("NTFY_TOPIC_ERROR", "ntfy-error-1978")
    monkeypatch.setenv("LLM_QUOTA_LOW_WARN_FRACTION", "0.10")
    monkeypatch.setenv("LLM_QUOTA_ALERT_COOLDOWN_SEC", "14400")

    # 1. Argon at 0%, Claude at 100% -> triggers argon_low alert on ntfy-error-1978
    snap_low = {
        "checked_ts": time.time(),
        "available_credits": 0.0,
        "buckets": {
            "claude-sonnet-5-5-medium": 1.0,
            "gpt-oss-120b-medium": 1.0,
        },
        "last_alert_ts": 0.0,
        "last_alert_tier": "ok",
    }
    GeminiClient._maybe_notify_low_quota(snap_low)
    assert len(alerts) == 1
    assert "0%" in alerts[0]["title"]
    assert snap_low["last_alert_tier"] == "argon_low"

    # 2. Repeated check within 4h cooldown at same severity does NOT spam
    GeminiClient._maybe_notify_low_quota(snap_low)
    assert len(alerts) == 1

    # 3. Escalation: Claude also drops to 5% (<= 10%) -> alerts immediately despite cooldown
    snap_low["buckets"]["claude-sonnet-5-5-medium"] = 0.05
    GeminiClient._maybe_notify_low_quota(snap_low)
    assert len(alerts) == 2
    assert snap_low["last_alert_tier"] == "all_paid_low"

    # 4. User tops up credits -> alert tier resets to "ok"
    snap_low["available_credits"] = 50.0
    GeminiClient._maybe_notify_low_quota(snap_low)
    assert snap_low["last_alert_tier"] == "ok"

    # 5. _extract_available_credits does not double-count identical tier IDs in paidTier and currentTier
    dup_tier_payload = {
        "paidTier": {"id": "g1-ultra-tier", "availableCredits": [{"creditAmount": 25.0}]},
        "currentTier": {"id": "g1-ultra-tier", "availableCredits": [{"creditAmount": 25.0}]},
    }
    assert GeminiClient._extract_available_credits(dup_tier_payload) == 25.0

    # 6. Runtime credit exhaustion on gemini-4-argon-medium clears shared Gemini/Argon pool buckets & available_credits
    quota_state_file.write_text(
        json.dumps({
            "checked_ts": time.time(),
            "available_credits": 10.0,
            "buckets": {
                "gemini-4-argon-high": 0.8,
                "gemini-4-argon-medium": 0.8,
                "gemini-4-argon-low": 0.8,
                "gemini-3.8-flash-medium": 0.8,
                "gemini-3.8-flash-low": 0.8,
                "claude-sonnet-5-5-medium": 1.0,
                "gpt-oss-120b-medium": 1.0,
            },
            "last_alert_ts": 0.0,
            "last_alert_tier": "ok",
        }),
        encoding="utf-8",
    )
    alerts.clear()
    GeminiClient._record_model_quota_exhausted("gemini-4-argon-medium")
    assert len(alerts) == 1
    assert GeminiClient._QUOTA_MEM_CACHE["available_credits"] == 0.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-4-argon-low"] == 0.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-4-argon-high"] == 0.0

    # 7. filter_models_with_confirmed_credits never returns an empty list if all buckets report 0.0
    all_zero_snap = {
        "checked_ts": time.time(),
        "available_credits": 0.0,
        "buckets": {"claude-sonnet-5-5-medium": 0.0, "gpt-oss-120b-medium": 0.0},
        "runtime_exhausted": {},
    }
    cands = ["claude-sonnet-5-5-medium", "gpt-oss-120b-medium"]
    assert GeminiClient.filter_models_with_confirmed_credits(cands, all_zero_snap) == cands








