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


def test_unified_ai_agent_base_classes_and_query_topic_json():
    """Verify PreTradeLLMGuardBase, CachedTopicAssessmentAgent, and query_topic_json normalization."""
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro_sentiment_ssot import CachedTopicAssessmentAgent
    from intelligence.sentiment.gemini_client import GeminiClient, query_topic_json
    from intelligence.sentiment.guards import HighStakeGuard, PostStopReentryGuard, PreTradeLLMGuardBase
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor

    assert issubclass(HighStakeGuard, PreTradeLLMGuardBase)
    assert issubclass(PostStopReentryGuard, PreTradeLLMGuardBase)
    assert issubclass(GeopoliticalThreatAnalyzer, CachedTopicAssessmentAgent)
    assert issubclass(SentimentAdvisor, CachedTopicAssessmentAgent)

    recorded_calls = []

    class _SpyClient(GeminiClient):
        def query_json(self, prompt, **kwargs):
            recorded_calls.append((prompt, kwargs))
            return {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"}

    spy = _SpyClient()
    res = spy.query_topic_json("guard_high_stake", "Evaluate BUY", subtitle="BUY BTCUSDC €1,000.00")
    assert res == {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"}
    assert len(recorded_calls) == 1
    _, kw = recorded_calls[0]
    assert kw["topic_key"] == "guard_high_stake"
    assert "BUY BTCUSDC €1,000.00" in kw["thread_title"]
    assert kw["timeout_sec"] == 120.0
    assert query_topic_json(None, "guard_high_stake", "noop") is None


def test_pretrade_guard_cross_mode_cache_isolation_and_high_stake_mode():
    """Verify PostStopReentryGuard and HighStakeGuard adapt cached decisions across enforce <-> shadow modes."""
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.guards import HighStakeGuard, PostStopReentryGuard
    from intelligence.sentiment.guards import high_stake_guard as hsg_mod
    from intelligence.sentiment.guards import post_stop_reentry_guard as psg_mod

    psg_mod._POST_STOP_DECISION_CACHE.clear()
    hsg_mod._HIGH_STAKE_DECISION_CACHE.clear()

    class _VetoClient(GeminiClient):
        def __init__(self):
            self.calls = 0

        def query_json(self, prompt, **kwargs):
            self.calls += 1
            return {
                "decision": "REJECTED",
                "suggested_scale": 0.0,
                "reason": "Momentum downward",
            }

    client = _VetoClient()
    sell_ts = time.time() - 60.0

    # 1. Enforce guard caches VETO; subsequent shadow guard on same stop must NOT block
    g_enforce = PostStopReentryGuard(gemini_client=client, mode="enforce", notify_enabled=False, cache_ttl_sec=60.0)
    d_enf = g_enforce.check(
        "ETHUSDC",
        current_price=2010.0,
        last_sell_price=2000.0,
        trough=1950.0,
        bounce_pct=0.03,
        qty=1.0,
        notional_eur=300.0,
        last_sell_time_s=sell_ts,
        telemetry={},
    )
    assert d_enf.allowed is False
    assert client.calls == 1

    g_shadow = PostStopReentryGuard(gemini_client=client, mode="shadow", notify_enabled=False, cache_ttl_sec=60.0)
    d_shd = g_shadow.check(
        "ETHUSDC",
        current_price=2010.0,
        last_sell_price=2000.0,
        trough=1950.0,
        bounce_pct=0.03,
        qty=1.0,
        notional_eur=300.0,
        last_sell_time_s=sell_ts,
        telemetry={},
    )
    assert client.calls == 1  # Served from cache
    assert d_shd.allowed is True
    assert d_shd.reason.startswith("shadow_mode")
    assert d_shd.shadow_decision is not None
    assert d_shd.shadow_decision.allowed is False

    # 2. Reverse: Shadow guard populates cache first; subsequent enforce guard must enforce the underlying VETO
    hsg_mod._HIGH_STAKE_DECISION_CACHE.clear()
    hs_shadow = HighStakeGuard(
        min_notional_eur=500.0, gemini_client=client, mode="shadow", notify_enabled=False, cache_ttl_sec=60.0
    )
    hs_d_shd = hs_shadow.check("BTCUSDC", "BUY", 85000.0, 0.02, notional_eur=1200.0, telemetry={})
    assert client.calls == 2
    assert hs_d_shd.allowed is True
    assert hs_d_shd.shadow_decision is not None and hs_d_shd.shadow_decision.allowed is False

    hs_enforce = HighStakeGuard(
        min_notional_eur=500.0, gemini_client=client, mode="enforce", notify_enabled=False, cache_ttl_sec=60.0
    )
    hs_d_enf = hs_enforce.check("BTCUSDC", "BUY", 85000.0, 0.02, notional_eur=1200.0, telemetry={})
    assert client.calls == 2  # Served from cache
    assert hs_d_enf.allowed is False
    assert "Momentum downward" in hs_d_enf.reason

    # 3. HighStakeGuard(mode="off") immediately allows without calling Gemini
    hs_off = HighStakeGuard(min_notional_eur=500.0, gemini_client=client, mode="off", notify_enabled=False)
    hs_d_off = hs_off.check("BTCUSDC", "BUY", 85000.0, 0.02, notional_eur=1200.0, telemetry={})
    assert hs_d_off.allowed is True
    assert hs_d_off.reason == "high_stake_guard_disabled"
    assert client.calls == 2


def test_macro_analyzer_and_daemon_apply_retry_backoff_on_llm_failure(monkeypatch, tmp_path):
    """Verify MacroAnalyzer.run_cycle and IntelligenceTelemetryDaemon.run_cycle back off via compute_step_retry_ts."""
    from intelligence.daemon import IntelligenceTelemetryDaemon
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisorAssessment

    analyzer = MacroAnalyzer(
        cache_dir=str(tmp_path),
        macro_llm_interval_sec=1800.0,
        advisor_interval_sec=10800.0,
    )
    analyzer._last_macro_llm_ts = 0.0
    analyzer._last_advisor_ts = 0.0

    monkeypatch.setattr(analyzer, "write_heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(analyzer, "prune_cli_threads", lambda *a, **k: 0)
    monkeypatch.setattr(analyzer, "assess_geopolitical_threat", lambda **kw: False)
    monkeypatch.setattr(analyzer, "assess_market_advisor", lambda **kw: False)

    now_ts = time.time()
    res = analyzer.run_cycle(now=now_ts)
    assert res["geopolitical_updated"] is False
    assert res["advisor_updated"] is False
    # Instead of staying at 0.0 (hammering every 5s), both timestamps advance so next retry is ~300s away
    eff_geo_int = res["effective_geo_interval_sec"]
    eff_adv_int_1 = res["effective_advisor_interval_sec"]
    assert analyzer._last_macro_llm_ts > now_ts - eff_geo_int
    assert analyzer._last_advisor_ts > now_ts - eff_adv_int_1

    # Also test when SentimentAdvisor falls back to a stale cached assessment on LLM failure
    stale_ts = now_ts - max(12000.0, eff_adv_int_1 + 1200.0)
    analyzer.sentiment_advisor._last_eval_ts = stale_ts
    analyzer.sentiment_advisor._cached_assessment = SentimentAdvisorAssessment(
        market_bias="NEUTRAL",
        risk_level="LOW",
        confidence=0.5,
        summary="stale",
        key_risks=[],
        recommended_action="HOLD",
        ts=stale_ts,
    )
    analyzer._last_advisor_ts = stale_ts
    monkeypatch.setattr(analyzer, "assess_market_advisor", lambda **kw: True)
    res2 = analyzer.run_cycle(now=now_ts)
    assert res2["advisor_updated"] is False
    # Backed off by 25% of eff_adv_int rather than locking out for the full interval
    eff_adv_int = res2["effective_advisor_interval_sec"]
    assert abs((analyzer._last_advisor_ts + eff_adv_int) - (now_ts + eff_adv_int * 0.25)) < 5.0

    # IntelligenceTelemetryDaemon.run_cycle also backs off when update_macro() returns False
    daemon = IntelligenceTelemetryDaemon(cache_dir=str(tmp_path), symbols=[])
    daemon._last_macro_ts = 0.0
    monkeypatch.setattr(daemon, "_step_external_symbols", lambda *a, **k: None)
    monkeypatch.setattr(daemon, "write_heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(daemon, "update_macro", lambda **kw: False)
    d_res = daemon.run_cycle(now=now_ts)
    assert d_res["macro_updated"] is False
    assert daemon._last_macro_ts > 0.0


def test_orderbook_depth_collector_uses_eur_endpoints_for_eur_pairs(monkeypatch, tmp_path):
    """Verify OrderbookDepthCollector.fetch queries EUR orderbook endpoints for EUR-quoted symbols."""
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector

    collector = OrderbookDepthCollector(cache_dir=str(tmp_path))
    requested_urls = []

    def _fake_fetch_json_dict(url, *args, **kwargs):
        requested_urls.append(url)
        return {
            "bids": [["60000.0", "1.5"], ["59990.0", "2.0"]],
            "asks": [["60010.0", "1.2"], ["60020.0", "1.8"]],
        }

    monkeypatch.setattr(collector, "_fetch_json_dict", _fake_fetch_json_dict)
    snap = collector.fetch("XBTEUR", allow_network=True)
    assert snap is not None
    assert snap.mid_price == 60005.0
    assert requested_urls
    assert "symbol=BTCEUR" in requested_urls[0]
    assert "BTCUSDT" not in requested_urls[0]


def test_compact_model_candidates_skips_failed_attempts_and_runtime_exhausted(monkeypatch):
    """Verify get_compact_model_candidates excludes models from earlier failed attempts and runtime-exhausted pools."""
    from intelligence.sentiment.gemini_client import GeminiClient

    client = GeminiClient()
    models_to_try = [
        "gpt-oss-120b-medium",
        "gemini-4-argon-low",
        "claude-sonnet-5-5-low",
        "gemini-3.8-flash-low",
    ]
    # Suppose attempt_idx=2 (gpt-oss-120b-medium failed at idx 0, gemini-4-argon-low quota-exhausted at idx 1)
    monkeypatch.setattr(
        GeminiClient,
        "_QUOTA_MEM_CACHE",
        {
            "runtime_exhausted": {
                "gemini-4-argon-low": 9999999999.0,
                "gemini-3.8-flash-low": 9999999999.0,
            }
        },
    )
    cands = client.get_compact_model_candidates(models_to_try, attempt_idx=2)
    assert "gpt-oss-120b-medium" not in cands
    assert "gemini-4-argon-low" not in cands
    assert "gemini-3.8-flash-low" not in cands
    assert cands == ["claude-sonnet-5-5-low"]


def test_find_duplicate_ai_open_limit_order_matches_wallet_clamped_intent():
    """Verify find_duplicate_ai_open_limit_order recognizes a resting AI limit order clamped by wallet balance."""
    from providers.execution_audit import find_duplicate_ai_open_limit_order

    class _StubInst:
        symbol = "TAOUSD"

        def open_orders(self):
            return [
                {
                    "orderId": "O-CLAMPED-1",
                    "side": "BUY",
                    "price": "100.0",
                    "origQty": "3.2",
                    "executedQty": "0.0",
                }
            ]

    prior_intents = [
        {
            "intent_id": "intent_1700000000_buy_taousd",
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "binance",
            "suggested_notional_usd": 500.0,
            "clamped_notional_usd": 320.0,
            "order_type": "LIMIT",
            "limit_price": 100.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "O-CLAMPED-1"},
            "history": [{"action": "EXECUTED", "order_id": "O-CLAMPED-1", "notional_usd": 320.0}],
        }
    ]

    # Same unclamped target notional ($500) -> matches existing $320 wallet-clamped order
    dup = find_duplicate_ai_open_limit_order(
        _StubInst(),
        "BUY",
        100.0,
        500.0,
        intents=prior_intents,
    )
    assert dup == "O-CLAMPED-1"

    # Intentionally changed target notional ($200) -> does NOT match the $500->$320 order
    dup_changed = find_duplicate_ai_open_limit_order(
        _StubInst(),
        "BUY",
        100.0,
        200.0,
        intents=prior_intents,
    )
    assert dup_changed is None


def test_equity_symbol_normalization_and_pretrade_telemetry_cache_isolation(tmp_path):
    """Verify delimiter-stripped USEQ/_EQ symbol normalization and custom cachedb_dir isolation in collect_pretrade_telemetry."""
    import json
    from cachepaths import (
        cache_symbol_variants,
        canonical_symbol_key,
        normalize_cache_base_asset,
        split_symbol_assets,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import collect_pretrade_telemetry

    assert normalize_cache_base_asset("SPCXUSEQ") == "SPCX"
    assert split_symbol_assets("SPCXUSEQ") == ("SPCX", "USD")
    assert canonical_symbol_key("SPCXUSEQ") == "SPCXUSD"
    assert split_symbol_assets("RGNT_EQ") == ("RGNT", "USD")
    assert canonical_symbol_key("RGNT_EQ") == "RGNTUSD"
    assert "SPCXUSEQUSD" not in cache_symbol_variants("SPCX_US_EQ")

    # Write a USD orderbook file and an EUR orderbook file in tmp_path
    (tmp_path / "orderbook_depth_BTCUSD.json").write_text(
        json.dumps({"symbol": "BTCUSD", "imbalance_ratio": 0.90, "mid_price": 70000.0}),
        encoding="utf-8",
    )
    (tmp_path / "orderbook_depth_BTCEUR.json").write_text(
        json.dumps({"symbol": "BTCEUR", "imbalance_ratio": 0.25, "mid_price": 64000.0}),
        encoding="utf-8",
    )
    telem = collect_pretrade_telemetry("BTCEUR", cachedb_dir=str(tmp_path))
    assert "25% bids vs 75% asks" in telem.get("Orderbook Depth Imbalance", "")


def test_harvest_recent_orders_recomputes_flat_after_exit_when_free_base_populated_from_cycle(tmp_path):
    """When pos['free_base'] is None and populated from a dust cycle qty (< min_notional) after a SELL exit, stale anchors must be cleared."""
    import os
    from intelligence.sentiment.collectors.portfolio_collector import harvest_recent_orders_and_enrich_positions

    pos = {
        "venue": "kraken",
        "symbol": "SOLUSD",
        "quote": "USD",
        "free_base": None,
        "free_quote": 500.0,
        "price": 100.0,
        "exchange_min_notional_usd": 10.0,
        "trailing_peak": 125.0,
        "dd_from_peak_pct": -20.0,
    }
    kraken_cycles = {
        "SOLUSD": {
            "qty": 0.02,  # $2.00 notional < $10.00 exchange_min_notional_usd (dust after SELL exit)
            "cost_usd": 2.40,
            "avg_cost_px": 120.0,
            "pos_peak": 125.0,
            "dd_pct": -16.67,
        }
    }
    fake_files = {
        "cache_trade_kraken.json": {
            "items": {
                "SOLUSD": [
                    {"side": "BUY", "price": 120.0, "qty": 1.0, "executedQty": 1.0, "timestamp": 1700000000000, "status": "FILLED"},
                    {"side": "SELL", "price": 115.0, "qty": 0.98, "executedQty": 0.98, "timestamp": 1700000100000, "status": "FILLED"},
                ]
            }
        }
    }
    harvest_recent_orders_and_enrich_positions(
        str(tmp_path),
        {},
        [pos],
        {"SOLUSD": {"price": 100.0}},
        kraken_cycles,
        {},
        {},
        read_json_fn=lambda p: fake_files.get(os.path.basename(p), {}),
    )
    assert pos["free_base"] == 0.02
    assert "trailing_peak" not in pos
    assert "active_avg_buy_px" not in pos
    assert "tactical_avg_buy_px" not in pos
    assert "pnl_vs_tactical_avg_pct" not in pos
    assert kraken_cycles["SOLUSD"]["avg_cost_px"] is None
    assert kraken_cycles["SOLUSD"]["cost_usd"] == 0.0


def test_flat_cycle_clears_tactical_avg_and_semantic_stem_and_macro_analyzer_whale_flip(tmp_path):
    """Verify tactical_avg_buy_px cleanup on cyc_flat, silent-e/comma stem normalization, str obj_attr safety, and MacroAnalyzer whale flip interval wiring."""
    import json
    import os
    from types import SimpleNamespace
    from botcore import obj_attr
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.macro_sentiment_ssot import (
        _normalize_semantic_stem,
        headlines_are_semantic_duplicates,
        resolve_macro_economic_calendar,
    )
    from intelligence.sentiment.collectors.portfolio_collector import harvest_recent_orders_and_enrich_positions

    # 1. obj_attr on str returns default instead of bound str method; resolve_macro_economic_calendar accepts str headlines
    assert obj_attr("US CPI print", "title", "") == ""
    cal = resolve_macro_economic_calendar(now_ts=1762196700.0, breaking_news=["US CPI inflation print rises"])
    assert cal["macro_release_confirmed"] is True
    assert cal["detected_macro_headlines"] == ["US CPI inflation print rises"]

    # 2. Semantic stemmer aligns silent-e verbs, preserves -is/-us/-as nouns, and normalizes comma numbers
    assert _normalize_semantic_stem("surges") == _normalize_semantic_stem("surged") == _normalize_semantic_stem("surging")
    assert _normalize_semantic_stem("crashes") == _normalize_semantic_stem("crashed") == _normalize_semantic_stem("crashing")
    assert _normalize_semantic_stem("plunging") == _normalize_semantic_stem("plunges")
    assert _normalize_semantic_stem("crisis") == "crisis"
    assert _normalize_semantic_stem("texas") == "texas"
    assert headlines_are_semantic_duplicates(
        "Bitcoin surges past $70,000 after Fed announces rate cut",
        "Bitcoin surged past $70000 after Fed announced rate cuts",
    ) is True

    # 3. cyc_flat clears tactical_avg_buy_px and pnl_vs_tactical_avg_pct from pos and ro
    pos = {
        "venue": "kraken",
        "symbol": "TAOUSD",
        "quote": "USD",
        "free_base": 0.0,
        "free_quote": 500.0,
        "price": 225.0,
        "exchange_min_notional_usd": 10.0,
    }
    krk_cycles = {"TAOUSD": {"qty": 0.0, "cost_usd": 0.0, "avg_cost_px": None}}
    fake_files = {
        "cache_trade_kraken.json": {
            "items": {
                "TAOUSD": [
                    {"side": "BUY", "price": 244.40, "qty": 10.0, "executedQty": 10.0, "timestamp": 1000, "status": "FILLED"},
                    {"side": "SELL", "price": 226.40, "qty": 4.0, "executedQty": 4.0, "timestamp": 2000, "status": "FILLED"},
                    {"side": "BUY", "price": 223.00, "qty": 4.0, "executedQty": 4.0, "timestamp": 3000, "status": "FILLED"},
                ]
            }
        }
    }
    ro_out = harvest_recent_orders_and_enrich_positions(
        str(tmp_path),
        {},
        [pos],
        {"TAOUSD": {"price": 225.0}},
        krk_cycles,
        {},
        {},
        read_json_fn=lambda p: fake_files.get(os.path.basename(p), {}),
    )
    assert "tactical_avg_buy_px" not in pos
    assert "pnl_vs_tactical_avg_pct" not in pos
    assert "tactical_avg_buy_px" not in ro_out.get("TAOUSD", {})

    # 4. MacroAnalyzer wires btc_taker_ratio and prev_btc_taker_ratio into compute_dynamic_sentiment_interval
    (tmp_path / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps({"symbol": "BTCUSDC", "taker_buy_sell_ratio": 1.42, "ts": 1000.0}),
        encoding="utf-8",
    )
    analyzer = MacroAnalyzer(cache_dir=str(tmp_path), symbols=["BTCUSDC"])
    analyzer._last_btc_taker_ratio = 0.68
    analyzer._last_advisor_ts = 900.0
    analyzer.sentiment_advisor._cached_assessment = SimpleNamespace(
        market_bias="BEARISH",
        risk_level="HIGH",
        recommended_action="HALT_BUYS",
        is_fallback=False,
        ts=900.0,
    )
    analyzer.geopolitical_analyzer._cached_assessment = SimpleNamespace(
        threat_level="NORMAL",
        risk_score=0.1,
        recommended_brake="NONE",
        headlines_analyzed=0,
        is_fallback=False,
        ts=900.0,
    )
    analyzer.assess_geopolitical_threat = lambda **kw: False
    analyzer.assess_market_advisor = lambda **kw: True
    res = analyzer.run_cycle(force=False, now=1000.0)
    assert res["effective_advisor_interval_sec"] <= 300.0
    assert "btc_whale_flow_regime_flip" in res["advisor_interval_reason"]


def test_patch_uniqueness_and_news_semantic_cleanup_and_position_pct_fallback(tmp_path):
    from datetime import datetime, timezone
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsHeadline
    from orchestratorOS.lib.technical_repair_ssot import apply_code_patch_safely, apply_reconciliation_plan
    from orchestratorOS.supervisor.prompt_formatters import _format_tradable_position_line

    # 1a. apply_reconciliation_plan must reject missing/None replace_block instead of coercing to ""
    called_patches = []
    plan_missing_replace = {
        "code_remediation": {
            "has_fix": True,
            "target_file": "botcore.py",
            "search_block": "OLD_CODE = 1",
            "replace_block": None,
            "rationale": "malformed patch without replace_block",
        }
    }
    changed, passed = apply_reconciliation_plan(
        plan_missing_replace,
        str(tmp_path),
        is_enforce=True,
        execute_action_fn=lambda a: False,
        apply_patch_fn=lambda **kw: (called_patches.append(kw) or True),
    )
    assert changed is False
    assert passed is None
    assert called_patches == []

    # 1b. apply_code_patch_safely must refuse ambiguous non-unique search_block matches
    target_py = tmp_path / "botcore.py"
    orig_text = "x = 1\ny = 2\nx = 1\n"
    target_py.write_text(orig_text, encoding="utf-8")
    assert (
        apply_code_patch_safely(
            target_file=str(target_py),
            search_block="x = 1",
            replace_block="x = 99",
            rationale="ambiguous block",
            workspace_dir=str(tmp_path),
        )
        is False
    )
    assert target_py.read_text(encoding="utf-8") == orig_text

    # 2. NewsFeedCollector._aggregate deduplicates pre-existing semantic duplicates in _aggregated_pool
    collector = NewsFeedCollector(history_file=str(tmp_path / "news_feed_collect.json"))
    now_ts = float(datetime(2026, 11, 3, 10, 10, 0, tzinfo=timezone.utc).timestamp())
    h1 = NewsHeadline(
        title="Iran launches massive missile attack on Israel",
        source="Reuters",
        pub_date="Tue, 03 Nov 2026 10:00:00 GMT",
        url="https://example.com/reuters-1",
        matched_keywords=("missile", "attack"),
        is_high_severity=True,
    )
    h2 = NewsHeadline(
        title="Iran launches massive missile attacks against Israel - Bloomberg",
        source="Bloomberg",
        pub_date="Tue, 03 Nov 2026 10:05:00 GMT",
        url="https://example.com/bloomberg-2",
        matched_keywords=("missile", "attack", "war"),
        is_high_severity=True,
    )
    collector._aggregated_pool = {
        "iran launches massive missile attack on israel": (now_ts - 300.0, h1),
        "iran launches massive missile attacks against israel": (now_ts - 120.0, h2),
    }
    snap = collector._aggregate([], now=now_ts)
    assert len(collector._aggregated_pool) == 1
    assert len(snap.headlines) == 1
    assert "war" in snap.headlines[0].matched_keywords

    # 3. _format_tradable_position_line computes fallback pct or omits None% suffix
    line_with_px = _format_tradable_position_line(
        {
            "venue": "kraken",
            "symbol": "BTCUSD",
            "base": "BTC",
            "quote": "USD",
            "price": 105.0,
            "free_base": 1.0,
            "value_usd": 105.0,
            "free_quote": 500.0,
            "active_avg_buy_px": 100.0,
            "pnl_vs_avg_pct": None,
        }
    )
    assert "avg_buy_px=100.0 (pnl_vs_avg=5.0%)" in line_with_px
    assert "None%" not in line_with_px

    line_without_px = _format_tradable_position_line(
        {
            "venue": "kraken",
            "symbol": "BTCUSD",
            "base": "BTC",
            "quote": "USD",
            "price": None,
            "free_base": 1.0,
            "value_usd": None,
            "free_quote": 500.0,
            "active_avg_buy_px": 100.0,
            "pnl_vs_avg_pct": None,
        }
    )
    assert "avg_buy_px=100.0" in line_without_px
    assert "None%" not in line_without_px


def test_telemetry_deep_bug_hunt_fixes_a_through_g():
    """Verify fixes for Bugs A-G discovered in live supervisor prompt telemetry audit."""
    from datetime import datetime, timezone
    from intelligence.macro_sentiment_ssot import (
        detect_active_binance_symbols,
        match_high_severity_keywords,
        resolve_macro_economic_calendar,
    )
    from intelligence.sentiment.collectors.portfolio_collector import (
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
    )
    from orchestratorOS.supervisor.prompt_formatters import (
        _format_market_prices_section,
        _format_portfolio_section,
    )

    # Bug A: detect_active_binance_symbols includes TAOUSD alongside TAOUSDC when configured on Kraken
    active_syms = detect_active_binance_symbols(include_kraken_cache=True)
    assert "TAOUSD" in active_syms
    assert "HYPEUSD" in active_syms

    # Bug B: Midterm election 'inflation' liveblog does not trigger tomorrow's CPI/NFP release confirmation
    now_tue = float(datetime(2026, 11, 3, 20, 26, 0, tzinfo=timezone.utc).timestamp())
    cal = resolve_macro_economic_calendar(
        now_ts=now_tue,
        breaking_news=["Midterm elections 2026 live updates: Economy, inflation and other voter concerns"],
    )
    assert cal["macro_release_confirmed"] is False
    assert all("CPI / PPI / NFP" not in u for u in cal["upcoming_windows"])

    # Even when today has a legitimate macro headline, tomorrow's 15:30 EET window remains routine
    cal_with_today_print = resolve_macro_economic_calendar(
        now_ts=now_tue,
        breaking_news=["Job openings posted big September gain in positive sign for labor market ahead of payrolls report"],
    )
    assert cal_with_today_print["macro_release_confirmed"] is True
    assert any("Pre-Market Macro & Treasury Liquidity Window (08:30 EST) at Wed 15:30" in u for u in cal_with_today_print["upcoming_windows"])
    assert all("CPI / PPI / NFP" not in u for u in cal_with_today_print["upcoming_windows"])

    # Bug C: Partisan election war opinion & single-stock merchandise plunge are excluded from [HIGH-SEV]
    assert match_high_severity_keywords("DeSantis: War in Iran 'big, big drag' on Republicans - The Hill") == ()
    assert match_high_severity_keywords("Shopify Stock Plunges. Slowing Merchandise Sales Flash a Warning Sign.") == ()
    assert "war" in match_high_severity_keywords("Iran launches missile strike as war escalates in Middle East")

    # Bug D: enrich_open_limit_orders_tca ignores top-of-book touch walls (<10bps) and never recommends crossing best_bid/ask
    hype_orders = [{"symbol": "HYPEUSDC", "side": "SELL", "price": 88.0}]
    enrich_open_limit_orders_tca(
        hype_orders,
        {"HYPEUSDC": {"price": 85.1165, "vol_1h_pct": 0.89}},
        {"HYPEUSDC": {"ob_mid_price": 85.1165, "spread_bps": 0.12, "ask_wall_px": 85.122, "ask_wall_usd": 160919.0}},
    )
    assert "wall_queue_pos" not in hype_orders[0]
    assert "recommended_front_run_px" not in hype_orders[0]

    btc_orders = [{"symbol": "BTCUSDC", "side": "SELL", "price": 78950.0}]
    enrich_open_limit_orders_tca(
        btc_orders,
        {"BTCUSDC": {"price": 77536.43, "vol_1h_pct": 0.23}},
        {"BTCUSDC": {"ob_mid_price": 77536.435, "spread_bps": 0.01, "ask_wall_px": 78000.0, "ask_wall_usd": 175985.0}},
    )
    assert btc_orders[0]["wall_queue_pos"] == "BEHIND_ASK_WALL"
    assert btc_orders[0]["recommended_front_run_px"] == 77995.0

    # Bug E: compute_enterprise_portfolio_risk_metrics separates free_cash_usd from locked_quote_usd
    ps_mock = {
        "tradable_positions": [
            {"venue": "binance", "symbol": "BTCUSDC", "value_usd": 10000.0, "locked_base_usd": 0.0},
        ],
        "venue_quote_cash": {
            "binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1000.0, "locked_quote_usd": 4000.0},
        },
        "recent_orders": {},
    }
    risk = compute_enterprise_portfolio_risk_metrics(ps_mock, {"BTCUSDC": {"price": 77500.0}}, {})
    assert risk["unified_nav_usd"] == 15000.0
    assert risk["global_free_cash_usd"] == 1000.0
    assert risk["global_free_cash_pct"] == 6.7
    assert risk["locked_quote_usd"] == 4000.0
    assert risk["locked_quote_pct"] == 26.7
    ps_mock["portfolio_risk_engine"] = risk
    port_lines = "\n".join(_format_portfolio_section(ps_mock, "crypto"))
    assert "free_cash=$1,000.00 (6.7%), locked_quote=$4,000.00 (26.7%)" in port_lines

    # Bug G: Sub-cent mid price and sub-0.5bps sibling delta formatting in Section 3A
    lines_3a = "\n".join(
        _format_market_prices_section(
            {
                "HYPEUSDC": {"price": 85.1165, "chg_15m_pct": 0.08},
                "HYPEUSD": {"price": 85.12, "chg_15m_pct": 0.08},
            }
        )
    )
    assert "HYPEUSDC (sibling venue: HYPEUSD=$85.12 [+0.4bps]): Price=$85.1165" in lines_3a


def test_macro_surprise_word_boundaries_fomc_gating_and_cross_venue_routing(tmp_path):
    import os
    from datetime import datetime, timezone
    from intelligence.macro_sentiment_ssot import (
        parse_live_macro_release_surprises,
        resolve_macro_economic_calendar,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import compute_deribit_options_surface
    from intelligence.sentiment.collectors.portfolio_collector import (
        _infer_symbol_venue_and_fees_bps,
        compute_enterprise_portfolio_risk_metrics,
        harvest_recent_orders_and_enrich_positions,
        sync_sibling_symbol_telemetry,
    )

    # 1. parse_live_macro_release_surprises ignores 'at 08:30' and 'flat 0.2%' while matching 'Est. 3.1%' / 'Prev. 3.2%'
    surprises = parse_live_macro_release_surprises(
        [
            "US CPI released at 08:30 ET: Actual 3.4% vs Est. 3.1%, Prev. 3.2%",
            "US PPI stays flat 0.2% vs Est. 0.3% in October report",
        ]
    )
    assert len(surprises) == 1
    assert surprises[0]["event"] == "US CPI"
    assert surprises[0]["actual"] == "3.4%"
    assert surprises[0]["consensus"] == "3.1%"
    assert surprises[0]["prior"] == "3.2%"
    assert surprises[0]["stance"] == "HAWKISH_HEADWIND"

    # 2. Non-FOMC day with CPI headline does not mislabel 21:00 EET window as FOMC Rate Decision
    # 2026-11-04 18:30 UTC = 20:30 EET (30m before 21:00 EET window, not an FOMC date)
    non_fomc_ts = datetime(2026, 11, 4, 18, 30, 0, tzinfo=timezone.utc).timestamp()
    cal = resolve_macro_economic_calendar(now_ts=non_fomc_ts, breaking_news=["US CPI inflation rises 3.4%"])
    assert "FOMC Rate Decision" not in (cal.get("next_event") or "")

    # 3. _infer_symbol_venue_and_fees_bps & sync_sibling_symbol_telemetry skip same-venue TAOUSDT and pair TAOUSDC with Kraken TAOUSD
    assert _infer_symbol_venue_and_fees_bps("TAOUSDT", {})[0] == "binance"
    assert _infer_symbol_venue_and_fees_bps("BTCEUR", {})[0] == "kraken"
    prices_map = {
        "TAOUSDC": {"price": 400.0},
        "TAOUSDT": {"price": 400.1},
        "TAOUSD": {"price": 402.0},
    }
    micro_map = {
        "TAOUSDC": {"ob_mid_price": 400.0, "spread_bps": 2.0},
        "TAOUSDT": {"ob_mid_price": 400.1, "spread_bps": 2.0},
        "TAOUSD": {"ob_mid_price": 402.0, "spread_bps": 4.0},
    }
    sync_sibling_symbol_telemetry(prices_map, micro_map)
    assert micro_map["TAOUSDC"]["cross_venue_peer"] == "TAOUSD"
    assert micro_map["TAOUSDC"]["best_buy_venue"] == "binance:TAOUSDC"
    assert micro_map["TAOUSDC"]["best_sell_venue"] == "kraken:TAOUSD"

    # 4. cyc_flat clears swing_subcycle_avg_px, pnl_vs_swing_avg_pct, and open_lots
    pos = {
        "venue": "kraken",
        "symbol": "TAOUSD",
        "quote": "USD",
        "free_base": 0.0,
        "free_quote": 500.0,
        "price": 225.0,
        "exchange_min_notional_usd": 10.0,
    }
    krk_cycles = {"TAOUSD": {"qty": 0.0, "cost_usd": 0.0, "avg_cost_px": None}}
    fake_files = {
        "cache_trade_kraken.json": {
            "items": {
                "TAOUSD": [
                    {"side": "BUY", "price": 240.0, "qty": 2.0, "executedQty": 2.0, "timestamp": 1000, "status": "FILLED"},
                ]
            }
        }
    }
    ro_out = harvest_recent_orders_and_enrich_positions(
        str(tmp_path),
        {},
        [pos],
        {"TAOUSD": {"price": 225.0}},
        krk_cycles,
        {},
        {},
        read_json_fn=lambda p: fake_files.get(os.path.basename(p), {}),
    )
    assert "swing_subcycle_avg_px" not in pos
    assert "pnl_vs_swing_avg_pct" not in pos
    assert "open_lots" not in pos
    assert "swing_subcycle_avg_px" not in ro_out.get("TAOUSD", {})
    assert "open_lots" not in ro_out.get("TAOUSD", {})

    # 5. compute_enterprise_portfolio_risk_metrics computes clean risk metrics without noisy fee-drag/markout
    risk = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": [{"venue": "kraken", "symbol": "BTCUSD", "value_usd": 500.0, "free_quote": 500.0, "quote": "USD"}],
            "venue_quote_cash": {"kraken:USD": {"venue": "kraken", "free": 500.0}},
            "recent_orders": {"BTCUSD": {"lifo_realized_pnl_usd": 12.5, "last_buy": {"price": 100000.0, "qty": 0.005}}},
        },
        {"BTCUSD": {"price": 101000.0, "vol_1h_pct": 1.0}},
        {"BTCUSD": {"spread_bps": 2.0}},
    )
    assert risk["unified_nav_usd"] == 1000.0
    assert risk["subcycle_lifo_realized_pnl_usd"] == 12.5
    assert "avg_entry_markout_bps" not in risk
    assert "est_recent_fee_drag_usd" not in risk

    # 6. compute_deribit_options_surface ignores non-C/P instruments and excludes bogus gamma pin / dealer gamma fields
    surf = compute_deribit_options_surface(
        [
            {"instrument_name": "BTC-28NOV26-90000-C", "open_interest": 100.0, "mark_iv": 52.0, "underlying_price": 90000.0},
            {"instrument_name": "BTC-28NOV26-90000-P", "open_interest": 80.0, "mark_iv": 54.0, "underlying_price": 90000.0},
            {"instrument_name": "BTC-28NOV26-91000-F", "open_interest": 9999.0, "mark_iv": 0.0, "underlying_price": 90000.0},
        ],
        btc_dvol=50.0,
        btc_rv=45.0,
    )
    assert surf["btc_pc_oi_ratio"] == 0.8
    assert "btc_gamma_pin_strike" not in surf
    assert "dealer_gamma_regime" not in surf


def test_enrich_cross_venue_trends_preserves_7d_range_over_short_watcher_history():
    """Verify newly initialized <5d watcher_price_items do not overwrite 7d OHLC range metrics."""
    from intelligence.sentiment.collectors.portfolio_collector import enrich_cross_venue_trends

    now_ms = 1_780_000_000_000
    prices_and_trends = {
        "ADAUSD": {
            "price": 0.1912,
            "chg_7d_pct": -10.02,
            "pos_7d_pct": 19.8,
            "high_7d": 0.214343,
            "low_7d": 0.185479,
            "swings_7d": ["0.214343@-140h(H)", "0.185479@-48h(L)"],
        }
    }
    short_watcher_series = [
        [now_ms - 1800_000 + i * 15_000, 0.1900 + (i % 5) * 0.0003]
        for i in range(120)
    ]
    enrich_cross_venue_trends(
        prices_and_trends=prices_and_trends,
        trends_data={},
        kalman_data={},
        long_items={},
        priceanalysis_data={},
        t_trend_map={},
        watcher_price_items={"ADAUSD": short_watcher_series},
    )
    assert prices_and_trends["ADAUSD"]["chg_7d_pct"] == -10.02
    assert prices_and_trends["ADAUSD"]["high_7d"] == 0.214343
    assert prices_and_trends["ADAUSD"]["low_7d"] == 0.185479
    assert len(prices_and_trends["ADAUSD"]["swings_7d"]) == 2


def test_subcycle_lifo_pnl_reset_open_order_deep_passive_and_cycle_wallet_sync(monkeypatch):
    """Verify 72h LIFO PnL reset, DEEP_PASSIVE_BID/FAR_ABOVE_ASK_WALL TCA, Hyperliquid deep L2 book, and EET clock."""
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
    from intelligence.sentiment.collectors.microstructure_collector import (
        compute_liquidation_clusters,
        extract_symbol_recent_fills,
    )
    from intelligence.sentiment.collectors.portfolio_collector import (
        enrich_open_limit_orders_tca,
        harvest_recent_orders_and_enrich_positions,
    )
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import _format_tradable_position_line

    # 1. Operational clock header uses Europe/Bucharest (EET / EEST)
    prompt_txt = build_supervisor_prompt(sensors=object(), logs=[])
    clock_line = next(l for l in prompt_txt.splitlines() if l.startswith("### 0. RUN TIMESTAMP & OPERATIONAL CLOCK:"))
    assert clock_line.endswith((" EET", " EEST"))

    # 2. extract_symbol_recent_fills resets LIFO PnL across flattened cycles & scopes to 72h swing window
    now_ms = 1_780_000_000_000
    day_ms = 86_400_000
    fills = extract_symbol_recent_fills(
        [
            # Closed historical cycle 10 days ago (+$500 profit) — must NOT leak into active sub-cycle LIFO PnL
            {"orderId": "1", "side": "BUY", "price": 100.0, "executedQty": 5.0, "status": "FILLED", "time": now_ms - 10 * day_ms},
            {"orderId": "2", "side": "SELL", "price": 200.0, "executedQty": 5.0, "status": "FILLED", "time": now_ms - 9 * day_ms},
            # Active cycle: core lot 5 days ago + tactical swing lot 2h ago sold 1h ago (+$12 profit)
            {"orderId": "3", "side": "BUY", "price": 250.0, "executedQty": 2.0, "status": "FILLED", "time": now_ms - 5 * day_ms},
            {"orderId": "4", "side": "BUY", "price": 220.0, "executedQty": 1.0, "status": "FILLED", "time": now_ms - 7_200_000},
            {"orderId": "5", "side": "SELL", "price": 232.0, "executedQty": 1.0, "status": "FILLED", "time": now_ms - 3_600_000},
        ],
        symbol="TAOUSDC",
    )
    assert fills is not None
    assert fills["lifo_realized_pnl_usd"] == 12.0
    assert len(fills["open_lots"]) == 1
    assert fills["open_lots"][0]["entry_px"] == 250.0

    # 3. enrich_open_limit_orders_tca distinguishes DEEP_PASSIVE_BID & FAR_ABOVE_ASK_WALL from near-wall orders
    open_orders = [
        {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 75000.0, "orig_qty": 0.01},
        {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 77400.0, "orig_qty": 0.01},
        {"venue": "binance", "symbol": "ARBUSDC", "side": "SELL", "price": 0.165, "orig_qty": 1000.0},
    ]
    enrich_open_limit_orders_tca(
        open_orders,
        prices_and_trends={"BTCUSDC": {"price": 77836.0, "vol_1h_pct": 0.8}, "ARBUSDC": {"price": 0.158, "vol_1h_pct": 1.2}},
        microstructure={
            "BTCUSDC": {"ob_mid_price": 77836.0, "best_bid": 77835.0, "bid_wall_px": 77500.0, "bid_wall_usd": 175000.0},
            "ARBUSDC": {"ob_mid_price": 0.158, "best_ask": 0.1581, "ask_wall_px": 0.1597, "ask_wall_usd": 98000.0},
        },
    )
    assert open_orders[0]["wall_queue_pos"] == "DEEP_PASSIVE_BID"
    assert "recommended_front_run_px" not in open_orders[0]
    assert open_orders[1]["wall_queue_pos"] == "BEHIND_BID_WALL"
    assert open_orders[1]["recommended_front_run_px"] > 77500.0
    assert open_orders[2]["wall_queue_pos"] == "FAR_ABOVE_ASK_WALL"
    assert "recommended_front_run_px" not in open_orders[2]

    # 4. _format_tradable_position_line deduplicates trail_peak and tactical_avg_px when equal to avg_buy_px / last_buy_px
    pos_line = _format_tradable_position_line(
        {
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "base": "HYPE",
            "quote": "USD",
            "free_base": 0.540276,
            "value_usd": 46.33,
            "free_quote": 0.34,
            "price": 85.75,
            "active_avg_buy_px": 91.209992,
            "pnl_vs_avg_pct": -5.99,
            "tactical_avg_buy_px": 87.267,
            "pnl_vs_tactical_avg_pct": -1.74,
            "last_buy_px": 87.267,
            "pnl_vs_last_buy_pct": -1.74,
            "trailing_peak": 91.21,
            "dd_from_peak_pct": -5.99,
        }
    )
    assert "avg_buy_px=91.209992" in pos_line
    assert "trail_peak=" not in pos_line
    assert "tactical_avg_px=" not in pos_line
    assert "last_buy_px=87.267" in pos_line

    # 5. harvest_recent_orders_and_enrich_positions syncs stale Hyperliquid cycle qty (free + locked) & avg_cost
    hl_cycles = {"HYPEUSDC": {"qty": 7.603274, "cost_usd": 660.43, "avg_cost_px": 86.86174, "pos_peak": 92.696}}
    tradable = [
        {
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "base": "HYPE",
            "quote": "USDC",
            "free_base": 9.826927,
            "locked_base_qty": 5.68,
            "price": 85.75,
        }
    ]
    harvest_recent_orders_and_enrich_positions(
        "/nonexistent_cache_dir",
        {},
        tradable,
        {"HYPEUSDC": {"price": 85.75}},
        {},
        hl_cycles,
        {},
    )
    assert hl_cycles["HYPEUSDC"]["qty"] == round(9.826927 + 5.68, 6)
    assert hl_cycles["HYPEUSDC"]["cost_usd"] == round(hl_cycles["HYPEUSDC"]["qty"] * 86.86174, 2)

    # 6. OrderbookDepthCollector extends shallow 20-tick Hyperliquid l2Book with nSigFigs=4 & selects non-touch walls
    ob_col = OrderbookDepthCollector(cache_ttl_sec=60.0, cache_dir=None)
    calls = []

    def _fake_fetch_json(url, label, lookup_sym, **kwargs):
        calls.append(url)
        if "nSigFigs=4" in url:
            return {
                "levels": [
                    [{"px": "85.70", "sz": "100.0"}, {"px": "85.20", "sz": "1500.0"}, {"px": "84.80", "sz": "800.0"}],
                    [{"px": "85.80", "sz": "100.0"}, {"px": "86.30", "sz": "1800.0"}, {"px": "86.70", "sz": "700.0"}],
                ]
            }
        return {
            "levels": [
                [{"px": "85.750", "sz": "500.0"}, {"px": "85.740", "sz": "100.0"}],
                [{"px": "85.760", "sz": "400.0"}, {"px": "85.770", "sz": "100.0"}],
            ]
        }

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_fetch_json)
    snap = ob_col.fetch("HYPEUSDC", allow_network=True, now=1_780_000_000.0)
    assert snap is not None
    assert any("nSigFigs=4" in u for u in calls)
    assert snap.best_bid == 85.75
    assert snap.best_ask == 85.76
    assert snap.largest_bid_wall_price == 85.20
    assert snap.largest_ask_wall_price == 86.30

    # 7. compute_liquidation_clusters blends near-spot walls (>=0.45%) and reads kalman_vel
    liq = compute_liquidation_clusters(
        {"price": 100.0, "vol_1h_pct": 1.0, "kalman_vel": 0.25, "chg_15m_pct": 0.4},
        {"oi_usd": 50_000_000.0, "whale_long_pct": 0.58, "ask_wall_px": 100.6, "bid_wall_px": 98.0, "taker_buy_sell_ratio": 1.25, "ob_imbalance": 0.62},
    )
    assert liq["short_liq_dist_pct"] < 1.35
    assert liq["liq_magnet_bias"] == "SHORT_SQUEEZE_MAGNET_UP"


def test_open_order_live_dist_recompute_bimodal_vah_and_range_ob_sync(tmp_path):
    """Verify live dist_from_spot_pct recomputation, venue tick-aligned front-run price, bimodal VAH > POC, and 24h/7d/6h-bottom ob_mid sync."""
    from intelligence.sentiment.collectors.microstructure_collector import (
        compute_volume_profile_metrics,
        load_symbol_microstructure,
    )
    from intelligence.sentiment.collectors.portfolio_collector import (
        enrich_open_limit_orders_tca,
        sync_sibling_symbol_telemetry,
    )

    # 1. enrich_open_limit_orders_tca overwrites stale cached dist_from_spot_pct and uses venue tick (0.01 on Kraken >=10)
    orders = [
        {
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "side": "BUY",
            "price": 84.0,
            "dist_from_spot_pct": -2.27,  # stale cached distance from old price 85.95
            "recommended_front_run_px": 84.9999,
        }
    ]
    enrich_open_limit_orders_tca(
        orders,
        {"HYPEUSD": {"price": 85.405, "vol_1h_pct": 0.55}},
        {"HYPEUSD": {"ob_mid_price": 85.405, "best_bid": 85.40, "best_ask": 85.41, "spread_bps": 1.17, "bid_wall_px": 84.53, "bid_wall_usd": 380000.0}},
    )
    assert orders[0]["dist_from_spot_pct"] == -1.65
    assert orders[0]["wall_queue_pos"] == "BEHIND_BID_WALL"
    assert orders[0]["recommended_front_run_px"] == 84.54

    # 2. compute_volume_profile_metrics swaps poc_idx with hvn2_idx on bimodal profile so VAH > POC (never clamped VAH == POC)
    bimodal_series = []
    t0 = 1_780_000_000_000
    for i in range(40):
        bimodal_series.append([t0 + i * 60000, 76840.0 + (i % 3) * 5.0, 100.0])
    for i in range(38):
        bimodal_series.append([t0 + (40 + i) * 60000, 77590.0 + (i % 3) * 5.0, 95.0])
    bimodal_series.append([t0 + 80 * 60000, 76285.0, 10.0])
    bimodal_series.append([t0 + 81 * 60000, 77990.0, 10.0])
    bimodal_series.append([t0 + 82 * 60000, 77830.0, 20.0])
    vp = compute_volume_profile_metrics(bimodal_series)
    assert vp["poc_px"] > 77500.0
    assert vp["vah_px"] > vp["poc_px"]
    assert vp["val_px"] < vp["poc_px"]

    # 3. sync_sibling_symbol_telemetry reconciles 24h_range, 7d_range, swing_wave, and recent_1h_low_px with updated price
    prices_and_trends = {
        "HYPEUSDC": {
            "price": 85.40,
            "low_24h": 82.51,
            "high_24h": 87.69,
            "pos_24h_pct": 66.8,
            "dd_24h_high_pct": -1.96,
            "bounce_24h_low_pct": 4.19,
            "low_7d": 78.16,
            "high_7d": 96.81,
            "pos_7d_pct": 42.0,
            "swing_low_px": 82.69,
            "swing_high_px": 87.41,
            "rally_from_swing_low_pct": 3.97,
            "pullback_from_swing_high_pct": -1.65,
            "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
            "prior_6h_low_px": 84.89,
            "recent_1h_low_px": 85.6315,
            "higher_low_vs_prior_pct": 0.87,
        }
    }
    sync_sibling_symbol_telemetry(prices_and_trends, {})
    hp = prices_and_trends["HYPEUSDC"]
    assert hp["pos_24h_pct"] == 55.8
    assert hp["dd_24h_high_pct"] == -2.61
    assert hp["recent_1h_low_px"] == 85.4
    assert hp["higher_low_vs_prior_pct"] == 0.6

    # 4. load_symbol_microstructure rounds oi_usd and taker_buy/sell_vol_usd to 2 decimals
    m_info = load_symbol_microstructure(
        str(tmp_path),
        "ARBUSDC",
        deriv={"funding_rate": 0.0001, "open_interest_usd": 35837274.643851936},
        whale={"taker_buy_vol_usd": 1267761.0736800001, "taker_sell_vol_usd": 1046782.1292599998},
    )
    assert m_info["oi_usd"] == 35837274.64
    assert m_info["taker_buy_vol_usd"] == 1267761.07
    assert m_info["taker_sell_vol_usd"] == 1046782.13


    # 8. Fee tuple unpacking, 7d>=24h range invariant, VA regime vs va_pos, and local L2 spot_cvd isolation
    from intelligence.sentiment.collectors.portfolio_collector import (
        _infer_symbol_venue_and_fees_bps,
        sync_sibling_symbol_telemetry,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import _build_vp_position_dict

    venue, maker_bps, taker_bps, half_sp = _infer_symbol_venue_and_fees_bps("HYPEUSD", {"spread_bps": 4.0})
    assert venue == "kraken"
    assert isinstance(maker_bps, float) and maker_bps == 16.0
    assert isinstance(taker_bps, float) and taker_bps == 26.0
    assert half_sp == 2.0

    vp_pos = _build_vp_position_dict(85.8225, 85.839, 83.7566, 86.9859)
    assert vp_pos["va_pos_pct"] == 64.0
    assert vp_pos["va_regime"] == "INSIDE_VA_UPPER"

    p_trends = {
        "HYPEUSDC": {"price": 85.82, "low_24h": 83.10, "high_24h": 87.50, "low_7d": 83.50, "high_7d": 87.20, "chg_15m_pct": 0.4},
        "HYPEUSD": {"price": 85.80, "low_24h": 83.10, "high_24h": 87.50, "low_7d": 83.50, "high_7d": 87.20, "chg_15m_pct": 0.4},
    }
    m_map = {
        "HYPEUSDC": {
            "bid_depth_usd": 480000.0,
            "ask_depth_usd": 480000.0,
            "spot_cvd_15m_usd": 342100.0,
            "cvd_15m_usd": 120000.0,
            "oi_usd": 50000000.0,
        },
        "HYPEUSD": {
            "bid_depth_usd": 5800.0,
            "ask_depth_usd": 6800.0,
        },
    }
    sync_sibling_symbol_telemetry(p_trends, m_map)
    assert p_trends["HYPEUSDC"]["low_7d"] <= p_trends["HYPEUSDC"]["low_24h"]
    assert p_trends["HYPEUSDC"]["high_7d"] >= p_trends["HYPEUSDC"]["high_24h"]
    assert abs(m_map["HYPEUSD"]["spot_cvd_15m_usd"]) <= (5800.0 + 6800.0) * 0.85

    # 9. Prompt anti-pollution: cold-start Kalman override, multi-horizon selloff spot_cvd guard, diplomatic news severity, parenthesized advisor ratio
    from intelligence.macro.lexicon_ssot import filter_valid_severity_keywords
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import enrich_sentiment_advisor_live_flow_divergence
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_microstructure_cvd
    from intelligence.sentiment.evaluators.price_series_evaluator import _apply_24h_series_enrichment

    cold_entry = {"price": 77800.0, "kalman_vel": 0.0, "kalman_vel_std": 6.0, "kalman_trend": "FLAT"}
    dense_ser = [[(1700000000 + i * 30) * 1000, 77000.0 + i * 8.0] for i in range(100)]
    _apply_24h_series_enrichment("BTCUSDC", cold_entry, {}, dense_ser, None)
    assert cold_entry["kalman_vel"] != 0.0

    selloff_m = {
        "bid_depth_usd": 105000.0,
        "ask_depth_usd": 60000.0,
        "taker_buy_vol_usd": 1925000.0,
        "taker_sell_vol_usd": 2015000.0,
        "cvd_15m_usd": -90000.0,
        "cvd_5m_usd": -645000.0,
        "cvd_1h_usd": -1520000.0,
        "funding_rate": 0.0000125,
    }
    enrich_microstructure_cvd(selloff_m, {"price": 85.64, "chg_15m_pct": -0.05})
    assert selloff_m["spot_perp_flow_regime"] == "BALANCED_CROSS_VENUE_FLOW"

    assert list(
        filter_valid_severity_keywords(
            ["war", "iran war"], title="Oil prices drop on fresh signs of US-Iran diplomatic push"
        )
    ) == []
    assert list(
        filter_valid_severity_keywords(
            ["war", "missile"], title="Oil jumps after missile strike in Middle East war"
        )
    ) == ["war", "missile"]

    sa_doc = {
        "age_min": 53.3,
        "recommended_action": "HOLD",
        "market_bias": "NEUTRAL",
        "summary": "Orderly de-risking alongside negative 1h CVD (-$5.87M) and weak taker buy/sell ratio (0.5627).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_doc, {"BTCUSDC": {"taker_buy_sell_ratio": 1.1172, "cvd_15m_usd": 1480000.0}}
    )
    assert "LIVE_FLOW_RECOVERY" in sa_doc.get("live_flow_divergence", "")
    assert "advisor ratio=0.56" in sa_doc.get("live_flow_divergence", "")

    sa_doc_cooled = {
        "age_min": 32.0,
        "recommended_action": "HOLD",
        "market_bias": "BULLISH",
        "summary": "Near-term taker flow has cooled (0.8001) amid low-volume Asian overnight trading.",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_doc_cooled, {"BTCUSDC": {"taker_buy_sell_ratio": 1.22, "cvd_15m_usd": 1800000.0}}
    )
    assert "LIVE_FLOW_RECOVERY" in sa_doc_cooled.get("live_flow_divergence", "")
    assert "advisor ratio=0.80" in sa_doc_cooled.get("live_flow_divergence", "")

    # 10. Near-spot wall TCA (2..10 bps), 5-digit tape_15m decimals, and AI intent fill synthesis
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_open_limit_orders_tca,
        extract_symbol_recent_fills,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import recalculate_live_price_geometry

    open_orders = [
        {"venue": "binance", "symbol": "BTCUSDC", "side": "SELL", "price": 78950.0, "qty": 0.01},
        {"venue": "binance", "symbol": "BTCUSDC", "side": "SELL", "price": 78950.0, "qty": 0.01},
    ]
    enrich_open_limit_orders_tca(
        open_orders[:1],
        {"BTCUSDC": {"price": 77812.35}},
        {"BTCUSDC": {"ask_wall_px": 77832.39, "ask_wall_usd": 144901.0, "ask_depth_usd": 160000.0}},
    )
    assert open_orders[0]["wall_queue_pos"] == "BEHIND_ASK_WALL"
    assert open_orders[0]["ask_wall_ahead_px"] == 77832.39
    assert "recommended_front_run_px" not in open_orders[0]

    enrich_open_limit_orders_tca(
        open_orders[1:],
        {"BTCUSDC": {"price": 77783.39}},
        {"BTCUSDC": {"ask_wall_px": 77862.06, "ask_wall_usd": 122179.0, "ask_depth_usd": 140000.0}},
    )
    assert open_orders[1]["wall_queue_pos"] == "BEHIND_ASK_WALL"
    assert "recommended_front_run_px" not in open_orders[1]

    from orchestratorOS.supervisor.prompt_sections import _fmt_px

    assert _fmt_px(85.6505) == "$85.6505"

    btc_geom = {"price": 77812.35, "recent_tape_15m": ["-15m:77750", "now:77800"]}
    recalculate_live_price_geometry(btc_geom, 77812.35)
    assert btc_geom["recent_tape_15m"][-1] == "now:77812.35"

    synth_fills = extract_symbol_recent_fills(
        [],
        symbol="NVDA_US_EQ",
        ai_intents=[
            {
                "symbol": "NVDA_US_EQ",
                "action": "BUY",
                "status": "EXECUTED",
                "exchange_status": "FILLED",
                "order_id": "ai-nvda-1",
                "executed_qty": 0.96,
                "executed_avg_price": 207.93,
                "executed_ts": 1700000000.0,
            }
        ],
    )
    assert synth_fills["last_buy"]["price"] == 207.93
    assert synth_fills["last_buy"]["actor"] == "AI"

    # 11. 1-minute 1h bar vol_1h_pct override, weak ob_imbalance spot_cvd guard, and resting order / xStock prompt deduplication
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section

    inflated_vol_entry = {"price": 227.2, "vol_1h_pct": 1.3634, "adapt_reentry_pct": 2.727, "adapt_dca_pct": 1.363, "slope_3_7m": 0.001, "slope_15m": 0.001, "slope_2_5h": -0.0001}
    one_hour_bars = [[(1700000000 + i * 60) * 1000, 227.0 + (i % 4) * 0.08] for i in range(60)]
    _apply_24h_series_enrichment("TAOUSDC", inflated_vol_entry, {}, one_hour_bars, None)
    assert inflated_vol_entry["vol_1h_pct"] < 0.8

    weak_tao_m = {
        "ob_imbalance": 0.545,
        "bid_depth_usd": 622811.0,
        "ask_depth_usd": 519042.0,
        "taker_buy_vol_usd": 75293.84,
        "taker_sell_vol_usd": 99147.6,
        "cvd_15m_usd": -23900.0,
        "cvd_5m_usd": 6200.0,
        "cvd_1h_usd": 44100.0,
        "funding_rate": 0.00005,
    }
    enrich_microstructure_cvd(weak_tao_m, {"price": 227.15, "chg_15m_pct": 0.04})
    assert weak_tao_m["spot_perp_flow_regime"] == "BALANCED_CROSS_VENUE_FLOW"

    sec_3c_lines = "\n".join(
        _format_portfolio_section(
            {
                "open_limit_orders": [{"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "order_id": "101", "price": 75000.0, "qty": 0.01, "filled_qty": 0.0, "status": "NEW"}],
                "last_cycle_execution_status": [
                    {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "status": "OPEN_RESTING", "order_id": "101", "age_min": 500.0},
                    {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "status": "FILLED", "order_id": "202", "fill_px": 227.0, "age_min": 5.0},
                ],
                "recent_ai_intents": [
                    {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "status": "EXECUTED", "venue_status": "NEW", "order_id": "101", "age_h": 8.3},
                    {"decision": "BUY", "symbol": "TAOUSDC", "venue": "binance", "status": "EXECUTED", "venue_status": "FILLED", "order_id": "202", "age_h": 0.1},
                ],
                "kraken_xstocks": {
                    "underlying": "SPCX",
                    "t212_sibling": "SPCX_US_EQ",
                    "holdings": {"AAPLx.T": 2e-06, "SPCXx.T": 15.74655},
                    "listed_pair": None,
                    "t212_sibling_cycle": {"qty": 7.04, "dd_pct": -2.54, "pnl_vs_avg_pct": -2.54, "last_exit_kind": None},
                },
            },
            asset_scope="crypto",
            include_macro_benchmarks=False,
        )
    )
    assert "AAPLx.T" not in sec_3c_lines
    assert "pnl_vs_avg_pct" not in sec_3c_lines
    assert "'age_min': 500.0" not in sec_3c_lines
    assert "'order_id': '202'" in sec_3c_lines

    # 12. Short-series Kalman warm-seed alignment and prompt fallback RESTING_IN_BOOK normalization
    import os
    import shadow_signals as ss

    tracker = ss.KalmanStateTracker(journal=ss.KalmanJournal(fixed_path=os.devnull), compute_vol=False)
    # 122 samples at 1s with KALMAN_SAMPLE_SEC=60: last sample is at ts=1000, prior 60s steps at idx 61 and idx 1
    big_p = [100.0 + i * 0.1 for i in range(121)] + [100.0 + 120 * 0.1]  # zero 1s tick between idx 120 and 121
    k_fields = tracker.update("TESTUSDC", 1000.0, big_p[-1], 0.05, big_p, 1.0)
    assert k_fields["kalman_vel"] > 0.0

    sec_3c_fallback = "\n".join(
        _format_portfolio_section(
            {
                "open_orders": [{"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "order_id": "101", "price": 75000.0, "qty": 0.01}],
                "recent_ai_intents": [
                    {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "status": "EXECUTED", "venue_status": "NEW", "order_id": "101", "age_h": 8.3},
                ],
            },
            asset_scope="crypto",
            include_macro_benchmarks=False,
        )
    )
    assert "RESTING_IN_BOOK" in sec_3c_fallback

    # 13. Sibling microstructure flow/liq reconciliation, squeeze-vs-dump guard, and T212 chart includePrePost
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        reconcile_sibling_flow_and_liq_regimes,
    )
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section

    contradictory_hype = {
        "ob_imbalance": 0.65,
        "bid_depth_usd": 500000.0,
        "ask_depth_usd": 250000.0,
        "taker_buy_vol_usd": 30000.0,
        "taker_sell_vol_usd": 95000.0,
        "cvd_15m_usd": -65000.0,
        "cvd_5m_usd": -25000.0,
        "cvd_1h_usd": -120000.0,
        "funding_rate": -0.0002,
    }
    enrich_microstructure_cvd(contradictory_hype, {"price": 34.4, "chg_15m_pct": -0.3})
    assert contradictory_hype["cvd_regime"] == "aggressive_market_dumps"
    assert contradictory_hype["spot_perp_flow_regime"] != "INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE"

    sib_micro = {
        "HYPEUSDC": {
            "ob_mid_price": 34.45,
            "ob_imbalance": 0.34,
            "bid_depth_usd": 146000.0,
            "ask_depth_usd": 286000.0,
            "spot_cvd_15m_usd": -183600.0,
            "cvd_15m_usd": -9600.0,
            "cvd_regime": "balanced_flow",
            "spot_perp_flow_regime": "RETAIL_PERP_LONG_TRAP",
            "liq_magnet_bias": "BALANCED_LIQ_MAGNET",
            "funding_rate": 0.00005,
        },
        "HYPEUSD": {
            "ob_mid_price": 34.44,
            "ob_imbalance": 0.67,
            "bid_depth_usd": 470000.0,
            "ask_depth_usd": 230000.0,
            "spot_cvd_15m_usd": 230800.0,
            "cvd_15m_usd": -9600.0,
            "cvd_regime": "aggressive_market_dumps",
            "spot_perp_flow_regime": "SYNCHRONIZED_SPOT_PERP_BUYING",
            "liq_magnet_bias": "LONG_SWEEP_MAGNET_DOWN",
            "funding_rate": 0.00005,
        },
    }
    sib_pt = {
        "HYPEUSDC": {"price": 34.45, "chg_15m_pct": 0.38},
        "HYPEUSD": {"price": 34.44, "chg_15m_pct": 0.40},
    }
    reconcile_sibling_flow_and_liq_regimes(sib_micro, sib_pt)
    assert sib_micro["HYPEUSDC"]["spot_perp_flow_regime"] == "BALANCED_CROSS_VENUE_FLOW"
    assert sib_micro["HYPEUSD"]["spot_perp_flow_regime"] == "BALANCED_CROSS_VENUE_FLOW"
    assert sib_micro["HYPEUSDC"]["liq_magnet_bias"] == sib_micro["HYPEUSD"]["liq_magnet_bias"]
    assert sib_micro["HYPEUSDC"]["cvd_regime"] == sib_micro["HYPEUSD"]["cvd_regime"]

    sec_3b_text = "\n".join(_format_microstructure_section(sib_micro, sib_pt))
    assert "RETAIL_PERP_LONG_TRAP" not in sec_3b_text
    assert "SYNCHRONIZED_SPOT_PERP_BUYING" not in sec_3b_text

    from botcore import load_local_module

    md_mod = load_local_module(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "212trading"),
        "market_data",
    )
    captured_urls: list[str] = []

    def _fake_http_get(url: str, **_kwargs):
        captured_urls.append(url)
        return (
            200,
            b'{"chart": {"result": [{"timestamp": [1700000000], "indicators": {"quote": [{"close": [207.61], "open": [207.5], "high": [207.7], "low": [207.4], "volume": [1000]}]}}]}}',
        )

    orig_http_get = md_mod.http_get
    try:
        md_mod.http_get = _fake_http_get
        _meta, bars = md_mod._chart("NVDA", "5d", "5m")
    finally:
        md_mod.http_get = orig_http_get
    assert len(bars) == 1 and bars[0][1] == 207.61
    assert captured_urls and "includePrePost=true" in captured_urls[0]





def test_open_lots_lowest_buy_px_geopolitical_deescalation_and_hl_nsigfigs3(monkeypatch):
    """Verify active open_lots lowest_buy_px, geopolitical zero-high-sev de-escalation, intra-clause taker flow regex, and HL nSigFigs=3 walls."""
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.macro.calendar_ssot import (
        compute_dynamic_geopolitical_interval,
        compute_dynamic_sentiment_interval,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        enrich_sentiment_advisor_live_flow_divergence,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
    )

    # 1. lowest_buy_px uses active open_lots (91.21) instead of closed-cycle recent_buys (75.55)
    pos_list = [{"venue": "kraken", "symbol": "HYPEUSD", "free_base": 0.54, "price": 85.8}]
    venue_ro = {
        "kraken": {
            "HYPEUSD": {
                "last_action_side": "SELL",
                "active_avg_buy_px": 91.209992,
                "last_buy": {"price": 91.209992, "qty": 4.9358},
                "last_sell": {"price": 87.120002, "qty": 2.2951},
                "recent_buys": [
                    {"price": 79.04, "qty": 0.295},
                    {"price": 77.41, "qty": 4.198},
                    {"price": 75.55, "qty": 1.382},
                    {"price": 91.209992, "qty": 4.9358},
                ],
                "open_lots": [{"lot_id": "L4", "qty": 0.54, "entry_px": 91.209992}],
            }
        }
    }
    enrich_positions_with_orders_and_cycles(
        pos_list,
        {"HYPEUSD": {"price": 85.8}},
        {},
        venue_ro,
        {},
        {"HYPEUSD": {"qty": 0.54, "avg_cost_px": 91.209992, "entry_price": 91.209992}},
        {},
        {},
        {},
    )
    assert pos_list[0]["lowest_buy_px"] == 91.209992

    # 2. Geopolitical de-escalation recheck when high_severity_count == 0 under ELEVATED
    geo_itv, geo_reason = compute_dynamic_geopolitical_interval(
        base_interval_sec=16200.0,
        last_eval_ts=1_700_000_000.0,
        now_ts=1_700_002_000.0,
        cached_assessment={"threat_level": "ELEVATED", "is_fallback": False},
        news_snapshot={"headlines": [{"title": "Routine earnings update", "is_high_severity": False}], "high_severity_count": 0},
        headlines_changed=False,
    )
    assert geo_reason == "threat_deescalation_zero_high_sev_recheck"
    assert geo_itv <= 3600.0

    # 3. Intra-clause taker flow phrase "(0.8732 ratio, 1h CVD -$1.73M)" extracts 0.8732
    sa_clause = {
        "market_bias": "NEUTRAL",
        "recommended_action": "HOLD",
        "age_min": 25.0,
        "summary": "24h breadth remains positive while short-term Asian session taker flow is mildly subdued (0.8732 ratio, 1h CVD -$1.73M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_clause,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.28, "cvd_15m_usd": 2_100_000.0}},
    )
    assert "LIVE_FLOW_RECOVERY" in sa_clause.get("live_flow_divergence", "")
    assert "advisor ratio=0.87" in sa_clause.get("live_flow_divergence", "")

    # Sudden BTC selloff under NEUTRAL / HOLD / risk=LOW triggers LIVE_SELL_REVERSAL
    sa_selloff_neutral = {
        "market_bias": "NEUTRAL",
        "risk_level": "LOW",
        "recommended_action": "HOLD",
        "age_min": 33.0,
        "summary": "24h breadth remains positive while short-term Asian session taker flow is mildly subdued (0.8732 ratio, 1h CVD -$1.73M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_selloff_neutral,
        {"BTCUSDC": {"taker_buy_sell_ratio": 0.5552, "cvd_15m_usd": -43_300_000.0, "cvd_1h_usd": -48_400_000.0}},
    )
    assert "LIVE_SELL_REVERSAL" in sa_selloff_neutral.get("live_flow_divergence", "")
    assert "NEUTRAL/HOLD (advisor ratio=0.87)" in sa_selloff_neutral.get("live_flow_divergence", "")

    # Symmetric bullish surge under NEUTRAL / HOLD triggers LIVE_BUY_REVERSAL
    sa_surge_neutral = {
        "market_bias": "NEUTRAL",
        "risk_level": "LOW",
        "recommended_action": "HOLD",
        "age_min": 33.0,
        "summary": "24h breadth remains positive while short-term Asian session taker flow is mildly subdued (0.8732 ratio, 1h CVD -$1.73M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_surge_neutral,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.42, "cvd_15m_usd": 18_500_000.0}},
    )
    assert "LIVE_BUY_REVERSAL" in sa_surge_neutral.get("live_flow_divergence", "")
    assert "NEUTRAL/HOLD (advisor ratio=0.87)" in sa_surge_neutral.get("live_flow_divergence", "")

    _, adv_reason = compute_dynamic_sentiment_interval(
        base_interval_sec=10800.0,
        last_eval_ts=1_700_000_000.0,
        now_ts=1_700_001_800.0,
        cached_assessment=sa_clause,
        btc_taker_ratio=1.30,
        prev_btc_taker_ratio=None,
    )
    assert adv_reason == "btc_whale_flow_regime_flip_0.87_to_1.30"

    _, adv_sell_reason = compute_dynamic_sentiment_interval(
        base_interval_sec=10800.0,
        last_eval_ts=1_700_000_000.0,
        now_ts=1_700_001_800.0,
        cached_assessment=sa_selloff_neutral,
        btc_taker_ratio=0.5552,
        prev_btc_taker_ratio=None,
    )
    assert adv_sell_reason == "btc_whale_flow_regime_flip_0.87_to_0.56"

    # 4. OrderbookDepthCollector extends HL l2Book to nSigFigs=3 when nSigFigs=4 is <1% deep & skips <10bps micro-ticks
    ob_col = OrderbookDepthCollector(cache_ttl_sec=60.0, cache_dir=None)
    hl_calls = []

    def _fake_hl_fetch(url, label, lookup_sym, **kwargs):
        hl_calls.append(url)
        if "nSigFigs=3" in url:
            return {
                "levels": [
                    [{"px": "85.50", "sz": "100.0"}, {"px": "85.10", "sz": "2500.0"}, {"px": "84.50", "sz": "900.0"}],
                    [{"px": "86.00", "sz": "100.0"}, {"px": "86.60", "sz": "2200.0"}, {"px": "87.00", "sz": "800.0"}],
                ]
            }
        if "nSigFigs=4" in url:
            return {
                "levels": [
                    [{"px": "85.77", "sz": "800.0"}, {"px": "85.60", "sz": "200.0"}],
                    [{"px": "85.81", "sz": "400.0"}, {"px": "85.95", "sz": "200.0"}],
                ]
            }
        return {
            "levels": [
                [{"px": "85.774", "sz": "500.0"}, {"px": "85.772", "sz": "900.0"}],
                [{"px": "85.775", "sz": "400.0"}, {"px": "85.808", "sz": "450.0"}],
            ]
        }

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_hl_fetch)
    snap_hl = ob_col.fetch("HYPEUSDC", allow_network=True, now=1_780_000_100.0)
    assert snap_hl is not None
    assert any("nSigFigs=4" in u for u in hl_calls)
    assert any("nSigFigs=3" in u for u in hl_calls)
    assert snap_hl.largest_bid_wall_price == 85.10
    assert snap_hl.largest_ask_wall_price == 86.60

    # 5. Per-symbol locked_quote_usd & replace_max_buy_usd isolation in Section 3C vs venue_quote_cash
    from types import SimpleNamespace
    from intelligence.sentiment.collectors.portfolio_collector import harvest_tradable_positions_and_orders
    from orchestratorOS.supervisor.prompt_sections import _format_tradable_position_line, _format_recent_order_line

    def _mk_binance_inst(sym, base, free_b, open_ords):
        return SimpleNamespace(
            symbol=sym,
            provider_name="binance",
            base=base,
            quote="USDC",
            params={},
            _provider=SimpleNamespace(
                get_current_price=lambda s: {"BTCUSDC": 75800.0, "TAOUSDC": 235.5, "ARBUSDC": 0.1235}.get(s, 0.0),
                free_balance=lambda a: free_b if a == base else 1288.04,
            ),
            open_orders=lambda: open_ords,
        )

    inst_map_bn = {
        "BINANCE_BTC": _mk_binance_inst(
            "BTCUSDC", "BTC", 0.05,
            [{"orderId": 1001, "symbol": "BTCUSDC", "side": "BUY", "type": "LIMIT", "price": "75000.0", "origQty": "0.01333", "executedQty": "0.0", "status": "NEW"}],
        ),
        "BINANCE_TAO": _mk_binance_inst(
            "TAOUSDC", "TAO", 4.28,
            [{"orderId": 721126902, "symbol": "TAOUSDC", "side": "BUY", "type": "LIMIT", "price": "220.0", "origQty": "2.2727", "executedQty": "0.0", "status": "NEW"}],
        ),
        "BINANCE_ARB": _mk_binance_inst(
            "ARBUSDC", "ARB", 2441.5,
            [{"orderId": 105396008, "symbol": "ARBUSDC", "side": "SELL", "type": "LIMIT", "price": "0.165", "origQty": "1220.7", "executedQty": "0.0", "status": "NEW"}],
        ),
    }
    pos_list, _, venue_cash, _ = harvest_tradable_positions_and_orders(
        inst_map_bn,
        {"BTCUSDC": {"price": 75800.0}, "TAOUSDC": {"price": 235.5}, "ARBUSDC": {"price": 0.1235}},
        {},
        {},
        gateway_max_usd=2000.0,
    )
    # Venue-level cash tracks total locked quote across all symbols on binance ($999.75 + $499.99 = $1499.74)
    assert venue_cash["binance:USDC"]["locked_quote_usd"] == 1499.74
    pos_by_sym = {p["symbol"]: p for p in pos_list}
    # ARBUSDC has 0 BUY open orders -> no symbol-level locked_quote_usd or replace_max_buy_usd inflation
    assert pos_by_sym["ARBUSDC"].get("locked_quote_usd", 0.0) == 0.0
    assert "replace_max_buy_usd" not in pos_by_sym["ARBUSDC"]
    # TAOUSDC only locks $499.99 in its own BUY order -> replace_max_buy_usd is (1288.04 + 499.99)*0.995 = 1779.09
    assert pos_by_sym["TAOUSDC"]["locked_quote_usd"] == 499.99
    assert pos_by_sym["TAOUSDC"]["replace_max_buy_usd"] == 1779.09
    # BTCUSDC locks $999.75 in its own BUY order -> replace_max_buy_usd is capped at max_order_usd ($2000.0)
    assert pos_by_sym["BTCUSDC"]["locked_quote_usd"] == 999.75
    assert pos_by_sym["BTCUSDC"]["replace_max_buy_usd"] == 2000.0

    # 6. _same_anchor_px preserves distinct multi-tranche tactical_avg_buy_px on BTCUSDC (1.5 bps diff vs last_buy_px) and locked_* is not duplicated in limits(...)
    btc_pos_line = _format_tradable_position_line(
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "base": "BTC",
            "quote": "USDC",
            "free_base": 0.05288,
            "free_quote": 1288.04,
            "locked_base_qty": 0.01,
            "locked_quote_usd": 999.75,
            "price": 75807.29,
            "value_usd": 4008.69,
            "active_avg_buy_px": 77959.914388,
            "tactical_avg_buy_px": 77590.206667,
            "swing_subcycle_avg_px": 77376.950854,
            "last_buy_px": 77578.56,
        }
    )
    assert "avg_buy_px=77959.914388" in btc_pos_line
    assert "tactical_avg_px=77590.206667" in btc_pos_line
    assert "swing_avg_px=77376.950854" in btc_pos_line
    assert "last_buy_px=77578.56" in btc_pos_line
    assert btc_pos_line.count("locked_base_qty=") == 1
    assert btc_pos_line.count("locked_quote_usd=") == 1

    btc_ord_line = _format_recent_order_line(
        "BTCUSDC",
        {
            "venue": "binance",
            "last_action_side": "BUY",
            "active_avg_buy_px": 77959.914388,
            "tactical_avg_buy_px": 77590.206667,
            "swing_subcycle_avg_px": 77376.950854,
            "last_buy": {"price": 77578.56, "qty": 0.00644, "time": "2026-10-30 22:08:52 UTC"},
        },
        1700000000000.0,
    )
    assert "Tactical Sub-Cycle Avg BUY=77590.206667" in btc_ord_line
    assert "Swing Sub-Cycle Avg BUY=77376.950854" in btc_ord_line

    # 7. T212 cycle last_buy_price/last_sell_price sync in crypto scope & 4-decimal _US_EQ 7d range rounding
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_cross_venue_trends
    from intelligence.sentiment.evaluators.portfolio_evaluator import reconcile_venue_cycles_with_recent_orders
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context

    t212_cycs = {
        "NVDA_US_EQ": {"qty": 0.03, "avg_cost_px": 208.0, "entry_price": 207.9271, "last_buy_price": 207.9271, "dd_pct": -1.15},
    }
    v_ro = {
        "t212": {
            "NVDA_US_EQ": {
                "last_buy": {"price": 207.93, "qty": 0.96, "ts_ms": 1793720000000},
                "last_sell": {"price": 207.9, "qty": 0.93, "ts_ms": 1793724706000},
                "recent_buys": [{"price": 207.93, "qty": 0.96, "ts_ms": 1793720000000}],
                "recent_sells": [{"price": 207.9, "qty": 0.93, "ts_ms": 1793724706000}],
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders({}, {}, t212_cycs, v_ro, {}, {"NVDA_US_EQ": {"price": 205.6}})
    assert t212_cycs["NVDA_US_EQ"]["last_buy_price"] == 207.93
    assert t212_cycs["NVDA_US_EQ"]["last_sell_price"] == 207.9
    cmc = build_cross_market_context({"NVDA_US_EQ": {"price": 205.6, "chg_15m_pct": -0.28}}, {}, t212_cycs)
    assert cmc["equity_bellwethers"]["NVDA_US_EQ"]["t212_cycle"]["last_buy_price"] == 207.93
    assert cmc["equity_bellwethers"]["NVDA_US_EQ"]["t212_cycle"]["last_sell_price"] == 207.9

    pt_eq = {"SPCX_US_EQ": {"price": 139.72, "trend": 0, "growth_coefficient": 0.0}}
    enrich_cross_venue_trends(
        pt_eq,
        {},
        {},
        {},
        {},
        {},
        {"SPCX": [[1793100000000, 148.5800018310547], [1793400000000, 134.6649932861328], [1793700000000, 141.31]]},
    )
    assert pt_eq["SPCX_US_EQ"]["low_7d"] == 134.665
    assert pt_eq["SPCX_US_EQ"]["high_7d"] == 148.58
    assert pt_eq["SPCX_US_EQ"]["chg_7d_pct"] == round(((141.31 - 148.5800018310547) / 148.5800018310547) * 100.0, 2)

    # 8. Post-DCA last_exit_kind remains REDUCE when a partial sell occurred below pre-DCA entry_price
    k_cycs_dca = {
        "HYPEUSD": {
            "qty": 6.492657,
            "cost_usd": 549.28,
            "avg_cost_px": 84.597119,
            "entry_price": 91.209992,
            "_entry_ts_ms": 1793690000000,
            "last_buy_price": 84.0,
            "last_sell_price": 93.299997,
            "last_exit_kind": "TP",
            "realized_pnl": -189.93,
            "dd_pct": 0.44,
        }
    }
    v_ro_dca = {
        "kraken": {
            "HYPEUSD": {
                "last_buy": {"price": 84.0, "qty": 5.95238095, "ts_ms": 1793752000000},
                "last_sell": {"price": 87.120002, "qty": 2.2951572, "ts_ms": 1793721000000},
                "recent_buys": [
                    {"price": 91.209992, "qty": 2.835433, "ts_ms": 1793690000000},
                    {"price": 84.0, "qty": 5.95238095, "ts_ms": 1793752000000},
                ],
                "recent_sells": [
                    {"price": 87.120002, "qty": 2.2951572, "ts_ms": 1793721000000},
                ],
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        k_cycs_dca,
        {},
        {},
        v_ro_dca,
        {("kraken", "HYPEUSD"): 6.492657},
        {"HYPEUSD": {"price": 84.97}},
    )
    assert k_cycs_dca["HYPEUSD"]["last_sell_price"] == 87.120002
    assert k_cycs_dca["HYPEUSD"]["last_exit_kind"] == "REDUCE"




def test_post_rth_equities_series_enrichment_short_slopes_scale_and_daemon_log_recovery(tmp_path):
    """Verify:
    - enrich_24h_and_microstructure seeds pos:t212:* positions (e.g. SPCX_US_EQ) from venue_balances_state.json even without a .state_*.json file
    - post-RTH US equity series (>20m since RTH close) do not append a multi-hour gap that collapses 15m/30m/1h/3h and resets kalman_vel, while recalculating live range/24h/7d geometry
    - _compute_short_slopes_from_series keeps growth_* on the same scale as slope_* instead of inflating by 150,000x
    - harvest_log_anomalies_from_dirs deduplicates unstamped command stdout echoed in ERROR lines and clears resolved daemon errors after 'Daemon status: active'
    """
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from intelligence.sentiment.evaluators.price_series_evaluator import _compute_short_slopes_from_series
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # 1. Post-RTH SPCX_US_EQ seeded from venue_balances_state.json + 24h RTH series ending 3 hours ago
    cache_dir = tmp_path / "cachedb"
    prices_dir = cache_dir / "prices"
    state_dir = cache_dir / "state"
    for d in (prices_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    now_ts = 1_793_751_600.0
    rth_close_ms = int((now_ts - 3.0 * 3600.0) * 1000)
    rth_bars = []
    for idx in range(78):
        t_ms = rth_close_ms - (77 - idx) * 300_000
        px = round(140.0 + 0.02 * idx + (0.15 if idx % 3 == 0 else -0.05), 4)
        rth_bars.append([t_ms, px])
    # Force distinct 15m, 30m, 1h, 3h bars before RTH close (last bar = 141.31)
    rth_bars[-1][1] = 141.31
    rth_bars[-4][1] = 141.60   # 15m before close
    rth_bars[-7][1] = 141.95   # 30m before close
    rth_bars[-13][1] = 141.81  # 1h before close
    rth_bars[-37][1] = 140.90  # 3h before close

    (prices_dir / "cache_24_price_SPCX_US_EQ.json").write_text(
        json.dumps({"items": {"SPCX_US_EQ": rth_bars}}),
        encoding="utf-8",
    )
    (state_dir / "venue_balances_state.json").write_text(
        json.dumps({"pos:t212:SPCX_US_EQ": {"symbol": "SPCX_US_EQ", "free": 12.14, "price": 141.20}}),
        encoding="utf-8",
    )

    pt_map = {"SPCX_US_EQ": {"price": 141.31, "trend": 0, "growth_coefficient": 0.0}}
    enrich_24h_and_microstructure(str(cache_dir), str(tmp_path), pt_map, {}, {}, {}, now_ts)
    spcx = pt_map["SPCX_US_EQ"]
    assert abs(spcx["price"] - 141.20) < 1e-6
    # 15m, 30m, 1h, 3h must reflect distinct RTH session horizons rather than collapsing to identical values
    assert len({spcx["chg_15m_pct"], spcx["chg_30m_pct"], spcx["chg_1h_pct"], spcx["chg_3h_pct"]}) >= 3
    assert abs(spcx["kalman_vel"]) > 0.0
    assert "recent_tape_15m" not in spcx

    # 2. _compute_short_slopes_from_series keeps growth_2_5h on the same scale as slope_2_5h
    slopes = _compute_short_slopes_from_series(rth_bars)
    assert "slope_2_5h" in slopes and "growth_2_5h" in slopes
    assert abs(slopes["growth_2_5h"]) < 0.05

    # 3. harvest_log_anomalies_from_dirs deduplicates unstamped echo and suppresses recovered manage_agy daemon errors
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    now_s = time.time()
    s_fmt = lambda ago: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s - ago))  # noqa: E731
    agy_log = log_dir / "manage_agy.log"
    agy_log.write_text(
        f"[{s_fmt(1800)}] [manage_agy] WARNING: Daemon inactive/dead. Attempting automatic recovery...\n"
        "Remote-control daemon stopped.\n"
        "  Error: not signed in: run `agy`, sign in, then run `agy remote-control start`\n"
        f"[{s_fmt(1797)}] [manage_agy] ERROR: Failed to recover daemon. Output: Remote-control daemon stopped.   Error: not signed in: run `agy`, sign in, then run `agy remote-control start`\n"
        f"[{s_fmt(600)}] [manage_agy] Remote-control daemon started.\n"
        f"[{s_fmt(600)}] [manage_agy] Daemon status: active\n",
        encoding="utf-8",
    )
    summary_recovered = harvest_log_anomalies_from_dirs([str(log_dir)], lookback_hours=1.0)
    assert summary_recovered.error_count == 0
    assert summary_recovered.persistent_errors == []

    # When the failure is at the very tail without subsequent recovery, it is counted once (not double-counted)
    agy_log.write_text(
        f"[{s_fmt(60)}] [manage_agy] WARNING: Daemon inactive/dead. Attempting automatic recovery...\n"
        "  Error: not signed in: run `agy`, sign in, then run `agy remote-control start`\n"
        f"[{s_fmt(58)}] [manage_agy] ERROR: Failed to recover daemon. Output: Error: not signed in: run `agy`, sign in, then run `agy remote-control start`\n",
        encoding="utf-8",
    )
    summary_unrecovered = harvest_log_anomalies_from_dirs([str(log_dir)], lookback_hours=1.0)
    assert summary_unrecovered.error_count == 1
    assert len(summary_unrecovered.persistent_errors) == 1
    assert summary_unrecovered.persistent_errors[0]["unresolved_at_tail"] is True


def test_sentiment_advisor_subdomain_cache_dir_pretrade_walls_fills_and_traceback_harvesting(tmp_path):
    """Verify:
    - Bug 8: SentimentAdvisor._effective_cache_dir strips CACHEDB_SUBDOMAINS ('macro') and resolves prices/cache_24_price_BTCUSDC.json so BTC Intraday Structure is included in prompt
    - Bug 9: populate_market_and_portfolio_snapshot enriches sentiment_advisor live_flow_divergence before scope='equities' clears snapshot.microstructure
    - Bug 10: compute_dynamic_sentiment_interval extracts fallback prev_fg_val and prev_adv from cached_assessment summary/key_risks after daemon restart
    - Bug 11: collect_pretrade_telemetry scales orderbook wall threshold by total depth, prefers same-quote-family venue fills (HYPEUSDC vs HYPEUSD), and formats missing last_sell as 'none'
    - Bug 12: harvest_log_anomalies_from_dirs captures Python terminal exception lines (RuntimeError:, TimeoutError:, ReadTimeout:) while skipping interior traceback boilerplate
    """
    from types import SimpleNamespace
    from intelligence.macro.calendar_ssot import compute_dynamic_sentiment_interval
    from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        apply_asset_scope_filter,
        enrich_sentiment_advisor_live_flow_divergence,
    )
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # 1. Bug 8: SentimentAdvisor with cache_file inside cachedb/macro/ finds cachedb/prices/cache_24_price_BTCUSDC.json
    c_root = tmp_path / "cachedb"
    (c_root / "macro").mkdir(parents=True, exist_ok=True)
    (c_root / "prices").mkdir(parents=True, exist_ok=True)
    now_ms = int(1_793_751_600 * 1000)
    btc_pts = [[now_ms - (30 - i) * 300_000, 76000.0 + i * 25.0] for i in range(31)]
    (c_root / "prices" / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": btc_pts}}),
        encoding="utf-8",
    )
    captured_prompts = []

    class _DummyGemini:
        def query_json(self, prompt, **kw):
            captured_prompts.append(prompt)
            return {
                "market_bias": "NEUTRAL",
                "risk_level": "MODERATE",
                "confidence": 0.7,
                "summary": "Balanced market.",
                "key_risks": ["Macro"],
                "recommended_action": "HOLD",
            }

    adv = SentimentAdvisor(
        gemini_client=_DummyGemini(),
        cache_file=str(c_root / "macro" / "sentiment_eval.json"),
    )
    assert adv._effective_cache_dir() == str(c_root)
    adv.review(force_refresh=True, now_ts=1_793_751_600.0)
    assert captured_prompts and "- BTC Intraday Structure:" in captured_prompts[0]

    # 2. Bug 9: enrich_sentiment_advisor_live_flow_divergence runs before equities scope clears microstructure
    sa_dict = {
        "market_bias": "BEARISH",
        "recommended_action": "DEFENSIVE",
        "age_min": 40.0,
        "summary": "Seller dominance with weak taker buy/sell ratio (0.5627).",
    }
    micro = {"BTCUSDC": {"taker_buy_sell_ratio": 1.15, "cvd_15m_usd": 1_500_000.0}}
    enrich_sentiment_advisor_live_flow_divergence(sa_dict, micro)
    _, filtered_micro = apply_asset_scope_filter({}, micro, {}, "equities", {}, {})
    assert filtered_micro == {}
    assert "live_flow_divergence" in sa_dict
    assert "LIVE_FLOW_RECOVERY" in sa_dict["live_flow_divergence"]

    # 3. Bug 10: compute_dynamic_sentiment_interval fallback extraction of prev_fg_val and prev_adv from cached_assessment
    cached_assess = SimpleNamespace(
        risk_level="MODERATE",
        market_bias="NEUTRAL",
        recommended_action="HOLD",
        summary="24h breadth shows 80.9% advancing whileperp flow cools.",
        key_risks=["Sticky macro fear (F&G at 26) despite relief."],
    )
    iv_fg, reason_fg = compute_dynamic_sentiment_interval(
        base_interval_sec=3600.0,
        last_eval_ts=1000.0,
        now_ts=2000.0,
        cached_assessment=cached_assess,
        fear_greed=SimpleNamespace(value=35, is_extreme_fear=False, is_extreme_greed=False),
        breadth=SimpleNamespace(advance_ratio=0.75, median_change_pct=1.0, regime="BULLISH"),
        prev_fear_greed_val=None,
        prev_advance_ratio=None,
    )
    assert reason_fg == "fear_greed_jump_+9pts"
    assert iv_fg <= 900.0

    iv_br, reason_br = compute_dynamic_sentiment_interval(
        base_interval_sec=3600.0,
        last_eval_ts=1000.0,
        now_ts=2000.0,
        cached_assessment=cached_assess,
        fear_greed=SimpleNamespace(value=28, is_extreme_fear=False, is_extreme_greed=False),
        breadth=SimpleNamespace(advance_ratio=0.45, median_change_pct=-0.5, regime="NEUTRAL"),
        prev_fear_greed_val=None,
        prev_advance_ratio=None,
    )
    assert reason_br == "breadth_shift_-0.36"
    assert iv_br <= 900.0

    # 4. Bug 11: load_symbol_best_recent_fills prefers same-quote-family venue fills (HYPEUSDC -> Hyperliquid over Kraken HYPEUSD)
    orders_dir = c_root / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    (orders_dir / "cache_trade_kraken.json").write_text(
        json.dumps({
            "items": {
                "HYPEUSD": [
                    {"order_id": "K1", "side": "buy", "status": "closed", "executed_qty": 5.95, "price": 84.0, "update_time": 2000}
                ]
            }
        }),
        encoding="utf-8",
    )
    (orders_dir / "cache_trade_hyperliquid.json").write_text(
        json.dumps({
            "items": {
                "HYPE": [
                    {"order_id": "HL1", "side": "buy", "status": "filled", "executed_qty": 1.64, "price": 87.267, "update_time": 1500}
                ]
            }
        }),
        encoding="utf-8",
    )
    hl_fills = load_symbol_best_recent_fills(str(c_root), "HYPEUSDC", ["HYPEUSDC", "HYPEUSDT", "HYPEUSD", "HYPE"])
    kr_fills = load_symbol_best_recent_fills(str(c_root), "HYPEUSD", ["HYPEUSD", "HYPEUSDC", "HYPEUSDT", "HYPE"])
    assert hl_fills is not None and abs(float(hl_fills["last_buy"]["price"]) - 87.267) < 1e-4
    assert kr_fills is not None and abs(float(kr_fills["last_buy"]["price"]) - 84.0) < 1e-4

    # 5. Bug 12: harvest_log_anomalies_from_dirs captures Python terminal exception lines and skips traceback boilerplate
    tb_dir = tmp_path / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    now_s = time.time()
    clk = time.strftime("%H:%M:%S", time.localtime(now_s - 30))
    (tb_dir / "HL-bot.log").write_text(
        f"[{clk}]   [STRAT] state loaded\n"
        "Traceback (most recent call last):\n"
        '  File "/app/urllib3/util/retry.py", line 474, in increment\n'
        "    raise reraise(type(error), error, _stacktrace)\n"
        "urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='api.hyperliquid.xyz', port=443): Read timed out.\n\n"
        "The above exception was the direct cause of the following exception:\n\n"
        "Traceback (most recent call last):\n"
        '  File "/app/strategies/spot_engine.py", line 402, in __init__\n'
        '    raise RuntimeError(f"Cannot start {pair}: venue precision is unavailable") from exc\n'
        "RuntimeError: Cannot start HYPE: venue precision is unavailable\n",
        encoding="utf-8",
    )
    tb_summary = harvest_log_anomalies_from_dirs([str(tb_dir)], lookback_hours=1.0)
    samples = [pe["sample"] for pe in tb_summary.persistent_errors]
    assert any("RuntimeError: Cannot start HYPE: venue precision is unavailable" in s for s in samples)
    assert any("ReadTimeoutError:" in s for s in tb_summary.critical_events)
    assert not any(s == "Traceback (most recent call last):" for s in tb_summary.critical_events)
    assert not any("The above exception was the direct cause" in s for s in tb_summary.critical_events)
    assert not any("raise reraise" in s for s in tb_summary.critical_events)

    # 6. Bug 13: parse_venue_cycle_state sets last_exit_kind="REDUCE" when last_exit_kind is None and sell_p < avg_cost_px * 0.995
    from intelligence.sentiment.evaluators.portfolio_evaluator import parse_venue_cycle_state
    spcx_cyc = parse_venue_cycle_state(
        {
            "qty": 13.71,
            "cost_usd": 2033.24,
            "last_sell_price": 138.9,
            "last_exit_kind": None,
            "realized_pnl": -115.65,
            "entry_ts": 1700000000000,
        },
        "SPCX_US_EQ",
        {},
    )
    assert spcx_cyc["last_exit_kind"] == "REDUCE"

    # 7. Bug 14: enrich_24h_and_microstructure recenters stale ob_mid_price and walls on multi-venue USD pair (HYPEUSD)
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    ob_dir = tmp_path / "ob_recenter"
    (ob_dir / "telemetry").mkdir(parents=True, exist_ok=True)
    (ob_dir / "telemetry" / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps(
            {
                "symbol": "HYPEUSD",
                "timestamp": now_s,
                "mid_price": 86.405,
                "best_bid": 86.38,
                "best_ask": 86.43,
                "spread_bps": 5.79,
                "imbalance_ratio": 0.12,
                "bid_depth_usd": 45000.0,
                "ask_depth_usd": 35000.0,
                "largest_bid_wall_price": 85.50,
                "largest_bid_wall_usd": 12000.0,
                "largest_ask_wall_price": 87.24,
                "largest_ask_wall_usd": 11000.0,
            }
        ),
        encoding="utf-8",
    )
    pt_hype = {"HYPEUSD": {"price": 83.79}, "HYPEUSDC": {"price": 83.8075}}
    micro_hype: dict = {}
    enrich_24h_and_microstructure(str(ob_dir), str(tmp_path), pt_hype, micro_hype, {}, {}, now_s)
    assert abs(float(micro_hype["HYPEUSD"]["ob_mid_price"]) - 83.79) < 1e-4
    assert float(micro_hype["HYPEUSD"]["bid_wall_px"]) < 83.79
    assert float(micro_hype["HYPEUSD"]["ask_wall_px"]) > 83.79

    # 8. Bug 15: build_cross_market_context includes last_exit_kind, pos_peak, dd_from_peak_pct, realized_pnl_usd in t212_cycle
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    cm_ctx = build_cross_market_context(
        {"SPCX_US_EQ": {"price": 138.36}},
        {},
        {"SPCX_US_EQ": spcx_cyc},
    )
    assert cm_ctx["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]["last_exit_kind"] == "REDUCE"
    assert cm_ctx["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]["realized_pnl_usd"] == -115.65

    # 9. Bug 16: _format_market_prices_section omits vs_btc(7d=...) when symbol has sub-24h history matching 24h range
    from orchestratorOS.supervisor.prompt_sections import _format_market_prices_section
    pt_short: dict = {
        "BTCUSDC": {
            "price": 75575.56,
            "trend": -1,
            "growth_coefficient": -0.5,
            "chg_15m_pct": -0.10,
            "chg_1h_pct": -0.30,
            "chg_24h_pct": -2.10,
            "chg_7d_pct": -4.82,
        },
        "ADAUSD": {
            "price": 0.18783,
            "trend": 0,
            "growth_coefficient": 0.0,
            "chg_15m_pct": 0.05,
            "chg_1h_pct": 0.10,
            "chg_7d_pct": -1.22,
            "low_24h": 0.186567,
            "high_24h": 0.191635,
            "low_7d": 0.186567,
            "high_7d": 0.191635,
        },
    }
    pt_lines = _format_market_prices_section(pt_short)
    ada_line = next(l for l in pt_lines if "ADAUSD:" in l)
    assert "1h=+0.40%" in ada_line
    assert "7d=+3.60%" not in ada_line
    assert "7d_range=" not in ada_line

    # 10. Bug 17: build_cross_market_context omits sub-24h chg_7d_pct on crypto_bellwethers when matching 24h range
    cm_short = build_cross_market_context(pt_short, {}, {})
    assert "chg_7d_pct" not in cm_short["crypto_bellwethers"]["ADA"]
    assert cm_short["crypto_bellwethers"]["BTC"]["chg_7d_pct"] == -4.82

    # 11. Bug 18: enrich_open_limit_orders_tca provides recommended_front_run_px across the full BEHIND_BID_WALL window (<=1.5% from wall, >=15bps from spot)
    from intelligence.sentiment.evaluators.tca_risk_evaluator import enrich_open_limit_orders_tca
    tao_orders = [
        {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "price": 220.0, "qty": 2.2727}
    ]
    enrich_open_limit_orders_tca(
        tao_orders,
        {"TAOUSDC": {"price": 223.4, "vol_1h_pct": 1.23}},
        {
            "TAOUSDC": {
                "ob_mid_price": 223.4,
                "best_bid": 223.3,
                "best_ask": 223.5,
                "spread_bps": 8.95,
                "bid_wall_px": 222.9,
                "bid_wall_usd": 14276.0,
            }
        },
    )
    assert tao_orders[0]["wall_queue_pos"] == "BEHIND_BID_WALL"
    assert tao_orders[0]["recommended_front_run_px"] == 222.92

    # 12. Bug 19: _populate_cycle_fallback_recent_order populates ro["last_sell"] from cycle last_sell_price when missing from trade cache
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    ada_pos = [{"venue": "kraken", "symbol": "ADAUSD", "free_base": 4294.02, "price": 0.18783}]
    ada_ro = {
        "last_action_side": "BUY",
        "last_action_actor": "BOT",
        "last_buy": {"price": 0.224378, "qty": 1448.45, "actor": "BOT"},
        "last_sell": None,
        "active_avg_buy_px": 0.227623,
        "recent_buys": [{"price": 0.229281, "qty": 2834.95}, {"price": 0.224378, "qty": 1448.45}],
    }
    ada_ro_map = {"ADAUSD": ada_ro}
    enrich_positions_with_orders_and_cycles(
        ada_pos,
        {"ADAUSD": {"price": 0.18783}},
        {},
        {"kraken": ada_ro_map},
        ada_ro_map,
        {"ADAUSD": {"qty": 4294.02, "avg_cost_px": 0.227623, "last_buy_price": 0.224378, "last_sell_price": 0.238254}},
        {},
        {},
        {},
    )
    assert ada_ro["last_sell"] is not None
    assert ada_ro["last_sell"]["price"] == 0.238254

    # 13. Bugs 20, 21, 22: collect_pretrade_telemetry includes bimodal sec_POC/LVN/shape, recenters stale USD orderbook walls, and formats price-only fills
    import intelligence.sentiment.collectors.pretrade_telemetry_collector as ptc_mod
    pt_cache = tmp_path / "pt_cache"
    pt_cache.mkdir(parents=True, exist_ok=True)
    (pt_cache / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps({
            "symbol": "HYPEUSD",
            "imbalance_ratio": 0.45,
            "mid_price": 86.405,
            "best_bid": 86.38,
            "best_ask": 86.43,
            "spread_bps": 5.79,
            "bid_depth_usd": 450000.0,
            "ask_depth_usd": 470000.0,
            "depth_1pct_bid_usd": 310000.0,
            "depth_1pct_ask_usd": 330000.0,
            "largest_bid_wall_price": 85.50,
            "largest_bid_wall_usd": 384025.0,
            "largest_ask_wall_price": 87.24,
            "largest_ask_wall_usd": 391866.0,
        }),
        encoding="utf-8",
    )
    bimodal_pts = (
        [[now_ms - (40 - i) * 300_000, 86.40 + (i % 3) * 0.02] for i in range(20)]
        + [[now_ms - (20 - i) * 300_000, 83.75 + (i % 3) * 0.02] for i in range(19)]
        + [[now_ms, 83.79]]
    )
    (pt_cache / "cache_24_price_HYPEUSD.json").write_text(
        json.dumps({"items": {"HYPEUSD": bimodal_pts}}),
        encoding="utf-8",
    )
    (pt_cache / "cache_current_price.json").write_text(
        json.dumps({"items": {"HYPEUSD": {"price": 83.79}}}),
        encoding="utf-8",
    )
    orig_best_fills = ptc_mod.load_symbol_best_recent_fills
    try:
        ptc_mod.load_symbol_best_recent_fills = lambda *a, **kw: {
            "last_action_side": "BUY",
            "last_buy": {"price": 84.0, "qty": 5.95, "actor": "BOT"},
            "last_sell": {"price": 87.12, "qty": None, "actor": "CYCLE"},
        }
        pt_hype = ptc_mod.collect_pretrade_telemetry("HYPEUSD", cachedb_dir=str(pt_cache))
    finally:
        ptc_mod.load_symbol_best_recent_fills = orig_best_fills
    vp_str = pt_hype.get("24h Volume Profile (POC / Value Area)", "")
    assert "sec_POC=$" in vp_str and "LVN=$" in vp_str and "shape=BIMODAL" in vp_str
    # Bid wall at $85.50 on stale mid $86.405 is recentered below live $83.79 (~$82.91)
    assert "@ $82.91" in pt_hype.get("Orderbook Bid Wall Ahead", "")
    assert "@ $84.60" in pt_hype.get("Orderbook Ask Wall Ahead", "")
    assert "Last SELL=87.12 (by=CYCLE)" in pt_hype.get("Recent Venue Fills", "")

    # 14. Bugs 23, 24, 25: _format_recent_order_line 20bps swing dedup, build_cross_market_context bimodal/spot_cvd fields, and macro_intelligence.conf config audit
    from orchestratorOS.supervisor.prompt_sections import _format_recent_order_line
    from orchestratorOS.lib.technical_repair_ssot import audit_workspace_configs
    hype_ro_line = _format_recent_order_line(
        "HYPEUSDC",
        {
            "last_action_side": "BUY",
            "last_action_actor": "AI",
            "last_buy": {"price": 87.267, "qty": 1.64, "actor": "AI"},
            "last_sell": {"price": 87.733, "qty": 1.7, "actor": "AI"},
            "active_avg_buy_px": 86.861664,
            "swing_subcycle_avg_px": 86.852582,
        },
        now_ms,
    )
    assert "Active Cycle Avg BUY=86.861664" in hype_ro_line
    assert "Swing Sub-Cycle Avg BUY=" not in hype_ro_line

    cm_bimodal = build_cross_market_context(
        {
            "ARBUSDC": {
                "price": 0.15565,
                "poc_px": 0.158096,
                "sec_poc_px": 0.152523,
                "lvn_px": 0.154393,
                "vah_px": 0.1626,
                "val_px": 0.153654,
                "va_regime": "INSIDE_VA_LOWER",
                "profile_shape": "BIMODAL",
            }
        },
        {
            "ARBUSDC": {
                "cvd_15m_usd": 48073.0,
                "cvd_regime": "aggressive_buying",
                "spot_cvd_15m_usd": 12300.0,
                "spot_perp_flow_regime": "SYNCHRONIZED_SPOT_PERP_BUYING",
                "perp_spot_basis_bps": -0.2,
            }
        },
        {},
    )
    arb_cb = cm_bimodal["crypto_bellwethers"]["ARB"]
    assert arb_cb["sec_poc_px"] == 0.152523
    assert arb_cb["lvn_px"] == 0.154393
    assert arb_cb["profile_shape"] == "BIMODAL"
    assert arb_cb["spot_cvd_15m_usd"] == 12300.0
    assert arb_cb["spot_perp_flow_regime"] == "SYNCHRONIZED_SPOT_PERP_BUYING"
    assert arb_cb["perp_spot_basis_bps"] == -0.2

    bad_cfg_ws = tmp_path / "bad_cfg_ws"
    bad_cfg_ws.mkdir(parents=True, exist_ok=True)
    (bad_cfg_ws / "macro_intelligence.conf").write_text("[feeds]\nonly_feeds = 1\n", encoding="utf-8")
    cfg_res = audit_workspace_configs(str(bad_cfg_ws))
    assert "macro_intelligence.conf" in cfg_res["checked_files"]
    assert any("missing required [sessions] section" in iss for iss in cfg_res["issues"])

    # Bugs 26, 27 & 28:
    # - extract_high_severity_headlines_and_digest filters reference/encyclopedia articles and syndicated semantic duplicates
    # - compute_dynamic_geopolitical_interval ignores syndicated rewrites of already-digested headlines
    # - GeopoliticalThreatAnalyzer._headline_age_suffix formats age=...m for recent headlines
    from intelligence.macro_sentiment_ssot import (
        compute_dynamic_geopolitical_interval,
        extract_high_severity_headlines_and_digest,
    )
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer

    raw_snap_syndicated = {
        "headlines": [
            {
                "title": "Binance fake job interviews linked to $6.3bn crypto hack - Reuters",
                "source": "Reuters",
                "url": "https://example.com/reuters-1",
                "is_high_severity": True,
                "first_seen_ts": (now_ms / 1000.0) - 180.0 * 60.0,
            },
            {
                "title": "Hackers used fake job interviews in $6.3bn Binance crypto hack - Bloomberg",
                "source": "Bloomberg",
                "url": "https://example.com/bloomberg-2",
                "is_high_severity": True,
                "first_seen_ts": (now_ms / 1000.0) - 40.0 * 60.0,
            },
            {
                "title": "2026 Iran war | Oil, Explained, History - Britannica",
                "source": "Britannica",
                "url": "https://www.britannica.com/event/2026-iran-war",
                "is_high_severity": True,
                "first_seen_ts": (now_ms / 1000.0) - 20.0 * 60.0,
            },
        ],
        "high_severity_count": 3,
        "has_critical_shock_keywords": False,
    }
    high_sev_dedup, digest_dedup = extract_high_severity_headlines_and_digest(raw_snap_syndicated)
    assert len(high_sev_dedup) == 1
    assert "Reuters" in digest_dedup
    assert "Bloomberg" not in digest_dedup
    assert "Britannica" not in digest_dedup

    itv_synd, reason_synd = compute_dynamic_geopolitical_interval(
        base_interval_sec=16200.0,
        last_eval_ts=(now_ms / 1000.0) - 1200.0,
        now_ts=(now_ms / 1000.0),
        cached_assessment={"threat_level": "ELEVATED", "is_fallback": False},
        news_snapshot={
            "headlines": [
                {
                    "title": "Hackers used fake job interviews in $6.3bn Binance crypto hack - Bloomberg",
                    "source": "Bloomberg",
                    "is_high_severity": True,
                }
            ],
            "high_severity_count": 1,
            "has_critical_shock_keywords": False,
        },
        headlines_changed=True,
        last_headlines_digest="Binance fake job interviews linked to $6.3bn crypto hack - Reuters",
    )
    assert itv_synd == 16200.0
    assert reason_synd == "standard_geopolitical_ttl_elevated"

    age_sfx = GeopoliticalThreatAnalyzer._headline_age_suffix(
        high_sev_dedup[0],
        now_ms / 1000.0,
    )
    assert age_sfx == ", age=180m"

    # Bugs 29, 30 & 31:
    # - compute_volume_profile_metrics omits sec_poc_px/lvn_px when secondary cluster is <0.5% from POC or lacks a >=10% valley
    # - reconcile_venue_cycles_with_recent_orders replaces stale prior-cycle STOP with REDUCE and syncs dca_buys on already-deducted partial sells
    # - load_symbol_microstructure selects freshest whale_snapshot across variants while keeping venue-specific orderbook_depth
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_microstructure
    from intelligence.sentiment.evaluators.portfolio_evaluator import reconcile_venue_cycles_with_recent_orders
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_volume_profile_metrics

    unimodal_plateau = (
        [[now_ms - (40 - i) * 60_000, 226.45 + (i % 3) * 0.05] for i in range(20)]
        + [[now_ms - (20 - i) * 60_000, 227.15 + (i % 3) * 0.05] for i in range(20)]
    )
    vp_uni = compute_volume_profile_metrics(unimodal_plateau)
    assert vp_uni.get("profile_shape") == "UNIMODAL"
    assert vp_uni.get("sec_poc_px") is None
    assert vp_uni.get("lvn_px") is None

    rgnt_cycs = {
        "RGNT_US_EQ": {
            "qty": 122.51,
            "cost_usd": 173.96,
            "avg_cost_px": 1.42,
            "entry_price": 1.42,
            "_entry_ts_ms": now_ms - 86400_000 * 20,
            "_last_sell_ts_ms": now_ms - 86400_000 * 20,
            "last_buy_price": 1.38,
            "last_sell_price": 1.37,
            "last_exit_kind": "STOP",
            "dca_buys": 0,
        }
    }
    rgnt_ro = {
        "t212": {
            "RGNT_US_EQ": {
                "last_buy": {"price": 1.38, "qty": 72.14, "ts_ms": now_ms - 3600_000 * 20},
                "last_sell": {"price": 1.37, "qty": 36.26, "ts_ms": now_ms - 3600_000 * 50},
                "recent_buys": [
                    {"price": 1.38, "qty": 32.6, "ts_ms": now_ms - 3600_000 * 44},
                    {"price": 1.38, "qty": 72.14, "ts_ms": now_ms - 3600_000 * 20},
                ],
                "recent_sells": [
                    {"price": 1.37, "qty": 36.26, "ts_ms": now_ms - 3600_000 * 50},
                ],
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        {},
        {},
        rgnt_cycs,
        rgnt_ro,
        {("t212", "RGNT_US_EQ"): 122.51},
        {"RGNT_US_EQ": {"price": 1.415}},
    )
    assert rgnt_cycs["RGNT_US_EQ"]["last_exit_kind"] == "REDUCE"
    assert rgnt_cycs["RGNT_US_EQ"]["dca_buys"] == 1

    tao_cache = tmp_path / "tao_micro"
    tao_cache.mkdir(parents=True, exist_ok=True)
    (tao_cache / "orderbook_depth_TAOUSD.json").write_text(
        json.dumps({"symbol": "TAOUSD", "mid_price": 223.35, "best_bid": 223.13, "best_ask": 223.57, "spread_bps": 19.7}),
        encoding="utf-8",
    )
    (tao_cache / "whale_snapshot_TAOUSD.json").write_text(
        json.dumps({"symbol": "TAOUSD", "ts": 1000.0, "taker_buy_sell_ratio": 0.5035, "taker_buy_vol_usd": 588000.0, "taker_sell_vol_usd": 1169000.0}),
        encoding="utf-8",
    )
    (tao_cache / "whale_snapshot_TAOUSDC.json").write_text(
        json.dumps({"symbol": "TAOUSDC", "ts": 2000.0, "taker_buy_sell_ratio": 1.4839, "taker_buy_vol_usd": 926000.0, "taker_sell_vol_usd": 624000.0}),
        encoding="utf-8",
    )
    m_tao = load_symbol_microstructure(str(tao_cache), "TAOUSD")
    assert m_tao is not None
    assert m_tao["spread_bps"] == 19.7
    assert m_tao["taker_buy_sell_ratio"] == 1.4839
    assert m_tao["cvd_15m_usd"] > 0

    # Bugs 33, 34 & 35:
    # - collect_pretrade_telemetry selects freshest whale_snapshot across variants
    # - harvest_heartbeats_and_state_files excludes stale macro_analyzer.heartbeat (>1800s)
    # - _format_sre_and_fleet_section reports 'No running heartbeats (all fleet processes stopped)' when 0 processes run
    from orchestratorOS.lib.technical_repair_ssot import LogEventSummary, harvest_heartbeats_and_state_files
    from orchestratorOS.supervisor.prompt_sections import _format_sre_and_fleet_section

    pt_tao = ptc_mod.collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(tao_cache))
    assert "Buyer/Seller ratio: 1.48" in pt_tao.get("Whale Taker Flow", "")
    assert "aggressive_buying" in pt_tao.get("15m Cumulative Volume Delta (CVD)", "")

    hb_dir_stale = tmp_path / "hb_stale_check"
    hb_dir_stale.mkdir(parents=True, exist_ok=True)
    (hb_dir_stale / "macro_analyzer.heartbeat").write_text(
        json.dumps({"status": "alive", "ts": 1000.0}),
        encoding="utf-8",
    )
    (hb_dir_stale / "globaltelemetry_collector.heartbeat").write_text(
        json.dumps({"status": "alive", "ts": 4500.0}),
        encoding="utf-8",
    )
    act_hbs, _ = harvest_heartbeats_and_state_files(str(hb_dir_stale), str(tmp_path), 5000.0)
    assert "macro_analyzer.heartbeat" not in act_hbs
    assert "globaltelemetry_collector.heartbeat" in act_hbs
    assert act_hbs["globaltelemetry_collector.heartbeat"]["age_seconds"] == 500.0

    sre_lines = _format_sre_and_fleet_section(
        SimpleNamespace(
            active_heartbeats={},
            state_files={},
            fleet_processes=[{"name": "HL-bot", "is_running": False, "heartbeat_fresh": False}],
            system_and_vpn={},
        ),
        LogEventSummary(),
        1.0,
    )
    sec_2b = next(l for l in sre_lines if "### 2B." in l)
    assert "0/1 running" in sec_2b
    assert "No running heartbeats (all fleet processes stopped)" in sec_2b
    assert "All heartbeats fresh" not in sec_2b

    # Bugs 36, 37 & 38:
    # - load_symbol_best_recent_fills supplements missing last_sell from .state_<SYM>.json
    # - collect_pretrade_telemetry evaluates orderbook walls against per-side depth (5% of side depth)
    # - collect_pretrade_telemetry populates '7d Price Range & Swing Structure' from cache_watcher_all_price.json when 24h series is absent
    ws_36 = tmp_path / "ws_36"
    cdir_36 = ws_36 / "cachedb"
    kr_36 = ws_36 / "kraken"
    cdir_36.mkdir(parents=True, exist_ok=True)
    kr_36.mkdir(parents=True, exist_ok=True)
    (cdir_36 / "cache_trade_kraken.json").write_text(
        json.dumps({"items": {"ADAUSD": [{"side": "BUY", "price": 0.224378, "qty": 1448.44, "timestamp": 1773338995022, "status": "FILLED"}]}}),
        encoding="utf-8",
    )
    (kr_36 / ".state_ADAUSD.json").write_text(
        json.dumps({"qty": 4283.15, "entry_price": 0.228669, "last_buy_price": 0.224378, "last_sell_price": 0.238254, "last_sell_ts": 1773332904.415}),
        encoding="utf-8",
    )
    (cdir_36 / "orderbook_depth_TAOUSD.json").write_text(
        json.dumps({
            "symbol": "TAOUSD",
            "mid_price": 223.35,
            "bid_depth_usd": 495383.0,
            "ask_depth_usd": 414240.0,
            "imbalance_ratio": 0.55,
            "largest_ask_wall_usd": 27705.0,
            "largest_ask_wall_price": 224.28,
            "largest_bid_wall_usd": 51581.0,
            "largest_bid_wall_price": 220.35,
        }),
        encoding="utf-8",
    )
    (cdir_36 / "cache_watcher_all_price.json").write_text(
        json.dumps({"items": {"SPCX": [[1793000000000, 145.0], [1793300000000, 135.0], [1793600000000, 142.0], [1793750000000, 138.36]]}}),
        encoding="utf-8",
    )
    pt_ada_36 = ptc_mod.collect_pretrade_telemetry("ADAUSD", cachedb_dir=str(cdir_36))
    assert "Last SELL=0.238254 (by=BOT)" in pt_ada_36.get("Recent Venue Fills", "")
    pt_tao_37 = ptc_mod.collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(cdir_36))
    assert pt_tao_37.get("Orderbook Ask Wall Ahead") == "$27,705 @ $224.28"
    assert pt_tao_37.get("Orderbook Bid Wall Ahead") == "$51,581 @ $220.35"
    pt_spcx_38 = ptc_mod.collect_pretrade_telemetry("SPCX_US_EQ", cachedb_dir=str(cdir_36))
    assert "7d Range=[$135.0, $145.0]" in pt_spcx_38.get("7d Price Range & Swing Structure", "")
    assert "swing_wave(" in pt_spcx_38.get("7d Price Range & Swing Structure", "")

    # Bugs 39, 40 & 41:
    # - update_microstructure_cycle_deltas does not pollute ob_imbalance_delta from sibling venue history and persists missing sibling keys
    # - compute_enterprise_portfolio_risk_metrics scales equity VaR appropriately and weights avg_corr_24h by USD exposure; build_cross_market_context includes 7d range for equities without 24h series
    from intelligence.sentiment.evaluators.microstructure_evaluator import update_microstructure_cycle_deltas
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        build_cross_market_context,
        compute_enterprise_portfolio_risk_metrics,
    )

    mhist_39 = cdir_36 / "microstructure_cycle_history.json"
    mhist_39.write_text(
        json.dumps({"HYPEUSD": [{"ts": 1793750000.0, "ob_imbalance": 0.599, "whale_long_pct": 0.65, "oi_usd": 1900000000.0}]}),
        encoding="utf-8",
    )
    micro_39 = {
        "HYPEUSD": {"ob_imbalance": 0.599, "whale_long_pct": 0.67, "oi_usd": 1900000000.0},
        "HYPEUSDC": {"ob_imbalance": 0.535, "whale_long_pct": 0.67, "oi_usd": 1900000000.0},
    }
    update_microstructure_cycle_deltas(str(cdir_36), micro_39, now_ts=1793750600.0)
    assert "ob_imbalance_delta" not in micro_39["HYPEUSDC"].get("cycle_delta", {})
    assert micro_39["HYPEUSDC"]["cycle_delta"]["whale_long_delta_pct"] == 0.02
    saved_mh_39 = json.loads(mhist_39.read_text(encoding="utf-8"))
    assert "HYPEUSDC" in saved_mh_39 and len(saved_mh_39["HYPEUSDC"]) == 1

    risk_eq_40 = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": [
                {"venue": "t212", "symbol": "SPCX_US_EQ", "value_usd": 974.46},
                {"venue": "t212", "symbol": "NVDA_US_EQ", "value_usd": 6.20},
            ],
            "venue_quote_cash": {"t212:USD": {"venue": "t212", "free": 92.18}},
        },
        {"SPCX_US_EQ": {"price": 138.36, "span_7d_pct": 10.33, "corr_btc_24h": 0.25}, "NVDA_US_EQ": {"price": 206.69, "corr_btc_24h": 0.80}},
        {},
    )
    assert risk_eq_40["intraday_var_95_pct"] < 4.5
    assert abs(risk_eq_40["avg_corr_vs_btc_24h"] - 0.25) <= 0.01

    cmc_40 = build_cross_market_context(
        {"SPCX_US_EQ": {"price": 138.36, "chg_7d_pct": -4.89, "pos_7d_pct": 26.6, "low_7d": 134.665, "high_7d": 148.58}},
        {},
        {},
    )
    assert cmc_40["equity_bellwethers"]["SPCX_US_EQ"]["pos_7d_pct"] == 26.6
    assert cmc_40["equity_bellwethers"]["SPCX_US_EQ"]["low_7d"] == 134.665

    # Bugs 42, 43 & 44:
    # - sync_sibling_symbol_telemetry does not copy bimodal sec_poc_px/lvn_px onto a sibling that already computed its own unimodal poc_px
    # - enrich_sentiment_advisor_live_flow_divergence includes 1h_tr and cvd_1h alongside 15m flow when present
    # - collect_pretrade_telemetry includes flow_horizons and spot_cvd_15m in '15m Cumulative Volume Delta (CVD)'
    from intelligence.sentiment.evaluators.microstructure_evaluator import sync_sibling_symbol_telemetry

    pt_42 = {
        "TAOUSDC": {"price": 223.4, "poc_px": 226.49, "sec_poc_px": 221.5, "lvn_px": 224.0, "profile_shape": "BIMODAL"},
        "TAOUSD": {"price": 223.34, "poc_px": 226.8, "profile_shape": "UNIMODAL"},
    }
    sync_sibling_symbol_telemetry(pt_42, {})
    assert "sec_poc_px" not in pt_42["TAOUSD"]
    assert "lvn_px" not in pt_42["TAOUSD"]
    assert pt_42["TAOUSD"]["profile_shape"] == "UNIMODAL"

    sa_43 = {"market_bias": "NEUTRAL", "recommended_action": "HOLD", "age_min": 45.0, "summary": "Quiet taker ratio 0.87."}
    enrich_sentiment_advisor_live_flow_divergence(
        sa_43,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.38, "cvd_15m_usd": 26_040_000.0, "taker_ratio_1h": 0.78, "cvd_1h_usd": -117_130_000.0}},
    )
    assert "1h_tr=0.78" in sa_43["live_flow_divergence"]
    assert "cvd_1h=-117.13M" in sa_43["live_flow_divergence"]

    (cdir_36 / "whale_snapshot_TAOUSD.json").write_text(
        json.dumps({
            "symbol": "TAOUSD",
            "top_traders_long_pct": 0.745,
            "taker_buy_sell_ratio": 0.5035,
            "taker_buy_vol_usd": 589331.0,
            "taker_sell_vol_usd": 1170479.0,
            "taker_ratio_5m": 1.07,
            "cvd_5m_usd": 9600.0,
            "taker_ratio_1h": 0.76,
            "cvd_1h_usd": -722300.0,
            "spot_cvd_15m_usd": 41800.0,
        }),
        encoding="utf-8",
    )
    pt_tao_44 = ptc_mod.collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(cdir_36))
    assert "flow_horizons(5m=tr:1.07/cvd:$+9,600, 1h=tr:0.76/cvd:$-722,300)" in pt_tao_44.get("15m Cumulative Volume Delta (CVD)", "")
    assert "spot_cvd_15m=$+28,400 (BALANCED_CROSS_VENUE_FLOW)" in pt_tao_44.get("15m Cumulative Volume Delta (CVD)", "")

    # Bugs 45, 46 & 47:
    # - build_cross_market_context preserves 3 significant digits for small slopes (<1e-4) instead of rounding to 6 decimal places
    # - collect_pretrade_telemetry recalculates m24 geometry with live orderbook mid price and emits 7d range even when m24 is present
    cmc_46 = build_cross_market_context(
        {"ADAUSD": {"price": 0.18783, "slope_3_7m": 6.48e-07, "slope_15m": 1.55e-06, "slope_2_5h": -6.15e-07}},
        {},
        {},
    )
    assert cmc_46["crypto_bellwethers"]["ADA"]["slope_3_7m"] == 6.48e-07
    assert cmc_46["crypto_bellwethers"]["ADA"]["slope_15m"] == 1.55e-06
    assert cmc_46["crypto_bellwethers"]["ADA"]["slope_2_5h"] == -6.15e-07

    (cdir_36 / "cache_24_price_TAOUSD.json").write_text(
        json.dumps({"items": {"TAOUSD": [[1793670000000 + i * 3600000, 222.0 + (i % 5) * 0.5] for i in range(25)]}}),
        encoding="utf-8",
    )
    (cdir_36 / "cache_watcher_all_price.json").write_text(
        json.dumps({
            "items": {
                "SPCX": [[1793000000000, 145.0], [1793300000000, 135.0], [1793600000000, 142.0], [1793750000000, 138.36]],
                "TAOUSD": [[1793100000000, 250.0], [1793400000000, 220.0], [1793750000000, 223.35]],
            }
        }),
        encoding="utf-8",
    )
    pt_tao_45 = ptc_mod.collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(cdir_36))
    assert "24h Multi-Horizon Price Evolution" in pt_tao_45
    assert "7d Price Range & Swing Structure" in pt_tao_45
    assert "7d Range=[$220.0, $250.0]" in pt_tao_45["7d Price Range & Swing Structure"]

    # Bugs 48, 49 & 50:
    # - harvest_macro_and_news_sensors clamps macro_benchmarks low_24h/high_24h to include price
    # - load_symbol_price_series appends newer series_24h ticks after arch_7d_end_ts to vals_7d so sub-7d troughs are preserved
    # - summarize_recent_ai_intents does not let open resting orders with age_h > 0.5 consume the 8-slot non-resting quota
    from intelligence.sentiment.collectors.macro_telemetry_collector import harvest_macro_and_news_sensors
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series
    from providers.execution_audit import extract_intent_notional, summarize_recent_ai_intents

    (cdir_36 / "macro_benchmarks_collect.json").write_text(
        json.dumps({
            "ts": 1793810000.0,
            "items": {
                "SPY": {"symbol": "SPY", "price": 757.09, "low_24h": 753.35, "high_24h": 757.01, "chg_24h_pct": 0.44},
                "IBIT": {"symbol": "IBIT", "price": 44.31, "low_24h": 43.66, "high_24h": 44.05, "chg_24h_pct": 0.61},
            },
        }),
        encoding="utf-8",
    )
    ms_48, _ = harvest_macro_and_news_sensors(str(cdir_36), 1793810060.0)
    assert ms_48["macro_benchmarks"]["items"]["SPY"]["high_24h"] == 757.09
    assert ms_48["macro_benchmarks"]["items"]["IBIT"]["high_24h"] == 44.31

    (cdir_36 / "cache_365_price_ADAUSD.jsonl").write_text(
        "\n".join(json.dumps([1793700000000 + i * 1000000, 0.1900 - i * 0.0001]) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    (cdir_36 / "cache_24_price_ADAUSD.json").write_text(
        json.dumps({
            "items": {
                "ADAUSD": [
                    [1793709000000, 0.1891],
                    [1793710000000, 0.186567],
                    [1793711000000, 0.18783],
                ]
            }
        }),
        encoding="utf-8",
    )
    _, vals_7d_49 = load_symbol_price_series(str(cdir_36), "ADAUSD", {})
    assert min(vals_7d_49) == 0.186567

    intents_50 = [
        {"decision": "BUY", "symbol": "BTCUSDC", "venue": "binance", "order_type": "MARKET", "status": "EXECUTED", "suggested_notional_usd": 100.0 + k, "timestamp": 1793700000.0 + k * 60.0, "execution_payload": {"orderId": 1000 + k, "status": "FILLED", "executedQty": "0.001", "price": "77000.0"}}
        for k in range(8)
    ] + [
        {"decision": "SELL", "symbol": "TAOUSDC", "venue": "binance", "order_type": "LIMIT", "limit_price": 231.5, "status": "EXECUTED", "suggested_notional_usd": 500.0, "timestamp": 1793705000.0, "execution_payload": {"orderId": 9999, "status": "NEW", "origQty": "2.0", "executedQty": "0.0", "price": "231.5"}},
    ]
    open_50 = [{"venue": "binance", "symbol": "TAOUSDC", "order_id": "9999", "side": "SELL", "price": 231.5, "orig_qty": 2.0, "executed_qty": 0.0, "status": "NEW"}]
    sum_50 = summarize_recent_ai_intents(intents_50, "crypto", 1793710000.0, extract_intent_notional, open_orders=open_50)
    assert len([x for x in sum_50 if x.get("order_id") != "9999"]) == 8

    # Bugs 51 & 52:
    # - harvest_heartbeats_and_state_files uses payload ts/timestamp for fleet .heartbeat files in system/heartbeat
    # - collect_pretrade_telemetry includes major_low and swings_7d/avg_amp in 7d Price Range & Swing Structure
    from orchestratorOS.lib.technical_repair_ssot import harvest_heartbeats_and_state_files

    hb_sys_dir = cdir_36 / "system" / "heartbeat"
    hb_sys_dir.mkdir(parents=True, exist_ok=True)
    (hb_sys_dir / "HL-bot.heartbeat").write_text(
        json.dumps({"ts": 1793710000.0, "status": "OK", "service": "HL-bot"}),
        encoding="utf-8",
    )
    (hb_sys_dir / "Stale-bot.heartbeat").write_text(
        json.dumps({"ts": 1793700000.0, "status": "OK", "service": "Stale-bot"}),
        encoding="utf-8",
    )
    hbs_51, _ = harvest_heartbeats_and_state_files(str(cdir_36), str(tmp_path), 1793710045.0)
    assert "HL-bot.heartbeat" in hbs_51
    assert hbs_51["HL-bot.heartbeat"]["age_seconds"] == 45.0
    assert "Stale-bot.heartbeat" not in hbs_51

    (cdir_36 / "cache_365_price_TAOUSDC.jsonl").write_text(
        "\n".join(
            json.dumps([1793600000000 + i * 3600000, px_v])
            for i, px_v in enumerate([240.0, 210.0, 235.0, 215.0, 245.0, 220.0, 232.0, 228.0])
        )
        + "\n",
        encoding="utf-8",
    )
    pt_tao_52 = ptc_mod.collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cdir_36))
    assert "swings_7d=[" in pt_tao_52.get("7d Price Range & Swing Structure", "")
    assert "avg_amp=" in pt_tao_52.get("7d Price Range & Swing Structure", "")

    # Bugs 53 & 54:
    # - parse_venue_cycle_state avoids cent-rounding quantization on small single-lot qty (0.03 * 207.92708333 -> cost_usd=6.24)
    #   and reconcile_venue_cycles_with_recent_orders updates last_exit_kind to REDUCE on newer scratch/loss sells below entry/avg
    # - build_cross_market_context includes 7d swing structure metrics for equity bellwethers when 24h series is absent
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        parse_venue_cycle_state,
        reconcile_venue_cycles_with_recent_orders,
    )

    nvda_cyc_53 = parse_venue_cycle_state(
        {
            "qty": 0.03,
            "cost_usd": 6.24,
            "dca_buys": 0,
            "entry_price": 207.92708333,
            "last_buy_price": 207.92708333,
            "last_sell_price": 208.1,
            "last_sell_ts": 1793718965.337,
            "last_exit_kind": "TP",
        },
        "NVDA_US_EQ",
        {"NVDA_US_EQ": {"price": 206.69}},
        avg_decimals=4,
        include_meta=True,
    )
    assert nvda_cyc_53["avg_cost_px"] == 207.9271
    t212_cycs_53 = {"NVDA_US_EQ": nvda_cyc_53}
    reconcile_venue_cycles_with_recent_orders(
        {},
        {},
        t212_cycs_53,
        {
            "t212": {
                "NVDA_US_EQ": {
                    "active_avg_buy_px": 207.93,
                    "active_cycle_rem_qty": 0.03,
                    "lifo_realized_pnl_usd": -0.03,
                    "last_buy": {"price": 207.93, "qty": 0.96, "ts_ms": 1793720000000},
                    "last_sell": {"price": 207.9, "qty": 0.93, "ts_ms": 1793724706157},
                    "recent_buys": [{"price": 207.93, "qty": 0.96, "ts_ms": 1793720000000}],
                    "recent_sells": [{"price": 207.9, "qty": 0.93, "ts_ms": 1793724706157}],
                }
            }
        },
        {("t212", "NVDA_US_EQ"): 0.03},
        {"NVDA_US_EQ": {"price": 206.69}},
    )
    assert t212_cycs_53["NVDA_US_EQ"]["last_exit_kind"] == "REDUCE"

    cmc_54 = build_cross_market_context(
        {
            "SPCX_US_EQ": {
                "price": 138.36,
                "chg_7d_pct": -4.89,
                "pos_7d_pct": 26.6,
                "low_7d": 134.665,
                "high_7d": 148.58,
                "dd_7d_high_pct": -6.88,
                "bounce_7d_low_pct": 2.74,
                "span_7d_pct": 10.33,
                "swing_low_px": 136.14,
                "swing_high_px": 142.468,
                "rally_from_swing_low_pct": 1.63,
                "pullback_from_swing_high_pct": -2.88,
            }
        },
        {},
        {},
    )
    assert cmc_54["equity_bellwethers"]["SPCX_US_EQ"]["swing_low_px"] == 136.14
    assert cmc_54["equity_bellwethers"]["SPCX_US_EQ"]["swing_high_px"] == 142.468

    # Bug 55: collect_pretrade_telemetry includes funding_ann, perp_basis, 1h_chg on OI, and Cross-Venue Relative Value (Arbitrage)
    (cdir_36 / "orderbook_depth_TAOUSDC.json").write_text(
        json.dumps(
            {
                "bid_depth_usd": 120000.0,
                "ask_depth_usd": 90000.0,
                "imbalance_pct": 14.29,
                "spread_bps": 1.2,
                "mid_price": 228.5,
                "depth_1pct_bid_usd": 115000.0,
                "depth_1pct_ask_usd": 88000.0,
                "bids": [[228.45, 50.0]],
                "asks": [[228.55, 50.0]],
            }
        ),
        encoding="utf-8",
    )
    (cdir_36 / "orderbook_depth_TAOUSD.json").write_text(
        json.dumps(
            {
                "bid_depth_usd": 45000.0,
                "ask_depth_usd": 38000.0,
                "imbalance_pct": 8.43,
                "spread_bps": 4.5,
                "mid_price": 228.1,
                "depth_1pct_bid_usd": 42000.0,
                "depth_1pct_ask_usd": 35000.0,
                "bids": [[228.05, 20.0]],
                "asks": [[228.15, 20.0]],
            }
        ),
        encoding="utf-8",
    )
    (cdir_36 / "whale_snapshot_TAOUSDC.json").write_text(
        json.dumps(
            {
                "funding_rate": 0.0001,
                "oi_usd": 125000000.0,
                "oi_1h_chg_pct": 1.45,
                "cvd_15m_usd": 85000.0,
                "spot_cvd_15m_usd": 110000.0,
            }
        ),
        encoding="utf-8",
    )
    pt_tao_55 = ptc_mod.collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cdir_36))
    assert "funding_ann=" in pt_tao_55.get("Perpetual Funding Rate", "")
    assert "perp_basis=" in pt_tao_55.get("Perpetual Funding Rate", "")
    assert "1h_chg=+1.45%" in pt_tao_55.get("Derivatives Open Interest", "")
    assert "arb_vs_TAOUSD=" in pt_tao_55.get("Cross-Venue Relative Value (Arbitrage)", "")
    assert "net_buy_edge=" in pt_tao_55.get("Cross-Venue Relative Value (Arbitrage)", "")

    # Bug 57 & 58: suppress duplicate sub-24h swing_wave in Section 3A; include 90s slope & sub-cycle cost anchors in pretrade
    sub24_sec = _format_market_prices_section(
        {
            "ADAUSD": {
                "price": 0.1878,
                "chg_15m_pct": 0.12,
                "low_24h": 0.186567,
                "high_24h": 0.191635,
                "chg_7d_pct": -1.5,
                "low_7d": 0.186567,
                "high_7d": 0.191635,
                "swing_low_px": 0.186567,
                "swing_high_px": 0.191635,
                "rally_from_swing_low_pct": 0.68,
                "pullback_from_swing_high_pct": -1.99,
            }
        }
    )
    assert "swing_wave(" not in sub24_sec
    t24_dir = cdir_36 / "24trend"
    t24_dir.mkdir(exist_ok=True)
    (t24_dir / "live_TAOUSDC.json").write_text(
        json.dumps(
            {
                "90": {"slope_full": 0.00125},
                "222": {"slope_full": 0.00085},
                "900": {"slope_full": 0.00045},
                "9000": {"slope_full": 0.00015},
            }
        ),
        encoding="utf-8",
    )
    (cdir_36 / "cache_order.json").write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": [
                        {"orderId": "1", "side": "BUY", "price": 260.0, "executedQty": 2.0, "timestamp": 1700000000000, "status": "FILLED"},
                        {"orderId": "2", "side": "BUY", "price": 224.0, "executedQty": 1.0, "timestamp": 1700500000000, "status": "FILLED"},
                        {"orderId": "3", "side": "BUY", "price": 226.0, "executedQty": 1.0, "timestamp": 1700501800000, "status": "FILLED"},
                        {"orderId": "4", "side": "SELL", "price": 228.0, "executedQty": 0.5, "timestamp": 1700503600000, "status": "FILLED"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pt_tao_58 = ptc_mod.collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cdir_36))
    assert "90s=+0.00125" in pt_tao_58.get("Dynamic Multi-Window Slopes", "")
    assert "Active Cycle Avg BUY=" in pt_tao_58.get("Recent Venue Fills", "")
    assert "Tactical Sub-Cycle Avg BUY=" in pt_tao_58.get("Recent Venue Fills", "")
    assert "Sub-Cycle LIFO Realized PnL=$+1.00" in pt_tao_58.get("Recent Venue Fills", "")

    # Bug 59 & 60: build_cross_market_context includes slope_90s and t212_cycle cost_usd/dca_buys
    cmc_60 = build_cross_market_context(
        {"BTCUSDC": {"price": 77000.0, "slope_90s": 0.11591, "slope_15m": 0.17415}},
        {},
        {"SPCX_US_EQ": {"qty": 7.04, "cost_usd": 1021.14, "avg_cost_px": 144.9872, "dca_buys": 1}},
    )
    assert cmc_60["crypto_bellwethers"]["BTC"]["slope_90s"] == 0.11591
    assert cmc_60["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]["cost_usd"] == 1021.14
    assert cmc_60["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]["dca_buys"] == 1

    # Discrepancy AG & AI: 22h watcher/archive span (<5d, >=18h) backfills full 7d OHLC range; de-escalation headline filtered
    from intelligence.macro.lexicon_ssot import filter_valid_severity_keywords
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_cross_venue_trends

    assert (
        filter_valid_severity_keywords(
            ["escalation", "iran escalation"],
            title="Oil Prices to Come Under Pressure if Midterms Outcome Halts Iran Escalation - WSJ",
        )
        == ()
    )
    now_ag = 1_793_800_000.0
    watcher_ag = {"ADAUSD": [[int((now_ag - 22 * 3600) * 1000), 0.18252], [int(now_ag * 1000), 0.1879]]}
    (cdir_36 / "cache_24_price_ADAUSD.json").write_text(json.dumps({"items": watcher_ag}), encoding="utf-8")
    pt_ag = {"ADAUSD": {"price": 0.1879}}
    enrich_cross_venue_trends(pt_ag, {}, {}, {}, {}, {}, watcher_ag)
    assert pt_ag["ADAUSD"].get("_partial_7d_span_ms") is not None

    class _FakeAdaProvider:
        def ohlc_series(self, sym, interval):
            from types import SimpleNamespace

            n_bars = 168 if interval == 60 else 42
            step_s = interval * 60
            ts_list = [now_ag - (n_bars - 1 - idx) * step_s for idx in range(n_bars)]
            closes = [0.2072 if idx == 0 else (0.214343 if idx == 30 else 0.1879) for idx in range(n_bars)]
            return SimpleNamespace(timestamps=ts_list, closes=closes, lows=closes, highs=closes, volumes=[1000.0] * n_bars)

    from types import SimpleNamespace as _SNS

    enrich_24h_and_microstructure(
        str(cdir_36),
        str(tmp_path),
        pt_ag,
        {},
        {},
        {"ADAUSD": _SNS(_provider=_FakeAdaProvider(), provider_name="kraken")},
        now_ag,
    )
    enrich_cross_venue_trends(pt_ag, {}, {}, {}, {}, {}, watcher_ag)
    assert pt_ag["ADAUSD"]["high_7d"] == 0.214343
    assert pt_ag["ADAUSD"]["low_7d"] == 0.18252
    assert "_partial_7d_span_ms" not in pt_ag["ADAUSD"]

    # Wave 3: lowest_buy_px <= last_buy_px when LIFO pops dip buy + stale >14d open_lots ignored; va_pos_pct clamped [-100, 200]; sparse horizon dedup
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from intelligence.sentiment.evaluators.price_series_evaluator import _build_vp_position_dict, compute_24h_price_metrics

    pos_w3 = [
        {"venue": "binance", "symbol": "TAOUSDC", "free_base": 6.0, "price": 233.6},
        {"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.01, "price": 80100.0},
    ]
    now_ms_w3 = 1_793_800_000_000
    ro_w3 = {
        "binance": {
            "TAOUSDC": {
                "last_action_side": "SELL",
                "active_avg_buy_px": 248.3,
                "last_buy": {"price": 220.0, "qty": 2.27, "ts_ms": now_ms_w3 - 3600_000},
                "last_sell": {"price": 238.4, "qty": 4.19, "ts_ms": now_ms_w3 - 1800_000},
                "recent_buys": [{"price": 221.0, "qty": 2.11}, {"price": 220.0, "qty": 2.27}],
                "open_lots": [{"lot_id": "L1", "qty": 6.0, "entry_px": 232.3, "ts_ms": now_ms_w3 - 7200_000}],
            },
            "BTCUSDC": {
                "last_action_side": "SELL",
                "active_avg_buy_px": 77959.9,
                "last_buy": {"price": 77578.56, "qty": 0.00644, "ts_ms": now_ms_w3 - 3600_000},
                "recent_buys": [{"price": 77208.59, "qty": 0.00014}, {"price": 77578.56, "qty": 0.00644}],
                "open_lots": [
                    {"lot_id": "L0", "qty": 0.0001, "entry_px": 62805.19, "ts_ms": now_ms_w3 - 30 * 86400 * 1000},
                    {"lot_id": "L2", "qty": 0.01, "entry_px": 77352.51, "ts_ms": now_ms_w3 - 7200_000},
                ],
            },
        }
    }
    enrich_positions_with_orders_and_cycles(
        pos_w3,
        {"TAOUSDC": {"price": 233.6}, "BTCUSDC": {"price": 80100.0}},
        {},
        ro_w3,
        {},
        {},
        {},
        {},
        {},
    )
    assert pos_w3[0]["lowest_buy_px"] == 220.0
    assert pos_w3[1]["lowest_buy_px"] == 77352.51

    vp_gap = _build_vp_position_dict(134.45, 141.18, 140.36, 142.47)
    assert vp_gap["va_pos_pct"] == -100.0

    # Sparse overnight gap between -3h and -13h in a >=25 point series: 6h and 12h would hit the same bar at -13h
    sparse_pts = [[now_ms_w3 - 24 * 3600_000 + i * 1800_000, 100.0 + i * 0.1] for i in range(23)]  # -24h .. -13h
    sparse_pts += [[now_ms_w3 - 180 * 60_000 + i * 10 * 60_000, 105.0 + i * 0.1] for i in range(19)]  # -3h .. now
    m24_sparse = compute_24h_price_metrics(sparse_pts)
    assert "chg_6h_pct" in m24_sparse
    assert "chg_12h_pct" not in m24_sparse

    # Bugs 13, 14, 15 & 16:
    # - harvest_log_anomalies_from_dirs skips __orchestrator_intent__ IPC lines, deduplicates chained tracebacks, and clears pre-startup ImportError after clean strategy startup
    # - enrich_sentiment_advisor_live_flow_divergence does not trigger LIVE_BUY_REVERSAL on NEUTRAL/HOLD when live_tr < 1.25 (e.g. 1.16)
    # - _apply_24h_series_enrichment computes fallback slopes/vol_1h for equities; collect_pretrade_telemetry syncs live equity price and strips post-RTH short tape
    ipc_log_dir = tmp_path / "ipc_and_krk_logs"
    ipc_log_dir.mkdir(parents=True, exist_ok=True)
    clk_now = time.strftime("%H:%M:%S", time.localtime(time.time() - 20))
    (ipc_log_dir / "macro_analyzer.log").write_text(
        '{"__orchestrator_intent__": "ntfy_webhook", "alerts": [{"title": "HARD_VETO_NEW_BUYS Error in feed"}]}\n',
        encoding="utf-8",
    )
    (ipc_log_dir / "Kraken-HYPE.log").write_text(
        "Traceback (most recent call last):\n"
        '  File "/app/kraken/kraken_bot.py", line 25, in <module>\n'
        "    from kraken.kraken_client import KrakenRestClient\n"
        "ImportError: cannot import name 'normalize_epoch_sec' from 'botcore'\n\n"
        "During handling of the above exception, another exception occurred:\n\n"
        "Traceback (most recent call last):\n"
        '  File "/app/kraken/kraken_bot.py", line 28, in <module>\n'
        "    from kraken_client import KrakenRestClient\n"
        "ImportError: cannot import name 'normalize_epoch_sec' from 'botcore'\n"
        f"[{clk_now}]   [STRAT] state loaded\n"
        "=== STRATEGIE KRAKEN PORNITA ===\n",
        encoding="utf-8",
    )
    ipc_summary = harvest_log_anomalies_from_dirs([str(ipc_log_dir)], lookback_hours=1.0)
    assert ipc_summary.error_count == 0
    assert ipc_summary.warning_count == 0
    assert ipc_summary.critical_events == []
    assert ipc_summary.guard_interventions == []
    assert ipc_summary.persistent_errors == []

    sa_neutral_mild = {
        "market_bias": "NEUTRAL",
        "recommended_action": "HOLD",
        "age_min": 24.5,
        "summary": "Balanced flow with taker buy/sell ratio (0.89).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_neutral_mild,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.1592, "cvd_15m_usd": 7_270_000.0, "taker_ratio_1h": 0.71, "cvd_1h_usd": -61_090_000.0}},
    )
    assert "live_flow_divergence" not in sa_neutral_mild
    from intelligence.sentiment.evaluators.price_series_evaluator import _apply_24h_series_enrichment

    spcx_eq_entry: dict = {"price": 134.54}
    spcx_eq_pts = [[now_ms_w3 - (30 - i) * 300_000, 132.0 + i * 0.08] for i in range(31)]
    _apply_24h_series_enrichment("SPCX_US_EQ", spcx_eq_entry, compute_24h_price_metrics(spcx_eq_pts), spcx_eq_pts, None)
    assert spcx_eq_entry.get("slope_15m") is not None and spcx_eq_entry.get("vol_1h_pct") is not None

    # Bugs 61 & 62:
    # - Section 3A clamps raw pinfo['va_pos_pct'] to [-100.0, 200.0]
    # - enrich_sentiment_advisor_live_flow_divergence distinguishes LIVE_15M_BOUNCE_VS_1H_SELL under NEUTRAL/HOLD when 1h flow is deeply negative
    lines_61 = _format_market_prices_section(
        {"SPCX_US_EQ": {"price": 134.54, "poc_px": 141.18, "val_px": 140.36, "vah_px": 142.47, "va_pos_pct": -275.3, "va_regime": "BELOW_VAL"}}
    )
    assert "va_pos=-100.0%" in "\n".join(lines_61)

    sa_62 = {
        "market_bias": "NEUTRAL",
        "recommended_action": "HOLD",
        "age_min": 40.0,
        "summary": "Balanced flow with taker buy/sell ratio (0.87).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_62,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.38, "cvd_15m_usd": 26_040_000.0, "taker_ratio_1h": 0.78, "cvd_1h_usd": -117_130_000.0}},
    )
    assert sa_62["live_flow_divergence"].startswith("LIVE_15M_BOUNCE_VS_1H_SELL (")

    # Discrepancy AJ & AK: NEUTRAL/HOLD advisor with cited_tr=0.58 still triggers LIVE_SELL_REVERSAL on heavy live dump;
    # compute_swing_wave_metrics promotes prior completed trough when cur_v == lo_px on pullback
    sa_neutral_cited_low = {
        "market_bias": "NEUTRAL",
        "recommended_action": "HOLD",
        "age_min": 27.0,
        "summary": "Voluntary long_unwinding (0.58 taker buy/sell ratio) with passive spot bids supporting HOLD.",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_neutral_cited_low,
        {"BTCUSDC": {"taker_buy_sell_ratio": 0.7363, "cvd_15m_usd": -13_020_000.0, "taker_ratio_1h": 0.81, "cvd_1h_usd": -34_370_000.0}},
    )
    assert "LIVE_SELL_REVERSAL" in sa_neutral_cited_low.get("live_flow_divergence", "")
    assert "NEUTRAL/HOLD (advisor ratio=0.58)" in sa_neutral_cited_low.get("live_flow_divergence", "")

    from intelligence.sentiment.evaluators.price_series_evaluator import compute_swing_wave_metrics

    sw_ak = compute_swing_wave_metrics([0.193, 0.18252, 0.201272, 0.194847])
    assert sw_ak["swing_low_px"] == 0.18252
    assert sw_ak["swing_high_px"] == 0.201272
    assert sw_ak["rally_from_swing_low_pct"] == 6.75
    assert sw_ak["pullback_from_swing_high_pct"] == -3.19

    # Wave 6: collect_pretrade_telemetry omits deduplicated/stale horizons and sub-2.5h slopes on sparse series
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry

    pt_sparse_dir = tmp_path / "pt_sparse"
    pt_sparse_dir.mkdir(parents=True, exist_ok=True)
    rgnt_pts_pt = [[now_ms_w3 - 24 * 3600_000 + i * 2 * 3600_000, round(1.36 + (i % 3) * 0.02, 4), 1000.0] for i in range(11)]
    rgnt_pts_pt += [
        [now_ms_w3 - 4 * 3600_000, 1.3717, 1500.0],
        [now_ms_w3 - 3 * 3600_000, 1.36, 1200.0],
        [now_ms_w3 - 10 * 60_000, 1.37, 800.0],
    ]
    (pt_sparse_dir / "cache_24_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": rgnt_pts_pt}}),
        encoding="utf-8",
    )
    pt_rgnt = collect_pretrade_telemetry("RGNT_US_EQ", cachedb_dir=str(pt_sparse_dir))
    assert "30m=+0.00%" not in pt_rgnt.get("24h Multi-Horizon Price Evolution", "")
    assert "1h=+0.00%" not in pt_rgnt.get("24h Multi-Horizon Price Evolution", "")
    assert "3.7m=+0.00000" not in pt_rgnt.get("Dynamic Multi-Window Slopes", "")
    assert "15m=+0.00000" not in pt_rgnt.get("Dynamic Multi-Window Slopes", "")
    assert "2.5h=" in pt_rgnt.get("Dynamic Multi-Window Slopes", "")

    # Wave 8: 5m OHLC bars + live tick deduplicate tape_15m (-3m vs -1m) and do not fabricate 90s slope or duplicate growth_3_7m
    from intelligence.sentiment.evaluators.price_series_evaluator import _compute_short_slopes_from_series

    eq_5m_bars = [
        [now_ms_w3 - (18.5 - 5.0 * idx) * 60_000, round(200.0 + idx * 0.25, 2)]
        for idx in range(-30, 4)
    ]
    eq_5m_bars[-4] = [now_ms_w3 - int(18.5 * 60_000), 202.27]
    eq_5m_bars[-3] = [now_ms_w3 - int(13.5 * 60_000), 202.14]
    eq_5m_bars[-2] = [now_ms_w3 - int(8.5 * 60_000), 202.10]
    eq_5m_bars[-1] = [now_ms_w3 - int(3.5 * 60_000), 202.41]
    eq_5m_live = eq_5m_bars + [[now_ms_w3, 202.44]]
    m24_eq_live = compute_24h_price_metrics(eq_5m_live)
    assert m24_eq_live.get("recent_tape_15m") == [
        "-15m:202.27",
        "-10m:202.14",
        "-5m:202.1",
        "-3m:202.41",
        "now:202.44",
    ]
    slopes_eq_live = _compute_short_slopes_from_series(eq_5m_live)
    assert "slope_90s" not in slopes_eq_live
    assert "slope_3_7m" in slopes_eq_live and "growth_3_7m" not in slopes_eq_live
    assert "slope_15m" in slopes_eq_live and "growth_15m" in slopes_eq_live

    # Wave 9: parse_cited_taker_ratio prefers overall/15m over 5m sub-horizon and 3m horizon does not steal -6.4m 5m bar
    sa_overall_vs_5m = {
        "directional_bias": "NEUTRAL",
        "recommended_action": "HOLD",
        "age_min": 17.3,
        "summary": "BTC holding above $80k (5m taker ratio 1.35, +$13.9M 5m CVD; overall taker ratio recovering to 0.95).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_overall_vs_5m,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.36, "cvd_15m_usd": 21_030_000.0, "taker_ratio_1h": 0.96, "cvd_1h_usd": -10_750_000.0}},
    )
    assert "advisor ratio=0.95" in sa_overall_vs_5m.get("live_flow_divergence", "")
    assert "advisor ratio=1.35" not in sa_overall_vs_5m.get("live_flow_divergence", "")

    eq_5m_1m37 = [
        [now_ms_w3 - int((16.37 - 5.0 * idx) * 60_000), round(200.0 + idx * 0.25, 2)]
        for idx in range(-30, 4)
    ]
    eq_5m_1m37[-4] = [now_ms_w3 - int(16.37 * 60_000), 202.12]
    eq_5m_1m37[-3] = [now_ms_w3 - int(11.37 * 60_000), 202.35]
    eq_5m_1m37[-2] = [now_ms_w3 - int(6.37 * 60_000), 202.21]
    eq_5m_1m37[-1] = [now_ms_w3 - int(1.37 * 60_000), 202.12]
    m24_1m37 = compute_24h_price_metrics(eq_5m_1m37 + [[now_ms_w3, 202.14]])
    assert "chg_1m_pct" in m24_1m37 and "chg_5m_pct" in m24_1m37 and "chg_3m_pct" not in m24_1m37

    # Wave 10: Discrepancies AL & AM (LIVE_15M_BOUNCE_VS_1H_SELL at 1h_tr=0.93/cvd_1h=-19.10M, _fmt_slope_val 5e-9 precision, sibling 90s sync)
    from orchestratorOS.supervisor.prompt_sections import _fmt_slope_val
    from intelligence.sentiment.evaluators.microstructure_evaluator import sync_sibling_symbol_telemetry

    sa_al = {
        "directional_bias": "NEUTRAL",
        "recommended_action": "HOLD",
        "age_min": 11.3,
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_al,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.30, "cvd_15m_usd": 27_520_000.0, "taker_ratio_1h": 0.93, "cvd_1h_usd": -19_100_000.0}},
    )
    assert sa_al.get("live_flow_divergence", "").startswith("LIVE_15M_BOUNCE_VS_1H_SELL")

    assert _fmt_slope_val(9.71e-08, 5) == "+9.71e-08"
    assert _fmt_slope_val(4.97e-10, 5) == "+0.00000"

    pt_sib_90s = {
        "TAOUSDC": {"price": 232.9, "slope_90s": 0.0, "growth_90s": 0.0, "gradient_recent_fast": 0.0},
        "TAOUSD": {"price": 232.8628, "slope_90s": -0.00195, "growth_90s": 0.00043, "gradient_recent_fast": -0.0026},
    }
    sync_sibling_symbol_telemetry(pt_sib_90s, {})
    assert pt_sib_90s["TAOUSDC"]["slope_90s"] == -0.00195
    assert pt_sib_90s["TAOUSDC"]["growth_90s"] == 0.00043
    assert pt_sib_90s["TAOUSDC"]["gradient_recent_fast"] == -0.0026

    # Bugs 63, 64 & 65:
    # - load_symbol_price_series falls back to dense (>10 pts in 24h) cache_watcher_all_price.json when cache_24_price/archive are absent
    # - enrich_24h_and_microstructure populates 24h Volume Profile, slopes, and vol_1h_pct for SPCX_US_EQ from dense cache_watcher_all_price.json
    # - collect_pretrade_telemetry loads cache_kalman_price.json into t_entry for liquidation cluster velocity tilt
    cdir_64 = tmp_path / "cachedb_64"
    cdir_64.mkdir(exist_ok=True)
    spcx_dense_watcher = [[now_ms_w3 - (30 - i) * 3600_000, 148.58 - i * 0.25] for i in range(31)]
    (cdir_64 / "cache_watcher_all_price.json").write_text(
        json.dumps({"items": {"SPCX": spcx_dense_watcher}}),
        encoding="utf-8",
    )
    s24_64, v7d_64 = load_symbol_price_series(str(cdir_64), "SPCX_US_EQ")
    assert len(s24_64) >= 24
    assert len(v7d_64) >= 24
    pt_64 = {"SPCX_US_EQ": {"price": 138.36}}
    enrich_24h_and_microstructure(str(cdir_64), str(tmp_path), pt_64, {}, {}, {}, now_ms_w3 / 1000.0)
    assert pt_64["SPCX_US_EQ"].get("poc_px") is not None
    assert pt_64["SPCX_US_EQ"].get("slope_15m") is not None
    assert pt_64["SPCX_US_EQ"].get("vol_1h_pct") is not None
    (cdir_64 / "venue_balances_state.json").write_text(
        json.dumps({"pos:trading212:SPCX_US_EQ": {"symbol": "SPCX_US_EQ", "price": 138.36, "qty": 1.0}}),
        encoding="utf-8",
    )
    pt_spcx_64 = collect_pretrade_telemetry("SPCX_US_EQ", cachedb_dir=str(cdir_64))
    assert "24h Multi-Horizon Price Evolution" in pt_spcx_64
    assert "24h Volume Profile (POC / Value Area)" in pt_spcx_64
    assert "7d Price Range & Swing Structure" in pt_spcx_64

    # Bugs 66, 67 & 68:
    # - Bug 66: Market-closed flat runs (>=4 identical prices) in cache_watcher_all_price.json are collapsed for equities so RTH horizons, slopes, vol_1h_pct, and TPO POC are not zeroed/distorted by overnight/post-close duplicate ticks
    # - Bug 67: collect_pretrade_telemetry reads cache_kalman_trend.json (as well as cache_kalman_price.json)
    cdir_66 = tmp_path / "cachedb_66"
    cdir_66.mkdir(exist_ok=True)
    spcx_flat_runs = (
        [[now_ms_w3 - (28 * 12 - i) * 300_000, 136.3699951171875] for i in range(180)]
        + [[now_ms_w3 - (13 * 12 - i) * 300_000, 140.50 + (i % 7) * 0.15 + i * 0.01] for i in range(78)]
        + [[now_ms_w3 - (6 * 12 - i) * 300_000, 141.30999755859375] for i in range(72)]
    )
    (cdir_66 / "cache_watcher_all_price.json").write_text(
        json.dumps({"items": {"SPCX": spcx_flat_runs}}),
        encoding="utf-8",
    )
    (cdir_66 / "cache_kalman_trend.json").write_text(
        json.dumps({"items": {"SPCX": {"kalman_vel": -0.02388, "kalman_vel_std": 0.49, "kalman_trend": "FLAT"}}}),
        encoding="utf-8",
    )
    pt_66 = {"SPCX_US_EQ": {"price": 138.36}}
    enrich_24h_and_microstructure(str(cdir_66), str(tmp_path), pt_66, {}, {}, {}, now_ms_w3 / 1000.0)
    assert pt_66["SPCX_US_EQ"].get("chg_15m_pct") != 0.0
    assert pt_66["SPCX_US_EQ"].get("slope_15m") != 0.0
    assert (pt_66["SPCX_US_EQ"].get("vol_1h_pct") or 0.0) > 0.0
    assert (pt_66["SPCX_US_EQ"].get("poc_px") or 0.0) > 140.0
    assert pt_66["SPCX_US_EQ"].get("low_24h") == 136.37
    pt_spcx_66 = collect_pretrade_telemetry("SPCX_US_EQ", cachedb_dir=str(cdir_66))
    assert "15m=+0.00%, 30m=+0.00%, 1h=+0.00%" not in pt_spcx_66.get("24h Multi-Horizon Price Evolution", "")
    assert "POC=$136." not in pt_spcx_66.get("24h Volume Profile (POC / Value Area)", "")

    # Wave 11: Discrepancies AN, AP, AQ
    # - Discrepancy AP: grizzly bear attack headline with "killed" excluded from [HIGH-SEV] crisis keywords
    from intelligence.macro.lexicon_ssot import filter_valid_severity_keywords

    assert filter_valid_severity_keywords(
        ["attack"],
        title="Mother and daughter both killed in grizzly bear attack, Canadian authorities confirm - The Guardian",
    ) == ()

    # - Discrepancy AQ: zero-volume interior synthetic spike bar filtered out in _extract_sorted_ts_px and compute_volume_profile_metrics
    rgnt_series = [
        [now_ms_w3 - 40 * 60_000, 1.35, 210.0],
        [now_ms_w3 - 35 * 60_000, 1.355, 180.0],
        [now_ms_w3 - 30 * 60_000, 1.36, 300.0],
        [now_ms_w3 - 26 * 60_000, 1.36, 144.0],
        [now_ms_w3 - 16 * 60_000, 1.2635, 0.0],  # zero-volume synthetic outlier bar (-7.1% spike)
        [now_ms_w3 - 11 * 60_000, 1.355, 439.0],
        [now_ms_w3, 1.32, 0.0],
    ]
    rgnt_m24 = compute_24h_price_metrics(rgnt_series)
    assert rgnt_m24["low_24h"] == 1.32
    assert rgnt_m24["chg_15m_pct"] < 0.0

    # - Discrepancy AN: BTCUSDC HIGHER_LOW_DOUBLE_BOTTOM preserved during 1h support retest (cur_px >= recent_low > prior_low * 1.002)
    btc_hl_retest = [
        [now_ms_w3 - 300 * 60_000, 80455.0],
        [now_ms_w3 - 240 * 60_000, 78950.0],
        [now_ms_w3 - 180 * 60_000, 79600.0],
        [now_ms_w3 - 45 * 60_000, 80200.0],
        [now_ms_w3 - 20 * 60_000, 79890.07],
        [now_ms_w3, 79989.95],
    ]
    btc_m24_hl = compute_24h_price_metrics(btc_hl_retest)
    assert btc_m24_hl.get("bottom_structure_6h") == "HIGHER_LOW_DOUBLE_BOTTOM"
    assert btc_m24_hl.get("higher_low_vs_prior_pct") == 1.19

    # - Live tape_15m / chg_1m..15m sync in recalculate_live_price_geometry, 5m-bar slope_15m alignment, and ask_absorbing spot_perp_flow_regime symmetry
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_microstructure_cvd
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _compute_short_slopes_from_series,
        recalculate_live_price_geometry,
    )

    p_hype = {
        "price": 83.57,
        "chg_1m_pct": 0.02,
        "chg_3m_pct": 0.14,
        "chg_5m_pct": 0.01,
        "chg_15m_pct": 0.42,
        "recent_tape_15m": ["-15m:83.218", "-10m:83.4275", "-5m:83.559", "-3m:83.4535", "-1m:83.5525", "now:83.57"],
        "low_15m": 83.218,
        "high_15m": 83.57,
    }
    recalculate_live_price_geometry(p_hype, 83.4905)
    assert p_hype["recent_tape_15m"][-1] == "now:83.4905"
    assert p_hype["chg_1m_pct"] == -0.07
    assert p_hype["chg_5m_pct"] == -0.08
    assert p_hype["chg_15m_pct"] == 0.33

    nvda_5m_series = [
        [now_ms_w3 - (12 - i) * 300_000 - 150_000, 203.0 - i * 0.1]
        for i in range(9)
    ] + [
        [now_ms_w3 - 1050_000, 202.09],
        [now_ms_w3 - 750_000, 200.8999],
        [now_ms_w3 - 450_000, 201.345],
        [now_ms_w3 - 150_000, 200.99],
        [now_ms_w3, 200.99],
    ]
    nvda_m24 = compute_24h_price_metrics(nvda_5m_series)
    nvda_slopes = _compute_short_slopes_from_series(nvda_5m_series)
    assert nvda_m24["chg_15m_pct"] < 0.0
    assert nvda_slopes["slope_15m"] < 0.0
    assert (nvda_slopes.get("growth_15m") or 0.0) <= 0.0

    ada_m = {"cvd_15m_usd": 37300.0, "cvd_5m_usd": -130100.0, "cvd_1h_usd": -845800.0, "ob_imbalance": 0.62, "bid_depth_usd": 310000.0, "ask_depth_usd": 190000.0}
    ada_p = {"price": 0.195748, "chg_15m_pct": -0.46, "vol_15m_usd": 150000.0}
    enrich_microstructure_cvd(ada_m, ada_p)
    assert ada_m["cvd_regime"] == "ask_absorbing"
    assert ada_m["spot_perp_flow_regime"] != "SYNCHRONIZED_SPOT_PERP_BUYING"

    # Wave 12: Discrepancies AR, AS, AT, AU
    from datetime import datetime, timezone
    from intelligence.macro.calendar_ssot import resolve_macro_economic_calendar
    from intelligence.sentiment.evaluators.price_series_evaluator import apply_live_24trend_windows, apply_short_trend_item

    # AR: At 23:20 EET (21:20 UTC on a Wednesday in standard time), US_WALL_STREET_CLOSE (23:00 EET) is no longer ACTIVE VOLATILITY WINDOW
    cal_ar = resolve_macro_economic_calendar(datetime(2026, 11, 4, 21, 20, 0, tzinfo=timezone.utc).timestamp())
    assert cal_ar["session_key"] == "ASIA_OVERNIGHT_CALM"
    assert "US Wall Street Cash Close" not in cal_ar["next_high_impact_event"]

    # AS: apply_short_trend_item / apply_live_24trend_windows fall back to non-zero 90s/222s/900s gradients when primary window is 0.0
    ada_entry_as: dict = {}
    apply_short_trend_item(
        ada_entry_as,
        {"gradient_recent": 0.0, "gradient_recent_fast": 0.0, "epsilon": 1.6e-05, "trend": 0, "slope": -3.19e-07},
    )
    apply_live_24trend_windows(
        ada_entry_as,
        {
            "90": {"slope_full": 6.46e-06, "gradient_recent_fast": 6.46e-06, "gradient_recent": 6.46e-06},
            "222": {"slope_full": 0.0, "gradient_recent_fast": 0.0, "gradient_recent": 0.0},
            "900": {"slope_full": 2.83e-06, "gradient_recent": 2.83e-06},
        },
    )
    assert abs(ada_entry_as["gradient_recent_fast"] - 6.46e-06) < 1e-10
    assert abs(ada_entry_as["gradient_recent"] - 6.46e-06) < 1e-10

    # AT & AU: T212 cached equity position adopts fresh (<20m) post-market bar even when venue_balances_state.json mtime is newer;
    # and when post-RTH price differs by >0.5% (e.g. RGNT_US_EQ $1.31 vs $1.36), m24 horizons, slopes, and kalman_vel reflect $1.31
    cdir_at = tmp_path / "cachedb_at"
    (cdir_at / "prices").mkdir(parents=True, exist_ok=True)
    (cdir_at / "state").mkdir(parents=True, exist_ok=True)
    now_at = 1_793_827_200.0
    nvda_bars_at = [
        [int((now_at - (32 - i) * 300) * 1000), round(195.0 - i * 0.1, 4), 1000.0]
        for i in range(30)
    ] + [[int((now_at - 300) * 1000), 191.51, 1200.0]]
    rgnt_bars_au = [
        [int((now_at - (30 - i) * 300 - 1800) * 1000), 1.36, 500.0]
        for i in range(31)
    ]
    rgnt_bars_au[0][1] = 1.41
    (cdir_at / "prices" / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": nvda_bars_at}}),
        encoding="utf-8",
    )
    (cdir_at / "prices" / "cache_24_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": rgnt_bars_au}}),
        encoding="utf-8",
    )
    (cdir_at / "state" / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:t212:NVDA_US_EQ": {"symbol": "NVDA_US_EQ", "free": 0.032, "price": 191.95, "ts": now_at - 1800.0},
                "pos:t212:RGNT_US_EQ": {"symbol": "RGNT_US_EQ", "free": 122.51, "price": 1.31, "ts": now_at - 1800.0},
            }
        ),
        encoding="utf-8",
    )
    pt_at: dict = {}
    enrich_24h_and_microstructure(str(cdir_at), str(tmp_path), pt_at, {}, {}, {}, now_at)
    assert pt_at["NVDA_US_EQ"]["price"] == 191.51
    assert pt_at["RGNT_US_EQ"]["price"] == 1.31
    assert pt_at["RGNT_US_EQ"]["chg_15m_pct"] < 0.0
    assert pt_at["RGNT_US_EQ"]["slope_3_7m"] < 0.0
    assert pt_at["RGNT_US_EQ"]["kalman_vel"] < 0.0

    # - Exclude first-person reader-advice columns, suppress non-zero growth_15m when slope_15m == 0 on 5m-bar series,
    #   and clamp LONG_SWEEP_MAGNET_DOWN to BALANCED_LIQ_MAGNET during aggressive_buying + SYNCHRONIZED_SPOT_PERP_BUYING
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item
    from intelligence.sentiment.evaluators.microstructure_evaluator import compute_liquidation_clusters

    assert is_excluded_reference_news_item(
        "‘It’s good to be old’: I’ve been invited to a dozen free meals that turn into annuity sales pitches. Should I stop going?",
        "MarketWatch.com - Top Stories",
    ) is True
    assert is_excluded_reference_news_item(
        "‘I have no financial safety net’: I’m 66, rely on Social Security and have $500 in savings. What politician speaks for me?",
        "MarketWatch.com - Top Stories",
    ) is True

    rgnt_flat_15m = [
        [now_ms_w3 - (12 - i) * 300_000, 1.37]
        for i in range(9)
    ] + [
        [now_ms_w3 - 900_000, 1.32],
        [now_ms_w3 - 600_000, 1.36],
        [now_ms_w3, 1.32],
    ]
    rgnt_flat_slopes = _compute_short_slopes_from_series(rgnt_flat_15m)
    assert rgnt_flat_slopes["slope_15m"] == 0.0
    assert "growth_15m" not in rgnt_flat_slopes

    tao_liq = compute_liquidation_clusters(
        {"price": 234.2, "vol_1h_pct": 0.7381, "kalman_vel": 0.08066, "chg_15m_pct": 0.17, "val_px": 219.6, "vah_px": 235.77, "low_24h": 217.6, "high_24h": 239.8},
        {"oi_usd": 53_372_124.26, "whale_long_pct": 0.7639, "bid_wall_px": 233.6, "ask_wall_px": 234.6, "taker_buy_sell_ratio": 1.0172, "ob_imbalance": 0.586, "cvd_regime": "aggressive_buying", "spot_perp_flow_regime": "SYNCHRONIZED_SPOT_PERP_BUYING"},
    )
    assert tao_liq["liq_magnet_bias"] != "LONG_SWEEP_MAGNET_DOWN"

    # Bugs 69 & 71:
    # - Bug 69: collect_pretrade_telemetry applies BTCUSDC ob_mid_price geometry to btc_m24 and includes 7d in vs_btc(...)
    # - Bug 71: apply_live_24trend_windows falls back growth_90s (<5e-8) to non-zero 90s slope_full (>=5e-8) so g_coeff(90s=...) is not +0.00000
    ada_71: dict = {}
    apply_live_24trend_windows(
        ada_71,
        {"90": {"slope_full": 1.5096658654566196e-06, "growth_coefficient": 4.599450542363896e-08}},
    )
    assert _fmt_slope_val(ada_71["growth_90s"]) == "+1.51e-06"
    assert _fmt_slope_val(4.97e-10) == "+0.00000"
    cdir_69 = tmp_path / "cachedb_69"
    cdir_69.mkdir(exist_ok=True)
    btc_24_69 = [[int((now_at - (24 * 12 - i) * 300) * 1000), 77000.0] for i in range(24 * 12)] + [
        [int(now_at * 1000), 76990.51]
    ]
    tao_24_69 = [[int((now_at - (24 * 12 - i) * 300) * 1000), 224.0] for i in range(24 * 12)] + [
        [int(now_at * 1000), 223.40]
    ]
    btc_365_69 = [[int((now_at - (7 * 24 - i) * 3600) * 1000), 80850.0] for i in range(7 * 24)] + [
        [int(now_at * 1000), 76990.51]
    ]
    tao_365_69 = [[int((now_at - (7 * 24 - i) * 3600) * 1000), 257.70] for i in range(7 * 24)] + [
        [int(now_at * 1000), 223.40]
    ]
    (cdir_69 / "cache_24_price_BTCUSDC.json").write_text(json.dumps({"items": {"BTCUSDC": btc_24_69}}), encoding="utf-8")
    (cdir_69 / "cache_24_price_TAOUSDC.json").write_text(json.dumps({"items": {"TAOUSDC": tao_24_69}}), encoding="utf-8")
    (cdir_69 / "cache_365_price_BTCUSDC.json").write_text(json.dumps({"items": {"BTCUSDC": btc_365_69}}), encoding="utf-8")
    (cdir_69 / "cache_365_price_TAOUSDC.json").write_text(json.dumps({"items": {"TAOUSDC": tao_365_69}}), encoding="utf-8")
    (cdir_69 / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps({"mid_price": 76958.975, "best_bid": 76958.97, "best_ask": 76958.98, "spread_bps": 0.0013}),
        encoding="utf-8",
    )
    (cdir_69 / "orderbook_depth_TAOUSDC.json").write_text(
        json.dumps({"mid_price": 223.40, "best_bid": 223.30, "best_ask": 223.50, "spread_bps": 8.95}),
        encoding="utf-8",
    )
    pt_tao_69 = collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cdir_69))
    rel_69 = pt_tao_69.get("Relative Strength vs BTC (24h Benchmark)", "")
    assert "7d=" in rel_69
    assert "5m=-0.22%" in rel_69

    # Wave 13: Discrepancies AV & AW
    from collections import deque
    import shadow_signals as ss_aw

    # AV: On FOMC day (2026-11-05 at 00:05 EET = 2026-11-04 22:05 UTC), upcoming_windows includes the 21:00 EET FOMC window
    cal_av = resolve_macro_economic_calendar(datetime(2026, 11, 4, 22, 5, 0, tzinfo=timezone.utc).timestamp())
    assert cal_av["fomc_today"] is True
    assert any("FOMC Rate Decision" in w for w in cal_av["upcoming_windows"])

    # AW: KalmanStateTracker.update warm-starts from a collections.deque big_prices (PriceWindow.prices)
    tracker_aw = ss_aw.KalmanStateTracker(state_path=None, compute_vol=False)
    deque_prices_aw = deque(220.0 + i * 0.005 for i in range(3600))
    k_aw = tracker_aw.update("TAOUSDC", 1_793_830_000.0, 238.0, epsilon=0.05, big_prices=deque_prices_aw, big_sample_rate=1.0)
    assert k_aw["kalman_vel"] > 0.0
    assert k_aw["kalman_vel_std"] < 5.0

    # Wave 14: Bugs 74-78
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item
    from intelligence.sentiment.evaluators.price_series_evaluator import apply_short_trend_item

    # Bug 74: apply_short_trend_item falls back sub-threshold slope_full (<5e-8) to gradient_recent and sub-threshold growth_coefficient to slope_full
    entry_74: dict = {}
    apply_short_trend_item(
        entry_74,
        {
            "dynamic": {
                "222": {"slope_full": -2.63e-8, "gradient_recent": -2.61e-7, "growth_coefficient": -1.07e-6},
                "9000": {"slope_full": 4.00e-7, "gradient_recent": 3.50e-7, "growth_coefficient": 1.20e-8},
            }
        },
    )
    assert abs(entry_74["slope_3_7m"] - (-2.61e-7)) < 1e-12
    assert abs(entry_74["growth_2_5h"] - 4.00e-7) < 1e-12

    # Bug 76: is_excluded_reference_news_item filters first-person plural ("We're...") reader-advice columns
    assert is_excluded_reference_news_item(
        "‘It seems bizarre’: We’re in our 80s with $950,000 in traditional IRAs. Why is our adviser pushing Roth conversions?"
    ) is True

    # Bug 77: compute_liquidation_clusters clamps LONG_SWEEP_MAGNET_DOWN when slope_15m > 0, aggressive_buying, SYNCHRONIZED_SPOT_PERP_BUYING, and strong taker/OB imbalance
    liq_77 = compute_liquidation_clusters(
        {"price": 0.41585, "chg_15m_pct": 0.46, "kalman_vel": -0.011, "slope_15m": 1.43e-6, "volatility_1h_pct": 0.25},
        {
            "whale_long_pct": 0.74,
            "taker_buy_sell_ratio": 1.2659,
            "ob_imbalance": 0.598,
            "cvd_regime": "aggressive_buying",
            "spot_perp_flow_regime": "SYNCHRONIZED_SPOT_PERP_BUYING",
            "oi_usd": 200_000_000.0,
        },
    )
    assert liq_77["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    # Wave 14: Bugs 1-6 (Sibling Long Trend Inheritance, Weibull est_rem, Cross-Venue Spot CVD, Flow-Venue OI vs L1 OI, Fill-Time Execution Status, and Kalman/Short-Trend Vol Sync)
    from intelligence.internal.state.survival import estimate_remaining_trend_hours
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_microstructure
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        enrich_cross_venue_trends,
        enrich_microstructure_cvd,
        reconcile_sibling_flow_and_liq_regimes,
    )
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section
    from providers.execution_audit import (
        build_last_cycle_execution_status,
        extract_intent_notional,
        reconcile_intents_from_execution_audit,
    )

    # Bug 1 & Bug 2: HYPEUSDC inherits HYPEUSD long_trend when HYPEUSDC is None in priceanalysis.json, and est_rem uses Weibull conditional remaining life
    pt_b12 = {"HYPEUSDC": {"price": 86.25}, "HYPEUSD": {"price": 86.25}}
    pa_b12 = {
        "timestamp": 1_793_830_000,
        "items": {
            "HYPEUSDC": None,
            "HYPEUSD": {"direction": "up", "duration_seconds": 240.0 * 3600.0, "estimated_future_hours": 120.0, "timestamp": 1_793_830_000},
        },
    }
    lt_b12 = {
        "timestamp": 1_793_820_000,
        "items": {
            "HYPEUSDC": [{"direction": "up", "duration_seconds": 392.0 * 3600.0, "estimated_future_hours": 196.0, "timestamp": 1_793_820_000}],
            "HYPEUSD": [{"direction": "up", "duration_seconds": 240.0 * 3600.0, "estimated_future_hours": 120.0, "timestamp": 1_793_830_000}],
        },
    }
    tt_b12 = {"HYPEUSD": {"median_d": 3.6, "p90_d": 6.8, "p_cont": {"10": 0.25}}}
    enrich_cross_venue_trends(pt_b12, {}, {}, lt_b12, pa_b12, tt_b12, {})
    assert pt_b12["HYPEUSDC"]["long_trend_hours"] == 240.0
    assert pt_b12["HYPEUSD"]["long_trend_hours"] == 240.0
    assert pt_b12["HYPEUSDC"]["long_trend_est_rem_hours"] == pt_b12["HYPEUSD"]["long_trend_est_rem_hours"]
    assert pt_b12["HYPEUSDC"]["long_trend_est_rem_hours"] < 80.0
    assert estimate_remaining_trend_hours(240.0, median_d=3.6, p90_d=6.8, p_cont={"10": 0.25}) < 80.0

    # Bug 3: Cross-venue Spot CVD unification across HYPEUSDC and HYPEUSD prevents opposite-sign synthetic spot_cvd_15m_usd
    micro_b3 = {
        "HYPEUSDC": {
            "ob_imbalance": 0.441,
            "bid_depth_usd": 620_000.0,
            "ask_depth_usd": 785_000.0,
            "taker_buy_vol_usd": 1_500_000.0,
            "taker_sell_vol_usd": 1_915_600.0,
        },
        "HYPEUSD": {
            "ob_imbalance": 0.564,
            "bid_depth_usd": 650_000.0,
            "ask_depth_usd": 502_000.0,
            "taker_buy_vol_usd": 1_500_000.0,
            "taker_sell_vol_usd": 1_915_600.0,
        },
    }
    enrich_microstructure_cvd(micro_b3["HYPEUSDC"], {"price": 86.25, "chg_15m_pct": -0.1})
    enrich_microstructure_cvd(micro_b3["HYPEUSD"], {"price": 86.25, "chg_15m_pct": -0.1})
    reconcile_sibling_flow_and_liq_regimes(micro_b3, {"HYPEUSDC": {"price": 86.25, "chg_15m_pct": -0.1}, "HYPEUSD": {"price": 86.25, "chg_15m_pct": -0.1}})
    assert micro_b3["HYPEUSDC"]["spot_cvd_15m_usd"] == micro_b3["HYPEUSD"]["spot_cvd_15m_usd"]

    # Bug 4: Flow-venue OI ($403.2M) vs Native L1 OI ($1.86B) in load_symbol_microstructure and Section 3B formatting
    cdir_b4 = tmp_path / "cachedb_b4"
    cdir_b4.mkdir(exist_ok=True)
    m_b4 = load_symbol_microstructure(
        str(cdir_b4),
        "HYPEUSDC",
        ob={"symbol": "HYPEUSDC", "mid_price": 86.25, "imbalance_ratio": 0.52, "bid_depth_usd": 500_000.0, "ask_depth_usd": 460_000.0},
        deriv={"open_interest_usd": 1_863_328_553.78, "funding_rate": 0.0000125},
        whale={"open_interest_usd": 403_209_711.39, "top_traders_long_pct": 0.6909, "taker_buy_vol_usd": 1_730_000.0, "taker_sell_vol_usd": 1_520_000.0},
    )
    assert m_b4 is not None
    assert m_b4["oi_usd"] == 403_209_711.39
    assert m_b4["native_l1_oi_usd"] == 1_863_328_553.78
    sec3b_b4 = "\n".join(_format_microstructure_section({"HYPEUSDC": m_b4}, {"HYPEUSDC": {"price": 86.25}}))
    assert "oi_usd=403209711.39 (l1_oi=$1.86B)" in sec3b_b4

    # Bug 5: Fill-time age_min & 60m terminal cutoff in build_last_cycle_execution_status + filled_at backfill in reconcile_intents_from_execution_audit
    aud_dir_b5 = tmp_path / "audit_b5"
    aud_dir_b5.mkdir(exist_ok=True)
    (aud_dir_b5 / "execution_audit_2026-11-04.jsonl").write_text(
        json.dumps({
            "ts": 1_793_821_786.0,
            "event": "order_terminal",
            "intent_id": "intent_1793805953_buy_hypeusdc",
            "order_id": "594571495032",
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "side": "BUY",
            "venue_status": "FILLED",
            "lifecycle_status": "filled",
            "filled_qty": 9.3,
            "price": 86.0,
            "cost": 799.8,
        }) + "\n",
        encoding="utf-8",
    )
    intents_b5 = [
        {
            "intent_id": "intent_1793805953_buy_hypeusdc",
            "timestamp": 1_793_805_953.8,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "limit_price": 86.0,
            "reference_price": 86.44,
            "suggested_notional_usd": 800.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": 594571495032, "status": "FILLED", "executedQty": "9.3", "price": 86.0, "reconciled_terminal": True},
        }
    ]
    assert reconcile_intents_from_execution_audit(intents_b5, str(aud_dir_b5)) is True
    assert intents_b5[0]["execution_payload"]["filled_at"] == 1_793_821_786.0
    assert reconcile_intents_from_execution_audit(intents_b5, str(aud_dir_b5)) is False
    # 12 minutes after fill (even though placed 4.6 hours earlier): reported with age_min=12.0 and placed_age_min=275.9
    st_fresh_b5 = build_last_cycle_execution_status(intents_b5, "crypto", 1_793_821_786.0 + 720.0, extract_intent_notional)
    assert isinstance(st_fresh_b5, dict)
    assert st_fresh_b5["age_min"] == 12.0
    assert st_fresh_b5["placed_age_min"] == 275.9
    # 2.5 hours after fill: expired from last_cycle_execution_status (>60m terminal cutoff)
    assert build_last_cycle_execution_status(intents_b5, "crypto", 1_793_821_786.0 + 9000.0, extract_intent_notional) is None

    # Bugs 79-82: Yahoo partial-bar duplicate deduplication in _extract_sorted_ts_px, L1 vs Flow OI cycle_delta guard, and pre-trade OI/sibling 24h series sync
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _compute_short_slopes_from_series,
        compute_24h_price_metrics,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import update_microstructure_cycle_deltas

    # Bug 79: 5-minute equity series with trailing partial sub-5m duplicate Yahoo quote drops the duplicate bar
    base_ms_79 = 1_793_829_000_000
    eq_series_79 = [[base_ms_79 + i * 300_000, 134.0 + i * 0.08, 100.0] for i in range(13)]
    eq_series_79[-2][1] = 134.78
    eq_series_79[-1][1] = 135.05
    # Append trailing partial sub-5m duplicate quote 260s after last 5-minute bar
    eq_series_79.append([eq_series_79[-1][0] + 260_000, 135.05, 0.0])
    slopes_79 = _compute_short_slopes_from_series(eq_series_79)
    assert slopes_79["slope_3_7m"] > 0.0
    m24_79 = compute_24h_price_metrics(eq_series_79)
    assert "chg_3m_pct" not in m24_79
    assert m24_79["chg_5m_pct"] == 0.2

    # Bug 80: update_microstructure_cycle_deltas compares native_l1_oi_usd against prior L1 oi_usd instead of emitting -77.96%
    cdir_b80 = tmp_path / "cachedb_b80"
    cdir_b80.mkdir(exist_ok=True)
    (cdir_b80 / "microstructure_cycle_history.json").write_text(
        json.dumps({"HYPEUSDC": [{"ts": 1_793_830_000.0, "oi_usd": 1_901_470_497.0}]}),
        encoding="utf-8",
    )
    micro_b80 = {"HYPEUSDC": {"oi_usd": 419_168_669.47, "native_l1_oi_usd": 1_901_470_497.0}}
    update_microstructure_cycle_deltas(str(cdir_b80), micro_b80, now_ts=1_793_830_900.0)
    assert micro_b80["HYPEUSDC"]["cycle_delta"]["oi_delta_pct"] == 0.0

    # Bug 84: _format_portfolio_section omits last_cycle_execution_status when all entries are >30m OPEN_RESTING already in open_limit_orders
    from orchestratorOS.supervisor.prompt_sections import _format_market_prices_section, _format_portfolio_section

    sec3c_b84 = "\n".join(
        _format_portfolio_section(
            {
                "open_limit_orders": [
                    {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "order_id": "101", "price": 78620.0, "qty": 0.01, "filled_qty": 0.0, "status": "NEW"},
                    {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "order_id": "721398249", "price": 228.5, "qty": 3.5, "filled_qty": 0.0, "status": "NEW"},
                ],
                "last_cycle_execution_status": [
                    {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "status": "OPEN_RESTING", "order_id": "101", "age_min": 512.9},
                    {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "status": "OPEN_RESTING", "order_id": "721398249", "age_min": 512.8},
                ],
            },
            asset_scope="crypto",
            include_macro_benchmarks=False,
        )
    )
    assert "last_cycle_execution_status:" not in sec3c_b84

    # Bug 85: _format_market_prices_section does not leak sibling sec_poc_px/lvn_px onto a symbol with its own UNIMODAL Volume Profile
    sec3a_b85 = "\n".join(
        _format_market_prices_section(
            {
                "HYPEUSDC": {
                    "price": 83.7985,
                    "poc_px": 84.6045,
                    "val_px": 82.6535,
                    "vah_px": 85.9865,
                    "va_pos_pct": 34.4,
                    "va_regime": "INSIDE_VA_LOWER",
                    "profile_shape": "UNIMODAL",
                    "vp_source": "TPO",
                },
                "HYPEUSD": {
                    "price": 83.785,
                    "poc_px": 84.6844,
                    "sec_poc_px": 86.6434,
                    "lvn_px": 86.0215,
                    "val_px": 82.6498,
                    "vah_px": 86.1583,
                    "va_pos_pct": 32.4,
                    "va_regime": "INSIDE_VA_LOWER",
                    "profile_shape": "BIMODAL",
                    "vp_source": "TPO",
                },
            }
        )
    )
    assert "sec_POC=86.6434" not in sec3a_b85
    assert "LVN=86.0215" not in sec3a_b85

    # Bug 86: T212 position price seeded from cache uses pos_ts (not wall-clock now_ms) when appended to 24h series
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure

    cdir_b86 = tmp_path / "cachedb_b86"
    (cdir_b86 / "prices").mkdir(parents=True, exist_ok=True)
    base_eq_ms = 1_775_851_200_000  # 20:00 UTC close
    pts_eq_b86 = [[base_eq_ms - (15 - i) * 60_000, 188.0 + i * 0.04] for i in range(16)]
    (cdir_b86 / "prices" / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": pts_eq_b86}}),
        encoding="utf-8",
    )
    (cdir_b86 / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:t212:NVDA_US_EQ": {
                    "symbol": "NVDA_US_EQ",
                    "free": 6.45,
                    "price": 188.63,
                    "ts": (base_eq_ms / 1000.0) + 25.0,
                }
            }
        ),
        encoding="utf-8",
    )
    pt_b86_a: dict = {}
    pt_b86_b: dict = {}
    enrich_24h_and_microstructure(str(cdir_b86), str(cdir_b86), pt_b86_a, {}, {}, {}, now_ts=(base_eq_ms / 1000.0) + 60.0)
    enrich_24h_and_microstructure(str(cdir_b86), str(cdir_b86), pt_b86_b, {}, {}, {}, now_ts=(base_eq_ms / 1000.0) + 900.0)
    assert pt_b86_a["NVDA_US_EQ"]["slope_15m"] == pt_b86_b["NVDA_US_EQ"]["slope_15m"]

    # Bug 87: harvest_log_anomalies_from_dirs suppresses recovered rsync errors in sync_wsl_cache.log
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    ldir_b87 = tmp_path / "logs_b87"
    ldir_b87.mkdir(exist_ok=True)
    t_err_87 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 3600))
    t_ok_87 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 600))
    (ldir_b87 / "sync_wsl_cache.log").write_text(
        f"[{t_err_87}] [rsync_cachedb] Syncing prod:/home/predut/mptrade/cachedb/ -> /home/mariusp/mptrade/cachedb/\n"
        f"rsync error: unexplained error (code 255) at io.c(232) [Receiver=3.2.7]\n"
        f"[{t_err_87}] [rsync_cachedb] ERROR: rsync (prod:/home/predut/mptrade/cachedb/ -> /home/mariusp/mptrade/cachedb/) failed with code 255\n"
        f"[{t_ok_87}] [rsync_cachedb] Syncing prod:/home/predut/mptrade/cachedb/ -> /home/mariusp/mptrade/cachedb/\n"
        f"sent 2,922 bytes  received 16,668,782 bytes  952,668.80 bytes/sec\n"
        f"total size is 2,340,399,369  speedup is 140.38\n",
        encoding="utf-8",
    )
    anom_b87 = harvest_log_anomalies_from_dirs([str(ldir_b87)], lookback_hours=4.0)
    assert anom_b87.error_count == 0
    assert anom_b87.persistent_errors == []

    # Bug 88: reconcile_venue_cycles_with_recent_orders updates last_exit_kind='REDUCE' and dca_buys=2 on HYPEUSD
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        extract_symbol_recent_fills,
        reconcile_venue_cycles_with_recent_orders,
    )

    kr_cyc_b88 = {
        "HYPEUSD": {
            "qty": 6.492657,
            "cost_usd": 556.83,
            "avg_cost_px": 85.762381,
            "entry_price": 91.209992,
            "last_buy_price": 84.0,
            "last_sell_price": 93.299997,
            "last_exit_kind": "TP",
            "dca_buys": 1,
            "_entry_ts_ms": 1_793_124_116_610,
            "_last_sell_ts_ms": 1_793_368_889_035,
        }
    }
    v_ro_b88 = {
        "kraken": {
            "HYPEUSD": {
                "last_sell": {"price": 87.120002, "qty": 2.295157, "ts_ms": 1_793_721_326_834, "actor": "AI"},
                "recent_buys": [
                    {"price": 91.209992, "qty": 2.294924, "ts_ms": 1_793_666_393_168, "actor": "BOT"},
                    {"price": 84.0, "qty": 5.952381, "ts_ms": 1_793_761_201_845, "actor": "BOT"},
                    {"price": 82.5, "qty": 1.81818, "ts_ms": 1_793_781_037_619, "actor": "AI"},
                ],
                "active_avg_buy_px": 85.158364,
                "active_cycle_rem_qty": 8.308042,
                "lifo_realized_pnl_usd": -9.39,
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        kr_cyc_b88,
        {},
        {},
        v_ro_b88,
        {("kraken", "HYPEUSD"): 8.310839},
        {"HYPEUSD": {"price": 83.75}},
    )
    assert kr_cyc_b88["HYPEUSD"]["last_sell_price"] == 87.120002
    assert kr_cyc_b88["HYPEUSD"]["last_exit_kind"] == "REDUCE"
    assert kr_cyc_b88["HYPEUSD"]["dca_buys"] == 2

    # Bug 89: extract_symbol_recent_fills preserves lifo_realized_pnl_usd across core-inventory partial sells followed by intraday scale-in BUY
    t0_b89 = 1_793_722_000_000
    fills_b89 = extract_symbol_recent_fills(
        [
            {"side": "BUY", "price": 7.98, "qty": 25.062657, "ts_ms": t0_b89},
            {"side": "BUY", "price": 7.46, "qty": 53.654952, "ts_ms": t0_b89 + 3_600_000},
            {"side": "SELL", "price": 7.57, "qty": 72.990686, "ts_ms": t0_b89 + 7_200_000},
            {"side": "SELL", "price": 7.57, "qty": 72.990686, "ts_ms": t0_b89 + 7_205_000},
            {"side": "BUY", "price": 7.48, "qty": 47.619048, "ts_ms": t0_b89 + 8_600_000},
        ]
    )
    assert fills_b89 is not None
    assert fills_b89["lifo_realized_pnl_usd"] == -4.37





def test_active_7d_vals_ohlc_override_pretrade_sibling_90s_and_stdin_log_filtering(tmp_path):
    """Verify:
    - Bug 17: enrich_24h_and_microstructure tracks active_7d_vals from _load_stitched_7d_ohlc_closes so ob_mid_m price adjustment does not overwrite chg_7d_pct with partial 1-day arch_vals_7d[0]
    - Bug 18: collect_pretrade_telemetry backfills zero slope_90s/growth_90s/trend_90s from non-diverged sibling live_<SYM>.json (e.g. ADAUSDC for ADAUSD)
    - Bug 20: harvest_log_anomalies_from_dirs ignores interactive [<stdin>:NN] lines and clears manage_agy errors after 'ui-sync: check healthy'
    """
    from types import SimpleNamespace
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    cdir = tmp_path / "cachedb"
    for sub in ("prices", "trends/24trend", "telemetry", "24trend"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)

    now_ts = 1_793_826_000.0
    now_ms = int(now_ts * 1000)

    # Partial 1.08-day archive starting at 0.184033 (+4.48% to 0.19228)
    arch_rows = [
        {"s": "ADAUSD", "i": [now_ms - int(1.08 * 86400 * 1000) + i * 3600_000, 0.184033 + i * 0.0003]}
        for i in range(26)
    ]
    (cdir / "prices" / "cache_365_price_ADAUSD.jsonl").write_text(
        "\n".join(json.dumps(r) for r in arch_rows) + "\n",
        encoding="utf-8",
    )
    s24_pts = [[now_ms - (24 - i) * 3600_000, 0.188 + i * 0.00018] for i in range(25)]
    (cdir / "prices" / "cache_24_price_ADAUSD.json").write_text(
        json.dumps({"items": {"ADAUSD": s24_pts}}),
        encoding="utf-8",
    )
    (cdir / "telemetry" / "orderbook_depth_ADAUSDC.json").write_text(
        json.dumps({"symbol": "ADAUSDC", "mid_price": 0.19228, "best_bid": 0.1922, "best_ask": 0.19236, "bid_depth_usd": 120_000.0, "ask_depth_usd": 110_000.0}),
        encoding="utf-8",
    )

    # Full 7-day Kraken OHLC closes starting at 0.199018 (-3.39% to 0.19228, high=0.206197)
    ohlc_ts = [now_ts - (168 - i) * 3600.0 for i in range(169)]
    ohlc_closes = [0.199018] + [0.206197 if i == 40 else (0.18252 if i == 120 else 0.193) for i in range(1, 168)] + [0.19225]
    dummy_inst = SimpleNamespace(
        _provider=SimpleNamespace(
            ohlc_series=lambda sym, interval: SimpleNamespace(timestamps=ohlc_ts, closes=ohlc_closes, volumes=[1000.0] * len(ohlc_ts))
        )
    )
    pt_map = {"ADAUSD": {"price": 0.19225, "trend": 0, "growth_coefficient": 0.0}}
    micro_map: dict = {}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_map, micro_map, {}, {"ADAUSD": dummy_inst}, now_ts)
    assert pt_map["ADAUSD"]["high_7d"] == 0.206197
    assert pt_map["ADAUSD"]["chg_7d_pct"] == -3.39

    # Bug 18: live_ADAUSD.json has 0.0 in window '90', while sibling live_ADAUSDC.json has non-zero slope_90s
    (cdir / "trends" / "24trend" / "live_ADAUSD.json").write_text(
        json.dumps({
            "windows": {
                "90": {"slope_full": 0.0, "growth_coefficient": 0.0, "final_trend": 0, "gradient_recent": 0.0, "current_price": 0.19225},
                "222": {"slope_full": -2.51e-05, "growth_coefficient": -1.3e-04, "final_trend": -1, "gradient_recent": -2.51e-05, "current_price": 0.19225},
                "900": {"slope_full": -5.66e-05, "growth_coefficient": -2.9e-04, "final_trend": -1, "gradient_recent": -5.66e-05, "current_price": 0.19225},
                "9000": {"slope_full": -3.19e-06, "growth_coefficient": -1.6e-05, "final_trend": 0, "gradient_recent": -3.19e-06, "current_price": 0.19225},
            }
        }),
        encoding="utf-8",
    )
    (cdir / "trends" / "24trend" / "live_ADAUSDC.json").write_text(
        json.dumps({
            "windows": {
                "90": {"slope_full": -2.46e-05, "growth_coefficient": -1.28e-04, "final_trend": -1, "gradient_recent": -2.46e-05, "current_price": 0.19228},
                "222": {"slope_full": -2.51e-05, "growth_coefficient": -1.3e-04, "final_trend": -1, "gradient_recent": -2.51e-05, "current_price": 0.19228},
            }
        }),
        encoding="utf-8",
    )
    pre_ada = collect_pretrade_telemetry("ADAUSD", cachedb_dir=str(cdir))
    assert "90s=-2.46e-05" in pre_ada["Dynamic Multi-Window Slopes"]

    # Bug 20: [<stdin>:NN] lines ignored & ui-sync: check healthy clears prior manage_agy error
    ldir = tmp_path / "logs_b20"
    ldir.mkdir(exist_ok=True)
    s_fmt = lambda ago: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - ago))  # noqa: E731
    (ldir / "app_2026-11-04.log").write_text(
        "23:56:23 [<stdin>:34] FAIL: offline/research/intelligence_backtest.py -> AttributeError: 'NoneType' object has no attribute '__dict__'\n"
        "23:56:45 [<stdin>:35] FAIL: offline/research/kraken_continuous_grid/fetch.py -> FileNotFoundError: [Errno 2] No such file or directory\n",
        encoding="utf-8",
    )
    (ldir / "manage_agy.log").write_text(
        f"[{s_fmt(1200)}] [manage_agy] ERROR: ui_sync_idle_sec / AGY_UI_SYNC_IDLE_SEC missing or invalid ('[23:53:02]')\n"
        f"[{s_fmt(300)}] [manage_agy] ui-sync: check healthy (rc=10, idle_sec=120s).\n",
        encoding="utf-8",
    )
    anom = harvest_log_anomalies_from_dirs([str(ldir)], lookback_hours=2.0)
    assert anom.error_count == 0
    assert anom.persistent_errors == []
    assert not any("<stdin>" in ev for ev in anom.critical_events)


def test_bugs_21_to_26_tape5m_geodeescalation_advisor_lows_whale_str_and_taousdc_vp(tmp_path):
    """Verify Bugs 21-26:
    - Bug 21: 5m bar series with live sub-5m points at +4.7m after last 5m bar retains chg_5m_pct and -5m in tape_15m
    - Bug 22: NewsFeedCollector preserves first_seen_ts on NewsHeadline & GeopoliticalThreatAnalyzer de-escalates ELEVATED after 90m with no fresh high-sev headlines
    - Bug 23: SentimentAdvisor computes fallback prior_6h_low and 1h_low from BTC 24h series when bottom_structure_6h is NONE
    - Bug 24: load_cached_whale_telemetry accepts string symbol 'BTCUSDC' without iterating char-by-char
    - Bug 25: harvest_log_anomalies_from_dirs suppresses recovered [kraken_cache][ws] error: ping/pong timed out when followed by subscribed ownTrades
    - Bug 26: _resolve_fallback_ohlc_fn prefers binance:<SYM> over hyperliquid:<SYM> for multi-venue USDC pairs like TAOUSDC
    """
    import time
    from types import SimpleNamespace
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics
    from intelligence.sentiment.collectors.microstructure_collector import _resolve_fallback_ohlc_fn
    from intelligence.macro.news_feed_collector import NewsFeedCollector
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer, GeopoliticalThreatAssessment
    from intelligence.macro.calendar_ssot import compute_dynamic_geopolitical_interval
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from intelligence.macro_sentiment_ssot import load_cached_whale_telemetry
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # Bug 21: 5m bars + -3m/-1m/now at +4.68m after last 5m bar
    base_ms = 1_793_800_000_000
    pts_5m = [[base_ms - (12 - i) * 300_000, 200.0 + i * 0.1] for i in range(13)]
    now_ms = base_ms + 281_000  # +4.683m after last 5m bar at base_ms
    pts_merged = pts_5m + [[now_ms - 180_000, 201.27], [now_ms - 60_000, 201.28], [now_ms, 201.29]]
    m24 = compute_24h_price_metrics(pts_merged)
    assert "chg_5m_pct" in m24
    assert any(p.startswith("-5m:") for p in m24.get("recent_tape_15m", []))
    assert any(p.startswith("-3m:") for p in m24.get("recent_tape_15m", []))

    # Bug 22: NewsFeedCollector preserves first_seen_ts & GeopoliticalThreatAnalyzer de-escalates after 90m
    cdir = tmp_path / "cachedb"
    (cdir / "macro").mkdir(parents=True, exist_ok=True)
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    now_s = time.time()
    hist_path = cdir / "macro" / "news_feed_collect.json"
    hist_path.write_text(
        json.dumps([
            {
                "title": "Ukraine raises death toll from Russian missile strike",
                "source": "BBC World",
                "pub_date": "",
                "url": "https://bbc.co.uk/news/1",
                "matched_keywords": ["missile", "strike"],
                "is_high_severity": True,
                "first_seen_ts": now_s - 7200.0,  # 120m old (>90m)
            }
        ]),
        encoding="utf-8",
    )
    nf = NewsFeedCollector(history_file=str(hist_path))
    assert nf._cache_dir() == str(cdir)
    snap = nf.get_pool_snapshot(now_ts=now_s)
    assert len(snap.headlines) == 1
    assert snap.headlines[0].first_seen_ts == now_s - 7200.0

    geo_state = cdir / "macro" / "geopolitical_eval.json"
    geo = GeopoliticalThreatAnalyzer(state_file=str(geo_state), cache_ttl_sec=16200.0)
    geo._cached_assessment = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.3,
        summary="Prior elevated tension.",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=1,
        ts=now_s - 6000.0,
    )
    geo._last_eval_ts = now_s - 6000.0
    geo._last_headlines_digest = "Ukraine raises death toll from Russian missile strike"
    itv_geo, reason_geo = compute_dynamic_geopolitical_interval(
        base_interval_sec=16200.0,
        last_eval_ts=now_s - 6000.0,
        now_ts=now_s,
        cached_assessment=geo._cached_assessment,
        news_snapshot=snap,
        headlines_changed=False,
        last_headlines_digest=geo._last_headlines_digest,
    )
    assert itv_geo == 1800.0
    assert reason_geo == "threat_deescalation_zero_high_sev_recheck"
    deesc = geo.assess(snap, now_ts=now_s)
    assert deesc.threat_level == "NORMAL"
    assert deesc.recommended_brake == "NONE"
    # Even with force_refresh=True or recent eval_ts (<30m), >=90m-aged headlines stay de-escalated to NORMAL
    deesc_forced = geo.assess(snap, force_refresh=True, now_ts=now_s)
    assert deesc_forced.threat_level == "NORMAL"
    assert deesc_forced.recommended_brake == "NONE"

    # Bug 23 & 24: load_cached_whale_telemetry string symbol & SentimentAdvisor prior_6h_low/1h_low fallback
    (cdir / "telemetry" / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps({"symbol": "BTCUSDC", "taker_buy_sell_ratio": 1.12, "open_interest_change_pct": 0.45, "divergence_regime": "accumulation"}),
        encoding="utf-8",
    )
    w_snap = load_cached_whale_telemetry(str(cdir), "BTCUSDC")
    assert isinstance(w_snap, dict) and w_snap.get("taker_buy_sell_ratio") == 1.12

    btc_24h_pts = [[int((now_s - (24 - i) * 3600) * 1000), 78000.0 + i * 100.0] for i in range(25)]
    (cdir / "prices" / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": btc_24h_pts}}),
        encoding="utf-8",
    )
    captured_prompts: list[str] = []
    dummy_gemini = SimpleNamespace(query_json=lambda prompt, **kw: (captured_prompts.append(prompt) or None))
    sa = SentimentAdvisor(gemini_client=dummy_gemini, cache_file=str(cdir / "macro" / "sentiment_eval.json"))
    sa.review(fear_greed=None, breadth=None, whale_telemetry=w_snap, force_refresh=True)
    assert captured_prompts
    assert "prior_6h_low=$0.00" not in captured_prompts[0]
    assert "Whales & Derivatives:" in captured_prompts[0]

    # Bug 25: Kraken-cache ws ping/pong timeout followed by subscribed ownTrades is suppressed
    ldir = tmp_path / "logs_b25"
    ldir.mkdir(exist_ok=True)
    t_err = time.strftime("%H:%M:%S", time.localtime(now_s - 1800))
    t_rec = time.strftime("%H:%M:%S", time.localtime(now_s - 1795))
    (ldir / "Kraken-cache.log").write_text(
        f"[{t_err}] [kraken_cache][ws] error: ping/pong timed out\n" * 3
        + f"[{t_rec}] [kraken_cache][ws] subscribed ownTrades\n",
        encoding="utf-8",
    )
    anom_25 = harvest_log_anomalies_from_dirs([str(ldir)], lookback_hours=2.0)
    assert anom_25.error_count == 0
    assert anom_25.persistent_errors == []

    # Bug 26: _resolve_fallback_ohlc_fn prefers binance:TAOUSDC over hyperliquid:TAOUSDC
    bin_fn = lambda s, i: "binance_ohlc"  # noqa: E731
    hl_fn = lambda s, i: "hl_ohlc"  # noqa: E731
    inst_map = {
        "TAOUSDC": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=hl_fn)),
        "binance:TAOUSDC": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=bin_fn)),
        "hyperliquid:TAOUSDC": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=hl_fn)),
    }
    assert _resolve_fallback_ohlc_fn("TAOUSDC", inst_map) is bin_fn

    # Bug 29: parse_cited_taker_ratio handles parenthetical '(ratio 0.60, 1h CVD -$25.46M)' and ranks 1h above 5m
    from intelligence.macro.calendar_ssot import parse_cited_taker_ratio

    live_summary = (
        "and a sharp 5m taker buy resurgence (ratio 1.64, CVD +$9.25M). "
        "Despite short-term bid absorption, aggregate 1h taker flow remains net negative (ratio 0.60, 1h CVD -$25.46M)"
    )
    assert parse_cited_taker_ratio(live_summary) == 0.60

    # Bug 30b: extract_symbol_recent_fills clears lifo_realized_pnl_usd when a full exit of >7d-old lots is followed by a new cycle BUY
    from intelligence.sentiment.evaluators.portfolio_evaluator import extract_symbol_recent_fills

    tao_fills = extract_symbol_recent_fills(
        [
            {"side": "BUY", "price": 272.94, "qty": 1.19073, "ts_ms": 1791889520952},
            {"side": "BUY", "price": 264.7238, "qty": 0.58748, "ts_ms": 1792030636688},
            {"side": "BUY", "price": 256.0432, "qty": 1.26932, "ts_ms": 1792072494450},
            {"side": "SELL", "price": 226.016729, "qty": 7.40379, "ts_ms": 1793319677319},
            {"side": "BUY", "price": 235.741094, "qty": 3.70189, "ts_ms": 1793331231572},
        ],
        symbol="TAOUSD",
    )
    assert tao_fills is not None
    assert tao_fills.get("lifo_realized_pnl_usd", 0.0) == 0.0

    # Bug 31a & 31b: execute_enforce_intent_order retries T212 insufficient-free-for-stocks-buy with 0.94x downscale,
    # and harvest_log_anomalies_from_dirs suppresses insufficient-free-for-stocks-buy followed by 'Successfully placed BUY'
    from providers.execution_audit import execute_enforce_intent_order

    t212_calls: list[float] = []

    def _fake_place(**kw):
        t212_calls.append(kw["qty"])
        if len(t212_calls) == 1:
            kw["_outcome_context"].update({"state": "refused", "reason": "T212 HTTP 400: {'type': '/api-errors/insufficient-free-for-stocks-buy'}"})
            return None
        return {"id": "59907459622", "orderId": "59907459622"}

    dummy_t212_inst = SimpleNamespace(
        symbol="NVDA_US_EQ",
        provider_name="t212",
        _provider=SimpleNamespace(round_quantity=lambda sym, q: round(q, 2)),
        place=_fake_place,
    )
    intent_31 = {"intent_id": "intent_1_buy_nvda", "history": []}
    execute_enforce_intent_order(
        dummy_t212_inst,
        intent_31,
        intent_id="intent_1_buy_nvda",
        decision="BUY",
        target_qty=0.44,
        price=201.73,
        effective_usd=88.76,
        guard_allowed=True,
        guard_reason="OK",
        thesis="test",
        trade_tag="[auto_sharetrade_executor]",
        now_ts=now_s,
        bypass_guards=frozenset({"all"}),
    )
    assert len(t212_calls) == 2
    assert t212_calls[0] == 0.44
    assert t212_calls[1] == 0.41
    assert intent_31["status"] == "EXECUTED"

    ldir_31 = tmp_path / "logs_b31"
    ldir_31.mkdir(exist_ok=True)
    interleaved = "".join(f"{t_err} [INFO] [INTENT GATEWAY] auto_cryptotrade_executor decision is HOLD (cycle {i}).\n" for i in range(35))
    (ldir_31 / "auto_sharetrade_executor.log").write_text(
        f"{t_err} [T212] place_order NVDA_US_EQ failed (HTTP 400): {{'type': '/api-errors/insufficient-free-for-stocks-buy'}}\n"
        f"{t_err} [NVDA_US_EQ] BUY BLOCKED (pre-submit): T212 HTTP 400: {{'type': '/api-errors/insufficient-free-for-stocks-buy'}}\n"
        f"{interleaved}"
        f"{t_rec} [INFO] [LLM TRADER · ENFORCE] Successfully placed BUY NVDA_US_EQ: {{'id': 59907459622}}\n",
        encoding="utf-8",
    )
    anom_31 = harvest_log_anomalies_from_dirs([str(ldir_31)], lookback_hours=2.0)
    assert anom_31.error_count == 0


def test_bugs_90_91_92_active_cycle_recent_buys_resource_watchdog_sustained_and_5m_flow_reversal(tmp_path):
    import json
    import time
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        enrich_sentiment_advisor_live_flow_divergence,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        extract_symbol_recent_fills,
        reconcile_venue_cycles_with_recent_orders,
    )
    from orchestratorOS.lib.technical_repair_ssot import gather_system_and_vpn_telemetry

    # 1. Bug 90: extract_symbol_recent_fills scopes recent_buys to the active cycle after a full exit,
    #    and prunes >7d-old closed-cycle buys even when oversold swing PnL is checked
    tao_fills = extract_symbol_recent_fills(
        [
            {"side": "BUY", "price": 272.94, "qty": 1.19073, "ts_ms": 1791889520952},
            {"side": "BUY", "price": 264.7238, "qty": 0.58748, "ts_ms": 1792030636688},
            {"side": "BUY", "price": 256.0432, "qty": 1.26932, "ts_ms": 1792072494450},
            {"side": "SELL", "price": 226.016729, "qty": 7.40379, "ts_ms": 1793319677319},
            {"side": "BUY", "price": 235.741094, "qty": 3.70189, "ts_ms": 1793331231572},
        ],
        symbol="TAOUSD",
    )
    assert tao_fills is not None
    assert [b["price"] for b in tao_fills["recent_buys"]] == [235.741094]

    hype_fills = extract_symbol_recent_fills(
        [
            {"side": "BUY", "price": 75.55, "qty": 1.38285373, "ts_ms": 1792831547936},
            {"side": "SELL", "price": 93.3, "qty": 17.88, "ts_ms": 1793287711318},
            {"side": "BUY", "price": 91.209992, "qty": 4.9358342, "ts_ms": 1793365875456},
            {"side": "SELL", "price": 87.120002, "qty": 4.398355, "ts_ms": 1793721326834},
            {"side": "BUY", "price": 84.0, "qty": 5.95238095, "ts_ms": 1793751791558},
            {"side": "BUY", "price": 82.5, "qty": 1.81818181, "ts_ms": 1793756214195},
        ],
        symbol="HYPEUSD",
    )
    assert hype_fills is not None
    assert [b["price"] for b in hype_fills["recent_buys"]] == [91.209992, 84.0, 82.5]

    ro_tao_c = {
        "last_sell": {"price": 226.016729, "qty": 7.40379, "ts_ms": 1793319677319},
        "recent_buys": [
            {"price": 256.0432, "qty": 1.26932, "ts_ms": 1792072494450},
            {"price": 235.741094, "qty": 3.70189, "ts_ms": 1793331231572},
        ],
        "lifo_realized_pnl_usd": -116.73,
    }
    reconcile_venue_cycles_with_recent_orders(
        {"TAOUSD": {"qty": 7.38813, "cost_usd": 2105.51, "avg_cost_px": 284.98, "entry_price": 305.469, "_entry_ts_ms": 1791188569000}},
        {},
        {},
        {"kraken": {"TAOUSD": ro_tao_c}},
        {("kraken", "TAOUSD"): 3.706878},
        {"TAOUSD": {"price": 236.4}},
    )
    assert [b["price"] for b in ro_tao_c["recent_buys"]] == [235.741094]

    # 2. Bug 91: gather_system_and_vpn_telemetry filters out stale (>1h) or 1-poll transient resource_watchdog_offenders
    now_s = time.time()
    ws_stale = tmp_path / "ws_stale"
    ws_stale.mkdir(parents=True, exist_ok=True)
    (ws_stale / ".resource_watchdog_state.json").write_text(
        json.dumps({
            "last_run": now_s - 86400.0,
            "processes": {"101:1": {"pid": 101, "command": "python macro_analyzer.py"}},
            "tracked": {"101:1": {"high": 5, "alerted": True, "last_reasons": ["CPU 51.4%"]}},
        }),
        encoding="utf-8",
    )
    telem_stale = gather_system_and_vpn_telemetry(str(ws_stale), base_resources={})
    assert "resource_watchdog_offenders" not in telem_stale

    ws_fresh = tmp_path / "ws_fresh"
    ws_fresh.mkdir(parents=True, exist_ok=True)
    (ws_fresh / ".resource_watchdog_state.json").write_text(
        json.dumps({
            "last_run": now_s - 60.0,
            "processes": {
                "101:1": {"pid": 101, "command": "python macro_analyzer.py"},
                "202:2": {"pid": 202, "command": "python rtrade.py"},
            },
            "tracked": {
                "101:1": {"high": 1, "alerted": False, "last_reasons": ["CPU 51.4%"]},
                "202:2": {"high": 3, "alerted": True, "last_reasons": ["CPU 95.0%"]},
            },
        }),
        encoding="utf-8",
    )
    telem_fresh = gather_system_and_vpn_telemetry(str(ws_fresh), base_resources={})
    assert telem_fresh.get("resource_watchdog_offenders") == [
        {"key": "202:2", "command": "python rtrade.py", "reasons": ["CPU 95.0%"], "consecutive_high": 3}
    ]

    # 3. Bug 92: enrich_sentiment_advisor_live_flow_divergence detects LIVE_5M_SELL_REVERSAL / LIVE_5M_BUY_REVERSAL
    sa_bull_5m = {
        "market_bias": "BULLISH",
        "recommended_action": "ACCUMULATE",
        "age_min": 16.0,
        "summary": "BTC pushes higher propelled by taker buying (taker buy/sell ratio 1.23, 5m ratio 2.34, 5m CVD +$16.21M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_bull_5m,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 1.058,
                "cvd_15m_usd": 3_030_000.0,
                "taker_ratio_5m": 0.51,
                "cvd_5m_usd": -8_390_000.0,
                "taker_ratio_1h": 1.00,
                "cvd_1h_usd": -791_200.0,
            }
        },
    )
    assert sa_bull_5m.get("live_flow_divergence", "").startswith("LIVE_5M_SELL_REVERSAL (")
    assert "5m_tr=0.51, cvd_5m=-8.39M" in sa_bull_5m["live_flow_divergence"]

    # 4. Bug 32a & 32b: Do not emit false LIVE_SELL_REVERSAL or 300s seller-reversal loop when BULLISH/HOLD
    #    summary already cited low taker buy/sell ratio (e.g. 0.56) and live taker_ratio is 0.67
    from types import SimpleNamespace
    from intelligence.macro.calendar_ssot import compute_dynamic_sentiment_interval

    sa_bull_hold_already_low_tr = {
        "market_bias": "BULLISH",
        "recommended_action": "HOLD",
        "age_min": 10.2,
        "summary": "Broad 24h market breadth sustains structure. However, persistent taker sell dominance (taker buy/sell ratio 0.56, 4h CVD -$51.3M) indicates passive absorption.",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_bull_hold_already_low_tr,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.6667,
                "cvd_15m_usd": -12_173_136.0,
                "taker_ratio_5m": 0.4735,
                "cvd_5m_usd": -9_056_261.0,
                "taker_ratio_1h": 0.8172,
                "cvd_1h_usd": -32_628_049.0,
            }
        },
    )
    assert "live_flow_divergence" not in sa_bull_hold_already_low_tr

    sa_caution_hold_already_low_tr = {
        "market_bias": "CAUTION",
        "recommended_action": "HOLD",
        "age_min": 11.5,
        "summary": "Intraday derivatives shifted into aggressive_shorting with heavy taker sell volume (taker ratio 0.53, 5m CVD -$31.3M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_caution_hold_already_low_tr,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.55,
                "cvd_15m_usd": -12_000_000.0,
            }
        },
    )
    assert "live_flow_divergence" not in sa_caution_hold_already_low_tr

    cached_bull_hold = SimpleNamespace(
        market_bias="BULLISH",
        recommended_action="HOLD",
        risk_level="MODERATE",
        summary=sa_bull_hold_already_low_tr["summary"],
    )
    itv_sec, itv_reason = compute_dynamic_sentiment_interval(
        base_interval_sec=3600.0,
        last_eval_ts=1793841146.5,
        now_ts=1793841270.0,
        cached_assessment=cached_bull_hold,
        btc_taker_ratio=0.56,
        prev_btc_taker_ratio=0.56,
    )
    assert "btc_seller_reversal" not in itv_reason
    assert itv_sec > 300.0


def test_bugs_93_94_95_pretrade_venue_ob_isolation_watchdog_state_cleanup_and_partial_trim_scalein_dca(tmp_path):
    import json
    import os
    import time
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.sentiment.collectors.portfolio_collector import harvest_bot_runtime_states
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.portfolio_evaluator import reconcile_venue_cycles_with_recent_orders
    from orchestratorOS.lib.technical_repair_ssot import harvest_heartbeats_and_state_files
    import external_order_guard as eog

    # 1. Bug 93: collect_pretrade_telemetry isolates HYPEUSD/TAOUSD orderbooks on disk even when singleton
    #    OrderbookDepthCollector in-memory cache holds a fresh HYPEUSDC snapshot
    now_s = time.time()
    (tmp_path / "orderbook_depth_HYPEUSDC.json").write_text(
        json.dumps({
            "symbol": "HYPEUSDC",
            "mid_price": 84.303,
            "bid_depth_usd": 12_969_835.0,
            "ask_depth_usd": 14_760_427.0,
            "imbalance_ratio": 0.468,
            "largest_bid_wall_usd": 1_948_059.0,
            "largest_bid_wall_price": 83.90,
            "largest_ask_wall_usd": 2_490_997.0,
            "largest_ask_wall_price": 84.60,
            "ts": now_s - 2.0,
            "best_bid": 84.298,
            "best_ask": 84.308,
            "spread_bps": 1.19,
            "depth_1pct_bid_usd": 11_431_151.0,
            "depth_1pct_ask_usd": 11_866_477.0,
        }),
        encoding="utf-8",
    )
    (tmp_path / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps({
            "symbol": "HYPEUSD",
            "mid_price": 84.29,
            "bid_depth_usd": 2_084_086.0,
            "ask_depth_usd": 1_497_495.0,
            "imbalance_ratio": 0.582,
            "largest_bid_wall_usd": 377_879.0,
            "largest_bid_wall_price": 83.37,
            "largest_ask_wall_usd": 383_098.0,
            "largest_ask_wall_price": 85.14,
            "ts": now_s - 40.0,
            "best_bid": 84.28,
            "best_ask": 84.30,
            "spread_bps": 2.37,
            "depth_1pct_bid_usd": 1_265_681.0,
            "depth_1pct_ask_usd": 988_184.0,
        }),
        encoding="utf-8",
    )
    col = OrderbookDepthCollector(cache_dir=str(tmp_path))
    col.record_snapshot(
        OrderbookSnapshot(
            symbol="HYPEUSDC",
            mid_price=84.303,
            bid_depth_usd=12_969_835.0,
            ask_depth_usd=14_760_427.0,
            imbalance_ratio=0.468,
            largest_bid_wall_usd=1_948_059.0,
            largest_bid_wall_price=83.90,
            largest_ask_wall_usd=2_490_997.0,
            largest_ask_wall_price=84.60,
            ts=now_s - 1.0,
            best_bid=84.298,
            best_ask=84.308,
            spread_bps=1.19,
            depth_1pct_bid_usd=11_431_151.0,
            depth_1pct_ask_usd=11_866_477.0,
        )
    )
    prev_col = eog._orderbook_collector
    try:
        eog._orderbook_collector = col
        pt_hype_kr = collect_pretrade_telemetry("HYPEUSD", cachedb_dir=str(tmp_path))
    finally:
        eog._orderbook_collector = prev_col
    assert pt_hype_kr.get("Orderbook Depth Imbalance") == "0.58 (58% bids vs 42% asks)"
    assert pt_hype_kr.get("Orderbook Bid-Ask Spread") == "2.37 bps"
    assert "Cross-Venue Relative Value (Arbitrage)" in pt_hype_kr
    assert "arb_vs_HYPEUSDC=" in pt_hype_kr["Cross-Venue Relative Value (Arbitrage)"]

    # 2. Bug 94: harvest_heartbeats_and_state_files rounds age_hours and filters >24h retired watchdog states;
    #    harvest_bot_runtime_states rounds IEEE-754 float artifacts in assetguardian total_filled_qty
    ws_dir = tmp_path / "ws_state"
    ws_dir.mkdir(parents=True, exist_ok=True)
    rw_p = ws_dir / ".resource_watchdog_state.json"
    rw_p.write_text("{}", encoding="utf-8")
    os.utime(rw_p, (now_s - 936.6, now_s - 936.6))
    wd_p = ws_dir / ".watchdog_state.json"
    wd_p.write_text("{}", encoding="utf-8")
    os.utime(wd_p, (now_s - 400 * 3600.0, now_s - 400 * 3600.0))
    _, st_files = harvest_heartbeats_and_state_files(str(tmp_path), str(ws_dir), now_s)
    assert st_files == {".resource_watchdog_state.json": {"age_hours": 0.2602}}

    (tmp_path / "assetguardian_state.json").write_text(
        json.dumps({"symbols": {"ARBUSDC": {"sell": {"initial_qty": 316.552935, "total_filled_qty": 63.3 + 37.9}}}}),
        encoding="utf-8",
    )
    bot_st = harvest_bot_runtime_states(str(tmp_path), now_ts=now_s)
    assert bot_st["assetguardian_state"]["ARBUSDC"]["sell"]["total_filled_qty"] == 101.2

    # 3. Bug 95: reconcile_venue_cycles_with_recent_orders increments dca_buys when partial core remains + scale-in buy
    t212_cycs = {
        "NVDA_US_EQ": {
            "qty": 0.41,
            "cost_usd": 84.39,
            "avg_cost_px": 205.8293,
            "entry_price": 207.9271,
            "last_buy_price": 205.6579,
            "last_sell_price": 208.1,
            "last_exit_kind": "TP",
            "dca_buys": 0,
            "_last_sell_ts_ms": 1793718965337,
        }
    }
    ro_nvda = {
        "last_sell": {"price": 207.9, "qty": 0.93, "ts_ms": 1793724500000},
        "recent_buys": [
            {"price": 207.93, "qty": 0.96, "ts_ms": 1793710000000},
            {"price": 205.65, "qty": 0.38, "ts_ms": 1793804500000},
        ],
        "active_cycle_rem_qty": 0.41,
        "lifo_realized_pnl_usd": 8.20,
    }
    reconcile_venue_cycles_with_recent_orders(
        {},
        {},
        t212_cycs,
        {"t212": {"NVDA_US_EQ": ro_nvda}},
        {("t212", "NVDA_US_EQ"): 0.41},
        {"NVDA_US_EQ": {"price": 198.55}},
    )
    assert t212_cycs["NVDA_US_EQ"]["dca_buys"] == 1
    assert t212_cycs["NVDA_US_EQ"]["last_sell_price"] == 207.9
    assert t212_cycs["NVDA_US_EQ"]["last_exit_kind"] == "TP"


def test_stale_major_swing_low_cleanup_and_guard_anomaly_scanner(tmp_path):
    """Verify stale major_swing_low_px is cleared when swing_low_px is lower, and /compact warnings do not trigger GUARD."""
    import pytest
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        recalculate_live_price_geometry,
    )
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # 1. recalculate_live_price_geometry drops major_swing_low_px when maj_lo_new >= swing_low_px
    pinfo = {
        "price": 133.27,
        "swing_low_px": 129.79,
        "swing_high_px": 139.96,
        "rally_from_swing_low_pct": 2.68,
        "pullback_from_swing_high_pct": -4.78,
        "major_swing_low_px": 133.74,
        "rally_from_major_low_pct": 0.0,
    }
    recalculate_live_price_geometry(pinfo, 133.27)
    assert "major_swing_low_px" not in pinfo
    assert "rally_from_major_low_pct" not in pinfo

    # When major_swing_low_px is genuinely lower than swing_low_px, it is preserved
    pinfo_valid = {
        "price": 135.0,
        "swing_low_px": 133.0,
        "swing_high_px": 140.0,
        "major_swing_low_px": 129.79,
    }
    recalculate_live_price_geometry(pinfo_valid, 135.0)
    assert pinfo_valid.get("major_swing_low_px") == 129.79
    assert pinfo_valid.get("rally_from_major_low_pct") == pytest.approx(((135.0 - 129.79) / 129.79) * 100.0, abs=0.01)

    # 2. harvest_log_anomalies_from_dirs ignores /compact warning lines, ignores routine Existing SELL order blocks,
    #    and deduplicates repeated guard signatures by normalized signature
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "python_orchestrator.log"
    log_file.write_text(
        "2026-11-05 03:37:48,420 [WARNING] In-place /compact for topic 'auto_technical_repair' conversation d9d22d76 failed: "
        "Command '['/home/predut/.local/bin/agy-bot', '-p', '/compact Summarize prior turns in place, preserving open positions, recent decisions, and active risk guards.']' timed out after 120.0 seconds\n"
        "03:50:19 [BTCUSDC] Existing SELL order blocks AssetGuardian tier: ['10723115083']\n"
        "2026-11-05 03:38:00,000 [INFO] [GUARD] AssetGuardian tier 1 triggered for ETHUSDC\n"
        "2026-11-05 03:39:00,000 [INFO] [GUARD] AssetGuardian tier 1 triggered for ETHUSDC\n",
        encoding="utf-8",
    )
    res = harvest_log_anomalies_from_dirs([str(log_dir)], 1)
    assert res.error_count == 0
    assert res.warning_count == 2
    assert res.guard_interventions == ["2026-11-05 03:39:00,000 [INFO] [GUARD] AssetGuardian tier 1 triggered for ETHUSDC"]

    # 3. enrich_microstructure_cvd treats negative 15m CVD with high bid dominance (>=0.65), accumulation regime,
    #    and mild 15m drift (-0.09% >= -0.15%) as spot_absorbing rather than aggressive_market_dumps
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_microstructure_cvd

    btc_micro = {"cvd_15m_usd": -2_225_828.0, "ob_imbalance": 0.71, "divergence_regime": "accumulation"}
    enrich_microstructure_cvd(btc_micro, {"price": 80673.07, "chg_15m_pct": -0.09})
    assert btc_micro["cvd_regime"] == "spot_absorbing"

    # 4. harvest_tradable_positions_and_orders preserves cached position price and ts when get_current_price returns None
    import json
    from types import SimpleNamespace
    from intelligence.sentiment.collectors.portfolio_collector import harvest_tradable_positions_and_orders

    vb_path = tmp_path / "venue_balances_state.json"
    vb_path.write_text(
        json.dumps(
            {
                "pos:t212:NVDA_US_EQ": {
                    "venue": "t212",
                    "symbol": "NVDA_US_EQ",
                    "price": 199.22,
                    "free_base": 0.41,
                    "ts": 1793844937.278,
                }
            }
        ),
        encoding="utf-8",
    )
    fake_prov = SimpleNamespace(
        get_current_price=lambda s: None,
        free_balance=lambda a: 0.41 if a == "NVDA_US_EQ" else 14.01,
    )
    fake_inst = SimpleNamespace(
        symbol="NVDA_US_EQ",
        provider_name="t212",
        base="NVDA_US_EQ",
        quote="USD",
        _provider=fake_prov,
        params={},
        open_orders=lambda: [],
    )
    pt_eq = {}
    pos_list, _, _, _ = harvest_tradable_positions_and_orders(
        {"T212_NVDA": fake_inst},
        pt_eq,
        {},
        {},
        cache_dir=str(tmp_path),
    )
    assert pos_list[0]["price"] == 199.22
    assert pt_eq["NVDA_US_EQ"]["price"] == 199.22
    saved_vb = json.loads(vb_path.read_text(encoding="utf-8"))
    assert saved_vb["pos:t212:NVDA_US_EQ"]["price"] == 199.22
    assert saved_vb["pos:t212:NVDA_US_EQ"]["ts"] == 1793844937.278


