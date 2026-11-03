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
    assert analyzer._last_macro_llm_ts > now_ts - 1800.0
    assert analyzer._last_advisor_ts > now_ts - 10800.0

    # Also test when SentimentAdvisor falls back to a stale cached assessment on LLM failure
    stale_ts = now_ts - 12000.0
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

    # 5. compute_enterprise_portfolio_risk_metrics falls back to last_buy/last_sell when recent_buys/recent_sells omitted
    risk = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": [{"venue": "kraken", "symbol": "BTCUSD", "value_usd": 500.0, "free_quote": 500.0, "quote": "USD"}],
            "venue_quote_cash": {"kraken:USD": {"venue": "kraken", "free": 500.0}},
            "recent_orders": {"BTCUSD": {"last_buy": {"price": 100000.0, "qty": 0.005}}},
        },
        {"BTCUSD": {"price": 101000.0, "volatility_1h_pct": 1.0}},
        {"BTCUSD": {"spread_bps": 2.0}},
    )
    assert risk["avg_entry_markout_bps"] == 100.0
    assert risk["post_trade_win_rate_pct"] == 100.0
    assert risk["est_recent_fee_drag_usd"] == 1.3  # $500 notional * 26 bps Kraken taker fee = $1.30

    # 6. compute_deribit_options_surface ignores non-C/P instruments when building strike_oi
    surf = compute_deribit_options_surface(
        [
            {"instrument_name": "BTC-28NOV26-90000-C", "open_interest": 100.0, "mark_iv": 52.0, "underlying_price": 90000.0},
            {"instrument_name": "BTC-28NOV26-90000-P", "open_interest": 80.0, "mark_iv": 54.0, "underlying_price": 90000.0},
            {"instrument_name": "BTC-28NOV26-91000-F", "open_interest": 9999.0, "mark_iv": 0.0, "underlying_price": 90000.0},
        ],
        btc_dvol=50.0,
        btc_rv=45.0,
    )
    assert surf["btc_gamma_pin_strike"] == 90000

