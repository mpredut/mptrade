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
