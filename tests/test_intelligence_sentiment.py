"""Unit tests for Pillar 3: Sentiment Intelligence (Fear & Greed, Market Breadth, Contrarian Triggers, Euphoria/Panic Guards)."""
import ast
import cacheManager as cm
import external_order_guard as eog
import intelligence.sentiment.collectors.pretrade_telemetry_collector as ptc_mod
import intelligence.sentiment.gemini_client as gc_mod
import json
import llm_order_guard
import math
import order_guard
import order_retry
import os
import pathlib
import pytest
import shadow_signals as ss
import sqlite3
import subprocess
import time
import urllib.request
from botcore import obj_attr
from cachepaths import (
    cache_symbol_variants,
    canonical_symbol_key,
    ensure_equity_suffix,
    filter_asset_class_variants,
    has_equity_suffix,
    normalize_cache_base_asset,
    resolve_cache_subdomain,
    split_symbol_assets,
)
from collections import deque
from datetime import datetime, timezone
from instrument_registry import is_equity_symbol
from intelligence.composite import CompositeMarketIntelligence
from intelligence.daemon import IntelligenceTelemetryDaemon
from intelligence.external.collectors.derivatives_telemetry_collector import (
    DerivativesTelemetry,
    DerivativesTelemetryCollector,
)
from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
from intelligence.internal.state.survival import estimate_remaining_trend_hours
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerSide
from intelligence.macro.calendar_ssot import (
    compute_dynamic_geopolitical_interval,
    compute_dynamic_sentiment_interval,
    parse_cited_cvd_usd,
    parse_cited_taker_ratio,
    parse_live_macro_release_surprises,
    resolve_macro_economic_calendar,
)
from intelligence.macro.daemon_ssot import CachedTopicAssessmentAgent, detect_active_binance_symbols
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer, GeopoliticalThreatAssessment
from intelligence.macro.lexicon_ssot import (
    _BROAD_MARKET_INDEX_CONTEXT_WORDS,
    _is_editorial_clickbait_or_advice_title,
    _is_noise_dynamic_keyword,
    _normalize_semantic_stem,
    _strip_headline_publisher_suffix,
    discover_dynamic_terms_from_headlines,
    filter_valid_severity_keywords,
    has_financial_or_geopolitical_context,
    headlines_are_semantic_duplicates,
    is_excluded_reference_news_item,
    is_off_topic_general_news_title,
    match_high_severity_keywords,
)
from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsHeadline
from intelligence.macro_analyzer import MacroAnalyzer
from intelligence.macro_sentiment_ssot import (
    extract_high_severity_headlines_and_digest,
    get_breadth_excluded_bases,
    load_cached_fear_greed_snapshot,
    load_cached_market_breadth_snapshot,
    load_cached_whale_telemetry,
)
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector, FearGreedSnapshot
from intelligence.sentiment.collectors.macro_telemetry_collector import (
    PortfolioAndVenueCyclesCollector,
    collect_macro_benchmarks_snapshot,
    compute_deribit_options_surface,
    harvest_macro_and_news_sensors,
)
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector, MarketBreadthSnapshot
from intelligence.sentiment.collectors.microstructure_collector import (
    load_stitched_7d_ohlc_closes,
    resolve_fallback_ohlc_fn,
    enrich_24h_and_microstructure,
    load_symbol_microstructure,
    load_symbol_price_series,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    _load_cached_ai_intents_list,
    harvest_bot_runtime_states,
    harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders,
    harvest_venue_cycles,
    load_symbol_best_recent_fills,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    apply_asset_scope_filter,
    enrich_sentiment_advisor_live_flow_divergence,
    format_macro_benchmarks_summary_line,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _infer_symbol_venue_and_fees_bps,
    align_equity_orderbook_to_spot,
    compute_liquidation_clusters,
    enrich_cross_venue_trends,
    enrich_microstructure_cvd,
    reconcile_sibling_flow_and_liq_regimes,
    sync_sibling_symbol_telemetry,
    update_microstructure_cycle_deltas,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    _populate_cycle_fallback_recent_order,
    _sync_cycle_wallet_qty_and_avg,
    apply_cycle_fallback_to_recent_fills,
    enrich_positions_with_orders_and_cycles,
    extract_symbol_recent_fills,
    parse_venue_cycle_state,
    reconcile_venue_cycles_with_recent_orders,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    apply_24h_series_enrichment,
    build_vp_position_dict,
    compute_short_slopes_from_series,
    _extract_sorted_ts_px,
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_24h_price_metrics,
    compute_7d_range_metrics,
    compute_swing_wave_metrics,
    compute_volume_profile_metrics,
    recalculate_live_price_geometry,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    build_cross_market_context,
    compute_enterprise_portfolio_risk_metrics,
    enrich_open_limit_orders_tca,
)
from intelligence.sentiment.gemini_client import GeminiClient, query_topic_json
from intelligence.sentiment.guards import high_stake_guard as hsg_mod, post_stop_reentry_guard as psg_mod
from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
from intelligence.sentiment.guards.high_stake_guard import (
    HighStakeGuard,
    PreTradeLLMGuardBase,
    _HIGH_STAKE_DECISION_CACHE,
    collect_pretrade_telemetry,
    evaluate_llm_guard_response,
    resolve_llm_guard_cache_ttl,
)
from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
from intelligence.sentiment.sentiment_advisor import SentimentAdvisor, SentimentAdvisorAssessment
from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger
from market_regime import MarketRegimeContext, MarketRegimeDecision
from orchestratorOS.agy.summary_store import (
    conversation_db,
    read_summary_row,
    restore_conversation_state,
    snapshot_conversation_state,
    summaries_db,
)
from orchestratorOS.agy.topics import TopicSession
from orchestratorOS.lib.technical_repair_ssot import (
    LogEventSummary,
    apply_code_patch_safely,
    apply_reconciliation_plan,
    audit_workspace_configs,
    gather_system_and_vpn_telemetry,
    harvest_heartbeats_and_state_files,
    harvest_log_anomalies_from_dirs,
    normalize_error_signature,
)
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
from orchestratorOS.supervisor.prompt_sections import (
    _fmt_px,
    _fmt_slope_val,
    _format_market_prices_section,
    _format_microstructure_section,
    _format_portfolio_section,
    _format_recent_order_line,
    _format_sre_and_fleet_section,
    _format_tradable_position_line,
    _format_venue_cycle_line,
)
from providers.execution_audit import (
    build_last_cycle_execution_status,
    cancel_existing_ai_open_orders,
    execute_enforce_intent_order,
    extract_intent_notional,
    find_duplicate_ai_open_limit_order,
    reconcile_intents_from_execution_audit,
    summarize_recent_ai_intents,
)
from providers.order_ohlc_utils import (
    extract_client_order_id,
    format_actor_tag,
    normalize_open_order,
    resolve_order_actor,
    resolve_order_origin,
)
from providers.quantity import extract_regime_buy_scale
from state_io import write_meta_json
from tests.conftest import DummyGeminiClient, MockHttpResponse, load_venue_modules
from types import SimpleNamespace
from unittest.mock import MagicMock, patch



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
        client = GeminiClient(
            custom_runner=lambda *_: '{"decision": "APPROVED", "suggested_scale": 1.0, "reason": "All metrics green"}'
        )
        res = client.query_json("Test prompt")
        assert res is not None
        assert res["decision"] == "APPROVED"
        assert res["suggested_scale"] == 1.0

    def test_query_json_markdown_wrapped(self):
        client = GeminiClient(
            custom_runner=lambda *_: '```json\n{\n  "decision": "DOWNSCALE",\n  "suggested_scale": 0.5,\n  "reason": "Overleveraged"\n}\n```'
        )
        res = client.query_json("Test prompt")
        assert res is not None
        assert res["decision"] == "DOWNSCALE"
        assert res["suggested_scale"] == 0.5


class TestSentimentAdvisor:
    """Tests for periodic SentimentAdvisor macro evaluations."""

    def test_advisor_review_and_caching(self, tmp_path):
        client = DummyGeminiClient({
            "market_bias": "BULLISH",
            "risk_level": "MODERATE",
            "confidence": 0.85,
            "summary": "Macro indicators suggest continued institutional accumulation.",
            "key_risks": ["Upcoming CPI release", "Derivatives OI peak"],
            "recommended_action": "ACCUMULATE",
        })
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
        client = DummyGeminiClient()
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        # 500 EUR order -> instant pass
        dec = guard.check("BTCUSDT", "BUY", price=50000.0, qty=0.01, notional_eur=500.0)
        assert dec.allowed is True
        assert "below_high_stake_threshold" in dec.reason
        assert client.call_count == 0

    def test_high_stake_approved(self):
        client = DummyGeminiClient({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Healthy trend and low funding"})
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.025, notional_eur=1500.0)
        assert "1,500.00 EUR" in client.last_prompt
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.NONE
        assert "Gemini approved" in dec.reason

    def test_high_stake_vetoed(self):
        client = DummyGeminiClient({"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Massive whale sell wall and euphoric top"})
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=70000.0, qty=0.03, notional_eur=2100.0)
        assert dec.allowed is False
        assert dec.brake_action == BrakeAction.HARD_VETO
        assert "Gemini vetoed" in dec.reason

    def test_high_stake_downscaled(self):
        client = DummyGeminiClient({"decision": "DOWNSCALE", "suggested_scale": 0.35, "reason": "Caution: funding is elevated"})
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.02, notional_eur=1200.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
        assert dec.suggested_scale == 0.35

    def test_high_stake_downscaled_percentage_scale(self):
        client = DummyGeminiClient({"decision": "DOWNSCALE", "suggested_scale": 25, "reason": "Scale to 25%"})
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.02, notional_eur=1200.0)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
        assert dec.suggested_scale == 0.25


class TestOrderGuardWithGemini:
    """Integration test for order_guard.check_intelligence_guards with Gemini mode."""

    def test_check_intelligence_guards_with_gemini_shadow(self, monkeypatch):

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


        mock_guard = HighStakeGuard(
            gemini_client=DummyGeminiClient({"decision": "REJECTED", "suggested_scale": 0.0, "reason": "Risky market"}),
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


        called_notionals = []
        mock_guard = HighStakeGuard(
            gemini_client=DummyGeminiClient({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Good trade"}),
            min_notional_eur=1000.0,
        )
        orig_check = mock_guard.check
        mock_guard.check = lambda symbol, side, price, qty, notional_eur=None, **kw: (
            called_notionals.append(notional_eur)
            or orig_check(symbol, side, price, qty, notional_eur=notional_eur)
        )
        monkeypatch.setattr("intelligence.sentiment.guards.high_stake_guard.HighStakeGuard", lambda **kw: mock_guard)

        # Provider mockup
        p_stub = SimpleNamespace(name="binance", last_opposite_fill=lambda *a, **k: 70000.0)

        # BUY 0.05 BTC @ 65,000 = 3,250 EUR > 1000 EUR min
        allowed = order_guard.profit_guard(p_stub, "BTCUSDC", "BUY", 65000.0, 1.15, qty=0.05)
        assert allowed is True
        assert len(called_notionals) == 1
        assert called_notionals[0] == 65000.0 * 0.05  # 3,250

    def test_symbol_regime_fallback_when_allow_fallback_true(self, monkeypatch):

        snapshot_called = False
        def mock_snapshot(symbol, now=None):
            nonlocal snapshot_called
            snapshot_called = True
            return {"gradient_recent": 0.8, "epsilon": 0.1, "ts": time.time()}

        monkeypatch.setattr(
            "cacheManager.get_short_trend_manager",
            lambda: SimpleNamespace(get_fresh_snapshot=mock_snapshot),
        )

        # Calling symbol_regime directly without snapshot_resolver falls back gracefully
        decision = order_guard.symbol_regime("BTCUSDT", allow_fallback=True)
        assert snapshot_called is True
        assert decision.regime == "bull"

    def test_macro_shadow_guard_notification_title_and_cooldown(self, monkeypatch):
        """Verify macro shadow guard sends clear VETO titles and deduplicates consecutive calls."""

        margins = {
            **order_guard._load_margins(),
            "intelligence_guards_mode": "shadow",
            "geopolitical_guard_mode": "shadow",
            "shadow_notify": 1.0,
            "default": 1.15,
        }
        monkeypatch.setattr(order_guard, "_load_margins", lambda: margins)
        llm_order_guard._GEO_SHADOW_NOTIFY_COOLDOWN.clear()

        shock_geo = GeopoliticalThreatAssessment(
            threat_level="CRITICAL_SHOCK",
            risk_score=0.95,
            summary="Kinetic strikes on energy hubs",
            recommended_brake="HALT_ALL",
            headlines_analyzed=10,
            ts=time.time(),
        )

        monkeypatch.setattr(
            "intelligence.macro.geopolitical_analyzer.GeopoliticalThreatAnalyzer",
            lambda **_kw: SimpleNamespace(_cached_assessment=shock_geo),
        )

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

        client = DummyGeminiClient({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "Telemetry looks solid"})
        guard = HighStakeGuard(gemini_client=client, min_notional_eur=1000.0)

        explicit_telemetry = {
            "Orderbook Depth Imbalance": "0.85 (85% bids vs 15% asks)",
            "Perpetual Funding Rate": "+0.0001% (8h)",
            "Whale Positioning": "65.0% Top Traders Long",
            "Technical Trend Regime": "bull (gradient=+0.0012, epsilon=0.0004)",
        }

        dec = guard.check("BTCUSDT", "BUY", price=60000.0, qty=0.025, notional_eur=1500.0, telemetry=explicit_telemetry)
        assert dec.allowed is True
        assert len(client.prompts) == 1
        prompt = client.last_prompt

        assert "Orderbook Depth Imbalance: 0.85 (85% bids vs 15% asks)" in prompt
        assert "Perpetual Funding Rate: +0.0001% (8h)" in prompt
        assert "Whale Positioning: 65.0% Top Traders Long" in prompt
        assert "Technical Trend Regime: bull (gradient=+0.0012, epsilon=0.0004)" in prompt
        assert "€1,500.00 EUR" in prompt

    def test_gemini_client_fallback_chain_and_non_gemini_rest(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_COMPAT_FALLBACK_URL", raising=False)
        monkeypatch.delenv("OPENAI_COMPAT_FALLBACK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_COMPAT_FALLBACK_MODEL", raising=False)

        # 1. Test fallback models initialization across Flash tiers
        client = GeminiClient(
            model="primary-failing-model",
            fallback_models=["gemini-3.8-flash-medium", "gemini-3.8-flash-low"],
            api_key=None,
        )
        assert client.fallback_models == ("gemini-3.8-flash-medium", "gemini-3.8-flash-low")

        # 2. Test REST fallback cleans non-Gemini model names safely

        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}'
        mock_resp.__enter__.return_value = mock_resp

        mock_open = MagicMock(return_value=mock_resp)

        orig_urlopen = urllib.request.urlopen
        try:
            urllib.request.urlopen = mock_open
            rest_client = GeminiClient(
                model="claude-sonnet-5-5-medium",
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
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps({"response": f"success from {used_m}"}))

        mock_run = MagicMock(side_effect=mock_subproc_run)
        orig_run = subprocess.run
        try:
            subprocess.run = mock_run
            cli_client = GeminiClient(
                model="failing-model",
                fallback_models=["gemini-3.8-flash-medium", "gemini-3.8-flash-low"],
                cli_path=__file__,
                api_key=None,
            )
            # First test: primary fails, secondary succeeds
            res = cli_client.query_text("sample prompt")
            assert res == "success from gemini-3.8-flash-medium"
            assert attempted_models == ["failing-model", "gemini-3.8-flash-medium"]

            # Second test: both fail, third (gemini-3.8-flash-low) succeeds
            attempted_models.clear()
            def mock_subproc_run_flash_low(cmd, **kwargs):
                m_idx = cmd.index("--model") + 1
                used_m = cmd[m_idx]
                attempted_models.append(used_m)
                if used_m in ("failing-model", "gemini-3.8-flash-medium"):
                    return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="Rate limit")
                return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps({"response": f"success from {used_m}"}))

            subprocess.run = MagicMock(side_effect=mock_subproc_run_flash_low)
            res2 = cli_client.query_text("sample prompt 2")
            assert res2 == "success from gemini-3.8-flash-low"
            assert attempted_models == ["failing-model", "gemini-3.8-flash-medium", "gemini-3.8-flash-low"]

            # 4. Topic session with credit/quota exhaustion skips redundant one-off retry on the same model

            cli_invocations = []
            def mock_subproc_credit(cmd, **kwargs):
                m_idx = cmd.index("--model") + 1
                used_m = cmd[m_idx]
                has_conv = "--conversation" in cmd
                cli_invocations.append((used_m, has_conv))
                if used_m in ("failing-model", "gemini-3.8-flash-medium"):
                    return subprocess.CompletedProcess(
                        cmd,
                        returncode=3,
                        stdout="",
                        stderr='error: Your AI credits balance is too low to continue.\nAGY_ERROR: {"status":"RESOURCE_EXHAUSTED"}',
                    )
                return subprocess.CompletedProcess(
                    cmd,
                    returncode=0,
                    stdout=json.dumps({"response": f"success from {used_m}", "conversation_id": "cid-topic-1"}),
                )

            subprocess.run = MagicMock(side_effect=mock_subproc_credit)
            with patch.object(gc_mod, "prepare_topic_session", return_value=TopicSession("cid-topic-1", "", False)), \
                 patch.object(gc_mod, "record_topic_turn"):
                res3 = cli_client.query_text("sample prompt 3", topic_key="auto_cryptotrade_executor")
            assert res3 == "success from gemini-3.8-flash-low"
            assert cli_invocations == [
                ("failing-model", True),
                ("gemini-3.8-flash-medium", True),
                ("gemini-3.8-flash-low", True),
            ]

            # 5. Unrecognized model error on gemini-4-neon-medium does not poison gemini-3.8-flash-* in runtime_exhausted
            unrec_invocations = []
            def mock_subproc_unrecognized(cmd, **kwargs):
                m_idx = cmd.index("--model") + 1
                used_m = cmd[m_idx]
                unrec_invocations.append(used_m)
                if used_m == "gemini-4-neon-medium":
                    return subprocess.CompletedProcess(
                        cmd,
                        returncode=1,
                        stdout="",
                        stderr='error: invalid model selection: model "gemini-4-neon-medium" is not recognized as a known model.',
                    )
                return subprocess.CompletedProcess(
                    cmd,
                    returncode=0,
                    stdout=json.dumps({"response": f"success from {used_m}"}),
                )

            subprocess.run = MagicMock(side_effect=mock_subproc_unrecognized)
            tier_client = GeminiClient(
                model="gemini-4-neon-medium",
                fallback_models=["gemini-3.8-flash-medium", "gemini-3.8-flash-low"],
                cli_path=__file__,
                api_key=None,
            )
            q_file = str(tmp_path / "quota_unrec.json")
            GeminiClient._QUOTA_MEM_CACHE = {}
            with patch.object(GeminiClient, "_get_quota_state_path", staticmethod(lambda: q_file)), \
                 patch.object(GeminiClient, "probe_live_model_quota", return_value=None):
                res4 = tier_client.query_text("sample prompt 4")
            assert res4 == "success from gemini-3.8-flash-medium"
            assert unrec_invocations == ["gemini-4-neon-medium", "gemini-3.8-flash-medium"]
        finally:
            subprocess.run = orig_run


def test_order_actor_resolution_and_recent_fills_tagging():
    """Verify AI vs BOT actor resolution on orders, open orders, and recent fills extraction."""

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
    # monitororder replacements of manual orders and venue-native manual orders classify as MANUAL with manual module
    assert resolve_order_origin({"clientOrderId": "MO_MN_OR_MainThread_1A2B"})[:2] == ("MANUAL", "manual")
    assert resolve_order_origin({"clientOrderId": "web_8f2c1d9e"})[:2] == ("MANUAL", "manual")
    assert resolve_order_origin({"clientOrderId": "and_8f2c1d9e"})[:2] == ("MANUAL", "manual")
    assert resolve_order_actor({"clientOrderId": "and_8f2c1d9e"}) == "MANUAL"
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
    norm_manual = normalize_open_order({"orderId": "xyz", "clientOrderId": "web_1", "price": 100, "origQty": 1}, venue="BINANCE", symbol="BTCUSDC")
    assert norm_manual["actor"] == "MANUAL"
    assert norm_manual["module"] == "manual"
    assert norm_manual["clientOrderId"] == "web_1"

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

    # Phase 1: Neon models exhausted (absent from buckets, 0 availableCredits), Claude has 100% quota
    api_state = {
        "buckets": [
            {"modelId": "claude-sonnet-5-5-medium", "remainingFraction": 1.0},
            {"modelId": "claude-sonnet-5-5-low", "remainingFraction": 1.0},
        ],
        "tier": "g1-ultra-tier",
        "companionProject": "",
        "availableCredits": [],
    }
    quota_req_bodies = []

    def _fake_urlopen(req, timeout=2.5):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        assert req.headers.get("User-agent") == "antigravity/1.4.12"
        if "retrieveUserQuota" in url:
            body_obj = json.loads((req.data or b"{}").decode("utf-8"))
            quota_req_bodies.append(body_obj)
            if api_state["tier"] == "free-tier" and body_obj.get("project") != "aicode-consumers":
                raise AssertionError("retrieveUserQuota on free-tier requires cloudaicompanionProject")
            return MockHttpResponse({"buckets": api_state["buckets"]})
        if "loadCodeAssist" in url:
            payload = {
                "currentTier": {"id": api_state["tier"]},
                "paidTier": {"id": api_state["tier"], "availableCredits": api_state["availableCredits"]},
            }
            if api_state["companionProject"]:
                payload["cloudaicompanionProject"] = api_state["companionProject"]
            return MockHttpResponse(payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setenv("LLM_DEFAULT_MODEL", "gemini-4-neon-medium")
    monkeypatch.setenv("LLM_HIGH_STAKE_MODEL", "gemini-4-neon-high")
    monkeypatch.setenv("LLM_LOW_STAKE_MODEL", "gemini-4-neon-low")
    monkeypatch.setenv("LLM_COMPACT_MODEL", "gemini-3.8-flash-low")
    monkeypatch.setenv(
        "LLM_FALLBACK_MODELS",
        "gemini-4-neon-medium,gemini-4-neon-low,claude-sonnet-5-5-medium,claude-sonnet-5-5-low,gemini-3.8-flash-medium,gemini-3.8-flash-low",
    )
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    snap1 = GeminiClient.probe_live_model_quota(bot_cli_base=str(bot_cli_base), force_refresh=True)
    assert snap1 is not None
    assert snap1["available_credits"] == 0.0

    client = GeminiClient(cli_path=str(cli_bin))
    med_candidates = client.get_model_candidates("gemini-4-neon-medium")
    filtered_med = GeminiClient.filter_models_with_confirmed_credits(med_candidates, snap1)
    # Neon models (0% quota) are skipped immediately; paid claude-sonnet-5-5-medium is first!
    assert filtered_med == [
        "claude-sonnet-5-5-medium",
        "claude-sonnet-5-5-low",
    ]

    low_candidates = client.get_model_candidates("gemini-4-neon-low")
    filtered_low = GeminiClient.filter_models_with_confirmed_credits(low_candidates, snap1)
    assert filtered_low[0] == "claude-sonnet-5-5-low"

    # Phase 2: User tops up paid credits (or 5h window resets) -> Neon dynamically restored on next probe!
    api_state["availableCredits"] = [{"creditAmount": 25.0}]
    snap2 = GeminiClient.probe_live_model_quota(bot_cli_base=str(bot_cli_base), force_refresh=True)
    assert snap2 is not None
    assert snap2["available_credits"] == 25.0
    restored_med = GeminiClient.filter_models_with_confirmed_credits(med_candidates, snap2)
    assert restored_med[0] == "gemini-4-neon-medium"

    # Phase 3: Consumer free-tier passes cloudaicompanionProject to retrieveUserQuota and routes directly to gemini-3.8-flash-*
    api_state["tier"] = "free-tier"
    api_state["companionProject"] = "aicode-consumers"
    api_state["availableCredits"] = []
    api_state["buckets"] = [
        {"modelId": "gemini-3.8-flash-tiered", "remainingFraction": 0.83},
    ]
    quota_req_bodies.clear()
    snap3 = GeminiClient.probe_live_model_quota(bot_cli_base=str(bot_cli_base), force_refresh=True)
    assert snap3 is not None
    assert quota_req_bodies == [{"project": "aicode-consumers"}]
    assert snap3["buckets"]["gemini-4-neon-medium"] == 0.0
    assert snap3["buckets"]["gemini-4-neon-xhigh"] == 0.0
    assert snap3["buckets"]["gemini-3.8-flash-medium"] == 0.83
    assert snap3["buckets"]["gemini-3.8-flash-high"] == 0.83
    assert GeminiClient.filter_models_with_confirmed_credits(med_candidates, snap3) == [
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-low",
    ]
    high_candidates = client.get_model_candidates("gemini-4-neon-high")
    assert GeminiClient.filter_models_with_confirmed_credits(high_candidates, snap3) == [
        "gemini-3.8-flash-high",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-low",
    ]
    xhigh_candidates = client.get_model_candidates("gemini-4-neon-xhigh")
    assert GeminiClient.filter_models_with_confirmed_credits(xhigh_candidates, snap3) == [
        "gemini-3.8-flash-high",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-low",
    ]


def test_gemini_client_low_quota_error_ntfy_alert_and_cooldown(tmp_path, monkeypatch):
    """Verify NTFY_TOPIC_ERROR alert fires when Neon quota is near zero (<=10%) and resets on top-up."""

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

    # 1. Primary Gemini family at 0%, Claude at 100% -> triggers primary_low alert on ntfy-error-1978
    snap_low = {
        "checked_ts": time.time(),
        "available_credits": 0.0,
        "buckets": {
            "claude-sonnet-5-5-medium": 1.0,
            "gemini-3.8-flash-medium": 1.0,
        },
        "last_alert_ts": 0.0,
        "last_alert_tier": "ok",
    }
    GeminiClient._maybe_notify_low_quota(snap_low)
    assert len(alerts) == 1
    assert "0%" in alerts[0]["title"]
    assert snap_low["last_alert_tier"] == "primary_low"

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

    # 6. Runtime credit exhaustion on gemini-4-neon-medium clears shared Gemini/Neon pool buckets & available_credits,
    # whereas an unrecognized model slug marks only that single model without poisoning Flash or available_credits.
    quota_state_file.write_text(
        json.dumps({
            "checked_ts": time.time(),
            "available_credits": 10.0,
            "buckets": {
                "gemini-4-neon-xhigh": 0.8,
                "gemini-4-neon-high": 0.8,
                "gemini-4-neon-medium": 0.8,
                "gemini-4-neon-low": 0.8,
                "gemini-3.8-flash-medium": 0.8,
                "gemini-3.8-flash-low": 0.8,
                "claude-sonnet-5-5-medium": 1.0,
            },
            "last_alert_ts": 0.0,
            "last_alert_tier": "ok",
        }),
        encoding="utf-8",
    )
    alerts.clear()
    GeminiClient._record_model_quota_exhausted("gemini-4-neon-medium", model_unrecognized=True)
    assert len(alerts) == 0
    assert GeminiClient._QUOTA_MEM_CACHE["available_credits"] == 10.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-4-neon-medium"] == 0.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-3.8-flash-medium"] == 0.8
    assert "gemini-3.8-flash-medium" not in GeminiClient._QUOTA_MEM_CACHE["runtime_exhausted"]

    GeminiClient._record_model_quota_exhausted("gemini-4-neon-medium")
    assert len(alerts) == 1
    assert GeminiClient._QUOTA_MEM_CACHE["available_credits"] == 0.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-4-neon-low"] == 0.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-4-neon-high"] == 0.0
    assert GeminiClient._QUOTA_MEM_CACHE["buckets"]["gemini-4-neon-xhigh"] == 0.0

    # 7. filter_models_with_confirmed_credits never returns an empty list if all buckets report 0.0
    all_zero_snap = {
        "checked_ts": time.time(),
        "available_credits": 0.0,
        "buckets": {"claude-sonnet-5-5-medium": 0.0, "gemini-3.8-flash-low": 0.0},
        "runtime_exhausted": {},
    }
    cands = ["claude-sonnet-5-5-medium", "gemini-3.8-flash-low"]
    assert GeminiClient.filter_models_with_confirmed_credits(cands, all_zero_snap) == cands

    # 8. Interrupted CLI process (SIGTERM / "error: interrupted") is never mistaken for quota exhaustion
    # even when stdout contains buffered prompt/context text mentioning "rate limit"

    assert resolve_cache_subdomain("llm_quota_status.json") == "llm"
    interrupted_proc = subprocess.CompletedProcess(
        ["agy"],
        returncode=1,
        stdout="Context: T212 rate limit spacing is 10s",
        stderr="error: interrupted\n",
    )
    assert GeminiClient._is_cli_interrupted(interrupted_proc) is True
    assert GeminiClient._is_quota_or_credit_error(interrupted_proc) is False
    sigterm_proc = subprocess.CompletedProcess(
        ["agy"],
        returncode=-15,
        stdout="RESOURCE_EXHAUSTED in prompt example",
        stderr="",
    )
    assert GeminiClient._is_cli_interrupted(sigterm_proc) is True
    assert GeminiClient._is_quota_or_credit_error(sigterm_proc) is False
    non_quota_err_proc = subprocess.CompletedProcess(
        ["agy"],
        returncode=1,
        stdout="Prompt text mentioning rate limit and quota exceeded",
        stderr="error: connection reset by peer",
    )
    assert GeminiClient._is_quota_or_credit_error(non_quota_err_proc) is False


def test_unified_ai_agent_base_classes_and_query_topic_json():
    """Verify PreTradeLLMGuardBase, CachedTopicAssessmentAgent, and query_topic_json normalization."""

    assert issubclass(HighStakeGuard, PreTradeLLMGuardBase)
    assert issubclass(PostStopReentryGuard, PreTradeLLMGuardBase)
    assert issubclass(GeopoliticalThreatAnalyzer, CachedTopicAssessmentAgent)
    assert issubclass(SentimentAdvisor, CachedTopicAssessmentAgent)

    spy = DummyGeminiClient({"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"})
    res = spy.query_topic_json("guard_high_stake", "Evaluate BUY", subtitle="BUY BTCUSDC €1,000.00")
    assert res == {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"}
    assert len(spy.calls) == 1
    _, kw = spy.calls[0]
    assert kw["topic_key"] == "guard_high_stake"
    assert "BUY BTCUSDC €1,000.00" in kw["thread_title"]
    assert kw["timeout_sec"] == 240.0
    assert query_topic_json(None, "guard_high_stake", "noop") is None


def test_pretrade_guard_cross_mode_cache_isolation_and_high_stake_mode():
    """Verify PostStopReentryGuard and HighStakeGuard adapt cached decisions across enforce <-> shadow modes."""

    psg_mod._POST_STOP_DECISION_CACHE.clear()
    hsg_mod._HIGH_STAKE_DECISION_CACHE.clear()

    client = DummyGeminiClient({
        "decision": "REJECTED",
        "suggested_scale": 0.0,
        "reason": "Momentum downward",
    })
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
    assert client.call_count == 1

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
    assert client.call_count == 1  # Served from cache
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
    assert client.call_count == 2
    assert hs_d_shd.allowed is True
    assert hs_d_shd.shadow_decision is not None and hs_d_shd.shadow_decision.allowed is False

    hs_enforce = HighStakeGuard(
        min_notional_eur=500.0, gemini_client=client, mode="enforce", notify_enabled=False, cache_ttl_sec=60.0
    )
    hs_d_enf = hs_enforce.check("BTCUSDC", "BUY", 85000.0, 0.02, notional_eur=1200.0, telemetry={})
    assert client.call_count == 2  # Served from cache
    assert hs_d_enf.allowed is False
    assert "Momentum downward" in hs_d_enf.reason

    # 3. HighStakeGuard(mode="off") immediately allows without calling Gemini
    hs_off = HighStakeGuard(min_notional_eur=500.0, gemini_client=client, mode="off", notify_enabled=False)
    hs_d_off = hs_off.check("BTCUSDC", "BUY", 85000.0, 0.02, notional_eur=1200.0, telemetry={})
    assert hs_d_off.allowed is True
    assert hs_d_off.reason == "high_stake_guard_disabled"
    assert client.call_count == 2


def test_macro_analyzer_and_daemon_apply_retry_backoff_on_llm_failure(monkeypatch, tmp_path):
    """Verify MacroAnalyzer.run_cycle and IntelligenceTelemetryDaemon.run_cycle back off via compute_step_retry_ts."""

    analyzer = MacroAnalyzer(
        cache_dir=str(tmp_path),
        macro_llm_interval_sec=1800.0,
        advisor_interval_sec=10800.0,
    )
    analyzer._last_macro_llm_ts = 0.0
    analyzer._last_advisor_ts = 0.0

    monkeypatch.setattr(analyzer, "write_heartbeat", lambda *a, **k: None)
    monkeypatch.setattr("intelligence.macro_analyzer.prune_automated_cli_threads", lambda *a, **k: 0)
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

    client = GeminiClient()
    models_to_try = [
        "gemini-4-neon-medium",
        "gemini-4-neon-low",
        "claude-sonnet-5-5-low",
        "gemini-3.8-flash-low",
    ]
    # Suppose attempt_idx=2 (gemini-4-neon-medium failed at idx 0, gemini-4-neon-low quota-exhausted at idx 1)
    monkeypatch.setattr(
        GeminiClient,
        "_QUOTA_MEM_CACHE",
        {
            "runtime_exhausted": {
                "gemini-4-neon-low": 9999999999.0,
                "gemini-3.8-flash-low": 9999999999.0,
            }
        },
    )
    cands = client.get_compact_model_candidates(models_to_try, attempt_idx=2)
    assert "gemini-4-neon-medium" not in cands
    assert "gemini-4-neon-low" not in cands
    assert "gemini-3.8-flash-low" not in cands
    assert cands == ["claude-sonnet-5-5-low"]


def test_find_duplicate_ai_open_limit_order_matches_wallet_clamped_intent():
    """Verify find_duplicate_ai_open_limit_order recognizes a resting AI limit order clamped by wallet balance."""

    stub_inst = SimpleNamespace(
        symbol="TAOUSD",
        open_orders=lambda: [
            {
                "orderId": "O-CLAMPED-1",
                "side": "BUY",
                "price": "100.0",
                "origQty": "3.2",
                "executedQty": "0.0",
            }
        ],
    )

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
        stub_inst,
        "BUY",
        100.0,
        500.0,
        intents=prior_intents,
    )
    assert dup == "O-CLAMPED-1"

    # Intentionally changed target notional ($200) -> does NOT match the $500->$320 order
    dup_changed = find_duplicate_ai_open_limit_order(
        stub_inst,
        "BUY",
        100.0,
        200.0,
        intents=prior_intents,
    )
    assert dup_changed is None


def test_equity_symbol_normalization_and_pretrade_telemetry_cache_isolation(tmp_path):
    """Verify delimiter-stripped USEQ/_EQ symbol normalization and custom cachedb_dir isolation in collect_pretrade_telemetry."""

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

    # Bug A: detect_active_binance_symbols keeps enabled cross-venue same-base pairs (HYPEUSD alongside HYPEUSDC)
    # and Binance ADAUSDC/TAOUSDC, while excluding disabled Kraken pairs (TAOUSD)
    active_syms = detect_active_binance_symbols(include_kraken_cache=True)
    assert "TAOUSDC" in active_syms
    assert "ADAUSDC" in active_syms
    assert "HYPEUSDC" in active_syms
    assert "HYPEUSD" in active_syms
    assert "TAOUSD" not in active_syms

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

    # 2. Non-FOMC day with CPI/Fed-rate-hike headline does not mislabel 21:00 EET window as FOMC Rate Decision
    # 2026-11-04 18:30 UTC = 20:30 EET (30m before 21:00 EET window, not an FOMC date)
    non_fomc_ts = datetime(2026, 11, 4, 18, 30, 0, tzinfo=timezone.utc).timestamp()
    cal = resolve_macro_economic_calendar(
        now_ts=non_fomc_ts,
        breaking_news=["Hot U.S. inflation hasn’t cooled off. Rising consumer prices to keep Fed rate hike on track."],
    )
    assert "FOMC Rate Decision" not in (cal.get("next_high_impact_event") or "")
    assert "Afternoon Treasury Auction & Liquidity Window" in (cal.get("next_high_impact_event") or "")

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

    # 1. Operational clock header uses Europe/Bucharest (EET / EEST)
    prompt_txt = build_supervisor_prompt(sensors=object(), logs=[])
    clock_line = next(l for l in prompt_txt.splitlines() if l.startswith("### 0. RUN TIMESTAMP & OPERATIONAL CLOCK:"))
    assert clock_line.endswith((" EET", " EEST"))

    # 2. extract_symbol_recent_fills resets LIFO PnL across flattened cycles & scopes to 72h swing window
    now_ms = 1_780_000_000_000
    day_ms = 86_400_000
    fills = extract_symbol_recent_fills(
        [
            # Closed historical cycle 10 days ago (+$500 profit) -- must NOT leak into active sub-cycle LIFO PnL
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

    venue, maker_bps, taker_bps, half_sp = _infer_symbol_venue_and_fees_bps("HYPEUSD", {"spread_bps": 4.0})
    assert venue == "kraken"
    assert isinstance(maker_bps, float) and maker_bps == 16.0
    assert isinstance(taker_bps, float) and taker_bps == 26.0
    assert half_sp == 2.0

    vp_pos = build_vp_position_dict(85.8225, 85.839, 83.7566, 86.9859)
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

    cold_entry = {"price": 77800.0, "kalman_vel": 0.0, "kalman_vel_std": 6.0, "kalman_trend": "FLAT"}
    dense_ser = [[(1700000000 + i * 30) * 1000, 77000.0 + i * 8.0] for i in range(100)]
    apply_24h_series_enrichment("BTCUSDC", cold_entry, {}, dense_ser, None)
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

    inflated_vol_entry = {"price": 227.2, "vol_1h_pct": 1.3634, "adapt_reentry_pct": 2.727, "adapt_dca_pct": 1.363, "slope_3_7m": 0.001, "slope_15m": 0.001, "slope_2_5h": -0.0001}
    one_hour_bars = [[(1700000000 + i * 60) * 1000, 227.0 + (i % 4) * 0.08] for i in range(60)]
    apply_24h_series_enrichment("TAOUSDC", inflated_vol_entry, {}, one_hour_bars, None)
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


    md_mod = load_venue_modules("212trading", "market_data")
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
    - compute_short_slopes_from_series keeps growth_* on the same scale as slope_* instead of inflating by 150,000x
    - harvest_log_anomalies_from_dirs deduplicates unstamped command stdout echoed in ERROR lines and clears resolved daemon errors after 'Daemon status: active'
    """

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

    # 2. compute_short_slopes_from_series keeps growth_2_5h on the same scale as slope_2_5h
    slopes = compute_short_slopes_from_series(rth_bars)
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
    dummy_gemini = DummyGeminiClient({
        "market_bias": "NEUTRAL",
        "risk_level": "MODERATE",
        "confidence": 0.7,
        "summary": "Balanced market.",
        "key_risks": ["Macro"],
        "recommended_action": "HOLD",
    })
    adv = SentimentAdvisor(
        gemini_client=dummy_gemini,
        cache_file=str(c_root / "macro" / "sentiment_eval.json"),
    )
    assert adv._effective_cache_dir() == str(c_root)
    adv.review(force_refresh=True, now_ts=1_793_751_600.0)
    assert dummy_gemini.last_prompt and "- BTC Intraday Structure:" in dummy_gemini.last_prompt

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
    cm_ctx = build_cross_market_context(
        {"SPCX_US_EQ": {"price": 138.36}},
        {},
        {"SPCX_US_EQ": spcx_cyc},
    )
    assert cm_ctx["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]["last_exit_kind"] == "REDUCE"
    assert cm_ctx["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]["realized_pnl_usd"] == -115.65

    # 9. Bug 16: _format_market_prices_section omits vs_btc(7d=...) when symbol has sub-24h history matching 24h range
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
    assert "sec_POC=$" in vp_str and "LVN=$" in vp_str
    # Bid wall at $85.50 on stale mid $86.405 is recentered below live $83.79 (~$82.91)
    assert "@ $82.91" in pt_hype.get("Orderbook Bid Wall Ahead", "")
    assert "@ $84.60" in pt_hype.get("Orderbook Ask Wall Ahead", "")
    assert "Last SELL=87.12 (by=CYCLE)" in pt_hype.get("Recent Venue Fills", "")

    # 14. Bugs 23, 24, 25: _format_recent_order_line 20bps swing dedup, build_cross_market_context bimodal/spot_cvd fields, and macro_intelligence.conf config audit
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

    # Also verify when _last_sell_ts_ms already matches last_sell.ts_ms on an already-deducted partial sell
    rgnt_cycs_synced = {
        "RGNT_US_EQ": {
            "qty": 122.51,
            "cost_usd": 176.35,
            "avg_cost_px": 1.4395,
            "entry_price": 1.50,
            "_last_sell_ts_ms": now_ms - 3600_000 * 50,
            "last_buy_price": 1.43,
            "last_sell_price": 1.37,
            "last_exit_kind": "STOP",
            "dca_buys": 0,
        }
    }
    rgnt_ro_synced = {
        "t212": {
            "RGNT_US_EQ": {
                "active_cycle_rem_qty": 122.51,
                "active_avg_buy_px": 1.4395,
                "lifo_realized_pnl_usd": -12.85,
                "last_buy": {"price": 1.43, "qty": 72.14, "ts_ms": now_ms - 3600_000 * 20},
                "last_sell": {"price": 1.37, "qty": 72.46, "ts_ms": now_ms - 3600_000 * 50},
                "recent_buys": [
                    {"price": 1.42, "qty": 57.14, "ts_ms": now_ms - 3600_000 * 60},
                    {"price": 1.43, "qty": 32.6, "ts_ms": now_ms - 3600_000 * 50 + 1000},
                    {"price": 1.43, "qty": 72.14, "ts_ms": now_ms - 3600_000 * 20},
                ],
                "recent_sells": [
                    {"price": 1.37, "qty": 72.46, "ts_ms": now_ms - 3600_000 * 50},
                ],
            }
        }
    }
    reconcile_venue_cycles_with_recent_orders(
        {},
        {},
        rgnt_cycs_synced,
        rgnt_ro_synced,
        {("t212", "RGNT_US_EQ"): 122.51},
        {"RGNT_US_EQ": {"price": 1.32}},
    )
    assert rgnt_cycs_synced["RGNT_US_EQ"]["last_exit_kind"] == "REDUCE"

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

    pt_tao = ptc_mod.collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(tao_cache))
    assert "Buyer/Seller ratio: 1.48" in pt_tao.get("Whale Taker Flow", "")
    assert pt_tao.get("15m Cumulative Volume Delta (CVD)", "").startswith("$+")

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
    assert "spot_cvd_15m=$+28,400" in pt_tao_44.get("15m Cumulative Volume Delta (CVD)", "")

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
    (cdir_36 / "cache_order_binance.json").write_text(
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

    def _ada_ohlc_series(sym, interval):
        n_bars = 168 if interval == 60 else 42
        step_s = interval * 60
        ts_list = [now_ag - (n_bars - 1 - idx) * step_s for idx in range(n_bars)]
        closes = [0.2072 if idx == 0 else (0.214343 if idx == 30 else 0.1879) for idx in range(n_bars)]
        return SimpleNamespace(timestamps=ts_list, closes=closes, lows=closes, highs=closes, volumes=[1000.0] * n_bars)

    enrich_24h_and_microstructure(
        str(cdir_36),
        str(tmp_path),
        pt_ag,
        {},
        {},
        {"ADAUSD": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=_ada_ohlc_series), provider_name="kraken")},
        now_ag,
    )
    enrich_cross_venue_trends(pt_ag, {}, {}, {}, {}, {}, watcher_ag)
    assert pt_ag["ADAUSD"]["high_7d"] == 0.214343
    assert pt_ag["ADAUSD"]["low_7d"] == 0.18252
    assert "_partial_7d_span_ms" not in pt_ag["ADAUSD"]

    # Wave 3: lowest_buy_px <= last_buy_px when LIFO pops dip buy + stale >14d open_lots ignored; va_pos_pct clamped [-100, 200]; sparse horizon dedup

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

    vp_gap = build_vp_position_dict(134.45, 141.18, 140.36, 142.47)
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
    # - apply_24h_series_enrichment computes fallback slopes/vol_1h for equities; collect_pretrade_telemetry syncs live equity price and strips post-RTH short tape
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
        f"{clk_now} [unknown] [STDERR] Traceback (most recent call last):\n"
        f'{clk_now} [unknown] [STDERR]   File "/app/macro_telemetry_collector.py", line 583, in harvest_macro_and_news_sensors\n'
        f"{clk_now} [unknown] [STDERR]     except Exception:\n"
        "UnboundLocalError: cannot access local variable 'datetime' where it is not associated with a value\n"
        f"{clk_now} [BINANCE][GUARD] account cache is not fresh: account_cache_writer_not_running\n"
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

    spcx_eq_entry: dict = {"price": 134.54}
    spcx_eq_pts = [[now_ms_w3 - (30 - i) * 300_000, 132.0 + i * 0.08] for i in range(31)]
    apply_24h_series_enrichment("SPCX_US_EQ", spcx_eq_entry, compute_24h_price_metrics(spcx_eq_pts), spcx_eq_pts, None)
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


    sw_ak = compute_swing_wave_metrics([0.193, 0.18252, 0.201272, 0.194847])
    assert sw_ak["swing_low_px"] == 0.18252
    assert sw_ak["swing_high_px"] == 0.201272
    assert sw_ak["rally_from_swing_low_pct"] == 6.75
    assert sw_ak["pullback_from_swing_high_pct"] == -3.19

    # Wave 6: collect_pretrade_telemetry omits deduplicated/stale horizons and sub-2.5h slopes on sparse series

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
    slopes_eq_live = compute_short_slopes_from_series(eq_5m_live)
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
    nvda_slopes = compute_short_slopes_from_series(nvda_5m_series)
    assert nvda_m24["chg_15m_pct"] < 0.0
    assert nvda_slopes["slope_15m"] < 0.0
    assert (nvda_slopes.get("growth_15m") or 0.0) <= 0.0

    ada_m = {"cvd_15m_usd": 37300.0, "cvd_5m_usd": -130100.0, "cvd_1h_usd": -845800.0, "ob_imbalance": 0.62, "bid_depth_usd": 310000.0, "ask_depth_usd": 190000.0}
    ada_p = {"price": 0.195748, "chg_15m_pct": -0.46, "vol_15m_usd": 150000.0}
    enrich_microstructure_cvd(ada_m, ada_p)
    assert ada_m["cvd_regime"] == "ask_absorbing"
    assert ada_m["spot_perp_flow_regime"] != "SYNCHRONIZED_SPOT_PERP_BUYING"

    # Wave 12: Discrepancies AR, AS, AT, AU

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
    rgnt_flat_slopes = compute_short_slopes_from_series(rgnt_flat_15m)
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

    # AV: On FOMC day (2026-11-05 at 00:05 EET = 2026-11-04 22:05 UTC), upcoming_windows includes the 21:00 EET FOMC window
    cal_av = resolve_macro_economic_calendar(datetime(2026, 11, 4, 22, 5, 0, tzinfo=timezone.utc).timestamp())
    assert cal_av["fomc_today"] is True
    assert any("FOMC Rate Decision" in w for w in cal_av["upcoming_windows"])

    # AW: KalmanStateTracker.update warm-starts from a collections.deque big_prices (PriceWindow.prices)
    tracker_aw = ss.KalmanStateTracker(state_path=None, compute_vol=False)
    deque_prices_aw = deque(220.0 + i * 0.005 for i in range(3600))
    k_aw = tracker_aw.update("TAOUSDC", 1_793_830_000.0, 238.0, epsilon=0.05, big_prices=deque_prices_aw, big_sample_rate=1.0)
    assert k_aw["kalman_vel"] > 0.0
    assert k_aw["kalman_vel_std"] < 5.0

    # Wave 14: Bugs 74-78

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

    # Bug 79: 5-minute equity series with trailing partial sub-5m duplicate Yahoo quote drops the duplicate bar
    base_ms_79 = 1_793_829_000_000
    eq_series_79 = [[base_ms_79 + i * 300_000, 134.0 + i * 0.08, 100.0] for i in range(13)]
    eq_series_79[-2][1] = 134.78
    eq_series_79[-1][1] = 135.05
    # Append trailing partial sub-5m duplicate quote 260s after last 5-minute bar
    eq_series_79.append([eq_series_79[-1][0] + 260_000, 135.05, 0.0])
    slopes_79 = compute_short_slopes_from_series(eq_series_79)
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
    - Bug 17: enrich_24h_and_microstructure tracks active_7d_vals from load_stitched_7d_ohlc_closes so ob_mid_m price adjustment does not overwrite chg_7d_pct with partial 1-day arch_vals_7d[0]
    - Bug 18: collect_pretrade_telemetry backfills zero slope_90s/growth_90s/trend_90s from non-diverged sibling live_<SYM>.json (e.g. ADAUSDC for ADAUSD)
    - Bug 20: harvest_log_anomalies_from_dirs ignores interactive [<stdin>:NN] lines and clears manage_agy errors after 'ui-sync: check healthy'
    """

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
    anom = harvest_log_anomalies_from_dirs([str(ldir)], lookback_hours=24.0)
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
    - Bug 26: resolve_fallback_ohlc_fn prefers binance:<SYM> over hyperliquid:<SYM> for multi-venue USDC pairs like TAOUSDC
    """

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

    # Bug 26: resolve_fallback_ohlc_fn prefers binance:TAOUSDC over hyperliquid:TAOUSDC
    bin_fn = lambda s, i: "binance_ohlc"  # noqa: E731
    hl_fn = lambda s, i: "hl_ohlc"  # noqa: E731
    inst_map = {
        "TAOUSDC": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=hl_fn)),
        "binance:TAOUSDC": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=bin_fn)),
        "hyperliquid:TAOUSDC": SimpleNamespace(_provider=SimpleNamespace(ohlc_series=hl_fn)),
    }
    assert resolve_fallback_ohlc_fn("TAOUSDC", inst_map) is bin_fn

    # Bug 29: parse_cited_taker_ratio handles parenthetical '(ratio 0.60, 1h CVD -$25.46M)' and ranks 1h above 5m

    live_summary = (
        "and a sharp 5m taker buy resurgence (ratio 1.64, CVD +$9.25M). "
        "Despite short-term bid absorption, aggregate 1h taker flow remains net negative (ratio 0.60, 1h CVD -$25.46M)"
    )
    assert parse_cited_taker_ratio(live_summary) == 0.60

    # Bug 30b: extract_symbol_recent_fills clears lifo_realized_pnl_usd when a full exit of >7d-old lots is followed by a new cycle BUY

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
    res = harvest_log_anomalies_from_dirs([str(log_dir)], 999999)
    assert res.error_count == 0
    assert res.warning_count == 2
    assert res.guard_interventions == ["2026-11-05 03:39:00,000 [INFO] [GUARD] AssetGuardian tier 1 triggered for ETHUSDC"]

    # 3. enrich_microstructure_cvd treats negative 15m CVD with high bid dominance (>=0.65), accumulation regime,
    #    and mild 15m drift (-0.09% >= -0.15%) as spot_absorbing rather than aggressive_market_dumps

    btc_micro = {"cvd_15m_usd": -2_225_828.0, "ob_imbalance": 0.71, "divergence_regime": "accumulation"}
    enrich_microstructure_cvd(btc_micro, {"price": 80673.07, "chg_15m_pct": -0.09})
    assert btc_micro["cvd_regime"] == "spot_absorbing"

    # 4. harvest_tradable_positions_and_orders preserves cached position price and ts when get_current_price returns None

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


def test_bugs_96_to_98_breadth_base_dedup_multi_venue_orderbook_isolation_and_slope_zero_sign(tmp_path, monkeypatch):
    """Verify:
    - Bug 96: MarketBreadthCollector.parse_tickers and harvest_macro_and_news_sensors deduplicate top_gainers and top_losers by canonical base asset (e.g. ZECUSDC + ZECUSDT do not occupy 2 of 3 top-gainer slots)
    - Bug 97: OrderbookDepthCollector.fetch isolates multi-venue symbols (TAOUSDC on Binance vs TAOUSD on Kraken; HYPEUSDC on Hyperliquid vs HYPEUSD on Kraken) so transient primary-venue timeouts fall back to _fallback_stale instead of polluting orderbook_depth_<SYM>.json or _preferred_url with a sibling venue's book
    - Bug 98: build_cross_market_context normalizes -0.0 slope values to +0.0
    """

    # 1. Bug 96: MarketBreadthCollector.parse_tickers deduplicates ZECUSDC and ZECUSDT in top_gainers
    tickers = [
        {"symbol": "INJUSDT", "lastPrice": "25.0", "priceChangePercent": "12.994", "quoteVolume": "25000000"},
        {"symbol": "ZECUSDC", "lastPrice": "45.0", "priceChangePercent": "9.939", "quoteVolume": "15000000"},
        {"symbol": "ZECUSDT", "lastPrice": "44.9", "priceChangePercent": "9.833", "quoteVolume": "35000000"},
        {"symbol": "SOLUSDT", "lastPrice": "180.0", "priceChangePercent": "7.500", "quoteVolume": "80000000"},
        {"symbol": "BTCUSDC", "lastPrice": "80600.0", "priceChangePercent": "5.100", "quoteVolume": "500000000"},
        {"symbol": "HUMAUSDT", "lastPrice": "0.05", "priceChangePercent": "-18.098", "quoteVolume": "20000000"},
        {"symbol": "HUMAUSDC", "lastPrice": "0.05", "priceChangePercent": "-17.950", "quoteVolume": "15000000"},
        {"symbol": "TRUMPUSDT", "lastPrice": "8.5", "priceChangePercent": "-17.784", "quoteVolume": "40000000"},
        {"symbol": "BMTUSDT", "lastPrice": "0.12", "priceChangePercent": "-9.710", "quoteVolume": "18000000"},
    ]
    mb_snap = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=5_000_000.0, now_ts=1793845000.0)
    assert mb_snap is not None
    assert mb_snap.total_count == 7
    assert mb_snap.advancing_count == 4
    assert mb_snap.declining_count == 3
    assert [g[0] for g in mb_snap.top_gainers] == ["INJUSDT", "ZECUSDT", "SOLUSDT"]
    assert [l[0] for l in mb_snap.top_losers] == ["HUMAUSDT", "TRUMPUSDT", "BMTUSDT"]

    # Also verify harvest_macro_and_news_sensors deduplicates cached market_breadth_collect.json top_gainers/top_losers
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    (macro_dir / "market_breadth_collect.json").write_text(
        json.dumps(
            {
                "advance_ratio": 0.7634,
                "regime": "BULLISH_BREADTH",
                "top_gainers": [["INJUSDT", 12.994], ["ZECUSDC", 9.939], ["ZECUSDT", 9.833], ["SOLUSDT", 7.5]],
                "top_losers": [["HUMAUSDT", -18.098], ["HUMAUSDC", -17.95], ["TRUMPUSDT", -17.784]],
                "ts": 1793845000.0,
            }
        ),
        encoding="utf-8",
    )
    macro_sensors, _ = harvest_macro_and_news_sensors(str(tmp_path), 1793845060.0)
    mb_cached = macro_sensors["market_breadth"]
    assert [m[0] for m in mb_cached["top_gainers"]] == ["INJUSDT", "ZECUSDC", "SOLUSDT"]
    assert [m[0] for m in mb_cached["top_losers"]] == ["HUMAUSDT", "TRUMPUSDT"]

    # 2. Bug 97: OrderbookDepthCollector.fetch does not cross-pollute TAOUSDC with Kraken TAOUSD on transient Binance failure
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), cache_ttl_sec=10.0, disk_ttl_sec=60.0)
    ob_col.record_snapshot(
        OrderbookSnapshot(
            symbol="TAOUSDC",
            mid_price=235.75,
            bid_depth_usd=169765.0,
            ask_depth_usd=101824.0,
            imbalance_ratio=0.625,
            largest_bid_wall_usd=30224.0,
            largest_bid_wall_price=235.3,
            largest_ask_wall_usd=18009.0,
            largest_ask_wall_price=236.5,
            ts=1000.0,
            best_bid=235.7,
            best_ask=235.8,
            spread_bps=4.24,
        ),
        save_to_disk=True,
    )
    called_urls = []

    def _fake_fetch_binance_down(url, *args, **kwargs):
        called_urls.append(url)
        if "binance.com" in url:
            raise RuntimeError("transient binance timeout")
        return {"result": {"TAOUSD": {"bids": [["235.8", "3000.0"]], "asks": [["235.9", "1500.0"]]}}}

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_fetch_binance_down)
    snap_tao = ob_col.fetch("TAOUSDC", allow_network=True, now=1100.0)
    assert snap_tao is not None
    assert snap_tao.bid_depth_usd == 169765.0
    assert all("kraken.com" not in u and "hyperliquid.xyz" not in u for u in called_urls)
    assert "TAOUSDC" not in ob_col._preferred_url

    # 3. Bug 98: build_cross_market_context normalizes -0.0 slope values to +0.0
    cm_ctx = build_cross_market_context(
        {"ARBUSDC": {"price": 0.16335, "slope_90s": -0.0, "kalman_vel": -0.0}},
        {},
    )
    arb_cb = cm_ctx["crypto_bellwethers"]["ARB"]
    assert arb_cb["slope_90s"] == 0.0
    assert math.copysign(1.0, arb_cb["slope_90s"]) == 1.0
    assert math.copysign(1.0, arb_cb["kalman_vel"]) == 1.0


def test_parse_cited_taker_ratio_key_risks_and_guard_bypass_notice(tmp_path):
    """Verify:
    - parse_cited_taker_ratio parses plural 'taker ratios (overall 0.60, 5m 0.67)' across summary + key_risks and ranks overall (0.60) above 1h (1.03)
    - enrich_sentiment_advisor_live_flow_divergence does not emit false LIVE_SELL_REVERSAL on BULLISH/ACCUMULATE when key_risks already cited overall 0.60 and live taker_ratio improved to 0.70
    - harvest_log_anomalies_from_dirs ignores informational '[GUARD] ... the historical price reference was explicitly bypassed' lines
    """

    summary = (
        "BTC orderbook depth is heavily bid-dominated (0.69 imbalance) with positive 1h net buying flow "
        "(1h taker ratio 1.03, 1h CVD +$5.67M) and negative funding (-0.31 bps)."
    )
    key_risks = [
        "Deeply negative 4h cumulative volume delta (-$107.7M) and sub-1.0 short-term taker ratios (overall 0.60, 5m 0.67) show persistent active selling into limit bids."
    ]
    assert parse_cited_taker_ratio((summary, key_risks)) == 0.60

    sa_acc = {
        "market_bias": "BULLISH",
        "recommended_action": "ACCUMULATE",
        "age_min": 14.9,
        "summary": summary,
        "key_risks": key_risks,
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_acc,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.7035,
                "cvd_15m_usd": -8_660_000.0,
                "taker_ratio_5m": 0.7478,
                "cvd_5m_usd": -2_870_000.0,
                "taker_ratio_1h": 0.9613,
                "cvd_1h_usd": -7_410_000.0,
            }
        },
    )
    assert "live_flow_divergence" not in sa_acc

    log_file = tmp_path / "bot_tao.log"
    log_file.write_text(
        "04:41:11 [GUARD] SELL TAOUSDC: the historical price reference was explicitly bypassed; the quantity/weight guard remains active\n"
        "04:41:12 [GUARD] BUY TAOUSDC: the historical price reference was explicitly bypassed; the quantity/weight guard remains active\n",
        encoding="utf-8",
    )
    res = harvest_log_anomalies_from_dirs([str(tmp_path)], 999999)
    assert res.guard_interventions == []
    assert res.critical_events == []


    btc_neutral_abs = {
        "ob_imbalance": 0.718,
        "bid_depth_usd": 8_952_909.0,
        "ask_depth_usd": 3_513_606.0,
        "taker_buy_vol_usd": 20_543_994.72,
        "taker_sell_vol_usd": 21_210_688.22,
        "cvd_15m_usd": -666_694.0,
        "divergence_regime": "neutral",
    }
    enrich_microstructure_cvd(btc_neutral_abs, {"price": 80514.3, "chg_15m_pct": -0.07})
    assert btc_neutral_abs["cvd_regime"] == "spot_absorbing"

    ada_accum_abs = {
        "ob_imbalance": 0.59,
        "bid_depth_usd": 1_165_159.0,
        "ask_depth_usd": 724_691.0,
        "cvd_15m_usd": -47_336.0,
        "divergence_regime": "accumulation",
    }
    enrich_microstructure_cvd(ada_accum_abs, {"price": 0.19835, "chg_15m_pct": -0.07})
    assert ada_accum_abs["cvd_regime"] == "spot_absorbing"


    (tmp_path / "orderbook_depth_HYPEUSD.json").write_text(
        json.dumps({"symbol": "HYPEUSD", "mid_price": 84.0, "imbalance_ratio": 0.615, "ts": time.time()}),
        encoding="utf-8",
    )
    pre_hype = collect_pretrade_telemetry("HYPEUSD", cachedb_dir=str(tmp_path))
    assert pre_hype["Orderbook Depth Imbalance"] == "0.62 (62% bids vs 38% asks)"


def test_bugs_99_to_102_breadth_exclusions_btc_vol_share_neg_zero_benchmarks_and_suppressed_growth_trend(tmp_path):

    # Bug 99 & Bug 100: MarketBreadthCollector excludes tokenized equities/ETFs (MSTRBUSDT, SOXLBUSDT),
    # newer stablecoins (USD1USDT), and leveraged tokens (BTCUPUSDT), does NOT falsely exclude NEURUSDT,
    # and includes BTCFDUSD / ETHFDUSD volume in btc_vol_usd / eth_vol_usd rather than altcoin volume.
    tickers = [
        {"symbol": "BTCUSDT", "lastPrice": "80000.0", "priceChangePercent": "2.0", "quoteVolume": "100000000.0"},
        {"symbol": "BTCFDUSD", "lastPrice": "80010.0", "priceChangePercent": "2.0", "quoteVolume": "100000000.0"},
        {"symbol": "ETHUSDT", "lastPrice": "2500.0", "priceChangePercent": "3.0", "quoteVolume": "50000000.0"},
        {"symbol": "ETHFDUSD", "lastPrice": "2501.0", "priceChangePercent": "3.0", "quoteVolume": "50000000.0"},
        {"symbol": "SOLUSDT", "lastPrice": "150.0", "priceChangePercent": "8.5", "quoteVolume": "40000000.0"},
        {"symbol": "TAOUSDT", "lastPrice": "240.0", "priceChangePercent": "6.2", "quoteVolume": "30000000.0"},
        {"symbol": "NEURUSDT", "lastPrice": "0.05", "priceChangePercent": "5.1", "quoteVolume": "20000000.0"},
        {"symbol": "ARBUSDT", "lastPrice": "0.15", "priceChangePercent": "-3.4", "quoteVolume": "10000000.0"},
        {"symbol": "MSTRBUSDT", "lastPrice": "350.0", "priceChangePercent": "15.0", "quoteVolume": "25000000.0"},
        {"symbol": "SOXLBUSDT", "lastPrice": "28.0", "priceChangePercent": "-18.0", "quoteVolume": "25000000.0"},
        {"symbol": "USD1USDT", "lastPrice": "1.0", "priceChangePercent": "0.01", "quoteVolume": "50000000.0"},
        {"symbol": "BTCUPUSDT", "lastPrice": "12.0", "priceChangePercent": "22.0", "quoteVolume": "20000000.0"},
    ]
    snap = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=5_000_000.0, now_ts=1_776_000_000.0)
    assert snap is not None
    gainer_syms = [s for s, _ in snap.top_gainers]
    loser_syms = [s for s, _ in snap.top_losers]
    assert "MSTRBUSDT" not in gainer_syms
    assert "BTCUPUSDT" not in gainer_syms
    assert "SOXLBUSDT" not in loser_syms
    assert "NEURUSDT" in gainer_syms
    # Total eligible crypto vol = 100M (BTCUSDT) + 100M (BTCFDUSD) + 50M (ETHUSDT) + 50M (ETHFDUSD) + 100M (alts) = 400M
    assert snap.btc_vol_share_pct == 50.0
    assert snap.eth_vol_share_pct == 25.0
    assert snap.alt_vol_share_pct == 25.0

    # Bug 99 & Bug 101 in harvest_macro_and_news_sensors: cached market_breadth filters MSTRBUSDT/SOXLBUSDT
    # and cached macro_benchmarks normalizes -0.00% in items and summary_line.
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    (macro_dir / "market_breadth_collect.json").write_text(
        json.dumps(
            {
                "timestamp": 1_776_000_000.0,
                "top_gainers": [["INJUSDT", 12.45], ["MSTRBUSDT", 9.98], ["ZECUSDC", 8.1]],
                "top_losers": [["HUMAUSDT", -19.1], ["SOXLBUSDT", -6.85], ["TRUMPUSDT", -5.2]],
            }
        ),
        encoding="utf-8",
    )
    (macro_dir / "macro_benchmarks_collect.json").write_text(
        json.dumps(
            {
                "ts": 1_776_000_000.0,
                "items": {
                    "DXY": {
                        "symbol": "DXY",
                        "unit": "index",
                        "price": 101.3,
                        "chg_24h_pct": -0.27,
                        "chg_1h_pct": -0.0,
                        "chg_15m_pct": 0.01,
                    }
                },
                "summary_line": "DXY=101.30 (-0.27% 24h, -0.00% 1h, +0.01% 15m)",
            }
        ),
        encoding="utf-8",
    )
    sensors, _ = harvest_macro_and_news_sensors(str(tmp_path), now_ts=1_776_000_030.0)
    mb_harvested = sensors["market_breadth"]
    assert mb_harvested is not None
    assert [m[0] for m in mb_harvested["top_gainers"]] == ["INJUSDT", "ZECUSDC"]
    assert [m[0] for m in mb_harvested["top_losers"]] == ["HUMAUSDT", "TRUMPUSDT"]
    bm_harvested = sensors["macro_benchmarks"]
    assert bm_harvested is not None
    assert "-0.00%" not in bm_harvested["summary_line"]
    assert "+0.00% 1h" in bm_harvested["summary_line"]
    assert "-0.00%" not in format_macro_benchmarks_summary_line(bm_harvested)

    # Bug 102: when growth_15m is suppressed on a 5m-cadence series with flat 15m endpoints (slope_15m == 0.0),
    # trend_15m must use slope_15m (0) rather than the suppressed intra-window growth_15m (-1).
    t0_ms = 1_776_000_000_000
    pts_5m = [[t0_ms + i * 300_000, 100.0] for i in range(12)]
    pts_5m[-2][1] = pts_5m[-3][1] = 100.5  # intermediate bar higher than both -15m (100.0) and now (100.0)
    slopes_5m = compute_short_slopes_from_series(pts_5m)
    assert slopes_5m["slope_15m"] == 0.0
    assert "growth_15m" not in slopes_5m
    assert slopes_5m["trend_15m"] == 0


def test_recalculate_live_price_geometry_6h_bottom_invalidation_and_equal_low_bounce():

    # 1. When LIQUIDITY_SWEEP_RECLAIM drops back below prior_6h_low_px (>2.5% above recent_1h_low_px),
    # bottom_structure_6h is cleared rather than left as an un-reclaimed LIQUIDITY_SWEEP_RECLAIM.
    pinfo_broken = {
        "prior_6h_low_px": 100.0,
        "recent_1h_low_px": 95.0,
        "bottom_structure_6h": "LIQUIDITY_SWEEP_RECLAIM",
        "reclaim_above_prior_low_pct": 0.5,
    }
    recalculate_live_price_geometry(pinfo_broken, 98.0)
    assert "bottom_structure_6h" not in pinfo_broken
    assert "reclaim_above_prior_low_pct" not in pinfo_broken

    # 2. When FRESH_6H_LOW_TEST with equal lows (recent_1h_low_px == prior_6h_low_px) bounces >0.2% above prior_6h_low_px,
    # it upgrades to HIGHER_LOW_DOUBLE_BOTTOM with higher_low_vs_prior_pct == 0.0.
    pinfo_eq = {
        "prior_6h_low_px": 100.0,
        "recent_1h_low_px": 100.0,
        "bottom_structure_6h": "FRESH_6H_LOW_TEST",
    }
    recalculate_live_price_geometry(pinfo_eq, 100.5)
    assert pinfo_eq["bottom_structure_6h"] == "HIGHER_LOW_DOUBLE_BOTTOM"
    assert pinfo_eq["higher_low_vs_prior_pct"] == 0.0


def test_parenthetical_multi_item_taker_ratio_bullish_aligned_flow_and_no_negative_zero_dd():

    # 1. Multi-item parenthetical where 5m comes BEFORE overall/15m/1h
    assert parse_cited_taker_ratio("taker ratios (5m 0.67, overall 0.60)") == 0.60
    assert parse_cited_taker_ratio("aggressive short-term taker buying (5m ratio 2.5385, 15m ratio 1.4319)") == 1.4319
    assert parse_cited_taker_ratio("aggressive short-term taker buying (5m ratio 2.5385, 1h ratio 1.1636)") == 1.1636

    # 2. BULLISH advisor that already cited bullish taker buying (cited_tr >= 1.10) does not trigger false LIVE_FLOW_RECOVERY
    sa_bull_aligned = {
        "market_bias": "BULLISH",
        "recommended_action": "HOLD",
        "age_min": 16.0,
        "summary": "BTC maintains strong bullish momentum backed by aggressive short-term taker buying (1h taker ratio 1.1636, 5m ratio 2.5385).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_bull_aligned,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.48, "taker_ratio_1h": 1.32, "cvd_15m_usd": 8_000_000.0}},
    )
    assert "live_flow_divergence" not in sa_bull_aligned

    # 3. Tiny sub-basis-point drawdown from peak (e.g. TAOUSD 237.35145 vs trail_peak 237.3584) formats as 0.0%, never -0.0%
    pos_tao = [
        {
            "venue": "kraken",
            "symbol": "TAOUSD",
            "quote": "USD",
            "free_base": 0.073518,
            "locked_base_qty": 3.63336,
            "price": 235.0,
            "trailing_peak": 237.3584,
            "active_avg_buy_px": 235.741094,
        }
    ]
    krk_cycles = {
        "TAOUSD": {
            "qty": 3.706878,
            "cost_usd": 873.86,
            "avg_cost_px": 235.741094,
            "entry_price": 235.741094,
            "pos_peak": 237.3584,
            "dca_buys": 0,
        }
    }
    enrich_positions_with_orders_and_cycles(
        pos_tao,
        {"TAOUSD": {"price": 237.35145}},
        {},
        {},
        {},
        krk_cycles,
        {},
        {},
        {},
    )
    pos_line = _format_tradable_position_line(pos_tao[0])
    cyc_line = _format_venue_cycle_line("kraken", "TAOUSD", krk_cycles["TAOUSD"])
    assert "dd_from_peak=-0.0%" not in pos_line
    assert "dd_from_peak=0.0%" in pos_line
    assert "dd_from_peak=-0.0%" not in cyc_line
    assert "dd_from_peak=0.0%" in cyc_line


def test_lexicon_noise_pruning_publisher_suffix_stale_swing_lots_and_manifest_heartbeats(tmp_path):

    # 1. Bug 103: Partisan election war post-mortems, fast-food discount wars, and sports striker/golf liquidation
    #    do not trigger severity keywords or discover dynamic crisis bigrams
    assert (
        filter_valid_severity_keywords(
            ["war", "iran war"],
            title="Trump blames Iran war for midterms losses as GOP concedes role",
        )
        == ()
    )
    assert (
        filter_valid_severity_keywords(
            ["war", "discount wars"],
            title="McDonald's tests new value menu as fast-food discount wars heat up",
        )
        == ()
    )
    assert (
        filter_valid_severity_keywords(
            ["liquidation", "strike"],
            title="LIV Golf faces liquidation as Tenerife striker signs new deal",
        )
        == ()
    )
    discovered = discover_dynamic_terms_from_headlines(
        [
            {"title": "Trump blames Iran war for midterms losses - CNN", "source": "CNN", "url": "https://a.com/1"},
            {"title": "Trump blames Iran war for midterms losses - BBC", "source": "BBC", "url": "https://b.com/2"},
            {"title": "McDonald's tests new menu as fast-food discount wars heat up - CNBC", "source": "CNBC", "url": "https://c.com/3"},
            {"title": "Burger chains battle as fast-food discount wars heat up - Reuters", "source": "Reuters", "url": "https://d.com/4"},
            {"title": "LIV Golf faces liquidation after tournament - ESPN", "source": "ESPN", "url": "https://e.com/5"},
            {"title": "LIV Golf faces liquidation in sports shakeup - Sky", "source": "Sky", "url": "https://f.com/6"},
        ],
        persist=False,
    )
    assert discovered == ()

    # 2. Bug 104: _strip_headline_publisher_suffix preserves internal em-dashes and strips multi-segment publisher taglines
    assert (
        _strip_headline_publisher_suffix(
            "Stock Market Today: Dow Jumps 1,500 Points On Sweep — Live Updates - WSJ"
        )
        == "Stock Market Today: Dow Jumps 1,500 Points On Sweep — Live Updates"
    )
    assert (
        _strip_headline_publisher_suffix(
            "Stock Market Today: Dow Jumps 1,500 Points On Sweep — Live Updates"
        )
        == "Stock Market Today: Dow Jumps 1,500 Points On Sweep — Live Updates"
    )
    assert (
        _strip_headline_publisher_suffix(
            "JetBlue raises baggage fees for a second time as jet fuel prices stay elevated - ABC News - Breaking News, Latest News and Videos"
        )
        == "JetBlue raises baggage fees for a second time as jet fuel prices stay elevated"
    )

    # 3. Bug 105: extract_symbol_recent_fills does not promote >72h legacy lots or single-lot LIFO remainders as tactical_avg_buy_px after sells
    now_ms = 1_793_805_000_000
    day_ms = 86_400_000
    fills_stale_core = extract_symbol_recent_fills(
        [
            {"orderId": "1", "side": "BUY", "price": 258.5, "executedQty": 5.0, "status": "FILLED", "time": now_ms - 12 * day_ms},
            {"orderId": "2", "side": "BUY", "price": 236.4, "executedQty": 4.5, "status": "FILLED", "time": now_ms - 5 * day_ms},
            {"orderId": "3", "side": "BUY", "price": 224.5, "executedQty": 4.0, "status": "FILLED", "time": now_ms - 1 * day_ms},
            {"orderId": "4", "side": "SELL", "price": 234.8, "executedQty": 4.0, "status": "FILLED", "time": now_ms - 3_600_000},
        ],
        symbol="TAOUSDC",
    )
    assert fills_stale_core is not None
    assert fills_stale_core["swing_subcycle_avg_px"] is None
    assert "tactical_avg_buy_px" not in fills_stale_core

    fills_single_rem = extract_symbol_recent_fills(
        [
            {"orderId": "1", "side": "BUY", "price": 0.1595, "executedQty": 1000.0, "status": "FILLED", "time": now_ms - 2 * day_ms},
            {"orderId": "2", "side": "BUY", "price": 0.1540, "executedQty": 4000.0, "status": "FILLED", "time": now_ms - 1 * day_ms},
            {"orderId": "3", "side": "SELL", "price": 0.1640, "executedQty": 4774.3, "status": "FILLED", "time": now_ms - 1_800_000},
        ],
        symbol="ARBUSDC",
    )
    assert fills_single_rem is not None
    assert fills_single_rem["swing_subcycle_avg_px"] is None
    assert "tactical_avg_buy_px" not in fills_single_rem

    # 4. Bug 106: harvest_heartbeats_and_state_files honors procs.conf hb_stale_s, and _format_sre_and_fleet_section separates stopped process heartbeats
    ws_dir = tmp_path / "ws_hb"
    c_dir = ws_dir / "cachedb"
    (c_dir / "system" / "heartbeat").mkdir(parents=True, exist_ok=True)
    (ws_dir / "procs.conf").write_text(
        "cacheManager.py|$ROOT|python cacheManager.py|cacheManager|cachedb/cacheManager.heartbeat|180|fleet\n"
        "xStock-watch.py|$ROOT|python xStock-watch.py|xStock-watch|cachedb/xStock-watch.heartbeat|1800|bot\n",
        encoding="utf-8",
    )
    now_s = 1_800_000_000.0
    (c_dir / "system" / "heartbeat" / "cacheManager.heartbeat").write_text(
        json.dumps({"status": "alive", "ts": now_s - 400.0}),
        encoding="utf-8",
    )
    (c_dir / "system" / "heartbeat" / "xStock-watch.heartbeat").write_text(
        json.dumps({"status": "alive", "ts": now_s - 400.0}),
        encoding="utf-8",
    )
    hbs_manifest, _ = harvest_heartbeats_and_state_files(str(c_dir), str(ws_dir), now_s)
    assert "cacheManager.heartbeat" not in hbs_manifest
    assert "xStock-watch.heartbeat" in hbs_manifest

    sre_lines = _format_sre_and_fleet_section(
        SimpleNamespace(
            active_heartbeats={"xStock-watch.heartbeat": {"age_seconds": 400.0}, "auto_cryptotrade_executor.heartbeat": {"age_seconds": 60.0}},
            state_files={},
            fleet_processes=[{"name": "xStock-watch", "is_running": False, "heartbeat_fresh": True}],
            system_and_vpn={},
        ),
        LogEventSummary(),
        1.0,
    )
    hb_line = next(l for l in sre_lines if l.startswith("- Active Heartbeats:"))
    assert "['auto_cryptotrade_executor.heartbeat']" in hb_line
    assert "disk heartbeats for stopped processes: ['xStock-watch.heartbeat']" in hb_line


def test_geopolitical_zero_high_sev_digest_reset_and_false_positive_shield_cleanup(tmp_path):

    # 1. extract_high_severity_headlines_and_digest drops headlines whose matched_keywords are all excluded metaphors
    snap_talent_war = {
        "headlines": [
            {
                "title": "Inside the chip talent war: Trade secrets and rising Chinese rivals",
                "source": "CNBC",
                "matched_keywords": ["war"],
                "is_high_severity": True,
            }
        ]
    }
    high_sev, digest = extract_high_severity_headlines_and_digest(snap_talent_war)
    assert high_sev == []
    assert digest == ""
    assert (
        filter_valid_severity_keywords(
            ["war", "iran war"],
            title="JD Vance Breaks With Trump on Iran War and Israel, Positions Himself Ahead of 2028",
        )
        == ()
    )

    # 2. GeopoliticalThreatAnalyzer.assess immediately resets cached headline-based assessment to calm NORMAL
    #    when news_snapshot has headlines and high_sev is empty, even before cache_ttl_sec expires
    geo_file = tmp_path / "geopolitical_eval.json"
    geo_file.write_text(
        json.dumps(
            {
                "threat_level": "NORMAL",
                "risk_score": 0.12,
                "summary": "The story highlights competitive pressures in the semiconductor talent market.",
                "recommended_brake": "NONE",
                "headlines_analyzed": 1,
                "ts": 1_793_850_717.0,
                "is_fallback": False,
                "headlines_digest": "Inside the chip talent war: Trade secrets and rising Chinese rivals",
            }
        ),
        encoding="utf-8",
    )
    analyzer = GeopoliticalThreatAnalyzer(state_file=str(geo_file), cache_ttl_sec=16200.0)
    res_calm = analyzer.assess(snap_talent_war, now_ts=1_793_851_000.0)
    assert res_calm.threat_level == "NORMAL"
    assert res_calm.risk_score == 0.1
    assert res_calm.summary == "Calm geopolitical and energy climate. No systemic shock detected."
    assert analyzer._last_headlines_digest == ""

    # 3. harvest_macro_and_news_sensors resets geopolitical_threat when its headlines_digest only contains excluded metaphors
    c_dir = tmp_path / "cachedb_geo"
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "geopolitical_eval.json").write_text(
        json.dumps(
            {
                "threat_level": "NORMAL",
                "risk_score": 0.12,
                "summary": "The story highlights competitive pressures in the semiconductor talent market.",
                "recommended_brake": "NONE",
                "headlines_analyzed": 1,
                "ts": 1_793_850_717.0,
                "is_fallback": False,
                "headlines_digest": "Inside the chip talent war: Trade secrets and rising Chinese rivals",
            }
        ),
        encoding="utf-8",
    )
    (c_dir / "news_feed_collect.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "title": "Inside the chip talent war: Trade secrets and rising Chinese rivals",
                        "source": "CNBC",
                        "matched_keywords": ["war"],
                        "is_high_severity": True,
                        "first_seen_ts": 1_793_850_700.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sensors, b_news = harvest_macro_and_news_sensors(str(c_dir), 1_793_851_300.0)
    assert len(b_news) == 1
    assert b_news[0]["is_high_severity"] is False
    assert sensors["geopolitical_threat"]["summary"] == "Calm geopolitical and energy climate. No systemic shock detected."
    assert sensors["geopolitical_threat"]["risk_score"] == 0.1
    assert sensors["geopolitical_threat"]["headlines_digest"] == ""


def test_position_peak_24h_reconciliation_and_semantic_dup_and_offtopic_news(tmp_path):

    # 1. Held positions with missing or un-warmed trailing_peak (equal to active_avg_buy_px) and age >= 12h
    #    reconcile trailing_peak and cycle pos_peak with high_24h and last_sell_px
    now_ms = int(time.time() * 1000)
    old_buy_ts = now_ms - 48 * 3600 * 1000
    recent_sell_ts = now_ms - 13 * 3600 * 1000
    positions = [
        {
            "venue": "kraken",
            "symbol": "TAOUSD",
            "base": "TAO",
            "quote": "USD",
            "free_base": 3.7024,
            "price": 234.9376,
            "trailing_peak": 235.7411,
        },
        {
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "base": "HYPE",
            "quote": "USDC",
            "free_base": 15.5746,
            "price": 83.5255,
            "trailing_peak": None,
        },
    ]
    prices_and_trends = {
        "TAOUSD": {"price": 234.9376, "high_24h": 240.1108, "low_24h": 220.502},
        "HYPEUSDC": {"price": 83.5255, "high_24h": 88.7365, "low_24h": 83.400},
        "NVDA_US_EQ": {"price": 198.41, "high_24h": 210.7, "low_24h": 198.41},
    }
    venue_ro = {
        "kraken": {
            "TAOUSD": {
                "last_action_side": "BUY",
                "active_avg_buy_px": 235.741094,
                "last_buy": {"price": 235.741094, "qty": 3.7024, "ts_ms": old_buy_ts},
                "open_lots": [{"qty": 3.7024, "entry_px": 235.741094, "ts_ms": old_buy_ts}],
            }
        },
        "hyperliquid": {
            "HYPEUSDC": {
                "last_action_side": "BUY",
                "active_avg_buy_px": 85.658865,
                "last_buy": {"price": 83.8, "qty": 5.75, "ts_ms": now_ms - 8 * 3600 * 1000},
                "last_sell": {"price": 88.0, "qty": 5.68, "ts_ms": recent_sell_ts},
                "open_lots": [
                    {"qty": 9.8246, "entry_px": 86.75, "ts_ms": old_buy_ts},
                    {"qty": 5.75, "entry_px": 83.8, "ts_ms": now_ms - 8 * 3600 * 1000},
                ],
            }
        },
        "t212": {
            "NVDA_US_EQ": {
                "last_action_side": "BUY",
                "active_avg_buy_px": 205.8293,
                "last_buy": {"price": 205.65, "qty": 0.38, "ts_ms": recent_sell_ts},
                "last_sell": {"price": 207.9, "qty": 0.93, "ts_ms": old_buy_ts},
                "open_lots": [
                    {"qty": 0.03, "entry_px": 207.93, "ts_ms": old_buy_ts},
                    {"qty": 0.38, "entry_px": 205.65, "ts_ms": recent_sell_ts},
                ],
            }
        },
    }
    krk_cycles = {"TAOUSD": {"qty": 3.7024, "avg_cost_px": 235.741094, "pos_peak": None}}
    hl_cycles = {"HYPEUSDC": {"qty": 15.5746, "avg_cost_px": 85.658865, "pos_peak": None}}
    t212_cycles = {"NVDA_US_EQ": {"qty": 0.41, "avg_cost_px": 205.8293, "entry_price": 207.9271, "pos_peak": None}}
    enrich_positions_with_orders_and_cycles(
        positions,
        prices_and_trends,
        {},
        venue_ro,
        {},
        krk_cycles,
        hl_cycles,
        t212_cycles,
        {},
    )
    assert positions[0]["trailing_peak"] == 240.1108
    assert krk_cycles["TAOUSD"]["pos_peak"] == 240.1108
    assert positions[0]["dd_from_peak_pct"] < 0.0
    assert positions[1]["trailing_peak"] == 88.7365
    assert hl_cycles["HYPEUSDC"]["pos_peak"] == 88.7365
    assert positions[1]["dd_from_peak_pct"] < 0.0
    assert t212_cycles["NVDA_US_EQ"]["pos_peak"] == 210.7
    assert t212_cycles["NVDA_US_EQ"]["dd_from_peak_pct"] == -5.83

    # 2. Syndicated partisan rewrites sharing >= 3 specific entity stems and >= 0.40 overlap are deduplicated
    assert headlines_are_semantic_duplicates(
        "JD Vance Breaks With Trump on Iran War and Israel, Positions Himself Ahead of 2028",
        "Vance walks Trump tightrope with scathing election critique blaming Iran war",
    )
    assert not headlines_are_semantic_duplicates(
        "Iran war sends crude oil above $95 in London trading",
        "Iran war threatens global shipping lanes in Red Sea",
    )

    # 3. Off-topic sports/tabloid filler is deprioritized behind financial/geopolitical news when selecting top 6
    assert is_off_topic_general_news_title("World record 53 red cards shown in Brazil football match")
    assert not is_off_topic_general_news_title("Inside the chip talent war: Trade secrets and rising Chinese rivals")
    c_dir = tmp_path / "cachedb_news_rank"
    c_dir.mkdir(parents=True, exist_ok=True)
    now_s = 1_793_852_000.0
    (c_dir / "news_feed_collect.json").write_text(
        json.dumps(
            {
                "items": [
                    {"title": "World record 53 red cards shown in Brazil football match", "source": "BBC", "first_seen_ts": now_s - 60},
                    {"title": "JD Vance Breaks With Trump on Iran War and Israel, Positions Himself Ahead of 2028", "source": "NYT", "first_seen_ts": now_s - 120},
                    {"title": "Vance walks Trump tightrope with scathing election critique blaming Iran war", "source": "The Hill", "first_seen_ts": now_s - 180},
                    {"title": "Americans are turning on the Iran war", "source": "Yahoo", "first_seen_ts": now_s - 240},
                    {"title": "Inside the chip talent war: Trade secrets and rising Chinese rivals", "source": "CNBC", "first_seen_ts": now_s - 300},
                    {"title": "Stock futures are little changed after big rally as traders monitor oil prices", "source": "CNBC", "first_seen_ts": now_s - 360},
                    {"title": "Treasury yields hold steady ahead of consumer inflation report", "source": "Reuters", "first_seen_ts": now_s - 420},
                    {"title": "Central banks weigh interest rate cuts amid slowing global trade", "source": "FT", "first_seen_ts": now_s - 480},
                ]
            }
        ),
        encoding="utf-8",
    )
    _, top_news = harvest_macro_and_news_sensors(str(c_dir), now_s)
    top_titles = [n["title"] for n in top_news]
    assert len(top_titles) == 6
    assert "World record 53 red cards shown in Brazil football match" not in top_titles
    assert sum(1 for t in top_titles if "Vance" in t) == 1

    # 4. Meta/adjective noise bigrams are rejected by _is_noise_dynamic_keyword
    for bad_bg in ("end iran", "amid iran", "war fresh", "shock breaking", "alerts hacked", "ryanair freeze"):
        assert _is_noise_dynamic_keyword(bad_bg) is True


def test_asset_value_inner_timestamp_age_partisan_election_news_filter_and_pretrade_fill_ages(tmp_path):

    # 1. Bug 110: cache_asset_value.jsonl inner "i.timestamp" extracts total_value_usdc_age_min
    cdir = tmp_path / "cachedb"
    orders_dir = cdir / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    now_s = 1_793_855_000.0
    (orders_dir / "cache_asset_value.jsonl").write_text(
        json.dumps(
            {
                "s": "TOTAL",
                "i": {
                    "timestamp": int(now_s - 750),
                    "datetime_local": "2026-11-05T06:53:44",
                    "total_value_usdc": 11330.48,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ps: dict = {}
    PortfolioAndVenueCyclesCollector(start_sync=False).collect(
        str(cdir),
        "*",
        workspace_dir=str(tmp_path),
        logs_dirs=[],
        portfolio_summary=ps,
        prices_and_trends={},
        microstructure={},
        watcher_price_items={},
        inst_by_sym={},
        now_ts=now_s,
    )
    assert ps["total_value_usdc"] == 11330.48
    assert ps["total_value_usdc_age_min"] == 12.5

    # 2. Bug 111: Partisan election commentary mentioning generic war/prices without market/Fed/index context is deprioritized
    assert has_financial_or_geopolitical_context(
        "Video | Trump's Broken Peace Promises, Iran War And Rising Prices Cost Republicans In US Midterms"
    ) is False
    assert is_off_topic_general_news_title(
        "McDonald's tests new value menu as fast-food discount wars heat up"
    ) is True
    assert has_financial_or_geopolitical_context(
        "Stock futures are little changed after big rally following Democrats' decisive midterm wins"
    ) is True
    assert is_off_topic_general_news_title(
        "Stock futures are little changed after big rally following Democrats' decisive midterm wins"
    ) is False

    # 3. Bug 112: collect_pretrade_telemetry Recent Venue Fills includes Last BUY Age, Last SELL Age, and Recent BUY/SELL Tranches
    now_ms = int(time.time() * 1000)
    (orders_dir / "cache_order_binance.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "b1",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 77350.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 773.5,
                            "updateTime": now_ms - int(48 * 3_600_000),
                            "clientOrderId": "AIBTC1",
                        },
                        {
                            "orderId": "b2",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 77580.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 775.8,
                            "updateTime": now_ms - int(36 * 3_600_000),
                            "clientOrderId": "AIBTC2",
                        },
                        {
                            "orderId": "s1",
                            "side": "SELL",
                            "status": "FILLED",
                            "price": 79950.0,
                            "executedQty": 0.005,
                            "cummulativeQuoteQty": 399.75,
                            "updateTime": now_ms - int(14 * 3_600_000),
                            "clientOrderId": "AIBTC3",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pt = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir))
    rvf = pt.get("Recent Venue Fills", "")
    assert "Last BUY=0.01 @ 77580.0 (by=BOT)" in rvf
    assert "Last SELL=0.005 @ 79950.0 (by=BOT)" in rvf
    assert "Last BUY Age=36.0h" in rvf
    assert "Last SELL Age=14.0h" in rvf
    assert "Recent BUY Tranches=[0.01@77350.0, 0.01@77580.0]" in rvf

    (orders_dir / "cache_order_binance.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "b1",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 77350.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 773.5,
                            "updateTime": now_ms - int(48 * 3_600_000),
                            "clientOrderId": "AIBTC1",
                        },
                        {
                            "orderId": "b2",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 77580.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 775.8,
                            "updateTime": now_ms - int(36 * 3_600_000),
                            "clientOrderId": "AIBTC2",
                        },
                        {
                            "orderId": "s1",
                            "side": "SELL",
                            "status": "FILLED",
                            "price": 79950.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 799.5,
                            "updateTime": now_ms - int(14 * 3_600_000),
                            "clientOrderId": "AIBTC3",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pt_harvested = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir))
    rvf_harvested = pt_harvested.get("Recent Venue Fills", "")
    assert "Last BUY=0.01 @ 77580.0 (by=BOT, lot_harvested)" in rvf_harvested
    assert "Open LIFO Lots=[0.01@77350.0]" in rvf_harvested
    assert "Active Cycle Avg BUY=77465.0" in rvf_harvested

    (orders_dir / "cache_order_binance.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "b1",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 75000.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 750.0,
                            "updateTime": now_ms - int(48 * 3_600_000),
                            "clientOrderId": "AIBTC1",
                        },
                        {
                            "orderId": "s1",
                            "side": "SELL",
                            "status": "FILLED",
                            "price": 76000.0,
                            "executedQty": 0.005,
                            "cummulativeQuoteQty": 380.0,
                            "updateTime": now_ms - int(36 * 3_600_000),
                            "clientOrderId": "AIBTC2",
                        },
                        {
                            "orderId": "b2",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 78000.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 780.0,
                            "updateTime": now_ms - int(24 * 3_600_000),
                            "clientOrderId": "AIBTC3",
                        },
                        {
                            "orderId": "b3",
                            "side": "BUY",
                            "status": "FILLED",
                            "price": 77600.0,
                            "executedQty": 0.01,
                            "cummulativeQuoteQty": 776.0,
                            "updateTime": now_ms - int(12 * 3_600_000),
                            "clientOrderId": "AIBTC4",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pt_higher_sub = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir))
    rvf_higher_sub = pt_higher_sub.get("Recent Venue Fills", "")
    assert "Active Cycle Avg BUY=" in rvf_higher_sub
    assert "Tactical Sub-Cycle Avg BUY=" not in rvf_higher_sub
    assert "Swing Sub-Cycle Avg BUY=" not in rvf_higher_sub

    # 4. Bug 113: normalize_error_signature strips [file.py:line] [STDERR] tags and harvest_log_anomalies_from_dirs
    #    clears recovered transient -2015 IP/auth failover bursts once subsequent bot cycles succeed

    raw_ws = "WS error: login failed: {'id': 'login', 'status': 401, 'error': {'code': -2015, 'msg': 'Invalid API-key, IP, or permissions for action.'}}"
    wrapped_ws = f"06:46:47 [__init__.py:1163] [STDERR] {raw_ws}"
    assert normalize_error_signature(raw_ws) == normalize_error_signature(wrapped_ws)

    ldir_rec = tmp_path / "logs_rec"
    ldir_rec.mkdir(parents=True, exist_ok=True)
    clk_err = time.strftime("%H:%M:%S", time.localtime(time.time() - 900))
    clk_ok = time.strftime("%H:%M:%S", time.localtime(time.time() - 60))
    (ldir_rec / "assetguardian.log").write_text(
        "\n".join([
            f"{clk_err} get_free_balance: Error for TAO: APIError(code=-2015): Invalid API-key, IP, or permissions for action.",
            f"{clk_err} get_free_balance: Error for TAO: APIError(code=-2015): Invalid API-key, IP, or permissions for action.",
            f"{clk_ok} [TAOUSDC] Existing SELL order blocks AssetGuardian tier: ['721391298']",
            *(f"{clk_ok} [DEBUG] sleep 30.0s before next cycle {i}" for i in range(12)),
        ]),
        encoding="utf-8",
    )
    (ldir_rec / "monitortrades_2026-11-05.log").write_text(
        "\n".join([
            f"{clk_err} [__init__.py:1163] [STDERR] {raw_ws}",
            f"{clk_err} [__init__.py:1163] [STDERR] {raw_ws}",
            f"{clk_ok} get_trade_orders:           Total found 21 orders in the last 4.0 days.",
            *(f"{clk_ok} For BTCUSDC {i}" for i in range(12)),
        ]),
        encoding="utf-8",
    )
    (ldir_rec / "Binance-trailing.log").write_text(
        "\n".join([
            "get_account_assets_balances: Error reading balances: APIError(code=-2015): Invalid API-key, IP, or permissions for action.",
            f"[{clk_err}]   ! [TRAIL] empty or invalid balance snapshot — skipping the tick",
            f"[{clk_ok}] === TRAILING STOP started — ACTIVE (it really sells) ===",
            *(f"[{clk_ok}]   [TRAIL] BTCUSDC: 80585.6600  peak 86986.5700  sells below 71763.9202 (-17.5%) {i}" for i in range(12)),
        ]),
        encoding="utf-8",
    )
    anom_rec = harvest_log_anomalies_from_dirs([str(ldir_rec)], lookback_hours=1.0)
    assert anom_rec.error_count == 0
    assert anom_rec.persistent_errors == []
    assert anom_rec.critical_events == []


    assert parse_cited_taker_ratio(
        "A sharp 5-minute taker sell spike (5m ratio 0.3511, 5m CVD -$13.08M) keeps the aggregate taker ratio at 0.8326."
    ) == 0.8326
    assert parse_cited_taker_ratio(
        ["Paired with short-window taker sell pressure (aggregate ratio 0.6369, 5m CVD -$2.76M)"]
    ) == 0.6369

    # 5. Non-tradable cycle pos_peak reconciliation selects quote variant with positive high_24h
    hl_nontrad_cycles = {"HYPE": {"qty": 10.0, "cost_usd": 380.0, "avg_cost_px": 38.0, "pos_peak": 38.0}}
    enrich_positions_with_orders_and_cycles(
        [],
        {"HYPE": {"price": 40.0}, "HYPEUSDC": {"price": 40.0, "high_24h": 44.5}},
        {},
        {"hyperliquid": {"HYPE": {"last_buy": {"price": 38.0, "qty": 10.0, "ts_ms": now_ms - 24 * 3_600_000}}}},
        {},
        {},
        hl_nontrad_cycles,
        {},
        {},
    )
    assert hl_nontrad_cycles["HYPE"]["pos_peak"] == 44.5

    # 6. Bug 58: collect_pretrade_telemetry sibling p_est uses sibling's own cache_kalman_trend / cache_short_trend
    #    entry so TAOUSD and TAOUSDC produce identical liquidation magnet bias
    cdir_sib = tmp_path / "cachedb_sib_kal"
    cdir_sib.mkdir(parents=True, exist_ok=True)
    tao_pts_sib = [
        [now_ms - (288 - i) * 300_000, 226.60 if i == 50 else (239.80 if i == 150 else 233.50)]
        for i in range(288)
    ] + [[now_ms - 900_000, 233.50], [now_ms, 233.90]]
    (cdir_sib / "cache_24_price_TAOUSDC.json").write_text(json.dumps({"items": {"TAOUSDC": tao_pts_sib}}), encoding="utf-8")
    (cdir_sib / "cache_24_price_TAOUSD.json").write_text(json.dumps({"items": {"TAOUSD": tao_pts_sib}}), encoding="utf-8")
    (cdir_sib / "cache_kalman_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": {"kalman_vel": 0.02312, "kalman_vel_std": 0.04, "kalman_trend": 1, "vol_1h_pct": 0.7118},
                    "TAOUSD": {"kalman_vel": -0.03367, "kalman_vel_std": 0.04, "kalman_trend": -1, "vol_1h_pct": 0.7118},
                }
            }
        ),
        encoding="utf-8",
    )
    (cdir_sib / "orderbook_depth_TAOUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSDC",
                "mid_price": 233.90,
                "best_bid": 233.80,
                "best_ask": 234.00,
                "spread_bps": 8.55,
                "bid_depth_usd": 139282.0,
                "ask_depth_usd": 101516.0,
                "imbalance_ratio": 0.578,
                "largest_bid_wall_price": 233.50,
                "largest_bid_wall_usd": 20987.0,
                "largest_ask_wall_price": 234.30,
                "largest_ask_wall_usd": 14320.0,
                "ts": now_s,
            }
        ),
        encoding="utf-8",
    )
    (cdir_sib / "orderbook_depth_TAOUSD.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSD",
                "mid_price": 233.96,
                "best_bid": 233.88,
                "best_ask": 234.04,
                "spread_bps": 6.85,
                "bid_depth_usd": 727520.0,
                "ask_depth_usd": 284972.0,
                "imbalance_ratio": 0.719,
                "largest_bid_wall_price": 230.47,
                "largest_bid_wall_usd": 64969.0,
                "largest_ask_wall_price": 235.38,
                "largest_ask_wall_usd": 26378.0,
                "ts": now_s,
            }
        ),
        encoding="utf-8",
    )
    (cdir_sib / "derivatives_telemetry_TAOUSDC.json").write_text(
        json.dumps({"symbol": "TAOUSDC", "funding_rate": 0.00005, "open_interest_usd": 52918312.0, "ts": now_s}),
        encoding="utf-8",
    )
    (cdir_sib / "whale_snapshot_TAOUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSDC",
                "top_traders_long_pct": 0.7649,
                "top_traders_long_ratio": 3.25,
                "taker_buy_sell_ratio": 1.0694,
                "taker_buy_vol_usd": 214039.43,
                "taker_sell_vol_usd": 200152.11,
                "open_interest_usd": 52918312.0,
                "open_interest_1h_change_pct": -0.93,
                "divergence_regime": "long_unwinding",
                "ts": now_s,
            }
        ),
        encoding="utf-8",
    )
    pt_usdc_sib = collect_pretrade_telemetry("TAOUSDC", cachedb_dir=str(cdir_sib))
    pt_usd_sib = collect_pretrade_telemetry("TAOUSD", cachedb_dir=str(cdir_sib))
    assert "LONG liq=$" in pt_usdc_sib["Liquidation Clusters (LONG vs SHORT)"]
    assert "LONG liq=$" in pt_usd_sib["Liquidation Clusters (LONG vs SHORT)"]


def test_6h_bottom_dead_zone_sibling_low_guard_cross_market_bounds_and_military_news_filter():

    # 1. Bug 114: compute_24h_price_metrics detects HIGHER_LOW_DOUBLE_BOTTOM when recent_low is slightly above prior_low (+0.02%)
    # and cur_px has bounced +0.11%..+0.13% above recent_low (e.g. TAOUSDC prior_low=233.3, recent_low=233.35, cur_px=233.65).
    now_ms = 1_793_854_000_000
    series_tao = [
        [now_ms - 5 * 3_600_000, 238.0],
        [now_ms - 4 * 3_600_000, 236.0],
        [now_ms - 3 * 3_600_000, 233.3],
        [now_ms - 2 * 3_600_000, 235.0],
        [now_ms - 45 * 60_000, 234.0],
        [now_ms - 15 * 60_000, 233.35],
        [now_ms, 233.65],
    ]
    m_tao = compute_24h_price_metrics(series_tao)
    assert m_tao.get("bottom_structure_6h") == "HIGHER_LOW_DOUBLE_BOTTOM"
    assert m_tao.get("prior_6h_low_px") == 233.3
    assert m_tao.get("recent_1h_low_px") == 233.35
    assert m_tao.get("higher_low_vs_prior_pct") == 0.02

    # And when cur_px is sitting directly on the higher low (233.35 vs 233.30), it classifies as FRESH_6H_LOW_TEST
    series_tao_test = [*series_tao[:-1], [now_ms, 233.35]]
    m_tao_test = compute_24h_price_metrics(series_tao_test)
    assert m_tao_test.get("bottom_structure_6h") == "FRESH_6H_LOW_TEST"
    recalculate_live_price_geometry(m_tao_test, 233.65)
    assert m_tao_test.get("bottom_structure_6h") == "HIGHER_LOW_DOUBLE_BOTTOM"

    # 2. Bug 114b: sync_sibling_symbol_telemetry does not copy sibling's bottom_structure_6h when sibling's recent_1h_low_px > low_15m
    pt_map = {
        "TAOUSDC": {"price": 233.65, "chg_6h_pct": -0.49, "low_15m": 233.35, "high_15m": 234.2, "low_24h": 224.8, "high_24h": 239.8},
        "TAOUSD": {
            "price": 233.697,
            "chg_6h_pct": -0.40,
            "low_24h": 224.87,
            "high_24h": 240.11,
            "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
            "prior_6h_low_px": 233.4,
            "recent_1h_low_px": 233.47,
            "higher_low_vs_prior_pct": 0.03,
        },
    }
    sync_sibling_symbol_telemetry(pt_map, {})
    assert "bottom_structure_6h" not in pt_map["TAOUSDC"]

    # 3. Bug 114c: build_cross_market_context exposes low_24h, high_24h, bottom_structure_6h, prior_6h_low_px, recent_1h_low_px
    cmc = build_cross_market_context(
        {
            "TAOUSDC": {
                "price": 233.65,
                "pos_24h_pct": 59.0,
                "low_24h": 224.8,
                "high_24h": 239.8,
                "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
                "prior_6h_low_px": 233.3,
                "recent_1h_low_px": 233.35,
            },
            "NVDA_US_EQ": {
                "price": 198.01,
                "pos_24h_pct": 0.0,
                "low_24h": 198.01,
                "high_24h": 210.7,
                "bottom_structure_6h": "FRESH_6H_LOW_TEST",
                "prior_6h_low_px": 200.28,
                "recent_1h_low_px": 198.01,
            },
        },
        {},
        {},
    )
    assert cmc["crypto_bellwethers"]["TAO"]["low_24h"] == 224.8
    assert cmc["crypto_bellwethers"]["TAO"]["bottom_structure_6h"] == "HIGHER_LOW_DOUBLE_BOTTOM"
    assert cmc["equity_bellwethers"]["NVDA_US_EQ"]["low_24h"] == 198.01
    assert cmc["equity_bellwethers"]["NVDA_US_EQ"]["bottom_structure_6h"] == "FRESH_6H_LOW_TEST"

    # 4. Bug 116: has_financial_or_geopolitical_context filters non-theater local military headlines and partisan prediction-market stories
    assert not has_financial_or_geopolitical_context("El Salvador’s Bukele touts military intervention in Haiti")
    assert not has_financial_or_geopolitical_context(
        "Prediction markets cement their status after forecasting Democratic sweep of Congress"
    )
    assert has_financial_or_geopolitical_context("Pentagon deploys additional military assets to Middle East")
    assert has_financial_or_geopolitical_context("Futures Mixed After Postelection Rally; Three Giants Are Buys")
    assert has_financial_or_geopolitical_context(
        'Russia, China may be more aggressive if they feel Trump is on "his way out": Brookings'
    )


def test_advisor_1h_cvd_sign_flip_divergence_and_liq_magnet_multi_horizon_flow(tmp_path):

    macro_dir = tmp_path / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bug 117: parse_cited_cvd_usd & enrich_sentiment_advisor_live_flow_divergence detect 1h CVD sign flips
    adv_summary = (
        "BTC trades at $80,416.12 as strong short-term taker buying (aggregate ratio 1.6606, 5m CVD +$5.69M) "
        "meets stabilizing OI, while 1h CVD remains slightly negative (-$2.48M)."
    )
    assert parse_cited_cvd_usd(adv_summary, horizon="1h") == -2_480_000.0
    assert parse_cited_cvd_usd(adv_summary, horizon="5m") == 5_690_000.0

    sa_pos_flip = {
        "market_bias": "BULLISH",
        "recommended_action": "HOLD",
        "age_min": 32.2,
        "summary": adv_summary,
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_pos_flip,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 1.4956,
                "cvd_15m_usd": 10_660_000.0,
                "taker_ratio_5m": 1.08,
                "cvd_5m_usd": 380_200.0,
                "taker_ratio_1h": 1.16,
                "cvd_1h_usd": 14_710_000.0,
            }
        },
    )
    assert "LIVE_1H_CVD_FLIP_POSITIVE" in sa_pos_flip.get("live_flow_divergence", "")
    assert "advisor 1h_cvd=-2.48M" in sa_pos_flip.get("live_flow_divergence", "")

    sa_neg_flip = {
        "market_bias": "BEARISH",
        "recommended_action": "HOLD",
        "age_min": 25.0,
        "summary": "Short-term selling (aggregate ratio 0.62) meets positive 1h CVD (+$6.40M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_neg_flip,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.68,
                "cvd_15m_usd": -4_200_000.0,
                "taker_ratio_1h": 0.82,
                "cvd_1h_usd": -9_100_000.0,
            }
        },
    )
    assert "LIVE_1H_CVD_FLIP_NEGATIVE" in sa_neg_flip.get("live_flow_divergence", "")
    assert "advisor 1h_cvd=+6.40M" in sa_neg_flip.get("live_flow_divergence", "")

    # 2. Bug 118: compute_liquidation_clusters neutralizes LONG_SWEEP_MAGNET_DOWN when ask_absorbing + strong taker/5m/OB/spot flow
    tao_liq = compute_liquidation_clusters(
        {
            "price": 232.95,
            "vol_1h_pct": 0.565,
            "kalman_vel": -0.01422,
            "kalman_vel_std": 0.045,
            "chg_15m_pct": -0.32,
            "val_px": 229.66,
            "vah_px": 238.02,
            "low_24h": 226.65,
            "high_24h": 239.8,
        },
        {
            "oi_usd": 52_694_654.38,
            "whale_long_pct": 0.7652,
            "funding_rate": 0.00005,
            "ob_imbalance": 0.602,
            "bid_wall_px": 232.5,
            "ask_wall_px": 233.5,
            "taker_buy_sell_ratio": 1.4194,
            "cvd_15m_usd": 78_947.0,
            "cvd_regime": "ask_absorbing",
            "spot_cvd_15m_usd": 39_727.0,
            "taker_ratio_5m": 2.22,
            "cvd_5m_usd": 77_397.0,
            "taker_ratio_1h": 0.88,
            "cvd_1h_usd": -131_091.0,
        },
    )
    assert tao_liq["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    # 3. Parenthesized number-first horizon taker buying list ("taker buying (1.5385 overall, 2.0885 5m, 1.1871 1h)")

    sa_summary_iter19 = (
        "European morning accumulation is actively underway as BTC advances to $80,487.51 (81% range position) "
        "backed by expanding 24h market breadth (68.8% advancing) and aggressive multi-timeframe taker buying "
        "(1.5385 overall, 2.0885 5m, 1.1871 1h)."
    )
    assert parse_cited_taker_ratio(sa_summary_iter19) == 1.5385
    assert parse_cited_taker_ratio("taker ratio (0.566 5m, 1.0944 1h)") == 1.0944

    sa_fade_iter19 = {
        "age_min": 18.0,
        "directional_bias": "BULLISH",
        "recommended_action": "ACCUMULATE",
        "summary": sa_summary_iter19,
        "key_risks": [],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_fade_iter19,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.7541,
                "cvd_15m_usd": -4_189_363.0,
                "taker_ratio_5m": 0.80,
                "cvd_5m_usd": -617_830.0,
                "taker_ratio_1h": 1.07,
                "cvd_1h_usd": 6_190_368.0,
            }
        },
    )
    assert "LIVE_FLOW_FADE" in sa_fade_iter19.get("live_flow_divergence", "")
    assert "advisor ratio=1.54" in sa_fade_iter19.get("live_flow_divergence", "")

    # 4. Multi-horizon positive flow alignment neutralizes LONG_SWEEP_MAGNET_DOWN on HYPEUSDC
    hype_liq_iter19 = compute_liquidation_clusters(
        {
            "price": 83.2845,
            "vol_1h_pct": 0.5017,
            "kalman_vel": -0.00384,
            "kalman_vel_std": 0.04,
            "chg_15m_pct": -0.01,
            "slope_3_7m": 0.00033,
            "val_px": 84.033,
            "vah_px": 86.987,
            "low_24h": 82.7275,
            "high_24h": 88.7365,
        },
        {
            "oi_usd": 401_073_945.41,
            "whale_long_pct": 0.7033,
            "funding_rate": 0.0000125,
            "ob_imbalance": 0.52,
            "bid_wall_px": 82.9,
            "ask_wall_px": 83.6,
            "taker_buy_sell_ratio": 1.0426,
            "cvd_15m_usd": 32_601.0,
            "cvd_regime": "ask_absorbing",
            "spot_cvd_15m_usd": 155_135.0,
            "taker_ratio_5m": 1.51,
            "cvd_5m_usd": 86_503.0,
            "taker_ratio_1h": 1.03,
            "cvd_1h_usd": 146_996.0,
        },
    )
    assert hype_liq_iter19["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    # 5. Spot-absorbing accumulation rally (ARBUSDC) and strong 15m taker + spot buying (HYPEUSDC)
    #    neutralize LONG_SWEEP_MAGNET_DOWN, and harvest_macro_and_news_sensors strips stale US cash-session
    #    1h/15m deltas outside RTH when last_bar_ts is stale or legacy-absent.
    arb_liq_iter20 = compute_liquidation_clusters(
        {
            "price": 0.162888,
            "vol_1h_pct": 0.62,
            "kalman_vel": 0.03051,
            "kalman_vel_std": 0.04,
            "chg_15m_pct": 0.31,
            "slope_15m": 5.2e-7,
            "val_px": 0.1597,
            "vah_px": 0.1684,
            "low_24h": 0.1585,
            "high_24h": 0.1695,
        },
        {
            "oi_usd": 47_600_000.0,
            "whale_long_pct": 0.7524,
            "funding_rate": -0.000018,
            "ob_imbalance": 0.627,
            "div_regime": "accumulation",
            "taker_buy_sell_ratio": 0.4869,
            "cvd_15m_usd": -306_600.0,
            "cvd_regime": "spot_absorbing",
            "spot_cvd_15m_usd": 23_000.0,
        },
    )
    assert arb_liq_iter20["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    hype_liq_iter20 = compute_liquidation_clusters(
        {
            "price": 83.37,
            "vol_1h_pct": 0.49,
            "kalman_vel": 0.02391,
            "kalman_vel_std": 0.04,
            "chg_15m_pct": 0.01,
            "val_px": 84.03,
            "vah_px": 86.99,
            "low_24h": 82.73,
            "high_24h": 88.74,
        },
        {
            "oi_usd": 401_580_000.0,
            "whale_long_pct": 0.7033,
            "funding_rate": 0.0000125,
            "ob_imbalance": 0.517,
            "taker_buy_sell_ratio": 1.2103,
            "cvd_15m_usd": 201_400.0,
            "cvd_regime": "ask_absorbing",
            "spot_cvd_15m_usd": 114_400.0,
            "taker_ratio_5m": 0.81,
            "cvd_5m_usd": -19_100.0,
            "taker_ratio_1h": 0.93,
            "cvd_1h_usd": -377_400.0,
        },
    )
    assert hype_liq_iter20["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    (macro_dir / "macro_benchmarks_collect.json").write_text(
        json.dumps(
            {
                "ts": 1_793_859_927.0,  # 2026-11-05 01:25 EST (outside US cash session)
                "items": {
                    "SPY": {
                        "symbol": "SPY",
                        "unit": "usd",
                        "price": 765.31,
                        "chg_24h_pct": 1.45,
                        "chg_1h_pct": -0.27,
                        "chg_15m_pct": -0.27,
                    },
                    "ES=F": {
                        "symbol": "ES=F",
                        "unit": "usd",
                        "price": 7705.25,
                        "chg_24h_pct": -0.20,
                        "chg_1h_pct": 0.05,
                        "chg_15m_pct": 0.06,
                    },
                    "US10Y": {
                        "symbol": "US10Y",
                        "unit": "yield_pct",
                        "price": 4.894,
                        "chg_24h_pct": -50.3,
                        "low_24h": 4.884,
                        "high_24h": 4.958,
                    },
                },
                "summary_line": "SPY=$765.31 (+1.45% 24h, -0.27% 1h, -0.27% 15m) | ES=F=$7,705.25 (-0.20% 24h, +0.05% 1h, +0.06% 15m) | US10Y=4.89% (-50.30% 24h)",
            }
        ),
        encoding="utf-8",
    )
    sensors_iter20, _ = harvest_macro_and_news_sensors(str(tmp_path), now_ts=1_793_861_400.0)
    bm_iter20 = sensors_iter20["macro_benchmarks"]
    assert bm_iter20 is not None
    assert "chg_1h_pct" not in bm_iter20["items"]["SPY"]
    assert "chg_15m_pct" not in bm_iter20["items"]["SPY"]
    assert bm_iter20["items"]["ES=F"]["chg_1h_pct"] == 0.05
    assert bm_iter20["items"]["US10Y"]["chg_24h_pct"] == -1.29
    assert "SPY=$765.31 (+1.45% 24h)" in bm_iter20["summary_line"]
    assert "ES=F=$7,705.25 (-0.20% 24h, +0.05% 1h, +0.06% 15m)" in bm_iter20["summary_line"]
    assert "US10Y=4.89% (-1.29% 24h)" in bm_iter20["summary_line"]

    # 5. Slash-separated horizon taker ratio list, aggressive_buying on near-flat 15m with bid dominance, and Nintendo Switch price filter

    sa_slash_sum = (
        "Constructive 24h market breadth (67.5% advancing, median +2.64%) and sustained 5m/1h/4h taker buy dominance "
        "(1.1211 / 1.2618 / 1.0985) confirm institutional bid absorption despite minor top-of-range supply."
    )
    assert parse_cited_taker_ratio(sa_slash_sum) == 1.2618
    _, slash_reason = compute_dynamic_sentiment_interval(
        base_interval_sec=14400.0,
        last_eval_ts=1_793_862_562.0,
        now_ts=1_793_862_730.0,
        cached_assessment=SimpleNamespace(
            market_bias="BULLISH",
            risk_level="MODERATE",
            recommended_action="ACCUMULATE",
            summary=sa_slash_sum,
            key_risks=[],
            is_fallback=False,
            ts=1_793_862_562.0,
        ),
        btc_taker_ratio=1.37,
        prev_btc_taker_ratio=0.97,
    )
    assert "btc_whale_flow_regime_flip" not in slash_reason

    btc_m_iter20 = {
        "cvd_15m_usd": 12_564_049.0,
        "spot_cvd_15m_usd": 1_587_017.0,
        "ob_imbalance": 0.705,
        "taker_buy_sell_ratio": 1.4832,
        "cvd_1h_usd": 12_830_000.0,
    }
    enrich_microstructure_cvd(btc_m_iter20, {"price": 80604.14, "chg_15m_pct": -0.01})
    assert btc_m_iter20["cvd_regime"] == "aggressive_buying"
    assert btc_m_iter20["spot_perp_flow_regime"] == "SYNCHRONIZED_SPOT_PERP_BUYING"

    ada_m_iter20 = {
        "cvd_15m_usd": 165_000.0,
        "spot_cvd_15m_usd": 30_600.0,
        "ob_imbalance": 0.605,
        "taker_buy_sell_ratio": 1.9652,
    }
    enrich_microstructure_cvd(ada_m_iter20, {"price": 0.1978, "chg_15m_pct": 0.02})
    assert ada_m_iter20["cvd_regime"] == "aggressive_buying"
    assert ada_m_iter20["spot_perp_flow_regime"] == "SYNCHRONIZED_SPOT_PERP_BUYING"

    tao_m_flat15 = {
        "cvd_15m_usd": 30_380.0,
        "spot_cvd_15m_usd": 18_051.0,
        "ob_imbalance": 0.614,
        "taker_buy_sell_ratio": 1.2287,
        "cvd_1h_usd": 17_900.0,
    }
    enrich_microstructure_cvd(tao_m_flat15, {"price": 235.4, "chg_15m_pct": 0.0})
    assert tao_m_flat15["cvd_regime"] == "aggressive_buying"
    assert tao_m_flat15["spot_perp_flow_regime"] == "SYNCHRONIZED_SPOT_PERP_BUYING"

    sa_6m_bull = {
        "directional_bias": "BULLISH",
        "recommended_action": "ACCUMULATE",
        "age_min": 6.7,
        "summary": "BTC advancing as aggressive taker buy flow returns (taker buy/sell ratio 1.5621, 5m CVD +$0.82M).",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_6m_bull,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.6002,
                "cvd_15m_usd": -29_800_000.0,
                "taker_ratio_5m": 0.43,
                "cvd_5m_usd": -35_460_000.0,
                "taker_ratio_1h": 0.78,
                "cvd_1h_usd": -33_060_000.0,
            }
        },
    )
    assert sa_6m_bull.get("live_flow_divergence", "").startswith("LIVE_SELL_REVERSAL (")

    assert not has_financial_or_geopolitical_context("Nintendo Switch 2 sales plunge 41% as price hikes and memory crunch bite")
    assert has_financial_or_geopolitical_context("Oil prices slide as Asian stocks rise")

    # 6. Clickbait survival-strategy opinion column exclusion and recovered account_cache_not_fresh startup log filter

    assert is_excluded_reference_news_item(
        "Most investors are blind to the possibility of a market crash. Here is the five-part strategy to survive one."
    )
    rtrade_log_dir = tmp_path / "logs_iter21"
    rtrade_log_dir.mkdir(parents=True, exist_ok=True)
    (rtrade_log_dir / "rtrade.log").write_text(
        "\n".join(
            [
                "09:21:27 [BINANCE][GUARD] account cache is not fresh: account_cache_initializing",
                "09:21:27 [TAOUSDC] BUY BLOCKED (pre-submit): account_cache_not_fresh",
                "09:22:01 apply_weight_limit -> BUY TAOUSDC, Available qty 27.82, Final qty: 2.12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    anom_iter21 = harvest_log_anomalies_from_dirs([str(rtrade_log_dir)], lookback_hours=24.0)
    assert anom_iter21.error_count == 0
    assert not anom_iter21.persistent_errors
    assert not anom_iter21.guard_interventions

    # 7. Multi-horizon parenthesized CVD parsing, horizon-specific taker ratio extraction, and 5m flow fade/recovery divergence
    sa_multi_sum = (
        "European morning accumulation is actively underway as BTC advances to $80,487.51 (81% range position) "
        "backed by expanding 24h market breadth (68.8% advancing) and aggressive multi-timeframe taker buying "
        "(1.5385 overall, 2.0885 5m, 1.1871 1h). Stabilizing open interest (+0.07% 1h to $7.71B) alongside "
        "strong positive CVD inflows (+$17.56M 1h / +$34.59M 4h) confirms constructive institutional bid absorption."
    )
    assert parse_cited_taker_ratio(sa_multi_sum) == 1.5385
    assert parse_cited_taker_ratio(sa_multi_sum, horizon="5m") == 2.0885
    assert parse_cited_taker_ratio(sa_multi_sum, horizon="1h") == 1.1871
    assert parse_cited_taker_ratio("5m/15m taker buying (2.09, 1.18)") == 1.18
    assert parse_cited_taker_ratio("5m/15m taker buying (2.09, 1.18)", horizon="5m") == 2.09
    assert parse_cited_cvd_usd(sa_multi_sum, horizon="1h") == 17_560_000.0
    assert parse_cited_cvd_usd(sa_multi_sum, horizon="4h") == 34_590_000.0
    sa_slash_cvd_sum = (
        "Broader structural support remains intact with positive 1h open interest growth (+0.18% to $7.72B) "
        "and sustained 1h/4h CVD inflows (+$11.72M / +$36.57M), favoring a disciplined HOLD posture."
    )
    assert parse_cited_cvd_usd(sa_slash_cvd_sum, horizon="1h") == 11_720_000.0
    assert parse_cited_cvd_usd(sa_slash_cvd_sum, horizon="4h") == 36_570_000.0
    assert parse_cited_cvd_usd(sa_slash_cvd_sum, horizon="5m") is None

    ada_abs_m = {
        "cvd_15m_usd": -111_400.0,
        "spot_cvd_15m_usd": 25_800.0,
        "ob_imbalance": 0.582,
        "taker_buy_sell_ratio": 0.7353,
        "cvd_1h_usd": 78_800.0,
    }
    enrich_microstructure_cvd(ada_abs_m, {"price": 0.197429, "chg_15m_pct": -0.12})
    assert ada_abs_m["cvd_regime"] == "spot_absorbing"

    arb_liq_5m = compute_liquidation_clusters(
        {
            "price": 0.16335,
            "vol_1h_pct": 0.8516,
            "kalman_vel": -0.03124,
            "chg_15m_pct": 0.21,
            "val_px": 0.162954,
            "vah_px": 0.169546,
            "low_24h": 0.1595,
            "high_24h": 0.1723,
        },
        {
            "oi_usd": 34_247_744.15,
            "whale_long_pct": 0.6195,
            "funding_rate": 0.00008435,
            "ob_imbalance": 0.565,
            "bid_wall_px": 0.1623,
            "ask_wall_px": 0.1647,
            "taker_buy_sell_ratio": 0.367,
            "cvd_15m_usd": -364_600.0,
            "cvd_regime": "spot_absorbing",
            "spot_cvd_15m_usd": 12_300.0,
            "taker_ratio_5m": 2.68,
            "cvd_5m_usd": 18_400.0,
            "taker_ratio_1h": 0.59,
            "cvd_1h_usd": -401_200.0,
        },
    )
    assert arb_liq_5m["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    sa_5m_fade = {
        "market_bias": "BULLISH",
        "recommended_action": "ACCUMULATE",
        "age_min": 62.7,
        "summary": sa_multi_sum,
        "key_risks": [],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_5m_fade,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 1.5231,
                "cvd_15m_usd": 13_270_000.0,
                "taker_ratio_5m": 0.8317,
                "cvd_5m_usd": -1_640_000.0,
                "taker_ratio_1h": 1.1045,
                "cvd_1h_usd": 8_780_000.0,
            }
        },
    )
    assert "LIVE_5M_FLOW_FADE" in sa_5m_fade.get("live_flow_divergence", "")
    assert "5m_ratio=2.09" in sa_5m_fade.get("live_flow_divergence", "")

    sa_5m_rec = {
        "market_bias": "BEARISH",
        "recommended_action": "REDUCE",
        "age_min": 35.0,
        "summary": "Heavy distribution continues with taker selling (0.82 overall, 0.54 5m, 0.78 1h) and CVD (-$8.50M 1h).",
        "key_risks": [],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_5m_rec,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.88,
                "cvd_15m_usd": -1_200_000.0,
                "taker_ratio_5m": 1.42,
                "cvd_5m_usd": 1_850_000.0,
                "taker_ratio_1h": 0.85,
                "cvd_1h_usd": -4_100_000.0,
            }
        },
    )
    assert "LIVE_5M_FLOW_RECOVERY" in sa_5m_rec.get("live_flow_divergence", "")
    assert "5m_ratio=0.54" in sa_5m_rec.get("live_flow_divergence", "")

    # Bug 70: TAOUSDC near-flat chg_15m (-0.06% / +0.02%) with strong ob_imbalance (0.631),
    # positive chg_5m (+0.15%), spot_cvd_15m > 0, and kalman_vel > 0 should classify as
    # spot_absorbing and neutralize LONG_SWEEP_MAGNET_DOWN to BALANCED_LIQ_MAGNET.
    tao_minfo = {
        "taker_buy_sell_ratio": 0.8572,
        "cvd_15m_usd": -27_871.0,
        "cvd_1h_usd": -489_200.0,
        "ob_imbalance": 0.632,
        "divergence_regime": "neutral",
        "bid_depth_usd": 389_000.0,
        "ask_depth_usd": 227_000.0,
    }
    tao_pinfo = {"price": 234.7, "chg_15m_pct": -0.06, "chg_5m_pct": 0.15}
    enrich_microstructure_cvd(tao_minfo, tao_pinfo)
    assert tao_minfo["cvd_regime"] == "spot_absorbing"

    tao_liq = compute_liquidation_clusters(
        {
            "price": 234.77,
            "chg_15m_pct": 0.02,
            "chg_5m_pct": 0.19,
            "kalman_vel": 0.07482,
            "val_px": 234.09,
            "vah_px": 244.71,
            "low_24h": 230.89,
            "high_24h": 247.34,
        },
        {
            "oi_usd": 17_847_005.1,
            "whale_long_pct": 0.7674,
            "funding_rate": -0.00008365,
            "ob_imbalance": 0.631,
            "bid_wall_px": 234.34,
            "ask_wall_px": 235.16,
            "taker_buy_sell_ratio": 0.8572,
            "cvd_15m_usd": -27_900.0,
            "cvd_regime": "spot_absorbing",
            "spot_cvd_15m_usd": 7_500.0,
            "taker_ratio_5m": 0.98,
            "cvd_5m_usd": -1_600.0,
            "taker_ratio_1h": 0.65,
            "cvd_1h_usd": -489_200.0,
        },
    )
    assert tao_liq["liq_magnet_bias"] == "BALANCED_LIQ_MAGNET"

    # Wave 12:
    # 1. LIVE_1H_CVD_FLIP_NEGATIVE triggers when cited 1h CVD (+11.72M) flips to -10.79M with live_tr=0.9765 (<=1.0) and live_tr_1h=0.93 (<=0.95)
    sa_1h_cvd_flip = {
        "market_bias": "BULLISH",
        "recommended_action": "HOLD",
        "age_min": 24.3,
        "summary": (
            "BTC is digesting recent European morning gains at $80,552.01 (82% range position) as short-term "
            "taker sell pressure (0.6894 overall, 0.4109 5m, -$5.42M 5m CVD) cools the divergence regime back to "
            "'neutral'. Broader structural support remains intact with constructive 24h market breadth (63.2% advancing), "
            "positive 1h open interest growth (+0.18% to $7.72B), and sustained 1h/4h CVD inflows (+$11.72M / +$36.57M), "
            "favoring a disciplined HOLD posture."
        ),
        "key_risks": [
            "Acute 5m taker selling (0.4109 ratio, -$5.42M CVD) alongside moderating 24h breadth could push price down."
        ],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_1h_cvd_flip,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.9765,
                "cvd_15m_usd": -1_306_139.0,
                "taker_ratio_5m": 1.01,
                "cvd_5m_usd": 209_046.0,
                "taker_ratio_1h": 0.93,
                "cvd_1h_usd": -10_794_598.0,
            }
        },
    )
    assert "LIVE_1H_CVD_FLIP_NEGATIVE" in sa_1h_cvd_flip.get("live_flow_divergence", "")
    assert "1h_cvd=+11.72M" in sa_1h_cvd_flip.get("live_flow_divergence", "")

    # 2. compute_dynamic_sentiment_interval triggers 300s refresh when cited_tr <= 0.70 (0.6894) recovers to cur_tr >= 0.95 (0.9765)
    iv_w12, rsn_w12 = compute_dynamic_sentiment_interval(
        base_interval_sec=14400.0,
        last_eval_ts=1_793_863_600.0,
        now_ts=1_793_865_060.0,
        cached_assessment=SimpleNamespace(
            market_bias="BULLISH",
            risk_level="MODERATE",
            recommended_action="HOLD",
            summary=sa_1h_cvd_flip["summary"],
            key_risks=sa_1h_cvd_flip["key_risks"],
            is_fallback=False,
            ts=1_793_863_600.0,
        ),
        btc_taker_ratio=0.9765,
        prev_btc_taker_ratio=0.94,
    )
    assert iv_w12 <= 300.0
    assert "btc_whale_flow_regime_flip_0.69_to_0.98" in rsn_w12

    # 3. reconcile_sibling_flow_and_liq_regimes & update_microstructure_cycle_deltas unify shared perp OI and liq USD across USD siblings

    m_tao = {
        "TAOUSDC": {
            "ob_mid_price": 235.55,
            "ob_imbalance": 0.674,
            "bid_depth_usd": 199619.0,
            "ask_depth_usd": 96506.0,
            "oi_usd": 53007961.68,
            "oi_1h_chg_pct": -0.13,
            "cvd_15m_usd": 119951.0,
            "long_liq_usd": 485000.0,
            "short_liq_usd": 255600.0,
            "long_liq_deep_usd": 831500.0,
            "short_liq_deep_usd": 438200.0,
            "liq_magnet_bias": "SHORT_SQUEEZE_MAGNET_UP",
        },
        "TAOUSD": {
            "ob_mid_price": 235.63,
            "ob_imbalance": 0.581,
            "bid_depth_usd": 672893.0,
            "ask_depth_usd": 486217.0,
            "oi_usd": 53057438.52,
            "oi_1h_chg_pct": -0.13,
            "cvd_15m_usd": 119951.0,
            "long_liq_usd": 485500.0,
            "short_liq_usd": 255900.0,
            "long_liq_deep_usd": 832200.0,
            "short_liq_deep_usd": 438700.0,
            "liq_magnet_bias": "SHORT_SQUEEZE_MAGNET_UP",
        },
    }
    reconcile_sibling_flow_and_liq_regimes(m_tao, {"TAOUSDC": {"price": 235.55}, "TAOUSD": {"price": 235.63}})
    assert m_tao["TAOUSD"]["oi_usd"] == 53007961.68
    assert m_tao["TAOUSD"]["long_liq_usd"] == 485000.0
    assert m_tao["TAOUSD"]["short_liq_usd"] == 255600.0
    cdir_tao = tmp_path / "cachedb_tao_cd"
    cdir_tao.mkdir(parents=True, exist_ok=True)
    (cdir_tao / "microstructure_cycle_history.json").write_text(
        json.dumps(
            {
                "TAOUSDC": [{"ts": 1_793_866_320.0, "ob_imbalance": 0.637, "cvd_15m_usd": 254362.0, "oi_usd": 52878324.1}],
                "TAOUSD": [{"ts": 1_793_866_320.0, "ob_imbalance": 0.569, "cvd_15m_usd": 254362.0, "oi_usd": 52877862.19}],
            }
        ),
        encoding="utf-8",
    )
    update_microstructure_cycle_deltas(str(cdir_tao), m_tao, now_ts=1_793_866_750.0)
    assert m_tao["TAOUSD"]["cycle_delta"]["oi_delta_pct"] == m_tao["TAOUSDC"]["cycle_delta"]["oi_delta_pct"]
    assert m_tao["TAOUSD"]["cycle_delta"]["ob_imbalance_delta"] != m_tao["TAOUSDC"]["cycle_delta"]["ob_imbalance_delta"]

    # 4. Cross-venue ob_imbalance in reconcile_sibling_flow_and_liq_regimes preserves INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE
    m_tao_abs = {
        "TAOUSDC": {
            "ob_mid_price": 235.15,
            "ob_imbalance": 0.576,  # just below 0.58 single-venue threshold
            "bid_depth_usd": 179129.0,
            "ask_depth_usd": 131701.0,
            "taker_buy_vol_usd": 798258.0,
            "taker_sell_vol_usd": 929869.0,
            "cvd_15m_usd": -131611.0,
            "cvd_5m_usd": -84534.0,
            "cvd_1h_usd": -117761.0,
            "funding_rate": -0.000018,
            "cvd_regime": "spot_absorbing",
            "_synthetic_spot_cvd": True,
        },
        "TAOUSD": {
            "ob_mid_price": 235.16,
            "ob_imbalance": 0.675,  # strong bid depth on sibling venue -> cross-venue ob_imb = 0.6547 >= 0.58
            "bid_depth_usd": 653789.0,
            "ask_depth_usd": 307617.0,
            "taker_buy_vol_usd": 798258.0,
            "taker_sell_vol_usd": 929869.0,
            "cvd_15m_usd": -131611.0,
            "cvd_5m_usd": -84534.0,
            "cvd_1h_usd": -117761.0,
            "funding_rate": -0.000018,
            "cvd_regime": "spot_absorbing",
            "_synthetic_spot_cvd": True,
        },
    }
    pt_tao_abs = {
        "TAOUSDC": {"price": 235.15, "chg_15m_pct": 0.08, "chg_5m_pct": 0.04},
        "TAOUSD": {"price": 235.16, "chg_15m_pct": 0.08, "chg_5m_pct": 0.04},
    }
    reconcile_sibling_flow_and_liq_regimes(m_tao_abs, pt_tao_abs)
    assert m_tao_abs["TAOUSDC"]["spot_perp_flow_regime"] == "INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE"
    assert m_tao_abs["TAOUSD"]["spot_perp_flow_regime"] == "INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE"

    # 5. enrich_positions_with_orders_and_cycles recovers multi-day cycle peak from cache_365_price_<sym>.jsonl when >24h old

    cdir_nvda = tmp_path / "cachedb_nvda_pk"
    (cdir_nvda / "prices").mkdir(parents=True, exist_ok=True)
    now_ms_pk = int(time.time() * 1000)
    entry_ts_ms = now_ms_pk - int(43.6 * 3600 * 1000)
    peak_25h_ago_ms = now_ms_pk - int(24.6 * 3600 * 1000)
    (cdir_nvda / "prices" / "cache_365_price_NVDA_US_EQ.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"i": [entry_ts_ms, 207.93]}),
                json.dumps({"i": [peak_25h_ago_ms, 210.70]}),
                json.dumps({"i": [now_ms_pk - 3600_000, 209.52]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pos_nvda = [
        {
            "venue": "t212",
            "symbol": "NVDA_US_EQ",
            "free_base": 2.910252,
            "price": 206.69,
            "trailing_peak": 207.93,
            "active_avg_buy_px": 207.93,
        }
    ]
    t212_cyc_nvda = {
        "NVDA_US_EQ": {
            "qty": 2.910252,
            "entry_price": 207.93,
            "avg_cost_px": 207.93,
            "pos_peak": 207.93,
        }
    }
    v_ro_nvda = {
        "t212": {
            "NVDA_US_EQ": {
                "last_action_side": "BUY",
                "active_avg_buy_px": 207.93,
                "open_lots": [{"lot_id": 28, "qty": 0.03, "entry_px": 207.93, "ts_ms": entry_ts_ms}],
                "last_buy": {"price": 207.93, "qty": 0.03, "ts_ms": entry_ts_ms},
            }
        }
    }
    pt_nvda = {"NVDA_US_EQ": {"price": 206.69, "high_24h": 209.52}}
    enrich_positions_with_orders_and_cycles(
        pos_nvda,
        pt_nvda,
        {},
        v_ro_nvda,
        v_ro_nvda["t212"],
        {},
        {},
        t212_cyc_nvda,
        {},
        cache_dir=str(cdir_nvda),
    )
    assert pos_nvda[0]["trailing_peak"] == 210.7
    assert t212_cyc_nvda["NVDA_US_EQ"]["pos_peak"] == 210.7

    # apply_asset_scope_filter("equities") preserves equity microstructure while removing crypto microstructure

    eq_ps_summary = {
        "total_value_usdc": 2000.0,
        "total_value_usdc_age_min": 1.2,
        "venues_value_usd": {"binance": 2000.0, "kraken": 950.0, "t212": 100.0},
        "portfolio_total_usd": 3050.0,
    }
    _, eq_micro_filtered = apply_asset_scope_filter(
        {"NVDA_US_EQ": {"price": 206.69}, "BTCUSDC": {"price": 85000.0}},
        {"NVDA_US_EQ": {"ob_imbalance": 0.58, "whale_long_pct": 0.62}, "BTCUSDC": {"ob_imbalance": 0.45}},
        eq_ps_summary,
        "equities",
        {},
        {"NVDA_US_EQ": {}},
    )
    assert "NVDA_US_EQ" in eq_micro_filtered and "BTCUSDC" not in eq_micro_filtered
    assert "total_value_usdc" not in eq_ps_summary and "total_value_usdc_age_min" not in eq_ps_summary
    assert eq_ps_summary.get("venues_value_usd") == {"t212": 100.0}
    assert eq_ps_summary.get("portfolio_total_usd") == 100.0
    cm_ctx = build_cross_market_context(
        {"NVDA_US_EQ": {"price": 206.69, "chg_24h_pct": 1.2}},
        {"NVDA_US_EQ": {"ob_imbalance": 0.58, "whale_long_pct": 0.62, "cvd_15m_usd": 125000.0}},
        {"NVDA_US_EQ": {"qty": 2.91, "avg_cost_px": 207.93}},
    )
    assert cm_ctx["equity_bellwethers"]["NVDA_US_EQ"]["ob_imbalance"] == 0.58
    assert cm_ctx["equity_bellwethers"]["NVDA_US_EQ"]["whale_long_pct"] == 0.62

    # 6. harvest_log_anomalies_from_dirs clears pre-restart SyntaxError and NameError once daemon/cycle recovery lines appear
    log_dir_rec = tmp_path / "logs_rec"
    log_dir_rec.mkdir(parents=True, exist_ok=True)
    now_s_rec = time.time()
    fmt_rec = lambda ago: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s_rec - ago))  # noqa: E731
    (log_dir_rec / "tradeall.log").write_text(
        f"{fmt_rec(180)} [TrendCoordinator] Error while evaluating BTCUSDC: name 'sys' is not defined\n"
        f"{fmt_rec(60)} [TRADEALL] BTCUSDC resolved trend: flat\n",
        encoding="utf-8",
    )
    (log_dir_rec / "globaltelemetry_collector.log").write_text(
        "SyntaxError: invalid decimal literal\n"
        f"{fmt_rec(120)} [INFO] [intelligence.telemetry_collector] Starting GlobalTelemetryCollector for symbols: BTCUSDC\n"
        f"{fmt_rec(60)} [INFO] [intelligence.telemetry_collector] Global telemetry cycle active: 10 symbols monitored\n",
        encoding="utf-8",
    )
    rec_sum = harvest_log_anomalies_from_dirs([str(log_dir_rec)], lookback_hours=1.0)
    assert rec_sum.error_count == 0
    assert rec_sum.persistent_errors == []

    # 7. load_symbol_price_series strips off-RTH flat 2-element poller ticks after 3-element OHLC bars
    #    and apply_24h_series_enrichment heals vol_1h_pct == 0.0 from RTH bars

    cdir_rgnt = tmp_path / "cachedb_rgnt"
    (cdir_rgnt / "prices").mkdir(parents=True, exist_ok=True)
    rth_close_ms = now_ms_pk - 14 * 3600_000
    rth_bars_21 = [
        [rth_close_ms - (20 - idx) * 900_000, round(1.41 - 0.004 * idx + (0.01 if idx % 2 else 0.0), 4), 500.0 + idx * 25.0]
        for idx in range(21)
    ]
    rth_bars_21[-1] = [rth_close_ms, 1.32, 708.0]
    flat_off_rth_ticks = [[rth_close_ms + 13 * 3600_000 + idx * 60_000, 1.32] for idx in range(30)]
    (cdir_rgnt / "prices" / "cache_24_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": rth_bars_21 + flat_off_rth_ticks}}),
        encoding="utf-8",
    )
    older_rth_bars = [
        [rth_close_ms - 86400_000 + idx * 3600_000, round(1.38 - 0.002 * idx, 4), 400.0]
        for idx in range(8)
    ]
    (cdir_rgnt / "prices" / "cache_365_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": older_rth_bars + rth_bars_21}}),
        encoding="utf-8",
    )
    s24_rgnt, _ = load_symbol_price_series(
        str(cdir_rgnt),
        "RGNT_US_EQ",
        current_price_items={"RGNT_US_EQ": [[rth_close_ms + 14 * 3600_000, 1.32]]},
    )
    assert s24_rgnt[-1][0] == rth_close_ms
    assert all(len(pt) >= 3 for pt in s24_rgnt)
    m24_rgnt = compute_24h_price_metrics(s24_rgnt)
    assert m24_rgnt.get("vp_source") == "REAL_VOL"
    assert m24_rgnt.get("chg_1h_pct") is not None
    assert m24_rgnt.get("chg_24h_pct") is not None
    entry_rgnt = {
        "price": 1.32,
        "vol_1h_pct": 0.0,
        "adapt_reentry_pct": 0.0,
        "adapt_dca_pct": 0.0,
        "kalman_vel": 0.0,
        "kalman_vel_std": 0.008,
        "slope_15m": 0.0,
        "slope_2_5h": 0.0,
        "growth_15m": 0.0,
        "growth_2_5h": 0.0,
    }
    apply_24h_series_enrichment("RGNT_US_EQ", entry_rgnt, m24_rgnt, s24_rgnt, None)
    assert entry_rgnt["vol_1h_pct"] > 0.0
    assert entry_rgnt["adapt_reentry_pct"] > 0.0
    assert entry_rgnt["adapt_dca_pct"] > 0.0
    assert entry_rgnt["kalman_vel"] != 0.0
    assert entry_rgnt["slope_2_5h"] != 0.0

    # Also strips flat off-RTH tail when series consists purely of 2-element [ts, px] points
    cdir_rgnt_2el = tmp_path / "cachedb_rgnt_2el"
    (cdir_rgnt_2el / "prices").mkdir(parents=True, exist_ok=True)
    rth_2el_pts = [[pt[0], pt[1]] for pt in rth_bars_21]
    (cdir_rgnt_2el / "prices" / "cache_24_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": rth_2el_pts + flat_off_rth_ticks}}),
        encoding="utf-8",
    )
    s24_rgnt_2el, _ = load_symbol_price_series(str(cdir_rgnt_2el), "RGNT_US_EQ")
    assert s24_rgnt_2el[-1][0] == rth_close_ms

    # 8. OrderbookDepthCollector._fetch_equity_orderbook rejects wide off-RTH stub L1 quotes
    #    (e.g. 1.18/1.53 around 1.32) and keeps bid_wall_px < mid_price < ask_wall_px

    ob_col_eq = OrderbookDepthCollector(cache_dir=str(cdir_rgnt))
    ob_col_eq._fetch_equity_market_context = lambda sym, now, **kw: {
        "ticker": "RGNT",
        "last_price": 1.32,
        "best_bid": 1.18,
        "best_ask": 1.53,
        "bid_size_sh": 500.0,
        "ask_size_sh": 500.0,
        "daily_vol_sh": 15000.0,
        "bars": [
            (now - (10 - i) * 300.0, 1.32, 1.33, 1.31, 1.32, 800.0)
            for i in range(10)
        ],
    }
    snap_eq_ob = ob_col_eq._fetch_equity_orderbook("RGNT_US_EQ", now_s_rec)
    assert snap_eq_ob is not None
    assert abs(snap_eq_ob.mid_price - 1.32) < 1e-4
    assert snap_eq_ob.spread_bps < 100.0
    assert snap_eq_ob.largest_bid_wall_price < snap_eq_ob.mid_price
    assert snap_eq_ob.largest_ask_wall_price > snap_eq_ob.mid_price


    eq_m_stub = {
        "ob_mid_price": 1.355,
        "best_bid": 1.18,
        "best_ask": 1.53,
        "spread_bps": 2583.0,
        "bid_wall_px": 1.32,
        "ask_wall_px": 1.32,
    }
    assert align_equity_orderbook_to_spot(eq_m_stub, 1.32) is True
    assert eq_m_stub["ob_mid_price"] == 1.32
    assert eq_m_stub["spread_bps"] < 100.0
    assert eq_m_stub["bid_wall_px"] < 1.32 < eq_m_stub["ask_wall_px"]

    # 9. filter_valid_severity_keywords excludes war-ending de-escalation headlines ("even if the Iran war ends soon")
    #    and compute_dynamic_geopolitical_interval / harvest_macro_and_news_sensors reconcile 90m de-escalation summary
    #    when fresh (<90m) high-severity headlines arrive

    assert (
        filter_valid_severity_keywords(
            ["war", "iran war"],
            title="Interest rates could go significantly higher if oil prices keep climbing — even if the Iran war ends soon",
        )
        == ()
    )
    assert (
        filter_valid_severity_keywords(
            ["war", "iran war"],
            title="Peace talks to end Iran war collapse as negotiations fail",
        )
        == ("war", "iran war")
    )
    itv_deesc_res, rsn_deesc_res = compute_dynamic_geopolitical_interval(
        base_interval_sec=16200.0,
        last_eval_ts=now_s_rec - 2400.0,
        now_ts=now_s_rec,
        cached_assessment=SimpleNamespace(
            threat_level="NORMAL",
            is_fallback=False,
            summary="Geopolitical threat de-escalated to NORMAL: no fresh high-severity macro headlines in the last 90 minutes.",
        ),
        news_snapshot={
            "high_severity_count": 1,
            "has_critical_shock_keywords": False,
            "headlines": [
                {
                    "title": "Sudden naval blockade escalates shipping tensions",
                    "source": "Reuters",
                    "is_high_severity": True,
                    "first_seen_ts": now_s_rec - 300.0,
                }
            ],
        },
        headlines_changed=True,
        last_headlines_digest="Old headline from 3 hours ago",
    )
    assert itv_deesc_res == 0.0
    assert rsn_deesc_res == "fresh_high_sev_after_90m_deescalation"

    # 10. enrich_24h_and_microstructure caps stale T212 position ts against cur_p_items and folds sparse (<15m) post-gap premarket stubs

    cdir_rgnt_gap = tmp_path / "cachedb_rgnt_gap"
    (cdir_rgnt_gap / "prices").mkdir(parents=True, exist_ok=True)
    (cdir_rgnt_gap / "state").mkdir(parents=True, exist_ok=True)
    rth_end_ms = int((now_s_rec - 16.0 * 3600.0) * 1000)
    rgnt_gap_bars = [
        [rth_end_ms - (30 - i) * 300_000, round(1.38 - (i % 5) * 0.01, 4), 600.0]
        for i in range(30)
    ] + [
        [rth_end_ms, 1.32, 708.0],
        [int((now_s_rec - 300.0) * 1000), 1.27, 0.0],
        [int((now_s_rec - 97.0) * 1000), 1.27, 0.0],
    ]
    (cdir_rgnt_gap / "prices" / "cache_24_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": rgnt_gap_bars}}),
        encoding="utf-8",
    )
    (cdir_rgnt_gap / "state" / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:t212:RGNT_US_EQ": {
                    "symbol": "RGNT_US_EQ",
                    "free": 122.51,
                    "price": 1.27,
                    "ts": now_s_rec - 60.0,
                }
            }
        ),
        encoding="utf-8",
    )
    pt_rgnt_gap = {"RGNT_US_EQ": {"price": 1.32, "trend": 0, "growth_coefficient": 0.0}}
    cur_p_rgnt = {"RGNT_US_EQ": [[rth_end_ms, 1.32]]}
    enrich_24h_and_microstructure(str(cdir_rgnt_gap), str(tmp_path), pt_rgnt_gap, {}, cur_p_rgnt, {}, now_s_rec)
    assert pt_rgnt_gap["RGNT_US_EQ"]["price"] == 1.27
    assert pt_rgnt_gap["RGNT_US_EQ"].get("chg_15m_pct") is not None
    assert pt_rgnt_gap["RGNT_US_EQ"].get("chg_1h_pct") is not None
    assert pt_rgnt_gap["RGNT_US_EQ"].get("slope_15m") is not None

    # 10h. extract_symbol_recent_fills preserves middle lots up to 32 open lots and retains open_lots[0] when > 32 lots exist
    # so enrich_positions_with_orders_and_cycles recovers both exact LIFO VWAP and true cycle peak since the first open lot.
    base_lot_ms = int((now_s_rec - 20 * 86400.0) * 1000)
    many_orders = [
        {"symbol": "SPCX_US_EQ", "side": "BUY", "price": 159.58, "qty": 0.53, "executedQty": 0.53, "timestamp": base_lot_ms, "status": "FILLED"}
    ] + [
        {
            "symbol": "SPCX_US_EQ",
            "side": "BUY",
            "price": 140.0 + i * 0.2,
            "qty": 0.5,
            "executedQty": 0.5,
            "timestamp": int((now_s_rec - (7 - i) * 3600.0) * 1000),
            "status": "FILLED",
        }
        for i in range(7)
    ]
    fills_many = extract_symbol_recent_fills(many_orders, symbol="SPCX_US_EQ")
    assert fills_many is not None
    assert len(fills_many["open_lots"]) == 8
    assert fills_many["open_lots"][0]["ts_ms"] == base_lot_ms

    # 10i. summarize_recent_ai_intents and build_last_cycle_execution_status surface failure/guard `reason` on FAILED/BLOCKED_BY_GUARD intents.

    failed_intents = [
        {
            "timestamp": now_s_rec - 120.0,
            "symbol": "ARBUSDC",
            "venue": "binance",
            "decision": "SELL",
            "order_type": "MARKET",
            "status": "FAILED",
            "reference_price": 0.1622,
            "suggested_notional_usd": 772.0,
            "history": [
                {
                    "ts": now_s_rec - 118.0,
                    "action": "ORDER_GATEWAY_FAILED",
                    "reason": "OrderGuard blocked binance MARKET SELL ARBUSDC: account_cache_not_fresh",
                }
            ],
        }
    ]
    failed_sum = summarize_recent_ai_intents(failed_intents, "crypto", now_s_rec, extract_intent_notional)
    assert len(failed_sum) == 1
    assert failed_sum[0].get("reason") == "OrderGuard blocked binance MARKET SELL ARBUSDC: account_cache_not_fresh"
    failed_last_st = build_last_cycle_execution_status(failed_intents, "crypto", now_s_rec, extract_intent_notional)
    failed_rec = failed_last_st[0] if isinstance(failed_last_st, list) else failed_last_st
    assert isinstance(failed_rec, dict)
    assert failed_rec.get("reason") == "OrderGuard blocked binance MARKET SELL ARBUSDC: account_cache_not_fresh"

    # 10j. apply_short_trend_item & apply_24h_series_enrichment replace stale (>10m behind 24h series) or 1-sample short trends with fresh series slopes.

    nvda_entry_st: dict = {}
    stale_tinfo = {
        "ts": now_s_rec - 10800.0,
        "current_price": 198.59,
        "final_trend": -1,
        "growth_coefficient": 0.0,
        "slope_full": 0.0,
        "gradient_recent": 0.0,
        "gradient_recent_fast": 0.0,
        "dynamic": {
            "90": {"n_samples": 1, "slope_full": 0.0, "growth_coefficient": 0.0, "gradient_recent": 0.0},
            "222": {"n_samples": 1, "slope_full": 0.0, "growth_coefficient": 0.0, "gradient_recent": 0.0},
            "900": {"n_samples": 4, "slope_full": -0.10833, "growth_coefficient": -0.05098, "gradient_recent": 0.115},
            "9000": {"n_samples": 29, "slope_full": 0.00074, "growth_coefficient": -0.01533, "gradient_recent": -0.025},
        },
    }
    apply_short_trend_item(nvda_entry_st, stale_tinfo, fallback_price=196.70)
    assert nvda_entry_st.get("slope_3_7m") is None
    fresh_s24 = [[int((now_s_rec - (40 - i) * 300.0) * 1000), round(195.0 + i * 0.05, 4)] for i in range(41)]
    apply_24h_series_enrichment("NVDA_US_EQ", nvda_entry_st, compute_24h_price_metrics(fresh_s24), fresh_s24, None)
    assert nvda_entry_st.get("slope_3_7m") is not None and nvda_entry_st["slope_3_7m"] > 0.0
    assert nvda_entry_st.get("growth_3_7m") is None
    assert nvda_entry_st.get("slope_15m") is not None and nvda_entry_st["slope_15m"] > 0.0
    assert abs(nvda_entry_st.get("gradient_recent", 0.0) - 0.115) > 1e-6


    pt_stale_dir = tmp_path / "pt_stale_slopes"
    pt_stale_dir.mkdir(parents=True, exist_ok=True)
    (pt_stale_dir / "cache_short_trend.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": stale_tinfo}}),
        encoding="utf-8",
    )
    (pt_stale_dir / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps(fresh_s24),
        encoding="utf-8",
    )
    pt_nvda_stale = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(pt_stale_dir))
    assert "15m=-0.10833" not in str(pt_nvda_stale.get("Dynamic Multi-Window Slopes"))
    assert "15m=+" in str(pt_nvda_stale.get("Dynamic Multi-Window Slopes"))

    # 10k. _extract_sorted_ts_px strips multi-tick zero-volume plateaus & apply_24h_series_enrichment replaces inflated vol_1h_pct on sparse 1h series.

    rgnt_s24 = [
        [int((now_s_rec - 86400 + i * 1800) * 1000), round(1.38 - (i % 3) * 0.01, 4), 400.0]
        for i in range(25)
    ] + [
        [int((now_s_rec - 3900) * 1000), 1.32, 500.0],
        [int((now_s_rec - 3800) * 1000), 1.27, 0.0],
        [int((now_s_rec - 3700) * 1000), 1.27, 0.0],
        [int((now_s_rec - 3500) * 1000), 1.33, 600.0],
    ] + [
        [int((now_s_rec - (15 - j) * 60) * 1000), 1.29 if j < 10 else 1.28]
        for j in range(16)
    ]
    extracted_rgnt = _extract_sorted_ts_px(rgnt_s24)
    assert all(abs(px - 1.27) > 1e-6 for _, px in extracted_rgnt)

    rgnt_cdir = tmp_path / "rgnt_cdir"
    rgnt_cdir.mkdir(parents=True, exist_ok=True)
    (rgnt_cdir / "cache_24_price_RGNT_US_EQ.json").write_text(json.dumps(rgnt_s24), encoding="utf-8")
    loaded_rgnt_24h, loaded_rgnt_7d = load_symbol_price_series(str(rgnt_cdir), "RGNT_US_EQ")
    assert all(abs(pt[1] - 1.27) > 1e-6 for pt in loaded_rgnt_24h)
    assert all(abs(px - 1.27) > 1e-6 for px in loaded_rgnt_7d)

    rgnt_entry = {
        "price": 1.28,
        "vol_1h_pct": 16.4353,
        "adapt_reentry_pct": 32.871,
        "adapt_dca_pct": 16.435,
        "_short_trend_ts": now_s_rec,
    }
    apply_24h_series_enrichment("RGNT_US_EQ", rgnt_entry, compute_24h_price_metrics(rgnt_s24), rgnt_s24, None)
    assert 0.1 < rgnt_entry["vol_1h_pct"] < 4.0
    assert rgnt_entry["adapt_reentry_pct"] < 8.0
    assert rgnt_entry["adapt_dca_pct"] < 4.0

    rgnt_s24_2ticks = [
        [int((now_s_rec - 86400 + i * 1800) * 1000), round(1.38 - (i % 3) * 0.01, 4), 400.0]
        for i in range(25)
    ] + [
        [int((now_s_rec - 2700) * 1000), 1.23, 500.0],
        [int((now_s_rec - 10) * 1000), 1.20],
    ]
    rgnt_entry_2 = {
        "price": 1.20,
        "vol_1h_pct": 6.6154,
        "adapt_reentry_pct": 13.231,
        "adapt_dca_pct": 6.615,
        "_short_trend_ts": now_s_rec,
    }
    apply_24h_series_enrichment("RGNT_US_EQ", rgnt_entry_2, compute_24h_price_metrics(rgnt_s24_2ticks), rgnt_s24_2ticks, None)
    assert 0.1 < rgnt_entry_2["vol_1h_pct"] < 4.0


    dt_neg_zero = DerivativesTelemetry.from_dict(
        {"funding_rate": -0.0, "predicted_funding_rate": -0.0, "perp_spot_basis_bps": -0.0, "funding_annualized_pct": -0.0},
        symbol="NVDA_US_EQ",
    )
    assert not str(dt_neg_zero.funding_rate).startswith("-")
    assert not str(dt_neg_zero.predicted_funding_rate).startswith("-")
    assert not str(dt_neg_zero.perp_spot_basis_bps).startswith("-")
    assert not str(dt_neg_zero.funding_annualized_pct).startswith("-")

    ldir_agy_int = tmp_path / "logs_agy_int"
    ldir_agy_int.mkdir(exist_ok=True)
    t_agy_s = time.strftime("%H:%M:%S", time.localtime(now_s_rec - 60))
    (ldir_agy_int / "assetguardian.log").write_text(
        f"agy CLI with model 'gemini-4-argon-high' failed (code 1): error: interrupted\n"
        f"{t_agy_s}  Started. check_interval=54.0s, sell_tiers=((9.0, 0.12),)\n"
        f"Exception ignored in: <module 'threading' from '/usr/lib/python3.12/threading.py'>\n"
        f"Traceback (most recent call last):\n"
        f"  File \"/usr/lib/python3.12/threading.py\", line 1573, in _shutdown\n"
        f"SystemExit: 0\n",
        encoding="utf-8",
    )
    anom_agy_int = harvest_log_anomalies_from_dirs([str(ldir_agy_int)], lookback_hours=2.0)
    assert anom_agy_int.error_count == 0


    mixed_vp_series = [
        [int((now_s_rec - 86400 + i * 3600) * 1000), round(1.38 - (i % 4) * 0.03, 4), 600.0]
        for i in range(20)
    ] + [[int((now_s_rec - (90 - j) * 10) * 1000), 1.20] for j in range(90)]
    vp_mixed = compute_volume_profile_metrics(mixed_vp_series)
    assert vp_mixed.get("vp_source") == "REAL_VOL"

    stale_vol_series = [
        [int((now_s_rec - 86400 + i * 300) * 1000), round(202.0 + (i % 4) * 0.5, 4), 1500.0]
        for i in range(20)
    ] + [[int((now_s_rec - 7200 + j * 30) * 1000), round(185.0 + (j % 5) * 0.2, 4)] for j in range(120)]
    vp_stale_vol = compute_volume_profile_metrics(stale_vol_series)
    assert vp_stale_vol.get("vp_source") == "TPO"
    assert vp_stale_vol.get("poc_px", 0.0) < 190.0

    cdir_eq_ohlc = tmp_path / "cdir_eq_ohlc"
    (cdir_eq_ohlc / "prices").mkdir(parents=True, exist_ok=True)
    (cdir_eq_ohlc / "prices" / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": stale_vol_series}}),
        encoding="utf-8",
    )
    fresh_ohlc24 = [
        [int((now_s_rec - 86400 + i * 1800) * 1000), round(200.0 - i * 0.3, 4), 1200.0]
        for i in range(44)
    ] + [
        [int((now_s_rec - 3600 + k * 300) * 1000), round(185.0 + (k % 3) * 0.25, 4), 2400.0]
        for k in range(12)
    ]
    (cdir_eq_ohlc / "prices" / "cache_ohlc_24h_NVDA_US_EQ.json").write_text(
        json.dumps({"ts": now_s_rec, "symbol": "NVDA_US_EQ", "points": fresh_ohlc24}),
        encoding="utf-8",
    )
    merged_nvda_24h, _ = load_symbol_price_series(str(cdir_eq_ohlc), "NVDA_US_EQ")
    vp_merged_nvda = compute_volume_profile_metrics(merged_nvda_24h)
    assert vp_merged_nvda.get("vp_source") == "REAL_VOL"
    assert 184.0 <= vp_merged_nvda.get("poc_px", 0.0) <= 190.0


    dt_col_eq = DerivativesTelemetryCollector(cache_dir=str(cdir_eq_ohlc))
    dt_col_eq._fetch_equity_market_context = lambda sym, now, **kw: {
        "ticker": "RGNT",
        "last_price": 1.24,
        "call_oi": 0.0,
        "put_oi": 0.0,
        "bars": [(now - 3600.0, 1.20, 1.20, 1.20, 1.20, 500.0)],
    }
    rgnt_dt = dt_col_eq._fetch_equity_derivatives("RGNT_US_EQ", now_s_rec)
    assert rgnt_dt is not None
    assert rgnt_dt.perp_spot_basis_bps == 0.0

    # Sub-5m ticks in [ohlc_last - 15m, ohlc_last] are preserved when merging cache_ohlc_24h
    last_bar_ms = fresh_ohlc24[-1][0]
    p24_with_sub5m = stale_vol_series + [
        [last_bar_ms - 180_000, 184.50],
        [last_bar_ms - 60_000, 184.80],
        [last_bar_ms + 10_000, 185.60],
    ]
    (cdir_eq_ohlc / "prices" / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": p24_with_sub5m}}),
        encoding="utf-8",
    )
    merged_sub5m_24h, _ = load_symbol_price_series(str(cdir_eq_ohlc), "NVDA_US_EQ")
    m24_sub5m = compute_24h_price_metrics(merged_sub5m_24h)
    assert "chg_1m_pct" in m24_sub5m and m24_sub5m["chg_1m_pct"] > 0.3
    assert "chg_3m_pct" in m24_sub5m and m24_sub5m["chg_3m_pct"] > 0.5

    # Cache24PriceManager preserves 3-element OHLCV bars written to disk with .meta when saving newer live tick

    c24_eq_file = str(cdir_eq_ohlc / "prices" / "cache_24_price_NVDA_US_EQ.json")
    mgr_eq = cm.Cache24PriceManager(60, ["NVDA_US_EQ"], c24_eq_file, read_only=True)
    mgr_eq.save_state = True
    json.dump(
        {"items": {"NVDA_US_EQ": fresh_ohlc24}, "fetchtime": {"NVDA_US_EQ": last_bar_ms + 5000}},
        open(c24_eq_file, "w", encoding="utf-8"),
    )
    write_meta_json(
        c24_eq_file,
        {
            "fetchtime": {"NVDA_US_EQ": last_bar_ms + 5000},
            "counts": {"NVDA_US_EQ": len(fresh_ohlc24)},
            "max_ts": last_bar_ms + 5000,
            "saved_at": last_bar_ms + 5000,
        },
    )
    mgr_eq.on_price_update("NVDA_US_EQ", last_bar_ms + 15000, 186.10)
    assert mgr_eq.save_state_to_file() is True
    saved_eq_rows = json.loads(open(c24_eq_file, encoding="utf-8").read())["items"]["NVDA_US_EQ"]
    assert sum(1 for r in saved_eq_rows if len(r) >= 3 and r[2] > 0) == len(fresh_ohlc24)
    assert saved_eq_rows[-1] == [last_bar_ms + 15000, 186.10]

    # Sub-$5 equity depth_1pct_ask_usd floor, unquoted first-person/local-crime headline filtering, and >=$20M 1h CVD flip at 5-15m advisor age

    ob_col_eq = OrderbookDepthCollector(cache_dir=str(cdir_eq_ohlc))
    ob_col_eq._fetch_equity_market_context = lambda sym, now, **kw: {
        "ticker": "RGNT",
        "last_price": 1.21,
        "best_bid": 1.205,
        "best_ask": 1.215,
        "bid_size_sh": 0.0,
        "ask_size_sh": 5.6,
        "bars": [(now - 300.0 * k, 1.21, 1.23, 1.19, 1.21, 8000.0) for k in range(20)],
    }
    rgnt_ob = ob_col_eq._fetch_equity_orderbook("RGNT_US_EQ", now_s_rec)
    assert rgnt_ob is not None
    assert rgnt_ob.depth_1pct_ask_usd >= round(rgnt_ob.ask_depth_usd * (1.0 / 1.5), 2)

    rgnt_minfo = {
        "ob_mid_price": 1.21,
        "best_bid": 1.205,
        "best_ask": 1.215,
        "bid_wall_px": 1.21,
        "ask_wall_px": 1.21,
        "spread_bps": 82.6,
        "bid_depth_usd": 1197.0,
        "ask_depth_usd": 1401.0,
        "depth_1pct_bid_usd": 798.0,
        "depth_1pct_ask_usd": 7.0,
        "est_slip_2k_bps": 250.0,
    }
    assert align_equity_orderbook_to_spot(rgnt_minfo, 1.21) is True
    enrich_microstructure_cvd(rgnt_minfo, {"price": 1.21})
    assert rgnt_minfo["depth_1pct_ask_usd"] >= 934.0
    assert rgnt_minfo["est_slip_2k_bps"] < 200.0
    assert rgnt_minfo["bid_wall_px"] == 1.208
    assert rgnt_minfo["ask_wall_px"] == 1.212

    camper_title = "I’m 65 and worth more than $5.6 million. Can I buy a $200,000 camper even though the price feels ‘crazy high’?"
    assert is_excluded_reference_news_item(camper_title) is True
    assert has_financial_or_geopolitical_context(camper_title) is False
    stabbing_title = "Jordanian officer killed in stabbing attack in Amman"
    assert has_financial_or_geopolitical_context(stabbing_title) is False
    assert is_off_topic_general_news_title(stabbing_title) is True

    sa_fast_1h_flip = {
        "market_bias": "CAUTION",
        "recommended_action": "ACCUMULATE",
        "age_min": 8.9,
        "summary": "BTC is defending a 6h HIGHER_LOW_DOUBLE_BOTTOM with renewed 5m buyer aggression.",
        "key_risks": [
            "Derivative shift to 'short_building' with negative 1h/4h CVD (-$13.2M / -$91.2M) and crowded top-trader longs."
        ],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_fast_1h_flip,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 1.3831,
                "cvd_15m_usd": 39_340_000.0,
                "taker_ratio_5m": 2.21,
                "cvd_5m_usd": 33_760_000.0,
                "taker_ratio_1h": 1.06,
                "cvd_1h_usd": 37_350_000.0,
            }
        },
    )
    assert "LIVE_1H_CVD_FLIP_POSITIVE" in sa_fast_1h_flip.get("live_flow_divergence", "")
    assert "advisor 1h_cvd=-13.20M" in sa_fast_1h_flip.get("live_flow_divergence", "")


def test_cross_venue_heavy_shorting_cvd_and_stale_partial_fill_dedup():
    """Verify:
    - Cross-venue sibling orderbook depth does not misclassify a symbol's 15m CVD as spot_absorbing when the symbol's own venue book is ask-heavy (<0.50), taker_ratio <= 0.50, and div_regime is aggressive_shorting
    - _format_portfolio_section suppresses >30m PARTIALLY_FILLED duplicates from last_cycle_execution_status and recent_ai_intents when already present in open_limit_orders
    - build_last_cycle_execution_status surfaces filled_notional_usd alongside order notional_usd for PARTIALLY_FILLED orders
    """

    m_tao_dump = {
        "TAOUSDC": {
            "ob_mid_price": 216.9,
            "ob_imbalance": 0.477,
            "bid_depth_usd": 128000.0,
            "ask_depth_usd": 140000.0,
            "taker_buy_sell_ratio": 0.3529,
            "taker_buy_vol_usd": 467962.45,
            "taker_sell_vol_usd": 1325897.20,
            "cvd_15m_usd": -857934.75,
            "divergence_regime": "aggressive_shorting",
            "_synthetic_spot_cvd": True,
        },
        "TAOUSD": {
            "ob_mid_price": 217.15,
            "ob_imbalance": 0.620,
            "bid_depth_usd": 645000.0,
            "ask_depth_usd": 395000.0,
            "taker_buy_sell_ratio": 0.3529,
            "taker_buy_vol_usd": 467962.45,
            "taker_sell_vol_usd": 1325897.20,
            "cvd_15m_usd": -857934.75,
            "divergence_regime": "aggressive_shorting",
            "_synthetic_spot_cvd": True,
        },
    }
    pt_tao_dump = {
        "TAOUSDC": {"price": 216.9, "chg_15m_pct": -0.02, "chg_5m_pct": 0.08},
        "TAOUSD": {"price": 217.15, "chg_15m_pct": -0.02, "chg_5m_pct": 0.08},
    }
    reconcile_sibling_flow_and_liq_regimes(m_tao_dump, pt_tao_dump)
    assert m_tao_dump["TAOUSDC"]["cvd_regime"] == "aggressive_market_dumps"
    assert m_tao_dump["TAOUSD"]["cvd_regime"] == "aggressive_market_dumps"

    # Also verify USDT + USD sibling pair (where USD has larger depth than USDT) does not override USDT's aggressive_market_dumps
    m_tao_usdt_dump = {
        "TAOUSDT": dict(m_tao_dump["TAOUSDC"], _cross_venue_spot_cvd=False, _synthetic_spot_cvd=True),
        "TAOUSD": dict(m_tao_dump["TAOUSD"], _cross_venue_spot_cvd=False, _synthetic_spot_cvd=True),
    }
    reconcile_sibling_flow_and_liq_regimes(
        m_tao_usdt_dump,
        {"TAOUSDT": {"price": 216.9, "chg_15m_pct": -0.02, "chg_5m_pct": 0.08}, "TAOUSD": {"price": 217.15, "chg_15m_pct": -0.02, "chg_5m_pct": 0.08}},
    )
    assert m_tao_usdt_dump["TAOUSDT"]["cvd_regime"] == "aggressive_market_dumps"
    assert m_tao_usdt_dump["TAOUSD"]["cvd_regime"] == "aggressive_market_dumps"

    st_pf = build_last_cycle_execution_status(
        [
            {
                "timestamp": 1_793_800_000.0,
                "decision": "SELL",
                "symbol": "ARBUSDC",
                "venue": "binance",
                "order_type": "LIMIT",
                "limit_price": 0.1572,
                "reference_price": 0.1516,
                "clamped_notional_usd": 1400.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "563119781", "status": "NEW"},
            }
        ],
        "crypto",
        1_793_800_600.0,
        extract_intent_notional,
        open_orders=[
            {
                "venue": "binance",
                "symbol": "ARBUSDC",
                "side": "SELL",
                "order_id": "563119781",
                "price": 0.1572,
                "orig_qty": 8905.8,
                "filled_qty": 773.9,
                "qty": 8131.9,
                "status": "PARTIALLY_FILLED",
            }
        ],
    )
    assert isinstance(st_pf, dict)
    assert st_pf["status"] == "PARTIALLY_FILLED"
    assert st_pf["filled_notional_usd"] == 121.66
    assert st_pf["notional_usd"] == 1400.0

    sec3c_pf = "\n".join(
        _format_portfolio_section(
            {
                "open_limit_orders": [
                    {
                        "venue": "binance",
                        "symbol": "ARBUSDC",
                        "side": "SELL",
                        "order_id": "563119781",
                        "price": 0.1572,
                        "orig_qty": 8905.8,
                        "filled_qty": 773.9,
                        "remaining_qty": 8131.9,
                        "status": "PARTIALLY_FILLED",
                    }
                ],
                "last_cycle_execution_status": {
                    "venue": "binance",
                    "symbol": "ARBUSDC",
                    "side": "SELL",
                    "order_type": "LIMIT",
                    "status": "PARTIALLY_FILLED",
                    "order_id": "563119781",
                    "age_min": 105.9,
                },
                "recent_ai_intents": [
                    {
                        "decision": "SELL",
                        "symbol": "ARBUSDC",
                        "venue": "binance",
                        "order_type": "LIMIT",
                        "status": "EXECUTED",
                        "venue_status": "PARTIALLY_FILLED",
                        "order_id": "563119781",
                        "age_h": 1.77,
                    },
                    {
                        "decision": "BUY",
                        "symbol": "BTCUSDC",
                        "venue": "binance",
                        "order_type": "LIMIT",
                        "status": "EXECUTED",
                        "venue_status": "FILLED",
                        "order_id": "999",
                        "age_h": 0.2,
                    },
                ],
            },
            asset_scope="crypto",
            include_macro_benchmarks=False,
        )
    )
    assert "last_cycle_execution_status:" not in sec3c_pf
    assert "563119781" not in sec3c_pf.split("Recent Autonomous Intents:")[-1]


def test_prompt_2042_audit_breadth_exclusions_advisor_pullback_news_filler_and_spcx_cvd(tmp_path):
    """Verify 20:42 EET prompt audit fixes:
    1) MarketBreadthCollector excludes USD stablecoins (UUSDT, XUSDUSDT, RWUSDUSDT) and -B tokenized stocks (SNXXBUSDT, KORUBUSDT, SNDKBUSDT).
    2) enrich_sentiment_advisor_live_flow_divergence emits LIVE_15M_PULLBACK_VS_1H_BUY on BULLISH/ACCUMULATE when 15m tr > 0.75 and 1h flow is strongly positive.
    3) filter_valid_severity_keywords & is_off_topic_general_news_title filter human-interest war filler ('Alaskans heating bills', 'cruise ship Cambodia cabin').
    4) enrich_microstructure_cvd classifies ask-heavy (ob_imbalance <= 0.46) heavy dumping (taker_ratio <= 0.50, spot_cvd < 0) as aggressive_market_dumps / SYNCHRONIZED_SPOT_PERP_SELLING even when chg_15m >= 0.05.
    """

    # 1. MarketBreadthCollector excludes UUSDT, XUSDUSDT, RWUSDUSDT, SNXXBUSDT, KORUBUSDT, SNDKBUSDT
    #    and prioritizes actual positive gainers ($5M..$10M) over negative $10M+ tickers in top_gainers
    raw_tickers = [
        {"symbol": "INJUSDT", "lastPrice": "12.50", "priceChangePercent": "3.564", "quoteVolume": "25000000"},
        {"symbol": "TONUSDT", "lastPrice": "4.80", "priceChangePercent": "0.953", "quoteVolume": "8000000"},
        {"symbol": "XTZUSDT", "lastPrice": "0.75", "priceChangePercent": "0.909", "quoteVolume": "6500000"},
        {"symbol": "TRXUSDT", "lastPrice": "0.18", "priceChangePercent": "-0.625", "quoteVolume": "50000000"},
        {"symbol": "UUSDT", "lastPrice": "1.0003", "priceChangePercent": "0.070", "quoteVolume": "61000000"},
        {"symbol": "XUSDUSDT", "lastPrice": "1.0013", "priceChangePercent": "0.050", "quoteVolume": "15800000"},
        {"symbol": "RWUSDUSDT", "lastPrice": "0.9992", "priceChangePercent": "-0.030", "quoteVolume": "20500000"},
        {"symbol": "SNXXBUSDT", "lastPrice": "45.00", "priceChangePercent": "-38.704", "quoteVolume": "14000000"},
        {"symbol": "KORUBUSDT", "lastPrice": "12.00", "priceChangePercent": "-33.614", "quoteVolume": "13000000"},
        {"symbol": "SNDKBUSDT", "lastPrice": "31.00", "priceChangePercent": "-19.702", "quoteVolume": "11000000"},
        {"symbol": "HUMAUSDT", "lastPrice": "0.032", "priceChangePercent": "-18.754", "quoteVolume": "15000000"},
        {"symbol": "RVNUSDC", "lastPrice": "0.015", "priceChangePercent": "-14.252", "quoteVolume": "12000000"},
        {"symbol": "UNIUSDT", "lastPrice": "6.80", "priceChangePercent": "-10.674", "quoteVolume": "45000000"},
        {"symbol": "BTCUSDC", "lastPrice": "75878.0", "priceChangePercent": "-5.48", "quoteVolume": "500000000"},
        {"symbol": "ETHUSDC", "lastPrice": "2247.34", "priceChangePercent": "-7.21", "quoteVolume": "300000000"},
    ]
    snap = MarketBreadthCollector.parse_tickers(raw_tickers, now_ts=1_793_904_141.0)
    assert snap is not None
    gainer_syms = [s for s, _ in snap.top_gainers]
    loser_syms = [s for s, _ in snap.top_losers]
    assert gainer_syms == ["INJUSDT", "TONUSDT", "XTZUSDT"]
    assert "UUSDT" not in gainer_syms and "XUSDUSDT" not in gainer_syms
    assert loser_syms == ["HUMAUSDT", "RVNUSDC", "UNIUSDT"]
    assert "SNXXBUSDT" not in loser_syms and "KORUBUSDT" not in loser_syms and "SNDKBUSDT" not in loser_syms

    # 2. Macro Sentiment Advisor: 15m pullback against strong +$28.71M 1h CVD on BULLISH/ACCUMULATE
    sa_bull = {
        "market_bias": "BULLISH",
        "risk_level": "MODERATE",
        "recommended_action": "ACCUMULATE",
        "age_min": 13.0,
        "summary": "BTC has reclaimed $76,049 as derivatives confirm an accumulation regime.",
        "key_risks": [],
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_bull,
        {
            "BTCUSDC": {
                "taker_buy_sell_ratio": 0.8839,
                "cvd_15m_usd": -9_790_000.0,
                "taker_ratio_5m": 0.90,
                "cvd_5m_usd": -1_890_000.0,
                "taker_ratio_1h": 1.08,
                "cvd_1h_usd": 28_710_000.0,
            }
        },
    )
    assert sa_bull["live_flow_divergence"].startswith("LIVE_15M_PULLBACK_VS_1H_BUY (")

    # 3. Human-interest war filler excluded from [HIGH-SEV] and flagged as off-topic general news
    alaska_title = "As winter sets in, the Iran war has some Alaskans dreading their heating bills"
    cruise_title = (
        "A cruise ship sold as a forever home left 203 residents in Cambodia "
        "as Iran war fuel costs hit up to $8,000 a cabin"
    )
    assert filter_valid_severity_keywords(["war", "iran war"], title=alaska_title) == ()
    assert filter_valid_severity_keywords(["war", "iran war"], title=cruise_title) == ()
    assert match_high_severity_keywords(alaska_title) == ()
    assert match_high_severity_keywords(cruise_title) == ()
    assert is_off_topic_general_news_title(alaska_title) is True
    assert is_off_topic_general_news_title(cruise_title) is True

    # 4. SPCX_US_EQ ask-heavy book (ob_imbalance=0.44) + heavy selling (tr=0.3658, cvd_15m=-$150.9M, spot_cvd=-$7.43M)
    spcx_micro = {
        "ob_imbalance": 0.44,
        "bid_depth_usd": 17_485_000.0,
        "ask_depth_usd": 22_050_000.0,
        "taker_buy_sell_ratio": 0.3658,
        "taker_buy_vol_usd": 87_000_000.0,
        "taker_sell_vol_usd": 237_943_690.0,
        "cvd_15m_usd": -150_943_690.0,
        "cvd_5m_usd": -45_000_000.0,
        "cvd_1h_usd": -210_000_000.0,
    }
    enrich_microstructure_cvd(spcx_micro, {"price": 75878.0, "chg_15m_pct": 0.27})
    assert spcx_micro["cvd_regime"] == "aggressive_market_dumps"
    assert spcx_micro["spot_cvd_15m_usd"] < 0
    assert spcx_micro["spot_perp_flow_regime"] == "SYNCHRONIZED_SPOT_PERP_SELLING"


    excl_bases = get_breadth_excluded_bases()
    assert "RGNTB" in excl_bases and "NVDAB" in excl_bases and "SNDKB" in excl_bases


    assert is_excluded_reference_news_item(
        "Prediction: These Will Be Nvidia's and Palantir's Stock Prices by 2027 (Hint: Only 1 Is a Buy)"
    ) is True
    assert is_excluded_reference_news_item(
        "Three consumer stocks to buy after the midterm elections"
    ) is True
    assert is_excluded_reference_news_item(
        "Historically Bullish Signal Emerges for Nvidia Stock"
    ) is True

    pt_zero_k = {"RGNT_US_EQ": {"price": 1.17, "kalman_vel": -0.2842, "kalman_vel_std": 0.59, "kalman_trend": 0}}
    enrich_cross_venue_trends(
        pt_zero_k,
        {},
        {"RGNT_US_EQ": {"kalman_vel": -0.0, "kalman_vel_std": 0.21944, "kalman_trend": 0}},
        {},
        {},
        {},
        {},
    )
    assert pt_zero_k["RGNT_US_EQ"]["kalman_vel"] == -0.2842

    # Conversely, if entry has zero kalman_vel (-0.0) and kalman_data has real non-zero kalman_vel, it upgrades
    pt_upgrade_k = {"RGNT_US_EQ": {"price": 1.17, "kalman_vel": -0.0, "kalman_vel_std": 0.21944, "kalman_trend": 0}}
    enrich_cross_venue_trends(
        pt_upgrade_k,
        {},
        {"RGNT_US_EQ": {"kalman_vel": -0.2842, "kalman_vel_std": 0.59, "kalman_trend": -1}},
        {},
        {},
        {},
        {},
    )
    assert pt_upgrade_k["RGNT_US_EQ"]["kalman_vel"] == -0.2842

    sec_3b_eq = "\n".join(
        _format_microstructure_section(
            {
                "RGNT_US_EQ": {
                    "ob_imbalance": 0.36,
                    "bid_depth_usd": 1563.0,
                    "ask_depth_usd": 2776.0,
                    "bid_wall_px": 1.168,
                    "bid_wall_usd": 1563.0,
                    "ask_wall_px": 1.18,
                    "ask_wall_usd": 2723.0,
                    "funding_rate": 0.0,
                    "pred_funding_rate": 0.0,
                    "oi_usd": 0.0,
                    "oi_1h_chg_pct": 0.0,
                    "perp_spot_basis_bps": 0.0,
                    "funding_annualized_pct": 0.0,
                }
            },
            {"RGNT_US_EQ": {"price": 1.17}},
        )
    )
    assert "perp_basis=+0.0bps" not in sec_3b_eq
    assert "funding_ann=+0.0%" not in sec_3b_eq
    assert "funding=0.0" not in sec_3b_eq
    assert "oi_usd=0.0" not in sec_3b_eq
    assert "liq_clusters(" not in sec_3b_eq

    # Ask-heavy book (ob_imbalance=0.473) + negative CVD (-$182.18M) keeps spot_cvd_15m_usd negative even when chg_15m_pct=+1.11%
    spcx_ask_heavy = {
        "ob_imbalance": 0.473,
        "bid_depth_usd": 203_688_580.0,
        "ask_depth_usd": 226_868_059.0,
        "taker_buy_sell_ratio": 0.7066,
        "taker_buy_vol_usd": 438_685_584.08,
        "taker_sell_vol_usd": 620_869_037.67,
        "cvd_15m_usd": -182_183_454.0,
    }
    enrich_microstructure_cvd(spcx_ask_heavy, {"price": 123.52, "chg_15m_pct": 1.11})
    assert spcx_ask_heavy["spot_cvd_15m_usd"] < 0.0

    # Sibling reconciliation also preserves negative synthetic spot CVD when aggregate book is ask-heavy and CVD < 0

    sib_micro = {
        "BTCUSDC": dict(spcx_ask_heavy, ob_mid_price=70000.0),
        "BTCUSDT": dict(spcx_ask_heavy, ob_mid_price=70000.0),
    }
    sib_pt = {
        "BTCUSDC": {"price": 70000.0, "chg_15m_pct": 1.11},
        "BTCUSDT": {"price": 70000.0, "chg_15m_pct": 1.11},
    }
    reconcile_sibling_flow_and_liq_regimes(sib_micro, sib_pt)
    assert sib_micro["BTCUSDC"]["spot_cvd_15m_usd"] < 0.0
    assert sib_micro["BTCUSDT"]["spot_cvd_15m_usd"] < 0.0

    # Zero-OI equity with whale_long_pct=0.0 and taker_buy_sell_ratio=0.0 normalizes taker_buy_sell_ratio from taker_ratio_1h

    eq_m = load_symbol_microstructure(
        "",
        "SPCX_US_EQ",
        lambda _p: None,
        ob={"best_bid": 123.50, "best_ask": 123.54, "bid_depth_usd": 100000.0, "ask_depth_usd": 120000.0},
        deriv={"funding_rate": 0.0, "open_interest_usd": 0.0},
        whale={
            "top_traders_long_pct": 0.0,
            "top_traders_long_ratio": 0.0,
            "taker_buy_sell_ratio": 0.0,
            "taker_ratio_1h": 0.80,
            "taker_buy_vol_usd": 0.0,
            "taker_sell_vol_usd": 90000.0,
            "open_interest_usd": 0.0,
        },
    )
    assert eq_m["whale_long_pct"] == 0.4444
    assert eq_m["taker_buy_sell_ratio"] == 0.80
    assert eq_m["taker_buy_vol_usd"] > 0.0

    # update_microstructure_cycle_deltas ignores unnormalized whale_long_pct=0.0 in ref_snap (t-360s) on zero-OI symbols

    now_cd = 1_793_910_000.0
    (tmp_path / "microstructure_cycle_history.json").write_text(
        json.dumps(
            {
                "SPCX_US_EQ": [
                    {"ts": now_cd - 360.0, "ob_imbalance": 0.45, "cvd_15m_usd": -100000.0, "whale_long_pct": 0.0},
                    {"ts": now_cd - 200.0, "ob_imbalance": 0.46, "cvd_15m_usd": -90000.0, "whale_long_pct": 0.40},
                ]
            }
        ),
        encoding="utf-8",
    )
    cd_map = {
        "SPCX_US_EQ": {
            "ob_imbalance": 0.47,
            "cvd_15m_usd": -80000.0,
            "whale_long_pct": 0.4444,
            "oi_usd": 0.0,
            "taker_buy_sell_ratio": 0.80,
        }
    }
    update_microstructure_cycle_deltas(str(tmp_path), cd_map, now_ts=now_cd)
    assert cd_map["SPCX_US_EQ"]["cycle_delta"]["prev_whale_long_pct"] == 0.40
    assert cd_map["SPCX_US_EQ"]["cycle_delta"]["whale_long_delta_pct"] == 0.0444

    # Closed cash benchmark omits stale 1h/15m intraday deltas in summary line

    closed_line = format_macro_benchmarks_summary_line(
        {
            "items": {
                "QQQ": {
                    "price": 651.92,
                    "chg_24h_pct": -5.79,
                    "chg_1h_pct": 0.65,
                    "chg_15m_pct": 0.19,
                    "market_state": "CLOSED",
                }
            }
        }
    )
    assert "QQQ=$651.92 (-5.79% 24h, closed)" in closed_line
    assert "+0.65% 1h" not in closed_line

    # Equity with options OI normalizes legacy perp_spot_basis_bps to 0.0, rescales <0.176 5m CLV ratio, and omits zero funding/perp_basis
    nvda_m = load_symbol_microstructure(
        "",
        "NVDA_US_EQ",
        lambda _p: None,
        ob={"best_bid": 186.15, "best_ask": 186.25, "bid_depth_usd": 472_826_496.0, "ask_depth_usd": 522_399_263.0},
        deriv={"open_interest_usd": 285_486_813_126.0, "funding_rate": -2e-6, "predicted_funding_rate": -2e-6, "funding_annualized_pct": -0.22, "perp_spot_basis_bps": 49.0},
        whale={
            "open_interest_usd": 285_721_421_256.0,
            "open_interest_1h_change_pct": 0.21,
            "top_traders_long_pct": 0.4996,
            "taker_buy_sell_ratio": 1.1479,
            "taker_buy_vol_usd": 1_697_892_844.83,
            "taker_sell_vol_usd": 1_479_124_636.85,
            "taker_ratio_5m": 0.0754,
            "cvd_5m_usd": -1_163_122_701.0,
        },
    )
    assert nvda_m is not None
    assert nvda_m["perp_spot_basis_bps"] == 0.0
    assert nvda_m["funding_rate"] == 0.0
    assert nvda_m["funding_annualized_pct"] == 0.0
    assert nvda_m["taker_ratio_5m"] >= 0.1765
    assert abs(nvda_m["cvd_5m_usd"]) < 1_000_000_000.0
    sec_3b_nvda = "\n".join(_format_microstructure_section({"NVDA_US_EQ": nvda_m}, {"NVDA_US_EQ": {"price": 186.2}}))
    assert "perp_basis=" not in sec_3b_nvda
    assert "funding=" not in sec_3b_nvda
    assert "funding_ann=" not in sec_3b_nvda
    assert "oi_usd=285486813126.0" in sec_3b_nvda

    # Volume profile Winsorizes single-bar block volume spikes (8x median) and compute_24h_price_metrics resolves 30m from -25m across post-market gaps

    rgnt_block_series = [
        [int((now_cd - 86400 + i * 1800) * 1000), round(1.35 - (i % 6) * 0.03, 4), 1200.0]
        for i in range(20)
    ] + [
        [int((now_cd - 3600) * 1000), 1.20, 45140.0],
        [int((now_cd - 1500) * 1000), 1.19, 800.0],
        [int((now_cd - 900) * 1000), 1.18, 600.0],
        [int((now_cd - 300) * 1000), 1.21, 500.0],
        [int(now_cd * 1000), 1.22, 400.0],
    ]
    m24_rgnt_block = compute_24h_price_metrics(rgnt_block_series)
    assert m24_rgnt_block.get("chg_30m_pct") is not None
    assert m24_rgnt_block["vah_px"] - m24_rgnt_block["val_px"] >= 0.03
    assert m24_rgnt_block["va_pos_pct"] < 150.0

    sw_bounce = compute_swing_wave_metrics([1.30, 1.36, 1.17, 1.22])
    assert sw_bounce["swing_low_px"] == 1.17
    assert sw_bounce["swing_high_px"] == 1.36
    assert sw_bounce["rally_from_swing_low_pct"] == 4.27
    assert sw_bounce["pullback_from_swing_high_pct"] == -10.29

    # load_stitched_7d_ohlc_closes anchors 24h peak/trough at their chronological tail extrema and recalculate_live_price_geometry promotes major_swing_low_px when p_cur == swing_low_px

    fake_ohlc = lambda _s, _tf: SimpleNamespace(
        timestamps=[now_cd - (10 - i) * 3600 for i in range(10)],
        closes=[0.195, 0.198, 0.191, 0.186, 0.1783, 0.181, 0.1852, 0.1845, 0.1841, 0.18398],
    )
    stitched_7d = load_stitched_7d_ohlc_closes(
        fake_ohlc, "ADAUSD", now_cd, {"low_24h": 0.177602, "high_24h": 0.199354}, 0.183943
    )
    assert stitched_7d.index(0.199354) < stitched_7d.index(0.177602)
    m7_ada = compute_7d_range_metrics(stitched_7d)
    assert m7_ada["swing_low_px"] == 0.177602
    assert m7_ada["rally_from_swing_low_pct"] > 3.0

    pinfo_ada = {
        "price": 0.18398,
        "swing_low_px": 0.18398,
        "swing_high_px": 0.199354,
        "rally_from_swing_low_pct": 0.0,
        "pullback_from_swing_high_pct": -7.71,
        "major_swing_low_px": 0.177602,
        "rally_from_major_low_pct": 3.59,
    }
    recalculate_live_price_geometry(pinfo_ada, 0.183943)
    assert pinfo_ada["swing_low_px"] == 0.177602
    assert pinfo_ada["rally_from_swing_low_pct"] == 3.57
    assert "major_swing_low_px" not in pinfo_ada

    # Cycle high seeding anchors _prices_diverged against max(pos_px, avg_cost_px, entry_price) so >26% peak drawdowns preserve trail_peak/pos_peak

    cdir_pk = tmp_path / "cdir_pk"
    cdir_pk.mkdir(exist_ok=True)
    oldest_rgnt_ms = int((now_cd - 5 * 86400) * 1000)
    (cdir_pk / "cache_365_price_RGNT_US_EQ.jsonl").write_text(
        json.dumps({"s": "RGNT_US_EQ", "i": [oldest_rgnt_ms + 3600000, 1.60]}) + "\n",
        encoding="utf-8",
    )
    pos_rgnt_dd = [{"venue": "t212", "symbol": "RGNT_US_EQ", "price": 1.17, "free_base": 122.51, "value_usd": 143.34}]
    t212_cyc_dd = {
        "RGNT_US_EQ": {
            "qty": 122.51,
            "cost_usd": 176.35,
            "avg_cost_px": 1.4395,
            "entry_price": 1.4395,
            "pos_peak": 1.4395,
            "last_buy_price": 1.43,
            "last_sell_price": 1.37,
        }
    }
    ro_rgnt_dd = {
        "open_lots": [{"qty": 122.51, "price": 1.4395, "ts_ms": oldest_rgnt_ms}],
        "active_avg_buy_px": 1.4395,
        "last_buy": {"price": 1.43, "qty": 72.14, "ts_ms": oldest_rgnt_ms + 7200000},
    }
    enrich_positions_with_orders_and_cycles(
        pos_rgnt_dd,
        {"RGNT_US_EQ": {"price": 1.17, "high_24h": 1.36}},
        {},
        {"t212": {"RGNT_US_EQ": ro_rgnt_dd}},
        {"RGNT_US_EQ": ro_rgnt_dd},
        {},
        {},
        t212_cyc_dd,
        {},
        cache_dir=str(cdir_pk),
    )
    assert pos_rgnt_dd[0]["trailing_peak"] == 1.6
    assert pos_rgnt_dd[0]["dd_from_peak_pct"] == -26.88
    assert t212_cyc_dd["RGNT_US_EQ"]["pos_peak"] == 1.6
    assert t212_cyc_dd["RGNT_US_EQ"]["dd_from_peak_pct"] == -26.88

    # Bug 126: parse_cited_taker_ratio recognizes "headline" ratio token in parenthesized multi-horizon groups

    sa_headline_sell = (
        "BTC broke below its prior 6h low ($75,534.30) to a new 1h low of $75,322.63 "
        "amid aggressive short-term taker selling (0.6409 headline, 0.3151 5m ratio, -$20.16M 5m CVD)."
    )
    assert parse_cited_taker_ratio(sa_headline_sell) == 0.6409
    assert parse_cited_taker_ratio(sa_headline_sell, horizon="headline") == 0.6409
    assert parse_cited_taker_ratio(sa_headline_sell, horizon="5m") == 0.3151

    sa_headline_buy = (
        "BTC reclaimed the prior 6h low ($75,530.70) to trade at $75,612.00, "
        "supported by strong taker buying (1.5639 headline, 1.6606 5m ratio, +$7.68M 5m CVD)."
    )
    assert parse_cited_taker_ratio(sa_headline_buy) == 1.5639
    assert parse_cited_taker_ratio(sa_headline_buy, horizon="5m") == 1.6606

    # Bug 127: collect_pretrade_telemetry prefers fresher venue_balances_state pos:t212:* price over stale timestamped cache_current_price tick and normalizes Whale Taker Flow on zero-OI equities

    cdir_eq_vb = tmp_path / "cdir_eq_vb"
    cdir_eq_vb.mkdir(exist_ok=True)
    stale_rth_ms = int((now_cd - 36000) * 1000)
    eq_rth_series = [
        [stale_rth_ms - (24 - i) * 3600 * 1000, round(200.0 - i * 0.75, 4), 1000.0]
        for i in range(25)
    ]
    (cdir_eq_vb / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": eq_rth_series}}),
        encoding="utf-8",
    )
    (cdir_eq_vb / "cache_current_price.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": [[stale_rth_ms, 182.0]]}}),
        encoding="utf-8",
    )
    (cdir_eq_vb / "venue_balances_state.json").write_text(
        json.dumps({"pos:t212:NVDA_US_EQ": {"symbol": "NVDA_US_EQ", "price": 184.88, "ts": now_cd - 30.0}}),
        encoding="utf-8",
    )
    (cdir_eq_vb / "orderbook_depth_NVDA_US_EQ.json").write_text(
        json.dumps({
            "symbol": "NVDA_US_EQ",
            "imbalance_ratio": 0.48,
            "mid_price": 182.0,
            "best_bid": 181.95,
            "best_ask": 182.05,
            "spread_bps": 5.49,
            "bid_depth_usd": 400000000.0,
            "ask_depth_usd": 420000000.0,
            "largest_bid_wall_price": 181.78,
            "largest_bid_wall_usd": 350000000.0,
            "largest_ask_wall_price": 183.68,
            "largest_ask_wall_usd": 320000000.0,
        }),
        encoding="utf-8",
    )
    (cdir_eq_vb / "whale_snapshot_NVDA_US_EQ.json").write_text(
        json.dumps({
            "symbol": "NVDA_US_EQ",
            "top_traders_long_pct": 0.0,
            "taker_buy_sell_ratio": 0.80,
            "taker_ratio_1h": 0.179,
            "taker_buy_vol_usd": 0.0,
            "taker_sell_vol_usd": 8874.08,
            "open_interest_usd": 0.0,
        }),
        encoding="utf-8",
    )
    tel_eq_vb = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(cdir_eq_vb))
    assert "184.66" in str(tel_eq_vb.get("Orderbook Bid Wall Ahead"))
    assert "186.59" in str(tel_eq_vb.get("Orderbook Ask Wall Ahead")) or "186.58" in str(tel_eq_vb.get("Orderbook Ask Wall Ahead"))
    assert tel_eq_vb.get("Whale Taker Flow") == "Buyer/Seller ratio: 0.18"

    # Bug 128: collect_pretrade_telemetry does not overwrite newer Yahoo RTH bar price with older cached spot price timestamp
    cdir_eq_old_cp = tmp_path / "cdir_eq_old_cp"
    cdir_eq_old_cp.mkdir(exist_ok=True)
    rth_close_ms = int((now_cd - 7200) * 1000)
    older_cp_ms = int((now_cd - 36000) * 1000)
    eq_rth_bars_25 = [
        [rth_close_ms - (24 - i) * 3600 * 1000, round(180.0 + i * 0.2, 4), 1000.0]
        for i in range(25)
    ]
    (cdir_eq_old_cp / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": eq_rth_bars_25}}),
        encoding="utf-8",
    )
    (cdir_eq_old_cp / "cache_current_price.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": [[older_cp_ms, 182.0]]}}),
        encoding="utf-8",
    )
    tel_eq_old_cp = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(cdir_eq_old_cp))
    assert "24h=+2.67%" in str(tel_eq_old_cp.get("24h Multi-Horizon Price Evolution"))

    # Bug 129: collect_pretrade_telemetry reports micro-cap equity orderbook walls (>= $500) with 4-decimal precision
    (cdir_eq_old_cp / "orderbook_depth_RGNT_US_EQ.json").write_text(
        json.dumps({
            "symbol": "RGNT_US_EQ",
            "imbalance_ratio": 0.362,
            "mid_price": 1.17,
            "best_bid": 1.1651,
            "best_ask": 1.1749,
            "spread_bps": 83.33,
            "bid_depth_usd": 1579.0,
            "ask_depth_usd": 2779.0,
            "largest_bid_wall_price": 1.1681,
            "largest_bid_wall_usd": 1544.0,
            "largest_ask_wall_price": 1.1797,
            "largest_ask_wall_usd": 2600.0,
        }),
        encoding="utf-8",
    )
    tel_rgnt = collect_pretrade_telemetry("RGNT_US_EQ", cachedb_dir=str(cdir_eq_old_cp))
    assert tel_rgnt.get("Orderbook Bid Wall Ahead") == "$1,544 @ $1.1681"
    assert tel_rgnt.get("Orderbook Ask Wall Ahead") == "$2,600 @ $1.1797"

    # Bug 130: Flat short-window plateau falls back to 900s/9000s epsilon and backfills zeroed fast slopes from live price geometry
    ada_entry = {
        "price": 0.4310,
        "recent_tape_15m": ["-15m:0.4305", "-5m:0.4310", "-3m:0.4310", "-1m:0.4310", "now:0.4310"],
        "chg_1m_pct": 0.0,
        "chg_3m_pct": 0.0,
        "low_24h": 0.4200,
        "high_24h": 0.4400,
    }
    apply_short_trend_item(
        ada_entry,
        {
            "slope_full": 0.0,
            "growth_coefficient": 0.0,
            "gradient_recent": 0.0,
            "gradient_recent_fast": 0.0,
            "epsilon": 0.0,
            "final_trend": 0,
            "dynamic": {
                "90": {"slope_full": 0.0, "growth_coefficient": 0.0, "gradient_recent": 0.0, "epsilon": 0.0, "trend": 0},
                "222": {"slope_full": 0.0, "growth_coefficient": 0.0, "gradient_recent": 0.0, "epsilon": 0.0, "trend": 0},
                "900": {"slope_full": -5e-07, "growth_coefficient": -1e-06, "gradient_recent": -2e-05, "epsilon": 2.1e-05, "trend": 0},
                "9000": {"slope_full": 1.1e-06, "growth_coefficient": 2.5e-06, "gradient_recent": 1.5e-05, "epsilon": 1.9e-05, "trend": 1},
            },
        },
    )
    assert abs(ada_entry["epsilon"] - 2.1e-05) < 1e-09
    recalculate_live_price_geometry(ada_entry, 0.4316)
    assert abs(ada_entry["chg_1m_pct"] - 0.14) <= 0.01
    assert ada_entry["slope_90s"] > 0.0
    assert ada_entry["slope_3_7m"] > 0.0
    assert ada_entry["gradient_recent_fast"] > 0.0
    ada_live_entry = {"price": 0.4310}
    apply_live_24trend_windows(
        ada_live_entry,
        {
            "90": {"slope_full": 0.0, "growth_coefficient": 0.0, "gradient_recent": 0.0, "epsilon": 0.0, "trend": 0},
            "222": {"slope_full": 0.0, "growth_coefficient": 0.0, "gradient_recent": 0.0, "epsilon": 0.0, "trend": 0},
            "900": {"slope_full": -5e-07, "growth_coefficient": -1e-06, "gradient_recent": -2e-05, "epsilon": 2.3e-05, "trend": 0},
        },
    )
    assert abs(ada_live_entry["epsilon"] - 2.3e-05) < 1e-09

    # Bug 131: update_microstructure_cycle_deltas does not fabricate phantom d_cvd from >15m-old history when metric is frozen
    cdir_cd_frozen = tmp_path / "cdir_cd_frozen"
    cdir_cd_frozen.mkdir(exist_ok=True)
    (cdir_cd_frozen / "microstructure_cycle_history.json").write_text(
        json.dumps(
            {
                "NVDA_US_EQ": [
                    {"ts": now_cd - 2400.0, "ob_imbalance": 0.469, "cvd_15m_usd": 218000000.0, "whale_long_pct": 0.4996},
                    {"ts": now_cd - 1200.0, "ob_imbalance": 0.469, "cvd_15m_usd": -352000000.0, "whale_long_pct": 0.4996},
                    {"ts": now_cd - 360.0, "ob_imbalance": 0.469, "cvd_15m_usd": -352000000.0, "whale_long_pct": 0.4996},
                ]
            }
        ),
        encoding="utf-8",
    )
    cd_frozen_map = {
        "NVDA_US_EQ": {
            "ob_imbalance": 0.469,
            "cvd_15m_usd": -352000000.0,
            "whale_long_pct": 0.4996,
            "oi_usd": 285000000000.0,
            "taker_buy_sell_ratio": 0.8351,
        }
    }
    update_microstructure_cycle_deltas(str(cdir_cd_frozen), cd_frozen_map, now_ts=now_cd)
    assert cd_frozen_map["NVDA_US_EQ"]["cycle_delta"]["cvd_15m_delta_usd"] == 0.0

    # Bug 132: collect_macro_benchmarks_snapshot preserves true 24h bar change for VIX/GTH indices and pre-market ETFs when Yahoo resets fulldayChangePercent/previousClose

    def _fake_vix_gth_fetch(url, timeout=3.5, headers=None):
        if "%5EVIX" in url or "^VIX" in url:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "regularMarketPrice": 27.60,
                                "previousClose": 28.18,
                                "chartPreviousClose": 28.18,
                                "hasPrePostMarketData": False,
                            },
                            "timestamp": [int(now_cd - 86400), int(now_cd - 43200), int(now_cd)],
                            "indicators": {"quote": [{"close": [17.08, 28.18, 27.60]}]},
                        }
                    ]
                }
            }
        if "SPY" in url:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "regularMarketPrice": 733.0,
                                "fulldayPrice": 736.25,
                                "fulldayChangePercent": 0.42,
                                "previousClose": 766.81,
                                "chartPreviousClose": 766.81,
                                "hasPrePostMarketData": True,
                                "marketState": "PRE",
                            },
                            "timestamp": [int(now_cd - 86100), int(now_cd - 43200), int(now_cd)],
                            "indicators": {"quote": [{"close": [766.18, 733.0, 736.25]}]},
                        }
                    ]
                }
            }
        return {}

    vix_snap = collect_macro_benchmarks_snapshot(
        str(tmp_path / "cdir_vix_gth"), now_ts=now_cd, persist=False, fetch_fn=_fake_vix_gth_fetch
    )
    assert vix_snap is not None
    vix_item = vix_snap["items"]["VIX"]
    assert abs(vix_item["chg_24h_pct"] - 61.59) <= 0.02
    assert abs(vix_item["globex_session_chg_pct"] - (-2.06)) <= 0.02
    spy_item = vix_snap["items"]["SPY"]
    assert abs(spy_item["chg_24h_pct"] - (-3.91)) <= 0.02
    assert abs(spy_item["globex_session_chg_pct"] - 0.42) <= 0.02

    # Bug 133: collect_pretrade_telemetry & load_symbol_best_recent_fills normalize underscore symbols (HYPE_USDC)
    cdir_us = tmp_path / "cdir_underscore_sym"
    cdir_us.mkdir(exist_ok=True)
    hype_bars = [
        [int((now_cd - (20 - i) * 60) * 1000), round(77.50 + i * 0.01, 4), 100.0]
        for i in range(21)
    ]
    (cdir_us / "cache_24_price_HYPEUSDC.json").write_text(
        json.dumps({"items": {"HYPEUSDC": hype_bars}}),
        encoding="utf-8",
    )
    (cdir_us / "orderbook_depth_HYPEUSDC.json").write_text(
        json.dumps({
            "symbol": "HYPEUSDC",
            "imbalance_ratio": 0.91,
            "mid_price": 77.3135,
            "best_bid": 77.312,
            "best_ask": 77.315,
            "spread_bps": 0.39,
            "bid_depth_usd": 532631.0,
            "ask_depth_usd": 53009.0,
        }),
        encoding="utf-8",
    )
    (cdir_us / "cache_order_kraken.json").write_text(
        json.dumps({
            "items": {
                "HYPEUSD": [
                    {"id": "k-buy-1", "side": "buy", "status": "closed", "price": 80.0, "filled": 25.0, "timestamp": int((now_cd - 3600) * 1000)},
                ]
            }
        }),
        encoding="utf-8",
    )
    (cdir_us / "cache_trade_hyperliquid.json").write_text(
        json.dumps({
            "items": {
                "HYPE": [
                    {"oid": "hl-buy-1", "side": "B", "status": "filled", "px": 83.8, "sz": 5.75, "time": int((now_cd - 7200) * 1000)},
                ]
            }
        }),
        encoding="utf-8",
    )
    tel_hype_us = collect_pretrade_telemetry("HYPE_USDC", cachedb_dir=str(cdir_us))
    assert "now:77.3135" in str(tel_hype_us.get("24h Multi-Horizon Price Evolution"))
    assert "5.75 @ 83.8" in str(tel_hype_us.get("Recent Venue Fills"))

    # Bug 134: _format_venue_cycle_line reconciles pnl_vs_last_buy without 2-decimal dd_pct rounding drift

    spcx_cyc_line = _format_venue_cycle_line(
        "t212",
        "SPCX_US_EQ",
        {
            "qty": 19.627275,
            "cost_usd": 2845.73,
            "avg_cost_px": 144.988,
            "entry_price": 149.84,
            "last_buy_price": 140.51,
            "last_sell_price": 152.71,
            "last_exit_kind": "TP",
            "dca_buys": 1,
            "dd_pct": -12.9,
            "pnl_vs_avg_pct": -12.9,
        },
    )
    assert "(pnl_vs_last_buy=-10.13%)" in spcx_cyc_line

    # Bug 133: enrich_cross_venue_trends aligns watcher 7d chg_7d_pct with live sym_px_cur across pre/post ob_mid_price updates

    w_hype_7d = {
        "HYPE": [
            [1793406572768, 89.865],
            [1793670000000, 89.9475],
            [1793946582910, 77.7125],
        ]
    }
    pt_hype_7d = {"HYPEUSDC": {"price": 77.703}}
    enrich_cross_venue_trends(pt_hype_7d, {}, {}, {}, {}, {}, w_hype_7d)
    assert pt_hype_7d["HYPEUSDC"]["chg_7d_pct"] == -13.53
    pt_hype_7d["HYPEUSDC"]["price"] = 77.5095
    enrich_cross_venue_trends(pt_hype_7d, {}, {}, {}, {}, {}, w_hype_7d)
    assert pt_hype_7d["HYPEUSDC"]["chg_7d_pct"] == -13.75

    # Bug 134: fresh orderbook_depth_<SYM>.json (>60s newer than stale 24h series) updates price & chg_7d_pct even on >1% move or >10% 7d drawdown

    cdir_134 = tmp_path / "b134_cache"
    cdir_134.mkdir()
    now_134 = time.time()
    s24_ts_ms = int((now_134 - 3600.0) * 1000)
    (cdir_134 / "cache_24_price_TAOUSDC.json").write_text(
        json.dumps(
            {
                "items": {
                    "TAOUSDC": [
                        [s24_ts_ms - 4 * 900_000, 218.0],
                        [s24_ts_ms - 3 * 900_000, 218.2],
                        [s24_ts_ms - 2 * 900_000, 218.3],
                        [s24_ts_ms - 900_000, 218.4],
                        [s24_ts_ms, 218.5],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (cdir_134 / "cache_sparse_price_TAOUSDC.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"s": "TAOUSDC", "i": [s24_ts_ms - 6 * 86400 * 1000, 255.0]}),
                json.dumps({"s": "TAOUSDC", "i": [s24_ts_ms, 218.5]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cdir_134 / "orderbook_depth_TAOUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSDC",
                "ts": now_134 - 10.0,
                "mid_price": 221.2,
                "best_bid": 221.1,
                "best_ask": 221.3,
                "spread_bps": 9.05,
                "imbalance": 0.12,
                "bid_depth_usd": 50000.0,
                "ask_depth_usd": 40000.0,
            }
        ),
        encoding="utf-8",
    )
    pt_134 = {"TAOUSDC": {"price": 218.5}}
    micro_134: dict = {}
    enrich_24h_and_microstructure(str(cdir_134), str(tmp_path), pt_134, micro_134, {}, {}, now_134)
    assert pt_134["TAOUSDC"]["price"] == 221.2
    assert pt_134["TAOUSDC"]["chg_7d_pct"] == -13.25
    (cdir_134 / "cache_current_price.json").write_text(
        json.dumps({"items": {"TAOUSD": [[s24_ts_ms, 218.5]], "TAOUSDC": [[s24_ts_ms, 218.5]]}}),
        encoding="utf-8",
    )
    (cdir_134 / "orderbook_depth_TAOUSD.json").write_text(
        json.dumps(
            {
                "symbol": "TAOUSD",
                "ts": now_134 - 10.0,
                "mid_price": 221.18,
                "best_bid": 221.08,
                "best_ask": 221.28,
                "spread_bps": 9.04,
                "imbalance": 0.10,
                "bid_depth_usd": 35000.0,
                "ask_depth_usd": 30000.0,
            }
        ),
        encoding="utf-8",
    )
    (cdir_134 / "cache_ohlc_24h_TAOUSDC.json").write_text(
        json.dumps(
            {
                "ts": now_134 - 5.0,
                "points": [
                    [int((now_134 - (20 - idx) * 3600.0) * 1000), 218.5 + (idx % 3) * 0.8, 120.0 + idx * 10.0]
                    for idx in range(21)
                ]
                + [[int((now_134 - 60.0) * 1000), 221.15, 250.0]],
            }
        ),
        encoding="utf-8",
    )

    os.environ["MPTRADE_OFFLINE_DISK_CACHE"] = "1"
    try:
        pt_tele_134 = collect_pretrade_telemetry("TAO_USDC", cachedb_dir=str(cdir_134))
    finally:
        os.environ.pop("MPTRADE_OFFLINE_DISK_CACHE", None)
    assert "24h Range=[$218.0, $221.2]" in str(pt_tele_134.get("24h Multi-Horizon Price Evolution", ""))
    assert "-13.25%" in str(pt_tele_134.get("7d Price Range & Swing Structure", ""))
    assert "vp_source=REAL_VOL" in str(pt_tele_134.get("24h Volume Profile (POC / Value Area)", ""))
    assert "arb_vs_TAOUSD=+0.9bps" in str(pt_tele_134.get("Cross-Venue Relative Value (Arbitrage)", ""))

    # Bug 135: collect_pretrade_telemetry reconciles low_7d <= low_24h even when price bounced >10% above low_7d
    cdir_135 = tmp_path / "b135_cache"
    cdir_135.mkdir()
    (cdir_135 / "cache_24_price_ARBUSDC.json").write_text(
        json.dumps({"items": {"ARBUSDC": [[s24_ts_ms - 3600_000, 0.15], [s24_ts_ms, 0.1679]]}}),
        encoding="utf-8",
    )
    (cdir_135 / "cache_sparse_price_ARBUSDC.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"s": "ARBUSDC", "i": [s24_ts_ms - 6 * 86400 * 1000, 0.1715]}),
                json.dumps({"s": "ARBUSDC", "i": [s24_ts_ms - 3 * 86400 * 1000, 0.1504]}),
                json.dumps({"s": "ARBUSDC", "i": [s24_ts_ms, 0.1679]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pt_tele_135 = collect_pretrade_telemetry("ARB_USDC", cachedb_dir=str(cdir_135))
    assert "7d Range=[$0.15, $0.1715]" in str(pt_tele_135.get("7d Price Range & Swing Structure", ""))



def test_premarket_gap_preservation_and_symmetric_sell_late_scale(tmp_path, monkeypatch):
    """Verify PRE-market >1% gap bars are preserved across _chart and enrich_24h_and_microstructure,
    SELL late intelligence scale applies _suggested_scale_by_side['SELL'], and cancel_existing_ai_open_orders
    discards superseded submit_pending order_retry entries."""

    # 1. Pre-market bars after 8h overnight gap (>1% above regularMarketPrice) are preserved in enrich_24h_and_microstructure
    cdir = tmp_path / "premarket_cache"
    cdir.mkdir()
    now_s = 1_793_958_000.0
    rth_close_ms = int((now_s - 43200.0) * 1000)   # 12h ago (yesterday RTH close, volume > 0)
    post_end_ms = int((now_s - 32400.0) * 1000)    # 9h ago (yesterday post-market end, volume = 0)
    pre_start_ms = int((now_s - 3600.0) * 1000)    # 1h ago (today pre-market start after 8h gap, volume = 0, +4.3% higher)
    bars_5d = [
        [rth_close_ms - 86400_000, 198.0, 1000.0],
        [rth_close_ms - 300_000, 181.40, 2500.0],
        [rth_close_ms, 181.59, 3200.0],
        [post_end_ms, 180.50, 0.0],
        [pre_start_ms, 187.59, 0.0],
        [int((now_s - 1800.0) * 1000), 188.04, 0.0],
        [int((now_s - 900.0) * 1000), 189.47, 0.0],
        [int((now_s - 300.0) * 1000), 189.30, 0.0],
        [int(now_s * 1000), 189.47, 0.0],
    ]
    (cdir / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": bars_5d}}),
        encoding="utf-8",
    )
    pt_nvda = {"NVDA_US_EQ": {"price": 189.47}}
    micro_nvda: dict = {}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_nvda, micro_nvda, {}, {}, now_s)
    # 15m delta must compare 189.47 against 189.47 (0.0%), NOT against yesterday's 180.50 (+4.97%)
    assert abs(float(pt_nvda["NVDA_US_EQ"]["chg_15m_pct"])) < 0.05
    assert abs(float(pt_nvda["NVDA_US_EQ"]["chg_30m_pct"]) - ((189.47 - 188.04) / 188.04 * 100.0)) < 0.05
    assert abs(float(pt_nvda["NVDA_US_EQ"]["chg_1h_pct"]) - ((189.47 - 187.59) / 187.59 * 100.0)) < 0.05

    # 2. Symmetric SELL late intelligence scale reads _suggested_scale_by_side['SELL'] while ignoring regime suggested_scale
    ctx_sell = {"suggested_scale": 0.25, "_suggested_scale_by_side": {"BUY": 0.25}}
    assert extract_regime_buy_scale(ctx_sell, side="SELL") is None
    assert extract_regime_buy_scale(ctx_sell, side="BUY") == 0.25
    ctx_sell["_suggested_scale_by_side"]["SELL"] = 0.31
    assert extract_regime_buy_scale(ctx_sell, side="SELL") == 0.31

    # 3. cancel_existing_ai_open_orders discards superseded submit_pending order_retry records when include_bot_orders=True
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(order_retry, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(order_retry, "LOCK_FILE", str(l_file))
    q_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "bot_deferred_sell_1",
                        "prefix": "MO_TA_OR",
                        "symbol": "BTCUSDC",
                        "side": "SELL",
                        "qty": 0.02153,
                        "created_ts": now_s - 20.0,
                        "revision": 0,
                        "lifecycle": "submit_pending",
                        "place_kwargs": {"client_order_id": "MO_TA_OR_bot_deferred_sell_1"},
                    }
                ),
                json.dumps(
                    {
                        "id": "ai_accepted_sell_1",
                        "prefix": "OR",
                        "intent_id": "intent_1_sell_btcusdc",
                        "symbol": "BTCUSDC",
                        "side": "SELL",
                        "qty": 0.00771,
                        "order_id": "10744797899",
                        "created_ts": now_s - 60.0,
                        "revision": 0,
                        "lifecycle": "accepted",
                        "place_kwargs": {"client_order_id": "AI_intent_1_sell_btcusdc"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dummy_inst = SimpleNamespace(
        symbol="BTCUSDC",
        open_orders=lambda: [{"orderId": "10744856617", "clientOrderId": "MO_TA_OR_1", "side": "SELL", "price": "76046.32"}],
        cancel_order=lambda oid: True,
    )
    canceled = cancel_existing_ai_open_orders(dummy_inst, "SELL", include_bot_orders=True)
    assert canceled == ["10744856617"]
    remaining_ids = [r.get("id") for r in order_retry.load_all()]
    assert remaining_ids == ["ai_accepted_sell_1"]


def test_headlines_are_semantic_duplicates_syndicated_ai_stocks_rewrites():
    """Verify syndicated rewrites sharing 2-letter acronyms (AI/US) and magnitude nouns (trillions) deduplicate."""

    assert headlines_are_semantic_duplicates(
        "AI stocks climb after trillions wiped from market values",
        "AI stocks bounce back after trillions wiped from market values",
    )
    assert not headlines_are_semantic_duplicates(
        "AI stocks climb after trillions wiped from market values",
        "US labor market regains ground as job growth surges",
    )


def test_b1_through_b4_and_cached_ai_intents_collector_fallback(tmp_path, monkeypatch):
    """Verify Hyperliquid 1h annualized funding, lexicon clickbait/wall filters, lot_harvested ts guard, and cached AI intents collector fallback."""

    # B1: Hyperliquid preserves raw 1h funding_rate while annualizing over 24h/day (24 * 365 * 100)
    col = DerivativesTelemetryCollector(cache_dir=str(tmp_path), cache_ttl_sec=0.0, disk_ttl_sec=0.0)
    monkeypatch.setattr(
        col,
        "_fetch_hyperliquid_asset_ctx",
        lambda sym, timeout=4.0: {"funding": "0.0000125", "openInterest": "1000", "markPx": "80.0", "oraclePx": "80.0"},
    )
    hl_tel = col._fetch_hyperliquid_derivatives("HYPEUSDC", now=1000.0)
    assert hl_tel is not None
    assert hl_tel.funding_rate == pytest.approx(0.0000125)
    assert hl_tel.funding_annualized_pct == pytest.approx(10.95)

    # B2: Standalone 'wall' is excluded from _BROAD_MARKET_INDEX_CONTEXT_WORDS and 'may be a bargain' is caught as clickbait
    assert "wall" not in _BROAD_MARKET_INDEX_CONTEXT_WORDS
    assert _is_editorial_clickbait_or_advice_title("nvidia (nvda) stock may be a bargain after pullback")

    # B4: When last_buy ts_ms is strictly newer than last_sell ts_ms, do not mark lot_harvested
    ro_line = _format_recent_order_line(
        "TAOUSDC",
        {
            "last_action_side": "BUY",
            "last_action_actor": "AI",
            "last_buy": {"price": 220.0, "qty": 1.5, "actor": "AI", "ts_ms": 2_000_000},
            "last_sell": {"price": 235.0, "qty": 1.5, "actor": "AI", "ts_ms": 1_000_000},
            "open_lots": [{"lot_id": "L1", "qty": 1.5, "entry_px": 240.0}],
        },
        now_ms=2_500_000,
    )
    assert "lot_harvested" not in ro_line

    # B14: _load_cached_ai_intents_list resolves both state/ and root autonomous_trade_intents_state.json
    (tmp_path / "autonomous_trade_intents_state.json").write_text(
        json.dumps({"intents": [{"intent_id": "id_1", "venue": "binance", "symbol": "BTCUSDC", "status": "EXECUTED"}]}),
        encoding="utf-8",
    )
    loaded = _load_cached_ai_intents_list(str(tmp_path))
    assert len(loaded) == 1
    assert loaded[0]["intent_id"] == "id_1"


def test_harvest_positions_dedup_bare_equity_pretrade_and_assetguardian_recovery(tmp_path):
    """Verify duplicate (venue, symbol) inst_map dedup, bare equity pretrade funding suppression, and recovered AssetValue log error suppression."""

    # 1. Duplicate (venue, symbol) alias keys in inst_map do not duplicate positions, open orders, or locked_base_qty
    prov = SimpleNamespace(
        free_balance=lambda asset: 19.194454 if asset == "HYPE" else 581.83,
        usd_cash_breakdown=None,
    )
    open_list = [{"orderId": "O6SSYE-QJKQD-GTMQKU", "side": "SELL", "type": "LIMIT", "price": 81.45, "origQty": 14.0, "executedQty": 0.0, "status": "NEW"}]
    inst_conf = SimpleNamespace(
        symbol="HYPEUSD",
        provider_name="kraken",
        base="HYPE",
        quote="USD",
        params={"trailing.pct": 28.0, "mt.gain": 9.0, "mt.lost": 4.25},
        _provider=prov,
        open_orders=lambda: list(open_list),
    )
    inst_alias = SimpleNamespace(
        symbol="HYPEUSD",
        provider_name="kraken",
        base="HYPE",
        quote="USD",
        params={},
        _provider=prov,
        open_orders=lambda: list(open_list),
    )
    prices_trends = {"HYPEUSD": {"price": 75.0, "trend": 1, "growth_coefficient": 0.5}}
    positions, open_orders, _, _ = harvest_tradable_positions_and_orders(
        {"KRAKEN_HYPE": inst_alias, "KRAKEN_HYPEUSD": inst_conf},
        prices_trends,
        {},
        {},
        cache_dir=str(tmp_path),
    )
    assert len(positions) == 1
    assert len(open_orders) == 1
    assert positions[0]["locked_base_qty"] == pytest.approx(14.0)
    assert positions[0]["trailing_pct_cfg"] == 28.0

    # 2. Bare equity ticker "NVDA" suppresses 0.0 Perpetual Funding Rate in collect_pretrade_telemetry
    cdir = tmp_path / "eq_cache"
    cdir.mkdir()
    (cdir / "cache_current_price.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": {"price": 175.5, "timestamp": 1794240000000}}}),
        encoding="utf-8",
    )
    (cdir / "derivatives_telemetry_state.json").write_text(
        json.dumps({
            "items": {
                "NVDA_US_EQ": {
                    "symbol": "NVDA_US_EQ",
                    "funding_rate": 0.0,
                    "funding_annualized_pct": 0.0,
                    "perp_spot_basis_bps": 0.0,
                    "open_interest_usd": 280_000_000_000.0,
                }
            }
        }),
        encoding="utf-8",
    )
    tel_nvda = collect_pretrade_telemetry("NVDA", cachedb_dir=str(cdir))
    assert "Perpetual Funding Rate" not in tel_nvda
    assert is_equity_symbol("NVDA") is True
    assert is_equity_symbol("RNDR") is False

    # 3. Recovered AssetValue cache error in assetguardian.log is suppressed once normal cycles resume
    ldir = tmp_path / "logs"
    ldir.mkdir()
    (ldir / "assetguardian.log").write_text(
        "17:59:38 [assetguardian.py:435] ERROR reading AssetValue cache via cacheManager: stat: path should be string, bytes, os.PathLike or integer, not NoneType\n"
        "18:00:32 [BTCUSDC] price=74849.18000000, window_min/max=73501.11000000/75905.55000000, minutes_back=1440.0\n",
        encoding="utf-8",
    )
    (ldir / ".inspect_all_prompts_2026-11-09.log").write_text(
        "19:45:30 ERROR fake prompt inspection log line\n",
        encoding="utf-8",
    )
    summary = harvest_log_anomalies_from_dirs([str(ldir)], lookback_hours=1.0)
    assert summary.error_count == 0
    assert not summary.persistent_errors

    # 4. Editorial clickbait filter catches 'could steal the show'
    assert _is_editorial_clickbait_or_advice_title(
        "nvidia gets the spotlight, but this ai stock could steal the show in 2027"
    )

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_highstakeguard_check_honors_now_and_caches_timeout_fallback_iter16_s4(monkeypatch):
    """[ITER16 #4] HighStakeGuard.check honors `now` and caches timeout fallback decisions."""
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

def test_compositemarketintelligence_evaluate_guards_forwards_dur_sec_and_now_to_gemini_guard_iter17_s3():
    """[ITER17 #3] CompositeMarketIntelligence.evaluate_guards forwards `dur_sec` and `now` to gemini_guard.check."""
    # 3. CompositeMarketIntelligence.evaluate_guards forwards `dur_sec` and `now` to gemini_guard.check
    captured_gemini_kwargs = {}
    spy_gemini_guard = SimpleNamespace(
        min_notional_eur=700.0,
        check=lambda symbol, side, price, qty, **kwargs: (
            captured_gemini_kwargs.update(kwargs) or GuardDecision.allow("HighStakeGuard", "spy_ok")
        ),
    )

    comp = CompositeMarketIntelligence(gemini_guard=spy_gemini_guard)
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

def test_highstakeguard_helpers_nan_bool_resilience_iter31_s3(monkeypatch):
    """[ITER31 #3] HighStakeGuard & helpers NaN/bool resilience."""
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

def test_evaluate_high_stake_order_guard_with_nan_computed_notional_does_not_invoke_highstakeguard_iter34_s5(monkeypatch):
    """[ITER34 #5] evaluate_high_stake_order_guard with NaN computed_notional does not invoke HighStakeGuard."""
    # 5. evaluate_high_stake_order_guard with NaN computed_notional does not invoke HighStakeGuard
    called_hs = []
    monkeypatch.setattr(
        hsg_mod,
        "HighStakeGuard",
        lambda *a, **kw: SimpleNamespace(
            check=lambda *ca, **ckw: (
                called_hs.append((ca, ckw))
                or SimpleNamespace(allowed=False, reason="veto", suggested_scale=0.0, brake_action="VETO")
            )
        ),
    )
    monkeypatch.setattr(hsg_mod, "is_llm_guard_exempt", lambda: False)

    ok_nan, r_nan, sc_nan = llm_order_guard.evaluate_high_stake_order_guard(
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

def test_triggers_sanitize_bool_nan_inf_constructor_parameters_iter54_s1():
    """[ITER54 #1] Triggers sanitize bool/NaN/Inf constructor parameters."""
    # 1. Triggers sanitize bool/NaN/Inf constructor parameters
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

def test_extract_symbol_recent_fills_supports_binance_isbuyer_and_kraken_type_formats_iter101_s4():
    """[ITER101 #4] extract_symbol_recent_fills supports Binance ('isBuyer') and Kraken ('type') formats."""
    # 4. extract_symbol_recent_fills supports Binance ("isBuyer") and Kraken ("type") formats
    fills_binance = extract_symbol_recent_fills([
        {"isBuyer": True, "price": "100.0", "executedQty": "2.0", "time": 1000},
        {"isBuyer": "false", "price": "110.0", "executedQty": "1.0", "time": 2000},
    ])
    assert fills_binance is not None
    assert fills_binance["last_action_side"] == "SELL"
    assert fills_binance["last_buy"]["price"] == pytest.approx(100.0)
    assert fills_binance["last_sell"]["price"] == pytest.approx(110.0)
    assert fills_binance["active_avg_buy_px"] == pytest.approx(100.0)

    fills_kraken = extract_symbol_recent_fills([
        {"type": "buy", "price": "50.0", "qty": "4.0", "timestamp": 1000},
        {"type": "s", "price": "55.0", "qty": "4.0", "timestamp": 2000},
    ])
    assert fills_kraken is not None
    assert fills_kraken["last_action_side"] == "SELL"
    assert fills_kraken["last_buy"]["price"] == pytest.approx(50.0)
    assert fills_kraken["last_sell"]["price"] == pytest.approx(55.0)
    assert fills_kraken["active_avg_buy_px"] is None

def test_feargreedcollector_parse_payload_tolerates_malformed_timestamp_and_bool_value_classification_iter103_s6():
    """[ITER103 #6] FearGreedCollector.parse_payload tolerates malformed timestamp and bool value_classification."""
    # 6. FearGreedCollector.parse_payload tolerates malformed timestamp and bool value_classification
    fg_snap = FearGreedCollector.parse_payload({"data": [{"value": "82", "value_classification": True, "timestamp": "bad_ts"}]})
    assert fg_snap is not None and fg_snap.value == 82 and fg_snap.classification == "Neutral" and fg_snap.timestamp > 0

def test_marketbreadthcollector_parse_tickers_ignores_non_str_symbol_iter104_s3():
    """[ITER104 #3] MarketBreadthCollector.parse_tickers ignores non-str symbol."""
    # 3. MarketBreadthCollector.parse_tickers ignores non-str symbol
    tickers = [
        {"symbol": True, "quoteVolume": "10000000", "priceChangePercent": "5.0"},
        {"symbol": "BTCUSDT", "quoteVolume": "10000000", "priceChangePercent": "2.0"},
        {"symbol": "ETHUSDT", "quoteVolume": "10000000", "priceChangePercent": "3.0"},
        {"symbol": "SOLUSDT", "quoteVolume": "10000000", "priceChangePercent": "4.0"},
        {"symbol": "BNBUSDT", "quoteVolume": "10000000", "priceChangePercent": "1.0"},
        {"symbol": "XRPUSDT", "quoteVolume": "10000000", "priceChangePercent": "-1.0"},
    ]
    mb_snap = MarketBreadthCollector.parse_tickers(tickers, min_volume_usd=1_000_000.0, now_ts=1000.0)
    assert mb_snap is not None and mb_snap.total_count == 5

def test_collect_pretrade_telemetry_skips_nan_wall_prices_and_nan_funding_iter98_s4(tmp_path, monkeypatch):
    """[ITER98 #4] collect_pretrade_telemetry: skips NaN wall prices and NaN funding/whale fields."""
    # 4. collect_pretrade_telemetry: skips NaN wall prices and NaN funding/whale fields
    monkeypatch.setattr(
        "external_order_guard.get_orderbook_collector",
        lambda: SimpleNamespace(
            fetch=lambda sym, allow_network=False: SimpleNamespace(
                imbalance_ratio=0.6,
                largest_ask_wall_usd=250_000.0,
                largest_ask_wall_price=float("nan"),
                largest_bid_wall_usd=250_000.0,
                largest_bid_wall_price=50000.0,
            )
        ),
    )
    monkeypatch.setattr(
        "external_order_guard.get_derivatives_collector",
        lambda: SimpleNamespace(
            fetch=lambda sym, allow_network=False: SimpleNamespace(
                funding_rate=float("nan"),
                open_interest_usd=float("inf"),
            )
        ),
    )
    tel = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(tmp_path))
    assert tel["Orderbook Ask Wall Ahead"] == "None detected within 2%"
    assert "$250,000 @ $50,000.00" in tel["Orderbook Bid Wall Ahead"]
    assert "Perpetual Funding Rate" not in tel
    assert "Derivatives Open Interest" not in tel

def test_feargreedcollector_parse_payload_normalizes_millisecond_timestamps_1e11_to_seconds_iter123_s3():
    """[ITER123 #3] FearGreedCollector.parse_payload normalizes millisecond timestamps (> 1e11) to seconds."""
    # 3. FearGreedCollector.parse_payload normalizes millisecond timestamps (> 1e11) to seconds
    fg_snap = FearGreedCollector.parse_payload({"data": [{"value": "42", "value_classification": "Fear", "timestamp": "1760000000000"}]})
    assert fg_snap is not None and fg_snap.timestamp == 1760000000.0

def test_sentimentadvisor_filters_boolean_blank_key_risks_and_boolean_summary_iter123_s4(tmp_path):
    """[ITER123 #4] SentimentAdvisor filters boolean/blank key_risks and boolean summary from disk."""
    # 4. SentimentAdvisor filters boolean/blank key_risks and boolean summary from disk
    adv_path = tmp_path / "sentiment_eval.json"
    adv_path.write_text(
        json.dumps({
            "market_bias": "BULLISH",
            "risk_level": "LOW",
            "confidence": 0.8,
            "summary": True,
            "key_risks": ["Valid risk", True, "   ", None],
            "recommended_action": "ACCUMULATE",
            "ts": time.time(),
        }),
        encoding="utf-8",
    )
    adv = SentimentAdvisor(cache_file=str(adv_path))
    assert adv._cached_assessment is not None
    assert adv._cached_assessment.summary == ""
    assert adv._cached_assessment.key_risks == ["Valid risk"]

def test_geminiclient__remove_detached_thread_blocks_path_traversal_cids_iter123_s5(tmp_path):
    """[ITER123 #5] GeminiClient._remove_detached_thread blocks path traversal cids."""
    # 5. GeminiClient._remove_detached_thread blocks path traversal cids
    brain_dir = tmp_path / "brain" / "keep_me"
    brain_dir.mkdir(parents=True)
    GeminiClient._remove_detached_thread(str(tmp_path), "../brain")
    GeminiClient._remove_detached_thread(str(tmp_path), "keep_me/..")
    assert brain_dir.exists()
    GeminiClient._remove_detached_thread(str(tmp_path), "keep_me")
    assert not brain_dir.exists()

def test_extremegreedguard_clamps_downscale_threshold_hard_veto_threshold_iter125_s3():
    """[ITER125 #3] ExtremeGreedGuard clamps downscale_threshold <= hard_veto_threshold."""
    # 3. ExtremeGreedGuard clamps downscale_threshold <= hard_veto_threshold
    eg_guard = ExtremeGreedGuard(downscale_threshold=92, hard_veto_threshold=85)
    assert eg_guard.downscale_threshold == 85
    assert eg_guard.hard_veto_threshold == 85

def test_evaluate_llm_guard_response_handles_boolean_decision_reason_and_deny_reject_iter125_s5():
    """[ITER125 #5] evaluate_llm_guard_response handles boolean decision/reason and DENY/REJECT synonyms."""
    # 5. evaluate_llm_guard_response handles boolean decision/reason and DENY/REJECT synonyms
    dec_llm = evaluate_llm_guard_response(
        "HighStakeGuard",
        "BUY",
        {"decision": "DENIED", "reason": True, "suggested_scale": True},
        fallback_action="allow",
        notional_eur=1500.0,
    )
    assert dec_llm.allowed is False
    assert "evaluated by LLM" in dec_llm.reason

def test_meanreversiontrigger_sentimentcontrariantrigger_and_marketbreadthtrigger_normalize_inverted_thresholds_iter126_s3():
    """[ITER126 #3] MeanReversionTrigger, SentimentContrarianTrigger, and MarketBreadthTrigger normalize inverted thresholds."""
    # 3. MeanReversionTrigger, SentimentContrarianTrigger, and MarketBreadthTrigger normalize inverted thresholds
    mrt = MeanReversionTrigger(rsi_low=75.0, rsi_high=25.0, pb_low=0.9, pb_high=0.1)
    assert mrt.rsi_low == 25.0 and mrt.rsi_high == 75.0
    assert mrt.pb_low == 0.1 and mrt.pb_high == 0.9

    sct = SentimentContrarianTrigger(extreme_fear_entry=85, extreme_greed_exit=20)
    assert sct.extreme_fear_entry == 20 and sct.extreme_greed_exit == 85
    sig_sc, ev_sc = sct.evaluate("BTCUSDC", 100.0, type("FG", (), {"value": 15, "classification": True, "trend_7d_change": -5, "historical_values": []})())
    assert sig_sc == "CONTRARIAN_BUY" and "Unknown" in ev_sc.reason

    mbt = MarketBreadthTrigger(
        washout_advance_threshold=0.90,
        blowoff_advance_threshold=0.10,
        washout_median_threshold=6.0,
        blowoff_median_threshold=-4.0,
    )
    assert mbt.washout_advance_threshold == 0.10 and mbt.blowoff_advance_threshold == 0.90
    assert mbt.washout_median_threshold == -4.0 and mbt.blowoff_median_threshold == 6.0

def test_collect_pretrade_telemetry_extracts_technical_trend_regime_from_dict_regime_context_iter174_s5():
    """[ITER174 #5] collect_pretrade_telemetry extracts Technical Trend Regime from dict regime_context."""
    # 5. collect_pretrade_telemetry extracts Technical Trend Regime from dict regime_context
    telem = collect_pretrade_telemetry(
        "BTCUSDC",
        regime_context={"resolved_trend": "bull", "decision": {"gradient": 0.0123, "epsilon": 0.005}},
        dur_sec=7200.0,
    )
    assert "bull (gradient=+0.0123, epsilon=0.0050)" in telem.get("Technical Trend Regime", "")

def test_extremegreedguard_and_panicwashoutguard_accept_bid_long_and_dict_iter176_s1():
    """[ITER176 #1] ExtremeGreedGuard and PanicWashoutGuard accept BID/LONG and dict snapshots."""
    # 1. ExtremeGreedGuard and PanicWashoutGuard accept BID/LONG and dict snapshots
    eg_dec = ExtremeGreedGuard().check("BTCUSDC", "BID", {"value": 92, "classification": "Extreme Greed"})
    assert eg_dec.allowed is False and "veto_extreme_market_euphoria" in eg_dec.reason

    pw_dec = PanicWashoutGuard().check(
        "BTCUSDC",
        "LONG",
        fear_greed_snapshot={"value": 8, "historical_values": [8, 22]},
    )
    assert pw_dec.allowed is False and "defer_accelerating_panic_capitulation" in pw_dec.reason

def test_geminiclient_query_text_tolerates_malformed_none_timestamp_in__cache_iter179_s2():
    """[ITER179 #2] GeminiClient.query_text tolerates malformed/None timestamp in _cache."""
    # 2. GeminiClient.query_text tolerates malformed/None timestamp in _cache
    client = GeminiClient(custom_runner=lambda p, m, t: "fresh_response")
    client._cache["test_prompt"] = (None, "stale")
    assert client.query_text("test_prompt", cache_ttl_sec=60.0) == "fresh_response"

def test_resolve_llm_guard_cache_ttl_respects_explicit_ttl_custom_client_and_custom_iter183_s3():
    """[ITER183 #3] resolve_llm_guard_cache_ttl respects explicit TTL, custom client, and custom runner."""
    # 3. resolve_llm_guard_cache_ttl respects explicit TTL, custom client, and custom runner
    assert resolve_llm_guard_cache_ttl(120.0, True, SimpleNamespace(_custom_runner=True), 600.0) == pytest.approx(120.0)
    assert resolve_llm_guard_cache_ttl(None, False, SimpleNamespace(_custom_runner=None), 600.0) == pytest.approx(600.0)
    assert resolve_llm_guard_cache_ttl(None, True, SimpleNamespace(_custom_runner=None), 600.0) == pytest.approx(0.0)
    assert resolve_llm_guard_cache_ttl(None, False, SimpleNamespace(_custom_runner=lambda p: {}), 600.0) == pytest.approx(0.0)


def test_post_rth_equity_24_5_move_up_to_35pct_aligns_series_orderbook_and_pretrade(tmp_path):
    """Ensure >10% (<=35%) 24/5 equity moves (e.g. RGNT_US_EQ $1.27 -> $1.52) stay coherent across Section 3A, 3B, and pretrade telemetry."""

    now_s = 1_794_295_500.0
    rth_close_ms = int((now_s - 10 * 3600.0) * 1000)
    rgnt_bars = [
        [rth_close_ms - (24 - i) * 3600 * 1000, round(1.19 + (i % 6) * 0.07, 4), 1200.0]
        for i in range(24)
    ] + [[rth_close_ms, 1.27, 2500.0]]
    (tmp_path / "cache_24_price_RGNT_US_EQ.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": rgnt_bars}}),
        encoding="utf-8",
    )
    (tmp_path / "cache_current_price.json").write_text(
        json.dumps({"items": {"RGNT_US_EQ": [[rth_close_ms, 1.27]]}}),
        encoding="utf-8",
    )
    (tmp_path / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:t212:RGNT_US_EQ": {
                    "venue": "t212",
                    "symbol": "RGNT_US_EQ",
                    "free_base": 3.93,
                    "price": 1.52,
                    "ts": now_s - 20.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "orderbook_depth_RGNT_US_EQ.json").write_text(
        json.dumps(
            {
                "symbol": "RGNT_US_EQ",
                "imbalance_ratio": 0.45,
                "mid_price": 1.26,
                "best_bid": 1.255,
                "best_ask": 1.265,
                "spread_bps": 79.4,
                "bid_depth_usd": 2000.0,
                "ask_depth_usd": 3000.0,
                "largest_bid_wall_price": 1.22,
                "largest_bid_wall_usd": 510.0,
                "largest_ask_wall_price": 1.28,
                "largest_ask_wall_usd": 633.0,
            }
        ),
        encoding="utf-8",
    )

    pt_map = {"RGNT_US_EQ": {"price": 1.52, "trend": 0, "growth_coefficient": 0.0}}
    micro_map = {}
    enrich_24h_and_microstructure(
        str(tmp_path),
        str(tmp_path),
        pt_map,
        micro_map,
        {"RGNT_US_EQ": [[rth_close_ms, 1.27]]},
        {},
        now_s,
        allow_network=False,
    )
    assert pt_map["RGNT_US_EQ"]["price"] == pytest.approx(1.52)
    assert pt_map["RGNT_US_EQ"]["chg_24h_pct"] == pytest.approx(round(((1.52 / 1.19) - 1.0) * 100.0, 2))
    assert micro_map["RGNT_US_EQ"]["ob_mid_price"] == pytest.approx(1.52)
    assert micro_map["RGNT_US_EQ"]["ask_wall_px"] > 1.52

    tel_rgnt = collect_pretrade_telemetry("RGNT_US_EQ", cachedb_dir=str(tmp_path), allow_network=False)
    assert f"24h=+{round(((1.52 / 1.19) - 1.0) * 100.0, 2):.2f}%" in str(
        tel_rgnt.get("24h Multi-Horizon Price Evolution")
    )
    assert "1.5441" in str(tel_rgnt.get("Orderbook Ask Wall Ahead"))


def test_equity_suffix_ssot_primitives_and_zero_manual_endswith_ast():
    """Verify cachepaths equity-suffix SSOT helpers and lock zero manual _EQ/_US_EQ endswith checks."""

    assert has_equity_suffix("NVDA_US_EQ") is True
    assert has_equity_suffix("asml_eq") is True
    assert has_equity_suffix("NVDAUSEQ") is False
    assert has_equity_suffix("NVDAUSEQ", include_compact=True) is True
    assert has_equity_suffix("BTCUSDC", include_compact=True) is False
    assert has_equity_suffix(True) is False

    assert ensure_equity_suffix("nvda") == "NVDA_US_EQ"
    assert ensure_equity_suffix("NVDA_US_EQ") == "NVDA_US_EQ"
    assert ensure_equity_suffix("ASML_EQ") == "ASML_EQ"
    assert ensure_equity_suffix("BTCUSDC", is_equity=False) == "BTCUSDC"
    assert ensure_equity_suffix(False) == ""

    cands = ("NVDA", "NVDA_US_EQ", "NVDAUSEQ", "NVDAUSDC", "NVDAUSD")
    assert filter_asset_class_variants("NVDA_US_EQ", cands) == ("NVDA", "NVDA_US_EQ", "NVDAUSEQ")
    assert filter_asset_class_variants("NVDAUSDC", cands) == ("NVDA", "NVDAUSEQ", "NVDAUSDC", "NVDAUSD")
    assert filter_asset_class_variants("NVDAUSDC", cands, include_compact_equity=True) == (
        "NVDA",
        "NVDAUSDC",
        "NVDAUSD",
    )

    from verify_tools.ssot_ast_audit import audit_equity_suffix_endswith_ast

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    violations = audit_equity_suffix_endswith_ast(repo_root)
    assert violations == [], f"Manual _EQ/_US_EQ .endswith() found outside cachepaths.py: {violations}"


def test_ai_data_coherence_and_quota_fingerprint_fixes(tmp_path, monkeypatch):
    """Verify all 6 live AI prompt data-coherence fixes and instant OAuth token fingerprint cache invalidation."""
    import order_retry as _or_mod
    from intelligence.sentiment.collectors.portfolio_collector import BotRuntimeStatesCollector
    from orchestratorOS.supervisor.prompt_sections import (
        _format_microstructure_section,
        _format_tradable_position_line,
        _format_venue_cycle_line,
    )

    now_s = 1_794_500_000.0

    # Fix #1A: Equity synthetic orderbook ignores out-of-band historical bars and caps MOO/MOC cross spikes
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path))
    out_of_band_bars = [
        (now_s - (80 - i) * 300.0, 196.0, 197.0, 195.0, 196.5, 4_000_000.0)
        for i in range(60)
    ]
    in_band_bars = [
        (now_s - (15 - i) * 300.0, 206.6, 207.4, 206.1, 206.95, 1_200_000.0)
        for i in range(15)
    ]
    ob_col._fetch_equity_market_context = lambda sym, now, **kw: {
        "ticker": "NVDA",
        "last_price": 206.95,
        "best_bid": 206.93,
        "best_ask": 206.97,
        "bid_size_sh": 400.0,
        "ask_size_sh": 400.0,
        "daily_vol_sh": 35_000_000.0,
        "bars": out_of_band_bars + in_band_bars,
    }
    nvda_ob = ob_col._fetch_equity_orderbook("NVDA_US_EQ", now_s)
    assert nvda_ob is not None
    assert nvda_ob.largest_bid_wall_usd < 15_000_000.0
    assert nvda_ob.bid_depth_usd < 25_000_000.0
    assert abs(nvda_ob.largest_bid_wall_price - round(206.95 * 0.991, 2)) > 0.25

    # Fix #1B: Section 3B suppresses perp liq_clusters on _US_EQ even when Cboe options OI > 0
    sec_3b_lines = _format_microstructure_section(
        {
            "NVDA_US_EQ": {
                "ob_imbalance": 0.54,
                "bid_depth_usd": 4_200_000.0,
                "ask_depth_usd": 3_600_000.0,
                "bid_wall_px": 206.10,
                "bid_wall_usd": 1_100_000.0,
                "ask_wall_px": 207.40,
                "ask_wall_usd": 950_000.0,
                "whale_long_pct": 0.58,
                "taker_buy_sell_ratio": 1.12,
                "taker_buy_vol_usd": 8_500_000.0,
                "taker_sell_vol_usd": 7_590_000.0,
                "oi_usd": 285486813126.0,
                "oi_1h_chg_pct": 0.4,
                "long_liq_px": 195.0,
                "short_liq_px": 219.0,
            }
        },
        {"NVDA_US_EQ": {"price": 206.95}},
    )
    sec_3b_txt = "\n".join(sec_3b_lines)
    assert "oi_usd=285486813126.0 (cboe_options_oi)" in sec_3b_txt
    assert "liq_clusters(" not in sec_3b_txt

    # Fix #2 & #5B: AVCO vs remaining LIFO VWAP on NVDA_US_EQ and older_core_qty on BTCUSDC
    nvda_pos_line = _format_tradable_position_line(
        {
            "venue": "t212",
            "symbol": "NVDA_US_EQ",
            "base": "NVDA",
            "quote": "USD",
            "free_base": 0.41,
            "value_usd": 84.85,
            "free_quote": 1240.0,
            "price": 206.95,
            "active_avg_buy_px": 167.1951,
            "pnl_vs_avg_pct": 23.78,
            "open_lots": [
                {"lot_id": "N1", "qty": 0.38, "entry_px": 205.65},
                {"lot_id": "N2", "qty": 0.03, "entry_px": 207.93},
            ],
        }
    )
    assert "lifo_rem_avg_px=205.816829 (pnl_vs_lifo=+0.55%)" in nvda_pos_line
    assert "older_core_qty=" not in nvda_pos_line

    nvda_cyc_line = _format_venue_cycle_line(
        "t212",
        "NVDA_US_EQ",
        {
            "qty": 0.41,
            "cost_usd": 68.55,
            "avg_cost_px": 167.1951,
            "entry_price": 167.1951,
            "last_buy_price": 207.93,
            "dca_buys": 2,
            "pnl_vs_avg_pct": 23.78,
            "_spot_px": 206.95,
            "_open_lots": [
                {"qty": 0.38, "entry_px": 205.65},
                {"qty": 0.03, "entry_px": 207.93},
            ],
        },
    )
    assert "lifo_rem_avg=205.816829 (pnl_vs_lifo=+0.55%)" in nvda_cyc_line

    btc_pos_line = _format_tradable_position_line(
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "base": "BTC",
            "quote": "USDC",
            "free_base": 0.055436,
            "value_usd": 4102.26,
            "free_quote": 2500.0,
            "price": 74000.0,
            "active_avg_buy_px": 73255.12,
            "open_lots": [
                {"lot_id": "B1", "qty": 0.00696, "entry_px": 73415.99},
                {"lot_id": "B2", "qty": 0.00691, "entry_px": 73660.27},
            ],
        }
    )
    assert "older_core_qty=0.041566_BTC" in btc_pos_line

    # Fix #4: HighStakeGuard dynamic USDC/USD currency, equity persona, and clean grammar
    class _SpyPromptClient:
        def __init__(self):
            self.last_prompt = ""

        def query_json(self, prompt, **_kw):
            self.last_prompt = prompt
            return {"decision": "APPROVED", "suggested_scale": 1.0, "reason": "ok"}

    spy_hs = _SpyPromptClient()
    monkeypatch.setattr("intelligence.sentiment.guards.high_stake_guard._HIGH_STAKE_DECISION_CACHE", {})
    monkeypatch.setattr(
        "intelligence.sentiment.guards.high_stake_guard.query_topic_json",
        lambda client, topic_key, prompt, **kw: spy_hs.query_json(prompt),
    )
    hs_usd = HighStakeGuard(gemini_client=spy_hs, min_notional_eur=700.0, min_notional_usd=700.0)
    hs_usd.check("SOLUSDC", "BUY", price=143.51, qty=8.0, notional_eur=1148.09, telemetry={"Price": "$143.51"}, now=now_s)
    assert "$1,148.09 USDC" in spy_hs.last_prompt
    assert "$700.00 USDC" in spy_hs.last_prompt
    assert "€" not in spy_hs.last_prompt
    assert "evaluate if executing" not in spy_hs.last_prompt
    assert "you must evaluate whether this entry is sound" in spy_hs.last_prompt

    hs_usd.check("NVDA_US_EQ", "BUY", price=206.95, qty=8.0, notional_eur=1655.60, telemetry={"Price": "$206.95"}, now=now_s + 5.0)
    assert "principal quantitative equities and multi-asset risk officer" in spy_hs.last_prompt
    assert "$1,655.60 USD" in spy_hs.last_prompt

    # Fix #5A: kalman_primary_up fast impulse TTL = hours * 3600 (1080s for 0.3h) + zombie refused filter
    zombie_rec = {
        "id": "bcc26acf413f4a60a5bd7244312eec18",
        "venue": "binance",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 0.0068,
        "requested_price": 73897.15,
        "kind": "kalman_primary_up",
        "place_kwargs": {"hours": 0.3, "kind": "kalman_primary_up", "module": "tradeall"},
        "created_ts": now_s - 7200.0,
        "attempts": 200,
        "lifecycle": "submit_pending",
        "submission_state": "refused",
        "safe_to_discard": True,
        "last_failure_reason": "qty_zero_after_policy",
    }
    assert _or_mod.resolve_record_ttl(zombie_rec) == pytest.approx(1080.0)
    assert _or_mod.is_expired(zombie_rec, now=now_s) is True

    cdir_rq = tmp_path / "cachedb_rq"
    (cdir_rq / "orders").mkdir(parents=True, exist_ok=True)
    non_impulse_refused = dict(zombie_rec, id="ref2", kind="custom_grid", created_ts=now_s - 60.0)
    (cdir_rq / "orders" / "order_retry_queue.jsonl").write_text(
        json.dumps(zombie_rec) + "\n" + json.dumps(non_impulse_refused) + "\n",
        encoding="utf-8",
    )
    rt_states = BotRuntimeStatesCollector().collect(str(cdir_rq), now=now_s)
    assert "active_retry_queue" not in rt_states

    # Fix #6: Instant OAuth token fingerprint invalidation across account switch
    q_dir = tmp_path / "q_switch"
    q_dir.mkdir(parents=True, exist_ok=True)
    tok_file = q_dir / "antigravity-oauth-token"
    q_state_file = q_dir / "llm_quota_status.json"
    tok_file.write_text(
        json.dumps({"token": {"access_token": "acct-A-tok", "expiry_timestamp": int((time.time() + 3600) * 1000)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", staticmethod(lambda: str(q_state_file)))
    monkeypatch.setattr("intelligence.sentiment.llm_quota_router.resolve_bot_cli_base", lambda: str(q_dir))
    GeminiClient._QUOTA_MEM_CACHE = {}

    acct_tier = {"id": "free-tier", "frac": 0.0, "seen_tokens": []}

    def _mock_post(url, *args, **kwargs):
        hdrs = kwargs.get("headers") or {}
        acct_tier["seen_tokens"].append(hdrs.get("Authorization"))
        if "loadCodeAssist" in url:
            return {"currentTier": {"id": acct_tier["id"]}, "paidTier": {"id": acct_tier["id"], "availableCredits": []}}
        return {"buckets": [{"modelId": "gemini-3.8-flash-tiered", "remainingFraction": acct_tier["frac"]}]}

    monkeypatch.setattr("intelligence.sentiment.llm_quota_router.fetch_json_post", _mock_post)
    s_a = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=True)
    assert s_a is not None and s_a["tier_id"] == "free-tier"
    GeminiClient._record_model_quota_exhausted("gemini-4-neon-medium", ttl_sec=120.0)

    # Switch OAuth token WITHOUT force_refresh -> must bypass 30s cache in 0ms and clear old runtime_exhausted!
    tok_file.write_text(
        json.dumps({"token": {"access_token": "acct-B-ultra-tok", "expiry_timestamp": int((time.time() + 7200) * 1000)}}),
        encoding="utf-8",
    )
    acct_tier["id"] = "g1-ultra-tier"
    acct_tier["frac"] = 1.0
    s_b = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=False)
    assert s_b is not None
    assert s_b["tier_id"] == "g1-ultra-tier"
    cfg_def = os.environ["LLM_DEFAULT_MODEL"].strip()
    assert s_b["buckets"][cfg_def] == 1.0
    assert s_b["buckets"][f"{cfg_def.rsplit('-', 1)[0]}-xhigh"] == 1.0
    assert s_b["runtime_exhausted"] == {}
    assert acct_tier["seen_tokens"][-1] == "Bearer acct-B-ultra-tok"


def test_kinematic_velocity_acceleration_and_impulse_wakeup(tmp_path, monkeypatch):
    """Verify kinematic velocity/acceleration telemetry, unbiased prompt autonomy, and mid-cycle impulse wakeup."""
    import botcore
    from cachetrend import CacheShortTrendManager
    from intelligence.globaltelemetry_collector import (
        GlobalTelemetryCollector,
        evaluate_kinematic_impulse_symbols,
    )
    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
    from orchestratorOS.supervisor.autonomous_ai_reconciler import LogEventSummary
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_series_kinematics
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import _format_market_prices_section

    now_s = 1_794_500_120.0  # offset = 120s into 360s cron cycle (valid mid-cycle window)
    now_ms = int(now_s * 1000.0)

    # 1. Analytical quadratic kinematics on accelerating crash vs decelerating bottom vs flat series
    crash_series = []
    for k in range(31):
        tau_m = -5.0 + k * (5.0 / 30.0)  # -5.0 .. 0.0 minutes
        t_ms = now_ms + int(tau_m * 60_000.0)
        # Concave-down accelerating drop: y(tau) = 72000 * (1 - 0.0040*(tau+5)/5 - 0.0045*((tau+5)/5)^2)
        u = (tau_m + 5.0) / 5.0
        px = 72_000.0 * (1.0 - 0.0015 * u - 0.0055 * (u ** 2))
        crash_series.append([t_ms, round(px, 4)])

    kin_crash = compute_series_kinematics(crash_series)
    assert kin_crash["vel_90s_pct_min"] < -0.20
    assert kin_crash["accel_90s_pct_min2"] < -0.04
    assert kin_crash["vel_5m_pct_min"] < -0.20
    assert kin_crash["accel_5m_pct_min2"] < -0.04

    bottom_series = []
    for k in range(31):
        tau_m = -5.0 + k * (5.0 / 30.0)
        t_ms = now_ms + int(tau_m * 60_000.0)
        # Concave-up decelerating bottom / V-turn: minimum around tau=-1.2m, curving upward by tau=0
        px = 71_200.0 * (1.0 + 0.0005 * ((tau_m + 1.2) ** 2))
        bottom_series.append([t_ms, round(px, 4)])

    kin_bottom = compute_series_kinematics(bottom_series)
    assert kin_bottom["accel_90s_pct_min2"] > 0.05
    assert kin_bottom["accel_5m_pct_min2"] > 0.05

    flat_series = [[now_ms - (30 - k) * 10_000, 72_000.0] for k in range(31)]
    assert compute_series_kinematics(flat_series) == {}

    # 2. CacheShortTrendManager publishes vel_90s/accel_90s and vel_5m/accel_5m
    from types import SimpleNamespace
    import pricewindow as pw

    st_dir = tmp_path / "st_cache"
    st_dir.mkdir(parents=True, exist_ok=True)
    mgr = CacheShortTrendManager(
        symbols=["BTCUSDC"],
        filename=str(st_dir / "cache_short_trend.json"),
        data_sync_mode="producer",
    )
    mgr.windows["BTCUSDC"] = {
        sec: pw.PriceWindow("BTCUSDC", max(10, int(sec / 5.0)), sample_rate_sec=5.0)
        for sec in mgr.window_seconds
    }
    mgr.analyzers["BTCUSDC"] = {sec: pw.WindowAnalyzer(w) for sec, w in mgr.windows["BTCUSDC"].items()}
    last_px_obs = 72_500.0
    for k in range(40):
        t_obs = now_s - (39 - k) * 5.0
        u = k / 39.0
        last_px_obs = 72_500.0 * (1.0 - 0.0010 * u - 0.0050 * (u ** 2))
        for w_obj in mgr.windows["BTCUSDC"].values():
            w_obj.process_price(last_px_obs)
    mgr.current_price_mgr = SimpleNamespace(
        get_cached_price_observation=lambda _sym, _stale: (now_s, last_px_obs)
    )
    mgr.evaluate_full("BTCUSDC")
    st_item = mgr._mem.get("BTCUSDC") or {}
    assert st_item.get("vel_90s_pct_min") is not None and st_item["vel_90s_pct_min"] < 0.0
    assert st_item.get("accel_90s_pct_min2") is not None and st_item["accel_90s_pct_min2"] < 0.0

    # 3. Section 3A & Pre-trade Telemetry render kinematics(...) and kalman_z
    sec_3a = "\n".join(
        _format_market_prices_section(
            {
                "BTCUSDC": {
                    "price": 71_492.0,
                    "chg_5m_pct": -0.68,
                    "chg_1h_pct": -1.42,
                    "slope_90s": -0.00612,
                    "slope_15m": -0.00215,
                    "kalman_vel": -0.3412,
                    "kalman_vel_std": 0.115,
                    "vel_90s_pct_min": -0.4125,
                    "accel_90s_pct_min2": -0.2240,
                    "vel_5m_pct_min": -0.2650,
                    "accel_5m_pct_min2": -0.0910,
                }
            }
        )
    )
    assert "kinematics(vel_90s=-0.4125%/m, acc_90s=-0.2240%/m^2, vel_5m=-0.2650%/m, acc_5m=-0.0910%/m^2, kalman_z=-2.97)" in sec_3a

    pt_dir = tmp_path / "pt_cache"
    pt_dir.mkdir(parents=True, exist_ok=True)
    (pt_dir / "cache_24_price_BTCUSDC.json").write_text(json.dumps(crash_series), encoding="utf-8")
    (pt_dir / "cache_short_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": {
                        "current_price": crash_series[-1][1],
                        "ts": now_s,
                        "slopes": {"90": -0.0052, "222": -0.0041, "900": -0.0020},
                        "vel_90s_pct_min": kin_crash["vel_90s_pct_min"],
                        "accel_90s_pct_min2": kin_crash["accel_90s_pct_min2"],
                        "vel_5m_pct_min": kin_crash["vel_5m_pct_min"],
                        "accel_5m_pct_min2": kin_crash["accel_5m_pct_min2"],
                        "kalman_vel": -0.28,
                        "kalman_vel_std": 0.10,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pt_snap = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(pt_dir), now_ts=now_s, allow_network=False)
    assert "Kinematic Velocity & Acceleration" in pt_snap
    assert "90s: vel=-" in pt_snap["Kinematic Velocity & Acceleration"]
    assert "acc=-" in pt_snap["Kinematic Velocity & Acceleration"]
    assert "kalman_z=-2.80" in pt_snap["Kinematic Velocity & Acceleration"]

    # 4. Unbiased symmetric autonomy prompt check
    prompt_txt = build_supervisor_prompt(
        TelemetrySensorSnapshot(prices_and_trends={"BTCUSDC": {"price": 71_500.0}}),
        LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0),
        botcore.topic_spec("auto_cryptotrade_executor"),
        1.0,
    )
    assert "UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING" in prompt_txt
    assert "even at breakeven or below cost basis" in prompt_txt
    assert "kinematics(vel_90s, acc_90s, vel_5m, acc_5m, kalman_z)" in prompt_txt

    # 5. Event-driven kinematic impulse wakeup (symmetric drop/spike, equity filter, cooldown & cron guards)
    shocked_syms = evaluate_kinematic_impulse_symbols(
        {
            "BTCUSDC": {"vel_90s_pct_min": -0.31, "accel_90s_pct_min2": -0.19},
            "BTCUSDT": {"vel_90s_pct_min": -0.32, "accel_90s_pct_min2": -0.20},
            "TAOUSDC": {"vel_90s_pct_min": 0.29, "accel_90s_pct_min2": 0.18},  # symmetric upward breakout
            "ETHUSDC": {"vel_90s_pct_min": -0.06, "accel_90s_pct_min2": -0.02},  # normal noise -> ignored
            "NVDA_US_EQ": {"vel_90s_pct_min": -0.80, "accel_90s_pct_min2": -0.50},  # equity -> ignored
        }
    )
    assert shocked_syms == ["BTCUSDC", "TAOUSDC"]

    ws_dir = tmp_path / "ws"
    c_dir = ws_dir / "cachedb"
    (ws_dir / "orchestratorOS").mkdir(parents=True, exist_ok=True)
    runner_sh = ws_dir / "orchestratorOS" / "run_python.sh"
    runner_sh.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "cache_short_trend.json").write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": {
                        "current_price": 72_300.0,
                        "ts": now_s,
                        "vel_90s_pct_min": -0.38,
                        "accel_90s_pct_min2": -0.21,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_IMPULSE_TRIGGER_ENABLED", "true")
    monkeypatch.setenv("AI_IMPULSE_REQUIRE_ACTIVE_CRON", "false")
    monkeypatch.setenv("AI_IMPULSE_COOLDOWN_SEC", "150")
    orig_isfile = os.path.isfile
    monkeypatch.setattr(
        "os.path.isfile",
        lambda p: True if str(p) in {"/usr/bin/flock", "/usr/bin/timeout", str(runner_sh)} else orig_isfile(p),
    )
    spawned_calls = []

    class _DummyPopen:
        def __init__(self, args, **kwargs):
            spawned_calls.append((list(args), kwargs))

    monkeypatch.setattr("intelligence.globaltelemetry_collector.subprocess.Popen", _DummyPopen)
    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC", "ETHUSDC"], cache_dir=str(c_dir), skip_external=True, skip_news=True)

    # Near cron boundary (:05:30 -> 330s into 360s cycle, <45s before next cron) -> skipped
    assert gtc.maybe_trigger_kinematic_ai_wakeup(now=1_794_500_000.0 - (1_794_500_000.0 % 360.0) + 330.0) is None
    assert len(spawned_calls) == 0

    # Valid mid-cycle offset (120s into 360s cycle) -> spawns detached flock wakeup
    from cachepaths import resolve_cache_subdomain, resolve_cachedb_filepath

    assert resolve_cache_subdomain("kinematic_impulse_signal.json") == "telemetry"
    mid_ts = 1_794_500_000.0 - (1_794_500_000.0 % 360.0) + 120.0
    fired = gtc.maybe_trigger_kinematic_ai_wakeup(now=mid_ts)
    assert fired == ["BTCUSDC"]
    assert os.path.isfile(str(resolve_cachedb_filepath(str(c_dir / "kinematic_impulse_signal.json"))))
    assert len(spawned_calls) == 1
    assert "--focus-symbols" in spawned_calls[0][0]
    assert "BTCUSDC" in spawned_calls[0][0]

    # Immediate repeat within 150s cooldown -> skipped
    assert gtc.maybe_trigger_kinematic_ai_wakeup(now=mid_ts + 30.0) is None
    assert len(spawned_calls) == 1


def test_short_venue_cycle_pnl_symmetry_and_shared_fallback_recent_fills(tmp_path):
    """Verify SHORT cycle signed PnL, REDUCE classification, prompt formatting, and SSOT cycle-fallback synthesis."""
    ws_dir = tmp_path / "ws_short"
    hl_dir = ws_dir / "hyperliquid"
    c_dir = ws_dir / "cachedb"
    hl_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)

    # 1. Active SHORT cycle at avg=100, spot=90 -> +10% unrealized profit (not -10% long drawdown),
    #    and above-entry cover (last_close_price=102, realized_net=-15) -> last_exit_kind="REDUCE".
    (hl_dir / ".state_HYPE_SHORT.json").write_text(
        json.dumps(
            {
                "qty": 2.0,
                "cost": 200.0,
                "entry_price": 102.0,
                "last_open_price": 98.0,
                "last_close_price": 103.5,
                "dca_buys": 1,
                "realized_net": -15.0,
                "entry_ts": 1_700_000_000,
            }
        ),
        encoding="utf-8",
    )
    _krk_c, hl_cycles, _t212_c = harvest_venue_cycles(
        str(ws_dir),
        {"HYPEUSDC": {"price": 90.0}},
        {},
    )
    short_cyc = hl_cycles["HYPEUSDC"]
    assert short_cyc["direction"] == "SHORT"
    assert short_cyc["avg_cost_px"] == pytest.approx(100.0)
    assert short_cyc["last_buy_price"] == pytest.approx(98.0)
    assert short_cyc["pnl_vs_avg_pct"] == pytest.approx(10.0)
    assert short_cyc["dd_pct"] == pytest.approx(10.0)
    assert short_cyc["last_exit_kind"] == "REDUCE"

    fmt_short = _format_venue_cycle_line("hyperliquid", "HYPEUSDC", short_cyc)
    assert "direction=SHORT" in fmt_short
    assert "pnl_vs_avg=+10.00%" in fmt_short
    assert "pnl_vs_last_buy=+8.16%" in fmt_short

    # Wallet sync preserves SHORT sign when price rises above avg (110 vs 100 -> -10% drawdown).
    _sync_cycle_wallet_qty_and_avg(short_cyc, qty=2.0, avg_px=100.0, spot_px=110.0, avg_dec=6)
    assert short_cyc["pnl_vs_avg_pct"] == pytest.approx(-10.0)

    # 2. Shared apply_cycle_fallback_to_recent_fills matches both _populate_cycle_fallback_recent_order and load_symbol_best_recent_fills.
    t212_cyc = {
        "qty": 5.0,
        "avg_cost_px": 210.0,
        "entry_price": 210.0,
        "last_buy_price": 210.0,
        "last_sell_price": 225.0,
        "dca_buys": 0,
        "_entry_ts_ms": 1_700_000_000_000,
        "_last_sell_ts_ms": 1_700_003_600_000,
    }
    ro_map: dict = {}
    v_ro: dict = {}
    ro_eval = _populate_cycle_fallback_recent_order("t212", "NVDA_US_EQ", t212_cyc, True, None, ro_map, v_ro, {})
    ro_direct = apply_cycle_fallback_to_recent_fills("t212", t212_cyc, True, None, enrich_all_venues=True)
    assert ro_eval == ro_direct
    assert ro_eval is not None and ro_eval["active_avg_buy_px"] == pytest.approx(210.0)
    assert ro_eval["last_buy"]["qty"] == pytest.approx(5.0)
    assert ro_eval["last_sell"]["price"] == pytest.approx(225.0)
    best_hl = load_symbol_best_recent_fills(str(c_dir), "HYPEUSDC", ["HYPEUSDC", "HYPE"])
    assert best_hl is not None
    assert best_hl["venue"] == "hyperliquid"
    assert best_hl["active_avg_buy_px"] == pytest.approx(100.0)
    assert best_hl["last_buy"]["price"] == pytest.approx(98.0)


def test_24h_price_metrics_bisect_parity_and_series_memo_shallow_copy(tmp_path, monkeypatch):
    """Verify bisect-based compute_24h_price_metrics + TPO downsampling match legacy linear scans bit-for-bit, call _extract_sorted_ts_px once, and avoid inner sub-list reallocation on _PRICE_SERIES_MEMO hit."""
    import intelligence.sentiment.evaluators.price_series_evaluator as pse_mod
    import intelligence.macro_sentiment_ssot as ssot_mod

    def _legacy_compute_24h_and_tpo(series: list) -> dict:
        clean_series = ssot_mod._extract_sorted_ts_px(series)
        if len(clean_series) < 2:
            return {}
        now_ms, cur_px = clean_series[-1]
        span_ms = now_ms - clean_series[0][0]
        out = pse_mod.compute_horizon_range_dict(
            cur_px,
            round(min(x[1] for x in clean_series), 6),
            round(max(x[1] for x in clean_series), 6),
            "24h",
        )
        seen_sample_tms: set = set()
        dedup_sparse_samples = len(clean_series) >= 12
        for key, mins, min_span_ms, max_gap_mins in (
            ("chg_1m_pct", 1.0, 60000.0, 1.999),
            ("chg_3m_pct", 3.0, 180000.0, 1.999),
            ("chg_5m_pct", 5.0, 300000.0, 5.2),
            ("chg_15m_pct", 15.0, 0.0, 14.99 if dedup_sparse_samples else 0.0),
            ("chg_30m_pct", 30.0, 0.0, 29.99 if dedup_sparse_samples else 0.0),
            ("chg_1h_pct", 60.0, 0.0, 119.99 if dedup_sparse_samples else 0.0),
            ("chg_3h_pct", 180.0, 0.0, 179.99 if dedup_sparse_samples else 0.0),
            ("chg_6h_pct", 360.0, 3600000.0 * 4, 0.0),
            ("chg_12h_pct", 720.0, 3600000.0 * 9, 0.0),
            ("chg_24h_pct", 1440.0, 3600000.0 * 18, 0.0),
        ):
            tms, p_ref = next(
                ((t_v, p_v) for t_v, p_v in reversed(clean_series) if t_v <= now_ms - mins * 60000.0),
                (None, clean_series[0][1]),
            )
            valid_back = p_ref > 0 and span_ms >= min_span_ms and (
                max_gap_mins <= 0.0 or (tms is not None and tms >= now_ms - (mins + max_gap_mins) * 60000.0)
            )
            if valid_back and dedup_sparse_samples and tms is not None and mins < 1440.0 and tms in seen_sample_tms:
                valid_back = False
            if not valid_back and 15.0 <= mins <= 180.0 and span_ms >= max(min_span_ms, mins * 60000.0):
                target_ms = now_ms - mins * 60000.0
                nxt_t = next((t_f for t_f, _ in clean_series if tms is not None and t_f > tms), None)
                has_overnight_gap = (
                    tms is not None
                    and nxt_t is not None
                    and tms <= target_ms < nxt_t
                    and (nxt_t - tms) > 4.0 * 3600000.0
                )
                if mins <= 30.0 or has_overnight_gap:
                    max_fwd_ms = now_ms - (30.0 if mins == 60.0 else mins * 0.75) * 60000.0
                    fwd_pt = next(
                        ((t_f, p_f) for t_f, p_f in clean_series if target_ms < t_f <= max_fwd_ms and t_f not in seen_sample_tms),
                        None,
                    )
                    if fwd_pt is not None and fwd_pt[1] > 0:
                        tms, p_ref, valid_back = fwd_pt[0], fwd_pt[1], True
                    elif has_overnight_gap and p_ref > 0 and tms not in seen_sample_tms:
                        valid_back = True
            if valid_back:
                if tms is not None:
                    seen_sample_tms.add(tms)
                out[key] = pse_mod.pct_delta(cur_px, p_ref)
        recent_15m = [p_v for tms, p_v in clean_series if tms >= now_ms - 15.0 * 60000.0]
        if len(recent_15m) >= 2:
            eff_dec_15m = pse_mod._auto_px_decimals(cur_px)
            tape_pts: list = []
            tape_vals: list = []
            seen_tape_tms: set = set()
            for m_off, lbl in ((15.0, "-15m"), (10.0, "-10m"), (5.0, "-5m"), (3.0, "-3m"), (1.0, "-1m"), (0.0, "now")):
                if m_off == 0.0:
                    tape_pts.append(f"now:{f'{round(cur_px, eff_dec_15m):.{eff_dec_15m}f}'.rstrip('0').rstrip('.')}")
                    tape_vals.append(cur_px)
                    continue
                if span_ms < m_off * 60000.0:
                    continue
                max_tape_gap = 5.2 if m_off == 5.0 else (1.999 if m_off in (1.0, 3.0) else max(3.0, m_off * 0.5))
                t_tap, p_tap = next(
                    ((t_v, p_v) for t_v, p_v in reversed(clean_series) if t_v <= now_ms - m_off * 60000.0),
                    (None, clean_series[0][1]),
                )
                if t_tap is None or t_tap < now_ms - (m_off + max_tape_gap) * 60000.0 or t_tap in seen_tape_tms:
                    continue
                seen_tape_tms.add(t_tap)
                tape_pts.append(f"{lbl}:{f'{round(p_tap, eff_dec_15m):.{eff_dec_15m}f}'.rstrip('0').rstrip('.')}")
                tape_vals.append(p_tap)
            if len(recent_15m) >= 3 or len(tape_pts) >= 3:
                all_15m_vals = recent_15m + tape_vals if len(tape_pts) >= 3 else recent_15m
                l15, h15 = round(min(all_15m_vals), eff_dec_15m), round(max(all_15m_vals), eff_dec_15m)
                out["low_15m"], out["high_15m"] = l15, h15
                if h15 > l15 > 0:
                    out["pos_15m_pct"] = pse_mod.range_position_pct(cur_px, l15, h15)
                    out["span_15m_pct"] = pse_mod.pct_delta(h15, l15)
            if len(tape_pts) >= 3:
                out["recent_tape_15m"] = tape_pts
        if span_ms >= 3600000.0 * 2.0:
            win_6h = [(tms, p_v) for tms, p_v in clean_series if tms >= now_ms - 360.0 * 60000.0]
            cutoff_1h = now_ms - 60.0 * 60000.0
            prior_pts = [p_v for tms, p_v in win_6h if tms < cutoff_1h]
            recent_pts = [p_v for tms, p_v in win_6h if tms >= cutoff_1h]
            if len(prior_pts) >= 2 and len(recent_pts) >= 2:
                high_6h, prior_low, recent_low = max(p_v for _, p_v in win_6h), min(prior_pts), min(recent_pts)
                low_6h = min(prior_low, recent_low)
                dd_low_6h = ((low_6h / high_6h) - 1.0) * 100.0 if high_6h > 0.0 else 0.0
                pos_6h = ((cur_px - low_6h) / (high_6h - low_6h)) if high_6h > low_6h else 1.0
                dd_24h = out.get("dd_24h_high_pct") or 0.0
                pos_24h = out.get("pos_24h_pct") if out.get("pos_24h_pct") is not None else 100.0
                has_meaningful_bottom_dd = dd_low_6h <= -1.45 or (
                    dd_low_6h <= -0.85 and dd_24h <= -1.45 and pos_24h <= 35.0
                )
                if has_meaningful_bottom_dd and pos_6h <= 0.70 and prior_low > 0.0 and recent_low > 0.0:
                    eff_dec = pse_mod._auto_px_decimals(cur_px)
                    if recent_low < prior_low and cur_px > prior_low:
                        out["bottom_structure_6h"] = "LIQUIDITY_SWEEP_RECLAIM"
                        out["reclaim_above_prior_low_pct"] = pse_mod.pct_delta(cur_px, prior_low)
                    elif (
                        recent_low >= prior_low
                        and ((recent_low / prior_low) - 1.0) * 100.0 <= 4.5
                        and (
                            cur_px > recent_low * 1.002
                            or (recent_low > prior_low * 1.002 and cur_px >= recent_low)
                            or (recent_low > prior_low and cur_px > recent_low * 1.0005 and cur_px > prior_low * 1.001)
                        )
                    ):
                        out["bottom_structure_6h"] = "HIGHER_LOW_DOUBLE_BOTTOM"
                        out["higher_low_vs_prior_pct"] = pse_mod.pct_delta(recent_low, prior_low)
                    elif (
                        (recent_low <= prior_low and cur_px <= prior_low and ((cur_px / recent_low) - 1.0) * 100.0 <= 2.5)
                        or (
                            recent_low >= prior_low
                            and ((recent_low / prior_low) - 1.0) * 100.0 <= 0.20
                            and ((cur_px / prior_low) - 1.0) * 100.0 <= 0.25
                        )
                    ):
                        out["bottom_structure_6h"] = "FRESH_6H_LOW_TEST"
                    if "bottom_structure_6h" in out:
                        out["prior_6h_low_px"], out["recent_1h_low_px"] = round(prior_low, eff_dec), round(recent_low, eff_dec)
        # Force 3-col zero-vol path in compute_volume_profile_metrics to compare legacy full-array TPO downsampling
        legacy_vp = compute_volume_profile_metrics([[t_v, p_v, 0.0] for t_v, p_v in clean_series])
        out.update(legacy_vp)
        return out

    base_ms = 1_750_000_000_000.0
    # 1. Dense >4000-point 24h series with duplicate boundary timestamps and LIQUIDITY_SWEEP_RECLAIM
    dense_series = []
    for i in range(6017):
        t_ms = base_ms + i * 14_350.0
        px = 100.0 - (i / 6016.0) * 3.0 + math.sin(i * 0.13) * 0.45
        dense_series.append([t_ms, round(px, 4)])
        if i in (2500, 5950):
            dense_series.append([t_ms, round(px + 0.08, 4)])
    dense_series[-10][1] = 95.80
    dense_series[-1][1] = 96.85
    _faza1_additive_keys = {
        "atr_15m_pct",
        "atr_1h_pct",
        "atr_4h_pct",
        "session_vwap_px",
        "dist_vs_session_vwap_pct",
        "session_vwap_sigma_pct",
        "session_vwap_z",
        "session_vwap_anchor",
        "session_extremes",
    }

    def _strip_faza1(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in _faza1_additive_keys}

    assert _strip_faza1(compute_24h_price_metrics(dense_series)) == _legacy_compute_24h_and_tpo(dense_series)

    # 2. Sparse series with overnight gap (>4h) triggering forward-point fallback and HIGHER_LOW_DOUBLE_BOTTOM
    now_ms = base_ms + 24 * 3600_000.0
    sparse_overnight = [
        [now_ms - 24 * 3600_000.0, 210.0],
        [now_ms - 18 * 3600_000.0, 208.5],
        [now_ms - 12 * 3600_000.0, 207.0],
        [now_ms - 6.5 * 3600_000.0, 205.0],
        [now_ms - 5.0 * 3600_000.0, 206.0],
        [now_ms - 4.8 * 3600_000.0, 200.0],
        [now_ms - 52 * 60_000.0, 200.4],
        [now_ms - 45 * 60_000.0, 200.2],
        [now_ms - 26 * 60_000.0, 200.6],
        [now_ms - 14 * 60_000.0, 200.8],
        [now_ms - 9 * 60_000.0, 200.9],
        [now_ms - 4.5 * 60_000.0, 201.0],
        [now_ms - 2.5 * 60_000.0, 201.1],
        [now_ms - 0.8 * 60_000.0, 201.15],
        [now_ms, 201.2],
    ]
    res_sparse = compute_24h_price_metrics(sparse_overnight)
    assert _strip_faza1(res_sparse) == _legacy_compute_24h_and_tpo(sparse_overnight)
    assert res_sparse.get("bottom_structure_6h") == "HIGHER_LOW_DOUBLE_BOTTOM"

    # 3. _extract_sorted_ts_px is called only once inside compute_24h_price_metrics on 2-col series
    extract_calls = []
    real_extract = pse_mod._extract_sorted_ts_px

    def _counting_extract(ser):
        ssot_mod._SORTED_TS_PX_MEMO.clear()
        extract_calls.append(len(ser))
        return real_extract(ser)

    monkeypatch.setattr(pse_mod, "_extract_sorted_ts_px", _counting_extract)
    res_counted = compute_24h_price_metrics(dense_series)
    assert len(extract_calls) == 1
    assert _strip_faza1(res_counted) == _legacy_compute_24h_and_tpo(dense_series)
    monkeypatch.setattr(pse_mod, "_extract_sorted_ts_px", real_extract)

    # 4. load_symbol_price_series memo hit returns distinct outer list without reallocating inner row lists
    from cacheprice import Cache24PriceManager

    c_dir = tmp_path / "cachedb"
    p24_file = c_dir / Cache24PriceManager.disk_filename_for_symbol("BTCUSDC")
    p24_file.parent.mkdir(parents=True, exist_ok=True)
    live_now_ms = round(time.time() * 1000.0)
    p24_file.write_text(
        json.dumps(
            {
                "BTCUSDC": [
                    [live_now_ms - 600_000.0, 80000.0],
                    [live_now_ms - 300_000.0, 80050.0],
                    [live_now_ms - 60_000.0, 80100.0],
                ]
            }
        ),
        encoding="utf-8",
    )
    s1, _ = load_symbol_price_series(str(c_dir), "BTCUSDC", current_price_items={})
    s2, _ = load_symbol_price_series(str(c_dir), "BTCUSDC", current_price_items={})
    s3, _ = load_symbol_price_series(str(c_dir), "BTCUSDC", current_price_items={})
    assert isinstance(s2[-1], list)
    assert s2[-1] == [live_now_ms - 60_000.0, 80100.0]
    assert s1 is not s2 and s2 is not s3
    assert s1[0] is s2[0] and s2[0] is s3[0] and s2[-1] is s3[-1]
    assert ssot_mod._series_memo_key(s1) == ssot_mod._series_memo_key(s2)
    seeded_pts = ssot_mod._SORTED_TS_PX_MEMO.get(ssot_mod._series_memo_key(s2))
    assert seeded_pts is not None and id(seeded_pts) in ssot_mod._TWO_COL_TS_PX_IDS
    s2.append([live_now_ms, 80200.0])
    s4, _ = load_symbol_price_series(str(c_dir), "BTCUSDC", current_price_items={})
    assert len(s4) == 3

    # 5. Evicting _SORTED_TS_PX_MEMO >32 times never leaves dangling id() entries in _TWO_COL_TS_PX_IDS
    #    that could misclassify a subsequent 3-col real-volume series as TPO
    for k in range(50):
        ssot_mod._extract_sorted_ts_px(
            [[live_now_ms - (30 - i) * 60_000.0, 100.0 + i + k * 0.1] for i in range(21)]
        )
        s3_vol = [
            [live_now_ms - (30 - i) * 60_000.0, 1.30 + (i % 3) * 0.01, 500.0 + i]
            for i in range(21)
        ]
        assert compute_24h_price_metrics(s3_vol).get("vp_source") == "REAL_VOL"
    assert len(ssot_mod._SORTED_TS_PX_MEMO) <= ssot_mod._SORTED_TS_PX_MAX_ENTRIES
    assert len(ssot_mod._TWO_COL_TS_PX_IDS) <= ssot_mod._SORTED_TS_PX_MAX_ENTRIES

    # 6. 2-col series memos anchor only endpoint rows (never pinning full outer series lists),
    #    and _ARCH_FILE_MEMO / _OHLC_24H_DISK_MEMO overwrite in place on file mtime/size updates.
    import intelligence.sentiment.collectors.microstructure_collector as mc_mod

    transient_alt = [[live_now_ms - (20 - i) * 300_000.0, 200.0 + i] for i in range(21)]
    transient_btc = [[live_now_ms - (20 - i) * 300_000.0, 80_000.0 + i * 10.0] for i in range(21)]
    pse_mod.compute_24h_price_metrics(transient_alt)
    pse_mod.compute_cross_btc_metrics(transient_alt, transient_btc)
    alt_key = ssot_mod._series_memo_key(transient_alt)
    assert ssot_mod._SORTED_TS_PX_ANCHORS[alt_key] == (
        transient_alt[0],
        transient_alt[len(transient_alt) // 2],
        transient_alt[-1],
    )
    assert pse_mod._BUCKET_5M_MEMO[alt_key][0] == ssot_mod._SORTED_TS_PX_ANCHORS[alt_key]

    mc_mod._ARCH_FILE_MEMO.clear()
    arch_file = c_dir / "prices" / "cache_365_price_BTCUSDC.jsonl"
    arch_file.parent.mkdir(parents=True, exist_ok=True)
    arch_file.write_text(
        "\n".join(
            json.dumps({"s": "BTCUSDC", "i": [live_now_ms - (5 - i) * 60_000.0, 80_000.0 + i]})
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    rows_v1 = mc_mod._read_archive_series_file(
        str(arch_file), ["BTCUSDC"], {"BTCUSDC"}, ("BTCUSDC",), mc_mod.read_json_recovering, True
    )
    assert len(rows_v1) == 5 and len(mc_mod._ARCH_FILE_MEMO) == 1
    with arch_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"s": "BTCUSDC", "i": [live_now_ms + 60_000.0, 80_500.0]}) + "\n")
    rows_v2 = mc_mod._read_archive_series_file(
        str(arch_file), ["BTCUSDC"], {"BTCUSDC"}, ("BTCUSDC",), mc_mod.read_json_recovering, True
    )
    assert len(rows_v2) == 6 and rows_v2[-1][1] == 80_500.0
    assert len(mc_mod._ARCH_FILE_MEMO) == 1


def test_prompt_audit_15_full_telemetry_and_clarity_fixes(tmp_path, monkeypatch):
    """Regression suite verifying all 15 prompt telemetry, freshness, symmetry, and clarity fixes."""
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector
    from intelligence.sentiment.collectors.macro_telemetry_collector import MarketAndPortfolioSnapshotOrchestrator
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector, VenueCyclesCollector
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    import logging
    from providers.base import MarketDataProvider
    from providers.binance_provider import BinanceProvider
    from providers.kraken_provider import KrakenProvider

    # 1. BY DESIGN: Complete per-provider L2 isolation with ERROR logging and zero cross-venue book contamination
    MarketDataProvider._ORDERBOOK_PROV_ERR_TS.clear()
    err_records: list[str] = []
    vm_logger = logging.getLogger("providers.venue_metadata")
    err_handler = logging.Handler(level=logging.ERROR)
    err_handler.emit = lambda rec: err_records.append(rec.getMessage())  # type: ignore[method-assign]
    vm_logger.addHandler(err_handler)
    try:
        monkeypatch.setattr(
            KrakenProvider,
            "public_orderbook_depth_urls",
            classmethod(lambda cls, sym: (_ for _ in ()).throw(NameError("simulated broken import"))),
        )
        urls_btc, _ = MarketDataProvider.resolve_orderbook_depth_urls("BTCUSDC", venue="binance", multi_venue=True)
        assert any("api.binance.com/api/v3/depth?symbol=BTCUSDC" in u for u in urls_btc)
        assert not any("kraken.com" in u or "hyperliquid.xyz" in u for u in urls_btc)
        assert MarketDataProvider.resolve_orderbook_depth_urls("TAOUSD", venue="kraken", multi_venue=True) == ((), "")
        assert any("NameError: simulated broken import" in msg for msg in err_records)
        monkeypatch.setattr(
            BinanceProvider,
            "extract_orderbook_sides",
            staticmethod(lambda raw: (_ for _ in ()).throw(RuntimeError("broken extractor"))),
            raising=False,
        )
        norm_krk = MarketDataProvider.normalize_orderbook_payload(
            {"result": {"TAOUSD": {"bids": [["195.0", "2.0"]], "asks": [["195.2", "2.0"]]}}}
        )
        assert norm_krk is not None and norm_krk["bids"][0][0] == pytest.approx(195.0)
    finally:
        vm_logger.removeHandler(err_handler)
    monkeypatch.undo()

    # 2 & 14. Stale ob_mid (<1% away) NEVER overwrites 0.5s live price/tape_15m, and offline cache_ohlc_24h upgrades TPO -> REAL_VOL
    cdir = tmp_path / "cachedb"
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    now_s = 1_795_000_000.0
    now_ms = int(now_s * 1000)
    s24_pts = [[now_ms - (30 - i) * 60_000, 64_200.0 + i * 2.0] for i in range(30)] + [[now_ms - 500, 64_148.31]]
    (cdir / "prices" / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": s24_pts}}), encoding="utf-8"
    )
    ohlc_pts = [[now_ms - (24 - i) * 3_600_000, 64_000.0 + (i % 5) * 60.0, 120.0 + i * 10.0] for i in range(25)]
    ohlc_pts[-1][1] = 64_148.31
    (cdir / "prices" / "cache_ohlc_24h_BTCUSDC.json").write_text(
        json.dumps({"points": ohlc_pts}), encoding="utf-8"
    )
    (cdir / "telemetry" / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "ts": now_s - 352.4,
                "mid_price": 64_296.85,
                "best_bid": 64_296.80,
                "best_ask": 64_296.90,
                "imbalance_ratio": 0.55,
                "bid_depth_usd": 3_930_000.0,
                "ask_depth_usd": 2_980_000.0,
                "depth_1pct_bid_usd": 3_930_000.0,
                "depth_1pct_ask_usd": 2_980_000.0,
                "book_span_pct": 0.07,
            }
        ),
        encoding="utf-8",
    )
    cp_items = {"BTCUSDC": [[now_ms - 500, 64_148.31]]}
    pt_map = {"BTCUSDC": {"price": 64_148.31}}
    micro_map: dict = {}
    enrich_24h_and_microstructure(
        str(cdir), str(tmp_path), pt_map, micro_map, cp_items, {}, now_s, allow_network=False, persist=False
    )
    assert pt_map["BTCUSDC"]["price"] == pytest.approx(64_148.31)
    assert "64148.31" in str(pt_map["BTCUSDC"]["recent_tape_15m"][-1])
    assert pt_map["BTCUSDC"].get("vp_source") == "REAL_VOL"
    assert micro_map["BTCUSDC"].get("book_span_pct") == pytest.approx(0.07)

    # 3. Section 0 Snapshot Domain Ages reports orderbook_newest/oldest and flags [stale]
    (cdir / "cache_current_price.json").write_text(
        json.dumps({"timestamp": now_s - 0.5, "items": cp_items}), encoding="utf-8"
    )
    snap_orch = MarketAndPortfolioSnapshotOrchestrator().read_snapshot(
        str(cdir), workspace_dir=str(tmp_path), scope="crypto", now_ts=now_s
    )
    assert snap_orch.source_ages_sec.get("orderbook_newest") == pytest.approx(352.4, abs=1.0)
    prompt_s0 = build_supervisor_prompt(snap_orch, SimpleNamespace(total_scanned_files=0, error_count=0, warning_count=0, critical_events=[], guard_interventions=[]))
    assert "orderbook_newest=352.4s[stale]" in prompt_s0

    # 4 & 5 & 8 & 13. Tokenized xStocks skip Binance Futures, omit liq_clusters/None whale fields, tag spot_cvd/liq_clusters/book_window
    wc_tok = WhalePositioningCollector(cache_dir=str(cdir))
    dc_tok = DerivativesTelemetryCollector(cache_dir=str(cdir))
    assert wc_tok.fetch("SPCXXUSD", allow_network=True, now=now_s) is None
    assert dc_tok.fetch("ANTXUSD", allow_network=True, now=now_s) is None
    assert compute_liquidation_clusters({"price": 142.0}, {"bid_depth_usd": 50_000.0, "ask_depth_usd": 50_000.0}, symbol="SPCXxUSD") == {}

    ob_col_cap = OrderbookDepthCollector(cache_dir=str(cdir), cache_ttl_sec=0.0)
    monkeypatch.setattr(
        ob_col_cap,
        "_fetch_json_dict",
        lambda *a, **kw: {
            "bids": [[str(64200.0 - i * 0.3), "0.5"] for i in range(120)],
            "asks": [[str(64200.1 + i * 0.3), "0.5"] for i in range(120)],
        },
    )
    ob_snap_cap = ob_col_cap.fetch("BTCUSDC", allow_network=True, now=now_s)
    assert ob_snap_cap is not None and 0.0 < ob_snap_cap.book_span_pct < 0.20

    sec3b_txt = "\n".join(
        _format_microstructure_section(
            {
                "BTCUSDC": {
                    **micro_map["BTCUSDC"],
                    "whale_long_pct": 0.58,
                    "taker_buy_sell_ratio": 1.12,
                    "taker_buy_vol_usd": 500_000.0,
                    "taker_sell_vol_usd": 446_000.0,
                    "cvd_15m_usd": 54_000.0,
                    "oi_usd": 250_000_000.0,
                    "funding_rate": 0.0001,
                },
                "SPCXxUSD": {
                    "ob_imbalance": 0.52,
                    "bid_depth_usd": 25_000.0,
                    "ask_depth_usd": 23_000.0,
                    "bid_wall_px": 141.0,
                    "bid_wall_usd": 8_000.0,
                    "ask_wall_px": 143.0,
                    "ask_wall_usd": 7_500.0,
                    "ob_mid_price": 142.0,
                    "spread_bps": 14.0,
                },
            },
            {"BTCUSDC": pt_map["BTCUSDC"], "SPCXxUSD": {"price": 142.0, "chg_15m_pct": 0.2}},
            now_ts=now_s,
        )
    )
    assert "book_window=+/-0.07%[level_cap]" in sec3b_txt
    assert "[est:ob+15m]" in sec3b_txt
    assert "src=model_est)" in sec3b_txt
    spcx_line_3b = next(ln for ln in sec3b_txt.splitlines() if "SPCXxUSD" in ln)
    assert "perp_derivatives=N/A[spot_tokenized_equity]" in spcx_line_3b
    assert "whale_long_pct=None" not in spcx_line_3b
    assert "liq_clusters(" not in spcx_line_3b

    real_sp = {"cvd_15m_usd": 50_000.0, "spot_cvd_15m_usd": 19_500.0, "spot_cvd_source": "real_spot", "bid_depth_usd": 100_000.0, "ask_depth_usd": 100_000.0}
    enrich_microstructure_cvd(real_sp, {"price": 100.0, "chg_15m_pct": -1.5})
    assert real_sp["spot_cvd_15m_usd"] == pytest.approx(19_500.0)

    # 6. KrakenXstocksCollector separates SPCXx.T holding from XSTOCK_YAHOO=ANT (ignoring dust AAPLx.T and normalizing SPCXXUSD)
    krk_dir = tmp_path / "kraken"
    krk_dir.mkdir(parents=True, exist_ok=True)
    (krk_dir / "config.env").write_text("XSTOCK_YAHOO=ANT\nXSTOCK_ALLOC_PRICE=141.75\n", encoding="utf-8")
    (krk_dir / "xstock_state.json").write_text(
        json.dumps({"known_assets": {"AAPLx.T": 2e-06, "SPCXx.T": 15.74655}}), encoding="utf-8"
    )
    xs_out = KrakenXstocksCollector().collect(
        str(cdir), kraken_dir=str(krk_dir), prices_and_trends={"SPCXxUSD": {"price": 141.97}}
    )
    assert xs_out["underlying"] == "SPCX"
    assert xs_out["t212_sibling"] == "SPCX_US_EQ"
    assert xs_out["holdings"] == {"SPCXx.T": 15.74655}
    assert "holdings_underlying_map" not in xs_out
    assert xs_out["watched_ipo_underlying"] == "ANT"
    assert xs_out["live_price"] == pytest.approx(141.97)
    (krk_dir / "xstock_state.json").write_text(
        json.dumps({"known_assets": {"SPCXx.T": 15.74655}, "pair": "SPCXXUSD"}), encoding="utf-8"
    )
    xs_pair_out = KrakenXstocksCollector().collect(
        str(cdir), kraken_dir=str(krk_dir), prices_and_trends={"SPCXxUSD": {"price": 141.97}}
    )
    assert xs_pair_out["underlying"] == "SPCX"
    assert xs_pair_out["listed_pair"] == "SPCXxUSD"

    asym_3b_txt = "\n".join(
        _format_microstructure_section(
            {
                "BTCUSDC": {
                    "ob_imbalance": 0.62,
                    "bid_depth_usd": 3_158_263.0,
                    "ask_depth_usd": 1_919_767.0,
                    "depth_1pct_bid_usd": 3_158_263.0,
                    "depth_1pct_ask_usd": 1_917_693.0,
                    "book_span_pct": 0.495,
                    "est_slip_2k_bps": 0.5,
                }
            },
            {"BTCUSDC": {"price": 63_885.6}},
            now_ts=now_s,
        )
    )
    assert "book_window=+/-0.495%[level_cap]" in asym_3b_txt

    # BY DESIGN: Tokenized xStock depth URLs resolve exclusively to Kraken tokenized_asset (never Binance Spot/Futures or Hyperliquid)
    spcx_depth_urls, spcx_hl_url = MarketDataProvider.resolve_orderbook_depth_urls("SPCXXUSD")
    assert spcx_hl_url == ""
    assert all("api.kraken.com" in u and "asset_class=tokenized_asset" in u for u in spcx_depth_urls)
    assert not any("binance.com" in u or "hyperliquid.xyz" in u for u in spcx_depth_urls)

    # 8b. BY DESIGN: fill_prob_1h_pct barrier-touch calibration separates near (+0.8%), multi-sigma (+1.76%), and far-OTM (+22.6%)
    lim_orders = [
        {"venue": "binance", "symbol": "TAOUSDC", "side": "SELL", "price": 197.50, "dist_from_spot_pct": 0.80},
        {"venue": "binance", "symbol": "TAOUSDC", "side": "SELL", "price": 199.40, "dist_from_spot_pct": 1.76},
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "side": "SELL", "price": 84.45, "dist_from_spot_pct": 22.6},
    ]
    enrich_open_limit_orders_tca(
        lim_orders,
        {"TAOUSDC": {"price": 195.95, "vol_1h_pct": 0.45, "span_24h_pct": 4.2}, "HYPEUSDC": {"price": 68.88, "vol_1h_pct": 0.8, "span_24h_pct": 5.0}},
        {},
    )
    assert lim_orders[0]["fill_prob_1h_pct"] > 12.0
    assert 1.0 < lim_orders[1]["fill_prob_1h_pct"] < 10.0
    assert lim_orders[2]["fill_prob_1h_pct"] <= 0.2

    # 9, 10, 11, 12. BY DESIGN: Preserve prior-cycle SL across cycle boundary (last_sell_ts < entry_ts), upgrade in-cycle sell SL -> TP,
    # preserve both zero-sell and active-subcycle cumulative PnL, ignore PAPER-* state, and keep crypto scope US_EQ bellwethers compact
    hl_dir = tmp_path / "hyperliquid"
    hl_dir.mkdir(parents=True, exist_ok=True)
    (hl_dir / ".state_HYPE_long.json").write_text(
        json.dumps({"qty": 0.85, "cost": 50.19, "orders": [{"oid": "PAPER-2", "side": "sell", "px": 60.0, "sz": 0.85}]}),
        encoding="utf-8",
    )
    _, hl_cycs_paper, _ = VenueCyclesCollector().collect(str(cdir), workspace_dir=str(tmp_path))
    assert "HYPEUSDC" not in hl_cycs_paper

    pos_sym = [
        {"venue": "kraken", "symbol": "HYPEUSD", "free_base": 26.32, "price": 67.92, "last_sell_px": 68.900061, "lifo_realized_pnl_usd": 0.0},
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "free_base": 4.0, "price": 68.80, "last_sell_px": 69.20, "lifo_realized_pnl_usd": 5.31},
    ]
    krk_cycs = {"HYPEUSD": {"qty": 26.32, "cost_usd": 1808.27, "avg_cost_px": 68.69, "entry_price": 68.69, "last_sell_price": 68.67, "last_exit_kind": "SL", "realized_pnl_usd": -189.93}}
    hl_cycs = {"HYPEUSDC": {"qty": 4.0, "cost_usd": 274.0, "avg_cost_px": 68.50, "entry_price": 68.50, "last_sell_price": 68.40, "last_exit_kind": "SL", "realized_pnl_usd": 179.25}}
    enrich_positions_with_orders_and_cycles(
        pos_sym,
        {"HYPEUSD": {"price": 67.92}, "HYPEUSDC": {"price": 68.80}},
        {},
        {
            "kraken": {"HYPEUSD": {"lifo_realized_pnl_usd": 0.0, "last_sell": {"price": 68.900061, "ts_ms": 1_794_627_083_000}, "last_buy": {"price": 68.69, "ts_ms": 1_794_631_269_000}}},
            "hyperliquid": {"HYPEUSDC": {"lifo_realized_pnl_usd": 5.31, "last_sell": {"price": 69.20, "ts_ms": 1_794_700_000_000}, "last_buy": {"price": 68.50, "ts_ms": 1_794_600_000_000}}},
        },
        {},
        krk_cycs,
        hl_cycs,
        {},
        {("kraken:entry", "HYPEUSD"): 1_794_631_269_000, ("hyperliquid:entry", "HYPEUSDC"): 1_794_600_000_000},
    )
    assert krk_cycs["HYPEUSD"]["last_exit_kind"] == "SL"
    assert hl_cycs["HYPEUSDC"]["last_exit_kind"] == "TP"
    assert krk_cycs["HYPEUSD"]["realized_pnl_usd"] == pytest.approx(-189.93)
    assert hl_cycs["HYPEUSDC"]["realized_pnl_usd"] == pytest.approx(5.31)
    assert hl_cycs["HYPEUSDC"]["cum_realized_pnl_usd"] == pytest.approx(179.25)

    cmc_payload = {
        "equity_bellwethers": {
            "NVDA_US_EQ": {
                "price": 160.8513,
                "quote_source": "yahoo_ext",
                "quote_age_min": 1.8,
                "t212_close_px": 155.15,
                "chg_24h_pct": 3.67,
                "t212_cycle": {"qty": 3.0, "avg_cost_px": 150.0, "pnl_vs_avg_pct": 7.23},
            },
            "RGNT_US_EQ": {"price": 1.48, "quote_source": "t212_rth_close", "quote_age_min": 930.0, "chg_24h_pct": -1.2},
        }
    }
    sec3c_out = "\n".join(
        _format_portfolio_section(
            {
                "portfolio_risk_engine": {
                    "unified_nav_usd": 16107.24,
                    "invested_exposure_usd": 13190.0,
                    "global_free_cash_usd": 2917.24,
                    "global_free_cash_pct": 18.1,
                    "venue_nav_breakdown_usd": {"binance": 11000.0, "kraken": 3107.24, "hyperliquid": 2000.0},
                    "portfolio_btc_beta": 0.85,
                    "avg_corr_vs_btc_24h": 0.88,
                    "intraday_var_95_usd": 420.0,
                    "intraday_var_95_pct": 2.61,
                },
                "tradable_positions": pos_sym,
                "kraken_cycles": krk_cycs,
                "hyperliquid_cycles": hl_cycs,
                "cross_market_context": cmc_payload,
            },
            asset_scope="crypto",
            include_macro_benchmarks=False,
        )
    )
    assert "Portfolio Risk Engine (Crypto Scope: binance + kraken + hyperliquid; excl. T212 equities): crypto_nav=$16,107.24" in sec3c_out
    assert "realized_pnl=$-189.93 (subcycle=$0.00)" in sec3c_out
    assert "realized_pnl=$5.31 (cum_cycles=$179.25)" in sec3c_out
    assert "last_sell=68.900061 (SL)" in sec3c_out
    assert "'quote_source': 'yahoo_ext'" in sec3c_out
    assert "'t212_close_px': 155.15" in sec3c_out
    assert "'RGNT_US_EQ'" not in sec3c_out
    assert "'t212_cycle'" not in sec3c_out

    sec3c_all = "\n".join(
        _format_portfolio_section({"cross_market_context": cmc_payload}, asset_scope="all", include_macro_benchmarks=False)
    )
    assert "'RGNT_US_EQ'" in sec3c_all
    assert "'t212_cycle'" in sec3c_all


def test_item_1_2_ob_freshness_ssot_and_no_sub_1pct_wall_rescaling(tmp_path):
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        is_ob_live_and_newer,
        is_ob_mid_fresher_than_series,
    )
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from state_io import atomic_write_json

    now_s = 1_794_680_000.0
    # 1. SSOT freshness bounds: >60s old or older than live 0.5s price tick must be rejected.
    assert is_ob_live_and_newer(now_s - 10.0, now_s, (int((now_s - 15.0) * 1000), 67.08), has_live_cp_point=True) is True
    assert is_ob_live_and_newer(now_s - 10.0, now_s, (int((now_s - 0.5) * 1000), 67.08), has_live_cp_point=True) is False
    assert is_ob_live_and_newer(now_s - 95.0, now_s, (int((now_s - 180.0) * 1000), 67.08), has_live_cp_point=True) is False
    assert (
        is_ob_mid_fresher_than_series(
            67.08, 67.045, now_s - 180.0, now_s, (int((now_s - 300.0) * 1000), 67.08), is_exact_ob_sym=True
        )
        is False
    )

    # 2. Normal <=1% intraday drift between 0.5s spot ($67.08) and resting Kraken L2 mid ($67.045)
    #    must neither overwrite live spot nor mutate real 2-decimal resting L2 wall prices.
    cdir = tmp_path / "cachedb"
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    cp_items = {"HYPEUSD": [[int((now_s - 0.5) * 1000), 67.08]]}
    atomic_write_json(
        str(cdir / "prices" / "cache_current_price.json"),
        {"items": cp_items},
    )
    atomic_write_json(
        str(cdir / "prices" / "cache_24trend_HYPEUSD.json"),
        {"series": [[int((now_s - 900.0) * 1000), 66.80], [int((now_s - 5.0) * 1000), 67.06]]},
    )
    atomic_write_json(
        str(cdir / "telemetry" / "orderbook_depth_HYPEUSD.json"),
        {
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "timestamp": now_s - 12.0,
            "mid_price": 67.045,
            "best_bid": 67.04,
            "best_ask": 67.05,
            "bid_wall_px": 66.59,
            "ask_wall_px": 67.74,
            "bid_wall_usd": 14500.0,
            "ask_wall_usd": 18200.0,
            "bid_depth_usd": 52000.0,
            "ask_depth_usd": 55000.0,
            "imbalance": -0.028,
        },
    )
    pt_map = {"HYPEUSD": {"price": 67.08}}
    micro_map: dict = {}
    enrich_24h_and_microstructure(
        str(cdir),
        str(tmp_path),
        pt_map,
        micro_map,
        cp_items,
        {},
        now_s,
        allow_network=False,
        persist=False,
    )
    assert pt_map["HYPEUSD"]["price"] == pytest.approx(67.08)
    assert micro_map["HYPEUSD"]["ob_mid_price"] == pytest.approx(67.045)
    assert micro_map["HYPEUSD"]["bid_wall_px"] == pytest.approx(66.59)
    assert micro_map["HYPEUSD"]["ask_wall_px"] == pytest.approx(67.74)


def test_item_1_3_and_ssot_unification_contracts(tmp_path):
    import copy
    from types import SimpleNamespace
    from intelligence.macro_sentiment_ssot import (
        TelemetrySensorSnapshot,
        format_telemetry_domain_age_tag,
        resolve_telemetry_domain_stale_thresholds_sec,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import MarketAndPortfolioSnapshotOrchestrator
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        select_reconciled_spot_or_ob_price,
        should_rescale_usd_sibling_walls,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        classify_reconciled_cycle_exit_kind,
        compute_cycle_subcycle_realized_pnl_view,
    )
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from state_io import atomic_write_json

    # 1. Item 1.3: Symmetric Section 0 [stale] + 60s crypto oldest jitter tolerance + RTH [market_closed] on equities
    crypto_thrs = resolve_telemetry_domain_stale_thresholds_sec("crypto", executor_interval_sec=360.0)
    eq_thrs = resolve_telemetry_domain_stale_thresholds_sec("equities", executor_interval_sec=360.0)
    assert crypto_thrs["orderbook_newest"] == pytest.approx(30.0)
    assert crypto_thrs["orderbook_oldest"] == pytest.approx(60.0)
    assert eq_thrs["orderbook_oldest"] == pytest.approx(120.0)
    assert format_telemetry_domain_age_tag("orderbook_oldest", 37.1, asset_scope="crypto", thresholds=crypto_thrs) == ""
    assert format_telemetry_domain_age_tag("orderbook_oldest", 65.0, asset_scope="crypto", thresholds=crypto_thrs) == "[stale]"
    assert format_telemetry_domain_age_tag("prices_oldest", 180.0, asset_scope="crypto", thresholds=crypto_thrs) == "[stale]"
    assert format_telemetry_domain_age_tag("short_trends", 650.0, asset_scope="crypto", thresholds=crypto_thrs) == "[stale]"
    assert format_telemetry_domain_age_tag("orderbook_oldest", 37.1, asset_scope="equities", us_rth_open=True, thresholds=eq_thrs) == ""
    assert (
        format_telemetry_domain_age_tag("orderbook_newest", 54000.0, asset_scope="equities", us_rth_open=False, thresholds=eq_thrs)
        == "[market_closed]"
    )
    assert (
        format_telemetry_domain_age_tag("geopolitical_threat", 40000.0, asset_scope="equities", us_rth_open=False, thresholds=eq_thrs)
        == "[stale]"
    )

    # Sunday UTC (outside US RTH): 2026-11-15 12:00:00 UTC = 1794744000
    sun_ts = 1_794_744_000.0
    eq_snap = TelemetrySensorSnapshot(
        evaluated_at_ts=sun_ts,
        source_ages_sec={
            "prices_newest": 54000.0,
            "orderbook_newest": 54000.0,
            "venue_balances": 54000.0,
            "geopolitical_threat": 40000.0,
        },
    )
    eq_prompt = build_supervisor_prompt(eq_snap, [], topic_key="auto_sharetrade_executor")
    assert "prices_newest=54000.0s[market_closed]" in eq_prompt
    assert "orderbook_newest=54000.0s[market_closed]" in eq_prompt
    assert "venue_balances=54000.0s[market_closed]" in eq_prompt
    assert "geopolitical_threat=40000.0s[stale]" in eq_prompt

    # Active-fleet filtering & casefold deduplication in MarketAndPortfolioSnapshotOrchestrator
    cdir = tmp_path / "cachedb"
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "state").mkdir(parents=True, exist_ok=True)
    now_s = 1_794_680_000.0
    atomic_write_json(
        str(cdir / "prices" / "cache_current_price.json"),
        {"items": {"SPCXxUSD": [[int((now_s - 1.0) * 1000), 142.5]], "UNHELDUSDC": [[int((now_s - 2.0) * 1000), 1.2]]}},
    )
    atomic_write_json(
        str(cdir / "telemetry" / "orderbook_depth_SPCXxUSD.json"),
        {"symbol": "SPCXxUSD", "venue": "kraken", "timestamp": now_s - 8.0, "mid_price": 142.5, "best_bid": 142.4, "best_ask": 142.6},
    )
    atomic_write_json(
        str(cdir / "telemetry" / "orderbook_depth_UNHELDUSDC.json"),
        {"symbol": "UNHELDUSDC", "venue": "binance", "timestamp": now_s - 900.0, "mid_price": 1.2, "best_bid": 1.19, "best_ask": 1.21},
    )
    atomic_write_json(
        str(cdir / "state" / "venue_balances_state.json"),
        {
            "open_orders_verified_ts:kraken": now_s - 5.0,
            "pos:kraken:SPCXxUSD": {"venue": "kraken", "symbol": "SPCXxUSD", "free_base": 2.5, "price": 142.5, "ts": now_s - 5.0},
        },
    )
    snap_pre = TelemetrySensorSnapshot(
        microstructure={
            "SPCXxUSD": {"ob_mid_price": 142.5, "_ob_raw_ts": now_s - 8.0},
            "SPCXXUSD": {"ob_mid_price": 142.5, "_ob_raw_ts": now_s - 8.0},
            "UNHELDUSDC": {"ob_mid_price": 1.2, "_ob_raw_ts": now_s - 900.0},
        },
    )
    orch = MarketAndPortfolioSnapshotOrchestrator(start_sync=False)
    # With a single active Kraken holding (SPCXxUSD, stored under 2 case keys) + stale unheld reference UNHELDUSDC:
    # UNHELDUSDC must be excluded from orderbook_oldest, and SPCXxUSD/SPCXXUSD deduplicated to 1 active symbol.
    res_snap = orch.collect(
        str(cdir),
        "*",
        snapshot=snap_pre,
        workspace_dir=str(tmp_path),
        scope="crypto",
        now_ts=now_s,
        allow_network=False,
        persist=False,
    )
    assert res_snap.source_ages_sec["orderbook_newest"] == pytest.approx(8.0)
    assert "orderbook_oldest" not in res_snap.source_ages_sec

    # 2. Unification 1 & 2: Pure SSOT compute + Single-owner non-mutating formatter contract
    assert (
        select_reconciled_spot_or_ob_price(
            67.08, 67.045, now_s - 12.0, now_s, (int((now_s - 0.5) * 1000), 67.08), is_exact_ob_sym=True, has_live_cp_point=True
        )
        is None
    )
    assert should_rescale_usd_sibling_walls("HYPEUSD", 67.08, 67.045, 67.08) is False
    assert classify_reconciled_cycle_exit_kind("SL", 68.900061, 68.69, sell_from_prior_cycle=True, allow_stop_upgrade=True) is None
    assert classify_reconciled_cycle_exit_kind("SL", 68.900061, 68.69, sell_from_prior_cycle=False, allow_stop_upgrade=True) == "TP"

    ps_in = {
        "tradable_positions": [{"venue": "hyperliquid", "symbol": "HYPEUSDC", "price": 67.10, "lifo_realized_pnl_usd": 5.31}],
        "hyperliquid_cycles": {"HYPEUSDC": {"qty": 10.0, "avg_cost_px": 66.0, "realized_pnl_usd": 179.25, "last_exit_kind": "TP"}},
    }
    ps_before = copy.deepcopy(ps_in)
    rendered_3c = "\n".join(_format_portfolio_section(ps_in, asset_scope="crypto", include_macro_benchmarks=False))
    assert ps_in == ps_before
    assert "realized_pnl=$5.31 (cum_cycles=$179.25)" in rendered_3c
    view_d = compute_cycle_subcycle_realized_pnl_view(ps_in["hyperliquid_cycles"]["HYPEUSDC"], 5.31)
    assert view_d["realized_pnl_usd"] == pytest.approx(5.31)
    assert view_d["cum_realized_pnl_usd"] == pytest.approx(179.25)


def test_item_2_1_tokenized_xstock_base_isolation_no_retry_spin_and_pretrade_tag(tmp_path):
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_microstructure
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from state_io import atomic_write_json

    now_s = 1_795_018_000.0
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)

    # Even if stale/base-variant disk files exist for SPCX / SPCXXUSD, tokenized xStock collectors and loaders must return None / ignore them.
    for pfx in ("whale_snapshot_", "derivatives_telemetry_"):
        for s_bad in ("SPCX", "SPCXXUSD", "SPCXxUSD"):
            atomic_write_json(
                str(cdir / "telemetry" / f"{pfx}{s_bad}.json"),
                {
                    "symbol": s_bad,
                    "ts": now_s - 5.0,
                    "funding_rate": 0.0002,
                    "open_interest_usd": 999_999.0,
                    "top_traders_long_pct": 0.65,
                    "taker_buy_sell_ratio": 1.35,
                },
            )
    atomic_write_json(
        str(cdir / "telemetry" / "orderbook_depth_SPCXxUSD.json"),
        {
            "symbol": "SPCXxUSD",
            "venue": "kraken",
            "ts": now_s - 3.0,
            "mid_price": 144.60,
            "best_bid": 144.58,
            "best_ask": 144.62,
            "bid_depth_usd": 50000.0,
            "ask_depth_usd": 60000.0,
            "imbalance_ratio": 0.455,
            "largest_bid_wall_price": 144.20,
            "largest_bid_wall_usd": 20000.0,
            "largest_ask_wall_price": 145.10,
            "largest_ask_wall_usd": 25000.0,
        },
    )
    atomic_write_json(
        str(cdir / "prices" / "cache_current_price.json"),
        {"items": {"SPCXxUSD": [[int((now_s - 1.0) * 1000), 144.60]]}},
    )

    wc = WhalePositioningCollector(cache_dir=str(cdir), start_sync=False)
    dc = DerivativesTelemetryCollector(cache_dir=str(cdir), start_sync=False)
    assert wc.fetch("SPCXxUSD", allow_network=False, now=now_s) is None
    assert dc.fetch("SPCXXUSD", allow_network=False, now=now_s) is None

    m_tok = load_symbol_microstructure(str(cdir), "SPCXxUSD")
    assert m_tok is not None
    assert m_tok.get("ob_mid_price") == pytest.approx(144.60)
    assert m_tok.get("whale_long_pct") is None
    assert m_tok.get("funding_rate") is None
    assert m_tok.get("oi_usd") is None

    pt_tok = collect_pretrade_telemetry("SPCXxUSD", cachedb_dir=str(cdir), allow_network=False, now_ts=now_s)
    assert pt_tok.get("Perpetual Derivatives") == "N/A (spot tokenized equity)"
    assert "Perpetual Funding Rate" not in pt_tok
    assert "Derivatives Open Interest" not in pt_tok
    assert "Whale Positioning" not in pt_tok

    # GlobalTelemetryCollector._step_external_symbols skips deriv/whale on tokenized xStocks instead of 0.25x retry spinning.
    gtc = GlobalTelemetryCollector(symbols=["SPCXxUSD"], cache_dir=str(cdir), skip_news=True)
    deriv_calls, whale_calls, ob_calls = [], [], []
    gtc.update_orderbook = lambda sym: (ob_calls.append(sym) or True)
    gtc.update_derivatives = lambda sym: (deriv_calls.append(sym) or False)
    gtc.update_whale = lambda sym: (whale_calls.append(sym) or False)
    res_step = gtc._new_external_cycle_results(now_s)
    gtc._step_external_symbols(now_s, False, res_step, bounded=True)
    assert ob_calls == ["SPCXxUSD"]
    assert deriv_calls == []
    assert whale_calls == []
    assert "SPCXxUSD" not in gtc._last_derivatives_ts
    assert "SPCXxUSD" not in gtc._last_whale_ts


def test_item_2_2_no_synthetic_oi_mult_12_and_compact_equity_liq_isolation():
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        compute_liquidation_clusters,
        sync_sibling_symbol_telemetry,
    )
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section

    # 1. Spot-only / missing-OI crypto pairs never fabricate synthetic OI = (bid_depth + ask_depth) * 12.
    assert (
        compute_liquidation_clusters(
            {"price": 0.15, "vol_1h_pct": 1.0},
            {"bid_depth_usd": 50_000.0, "ask_depth_usd": 50_000.0, "oi_usd": 0.0},
            symbol="ADAUSD",
        )
        == {}
    )
    assert (
        compute_liquidation_clusters(
            {"price": 0.15, "vol_1h_pct": 1.0},
            {"bid_depth_usd": 50_000.0, "ask_depth_usd": 50_000.0},
            symbol="ADAUSD",
        )
        == {}
    )

    # 2. Compact equity tickers (NVDA, SPCX) without _US_EQ suffix are caught by SSOT is_equity_symbol
    #    even when Cboe options OI is present, and residual cached liq_* keys are stripped cleanly.
    assert (
        compute_liquidation_clusters(
            {"price": 152.0, "vol_1h_pct": 1.1},
            {"oi_usd": 303_000_000_000.0, "bid_depth_usd": 200_000.0, "ask_depth_usd": 200_000.0},
            symbol="NVDA",
        )
        == {}
    )
    stale_eq_micro = {
        "NVDA": {
            "ob_imbalance": 0.52,
            "bid_depth_usd": 200_000.0,
            "ask_depth_usd": 190_000.0,
            "oi_usd": 303_000_000_000.0,
            "long_liq_px": 148.0,
            "short_liq_px": 156.0,
            "liq_magnet_bias": "BALANCED_LIQ_MAGNET",
        },
        "SPCXxUSD": {
            "ob_imbalance": 0.45,
            "bid_depth_usd": 45_000.0,
            "ask_depth_usd": 55_000.0,
            "long_liq_px": 140.0,
            "short_liq_px": 148.0,
        },
    }
    eq_pt = {"NVDA": {"price": 152.0}, "SPCXxUSD": {"price": 144.5}}
    sec3b = "\n".join(_format_microstructure_section(stale_eq_micro, eq_pt))
    assert "liq_clusters(" not in sec3b
    sync_sibling_symbol_telemetry(eq_pt, stale_eq_micro, persist_cycle_deltas=False, cycle_history={})
    assert "long_liq_px" not in stale_eq_micro["NVDA"]
    assert "long_liq_px" not in stale_eq_micro["SPCXxUSD"]


def test_item_2_3_kraken_xstocks_active_pairs_dynamic_directive_and_ipo_isolation(tmp_path):
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from state_io import atomic_write_json

    krk_dir = tmp_path / "kraken"
    krk_dir.mkdir(parents=True, exist_ok=True)
    (krk_dir / "config.env").write_text(
        "XSTOCK_YAHOO=ANT\n"
        "XSTOCK_ALLOC_PRICE=0\n"
        "XSTOCK_ACTIVE_PAIRS=SPCXxUSD:141.75\n"
        "XSTOCK_TP_ALERT_PCT=10\n"
        "XSTOCK_TP2_ALERT_PCT=16\n"
        "XSTOCK_SL_ALERT_PCT=12\n"
        "XSTOCK_BOT_TP_PCT=9.0\n"
        "XSTOCK_BOT_TP_TRAIL_PCT=3.0\n"
        "XSTOCK_BOT_DCA_DROP_PCT=3.0\n",
        encoding="utf-8",
    )
    # Even if ANTxUSD lists with 0 held shares while 10.25 SPCXx.T is held, underlying stays SPCX,
    # listed_pair/alloc_price/pnl_vs_alloc_pct resolve from XSTOCK_ACTIVE_PAIRS, directive uses live 10.25 qty,
    # and ANT IPO alerts stay isolated under watched_ipo_alerts.
    atomic_write_json(
        str(krk_dir / "xstock_state.json"),
        {"known_assets": {"SPCXx.T": 10.25, "AAPLx.T": 2e-06}, "pair": "ANTXUSD", "allocated": None},
    )
    xs = KrakenXstocksCollector().collect(
        str(tmp_path),
        kraken_dir=str(krk_dir),
        prices_and_trends={"SPCXxUSD": {"price": 148.8375}},
    )
    assert xs is not None
    assert xs["underlying"] == "SPCX"
    assert xs["t212_sibling"] == "SPCX_US_EQ"
    assert xs["listed_pair"] == "SPCXxUSD"
    assert xs["alloc_price"] == pytest.approx(141.75)
    assert xs["pnl_vs_alloc_pct"] == pytest.approx(5.0)
    assert xs["watched_ipo_underlying"] == "ANT"
    assert xs["watched_ipo_alerts"] == {
        "tp1_alert_pct": 10.0,
        "tp2_alert_pct": 16.0,
        "sl_alert_pct": 12.0,
        "listed_pair": "ANTxUSD",
    }
    assert "tp1_alert_pct" not in xs
    assert "10.25 SPCXx.T" in xs["active_swing_directive"]
    assert "15.74655" not in xs["active_swing_directive"]
    assert "TP +9% / Trail 3% / DCA -3%" in xs["active_swing_directive"]


def test_item_2_4_us_equities_rth_no_fake_close_px_and_sec3a_quote_meta(tmp_path, monkeypatch):
    from intelligence.sentiment.collectors.microstructure_collector import enrich_24h_and_microstructure
    from orchestratorOS.supervisor.prompt_sections import _format_market_prices_section, _format_portfolio_section
    from state_io import atomic_write_json

    cdir = tmp_path / "cachedb"
    (cdir / "state").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    # Open US RTH timestamp: Wed Nov 18 2026 16:00:00 UTC (11:00 EST) = 1795017600
    rth_ts = 1_795_017_600.0
    atomic_write_json(
        str(cdir / "state" / "venue_balances_state.json"),
        {"pos:t212:NVDA_US_EQ": {"venue": "t212", "symbol": "NVDA_US_EQ", "free_base": 3.0, "price": 152.45, "ts": rth_ts - 90.0}},
    )
    atomic_write_json(
        str(cdir / "prices" / "cache_24_price_NVDA_US_EQ.json"),
        {"items": {"NVDA_US_EQ": [[int((rth_ts - 300.0) * 1000), 152.10], [int((rth_ts - 10.0) * 1000), 152.37]]}},
    )
    pt_rth = {"NVDA_US_EQ": {"price": 152.45}}
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_rth, {}, {}, {}, rth_ts, allow_network=False, persist=False)
    assert pt_rth["NVDA_US_EQ"]["quote_source"] == "yahoo_rth"
    assert "t212_close_px" not in pt_rth["NVDA_US_EQ"]

    sec3a_eq = "\n".join(
        _format_market_prices_section(
            {"NVDA_US_EQ": {"price": 158.56, "quote_source": "yahoo_ext", "quote_age_min": 1.2, "t212_close_px": 155.15, "chg_24h_pct": 2.2}}
        )
    )
    assert "Price=$158.56 [src=yahoo_ext, age=1.2m, t212_close=$155.15] | 24h=+2.20%" in sec3a_eq

    cmc = {"equity_bellwethers": {"NVDA_US_EQ": {"price": 158.56, "quote_source": "yahoo_ext", "t212_cycle": {"qty": 3.0}}}}
    assert "Multi-Horizon Deltas):" in "\n".join(_format_portfolio_section({"cross_market_context": cmc}, asset_scope="crypto", include_macro_benchmarks=False))
    assert "Multi-Horizon Deltas & Active Cycles):" in "\n".join(_format_portfolio_section({"cross_market_context": cmc}, asset_scope="all", include_macro_benchmarks=False))


def test_item_3_1_real_spot_cvd_across_binance_kraken_hyperliquid_and_pretrade_symmetry(tmp_path, monkeypatch):
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.microstructure_evaluator import reconcile_sibling_flow_and_liq_regimes
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from state_io import atomic_write_json

    now_s = 1_795_032_000.0
    # 1. Polymorphic 3-venue real spot CVD extraction (Binance 5m klines, Kraken Trades, Hyperliquid @107 1m spot candles)
    bn_cvd = BinanceProvider.extract_spot_cvd_15m_usd(
        [
            [0, "63000", "63100", "62900", "63050", "10", 0, "1000000.0", 500, "6", "620000.0", "0"],
            [0, "63050", "63200", "63000", "63180", "12", 0, "1200000.0", 600, "8", "800000.0", "0"],
        ]
    )
    assert bn_cvd == pytest.approx(640_000.0)

    krk_cvd = KrakenProvider.extract_spot_cvd_15m_usd(
        {
            "error": [],
            "result": {
                "HYPEUSD": [
                    ["67.00", "100.0", now_s - 120.0, "b", "l", "", 1],
                    ["66.95", "40.0", now_s - 60.0, "s", "m", "", 2],
                ],
                "last": "1795031940000000000",
            },
        },
        since_ts=now_s - 900.0,
    )
    assert krk_cvd == pytest.approx(4022.0)

    hl_cvd = HyperliquidProvider.extract_spot_cvd_15m_usd(
        [{"o": "66.8", "h": "67.2", "l": "66.8", "c": "67.2", "v": "500.0"}]
    )
    assert hl_cvd is not None and hl_cvd > 20_000.0

    # 2. OrderbookDepthCollector attaches real_spot CVD per venue and preserves it across sibling reconciliation & pretrade
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    ob_col = OrderbookDepthCollector(cache_dir=str(cdir), cache_ttl_sec=0.0)
    monkeypatch.setattr(
        ob_col,
        "_fetch_json_dict",
        lambda url, label, lookup_sym, **kw: {"bids": [["67.00", "200.0"]], "asks": [["67.02", "180.0"]]},
    )
    monkeypatch.setattr(
        ob_col,
        "_fetch_spot_cvd_payload",
        lambda url, post_body: {
            "error": [],
            "result": {"HYPEUSD": [["67.00", "100.0", now_s - 30.0, "b", "l", "", 1]], "last": "1"},
        },
    )
    snap_krk = ob_col.fetch("HYPEUSD", allow_network=True, now=now_s)
    assert snap_krk is not None
    assert snap_krk.spot_cvd_source == "real_spot"
    assert snap_krk.spot_cvd_15m_usd == pytest.approx(6700.0)

    # Cross-venue sibling reconciliation must NOT overwrite real_spot CVD with synthetic est_ob_15m
    sib_m = {
        "HYPEUSDC": {"bid_depth_usd": 50000.0, "ask_depth_usd": 40000.0, "cvd_15m_usd": 12000.0, "spot_cvd_15m_usd": 24000.0, "spot_cvd_source": "real_spot"},
        "HYPEUSD": {"bid_depth_usd": 30000.0, "ask_depth_usd": 35000.0, "cvd_15m_usd": 12000.0, "spot_cvd_15m_usd": 6700.0, "spot_cvd_source": "real_spot"},
    }
    reconcile_sibling_flow_and_liq_regimes(sib_m, {"HYPEUSDC": {"price": 67.0}, "HYPEUSD": {"price": 67.0}})
    assert sib_m["HYPEUSDC"]["spot_cvd_15m_usd"] == pytest.approx(24000.0)
    assert sib_m["HYPEUSD"]["spot_cvd_15m_usd"] == pytest.approx(6700.0)

    # Pretrade & Supervisor 3B emit symmetric [real_spot] vs [est:ob+15m] tags
    atomic_write_json(str(cdir / "prices" / "cache_current_price.json"), {"items": {"HYPEUSD": [[int(now_s * 1000), 67.0]]}})
    atomic_write_json(
        str(cdir / "telemetry" / "whale_snapshot_HYPEUSD.json"),
        {"symbol": "HYPEUSD", "ts": now_s - 5.0, "top_traders_long_pct": 0.55, "taker_buy_sell_ratio": 1.12, "taker_buy_vol_usd": 60000.0, "taker_sell_vol_usd": 48000.0},
    )
    pt_hype = collect_pretrade_telemetry("HYPEUSD", cachedb_dir=str(cdir), allow_network=False, now_ts=now_s)
    assert "spot_cvd_15m=$+6,700[real_spot]" in pt_hype.get("15m Cumulative Volume Delta (CVD)", "")
    sec3b_real = "\n".join(_format_microstructure_section({"HYPEUSD": sib_m["HYPEUSD"]}, {"HYPEUSD": {"price": 67.0}}, now_ts=now_s))
    assert "spot_cvd_15m=+$6.7K[real_spot]" in sec3b_real
    assert BinanceProvider.resolve_venue_for_orderbook_url("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500") == "binance"
    assert BinanceProvider.resolve_venue_for_orderbook_url("https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500") == ""
    assert BinanceProvider.resolve_venue_for_orderbook_url("https://api.kraken.com/0/public/Depth?pair=HYPEUSD&count=200") == "kraken"
    assert BinanceProvider.resolve_venue_for_orderbook_url("https://api.hyperliquid.xyz/info?type=l2Book&coin=HYPE") == "hyperliquid"


def test_item_3_2_pretrade_liq_clusters_model_est_tag(tmp_path):
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from state_io import atomic_write_json

    now_s = 1_795_032_500.0
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(cdir / "prices" / "cache_current_price.json"), {"items": {"BTCUSDC": [[int(now_s * 1000), 63000.0]]}})
    atomic_write_json(
        str(cdir / "telemetry" / "derivatives_telemetry_BTCUSDC.json"),
        {"symbol": "BTCUSDC", "ts": now_s - 5.0, "funding_rate": 0.0001, "open_interest_usd": 5_000_000_000.0},
    )
    pt = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir), allow_network=False, now_ts=now_s)
    assert pt.get("Liquidation Clusters (LONG vs SHORT)", "").endswith(", src=model_est")


def test_item_3_3_directional_fill_prob_depth_scaled_wall_penalty_and_4dec_front_run():
    from intelligence.sentiment.evaluators.tca_risk_evaluator import enrich_open_limit_orders_tca

    # 1. Marketable resting BUY (>= spot) or SELL (<= spot) has 99.0% fill prob instead of abs(dist) penalty.
    mkt_orders = [
        {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "price": 195.0},
        {"venue": "binance", "symbol": "TAOUSDC", "side": "SELL", "price": 185.0},
    ]
    enrich_open_limit_orders_tca(mkt_orders, {"TAOUSDC": {"price": 190.0, "vol_1h_pct": 1.0}}, {})
    assert mkt_orders[0]["fill_prob_1h_pct"] == pytest.approx(99.0)
    assert mkt_orders[1]["fill_prob_1h_pct"] == pytest.approx(99.0)

    # 2. Micro $6,000 wall on $5M BTCUSDC depth is ignored (<2% depth), whereas heavy $1.5M wall penalizes strongly.
    btc_orders = [
        {"venue": "binance", "symbol": "BTCUSDC", "side": "SELL", "price": 63500.0},
        {"venue": "binance", "symbol": "BTCUSDC", "side": "SELL", "price": 63500.0},
    ]
    enrich_open_limit_orders_tca(
        [btc_orders[0]],
        {"BTCUSDC": {"price": 63000.0, "vol_1h_pct": 0.9}},
        {"BTCUSDC": {"ob_mid_price": 63000.0, "ask_depth_usd": 5_000_000.0, "ask_wall_px": 63200.0, "ask_wall_usd": 6_000.0}},
    )
    enrich_open_limit_orders_tca(
        [btc_orders[1]],
        {"BTCUSDC": {"price": 63000.0, "vol_1h_pct": 0.9}},
        {"BTCUSDC": {"ob_mid_price": 63000.0, "ask_depth_usd": 1_800_000.0, "ask_wall_px": 63200.0, "ask_wall_usd": 1_500_000.0}},
    )
    assert "wall_queue_pos" not in btc_orders[0]
    assert btc_orders[1]["wall_queue_pos"] == "BEHIND_ASK_WALL"
    assert btc_orders[1]["fill_prob_1h_pct"] < btc_orders[0]["fill_prob_1h_pct"] * 0.70

    # 3. 4-decimal Kraken TAOUSD BBO/wall preserves 4-decimal recommended_front_run_px.
    tao_krk = [{"venue": "kraken", "symbol": "TAOUSD", "side": "SELL", "price": 193.0}]
    enrich_open_limit_orders_tca(
        tao_krk,
        {"TAOUSD": {"price": 190.78, "vol_1h_pct": 1.1}},
        {
            "TAOUSD": {
                "ob_mid_price": 190.7785,
                "best_bid": 190.6862,
                "best_ask": 190.8707,
                "ask_depth_usd": 600_000.0,
                "ask_wall_px": 192.2305,
                "ask_wall_usd": 178_093.0,
            }
        },
    )
    assert tao_krk[0]["wall_queue_pos"] == "BEHIND_ASK_WALL"
    assert tao_krk[0]["recommended_front_run_px"] == pytest.approx(192.2113, abs=0.001)
    assert round(tao_krk[0]["recommended_front_run_px"], 2) != tao_krk[0]["recommended_front_run_px"]


def test_item_3_4_realized_pnl_flat_cleanup_and_sibling_cum_symmetry():
    """Verify closed/dust positions clear lifo_realized_pnl_usd and cycle dicts preserve distinct cum_realized_pnl_usd without duplicate subcycle keys."""
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        build_cross_market_context,
        compute_enterprise_portfolio_risk_metrics,
    )
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section

    # 1. Flattened/dust position (Kraken ADAUSD $0.13 < $1.00 min_notional after SELL) clears lifo_realized_pnl_usd
    # so closed historical losses do not pollute active subcycle_lifo_realized_pnl_usd in Risk Engine.
    ro_ada = {
        "venue": "kraken",
        "last_action_side": "SELL",
        "last_sell": {"price": 0.165, "qty": 1500.0, "ts_ms": 1_794_500_000_000},
        "lifo_realized_pnl_usd": -245.93,
    }
    ro_btc = {
        "venue": "binance",
        "last_action_side": "SELL",
        "last_sell": {"price": 63500.0, "qty": 0.01, "ts_ms": 1_795_000_000_000},
        "lifo_realized_pnl_usd": 14.64,
    }
    tradable_positions = [
        {
            "venue": "kraken",
            "symbol": "ADAUSD",
            "base": "ADA",
            "quote": "USD",
            "price": 0.150883,
            "free_base": 0.840208,
            "value_usd": 0.13,
            "free_quote": 3.38,
            "exchange_min_notional_usd": 1.0,
        },
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "base": "BTC",
            "quote": "USDC",
            "price": 62400.0,
            "free_base": 0.031377,
            "value_usd": 1957.92,
            "free_quote": 1739.67,
            "exchange_min_notional_usd": 5.0,
        },
    ]
    recent_orders = {"ADAUSD": ro_ada, "BTCUSDC": ro_btc}
    venue_ro = {"kraken": {"ADAUSD": ro_ada}, "binance": {"BTCUSDC": ro_btc}}
    enrich_positions_with_orders_and_cycles(
        tradable_positions,
        {"ADAUSD": {"price": 0.150883}, "BTCUSDC": {"price": 62400.0}},
        {},
        venue_ro,
        recent_orders,
        {},
        {},
        {},
        {},
    )
    assert "lifo_realized_pnl_usd" not in ro_ada
    assert "lifo_realized_pnl_usd" not in tradable_positions[0]
    risk = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": tradable_positions,
            "venue_quote_cash": {"binance:USDC": {"venue": "binance", "free": 1739.67}},
            "recent_orders": recent_orders,
        },
        {"BTCUSDC": {"price": 62400.0, "vol_1h_pct": 0.9}},
        {},
    )
    assert risk["subcycle_lifo_realized_pnl_usd"] == pytest.approx(14.64)

    # 2. Crypto scope preserves cum_realized_pnl_usd (-99.53) in kraken_xstocks.t212_sibling_cycle without duplicate subcycle key;
    # Equities/All scope omits cum_realized_pnl_usd from t212_sibling_cycle because t212_cycles renders it right above (2 decimals).
    sib_cycle = {
        "qty": 3.262965,
        "avg_cost_px": 144.9872,
        "dd_pct": -1.29,
        "realized_pnl_usd": 5.31,
        "subcycle_realized_pnl_usd": 5.31,
        "cum_realized_pnl_usd": -99.53,
    }
    sec_crypto = "\n".join(
        _format_portfolio_section(
            {
                "kraken_xstocks": {
                    "underlying": "SPCX",
                    "listed_pair": "SPCXxUSD",
                    "t212_sibling": "SPCX_US_EQ",
                    "t212_sibling_cycle": dict(sib_cycle),
                },
            },
            "crypto",
        )
    )
    assert "'realized_pnl_usd': 5.31" in sec_crypto
    assert "'cum_realized_pnl_usd': -99.53" in sec_crypto
    assert "'subcycle_realized_pnl_usd'" not in sec_crypto

    sec_all = "\n".join(
        _format_portfolio_section(
            {
                "t212_cycles": {
                    "NVDA_US_EQ": {
                        "qty": 0.41,
                        "cost_usd": 68.55,
                        "avg_cost_px": 167.1951,
                        "realized_pnl_usd": 10.35,
                        "subcycle_realized_pnl_usd": 10.35,
                        "cum_realized_pnl_usd": 21.2,
                    },
                    "SPCX_US_EQ": dict(sib_cycle),
                },
                "kraken_xstocks": {
                    "underlying": "SPCX",
                    "listed_pair": "SPCXxUSD",
                    "t212_sibling": "SPCX_US_EQ",
                    "t212_sibling_cycle": dict(sib_cycle),
                },
            },
            "all",
        )
    )
    assert "realized_pnl=$10.35 (cum_cycles=$21.20)" in sec_all
    assert "realized_pnl=$5.31 (cum_cycles=$-99.53)" in sec_all
    assert "'cum_realized_pnl_usd'" not in sec_all
    assert "'subcycle_realized_pnl_usd'" not in sec_all

    sec_eq = "\n".join(
        _format_portfolio_section(
            {
                "t212_cycles": {"SPCX_US_EQ": dict(sib_cycle)},
                "kraken_xstocks": {
                    "underlying": "SPCX",
                    "listed_pair": "SPCXxUSD",
                    "t212_sibling": "SPCX_US_EQ",
                    "t212_sibling_cycle": dict(sib_cycle),
                },
            },
            "equities",
        )
    )
    assert "SPCX_US_EQ: qty=3.262965" in sec_eq
    assert "'t212_sibling_cycle'" not in sec_eq

    # 3. build_cross_market_context preserves distinct cum_realized_pnl_usd / zero subcycle_realized_pnl_usd on equity_bellwethers.
    cmc = build_cross_market_context(
        {"SPCX_US_EQ": {"price": 143.11}, "RGNT_US_EQ": {"price": 1.15}},
        {},
        all_t212_cycles={
            "SPCX_US_EQ": dict(sib_cycle),
            "RGNT_US_EQ": {
                "qty": 3.93,
                "avg_cost_px": 1.4346,
                "realized_pnl_usd": -103.41,
                "subcycle_realized_pnl_usd": 0.0,
                "cum_realized_pnl_usd": -103.41,
            },
        },
    )
    spcx_bw_cyc = cmc["equity_bellwethers"]["SPCX_US_EQ"]["t212_cycle"]
    assert spcx_bw_cyc["realized_pnl_usd"] == pytest.approx(5.31)
    assert spcx_bw_cyc["cum_realized_pnl_usd"] == pytest.approx(-99.53)
    assert "subcycle_realized_pnl_usd" not in spcx_bw_cyc
    rgnt_bw_cyc = cmc["equity_bellwethers"]["RGNT_US_EQ"]["t212_cycle"]
    assert rgnt_bw_cyc["realized_pnl_usd"] == pytest.approx(-103.41)
    assert rgnt_bw_cyc["subcycle_realized_pnl_usd"] == pytest.approx(0.0)
    assert "cum_realized_pnl_usd" not in rgnt_bw_cyc


def test_item_3_5_reconciled_lifo_tp_stop_upgrade_and_cycle_boundary():
    """Verify profitable LIFO dip-harvests preserve/restore TP, verified in-cycle sells upgrade SL->TP even when sell_changed=False, and prior-cycle SL stays protected."""
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
        parse_venue_cycle_state,
        reconcile_venue_cycles_with_recent_orders,
    )

    # 1. Live PROD NVDA_US_EQ & SPCX_US_EQ: profitable LIFO dip harvest (155.97 -> 163.31, +$10.35; 140.51 -> 142.54, +$5.31)
    # raises remaining core bag's avg_cost_px above sell_px (167.20 > 163.31; 144.99 > 142.54) -- must stay/restore TP, never REDUCE.
    nvda_cyc = parse_venue_cycle_state(
        {
            "qty": 0.41,
            "cost_usd": 68.55,
            "entry_price": 207.93,
            "last_buy_price": 155.97,
            "last_sell_price": 163.31,
            "last_sell_ts": 1_794_497_684.5,
            "last_exit_kind": "TP",
            "realized_net_usd": 21.20,
        },
        "NVDA_US_EQ",
        {"NVDA_US_EQ": {"price": 150.81}},
        avg_decimals=4,
        include_meta=True,
    )
    spcx_cyc = parse_venue_cycle_state(
        {
            "qty": 3.262965,
            "cost_usd": 473.09,
            "entry_price": 144.9872,
            "last_buy_price": 144.9872,
            "last_sell_price": 142.54,
            "last_sell_ts": 1_794_240_254.0,
            "last_exit_kind": "TP",
            "realized_net_usd": -99.53,
        },
        "SPCX_US_EQ",
        {"SPCX_US_EQ": {"price": 143.11}},
        avg_decimals=4,
        include_meta=True,
    )
    t212_cycs = {"NVDA_US_EQ": nvda_cyc, "SPCX_US_EQ": spcx_cyc}
    v_ro_t212 = {
        "t212": {
            "NVDA_US_EQ": {
                "last_action_side": "SELL",
                "active_avg_buy_px": 167.1992,
                "lifo_realized_pnl_usd": 10.35,
                "last_buy": {"price": 155.97, "qty": 1.41, "ts_ms": 1_794_494_062_000},
                "last_sell": {"price": 163.31, "qty": 1.41, "ts_ms": 1_794_497_663_000},
                "recent_buys": [
                    {"price": 207.93, "qty": 0.96, "ts_ms": 1_793_719_666_000},
                    {"price": 155.97, "qty": 1.41, "ts_ms": 1_794_494_062_000},
                ],
                "recent_sells": [{"price": 163.31, "qty": 1.41, "ts_ms": 1_794_497_663_000}],
            },
            "SPCX_US_EQ": {
                "last_action_side": "SELL",
                "active_avg_buy_px": 144.9663,
                "lifo_realized_pnl_usd": 5.31,
                "last_buy": {"price": 140.51, "qty": 0.71, "ts_ms": 1_793_724_724_000},
                "last_sell": {"price": 142.54, "qty": 2.04, "ts_ms": 1_794_240_243_000},
                "recent_buys": [
                    {"price": 141.80, "qty": 1.41, "ts_ms": 1_793_719_675_000},
                    {"price": 140.51, "qty": 0.71, "ts_ms": 1_793_724_724_000},
                ],
                "recent_sells": [{"price": 142.54, "qty": 2.04, "ts_ms": 1_794_240_243_000}],
            },
        }
    }
    ts_map = reconcile_venue_cycles_with_recent_orders(
        {},
        {},
        t212_cycs,
        v_ro_t212,
        {("t212", "NVDA_US_EQ"): 0.41, ("t212", "SPCX_US_EQ"): 3.262965},
        {"NVDA_US_EQ": {"price": 150.81}, "SPCX_US_EQ": {"price": 143.11}},
    )
    assert t212_cycs["NVDA_US_EQ"]["last_exit_kind"] == "TP"
    assert t212_cycs["SPCX_US_EQ"]["last_exit_kind"] == "TP"

    # 2. Verified in-cycle profitable sell upgrades stale SL -> TP even when prev_cyc_sell == new_cyc_sell (sell_changed=False),
    # while prior-cycle SL (Cycle 17 SL @ 68.90 before Cycle 18 ENTRY @ 68.69) stays SL.
    pos_list = [
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "free_base": 4.0, "price": 68.80, "last_sell_px": 69.20},
        {"venue": "kraken", "symbol": "HYPEUSD", "free_base": 26.32, "price": 67.92, "last_sell_px": 68.900061},
    ]
    hl_cycs = {"HYPEUSDC": {"qty": 4.0, "cost_usd": 274.0, "avg_cost_px": 68.50, "entry_price": 68.50, "last_sell_price": 69.20, "last_exit_kind": "SL"}}
    krk_cycs = {"HYPEUSD": {"qty": 26.32, "cost_usd": 1808.27, "avg_cost_px": 68.69, "entry_price": 68.69, "last_sell_price": 68.900061, "last_exit_kind": "SL"}}
    enrich_positions_with_orders_and_cycles(
        pos_list,
        {"HYPEUSDC": {"price": 68.80}, "HYPEUSD": {"price": 67.92}},
        {},
        {
            "hyperliquid": {"HYPEUSDC": {"last_sell": {"price": 69.20, "ts_ms": 1_794_700_000_000}, "last_buy": {"price": 68.50, "ts_ms": 1_794_600_000_000}}},
            "kraken": {"HYPEUSD": {"last_sell": {"price": 68.900061, "ts_ms": 1_794_627_083_000}, "last_buy": {"price": 68.69, "ts_ms": 1_794_631_269_000}}},
        },
        {},
        krk_cycs,
        hl_cycs,
        {},
        {**ts_map, ("hyperliquid:entry", "HYPEUSDC"): 1_794_600_000_000, ("kraken:entry", "HYPEUSD"): 1_794_631_269_000},
    )
    assert hl_cycs["HYPEUSDC"]["last_exit_kind"] == "TP"
    assert krk_cycs["HYPEUSD"]["last_exit_kind"] == "SL"


def test_item_3_6_scoped_risk_engine_xstock_beta_and_shadow_state_exclusion(tmp_path):
    """Verify shadow/paper/replay .state_*.json exclusion across venues, xStock equity beta/corr in Risk Engine, and BTC-free equities Risk Engine formatting."""
    from cachepaths import is_shadow_or_paper_state_filename
    from intelligence.sentiment.collectors.portfolio_collector import VenueCyclesCollector
    from intelligence.sentiment.evaluators.tca_risk_evaluator import compute_enterprise_portfolio_risk_metrics
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from state_io import atomic_write_json

    # 1. Filename + payload shadow/paper/replay filtering across Kraken, Hyperliquid, and T212
    assert is_shadow_or_paper_state_filename(".state_REPLAY.json") is True
    assert is_shadow_or_paper_state_filename(".state_PAPER_HYPE.json") is True
    assert is_shadow_or_paper_state_filename(".state_NVDA_US_EQ_PAPER.json") is True
    assert is_shadow_or_paper_state_filename(".state_HYPEUSD.json") is False
    assert is_shadow_or_paper_state_filename(".state_SPCXxUSD.json") is False
    assert is_shadow_or_paper_state_filename(".state_NVDA_US_EQ.json") is False

    for sub in ("kraken", "hyperliquid", "212trading", "cachedb"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    # Filename-level shadow/replay states (even with non-empty qty and empty orders) must be skipped
    atomic_write_json(str(tmp_path / "hyperliquid" / ".state_REPLAY.json"), {"qty": 10.0, "cost": 700.0, "orders": []})
    atomic_write_json(str(tmp_path / "212trading" / ".state_PAPER_NVDA_US_EQ.json"), {"qty": 5.0, "cost_usd": 750.0, "orders": []})
    # Payload-level paper/shadow states (orders=[] after paper fill, or mode="paper" / last_hist_fill_id="PAPER-9") must be skipped
    atomic_write_json(
        str(tmp_path / "hyperliquid" / ".state_HYPE_long.json"),
        {"qty": 0.85, "cost": 50.19, "orders": [], "last_hist_fill_id": "PAPER-9"},
    )
    atomic_write_json(
        str(tmp_path / "kraken" / ".state_SOLUSD.json"),
        {"qty": 12.0, "cost": 1800.0, "mode": "paper", "orders": []},
    )
    # Real live state must be preserved
    atomic_write_json(
        str(tmp_path / "hyperliquid" / ".state_HYPE.json"),
        {"qty": 15.91, "cost": 1357.91, "entry_price": 92.696, "orders": []},
    )
    krk_c, hl_c, t212_c = VenueCyclesCollector().collect(str(tmp_path / "cachedb"), workspace_dir=str(tmp_path))
    assert krk_c == {}
    assert list(hl_c.keys()) == ["HYPEUSDC"]
    assert hl_c["HYPEUSDC"]["qty"] == pytest.approx(15.91)
    assert t212_c == {}

    # 2. Tokenized xStock (SPCXXUSD / SPCXxUSD) uses equity beta/corr priors (0.35 / 0.30) rather than 1.15 / 0.78 altcoin priors
    risk_xstock = compute_enterprise_portfolio_risk_metrics(
        {
            "tradable_positions": [
                {"venue": "kraken", "symbol": "SPCXXUSD", "base": "SPCXX", "quote": "USD", "value_usd": 2000.0, "free_quote": 0.0}
            ],
            "venue_quote_cash": {},
            "recent_orders": {},
        },
        {"SPCXXUSD": {"price": 143.0, "vol_1h_pct": 0.5}},
        {},
    )
    assert risk_xstock["portfolio_btc_beta"] == pytest.approx(0.35)
    assert risk_xstock["corr_vs_btc_24h"] == {"SPCXx": 0.3}

    # 3. Equities Scope Risk Engine omits BTC beta/corr while Crypto & All scopes include them
    sec_eq = "\n".join(
        _format_portfolio_section(
            {
                "portfolio_risk_engine": {
                    "unified_nav_usd": 1257.48,
                    "invested_exposure_usd": 534.11,
                    "global_free_cash_usd": 723.37,
                    "global_free_cash_pct": 57.5,
                    "venue_nav_breakdown_usd": {"t212": 1257.48},
                    "portfolio_btc_beta": 0.15,
                    "avg_corr_vs_btc_24h": 0.3,
                    "intraday_var_95_usd": 14.68,
                    "intraday_var_95_pct": 1.17,
                    "subcycle_lifo_realized_pnl_usd": 22.99,
                }
            },
            "equities",
            include_macro_benchmarks=False,
        )
    )
    assert "Portfolio Risk Engine (Equities Scope: t212; excl. crypto venues): equities_nav=$1,257.48" in sec_eq
    assert "portfolio_btc_beta=" not in sec_eq
    assert "avg_corr_24h=" not in sec_eq
    assert "intraday_var_95=$14.68 (1.17%)" in sec_eq
    assert "subcycle_lifo_pnl=$+22.99" in sec_eq


def test_item_3_7_book_window_level_cap_symmetry_and_thin_book_isolation(tmp_path):
    """Verify truncated REST L2 book_window=+/-X%[level_cap] is symmetric across Section 3B and Pretrade and never triggers on <100-level thin sides."""
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section
    from state_io import atomic_write_json

    now_s = 1_795_061_600.0
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)

    # 1. Pretrade GLLM telemetry surfaces book_window=+/-0.07%[level_cap] symmetrically with Supervisor Section 3B
    atomic_write_json(
        str(cdir / "prices" / "cache_current_price.json"),
        {"items": {"BTCUSDC": [[int((now_s - 0.5) * 1000), 63_280.0]]}},
    )
    atomic_write_json(
        str(cdir / "telemetry" / "orderbook_depth_BTCUSDC.json"),
        {
            "symbol": "BTCUSDC",
            "venue": "binance",
            "ts": now_s - 5.0,
            "mid_price": 63_280.0,
            "best_bid": 63_279.99,
            "best_ask": 63_280.01,
            "spread_bps": 0.003,
            "bid_depth_usd": 3_100_000.0,
            "ask_depth_usd": 3_000_000.0,
            "depth_1pct_bid_usd": 3_100_000.0,
            "depth_1pct_ask_usd": 3_000_000.0,
            "imbalance_ratio": 0.508,
            "largest_bid_wall_usd": 2_000.0,
            "largest_bid_wall_price": 63_250.0,
            "largest_ask_wall_usd": 2_000.0,
            "largest_ask_wall_price": 63_310.0,
            "book_span_pct": 0.07,
        },
    )
    pt_res = collect_pretrade_telemetry(
        "BTCUSDC",
        cachedb_dir=str(cdir),
        allow_network=False,
    )
    tel = pt_res.get("telemetry") if isinstance(pt_res, dict) and "telemetry" in pt_res else pt_res
    assert "book_window=+/-0.07%[level_cap]" in tel["Orderbook Depth Within 1%"]
    assert tel["Orderbook Ask Wall Ahead"] == "None detected within +/-0.07%[level_cap]"

    sec_3b = "\n".join(
        _format_microstructure_section(
            {
                "BTCUSDC": {
                    "ob_mid_price": 63_280.0,
                    "spread_bps": 0.003,
                    "bid_depth_usd": 3_100_000.0,
                    "ask_depth_usd": 3_000_000.0,
                    "depth_1pct_bid_usd": 3_100_000.0,
                    "depth_1pct_ask_usd": 3_000_000.0,
                    "book_span_pct": 0.07,
                    "est_slip_2k_bps": 0.5,
                }
            },
            {"BTCUSDC": {"price": 63_280.0}},
            now_ts=now_s,
        )
    )
    assert "book_window=+/-0.07%[level_cap]" in sec_3b

    # 2. Thin uncapped ask side (<100 levels at 0.60% while 120-level bid side reaches 2.0%) is NOT falsely tagged [level_cap]
    col = OrderbookDepthCollector(cache_dir=str(cdir))
    bids_120 = [[100.0 - i * 0.02, 1.0] for i in range(1, 121)]
    asks_12 = [[100.0 + i * 0.05, 1.0] for i in range(1, 13)]
    col._fetch_json_dict = lambda *_a, **_kw: {"bids": bids_120, "asks": asks_12}
    col._resolve_real_spot_cvd_15m = lambda *_a, **_kw: None
    snap_thin = col.fetch("TAOUSD", allow_network=True, now=now_s)
    assert snap_thin is not None
    assert snap_thin.book_span_pct == pytest.approx(1.5)


def test_item_3_8_offline_real_vol_upgrade_xstock_case_and_last_cycle_exec_alignment(tmp_path):
    """Verify offline TPO->REAL_VOL upgrade across case-sensitive Kraken xStocks (SPCXXUSD vs SPCXxUSD) and US equities, plus last_cycle_execution_status <=60m fill / >30m resting dedup alignment."""
    import time as _time
    from types import SimpleNamespace
    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from state_io import atomic_write_json

    now_s = _time.time()
    cdir = tmp_path / "cachedb"
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)

    # 1. Case-sensitive Linux files: cache_24_price_SPCXxUSD.json (tick-only TPO) + cache_ohlc_24h_SPCXXUSD.json (REAL_VOL)
    #    queried via Kraken gateway symbol "SPCXXUSD" with allow_network=False must resolve series AND upgrade TPO -> REAL_VOL.
    t0_ms = int((now_s - 22 * 3600.0) * 1000)
    tick_rows = [[t0_ms + i * 3600_000, 142.0 + (i % 4) * 0.6] for i in range(23)]
    ohlc_rows = [[t0_ms + i * 3600_000, 142.0 + (i % 4) * 0.6, 150.0 + i * 10.0] for i in range(23)]
    atomic_write_json(str(cdir / "prices" / "cache_24_price_SPCXxUSD.json"), {"items": {"SPCXxUSD": tick_rows}})
    atomic_write_json(str(cdir / "prices" / "cache_ohlc_24h_SPCXXUSD.json"), {"points": ohlc_rows})

    spcx_pt = collect_pretrade_telemetry("SPCXXUSD", cachedb_dir=str(cdir), allow_network=False)
    spcx_tel = spcx_pt.get("telemetry") if isinstance(spcx_pt, dict) and "telemetry" in spcx_pt else spcx_pt
    assert "vp_source=REAL_VOL" in str(spcx_tel.get("24h Volume Profile (POC / Value Area)", ""))

    # 2. Offline US equity tick-only 24h series upgrades TPO -> REAL_VOL from local cache_ohlc_24h_NVDA_US_EQ.json with zero network I/O.
    eq_ticks = [[t0_ms + i * 3600_000, 151.0 + (i % 3) * 0.8] for i in range(22)]
    eq_ohlc = [[t0_ms + i * 3600_000, 151.0 + (i % 3) * 0.8, 2500.0 + i * 50.0] for i in range(22)]
    atomic_write_json(str(cdir / "prices" / "cache_24_price_NVDA_US_EQ.json"), {"items": {"NVDA_US_EQ": eq_ticks}})
    atomic_write_json(str(cdir / "prices" / "cache_ohlc_24h_NVDA_US_EQ.json"), {"points": eq_ohlc})

    nvda_pt = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(cdir), allow_network=False)
    nvda_tel = nvda_pt.get("telemetry") if isinstance(nvda_pt, dict) and "telemetry" in nvda_pt else nvda_pt
    assert "vp_source=REAL_VOL" in str(nvda_tel.get("24h Volume Profile (POC / Value Area)", ""))

    # 3. last_cycle_execution_status keeps <=60m fill (normalizing SPCXXUSD -> SPCXxUSD and unwrapping 1-element list)
    #    while deduplicating >30m OPEN_RESTING orders already shown in open_limit_orders; prompt instructions state <=60m / <=30m.
    sec3c = "\n".join(
        _format_portfolio_section(
            {
                "open_limit_orders": [
                    {"venue": "kraken", "symbol": "TAOUSD", "side": "SELL", "price": 236.8, "qty": 3.37, "order_id": "OID-REST-1", "age": "319.5h"}
                ],
                "last_cycle_execution_status": [
                    {"venue": "kraken", "symbol": "TAOUSD", "side": "SELL", "status": "OPEN_RESTING", "order_id": "OID-REST-1", "age_min": 19170.0},
                    {"venue": "kraken", "symbol": "SPCXXUSD", "side": "SELL", "status": "FILLED", "fill_px": 144.89, "filled_qty": 5.52, "slippage_bps": -63.2, "order_id": "OID-FILL-2", "age_min": 42.0},
                ],
                "recent_ai_intents": [
                    {"decision": "SELL", "symbol": "SPCXXUSD", "venue": "kraken", "order_type": "LIMIT", "status": "EXECUTED", "venue_status": "FILLED", "order_id": "OID-FILL-2", "age_h": 0.7}
                ],
            },
            "crypto",
            include_macro_benchmarks=False,
        )
    )
    assert "last_cycle_execution_status: {'venue': 'kraken', 'symbol': 'SPCXxUSD'" in sec3c
    assert "OID-REST-1" not in sec3c.split("last_cycle_execution_status:")[1].splitlines()[0]
    assert "'symbol': 'SPCXxUSD'" in sec3c

    sup_prompt = build_supervisor_prompt(
        TelemetrySensorSnapshot(evaluated_at_ts=now_s),
        SimpleNamespace(total_scanned_files=0, error_count=0, warning_count=0, critical_events=[], guard_interventions=[]),
        topic_key="auto_cryptotrade_executor",
    )
    assert "recent `<=60m` fill/rejection or `<=30m` resting order" in sup_prompt


def test_burnin_antxusd_empty_book_backoff_and_equities_xstock_readonly_framing(tmp_path, caplog):
    """Verify pre-IPO Kraken empty-book backoff (0 WARNING spam, no 0.25x spin) and read-only xStock sibling framing on equities scope."""
    from cachepaths import cache_ohlc_24h_filename
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector

    assert cache_ohlc_24h_filename("SPCXxUSD") == "cache_ohlc_24h_SPCXxUSD.json"
    assert cache_ohlc_24h_filename("SPCXxUSD", preserve_case=True) == "cache_ohlc_24h_SPCXxUSD.json"

    now_s = 1_776_500_000.0
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), start_sync=True)
    calls = []

    def _fake_fetch_empty_book(url, *_args, **_kwargs):
        calls.append(url)
        return {"error": [], "result": {"ANTxUSD": {"asks": [], "bids": []}}}

    ob_col._fetch_json_dict = _fake_fetch_empty_book
    caplog.clear()
    with caplog.at_level("WARNING", logger="intelligence.external.orderbook_depth"):
        res1 = ob_col.fetch("ANTxUSD", allow_network=True, now=now_s)
        res2 = ob_col.fetch("ANTxUSD", allow_network=True, now=now_s + 7.5)

    assert res1 is None and res2 is None
    assert len(calls) == 1
    assert ob_col.is_unlisted_xstock_backoff_active("ANTxUSD", now_s + 60.0) is True
    assert ob_col.is_unlisted_xstock_backoff_active("ANTxUSD", now_s + 301.0) is False
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []

    # Active xStock with an existing cached orderbook must still warn & fall back to cached on unexpected empty book
    seeded_snap = OrderbookSnapshot(
        symbol="SPCXxUSD",
        mid_price=146.5,
        bid_depth_usd=12000.0,
        ask_depth_usd=11000.0,
        imbalance_ratio=0.522,
        largest_bid_wall_usd=3500.0,
        largest_bid_wall_price=146.0,
        largest_ask_wall_usd=3200.0,
        largest_ask_wall_price=147.0,
        ts=now_s - 120.0,
        best_bid=146.4,
        best_ask=146.6,
        spread_bps=13.66,
    )
    ob_col._commit_snapshot("SPCXxUSD", now_s - 120.0, seeded_snap)
    caplog.clear()
    with caplog.at_level("WARNING", logger="intelligence.external.orderbook_depth"):
        fallback_snap = ob_col.fetch("SPCXxUSD", allow_network=True, now=now_s)
    assert fallback_snap is not None and fallback_snap.mid_price == 146.5
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1

    # GlobalTelemetryCollector advances _last_orderbook_ts to current_ts (no 0.25x retry spin) during pre-IPO backoff
    daemon = GlobalTelemetryCollector(symbols=["ANTxUSD"], cache_dir=str(tmp_path), skip_external=False)
    daemon.orderbook_collector = ob_col
    results = daemon._new_external_cycle_results(now_s + 10.0)
    daemon._step_external_symbols(now_s + 10.0, False, results, bounded=True)
    assert daemon._last_orderbook_ts["ANTxUSD"] == now_s + 10.0

    xs_payload = {
        "underlying": "SPCX",
        "listed_pair": "SPCXxUSD",
        "live_price": 146.84,
        "t212_sibling": "SPCX_US_EQ",
        "watched_ipo_underlying": "ANT",
        "watched_ipo_alerts": [{"title": "Anthropic IPO rumor"}],
    }
    eq_lines = "\n".join(_format_portfolio_section({"kraken_xstocks": xs_payload}, "equities", include_macro_benchmarks=False))
    cr_lines = "\n".join(_format_portfolio_section({"kraken_xstocks": xs_payload}, "crypto", include_macro_benchmarks=False))
    assert "Cross-Venue Kraken Tokenized Sibling Reference (READ-ONLY in Equities Scope;" in eq_lines
    assert "API-TRADABLE 24/7 on 'kraken'" not in eq_lines
    assert "watched_ipo_alerts" not in eq_lines
    assert "API-TRADABLE 24/7 on 'kraken' at 0bps maker/taker fee" in cr_lines
    assert "ANTxUSD IPO Playbook" in cr_lines


def test_phase1_symbol_identity_3tier_asset_policy_and_spot_ob_reconciler_ssot():
    """Verify SymbolIdentity, 3-tier AssetSessionPolicy hierarchy (Crypto/TokenizedEquity/RthEquity), and unified SpotOrderbookPriceReconcilerEvaluator."""
    from cachepaths import cache_24_price_filename, cache_ohlc_24h_filename, price_cache_filename_variants
    from instrument_registry import resolve_symbol_identity
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        SpotOrderbookPriceReconcilerEvaluator,
        reconcile_spot_and_orderbook_microstructure,
    )
    from intelligence.sentiment.evaluators.asset_session_policy import (
        AssetClassPolicyInterface,
        AssetClassPolicyRegistry,
        AssetSessionPolicy,
        Continuous24x7SessionPolicy,
        Crypto24x7SessionPolicy,
        EquityRthSessionPolicy,
        TokenizedEquity24x7SessionPolicy,
        resolve_asset_session_policy,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import TelemetryEvaluatorFactory

    id_btc = resolve_symbol_identity("BTCUSDC")
    assert id_btc.asset_class == "crypto" and id_btc.is_crypto and id_btc.has_crypto_perpetuals and not id_btc.uses_equity_risk_priors

    id_xstock = resolve_symbol_identity("SPCXxUSD")
    assert id_xstock.asset_class == "tokenized_equity"
    assert id_xstock.storage_key == "SPCXxUSD"
    assert id_xstock.wire_symbol == "SPCXxUSD"
    assert id_xstock.display_symbol == "SPCXxUSD"
    assert not id_xstock.has_crypto_perpetuals and not id_xstock.has_liquidation_clusters and id_xstock.uses_equity_risk_priors
    xstock_cands = id_xstock.candidate_cache_symbols()
    assert "SPCXxUSD" in xstock_cands
    assert price_cache_filename_variants("SPCXxUSD", cache_ohlc_24h_filename)[0] == "cache_ohlc_24h_SPCXxUSD.json"
    assert price_cache_filename_variants("SPCXxUSD", cache_24_price_filename)[0] == "cache_24_price_SPCXxUSD.jsonl"

    id_eq = resolve_symbol_identity("NVDA_US_EQ")
    assert id_eq.asset_class == "rth_equity" and id_eq.is_rth_equity and not id_eq.has_crypto_perpetuals and id_eq.uses_equity_risk_priors

    with pytest.raises(TypeError):
        AssetSessionPolicy()
    with pytest.raises(TypeError):
        Continuous24x7SessionPolicy()

    p_crypto = resolve_asset_session_policy("HYPEUSDC")
    p_tok = resolve_asset_session_policy("SPCXXUSD")
    p_eq = resolve_asset_session_policy("SPCX_US_EQ")
    assert isinstance(p_crypto, Crypto24x7SessionPolicy) and isinstance(p_crypto, Continuous24x7SessionPolicy) and isinstance(p_crypto, AssetClassPolicyInterface)
    assert isinstance(p_tok, TokenizedEquity24x7SessionPolicy) and isinstance(p_tok, Continuous24x7SessionPolicy)
    assert isinstance(p_eq, EquityRthSessionPolicy) and AssetClassPolicyRegistry.get("tokenized_equity") is p_tok
    assert p_crypto.has_crypto_perpetuals is True and p_tok.has_crypto_perpetuals is False and p_eq.has_crypto_perpetuals is False
    assert p_tok.allow_live_vp_ohlc_network is True and p_eq.allow_live_vp_ohlc_network is False
    assert p_eq.ohlc_vp_live_clock_skew_ms > p_tok.ohlc_vp_live_clock_skew_ms

    reconciler = TelemetryEvaluatorFactory.create("micro_spot_ob_reconciler")
    assert isinstance(reconciler, SpotOrderbookPriceReconcilerEvaluator)
    m_fiat = {"ob_mid_price": 200.0, "best_bid": 199.9, "best_ask": 200.1, "bid_wall_px": 198.0, "ask_wall_px": 202.0}
    new_px, adopted_px, act = reconcile_spot_and_orderbook_microstructure(
        "TAOUSD",
        196.0,
        m_fiat,
        is_usd_sym=True,
        is_eq_sym=False,
        is_exact_ob_sym=True,
        ob_ts_sec=1000.0,
        now_ts=1200.0,
        last_series_pt=(1_190_000.0, 196.0),
    )
    assert act == "rescaled_fiat_walls" and adopted_px is None and new_px == 196.0
    assert m_fiat["ob_mid_price"] == 196.0 and m_fiat["ask_wall_px"] < 200.0


def test_phase3_prompt_view_evaluators_3tier_and_section_parity() -> None:
    """Verify 3-tier PromptViewEvaluatorBase hierarchy and pure Section 3A/3B/3C view models."""
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        TelemetryEvaluatorFactory,
        TelemetryEvaluatorInterface,
    )
    from intelligence.sentiment.evaluators.prompt_view_evaluator import (
        MarketPricesPromptViewEvaluator,
        MicrostructurePromptViewEvaluator,
        PortfolioPromptViewEvaluator,
        PromptViewEvaluatorBase,
        coalesce_case_variants,
        evaluate_market_prices_prompt_view,
        evaluate_microstructure_prompt_view,
        evaluate_portfolio_prompt_view,
    )
    from orchestratorOS.supervisor.prompt_sections import (
        _format_market_prices_section,
        _format_microstructure_section,
        _format_portfolio_section,
    )

    with pytest.raises(TypeError):
        PromptViewEvaluatorBase()

    ev_3a = TelemetryEvaluatorFactory.create("prompt_market_prices_view")
    ev_3b = TelemetryEvaluatorFactory.create("prompt_microstructure_view")
    ev_3c = TelemetryEvaluatorFactory.create("prompt_portfolio_view")
    assert isinstance(ev_3a, MarketPricesPromptViewEvaluator) and isinstance(ev_3a, TelemetryEvaluatorInterface)
    assert isinstance(ev_3b, MicrostructurePromptViewEvaluator)
    assert isinstance(ev_3c, PortfolioPromptViewEvaluator)

    merged_case = coalesce_case_variants({
        "SPCXXUSD": {"price": 122.0, "low_24h": 115.0},
        "SPCXxUSD": {"price": 122.5, "high_24h": 126.0},
    })
    assert list(merged_case.keys()) == ["SPCXxUSD"]
    assert merged_case["SPCXxUSD"]["price"] == 122.5
    assert merged_case["SPCXxUSD"]["low_24h"] == 115.0
    assert merged_case["SPCXxUSD"]["high_24h"] == 126.0

    pt_3a = {
        "BTCUSDC": {"price": 75000.0, "chg_24h_pct": 1.5},
        "BTCUSD": {"price": 74990.0, "chg_24h_pct": 1.4, "poc_px": 74800.0, "val_px": 74200.0, "vah_px": 75400.0},
    }
    view_3a = evaluate_market_prices_prompt_view(pt_3a)
    assert view_3a["btc_entry"]["price"] == 75000.0
    assert view_3a["btc_entry"]["poc_px"] == 74800.0
    assert len(view_3a["rows"]) == 1
    sym_0, pinfo_0, sibs_0 = view_3a["rows"][0]
    assert sym_0 == "BTCUSDC" and pinfo_0["poc_px"] == 74800.0 and len(sibs_0) == 1
    assert "sibling venue: BTCUSD=" in "\n".join(_format_market_prices_section(pt_3a))

    micro_3b = {
        "SPCXXUSD": {
            "ob_mid_price": 122.5,
            "best_bid": 122.4,
            "best_ask": 122.6,
            "bid_depth_usd": 25000.0,
            "ask_depth_usd": 30000.0,
            "long_liq_px": 110.0,
            "liq_magnet_bias": "LONG_LIQ_BELOW",
        }
    }
    rows_3b = evaluate_microstructure_prompt_view(micro_3b, {"SPCXxUSD": {"price": 122.5, "chg_15m_pct": 0.4}})
    assert len(rows_3b) == 1
    assert rows_3b[0]["sym_is_tok"] is True and rows_3b[0]["sym_no_perp"] is True
    assert "long_liq_px" not in rows_3b[0]["minfo"]
    assert "liq_clusters(" not in "\n".join(
        _format_microstructure_section(micro_3b, {"SPCXxUSD": {"price": 122.5}})
    )

    ps_3c = {
        "open_limit_orders": [
            {
                "symbol": "SPCXxUSD",
                "side": "BUY",
                "qty": 2.0,
                "orig_qty": 2.0,
                "remaining_qty": 2.0,
                "filled_qty": 0.0,
                "order_id": "OID-1",
                "orderId": "OID-1",
            }
        ],
        "tradable_positions": [
            {"venue": "kraken", "symbol": "SPCXxUSD", "price": 122.5, "lifo_realized_pnl_usd": 14.5}
        ],
        "kraken_cycles": {
            "SPCXxUSD": {"open_qty": 1.0, "avg_buy": 120.0, "cycle_realized_pnl_usd": 4.25}
        },
        "kraken_xstocks": {
            "listed_pair": "SPCXxUSD",
            "watched_ipo_underlying": "ANT",
            "holdings": {"SPCXx": 1.0, "DUSTx": 0.0},
            "t212_sibling": "SPCX_US_EQ",
            "t212_sibling_cycle": {"open_qty": 2.0, "avg_buy": 121.0, "dd_pct": -1.2, "pnl_vs_avg_pct": -1.2},
        },
        "cross_market_context": {
            "equity_bellwethers": {
                "NVDA_US_EQ": {"price": 185.0, "chg_24h_pct": 1.1, "va_regime": "INSIDE_VALUE", "extra_noise": 99},
                "RGNT_US_EQ": {"price": 1.18, "chg_24h_pct": -0.5},
            }
        },
        "last_cycle_execution_status": [
            {"symbol": "SPCXXUSD", "status": "OPEN_RESTING", "age_min": 45.0, "order_id": "OID-1"},
            {"symbol": "SPCXXUSD", "status": "FILLED", "age_min": 5.0, "order_id": "OID-2"},
        ],
        "assetguardian_state": {
            "BTCUSDC": {
                "tier_1": {"active": False, "pending": False, "completed_tiers": [1]},
                "tier_2": {"active": True, "target_px": 72000.0, "terminal_orders": ["old"]},
            }
        },
        "recent_ai_intents": [
            {"symbol": "SPCXXUSD", "side": "HOLD", "status": "SKIPPED"},
            {"symbol": "SPCXXUSD", "side": "BUY", "status": "EXECUTED", "venue_status": "OPEN", "order_id": "OID-9", "age_h": 0.2},
        ],
    }
    view_3c_crypto = evaluate_portfolio_prompt_view(ps_3c, "crypto")
    assert "orderId" not in view_3c_crypto["clean_open_limit_orders"][0]
    assert "orig_qty" not in view_3c_crypto["clean_open_limit_orders"][0]
    assert "RGNT_US_EQ" not in view_3c_crypto["equity_bellwethers"]
    assert "extra_noise" not in view_3c_crypto["equity_bellwethers"]["NVDA_US_EQ"]
    assert view_3c_crypto["clean_last_cycle_execution_status"]["symbol"] == "SPCXxUSD"
    rt_map = dict((k, v) for k, _lbl, v in view_3c_crypto["runtime_states"])
    assert "tier_1" not in rt_map["assetguardian_state"]["BTCUSDC"]
    assert "terminal_orders" not in rt_map["assetguardian_state"]["BTCUSDC"]["tier_2"]
    assert rt_map["recent_ai_intents"][0]["status"] == "RESTING_IN_BOOK"
    assert rt_map["recent_ai_intents"][0]["symbol"] == "SPCXxUSD"

    view_3c_eq = evaluate_portfolio_prompt_view(ps_3c, "equities")
    assert "watched_ipo_underlying" not in view_3c_eq["kraken_xstocks"]["payload"]
    sec_3c_eq = "\n".join(_format_portfolio_section(ps_3c, "equities"))
    assert "READ-ONLY in Equities Scope" in sec_3c_eq


def test_phase4_portfolio_flat_state_transitions_and_intent_order_matching_ssot(tmp_path) -> None:
    """Verify canonical flat cycle/position state transitions and unified SSOT intent-order matching."""
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        extract_symbol_recent_fills,
        reset_flat_cycle_state,
        reset_flat_position_and_recent_order_state,
    )
    from providers.intent_reconciliation import reconcile_intents_from_execution_audit
    from providers.intent_state import (
        ai_intent_client_order_id,
        extract_intent_notional,
        extract_intent_primary_order_id,
        extract_intent_timestamp_s,
        order_matches_intent_client_id,
        symbols_match_intent_scope_and_quote,
    )
    from providers.intent_telemetry import _resolve_intent_execution_details

    cyc_d = {"qty": 1.5, "cost_usd": 180.0, "avg_cost_px": 120.0, "dd_pct": -1.5, "pnl_vs_avg_pct": -1.5, "pos_peak": 125.0, "dd_from_peak_pct": -5.2}
    reset_flat_cycle_state(cyc_d, qty=0.0, clear_peak=True)
    assert cyc_d["qty"] == 0.0 and cyc_d["cost_usd"] == 0.0 and cyc_d["avg_cost_px"] is None
    assert "pos_peak" not in cyc_d and "dd_from_peak_pct" not in cyc_d

    ro_d = {"active_avg_buy_px": 120.0, "open_lots": [{"lot_id": "L1"}], "lifo_realized_pnl_usd": 8.5, "last_sell": {"price": 124.0}}
    pos_d = {"active_avg_buy_px": 120.0, "pnl_vs_avg_pct": 2.0, "open_lots": [{"lot_id": "L1"}], "trailing_peak": 125.0, "dd_from_peak_pct": -1.0, "lifo_realized_pnl_usd": 8.5}
    reset_flat_position_and_recent_order_state(ro_d, pos_d, full_pos_clear=True)
    assert "active_avg_buy_px" not in ro_d and "open_lots" not in ro_d and ro_d["last_sell"]["price"] == 124.0
    assert "active_avg_buy_px" not in pos_d and "trailing_peak" not in pos_d and "lifo_realized_pnl_usd" not in pos_d

    # Legacy intent dict with 'id' (intent_id) must never mistake 'id' for a venue order ID
    legacy_intent = {
        "id": "intent_1700000000_buy_btcusdc",
        "executed_at": 1_700_000_050_000,
        "symbol": "BTCUSDC",
        "venue": "binance",
        "decision": "BUY",
        "order_type": "LIMIT",
        "limit_price": 74000.0,
        "status": "EXECUTED",
        "history": [{"action": "EXECUTED", "order_id": "VENUE-OID-77"}],
    }
    raw_ts, ts_sec = extract_intent_timestamp_s(legacy_intent)
    assert raw_ts == 1_700_000_050_000 and ts_sec == 1_700_000_050.0
    assert extract_intent_primary_order_id(legacy_intent, include_top_level=True) == "VENUE-OID-77"
    assert extract_intent_primary_order_id({"id": "intent_only"}, default=None, include_top_level=True) is None

    assert symbols_match_intent_scope_and_quote("BTCUSDC", "BTCUSD", venue_a="binance", venue_b="binance") is True
    assert symbols_match_intent_scope_and_quote("NVDA_US_EQ", "NVDAUSDC", venue_a="t212", venue_b="binance") is False

    cid_native = ai_intent_client_order_id("intent_1700000000_buy_btcusdc")
    assert order_matches_intent_client_id({"clientOrderId": cid_native}, {"intent_id": "intent_1700000000_buy_btcusdc"}) is True
    assert order_matches_intent_client_id({"clientOrderId": "TA_MainThread_123"}, {"intent_id": "intent_1700000000_buy_btcusdc"}) is False

    det = _resolve_intent_execution_details(
        {
            "intent_id": "intent_1700000000_buy_btcusdc",
            "timestamp": 1_700_000_000.0,
            "symbol": "BTCUSDC",
            "venue": "binance",
            "decision": "BUY",
            "order_type": "LIMIT",
            "limit_price": 74000.0,
            "suggested_notional_usd": 150.0,
            "status": "PENDING",
        },
        1_700_000_030.0,
        extract_intent_notional,
        open_orders=[{"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 74000.0, "clientOrderId": cid_native}],
    )
    assert det["exec_status"] == "OPEN_RESTING" and det["is_open_in_venue"] is True

    fills_out = extract_symbol_recent_fills(
        [
            {"symbol": "SPCXxUSD", "orderId": "K-1", "side": "BUY", "price": 120.0, "executedQty": 2.0, "timestamp": 1_700_000_000_000},
            {"symbol": "SPCXxUSD", "orderId": "K-2", "side": "BUY", "price": 116.0, "executedQty": 1.0, "timestamp": 1_700_000_600_000},
            {"symbol": "SPCXxUSD", "orderId": "K-3", "side": "SELL", "price": 119.0, "executedQty": 1.0, "timestamp": 1_700_001_200_000},
        ],
        known_ai_order_ids={"K-2"},
        symbol="SPCXxUSD",
    )
    assert fills_out is not None
    assert fills_out["lifo_realized_pnl_usd"] == 3.0
    assert len(fills_out["open_lots"]) == 1 and fills_out["open_lots"][0]["qty"] == 2.0

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "execution_audit_2023-11-14.jsonl").write_text(
        json.dumps(
            {
                "ts": 1_700_000_010.0,
                "event": "order_terminal",
                "intent_id": "intent_rec_1",
                "venue": "kraken",
                "symbol": "SPCXxUSD",
                "side": "buy",
                "order_id": "KRK-OID-1",
                "status": "closed",
                "venue_status": "FILLED",
                "filled_qty": 1.5,
                "cost": 183.0,
                "qty": 1.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    intents_rec = [{"intent_id": "intent_rec_1", "timestamp": 1_700_000_005.0, "symbol": "SPCXxUSD", "venue": "kraken", "decision": "BUY", "status": "FAILED"}]
    assert reconcile_intents_from_execution_audit(intents_rec, str(audit_dir)) is True
    assert intents_rec[0]["status"] == "EXECUTED"
    assert intents_rec[0]["execution_payload"]["orderId"] == "KRK-OID-1"


def test_telemetry_audit_7_fixes_stale_ob_kinematics_rvol_us_open_ant_realized_liq_and_prompt_compression(tmp_path, monkeypatch):
    """End-to-end regression lock for all 7 telemetry audit fixes."""
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import pricewindow as pw
    from botcore import topic_spec
    from instrument_registry import (
        discover_tradable_specs,
        is_equity_symbol,
        load_registry,
        resolve_instrument_spec,
        tracked_price_instruments,
    )
    from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.macro.calendar_ssot import compute_dynamic_sentiment_interval
    from intelligence.macro_sentiment_ssot import (
        is_us_cash_open_volatility_window,
        resolve_effective_macro_benchmarks_interval_sec,
    )
    from intelligence.sentiment.collectors.microstructure_collector import load_symbol_microstructure
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_24h_price_metrics,
        compute_series_kinematics,
        compute_volume_rvol_metrics,
    )
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_portfolio_prompt_view
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        build_cross_market_context,
        enrich_open_limit_orders_tca,
    )
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import (
        _format_macro_and_news_section,
        _format_microstructure_section,
    )
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from state_io import atomic_write_json

    now_s = 1_795_099_695.0  # 2026-11-19 16:48:15 EET (09:48:15 EST)

    # 1. Stale orderbook desync: open_limit_orders anchors to live Price ($143.36, +2.18%) not stale ob_mid ($144.52, +1.36%),
    #    flags [ob_stale] when ob_age > 30s, and skips stale wall front-run penalties.
    stale_oo = [{"venue": "kraken", "symbol": "SPCXxUSD", "side": "SELL", "price": 146.48}]
    enrich_open_limit_orders_tca(
        stale_oo,
        {"SPCXxUSD": {"price": 143.36, "vol_1h_pct": 1.2}},
        {
            "SPCXxUSD": {
                "ob_mid_price": 144.52,
                "best_bid": 144.40,
                "best_ask": 144.64,
                "ask_depth_usd": 50_000.0,
                "ask_wall_px": 145.50,
                "ask_wall_usd": 20_000.0,
                "_ob_raw_ts": now_s - 123.7,
            }
        },
        now_ts=now_s,
    )
    assert stale_oo[0]["dist_from_spot_pct"] == pytest.approx(2.18, abs=0.01)
    assert stale_oo[0]["ob_flag"] == "[ob_stale]"
    assert stale_oo[0]["ob_age_s"] == pytest.approx(123.7, abs=0.1)
    assert "recommended_front_run_px" not in stale_oo[0]
    assert "wall_queue_pos" not in stale_oo[0]

    # When orderbook refreshes (age=5s <= 30s), [ob_stale] clears and fresh wall analysis runs
    enrich_open_limit_orders_tca(
        stale_oo,
        {"SPCXxUSD": {"price": 143.36, "vol_1h_pct": 1.2}},
        {
            "SPCXxUSD": {
                "ob_mid_price": 143.36,
                "best_bid": 143.30,
                "best_ask": 143.42,
                "ask_depth_usd": 50_000.0,
                "ask_wall_px": 145.50,
                "ask_wall_usd": 20_000.0,
                "_ob_raw_ts": now_s - 5.0,
            }
        },
        now_ts=now_s,
    )
    assert "ob_flag" not in stale_oo[0]
    assert "ob_age_s" not in stale_oo[0]
    assert stale_oo[0]["wall_queue_pos"] == "BEHIND_ASK_WALL"

    # 2. Quadratic boundary overshoot guard on sharp step-drops (no sign flip opposite to falling slopes)
    now_ms = int(now_s * 1000.0)
    knots = [(-300, 145.80), (-240, 145.55), (-180, 145.26), (-150, 145.20), (-120, 144.00), (-90, 143.84), (-60, 143.58), (-30, 143.45), (0, 143.36)]
    mono_drop_pts = []
    for i in range(len(knots) - 1):
        t0, p0 = knots[i]
        t1, p1 = knots[i + 1]
        for s in range(t0, t1, 5):
            mono_drop_pts.append([now_ms + s * 1000.0, p0 + ((s - t0) / (t1 - t0)) * (p1 - p0)])
    mono_drop_pts.append([now_ms, knots[-1][1]])
    kin_drop = compute_series_kinematics(mono_drop_pts)
    assert kin_drop["vel_90s_pct_min"] <= 0.0

    step_knots = [(-90, 145.20), (-75, 143.65), (-60, 143.58), (-30, 143.45), (0, 143.36)]
    w_step = pw.PriceWindow("SPCXxUSD", 100, sample_rate_sec=5.0)
    for i in range(len(step_knots) - 1):
        t0, p0 = step_knots[i]
        t1, p1 = step_knots[i + 1]
        for s in range(t0, t1, 5):
            w_step.process_price(p0 + ((s - t0) / (t1 - t0)) * (p1 - p0))
    w_step.process_price(143.36)
    vel_step, acc_step = w_step.calculate_acceleration(90.0)
    assert vel_step <= 0.0
    assert acc_step > 0.0  # Deceleration of the fall is still detected accurately

    # 3. Spot total volume (15m/1h) & RVOL across providers, 24h OHLC bars, and Section 3B formatting
    bn_cvd, bn_vol = BinanceProvider.extract_spot_flow_15m_usd(
        [
            [0, "100", "101", "99", "100.5", "10", 0, "100000.0", 50, "6", "60000.0", "0"],
            [0, "100.5", "102", "100", "101.0", "12", 0, "150000.0", 60, "8", "100000.0", "0"],
        ]
    )
    assert bn_cvd == pytest.approx(70_000.0)
    assert bn_vol == pytest.approx(250_000.0)

    krk_cvd, krk_vol = KrakenProvider.extract_spot_flow_15m_usd(
        {
            "error": [],
            "result": {
                "SPCXxUSD": [
                    ["144.0", "10.0", now_s - 120.0, "b", "l", "", 1],
                    ["143.5", "30.0", now_s - 60.0, "s", "m", "", 2],
                ],
                "last": "1",
            },
        },
        since_ts=now_s - 900.0,
    )
    assert krk_cvd == pytest.approx(1440.0 - 4305.0, abs=1.0)
    assert krk_vol == pytest.approx(1440.0 + 4305.0, abs=1.0)

    hl_cvd, hl_vol = HyperliquidProvider.extract_spot_flow_15m_usd(
        [{"o": "40.0", "h": "41.0", "l": "40.0", "c": "41.0", "v": "500.0"}]
    )
    assert hl_cvd > 0.0 and hl_vol > 15_000.0

    # 24h 5m OHLC bars with 2x surge in the latest 15m -> rvol_15m ~ 2.0x
    ohlc_5m_bars = []
    for i in range(288):  # 24h of 5m bars
        t_b = now_ms - (287 - i) * 300_000
        vol_b = 20.0 if i >= 285 else 10.0
        ohlc_5m_bars.append([t_b, 100.0, vol_b])
    rvol_m = compute_volume_rvol_metrics(ohlc_5m_bars)
    assert rvol_m["spot_vol_15m_usd"] == pytest.approx(6_000.0)
    assert rvol_m["rvol_15m"] == pytest.approx(1.98, abs=0.05)
    assert rvol_m["spot_vol_1h_usd"] == pytest.approx(15_000.0)
    assert rvol_m["rvol_1h"] == pytest.approx(1.24, abs=0.05)
    m24_with_rvol = compute_24h_price_metrics(ohlc_5m_bars)
    assert m24_with_rvol["rvol_15m"] == pytest.approx(rvol_m["rvol_15m"])

    sec3b_rvol = "\n".join(
        _format_microstructure_section(
            {
                "SPCXxUSD": {
                    "ob_imbalance": 0.45,
                    "bid_depth_usd": 40_000.0,
                    "ask_depth_usd": 50_000.0,
                    "spot_cvd_15m_usd": -2600.0,
                    "spot_cvd_source": "real_spot",
                    "spot_vol_15m_usd": 5745.0,
                }
            },
            {"SPCXxUSD": {"price": 143.36, "spot_vol_24h_avg_15m_usd": 3000.0, "spot_vol_1h_usd": 20000.0, "rvol_1h": 1.67}},
            now_ts=now_s,
        )
    )
    assert "spot_cvd_15m=-$2.6K[real_spot], spot_vol_15m=$5.7K (rvol_15m=1.92x), spot_vol_1h=$20.0K (rvol_1h=1.67x)" in sec3b_rvol

    # 4. US Cash Open volatility window (09:25-10:15 NY / 16:25-17:15 EET) accelerates macro_benchmarks to 45s & sentiment to 5m
    ts_us_open = datetime(2026, 11, 19, 14, 46, 0, tzinfo=timezone.utc).timestamp()  # 09:46 EST / 16:46 EET
    ts_calm_eu = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc).timestamp()   # 07:00 EST / 14:00 EET
    ts_sunday = datetime(2026, 11, 22, 14, 46, 0, tzinfo=timezone.utc).timestamp()
    assert is_us_cash_open_volatility_window(ts_us_open) is True
    assert is_us_cash_open_volatility_window(ts_calm_eu) is False
    assert is_us_cash_open_volatility_window(ts_sunday) is False
    assert resolve_effective_macro_benchmarks_interval_sec(300.0, now_ts=ts_us_open) == pytest.approx(45.0)
    assert resolve_effective_macro_benchmarks_interval_sec(300.0, now_ts=ts_calm_eu) == pytest.approx(300.0)
    gtc_open = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_news=True, skip_external=True)
    assert gtc_open._effective_macro_benchmarks_interval_sec(ts_us_open) == pytest.approx(45.0)
    assert gtc_open._effective_macro_benchmarks_interval_sec(ts_calm_eu) == pytest.approx(300.0)
    gtc_open.macro_benchmarks_interval_sec = 3600.0
    assert gtc_open._effective_macro_benchmarks_interval_sec(ts_us_open) == pytest.approx(3600.0)
    sent_open_itv, sent_open_reason = compute_dynamic_sentiment_interval(now_ts=ts_us_open)
    assert sent_open_itv == pytest.approx(300.0)
    assert "us_cash_open_volatility_window" in sent_open_reason

    # 5. Official Nasdaq RTH ticker ANT_US_EQ remains tracked for market/bellwether analysis while blocked from live/AI trading
    all_syms, _, nb_prov = tracked_price_instruments()
    assert "ANT_US_EQ" in all_syms and nb_prov["ANT_US_EQ"] == "t212"
    assert "ANTxUSD" in all_syms and nb_prov["ANTxUSD"] == "kraken"
    assert is_equity_symbol("ANT_US_EQ") is True
    eq_tradable = discover_tradable_specs(asset_scope="equities", include_closed_rth=True)
    assert "T212_ANT" not in eq_tradable  # Non-tradable (config.ant.env.off + ai.tradable=no): price/bellwether telemetry only
    assert "T212_SPCX" in eq_tradable
    assert resolve_instrument_spec("ANT_US_EQ", "t212", load_registry()) is None
    cmc_ant = build_cross_market_context(
        {
            "ANT_US_EQ": {"price": 103.10, "quote_source": "yahoo_rth", "quote_age_min": 0.5, "chg_15m_pct": 0.45, "chg_24h_pct": 24.2},
            "SPCX_US_EQ": {"price": 143.50, "quote_source": "yahoo_rth", "chg_24h_pct": -1.1},
            "RGNT_US_EQ": {"price": 1.20, "chg_24h_pct": -0.5},
        },
        {},
    )
    view_ant = evaluate_portfolio_prompt_view({"cross_market_context": cmc_ant}, asset_scope="crypto")
    assert "ANT_US_EQ" in view_ant["equity_bellwethers"]
    assert view_ant["equity_bellwethers"]["ANT_US_EQ"]["quote_source"] == "yahoo_rth"
    assert "RGNT_US_EQ" not in view_ant["equity_bellwethers"]

    # 6. Realized Bybit forced liquidations persisted & rendered alongside liq_clusters(src=model_est)
    cdir = tmp_path / "rliq_cache"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        str(cdir / "telemetry" / "derivatives_telemetry_BTCUSDC.json"),
        {"symbol": "BTCUSDC", "ts": now_s - 5.0, "funding_rate": 0.0001, "open_interest_usd": 5_000_000_000.0},
    )
    gtc_liq = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(cdir), skip_news=True, skip_external=True)
    bybit_col = BybitLiquidationCollector(symbols=["BTCUSDT"], start_sync=False)
    bybit_col.record_event("BTCUSDT", "BUY", 150_000.0, ts=now_s - 120.0)   # LONG liquidation
    bybit_col.record_event("BTCUSDT", "SELL", 40_000.0, ts=now_s - 60.0)   # SHORT liquidation
    gtc_liq.bybit_liq_collector = bybit_col
    assert gtc_liq.update_realized_liquidations(now=now_s) is True
    m_btc = load_symbol_microstructure(str(cdir), "BTCUSDC")
    assert m_btc["realized_liq_15m_long_usd"] == pytest.approx(150_000.0)
    assert m_btc["realized_liq_15m_short_usd"] == pytest.approx(40_000.0)
    sec3b_liq = "\n".join(_format_microstructure_section({"BTCUSDC": m_btc}, {"BTCUSDC": {"price": 90_000.0}}, now_ts=now_s))
    assert "src=model_est" in sec3b_liq
    assert "realized_liq_15m(long=$150.0K, short=$40.0K, n=2)" in sec3b_liq

    # 7. Compressed static prompt rules (>50% reduction vs 17.2KB/2300w baseline) & bounded sentiment_advisor summary
    compact_prompt = build_supervisor_prompt(
        SimpleNamespace(),
        SimpleNamespace(total_scanned_files=0, error_count=0, warning_count=0, critical_events=[], guard_interventions=[]),
        topic_spec("auto_cryptotrade_executor"),
    )
    assert len(compact_prompt) < 8500
    assert len(compact_prompt.split()) < 1100
    long_sa = "First factual macro sentence with BTC metrics and CB premium +4bps. " * 10
    sec1_lines = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(sentiment_advisor={"confidence": 0.85, "age_min": 2.0, "summary": long_sa}),
            asset_scope="crypto",
        )
    )
    sa_out = next(ln for ln in sec1_lines.splitlines() if "Macro Sentiment Advisor:" in ln)
    assert len(sa_out.split("summary=", 1)[1]) <= 360


def test_mixed_case_xstock_routing_frozen_ob_suppression_bnb_reserve_and_limit_px_sync(tmp_path):
    """Verify:
    1. Mixed-case tokenized xStocks (SPCXxUSD, ANTxUSD) route Depth, Spot Flow, and Historical OHLC exclusively to Kraken tokenized_asset (zero Binance Futures ANTUSDT leak or SPCXUSD 92m freeze).
    2. Severely stale/frozen orderbooks (>60s xStock, >300s fiat USD) never rescale synthetic ghost walls or adopt stale OB mid when live spot is missing, and Section 3B suppresses frozen walls/imbalance/spot_cvd_15m.
    3. Total Binance Portfolio Value explicitly labels the persisted BNB fee reserve on live_reconciled (and strips it in equities scope).
    4. last_cycle_execution_status prefers exchange-confirmed resting limit price over unrounded requested limit_price.
    """
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import apply_asset_scope_filter
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        reconcile_spot_and_orderbook_microstructure,
        select_reconciled_spot_or_ob_price,
    )
    from orchestratorOS.supervisor.prompt_sections import (
        _format_microstructure_section,
        _format_portfolio_section,
    )
    from providers.base import MarketDataProvider
    from providers.market_api import MarketApi

    now_s = 1_795_107_000.0

    # 1. Mixed-case xStock Depth & Spot Flow URLs isolate exclusively to Kraken tokenized_asset
    for sym in ("SPCXxUSD", "ANTxUSD"):
        urls, hl_url = MarketDataProvider.resolve_orderbook_depth_urls(sym)
        assert hl_url == ""
        assert urls == (f"https://api.kraken.com/0/public/Depth?pair={sym}&count=200&asset_class=tokenized_asset",)
        flow_spec, _ = MarketDataProvider.resolve_spot_flow_15m_spec(sym, venue="kraken", now_ts=now_s)
        assert flow_spec is not None and f"pair={sym}" in flow_spec[0] and "asset_class=tokenized_asset" in flow_spec[0]

    # OrderbookDepthCollector preserves mixed-case wire symbol for both Depth and Spot Trades and backs off empty pre-IPO book even with stale cache
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "state").mkdir(parents=True, exist_ok=True)
    col = OrderbookDepthCollector(cache_dir=str(cdir), start_sync=True)
    fetched_urls: list[str] = []

    def _fake_depth(url, kind, sym, **kw):
        fetched_urls.append(url)
        if "pair=SPCXxUSD" in url:
            return {
                "error": [],
                "result": {
                    "SPCXxUSD": {
                        "bids": [["139.95", "100.0", 1], ["138.50", "500.0", 1]],
                        "asks": [["139.99", "100.0", 1], ["141.50", "600.0", 1]],
                    }
                },
            }
        return {"error": [], "result": {"ANTxUSD": {"asks": [], "bids": []}}}

    col._fetch_json_dict = _fake_depth
    col._fetch_spot_cvd_payload = lambda url, post_body: {
        "error": [],
        "result": {"SPCXxUSD": [["139.97", "20.0", now_s - 60.0, "b", "m", ""]], "last": "1"},
    }
    snap_spcx = col.fetch("SPCXxUSD", allow_network=True, now=now_s)
    assert snap_spcx is not None and snap_spcx.mid_price == pytest.approx(139.97)
    assert snap_spcx.spot_cvd_15m_usd == pytest.approx(2799.0)
    assert snap_spcx.spot_vol_15m_usd == pytest.approx(2799.0)
    assert all("api.kraken.com" in u and "pair=SPCXxUSD" in u and "asset_class=tokenized_asset" in u for u in fetched_urls)

    # Pre-IPO empty ANTxUSD book triggers quiet backoff even if an older (>600s) stale cache exists
    col._commit_snapshot(
        "ANTxUSD",
        now_s - 700.0,
        OrderbookSnapshot(
            symbol="ANTxUSD", mid_price=107.125, bid_depth_usd=1000.0, ask_depth_usd=1000.0,
            imbalance_ratio=0.5, largest_bid_wall_usd=500.0, largest_bid_wall_price=106.0,
            largest_ask_wall_usd=500.0, largest_ask_wall_price=108.0, ts=now_s - 700.0,
        ),
    )
    fetched_urls.clear()
    col.fetch("ANTxUSD", allow_network=True, now=now_s)
    assert fetched_urls == ["https://api.kraken.com/0/public/Depth?pair=ANTxUSD&count=200&asset_class=tokenized_asset"]
    assert col.is_unlisted_xstock_backoff_active("ANTxUSD", now_s) is True

    # MarketApi.fetch_historical_ohlc_points routes tokenized xStocks exclusively to Kraken (never Binance)
    ohlc_calls: list[tuple[str, str]] = []

    class _FakeBinance:
        name = "Binance"
        _ASSET_SCOPE = "crypto"
        _HISTORICAL_OHLC_PRIORITY = 1
        _SUPPORTS_TOKENIZED_ASSETS = False

        def fetch_historical_ohlc_points(self, sym, days, interval="1h"):
            ohlc_calls.append(("binance", sym))
            raise AssertionError("Binance must not be queried for Kraken tokenized xStocks")

    class _FakeKraken:
        name = "Kraken"
        _ASSET_SCOPE = "crypto"
        _HISTORICAL_OHLC_PRIORITY = 2
        _SUPPORTS_TOKENIZED_ASSETS = True

        def fetch_historical_ohlc_points(self, sym, days, interval="1h"):
            ohlc_calls.append(("kraken", sym))
            return [(now_s - 3600.0, 140.0), (now_s, 139.97)] if sym == "SPCXxUSD" else []

    fake_api = MarketApi(providers=[_FakeBinance(), _FakeKraken()])
    assert len(fake_api.fetch_historical_ohlc_points("SPCXxUSD", 7, "1h")) == 2
    assert fake_api.fetch_historical_ohlc_points("ANTxUSD", 7, "1h") == []
    assert ohlc_calls == [("kraken", "SPCXxUSD"), ("kraken", "ANTxUSD")]

    # 2. Frozen orderbook (>60s xStock, >300s fiat USD) blocks wall rescaling, blocks stale adoption, and suppresses Section 3B walls/imbalance/CVD
    m_spcx_stale = {
        "ob_mid_price": 144.52, "best_bid": 144.50, "best_ask": 144.54,
        "bid_wall_px": 143.01, "bid_wall_usd": 12000.0,
        "ask_wall_px": 146.50, "ask_wall_usd": 15000.0,
        "ob_imbalance": 0.468, "bid_depth_usd": 50000.0, "ask_depth_usd": 56000.0,
        "spot_cvd_15m_usd": -2600.0, "spot_cvd_source": "real_spot",
        "_ob_raw_ts": now_s - 5500.0,
    }
    _, _, act_stale = reconcile_spot_and_orderbook_microstructure(
        "SPCXxUSD", 139.97, m_spcx_stale,
        is_usd_sym=True, is_eq_sym=False, is_exact_ob_sym=True,
        ob_ts_sec=now_s - 5500.0, now_ts=now_s,
        last_series_pt=((now_s - 5.0) * 1000.0, 139.97),
    )
    assert act_stale == "none"
    assert m_spcx_stale["bid_wall_px"] == 143.01 and m_spcx_stale["ask_wall_px"] == 146.50
    assert select_reconciled_spot_or_ob_price(0.0, 107.125, now_s - 600.0, now_s, is_exact_ob_sym=True) is None
    sec3b_frozen = "\n".join(
        _format_microstructure_section({"SPCXxUSD": m_spcx_stale}, {"SPCXxUSD": {"price": 139.97}}, now_ts=now_s)
    )
    assert "orderbook=STALE_FROZEN(5500s; walls/imbalance/spot_cvd_15m suppressed)" in sec3b_frozen
    assert "bid_wall=" not in sec3b_frozen and "ask_wall=" not in sec3b_frozen and "spot_cvd_15m=" not in sec3b_frozen

    # 3. Total Binance Portfolio Value explicitly labels persisted BNB fee reserve on live_reconciled & strips in equities scope
    ps_bnb = {
        "total_value_usdc": 10032.06,
        "total_value_usdc_age_min": 1.2,
        "live_binance_equity_usdc": 9838.06,
        "binance_bnb_fee_reserve_usdc": 194.0,
    }
    sec3c_bnb = "\n".join(_format_portfolio_section(ps_bnb, asset_scope="crypto", include_macro_benchmarks=False))
    assert "live_reconciled=$9,838.06 USDC [excl. ~$194.00 BNB fee reserve]" in sec3c_bnb
    eq_ps = dict(ps_bnb)
    apply_asset_scope_filter({}, {}, eq_ps, "equities", {}, {})
    assert "binance_bnb_fee_reserve_usdc" not in eq_ps

    # 4. last_cycle_execution_status prefers exchange-confirmed resting price (190.8) over unrounded requested limit_price (190.85)
    st_tao = build_last_cycle_execution_status(
        [
            {
                "timestamp": now_s - 30.0,
                "decision": "BUY",
                "symbol": "TAOUSDC",
                "venue": "binance",
                "order_type": "LIMIT",
                "limit_price": 190.85,
                "reference_price": 191.2,
                "suggested_notional_usd": 500.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "726658530", "status": "NEW", "price": "190.80000000"},
            }
        ],
        "crypto",
        now_s,
        extract_intent_notional,
        open_orders=[{"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "order_id": "726658530", "price": 190.8}],
    )
    assert st_tao is not None and st_tao["limit_px"] == pytest.approx(190.8)

    # 5. Reanalysis polish: ensure_fresh_symbol_microstructure recognizes mixed-case orderbook_depth_SPCXxUSD.json without redundant refetch,
    # and should_rescale_usd_sibling_walls refuses rescaling severely stale books (>60s xStock, >600s STALE_FROZEN fiat USD)
    from intelligence.sentiment.collectors.microstructure_collector import ensure_fresh_symbol_microstructure
    from intelligence.sentiment.evaluators.microstructure_evaluator import should_rescale_usd_sibling_walls

    assert should_rescale_usd_sibling_walls("TAOUSD", 196.0, 200.0, 196.0, ob_ts=now_s - 700.0, now_ts=now_s) is False
    assert should_rescale_usd_sibling_walls("TAOUSD", 196.0, 200.0, 196.0, ob_ts=now_s - 120.0, now_ts=now_s) is True
    assert should_rescale_usd_sibling_walls("SPCXxUSD", 139.97, 144.52, 139.97, ob_ts=now_s - 90.0, now_ts=now_s) is False
    fresh_ob_path = cdir / "telemetry" / "orderbook_depth_SPCXxUSD.json"
    assert fresh_ob_path.exists()
    os.utime(fresh_ob_path, (now_s - 5.0, now_s - 5.0))
    assert ensure_fresh_symbol_microstructure(str(cdir), "SPCXxUSD", now_ts=now_s, max_age_ob_sec=180.0, allow_network=True) is False


def test_ghost_cvd_delta_macro_bar_age_closed_equity_intraday_and_agg_spot_flow(tmp_path):
    """Verify ghost d_cvd suppression, macro benchmark bar_age/CLOSED, closed US equity delta pruning, rvol_15m=0.00x, and agg spot CVD."""
    from intelligence.sentiment.collectors.macro_telemetry_collector import MacroAndNewsSensorsCollector
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        enrich_microstructure_cvd,
        evaluate_microstructure_cycle_deltas,
        format_spot_volume_rvol_parts,
    )

    now_s = 1_795_000_000.0
    bar_ts_ms = int((now_s - 180.0) * 1000)

    # 1. Ghost d_cvd from mark_px jitter on same upstream Binance 5m bar is suppressed (both explicit taker_bar_ts_ms and legacy fallback)
    hist_same_bar = {
        "BTCUSDC": [
            {
                "ts": now_s - 360.0,
                "ob_imbalance": 0.55,
                "cvd_15m_usd": 8_950_000.0,
                "whale_long_pct": 0.65,
                "oi_usd": 5_000_000_000.0,
                "taker_buy_sell_ratio": 1.0915,
                "taker_bar_ts_ms": bar_ts_ms,
            }
        ],
        "ARBUSDC": [
            {
                "ts": now_s - 360.0,
                "ob_imbalance": 0.50,
                "cvd_15m_usd": 4_159_566.0,
                "whale_long_pct": 0.60,
                "oi_usd": 50_000_000.0,
                "taker_buy_sell_ratio": 1.1234,
            }
        ],
    }
    micro_jitter = {
        "BTCUSDC": {
            "ob_imbalance": 0.57,
            "cvd_15m_usd": 8_956_500.0,  # +$6.5K jitter on exact same bar_ts_ms
            "whale_long_pct": 0.65,
            "oi_usd": 5_000_000_000.0,
            "taker_buy_sell_ratio": 1.0915,
            "_taker_bar_ts_ms": bar_ts_ms,
        },
        "ARBUSDC": {
            "ob_imbalance": 0.51,
            "cvd_15m_usd": 4_156_647.0,  # -$2,919 (<2%) jitter with identical 4-dec taker_buy_sell_ratio
            "whale_long_pct": 0.60,
            "oi_usd": 50_000_000.0,
            "taker_buy_sell_ratio": 1.1234,
        },
    }
    out_m, out_hist, _ = evaluate_microstructure_cycle_deltas(micro_jitter, hist_same_bar, now_ts=now_s)
    assert out_m["BTCUSDC"]["cycle_delta"]["cvd_15m_delta_usd"] == 0.0
    assert out_m["ARBUSDC"]["cycle_delta"]["cvd_15m_delta_usd"] == 0.0
    assert out_hist["BTCUSDC"][-1]["taker_bar_ts_ms"] == bar_ts_ms
    sec_3b_no_ghost = "\n".join(
        _format_microstructure_section(out_m, {"BTCUSDC": {"price": 62000.0}, "ARBUSDC": {"price": 0.12}}, now_ts=now_s)
    )
    assert "d_cvd=" not in sec_3b_no_ghost

    # 2. Macro benchmarks expose real bar_age_min and mark non-prepost assets (US10Y) CLOSED when stale >45m
    mb_raw = {
        "ts": now_s - 60.0,
        "items": {
            "ES=F": {
                "price": 6500.0,
                "unit": "usd",
                "chg_24h_pct": 0.45,
                "chg_1h_pct": 0.12,
                "chg_15m_pct": 0.04,
                "last_bar_ts": int(now_s - 600.0),  # 10m Yahoo delay
            },
            "US10Y": {
                "price": 4.25,
                "unit": "yield_pct",
                "chg_24h_pct": 0.10,
                "chg_1h_pct": 0.05,
                "chg_15m_pct": 0.02,
                "last_bar_ts": int(now_s - 12.8 * 3600.0),  # 12.8h old without market_state
            },
        },
    }
    mb_norm = MacroAndNewsSensorsCollector._normalize_macro_benchmarks_items(
        mb_raw, cache_dir=str(tmp_path), now_ts=now_s
    )
    assert mb_norm["items"]["ES=F"]["bar_age_min"] == pytest.approx(10.0)
    assert mb_norm["items"]["US10Y"]["market_state"] == "CLOSED"
    assert "chg_15m_pct" not in mb_norm["items"]["US10Y"]
    assert "chg_1h_pct" not in mb_norm["items"]["US10Y"]
    mb_line = format_macro_benchmarks_summary_line(mb_norm)
    assert "ES=F=$6,500.00 (+0.45% 24h, +0.12% 1h, +0.04% 15m, bar_age=10m)" in mb_line
    assert "US10Y=4.25% (+0.10% 24h, closed, bar_age=12.8h)" in mb_line

    # 3. Closed US equity (quote_age_min > 30) strips stale chg_15m_pct / chg_1h_pct in CrossMarketContextEvaluator
    cm_closed = build_cross_market_context(
        {
            "NVDA_US_EQ": {
                "price": 180.0,
                "quote_source": "t212_rth_close",
                "quote_age_min": 698.8,
                "chg_15m_pct": 5.23,
                "chg_30m_pct": 4.10,
                "chg_1h_pct": 1.47,
                "chg_24h_pct": 2.15,
            }
        },
        {},
        {},
        macro_benchmarks={
            "items": {
                "SPY": {"price": 600.0, "chg_24h_pct": 0.2, "market_state": "CLOSED"},
                "ES=F": {"price": 605.0, "chg_24h_pct": 1.0, "globex_session_chg_pct": 0.8},
            }
        },
    )
    nvda_cb = cm_closed["equity_bellwethers"]["NVDA_US_EQ"]
    assert "chg_15m_pct" not in nvda_cb
    assert "chg_30m_pct" not in nvda_cb
    assert "chg_1h_pct" not in nvda_cb
    assert nvda_cb["chg_24h_pct"] == pytest.approx(2.15)
    assert nvda_cb["futures_lead_pct"] == pytest.approx(0.8)

    # 4. Thin USDC pair uses agg_spot_cvd_15m_usd for spot_perp_flow_regime, shows rvol_15m=0.00x, and renders agg_spot_cvd_15m
    arb_m = {
        "ob_imbalance": 0.56,
        "bid_depth_usd": 50_000.0,
        "ask_depth_usd": 40_000.0,
        "cvd_15m_usd": 80_000.0,
        "spot_cvd_15m_usd": 6.0,
        "spot_cvd_source": "real_spot",
        "spot_vol_15m_usd": 6.0,
        "agg_spot_cvd_15m_usd": 120_000.0,
        "agg_spot_vol_15m_usd": 250_000.0,
        "agg_spot_cvd_source": "usdc+usdt",
    }
    enrich_microstructure_cvd(arb_m, {"price": 0.12, "chg_15m_pct": 0.4, "spot_vol_24h_avg_15m_usd": 9550.0})
    assert arb_m["rvol_15m"] == 0.0
    assert arb_m["spot_perp_flow_regime"] == "SYNCHRONIZED_SPOT_PERP_BUYING"
    vol_parts = format_spot_volume_rvol_parts(arb_m, lambda v: f"${v:,.0f}")
    assert any("rvol_15m=0.00x" in p for p in vol_parts)
    sec_3b_arb = "\n".join(_format_microstructure_section({"ARBUSDC": arb_m}, {"ARBUSDC": {"price": 0.12}}, now_ts=now_s))
    assert "agg_spot_cvd_15m=+$120.0K[usdc+usdt, vol=$250.0K]" in sec_3b_arb


def test_sentiment_advisor_quantitative_prompt_enrichment_geopolitical_context_and_no_labels(tmp_path):
    """Verify SentimentAdvisor & MacroAnalyzer prompt quality fixes:
    - Structured Whales & Derivatives with annualized funding, perp_spot_basis_bps, explicit 15m/cvd_15m, no raw dict/sci-notation/labels
    - Full quantitative MarketBreadthSnapshot fields (counts, mean, dispersion_std, ETH, ETH/BTC, vol_share) without regime labels
    - Enriched BTC Intraday Structure (15m/1h/6h/24h, bounce_24h_low, dd_24h_high, VP levels, bottom_6h, hl_vs_prior)
    - Factual Geopolitical Context (risk_score, age, summary) wired into SentimentAdvisor & MacroAnalyzer without brake labels
    - Balanced ACCUMULATE guidance and label-free Cross-Asset instructions in SentimentAdvisor & GeopoliticalThreatAnalyzer
    """
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.macro_sentiment_ssot import (
        MarketBreadthSnapshot,
        load_cached_derivatives_telemetry,
    )

    cdir = tmp_path / "cachedb"
    (cdir / "macro").mkdir(parents=True, exist_ok=True)
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    (cdir / "prices").mkdir(parents=True, exist_ok=True)
    now_s = 1_795_167_600.0
    now_ms = int(now_s * 1000)

    (cdir / "telemetry" / "whale_snapshot_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "top_traders_long_ratio": 1.8194,
                "top_traders_long_pct": 0.6453,
                "taker_buy_sell_ratio": 0.6538,
                "taker_buy_vol_usd": 43_599_943.94,
                "taker_sell_vol_usd": 66_683_670.12,
                "open_interest_usd": 5_965_272_558.644072,
                "open_interest_1h_change_pct": -0.93,
                "divergence_regime": "long_unwinding",
                "ts": now_s - 20.0,
                "funding_rate": -1.8e-06,
                "taker_ratio_5m": 0.6361,
                "taker_ratio_1h": 0.849,
                "taker_ratio_4h": 1.0186,
                "cvd_5m_usd": -8_169_535.0,
                "cvd_1h_usd": -33_507_458.0,
                "cvd_4h_usd": 17_555_590.0,
            }
        ),
        encoding="utf-8",
    )
    (cdir / "telemetry" / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "funding_rate": -1.8e-06,
                "open_interest_usd": 5_965_557_327.92,
                "perp_spot_basis_bps": -4.31,
                "funding_annualized_pct": -0.2,
                "ts": now_s - 20.0,
            }
        ),
        encoding="utf-8",
    )
    (cdir / "macro" / "geopolitical_eval.json").write_text(
        json.dumps(
            {
                "threat_level": "ELEVATED",
                "risk_score": 0.66,
                "recommended_brake": "DOWNSCALE_50",
                "headlines_analyzed": 7,
                "summary": "Missile barrage on Kyiv damaged embassies alongside Houthi attacks in Riyadh.",
                "ts": now_s - 1800.0,
            }
        ),
        encoding="utf-8",
    )
    (cdir / "macro" / "market_breadth_collect.json").write_text(
        json.dumps(
            {
                "advance_ratio": 0.1186,
                "advancing_count": 7,
                "declining_count": 52,
                "total_count": 59,
                "median_change_pct": -4.388,
                "mean_change_pct": -3.448,
                "dispersion_std": 9.36,
                "regime": "PANIC_WASHOUT",
                "eth_price": 1743.71,
                "eth_chg_24h_pct": -3.41,
                "eth_btc_ratio": 0.02832,
                "eth_btc_chg_24h_pct": 0.07,
                "btc_vol_share_pct": 44.9,
                "eth_vol_share_pct": 21.72,
                "alt_vol_share_pct": 33.38,
                "rotation_regime": "BALANCED_ROTATION",
                "ts": now_s - 60.0,
            }
        ),
        encoding="utf-8",
    )
    # Build 24h BTC series with a clear 6h drawdown (>1.5%), higher 1h low, and multi-horizon moves
    btc_pts = []
    for mins_ago in range(24 * 60, -1, -5):
        tms = now_ms - mins_ago * 60_000
        if mins_ago > 360:
            px = 64000.0 - (24 * 60 - mins_ago) * 1.4
        elif mins_ago > 60:
            px = 62500.0 if mins_ago >= 300 else (61327.0 if mins_ago == 180 else 61800.0)
        else:
            px = 61550.0 if mins_ago == 30 else 61750.0
        btc_pts.append([tms, px])
    (cdir / "prices" / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": btc_pts}}),
        encoding="utf-8",
    )

    loaded_deriv = load_cached_derivatives_telemetry(str(cdir), symbols=["BTCUSDC"])
    assert isinstance(loaded_deriv, dict)
    assert loaded_deriv["perp_spot_basis_bps"] == pytest.approx(-4.31)
    assert loaded_deriv["funding_annualized_pct"] == pytest.approx(-0.2)

    dummy_client = DummyGeminiClient(
        {
            "market_bias": "CAUTION",
            "risk_level": "HIGH",
            "confidence": 0.80,
            "summary": "Bearish macro and derivatives flow.",
            "key_risks": ["Geopolitical escalation"],
            "recommended_action": "HOLD",
        }
    )
    ma = MacroAnalyzer(cache_dir=str(cdir), symbols=["BTCUSDC"])
    ma.sentiment_advisor.gemini_client = dummy_client
    ma.gemini_advisor = ma.sentiment_advisor
    assert ma.assess_market_advisor(force=True, now=now_s) is True

    prompt = dummy_client.last_prompt
    assert prompt
    # 1. Breadth includes quantitative detail + ETH/BTC + volume shares without heuristic labels
    assert "11.9% advancing (7/59), median -4.39%, mean -3.45%, dispersion_std 9.36%" in prompt
    assert "ETH=$1,743.71 (-3.41% 24h), ETH/BTC=0.02832 (+0.07% 24h)" in prompt
    assert "vol_share(BTC 44.9% / ETH 21.7% / ALT 33.4%)" in prompt
    assert "PANIC_WASHOUT" not in prompt
    assert "BALANCED_ROTATION" not in prompt

    # 2. Whales & Derivatives is clean, structured, annualized, with basis + explicit 15m/CVD, no raw dict/sci-notation
    assert "- Whales & Derivatives: BTCUSDC | OI=$5.97B (-0.93% 1h) | top_long=64.5% (ratio=1.82)" in prompt
    assert "funding=-0.20% ann | perp_spot_basis=-4.31bps" in prompt
    assert "5m(tr=0.64, cvd=-$8.17M)" in prompt
    assert "15m(tr=0.65, cvd=-$23.08M)" in prompt
    assert "1h(tr=0.85, cvd=-$33.51M)" in prompt
    assert "4h(tr=1.02, cvd=+$17.56M)" in prompt
    assert "e-06" not in prompt
    assert "'ts':" not in prompt
    assert "long_unwinding" not in prompt

    # 3. Geopolitical Context is present factually without brake/threat directive labels
    assert "- Geopolitical Context: risk_score=0.66 (age=30m, headlines=7) | summary=Missile barrage on Kyiv" in prompt
    assert "DOWNSCALE_50" not in prompt
    assert "ELEVATED" not in prompt

    # 4. BTC Intraday Structure exposes multi-horizon changes, range geometry, VP levels, and bottom_6h + hl_vs_prior
    assert "- BTC Intraday Structure: price=$61750.00 (15m " in prompt
    assert ", 1h " in prompt
    assert ", 6h " in prompt
    assert "bounce_24h_low=" in prompt
    assert "dd_24h_high=" in prompt
    assert "VP(VAL=$" in prompt
    assert "bottom_6h(prior_6h_low=$61327.00, 1h_low=$61550.00)" in prompt
    assert "hl_vs_prior=+0.36%" in prompt

    # 5. Balanced framework: no nonexistent fields, no stale heuristic label tokens, no unconditional ACCUMULATE shift
    for stale_tok in (
        "advancing_pct",
        "avg_change_24h_pct",
        "bounce_from_low_pct",
        "FUTURES_PREMARKET_GAP_UP",
        "FUTURES_PREMARKET_GAP_DOWN",
        "COUPLED_RISK_ON",
        "CRYPTO_BULLISH_DECOUPLING",
        "EQUITY_BULLISH_DECOUPLING",
        "shift recommended_action to 'ACCUMULATE'",
    ):
        assert stale_tok not in prompt

    # 6. GeopoliticalThreatAnalyzer prompt also avoids stale heuristic cross-asset regime labels
    geo_client = DummyGeminiClient(
        {
            "threat_level": "NORMAL",
            "risk_score": 0.15,
            "summary": "Calm markets.",
            "recommended_brake": "NONE",
        }
    )
    geo = GeopoliticalThreatAnalyzer(gemini_client=geo_client, state_file=str(cdir / "macro" / "geo_test.json"))
    geo.assess(
        {
            "headlines": [
                {
                    "title": "Regional sanctions update reported",
                    "source": "Reuters",
                    "published_ts": now_s - 600.0,
                    "matched_keywords": ["sanctions"],
                    "is_high_severity": True,
                }
            ],
            "high_severity_count": 1,
            "has_critical_shock_keywords": False,
            "ts": now_s,
        },
        force_refresh=True,
        now_ts=now_s,
    )
    for stale_geo_tok in ("COUPLED_RISK_ON", "CRYPTO_BULLISH_DECOUPLING", "FUTURES_PREMARKET_GAP_UP", "Cross_Asset"):
        assert stale_geo_tok not in geo_client.last_prompt

    # 7. Section 3B/3C prompt hygiene: deduplicate raw sci-notation funding when funding_ann present,
    #    format dust free_base without sci-notation, drop raw intent ts when age_h present, and drop [HEALTHY]/[CAUTION] labels
    from providers.ai_survival_guard import format_ai_decision_scorecard_line
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section

    sec_3b_clean = "\n".join(
        _format_microstructure_section(
            {
                "BTCUSDC": {
                    "ob_imbalance": 0.65,
                    "bid_depth_usd": 5_000_000.0,
                    "ask_depth_usd": 3_000_000.0,
                    "funding_rate": -4.24e-06,
                    "pred_funding_rate": -4.24e-06,
                    "funding_annualized_pct": -0.5,
                    "perp_spot_basis_bps": -4.8,
                    "oi_usd": 5_980_803_551.77,
                    "oi_1h_chg_pct": -0.54,
                    "whale_long_pct": 0.645,
                    "taker_buy_sell_ratio": 1.15,
                    "taker_buy_vol_usd": 45_000_000.0,
                    "taker_sell_vol_usd": 39_000_000.0,
                    "cvd_15m_usd": 6_000_000.0,
                }
            },
            {"BTCUSDC": {"price": 61_678.0}},
            now_ts=now_s,
        )
    )
    assert "funding_ann=-0.5%" in sec_3b_clean
    assert "e-06" not in sec_3b_clean
    assert "(pred=" not in sec_3b_clean

    sc_ok = format_ai_decision_scorecard_line(
        {"count": 4, "wins": 4, "scratches": 0, "losses": 0, "avg_edge_pct": 2.81, "net_usd": 49.0, "status": "HEALTHY"}
    )
    assert sc_ok == "AI Decision Scorecard (48h, n=4): 4W/0S/0L, avg=+2.81%, net=+$49"
    assert "[HEALTHY]" not in sc_ok
    sc_cd = format_ai_decision_scorecard_line(
        {
            "count": 3, "wins": 0, "scratches": 0, "losses": 3, "avg_edge_pct": -1.5, "net_usd": -120.0,
            "loss_streak": 3, "status": "BUY_SHADOW_COOLDOWN", "cooldown_remaining_h": 2.4,
        }
    )
    assert "[BUY_SHADOW_COOLDOWN: 2.4h left]" in sc_cd

    sec_3c_clean = "\n".join(
        _format_portfolio_section(
            {
                "tradable_positions": [
                    {
                        "venue": "kraken",
                        "symbol": "TAOUSD",
                        "base": "TAO",
                        "quote": "USD",
                        "free_base": 3e-06,
                        "value_usd": 0.0,
                        "free_quote": 4478.99,
                        "price": 186.4,
                    }
                ],
                "recent_ai_intents": [
                    {
                        "decision": "SELL",
                        "symbol": "TAOUSDC",
                        "venue": "binance",
                        "order_type": "LIMIT",
                        "status": "EXECUTED",
                        "notional_usd": 450.0,
                        "ts": 1795158520.6662192,
                        "venue_status": "FILLED",
                        "age_h": 2.5,
                    }
                ],
            }
        )
    )
    assert "free_TAO=0.000003 ($0.0 USD)" in sec_3c_clean
    assert "3e-06" not in sec_3c_clean
    assert "'ts':" not in sec_3c_clean
    assert "'age_h': 2.5" in sec_3c_clean

    from intelligence.sentiment.sentiment_advisor import _format_geopolitical_line

    long_geo = _format_geopolitical_line(
        {"risk_score": 0.68, "headlines_analyzed": 7, "ts": now_s - 600.0, "summary": "word " * 120 + "Coinbase"},
        now_s,
    )
    assert long_geo is not None and long_geo.endswith("word...")




def test_etapa0_derivatives_short_horizon_deltas_and_rendering(tmp_path):
    """Verify Q1 oi_15m_chg_pct, basis_delta_15m_bps, funding_delta_1h_bps, 90-entry history retention, sibling sync, and rendering."""
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import PretradeOrderbookCvdAndLiqStage
    from intelligence.sentiment.evaluators.microstructure_evaluator import evaluate_microstructure_cycle_deltas

    now_s = 1_795_200_000.0

    # 1. load_symbol_microstructure extracts oi_15m_chg_pct from whale snapshot
    loaded = load_symbol_microstructure(
        str(tmp_path),
        "BTCUSDC",
        whale={"open_interest_1h_change_pct": 2.4, "open_interest_15m_change_pct": 1.15},
    )
    assert loaded["oi_1h_chg_pct"] == pytest.approx(2.4)
    assert loaded["oi_15m_chg_pct"] == pytest.approx(1.15)

    # 2. evaluate_microstructure_cycle_deltas retains >12 (up to 90) 1m snapshots and computes 15m basis + 1h funding deltas
    hist_1m = []
    for mins_ago in range(70, 0, -1):
        rec = {
            "ts": now_s - mins_ago * 60.0,
            "ob_imbalance": 0.52,
            "cvd_15m_usd": 100_000.0,
            "whale_long_pct": 0.60,
            "oi_usd": 5_000_000_000.0,
            "funding_rate": 0.00005 if mins_ago >= 60 else 0.00010,
            "perp_spot_basis_bps": 1.0 if mins_ago > 15 else 2.0,
        }
        hist_1m.append(rec)

    micro_map = {
        "BTCUSDC": {
            "ob_imbalance": 0.58,
            "bid_depth_usd": 1_000_000.0,
            "ask_depth_usd": 800_000.0,
            "cvd_15m_usd": 500_000.0,
            "whale_long_pct": 0.62,
            "taker_buy_sell_ratio": 1.25,
            "oi_usd": 5_100_000_000.0,
            "oi_1h_chg_pct": 2.4,
            "oi_15m_chg_pct": 1.15,
            "funding_rate": 0.00015,
            "pred_funding_rate": 0.00016,
            "funding_annualized_pct": 16.4,
            "perp_spot_basis_bps": 6.5,
        },
        "BTCUSD": {
            "ob_imbalance": 0.54,
            "bid_depth_usd": 200_000.0,
            "ask_depth_usd": 180_000.0,
            "ob_mid_price": 62005.0,
            "spread_bps": 2.0,
        },
    }
    out_m, out_hist, _ = evaluate_microstructure_cycle_deltas(micro_map, {"BTCUSDC": hist_1m}, now_ts=now_s)
    assert len(out_hist["BTCUSDC"]) == 71
    assert out_hist["BTCUSDC"][-1]["funding_rate"] == pytest.approx(0.00015)
    assert out_hist["BTCUSDC"][-1]["perp_spot_basis_bps"] == pytest.approx(6.5)
    assert out_m["BTCUSDC"]["basis_delta_15m_bps"] == pytest.approx(4.5)
    assert out_m["BTCUSDC"]["funding_delta_1h_bps"] == pytest.approx(1.0)

    # 3. Sibling sync propagates oi_15m_chg_pct, basis_delta_15m_bps, and funding_delta_1h_bps to BTCUSD
    prices_map = {"BTCUSDC": {"price": 62000.0}, "BTCUSD": {"price": 62005.0}}
    sync_sibling_symbol_telemetry(prices_map, out_m)
    assert out_m["BTCUSD"]["oi_15m_chg_pct"] == pytest.approx(1.15)
    assert out_m["BTCUSD"]["basis_delta_15m_bps"] == pytest.approx(4.5)
    assert out_m["BTCUSD"]["funding_delta_1h_bps"] == pytest.approx(1.0)

    # 4. Section 3B and Pretrade telemetry render compact delta suffixes
    sec_3b = "\n".join(_format_microstructure_section(out_m, prices_map, now_ts=now_s))
    assert "perp_basis=+6.5bps(d15m=+4.5bps)" in sec_3b
    assert "funding_ann=+16.4%(d1h=+1.00bps)" in sec_3b
    assert "oi_15m_chg=+1.15%" in sec_3b

    pt_tel = {}
    PretradeOrderbookCvdAndLiqStage._format_derivatives_and_whale_lines(
        pt_tel, out_m["BTCUSDC"], is_eq_sym=False, is_tok_xstock_pt=False
    )
    assert "fund_d1h=+1.00bps" in pt_tel["Perpetual Funding Rate"]
    assert "basis_d15m=+4.5bps" in pt_tel["Perpetual Funding Rate"]
    assert "15m_chg=+1.15%" in pt_tel["Derivatives Open Interest"]


def test_faza1_multi_horizon_atr_session_vwap_and_intraday_extremes(tmp_path):
    """Verify Faza 1: multi-horizon Wilder ATR (15m/1h/4h), UTC/RTH Session VWAP + bands/z, Asia/London/NY extremes, sibling isolation, and rendering."""
    from types import SimpleNamespace
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import Pretrade24hAndBtcFormatStage
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        build_compact_4h_ohlc_bars,
        compute_atr_multi_horizon,
        compute_intraday_session_extremes,
        compute_session_vwap_metrics,
    )

    # Anchor at 2026-11-20 15:00:00 UTC (Asia closed, London active, NY open 90m)
    day_utc_00_ms = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    now_utc_15_ms = datetime(2026, 11, 20, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0

    # 1. Crypto 24h dense series (1m steps over 24h) + 7d archive -> compact 4h bars & monotonic ATR(15m < 1h < 4h)
    arch_7d = []
    for h_back in range(168, 0, -1):
        t_h = now_utc_15_ms - h_back * 3_600_000.0
        base_p = 60000.0 + math.sin(h_back * 0.25) * 1400.0
        arch_7d.append([t_h, round(base_p, 2)])
        arch_7d.append([t_h + 1_800_000.0, round(base_p + 280.0 * (1 if h_back % 2 == 0 else -1), 2)])
    bars_4h_7d = build_compact_4h_ohlc_bars(arch_7d, [], [])
    assert 35 <= len(bars_4h_7d) <= 46
    assert all(len(r) == 5 and r[3] >= r[1] >= r[4] for r in bars_4h_7d)

    crypto_24h = []
    for m_back in range(24 * 60, -1, -1):
        t_m = now_utc_15_ms - m_back * 60_000.0
        px_m = 62000.0 + math.sin(m_back * 0.08) * 350.0 + math.sin(m_back * 0.5) * 90.0
        # Inject distinctive session extremes inside today's Asia (02:00 UTC), London (10:00 UTC), NY (14:00 UTC)
        if abs(t_m - (day_utc_00_ms + 2 * 3_600_000.0)) < 30_000.0:
            px_m = 61100.0
        elif abs(t_m - (day_utc_00_ms + 5 * 3_600_000.0)) < 30_000.0:
            px_m = 62900.0
        elif abs(t_m - (day_utc_00_ms + 10 * 3_600_000.0)) < 30_000.0:
            px_m = 63150.0
        elif abs(t_m - (day_utc_00_ms + 14 * 3_600_000.0)) < 30_000.0:
            px_m = 61400.0
        crypto_24h.append([t_m, round(px_m, 2)])
    crypto_24h[-1][1] = 62100.0

    m24_btc = compute_24h_price_metrics(crypto_24h, is_equity=False, series_7d_ts_px=bars_4h_7d)
    assert m24_btc["atr_15m_pct"] > 0.0
    assert m24_btc["atr_1h_pct"] > m24_btc["atr_15m_pct"]
    assert m24_btc["atr_4h_pct"] > m24_btc["atr_1h_pct"]
    sess_ex = m24_btc["session_extremes"]
    assert sess_ex["asia"][0] == pytest.approx(61100.0)
    assert sess_ex["asia"][1] == pytest.approx(62900.0)
    assert sess_ex["london"][1] == pytest.approx(63150.0)
    assert sess_ex["ny"][0] == pytest.approx(61400.0)

    # At 06:00 UTC, only Asia is emitted (London 07:00 & NY 13:30 have not opened yet)
    early_asia_pts = [pt for pt in crypto_24h if pt[0] <= day_utc_00_ms + 6 * 3_600_000.0]
    early_sess = compute_intraday_session_extremes(early_asia_pts, is_equity=False).get("session_extremes") or {}
    assert set(early_sess.keys()) == {"asia"}

    # 2. Crypto Session VWAP anchored to 00:00 UTC (excludes prior UTC day bars) + live price geometry sync
    ohlc_crypto = [
        [day_utc_00_ms - 3_600_000.0, 50000.0, 9999.0],  # Yesterday UTC -- MUST be excluded
        [day_utc_00_ms + 900_000.0, 62000.0, 100.0],
        [day_utc_00_ms + 1_800_000.0, 62200.0, 100.0],
        [day_utc_00_ms + 2_700_000.0, 62400.0, 200.0],
    ]
    # Expected VWAP = (62000*100 + 62200*100 + 62400*200) / 400 = 62250.0
    vwap_c = compute_session_vwap_metrics(ohlc_crypto, is_equity=False, cur_px=62100.0)
    assert vwap_c["session_vwap_px"] == pytest.approx(62250.0)
    assert vwap_c["session_vwap_anchor"] == "utc_00"
    assert vwap_c["dist_vs_session_vwap_pct"] == pytest.approx(((62100.0 / 62250.0) - 1.0) * 100.0, abs=0.01)
    assert vwap_c["session_vwap_sigma_pct"] > 0.0
    assert vwap_c["session_vwap_z"] < 0.0
    m24_btc.update(vwap_c)

    # Live price geometry recomputes dist_vs_session_vwap_pct and flips z-score sign above VWAP
    recalculate_live_price_geometry(m24_btc, 62500.0)
    assert m24_btc["dist_vs_session_vwap_pct"] == pytest.approx(((62500.0 / 62250.0) - 1.0) * 100.0, abs=0.01)
    assert m24_btc["session_vwap_z"] > 0.0

    # Degenerate <2 bars after UTC midnight returns {}
    assert compute_session_vwap_metrics(
        [[day_utc_00_ms - 900_000.0, 61000.0, 50.0], [day_utc_00_ms + 300_000.0, 62000.0, 10.0]],
        is_equity=False,
    ) == {}

    # 3. US Equity RTH Session VWAP (09:30-16:00 NY), overnight gap immunity on intraday ATR, no crypto sessions, post-RTH override preservation
    # Day 1 RTH (Nov 18, 2026 EST = UTC-5 -> 14:30-21:00 UTC) around $150; Day 2 RTH (Nov 19) gaps +10% to $165
    d1_rth0 = datetime(2026, 11, 18, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    d2_rth0 = datetime(2026, 11, 19, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    eq_bars = []
    for i in range(78):  # 6.5h of 5m bars on Day 1 around $150 +/- $0.30
        eq_bars.append([d1_rth0 + i * 300_000.0, round(150.0 + (0.30 if i % 2 == 0 else -0.30), 4), 1_000_000.0])
    # Zero-volume pre-market tick on Day 2 before 09:30 NY
    eq_bars.append([d2_rth0 - 1_800_000.0, 166.0, 0.0])
    for i in range(78):  # 6.5h of 5m bars on Day 2 around $165 +/- $0.33 (~0.4% intrabar range)
        eq_bars.append([d2_rth0 + i * 300_000.0, round(165.0 + (0.33 if i % 2 == 0 else -0.33), 4), 2_000_000.0])
    # Zero-volume after-hours tick after 16:00 NY
    eq_bars.append([d2_rth0 + 79 * 300_000.0, 168.0, 0.0])

    m24_eq = compute_24h_price_metrics(eq_bars, is_equity=True)
    assert "session_extremes" not in m24_eq
    assert m24_eq["session_vwap_anchor"] == "rth"
    # Day 2 RTH mean is exactly $165.00 (excludes Day 1 $150 and zero-vol $166/$168 ticks)
    assert m24_eq["session_vwap_px"] == pytest.approx(165.0, abs=0.05)
    # Overnight +10% gap must NOT inflate 15m ATR to >2%
    assert 0.1 < m24_eq["atr_15m_pct"] < 1.0
    assert 0.1 < m24_eq["atr_1h_pct"] < 1.2

    # Post-RTH price override preserves RTH Session VWAP anchor
    eq_policy = resolve_asset_session_policy("NVDA_US_EQ")
    _, m24_eq_ov, _ = eq_policy.apply_post_rth_price_override(eq_bars, dict(m24_eq), 170.0)
    assert m24_eq_ov["session_vwap_px"] == pytest.approx(m24_eq["session_vwap_px"])
    assert m24_eq_ov["session_vwap_anchor"] == "rth"

    # 4. Sibling sync shares ATR across all quotes but isolates VWAP & session extremes between USD and EUR
    p_sib = {
        "BTCUSDC": {"price": 62500.0, **m24_btc},
        "BTCUSD": {"price": 62505.0},
        "BTCEUR": {"price": 57500.0},
    }
    sync_sibling_symbol_telemetry(p_sib, {})
    assert p_sib["BTCUSD"]["atr_1h_pct"] == pytest.approx(m24_btc["atr_1h_pct"])
    assert p_sib["BTCUSD"]["session_vwap_px"] == pytest.approx(62250.0)
    assert p_sib["BTCEUR"]["atr_1h_pct"] == pytest.approx(m24_btc["atr_1h_pct"])
    assert "session_vwap_px" not in p_sib["BTCEUR"]
    assert "session_extremes" not in p_sib["BTCEUR"]

    # 5. Section 3A & Pretrade telemetry render compact ATR, Session VWAP, and sessions
    sec_3a = "\n".join(_format_market_prices_section({"BTCUSDC": {"price": 62500.0, "vol_1h_pct": 0.91, **m24_btc}}))
    assert "session_vwap=$62,250.00 (+0.40%, z=" in sec_3a
    assert "utc_00)" in sec_3a
    assert "sessions(asia=[61100.0, 62900.0], ldn=" in sec_3a
    assert f"atr(15m={m24_btc['atr_15m_pct']:.2f}%, 1h={m24_btc['atr_1h_pct']:.2f}%, 4h={m24_btc['atr_4h_pct']:.2f}%)" in sec_3a
    assert "vol_1h=" not in sec_3a
    sec_3a_no_atr1h = "\n".join(_format_market_prices_section({"BTCUSDC": {"price": 62500.0, "vol_1h_pct": 0.91}}))
    assert "vol_1h=0.91%" in sec_3a_no_atr1h

    pt_ctx = SimpleNamespace(
        cachedb_dir=str(tmp_path),
        symbol="BTCUSDC",
        telemetry={},
        m24=m24_btc,
        alt_series_24h=crypto_24h,
        cur_p_items={},
        cur_px_est=62500.0,
        rj=lambda _p: {},
    )
    Pretrade24hAndBtcFormatStage().evaluate(pt_ctx)
    assert "Multi-Timeframe ATR" in pt_ctx.telemetry
    assert f"1h={m24_btc['atr_1h_pct']:.2f}%" in pt_ctx.telemetry["Multi-Timeframe ATR"]
    assert "Session VWAP & Intraday Structure" in pt_ctx.telemetry
    assert "Session VWAP=$62,250.00 (+0.40%" in pt_ctx.telemetry["Session VWAP & Intraday Structure"]
    assert "sessions(asia=[61100.0, 62900.0]" in pt_ctx.telemetry["Session VWAP & Intraday Structure"]


def test_faza1_ohlc_hlo_preservation_mixed_stitched_atr_and_typical_vwap():
    """Verify 6-col [ts, close, vol, high, low, open] preservation, mixed-series resampling, short 5m 7d guard, and HLC/3 VWAP."""
    from types import SimpleNamespace
    from providers.ohlc_utils import normalize_yahoo_5d_chart_bars
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _extract_ohlc_series_points,
        compute_atr_multi_horizon,
        compute_session_vwap_metrics,
    )

    day0_s = datetime(2026, 11, 20, 14, 30, 0, tzinfo=timezone.utc).timestamp()
    # 1. Yahoo dict payload preserves [ts_ms, close, vol, high, low, open]
    y_chart = {
        "timestamp": [day0_s + i * 300.0 for i in range(6)],
        "indicators": {
            "quote": [
                {
                    "open": [100.0 + i for i in range(6)],
                    "high": [102.0 + i for i in range(6)],
                    "low": [98.0 + i for i in range(6)],
                    "close": [101.0 + i for i in range(6)],
                    "volume": [1000.0 + i * 100.0 for i in range(6)],
                }
            ]
        },
        "meta": {"regularMarketPrice": 106.0},
    }
    pts_5d, pts_24h, _, _, _, _, _ = normalize_yahoo_5d_chart_bars(y_chart, [], day0_s + 3600.0, eq_rth_open=True)
    assert len(pts_5d) == 6
    assert pts_5d[0] == [math.trunc(day0_s * 1000), 101.0, 1000.0, 102.0, 98.0, 100.0]
    assert len(pts_24h) == 6 and len(pts_24h[-1]) == 6

    # Tuple payload stays 2-col / 3-col backwards-compatible when H/L absent and preserves 6-col when present
    pts_2c, _, _, _, _, _, _ = normalize_yahoo_5d_chart_bars(
        ({}, [[day0_s, 101.0], [day0_s + 300.0, 102.0]]), [], day0_s + 600.0
    )
    assert pts_2c == [[math.trunc(day0_s * 1000), 101.0], [math.trunc((day0_s + 300.0) * 1000), 102.0]]
    pts_6c, _, _, _, _, _, _ = normalize_yahoo_5d_chart_bars(
        ({}, [[day0_s, 101.0, 500.0, 103.0, 99.0, 100.5], [day0_s + 300.0, 102.0, 600.0, 104.0, 100.0, 101.0]]),
        [],
        day0_s + 600.0,
    )
    assert pts_6c[0] == [math.trunc(day0_s * 1000), 101.0, 500.0, 103.0, 99.0, 100.5]

    # 2. _extract_ohlc_series_points preserves H/L/O from OHLCSeries-like objects
    ohlc_obj = SimpleNamespace(
        timestamps=[day0_s, day0_s + 900.0],
        closes=[100.0, 101.0],
        volumes=[250.0, 300.0],
        highs=[103.0, 104.0],
        lows=[97.0, 98.0],
        opens=[99.0, 100.0],
    )
    extracted = _extract_ohlc_series_points(ohlc_obj, min_ts=day0_s - 60.0)
    assert extracted[0] == [round(day0_s * 1000.0), 100.0, 250.0, 103.0, 97.0, 99.0]
    assert extracted[1] == [round((day0_s + 900.0) * 1000.0), 101.0, 300.0, 104.0, 98.0, 100.0]

    # 3. Mixed stitched series (10 leading 2-col rows + 16 trailing 6-col rows) still uses intrabar H/L
    base_ms = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    mixed_rows = [[base_ms + i * 900_000.0, 100.0] for i in range(10)]
    for j in range(10, 26):
        # Close stays flat near 100.0 +/- 0.1, but intrabar H/L swings 103.0 / 97.0 (6% range)
        c_v = 100.1 if j % 2 == 0 else 99.9
        mixed_rows.append([base_ms + j * 900_000.0, c_v, 100.0, 103.0, 97.0, 100.0])
    close_only_rows = [[r[0], r[1]] for r in mixed_rows]
    atr_mixed = compute_atr_multi_horizon(mixed_rows, cur_px=100.0)
    atr_close_only = compute_atr_multi_horizon(close_only_rows, cur_px=100.0)
    assert atr_mixed["atr_15m_pct"] > 3.0
    assert atr_close_only["atr_15m_pct"] < 0.5

    # Short <= 60-bar 5m 6-col series_7d_ts_px is NOT misclassified as compact 4h and contributes to 1h ATR
    short_5m_7d = [
        [base_ms - (36 - k) * 300_000.0, 100.0, 50.0, 102.0, 98.0, 100.0]
        for k in range(36)
    ]  # 3 hours of 5m bars (36 <= 60, avg step 5m < 2h)
    short_24h = [
        [base_ms + k * 900_000.0, 100.0, 50.0, 102.0, 98.0, 100.0]
        for k in range(16)
    ]  # 4 hours of 15m bars -> alone only 4 hourly bars (<6), merged with 3h 5m history -> 7 hourly bars -> atr_1h_pct populated!
    assert "atr_1h_pct" not in compute_atr_multi_horizon(short_24h, cur_px=100.0)
    atr_with_short_7d = compute_atr_multi_horizon(short_24h, series_7d_ts_px=short_5m_7d, cur_px=100.0)
    assert atr_with_short_7d["atr_1h_pct"] == pytest.approx(4.0, abs=0.1)

    # 4. Session VWAP uses typical price (H + L + C) / 3 on 5/6-col OHLC rows
    # Row 0: H=106, L=100, C=100 -> TP=102.0, vol=100
    # Row 1: H=109, L=103, C=103 -> TP=105.0, vol=300
    # Expected VWAP = (102*100 + 105*300) / 400 = 104.25
    vwap_hlc = compute_session_vwap_metrics(
        [
            [base_ms + 900_000.0, 100.0, 100.0, 106.0, 100.0, 101.0],
            [base_ms + 1_800_000.0, 103.0, 300.0, 109.0, 103.0, 104.0],
        ],
        is_equity=False,
        cur_px=104.25,
    )
    assert vwap_hlc["session_vwap_px"] == pytest.approx(104.25)
    assert vwap_hlc["dist_vs_session_vwap_pct"] == pytest.approx(0.0)


def test_audit_fixes_equity_post_rth_vwap_atr_and_ohlc_hlo_disk_fallback(tmp_path):
    """Verify audit fixes: post-RTH VWAP dist/z recalculation + RTH ATR preservation, and 6-col H/L/O preservation in upgrade_tpo_vp_from_ohlc."""
    import json
    from intelligence.sentiment.collectors.microstructure_collector import upgrade_tpo_vp_from_ohlc
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics

    # 1. Equity series ending on final high-volume RTH bar (no zero-vol AH bar) + 5d history for atr_4h_pct
    d2_rth0 = datetime(2026, 11, 19, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    eq_rth_only = [
        [d2_rth0 + i * 300_000.0, round(165.0 + (0.40 if i % 2 == 0 else -0.40), 4), 2_000_000.0]
        for i in range(78)
    ]
    hist_5d = []
    for day_off in range(5, 0, -1):
        base_d = d2_rth0 - day_off * 86_400_000.0
        for i in range(78):
            hist_5d.append([base_d + i * 300_000.0, round(160.0 + day_off + (0.40 if i % 2 == 0 else -0.40), 4), 1_000_000.0])

    m24_rth = compute_24h_price_metrics(eq_rth_only, is_equity=True, series_7d_ts_px=hist_5d)
    assert m24_rth.get("atr_4h_pct", 0.0) > 0.0
    assert m24_rth["session_vwap_px"] == pytest.approx(165.0, abs=0.05)

    # Extended-hours override +3% to $170.0 must preserve RTH VWAP & ATR (including atr_4h_pct) and recompute dist/z vs restored RTH VWAP
    eq_policy = resolve_asset_session_policy("NVDA_US_EQ")
    _, m24_ov, _ = eq_policy.apply_post_rth_price_override(eq_rth_only, dict(m24_rth), 170.0)
    vwap_px = m24_rth["session_vwap_px"]
    sig_pct = m24_rth["session_vwap_sigma_pct"]
    expected_dist = round(((170.0 - vwap_px) / vwap_px) * 100.0, 2)
    expected_z = round((170.0 - vwap_px) / (vwap_px * (sig_pct / 100.0)), 2)
    assert m24_ov["session_vwap_px"] == pytest.approx(vwap_px)
    assert m24_ov["dist_vs_session_vwap_pct"] == pytest.approx(expected_dist, abs=0.02)
    assert m24_ov["session_vwap_z"] == pytest.approx(expected_z, abs=0.05)
    assert m24_ov["atr_4h_pct"] == pytest.approx(m24_rth["atr_4h_pct"])
    assert m24_ov["atr_15m_pct"] == pytest.approx(m24_rth["atr_15m_pct"])

    # 2. upgrade_tpo_vp_from_ohlc disk fallback preserves 6-col [ts, close, vol, high, low, open] for HLC/3 Session VWAP
    now_s = datetime(2026, 11, 20, 6, 0, 0, tzinfo=timezone.utc).timestamp()
    day_utc_00_ms = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    ohlc_pts = []
    # 18h yesterday (non-flat closes [99.5, 106.5] for REAL_VOL VP) + 6h today (close=100, high=106, low=100 -> TP=102.0)
    start_ms = day_utc_00_ms - 18 * 3_600_000.0
    for idx in range(96):
        ts_b = int(start_ms + idx * 900_000.0)
        if ts_b < day_utc_00_ms:
            c_b = 99.5 if idx % 2 == 0 else 106.5
            ohlc_pts.append([ts_b, c_b, 50.0, max(c_b, 106.5), min(c_b, 99.5), c_b])
        else:
            ohlc_pts.append([ts_b, 100.0, 50.0, 106.0, 100.0, 100.0])
    ohlc_file = tmp_path / "cache_ohlc_24h_BTCUSDC.json"
    ohlc_file.write_text(json.dumps({"ts": now_s, "symbol": "BTCUSDC", "points": ohlc_pts}), encoding="utf-8")
    # Age mtime > 3600s so load_stitched_24h_ohlc_points returns [] and forces the cand_vp disk fallback path
    stale_mtime = time.time() - 5000.0
    os.utime(ohlc_file, (stale_mtime, stale_mtime))

    m24_crypto = {"vp_source": "TPO", "low_24h": 99.0, "high_24h": 107.0}
    upgrade_tpo_vp_from_ohlc(
        "BTCUSDC",
        m24_crypto,
        cur_px=100.0,
        now_ts=now_s,
        s24_is_live_clock=True,
        cache_dir=str(tmp_path),
        allow_network=False,
    )
    assert m24_crypto.get("vp_source") == "REAL_VOL"
    # Close-only VWAP would be 100.0; preserved H/L/O typical price VWAP is exactly (106+100+100)/3 = 102.0
    assert m24_crypto["session_vwap_px"] == pytest.approx(102.0, abs=0.05)


def test_openai_compat_rest_adapter_resilience_and_direct_ab_routing(monkeypatch):
    """Verify decoupled OpenAI-compatible REST adapter: URL/prefix normalization, reasoning strip, resilience fallback, and direct A/B routing."""
    from types import SimpleNamespace
    import intelligence.sentiment.gemini_client as gc_mod
    from intelligence.sentiment.openai_compat_rest import (
        extract_openai_compat_text,
        format_openai_compat_model_id,
        normalize_openai_compat_url,
        strip_openai_compat_prefix,
    )

    # 1. Pure helpers
    assert normalize_openai_compat_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1/chat/completions"
    assert normalize_openai_compat_url("https://api.z.ai/api/paas/v4/chat/completions") == "https://api.z.ai/api/paas/v4/chat/completions"
    assert normalize_openai_compat_url("ftp://bad.example.com") == ""
    assert strip_openai_compat_prefix("openai_compat:Qwen/Qwen3-Max") == (True, "Qwen/Qwen3-Max")
    assert strip_openai_compat_prefix("gemini-4-neon-high") == (False, "gemini-4-neon-high")
    assert format_openai_compat_model_id("glm-5.3") == "openai_compat:glm-5.3"

    # 2. Content extraction (string, multimodal text list, <think> reasoning block)
    assert extract_openai_compat_text(
        {"choices": [{"message": {"content": "<think>internal chain</think>\n{\"decision\":\"HOLD\"}"}}]}
    ) == '{"decision":"HOLD"}'
    assert extract_openai_compat_text(
        {"choices": [{"message": {"content": [{"type": "text", "text": "{\"decision\":\"BUY\"}"}]}}]}
    ) == '{"decision":"BUY"}'

    # 3. Direct A/B routing via openai_compat:<slug> bypasses agy CLI completely and tags last_model_used
    posted_calls = []

    def _fake_post(url, payload, *, timeout=5.0, headers=None):
        posted_calls.append((url, payload, timeout, headers))
        mdl = payload.get("model")
        if mdl == "bad-non-json":
            return {"choices": [{"message": {"content": "not json at all"}}]}
        return {
            "choices": [
                {"message": {"content": f"<think>reasoning</think>\n{{\"decision\":\"HOLD\",\"via\":\"{mdl}\"}}"}}
            ]
        }

    monkeypatch.setattr(gc_mod, "fetch_json_post", _fake_post)
    monkeypatch.setenv("OPENAI_COMPAT_FALLBACK_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_COMPAT_FALLBACK_API_KEY", "sk-test-secret")
    monkeypatch.setenv("OPENAI_COMPAT_FALLBACK_MODEL", "bad-non-json,qwen3-max")

    cli_invocations = []

    def _fail_cli(self, *args, **kwargs):
        cli_invocations.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="quota exhausted")

    client = gc_mod.GeminiClient(
        model="gemini-4-neon-medium",
        api_key="google-rest-key",
        cli_path="/bin/sh",
        project_id="mptrade",
    )
    monkeypatch.setattr(client, "_execute_cli", _fail_cli.__get__(client, gc_mod.GeminiClient))

    direct_res = client.query_json("Analyze BTC", model="openai_compat:glm-5.3", timeout_sec=20.0)
    assert direct_res == {"decision": "HOLD", "via": "glm-5.3"}
    assert client.last_model_used == "openai_compat:glm-5.3"
    assert len(cli_invocations) == 0
    assert posted_calls[-1][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert posted_calls[-1][3]["Authorization"] == "Bearer sk-test-secret"

    # 4. Resilience fallback when agy CLI fails: tries bad-non-json (skipped on require_json), succeeds on qwen3-max
    posted_calls.clear()
    monkeypatch.setattr(client, "get_model_candidates", lambda m: [m])
    monkeypatch.setattr(client, "_is_quota_or_credit_error", lambda _p: False)
    fb_res = client.query_json("Analyze ETH", timeout_sec=25.0)
    assert fb_res == {"decision": "HOLD", "via": "qwen3-max"}
    assert client.last_model_used == "openai_compat:qwen3-max"
    assert len(cli_invocations) == 1
    assert [c[1]["model"] for c in posted_calls] == ["bad-non-json", "qwen3-max"]

    # 5. Falls through cleanly to Google REST when OpenAI-compatible URL is not configured
    posted_calls.clear()
    monkeypatch.delenv("OPENAI_COMPAT_FALLBACK_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_FALLBACK_MODEL", raising=False)

    def _google_only_post(url, payload, *, timeout=5.0, headers=None):
        posted_calls.append((url, payload))
        return {"candidates": [{"content": {"parts": [{"text": '{"decision":"HOLD","src":"google_rest"}'}]}}]}

    monkeypatch.setattr(gc_mod, "fetch_json_post", _google_only_post)
    g_res = client.query_json("Analyze SOL", timeout_sec=20.0)
    assert g_res == {"decision": "HOLD", "src": "google_rest"}
    assert "generativelanguage.googleapis.com" in posted_calls[0][0]


def test_root_cause_normalizations_ohlc_rth_anchors_and_model_cache_isolation():
    """Verify SSOT normalizations: build/normalize_ohlc_row invariants, override_series_tail_close, restore_rth_metric_anchors, model-scoped GeminiClient._cache, and degraded model prefix strip."""
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.llm_quota_router import is_degraded_trading_model
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        override_series_tail_close,
        restore_rth_metric_anchors,
    )
    from providers.ohlc_utils import build_ohlc_row, normalize_ohlc_row

    # 1. Canonical OHLC row builder/normalizer enforces low <= close <= high and [lo, hi] bounds
    row_inv = build_ohlc_row(1_795_000_000.0, 105.0, 10.0, 102.0, 99.0, 106.0)
    assert row_inv == [1_795_000_000_000, 105.0, 10.0, 105.0, 99.0, 105.0]
    row_clamped = normalize_ohlc_row([1_795_000_000_000, 105.0, 10.0, 112.0, 95.0, 101.0], clamp_bounds=(98.0, 108.0))
    assert row_clamped == [1_795_000_000_000, 105.0, 10.0, 108.0, 98.0, 101.0]

    # 2. override_series_tail_close drops stale H/L/O to avoid close > high OHLC violation
    ov = override_series_tail_close(
        [[1_795_000_000_000, 100.0, 50.0, 101.0, 99.0, 100.0], [1_795_000_300_000, 101.0, 60.0, 102.0, 100.0, 101.0]],
        107.0,
    )
    assert ov[-1] == [1_795_000_300_000, 107.0, 60.0]

    # 3. restore_rth_metric_anchors restores VP/VWAP/ATR atomically and realigns distances & z-score to live_px
    m24_mut = {"poc_px": 107.0, "val_px": 106.0, "vah_px": 108.0, "session_vwap_px": 106.5, "atr_15m_pct": 2.9}
    restore_rth_metric_anchors(
        m24_mut,
        vp_src={"poc_px": 100.0, "val_px": 99.0, "vah_px": 101.0, "vp_source": "REAL_VOL"},
        vwap_src={"session_vwap_px": 100.0, "session_vwap_sigma_pct": 1.0, "session_vwap_anchor": "rth"},
        atr_src={"atr_15m_pct": 0.45, "atr_1h_pct": 0.85, "atr_4h_pct": 1.75},
        live_px=103.0,
    )
    assert m24_mut["poc_px"] == 100.0 and m24_mut["poc_dist_pct"] == pytest.approx(3.0)
    assert m24_mut["session_vwap_px"] == 100.0 and m24_mut["dist_vs_session_vwap_pct"] == pytest.approx(3.0)
    assert m24_mut["session_vwap_z"] == pytest.approx(3.0)
    assert m24_mut["atr_15m_pct"] == 0.45 and m24_mut["atr_4h_pct"] == 1.75

    # 4. GeminiClient._cache isolates distinct models for the same prompt (A/B safe) & degraded classifier strips openai_compat:
    runner_calls = []
    gc = GeminiClient(
        model="gemini-4-neon-medium",
        custom_runner=lambda p, m, t: (runner_calls.append(m) or f'{{"model":"{m}"}}'),
    )
    r1 = gc.query_json("same prompt", model="gemini-4-neon-medium", cache_ttl_sec=60.0)
    r2 = gc.query_json("same prompt", model="openai_compat:glm-5.3", cache_ttl_sec=60.0)
    r3 = gc.query_json("same prompt", model="openai_compat:glm-5.3", cache_ttl_sec=60.0)
    assert r1 == {"model": "gemini-4-neon-medium"}
    assert r2 == {"model": "openai_compat:glm-5.3"}
    assert r3 == {"model": "openai_compat:glm-5.3"}
    assert runner_calls == ["gemini-4-neon-medium", "openai_compat:glm-5.3"]
    assert is_degraded_trading_model("openai_compat:glm-5.3") is False
    assert is_degraded_trading_model("openai_compat:gemini-2.5-flash") is True
    assert is_degraded_trading_model("openai_compat:qwen-low") is True


def test_stabilize_faza1_and_audit_bugfixes(tmp_path, monkeypatch):
    """Verify Faza 1 stabilization and real audit bugfixes: pretrade basis/fund deltas, PRE/POST benchmark preservation,
    OKX ctVal periodic retry + Binance quote filter, 4h ATR tail-bar & divergence guards, equity pre-market VWAP, and spot-only Section 3B CVD."""
    import asyncio
    from intelligence.external.collectors import bybit_liquidations_collector as liq_mod
    from intelligence.sentiment.collectors.macro_telemetry_collector import MacroAndNewsSensorsCollector
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_atr_multi_horizon,
        compute_session_vwap_metrics,
        format_session_vwap_summary,
    )

    # 1. Pre-trade telemetry loads microstructure_cycle_history.json read-only and populates basis_d15m + fund_d1h
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    now_s = 1_795_100_000.0
    hist_path = cdir / "telemetry" / "microstructure_cycle_history.json"
    hist_orig = {
        "BTCUSDC": [
            {"ts": now_s - 3600.0, "funding_rate": 0.00010, "perp_spot_basis_bps": -2.0},
            {"ts": now_s - 900.0, "funding_rate": 0.00015, "perp_spot_basis_bps": -1.0},
        ]
    }
    hist_path.write_text(json.dumps(hist_orig), encoding="utf-8")
    (cdir / "telemetry" / "orderbook_depth_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "mid_price": 62000.0,
                "best_bid": 61999.5,
                "best_ask": 62000.5,
                "bid_depth_usd": 500000.0,
                "ask_depth_usd": 450000.0,
                "imbalance_ratio": 0.53,
                "spread_bps": 0.16,
                "ts": now_s - 5.0,
            }
        ),
        encoding="utf-8",
    )
    (cdir / "telemetry" / "derivatives_telemetry_BTCUSDC.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDC",
                "funding_rate": 0.00025,
                "funding_annualized_pct": 27.38,
                "perp_spot_basis_bps": 4.5,
                "open_interest_usd": 1_500_000_000.0,
                "ts": now_s - 10.0,
            }
        ),
        encoding="utf-8",
    )
    pt_btc = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir), now_ts=now_s)
    fr_line = pt_btc.get("Perpetual Funding Rate", "")
    assert "basis_d15m=+5.5bps" in fr_line
    assert "fund_d1h=+1.50bps" in fr_line
    # Confirm read-only: cycle history file on disk was NOT modified
    assert json.loads(hist_path.read_text(encoding="utf-8")) == hist_orig

    # 2. _normalize_macro_benchmarks_items preserves PRE/POST market_state when bar_age_s > 2700s while stripping stale intraday deltas
    mb_norm = MacroAndNewsSensorsCollector._normalize_macro_benchmarks_items(
        {
            "ts": now_s,
            "items": {
                "SPY": {
                    "price": 595.0,
                    "chg_24h_pct": 0.45,
                    "chg_1h_pct": 0.12,
                    "chg_15m_pct": 0.05,
                    "market_state": "PRE",
                    "last_bar_ts": now_s - 3600.0,
                },
                "US10Y": {
                    "price": 4.25,
                    "chg_24h_pct": -0.3,
                    "chg_1h_pct": 0.08,
                    "chg_15m_pct": 0.02,
                    "market_state": "REGULAR",
                    "last_bar_ts": now_s - 3600.0,
                },
            },
        },
        cache_dir=str(cdir),
        now_ts=now_s,
    )
    assert mb_norm["items"]["SPY"]["market_state"] == "PRE"
    assert "chg_15m_pct" not in mb_norm["items"]["SPY"]
    assert "chg_1h_pct" not in mb_norm["items"]["SPY"]
    assert mb_norm["items"]["US10Y"]["market_state"] == "CLOSED"
    assert "chg_15m_pct" not in mb_norm["items"]["US10Y"]

    # 3. BinanceFuturesLiquidationCollector filters non-USDT/USDC symbols; OkxLiquidationCollector self-heals ctVal on tick
    bf_col = liq_mod.BinanceFuturesLiquidationCollector(symbols=["BTCUSDT"], start_sync=False)
    bf_col._ingest(
        json.dumps(
            {
                "e": "forceOrder",
                "E": int(now_s * 1000),
                "o": {"s": "BTCUSD_PERP", "S": "SELL", "q": "10", "ap": "62000.0", "T": int(now_s * 1000)},
            }
        )
    )
    assert bf_col.get_summary("BTCUSDT", window_sec=900.0, now=now_s).events_count == 0
    bf_col._ingest(
        json.dumps(
            {
                "e": "forceOrder",
                "E": int(now_s * 1000),
                "o": {"s": "BTCUSDT", "S": "SELL", "q": "0.5", "ap": "62000.0", "T": int(now_s * 1000)},
            }
        )
    )
    assert bf_col.get_summary("BTCUSDT", window_sec=900.0, now=now_s).long_liq_usd == pytest.approx(31000.0)

    okx_calls = []

    def _fake_okx_fetch(_url, **_kw):
        okx_calls.append(1)
        if len(okx_calls) == 1:
            raise RuntimeError("transient VPN startup outage")
        return {"data": [{"instId": "BTC-USDT-SWAP", "ctVal": "0.01"}]}

    monkeypatch.setattr(liq_mod, "fetch_public_json", _fake_okx_fetch)
    okx_col = liq_mod.OkxLiquidationCollector(symbols=["BTCUSDT"], start_sync=False, ct_val_retry_sec=300.0)
    okx_col._before_session()
    assert okx_col._ct_val_by_inst == {}
    # Before retry interval elapses, does not re-fetch
    asyncio.run(okx_col._on_session_tick(okx_col._last_ct_val_attempt_ts + 60.0))
    assert len(okx_calls) == 1
    # After retry interval elapses, self-heals ctVal without requiring WS disconnect
    asyncio.run(okx_col._on_session_tick(okx_col._last_ct_val_attempt_ts + 305.0))
    assert len(okx_calls) == 2
    assert okx_col._ct_val_by_inst.get("BTC-USDT-SWAP") == pytest.approx(0.01)

    # 4. 4h ATR excludes a 2-minute forming tail bar even when merged with compact 7d 4h bars AND rejects diverged 7d archives
    base_4h_ms = 1_795_008_000_000.0  # Aligned to 4h boundary
    compact_7d_4h = [
        [base_4h_ms - (40 - i) * 14_400_000.0, 62000.0, 0.0, 62620.0, 61380.0]
        for i in range(40)
    ]  # 2.0% true range per 4h bar
    dense_24h = []
    for b_idx in range(6):
        b_start = base_4h_ms + b_idx * 14_400_000.0
        for m_step in range(0, 240, 15):
            px_v = 62620.0 if (m_step // 15) % 2 == 0 else 61380.0
            dense_24h.append([b_start + m_step * 60_000.0, px_v])
    atr_clean = compute_atr_multi_horizon(dense_24h, series_7d_ts_px=compact_7d_4h, cur_px=62000.0)
    # Append a tiny 2-minute forming tail 4h bar with $1 range (should be excluded!)
    dense_with_tail = dense_24h + [
        [base_4h_ms + 6 * 14_400_000.0, 62000.0],
        [base_4h_ms + 6 * 14_400_000.0 + 120_000.0, 62001.0],
    ]
    atr_with_tail = compute_atr_multi_horizon(dense_with_tail, series_7d_ts_px=compact_7d_4h, cur_px=62000.0)
    assert atr_with_tail["atr_4h_pct"] == pytest.approx(atr_clean["atr_4h_pct"], abs=0.02)

    # Diverged 7d archive (e.g. pre-split $200 vs current $62000) must be ignored instead of exploding atr_4h_pct
    diverged_7d = [[base_4h_ms - (40 - i) * 14_400_000.0, 200.0, 0.0, 205.0, 195.0] for i in range(40)]
    atr_div = compute_atr_multi_horizon(dense_24h, series_7d_ts_px=diverged_7d, cur_px=62000.0)
    assert atr_div["atr_4h_pct"] < 5.0

    # 5. US equity pre-market bars preserve prior RTH Session VWAP; crypto 00:02 UTC 0-vol tick preserves fresh UTC VWAP
    d1_rth0 = datetime(2026, 11, 19, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    d2_pre0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0  # 05:00 NY pre-market
    eq_pre_bars = [
        [d1_rth0 + i * 300_000.0, 180.0, 500_000.0] for i in range(10)
    ] + [
        [d2_pre0, 185.0, 1200.0],  # Single positive-vol pre-market print today
        [d2_pre0 + 300_000.0, 185.5, 900.0],
    ]
    eq_vwap_pre = compute_session_vwap_metrics(eq_pre_bars, is_equity=True, cur_px=185.5)
    assert eq_vwap_pre["session_vwap_anchor"] == "prev_rth"
    assert eq_vwap_pre["session_vwap_px"] == pytest.approx(180.0)

    utc_mid_ms = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    crypto_midnight = [
        [utc_mid_ms - 900_000.0, 62000.0, 100.0],
        [utc_mid_ms - 600_000.0, 62200.0, 100.0],
        [utc_mid_ms - 300_000.0, 62400.0, 200.0],
        [utc_mid_ms + 120_000.0, 62500.0, 0.0],  # 00:02 UTC 0-vol live tick
    ]
    c_vwap_mid = compute_session_vwap_metrics(crypto_midnight, is_equity=False, cur_px=62500.0)
    assert c_vwap_mid["session_vwap_px"] == pytest.approx(62250.0)
    assert format_session_vwap_summary(c_vwap_mid, cur_px=62500.0).startswith("$62,250.00 (")

    # 6. Section 3B emits real spot CVD & agg_spot_cvd_15m even when perp cvd_15m_usd is None (spot-only / missing whale)
    sec_3b_spot = "\n".join(
        _format_microstructure_section(
            {
                "ARBUSDC": {
                    "ob_imbalance": 0.55,
                    "bid_depth_usd": 50000.0,
                    "ask_depth_usd": 40000.0,
                    "spread_bps": 3.5,
                    "spot_cvd_15m_usd": -12500.0,
                    "spot_cvd_source": "real_spot",
                    "agg_spot_cvd_15m_usd": 145000.0,
                    "agg_spot_cvd_source": "usdc+usdt",
                    "agg_spot_vol_15m_usd": 620000.0,
                }
            },
            {"ARBUSDC": {"price": 0.25}},
        )
    )
    assert "spot_cvd_15m=-$12.5K[real_spot]" in sec_3b_spot
    assert "agg_spot_cvd_15m=+$145.0K[usdc+usdt, vol=$620.0K]" in sec_3b_spot


def test_faza2_prep_seams_zero_behavior_change(tmp_path):
    """Verify Faza 2 preparatory SSOT seams preserve exact live behavior while exposing clean extension hooks."""
    from intelligence.macro.calendar_ssot import (
        _collect_routine_session_windows,
        _format_calendar_countdown,
        _load_macro_calendar_windows,
        _primary_eet_dt,
    )
    from intelligence.macro_sentiment_ssot import (
        get_breadth_excluded_bases,
        get_tracked_equity_base_tickers,
        resolve_secondary_ny_datetime,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import MacroTelemetryCollectorBase
    from orchestratorOS.supervisor.prompt_sections import _format_macro_calendar_line

    # 1. Shared NY timezone resolver matches America/New_York EST/EDT accurately
    ts_est = datetime(2026, 11, 20, 14, 30, 0, tzinfo=timezone.utc).timestamp()
    dt_ny = resolve_secondary_ny_datetime(ts_est)
    assert dt_ny.hour == 9 and dt_ny.minute == 30

    # 2. Tracked equity base ticker SSOT feeds breadth exclusions and supports tokenized Kraken xStocks
    t212_only = get_tracked_equity_base_tickers(include_tokenized=False)
    with_tok = get_tracked_equity_base_tickers(include_tokenized=True)
    assert set(t212_only) <= set(with_tok)
    excluded = get_breadth_excluded_bases()
    for base in t212_only:
        assert f"{base}B" in excluded and f"{base}XB" in excluded

    # 3. Calendar routine-window seam supports skip_slot_fn without altering default behavior
    assert _format_calendar_countdown(10.0) == "10m"
    assert _format_calendar_countdown(120.0) == "2h"
    assert _format_calendar_countdown(135.0) == "2h15m"
    windows, routine_labels = _load_macro_calendar_windows()
    dt_eet = _primary_eet_dt(ts_est)
    _, _, up_all = _collect_routine_session_windows(
        ts_est,
        dt_eet,
        windows=windows,
        routine_labels=routine_labels,
        fomc_dates=set(),
        macro_release_confirmed=False,
        has_fomc_headline=False,
    )
    _, _, up_skip_open = _collect_routine_session_windows(
        ts_est,
        dt_eet,
        windows=windows,
        routine_labels=routine_labels,
        fomc_dates=set(),
        macro_release_confirmed=False,
        has_fomc_headline=False,
        skip_slot_fn=lambda _d, code, _t: code == "US_WALL_STREET_OPEN",
    )
    assert any(x[1] == "US_WALL_STREET_OPEN" for x in up_all)
    assert all(x[1] != "US_WALL_STREET_OPEN" for x in up_skip_open)

    # 4. MacroTelemetryCollectorBase shared cache persistence/read helpers + Section 1 calendar line helper
    out_p = tmp_path / "macro_benchmarks_collect.json"
    MacroTelemetryCollectorBase._persist_cache_json(str(out_p), {"ts": ts_est, "items": {"SPY": {"price": 595.0}}})
    loaded, _ = MacroTelemetryCollectorBase._read_cached_dict(
        str(tmp_path), "macro_benchmarks_files", "macro_benchmarks_collect.json"
    )
    assert isinstance(loaded, dict) and loaded["items"]["SPY"]["price"] == 595.0
    assert _format_macro_calendar_line(None) is None
    cal_line = _format_macro_calendar_line(
        {
            "session_label": "EU/US Overlap",
            "next_high_impact_event": "US Wall Street Cash Open at 16:30 EET in 10m",
            "next_event_is_high_impact": False,
            "fomc_today": False,
        }
    )
    assert cal_line is not None and 'next_routine_session_window="US Wall Street Cash Open at 16:30 EET in 10m"' in cal_line


def test_provider_ohlc_hlo_preservation_and_pretrade_multi_venue_parity(tmp_path, monkeypatch):
    """Verify Binance/HL/Kraken/Yahoo OHLC H/L/O propagation and Pre-Trade agg_spot_cvd_15m + multi-venue liq parity."""
    from types import SimpleNamespace
    from botcore import ClosedPriceSeries
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        format_agg_spot_cvd_part,
        format_realized_liq_sources_tag,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import _extract_ohlc_series_points
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.ohlc_utils import extract_yahoo_intraday_chart_bars
    from state_io import save_json_state
    import providers.binance_provider as bn_mod
    from kraken.kraken_client import KrakenClient

    # 1. ClosedPriceSeries.from_raw_candles + BinanceProvider + HyperliquidProvider preserve H/L/O
    raw_bn = [
        [1700000000000, "100.0", "103.5", "98.5", "101.0", "25.0", 1700000300000],
        [1700000300000, "101.0", "104.0", "99.0", "102.0", "30.0", 1700000600000],
        [1700000600000, "102.0", "105.0", "101.0", "103.0", "10.0", 1700000900000],
    ]
    monkeypatch.setattr(bn_mod, "_get_bapi", lambda: SimpleNamespace(client=SimpleNamespace(get_klines=lambda **_kw: raw_bn)))
    bn_ser = BinanceProvider().ohlc_series("BTCUSDC", 5)
    assert bn_ser.closes == (101.0, 102.0)
    assert bn_ser.highs == (103.5, 104.0)
    assert bn_ser.lows == (98.5, 99.0)
    assert bn_ser.opens == (100.0, 101.0)
    pts_bn = _extract_ohlc_series_points(bn_ser, min_ts=1699999000.0)
    assert pts_bn[0] == [1700000300000, 101.0, 25.0, 103.5, 98.5, 100.0]

    hl_candles = [
        {"o": "64.0", "h": "66.5", "l": "63.2", "c": "65.1", "v": "1200.0", "T": 1700000300000},
        {"o": "65.1", "h": "67.0", "l": "64.8", "c": "66.0", "v": "900.0", "T": 1700000600000},
    ]
    hl_prov = HyperliquidProvider()
    monkeypatch.setattr(
        hl_prov,
        "_require_hl_client_and_pair",
        lambda _caller: (SimpleNamespace(candles=lambda _p, _iv, _lh: hl_candles), "HYPE/USDC", None),
    )
    hl_ser = hl_prov.ohlc_series("HYPEUSDC", 5)
    assert hl_ser.closes == (65.1,)
    assert hl_ser.highs == (66.5,)
    assert hl_ser.lows == (63.2,)
    assert hl_ser.opens == (64.0,)

    # 2. KrakenClient + KrakenProvider preserve H/L/O without altering ohlc_closes_with_timestamps return shape
    kc = KrakenClient.__new__(KrakenClient)
    kc._public = lambda _ep, _params: {
        "ADAUSD": [
            [1700000000, "0.148", "0.153", "0.146", "0.150", "0.149", "50000.0", 42],
            [1700000300, "0.150", "0.155", "0.149", "0.151", "0.150", "10000.0", 12],
        ],
        "last": 1700000300,
    }
    kp = KrakenProvider()
    monkeypatch.setattr(kp, "_client", lambda: kc)
    kr_ser = kp.ohlc_series("ADAUSD", 5)
    assert kr_ser.closes == (0.15,)
    assert kr_ser.highs == (0.153,)
    assert kr_ser.lows == (0.146,)
    assert kr_ser.opens == (0.148,)
    assert kr_ser.volumes == (50000.0,)

    # 3. Yahoo extract_yahoo_intraday_chart_bars preserves 6-tuple H/L/O and 212trading.market_data handles it cleanly
    now_s = math.trunc(time.time())
    y_payload = {
        "chart": {
            "result": [
                {
                    "meta": {"regularMarketPrice": 153.0, "regularMarketTime": now_s, "marketState": "REGULAR"},
                    "timestamp": [now_s - 1200, now_s - 900, now_s - 600, now_s - 300, now_s],
                    "indicators": {
                        "quote": [
                            {
                                "open": [149.0, 149.5, 150.0, 150.5, 151.0],
                                "high": [150.2, 150.8, 151.2, 151.8, 153.6],
                                "low": [148.4, 149.1, 149.4, 150.1, 150.7],
                                "close": [149.5, 150.0, 150.5, 151.0, 153.0],
                                "volume": [900, 1100, 1000, 1200, 1500],
                            }
                        ]
                    },
                }
            ]
        }
    }
    _meta, y_bars = extract_yahoo_intraday_chart_bars(y_payload)
    assert len(y_bars) == 5
    assert y_bars[0] == (now_s - 1200, 149.5, 900.0, 150.2, 148.4, 149.0)
    md_mod = load_venue_modules("212trading", "market_data")
    monkeypatch.setattr(md_mod, "_can_use_ssot_equity_cache", lambda _sym: None)
    monkeypatch.setattr(md_mod, "_chart", lambda _sym, **_kw: (_meta, y_bars))
    assert md_mod.trend_slope_pct("NVDA", bars=5) > 0.0
    chk = md_mod.check_market("NVDA")
    assert chk is not None and chk["trading"] is True and chk["volume"] == 5700.0

    # 4. SSOT formatters + Pre-Trade telemetry surface agg_spot_cvd_15m and multi-venue realized liquidations cleanly
    assert format_agg_spot_cvd_part(
        {"agg_spot_cvd_15m_usd": 59800.0, "agg_spot_vol_15m_usd": 155700.0, "agg_spot_cvd_source": "usdc+usdt"},
        lambda v: f"${v:+,.0f}",
        lambda v: f"${v:,.0f}",
    ) == "agg_spot_cvd_15m=$+59,800[usdc+usdt, vol=$155,700]"
    assert format_realized_liq_sources_tag({"realized_liq_sources": ["bybit_ws", "binance_futures_ws", "okx_ws"]}) == "bybit+binance+okx"
    assert format_realized_liq_sources_tag({"realized_liq_sources": ["bybit_ws"]}, multi_only=True) == ""
    assert format_realized_liq_sources_tag({"realized_liq_sources": ["bybit_ws"]}, multi_only=False) == "bybit"

    now_pt = time.time()
    cdir = tmp_path / "cachedb"
    (cdir / "telemetry").mkdir(parents=True, exist_ok=True)
    save_json_state(
        str(cdir / "telemetry" / "orderbook_depth_BTCUSDC.json"),
        {
            "symbol": "BTCUSDC",
            "mid_price": 62000.0,
            "best_bid": 61999.5,
            "best_ask": 62000.5,
            "spread_bps": 0.16,
            "bid_depth_usd": 6_000_000.0,
            "ask_depth_usd": 5_000_000.0,
            "depth_1pct_bid_usd": 5_500_000.0,
            "depth_1pct_ask_usd": 4_500_000.0,
            "imbalance_ratio": 0.545,
            "largest_bid_wall_usd": 1_000_000.0,
            "largest_bid_wall_price": 61500.0,
            "largest_ask_wall_usd": 800_000.0,
            "largest_ask_wall_price": 62500.0,
            "book_span_pct": 1.3,
            "spot_cvd_15m_usd": 450_000.0,
            "spot_cvd_source": "real_spot",
            "spot_vol_15m_usd": 2_000_000.0,
            "agg_spot_cvd_15m_usd": -125_000.0,
            "agg_spot_vol_15m_usd": 9_500_000.0,
            "agg_spot_cvd_source": "usdc+usdt",
            "ts": now_pt,
        },
    )
    save_json_state(
        str(cdir / "telemetry" / "whale_snapshot_BTCUSDC.json"),
        {
            "symbol": "BTCUSDC",
            "top_traders_long_pct": 0.65,
            "top_traders_long_ratio": 1.86,
            "taker_buy_sell_ratio": 0.62,
            "taker_buy_vol_usd": 45_000_000.0,
            "taker_sell_vol_usd": 72_000_000.0,
            "open_interest_usd": 5_900_000_000.0,
            "open_interest_1h_change_pct": -0.45,
            "open_interest_15m_change_pct": -0.35,
            "ts": now_pt,
        },
    )
    save_json_state(
        str(cdir / "telemetry" / "realized_liquidations.json"),
        {
            "ts": now_pt,
            "source": "multi_venue_ws",
            "sources": ["bybit_ws", "binance_futures_ws", "okx_ws"],
            "connected_sources": ["bybit_ws", "binance_futures_ws", "okx_ws"],
            "connected": True,
            "symbols": {
                "BTCUSDT": {
                    "long_15m_usd": 114_711.0,
                    "short_15m_usd": 0.0,
                    "net_15m_usd": -114_711.0,
                    "events_15m": 27,
                }
            },
        },
    )
    pt_tel = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir), allow_network=False, now_ts=now_pt)
    assert "agg_spot_cvd_15m=$-125,000[usdc+usdt, vol=$9,500,000]" in pt_tel["15m Cumulative Volume Delta (CVD)"]
    assert "Realized Liquidations (15m Bybit)" not in pt_tel
    assert pt_tel["Realized Liquidations (15m)"] == "long=$114,711, short=$0, events=27, src=bybit+binance+okx"



def test_faza2_proactive_macro_calendar_earnings_and_pretrade_catalyst(tmp_path, monkeypatch):
    """Verify Faza 2 end-to-end: ForexFactory + Nasdaq collector, partial-failure resilience, static BLS/BEA/FOMC
    dedup & DST accuracy, pre_event_risk_window, Section 1 rendering, Pre-Trade catalyst, and GTC daemon step."""
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.macro.calendar_ssot import (
        format_pretrade_macro_catalyst_line,
        resolve_macro_economic_calendar,
    )
    from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        MacroAndNewsSensorsCollector,
        collect_macro_calendar_snapshot,
    )
    from orchestratorOS.supervisor.prompt_sections import _format_macro_calendar_line

    # 1. Interval scales mathematically from T_exec (60 * 360s = 21600s = 6h)
    assert resolve_derived_telemetry_intervals(executor_interval_sec=360.0)["macro_calendar_interval_sec"] == 21600.0

    # 2. MacroEconomicCalendarCollector filters ForexFactory (USD High + whitelisted Medium; drops EUR & non-whitelisted Medium)
    #    and Nasdaq earnings (tracked holdings + bellwethers; drops noise symbols; maps BMO/AMC/TBD)
    # Test date: Wed 2026-11-18 13:00 UTC (08:00 EST / 15:00 EET — 30m before 08:30 EST Core CPI)
    now_cpi_pre = datetime(2026, 11, 18, 13, 0, 0, tzinfo=timezone.utc).timestamp()
    ff_payload = [
        {
            "title": "Core CPI m/m",
            "country": "USD",
            "date": "2026-11-18T08:30:00-05:00",
            "impact": "High",
            "forecast": "0.3%",
            "previous": "0.2%",
        },
        {
            "title": "Unemployment Claims",
            "country": "USD",
            "date": "2026-11-19T08:30:00-05:00",
            "impact": "Medium",
            "forecast": "215K",
            "previous": "210K",
        },
        {
            "title": "Natural Gas Storage",
            "country": "USD",
            "date": "2026-11-19T10:30:00-05:00",
            "impact": "Medium",
            "forecast": "10B",
            "previous": "20B",
        },
        {
            "title": "German Flash PMI",
            "country": "EUR",
            "date": "2026-11-18T03:30:00-05:00",
            "impact": "High",
            "forecast": "48.0",
            "previous": "47.5",
        },
    ]
    nq_by_date = {
        "2026-11-18": {
            "data": {
                "rows": [
                    {"symbol": "NVDA", "name": "NVIDIA Corp", "time": "time-after-hours", "epsForecast": "$0.85", "fiscalQuarterEnding": "Oct/2026"},
                    {"symbol": "NOISE", "name": "Tiny Co", "time": "time-pre-market", "epsForecast": "$0.10"},
                ]
            }
        },
        "2026-11-19": {
            "data": {
                "rows": [
                    {"symbol": "AAPL", "name": "Apple Inc", "time": "time-after-hours", "epsForecast": "$1.62", "fiscalQuarterEnding": "Sep/2026"},
                ]
            }
        },
    }

    def _fake_fetch(url, **_kw):
        if "ff_calendar" in url:
            return ff_payload
        for d_iso, body in nq_by_date.items():
            if d_iso in url:
                return body
        return None

    cdir = tmp_path / "cachedb"
    (cdir / "macro").mkdir(parents=True, exist_ok=True)
    snap = collect_macro_calendar_snapshot(
        str(cdir),
        now_ts=now_cpi_pre,
        persist=True,
        fetch_fn=_fake_fetch,
        tracked_equity_tickers=["NVDA", "SPCX"],
    )
    assert snap is not None and snap["events_ok"] is True and snap["earnings_ok"] is True
    ev_titles = [e["title"] for e in snap["events"]]
    assert ev_titles == ["Core CPI m/m", "Unemployment Claims"]
    assert [e["symbol"] for e in snap["earnings"]] == ["NVDA", "AAPL"]
    assert snap["earnings"][0]["timing"] == "AMC" and snap["earnings"][0]["is_portfolio_holding"] is True
    assert snap["earnings"][1]["is_portfolio_holding"] is False
    assert (cdir / "macro" / "macro_calendar_collect.json").exists()

    # Partial failure (Nasdaq down, ForexFactory up) preserves prior fresh earnings from cache
    def _ff_only_fetch(url, **_kw):
        if "ff_calendar" in url:
            return ff_payload
        raise RuntimeError("nasdaq timeout")

    snap_partial = collect_macro_calendar_snapshot(
        str(cdir),
        now_ts=now_cpi_pre + 600.0,
        persist=True,
        fetch_fn=_ff_only_fetch,
        tracked_equity_tickers=["NVDA", "SPCX"],
    )
    assert snap_partial is not None and snap_partial["events_ok"] is True and snap_partial["earnings_ok"] is False
    assert [e["symbol"] for e in snap_partial["earnings"]] == ["NVDA", "AAPL"]

    # Complete failure returns None without clobbering existing cached file
    assert (
        collect_macro_calendar_snapshot(
            str(cdir),
            now_ts=now_cpi_pre + 1200.0,
            persist=True,
            fetch_fn=lambda _u, **_kw: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        is None
    )

    # 3. Pure resolver merges live snapshot, triggers pre_event_risk_window (30m <= 45m), replaces generic 08:30 slot, renders Section 1
    cal = resolve_macro_economic_calendar(
        now_ts=now_cpi_pre,
        calendar_snapshot=snap,
        tracked_equity_tickers=["NVDA", "SPCX"],
    )
    assert cal["macro_release_confirmed"] is True
    assert cal["next_event_is_high_impact"] is True
    assert "Core CPI m/m [HIGH] (fc=0.3%, prev=0.2%) at 15:30 EET in 30m" == cal["next_high_impact_event"]
    assert cal["pre_event_risk_window"] is True
    assert "Core CPI m/m [HIGH] in 30m (fc=0.3%, prev=0.2%)" == cal["pre_event_risk_reason"]
    assert len(cal["confirmed_events_today"]) == 1
    assert len(cal["earnings_today"]) == 1 and cal["earnings_today"][0]["symbol"] == "NVDA"
    assert len(cal["upcoming_earnings"]) == 1 and cal["upcoming_earnings"][0]["symbol"] == "AAPL"
    assert all("Pre-Market Macro & Treasury Liquidity Window" not in w for w in cal["upcoming_windows"][:2])

    sec1_line = _format_macro_calendar_line(cal)
    assert sec1_line is not None
    assert 'PRE_EVENT_RISK="Core CPI m/m [HIGH] in 30m (fc=0.3%, prev=0.2%)"' in sec1_line
    assert "confirmed_macro_today=" in sec1_line
    assert "earnings_today=['NVDA (AMC, eps_fc=$0.85) [HOLDING]']" in sec1_line
    assert "upcoming_earnings=['2026-11-19 AAPL (AMC, eps_fc=$1.62)']" in sec1_line

    # 4. Portfolio-holding earnings within 60m of AMC (15:30 EST = 20:30 UTC -> 35m before 16:05 EST AMC) triggers pre_event_risk_window
    now_nvda_amc = datetime(2026, 11, 18, 20, 30, 0, tzinfo=timezone.utc).timestamp()
    cal_amc = resolve_macro_economic_calendar(
        now_ts=now_nvda_amc,
        calendar_snapshot=snap,
        tracked_equity_tickers=["NVDA"],
    )
    assert cal_amc["pre_event_risk_window"] is True
    assert "NVDA earnings AMC in 35m" == cal_amc["pre_event_risk_reason"]
    cat_nvda = format_pretrade_macro_catalyst_line(cal_amc, symbol="NVDA_US_EQ")
    assert cat_nvda == "PRE_EVENT_RISK: NVDA earnings AMC in 35m"

    # 5. Static BLS/BEA/FOMC fallback handles US/EU DST mismatch week accurately (2026-03-18 FOMC: 14:00 EDT = 20:00 EET!)
    ts_dst_fomc = datetime(2026, 3, 18, 17, 30, 0, tzinfo=timezone.utc).timestamp()  # 13:30 EDT / 19:30 EET (30m before 14:00 EDT)
    cal_dst = resolve_macro_economic_calendar(now_ts=ts_dst_fomc)
    assert cal_dst["fomc_today"] is True
    assert "FOMC Rate Decision [HIGH] at 20:00 EET in 30m" == cal_dst["next_high_impact_event"]
    assert cal_dst["pre_event_risk_window"] is True
    assert all("Afternoon Treasury Auction" not in w for w in cal_dst["upcoming_windows"])

    # Static BLS CPI fallback (2026-11-10) deduplicates cleanly when live ForexFactory CPI exists on same date
    ts_static_cpi = datetime(2026, 11, 10, 13, 0, 0, tzinfo=timezone.utc).timestamp()
    cal_static_cpi = resolve_macro_economic_calendar(now_ts=ts_static_cpi)
    assert cal_static_cpi["pre_event_risk_window"] is True
    assert "US CPI Inflation Report [HIGH] in 30m" == cal_static_cpi["pre_event_risk_reason"]
    cal_dedup_cpi = resolve_macro_economic_calendar(
        now_ts=ts_static_cpi,
        calendar_snapshot={
            "ts": ts_static_cpi,
            "events": [
                {
                    "ts": datetime(2026, 11, 10, 13, 30, 0, tzinfo=timezone.utc).timestamp(),
                    "date_iso": "2026-11-10",
                    "ny_time": "08:30",
                    "title": "CPI y/y",
                    "impact": "HIGH",
                    "forecast": "2.8%",
                    "previous": "2.9%",
                }
            ],
        },
    )
    assert len(cal_dedup_cpi["confirmed_events_today"]) == 1
    assert cal_dedup_cpi["confirmed_events_today"][0]["title"] == "CPI y/y"

    # 6. MacroAndNewsSensorsCollector + Pre-Trade MacroPillarsStage + GlobalTelemetryCollector integration
    sensors, _ = MacroAndNewsSensorsCollector(start_sync=False).collect(
        str(cdir), now_ts=now_cpi_pre, allow_network=False, persist=False
    )
    assert sensors["macro_calendar"]["pre_event_risk_window"] is True
    assert sensors["macro_calendar"]["earnings_today"][0]["symbol"] == "NVDA"

    pt_cal = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(cdir), now_ts=now_cpi_pre)
    assert "Upcoming Macro / Earnings Catalyst" in pt_cal
    assert "PRE_EVENT_RISK: Core CPI m/m [HIGH] in 30m" in pt_cal["Upcoming Macro / Earnings Catalyst"]
    assert "Earnings TODAY: NVDA (AMC, eps_fc=$0.85) [HOLDING]" in pt_cal["Upcoming Macro / Earnings Catalyst"]
    # Empty cache dir remains unavailable (static dates alone must not fabricate a populated snapshot)
    empty_dir = tmp_path / "empty_cachedb"
    empty_dir.mkdir()
    assert collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(empty_dir), now_ts=ts_static_cpi) == {}

    # GlobalTelemetryCollector.update_macro_calendar persists snapshot cleanly
    monkeypatch.setattr(
        "intelligence.globaltelemetry_collector.collect_macro_calendar_snapshot",
        lambda cache_dir, **_kw: collect_macro_calendar_snapshot(
            cache_dir, now_ts=now_cpi_pre, persist=True, fetch_fn=_fake_fetch, tracked_equity_tickers=["NVDA"]
        ),
    )
    gtc_dir = tmp_path / "gtc_cache"
    (gtc_dir / "macro").mkdir(parents=True, exist_ok=True)
    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(gtc_dir), skip_external=True, skip_news=True)
    assert gtc.update_macro_calendar(force=True, now=now_cpi_pre) is True
    assert (gtc_dir / "macro" / "macro_calendar_collect.json").exists()

    # 7. Restart 429 guard (skip_if_fresh_sec) skips network when complete cache is fresh, heals partial failures,
    #    and live mode stamps post-fetch completion wall-clock ts (no pre-fetch backdating).
    fetch_calls = []

    def _counting_fetch(url, **_kw):
        fetch_calls.append(url)
        return _fake_fetch(url)

    fresh_skip = collect_macro_calendar_snapshot(
        str(gtc_dir),
        now_ts=now_cpi_pre + 300.0,
        persist=True,
        fetch_fn=_counting_fetch,
        tracked_equity_tickers=["NVDA"],
        skip_if_fresh_sec=1800.0,
    )
    assert fresh_skip is not None and fetch_calls == []

    # Partial-failure cache on disk (earnings_ok=False) bypasses skip_if_fresh_sec so it self-heals
    heal_snap = collect_macro_calendar_snapshot(
        str(cdir),
        now_ts=now_cpi_pre + 900.0,
        persist=True,
        fetch_fn=_counting_fetch,
        tracked_equity_tickers=["NVDA"],
        skip_if_fresh_sec=1800.0,
    )
    assert heal_snap is not None and heal_snap["earnings_ok"] is True and len(fetch_calls) == 3

    # Post-fetch completion wall-clock ts when now_ts=None (no pre-fetch backdating)
    import intelligence.sentiment.collectors.macro_telemetry_collector as mtc_mod

    clock_vals = iter([now_cpi_pre, now_cpi_pre + 4.5])
    monkeypatch.setattr(mtc_mod, "resolve_now_ts", lambda v=None: v if v is not None else next(clock_vals))
    live_ts_snap = collect_macro_calendar_snapshot(
        str(tmp_path / "live_ts_dir"),
        now_ts=None,
        persist=False,
        fetch_fn=_fake_fetch,
        tracked_equity_tickers=["NVDA"],
    )
    assert live_ts_snap is not None and live_ts_snap["ts"] == pytest.approx(now_cpi_pre + 4.5)


def test_prompt_telemetry_audit_fixes_critical_and_medium(tmp_path):
    """Regression suite covering all 9 Critical & Medium auto_cryptotrade_executor telemetry audit fixes."""
    from instrument_registry import resolve_symbol_identity
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        compute_futures_premarket_lead,
        format_macro_benchmarks_summary_line,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        compute_liquidation_clusters,
        enrich_microstructure_cvd,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _guard_quadratic_kinematic_overshoot,
        apply_24h_series_enrichment,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import enrich_open_limit_orders_tca
    from orchestratorOS.supervisor.crypto_executor_delta_gate import _evaluate_delta_gate_trade_plans
    from orchestratorOS.supervisor.prompt_sections import _format_tradable_position_line
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import summarize_active_ai_trade_plans
    from providers.order_id_context import resolve_order_actor
    from state_io import read_jsonl_records_since_span

    # 1. Identity cache coherence + frozenset actor memo + open-order age clock synchronization
    id_a = resolve_symbol_identity("SPCXXUSD")
    id_b = resolve_symbol_identity("SPCXXUSD")
    assert id_a is id_b and id_a.display_symbol == "SPCXxUSD" and id_a.is_tokenized_equity
    known_fset = frozenset({"727115018", "OYKJZC-OYNBR-DQUQJD"})
    assert resolve_order_actor({"orderId": "727115018"}, known_fset) == "AI"
    assert resolve_order_actor({"orderId": "999999"}, known_fset) == "BOT"

    t_now = 1_795_176_000.0
    oo_list = [{
        "venue": "binance",
        "symbol": "TAOUSDC",
        "side": "SELL",
        "price": 186.6,
        "qty": 1.95,
        "created_ts": t_now - 372.0,
        "age_min": 8.8,
        "age": "8.8m",
    }]
    enrich_open_limit_orders_tca(oo_list, {"TAOUSDC": {"price": 186.4, "vol_1h_pct": 1.2}}, {}, now_ts=t_now)
    assert oo_list[0]["age_min"] == 6.2 and oo_list[0]["age"] == "6.2m"

    # 2. Kinematics staleness (>45s) & sign-contradiction refresh + quadratic reversal overshoot clamp
    base_ms = int((t_now - 360.0) * 1000.0)
    falling_series = [[base_ms + i * 5000, round(64.5 - i * 0.01, 4)] for i in range(73)]
    ent_stale = {
        "price": falling_series[-1][1],
        "chg_1m_pct": -0.18,
        "slope_90s": -0.002,
        "gradient_recent_fast": -0.002,
        "vel_90s_pct_min": 0.25,
        "accel_90s_pct_min2": 0.10,
        "vel_5m_pct_min": 0.12,
        "accel_5m_pct_min2": 0.05,
        "_short_trend_ts": t_now - 90.0,
    }
    apply_24h_series_enrichment("HYPEUSDC", ent_stale, {"chg_1m_pct": -0.18}, falling_series, None)
    assert ent_stale["vel_90s_pct_min"] < 0.0 and ent_stale["vel_5m_pct_min"] < 0.0

    # Even with fresh short_trend_ts (<45s), contradictory positive vel_90s vs negative 1m & slope_90s is refreshed
    ent_contra = {
        "price": falling_series[-1][1],
        "chg_1m_pct": -0.18,
        "slope_90s": -0.002,
        "gradient_recent_fast": -0.002,
        "vel_90s_pct_min": 0.17,
        "accel_90s_pct_min2": 0.40,
        "_short_trend_ts": t_now - 10.0,
    }
    apply_24h_series_enrichment("ADAUSD", ent_contra, {"chg_1m_pct": -0.18}, falling_series, None)
    assert ent_contra["vel_90s_pct_min"] < 0.0

    pts_pullback = [(-5.0, 0.0), (-4.0, 0.15), (-3.0, 0.30), (-2.0, 0.42), (-1.0, 0.45), (-0.5, 0.43), (0.0, 0.41)]
    clamped_v, _ = _guard_quadratic_kinematic_overshoot(pts_pullback, -0.50, -0.40)
    assert abs(clamped_v) <= 0.16 and clamped_v < 0.0

    # 3. Leading zero-volume pre-IPO indicative row stripped on RTH equity 24h series (ANT_US_EQ)
    eq_pol = resolve_asset_session_policy("ANT_US_EQ")
    raw_ant = [
        [1_795_098_900_000, 83.0, 0.0],
        [1_795_106_700_000, 111.68, 160_677_756.0],
        [1_795_107_000_000, 111.51, 70_385_284.0],
        [1_795_120_000_000, 105.03, 51_622_553.0],
        [1_795_150_000_000, 101.22, 64_818_040.0],
        [1_795_176_000_000, 90.02, 12_000_000.0],
    ]
    cleaned_ant = eq_pol.clean_raw_24h_series(raw_ant, "ANT_US_EQ")
    assert cleaned_ant[0][1] == pytest.approx(111.68)
    m24_ant = compute_24h_price_metrics(cleaned_ant, is_equity=True)
    assert m24_ant["chg_24h_pct"] < -15.0 and m24_ant["low_24h"] == pytest.approx(90.02)

    # 4. Binary-seek 7d JSONL recovery when sparse archive < 5d and dense 365d file > 512KB
    dense_cdir = tmp_path / "dense_7d"
    (dense_cdir / "prices").mkdir(parents=True)
    end_ms = int(t_now * 1000.0)
    start_10d_ms = end_ms - 10 * 86400 * 1000
    f365 = dense_cdir / "prices" / "cache_365_price_SPCXxUSD.jsonl"
    with open(f365, "w", encoding="utf-8") as fh:
        for i in range(12_000):
            ts_i = start_10d_ms + i * 72_000
            px_i = 140.0 + (7.2 if i == 6000 else (-8.0 * i / 12_000.0))
            fh.write(json.dumps({"s": "SPCXxUSD", "i": [ts_i, round(px_i, 4)]}) + "\n")
    assert f365.stat().st_size > 512 * 1024
    recs_7d = read_jsonl_records_since_span(str(f365), 7.05 * 86400 * 1000, max_records=2000)
    assert len(recs_7d) >= 500
    assert (recs_7d[-1]["i"][0] - recs_7d[0]["i"][0]) >= 6.5 * 86400 * 1000

    # 5. Spot volume monotonicity (spot_vol_1h >= spot_vol_15m) & consistent RVOL scaling without avg_15m_base
    m_vol = {"spot_vol_15m_usd": 154_700.0}
    p_vol = {"spot_vol_15m_usd": 40_000.0, "spot_vol_1h_usd": 107_200.0, "rvol_1h": 0.54}
    enrich_microstructure_cvd(m_vol, p_vol)
    assert m_vol["spot_vol_1h_usd"] >= m_vol["spot_vol_15m_usd"]
    assert m_vol["rvol_1h"] > 0.54

    # 6. Exact-symbol derivatives precedence (HYPEUSDC L1 over newer HYPEUSD proxy) + L1 OI liquidation cluster scaling
    d_cdir = tmp_path / "deriv_pref"
    (d_cdir / "telemetry").mkdir(parents=True)
    (d_cdir / "telemetry" / "derivatives_telemetry_HYPEUSDC.json").write_text(
        json.dumps({"symbol": "HYPEUSDC", "open_interest_usd": 1_180_000_000.0, "funding_rate": 1.25e-5, "funding_annualized_pct": 10.95, "ts": t_now - 10.0}),
        encoding="utf-8",
    )
    (d_cdir / "telemetry" / "derivatives_telemetry_HYPEUSD.json").write_text(
        json.dumps({"symbol": "HYPEUSD", "open_interest_usd": 320_000_000.0, "funding_rate": 1.37e-5, "funding_annualized_pct": 1.50, "ts": t_now - 2.0}),
        encoding="utf-8",
    )
    (d_cdir / "telemetry" / "whale_snapshot_HYPEUSDC.json").write_text(
        json.dumps({"symbol": "HYPEUSDC", "open_interest_usd": 320_000_000.0, "top_traders_long_pct": 0.70, "taker_buy_vol_usd": 3_000_000.0, "taker_sell_vol_usd": 4_000_000.0, "ts": t_now - 5.0}),
        encoding="utf-8",
    )
    m_hype = load_symbol_microstructure(str(d_cdir), "HYPEUSDC")
    assert m_hype is not None
    assert m_hype["native_l1_oi_usd"] == pytest.approx(1_180_000_000.0)
    assert m_hype["funding_annualized_pct"] == pytest.approx(10.95)
    liq_l1 = compute_liquidation_clusters({"price": 63.5, "vol_1h_pct": 1.5}, m_hype, symbol="HYPEUSDC")
    assert liq_l1["long_liq_usd"] > 8_000_000.0

    # 7. Multi-tranche resting Active AI Trade Plans + severity-preserving delta-gate state aggregation
    multi_intents = [
        {
            "intent_id": "tao_sell_1",
            "timestamp": t_now - 600.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 186.6,
            "invalidation_px": 189.6,
            "target_px": 183.5,
            "suggested_notional_usd": 365.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "727115018", "status": "NEW"},
        },
        {
            "intent_id": "tao_sell_2",
            "timestamp": t_now - 300.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 189.2,
            "invalidation_px": 192.5,
            "target_px": 184.0,
            "suggested_notional_usd": 371.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "727115947", "status": "NEW"},
        },
    ]
    open_tao = [
        {"venue": "binance", "symbol": "TAOUSDC", "side": "SELL", "order_id": "727115018", "price": 186.6, "qty": 1.95, "status": "NEW"},
        {"venue": "binance", "symbol": "TAOUSDC", "side": "SELL", "order_id": "727115947", "price": 189.2, "qty": 1.96, "status": "NEW"},
    ]
    multi_plans = summarize_active_ai_trade_plans(
        multi_intents, "crypto", t_now, extract_intent_notional, open_orders=open_tao, prices_and_trends={"TAOUSDC": {"price": 190.0}}
    )
    assert len(multi_plans) == 2
    assert {p["entry_px"] for p in multi_plans} == {186.6, 189.2}
    p_states, p_trig = _evaluate_delta_gate_trade_plans(
        {"active_ai_trade_plans": multi_plans}, {"TAOUSDC": 190.0}, {"binance:TAOUSDC:SELL": "ACTIVE"}
    )
    assert p_states["binance:TAOUSDC:SELL"] == "INVALIDATED" and "trade_plan_invalidated" in p_trig

    # 8. Futures_Lead suppressed in live US Pre-Market when QQQ/SPY already price the Globex move
    premarket_live_mb = {
        "items": {
            "SPY": {"price": 710.63, "chg_24h_pct": -3.80, "chg_1h_pct": -0.15, "chg_15m_pct": -0.01, "market_state": "PRE", "last_bar_ts": t_now - 60.0},
            "QQQ": {"price": 635.18, "chg_24h_pct": -5.58, "chg_1h_pct": -0.07, "chg_15m_pct": 0.10, "market_state": "PRE", "last_bar_ts": t_now - 60.0},
            "ES=F": {"price": 7139.5, "chg_24h_pct": -3.82, "globex_session_chg_pct": -1.48, "chg_1h_pct": -0.20, "chg_15m_pct": -0.04, "last_bar_ts": t_now - 120.0},
            "NQ=F": {"price": 26133.75, "chg_24h_pct": -5.76, "globex_session_chg_pct": -1.75, "chg_1h_pct": -0.25, "chg_15m_pct": -0.11, "last_bar_ts": t_now - 120.0},
        },
        "futures_lead_pct": -1.75,
        "futures_lead_regime": "FUTURES_PREMARKET_GAP_DOWN",
    }
    assert compute_futures_premarket_lead(premarket_live_mb) == {}
    assert "Futures_Lead=" not in format_macro_benchmarks_summary_line(premarket_live_mb)

    # 9. venue_fee_per_leg reports conservative upper bound max(maker, taker) for Binance VIP0+BNB USDC (7.5 > 7.12)
    bn_line = _format_tradable_position_line({
        "venue": "binance",
        "symbol": "BTCUSDC",
        "base": "BTC",
        "quote": "USDC",
        "price": 61500.0,
        "free_base": 0.03,
        "value_usd": 1845.0,
        "free_quote": 5000.0,
        "fee_tier": "VIP0_BNB",
        "maker_fee_bps": 7.5,
        "taker_fee_bps": 7.12,
        "spread_bps": 0.01,
    })
    assert "venue_fee_per_leg=7.5bps (tier=VIP0_BNB, maker=7.5bps/taker=7.12bps" in bn_line


def test_faza3a_prep_ssot_seams_zero_behavior_change(tmp_path):
    """Verify Faza 3A SSOT & SoC infrastructure seams preserve exact runtime behavior while unifying equity session & Yahoo chart parsing."""
    from datetime import datetime
    from botcore import ET, is_us_extended_hours_open, is_us_rth_open, resolve_us_equity_session_state
    from cacheprice import Cache24PriceManager, CacheCurrentPriceManager
    from instrument_registry import is_symbol_market_open, resolve_symbol_market_session
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        EQUITY_QUOTE_META_KEYS,
        format_equity_quote_meta_suffix,
    )
    from providers.ohlc_utils import extract_yahoo_chart_latest_price, extract_yahoo_chart_ohlcv_bars

    # 1. Canonical US equity session state SSOT (Mon-Fri ET windows + weekend CLOSED)
    ts_pre = datetime(2026, 11, 18, 7, 15, tzinfo=ET).timestamp()
    ts_rth = datetime(2026, 11, 18, 11, 30, tzinfo=ET).timestamp()
    ts_post = datetime(2026, 11, 18, 17, 45, tzinfo=ET).timestamp()
    ts_night = datetime(2026, 11, 18, 22, 0, tzinfo=ET).timestamp()
    ts_sat = datetime(2026, 11, 21, 12, 0, tzinfo=ET).timestamp()

    assert resolve_us_equity_session_state(ts_pre) == "PRE"
    assert resolve_us_equity_session_state(ts_rth) == "REGULAR"
    assert resolve_us_equity_session_state(ts_post) == "POST"
    assert resolve_us_equity_session_state(ts_night) == "CLOSED"
    assert resolve_us_equity_session_state(ts_sat) == "CLOSED"

    assert is_us_rth_open(ts_rth) is True
    assert is_us_rth_open(ts_pre) is False and is_us_rth_open(ts_post) is False and is_us_rth_open(ts_sat) is False
    assert is_us_extended_hours_open(ts_pre) is True and is_us_extended_hours_open(ts_post) is True
    assert is_us_extended_hours_open(ts_rth) is False and is_us_extended_hours_open(ts_night) is False

    # Symbol-level session SSOT keeps 24X7 for crypto/tokenized xStocks and delegates equities
    assert resolve_symbol_market_session("BTCUSDC", ts_pre) == "24X7"
    assert resolve_symbol_market_session("SPCXxUSD", ts_post) == "24X7"
    assert resolve_symbol_market_session("NVDA_US_EQ", ts_pre) == "PRE"
    assert is_symbol_market_open("NVDA_US_EQ", ts_rth) is True
    assert is_symbol_market_open("NVDA_US_EQ", ts_pre) is False

    # 2. Polymorphic AssetSessionPolicy contract
    crypto_pol = resolve_asset_session_policy("BTCUSDC")
    tok_pol = resolve_asset_session_policy("SPCXxUSD")
    eq_pol = resolve_asset_session_policy("NVDA_US_EQ")
    assert crypto_pol.supports_extended_hours is False and crypto_pol.resolve_session_state(ts_pre) == "24X7"
    assert tok_pol.supports_extended_hours is False and tok_pol.resolve_session_state(ts_post) == "24X7"
    assert eq_pol.supports_extended_hours is True and eq_pol.resolve_session_state(ts_pre) == "PRE"
    assert eq_pol.resolve_session_state(ts_rth) == "REGULAR" and eq_pol.resolve_session_state(ts_post) == "POST"

    # 3. Unified Yahoo v8 chart OHLCV & opt-in extended-hours price extraction
    yh_payload = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "marketState": "PRE",
                        "regularMarketPrice": 140.0,
                        "preMarketPrice": 144.25,
                    },
                    "timestamp": [1_795_000_600, 1_795_000_300],
                    "indicators": {
                        "quote": [
                            {
                                "open": [140.0, 139.0],
                                "high": [141.0, 140.5],
                                "low": [139.5, 138.8],
                                "close": [140.2, 139.8],
                                "volume": [1200, 800],
                            }
                        ]
                    },
                }
            ]
        }
    }
    meta, bars = extract_yahoo_chart_ohlcv_bars(yh_payload)
    assert isinstance(meta, dict) and len(bars) == 2
    assert bars[0][0] < bars[1][0] and bars[0][4] == pytest.approx(139.8) and bars[1][4] == pytest.approx(140.2)
    assert extract_yahoo_chart_latest_price(yh_payload) == pytest.approx(139.8)
    assert extract_yahoo_chart_latest_price(yh_payload, include_extended=True, now_ts=ts_pre) == pytest.approx(144.25)

    # 4. Cache24PriceManager & CacheCurrentPriceManager Faza 3A extended-hours cadence & CLOSED gate
    p24 = Cache24PriceManager(60, ["NVDA_US_EQ"], str(tmp_path / "p24.json"), data_sync_mode="disk_reader")
    assert p24._resolve_equity_ohlc_refresh_interval_sec("NVDA_US_EQ", ts_rth, row_count=50, has_pos_vol_ohlc=True) == (
        True,
        p24.EQUITY_OHLC_RTH_REFRESH_SEC,
    )
    assert p24._resolve_equity_ohlc_refresh_interval_sec("NVDA_US_EQ", ts_pre, row_count=50, has_pos_vol_ohlc=True) == (
        False,
        round(p24.EQUITY_OHLC_RTH_REFRESH_SEC * 2.0, 1),
    )
    assert p24._resolve_equity_ohlc_refresh_interval_sec("NVDA_US_EQ", ts_post, row_count=50, has_pos_vol_ohlc=True) == (
        False,
        round(p24.EQUITY_OHLC_RTH_REFRESH_SEC * 2.0, 1),
    )
    assert p24._resolve_equity_ohlc_refresh_interval_sec("NVDA_US_EQ", ts_night, row_count=50, has_pos_vol_ohlc=True) == (
        False,
        p24.EQUITY_OHLC_OFF_RTH_REFRESH_SEC,
    )
    cp = CacheCurrentPriceManager(60, ["NVDA_US_EQ"], str(tmp_path / "cp.json"), data_sync_mode="disk_reader")
    cp._enforce_rth_gate = True
    # Stale cached quote from previous session -> polls during RTH and PRE/POST, stays paused during CLOSED night/weekend
    cp.cache["NVDA_US_EQ"] = [[int((ts_pre - 3600.0) * 1000), 140.0]]
    assert cp._should_skip_equity_poll_for_session("NVDA_US_EQ", ts_rth) is False
    assert cp._should_skip_equity_poll_for_session("NVDA_US_EQ", ts_pre) is False
    assert cp._should_skip_equity_poll_for_session("NVDA_US_EQ", ts_post) is False
    assert cp._should_skip_equity_poll_for_session("NVDA_US_EQ", ts_night) is True
    assert cp._should_skip_equity_poll_for_session("NVDA_US_EQ", ts_sat) is True
    # Fresh extended-hours quote within ext_gap_sec throttles polling until cadence elapses
    cp.cache["NVDA_US_EQ"] = [[int((ts_pre - 10.0) * 1000), 144.25]]
    assert cp._should_skip_equity_poll_for_session("NVDA_US_EQ", ts_pre) is True

    # 5. Equity quote metadata SSOT preserves byte-identical legacy formatting and supports PRE/POST tags
    assert "quote_source" in EQUITY_QUOTE_META_KEYS and "market_state" in EQUITY_QUOTE_META_KEYS
    assert format_equity_quote_meta_suffix(
        {"quote_source": "yahoo_ext", "quote_age_min": 2.5, "t212_close_px": 140.0}
    ) == " [src=yahoo_ext, age=2.5m, t212_close=$140.0]"
    assert format_equity_quote_meta_suffix(
        {"market_state": "PRE", "quote_source": "yahoo_ext", "quote_age_min": 1.0, "ext_vs_rth_pct": 2.15, "t212_close_px": 140.0}
    ) == " [PRE, src=yahoo_ext, age=1.0m, ext_vs_rth=+2.15%, t212_close=$140.0]"


def test_prompt_telemetry_audit_fixes_low_issues(tmp_path):
    """Regression coverage for LOW prompt audit fixes (#11 tokenized vs_btc, #12 news path & noise, #13 BNBUSDC fee reserve)."""
    from intelligence.macro.lexicon_ssot import (
        has_financial_or_geopolitical_context,
        is_excluded_reference_news_item,
        is_off_topic_general_news_title,
    )
    from intelligence.sentiment.collectors.portfolio_collector import harvest_recent_orders_and_enrich_positions
    from intelligence.sentiment.evaluators.price_series_evaluator import apply_24h_series_enrichment
    from orchestratorOS.supervisor.prompt_sections import (
        _format_macro_and_news_section,
        _format_market_prices_section,
    )
    from state_io import atomic_write_json

    t_now = 1_795_180_000.0

    # 11. Tokenized xStocks (SPCXxUSD, ANTxUSD) skip cross-BTC calculation and omit vs_btc / alt_btc_24h in Section 3
    btc_24h = [[int((t_now - (24 - i) * 3600) * 1000), 60000.0 + i * 50.0] for i in range(25)]
    spcx_24h = [[int((t_now - (24 - i) * 3600) * 1000), 140.0 + i * 0.2] for i in range(25)]
    spcx_entry = {"price": 144.8, "chg_15m_pct": 0.1, "chg_1h_pct": 0.3, "chg_24h_pct": 3.4}
    apply_24h_series_enrichment("SPCXxUSD", spcx_entry, {"chg_24h_pct": 3.4}, spcx_24h, btc_24h)
    assert "beta_btc_24h" not in spcx_entry
    assert "corr_btc_24h" not in spcx_entry
    assert "alt_btc_pos_24h_pct" not in spcx_entry

    sec3_lines = _format_market_prices_section(
        {
            "BTCUSDC": {"price": 61200.0, "chg_15m_pct": 0.05, "chg_1h_pct": 0.2, "chg_24h_pct": 2.0},
            "TAOUSDC": {"price": 185.0, "chg_15m_pct": 0.4, "chg_1h_pct": 1.1, "chg_24h_pct": 4.0, "beta_btc_24h": 1.4, "corr_btc_24h": 0.72},
            "SPCXxUSD": {"price": 144.8, "chg_15m_pct": 0.1, "chg_1h_pct": 0.3, "chg_24h_pct": 3.4, "beta_btc_24h": 0.9, "corr_btc_24h": 0.5},
            "ANTxUSD": {"price": 51.0, "chg_15m_pct": -0.2, "chg_1h_pct": -0.5, "chg_24h_pct": -2.0},
        },
    )
    sec3_by_sym = {ln.split(":", 1)[0].replace("* ", "").strip(): ln for ln in sec3_lines if ln.startswith("  * ")}
    assert "vs_btc(" in sec3_by_sym["TAOUSDC"] and "alt_btc_24h(" in sec3_by_sym["TAOUSDC"]
    assert "vs_btc(" not in sec3_by_sym["SPCXxUSD"] and "alt_btc_24h(" not in sec3_by_sym["SPCXxUSD"]
    assert "vs_btc(" not in sec3_by_sym["ANTxUSD"] and "alt_btc_24h(" not in sec3_by_sym["ANTxUSD"]

    # 12. Breaking News Feed path reflects cachedb/macro/news_feed_collect.json and filters personal-finance/fashion/sports-drone noise
    macro_snap = SimpleNamespace(
        macro_and_sentiment={},
        fear_greed=None,
        market_breadth=None,
        sentiment_advisor=None,
        geopolitical_threat=None,
        portfolio_summary={},
        breaking_news=[{"title": "Treasury yields fall sharply as investors seek safe havens", "age_min": 5.0}],
        macro_benchmarks=None,
        macro_calendar=None,
    )
    sec3a_txt = "\n".join(_format_macro_and_news_section(macro_snap, asset_scope="crypto"))
    assert "cachedb/macro/news_feed_collect.json" in sec3a_txt

    assert is_excluded_reference_news_item(
        "I’m terrified of a market crash. Should I put $400,000 of my $1 million retirement savings into an annuity?",
        "MarketWatch",
        "",
    )
    assert is_excluded_reference_news_item(
        "These 5 defense stocks are drawing millions in insider bets ahead of military-spending boom in 2027",
        "MarketWatch",
        "",
    )
    drone_sports = "Drone operator removes Man United training footage and issues apology"
    assert has_financial_or_geopolitical_context(drone_sports) is False
    assert is_off_topic_general_news_title(drone_sports) is True
    fashion_fluff = "Rejecting cheap fabrics, higher prices: Why shoppers are asking for better clothes"
    assert has_financial_or_geopolitical_context(fashion_fluff) is False
    assert is_off_topic_general_news_title(fashion_fluff) is True
    # Genuine geopolitical drone strike headlines remain high-relevance financial/geopolitical context
    assert has_financial_or_geopolitical_context("Iranian drone strike targets Gulf tanker near Strait of Hormuz") is True

    # 13. Non-tradable BNBUSDC fee-reserve fills are excluded from recent_orders unless BNBUSDC is actively configured/tradable
    cdir = tmp_path / "cachedb"
    ord_dir = cdir / "orders"
    ord_dir.mkdir(parents=True, exist_ok=True)
    orders_payload = {
        "items": {
            "BNBUSDC": [
                {"orderId": 9001, "symbol": "BNBUSDC", "side": "BUY", "status": "FILLED", "price": 620.0, "quantity": 0.32, "timestamp": int((t_now - 3600) * 1000)}
            ],
            "BTCUSDC": [
                {"orderId": 9002, "symbol": "BTCUSDC", "side": "BUY", "status": "FILLED", "price": 61000.0, "quantity": 0.01, "timestamp": int((t_now - 1800) * 1000)}
            ],
        }
    }
    atomic_write_json(str(ord_dir / "cache_order_binance.json"), orders_payload)
    res_excl = harvest_recent_orders_and_enrich_positions(
        str(cdir),
        inst_by_sym={"BTCUSDC": SimpleNamespace(symbol="BTCUSDC", provider_name="binance")},
        tradable_positions=[],
        prices_and_trends={"BTCUSDC": {"price": 61200.0}},
        kraken_cycles={},
        hyperliquid_cycles={},
        t212_cycles={},
        allow_network=False,
    )
    assert "BTCUSDC" in res_excl
    assert "BNBUSDC" not in res_excl

    # If BNBUSDC IS explicitly configured as a tradable instrument, it must remain visible
    res_incl = harvest_recent_orders_and_enrich_positions(
        str(cdir),
        inst_by_sym={
            "BTCUSDC": SimpleNamespace(symbol="BTCUSDC", provider_name="binance"),
            "BNBUSDC": SimpleNamespace(symbol="BNBUSDC", provider_name="binance"),
        },
        tradable_positions=[],
        prices_and_trends={"BTCUSDC": {"price": 61200.0}, "BNBUSDC": {"price": 622.0}},
        kraken_cycles={},
        hyperliquid_cycles={},
        t212_cycles={},
        allow_network=False,
    )
    assert "BNBUSDC" in res_incl


def test_faza3a_extended_hours_equities_live_pricing_and_xstock_basis(tmp_path, monkeypatch):
    """Verify Faza 3A end-to-end: Yahoo PRE/POST live pricing in market_data, MarketApi, base_collector, microstructure_collector, and KrakenXstocksCollector."""
    from datetime import datetime
    from types import SimpleNamespace
    from botcore import ET
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.sentiment.collectors.microstructure_collector import Fleet24hAndMicrostructureEnricher
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from orchestratorOS.supervisor.prompt_sections import _format_market_prices_section
    from providers.market_api import MarketApi
    from state_io import save_json_state

    ts_pre = datetime(2026, 11, 18, 7, 30, tzinfo=ET).timestamp()
    ts_post = datetime(2026, 11, 18, 17, 30, tzinfo=ET).timestamp()

    # 1. 212trading/market_data.get_price_usd returns live PRE/POST extended price over stale RTH bars
    md_mod = load_venue_modules("212trading", "market_data")
    monkeypatch.setattr(md_mod, "_can_use_ssot_equity_cache", lambda _sym: None)
    monkeypatch.setattr(
        md_mod,
        "_chart",
        lambda _sym, **_kw: (
            {"marketState": "PRE", "regularMarketPrice": 140.0, "preMarketPrice": 145.6},
            [(int(ts_pre - 40000), 140.0, 5000.0)],
        ),
    )
    assert md_mod.get_price_usd("NVDA") == pytest.approx(145.6)

    # 2. MarketApi.get_current_price(..., prefer_public_extended=True) uses polymorphic fetch_public_market_price_usd
    class _FakeEqProvider:
        name = "T212"
        _ASSET_SCOPE = "equities"

        def supports_symbol(self, _s):
            return True

        def get_current_price(self, _s):
            return 140.0  # frozen broker RTH close

        @classmethod
        def fetch_public_market_price_usd(cls, _s):
            return 145.6  # live Yahoo PRE quote

    api = MarketApi(providers=[_FakeEqProvider()])
    assert api.get_current_price("NVDA_US_EQ") == pytest.approx(140.0)
    assert api.get_current_price("NVDA_US_EQ", prefer_public_extended=True) == pytest.approx(145.6)

    # 3. _DiskBackedSymbolCollector._fetch_equity_market_context captures live POST price & ext_vs_rth_pct
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path / "eq_ctx"), cache_ttl_sec=0.0, disk_ttl_sec=0.0, start_sync=True)
    ob_col._EQUITY_CTX_MEMO.clear()

    def _fake_eq_fetch(url, label, _sym, *, timeout=2.5, user_agent=""):
        if label == "nasdaq_quote":
            return {"data": {"primaryData": {"lastSalePrice": "$140.00", "volume": "50,000,000"}}}
        if label == "yahoo_chart_5m":
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "marketState": "POST",
                                "regularMarketPrice": 140.0,
                                "postMarketPrice": 147.0,
                            },
                            "timestamp": [int(ts_post - 600), int(ts_post - 300)],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [146.0, 146.5],
                                        "high": [146.8, 147.2],
                                        "low": [145.8, 146.4],
                                        "close": [146.5, 147.0],
                                        "volume": [1000, 1200],
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected call: {label} {url}")

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_eq_fetch)
    eq_ctx = ob_col._fetch_equity_market_context("NVDA_US_EQ", ts_post, include_options=False)
    assert eq_ctx is not None
    assert eq_ctx["last_price"] == pytest.approx(147.0)
    assert eq_ctx["market_state"] == "POST"
    assert eq_ctx["regular_close_price"] == pytest.approx(140.0)
    assert eq_ctx["extended_price"] == pytest.approx(147.0)
    assert eq_ctx["ext_vs_rth_pct"] == pytest.approx(5.0)

    # 4. Fleet24hAndMicrostructureEnricher._seed_t212_positions keeps fresher extended-hours cache_current_price
    #    over older frozen pos:t212 close and renders [PRE, src=yahoo_ext, ..., ext_vs_rth=+4.00%, t212_close=$140.0] in Section 3A
    pt_map = {"NVDA_US_EQ": {"price": 145.6, "trend": 0, "growth_coefficient": 0.0}}
    cp_map = {"NVDA_US_EQ": [[int((ts_pre - 30.0) * 1000), 145.6]]}
    cached_bals = {"pos:t212:NVDA_US_EQ": {"symbol": "NVDA_US_EQ", "free": 2.0, "price": 140.0, "ts": ts_pre - 36000.0}}
    Fleet24hAndMicrostructureEnricher._seed_t212_positions(
        "",
        pt_map,
        cp_map,
        {},
        cached_bals,
        "",
        0.0,
        ts_pre,
        lambda _p: None,
        False,
    )
    nv_entry = pt_map["NVDA_US_EQ"]
    assert nv_entry["price"] == pytest.approx(145.6)
    assert nv_entry["quote_source"] == "yahoo_ext"
    assert nv_entry["market_state"] == "PRE"
    assert nv_entry["t212_close_px"] == pytest.approx(140.0)
    assert nv_entry["ext_vs_rth_pct"] == pytest.approx(4.0)
    sec3a = "\n".join(_format_market_prices_section(pt_map))
    assert "[PRE, src=yahoo_ext, age=0.5m, ext_vs_rth=+4.00%, t212_close=$140.0]" in sec3a

    # 5. KrakenXstocksCollector surfaces underlying live PRE/POST price, session, and xstock premium/discount basis
    k_dir = tmp_path / "kraken_xst"
    k_dir.mkdir()
    save_json_state(
        str(k_dir / "xstock_state.json"),
        {"known_assets": {"SPCXx": 10.0}, "pair": "SPCXxUSD"},
    )
    xst_res = KrakenXstocksCollector(start_sync=False).collect(
        str(k_dir),
        kraken_dir=str(k_dir),
        prices_and_trends={
            "SPCXxUSD": {"price": 153.0},
            "SPCX_US_EQ": {"price": 150.0, "market_state": "PRE", "quote_source": "yahoo_ext"},
        },
        all_t212_cycles={},
    )
    assert xst_res is not None
    assert xst_res["live_price"] == pytest.approx(153.0)
    assert xst_res["underlying_live_price"] == pytest.approx(150.0)
    assert xst_res["underlying_session"] == "PRE"
    assert xst_res["xstock_vs_underlying_pct"] == pytest.approx(2.0)


def test_kinematic_reversal_overshoot_parity_and_filled_limit_px_precision():
    """Verify PriceWindow & series kinematics clamp partial-tail reversal ringing to physical speed, refresh contradictory 5m kinematics, and preserve clean submitted limit_px on FILLED Kraken orders."""
    import pricewindow as pw
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        apply_24h_series_enrichment,
        compute_series_kinematics,
    )
    from providers.intent_telemetry import build_last_cycle_execution_status
    from providers.intent_state import extract_intent_notional

    # 1. Step-drop (-0.25% early) + tiny +0.11% late bounce + flat tail: must NOT explode to >+0.40%/m
    t0_ms = 1_795_180_000_000
    step_prices = (
        [0.14611] * 20
        + [0.14575] * 40
        + [0.14580] * 10
        + [0.14585] * 8
        + [0.14594] * 13
    )
    win = pw.PriceWindow("ADAUSD", 91, sample_rate_sec=1.0, window_seconds=90.0)
    for px in step_prices:
        win.process_price(px)
    v90_pw, a90_pw = win.calculate_acceleration(window_sec=90.0)
    assert 0.0 < v90_pw <= 0.33
    assert a90_pw > 0.0

    series_90 = [[t0_ms + i * 1000, px] for i, px in enumerate(step_prices)]
    kin_s = compute_series_kinematics(series_90)
    assert 0.0 < kin_s["vel_90s_pct_min"] <= 0.33
    assert abs(v90_pw - kin_s["vel_90s_pct_min"]) < 0.06

    # 2. Stale positive vel_5m contradicting negative 5m & 1m tape triggers live series_24h kinematics refresh
    s_5m = [[t0_ms + i * 1000, 63.20 - 0.0002 * i] for i in range(301)]
    ent = {
        "price": s_5m[-1][1],
        "chg_1m_pct": -0.13,
        "chg_5m_pct": -0.09,
        "slope_90s": 0.00034,
        "vel_90s_pct_min": -0.2642,
        "accel_90s_pct_min2": -0.2798,
        "vel_5m_pct_min": 0.1183,
        "accel_5m_pct_min2": 0.0352,
    }
    apply_24h_series_enrichment("HYPEUSDC", ent, {"chg_1m_pct": -0.13, "chg_5m_pct": -0.09}, s_5m, None)
    assert ent["vel_5m_pct_min"] < 0.0
    assert ent["vel_90s_pct_min"] < 0.0
    assert ent["slope_90s"] < 0.0

    # 3. FILLED Kraken LIMIT order keeps exact submitted limit_price (130.07) instead of cost/vol float dust (130.070001)
    now_s = t0_ms / 1000.0 + 300.0
    st_filled = build_last_cycle_execution_status(
        [
            {
                "timestamp": now_s - 600.0,
                "decision": "SELL",
                "symbol": "SPCXxUSD",
                "venue": "kraken",
                "order_type": "LIMIT",
                "limit_price": 130.07,
                "reference_price": 130.07,
                "suggested_notional_usd": 646.31,
                "status": "EXECUTED",
                "execution_payload": {
                    "orderId": "OLZEGS-C2J2M-OTMPY2",
                    "status": "FILLED",
                    "price": 130.070001,
                    "filled_qty": 4.968943,
                    "fill_ts": now_s - 300.0,
                },
            }
        ],
        "crypto",
        now_s,
        extract_intent_notional,
        open_orders=[],
    )
    assert st_filled is not None
    assert st_filled["limit_px"] == pytest.approx(130.07)
    assert st_filled["fill_px"] == pytest.approx(130.070001)

    # 4. Historical WWII war-crimes essays must not match [HIGH-SEV] 'war' or rank as macro/geo breaking news
    from intelligence.macro.lexicon_ssot import (
        filter_valid_severity_keywords,
        has_financial_or_geopolitical_context,
        is_off_topic_general_news_title,
        match_high_severity_keywords,
    )
    from intelligence.globaltelemetry_collector import evaluate_kinematic_impulse_symbols

    nazi_title = "When Nazis were held responsible for war crimes"
    assert match_high_severity_keywords(nazi_title) == ()
    assert filter_valid_severity_keywords(["war"], title=nazi_title) == ()
    assert has_financial_or_geopolitical_context(nazi_title) is False
    assert is_off_topic_general_news_title(nazi_title) is True

    # 5. Kinematic impulse wakeup suppresses counter-trend micro-bounces and flat-plateau single ticks when slope_90s disproves them
    imp_res = evaluate_kinematic_impulse_symbols(
        {
            # Counter-trend micro-bounce: slope_90s negative (-0.17%/m), vel_90s positive (+0.39%/m), vel_5m negative -> ignored
            "ADAUSD": {
                "price": 0.14488,
                "slope_90s": -4.01e-06,
                "vel_90s_pct_min": 0.3931,
                "accel_90s_pct_min2": 0.6567,
                "vel_5m_pct_min": -0.1097,
            },
            # Flat-plateau +18bps single tick: slope_90s only +0.09%/m (< 0.15%/m) -> ignored
            "ARBUSDC": {
                "price": 0.1172,
                "slope_90s": 1.75e-06,
                "vel_90s_pct_min": 0.41,
                "accel_90s_pct_min2": 0.52,
                "vel_5m_pct_min": 0.04,
            },
            # Genuine corroborated crash: slope_90s -0.28%/m, vel_90s -0.39%/m -> triggers!
            "HYPEUSDC": {
                "price": 62.48,
                "slope_90s": -0.0029,
                "vel_90s_pct_min": -0.39,
                "accel_90s_pct_min2": -0.28,
                "vel_5m_pct_min": -0.26,
            },
        }
    )
    assert imp_res == ["HYPEUSDC"]


def test_faza3b_prep_ssot_seams_zero_behavior_change(tmp_path, monkeypatch):
    """Verify Faza 3B infrastructure prep SSOT seams: fetch_public_text, macro cache constants/registry, breadth serializer, compact USD formatter, and collector/daemon lifecycle helpers."""
    import botcore
    import cachepaths
    import core.http as core_http
    import intelligence.macro_sentiment_ssot as mss
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        MacroAndNewsSensorsCollector,
        MacroEconomicCalendarCollector,
        MacroTelemetryCollectorBase,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import format_compact_usd
    import intelligence.sentiment.sentiment_advisor as sa_mod
    from orchestratorOS.lib.technical_repair_ssot import build_sensor_health_summary
    from orchestratorOS.supervisor.prompt_sections import (
        _fmt_liq_usd,
        _fmt_signed_usd,
        _resolve_macro_sensor,
    )

    # 1. Level-0 fetch_public_text + MacroTelemetryCollectorBase._resolve_text_fetch_fn monkeypatch resolution
    monkeypatch.setattr(core_http, "fetch_public_bytes", lambda url, **_kw: f"CSV:{url}".encode("utf-8"))
    assert botcore.fetch_public_text("https://example.test/etf.html") == "CSV:https://example.test/etf.html"
    monkeypatch.setattr(mss, "fetch_public_text", lambda url, **_kw: f"PATCHED:{url}", raising=False)
    assert MacroTelemetryCollectorBase._resolve_text_fetch_fn()("https://example.test/x") == "PATCHED:https://example.test/x"
    assert MacroTelemetryCollectorBase._resolve_text_fetch_fn(lambda u, **_kw: f"EXPLICIT:{u}")("u") == "EXPLICIT:u"

    # 2. Macro cache filename constants & declarative sensor registry parity
    macro_exact = set(cachepaths.CACHEDB_SUBDOMAINS["macro"]["exact"])
    for fname in (
        cachepaths.CACHE_FEAR_GREED_FILENAME,
        cachepaths.CACHE_MARKET_BREADTH_FILENAME,
        cachepaths.CACHE_MACRO_BENCHMARKS_FILENAME,
        cachepaths.CACHE_MACRO_CALENDAR_FILENAME,
        cachepaths.CACHE_MACRO_LEXICON_FILENAME,
        cachepaths.CACHE_NEWS_FEED_FILENAME,
        cachepaths.CACHE_SENTIMENT_EVAL_FILENAME,
        cachepaths.CACHE_GEOPOLITICAL_EVAL_FILENAME,
    ):
        assert fname in macro_exact
    assert MacroAndNewsSensorsCollector._MACRO_SENSOR_CACHE_SPECS == mss.MACRO_SENSOR_CACHE_SPECS
    assert mss.MACRO_SENSOR_HEALTH_KEYS == ("fear_greed", "market_breadth", "sentiment_advisor", "geopolitical_threat", "macro_benchmarks", "etf_flows")
    assert mss.CRYPTO_ONLY_MACRO_SENSOR_KEYS == frozenset({"fear_greed", "market_breadth", "sentiment_advisor", "etf_flows"})
    assert "macro_calendar" in mss.SYNTHETIC_MACRO_SENSOR_KEYS
    sh = build_sensor_health_summary(SimpleNamespace(fear_greed={"value": 50}, macro_benchmarks={"items": {}}))
    assert sh["fear_greed_present"] is True and sh["macro_benchmarks_present"] is True and sh["market_breadth_present"] is False and sh["etf_flows_present"] is False

    # 3. Unified MarketBreadthSnapshot serialization + fresh guard + cached macro snapshot loader
    t_now = 1_795_200_000.0
    mb_snap = mss.MarketBreadthSnapshot(
        advance_ratio=0.62,
        advancing_count=62,
        declining_count=38,
        total_count=100,
        median_change_pct=1.25,
        mean_change_pct=1.40,
        dispersion_std=2.1,
        regime="BULLISH_BREADTH",
        top_gainers=(("BTCUSDT", 4.5),),
        top_losers=(("XRPUSDT", -2.1),),
        ts=t_now,
        btc_price=95000.0,
        btc_vol_share_pct=41.2,
        rotation_regime="ALT_INFLOW_ROTATION",
    )
    mb_dict = mb_snap.to_cache_dict(now_ts=t_now)
    assert mb_dict["total_symbols"] == 100 and mb_dict["total_count"] == 100
    assert mb_dict["top_gainers"] == [["BTCUSDT", 4.5]] and mb_dict["rotation_regime"] == "ALT_INFLOW_ROTATION"
    assert mb_dict == mss.serialize_market_breadth_snapshot(mb_snap, now_ts=t_now)
    assert mss.resolve_macro_fresh_guard_sec(21600.0) == pytest.approx(1800.0)
    assert mss.resolve_macro_fresh_guard_sec(3600.0) == pytest.approx(900.0)

    # 4. Unified compact USD formatter across macro_telemetry_evaluator, sentiment_advisor, and prompt_sections
    assert sa_mod._fmt_compact_usd is format_compact_usd
    assert format_compact_usd(1_250_000_000.0) == "$1.25B"
    assert format_compact_usd(-185_400_000.0, signed=True) == "-$185.40M"
    assert format_compact_usd(312_500_000.0, signed=True) == "+$312.50M"
    assert format_compact_usd(42_500.0) == "$42.5K"
    assert format_compact_usd(-0.2, signed=True) == "+$0"
    assert format_compact_usd(-0.2, signed=False) == "$0"
    assert _fmt_liq_usd(2_500_000.0) == "$2.50M"
    assert _fmt_signed_usd(-2_500_000.0) == "-$2.50M"

    # 5. MacroTelemetryCollectorBase lifecycle helpers, backfill helper, GlobalTelemetryCollector wrapper & Section 1 resolver
    cdir = tmp_path / "cachedb"
    cdir.mkdir()
    assert MacroEconomicCalendarCollector._build_fetch_deadline(None) is not None
    assert MacroEconomicCalendarCollector._build_fetch_deadline(lambda *_a, **_kw: None) is None
    assert MacroTelemetryCollectorBase._persist_snapshot(str(cdir), cachepaths.CACHE_MACRO_CALENDAR_FILENAME, {"ts": t_now, "events_ok": True}) is True
    loaded_cal = mss.load_cached_macro_snapshot(str(cdir), "macro_calendar_files", cachepaths.CACHE_MACRO_CALENDAR_FILENAME)
    assert isinstance(loaded_cal, dict) and loaded_cal["events_ok"] is True
    assert MacroTelemetryCollectorBase._is_snapshot_within_age({"ts": t_now - 100.0}, now_ts=t_now, max_age_sec=300.0) is True
    assert MacroTelemetryCollectorBase._is_snapshot_within_age({"ts": t_now - 300.0}, now_ts=t_now, max_age_sec=300.0, inclusive=False) is False
    assert MacroTelemetryCollectorBase._is_snapshot_within_age({"ts": t_now - 300.0}, now_ts=t_now, max_age_sec=300.0, inclusive=True) is True

    called = {"n": 0}

    def _fake_live():
        called["n"] += 1
        return {"ts": t_now, "live": True}

    kept = MacroAndNewsSensorsCollector._maybe_backfill_cached_macro_snapshot(
        {"ts": t_now - 60.0}, "", now_ts=t_now, max_age_sec=300.0, collect_fn=_fake_live
    )
    assert kept == {"ts": t_now - 60.0} and called["n"] == 0
    refreshed = MacroAndNewsSensorsCollector._maybe_backfill_cached_macro_snapshot(
        {"ts": t_now - 600.0}, "", now_ts=t_now, max_age_sec=300.0, collect_fn=_fake_live
    )
    assert refreshed == {"ts": t_now, "live": True} and called["n"] == 1

    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(cdir), skip_external=True, skip_news=True)
    assert gtc._run_macro_snapshot_update(
        label="test_macro",
        now=t_now,
        collect_fn=lambda eff: {"ts": eff, "ok": True},
        is_valid_fn=lambda s: bool(s.get("ok")),
    ) is True
    assert gtc._run_macro_snapshot_update(
        label="test_macro_err",
        now=t_now,
        collect_fn=lambda _eff: (_ for _ in ()).throw(RuntimeError("boom")),
        is_valid_fn=lambda s: bool(s),
    ) is False
    assert _resolve_macro_sensor(SimpleNamespace(macro_benchmarks=None), {"macro_benchmarks": {"x": 1}}, "macro_benchmarks") == {"x": 1}
    assert _resolve_macro_sensor(SimpleNamespace(macro_benchmarks=None), {}, "macro_benchmarks", ps={"macro_benchmarks": {"x": 2}}) == {"x": 2}


def test_iter3_premarket_lead_rollover_kinematics_impulse_capacity_and_limit_dust_snap():
    """Verify live PRE delayed-futures suppression, U-bounce rollover kinematic guard, unactionable impulse filtering, and LIMIT float-dust snapping."""
    import pricewindow as pw
    from intelligence.globaltelemetry_collector import _filter_actionable_impulse_symbols
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        compute_futures_premarket_lead,
        format_macro_benchmarks_summary_line,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_series_kinematics
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import (
        summarize_active_ai_trade_plans,
        summarize_recent_ai_intents,
    )

    t_now = 1_795_183_036.0

    # 1. Unactionable ($0 buy / $0 sell) symbols such as flat SELL-only xStocks must not trigger impulse wakeups
    vb_state = {
        "pos:kraken:ANTxUSD": {"symbol": "ANTxUSD", "effective_max_buy_usd": 0.0, "effective_max_sell_usd": 0.0},
        "pos:kraken:SPCXxUSD": {"symbol": "SPCXxUSD", "effective_max_buy_usd": 0.0, "effective_max_sell_usd": 615.0},
        "pos:binance:BTCUSDC": {"symbol": "BTCUSDC", "effective_max_buy_usd": 2000.0, "effective_max_sell_usd": 2000.0},
    }
    assert _filter_actionable_impulse_symbols(["ANTxUSD"], vb_state) == []
    assert _filter_actionable_impulse_symbols(["ANTxUSD", "SPCXxUSD", "BTCUSDC"], vb_state) == ["SPCXxUSD", "BTCUSDC"]
    assert _filter_actionable_impulse_symbols(["ANTxUSD"], {}) == ["ANTxUSD"]

    # 2. Delayed Yahoo CME futures (14m behind live PRE QQQ/SPY) must NOT emit inverted Futures_Lead
    delayed_premarket_mb = {
        "items": {
            "SPY": {"price": 709.43, "chg_24h_pct": -3.86, "chg_15m_pct": 0.13, "market_state": "PRE", "last_bar_ts": t_now},
            "QQQ": {"price": 634.25, "chg_24h_pct": -5.59, "chg_15m_pct": 0.28, "market_state": "PRE", "last_bar_ts": t_now},
            "ES=F": {"price": 7109.75, "chg_24h_pct": -4.08, "last_bar_ts": t_now - 840.0},
            "NQ=F": {"price": 26022.50, "chg_24h_pct": -5.91, "last_bar_ts": t_now - 840.0},
        },
        "futures_lead_pct": -0.32,
        "futures_lead_regime": "FUTURES_PREMARKET_GAP_DOWN",
    }
    assert compute_futures_premarket_lead(delayed_premarket_mb) == {}
    assert "Futures_Lead=" not in format_macro_benchmarks_summary_line(delayed_premarket_mb)

    # Overnight CLOSED ETFs still emit Globex Futures_Lead
    overnight_closed_mb = {
        "items": {
            "SPY": {"price": 709.43, "chg_24h_pct": -3.86, "market_state": "CLOSED", "last_bar_ts": t_now - 7200.0},
            "QQQ": {"price": 634.25, "chg_24h_pct": -5.59, "market_state": "CLOSED", "last_bar_ts": t_now - 7200.0},
            "ES=F": {"price": 7109.75, "chg_24h_pct": -4.30, "globex_session_chg_pct": -0.45, "last_bar_ts": t_now - 600.0},
            "NQ=F": {"price": 26022.50, "chg_24h_pct": -6.44, "globex_session_chg_pct": -0.85, "last_bar_ts": t_now - 600.0},
        },
    }
    assert compute_futures_premarket_lead(overnight_closed_mb)["futures_lead_pct"] == -0.85

    # 3. Failed mid-window bounce rollover (drop -> mid bounce -> final-minute rollover) must NOT emit positive vel_5m
    # 301 samples at 1s spacing (5.0m): drop from 61880 to 61750 (-5m..-2.2m), bounce to 61815 (-2.2m..-0.6m), roll over to 61750 (-0.6m..0m)
    prices_down_roll = []
    for sec in range(301):
        if sec <= 168:
            px = 61880.0 - (130.0 * (sec / 168.0))
        elif sec <= 264:
            px = 61750.0 + (65.0 * ((sec - 168) / 96.0))
        else:
            px = 61815.0 - (65.0 * ((sec - 264) / 36.0))
        prices_down_roll.append(px)
    series_down = [[(t_now - 300.0 + idx) * 1000.0, px] for idx, px in enumerate(prices_down_roll)]
    kin_down = compute_series_kinematics(series_down)
    assert kin_down["vel_5m_pct_min"] < 0.0
    assert kin_down["accel_5m_pct_min2"] < 0.0
    win_down = pw.PriceWindow("BTCUSDC", 301, sample_rate_sec=1.0, initial_prices=prices_down_roll, window_seconds=300.0)
    v_pw_down, a_pw_down = win_down.calculate_acceleration(window_sec=300.0)
    assert v_pw_down < 0.0 and a_pw_down < 0.0

    # Symmetric inverted-U rollover (rally -> mid dip -> final-minute reclaim) must emit positive vel_5m & accel_5m
    prices_up_roll = [2.0 * 61880.0 - px for px in prices_down_roll]
    series_up = [[(t_now - 300.0 + idx) * 1000.0, px] for idx, px in enumerate(prices_up_roll)]
    kin_up = compute_series_kinematics(series_up)
    assert kin_up["vel_5m_pct_min"] > 0.0
    assert kin_up["accel_5m_pct_min2"] > 0.0
    win_up = pw.PriceWindow("BTCUSDC", 301, sample_rate_sec=1.0, initial_prices=prices_up_roll, window_seconds=300.0)
    v_pw_up, a_pw_up = win_up.calculate_acceleration(window_sec=300.0)
    assert v_pw_up > 0.0 and a_pw_up > 0.0

    # 4. Historical FILLED LIMIT float dust (cost/vol = 130.070001 vs limit 130.07) snaps cleanly while real fill improvement is preserved
    intents = [
        {
            "venue": "kraken",
            "symbol": "SPCXxUSD",
            "decision": "SELL",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "limit_price": 130.07,
            "target_px": 128.89,
            "invalidation_px": 131.2,
            "suggested_notional_usd": 646.31,
            "ts": t_now - 3600.0,
            "execution_payload": {
                "order_id": "OLZEGS-C2J2M-OTMPY2",
                "status": "FILLED",
                "price": 130.0700008,
                "executedQty": "4.968943",
            },
        },
        {
            "venue": "binance",
            "symbol": "TAOUSDC",
            "decision": "SELL",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "limit_price": 184.45,
            "target_px": 182.11,
            "invalidation_px": 187.5,
            "suggested_notional_usd": 722.04,
            "ts": t_now - 3000.0,
            "execution_payload": {
                "order_id": "727133845",
                "status": "FILLED",
                "price": 184.4,
                "executedQty": "3.9156",
            },
        },
    ]
    rec_sum = summarize_recent_ai_intents(intents, "crypto", t_now, extract_intent_notional)
    by_sym_rec = {row["symbol"]: row for row in rec_sum}
    assert by_sym_rec["SPCXxUSD"]["fill_px"] == 130.07
    assert by_sym_rec["TAOUSDC"]["fill_px"] == 184.4
    plans = summarize_active_ai_trade_plans(
        intents,
        "crypto",
        t_now,
        extract_intent_notional,
        prices_and_trends={"SPCXxUSD": {"price": 130.1}, "TAOUSDC": {"price": 185.8}},
    )
    by_sym_plan = {row["symbol"]: row for row in plans}
    assert by_sym_plan["SPCXxUSD"]["entry_px"] == 130.07
    assert by_sym_plan["TAOUSDC"]["entry_px"] == 184.4

    # 5. Locked-quote resting order capacity remains actionable; post-eval <0.35% price chop suppresses duplicate impulse wakeups
    from intelligence.globaltelemetry_collector import _filter_unmoved_since_eval_symbols

    vb_locked_order = {
        "pos:kraken:ADAUSD": {
            "venue": "kraken",
            "symbol": "ADAUSD",
            "effective_max_buy_usd": 0.0,
            "effective_max_sell_usd": 0.0,
            "replace_max_buy_usd": 250.0,
        },
    }
    assert _filter_actionable_impulse_symbols(
        ["ADAUSD"],
        vb_locked_order,
        open_orders=[{"venue": "kraken", "symbol": "ADAUSD", "order_id": "OYXYYN-2Z243-T3P6XZ"}],
    ) == ["ADAUSD"]

    hold_eval_doc = {
        "ts": t_now - 120.0,
        "model_used": "gemini-4-neon-high",
        "market_trading_decisions": [{"decision": "HOLD", "symbol": "ADAUSD"}],
        "delta_gate_meta": {"eval_prices": {"ADAUSD": 0.1464, "HYPEUSDC": 63.10, "BTCUSDC": 61800.0}},
    }
    pt_live = {
        "ADAUSD": {"price": 0.14646},     # +0.04% (< 0.35%) -> suppressed
        "HYPEUSDC": {"price": 63.00},     # -0.16% (< 0.35%) -> suppressed
        "BTCUSDC": {"price": 61520.0},    # -0.45% (>= 0.35%) -> wakes up!
    }
    assert (
        _filter_unmoved_since_eval_symbols(
            ["ADAUSD", "HYPEUSDC", "BTCUSDC"],
            pt_live,
            hold_eval_doc,
            eff_now=t_now,
            max_idle_sec=1080.0,
            min_move_pct=0.35,
        )
        == ["BTCUSDC"]
    )
    actionable_eval_doc = {**hold_eval_doc, "market_trading_decisions": [{"decision": "BUY", "symbol": "ADAUSD"}]}
    assert _filter_unmoved_since_eval_symbols(
        ["ADAUSD"],
        pt_live,
        actionable_eval_doc,
        eff_now=t_now,
        max_idle_sec=1080.0,
        min_move_pct=0.35,
    ) == ["ADAUSD"]


def test_faza3b_spot_etf_flows_collector_parser_pretrade_and_prompt(tmp_path, monkeypatch):
    """End-to-end verification of Faza 3B: Farside BTC/ETH HTML parser, SpotEtfFlowsCollector, daemon update, Section 0/1 prompt, Pre-Trade pillar, and SentimentAdvisor."""
    import cachepaths
    import intelligence.macro_sentiment_ssot as mss
    from intelligence.external.collectors.base_collector import TelemetryCollectorFactory
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        MarketAndPortfolioSnapshotOrchestrator,
        SpotEtfFlowsCollector,
        collect_spot_etf_flows_snapshot,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
        MacroPillarsStage,
        read_pretrade_telemetry_snapshot,
        resolve_pretrade_source_intervals_sec,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        format_etf_flows_summary_line,
        format_pretrade_etf_flows_line,
        parse_farside_etf_flows_html,
    )
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

    btc_html = """
    <html><body><table>
      <tr><th></th><th></th><th></th><th></th><th></th><th>Total</th></tr>
      <tr><th></th><th>IBIT</th><th>FBTC</th><th>ARKB</th><th>GBTC</th><th></th></tr>
      <tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>0.21%</td><td>1.50%</td><td></td></tr>
      <tr><td>16 Nov 2026</td><td>120.5</td><td>30.0</td><td>0.0</td><td>(10.5)</td><td>140.0</td></tr>
      <tr><td>17 Nov 2026</td><td>(110.0)</td><td>(20.0)</td><td>0.0</td><td>(5.0)</td><td>(135.0)</td></tr>
      <tr><td>18 Nov 2026</td><td>(200.0)</td><td>(50.0)</td><td>(10.0)</td><td>(20.0)</td><td>(280.0)</td></tr>
      <tr><td>19 Nov 2026</td><td>(541.1)</td><td>(45.9)</td><td>(31.3)</td><td>(13.9)</td><td>(632.2)</td></tr>
      <tr><td>20 Nov 2026</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.0</td></tr>
      <tr><td>Total</td><td>63,370</td><td>10,037</td><td>854</td><td>(28,235)</td><td>53,580</td></tr>
      <tr><td>Average</td><td>88.3</td><td>14.0</td><td>1.2</td><td>(39.3)</td><td>74.6</td></tr>
    </table></body></html>
    """
    eth_html = """
    <html><body><table>
      <tr><th></th><th></th><th></th><th></th><th>Total</th></tr>
      <tr><th></th><th>ETHA</th><th>FETH</th><th>ETHE</th><th></th></tr>
      <tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>2.50%</td><td></td></tr>
      <tr><td>Staking fee</td><td>N/A</td><td>N/A</td><td>23%</td><td></td></tr>
      <tr><td>18 Nov 2026</td><td>(40.0)</td><td>(5.0)</td><td>(15.0)</td><td>(60.0)</td></tr>
      <tr><td>19 Nov 2026</td><td>(124.2)</td><td>-</td><td>(5.2)</td><td>(129.4)</td></tr>
      <tr><td>20 Nov 2026</td><td>-</td><td>-</td><td>-</td><td>0.0</td></tr>
      <tr><td>Total</td><td>11,043</td><td>2,436</td><td>(5,542)</td><td>11,344</td></tr>
    </table></body></html>
    """

    # 1. Pure Farside HTML parser skips unreported placeholder day (20 Nov), detects partial reporting, streaks, and regimes
    btc_parsed = parse_farside_etf_flows_html(btc_html, asset="BTC")
    assert btc_parsed is not None
    assert btc_parsed["latest_date"] == "2026-11-19"
    assert btc_parsed["pending_date"] == "2026-11-20"
    assert btc_parsed["latest_net_flow_musd"] == pytest.approx(-632.2)
    assert btc_parsed["latest_net_flow_usd"] == pytest.approx(-632_200_000.0)
    assert btc_parsed["is_partial"] is False
    assert btc_parsed["reported_issuers"] == 4 and btc_parsed["total_issuers"] == 4
    assert btc_parsed["streak_days"] == -3
    assert btc_parsed["sum_3d_musd"] == pytest.approx(-1047.2)
    assert btc_parsed["sum_5d_musd"] == pytest.approx(-907.2)
    assert btc_parsed["flow_regime"] == "STRONG_OUTFLOW"
    assert list(btc_parsed["top_issuers_musd"].keys()) == ["IBIT", "FBTC", "ARKB"]

    eth_parsed = parse_farside_etf_flows_html(eth_html, asset="ETH")
    assert eth_parsed is not None
    assert eth_parsed["latest_date"] == "2026-11-19"
    assert eth_parsed["is_partial"] is True
    assert eth_parsed["reported_issuers"] == 2 and eth_parsed["total_issuers"] == 3
    assert eth_parsed["streak_days"] == -2
    assert eth_parsed["flow_regime"] == "STRONG_OUTFLOW"

    # 2. SpotEtfFlowsCollector fetches, persists atomically, honors skip_if_fresh_sec, preserves partial fallback, and supports read_snapshot
    t_now = 1_795_200_000.0
    cdir = tmp_path / "cachedb"
    cdir.mkdir()
    fetch_calls = []

    def _fake_fetch_text(url, **_kw):
        fetch_calls.append(url)
        if "/btc/" in url:
            return btc_html
        if "/eth/" in url:
            return eth_html
        raise RuntimeError(f"Unexpected URL: {url}")

    snap = collect_spot_etf_flows_snapshot(
        str(cdir),
        now_ts=t_now,
        persist=True,
        fetch_text_fn=_fake_fetch_text,
    )
    assert snap is not None and snap["btc_ok"] is True and snap["eth_ok"] is True
    assert len(fetch_calls) == 2
    assert "BTC_ETF(2026-11-19): -$632.20M [IBIT -$541.10M, FBTC -$45.90M, ARKB -$31.30M]" in snap["summary_line"]
    assert "5d=-$907.20M, streak=-3d)" in snap["summary_line"]
    assert "STRONG_OUTFLOW" not in snap["summary_line"]
    assert "ETH_ETF(2026-11-19): -$129.40M [ETHA -$124.20M, ETHE -$5.20M]" in snap["summary_line"]
    assert "partial=2/3" in snap["summary_line"]

    # Fresh guard avoids redundant network calls
    snap_cached = collect_spot_etf_flows_snapshot(
        str(cdir),
        now_ts=t_now + 60.0,
        persist=True,
        fetch_text_fn=_fake_fetch_text,
        skip_if_fresh_sec=1800.0,
    )
    assert snap_cached is not None and len(fetch_calls) == 2

    # Partial failure (ETH fails, BTC succeeds) retains recent cached ETH asset
    def _btc_only_fetch(url, **_kw):
        if "/btc/" in url:
            return btc_html
        raise RuntimeError("eth timeout")

    snap_partial = collect_spot_etf_flows_snapshot(
        str(cdir),
        now_ts=t_now + 3600.0,
        persist=True,
        fetch_text_fn=_btc_only_fetch,
    )
    assert snap_partial is not None and snap_partial["btc_ok"] is True and snap_partial["eth_ok"] is False
    assert "BTC" in snap_partial["assets"] and "ETH" in snap_partial["assets"]

    # Consumer read_snapshot does zero network I/O
    ro_col = TelemetryCollectorFactory.create("spot_etf_flows", start_sync=False)
    assert isinstance(ro_col, SpotEtfFlowsCollector)
    ro_snap = ro_col.read_snapshot(str(cdir))
    assert isinstance(ro_snap, dict) and ro_snap["assets"]["BTC"]["latest_date"] == "2026-11-19"
    assert mss.load_cached_etf_flows_snapshot(str(cdir)) is not None

    # 3. GlobalTelemetryCollector update_etf_flows + heartbeat interval wiring
    monkeypatch.setattr(mss, "fetch_public_text", _fake_fetch_text, raising=False)
    gtc = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(cdir), skip_external=True, skip_news=True)
    assert gtc.etf_flows_interval_sec == pytest.approx(mss.resolve_derived_telemetry_intervals()["etf_flows_interval_sec"])
    assert gtc.update_etf_flows(force=True, now=t_now + 7200.0) is True
    assert "etf_flows" in resolve_pretrade_source_intervals_sec("CRYPTO")

    # 4. Snapshot orchestrator + Section 0/1 prompt: visible on crypto scope, stripped on equities scope
    orch = MarketAndPortfolioSnapshotOrchestrator(start_sync=False)
    crypto_snap = orch.read_snapshot(str(cdir), scope="crypto", now_ts=t_now + 7260.0)
    assert isinstance(crypto_snap.etf_flows, dict) and "etf_flows" in crypto_snap.source_ages_sec
    crypto_prompt = build_supervisor_prompt(crypto_snap, SimpleNamespace(), topic_key="auto_cryptotrade_executor")
    assert "- US Spot Crypto ETF Net Flows (Farside Daily):" in crypto_prompt
    assert "BTC_ETF(2026-11-19): -$632.20M" in crypto_prompt
    assert "STRONG_OUTFLOW" not in crypto_prompt
    assert "etf_flows=" in crypto_prompt

    eq_snap = orch.read_snapshot(str(cdir), scope="equities", now_ts=t_now + 7260.0)
    assert eq_snap.etf_flows is None
    eq_prompt = build_supervisor_prompt(eq_snap, SimpleNamespace(), topic_key="auto_sharetrade_executor")
    assert "US Spot Crypto ETF Net Flows" not in eq_prompt
    assert "etf_flows=" not in eq_prompt

    # 5. Pre-Trade MacroPillarsStage conditional injection (crypto leads with ETH for ETHUSDC, BTC for BTCUSDC; omitted on NVDA_US_EQ)
    assert format_pretrade_etf_flows_line(snap, symbol="NVDA_US_EQ") == ""
    assert format_pretrade_etf_flows_line(snap, symbol="ETHUSDC").startswith("ETH_ETF(")
    assert format_pretrade_etf_flows_line(snap, symbol="BTCUSDC").startswith("BTC_ETF(")
    for sym_pt, expect_present in (("ETHUSDC", True), ("NVDA_US_EQ", False)):
        pt_snap = read_pretrade_telemetry_snapshot(
            sym_pt,
            cachedb_dir=str(cdir),
            stages=(MacroPillarsStage(),),
            now_ts=t_now + 7260.0,
        )
        if expect_present:
            assert "US Spot ETF Net Flows" in pt_snap.data
            assert pt_snap.data["US Spot ETF Net Flows"].startswith("ETH_ETF(")
            assert "etf_flows" in pt_snap.pillar_timestamps
        else:
            assert "US Spot ETF Net Flows" not in pt_snap.data

    # 6. SentimentAdvisor includes cached Spot ETF Net Flows in its LLM prompt context
    captured_prompts = []

    class _FakeGemini:
        def query_json(self, prompt_txt, **_kw):
            captured_prompts.append(prompt_txt)
            return {
                "market_bias": "CAUTION",
                "risk_level": "HIGH",
                "confidence": 0.8,
                "summary": "Heavy institutional ETF outflows.",
                "key_risks": ["BTC ETF outflows"],
                "recommended_action": "HOLD",
            }

    adv = SentimentAdvisor(
        gemini_client=_FakeGemini(),
        cache_file=str(cdir / cachepaths.CACHE_SENTIMENT_EVAL_FILENAME),
    )
    adv.review(force_refresh=True, now_ts=t_now + 7260.0, allow_llm=True)
    assert captured_prompts and "- US Spot Crypto ETF Net Flows (Farside Daily):" in captured_prompts[0]
    assert format_etf_flows_summary_line(snap) in captured_prompts[0]


def test_iter4_slope_unit_scaling_resting_plan_target_hit_and_post_sell_lifo_reset():
    """Verify price-unit slope scaling on s90_flipped, no TARGET_HIT on unfilled RESTING plans, and post-flat Binance LIFO lot/PnL reset."""
    from intelligence.sentiment.evaluators.portfolio_evaluator import PositionCostBasisEnrichmentEvaluator
    from intelligence.sentiment.evaluators.price_series_evaluator import apply_24h_series_enrichment
    from orchestratorOS.supervisor.crypto_executor_delta_gate import _evaluate_delta_gate_trade_plans
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import (
        evaluate_trade_plan_status,
        summarize_active_ai_trade_plans,
    )

    t_now = 1_795_185_000.0

    # 1. Price-unit scaling on s90_flipped prevents 500x unit mismatch & sign contradiction on low-price coins (ARBUSDC)
    # Build 5m series with a noisy outlier spike at -90s (0.1186) followed by steady rise 0.1182 -> 0.1185
    series_arb = []
    for sec in range(301):
        ts_ms = (t_now - 300.0 + sec) * 1000.0
        if sec == 210:
            px = 0.11865  # 1-tick outlier at -90s
        elif sec < 210:
            px = 0.1182
        else:
            px = 0.1182 + 0.0003 * ((sec - 210) / 90.0)
        series_arb.append([ts_ms, px])
    entry_arb = {
        "price": 0.1185,
        "slope_90s": -2.0e-6,
        "slope_3_7m": 1.8e-7,
        "slope_15m": 7.7e-7,
        "slope_2_5h": 4.1e-7,
        "gradient_recent": -3.1e-6,
        "gradient_recent_fast": -2.0e-6,
        "epsilon": 9.6e-5,
        "vel_90s_pct_min": -0.25,  # stale opposite-sign vel triggers kin_contradicts_short -> recompute + s90_flipped
        "accel_90s_pct_min2": -0.10,
        "short_trend_ts": t_now,
    }
    apply_24h_series_enrichment("ARBUSDC", entry_arb, {"chg_1m_pct": 0.08}, series_arb, None)
    assert entry_arb["vel_90s_pct_min"] > 0.0
    assert entry_arb["slope_90s"] > 0.0
    assert entry_arb["slope_90s"] < 5.0e-5  # price-unit ($/sec), NOT ~0.002 ((dp/p)/min)
    assert entry_arb["gradient_recent_fast"] > 0.0
    assert entry_arb["gradient_recent_fast"] < 5.0e-5

    # 2. Unfilled RESTING orders must NEVER report TARGET_HIT or trigger delta-gate target_hit wakeups (INVALIDATED still works)
    assert evaluate_trade_plan_status("BUY", 0.1472, 0.139, 0.147, is_resting=True) == "ACTIVE"
    assert evaluate_trade_plan_status("BUY", 0.1385, 0.139, 0.147, is_resting=True) == "INVALIDATED"
    assert evaluate_trade_plan_status("BUY", 0.1472, 0.139, 0.147, is_resting=False) == "TARGET_HIT"
    assert evaluate_trade_plan_status("SELL", 180.0, 192.0, 184.0, is_resting=True) == "ACTIVE"
    assert evaluate_trade_plan_status("SELL", 193.0, 192.0, 184.0, is_resting=True) == "INVALIDATED"

    resting_ada_intent = [
        {
            "intent_id": "ada_buy_rest",
            "timestamp": t_now - 3600.0,
            "decision": "BUY",
            "symbol": "ADAUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 0.1418,
            "target_px": 0.147,
            "invalidation_px": 0.139,
            "suggested_notional_usd": 250.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "OYXYYN-2Z243-T3P6XZ", "status": "NEW"},
        }
    ]
    open_ada = [
        {"venue": "kraken", "symbol": "ADAUSD", "side": "BUY", "order_id": "OYXYYN-2Z243-T3P6XZ", "price": 0.1418, "qty": 1763.05, "status": "NEW"}
    ]
    ada_plans = summarize_active_ai_trade_plans(
        resting_ada_intent, "crypto", t_now, extract_intent_notional, open_orders=open_ada, prices_and_trends={"ADAUSD": {"price": 0.147126}}
    )
    assert len(ada_plans) == 1 and ada_plans[0]["order_state"] == "RESTING" and ada_plans[0]["status"] == "ACTIVE"
    dg_states, dg_trig = _evaluate_delta_gate_trade_plans(
        {"active_ai_trade_plans": ada_plans}, {"ADAUSD": 0.147126}, {"kraken:ADAUSD:BUY": "ACTIVE"}
    )
    assert dg_states["kraken:ADAUSD:BUY"] == "ACTIVE" and dg_trig == ""

    # 3. Fresh post-sell buy matching >=98% of live wallet prunes pre-sell phantom fee-drift lots & resets closed-cycle LIFO PnL
    ls_ts_ms = int((t_now - 24000.0) * 1000)
    new_buy_ts_ms = int((t_now - 300.0) * 1000)
    ro_arb = {
        "last_action_side": "BUY",
        "last_buy": {"price": 0.1184, "qty": 3489.5, "ts_ms": new_buy_ts_ms},
        "last_sell": {"price": 0.1201, "qty": 3584.5, "ts_ms": ls_ts_ms},
        "active_avg_buy_px": 0.118715,
        "lifo_realized_pnl_usd": -310.86,
        "open_lots": [
            {"lot_id": "L27", "qty": 84.1, "entry_px": 0.1595, "ts_ms": ls_ts_ms - 15 * 86400 * 1000},
            {"lot_id": "L51", "qty": 27.9, "entry_px": 0.1357, "ts_ms": ls_ts_ms - 4 * 86400 * 1000},
            {"lot_id": "L59", "qty": 3489.5, "entry_px": 0.1184, "ts_ms": new_buy_ts_ms},
        ],
    }
    pos_arb = {"venue": "binance", "symbol": "ARBUSDC", "price": 0.1186, "free_base": 3493.14511, "value_usd": 414.29}
    PositionCostBasisEnrichmentEvaluator._enrich_position_from_recent_orders(
        pos_arb,
        ro_arb,
        p_sym="ARBUSDC",
        p_quote_variants=("ARBUSDC",),
        pos_px_f=0.1186,
        held_cur=3493.14511,
        is_flat_balance=False,
        is_flat_after_exit=False,
        prices_and_trends={},
        cache_dir="",
        cycle_high_memo={},
    )
    assert len(pos_arb["open_lots"]) == 1 and pos_arb["open_lots"][0]["entry_px"] == 0.1184
    assert pos_arb["active_avg_buy_px"] == pytest.approx(0.1184)
    assert "lifo_realized_pnl_usd" not in pos_arb
    assert ro_arb["lifo_realized_pnl_usd"] == 0.0


def test_phases_2_to_3b_code_review_edge_case_fixes(tmp_path, monkeypatch):
    """Regression coverage for hidden edge cases found in Phases 2/3A/3B review: inverted PRE quote metadata, PRE equity memo throttle, frozen xStock sibling session, MEDIUM-before-HIGH catalyst shadowing, and Faza 3B partial-failure/Total=0.0/tokenized-equity guards."""
    from datetime import datetime
    from botcore import ET
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.macro.calendar_ssot import format_pretrade_macro_catalyst_line
    from intelligence.sentiment.collectors.macro_telemetry_collector import collect_spot_etf_flows_snapshot
    from intelligence.sentiment.collectors.microstructure_collector import Fleet24hAndMicrostructureEnricher
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        format_pretrade_etf_flows_line,
        parse_farside_etf_flows_html,
    )
    from state_io import save_json_state

    ts_pre = datetime(2026, 11, 18, 8, 15, tzinfo=ET).timestamp()

    # 1. Active PRE cache_current_price must NOT be inverted into t212_close_px when pos:t212 has newer ts;
    #    and matching extended vs close price clears stale close/ext_vs_rth metadata
    pt_inv = {"NVDA_US_EQ": {"price": 141.05, "trend": 0, "growth_coefficient": 0.0}}
    cp_inv = {"NVDA_US_EQ": [[int((ts_pre - 35.0) * 1000), 141.05]]}
    bals_inv = {"pos:t212:NVDA_US_EQ": {"symbol": "NVDA_US_EQ", "free": 2.0, "price": 145.98, "ts": ts_pre - 5.0}}
    Fleet24hAndMicrostructureEnricher._seed_t212_positions(
        "", pt_inv, cp_inv, {}, bals_inv, "", 0.0, ts_pre, lambda _p: None, False
    )
    assert pt_inv["NVDA_US_EQ"].get("quote_source") != "yahoo_ext_cached"
    assert pt_inv["NVDA_US_EQ"].get("t212_close_px") != pytest.approx(141.05)

    pt_eq = {
        "NVDA_US_EQ": {
            "price": 145.98,
            "t212_close_px": 140.0,
            "ext_vs_rth_pct": 4.27,
            "trend": 0,
            "growth_coefficient": 0.0,
        }
    }
    cp_eq = {"NVDA_US_EQ": [[int((ts_pre - 10.0) * 1000), 145.98]]}
    bals_eq = {"pos:t212:NVDA_US_EQ": {"symbol": "NVDA_US_EQ", "free": 2.0, "price": 145.98, "ts": ts_pre - 36000.0}}
    Fleet24hAndMicrostructureEnricher._seed_t212_positions(
        "", pt_eq, cp_eq, {}, bals_eq, "", 0.0, ts_pre, lambda _p: None, False
    )
    assert "t212_close_px" not in pt_eq["NVDA_US_EQ"]
    assert "ext_vs_rth_pct" not in pt_eq["NVDA_US_EQ"]

    # 2. Equity context memo throttles to >=120s during PRE/POST (cutting Nasdaq/Yahoo polling 4x outside RTH)
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path / "ob_pre"), cache_ttl_sec=0.0, disk_ttl_sec=0.0, start_sync=True)
    ob_col._EQUITY_CTX_MEMO.clear()
    fetch_cnt = {"n": 0}

    def _fake_eq_fetch(_url, label, _sym, **_kw):
        fetch_cnt["n"] += 1
        if label == "nasdaq_quote":
            return {"data": {"primaryData": {"lastSalePrice": "$145.98"}}}
        return {
            "chart": {
                "result": [
                    {
                        "meta": {"marketState": "PRE", "regularMarketPrice": 145.98, "preMarketPrice": 144.0},
                        "timestamp": [int(ts_pre - 300)],
                        "indicators": {"quote": [{"open": [144.0], "high": [144.2], "low": [143.9], "close": [144.0], "volume": [100]}]},
                    }
                ]
            }
        }

    monkeypatch.setattr(ob_col, "_fetch_json_dict", _fake_eq_fetch)
    assert ob_col._fetch_equity_market_context("NVDA_US_EQ", ts_pre, include_options=False) is not None
    calls_first = fetch_cnt["n"]
    # 60s later during PRE must still hit memo (zero additional HTTP calls under >=120s extended TTL)
    assert ob_col._fetch_equity_market_context("NVDA_US_EQ", ts_pre + 60.0, include_options=False) is not None
    assert fetch_cnt["n"] == calls_first

    # 3. KrakenXstocksCollector marks frozen t212_rth_close sibling as RTH_CLOSE instead of current wall-clock session
    k_dir = tmp_path / "kraken_rth_close"
    k_dir.mkdir()
    save_json_state(str(k_dir / "xstock_state.json"), {"known_assets": {"SPCXx": 10.0}, "pair": "SPCXxUSD"})
    xst = KrakenXstocksCollector(start_sync=False).collect(
        str(k_dir),
        kraken_dir=str(k_dir),
        prices_and_trends={
            "SPCXxUSD": {"price": 152.0},
            "SPCX_US_EQ": {"price": 150.0, "quote_source": "t212_rth_close"},
        },
        all_t212_cycles={},
    )
    assert xst is not None and xst["underlying_session"] == "RTH_CLOSE"

    # 4. Earlier MEDIUM macro event does NOT shadow a HIGH-impact release within <=240m in Pre-Trade catalyst line
    cal_mixed = {
        "pre_event_risk_window": False,
        "next_confirmed_event": {
            "title": "Unemployment Claims",
            "impact": "MEDIUM",
            "minutes_until": 15.0,
            "summary": "Unemployment Claims [MEDIUM] at 15:30 EET",
        },
        "confirmed_events_today": [
            {
                "title": "Unemployment Claims",
                "impact": "MEDIUM",
                "minutes_until": 15.0,
                "summary": "Unemployment Claims [MEDIUM] at 15:30 EET",
            },
            {
                "title": "Core PCE Price Index m/m",
                "impact": "HIGH",
                "minutes_until": 105.0,
                "summary": "Core PCE Price Index m/m [HIGH] at 17:00 EET",
            },
        ],
    }
    cat_line = format_pretrade_macro_catalyst_line(cal_mixed, symbol="BTCUSDC")
    assert cat_line is not None and "Core PCE Price Index" in cat_line

    # 5a. Partial Farside row where Total lags at 0.0 uses sum(issuers_map), handles colspan, and ignores tokenized equities
    partial_lag_html = """
    <table>
      <tr><th colspan="4"></th><th>Total</th></tr>
      <tr><th></th><th>IBIT</th><th>FBTC</th><th>GBTC</th><th></th></tr>
      <tr><td>19 Nov 2026</td><td>180.5</td><td>-</td><td>(20.5)</td><td>0.0</td></tr>
      <tr><td>20 Nov 2026</td><td>-</td><td>-</td><td>-</td><td>0.0</td></tr>
    </table>
    """
    parsed_partial = parse_farside_etf_flows_html(partial_lag_html, asset="BTC")
    assert parsed_partial is not None
    assert parsed_partial["latest_net_flow_musd"] == pytest.approx(160.0)
    assert parsed_partial["is_partial"] is True
    assert parsed_partial["pending_date"] == "2026-11-20"
    assert format_pretrade_etf_flows_line({"assets": {"BTC": parsed_partial}}, symbol="SPCXxUSD") == ""
    assert format_pretrade_etf_flows_line({"assets": {"BTC": parsed_partial}}, symbol="ANTxUSD") == ""
    assert format_pretrade_etf_flows_line({"assets": {"BTC": parsed_partial}}, symbol="BTCUSDC") != ""

    # 5b. SpotEtfFlowsCollector retries degraded cache (eth_ok=False) even inside skip_if_fresh_sec
    cdir_etf = tmp_path / "cachedb"
    cdir_etf.mkdir()
    eth_ok_html = """
    <table>
      <tr><th></th><th>ETHA</th><th>Total</th></tr>
      <tr><td>19 Nov 2026</td><td>(50.0)</td><td>(50.0)</td></tr>
    </table>
    """
    calls_etf = {"n": 0}

    def _both_ok(url, **_kw):
        calls_etf["n"] += 1
        return partial_lag_html if "/btc/" in url else eth_ok_html

    # Seed degraded snapshot (btc_ok=True, eth_ok=False)
    save_json_state(
        str(cdir_etf / "etf_flows_collect.json"),
        {
            "ts": ts_pre,
            "btc_ok": True,
            "eth_ok": False,
            "assets": {"BTC": parsed_partial, "ETH": {"latest_date": "2026-11-18", "latest_net_flow_musd": -10.0}},
        },
    )
    res_retry = collect_spot_etf_flows_snapshot(
        str(cdir_etf),
        now_ts=ts_pre + 60.0,
        persist=True,
        fetch_text_fn=_both_ok,
        skip_if_fresh_sec=1800.0,
    )
    assert res_retry is not None and res_retry["eth_ok"] is True and calls_etf["n"] == 2


def test_post_us_open_vwap_rvol_equity_placeholder_and_bellwether_funding():
    """Verify post-US-open telemetry fixes: prev_rth anchor & Kish n_eff guard, active-session RVOL span & freshness, no synthetic whale_long_pct, and annualized bellwether funding."""
    from datetime import datetime, timezone
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_session_vwap_metrics,
        compute_volume_rvol_metrics,
    )
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_portfolio_prompt_view
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section

    # 1. Session VWAP tags yesterday's RTH as 'prev_rth' at RTH open (<2 today bars) or when now_ts is next day,
    #    and suppresses degenerate z-score when opening-bell bar dominates >80% of volume (Kish n_eff < 2.0).
    d1_rth0 = datetime(2026, 11, 19, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    d2_rth0 = datetime(2026, 11, 20, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    d1_bars = [[d1_rth0 + i * 300_000.0, 150.0 + (0.2 if i % 2 == 0 else -0.2), 1_000_000.0] for i in range(10)]
    open_1bar = d1_bars + [[d2_rth0, 141.0, 40_000_000.0], [d2_rth0 + 180_000.0, 141.5, 0.0]]
    v_open = compute_session_vwap_metrics(open_1bar, is_equity=True, cur_px=141.5)
    assert v_open["session_vwap_anchor"] == "prev_rth"
    assert v_open["session_vwap_px"] == pytest.approx(150.0, abs=0.05)

    # Yesterday-only bars evaluated at today's RTH open via now_ts -> 'prev_rth' (e.g. RGNT_US_EQ with 0 bars today)
    v_rgnt = compute_session_vwap_metrics(d1_bars, is_equity=True, cur_px=0.89, now_ts=(d2_rth0 + 600_000.0) / 1000.0)
    assert v_rgnt["session_vwap_anchor"] == "prev_rth"

    # Opening-bell volume skew (bar 1 = 85% of session volume -> n_eff ~ 1.37 < 2.0) suppresses bogus z-score
    skewed_open = [
        [d2_rth0, 130.63, 17_200_000.0],
        [d2_rth0 + 300_000.0, 130.76, 3_100_000.0],
        [d2_rth0 + 600_000.0, 130.66, 620_000.0],
    ]
    v_skew = compute_session_vwap_metrics(skewed_open, is_equity=True, cur_px=131.0)
    assert v_skew["session_vwap_anchor"] == "rth"
    assert v_skew["session_vwap_px"] == pytest.approx(130.65, abs=0.05)
    assert "session_vwap_z" not in v_skew and "session_vwap_sigma_pct" not in v_skew

    # 2. Active-session RVOL excludes 17.5h overnight market closure (no 3.7x US equity inflation)
    #    and returns {} when latest volume bar is >1h older than now_ts.
    eq_rvol_bars = (
        [[d1_rth0 + i * 300_000.0, 100.0, 100.0] for i in range(78)]  # 6.5h Day 1 RTH at $10k/5m
        + [[d2_rth0 + i * 300_000.0, 100.0, 200.0] for i in range(3)]  # 15m Day 2 open at $20k/5m (2x baseline)
    )
    rvol_eq = compute_volume_rvol_metrics(eq_rvol_bars, now_ts=(d2_rth0 + 700_000.0) / 1000.0)
    assert rvol_eq["spot_vol_15m_usd"] == pytest.approx(60_000.0)
    # Active span is ~6.75h (NOT 24h), so 2x opening surge reads ~1.9x instead of ~7x
    assert 1.7 <= rvol_eq["rvol_15m"] <= 2.2
    # Stale yesterday bars (17.5h old vs today's now_ts) must NOT leak yesterday closing volume as live spot_vol_15m
    assert compute_volume_rvol_metrics(d1_bars, now_ts=(d2_rth0 + 600_000.0) / 1000.0) == {}

    # 3. Equity Section 3B with no fresh volume bars and no options OI omits synthetic whale_long_pct=0.5
    sec3b_eq = "\n".join(
        _format_microstructure_section(
            {
                "RGNT_US_EQ": {
                    "ob_imbalance": 0.315,
                    "bid_depth_usd": 3.0,
                    "ask_depth_usd": 7.0,
                    "whale_long_pct": 0.5,
                    "taker_buy_vol_usd": 0.0,
                    "taker_sell_vol_usd": 0.0,
                    "oi_usd": 0.0,
                }
            },
            {"RGNT_US_EQ": {"price": 0.89}},
            now_ts=(d2_rth0 + 600_000.0) / 1000.0,
        )
    )
    assert "intraday_rth_flow=N/A[no_fresh_vol_bars]" in sec3b_eq
    assert "whale_long_pct=0.5" not in sec3b_eq

    # 4. Crypto bellwethers carry funding_annualized_pct and prompt view drops raw sci-notation funding_rate
    cmc = build_cross_market_context(
        {"BTCUSDC": {"price": 62000.0, "chg_24h_pct": -3.0}},
        {"BTCUSDC": {"ob_imbalance": 0.55, "funding_rate": 6.666e-5, "funding_annualized_pct": 7.3}},
        {},
    )
    assert cmc["crypto_bellwethers"]["BTC"]["funding_annualized_pct"] == pytest.approx(7.3)
    pv_eq = evaluate_portfolio_prompt_view({"cross_market_context": cmc}, asset_scope="equities")
    btc_bw = pv_eq["crypto_bellwethers"]["BTC"]
    assert btc_bw["funding_annualized_pct"] == pytest.approx(7.3)
    assert "funding_rate" not in btc_bw


def test_iter5_n_wave_5m_kinematics_guard_and_retry_queue_price_age(tmp_path):
    """Verify N-wave 5m breakout/breakdown kinematics guard and retry-queue requested_price rounding + age_min."""
    from intelligence.sentiment.collectors.portfolio_collector import BotRuntimeStatesCollector
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_series_kinematics

    t_now = 1_795_186_821.0

    # 1. Bullish N-wave (rally -> mid-dip -> final-minute breakout to fresh 5m high) must NOT emit negative vel_5m
    # Mirrors live TAOUSDC 17:00:21 profile: 0.00% -> +0.22% (-4m) -> -0.05% (-0.5m) -> +0.32% (0m)
    prices_n_up = []
    for sec in range(301):
        if sec <= 60:
            pct = 0.22 * (sec / 60.0)
        elif sec <= 265:
            pct = 0.22 - 0.27 * ((sec - 60) / 205.0)
        else:
            pct = -0.05 + 0.37 * ((sec - 265) / 35.0)
        prices_n_up.append(185.65 * (1.0 + pct / 100.0))
    series_n_up = [[(t_now - 300.0 + idx) * 1000.0, px] for idx, px in enumerate(prices_n_up)]
    kin_up = compute_series_kinematics(series_n_up)
    assert kin_up["vel_5m_pct_min"] > 0.0
    assert kin_up["accel_5m_pct_min2"] > 0.0

    # Symmetric bearish N-wave (drop -> mid-bounce -> final-minute breakdown to fresh 5m low) must emit negative vel_5m
    prices_n_dn = [2.0 * 185.65 - px for px in prices_n_up]
    series_n_dn = [[(t_now - 300.0 + idx) * 1000.0, px] for idx, px in enumerate(prices_n_dn)]
    kin_dn = compute_series_kinematics(series_n_dn)
    assert kin_dn["vel_5m_pct_min"] < 0.0
    assert kin_dn["accel_5m_pct_min2"] < 0.0

    # 2. BotRuntimeStatesCollector rounds IEEE-754 float dust on requested_price and reports age_min
    cdir = tmp_path / "cachedb"
    orders_dir = cdir / "orders"
    orders_dir.mkdir(parents=True)
    rec = {
        "id": "r1",
        "provider_name": "Binance",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 0.00878,
        "requested_price": 61561.98597200001,
        "kind": "accel_surge_up",
        "attempts": 2,
        "last_failure_reason": "profit_guard",
        "created_ts": t_now - 900.0,
        "lifecycle": "submit_pending",
        "expires_at_ts": t_now + 3600.0,
    }
    (orders_dir / "order_retry_queue.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    rt_out = BotRuntimeStatesCollector(start_sync=False).collect(str(cdir), now_ts=t_now)
    rq = rt_out["active_retry_queue"][0]
    assert rq["requested_price"] == 61561.985972
    assert rq["age_min"] == 15.0


def test_iter6_deribit_carryover_active_cycle_lifo_and_sibling_whale_freshness(tmp_path, monkeypatch):
    """Verify Deribit/Coinbase carryover on transient 503, active-cycle LIFO reconciliation, and freshest sibling whale sync."""
    from intelligence.sentiment.collectors import macro_telemetry_collector as mtc_mod
    from intelligence.sentiment.evaluators.microstructure_evaluator import reconcile_sibling_flow_and_liq_regimes
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        PositionCostBasisEnrichmentEvaluator,
        extract_symbol_recent_fills,
    )
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_microstructure_prompt_view

    t_now = 1_795_206_900.0

    # 1. MacroBenchmarksSnapshotCollector carries forward recent Deribit options surface & Coinbase premium on HTTP 503,
    #    and expires them cleanly once older than their staleness cap.
    cdir = tmp_path / "cachedb"
    macro_dir = cdir / "macro"
    macro_dir.mkdir(parents=True)
    prev_mb = {
        "ts": t_now - 600.0,
        "deribit_ts": t_now - 600.0,
        "coinbase_premium_ts": t_now - 120.0,
        "items": {"SPY": {"symbol": "SPY", "price": 690.0, "chg_24h_pct": -1.5}},
        "coinbase_premium_bps": -10.5,
        "coinbase_premium_regime": "US_SPOT_SELLING_PRESSURE",
        "btc_dvol": 56.4,
        "btc_rv_pct": 48.2,
        "btc_vrp_pct": 8.2,
        "btc_pc_oi_ratio": 0.64,
        "btc_skew_25d_pct": 5.1,
        "eth_dvol": 68.1,
        "eth_rv_pct": 61.0,
        "eth_vrp_pct": 7.1,
        "eth_pc_oi_ratio": 0.55,
        "eth_skew_25d_pct": 6.4,
        "options_vol_regime": "NORMAL_IV",
    }
    (macro_dir / "macro_benchmarks_collect.json").write_text(json.dumps(prev_mb), encoding="utf-8")

    def _mock_fetch_503(url, timeout=3.0, headers=None):
        if "yahoo" in url:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {"regularMarketPrice": 691.0, "previousClose": 700.0, "marketState": "REGULAR"},
                            "timestamp": [int(t_now - 3600), int(t_now)],
                            "indicators": {"quote": [{"close": [695.0, 691.0]}]},
                        }
                    ]
                }
            }
        raise RuntimeError("HTTP 503 system_maintenance")

    monkeypatch.setattr(mtc_mod, "fetch_public_json", _mock_fetch_503)
    snap_carried = mtc_mod.MacroBenchmarksSnapshotCollector().collect(str(cdir), now_ts=t_now, persist=True)
    assert snap_carried is not None
    assert snap_carried["btc_dvol"] == 56.4
    assert snap_carried["eth_dvol"] == 68.1
    assert snap_carried["btc_skew_25d_pct"] == 5.1
    assert snap_carried["options_vol_regime"] == "NORMAL_IV"
    assert snap_carried["deribit_ts"] == pytest.approx(t_now - 600.0)
    assert snap_carried["coinbase_premium_bps"] == -10.5
    assert "BTC_DVOL=56.4" in snap_carried["summary_line"]
    assert "ETH_DVOL=68.1" in snap_carried["summary_line"]
    assert "deribit_ts" not in snap_carried["summary_line"]

    # After staleness cap (> 3600s for Deribit, > 300s for Coinbase), stale carryover expires cleanly
    snap_expired = mtc_mod.MacroBenchmarksSnapshotCollector().collect(str(cdir), now_ts=t_now + 4000.0, persist=False)
    assert snap_expired is not None
    assert "btc_dvol" not in snap_expired
    assert "eth_dvol" not in snap_expired
    assert "coinbase_premium_bps" not in snap_expired

    # 2. Active-cycle LIFO reconciliation drops phantom pre-flat losses (TAOUSDC) while preserving genuine held-core swing PnL (BTCUSDC)
    t_ms = int(t_now * 1000)
    tao_orders = [
        # Ancient phantom fee-drift lot (never closed in order log due to fee dust, not held in wallet)
        {"symbol": "TAOUSDC", "side": "BUY", "executedQty": "62.31", "price": "320.0", "updateTime": t_ms - 60 * 86400 * 1000, "status": "FILLED"},
        # Prior cycle closed 100% flat before current cycle
        {"symbol": "TAOUSDC", "side": "BUY", "executedQty": "3.92", "price": "190.0", "updateTime": t_ms - 12 * 3600 * 1000, "status": "FILLED"},
        {"symbol": "TAOUSDC", "side": "SELL", "executedQty": "3.92", "price": "184.4", "updateTime": t_ms - 10 * 3600 * 1000, "status": "FILLED"},
        {"symbol": "TAOUSDC", "side": "SELL", "executedQty": "3.90", "price": "184.4", "updateTime": t_ms - 9 * 3600 * 1000, "status": "FILLED"},
        # Brand-new active cycle + two partial harvests (+0.28 net LIFO)
        {"symbol": "TAOUSDC", "side": "BUY", "executedQty": "2.2365", "price": "178.8", "updateTime": t_ms - 5 * 3600 * 1000, "status": "FILLED"},
        {"symbol": "TAOUSDC", "side": "BUY", "executedQty": "13.7709", "price": "176.5", "updateTime": t_ms - 4 * 3600 * 1000, "status": "FILLED"},
        {"symbol": "TAOUSDC", "side": "SELL", "executedQty": "11.3571", "price": "176.1", "updateTime": t_ms - 3 * 3600 * 1000, "status": "FILLED"},
        {"symbol": "TAOUSDC", "side": "SELL", "executedQty": "2.4089", "price": "178.5", "updateTime": t_ms - 2 * 3600 * 1000, "status": "FILLED"},
    ]
    tao_ro = extract_symbol_recent_fills(tao_orders, symbol="TAOUSDC")
    assert tao_ro is not None and tao_ro["lifo_realized_pnl_usd"] < -500.0
    tao_pos = {"venue": "binance", "symbol": "TAOUSDC", "price": 175.65, "free_base": 0.002259, "locked_base_qty": 2.2431}
    PositionCostBasisEnrichmentEvaluator._enrich_position_from_recent_orders(
        tao_pos,
        tao_ro,
        p_sym="TAOUSDC",
        p_quote_variants=("TAOUSDC",),
        pos_px_f=175.65,
        held_cur=2.245359,
        is_flat_balance=False,
        is_flat_after_exit=False,
        prices_and_trends={"TAOUSDC": {"price": 175.65}},
        cache_dir="",
        cycle_high_memo={},
    )
    assert tao_ro["lifo_realized_pnl_usd"] == pytest.approx(0.28, abs=0.02)
    assert tao_pos["lifo_realized_pnl_usd"] == pytest.approx(0.28, abs=0.02)

    # BTCUSDC-style held core lot + swing loss in active cycle stays intact
    btc_orders = [
        {"symbol": "BTCUSDC", "side": "BUY", "executedQty": "0.020", "price": "70000.0", "updateTime": t_ms - 30 * 86400 * 1000, "status": "FILLED"},
        {"symbol": "BTCUSDC", "side": "BUY", "executedQty": "0.030", "price": "64000.0", "updateTime": t_ms - 5 * 86400 * 1000, "status": "FILLED"},
        {"symbol": "BTCUSDC", "side": "BUY", "executedQty": "0.010", "price": "60000.0", "updateTime": t_ms - 6 * 3600 * 1000, "status": "FILLED"},
        {"symbol": "BTCUSDC", "side": "SELL", "executedQty": "0.010", "price": "58500.0", "updateTime": t_ms - 2 * 3600 * 1000, "status": "FILLED"},
    ]
    btc_ro = extract_symbol_recent_fills(btc_orders, symbol="BTCUSDC")
    btc_pos = {"venue": "binance", "symbol": "BTCUSDC", "price": 58460.0, "free_base": 0.030, "locked_base_qty": 0.0}
    PositionCostBasisEnrichmentEvaluator._enrich_position_from_recent_orders(
        btc_pos,
        btc_ro,
        p_sym="BTCUSDC",
        p_quote_variants=("BTCUSDC",),
        pos_px_f=58460.0,
        held_cur=0.030,
        is_flat_balance=False,
        is_flat_after_exit=False,
        prices_and_trends={"BTCUSDC": {"price": 58460.0}},
        cache_dir="",
        cycle_high_memo={},
    )
    assert btc_ro["lifo_realized_pnl_usd"] == pytest.approx(-15.0, abs=0.05)
    assert btc_pos["lifo_realized_pnl_usd"] == pytest.approx(-15.0, abs=0.05)

    # 3. Fresher sibling Binance Futures whale snapshot propagates to primary TAOUSDC and cleanly deduplicates TAOUSD
    sib_pt = {
        "TAOUSDC": {"price": 175.85, "chg_15m_pct": -0.03},
        "TAOUSD": {"price": 175.61, "chg_15m_pct": -0.03},
    }
    sib_micro = {
        "TAOUSDC": {
            "ob_mid_price": 175.85,
            "ob_imbalance": 0.45,
            "bid_depth_usd": 89594.0,
            "ask_depth_usd": 109489.0,
            "oi_usd": 40053156.04,
            "whale_long_pct": 0.6902,
            "taker_buy_sell_ratio": 1.2743,
            "taker_ratio_5m": 0.84,
            "cvd_5m_usd": -22400.0,
            "cvd_15m_usd": 107500.0,
            "_deriv_ts": t_now - 19.8,
            "_whale_ts": t_now - 66.8,
        },
        "TAOUSD": {
            "ob_mid_price": 175.61,
            "ob_imbalance": 0.527,
            "bid_depth_usd": 864533.0,
            "ask_depth_usd": 777400.0,
            "oi_usd": 40053156.04,
            "whale_long_pct": 0.6898,
            "taker_buy_sell_ratio": 1.2362,
            "taker_ratio_5m": 1.02,
            "cvd_5m_usd": 2200.0,
            "cvd_15m_usd": 83300.0,
            "_deriv_ts": t_now - 19.8,
            "_whale_ts": t_now - 4.1,
        },
    }
    reconcile_sibling_flow_and_liq_regimes(sib_micro, sib_pt)
    assert sib_micro["TAOUSDC"]["taker_buy_sell_ratio"] == pytest.approx(1.2362)
    assert sib_micro["TAOUSDC"]["taker_ratio_5m"] == pytest.approx(1.02)
    assert sib_micro["TAOUSDC"]["cvd_5m_usd"] == pytest.approx(2200.0)
    assert sib_micro["TAOUSDC"]["_whale_ts"] == pytest.approx(t_now - 4.1)

    rows = evaluate_microstructure_prompt_view(sib_micro, sib_pt)
    by_sym = {r["sym"]: r for r in rows}
    assert by_sym["TAOUSDC"]["shared_perp_sym"] is None
    assert by_sym["TAOUSD"]["shared_perp_sym"] == "TAOUSDC"


def test_iter7_sell_only_trade_plan_delta_gate_and_tranche_formatting():
    """Verify SELL-only ghost trade plans are pruned, Delta-Gate ignores unactionable $0/$0 plans, and tranches/OI avoid float dust/-0.0."""
    from types import SimpleNamespace
    from intelligence.sentiment.evaluators.portfolio_evaluator import format_recent_fill_subcycle_parts
    from orchestratorOS.supervisor.crypto_executor_delta_gate import evaluate_crypto_executor_delta_gate
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import summarize_active_ai_trade_plans

    t_now = 1_795_208_051.0

    # 1. summarize_active_ai_trade_plans drops FILLED SELL plans on SELL-only instruments (max_buy=$0, e.g. Kraken SPCXxUSD)
    #    while preserving FILLED SELL plans on tradable crypto (TAOUSD max_buy=$2000) and RESTING SELL plans on xStocks.
    intents = [
        {
            "status": "EXECUTED",
            "venue": "kraken",
            "symbol": "SPCXxUSD",
            "decision": "SELL",
            "order_type": "LIMIT",
            "timestamp": t_now - 8 * 3600.0,
            "target_px": 128.89,
            "invalidation_px": 131.2,
            "execution_payload": {"status": "FILLED", "price": 130.07, "filled_qty": 4.968943},
        },
        {
            "status": "EXECUTED",
            "venue": "kraken",
            "symbol": "TAOUSD",
            "decision": "SELL",
            "order_type": "LIMIT",
            "timestamp": t_now - 10 * 3600.0,
            "target_px": 184.2,
            "invalidation_px": 192.5,
            "execution_payload": {"status": "FILLED", "price": 187.42, "filled_qty": 0.61499},
        },
        {
            "status": "EXECUTED",
            "venue": "kraken",
            "symbol": "ANTxUSD",
            "decision": "SELL",
            "order_type": "LIMIT",
            "timestamp": t_now - 1800.0,
            "target_px": 85.0,
            "invalidation_px": 92.0,
            "execution_payload": {"status": "OPEN", "order_id": "O-ANT-1", "price": 90.0},
        },
    ]
    positions = [
        {"venue": "kraken", "symbol": "SPCXxUSD", "price": 128.625, "value_usd": 0.0, "effective_max_buy_usd": 0.0, "effective_max_sell_usd": 0.0},
        {"venue": "kraken", "symbol": "TAOUSD", "price": 175.20, "value_usd": 0.0, "effective_max_buy_usd": 2000.0, "effective_max_sell_usd": 0.0},
        {"venue": "kraken", "symbol": "ANTxUSD", "price": 87.505, "value_usd": 200.0, "effective_max_buy_usd": 0.0, "effective_max_sell_usd": 200.0},
    ]
    open_orders = [{"venue": "kraken", "symbol": "ANTxUSD", "side": "SELL", "order_id": "O-ANT-1", "price": 90.0, "status": "OPEN"}]
    plans = summarize_active_ai_trade_plans(
        intents,
        "crypto",
        t_now,
        extract_intent_notional,
        tradable_positions=positions,
        open_orders=open_orders,
    )
    plan_syms = {p["symbol"] for p in plans}
    assert "SPCXxUSD" not in plan_syms
    assert "TAOUSD" in plan_syms
    assert "ANTxUSD" in plan_syms

    # 2. Delta-Gate ignores trade_plan_target_hit / invalidated on unactionable $0/$0 symbols (no high-effort wakeup)
    sensors = SimpleNamespace(
        prices_and_trends={"BTCUSDC": {"price": 58300.0}, "SPCXxUSD": {"price": 128.625}},
        portfolio_summary={
            "tradable_positions": [
                {"venue": "binance", "symbol": "BTCUSDC", "price": 58300.0, "effective_max_buy_usd": 2000.0, "effective_max_sell_usd": 2000.0},
                {"venue": "kraken", "symbol": "SPCXxUSD", "price": 128.625, "effective_max_buy_usd": 0.0, "effective_max_sell_usd": 0.0},
            ],
            "active_ai_trade_plans": [
                {"venue": "kraken", "symbol": "SPCXxUSD", "side": "SELL", "order_state": "FILLED", "target_px": 128.89, "invalidation_px": 131.2, "status": "TARGET_HIT"},
            ],
            "open_limit_orders": [],
            "recent_orders": {},
        },
        sentiment_advisor={"ts": t_now - 1800.0},
        geopolitical_threat={"ts": t_now - 1200.0, "risk_score": 0.4},
        source_timestamps={},
    )
    prev_eval = {
        "ts": t_now - 360.0,
        "decision": "HOLD",
        "model_used": "gemini-4-neon-medium",
        "delta_gate_meta": {
            "eval_prices": {"BTCUSDC": 58310.0, "SPCXxUSD": 128.95},
            "macro_ts": t_now - 1800.0,
            "geo_ts": t_now - 1200.0,
            "latest_fill_ts": 0.0,
            "open_order_ids": [],
            "order_attention_flags": {},
            "plan_states": {"kraken:SPCXxUSD:SELL": "ACTIVE"},
        },
    }
    should_skip, reason, escalate_hi, _meta = evaluate_crypto_executor_delta_gate(
        sensors, prev_eval, now_ts=t_now
    )
    assert should_skip is True
    assert "quiet_flat" in reason
    assert escalate_hi is False

    # 3. format_recent_fill_subcycle_parts strips IEEE-754 float dust on tranche qty while preserving 8-decimal precision
    parts = format_recent_fill_subcycle_parts(
        {
            "recent_buys": [
                {"qty": 5.7299999999999995, "price": 61.02},
                {"qty": 6.721433899999999, "price": 66.95},
            ],
            "recent_sells": [
                {"qty": 3.0970999999999997, "price": 185.98},
                {"qty": 2155.17241379, "price": 0.1392},
            ],
        }
    )
    joined_tr = " | ".join(parts)
    assert "5.73@61.02" in joined_tr
    assert "6.7214339@66.95" in joined_tr
    assert "3.0971@185.98" in joined_tr
    assert "2155.17241379@0.1392" in joined_tr
    assert "999999" not in joined_tr

    # 4. Section 3B normalizes negative-zero oi_1h_chg (-0.0 -> 0.0)
    sec3b = "\n".join(
        _format_microstructure_section(
            {
                "BTCUSDC": {
                    "ob_imbalance": 0.6,
                    "bid_depth_usd": 1000000.0,
                    "ask_depth_usd": 700000.0,
                    "ob_mid_price": 58300.0,
                    "oi_usd": 5500000000.0,
                    "oi_1h_chg_pct": -0.0,
                    "funding_annualized_pct": -5.0,
                    "whale_long_pct": 0.63,
                    "taker_buy_sell_ratio": 0.9,
                    "taker_buy_vol_usd": 100000.0,
                    "taker_sell_vol_usd": 110000.0,
                }
            },
            {"BTCUSDC": {"price": 58300.0}},
        )
    )
    assert "oi_1h_chg=0.0%" in sec3b
    assert "oi_1h_chg=-0.0%" not in sec3b


def test_iter8_equity_ext_hours_24h_preservation_and_sudan_news_filter():
    """Verify continuous Yahoo extended-hours vol=0 equity bars stay intact for 24h delta and non-financial humanitarian war filler is filtered."""
    from intelligence.macro.lexicon_ssot import filter_valid_severity_keywords
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_24h_price_metrics,
        prepend_older_non_diverged_points,
    )

    # 1. EquityRthSessionPolicy.clean_raw_24h_series preserves continuous extended-hours vol=0 bars before RTH open
    #    (e.g., NVDA_US_EQ after 16:00 EST close when prior-day post-market + pre-market precede 09:30 EST RTH open)
    eq_pol = resolve_asset_session_policy("NVDA_US_EQ")
    t_rth0 = 1_795_185_000_000
    ext_bars = [
        [t_rth0 - 17 * 3600 * 1000, 145.31, 0.0, 145.98, 145.30, 145.54],
        [t_rth0 - 16 * 3600 * 1000, 144.80, 0.0, 145.10, 144.40, 145.00],
        [t_rth0 - 15 * 3600 * 1000, 144.20, 0.0, 144.60, 144.00, 144.50],
        [t_rth0 - 600_000, 143.50, 0.0, 143.70, 143.10, 143.40],
        [t_rth0 - 300_000, 143.64, 0.0, 144.27, 143.58, 143.58],
    ]
    rth_bars = [
        [t_rth0 + i * 300_000, round(140.82 - i * 0.25, 2), 12_000_000.0]
        for i in range(60)
    ]
    cleaned_nvda = eq_pol.clean_raw_24h_series(ext_bars + rth_bars, "NVDA_US_EQ")
    assert cleaned_nvda[0][1] == pytest.approx(145.31)
    m24_nvda = compute_24h_price_metrics(cleaned_nvda, is_equity=True)
    assert m24_nvda["chg_24h_pct"] is not None and m24_nvda["chg_24h_pct"] < -10.0

    # Flat multi-row pre-IPO indicative stub with >5% jump into IPO open is still stripped
    flat_stub = [
        [t_rth0 - 3600_000, 83.0, 0.0],
        [t_rth0 - 1800_000, 83.0, 0.0],
        [t_rth0 - 300_000, 83.0, 0.0],
        [t_rth0, 111.68, 160_000_000.0],
        [t_rth0 + 300_000, 111.50, 70_000_000.0],
        [t_rth0 + 600_000, 108.00, 50_000_000.0],
        [t_rth0 + 900_000, 106.50, 40_000_000.0],
    ]
    cleaned_stub = eq_pol.clean_raw_24h_series(flat_stub, "ANT_US_EQ")
    assert cleaned_stub[0][1] == pytest.approx(111.68)

    # 2. Equity divergence threshold (0.35) allows stitching -14% single-day equity crash slices
    older_eq = [[t_rth0 - 86_400_000, 146.0], [t_rth0 - 3600_000, 145.0]]
    cur_eq = [[t_rth0, 125.5], [t_rth0 + 300_000, 127.8]]
    stitched_eq = prepend_older_non_diverged_points(cur_eq, older_eq, threshold=0.35)
    assert len(stitched_eq) == 4 and stitched_eq[0][1] == pytest.approx(146.0)

    # 3. Humanitarian/education war filler headline is excluded while kinetic oil-theater strikes remain valid
    assert filter_valid_severity_keywords(
        ["war"], title="Sudan’s war leaves disabled children missing out on education"
    ) == ()
    assert "airstrike" in filter_valid_severity_keywords(
        ["airstrike"], title="Airstrike hits major oil terminal in Sudan"
    )


def test_iter9_prune_superseded_open_lots_after_partial_sell_reconciles_active_lifo_pnl():
    """Verify partial take-profit sells on an active cycle do not resurrect dead-cycle phantom lots or prior-cycle realized LIFO losses."""
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        SymbolRecentFillsEvaluator,
        enrich_positions_with_orders_and_cycles,
    )

    now_ms = 1_795_211_500_000
    day_ms = 86_400_000
    # 1. Exact ARBUSDC pattern: ancient dust lots (18d & 4.8d old) left behind by missing older fills,
    #    followed by prior-cycle exit sells (-$312.61), then fresh cycle buys (2624.6@0.1143 + 5162.1@0.1137),
    #    and a profitable partial take-profit sell (5096.6@0.1138 -> +$0.51).
    arb_orders = [
        {"orderId": "1", "side": "BUY", "price": 0.1595, "executedQty": 84.1, "status": "FILLED", "time": now_ms - 18 * day_ms},
        {"orderId": "2", "side": "BUY", "price": 0.1357, "executedQty": 27.8, "status": "FILLED", "time": now_ms - 4 * day_ms - 18 * 3600_000},
        {"orderId": "3", "side": "BUY", "price": 0.1310, "executedQty": 19746.3, "status": "FILLED", "time": now_ms - 2 * day_ms},
        {"orderId": "4a", "side": "SELL", "price": 0.1180, "executedQty": 15000.0, "status": "FILLED", "time": now_ms - 15 * 3600_000},
        {"orderId": "4b", "side": "SELL", "price": 0.1180, "executedQty": 4746.3, "status": "FILLED", "time": now_ms - 14 * 3600_000},
        {"orderId": "5", "side": "BUY", "price": 0.1143, "executedQty": 2624.6, "status": "FILLED", "time": now_ms - 6 * 3600_000},
        {"orderId": "6", "side": "BUY", "price": 0.1137, "executedQty": 5162.1, "status": "FILLED", "time": now_ms - 5 * 3600_000},
        {"orderId": "7", "side": "SELL", "price": 0.1138, "executedQty": 5096.6, "status": "FILLED", "time": now_ms - 120_000},
    ]
    ev = SymbolRecentFillsEvaluator()
    arb_ro = ev.evaluate(arb_orders, symbol="ARBUSDC")
    assert arb_ro is not None
    # Raw replay still carries ancient L1/L2 dust lots and prior-cycle realized loss
    assert len(arb_ro["open_lots"]) == 4
    assert arb_ro["lifo_realized_pnl_usd"] < -200.0

    # 2. Multi-tranche core+tactical position (BTCUSDC-style): older core lots ARE genuinely held
    #    and must NOT be pruned when newer lots alone do not cover held_qty.
    btc_orders = [
        {"orderId": "10", "side": "BUY", "price": 117000.0, "executedQty": 0.015, "status": "FILLED", "time": now_ms - 60 * day_ms},
        {"orderId": "11", "side": "BUY", "price": 64300.0, "executedQty": 0.023, "status": "FILLED", "time": now_ms - 5 * day_ms},
        {"orderId": "12", "side": "BUY", "price": 63000.0, "executedQty": 0.010, "status": "FILLED", "time": now_ms - 12 * 3600_000},
        {"orderId": "13", "side": "SELL", "price": 61500.0, "executedQty": 0.010, "status": "FILLED", "time": now_ms - 7 * 3600_000},
        {"orderId": "14", "side": "BUY", "price": 59400.0, "executedQty": 0.020, "status": "FILLED", "time": now_ms - 5 * 3600_000},
        {"orderId": "15", "side": "BUY", "price": 58420.0, "executedQty": 0.010, "status": "FILLED", "time": now_ms - 3 * 3600_000},
    ]
    btc_ro = ev.evaluate(btc_orders, symbol="BTCUSDC")
    assert btc_ro is not None

    positions = [
        {
            "venue": "binance",
            "symbol": "ARBUSDC",
            "free_base": 91.74511,
            "free_base_qty": 91.74511,
            "locked_base_qty": 2601.9,
            "price": 0.1135,
            "value_usd": 305.73,
        },
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "free_base": 0.033,
            "free_base_qty": 0.033,
            "locked_base_qty": 0.020,
            "price": 58963.0,
            "value_usd": 3125.04,
        },
    ]
    recent_orders = {"ARBUSDC": arb_ro, "BTCUSDC": btc_ro}
    enrich_positions_with_orders_and_cycles(
        positions,
        {"ARBUSDC": {"price": 0.1135}, "BTCUSDC": {"price": 58963.0}},
        {},
        {"binance": recent_orders},
        recent_orders,
        {},
        {},
        {},
        {},
    )
    arb_pos, btc_pos = positions[0], positions[1]
    assert len(arb_pos["open_lots"]) == 2
    assert [lt["entry_px"] for lt in arb_pos["open_lots"]] == [0.1143, 0.1137]
    assert arb_pos["active_avg_buy_px"] == pytest.approx(0.114285, abs=1e-6)
    assert arb_pos["lifo_realized_pnl_usd"] == pytest.approx(0.51, abs=0.01)
    assert recent_orders["ARBUSDC"]["lifo_realized_pnl_usd"] == pytest.approx(0.51, abs=0.01)
    assert "tactical_avg_buy_px" not in arb_pos

    # BTCUSDC sheds only the ancient 60d phantom lot (0.015), keeping its real 5d core (0.023) + 3h-5h tactical (0.030) = 0.053 BTC
    assert btc_pos["lifo_realized_pnl_usd"] == pytest.approx(-15.0, abs=0.01)
    assert btc_pos["tactical_avg_buy_px"] == pytest.approx((0.020 * 59400.0 + 0.010 * 58420.0) / 0.030, abs=0.05)
    assert sum(lt["qty"] for lt in recent_orders["BTCUSDC"]["open_lots"]) == pytest.approx(0.053, abs=1e-5)


def test_oauth_access_token_refresh_preserves_alert_cooldown_and_runtime_exhausted(tmp_path, monkeypatch):
    """Hourly OAuth access_token rotation on the same account must not reset alert cooldown or runtime_exhausted."""
    import orchestratorOS.lib.watchdog_common as wc_mod
    from intelligence.sentiment.gemini_client import GeminiClient

    q_dir = tmp_path / "q_refresh"
    q_dir.mkdir(parents=True, exist_ok=True)
    tok_file = q_dir / "antigravity-oauth-token"
    q_state_file = q_dir / "llm_quota_status.json"
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", staticmethod(lambda: str(q_state_file)))
    monkeypatch.setattr("intelligence.sentiment.llm_quota_router.resolve_bot_cli_base", lambda: str(q_dir))
    monkeypatch.setenv("LLM_QUOTA_ALERT_COOLDOWN_SEC", "14400")
    GeminiClient._QUOTA_MEM_CACHE = {}

    alerts = []
    monkeypatch.setattr(wc_mod, "send_ntfy", lambda title, msg, **kw: alerts.append((title, msg)))

    paid_model = os.environ["LLM_DEFAULT_MODEL"].strip()
    mock_state = {"paid_frac": 0.0, "claude_frac": 0.8, "flash_frac": 1.0, "tier": "g1-pro-tier"}

    def _mock_post(url, *args, **kwargs):
        if "loadCodeAssist" in url:
            return {
                "currentTier": {"id": mock_state["tier"]},
                "paidTier": {"id": mock_state["tier"], "availableCredits": []},
            }
        return {
            "buckets": [
                {"modelId": paid_model, "remainingFraction": mock_state["paid_frac"]},
                {"modelId": "claude-sonnet-4-6-thinking", "remainingFraction": mock_state["claude_frac"]},
                {"modelId": "gemini-3-flash", "remainingFraction": mock_state["flash_frac"]},
            ]
        }

    monkeypatch.setattr("intelligence.sentiment.llm_quota_router.fetch_json_post", _mock_post)

    # 1. Initial exhausted probe on account A sends exactly 1 alert
    tok_file.write_text(
        json.dumps({
            "id_token_email": "bot-acct-a@example.com",
            "token": {
                "access_token": "ya29.acctA-hour1",
                "refresh_token": "1//acctA-refresh",
                "expires_in": 3599,
            },
        }),
        encoding="utf-8",
    )
    snap1 = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=True)
    assert snap1 is not None
    assert len(alerts) == 1
    assert snap1["last_alert_tier"] == "primary_low"
    fp_a = snap1["oauth_token_fp"]
    alert_ts_1 = snap1["last_alert_ts"]
    assert alert_ts_1 > 0.0

    GeminiClient._record_model_quota_exhausted(paid_model, ttl_sec=600.0)

    # 2. Routine hourly access_token refresh on the SAME account must preserve fp, cooldown, and runtime_exhausted
    tok_file.write_text(
        json.dumps({
            "id_token_email": "bot-acct-a@example.com",
            "token": {
                "access_token": "ya29.acctA-hour2",
                "refresh_token": "1//acctA-refresh",
                "expires_in": 3599,
            },
        }),
        encoding="utf-8",
    )
    snap2 = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=True)
    assert snap2 is not None
    assert snap2["oauth_token_fp"] == fp_a
    assert snap2["last_alert_ts"] == pytest.approx(alert_ts_1, abs=1e-6)
    assert snap2["last_alert_tier"] == "primary_low"
    assert paid_model in snap2["runtime_exhausted"]
    assert len(alerts) == 1  # Cooldown respected: no hourly duplicate alert

    # 3. Refresh-token fallback when id_token_email is absent also stays stable across access_token rotation
    tok_file.write_text(
        json.dumps({"token": {"access_token": "ya29.rt-hour1", "refresh_token": "1//stable-rt"}}),
        encoding="utf-8",
    )
    snap_rt1 = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=True)
    assert snap_rt1 is not None
    tok_file.write_text(
        json.dumps({"token": {"access_token": "ya29.rt-hour2", "refresh_token": "1//stable-rt"}}),
        encoding="utf-8",
    )
    snap_rt2 = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=True)
    assert snap_rt2 is not None
    assert snap_rt2["oauth_token_fp"] == snap_rt1["oauth_token_fp"]
    assert len(alerts) == 2  # One alert for the account change to rt-only, zero for its hourly refresh

    # 4. Genuine account switch (different id_token_email) invalidates cache immediately and clears runtime_exhausted
    GeminiClient._record_model_quota_exhausted(paid_model, ttl_sec=600.0)
    mock_state["paid_frac"] = 1.0
    mock_state["tier"] = "g1-ultra-tier"
    tok_file.write_text(
        json.dumps({
            "id_token_email": "bot-acct-b@example.com",
            "token": {
                "access_token": "ya29.acctB-fresh",
                "refresh_token": "1//acctB-refresh",
                "expires_in": 3599,
            },
        }),
        encoding="utf-8",
    )
    snap_b = GeminiClient.probe_live_model_quota(bot_cli_base=str(q_dir), force_refresh=False)
    assert snap_b is not None
    assert snap_b["oauth_token_fp"] != snap_rt2["oauth_token_fp"]
    assert snap_b["tier_id"] == "g1-ultra-tier"
    assert snap_b["buckets"][paid_model] == 1.0
    assert snap_b["runtime_exhausted"] == {}
    assert snap_b["last_alert_tier"] == "ok"


def test_va_breakout_hold_duration_and_taker_ratio_cycle_delta() -> None:
    """Verify symmetric Value-Area breakout hold duration (above_vah/below_val) and deadbanded d_tr cycle_delta."""
    from intelligence.sentiment.evaluators.microstructure_evaluator import evaluate_microstructure_cycle_deltas
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_va_breakout_hold_min,
        compute_volume_profile_metrics,
        format_va_hold_suffix,
        recalculate_live_price_geometry,
        restore_rth_metric_anchors,
    )
    from orchestratorOS.supervisor.prompt_sections import (
        _format_market_prices_section,
        _format_microstructure_section,
    )

    t0_ms = 1_795_000_000_000.0
    min_ms = 60_000.0

    # 1. Sustained breakout above VAH (35m hold) and symmetric breakdown below VAL (40m hold), zero emission inside VA
    base_range = [[t0_ms + i * 5.0 * min_ms, 100.0 + (i % 5) * 0.5, 500.0] for i in range(40)]
    base_range.append([base_range[-1][0] + 5.0 * min_ms, 100.5, 500.0])  # Anchor last base bar strictly inside VA
    last_base_ms = base_range[-1][0]
    above_tail = [[last_base_ms + (k + 1) * 5.0 * min_ms, 112.0 + k * 0.2, 50.0] for k in range(8)]
    vp_above = compute_volume_profile_metrics(base_range + above_tail)
    assert vp_above["va_regime"] == "ABOVE_VAH"
    assert vp_above["va_hold_min"] == pytest.approx(35.0)
    assert format_va_hold_suffix(vp_above) == ", above_vah=35m"

    below_tail = [[last_base_ms + (k + 1) * 5.0 * min_ms, 88.0 - k * 0.2, 50.0] for k in range(9)]
    vp_below = compute_volume_profile_metrics(base_range + below_tail)
    assert vp_below["va_regime"] == "BELOW_VAL"
    assert vp_below["va_hold_min"] == pytest.approx(40.0)
    assert format_va_hold_suffix(vp_below) == ", below_val=40m"

    inside_tail = [[last_base_ms + (k + 1) * 5.0 * min_ms, 101.0, 50.0] for k in range(4)]
    vp_inside = compute_volume_profile_metrics(base_range + inside_tail)
    assert vp_inside["va_regime"] in ("INSIDE_VA_UPPER", "INSIDE_VA_LOWER")
    assert "va_hold_min" not in vp_inside
    assert format_va_hold_suffix(vp_inside) == ""

    # 2. Fresh 1-sample breakout yields 0.0m hold; >1h overnight gap stops hold accumulation
    fresh_pts = [(t0_ms, 100.0), (t0_ms + 20.0 * min_ms, 100.0), (t0_ms + 25.0 * min_ms, 115.0)]
    assert compute_va_breakout_hold_min(fresh_pts, 98.0, 105.0) == 0.0
    assert format_va_hold_suffix({"va_regime": "ABOVE_VAH", "va_hold_min": 0.0}) == ", above_vah=0m"

    gapped_pts = [
        (t0_ms, 100.0),
        (t0_ms + 10.0 * min_ms, 110.0),
        (t0_ms + 200.0 * min_ms, 111.0),  # >1h closed-market gap
        (t0_ms + 215.0 * min_ms, 112.0),
    ]
    assert compute_va_breakout_hold_min(gapped_pts, 98.0, 105.0) == pytest.approx(15.0)

    # 3. Live price geometry clears va_hold_min on reversion inside VA, resets to 0.0 on fresh flip, and RTH restore drops stale hold
    live_entry = {"price": 113.4, **vp_above}
    recalculate_live_price_geometry(live_entry, vp_above["poc_px"])
    assert "va_hold_min" not in live_entry
    recalculate_live_price_geometry(live_entry, vp_above["vah_px"] * 1.02)
    assert live_entry["va_regime"] == "ABOVE_VAH"
    assert live_entry["va_hold_min"] == 0.0

    rth_entry = {"price": 113.4, **vp_above}
    restore_rth_metric_anchors(
        rth_entry,
        vp_src={"poc_px": 101.0, "val_px": 99.0, "vah_px": 103.0, "vp_source": "REAL_VOL"},
        live_px=113.4,
    )
    assert "va_hold_min" not in rth_entry

    # 4. Section 3A prompt formatting emits above_vah / below_val conditionally and omits inside VA
    sec_3a = "\n".join(
        _format_market_prices_section(
            {
                "BTCUSDC": {"price": 113.4, **vp_above},
                "TAOUSDC": {"price": 86.4, **vp_below},
                "ARBUSDC": {"price": 101.0, **vp_inside},
            }
        )
    )
    assert "above_vah=35m" in sec_3a
    assert "below_val=40m" in sec_3a
    arb_line = next(ln for ln in sec_3a.splitlines() if "ARBUSDC" in ln)
    assert "above_vah=" not in arb_line and "below_val=" not in arb_line

    # 5. taker_ratio_delta steps back to prior distinct 5m bar consistently with cvd_15m_delta and applies 0.05 prompt deadband
    now_s = 1_795_100_000.0
    bar_prev_ms = int((now_s - 600.0) * 1000)
    bar_cur_ms = int((now_s - 180.0) * 1000)
    hist_tr = {
        "BTCUSDC": [
            {
                "ts": now_s - 600.0,
                "ob_imbalance": 0.48,
                "cvd_15m_usd": -2_000_000.0,
                "oi_usd": 5_000_000_000.0,
                "taker_buy_sell_ratio": 0.82,
                "taker_bar_ts_ms": bar_prev_ms,
            },
            {
                "ts": now_s - 360.0,
                "ob_imbalance": 0.51,
                "cvd_15m_usd": 3_000_000.0,
                "oi_usd": 5_005_000_000.0,
                "taker_buy_sell_ratio": 1.18,
                "taker_bar_ts_ms": bar_cur_ms,
            },
        ],
        "ARBUSDC": [
            {
                "ts": now_s - 360.0,
                "ob_imbalance": 0.50,
                "cvd_15m_usd": 100_000.0,
                "oi_usd": 50_000_000.0,
                "taker_buy_sell_ratio": 1.00,
                "taker_bar_ts_ms": bar_prev_ms,
            }
        ],
    }
    micro_tr = {
        "BTCUSDC": {
            "ob_imbalance": 0.55,
            "cvd_15m_usd": 3_005_000.0,
            "oi_usd": 5_010_000_000.0,
            "taker_buy_sell_ratio": 1.18,
            "_taker_bar_ts_ms": bar_cur_ms,
        },
        "ARBUSDC": {
            "ob_imbalance": 0.52,
            "cvd_15m_usd": 120_000.0,
            "oi_usd": 50_100_000.0,
            "taker_buy_sell_ratio": 1.03,  # +0.03 < 0.05 deadband
            "_taker_bar_ts_ms": bar_cur_ms,
        },
    }
    out_micro_tr, _, _ = evaluate_microstructure_cycle_deltas(micro_tr, hist_tr, now_ts=now_s)
    assert out_micro_tr["BTCUSDC"]["cycle_delta"]["taker_ratio_delta"] == pytest.approx(0.36)
    assert out_micro_tr["BTCUSDC"]["cycle_delta"]["cvd_15m_delta_usd"] == pytest.approx(5_005_000.0)
    assert out_micro_tr["ARBUSDC"]["cycle_delta"]["taker_ratio_delta"] == pytest.approx(0.03)

    sec_3b = "\n".join(
        _format_microstructure_section(
            out_micro_tr,
            {"BTCUSDC": {"price": 62000.0}, "ARBUSDC": {"price": 0.12}},
            now_ts=now_s,
        )
    )
    btc_3b = next(ln for ln in sec_3b.splitlines() if "BTCUSDC" in ln)
    arb_3b = next(ln for ln in sec_3b.splitlines() if "ARBUSDC" in ln)
    assert "d_tr=+0.36" in btc_3b
    assert "d_tr=" not in arb_3b


def test_disabled_registry_symbols_xstock_longterm_cost_and_partial_24h_vp(tmp_path):
    """Verify disabled instruments exclusion, SPCXxUSD live cycle cost/MANUAL actor/long-term directive, and partial-24h VP upgrade."""
    from instrument_registry import disabled_registry_symbols
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        MarketAndPortfolioSnapshotOrchestrator,
        _harvest_initial_prices_and_trends_impl,
    )
    from intelligence.sentiment.collectors.microstructure_collector import (
        discover_reference_microstructure_symbols,
        upgrade_tpo_vp_from_ohlc,
    )
    from intelligence.sentiment.collectors.portfolio_collector import (
        KrakenXstocksCollector,
        RecentOrdersAndPositionEnricherCollector,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        apply_cycle_fallback_to_recent_fills,
        parse_venue_cycle_state,
    )
    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot

    ws = tmp_path
    cdir = ws / "cachedb"
    tel_dir = cdir / "telemetry"
    ord_dir = cdir / "orders"
    krk_dir = ws / "kraken"
    for d in (tel_dir, ord_dir, krk_dir):
        d.mkdir(parents=True, exist_ok=True)

    (ws / "instruments.conf").write_text(
        "[BINANCE_ADA]\nprovider = binance\nsymbol = ADAUSDC\nbase = ADA\nquote = USDC\n"
        "enabled = yes\nisolation = own_ledger\nmarket_hours = 24x7\n"
        "role.mt = yes\nrole.tradeall_fire = no\nrole.kalman_primary = no\n"
        "role.trailing = no\nrole.assetguardian = no\nrole.archive = yes\nrole.rtrade = no\nrole.force_sell = no\n\n"
        "[KRAKEN_ADA]\nprovider = kraken\nsymbol = ADAUSD\nbase = ADA\nquote = USD\n"
        "enabled = no\nisolation = own_ledger\nmarket_hours = 24x7\n"
        "role.mt = no\nrole.tradeall_fire = no\nrole.kalman_primary = no\n"
        "role.trailing = no\nrole.assetguardian = no\nrole.archive = no\nrole.rtrade = no\nrole.force_sell = no\n",
        encoding="utf-8",
    )
    assert "ADAUSD" in disabled_registry_symbols(ws)
    assert "ADAUSDC" not in disabled_registry_symbols(ws)

    now_s = 1_795_530_000.0
    prices_doc = {
        "items": {
            "ADAUSDC": [[int((now_s - 2.0) * 1000), 0.151]],
            "ADAUSD": [[int((now_s - 65_000.0) * 1000), 0.148]],
        }
    }
    pt = _harvest_initial_prices_and_trends_impl(str(cdir), prices_doc, {}, workspace_dir=str(ws))
    assert "ADAUSDC" in pt and "ADAUSD" not in pt

    (tel_dir / "orderbook_depth_ADAUSD.json").write_text(
        json.dumps({"symbol": "ADAUSD", "bid_depth_usd": 5000.0, "timestamp": now_s - 200.0}),
        encoding="utf-8",
    )
    ref_syms = discover_reference_microstructure_symbols(str(cdir), {"ADAUSDC": {"bid_depth_usd": 50_000.0}})
    assert "ADAUSD" not in ref_syms

    snap = TelemetrySensorSnapshot()
    snap.prices_and_trends = dict(pt)
    snap.microstructure = {}
    MarketAndPortfolioSnapshotOrchestrator._stamp_snapshot_domain_timestamps(
        snap,
        scope="crypto",
        allow_network=False,
        persist=False,
        now_ts=now_s,
        cache_dir=str(cdir),
        read_json_fn=lambda _p: None,
        macro_sensors={},
        prices_data=prices_doc,
        cur_p_items=prices_doc["items"],
        trends_data={},
        inst_by_sym={},
        t212_cycles={},
    )
    assert snap.source_ages_sec.get("prices_newest", 999.0) < 10.0
    assert snap.source_ages_sec.get("prices_oldest", 0.0) < 10.0

    (ord_dir / "cache_orders_filled_kraken.json").write_text(
        json.dumps({"items": {"ADAUSD": [{"orderId": "k1", "side": "SELL", "price": 0.148, "executedQty": 100.0, "timestamp": int((now_s - 70000) * 1000)}]}}),
        encoding="utf-8",
    )
    ro_out = RecentOrdersAndPositionEnricherCollector(start_sync=False).collect(
        str(cdir), inst_by_sym={}, tradable_positions=[], prices_and_trends=pt, allow_network=False
    )
    assert "ADAUSD" not in ro_out

    # 2. KrakenXstocksCollector prefers live active cycle cost ($134.844) over static alloc ($141.75),
    # reflects OBSERVE_ONLY + dynamic order_guard.conf 6% threshold, and isolates dormant IPO fields when XSTOCK_REGEX/XSTOCK_YAHOO are empty
    (krk_dir / "config.env").write_text(
        "XSTOCK_REGEX=\nXSTOCK_YAHOO=\nXSTOCK_ACTIVE_PAIRS=SPCXxUSD:141.75\nXSTOCK_BOT_OBSERVE_ONLY=true\n"
        "XSTOCK_BOT_TP_PCT=9.0\nXSTOCK_TP_ALERT_PCT=10\nXSTOCK_TP2_ALERT_PCT=16\nXSTOCK_SL_ALERT_PCT=12\n",
        encoding="utf-8",
    )
    (tmp_path / "order_guard.conf").write_text(
        "autonomous_trading_xstock_sell_min_profit_pct = 6.0\n"
        "autonomous_trading_xstock_sell_min_target_drop_pct = 4.0\n",
        encoding="utf-8",
    )
    (krk_dir / ".xstock_state.json").write_text(
        json.dumps({"known_assets": {"SPCXx.T": 24.472724}, "pair": "SPCXxUSD", "allocated": {"asset": "SPCXx.T", "qty": 4.27}}),
        encoding="utf-8",
    )
    (krk_dir / ".state_SPCXxUSD.json").write_text(
        json.dumps({"qty": 24.472724, "cost": 3300.0, "entry_price": 134.844006, "entry_ts": now_s - 300.0, "adopted": True}),
        encoding="utf-8",
    )
    xs = KrakenXstocksCollector(start_sync=False).collect(
        str(cdir), kraken_dir=str(krk_dir), prices_and_trends={"SPCXxUSD": {"price": 129.58}}
    )
    assert xs is not None
    assert xs["alloc_price"] == pytest.approx(134.844006, rel=1e-4)
    assert xs["pnl_vs_alloc_pct"] == pytest.approx(((129.58 - 134.844006) / 134.844006) * 100.0, abs=0.05)
    assert "LONG-TERM INVESTMENT" in xs["active_swing_directive"]
    assert "OBSERVE-ONLY" in xs["active_swing_directive"]
    assert "+6%" in xs["active_swing_directive"]
    assert "4%" in xs["active_swing_directive"]
    assert "HEAVY re-entry commission" in xs["active_swing_directive"]
    assert "tp1_alert_pct" not in xs
    assert "watched_ipo_playbook" not in xs
    assert "watched_ipo_underlying" not in xs
    assert xs["allocated"] is None

    # 3. Adopted / SELL-only tokenized equity BUY fallback sets actor="MANUAL" (not "BOT")
    parsed_cyc = parse_venue_cycle_state(
        {"qty": 24.472724, "cost": 3300.0, "entry_price": 134.844006, "entry_ts": now_s - 300.0, "adopted": True},
        "SPCXxUSD",
        {"SPCXxUSD": {"price": 129.58}},
        include_meta=True,
    )
    assert parsed_cyc.get("_adopted") is True
    ro_xs = apply_cycle_fallback_to_recent_fills("kraken", parsed_cyc, True, None, symbol="SPCXxUSD")
    assert ro_xs is not None
    assert ro_xs["last_action_actor"] == "MANUAL"
    assert ro_xs["last_buy"]["actor"] == "MANUAL"

    # 3b. Real Kraken filled BUY order on SELL-only tokenized equity (no bot/AI origin) also resolves actor="MANUAL",
    # and subsumed child trade-ID rows (_fillIds inside parent O... order) are never double-counted in LIFO PnL/open_lots!
    from intelligence.sentiment.evaluators.portfolio_evaluator import extract_symbol_recent_fills
    fills_xs = extract_symbol_recent_fills(
        [
            {"orderId": "TSR6LOW-BZ47F-UY2BAV", "_fillIds": ["TSR6LOW-BZ47F-UY2BAV"], "side": "BUY", "price": 134.844006, "quantity": 24.472723, "timestamp": int((now_s - 600) * 1000)},
            {"orderId": "TAIDKY-CX6QX-MQEL6T", "_fillIds": ["TAIDKY-CX6QX-MQEL6T"], "side": "SELL", "price": 125.64002, "quantity": 1.984, "timestamp": int((now_s - 310) * 1000)},
            {"orderId": "TJJTVY-32PLJ-DY7QOY", "_fillIds": ["TJJTVY-32PLJ-DY7QOY"], "side": "SELL", "price": 125.64002, "quantity": 1.984, "timestamp": int((now_s - 305) * 1000)},
            {"orderId": "OFMFTT-JDI7H-72NOAO", "_fillIds": ["TAIDKY-CX6QX-MQEL6T", "TJJTVY-32PLJ-DY7QOY", "TAVULX-HCEZI-BUXGTE"], "side": "SELL", "price": 125.64, "quantity": 12.336835, "timestamp": int((now_s - 300) * 1000)},
            {"orderId": "TAVULX-HCEZI-BUXGTE", "_fillIds": ["TAVULX-HCEZI-BUXGTE"], "side": "SELL", "price": 125.639996, "quantity": 8.368835, "timestamp": int((now_s - 300) * 1000)},
        ],
        known_ai_order_ids={"OFMFTT-JDI7H-72NOAO"},
        symbol="SPCXxUSD",
    )
    assert fills_xs is not None
    assert fills_xs["last_action_side"] == "SELL"
    assert fills_xs["last_action_actor"] == "AI"
    assert fills_xs["last_buy"]["actor"] == "MANUAL"
    assert fills_xs["last_sell"]["actor"] == "AI"
    assert fills_xs["last_sell"]["qty"] == pytest.approx(12.336835)
    assert len(fills_xs["recent_sells"]) == 1
    assert len(fills_xs["open_lots"]) == 1
    assert fills_xs["open_lots"][0]["qty"] == pytest.approx(12.135888, abs=1e-4)
    assert fills_xs["lifo_realized_pnl_usd"] == pytest.approx(-113.55, abs=0.2)

    # 3c. Observe-only symbols (ANTxUSD) never trigger Delta Gate wakeups or high-effort model escalations,
    # and completely empty observe-only venue cycles (ANT_US_EQ) are omitted from Section 3C anchors.
    from orchestratorOS.supervisor.crypto_executor_delta_gate import evaluate_crypto_executor_delta_gate
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_portfolio_prompt_view
    dg_skip, dg_reason, dg_hi, _ = evaluate_crypto_executor_delta_gate(
        SimpleNamespace(
            prices_and_trends={"BTCUSDC": {"price": 59500.0}, "ANTxUSD": {"price": 79.0}},
            portfolio_summary={"tradable_positions": [{"venue": "binance", "symbol": "BTCUSDC", "effective_max_buy_usd": 2000.0, "effective_max_sell_usd": 1000.0}]},
            sentiment_advisor=None,
            geopolitical_threat=None,
            source_timestamps={},
        ),
        {
            "ts": now_s - 360.0,
            "market_trading_decision": {"decision": "HOLD"},
            "delta_gate_meta": {"eval_prices": {"BTCUSDC": 59510.0, "ANTxUSD": 77.0}},
        },
        now_ts=now_s,
    )
    assert dg_skip is True, (dg_skip, dg_reason, dg_hi)
    assert dg_hi is False
    pv_obs = evaluate_portfolio_prompt_view(
        {
            "t212_cycles": {
                "ANT_US_EQ": {"qty": 0.0, "cost_usd": 0.0, "last_buy_price": None, "last_sell_price": None, "realized_pnl_usd": 0.0},
                "SPCX_US_EQ": {"qty": 4.44, "cost_usd": 635.94, "avg_cost_px": 143.13},
            }
        },
        asset_scope="equities",
    )
    t212_syms_in_view = [s for _, vid, items in pv_obs["venue_cycles"] if vid == "t212" for s, _ in items]
    assert t212_syms_in_view == ["SPCX_US_EQ"]

    # 3d. Flat observe-only symbols (ANTxUSD) omit stale Last Action lines from Section 3C, while closed-cycle loss
    # exits on T212 (e.g. NVDA_US_EQ sold at $156.15 vs $164.94 buy, realized_pnl < 0) reclassify bogus TP -> REDUCE.
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    sec3c_lines = "\n".join(
        _format_portfolio_section(
            {
                "tradable_positions": [
                    {"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.02, "free_quote": 1000.0, "price": 60000.0}
                ],
                "recent_orders": {
                    "ANTxUSD": {"last_action_side": "SELL", "last_action_actor": "AI", "last_sell": {"qty": 0.26, "price": 98.48, "ts_ms": int((now_s - 400000.0) * 1000)}},
                    "BTCUSDC": {"last_action_side": "BUY", "last_action_actor": "AI", "last_buy": {"qty": 0.02, "price": 60000.0, "ts_ms": int((now_s - 600.0) * 1000)}},
                },
            },
            asset_scope="crypto",
        )
    )
    assert "BTCUSDC Last Action=" in sec3c_lines
    assert "ANTxUSD Last Action=" not in sec3c_lines

    from intelligence.sentiment.collectors.portfolio_collector import (
        harvest_recent_orders_and_enrich_positions,
    )
    t212_cache_dir = tmp_path / "t212_loss_exit"
    t212_cache_dir.mkdir(parents=True, exist_ok=True)
    (t212_cache_dir / "cache_trade_t212.json").write_text(
        json.dumps(
            {
                "items": {
                    "NVDA_US_EQ": [
                        {"symbol": "NVDA_US_EQ", "orderId": "B_NVDA", "price": 164.94, "qty": 1.09, "time": int((now_s - 86400.0) * 1000), "side": "BUY"},
                        {"symbol": "NVDA_US_EQ", "orderId": "S_NVDA", "price": 156.15, "qty": 1.09, "time": int((now_s - 3600.0) * 1000), "side": "SELL"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    nvda_cyc = parse_venue_cycle_state(
        {
            "qty": 0.0,
            "cost_usd": 0.0,
            "entry_price": None,
            "last_buy_price": None,
            "last_sell_price": 156.15,
            "last_sell_ts": now_s - 3600.0,
            "last_exit_kind": "TP",
            "realized_net_usd": -29.26,
        },
        "NVDA_US_EQ",
        {"NVDA_US_EQ": {"price": 153.0}},
        include_meta=True,
    )
    t212_cycs = {"NVDA_US_EQ": nvda_cyc}
    t212_pos = [{"venue": "t212", "symbol": "NVDA_US_EQ", "base": "NVDA_US_EQ", "quote": "USD", "price": 153.0, "free_base": 0.0, "free_quote": 579.0}]
    harvest_recent_orders_and_enrich_positions(
        str(t212_cache_dir), {}, t212_pos, {"NVDA_US_EQ": {"price": 153.0}}, {}, {}, t212_cycs
    )
    assert t212_cycs["NVDA_US_EQ"]["last_exit_kind"] == "REDUCE"

    # 4. Partial local 24h tick history (<22h) still upgrades TPO -> REAL_VOL when overlap matches
    ohlc_pts = []
    for h in range(24):
        ts_ms = int((now_s - (23 - h) * 3600.0) * 1000)
        px = 0.1547 if h < 4 else 0.1495
        ohlc_pts.append([ts_ms, px, 10000.0, px * 1.002, px * 0.998, px])
    (cdir / "cache_ohlc_24h_ADAUSDC.json").write_text(json.dumps({"points": ohlc_pts}), encoding="utf-8")
    m24_partial = {"vp_source": "TPO", "low_24h": 0.1470, "high_24h": 0.1507}
    upgraded = upgrade_tpo_vp_from_ohlc(
        "ADAUSDC",
        m24_partial,
        0.1495,
        now_s,
        s24_is_live_clock=True,
        cache_dir=str(cdir),
        allow_network=False,
        series_24h_start_ms=(now_s - 18.0 * 3600.0) * 1000.0,
    )
    assert len(upgraded) >= 24
    assert m24_partial["vp_source"] == "REAL_VOL"
    assert "session_vwap_px" in m24_partial or "poc_px" in m24_partial


def test_openai_compat_soft_conserve_stateless_topics_reversible_fallback_and_memory(tmp_path, monkeypatch):
    """Verify <=25% stateless-topic GLM routing, auto_technical_repair exclusion, reversible agy fallback, and rolling_history continuity."""
    from types import SimpleNamespace
    import intelligence.sentiment.gemini_client as gc_mod
    from intelligence.sentiment.openai_compat_rest import extract_openai_compat_text
    from orchestratorOS.agy.topics import load_topic_registry, prepare_topic_session, record_topic_turn

    reg_path = tmp_path / "topic_registry.json"
    monkeypatch.setenv("LLM_TOPIC_REGISTRY_PATH", str(reg_path))
    monkeypatch.setenv("OPENAI_COMPAT_FALLBACK_URL", "https://api.z.ai/api/coding/paas/v4")
    monkeypatch.setenv("OPENAI_COMPAT_FALLBACK_API_KEY", "sk-zai-coding-test")
    monkeypatch.setenv("OPENAI_COMPAT_FALLBACK_MODEL", "glm-5.4,glm-5.3")
    monkeypatch.setenv("LLM_QUOTA_SOFT_CONSERVE_FRACTION", "0.25")
    monkeypatch.delenv("LLM_TOPIC_MODEL_MACRO_SENTIMENT", raising=False)

    # Seed existing agy thread in topic registry so we verify rolling_history injection + needs_full_prompt=True
    record_topic_turn(
        "macro_sentiment",
        "cid-existing-macro",
        '{"market_bias": "NEUTRAL", "summary": "prior agy decision"}',
        registry_path=str(reg_path),
    )

    # 1. Z.ai reasoning_content format: reads content cleanly and ignores reasoning_content; rejects length truncation
    assert extract_openai_compat_text(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"market_bias":"BULLISH","summary":"glm ok"}',
                        "reasoning_content": "Internal reasoning chain mentioning BEARISH",
                    },
                }
            ]
        },
        require_complete=True,
    ) == '{"market_bias":"BULLISH","summary":"glm ok"}'

    posted = []
    cli_calls = []
    oai_should_fail = {"flag": False}

    def _fake_post(url, payload, *, timeout=5.0, headers=None):
        posted.append((url, payload))
        if oai_should_fail["flag"]:
            raise RuntimeError("Z.ai 503 temporary outage")
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"market_bias":"BULLISH","summary":"via_glm_5_4"}',
                        "reasoning_content": "chain of thought",
                    },
                }
            ]
        }

    def _fake_cli(self, resume, text, call_timeout, cli_model, *args, **kwargs):
        cli_calls.append(cli_model)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"response": '{"market_bias":"NEUTRAL","summary":"via_agy_fallback"}', "conversation_id": "cid-existing-macro"}),
            stderr="",
        )

    monkeypatch.setattr(gc_mod, "fetch_json_post", _fake_post)
    client = gc_mod.GeminiClient(cli_path="/bin/sh", project_id="mptrade")
    monkeypatch.setattr(client, "_execute_cli", _fake_cli.__get__(client, gc_mod.GeminiClient))
    monkeypatch.setattr(
        client,
        "probe_live_model_quota",
        lambda **_kw: {
            "available_credits": 0.0,
            "buckets": {
                "gemini-4-neon-xhigh": 0.20,
                "gemini-4-neon-high": 0.20,
                "gemini-4-neon-medium": 0.20,
                "gemini-4-neon-low": 0.20,
            },
        },
    )

    # 2. Stateless topic (macro_sentiment) at 20% <= 25% routes automatically to glm-5.4, injects rolling_history,
    #    records the GLM decision, and marks needs_full_prompt=True while keeping existing agy conversation_id intact.
    res_macro = client.query_topic_json("macro_sentiment", "Evaluate macro regime")
    assert res_macro == {"market_bias": "BULLISH", "summary": "via_glm_5_4"}
    assert client.last_model_used == "openai_compat:glm-5.4"
    assert len(cli_calls) == 0
    assert posted[-1][0] == "https://api.z.ai/api/coding/paas/v4/chat/completions"
    assert posted[-1][1]["model"] == "glm-5.4"
    assert "prior agy decision" in posted[-1][1]["messages"][0]["content"]
    reg_after = load_topic_registry(str(reg_path))
    macro_entry = reg_after["macro_sentiment"]
    assert macro_entry["conversation_id"] == "cid-existing-macro"
    assert macro_entry["needs_full_prompt"] is True
    assert macro_entry["needs_history_sync"] is True
    assert any("via_glm_5_4" in str(h.get("summary") or "") for h in macro_entry["rolling_history"])

    # 2b. When agy resumes cid-existing-macro after GLM turns, prepare_topic_session injects rolling_history
    #     containing GLM's decision, and the subsequent agy turn clears needs_history_sync back to False.
    import orchestratorOS.agy.topics as agy_topics_mod
    monkeypatch.setattr(agy_topics_mod, "sync_topic_thread_from_ui", lambda *_a, **_kw: None)
    monkeypatch.setattr(agy_topics_mod, "get_thread_step_count", lambda *_a, **_kw: (True, 4, "IDLE"))
    sess_after_glm = prepare_topic_session("macro_sentiment", registry_path=str(reg_path))
    assert sess_after_glm.conversation_id == "cid-existing-macro"
    assert "via_glm_5_4" in sess_after_glm.history_prefix
    record_topic_turn(
        "macro_sentiment",
        "cid-existing-macro",
        '{"market_bias": "NEUTRAL", "summary": "agy_resumed_clean"}',
        registry_path=str(reg_path),
    )
    sess_after_agy = prepare_topic_session("macro_sentiment", registry_path=str(reg_path))
    assert sess_after_agy.conversation_id == "cid-existing-macro"
    assert sess_after_agy.history_prefix == ""

    # 3. auto_technical_repair (can_self_heal=True) is NEVER shifted to openai_compat even at 20% quota
    posted.clear()
    cli_calls.clear()
    res_repair = client.query_topic_json("auto_technical_repair", "Audit fleet")
    assert res_repair == {"market_bias": "NEUTRAL", "summary": "via_agy_fallback"}
    assert len(posted) == 0
    assert len(cli_calls) == 1

    # 4. Reversible safety net: when Z.ai fails transiently on allow_cli_fallback=True (executor soft-conserve)
    #    OR on stateless topic soft-conserve, GeminiClient falls back cleanly to agy CLI without double-hitting Z.ai
    oai_should_fail["flag"] = True
    posted.clear()
    cli_calls.clear()
    res_rev_exec = client.query_topic_json(
        "auto_cryptotrade_executor",
        "Evaluate BTC",
        model="openai_compat:glm-5.4",
        allow_cli_fallback=True,
    )
    assert res_rev_exec == {"market_bias": "NEUTRAL", "summary": "via_agy_fallback"}
    assert [p[1]["model"] for p in posted] == ["glm-5.4", "glm-5.3"]
    assert cli_calls == ["gemini-4-neon-medium"]

    posted.clear()
    cli_calls.clear()
    res_rev_macro = client.query_topic_json("macro_sentiment", "Evaluate macro during Z.ai outage")
    assert res_rev_macro == {"market_bias": "NEUTRAL", "summary": "via_agy_fallback"}
    assert [p[1]["model"] for p in posted] == ["glm-5.4", "glm-5.3"]
    assert cli_calls == ["gemini-4-neon-high"]

    # 5. Explicit A/B override (allow_cli_fallback=False) still does NOT fall back to agy CLI
    posted.clear()
    cli_calls.clear()
    res_strict = client.query_json("Direct A/B test", model="openai_compat:glm-5.4")
    assert res_strict is None
    assert len(cli_calls) == 0









