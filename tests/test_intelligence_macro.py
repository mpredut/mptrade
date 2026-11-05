"""Unit tests for Macro Geopolitical & Energy Shock Shield (Pillar Macro) and MacroAnalyzer."""
from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

from intelligence.composite import CompositeMarketIntelligence
from intelligence.internal.guards.guard_decision import BrakeAction
from intelligence.internal.triggers.trigger_event import TriggerSide
from intelligence.macro.geopolitical_analyzer import (
    GeopoliticalThreatAnalyzer,
    GeopoliticalThreatAssessment,
)
from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
from intelligence.macro.news_feed_collector import (
    NewsFeedCollector,
    NewsFeedSnapshot,
)
from intelligence.macro_analyzer import MacroAnalyzer
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.sentiment_advisor import SentimentAdvisorAssessment


SAMPLE_RSS_XML_WAR = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Breaking: Major military strike hits energy pipeline in Middle East</title>
      <link>https://news.example.com/item1</link>
      <pubDate>Sun, 04 Oct 2026 10:00:00 GMT</pubDate>
      <source>Reuters</source>
    </item>
    <item>
      <title>Oil embargo declared amid escalating war in critical shipping corridor</title>
      <link>https://news.example.com/item2</link>
      <pubDate>Sun, 04 Oct 2026 09:30:00 GMT</pubDate>
      <source>Bloomberg</source>
    </item>
    <item>
      <title>Strait of Hormuz tensions spike after naval clash</title>
      <link>https://news.example.com/item3</link>
      <pubDate>Sun, 04 Oct 2026 09:00:00 GMT</pubDate>
      <source>Al Jazeera</source>
    </item>
    <item>
      <title>Central banks meet to discuss ordinary quarterly rate path</title>
      <link>https://news.example.com/item4</link>
      <pubDate>Sun, 04 Oct 2026 08:00:00 GMT</pubDate>
      <source>Financial Times</source>
    </item>
  </channel>
</rss>
"""

SAMPLE_RSS_XML_CALM = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Local tech conference announces speaker lineup for next month</title>
      <link>https://news.example.com/item10</link>
      <pubDate>Sun, 04 Oct 2026 08:00:00 GMT</pubDate>
      <source>TechDaily</source>
    </item>
    <item>
      <title>Quarterly cloud software adoption rises across Europe</title>
      <link>https://news.example.com/item11</link>
      <pubDate>Sun, 04 Oct 2026 07:30:00 GMT</pubDate>
      <source>CloudReport</source>
    </item>
  </channel>
</rss>
"""


class TestNewsFeedCollector:
    """Tests for RSS news feed parser and severity keyword filter."""

    def test_parse_rss_xml_with_war_headlines(self):
        snapshot = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_WAR)
        assert snapshot is not None
        assert snapshot.total_fetched == 4
        assert snapshot.high_severity_count == 3
        assert snapshot.has_critical_shock_keywords is True

        h1 = snapshot.headlines[0]
        assert "military strike" in h1.matched_keywords or "war" in h1.matched_keywords
        assert h1.is_high_severity is True

    def test_parse_rss_xml_calm_headlines(self):
        snapshot = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_CALM)
        assert snapshot is not None
        assert snapshot.total_fetched == 2
        assert snapshot.high_severity_count == 0
        assert snapshot.has_critical_shock_keywords is False

    def test_caching(self):
        collector = NewsFeedCollector(cache_ttl_sec=300.0)
        dummy_snapshot = NewsFeedSnapshot(
            headlines=(),
            total_fetched=0,
            high_severity_count=0,
            has_critical_shock_keywords=False,
            ts=time.time(),
        )
        collector._cached_snapshot = dummy_snapshot
        collector._last_fetch_ts = time.time()
        assert collector.fetch(force_refresh=False) == dummy_snapshot

    def test_news_feed_collector_aggregation(self, tmp_path):
        history_file = str(tmp_path / "news_history.json")
        collector = NewsFeedCollector(
            cache_ttl_sec=0.0,
            history_file=history_file,
            rolling_window_sec=7200.0,
        )

        snap1 = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_CALM, now_ts=1000.0)
        agg1 = collector._aggregate(snap1, now=1000.0)
        assert agg1.total_fetched == 2

        snap2 = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_WAR, now_ts=1060.0)
        agg2 = collector._aggregate(snap2, now=1060.0)
        assert agg2.total_fetched == 6
        assert agg2.high_severity_count == 3

        agg3 = collector._aggregate(snap2, now=1120.0)
        assert agg3.total_fetched == 6


class TestGeopoliticalThreatAnalyzer:
    """Tests for GeopoliticalThreatAnalyzer two-stage filter and Gemini synthesis."""

    def test_calm_news_skips_llm(self):
        def failing_runner(prompt, model, timeout):
            raise AssertionError("LLM should not be called when news is calm")

        client = GeminiClient(custom_runner=failing_runner)
        analyzer = GeopoliticalThreatAnalyzer(gemini_client=client)

        calm_snapshot = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_CALM)
        assessment = analyzer.assess(calm_snapshot, force_refresh=True)

        assert assessment.threat_level == "NORMAL"
        assert assessment.recommended_brake == "NONE"
        assert assessment.risk_score <= 0.2

    def test_war_news_calls_llm(self, tmp_path):
        def mock_llm_runner(prompt, model, timeout):
            assert "Strait of Hormuz" in prompt
            return json.dumps({
                "threat_level": "CRITICAL_SHOCK",
                "risk_score": 0.95,
                "summary": "Imminent oil supply cut and naval escalation threaten severe risk-off crypto dump.",
                "recommended_brake": "HARD_VETO_NEW_BUYS",
            })

        client = GeminiClient(custom_runner=mock_llm_runner)
        state_file = str(tmp_path / "geopolitical_state.json")
        analyzer = GeopoliticalThreatAnalyzer(gemini_client=client, state_file=state_file)

        war_snapshot = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_WAR)
        assessment = analyzer.assess(war_snapshot, force_refresh=True)

        assert assessment.threat_level == "CRITICAL_SHOCK"
        assert assessment.recommended_brake == "HARD_VETO_NEW_BUYS"
        assert assessment.risk_score == 0.95
        assert "oil supply cut" in assessment.summary

    def test_macro_llm_interval_caching(self, tmp_path, monkeypatch):
        calls = []

        def counting_runner(prompt, model, timeout):
            calls.append(prompt)
            return json.dumps({
                "threat_level": "ELEVATED",
                "risk_score": 0.6,
                "summary": "Regional tensions require defensive sizing.",
                "recommended_brake": "DOWNSCALE_50",
            })

        client = GeminiClient(custom_runner=counting_runner)
        state_file = str(tmp_path / "geopolitical_state.json")
        analyzer = GeopoliticalThreatAnalyzer(
            gemini_client=client,
            cache_ttl_sec=7200.0,
            state_file=state_file,
        )

        war_snapshot = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_WAR)
        t0 = 1000000.0

        res1 = analyzer.assess(war_snapshot, now_ts=t0)
        assert len(calls) == 1
        assert res1.threat_level == "ELEVATED"

        res2 = analyzer.assess(war_snapshot, now_ts=t0 + 3600.0)
        assert len(calls) == 1
        assert res2 == res1

        res3 = analyzer.assess(war_snapshot, now_ts=t0 + 7300.0)
        assert len(calls) == 1
        assert res3 == res1

        war_xml_2 = SAMPLE_RSS_XML_WAR.replace(b"Major military strike", b"Second military strike")
        war_snap_2 = NewsFeedCollector.parse_rss_xml(war_xml_2)
        analyzer.assess(war_snap_2, now_ts=t0 + 7300.0)
        assert len(calls) == 2

    def test_llm_timeout_fallback_uses_short_retry_ttl_and_avoids_false_critical_shock(self, tmp_path):
        calls = []

        def timeout_then_recover(prompt, model, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                return None  # Simulate CLI timeout
            return json.dumps({
                "threat_level": "NORMAL",
                "risk_score": 0.20,
                "summary": "Background conflict coverage without acute macro escalation.",
                "recommended_brake": "NONE",
            })

        generic_war_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item><title>Commentary on ongoing war and sanctions debate</title></item>
    <item><title>Analysts discuss military strike history and conflict</title></item>
    <item><title>Parliament reviews war budget and sanctions package</title></item>
  </channel>
</rss>
"""
        snap = NewsFeedCollector.parse_rss_xml(generic_war_xml)
        assert snap.high_severity_count == 3

        client = GeminiClient(custom_runner=timeout_then_recover)
        state_file = str(tmp_path / "geopolitical_state.json")
        analyzer = GeopoliticalThreatAnalyzer(
            gemini_client=client,
            cache_ttl_sec=16200.0,
            fallback_retry_ttl_sec=120.0,
            state_file=state_file,
        )

        t0 = 2_000_000.0
        res1 = analyzer.assess(snap, now_ts=t0)
        assert len(calls) == 1
        assert calls[0] == 120.0
        # Generic war mentions without acute shock phrases must not trigger CRITICAL_SHOCK
        assert res1.threat_level == "ELEVATED"
        assert res1.recommended_brake == "DOWNSCALE_50"
        assert res1.is_fallback is True

        # Within 120s fallback retry window, cached fallback is returned
        res_cached = analyzer.assess(snap, now_ts=t0 + 60.0)
        assert len(calls) == 1
        assert res_cached == res1

        # After 120s (well before the 16200s normal TTL), LLM is retried even on identical headlines
        res2 = analyzer.assess(snap, now_ts=t0 + 125.0)
        assert len(calls) == 2
        assert res2.threat_level == "NORMAL"
        assert res2.recommended_brake == "NONE"
        assert res2.is_fallback is False

        # Subsequent poll on the same headlines with NORMAL cached assessment must NOT re-trigger LLM
        res3 = analyzer.assess(snap, now_ts=t0 + 130.0)
        assert len(calls) == 2
        assert res3 == res2



class TestGeopoliticalShockGuard:
    """Tests for GeopoliticalShockGuard emergency braking."""

    def test_critical_shock_vetoes_buy(self):
        guard = GeopoliticalShockGuard()
        assessment = GeopoliticalThreatAssessment(
            threat_level="CRITICAL_SHOCK",
            risk_score=0.95,
            summary="War outbreak in shipping corridor",
            recommended_brake="HARD_VETO_NEW_BUYS",
            headlines_analyzed=5,
            ts=time.time(),
        )

        dec = guard.check("BTCUSDT", "BUY", assessment)
        assert dec.allowed is False
        assert dec.brake_action == BrakeAction.HARD_VETO
        assert "veto_critical_geopolitical_shock" in dec.reason

    def test_elevated_shock_downscales_buy(self):
        guard = GeopoliticalShockGuard(downscale_factor=0.50)
        assessment = GeopoliticalThreatAssessment(
            threat_level="ELEVATED",
            risk_score=0.60,
            summary="Rising regional border tensions",
            recommended_brake="DOWNSCALE_50",
            headlines_analyzed=3,
            ts=time.time(),
        )

        dec = guard.check("BTCUSDT", "BUY", assessment)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.DOWNSCALE_QTY
        assert dec.suggested_scale == 0.50

    def test_normal_climate_allows_buy(self):
        guard = GeopoliticalShockGuard()
        assessment = GeopoliticalThreatAssessment(
            threat_level="NORMAL",
            risk_score=0.10,
            summary="Quiet global macro climate",
            recommended_brake="NONE",
            headlines_analyzed=10,
            ts=time.time(),
        )

        dec = guard.check("BTCUSDT", "BUY", assessment)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.NONE

    def test_sell_always_allowed(self):
        guard = GeopoliticalShockGuard()
        assessment = GeopoliticalThreatAssessment(
            threat_level="CRITICAL_SHOCK",
            risk_score=0.99,
            summary="Full scale war",
            recommended_brake="HARD_VETO_NEW_BUYS",
            headlines_analyzed=15,
            ts=time.time(),
        )

        dec = guard.check("BTCUSDT", "SELL", assessment)
        assert dec.allowed is True
        assert dec.brake_action == BrakeAction.NONE


class TestCompositeWithGeopolitical:
    """Integration test verifying CompositeMarketIntelligence respects GeopoliticalShockGuard."""

    def test_composite_geopolitical_veto(self):
        composite = CompositeMarketIntelligence()
        critical_assessment = GeopoliticalThreatAssessment(
            threat_level="CRITICAL_SHOCK",
            risk_score=0.92,
            summary="Major energy infrastructure strike",
            recommended_brake="HARD_VETO_NEW_BUYS",
            headlines_analyzed=4,
            ts=time.time(),
        )

        evaluation = composite.evaluate(
            symbol="BTCUSDT",
            price=65000.0,
            gradient=0.04,
            epsilon=0.01,
            geopolitical_assessment=critical_assessment,
        )

        assert evaluation.active_trigger is not None
        assert evaluation.active_trigger.side == TriggerSide.BUY
        assert evaluation.guard_decision.allowed is False
        assert evaluation.guard_decision.brake_action == BrakeAction.HARD_VETO
        assert evaluation.can_execute is False
        assert evaluation.metrics["geopolitical_threat"] == "CRITICAL_SHOCK"


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def test_macro_analyzer_initialization(temp_cache_dir):
    analyzer = MacroAnalyzer(
        cache_dir=temp_cache_dir,
        macro_llm_interval_sec=3600.0,
        advisor_interval_sec=1800.0,
        thread_prune_interval_sec=7200.0,
    )
    assert analyzer.macro_llm_interval_sec == 3600.0
    assert analyzer.advisor_interval_sec == 1800.0
    assert analyzer.thread_prune_interval_sec == 7200.0


def test_macro_analyzer_run_cycle_mocked(temp_cache_dir):
    analyzer = MacroAnalyzer(
        cache_dir=temp_cache_dir,
        macro_llm_interval_sec=3600.0,
        advisor_interval_sec=1800.0,
    )

    mock_geo = GeopoliticalThreatAssessment(
        threat_level="NORMAL",
        risk_score=0.1,
        summary="Calm macroeconomic backdrop.",
        recommended_brake="NONE",
        headlines_analyzed=5,
        ts=time.time(),
    )
    mock_advisor = SentimentAdvisorAssessment(
        market_bias="NEUTRAL",
        risk_level="MODERATE",
        confidence=0.7,
        summary="Market consolidating.",
        key_risks=["volatility"],
        recommended_action="HOLD",
        ts=time.time(),
    )

    with patch.object(analyzer.geopolitical_analyzer, "assess", return_value=mock_geo), \
         patch.object(analyzer.sentiment_advisor, "review", return_value=mock_advisor), \
         patch.object(analyzer, "prune_cli_threads", return_value=3):

        res = analyzer.run_cycle(force=True)

        assert res["geopolitical_updated"] is True
        assert res["advisor_updated"] is True
        assert res["threads_pruned"] == 3

        hb_path = os.path.join(temp_cache_dir, "macro_analyzer.heartbeat")
        assert os.path.exists(hb_path)
        with open(hb_path, "r", encoding="utf-8") as f:
            hb = json.load(f)
        assert hb["status"] == "alive"
        assert hb["last_run"]["geopolitical_updated"] is True


def test_macro_analyzer_default_3h_interval(temp_cache_dir, monkeypatch):
    import sys
    from intelligence import macro_analyzer

    analyzer = MacroAnalyzer(cache_dir=temp_cache_dir)
    assert analyzer.advisor_interval_sec == 10800.0

    captured = {}

    class DummyAnalyzer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_cycle(self, force=False):
            pass

    monkeypatch.setattr(macro_analyzer, "MacroAnalyzer", DummyAnalyzer)
    monkeypatch.setattr(sys, "argv", ["macro_analyzer.py", "--once"])
    macro_analyzer.main()
    assert captured["advisor_interval_sec"] is None
    assert captured["macro_llm_interval_sec"] is None


def test_macro_advisor_notification_dispatch(temp_cache_dir):
    analyzer = MacroAnalyzer(cache_dir=temp_cache_dir, advisor_interval_sec=10800.0)
    analyzer.shadow_notify = True

    mock_advisor = SentimentAdvisorAssessment(
        market_bias="BULLISH",
        risk_level="LOW",
        confidence=0.85,
        summary="Positive macro momentum and accumulation.",
        key_risks=[],
        recommended_action="ACCUMULATE",
        ts=time.time(),
    )

    with patch.object(analyzer.sentiment_advisor, "review", return_value=mock_advisor), \
         patch("notify_engine.alertnotifiers.notify") as mock_notify:
        ok = analyzer.assess_market_advisor(force=True)
        assert ok is True
        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs.get("source") == "macro_shadow"
        assert kwargs.get("title") == "🟢 [auto_macro_sentiment] ACCUMULATE (BULLISH)"
        assert kwargs.get("symbol") == "GLOBAL"


def test_macro_geopolitical_notification_dispatch(temp_cache_dir):
    analyzer = MacroAnalyzer(cache_dir=temp_cache_dir, macro_llm_interval_sec=16200.0)
    analyzer.shadow_notify = True

    mock_geo = GeopoliticalThreatAssessment(
        threat_level="ELEVATED",
        risk_score=0.65,
        summary="Regional shipping tensions elevated.",
        recommended_brake="DOWNSCALE_50",
        headlines_analyzed=4,
        ts=time.time(),
        is_fallback=False,
    )

    with patch.object(analyzer.geopolitical_analyzer, "assess", return_value=mock_geo), \
         patch("notify_engine.alertnotifiers.notify") as mock_notify:
        ok = analyzer.assess_geopolitical_threat(force=True)
        assert ok is True
        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs.get("source") == "macro_shadow"
        assert kwargs.get("title") == "🛡 [auto_macro_geopolitical] DOWNSCALE_50 (ELEVATED)"
        assert kwargs.get("symbol") == "GLOBAL"


def test_macro_advisor_skips_redundant_cached_notifications(temp_cache_dir):
    analyzer = MacroAnalyzer(cache_dir=temp_cache_dir, advisor_interval_sec=10800.0)
    analyzer.shadow_notify = True
    now = time.time()
    mock_advisor = SentimentAdvisorAssessment(
        market_bias="CAUTION",
        risk_level="MODERATE",
        confidence=0.75,
        summary="Market consolidating.",
        key_risks=["risk"],
        recommended_action="HOLD",
        ts=now,
    )

    with patch.object(analyzer.sentiment_advisor, "review", return_value=mock_advisor), \
         patch("notify_engine.alertnotifiers.notify") as mock_notify:
        ok = analyzer.assess_market_advisor(force=True)
        assert ok is True
        assert mock_notify.call_count == 1

        ok2 = analyzer.assess_market_advisor(force=True)
        assert ok2 is True
        assert mock_notify.call_count == 1


class TestMacroSentimentSSOTAndDynamicIntervals:
    """Tests for GlobalSessionContext, session checkpoints, dynamic interval scaling, and SSOT normalizers."""

    def test_ssot_normalizers(self):
        from intelligence.macro_sentiment_ssot import (
            normalize_market_bias,
            normalize_recommended_action,
            normalize_recommended_brake,
            normalize_risk_level,
            normalize_threat_level,
        )

        assert normalize_market_bias("risk_on") == "BULLISH"
        assert normalize_market_bias("defensive") == "CAUTION"
        assert normalize_market_bias("risk_off") == "BEARISH"
        assert normalize_risk_level("elevated") == "HIGH"
        assert normalize_risk_level("severe") == "EXTREME"
        assert normalize_recommended_action("buy") == "ACCUMULATE"
        assert normalize_recommended_action("take_profit") == "TRIM_PROFITS"
        assert normalize_threat_level("critical") == "CRITICAL_SHOCK"
        assert normalize_recommended_brake("HARD_VETO", threat_level="CRITICAL_SHOCK") == "HARD_VETO_NEW_BUYS"
        assert normalize_recommended_brake("DOWNSCALE_QTY", threat_level="ELEVATED") == "DOWNSCALE_50"
        assert normalize_recommended_brake("ALLOW", threat_level="NORMAL") == "NONE"

    def test_global_trading_sessions_europe_morning_buy_and_evening_close(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from intelligence.macro_sentiment_ssot import (
            did_cross_session_checkpoint,
            resolve_global_trading_session,
        )

        tz_eet = ZoneInfo("Europe/Bucharest")

        # Tuesday 09:00 EET -> EU_OPEN_ACCUMULATION (European morning buy/accumulation)
        t_eu_morning = datetime(2026, 10, 27, 9, 0, tzinfo=tz_eet).timestamp()
        ctx_morning = resolve_global_trading_session(t_eu_morning)
        assert ctx_morning.session_key == "EU_OPEN_ACCUMULATION"
        assert ctx_morning.is_active_western_session is True
        assert ctx_morning.is_peak_transition_window is True
        assert "accumulate" in ctx_morning.trader_flow_profile.lower()

        # Tuesday 16:45 EET -> US_OPEN_OVERLAP (EU afternoon + Wall Street open)
        t_us_open = datetime(2026, 10, 27, 16, 45, tzinfo=tz_eet).timestamp()
        ctx_overlap = resolve_global_trading_session(t_us_open)
        assert ctx_overlap.session_key == "US_OPEN_OVERLAP"
        assert ctx_overlap.is_peak_transition_window is True

        # Tuesday 19:00 EET -> EU_EVENING_CLOSE (European traders closing positions / profit-taking)
        t_eu_evening = datetime(2026, 10, 27, 19, 0, tzinfo=tz_eet).timestamp()
        ctx_evening = resolve_global_trading_session(t_eu_evening)
        assert ctx_evening.session_key == "EU_EVENING_CLOSE"
        assert ctx_evening.is_active_western_session is True
        assert ctx_evening.is_peak_transition_window is True
        assert "close intraday positions" in ctx_evening.trader_flow_profile.lower()

        # Wednesday 03:30 EET -> ASIA_OVERNIGHT_CALM
        t_asia = datetime(2026, 10, 28, 3, 30, tzinfo=tz_eet).timestamp()
        ctx_asia = resolve_global_trading_session(t_asia)
        assert ctx_asia.session_key == "ASIA_OVERNIGHT_CALM"
        assert ctx_asia.is_active_western_session is False

        # Checkpoints: 07:30 -> 08:20 EET crosses EU_MORNING_BUY_OPEN; 17:30 -> 18:35 EET crosses EU_EVENING_POSITION_CLOSE
        t_0730 = datetime(2026, 10, 27, 7, 30, tzinfo=tz_eet).timestamp()
        t_0820 = datetime(2026, 10, 27, 8, 20, tzinfo=tz_eet).timestamp()
        assert did_cross_session_checkpoint(t_0730, t_0820) == "EU_MORNING_BUY_OPEN"

        t_1730 = datetime(2026, 10, 27, 17, 30, tzinfo=tz_eet).timestamp()
        t_1835 = datetime(2026, 10, 27, 18, 35, tzinfo=tz_eet).timestamp()
        assert did_cross_session_checkpoint(t_1730, t_1835) == "EU_EVENING_POSITION_CLOSE"

    def test_dynamic_sentiment_interval_adapts_to_context(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from intelligence.macro_sentiment_ssot import compute_dynamic_sentiment_interval
        from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
        from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

        tz_eet = ZoneInfo("Europe/Bucharest")
        base = 10800.0  # 3h

        # 1. Global calm overnight (02:00 EET, neutral breadth & F&G) -> stretches to 14400s (4.0h)
        t_night = datetime(2026, 10, 27, 2, 0, tzinfo=tz_eet).timestamp()
        calm_fg = FearGreedSnapshot(
            value=50,
            classification="Neutral",
            timestamp=t_night,
            historical_values=(50, 49, 51),
            trend_7d_change=0,
            is_extreme_fear=False,
            is_extreme_greed=False,
        )
        calm_br = MarketBreadthSnapshot(
            advance_ratio=0.52,
            advancing_count=26,
            declining_count=24,
            total_count=50,
            median_change_pct=0.4,
            mean_change_pct=0.5,
            dispersion_std=1.2,
            regime="NEUTRAL",
            top_gainers=(),
            top_losers=(),
            ts=t_night,
        )
        itv_calm, reason_calm = compute_dynamic_sentiment_interval(
            base_interval_sec=base,
            last_eval_ts=t_night - 3600.0,
            now_ts=t_night,
            fear_greed=calm_fg,
            breadth=calm_br,
        )
        assert itv_calm == 14400.0
        assert "global_calm_quiet" in reason_calm

        # 2. European morning buy window (09:30 EET, after 08:15 checkpoint) -> 5400s (1.5h)
        t_eu_buy = datetime(2026, 10, 27, 9, 30, tzinfo=tz_eet).timestamp()
        itv_eu, reason_eu = compute_dynamic_sentiment_interval(
            base_interval_sec=base,
            last_eval_ts=t_eu_buy - 1800.0,
            now_ts=t_eu_buy,
            fear_greed=calm_fg,
            breadth=calm_br,
        )
        assert itv_eu == 5400.0
        assert "EU_OPEN_ACCUMULATION" in reason_eu

        # 3. Crossing European evening close checkpoint (17:40 -> 18:35 EET) -> 2700s (45m)
        t_1740 = datetime(2026, 10, 27, 17, 40, tzinfo=tz_eet).timestamp()
        t_1835 = datetime(2026, 10, 27, 18, 35, tzinfo=tz_eet).timestamp()
        itv_cp, reason_cp = compute_dynamic_sentiment_interval(
            base_interval_sec=base,
            last_eval_ts=t_1740,
            now_ts=t_1835,
            fear_greed=calm_fg,
            breadth=calm_br,
        )
        assert itv_cp == 2700.0
        assert reason_cp == "session_checkpoint_EU_EVENING_POSITION_CLOSE"

        # 4. Rapid F&G jump (>= 8 pts) or PANIC_WASHOUT -> 1800s (30m)
        itv_jump, reason_jump = compute_dynamic_sentiment_interval(
            base_interval_sec=base,
            last_eval_ts=t_night - 2000.0,
            now_ts=t_night,
            fear_greed=calm_fg,
            breadth=calm_br,
            prev_fear_greed_val=38,
        )
        assert itv_jump == 1800.0
        assert "fear_greed_jump" in reason_jump

        # 5. Critical 15:30-18:30 EET US/EU overlap window -> 1080s (18m)
        t_overlap = datetime(2026, 10, 27, 17, 0, tzinfo=tz_eet).timestamp()
        itv_ovl, reason_ovl = compute_dynamic_sentiment_interval(
            base_interval_sec=base,
            last_eval_ts=t_overlap - 1200.0,
            now_ts=t_overlap,
            fear_greed=calm_fg,
            breadth=calm_br,
            geopolitical_threat_level="ELEVATED",
        )
        assert itv_ovl == 1080.0
        assert "critical_us_eu_overlap_window" in reason_ovl

    def test_elevated_threat_escalates_immediately_on_new_acute_shock_headline(self, tmp_path):
        calls = []

        def mock_runner(prompt, model, timeout):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps({
                    "threat_level": "ELEVATED",
                    "risk_score": 0.65,
                    "summary": "Sanctions announced.",
                    "recommended_brake": "DOWNSCALE_50",
                })
            return json.dumps({
                "threat_level": "CRITICAL_SHOCK",
                "risk_score": 0.96,
                "summary": "Strait of Hormuz blockade declared.",
                "recommended_brake": "HARD_VETO_NEW_BUYS",
            })

        elevated_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>News</title>
  <item><title>New sanctions package debated after border conflict</title></item>
</channel></rss>"""

        acute_escalation_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>News</title>
  <item><title>Breaking: Strait of Hormuz closed after declared war and nuclear strike threat</title></item>
  <item><title>New sanctions package debated after border conflict</title></item>
</channel></rss>"""

        client = GeminiClient(custom_runner=mock_runner)
        analyzer = GeopoliticalThreatAnalyzer(
            gemini_client=client,
            cache_ttl_sec=16200.0,
            state_file=str(tmp_path / "geo_escalate.json"),
        )
        t0 = 1_700_000_000.0
        snap1 = NewsFeedCollector.parse_rss_xml(elevated_xml, now_ts=t0)
        res1 = analyzer.assess(snap1, now_ts=t0)
        assert res1.threat_level == "ELEVATED"
        assert len(calls) == 1

        # Only 10 minutes later (t0 + 600s << 16200s TTL), an acute shock headline arrives:
        # must bypass TTL immediately and escalate to CRITICAL_SHOCK!
        snap2 = NewsFeedCollector.parse_rss_xml(acute_escalation_xml, now_ts=t0 + 600.0)
        res2 = analyzer.assess(snap2, now_ts=t0 + 600.0)
        assert len(calls) == 2
        assert res2.threat_level == "CRITICAL_SHOCK"
        assert res2.recommended_brake == "HARD_VETO_NEW_BUYS"

    def test_macro_intelligence_conf_and_dynamic_lexicon_discovery(self, tmp_path):
        from intelligence.macro_sentiment_ssot import (
            build_dynamic_rss_url,
            discover_dynamic_terms_from_headlines,
            get_active_acute_shock_phrases,
            get_active_severity_keywords,
            get_breadth_excluded_bases,
            get_breadth_quote_suffixes,
            get_cache_filenames,
            get_fear_greed_endpoint_url,
            get_market_breadth_endpoint_url,
            get_specialized_rss_urls,
            load_dynamic_lexicon,
            match_high_severity_keywords,
            record_dynamic_crisis_terms_from_assessment,
        )

        # 1. Config-driven endpoints, exclusions, and specialized feeds
        assert "alternative.me" in get_fear_greed_endpoint_url()
        assert "binance.com" in get_market_breadth_endpoint_url()
        assert "USDC" in get_breadth_excluded_bases()
        assert "USDT" in get_breadth_quote_suffixes()
        assert "sentiment_eval.json" in get_cache_filenames("sentiment_advisor_files")

        urls = get_specialized_rss_urls(cache_dir=str(tmp_path))
        assert len(urls) >= 5
        assert any("bbci.co.uk" in u for u in urls)
        assert any("aljazeera.com" in u for u in urls)
        assert any("feeds.finance.yahoo.com" in u and "NVDA" in u for u in urls)

        from intelligence.macro_sentiment_ssot import (
            build_equity_tickers_rss_url,
            has_tracked_equity_entity,
            is_excluded_reference_news_item,
        )

        eq_rss = build_equity_tickers_rss_url()
        assert "NVDA" in eq_rss and "SPCX" in eq_rss and "RGNT" in eq_rss
        assert has_tracked_equity_entity("Morgan Stanley Sees SpaceX Stock Hitting $300 as AI Forecasts Explode")
        assert has_tracked_equity_entity("AMD Faces Nvidia Software Hurdle, Morningstar Says")
        assert not has_tracked_equity_entity("2 Bank Stocks to Buy After Their Third-Quarter Earnings Reports")
        assert is_excluded_reference_news_item(
            "2 Bank Stocks to Buy After Their Third-Quarter Earnings Reports",
            "Yahoo! Finance: NVDA SPCX RGNT News",
            "https://finance.yahoo.com/news/2-bank-stocks",
        )
        assert not is_excluded_reference_news_item(
            "Morgan Stanley Sees SpaceX Stock Hitting $300 as AI Forecasts Explode",
            "Yahoo! Finance: NVDA SPCX RGNT News",
            "https://finance.yahoo.com/news/morgan-stanley-spacex",
        )

        # 2. Automatic discovery of emerging crisis bigrams from multi-source headlines
        multi_source_headlines = [
            {
                "title": "Urgent: Bosphorus tanker blockade halts regional shipping corridor",
                "source": "BBC World",
            },
            {
                "title": "Markets tumble after Bosphorus tanker crisis escalates into naval standoff",
                "source": "Al Jazeera",
            },
        ]
        discovered = discover_dynamic_terms_from_headlines(
            multi_source_headlines,
            cache_dir=str(tmp_path),
            now_ts=1_700_000_000.0,
            persist=True,
        )
        assert "bosphorus tanker" in discovered
        active_kws = get_active_severity_keywords(cache_dir=str(tmp_path), now_ts=1_700_000_100.0)
        assert "bosphorus tanker" in active_kws

        # Verify dynamic RSS search URL now incorporates the newly discovered term
        dyn_url = build_dynamic_rss_url(cache_dir=str(tmp_path), now_ts=1_700_000_100.0)
        assert "bosphorus" in dyn_url.lower()

        # 3. AI Assessment feedback loop persists critical crisis terms into dynamic lexicon
        recorded = record_dynamic_crisis_terms_from_assessment(
            "Emergency Panama canal drought embargo threatens global LNG supply chains.",
            ["Panama canal blockade", "LNG supply embargo"],
            is_acute_shock=True,
            cache_dir=str(tmp_path),
            now_ts=1_700_000_200.0,
        )
        assert len(recorded) > 0
        lex_state = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=1_700_000_200.0)
        assert len(lex_state["dynamic_severity_keywords"]) > 0
        assert len(lex_state["dynamic_acute_shock_phrases"]) > 0
        matched = match_high_severity_keywords(
            "Shipping rates spike on Panama canal blockade",
            cache_dir=str(tmp_path),
            now_ts=1_700_000_250.0,
        )
        assert len(matched) > 0
        assert len(get_active_acute_shock_phrases(cache_dir=str(tmp_path), now_ts=1_700_000_250.0)) > 0

        # 4. compute_dynamic_geopolitical_interval only returns 0.0s on ELEVATED when a NEW acute phrase arrives
        from intelligence.macro_sentiment_ssot import compute_dynamic_geopolitical_interval

        existing_digest = "strait of hormuz shipping update|fed inflation remarks"
        non_acute_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>News</title>
  <item><title>Strait of Hormuz shipping update</title></item>
  <item><title>Emergency sanctions tariff announcement</title></item>
</channel></rss>"""
        non_acute_change_snap = NewsFeedCollector.parse_rss_xml(non_acute_xml, now_ts=1_700_000_100.0)
        itv_no_new_acute, reason_no_new = compute_dynamic_geopolitical_interval(
            base_interval_sec=16200.0,
            last_eval_ts=1_700_000_000.0,
            now_ts=1_700_000_120.0,
            cached_assessment={"threat_level": "ELEVATED", "is_fallback": False},
            news_snapshot=non_acute_change_snap,
            headlines_changed=True,
            last_headlines_digest=existing_digest,
        )
        assert itv_no_new_acute == 3600.0
        assert reason_no_new == "elevated_threat_new_headlines_1h"

        # 5. NORMAL assessment on already-digested shock headlines must NOT loop at 0.0s
        itv_normal_same, reason_normal_same = compute_dynamic_geopolitical_interval(
            base_interval_sec=16200.0,
            last_eval_ts=1_700_000_000.0,
            now_ts=1_700_000_015.0,
            cached_assessment={"threat_level": "NORMAL", "is_fallback": False},
            news_snapshot=non_acute_change_snap,
            headlines_changed=False,
            last_headlines_digest=existing_digest,
        )
        assert itv_normal_same == 16200.0
        assert reason_normal_same == "standard_geopolitical_ttl_normal"

        # Whereas a brand-new acute phrase under NORMAL still triggers 0.0s immediately
        itv_normal_fresh, reason_normal_fresh = compute_dynamic_geopolitical_interval(
            base_interval_sec=16200.0,
            last_eval_ts=1_700_000_000.0,
            now_ts=1_700_000_015.0,
            cached_assessment={"threat_level": "NORMAL", "is_fallback": False},
            news_snapshot=non_acute_change_snap,
            headlines_changed=True,
            last_headlines_digest="fed inflation remarks",
        )
        assert itv_normal_fresh == 0.0
        assert reason_normal_fresh == "acute_shock_keyword_alert"

    def test_dynamic_telemetry_interval_scaling_from_executor_cadence(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone
        from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
        from intelligence.macro_analyzer import MacroAnalyzer
        from intelligence.macro_sentiment_ssot import (
            compute_dynamic_geopolitical_interval,
            compute_dynamic_sentiment_interval,
            resolve_crypto_executor_interval_sec,
            resolve_derived_telemetry_intervals,
        )

        # 1. Baseline 6-minute (360s) cadence
        base_itvs = resolve_derived_telemetry_intervals(executor_interval_sec=360.0)
        assert base_itvs["crypto_executor_interval_sec"] == 360.0
        assert base_itvs["share_executor_interval_sec"] == 1800.0
        assert base_itvs["ui_sync_interval_sec"] == 720.0
        assert base_itvs["ui_sync_idle_sec"] == 120.0
        assert base_itvs["orderbook_interval_sec"] == 15.0
        assert base_itvs["derivatives_interval_sec"] == 30.0
        assert base_itvs["whale_interval_sec"] == 60.0
        assert base_itvs["news_interval_sec"] == 120.0
        assert base_itvs["breadth_interval_sec"] == 300.0
        assert base_itvs["macro_benchmarks_interval_sec"] == 300.0
        assert base_itvs["fear_greed_interval_sec"] == 1800.0
        assert base_itvs["sensor_stale_age_min"] == 5.5
        assert base_itvs["unfilled_order_cancel_grace_sec"] == 600.0
        assert base_itvs["peak_overlap_sentiment_interval_sec"] == 1080.0
        assert base_itvs["default_sentiment_interval_sec"] == 10800.0
        assert base_itvs["default_geopolitical_interval_sec"] == 16200.0

        # 2. Automatic propagation when crontab changes from */6 to */10 (10m = 600s)
        cron_10m = tmp_path / "crontab.prod.txt"
        cron_10m.write_text(
            "*/10 * * * * cd $HOME/mptrade && ./orchestratorOS/run_python.sh "
            "orchestratorOS/supervisor/autonomous_ai_reconciler.py --topic-key auto_cryptotrade_executor\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MPTRADE_CRONTAB_PATH", str(cron_10m))
        assert resolve_crypto_executor_interval_sec() == 600.0

        scaled_itvs = resolve_derived_telemetry_intervals()
        assert scaled_itvs["crypto_executor_interval_sec"] == 600.0
        assert scaled_itvs["share_executor_interval_sec"] == 3000.0
        assert scaled_itvs["ui_sync_interval_sec"] == 1200.0
        assert scaled_itvs["ui_sync_idle_sec"] == 200.0
        assert scaled_itvs["orderbook_interval_sec"] == 25.0
        assert scaled_itvs["derivatives_interval_sec"] == 50.0
        assert scaled_itvs["whale_interval_sec"] == 100.0
        assert scaled_itvs["news_interval_sec"] == 200.0
        assert scaled_itvs["breadth_interval_sec"] == 500.0
        assert scaled_itvs["macro_benchmarks_interval_sec"] == 500.0
        assert scaled_itvs["fear_greed_interval_sec"] == 3000.0
        assert round(scaled_itvs["sensor_stale_age_min"], 2) == 9.17
        assert scaled_itvs["unfilled_order_cancel_grace_sec"] == 1000.0
        assert scaled_itvs["peak_overlap_sentiment_interval_sec"] == 1800.0
        assert scaled_itvs["min_dynamic_sentiment_interval_sec"] == 3000.0
        assert scaled_itvs["default_sentiment_interval_sec"] == 18000.0
        assert scaled_itvs["default_geopolitical_interval_sec"] == 27000.0

        # 3. GlobalTelemetryCollector and MacroAnalyzer automatically inherit the scaled 10m intervals
        gtc_10m = GlobalTelemetryCollector(symbols=["BTCUSDC"], cache_dir=str(tmp_path), skip_news=True, skip_external=True)
        assert gtc_10m.orderbook_interval_sec == 25.0
        assert gtc_10m.derivatives_interval_sec == 50.0
        assert gtc_10m.whale_interval_sec == 100.0
        assert gtc_10m.news_interval_sec == 200.0
        assert gtc_10m.breadth_interval_sec == 500.0
        assert gtc_10m.macro_benchmarks_interval_sec == 500.0
        assert gtc_10m.fear_greed_interval_sec == 3000.0

        ma_10m = MacroAnalyzer(cache_dir=str(tmp_path), symbols=["BTCUSDC"])
        assert ma_10m.advisor_interval_sec == 18000.0
        assert ma_10m.macro_llm_interval_sec == 27000.0

        # Peak overlap window (16:00 EET = 14:00 UTC) scales from 1080s (18m) to 1800s (30m)
        ts_1600_eet = datetime(2026, 2, 11, 14, 0, 0, tzinfo=timezone.utc).timestamp()
        sent_itv, sent_reason = compute_dynamic_sentiment_interval(now_ts=ts_1600_eet)
        assert sent_itv == 1800.0
        assert "critical_us_eu_overlap_window" in sent_reason

        geo_itv, geo_reason = compute_dynamic_geopolitical_interval(now_ts=ts_1600_eet)
        assert geo_reason == "global_daytime_calm"
        assert geo_itv == pytest.approx(min(scaled_itvs["max_dynamic_geopolitical_interval_sec"], 27000.0 * 1.15))

    def test_freeze_stem_excludes_freezers_and_normal_assessment_demotes_dynamic_acute_terms(self, tmp_path):
        from intelligence.macro_sentiment_ssot import (
            demote_dynamic_acute_terms,
            discover_dynamic_terms_from_headlines,
            get_active_acute_shock_phrases,
            load_dynamic_lexicon,
            match_high_severity_keywords,
            update_dynamic_lexicon,
        )

        # 1. "freezers" must not match the "freeze" crisis stem unless an actual crisis word is in the bigram
        freezer_headlines = [
            {"title": "Shop freezers in seven European countries hit by outage", "source": "BBC"},
            {"title": "Supermarket fridges and freezers in seven regions affected", "source": "Reuters"},
        ]
        discovered_freezers = discover_dynamic_terms_from_headlines(
            freezer_headlines,
            cache_dir=str(tmp_path),
            now_ts=1_700_000_000.0,
            persist=True,
        )
        assert "freezers seven" not in discovered_freezers
        assert "supermarket fridges" not in discovered_freezers
        assert match_high_severity_keywords("Shop freezers in seven countries", cache_dir=str(tmp_path)) == ()

        # 2. Dynamic "acute" terms are demoted to "high" once LLM confirms threat_level != CRITICAL_SHOCK
        update_dynamic_lexicon(
            ["iran war"],
            new_acute_terms=["iran war"],
            cache_dir=str(tmp_path),
            source="llm_assessment_feedback",
            now_ts=1_700_000_000.0,
        )
        assert "iran war" in get_active_acute_shock_phrases(cache_dir=str(tmp_path), now_ts=1_700_000_010.0)

        def normal_llm_runner(prompt, model, timeout):
            return json.dumps({
                "threat_level": "NORMAL",
                "risk_score": 0.25,
                "summary": "Ongoing rhetoric without fresh systemic escalation.",
                "recommended_brake": "NONE",
            })

        analyzer = GeopoliticalThreatAnalyzer(
            gemini_client=GeminiClient(custom_runner=normal_llm_runner),
            cache_ttl_sec=16200.0,
            state_file=str(tmp_path / "geopolitical_eval.json"),
        )
        war_snap = NewsFeedCollector.parse_rss_xml(SAMPLE_RSS_XML_WAR, now_ts=1_700_000_020.0, cache_dir=str(tmp_path))
        res = analyzer.assess(war_snap, force_refresh=True, now_ts=1_700_000_020.0)
        assert res.threat_level == "NORMAL"

        lex_after = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=1_700_000_030.0)
        assert "iran war" in lex_after["high_severity_terms"]
        assert "iran war" not in lex_after["acute_shock_phrases"]
        assert demote_dynamic_acute_terms(cache_dir=str(tmp_path), now_ts=1_700_000_030.0) is False

    def test_geopolitical_surge_counts_only_new_headlines_and_preserves_digest_until_stale(self, tmp_path):
        from intelligence.macro_sentiment_ssot import (
            compute_dynamic_geopolitical_interval,
            extract_high_severity_headlines_and_digest,
        )

        # Simulate a 6h rolling pool with high_severity_count=79 where 11 of top 12 were already in last_headlines_digest
        old_titles = [f"Tariff trade negotiations round {i}" for i in range(11)]
        prev_digest = "|".join(sorted(t.lower() for t in old_titles))
        one_new_headlines = [
            {"title": "Central bank policy review scheduled", "source": "Reuters", "is_high_severity": True},
            *[{"title": t, "source": "Reuters", "is_high_severity": True} for t in old_titles],
        ]
        snap_one_new = {
            "headlines": one_new_headlines,
            "high_severity_count": 79,
            "has_critical_shock_keywords": False,
        }
        itv_one_new, reason_one_new = compute_dynamic_geopolitical_interval(
            base_interval_sec=16200.0,
            last_eval_ts=1_700_000_000.0,
            now_ts=1_700_000_600.0,
            cached_assessment={"threat_level": "NORMAL", "is_fallback": False},
            news_snapshot=snap_one_new,
            headlines_changed=True,
            last_headlines_digest=prev_digest,
        )
        assert itv_one_new == 16200.0
        assert reason_one_new == "standard_geopolitical_ttl_normal"

        # Whereas 3+ genuinely new high-severity headlines in top 12 trigger high_severity_headline_surge_3
        three_new_headlines = [
            {"title": "Sanctions package announced on oil exports", "source": "Reuters", "is_high_severity": True},
            {"title": "Military drills near shipping lane", "source": "BBC", "is_high_severity": True},
            {"title": "Tariff hike on industrial imports", "source": "CNBC", "is_high_severity": True},
            *[{"title": t, "source": "Reuters", "is_high_severity": True} for t in old_titles[:9]],
        ]
        snap_three_new = {
            "headlines": three_new_headlines,
            "high_severity_count": 81,
            "has_critical_shock_keywords": False,
        }
        itv_three_new, reason_three_new = compute_dynamic_geopolitical_interval(
            base_interval_sec=16200.0,
            last_eval_ts=1_700_000_000.0,
            now_ts=1_700_000_600.0,
            cached_assessment={"threat_level": "NORMAL", "is_fallback": False},
            news_snapshot=snap_three_new,
            headlines_changed=True,
            last_headlines_digest=prev_digest,
        )
        assert itv_three_new == 2700.0
        assert reason_three_new == "high_severity_headline_surge_3"

        # GeopoliticalThreatAnalyzer.assess must not overwrite _last_headlines_digest while cache is still fresh
        calls = []

        def counting_runner(prompt, model, timeout):
            calls.append(prompt)
            return json.dumps({
                "threat_level": "NORMAL",
                "risk_score": 0.15,
                "summary": "Routine macro headlines.",
                "recommended_brake": "NONE",
            })

        analyzer = GeopoliticalThreatAnalyzer(
            gemini_client=GeminiClient(custom_runner=counting_runner),
            cache_ttl_sec=16200.0,
            state_file=str(tmp_path / "geo_digest_preserve.json"),
        )
        snap_initial = {"headlines": [{"title": t, "source": "Reuters", "is_high_severity": True} for t in old_titles[:4]]}
        _, initial_digest = extract_high_severity_headlines_and_digest(snap_initial)
        analyzer.assess(snap_initial, now_ts=1_700_000_000.0)
        assert len(calls) == 1
        assert analyzer._last_headlines_digest == initial_digest

        # Call while not stale with modified headlines: must return cached assessment WITHOUT overwriting _last_headlines_digest
        analyzer.assess(snap_one_new, now_ts=1_700_000_600.0)
        assert len(calls) == 1
        assert analyzer._last_headlines_digest == initial_digest
        # Once stale (after 16200s), the new digest must be evaluated by LLM
        analyzer.assess(snap_one_new, now_ts=1_700_017_000.0)
        assert len(calls) == 2

    def test_news_feed_collector_no_resurrection_after_rolling_window_and_pubdate_filtering(self, tmp_path):
        import email.utils
        from intelligence.macro_sentiment_ssot import discover_dynamic_terms_from_headlines

        history_file = str(tmp_path / "news_resurrect.json")
        collector = NewsFeedCollector(
            cache_ttl_sec=0.0,
            history_file=history_file,
            rolling_window_sec=21600.0,
        )
        t0 = 1_793_660_000.0
        fresh_pub = email.utils.formatdate(t0 - 120.0, usegmt=True)
        stale_25h_pub = email.utils.formatdate(t0 - 90_000.0, usegmt=True)
        rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>News</title>
  <item>
    <title>Breaking naval clash near shipping lane</title>
    <pubDate>{fresh_pub}</pubDate>
    <source>BBC</source>
  </item>
  <item>
    <title>Yesterday war commentary from archives</title>
    <pubDate>{stale_25h_pub}</pubDate>
    <source>France 24</source>
  </item>
</channel></rss>""".encode("utf-8")

        snap = NewsFeedCollector.parse_rss_xml(rss_xml, now_ts=t0, cache_dir=str(tmp_path))
        agg1 = collector._aggregate(snap, now=t0)
        # Only the fresh headline (< 6h old) enters the rolling pool; the 25h-old pubDate item is excluded
        assert agg1.total_fetched == 1
        assert agg1.headlines[0].title == "Breaking naval clash near shipping lane"

        # 6h + 5m later, the first headline ages out and must NOT be resurrected when the RSS feed still serves it
        agg_expired = collector._aggregate(snap, now=t0 + 21900.0)
        assert agg_expired.total_fetched == 0
        agg_next_poll = collector._aggregate(snap, now=t0 + 22020.0)
        assert agg_next_poll.total_fetched == 0

        # Single-source revised headlines must not discover non-crisis neighbor bigrams like "worse chip"
        single_source_revisions = [
            {"title": "The Intel stock collapse is getting worse and the $3.5 trillion chip rout is spreading - Yahoo Finance", "source": "Yahoo Finance"},
            {"title": "The Intel stock collapse is getting worse and the $3.7 trillion chip rout is spreading - Yahoo Finance", "source": "Yahoo Finance"},
        ]
        discovered_single = discover_dynamic_terms_from_headlines(
            single_source_revisions,
            cache_dir=str(tmp_path),
            now_ts=t0,
            persist=False,
        )
        assert "worse chip" not in discovered_single
        assert "stock collapse" in discovered_single

        # Archival RSS items older than 7 days (e.g., months-old Google News background articles)
        # must not return None from _parse_recent_pub_date_ts, must not crowd out fresh items in parse_rss_xml,
        # and must not pollute dynamic lexicon with editorial words like "war timeline" or "war exposes".
        from intelligence.macro_sentiment_ssot import load_dynamic_lexicon, update_dynamic_lexicon

        archival_items = "\n".join(
            f"""  <item>
    <title>Iran war timeline and exposes #{idx} - CNN</title>
    <pubDate>Fri, 15 May 2026 07:00:00 GMT</pubDate>
    <source>Source{idx}</source>
  </item>"""
            for idx in range(30)
        )
        mixed_archival_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>News</title>
{archival_items}
  <item>
    <title>Red Sea missile attack disrupts container shipping</title>
    <pubDate>{fresh_pub}</pubDate>
    <source>Reuters</source>
  </item>
</channel></rss>""".encode("utf-8")

        snap_mixed = NewsFeedCollector.parse_rss_xml(mixed_archival_xml, now_ts=t0, cache_dir=str(tmp_path))
        assert snap_mixed is not None
        assert len(snap_mixed.headlines) == 1
        assert snap_mixed.headlines[0].title == "Red Sea missile attack disrupts container shipping"

        update_dynamic_lexicon(
            [
                "older blockade", "war timeline", "war exposes", "crash ahead",
                "attack debunked", "unpriced shock", "train crash", "jet crash",
                "florida jet", "alert apps", "worse chip",
            ],
            cache_dir=str(tmp_path),
            now_ts=t0 - 3600.0,
        )
        update_dynamic_lexicon(
            ["newest embargo"],
            cache_dir=str(tmp_path),
            now_ts=t0,
        )
        lex = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=t0)
        high_terms = lex["high_severity_terms"]
        assert high_terms[0] == "newest embargo"
        assert "older blockade" in high_terms
        for noise_term in (
            "war timeline", "war exposes", "crash ahead", "attack debunked",
            "unpriced shock", "train crash", "jet crash", "florida jet",
            "alert apps", "worse chip",
        ):
            assert noise_term not in high_terms

        # Same-URL RSS headline revisions (e.g. BBC crime story edited from "baseball bat" to "bat")
        # must deduplicate in the rolling pool and must not self-promote local crime bigrams like
        # "allegedly attacking" or "attacking one".
        bbc_rev1_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>BBC News</title>
  <item>
    <title>Man and boy arrested after allegedly attacking One Nation volunteer with baseball bat in Australia</title>
    <link>https://www.bbc.co.uk/news/articles/cwly3jd27m9xo?at_medium=RSS&amp;at_campaign=rss</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>BBC News</source>
  </item>
</channel></rss>""".encode("utf-8")
        bbc_rev2_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>BBC News</title>
  <item>
    <title>Man and boy arrested after allegedly attacking One Nation volunteer with bat</title>
    <link>https://www.bbc.co.uk/news/articles/cwly3jd27m9xo?at_medium=RSS&amp;at_campaign=rss</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>BBC News</source>
  </item>
</channel></rss>""".encode("utf-8")

        url_collector = NewsFeedCollector(
            history_file=str(tmp_path / "url_dedup_history.json"),
            rolling_window_sec=21600.0,
        )
        snap_r1 = NewsFeedCollector.parse_rss_xml(bbc_rev1_xml, now_ts=t0, cache_dir=str(tmp_path))
        snap_r2 = NewsFeedCollector.parse_rss_xml(bbc_rev2_xml, now_ts=t0 + 60.0, cache_dir=str(tmp_path))
        url_collector._aggregate(snap_r1, now=t0)
        agg_r2 = url_collector._aggregate(snap_r2, now=t0 + 60.0)
        assert agg_r2.total_fetched == 1
        assert agg_r2.high_severity_count == 0
        assert agg_r2.headlines[0].title == "Man and boy arrested after allegedly attacking One Nation volunteer with bat"
        assert agg_r2.headlines[0].is_high_severity is False
        lex_after_bbc = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=t0 + 60.0)["high_severity_terms"]
        assert "allegedly attacking" not in lex_after_bbc
        assert "attacking one" not in lex_after_bbc

        # Re-ingesting the older revision (snap_r1) from a lagging/cached feed must NOT overwrite the newer revision
        agg_r1_replay = url_collector._aggregate(snap_r1, now=t0 + 120.0)
        assert agg_r1_replay.total_fetched == 1
        assert agg_r1_replay.headlines[0].title == "Man and boy arrested after allegedly attacking One Nation volunteer with bat"

        # Seed "attack" keyword must not trigger high-severity on local crime/assault headlines
        bbc_bat_attack_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>BBC News</title>
  <item>
    <title>Boy, 15, arrested after volunteer allegedly hit with baseball bat in attack</title>
    <link>https://www.bbc.co.uk/news/articles/cwly3jd27m9xo</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>BBC News</source>
  </item>
</channel></rss>""".encode("utf-8")
        snap_bat = NewsFeedCollector.parse_rss_xml(bbc_bat_attack_xml, now_ts=t0 + 180.0, cache_dir=str(tmp_path))
        assert snap_bat is not None
        assert snap_bat.high_severity_count == 0
        assert snap_bat.headlines[0].is_high_severity is False
        assert snap_bat.headlines[0].matched_keywords == ()

        # Syndicated wire headlines across multiple local stations (- WSOC TV, - WKMG, - KSAT) and
        # cross-feed duplicates ("Title - BBC" vs "Title") must deduplicate to a single headline,
        # prefer the clean un-suffixed title, and not self-promote reporting bigrams.
        syndicated_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
  <item>
    <title>US has conducted defensive strikes on some Iranian targets in the Strait of Hormuz, US official says - WSOC TV</title>
    <link>https://news.google.com/rss/articles/CBMi_wsoc</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>WSOC TV</source>
  </item>
  <item>
    <title>US has conducted defensive strikes on some Iranian targets in the Strait of Hormuz, US official says - WKMG</title>
    <link>https://news.google.com/rss/articles/CBMi_wkmg</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>WKMG</source>
  </item>
  <item>
    <title>US has conducted defensive strikes on some Iranian targets in the Strait of Hormuz, US official says - KSAT</title>
    <link>https://news.google.com/rss/articles/CBMi_ksat</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>KSAT</source>
  </item>
  <item>
    <title>Russian bombing kills 26 people at Ukraine aid centre, Kyiv says - BBC</title>
    <link>https://news.google.com/rss/articles/CBMi_bbc</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>BBC</source>
  </item>
  <item>
    <title>Russian bombing kills 26 people at Ukraine aid centre, Kyiv says</title>
    <link>https://www.bbc.com/news/articles/c23456</link>
    <pubDate>{fresh_pub}</pubDate>
    <source>BBC World</source>
  </item>
</channel></rss>""".encode("utf-8")

        snap_synd = NewsFeedCollector.parse_rss_xml(syndicated_xml, now_ts=t0 + 200.0, cache_dir=str(tmp_path))
        assert snap_synd is not None
        assert snap_synd.total_fetched == 2
        assert snap_synd.high_severity_count == 1
        synd_titles = [h.title for h in snap_synd.headlines]
        assert "Russian bombing kills 26 people at Ukraine aid centre, Kyiv says" in synd_titles
        assert "Russian bombing kills 26 people at Ukraine aid centre, Kyiv says - BBC" not in synd_titles

        synd_collector = NewsFeedCollector(
            history_file=str(tmp_path / "synd_dedup_history.json"),
            rolling_window_sec=21600.0,
        )
        agg_synd = synd_collector._aggregate(snap_synd, now=t0 + 200.0)
        assert agg_synd.total_fetched == 2
        assert agg_synd.high_severity_count == 1
        lex_after_synd = load_dynamic_lexicon(cache_dir=str(tmp_path), now_ts=t0 + 200.0)["high_severity_terms"]
        for noise_bg in (
            "conducts defensive",
            "conducted defensive",
            "carries defensive",
            "iranian targets",
            "war conducts",
            "reportedly strikes",
            "strikes reported",
            "media strikes",
        ):
            assert noise_bg not in lex_after_synd


def test_binance_account_stream_dynamic_recv_timeout_and_response_health(monkeypatch):
    import asyncio
    from binance_api import bapi_ws

    sent_frames = []
    recv_timeouts = []
    healthy_count = 0

    class FakeWS:
        def __init__(self):
            self.step = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, msg):
            sent_frames.append(json.loads(msg))

        async def recv(self):
            self.step += 1
            if self.step == 1:
                return json.dumps({"id": "login", "status": 200})
            if self.step == 2:
                return json.dumps({"id": "sub", "status": 200})
            stream._stop_event.set()
            raise asyncio.TimeoutError()

    orig_wait_for = asyncio.wait_for

    async def tracking_wait_for(coro, timeout=None):
        recv_timeouts.append(timeout)
        return await orig_wait_for(coro, timeout=timeout)

    monkeypatch.setattr(bapi_ws.websockets, "connect", lambda *a, **kw: FakeWS())
    monkeypatch.setattr(bapi_ws.asyncio, "wait_for", tracking_wait_for)

    def on_healthy():
        nonlocal healthy_count
        healthy_count += 1

    stream = bapi_ws.BinanceAccountStream(
        on_event=lambda e: None,
        on_healthy=on_healthy,
        on_unhealthy=lambda: None,
        on_available=lambda a: None,
        keepalive_sec=1800,
        loss_timeout_sec=40,
    )
    stream._signing_key = "dummy_key"
    monkeypatch.setattr(bapi_ws.u, "_sign_ed25519", lambda k, p: "sig")

    asyncio.run(stream._connect_and_run())

    # recv_timeouts[0] is 10s login timeout; recv_timeouts[1] and [2] are bounded by loss_timeout_sec / 2 (<= 20.0s, not 30.0s)
    assert recv_timeouts[0] == 10
    assert all(1.0 <= t <= 20.0 for t in recv_timeouts[1:])
    # Both initial subscribe and the 200 OK "sub" response mark the stream healthy
    assert healthy_count >= 2


def test_contiguous_dynamic_lexicon_runs_test_log_isolation_and_dotenv_dedup(tmp_path, monkeypatch):
    """Verify contiguous token run extraction in dynamic lexicon discovery, test-log isolation in
    harvest_log_anomalies_from_dirs / log.py, and duplicate load_dotenv log suppression in botcore.
    """
    import logging
    import botcore
    import log as log_mod
    from intelligence.macro_sentiment_ssot import (
        discover_dynamic_terms_from_headlines,
        record_dynamic_crisis_terms_from_assessment,
    )
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # 1. Cross-punctuation and cross-stopword synthetic bigrams are never discovered
    headlines = [
        {
            "title": "North Korea denies role in record Binance hack, calls US charge 'provocation' - DW.com",
            "source": "DW.com",
            "url": "https://dw.com/a1",
        },
        {
            "title": "North Korea denies involvement in $6.3bn Binance hack and calls FBI accusation 'political provocation' - AOL.com",
            "source": "AOL.com",
            "url": "https://aol.com/a2",
        },
        {
            "title": "US strikes Iran's Qeshm Island on eve of midterms - FT",
            "source": "FT",
            "url": "https://ft.com/b1",
        },
        {
            "title": "U.S. reportedly strikes Iran's Qeshm Island on eve of election - Guardian",
            "source": "Guardian",
            "url": "https://theguardian.com/b2",
        },
        {
            "title": "Stocks mostly down as AI concerns top plunge in oil prices - Yahoo",
            "source": "Yahoo",
            "url": "https://yahoo.com/c1",
        },
        {
            "title": "Markets slide after sudden plunge in oil prices - CNBC",
            "source": "CNBC",
            "url": "https://cnbc.com/c2",
        },
    ]
    discovered = discover_dynamic_terms_from_headlines(
        headlines,
        cache_dir=str(tmp_path),
        now_ts=1_793_686_000.0,
        persist=True,
    )
    assert "binance hack" in discovered
    assert "strikes iran" in discovered
    for noise_bg in ("hack calls", "iran eve", "iran qeshm", "qeshm island", "oil prices"):
        assert noise_bg not in discovered

    rec = record_dynamic_crisis_terms_from_assessment(
        "Exchange was hacked. Emergency measures followed; hacks in Uganda and shock. Emergency sanction discussed.",
        ["Binance hack", "hacks, Ukraine"],
        cache_dir=str(tmp_path),
        now_ts=1_793_686_010.0,
    )
    assert "binance hack" in rec
    for cross_punct in ("hacked emergency", "hacks uganda", "shock emergency", "emergency sanction", "hacks ukraine"):
        assert cross_punct not in rec

    # 2. harvest_log_anomalies_from_dirs ignores pytest_*, __main___*, py.test_*, and c_* logs
    log_dir = tmp_path / "logger"
    log_dir.mkdir()
    (log_dir / "pytest_2026-11-03.log").write_text(
        "08:00:00 [GUARD] BUY BTCUSDC vetoed by unit test\n08:00:01 CRITICAL simulated test failure\n",
        encoding="utf-8",
    )
    (log_dir / "__main___2026-11-03.log").write_text(
        "08:00:02 [LLM GUARD] SELL ETHUSDC vetoed in test\n08:00:03 ERROR simulated main test\n",
        encoding="utf-8",
    )
    (log_dir / "c_2026-11-03.log").write_text(
        "08:00:04 Traceback (most recent call last): ad-hoc python -c\n",
        encoding="utf-8",
    )
    (log_dir / "rtrade_2026-11-03.log").write_text(
        "08:00:05 [GUARD] BUY TAOUSDC: real guard intervention\n",
        encoding="utf-8",
    )
    summary = harvest_log_anomalies_from_dirs([str(log_dir)], lookback_hours=24.0)
    assert summary.total_scanned_files == 1
    assert summary.error_count == 0
    assert summary.warning_count == 1
    assert len(summary.guard_interventions) == 1
    assert "TAOUSDC" in summary.guard_interventions[0]

    # Verify _DailyFileHandler suppresses writes to default _DEFAULT_LOG_FOLDER under pytest
    handler = log_mod._DailyFileHandler("pytest")
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "unit test msg", (), None)
    old_folder = log_mod._log_folder
    try:
        log_mod._log_folder = os.path.normpath(os.path.abspath(log_mod._DEFAULT_LOG_FOLDER))
        handler.emit(record)
        assert handler._stream is None
    finally:
        log_mod._log_folder = old_folder
        handler.close()

    # 3. botcore.load_dotenv logs '.env loaded from ...' only once per resolved path unless override=True
    env_file = tmp_path / "dedup_test.env"
    env_file.write_text("DEDUP_TEST_KEY=1\n", encoding="utf-8")
    logged_msgs = []
    monkeypatch.setattr(botcore, "log", lambda m: logged_msgs.append(m))
    monkeypatch.delenv("DEDUP_TEST_KEY", raising=False)
    botcore._LOADED_DOTENV_PATHS.discard(os.path.abspath(str(env_file)))

    botcore.load_dotenv(str(env_file))
    botcore.load_dotenv(str(env_file))
    assert len(logged_msgs) == 1
    botcore.load_dotenv(str(env_file), override=True)
    assert len(logged_msgs) == 2


def test_cross_asset_regime_vix_and_symmetric_ai_telemetry(tmp_path):
    """Verify VIX benchmark spec, symmetric cross_asset_regime classification, and unified Crypto/Equity AI telemetry."""
    import json
    from types import SimpleNamespace
    from intelligence.macro_sentiment_ssot import (
        CROSS_ASSET_REGIMES,
        get_macro_benchmark_specs,
        get_yahoo_chart_url,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        MACRO_BENCHMARK_SPECS,
        compute_cross_asset_regime,
        compute_equity_macro_regime,
        format_equity_macro_regime_line,
        format_macro_benchmarks_summary_line,
    )
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from orchestratorOS.supervisor.prompt_sections import _format_macro_and_news_section

    # 1. Declarative SSOT config includes VIX (^VIX) and cross_asset_regimes
    specs = get_macro_benchmark_specs()
    assert specs == MACRO_BENCHMARK_SPECS
    assert ("VIX", "^VIX", "index") in specs
    assert "%5EVIX" in get_yahoo_chart_url("^VIX")
    assert set(CROSS_ASSET_REGIMES) == {
        "COUPLED_RISK_ON",
        "COUPLED_RISK_OFF",
        "CRYPTO_BULLISH_DECOUPLING",
        "EQUITY_BULLISH_DECOUPLING",
        "NEUTRAL",
    }

    # 2. VIX surge triggers RISK_OFF and blocks false RISK_ON
    assert (
        compute_equity_macro_regime(
            {
                "items": {
                    "SPY": {"price": 580.0, "chg_24h_pct": -0.10},
                    "VIX": {"price": 24.5, "chg_24h_pct": 11.2},
                }
            }
        )
        == "RISK_OFF"
    )
    assert (
        compute_equity_macro_regime(
            {
                "items": {
                    "SPY": {"price": 580.0, "chg_24h_pct": 0.40},
                    "VIX": {"price": 24.5, "chg_24h_pct": 9.5},
                }
            }
        )
        == "NEUTRAL"
    )

    # 3. Symmetric Cross-Asset Regime classification (COUPLED_RISK_ON/OFF, CRYPTO/EQUITY_BULLISH_DECOUPLING)
    assert (
        compute_cross_asset_regime(
            {
                "items": {
                    "SPY": {"price": 585.0, "chg_24h_pct": 0.65},
                    "QQQ": {"price": 505.0, "chg_24h_pct": 0.90},
                    "IBIT": {"price": 46.2, "chg_24h_pct": 2.40},
                },
                "coinbase_premium_bps": 7.5,
            }
        )
        == "COUPLED_RISK_ON"
    )
    assert (
        compute_cross_asset_regime(
            {
                "items": {
                    "SPY": {"price": 575.0, "chg_24h_pct": -0.80},
                    "QQQ": {"price": 495.0, "chg_24h_pct": -1.10},
                    "IBIT": {"price": 43.1, "chg_24h_pct": -2.10},
                },
                "coinbase_premium_bps": -8.2,
            }
        )
        == "COUPLED_RISK_OFF"
    )
    decouple_crypto = {
        "items": {
            "SPY": {"price": 576.0, "chg_24h_pct": -0.75},
            "QQQ": {"price": 496.0, "chg_24h_pct": -0.95},
            "VIX": {"price": 21.4, "chg_24h_pct": 6.5, "unit": "index"},
            "IBIT": {"price": 46.8, "chg_24h_pct": 1.85},
        },
        "coinbase_premium_bps": 9.4,
        "coinbase_premium_regime": "US_INSTITUTIONAL_BUYING",
        "btc_dvol": 51.2,
        "options_vol_regime": "NORMAL_IV",
        "btc_rv_pct": 44.0,
        "btc_vrp_pct": 7.2,
        "btc_pc_oi_ratio": 0.48,
        "btc_skew_25d_pct": -1.25,
    }
    assert compute_cross_asset_regime(decouple_crypto) == "CRYPTO_BULLISH_DECOUPLING"
    assert compute_cross_asset_regime(decouple_crypto, btc_chg_24h_pct=0.0) == "CRYPTO_BULLISH_DECOUPLING"
    assert (
        compute_cross_asset_regime(
            {
                "items": {
                    "SPY": {"price": 586.0, "chg_24h_pct": 0.70},
                    "QQQ": {"price": 508.0, "chg_24h_pct": 1.05},
                    "IBIT": {"price": 43.5, "chg_24h_pct": -1.40},
                },
                "coinbase_premium_bps": 0.0,
            },
            btc_chg_24h_pct=0.0,
        )
        == "EQUITY_BULLISH_DECOUPLING"
    )
    assert (
        compute_cross_asset_regime(
            {
                "items": {
                    "SPY": {"price": 586.0, "chg_24h_pct": 0.70},
                    "QQQ": {"price": 508.0, "chg_24h_pct": 1.05},
                    "IBIT": {"price": 43.5, "chg_24h_pct": -1.40},
                },
                "coinbase_premium_bps": -6.8,
            }
        )
        == "EQUITY_BULLISH_DECOUPLING"
    )

    # 4. Unified summary & symmetric visibility across PreTrade Telemetry, SentimentAdvisor, and Prompt Sections
    summary_line = format_macro_benchmarks_summary_line(decouple_crypto)
    assert "VIX=21.40 (+6.50% 24h)" in summary_line
    assert "IBIT=$46.80 (+1.85% 24h)" in summary_line
    assert "CB_Premium=+9.4bps (US_INSTITUTIONAL_BUYING)" in summary_line
    assert "BTC_DVOL=51.2% (NORMAL_IV)" in summary_line
    assert "Options_Surface(RV=44.0%, VRP=+7.2%, PC_OI=0.48, Skew_25d=-1.25%)" in summary_line
    assert "Cross_Asset=CRYPTO_BULLISH_DECOUPLING" in summary_line

    eq_macro_line = format_equity_macro_regime_line(decouple_crypto)
    assert eq_macro_line.startswith("RISK_OFF (")
    assert "IBIT=$46.80" in eq_macro_line
    assert "CB_Premium=+9.4bps" in eq_macro_line
    assert "Cross_Asset=CRYPTO_BULLISH_DECOUPLING" in eq_macro_line

    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "macro_benchmarks_collect.json").write_text(json.dumps(decouple_crypto), encoding="utf-8")

    # Both Crypto and Equities pre-trade telemetry receive the full Equity & Macro Regime line
    pt_crypto = collect_pretrade_telemetry("BTCUSDC", cachedb_dir=str(cdir))
    pt_equity = collect_pretrade_telemetry("NVDA_US_EQ", cachedb_dir=str(cdir))
    assert pt_crypto.get("Equity & Macro Regime") == eq_macro_line
    assert pt_equity.get("Equity & Macro Regime") == eq_macro_line

    # Both Crypto and Equities prompt sections receive the full cross-asset benchmark line
    sensors_ns = SimpleNamespace(
        fear_greed={"value": 48, "classification": "Neutral"},
        market_breadth={"advance_ratio": 0.52, "regime": "NEUTRAL"},
        sentiment_advisor={"market_bias": "NEUTRAL", "risk_level": "MODERATE", "recommended_action": "HOLD"},
        geopolitical_threat={"threat_level": "NORMAL", "risk_score": 0.1, "recommended_brake": "NONE"},
        macro_benchmarks=decouple_crypto,
        breaking_news=[],
    )
    sec1_eq = "\n".join(_format_macro_and_news_section(sensors_ns, asset_scope="equities"))
    sec1_cr = "\n".join(_format_macro_and_news_section(sensors_ns, asset_scope="crypto"))
    for sec_txt in (sec1_eq, sec1_cr):
        assert "VIX=21.40" in sec_txt
        assert "IBIT=$46.80" in sec_txt
        assert "CB_Premium=+9.4bps" in sec_txt
        assert "BTC_DVOL=51.2%" in sec_txt
        assert "Options_Surface(" in sec_txt
        assert "Cross_Asset=CRYPTO_BULLISH_DECOUPLING" in sec_txt

    # SentimentAdvisor.review injects Cross-Asset Macro & Options Surface into the LLM prompt
    captured_sa_prompts = []

    def _sa_runner(prompt, model, timeout):
        captured_sa_prompts.append(prompt)
        return json.dumps(
            {
                "market_bias": "BULLISH",
                "risk_level": "MODERATE",
                "confidence": 0.82,
                "summary": "Institutional US spot accumulation decoupling from equity pullback.",
                "key_risks": ["Equity index volatility"],
                "recommended_action": "ACCUMULATE",
            }
        )

    sa = SentimentAdvisor(
        gemini_client=GeminiClient(custom_runner=_sa_runner),
        cache_file=str(cdir / "sentiment_eval.json"),
        cache_ttl_sec=300.0,
    )
    sa_res = sa.review(force_refresh=True, now_ts=1_793_860_000.0)
    assert sa_res is not None and sa_res.market_bias == "BULLISH"
    assert len(captured_sa_prompts) == 1
    assert "- Cross-Asset Macro & Options Surface:" in captured_sa_prompts[0]
    assert "Cross_Asset=CRYPTO_BULLISH_DECOUPLING" in captured_sa_prompts[0]

    # 5. Dynamic Sentiment Interval & Stale SentimentAdvisor Live Institutional Divergence (BUY & SELL symmetry)
    from intelligence.macro.calendar_ssot import compute_dynamic_sentiment_interval
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        enrich_sentiment_advisor_live_flow_divergence,
    )
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

    # Use a timestamp outside 15:30-18:30 EET overlap so institutional divergence rule is evaluated
    non_overlap_ts = 1_793_840_000.0
    bear_cached = SimpleNamespace(
        market_bias="BEARISH",
        risk_level="MODERATE",
        recommended_action="DEFENSIVE",
        summary="Prior defensive posture.",
        key_risks=[],
    )
    itv_buy_div, rsn_buy_div = compute_dynamic_sentiment_interval(
        base_interval_sec=10800.0,
        last_eval_ts=non_overlap_ts - 1800.0,
        now_ts=non_overlap_ts,
        cached_assessment=bear_cached,
        btc_taker_ratio=1.05,
        macro_benchmarks=decouple_crypto,
    )
    assert itv_buy_div <= 300.0
    assert "institutional_buy_divergence_crypto_bullish_decoupling_+9.4bps" in rsn_buy_div

    bull_cached = SimpleNamespace(
        market_bias="BULLISH",
        risk_level="MODERATE",
        recommended_action="ACCUMULATE",
        summary="Prior accumulate posture.",
        key_risks=[],
    )
    decouple_equity = {
        "cross_asset_regime": "EQUITY_BULLISH_DECOUPLING",
        "coinbase_premium_bps": -7.5,
        "IBIT": {"price": 43.5, "chg_1h_pct": -0.95, "chg_24h_pct": -1.40},
    }
    itv_sell_div, rsn_sell_div = compute_dynamic_sentiment_interval(
        base_interval_sec=10800.0,
        last_eval_ts=non_overlap_ts - 1800.0,
        now_ts=non_overlap_ts,
        cached_assessment=bull_cached,
        btc_taker_ratio=0.98,
        macro_benchmarks=decouple_equity,
    )
    assert itv_sell_div <= 300.0
    assert "institutional_sell_divergence_equity_bullish_decoupling_-7.5bps" in rsn_sell_div

    sa_stale_bear = {
        "age_min": 22.0,
        "recommended_action": "DEFENSIVE",
        "market_bias": "BEARISH",
        "summary": "Bearish macro posture",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_stale_bear,
        {"BTCUSDC": {"taker_buy_sell_ratio": 1.02}},
        macro_benchmarks=decouple_crypto,
    )
    assert "LIVE_INSTITUTIONAL_BUY_DECOUPLING" in str(sa_stale_bear.get("live_flow_divergence"))
    assert "Cross_Asset=CRYPTO_BULLISH_DECOUPLING" in str(sa_stale_bear.get("live_flow_divergence"))

    sa_stale_bull = {
        "age_min": 22.0,
        "recommended_action": "ACCUMULATE",
        "market_bias": "BULLISH",
        "summary": "Bullish macro posture",
    }
    enrich_sentiment_advisor_live_flow_divergence(
        sa_stale_bull,
        {"BTCUSDC": {"taker_buy_sell_ratio": 0.98}},
        macro_benchmarks=decouple_equity,
    )
    assert "LIVE_INSTITUTIONAL_SELL_DECOUPLING" in str(sa_stale_bull.get("live_flow_divergence"))
    assert "Cross_Asset=EQUITY_BULLISH_DECOUPLING" in str(sa_stale_bull.get("live_flow_divergence"))

    # 6. Pre-Trade Guards (HighStakeGuard BUY/SELL & PostStopReentryGuard) & Executor Prompt Policy Symmetry
    hsg = HighStakeGuard(gemini_client=GeminiClient(custom_runner=_sa_runner))
    hsg_buy_prompt = hsg._build_high_stake_prompt("BUY", 1200.0, f"- Equity & Macro Regime: {eq_macro_line}")
    hsg_sell_prompt = hsg._build_high_stake_prompt("SELL", 1200.0, f"- Equity & Macro Regime: {eq_macro_line}")
    psg_prompt = PostStopReentryGuard._build_post_stop_prompt("BTCUSDC", f"- Equity & Macro Regime: {eq_macro_line}")
    for gp in (hsg_buy_prompt, hsg_sell_prompt, psg_prompt):
        assert "CROSS-ASSET MACRO, VIX & INSTITUTIONAL FLOW" in gp
        assert "CRYPTO_BULLISH_DECOUPLING" in gp
        assert "EQUITY_BULLISH_DECOUPLING" in gp
        assert "CB_Premium" in gp
        assert "Options_Surface" in gp

    sup_cr = build_supervisor_prompt(sensors_ns, SimpleNamespace(), topic_key="auto_cryptotrade_executor")
    sup_eq = build_supervisor_prompt(sensors_ns, SimpleNamespace(), topic_key="auto_sharetrade_executor")
    for sp in (sup_cr, sup_eq):
        assert "CROSS-ASSET MACRO, VIX & INSTITUTIONAL CRYPTO/OPTIONS SURFACE SYMMETRY" in sp
        assert "CRYPTO_BULLISH_DECOUPLING" in sp
        assert "EQUITY_BULLISH_DECOUPLING" in sp













