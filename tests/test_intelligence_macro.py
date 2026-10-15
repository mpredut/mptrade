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


def test_macro_analyzer_default_3h_interval(temp_cache_dir):
    analyzer = MacroAnalyzer(cache_dir=temp_cache_dir)
    assert analyzer.advisor_interval_sec == 10800.0


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
        assert "BULLISH" in kwargs.get("title")
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
