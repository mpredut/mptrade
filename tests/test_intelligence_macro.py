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
        assert calls[0] == 25.0
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
        assert "sentiment_advisor_eval.json" in get_cache_filenames("sentiment_advisor_files")

        urls = get_specialized_rss_urls(cache_dir=str(tmp_path))
        assert len(urls) >= 5
        assert any("bbci.co.uk" in u for u in urls)
        assert any("aljazeera.com" in u for u in urls)

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


