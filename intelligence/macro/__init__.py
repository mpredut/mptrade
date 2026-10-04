"""Macro intelligence: Geopolitical, War, and Energy Crisis Black Swan Shield.

Separated cleanly from internal price mechanics and exchange flow telemetry:
- news_feed_collector: Fetches unauthenticated public RSS headlines regarding war and energy crises.
- geopolitical_analyzer: Two-stage filter (keyword screening + Google Gemini LLM synthesis).
- geopolitical_guard: Emergency brake vetoing BUYs during systemic macro shocks.
"""

from __future__ import annotations

from intelligence.macro.news_feed_collector import (
    NewsFeedCollector,
    NewsFeedSnapshot,
    NewsHeadline,
)
from intelligence.macro.geopolitical_analyzer import (
    GeopoliticalThreatAnalyzer,
    GeopoliticalThreatAssessment,
)
from intelligence.macro.geopolitical_guard import (
    GeopoliticalShockGuard,
)

__all__ = [
    "NewsFeedCollector",
    "NewsFeedSnapshot",
    "NewsHeadline",
    "GeopoliticalThreatAnalyzer",
    "GeopoliticalThreatAssessment",
    "GeopoliticalShockGuard",
]
