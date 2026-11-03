"""Macro intelligence: Geopolitical, War, and Energy Crisis Black Swan Shield.

Separated cleanly from internal price mechanics and exchange flow telemetry:
- news_feed_collector: Fetches unauthenticated public RSS headlines regarding war and energy crises.
- geopolitical_analyzer: Two-stage filter (keyword screening + Google Gemini LLM synthesis).
- geopolitical_guard: Emergency brake vetoing BUYs during systemic macro shocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.macro.geopolitical_analyzer import (
        GeopoliticalThreatAnalyzer,
        GeopoliticalThreatAssessment,
    )
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
    from intelligence.macro.news_feed_collector import (
        NewsFeedCollector,
        NewsFeedSnapshot,
        NewsHeadline,
    )

__all__ = [
    "NewsFeedCollector",
    "NewsFeedSnapshot",
    "NewsHeadline",
    "GeopoliticalThreatAnalyzer",
    "GeopoliticalThreatAssessment",
    "GeopoliticalShockGuard",
]


def __getattr__(name: str) -> Any:
    if name in ("NewsFeedCollector", "NewsFeedSnapshot", "NewsHeadline"):
        from intelligence.macro import news_feed_collector as _nfc

        return getattr(_nfc, name)
    if name in ("GeopoliticalThreatAnalyzer", "GeopoliticalThreatAssessment"):
        from intelligence.macro import geopolitical_analyzer as _ga

        return getattr(_ga, name)
    if name == "GeopoliticalShockGuard":
        from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard

        return GeopoliticalShockGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
