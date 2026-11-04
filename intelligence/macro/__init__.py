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

from botcore import bind_lazy_module_exports

_LAZY_IMPORTS: dict[str, str] = {
    "NewsFeedCollector": "intelligence.macro.news_feed_collector",
    "NewsFeedSnapshot": "intelligence.macro.news_feed_collector",
    "NewsHeadline": "intelligence.macro.news_feed_collector",
    "GeopoliticalThreatAnalyzer": "intelligence.macro.geopolitical_analyzer",
    "GeopoliticalThreatAssessment": "intelligence.macro.geopolitical_analyzer",
    "GeopoliticalShockGuard": "intelligence.macro.geopolitical_guard",
}

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS)
