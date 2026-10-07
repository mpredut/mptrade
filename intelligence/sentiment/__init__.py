"""Market sentiment intelligence (Pillar 3: Fear & Greed, Market Breadth, Macro Dispersion, LLM Reasoning).

Integrates sentiment telemetry, contrarian drivers, euphoric/panic brakes, and High-Stake Guard:
- collectors:
  * FearGreedCollector (Alternative.me Crypto Fear & Greed Index with 14d trend)
  * MarketBreadthCollector (Binance 24h market-wide advance/decline and dispersion)
- advisor:
  * SentimentAdvisor (Periodic macro market synthesis via LLM)
- triggers:
  * SentimentContrarianTrigger (Extreme fear dip-buying and extreme greed distribution)
  * MarketBreadthTrigger (Capitulation washout bounces and blowoff exhaustion trimming)
- guards:
  * ExtremeGreedGuard (Anti-FOMO / top-buying brake at euphoric sentiment peaks)
  * PanicWashoutGuard (Anti-falling-knife protection during market-wide crashes)
  * HighStakeGuard (LLM pre-flight risk vetting for purchases >= 1000 EUR)
"""

from __future__ import annotations

from intelligence.sentiment.gemini_client import (
    GeminiClient,
)
from intelligence.sentiment.collectors.fear_greed_collector import (
    FearGreedCollector,
    FearGreedSnapshot,
)
from intelligence.sentiment.collectors.market_breadth_collector import (
    MarketBreadthCollector,
    MarketBreadthSnapshot,
)
from intelligence.sentiment.sentiment_advisor import (
    SentimentAdvisor,
    SentimentAdvisorAssessment,
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
from intelligence.sentiment.guards.high_stake_guard import (
    HighStakeGuard,
)

__all__ = [
    "GeminiClient",
    "FearGreedCollector",
    "FearGreedSnapshot",
    "MarketBreadthCollector",
    "MarketBreadthSnapshot",
    "SentimentAdvisor",
    "SentimentAdvisorAssessment",
    "SentimentContrarianTrigger",
    "MarketBreadthTrigger",
    "ExtremeGreedGuard",
    "PanicWashoutGuard",
    "HighStakeGuard",
]
