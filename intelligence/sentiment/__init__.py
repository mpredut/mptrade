"""Market sentiment intelligence (Pillar 3: Fear & Greed, Market Breadth, Macro Dispersion).

Integrates sentiment telemetry, contrarian drivers, and euphoric/panic brakes:
- collectors:
  * FearGreedCollector (Alternative.me Crypto Fear & Greed Index with 14d trend)
  * MarketBreadthCollector (Binance 24h market-wide advance/decline and dispersion)
- triggers:
  * SentimentContrarianTrigger (Extreme fear dip-buying and extreme greed distribution)
  * MarketBreadthTrigger (Capitulation washout bounces and blowoff exhaustion trimming)
- guards:
  * ExtremeGreedGuard (Anti-FOMO / top-buying brake at euphoric sentiment peaks)
  * PanicWashoutGuard (Anti-falling-knife protection during market-wide crashes)
"""

from __future__ import annotations

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

__all__ = [
    "FearGreedCollector",
    "FearGreedSnapshot",
    "MarketBreadthCollector",
    "MarketBreadthSnapshot",
    "SentimentContrarianTrigger",
    "MarketBreadthTrigger",
    "ExtremeGreedGuard",
    "PanicWashoutGuard",
]
