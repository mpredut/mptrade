"""Market sentiment intelligence (Pillar 3: Fear & Greed, Market Breadth, Macro Dispersion, LLM Reasoning).

Integrates sentiment telemetry, contrarian drivers, euphoric/panic brakes, and High-Stake Guard:
- collectors:
  * FearGreedCollector (Alternative.me Crypto Fear & Greed Index with 14d trend)
  * MarketBreadthCollector (Binance 24h market-wide advance/decline and dispersion)
  * microstructure_collector (24h/7d price series, orderbook/derivatives/whale cache loader, reference symbol discovery)
  * portfolio_collector (spot positions, venue DCA cycles, trailing peaks, recent fills, runtime states)
  * macro_telemetry_collector (macro benchmarks, news sensors, snapshot orchestrator, pre-trade telemetry)
- evaluators:
  * price_series_evaluator (24h/7d price geometry, volume profile POC/VAH/VAL, swing waves, cross-BTC relative strength)
  * microstructure_evaluator (CVD regimes, liquidation clusters, cross-venue arbitrage, cycle deltas, sibling sync)
  * portfolio_evaluator (FIFO/LIFO order-lot accounting, limit-order TCA, cycle reconciliation, enterprise risk metrics)
  * macro_telemetry_evaluator (Deribit options surface, macro benchmark summary formatting, asset scope filtering)
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
