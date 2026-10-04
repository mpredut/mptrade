"""Root intelligence package unifying internal, external, and sentiment market regimes."""
from __future__ import annotations

# Pillar 1: Internal dynamics (Regime, Kalman, Gradient, Weibull, Volatility)
from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard

from intelligence.internal.state.volatility import calculate_volatility_1h, adaptive_thresholds, vol_1h_pct
from intelligence.internal.state.survival import get_trend_survival_metrics, estimate_T, hybrid_T
from intelligence.internal.state.persistence import calculate_mann_kendall, calculate_hurst_exponent, classify_hurst_regime

# Pillar 2: External telemetry (Whales, Liquidations, OI, Orderbook)
from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector, LiquidationSummary
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry, DerivativesTelemetryCollector
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot, WhalePositioningCollector
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot, OrderbookDepthCollector

from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger

from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard

# Pillar 3: Sentiment & LLM telemetry (Fear & Greed, Market Breadth, Gemini Advisor & High-Stake Guard)
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector, FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector, MarketBreadthSnapshot
from intelligence.sentiment.collectors.gemini_advisor import GeminiMarketAdvisor, GeminiMacroAssessment
from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger
from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard

# Unified Coordinator
from intelligence.composite import CompositeMarketIntelligence, MarketIntelligenceEvaluation

__all__ = [
    # Pillar 1
    "TriggerEvent",
    "TriggerAction",
    "TriggerSide",
    "KalmanTrendTrigger",
    "LinearGradientTrigger",
    "MeanReversionTrigger",
    "GuardDecision",
    "BrakeAction",
    "ParabolicSurgeGuard",
    "WeibullExhaustionGuard",
    "NoiseFloorGuard",
    "TrendSignificanceGuard",
    "calculate_volatility_1h",
    "adaptive_thresholds",
    "vol_1h_pct",
    "get_trend_survival_metrics",
    "estimate_T",
    "hybrid_T",
    "calculate_mann_kendall",
    "calculate_hurst_exponent",
    "classify_hurst_regime",
    # Pillar 2
    "BybitLiquidationCollector",
    "LiquidationSummary",
    "DerivativesTelemetry",
    "DerivativesTelemetryCollector",
    "WhalePositioningSnapshot",
    "WhalePositioningCollector",
    "OrderbookSnapshot",
    "OrderbookDepthCollector",
    "LiquidationCapitulationTrigger",
    "WhaleAccumulationTrigger",
    "LiquidationCascadeGuard",
    "FundingCrowdingGuard",
    "WhaleDivergenceGuard",
    "OrderbookWallGuard",
    # Pillar 3
    "GeminiClient",
    "FearGreedCollector",
    "FearGreedSnapshot",
    "MarketBreadthCollector",
    "MarketBreadthSnapshot",
    "GeminiMarketAdvisor",
    "GeminiMacroAssessment",
    "SentimentContrarianTrigger",
    "MarketBreadthTrigger",
    "ExtremeGreedGuard",
    "PanicWashoutGuard",
    "GeminiHighStakeGuard",
    # Coordinator
    "CompositeMarketIntelligence",
    "MarketIntelligenceEvaluation",
]
