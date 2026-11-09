"""Root intelligence package unifying internal, external, sentiment, and macro geopolitical regimes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.composite import CompositeMarketIntelligence, MarketIntelligenceEvaluation
    from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector, LiquidationSummary
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetry, DerivativesTelemetryCollector
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector, WhalePositioningSnapshot
    from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
    from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
    from intelligence.internal.guards.noise_guard import NoiseFloorGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.internal.state.persistence import calculate_hurst_exponent, calculate_mann_kendall, classify_hurst_regime
    from intelligence.internal.state.survival import estimate_T, get_trend_survival_metrics, hybrid_T
    from intelligence.internal.state.volatility import (
        adaptive_thresholds,
        calculate_volatility_1h,
        get_kalman_state_path,
        resolve_cached_volatility_1h,
        resolve_cached_volatility_1h_pct,
        vol_1h_pct,
        vol_pct_to_ratio,
        vol_ratio_to_pct,
    )
    from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
    from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
    from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer, GeopoliticalThreatAssessment
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsFeedSnapshot, NewsHeadline
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector, FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector, MarketBreadthSnapshot
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
    from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor, SentimentAdvisorAssessment
    from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
    from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger

_LAZY_IMPORTS: dict[str, str] = {
    # Pillar 1: Internal dynamics
    "TriggerEvent": "intelligence.internal.triggers.trigger_event",
    "TriggerAction": "intelligence.internal.triggers.trigger_event",
    "TriggerSide": "intelligence.internal.triggers.trigger_event",
    "KalmanTrendTrigger": "intelligence.internal.triggers.kalman_trigger",
    "LinearGradientTrigger": "intelligence.internal.triggers.gradient_trigger",
    "MeanReversionTrigger": "intelligence.internal.triggers.mean_reversion_trigger",
    "GuardDecision": "intelligence.internal.guards.guard_decision",
    "BrakeAction": "intelligence.internal.guards.guard_decision",
    "ParabolicSurgeGuard": "intelligence.internal.guards.parabolic_guard",
    "WeibullExhaustionGuard": "intelligence.internal.guards.exhaustion_guard",
    "NoiseFloorGuard": "intelligence.internal.guards.noise_guard",
    "TrendSignificanceGuard": "intelligence.internal.guards.trend_significance_guard",
    "calculate_volatility_1h": "intelligence.internal.state.volatility",
    "adaptive_thresholds": "intelligence.internal.state.volatility",
    "vol_1h_pct": "intelligence.internal.state.volatility",
    "vol_pct_to_ratio": "intelligence.internal.state.volatility",
    "vol_ratio_to_pct": "intelligence.internal.state.volatility",
    "get_kalman_state_path": "intelligence.internal.state.volatility",
    "resolve_cached_volatility_1h": "intelligence.internal.state.volatility",
    "resolve_cached_volatility_1h_pct": "intelligence.internal.state.volatility",
    "get_trend_survival_metrics": "intelligence.internal.state.survival",
    "estimate_T": "intelligence.internal.state.survival",
    "hybrid_T": "intelligence.internal.state.survival",
    "calculate_mann_kendall": "intelligence.internal.state.persistence",
    "calculate_hurst_exponent": "intelligence.internal.state.persistence",
    "classify_hurst_regime": "intelligence.internal.state.persistence",
    # Pillar 2: External telemetry
    "BybitLiquidationCollector": "intelligence.external.collectors.bybit_liquidations_collector",
    "LiquidationSummary": "intelligence.external.collectors.bybit_liquidations_collector",
    "DerivativesTelemetry": "intelligence.external.collectors.derivatives_telemetry_collector",
    "DerivativesTelemetryCollector": "intelligence.external.collectors.derivatives_telemetry_collector",
    "WhalePositioningSnapshot": "intelligence.external.collectors.whale_positioning_collector",
    "WhalePositioningCollector": "intelligence.external.collectors.whale_positioning_collector",
    "OrderbookSnapshot": "intelligence.external.collectors.orderbook_depth_collector",
    "OrderbookDepthCollector": "intelligence.external.collectors.orderbook_depth_collector",
    "LiquidationCapitulationTrigger": "intelligence.external.triggers.liquidation_trigger",
    "WhaleAccumulationTrigger": "intelligence.external.triggers.whale_accumulation_trigger",
    "LiquidationCascadeGuard": "intelligence.external.guards.cascade_guard",
    "FundingCrowdingGuard": "intelligence.external.guards.funding_crowding_guard",
    "WhaleDivergenceGuard": "intelligence.external.guards.whale_divergence_guard",
    "OrderbookWallGuard": "intelligence.external.guards.orderbook_wall_guard",
    "SmartStopLossGuard": "intelligence.external.guards.smart_stop_loss_guard",
    # Pillar 3: Sentiment & LLM telemetry
    "GeminiClient": "intelligence.sentiment.gemini_client",
    "FearGreedCollector": "intelligence.sentiment.collectors.fear_greed_collector",
    "FearGreedSnapshot": "intelligence.sentiment.collectors.fear_greed_collector",
    "MarketBreadthCollector": "intelligence.sentiment.collectors.market_breadth_collector",
    "MarketBreadthSnapshot": "intelligence.sentiment.collectors.market_breadth_collector",
    "SentimentAdvisor": "intelligence.sentiment.sentiment_advisor",
    "SentimentAdvisorAssessment": "intelligence.sentiment.sentiment_advisor",
    "SentimentContrarianTrigger": "intelligence.sentiment.triggers.sentiment_contrarian_trigger",
    "MarketBreadthTrigger": "intelligence.sentiment.triggers.market_breadth_trigger",
    "ExtremeGreedGuard": "intelligence.sentiment.guards.extreme_greed_guard",
    "PanicWashoutGuard": "intelligence.sentiment.guards.panic_washout_guard",
    "HighStakeGuard": "intelligence.sentiment.guards.high_stake_guard",
    # Macro Geopolitical & Energy Shock Shield
    "NewsFeedCollector": "intelligence.macro.news_feed_collector",
    "NewsFeedSnapshot": "intelligence.macro.news_feed_collector",
    "NewsHeadline": "intelligence.macro.news_feed_collector",
    "GeopoliticalThreatAnalyzer": "intelligence.macro.geopolitical_analyzer",
    "GeopoliticalThreatAssessment": "intelligence.macro.geopolitical_analyzer",
    "GeopoliticalShockGuard": "intelligence.macro.geopolitical_guard",
    # Unified Coordinator
    "CompositeMarketIntelligence": "intelligence.composite",
    "MarketIntelligenceEvaluation": "intelligence.composite",
}

_SUBMODULES = frozenset({
    "cli",
    "composite",
    "daemon",
    "external",
    "globaltelemetry_collector",
    "internal",
    "macro",
    "macro_analyzer",
    "macro_sentiment_ssot",
    "sentiment",
})

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS, _SUBMODULES)
