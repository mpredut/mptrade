"""Root intelligence package for market state, triggers, and guards."""
from __future__ import annotations

from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.state.volatility import calculate_volatility_1h, adaptive_thresholds
from intelligence.internal.state.survival import get_trend_survival_metrics
from intelligence.composite import CompositeMarketIntelligence

__all__ = [
    "TriggerEvent",
    "TriggerAction",
    "TriggerSide",
    "KalmanTrendTrigger",
    "LinearGradientTrigger",
    "GuardDecision",
    "BrakeAction",
    "ParabolicSurgeGuard",
    "WeibullExhaustionGuard",
    "NoiseFloorGuard",
    "calculate_volatility_1h",
    "adaptive_thresholds",
    "get_trend_survival_metrics",
    "CompositeMarketIntelligence",
]
