"""Internal intelligence package for local price and statistical regime."""
from __future__ import annotations

from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger, KalmanTrend
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard

from intelligence.internal.state.volatility import calculate_volatility_1h, adaptive_thresholds, vol_1h_pct
from intelligence.internal.state.survival import get_trend_survival_metrics
from intelligence.internal.state.persistence import (
    calculate_mann_kendall,
    calculate_hurst_exponent,
    classify_hurst_regime,
)

__all__ = [
    "TriggerEvent",
    "TriggerAction",
    "TriggerSide",
    "KalmanTrendTrigger",
    "KalmanTrend",
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
    "calculate_mann_kendall",
    "calculate_hurst_exponent",
    "classify_hurst_regime",
]
