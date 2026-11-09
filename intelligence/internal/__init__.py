"""Internal intelligence package for local price and statistical regime."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
    from intelligence.internal.guards.noise_guard import NoiseFloorGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.internal.state.persistence import (
        calculate_hurst_exponent,
        calculate_mann_kendall,
        classify_hurst_regime,
    )
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
    from intelligence.internal.triggers.kalman_trigger import KalmanTrend, KalmanTrendTrigger
    from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide

_LAZY_IMPORTS: dict[str, str] = {
    "TriggerEvent": "intelligence.internal.triggers.trigger_event",
    "TriggerAction": "intelligence.internal.triggers.trigger_event",
    "TriggerSide": "intelligence.internal.triggers.trigger_event",
    "KalmanTrendTrigger": "intelligence.internal.triggers.kalman_trigger",
    "KalmanTrend": "intelligence.internal.triggers.kalman_trigger",
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
}

_SUBMODULES = frozenset({"guards", "state", "triggers"})

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS, _SUBMODULES)
