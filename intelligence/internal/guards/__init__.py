"""Risk and market state guards (Brakes and Vetoes)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
    from intelligence.internal.guards.noise_guard import NoiseFloorGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard

_LAZY_IMPORTS: dict[str, str] = {
    "GuardDecision": "intelligence.internal.guards.guard_decision",
    "BrakeAction": "intelligence.internal.guards.guard_decision",
    "ParabolicSurgeGuard": "intelligence.internal.guards.parabolic_guard",
    "WeibullExhaustionGuard": "intelligence.internal.guards.exhaustion_guard",
    "NoiseFloorGuard": "intelligence.internal.guards.noise_guard",
    "TrendSignificanceGuard": "intelligence.internal.guards.trend_significance_guard",
}

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS)
