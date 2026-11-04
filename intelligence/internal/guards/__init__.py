"""Risk and market state guards (Brakes and Vetoes)."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
    from intelligence.internal.guards.noise_guard import NoiseFloorGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard

_LAZY_IMPORTS: dict[str, str] = {
    "GuardDecision": "intelligence.internal.guards.guard_decision",
    "BrakeAction": "intelligence.internal.guards.guard_decision",
    "ParabolicSurgeGuard": "intelligence.internal.guards.parabolic_guard",
    "WeibullExhaustionGuard": "intelligence.internal.guards.exhaustion_guard",
    "NoiseFloorGuard": "intelligence.internal.guards.noise_guard",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_IMPORTS.get(name)
    if mod_name is not None:
        val = getattr(importlib.import_module(mod_name), name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
