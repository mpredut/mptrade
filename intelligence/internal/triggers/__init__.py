"""Directional entry and exit triggers."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
    from intelligence.internal.triggers.kalman_trigger import KalmanTrend, KalmanTrendTrigger
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide

_LAZY_IMPORTS: dict[str, str] = {
    "TriggerEvent": "intelligence.internal.triggers.trigger_event",
    "TriggerAction": "intelligence.internal.triggers.trigger_event",
    "TriggerSide": "intelligence.internal.triggers.trigger_event",
    "KalmanTrend": "intelligence.internal.triggers.kalman_trigger",
    "KalmanTrendTrigger": "intelligence.internal.triggers.kalman_trigger",
    "LinearGradientTrigger": "intelligence.internal.triggers.gradient_trigger",
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
