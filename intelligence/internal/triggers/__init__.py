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

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS)
