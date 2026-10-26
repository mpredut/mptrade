"""Linear regression slope and signal-to-noise ratio directional trigger."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide


class LinearGradientTrigger:
    """Emits Entry/Exit triggers based on linear regression gradient exceeding noise threshold."""

    def __init__(self, strength_threshold: float = 2.0):
        st = 2.0
        if strength_threshold is not None and not isinstance(strength_threshold, bool):
            try:
                sf = float(strength_threshold)
                if math.isfinite(sf) and sf > 0.0:
                    st = sf
            except (TypeError, ValueError, OverflowError):
                pass
        self.strength_threshold = st
        self.last_state = "neutral"  # "bull", "bear", "neutral"

    def evaluate(
        self,
        symbol: str,
        price: float,
        gradient: float,
        epsilon: float,
        ts: Optional[float] = None,
    ) -> Tuple[str, Optional[TriggerEvent]]:
        """Evaluate gradient against epsilon noise floor.

        Returns (regime_str, Optional[TriggerEvent]).
        """
        if (
            price is None
            or gradient is None
            or epsilon is None
            or isinstance(price, bool)
            or isinstance(gradient, bool)
            or isinstance(epsilon, bool)
        ):
            return "unknown", None
        try:
            price = float(price)
            gradient = float(gradient)
            epsilon = float(epsilon)
        except (TypeError, ValueError):
            return "unknown", None
        if (
            not math.isfinite(price)
            or price <= 0
            or not math.isfinite(gradient)
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            return "unknown", None

        ts_f = 0.0
        if ts is not None and not isinstance(ts, bool):
            try:
                tf = float(ts)
                if math.isfinite(tf) and tf > 0.0:
                    ts_f = tf
            except (TypeError, ValueError, OverflowError):
                ts_f = 0.0

        strength = abs(gradient) / epsilon
        regime = "sideways"
        if strength > self.strength_threshold:
            regime = "bull" if gradient > 0 else "bear"

        trigger: Optional[TriggerEvent] = None
        if regime != self.last_state:
            if regime == "bull":
                trigger = TriggerEvent(
                    action=TriggerAction.ENTRY,
                    side=TriggerSide.BUY,
                    source="gradient_trigger",
                    symbol=symbol,
                    price=price,
                    strength=strength,
                    ts=ts_f,
                    velocity=gradient,
                    confidence=strength,
                    reason="gradient_breakout_bull",
                    metadata={"gradient": gradient, "epsilon": epsilon, "strength": strength},
                )
            elif regime == "bear":
                trigger = TriggerEvent(
                    action=TriggerAction.EXIT,
                    side=TriggerSide.SELL,
                    source="gradient_trigger",
                    symbol=symbol,
                    price=price,
                    strength=strength,
                    ts=ts_f,
                    velocity=gradient,
                    confidence=strength,
                    reason="gradient_breakout_bear",
                    metadata={"gradient": gradient, "epsilon": epsilon, "strength": strength},
                )
            self.last_state = regime

        return regime, trigger
