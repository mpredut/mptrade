"""Linear regression slope and signal-to-noise ratio directional trigger."""
from __future__ import annotations

from typing import Optional, Tuple

from botcore import safe_float, safe_pos_float
from intelligence.internal.triggers.trigger_event import TriggerEvent


class LinearGradientTrigger:
    """Emits Entry/Exit triggers based on linear regression gradient exceeding noise threshold."""

    def __init__(self, strength_threshold: float = 2.0):
        self.strength_threshold = safe_pos_float(strength_threshold, default=2.0) or 2.0
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
        price_f = safe_pos_float(price, default=None)
        grad_f = safe_float(gradient, default=None)
        eps_f = safe_pos_float(epsilon, default=None)
        if price_f is None or grad_f is None or eps_f is None:
            return "unknown", None
        price, gradient, epsilon = price_f, grad_f, eps_f
        ts_f = safe_pos_float(ts, default=0.0) or 0.0

        strength = abs(gradient) / epsilon
        regime = "sideways"
        if strength > self.strength_threshold:
            regime = "bull" if gradient > 0 else "bear"

        trigger: Optional[TriggerEvent] = None
        if regime != self.last_state:
            if regime in ("bull", "bear"):
                trigger = TriggerEvent.directional(
                    regime == "bull",
                    source="gradient_trigger",
                    symbol=symbol,
                    price=price,
                    strength=strength,
                    ts=ts_f,
                    velocity=gradient,
                    confidence=strength,
                    reason_bull="gradient_breakout_bull",
                    reason_bear="gradient_breakout_bear",
                    metadata={"gradient": gradient, "epsilon": epsilon, "strength": strength},
                )
            self.last_state = regime

        return regime, trigger
