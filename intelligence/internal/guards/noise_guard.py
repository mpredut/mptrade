"""Noise floor guard: defers trading during dead, flat chop or within the noise threshold."""
from __future__ import annotations

import math
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision


class NoiseFloorGuard:
    """Noise floor guard preventing orders when market movement is indistinguishable from noise."""

    def __init__(self, min_strength_ratio: float = 1.0):
        self.min_strength_ratio = float(min_strength_ratio)

    def check(
        self,
        symbol: str,
        side: str,
        gradient: float,
        epsilon: float,
    ) -> GuardDecision:
        """Evaluate if the instantaneous gradient exceeds the statistical noise floor."""
        if not math.isfinite(gradient) or not math.isfinite(epsilon):
            return GuardDecision.allow("NoiseFloorGuard", "non_finite_signal_allowed")

        abs_g = abs(gradient)
        eps = abs(epsilon)

        if eps <= 0:
            return GuardDecision.allow("NoiseFloorGuard", "noise_floor_zero")

        strength = abs_g / eps
        if strength < self.min_strength_ratio:
            return GuardDecision.defer(
                "NoiseFloorGuard",
                f"signal_in_noise_floor (|gradient|={abs_g:.6f} <= epsilon={eps:.6f}, strength={strength:.2f})",
                strength=strength,
                gradient=gradient,
                epsilon=epsilon,
            )

        return GuardDecision.allow(
            "NoiseFloorGuard",
            f"signal_above_noise (strength={strength:.2f} >= {self.min_strength_ratio:.2f})",
            strength=strength,
        )
