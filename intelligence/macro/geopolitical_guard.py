"""Geopolitical shock guard protecting against flash dumps from war and energy crises."""
from __future__ import annotations

import logging
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment

logger = logging.getLogger("intelligence.macro.geopolitical_guard")


class GeopoliticalShockGuard:
    """Black Swan & Shock Shield braking BUY orders during major geopolitical or energy crises.

    - CRITICAL_SHOCK: Complete veto on new BUY entries to prevent catching macro-driven liquidation knives.
    - ELEVATED: Downscales BUY order size (50%) to limit exposure while navigating uncertainty.
    - NORMAL: 100% green light (0ms overhead, instant pass).
    - SELL / Stop-Loss orders are never restricted (cash preservation allowed).
    """

    def __init__(
        self,
        downscale_factor: float = 0.50,
    ) -> None:
        self.downscale_factor = downscale_factor

    def check(
        self,
        symbol: str,
        side: str,
        assessment: Optional[GeopoliticalThreatAssessment],
    ) -> GuardDecision:
        """Evaluate order against current geopolitical threat assessment."""
        # Only BUY orders are restrained; SELL / Stop-Loss orders always pass freely
        if side.upper() != "BUY":
            return GuardDecision.allow("GeopoliticalShockGuard", "sell_permitted_in_shock")

        if assessment is None:
            return GuardDecision.allow("GeopoliticalShockGuard", "no_geopolitical_data")

        level = assessment.threat_level.upper()
        brake = assessment.recommended_brake.upper()

        if level == "CRITICAL_SHOCK" or brake == "HARD_VETO_NEW_BUYS":
            return GuardDecision.veto(
                "GeopoliticalShockGuard",
                f"veto_critical_geopolitical_shock ({assessment.summary})",
                threat_level=level,
                risk_score=assessment.risk_score,
            )

        if level == "ELEVATED" or brake == "DOWNSCALE_50":
            return GuardDecision.downscale(
                "GeopoliticalShockGuard",
                scale=self.downscale_factor,
                reason=f"downscale_elevated_geopolitical_risk ({assessment.summary})",
                threat_level=level,
                risk_score=assessment.risk_score,
            )

        return GuardDecision.allow("GeopoliticalShockGuard", "geopolitical_climate_normal")
