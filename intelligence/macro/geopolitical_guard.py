"""Geopolitical shock guard protecting against flash dumps from war and energy crises."""
from __future__ import annotations

from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    normalize_order_side,
    normalize_recommended_brake,
    normalize_threat_level,
    obj_attr,
    obj_float,
)


class GeopoliticalShockGuard:
    """Black Swan & Shock Shield braking BUY orders during major geopolitical or energy crises."""

    def __init__(self, downscale_factor: float = 0.50) -> None:
        df = clean_scale(downscale_factor, default=0.50, allow_percent=True)
        self.downscale_factor = df if df > 0.0 else 0.50

    def check(
        self,
        symbol: str,
        side: str,
        assessment: Optional[GeopoliticalThreatAssessment],
    ) -> GuardDecision:
        """Evaluate order against current geopolitical threat assessment."""
        if normalize_order_side(side) != "BUY":
            return GuardDecision.allow("GeopoliticalShockGuard", "sell_permitted_in_shock")
        if assessment is None:
            return GuardDecision.allow("GeopoliticalShockGuard", "no_geopolitical_data")

        level = normalize_threat_level(obj_attr(assessment, "threat_level"), default="NORMAL")
        brake = normalize_recommended_brake(obj_attr(assessment, "recommended_brake"), threat_level=level)
        raw_summary = obj_attr(assessment, "summary")
        summary = str(raw_summary).strip() if (isinstance(raw_summary, str) and raw_summary.strip()) else "geopolitical_shock"
        risk_score = obj_float(assessment, "risk_score", 0.0)

        if level == "CRITICAL_SHOCK" or brake == "HARD_VETO_NEW_BUYS":
            return GuardDecision.veto(
                "GeopoliticalShockGuard",
                f"veto_critical_geopolitical_shock ({summary})",
                threat_level=level,
                risk_score=risk_score,
            )
        if level == "ELEVATED" or brake == "DOWNSCALE_50":
            return GuardDecision.downscale(
                "GeopoliticalShockGuard",
                scale=self.downscale_factor,
                reason=f"downscale_elevated_geopolitical_risk ({summary})",
                threat_level=level,
                risk_score=risk_score,
            )
        return GuardDecision.allow("GeopoliticalShockGuard", "geopolitical_climate_normal")
