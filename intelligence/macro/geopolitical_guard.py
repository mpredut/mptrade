"""Geopolitical shock guard protecting against flash dumps from war and energy crises."""
from __future__ import annotations

from typing import Optional

from botcore import normalize_side
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    OrderGuardEvaluatorBase,
    OrderGuardEvaluatorFactory,
)
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    normalize_recommended_brake,
    normalize_threat_level,
    obj_attr,
)


class GeopoliticalShockGuard(OrderGuardEvaluatorBase):
    """Black Swan & Shock Shield braking BUY orders during major geopolitical or energy crises."""

    cls_name = "GeopoliticalShockGuard"
    GUARD_ID = "macro_geopolitical_shock"

    def __init__(self, downscale_factor: float = 0.50) -> None:
        super().__init__()
        self.downscale_factor = clean_scale(downscale_factor, default=0.50, allow_percent=True, min_positive=True)

    def check(
        self,
        symbol: str,
        side: str,
        assessment: Optional[GeopoliticalThreatAssessment],
        *,
        market_absorbed: bool = False,
    ) -> GuardDecision:
        """Evaluate order against current geopolitical threat assessment."""
        if normalize_side(side) != "BUY":
            return GuardDecision.allow("GeopoliticalShockGuard", "sell_permitted_in_shock")
        if assessment is None:
            return GuardDecision.allow("GeopoliticalShockGuard", "no_geopolitical_data")

        level = normalize_threat_level(obj_attr(assessment, "threat_level"), default="NORMAL")
        brake = normalize_recommended_brake(obj_attr(assessment, "recommended_brake"), threat_level=level)
        raw_summary = obj_attr(assessment, "summary")
        summary = str(raw_summary).strip() if (isinstance(raw_summary, str) and raw_summary.strip()) else "geopolitical_shock"
        risk_score = clean_scale(obj_attr(assessment, "risk_score"), default=0.0, allow_percent=True)

        if level == "CRITICAL_SHOCK" or brake == "HARD_VETO_NEW_BUYS":
            return GuardDecision.veto(
                "GeopoliticalShockGuard",
                f"veto_critical_geopolitical_shock ({summary})",
                threat_level=level,
                risk_score=risk_score,
            )
        if level == "ELEVATED" or brake == "DOWNSCALE_50":
            if market_absorbed:
                return GuardDecision.allow(
                    "GeopoliticalShockGuard",
                    f"geopolitical_elevated_absorbed_by_market ({summary})",
                    threat_level=level,
                    risk_score=risk_score,
                )
            return GuardDecision.downscale(
                "GeopoliticalShockGuard",
                scale=self.downscale_factor,
                reason=f"downscale_elevated_geopolitical_risk ({summary})",
                threat_level=level,
                risk_score=risk_score,
            )
        return GuardDecision.allow("GeopoliticalShockGuard", "geopolitical_climate_normal")


OrderGuardEvaluatorFactory.register(GeopoliticalShockGuard.GUARD_ID, GeopoliticalShockGuard)
