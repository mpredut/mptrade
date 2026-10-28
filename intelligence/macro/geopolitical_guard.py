"""Geopolitical shock guard protecting against flash dumps from war and energy crises."""
from __future__ import annotations

import logging
import math
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
from intelligence.macro_sentiment_ssot import (
    normalize_recommended_brake,
    normalize_threat_level,
)

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
        df = 0.50
        if downscale_factor is not None and not isinstance(downscale_factor, bool):
            try:
                f = float(downscale_factor)
                if math.isfinite(f) and 1.0 < f <= 100.0:
                    f = f / 100.0
                if math.isfinite(f) and 0.0 < f <= 1.0:
                    df = f
            except (TypeError, ValueError):
                pass
        self.downscale_factor = df

    def check(
        self,
        symbol: str,
        side: str,
        assessment: Optional[GeopoliticalThreatAssessment],
    ) -> GuardDecision:
        """Evaluate order against current geopolitical threat assessment."""
        # Only BUY orders are restrained; SELL / Stop-Loss orders always pass freely
        if not isinstance(side, str) or side.strip().upper() not in ("BUY", "B", "BID", "LONG"):
            return GuardDecision.allow("GeopoliticalShockGuard", "sell_permitted_in_shock")

        if assessment is None:
            return GuardDecision.allow("GeopoliticalShockGuard", "no_geopolitical_data")

        def _attr(key: str, default: object = None) -> object:
            return assessment.get(key, default) if isinstance(assessment, dict) else getattr(assessment, key, default)

        raw_level = _attr("threat_level", None)
        level = normalize_threat_level(raw_level, default="NORMAL")
        raw_brake = _attr("recommended_brake", None)
        brake = normalize_recommended_brake(raw_brake, threat_level=level)
        raw_summary = _attr("summary", None)
        summary = str(raw_summary).strip() if (isinstance(raw_summary, str) and raw_summary.strip()) else "geopolitical_shock"
        raw_risk = _attr("risk_score", 0.0)
        risk_score = 0.0
        if raw_risk is not None and not isinstance(raw_risk, bool):
            try:
                rf = float(raw_risk)
                if math.isfinite(rf):
                    risk_score = rf
            except (TypeError, ValueError):
                pass

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
