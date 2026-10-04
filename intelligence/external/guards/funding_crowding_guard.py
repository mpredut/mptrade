"""Perpetual funding crowding guard.

Protects against initiating positions into hyper-crowded, over-leveraged markets.
Extreme positive funding (> 0.05% / 8h) warns of crowded long squeeze risk.
Extreme negative funding (< -0.05% / 8h) warns of crowded short squeeze risk.
"""

from __future__ import annotations

from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.derivatives_telemetry import (
    DerivativesTelemetry,
    DerivativesTelemetryCollector,
)


class FundingCrowdingGuard:
    """Downscales or defers orders when perpetual market positioning is excessively crowded."""

    def __init__(
        self,
        collector: Optional[DerivativesTelemetryCollector] = None,
        max_long_funding_rate: float = 0.0005,      # +0.05% per 8h
        min_short_funding_rate: float = -0.0005,    # -0.05% per 8h
        crowding_policy: str = "downscale",         # "downscale" or "defer"
        crowded_scale: float = 0.50,
    ) -> None:
        self.collector = collector
        self.max_long_funding_rate = max_long_funding_rate
        self.min_short_funding_rate = min_short_funding_rate
        self.crowding_policy = crowding_policy
        self.crowded_scale = crowded_scale

    def check(
        self,
        symbol: str,
        side: str,
        telemetry: Optional[DerivativesTelemetry] = None,
    ) -> GuardDecision:
        """Evaluate if perpetual funding rate warrants a brake."""
        side_norm = side.upper()
        if telemetry is None and self.collector is not None:
            telemetry = self.collector.fetch(symbol)

        if telemetry is None:
            return GuardDecision.allow(
                guard_name="FundingCrowdingGuard",
                reason="no_telemetry_data",
            )

        fr = telemetry.funding_rate

        # 1. Entering BUY when market is hyper-crowded long
        if side_norm == "BUY" and fr >= self.max_long_funding_rate:
            reason = f"long_crowding_extreme (funding_rate={fr*100:.3f}% >= {self.max_long_funding_rate*100:.3f}%)"
            if self.crowding_policy == "defer":
                return GuardDecision.defer(
                    guard_name="FundingCrowdingGuard",
                    reason=reason,
                    funding_rate=fr,
                )
            return GuardDecision.downscale(
                guard_name="FundingCrowdingGuard",
                scale=self.crowded_scale,
                reason=reason,
                funding_rate=fr,
            )

        # 2. Entering SELL when market is hyper-crowded short
        if side_norm == "SELL" and fr <= self.min_short_funding_rate:
            reason = f"short_crowding_extreme (funding_rate={fr*100:.3f}% <= {self.min_short_funding_rate*100:.3f}%)"
            if self.crowding_policy == "defer":
                return GuardDecision.defer(
                    guard_name="FundingCrowdingGuard",
                    reason=reason,
                    funding_rate=fr,
                )
            return GuardDecision.downscale(
                guard_name="FundingCrowdingGuard",
                scale=self.crowded_scale,
                reason=reason,
                funding_rate=fr,
            )

        return GuardDecision.allow(
            guard_name="FundingCrowdingGuard",
            reason=f"funding_normal ({fr*100:+.3f}%)",
            funding_rate=fr,
        )
