"""Perpetual funding crowding guard.

Protects against initiating positions into hyper-crowded, over-leveraged markets.
Extreme positive funding (> 0.05% / 8h) warns of crowded long squeeze risk.
Extreme negative funding (< -0.05% / 8h) warns of crowded short squeeze risk.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from botcore import safe_float, safe_pos_float
from intelligence.internal.guards.guard_decision import (
    BrakeAction,
    GuardDecision,
    fetch_guard_snapshot,
    normalize_guard_side,
    snap_attr,
)
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
        self.max_long_funding_rate = safe_pos_float(max_long_funding_rate, default=0.0005) or 0.0005
        min_short = safe_float(min_short_funding_rate, default=-0.0005) or -0.0005
        self.min_short_funding_rate = min_short if min_short < 0.0 else -0.0005
        pol_norm = str(crowding_policy or "downscale").strip().lower() if not isinstance(crowding_policy, bool) else "downscale"
        self.crowding_policy = pol_norm if pol_norm in ("downscale", "defer") else "downscale"
        scale_f = safe_pos_float(crowded_scale, default=0.50) or 0.50
        self.crowded_scale = scale_f if scale_f <= 1.0 else 0.50

    def check(
        self,
        symbol: str,
        side: str,
        telemetry: Optional[DerivativesTelemetry] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate if perpetual funding rate warrants a brake."""
        side_norm = normalize_guard_side(side)
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow(
                guard_name="FundingCrowdingGuard",
                reason="unsupported_side_allowed",
            )
        if telemetry is None and self.collector is not None:
            telemetry = fetch_guard_snapshot(self.collector, symbol, now=now)

        if telemetry is None:
            return GuardDecision.allow(
                guard_name="FundingCrowdingGuard",
                reason="no_telemetry_data",
            )

        fr = safe_float(snap_attr(telemetry, "funding_rate"), default=0.0) or 0.0

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
