"""Weibull trend exhaustion guard: prevents entering trends that have reached their late-life percentile."""
from __future__ import annotations

import math
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision


class WeibullExhaustionGuard:
    """Exhaustion guard that calculates whether a persistent trend has reached statistical maturity.

    Uses the empirical survival distribution (Weibull model from trend_survival).
    If a trend duration exceeds the P90 threshold (90% of all historical episodes died
    before this duration), new trend-following entries are either downscaled or vetoed
    to prevent buying at the exhausted tail of an old move.
    """

    def __init__(
        self,
        default_p90_days: float = 7.0,
        policy: str = "downscale",  # "downscale" | "veto"
        exhausted_scale: float = 0.25,
    ):
        self.default_p90_days = float(default_p90_days)
        self.policy = policy.lower()
        self.exhausted_scale = float(exhausted_scale)

    def check(
        self,
        symbol: str,
        side: str,
        trend_duration_seconds: float,
        p90_days: Optional[float] = None,
        median_days: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate if the trend is mature/exhausted for the requested side."""
        side_u = str(side or "").upper()
        if side_u != "BUY":
            # Exiting an exhausted trend is encouraged, never blocked.
            return GuardDecision.allow("WeibullExhaustionGuard", "non_buy_order")

        if not math.isfinite(trend_duration_seconds) or trend_duration_seconds <= 0:
            return GuardDecision.allow("WeibullExhaustionGuard", "no_duration_data")

        duration_days = trend_duration_seconds / 86400.0
        limit_p90 = float(p90_days) if p90_days and p90_days > 0 else self.default_p90_days

        if duration_days > limit_p90:
            reason = (
                f"trend_exhausted (duration={duration_days:.1f}d > P90={limit_p90:.1f}d, "
                f"median={median_days or 3.0:.1f}d)"
            )
            if self.policy == "veto":
                return GuardDecision.veto(
                    "WeibullExhaustionGuard",
                    reason,
                    duration_days=duration_days,
                    p90_days=limit_p90,
                )
            else:
                return GuardDecision.downscale(
                    "WeibullExhaustionGuard",
                    scale=self.exhausted_scale,
                    reason=reason,
                    duration_days=duration_days,
                    p90_days=limit_p90,
                )

        return GuardDecision.allow(
            "WeibullExhaustionGuard",
            f"trend_healthy (duration={duration_days:.1f}d <= P90={limit_p90:.1f}d)",
            duration_days=duration_days,
            p90_days=limit_p90,
        )
