"""Perpetual funding crowding guard.

Protects against initiating positions into hyper-crowded, over-leveraged markets.
Extreme positive funding (> 0.05% / 8h) warns of crowded long squeeze risk.
Extreme negative funding (< -0.05% / 8h) warns of crowded short squeeze risk.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.derivatives_telemetry import (
    DerivativesTelemetry,
    DerivativesTelemetryCollector,
)


def _safe_float(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


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
        max_long = _safe_float(max_long_funding_rate, 0.0005)
        self.max_long_funding_rate = max_long if max_long > 0.0 else 0.0005
        min_short = _safe_float(min_short_funding_rate, -0.0005)
        self.min_short_funding_rate = min_short if min_short < 0.0 else -0.0005
        pol_norm = str(crowding_policy or "downscale").strip().lower() if not isinstance(crowding_policy, bool) else "downscale"
        self.crowding_policy = pol_norm if pol_norm in ("downscale", "defer") else "downscale"
        scale_f = _safe_float(crowded_scale, 0.50)
        self.crowded_scale = scale_f if 0.0 < scale_f <= 1.0 else 0.50

    def check(
        self,
        symbol: str,
        side: str,
        telemetry: Optional[DerivativesTelemetry] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate if perpetual funding rate warrants a brake."""
        side_norm = str(side or "").strip().upper() if isinstance(side, str) else ""
        if side_norm in ("B", "BUY", "BID", "LONG"):
            side_norm = "BUY"
        elif side_norm in ("S", "SELL", "ASK", "A", "SHORT"):
            side_norm = "SELL"
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow(
                guard_name="FundingCrowdingGuard",
                reason="unsupported_side_allowed",
            )
        if telemetry is None and self.collector is not None:
            try:
                telemetry = self.collector.fetch(symbol, now=now) if now is not None else self.collector.fetch(symbol)
            except TypeError:
                telemetry = self.collector.fetch(symbol)

        if telemetry is None:
            return GuardDecision.allow(
                guard_name="FundingCrowdingGuard",
                reason="no_telemetry_data",
            )

        raw_fr = telemetry.get("funding_rate") if isinstance(telemetry, dict) else getattr(telemetry, "funding_rate", None)
        fr = _safe_float(raw_fr, 0.0)

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
