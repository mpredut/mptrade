"""Perpetual funding crowding guard.

Protects against initiating positions into hyper-crowded, over-leveraged markets.
Extreme positive funding (> 0.05% / 8h) warns of crowded long squeeze risk.
Extreme negative funding (< -0.05% / 8h) warns of crowded short squeeze risk.
"""

from __future__ import annotations

from typing import Optional

from botcore import (
    clean_scale,
    obj_attr,
    safe_float,
    safe_pos_float,
)
from instrument_registry import venue_for_symbol
from providers.base import MarketDataProvider
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    OrderGuardEvaluatorBase,
    OrderGuardEvaluatorFactory,
)
from intelligence.external.collectors.derivatives_telemetry_collector import (
    DerivativesTelemetry,
    DerivativesTelemetryCollector,
)


class FundingCrowdingGuard(OrderGuardEvaluatorBase):
    """Downscales or defers orders when perpetual market positioning is excessively crowded."""

    cls_name = "FundingCrowdingGuard"
    GUARD_ID = "external_funding_crowding"
    _EMPTY_SNAPSHOT_REASON = "no_telemetry_data"

    def __init__(
        self,
        collector: Optional[DerivativesTelemetryCollector] = None,
        max_long_funding_rate: float = 0.0005,      # +0.05% per 8h
        min_short_funding_rate: float = -0.0005,    # -0.05% per 8h
        crowding_policy: str = "downscale",         # "downscale" or "defer"
        crowded_scale: float = 0.50,
    ) -> None:
        super().__init__(collector=collector)
        self.max_long_funding_rate = safe_pos_float(max_long_funding_rate, default=0.0005) or 0.0005
        min_short = safe_float(min_short_funding_rate, default=-0.0005) or -0.0005
        self.min_short_funding_rate = min_short if min_short < 0.0 else -0.0005
        pol_norm = str(crowding_policy or "downscale").strip().lower() if not isinstance(crowding_policy, bool) else "downscale"
        self.crowding_policy = pol_norm if pol_norm in ("downscale", "defer") else "downscale"
        self.crowded_scale = clean_scale(
            crowded_scale,
            default=0.50,
            min_positive=True,
            allow_percent=True,
        )

    def _evaluate_external_snapshot(
        self, side_norm: str, telemetry: DerivativesTelemetry
    ) -> GuardDecision:
        raw_fr = safe_float(obj_attr(telemetry, "funding_rate"), default=0.0) or 0.0
        ann_pct = safe_float(obj_attr(telemetry, "funding_annualized_pct"), default=None)
        if ann_pct is not None and (ann_pct != 0.0 or raw_fr == 0.0):
            # Normalize to 8h-equivalent rate: annualized_pct / (3 * 365 * 100)
            fr = ann_pct / (3.0 * 365.0 * 100.0)
        else:
            sym = obj_attr(telemetry, "symbol") or ""
            has_l1 = (safe_pos_float(obj_attr(telemetry, "native_l1_oi_usd"), default=0.0) or 0.0) > 0.0
            if not has_l1 and isinstance(sym, str) and sym.strip():
                has_l1 = MarketDataProvider.has_native_l1_derivatives(venue_for_symbol(sym))
            fr = raw_fr * 8.0 if has_l1 else raw_fr
        reason = (
            f"long_crowding_extreme (funding_rate={fr*100:.3f}% >= {self.max_long_funding_rate*100:.3f}%)"
            if side_norm == "BUY" and fr >= self.max_long_funding_rate
            else (
                f"short_crowding_extreme (funding_rate={fr*100:.3f}% <= {self.min_short_funding_rate*100:.3f}%)"
                if side_norm == "SELL" and fr <= self.min_short_funding_rate
                else None
            )
        )
        if reason is not None:
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


OrderGuardEvaluatorFactory.register(FundingCrowdingGuard.GUARD_ID, FundingCrowdingGuard)
