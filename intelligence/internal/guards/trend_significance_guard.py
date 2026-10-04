"""Statistical trend significance and persistence guard.

Protects against initiating trend entries on price action that is statistically indistinguishable
from Brownian noise (Mann-Kendall p-value > alpha) or executing momentum strategies in anti-persistent/random-walk regimes (Hurst H <= 0.50).
"""

from __future__ import annotations

from typing import Sequence
from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.internal.state.persistence import (
    calculate_mann_kendall,
    calculate_hurst_exponent,
    classify_hurst_regime,
)


class TrendSignificanceGuard:
    """Blocks or defers trend-following entries on noisy or non-persistent price data."""

    def __init__(
        self,
        max_p_value: float = 0.05,
        min_hurst: float | None = None,
    ) -> None:
        """
        Args:
            max_p_value: Maximum allowable two-sided p-value from Mann-Kendall test.
                         Higher values indicate trend slope is indistinguishable from random noise.
            min_hurst: If specified (e.g. 0.50), requires Hurst exponent >= min_hurst for trend entries.
        """
        self.max_p_value = max_p_value
        self.min_hurst = min_hurst

    def evaluate(
        self,
        trigger: TriggerEvent,
        price_history: Sequence[float],
    ) -> GuardDecision:
        """Evaluate if the price series statistically supports the trigger action."""
        # Only evaluate entries (momentum / trend buying or selling)
        if trigger.action != TriggerAction.ENTRY or trigger.side == TriggerSide.HOLD:
            return GuardDecision.allow(
                guard_name="TrendSignificanceGuard",
                reason="not_an_entry_trigger",
            )

        if not price_history or len(price_history) < 8:
            return GuardDecision.allow(
                guard_name="TrendSignificanceGuard",
                reason="insufficient_history_for_mk_test",
            )

        # 1. Non-parametric Mann-Kendall significance test
        s, z, p = calculate_mann_kendall(price_history)

        # If direction from trigger conflicts with statistical trend direction
        if trigger.side == TriggerSide.BUY and z < 0:
            return GuardDecision.veto(
                guard_name="TrendSignificanceGuard",
                reason=f"trend_direction_inverted (Z={z:.2f} negative for BUY)",
            )
        if trigger.side == TriggerSide.SELL and z > 0:
            return GuardDecision.veto(
                guard_name="TrendSignificanceGuard",
                reason=f"trend_direction_inverted (Z={z:.2f} positive for SELL)",
            )

        if p > self.max_p_value:
            return GuardDecision.defer(
                guard_name="TrendSignificanceGuard",
                reason=f"trend_slope_not_statistically_significant (p={p:.4f} > {self.max_p_value})",
            )

        # 2. Hurst exponent check (if required and enough data points)
        if self.min_hurst is not None and len(price_history) >= 65:
            h = calculate_hurst_exponent(price_history)
            regime = classify_hurst_regime(h)
            if h is not None and h < self.min_hurst:
                return GuardDecision.defer(
                    guard_name="TrendSignificanceGuard",
                    reason=f"hurst_regime_{regime} (H={h:.2f} < {self.min_hurst})",
                )

        return GuardDecision.allow(
            guard_name="TrendSignificanceGuard",
            reason=f"trend_statistically_significant (p={p:.4f}, Z={z:.2f})",
        )
