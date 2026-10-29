"""Statistical trend significance and persistence guard.

Protects against initiating trend entries on price action that is statistically indistinguishable
from Brownian noise (Mann-Kendall p-value > alpha) or executing momentum strategies in anti-persistent/random-walk regimes (Hurst H <= 0.50).
"""

from __future__ import annotations

from typing import Sequence
from botcore import safe_float, safe_nonneg_float
from intelligence.internal.guards.guard_decision import (
    BrakeAction,
    GuardDecision,
    normalize_guard_side,
    snap_attr,
)
from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.internal.state.persistence import (
    _extract_finite_prices,
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
        self.max_p_value = safe_nonneg_float(max_p_value, default=0.05) or 0.05
        self.min_hurst = safe_float(min_hurst, default=None)

    def evaluate(
        self,
        trigger: TriggerEvent,
        price_history: Sequence[float],
    ) -> GuardDecision:
        """Evaluate if the price series statistically supports the trigger action."""
        # Only evaluate entries (momentum / trend buying or selling)
        if trigger is None:
            return GuardDecision.allow(
                guard_name="TrendSignificanceGuard",
                reason="not_an_entry_trigger",
            )
        raw_action = snap_attr(trigger, "action")
        raw_side = snap_attr(trigger, "side")
        action_u = (
            raw_action.value
            if isinstance(raw_action, TriggerAction)
            else ("" if isinstance(raw_action, bool) or raw_action is None else str(raw_action).strip().upper())
        )
        side_u = normalize_guard_side(
            raw_side.value
            if isinstance(raw_side, TriggerSide)
            else ("" if isinstance(raw_side, bool) or raw_side is None else str(raw_side))
        )
        if action_u != TriggerAction.ENTRY.value or side_u == TriggerSide.HOLD.value or side_u not in (TriggerSide.BUY.value, TriggerSide.SELL.value):
            return GuardDecision.allow(
                guard_name="TrendSignificanceGuard",
                reason="not_an_entry_trigger",
            )

        clean_prices = _extract_finite_prices(price_history)
        if len(clean_prices) < 8:
            return GuardDecision.allow(
                guard_name="TrendSignificanceGuard",
                reason="insufficient_history_for_mk_test",
            )

        # 1. Non-parametric Mann-Kendall significance test
        s, z, p = calculate_mann_kendall(clean_prices)

        # If direction from trigger conflicts with statistical trend direction
        if side_u == TriggerSide.BUY and z < 0:
            return GuardDecision.veto(
                guard_name="TrendSignificanceGuard",
                reason=f"trend_direction_inverted (Z={z:.2f} negative for BUY)",
            )
        if side_u == TriggerSide.SELL and z > 0:
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
        if self.min_hurst is not None and len(clean_prices) >= 65:
            h = calculate_hurst_exponent(clean_prices)
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
