"""Liquidation cascade guard.

Prevents catching a falling knife during an active, explosive liquidation cascade.
Defers execution until the cascade rate has decelerated.
"""

from __future__ import annotations

import time
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.bybit_liquidations import (
    BybitLiquidationCollector,
    LiquidationSummary,
)


class LiquidationCascadeGuard:
    """Blocks or defers orders during active real-time liquidation storms."""

    def __init__(
        self,
        collector: Optional[BybitLiquidationCollector] = None,
        max_active_cascade_usd: float = 500_000.0,
        active_window_sec: float = 60.0,
    ) -> None:
        """
        Args:
            max_active_cascade_usd: Dollar threshold in active_window_sec to qualify as an active cascade.
            active_window_sec: Window for instantaneous cascade velocity (default 60s).
        """
        self.collector = collector
        self.max_active_cascade_usd = max_active_cascade_usd
        self.active_window_sec = active_window_sec

    def check(
        self,
        symbol: str,
        side: str,
        summary: Optional[LiquidationSummary] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate whether an active cascade prevents order execution."""
        side_norm = side.upper()
        # Cascade guard primarily protects BUY orders against active long cascades (forced dump)
        if side_norm != "BUY":
            return GuardDecision.allow(
                guard_name="LiquidationCascadeGuard",
                reason="non_buy_order_allowed",
            )

        if summary is None and self.collector is not None:
            summary = self.collector.get_summary(symbol, window_sec=self.active_window_sec)

        if summary is None:
            return GuardDecision.allow(
                guard_name="LiquidationCascadeGuard",
                reason="no_liquidation_data",
            )

        # If long liquidations in the short window exceed the cascade threshold, it's an active knife
        if summary.long_liq_usd >= self.max_active_cascade_usd:
            return GuardDecision.defer(
                guard_name="LiquidationCascadeGuard",
                reason=f"active_long_liquidation_cascade (${summary.long_liq_usd:,.0f} in {self.active_window_sec:.0f}s)",
                long_liq_usd=summary.long_liq_usd,
                active_window_sec=self.active_window_sec,
            )

        return GuardDecision.allow(
            guard_name="LiquidationCascadeGuard",
            reason="liquidation_flow_stable",
            long_liq_usd=summary.long_liq_usd,
        )
