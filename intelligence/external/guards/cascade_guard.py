"""Liquidation cascade guard.

Prevents catching a falling knife during an active, explosive liquidation cascade.
Defers execution until the cascade rate has decelerated.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from intelligence.internal.guards.guard_decision import (
    BrakeAction,
    GuardDecision,
    normalize_guard_side,
    safe_guard_float as _safe_float,
    safe_pos_guard_float as _safe_pos_float,
    snap_attr,
)
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
        self.max_active_cascade_usd = _safe_pos_float(max_active_cascade_usd, 500_000.0)
        self.active_window_sec = _safe_pos_float(active_window_sec, 60.0)

    def check(
        self,
        symbol: str,
        side: str,
        summary: Optional[LiquidationSummary] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate whether an active cascade prevents order execution."""
        side_norm = normalize_guard_side(side)
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow(
                guard_name="LiquidationCascadeGuard",
                reason="unsupported_side_allowed",
            )

        if summary is None and self.collector is not None:
            try:
                summary = self.collector.get_summary(
                    symbol, window_sec=self.active_window_sec, now=now
                )
            except TypeError:
                summary = self.collector.get_summary(
                    symbol, window_sec=self.active_window_sec
                )

        if summary is None:
            return GuardDecision.allow(
                guard_name="LiquidationCascadeGuard",
                reason="no_liquidation_data",
            )

        long_liq = _safe_float(snap_attr(summary, "long_liq_usd"), 0.0)
        short_liq = _safe_float(snap_attr(summary, "short_liq_usd"), 0.0)

        # 1. BUY: If long liquidations in the short window exceed the cascade threshold, it's an active knife
        if side_norm == "BUY" and long_liq >= self.max_active_cascade_usd:
            return GuardDecision.defer(
                guard_name="LiquidationCascadeGuard",
                reason=f"active_long_liquidation_cascade (${long_liq:,.0f} in {self.active_window_sec:.0f}s)",
                long_liq_usd=long_liq,
                active_window_sec=self.active_window_sec,
            )

        # 2. SELL: If short liquidations in the short window exceed the cascade threshold, it's an active squeeze
        if side_norm == "SELL" and short_liq >= self.max_active_cascade_usd:
            return GuardDecision.defer(
                guard_name="LiquidationCascadeGuard",
                reason=f"active_short_liquidation_cascade (${short_liq:,.0f} in {self.active_window_sec:.0f}s)",
                short_liq_usd=short_liq,
                active_window_sec=self.active_window_sec,
            )

        return GuardDecision.allow(
            guard_name="LiquidationCascadeGuard",
            reason="liquidation_flow_stable",
            long_liq_usd=long_liq,
            short_liq_usd=short_liq,
        )
