"""Whale positioning and Open Interest divergence guard.

Guards against buying false breakout rallies fueled only by short-covering (falling OI)
rather than genuine whale capital accumulation (rising OI).
"""

from __future__ import annotations

from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.whale_positioning import (
    WhalePositioningCollector,
    WhalePositioningSnapshot,
)


class WhaleDivergenceGuard:
    """Blocks or defers orders when whale positioning or OI flow contradicts the move."""

    def __init__(
        self,
        collector: Optional[WhalePositioningCollector] = None,
        block_short_covering_fakeout: bool = True,
        block_aggressive_shorting: bool = True,
    ) -> None:
        self.collector = collector
        self.block_short_covering_fakeout = block_short_covering_fakeout
        self.block_aggressive_shorting = block_aggressive_shorting

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[WhalePositioningSnapshot] = None,
    ) -> GuardDecision:
        """Evaluate if whale positioning or OI divergence conflicts with the order."""
        side_norm = side.upper()
        if snapshot is None and self.collector is not None:
            snapshot = self.collector.fetch(symbol)

        if snapshot is None:
            return GuardDecision.allow(
                guard_name="WhaleDivergenceGuard",
                reason="no_whale_snapshot_data",
            )

        regime = snapshot.divergence_regime

        # 1. Buying into a short-covering fakeout (price up, but OI falling: no new whale money!)
        if side_norm == "BUY" and self.block_short_covering_fakeout and regime == "short_covering":
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"short_covering_fakeout (OI_1h={snapshot.open_interest_1h_change_pct:+.1f}%, no whale accumulation)",
                open_interest_change=snapshot.open_interest_1h_change_pct,
                divergence_regime=regime,
            )

        # 2. Buying into aggressive whale short expansion (price down, but OI expanding: heavy short momentum)
        if side_norm == "BUY" and self.block_aggressive_shorting and regime == "aggressive_shorting":
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"aggressive_whale_shorting (OI_1h={snapshot.open_interest_1h_change_pct:+.1f}% expanding down)",
                open_interest_change=snapshot.open_interest_1h_change_pct,
                divergence_regime=regime,
            )

        # 3. Selling into massive whale accumulation / squeeze
        if side_norm == "SELL" and regime == "accumulation" and snapshot.top_traders_long_pct >= 0.70:
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"whale_accumulation_active (whales {snapshot.top_traders_long_pct*100:.1f}% long, OI_1h={snapshot.open_interest_1h_change_pct:+.1f}%)",
                open_interest_change=snapshot.open_interest_1h_change_pct,
                divergence_regime=regime,
                top_long_pct=snapshot.top_traders_long_pct,
            )

        return GuardDecision.allow(
            guard_name="WhaleDivergenceGuard",
            reason=f"whale_flow_consistent ({regime})",
            divergence_regime=regime,
            top_long_pct=snapshot.top_traders_long_pct,
        )
