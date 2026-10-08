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
        block_liquidation_cascade: bool = True,
        severe_cascade_oi_pct: float = -5.0,
        min_taker_ratio: float = 0.65,
    ) -> None:
        self.collector = collector
        self.block_short_covering_fakeout = block_short_covering_fakeout
        self.block_aggressive_shorting = block_aggressive_shorting
        self.block_liquidation_cascade = block_liquidation_cascade
        self.severe_cascade_oi_pct = severe_cascade_oi_pct
        self.min_taker_ratio = min_taker_ratio

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

        # 3. Buying into active severe liquidation cascade (falling price, OI crashing <= severe_cascade_oi_pct, weak taker buy)
        if (
            side_norm == "BUY"
            and self.block_liquidation_cascade
            and regime == "long_liquidation"
            and snapshot.open_interest_1h_change_pct <= self.severe_cascade_oi_pct
            and snapshot.taker_buy_sell_ratio < 0.85
        ):
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"liquidation_cascade_active (OI_1h={snapshot.open_interest_1h_change_pct:+.1f}%, taker_ratio={snapshot.taker_buy_sell_ratio:.2f})",
                open_interest_change=snapshot.open_interest_1h_change_pct,
                divergence_regime=regime,
                taker_ratio=snapshot.taker_buy_sell_ratio,
            )

        # 4. Severe taker selling panic (< min_taker_ratio, heavy market dumping)
        if side_norm == "BUY" and snapshot.taker_buy_sell_ratio < self.min_taker_ratio:
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"severe_taker_selling_dominated (taker_ratio={snapshot.taker_buy_sell_ratio:.2f} < {self.min_taker_ratio:.2f})",
                taker_ratio=snapshot.taker_buy_sell_ratio,
                divergence_regime=regime,
            )

        # 5. Selling into massive whale accumulation / squeeze
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
