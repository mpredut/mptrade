"""Panic washout guard protecting against catching falling knives during macro panics."""
from __future__ import annotations

import logging
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger("intelligence.sentiment.panic_washout_guard")


class PanicWashoutGuard:
    """Guards against uncontained macro panics and falling knives.

    During market-wide liquidation storms or accelerating panic drops:
    - Defers or downscales standard BUY orders until selling decelerates.
    - Permits SELL / stop-loss execution without interference.
    """

    def __init__(
        self,
        severe_panic_fng_threshold: int = 12,
        panic_advance_threshold: float = 0.10,
        downscale_factor: float = 0.50,
    ) -> None:
        self.severe_panic_fng_threshold = severe_panic_fng_threshold
        self.panic_advance_threshold = panic_advance_threshold
        self.downscale_factor = downscale_factor

    def check(
        self,
        symbol: str,
        side: str,
        *,
        breadth_snapshot: Optional[MarketBreadthSnapshot] = None,
        fear_greed_snapshot: Optional[FearGreedSnapshot] = None,
        asset_24h_change_pct: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order against macro panic and market-wide capitulation."""
        # Only guard BUY orders; stop-losses or exits are never impeded
        if side.upper() != "BUY":
            return GuardDecision.allow("PanicWashoutGuard", "sell_permitted_in_panic")

        # 1. Check severe accelerating Fear & Greed free-fall
        if fear_greed_snapshot is not None:
            val = fear_greed_snapshot.value
            history = fear_greed_snapshot.historical_values
            if val <= self.severe_panic_fng_threshold:
                # If sentiment plummeted sharply (e.g. > 8 pt drop from previous day)
                if len(history) >= 2 and (history[1] - history[0]) > 8:
                    return GuardDecision.defer(
                        "PanicWashoutGuard",
                        f"defer_accelerating_panic_capitulation (F&G={val}, 1d drop={history[1] - history[0]})",
                        fear_greed_value=val,
                    )

        # 2. Check Market Breadth Washout
        if breadth_snapshot is not None:
            if (
                breadth_snapshot.advance_ratio <= self.panic_advance_threshold
                and breadth_snapshot.median_change_pct <= -5.0
            ):
                # If asset is also plunging heavily (> 7% drop), defer entry
                if asset_24h_change_pct is not None and asset_24h_change_pct <= -7.0:
                    return GuardDecision.defer(
                        "PanicWashoutGuard",
                        f"defer_market_panic_washout (adv_ratio={breadth_snapshot.advance_ratio:.2f}, asset_change={asset_24h_change_pct:.1f}%)",
                        advance_ratio=breadth_snapshot.advance_ratio,
                    )
                # Otherwise, allow with reduced risk allocation
                return GuardDecision.downscale(
                    "PanicWashoutGuard",
                    scale=self.downscale_factor,
                    reason=f"downscale_market_panic_washout (adv_ratio={breadth_snapshot.advance_ratio:.2f})",
                    advance_ratio=breadth_snapshot.advance_ratio,
                )

        return GuardDecision.allow("PanicWashoutGuard", "macro_conditions_stable")
