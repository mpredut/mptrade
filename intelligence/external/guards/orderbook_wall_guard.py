"""Order book depth imbalance and whale limit wall guard.

Prevents buying directly into massive whale sell walls (resistance) or selling directly
into massive whale buy walls (support).
"""

from __future__ import annotations

from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.orderbook_depth import (
    OrderbookDepthCollector,
    OrderbookSnapshot,
)


class OrderbookWallGuard:
    """Defers or downscales orders when order book depth or whale walls oppose the trade."""

    def __init__(
        self,
        collector: Optional[OrderbookDepthCollector] = None,
        min_buy_imbalance: float = 0.25,     # Below 0.25 (asks 3:1 over bids) is unfavorable for BUY
        max_sell_imbalance: float = 0.75,    # Above 0.75 (bids 3:1 over asks) is unfavorable for SELL
        whale_wall_usd_limit: float = 1_000_000.0,
    ) -> None:
        self.collector = collector
        self.min_buy_imbalance = min_buy_imbalance
        self.max_sell_imbalance = max_sell_imbalance
        self.whale_wall_usd_limit = whale_wall_usd_limit

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[OrderbookSnapshot] = None,
    ) -> GuardDecision:
        """Evaluate order book liquidity and limit walls."""
        side_norm = side.upper()
        if snapshot is None and self.collector is not None:
            snapshot = self.collector.fetch(symbol)

        if snapshot is None:
            return GuardDecision.allow(
                guard_name="OrderbookWallGuard",
                reason="no_orderbook_snapshot",
            )

        imb = snapshot.imbalance_ratio

        # 1. Evaluate BUY side against ask walls and heavy ask imbalance
        if side_norm == "BUY":
            if imb < self.min_buy_imbalance:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"orderbook_heavily_ask_dominated (imbalance={imb:.2f} < {self.min_buy_imbalance:.2f})",
                    imbalance=imb,
                )
            if snapshot.largest_ask_wall_usd >= self.whale_wall_usd_limit:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"whale_sell_wall_blocking (${snapshot.largest_ask_wall_usd:,.0f} @ {snapshot.largest_ask_wall_price:.2f})",
                    wall_usd=snapshot.largest_ask_wall_usd,
                    wall_price=snapshot.largest_ask_wall_price,
                )

        # 2. Evaluate SELL side against bid walls and heavy bid imbalance
        if side_norm == "SELL":
            if imb > self.max_sell_imbalance:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"orderbook_heavily_bid_supported (imbalance={imb:.2f} > {self.max_sell_imbalance:.2f})",
                    imbalance=imb,
                )
            if snapshot.largest_bid_wall_usd >= self.whale_wall_usd_limit:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"whale_buy_wall_supporting (${snapshot.largest_bid_wall_usd:,.0f} @ {snapshot.largest_bid_wall_price:.2f})",
                    wall_usd=snapshot.largest_bid_wall_usd,
                    wall_price=snapshot.largest_bid_wall_price,
                )

        return GuardDecision.allow(
            guard_name="OrderbookWallGuard",
            reason=f"orderbook_liquidity_clear (imbalance={imb:.2f})",
            imbalance=imb,
        )
