"""Order book depth imbalance and whale limit wall guard.

Prevents buying directly into massive whale sell walls (resistance) or selling directly
into massive whale buy walls (support).
"""

from __future__ import annotations

from typing import Optional

from botcore import safe_nonneg_float, safe_pos_float
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    fetch_guard_snapshot,
    normalize_guard_side,
    snap_attr,
)
from intelligence.external.collectors.orderbook_depth import (
    OrderbookDepthCollector,
    OrderbookSnapshot,
)
from intelligence.macro_sentiment_ssot import clean_scale


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
        self.min_buy_imbalance = clean_scale(min_buy_imbalance, default=0.25, min_positive=True)
        self.max_sell_imbalance = clean_scale(max_sell_imbalance, default=0.75, min_positive=True)
        self.whale_wall_usd_limit = safe_pos_float(whale_wall_usd_limit, default=1_000_000.0) or 1_000_000.0

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[OrderbookSnapshot] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order book liquidity and limit walls."""
        side_norm = normalize_guard_side(side)
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow(
                guard_name="OrderbookWallGuard",
                reason="unsupported_side_allowed",
            )
        if snapshot is None and self.collector is not None:
            snapshot = fetch_guard_snapshot(self.collector, symbol, now=now)

        if snapshot is None:
            return GuardDecision.allow(
                guard_name="OrderbookWallGuard",
                reason="no_orderbook_snapshot",
            )

        raw_imb = snap_attr(snapshot, "imbalance_ratio")
        if raw_imb is None:
            raw_imb = snap_attr(snapshot, "ob_imbalance")
        imb = clean_scale(raw_imb, default=0.5, allow_percentage=True)
        raw_ask_usd = snap_attr(snapshot, "largest_ask_wall_usd")
        ask_wall_usd = safe_nonneg_float(
            raw_ask_usd if raw_ask_usd is not None else snap_attr(snapshot, "ask_wall_usd"),
            default=0.0,
        )
        raw_ask_px = snap_attr(snapshot, "largest_ask_wall_price")
        ask_wall_px = safe_nonneg_float(
            raw_ask_px if raw_ask_px is not None else snap_attr(snapshot, "ask_wall_px"),
            default=0.0,
        )
        raw_bid_usd = snap_attr(snapshot, "largest_bid_wall_usd")
        bid_wall_usd = safe_nonneg_float(
            raw_bid_usd if raw_bid_usd is not None else snap_attr(snapshot, "bid_wall_usd"),
            default=0.0,
        )
        raw_bid_px = snap_attr(snapshot, "largest_bid_wall_price")
        bid_wall_px = safe_nonneg_float(
            raw_bid_px if raw_bid_px is not None else snap_attr(snapshot, "bid_wall_px"),
            default=0.0,
        )

        # 1. Evaluate BUY side against ask walls and heavy ask imbalance
        if side_norm == "BUY":
            if imb < self.min_buy_imbalance:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"orderbook_heavily_ask_dominated (imbalance={imb:.2f} < {self.min_buy_imbalance:.2f})",
                    imbalance=imb,
                )
            if ask_wall_usd >= self.whale_wall_usd_limit:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"whale_sell_wall_blocking (${ask_wall_usd:,.0f} @ {ask_wall_px:.2f})",
                    wall_usd=ask_wall_usd,
                    wall_price=ask_wall_px,
                )

        # 2. Evaluate SELL side against bid walls and heavy bid imbalance
        if side_norm == "SELL":
            if imb > self.max_sell_imbalance:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"orderbook_heavily_bid_supported (imbalance={imb:.2f} > {self.max_sell_imbalance:.2f})",
                    imbalance=imb,
                )
            if bid_wall_usd >= self.whale_wall_usd_limit:
                return GuardDecision.defer(
                    guard_name="OrderbookWallGuard",
                    reason=f"whale_buy_wall_supporting (${bid_wall_usd:,.0f} @ {bid_wall_px:.2f})",
                    wall_usd=bid_wall_usd,
                    wall_price=bid_wall_px,
                )

        return GuardDecision.allow(
            guard_name="OrderbookWallGuard",
            reason=f"orderbook_liquidity_clear (imbalance={imb:.2f})",
            imbalance=imb,
        )
