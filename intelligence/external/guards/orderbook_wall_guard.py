"""Order book depth imbalance and whale limit wall guard.

Prevents buying directly into massive whale sell walls (resistance) or selling directly
into massive whale buy walls (support).
"""

from __future__ import annotations

import math
from typing import Any, Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.orderbook_depth import (
    OrderbookDepthCollector,
    OrderbookSnapshot,
)


def _safe_float(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


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
        min_buy = _safe_float(min_buy_imbalance, 0.25)
        self.min_buy_imbalance = min_buy if 0.0 <= min_buy <= 1.0 else 0.25
        max_sell = _safe_float(max_sell_imbalance, 0.75)
        self.max_sell_imbalance = max_sell if 0.0 <= max_sell <= 1.0 else 0.75
        wall_lim = _safe_float(whale_wall_usd_limit, 1_000_000.0)
        self.whale_wall_usd_limit = wall_lim if wall_lim > 0.0 else 1_000_000.0

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[OrderbookSnapshot] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order book liquidity and limit walls."""
        side_norm = str(side or "").strip().upper() if isinstance(side, str) else ""
        if side_norm in ("B", "BUY", "BID", "LONG"):
            side_norm = "BUY"
        elif side_norm in ("S", "SELL", "ASK", "A", "SHORT"):
            side_norm = "SELL"
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow(
                guard_name="OrderbookWallGuard",
                reason="unsupported_side_allowed",
            )
        if snapshot is None and self.collector is not None:
            try:
                snapshot = self.collector.fetch(symbol, now=now) if now is not None else self.collector.fetch(symbol)
            except TypeError:
                snapshot = self.collector.fetch(symbol)

        if snapshot is None:
            return GuardDecision.allow(
                guard_name="OrderbookWallGuard",
                reason="no_orderbook_snapshot",
            )

        imb = _safe_float(getattr(snapshot, "imbalance_ratio", None), 0.5)
        ask_wall_usd = _safe_float(getattr(snapshot, "largest_ask_wall_usd", None), 0.0)
        ask_wall_px = _safe_float(getattr(snapshot, "largest_ask_wall_price", None), 0.0)
        bid_wall_usd = _safe_float(getattr(snapshot, "largest_bid_wall_usd", None), 0.0)
        bid_wall_px = _safe_float(getattr(snapshot, "largest_bid_wall_price", None), 0.0)

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
