"""Whale accumulation and distribution trigger.

Generates actionable entry/exit signals by monitoring:
- Top 20% whale trader positioning (long/short ratio)
- Aggressive taker buy/sell volume dominance
- Net Open Interest capital injection
"""

from __future__ import annotations

import time
from typing import Optional

from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.external.collectors.whale_positioning import (
    WhalePositioningCollector,
    WhalePositioningSnapshot,
)


class WhaleAccumulationTrigger:
    """Generates directional entries/exits when whales aggressively accumulate or distribute."""

    def __init__(
        self,
        collector: Optional[WhalePositioningCollector] = None,
        min_top_traders_long_pct: float = 0.60,    # Whales must be >= 60% long
        min_taker_buy_ratio: float = 1.25,         # Taker aggressive buys >= 1.25x sells
        min_top_traders_short_pct: float = 0.60,   # Whales must be >= 60% short for exit
        min_taker_sell_ratio: float = 0.80,        # Taker aggressive sells dominate (<0.80)
    ) -> None:
        self.collector = collector
        self.min_top_traders_long_pct = min_top_traders_long_pct
        self.min_taker_buy_ratio = min_taker_buy_ratio
        self.min_top_traders_short_pct = min_top_traders_short_pct
        self.min_taker_sell_ratio = min_taker_sell_ratio

    def evaluate(
        self,
        symbol: str,
        price: float,
        snapshot: Optional[WhalePositioningSnapshot] = None,
        now: Optional[float] = None,
    ) -> Optional[TriggerEvent]:
        """Evaluate if whale capital flow triggers an entry or exit."""
        t_now = now or time.time()
        if snapshot is None and self.collector is not None:
            snapshot = self.collector.fetch(symbol)

        if snapshot is None:
            return None

        # 1. Whale Accumulation Entry (BUY)
        # Whales are positioned long, taker buys dominate, and fresh capital entered (OI rising)
        is_whale_long = snapshot.top_traders_long_pct >= self.min_top_traders_long_pct
        is_aggressive_buying = snapshot.taker_buy_sell_ratio >= self.min_taker_buy_ratio
        is_capital_entering = snapshot.open_interest_1h_change_pct >= 0.0

        if is_whale_long and is_aggressive_buying and is_capital_entering:
            confidence = min(1.0, (snapshot.top_traders_long_pct - 0.5) * 2.0)
            return TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="WhaleAccumulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(snapshot.taker_buy_sell_ratio, 2),
                ts=t_now,
                confidence=round(confidence, 2),
                reason="whale_aggressive_accumulation",
                metadata={
                    "top_long_pct": snapshot.top_traders_long_pct,
                    "taker_ratio": snapshot.taker_buy_sell_ratio,
                    "oi_change_pct": snapshot.open_interest_1h_change_pct,
                    "regime": snapshot.divergence_regime,
                },
            )

        # 2. Whale Distribution Exit (SELL)
        # Whales are positioned short and taker sells dominate
        top_short_pct = 1.0 - snapshot.top_traders_long_pct
        is_whale_short = top_short_pct >= self.min_top_traders_short_pct
        is_aggressive_selling = snapshot.taker_buy_sell_ratio <= self.min_taker_sell_ratio

        if is_whale_short and is_aggressive_selling:
            confidence = min(1.0, (top_short_pct - 0.5) * 2.0)
            return TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="WhaleAccumulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(1.0 / max(0.01, snapshot.taker_buy_sell_ratio), 2),
                ts=t_now,
                confidence=round(confidence, 2),
                reason="whale_aggressive_distribution",
                metadata={
                    "top_short_pct": top_short_pct,
                    "taker_ratio": snapshot.taker_buy_sell_ratio,
                    "oi_change_pct": snapshot.open_interest_1h_change_pct,
                    "regime": snapshot.divergence_regime,
                },
            )

        return None
