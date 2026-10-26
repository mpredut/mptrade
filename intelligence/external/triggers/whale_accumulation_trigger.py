"""Whale accumulation and distribution trigger.

Generates actionable entry/exit signals by monitoring:
- Top 20% whale trader positioning (long/short ratio)
- Aggressive taker buy/sell volume dominance
- Net Open Interest capital injection
"""

from __future__ import annotations

import math
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
        if price is None or isinstance(price, bool):
            return None
        try:
            price_f = float(price)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(price_f) and price_f > 0.0):
            return None
        price = price_f

        t_now = time.time()
        if now is not None and not isinstance(now, bool):
            try:
                nf = float(now)
                if math.isfinite(nf) and nf > 0.0:
                    t_now = nf
            except (TypeError, ValueError, OverflowError):
                pass
        if snapshot is None and self.collector is not None:
            try:
                snapshot = self.collector.fetch(symbol, now=t_now) if now is not None else self.collector.fetch(symbol)
            except TypeError:
                snapshot = self.collector.fetch(symbol)

        if snapshot is None:
            return None

        raw_long_pct = getattr(snapshot, "top_traders_long_pct", None)
        raw_taker_ratio = getattr(snapshot, "taker_buy_sell_ratio", None)
        raw_oi_chg = getattr(snapshot, "open_interest_1h_change_pct", None)
        if (
            raw_long_pct is None
            or raw_taker_ratio is None
            or raw_oi_chg is None
            or isinstance(raw_long_pct, bool)
            or isinstance(raw_taker_ratio, bool)
            or isinstance(raw_oi_chg, bool)
        ):
            return None
        try:
            top_long_pct = float(raw_long_pct)
            taker_ratio = float(raw_taker_ratio)
            oi_chg = float(raw_oi_chg)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(top_long_pct) and math.isfinite(taker_ratio) and math.isfinite(oi_chg)):
            return None

        # 1. Whale Accumulation Entry (BUY)
        # Whales are positioned long, taker buys dominate, and fresh capital entered (OI rising)
        is_whale_long = top_long_pct >= self.min_top_traders_long_pct
        is_aggressive_buying = taker_ratio >= self.min_taker_buy_ratio
        is_capital_entering = oi_chg >= 0.0

        if is_whale_long and is_aggressive_buying and is_capital_entering:
            confidence = min(1.0, (top_long_pct - 0.5) * 2.0)
            return TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="WhaleAccumulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(taker_ratio, 2),
                ts=t_now,
                confidence=round(confidence, 2),
                reason="whale_aggressive_accumulation",
                metadata={
                    "top_long_pct": top_long_pct,
                    "taker_ratio": taker_ratio,
                    "oi_change_pct": oi_chg,
                    "regime": snapshot.divergence_regime,
                },
            )

        # 2. Whale Distribution Exit (SELL)
        # Whales are positioned short and taker sells dominate
        top_short_pct = 1.0 - top_long_pct
        is_whale_short = top_short_pct >= self.min_top_traders_short_pct
        is_aggressive_selling = taker_ratio <= self.min_taker_sell_ratio

        if is_whale_short and is_aggressive_selling:
            confidence = min(1.0, (top_short_pct - 0.5) * 2.0)
            return TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="WhaleAccumulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(1.0 / max(0.01, taker_ratio), 2),
                ts=t_now,
                confidence=round(confidence, 2),
                reason="whale_aggressive_distribution",
                metadata={
                    "top_short_pct": top_short_pct,
                    "taker_ratio": taker_ratio,
                    "oi_change_pct": oi_chg,
                    "regime": snapshot.divergence_regime,
                },
            )

        return None
