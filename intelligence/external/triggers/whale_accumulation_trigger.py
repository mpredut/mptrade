"""Whale accumulation and distribution trigger.

Generates actionable entry/exit signals by monitoring:
- Top 20% whale trader positioning (long/short ratio)
- Aggressive taker buy/sell volume dominance
- Net Open Interest capital injection
"""

from __future__ import annotations

from typing import Optional

from botcore import safe_float, safe_pos_float
from intelligence.internal.guards.guard_decision import fetch_guard_snapshot, snap_attr
from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.external.collectors.whale_positioning import (
    WhalePositioningCollector,
    WhalePositioningSnapshot,
)
from intelligence.macro_sentiment_ssot import resolve_now_ts


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
        long_pct = safe_pos_float(min_top_traders_long_pct, default=0.60) or 0.60
        self.min_top_traders_long_pct = long_pct if long_pct <= 1.0 else 0.60
        self.min_taker_buy_ratio = safe_pos_float(min_taker_buy_ratio, default=1.25) or 1.25
        short_pct = safe_pos_float(min_top_traders_short_pct, default=0.60) or 0.60
        self.min_top_traders_short_pct = short_pct if short_pct <= 1.0 else 0.60
        self.min_taker_sell_ratio = safe_pos_float(min_taker_sell_ratio, default=0.80) or 0.80

    def evaluate(
        self,
        symbol: str,
        price: float,
        snapshot: Optional[WhalePositioningSnapshot] = None,
        now: Optional[float] = None,
    ) -> Optional[TriggerEvent]:
        """Evaluate if whale capital flow triggers an entry or exit."""
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return None
        price = price_f

        t_now = resolve_now_ts(now)
        if snapshot is None and self.collector is not None:
            snapshot = fetch_guard_snapshot(self.collector, symbol, now=t_now if now is not None else None)

        if snapshot is None:
            return None

        top_long_pct = safe_float(snap_attr(snapshot, "top_traders_long_pct", None), default=None)
        taker_ratio = safe_float(snap_attr(snapshot, "taker_buy_sell_ratio", None), default=None)
        oi_chg = safe_float(snap_attr(snapshot, "open_interest_1h_change_pct", None), default=None)
        raw_regime = snap_attr(snapshot, "divergence_regime", "neutral")
        regime_str = raw_regime.strip() if isinstance(raw_regime, str) and raw_regime.strip() else "neutral"
        if top_long_pct is None or taker_ratio is None or oi_chg is None:
            return None

        # 1. Whale Accumulation Entry (BUY)
        # Whales are positioned long, taker buys dominate, and fresh capital entered (OI rising)
        is_whale_long = top_long_pct >= self.min_top_traders_long_pct
        is_aggressive_buying = taker_ratio >= self.min_taker_buy_ratio
        is_capital_entering = oi_chg >= 0.0

        if is_whale_long and is_aggressive_buying and is_capital_entering:
            confidence = max(0.0, min(1.0, (top_long_pct - 0.5) * 2.0))
            return TriggerEvent.directional(
                True,
                source="WhaleAccumulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(taker_ratio, 2),
                ts=t_now,
                confidence=round(confidence, 2),
                reason_bull="whale_aggressive_accumulation",
                reason_bear="",
                metadata={
                    "top_long_pct": top_long_pct,
                    "taker_ratio": taker_ratio,
                    "oi_change_pct": oi_chg,
                    "regime": regime_str,
                },
            )

        # 2. Whale Distribution Exit (SELL)
        # Whales are positioned short and taker sells dominate
        top_short_pct = 1.0 - top_long_pct
        is_whale_short = top_short_pct >= self.min_top_traders_short_pct
        is_aggressive_selling = taker_ratio <= self.min_taker_sell_ratio

        if is_whale_short and is_aggressive_selling:
            confidence = max(0.0, min(1.0, (top_short_pct - 0.5) * 2.0))
            return TriggerEvent.directional(
                False,
                source="WhaleAccumulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(1.0 / max(0.01, taker_ratio), 2),
                ts=t_now,
                confidence=round(confidence, 2),
                reason_bull="",
                reason_bear="whale_aggressive_distribution",
                metadata={
                    "top_short_pct": top_short_pct,
                    "taker_ratio": taker_ratio,
                    "oi_change_pct": oi_chg,
                    "regime": regime_str,
                },
            )

        return None
