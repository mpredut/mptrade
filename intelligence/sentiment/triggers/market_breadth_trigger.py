"""Market breadth capitulation bounce and blowoff exhaustion trigger."""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger("intelligence.sentiment.market_breadth_trigger")


class MarketBreadthTrigger:
    """Emits directional triggers based on cross-asset market participation extremes.

    - ENTRY BUY: Triggered on capitulation washouts (advance_ratio <= 0.15, median_change <= -4.0%) when relative strength appears.
    - EXIT SELL: Triggered on euphoric blowoffs (advance_ratio >= 0.88, median_change >= 6.0%) to trim into overheating rallies.
    """

    def __init__(
        self,
        washout_advance_threshold: float = 0.15,
        washout_median_threshold: float = -4.0,
        blowoff_advance_threshold: float = 0.88,
        blowoff_median_threshold: float = 6.0,
    ) -> None:
        self.washout_advance_threshold = washout_advance_threshold
        self.washout_median_threshold = washout_median_threshold
        self.blowoff_advance_threshold = blowoff_advance_threshold
        self.blowoff_median_threshold = blowoff_median_threshold

    def evaluate(
        self,
        symbol: str,
        price: float,
        snapshot: Optional[MarketBreadthSnapshot],
        *,
        asset_24h_change_pct: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[TriggerEvent]]:
        """Evaluate market breadth snapshot and generate actionable triggers."""
        if snapshot is None:
            return None, None

        now = ts if ts is not None else time.time()

        # 1. Capitulation Washout Bounce Entry
        if (
            snapshot.advance_ratio <= self.washout_advance_threshold
            and snapshot.median_change_pct <= self.washout_median_threshold
        ):
            # If asset is outperforming the bleeding median (relative strength leader)
            relative_strength = False
            if asset_24h_change_pct is not None and asset_24h_change_pct > snapshot.median_change_pct:
                relative_strength = True

            strength = round(min(0.95, 0.70 + (1.0 - snapshot.advance_ratio) * 0.25), 2)
            event = TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="market_breadth_washout",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"market_washout_capitulation_bounce (adv_ratio={snapshot.advance_ratio:.2f}, median={snapshot.median_change_pct:.1f}%)",
                metadata={
                    "advance_ratio": snapshot.advance_ratio,
                    "median_change_pct": snapshot.median_change_pct,
                    "relative_strength": relative_strength,
                    "regime": snapshot.regime,
                },
            )
            return "BREADTH_WASHOUT_BUY", event

        # 2. Overheated Blowoff Distribution Exit
        if (
            snapshot.advance_ratio >= self.blowoff_advance_threshold
            and snapshot.median_change_pct >= self.blowoff_median_threshold
        ):
            strength = round(min(0.95, 0.70 + snapshot.advance_ratio * 0.25), 2)
            event = TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="market_breadth_blowoff",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"market_blowoff_overheated_exhaustion (adv_ratio={snapshot.advance_ratio:.2f}, median={snapshot.median_change_pct:.1f}%)",
                metadata={
                    "advance_ratio": snapshot.advance_ratio,
                    "median_change_pct": snapshot.median_change_pct,
                    "regime": snapshot.regime,
                },
            )
            return "BREADTH_BLOWOFF_SELL", event

        return None, None
