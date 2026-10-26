"""Market breadth capitulation bounce and blowoff exhaustion trigger."""
from __future__ import annotations

import logging
import math
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
        if snapshot is None or price is None or isinstance(price, bool):
            return None, None
        try:
            price_f = float(price)
            adv_r = float(snapshot.advance_ratio)
            med_c = float(snapshot.median_change_pct)
        except (TypeError, ValueError):
            return None, None
        if (
            isinstance(snapshot.advance_ratio, bool)
            or isinstance(snapshot.median_change_pct, bool)
            or not (math.isfinite(price_f) and price_f > 0.0 and math.isfinite(adv_r) and math.isfinite(med_c))
        ):
            return None, None
        price = price_f

        now = time.time()
        if ts is not None and not isinstance(ts, bool):
            try:
                tf = float(ts)
                if math.isfinite(tf) and tf > 0.0:
                    now = tf
            except (TypeError, ValueError):
                pass

        # 1. Capitulation Washout Bounce Entry
        if (
            adv_r <= self.washout_advance_threshold
            and med_c <= self.washout_median_threshold
        ):
            # If asset is outperforming the bleeding median (relative strength leader)
            relative_strength = False
            if asset_24h_change_pct is not None and not isinstance(asset_24h_change_pct, bool):
                try:
                    a24 = float(asset_24h_change_pct)
                    if math.isfinite(a24) and a24 > med_c:
                        relative_strength = True
                except (TypeError, ValueError):
                    pass

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
