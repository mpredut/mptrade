"""Market breadth capitulation bounce and blowoff exhaustion trigger."""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.macro_sentiment_ssot import clean_scale, obj_attr, resolve_now_ts, safe_float, safe_pos_float
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
        self.washout_advance_threshold = clean_scale(washout_advance_threshold, 0.15)
        self.washout_median_threshold = safe_float(washout_median_threshold, -4.0)
        self.blowoff_advance_threshold = clean_scale(blowoff_advance_threshold, 0.88)
        self.blowoff_median_threshold = safe_float(blowoff_median_threshold, 6.0)
        if self.washout_advance_threshold > self.blowoff_advance_threshold:
            self.washout_advance_threshold, self.blowoff_advance_threshold = (
                self.blowoff_advance_threshold,
                self.washout_advance_threshold,
            )
        if self.washout_median_threshold > self.blowoff_median_threshold:
            self.washout_median_threshold, self.blowoff_median_threshold = (
                self.blowoff_median_threshold,
                self.washout_median_threshold,
            )

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
        price_f = safe_pos_float(price, 0.0)
        if snapshot is None or price_f <= 0.0:
            return None, None
        adv_r = safe_float(obj_attr(snapshot, "advance_ratio", None), float("nan"))
        med_c = safe_float(obj_attr(snapshot, "median_change_pct", None), float("nan"))
        if not (math.isfinite(adv_r) and math.isfinite(med_c)):
            return None, None
        raw_regime = obj_attr(snapshot, "regime", None)
        regime = str(raw_regime).strip() if (isinstance(raw_regime, str) and raw_regime.strip()) else "NEUTRAL"
        price = price_f
        now = resolve_now_ts(ts)

        # 1. Capitulation Washout Bounce Entry
        if (
            adv_r <= self.washout_advance_threshold
            and med_c <= self.washout_median_threshold
        ):
            a24 = safe_float(asset_24h_change_pct, float("nan"))
            relative_strength = bool(math.isfinite(a24) and a24 > med_c)

            strength = round(min(0.95, 0.70 + (1.0 - adv_r) * 0.25), 2)
            event = TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="market_breadth_washout",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"market_washout_capitulation_bounce (adv_ratio={adv_r:.2f}, median={med_c:.1f}%)",
                metadata={
                    "advance_ratio": adv_r,
                    "median_change_pct": med_c,
                    "relative_strength": relative_strength,
                    "regime": regime,
                },
            )
            return "BREADTH_WASHOUT_BUY", event

        # 2. Overheated Blowoff Distribution Exit
        if (
            adv_r >= self.blowoff_advance_threshold
            and med_c >= self.blowoff_median_threshold
        ):
            strength = round(min(0.95, 0.70 + adv_r * 0.25), 2)
            event = TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="market_breadth_blowoff",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"market_blowoff_overheated_exhaustion (adv_ratio={adv_r:.2f}, median={med_c:.1f}%)",
                metadata={
                    "advance_ratio": adv_r,
                    "median_change_pct": med_c,
                    "regime": regime,
                },
            )
            return "BREADTH_BLOWOFF_SELL", event

        return None, None
