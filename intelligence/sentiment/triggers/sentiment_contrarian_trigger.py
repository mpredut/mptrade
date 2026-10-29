"""Contrarian sentiment trigger exploiting extreme market psychology mispricings."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.macro_sentiment_ssot import (
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_pos_float,
    safe_thresh as _safe_thresh,
)
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot

logger = logging.getLogger("intelligence.sentiment.sentiment_contrarian_trigger")


class SentimentContrarianTrigger:
    """Generates contrarian entry signals during Extreme Fear and exit signals during Extreme Greed.

    - ENTRY BUY: Triggered when Fear & Greed <= extreme_fear_entry (default 22) and sentiment has bottomed/stabilized.
    - EXIT SELL: Triggered when Fear & Greed >= extreme_greed_exit (default 85) to take profit in peak euphoria.
    """

    def __init__(
        self,
        extreme_fear_entry: int = 22,
        extreme_greed_exit: int = 85,
    ) -> None:
        self.extreme_fear_entry = _safe_thresh(extreme_fear_entry, 22)
        self.extreme_greed_exit = _safe_thresh(extreme_greed_exit, 85)
        if self.extreme_fear_entry > self.extreme_greed_exit:
            self.extreme_fear_entry, self.extreme_greed_exit = self.extreme_greed_exit, self.extreme_fear_entry

    def evaluate(
        self,
        symbol: str,
        price: float,
        snapshot: Optional[FearGreedSnapshot],
        *,
        ts: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[TriggerEvent]]:
        """Evaluate Fear & Greed snapshot and emit directional trigger events.

        Returns (signal_type, Optional[TriggerEvent]).
        """
        price_f = safe_pos_float(price, 0.0)
        if snapshot is None or price_f <= 0.0:
            return None, None
        val_f = safe_float(obj_attr(snapshot, "value", None), default=None)
        if val_f is None:
            return None, None
        price = price_f
        now = resolve_now_ts(ts)
        val = int(round(val_f)) if float(val_f).is_integer() else val_f
        raw_cls = obj_attr(snapshot, "classification", None)
        cls_label = str(raw_cls).strip() if (isinstance(raw_cls, str) and raw_cls.strip()) else "Unknown"
        trend_7d = obj_attr(snapshot, "trend_7d_change", None)

        # 1. Contrarian ENTRY BUY during Extreme Fear
        if val <= self.extreme_fear_entry:
            history = obj_attr(snapshot, "historical_values", None)
            if isinstance(history, (list, tuple)) and len(history) >= 2:
                h0 = safe_float(history[0], default=None)
                h1 = safe_float(history[1], default=None)
                if h0 is not None and h1 is not None and (h1 - h0) > 10:
                    return None, None

            strength = round(max(0.60, min(1.0, 1.0 - (val / 100.0))), 2)
            event = TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="sentiment_contrarian",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"extreme_fear_contrarian_dip_buy (F&G={val}, {cls_label})",
                metadata={
                    "fear_greed_value": val,
                    "fear_greed_classification": cls_label,
                    "trend_7d_change": trend_7d,
                },
            )
            return "CONTRARIAN_BUY", event

        # 2. Contrarian EXIT SELL during Extreme Greed
        if val >= self.extreme_greed_exit:
            strength = round(max(0.70, min(1.0, val / 100.0)), 2)
            event = TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="sentiment_contrarian",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"extreme_greed_contrarian_distribution (F&G={val}, {cls_label})",
                metadata={
                    "fear_greed_value": val,
                    "fear_greed_classification": cls_label,
                    "trend_7d_change": trend_7d,
                },
            )
            return "CONTRARIAN_SELL", event

        return None, None
