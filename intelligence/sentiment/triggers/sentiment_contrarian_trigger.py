"""Contrarian sentiment trigger exploiting extreme market psychology mispricings."""
from __future__ import annotations

from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerEvent
from botcore import safe_thresh
from intelligence.macro_sentiment_ssot import (
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot


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
        self.extreme_fear_entry = safe_thresh(extreme_fear_entry, 22)
        self.extreme_greed_exit = safe_thresh(extreme_greed_exit, 85)
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
        if snapshot is None or (price := safe_pos_float(price, default=None)) is None:
            return None, None
        extracted = FearGreedSnapshot.extract_value_and_label(
            snapshot, preserve_float=True
        )
        if extracted is None:
            return None, None
        val, cls_label = extracted
        now = resolve_now_ts(ts)
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
            event = TriggerEvent.directional(
                True,
                source="sentiment_contrarian",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason_bull=f"extreme_fear_contrarian_dip_buy (F&G={val}, {cls_label})",
                reason_bear="",
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
            event = TriggerEvent.directional(
                False,
                source="sentiment_contrarian",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason_bull="",
                reason_bear=f"extreme_greed_contrarian_distribution (F&G={val}, {cls_label})",
                metadata={
                    "fear_greed_value": val,
                    "fear_greed_classification": cls_label,
                    "trend_7d_change": trend_7d,
                },
            )
            return "CONTRARIAN_SELL", event

        return None, None
