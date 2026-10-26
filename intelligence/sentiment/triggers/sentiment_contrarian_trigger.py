"""Contrarian sentiment trigger exploiting extreme market psychology mispricings."""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
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
        def _safe_thresh(v, default: int) -> int:
            if v is None or isinstance(v, bool):
                return default
            try:
                f = float(v)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(f) or not (0.0 <= f <= 100.0):
                return default
            return int(round(f))

        self.extreme_fear_entry = _safe_thresh(extreme_fear_entry, 22)
        self.extreme_greed_exit = _safe_thresh(extreme_greed_exit, 85)

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
        if snapshot is None or price is None or isinstance(price, bool):
            return None, None
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            return None, None
        if not (math.isfinite(price_f) and price_f > 0.0):
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
        raw_val = getattr(snapshot, "value", None)
        if raw_val is None or isinstance(raw_val, bool):
            return None, None
        try:
            val_f = float(raw_val)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(val_f):
            return None, None
        val = int(round(val_f)) if float(val_f).is_integer() else val_f
        cls_label = getattr(snapshot, "classification", None) or "Unknown"
        trend_7d = getattr(snapshot, "trend_7d_change", None)

        # 1. Contrarian ENTRY BUY during Extreme Fear
        if val <= self.extreme_fear_entry:
            # Check for stabilization if history exists: avoid buying if sentiment is in free-fall (>10 pt drop today)
            history = getattr(snapshot, "historical_values", None)
            if isinstance(history, (list, tuple)) and len(history) >= 2:
                h0, h1 = history[0], history[1]
                if (
                    h0 is not None
                    and h1 is not None
                    and not isinstance(h0, bool)
                    and not isinstance(h1, bool)
                ):
                    try:
                        recent_drop = float(h1) - float(h0)
                        if math.isfinite(recent_drop) and recent_drop > 10:
                            # Still in sharp acceleration of fear, wait for stabilization
                            return None, None
                    except (TypeError, ValueError, OverflowError):
                        pass

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
