"""Liquidation capitulation and short squeeze trigger.

Identifies high-probability reversal points triggered by forced liquidations:
- Panic capitulation: Long liquidations dominate (forced selling cascade), signaling dip-buy entry.
- Short squeeze: Short liquidations dominate (forced buying cascade), signaling take-profit exit.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.external.collectors.bybit_liquidations import (
    BybitLiquidationCollector,
    LiquidationSummary,
)


class LiquidationCapitulationTrigger:
    """Generates actionable entry/exit signals from derivatives liquidation bursts."""

    def __init__(
        self,
        collector: Optional[BybitLiquidationCollector] = None,
        min_notional_usd: float = 100_000.0,
        ratio_threshold: float = 0.75,
        window_sec: float = 300.0,
    ) -> None:
        self.collector = collector
        self.min_notional_usd = min_notional_usd
        self.ratio_threshold = ratio_threshold
        self.window_sec = window_sec

    def evaluate(
        self,
        symbol: str,
        price: float,
        summary: Optional[LiquidationSummary] = None,
        now: Optional[float] = None,
    ) -> Optional[TriggerEvent]:
        """Evaluate current liquidation window for capitulation buy or squeeze exit."""
        if price is None or isinstance(price, bool):
            return None
        try:
            price_f = float(price)
        except (TypeError, ValueError):
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
            except (TypeError, ValueError):
                pass
        if summary is None and self.collector is not None:
            try:
                summary = (
                    self.collector.get_summary(symbol, window_sec=self.window_sec, now=t_now)
                    if now is not None
                    else self.collector.get_summary(symbol, window_sec=self.window_sec)
                )
            except TypeError:
                summary = self.collector.get_summary(symbol, window_sec=self.window_sec)

        if (
            summary is None
            or not math.isfinite(summary.total_usd)
            or not math.isfinite(summary.long_liq_usd)
            or not math.isfinite(summary.short_liq_usd)
            or not math.isfinite(summary.capitulation_ratio)
            or not math.isfinite(summary.squeeze_ratio)
            or summary.total_usd < self.min_notional_usd
        ):
            return None

        # 1. Long capitulation burst -> BUY entry
        if summary.capitulation_ratio >= self.ratio_threshold and summary.long_liq_usd >= self.min_notional_usd:
            return TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="LiquidationCapitulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(summary.capitulation_ratio, 2),
                ts=t_now,
                confidence=round(min(1.0, summary.long_liq_usd / (self.min_notional_usd * 2.0)), 2),
                reason="liquidation_capitulation_burst",
                metadata={
                    "long_liq_usd": summary.long_liq_usd,
                    "short_liq_usd": summary.short_liq_usd,
                    "capitulation_ratio": summary.capitulation_ratio,
                    "window_sec": summary.window_sec,
                },
            )

        # 2. Short squeeze burst -> SELL exit
        if summary.squeeze_ratio >= self.ratio_threshold and summary.short_liq_usd >= self.min_notional_usd:
            return TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="LiquidationCapitulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(summary.squeeze_ratio, 2),
                ts=t_now,
                confidence=round(min(1.0, summary.short_liq_usd / (self.min_notional_usd * 2.0)), 2),
                reason="liquidation_short_squeeze_burst",
                metadata={
                    "long_liq_usd": summary.long_liq_usd,
                    "short_liq_usd": summary.short_liq_usd,
                    "squeeze_ratio": summary.squeeze_ratio,
                    "window_sec": summary.window_sec,
                },
            )

        return None
