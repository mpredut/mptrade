"""Liquidation capitulation and short squeeze trigger.

Identifies high-probability reversal points triggered by forced liquidations:
- Panic capitulation: Long liquidations dominate (forced selling cascade), signaling dip-buy entry.
- Short squeeze: Short liquidations dominate (forced buying cascade), signaling take-profit exit.
"""

from __future__ import annotations

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
        t_now = now or time.time()
        if summary is None and self.collector is not None:
            summary = self.collector.get_summary(symbol, window_sec=self.window_sec)

        if summary is None or summary.total_usd < self.min_notional_usd:
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
