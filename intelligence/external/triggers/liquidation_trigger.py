"""Liquidation capitulation and short squeeze trigger.

Identifies high-probability reversal points triggered by forced liquidations:
- Panic capitulation: Long liquidations dominate (forced selling cascade), signaling dip-buy entry.
- Short squeeze: Short liquidations dominate (forced buying cascade), signaling take-profit exit.
"""

from __future__ import annotations

from typing import Any, Optional

from botcore import safe_float, safe_pos_float
from intelligence.internal.guards.guard_decision import snap_attr
from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.external.collectors.bybit_liquidations import (
    BybitLiquidationCollector,
    LiquidationSummary,
)
from intelligence.macro_sentiment_ssot import resolve_now_ts


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
        self.min_notional_usd = safe_pos_float(min_notional_usd, default=100_000.0) or 100_000.0
        rt = safe_pos_float(ratio_threshold, default=0.75) or 0.75
        self.ratio_threshold = rt if rt <= 1.0 else 0.75
        self.window_sec = safe_pos_float(window_sec, default=300.0) or 300.0

    def evaluate(
        self,
        symbol: str,
        price: float,
        summary: Optional[LiquidationSummary] = None,
        now: Optional[float] = None,
    ) -> Optional[TriggerEvent]:
        """Evaluate current liquidation window for capitulation buy or squeeze exit."""
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return None
        price = price_f

        t_now = resolve_now_ts(now)
        if summary is None and self.collector is not None:
            try:
                summary = (
                    self.collector.get_summary(symbol, window_sec=self.window_sec, now=t_now)
                    if now is not None
                    else self.collector.get_summary(symbol, window_sec=self.window_sec)
                )
            except TypeError:
                summary = self.collector.get_summary(symbol, window_sec=self.window_sec)

        if summary is None:
            return None

        long_liq = safe_float(snap_attr(summary, "long_liq_usd", None), default=None)
        short_liq = safe_float(snap_attr(summary, "short_liq_usd", None), default=None)
        total_usd = safe_float(snap_attr(summary, "total_usd", None), default=None)
        if isinstance(summary, dict) and "total_usd" not in summary and long_liq is not None and short_liq is not None:
            total_usd = long_liq + short_liq
        cap_ratio = safe_float(snap_attr(summary, "capitulation_ratio", None), default=None)
        if isinstance(summary, dict) and "capitulation_ratio" not in summary and total_usd is not None and total_usd > 0.0 and long_liq is not None:
            cap_ratio = long_liq / total_usd
        sq_ratio = safe_float(snap_attr(summary, "squeeze_ratio", None), default=None)
        if isinstance(summary, dict) and "squeeze_ratio" not in summary and total_usd is not None and total_usd > 0.0 and short_liq is not None:
            sq_ratio = short_liq / total_usd
        win_sec = safe_float(snap_attr(summary, "window_sec", None), default=self.window_sec)
        if (
            total_usd is None
            or long_liq is None
            or short_liq is None
            or cap_ratio is None
            or sq_ratio is None
            or total_usd < self.min_notional_usd
        ):
            return None

        denom = max(self.min_notional_usd * 2.0, 1e-9)

        # 1. Long capitulation burst -> BUY entry
        if cap_ratio >= self.ratio_threshold and long_liq >= self.min_notional_usd:
            return TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="LiquidationCapitulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(cap_ratio, 2),
                ts=t_now,
                confidence=round(min(1.0, long_liq / denom), 2),
                reason="liquidation_capitulation_burst",
                metadata={
                    "long_liq_usd": long_liq,
                    "short_liq_usd": short_liq,
                    "capitulation_ratio": cap_ratio,
                    "window_sec": win_sec,
                },
            )

        # 2. Short squeeze burst -> SELL exit
        if sq_ratio >= self.ratio_threshold and short_liq >= self.min_notional_usd:
            return TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="LiquidationCapitulationTrigger",
                symbol=symbol,
                price=price,
                strength=round(sq_ratio, 2),
                ts=t_now,
                confidence=round(min(1.0, short_liq / denom), 2),
                reason="liquidation_short_squeeze_burst",
                metadata={
                    "long_liq_usd": long_liq,
                    "short_liq_usd": short_liq,
                    "squeeze_ratio": sq_ratio,
                    "window_sec": win_sec,
                },
            )

        return None
