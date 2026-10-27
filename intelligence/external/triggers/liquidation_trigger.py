"""Liquidation capitulation and short squeeze trigger.

Identifies high-probability reversal points triggered by forced liquidations:
- Panic capitulation: Long liquidations dominate (forced selling cascade), signaling dip-buy entry.
- Short squeeze: Short liquidations dominate (forced buying cascade), signaling take-profit exit.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from intelligence.internal.triggers.trigger_event import TriggerEvent, TriggerAction, TriggerSide
from intelligence.external.collectors.bybit_liquidations import (
    BybitLiquidationCollector,
    LiquidationSummary,
)


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


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
        mn = _safe_float(min_notional_usd, 100_000.0)
        self.min_notional_usd = mn if (mn is not None and mn > 0.0) else 100_000.0
        rt = _safe_float(ratio_threshold, 0.75)
        self.ratio_threshold = rt if (rt is not None and 0.0 < rt <= 1.0) else 0.75
        ws = _safe_float(window_sec, 300.0)
        self.window_sec = ws if (ws is not None and ws > 0.0) else 300.0

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

        if summary is None:
            return None

        def _sum_attr(key: str, default: Any = None) -> Any:
            return summary.get(key, default) if isinstance(summary, dict) else getattr(summary, key, default)

        long_liq = _safe_float(_sum_attr("long_liq_usd", None), None)
        short_liq = _safe_float(_sum_attr("short_liq_usd", None), None)
        total_usd = _safe_float(_sum_attr("total_usd", None), None)
        if total_usd is None and long_liq is not None and short_liq is not None:
            total_usd = long_liq + short_liq
        cap_ratio = _safe_float(_sum_attr("capitulation_ratio", None), None)
        if cap_ratio is None and total_usd is not None and total_usd > 0.0 and long_liq is not None:
            cap_ratio = long_liq / total_usd
        sq_ratio = _safe_float(_sum_attr("squeeze_ratio", None), None)
        if sq_ratio is None and total_usd is not None and total_usd > 0.0 and short_liq is not None:
            sq_ratio = short_liq / total_usd
        win_sec = _safe_float(_sum_attr("window_sec", None), self.window_sec)
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
