"""Mean-reversion trigger based on Wilder's RSI and Bollinger Bands %B.

Adapted from research (offline/research/tradeall_trigger_gate/experiment_rsi_bollinger.py).

NOTE & RESEARCH FINDINGS (from docs/archive/RSI_BOLLINGER_REVIEW_2026-09-14.md):
- When tested across 329 days of live data, unconstrained mean-reversion triggers
  failed against Buy & Hold during persistent trend regimes by catching falling knives in bear trends.
- Therefore, this trigger is recommended ONLY when combined with regime awareness (e.g. Hurst H < 0.45
  or confirmed sideways/ranging market regime) or with protective guards.
"""

from __future__ import annotations

from collections import deque
from typing import Literal
from intelligence.internal.triggers.trigger_event import (
    TriggerEvent,
    TriggerAction,
    TriggerSide,
)


class MeanReversionTrigger:
    """Computes Wilder's RSI and Bollinger %B to identify oversold (buy) and overbought (sell) turning points."""

    def __init__(
        self,
        mode: Literal["rsi", "pb", "both"] = "both",
        rsi_period: int = 14,
        rsi_low: float = 30.0,
        rsi_high: float = 70.0,
        bb_period: int = 20,
        bb_k: float = 2.0,
        pb_low: float = 0.0,
        pb_high: float = 1.0,
    ) -> None:
        self.mode = mode
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.bb_period = bb_period
        self.bb_k = bb_k
        self.pb_low = pb_low
        self.pb_high = pb_high

        self.prev_price: float | None = None
        self.avg_gain: float | None = None
        self.avg_loss: float | None = None
        self._seed_gains: list[tuple[float, float]] = []
        self.bb_buf: deque[float] = deque(maxlen=bb_period)
        self.current_sign: int = 0  # +1 oversold (buy), -1 overbought (sell), 0 neutral

    def _calc_rsi(self, price: float) -> float | None:
        if self.prev_price is None:
            self.prev_price = price
            return None
            
        change = price - self.prev_price
        self.prev_price = price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self.avg_gain is None:
            self._seed_gains.append((gain, loss))
            if len(self._seed_gains) < self.rsi_period:
                return None
            self.avg_gain = sum(g for g, _ in self._seed_gains) / self.rsi_period
            self.avg_loss = sum(l for _, l in self._seed_gains) / self.rsi_period
        else:
            self.avg_gain = (self.avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self.avg_loss = (self.avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

        if self.avg_loss == 0.0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _calc_percent_b(self, price: float) -> float | None:
        self.bb_buf.append(price)
        if len(self.bb_buf) < self.bb_period:
            return None
            
        n = len(self.bb_buf)
        mean = sum(self.bb_buf) / n
        var = sum((p - mean) ** 2 for p in self.bb_buf) / n
        sd = var ** 0.5
        if sd == 0.0:
            return 0.5
            
        upper = mean + self.bb_k * sd
        lower = mean - self.bb_k * sd
        if upper == lower:
            return 0.5
        return (price - lower) / (upper - lower)

    def update(self, symbol: str, timestamp: float, price: float) -> TriggerEvent:
        """Feed latest price and return current TriggerEvent."""
        rsi = self._calc_rsi(price)
        pb = self._calc_percent_b(price)

        oversold = False
        overbought = False

        if self.mode == "rsi":
            if rsi is not None:
                oversold = rsi < self.rsi_low
                overbought = rsi > self.rsi_high
        elif self.mode == "pb":
            if pb is not None:
                oversold = pb < self.pb_low
                overbought = pb > self.pb_high
        else:  # "both"
            if rsi is not None and pb is not None:
                oversold = (rsi < self.rsi_low) and (pb < self.pb_low)
                overbought = (rsi > self.rsi_high) and (pb > self.pb_high)

        new_sign = 1 if oversold else (-1 if overbought else 0)
        self.current_sign = new_sign

        if new_sign == 1:
            side = TriggerSide.BUY
            action = TriggerAction.ENTRY
            conf = 1.0
        elif new_sign == -1:
            side = TriggerSide.SELL
            action = TriggerAction.EXIT
            conf = 1.0
        else:
            side = TriggerSide.HOLD
            action = TriggerAction.NEUTRAL
            conf = 0.0

        return TriggerEvent(
            action=action,
            side=side,
            source="MeanReversionTrigger",
            symbol=symbol,
            price=price,
            strength=float(abs(new_sign)),
            ts=timestamp,
            confidence=conf,
            metadata={
                "rsi": rsi,
                "percent_b": pb,
                "mode": self.mode,
                "current_sign": self.current_sign,
            },
        )
