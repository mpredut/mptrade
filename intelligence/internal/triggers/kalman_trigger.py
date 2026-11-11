"""Kalman-filter directional trigger generator (Entry on UP, Exit on DOWN)."""
from __future__ import annotations

from typing import Optional, Tuple

from botcore import normalize_epoch_sec, safe_pos_float
from intelligence.internal.triggers.trigger_event import TriggerEvent
from pricewindow import (
    GAP_RESET_SEC,
    KALMAN_QR,
    KalmanTrend,
)


class KalmanTrendTrigger:
    """Actionable trigger generator wrapping KalmanTrend state."""

    def __init__(self, qr: float = KALMAN_QR, gap_reset_sec: float = GAP_RESET_SEC):
        self.filter = KalmanTrend(qr=qr, gap_reset_sec=gap_reset_sec)

    def evaluate(
        self, symbol: str, ts: float, price: float, epsilon: Optional[float] = None
    ) -> Tuple[dict, Optional[TriggerEvent]]:
        """Update Kalman state and emit a TriggerEvent on confirmed directional transitions."""
        out = self.filter.update(ts, price, epsilon)
        trend = out["trend"]
        old_trend = out["old_trend"]

        trigger_event: Optional[TriggerEvent] = None
        if trend != old_trend and trend != 0:
            price_f = safe_pos_float(price, default=0.0) or 0.0
            ts_f = normalize_epoch_sec(ts, default=0.0, allow_zero=True) or 0.0
            confidence = (
                abs(out["vel"]) / max(out["vel_std"], 1e-6)
                if out["vel_std"] > 0
                else 1.0
            )
            if trend in (1, -1):
                trigger_event = TriggerEvent.directional(
                    trend == 1,
                    source="kalman_trigger",
                    symbol=symbol,
                    price=price_f,
                    strength=confidence,
                    ts=ts_f,
                    velocity=out["vel"],
                    confidence=confidence,
                    reason_bull="kalman_transition_up",
                    reason_bear="kalman_transition_down",
                    metadata=out,
                )

        return out, trigger_event
