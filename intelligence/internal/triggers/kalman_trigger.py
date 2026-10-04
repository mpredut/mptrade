"""Kalman-filter directional trigger generator (Entry on UP, Exit on DOWN)."""
from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

import numpy as np

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide


def _f_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default
    except ValueError:
        return default


KALMAN_QR = _f_env("SHADOW_KALMAN_QR", 0.0005)
CONF_ENTER = 1.64
CONF_EXIT = _f_env("SHADOW_KALMAN_EXIT", 0.8)
MIN_VEL_PCT_MIN = 0.005
DT_MIN = 0.05
GAP_RESET_SEC = 300.0


class KalmanTrend:
    """One-dimensional constant-velocity Kalman filter for level and velocity tracking."""

    def __init__(self, qr: float = KALMAN_QR):
        self.qr = qr
        self.x = None          # [level, velocity]
        self.P = None          # State covariance
        self.last_ts = None
        self.trend = 0         # Last confirmed direction: -1, 0, or +1

    def update(self, ts: float, price: float, epsilon: float | None) -> dict:
        """Run one predict/update step and return velocity and trend fields."""
        eps = float(epsilon) if epsilon else 0.0
        if eps <= 0:
            eps = max(price * 1e-4, 1e-9)
        R = eps * eps

        if self.x is None:
            self.x = np.array([price, 0.0])
            self.P = np.diag([R * 10.0, (price * 1e-3) ** 2])
            self.last_ts = ts
            return self._out(price, old_trend=self.trend)

        raw_dt = ts - self.last_ts
        if raw_dt > GAP_RESET_SEC:
            self.x = np.array([price, 0.0])
            self.P = np.diag([R * 10.0, (price * 1e-3) ** 2])
            self.last_ts = ts
            old_trend = self.trend
            out = self._out(price, old_trend=old_trend)
            self.trend = out["trend"]
            return out

        dt = max(raw_dt, DT_MIN)
        self.last_ts = ts

        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.qr * R
        Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                          [dt ** 2 / 2.0, dt]])
        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # Update (H = [1, 0])
        y = price - self.x[0]
        S = self.P[0, 0] + R
        K = self.P[:, 0] / S
        self.x = self.x + K * y
        self.P = self.P - np.outer(K, self.P[0, :])

        old_trend = self.trend
        out = self._out(price, old_trend=old_trend)
        self.trend = out["trend"]
        return out

    def _out(self, price: float, old_trend: int) -> dict:
        vel = float(self.x[1])
        vel_std = math.sqrt(max(float(self.P[1, 1]), 0.0))
        vel_pct_min = vel / price * 100.0 * 60.0
        std_pct_min = vel_std / price * 100.0 * 60.0
        trend = old_trend
        if old_trend == 0:
            if abs(vel_pct_min) > max(CONF_ENTER * std_pct_min, MIN_VEL_PCT_MIN):
                trend = 1 if vel_pct_min > 0 else -1
        else:
            if vel_pct_min * old_trend < 0 and abs(vel_pct_min) > CONF_ENTER * std_pct_min:
                trend = -old_trend
            elif abs(vel_pct_min) < CONF_EXIT * std_pct_min:
                trend = 0
        return {
            "vel": round(vel_pct_min, 5),
            "vel_std": round(std_pct_min, 5),
            "trend": trend,
            "old_trend": old_trend,
        }


class KalmanTrendTrigger:
    """Actionable trigger generator wrapping KalmanTrend state."""

    def __init__(self, qr: float = KALMAN_QR):
        self.filter = KalmanTrend(qr=qr)

    def evaluate(
        self, symbol: str, ts: float, price: float, epsilon: Optional[float] = None
    ) -> Tuple[dict, Optional[TriggerEvent]]:
        """Update Kalman state and emit a TriggerEvent on confirmed directional transitions."""
        out = self.filter.update(ts, price, epsilon)
        trend = out["trend"]
        old_trend = out["old_trend"]

        trigger_event: Optional[TriggerEvent] = None
        if trend != old_trend and trend != 0:
            confidence = (
                abs(out["vel"]) / max(out["vel_std"], 1e-6)
                if out["vel_std"] > 0
                else 1.0
            )
            if trend == 1:
                trigger_event = TriggerEvent(
                    action=TriggerAction.ENTRY,
                    side=TriggerSide.BUY,
                    source="kalman_trigger",
                    symbol=symbol,
                    price=price,
                    strength=confidence,
                    ts=ts,
                    velocity=out["vel"],
                    confidence=confidence,
                    reason="kalman_transition_up",
                    metadata=out,
                )
            elif trend == -1:
                trigger_event = TriggerEvent(
                    action=TriggerAction.EXIT,
                    side=TriggerSide.SELL,
                    source="kalman_trigger",
                    symbol=symbol,
                    price=price,
                    strength=confidence,
                    ts=ts,
                    velocity=out["vel"],
                    confidence=confidence,
                    reason="kalman_transition_down",
                    metadata=out,
                )

        return out, trigger_event
