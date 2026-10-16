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

    def __init__(self, qr: float = KALMAN_QR, gap_reset_sec: float = GAP_RESET_SEC):
        self.qr = qr
        self.gap_reset_sec = float(gap_reset_sec)
        self.x = None          # [level, velocity]
        self.P = None          # State covariance
        self._x0 = 0.0
        self._x1 = 0.0
        self._p00 = 0.0
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = 0.0
        self.last_ts = None
        self.trend = 0         # Last confirmed direction: -1, 0, or +1

    def _reset_state(self, price: float, R: float, ts: float) -> None:
        self._x0 = float(price)
        self._x1 = 0.0
        self._p00 = R * 10.0
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = (price * 1e-3) ** 2
        self.x = [self._x0, self._x1]
        self.last_ts = ts

    def update(self, ts: float, price: float, epsilon: float | None) -> dict:
        """Run one predict/update step and return velocity and trend fields."""
        eps = float(epsilon) if epsilon else 0.0
        if eps <= 0:
            eps = max(price * 1e-4, 1e-9)
        R = eps * eps

        if self.x is None:
            self._reset_state(price, R, ts)
            return self._out(price, old_trend=self.trend)

        raw_dt = ts - self.last_ts
        if raw_dt > self.gap_reset_sec:
            self._reset_state(price, R, ts)
            old_trend = self.trend
            out = self._out(price, old_trend=old_trend)
            self.trend = out["trend"]
            return out

        dt = max(raw_dt, DT_MIN)
        self.last_ts = ts

        # Normalize R across sampling intervals so multi-second steps (e.g. 60s)
        # do not cube dt against an already step-scaled measurement variance R.
        q = self.qr * (R / (max(dt, 1.0) ** 1.25))
        dt2 = dt * dt
        q00 = q * (dt ** 3 / 3.0)
        q01 = q * (dt2 / 2.0)
        q11 = q * dt

        # Predict: x = F @ x, P = F @ P @ F.T + Q
        x0 = self._x0 + dt * self._x1
        x1 = self._x1
        p00 = self._p00 + dt * (self._p10 + self._p01) + dt2 * self._p11 + q00
        p01 = self._p01 + dt * self._p11 + q01
        p10 = self._p10 + dt * self._p11 + q01
        p11 = self._p11 + q11

        # Update (H = [1, 0])
        y = price - x0
        S = p00 + R
        k0 = p00 / S
        k1 = p10 / S
        self._x0 = x0 + k0 * y
        self._x1 = x1 + k1 * y
        self.x = [self._x0, self._x1]
        self._p00 = p00 - k0 * p00
        self._p01 = p01 - k0 * p01
        self._p10 = p10 - k1 * p00
        self._p11 = p11 - k1 * p01

        old_trend = self.trend
        out = self._out(price, old_trend=old_trend, dt=dt)
        self.trend = out["trend"]
        return out

    def _out(self, price: float, old_trend: int, dt: float = 1.0) -> dict:
        vel = self._x1
        vel_std = math.sqrt(max(self._p11, 0.0))
        vel_pct_min = vel / price * 100.0 * 60.0
        std_pct_min = vel_std / price * 100.0 * 60.0
        min_vel = max(MIN_VEL_PCT_MIN, 0.045) if dt >= 30.0 else MIN_VEL_PCT_MIN
        trend = old_trend
        if old_trend == 0:
            if abs(vel_pct_min) > max(CONF_ENTER * std_pct_min, min_vel):
                trend = 1 if vel_pct_min > 0 else -1
        else:
            if vel_pct_min * old_trend < 0 and abs(vel_pct_min) > max(CONF_ENTER * std_pct_min, min_vel):
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
