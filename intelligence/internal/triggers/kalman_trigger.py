"""Kalman-filter directional trigger generator (Entry on UP, Exit on DOWN)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from botcore import float_env, safe_float, safe_pos_float
from intelligence.internal.triggers.trigger_event import TriggerEvent


def _f_env(name: str, default: float) -> float:
    return safe_pos_float(float_env(name), default)


KALMAN_QR = _f_env("SHADOW_KALMAN_QR", 0.0005)
CONF_ENTER = 1.64
CONF_EXIT = _f_env("SHADOW_KALMAN_EXIT", 0.8)
MIN_VEL_PCT_MIN = 0.005
DT_MIN = 0.05
GAP_RESET_SEC = 300.0


class KalmanTrend:
    """One-dimensional constant-velocity Kalman filter for level and velocity tracking."""

    def __init__(self, qr: float = KALMAN_QR, gap_reset_sec: float = GAP_RESET_SEC):
        self.qr = safe_pos_float(qr, default=KALMAN_QR) or KALMAN_QR
        self.gap_reset_sec = safe_pos_float(gap_reset_sec, default=GAP_RESET_SEC) or GAP_RESET_SEC
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
        ts_f = safe_float(ts, default=None)
        price_f = safe_pos_float(price, default=None)
        if ts_f is None or price_f is None:
            return {
                "vel": 0.0,
                "vel_std": 0.0,
                "trend": self.trend,
                "old_trend": self.trend,
            }
        ts, price = ts_f, price_f
        eps_f = safe_pos_float(epsilon, default=None)
        eps = eps_f if eps_f is not None else max(price * 1e-4, 1e-9)
        R = eps * eps

        last_ts_f = safe_float(self.last_ts, default=None)
        if self.x is None or last_ts_f is None:
            self._reset_state(price, R, ts)
            return self._out(price, old_trend=self.trend)

        raw_dt = ts - last_ts_f
        if raw_dt > self.gap_reset_sec or raw_dt < -1.0:
            self._reset_state(price, R, ts)
            old_trend = self.trend
            self.trend = 0
            out = self._out(price, old_trend=0)
            out["old_trend"] = old_trend
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
        denom = price if price and price > 0 else 1e-9
        vel_pct_min = vel / denom * 100.0 * 60.0
        std_pct_min = vel_std / denom * 100.0 * 60.0
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
            price_f = float(price)
            ts_f = float(ts)
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
