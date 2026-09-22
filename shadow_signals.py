#!/usr/bin/env python3
"""Kalman trend and adaptive-volatility signals consumed by tradeall.

The Kalman trend is no longer purely observational: tradeall uses it to gate all
orders and optionally initiate primary orders for configured symbols. Adaptive
reentry and DCA thresholds remain observational because tradeall has no consumer
for them. All fields are published to snapshots, and Kalman transitions are
journaled for visual and quantitative comparison.

KalmanTrend is a constant-velocity level/velocity filter that reports percentage
velocity, uncertainty, and a {-1, 0, +1} direction. vol_1h_pct estimates one-hour
volatility from log returns. ShadowJournal writes one condensed row per transition.
Environment variables configure process/measurement noise and adaptive multipliers.
"""
from __future__ import annotations

import math
import os
from state_io import atomic_write_json
from datetime import date

import numpy as np


def _f_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default
    except ValueError:
        return default


KALMAN_QR = _f_env("SHADOW_KALMAN_QR", 0.0005)   # Sweep: 94% stable after detection, ~15s latency.
K_REENTRY = _f_env("SHADOW_K_REENTRY", 2.0)
K_DCA = _f_env("SHADOW_K_DCA", 1.0)

# Feed Kalman at a subsampled cadence. One-second ticks track minute oscillations
# and produced thousands of daily transitions; 60 seconds yields about four BTC
# transitions per day, comparable with the primary model's timescale.
KALMAN_SAMPLE_SEC = _f_env("SHADOW_KALMAN_SAMPLE_SEC", 60.0)

CONF_ENTER = 1.64         # Enter direction above 1.64 standard deviations (~90%).
CONF_EXIT = _f_env("SHADOW_KALMAN_EXIT", 0.8)   # Exit only below 0.8 std for hysteresis.
MIN_VEL_PCT_MIN = 0.005   # Treat below 0.005%/minute as flat regardless of std.
DT_MIN = 0.05
# After a network or process gap, pre-gap velocity no longer describes current
# state. Reset after 300 seconds rather than capping dt and understating filter
# uncertainty. This matches tradeall's stale-signal gate.
GAP_RESET_SEC = 300.0


class KalmanTrend:
    """Apply a one-dimensional constant-velocity Kalman filter to one symbol.

    State is level and price-per-second velocity; observation is price. R comes
    from PriceWindow epsilon in absolute price units, and Q is KALMAN_QR times R
    discretized using actual elapsed time.
    """

    def __init__(self, qr: float = KALMAN_QR):
        self.qr = qr
        self.x = None          # [level, velocity]
        self.P = None          # State covariance.
        self.last_ts = None
        self.trend = 0         # Last confirmed direction: -1, 0, or +1.

    def update(self, ts: float, price: float, epsilon: float | None) -> dict:
        """Run one predict/update step and return velocity and trend fields."""
        eps = float(epsilon) if epsilon else 0.0
        if eps <= 0:
            eps = max(price * 1e-4, 1e-9)   # Warm-up assumes noise at 0.01% of price.
        R = eps * eps

        if self.x is None:
            self.x = np.array([price, 0.0])
            self.P = np.diag([R * 10.0, (price * 1e-3) ** 2])
            self.last_ts = ts
            return self._out(price, old_trend=self.trend)

        raw_dt = ts - self.last_ts
        if raw_dt > GAP_RESET_SEC:
            # Do not propagate stale velocity across a long gap. Reset as at warm-up
            # so zero velocity remains flat until enough new data accumulates.
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
        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # update (H = [1, 0])
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
        # Schmitt hysteresis enters at CONF_ENTER*std and exits below CONF_EXIT*std,
        # eliminating flicker around a single threshold.
        trend = old_trend
        if old_trend == 0:
            if abs(vel_pct_min) > max(CONF_ENTER * std_pct_min, MIN_VEL_PCT_MIN):
                trend = 1 if vel_pct_min > 0 else -1
        else:
            if vel_pct_min * old_trend < 0 and abs(vel_pct_min) > CONF_ENTER * std_pct_min:
                trend = -old_trend                      # Direct high-confidence flip.
            elif abs(vel_pct_min) < CONF_EXIT * std_pct_min:
                trend = 0
        return {"vel": round(vel_pct_min, 5), "vel_std": round(std_pct_min, 5),
                "trend": trend, "old_trend": old_trend}


def vol_1h_pct(prices, sample_rate_sec: float) -> float | None:
    """Estimate one-sigma hourly volatility from scaled log returns."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 20 or sample_rate_sec <= 0:
        return None
    p = p[p > 0]
    if len(p) < 20:
        return None
    rets = np.diff(np.log(p))
    std = float(np.std(rets))
    if std == 0.0:
        return 0.0
    return round(std * math.sqrt(3600.0 / sample_rate_sec) * 100.0, 4)


def adaptive_thresholds(vol1h: float | None) -> tuple[float | None, float | None]:
    """Return adaptive reentry and DCA percentages as multipliers of hourly vol."""
    if vol1h is None:
        return None, None
    return round(K_REENTRY * vol1h, 3), round(K_DCA * vol1h, 3)


_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOG_DIR = os.path.join(_ROOT, "logger")


class ShadowJournal:
    """Write sanitized pipe-delimited signal transitions without affecting host.

    Format is ts|symbol|signal|event|state|old_state|price|vel|vel_std. Live files
    rotate daily; backtests use a flat fixed_path file.
    """

    def __init__(self, out_dir: str = _DEFAULT_LOG_DIR, fixed_path: str | None = None):
        self.out_dir = out_dir
        self.fixed_path = fixed_path

    @staticmethod
    def _sanitize(value) -> str:
        return str(value).replace("|", "/").replace("\n", " ") if value is not None else ""

    def _path(self) -> str:
        if self.fixed_path:
            return self.fixed_path
        return os.path.join(self.out_dir, f"tradeall_shadow_{date.today().isoformat()}.log")

    def log_transition(self, ts: float, symbol: str, signal: str, state, old_state,
                       price, vel="", vel_std="") -> None:
        try:
            if not self.fixed_path:
                os.makedirs(self.out_dir, exist_ok=True)
            cols = [ts, symbol, signal, "trend_start", state, old_state, price, vel, vel_std]
            with open(self._path(), "a", encoding="utf-8") as f:
                f.write("|".join(self._sanitize(c) for c in cols) + "\n")
        except Exception as e:  # noqa: BLE001 — Logging must not stop the host.
            print(f"[shadow_signals] error writing shadow journal: {e}")


class ShadowSet:
    """Maintain Kalman/adaptive signals and their journal for a symbol set.

    One update call returns snapshot fields. ``state_path`` stores the latest
    per-symbol state because cacheManager, not tradeall, owns the trend-cache file;
    the monitor combines this state file with that snapshot.
    """

    def __init__(self, journal: ShadowJournal | None = None,
                 state_path: str | None = None, state_min_interval: float = 1.0):
        self.journal = journal or ShadowJournal()
        self.state_path = state_path
        self.state_min_interval = state_min_interval
        self._state: dict = {}
        self._last_state_write = 0.0
        self._kalman: dict = {}
        self._last_fed: dict = {}      # Last Kalman feed timestamp per symbol.
        self._last_kfields: dict = {}  # Last Kalman fields between feeds.
        self._fed_prices: dict = {}    # Fed prices for step-scale epsilon.

    def _write_state(self, now: float) -> None:
        if not self.state_path or (now - self._last_state_write) < self.state_min_interval:
            return
        try:
            atomic_write_json(self.state_path, self._state)
            self._last_state_write = now
        except Exception as e:  # noqa: BLE001
            print(f"[shadow_signals] error writing shadow state: {e}")

    def current_trend(self, symbol: str) -> tuple:
        """Return Kalman trend and state age for tradeall's live order gate."""
        st = self._state.get(symbol)
        if not st:
            return None, 1e18
        import time as _t
        return st.get("kalman_trend"), _t.time() - st.get("ts", 0)

    def update(self, symbol: str, ts: float, price: float, epsilon: float | None,
               big_prices, big_sample_rate: float) -> dict:
        # Feed only at KALMAN_SAMPLE_SEC and reuse fields between feeds so the
        # snapshot remains populated.
        last_fed = self._last_fed.get(symbol, -1e18)
        if ts - last_fed >= KALMAN_SAMPLE_SEC:
            kf = self._kalman.get(symbol)
            if kf is None:
                kf = self._kalman[symbol] = KalmanTrend()
            # Measure R at the Kalman step scale rather than one-second tick scale,
            # which would appear unrealistically precise and flicker. Derive it from
            # the subsampled fed prices, falling back to caller epsilon during warm-up.
            fed = self._fed_prices.setdefault(symbol, [])
            fed.append(price)
            if len(fed) > 60:
                fed.pop(0)
            if len(fed) >= 5:
                import numpy as _np
                eps_eff = float(_np.std(_np.gradient(_np.asarray(fed))))
            else:
                eps_eff = epsilon
            k = kf.update(ts, price, eps_eff)
            self._last_fed[symbol] = ts
            self._last_kfields[symbol] = k
            if k["trend"] != k["old_trend"]:
                self.journal.log_transition(ts, symbol, "kalman", k["trend"], k["old_trend"],
                                             price, k["vel"], k["vel_std"])
        else:
            k = self._last_kfields.get(symbol,
                                        {"vel": 0.0, "vel_std": 0.0, "trend": 0, "old_trend": 0})

        v1h = vol_1h_pct(big_prices, big_sample_rate)
        adapt_re, adapt_dca = adaptive_thresholds(v1h)
        fields = {
            "kalman_vel": k["vel"], "kalman_vel_std": k["vel_std"],
            "kalman_trend": k["trend"],
            "vol_1h_pct": v1h, "adapt_reentry_pct": adapt_re, "adapt_dca_pct": adapt_dca,
        }
        self._state[symbol] = {**fields, "ts": ts, "price": price}
        self._write_state(ts)
        return fields
