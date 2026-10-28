#!/usr/bin/env python3
"""Kalman trend and adaptive-volatility signals consumed by tradeall.

The Kalman trend is no longer purely observational: tradeall uses it to gate all
orders and optionally initiate primary orders for configured symbols. Adaptive
reentry and DCA thresholds remain observational because tradeall has no consumer
for them. All fields are published to snapshots, and Kalman transitions are
journaled for visual and quantitative comparison.

KalmanTrend is a constant-velocity level/velocity filter that reports percentage
velocity, uncertainty, and a {-1, 0, +1} direction. vol_1h_pct estimates one-hour
volatility from log returns. KalmanJournal writes one condensed row per transition.
Environment variables configure process/measurement noise and adaptive multipliers.
"""
from __future__ import annotations

import math
import os
import time
from state_io import atomic_write_json, read_recovering_previous, save_json_state
from datetime import date

import numpy as np
from botcore import float_env, safe_float, safe_nonneg_float, safe_pos_float


def _f_env(name: str, default: float) -> float:
    val = float_env(name)
    return val if (val is not None and val > 0.0) else default


# Feed Kalman at a subsampled cadence. One-second ticks track minute oscillations
# and produced thousands of daily transitions; 60 seconds yields about four BTC
# transitions per day, comparable with the primary model's timescale.
KALMAN_SAMPLE_SEC = _f_env("SHADOW_KALMAN_SAMPLE_SEC", 60.0)

from intelligence.internal.triggers.kalman_trigger import (
    KalmanTrend,
    KALMAN_QR,
    CONF_ENTER,
    CONF_EXIT,
    MIN_VEL_PCT_MIN,
    DT_MIN,
    GAP_RESET_SEC,
)
from intelligence.internal.state.volatility import (
    vol_1h_pct,
    adaptive_thresholds,
    K_REENTRY_DEFAULT as K_REENTRY,
    K_DCA_DEFAULT as K_DCA,
)


_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOG_DIR = os.path.join(_ROOT, "logger")


class KalmanJournal:
    """Write sanitized pipe-delimited signal transitions without affecting host.

    Format is ts|symbol|signal|event|state|old_state|price|vel|vel_std. Live files
    rotate daily; backtests use a flat fixed_path file.
    """

    def __init__(self, out_dir: str = _DEFAULT_LOG_DIR, fixed_path: str | None = None):
        self.out_dir = out_dir
        self.fixed_path = fixed_path

    @staticmethod
    def _sanitize(value) -> str:
        return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ") if value is not None else ""

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
            print(f"[shadow_signals] error writing kalman journal: {e}")


class KalmanStateTracker:
    """Maintain Kalman/adaptive signals and their journal for a symbol set.

    One update call returns snapshot fields. ``state_path`` stores the latest
    per-symbol state because cacheManager, not tradeall, owns the trend-cache file;
    the monitor combines this state file with that snapshot.
    """

    def __init__(self, journal: KalmanJournal | None = None,
                 state_path: str | None = None, state_min_interval: float = 1.0,
                 now_fn=None, compute_vol: bool = True):
        self.journal = journal or KalmanJournal()
        # current_trend() ages the signal against this clock. Live uses wall time; a
        # replay must inject its simulated clock, or every signal looks months old and
        # the order gate blocks every BUY.
        self._now = now_fn or time.time
        self.state_path = state_path
        self.state_min_interval = safe_nonneg_float(state_min_interval, 1.0)
        self.compute_vol = bool(compute_vol)
        self._state: dict = {}
        if self.state_path:
            self._port_and_load_initial_state()
        self._last_state_write = 0.0
        self._kalman: dict = {}
        self._last_fed: dict = {}      # Last Kalman feed timestamp per symbol.
        self._last_kfields: dict = {}  # Last Kalman fields between feeds.
        self._fed_prices: dict = {}    # Fed prices for step-scale epsilon.
        self._last_vol_time: dict = {} # Throttling for 1h volatility computation.
        self._last_vol_fields: dict = {}

    def _port_and_load_initial_state(self) -> None:
        """Port legacy shadow_state.json to kalman_state.json if needed and load initial state."""
        import json
        try:
            if (
                os.path.basename(self.state_path) == "kalman_state.json"
                and not os.path.exists(self.state_path)
            ):
                legacy_path = os.path.join(os.path.dirname(self.state_path), "shadow_state.json")
                if os.path.exists(legacy_path):
                    os.replace(legacy_path, self.state_path)
            loaded = read_recovering_previous(self.state_path, json.loads, default=None)
            if isinstance(loaded, dict):
                self._state = loaded
        except Exception as e:  # noqa: BLE001
            print(f"[shadow_signals] error loading initial kalman state: {e}")

    def _write_state(self, now: float) -> None:
        if self._last_state_write > now:
            self._last_state_write = 0.0
        if not self.state_path or (now - self._last_state_write) < self.state_min_interval:
            return
        try:
            save_json_state(self.state_path, self._state)
            self._last_state_write = now
        except Exception as e:  # noqa: BLE001
            print(f"[shadow_signals] error writing shadow state: {e}")

    def current_trend(self, symbol: str) -> tuple:
        """Return Kalman trend and state age for tradeall's live order gate."""
        st = self._state.get(symbol)
        if not isinstance(st, dict) and isinstance(symbol, str):
            sym_u = symbol.strip().upper()
            if sym_u:
                st = (
                    self._state.get(sym_u)
                    or self._state.get(sym_u.replace("/", "").replace("-", "").replace("_", ""))
                )
        if not isinstance(st, dict) or not st:
            return None, 1e18
        ts_f = safe_pos_float(st.get("ts"), default=None)
        now_f = safe_float(self._now(), default=None)
        kt_f = safe_float(st.get("kalman_trend"), default=None)
        if ts_f is None or now_f is None or kt_f not in (-1.0, 0.0, 1.0):
            return None, 1e18
        age = now_f - ts_f
        if age < -60.0:
            return None, 1e18
        return int(kt_f), max(0.0, age)

    @staticmethod
    def _grad_std(fed: list[float]) -> float:
        n = len(fed)
        if n <= 2:
            return 0.0
        g0 = fed[1] - fed[0]
        gn = fed[-1] - fed[-2]
        sum_g = g0 + gn
        sum_g2 = g0 * g0 + gn * gn
        for i in range(1, n - 1):
            gi = (fed[i + 1] - fed[i - 1]) * 0.5
            sum_g += gi
            sum_g2 += gi * gi
        mean = sum_g / n
        return math.sqrt(max(0.0, sum_g2 / n - mean * mean))

    def _build_fields(self, k: dict, v1h: float | None) -> dict:
        adapt_reentry, adapt_dca = adaptive_thresholds(v1h, k_reentry=K_REENTRY, k_dca=K_DCA)
        return {
            "kalman_vel": k["vel"],
            "kalman_vel_std": k["vel_std"],
            "kalman_trend": k["trend"],
            "vol_1h_pct": v1h,
            "adapt_reentry_pct": adapt_reentry,
            "adapt_dca_pct": adapt_dca,
        }

    def update(self, symbol: str, ts: float, price: float, epsilon: float | None,
               big_prices, big_sample_rate: float) -> dict:
        # Feed only at KALMAN_SAMPLE_SEC and reuse fields between feeds so the
        # snapshot remains populated.
        ts_f = safe_float(ts, default=None)
        price_f = safe_pos_float(price, default=None)
        if ts_f is None or price_f is None:
            k = self._last_kfields.get(
                symbol, {"vel": 0.0, "vel_std": 0.0, "trend": 0, "old_trend": 0}
            )
            return self._build_fields(k, self._last_vol_fields.get(symbol))
        ts = ts_f
        price = price_f
        sr_val = safe_pos_float(big_sample_rate, 1.0)
        eps_val = safe_nonneg_float(epsilon, 0.0)
        last_fed = self._last_fed.get(symbol, -1e18)
        if last_fed > ts:
            last_fed = -1e18
        if ts - last_fed >= KALMAN_SAMPLE_SEC:
            kf = self._kalman.get(symbol)
            if kf is None:
                kf = self._kalman[symbol] = KalmanTrend(qr=KALMAN_QR, gap_reset_sec=GAP_RESET_SEC)
            # Measure R at the Kalman step scale rather than one-second tick scale,
            # which would appear unrealistically precise and flicker. Derive it from
            # the subsampled fed prices, falling back to caller epsilon during warm-up.
            fed = self._fed_prices.setdefault(symbol, [])
            fed.append(price)
            if len(fed) > 60:
                fed.pop(0)
            if len(fed) >= 5:
                eps_eff = self._grad_std(fed)
            else:
                step_scale = math.sqrt(KALMAN_SAMPLE_SEC / max(sr_val, 0.1))
                base_eps = max(eps_val * step_scale, price * 0.0008)
                if len(fed) >= 2:
                    eps_eff = max(base_eps, self._grad_std(fed))
                else:
                    eps_eff = base_eps
            k = kf.update(ts, price, eps_eff)
            self._last_fed[symbol] = ts
            self._last_kfields[symbol] = k
            if k["trend"] != k["old_trend"]:
                self.journal.log_transition(ts, symbol, "kalman", k["trend"], k["old_trend"],
                                             price, k["vel"], k["vel_std"])
        else:
            k = self._last_kfields.get(symbol,
                                        {"vel": 0.0, "vel_std": 0.0, "trend": 0, "old_trend": 0})

        if self.compute_vol:
            # Throttle heavy volatility & log-return computation on big_prices (thousands of points)
            last_vol_t = self._last_vol_time.get(symbol, -1e18)
            if last_vol_t > ts:
                last_vol_t = -1e18
            if (ts - last_vol_t) >= 2.0 or (ts - last_fed >= KALMAN_SAMPLE_SEC):
                v1h = vol_1h_pct(big_prices, big_sample_rate)
                self._last_vol_time[symbol] = ts
                self._last_vol_fields[symbol] = v1h
            else:
                v1h = self._last_vol_fields.get(symbol)
        else:
            v1h = None
        fields = self._build_fields(k, v1h)
        self._state[symbol] = {**fields, "ts": ts, "price": price}
        self._write_state(ts)
        return fields
