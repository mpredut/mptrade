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
from datetime import date

from botcore import (
    float_env,
    load_shadow_environment,
    normalize_epoch_sec,
    pos_float_env,
    prepare_shadow_runtime,
    require_shadow_interval,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    sanitize_pipe_field,
)
from cachepaths import cache_symbol_variants
from instrument_registry import casefold_get
from intelligence.internal.state.volatility import (
    K_DCA_DEFAULT as K_DCA,
    K_REENTRY_DEFAULT as K_REENTRY,
    adaptive_thresholds,
    vol_1h_pct,
)
from intelligence.internal.triggers.kalman_trigger import (
    GAP_RESET_SEC,
    KALMAN_QR,
    KalmanTrend,
)
from state_io import append_text_line, durable_replace_file, read_json_dict, save_json_state


# Feed Kalman at a subsampled cadence. One-second ticks track minute oscillations
# and produced thousands of daily transitions; 60 seconds yields about four BTC
# transitions per day, comparable with the primary model's timescale.
_f_env = pos_float_env
KALMAN_SAMPLE_SEC = pos_float_env("SHADOW_KALMAN_SAMPLE_SEC", 60.0)


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

    _sanitize = staticmethod(sanitize_pipe_field)

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
            append_text_line(self._path(), "|".join(self._sanitize(c) for c in cols))
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
        try:
            if (
                os.path.basename(self.state_path) in ("cache_kalman_trend.json", "kalman_state.json")
                and not os.path.exists(self.state_path)
            ):
                for legacy_name in ("kalman_state.json", "shadow_state.json"):
                    legacy_path = os.path.join(os.path.dirname(self.state_path), legacy_name)
                    if legacy_path != self.state_path and os.path.exists(legacy_path):
                        durable_replace_file(legacy_path, self.state_path)
                        break
            loaded = read_json_dict(self.state_path, default=None, recover_previous=True)
            if isinstance(loaded, dict):
                self._state = loaded
        except Exception as e:  # noqa: BLE001
            print(f"[shadow_signals] error loading initial kalman state: {e}")

    def _write_state(self, now: float) -> None:
        if self._last_state_write > now:
            self._last_state_write = 0.0
        has_new_symbols = len(self._state) > getattr(self, "_last_written_count", 0)
        if not self.state_path or (not has_new_symbols and (now - self._last_state_write) < self.state_min_interval):
            return
        try:
            merged = dict(self._state)
            if os.path.exists(self.state_path):
                existing = read_json_dict(self.state_path, default=None, recover_previous=True)
                if isinstance(existing, dict):
                    for k, v in existing.items():
                        if not isinstance(v, dict):
                            continue
                        cur_v = merged.get(k)
                        if cur_v is None:
                            merged[k] = v
                        elif (
                            safe_nonneg_float(cur_v.get("kalman_vel_std"), 6.0) >= 5.99
                            and abs(safe_float(cur_v.get("kalman_vel"), 0.0)) < 1e-12
                            and safe_nonneg_float(v.get("kalman_vel_std"), 6.0) < 5.99
                        ):
                            merged[k] = {
                                **cur_v,
                                **{
                                    fk: v[fk]
                                    for fk in ("kalman_vel", "kalman_vel_std", "kalman_trend")
                                    if fk in v
                                },
                            }
                    self._state = merged
            save_json_state(self.state_path, merged)
            self._last_state_write = now
            self._last_written_count = len(merged)
        except Exception as e:  # noqa: BLE001
            print(f"[shadow_signals] error writing shadow state: {e}")

    def current_trend(self, symbol: str) -> tuple:
        """Return Kalman trend and state age for tradeall's live order gate."""
        st = self._state.get(symbol)
        if not isinstance(st, dict) and isinstance(symbol, str):
            st = casefold_get(self._state, symbol)
            if not isinstance(st, dict):
                for cand in cache_symbol_variants(symbol):
                    st = self._state.get(cand) if cand in self._state else casefold_get(self._state, cand)
                    if isinstance(st, dict):
                        break
        if not isinstance(st, dict) or not st:
            return None, 1e18
        ts_f = normalize_epoch_sec(st.get("ts"))
        now_f = normalize_epoch_sec(self._now(), allow_zero=True)
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
               big_prices, big_sample_rate: float, *, precomputed_vol_1h_pct: float | None = None) -> dict:
        # Feed only at KALMAN_SAMPLE_SEC and reuse fields between feeds so the
        # snapshot remains populated.
        if isinstance(symbol, str):
            norm_sym = symbol.strip().upper()
            if norm_sym:
                if symbol != norm_sym and symbol in self._state and norm_sym not in self._state:
                    self._state[norm_sym] = self._state.pop(symbol)
                elif symbol != norm_sym and symbol in self._state:
                    self._state.pop(symbol, None)
                symbol = norm_sym
        ts_f = normalize_epoch_sec(ts, allow_zero=True)
        price_f = safe_pos_float(price, default=None)
        if ts_f is None or price_f is None:
            k = self._last_kfields.get(
                symbol, {"vel": 0.0, "vel_std": 0.0, "trend": 0, "old_trend": 0}
            )
            return self._build_fields(k, self._last_vol_fields.get(symbol))
        ts, price = ts_f, price_f
        sr_val = safe_pos_float(big_sample_rate, 1.0)
        eps_val = safe_nonneg_float(epsilon, 0.0)
        last_fed = self._last_fed.get(symbol, -1e18)
        if last_fed > ts:
            last_fed = -1e18
        is_cold = len(self._fed_prices.get(symbol, ())) < 2
        has_big_warmup = (
            big_prices is not None
            and not isinstance(big_prices, (str, bytes, dict))
            and hasattr(big_prices, "__len__")
            and len(big_prices) >= 3
        )
        if ts - last_fed >= KALMAN_SAMPLE_SEC or (is_cold and has_big_warmup):
            kf = self._kalman.get(symbol)
            fed = self._fed_prices.setdefault(symbol, [])
            if kf is None or (is_cold and has_big_warmup):
                kf = self._kalman[symbol] = KalmanTrend(qr=KALMAN_QR, gap_reset_sec=GAP_RESET_SEC)
                fed.clear()
                if (
                    big_prices is not None
                    and not isinstance(big_prices, (str, bytes, dict))
                    and hasattr(big_prices, "__len__")
                    and len(big_prices) >= 2
                ):
                    seq_prices = big_prices if isinstance(big_prices, (list, tuple)) else list(big_prices)
                    step_n = max(1, int(round(KALMAN_SAMPLE_SEC / max(sr_val, 0.1))))
                    raw_hist = seq_prices[-1 - step_n :: -step_n][:60]
                    if not raw_hist and len(seq_prices) >= 2:
                        raw_hist = [seq_prices[0]]
                    step_dt = step_n * max(sr_val, 0.1)
                    hist_pts = [
                        (steps_back, p_h)
                        for steps_back, raw_p in enumerate(raw_hist, start=1)
                        if (ts - steps_back * step_dt) >= 0.0
                        and (
                            p_h := safe_pos_float(
                                raw_p[1] if isinstance(raw_p, (list, tuple)) and len(raw_p) >= 2 else raw_p,
                                default=None,
                            )
                        )
                        is not None
                    ][::-1]
                    if hist_pts:
                        for steps_back, p_h in hist_pts:
                            fed.append(p_h)
                            if len(fed) > 60:
                                fed.pop(0)
                            eps_h = self._grad_std(fed) if len(fed) >= 5 else max(eps_val * math.sqrt(KALMAN_SAMPLE_SEC / max(sr_val, 0.1)), p_h * 0.0008)
                            kf.update(ts - steps_back * step_dt, p_h, eps_h)
            # Measure R at the Kalman step scale rather than one-second tick scale,
            # which would appear unrealistically precise and flicker. Derive it from
            # the subsampled fed prices, falling back to caller epsilon during warm-up.
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

        if precomputed_vol_1h_pct is not None:
            v1h = safe_nonneg_float(precomputed_vol_1h_pct, default=None)
            self._last_vol_time[symbol] = ts
            self._last_vol_fields[symbol] = v1h
        elif self.compute_vol:
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
