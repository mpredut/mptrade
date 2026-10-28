"""Parabolic surge guard: prevents buying at the peak of vertical price spikes (Anti-FOMO)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    clean_guard_symbol_key,
    normalize_guard_side,
    safe_pos_guard_float,
)


class ParabolicSurgeGuard:
    """Anti-FOMO protection guard preventing BUY execution during parabolic spikes.

    Tracks rolling price history to detect sudden acceleration. If a rapid surge
    is detected, BUY orders are deferred until a healthy pullback occurs or the
    spike consolidates.
    """

    def __init__(
        self,
        surge_threshold_pct: float = 4.0,
        pullback_required_pct: float = 1.5,
        window_seconds: float = 7200.0,  # 2 hours
        vol_multiplier: float = 3.0,
    ):
        self.surge_threshold_pct = safe_pos_guard_float(surge_threshold_pct, 4.0)
        self.pullback_required_pct = safe_pos_guard_float(pullback_required_pct, 1.5, allow_zero=True)
        self.window_seconds = safe_pos_guard_float(window_seconds, 7200.0)
        self.vol_multiplier = safe_pos_guard_float(vol_multiplier, 3.0, allow_zero=True)

        # symbol -> {"peak": float, "surge_active": bool, "surge_start_px": float, "ts": float}
        self._states: Dict[str, dict] = {}

    def check(
        self,
        symbol: str,
        side: str,
        current_price: float,
        price_history: Optional[List[Tuple[float, float]]] = None,  # [(ts, price), ...]
        volatility_1h_pct: Optional[float] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate whether a parabolic surge (BUY) or vertical dump (SELL) defers the proposed order side."""
        side_u = normalize_guard_side(side)
        if side_u not in ("BUY", "SELL"):
            return GuardDecision.allow("ParabolicSurgeGuard", "unsupported_side_skipped")

        if current_price is None or isinstance(current_price, bool):
            return GuardDecision.allow("ParabolicSurgeGuard", "invalid_price_skipped")
        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            return GuardDecision.allow("ParabolicSurgeGuard", "invalid_price_skipped")
        if not math.isfinite(current_price) or current_price <= 0:
            return GuardDecision.allow("ParabolicSurgeGuard", "invalid_price_skipped")

        ts_now = 0.0
        if now is not None and not isinstance(now, bool):
            try:
                nf = float(now)
                if math.isfinite(nf) and nf > 0:
                    ts_now = nf
            except (TypeError, ValueError):
                pass
        history_seq = (
            list(price_history)
            if price_history is not None
            and not isinstance(price_history, (str, bytes, bytearray, dict, set, frozenset, bool))
            and hasattr(price_history, "__iter__")
            else []
        )
        if ts_now <= 0.0 and history_seq:
            for item in reversed(history_seq):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    raw_last_ts = item[0]
                    if not isinstance(raw_last_ts, bool):
                        hf = float(raw_last_ts)
                        if math.isfinite(hf) and hf > 0:
                            ts_now = max(ts_now, hf)
                except (TypeError, ValueError, IndexError):
                    continue

        # Dynamic threshold based on volatility if available
        threshold = self.surge_threshold_pct
        if volatility_1h_pct is not None and not isinstance(volatility_1h_pct, bool):
            try:
                vol_f = float(volatility_1h_pct)
                if math.isfinite(vol_f) and vol_f > 0:
                    threshold = max(self.surge_threshold_pct, vol_f * self.vol_multiplier)
            except (TypeError, ValueError):
                pass

        sym_norm = clean_guard_symbol_key(symbol)
        state_key = sym_norm if side_u == "BUY" else f"{sym_norm}:SELL"
        raw_st = self._states.get(state_key)
        if not isinstance(raw_st, dict):
            st = {
                "peak": current_price,
                "surge_active": False,
                "surge_start_px": current_price,
                "ts": ts_now,
                "disarmed_peak_ts": 0.0,
            }
            self._states[state_key] = st
        else:
            st = raw_st
            try:
                pk_f = float(st.get("peak")) if not isinstance(st.get("peak"), bool) else 0.0
                if not (math.isfinite(pk_f) and pk_f > 0):
                    st["peak"] = current_price
                else:
                    st["peak"] = pk_f
            except (TypeError, ValueError):
                st["peak"] = current_price
            try:
                ts_f = float(st.get("ts")) if not isinstance(st.get("ts"), bool) else 0.0
                st["ts"] = ts_f if (math.isfinite(ts_f) and ts_f >= 0.0) else 0.0
            except (TypeError, ValueError):
                st["ts"] = 0.0
            try:
                dp_f = float(st.get("disarmed_peak_ts")) if not isinstance(st.get("disarmed_peak_ts"), bool) else 0.0
                st["disarmed_peak_ts"] = dp_f if (math.isfinite(dp_f) and dp_f >= 0.0) else 0.0
            except (TypeError, ValueError):
                st["disarmed_peak_ts"] = 0.0
            st["surge_active"] = bool(st.get("surge_active", False))

        # Reset clock-skewed future timestamps
        if ts_now > 0 and float(st.get("disarmed_peak_ts", 0.0)) > ts_now + 5.0:
            st["disarmed_peak_ts"] = 0.0

        # Expire stale peak/trough state once a full window has elapsed without new extremes (or on backward clock skew)
        if ts_now > 0 and st.get("ts", 0) > 0 and (
            (ts_now - st["ts"]) >= self.window_seconds or st["ts"] > ts_now + 5.0
        ):
            st["surge_active"] = False
            st["peak"] = current_price
            st["surge_start_px"] = current_price
            st["ts"] = ts_now

        was_active = bool(st["surge_active"])

        # Update extreme (peak for BUY, trough for SELL) if current price extends an active move
        if was_active:
            if (side_u == "BUY" and current_price > st["peak"]) or (side_u == "SELL" and current_price < st["peak"]):
                st["peak"] = current_price
                st["ts"] = ts_now
        else:
            st["peak"] = current_price
            st["ts"] = ts_now

        # Check chronological directional move in rolling window if history provided
        if len(history_seq) >= 2:
            window_start_ts = max(ts_now - self.window_seconds, float(st.get("disarmed_peak_ts", 0.0)))
            max_allowed_ts = (ts_now + 5.0) if ts_now > 0 else float("inf")
            window_pts = []
            for item in history_seq:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    raw_t = item[0]
                    raw_p = item[1]
                    if isinstance(raw_t, bool) or isinstance(raw_p, bool):
                        continue
                    t_f = float(raw_t)
                    p_f = float(raw_p)
                    if math.isfinite(t_f) and math.isfinite(p_f) and p_f > 0 and window_start_ts <= t_f <= max_allowed_ts:
                        window_pts.append((t_f, p_f))
                except (TypeError, ValueError, IndexError):
                    continue
            if len(window_pts) > 1:
                window_pts.sort(key=lambda x: x[0])
            if ts_now >= window_start_ts and (not window_pts or window_pts[-1] != (float(ts_now), float(current_price))):
                window_pts.append((float(ts_now), float(current_price)))
            if window_pts:
                running_anchor = window_pts[0][1]
                best_move_pct = 0.0
                surge_start_px = running_anchor
                surge_extreme_px = window_pts[0][1]
                surge_extreme_ts = window_pts[0][0]
                for t, p in window_pts:
                    if side_u == "BUY":
                        if p < running_anchor:
                            running_anchor = p
                        move_pct = (p - running_anchor) / running_anchor * 100.0 if running_anchor > 0 else 0.0
                    else:
                        if p > running_anchor:
                            running_anchor = p
                        move_pct = (running_anchor - p) / running_anchor * 100.0 if running_anchor > 0 else 0.0
                    if move_pct >= best_move_pct:
                        best_move_pct = move_pct
                        surge_start_px = running_anchor
                        surge_extreme_px = p
                        surge_extreme_ts = t
                if best_move_pct >= threshold:
                    st["surge_active"] = True
                    st["surge_start_px"] = surge_start_px
                    if was_active:
                        st["peak"] = max(st["peak"], surge_extreme_px) if side_u == "BUY" else min(st["peak"], surge_extreme_px)
                    else:
                        st["peak"] = surge_extreme_px
                    st["ts"] = surge_extreme_ts
                elif st["surge_active"]:
                    if (side_u == "BUY" and surge_extreme_px > st["peak"]) or (side_u == "SELL" and surge_extreme_px < st["peak"]):
                        st["peak"] = surge_extreme_px
                        st["ts"] = surge_extreme_ts

        # If surge/dump is active, verify if price has retraced sufficiently from extreme
        # or if the spike has consolidated over the full window without making new extremes
        if st["surge_active"]:
            extreme = st["peak"]
            if side_u == "BUY":
                pullback_pct = (extreme - current_price) / extreme * 100.0 if extreme > 0 else 0.0
            else:
                pullback_pct = (current_price - extreme) / extreme * 100.0 if extreme > 0 else 0.0

            if ts_now > 0 and st.get("ts", 0) > 0 and (ts_now - st["ts"]) >= self.window_seconds:
                # Spike has consolidated past the window without new extremes; disarmed
                st["surge_active"] = False
                st["disarmed_peak_ts"] = max(float(st.get("disarmed_peak_ts", 0.0)), float(st.get("ts", 0.0)))
                st["peak"] = current_price
                st["ts"] = ts_now
            elif pullback_pct < self.pullback_required_pct:
                label = "parabolic_surge_active" if side_u == "BUY" else "parabolic_dump_active"
                ext_label = "peak" if side_u == "BUY" else "trough"
                return GuardDecision.defer(
                    "ParabolicSurgeGuard",
                    f"{label} ({ext_label}={extreme:.2f}, current={current_price:.2f}, "
                    f"pullback={pullback_pct:.2f}% < required {self.pullback_required_pct:.2f}%)",
                    peak=extreme,
                    pullback_pct=pullback_pct,
                    required_pullback=self.pullback_required_pct,
                )
            else:
                # Retracement achieved, surge/dump disarmed
                st["surge_active"] = False
                st["disarmed_peak_ts"] = max(float(st.get("disarmed_peak_ts", 0.0)), float(st.get("ts", 0.0)))
                st["peak"] = current_price
                st["ts"] = ts_now

        return GuardDecision.allow("ParabolicSurgeGuard", "normal_market_structure")
