"""Parabolic surge guard: prevents buying at the peak of vertical price spikes (Anti-FOMO)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from botcore import safe_float, safe_nonneg_float, safe_pos_float
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    clean_guard_symbol_key,
    normalize_guard_side,
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
        self.surge_threshold_pct = safe_pos_float(surge_threshold_pct, default=4.0) or 4.0
        self.pullback_required_pct = safe_nonneg_float(pullback_required_pct, default=1.5)
        self.window_seconds = safe_pos_float(window_seconds, default=7200.0) or 7200.0
        self.vol_multiplier = safe_nonneg_float(vol_multiplier, default=3.0)

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

        current_price_f = safe_pos_float(current_price, default=None)
        if current_price_f is None:
            return GuardDecision.allow("ParabolicSurgeGuard", "invalid_price_skipped")
        current_price = current_price_f

        ts_now = safe_pos_float(now, default=0.0) or 0.0
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
                hf = safe_pos_float(item[0], default=None)
                if hf is not None:
                    ts_now = max(ts_now, hf)

        # Dynamic threshold based on volatility if available
        vol_f = safe_pos_float(volatility_1h_pct, default=None)
        threshold = max(self.surge_threshold_pct, vol_f * self.vol_multiplier) if vol_f is not None else self.surge_threshold_pct

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
            st["peak"] = safe_pos_float(st.get("peak"), default=current_price) or current_price
            st["ts"] = safe_nonneg_float(st.get("ts"), default=0.0) or 0.0
            st["disarmed_peak_ts"] = safe_nonneg_float(st.get("disarmed_peak_ts"), default=0.0) or 0.0
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
                t_f = safe_float(item[0], default=None)
                p_f = safe_pos_float(item[1], default=None)
                if t_f is not None and p_f is not None and window_start_ts <= t_f <= max_allowed_ts:
                    window_pts.append((t_f, p_f))
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
                post_surge_anchor: Optional[float] = None
                rearmed_after_pullback = False
                for t, p in window_pts:
                    if side_u == "BUY":
                        if p < running_anchor:
                            running_anchor = p
                        move_pct = (p - running_anchor) / running_anchor * 100.0 if running_anchor > 0 else 0.0
                    else:
                        if p > running_anchor:
                            running_anchor = p
                        move_pct = (running_anchor - p) / running_anchor * 100.0 if running_anchor > 0 else 0.0

                    if best_move_pct >= threshold and surge_extreme_px > 0:
                        extends_extreme = (p >= surge_extreme_px) if side_u == "BUY" else (p <= surge_extreme_px)
                        if extends_extreme:
                            best_move_pct = max(best_move_pct, move_pct)
                            surge_extreme_px = p
                            surge_extreme_ts = t
                            post_surge_anchor = None
                            continue
                        retrace_pct = (
                            (surge_extreme_px - p) / surge_extreme_px * 100.0
                            if side_u == "BUY"
                            else (p - surge_extreme_px) / surge_extreme_px * 100.0
                        )
                        if retrace_pct >= self.pullback_required_pct:
                            if post_surge_anchor is None:
                                post_surge_anchor = p
                            else:
                                post_surge_anchor = (
                                    min(post_surge_anchor, p) if side_u == "BUY" else max(post_surge_anchor, p)
                                )
                        if post_surge_anchor is not None and post_surge_anchor > 0:
                            re_move_pct = (
                                (p - post_surge_anchor) / post_surge_anchor * 100.0
                                if side_u == "BUY"
                                else (post_surge_anchor - p) / post_surge_anchor * 100.0
                            )
                            if re_move_pct >= threshold:
                                running_anchor = post_surge_anchor
                                best_move_pct = re_move_pct
                                surge_start_px = post_surge_anchor
                                surge_extreme_px = p
                                surge_extreme_ts = t
                                post_surge_anchor = None
                                rearmed_after_pullback = True
                                continue

                    if move_pct >= best_move_pct:
                        best_move_pct = move_pct
                        surge_start_px = running_anchor
                        surge_extreme_px = p
                        surge_extreme_ts = t
                if best_move_pct >= threshold:
                    st["surge_active"] = True
                    st["surge_start_px"] = surge_start_px
                    if was_active and not rearmed_after_pullback:
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
