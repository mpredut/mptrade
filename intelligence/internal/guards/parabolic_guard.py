"""Parabolic surge guard: prevents buying at the peak of vertical price spikes (Anti-FOMO)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from intelligence.internal.guards.guard_decision import GuardDecision


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
        self.surge_threshold_pct = float(surge_threshold_pct)
        self.pullback_required_pct = float(pullback_required_pct)
        self.window_seconds = float(window_seconds)
        self.vol_multiplier = float(vol_multiplier)

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
        """Evaluate whether a parabolic surge blocks or defers the proposed order side."""
        side_u = str(side or "").upper()
        if side_u != "BUY":
            # Parabolic spikes are favorable for exits/sells, never blocked.
            return GuardDecision.allow("ParabolicSurgeGuard", "non_buy_order")

        if not math.isfinite(current_price) or current_price <= 0:
            return GuardDecision.allow("ParabolicSurgeGuard", "invalid_price_skipped")

        ts_now = now or (price_history[-1][0] if price_history else 0.0)

        # Dynamic threshold based on volatility if available
        threshold = self.surge_threshold_pct
        if volatility_1h_pct is not None and volatility_1h_pct > 0:
            threshold = max(self.surge_threshold_pct, volatility_1h_pct * self.vol_multiplier)

        st = self._states.setdefault(
            symbol,
            {
                "peak": current_price,
                "surge_active": False,
                "surge_start_px": current_price,
                "ts": ts_now,
                "disarmed_peak_ts": 0.0,
            },
        )

        # Expire stale peak/surge state once a full window has elapsed without new highs
        if ts_now > 0 and st.get("ts", 0) > 0 and (ts_now - st["ts"]) >= self.window_seconds:
            st["surge_active"] = False
            st["peak"] = current_price
            st["surge_start_px"] = current_price
            st["ts"] = ts_now

        # Update peak if current price makes a new high
        if current_price > st["peak"]:
            st["peak"] = current_price
            st["ts"] = ts_now

        # Check chronological upward move in rolling window if history provided
        if price_history and len(price_history) >= 2:
            window_start_ts = max(ts_now - self.window_seconds, float(st.get("disarmed_peak_ts", 0.0)))
            window_pts = [(float(t), float(p)) for (t, p) in price_history if t >= window_start_ts and p > 0]
            if ts_now >= window_start_ts and (not window_pts or window_pts[-1] != (float(ts_now), float(current_price))):
                window_pts.append((float(ts_now), float(current_price)))
            if window_pts:
                running_low = window_pts[0][1]
                best_move_pct = 0.0
                surge_low_px = running_low
                surge_peak_px = window_pts[0][1]
                surge_peak_ts = window_pts[0][0]
                for t, p in window_pts:
                    if p < running_low:
                        running_low = p
                    if running_low > 0:
                        move_pct = (p - running_low) / running_low * 100.0
                        if move_pct >= best_move_pct:
                            best_move_pct = move_pct
                            surge_low_px = running_low
                            surge_peak_px = p
                            surge_peak_ts = t
                if best_move_pct >= threshold:
                    st["surge_active"] = True
                    st["surge_start_px"] = surge_low_px
                    st["peak"] = max(st["peak"], surge_peak_px)
                    st["ts"] = surge_peak_ts
                elif st["surge_active"] and surge_peak_px > st["peak"]:
                    st["peak"] = surge_peak_px
                    st["ts"] = surge_peak_ts

        # If surge is active, verify if price has pulled back sufficiently from peak
        # or if the spike has consolidated over the full window without making new highs
        if st["surge_active"]:
            peak = st["peak"]
            pullback_pct = (peak - current_price) / peak * 100.0 if peak > 0 else 0.0

            if ts_now > 0 and st.get("ts", 0) > 0 and (ts_now - st["ts"]) >= self.window_seconds:
                # Spike has consolidated past the window without new highs; surge disarmed
                st["surge_active"] = False
                st["disarmed_peak_ts"] = max(float(st.get("disarmed_peak_ts", 0.0)), float(st.get("ts", 0.0)))
                st["peak"] = current_price
                st["ts"] = ts_now
            elif pullback_pct < self.pullback_required_pct:
                return GuardDecision.defer(
                    "ParabolicSurgeGuard",
                    f"parabolic_surge_active (peak={peak:.2f}, current={current_price:.2f}, "
                    f"pullback={pullback_pct:.2f}% < required {self.pullback_required_pct:.2f}%)",
                    peak=peak,
                    pullback_pct=pullback_pct,
                    required_pullback=self.pullback_required_pct,
                )
            else:
                # Pullback achieved, surge disarmed
                st["surge_active"] = False
                st["disarmed_peak_ts"] = max(float(st.get("disarmed_peak_ts", 0.0)), float(st.get("ts", 0.0)))
                st["peak"] = current_price
                st["ts"] = ts_now

        return GuardDecision.allow("ParabolicSurgeGuard", "normal_market_structure")
