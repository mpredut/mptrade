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
            {"peak": current_price, "surge_active": False, "surge_start_px": current_price, "ts": ts_now},
        )

        # Update peak
        if current_price > st["peak"]:
            st["peak"] = current_price
            st["ts"] = ts_now

        # Check window move if history provided
        if price_history and len(price_history) >= 2:
            window_start_ts = ts_now - self.window_seconds
            window_prices = [p for (t, p) in price_history if t >= window_start_ts and p > 0]
            if window_prices:
                low_px = min(window_prices)
                high_px = max(window_prices)
                if high_px > st["peak"]:
                    st["peak"] = high_px
                    st["ts"] = ts_now
                if low_px > 0:
                    window_move_pct = (st["peak"] - low_px) / low_px * 100.0
                    if window_move_pct >= threshold:
                        st["surge_active"] = True
                        st["surge_start_px"] = low_px

        # If surge is active, verify if price has pulled back sufficiently from peak
        # or if the spike has consolidated over the full window without making new highs
        if st["surge_active"]:
            peak = st["peak"]
            pullback_pct = (peak - current_price) / peak * 100.0 if peak > 0 else 0.0

            if ts_now > 0 and st.get("ts", 0) > 0 and (ts_now - st["ts"]) >= self.window_seconds:
                # Spike has consolidated past the window without new highs; surge disarmed
                st["surge_active"] = False
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

        return GuardDecision.allow("ParabolicSurgeGuard", "normal_market_structure")
