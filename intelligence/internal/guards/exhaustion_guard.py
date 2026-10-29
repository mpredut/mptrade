"""Weibull trend exhaustion guard: prevents entering trends that have reached their late-life percentile."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from botcore import safe_pos_float
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    normalize_guard_side,
)


def calibrate_weibull_params(median_days: float, p90_days: float) -> Tuple[float, float]:
    """Calibrate Weibull shape (k) and scale (lambda) from median (S=0.5) and P90 (S=0.1) durations.

    Survival function: S(t) = exp(-(t / lambda)^k)
    Hazard function:   h(t) = (k / lambda) * (t / lambda)^(k - 1)
    """
    med_in = safe_pos_float(median_days, default=3.0) or 3.0
    p90_in = safe_pos_float(p90_days, default=7.0) or 7.0
    med = max(0.1, med_in)
    p90 = max(med * 1.05, p90_in)
    ln2 = math.log(2.0)
    ln10 = math.log(10.0)
    ratio = p90 / med
    k = math.log(ln10 / ln2) / math.log(ratio) if ratio > 1.0 else 1.5
    k = max(0.2, min(k, 10.0))
    lam = med / (ln2 ** (1.0 / k))
    return k, lam


def weibull_survival_and_hazard(duration_days: float, median_days: float, p90_days: float) -> Tuple[float, float, float, float]:
    """Return (survival_prob S(t), hazard_rate h(t), shape k, scale lambda) at duration_days."""
    dur_d = safe_pos_float(duration_days, default=0.0) or 0.0
    if dur_d <= 0.0:
        k, lam = calibrate_weibull_params(median_days, p90_days)
        return 1.0, 0.0, k, lam
    k, lam = calibrate_weibull_params(median_days, p90_days)
    u = dur_d / max(lam, 1e-6)
    exponent = min(100.0, u ** k)
    survival = math.exp(-exponent)
    hazard = (k / max(lam, 1e-6)) * (u ** (k - 1.0))
    return survival, hazard, k, lam


class WeibullExhaustionGuard:
    """Exhaustion guard that calculates whether a persistent trend has reached statistical maturity.

    Uses the asymmetric Weibull survival distribution S(t) = exp(-(t / lambda)^k), which differs
    mathematically from the symmetric Gaussian bell curve (G_QLIMIT1):
    - G_QLIMIT1 (Gaussian + Lindy plateau) allocates the 24h cumulative portfolio budget across
      Zone 1 (0..T) and Zone 2 (T..1.4*T).
    - G_QLIMIT2 (WeibullExhaustionGuard) models the late-tail hazard / wear-out rate beyond the
      active Gaussian horizon (or beyond P90 when evaluated standalone), downscaling or vetoing
      individual orders when survival probability S(t) drops below 10% (t > limit_p90).
    """

    def __init__(
        self,
        default_p90_days: float = 7.0,
        policy: str = "downscale",  # "downscale" | "veto" | "off"
        exhausted_scale: float = 0.25,
    ):
        self.default_p90_days = safe_pos_float(default_p90_days, default=7.0) or 7.0
        self.policy = str(policy or "downscale").strip().lower() if not isinstance(policy, bool) else "downscale"
        self.exhausted_scale = min(1.0, safe_pos_float(exhausted_scale, default=0.25) or 0.25)

    def check(
        self,
        symbol: str,
        side: str,
        trend_duration_seconds: float,
        p90_days: Optional[float] = None,
        median_days: Optional[float] = None,
        trend_direction: Optional[str] = None,
        gaussian_horizon_days: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate if the trend is mature/exhausted for the requested side."""
        if self.policy == "off":
            return GuardDecision.allow("WeibullExhaustionGuard", "weibull_policy_off")

        side_u = normalize_guard_side(side)
        if side_u not in ("BUY", "SELL"):
            return GuardDecision.allow("WeibullExhaustionGuard", "unsupported_side")

        if trend_direction is not None and not isinstance(trend_direction, bool):
            raw_dir = str(trend_direction).strip().lower()
            if raw_dir not in ("", "unknown", "none"):
                if raw_dir in ("up", "bull", "uptrend", "bullish", "long"):
                    dir_norm = "bull"
                elif raw_dir in ("down", "bear", "downtrend", "bearish", "short"):
                    dir_norm = "bear"
                else:
                    dir_norm = "flat"
                if (side_u == "BUY" and dir_norm != "bull") or (side_u == "SELL" and dir_norm != "bear"):
                    return GuardDecision.allow(
                        "WeibullExhaustionGuard",
                        f"trend_direction_favorable ({dir_norm})",
                    )

        dur_sec = safe_pos_float(trend_duration_seconds, default=None)
        if dur_sec is None:
            return GuardDecision.allow("WeibullExhaustionGuard", "no_duration_data")

        duration_days = dur_sec / 86400.0
        base_p90 = safe_pos_float(p90_days, default=self.default_p90_days) or self.default_p90_days
        med_d = safe_pos_float(median_days, default=None) or min(3.0, base_p90 * 0.5)

        # Avoid overlapping with G_QLIMIT1's Gaussian Zone 1 & Zone 2 (0 .. 1.4*T) when provided:
        limit_p90 = base_p90
        gauss_h = safe_pos_float(gaussian_horizon_days, default=None)
        if gauss_h is not None:
            limit_p90 = max(limit_p90, gauss_h)

        survival_prob, hazard_rate, k_shape, lam_scale = weibull_survival_and_hazard(
            duration_days, med_d, limit_p90
        )

        if duration_days > limit_p90:
            reason = (
                f"trend_exhausted (duration={duration_days:.1f}d > P90={limit_p90:.1f}d, "
                f"median={med_d:.1f}d, S_weibull={survival_prob:.2f}, k={k_shape:.2f})"
            )
            if self.policy in ("veto", "block", "defer"):
                return GuardDecision.veto(
                    "WeibullExhaustionGuard",
                    reason,
                    duration_days=duration_days,
                    p90_days=limit_p90,
                    weibull_survival=round(survival_prob, 4),
                    weibull_hazard=round(hazard_rate, 4),
                    weibull_k=round(k_shape, 3),
                    weibull_lambda=round(lam_scale, 3),
                )
            else:
                return GuardDecision.downscale(
                    "WeibullExhaustionGuard",
                    scale=self.exhausted_scale,
                    reason=reason,
                    duration_days=duration_days,
                    p90_days=limit_p90,
                    weibull_survival=round(survival_prob, 4),
                    weibull_hazard=round(hazard_rate, 4),
                    weibull_k=round(k_shape, 3),
                    weibull_lambda=round(lam_scale, 3),
                )

        return GuardDecision.allow(
            "WeibullExhaustionGuard",
            f"trend_healthy (duration={duration_days:.1f}d <= P90={limit_p90:.1f}d, S_weibull={survival_prob:.2f})",
            duration_days=duration_days,
            p90_days=limit_p90,
            weibull_survival=round(survival_prob, 4),
            weibull_hazard=round(hazard_rate, 4),
            weibull_k=round(k_shape, 3),
            weibull_lambda=round(lam_scale, 3),
        )
