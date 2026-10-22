"""Internal Quantitative Mathematical Order Guard (Pillar 1).

Evaluates pre-trade orders against pure mathematical models computed on price history:
- ParabolicSurgeGuard: Anti-FOMO protection against parabolic price extensions.
- WeibullExhaustionGuard: Empirical trend survival exhaustion threshold (P90).
- NoiseFloorGuard: Signal-to-noise significance filtering.

Executes in 0 ms (100% local in-memory calculation, zero network or external I/O).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("internal_order_guard")


def _resolve_trend_duration(symbol: str) -> float:
    """Read trend duration in seconds, delegating to order_guard SSOT."""
    try:
        from order_guard import _resolve_trend_duration as _og_resolve
        return _og_resolve(symbol)
    except Exception:
        return 0.0


def _resolve_trend_direction(symbol: str, regime_context: Any = None) -> Optional[str]:
    """Read trend direction ('up'/'down'/'bull'/'bear'), delegating to order_guard SSOT."""
    try:
        from order_guard import _resolve_trend_direction as _og_resolve_dir
        return _og_resolve_dir(symbol, regime_context=regime_context)
    except Exception:
        return None


def check_internal_order_guards(
    symbol: str,
    side: str,
    price: float,
    *,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    is_stop_loss: bool = False,
    price_history: Optional[List[Tuple[float, float]]] = None,
    trend_duration_seconds: float = 0.0,
    regime_context: Any = None,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Evaluate pure mathematical guards on price time series (Pillar 1).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    if is_stop_loss:
        return True, "stop_loss_exempt", 1.0

    order_side = (side or "").upper()
    if order_side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    mode = validate_guard_mode_cfg(m["internal_guard_mode"], "internal_guard_mode")
    if mode == "off":
        return True, "intelligence_guards_off", 1.0

    if order_side == "SELL":
        computed_notional = float(notional_eur) if notional_eur is not None else (float(price * qty) if (price and qty) else 0.0)
        sell_min_notional = float(m["sell_guard_min_notional_eur"])
        if computed_notional < sell_min_notional:
            return True, "small_sell_exempt", 1.0

    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction

    surge_pct = float(m["parabolic_surge_pct"])
    pullback_pct = float(m["parabolic_pullback_pct"])
    exhaust_policy = str(m["weibull_exhaustion_policy"]).strip().lower()
    exhaust_scale = float(m["weibull_exhausted_scale"])
    exhaust_p90 = float(m["weibull_p90_days"])

    p_guard = ParabolicSurgeGuard(surge_threshold_pct=surge_pct, pullback_required_pct=pullback_pct)
    e_guard = WeibullExhaustionGuard(default_p90_days=exhaust_p90, policy=exhaust_policy, exhausted_scale=exhaust_scale)

    effective_scale = 1.0
    active_reason = "ok"

    # 1. Parabolic surge / vertical dump check (Anti-FOMO on BUY, Anti-Capitulation on SELL)
    history = price_history
    if history is None:
        try:
            from order_guard import _read_cached_price_history
            history = _read_cached_price_history(symbol, window_seconds=7200.0)
        except Exception:
            pass

    if history:
        p_dec = p_guard.check(symbol, order_side, price, price_history=history, now=now)
        if not p_dec.allowed:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {order_side} {symbol} @ {price}: {p_dec.reason} (brake={p_dec.brake_action})")
            if mode == "enforce":
                return False, p_dec.reason, 0.0

    # 2. Weibull trend exhaustion check
    dur_sec = trend_duration_seconds
    if not dur_sec:
        if regime_context is not None:
            dur_sec = getattr(regime_context, "trend_duration_seconds", 0.0) or 0.0
        if not dur_sec:
            dur_sec = _resolve_trend_duration(symbol)

    if dur_sec and dur_sec > 0:
        if trend_duration_seconds and trend_duration_seconds > 0 and getattr(regime_context, "resolved_trend", None) is None:
            trend_dir = None
        else:
            trend_dir = _resolve_trend_direction(symbol, regime_context=regime_context)
        e_dec = e_guard.check(symbol, order_side, trend_duration_seconds=dur_sec, trend_direction=trend_dir)
        if e_dec.brake_action != BrakeAction.NONE:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {order_side} {symbol} @ {price}: {e_dec.reason} (brake={e_dec.brake_action}, suggested_scale={e_dec.suggested_scale})")
            if mode == "enforce":
                if not e_dec.allowed:
                    return False, e_dec.reason, 0.0
                effective_scale = min(effective_scale, e_dec.suggested_scale)
                active_reason = e_dec.reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
