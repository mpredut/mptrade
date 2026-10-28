"""Quantitative Limit Order Guards (G_QLIMIT1 & G_QLIMIT2).

Symmetric coordinator for internal quantitative limits computed from local price & trade history:
- G_QLIMIT1 (QuantityPolicyGuard): 24h cumulative portfolio exposure cap using the symmetric
  Gaussian bell curve + Lindy plateau over the empirical horizon T AUTO (0..T Zone 1, T..1.4T Zone 2).
- G_QLIMIT2 (InternalQuantLimitGuard): Per-order internal mathematical brakes:
  1) ParabolicSurgeGuard: 2h rolling spike/dump protection (Anti-FOMO on BUY, Anti-Capitulation on SELL).
  2) WeibullExhaustionGuard: Asymmetric Weibull survival hazard S(t) = exp(-(t/lambda)^k) active
     beyond the Gaussian persistence horizon (t > max(P90, 1.4*T)) so G_QLIMIT2 never overlaps
     with G_QLIMIT1's Zone 1 & Zone 2 allocation.

Executes in 0 ms (100% local in-memory calculation, zero network or external I/O).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from botcore import safe_nonneg_float
from intelligence.macro_sentiment_ssot import clean_scale, obj_attr, safe_pos_float
from providers.base import normalize_side

logger = logging.getLogger("qlimit_order_guard")


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


def resolve_cached_gaussian_horizon(symbol: str, exceed_percent: float = 0.4) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Read cached (p90_days, median_days, gaussian_horizon_days=T*(1+exceed_percent)) in 0 ms without network I/O."""
    try:
        from intelligence.internal.state.survival import (
            _effective_t_cache_file,
            _lookup_t_cache_entry,
            _read_t_cache_dict,
        )
        data = _read_t_cache_dict(_effective_t_cache_file())
        if isinstance(data, dict):
            ent = _lookup_t_cache_entry(data, symbol)
            if isinstance(ent, dict):
                import math

                p90_d = safe_pos_float(ent.get("p90_d"), 0.0) or None
                med_d = safe_pos_float(ent.get("median_d"), 0.0) or None
                t_val = safe_pos_float(ent.get("T"), 0.0) or None
                ex_pct = safe_nonneg_float(exceed_percent, default=0.4)
                if ex_pct is None:
                    ex_pct = 0.4
                gauss_h = round(t_val * (1.0 + ex_pct), 4) if t_val is not None else None
                return p90_d, med_d, gauss_h
    except Exception:
        pass
    return None, None, None


def check_qlimit1_weight_guard(
    symbol: str,
    side: str,
    price: float,
    required_qty: float,
    available_qty: float,
    traded_24h_value: float = 0.0,
    provider_name: str = "binance",
) -> float:
    """Evaluate G_QLIMIT1 (24h Gaussian weight-capped quantity)."""
    import order_guard
    weight = order_guard.resolve_trade_weight(
        symbol, side, provider_name=provider_name, raise_on_error=True
    )
    return order_guard.compute_weight_capped_qty(
        weight=weight,
        price=price,
        required_qty=required_qty,
        available_qty=available_qty,
        traded_24h_value=traded_24h_value,
    )


def check_qlimit2_order_guards(
    symbol: str,
    side: str,
    price: float,
    *,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    price_history: Optional[List[Tuple[float, float]]] = None,
    trend_duration_seconds: float = 0.0,
    regime_context: Any = None,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
    bypass_guards: Optional[set[str]] = None,
) -> Tuple[bool, str, float]:
    """Evaluate G_QLIMIT2 (ParabolicSurgeGuard + WeibullExhaustionGuard).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards) if bypass_guards else set()
    if "all" in bypassed or "*" in bypassed or GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(bypassed):
        return True, "stop_loss_exempt", 1.0

    order_side = normalize_side(side)
    if order_side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    m = _load_margins() if margins is None else margins

    if "intelligence_guards_mode" in m:
        master_mode = validate_guard_mode_cfg(m["intelligence_guards_mode"], "intelligence_guards_mode")
        if master_mode == "off":
            return True, "intelligence_guards_off", 1.0
    mode = validate_guard_mode_cfg(m["internal_guard_mode"], "internal_guard_mode")
    if mode == "off":
        return True, "intelligence_guards_off", 1.0

    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction

    surge_pct = m["parabolic_surge_pct"]
    pullback_pct = m["parabolic_pullback_pct"]
    exhaust_policy = str(m["weibull_exhaustion_policy"] or "").strip().lower()
    exhaust_scale = m["weibull_exhausted_scale"]
    exhaust_p90 = m["weibull_p90_days"]

    p_guard = ParabolicSurgeGuard(surge_threshold_pct=surge_pct, pullback_required_pct=pullback_pct)
    e_guard = WeibullExhaustionGuard(default_p90_days=exhaust_p90, policy=exhaust_policy, exhausted_scale=exhaust_scale)

    effective_scale = 1.0
    active_reason = "ok"

    # 1. Parabolic surge / vertical dump check (Anti-FOMO on BUY, Anti-Capitulation on SELL)
    history = price_history
    if history is None:
        try:
            from order_guard import _read_cached_price_history
            try:
                history = _read_cached_price_history(symbol, window_seconds=7200.0, now=now)
            except TypeError:
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

    # 2. Weibull trend exhaustion check (calibrated past G_QLIMIT1's Gaussian Zone 2 when resolved from live cache)
    dur_sec = safe_pos_float(trend_duration_seconds, 0.0)
    explicit_duration_override = dur_sec > 0.0
    if dur_sec <= 0.0:
        if regime_context is not None:
            dur_sec = safe_pos_float(obj_attr(regime_context, "trend_duration_seconds", 0.0), 0.0)
        if dur_sec <= 0.0:
            dur_sec = safe_pos_float(_resolve_trend_duration(symbol), 0.0)

    if dur_sec > 0.0:
        ctx_rt = obj_attr(regime_context, "resolved_trend") or obj_attr(regime_context, "regime")
        if explicit_duration_override and ctx_rt is None:
            trend_dir = None
        else:
            trend_dir = _resolve_trend_direction(symbol, regime_context=regime_context)

        gauss_horizon_d = None
        p90_d = None
        median_d = None
        if not explicit_duration_override:
            qlimit1_active = not GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(bypassed)
            p90_d, median_d, cached_gauss_h = resolve_cached_gaussian_horizon(symbol)
            if qlimit1_active and cached_gauss_h:
                gauss_horizon_d = cached_gauss_h

        e_dec = e_guard.check(
            symbol,
            order_side,
            trend_duration_seconds=dur_sec,
            p90_days=p90_d,
            median_days=median_d,
            trend_direction=trend_dir,
            gaussian_horizon_days=gauss_horizon_d,
        )
        if e_dec.brake_action != BrakeAction.NONE:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {order_side} {symbol} @ {price}: {e_dec.reason} (brake={e_dec.brake_action}, suggested_scale={e_dec.suggested_scale})")
            if mode == "enforce":
                if not e_dec.allowed:
                    return False, e_dec.reason, 0.0
                s_val = safe_pos_float(e_dec.suggested_scale, 0.0)
                if s_val <= 0.0:
                    return False, e_dec.reason, 0.0
                effective_scale = min(effective_scale, min(1.0, s_val))
                active_reason = e_dec.reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


check_internal_order_guards = check_qlimit2_order_guards
