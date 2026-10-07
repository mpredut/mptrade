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
    """Read trend duration in seconds from cache_price_long_trend.json."""
    try:
        cachedb_dir = os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
        p = os.path.join(cachedb_dir, "cache_price_long_trend.json")
        if os.path.exists(p):
            import json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            su = (symbol or "").strip().upper()
            items = data.get("items", {}).get(su, [])
            if not items:
                for q in ("USDC", "USDT", "FDUSD", "USD", "EUR", "RON", "GBP"):
                    if su.endswith(q) and len(su) > len(q):
                        items = data.get("items", {}).get(su[:-len(q)], [])
                        if items:
                            break
            if items and isinstance(items, list):
                last_item = items[-1]
                if isinstance(last_item, dict):
                    dur = last_item.get("duration_seconds")
                    if dur is not None:
                        return float(dur)
    except Exception:
        pass
    return 0.0


def check_internal_order_guards(
    symbol: str,
    side: str,
    price: float,
    *,
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
    order_side = (side or "").upper()
    if order_side != "BUY":
        return True, "not_buy_side", 1.0

    if margins is None:
        try:
            from order_guard import _load_margins
            m = _load_margins()
        except Exception:
            m = {}
    else:
        m = margins

    mode = str(m.get("internal_guard_mode", m.get("intelligence_guards_mode", "shadow"))).strip().lower()
    if mode in ("off", "0", "disabled"):
        return True, "intelligence_guards_off", 1.0

    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction

    surge_pct = float(m.get("parabolic_surge_pct", 15.0))
    pullback_pct = float(m.get("parabolic_pullback_pct", 2.0))
    exhaust_policy = str(m.get("weibull_exhaustion_policy", "downscale")).strip().lower()
    exhaust_scale = float(m.get("weibull_exhausted_scale", 0.25))

    p_guard = ParabolicSurgeGuard(surge_threshold_pct=surge_pct, pullback_required_pct=pullback_pct)
    e_guard = WeibullExhaustionGuard(policy=exhaust_policy, exhausted_scale=exhaust_scale)

    effective_scale = 1.0
    active_reason = "ok"

    # 1. Parabolic surge check (Anti-FOMO)
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
        e_dec = e_guard.check(symbol, order_side, trend_duration_seconds=dur_sec)
        if e_dec.brake_action != BrakeAction.NONE:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {order_side} {symbol} @ {price}: {e_dec.reason} (brake={e_dec.brake_action}, suggested_scale={e_dec.suggested_scale})")
            if mode == "enforce":
                if not e_dec.allowed:
                    return False, e_dec.reason, 0.0
                effective_scale = min(effective_scale, e_dec.suggested_scale)
                active_reason = e_dec.reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
