"""Macro, External Microstructure, and LLM Intelligence Order Guard Coordinator.

Composes and orchestrates:
- microstructure_order_guard.py (Pillar 2: Orderbook Depth, Funding Crowding, Whale Flow)
- geopolitical_order_guard.py (Pillar 4: Geopolitical & Macro Shock Shield)
- GeminiHighStakeGuard (Pillar 3: Event-driven LLM vetting for BUY >= 1000 EUR)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from microstructure_order_guard import (
    check_microstructure_order_guards,
    get_derivatives_collector,
    get_orderbook_collector,
    get_whale_collector,
)
from geopolitical_order_guard import check_geopolitical_order_guards

logger = logging.getLogger("macro_order_guard")

_GEMINI_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str], float] = {}


def check_macro_order_guards(
    provider: Optional[str],
    symbol: str,
    order_type: str,
    price: float,
    *,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    regime_context: Any = None,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Evaluate external microstructure, geopolitical shock, and high-stake LLM guards.

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    side = (order_type or "").upper()
    if side != "BUY":
        return True, "not_buy_side", 1.0

    if margins is None:
        try:
            from order_guard import _load_margins
            m = _load_margins()
        except Exception:
            m = {}
    else:
        m = margins

    from intelligence.internal.guards.guard_decision import BrakeAction

    effective_scale = 1.0
    active_reason = "ok"
    shadow_notify = bool(int(float(m.get("shadow_notify", 1.0))))
    computed_notional = notional_eur
    if computed_notional is None and qty is not None and price > 0:
        computed_notional = price * qty

    # 1. LLM Pre-Trade Guard (Pillar 3: >= 1000 EUR purchases)
    llm_mode = str(
        m.get("llm_guard_mode",
        m.get("high_stake_guard_mode",
        m.get("macro_stake_guard_mode",
        m.get("macro_guard_mode",
        m.get("gemini_guard_mode", "shadow")))))
    ).strip().lower()

    if llm_mode not in ("off", "0", "disabled"):
        min_notional = float(
            m.get("llm_min_notional_eur",
            m.get("high_stake_min_notional_eur",
            m.get("macro_stake_min_notional_eur",
            m.get("gemini_min_notional_eur", 1000.0))))
        )
        timeout_sec = float(
            m.get("llm_timeout_sec",
            m.get("high_stake_timeout_sec",
            m.get("gemini_timeout_sec", 25.0)))
        )
        fallback = str(
            m.get("llm_fallback",
            m.get("high_stake_fallback",
            m.get("gemini_fallback", "allow")))
        ).strip().lower()

        if computed_notional is not None and computed_notional >= min_notional:
            from intelligence.sentiment.guards import gemini_high_stake_guard
            if getattr(gemini_high_stake_guard, "GeminiHighStakeGuard", None) not in (gemini_high_stake_guard.LLMHighStakeGuard, getattr(gemini_high_stake_guard, "_ORIGINAL_LLM_HIGH_STAKE_GUARD", None)):
                GuardCls = gemini_high_stake_guard.GeminiHighStakeGuard
            else:
                GuardCls = gemini_high_stake_guard.LLMHighStakeGuard
            llm_guard = GuardCls(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
            try:
                g_dec = llm_guard.check(
                    symbol,
                    side,
                    price,
                    qty if qty is not None else 1.0,
                    notional_eur=computed_notional,
                    regime_context=regime_context,
                )
            except TypeError:
                g_dec = llm_guard.check(
                    symbol,
                    side,
                    price,
                    qty if qty is not None else 1.0,
                    notional_eur=computed_notional,
                )
            if regime_context is not None:
                try:
                    object.__setattr__(regime_context, "_gemini_evaluated_notional", computed_notional)
                    object.__setattr__(regime_context, "_gemini_decision", (g_dec.allowed, g_dec.reason, g_dec.suggested_scale))
                except Exception:
                    pass
            prefix = "[LLM_GUARD_SHADOW]" if llm_mode == "shadow" else "[LLM_GUARD_ENFORCE]"
            print(f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} (brake={g_dec.brake_action}, suggested_scale={g_dec.suggested_scale})")
            if llm_mode == "shadow" and shadow_notify:
                cd_key = (symbol, side)
                now_ts = now if now is not None else time.time()
                if now_ts - _GEMINI_SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                    _GEMINI_SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                    try:
                        from notify_engine.alertnotifiers import notify
                        if not g_dec.allowed:
                            g_title = f"🛡 [LLM SHADOW VETO] Would Block {side} {symbol}"
                        elif g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                            g_title = f"🛡 [LLM SHADOW DOWNSCALE] Would Scale {int(g_dec.suggested_scale*100)}% {side} {symbol}"
                        else:
                            g_title = f"🧭 [LLM SHADOW APPROVE] Approved {side} {symbol}"
                        notify(
                            title=g_title,
                            body=f"High-stake order €{computed_notional:.2f} evaluated: {g_dec.reason} (scale={g_dec.suggested_scale})",
                            source="macro_shadow",
                            symbol=symbol,
                        )
                    except Exception:
                        pass
                if llm_mode == "enforce":
                    if not g_dec.allowed:
                        return False, g_dec.reason, 0.0
                    effective_scale = min(effective_scale, g_dec.suggested_scale)
                    active_reason = g_dec.reason

    # 2. Geopolitical & Energy Shock Guard (Pillar 4)
    geo_ok, geo_reason, geo_scale = check_geopolitical_order_guards(
        symbol=symbol,
        side=side,
        margins=m,
        now=now,
    )
    if not geo_ok:
        return False, geo_reason, 0.0
    if geo_scale < 1.0:
        effective_scale = min(effective_scale, geo_scale)
        active_reason = geo_reason

    # 3. External Microstructure Guards (Pillar 2: Wall, Funding, Whale)
    micro_ok, micro_reason, micro_scale = check_microstructure_order_guards(
        symbol=symbol,
        side=side,
        price=price,
        margins=m,
        now=now,
    )
    if not micro_ok:
        return False, micro_reason, 0.0
    if micro_scale < 1.0:
        effective_scale = min(effective_scale, micro_scale)
        active_reason = micro_reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
