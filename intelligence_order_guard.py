"""Intelligence & LLM Pre-Trade Order Guard Coordinator (Pillar 3).

Evaluates high-stake orders and macro threat context using LLM reasoning:
- HighStakeGuard: Real-time LLM validation for high-notional orders (BUY >= 1000 EUR).
  Sub-1000 EUR orders pass with 0 ms latency (zero overhead).
- GeopoliticalShockGuard: Evaluates acute conflict, war escalation, and energy supply crisis threats.
- Sentiment Advisor context: Incorporates 3h macro sentiment evaluation (sentiment_advisor_eval.json).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from geopolitical_order_guard import check_geopolitical_order_guards

logger = logging.getLogger("intelligence_order_guard")

_LLM_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str], float] = {}


def check_intelligence_order_guards(
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
    """Evaluate external microstructure, geopolitical shock, and high-stake LLM guards (Pillars 2, 3, 4).

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

    # Compute notional EUR
    computed_notional = float(notional_eur) if notional_eur is not None else float(price * (qty or 1.0))

    # 1. High-Stake LLM Pre-Trade Guard (Pillar 3)
    llm_mode = str(m.get("llm_guard_mode", m.get("high_stake_guard_mode", m.get("gemini_guard_mode", "shadow")))).strip().lower()
    if llm_mode not in ("off", "0", "disabled"):
        min_notional = float(m.get("llm_min_notional_eur", m.get("high_stake_min_notional_eur", m.get("gemini_min_notional_eur", 1000.0))))
        if computed_notional >= min_notional:
            valid_context = regime_context is not None and hasattr(regime_context, "__dict__")
            prev_notional = getattr(regime_context, "_gemini_evaluated_notional", None) if valid_context else None
            needs_eval = (prev_notional is None) or (abs(computed_notional - prev_notional) > 1.0)
            if needs_eval:
                dur_sec = getattr(regime_context, "resolved_duration_sec", None) if valid_context else None
                from intelligence.sentiment.guards import high_stake_guard
                if getattr(high_stake_guard, "GeminiHighStakeGuard", None) not in (
                    high_stake_guard.LLMHighStakeGuard,
                    getattr(high_stake_guard, "_ORIGINAL_LLM_HIGH_STAKE_GUARD", None)
                ):
                    GuardCls = high_stake_guard.GeminiHighStakeGuard
                else:
                    GuardCls = high_stake_guard.HighStakeGuard
                timeout_sec = float(m.get("llm_timeout_sec", m.get("high_stake_timeout_sec", m.get("gemini_timeout_sec", 25.0))))
                fallback = str(m.get("llm_fallback", m.get("high_stake_fallback", m.get("gemini_fallback", "allow")))).strip().lower()
                llm_guard = GuardCls(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
                try:
                    g_dec = llm_guard.check(
                        symbol,
                        side,
                        price,
                        qty if qty is not None else 1.0,
                        notional_eur=computed_notional,
                        regime_context=regime_context if valid_context else None,
                        dur_sec=dur_sec,
                    )
                except TypeError:
                    g_dec = llm_guard.check(
                        symbol,
                        side,
                        price,
                        qty if qty is not None else 1.0,
                        notional_eur=computed_notional,
                    )
                if valid_context:
                    try:
                        object.__setattr__(regime_context, "_gemini_evaluated_notional", computed_notional)
                        object.__setattr__(regime_context, "_gemini_decision", (g_dec.allowed, g_dec.reason, g_dec.suggested_scale))
                    except Exception:
                        pass

                prefix = "[LLM_GUARD_SHADOW]" if llm_mode == "shadow" else "[LLM_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} (brake={g_dec.brake_action}, suggested_scale={g_dec.suggested_scale})")

                # Shadow Phone Alert via ntfy
                now_ts = float(now) if now is not None else time.time()
                cd_key = (symbol, side)
                if now_ts - _LLM_SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                    _LLM_SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                    try:
                        from notify_engine.alertnotifiers import notify
                        scale_pct = int(g_dec.suggested_scale * 100)
                        if not g_dec.allowed:
                            g_title = f"🛡 [P3 · HIGH-STAKE] BLOCK {side} {symbol}"
                        elif g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                            g_title = f"🛡 [P3 · HIGH-STAKE] SCALE {scale_pct}% {side} {symbol}"
                        else:
                            g_title = f"🧭 [P3 · HIGH-STAKE] ACCEPT {side} {symbol}"

                        clean_reason = g_dec.reason
                        for pfx in (
                            "Gemini vetoed high-stake BUY: ", "LLM vetoed high-stake BUY: ",
                            "Gemini approved high-stake BUY: ", "LLM approved high-stake BUY: ",
                            "Gemini downscaled high-stake BUY: ", "LLM downscaled high-stake BUY: ",
                        ):
                            if clean_reason.startswith(pfx):
                                clean_reason = clean_reason[len(pfx):].strip()
                                break

                        price_str = f"${price:,.2f}" if price else ""
                        qty_str = f"{qty:g} " if qty else ""
                        order_header = f"Order: {side} {qty_str}{symbol}"
                        if price_str:
                            order_header += f" @ {price_str}"
                        order_header += f" · €{computed_notional:,.0f} (Scale: {scale_pct}%)"
                        g_body = f"{order_header}\n{clean_reason}"
                        notify(
                            title=g_title,
                            body=g_body,
                            source="order_guard",
                            symbol=symbol,
                        )
                    except Exception:
                        pass
                if llm_mode == "enforce":
                    if not g_dec.allowed:
                        return False, g_dec.reason, 0.0
                    effective_scale = min(effective_scale, g_dec.suggested_scale)
                    active_reason = g_dec.reason

    # 2. Geopolitical & Energy Shock Guard (Pillar 3 Sub-Guard)
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

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
