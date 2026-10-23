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
import re
import time
from typing import Any, Dict, Optional, Tuple

from geopolitical_order_guard import check_geopolitical_order_guards

logger = logging.getLogger("intelligence_order_guard")

_LLM_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str], float] = {}


def evaluate_high_stake_order_guard(
    *,
    symbol: str,
    side: str,
    price: float,
    qty: Optional[float],
    computed_notional: Optional[float],
    is_stop_loss: bool,
    regime_context: Any,
    valid_context: bool,
    cache_key: Tuple[Any, ...],
    margins: Dict[str, Any],
    bypassed: set[str],
    now: Optional[float] = None,
    dur_sec: Optional[float] = None,
    fallback_notional: Optional[float] = None,
    notify_on_eval: bool = True,
) -> Tuple[bool, str, float]:
    """SSOT evaluator for HighStakeGuard (GI3) across both initial and cached-context re-evaluations."""
    from guard_hierarchy import GUARD_REGISTRY
    from intelligence.internal.guards.guard_decision import BrakeAction
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard, is_llm_guard_exempt
    from order_guard import validate_guard_mode_cfg

    skip_llm = GUARD_REGISTRY["GI3"].is_bypassed(bypassed) or is_llm_guard_exempt()
    llm_mode = validate_guard_mode_cfg(margins["llm_guard_mode"], "llm_guard_mode")
    if skip_llm or llm_mode == "off":
        return True, "ok", 1.0

    min_notional = float(margins["llm_min_notional_eur"])
    if computed_notional is None or computed_notional < min_notional:
        return True, "ok", 1.0

    decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None) if valid_context else None
    entry_for_key = decisions_map.get(cache_key, {}) if isinstance(decisions_map, dict) else {}
    if "gemini_notional" in entry_for_key:
        prev_notional = entry_for_key.get("gemini_notional")
    elif valid_context and getattr(regime_context, "_gemini_decision_key", cache_key) == cache_key:
        prev_notional = getattr(regime_context, "_gemini_evaluated_notional", None)
    else:
        prev_notional = None
    if prev_notional is None and fallback_notional is not None:
        prev_notional = entry_for_key.get("notional", fallback_notional)

    notional_jump = (
        prev_notional is not None
        and (computed_notional - prev_notional) / max(prev_notional, 1.0) > 0.20
    )
    needs_eval = prev_notional is None or prev_notional < min_notional or notional_jump
    if needs_eval:
        if dur_sec is None and valid_context:
            dur_sec = (
                getattr(regime_context, "trend_duration_seconds", None)
                or getattr(regime_context, "resolved_duration_sec", None)
            )
        timeout_sec = float(margins["llm_timeout_sec"])
        fallback = str(margins["llm_fallback"]).strip().lower()
        llm_guard = HighStakeGuard(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
        try:
            g_dec = llm_guard.check(
                symbol,
                side,
                price,
                qty if qty is not None else 1.0,
                notional_eur=computed_notional,
                regime_context=regime_context if valid_context else None,
                dur_sec=dur_sec,
                is_stop_loss=is_stop_loss,
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
                g_tuple = (g_dec.allowed, g_dec.reason, g_dec.suggested_scale)
                if not isinstance(decisions_map, dict):
                    decisions_map = {}
                entry = decisions_map.get(cache_key, {})
                entry["gemini_notional"] = computed_notional
                entry["gemini_decision"] = g_tuple
                decisions_map[cache_key] = entry
                object.__setattr__(regime_context, "_intelligence_decisions_by_key", decisions_map)
                object.__setattr__(regime_context, "_gemini_decision_key", cache_key)
                object.__setattr__(regime_context, "_gemini_evaluated_notional", computed_notional)
                object.__setattr__(regime_context, "_gemini_decision", g_tuple)
            except Exception:
                pass

        if notify_on_eval or not g_dec.allowed or g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
            prefix = "[LLM_GUARD_SHADOW]" if llm_mode == "shadow" else "[LLM_GUARD_ENFORCE]"
            print(
                f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} "
                f"(brake={g_dec.brake_action}, suggested_scale={g_dec.suggested_scale})"
            )

        if notify_on_eval:
            now_ts = float(now) if now is not None else time.time()
            cd_key = (symbol, side)
            if now_ts - _LLM_SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                _LLM_SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                try:
                    from notify_engine.alertnotifiers import notify

                    scale_pct = int(g_dec.suggested_scale * 100)
                    if not g_dec.allowed:
                        g_title = f"🛡 [P3C · HIGH-STAKE] BLOCK {side} {symbol}"
                    elif g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                        g_title = f"🛡 [P3C · HIGH-STAKE] SCALE {scale_pct}% {side} {symbol}"
                    else:
                        g_title = f"🧭 [P3C · HIGH-STAKE] ACCEPT {side} {symbol}"

                    meta_reason = (getattr(g_dec, "metadata", None) or {}).get("gemini_reason")
                    clean_reason = str(meta_reason or re.sub(
                        r"^(?:Gemini|LLM) (?:vetoed|approved|downscaled) high-stake (?:BUY|SELL):\s*",
                        "",
                        g_dec.reason,
                    )).strip()

                    price_str = f"${price:,.2f}" if price else ""
                    qty_str = f"{qty:g} " if qty else ""
                    order_header = f"Order: {side} {qty_str}{symbol}"
                    if price_str:
                        order_header += f" @ {price_str}"
                    order_header += f" · €{computed_notional:,.0f} (Scale: {scale_pct}%)"
                    notify(
                        title=g_title,
                        body=f"{order_header}\n{clean_reason}",
                        source="order_guard",
                        symbol=symbol,
                    )
                except Exception:
                    pass

        if llm_mode == "enforce":
            if not g_dec.allowed:
                return False, g_dec.reason, 0.0
            if g_dec.suggested_scale < 1.0:
                return True, g_dec.reason, g_dec.suggested_scale
        return True, "ok", 1.0

    if "gemini_decision" in entry_for_key:
        cached_g_dec = entry_for_key.get("gemini_decision")
    elif valid_context and getattr(regime_context, "_gemini_decision_key", cache_key) == cache_key:
        cached_g_dec = getattr(regime_context, "_gemini_decision", None)
    else:
        cached_g_dec = None
    if cached_g_dec is not None and llm_mode == "enforce":
        g_allowed, g_reason, g_scale = cached_g_dec
        if not g_allowed:
            return False, g_reason, 0.0
        if g_scale < 1.0:
            return True, g_reason, g_scale
    return True, "ok", 1.0


def check_intelligence_order_guards(
    provider: Optional[str],
    symbol: str,
    order_type: str,
    price: float,
    *,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    is_stop_loss: bool = False,
    regime_context: Any = None,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
    bypass_guards: Optional[set[str]] = None,
) -> Tuple[bool, str, float]:
    """Evaluate external microstructure, geopolitical shock, and high-stake LLM guards (Pillars 2, 3, 4).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    from guard_hierarchy import normalize_bypassed_guards, GUARD_REGISTRY
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if is_stop_loss or "all" in bypassed or "*" in bypassed:
        return True, "stop_loss_exempt", 1.0

    side = (order_type or "").upper()
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    from intelligence.internal.guards.guard_decision import BrakeAction

    effective_scale = 1.0
    active_reason = "ok"

    # Compute notional EUR
    computed_notional = float(notional_eur) if notional_eur is not None else (float(price * qty) if (price and qty) else 0.0)

    if side == "SELL":
        sell_min_notional = float(m["sell_guard_min_notional_eur"])
        if computed_notional < sell_min_notional:
            return True, "small_sell_exempt", 1.0

    # 1. High-Stake LLM Pre-Trade Guard (Pillar 3 / GI3)
    valid_context = False
    if regime_context is not None and hasattr(regime_context, "__dict__"):
        if hasattr(regime_context, "is_valid_for"):
            max_age = float(m.get("regime_context_max_age_sec", 300.0))
            has_id = bool(getattr(regime_context, "symbol", None) and getattr(regime_context, "provider", None))
            valid_context = regime_context.is_valid_for(
                symbol=symbol,
                provider=provider,
                max_age_seconds=max_age,
                now=now,
                require_identity=has_id,
            )
        else:
            valid_context = True
    hs_ok, hs_reason, hs_scale = evaluate_high_stake_order_guard(
        symbol=symbol,
        side=side,
        price=price,
        qty=qty,
        computed_notional=computed_notional,
        is_stop_loss=is_stop_loss,
        regime_context=regime_context,
        valid_context=valid_context,
        cache_key=(side, frozenset(bypassed)),
        margins=m,
        bypassed=bypassed,
        now=now,
        notify_on_eval=True,
    )
    if not hs_ok:
        return False, hs_reason, 0.0
    if hs_scale < 1.0:
        effective_scale = min(effective_scale, hs_scale)
        active_reason = hs_reason

    # 2. Geopolitical & Energy Shock Guard (Pillar 4 / GI4)
    skip_geo = GUARD_REGISTRY["GI4"].is_bypassed(bypassed)
    if not skip_geo:
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
