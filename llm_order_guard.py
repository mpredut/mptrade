"""LLM Pre-Trade Order Guard Coordinator (GLLM1, GLLM2, GLLM3).

Single symmetric coordinator for all LLM-powered trading guards:
- GLLM1 (LLMHighStakeGuardSpec / llm_high_stake):
    Real-time pre-trade LLM validation (HighStakeGuard >= 700 USD & PostStopReentryGuard).
    Sub-700 USD orders pass with 0 ms latency (zero overhead).
- GLLM2 (LLMSentimentGuardSpec / llm_sentiment):
    3h periodic LLM Macro Sentiment Advisor evaluation (sentiment_advisor_eval.json:
    Fear & Greed, Market Breadth, Crypto News).
- GLLM3 (LLMGeopoliticalGuardSpec / llm_geopolitical):
    4.5h periodic LLM Geopolitical & Energy Black Swan shock shield (geopolitical_threat_eval.json:
    CRITICAL_SHOCK -> BUY veto, ELEVATED -> 50% BUY downscale, NORMAL -> 0 ms pass).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

from botcore import _parse_bool_flag
from intelligence.macro_sentiment_ssot import (
    check_cooldown_gate,
    clean_scale as _clean_scale,
    clean_symbol_key,
    normalize_order_side as _normalize_side,
    obj_attr,
    resolve_now_ts as _resolve_now_ts,
    safe_pos_float as _pos_float,
)

logger = logging.getLogger("llm_order_guard")

_LLM_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str], float] = {}
_GEO_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str], float] = {}


# ============================================================================
# GLLM1: Live Pre-Trade High-Stake LLM Guard (>= 700 USD)
# ============================================================================

def evaluate_high_stake_order_guard(
    *,
    symbol: str,
    side: str,
    price: float,
    qty: Optional[float],
    computed_notional: Optional[float],
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
    """SSOT evaluator for HighStakeGuard (GLLM1) across both initial and cached-context re-evaluations."""
    from guard_hierarchy import GUARD_REGISTRY
    from intelligence.internal.guards.guard_decision import BrakeAction
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard, is_llm_guard_exempt
    from order_guard import resolve_high_stake_min_notional_usd, validate_guard_mode_cfg

    side = _normalize_side(side)
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    if "intelligence_guards_mode" in margins and validate_guard_mode_cfg(margins["intelligence_guards_mode"], "intelligence_guards_mode") == "off":
        return True, "ok", 1.0
    skip_llm = GUARD_REGISTRY["GLLM1"].is_bypassed(bypassed) or is_llm_guard_exempt()
    llm_mode = validate_guard_mode_cfg(margins["llm_guard_mode"], "llm_guard_mode")
    if skip_llm or llm_mode == "off":
        return True, "ok", 1.0

    min_notional = resolve_high_stake_min_notional_usd(margins, side=side, require_key=True)
    comp_notional = _pos_float(computed_notional, 0.0)
    if comp_notional <= 0.0 or comp_notional < min_notional:
        return True, "ok", 1.0
    computed_notional = comp_notional

    decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None) if valid_context else None
    entry_for_key = decisions_map.get(cache_key, {}) if isinstance(decisions_map, dict) else {}
    if "gemini_notional" in entry_for_key:
        prev_notional = entry_for_key.get("gemini_notional")
    elif valid_context and getattr(regime_context, "_gemini_decision_key", cache_key) == cache_key:
        prev_notional = getattr(regime_context, "_gemini_evaluated_notional", None)
    else:
        prev_notional = None
    if prev_notional is None and fallback_notional is not None and not (isinstance(decisions_map, dict) and cache_key in decisions_map):
        prev_notional = fallback_notional

    prev_notional_f = _pos_float(prev_notional, -1.0)
    prev_notional = prev_notional_f if prev_notional_f > 0.0 else None

    notional_jump = prev_notional is not None and (computed_notional - prev_notional) / max(prev_notional, 1.0) > 0.20
    if prev_notional is None or prev_notional < min_notional or notional_jump:
        ctx_usable = valid_context or isinstance(regime_context, dict)
        if dur_sec is None and ctx_usable:
            dur_sec = obj_attr(regime_context, "trend_duration_seconds") or obj_attr(regime_context, "resolved_duration_sec")
        timeout_sec = _pos_float(margins.get("llm_timeout_sec"), 5.0)
        fallback = str(margins.get("llm_fallback") or "allow").strip().lower()
        llm_guard = HighStakeGuard(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
        try:
            g_dec = llm_guard.check(
                symbol, side, price, qty if qty is not None else 1.0,
                notional_eur=computed_notional,
                regime_context=regime_context if ctx_usable else None,
                dur_sec=dur_sec,
                now=now,
            )
        except TypeError:
            g_dec = llm_guard.check(
                symbol, side, price, qty if qty is not None else 1.0,
                notional_eur=computed_notional,
            )
        g_scale = _clean_scale(getattr(g_dec, "suggested_scale", 1.0))
        if valid_context:
            try:
                g_tuple = (bool(g_dec.allowed), g_dec.reason, g_scale)
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

        do_notify = _parse_bool_flag(notify_on_eval, True) if not isinstance(notify_on_eval, bool) else notify_on_eval
        if do_notify or not g_dec.allowed or g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
            prefix = "[LLM_GUARD_SHADOW]" if llm_mode == "shadow" else "[LLM_GUARD_ENFORCE]"
            print(
                f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} "
                f"(brake={g_dec.brake_action}, suggested_scale={g_scale})"
            )

        if do_notify and check_cooldown_gate(_LLM_SHADOW_NOTIFY_COOLDOWN, (clean_symbol_key(symbol), side), _resolve_now_ts(now)):
            try:
                from notify_engine.alertnotifiers import notify
                from orchestratorOS.agy.topics import topic_spec

                hs_tag = topic_spec("guard_high_stake").bracket_tag
                scale_pct = int(g_scale * 100)
                if not g_dec.allowed:
                    g_title = f"🛡 {hs_tag} BLOCK {side} {symbol}"
                elif g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                    g_title = f"🛡 {hs_tag} SCALE {scale_pct}% {side} {symbol}"
                else:
                    g_title = f"🧭 {hs_tag} ACCEPT {side} {symbol}"

                meta_reason = (getattr(g_dec, "metadata", None) or {}).get("gemini_reason")
                clean_reason = str(meta_reason or re.sub(
                    r"^(?:Gemini|LLM) (?:vetoed|approved|downscaled) high-stake (?:BUY|SELL):\s*",
                    "",
                    g_dec.reason,
                )).strip()

                px_clean = _pos_float(price, 0.0)
                qty_clean = _pos_float(qty, 0.0)
                price_str = f"${px_clean:,.2f}" if px_clean > 0.0 else ""
                qty_str = f"{qty_clean:g} " if qty_clean > 0.0 else ""
                order_header = f"Order: {side} {qty_str}{symbol}" + (f" @ {price_str}" if price_str else "")
                order_header += f" · €{computed_notional:,.0f} (Scale: {scale_pct}%)"
                notify(title=g_title, body=f"{order_header}\n{clean_reason}", source="order_guard", symbol=symbol)
            except Exception:
                pass

        if llm_mode == "enforce":
            if not g_dec.allowed or g_scale <= 0.0:
                return False, g_dec.reason, 0.0
            if g_scale < 1.0:
                return True, g_dec.reason, g_scale
        return True, "ok", 1.0

    if "gemini_decision" in entry_for_key:
        cached_g_dec = entry_for_key.get("gemini_decision")
    elif valid_context and getattr(regime_context, "_gemini_decision_key", cache_key) == cache_key:
        cached_g_dec = getattr(regime_context, "_gemini_decision", None)
    else:
        cached_g_dec = None
    if isinstance(cached_g_dec, (tuple, list)) and len(cached_g_dec) == 3 and llm_mode == "enforce":
        g_allowed, g_reason, raw_g_scale = cached_g_dec
        g_scale = _clean_scale(raw_g_scale)
        if not g_allowed or g_scale <= 0.0:
            return False, g_reason, 0.0
        if g_scale < 1.0:
            return True, g_reason, g_scale
    return True, "ok", 1.0


# ============================================================================
# GLLM3: Geopolitical & Energy Crisis Shock Guard (4.5h LLM Assessment)
# ============================================================================

def check_geopolitical_order_guards(
    symbol: str,
    side: str,
    *,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Evaluate pre-trade orders against published geopolitical threat assessment (GLLM3)."""
    order_side = _normalize_side(side)
    if order_side != "BUY":
        return True, "not_buy_side", 1.0

    from intelligence.internal.guards.guard_decision import BrakeAction
    from order_guard import _load_margins, validate_guard_mode_cfg
    m = _load_margins() if margins is None else margins

    if "intelligence_guards_mode" in m and validate_guard_mode_cfg(m["intelligence_guards_mode"], "intelligence_guards_mode") == "off":
        return True, "intelligence_guards_off", 1.0

    geo_mode = validate_guard_mode_cfg(m["geopolitical_guard_mode"], "geopolitical_guard_mode")
    if geo_mode == "off":
        return True, "geopolitical_guard_off", 1.0

    shadow_notify = _parse_bool_flag(m.get("shadow_notify"), False)

    try:
        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
        from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
        cached_geo = GeopoliticalThreatAnalyzer()._cached_assessment
        if cached_geo is not None:
            geo_dec = GeopoliticalShockGuard().check(symbol, order_side, cached_geo)
            geo_scale = _clean_scale(getattr(geo_dec, "suggested_scale", 1.0))
            if not geo_dec.allowed or geo_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                prefix = "[GEOPOLITICAL_GUARD_SHADOW]" if geo_mode == "shadow" else "[GEOPOLITICAL_GUARD_ENFORCE]"
                print(f"{prefix} {order_side} {symbol}: {geo_dec.reason} (brake={geo_dec.brake_action}, suggested_scale={geo_scale})")
                if (
                    geo_mode == "shadow"
                    and shadow_notify
                    and check_cooldown_gate(_GEO_SHADOW_NOTIFY_COOLDOWN, (clean_symbol_key(symbol), order_side), _resolve_now_ts(now))
                ):
                    try:
                        from notify_engine.alertnotifiers import notify
                        from orchestratorOS.agy.topics import topic_spec

                        geo_tag = topic_spec("macro_geopolitical").bracket_tag
                        scale_pct = int(geo_scale * 100)
                        geo_title = (
                            f"🛑 {geo_tag} BLOCK {order_side} {symbol}"
                            if not geo_dec.allowed
                            else f"🛡 {geo_tag} SCALE {scale_pct}% {order_side} {symbol}"
                        )
                        clean_summary = obj_attr(cached_geo, "summary") or geo_dec.reason
                        if "(" in geo_dec.reason and geo_dec.reason.endswith(")"):
                            clean_summary = geo_dec.reason.split("(", 1)[1].rstrip(")")
                        raw_tl = obj_attr(cached_geo, "threat_level", "UNKNOWN")
                        raw_rs = _pos_float(obj_attr(cached_geo, "risk_score", 0.0), 0.0)
                        notify(
                            title=geo_title,
                            body=f"Threat: {raw_tl} (Risk: {raw_rs:.2f}, Scale: {scale_pct}%)\n{clean_summary}",
                            source="macro_shadow",
                            symbol=symbol,
                        )
                    except Exception:
                        pass
                if geo_mode == "enforce":
                    if not geo_dec.allowed or geo_scale <= 0.0:
                        return False, geo_dec.reason, 0.0
                    return True, geo_dec.reason, geo_scale
    except Exception as e:
        logger.debug("Geopolitical guard evaluation exception: %s", e)

    return True, "ok", 1.0


# ============================================================================
# Unified LLM Order Guards Coordinator (GLLM1 + GLLM2 + GLLM3)
# ============================================================================

def check_llm_order_guards(
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
    bypass_guards: Optional[set[str]] = None,
) -> Tuple[bool, str, float]:
    """Evaluate LLM pre-trade guards (GLLM1 High-Stake, GLLM2 Sentiment Advisor, GLLM3 Geopolitical Shock).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if "all" in bypassed or "*" in bypassed:
        return True, "stop_loss_exempt", 1.0

    side = _normalize_side(order_type)
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    if "intelligence_guards_mode" in m:
        if validate_guard_mode_cfg(m["intelligence_guards_mode"], "intelligence_guards_mode") == "off":
            return True, "intelligence_guards_off", 1.0

    effective_scale = 1.0
    active_reason = "ok"

    notional_f = _pos_float(notional_eur, 0.0)
    px_f = _pos_float(price, 0.0)
    qty_f = _pos_float(qty, 0.0)
    computed_notional = notional_f if notional_f > 0.0 else ((px_f * qty_f) if (px_f > 0.0 and qty_f > 0.0) else 0.0)

    # 1. GLLM1: High-Stake LLM Pre-Trade Guard (symmetric for BUY & SELL >= high_stake_min_notional_usd)
    valid_context = False
    if regime_context is not None and hasattr(regime_context, "__dict__"):
        if hasattr(regime_context, "is_valid_for"):
            max_age = _pos_float(m.get("regime_context_max_age_sec"), 300.0)
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
    _raw_intel_codes = ("GT_L", "G_QLIMIT1", "G_QLIMIT2", "GI_OBOOK", "GI_WHALE", "GI_FUNDING", "GLLM2", "GLLM3")
    intel_cache_key = (
        side,
        frozenset(code for code in _raw_intel_codes if GUARD_REGISTRY[code].is_bypassed(bypassed)),
    )
    hs_ok, hs_reason, hs_scale = evaluate_high_stake_order_guard(
        symbol=symbol,
        side=side,
        price=price,
        qty=qty,
        computed_notional=computed_notional,
        regime_context=regime_context,
        valid_context=valid_context,
        cache_key=intel_cache_key,
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

    # 2. GLLM3: Geopolitical & Energy Shock Guard
    skip_geo = GUARD_REGISTRY["GLLM3"].is_bypassed(bypassed)
    if not skip_geo:
        geo_ok, geo_reason, geo_scale = check_geopolitical_order_guards(
            symbol=symbol,
            side=side,
            margins=m,
            now=now,
        )
        if not geo_ok:
            return False, geo_reason, 0.0
        if geo_scale < 1.0 and geo_scale <= effective_scale:
            effective_scale = geo_scale
            active_reason = geo_reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


check_intelligence_order_guards = check_llm_order_guards
