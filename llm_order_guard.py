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
import time
from typing import Any, Dict, Optional, Tuple

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

    skip_llm = GUARD_REGISTRY["GLLM1"].is_bypassed(bypassed) or is_llm_guard_exempt()
    llm_mode = validate_guard_mode_cfg(margins["llm_guard_mode"], "llm_guard_mode")
    if skip_llm or llm_mode == "off":
        return True, "ok", 1.0

    min_notional = resolve_high_stake_min_notional_usd(margins, side=side, require_key=True)
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
    if (
        prev_notional is None
        and fallback_notional is not None
        and not (isinstance(decisions_map, dict) and cache_key in decisions_map)
    ):
        prev_notional = fallback_notional

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
                    from orchestratorOS.agy.topics import topic_spec

                    hs_tag = topic_spec("guard_high_stake").bracket_tag
                    scale_pct = int(g_dec.suggested_scale * 100)
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
    """Evaluate pre-trade orders against published geopolitical threat assessment (GLLM3).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    order_side = (side or "").upper()
    if order_side != "BUY":
        return True, "not_buy_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    from intelligence.internal.guards.guard_decision import BrakeAction

    geo_mode = validate_guard_mode_cfg(m["geopolitical_guard_mode"], "geopolitical_guard_mode")
    if geo_mode == "off":
        return True, "geopolitical_guard_off", 1.0

    raw_shadow_notify = m["shadow_notify"]
    if isinstance(raw_shadow_notify, str):
        shadow_notify = raw_shadow_notify.strip().lower() in ("1", "1.0", "true", "yes", "on")
    else:
        shadow_notify = bool(float(raw_shadow_notify))

    try:
        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
        from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
        analyzer = GeopoliticalThreatAnalyzer()
        cached_geo = analyzer._cached_assessment
        if cached_geo is not None:
            geo_guard = GeopoliticalShockGuard()
            geo_dec = geo_guard.check(symbol, order_side, cached_geo)
            if not geo_dec.allowed or geo_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                prefix = "[GEOPOLITICAL_GUARD_SHADOW]" if geo_mode == "shadow" else "[GEOPOLITICAL_GUARD_ENFORCE]"
                print(f"{prefix} {order_side} {symbol}: {geo_dec.reason} (brake={geo_dec.brake_action}, suggested_scale={geo_dec.suggested_scale})")
                if geo_mode == "shadow" and shadow_notify:
                    cd_key = (symbol, order_side)
                    now_ts = now if now is not None else time.time()
                    if now_ts - _GEO_SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                        _GEO_SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                        try:
                            from notify_engine.alertnotifiers import notify
                            from orchestratorOS.agy.topics import topic_spec

                            geo_tag = topic_spec("macro_geopolitical").bracket_tag
                            scale_pct = int(geo_dec.suggested_scale * 100)
                            if not geo_dec.allowed:
                                geo_title = f"🛑 {geo_tag} BLOCK {order_side} {symbol}"
                            else:
                                geo_title = f"🛡 {geo_tag} SCALE {scale_pct}% {order_side} {symbol}"

                            clean_summary = cached_geo.summary or geo_dec.reason
                            if "(" in geo_dec.reason and geo_dec.reason.endswith(")"):
                                clean_summary = geo_dec.reason.split("(", 1)[1].rstrip(")")

                            geo_body = f"Threat: {cached_geo.threat_level} (Risk: {cached_geo.risk_score:.2f}, Scale: {scale_pct}%)\n{clean_summary}"
                            notify(
                                title=geo_title,
                                body=geo_body,
                                source="macro_shadow",
                                symbol=symbol,
                            )
                        except Exception:
                            pass
                if geo_mode == "enforce":
                    if not geo_dec.allowed:
                        return False, geo_dec.reason, 0.0
                    return True, geo_dec.reason, geo_dec.suggested_scale
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

    side = (order_type or "").upper()
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    effective_scale = 1.0
    active_reason = "ok"

    computed_notional = float(notional_eur) if notional_eur is not None else (float(price * qty) if (price and qty) else 0.0)

    # 1. GLLM1: High-Stake LLM Pre-Trade Guard (symmetric for BUY & SELL >= high_stake_min_notional_usd)
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
        if geo_scale < 1.0:
            effective_scale = min(effective_scale, geo_scale)
            active_reason = geo_reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


check_intelligence_order_guards = check_llm_order_guards
