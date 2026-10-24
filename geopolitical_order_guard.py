"""Macro Geopolitical and Energy Crisis Order Guard (Pillar 3 · Intelligence Layer).

Evaluates pre-trade orders against geopolitical threat and systemic macro shock state:
- GeopoliticalShockGuard: Evaluates acute conflict, war escalation, and energy supply crisis threats.
- CRITICAL_SHOCK: Complete veto on new BUY entries (prevents catching liquidation knives).
- ELEVATED: Downscales new BUY entries to 50% exposure.
- NORMAL: Instant 0 ms green light.
- SELL / Stop-Loss orders are never restricted.

Reads published assessment from cachedb/geopolitical_threat_eval.json in sub-millisecond local time.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("geopolitical_order_guard")

_GEO_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str], float] = {}


def check_geopolitical_order_guards(
    symbol: str,
    side: str,
    *,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Evaluate pre-trade orders against published geopolitical threat assessment (Pillar 4).

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
