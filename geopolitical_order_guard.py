"""Macro Geopolitical and Energy Crisis Order Guard (Pillar 4).

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

    if margins is None:
        try:
            from order_guard import _load_margins
            m = _load_margins()
        except Exception:
            m = {}
    else:
        m = margins

    from intelligence.internal.guards.guard_decision import BrakeAction

    geo_mode = str(m.get("geopolitical_guard_mode", "shadow")).strip().lower()
    if geo_mode in ("off", "0", "disabled"):
        return True, "geopolitical_guard_off", 1.0

    shadow_notify = bool(int(float(m.get("shadow_notify", 1.0))))

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
                            if not geo_dec.allowed:
                                geo_title = f"🛡 [VETO] Would Block {order_side} {symbol}"
                            else:
                                geo_title = f"🛡 [DOWNSCALE] Would Scale {int(geo_dec.suggested_scale*100)}% {order_side} {symbol}"
                            notify(
                                title=geo_title,
                                body=f"Macro shock flagged: {geo_dec.reason} (threat={cached_geo.threat_level}, risk={cached_geo.risk_score:.2f})",
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
