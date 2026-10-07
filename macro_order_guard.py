"""Macro, External Microstructure, and LLM Intelligence Order Guard.

Evaluates pre-trade orders against external market telemetry and macro shock state:
- Pillar 4: Geopolitical & Energy Shock Guard (reads cachedb/geopolitical_threat_state.json).
- Pillar 2: Orderbook Depth & Ask Wall Guard (reads cachedb/orderbook_depth_*.json).
- Pillar 2: Whale Flow & Open Interest Divergence Guard (reads cachedb/whale_snapshot_*.json).
- Pillar 2: Perpetual Derivatives Funding Rate Crowding Guard (reads cachedb/derivatives_telemetry_*.json).
- Pillar 3: Google Gemini High-Stake Pre-Flight Risk Vetting (event-driven for purchases >= 1000 EUR).

Strictly decoupled from pure mathematical order guards (Pillar 1), allowing trading bots
and order_guard.py to evaluate macro intelligence with sub-millisecond local reads.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("macro_order_guard")

_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str, str], float] = {}
_whale_collector = None
_orderbook_collector = None
_derivatives_collector = None


def get_whale_collector():
    """Lazily load singleton whale positioning collector."""
    global _whale_collector
    import sys
    og = sys.modules.get("order_guard")
    if og and getattr(og, "_whale_collector", None) is not None:
        return og._whale_collector
    if _whale_collector is None:
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
        _whale_collector = WhalePositioningCollector()
    return _whale_collector


def get_orderbook_collector():
    """Lazily load singleton orderbook depth collector."""
    global _orderbook_collector
    import sys
    og = sys.modules.get("order_guard")
    if og and getattr(og, "_orderbook_collector", None) is not None:
        return og._orderbook_collector
    if _orderbook_collector is None:
        from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
        _orderbook_collector = OrderbookDepthCollector()
    return _orderbook_collector


def get_derivatives_collector():
    """Lazily load singleton derivatives telemetry collector."""
    global _derivatives_collector
    import sys
    og = sys.modules.get("order_guard")
    if og and getattr(og, "_derivatives_collector", None) is not None:
        return og._derivatives_collector
    if _derivatives_collector is None:
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
        _derivatives_collector = DerivativesTelemetryCollector()
    return _derivatives_collector


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
    """Evaluate external microstructure, macro geopolitical shock, and high-stake LLM guards.

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

    # 1. Google Gemini High-Stake Guard (> 1000 EUR purchases)
    gemini_mode = str(m.get("gemini_guard_mode", "shadow")).strip().lower()
    if gemini_mode not in ("off", "0", "disabled"):
        min_notional = float(m.get("gemini_min_notional_eur", 1000.0))
        if computed_notional is not None and computed_notional >= min_notional:
            from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
            timeout_sec = float(m.get("gemini_timeout_sec", 12.0))
            fallback = str(m.get("gemini_fallback", "allow")).strip().lower()
            g_guard = GeminiHighStakeGuard(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
            g_dec = g_guard.check(symbol, side, price, qty if qty is not None else 1.0, notional_eur=computed_notional)
            if regime_context is not None:
                try:
                    object.__setattr__(regime_context, "_gemini_evaluated_notional", computed_notional)
                    object.__setattr__(regime_context, "_gemini_decision", (g_dec.allowed, g_dec.reason, g_dec.suggested_scale))
                except Exception:
                    pass
            if not g_dec.allowed or g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                prefix = "[GEMINI_GUARD_SHADOW]" if gemini_mode == "shadow" else "[GEMINI_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} (brake={g_dec.brake_action}, suggested_scale={g_dec.suggested_scale})")
                if gemini_mode == "shadow" and shadow_notify:
                    cd_key = ("gemini", symbol, side)
                    now_ts = now if now is not None else time.time()
                    if now_ts - _SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                        _SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                        try:
                            from notify_engine.alertnotifiers import notify
                            if not g_dec.allowed:
                                g_title = f"🛡 [GEMINI SHADOW VETO] Would Block {side} {symbol}"
                            else:
                                g_title = f"🛡 [GEMINI SHADOW DOWNSCALE] Would Scale {int(g_dec.suggested_scale*100)}% {side} {symbol}"
                            notify(
                                title=g_title,
                                body=f"High-stake order €{computed_notional:.2f} flagged: {g_dec.reason} (brake={g_dec.brake_action}, scale={g_dec.suggested_scale})",
                                source="order_guard",
                                symbol=symbol,
                            )
                        except Exception:
                            pass
                if gemini_mode == "enforce":
                    if not g_dec.allowed:
                        return False, g_dec.reason, 0.0
                    effective_scale = min(effective_scale, g_dec.suggested_scale)
                    active_reason = g_dec.reason

    # 2. Geopolitical & Energy Shock Guard (Pillar 4 - Black Swan Shield)
    geo_mode = str(m.get("geopolitical_guard_mode", "shadow")).strip().lower()
    if geo_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
            from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
            analyzer = GeopoliticalThreatAnalyzer()
            cached_geo = analyzer._cached_assessment
            if cached_geo is not None:
                geo_guard = GeopoliticalShockGuard()
                geo_dec = geo_guard.check(symbol, side, cached_geo)
                if not geo_dec.allowed or geo_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                    prefix = "[GEOPOLITICAL_GUARD_SHADOW]" if geo_mode == "shadow" else "[GEOPOLITICAL_GUARD_ENFORCE]"
                    print(f"{prefix} {side} {symbol}: {geo_dec.reason} (brake={geo_dec.brake_action}, suggested_scale={geo_dec.suggested_scale})")
                    if geo_mode == "shadow" and shadow_notify:
                        cd_key = ("geopolitical", symbol, side)
                        now_ts = now if now is not None else time.time()
                        if now_ts - _SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                            _SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                            try:
                                from notify_engine.alertnotifiers import notify
                                if not geo_dec.allowed:
                                    geo_title = f"🛡 [VETO] Would Block {side} {symbol}"
                                else:
                                    geo_title = f"🛡 [DOWNSCALE] Would Scale {int(geo_dec.suggested_scale*100)}% {side} {symbol}"
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
                        effective_scale = min(effective_scale, geo_dec.suggested_scale)
                        active_reason = geo_dec.reason
        except Exception:
            pass

    # 3. Pillar 2: Whale Flow & Open Interest Divergence Guard
    whale_mode = str(m.get("whale_guard_mode", "shadow")).strip().lower()
    if whale_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
            w_snap = get_whale_collector().fetch(symbol, allow_network=False)
            w_guard = WhaleDivergenceGuard()
            w_dec = w_guard.check(symbol, side, snapshot=w_snap)
            if not w_dec.allowed or w_dec.brake_action == BrakeAction.DEFER_WAIT:
                prefix = "[WHALE_GUARD_SHADOW]" if whale_mode == "shadow" else "[WHALE_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol}: {w_dec.reason} (brake={w_dec.brake_action})")
                if whale_mode == "enforce":
                    if not w_dec.allowed:
                        return False, w_dec.reason, 0.0
        except Exception:
            pass

    # 4. Pillar 2: Orderbook Depth & Whale Limit Wall Guard
    ob_mode = str(m.get("orderbook_wall_guard_mode", "shadow")).strip().lower()
    if ob_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
            ob_snap = get_orderbook_collector().fetch(symbol, allow_network=False)
            min_imb = float(m.get("min_buy_imbalance", 0.25))
            wall_limit = float(m.get("whale_wall_usd_limit", 1_000_000.0))
            ob_guard = OrderbookWallGuard(
                min_buy_imbalance=min_imb,
                whale_wall_usd_limit=wall_limit,
            )
            ob_dec = ob_guard.check(symbol, side, snapshot=ob_snap)
            if not ob_dec.allowed or ob_dec.brake_action == BrakeAction.DEFER_WAIT:
                prefix = "[ORDERBOOK_GUARD_SHADOW]" if ob_mode == "shadow" else "[ORDERBOOK_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol}: {ob_dec.reason} (brake={ob_dec.brake_action})")
                if ob_mode == "enforce":
                    if not ob_dec.allowed:
                        return False, ob_dec.reason, 0.0
        except Exception:
            pass

    # 5. Pillar 2: Perpetual Derivatives Funding Rate Crowding Guard
    funding_mode = str(m.get("funding_guard_mode", "shadow")).strip().lower()
    if funding_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
            f_snap = get_derivatives_collector().fetch(symbol, allow_network=False)
            f_max = float(m.get("funding_max_long_rate", 0.0005))
            f_policy = str(m.get("funding_crowding_policy", "downscale")).strip().lower()
            f_scale = float(m.get("funding_crowded_scale", 0.50))
            f_guard = FundingCrowdingGuard(
                max_long_funding_rate=f_max,
                crowding_policy=f_policy,
                crowded_scale=f_scale,
            )
            f_dec = f_guard.check(symbol, side, telemetry=f_snap)
            if not f_dec.allowed or f_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                prefix = "[FUNDING_GUARD_SHADOW]" if funding_mode == "shadow" else "[FUNDING_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol}: {f_dec.reason} (brake={f_dec.brake_action}, suggested_scale={f_dec.suggested_scale})")
                if funding_mode == "enforce":
                    if not f_dec.allowed:
                        return False, f_dec.reason, 0.0
                    effective_scale = min(effective_scale, f_dec.suggested_scale)
                    active_reason = f_dec.reason
        except Exception:
            pass

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
