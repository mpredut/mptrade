"""External Market Intelligence Order Guards (GI_OBOOK, GI_WHALE, GI_FUNDING).

Evaluates pre-trade orders against external market microstructure math (NO LLM, 0 ms):
- GI_OBOOK (OrderbookWallGuard): Spot orderbook depth, bid/ask imbalance ratio, and large ask/bid walls.
- GI_WHALE (WhaleDivergenceGuard): Top traders Long/Short ratio, taker volume flow, and OI divergence.
- GI_FUNDING / GI_CROWD (FundingCrowdingGuard): Perpetual derivatives funding rate crowding & squeeze risk.

Executes in 0 ms (reads local cachedb/ telemetry, zero network calls during order placement).
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("external_order_guard")

_whale_collector = None
_orderbook_collector = None
_derivatives_collector = None
_MICRO_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str, str], float] = {}


def get_whale_collector():
    """Lazily load singleton whale positioning collector."""
    global _whale_collector
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
    og = sys.modules.get("order_guard")
    if og and getattr(og, "_derivatives_collector", None) is not None:
        return og._derivatives_collector
    if _derivatives_collector is None:
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
        _derivatives_collector = DerivativesTelemetryCollector()
    return _derivatives_collector


def check_external_order_guards(
    symbol: str,
    side: str,
    price: float,
    *,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
    bypass_guards: Optional[set[str]] = None,
) -> Tuple[bool, str, float]:
    """Evaluate external orderbook depth (GI_OBOOK), whale positioning (GI_WHALE), and funding (GI_FUNDING).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards) if bypass_guards else set()
    if "all" in bypassed or "*" in bypassed:
        return True, "stop_loss_exempt", 1.0

    order_side = (side or "").upper()
    if order_side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    if "intelligence_guards_mode" in m:
        master_mode = validate_guard_mode_cfg(m["intelligence_guards_mode"], "intelligence_guards_mode")
        if master_mode == "off":
            return True, "intelligence_guards_off", 1.0

    ext_mode = "enforce"
    if "external_guard_mode" in m:
        ext_mode = validate_guard_mode_cfg(m["external_guard_mode"], "external_guard_mode")
        if ext_mode == "off":
            return True, "external_guards_off", 1.0

    computed_notional = float(notional_eur) if notional_eur is not None else (float(price * qty) if (price and qty) else 0.0)

    from intelligence.internal.guards.guard_decision import BrakeAction

    effective_scale = 1.0
    active_reason = "ok"

    now_ts = float(now) if now is not None else time.time()
    shadow_notify_enabled = str(m["shadow_notify"]).strip().lower() in ("1", "1.0", "true", "yes", "on")

    def _send_micro_shadow_notify(
        guard_key: str,
        title: str,
        reason: str,
        scale_pct: Optional[int] = None,
    ) -> None:
        cd_key = (symbol, order_side, guard_key)
        if 0.0 <= (now_ts - _MICRO_SHADOW_NOTIFY_COOLDOWN.get(cd_key, -1e18)) < 1800.0:
            return
        _MICRO_SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
        try:
            from notify_engine.alertnotifiers import notify
            price_str = f" @ ${price:,.2f}" if price else ""
            scale_str = f" (Scale: {scale_pct}%)" if scale_pct is not None else ""
            order_header = f"Order: {order_side} {symbol}{price_str}{scale_str}"
            body = f"{order_header}\n{reason}"
            notify(
                title=title,
                body=body,
                source="order_guard",
                symbol=symbol,
            )
        except Exception:
            pass

    # 1. GI_WHALE: Whale Flow & Open Interest Divergence Guard
    whale_mode = validate_guard_mode_cfg(m["whale_guard_mode"], "whale_guard_mode")
    if ext_mode == "shadow" and whale_mode == "enforce":
        whale_mode = "shadow"
    if whale_mode != "off" and not GUARD_REGISTRY["GI_WHALE"].is_bypassed(bypassed):
        try:
            from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
            w_col = get_whale_collector()
            try:
                w_snap = w_col.fetch(symbol, allow_network=False, now=now_ts)
            except TypeError:
                w_snap = w_col.fetch(symbol, allow_network=False)
            sev_oi = float(m["severe_cascade_oi_pct"])
            min_tk = float(m["min_taker_ratio"])
            downscale_tk = float(m["taker_downscale_threshold"])
            downscaled_scale = float(m["taker_downscaled_scale"])
            w_guard = WhaleDivergenceGuard(
                severe_cascade_oi_pct=sev_oi,
                min_taker_ratio=min_tk,
                taker_downscale_threshold=downscale_tk,
                taker_downscaled_scale=downscaled_scale,
            )
            w_dec = w_guard.check(symbol, order_side, snapshot=w_snap)
            if not w_dec.allowed or w_dec.brake_action in (BrakeAction.DEFER_WAIT, BrakeAction.DOWNSCALE_QTY):
                prefix = "[WHALE_GUARD_SHADOW]" if whale_mode == "shadow" else "[WHALE_GUARD_ENFORCE]"
                print(f"{prefix} {order_side} {symbol}: {w_dec.reason} (brake={w_dec.brake_action})")
                if whale_mode == "shadow" and shadow_notify_enabled:
                    scale_pct = int(w_dec.suggested_scale * 100) if w_dec.suggested_scale < 1.0 else None
                    _send_micro_shadow_notify("whale", f"🛡 [GI_WHALE] {w_dec.brake_action.value} {order_side} {symbol}", w_dec.reason, scale_pct=scale_pct)
                if whale_mode == "enforce":
                    if not w_dec.allowed:
                        return False, w_dec.reason, 0.0
                    if w_dec.suggested_scale < 1.0:
                        effective_scale = min(effective_scale, w_dec.suggested_scale)
                        active_reason = w_dec.reason
        except Exception:
            pass

    # 2. GI_OBOOK: Orderbook Depth & Whale Limit Wall Guard
    ob_mode = validate_guard_mode_cfg(m["orderbook_wall_guard_mode"], "orderbook_wall_guard_mode")
    if ext_mode == "shadow" and ob_mode == "enforce":
        ob_mode = "shadow"
    if ob_mode != "off" and not GUARD_REGISTRY["GI_OBOOK"].is_bypassed(bypassed):
        try:
            from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
            ob_col = get_orderbook_collector()
            try:
                ob_snap = ob_col.fetch(symbol, allow_network=False, now=now_ts)
            except TypeError:
                ob_snap = ob_col.fetch(symbol, allow_network=False)
            min_imb = float(m["min_buy_imbalance"])
            max_imb = float(m["max_sell_imbalance"])
            wall_limit = float(m["whale_wall_usd_limit"])
            ob_guard = OrderbookWallGuard(
                min_buy_imbalance=min_imb,
                max_sell_imbalance=max_imb,
                whale_wall_usd_limit=wall_limit,
            )
            ob_dec = ob_guard.check(symbol, order_side, snapshot=ob_snap)
            if not ob_dec.allowed or ob_dec.brake_action in (BrakeAction.DEFER_WAIT, BrakeAction.DOWNSCALE_QTY):
                prefix = "[ORDERBOOK_GUARD_SHADOW]" if ob_mode == "shadow" else "[ORDERBOOK_GUARD_ENFORCE]"
                print(f"{prefix} {order_side} {symbol}: {ob_dec.reason} (brake={ob_dec.brake_action})")
                if ob_mode == "shadow" and shadow_notify_enabled:
                    scale_pct = int(ob_dec.suggested_scale * 100) if ob_dec.suggested_scale < 1.0 else None
                    action_str = f"SCALE {scale_pct}%" if (ob_dec.allowed and scale_pct is not None) else "DEFER"
                    _send_micro_shadow_notify("orderbook", f"🛡 [GI_OBOOK] {action_str} {order_side} {symbol}", ob_dec.reason, scale_pct=scale_pct)
                if ob_mode == "enforce":
                    if not ob_dec.allowed:
                        return False, ob_dec.reason, 0.0
                    if ob_dec.suggested_scale < 1.0 and ob_dec.suggested_scale <= effective_scale:
                        effective_scale = ob_dec.suggested_scale
                        active_reason = ob_dec.reason
        except Exception:
            pass

    # 3. GI_FUNDING (GI_CROWD): Perpetual Derivatives Funding Rate Crowding Guard
    funding_mode = validate_guard_mode_cfg(m["funding_guard_mode"], "funding_guard_mode")
    if ext_mode == "shadow" and funding_mode == "enforce":
        funding_mode = "shadow"
    if funding_mode != "off" and not GUARD_REGISTRY["GI_FUNDING"].is_bypassed(bypassed):
        try:
            from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
            f_col = get_derivatives_collector()
            try:
                f_snap = f_col.fetch(symbol, allow_network=False, now=now_ts)
            except TypeError:
                f_snap = f_col.fetch(symbol, allow_network=False)
            f_max = float(m["funding_max_long_rate"])
            f_min = float(m["funding_min_short_rate"])
            f_policy = str(m["funding_crowding_policy"]).strip().lower()
            f_scale = float(m["funding_crowded_scale"])
            f_guard = FundingCrowdingGuard(
                max_long_funding_rate=f_max,
                min_short_funding_rate=f_min,
                crowding_policy=f_policy,
                crowded_scale=f_scale,
            )
            f_dec = f_guard.check(symbol, order_side, telemetry=f_snap)
            if not f_dec.allowed or f_dec.brake_action in (BrakeAction.DEFER_WAIT, BrakeAction.DOWNSCALE_QTY):
                prefix = "[FUNDING_GUARD_SHADOW]" if funding_mode == "shadow" else "[FUNDING_GUARD_ENFORCE]"
                print(f"{prefix} {order_side} {symbol}: {f_dec.reason} (brake={f_dec.brake_action}, suggested_scale={f_dec.suggested_scale})")
                if funding_mode == "shadow" and shadow_notify_enabled:
                    if not f_dec.allowed:
                        _send_micro_shadow_notify("funding", f"🛡 [GI_FUNDING] DEFER {order_side} {symbol}", f_dec.reason)
                    else:
                        scale_pct = int(f_dec.suggested_scale * 100)
                        _send_micro_shadow_notify("funding", f"🛡 [GI_FUNDING] SCALE {scale_pct}% {order_side} {symbol}", f_dec.reason, scale_pct=scale_pct)
                if funding_mode == "enforce":
                    if not f_dec.allowed:
                        return False, f_dec.reason, 0.0
                    if f_dec.suggested_scale < 1.0 and f_dec.suggested_scale <= effective_scale:
                        effective_scale = f_dec.suggested_scale
                        active_reason = f_dec.reason
        except Exception:
            pass

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
