"""External Order Guard (Pillar 2).

Evaluates pre-trade orders against external market microstructure math (NO LLM, 0 ms):
- OrderbookWallGuard: Spot orderbook depth, bid/ask imbalance ratio, and large ask walls.
- FundingCrowdingGuard: Perpetual derivatives funding rate crowding & long squeeze risk.
- WhaleDivergenceGuard: Top traders Long/Short ratio, taker volume flow, and OI divergence.

Executes in 0 ms (reads local cachedb/ telemetry, zero network calls during order placement).
Symmetrically mirrors internal_order_guard.py (Pillar 1).
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
    is_stop_loss: bool = False,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Evaluate external orderbook depth, derivatives funding, and whale positioning guards (Pillar 2).

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    order_side = (side or "").upper()
    if is_stop_loss:
        return True, "stop_loss_exempt", 1.0

    if order_side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    from order_guard import _load_margins, validate_guard_mode_cfg
    if margins is None:
        m = _load_margins()
    else:
        m = margins

    # For SELL orders: small routine orders (< sell_guard_min_notional_eur) bypass with 0 ms overhead
    computed_notional = float(notional_eur) if notional_eur is not None else (float(price * qty) if (price and qty) else 0.0)
    if order_side == "SELL":
        sell_min_notional = float(m["sell_guard_min_notional_eur"])
        if computed_notional < sell_min_notional:
            return True, "small_sell_exempt", 1.0

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
        if now_ts - _MICRO_SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) < 1800.0:
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

    # 1. Whale Flow & Open Interest Divergence Guard
    whale_mode = validate_guard_mode_cfg(m["whale_guard_mode"], "whale_guard_mode")
    if whale_mode != "off":
        try:
            from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
            w_snap = get_whale_collector().fetch(symbol, allow_network=False)
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
                    _send_micro_shadow_notify("whale", f"🛡 [P2 · WHALE-FLOW] {w_dec.brake_action.value} {order_side} {symbol}", w_dec.reason, scale_pct=scale_pct)
                if whale_mode == "enforce":
                    if not w_dec.allowed:
                        return False, w_dec.reason, 0.0
                    if w_dec.suggested_scale < 1.0:
                        effective_scale = min(effective_scale, w_dec.suggested_scale)
                        active_reason = w_dec.reason
        except Exception:
            pass

    # 2. Orderbook Depth & Whale Limit Wall Guard
    ob_mode = validate_guard_mode_cfg(m["orderbook_wall_guard_mode"], "orderbook_wall_guard_mode")
    if ob_mode != "off":
        try:
            from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
            ob_snap = get_orderbook_collector().fetch(symbol, allow_network=False)
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
                    _send_micro_shadow_notify("orderbook", f"🛡 [P2 · ORDERBOOK] {action_str} {order_side} {symbol}", ob_dec.reason, scale_pct=scale_pct)
                if ob_mode == "enforce":
                    if not ob_dec.allowed:
                        return False, ob_dec.reason, 0.0
                    if ob_dec.suggested_scale < 1.0:
                        effective_scale = min(effective_scale, ob_dec.suggested_scale)
                        active_reason = ob_dec.reason
        except Exception:
            pass

    # 3. Perpetual Derivatives Funding Rate Crowding Guard
    funding_mode = validate_guard_mode_cfg(m["funding_guard_mode"], "funding_guard_mode")
    if funding_mode != "off":
        try:
            from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
            f_snap = get_derivatives_collector().fetch(symbol, allow_network=False)
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
                        _send_micro_shadow_notify("funding", f"🛡 [P2 · FUNDING] DEFER {order_side} {symbol}", f_dec.reason)
                    else:
                        scale_pct = int(f_dec.suggested_scale * 100)
                        _send_micro_shadow_notify("funding", f"🛡 [P2 · FUNDING] SCALE {scale_pct}% {order_side} {symbol}", f_dec.reason, scale_pct=scale_pct)
                if funding_mode == "enforce":
                    if not f_dec.allowed:
                        return False, f_dec.reason, 0.0
                    effective_scale = min(effective_scale, f_dec.suggested_scale)
                    active_reason = f_dec.reason
        except Exception:
            pass

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale
