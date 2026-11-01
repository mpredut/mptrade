"""External Market Intelligence Order Guards (GI_OBOOK, GI_WHALE, GI_FUNDING).

Evaluates pre-trade orders against external market microstructure math (NO LLM, 0 ms):
- GI_OBOOK (OrderbookWallGuard): Spot orderbook depth, bid/ask imbalance ratio, and large ask/bid walls.
- GI_WHALE (WhaleDivergenceGuard): Top traders Long/Short ratio, taker volume flow, and OI divergence.
- GI_FUNDING / GI_CROWD (FundingCrowdingGuard): Perpetual derivatives funding rate crowding & squeeze risk.

Executes in 0 ms (reads local cachedb/ telemetry, zero network calls during order placement).
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict, Optional, Tuple

from botcore import _parse_bool_flag, safe_float
from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
import intelligence.external.guards.funding_crowding_guard as fcg
import intelligence.external.guards.orderbook_wall_guard as owg
import intelligence.external.guards.whale_divergence_guard as wdg
from intelligence.internal.guards.guard_decision import BrakeAction, fetch_guard_snapshot
from intelligence.macro_sentiment_ssot import (
    check_cooldown_gate,
    clean_scale as _clean_scale,
    clean_symbol_key,
    normalize_order_side,
    resolve_now_ts,
    safe_pos_float,
)

logger = logging.getLogger("external_order_guard")

_whale_collector = None
_orderbook_collector = None
_derivatives_collector = None
_MICRO_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str, str], float] = {}


def _get_collector(attr_name: str, module_path: str, cls_name: str):
    og = sys.modules.get("order_guard")
    if og and getattr(og, attr_name, None) is not None:
        return getattr(og, attr_name)
    cur = globals().get(attr_name)
    if cur is None:
        cur = getattr(importlib.import_module(module_path), cls_name)()
        globals()[attr_name] = cur
    return cur


def get_whale_collector():
    """Lazily load singleton whale positioning collector."""
    return _get_collector("_whale_collector", "intelligence.external.collectors.whale_positioning", "WhalePositioningCollector")


def get_orderbook_collector():
    """Lazily load singleton orderbook depth collector."""
    return _get_collector("_orderbook_collector", "intelligence.external.collectors.orderbook_depth", "OrderbookDepthCollector")


def get_derivatives_collector():
    """Lazily load singleton derivatives telemetry collector."""
    return _get_collector("_derivatives_collector", "intelligence.external.collectors.derivatives_telemetry", "DerivativesTelemetryCollector")


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
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards) if bypass_guards else set()
    if "all" in bypassed or "*" in bypassed:
        return True, "stop_loss_exempt", 1.0

    order_side = normalize_order_side(side)
    if order_side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    sym_norm = clean_symbol_key(symbol)

    from order_guard import _load_margins, is_intelligence_master_off, validate_guard_mode_cfg
    m = _load_margins() if margins is None else margins

    if is_intelligence_master_off(m):
        return True, "intelligence_guards_off", 1.0

    ext_mode = "enforce"
    if "external_guard_mode" in m:
        ext_mode = validate_guard_mode_cfg(m["external_guard_mode"], "external_guard_mode")
        if ext_mode == "off":
            return True, "external_guards_off", 1.0

    price_f = safe_pos_float(price, 0.0)

    effective_scale = 1.0
    active_reason = "ok"

    now_ts = resolve_now_ts(now)
    shadow_notify_enabled = _parse_bool_flag(m.get("shadow_notify"), False)

    def _send_micro_shadow_notify(
        guard_key: str,
        title: str,
        reason: str,
        scale_pct: Optional[int] = None,
    ) -> None:
        cd_key = (sym_norm or symbol, order_side, guard_key)
        if not check_cooldown_gate(_MICRO_SHADOW_NOTIFY_COOLDOWN, cd_key, now_ts):
            return
        try:
            from notify_engine.alertnotifiers import notify
            price_str = f" @ ${price_f:,.2f}" if price_f > 0.0 else ""
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

    def _apply_micro_decision(
        dec: Any,
        mode: str,
        log_tag: str,
        notify_key: str,
        code_tag: str,
        *,
        log_scale: bool = False,
        use_brake_value_in_title: bool = False,
    ) -> Optional[Tuple[bool, str, float]]:
        nonlocal effective_scale, active_reason
        d_scale = _clean_scale(getattr(dec, "suggested_scale", 1.0))
        if not dec.allowed or dec.brake_action in (BrakeAction.DEFER_WAIT, BrakeAction.DOWNSCALE_QTY):
            prefix = f"[{log_tag}_GUARD_SHADOW]" if mode == "shadow" else f"[{log_tag}_GUARD_ENFORCE]"
            scale_sfx = f", suggested_scale={d_scale}" if log_scale else ""
            print(f"{prefix} {order_side} {symbol}: {dec.reason} (brake={dec.brake_action}{scale_sfx})")
            if mode == "shadow" and shadow_notify_enabled:
                scale_pct = int(d_scale * 100) if d_scale < 1.0 else None
                if use_brake_value_in_title:
                    action_str = str(dec.brake_action.value)
                else:
                    action_str = f"SCALE {scale_pct}%" if (dec.allowed and scale_pct is not None) else "DEFER"
                _send_micro_shadow_notify(notify_key, f"🛡 [{code_tag}] {action_str} {order_side} {symbol}", dec.reason, scale_pct=scale_pct)
            if mode == "enforce":
                if not dec.allowed or d_scale <= 0.0:
                    return False, dec.reason, 0.0
                if d_scale < 1.0 and d_scale <= effective_scale:
                    effective_scale = d_scale
                    active_reason = dec.reason
        return None

    def _eval_ext_guard(
        mode_key: str,
        guard_id: str,
        collector_fn,
        build_guard,
        kw_name: str,
        log_tag: str,
        notify_key: str,
        *,
        log_scale: bool = False,
        use_brake_value_in_title: bool = False,
    ) -> Optional[Tuple[bool, str, float]]:
        g_mode = validate_guard_mode_cfg(m[mode_key], mode_key)
        if ext_mode == "shadow" and g_mode == "enforce":
            g_mode = "shadow"
        if g_mode == "off" or GUARD_REGISTRY[guard_id].is_bypassed(bypassed):
            return None
        try:
            snap = fetch_guard_snapshot(collector_fn(), symbol, now=now_ts, allow_network=False)
            guard = build_guard()
            try:
                dec = guard.check(symbol, order_side, **{kw_name: snap}, now=now_ts)
            except TypeError:
                dec = guard.check(symbol, order_side, **{kw_name: snap})
            return _apply_micro_decision(
                dec, g_mode, log_tag, notify_key, guard_id,
                log_scale=log_scale, use_brake_value_in_title=use_brake_value_in_title,
            )
        except Exception:
            return None

    # 1. GI_WHALE: Whale Flow & Open Interest Divergence Guard
    if (veto := _eval_ext_guard(
        "whale_guard_mode", "GI_WHALE", get_whale_collector,
        lambda: wdg.WhaleDivergenceGuard(
            severe_cascade_oi_pct=m["severe_cascade_oi_pct"],
            min_taker_ratio=m["min_taker_ratio"],
            taker_downscale_threshold=m["taker_downscale_threshold"],
            taker_downscaled_scale=m["taker_downscaled_scale"],
        ),
        "snapshot", "WHALE", "whale", use_brake_value_in_title=True,
    )) is not None:
        return veto

    # 2. GI_OBOOK: Orderbook Depth & Whale Limit Wall Guard
    if (veto := _eval_ext_guard(
        "orderbook_wall_guard_mode", "GI_OBOOK", get_orderbook_collector,
        lambda: owg.OrderbookWallGuard(
            min_buy_imbalance=m["min_buy_imbalance"],
            max_sell_imbalance=m["max_sell_imbalance"],
            whale_wall_usd_limit=m["whale_wall_usd_limit"],
        ),
        "snapshot", "ORDERBOOK", "orderbook",
    )) is not None:
        return veto

    # 3. GI_FUNDING (GI_CROWD): Perpetual Derivatives Funding Rate Crowding Guard
    if (veto := _eval_ext_guard(
        "funding_guard_mode", "GI_FUNDING", get_derivatives_collector,
        lambda: fcg.FundingCrowdingGuard(
            max_long_funding_rate=m["funding_max_long_rate"],
            min_short_funding_rate=m["funding_min_short_rate"],
            crowding_policy=str(m["funding_crowding_policy"] or "").strip().lower(),
            crowded_scale=m["funding_crowded_scale"],
        ),
        "telemetry", "FUNDING", "funding", log_scale=True,
    )) is not None:
        return veto

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


def is_gi_whale_enforced(bypass_guards: Any = None, *, venue: str = "binance") -> bool:
    """Return True if GI_WHALE is actively enforced and not bypassed in order_guard.conf."""
    try:
        from order_guard import _load_margins, resolve_effective_bypass_guards
        m = _load_margins()
        for mode_key in ("intelligence_guards_mode", "external_guard_mode", "whale_guard_mode"):
            if str(m.get(mode_key, "enforce")).strip().lower() != "enforce":
                return False
        bypassed = resolve_effective_bypass_guards(venue, bypass_guards=bypass_guards, margins=m)
        if GUARD_REGISTRY["GI_WHALE"].is_bypassed(bypassed) or "all" in bypassed or "*" in bypassed:
            return False
    except Exception:
        pass
    return True


def evaluate_whale_spread_flow(
    symbol: str,
    side: str,
    *,
    bypass_guards: Any = None,
    enforced: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Check if whale taker flow or OI divergence conflicts with initiating a spread round."""
    side_u = (side or "").upper()
    if side_u not in {"BUY", "SELL"}:
        return False, "ok"
    if not (is_gi_whale_enforced(bypass_guards) if enforced is None else enforced):
        return False, "gi_whale_bypassed_or_off"
    try:
        collector = get_whale_collector()
        if collector is None:
            return False, "no_collector"
        snap = collector.fetch(symbol, allow_network=False)
        if snap is None:
            return False, "no_snapshot"
        ratio = safe_pos_float(getattr(snap, "taker_buy_sell_ratio", None), 0.0) or 0.0
        oi_pct = safe_float(getattr(snap, "open_interest_1h_change_pct", None), default=0.0) or 0.0
        div_regime = getattr(snap, "divergence_regime", None)
        if side_u == "BUY":
            if 0.0 < ratio < 0.80:
                return True, f"taker_sell_dominance (ratio={ratio:.2f} < 0.80)"
            if div_regime == "aggressive_shorting":
                return True, f"aggressive_shorting_divergence (OI={oi_pct:+.1f}%)"
        else:
            if ratio > 1.25:
                return True, f"taker_buy_dominance (ratio={ratio:.2f} > 1.25)"
            if div_regime in ("short_squeeze", "short_squeeze_primed"):
                return True, f"short_squeeze_divergence (OI={oi_pct:+.1f}%)"
    except Exception as exc:
        return False, f"error:{exc}"
    return False, "ok"


def resolve_whale_spread_adjustment(
    symbol: str,
    base_adjustment: float,
    *,
    bypass_guards: Any = None,
    enforced: Optional[bool] = None,
) -> float:
    """Widen grid quote adjustment when moderate taker flow pressure exists."""
    if not (is_gi_whale_enforced(bypass_guards) if enforced is None else enforced):
        return base_adjustment
    try:
        collector = get_whale_collector()
        if collector is not None:
            snap = collector.fetch(symbol, allow_network=False)
            ratio = (safe_pos_float(getattr(snap, "taker_buy_sell_ratio", None), 0.0) or 0.0) if snap is not None else 0.0
            if ratio > 0.0 and (ratio < 0.90 or ratio > (1.0 / 0.90)):
                return base_adjustment * 1.5
    except Exception:
        pass
    return base_adjustment

