"""Autonomous AI intent preflight, shadow observation, and live order execution."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from botcore import (
    format_notify_price,
    format_notify_qty,
    get_stale_cache_retry_delay_sec,
    normalize_side,
    safe_pos_float,
)
from cachepaths import is_equity_venue
from .base import MarketDataProvider
from .intent_state import (
    _intent_order_ids,
    _record_intent_state,
    _resolve_clamped_intent_notional,
    ai_intent_client_order_id,
    append_intent_history,
    collect_known_ai_order_ids,
    extract_intent_limit_price,
    extract_intent_order_type,
    normalize_intent_fields,
    parse_intent_confidence,
)
from .order_ohlc_utils import (
    extract_client_order_id,
    extract_executed_qty,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
    resolve_order_actor,
)

_TOPIC_BRACKET_TAGS: dict[str, str] = {
    "auto_technical_repair": "[auto_technical_repair]",
    "auto_cryptotrade_executor": "[auto_cryptotrade_executor]",
    "auto_trade_executor": "[auto_cryptotrade_executor]",
    "auto_sharetrade_executor": "[auto_sharetrade_executor]",
    "macro_geopolitical": "[auto_macro_geopolitical]",
    "macro_sentiment": "[auto_macro_sentiment]",
    "guard_high_stake": "[auto_guard_large_order]",
    "guard_post_stop": "[auto_guard_post_stop]",
}


def resolve_intent_topic_tag(intent: dict[str, Any], venue: str) -> str:
    """Resolve canonical topic bracket tag ('[auto_cryptotrade_executor]' or '[auto_sharetrade_executor]') for an intent."""
    raw_src = intent.get("created_by") if isinstance(intent, dict) else None
    src_topic = "" if raw_src is None or isinstance(raw_src, bool) else str(raw_src).strip()
    if src_topic in _TOPIC_BRACKET_TAGS:
        return _TOPIC_BRACKET_TAGS[src_topic]
    return MarketDataProvider.resolve_venue_intent_executor_tag(venue)


def record_shadow_intent_observation(
    inst: Any,
    intent: dict[str, Any],
    *,
    decision: str,
    target_qty: float,
    price: float,
    effective_usd: float,
    confidence: float,
    guard_allowed: bool,
    guard_reason: str,
    thesis: str,
    trade_tag: str,
    now_ts: float,
    is_limit: bool = False,
    dispatch_notify_fn: Optional[Callable[..., Any]] = None,
    log: Optional[Any] = None,
    retry_delay_sec: Optional[float] = None,
    sleep_fn: Optional[Callable[[float], Any]] = None,
) -> dict[str, Any]:
    """Record a shadow-mode intent observation and dispatch notification."""
    ord_lbl = f"{decision} LIMIT" if is_limit else decision
    qty_s = format_notify_qty(target_qty)
    px_log_s = format_notify_price(price, comma=False)
    px_body_s = format_notify_price(price, comma=True)
    if log:
        log.info(
            "[LLM TRADER · SHADOW] Would execute %s %s (qty=%s, price=$%s, notional=$%.2f) on %s. Thesis: %s",
            ord_lbl, inst.symbol, qty_s, px_log_s, effective_usd, inst.provider_name, thesis,
        )
    title = f"🟢 [{trade_tag.strip('[]')} · SHADOW] {ord_lbl} {inst.symbol} ${effective_usd:.0f}"
    body = (
        f"Venue: {inst.provider_name} · Qty: {qty_s} @ ${px_body_s} ({'LIMIT' if is_limit else 'MARKET'})\n"
        f"Confidence: {confidence:.2f} · Notional: ${effective_usd:.2f}\n"
        f"Operational Preflight: {'PASSED' if guard_allowed else guard_reason}\n"
        f"Thesis: {thesis}"
    )
    if callable(dispatch_notify_fn):
        dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

    extra_limit = {"order_type": "LIMIT", "limit_price": price} if is_limit else {}
    _record_intent_state(
        intent,
        "OBSERVED_SHADOW",
        now_ts,
        guard_allowed=guard_allowed,
        guard_reason=guard_reason,
        price=price,
        qty=target_qty,
        notional_usd=effective_usd,
        **extra_limit,
    )
    return intent


def execute_enforce_intent_order(
    inst: Any,
    intent: dict[str, Any],
    *,
    intent_id: str,
    decision: str,
    target_qty: float,
    price: float,
    effective_usd: float,
    guard_allowed: bool,
    guard_reason: str,
    thesis: str,
    trade_tag: str,
    now_ts: float,
    bypass_guards: Any,
    is_limit: bool = False,
    dispatch_notify_fn: Optional[Callable[..., Any]] = None,
    log: Optional[Any] = None,
    retry_delay_sec: Optional[float] = None,
    sleep_fn: Optional[Callable[[float], Any]] = None,
) -> dict[str, Any]:
    """Execute an enforce-mode trade intent via Instrument.place with retry on stale account cache."""
    ord_lbl = f"{decision} LIMIT" if is_limit else decision
    px_body_s = format_notify_price(price, comma=True)
    if not guard_allowed:
        if log:
            log.warning(
                "[LLM TRADER · ENFORCE] Operational guard blocked %s %s: %s",
                ord_lbl, inst.symbol, guard_reason,
            )
        title = f"🛡 {trade_tag} BLOCKED {ord_lbl} {inst.symbol}"
        body = (
            f"Venue: {inst.provider_name} · Price: ${px_body_s} · Notional: ${effective_usd:.0f}\n"
            f"Reason: {guard_reason}\n"
            f"Thesis: {thesis}"
        )
        if callable(dispatch_notify_fn):
            dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

        _record_intent_state(
            intent,
            "BLOCKED_BY_GUARD",
            now_ts,
            guard_reason=guard_reason,
            price=price,
            qty=target_qty,
            notional_usd=effective_usd,
        )
        return intent

    if log:
        log.warning(
            "[LLM TRADER · ENFORCE] Executing %s %s (qty=%s, price=$%s, notional=$%.2f) on %s...",
            ord_lbl, inst.symbol, format_notify_qty(target_qty), format_notify_price(price, comma=False), effective_usd, inst.provider_name,
        )

    try:
        intent_cid = ai_intent_client_order_id(intent_id)
        outcome_ctx: dict[str, Any] = {}
        order_res = None

        def _cleanup_refused_retry_records() -> None:
            if not intent_id:
                return
            try:
                MarketDataProvider.discard_refused_intent_retry_records(intent_id)
            except Exception:
                pass

        stale_cache_retries = 0
        retried_insufficient_free = False
        for _attempt in range(5):
            outcome_ctx.clear()
            place_kwargs: dict[str, Any] = {
                "side": decision,
                "qty": target_qty,
                "price": price,
                "market": not is_limit,
                "enforce_business_minimum": False,
                "wait_for_trend": False,
                "intent_id": intent_id,
                "client_order_id": intent_cid,
                "regime_context": None,
                "bypass_guards": bypass_guards,
                "_outcome_context": outcome_ctx,
            }
            if is_limit:
                place_kwargs["smart"] = False
            order_res = inst.place(**place_kwargs)
            if order_res:
                break
            refused_state = outcome_ctx.get("state") == "refused" or outcome_ctx.get("category") == "refused_pre_submit"
            reason_str = str(outcome_ctx.get("reason") or "")
            if (
                stale_cache_retries < 3
                and refused_state
                and reason_str in ("account_cache_not_fresh", "account_cache_snapshot_changed")
            ):
                stale_cache_retries += 1
                _cleanup_refused_retry_records()
                (sleep_fn if callable(sleep_fn) else time.sleep)(
                    get_stale_cache_retry_delay_sec(retry_delay_sec)
                )
                continue
            if (
                not retried_insufficient_free
                and refused_state
                and decision == "BUY"
                and "insufficient-free-for-stocks-buy" in reason_str.lower()
            ):
                retried_insufficient_free = True
                from .quantity import _floor_qty_decimals, round_provider_qty

                scaled_qty = _floor_qty_decimals(
                    round_provider_qty(getattr(inst, "_provider", None), str(getattr(inst, "symbol", "") or ""), target_qty * 0.94),
                    2,
                )
                if 0.0 < scaled_qty < target_qty and (price <= 0.0 or scaled_qty * price >= 1.0):
                    target_qty = scaled_qty
                    if price > 0.0:
                        effective_usd = round(target_qty * price, 2)
                    _cleanup_refused_retry_records()
                    continue
            break
        if order_res:
            _cleanup_refused_retry_records()
            if log:
                log.info(
                    "[LLM TRADER · ENFORCE] Successfully placed %s %s: %s",
                    ord_lbl, inst.symbol, order_res,
                )
            order_id_str = extract_order_id(order_res, default="N/A")
            intent["execution_payload"] = order_res
            extra_limit: dict[str, Any] = {"order_type": "LIMIT", "limit_price": price} if is_limit else {}
            if order_id_str and order_id_str != "N/A":
                extra_limit["order_id"] = order_id_str
            _record_intent_state(
                intent,
                "EXECUTED",
                now_ts,
                price=price,
                qty=round(target_qty, 8),
                notional_usd=effective_usd,
                payload=order_res,
                **extra_limit,
            )

            if callable(dispatch_notify_fn):
                try:
                    from notify_engine.alertnotifiers import dispatch_order_event_notification

                    dispatch_order_event_notification(
                        inst.symbol,
                        decision,
                        order_id_str,
                        price,
                        target_qty,
                        default_verb="PLACED LIMIT" if is_limit else "EXECUTED",
                        default_icon="📝" if is_limit else "🎉",
                        order_type="LIMIT" if is_limit else "MARKET",
                        venue=inst.provider_name,
                        owner="ai_executor",
                        source=None,
                        trade_tag=trade_tag,
                        notional_usd=effective_usd,
                        thesis=thesis,
                        notify_fn=dispatch_notify_fn,
                        include_price=False,
                    )
                except Exception as notify_exc:
                    if log:
                        log.debug("Notification failed after order execution: %s", notify_exc)
        else:
            refusal_reason = str(outcome_ctx.get("reason") or "Instrument.place returned None")
            refusal_cat = str(outcome_ctx.get("category") or "")
            if log:
                log.warning(
                    "[LLM TRADER · ENFORCE] Order placement refused or failed for %s %s: %s",
                    decision, inst.symbol, refusal_reason,
                )
            title = f"⚠️ {trade_tag} REFUSED {decision} {inst.symbol}"
            body = (
                f"Venue: {inst.provider_name} · Price: ${px_body_s} · Notional: ${effective_usd:.0f}\n"
                f"Reason: {refusal_reason}\n"
                f"Thesis: {thesis}"
            )
            if callable(dispatch_notify_fn):
                dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

            extra_cat = {"category": refusal_cat} if refusal_cat else {}
            _record_intent_state(intent, "FAILED", now_ts, reason=refusal_reason, **extra_cat)
            if outcome_ctx.get("state") == "refused":
                _cleanup_refused_retry_records()
    except Exception as e:
        if log:
            log.error("[LLM TRADER · ENFORCE] Exception executing %s %s: %s", decision, inst.symbol, e)
        title = f"🚨 {trade_tag} ERROR {decision} {inst.symbol}"
        body = f"Execution exception: {e}\nThesis: {thesis}"
        if callable(dispatch_notify_fn):
            dispatch_notify_fn(title=title, body=body, symbol=inst.symbol)

        _record_intent_state(intent, "FAILED", now_ts, action="ERROR", error=str(e))

    return intent


def cancel_existing_ai_open_orders(
    inst: Any,
    side: str,
    log: Optional[Any] = None,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    intents: Optional[list[dict[str, Any]]] = None,
    include_bot_orders: bool = False,
    any_side: bool = False,
    exclude_order_ids: Optional[set[str]] = None,
) -> list[str]:
    """Cancel prior unfilled AI-placed (or bot-placed when ``include_bot_orders=True``) open orders on the symbol."""
    canceled_ids: list[str] = []
    open_fn = getattr(inst, "open_orders", None)
    cancel_fn = getattr(inst, "cancel_order", None)
    want_side = normalize_side(side)
    if not callable(open_fn) or not callable(cancel_fn) or (not any_side and not want_side):
        return canceled_ids
    excluded = {str(x) for x in (exclude_order_ids or ()) if x}
    ai_oids = collect_known_ai_order_ids(intents, known_ai_order_ids)
    try:
        raw_open = open_fn()
        if isinstance(raw_open, list):
            for oo in raw_open:
                if not isinstance(oo, dict):
                    continue
                oo_side = extract_order_side(oo) or normalize_side(oo.get("side") or oo.get("type"))
                if not any_side and oo_side != want_side:
                    continue
                oid = extract_order_id(oo, default="")
                if oid and str(oid) in excluded:
                    continue
                is_ai_order = resolve_order_actor(oo, ai_oids) == "AI"
                if not is_ai_order and not include_bot_orders:
                    continue
                if oid:
                    try:
                        if cancel_fn(str(oid)) is False:
                            if log:
                                log.warning(
                                    "[INTENT GATEWAY] Venue refused cancel of prior open order %s on %s; it stays active.",
                                    oid, getattr(inst, "symbol", ""),
                                )
                            continue
                        canceled_ids.append(str(oid))
                        if log:
                            log.info(
                                "[INTENT GATEWAY] Canceled prior %s open %s order %s on %s before replacement.",
                                "AI" if is_ai_order else "BOT",
                                oo_side or side,
                                oid,
                                getattr(inst, "symbol", ""),
                            )
                    except Exception as ce:
                        if log:
                            log.debug("Could not cancel open order %s: %s", oid, ce)
    except Exception as e:
        if log:
            log.debug("Open order scan failed before replacement: %s", e)

    sym_str = str(getattr(inst, "symbol", "") or "").strip().upper()
    if sym_str and not hasattr(getattr(inst, "symbol", None), "_mock_name"):
        try:
            to_discard = MarketDataProvider.discard_superseded_symbol_retry_records(
                sym_str,
                want_side,
                canceled_ids=canceled_ids,
                ai_order_ids=ai_oids,
                include_bot_orders=include_bot_orders,
                any_side=any_side,
                cancel_fn_is_mock=hasattr(cancel_fn, "_mock_name"),
            )
            if to_discard and log:
                log.info(
                    "[INTENT GATEWAY] Discarded %d superseded order_retry record(s) on %s: %s",
                    len(to_discard), sym_str, to_discard,
                )
        except Exception as re_err:
            if log:
                log.debug("Order retry queue cleanup skipped on %s: %s", sym_str, re_err)
    return canceled_ids


def find_duplicate_ai_open_limit_order(
    inst: Any,
    side: str,
    limit_price: float,
    notional_usd: float,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    intents: Optional[list[dict[str, Any]]] = None,
    require_notional_match: bool = True,
) -> Optional[str]:
    """Return the venue order ID of an unfilled AI open limit order at the same side, price (<= 1 bps), and (when required) notional (<= 5%)."""
    open_fn = getattr(inst, "open_orders", None)
    want_side = normalize_side(side)
    if not callable(open_fn) or not want_side or not (limit_price and limit_price > 0) or not (notional_usd and notional_usd > 0):
        return None
    try:
        raw_open = open_fn()
        if not isinstance(raw_open, list):
            return None
        ai_oids = collect_known_ai_order_ids(intents, known_ai_order_ids)
        if isinstance(intents, dict):
            intents = intents.get("intents")
        intent_notional_by_id: dict[str, tuple[float, float]] = {}
        if require_notional_match and isinstance(intents, list):
            for item in intents:
                if not isinstance(item, dict):
                    continue
                p_sug = safe_pos_float(item.get("suggested_notional_usd"), default=None)
                p_clamp = safe_pos_float(item.get("clamped_notional_usd"), default=None)
                if p_clamp is None and isinstance(item.get("history"), list):
                    for h_entry in reversed(item["history"]):
                        if isinstance(h_entry, dict) and str(h_entry.get("action") or "").upper() in ("EXECUTED", "PLACED"):
                            p_clamp = safe_pos_float(h_entry.get("notional_usd"), default=None)
                            if p_clamp is not None:
                                break
                if p_sug is None or p_clamp is None:
                    continue
                for k in _intent_order_ids(item):
                    if k:
                        intent_notional_by_id[k] = (p_sug, p_clamp)
                        intent_notional_by_id[k.lower()] = (p_sug, p_clamp)
        for oo in raw_open:
            if not isinstance(oo, dict):
                continue
            oo_side = extract_order_side(oo) or normalize_side(oo.get("side") or oo.get("type"))
            if oo_side != want_side:
                continue
            if resolve_order_actor(oo, ai_oids) != "AI":
                continue
            if extract_executed_qty(oo) > 0:
                continue
            oo_px = extract_order_price(oo)
            if oo_px <= 0 or abs(oo_px - limit_price) / limit_price > 1e-4:
                continue
            oo_qty = extract_orig_qty(oo)
            oo_notional = oo_px * oo_qty
            if oo_notional <= 0:
                continue
            oid = extract_order_id(oo, default="")
            if require_notional_match:
                notional_ok = abs(oo_notional - notional_usd) <= max(1.0, 0.05 * max(notional_usd, oo_notional))
                if not notional_ok and intent_notional_by_id:
                    cid = extract_client_order_id(oo, default="")
                    prior_pair = (
                        intent_notional_by_id.get(str(oid))
                        or intent_notional_by_id.get(str(oid).lower())
                    ) if oid else None
                    if prior_pair is None and cid:
                        prior_pair = (
                            intent_notional_by_id.get(str(cid))
                            or intent_notional_by_id.get(str(cid).lower())
                        )
                    if prior_pair is not None:
                        p_sug, p_clamp = prior_pair
                        if (
                            abs(p_sug - notional_usd) <= max(1.0, 0.05 * max(notional_usd, p_sug))
                            and abs(oo_notional - p_clamp) <= max(1.0, 0.05 * max(p_clamp, oo_notional))
                        ):
                            notional_ok = True
                if not notional_ok:
                    continue
            if oid:
                return str(oid)
    except Exception:
        return None
    return None


def resolve_intent_execution_price(
    intent: dict[str, Any],
    *,
    intent_id: str,
    decision: str,
    symbol: str,
    live_price: float,
    now_ts: float,
    log: Optional[Any] = None,
) -> tuple[bool, float, bool]:
    """Validate and resolve MARKET or LIMIT execution price against live_price."""
    order_type = extract_intent_order_type(intent)
    limit_px = extract_intent_limit_price(intent, default=None)
    if order_type == "LIMIT":
        if limit_px is None or limit_px <= 0 or limit_px < live_price * 0.50 or limit_px > live_price * 2.00:
            fallback_mult = 0.995 if decision == "BUY" else 1.005
            price = round(live_price * fallback_mult, 8)
            if log:
                log.warning(
                    "[INTENT GATEWAY] Intent %s limit_price %r missing or out of [0.5x, 2.0x] bounds vs live $%.6g; "
                    "using passive fallback $%.6g for %s",
                    intent_id, limit_px, live_price, price, symbol,
                )
            append_intent_history(intent, {
                "ts": now_ts,
                "action": "FALLBACK_PASSIVE_LIMIT_PRICE",
                "requested_limit_price": limit_px,
                "fallback_limit_price": price,
                "live_price": live_price,
            })
        elif decision == "BUY" and limit_px > live_price:
            price = live_price
        elif decision == "SELL" and limit_px < live_price:
            price = live_price
        else:
            price = limit_px
        intent["order_type"] = "LIMIT"
        intent["limit_price"] = price
        return True, price, True
    intent["order_type"] = "MARKET"
    return True, live_price, False


def preflight_intent_and_resolve_price(
    intent: dict[str, Any],
    *,
    min_confidence: float,
    min_notional_usd: float,
    max_notional_usd: float,
    resolve_instrument_fn: Callable[..., Any],
    get_live_price_fn: Callable[[Any], Any],
    instruments_map: Optional[dict[str, Any]],
    now_ts: float,
    log: Optional[Any] = None,
) -> tuple[bool, str, str, str, str, float, Any, float, float, bool]:
    """Validate intent fields, confidence, instrument, and live/limit price before order execution."""
    intent_id, decision, symbol, venue = normalize_intent_fields(intent, now_ts)
    if decision not in ("BUY", "SELL"):
        _record_intent_state(intent, "REJECTED_INVALID_DECISION", now_ts, decision=decision)
        return False, intent_id, decision, symbol, venue, 0.0, None, 0.0, 0.0, False

    confidence = parse_intent_confidence(intent.get("confidence"), 0.0)
    intent["confidence"] = confidence
    if confidence < min_confidence:
        if log:
            log.info(
                "[INTENT GATEWAY] Intent %s rejected: confidence %.2f < %.2f",
                intent_id, confidence, min_confidence,
            )
        _record_intent_state(
            intent,
            "REJECTED_LOW_CONFIDENCE",
            now_ts,
            confidence=confidence,
            min_confidence=min_confidence,
        )
        return False, intent_id, decision, symbol, venue, confidence, None, 0.0, 0.0, False

    inst = resolve_instrument_fn(symbol, venue=venue, instruments_map=instruments_map)
    inst_sym_u = str(getattr(inst, "symbol", "") or "").strip().upper()
    inst_quote_u = str(getattr(inst, "quote", "") or "").strip().upper()
    if not inst or (
        not is_equity_venue(getattr(inst, "provider_name", ""))
        and ((len(inst_sym_u) > 3 and inst_sym_u.endswith("BTC")) or inst_quote_u == "BTC")
    ):
        if log:
            log.warning(
                "[INTENT GATEWAY] Intent %s rejected: unknown or non-fiat instrument %s on %s",
                intent_id, symbol, venue,
            )
        _record_intent_state(intent, "REJECTED_UNKNOWN_INSTRUMENT", now_ts, symbol=symbol, venue=venue)
        return False, intent_id, decision, symbol, venue, confidence, None, 0.0, 0.0, False

    effective_usd = _resolve_clamped_intent_notional(intent, min_notional_usd, max_notional_usd)

    live_price = safe_pos_float(get_live_price_fn(inst), default=None)
    if live_price is None:
        if log:
            log.warning("[INTENT GATEWAY] Intent %s rejected: invalid price for %s", intent_id, symbol)
        _record_intent_state(intent, "REJECTED_PRICE_UNAVAILABLE", now_ts, symbol=symbol)
        return False, intent_id, decision, symbol, venue, confidence, inst, effective_usd, 0.0, False

    intent["reference_price"] = round(live_price, 8)
    px_ok, price, is_limit = resolve_intent_execution_price(
        intent,
        intent_id=intent_id,
        decision=decision,
        symbol=symbol,
        live_price=live_price,
        now_ts=now_ts,
        log=log,
    )
    if not px_ok:
        return False, intent_id, decision, symbol, venue, confidence, inst, effective_usd, price, is_limit

    intent["requested_price"] = round(price, 8)
    return True, intent_id, decision, symbol, venue, confidence, inst, effective_usd, price, is_limit


__all__ = [
    "cancel_existing_ai_open_orders",
    "execute_enforce_intent_order",
    "find_duplicate_ai_open_limit_order",
    "preflight_intent_and_resolve_price",
    "record_shadow_intent_observation",
    "resolve_intent_execution_price",
    "resolve_intent_topic_tag",
]
