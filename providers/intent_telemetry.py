"""Pure deterministic AI trade intent cycle telemetry, fill resolution, and slippage evaluation."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from botcore import (
    _parse_bool_flag,
    _ts_ms_to_s,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    first_pos_float,
    normalize_timestamp_ms,
    resolve_now_ts,
    safe_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    compatible_quote_suffixes,
    filter_candidates_by_asset_class,
    is_equity_venue_or_suffix,
    matches_asset_scope,
    normalize_venue_name,
)
from instrument_registry import casefold_get, resolve_symbol_identity
from .base import MarketDataProvider
from .intent_state import (
    _clean_intent_symbol,
    _intent_venue_symbol_side_key,
    extract_intent_order_type,
    extract_intent_primary_order_id,
    extract_intent_side,
    extract_intent_timestamp_s,
    extract_intent_trade_plan_levels,
    intent_venue_client_order_ids,
    order_matches_intent_client_id,
    parse_pos_epoch_s,
    symbols_match_intent_scope_and_quote,
)
from .order_ohlc_utils import (
    extract_executed_qty,
    extract_orig_qty,
)
from .order_status_ssot import (
    RECONCILABLE_AI_INTENT_STATUSES,
    TERMINAL_CANCEL_VENUE_STATUSES,
    UNFILLED_ORDER_STATUSES,
    format_prompt_intent_status,
)


def _parse_pos_epoch_s(raw: Any) -> Optional[float]:
    """Parse an epoch-seconds, epoch-ms, or ISO-8601 timestamp into positive UTC seconds."""
    return parse_pos_epoch_s(raw)


def _filter_intents_by_scope(intents_data: Any, scope: str) -> list[dict[str, Any]]:
    """Filter intent records to those matching the requested asset scope ('crypto', 'equities', or 'all')."""
    if isinstance(intents_data, dict):
        intents_data = intents_data.get("intents")
    if not isinstance(intents_data, list):
        return []
    return [
        i
        for i in intents_data
        if isinstance(i, dict) and matches_asset_scope(i.get("symbol"), i.get("venue"), scope)
    ]


def _match_intent_open_order(
    intent: dict[str, Any],
    *,
    raw_intent_st: str,
    venue: Optional[str],
    symbol: str,
    side: str,
    order_type: str,
    order_id: Optional[str],
    open_orders: Optional[list[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    """Find a matching venue open order for an intent by order ID or venue-native client ID."""
    if not isinstance(open_orders, list) or (
        raw_intent_st != "EXECUTED" and raw_intent_st not in RECONCILABLE_AI_INTENT_STATUSES
    ):
        return None
    expected_cids: Optional[set[str]] = None
    for oo in open_orders:
        if not isinstance(oo, dict):
            continue
        oo_venue = normalize_venue_name(clean_scalar_str(oo.get("venue")) or "")
        oo_sym = _clean_intent_symbol(oo.get("symbol"))
        if not symbols_match_intent_scope_and_quote(
            oo_sym, symbol, venue_a=oo_venue, venue_b=venue, allow_missing_symbol=True
        ):
            continue
        oo_oid = extract_order_id(oo, default=None)
        if order_id is not None and oo_oid is not None and str(oo_oid) == str(order_id):
            return oo
        if (
            order_id is None
            and order_type == "LIMIT"
            and oo_venue == venue
            and oo_sym
            and (extract_order_side(oo) or str(oo.get("side") or "").strip().upper()) == side
        ):
            if expected_cids is None:
                expected_cids = intent_venue_client_order_ids(intent)
            if order_matches_intent_client_id(
                oo, intent, expected_cids=expected_cids, allow_untagged_ai_actor=True
            ):
                return oo
    return None


def _match_intent_recent_fill(
    *,
    venue: Optional[str],
    symbol: str,
    side: str,
    order_type: str,
    order_id: Optional[str],
    ts_val: float,
    is_eq_intent: bool,
    venue_st: str,
    fill_px: Optional[float],
    filled_qty: float,
    recent_orders: dict[str, Any],
) -> tuple[Optional[float], float, int]:
    """Match an executed intent against symbol ``recent_orders`` fill history."""
    raw_cands = cache_symbol_variants(symbol, quotes=compatible_quote_suffixes(symbol))
    sym_cands = filter_candidates_by_asset_class(raw_cands, is_equity=is_eq_intent)
    ro_entry = next(
        (
            val
            for key_cand in (*(f"{venue}:{cand}" for cand in sym_cands), *sym_cands)
            if isinstance((val := casefold_get(recent_orders, key_cand)), dict)
            and (
                ":" in key_cand
                or not venue
                or not val.get("venue")
                or normalize_venue_name(val.get("venue")) == venue
            )
        ),
        None,
    )
    if not isinstance(ro_entry, dict):
        return fill_px, filled_qty, 0
    side_key = "buy" if side == "BUY" else ("sell" if side == "SELL" else "")
    fill_cands: list[dict[str, Any]] = []
    if side_key:
        last_c = ro_entry.get(f"last_{side_key}")
        recent_c = ro_entry.get(f"recent_{side_key}s")
        for cand_f in ([last_c] if isinstance(last_c, dict) else []) + (
            list(reversed(recent_c)) if isinstance(recent_c, list) else []
        ):
            if isinstance(cand_f, dict) and cand_f not in fill_cands:
                fill_cands.append(cand_f)
    min_fill_ts_ms = ts_val * 1000.0 if order_type == "LIMIT" else (ts_val - 120.0) * 1000.0
    last_fill = None
    for cand_f in fill_cands:
        f_oid = clean_order_id(cand_f.get("order_id"))
        if order_id is not None and f_oid:
            if f_oid == str(order_id):
                last_fill = cand_f
                break
            continue
        f_ts_ms = normalize_timestamp_ms(cand_f.get("ts_ms"), default=0) or 0
        f_actor = str(cand_f.get("actor") or "").strip().upper()
        if f_actor != "BOT" and (ts_val <= 0 or f_ts_ms <= 0 or f_ts_ms >= min_fill_ts_ms):
            last_fill = cand_f
            break
    if not isinstance(last_fill, dict):
        return fill_px, filled_qty, 0
    if fill_px is None or (venue_st in UNFILLED_ORDER_STATUSES and filled_qty <= 0):
        fill_px = safe_pos_float(last_fill.get("price"), default=fill_px)
    if filled_qty <= 0:
        filled_qty = safe_pos_float(last_fill.get("qty"), default=0.0) or 0.0
    matched_fill_ts_ms = normalize_timestamp_ms(last_fill.get("ts_ms"), default=0) or 0
    return fill_px, filled_qty, matched_fill_ts_ms


def _compute_signed_exec_bps(side: str, fill_px: float, anchor_px: float) -> float:
    raw = (
        ((fill_px - anchor_px) / anchor_px) * 10_000.0
        if side == "BUY"
        else ((anchor_px - fill_px) / anchor_px) * 10_000.0
    )
    return round(raw, 2) + 0.0


def _classify_intent_exec_status_and_slippage(
    intent: dict[str, Any],
    *,
    last_hist: dict[str, Any],
    raw_intent_st: str,
    side: str,
    order_type: str = "",
    venue_st: str,
    matched_open: Optional[dict[str, Any]],
    is_unfilled_resting_limit: bool,
    is_unfilled_accepted_market: bool,
    fill_px: Optional[float],
    filled_qty: float,
) -> tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute canonical ``(exec_status, venue_st, fill_px, ref_px, slippage_bps, arrival_shortfall_bps)``."""
    if (
        fill_px is None
        and raw_intent_st in ("EXECUTED", "OBSERVED_SHADOW")
        and not is_unfilled_resting_limit
        and not is_unfilled_accepted_market
    ):
        fill_px = first_pos_float(
            last_hist.get("price"),
            intent.get("requested_price"),
            intent.get("limit_price"),
            intent.get("reference_price"),
        )

    exec_status = format_prompt_intent_status(
        raw_intent_st,
        venue_st,
        in_open_book=matched_open is not None,
        filled_qty=filled_qty,
        is_unfilled_resting_limit=is_unfilled_resting_limit,
        is_unfilled_accepted_market=is_unfilled_accepted_market,
        section="last_cycle_execution_status",
    )
    if matched_open is not None:
        if exec_status == "PARTIALLY_FILLED" and venue_st in UNFILLED_ORDER_STATUSES:
            venue_st = "PARTIALLY_FILLED"
    elif exec_status == "FILLED" and raw_intent_st == "EXECUTED" and venue_st in UNFILLED_ORDER_STATUSES:
        venue_st = "FILLED"

    eff_ot = clean_upper_scalar(order_type) or extract_intent_order_type(intent)
    lim_px = first_pos_float(
        matched_open.get("price") if isinstance(matched_open, dict) else None,
        intent.get("limit_price"),
        intent.get("requested_price") if eff_ot == "LIMIT" else None,
    )
    arrival_ref_px = first_pos_float(
        intent.get("reference_price"),
        intent.get("limit_price"),
        intent.get("requested_price"),
        last_hist.get("price"),
        fill_px,
    )
    ref_px = arrival_ref_px
    if exec_status in ("OPEN_RESTING", "ACCEPTED", *TERMINAL_CANCEL_VENUE_STATUSES) and filled_qty <= 0:
        fill_px = None
    slippage_bps: Optional[float] = None
    arrival_shortfall_bps: Optional[float] = None
    if (
        exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED")
        and fill_px is not None
        and fill_px > 0
    ):
        if eff_ot == "LIMIT" and lim_px is not None and lim_px > 0:
            slippage_bps = _compute_signed_exec_bps(side, fill_px, lim_px)
            if (
                arrival_ref_px is not None
                and arrival_ref_px > 0
                and abs(arrival_ref_px - lim_px) / lim_px > 1e-6
            ):
                arrival_shortfall_bps = _compute_signed_exec_bps(side, fill_px, arrival_ref_px)
        elif ref_px is not None and ref_px > 0:
            slippage_bps = _compute_signed_exec_bps(side, fill_px, ref_px)
        elif exec_status in ("FILLED", "SHADOW_SIMULATED"):
            slippage_bps = 0.0
    return exec_status, venue_st, fill_px, ref_px, slippage_bps, arrival_shortfall_bps


def _resolve_intent_fill_ts_and_reason(
    intent: dict[str, Any],
    *,
    exec_pay: dict[str, Any],
    last_hist: dict[str, Any],
    exec_status: str,
    eff_qty: Optional[float],
    ts_val: float,
    matched_fill_ts_ms: int,
) -> tuple[float, Optional[str]]:
    """Resolve effective fill timestamp and sanitized failure reason for an intent."""
    fill_ts_val = ts_val
    is_fill_exec = exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED") or (
        eff_qty is not None and eff_qty > 0
    )
    if is_fill_exec and isinstance(exec_pay, dict):
        for _ftk in ("filled_at", "filled_ts", "terminal_ts", "updated_ts", "updateTime", "workingTime", "transactTime"):
            if (_f_s := _parse_pos_epoch_s(exec_pay.get(_ftk))) is not None and _f_s >= ts_val:
                fill_ts_val = max(fill_ts_val, _f_s)
                break
    if is_fill_exec and matched_fill_ts_ms > 0:
        _mf_s = _ts_ms_to_s(safe_pos_float(matched_fill_ts_ms, default=0.0) or 0.0)
        if _mf_s > fill_ts_val:
            fill_ts_val = _mf_s
    if is_fill_exec and fill_ts_val <= ts_val and str(last_hist.get("action") or "") in (
        "RECONCILED_EXECUTED",
        "RECONCILED_PARTIAL",
    ):
        if (_lh_s := _parse_pos_epoch_s(last_hist.get("ts"))) is not None and _lh_s > fill_ts_val:
            fill_ts_val = _lh_s
    fail_reason: Optional[str] = None
    if exec_status not in ("FILLED", "PARTIALLY_FILLED", "OPEN_RESTING", "ACCEPTED", "SHADOW_SIMULATED"):
        raw_fail_reason = next(
            (
                s
                for cand in (
                    last_hist.get("reason"),
                    last_hist.get("guard_reason"),
                    last_hist.get("error"),
                    intent.get("guard_reason"),
                    intent.get("circuit_breaker_reason"),
                    intent.get("error"),
                    exec_pay.get("reason") if isinstance(exec_pay, dict) else None,
                    exec_pay.get("error") if isinstance(exec_pay, dict) else None,
                )
                if (s := clean_scalar_str(cand)) is not None
            ),
            None,
        )
        if raw_fail_reason:
            fail_reason = " ".join(raw_fail_reason.split())[:200] or None
    return fill_ts_val, fail_reason


def _resolve_intent_execution_details(
    i: dict[str, Any],
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Resolve fill price, filled quantity, execution status, and slippage (bps) for an intent."""
    raw_i_venue = clean_scalar_str(i.get("venue"))
    venue = (normalize_venue_name(raw_i_venue) or raw_i_venue) if raw_i_venue else None
    symbol = _clean_intent_symbol(i.get("symbol"))
    side = extract_intent_side(i) or clean_upper_scalar(i.get("decision")) or ""
    order_type = extract_intent_order_type(i)
    exec_pay = i.get("execution_payload") if isinstance(i.get("execution_payload"), dict) else {}
    hist = i.get("history") if isinstance(i.get("history"), list) else []
    last_hist = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    exec_quote_notional = first_pos_float(
        exec_pay.get("cummulativeQuoteQty"),
        exec_pay.get("cost"),
        exec_pay.get("filledValue"),
    )
    intent_order_notional = first_pos_float(
        i.get("clamped_notional_usd"),
        last_hist.get("notional_usd"),
        extract_notional_fn(i, default=None),
    )
    eff_notional = exec_quote_notional or intent_order_notional
    ts_raw, ts_val = extract_intent_timestamp_s(i)
    order_id = extract_intent_primary_order_id(i, default=None)
    raw_intent_st = clean_upper_scalar(i.get("status")) or ""
    is_eq_intent = is_equity_venue_or_suffix(symbol, venue)

    matched_open = _match_intent_open_order(
        i,
        raw_intent_st=raw_intent_st,
        venue=venue,
        symbol=symbol,
        side=side,
        order_type=order_type,
        order_id=order_id,
        open_orders=open_orders,
    )
    fill_px = extract_order_price(exec_pay, default=None)
    filled_qty = extract_executed_qty(exec_pay, default=0.0)
    orig_qty = extract_orig_qty(exec_pay, default=None) or safe_pos_float(last_hist.get("qty"), default=None)
    if matched_open is not None:
        open_filled = extract_executed_qty(matched_open, default=0.0)
        if open_filled > filled_qty:
            filled_qty = open_filled
        if fill_px is None:
            fill_px = safe_pos_float(matched_open.get("price"), default=None)

    venue_st = (
        clean_upper_scalar(exec_pay.get("status"))
        or ("RESTING" if (clean_scalar_str(exec_pay.get("msg")) or "").lower() == "resting" else "")
        or (
            "OPEN"
            if (
                order_type == "LIMIT"
                and filled_qty <= 0
                and isinstance(exec_pay, dict)
                and any(k in exec_pay for k in ("txid", "ordertxid", "descr", "orderId", "order_id", "oid"))
            )
            else ""
        )
    )
    if matched_open is not None and not venue_st:
        venue_st = clean_upper_scalar(matched_open.get("status")) or "OPEN"

    matched_fill_ts_ms = 0
    if (
        matched_open is None
        and raw_intent_st == "EXECUTED"
        and venue_st not in TERMINAL_CANCEL_VENUE_STATUSES
        and isinstance(recent_orders, dict)
        and symbol
    ):
        fill_px, filled_qty, matched_fill_ts_ms = _match_intent_recent_fill(
            venue=venue,
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_id=order_id,
            ts_val=ts_val,
            is_eq_intent=is_eq_intent,
            venue_st=venue_st,
            fill_px=fill_px,
            filled_qty=filled_qty,
            recent_orders=recent_orders,
        )

    is_unfilled_resting_limit = (
        order_type == "LIMIT"
        and (venue_st in UNFILLED_ORDER_STATUSES or not venue_st)
        and filled_qty <= 0
    )
    is_unfilled_accepted_market = (
        order_type != "LIMIT"
        and _parse_bool_flag(exec_pay.get("reconciled_from_audit"), False)
        and (venue_st in UNFILLED_ORDER_STATUSES or not venue_st)
        and filled_qty <= 0
    )
    venue_open_verified = (
        verified_open_venues is None
        or (
            bool(venue)
            and isinstance(verified_open_venues, (set, frozenset, list, tuple))
            and normalize_venue_name(venue) in verified_open_venues
        )
    )
    if (
        is_unfilled_resting_limit
        and matched_open is None
        and isinstance(open_orders, list)
        and venue_open_verified
        and ts_val > 0
        and (resolve_now_ts(now_ts) - ts_val) > MarketDataProvider.resolve_unfilled_order_cancel_grace_sec()
    ):
        venue_st = "CANCELED_OR_EXPIRED"

    exec_status, venue_st, fill_px, ref_px, slippage_bps, arrival_shortfall_bps = _classify_intent_exec_status_and_slippage(
        i,
        last_hist=last_hist,
        raw_intent_st=raw_intent_st,
        side=side,
        order_type=order_type,
        venue_st=venue_st,
        matched_open=matched_open,
        is_unfilled_resting_limit=is_unfilled_resting_limit,
        is_unfilled_accepted_market=is_unfilled_accepted_market,
        fill_px=fill_px,
        filled_qty=filled_qty,
    )
    eff_qty = (
        filled_qty
        if filled_qty > 0
        else (orig_qty if (orig_qty is not None and exec_status in ("FILLED", "SHADOW_SIMULATED")) else None)
    )
    limit_px = (
        first_pos_float(
            matched_open.get("price") if isinstance(matched_open, dict) else None,
            i.get("limit_price") if exec_status == "FILLED" else None,
            exec_pay.get("price") if isinstance(exec_pay, dict) else None,
            i.get("limit_price"),
        )
        if order_type == "LIMIT"
        else None
    )
    if (
        order_type == "LIMIT"
        and exec_status == "FILLED"
        and fill_px is not None
        and limit_px is not None
        and abs(fill_px - limit_px) <= limit_px * 1e-5
    ):
        fill_px = limit_px
    fill_ts_val, fail_reason = _resolve_intent_fill_ts_and_reason(
        i,
        exec_pay=exec_pay,
        last_hist=last_hist,
        exec_status=exec_status,
        eff_qty=eff_qty,
        ts_val=ts_val,
        matched_fill_ts_ms=matched_fill_ts_ms,
    )
    filled_notional_usd: Optional[float] = None
    if exec_status == "PARTIALLY_FILLED":
        raw_filled_notional = exec_quote_notional or (
            (eff_qty * fill_px)
            if (eff_qty is not None and eff_qty > 0 and fill_px is not None and fill_px > 0)
            else None
        )
        if raw_filled_notional is not None and raw_filled_notional > 0:
            filled_notional_usd = round(raw_filled_notional, 2)
        orig_order_px = limit_px or fill_px or ref_px
        eff_notional = (
            intent_order_notional
            or ((orig_qty * orig_order_px) if (orig_qty is not None and orig_order_px is not None) else None)
            or exec_quote_notional
        )
    return {
        "venue": venue,
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "raw_status": i.get("status"),
        "venue_status": venue_st or None,
        "exec_status": exec_status,
        "is_open_in_venue": matched_open is not None,
        "order_id": str(order_id) if order_id is not None else None,
        "fill_px": round(fill_px, 6) if fill_px is not None else None,
        "limit_px": round(limit_px, 6) if limit_px is not None else None,
        "ref_px": round(ref_px, 6) if ref_px is not None else None,
        "filled_qty": round(eff_qty, 8) if eff_qty is not None and eff_qty > 0 else None,
        "filled_notional_usd": filled_notional_usd,
        "notional_usd": round(eff_notional, 2) if eff_notional is not None else i.get("suggested_notional_usd"),
        "slippage_bps": slippage_bps,
        "arrival_shortfall_bps": arrival_shortfall_bps,
        "reason": fail_reason,
        "ts": ts_raw,
        "ts_val": ts_val,
        "fill_ts_val": fill_ts_val,
    }


def _is_open_resting_detail(det: dict[str, Any]) -> bool:
    return bool(
        _parse_bool_flag(det.get("is_open_in_venue"), False)
        and det.get("exec_status") in ("OPEN_RESTING", "PARTIALLY_FILLED")
    )


def _attach_intent_age_fields(
    target: dict[str, Any],
    det: dict[str, Any],
    now_ts: float,
    *,
    unit_sec: float,
    ndigits: int,
    suffix: str,
) -> None:
    now_f = safe_pos_float(now_ts, default=None)
    unit_f = safe_pos_float(unit_sec, default=None)
    if now_f is None or unit_f is None:
        return
    ts_val = safe_pos_float(det.get("ts_val"), default=0.0) or 0.0
    eff_ts = safe_pos_float(det.get("fill_ts_val"), default=None) or ts_val
    if eff_ts > 0 and now_f >= eff_ts:
        target[f"age_{suffix}"] = round((now_f - eff_ts) / unit_f, ndigits)
    if ts_val > 0 and eff_ts > ts_val + 60.0 and now_f >= ts_val:
        target[f"placed_age_{suffix}"] = round((now_f - ts_val) / unit_f, ndigits)


def _resolve_executed_open_resting_detail(
    intent_obj: dict[str, Any],
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> Optional[dict[str, Any]]:
    if str(intent_obj.get("status") or "").strip().upper() != "EXECUTED":
        return None
    det = _resolve_intent_execution_details(
        intent_obj,
        now_ts,
        extract_notional_fn,
        recent_orders=recent_orders,
        open_orders=open_orders,
        verified_open_venues=verified_open_venues,
    )
    return det if _is_open_resting_detail(det) else None


def summarize_recent_ai_intents(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Summarize the most recent AI trade intents filtered by asset scope."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)

    def _to_summary(i: dict[str, Any], det: dict[str, Any]) -> dict[str, Any]:
        intent_summary: dict[str, Any] = {
            "decision": i.get("decision") or det["side"],
            "symbol": i.get("symbol"),
            "venue": det["venue"],
            "order_type": det["order_type"],
            "status": det["raw_status"],
            "notional_usd": det["notional_usd"],
            "ts": det["ts"],
        }
        if i.get("limit_price") is not None:
            intent_summary["limit_price"] = i.get("limit_price")
        if i.get("invalidation_px") is not None:
            intent_summary["invalidation_px"] = i.get("invalidation_px")
        if i.get("target_px") is not None:
            intent_summary["target_px"] = i.get("target_px")
        if det["venue_status"]:
            intent_summary["venue_status"] = det["venue_status"]
        for k in ("order_id", "fill_px", "filled_qty", "filled_notional_usd", "slippage_bps", "arrival_shortfall_bps"):
            if det.get(k) is not None:
                intent_summary[k] = det[k]
        if det.get("reason"):
            intent_summary["reason"] = det["reason"]
        _attach_intent_age_fields(intent_summary, det, now_ts, unit_sec=3600.0, ndigits=2, suffix="h")
        return intent_summary

    recent_rev: list[dict[str, Any]] = []
    non_resting_count = 0
    seen_oids: set[str] = set()
    seen_sym_side: set[tuple[str, str, tuple[str, ...], str]] = set()
    recent_indices: set[int] = set()
    for idx in range(len(scoped_intents) - 1, -1, -1):
        if non_resting_count >= 8 or len(recent_rev) >= 16:
            break
        i = scoped_intents[idx]
        det = _resolve_intent_execution_details(
            i,
            now_ts,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        sym_side_key = _intent_venue_symbol_side_key(det.get("venue"), det.get("symbol"), det.get("side"))
        if (
            det.get("exec_status")
            in ("CANCELED", "CANCELLED", "EXPIRED", "CANCELED_OR_EXPIRED", "SKIPPED_DUPLICATE_OPEN_ORDER")
            and (safe_pos_float(det.get("filled_qty"), default=0.0) or 0.0) <= 0.0
            and sym_side_key in seen_sym_side
        ):
            continue
        seen_sym_side.add(sym_side_key)
        recent_indices.add(idx)
        if det["order_id"] is not None:
            seen_oids.add(str(det["order_id"]))
        sum_item = _to_summary(i, det)
        recent_rev.append(sum_item)
        if not (
            isinstance(open_orders, list)
            and open_orders
            and _is_open_resting_detail(det)
            and (sum_item.get("age_h") or 0.0) > 0.5
        ):
            non_resting_count += 1
    recent_intents_out: list[dict[str, Any]] = list(reversed(recent_rev))

    if isinstance(open_orders, list) and open_orders and len(scoped_intents) > len(recent_intents_out):
        older_resting: list[dict[str, Any]] = []
        for idx, i in enumerate(scoped_intents):
            if idx in recent_indices:
                continue
            det = _resolve_executed_open_resting_detail(
                i,
                now_ts,
                extract_notional_fn,
                recent_orders=recent_orders,
                open_orders=open_orders,
                verified_open_venues=verified_open_venues,
            )
            if det is not None and (det["order_id"] is None or str(det["order_id"]) not in seen_oids):
                if det["order_id"] is not None:
                    seen_oids.add(str(det["order_id"]))
                older_resting.append(_to_summary(i, det))
        if older_resting:
            recent_intents_out = older_resting + recent_intents_out
    return recent_intents_out


def build_last_cycle_execution_status(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
) -> Optional[Any]:
    """Build closed-loop mathematical fill confirmation (fill_px, slippage_bps, status) for the last cycle's intent(s)."""
    scoped_intents = _filter_intents_by_scope(intents_data, scope)
    if not scoped_intents:
        return None

    def _eff_intent_ts(intent_obj: dict[str, Any]) -> float:
        base_t = _parse_pos_epoch_s(intent_obj.get("timestamp")) or 0.0
        ep = intent_obj.get("execution_payload")
        if isinstance(ep, dict):
            ep_st = str(ep.get("status") or "").strip().upper()
            has_fill = (
                ep_st in ("FILLED", "PARTIALLY_FILLED", "CLOSED")
                or extract_executed_qty(ep, default=0.0) > 0.0
                or ep.get("filled_at") is not None
                or ep.get("filled_ts") is not None
            )
            if has_fill and ep_st not in TERMINAL_CANCEL_VENUE_STATUSES:
                for k in ("filled_at", "filled_ts", "terminal_ts", "updated_ts"):
                    if (ft := _parse_pos_epoch_s(ep.get(k))) is not None:
                        base_t = max(base_t, ft)
        return base_t

    last_ts = max((_eff_intent_ts(i) for i in scoped_intents), default=0.0)
    if last_ts > 0:
        cycle_intents = [
            i for i in scoped_intents if abs(_eff_intent_ts(i) - last_ts) <= 90.0
        ]
    else:
        cycle_intents = [scoped_intents[-1]]

    def _build_status_rec(det: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "venue": det["venue"],
            "symbol": det["symbol"],
            "side": det["side"],
            "order_type": det["order_type"],
            "status": det["exec_status"],
            "fill_px": det["fill_px"],
            "ref_px": det["ref_px"],
            "filled_qty": det["filled_qty"],
            "notional_usd": det["notional_usd"],
            "slippage_bps": det["slippage_bps"],
        }
        for k in ("filled_notional_usd", "limit_px", "order_id", "arrival_shortfall_bps"):
            if det.get(k) is not None:
                rec[k] = det[k]
        if det.get("reason"):
            rec["reason"] = det["reason"]
        _attach_intent_age_fields(rec, det, now_ts, unit_sec=60.0, ndigits=1, suffix="min")
        return rec

    out: list[dict[str, Any]] = []
    seen_oids: set[str] = set()
    seen_sym_keys: set[tuple[str, str, tuple[str, ...], str]] = set()
    cycle_ids = {id(i) for i in cycle_intents}
    now_f = safe_pos_float(now_ts, default=None)
    for i in cycle_intents:
        det = _resolve_intent_execution_details(
            i,
            now_ts,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        eff_ts_det = safe_pos_float(det.get("fill_ts_val"), default=None) or det["ts_val"]
        if (
            not _is_open_resting_detail(det)
            and det.get("exec_status") not in ("OPEN_RESTING", "PARTIALLY_FILLED")
            and eff_ts_det > 0
            and now_f is not None
            and now_f > eff_ts_det
            and (now_f - eff_ts_det) > 3600.0
        ):
            continue
        if det["order_id"] is not None:
            seen_oids.add(str(det["order_id"]))
        if _is_open_resting_detail(det):
            seen_sym_keys.add(_intent_venue_symbol_side_key(i.get("venue"), i.get("symbol"), extract_intent_side(i)))
        out.append(_build_status_rec(det))

    if isinstance(open_orders, list) and open_orders:
        carried_resting: list[dict[str, Any]] = []
        for i in reversed(scoped_intents):
            if id(i) in cycle_ids:
                continue
            det = _resolve_executed_open_resting_detail(
                i,
                now_ts,
                extract_notional_fn,
                recent_orders=recent_orders,
                open_orders=open_orders,
                verified_open_venues=verified_open_venues,
            )
            if det is None:
                continue
            oid_str = str(det["order_id"]) if det["order_id"] is not None else None
            s_key = _intent_venue_symbol_side_key(i.get("venue"), i.get("symbol"), extract_intent_side(i))
            if oid_str is not None:
                if oid_str in seen_oids:
                    continue
                seen_oids.add(oid_str)
            elif s_key in seen_sym_keys:
                continue
            seen_sym_keys.add(s_key)
            carried_resting.append(_build_status_rec(det))
        if carried_resting:
            out = list(reversed(carried_resting)) + out

    if not out:
        return None
    return out[0] if len(out) == 1 else out


def evaluate_trade_plan_status(
    side: str,
    spot_px: Optional[float],
    invalidation_px: Optional[float],
    target_px: Optional[float],
    *,
    is_resting: bool = False,
) -> str:
    """Evaluate live status ('ACTIVE' | 'TARGET_HIT' | 'INVALIDATED') of a structured AI trade plan."""
    s_norm = (clean_upper_scalar(side) or "").strip()
    px = safe_pos_float(spot_px, default=None)
    inv = safe_pos_float(invalidation_px, default=None)
    tgt = safe_pos_float(target_px, default=None)
    if px is None or s_norm not in ("BUY", "SELL"):
        return "ACTIVE"
    if s_norm == "BUY":
        if inv is not None and px <= inv:
            return "INVALIDATED"
        if not is_resting and tgt is not None and px >= tgt:
            return "TARGET_HIT"
        return "ACTIVE"
    if inv is not None and px >= inv:
        return "INVALIDATED"
    if not is_resting and tgt is not None and px <= tgt:
        return "TARGET_HIT"
    return "ACTIVE"


def _extract_price_bar_ts_hi_lo(raw_row: Any) -> Optional[tuple[float, float, float, float, bool]]:
    """Parse canonical OHLC ``[ts, close, vol, high, low, ...]`` or ``[ts, px]`` into ``(ts_s, close, hi, lo, has_hl)``."""
    if not isinstance(raw_row, (list, tuple)) or len(raw_row) < 2:
        return None
    ts_s = _parse_pos_epoch_s(raw_row[0])
    c_v = safe_pos_float(raw_row[1], default=None)
    if ts_s is None or c_v is None:
        return None
    hi_v = safe_pos_float(raw_row[3], default=None) if len(raw_row) >= 5 else None
    lo_v = safe_pos_float(raw_row[4], default=None) if len(raw_row) >= 5 else None
    if hi_v is not None and lo_v is not None and hi_v >= lo_v > 0.0:
        return ts_s, c_v, max(hi_v, c_v), min(lo_v, c_v), True
    return ts_s, c_v, c_v, c_v, False


def resolve_symbol_price_bars(
    price_bars_by_symbol: Optional[Mapping[str, Sequence[Any]]],
    venue: Any,
    symbol: Any,
) -> Optional[Sequence[Any]]:
    """Resolve chronological price/OHLC bars for ``(venue, symbol)`` respecting asset class and quote compatibility."""
    if not isinstance(price_bars_by_symbol, Mapping) or not price_bars_by_symbol:
        return None
    sym = _clean_intent_symbol(symbol)
    if not sym:
        return None
    ven = normalize_venue_name(clean_scalar_str(venue) or "")
    is_eq = is_equity_venue_or_suffix(sym, ven)
    raw_cands = cache_symbol_variants(sym, quotes=compatible_quote_suffixes(sym))
    sym_cands = [sym, *(c for c in filter_candidates_by_asset_class(raw_cands, is_equity=is_eq) if c != sym)]
    keys: list[str] = []
    if ven:
        keys.extend(f"{ven}:{c}" for c in sym_cands)
    keys.extend(sym_cands)
    for key in keys:
        bars = casefold_get(price_bars_by_symbol, key)
        if isinstance(bars, Sequence) and not isinstance(bars, (str, bytes)) and bars:
            return bars
    return None


def evaluate_trade_plan_path_status(
    side: str,
    spot_px: Optional[float],
    invalidation_px: Optional[float],
    target_px: Optional[float],
    *,
    start_ts: float = 0.0,
    end_ts: Optional[float] = None,
    price_bars: Optional[Sequence[Any]] = None,
    is_resting: bool = False,
    check_spot: bool = True,
) -> str:
    """Evaluate chronological trade-plan status over ``(start_ts, end_ts]`` bars before checking live ``spot_px``.

    Prevents already-resolved plans from resurrecting as ``ACTIVE`` after price rebounds back into the
    ``[target_px, invalidation_px]`` band, while avoiding pre-fill wick contamination on bars straddling ``start_ts``.
    """
    s_norm = (clean_upper_scalar(side) or "").strip()
    inv = safe_pos_float(invalidation_px, default=None)
    tgt = safe_pos_float(target_px, default=None)
    if s_norm not in ("BUY", "SELL") or (inv is None and tgt is None):
        return "ACTIVE"
    if price_bars and not is_resting:
        parsed: list[tuple[float, float, float, float, bool]] = []
        for row in price_bars:
            if (bar := _extract_price_bar_ts_hi_lo(row)) is not None:
                parsed.append(bar)
        parsed.sort(key=lambda b: b[0])
        prev_ts: Optional[float] = None
        for bt, c_v, hi_v, lo_v, has_hl in parsed:
            bar_span = 0.0
            if has_hl:
                dt_prev = (bt - prev_ts) if (prev_ts is not None and bt > prev_ts) else 300.0
                bar_span = min(14400.0, max(60.0, dt_prev))
            prev_ts = bt
            if start_ts > 0.0 and bt <= start_ts:
                continue
            if end_ts is not None and end_ts > 0.0 and bt > end_ts:
                break
            use_hl = has_hl and (start_ts <= 0.0 or (bt - bar_span) >= start_ts)
            eff_hi = hi_v if use_hl else c_v
            eff_lo = lo_v if use_hl else c_v
            if s_norm == "BUY":
                if inv is not None and eff_lo <= inv:
                    return "INVALIDATED"
                if tgt is not None and eff_hi >= tgt:
                    return "TARGET_HIT"
            else:
                if inv is not None and eff_hi >= inv:
                    return "INVALIDATED"
                if tgt is not None and eff_lo <= tgt:
                    return "TARGET_HIT"
    if check_spot:
        return evaluate_trade_plan_status(s_norm, spot_px, inv, tgt, is_resting=is_resting)
    return "ACTIVE"


def summarize_active_ai_trade_plans(
    intents_data: list[dict[str, Any]],
    scope: str,
    now_ts: float,
    extract_notional_fn: Callable[..., Any],
    *,
    tradable_positions: Optional[list[dict[str, Any]]] = None,
    prices_and_trends: Optional[dict[str, Any]] = None,
    recent_orders: Optional[dict[str, Any]] = None,
    open_orders: Optional[list[dict[str, Any]]] = None,
    verified_open_venues: Optional[set[str]] = None,
    price_bars_by_symbol: Optional[Mapping[str, Sequence[Any]]] = None,
    max_age_days: float = 14.0,
) -> list[dict[str, Any]]:
    """Build compact persistent exit/invalidation memory for active executed/resting AI trade plans.

    Retains at most one latest active plan per ``(venue, symbol)``, clearing superseded plans on
    newer AI actions or exited spot positions so structured exit memory survives ``/compact``.
    """
    scoped_intents = _filter_intents_by_scope(intents_data, scope)
    if not scoped_intents:
        return []

    eff_now = resolve_now_ts(now_ts)
    max_age_sec = max(0.0, safe_pos_float(max_age_days, default=14.0) or 14.0) * 86400.0
    pos_list = tradable_positions if isinstance(tradable_positions, list) else []
    pt_map = prices_and_trends if isinstance(prices_and_trends, dict) else {}

    def _find_matching_position(ven: str, sym: str) -> Optional[dict[str, Any]]:
        for p in pos_list:
            if not isinstance(p, dict):
                continue
            if symbols_match_intent_scope_and_quote(p.get("symbol"), sym, venue_a=p.get("venue"), venue_b=ven):
                return p
        return None

    def _resolve_spot_px(ven: str, sym: str, pos_hit: Optional[dict[str, Any]]) -> Optional[float]:
        if isinstance(pos_hit, dict):
            if (p_px := safe_pos_float(pos_hit.get("price"), default=None)) is not None:
                return p_px
        if pt_map:
            for cand in (sym, *cache_symbol_variants(sym, include_base=True)):
                pinfo = casefold_get(pt_map, cand)
                if isinstance(pinfo, dict):
                    if (c_px := safe_pos_float(pinfo.get("price"), default=None)) is not None:
                        return c_px
        return None

    plans_rev: list[dict[str, Any]] = []
    seen_inst_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    seen_resting_keys: set[tuple[str, str, tuple[str, ...], str, str]] = set()
    _non_active_exec_statuses = (
        "CANCELED",
        "CANCELLED",
        "EXPIRED",
        "CANCELED_OR_EXPIRED",
        "FAILED",
        "REJECTED",
        "BLOCKED_BY_GUARD",
        "SUBMIT_UNKNOWN",
        "SKIPPED_DUPLICATE_OPEN_ORDER",
    )

    for i in reversed(scoped_intents):
        if not isinstance(i, dict):
            continue
        if (clean_upper_scalar(i.get("status")) or "") != "EXECUTED":
            continue
        det = _resolve_intent_execution_details(
            i,
            eff_now,
            extract_notional_fn,
            recent_orders=recent_orders,
            open_orders=open_orders,
            verified_open_venues=verified_open_venues,
        )
        side = det.get("side") or ""
        if side not in ("BUY", "SELL"):
            continue
        filled_q = safe_pos_float(det.get("filled_qty"), default=0.0) or 0.0
        if det.get("exec_status") in _non_active_exec_statuses and filled_q <= 0.0:
            continue
        venue = det.get("venue") or ""
        sym = det.get("symbol") or ""
        v_k, b_k, q_k, _ = _intent_venue_symbol_side_key(venue, sym, side)
        inst_key = (v_k, b_k, q_k)
        is_resting = _is_open_resting_detail(det)
        if not is_resting:
            if inst_key in seen_inst_keys:
                continue
            seen_inst_keys.add(inst_key)

        eff_ts = safe_pos_float(det.get("fill_ts_val"), default=None) or safe_pos_float(det.get("ts_val"), default=0.0) or 0.0
        if max_age_sec > 0.0 and eff_ts > 0.0 and (eff_now - eff_ts) > max_age_sec:
            continue

        entry_px = (
            safe_pos_float(det.get("fill_px"), default=None)
            or safe_pos_float(det.get("limit_px"), default=None)
            or safe_pos_float(det.get("ref_px"), default=None)
        )
        if is_resting:
            rest_oid = str(det.get("order_id") or "") or (f"{round(entry_px, 6)}" if entry_px is not None else "")
            rest_key = (v_k, b_k, q_k, side, rest_oid)
            if rest_key in seen_resting_keys:
                continue
            seen_resting_keys.add(rest_key)
        inv_px, tgt_px = extract_intent_trade_plan_levels(i, side=side, ref_price=entry_px)
        if inv_px is None and tgt_px is None:
            continue

        pos_hit = _find_matching_position(venue, sym)
        if side == "BUY" and not is_resting and isinstance(pos_hit, dict):
            held_usd = (safe_float(pos_hit.get("value_usd"), 0.0) or 0.0) + (
                safe_float(pos_hit.get("locked_base_usd"), 0.0) or 0.0
            )
            if held_usd < 5.0:
                continue
        if side == "SELL" and not is_resting and isinstance(pos_hit, dict):
            if "effective_max_buy_usd" in pos_hit or "effective_max_sell_usd" in pos_hit:
                mb_u = safe_pos_float(pos_hit.get("effective_max_buy_usd"), default=0.0) or 0.0
                rmb_u = safe_pos_float(pos_hit.get("replace_max_buy_usd"), default=0.0) or 0.0
                if mb_u <= 0.0 and rmb_u <= 0.0:
                    continue

        spot_px = _resolve_spot_px(venue, sym, pos_hit)
        sym_bars = resolve_symbol_price_bars(price_bars_by_symbol, venue, sym)
        status = evaluate_trade_plan_path_status(
            side,
            spot_px,
            inv_px,
            tgt_px,
            start_ts=eff_ts,
            price_bars=sym_bars,
            is_resting=is_resting,
        )
        if (
            side == "SELL"
            and not is_resting
            and status in ("TARGET_HIT", "INVALIDATED")
            and eff_ts > 0.0
            and (eff_now - eff_ts) > 86400.0
        ):
            continue
        ident = resolve_symbol_identity(sym)
        disp_sym = ident.display_symbol if (ident.is_tokenized_equity and ident.display_symbol) else sym
        is_limit_anchor = bool(is_resting and entry_px is not None and entry_px > 0.0)
        dist_anchor = (
            entry_px
            if is_limit_anchor
            else (spot_px if (spot_px is not None and spot_px > 0.0) else entry_px)
        )

        plan_rec: dict[str, Any] = {
            "venue": venue,
            "symbol": disp_sym,
            "side": side,
            "order_state": "RESTING" if is_resting else "FILLED",
            "status": status,
            "dist_anchor": "limit" if is_limit_anchor else "spot",
        }
        if is_resting and det.get("order_id"):
            plan_rec["order_id"] = str(det["order_id"])
        if entry_px is not None:
            plan_rec["entry_px"] = round(entry_px, 6)
        if spot_px is not None:
            plan_rec["spot_px"] = round(spot_px, 6)
        if tgt_px is not None:
            plan_rec["target_px"] = round(tgt_px, 6)
            if dist_anchor is not None and dist_anchor > 0.0:
                plan_rec["target_dist_pct"] = round(((tgt_px - dist_anchor) / dist_anchor) * 100.0, 2)
        if inv_px is not None:
            plan_rec["invalidation_px"] = round(inv_px, 6)
            if dist_anchor is not None and dist_anchor > 0.0:
                plan_rec["invalidation_dist_pct"] = round(((inv_px - dist_anchor) / dist_anchor) * 100.0, 2)
        if eff_ts > 0.0 and eff_now >= eff_ts:
            plan_rec["age_h"] = round((eff_now - eff_ts) / 3600.0, 1)
        plans_rev.append(plan_rec)

    return list(reversed(plans_rev))


__all__ = [
    "_attach_intent_age_fields",
    "_filter_intents_by_scope",
    "_is_open_resting_detail",
    "_resolve_executed_open_resting_detail",
    "_resolve_intent_execution_details",
    "build_last_cycle_execution_status",
    "evaluate_trade_plan_path_status",
    "evaluate_trade_plan_status",
    "resolve_symbol_price_bars",
    "summarize_active_ai_trade_plans",
    "summarize_recent_ai_intents",
]
