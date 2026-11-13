"""Pure deterministic AI trade intent cycle telemetry, fill resolution, and slippage evaluation."""
from __future__ import annotations

from typing import Any, Callable, Optional

from botcore import (
    _parse_bool_flag,
    _parse_epoch_or_iso_s,
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
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    compatible_quote_suffixes,
    filter_candidates_by_asset_class,
    is_equity_venue_or_suffix,
    matches_asset_scope,
    normalize_venue_name,
    same_canonical_base_asset,
)
from instrument_registry import casefold_get
from .base import MarketDataProvider
from .execution_audit import _TERMINAL_CANCEL_STATUSES
from .intent_state import (
    _clean_intent_symbol,
    _intent_venue_symbol_side_key,
    extract_intent_order_type,
    extract_intent_side,
    intent_venue_client_order_ids,
)
from .order_id_context import (
    extract_client_order_id,
    resolve_order_actor,
)
from .order_ohlc_utils import (
    _UNFILLED_ORDER_STATUSES,
    extract_executed_qty,
    extract_orig_qty,
)


def _parse_pos_epoch_s(raw: Any) -> Optional[float]:
    """Parse an epoch-seconds, epoch-ms, or ISO-8601 timestamp into positive UTC seconds."""
    parsed = _parse_epoch_or_iso_s(raw)
    return _ts_ms_to_s(parsed) if (parsed is not None and parsed > 0.0) else None


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
    ts_raw: Any = None
    ts_val = 0.0
    for _tk in ("timestamp", "executed_at", "executed_ts", "created_at", "ts"):
        _cand_raw = i.get(_tk)
        if _cand_raw is None:
            continue
        if ts_raw is None:
            ts_raw = _cand_raw
        if (_parsed_s := _parse_pos_epoch_s(_cand_raw)) is not None:
            ts_raw = _cand_raw
            ts_val = _parsed_s
            break
    order_id = extract_order_id(exec_pay, default=None) or extract_order_id(
        last_hist.get("order_id") or last_hist.get("payload"), default=None
    )

    raw_intent_st = clean_upper_scalar(i.get("status")) or ""
    is_eq_intent = is_equity_venue_or_suffix(symbol, venue)
    matched_open = None
    expected_cids: Optional[set[str]] = None
    if isinstance(open_orders, list) and raw_intent_st in (
        "EXECUTED",
        "FAILED",
        "PENDING",
        "PENDING_EXECUTION",
        "SUBMIT_UNKNOWN",
    ):
        for oo in open_orders:
            if not isinstance(oo, dict):
                continue
            oo_venue = normalize_venue_name(clean_scalar_str(oo.get("venue")) or "")
            if oo_venue and venue and oo_venue != venue:
                continue
            oo_sym = _clean_intent_symbol(oo.get("symbol"))
            oo_is_eq = is_equity_venue_or_suffix(oo_sym, oo_venue)
            if (oo_venue or oo_sym) and (venue or symbol) and oo_is_eq != is_eq_intent:
                continue
            if oo_sym and symbol and (
                not same_canonical_base_asset(oo_sym, symbol)
                or compatible_quote_suffixes(oo_sym) != compatible_quote_suffixes(symbol)
            ):
                continue
            oo_oid = extract_order_id(oo, default=None)
            if order_id is not None and oo_oid is not None and str(oo_oid) == str(order_id):
                matched_open = oo
                break
            if (
                order_id is None
                and order_type == "LIMIT"
                and oo_venue == venue
                and oo_sym
                and (extract_order_side(oo) or str(oo.get("side") or "").strip().upper()) == side
            ):
                # Response-lost intent: adopt only the order carrying this intent's own client ID
                # (venue-native form), never a BOT / monitororder-repriced order or another AI intent's order.
                if expected_cids is None:
                    expected_cids = intent_venue_client_order_ids(i)
                oo_cid = (extract_client_order_id(oo, default="") or "").lower()
                if (oo_cid in expected_cids) if oo_cid else (resolve_order_actor(oo) == "AI"):
                    matched_open = oo
                    break

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
    if (
        matched_open is None
        and raw_intent_st == "EXECUTED"
        and venue_st not in _TERMINAL_CANCEL_STATUSES
        and isinstance(recent_orders, dict)
        and symbol
    ):
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
        if isinstance(ro_entry, dict):
            side_key = "buy" if side == "BUY" else ("sell" if side == "SELL" else "")
            # Scan newest-first across last_<side> and recent_<side>s: a later BOT fill on the same side
            # (e.g. a monitororder-repriced tradeall order) must not hide this intent's own AI fill.
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
            if isinstance(last_fill, dict):
                if fill_px is None or (venue_st in _UNFILLED_ORDER_STATUSES and filled_qty <= 0):
                    fill_px = safe_pos_float(last_fill.get("price"), default=fill_px)
                if filled_qty <= 0:
                    filled_qty = safe_pos_float(last_fill.get("qty"), default=0.0) or 0.0
                matched_fill_ts_ms = normalize_timestamp_ms(last_fill.get("ts_ms"), default=0) or 0
            else:
                matched_fill_ts_ms = 0
        else:
            matched_fill_ts_ms = 0
    else:
        matched_fill_ts_ms = 0
    is_unfilled_resting_limit = (
        order_type == "LIMIT"
        and (venue_st in _UNFILLED_ORDER_STATUSES or not venue_st)
        and filled_qty <= 0
    )
    is_unfilled_accepted_market = (
        order_type != "LIMIT"
        and _parse_bool_flag(exec_pay.get("reconciled_from_audit"), False)
        and (venue_st in _UNFILLED_ORDER_STATUSES or not venue_st)
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
    ):
        cancel_grace_sec = MarketDataProvider.resolve_unfilled_order_cancel_grace_sec()
        if (resolve_now_ts(now_ts) - ts_val) > cancel_grace_sec:
            venue_st = "CANCELED_OR_EXPIRED"
    if (
        fill_px is None
        and raw_intent_st in ("EXECUTED", "OBSERVED_SHADOW")
        and not is_unfilled_resting_limit
        and not is_unfilled_accepted_market
    ):
        fill_px = first_pos_float(
            last_hist.get("price"),
            i.get("requested_price"),
            i.get("limit_price"),
            i.get("reference_price"),
        )

    if matched_open is not None:
        exec_status = "PARTIALLY_FILLED" if filled_qty > 0 else "OPEN_RESTING"
        if exec_status == "PARTIALLY_FILLED" and venue_st in _UNFILLED_ORDER_STATUSES:
            venue_st = "PARTIALLY_FILLED"
    elif raw_intent_st == "EXECUTED":
        if venue_st in _TERMINAL_CANCEL_STATUSES:
            exec_status = venue_st
        elif is_unfilled_resting_limit:
            exec_status = "OPEN_RESTING"
        elif is_unfilled_accepted_market:
            exec_status = "ACCEPTED"
        elif venue_st == "PARTIALLY_FILLED":
            exec_status = "PARTIALLY_FILLED"
        else:
            exec_status = "FILLED"
            if venue_st in _UNFILLED_ORDER_STATUSES:
                venue_st = "FILLED"
    elif raw_intent_st == "OBSERVED_SHADOW":
        exec_status = "SHADOW_SIMULATED"
    else:
        exec_status = raw_intent_st or "UNKNOWN"

    ref_px = first_pos_float(
        i.get("reference_price"),
        i.get("limit_price"),
        i.get("requested_price"),
        last_hist.get("price"),
        fill_px,
    )
    if exec_status in ("OPEN_RESTING", "ACCEPTED", *_TERMINAL_CANCEL_STATUSES) and filled_qty <= 0:
        fill_px = None
    slippage_bps: Optional[float] = None
    if (
        exec_status in ("FILLED", "PARTIALLY_FILLED", "SHADOW_SIMULATED")
        and fill_px is not None
        and fill_px > 0
        and ref_px is not None
        and ref_px > 0
    ):
        raw_slip = (
            ((fill_px - ref_px) / ref_px) * 10_000.0
            if side == "BUY"
            else ((ref_px - fill_px) / ref_px) * 10_000.0
        )
        slippage_bps = round(raw_slip, 2)
    elif exec_status in ("FILLED", "SHADOW_SIMULATED") and fill_px is not None:
        slippage_bps = 0.0

    eff_qty = (
        filled_qty
        if filled_qty > 0
        else (orig_qty if (orig_qty is not None and exec_status in ("FILLED", "SHADOW_SIMULATED")) else None)
    )
    limit_px = (
        first_pos_float(
            i.get("limit_price"),
            matched_open.get("price") if isinstance(matched_open, dict) else None,
        )
        if order_type == "LIMIT"
        else None
    )
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
                    i.get("guard_reason"),
                    i.get("error"),
                    exec_pay.get("reason") if isinstance(exec_pay, dict) else None,
                    exec_pay.get("error") if isinstance(exec_pay, dict) else None,
                )
                if (s := clean_scalar_str(cand)) is not None
            ),
            None,
        )
        if raw_fail_reason:
            fail_reason = " ".join(raw_fail_reason.split())[:200] or None
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
        if det["venue_status"]:
            intent_summary["venue_status"] = det["venue_status"]
        for k in ("order_id", "fill_px", "filled_qty", "filled_notional_usd", "slippage_bps"):
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
            if has_fill and ep_st not in _TERMINAL_CANCEL_STATUSES:
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
        for k in ("filled_notional_usd", "limit_px", "order_id"):
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


__all__ = [
    "_attach_intent_age_fields",
    "_filter_intents_by_scope",
    "_is_open_resting_detail",
    "_resolve_executed_open_resting_detail",
    "_resolve_intent_execution_details",
    "build_last_cycle_execution_status",
    "summarize_recent_ai_intents",
]
