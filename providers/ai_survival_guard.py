"""Pure deterministic AI survival guards: anti-ping-pong flip protection and recent decision scorecard.

Single Source of Truth (SSOT) for AI-specific failure modes (whipsaw ping-pong reversals and
recent-decision loss streaks). Operates strictly on durable intent history, live spot prices,
and order_guard.conf thresholds in 0 ms with zero network calls or extra disk state.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Callable, Mapping, Optional, Sequence

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
    clean_scale,
    clean_upper_scalar,
    extract_current_price_item,
    extract_order_price,
    first_pos_float,
    fmt_signed,
    normalize_side,
    opposite_side,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    matches_asset_scope,
    normalize_venue_name,
)
from instrument_registry import canonical_display_symbol, casefold_get, resolve_symbol_identity
from .ai_circuit_breaker import (
    DEFAULT_FLIP_COOLDOWN_MIN,
    DEFAULT_FLIP_MIN_EDGE_PCT,
    DEFAULT_MAX_FLIPS_6H,
    DEFAULT_MIN_TARGET_RR,
    DEFAULT_SCORECARD_COOLDOWN_HOURS,
    DEFAULT_SCORECARD_MAX_LOSS_STREAK,
    DEFAULT_SCORECARD_WINDOW_TRADES,
    DEFAULT_TARGET_EV_FEE_MULT,
    DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
    DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
    DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
)
from .intent_state import (
    _extract_intent_id,
    _intent_executed_notional_usd,
    _intent_venue_symbol_side_key,
    collect_canceled_order_ids,
    extract_intent_horizon_hours,
    extract_intent_order_type,
    extract_intent_primary_order_id,
    extract_intent_side,
    extract_intent_timestamp_s,
    extract_intent_trade_plan_levels,
    parse_pos_epoch_s,
    symbols_match_intent_scope_and_quote,
)
from .intent_telemetry import (
    TERMINAL_TRADE_PLAN_STATUSES,
    _compute_signed_exec_bps,
    evaluate_trade_plan_path_status,
    evaluate_trade_plan_status,
    extract_frozen_trade_plan_status,
    resolve_symbol_price_bars,
)
from .order_ohlc_utils import extract_executed_qty
from .order_status_ssot import (
    ACCEPTED_WORKING_VENUE_STATUSES,
    NON_FILL_ORDER_STATUSES,
    TERMINAL_CANCEL_VENUE_STATUSES,
)
from .venue_metadata import ProviderVenueMetadataMixin

_FILLED_VENUE_STATUSES: frozenset[str] = frozenset({"FILLED", "CLOSED", "PARTIALLY_FILLED"})
_WIN_EDGE_PCT = 0.20
_LOSS_EDGE_PCT = -0.35
_OPEN_MTM_LOSS_EDGE_PCT = -1.50
_EMERGENCY_SELL_EDGE_MULT = 2.5
_MAX_BOT_FLIP_COOLDOWN_SEC = 1800.0
_WHIPSAW_WINDOW_SEC = 21600.0
_REFERENCE_RT_FEE_BPS = 20.0
_MIN_FLIP_EDGE_FLOOR_PCT = 0.15
_MIN_TARGET_EV_FLOOR_PCT = 0.10
_RESTING_UNFILLED_GRACE_SEC = 3600.0
_LIFO_REMAINDER_DUST_USD = 2.0
_LIFO_REMAINDER_DUST_MAX_FRAC = 0.05
DEFAULT_LIFO_MAX_PAIR_HOLD_SEC = 7.0 * 86400.0


def _is_terminal_or_stale_unfilled_intent(
    it: Mapping[str, Any],
    canceled_oids: set[str],
    *,
    now_ts: float,
    ref_ts: float,
    resting_grace_sec: float = _RESTING_UNFILLED_GRACE_SEC,
) -> bool:
    """Return True when an unfilled EXECUTED intent is canceled/expired or aged past active resting grace."""
    prim_oid = extract_intent_primary_order_id(it, default="", include_top_level=True)
    if prim_oid and str(prim_oid) in canceled_oids:
        return True
    ep = it.get("execution_payload") if isinstance(it.get("execution_payload"), Mapping) else {}
    ep_st = (clean_upper_scalar(ep.get("status")) if ep else "") or ""
    ep_msg = (clean_scalar_str(ep.get("msg")) or "").lower() if ep else ""
    if ep_st in TERMINAL_CANCEL_VENUE_STATUSES:
        return True
    if ep_st in ACCEPTED_WORKING_VENUE_STATUSES or ep_msg == "resting" or not ep_st:
        grace = max(0.0, safe_nonneg_float(resting_grace_sec, default=_RESTING_UNFILLED_GRACE_SEC))
        return (now_ts - ref_ts) >= grace
    return True


def _collect_canceled_order_ids(items: list[dict[str, Any]], exclude_iid: str = "") -> set[str]:
    return collect_canceled_order_ids(items, exclude_iid=exclude_iid)


def _has_explicit_fill_marker(it: dict[str, Any], ep: dict[str, Any], ep_st: str, ep_msg: str) -> bool:
    if ep_st in _FILLED_VENUE_STATUSES or ep_msg.startswith("filled"):
        return True
    for key in ("filled_at", "filled_ts"):
        if parse_pos_epoch_s(ep.get(key)) is not None:
            return True
    for key in ("avg_fill_price", "avgPrice", "fill_price"):
        if safe_pos_float(ep.get(key), default=None) is not None:
            return True
    return (
        parse_pos_epoch_s(it.get("filled_ts")) is not None
        or parse_pos_epoch_s(it.get("filled_at")) is not None
        or safe_pos_float(it.get("fill_px"), default=None) is not None
    )


def _is_filled_ai_intent(it: dict[str, Any], canceled_oids: set[str]) -> bool:
    """Return True when an executed AI intent actually filled (excludes canceled/expired/unfilled resting orders)."""
    if (clean_upper_scalar(it.get("status")) or "") != "EXECUTED":
        return False
    ep = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
    filled_q = extract_executed_qty(ep, default=0.0) if ep else 0.0
    if filled_q <= 0.0:
        filled_q = safe_nonneg_float(it.get("filled_qty"), default=0.0)
    if filled_q > 0.0:
        return True
    ep_st = (clean_upper_scalar(ep.get("status")) if ep else "") or ""
    ep_msg = (clean_scalar_str(ep.get("msg")) or "").lower() if ep else ""
    if ep_st in NON_FILL_ORDER_STATUSES or ep_msg == "resting":
        return False
    prim_oid = extract_intent_primary_order_id(it, default="", include_top_level=True)
    if prim_oid and str(prim_oid) in canceled_oids:
        return False
    if extract_intent_order_type(it) == "LIMIT":
        return _has_explicit_fill_marker(it, ep, ep_st, ep_msg)
    if _parse_bool_flag(ep.get("reconciled_from_audit"), False) and ep_st not in _FILLED_VENUE_STATUSES:
        return False
    return True


def _extract_intent_exec_price(it: dict[str, Any]) -> float:
    ep = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
    hist = it.get("history") if isinstance(it.get("history"), list) else []
    last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    ep_px = extract_order_price(ep, default=None) if ep else None
    px = first_pos_float(
        ep_px,
        ep.get("avg_fill_price"),
        ep.get("avgPrice"),
        ep.get("fill_price"),
        it.get("fill_px"),
        last_h.get("price"),
        it.get("limit_price"),
        it.get("requested_price"),
        it.get("reference_price"),
    )
    return px if px is not None else 0.0


def _extract_intent_exec_ts(it: dict[str, Any], *, max_ts: Optional[float] = None) -> float:
    _, raw_base_ts = extract_intent_timestamp_s(it)
    base_ts = raw_base_ts if (max_ts is None or raw_base_ts <= max_ts) else 0.0
    fill_ts = base_ts
    for key in ("filled_ts", "filled_at"):
        if (
            (ts_v := parse_pos_epoch_s(it.get(key))) is not None
            and ts_v >= base_ts
            and (max_ts is None or ts_v <= max_ts)
        ):
            fill_ts = max(fill_ts, ts_v)
            break
    ep = it.get("execution_payload")
    if isinstance(ep, dict):
        for key in ("filled_at", "filled_ts", "terminal_ts", "updated_ts", "updateTime", "workingTime", "transactTime"):
            if (
                (ts_v := parse_pos_epoch_s(ep.get(key))) is not None
                and ts_v >= base_ts
                and (max_ts is None or ts_v <= max_ts)
            ):
                fill_ts = max(fill_ts, ts_v)
                break
    if fill_ts <= base_ts:
        hist = it.get("history") if isinstance(it.get("history"), list) else []
        last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
        if str(last_h.get("action") or "") in ("RECONCILED_EXECUTED", "RECONCILED_PARTIAL"):
            if (
                (lh_ts := parse_pos_epoch_s(last_h.get("ts"))) is not None
                and lh_ts > fill_ts
                and (max_ts is None or lh_ts <= max_ts)
            ):
                fill_ts = lh_ts
    if fill_ts <= base_ts:
        for key in ("executed_at", "executed_ts"):
            if (
                (ts_v := parse_pos_epoch_s(it.get(key))) is not None
                and ts_v > fill_ts
                and (max_ts is None or ts_v <= max_ts)
            ):
                fill_ts = ts_v
                break
    return fill_ts


def _extract_intent_notional_usd(it: dict[str, Any]) -> float:
    return _intent_executed_notional_usd(it)


def compute_signed_price_edge_pct(side: str, entry_px: float, exit_px: float) -> float:
    """Return directional gross percentage price move from ``entry_px`` to ``exit_px`` (+ favorable)."""
    s_norm = normalize_side(side)
    ep = safe_pos_float(entry_px, default=None)
    xp = safe_pos_float(exit_px, default=None)
    if ep is None or xp is None or s_norm not in ("BUY", "SELL"):
        return 0.0
    return (
        ((xp - ep) / ep) * 100.0
        if s_norm == "BUY"
        else ((ep - xp) / ep) * 100.0
    )


def resolve_matched_notional_usd(entry_usd: float, exit_usd: float) -> float:
    """Return matched round-trip USD notional (``min`` when both positive, else ``max``)."""
    e_u = safe_nonneg_float(entry_usd, default=0.0)
    x_u = safe_nonneg_float(exit_usd, default=0.0)
    return min(e_u, x_u) if (e_u > 0.0 and x_u > 0.0) else max(e_u, x_u)


def _extract_lifo_row_qty(row: dict[str, Any]) -> tuple[float, bool]:
    """Return ``(positive_qty, has_explicit_qty)`` for a filled LIFO lot row."""
    if (q_exp := safe_pos_float(row.get("qty"), default=None)) is not None:
        return q_exp, True
    if (q_fill := safe_pos_float(row.get("filled_qty"), default=None)) is not None:
        return q_fill, True
    px = safe_pos_float(row.get("px") if row.get("px") is not None else row.get("price"), default=None)
    usd = safe_pos_float(row.get("usd") if row.get("usd") is not None else row.get("cost_usd"), default=None)
    if px is not None and usd is not None:
        return usd / px, False
    return 0.0, False


def _slice_lifo_row(row: dict[str, Any], matched_qty: float, total_qty: float) -> dict[str, Any]:
    """Return a proportional slice of ``row`` carrying ``matched_qty`` and scaled ``usd``/``cost_usd``."""
    sliced = dict(row)
    if total_qty <= 0.0 or matched_qty >= total_qty - 1e-12:
        if total_qty > 0.0:
            sliced["qty"] = total_qty
        return sliced
    ratio = matched_qty / total_qty
    sliced["qty"] = matched_qty
    if (usd_v := safe_pos_float(row.get("usd"), default=None)) is not None:
        sliced["usd"] = usd_v * ratio
    if (cost_v := safe_pos_float(row.get("cost_usd"), default=None)) is not None:
        sliced["cost_usd"] = cost_v * ratio
    return sliced


def match_filled_intent_rows_lifo(
    filled_rows: Sequence[dict[str, Any]],
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Chronologically match filled rows LIFO per ``inst_key``, consuming executed qty and retaining lot remainders."""
    sorted_rows = sorted(filled_rows, key=lambda r: r["ts"])
    # Each stack item is [row_dict, remaining_qty, original_qty, has_explicit_qty].
    pending_buys: dict[Any, list[list[Any]]] = {}
    reloaded_buys: dict[Any, list[list[Any]]] = {}
    pending_sells: dict[Any, list[list[Any]]] = {}
    closed_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pair_idx_by_src: dict[tuple[str, int, int], tuple[int, float, float, float, float]] = {}

    def _record_closed_pair(
        pair_kind: str,
        lot_row: dict[str, Any],
        lot_matched_q: float,
        lot_orig: float,
        incoming: dict[str, Any],
        in_matched_q: float,
        in_orig: float,
    ) -> None:
        # Coalesce split slices of the exact same entry/exit fills (e.g. a reload BUY split across
        # pending_buys + reloaded_buys and later closed by a single SELL) into one closed trade.
        src_key = (pair_kind, id(lot_row), id(incoming))
        if (prev := pair_idx_by_src.get(src_key)) is not None:
            idx, prev_lot_q, prev_lot_orig, prev_in_q, prev_in_orig = prev
            new_lot_q = min(prev_lot_orig, prev_lot_q + lot_matched_q)
            new_in_q = min(prev_in_orig, prev_in_q + in_matched_q)
            closed_pairs[idx] = (
                pair_kind,
                _slice_lifo_row(lot_row, new_lot_q, prev_lot_orig),
                _slice_lifo_row(incoming, new_in_q, prev_in_orig),
            )
            pair_idx_by_src[src_key] = (idx, new_lot_q, prev_lot_orig, new_in_q, prev_in_orig)
            return
        pair_idx_by_src[src_key] = (len(closed_pairs), lot_matched_q, lot_orig, in_matched_q, in_orig)
        closed_pairs.append(
            (pair_kind, _slice_lifo_row(lot_row, lot_matched_q, lot_orig), _slice_lifo_row(incoming, in_matched_q, in_orig))
        )

    def _dust_eps(orig_a: float, orig_b: float, explicit_a: bool, explicit_b: bool) -> float:
        if explicit_a and explicit_b:
            return 1e-9
        return max(1e-9, 0.01 * max(orig_a, orig_b, 1e-9))

    def _is_dust_remainder(rem_q: float, orig_q: float, row_obj: Mapping[str, Any], eps: float) -> bool:
        if rem_q <= eps:
            return True
        if orig_q > 0.0 and (rem_q / orig_q) <= _LIFO_REMAINDER_DUST_MAX_FRAC:
            px_v = safe_pos_float(
                row_obj.get("px") if row_obj.get("px") is not None else row_obj.get("price"),
                default=0.0,
            ) or 0.0
            if px_v > 0.0:
                rem_usd = rem_q * px_v
            else:
                orig_usd = safe_pos_float(
                    row_obj.get("usd") if row_obj.get("usd") is not None else row_obj.get("cost_usd"),
                    default=0.0,
                ) or 0.0
                rem_usd = orig_usd * (rem_q / orig_q) if orig_usd > 0.0 else 0.0
            if 0.0 < rem_usd < _LIFO_REMAINDER_DUST_USD:
                return True
        return False

    def _consume_stack(
        stack: list[list[Any]],
        incoming: dict[str, Any],
        rem_qty: float,
        orig_in_qty: float,
        in_explicit: bool,
        pair_kind: str,
    ) -> float:
        while stack and rem_qty > 1e-12:
            top = stack[-1]
            lot_row, lot_rem, lot_orig, lot_exp = top[0], top[1], top[2], top[3]
            if lot_orig <= 0.0 or orig_in_qty <= 0.0:
                stack.pop()
                closed_pairs.append((pair_kind, dict(lot_row), dict(incoming)))
                return 0.0
            if (not lot_exp) and (not in_explicit):
                lot_px = safe_pos_float(
                    lot_row.get("px") if lot_row.get("px") is not None else lot_row.get("price"),
                    default=0.0,
                ) or 0.0
                in_px = safe_pos_float(
                    incoming.get("px") if incoming.get("px") is not None else incoming.get("price"),
                    default=0.0,
                ) or 0.0
                lot_rem_usd = lot_rem * lot_px
                in_rem_usd = rem_qty * in_px
                max_u = max(lot_rem_usd, in_rem_usd)
                if max_u > 0.0 and abs(lot_rem_usd - in_rem_usd) / max_u <= 0.05:
                    stack.pop()
                    _record_closed_pair(pair_kind, lot_row, lot_rem, lot_orig, incoming, rem_qty, orig_in_qty)
                    return 0.0
            eps = _dust_eps(lot_orig, orig_in_qty, lot_exp, in_explicit)
            matched_q = min(lot_rem, rem_qty)
            _record_closed_pair(pair_kind, lot_row, matched_q, lot_orig, incoming, matched_q, orig_in_qty)
            lot_rem -= matched_q
            rem_qty -= matched_q
            if _is_dust_remainder(lot_rem, lot_orig, lot_row, eps):
                stack.pop()
            else:
                top[1] = lot_rem
            if _is_dust_remainder(rem_qty, orig_in_qty, incoming, eps):
                rem_qty = 0.0
                break
        return rem_qty

    for row in sorted_rows:
        side = normalize_side(row.get("side"))
        if side not in ("BUY", "SELL"):
            continue
        ikey = row["inst_key"]
        row_qty, row_exp = _extract_lifo_row_qty(row)
        if side == "BUY":
            s_queue = pending_sells.get(ikey)
            if s_queue:
                def _stack_rem_usd(queue: list[list[Any]]) -> float:
                    return sum(
                        (safe_pos_float(it[0].get("usd"), default=0.0) or 0.0) * (it[1] / it[2])
                        for it in queue
                        if it[2] > 0.0
                    )

                rem_sell_usd_before = _stack_rem_usd(s_queue)
                in_usd = safe_pos_float(row.get("usd"), default=0.0) or 0.0
                rem_after = _consume_stack(s_queue, row, row_qty, row_qty, row_exp, "SELL->BUY")
                consumed_sell_usd = max(0.0, rem_sell_usd_before - _stack_rem_usd(s_queue))
                if rem_after > 0.0 and _is_dust_remainder(rem_after, row_qty, row, 1e-12):
                    rem_after = 0.0
                # A spot reload BUY closes the SELL->BUY leg and opens live long inventory in
                # reloaded_buys without duplicating open BUY_MTM while still held.
                is_notional_reload = consumed_sell_usd > 0.0 and in_usd <= consumed_sell_usd * 1.05
                if rem_after <= 1e-12 or is_notional_reload:
                    reloaded_buys.setdefault(ikey, []).append([row, row_qty, row_qty, row_exp])
                else:
                    reloaded_buys.setdefault(ikey, []).append([row, row_qty - rem_after, row_qty, row_exp])
                    pending_buys.setdefault(ikey, []).append([row, rem_after, row_qty, row_exp])
            else:
                pending_buys.setdefault(ikey, []).append([row, row_qty, row_qty, row_exp])
        else:
            rem_sell = row_qty
            b_queue = pending_buys.get(ikey)
            if b_queue:
                rem_sell = _consume_stack(b_queue, row, rem_sell, row_qty, row_exp, "BUY->SELL")
            if rem_sell > 1e-12 and (rb_queue := reloaded_buys.get(ikey)):
                rem_sell = _consume_stack(rb_queue, row, rem_sell, row_qty, row_exp, "BUY->SELL")
            if rem_sell > 0.0 and _is_dust_remainder(rem_sell, row_qty, row, 1e-12):
                rem_sell = 0.0
            if rem_sell > 1e-12 or (row_qty <= 0.0 and not b_queue and not reloaded_buys.get(ikey)):
                pending_sells.setdefault(ikey, []).append([row, rem_sell, row_qty, row_exp])

    open_buys = [_slice_lifo_row(it[0], it[1], it[2]) for q in pending_buys.values() for it in q]
    open_sells = [_slice_lifo_row(it[0], it[1], it[2]) for q in pending_sells.values() for it in q]
    return closed_pairs, open_buys, open_sells


def match_filled_intent_rows_lifo_in_window(
    filled_rows: Sequence[dict[str, Any]],
    *,
    cutoff_ts: float = 0.0,
    max_pair_hold_sec: Optional[float] = DEFAULT_LIFO_MAX_PAIR_HOLD_SEC,
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Match a bounded-hydration chronological fill stream LIFO first, then filter outcomes to ``>= cutoff_ts``.

    - Hydrating from ``cutoff_ts - max_pair_hold_sec`` preserves genuine cross-boundary entries (preventing
      in-window stop-loss SELLs from inverting into phantom ``SELL->BUY`` wins) while preventing weeks-old
      ghost AI BUYs (already closed on a shared wallet by BOT/MANUAL actors) from poisoning recent PnL.
    """
    eff_max_hold = (
        safe_pos_float(max_pair_hold_sec, default=None)
        if max_pair_hold_sec is not None
        else None
    )
    if cutoff_ts > 0.0 and eff_max_hold is not None:
        hydrate_cutoff = cutoff_ts - eff_max_hold
        lifo_stream = [
            row for row in filled_rows
            if (safe_float(row.get("ts"), default=0.0) or 0.0) >= hydrate_cutoff
        ]
    else:
        lifo_stream = list(filled_rows)
    closed_pairs, open_buys, open_sells = match_filled_intent_rows_lifo(lifo_stream)
    if cutoff_ts <= 0.0:
        return closed_pairs, open_buys, open_sells
    win_closed = [
        pair for pair in closed_pairs
        if (ex_ts := (safe_float(pair[2].get("ts"), default=0.0) or 0.0)) >= cutoff_ts
        and (
            eff_max_hold is None
            or (ex_ts - (safe_float(pair[1].get("ts"), default=0.0) or 0.0)) <= eff_max_hold
        )
    ]
    win_open_buys = [
        row for row in open_buys
        if (safe_float(row.get("ts"), default=0.0) or 0.0) >= cutoff_ts
    ]
    win_open_sells = [
        row for row in open_sells
        if (safe_float(row.get("ts"), default=0.0) or 0.0) >= cutoff_ts
    ]
    return win_closed, win_open_buys, win_open_sells


def build_normalized_filled_ai_intent_row(
    it: Mapping[str, Any],
    *,
    venue: str,
    symbol: str,
    side: str,
    ts_s: float,
    px: float,
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
    include_slippage: bool = False,
    include_attribution: bool = False,
    model_used: str = "",
    override_codes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the canonical normalized filled-intent row shared by scorecard, calibration, and attribution."""
    ot = extract_intent_order_type(it)
    inv_px, tgt_px = extract_intent_trade_plan_levels(it, side=side, ref_price=px)
    live_rt = fee_resolver(venue, symbol, ot) if callable(fee_resolver) else None
    rt_bps = resolve_venue_round_trip_fee_bps(venue, symbol, order_type=ot, round_trip_fee_bps=live_rt)
    v_k, b_k, q_k, _ = _intent_venue_symbol_side_key(venue, symbol, side)
    exec_payload = it.get("execution_payload")
    exec_qty = (
        safe_pos_float(extract_executed_qty(exec_payload, default=0.0), default=None)
        if isinstance(exec_payload, Mapping)
        else None
    )
    row: dict[str, Any] = {
        "intent_id": _extract_intent_id(it),
        "ts": ts_s,
        "venue": venue,
        "symbol": symbol,
        "inst_key": (v_k, b_k, q_k),
        "side": side,
        "order_type": ot,
        "px": px,
        "qty": exec_qty,
        "usd": _extract_intent_notional_usd(it),
        "leg_fee_bps": rt_bps / 2.0,
        "rt_fee_bps": rt_bps,
        "invalidation_px": inv_px,
        "target_px": tgt_px,
        "horizon_h": extract_intent_horizon_hours(it),
        "confidence": clean_scale(it.get("confidence"), default=None, allow_percent=True) if it.get("confidence") is not None else None,
        "has_plan": _is_actionable_filled_trade_plan(venue, symbol, side, inv_px, tgt_px),
        "resolved_plan_status": extract_frozen_trade_plan_status(it),
    }
    if include_slippage:
        lim_slip, arr_sf, mkt_slip = extract_fill_execution_slippage(it, side=side, exec_px=px, order_type=ot)
        row["limit_slip_bps"] = lim_slip
        row["arrival_sf_bps"] = arr_sf
        row["market_slip_bps"] = mkt_slip
        row["limit_passive_offset_bps"] = extract_limit_passive_offset_bps(it, side=side, order_type=ot)
    if include_attribution:
        row["intent_id"] = _extract_intent_id(it)
        row["order_id"] = extract_intent_primary_order_id(it, default="", include_top_level=True)
        row["model_used"] = model_used or (clean_scalar_str(it.get("model_used")) or "unknown")
        row["override_codes"] = list(override_codes)
    return row


def format_signed_usd(val: Any, *, decimals: int = 2) -> str:
    """Format a signed USD value as ``+$X.XX`` / ``-$X.XX`` (or integer dollars when ``decimals=0``)."""
    v = safe_float(val, default=0.0) or 0.0
    prec = max(0, safe_nonneg_int(decimals, default=2))
    return f"+${abs(v):.{prec}f}" if v >= 0.0 else f"-${abs(v):.{prec}f}"


def extract_live_prices_from_cache_payload(raw_data: Any) -> dict[str, float]:
    """Parse ``cache_current_price.json`` payload items into a clean ``symbol -> positive_price`` map."""
    items = raw_data.get("items") if isinstance(raw_data, Mapping) else None
    prices: dict[str, float] = {}
    if isinstance(items, Mapping):
        for k, v in items.items():
            px = safe_pos_float(extract_current_price_item(v), default=None)
            if px is None and isinstance(v, Mapping):
                px = safe_pos_float(v.get("price"), default=None)
            if px is not None:
                prices[str(k)] = px
    return prices


def build_intent_live_price_map(
    prices_and_trends: Any,
    tradable_positions: Any = None,
) -> dict[str, Any]:
    """Build a unified symbol / ``venue:symbol`` live price lookup map for MTM scoring and calibration."""
    sc_prices: dict[str, Any] = dict(prices_and_trends) if isinstance(prices_and_trends, Mapping) else {}
    for tp in (tradable_positions if isinstance(tradable_positions, Sequence) and not isinstance(tradable_positions, (str, bytes)) else ()):
        if (
            isinstance(tp, Mapping)
            and tp.get("symbol")
            and (tp_px := safe_pos_float(tp.get("price"), default=None)) is not None
        ):
            tp_sym = str(tp["symbol"])
            sc_prices.setdefault(tp_sym, tp_px)
            if tp.get("venue"):
                sc_prices[f"{tp['venue']}:{tp_sym}"] = tp_px
    return sc_prices


def _resolve_live_price_for_symbol(
    live_prices: Mapping[str, Any],
    venue: str,
    symbol: str,
) -> Optional[float]:
    if not isinstance(live_prices, Mapping) or not live_prices:
        return None
    ven = normalize_venue_name(clean_scalar_str(venue) or "")
    sym = clean_scalar_str(symbol) or ""
    if not sym:
        return None
    cands: list[str] = []
    if ven:
        cands.append(f"{ven}:{sym}")
    cands.append(sym)
    for var in cache_symbol_variants(sym, include_base=True):
        if ven:
            cands.append(f"{ven}:{var}")
        cands.append(var)
    for key in cands:
        raw = casefold_get(live_prices, key)
        if isinstance(raw, Mapping):
            px = safe_pos_float(raw.get("price"), default=None)
        else:
            px = safe_pos_float(raw, default=None)
        if px is not None:
            return px
    return None


def resolve_venue_round_trip_fee_bps(
    venue: Any,
    symbol: Any = "",
    *,
    order_type: Any = "LIMIT",
    round_trip_fee_bps: Optional[float] = None,
) -> float:
    """Resolve declarative round-trip (2x leg) fee in basis points via venue metadata SSOT (0 ms, no I/O)."""
    if round_trip_fee_bps is not None:
        return round(safe_nonneg_float(round_trip_fee_bps, default=0.0), 4)
    ven = normalize_venue_name(clean_scalar_str(venue) or "")
    sym = clean_scalar_str(symbol) or ""
    if ProviderVenueMetadataMixin.supports_tokenized_assets(ven) and sym and resolve_symbol_identity(sym).is_tokenized_equity:
        return 0.0
    _, maker_bps, taker_bps, _ = ProviderVenueMetadataMixin.resolve_venue_default_fee_spec(ven)
    ot = clean_upper_scalar(order_type) or "LIMIT"
    leg_bps = safe_nonneg_float(taker_bps if ot == "MARKET" else maker_bps, default=10.0)
    return round(2.0 * leg_bps, 4)


def build_persisted_round_trip_fee_resolver(
    cache_dir: Optional[str],
    *,
    now_ts: Optional[float] = None,
) -> Optional[Callable[[str, str, str], Optional[float]]]:
    """Build a memoized ``(venue, symbol, order_type) -> round_trip_fee_bps`` resolver from persisted fee state."""
    cdir = clean_scalar_str(cache_dir)
    if not cdir:
        return None
    from .venue_fee_maintenance import read_persisted_venue_fee_spec

    eff_now = resolve_now_ts(now_ts)
    memo: dict[tuple[str, str, str], Optional[float]] = {}

    def _resolve(venue: str, symbol: str, order_type: str = "LIMIT") -> Optional[float]:
        ven = normalize_venue_name(clean_scalar_str(venue) or "")
        sym = canonical_display_symbol(symbol, venue=ven or None) or clean_scalar_str(symbol) or ""
        ot = clean_upper_scalar(order_type) or "LIMIT"
        key = (ven, sym, ot)
        if key in memo:
            return memo[key]
        res: Optional[float] = None
        try:
            spec = read_persisted_venue_fee_spec(ven, sym, cache_dir=cdir, now_ts=eff_now)
            if isinstance(spec, Mapping) and spec:
                leg_keys = ("taker_fee_bps", "taker_bps") if ot == "MARKET" else ("maker_fee_bps", "maker_bps")
                leg_bps = next(
                    (v for lk in leg_keys if (v := safe_nonneg_float(spec.get(lk), default=None)) is not None),
                    None,
                )
                if leg_bps is not None:
                    res = round(2.0 * leg_bps, 4)
        except Exception:
            res = None
        memo[key] = res
        return res

    return _resolve


def resolve_venue_flip_min_edge_pct(
    venue: Any,
    symbol: Any = "",
    *,
    order_type: Any = "LIMIT",
    base_min_edge_pct: float = DEFAULT_FLIP_MIN_EDGE_PCT,
    round_trip_fee_bps: Optional[float] = None,
) -> tuple[float, float]:
    """Scale flip minimum edge by venue round-trip fee anchored at Binance 20 bps RT.

    Returns ``(effective_min_edge_pct, venue_rt_fee_bps)``.
    With default ``base_min_edge_pct=0.60``:
    - Kraken xStocks (0 bps RT) -> 0.20%
    - Hyperliquid LIMIT (8 bps RT) -> 0.36% (MARKET 14 bps -> 0.48%)
    - Binance LIMIT/MARKET (20 bps RT) -> 0.60%
    - Trading212 (30 bps FX RT) -> 0.80%
    - Kraken Crypto LIMIT (32 bps RT) -> 0.84% (MARKET 52 bps -> 1.24%)
    """
    rt_bps = resolve_venue_round_trip_fee_bps(
        venue, symbol, order_type=order_type, round_trip_fee_bps=round_trip_fee_bps
    )
    base_edge = safe_nonneg_float(base_min_edge_pct, default=DEFAULT_FLIP_MIN_EDGE_PCT)
    if base_edge <= 0.0:
        return 0.0, rt_bps
    rt_pct = rt_bps / 100.0
    ref_pct = _REFERENCE_RT_FEE_BPS / 100.0
    adjusted = base_edge + 2.0 * (rt_pct - ref_pct)
    floor = max(_MIN_FLIP_EDGE_FLOOR_PCT, 1.5 * rt_pct) if base_edge >= 0.20 else base_edge
    return round(max(floor, adjusted), 4), rt_bps


def evaluate_ai_target_ev_guard(
    intent: dict[str, Any],
    *,
    decision: str,
    symbol: str,
    venue: str,
    price: float,
    target_ev_fee_mult: float = DEFAULT_TARGET_EV_FEE_MULT,
    min_target_rr: float = DEFAULT_MIN_TARGET_RR,
    round_trip_fee_bps: Optional[float] = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Pure 4B-1 guard blocking BUY entries and ``urgency='NORMAL'`` SELL intents whose explicit ``target_px`` has negative net EV or poor R:R.

    Returns ``(allowed, reason, metrics)``.
    - Never blocks intents without an explicit ``target_px`` (e.g. defensive exits or open-ended holds).
    - Bypasses only defensive ``SELL`` exits with ``urgency == 'HIGH'``; all ``BUY`` entries
      (including momentum/breakout entries) must satisfy both positive-EV and ``min_target_rr`` floors.
    - Blocks when gross distance from entry ``price`` to ``target_px`` is strictly below
      ``max(0.10%, target_ev_fee_mult * venue_round_trip_fee_pct)``, OR when both ``target_px``
      and ``invalidation_px`` are set and gross ``reward / risk < min_target_rr`` (e.g. after chasing a REPLACE).
    """
    mult = safe_nonneg_float(target_ev_fee_mult, default=DEFAULT_TARGET_EV_FEE_MULT)
    min_rr = safe_nonneg_float(min_target_rr, default=DEFAULT_MIN_TARGET_RR)
    if mult <= 0.0 and min_rr <= 0.0:
        return True, "", {}
    side = normalize_side(decision) or extract_intent_side(intent)
    if side not in ("BUY", "SELL"):
        return True, "", {}
    urgency = clean_upper_scalar(intent.get("urgency")) or "NORMAL"
    if side == "SELL" and urgency == "HIGH":
        return True, "", {"bypassed_urgency": "HIGH"}
    eff_px = safe_pos_float(price, default=0.0) or 0.0
    if eff_px <= 0.0:
        return True, "", {}
    inv_px, tgt_px = extract_intent_trade_plan_levels(intent, side=side, ref_price=eff_px)
    if tgt_px is None:
        return True, "", {}
    order_type = extract_intent_order_type(intent)
    rt_bps = resolve_venue_round_trip_fee_bps(
        venue, symbol, order_type=order_type, round_trip_fee_bps=round_trip_fee_bps
    )
    gross_pct = (
        ((tgt_px - eff_px) / eff_px) * 100.0
        if side == "BUY"
        else ((eff_px - tgt_px) / eff_px) * 100.0
    )
    min_target_pct = max(_MIN_TARGET_EV_FLOOR_PCT, mult * (rt_bps / 100.0)) if mult > 0.0 else 0.0
    metrics: dict[str, Any] = {
        "target_px": tgt_px,
        "target_gross_pct": round(gross_pct, 4),
        "min_target_ev_pct": round(min_target_pct, 4),
        "venue_rt_fee_bps": rt_bps,
    }
    if mult > 0.0 and gross_pct < min_target_pct:
        return (
            False,
            (
                f"ai_negative_ev_target({symbol}@{venue}:target={tgt_px:g}, "
                f"gross={gross_pct:.2f}%<min={min_target_pct:.2f}%, rt_fee={rt_bps:.0f}bps)"
            ),
            metrics,
        )
    if min_rr > 0.0 and inv_px is not None:
        risk_pct = (
            ((eff_px - inv_px) / eff_px) * 100.0
            if side == "BUY"
            else ((inv_px - eff_px) / eff_px) * 100.0
        )
        if risk_pct > 0.0:
            rr_ratio = gross_pct / risk_pct
            metrics["invalidation_px"] = inv_px
            metrics["risk_pct"] = round(risk_pct, 4)
            metrics["reward_risk_ratio"] = round(rr_ratio, 3)
            metrics["min_target_rr"] = round(min_rr, 3)
            if rr_ratio < min_rr:
                return (
                    False,
                    (
                        f"ai_poor_rr_plan({symbol}@{venue}:rr={rr_ratio:.2f}<min={min_rr:.2f}, "
                        f"reward={gross_pct:.2f}%, risk={risk_pct:.2f}%)"
                    ),
                    metrics,
                )
    return True, "", metrics


def evaluate_ai_tokenized_equity_sell_guard(
    intent: dict[str, Any],
    existing_intents: list[dict[str, Any]],
    *,
    decision: str,
    symbol: str,
    venue: str,
    price: float,
    confidence: float,
    now_ts: float,
    cost_basis_px: Optional[float] = None,
    min_profit_pct: float = DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
    min_target_drop_pct: float = DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
    min_confidence: float = DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
    cooldown_hours: float = DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
    crash_min_target_drop_pct: float = DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
    crash_min_confidence: float = DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
) -> tuple[bool, str, dict[str, Any]]:
    """Enforce long-term holding discipline on SELL-only tokenized equities (e.g. Kraken EU xStocks).

    Because EU Spot API cannot BUY tokenized equities and the owner must manually log in to rebuy
    with heavy re-entry fees/spread, exits must be rare and satisfy EITHER:
      (A) Profit-taking: unrealized profit >= ``min_profit_pct`` above active cost basis AND a
          high-conviction multi-day (1-2+ day) downside ``target_px`` >= ``min_target_drop_pct`` lower; OR
      (B) Imminent-crash capital preservation: ``urgency == "HIGH"`` AND ``confidence >= crash_min_confidence``
          AND an explicit crash ``target_px`` >= ``crash_min_target_drop_pct`` below sell price (bypassing the
          profit floor to stop severe losses while still enforcing the 24h sell cooldown).
    """
    side = normalize_side(decision) or extract_intent_side(intent)
    if side != "SELL":
        return True, "", {}
    ven = normalize_venue_name(clean_scalar_str(venue) or "")
    sym = clean_scalar_str(symbol) or ""
    if not (ven and sym and ProviderVenueMetadataMixin.supports_tokenized_assets(ven) and resolve_symbol_identity(sym).is_tokenized_equity):
        return True, "", {}

    req_profit_pct = safe_nonneg_float(min_profit_pct, default=DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT)
    req_drop_pct = safe_nonneg_float(min_target_drop_pct, default=DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT)
    req_conf = safe_nonneg_float(min_confidence, default=DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE)
    req_cd_h = safe_nonneg_float(cooldown_hours, default=DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS)
    req_crash_drop_pct = safe_nonneg_float(
        crash_min_target_drop_pct, default=DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT
    )
    req_crash_conf = max(
        req_conf,
        safe_nonneg_float(crash_min_confidence, default=DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE),
    )
    if req_profit_pct <= 0.0 and req_drop_pct <= 0.0 and req_conf <= 0.0 and req_cd_h <= 0.0:
        return True, "", {}

    eff_px = safe_pos_float(price, default=0.0) or 0.0
    eff_conf = safe_nonneg_float(confidence, default=0.0)
    eff_now = resolve_now_ts(now_ts)
    eff_cost = safe_pos_float(cost_basis_px, default=None)
    urgency_u = clean_upper_scalar(intent.get("urgency")) if isinstance(intent, dict) else None
    _, tgt_px = extract_intent_trade_plan_levels(intent, side="SELL", ref_price=eff_px) if eff_px > 0.0 else (None, None)
    drop_pct = (((eff_px - tgt_px) / eff_px) * 100.0) if (tgt_px is not None and eff_px > 0.0) else None
    metrics: dict[str, Any] = {
        "xstock_sell_min_profit_pct": req_profit_pct,
        "xstock_sell_min_target_drop_pct": req_drop_pct,
        "xstock_sell_min_confidence": req_conf,
        "xstock_sell_cooldown_hours": req_cd_h,
        "xstock_sell_crash_min_target_drop_pct": req_crash_drop_pct,
        "xstock_sell_crash_min_confidence": req_crash_conf,
    }
    if eff_cost is not None:
        metrics["xstock_cost_basis_px"] = round(eff_cost, 6)
        if eff_px > 0.0:
            metrics["xstock_profit_pct"] = round(((eff_px - eff_cost) / eff_cost) * 100.0, 3)
    if tgt_px is not None and drop_pct is not None:
        metrics["target_px"] = tgt_px
        metrics["xstock_target_drop_pct"] = round(drop_pct, 3)

    is_imminent_crash_exit = bool(
        req_crash_drop_pct > 0.0
        and urgency_u == "HIGH"
        and eff_conf >= req_crash_conf - 1e-9
        and drop_pct is not None
        and drop_pct >= req_crash_drop_pct - 1e-9
    )
    if is_imminent_crash_exit:
        metrics["xstock_crash_protection_exit"] = True

    if req_conf > 0.0 and eff_conf < req_conf - 1e-9:
        return (
            False,
            f"ai_xstock_longterm_low_confidence({sym}@{ven}:conf={eff_conf:.2f}<{req_conf:.2f})",
            metrics,
        )

    if not is_imminent_crash_exit:
        if req_profit_pct > 0.0:
            if eff_cost is None or eff_px <= 0.0:
                return (
                    False,
                    f"ai_xstock_longterm_missing_cost_basis({sym}@{ven}:cannot verify >={req_profit_pct:g}% profit)",
                    metrics,
                )
            profit_pct = ((eff_px - eff_cost) / eff_cost) * 100.0
            metrics["xstock_profit_pct"] = round(profit_pct, 3)
            if profit_pct < req_profit_pct - 1e-9:
                return (
                    False,
                    (
                        f"ai_xstock_longterm_min_profit({sym}@{ven}:px={eff_px:g} vs cost={eff_cost:g}, "
                        f"profit={fmt_signed(profit_pct, 2)}%<+{req_profit_pct:g}%)"
                    ),
                    metrics,
                )

        if req_drop_pct > 0.0 and eff_px > 0.0:
            if tgt_px is None or drop_pct is None:
                return (
                    False,
                    (
                        f"ai_xstock_longterm_missing_target_px({sym}@{ven}:requires explicit multi-day "
                        f"target_px >= {req_drop_pct:g}% below sell price)"
                    ),
                    metrics,
                )
            if drop_pct < req_drop_pct - 1e-9:
                return (
                    False,
                    (
                        f"ai_xstock_longterm_min_target_drop({sym}@{ven}:target={tgt_px:g}, "
                        f"drop={drop_pct:.2f}%<{req_drop_pct:g}% multi-day floor)"
                    ),
                    metrics,
                )

    if req_cd_h > 0.0:
        cur_iid = _extract_intent_id(intent)
        items = existing_intents if isinstance(existing_intents, list) else []
        canceled_oids = _collect_canceled_order_ids(items, exclude_iid=cur_iid)
        cd_sec = req_cd_h * 3600.0
        latest_sell_ts = 0.0
        for it in items:
            if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
                continue
            if not _is_filled_ai_intent(it, canceled_oids):
                continue
            if extract_intent_side(it) != "SELL":
                continue
            if not symbols_match_intent_scope_and_quote(
                it.get("symbol"), sym, venue_a=it.get("venue"), venue_b=ven
            ):
                continue
            ts_s = _extract_intent_exec_ts(it, max_ts=eff_now + 60.0)
            if 0.0 < ts_s <= eff_now + 60.0 and ts_s > latest_sell_ts:
                latest_sell_ts = ts_s
        if latest_sell_ts > 0.0 and (eff_now - latest_sell_ts) < cd_sec:
            elapsed_h = max(0.0, (eff_now - latest_sell_ts) / 3600.0)
            metrics["xstock_sell_elapsed_hours"] = round(elapsed_h, 2)
            return (
                False,
                f"ai_xstock_longterm_sell_cooldown({sym}@{ven}:dt={elapsed_h:.1f}h<{req_cd_h:g}h)",
                metrics,
            )

    return True, "", metrics


def evaluate_ai_ping_pong_guard(
    intent: dict[str, Any],
    existing_intents: list[dict[str, Any]],
    *,
    decision: str,
    symbol: str,
    venue: str,
    price: float,
    now_ts: float,
    flip_cooldown_min: float = DEFAULT_FLIP_COOLDOWN_MIN,
    flip_min_edge_pct: float = DEFAULT_FLIP_MIN_EDGE_PCT,
    max_flips_6h: int = DEFAULT_MAX_FLIPS_6H,
    spot_price: Optional[float] = None,
    recent_bot_fill: Optional[Mapping[str, Any]] = None,
    round_trip_fee_bps: Optional[float] = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Deterministic fee-aware anti-ping-pong guard protecting against rapid unprofitable opposite-side flips.

    Returns ``(allowed, reason, metrics)``.
    - Allows opposite-side flips immediately when the AI's pre-committed ``invalidation_px`` or
      ``target_px`` plan level is triggered.
    - Scales ``flip_min_edge_pct`` by venue round-trip fee (anchored at Binance 20 bps RT), and
      allows flips when favorable edge >= effective edge or on emergency defensive ``SELL`` drops
      <= ``-2.5 * effective_edge`` (or ``urgency == "HIGH"`` <= ``-effective_edge``).
    - Blocks narrow-band/adverse rapid flips within ``flip_cooldown_min`` (including fresh ``by=BOT`` fills
      within 30m) and caps 6h opposite-side reversals per ``(venue, symbol)`` at ``max_flips_6h``.
    """
    side = normalize_side(decision) or extract_intent_side(intent)
    if side not in ("BUY", "SELL"):
        return True, "", {}

    opp_side = opposite_side(side)
    cur_iid = _extract_intent_id(intent)
    eff_now = resolve_now_ts(now_ts)
    eff_px = safe_pos_float(price, default=0.0) or 0.0
    eff_spot = safe_pos_float(spot_price, default=eff_px) or eff_px
    cd_min = safe_nonneg_float(flip_cooldown_min, default=DEFAULT_FLIP_COOLDOWN_MIN)
    base_min_edge = safe_nonneg_float(flip_min_edge_pct, default=DEFAULT_FLIP_MIN_EDGE_PCT)
    order_type = extract_intent_order_type(intent)
    min_edge, rt_fee_bps = resolve_venue_flip_min_edge_pct(
        venue,
        symbol,
        order_type=order_type,
        base_min_edge_pct=base_min_edge,
        round_trip_fee_bps=round_trip_fee_bps,
    )
    cap_flips = safe_nonneg_int(max_flips_6h, default=DEFAULT_MAX_FLIPS_6H)
    emergency_drop_pct = min_edge * _EMERGENCY_SELL_EDGE_MULT
    urgency = clean_upper_scalar(intent.get("urgency")) or "NORMAL"

    if cd_min <= 0.0 and cap_flips <= 0:
        return True, "", {}

    items = existing_intents if isinstance(existing_intents, list) else []
    canceled_oids = _collect_canceled_order_ids(items, exclude_iid=cur_iid)
    whipsaw_cutoff = eff_now - _WHIPSAW_WINDOW_SEC

    sym_fills: list[tuple[float, str, float, dict[str, Any]]] = []
    latest_opp_ai: Optional[tuple[float, float, dict[str, Any]]] = None
    latest_same_ai_ts = 0.0
    for it in items:
        if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
            continue
        if not _is_filled_ai_intent(it, canceled_oids):
            continue
        if not symbols_match_intent_scope_and_quote(
            it.get("symbol"), symbol, venue_a=it.get("venue"), venue_b=venue
        ):
            continue
        it_side = extract_intent_side(it)
        if it_side not in ("BUY", "SELL"):
            continue
        ts_s = _extract_intent_exec_ts(it, max_ts=eff_now + 60.0)
        if ts_s <= 0.0 or ts_s > eff_now + 60.0:
            continue
        fill_px = _extract_intent_exec_price(it)
        if ts_s >= whipsaw_cutoff:
            sym_fills.append((ts_s, it_side, fill_px, it))
        if it_side == opp_side and (latest_opp_ai is None or ts_s >= latest_opp_ai[0]):
            latest_opp_ai = (ts_s, fill_px, it)
        elif it_side == side and ts_s > latest_same_ai_ts:
            latest_same_ai_ts = ts_s

    # Resolve any fresh by=BOT opposite-side fill within bot cooldown (<= 30m)
    cd_sec = cd_min * 60.0
    bot_cd_sec = min(cd_sec, _MAX_BOT_FLIP_COOLDOWN_SEC) if cd_sec > 0.0 else 0.0
    bot_opp_ts = 0.0
    bot_opp_px = 0.0
    if bot_cd_sec > 0.0 and isinstance(recent_bot_fill, Mapping):
        b_side = normalize_side(recent_bot_fill.get("side"))
        b_ts = parse_pos_epoch_s(recent_bot_fill.get("ts")) or 0.0
        b_px = safe_pos_float(recent_bot_fill.get("price"), default=0.0) or 0.0
        if (
            (not b_side or b_side == opp_side)
            and b_ts > 0.0
            and b_ts <= eff_now + 60.0
            and (eff_now - b_ts) <= bot_cd_sec
        ):
            bot_opp_ts, bot_opp_px = b_ts, b_px

    # 1. Pre-committed AI trade plan check (only for an active, unclosed opposite AI fill not superseded by a newer same-side or BOT fill)
    opp_ai_is_latest_side = bool(latest_opp_ai is not None and latest_opp_ai[0] > latest_same_ai_ts)
    plan_status = "ACTIVE"
    if opp_ai_is_latest_side and latest_opp_ai[0] >= bot_opp_ts:
        _, opp_px_ai, opp_intent_ai = latest_opp_ai
        frozen_st = extract_frozen_trade_plan_status(opp_intent_ai)
        if frozen_st in ("INVALIDATED", "TARGET_HIT"):
            return True, "", {"plan_triggered": frozen_st, "flip_bypassed_by_plan": True}
        if eff_spot > 0.0:
            inv_px, tgt_px = extract_intent_trade_plan_levels(
                opp_intent_ai, side=opp_side, ref_price=opp_px_ai if opp_px_ai > 0.0 else None
            )
            if inv_px is not None or tgt_px is not None:
                plan_status = evaluate_trade_plan_status(opp_side, eff_spot, inv_px, tgt_px)
                if plan_status in ("INVALIDATED", "TARGET_HIT"):
                    return True, "", {"plan_triggered": plan_status, "flip_bypassed_by_plan": True}

    # Count 6h opposite-side reversals on this symbol (strictly by=AI fills)
    sym_fills.sort(key=lambda r: r[0])
    flips_6h = 0
    prev_side_6h = ""
    for _, f_side, _, _ in sym_fills:
        if prev_side_6h and f_side != prev_side_6h:
            flips_6h += 1
        prev_side_6h = f_side

    metrics: dict[str, Any] = {
        "flips_6h": flips_6h,
        "last_6h_side": prev_side_6h or None,
        "venue_rt_fee_bps": rt_fee_bps,
        "effective_flip_min_edge_pct": min_edge,
    }

    # Resolve the most recent unclosed opposite-side fill within cooldown (AI fill or fresh BOT fill <= 30m)
    ref_opp_ts = 0.0
    ref_opp_px = 0.0
    ref_opp_actor = ""
    if cd_sec > 0.0 and opp_ai_is_latest_side and (eff_now - latest_opp_ai[0]) <= cd_sec:
        ref_opp_ts, ref_opp_px, ref_opp_actor = latest_opp_ai[0], latest_opp_ai[1], "AI"
    if bot_opp_ts > ref_opp_ts and bot_opp_ts > latest_same_ai_ts:
        ref_opp_ts, ref_opp_px, ref_opp_actor = bot_opp_ts, bot_opp_px, "BOT"

    fav_pct: Optional[float] = None
    if ref_opp_px > 0.0 and eff_px > 0.0:
        fav_pct = (
            ((eff_px - ref_opp_px) / ref_opp_px) * 100.0
            if side == "SELL"
            else ((ref_opp_px - eff_px) / ref_opp_px) * 100.0
        )
        metrics["flip_fav_pct"] = round(fav_pct, 3)

    is_emergency_sell = bool(
        side == "SELL"
        and fav_pct is not None
        and (
            fav_pct <= -emergency_drop_pct
            or (urgency == "HIGH" and fav_pct <= -min_edge)
        )
    )

    # 2. Opposite-side flip cooldown check
    if ref_opp_ts > 0.0:
        elapsed_min = max(0.0, (eff_now - ref_opp_ts) / 60.0)
        metrics["flip_elapsed_min"] = round(elapsed_min, 1)
        metrics["flip_ref_actor"] = ref_opp_actor
        has_valid_edge = fav_pct is not None and fav_pct >= min_edge
        if not has_valid_edge and not is_emergency_sell:
            fav_lbl = f"{fmt_signed(fav_pct, 2)}%" if fav_pct is not None else "n/a"
            return (
                False,
                (
                    f"ai_ping_pong_flip_cooldown({symbol}@{venue}:{opp_side}->{side} "
                    f"by={ref_opp_actor}, dt={elapsed_min:.1f}m<{cd_min:g}m, "
                    f"fav={fav_lbl}<+{min_edge:.2f}%)"
                ),
                metrics,
            )

    # 3. 6-hour whipsaw reversal cap per symbol (always allow emergency crash SELL vs latest AI BUY fill)
    if cap_flips > 0 and prev_side_6h and side != prev_side_6h and flips_6h >= cap_flips:
        ai_sell_fav_pct = fav_pct
        if side == "SELL" and ai_sell_fav_pct is None and latest_opp_ai is not None and latest_opp_ai[1] > 0.0 and eff_px > 0.0:
            ai_sell_fav_pct = ((eff_px - latest_opp_ai[1]) / latest_opp_ai[1]) * 100.0
        if not (side == "SELL" and ai_sell_fav_pct is not None and ai_sell_fav_pct <= -emergency_drop_pct):
            return (
                False,
                f"ai_ping_pong_whipsaw_cap({symbol}@{venue}:{flips_6h}/{cap_flips} flips in 6h)",
                metrics,
            )

    return True, "", metrics


def evaluate_ai_decision_scorecard(
    intents: list[dict[str, Any]],
    *,
    now_ts: float,
    live_prices: Optional[Mapping[str, Any]] = None,
    window_trades: int = DEFAULT_SCORECARD_WINDOW_TRADES,
    max_loss_streak: int = DEFAULT_SCORECARD_MAX_LOSS_STREAK,
    cooldown_hours: float = DEFAULT_SCORECARD_COOLDOWN_HOURS,
    lookback_hours: float = 48.0,
    min_mature_minutes: float = 60.0,
    scope: str = "all",
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
) -> dict[str, Any]:
    """Deterministically score recent filled ``by=AI`` decisions and compute BUY shadow cooldown state.

    - Closed ``BUY -> SELL`` round-trips score realized price edge (``LOSS`` < -0.35%).
    - Closed unmatched ``SELL -> BUY`` reloads score reload timing edge (``LOSS`` < -0.35%).
    - Open ``BUY`` entries score ``LOSS`` immediately when ``invalidation_px`` is breached, or
      once matured (>= ``min_mature_minutes``, default 60m) if mark-to-market < -1.50% (so
      normal intraday DCA dip noise above invalidation never triggers false BUY cooldowns).
    - Standalone unmatched ``SELL`` exits (e.g. taking profit on older inventory) are never penalized.
    """
    eff_now = resolve_now_ts(now_ts)
    win_n = safe_nonneg_int(window_trades, default=DEFAULT_SCORECARD_WINDOW_TRADES)
    streak_cap = safe_nonneg_int(max_loss_streak, default=DEFAULT_SCORECARD_MAX_LOSS_STREAK)
    cd_hours = safe_nonneg_float(cooldown_hours, default=DEFAULT_SCORECARD_COOLDOWN_HOURS)
    lb_sec = max(3600.0, (safe_pos_float(lookback_hours, default=48.0) or 48.0) * 3600.0)
    mature_sec = max(0.0, safe_nonneg_float(min_mature_minutes, default=60.0)) * 60.0
    prices_map = live_prices if isinstance(live_prices, Mapping) else {}
    items = intents if isinstance(intents, list) else []

    empty_res: dict[str, Any] = {
        "count": 0,
        "total_outcomes": 0,
        "window_trades": win_n,
        "wins": 0,
        "scratches": 0,
        "losses": 0,
        "net_usd": 0.0,
        "avg_edge_pct": 0.0,
        "loss_streak": 0,
        "weak_window": False,
        "latest_loss_ts": 0.0,
        "buy_shadow_active": False,
        "cooldown_remaining_h": 0.0,
        "unmatched_sells": 0,
        "status": "HEALTHY",
        "reason": "",
    }
    if win_n <= 0 or not items:
        return empty_res

    canceled_oids = _collect_canceled_order_ids(items)
    cutoff = eff_now - lb_sec

    filled_rows: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict) or not _is_filled_ai_intent(it, canceled_oids):
            continue
        sym = clean_scalar_str(it.get("symbol")) or ""
        ven = normalize_venue_name(clean_scalar_str(it.get("venue")) or "")
        if scope and scope.strip().lower() != "all" and not matches_asset_scope(sym, ven, scope):
            continue
        side = extract_intent_side(it)
        if side not in ("BUY", "SELL"):
            continue
        ts_s = _extract_intent_exec_ts(it, max_ts=eff_now + 60.0)
        px = _extract_intent_exec_price(it)
        if ts_s <= 0.0 or ts_s > eff_now + 60.0 or px <= 0.0:
            continue
        filled_rows.append(
            build_normalized_filled_ai_intent_row(
                it,
                venue=ven,
                symbol=sym,
                side=side,
                ts_s=ts_s,
                px=px,
                fee_resolver=fee_resolver,
            )
        )

    if not filled_rows:
        return empty_res

    def _classify(
        edge_pct: float,
        *,
        invalidated: bool = False,
        loss_threshold: float = _LOSS_EDGE_PCT,
    ) -> str:
        if invalidated or edge_pct < loss_threshold:
            return "LOSS"
        if edge_pct > _WIN_EDGE_PCT:
            return "WIN"
        return "SCRATCH"

    closed_pairs, open_buys, open_sells = match_filled_intent_rows_lifo_in_window(
        filled_rows, cutoff_ts=cutoff
    )
    unmatched_sells = len(open_sells)
    outcomes: list[dict[str, Any]] = []

    for kind, entry_row, exit_row in closed_pairs:
        gross_edge_pct = compute_signed_price_edge_pct(entry_row["side"], entry_row["px"], exit_row["px"])
        pair_rt_bps = 0.5 * (
            safe_nonneg_float(entry_row.get("rt_fee_bps"), default=0.0)
            + safe_nonneg_float(exit_row.get("rt_fee_bps"), default=0.0)
        )
        edge_pct = gross_edge_pct - (pair_rt_bps / 100.0)
        entry_usd = safe_pos_float(entry_row.get("usd"), default=0.0) or 0.0
        exit_usd = safe_pos_float(exit_row.get("usd"), default=0.0) or 0.0
        matched_usd = entry_usd if entry_usd > 0.0 else resolve_matched_notional_usd(entry_usd, exit_usd)
        outcomes.append({
            "ts": exit_row["ts"],
            "kind": kind,
            "symbol": exit_row["symbol"],
            "edge_pct": edge_pct,
            "usd_impact": matched_usd * (edge_pct / 100.0),
            "grade": _classify(edge_pct),
        })

    for buy_row in open_buys:
        spot_px = _resolve_live_price_for_symbol(prices_map, buy_row["venue"], buy_row["symbol"])
        if spot_px is None or spot_px <= 0.0:
            continue
        plan_st = evaluate_trade_plan_status(
            "BUY", spot_px, buy_row["invalidation_px"], buy_row["target_px"]
        )
        is_invalidated = (
            buy_row.get("resolved_plan_status") == "INVALIDATED" or plan_st == "INVALIDATED"
        )
        if not is_invalidated and (eff_now - buy_row["ts"]) < mature_sec:
            continue
        gross_edge_pct = compute_signed_price_edge_pct("BUY", buy_row["px"], spot_px)
        edge_pct = gross_edge_pct - (safe_nonneg_float(buy_row.get("rt_fee_bps"), default=0.0) / 100.0)
        outcomes.append({
            "ts": buy_row["ts"],
            "kind": "BUY_MTM",
            "symbol": buy_row["symbol"],
            "edge_pct": edge_pct,
            "usd_impact": buy_row["usd"] * (edge_pct / 100.0),
            "grade": _classify(
                edge_pct,
                invalidated=is_invalidated,
                loss_threshold=_OPEN_MTM_LOSS_EDGE_PCT,
            ),
        })

    if not outcomes:
        return empty_res

    outcomes.sort(key=lambda o: o["ts"])
    recent = outcomes[-win_n:]
    wins = sum(1 for o in recent if o["grade"] == "WIN")
    scratches = sum(1 for o in recent if o["grade"] == "SCRATCH")
    losses = sum(1 for o in recent if o["grade"] == "LOSS")
    net_usd = sum(o["usd_impact"] for o in recent)
    avg_edge = sum(o["edge_pct"] for o in recent) / len(recent)

    loss_streak = 0
    for o in reversed(recent):
        if o["grade"] == "LOSS":
            loss_streak += 1
        else:
            break

    weak_threshold = max(win_n - 1, streak_cap) if streak_cap > 0 else max(win_n - 1, 1)
    weak_window = bool(len(recent) >= win_n and losses >= weak_threshold and net_usd < 0.0)
    tripped_streak = bool(streak_cap > 0 and loss_streak >= streak_cap)
    latest_loss_ts = max((o["ts"] for o in recent if o["grade"] == "LOSS"), default=0.0)

    buy_shadow_active = False
    cooldown_remaining_h = 0.0
    reason = ""
    if (tripped_streak or weak_window) and cd_hours > 0.0 and latest_loss_ts > 0.0:
        cd_until = latest_loss_ts + cd_hours * 3600.0
        if cd_until > eff_now:
            buy_shadow_active = True
            cooldown_remaining_h = round((cd_until - eff_now) / 3600.0, 2)
            trig = f"streak={loss_streak}L" if tripped_streak else f"weak_window={losses}/{len(recent)}L"
            net_lbl = format_signed_usd(net_usd, decimals=0)
            reason = (
                f"ai_scorecard_buy_shadow_cooldown({trig}, net={net_lbl}, "
                f"remaining={cooldown_remaining_h:.1f}h)"
            )

    if buy_shadow_active:
        status = "BUY_SHADOW_COOLDOWN"
    elif streak_cap > 1 and loss_streak >= streak_cap - 1:
        status = "CAUTION"
    else:
        status = "HEALTHY"

    return {
        "count": len(recent),
        "total_outcomes": len(outcomes),
        "window_trades": win_n,
        "wins": wins,
        "scratches": scratches,
        "losses": losses,
        "net_usd": round(net_usd, 2),
        "avg_edge_pct": round(avg_edge, 2),
        "loss_streak": loss_streak,
        "weak_window": weak_window,
        "latest_loss_ts": round(latest_loss_ts, 3),
        "buy_shadow_active": buy_shadow_active,
        "cooldown_remaining_h": cooldown_remaining_h,
        "unmatched_sells": unmatched_sells,
        "status": status,
        "reason": reason,
    }


def format_ai_decision_scorecard_line(scorecard: Any) -> str:
    """Format a compact single-line summary of the AI Decision Scorecard for Section 3C."""
    if not isinstance(scorecard, Mapping):
        return ""
    cnt = safe_nonneg_int(scorecard.get("count"), default=0)
    if cnt <= 0:
        return ""
    tot = safe_nonneg_int(scorecard.get("total_outcomes"), default=0)
    w = safe_nonneg_int(scorecard.get("wins"), default=0)
    s = safe_nonneg_int(scorecard.get("scratches"), default=0)
    lo = safe_nonneg_int(scorecard.get("losses"), default=0)
    net_u = safe_float(scorecard.get("net_usd"), default=0.0) or 0.0
    avg_e = safe_float(scorecard.get("avg_edge_pct"), default=0.0) or 0.0
    streak = safe_nonneg_int(scorecard.get("loss_streak"), default=0)
    unm_sells = safe_nonneg_int(scorecard.get("unmatched_sells"), default=0)
    status = str(scorecard.get("status") or "HEALTHY").strip().upper()
    rem_h = safe_nonneg_float(scorecard.get("cooldown_remaining_h"), default=0.0)
    streak_sfx = f", streak={streak}L" if streak >= 2 else ""
    unm_sfx = f", excl_unmatched_sells={unm_sells}" if unm_sells > 0 else ""
    if status == "BUY_SHADOW_COOLDOWN" and rem_h > 0.0:
        tag_sfx = f" [BUY_SHADOW_COOLDOWN: {rem_h:.1f}h left]"
    else:
        tag_sfx = ""
    net_lbl = format_signed_usd(net_u, decimals=0)
    win_lbl = f"last {cnt}/{tot}" if tot > cnt else f"n={cnt}"
    return (
        f"AI Decision Scorecard (48h, {win_lbl}): "
        f"{w}W/{s}BE/{lo}L, avg={fmt_signed(avg_e, 2)}%, net={net_lbl}{streak_sfx}{unm_sfx}{tag_sfx}"
    )


_LIMIT_PASSIVE_OFFSET_BUCKETS: tuple[tuple[str, str, Optional[float], Optional[float]], ...] = (
    ("le_40bps", "<=40bp", None, 40.0),
    ("40_100bps", "40-100bp", 40.0, 100.0),
    ("gt_100bps", ">100bp", 100.0, None),
)


def extract_limit_passive_offset_bps(
    intent: Mapping[str, Any],
    *,
    side: str = "",
    order_type: str = "",
) -> Optional[float]:
    """Return signed passive LIMIT distance from arrival ``reference_price`` in bps (>0 passive, <0 aggressive)."""
    if not isinstance(intent, Mapping):
        return None
    ot = clean_upper_scalar(order_type) or extract_intent_order_type(intent)
    if ot != "LIMIT":
        return None
    side_u = normalize_side(side) or extract_intent_side(intent)
    if side_u not in ("BUY", "SELL"):
        return None
    lim_px = first_pos_float(intent.get("limit_price"), intent.get("requested_price"))
    ref_px = first_pos_float(intent.get("reference_price"))
    if lim_px is None or ref_px is None:
        return None
    raw_bps = ((ref_px - lim_px) / ref_px) * 10000.0 if side_u == "BUY" else ((lim_px - ref_px) / ref_px) * 10000.0
    return round(raw_bps, 2)


def summarize_limit_passive_offset_fill_rates(
    filled_offsets_bps: Sequence[float],
    unfilled_offsets_bps: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Partition filled vs. unfilled LIMIT passive offsets (bps from arrival spot) into empirical fill-rate bands."""
    f_vals = [v for raw in filled_offsets_bps if (v := safe_float(raw, default=None)) is not None]
    u_vals = [v for raw in unfilled_offsets_bps if (v := safe_float(raw, default=None)) is not None]
    out: dict[str, dict[str, Any]] = {}
    for key, label, lo_excl, hi_incl in _LIMIT_PASSIVE_OFFSET_BUCKETS:
        f_n = sum(
            1 for v in f_vals
            if (lo_excl is None or v > lo_excl) and (hi_incl is None or v <= hi_incl)
        )
        u_n = sum(
            1 for v in u_vals
            if (lo_excl is None or v > lo_excl) and (hi_incl is None or v <= hi_incl)
        )
        tot = f_n + u_n
        out[key] = {
            "label": label,
            "filled": f_n,
            "unfilled": u_n,
            "total": tot,
            "fill_rate_pct": round((f_n / tot) * 100.0, 1) if tot > 0 else None,
        }
    return out


def extract_fill_execution_slippage(
    intent: Mapping[str, Any],
    *,
    side: str,
    exec_px: float,
    order_type: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute ``(limit_slip_bps, arrival_sf_bps, market_slip_bps)`` for one filled intent."""
    side_u = normalize_side(side)
    ep = safe_pos_float(exec_px, default=None)
    if side_u not in ("BUY", "SELL") or ep is None or not isinstance(intent, Mapping):
        return None, None, None
    ot = clean_upper_scalar(order_type) or "LIMIT"
    lim_px = first_pos_float(intent.get("limit_price"), intent.get("requested_price") if ot == "LIMIT" else None)
    ref_px = first_pos_float(intent.get("reference_price"))
    limit_slip_bps: Optional[float] = None
    arrival_sf_bps: Optional[float] = None
    market_slip_bps: Optional[float] = None
    if ot == "LIMIT" and lim_px is not None:
        limit_slip_bps = _compute_signed_exec_bps(side_u, ep, lim_px)
        if ref_px is not None:
            arrival_sf_bps = _compute_signed_exec_bps(side_u, ep, ref_px)
    elif ot != "LIMIT" and ref_px is not None:
        market_slip_bps = _compute_signed_exec_bps(side_u, ep, ref_px)
    return limit_slip_bps, arrival_sf_bps, market_slip_bps


def summarize_execution_quality_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate LIMIT/MARKET fill counts and average slippage/arrival shortfall bps for a bucket of fills."""
    valid_rows = [r for r in rows if isinstance(r, Mapping)]
    lim_slips = [
        v for r in valid_rows
        if (v := safe_float(r.get("limit_slip_bps"), default=None)) is not None
    ]
    arr_sfs = [
        v for r in valid_rows
        if (v := safe_float(r.get("arrival_sf_bps"), default=None)) is not None
    ]
    mkt_slips = [
        v for r in valid_rows
        if (v := safe_float(r.get("market_slip_bps"), default=None)) is not None
    ]
    lim_filled = sum(1 for r in valid_rows if (clean_upper_scalar(r.get("order_type")) or "") == "LIMIT")
    return {
        "limit_filled": lim_filled,
        "market_filled": len(valid_rows) - lim_filled,
        "avg_limit_slippage_bps": round(sum(lim_slips) / len(lim_slips), 2) if lim_slips else None,
        "avg_arrival_shortfall_bps": round(sum(arr_sfs) / len(arr_sfs), 2) if arr_sfs else None,
        "avg_market_slippage_bps": round(sum(mkt_slips) / len(mkt_slips), 2) if mkt_slips else None,
    }


def summarize_execution_quality_by_venue_and_order_type(
    filled_rows: Sequence[Mapping[str, Any]],
    *,
    unfilled_counts: Optional[Mapping[tuple[str, str], int]] = None,
    unfilled_limit_offsets_by_venue: Optional[Mapping[str, Sequence[float]]] = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Group filled and unfilled execution TCA metrics by ``venue`` and by ``venue:order_type``."""
    valid_rows = [r for r in filled_rows if isinstance(r, Mapping)]
    unf_map: dict[tuple[str, str], int] = {}
    if isinstance(unfilled_counts, Mapping):
        for k, v in unfilled_counts.items():
            if isinstance(k, tuple) and len(k) == 2:
                ven_u = normalize_venue_name(clean_scalar_str(k[0]) or "") or "unknown"
                ot_u = clean_upper_scalar(k[1]) or "LIMIT"
                cnt_u = safe_nonneg_int(v, default=0)
                if cnt_u > 0:
                    unf_map[(ven_u, ot_u)] = unf_map.get((ven_u, ot_u), 0) + cnt_u
    unf_off_map: dict[str, list[float]] = {}
    if isinstance(unfilled_limit_offsets_by_venue, Mapping):
        for raw_ven, raw_offs in unfilled_limit_offsets_by_venue.items():
            ven_u = normalize_venue_name(clean_scalar_str(raw_ven) or "") or "unknown"
            if isinstance(raw_offs, Sequence) and not isinstance(raw_offs, (str, bytes)):
                vals = [v for x in raw_offs if (v := safe_float(x, default=None)) is not None]
                if vals:
                    unf_off_map.setdefault(ven_u, []).extend(vals)

    by_venue_rows: dict[str, list[Mapping[str, Any]]] = {}
    by_vot_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in valid_rows:
        ven = normalize_venue_name(clean_scalar_str(row.get("venue")) or "") or "unknown"
        ot = "LIMIT" if (clean_upper_scalar(row.get("order_type")) or "LIMIT") == "LIMIT" else "MARKET"
        by_venue_rows.setdefault(ven, []).append(row)
        by_vot_rows.setdefault((ven, ot), []).append(row)

    all_venues = sorted(set(by_venue_rows.keys()) | {v for v, _ in unf_map})
    by_venue: dict[str, dict[str, Any]] = {}
    for ven in all_venues:
        v_rows = by_venue_rows.get(ven, [])
        bucket = summarize_execution_quality_bucket(v_rows)
        lim_unf = unf_map.get((ven, "LIMIT"), 0)
        lim_tot = bucket["limit_filled"] + lim_unf
        notional_usd = sum(safe_nonneg_float(r.get("usd"), default=0.0) for r in v_rows)
        leg_fees = [
            f_v for r in v_rows
            if (f_v := safe_nonneg_float(r.get("leg_fee_bps"), default=None)) is not None
        ]
        f_offs = [
            off_v for r in v_rows
            if (clean_upper_scalar(r.get("order_type")) or "") == "LIMIT"
            and (off_v := safe_float(r.get("limit_passive_offset_bps"), default=None)) is not None
        ]
        u_offs = unf_off_map.get(ven, [])
        by_venue[ven] = {
            **bucket,
            "limit_unfilled": lim_unf,
            "limit_fill_rate_pct": round((bucket["limit_filled"] / lim_tot) * 100.0, 1) if lim_tot > 0 else None,
            "notional_usd": round(notional_usd, 2),
            "avg_leg_fee_bps": round(sum(leg_fees) / len(leg_fees), 2) if leg_fees else None,
            "med_filled_offset_bps": round(median(f_offs), 1) if f_offs else None,
            "med_unfilled_offset_bps": round(median(u_offs), 1) if u_offs else None,
        }

    all_vot = sorted(set(by_vot_rows.keys()) | set(unf_map.keys()))
    by_vot: dict[str, dict[str, Any]] = {}
    for ven, ot in all_vot:
        rows_k = by_vot_rows.get((ven, ot), [])
        bucket = summarize_execution_quality_bucket(rows_k)
        n_filled = len(rows_k)
        n_unf = unf_map.get((ven, ot), 0)
        n_sub = n_filled + n_unf
        notional_usd = sum(safe_nonneg_float(r.get("usd"), default=0.0) for r in rows_k)
        leg_fees = [
            f_v for r in rows_k
            if (f_v := safe_nonneg_float(r.get("leg_fee_bps"), default=None)) is not None
        ]
        avg_slip = bucket["avg_limit_slippage_bps"] if ot == "LIMIT" else bucket["avg_market_slippage_bps"]
        avg_arr = bucket["avg_arrival_shortfall_bps"] if ot == "LIMIT" else None
        ven_bkt = by_venue.get(ven, {})
        by_vot[f"{ven}:{ot}"] = {
            "venue": ven,
            "order_type": ot,
            "filled": n_filled,
            "unfilled": n_unf,
            "fill_rate_pct": round((n_filled / n_sub) * 100.0, 1) if n_sub > 0 else None,
            "notional_usd": round(notional_usd, 2),
            "avg_slippage_bps": avg_slip,
            "avg_arrival_shortfall_bps": avg_arr,
            "avg_leg_fee_bps": round(sum(leg_fees) / len(leg_fees), 2) if leg_fees else None,
            "med_filled_offset_bps": ven_bkt.get("med_filled_offset_bps") if ot == "LIMIT" else None,
            "med_unfilled_offset_bps": ven_bkt.get("med_unfilled_offset_bps") if ot == "LIMIT" else None,
        }
    return by_venue, by_vot


def _is_actionable_filled_trade_plan(
    venue: str,
    symbol: str,
    side: str,
    inv_px: Optional[float],
    tgt_px: Optional[float],
) -> bool:
    """Return True when a filled intent carries a directional plan on a two-way tradable instrument."""
    if inv_px is None and tgt_px is None:
        return False
    if (
        side == "SELL"
        and ProviderVenueMetadataMixin.supports_tokenized_assets(venue)
        and resolve_symbol_identity(symbol).is_tokenized_equity
    ):
        return False
    return True


def evaluate_filled_trade_plans(
    filled_rows: Sequence[Mapping[str, Any]],
    live_prices: Optional[Mapping[str, Any]] = None,
    *,
    price_bars_by_symbol: Optional[Mapping[str, Sequence[Any]]] = None,
    newly_resolved_out: Optional[dict[str, str]] = None,
    row_outcomes_out: Optional[list[dict[str, Any]]] = None,
    now_ts: Optional[float] = None,
) -> tuple[dict[str, int], dict[str, dict[str, int]], int]:
    """Evaluate every filled intent with a trade plan against frozen status, chronological bars, subsequent fills, and live spot."""
    prices_map = live_prices if isinstance(live_prices, Mapping) else {}
    eff_now = resolve_now_ts(now_ts)
    empty_side = lambda: {"TARGET_HIT": 0, "INVALIDATED": 0, "CLOSED_EARLY": 0, "ACTIVE": 0}
    status_counts = empty_side()
    by_side = {"BUY": empty_side(), "SELL": empty_side()}
    ordered = sorted(filled_rows, key=lambda r: safe_float(r.get("ts"), default=0.0) or 0.0)
    by_inst: dict[Any, list[Mapping[str, Any]]] = {}
    for row in ordered:
        by_inst.setdefault(row.get("inst_key"), []).append(row)
    planned_n = 0
    for inst_rows in by_inst.values():
        for idx, row in enumerate(inst_rows):
            if not row.get("has_plan"):
                continue
            side_k = normalize_side(row.get("side"))
            if side_k not in ("BUY", "SELL"):
                continue
            planned_n += 1
            entry_ts = safe_float(row.get("ts"), default=0.0) or 0.0
            hor_h = safe_pos_float(row.get("horizon_h"), default=None)
            sym_bars = resolve_symbol_price_bars(
                price_bars_by_symbol, row.get("venue"), row.get("symbol")
            )
            spot_px_cached: Optional[float] = None
            resolved_ts: Optional[float] = None
            if frozen_st := extract_frozen_trade_plan_status(row):
                final_st = frozen_st
            else:
                prev_ts = entry_ts
                final_st = None
                for nxt in inst_rows[idx + 1:]:
                    nxt_ts = safe_float(nxt.get("ts"), default=0.0) or 0.0
                    if sym_bars:
                        path_st = evaluate_trade_plan_path_status(
                            side_k,
                            None,
                            row.get("invalidation_px"),
                            row.get("target_px"),
                            start_ts=prev_ts,
                            end_ts=nxt_ts if nxt_ts > prev_ts else None,
                            price_bars=sym_bars,
                            check_spot=False,
                        )
                        if path_st in ("TARGET_HIT", "INVALIDATED"):
                            final_st = path_st
                            resolved_ts = nxt_ts if nxt_ts > prev_ts else prev_ts
                            break
                    st_nxt = evaluate_trade_plan_status(
                        side_k, nxt.get("px"), row.get("invalidation_px"), row.get("target_px")
                    )
                    if st_nxt in ("TARGET_HIT", "INVALIDATED"):
                        final_st = st_nxt
                        resolved_ts = nxt_ts if nxt_ts > 0.0 else prev_ts
                        break
                    if normalize_side(nxt.get("side")) != side_k:
                        final_st = "CLOSED_EARLY"
                        resolved_ts = nxt_ts if nxt_ts > 0.0 else prev_ts
                        break
                    if nxt_ts > prev_ts:
                        prev_ts = nxt_ts
                if final_st is None:
                    spot_px_cached = _resolve_live_price_for_symbol(
                        prices_map, str(row.get("venue") or ""), str(row.get("symbol") or "")
                    )
                    st = evaluate_trade_plan_path_status(
                        side_k,
                        spot_px_cached,
                        row.get("invalidation_px"),
                        row.get("target_px"),
                        start_ts=prev_ts,
                        end_ts=eff_now + 60.0,
                        price_bars=sym_bars,
                        check_spot=(spot_px_cached is not None and spot_px_cached > 0.0),
                    )
                    final_st = st if st in ("TARGET_HIT", "INVALIDATED", "ACTIVE") else "ACTIVE"
                    if final_st in ("TARGET_HIT", "INVALIDATED"):
                        resolved_ts = eff_now
                if (
                    newly_resolved_out is not None
                    and final_st in TERMINAL_TRADE_PLAN_STATUSES
                    and (final_st != "CLOSED_EARLY" or bool(sym_bars))
                    and (row_iid := clean_scalar_str(row.get("intent_id")))
                ):
                    newly_resolved_out[row_iid] = final_st
            status_counts[final_st] = status_counts.get(final_st, 0) + 1
            by_side[side_k][final_st] = by_side[side_k].get(final_st, 0) + 1
            if row_outcomes_out is not None:
                within_hor_st: Optional[str] = None
                if hor_h is not None and entry_ts > 0.0:
                    hor_end_ts = entry_ts + hor_h * 3600.0
                    if sym_bars:
                        in_win = eff_now <= hor_end_ts
                        sp = spot_px_cached if spot_px_cached is not None else (
                            _resolve_live_price_for_symbol(
                                prices_map, str(row.get("venue") or ""), str(row.get("symbol") or "")
                            ) if in_win else None
                        )
                        h_st = evaluate_trade_plan_path_status(
                            side_k,
                            sp if in_win else None,
                            row.get("invalidation_px"),
                            row.get("target_px"),
                            start_ts=entry_ts,
                            end_ts=hor_end_ts,
                            price_bars=sym_bars,
                            check_spot=(in_win and sp is not None and sp > 0.0),
                        )
                        within_hor_st = h_st if h_st in ("TARGET_HIT", "INVALIDATED") else ("EXPIRED" if eff_now > hor_end_ts else "ACTIVE")
                    elif final_st in ("TARGET_HIT", "INVALIDATED") and (resolved_ts is None or resolved_ts <= hor_end_ts):
                        within_hor_st = final_st
                    else:
                        within_hor_st = "EXPIRED" if eff_now > hor_end_ts else final_st
                row_outcomes_out.append({
                    "confidence": row.get("confidence"),
                    "horizon_h": hor_h,
                    "status": final_st,
                    "within_horizon_status": within_hor_st,
                })
    return status_counts, by_side, planned_n


def evaluate_ai_calibration_feedback(
    intents: list[dict[str, Any]],
    *,
    now_ts: float,
    live_prices: Optional[Mapping[str, Any]] = None,
    price_bars_by_symbol: Optional[Mapping[str, Sequence[Any]]] = None,
    lookback_days: float = 7.0,
    scope: str = "all",
    newly_resolved_out: Optional[dict[str, str]] = None,
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
) -> dict[str, Any]:
    """Compute 0ms closed-loop BUY/SELL trade-plan calibration and per-venue execution TCA for Section 3C."""
    eff_now = resolve_now_ts(now_ts)
    lb_days = safe_pos_float(lookback_days, default=7.0) or 7.0
    cutoff = eff_now - (lb_days * 86400.0)
    eff_scope = (clean_scalar_str(scope) or "all").lower()
    prices_map = live_prices if isinstance(live_prices, Mapping) else {}
    items = intents if isinstance(intents, list) else []

    empty_side = lambda: {"TARGET_HIT": 0, "INVALIDATED": 0, "CLOSED_EARLY": 0, "ACTIVE": 0}
    status_counts = empty_side()
    by_side = {"BUY": empty_side(), "SELL": empty_side()}
    if not items:
        return {
            "lookback_days": lb_days,
            "filled_total": 0,
            "planned_filled": 0,
            "status_counts": status_counts,
            "by_side": by_side,
            "resolved_hit_rate_pct": None,
            "venue_tca": {},
            "venue_order_type_tca": {},
            "limit_offset_fill_rates": {},
        }

    canceled_oids = _collect_canceled_order_ids(items)
    filled_rows: list[dict[str, Any]] = []
    unfilled_counts: dict[tuple[str, str], int] = {}
    unfilled_limit_offsets_by_venue: dict[str, list[float]] = {}
    all_unfilled_lim_offs: list[float] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        if (clean_upper_scalar(it.get("status")) or "") != "EXECUTED":
            continue
        sym = clean_scalar_str(it.get("symbol")) or ""
        ven = normalize_venue_name(clean_scalar_str(it.get("venue")) or "")
        if not sym:
            continue
        if eff_scope != "all" and not matches_asset_scope(sym, ven, eff_scope):
            continue
        _, base_ts = extract_intent_timestamp_s(it)
        exec_ts = _extract_intent_exec_ts(it, max_ts=eff_now + 60.0)
        ref_ts = exec_ts if exec_ts > 0.0 else base_ts
        if ref_ts <= 0.0 or ref_ts < cutoff or ref_ts > eff_now + 60.0:
            continue
        ot = extract_intent_order_type(it)
        if not _is_filled_ai_intent(it, canceled_oids):
            if _is_terminal_or_stale_unfilled_intent(it, canceled_oids, now_ts=eff_now, ref_ts=ref_ts):
                unf_ven = ven or "unknown"
                unf_ot = "LIMIT" if ot == "LIMIT" else "MARKET"
                k_unf = (unf_ven, unf_ot)
                unfilled_counts[k_unf] = unfilled_counts.get(k_unf, 0) + 1
                if unf_ot == "LIMIT":
                    unf_off = extract_limit_passive_offset_bps(it, order_type="LIMIT")
                    if unf_off is not None:
                        unfilled_limit_offsets_by_venue.setdefault(unf_ven, []).append(unf_off)
                        all_unfilled_lim_offs.append(unf_off)
            continue
        side = extract_intent_side(it)
        if side not in ("BUY", "SELL"):
            continue
        px = _extract_intent_exec_price(it)
        if px <= 0.0:
            continue
        filled_rows.append(
            build_normalized_filled_ai_intent_row(
                it,
                venue=ven,
                symbol=sym,
                side=side,
                ts_s=ref_ts,
                px=px,
                fee_resolver=fee_resolver,
                include_slippage=True,
            )
        )

    status_counts, by_side, planned_filled = evaluate_filled_trade_plans(
        filled_rows,
        prices_map,
        price_bars_by_symbol=price_bars_by_symbol,
        newly_resolved_out=newly_resolved_out,
        now_ts=eff_now,
    )
    resolved_n = status_counts["TARGET_HIT"] + status_counts["INVALIDATED"]
    hit_rate = round((status_counts["TARGET_HIT"] / resolved_n) * 100.0, 1) if resolved_n > 0 else None
    by_venue, by_vot = summarize_execution_quality_by_venue_and_order_type(
        filled_rows,
        unfilled_counts=unfilled_counts,
        unfilled_limit_offsets_by_venue=unfilled_limit_offsets_by_venue,
    )
    all_filled_lim_offs = [
        off_v for r in filled_rows
        if (clean_upper_scalar(r.get("order_type")) or "") == "LIMIT"
        and (off_v := safe_float(r.get("limit_passive_offset_bps"), default=None)) is not None
    ]
    lim_offset_fill_rates = summarize_limit_passive_offset_fill_rates(all_filled_lim_offs, all_unfilled_lim_offs)
    return {
        "lookback_days": lb_days,
        "filled_total": len(filled_rows),
        "planned_filled": planned_filled,
        "status_counts": status_counts,
        "by_side": by_side,
        "resolved_hit_rate_pct": hit_rate,
        "venue_tca": by_venue,
        "venue_order_type_tca": by_vot,
        "limit_offset_fill_rates": lim_offset_fill_rates,
    }


def format_ai_calibration_feedback_line(calibration: Any) -> str:
    """Format a compact single-line objective summary of trade-plan calibration and venue TCA for Section 3C."""
    if not isinstance(calibration, Mapping):
        return ""
    lb_days = safe_pos_float(calibration.get("lookback_days"), default=7.0) or 7.0
    planned_n = safe_nonneg_int(calibration.get("planned_filled"), default=0)
    by_side = calibration.get("by_side") if isinstance(calibration.get("by_side"), Mapping) else {}
    venue_tca = calibration.get("venue_tca") if isinstance(calibration.get("venue_tca"), Mapping) else {}
    segments: list[str] = []

    if planned_n > 0:
        side_parts: list[str] = []
        res_by_side: dict[str, int] = {}
        for sk in ("BUY", "SELL"):
            sc = by_side.get(sk)
            if not isinstance(sc, Mapping):
                continue
            t_n = safe_nonneg_int(sc.get("TARGET_HIT"), default=0)
            i_n = safe_nonneg_int(sc.get("INVALIDATED"), default=0)
            e_n = safe_nonneg_int(sc.get("CLOSED_EARLY"), default=0)
            a_n = safe_nonneg_int(sc.get("ACTIVE"), default=0)
            res_by_side[sk] = t_n + i_n
            if (t_n + i_n + e_n + a_n) > 0:
                side_parts.append(f"{sk}:{t_n}hit/{i_n}inv/{e_n}early/{a_n}open")
        hit_r = safe_float(calibration.get("resolved_hit_rate_pct"), default=None)
        if hit_r is not None:
            b_res, s_res = res_by_side.get("BUY", 0), res_by_side.get("SELL", 0)
            only_sfx = (
                "[SELL_only]"
                if (s_res > 0 and b_res == 0 and len(side_parts) > 1)
                else ("[BUY_only]" if (b_res > 0 and s_res == 0 and len(side_parts) > 1) else "")
            )
            side_parts.append(f"hit={hit_r:.0f}%{only_sfx}")
        inner = f" ({', '.join(side_parts)})" if side_parts else ""
        segments.append(f"plans={planned_n}{inner}")

    venue_parts: list[str] = []
    for ven, vt in sorted(venue_tca.items()):
        if not isinstance(vt, Mapping):
            continue
        lim_f = safe_nonneg_int(vt.get("limit_filled"), default=0)
        lim_u = safe_nonneg_int(vt.get("limit_unfilled"), default=0)
        mkt_f = safe_nonneg_int(vt.get("market_filled"), default=0)
        if (lim_f + lim_u + mkt_f) <= 0:
            continue
        v_bits: list[str] = []
        if (lim_f + lim_u) > 0:
            fr = safe_float(vt.get("limit_fill_rate_pct"), default=None)
            lim_hdr = f"LIM {lim_f}/{lim_f + lim_u}[{fr:.0f}%]" if (lim_u > 0 and fr is not None) else f"LIM {lim_f}"
            lim_s = safe_float(vt.get("avg_limit_slippage_bps"), default=None)
            arr_s = safe_float(vt.get("avg_arrival_shortfall_bps"), default=None)
            if lim_s is not None:
                lim_hdr += f" slip_bps={fmt_signed(lim_s, 1)}"
            if arr_s is not None:
                lim_hdr += f" arr_bps={fmt_signed(arr_s, 1)}"
            unf_off = safe_float(vt.get("med_unfilled_offset_bps"), default=None)
            if lim_u >= 3 and unf_off is not None:
                lim_hdr += f" unfilled_off_bps={fmt_signed(unf_off, 0)}"
            v_bits.append(lim_hdr)
        if mkt_f > 0:
            mkt_s = safe_float(vt.get("avg_market_slippage_bps"), default=None)
            mkt_hdr = f"MKT {mkt_f}" + (f" slip_bps={fmt_signed(mkt_s, 1)}" if mkt_s is not None else "")
            v_bits.append(mkt_hdr)
        if v_bits:
            venue_parts.append(f"{ven}: {' | '.join(v_bits)}")
    if venue_parts:
        segments.append(f"venue_tca({', '.join(venue_parts)})")

    lim_off_map = calibration.get("limit_offset_fill_rates")
    if isinstance(lim_off_map, Mapping):
        off_tot = 0
        off_unf = 0
        off_parts: list[str] = []
        for b_key, default_lbl, _, _ in _LIMIT_PASSIVE_OFFSET_BUCKETS:
            bkt = lim_off_map.get(b_key)
            if not isinstance(bkt, Mapping):
                continue
            b_f = safe_nonneg_int(bkt.get("filled"), default=0)
            b_u = safe_nonneg_int(bkt.get("unfilled"), default=0)
            b_t = b_f + b_u
            if b_t <= 0:
                continue
            off_tot += b_t
            off_unf += b_u
            b_fr = safe_float(bkt.get("fill_rate_pct"), default=None)
            if b_fr is None:
                b_fr = (b_f / b_t) * 100.0
            lbl = clean_scalar_str(bkt.get("label")) or default_lbl
            off_parts.append(f"{lbl}:{b_fr:.0f}%[{b_f}/{b_t}]")
        if off_parts and off_tot >= 5 and off_unf >= 1:
            segments.append(f"fill_by_offset({', '.join(off_parts)})")

    if not segments:
        return ""
    return f"AI Execution & Plan Calibration ({lb_days:g}d): {'; '.join(segments)}"


__all__ = [
    "build_intent_live_price_map",
    "build_normalized_filled_ai_intent_row",
    "build_persisted_round_trip_fee_resolver",
    "compute_signed_price_edge_pct",
    "evaluate_ai_calibration_feedback",
    "evaluate_ai_decision_scorecard",
    "evaluate_ai_ping_pong_guard",
    "evaluate_ai_target_ev_guard",
    "evaluate_ai_tokenized_equity_sell_guard",
    "evaluate_filled_trade_plans",
    "extract_fill_execution_slippage",
    "extract_limit_passive_offset_bps",
    "extract_live_prices_from_cache_payload",
    "format_ai_calibration_feedback_line",
    "format_ai_decision_scorecard_line",
    "format_signed_usd",
    "match_filled_intent_rows_lifo",
    "match_filled_intent_rows_lifo_in_window",
    "resolve_matched_notional_usd",
    "resolve_venue_flip_min_edge_pct",
    "resolve_venue_round_trip_fee_bps",
    "summarize_execution_quality_bucket",
    "summarize_execution_quality_by_venue_and_order_type",
    "summarize_limit_passive_offset_fill_rates",
]
