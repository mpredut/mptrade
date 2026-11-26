#!/usr/bin/env python3
"""
order_manager.py -- SPCX order-placement logic:
  * calculate quantity from a RON budget;
  * persist the exact intent before the non-idempotent POST;
  * perform one bounded submit/reconciliation step per invocation;
  * recover a lost response from active orders or portfolio delta;
  * never sleep or poll inside placement.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HERE)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from botcore import (  # noqa: E402
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    ensure_local_module_priority,
    extract_order_id,
    extract_order_side,
    first_pos_float,
    log,
    optional_env,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    sanitize_identifier,
)
ensure_local_module_priority(_HERE, "ipo_common", "market_data", "t212_broker_lifecycle", "t212_client")

from cachepaths import canonical_base_asset, same_canonical_base_asset as _t212_tickers_match  # noqa: E402
from ipo_common import notify  # noqa: E402
from notify_engine.notifyapi import emit_trade_notification  # noqa: E402
from providers.order_ledger import OrderEventType, get_default_ledger_writer  # noqa: E402
from market_data import get_price_usd, get_usd_ron  # noqa: E402
from state_io import encode_json_payload, load_json_state, save_json_state  # noqa: E402
from t212_broker_lifecycle import t212_active_order_matches, t212_portfolio_position  # noqa: E402
from t212_client import T212Client  # noqa: E402

T212_ORDER_TERMINAL = {"FILLED", "CANCELLED", "CANCELED", "EXPIRED", "REJECTED"}
T212_ORDER_OPEN = {
    "LOCAL", "UNCONFIRMED", "CONFIRMED", "NEW", "OPEN", "PENDING",
    "PARTIALLY_FILLED",
}
ABSENCE_CONFIRMATIONS = 2


# ---------------------------------------------------------------------------
# Quantity
# ---------------------------------------------------------------------------
def resolve_quantity(order_price: float,
                     order_qty: float | None,
                     order_budget_ron: float | None) -> float | None:
    """Resolve share quantity from fixed ORDER_QTY or a RON budget at the current FX rate."""
    qty_val = safe_pos_float(order_qty, None)
    if qty_val is not None:
        return qty_val
    budget = safe_pos_float(order_budget_ron, None)
    if budget is not None:
        price = safe_pos_float(order_price, None)
        rate = safe_pos_float(get_usd_ron(), None) if price is not None else None
        if price is None or rate is None:
            return None
        qty = budget / (price * rate)
        log(f"  [ORDER] {order_budget_ron} RON / ({order_price} USD \u00d7 {rate:.2f}) = {qty:.4f} shares")
        if qty < 1:
            log(f"  ! [ORDER] budget < the price of one share (~{price*rate:.0f} RON) "
                f"-> a FRACTIONAL order ({qty:.4f}). T212 may refuse fractional orders on a new instrument.")
        return qty
    return None


# ---------------------------------------------------------------------------
# Duplicate-prevention marker
# ---------------------------------------------------------------------------
def _marker_path(ticker: str, *, test: bool = False) -> str:
    """Namespace one-shot state by profile and ticker, never across accounts."""
    profile = sanitize_identifier(optional_env("IPO_PROFILE") or "default", sep="_", allow_dots_hyphens=True, strip_chars="._")
    raw_ticker = clean_upper_scalar(ticker) or ""
    symbol = sanitize_identifier(raw_ticker, sep="_", allow_dots_hyphens=True, strip_chars="._")
    suffix = ".test" if test else ""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f".t212_order.{profile}.{symbol}{suffix}.json",
    )


def _persist_marker(record: dict, path: str) -> None:
    try:
        save_json_state(path, record, indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot persist the T212 intent in {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Order-status polling
# ---------------------------------------------------------------------------
def _pos_abs_order_float(val: object) -> float | None:
    num = safe_float(val, default=None)
    return abs(num) if (num is not None and abs(num) > 0.0) else None


def _extract_t212_terminal_fill(info: dict, status: str) -> tuple[float, float]:
    """Return ``(filled_qty, fill_price)`` for a terminal T212 order without conflating ordered qty on cancel/expiry."""
    fq_abs = _pos_abs_order_float(info.get("filledQuantity"))
    if fq_abs is not None:
        fq = fq_abs
    elif status == "FILLED" and safe_float(info.get("filledQuantity"), default=None) is None:
        fq = (
            _pos_abs_order_float(info.get("quantity"))
            or _pos_abs_order_float(info.get("orderedQuantity"))
            or 0.0
        )
    else:
        fq = 0.0
    fval = _pos_abs_order_float(info.get("filledValue"))
    derived_px = (fval / fq) if (fval is not None and fq > 0) else None
    fp = first_pos_float(
        _pos_abs_order_float(info.get("fillPrice")),
        _pos_abs_order_float(info.get("averagePrice")),
        derived_px,
        _pos_abs_order_float(info.get("limitPrice")),
        default=0,
    )
    return fq, fp


def _emit_t212_order_notify(
    status: str, side: str, ticker: str, px: float, qty: float, order_id,
    *, desktop: bool, notify_price=None, detail: str | None = None,
) -> None:
    kw = {"source": "T212 order", "desktop": desktop}
    if notify_price is not None:
        kw["price"] = notify_price
    emit_trade_notification(
        notify, status, side, ticker, px, qty, qty * px, "USD", order_id,
        venue="t212", owner="order_manager", detail=detail, notify_kwargs=kw,
    )


def poll_order_until_terminal(client: T212Client, order_id, ticker: str,
                              desktop: bool = False) -> dict | bool | None:
    """Observe one status snapshot and return immediately."""
    info = client.get_order_status(order_id)
    if info is False:
        log(f"  [ORDER] status {order_id} absent from active and history endpoints")
        return False
    if not info:
        log(f"  [ORDER] status {order_id} unavailable - checking again on the next cycle")
        return None
    st = str(info.get("status") or "").upper()
    log(f"  [ORDER] status {order_id}: {st or 'UNKNOWN'}")
    if st not in T212_ORDER_TERMINAL:
        return info
    fq, fp = _extract_t212_terminal_fill(info, st)
    raw_q = safe_float(
        info.get("quantity") if info.get("quantity") is not None else info.get("orderedQuantity"),
        default=None,
    )
    side_in = extract_order_side(info) or ("SELL" if (raw_q is not None and raw_q < 0) else "BUY")
    notify_px = safe_pos_float(fp, 0.0)
    if st == "FILLED":
        log(f"  [ORDER] \u2713 FILLED: {fq} @ {fp} USD")
        _emit_t212_order_notify("FILLED", side_in, ticker, notify_px, fq, order_id, desktop=desktop, notify_price=notify_px)
    elif fq > 0:
        log(f"  [ORDER] \u25d0 PARTIAL {st}: {fq} @ {fp} USD")
        st_verb = "EXPIRED" if st == "EXPIRED" else "CANCELED"
        _emit_t212_order_notify(
            st_verb, side_in, ticker, notify_px, fq, order_id,
            desktop=desktop, notify_price=notify_px, detail="partial fill",
        )
    else:
        log(f"  [ORDER] \u2717 {st}")
        st_verb = "CANCELED" if st == "CANCELLED" else st
        ord_q = _pos_abs_order_float(info.get("quantity")) or _pos_abs_order_float(info.get("orderedQuantity")) or 0.0
        _emit_t212_order_notify(st_verb, side_in, ticker, notify_px, ord_q, order_id, desktop=desktop)
    return info


def _portfolio_qty(client: T212Client, ticker: str) -> float | None:
    pos = t212_portfolio_position(client, ticker)
    return pos[0] if pos is not None else None


def _matching_active_orders(client: T212Client, record: dict) -> list[dict] | None:
    orders = client.list_active_orders()
    if orders is None:
        return None
    if safe_pos_float(record.get("qty"), None) is None or safe_pos_float(record.get("limit_price"), None) is None:
        return None
    return [
        order
        for order in orders
        if t212_active_order_matches(
            order,
            target_ticker=record.get("ticker") or "",
            expected_side="BUY",
            expected_qty=record.get("qty"),
            expected_limit=record.get("limit_price"),
        )
    ]


def _confirm_fill_from_portfolio(
    client: T212Client,
    record: dict,
    marker_path: str,
    *,
    now: float,
    message: str,
) -> str | None:
    current_qty = _portfolio_qty(client, record["ticker"])
    if current_qty is None:
        return "waiting"
    before_f = safe_nonneg_float(record.get("before_qty"), None)
    if before_f is not None and current_qty > before_f + 1e-6:
        record.update({
            "lifecycle": "filled", "portfolio_fill_observed": True,
            "filled_qty": current_qty - before_f,
            "lookup_misses": 0, "last_observed_at": now,
        })
        _persist_marker(record, marker_path)
        log(message)
        return "complete"
    return None


def _classify_t212_venue_status(status: str, *, post_submit: bool = False) -> str:
    """Map an upper-cased T212 venue order status to a marker lifecycle token."""
    if status == "FILLED":
        return "filled"
    if status in {"CANCELLED", "CANCELED", "EXPIRED"}:
        return "expired" if status == "EXPIRED" else "canceled"
    if status == "REJECTED":
        return "submit_pending"
    if post_submit:
        return "accepted"
    if status in T212_ORDER_OPEN:
        return "open"
    # An undocumented status is not proof of either absence or terminality.
    return "status_unknown"


def _reconcile_record(client: T212Client, record: dict, marker_path: str,
                      *, now: float, retry_delay: float,
                      desktop: bool) -> str:
    """Return complete, terminal_failed, waiting, or retryable."""
    order_id = clean_order_id(record.get("order_id"))
    if order_id is not None:
        info = poll_order_until_terminal(client, order_id, record["ticker"], desktop)
        if info is None:
            return "waiting"
        if info is False:
            pf_outcome = _confirm_fill_from_portfolio(
                client,
                record,
                marker_path,
                now=now,
                message="  [ORDER] accepted order vanished from history; fill confirmed from portfolio",
            )
            if pf_outcome is not None:
                return pf_outcome
            misses = safe_nonneg_int(record.get("lookup_misses"), default=0) + 1
            record["lookup_misses"] = misses
            record["last_observed_at"] = now
            if misses >= ABSENCE_CONFIRMATIONS:
                record["lifecycle"] = "canceled"
                record["venue_status"] = "ABSENT"
                _persist_marker(record, marker_path)
                log(f"  [ORDER] \u2717 accepted order {order_id} definitively absent without portfolio growth")
                return "terminal_failed"
            _persist_marker(record, marker_path)
            return "waiting"
        record["lookup_misses"] = 0
        status = str(info.get("status") or "").upper()
        lifecycle = _classify_t212_venue_status(status, post_submit=False)
        record["venue_status"] = status
        record["last_observed_at"] = now
        record["lifecycle"] = lifecycle
        if lifecycle in {"filled", "canceled", "expired"}:
            fq, fp = _extract_t212_terminal_fill(info, status)
            if fq > 0:
                record["filled_qty"] = fq
                if fp > 0:
                    record["fill_price"] = fp
        if status == "REJECTED":
            raw_hist = record.get("venue_order_ids")
            history = list(raw_hist) if isinstance(raw_hist, (list, tuple)) else []
            if str(order_id) not in history:
                history.append(str(order_id))
            record.update({
                "order_id": None,
                "venue_order_ids": history,
                "lookup_misses": 0, "next_retry_at": now + retry_delay,
            })
        _persist_marker(record, marker_path)
        if lifecycle == "filled":
            return "complete"
        if lifecycle in {"canceled", "expired"}:
            return "terminal_failed"
        return "waiting"

    matches = _matching_active_orders(client, record)
    if matches is None:
        return "waiting"
    matched_oid = extract_order_id(matches[0]) if len(matches) == 1 else None
    if matched_oid is not None:
        record.update({
            "order_id": matched_oid, "lifecycle": "accepted",
            "lookup_misses": 0, "last_observed_at": now,
        })
        _persist_marker(record, marker_path)
        log(f"  [ORDER] lost response recovered: id={record['order_id']}")
        return "waiting"
    if len(matches) > 1:
        log("  ! [ORDER] " + "several T212 orders match the intent \u2014 keeping it pending")
        return "waiting"

    pf_outcome = _confirm_fill_from_portfolio(
        client,
        record,
        marker_path,
        now=now,
        message="  [ORDER] the submit was recovered from the growth of the T212 position",
    )
    if pf_outcome is not None:
        return pf_outcome

    misses = safe_nonneg_int(record.get("lookup_misses"), default=0) + 1
    record["lookup_misses"] = misses
    record["last_observed_at"] = now
    next_retry_at = safe_float(record.get("next_retry_at"), None)
    retry_d = safe_nonneg_float(retry_delay, default=0.0)
    if next_retry_at is None or next_retry_at > now + retry_d + 60.0:
        next_retry_at = now + retry_d
        record["next_retry_at"] = next_retry_at
    _persist_marker(record, marker_path)
    if misses < ABSENCE_CONFIRMATIONS or now < next_retry_at:
        return "waiting"
    return "retryable"


# ---------------------------------------------------------------------------
# Order placement with retry, price validation, and marker
# ---------------------------------------------------------------------------
def place_order_with_retry(
    client: T212Client,
    ticker: str,
    quantity: float,
    limit_price: float,
    validity: str,
    dry_run: bool,
    desktop: bool = False,
    retry_delay: int = 60,
    write_marker: bool = True,
) -> bool:
    """Perform one non-blocking lifecycle step for a persistent LIMIT BUY.

    ``True`` means the submit is accepted/recovered or an existing venue order remains
    owned. An ambiguous/pending submit returns ``False`` while its durable marker keeps
    it eligible for a later invocation. There is no in-process retry loop.
    """

    qty_f = safe_pos_float(quantity, default=None)
    price_f = safe_pos_float(limit_price, default=None)
    retry_delay_f = safe_nonneg_float(retry_delay, default=None)
    qty_r = round(qty_f, 2) if qty_f is not None else 0.0
    price_r = round(price_f, 2) if price_f is not None else 0.0
    param_err: str | None = None
    if (ticker := clean_scalar_str(ticker)) is None:
        param_err = "  ! [ORDER] ticker invalid"
    elif (validity := clean_upper_scalar(validity)) is None:
        param_err = "  ! [ORDER] validity invalid"
    else:
        if validity == "GTC":
            validity = "GOOD_TILL_CANCEL"
        if validity not in {"DAY", "GOOD_TILL_CANCEL"}:
            param_err = f"  ! [ORDER] validity invalid: {validity!r}"
        elif qty_f is None or price_f is None or retry_delay_f is None:
            param_err = "  ! [ORDER] invalid numeric parameters: bool not allowed"
        elif qty_r <= 0 or price_r <= 0:
            param_err = "  ! [ORDER] qty/price/retry_delay must be finite and valid"
    if param_err is not None:
        log(param_err)
        return False

    # Current price is informational only; do not change the user-selected limit.
    yahoo_sym = canonical_base_asset(ticker)
    current = get_price_usd(yahoo_sym)
    if current:
        log(f"  [ORDER] the current price of {yahoo_sym}: {current:.2f} USD  |  the limit: {price_r:.2f} USD")
        if price_r < current:
            log(f"  ! [ORDER] the limit {price_r} < the price {current:.2f} -> the order will sit waiting "
                f"(it only executes if the price falls to {price_r}).")

    if dry_run:
        log(f"  [ORDER] [DRY-RUN] LIMIT BUY {ticker}  qty={qty_r}  @ {price_r} USD  validity={validity}")
        log("  [ORDER] " + "Dry run \u2014 the order was NOT SENT. Set ORDER_EXECUTE=true in .env.")
        return True

    marker_path = _marker_path(ticker, test=not write_marker)
    now = time.time()
    try:
        record = load_json_state(marker_path, default_factory=lambda: None, fail_closed=True, label="T212 marker")
    except RuntimeError as exc:
        log(f"  ! [ORDER] unreadable T212 marker {marker_path}: {exc} \u2014 NOT sending without verifiable durable state")
        return False
    if record:
        # Legacy accepted markers stay blocking rather than silently creating a
        # second order whose identity cannot be reconstructed.
        ver_i = safe_int(record.get("version"), 1)
        if ver_i < 2:
            legacy = record.get("order") if isinstance(record.get("order"), dict) else {}
            legacy_id = extract_order_id(legacy)
            if legacy_id:
                info = poll_order_until_terminal(client, legacy_id, ticker, desktop)
                record["last_status"] = info.get("status") if isinstance(info, dict) else None
                _persist_marker(record, marker_path)
            log(f"  [ORDER] an existing legacy marker (v{ver_i}) \u2014 not duplicating the order")
            return True
        rec_qty_f = safe_pos_float(record.get("qty"), -1.0)
        rec_lim_f = safe_pos_float(record.get("limit_price"), -1.0)
        same_intent = (
            _t212_tickers_match(record.get("ticker") or "", ticker)
            and rec_qty_f > 0
            and rec_lim_f > 0
            and abs(rec_qty_f - qty_r) <= 0.011
            and abs(rec_lim_f - price_r) <= 0.011
        )
        lifecycle = (clean_scalar_str(record.get("lifecycle")) or "").lower()
        if not same_intent or lifecycle in {"canceled", "cancelled", "expired"}:
            log(
                "  ! [ORDER] another intent already exists in the marker \u2014 NOT overwriting"
                if not same_intent
                else f"  [ORDER] the intent is terminal ({lifecycle}); not resending it"
            )
            return False
        if lifecycle == "filled":
            return True
        outcome = _reconcile_record(
            client, record, marker_path, now=now,
            retry_delay=retry_delay_f, desktop=desktop,
        )
        if outcome == "terminal_failed":
            return False
        if outcome == "complete":
            return True
        if outcome == "waiting":
            # Do not report success merely because an unproven intent was durably
            # queued. A known venue order may safely remain open/status-unknown.
            return clean_order_id(record.get("order_id")) is not None
    else:
        before_qty = _portfolio_qty(client, ticker)
        if before_qty is None:
            log("  ! [ORDER] " + "the T212 portfolio is unavailable \u2014 NOT sending")
            return False
        record = {
            "version": 2,
            "intent_id": f"t212-one-shot-{ticker}-{uuid.uuid4().hex}",
            "ticker": ticker,
            "side": "BUY",
            "qty": qty_r,
            "limit_price": price_r,
            "validity": validity,
            "before_qty": before_qty,
            "created_at": now,
            "attempts": 0,
            "lookup_misses": 0,
            "lifecycle": "submit_pending",
            "next_retry_at": now,
        }
        active_matches = _matching_active_orders(client, record)
        if active_matches is None or active_matches:
            log(
                "  ! [ORDER] the active T212 orders are unavailable \u2014 NOT sending"
                if active_matches is None
                else "  ! [ORDER] an active order with the same ticker/side/qty/price already exists; "
                "not claiming it and NOT sending another"
            )
            return False

    att_i = safe_nonneg_int(record.get("attempts"), default=0)
    record["attempts"] = att_i + 1
    record["last_submit_at"] = now
    record["next_retry_at"] = now + retry_delay_f
    record["lookup_misses"] = 0
    record["lifecycle"] = "submit_pending"
    _persist_marker(record, marker_path)  # durability boundary before POST

    log(f"  [ORDER] LIMIT BUY {ticker}  qty={qty_r}  @ {price_r} USD  validity={validity}")
    ledger = get_default_ledger_writer()
    t212_iid = clean_order_id(record.get("intent_id"))
    ledger.record_event(
        OrderEventType.SUBMITTED,
        venue="t212",
        symbol=ticker,
        owner="t212_one_shot",
        side=record.get("side") or "BUY",
        order_type="LIMIT",
        qty=qty_r,
        px=price_r,
        iid=t212_iid,
        kind="t212_one_shot",
        src="t212_order_manager",
    )
    try:
        status, data = client.place_limit_order(ticker, qty_r, price_r, validity)
    except Exception as exc:  # response may have been lost after venue acceptance
        record["submit_error"] = f"{exc.__class__.__name__}: {exc}"
        _persist_marker(record, marker_path)
        ledger.record_event(
            OrderEventType.SUBMIT_UNKNOWN,
            venue="t212",
            symbol=ticker,
            owner="t212_one_shot",
            side=record.get("side") or "BUY",
            order_type="LIMIT",
            qty=qty_r,
            px=price_r,
            iid=t212_iid,
            reason=record["submit_error"],
            src="t212_order_manager",
        )
        log(f"  ! [ORDER] ambiguous submit response ({exc}) \u2014 intent stays pending")
        return False

    oid = extract_order_id(data) if isinstance(data, dict) else None
    if status in (200, 201) and oid is not None:
        venue_status = str(data.get("status") or "").upper()
        lifecycle = _classify_t212_venue_status(venue_status, post_submit=True)
        record.update({
            "order_id": None if venue_status == "REJECTED" else oid,
            "venue_status": venue_status,
            "lifecycle": lifecycle, "order": data,
        })
        if venue_status == "REJECTED":
            history = list(raw_hist) if isinstance(raw_hist := record.get("venue_order_ids"), (list, tuple)) else []
            record["venue_order_ids"] = history if str(oid) in history else [*history, str(oid)]
        _persist_marker(record, marker_path)
        ledger.record_event(
            OrderEventType.SUBMIT_REFUSED if venue_status == "REJECTED" else OrderEventType.ACKED,
            venue="t212",
            symbol=ticker,
            owner="t212_one_shot",
            side=record.get("side") or "BUY",
            order_type="LIMIT",
            qty=qty_r,
            px=price_r,
            iid=t212_iid,
            vid=oid,
            vst=venue_status or "NEW",
            src="t212_order_manager",
        )
        log(f"  [ORDER] submit response: id={oid}  status={venue_status or 'UNKNOWN'}")
        if lifecycle in {"canceled", "expired", "submit_pending"}:
            if lifecycle == "submit_pending":
                log("  ! [ORDER] order rejected; the intent stays queued for reconciliation/retry")
            return False
        _emit_t212_order_notify(
            "PLACED", record.get("side") or "BUY", ticker,
            safe_pos_float(price_r, 0.0), safe_pos_float(qty_r, 0.0), oid,
            desktop=desktop, notify_price=price_r, detail="LIMIT",
        )
        return True

    record["submit_error"] = f"T212 HTTP {status}: {encode_json_payload(data, default=str)[:400]}"
    _persist_marker(record, marker_path)
    ledger.record_event(
        OrderEventType.SUBMIT_REFUSED if (safe_int(status, 0) in range(400, 500)) else OrderEventType.SUBMIT_UNKNOWN,
        venue="t212",
        symbol=ticker,
        owner="t212_one_shot",
        side=record.get("side") or "BUY",
        order_type="LIMIT",
        qty=qty_r,
        px=price_r,
        iid=t212_iid,
        reason=record["submit_error"],
        src="t212_order_manager",
    )
    log(f"  ! [ORDER] {record['submit_error']} \u2014 non-blocking retry on the next cycle")
    return False
