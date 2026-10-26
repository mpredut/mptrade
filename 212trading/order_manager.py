#!/usr/bin/env python3
"""
order_manager.py — SPCX order-placement logic:
  * calculate quantity from a RON budget;
  * persist the exact intent before the non-idempotent POST;
  * perform one bounded submit/reconciliation step per invocation;
  * recover a lost response from active orders or portfolio delta;
  * never sleep or poll inside placement.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from state_io import atomic_write_json

try:
    from .ipo_common import log, now_str
    from .ipo_notify import notify
    from .market_data import get_usd_ron, get_price_usd, t212_to_yahoo
    from .t212_client import T212Client
except ImportError:
    from ipo_common import log, now_str
    from ipo_notify import notify
    from market_data import get_usd_ron, get_price_usd, t212_to_yahoo
    from t212_client import T212Client

# Legacy marker retained only for conservative migration of pre-v2 SPCX intents.
ORDER_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spcx_order_placed")

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
    if order_qty is not None and not isinstance(order_qty, bool):
        try:
            qty_val = float(order_qty)
        except (TypeError, ValueError, OverflowError):
            qty_val = 0.0
        if math.isfinite(qty_val) and qty_val > 0:
            return qty_val
    if order_budget_ron is not None and not isinstance(order_budget_ron, bool):
        if isinstance(order_price, bool):
            return None
        try:
            budget = float(order_budget_ron)
            price = float(order_price)
            rate = float(get_usd_ron() or 0.0)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(budget) and budget > 0
                and math.isfinite(price) and price > 0
                and math.isfinite(rate) and rate > 0):
            return None
        qty = budget / (price * rate)
        log(f"  [ORDER] {order_budget_ron} RON / ({order_price} USD × {rate:.2f}) = {qty:.4f} actiuni")
        if qty < 1:
            log(f"  ! [ORDER] budget < the price of one share (~{price*rate:.0f} RON) "
                f"-> a FRACTIONAL order ({qty:.4f}). T212 may refuse fractional orders on a new instrument.")
        return qty
    return None


# ---------------------------------------------------------------------------
# Duplicate-prevention marker
# ---------------------------------------------------------------------------
def order_already_placed() -> bool:
    """Conservatively report whether the legacy marker blocks a new submit."""
    try:
        record = _read_marker(ORDER_MARKER)
    except RuntimeError as exc:
        log(f"  ! [ORDER] marker ilizibil — fail closed: {exc}")
        return True
    return bool(record and str(record.get("lifecycle") or "").lower() not in {
        "canceled", "cancelled", "expired",
    })


def _read_marker(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"marker T212 ilizibil {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"marker T212 invalid {path}: obiect JSON asteptat")
    return value


def _safe_marker_component(value: str) -> str:
    if isinstance(value, bool):
        return "unknown"
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return component.strip("._") or "unknown"


def _marker_path(ticker: str, *, test: bool = False) -> str:
    """Namespace one-shot state by profile and ticker, never across accounts."""
    profile = _safe_marker_component(os.environ.get("IPO_PROFILE") or "default")
    raw_ticker = "" if isinstance(ticker, bool) or not isinstance(ticker, str) else ticker.strip().upper()
    symbol = _safe_marker_component(raw_ticker)
    suffix = ".test" if test else ""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f".t212_order.{profile}.{symbol}{suffix}.json",
    )


def _read_or_migrate_marker(ticker: str, marker_path: str) -> dict | None:
    record = _read_marker(marker_path)
    if record is not None or marker_path == ORDER_MARKER:
        return record

    # The old global file was named for SPCX. Migrate it only when its payload
    # proves the same ticker; an unrelated/unknown marker must never be guessed.
    legacy = _read_marker(ORDER_MARKER)
    if legacy is None:
        return None
    raw_order = legacy.get("order")
    order_dict = raw_order if isinstance(raw_order, dict) else {}
    legacy_ticker = str(
        legacy.get("ticker")
        or order_dict.get("ticker")
        or ""
    ).upper()
    if legacy_ticker != ticker.upper():
        return None
    _persist_marker(legacy, marker_path)
    log(f"  [ORDER] marker legacy migrat conservator: {marker_path}")
    return legacy


def _persist_marker(record: dict, path: str) -> None:
    try:
        atomic_write_json(path, record, indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot persist the T212 intent in {path}: {exc}") from exc


def _write_marker(ticker: str, result: dict) -> None:
    """Backward-compatible accepted-order marker writer."""
    _persist_marker({
        "version": 2, "at": now_str(), "ticker": ticker,
        "lifecycle": "accepted", "order_id": result.get("id"),
        "venue_status": result.get("status"), "order": result,
    }, _marker_path(ticker))
    log(f"  [ORDER] marker written: {_marker_path(ticker)}")


# ---------------------------------------------------------------------------
# Order-status polling
# ---------------------------------------------------------------------------
def poll_order_until_terminal(client: T212Client, order_id, ticker: str,
                              desktop: bool = False) -> dict | bool | None:
    """Observe one status snapshot and return immediately."""
    info = client.get_order_status(order_id)
    if info is False:
        log(f"  [ORDER] status {order_id} absent from active and history endpoints")
        return False
    if not info:
        log(f"  [ORDER] status {order_id} unavailable — checking again on the next cycle")
        return None
    st = str(info.get("status") or "").upper()
    log(f"  [ORDER] status {order_id}: {st or 'NECUNOSCUT'}")
    if st not in T212_ORDER_TERMINAL:
        return info
    fq = info.get("filledQuantity") or info.get("quantity") or info.get("orderedQuantity", 0)
    fp = info.get("fillPrice") or info.get("limitPrice", 0)
    if st == "FILLED":
        log(f"  [ORDER] ✓ FILLED: {fq} @ {fp} USD")
        try:
            fp_f = 0.0 if isinstance(fp, bool) else float(fp or 0.0)
            notify_px = fp_f if math.isfinite(fp_f) and fp_f > 0 else 0.0
        except (TypeError, ValueError, OverflowError):
            notify_px = 0.0
        notify(title=f"✓ Order filled: {ticker}", body=f"q{fq} @ {fp}",
               source="T212 order", price=notify_px, desktop=desktop)
    else:
        log(f"  [ORDER] ✗ {st}")
        notify(title=f"✗ Order {st}: {ticker}", body=f"id={order_id}",
               source="T212 order", desktop=desktop)
    return info


def _portfolio_qty(client: T212Client, ticker: str) -> float | None:
    portfolio = client.get_portfolio()
    if portfolio is None:
        return None
    for position in portfolio:
        if not isinstance(position, dict):
            continue
        if str(position.get("ticker") or "").upper() == ticker.upper():
            raw_q = position.get("quantity")
            if isinstance(raw_q, bool):
                return None
            try:
                q_val = float(raw_q or 0.0)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(q_val) or q_val < 0.0:
                return None
            return q_val
    return 0.0


def _matching_active_orders(client: T212Client, record: dict) -> list[dict] | None:
    orders = client.list_active_orders()
    if orders is None:
        return None
    raw_exp_qty = record.get("qty")
    raw_exp_lim = record.get("limit_price")
    if isinstance(raw_exp_qty, bool) or isinstance(raw_exp_lim, bool):
        return None
    try:
        expected_qty = float(raw_exp_qty)
        expected_limit = float(raw_exp_lim)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(expected_qty) and expected_qty > 0 and math.isfinite(expected_limit) and expected_limit > 0):
        return None
    matches = []
    for order in orders:
        if not isinstance(order, dict):
            continue
        instrument = order.get("instrument")
        nested_ticker = instrument.get("ticker") if isinstance(instrument, dict) else None
        ticker = str(order.get("ticker") or nested_ticker or "").upper()
        if ticker != str(record.get("ticker") or "").upper():
            continue
        raw_sq = order.get("quantity", order.get("orderedQuantity", order.get("qty")))
        raw_lim = order.get("limitPrice", order.get("limit"))
        if isinstance(raw_sq, bool) or isinstance(raw_lim, bool):
            continue
        try:
            signed_qty = float(raw_sq)
            qty = abs(signed_qty)
            limit = float(raw_lim)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(qty) and math.isfinite(limit)):
            continue
        side = str(order.get("side") or "").upper()
        if not side:
            side = "BUY" if signed_qty > 0 else "SELL" if signed_qty < 0 else ""
        if side != "BUY" or qty <= 0:
            continue
        if abs(qty - expected_qty) > max(0.011, expected_qty * 1e-6):
            continue
        if abs(limit - expected_limit) > 0.011:
            continue
        matches.append(order)
    return matches


def _reconcile_record(client: T212Client, record: dict, marker_path: str,
                      *, now: float, retry_delay: float,
                      desktop: bool) -> str:
    """Return complete, terminal_failed, waiting, or retryable."""
    order_id = record.get("order_id")
    if order_id is not None and not isinstance(order_id, bool) and str(order_id).strip():
        info = poll_order_until_terminal(client, order_id, record["ticker"], desktop)
        if info is None:
            return "waiting"
        if info is False:
            before_qty = record.get("before_qty")
            current_qty = _portfolio_qty(client, record["ticker"])
            if current_qty is None:
                return "waiting"
            before_f = None
            if before_qty is not None and not isinstance(before_qty, bool):
                try:
                    cand_b = float(before_qty)
                    if math.isfinite(cand_b) and cand_b >= 0.0:
                        before_f = cand_b
                except (TypeError, ValueError, OverflowError):
                    before_f = None
            if before_f is not None and current_qty > before_f + 1e-6:
                record.update({
                    "lifecycle": "filled", "portfolio_fill_observed": True,
                    "filled_qty": current_qty - before_f,
                    "lookup_misses": 0, "last_observed_at": now,
                })
                _persist_marker(record, marker_path)
                log("  [ORDER] accepted order vanished from history; fill confirmed from portfolio")
                return "complete"
            raw_m = record.get("lookup_misses")
            try:
                misses = (0 if isinstance(raw_m, bool) else max(0, int(raw_m or 0))) + 1
            except (TypeError, ValueError, OverflowError):
                misses = 1
            record["lookup_misses"] = misses
            record["last_observed_at"] = now
            if misses >= ABSENCE_CONFIRMATIONS:
                record["lifecycle"] = "canceled"
                record["venue_status"] = "ABSENT"
                _persist_marker(record, marker_path)
                log(f"  [ORDER] ✗ accepted order {order_id} definitively absent without portfolio growth")
                return "terminal_failed"
            _persist_marker(record, marker_path)
            return "waiting"
        record["lookup_misses"] = 0
        status = str(info.get("status") or "").upper()
        record["venue_status"] = status
        record["last_observed_at"] = now
        if status == "FILLED":
            record["lifecycle"] = "filled"
        elif status in {"CANCELLED", "CANCELED", "EXPIRED"}:
            record["lifecycle"] = "expired" if status == "EXPIRED" else "canceled"
        elif status == "REJECTED":
            raw_hist = record.get("venue_order_ids")
            history = list(raw_hist) if isinstance(raw_hist, (list, tuple)) else []
            if str(order_id) not in history:
                history.append(str(order_id))
            record.update({
                "lifecycle": "submit_pending", "order_id": None,
                "venue_order_ids": history,
                "lookup_misses": 0, "next_retry_at": now + retry_delay,
            })
        elif status in T212_ORDER_OPEN:
            record["lifecycle"] = "open"
        else:
            # An undocumented status is not proof of either absence or terminality.
            record["lifecycle"] = "status_unknown"
        _persist_marker(record, marker_path)
        if record["lifecycle"] == "filled":
            return "complete"
        if record["lifecycle"] in {"canceled", "expired"}:
            return "terminal_failed"
        return "waiting"

    matches = _matching_active_orders(client, record)
    if matches is None:
        return "waiting"
    if len(matches) == 1 and matches[0].get("id") is not None and not isinstance(matches[0].get("id"), bool) and str(matches[0].get("id")).strip():
        record.update({
            "order_id": str(matches[0]["id"]).strip(), "lifecycle": "accepted",
            "lookup_misses": 0, "last_observed_at": now,
        })
        _persist_marker(record, marker_path)
        log(f"  [ORDER] raspuns pierdut recuperat: id={record['order_id']}")
        return "waiting"
    if len(matches) > 1:
        log("  ! [ORDER] several T212 orders match the intent — keeping it pending")
        return "waiting"

    before_qty = record.get("before_qty")
    current_qty = _portfolio_qty(client, record["ticker"])
    if current_qty is None:
        return "waiting"
    before_f = None
    if before_qty is not None and not isinstance(before_qty, bool):
        try:
            cand_b = float(before_qty)
            if math.isfinite(cand_b) and cand_b >= 0.0:
                before_f = cand_b
        except (TypeError, ValueError, OverflowError):
            before_f = None
    if before_f is not None and current_qty is not None and current_qty > before_f + 1e-6:
        record.update({
            "lifecycle": "filled", "portfolio_fill_observed": True,
            "filled_qty": current_qty - before_f,
            "last_observed_at": now,
        })
        _persist_marker(record, marker_path)
        log("  [ORDER] the submit was recovered from the growth of the T212 position")
        return "complete"

    raw_m = record.get("lookup_misses")
    try:
        misses = (0 if isinstance(raw_m, bool) else max(0, int(raw_m or 0))) + 1
    except (TypeError, ValueError, OverflowError):
        misses = 1
    record["lookup_misses"] = misses
    record["last_observed_at"] = now
    raw_nra = record.get("next_retry_at")
    try:
        next_retry_at = 0.0 if isinstance(raw_nra, bool) else float(raw_nra or 0.0)
    except (TypeError, ValueError, OverflowError):
        next_retry_at = 0.0
    if not math.isfinite(next_retry_at) or next_retry_at > now + max(float(retry_delay), 0.0) + 60.0:
        next_retry_at = now + max(float(retry_delay), 0.0)
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

    if isinstance(ticker, bool) or not isinstance(ticker, str) or not ticker.strip():
        log("  ! [ORDER] ticker invalid")
        return False
    ticker = ticker.strip()
    if isinstance(validity, bool) or not isinstance(validity, str) or not validity.strip():
        log("  ! [ORDER] validity invalid")
        return False
    validity = validity.strip().upper()
    if validity == "GTC":
        validity = "GOOD_TILL_CANCEL"
    if validity not in {"DAY", "GOOD_TILL_CANCEL"}:
        log(f"  ! [ORDER] validity invalid: {validity!r}")
        return False
    if isinstance(quantity, bool) or isinstance(limit_price, bool) or isinstance(retry_delay, bool):
        log("  ! [ORDER] parametri numerici invalizi: bool nu este permis")
        return False
    try:
        qty_r = round(float(quantity), 2)
        price_r = round(float(limit_price), 2)
        retry_delay_f = float(retry_delay)
    except (TypeError, ValueError, OverflowError) as exc:
        log(f"  ! [ORDER] parametri numerici invalizi: {exc}")
        return False
    if (not math.isfinite(qty_r) or qty_r <= 0
            or not math.isfinite(price_r) or price_r <= 0
            or not math.isfinite(retry_delay_f) or retry_delay_f < 0):
        log("  ! [ORDER] qty/price/retry_delay must be finite and valid")
        return False

    # Current price is informational only; do not change the user-selected limit.
    current = get_price_usd(t212_to_yahoo(ticker))
    if current:
        log(f"  [ORDER] the current price of {t212_to_yahoo(ticker)}: {current:.2f} USD  |  the limit: {price_r:.2f} USD")
        if price_r < current:
            log(f"  ! [ORDER] the limit {price_r} < the price {current:.2f} -> the order will sit waiting "
                f"(it only executes if the price falls to {price_r}).")

    if dry_run:
        log(f"  [ORDER] [DRY-RUN] LIMIT BUY {ticker}  qty={qty_r}  @ {price_r} USD  validity={validity}")
        log("  [ORDER] Dry run — the order was NOT SENT. Set ORDER_EXECUTE=true in .env.")
        return True

    marker_path = _marker_path(ticker, test=not write_marker)
    now = time.time()
    try:
        record = _read_or_migrate_marker(ticker, marker_path)
    except RuntimeError as exc:
        log(f"  ! [ORDER] {exc} — NOT sending without verifiable durable state")
        return False
    if record:
        # Legacy accepted markers stay blocking rather than silently creating a
        # second order whose identity cannot be reconstructed.
        raw_ver = record.get("version")
        try:
            ver_i = 1 if (raw_ver is None or isinstance(raw_ver, bool)) else int(raw_ver)
        except (TypeError, ValueError, OverflowError):
            ver_i = 1
        if ver_i < 2:
            legacy = record.get("order") if isinstance(record.get("order"), dict) else {}
            legacy_id = legacy.get("id")
            if legacy_id:
                info = poll_order_until_terminal(client, legacy_id, ticker, desktop)
                record["last_status"] = info.get("status") if isinstance(info, dict) else None
                _persist_marker(record, marker_path)
            log("  [ORDER] an existing legacy marker — not duplicating the order")
            return True
        raw_rec_qty = record.get("qty")
        raw_rec_lim = record.get("limit_price")
        try:
            rec_qty_f = -1.0 if isinstance(raw_rec_qty, bool) else float(raw_rec_qty or 0.0)
            rec_lim_f = -1.0 if isinstance(raw_rec_lim, bool) else float(raw_rec_lim or 0.0)
        except (TypeError, ValueError, OverflowError):
            rec_qty_f, rec_lim_f = -1.0, -1.0
        same_intent = (
            str(record.get("ticker") or "").upper() == ticker.upper()
            and math.isfinite(rec_qty_f) and rec_qty_f > 0
            and math.isfinite(rec_lim_f) and rec_lim_f > 0
            and abs(rec_qty_f - qty_r) <= 0.011
            and abs(rec_lim_f - price_r) <= 0.011
        )
        if not same_intent:
            log("  ! [ORDER] another intent already exists in the marker — NOT overwriting")
            return False
        lifecycle = str(record.get("lifecycle") or "").lower()
        if lifecycle == "filled":
            return True
        if lifecycle in {"canceled", "cancelled", "expired"}:
            log(f"  [ORDER] the intent is terminal ({lifecycle}); not resending it")
            return False
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
            return bool(record.get("order_id"))
    else:
        before_qty = _portfolio_qty(client, ticker)
        if before_qty is None:
            log("  ! [ORDER] the T212 portfolio is unavailable — NOT sending")
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
        if active_matches is None:
            log("  ! [ORDER] the active T212 orders are unavailable — NOT sending")
            return False
        if active_matches:
            log(
                "  ! [ORDER] an active order with the same ticker/side/qty/price already exists; "
                "not claiming it and NOT sending another"
            )
            return False

    raw_att = record.get("attempts")
    try:
        att_i = 0 if (raw_att is None or isinstance(raw_att, bool)) else max(0, int(raw_att))
    except (TypeError, ValueError, OverflowError):
        att_i = 0
    record["attempts"] = att_i + 1
    record["last_submit_at"] = now
    record["next_retry_at"] = now + retry_delay_f
    record["lookup_misses"] = 0
    record["lifecycle"] = "submit_pending"
    _persist_marker(record, marker_path)  # durability boundary before POST

    log(f"  [ORDER] LIMIT BUY {ticker}  qty={qty_r}  @ {price_r} USD  validity={validity}")
    try:
        status, data = client.place_limit_order(ticker, qty_r, price_r, validity)
    except Exception as exc:  # response may have been lost after venue acceptance
        record["submit_error"] = f"{exc.__class__.__name__}: {exc}"
        _persist_marker(record, marker_path)
        log(f"  ! [ORDER] raspuns submit ambiguu ({exc}) — intentia ramane pending")
        return False

    if status in (200, 201) and isinstance(data, dict) and data.get("id") is not None and not isinstance(data.get("id"), bool) and str(data.get("id")).strip():
        oid = str(data["id"]).strip()
        venue_status = str(data.get("status") or "").upper()
        lifecycle = "filled" if venue_status == "FILLED" else "accepted"
        if venue_status in {"CANCELLED", "CANCELED", "EXPIRED"}:
            lifecycle = "expired" if venue_status == "EXPIRED" else "canceled"
        elif venue_status == "REJECTED":
            lifecycle = "submit_pending"
        record.update({
            "order_id": None if venue_status == "REJECTED" else oid,
            "venue_status": venue_status,
            "lifecycle": lifecycle, "order": data,
        })
        if venue_status == "REJECTED":
            raw_hist = record.get("venue_order_ids")
            history = list(raw_hist) if isinstance(raw_hist, (list, tuple)) else []
            if oid not in history:
                history.append(oid)
            record["venue_order_ids"] = history
        _persist_marker(record, marker_path)
        log(f"  [ORDER] raspuns submit: id={oid}  status={venue_status or 'NECUNOSCUT'}")
        if lifecycle in {"canceled", "expired"}:
            return False
        if lifecycle == "submit_pending":
            log("  ! [ORDER] order rejected; the intent stays queued for reconciliation/retry")
            return False
        notify(title="T212 order accepted!",
               body=f"LIMIT {ticker} qty={qty_r} @ {price_r} USD\nid={oid}",
               source="T212 order", price=price_r, desktop=desktop)
        return True

    record["submit_error"] = f"T212 HTTP {status}: {json.dumps(data, default=str)[:400]}"
    _persist_marker(record, marker_path)
    log(f"  ! [ORDER] {record['submit_error']} — retry non-blocant la ciclul urmator")
    return False
