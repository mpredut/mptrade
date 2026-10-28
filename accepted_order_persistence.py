"""Bounded fail-safe persistence for venue-accepted order claims."""
from __future__ import annotations

import math
import os
import time

import alertnotifiers as alert
from botcore import load_dotenv, required_float_env, required_int_env
from providers.base import _clean_single_order_id as _clean_id


_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, "config.env"))

ACCEPTED_TRACKING_PERSIST_ATTEMPTS = required_int_env(
    "ACCEPTED_TRACKING_PERSIST_ATTEMPTS")
ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC = required_float_env(
    "ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC")

if not 2 <= ACCEPTED_TRACKING_PERSIST_ATTEMPTS <= 10:
    raise ValueError(
        "ACCEPTED_TRACKING_PERSIST_ATTEMPTS must be between 2 and 10")
if (not math.isfinite(ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC)
        or not 0 < ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC <= 5):
    raise ValueError(
        "ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC must be finite, > 0, "
        "and <= 5")




def _accepted_is_durable(order_retry_module, claim, order) -> bool:
    """Recognize a first attempt that committed before reporting an I/O error."""
    try:
        if not isinstance(claim, dict):
            return False
        claim_id = _clean_id(claim.get("id"))
        expected_order_id = _clean_id(order_retry_module.order_id_from_response(order))
        pk = claim.get("place_kwargs")
        expected_client_id = _clean_id(pk.get("client_order_id")) if isinstance(pk, dict) else None
        if claim_id is None or expected_order_id is None or expected_client_id is None:
            return False
        for record in (order_retry_module.load_all() or ()):
            if not isinstance(record, dict) or _clean_id(record.get("id")) != claim_id:
                continue
            rpk = record.get("place_kwargs")
            current_client_id = _clean_id(rpk.get("client_order_id")) if isinstance(rpk, dict) else None
            current_order_id = _clean_id(record.get("order_id"))
            return (
                str(record.get("lifecycle") or "").strip().lower() == "accepted"
                and current_order_id == expected_order_id
                and current_client_id == expected_client_id
            )
    except Exception:  # noqa: BLE001 - the bounded exact retry remains authoritative.
        return False
    return False


_ALERTED_PERSISTENCE_FAILURES: dict[tuple, float] = {}
_PERSISTENCE_ALERT_COOLDOWN_SEC = 1800.0


def complete_accepted_claim(
        order_retry_module, claim, *, order, provider_name, symbol, side) -> bool:
    """Persist venue acceptance under the exact claim, retrying transient I/O.

    Failure leaves the producer-owned record intact for reconciliation-only
    recovery. It never falls back to a record-ID mutation that could overwrite a
    newer owner or revision.
    """
    last_error = None
    for attempt in range(ACCEPTED_TRACKING_PERSIST_ATTEMPTS):
        try:
            res = order_retry_module.complete_claim(
                claim, "accepted", order=order,
                provider_name=provider_name)
            if res is True or (not isinstance(res, (bool, str, float, dict, list, tuple, set)) and bool(res)):
                return True
            last_error = RuntimeError(
                "exact accepted-order claim completion returned false")
        except Exception as exc:  # noqa: BLE001 - retry only exact persistence.
            last_error = exc
        if _accepted_is_durable(order_retry_module, claim, order):
            return True
        if attempt + 1 < ACCEPTED_TRACKING_PERSIST_ATTEMPTS:
            time.sleep(ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC)

    from datetime import datetime
    try:
        order_id = _clean_id(order_retry_module.order_id_from_response(order))
    except Exception:
        order_id = None
    claim_dict = claim if isinstance(claim, dict) else {}
    pk = claim_dict.get("place_kwargs")
    client_order_id = _clean_id(pk.get("client_order_id")) if isinstance(pk, dict) else None
    qty = claim_dict.get("qty")
    px = claim_dict.get("requested_price") if claim_dict.get("requested_price") is not None else claim_dict.get("ref_price")
    if isinstance(qty, bool) or (isinstance(qty, float) and not math.isfinite(qty)):
        qty = None
    if isinstance(px, bool) or (isinstance(px, float) and not math.isfinite(px)):
        px = None
    val_str = "N/A"
    if qty is not None and px is not None and not isinstance(qty, bool) and not isinstance(px, bool):
        try:
            val_f = float(qty) * float(px)
            if math.isfinite(val_f):
                val_str = f"~${val_f:.2f}"
        except (TypeError, ValueError, OverflowError):
            val_str = "N/A"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail = (
        f"venue={provider_name} symbol={symbol} side={side} "
        f"qty={qty} price={px} value={val_str} time={now_str} "
        f"order_id={order_id} client_order_id={client_order_id} "
        f"attempts={ACCEPTED_TRACKING_PERSIST_ATTEMPTS} error={last_error}")
    print(f"CRITICAL: accepted order tracking could not be persisted: {detail}")
    dedup_key = (str(provider_name or ""), str(symbol or ""), str(side or ""), str(order_id or client_order_id or ""))
    now_mono = time.monotonic()
    last_alert_ts = _ALERTED_PERSISTENCE_FAILURES.get(dedup_key)
    if last_alert_ts is None or (now_mono - last_alert_ts) >= _PERSISTENCE_ALERT_COOLDOWN_SEC:
        _ALERTED_PERSISTENCE_FAILURES[dedup_key] = now_mono
        while len(_ALERTED_PERSISTENCE_FAILURES) > 1000:
            _ALERTED_PERSISTENCE_FAILURES.pop(next(iter(_ALERTED_PERSISTENCE_FAILURES)), None)
        alert.notify(
            title="🛑 CRITICAL accepted order tracking persistence failure",
            body=detail,
            source="order-persistence",
            symbol=str(symbol),
            email=True,
        )
    return False
