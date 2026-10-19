"""Bounded fail-safe persistence for venue-accepted order claims."""
from __future__ import annotations

import math
import os
import time

import alertnotifiers as alert
from botcore import load_dotenv, required_float_env, required_int_env


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
        expected_order_id = order_retry_module.order_id_from_response(order)
        expected_client_id = dict(
            claim.get("place_kwargs") or {}).get("client_order_id")
        for record in order_retry_module.load_all():
            if record.get("id") != claim.get("id"):
                continue
            current_client_id = dict(
                record.get("place_kwargs") or {}).get("client_order_id")
            return (
                str(record.get("lifecycle") or "").lower() == "accepted"
                and str(record.get("order_id") or "") == str(expected_order_id)
                and expected_order_id is not None
                and expected_client_id is not None
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
            if order_retry_module.complete_claim(
                    claim, "accepted", order=order,
                    provider_name=provider_name):
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
    order_id = order_retry_module.order_id_from_response(order)
    client_order_id = dict(
        claim.get("place_kwargs") or {}).get("client_order_id")
    qty = claim.get("qty")
    px = claim.get("requested_price") if claim.get("requested_price") is not None else claim.get("ref_price")
    try:
        val_str = f"~${float(qty) * float(px):.2f}" if (qty is not None and px is not None) else "N/A"
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
        alert.notify(
            title="🛑 CRITICAL accepted order tracking persistence failure",
            body=detail,
            source="order-persistence",
            symbol=str(symbol),
            email=True,
        )
    return False
