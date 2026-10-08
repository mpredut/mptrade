"""Delivery ownership regressions for monitororder replacements."""

from unittest.mock import patch
from types import SimpleNamespace

import pytest

import monitororder
import order_retry_worker
import accepted_order_persistence
from providers.strategy_executor import OrderStatus


@pytest.fixture(autouse=True)
def isolated_retry_queue(tmp_path, monkeypatch):
    """Prevent replacement tests from ever writing the runtime outbox."""
    monkeypatch.setattr(
        monitororder.order_retry, "QUEUE_FILE",
        str(tmp_path / "order_retry_queue.jsonl"))
    monkeypatch.setattr(
        monitororder.order_retry, "LOCK_FILE",
        str(tmp_path / "order_retry_queue.lock"))
    monkeypatch.setattr(monitororder.order_retry, "RETRY_ENABLED", True)
    monkeypatch.setattr(
        monitororder.mkt, "order_filter_refusal",
        lambda *_args, **_kwargs: None)
    monitororder.reset_monitororder_state_for_test()
    yield
    monitororder.reset_monitororder_state_for_test()


def test_failed_replacement_is_submitted_once_and_left_to_shared_outbox():
    failed_legacy = [{
        "symbol": "TAOUSDC", "order_type": "SELL",
        "price": 999.0, "quantity": 7.0,
    }]
    monitororder.initial_sell_prices.clear()
    open_orders = {"old-1": {"price": 100.0, "quantity": 2.0}}

    with (
        patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
        patch.object(monitororder.api, "get_current_price", return_value=100.1),
        patch.object(
            monitororder.mkt, "preflight_order", return_value=object()),
        patch.object(monitororder.api, "cancel_order", return_value=True),
        patch.object(
            monitororder.mkt, "order_status",
            return_value=OrderStatus(
                status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                venue_status="CANCELED")),
        patch.object(monitororder.mkt, "place", return_value=None) as place,
        patch.object(monitororder.time, "sleep") as sleep,
    ):
        monitororder.monitor_open_orders_by_type(
            "TAOUSDC", "SELL", failed_legacy,
        )

    place.assert_called_once()
    assert place.call_args.args[:2] == ("TAOUSDC", "SELL")
    assert place.call_args.kwargs["smart"] is False
    assert failed_legacy == [{
        "symbol": "TAOUSDC", "order_type": "SELL",
        "price": 999.0, "quantity": 7.0,
    }]
    sleep.assert_not_called()


def test_accepted_replacement_transfers_original_anchor_to_new_order_id():
    monitororder.initial_buy_prices.clear()
    open_orders = {"old-2": {"price": 100.0, "quantity": 1.5}}

    with (
        patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
        patch.object(monitororder.api, "get_current_price", return_value=100.1),
        patch.object(
            monitororder.mkt, "preflight_order", return_value=object()),
        patch.object(monitororder.api, "cancel_order", return_value=True),
        patch.object(
            monitororder.mkt, "order_status",
            return_value=OrderStatus(
                status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                venue_status="CANCELED")),
        patch.object(
            monitororder.mkt, "place",
            return_value={"orderId": "new-2"},
        ),
    ):
        monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")

    assert "old-2" not in monitororder.initial_buy_prices
    assert monitororder.initial_buy_prices["new-2"] == 100.0


def test_transient_accepted_replacement_tracking_failure_is_retried():
    open_orders = {"old-transient": {"price": 100.0, "quantity": 1.5}}
    real_complete = monitororder.order_retry.complete_claim
    attempts = []

    def transient_complete(*args, **kwargs):
        attempts.append((args, kwargs))
        if len(attempts) == 1:
            raise OSError("transient fsync failure")
        return real_complete(*args, **kwargs)

    with (
        patch.object(
            monitororder.api, "get_open_orders", return_value=open_orders),
        patch.object(
            monitororder.api, "get_current_price", return_value=100.1),
        patch.object(
            monitororder.mkt, "preflight_order", return_value=object()),
        patch.object(monitororder.api, "cancel_order", return_value=True),
        patch.object(
            monitororder.mkt, "order_status",
            return_value=OrderStatus(
                status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                venue_status="CANCELED")),
        patch.object(
            monitororder.mkt, "place",
            return_value={"orderId": "new-transient"}),
        patch.object(
            monitororder.order_retry, "complete_claim",
            side_effect=transient_complete),
        patch.object(
            accepted_order_persistence.time, "sleep") as retry_sleep,
        patch.object(
            accepted_order_persistence.alert, "notify") as critical_alert,
    ):
        monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")

    assert len(attempts) == 2
    retry_sleep.assert_called_once_with(
        accepted_order_persistence.
        ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC)
    critical_alert.assert_not_called()
    tracked = monitororder.order_retry.load_all()
    assert len(tracked) == 1
    assert tracked[0]["lifecycle"] == "accepted"
    assert tracked[0]["order_id"] == "new-transient"


def test_crash_after_accepted_replacement_is_reconciled_without_resubmit():
    open_orders = {"old-crash": {"price": 100.0, "quantity": 1.5}}
    submitted = []

    def accept_then_crash_window(*args, **kwargs):
        submitted.append((args, kwargs))
        return {"orderId": "new-crash"}

    with (
        patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
        patch.object(monitororder.api, "get_current_price", return_value=100.1),
        patch.object(monitororder.mkt, "preflight_order", return_value=object()),
        patch.object(monitororder.api, "cancel_order", return_value=True),
        patch.object(
            monitororder.mkt, "order_status",
            return_value=OrderStatus(
                status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                venue_status="CANCELED")),
        patch.object(monitororder.mkt, "place", side_effect=accept_then_crash_window),
        patch.object(
            monitororder.order_retry, "complete_claim",
            side_effect=SystemExit("simulated crash before acceptance persistence")),
    ):
        with pytest.raises(SystemExit, match="simulated crash"):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")

    pending = monitororder.order_retry.load_all()[0]
    assert len(submitted) == 1
    assert pending["submission_state"] == "producer_claimed"
    client_order_id = pending["place_kwargs"]["client_order_id"]

    class RecoveryMarket:
        def __init__(self):
            self.place_calls = []
            self.lookup_calls = []

        def reconciliation_capabilities(self, symbol, provider_name=None):
            return SimpleNamespace(lookup_by_client_order_id=True)

        def order_by_client_id(self, symbol, wanted, provider_name=None):
            self.lookup_calls.append((symbol, wanted, provider_name))
            assert wanted == client_order_id
            return {"orderId": "new-crash", "status": "NEW"}

        def place(self, *args, **kwargs):
            self.place_calls.append((args, kwargs))
            raise AssertionError("recovery must not submit the accepted replacement again")

    recovery = RecoveryMarket()
    with patch.object(
            monitororder.order_retry, "producer_claim_owner_state",
            return_value="dead"):
        stats = order_retry_worker.process_once(
            recovery,
            now=max(
                float(pending["claim_until"]) + 1.0,
                float(pending["created_ts"])
                + monitororder.order_retry.RETRY_INTERVAL_SEC + 1.0,
            ))

    assert stats["reconciled"] == 1
    assert recovery.place_calls == []
    assert recovery.lookup_calls == [
        ("TAOUSDC", client_order_id, "Binance")]
    tracked = monitororder.order_retry.load_all()[0]
    assert tracked["lifecycle"] == "accepted"
    assert tracked["order_id"] == "new-crash"
