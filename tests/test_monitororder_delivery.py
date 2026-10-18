"""Delivery ownership, notification, and TTL regressions for monitororder."""

import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pytest

import accepted_order_persistence
import monitororder
import order_retry_worker
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
    monkeypatch.setattr(
        monitororder.trade_cooldown, "get_last_trade_age",
        lambda *_args, **_kwargs: 999.0)
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


def test_rtrade_order_placement_does_not_emit_launched_notification():
    with patch("alertnotifiers.notify") as alert_mock:
        monitororder._notify_order_launched("TAOUSDC", "BUY", "rt-1", 255.5, 1.9, owner="rtrade")
    alert_mock.assert_not_called()


def test_tradeall_order_placement_emits_launched_notification():
    with patch("alertnotifiers.notify") as alert_mock:
        monitororder._notify_order_launched("TAOUSDC", "BUY", "ta-1", 255.5, 1.9, owner="tradeall")
    alert_mock.assert_called_once()
    assert alert_mock.call_args.kwargs["source"] == "monitororder"
    assert "tradeall" in alert_mock.call_args.kwargs["title"]


class TestMonitororderNotifications(unittest.TestCase):
    def test_warmup_does_not_notify_historical_open_orders(self):
        open_orders = {
            1001: {"price": 80000.0, "quantity": 0.1, "remainingQty": 0.1},
            1002: {"price": 81000.0, "quantity": 0.2, "remainingQty": 0.2},
        }
        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch("alertnotifiers.notify") as mock_notify:
            monitororder.warmup_open_orders()
            mock_notify.assert_not_called()
            self.assertIn("1001", monitororder._notified_placed_order_ids)
            self.assertIn("1002", monitororder._notified_placed_order_ids)

    def test_newly_launched_order_triggers_notification_and_deduplicates(self):
        with patch.object(monitororder.api, "get_open_orders", return_value={}):
            monitororder.warmup_open_orders()

        open_orders = {
            2001: {"price": 283.92, "quantity": 0.5, "remainingQty": 0.5}
        }
        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=250.0), \
             patch("alertnotifiers.notify") as mock_notify:
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_notify.assert_called_once_with(
                title="📝 BUY TAOUSDC @ 283.92",
                body="Order 2001 placed on venue. Qty: 0.5000 TAOUSDC (~141.96 USDC)",
                source="monitororder",
                symbol="TAOUSDC",
                price=283.92,
            )

            mock_notify.reset_mock()
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_notify.assert_not_called()

    def test_derived_replacement_order_does_not_trigger_notification(self):
        with patch.object(monitororder.api, "get_open_orders", return_value={}):
            monitororder.warmup_open_orders()

        initial_order = {
            3001: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0}
        }
        replacement_order = {"orderId": 9999}

        with patch.object(monitororder.api, "get_open_orders", return_value=initial_order), \
             patch.object(monitororder.api, "get_current_price", return_value=100.1), \
             patch.object(monitororder.mkt, "preflight_order", return_value=object()), \
             patch.object(monitororder.api, "cancel_order", return_value=True), \
             patch.object(monitororder.mkt, "order_status", return_value=MagicMock(venue_status="CANCELED", filled_qty=0.0)), \
             patch.object(monitororder.mkt, "place", return_value=replacement_order), \
             patch.object(monitororder.accepted_order_persistence, "complete_accepted_claim", return_value=True), \
             patch("alertnotifiers.notify") as mock_notify:
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")
            self.assertEqual(mock_notify.call_count, 1)
            self.assertIn("3001", mock_notify.call_args[1]["body"])
            self.assertIn("9999", monitororder._derived_replacement_order_ids)

            mock_notify.reset_mock()
            open_after_replace = {
                9999: {"price": 101.2, "quantity": 1.0, "remainingQty": 1.0}
            }
            with patch.object(monitororder.api, "get_open_orders", return_value=open_after_replace), \
                 patch.object(monitororder.api, "get_current_price", return_value=90.0):
                monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")
                mock_notify.assert_not_called()


class TestMonitororderTTLResolution:
    def test_module_default_ttl(self):
        assert monitororder.resolve_effective_order_ttl("1", "RT_rtrade-TAOUSDC_1_abc", owner="rtrade") == 300.0
        assert monitororder.resolve_effective_order_ttl("2", "TA_OR_abc_0", owner="tradeall") == 7200.0
        assert monitororder.resolve_effective_order_ttl("3", "MN_abc_0", owner="manual") == 86400.0
        assert monitororder.resolve_effective_order_ttl("4", "UNKNOWN_0", owner="unknown") == 14400.0

    def test_custom_ttl_from_map(self):
        custom_map = {
            "100": 600.0,
            "TA_OR_custom_0": 1800.0,
        }
        assert monitororder.resolve_effective_order_ttl("100", "TA_OR_abc_0", owner="tradeall", custom_ttl_map=custom_map) == 600.0
        assert monitororder.resolve_effective_order_ttl("101", "TA_OR_custom_0", owner="tradeall", custom_ttl_map=custom_map) == 1800.0

    def test_hard_cap_3_days_enforced_on_custom_ttl(self):
        ten_days = 10 * 86400.0
        custom_map = {"999": ten_days}
        effective = monitororder.resolve_effective_order_ttl("999", "MN_abc_0", owner="manual", custom_ttl_map=custom_map)
        assert effective == monitororder.MAX_GLOBAL_ORDER_TTL_SEC
        assert effective == 259200.0


class TestMonitororderTTLExpiration:
    def test_expired_order_is_canceled(self):
        now = time.time()
        open_orders = {
            "711457409": {
                "price": 277.4,
                "quantity": 1.8,
                "timestamp": now - 400.0,
                "clientOrderId": "RT_rtrade-TAOUSDC_1_abc",
            }
        }
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
            patch.object(monitororder.api, "get_current_price", return_value=278.0),
            patch.object(monitororder.mkt, "cancel_order") as mock_cancel,
            patch.object(monitororder, "_notify_order_expired") as mock_notify,
        ):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_cancel.assert_called_once_with("TAOUSDC", "711457409")
            mock_notify.assert_called_once()

    def test_non_expired_order_is_not_canceled(self):
        now = time.time()
        open_orders = {
            "711457410": {
                "price": 277.4,
                "quantity": 1.8,
                "timestamp": now - 100.0,
                "clientOrderId": "RT_rtrade-TAOUSDC_1_abc",
            }
        }
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
            patch.object(monitororder.api, "get_current_price", return_value=278.0),
            patch.object(monitororder.mkt, "cancel_order") as mock_cancel,
        ):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_cancel.assert_not_called()

    def test_hard_cap_cancels_even_if_custom_ttl_larger(self):
        now = time.time()
        open_orders = {
            "10544265098": {
                "price": 83000.0,
                "quantity": 0.01,
                "timestamp": now - (3.5 * 86400.0),
                "clientOrderId": "MN_OR_manual_0",
            }
        }
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
            patch.object(monitororder.api, "get_current_price", return_value=83500.0),
            patch.object(monitororder.mkt, "cancel_order") as mock_cancel,
            patch.object(monitororder, "_notify_order_expired") as mock_notify,
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "BUY")
            mock_cancel.assert_called_once_with("BTCUSDC", "10544265098")
            mock_notify.assert_called_once()


class TestOrderRetryTTL:
    def test_enqueue_persists_ttl_sec(self, tmp_path):
        import order_retry as oq
        with patch.object(oq, "QUEUE_FILE", str(tmp_path / "queue.jsonl")), \
             patch.object(oq, "LOCK_FILE", str(tmp_path / "queue.lock")):
            rec_id = oq.enqueue(
                "BTCUSDC", "BUY", 0.01,
                requested_price=80000.0, ref_price=80000.0,
                ttl_sec=1200.0
            )
            records = oq.load_all()
            found = [r for r in records if r.get("id") == rec_id]
            assert len(found) == 1
            assert found[0].get("ttl_sec") == 1200.0
            assert found[0].get("expires_at_ts") is not None

    def test_get_accepted_order_ttl_map(self, tmp_path):
        import order_retry as oq
        with patch.object(oq, "QUEUE_FILE", str(tmp_path / "queue.jsonl")), \
             patch.object(oq, "LOCK_FILE", str(tmp_path / "queue.lock")):
            oq.enqueue(
                "BTCUSDC", "BUY", 0.01,
                place_kwargs={"client_order_id": "TA_OR_test_1", "ttl_sec": 750.0},
                requested_price=80000.0, ref_price=80000.0,
            )
            ttl_map = oq.get_accepted_order_ttl_map()
            assert ttl_map.get("TA_OR_test_1") == 750.0

    def test_subdollar_buy_replacement_uses_positive_price(self, tmp_path):
        import order_retry as oq
        open_orders = {
            555001: {
                "price": 0.20,
                "quantity": 300.0,
                "remainingQty": 300.0,
            }
        }
        placed = []
        def fake_place(symbol, side, price, qty, **kw):
            placed.append((symbol, side, price, qty))
            return {"orderId": 555002}

        with patch.object(oq, "QUEUE_FILE", str(tmp_path / "queue.jsonl")), \
             patch.object(oq, "LOCK_FILE", str(tmp_path / "queue.lock")), \
             patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=0.1999), \
             patch.object(monitororder.mkt, "order_filter_refusal", return_value=None), \
             patch.object(monitororder.mkt, "preflight_order", return_value=object()), \
             patch.object(monitororder.api, "cancel_order", return_value=True), \
             patch.object(monitororder.mkt, "order_status", return_value=oq.OrderStatus("canceled", 0.0, 0.0, 0.0, venue_status="CANCELED")), \
             patch.object(monitororder.mkt, "place", side_effect=fake_place), \
             patch.object(monitororder.accepted_order_persistence, "complete_accepted_claim", return_value=True), \
             patch("alertnotifiers.notify"):
            monitororder.monitor_open_orders_by_type("ARBUSDC", "BUY")
        assert len(placed) == 1
        assert abs(placed[0][2] - (0.1999 * 0.999)) < 1e-9
