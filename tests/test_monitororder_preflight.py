import json
import os
import tempfile
import unittest
from unittest import mock

import monitororder
import order_retry_worker
from providers.strategy_executor import (
    OrderStatus,
    ProviderError,
    SubmissionRefused,
)


class MonitorOrderPreflightTest(unittest.TestCase):
    def setUp(self):
        monitororder.initial_sell_prices.clear()
        monitororder.initial_buy_prices.clear()
        self._temp_dir = tempfile.TemporaryDirectory()
        self._old_queue_file = monitororder.order_retry.QUEUE_FILE
        self._old_queue_lock = monitororder.order_retry.LOCK_FILE
        self._old_retry_enabled = monitororder.order_retry.RETRY_ENABLED
        self._old_retry_dedup = monitororder.order_retry.RETRY_DEDUP
        monitororder.order_retry.QUEUE_FILE = os.path.join(
            self._temp_dir.name, "order_retry_queue.jsonl")
        monitororder.order_retry.LOCK_FILE = os.path.join(
            self._temp_dir.name, "order_retry_queue.lock")
        monitororder.order_retry.RETRY_ENABLED = True
        monitororder.order_retry.RETRY_DEDUP = False
        self._filter_patcher = mock.patch.object(
            monitororder.mkt, "order_filter_refusal", return_value=None)
        self.filter_check = self._filter_patcher.start()

    def tearDown(self):
        monitororder.initial_sell_prices.clear()
        monitororder.initial_buy_prices.clear()
        monitororder.order_retry.QUEUE_FILE = self._old_queue_file
        monitororder.order_retry.LOCK_FILE = self._old_queue_lock
        monitororder.order_retry.RETRY_ENABLED = self._old_retry_enabled
        monitororder.order_retry.RETRY_DEDUP = self._old_retry_dedup
        self._filter_patcher.stop()
        self._temp_dir.cleanup()

    def test_filter_refusal_or_metadata_failure_preserves_live_order(self):
        for label, filter_result in (
            ("deterministic-refusal", "below_min_notional"),
            ("metadata-failure", ProviderError("rules unavailable")),
        ):
            with self.subTest(label=label):
                monitororder.order_retry.rewrite([])
                self.filter_check.reset_mock()
                self.filter_check.side_effect = None
                self.filter_check.return_value = None
                if isinstance(filter_result, Exception):
                    self.filter_check.side_effect = filter_result
                else:
                    self.filter_check.return_value = filter_result
                orders = {101: {"price": 100.0, "quantity": 2.0}}

                with (
                    mock.patch.object(
                        monitororder.api, "get_open_orders",
                        return_value=orders),
                    mock.patch.object(
                        monitororder.api, "get_current_price",
                        return_value=100.0),
                    mock.patch.object(
                        monitororder.u, "are_close", return_value=True),
                    mock.patch.object(
                        monitororder.mkt, "preflight_order") as preflight,
                    mock.patch.object(
                        monitororder.api, "cancel_order") as cancel,
                    mock.patch.object(monitororder.mkt, "place") as place,
                    mock.patch("builtins.print"),
                ):
                    monitororder.monitor_open_orders_by_type(
                        "BTCUSDC", "SELL")

                self.filter_check.assert_called_once_with(
                    "BTCUSDC",
                    "SELL",
                    101.1,
                    2.0,
                    market=False,
                    enforce_business_minimum=True,
                    provider_name="Binance",
                )
                preflight.assert_not_called()
                cancel.assert_not_called()
                place.assert_not_called()
                self.assertEqual(monitororder.order_retry.load_all(), [])

    def test_stale_cache_refusal_preserves_live_order_and_tracking(self):
        monitororder.initial_sell_prices[101] = 95.0
        orders = {101: {"price": 100.0, "quantity": 2.0}}

        with (
            mock.patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            mock.patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            mock.patch.object(monitororder.u, "are_close", return_value=True),
            mock.patch.object(
                monitororder.mkt,
                "preflight_order",
                side_effect=SubmissionRefused("account_cache_not_fresh"),
            ) as preflight,
            mock.patch.object(monitororder.api, "cancel_order") as cancel,
            mock.patch.object(monitororder.mkt, "place") as place,
            mock.patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        preflight.assert_called_once_with(
            "BTCUSDC", "SELL", 2.0, 101.1,
            market=False, kind="monitor_order_replace")
        cancel.assert_not_called()
        place.assert_not_called()
        self.assertEqual(
            list(monitororder.initial_sell_prices.items()), [(101, 95.0)])
        self.assertEqual(monitororder.order_retry.load_all(), [])

    def test_untyped_preflight_error_or_missing_permit_removes_awaiting_record(self):
        for label, preflight_result in (
                ("unexpected-error", RuntimeError("preflight failed")),
                ("missing-permit", None)):
            with self.subTest(label=label):
                if os.path.exists(monitororder.order_retry.QUEUE_FILE):
                    os.unlink(monitororder.order_retry.QUEUE_FILE)
                orders = {101: {"price": 100.0, "quantity": 2.0}}
                preflight_kwargs = (
                    {"side_effect": preflight_result}
                    if isinstance(preflight_result, Exception)
                    else {"return_value": preflight_result})
                with (
                    mock.patch.object(
                        monitororder.api, "get_open_orders",
                        return_value=orders),
                    mock.patch.object(
                        monitororder.api, "get_current_price",
                        return_value=100.0),
                    mock.patch.object(
                        monitororder.u, "are_close", return_value=True),
                    mock.patch.object(
                        monitororder.mkt, "preflight_order",
                        **preflight_kwargs),
                    mock.patch.object(
                        monitororder.api, "cancel_order") as cancel,
                    mock.patch.object(monitororder.mkt, "place") as place,
                    mock.patch("builtins.print"),
                ):
                    monitororder.monitor_open_orders_by_type(
                        "BTCUSDC", "SELL")

                cancel.assert_not_called()
                place.assert_not_called()
                self.assertEqual(monitororder.order_retry.load_all(), [])

    def test_success_preflights_before_cancel_and_guarded_place(self):
        events = []
        cache_permit = object()
        orders = {101: {"price": 100.0, "quantity": 2.0}}

        def preflight(*args, **kwargs):
            events.append(("preflight", args, kwargs))
            return cache_permit

        def cancel(*args, **kwargs):
            events.append(("cancel", args, kwargs))
            return True

        def place(*args, **kwargs):
            events.append(("place", args, kwargs))
            return {"orderId": 202}

        with (
            mock.patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            mock.patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            mock.patch.object(monitororder.u, "are_close", return_value=True),
            mock.patch.object(
                monitororder.mkt, "preflight_order", side_effect=preflight),
            mock.patch.object(
                monitororder.api, "cancel_order", side_effect=cancel),
            mock.patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                    venue_status="CANCELED")),
            mock.patch.object(monitororder.mkt, "place", side_effect=place),
            mock.patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        self.assertEqual([event[0] for event in events], [
            "preflight", "cancel", "place"])
        self.assertEqual(events[0][1], ("BTCUSDC", "SELL", 2.0, 101.1))
        self.assertEqual(events[0][2], {
            "market": False, "kind": "monitor_order_replace"})
        self.assertEqual(events[2][1], (
            "BTCUSDC", "SELL", 101.1, 2.0))
        place_kwargs = events[2][2]
        self.assertFalse(place_kwargs["smart"])
        self.assertEqual(place_kwargs["kind"], "monitor_order_replace")
        self.assertIs(place_kwargs["cache_permit"], cache_permit)
        self.assertTrue(place_kwargs["caller_owns_retry"])
        self.assertTrue(place_kwargs["client_order_id"].startswith("OR_"))
        self.assertIsInstance(place_kwargs["_outcome_context"], dict)
        self.assertEqual(
            list(monitororder.initial_sell_prices.items()), [(202, 100.0)])
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["lifecycle"], "accepted")
        self.assertEqual(persisted[0]["order_id"], "202")

    def test_cancel_response_loss_reconciles_canceled_order_and_replaces_it(self):
        permit = object()
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        with (
            mock.patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            mock.patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            mock.patch.object(monitororder.u, "are_close", return_value=True),
            mock.patch.object(
                monitororder.mkt, "preflight_order", return_value=permit),
            mock.patch.object(
                monitororder.api, "cancel_order", return_value=False),
            mock.patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                    venue_status="CANCELED")),
            mock.patch.object(
                monitororder.mkt, "place",
                return_value={"orderId": 202}) as place,
            mock.patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        place.assert_called_once()
        self.assertIs(place.call_args.kwargs["cache_permit"], permit)
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(persisted[0]["order_id"], "202")

    def test_partial_fill_replaces_only_the_unfilled_quantity(self):
        permit = object()
        orders = {101: {
            "price": 100.0,
            "quantity": 2.0,
            "executedQty": 0.0,
            "remainingQty": 2.0,
        }}
        with (
            mock.patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            mock.patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            mock.patch.object(monitororder.u, "are_close", return_value=True),
            mock.patch.object(
                monitororder.mkt, "preflight_order",
                return_value=permit) as preflight,
            mock.patch.object(
                monitororder.api, "cancel_order", return_value=True),
            mock.patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.75, cost=75.0, fee=0.01,
                    venue_status="CANCELED")),
            mock.patch.object(
                monitororder.mkt, "place",
                return_value={"orderId": 202}) as place,
            mock.patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        self.assertEqual(preflight.call_args.args[2], 2.0)
        self.assertEqual(place.call_args.args[3], 1.25)
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(persisted[0]["qty"], 1.25)

    def test_post_cancel_policy_refusal_keeps_serializable_durable_intent(self):
        for refusal_reason in ("profit_guard", "trend_deferred", "cooldown"):
            with self.subTest(refusal_reason=refusal_reason):
                if os.path.exists(monitororder.order_retry.QUEUE_FILE):
                    os.unlink(monitororder.order_retry.QUEUE_FILE)
                orders = {101: {"price": 100.0, "quantity": 2.0}}
                permit = object()

                def refuse(*args, **kwargs):
                    kwargs["_outcome_context"].update(
                        accepted=False, reason=refusal_reason, state="refused")
                    return None

                with (
                    mock.patch.object(
                        monitororder.api, "get_open_orders",
                        return_value=orders),
                    mock.patch.object(
                        monitororder.api, "get_current_price",
                        return_value=100.0),
                    mock.patch.object(
                        monitororder.u, "are_close", return_value=True),
                    mock.patch.object(
                        monitororder.mkt, "preflight_order",
                        return_value=permit),
                    mock.patch.object(
                        monitororder.api, "cancel_order", return_value=True),
                    mock.patch.object(
                        monitororder.mkt, "order_status",
                        return_value=OrderStatus(
                            status="canceled", filled_qty=0.0,
                            cost=0.0, fee=0.0,
                            venue_status="CANCELED")),
                    mock.patch.object(
                        monitororder.mkt, "place", side_effect=refuse),
                    mock.patch("builtins.print"),
                ):
                    monitororder.monitor_open_orders_by_type(
                        "BTCUSDC", "SELL")

                persisted = monitororder.order_retry.load_all()
                self.assertEqual(len(persisted), 1)
                self.assertEqual(
                    persisted[0]["last_failure_reason"], refusal_reason)
                self.assertNotIn("claim_token", persisted[0])
                serialized = json.dumps(persisted)
                self.assertNotIn("cache_permit", serialized)

    def test_post_cancel_submit_response_loss_stays_unknown_for_reconciliation(self):
        orders = {101: {"price": 100.0, "quantity": 2.0}}

        def lose_response(*args, **kwargs):
            kwargs["_outcome_context"].update(
                accepted=False,
                reason="response_without_order_id",
                state="unknown",
            )
            return None

        with (
            mock.patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            mock.patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            mock.patch.object(monitororder.u, "are_close", return_value=True),
            mock.patch.object(
                monitororder.mkt, "preflight_order", return_value=object()),
            mock.patch.object(
                monitororder.api, "cancel_order", return_value=True),
            mock.patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                    venue_status="CANCELED")),
            mock.patch.object(
                monitororder.mkt, "place", side_effect=lose_response),
            mock.patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        persisted = monitororder.order_retry.load_all()
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["lifecycle"], "submit_pending")
        self.assertEqual(persisted[0]["submission_state"], "unknown")
        self.assertEqual(
            persisted[0]["last_failure_reason"],
            "response_without_order_id")
        self.assertNotIn("cache_permit", json.dumps(persisted))

    def test_post_cancel_terminal_refusal_is_never_replayed(self):
        for refusal_reason in (
                "execution_disabled",
                "below_min_notional",
                "binance_filter_refused:quantity below lot size"):
            with self.subTest(refusal_reason=refusal_reason):
                monitororder.order_retry.rewrite([])
                monitororder.initial_sell_prices.clear()
                orders = {101: {"price": 100.0, "quantity": 2.0}}

                def terminal_refusal(*args, **kwargs):
                    kwargs["_outcome_context"].update(
                        accepted=False,
                        reason=refusal_reason,
                        state="refused",
                    )
                    return None

                with (
                    mock.patch.object(
                        monitororder.api, "get_open_orders",
                        return_value=orders),
                    mock.patch.object(
                        monitororder.api, "get_current_price",
                        return_value=100.0),
                    mock.patch.object(
                        monitororder.u, "are_close", return_value=True),
                    mock.patch.object(
                        monitororder.mkt, "preflight_order",
                        return_value=object()),
                    mock.patch.object(
                        monitororder.api, "cancel_order", return_value=True),
                    mock.patch.object(
                        monitororder.mkt, "order_status",
                        return_value=OrderStatus(
                            status="canceled", filled_qty=0.0,
                            cost=0.0, fee=0.0,
                            venue_status="CANCELED")),
                    mock.patch.object(
                        monitororder.mkt, "place",
                        side_effect=terminal_refusal) as place,
                    mock.patch("builtins.print"),
                ):
                    monitororder.monitor_open_orders_by_type(
                        "BTCUSDC", "SELL")

                place.assert_called_once()
                self.assertEqual(
                    monitororder.order_retry.load_all(), [])

        class RecoveryMarket:
            def __init__(self):
                self.place_calls = []

            def place(self, *args, **kwargs):
                self.place_calls.append((args, kwargs))
                return {"orderId": "must-not-submit"}

        recovery = RecoveryMarket()
        stats = order_retry_worker.process_once(recovery, now=10_000.0)
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(recovery.place_calls, [])

    def test_cooldown_active_preserves_live_order(self):
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        with (
            mock.patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            mock.patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            mock.patch.object(
                monitororder.u, "are_close", return_value=True),
            mock.patch.object(
                monitororder.trade_cooldown, "get_last_trade_age", return_value=5.0),
            mock.patch.object(
                monitororder.api, "cancel_order") as cancel_order,
            mock.patch.object(
                monitororder.mkt, "place") as place,
            mock.patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        cancel_order.assert_not_called()
        place.assert_not_called()
        self.assertEqual(monitororder.order_retry.load_all(), [])


if __name__ == "__main__":
    unittest.main()
