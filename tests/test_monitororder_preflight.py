import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import monitororder
import order_retry_worker
from providers.strategy_executor import (
    OrderStatus,
    ProviderError,
    SubmissionRefused,
)
from tests.conftest import TempDirTestCase


class MonitorOrderPreflightTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        monitororder.reset_monitororder_state_for_test()
        self._old_queue_file = monitororder.order_retry.QUEUE_FILE
        self._old_queue_lock = monitororder.order_retry.LOCK_FILE
        self._old_retry_enabled = monitororder.order_retry.RETRY_ENABLED
        self._old_retry_dedup = monitororder.order_retry.RETRY_DEDUP
        monitororder.order_retry.QUEUE_FILE = os.path.join(
            self.tmp, "order_retry_queue.jsonl")
        monitororder.order_retry.LOCK_FILE = os.path.join(
            self.tmp, "order_retry_queue.lock")
        monitororder.order_retry.RETRY_ENABLED = True
        monitororder.order_retry.RETRY_DEDUP = False
        self._filter_patcher = patch.object(
            monitororder.mkt, "order_filter_refusal", return_value=None)
        self.filter_check = self._filter_patcher.start()
        self._native_replace_patcher = patch.object(
            monitororder, "_native_replace_enabled_for", return_value=False)
        self._native_replace_patcher.start()

    def tearDown(self):
        monitororder.reset_monitororder_state_for_test()
        monitororder.order_retry.QUEUE_FILE = self._old_queue_file
        monitororder.order_retry.LOCK_FILE = self._old_queue_lock
        monitororder.order_retry.RETRY_ENABLED = self._old_retry_enabled
        monitororder.order_retry.RETRY_DEDUP = self._old_retry_dedup
        self._native_replace_patcher.stop()
        self._filter_patcher.stop()
        super().tearDown()

    def test_filter_refusal_or_metadata_failure_preserves_live_order(self):
        for label, filter_result in (
            ("deterministic-refusal", "below_min_notional"),
            ("metadata-failure", ProviderError("rules unavailable")),
        ):
            with self.subTest(label=label):
                monitororder.reset_monitororder_state_for_test()
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
                    patch.object(
                        monitororder.api, "get_open_orders",
                        return_value=orders),
                    patch.object(
                        monitororder.api, "get_current_price",
                        return_value=100.0),
                    patch.object(
                        monitororder, "are_close", return_value=True),
                    patch.object(
                        monitororder.mkt, "preflight_order") as preflight,
                    patch.object(
                        monitororder.api, "cancel_order") as cancel,
                    patch.object(monitororder.mkt, "place") as place,
                    patch("builtins.print"),
                ):
                    monitororder.monitor_open_orders_by_type(
                        "BTCUSDC", "SELL")

                self.filter_check.assert_called_once_with(
                    "BTCUSDC",
                    "SELL",
                    100.0 * (1.0 + monitororder.REPRICE_OFFSET),
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
            patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(
                monitororder.mkt,
                "preflight_order",
                side_effect=SubmissionRefused("account_cache_not_fresh"),
            ) as preflight,
            patch.object(monitororder.api, "cancel_order") as cancel,
            patch.object(monitororder.mkt, "place") as place,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        preflight.assert_called_once_with(
            "BTCUSDC", "SELL", 2.0, 100.0 * (1.0 + monitororder.REPRICE_OFFSET),
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
                monitororder.reset_monitororder_state_for_test()
                if os.path.exists(monitororder.order_retry.QUEUE_FILE):
                    os.unlink(monitororder.order_retry.QUEUE_FILE)
                orders = {101: {"price": 100.0, "quantity": 2.0}}
                preflight_kwargs = (
                    {"side_effect": preflight_result}
                    if isinstance(preflight_result, Exception)
                    else {"return_value": preflight_result})
                with (
                    patch.object(
                        monitororder.api, "get_open_orders",
                        return_value=orders),
                    patch.object(
                        monitororder.api, "get_current_price",
                        return_value=100.0),
                    patch.object(
                        monitororder, "are_close", return_value=True),
                    patch.object(
                        monitororder.mkt, "preflight_order",
                        **preflight_kwargs),
                    patch.object(
                        monitororder.api, "cancel_order") as cancel,
                    patch.object(monitororder.mkt, "place") as place,
                    patch("builtins.print"),
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
            patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(
                monitororder.mkt, "preflight_order", side_effect=preflight),
            patch.object(
                monitororder.api, "cancel_order", side_effect=cancel),
            patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                    venue_status="CANCELED")),
            patch.object(monitororder.mkt, "place", side_effect=place),
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        expected_px = 100.0 * (1.0 + monitororder.REPRICE_OFFSET)
        self.assertEqual([event[0] for event in events], [
            "preflight", "cancel", "place"])
        self.assertEqual(events[0][1], ("BTCUSDC", "SELL", 2.0, expected_px))
        self.assertEqual(events[0][2], {
            "market": False, "kind": "monitor_order_replace"})
        self.assertEqual(events[2][1], (
            "BTCUSDC", "SELL", expected_px, 2.0))
        place_kwargs = events[2][2]
        self.assertFalse(place_kwargs["smart"])
        self.assertEqual(place_kwargs["kind"], "monitor_order_replace")
        self.assertIs(place_kwargs["cache_permit"], cache_permit)
        self.assertTrue(place_kwargs["caller_owns_retry"])
        self.assertTrue(place_kwargs["client_order_id"].startswith(("MO_OR_", "OR_")))
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
            patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(
                monitororder.mkt, "preflight_order", return_value=permit),
            patch.object(
                monitororder.api, "cancel_order", return_value=False),
            patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0,
                    venue_status="CANCELED")),
            patch.object(
                monitororder.mkt, "place",
                return_value={"orderId": 202}) as place,
            patch("builtins.print"),
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
            patch.object(
                monitororder.api, "get_open_orders", return_value=orders),
            patch.object(
                monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(
                monitororder.mkt, "preflight_order",
                return_value=permit) as preflight,
            patch.object(
                monitororder.api, "cancel_order", return_value=True),
            patch.object(
                monitororder.mkt, "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.75, cost=75.0, fee=0.01,
                    venue_status="CANCELED")),
            patch.object(
                monitororder.mkt, "place",
                return_value={"orderId": 202}) as place,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        self.assertEqual(preflight.call_args.args[2], 2.0)
        self.assertEqual(place.call_args.args[3], 1.25)
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(persisted[0]["qty"], 1.25)

    def _run_post_cancel_replacement(self, *, place_side_effect=None, place_return=None, permit=None):
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        place_kwargs = {"side_effect": place_side_effect} if place_side_effect is not None else {"return_value": place_return}
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=permit or object()),
            patch.object(monitororder.api, "cancel_order", return_value=True) as cancel_order,
            patch.object(
                monitororder.mkt,
                "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0, venue_status="CANCELED"
                ),
            ),
            patch.object(monitororder.mkt, "place", **place_kwargs) as place,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")
        return cancel_order, place

    @staticmethod
    def _make_outcome_refusal(reason: str, state: str = "refused"):
        def _refuse(*args, **kwargs):
            kwargs["_outcome_context"].update(accepted=False, reason=reason, state=state)
            return None
        return _refuse

    def test_post_cancel_policy_refusal_keeps_serializable_durable_intent(self):
        for refusal_reason in ("profit_guard", "trend_deferred", "cooldown"):
            with self.subTest(refusal_reason=refusal_reason):
                if os.path.exists(monitororder.order_retry.QUEUE_FILE):
                    os.unlink(monitororder.order_retry.QUEUE_FILE)
                self._run_post_cancel_replacement(
                    place_side_effect=self._make_outcome_refusal(refusal_reason),
                    permit=object(),
                )

                persisted = monitororder.order_retry.load_all()
                self.assertEqual(len(persisted), 1)
                self.assertEqual(
                    persisted[0]["last_failure_reason"], refusal_reason)
                self.assertNotIn("claim_token", persisted[0])
                serialized = json.dumps(persisted)
                self.assertNotIn("cache_permit", serialized)

    def test_post_cancel_submit_response_loss_stays_unknown_for_reconciliation(self):
        self._run_post_cancel_replacement(
            place_side_effect=self._make_outcome_refusal("response_without_order_id", state="unknown"),
        )

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
                _, place = self._run_post_cancel_replacement(
                    place_side_effect=self._make_outcome_refusal(refusal_reason),
                )

                place.assert_called_once()
                self.assertEqual(
                    monitororder.order_retry.load_all(), [])

        recovery = SimpleNamespace(place_calls=[])
        recovery.place = lambda *args, **kwargs: (
            recovery.place_calls.append((args, kwargs)) or {"orderId": "must-not-submit"}
        )
        stats = order_retry_worker.process_once(recovery, now=10_000.0)
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(recovery.place_calls, [])

    def test_recent_trade_does_not_block_replacement(self):
        """Cooldown gate eliminated: recent trade does not block replacement preflight."""
        cancel_order, place = self._run_post_cancel_replacement(
            place_return={"orderId": "new-1"},
            permit=object(),
        )
        cancel_order.assert_called_once()
        place.assert_called_once()

    def test_native_replace_fast_path_and_fallbacks(self):
        """[NATIVE_REPLACE step 3] Verify native confirmed, replace_not_attempted, replace_cancel_only, and ambiguous client-ID recovery."""
        # 1. Native confirmed -> no cancel_order or order_status call; claim accepted and initial_prices updated.
        monitororder.order_retry.rewrite([])
        monitororder.initial_sell_prices.clear()
        monitororder.initial_sell_prices[101] = 95.0
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        permit1 = object()
        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=permit1) as preflight,
            patch.object(monitororder.api, "cancel_order") as cancel_order,
            patch.object(monitororder.mkt, "order_status") as order_status,
            patch.object(monitororder.mkt, "place", return_value={"orderId": 303}) as place,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        cancel_order.assert_not_called()
        order_status.assert_not_called()
        preflight.assert_called_once()
        place.assert_called_once()
        self.assertEqual(place.call_args.kwargs["replaces_order_id"], 101)
        self.assertEqual(place.call_args.kwargs["replaces_locked_balance"], 2.0)
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["lifecycle"], "accepted")
        self.assertEqual(persisted[0]["order_id"], "303")
        self.assertEqual(list(monitororder.initial_sell_prices.items()), [(303, 95.0)])

        # 2. replace_not_attempted -> normal cancel_order + order_status + fallback place.
        monitororder.order_retry.rewrite([])
        monitororder.initial_sell_prices.clear()
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        place_calls = []

        def _place_not_attempted(*args, **kwargs):
            place_calls.append((args, kwargs))
            if kwargs.get("replaces_order_id") is not None:
                kwargs["_outcome_context"].update(
                    accepted=False, reason="replace_not_attempted", state="refused"
                )
                return None
            return {"orderId": 304}

        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=object()) as preflight,
            patch.object(monitororder.api, "cancel_order", return_value=True) as cancel_order,
            patch.object(
                monitororder.mkt,
                "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.0, cost=0.0, fee=0.0, venue_status="CANCELED"
                ),
            ) as order_status,
            patch.object(monitororder.mkt, "place", side_effect=_place_not_attempted),
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        self.assertEqual(preflight.call_count, 2)
        cancel_order.assert_called_once_with("BTCUSDC", 101)
        order_status.assert_called_once_with("BTCUSDC", "101")
        self.assertEqual(len(place_calls), 2)
        self.assertNotIn("replaces_order_id", place_calls[1][1])
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(persisted[0]["lifecycle"], "accepted")
        self.assertEqual(persisted[0]["order_id"], "304")

        # 3. replace_cancel_only -> skips redundant cancel_order, reconciles CANCELED status, and places remaining qty.
        monitororder.order_retry.rewrite([])
        monitororder.initial_sell_prices.clear()
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        place_calls_co = []

        def _place_cancel_only(*args, **kwargs):
            place_calls_co.append((args, kwargs))
            if kwargs.get("replaces_order_id") is not None:
                kwargs["_outcome_context"].update(
                    accepted=False, reason="replace_cancel_only", state="refused"
                )
                return None
            return {"orderId": 305}

        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=object()),
            patch.object(monitororder.api, "cancel_order") as cancel_order,
            patch.object(
                monitororder.mkt,
                "order_status",
                return_value=OrderStatus(
                    status="canceled", filled_qty=0.5, cost=50.0, fee=0.0, venue_status="CANCELED"
                ),
            ) as order_status,
            patch.object(monitororder.mkt, "place", side_effect=_place_cancel_only),
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        cancel_order.assert_not_called()
        order_status.assert_called_once_with("BTCUSDC", "101")
        self.assertEqual(len(place_calls_co), 2)
        self.assertEqual(place_calls_co[1][0][3], 1.5)
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(persisted[0]["lifecycle"], "accepted")
        self.assertEqual(persisted[0]["order_id"], "305")
        self.assertEqual(persisted[0]["qty"], 1.5)

        # 4. Ambiguous native replace + found by client_order_id -> success without double placement.
        monitororder.order_retry.rewrite([])
        monitororder.initial_buy_prices.clear()
        orders = {101: {"price": 100.0, "quantity": 2.0}}

        def _place_ambiguous(*args, **kwargs):
            kwargs["_outcome_context"].update(
                accepted=False, reason="submit_ambiguous", state="unknown"
            )
            return None

        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=object()),
            patch.object(monitororder.api, "cancel_order") as cancel_order,
            patch.object(monitororder.mkt, "order_status") as order_status,
            patch.object(monitororder.mkt, "place", side_effect=_place_ambiguous) as place,
            patch.object(
                monitororder.mkt, "order_by_client_id", return_value={"orderId": 306}
            ) as order_by_cid,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "BUY")

        place.assert_called_once()
        self.assertEqual(place.call_args.kwargs["replaces_locked_balance"], 200.0)
        order_by_cid.assert_called_once()
        cancel_order.assert_not_called()
        order_status.assert_not_called()
        persisted = monitororder.order_retry.load_all()
        self.assertEqual(persisted[0]["lifecycle"], "accepted")
        self.assertEqual(persisted[0]["order_id"], "306")

        # 5. Refused native replace + fallback cancel sees order still OPEN -> resets submission_state="refused".
        monitororder.order_retry.rewrite([])
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=object()),
            patch.object(monitororder.api, "cancel_order", return_value=False),
            patch.object(
                monitororder.mkt,
                "order_status",
                return_value=OrderStatus(
                    status="open", filled_qty=0.0, cost=0.0, fee=0.0, venue_status="NEW"
                ),
            ),
            patch.object(monitororder.mkt, "place", side_effect=_place_not_attempted),
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        persisted_open = monitororder.order_retry.load_all()
        self.assertEqual(len(persisted_open), 1)
        self.assertEqual(persisted_open[0]["lifecycle"], "awaiting_cancel")
        self.assertEqual(persisted_open[0]["submission_state"], "refused")

        # 6. Pre-dispatch guard refusal during native replace preserves live order without cancel or second place.
        monitororder.order_retry.rewrite([])
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=object()),
            patch.object(monitororder.api, "cancel_order") as cancel_guard,
            patch.object(
                monitororder.mkt,
                "place",
                side_effect=self._make_outcome_refusal("daily_limit"),
            ) as place_guard,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")

        place_guard.assert_called_once()
        cancel_guard.assert_not_called()
        self.assertEqual(monitororder.order_retry.load_all(), [])

        # 7. Non-Binance provider (Kraken/Hyperliquid: requires_cache_permit=False, defers_preflight_quantity=True)
        # passes preflight_qty=None and succeeds when preflight_order returns None.
        monitororder.order_retry.rewrite([])
        monitororder.initial_sell_prices.clear()
        orders_kr = {"OTXID-KR-1": {"price": 200.0, "quantity": 1.5}}
        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders_kr),
            patch.object(monitororder.api, "get_current_price", return_value=200.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", return_value=None) as pf_kr,
            patch.object(monitororder.api, "cancel_order") as cancel_kr,
            patch.object(
                monitororder.mkt,
                "place",
                return_value={"orderId": "OTXID-KR-1", "amended_in_place": True},
            ) as place_kr,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("TAOUSD", "SELL")

        pf_kr.assert_called_once()
        self.assertIsNone(pf_kr.call_args.args[2])
        place_kr.assert_called_once()
        cancel_kr.assert_not_called()
        persisted_kr = monitororder.order_retry.load_all()
        self.assertEqual(len(persisted_kr), 1)
        self.assertEqual(persisted_kr[0]["lifecycle"], "accepted")
        self.assertEqual(persisted_kr[0]["order_id"], "OTXID-KR-1")

    def test_native_replace_not_attempted_refuses_fallback_cancel_when_repreflight_fails(self):
        """If native replace returns replace_not_attempted and re-preflight fails, do NOT cancel the live order."""
        orders = {101: {"price": 100.0, "quantity": 2.0}}
        initial_permit = object()
        pf_calls = [initial_permit, SubmissionRefused("account_cache_not_fresh")]

        def _pf(*_a, **_kw):
            nxt = pf_calls.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        with (
            patch.object(monitororder, "_native_replace_enabled_for", return_value=True),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order", side_effect=_pf),
            patch.object(
                monitororder.mkt,
                "place",
                side_effect=self._make_outcome_refusal("replace_not_attempted"),
            ) as place_mock,
            patch.object(monitororder.api, "cancel_order") as cancel_mock,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "BUY")

        place_mock.assert_called_once()
        cancel_mock.assert_not_called()
        self.assertEqual(monitororder.order_retry.load_all(), [])

    def test_non_binance_buy_reprice_checks_venue_account_cache_health_before_cancel(self):
        """Non-Binance BUY reprice aborts before cancel when venue account-cache health is stale."""
        orders = {"OTXID-KR-BUY": {"price": 100.0, "quantity": 2.0}}
        fake_kr = SimpleNamespace(
            requires_cache_permit=lambda: False,
            defers_preflight_quantity=lambda: True,
            require_account_cache_fresh_for_submit=lambda sym, side: (
                (_ for _ in ()).throw(SubmissionRefused("account_cache_not_fresh"))
                if side == "BUY"
                else None
            ),
        )
        with (
            patch.object(monitororder, "_resolve_provider", return_value=fake_kr),
            patch.object(monitororder.api, "get_open_orders", return_value=orders),
            patch.object(monitororder.api, "get_current_price", return_value=100.0),
            patch.object(monitororder, "are_close", return_value=True),
            patch.object(monitororder.mkt, "preflight_order") as pf_mock,
            patch.object(monitororder.api, "cancel_order") as cancel_mock,
            patch.object(monitororder.mkt, "place") as place_mock,
            patch("builtins.print"),
        ):
            monitororder.monitor_open_orders_by_type("HYPEUSD", "BUY")

        pf_mock.assert_not_called()
        cancel_mock.assert_not_called()
        place_mock.assert_not_called()
        self.assertEqual(monitororder.order_retry.load_all(), [])
