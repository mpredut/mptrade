"""Adversarial tests for the generic persistent tracked-order lifecycle."""
import copy
import unittest
from unittest import mock

from providers.strategy_executor import OrderStatus, SubmissionOutcome
from order_retry import (
    OrderSubmissionRefused,
    TrackedOrderLifecycle,
    propagate_submission_refusal,
)


class FakeMarketApi:
    def __init__(self):
        self.lookup = None
        self.statuses = [OrderStatus("open", 0.0, 0.0, 0.0)]
        self.lookup_calls = []
        self.status_calls = []
        self.cancel_calls = []
        self.cancel_error = None

    def order_by_client_id(self, symbol, client_order_id, *, provider_name=None):
        self.lookup_calls.append((symbol, client_order_id, provider_name))
        if isinstance(self.lookup, Exception):
            raise self.lookup
        return copy.deepcopy(self.lookup)

    def order_status(self, symbol, order_id, *, provider_name=None):
        self.status_calls.append((symbol, order_id, provider_name))
        value = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(value, Exception):
            raise value
        return value

    def cancel_order(self, symbol, order_id, *, provider_name=None):
        self.cancel_calls.append((symbol, order_id, provider_name))
        if self.cancel_error:
            raise self.cancel_error


class MemoryPending:
    def __init__(self):
        self.value = None
        self.history = []

    def persist(self, value):
        self.value = copy.deepcopy(value)
        self.history.append(copy.deepcopy(value))


class TrackedOrderLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.now = 10_000.0
        self.api = FakeMarketApi()
        self.lifecycle = TrackedOrderLifecycle(
            self.api, provider_name="binance", venue="Binance",
            missing_confirmations=2, max_age_seconds=900,
            clock=lambda: self.now,
        )
        self.store = MemoryPending()

    def intent(self, **overrides):
        values = {
            "intent_id": "intent-1",
            "client_order_id": "CID-1",
            "symbol": "TAOUSDC",
            "side": "SELL",
            "requested_qty": 3.0,
            "requested_price": 115.0,
            "kind": "ASSET_GUARDIAN_TIER",
            "attempt": 1,
            "metadata": {"threshold": 15.0},
        }
        values.update(overrides)
        return self.lifecycle.new_intent(**values)

    def test_intent_is_persisted_before_submit_side_effect(self):
        events = []

        def persist(value):
            events.append(("persist", copy.deepcopy(value)))

        def submit():
            events.append(("submit", None))
            return {"orderId": "7", "status": "NEW", "origQty": "3"}

        result = self.lifecycle.submit(self.intent(), persist=persist, submit=submit)

        self.assertEqual(events[0][0], "persist")
        self.assertEqual(events[1][0], "submit")
        self.assertTrue(result.order_known)
        self.assertEqual(result.outcome, "active")

    def test_submit_persists_venue_adjusted_quantity_and_price(self):
        result = self.lifecycle.submit(
            self.intent(), persist=self.store.persist,
            submit=lambda: {
                "orderId": "7", "status": "NEW",
                "origQty": "2.75", "price": "114.25",
            },
        )

        self.assertEqual(result.intent["submitted_qty"], 2.75)
        self.assertEqual(result.intent["submitted_price"], 114.25)
        self.assertEqual(self.store.value["submitted_price"], 114.25)

    def test_typed_acceptance_is_persisted_without_equating_it_to_fill(self):
        result = self.lifecycle.submit(
            self.intent(), persist=self.store.persist,
            submit=lambda: SubmissionOutcome("accepted", order_id="OID-9"),
        )

        self.assertEqual(result.outcome, "active")
        self.assertEqual(result.intent["submission_outcome"], "accepted")
        self.assertEqual(result.intent["order_id"], "OID-9")
        self.assertNotIn("terminal_status", result.intent)

    def test_typed_unknown_stays_persisted_for_reconciliation(self):
        result = self.lifecycle.submit(
            self.intent(), persist=self.store.persist,
            submit=lambda: SubmissionOutcome("unknown", reason="socket timeout"),
        )

        self.assertEqual(result.outcome, "active")
        self.assertFalse(result.order_known)
        self.assertEqual(result.intent["submission_outcome"], "unknown")
        self.assertEqual(result.intent["submit_error"], "socket timeout")

    def test_instrument_outcome_bridge_preserves_refused_vs_unknown(self):
        with self.assertRaisesRegex(
                OrderSubmissionRefused, "below_min_notional"):
            propagate_submission_refusal(
                None,
                {"state": "refused", "reason": "below_min_notional"},
            )
        self.assertIsNone(propagate_submission_refusal(
            None,
            {"state": "unknown", "reason": "response_without_order_id"},
        ))
        accepted = {"orderId": "9"}
        self.assertIs(
            propagate_submission_refusal(
                accepted,
                {"state": "accepted", "reason": None},
            ),
            accepted,
        )

    def test_pre_submit_refusal_is_not_classified_as_ambiguous(self):
        result = self.lifecycle.submit(
            self.intent(), persist=self.store.persist,
            submit=lambda: (_ for _ in ()).throw(
                OrderSubmissionRefused("trend_deferred")),
        )

        self.assertEqual(result.outcome, "refused")
        self.assertFalse(result.order_known)
        self.assertEqual(result.intent["refusal_reason"], "trend_deferred")
        self.assertEqual(result.intent["submit_status"], "refused_before_submit")
        self.assertEqual(self.store.value, result.intent)

    def test_lost_submit_response_recovers_by_client_id_without_resubmit(self):
        self.api.lookup = {"orderId": 8, "status": "NEW", "origQty": "3"}
        submit = mock.Mock(return_value=None)

        result = self.lifecycle.submit(
            self.intent(), persist=self.store.persist, submit=submit)
        result = self.lifecycle.reconcile(
            self.store.value, persist=self.store.persist)

        submit.assert_called_once_with()
        self.assertEqual(result.outcome, "active")
        self.assertEqual(result.intent["order_id"], "8")
        self.assertEqual(len(self.api.lookup_calls), 1)
        self.assertEqual(len(self.api.status_calls), 1)

    def test_submit_does_not_poll_or_wait_for_terminal_status(self):
        submit = mock.Mock(
            return_value={"orderId": "no-poll", "status": "FILLED"})

        result = self.lifecycle.submit(
            self.intent(), persist=self.store.persist, submit=submit)

        self.assertEqual(result.outcome, "active")
        self.assertTrue(result.order_known)
        self.assertEqual(self.api.lookup_calls, [])
        self.assertEqual(self.api.status_calls, [])
        self.assertEqual(self.api.cancel_calls, [])

    def test_confirmed_absence_requires_two_lookups_and_never_submits_in_reconcile(self):
        pending = self.intent()
        result1 = self.lifecycle.reconcile(pending, persist=self.store.persist)
        result2 = self.lifecycle.reconcile(
            self.store.value, persist=self.store.persist)

        self.assertEqual(result1.outcome, "active")
        self.assertEqual(result2.outcome, "absent")
        self.assertIsNone(self.store.value)
        self.assertEqual(len(self.api.lookup_calls), 2)
        self.assertEqual(self.api.status_calls, [])

    def test_at_least_once_policy_releases_lookup_error_for_strategy_retry(self):
        lifecycle = TrackedOrderLifecycle(
            self.api, provider_name="binance", retry_on_lookup_error=True,
            missing_confirmations=2, clock=lambda: self.now)
        pending = lifecycle.new_intent(
            intent_id="retry-1", client_order_id="CID-RETRY", symbol="TAOUSDC",
            side="BUY", requested_qty=1.0)
        self.api.lookup = RuntimeError("lookup transport down")

        result = lifecycle.reconcile(pending, persist=self.store.persist)

        self.assertEqual(result.outcome, "retryable")
        self.assertIsNone(self.store.value)
        self.assertEqual(len(self.api.lookup_calls), 1)
        self.assertEqual(self.api.status_calls, [])

    def test_terminal_truth_stays_persisted_until_strategy_acknowledges_it(self):
        pending = self.intent()
        pending["order_id"] = "9"
        self.api.statuses = [OrderStatus("closed", 3.0, 345.0, 0.2)]

        result1 = self.lifecycle.reconcile(pending, persist=self.store.persist)
        status_calls = len(self.api.status_calls)
        result2 = self.lifecycle.reconcile(
            self.store.value, persist=self.store.persist)

        self.assertEqual(result1.outcome, "terminal")
        self.assertEqual(result2.outcome, "terminal")
        self.assertEqual(result2.status.filled_qty, 3.0)
        self.assertIn("terminal_status", self.store.value)
        self.assertEqual(len(self.api.status_calls), status_calls)

    def test_old_owned_order_is_canceled_once_and_partial_terminal_is_returned(self):
        pending = self.intent()
        pending["order_id"] = "10"
        pending["created_at"] = self.now - 901
        self.api.statuses = [
            OrderStatus("open", 1.0, 115.0, 0.1),
            OrderStatus("canceled", 1.0, 115.0, 0.1),
        ]

        result = self.lifecycle.reconcile(pending, persist=self.store.persist)

        self.assertEqual(result.outcome, "terminal")
        self.assertEqual(result.status.status, "canceled")
        self.assertEqual(result.status.filled_qty, 1.0)
        self.assertEqual(
            self.api.cancel_calls, [("TAOUSDC", "10", "binance")])
        self.assertIn("cancel_attempted_at", result.intent)
        self.assertIn("terminal_status", self.store.value)

    def test_ambiguous_cancel_is_never_repeated(self):
        pending = self.intent()
        pending["order_id"] = "11"
        pending["created_at"] = self.now - 901
        self.api.cancel_error = RuntimeError("timeout")

        result1 = self.lifecycle.reconcile(pending, persist=self.store.persist)
        result2 = self.lifecycle.reconcile(
            self.store.value, persist=self.store.persist)

        self.assertEqual((result1.outcome, result2.outcome), ("active", "active"))
        self.assertEqual(len(self.api.cancel_calls), 1)
        self.assertIn("cancel_attempted_at", self.store.value)

    def test_invalid_created_at_stays_active_without_blind_cancel(self):
        pending = self.intent()
        pending["order_id"] = "12"
        pending["created_at"] = "broken"

        result = self.lifecycle.reconcile(pending, persist=self.store.persist)

        self.assertEqual(result.outcome, "active")
        self.assertEqual(self.api.cancel_calls, [])

    def test_persist_failure_before_submit_prevents_external_side_effect(self):
        submit = mock.Mock()

        def broken_persist(_value):
            raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            self.lifecycle.submit(
                self.intent(), persist=broken_persist, submit=submit)
        submit.assert_not_called()

    def test_metadata_cannot_override_lifecycle_fields(self):
        with self.assertRaisesRegex(ValueError, "reserved key"):
            self.intent(metadata={"order_id": "foreign"})
        with self.assertRaisesRegex(ValueError, "reserved key"):
            self.intent(metadata={"submitted_price": 1.0})

    def test_reconcile_clears_definitive_submit_refusal_without_order_id(self):
        pending = self.intent()
        pending["submission_outcome"] = "unknown"
        pending["submit_error"] = "ProviderError: submit_order HYPEUSD buy: EOrder:Insufficient funds"

        result = self.lifecycle.reconcile(pending, persist=self.store.persist)

        self.assertEqual(result.outcome, "refused")
        self.assertIsNone(self.store.value)
        self.assertEqual(self.api.lookup_calls, [])


if __name__ == "__main__":
    unittest.main()
