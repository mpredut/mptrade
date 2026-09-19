"""Characterisation and regressions for worker management in rtrade."""
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rtrade
from providers.strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    ProviderError,
    SubmissionRefused,
)
from rtrade_pair_store import RTradePairStore, rtrade_client_order_id


def _bot():
    bot = object.__new__(rtrade.TradingBot)
    bot.symbol = "TAOUSDC"
    bot.qty = 100.0
    bot.DEFAULT_ADJUSTMENT_PERCENT = 0.0064
    bot.filled_buy_price = 99.0
    bot.filled_sell_price = 101.0
    bot.buy_filled = False
    bot.sell_filled = False
    bot.lock = threading.Lock()
    bot.pair_store = SimpleNamespace(
        active=lambda _symbol: [],
        begin=lambda *_args, **_kwargs: None,
        checkpoint=lambda *_args, **_kwargs: None,
        checkpoint_many=lambda *_args, **_kwargs: None,
    )
    return bot


def _hard_stop_venue(root, store, *, preflight_error=None, submit_error=None):
    submit = Mock(return_value="M10")
    if submit_error is not None:
        submit.side_effect = submit_error
    preflight = Mock(return_value=None)
    if preflight_error is not None:
        preflight.side_effect = preflight_error
    executor = SimpleNamespace(
        name="Binance",
        free_balance=lambda _asset: 1.0,
        fee_cap_quantity=lambda *_args: 1.0,
        pair_precision=lambda _symbol: None,
        preflight_order=preflight,
        submit_order=submit,
    )
    with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
         patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
         patch.object(rtrade.mkt, "provider_by_name", return_value=executor):
        venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
    venue.current_price = lambda: 90.0
    return venue, executor


def _restart_merge_venue(root, store, statuses, lookups=None):
    class Executor:
        name = "Binance"

        def __init__(self):
            self.status_calls = []
            self.lookup_calls = []

        @staticmethod
        def free_balance(_asset):
            return 1000.0

        @staticmethod
        def reconciliation_capabilities():
            return OrderReconciliationCapabilities(
                lookup_by_client_order_id=True,
                status_by_order_id=True,
                cancel_by_order_id=True,
                list_open_orders=True,
            )

        def order_by_client_id(self, symbol, client_order_id):
            self.lookup_calls.append((symbol, client_order_id))
            return dict(lookups or {}).get(client_order_id)

        def order_status(self, symbol, order_id):
            self.status_calls.append((symbol, str(order_id)))
            return statuses[str(order_id)]

    executor = Executor()
    with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
         patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
         patch.object(rtrade.mkt, "provider_by_name", return_value=executor):
        venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
    return venue, executor


def _old_exposed_checkpoint(pair_id):
    return {
        "pair_id": pair_id,
        "qty": 1.0,
        "start_side": "BUY",
        "phase": "exposed",
        "reason": None,
        "shock": False,
        "elapsed_sec": 10.0,
        "first_fill_elapsed_sec": 1.0,
        "first_fill_side": "BUY",
        "stop_order_id": None,
        "limit_revisions": {"BUY": 0, "SELL": 0},
        "hard_stop_revision": 0,
        "hard_stop_reason": None,
        "tickets": [{
            "order_id": "entry", "side": "BUY", "price": 99.0,
            "qty": 1.0, "active": False, "pair_id": pair_id,
        }],
        "snapshots": {
            "entry": {
                "status": "closed", "filled_qty": 1.0,
                "cost": 99.0, "fee": 0.01,
            },
        },
    }


class RTradeThreadingTest(unittest.TestCase):
    def test_intent_recovery_config_fails_closed_on_missing_or_invalid_values(self):
        invalid = [
            (None, 1.0, 300.0),
            (2, float("nan"), 300.0),
            (2, 1.0, float("nan")),
            (1, 1.0, 300.0),
            (2, 1.0, 1.0),
        ]
        for confirmations, poll_sec, horizon_sec in invalid:
            with self.subTest(
                    confirmations=confirmations,
                    poll_sec=poll_sec, horizon_sec=horizon_sec):
                with self.assertRaises(ValueError):
                    rtrade._validate_intent_recovery_config(
                        confirmations, poll_sec, horizon_sec)

    def test_startup_cancels_the_orphaned_rtrade_order_automatically(self):
        bot = _bot()
        canceled = []
        executor = SimpleNamespace(
            open_orders=Mock(side_effect=[[
                {"orderId": "91", "clientOrderId": "RT_orphan"}
            ], []]),
            cancel_order=lambda symbol, order_id: canceled.append(
                (symbol, order_id)),
        )
        venue = SimpleNamespace(current_price=lambda: None, executor=executor)
        with patch.object(rtrade, "_LivePairVenue", return_value=venue), \
             patch.object(rtrade.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                bot._run_coordinator_forever()

        self.assertEqual(canceled, [("TAOUSDC", "91")])
        self.assertEqual(executor.open_orders.call_count, 2)

    def test_feature_flag_routes_to_single_coordinator_path(self):
        bot = _bot()
        with patch.object(rtrade, "RTRADE_PAIR_COORDINATOR_ENABLED", True), \
             patch.object(bot, "_run_coordinator_forever", return_value="coordinated") as run:
            self.assertEqual(bot.run(), "coordinated")
        run.assert_called_once_with()

    def test_coordinator_starts_multiple_independent_rounds_on_same_symbol(self):
        bot = _bot()
        coordinators = []

        class FakeCoordinator:
            def __init__(self, *_args, **kwargs):
                self.pair_id = f"pair-{len(coordinators) + 1}"
                self.start_side = kwargs.get("start_side")
                self.steps = []
                coordinators.append(self)

            def start(self, _price, pair_id=None):
                self.pair_id = pair_id or self.pair_id
                return SimpleNamespace(
                    terminal=False, pair_id=self.pair_id, phase="quoting",
                    reason=None)

            def step(self, now=None):
                self.steps.append(now)
                return SimpleNamespace(terminal=False)

        venue = SimpleNamespace(
            current_price=lambda: 100.0,
            executor=SimpleNamespace(open_orders=lambda _symbol: []))
        with patch.object(rtrade, "_LivePairVenue", return_value=venue), \
             patch.object(rtrade, "PairCoordinator", FakeCoordinator), \
             patch.object(rtrade, "RTRADE_PAIR_MAX_ACTIVE_ROUNDS", 2), \
             patch.object(rtrade, "RTRADE_PAIR_START_INTERVAL_SEC", 8), \
             patch.object(rtrade, "RTRADE_PAIR_DIRECTIONS", ("BUY", "SELL")), \
             patch.object(rtrade, "RTRADE_PAIR_POLL_SEC", 1), \
             patch.object(rtrade, "_trend_too_strong", return_value=False), \
             patch.object(rtrade.time, "monotonic", side_effect=[0.0, 8.0, 16.0]), \
             patch.object(rtrade.time, "sleep",
                          side_effect=[None, None, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                bot._run_coordinator_forever()

        self.assertEqual(len({c.pair_id for c in coordinators}), 2)
        self.assertTrue(all(len(c.pair_id) == 32 for c in coordinators))
        self.assertEqual([c.start_side for c in coordinators], ["BUY", "SELL"])
        self.assertEqual(coordinators[0].steps, [8.0, 16.0])
        self.assertEqual(coordinators[1].steps, [16.0])

    def test_checkpoint_failure_after_acceptance_retains_owner_and_blocks_rounds(self):
        bot = _bot()
        coordinators = []
        checkpoint_batches = []

        class FailingCheckpointStore:
            @staticmethod
            def active(_symbol):
                return []

            @staticmethod
            def checkpoint(*_args, **_kwargs):
                raise OSError("checkpoint fsync failed")

            @staticmethod
            def checkpoint_many(checkpoints):
                checkpoint_batches.append(len(list(checkpoints)))

        class AcceptedCoordinator:
            def __init__(self, *_args, **kwargs):
                self.pair_id = ""
                self.start_side = kwargs["start_side"]
                self.steps = 0
                coordinators.append(self)

            def start(self, _price, pair_id=None):
                self.pair_id = pair_id
                return SimpleNamespace(
                    terminal=False, pair_id=pair_id,
                    phase="quoting", reason=None)

            def step(self, now=None):
                self.steps += 1
                return SimpleNamespace(terminal=False)

            def export_state(self):
                return {
                    "pair_id": self.pair_id,
                    "qty": 1.0,
                    "start_side": self.start_side,
                    "phase": "quoting",
                    "tickets": [],
                }

        bot.pair_store = FailingCheckpointStore()
        venue = SimpleNamespace(
            current_price=lambda: 100.0,
            recovery_blocked=False,
            executor=SimpleNamespace(open_orders=lambda _symbol: []))
        with patch.object(rtrade, "_LivePairVenue", return_value=venue), \
             patch.object(rtrade, "PairCoordinator", AcceptedCoordinator), \
             patch.object(rtrade, "_trend_too_strong", return_value=False), \
             patch.object(rtrade.time, "monotonic", side_effect=[0.0, 1.0]), \
             patch.object(
                 rtrade.time, "sleep",
                 side_effect=[None, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                bot._run_coordinator_forever()

        self.assertEqual(len(coordinators), 1)
        self.assertEqual(coordinators[0].steps, 1)
        self.assertEqual(checkpoint_batches, [0, 1])

    def test_coordinator_backs_off_each_side_after_generic_place_failure(self):
        bot = _bot()
        starts = []

        class FailingCoordinator:
            def __init__(self, *_args, **kwargs):
                self.start_side = kwargs["start_side"]

            def start(self, _price, pair_id=None):
                self.pair_id = pair_id or f"pair-{len(starts) + 1}"
                starts.append(self.start_side)
                return SimpleNamespace(
                    terminal=True,
                    pair_id=f"pair-{len(starts)}",
                    phase="failed",
                    reason=f"{self.start_side.lower()}_place_failed",
                )

        venue = SimpleNamespace(
            current_price=lambda: 100.0,
            executor=SimpleNamespace(open_orders=lambda _symbol: []))
        with patch.object(rtrade, "_LivePairVenue", return_value=venue), \
             patch.object(rtrade, "PairCoordinator", FailingCoordinator), \
             patch.object(rtrade, "RTRADE_PAIR_MAX_ACTIVE_ROUNDS", 2), \
             patch.object(rtrade, "RTRADE_PAIR_START_INTERVAL_SEC", 8), \
             patch.object(rtrade, "RTRADE_PAIR_DIRECTIONS", ("BUY", "SELL")), \
             patch.object(rtrade, "RTRADE_PAIR_POLL_SEC", 1), \
             patch.object(rtrade, "RTRADE_PLACE_FAILURE_BACKOFF_SEC", 180), \
             patch.object(rtrade, "_trend_too_strong", return_value=False), \
             patch.object(rtrade.time, "monotonic", side_effect=[0.0, 8.0, 16.0]), \
             patch.object(rtrade.time, "sleep",
                          side_effect=[None, None, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                bot._run_coordinator_forever()

        self.assertEqual(starts, ["BUY", "SELL"])

    def test_place_failure_backoff_preserves_funds_specific_duration(self):
        with patch.object(rtrade, "RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC", 181), \
             patch.object(rtrade, "RTRADE_PLACE_FAILURE_BACKOFF_SEC", 37):
            self.assertEqual(
                rtrade._place_failure_backoff("buy_insufficient_funds:USDC"),
                ("BUY", 181),
            )
            self.assertEqual(
                rtrade._place_failure_backoff("sell_place_failed"),
                ("SELL", 37),
            )
            self.assertEqual(rtrade._place_failure_backoff("other"), (None, 0.0))

    def test_live_pair_adapter_passes_pair_id_and_owns_retry(self):
        executor = SimpleNamespace(free_balance=lambda _asset: 1000.0)
        order = {"orderId": 7, "price": "99.36", "origQty": "0.8"}
        with patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
             patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
             patch.object(rtrade.mkt, "place", return_value=order) as place:
            venue = rtrade._LivePairVenue("TAOUSDC")
            ticket = venue.place_limit("BUY", 99.36, 1.0, "pair-1")

        self.assertEqual((ticket.order_id, ticket.qty, ticket.pair_id),
                         ("7", 0.8, "pair-1"))
        kwargs = place.call_args.kwargs
        self.assertEqual(kwargs["cooldown_pair_id"], "pair-1")
        self.assertTrue(kwargs["caller_owns_retry"])
        self.assertFalse(kwargs["force"])
        self.assertFalse(kwargs["smart"])
        self.assertTrue(kwargs["client_order_id"].startswith("RT_"))
        self.assertEqual(len(kwargs["client_order_id"]), 35)

    def test_live_pair_limit_persists_canonical_intent_and_venue_values(self):
        executor = SimpleNamespace(free_balance=lambda _asset: 1000.0)
        order = {"orderId": 7, "price": "99.25", "origQty": "0.8"}
        with tempfile.TemporaryDirectory(prefix="rtrade-store-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
                patch.object(rtrade.mkt, "place", return_value=order):
                venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
                ticket = venue.place_limit("BUY", 99.36, 1.0, "pair-1")

            intent = store.active("TAOUSDC")[0]["intents"]["limit:BUY"]
        self.assertEqual(ticket.price, 99.25)
        self.assertEqual(ticket.qty, 0.8)
        self.assertEqual(intent["requested_price"], 99.36)
        self.assertEqual(intent["submitted_price"], 99.25)
        self.assertEqual(intent["submitted_qty"], 0.8)
        self.assertEqual(intent["order_id"], "7")

    def test_replacement_limit_uses_distinct_persisted_client_id(self):
        executor = SimpleNamespace(
            free_balance=lambda _asset: 1000.0,
            cancel_order=Mock(return_value=None),
        )
        responses = [
            {"orderId": 7, "price": "100.64", "origQty": "1.0"},
            {"orderId": 8, "price": "101.25", "origQty": "0.7"},
        ]
        with tempfile.TemporaryDirectory(prefix="rtrade-revision-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
                 patch.object(rtrade.mkt, "place", side_effect=responses) as place, \
                 patch("lock.trade_cooldown.release_pair_leg", return_value=True):
                venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
                old = venue.place_limit(
                    "SELL", 100.64, 1.0, "pair-reprice", revision=0)
                self.assertTrue(venue.cancel(old.order_id))
                new = venue.place_limit(
                    "SELL", 101.25, 0.7, "pair-reprice", revision=1)
                restarted = rtrade._LivePairVenue(
                    "TAOUSDC", pair_store=RTradePairStore(store.path))

            record = RTradePairStore(store.path).active("TAOUSDC")[0]
            initial = record["intents"]["limit:SELL"]
            replacement = record["intents"]["limit_1:SELL"]
            restored = restarted._canonical_intent(record, replacement)

        old_cid = place.call_args_list[0].kwargs["client_order_id"]
        new_cid = place.call_args_list[1].kwargs["client_order_id"]
        self.assertEqual(old_cid, rtrade_client_order_id(
            "pair-reprice", "SELL", "limit"))
        self.assertEqual(new_cid, rtrade_client_order_id(
            "pair-reprice", "SELL", "limit_1"))
        self.assertNotEqual(old_cid, new_cid)
        self.assertEqual((old.order_id, new.order_id), ("7", "8"))
        self.assertEqual(initial["client_order_id"], old_cid)
        self.assertEqual(replacement["client_order_id"], new_cid)
        self.assertEqual(restored["kind"], "limit_1")
        self.assertEqual(restored["client_order_id"], new_cid)

    def test_restart_merges_accepted_modifications_ahead_of_checkpoint(self):
        cases = [
            ("limit_replacement", "limit_1", 101.25, 0.7, "replacement-1", "open"),
            ("hard_stop", "hard_stop_1", 90.0, 1.0, "hard-stop-1", "open"),
        ]
        for kind, suffix, price, qty, order_id, expected_status in cases:
            with self.subTest(kind=kind):
                pair_id = f"pair-{kind}-crash"
                with tempfile.TemporaryDirectory(prefix=f"rtrade-merge-{kind}-") as root:
                    store = RTradePairStore(os.path.join(root, "pairs.json"))
                    state = _old_exposed_checkpoint(pair_id)
                    store.begin("TAOUSDC", pair_id, "BUY", 1.0)
                    store.checkpoint(pair_id, state, terminal=False)
                    client_id = rtrade_client_order_id(pair_id, "SELL", suffix)
                    store.intent(
                        pair_id, "SELL", price, qty, client_id,
                        kind=suffix, symbol="TAOUSDC")
                    store.accepted(pair_id, "SELL", order_id, kind=suffix)
                    venue, executor = _restart_merge_venue(
                        root, store, {
                            order_id: OrderStatus(expected_status, 0.0, 0.0, 0.0, "NEW")})
                    record = store.active("TAOUSDC")[0]

                    with patch.object(rtrade.mkt, "place") as submit:
                        merged = venue.merge_checkpoint_intents(record, state)

                    if kind == "limit_replacement":
                        restored = rtrade.PairCoordinator.from_state(
                            venue, rtrade.PairPolicy(adjustment_fraction=0.0064), merged)

                submit.assert_not_called()
                self.assertEqual(executor.lookup_calls, [])
                
                if kind == "limit_replacement":
                    self.assertEqual(
                        executor.status_calls, [("TAOUSDC", order_id)])
                    self.assertEqual(merged["limit_revisions"]["SELL"], 1)
                    self.assertEqual(restored.limit_revisions["SELL"], 1)
                    self.assertIn(order_id, [ticket.order_id for ticket in restored.tickets])
                else:
                    self.assertEqual(merged["phase"], "stopping")
                    self.assertEqual(merged["hard_stop_revision"], 1)
                    self.assertEqual(merged["stop_order_id"], order_id)
                    self.assertIn(order_id, [ticket["order_id"] for ticket in merged["tickets"]])

    def test_restart_pending_intent_recovers_by_client_id_without_submit(self):
        pair_id = "pair-pending-crash"
        with tempfile.TemporaryDirectory(prefix="rtrade-merge-pending-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            state = _old_exposed_checkpoint(pair_id)
            store.begin("TAOUSDC", pair_id, "BUY", 1.0)
            store.checkpoint(pair_id, state, terminal=False)
            client_id = rtrade_client_order_id(pair_id, "SELL", "limit_1")
            store.intent(
                pair_id, "SELL", 101.25, 0.7, client_id,
                kind="limit_1", symbol="TAOUSDC")
            venue, executor = _restart_merge_venue(
                root, store, {
                    "recovered-by-cid": OrderStatus(
                        "open", 0.0, 0.0, 0.0, "NEW")},
                lookups={
                    client_id: {
                        "orderId": "recovered-by-cid", "status": "NEW",
                        "price": "101.25", "origQty": "0.7",
                    },
                })
            record = store.active("TAOUSDC")[0]

            with patch.object(rtrade.mkt, "place") as submit:
                merged = venue.merge_checkpoint_intents(record, state)
            persisted = RTradePairStore(store.path).active(
                "TAOUSDC")[0]["intents"]["limit_1:SELL"]

        submit.assert_not_called()
        self.assertEqual(
            executor.lookup_calls, [("TAOUSDC", client_id)])
        self.assertEqual(persisted["order_id"], "recovered-by-cid")
        self.assertEqual(merged["limit_revisions"]["SELL"], 1)
        self.assertIn(
            "recovered-by-cid",
            [ticket["order_id"] for ticket in merged["tickets"]])

    def test_live_pair_guard_refusal_does_not_run_response_loss_recovery(self):
        class Executor:
            name = "Binance"

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def order_by_client_id(*_args, **_kwargs):
                raise AssertionError("pre-submit refusal must not query venue")

        def refuse(*_args, **kwargs):
            kwargs["_outcome_context"].update(
                accepted=False, reason="trend_deferred", state="refused")
            return None

        with tempfile.TemporaryDirectory(prefix="rtrade-refusal-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=Executor()), \
                 patch.object(rtrade.mkt, "place", side_effect=refuse) as place:
                venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
                ticket = venue.place_limit("SELL", 101.0, 1.0, "pair-1")

            intent = store.active("TAOUSDC")[0]["intents"]["limit:SELL"]

        self.assertIsNone(ticket)
        place.assert_called_once()
        self.assertEqual(intent["refusal_reason"], "trend_deferred")
        self.assertEqual(intent["submit_status"], "refused_before_submit")
        self.assertNotIn("order_id", intent)
        self.assertFalse(venue.recovery_blocked)

    def test_live_pair_recovers_lost_submit_response_without_resubmit(self):
        class Executor:
            name = "Binance"

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def reconciliation_capabilities():
                return OrderReconciliationCapabilities(
                    lookup_by_client_order_id=True,
                    status_by_order_id=True,
                    cancel_by_order_id=True,
                    list_open_orders=True,
                )

            @staticmethod
            def order_by_client_id(_symbol, _client_id):
                return {
                    "orderId": "88", "status": "NEW",
                    "price": "99.25", "origQty": "0.8",
                }

            @staticmethod
            def order_status(_symbol, _order_id):
                return OrderStatus("open", 0.0, 0.0, 0.0)

        with tempfile.TemporaryDirectory(prefix="rtrade-recover-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            client_id = rtrade_client_order_id("pair-1", "BUY", "limit")
            store.intent(
                "pair-1", "BUY", 99.36, 1.0, client_id,
                kind="limit", symbol="TAOUSDC", start_side="BUY")
            record = store.active("TAOUSDC")[0]
            stored = record["intents"]["limit:BUY"]
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=Executor()), \
                 patch.object(rtrade.mkt, "place") as place:
                venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
                ticket, snapshot = venue.recover_intent(record, stored)

            canonical = store.active("TAOUSDC")[0]["intents"]["limit:BUY"]
        place.assert_not_called()
        self.assertEqual(ticket.order_id, "88")
        self.assertEqual(snapshot.status, "open")
        self.assertEqual(canonical["order_id"], "88")

    def test_place_limit_recovers_response_loss_on_later_call_without_resubmit(self):
        class Executor:
            name = "Binance"

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def reconciliation_capabilities():
                return OrderReconciliationCapabilities(
                    lookup_by_client_order_id=True,
                    status_by_order_id=True,
                    cancel_by_order_id=True,
                    list_open_orders=True,
                )

            @staticmethod
            def order_by_client_id(_symbol, _client_id):
                return {
                    "orderId": "88", "status": "NEW",
                    "price": "99.25", "origQty": "0.8",
                }

            @staticmethod
            def order_status(_symbol, _order_id):
                return OrderStatus("open", 0.0, 0.0, 0.0)

        with tempfile.TemporaryDirectory(prefix="rtrade-live-loss-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                patch.object(rtrade.mkt, "provider_by_name", return_value=Executor()), \
                 patch.object(rtrade.mkt, "place", return_value=None) as place:
                venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
                ticket = venue.place_limit("BUY", 99.36, 1.0, "pair-1")
                self.assertIsNone(ticket)
                ticket = venue.place_limit("BUY", 99.36, 1.0, "pair-1")

            canonical = store.active("TAOUSDC")[0]["intents"]["limit:BUY"]
        place.assert_called_once()
        self.assertEqual(ticket.order_id, "88")
        self.assertEqual(canonical["order_id"], "88")
        self.assertFalse(venue.recovery_blocked)

    def test_place_limit_never_reuses_ambiguous_client_id_after_absence(self):
        class Executor:
            name = "Binance"

            def __init__(self):
                self.lookup_calls = 0

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def reconciliation_capabilities():
                return OrderReconciliationCapabilities(
                    lookup_by_client_order_id=True,
                    status_by_order_id=True,
                    cancel_by_order_id=True,
                    list_open_orders=True,
                )

            def order_by_client_id(self, _symbol, _client_id):
                self.lookup_calls += 1
                return None

            @staticmethod
            def order_status(_symbol, _order_id):
                return OrderStatus("open", 0.0, 0.0, 0.0)

        executor = Executor()
        recovery_now = [100.0]
        with tempfile.TemporaryDirectory(prefix="rtrade-live-retry-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
                 patch.object(rtrade.mkt, "place", return_value=None) as place:
                venue = rtrade._LivePairVenue(
                    "TAOUSDC", pair_store=store,
                    recovery_clock=lambda: recovery_now[0])
                self.assertIsNone(
                    venue.place_limit("BUY", 99.36, 1.0, "pair-1"))
                self.assertIsNone(
                    venue.place_limit("BUY", 99.36, 1.0, "pair-1"))
                # The same coordinator tick cannot count as another absence.
                self.assertIsNone(
                    venue.place_limit("BUY", 99.36, 1.0, "pair-1"))
                self.assertEqual(executor.lookup_calls, 1)
                recovery_now[0] += rtrade.RTRADE_PAIR_POLL_SEC
                self.assertIsNone(
                    venue.place_limit("BUY", 99.36, 1.0, "pair-1"))
                recovery_now[0] += rtrade.RTRADE_PAIR_POLL_SEC
                self.assertIsNone(
                    venue.place_limit("BUY", 99.36, 1.0, "pair-1"))

            canonical = store.active("TAOUSDC")[0]["intents"]["limit:BUY"]
        place.assert_called_once()
        self.assertEqual(executor.lookup_calls, 2)
        self.assertEqual(
            canonical["recovery_state"], "absence_confirmed_no_reuse")
        self.assertEqual(canonical["lookup_misses"], 2)

    def test_live_pair_finds_late_fill_after_first_absence_without_resubmit(self):
        class Executor:
            name = "Binance"

            def __init__(self):
                self.lookup_results = [
                    None,
                    {
                        "orderId": "89", "status": "FILLED",
                        "price": "99.3", "origQty": "0.9",
                    },
                ]

            @staticmethod
            def free_balance(_asset):
                return 1000.0

            @staticmethod
            def reconciliation_capabilities():
                return OrderReconciliationCapabilities(
                    lookup_by_client_order_id=True,
                    status_by_order_id=True,
                    cancel_by_order_id=True,
                    list_open_orders=True,
                )

            def order_by_client_id(self, _symbol, _client_id):
                return self.lookup_results.pop(0)

            @staticmethod
            def order_status(_symbol, _order_id):
                return OrderStatus("closed", 0.9, 89.37, 0.01)

        executor = Executor()
        recovery_now = [100.0]
        with tempfile.TemporaryDirectory(prefix="rtrade-retry-") as root:
            store = RTradePairStore(os.path.join(root, "pairs.json"))
            client_id = rtrade_client_order_id("pair-1", "BUY", "limit")
            store.intent(
                "pair-1", "BUY", 99.36, 1.0, client_id,
                kind="limit", symbol="TAOUSDC", start_side="BUY")
            record = store.active("TAOUSDC")[0]
            stored = record["intents"]["limit:BUY"]
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
                 patch.object(rtrade.mkt, "place") as place:
                venue = rtrade._LivePairVenue(
                    "TAOUSDC", pair_store=store,
                    recovery_clock=lambda: recovery_now[0])
                with self.assertRaisesRegex(
                        RuntimeError, "ambiguous"):
                    venue.recover_intent(record, stored)
                recovery_now[0] += rtrade.RTRADE_PAIR_POLL_SEC
                record = store.active("TAOUSDC")[0]
                stored = record["intents"]["limit:BUY"]
                ticket, snapshot = venue.recover_intent(record, stored)

            canonical = store.active("TAOUSDC")[0]["intents"]["limit:BUY"]
        place.assert_not_called()
        self.assertEqual(ticket.order_id, "89")
        self.assertEqual(snapshot.status, "closed")
        self.assertEqual(canonical["attempt"], 1)
        self.assertEqual(canonical["client_order_id"], client_id)

    def test_lookup_error_blocks_progress_and_keeps_intent(self):
        cases = ["recovery", "new_round"]
        for case in cases:
            with self.subTest(case=case):
                class Executor:
                    name = "Binance"

                    @staticmethod
                    def free_balance(_asset):
                        return 1000.0

                    @staticmethod
                    def reconciliation_capabilities():
                        return OrderReconciliationCapabilities(
                            lookup_by_client_order_id=True,
                            status_by_order_id=True,
                            cancel_by_order_id=True,
                            list_open_orders=True,
                        )

                    @staticmethod
                    def order_by_client_id(_symbol, _client_id):
                        raise TimeoutError("lookup unavailable")

                with tempfile.TemporaryDirectory(prefix=f"rtrade-ambiguous-{case}-") as root:
                    store = RTradePairStore(os.path.join(root, "pairs.json"))
                    client_id = rtrade_client_order_id("pair-1", "BUY", "limit")
                    
                    if case == "recovery":
                        store.intent(
                            "pair-1", "BUY", 99.36, 1.0, client_id,
                            kind="limit", symbol="TAOUSDC", start_side="BUY")
                        
                    with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                         patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                         patch.object(rtrade.mkt, "provider_by_name", return_value=Executor()), \
                         patch.object(rtrade.mkt, "place", return_value=None) as place:
                        venue = rtrade._LivePairVenue("TAOUSDC", pair_store=store)
                        if case == "recovery":
                            record = store.active("TAOUSDC")[0]
                            stored = record["intents"]["limit:BUY"]
                            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                                venue.recover_intent(record, stored)
                        else:
                            ticket = venue.place_limit("BUY", 99.36, 1.0, "pair-1")
                            self.assertIsNone(ticket)
                            ticket = venue.place_limit("BUY", 99.36, 1.0, "pair-1")
                            self.assertIsNone(ticket)

                    canonical = store.active("TAOUSDC")[0]["intents"]["limit:BUY"]

                if case == "recovery":
                    place.assert_not_called()
                else:
                    place.assert_called_once()
                    self.assertTrue(venue.recovery_blocked)
                    self.assertEqual(
                        venue.last_place_failure_reason("BUY"),
                        "buy_recovery_pending")
                self.assertIn("lookup_error", canonical)

    def test_live_pair_hard_stop_reconciles_and_uses_audited_market_exit(self):
        precision = SimpleNamespace(volume_decimals=3, order_min=0.001)
        executor = SimpleNamespace(
            name="Binance",
            free_balance=lambda _asset: 0.4,
            fee_cap_quantity=lambda *_args: 0.39,
            order_filter_refusal=Mock(return_value=None),
            pair_precision=lambda _symbol: precision,
            preflight_order=lambda *args, **kwargs: None,
            submit_order=lambda *args, **kwargs: "M9")
        with tempfile.TemporaryDirectory(prefix="rtrade-test-audit-") as audit_dir:
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": audit_dir}), \
                 patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
                 patch.object(rtrade.mkt, "get_current_price", return_value=90.0), \
                 patch.object(executor, "submit_order", wraps=executor.submit_order) as submit:
                venue = rtrade._LivePairVenue("TAOUSDC")
                ticket = venue.place_market_exit(
                    "SELL", 0.4, "fast_fill_hard_stop", pair_id="pair-9")

        self.assertEqual((ticket.order_id, ticket.qty, ticket.pair_id),
                         ("M9", 0.39, "pair-9"))
        args, kwargs = submit.call_args
        self.assertEqual(args[:4], ("TAOUSDC", "SELL", 0.39, None))
        self.assertTrue(kwargs["market"])
        self.assertIn("rtrade:fast_fill_hard_stop:pair-9", kwargs["kind"])
        executor.order_filter_refusal.assert_called_once_with(
            "TAOUSDC", "SELL", 90.0, 0.39, market=True,
            enforce_business_minimum=False)

    def test_hard_stop_refusal_preserves_clean_state(self):
        cases = [
            ("preflight", {"preflight_error": SubmissionRefused("cache_stale")}),
            ("submit", {"submit_error": SubmissionRefused("cache_stale")}),
        ]
        for phase, kwargs in cases:
            with self.subTest(phase=phase):
                with tempfile.TemporaryDirectory(prefix=f"rtrade-{phase}-refusal-") as root:
                    store_path = os.path.join(root, "pairs.json")
                    store = RTradePairStore(store_path)
                    store.begin("TAOUSDC", f"pair-{phase}", "SELL", 0.4)
                    venue, executor = _hard_stop_venue(root, store, **kwargs)
                    with self.assertRaisesRegex(SubmissionRefused, "cache_stale"):
                        venue.place_market_exit(
                            "SELL", 0.4, "hard_stop", pair_id=f"pair-{phase}")
                    reopened = RTradePairStore(store_path).active("TAOUSDC")[0]

                self.assertNotIn("hard_stop:SELL", reopened["intents"])
                if phase == "preflight":
                    executor.submit_order.assert_not_called()
                else:
                    executor.submit_order.assert_called_once()

    def test_post_cancel_hard_stop_refusal_retains_recoverable_intent(self):
        permit = object()
        with tempfile.TemporaryDirectory(prefix="rtrade-post-cancel-refusal-") as root:
            store_path = os.path.join(root, "pairs.json")
            store = RTradePairStore(store_path)
            store.begin("TAOUSDC", "pair-11b", "SELL", 0.4)
            venue, executor = _hard_stop_venue(
                root, store,
                submit_error=SubmissionRefused("account_cache_not_fresh"))
            with self.assertRaisesRegex(
                    SubmissionRefused, "account_cache_not_fresh"):
                venue.place_market_exit(
                    "SELL", 0.4, "hard_stop", pair_id="pair-11b",
                    cache_permit=permit)
            refused = store.active("TAOUSDC")[0]["intents"]["hard_stop:SELL"]
            self.assertEqual(
                refused["submit_status"], "refused_before_submit")
            self.assertEqual(
                refused["refusal_reason"], "account_cache_not_fresh")
            executor.submit_order.side_effect = None
            executor.submit_order.return_value = "M11b"
            ticket = venue.place_market_exit(
                "SELL", 0.4, "hard_stop", pair_id="pair-11b")
            reopened = RTradePairStore(store_path).active("TAOUSDC")[0]

        pending = reopened["intents"]["hard_stop:SELL"]
        self.assertEqual(ticket.order_id, "M11b")
        self.assertEqual(pending["order_id"], "M11b")
        self.assertEqual(
            pending["client_order_id"],
            rtrade_client_order_id("pair-11b", "SELL", "hard_stop"),
        )
        executor.preflight_order.assert_called_once()
        self.assertEqual(executor.submit_order.call_count, 2)
        self.assertIs(
            executor.submit_order.call_args_list[0].kwargs["cache_permit"],
            permit)

    def test_hard_stop_response_loss_reconciles_later_without_resubmit(self):
        class Executor:
            name = "Binance"

            def __init__(self):
                self.preflight_order = Mock(return_value=None)
                self.submit_order = Mock(
                    side_effect=ProviderError("response lost"))
                self.lookup_results = [
                    None,
                    {
                        "orderId": "M12", "status": "FILLED",
                        "price": "90", "origQty": "0.4",
                    },
                ]

            @staticmethod
            def free_balance(_asset):
                return 1.0

            @staticmethod
            def fee_cap_quantity(*_args):
                return 1.0

            @staticmethod
            def pair_precision(_symbol):
                return None

            @staticmethod
            def reconciliation_capabilities():
                return OrderReconciliationCapabilities(
                    lookup_by_client_order_id=True,
                    status_by_order_id=True,
                    cancel_by_order_id=True,
                    list_open_orders=True,
                )

            def order_by_client_id(self, _symbol, _client_id):
                return self.lookup_results.pop(0)

            @staticmethod
            def order_status(_symbol, _order_id):
                return OrderStatus("closed", 0.4, 36.0, 0.01)

        recovery_now = [100.0]
        with tempfile.TemporaryDirectory(prefix="rtrade-submit-unknown-") as root:
            store_path = os.path.join(root, "pairs.json")
            store = RTradePairStore(store_path)
            store.begin("TAOUSDC", "pair-12", "SELL", 0.4)
            executor = Executor()
            with patch.dict(os.environ, {"EXECUTION_AUDIT_DIR": root}), \
                 patch.object(
                     rtrade.mkt, "provider_name_for", return_value="Binance"), \
                 patch.object(
                     rtrade.mkt, "provider_by_name", return_value=executor):
                venue = rtrade._LivePairVenue(
                    "TAOUSDC", pair_store=store,
                    recovery_clock=lambda: recovery_now[0])
            venue.current_price = lambda: 90.0
            self.assertIsNone(venue.place_market_exit(
                "SELL", 0.4, "hard_stop", pair_id="pair-12"))
            self.assertIsNone(venue.place_market_exit(
                "SELL", 0.4, "hard_stop", pair_id="pair-12"))
            recovery_now[0] += rtrade.RTRADE_PAIR_POLL_SEC
            ticket = venue.place_market_exit(
                "SELL", 0.4, "hard_stop", pair_id="pair-12")
            reopened = RTradePairStore(store_path).active("TAOUSDC")[0]

        pending = reopened["intents"]["hard_stop:SELL"]
        self.assertEqual(ticket.order_id, "M12")
        self.assertEqual(pending["order_id"], "M12")
        self.assertEqual(
            pending["client_order_id"],
            rtrade_client_order_id("pair-12", "SELL", "hard_stop"),
        )
        executor.submit_order.assert_called_once()
        self.assertFalse(venue.recovery_blocked)

    def test_live_pair_cancel_releases_only_its_cooldown_leg(self):
        executor = SimpleNamespace(
            cancel_order=lambda *_args: None,
            free_balance=lambda _asset: 1000.0)
        order = {"orderId": 7, "price": "99.36", "origQty": "1"}
        with patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
             patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
             patch.object(rtrade.mkt, "place", return_value=order), \
             patch("lock.trade_cooldown.release_pair_leg", return_value=True) as release:
            venue = rtrade._LivePairVenue("TAOUSDC")
            venue.place_limit("BUY", 99.36, 1.0, "pair-1")
            self.assertTrue(venue.cancel("7"))

        release.assert_called_once_with("TAOUSDC", "pair-1", "BUY")

    def test_live_pair_adapter_rejects_insufficient_asset_before_submit(self):
        executor = SimpleNamespace(free_balance=lambda _asset: 0.0)
        with patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
             patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
             patch.object(rtrade.mkt, "place") as place:
            venue = rtrade._LivePairVenue("TAOUSDC")
            buy = venue.place_limit("BUY", 100.0, 1.0, "pair-buy")
            sell = venue.place_limit("SELL", 101.0, 1.0, "pair-sell")

        self.assertIsNone(buy)
        self.assertIsNone(sell)
        self.assertEqual(venue.last_place_failure_reason("BUY"),
                         "buy_insufficient_funds:USDC")
        self.assertEqual(venue.last_place_failure_reason("SELL"),
                         "sell_insufficient_funds:TAO")
        place.assert_not_called()

    def test_live_pair_adapter_leaves_partial_balance_clamp_to_provider(self):
        executor = SimpleNamespace(free_balance=lambda _asset: 10.0)
        adjusted = {"orderId": 8, "price": "100", "origQty": "0.099"}
        with patch.object(rtrade.mkt, "provider_name_for", return_value="Binance"), \
             patch.object(rtrade.mkt, "provider_by_name", return_value=executor), \
             patch.object(rtrade.mkt, "place", return_value=adjusted) as place:
            venue = rtrade._LivePairVenue("TAOUSDC")
            ticket = venue.place_limit("BUY", 100.0, 1.0, "pair-buy")

        self.assertEqual(ticket.qty, 0.099)
        place.assert_called_once()

    def test_pair_reuses_exactly_two_workers_between_rounds(self):
        bot = _bot()
        barrier = threading.Barrier(2)
        rounds = []

        def buy(_current, _filled):
            ident = threading.get_ident()
            barrier.wait(timeout=1.0)
            rounds[-1].append(ident)
            return 99.0

        def sell(_current, _filled):
            ident = threading.get_ident()
            barrier.wait(timeout=1.0)
            rounds[-1].append(ident)
            return 101.0

        bot.repetitive_buy = buy
        bot.repetitive_sell = sell
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rtrade-test") as executor:
            for _ in range(3):
                rounds.append([])
                self.assertEqual(bot._run_pair(executor, 100.0), (99.0, 101.0))

        worker_sets = [set(ids) for ids in rounds]
        self.assertTrue(all(len(ids) == 2 for ids in worker_sets))
        self.assertTrue(all(ids == worker_sets[0] for ids in worker_sets[1:]))

    def test_worker_exception_propagation_behavior(self):
        cases = ["immediate", "waits_for_other"]
        for case in cases:
            with self.subTest(case=case):
                bot = _bot()
                sell_started = threading.Event()
                release_sell = threading.Event()
                owner_finished = threading.Event()
                errors = []

                def fail(_current, _filled):
                    raise RuntimeError(f"buy worker failed in {case}")

                def slow_sell(_current, _filled):
                    sell_started.set()
                    release_sell.wait(timeout=1.0)
                    return 101.0

                bot.repetitive_buy = fail
                if case == "immediate":
                    bot.repetitive_sell = lambda _current, _filled: 101.0
                else:
                    bot.repetitive_sell = slow_sell

                if case == "immediate":
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        with self.assertRaisesRegex(RuntimeError, f"buy worker failed in {case}"):
                            bot._run_pair(executor, 100.0)
                else:
                    def run_owner(executor):
                        try:
                            bot._run_pair(executor, 100.0)
                        except Exception as exc:  # noqa: BLE001
                            errors.append(exc)
                        finally:
                            owner_finished.set()

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        owner = threading.Thread(target=run_owner, args=(executor,))
                        owner.start()
                        self.assertTrue(sell_started.wait(timeout=1.0))
                        self.assertFalse(
                            owner_finished.wait(timeout=0.05),
                            "the owner must not start another round while the old SELL is running",
                        )
                        release_sell.set()
                        owner.join(timeout=1.0)

                    self.assertFalse(owner.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertRegex(str(errors[0]), f"buy worker failed in {case}")

    def test_legacy_replacement_preflight_refusal_preserves_active_order(self):
        cases = (
            ("BUY", "repetitive_buy", 101.0, 99.36),
            ("SELL", "repetitive_sell", 99.0, 100.64),
        )
        for side, method_name, opposite_fill, expected_price in cases:
            with self.subTest(side=side):
                bot = _bot()
                preflight = Mock(
                    side_effect=SubmissionRefused("cache_stale"))
                executor = SimpleNamespace(preflight_order=preflight)
                first_order = {
                    "orderId": 7, "price": str(expected_price)}
                with patch.object(
                        rtrade.api, "get_current_price", return_value=100.0), \
                     patch.object(
                         rtrade, "_order_fully_filled", return_value=False), \
                     patch.object(
                         rtrade.mkt, "latest_fill_price", return_value=None), \
                     patch.object(
                         rtrade, "_cancel_order_confirmed") as cancel, \
                     patch.object(
                         rtrade.mkt, "place", return_value=first_order) as place, \
                     patch.object(
                         rtrade.mkt, "provider_name_for", return_value="Binance"), \
                     patch.object(
                         rtrade.mkt, "provider_by_name", return_value=executor), \
                     patch.object(rtrade.time, "sleep", return_value=None):
                    with self.assertRaisesRegex(SubmissionRefused, "cache_stale"):
                        getattr(bot, method_name)(100.0, opposite_fill)

                cancel.assert_not_called()
                place.assert_called_once()
                preflight.assert_called_once_with(
                    "TAOUSDC", side, 100.0, expected_price,
                    market=False, kind="rtrade_legacy_quote")

    def test_legacy_replacement_uses_terminal_unfilled_remainder(self):
        cases = (
            ("BUY", "repetitive_buy", 101.0, 99.36),
            ("SELL", "repetitive_sell", 99.0, 100.64),
        )
        for side, method_name, opposite_fill, first_price in cases:
            with self.subTest(side=side):
                bot = _bot()
                permit = object()
                calls = []

                def place(*args, **kwargs):
                    calls.append((args, kwargs))
                    if len(calls) == 1:
                        return {"orderId": 7, "price": str(first_price)}
                    raise RuntimeError("stop after observing replacement")

                with patch.object(
                        rtrade.api, "get_current_price", return_value=100.0), \
                     patch.object(
                         rtrade, "_order_fully_filled", return_value=False), \
                     patch.object(
                         rtrade.mkt, "latest_fill_price", return_value=None), \
                     patch.object(
                         rtrade.mkt, "cancel_order", return_value=None), \
                     patch.object(
                         rtrade.mkt, "order_status",
                         return_value=OrderStatus(
                             "canceled", 25.0, 2500.0, 0.1,
                             "CANCELED")), \
                     patch.object(
                         rtrade, "_preflight_routed_order",
                         return_value=permit), \
                     patch.object(
                         rtrade.mkt, "place", side_effect=place), \
                     patch.object(rtrade.time, "sleep", return_value=None):
                    with self.assertRaisesRegex(
                            RuntimeError, "stop after observing replacement"):
                        getattr(bot, method_name)(100.0, opposite_fill)

                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[1][0][1], side)
                self.assertEqual(calls[1][0][3], 75.0)
                self.assertIs(calls[1][1]["cache_permit"], permit)

    def test_legacy_cancel_full_fill_skips_same_side_replacement(self):
        cases = (
            ("BUY", "SELL", "repetitive_buy", 101.0, 99.36),
            ("SELL", "BUY", "repetitive_sell", 99.0, 100.64),
        )
        for side, followup_side, method_name, opposite_fill, first_price in cases:
            with self.subTest(side=side):
                bot = _bot()
                first_order = {"orderId": 7, "price": str(first_price)}
                with patch.object(
                        rtrade.api, "get_current_price", return_value=100.0), \
                     patch.object(
                         rtrade, "_order_fully_filled", return_value=False), \
                     patch.object(
                         rtrade.mkt, "latest_fill_price", return_value=None), \
                     patch.object(
                         rtrade.mkt, "cancel_order", return_value=None), \
                     patch.object(
                         rtrade.mkt, "order_status",
                         return_value=OrderStatus(
                             "canceled", 100.0, 10_000.0, 0.1,
                             "CANCELED")), \
                     patch.object(
                         rtrade, "_preflight_routed_order",
                         return_value=object()), \
                     patch.object(
                         rtrade, "_followup_force", return_value=False), \
                     patch.object(
                         rtrade.mkt, "place",
                         side_effect=[first_order, {"orderId": 8}]) as place, \
                     patch.object(rtrade.time, "sleep", return_value=None):
                    result = getattr(bot, method_name)(100.0, opposite_fill)

                self.assertEqual(result, first_price)
                self.assertEqual(len(place.call_args_list), 2)
                self.assertEqual(place.call_args_list[1].args[1], followup_side)
                self.assertEqual(place.call_args_list[1].args[3], bot.qty)

    def test_cancel_fill_followup_uses_same_trend_policy(self):
        """The sixth follow-up must not bypass _followup_force."""
        bot = _bot()
        first_order = {"orderId": 7, "price": "101.0"}
        with patch.object(rtrade.api, "get_current_price", return_value=100.0), \
             patch.object(rtrade, "_order_fully_filled", side_effect=[False, True]), \
             patch.object(rtrade.mkt, "latest_fill_price", return_value=None), \
             patch.object(rtrade, "_cancel_order_confirmed", return_value=False), \
             patch.object(rtrade, "_preflight_routed_order", return_value=None), \
             patch.object(rtrade.mkt, "place", side_effect=[first_order, {"orderId": 8}]) as place, \
             patch.object(rtrade, "_followup_force", return_value=False) as policy, \
             patch.object(rtrade.time, "sleep", return_value=None):
            result = bot.repetitive_sell(100.0, 100.0)

        self.assertEqual(result, 101.0)
        policy.assert_called_once_with("TAOUSDC", "BUY")
        self.assertFalse(place.call_args_list[1].kwargs["force"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
