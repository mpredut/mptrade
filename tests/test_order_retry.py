"""Tests for the replacement outbox store and its due/expiry policy.

All queue files are isolated in a temporary directory.
"""
import os
import sys
import time
import subprocess
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_retry as oq
from providers.strategy_executor import OrderReconciliationCapabilities, OrderStatus, ProviderError
from types import SimpleNamespace
import instrument as inst_mod
import order_lifecycle as ol
import pytest
from order_lifecycle import extract_order_restore_params
from providers.execution_audit import AuditedStrategyExecutor
from tests.conftest import DummySpotExecutionProvider, TempDirTestCase


class OrderRetryStoreTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        oq.QUEUE_FILE = os.path.join(self.tmp, "order_retry_queue.jsonl")
        oq.LOCK_FILE = os.path.join(self.tmp, "order_retry_queue.lock")
        self._reset_env(rewrite=False)

    def _terminal_remainder(self, *, terminal_at=1400.0):
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 2.0, {}, requested_price=100.0,
            now=1000.0, kind="terminal-remainder")
        self.assertTrue(oq.complete_claim(
            oq.claim([record_id], now=1001.0)[0],
            "accepted",
            order={"orderId": 91, "status": "NEW"},
            now=1001.0))
        claimed = oq.claim([record_id], now=terminal_at)[0]
        transition = oq.advance_claimed_status(
            claimed,
            OrderStatus(
                "expired", 0.5, 50.0, 0.01, venue_status="EXPIRED"),
            now=terminal_at,
        )
        self.assertEqual(transition.action, "retry_terminal")
        remainder = oq.get(record_id)
        self.assertEqual(remainder["lifecycle"], "submit_pending")
        self.assertTrue(remainder["order_history"])
        self.assertGreater(remainder["delivered_qty"], 0)
        return record_id, remainder

    def _reset_env(self, *, rewrite=True):
        if rewrite:
            oq.rewrite([])
        oq.RETRY_ENABLED = True
        oq.RETRY_INTERVAL_SEC = 300.0
        oq.RETRY_TTL_SEC = 86400.0
        oq.RETRY_MAX_ATTEMPTS = 0
        oq.RETRY_PRICE_TOL = 0.002
        oq.RETRY_DEDUP = True
        oq.RETRY_MAX_QUEUE = 500
        oq.RETRY_CLAIM_LEASE_SEC = 120.0
        oq.RETRY_NOT_FOUND_MAX_AGE_SEC = 30 * 24 * 60 * 60

    def test_enqueue_and_storage_basics(self):
        self._reset_env()
        with self.subTest(msg="enqueue_and_load"):
            i = oq.enqueue("BTCUSDC", "BUY", 1.0,
                           {"safeback_seconds": 999, "force": False, "smart": False}, now=1000.0)
            self.assertIsNotNone(i)
            items = oq.load_all()
            self.assertEqual(len(items), 1)
            r = items[0]
            self.assertEqual(r["symbol"], "BTCUSDC")
            self.assertEqual(r["side"], "BUY")
            self.assertEqual(r["qty"], 1.0)
            self.assertEqual(r["place_kwargs"]["safeback_seconds"], 999)
            self.assertEqual(r["attempts"], 0)
            self.assertNotIn("price", r)

        with self.subTest(msg="enqueue_captures_price_intent"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0, ref_price=62950.0,
                       now=1000.0)
            r = oq.load_all()[0]
            self.assertEqual(r["requested_price"], 63000.0)
            self.assertEqual(r["ref_price"], 62950.0)

        with self.subTest(msg="enqueue_disabled_returns_none"):
            self._reset_env()
            oq.RETRY_ENABLED = False
            self.assertIsNone(oq.enqueue("BTCUSDC", "BUY", 1.0, {}))
            self.assertEqual(oq.load_all(), [])

        with self.subTest(msg="multiple_appends_preserved"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0)
            oq.enqueue("TAOUSDC", "SELL", 2.0, {}, now=1001.0)
            self.assertEqual(len(oq.load_all()), 2)

        with self.subTest(msg="rewrite_removes_items"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0)
            oq.enqueue("TAOUSDC", "SELL", 2.0, {}, now=1001.0)
            items = oq.load_all()
            oq.rewrite([items[1]])
            rem = oq.load_all()
            self.assertEqual(len(rem), 1)
            self.assertEqual(rem[0]["symbol"], "TAOUSDC")

        with self.subTest(msg="invalid_intent_is_not_enqueued"):
            self._reset_env()
            self.assertIsNone(oq.enqueue("BTCUSDC", "BUY", None, {}, now=1000.0))
            self.assertIsNone(oq.enqueue("BTCUSDC", "HOLD", 1.0, {}, now=1000.0))
            self.assertIsNone(oq.enqueue("", "BUY", 1.0, {}, now=1000.0))

    def test_claimed_enqueue_and_owner_identity(self):
        self._reset_env()
        with self.subTest(msg="awaiting_cancel_enqueue_is_idempotent_without_mutating_claim"):
            first = oq.enqueue(
                "BTCUSDC", "SELL", 2.0, {"smart": False},
                requested_price=101.0, now=1000.0, provider_name="Binance",
                lifecycle="awaiting_cancel", replaces_order_id="old-7",
                replaces_original_qty=2.0)
            claimed = oq.claim([first], now=1000.0)[0]
            before = oq.get(first)

            second = oq.enqueue(
                "BTCUSDC", "SELL", 1.0, {"smart": True},
                requested_price=99.0, now=1001.0, provider_name="binance",
                lifecycle="awaiting_cancel", replaces_order_id="old-7",
                replaces_original_qty=2.0)

            self.assertEqual(second, first)
            self.assertEqual(oq.get(first), before)
            self.assertEqual(before["claim_token"], claimed["claim_token"])

        with self.subTest(msg="enqueue_claimed_is_atomic_unique_and_producer_owned"):
            self._reset_env()
            first = oq.enqueue_claimed(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                now=1000.0, provider_name="Binance")
            second = oq.enqueue_claimed(
                "BTCUSDC", "BUY", 2.0, {}, requested_price=99.0,
                now=1001.0, provider_name="Binance")

            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(first["submission_state"], "producer_claimed")
            self.assertEqual(first["claim_revision"], first["revision"])
            self.assertGreater(
                first["claim_until"],
                first["created_ts"] + oq.RETRY_INTERVAL_SEC)
            self.assertEqual(first["producer_pid"], os.getpid())
            self.assertTrue(first["producer_process_start_id"].startswith("linux:"))
            self.assertEqual(len(oq.load_all()), 2)

        with self.subTest(msg="begin_claimed_submit_renews_exact_claim_and_marks_dispatch"):
            self._reset_env()
            claimed = oq.enqueue_claimed(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                now=1000.0, provider_name="Binance", lease_sec=301.0)

            refreshed = oq.begin_claimed_submit(
                claimed, now=1100.0, lease_sec=500.0)

            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed["id"], claimed["id"])
            self.assertEqual(refreshed["claim_token"], claimed["claim_token"])
            self.assertEqual(refreshed["claim_revision"], claimed["revision"])
            self.assertEqual(refreshed["claim_until"], 1600.0)
            self.assertEqual(refreshed["submission_state"], "producer_claimed")
            self.assertEqual(refreshed, oq.get(claimed["id"]))

        with self.subTest(msg="begin_claimed_submit_refuses_stale_exact_identity"):
            self._reset_env()
            claimed = oq.enqueue_claimed(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                now=1000.0, provider_name="Binance", lease_sec=301.0)
            stale = dict(claimed)
            stale["claim_token"] = "not-the-owner"
            before = oq.get(claimed["id"])

            self.assertIsNone(oq.begin_claimed_submit(stale, now=1100.0))
            self.assertEqual(oq.get(claimed["id"]), before)

        with self.subTest(msg="producer_owner_identity_distinguishes_liveness_and_pid_reuse"):
            record = {
                "producer_pid": 123,
                "producer_process_start_id": "linux:boot-a:10",
            }
            with patch.object(
                    oq, "_process_start_identity",
                    return_value="linux:boot-a:10"):
                self.assertEqual(oq.producer_claim_owner_state(record), "alive")
            with patch.object(
                    oq, "_process_start_identity",
                    return_value="linux:boot-a:11"):
                self.assertEqual(
                    oq.producer_claim_owner_state(record), "mismatched")
            with patch.object(
                    oq, "_process_start_identity",
                    side_effect=ProcessLookupError):
                self.assertEqual(oq.producer_claim_owner_state(record), "dead")
            with patch.object(
                    oq, "_process_start_identity",
                    side_effect=PermissionError):
                self.assertEqual(oq.producer_claim_owner_state(record), "unknown")

    def test_deduplication_and_client_order_id_invariants(self):
        self._reset_env()
        with self.subTest(msg="never_overwrites_possibly_submitted_record"):
            original_id = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                failure_reason="submit_ambiguous", now=1000.0)
            original = oq.get(original_id)

            new_id = oq.enqueue(
                "BTCUSDC", "BUY", 2.0, {}, requested_price=99.0,
                failure_reason="profit_guard", now=1001.0)

            self.assertNotEqual(new_id, original_id)
            self.assertEqual(oq.get(original_id), original)
            self.assertEqual(len(oq.load_all()), 2)

        with self.subTest(msg="same_side_refreshes_not_duplicates"):
            self._reset_env()
            a = oq.enqueue(
                "BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0,
                failure_reason="profit_guard", now=1000.0)
            b = oq.enqueue("BTCUSDC", "SELL", 2.0, {}, requested_price=63100.0, now=1001.0)
            self.assertEqual(a, b)
            items = oq.load_all()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["requested_price"], 63100.0)
            self.assertEqual(items[0]["qty"], 2.0)
            self.assertEqual(items[0]["created_ts"], 1000.0)
            self.assertNotEqual(items[0]["place_kwargs"]["client_order_id"],
                                f"OR_{a[:24]}_0")

        with self.subTest(msg="legacy_dedup_never_overwrites_an_accepted_tracker"):
            self._reset_env()
            first = oq.enqueue(
                "BTCUSDC", "SELL", 1.0, {}, requested_price=100.0, now=1000.0)
            oq.complete_claim(
                oq.claim([first], now=1001.0)[0],
                "accepted",
                order={"orderId": 7, "status": "NEW"},
                now=1001.0)

            second = oq.enqueue(
                "BTCUSDC", "SELL", 2.0, {}, requested_price=101.0, now=1002.0)

            self.assertNotEqual(first, second)
            rows = oq.load_all()
            self.assertEqual(len(rows), 2)
            accepted = next(row for row in rows if row["id"] == first)
            self.assertEqual(accepted["lifecycle"], "accepted")
            self.assertEqual(accepted["order_id"], "7")

        with self.subTest(msg="claimed_dedup_intent_is_never_mutated_by_concurrent_enqueue"):
            self._reset_env()
            first = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                failure_reason="submit_pending", now=1000.0)
            claimed = oq.claim([first], now=1400.0)[0]
            first_before = oq.get(first)

            second = oq.enqueue(
                "BTCUSDC", "BUY", 2.0, {}, requested_price=99.0,
                failure_reason="submit_pending", now=1401.0)

            self.assertNotEqual(second, first)
            self.assertEqual(oq.get(first), first_before)
            self.assertTrue(oq.complete_claim(
                claimed, "accepted", now=1402.0,
                order={"orderId": 17, "status": "NEW"}))
            records = {record["id"]: record for record in oq.load_all()}
            self.assertEqual(set(records), {first, second})
            self.assertEqual(records[first]["lifecycle"], "accepted")
            self.assertEqual(records[first]["order_id"], "17")
            self.assertEqual(records[first]["qty"], 1.0)
            self.assertEqual(records[second]["lifecycle"], "submit_pending")
            self.assertEqual(records[second]["qty"], 2.0)

        with self.subTest(msg="client_order_id_is_stable_for_one_revision"):
            self._reset_env()
            rid = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
            rec = oq.load_all()[0]
            self.assertEqual(rec["place_kwargs"]["client_order_id"], f"OR_{rid[:24]}_0")
            claimed = oq.claim([rid], now=1400.0)[0]
            self.assertEqual(claimed["place_kwargs"]["client_order_id"],
                             rec["place_kwargs"]["client_order_id"])

        with self.subTest(msg="collapses_ladder_any_distance"):
            self._reset_env()
            oq.enqueue(
                "BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0,
                failure_reason="profit_guard", now=1000.0)
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=70000.0, now=1001.0)
            self.assertEqual(len(oq.load_all()), 1)
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=62000.0, now=1002.0)
            self.assertEqual(len(oq.load_all()), 2)

        with self.subTest(msg="preserves_attempts_on_refresh"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0, now=1000.0, attempts=3)
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63100.0, now=1001.0, attempts=0)
            self.assertEqual(oq.load_all()[0]["attempts"], 3)

        with self.subTest(msg="dedup_off_appends_always"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0,
                       now=1000.0, failure_reason="network")
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0,
                       now=1001.0, failure_reason="network")
            self.assertEqual(len(oq.load_all()), 2)

    def test_trend_consolidation_rules(self):
        self._reset_env()
        with self.subTest(msg="stream_keeps_only_latest_desired_exposure"):
            oq.RETRY_DEDUP = False
            first = oq.enqueue(
                "TAOUSDC", "SELL", 5.0, {}, requested_price=250.0, now=1000.0,
                failure_reason="trend_deferred", provider_name="binance",
                kind="trend_confirmed_down")
            latest = oq.enqueue(
                "TAOUSDC", "SELL", 2.0, {}, requested_price=240.0, now=1001.0,
                failure_reason="trend_deferred", provider_name="Binance",
                kind="trend_confirmed_down")
            records = oq.load_all()
            self.assertEqual(len(records), 1)
            self.assertNotEqual(first, latest)
            self.assertEqual(records[0]["id"], latest)
            self.assertEqual(records[0]["qty"], 2.0)
            self.assertEqual(records[0]["requested_price"], 240.0)

        with self.subTest(msg="profit_guard_stream_consolidates_when_dedup_off"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            first = oq.enqueue(
                "HYPEUSD", "BUY", 2.4, {}, requested_price=81.0, now=1000.0,
                failure_reason="profit_guard", provider_name="kraken")
            latest = oq.enqueue(
                "HYPEUSD", "BUY", 2.5, {}, requested_price=82.0, now=1001.0,
                failure_reason="profit_guard", provider_name="Kraken")
            records = oq.load_all()
            self.assertEqual(len(records), 1)
            self.assertNotEqual(first, latest)
            self.assertEqual(records[0]["id"], latest)
            self.assertEqual(records[0]["qty"], 2.5)

        with self.subTest(msg="preserves_independent_signal_kinds"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            for kind in ("kalman_primary_down", "trend_confirmed_down"):
                oq.enqueue(
                    "TAOUSDC", "SELL", 1.0, {}, requested_price=240.0,
                    now=1000.0, failure_reason="trend_deferred", kind=kind)
            self.assertEqual(len(oq.load_all()), 2)

        with self.subTest(msg="never_replaces_accepted_record"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            accepted_id = oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=240.0, now=1000.0,
                failure_reason="trend_deferred")
            records = oq.load_all()
            records[0]["lifecycle"] = "accepted"
            records[0]["order_id"] = "venue-1"
            oq.rewrite(records)
            latest = oq.enqueue(
                "TAOUSDC", "SELL", 2.0, {}, requested_price=241.0, now=1001.0,
                failure_reason="trend_deferred")
            self.assertEqual(
                {record["id"] for record in oq.load_all()}, {accepted_id, latest})

        with self.subTest(msg="enqueue_trend_consolidation_never_removes_claimed_record"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            first = oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=240.0,
                now=1000.0, failure_reason="trend_deferred",
                kind="trend_confirmed_down")
            claimed = oq.claim([first], now=1001.0)[0]
            before = oq.get(first)

            latest = oq.enqueue(
                "TAOUSDC", "SELL", 2.0, {}, requested_price=241.0,
                now=1002.0, failure_reason="trend_deferred",
                kind="trend_confirmed_down")

            records = {record["id"]: record for record in oq.load_all()}
            self.assertEqual(set(records), {first, latest})
            self.assertEqual(records[first], before)
            self.assertEqual(
                records[first]["claim_token"], claimed["claim_token"])

        with self.subTest(msg="startup_consolidation_compacts_existing_deferred_history"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            ids = []
            for now, qty in ((1000.0, 5.0), (1001.0, 3.0), (1002.0, 2.0)):
                ids.append(oq.enqueue(
                    "TAOUSDC", "SELL", qty, {}, requested_price=240.0, now=now,
                    failure_reason="network", kind="trend_confirmed_down"))
            records = oq.load_all()
            for record in records:
                record["last_failure_reason"] = "trend_deferred"
                record["safe_to_discard"] = True
            records.append({
                "id": "cooldown", "symbol": "TAOUSDC", "side": "SELL", "qty": 1.0,
                "created_ts": 1003.0, "lifecycle": "submit_pending", "revision": 0,
                "last_failure_reason": "cooldown",
            })
            oq.rewrite(records)

            removed = oq.consolidate_deferred_streams()

            remaining = oq.load_all()
            self.assertEqual(len(removed), 2)
            self.assertEqual({record["id"] for record in remaining}, {ids[-1], "cooldown"})
            self.assertEqual(next(r for r in remaining if r["id"] == ids[-1])["qty"], 2.0)

        with self.subTest(msg="startup_consolidation_never_removes_claimed_record"):
            self._reset_env()
            oq.RETRY_DEDUP = False
            ids = [
                oq.enqueue(
                    "TAOUSDC", "SELL", qty, {}, requested_price=240.0,
                    now=created, failure_reason="network",
                    kind="trend_confirmed_down")
                for created, qty in (
                    (1000.0, 5.0), (1001.0, 3.0), (1002.0, 2.0))
            ]
            records = oq.load_all()
            for record in records:
                record["last_failure_reason"] = "trend_deferred"
                record["safe_to_discard"] = True
            oq.rewrite(records)
            claimed = oq.claim([ids[0]], now=1100.0)[0]
            claimed_before = oq.get(ids[0])

            removed = oq.consolidate_deferred_streams()

            remaining = {record["id"]: record for record in oq.load_all()}
            self.assertEqual({record["id"] for record in removed}, {ids[1]})
            self.assertEqual(set(remaining), {ids[0], ids[2]})
            self.assertEqual(remaining[ids[0]], claimed_before)
            self.assertEqual(
                remaining[ids[0]]["claim_token"], claimed["claim_token"])

    def test_terminal_remainder_lifecycle_and_invariants(self):
        self._reset_env()
        with self.subTest(msg="capacity_never_evicts_terminal_remainder"):
            oq.RETRY_MAX_QUEUE = 1
            oq.RETRY_DEDUP = False
            record_id, remainder = self._terminal_remainder()
            remainder["safe_to_discard"] = True
            oq.rewrite([remainder])
            before = oq.get(record_id)

            refused = oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=200.0,
                now=1500.0, failure_reason="trend_deferred")

            self.assertIsNone(refused)
            self.assertEqual(oq.get(record_id), before)

        with self.subTest(msg="dedup_never_overwrites_terminal_remainder"):
            self._reset_env()
            record_id, remainder = self._terminal_remainder()

            new_id = oq.enqueue(
                "BTCUSDC", "BUY", 3.0, {}, requested_price=99.0,
                now=1500.0, failure_reason="account_cache_not_fresh",
                kind="terminal-remainder")

            self.assertNotEqual(new_id, record_id)
            self.assertEqual(oq.get(record_id), remainder)
            self.assertEqual(len(oq.load_all()), 2)

        with self.subTest(msg="inline_consolidation_never_deletes_terminal_remainder"):
            self._reset_env()
            record_id, _ = self._terminal_remainder()
            claimed = oq.claim([record_id], now=1700.0)[0]
            self.assertTrue(oq.complete_claim(
                claimed, "deferred", now=1701.0,
                failure_reason="trend_deferred", submission_state="refused"))
            protected = oq.get(record_id)
            self.assertNotIn("safe_to_discard", protected)

            new_id = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=98.0,
                now=1702.0, failure_reason="trend_deferred",
                kind="terminal-remainder")

            records = {record["id"]: record for record in oq.load_all()}
            self.assertEqual(set(records), {record_id, new_id})
            self.assertEqual(records[record_id], protected)

        with self.subTest(msg="startup_consolidation_never_deletes_terminal_remainder"):
            self._reset_env()
            record_id, _ = self._terminal_remainder()
            claimed = oq.claim([record_id], now=1700.0)[0]
            self.assertTrue(oq.complete_claim(
                claimed, "deferred", now=1701.0,
                failure_reason="trend_deferred", submission_state="refused"))
            protected = oq.get(record_id)
            protected["safe_to_discard"] = True
            oq.rewrite([protected])
            new_id = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=98.0,
                now=1702.0, failure_reason="trend_deferred",
                kind="terminal-remainder")
            before = {record["id"]: record for record in oq.load_all()}

            removed = oq.consolidate_deferred_streams()

            after = {record["id"]: record for record in oq.load_all()}
            self.assertEqual(removed, [])
            self.assertEqual(set(after), {record_id, new_id})
            self.assertEqual(after[record_id], before[record_id])

        with self.subTest(msg="terminal_remainder_gets_fresh_ttl_and_attempt_budget"):
            self._reset_env()
            oq.RETRY_MAX_ATTEMPTS = 1
            terminal_at = 1000.0 + oq.RETRY_TTL_SEC + 10.0
            record_id, remainder = self._terminal_remainder(
                terminal_at=terminal_at)

            self.assertEqual(remainder["attempts"], 0)
            self.assertEqual(remainder["ttl_started_ts"], terminal_at)
            self.assertEqual(remainder["last_attempt_ts"], terminal_at)
            self.assertNotIn("safe_to_discard", remainder)
            retry_at = terminal_at + oq.RETRY_INTERVAL_SEC
            self.assertFalse(oq.is_expired(remainder, now=retry_at))
            self.assertTrue(oq.is_due(remainder, now=retry_at))
            self.assertEqual(oq.get(record_id), remainder)

    def test_queue_capacity_and_eviction_policies(self):
        self._reset_env()
        with self.subTest(msg="full_queue_replaces_oldest_pending_record"):
            oq.RETRY_MAX_QUEUE = 2
            oq.RETRY_DEDUP = False
            oldest = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0,
                                now=1000.0, failure_reason="trend_deferred", kind="a")
            retained = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=2.0,
                                  now=1001.0, failure_reason="trend_deferred", kind="b")
            newest = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=3.0,
                                now=1002.0, failure_reason="trend_deferred", kind="c")
            self.assertIsNotNone(newest)
            ids = {record["id"] for record in oq.load_all()}
            self.assertNotIn(oldest, ids)
            self.assertIn(retained, ids)
            self.assertIn(newest, ids)

        with self.subTest(msg="full_queue_never_evicts_accepted_orders"):
            self._reset_env()
            oq.RETRY_MAX_QUEUE = 1
            oq.RETRY_DEDUP = False
            accepted_id = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=1.0, now=1000.0)
            records = oq.load_all()
            records[0]["lifecycle"] = "accepted"
            records[0]["order_id"] = "venue-1"
            oq.rewrite(records)
            self.assertIsNone(oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=2.0, now=1001.0))
            self.assertEqual([record["id"] for record in oq.load_all()], [accepted_id])

        with self.subTest(msg="full_queue_evicts_safe_deferred_before_ambiguous_record"):
            self._reset_env()
            oq.RETRY_MAX_QUEUE = 2
            oq.RETRY_DEDUP = False
            ambiguous = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=1.0,
                now=1000.0, failure_reason="submit_ambiguous")
            safe = oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=2.0,
                now=1001.0, failure_reason="trend_deferred", kind="safe")

            newest = oq.enqueue(
                "ETHUSDC", "BUY", 1.0, {}, requested_price=3.0,
                now=1002.0, failure_reason="cooldown")

            self.assertIsNotNone(newest)
            ids = {record["id"] for record in oq.load_all()}
            self.assertEqual(ids, {ambiguous, newest})
            self.assertNotIn(safe, ids)

        with self.subTest(msg="full_queue_refuses_when_all_records_are_ambiguous"):
            self._reset_env()
            oq.RETRY_MAX_QUEUE = 1
            oq.RETRY_DEDUP = False
            ambiguous = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=1.0,
                now=1000.0, failure_reason="response_without_order_id")
            before = oq.load_all()

            refused = oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=2.0,
                now=1001.0, failure_reason="trend_deferred")

            self.assertIsNone(refused)
            self.assertEqual(oq.load_all(), before)
            self.assertEqual(before[0]["id"], ambiguous)

        with self.subTest(msg="full_queue_never_evicts_claimed_deferred_record"):
            self._reset_env()
            oq.RETRY_MAX_QUEUE = 1
            oq.RETRY_DEDUP = False
            record_id = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=1.0,
                now=1000.0, failure_reason="trend_deferred")
            claimed = oq.claim([record_id], now=1001.0)[0]
            before = oq.get(record_id)

            refused = oq.enqueue(
                "TAOUSDC", "SELL", 1.0, {}, requested_price=2.0,
                now=1002.0, failure_reason="trend_deferred")

            self.assertIsNone(refused)
            self.assertEqual(oq.get(record_id), before)
            self.assertEqual(before["claim_token"], claimed["claim_token"])

        with self.subTest(msg="full_queue_never_evicts_awaiting_cancel_record"):
            self._reset_env()
            oq.RETRY_MAX_QUEUE = 1
            oq.RETRY_DEDUP = False
            waiting = oq.enqueue(
                "BTCUSDC", "SELL", 1.0, {}, requested_price=1.0,
                now=1000.0, lifecycle="awaiting_cancel",
                replaces_order_id="old-queue", replaces_original_qty=1.0)
            before = oq.load_all()

            refused = oq.enqueue(
                "TAOUSDC", "BUY", 1.0, {}, requested_price=2.0,
                now=1001.0, failure_reason="trend_deferred")

            self.assertIsNone(refused)
            self.assertEqual(oq.load_all(), before)
            self.assertEqual(before[0]["id"], waiting)

    def test_price_gate_behaviors(self):
        with self.subTest(msg="sell"):
            rec = {"side": "SELL", "requested_price": 100.0}
            self.assertTrue(oq.price_gate_ok(rec, 100.0))         # Equal -> ok.
            self.assertTrue(oq.price_gate_ok(rec, 101.0))         # Higher -> ok (you sell better).
            self.assertTrue(oq.price_gate_ok(rec, 99.9))          # Within the 0.2% tolerance.
            self.assertFalse(oq.price_gate_ok(rec, 95.0))         # Far below -> it waits.
            self.assertFalse(oq.price_gate_ok(rec, None))         # no price -> we do not decide

        with self.subTest(msg="buy"):
            rec = {"side": "BUY", "requested_price": 100.0}
            self.assertTrue(oq.price_gate_ok(rec, 100.0))
            self.assertTrue(oq.price_gate_ok(rec, 99.0))          # Lower -> ok (you buy cheaper).
            self.assertFalse(oq.price_gate_ok(rec, 105.0))        # Far above -> it waits.

        with self.subTest(msg="no intent skips"):
            # no captured requested_price (old/abnormal entry) -> conservative, do NOT retry blind
            self.assertFalse(oq.price_gate_ok({"side": "SELL"}, 50.0))
            self.assertFalse(oq.price_gate_ok({"side": "BUY", "requested_price": 0}, 50.0))
            self.assertFalse(oq.price_gate_ok({"side": "HOLD", "requested_price": 50}, 50.0))
            self.assertFalse(oq.price_gate_ok(
                {"side": "BUY", "requested_price": 50}, float("nan")))

    def test_record_resolution_and_accepted_state(self):
        self._reset_env()
        with self.subTest(msg="exact_record_resolution_does_not_remove_newer_revision"):
            rid = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                             failure_reason="profit_guard", now=1000.0)
            first = oq.get(rid)
            first_cid = first["place_kwargs"]["client_order_id"]
            oq.enqueue("BTCUSDC", "BUY", 2.0, {}, requested_price=99.0,
                       now=1001.0)

            self.assertNotEqual(oq.get(rid)["place_kwargs"]["client_order_id"], first_cid)
            self.assertEqual(oq.get(rid)["qty"], 2.0)

        with self.subTest(msg="complete_claim_accepted_keeps_exact_record_until_terminal"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                provider_name="Binance", intent_id="strategy-cycle-entry",
                kind="ENTRY", now=1000.0)
            claimed = oq.claim([rid], now=1050.0)[0]

            self.assertTrue(oq.complete_claim(
                claimed, "accepted", order={"orderId": 123, "status": "NEW"},
                now=1100.0))

            rec = oq.get(rid)
            self.assertEqual(rec["lifecycle"], "accepted")
            self.assertEqual(rec["order_id"], "123")
            self.assertEqual(rec["provider_name"], "Binance")
            self.assertEqual(rec["intent_id"], "strategy-cycle-entry")
            self.assertEqual(rec["kind"], "ENTRY")
            self.assertFalse(oq.is_expired(rec, now=1000.0 + 100 * 86400))
            self.assertFalse(oq.is_due(rec, now=1399.0))
            self.assertTrue(oq.is_due(rec, now=1400.0))

        with self.subTest(msg="order_id_parser_handles_provider_shapes"):
            self.assertEqual(oq.extract_order_id({"orderId": 1}), "1")
            self.assertEqual(oq.extract_order_id({"id": "two"}), "two")
            self.assertEqual(oq.extract_order_id({"txid": ["three"]}), "three")
            self.assertIsNone(oq.extract_order_id({"status": "NEW"}))

        with self.subTest(msg="success_does_not_delete_newer_refresh"):
            self._reset_env()
            rid = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
            claimed = oq.claim([rid], now=1400.0)[0]
            oq.enqueue("BTCUSDC", "BUY", 2.0, {}, requested_price=90.0, now=1401.0)
            oq.complete_claim(claimed, "success", now=1402.0)
            rec = oq.load_all()[0]
            self.assertEqual(rec["qty"], 2.0)
            self.assertEqual(rec["requested_price"], 90.0)
            self.assertNotIn("claim_token", rec)

        with self.subTest(msg="resolve_removes_only_matching_symbol_and_side"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0, now=1000.0)
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=2.0, now=1001.0)
            oq.enqueue("TAOUSDC", "BUY", 1.0, {}, requested_price=3.0, now=1002.0)

            self.assertEqual(oq.resolve("BTCUSDC", "buy"), 1)
            self.assertEqual(oq.resolve("BTCUSDC", "BUY"), 0)
            remaining = {(r["symbol"], r["side"]) for r in oq.load_all()}
            self.assertEqual(remaining, {("BTCUSDC", "SELL"), ("TAOUSDC", "BUY")})

        with self.subTest(msg="reenqueue_preserves_created_ts"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0,
                       now=5000.0, created_ts=1000.0, attempts=2)
            r = oq.load_all()[0]
            self.assertEqual(r["created_ts"], 1000.0)
            self.assertEqual(r["attempts"], 2)

        with self.subTest(msg="dedup_does_not_clobber_distinct_venues"):
            self._reset_env()
            id_bn = oq.enqueue(
                "HYPEUSDC", "BUY", 1.0, {}, requested_price=30.0,
                now=1000.0, failure_reason="trend_deferred", provider_name="binance",
            )
            id_hl = oq.enqueue(
                "HYPEUSDC", "BUY", 2.5, {}, requested_price=30.5,
                now=1001.0, failure_reason="trend_deferred", provider_name="hyperliquid",
            )
            self.assertNotEqual(id_bn, id_hl)
            queued = oq.load_all()
            self.assertEqual(len(queued), 2)
            by_venue = {r["provider_name"]: r["qty"] for r in queued}
            self.assertEqual(by_venue, {"binance": 1.0, "hyperliquid": 2.5})

        with self.subTest(msg="dedup_preserves_safe_to_discard_for_consolidated_deferrals_across_3plus_enqueues"):
            self._reset_env()
            id1 = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                now=1000.0, failure_reason="insufficient_funds", provider_name="binance",
            )
            id2 = oq.enqueue(
                "BTCUSDC", "BUY", 1.5, {}, requested_price=101.0,
                now=1010.0, failure_reason="insufficient_funds", provider_name="binance",
            )
            id3 = oq.enqueue(
                "BTCUSDC", "BUY", 2.0, {}, requested_price=102.0,
                now=1020.0, failure_reason="insufficient_funds", provider_name="binance",
            )
            self.assertEqual(id1, id2)
            self.assertEqual(id2, id3)
            queued = oq.load_all()
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["qty"], 2.0)
            self.assertTrue(queued[0].get("safe_to_discard"))

    def test_claim_leasing_and_recovery_semantics(self):
        self._reset_env()
        with self.subTest(msg="claim_leases_and_returns_without_removing"):
            a = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0, now=1000.0)
            oq.enqueue("TAOUSDC", "SELL", 2.0, {}, requested_price=2.0, now=1001.0)
            claimed = oq.claim([a])
            self.assertEqual(len(claimed), 1)
            self.assertEqual(claimed[0]["id"], a)
            rem = oq.load_all()
            self.assertEqual(len(rem), 2)
            leased = next(r for r in rem if r["id"] == a)
            self.assertTrue(leased["claim_token"])
            self.assertGreater(leased["claim_until"], time.time())

        with self.subTest(msg="claim_failure_releases_and_increments"):
            self._reset_env()
            rid = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0, now=1000.0)
            claimed = oq.claim([rid], now=1400.0)[0]
            self.assertTrue(oq.complete_claim(claimed, "failure", now=1401.0))
            rec = oq.load_all()[0]
            self.assertEqual(rec["attempts"], 1)
            self.assertEqual(rec["last_attempt_ts"], 1401.0)
            self.assertNotIn("claim_token", rec)

        with self.subTest(msg="claim_survives_crash_and_becomes_due_after_lease"):
            self._reset_env()
            rid = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0, now=1000.0)
            oq.claim([rid], now=1400.0, lease_sec=120.0)
            rec = oq.load_all()[0]
            self.assertFalse(oq.is_due(rec, now=1519.0))
            self.assertTrue(oq.is_due(rec, now=1521.0))

        with self.subTest(msg="claim_migrates_legacy_record_to_deterministic_client_id"):
            self._reset_env()
            oq.rewrite([{
                "id": "a" * 32, "symbol": "BTCUSDC", "side": "SELL", "qty": 1.0,
                "place_kwargs": {}, "requested_price": 100.0, "created_ts": 1000.0,
                "attempts": 0, "last_attempt_ts": 0.0,
            }])
            rec = oq.claim(["a" * 32], now=1400.0)[0]
            self.assertEqual(rec["place_kwargs"]["client_order_id"],
                             "OR_" + "a" * 24 + "_0")
            self.assertEqual(oq.load_all()[0]["place_kwargs"]["client_order_id"],
                             rec["place_kwargs"]["client_order_id"])

        with self.subTest(msg="claim_empty_noop"):
            self._reset_env()
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=1.0, now=1000.0)
            self.assertEqual(oq.claim([]), [])
            self.assertEqual(len(oq.load_all()), 1)

    def _accepted_claim(self, *, qty=1.0, now=1000.0):
        rid = oq.enqueue(
            "BTCUSDC", "BUY", qty, {}, requested_price=100.0, now=now)
        oq.complete_claim(
            oq.claim([rid], now=now + 100.0)[0],
            "accepted",
            order={"orderId": 77, "status": "NEW"},
            now=now + 100.0)
        return oq.claim([rid], now=now + 400.0)[0]

    def test_advance_claimed_status_behaviors(self):
        with self.subTest(msg="observes_partial_without_resubmit_state"):
            oq.rewrite([])
            claimed = self._accepted_claim(qty=2.0)
            status = OrderStatus(
                "open", 0.5, 49.0, 0.01, venue_status="PARTIALLY_FILLED")
            transition = oq.advance_claimed_status(claimed, status, now=1401.0)
            self.assertEqual(transition.action, "observed")
            self.assertTrue(transition.status_changed)
            record = oq.load_all()[0]
            self.assertEqual(record["lifecycle"], "accepted")
            self.assertEqual(record["filled_qty"], 0.5)
            self.assertNotIn("claim_token", record)

        with self.subTest(msg="fill_removes_active_record"):
            oq.rewrite([])
            claimed = self._accepted_claim()
            transition = oq.advance_claimed_status(
                claimed,
                OrderStatus("closed", 1.0, 100.0, 0.02, venue_status="FILLED"),
                now=1401.0,
            )
            self.assertEqual(transition.action, "filled")
            self.assertEqual(oq.load_all(), [])

        with self.subTest(msg="rejected_revises_only_remainder"):
            oq.rewrite([])
            claimed = self._accepted_claim(qty=2.0)
            transition = oq.advance_claimed_status(
                claimed,
                OrderStatus(
                    "canceled", 0.5, 49.0, 0.01, venue_status="REJECTED"),
                now=1401.0,
            )
            self.assertEqual(transition.action, "retry_terminal")
            self.assertEqual(transition.remaining_qty, 1.5)
            record = oq.load_all()[0]
            self.assertEqual(record["lifecycle"], "submit_pending")
            self.assertEqual(record["qty"], 1.5)
            self.assertEqual(record["revision"], 1)

        with self.subTest(msg="cancel_is_terminal_without_revision"):
            oq.rewrite([])
            claimed = self._accepted_claim(qty=2.0)
            transition = oq.advance_claimed_status(
                claimed,
                OrderStatus(
                    "canceled", 0.5, 49.0, 0.01, venue_status="CANCELED"),
                now=1401.0,
            )
            self.assertEqual(transition.action, "terminal")
            self.assertEqual(transition.remaining_qty, 1.5)
            self.assertEqual(oq.load_all(), [])

    def test_due_expiry_and_deferral_lifecycle(self):
        self._reset_env()
        with self.subTest(msg="is_due"):
            rec = {"last_attempt_ts": 1000.0}
            self.assertFalse(oq.is_due(rec, now=1000.0 + 100))   # < interval 300
            self.assertTrue(oq.is_due(rec, now=1000.0 + 300))    # >= interval

        with self.subTest(msg="is_expired_ttl"):
            rec = {
                "created_ts": 1000.0,
                "attempts": 0,
                "submission_state": "refused",
            }
            self.assertFalse(oq.is_expired(rec, now=1000.0 + 86400 - 1))
            self.assertTrue(oq.is_expired(rec, now=1000.0 + 86400 + 1))

        with self.subTest(msg="possibly_submitted_records_never_expire_automatically"):
            far_future = 1000.0 + 10 * oq.RETRY_TTL_SEC
            for label, state, reason in (
                    ("producer-owned", "producer_claimed", "submit_pending"),
                    ("unknown-response", "unknown", "provider_timeout"),
                    ("legacy-ambiguous", None, "response_without_order_id")):
                with self.subTest(label=label):
                    record = {
                        "created_ts": 1000.0,
                        "ttl_started_ts": 1000.0,
                        "attempts": 999,
                        "last_failure_reason": reason,
                    }
                    if state is not None:
                        record["submission_state"] = state
                    self.assertFalse(oq.is_expired(record, now=far_future))

        with self.subTest(msg="is_expired_max_attempts"):
            self._reset_env()
            oq.RETRY_MAX_ATTEMPTS = 3
            base = {"created_ts": 1e9, "submission_state": "refused"}
            self.assertFalse(oq.is_expired({**base, "attempts": 2}, now=1e9))
            self.assertTrue(oq.is_expired({**base, "attempts": 3}, now=1e9))

        with self.subTest(msg="trend_deferred_record_never_consumes_ttl_or_attempts"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                failure_reason="trend_deferred", now=1000.0)
            rec = oq.load_all()[0]
            self.assertFalse(oq.is_expired(rec, now=1000.0 + 10 * 86400))

            claimed = oq.claim([rid], now=1000.0 + 10 * 86400)[0]
            oq.complete_claim(
                claimed, "deferred", now=1000.0 + 10 * 86400,
                failure_reason="trend_deferred")
            rec = oq.load_all()[0]
            self.assertEqual(rec["attempts"], 0)
            self.assertEqual(rec["last_failure_reason"], "trend_deferred")
            self.assertFalse(oq.is_expired(rec, now=1000.0 + 20 * 86400))

        with self.subTest(msg="ttl_starts_fresh_after_trend_stops_deferring"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                failure_reason="trend_deferred", now=1000.0)
            transition = 1000.0 + 10 * 86400
            claimed = oq.claim([rid], now=transition)[0]
            oq.complete_claim(
                claimed, "failure", now=transition, failure_reason="cooldown")
            rec = oq.load_all()[0]
            self.assertEqual(rec["ttl_started_ts"], transition)
            self.assertFalse(oq.is_expired(rec, now=transition + 86400 - 1))
            self.assertTrue(oq.is_expired(rec, now=transition + 86400 + 1))

        with self.subTest(msg="account_cache_deferral_preserves_reason_attempts_and_ttl"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                failure_reason="account_cache_not_fresh", now=1000.0)
            original = oq.load_all()[0]
            original_ttl_start = original["ttl_started_ts"]

            deferred_at = 1000.0 + 10 * 86400
            claimed = oq.claim([rid], now=deferred_at)[0]
            oq.complete_claim(
                claimed, "deferred", now=deferred_at,
                failure_reason="account_cache_not_fresh")

            rec = oq.load_all()[0]
            self.assertEqual(rec["last_failure_reason"], "account_cache_not_fresh")
            self.assertEqual(rec["attempts"], 0)
            self.assertEqual(rec["ttl_started_ts"], original_ttl_start)
            self.assertFalse(oq.is_expired(rec, now=deferred_at + 20 * 86400))

        with self.subTest(msg="ttl_starts_fresh_after_account_cache_deferral_ends"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                failure_reason="account_cache_not_fresh", now=1000.0)
            transition = 1000.0 + 10 * 86400
            claimed = oq.claim([rid], now=transition)[0]
            oq.complete_claim(
                claimed, "failure", now=transition,
                failure_reason="provider_timeout")

            rec = oq.load_all()[0]
            self.assertEqual(rec["last_failure_reason"], "provider_timeout")
            self.assertEqual(rec["attempts"], 1)
            self.assertEqual(rec["ttl_started_ts"], transition)
            self.assertFalse(oq.is_expired(rec, now=transition + 86400 - 1))
            self.assertTrue(oq.is_expired(rec, now=transition + 86400 + 1))


    def test_import_rejects_unsafe_retry_configuration(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        valid_config = {
            "RETRY_ENABLED": "true",
            "RETRY_INTERVAL_SEC": "300",
            "RETRY_TTL_SEC": "86400",
            "RETRY_MAX_ATTEMPTS": "0",
            "RETRY_PRICE_TOL": "0.002",
            "RETRY_DEDUP": "false",
            "RETRY_MAX_QUEUE": "500",
            "RETRY_CLAIM_LEASE_SEC": "120",
            "RETRY_NOT_FOUND_MAX_AGE_SEC": "2592000",
        }
        invalid_values = (
            ("RETRY_INTERVAL_SEC", "0"),
            ("RETRY_INTERVAL_SEC", "nan"),
            ("RETRY_TTL_SEC", "-1"),
            ("RETRY_TTL_SEC", "inf"),
            ("RETRY_CLAIM_LEASE_SEC", "0"),
            ("RETRY_CLAIM_LEASE_SEC", "-inf"),
            ("RETRY_NOT_FOUND_MAX_AGE_SEC", "0"),
            ("RETRY_NOT_FOUND_MAX_AGE_SEC", "nan"),
            ("RETRY_NOT_FOUND_MAX_AGE_SEC", "inf"),
            ("RETRY_PRICE_TOL", "1"),
            ("RETRY_PRICE_TOL", "nan"),
            ("RETRY_MAX_ATTEMPTS", "-1"),
            ("RETRY_MAX_QUEUE", "-1"),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                env = os.environ.copy()
                env.update(valid_config)
                env[key] = value
                completed = subprocess.run(
                    [sys.executable, "-c", "import order_retry"],
                    cwd=root, env=env, capture_output=True, text=True,
                    check=False)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(key, completed.stdout + completed.stderr)

    def test_corrupt_middle_line_fails_closed_without_rewrite(self):
        record_id = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0)
        with open(oq.QUEUE_FILE, "rb") as queue_file:
            valid_line = queue_file.read()
        with open(oq.QUEUE_FILE, "ab") as queue_file:
            queue_file.write(b"{corrupt json\n")
            queue_file.write(valid_line)
        with open(oq.QUEUE_FILE, "rb") as queue_file:
            committed_bytes = queue_file.read()

        with self.assertRaises(oq.RetryQueueCorruptionError):
            oq.load_all()
        with self.assertRaises(oq.RetryQueueCorruptionError):
            oq.claim([record_id], now=2000.0)

        with open(oq.QUEUE_FILE, "rb") as queue_file:
            self.assertEqual(queue_file.read(), committed_bytes)

    def test_enqueue_rejects_invalid_explicit_timestamps(self):
        for bad_ts in (0, -10.0, float("nan"), float("inf"), "invalid", True):
            with self.subTest(now=bad_ts):
                self.assertIsNone(oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=bad_ts))
            with self.subTest(created_ts=bad_ts):
                self.assertIsNone(
                    oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0, created_ts=bad_ts)
                )
        self.assertEqual(oq.load_all(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_terminal_filter_refusal_classification_across_non_binance_providers_iter7_s1():
    """[ITER7 #1] Terminal filter refusal classification across non-Binance providers."""
    # 1. Terminal filter refusal classification across non-Binance providers
    assert oq.is_terminal_filter_refusal("filter_lot_size_min:0.001<0.05") is True
    assert oq.is_terminal_filter_refusal("qty_zero_or_negative") is True
    assert oq.is_terminal_filter_refusal("binance_filter_refused:MIN_NOTIONAL") is True
    assert oq.is_terminal_filter_refusal("below_min_notional") is True
    assert oq.is_terminal_filter_refusal("trend_deferred") is False

def test_order_retry_update_claimed_qty_persists_applied_scale_into_place_kwargs_on_disk_iter10_s2(tmp_path, monkeypatch):
    """[ITER10 #2] order_retry.update_claimed_qty persists applied_scale into place_kwargs on disk."""
    # 2. order_retry.update_claimed_qty persists applied_scale into place_kwargs on disk
    q_file = tmp_path / "pending_orders.json"
    l_file = tmp_path / "pending_orders.lock"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "LOCK_FILE", str(l_file))
    claim = oq.enqueue_claimed(
        "BTCUSDC", "BUY", 1.0, {"smart": False},
        requested_price=100.0, ref_price=100.0,
        failure_reason="submit_pending", provider_name="binance",
    )
    assert claim is not None
    updated = oq.update_claimed_qty(claim, 0.5, applied_scale=0.5)
    assert updated is not None
    assert updated["qty"] == pytest.approx(0.5)
    assert updated["place_kwargs"].get("applied_scale") == pytest.approx(0.5)

def test_instrument_place_persists_medium_trend_action_medium_trend_scale_from_self_params_iter11_s2(monkeypatch):
    """[ITER11 #2] Instrument.place persists medium_trend_action & medium_trend_scale from self.params into retry_kwargs."""
    # 2. Instrument.place persists medium_trend_action & medium_trend_scale from self.params into retry_kwargs
    enqueued_kwargs = []

    monkeypatch.setattr(
        oq,
        "enqueue",
        lambda _s, _side, _q, pkw, *_a, **_k: enqueued_kwargs.append(dict(pkw)) or {"id": "r1"},
    )
    inst_obj = inst_mod.Instrument.__new__(inst_mod.Instrument)
    inst_obj.name = "BTCUSDC"
    inst_obj.symbol = "BTCUSDC"
    inst_obj.provider_name = "binance"
    inst_obj.base = "BTC"
    inst_obj.quote = "USDC"
    inst_obj.enabled = True
    inst_obj._provider = DummySpotExecutionProvider(name="binance", free_bal=1000.0, wait_short_trend=True)
    inst_obj._api = SimpleNamespace(
        market_regime_context=lambda *_a, **_k: SimpleNamespace(resolved_trend="flat")
    )
    inst_obj.params = {"medium_trend_action": "scale", "medium_trend_scale": 0.4}
    assert inst_obj.place("BUY", 100.0, 1.0, smart=False) is None
    assert enqueued_kwargs
    assert enqueued_kwargs[-1].get("medium_trend_action") == "scale"
    assert enqueued_kwargs[-1].get("medium_trend_scale") == pytest.approx(0.4)

def test_order_retry_enqueue_serializes_set_frozenset_in_place_kwargs_cleanly_iter23_s5(tmp_path, monkeypatch):
    """[ITER23 #5] order_retry.enqueue serializes set/frozenset in place_kwargs cleanly."""
    # 5. order_retry.enqueue serializes set/frozenset in place_kwargs cleanly
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "LOCK_FILE", str(l_file))
    rec_id = oq.enqueue(
        "BTCUSDC",
        "BUY",
        0.1,
        place_kwargs={"bypass_guards": {"g_prof", "gllm1"}},
        requested_price=85000.0,
        ref_price=85000.0,
        failure_reason="trend_deferred",
    )
    assert rec_id is not None
    loaded_q = oq.load_all()
    assert len(loaded_q) == 1
    assert loaded_q[0]["place_kwargs"]["bypass_guards"] == ["g_prof", "gllm1"]

def test_order_lifecycle__lifecycle_finite_and_trackedorderlifecycle_reject_boolean_values_iter36_s2():
    """[ITER36 #2] order_lifecycle._lifecycle_finite and TrackedOrderLifecycle reject boolean values."""
    # 2. order_lifecycle._lifecycle_finite and TrackedOrderLifecycle reject boolean values
    assert ol._lifecycle_finite(True, positive=True) is None
    assert ol._lifecycle_finite(False, positive=False) is None
    assert ol._lifecycle_finite(1.5, positive=True) == 1.5

    with pytest.raises(ValueError):
        ol.TrackedOrderLifecycle(object(), provider_name="binance", missing_confirmations=True)
    with pytest.raises(ValueError):
        ol.TrackedOrderLifecycle(object(), provider_name="binance", max_age_seconds=True)

    tl = ol.TrackedOrderLifecycle(object(), provider_name="binance", clock=lambda: 1000.0)
    with pytest.raises(ValueError):
        tl.new_intent(
            intent_id="i1", client_order_id="c1", symbol="BTCUSDC",
            side="BUY", requested_qty=True, requested_price=100.0,
        )
    with pytest.raises(ValueError):
        tl.new_intent(
            intent_id="i1", client_order_id="c1", symbol="BTCUSDC",
            side="BUY", requested_qty=1.0, requested_price=True,
        )
    with pytest.raises(ValueError):
        tl.new_intent(
            intent_id="i1", client_order_id="c1", symbol="BTCUSDC",
            side="BUY", requested_qty=1.0, requested_price=100.0, attempt=True,
        )

def test_order_retry_enqueue_update_claimed_qty_resolve_record_ttl_reject_boolean_inputs_iter36_s3():
    """[ITER36 #3] order_retry.enqueue / update_claimed_qty / resolve_record_ttl reject boolean inputs."""
    # 3. order_retry.enqueue / update_claimed_qty / resolve_record_ttl reject boolean inputs
    assert oq.enqueue("BTCUSDC", "BUY", True) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, now=True) is None
    assert oq.update_claimed_qty({"id": "x"}, True) is None
    assert oq.resolve_record_ttl({"ttl_sec": True}) == float(oq.RETRY_TTL_SEC)

def test_order_outcomes_log_sanitizes_cr_lf_and_ignores_nan_bool_iter40_s1(monkeypatch, tmp_path):
    """[ITER40 #1] order_outcomes_log sanitizes CR/LF and ignores NaN/bool strength."""
    # 1. order_outcomes_log sanitizes CR/LF and ignores NaN/bool strength
    assert ol._sanitize_outcome_field("a|b\r\nc") == "a/b  c"
    monkeypatch.setattr(ol, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    ol.log_order_outcome(
        "BTCUSDC", "BUY", 60000.0, 0.1, "accepted", None, "test",
        regime_context=SimpleNamespace(regime="bull", strength=float("nan"), source="snap", fresh=True, fallback_used=False, reason="ok"),
    )
    logs = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "nan" not in content.lower()

def test_order_rollback_extract_order_restore_params_ignores_boolean_kind_and_level_iter40_s2():
    """[ITER40 #2] order_rollback.extract_order_restore_params ignores boolean kind and level."""
    # 2. order_rollback.extract_order_restore_params ignores boolean kind and level
    p_rest = ol.extract_order_restore_params({"side": "SELL", "price": 100.0, "qty": 1.0, "kind": True, "level": False})
    assert p_rest["kind"] == "TP"
    assert p_rest["level"] is None

def test_order_retry_valid_record_rejects_boolean_id_symbol_order_id_replaces_order_id_iter44_s2():
    """[ITER44 #2] order_retry: valid_record rejects boolean id/symbol/order_id/replaces_order_id/replaces_original_qty."""
    # 2. order_retry: valid_record rejects boolean id/symbol/order_id/replaces_order_id/replaces_original_qty
    base_rec = {
        "id": "rec1",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 0.5,
        "created_ts": 1700000000.0,
        "revision": 0,
        "lifecycle": "submit_pending",
    }
    assert oq.valid_record(base_rec) is True
    assert oq.valid_record({**base_rec, "id": True}) is False
    assert oq.valid_record({**base_rec, "symbol": True}) is False
    assert oq.valid_record({**base_rec, "lifecycle": "accepted", "order_id": True}) is False
    assert oq.valid_record({**base_rec, "lifecycle": "awaiting_cancel", "replaces_order_id": True, "replaces_original_qty": 1.0}) is False
    assert oq.valid_record({**base_rec, "lifecycle": "awaiting_cancel", "replaces_order_id": "123", "replaces_original_qty": True}) is False

def test_kalmanstatetracker_active_intents_iter54_s5():
    """[ITER54 #5] KalmanStateTracker & active_intents."""
    from active_intents import _intent_row
    from shadow_signals import KalmanStateTracker
    # 5. KalmanStateTracker & active_intents
    kst = KalmanStateTracker(state_min_interval=float("nan"), now_fn=lambda: 1000.0)
    assert kst.state_min_interval == 1.0
    kst._state["BTCUSDC"] = {"ts": 995.0, "kalman_trend": True}
    assert kst.current_trend("BTCUSDC") == (None, 1e18)
    kst._state["BTCUSDC"] = {"ts": 995.0, "kalman_trend": float("nan")}
    assert kst.current_trend("BTCUSDC") == (None, 1e18)
    kst._state["BTCUSDC"] = {"ts": 995.0, "kalman_trend": 1}
    assert kst.current_trend("BTCUSDC") == (1, 5.0)

    assert _intent_row({"id": "ord-1", "side": "BUY", "qty": True}, owner="test", source="s", location="$", inherited={}) is None
    assert _intent_row({"id": True, "side": "BUY", "qty": 1.0}, owner="test", source="s", location="$", inherited={}) is None

def test_active_intents__intent_row_ignores_boolean_ids_and_non_positive_iter62_s1():
    """[ITER62 #1] active_intents._intent_row ignores boolean IDs and non-positive/NaN quantities."""
    import active_intents
    # 1. active_intents._intent_row ignores boolean IDs and non-positive/NaN quantities
    assert active_intents._intent_row(
        {"order_id": True, "side": "BUY", "qty": 1.0},
        owner="kraken", source="s", location="$", inherited={},
    ) is None
    assert active_intents._intent_row(
        {"order_id": "ord-1", "side": "BUY", "qty": 0.0},
        owner="kraken", source="s", location="$", inherited={},
    ) is None
    valid_row = active_intents._intent_row(
        {"order_id": "ord-1", "side": "BUY", "qty": 2.5, "symbol": True},
        owner="kraken", source="s", location="$", inherited={"symbol": "HYPEUSD"},
    )
    assert valid_row is not None and valid_row["symbol"] == "HYPEUSD"

    exp = active_intents._extract_exposure(
        {"qty": 1.0, "cost": -50.0, "entry_price": -10.0, "symbol": "HYPEUSD"},
        owner="kraken", source="s", inherited={},
    )
    assert exp is not None and exp["cost"] is None and exp["entry_price"] is None

def test_order_rollback_extract_order_restore_params_guards_against_qty_price_overflow_iter71_s3():
    """[ITER71 #3] order_rollback.extract_order_restore_params guards against qty * price overflow."""
    # 3. order_rollback.extract_order_restore_params guards against qty * price overflow
    restored_params = extract_order_restore_params({"side": "BUY", "price": 1e200, "origQty": 1e200})
    assert restored_params["amount"] == 0.0

def test_order_retry_enqueue_rejects_whitespace_only_boolean_symbols_and_iter88_s4(monkeypatch, tmp_path):
    """[ITER88 #4] order_retry.enqueue rejects whitespace-only/boolean symbols and normalizes short-form B/S sides."""
    # 4. order_retry.enqueue rejects whitespace-only/boolean symbols and normalizes short-form B/S sides
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "LOCK_FILE", str(l_file))
    monkeypatch.setattr(oq, "RETRY_ENABLED", True)

    assert oq.enqueue("   ", "BUY", 1.0) is None
    assert oq.enqueue(True, "BUY", 1.0) is None
    assert oq.enqueue("BTCUSDC", True, 1.0) is None

    rec_id = oq.enqueue(" btcusdc ", " b ", 1.0, requested_price=65000.0)
    assert rec_id is not None
    loaded = oq.load_validated()
    assert len(loaded) == 1
    assert loaded[0]["symbol"] == "btcusdc"
    assert loaded[0]["side"] == "BUY"
    assert oq.price_gate_ok({"side": "b", "requested_price": 65000.0}, 64000.0, tol=0.0) is True
    assert oq.resolve(" btcusdc ", " b ") == 1

def test_order_retry_valid_record__same_deferred_stream_claim_discard_and_trackedorderlifecycle_validation_iter97_s2():
    """[ITER97 #2] order_retry.valid_record, _same_deferred_stream, claim/discard, and TrackedOrderLifecycle validation."""
    # 2. order_retry.valid_record, _same_deferred_stream, claim/discard, and TrackedOrderLifecycle validation
    base_rec = {
        "id": "rec1",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 1.0,
        "created_ts": 1000.0,
        "revision": 0,
        "lifecycle": "submit_pending",
        "last_failure_reason": "trend_deferred",
    }
    assert oq.valid_record({**base_rec, "revision": 1.5}) is False
    assert oq.valid_record({**base_rec, "side": " buy "}) is True
    assert oq._same_deferred_stream(
        {**base_rec, "symbol": "btcusdc", "side": " buy "},
        {**base_rec, "symbol": " BTCUSDC ", "side": "BUY"},
    ) is True
    assert oq.claim(None) == []
    assert oq.claim("rec1") == []
    assert oq.discard(None) == []
    assert oq.discard("rec1") == []

    with pytest.raises(ValueError, match="provider_name is required"):
        ol.TrackedOrderLifecycle(object(), provider_name=True)
    with pytest.raises(ValueError, match="missing_confirmations must be positive"):
        ol.TrackedOrderLifecycle(object(), provider_name="binance", missing_confirmations=1.5)

    lc = ol.TrackedOrderLifecycle(object(), provider_name=" binance ", clock=lambda: 1000.0)
    assert lc.provider_name == "binance"
    with pytest.raises(ValueError, match="must not be boolean"):
        lc.new_intent(intent_id=True, client_order_id="c1", symbol="BTCUSDC", side="BUY", requested_qty=1.0)
    with pytest.raises(ValueError, match="attempt must be positive"):
        lc.new_intent(intent_id="i1", client_order_id="c1", symbol="BTCUSDC", side="BUY", requested_qty=1.0, attempt=1.5)
    intent = lc.new_intent(intent_id=" i1 ", client_order_id=" c1 ", symbol=" BTCUSDC ", side=" b ", requested_qty=1.0)
    assert intent["side"] == "BUY"
    assert intent["symbol"] == "BTCUSDC"

def test_order_rollback_restore_cancelled_orders_rejects_non_list_and_bool_reason_iter106_s3():
    """[ITER106 #3] order_rollback: restore_cancelled_orders rejects non-list and bool reason."""
    # 3. order_rollback: restore_cancelled_orders rejects non-list and bool reason
    assert ol.restore_cancelled_orders(True, lambda _p: True, reason=True) == 0

def test_order_outcomes_log_log_order_outcome_sanitizes_boolean_fields_iter107_s3(tmp_path, monkeypatch):
    """[ITER107 #3] order_outcomes_log: log_order_outcome sanitizes boolean fields."""
    # 3. order_outcomes_log: log_order_outcome sanitizes boolean fields
    monkeypatch.setattr(ol, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    ol.log_order_outcome(
        symbol=True,
        side=True,
        price=True,
        qty=True,
        outcome=True,
        refuse_reason=True,
        motivation=True,
        caller=True,
        regime_context=SimpleNamespace(
            regime=True,
            strength=True,
            source=True,
            fresh=True,
            fallback_used=False,
            reason=True,
        ),
    )
    log_files = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8").strip()
    parts = content.split("|")
    assert parts[1:9] == ["", "", "", "", "", "", "", ""]
    assert parts[9:12] == ["", "", ""]
    assert parts[14] == ""

def test_order_retry_enqueue_rejects_bool_replaces_order_id_bool_metadata_fields_iter113_s1(monkeypatch):
    """[ITER113 #1] order_retry.enqueue rejects bool replaces_order_id, bool metadata fields, and fractional attempts."""
    monkeypatch.setattr(oq, "RETRY_ENABLED", True)
    # 1. order_retry.enqueue rejects bool replaces_order_id, bool metadata fields, and fractional attempts
    assert oq.enqueue(
        "BTCUSDC", "BUY", 1.0,
        lifecycle="awaiting_cancel", replaces_order_id=True, replaces_original_qty=1.0,
    ) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, intent_id=True) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, provider_name=True) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, kind=True) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, failure_reason=True) is None
    assert oq.enqueue("BTCUSDC", "BUY", 1.0, attempts=1.5) is None

def test_order_retry_get_and_claim_input_validation_iter113_s2():
    """[ITER113 #2] order_retry.get and claim input validation."""
    # 2. order_retry.get and claim input validation
    assert oq.get(True) is None
    with pytest.raises(ValueError):
        oq.claim(["id1"], lease_sec="bad")

def test_strategyexecutorlifecycleapi_rejects_bool_none_blank_order_id_and_client_order_id_iter113_s3():
    """[ITER113 #3] StrategyExecutorLifecycleApi rejects bool/None/blank order_id and client_order_id before str()."""
    # 3. StrategyExecutorLifecycleApi rejects bool/None/blank order_id and client_order_id before str()
    exec_stub = SimpleNamespace(
        reconciliation_capabilities=lambda: OrderReconciliationCapabilities(
            lookup_by_client_order_id=True, status_by_order_id=True, cancel_by_order_id=True
        ),
        order_by_client_id=lambda _s, cid: {"orderId": cid},
        order_status=lambda _s, oid: oid,
        cancel_order=lambda _s, oid: oid,
    )
    api_adapter = ol.StrategyExecutorLifecycleApi(exec_stub)
    with pytest.raises(ProviderError):
        api_adapter.order_by_client_id("BTCUSDC", True)
    with pytest.raises(ProviderError):
        api_adapter.order_status("BTCUSDC", None)
    with pytest.raises(ProviderError):
        api_adapter.cancel_order("BTCUSDC", "   ")

def test_auditedstrategyexecutor_ohlc_series_rejects_bool_invalid_interval_min_iter113_s4():
    """[ITER113 #4] AuditedStrategyExecutor.ohlc_series rejects bool/invalid interval_min."""
    # 4. AuditedStrategyExecutor.ohlc_series rejects bool/invalid interval_min
    audited = AuditedStrategyExecutor(SimpleNamespace(name="Stub", ohlc_closes=lambda _s, _i: [100.0]))
    with pytest.raises(ProviderError):
        audited.ohlc_series("BTCUSDC", True)
    with pytest.raises(ProviderError):
        audited.ohlc_series("BTCUSDC", 0)

def test_state_io_discard_generation_file_safely_ignores_bool_int_iter113_s5():
    """[ITER113 #5] state_io.discard_generation_file safely ignores bool/int."""
    import state_io as sio
    # 5. state_io.discard_generation_file safely ignores bool/int
    sio.discard_generation_file(True)
    sio.discard_generation_file(123)

def test_order_rollback_sanitizes_nan_level_in_extract_order_restore_params_iter121_s3():
    """[ITER121 #3] order_rollback sanitizes NaN level in extract_order_restore_params."""
    # 3. order_rollback sanitizes NaN level in extract_order_restore_params
    p = ol.extract_order_restore_params({"side": "SELL", "price": 100.0, "qty": 1.0, "level": float("nan")})
    assert p["level"] is None

class _DummyLifecycleExec:
    def __init__(self, name="kraken", *, echo_args=False):
        self.name = name
        self._echo_args = echo_args

    def reconciliation_capabilities(self):
        return OrderReconciliationCapabilities(
            lookup_by_client_order_id=True,
            status_by_order_id=True,
            cancel_by_order_id=True,
            not_found_reliable_for_seconds=3600.0,
        )

    def submit_order(self, symbol, side, qty, price=None, **kw):
        return "  OID-777  "

    def order_status(self, symbol, order_id):
        if self._echo_args:
            return {"symbol": symbol, "oid": order_id}
        return OrderStatus("open", filled_qty=0.0, cost=0.0, fee=0.0)

    def cancel_order(self, symbol, order_id):
        if self._echo_args:
            return {"symbol": symbol, "oid": order_id}
        return None

    def order_by_client_id(self, symbol, client_order_id):
        if self._echo_args:
            return {"symbol": symbol, "cid": client_order_id}
        return {"order_id": "OID-777", "status": "open"}


def test_auditedstrategyexecutor_normalizes_symbol_case_whitespace_and_order_id_across_iter123_s2(tmp_path):
    """[ITER123 #2] AuditedStrategyExecutor normalizes symbol case/whitespace and order_id across submit -> order_status & StrategyExecutorLifecycleApi rejects NaN/'nan'/'null' and normalizes integer float IDs."""
    from providers.execution_audit import ExecutionAudit
    # 2. AuditedStrategyExecutor normalizes symbol case/whitespace and order_id across submit -> order_status
    audit = ExecutionAudit(directory=str(tmp_path / "audit"))
    wrapped = AuditedStrategyExecutor(_DummyLifecycleExec("kraken"), audit=audit, venue="kraken")
    oid = wrapped.submit_order_with_intent("intent-abc", " xbtusd ", "buy", 0.01, 50000.0)
    assert oid == "OID-777"
    assert wrapped._intent("XBTUSD", "OID-777") == "intent-abc"
    for bad_cid in (float("nan"), "nan", "null", 12.5):
        with pytest.raises(ProviderError):
            wrapped.order_by_client_id("XBTUSD", bad_cid)


    # 3. StrategyExecutorLifecycleApi rejects NaN/"nan"/"null" and normalizes integer float IDs
    life_api = ol.StrategyExecutorLifecycleApi(wrapped)
    for bad_id in (float("nan"), "nan", "null", 12.5):
        with pytest.raises(ProviderError):
            life_api.order_status("XBTUSD", bad_id)
        with pytest.raises(ProviderError):
            life_api.cancel_order("XBTUSD", bad_id)
        with pytest.raises(ProviderError):
            life_api.order_by_client_id("XBTUSD", bad_id)
    assert life_api.order_by_client_id("XBTUSD", " cid-123 ") == {"order_id": "OID-777", "status": "open"}

def test_extract_order_restore_params_normalizes_string_level_to_float_and_rejects_iter140_s2():
    """[ITER140 #2] extract_order_restore_params normalizes string level to float and rejects sentinel/container level & kind."""
    # 2. extract_order_restore_params normalizes string level to float and rejects sentinel/container level & kind
    p_valid = extract_order_restore_params({"side": "SELL", "price": 100.0, "qty": 1.0, "level": "11.0", "kind": "TP"})
    assert p_valid["level"] == pytest.approx(11.0)
    assert isinstance(p_valid["level"], float)
    for bad_lvl in ("nan", "null", -2.0, 0.0, [5.0], {"lvl": 5.0}, True):
        p_bad = extract_order_restore_params({"side": "SELL", "price": 100.0, "qty": 1.0, "level": bad_lvl, "kind": bad_lvl})
        assert p_bad["level"] is None
        assert p_bad["kind"] == "TP"

def test_accepted_order_persistence__accepted_is_durable_normalizes_ids_rejects_sentinels_iter149_s2():
    """[ITER149 #2] accepted_order_persistence._accepted_is_durable normalizes IDs & rejects sentinels."""
    # 2. accepted_order_persistence._accepted_is_durable normalizes IDs & rejects sentinels
    class DummyRetryMod:
        def __init__(self, records):
            self._records = records

        def load_all(self):
            return self._records

    claim = {"id": "1001.0", "place_kwargs": {"client_order_id": "CID-1"}}
    mod_ok = DummyRetryMod(
        [{"id": "1001", "lifecycle": "accepted", "order_id": "777", "place_kwargs": {"client_order_id": "CID-1"}}],
    )
    assert ol._accepted_is_durable(mod_ok, claim, {"orderId": 777.0}) is True

    mod_bad_sentinel = DummyRetryMod(
        [{"id": "1001", "lifecycle": "accepted", "order_id": "0", "place_kwargs": {"client_order_id": "CID-1"}}],
    )
    assert ol._accepted_is_durable(mod_bad_sentinel, claim, {"orderId": "0"}) is False

def test_order_outcomes_log_sanitizes_nan_inf_containers_and_none_regime_context_iter151_s1(tmp_path):
    """[ITER151 #1] order_outcomes_log sanitizes NaN/Inf/containers and None regime_context.fresh/fallback_used."""
    import glob
    # 1. order_outcomes_log sanitizes NaN/Inf/containers and None regime_context.fresh/fallback_used
    orig_dir = ol.ORDER_OUTCOMES_LOG_DIR
    ol.ORDER_OUTCOMES_LOG_DIR = str(tmp_path / "outcomes")
    try:
        class DummyRegimeCtx:
            regime = "UP"
            strength = 0.75
            source = "short"
            fresh = None
            fallback_used = None
            reason = "ok"

        ol.log_order_outcome(
            "BTCUSDC",
            "BUY",
            float("nan"),
            float("inf"),
            "refused",
            ["bad"],
            "nan",
            caller=None,
            regime_context=DummyRegimeCtx(),
        )
        files = glob.glob(str(tmp_path / "outcomes" / "order_outcomes_*.log"))
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read().strip()
        parts = content.split("|")
        # price, qty, refuse_reason, caller, motivation, and fresh/fallback_used should be "" not "nan"/"inf"/"None"
        assert parts[3] == ""
        assert parts[4] == ""
        assert parts[6] == ""
        assert parts[7] == ""
        assert parts[8] == ""
        assert parts[12] == ""
        assert parts[13] == ""
    finally:
        ol.ORDER_OUTCOMES_LOG_DIR = orig_dir

def test_order_retry_valid_record_enqueue_sentinel_order_id_replaces_order_id_rejection_iter156_s2(tmp_path, monkeypatch):
    """[ITER156 #2] order_retry valid_record & enqueue sentinel order_id / replaces_order_id rejection."""
    # 2. order_retry valid_record & enqueue sentinel order_id / replaces_order_id rejection
    q_file = tmp_path / "order_retry_queue.jsonl"
    l_file = tmp_path / "order_retry_queue.lock"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "LOCK_FILE", str(l_file))
    monkeypatch.setattr(oq, "RETRY_ENABLED", True)

    assert oq.valid_record({
        "id": "r1",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 1.0,
        "created_ts": 1000.0,
        "revision": 0,
        "lifecycle": "accepted",
        "order_id": "0",
    }) is False
    assert oq.valid_record({
        "id": "r2",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "qty": 1.0,
        "created_ts": 1000.0,
        "revision": 0,
        "lifecycle": "awaiting_cancel",
        "replaces_order_id": "nan",
        "replaces_original_qty": 1.0,
    }) is False
    assert oq.enqueue(
        "BTCUSDC",
        "BUY",
        1.0,
        lifecycle="awaiting_cancel",
        replaces_order_id="0",
        replaces_original_qty=1.0,
    ) is None
    rec_id = oq.enqueue(
        "BTCUSDC",
        "BUY",
        1.0,
        lifecycle="awaiting_cancel",
        replaces_order_id="777.0",
        replaces_original_qty=1.0,
    )
    assert rec_id is not None
    loaded = oq.get(rec_id)
    assert loaded is not None and loaded["replaces_order_id"] == "777"
    # Deduplicates against integer 777
    rec_id_dup = oq.enqueue(
        "BTCUSDC",
        "BUY",
        1.0,
        lifecycle="awaiting_cancel",
        replaces_order_id=777,
        replaces_original_qty=1.0,
    )
    assert rec_id_dup == rec_id

def test_trackedorderresult_order_known_rejects_nan_none_iter157_s1():
    """[ITER157 #1] TrackedOrderResult.order_known rejects '0', 'nan', 'None'."""
    # 1. TrackedOrderResult.order_known rejects "0", "nan", "None"
    assert ol.TrackedOrderResult("active", {"order_id": "0"}).order_known is False
    assert ol.TrackedOrderResult("active", {"order_id": "nan"}).order_known is False
    assert ol.TrackedOrderResult("active", {"order_id": "555.0"}).order_known is True

def test_strategyexecutorlifecycleapi_rejects_and_normalizes_iter157_s2():
    """[ITER157 #2] StrategyExecutorLifecycleApi rejects '0' and normalizes .0."""
    # 2. StrategyExecutorLifecycleApi rejects "0" and normalizes .0
    adapter = ol.StrategyExecutorLifecycleApi(_DummyLifecycleExec("binance", echo_args=True))
    with pytest.raises(ProviderError):
        adapter.order_by_client_id("BTCUSDC", "0")
    with pytest.raises(ProviderError):
        adapter.order_status("BTCUSDC", "0")
    with pytest.raises(ProviderError):
        adapter.cancel_order("BTCUSDC", "nan")
    assert adapter.order_by_client_id("BTCUSDC", "888.0")["cid"] == "888"
    assert adapter.order_status("BTCUSDC", "999.0")["oid"] == "999"

def test_trackedorderlifecycle_retry_on_lookup_error_and_identity_order_fields_normalization_iter157_s3():
    """[ITER157 #3] TrackedOrderLifecycle retry_on_lookup_error and identity/order_fields normalization."""
    from providers.strategy_executor import SubmissionOutcome
    # 3. TrackedOrderLifecycle retry_on_lookup_error and identity/order_fields normalization
    lc_false = ol.TrackedOrderLifecycle(
        object(),
        provider_name="binance",
        retry_on_lookup_error="false",
        clock=lambda: 1000.0,
    )
    assert lc_false.retry_on_lookup_error is False
    lc_nan = ol.TrackedOrderLifecycle(
        object(),
        provider_name="binance",
        retry_on_lookup_error=float("nan"),
        clock=lambda: 1000.0,
    )
    assert lc_nan.retry_on_lookup_error is False

    for bad_kw in (
        {"intent_id": "0"},
        {"client_order_id": "nan"},
        {"symbol": "null"},
    ):
        kw = {
            "intent_id": "101.0",
            "client_order_id": "202.0",
            "symbol": "BTCUSDC",
            "side": "B",
            "requested_qty": 1.0,
        }
        kw.update(bad_kw)
        with pytest.raises(ValueError):
            lc_false.new_intent(**kw)

    intent = lc_false.new_intent(
        intent_id="101.0",
        client_order_id="202.0",
        symbol="BTCUSDC",
        side="B",
        requested_qty=1.0,
    )
    assert intent["intent_id"] == "101"
    assert intent["client_order_id"] == "202"
    assert intent["side"] == "BUY"

    assert ol.TrackedOrderLifecycle._order_fields({"orderId": "0"}) == {}
    assert ol.TrackedOrderLifecycle._order_fields({"orderId": "333.0"}) == {"order_id": "333"}
    with pytest.raises(ValueError):
        SubmissionOutcome("accepted", order_id="0")
    assert ol.TrackedOrderLifecycle._order_fields(
        SubmissionOutcome("unknown", native={"orderId": "0"})
    ) == {"submission_outcome": "unknown"}

def test_order_outcomes_log_sanitizes_regime_context_fields_and_non_numeric_price_iter160_s3(tmp_path, monkeypatch):
    """[ITER160 #3] order_outcomes_log: sanitizes regime_context fields and non-numeric price/qty strings."""
    # 3. order_outcomes_log: sanitizes regime_context fields and non-numeric price/qty strings
    monkeypatch.setattr(ol, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    assert ol._clean_numeric_field("not-a-number") == ""
    assert ol._clean_numeric_field("60000.5") == "60000.5"

    class _RegCtx:
        regime = "BULL|STRONG"
        strength = 0.75
        source = "None"
        fresh = "false"
        fallback_used = 1
        reason = "nan"

    ol.log_order_outcome(
        "BTCUSDC",
        "BUY",
        "not-a-number",
        0.01,
        "refused",
        "short_trend",
        "test",
        caller="unit",
        regime_context=_RegCtx(),
    )
    log_files = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(log_files) == 1
    row = log_files[0].read_text(encoding="utf-8").strip().split("|")
    assert row[3] == ""  # non-numeric price cleaned to empty
    assert row[9] == "BULL/STRONG"
    assert row[11] == ""  # "None" source cleaned to empty
    assert row[12] == "False"  # "false" string parsed as False
    assert row[13] == "True"  # 1 parsed as True
    assert row[14] == ""  # "nan" reason cleaned to empty

def test_accepted_order_persistence_complete_claim_returning_false_is_not_treated_as_iter161_s2(monkeypatch):
    """[ITER161 #2] accepted_order_persistence: complete_claim returning 'false' is not treated as success."""
    # 2. accepted_order_persistence: complete_claim returning "false" is not treated as success
    class _FakeRetry:
        @staticmethod
        def complete_claim(claim, status, **kw):
            return "false"

        @staticmethod
        def load_all():
            return None

    monkeypatch.setattr(ol.time, "sleep", lambda _s: None)
    notified = []
    monkeypatch.setattr(ol.alert, "notify", lambda **kw: notified.append(kw))
    ok = ol.complete_accepted_claim(
        _FakeRetry,
        {"id": "c1", "qty": float("nan"), "requested_price": True, "place_kwargs": {"client_order_id": "cid-1"}},
        order={"orderId": 555},
        provider_name="Binance",
        symbol="BTCUSDC",
        side="BUY",
    )
    assert ok is False
    assert len(notified) == 1
    assert "qty=None price=None" in notified[0]["body"]

def test_order_outcomes_log_with_dict_regime_context_and_primitive_regime_context_iter168_s5(tmp_path, monkeypatch):
    """[ITER168 #5] order_outcomes_log with dict regime_context and primitive regime_context."""
    # 5. order_outcomes_log with dict regime_context and primitive regime_context
    monkeypatch.setattr(ol, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path))
    ol.log_order_outcome(
        "BTCUSDC",
        "BUY",
        65000.0,
        0.01,
        "accepted",
        "",
        "entry",
        caller="test",
        regime_context={
            "regime": "bull",
            "strength": "0.75",
            "source": "cache",
            "fresh": "yes",
            "fallback_used": "no",
            "reason": "strong_trend",
        },
    )
    ol.log_order_outcome(
        "BTCUSDC",
        "SELL",
        66000.0,
        0.01,
        "accepted",
        "",
        "exit",
        caller="test",
        regime_context=False,
    )
    log_files = list(tmp_path.glob("order_outcomes_*.log"))
    assert len(log_files) == 1
    lines = log_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "bull|0.7500|cache|True|False|strong_trend" in lines[0]
    assert len(lines[1].split("|")) == 9

def test_instrument__parse_bool_flag_side_aliases_bid_ask_decimal_free_balance_iter171_s2(monkeypatch):
    """[ITER171 #2] Instrument._parse_bool_flag, side aliases (BID/ASK), Decimal free_balance, cancelorders='false', and GT_M dict/case."""
    from decimal import Decimal
    from providers.base import MarketDataProvider
    from providers.market_api import MarketApi
    from providers.quantity import QuantityDecision
    # 2. Instrument._parse_bool_flag, side aliases (BID/ASK), Decimal free_balance, cancelorders="false", and GT_M dict/case
    qd_calls = []

    class _TestProvider(MarketDataProvider):
        @property
        def name(self):
            return "TestProv"

        def get_current_price(self, symbol):
            return 100.0

        def supports_symbol(self, symbol):
            return True

        def free_balance(self, asset):
            return Decimal("500.0")

        def quantity_decision(self, symbol, side, price, qty, **kw):
            qd_calls.append(kw)
            return QuantityDecision(1.0, 5.0, 1.0, 5.0, 1.0)

        def place_order(self, symbol, side, price, qty, force=False, **kw):
            return {"orderId": "777", "side": side, "qty": qty}

    test_api = MarketApi([_TestProvider()])
    inst = inst_mod.Instrument(
        name="BTCUSDC",
        symbol="BTCUSDC",
        provider="TestProv",
        base="BTC",
        quote="USDC",
        params={"mt.flag": ""},
        api=test_api,
    )
    assert inst.param("mt", "flag", default=True, cast=bool) is True

    monkeypatch.setattr(inst_mod.order_guard, "daily_limit_guard", lambda *a, **kw: (True, None))
    monkeypatch.setattr(inst_mod.order_guard, "check_math_profit_reference", lambda *a, **kw: True)
    monkeypatch.setattr(inst_mod.order_guard, "profit_guard", lambda *a, **kw: True)
    monkeypatch.setattr(inst_mod.order_lifecycle, "log_order_outcome", lambda *a, **kw: None)

    outcome = {}
    res = inst.place(
        "BID",
        100.0,
        1.0,
        caller_owns_retry=True,
        wait_for_trend=False,
        cancelorders="false",
        medium_trend_action=" DEFER ",
        regime_context={"regime": " BEAR "},
        _outcome_context=outcome,
    )
    assert res is None and outcome.get("reason") == "medium_trend_deferred"
    assert qd_calls[-1]["cancelorders"] is False
    assert qd_calls[-1]["known_balance"] == pytest.approx(500.0)

def test_order_retry_price_gate_ok_supports_bid_long_and_ask_a_iter179_s4():
    """[ITER179 #4] order_retry.price_gate_ok supports BID/LONG and ASK/A/SHORT."""
    # 4. order_retry.price_gate_ok supports BID/LONG and ASK/A/SHORT
    assert oq.price_gate_ok({"side": "BID", "requested_price": 100.0}, 99.0, tol=0.0) is True
    assert oq.price_gate_ok({"side": "ASK", "requested_price": 100.0}, 101.0, tol=0.0) is True
