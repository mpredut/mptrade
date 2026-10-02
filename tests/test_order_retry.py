"""Tests for the replacement outbox store and its due/expiry policy.

All queue files are isolated in a temporary directory.
"""
import os
import sys
import time
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_retry as oq
from providers.strategy_executor import OrderStatus


class OrderRetryStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        oq.QUEUE_FILE = os.path.join(self.tmp, "order_retry_queue.jsonl")
        oq.LOCK_FILE = os.path.join(self.tmp, "order_retry_queue.lock")
        oq.RETRY_ENABLED = True
        oq.RETRY_INTERVAL_SEC = 300.0
        oq.RETRY_TTL_SEC = 86400.0
        oq.RETRY_MAX_ATTEMPTS = 0
        oq.RETRY_PRICE_TOL = 0.002
        oq.RETRY_DEDUP = True
        oq.RETRY_MAX_QUEUE = 500
        oq.RETRY_CLAIM_LEASE_SEC = 120.0
        oq.RETRY_NOT_FOUND_MAX_AGE_SEC = 30 * 24 * 60 * 60

    def _terminal_remainder(self, *, terminal_at=1400.0):
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 2.0, {}, requested_price=100.0,
            now=1000.0, kind="terminal-remainder")
        self.assertTrue(oq.mark_accepted(
            record_id, {"orderId": 91, "status": "NEW"}, now=1001.0))
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

    def _reset_env(self):
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
            a = oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0, now=1000.0)
            oq.mark_failure(
                a, "profit_guard", now=1000.0, submission_state="refused")
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
            oq.mark_accepted(first, {"orderId": 7, "status": "NEW"}, now=1001.0)

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
            first = oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=63000.0, now=1000.0)
            oq.mark_failure(
                first, "profit_guard", now=1000.0, submission_state="refused")
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
                             now=1000.0)
            first = oq.get(rid)
            first_cid = first["place_kwargs"]["client_order_id"]
            oq.mark_failure(
                rid, "profit_guard", now=1000.0, submission_state="refused")
            oq.enqueue("BTCUSDC", "BUY", 2.0, {}, requested_price=99.0,
                       now=1001.0)

            self.assertFalse(oq.resolve_record(rid, client_order_id=first_cid))
            self.assertEqual(oq.get(rid)["qty"], 2.0)

        with self.subTest(msg="mark_failure_preserves_pre_submit_client_id"):
            self._reset_env()
            rid = oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                             failure_reason="submit_pending", now=1000.0)
            client_id = oq.get(rid)["place_kwargs"]["client_order_id"]

            self.assertTrue(oq.mark_failure(
                rid, "response_without_order_id", now=1001.0,
                client_order_id=client_id))
            rec = oq.get(rid)
            self.assertEqual(rec["place_kwargs"]["client_order_id"], client_id)
            self.assertEqual(rec["last_failure_reason"], "response_without_order_id")

        with self.subTest(msg="mark_accepted_keeps_exact_record_until_terminal"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                provider_name="Binance", intent_id="strategy-cycle-entry",
                kind="ENTRY", now=1000.0)
            client_id = oq.get(rid)["place_kwargs"]["client_order_id"]

            self.assertTrue(oq.mark_accepted(
                rid, {"orderId": 123, "status": "NEW"}, now=1100.0,
                client_order_id=client_id))

            rec = oq.get(rid)
            self.assertEqual(rec["lifecycle"], "accepted")
            self.assertEqual(rec["order_id"], "123")
            self.assertEqual(rec["provider_name"], "Binance")
            self.assertEqual(rec["intent_id"], "strategy-cycle-entry")
            self.assertEqual(rec["kind"], "ENTRY")
            self.assertFalse(oq.is_expired(rec, now=1000.0 + 100 * 86400))
            self.assertFalse(oq.is_due(rec, now=1399.0))
            self.assertTrue(oq.is_due(rec, now=1400.0))

        with self.subTest(msg="mark_accepted_rejects_wrong_revision_client_id"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)

            self.assertFalse(oq.mark_accepted(
                rid, {"orderId": 123}, now=1100.0,
                client_order_id="different"))
            self.assertEqual(oq.get(rid)["lifecycle"], "submit_pending")

        with self.subTest(msg="legacy_resolve_record_cannot_remove_accepted_tracker"):
            self._reset_env()
            rid = oq.enqueue(
                "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
            cid = oq.get(rid)["place_kwargs"]["client_order_id"]
            self.assertTrue(oq.mark_accepted(
                rid, {"orderId": 91, "status": "NEW"}, now=1100.0,
                client_order_id=cid))

            self.assertFalse(oq.resolve_record(rid, client_order_id=cid))
            self.assertEqual(oq.get(rid)["order_id"], "91")

        with self.subTest(msg="order_id_parser_handles_provider_shapes"):
            self.assertEqual(oq.order_id_from_response({"orderId": 1}), "1")
            self.assertEqual(oq.order_id_from_response({"id": "two"}), "two")
            self.assertEqual(oq.order_id_from_response({"txid": ["three"]}), "three")
            self.assertIsNone(oq.order_id_from_response({"status": "NEW"}))

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
        oq.mark_accepted(
            rid, {"orderId": 77, "status": "NEW"}, now=now + 100.0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
