"""Focused tests for cross-process Binance account-cache reader versions."""

import gzip
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import binance_cache_health as health
import cacheManager as cm


def _order(order_id, timestamp):
    return {
        "orderId": order_id,
        "price": "100",
        "quantity": "1",
        "timestamp": timestamp,
        "side": "BUY",
    }


def _trade(trade_id, timestamp):
    return {
        "symbol": "BTCUSDC",
        "id": trade_id,
        "orderId": trade_id,
        "price": "100",
        "qty": "1",
        "time": timestamp,
        "isBuyer": True,
    }


class AccountCacheReaderSyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="account-cache-reader-sync-"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _reset_temp(self):
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory(
            prefix="account-cache-reader-sync-"
        )

    def _order_manager(self, items, fetchtime):
        path = os.path.join(self.tmp.name, "orders.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {"items": {"BTCUSDC": items},
                     "fetchtime": {"BTCUSDC": fetchtime}},
                    handle,
                )
        return cm.CacheOrderManager(
            3600, ["BTCUSDC"], path, api_client=cm.api
        )

    def _trade_manager(self, items, fetchtime):
        path = os.path.join(self.tmp.name, "trades.jsonl")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(
                        {"s": "BTCUSDC", "i": item},
                        separators=(",", ":"),
                    ) + "\n")
            with open(path + ".meta", "w", encoding="utf-8") as handle:
                json.dump({"fetchtime": {"BTCUSDC": fetchtime}}, handle)
        return cm.CacheTradeManager(
            3600, ["BTCUSDC"], path, api_client=cm.api
        )

    def test_order_reader_versions_and_reloads(self):
        with self.subTest(msg="reloads_new_version_and_new_fill"):
            self._reset_temp()
            first = _order(1, 1000)
            second = _order(2, 2000)
            writer = self._order_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            first_version = writer._persisted_data_version

            reader = self._order_manager([], 0)
            reader.ensure_persisted_version(first_version)
            self.assertEqual([first], reader.cache["BTCUSDC"])

            with writer.lock:
                writer.cache["BTCUSDC"].append(second)
                writer.fetchtime_time_per_symbol["BTCUSDC"] = 2000
            self.assertTrue(writer.save_state_to_file())
            reader.ensure_persisted_version(writer._persisted_data_version)
            self.assertEqual([first, second], reader.cache["BTCUSDC"])

        with self.subTest(msg="same_version_performs_no_reload"):
            self._reset_temp()
            writer = self._order_manager([_order(1, 1000)], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._order_manager([], 0)
            reader.ensure_persisted_version(writer._persisted_data_version)

            with patch.object(
                    reader, "_ensure_order_version") as reload_snapshot:
                reader.ensure_persisted_version(writer._persisted_data_version)
            reload_snapshot.assert_not_called()

        with self.subTest(msg="identical_snapshot_keeps_stable_version"):
            self._reset_temp()
            writer = self._order_manager([_order(1, 1000)], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            first_version = writer._persisted_data_version
            self.assertTrue(writer.save_state_to_file())
            self.assertEqual(first_version, writer._persisted_data_version)

        with self.subTest(msg="corruption_refuses_and_preserves_memory"):
            self._reset_temp()
            first = _order(1, 1000)
            writer = self._order_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._order_manager([], 0)
            reader.ensure_persisted_version(writer._persisted_data_version)
            previous = list(reader.cache["BTCUSDC"])

            with writer.lock:
                writer.cache["BTCUSDC"].append(_order(2, 2000))
                writer.fetchtime_time_per_symbol["BTCUSDC"] = 2000
            self.assertTrue(writer.save_state_to_file())
            new_version = writer._persisted_data_version
            with open(writer.filename, "w", encoding="utf-8") as handle:
                handle.write("{broken")

            with self.assertRaises(health.AccountCacheNotReady) as raised:
                reader.ensure_persisted_version(new_version)
            self.assertEqual(
                "order_cache_reader_not_current", raised.exception.reason
            )
            self.assertEqual(previous, reader.cache["BTCUSDC"])

        with self.subTest(msg="marker_manifest_mismatch_refuses"):
            self._reset_temp()
            writer = self._order_manager([_order(1, 1000)], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._order_manager([], 0)
            with self.assertRaises(health.AccountCacheNotReady):
                reader.ensure_persisted_version("snapshot:not-the-manifest")

    def test_trade_reader_versions_and_reloads(self):
        with self.subTest(msg="uses_only_new_suffix"):
            self._reset_temp()
            first = _trade(1, 1000)
            second = _trade(2, 2000)
            writer = self._trade_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._trade_manager([], 0)
            first_boundary = reader._loaded_committed_bytes

            with writer.lock:
                writer.cache["BTCUSDC"].append(second)
                writer.fetchtime_time_per_symbol["BTCUSDC"] = 2000
            self.assertTrue(writer.save_state_to_file())
            new_boundary = writer._persisted_committed_bytes
            with patch.object(
                    reader, "_read_trade_records_strict",
                    wraps=reader._read_trade_records_strict) as read_range:
                reader.ensure_persisted_version(writer._persisted_data_version)
            read_range.assert_called_once_with(first_boundary, new_boundary)
            self.assertEqual([first, second], reader.cache["BTCUSDC"])

        with self.subTest(msg="changed_stream_forces_full_reload"):
            self._reset_temp()
            writer = self._trade_manager([_trade(1, 1000)], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._trade_manager([], 0)
            old_stream = reader._loaded_stream_id

            self.assertTrue(writer.compact_jsonl())
            self.assertNotEqual(old_stream, writer._persisted_stream_id)
            with patch.object(
                    reader, "_read_trade_snapshot_strict",
                    wraps=reader._read_trade_snapshot_strict) as full_reload:
                reader.ensure_persisted_version(writer._persisted_data_version)
            full_reload.assert_called_once()

        with self.subTest(msg="corrupt_committed_content_refuses_and_preserves_memory"):
            self._reset_temp()
            first = _trade(1, 1000)
            writer = self._trade_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._trade_manager([], 0)
            previous = list(reader.cache["BTCUSDC"])

            with writer.lock:
                writer.cache["BTCUSDC"].append(_trade(2, 2000))
                writer.fetchtime_time_per_symbol["BTCUSDC"] = 2000
            self.assertTrue(writer.save_state_to_file())
            new_version = writer._persisted_data_version
            with open(writer.filename, "wb") as handle:
                handle.write(b"{broken\n")

            with self.assertRaises(health.AccountCacheNotReady) as raised:
                reader.ensure_persisted_version(new_version)
            self.assertEqual(
                "trade_cache_reader_not_current", raised.exception.reason
            )
            self.assertEqual(previous, reader.cache["BTCUSDC"])

        with self.subTest(msg="uncommitted_partial_tail_is_ignored"):
            self._reset_temp()
            first = _trade(1, 1000)
            writer = self._trade_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            committed = writer._persisted_committed_bytes
            with open(writer.filename, "ab") as handle:
                handle.write(b"{partial-uncommitted")
            self.assertGreater(os.path.getsize(writer.filename), committed)

            reader = self._trade_manager([], 0)
            self.assertEqual([first], reader.cache["BTCUSDC"])
            self.assertEqual(committed, reader._loaded_committed_bytes)

        with self.subTest(msg="dirty_reader_reloads_same_version_then_resumes_suffix"):
            self._reset_temp()
            first = _trade(1, 1000)
            second = _trade(2, 2000)
            local_only = _trade(999, 1500)
            writer = self._trade_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._trade_manager([], 0)
            version = writer._persisted_data_version

            with reader.lock:
                reader.cache["BTCUSDC"].extend([first, local_only])
                reader._mark_account_cache_dirty_locked()
            with patch.object(
                    reader, "_read_trade_snapshot_strict",
                    wraps=reader._read_trade_snapshot_strict) as full_reload:
                reader.ensure_persisted_version(version)
            full_reload.assert_called_once()
            self.assertEqual([first], reader.cache["BTCUSDC"])

            old_boundary = reader._loaded_committed_bytes
            with writer.lock:
                writer.cache["BTCUSDC"].append(second)
                writer.fetchtime_time_per_symbol["BTCUSDC"] = 2000
            self.assertTrue(writer.save_state_to_file())
            with patch.object(
                    reader, "_read_trade_records_strict",
                    wraps=reader._read_trade_records_strict) as suffix_read:
                reader.ensure_persisted_version(writer._persisted_data_version)
            suffix_read.assert_called_once_with(
                old_boundary, writer._persisted_committed_bytes
            )
            self.assertEqual([first, second], reader.cache["BTCUSDC"])

    def test_trade_compaction_and_rotation_behaviors(self):
        with self.subTest(msg="health_publish_failure_keeps_trade_data_and_meta_coherent"):
            self._reset_temp()
            writer = self._trade_manager([_trade(1, 1000)], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            with writer.lock:
                writer.cache["BTCUSDC"].append(_trade(2, 2000))
                writer.fetchtime_time_per_symbol["BTCUSDC"] = 2000

            with (
                patch.object(
                    writer, "_is_canonical_account_cache", return_value=True
                ),
                patch.object(
                    cm.account_cache_health,
                    "record_persisted_version",
                    side_effect=RuntimeError("marker unavailable"),
                ),
            ):
                self.assertTrue(writer.save_state_to_file())

            with open(writer.filename + ".meta", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(
                metadata["committed_bytes"], os.path.getsize(writer.filename)
            )
            reader = self._trade_manager([], 0)
            self.assertEqual(
                [_trade(1, 1000), _trade(2, 2000)],
                reader.cache["BTCUSDC"],
            )

        with self.subTest(msg="failed_trade_compaction_restores_prior_certified_generation"):
            self._reset_temp()
            first = _trade(1, 1000)
            writer = self._trade_manager([first], 1000)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            reader = self._trade_manager([], 0)
            old_version = writer._persisted_data_version
            old_memory = list(reader.cache["BTCUSDC"])

            with writer.lock:
                writer.cache["BTCUSDC"][0] = {
                    **writer.cache["BTCUSDC"][0],
                    "price": "200",
                }
                writer._mark_account_cache_dirty_locked()
            with patch.object(writer, "_write_meta", return_value=False):
                self.assertFalse(writer.compact_jsonl())

            with reader.lock:
                reader._mark_account_cache_dirty_locked()
            reader.ensure_persisted_version(old_version)
            self.assertEqual(old_memory, reader.cache["BTCUSDC"])
            restarted = cm.CacheTradeManager(
                3600, ["BTCUSDC"], writer.filename, api_client=cm.api
            )
            self.assertEqual(old_memory, restarted.cache["BTCUSDC"])
            self.assertFalse(os.path.exists(writer.filename + ".previous"))

        with self.subTest(msg="oversized_trade_rotation_publishes_readable_generation"):
            self._reset_temp()
            now_ms = 2_000_000_000_000
            trades = [_trade(index, now_ms + index) for index in range(1, 7)]
            writer = self._trade_manager(trades, now_ms)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            old_version = writer._persisted_data_version
            writer.MAX_FILE_BYTES = 1
            writer.ROTATE_KEEP_FRACTION = 0.5

            with (
                patch.object(
                    writer, "_is_canonical_account_cache", return_value=True
                ),
                patch.object(
                    cm.account_cache_health, "record_persisted_version"
                ) as publish,
            ):
                writer.maintain_append_persist()

            with open(writer.filename + ".meta", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertTrue(os.path.exists(writer.filename))
            self.assertEqual(
                metadata["committed_bytes"], os.path.getsize(writer.filename)
            )
            self.assertEqual(
                metadata["data_version"], writer._persisted_data_version
            )
            self.assertNotEqual(old_version, metadata["data_version"])
            publish.assert_called_once_with(
                "CacheTradeManager", metadata["data_version"]
            )
            self.assertFalse(os.path.exists(writer.filename + ".previous"))

            restarted = cm.CacheTradeManager(
                3600, ["BTCUSDC"], writer.filename, api_client=cm.api
            )
            self.assertEqual(trades[-3:], restarted.cache["BTCUSDC"])
            restarted.ensure_persisted_version(metadata["data_version"])

            archives = [
                path for path in os.listdir(self.tmp.name)
                if path.startswith(os.path.basename(writer.filename) + ".")
                and path.endswith(".archive.gz")
            ]
            self.assertEqual(1, len(archives))
            with gzip.open(
                    os.path.join(self.tmp.name, archives[0]),
                    "rt", encoding="utf-8") as handle:
                archived = [json.loads(line)["i"] for line in handle]
            self.assertEqual(trades, archived)

        with self.subTest(msg="failed_oversized_trade_rotation_preserves_certified_generation"):
            self._reset_temp()
            now_ms = 2_000_000_000_000
            trades = [_trade(index, now_ms + index) for index in range(1, 5)]
            writer = self._trade_manager(trades, now_ms)
            writer.save_state = True
            self.assertTrue(writer.save_state_to_file())
            writer.MAX_FILE_BYTES = 1
            writer.ROTATE_KEEP_FRACTION = 0.5
            old_version = writer._persisted_data_version
            old_cache = list(writer.cache["BTCUSDC"])
            with open(writer.filename, "rb") as handle:
                old_data = handle.read()
            with open(writer.filename + ".meta", "rb") as handle:
                old_meta = handle.read()

            with (
                patch.object(writer, "_write_meta", return_value=False),
                patch.object(
                    cm.account_cache_health, "record_persisted_version"
                ) as publish,
            ):
                writer.maintain_append_persist()

            publish.assert_not_called()
            self.assertEqual(old_version, writer._persisted_data_version)
            self.assertEqual(old_cache, writer.cache["BTCUSDC"])
            with open(writer.filename, "rb") as handle:
                self.assertEqual(old_data, handle.read())
            with open(writer.filename + ".meta", "rb") as handle:
                self.assertEqual(old_meta, handle.read())
            self.assertFalse(os.path.exists(writer.filename + ".previous"))
            self.assertFalse(any(
                ".archive" in path for path in os.listdir(self.tmp.name)
            ))

            restarted = cm.CacheTradeManager(
                3600, ["BTCUSDC"], writer.filename, api_client=cm.api
            )
            self.assertEqual(trades, restarted.cache["BTCUSDC"])
            restarted.ensure_persisted_version(old_version)

    def test_periodic_trade_persist_waits_for_generation_without_manager_lock(self):
        now_ms = 2_000_000_000_000
        writer = self._trade_manager([_trade(1, now_ms)], now_ms)
        writer.save_state = True
        self.assertTrue(writer.save_state_to_file())

        generation_held = threading.Event()
        allow_generation_owner = threading.Event()
        owner_has_manager_lock = threading.Event()
        periodic_attempted_generation = threading.Event()
        periodic_done = threading.Event()
        errors = []
        real_generation_lock = writer._trade_generation_lock

        def generation_owner():
            try:
                with real_generation_lock():
                    generation_held.set()
                    if not allow_generation_owner.wait(2):
                        raise RuntimeError(
                            "test did not release generation owner"
                        )
                    with writer.lock:
                        owner_has_manager_lock.set()
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        def observed_generation_lock():
            periodic_attempted_generation.set()
            return real_generation_lock()

        def periodic_persist():
            try:
                writer._persist_periodic_state(False)
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)
            finally:
                periodic_done.set()

        owner_thread = threading.Thread(
            target=generation_owner, name="generation-owner"
        )
        owner_thread.start()
        self.assertTrue(generation_held.wait(2))

        periodic_thread = threading.Thread(
            target=periodic_persist, name="periodic-persist"
        )
        try:
            with patch.object(
                    writer, "_trade_generation_lock",
                    side_effect=observed_generation_lock):
                periodic_thread.start()
                self.assertTrue(periodic_attempted_generation.wait(2))
                manager_lock_acquired = writer.lock.acquire(timeout=0.5)
                try:
                    self.assertTrue(
                        manager_lock_acquired,
                        "periodic persistence held the manager lock while "
                        "waiting for the Trade generation lock",
                    )
                finally:
                    if manager_lock_acquired:
                        writer.lock.release()
                allow_generation_owner.set()
                owner_thread.join(2)
                periodic_thread.join(2)
        finally:
            allow_generation_owner.set()
            owner_thread.join(2)
            periodic_thread.join(2)

        self.assertFalse(owner_thread.is_alive())
        self.assertFalse(periodic_thread.is_alive())
        self.assertTrue(owner_has_manager_lock.is_set())
        self.assertTrue(periodic_done.is_set())
        self.assertEqual([], errors)

    def test_equal_timestamp_stale_writer_cannot_replace_newer_trade_generation(self):
        first = _trade(1, 1000)
        second = _trade(2, 1000)
        stale_only = _trade(3, 1000)
        stale_writer = self._trade_manager([first], 1000)
        stale_writer.save_state = True
        self.assertTrue(stale_writer.save_state_to_file())
        first_version = stale_writer._persisted_data_version

        current_writer = cm.CacheTradeManager(
            3600, ["BTCUSDC"], stale_writer.filename, api_client=cm.api
        )
        current_writer.save_state = True
        self.assertEqual(
            first_version, current_writer._persisted_data_version
        )
        with current_writer.lock:
            current_writer.cache["BTCUSDC"].append(second)
            current_writer.fetchtime_time_per_symbol["BTCUSDC"] = 1000
            current_writer._mark_account_cache_dirty_locked()
        self.assertTrue(current_writer.save_state_to_file())
        current_version = current_writer._persisted_data_version
        self.assertNotEqual(first_version, current_version)
        with open(stale_writer.filename, "rb") as handle:
            current_data = handle.read()
        with open(stale_writer.filename + ".meta", "rb") as handle:
            current_meta = handle.read()

        with stale_writer.lock:
            stale_writer.cache["BTCUSDC"].append(stale_only)
            stale_writer.fetchtime_time_per_symbol["BTCUSDC"] = 1000
            stale_writer._mark_account_cache_dirty_locked()
        self.assertFalse(stale_writer.save_state_to_file())

        with open(stale_writer.filename, "rb") as handle:
            self.assertEqual(current_data, handle.read())
        with open(stale_writer.filename + ".meta", "rb") as handle:
            self.assertEqual(current_meta, handle.read())
        restarted = cm.CacheTradeManager(
            3600, ["BTCUSDC"], stale_writer.filename, api_client=cm.api
        )
        self.assertEqual([first, second], restarted.cache["BTCUSDC"])
        restarted.ensure_persisted_version(current_version)

    def test_latest_opposite_fill_uses_exchange_time_not_arrival_order(self):
        newer = {
            **_trade(101, 200),
            "price": "200",
        }
        older = {
            **_trade(100, 100),
            "price": "100",
        }
        manager = self._trade_manager([newer], 200)
        manager._persist_items("BTCUSDC", [older])
        self.assertEqual(
            [newer, older], manager.cache["BTCUSDC"]
        )
        self.assertEqual(
            200.0, manager.last_opposite_fill_price("BTCUSDC", "SELL")
        )

        manager.save_state = True
        self.assertTrue(manager.save_state_to_file())
        restarted = cm.CacheTradeManager(
            3600, ["BTCUSDC"], manager.filename, api_client=cm.api
        )
        self.assertEqual(
            200.0, restarted.last_opposite_fill_price("BTCUSDC", "SELL")
        )

    def test_order_aggregation_and_deduplication(self):
        with self.subTest(msg="rest_tranche_updates_existing_aggregate_atomically"):
            self._reset_temp()
            partial = {
                **_order(7, 1000),
                "price": 100.0,
                "quantity": 1.0,
                "status": "PARTIALLY_FILLED",
            }
            final_tranche = {
                **_order(7, 2000),
                "price": 120.0,
                "quantity": 1.0,
            }
            manager = self._order_manager([partial], 1000)
            manager._persist_items("BTCUSDC", [final_tranche])

            self.assertEqual(1, len(manager.cache["BTCUSDC"]))
            aggregate = manager.cache["BTCUSDC"][0]
            self.assertEqual(2.0, aggregate["quantity"])
            self.assertEqual(110.0, aggregate["price"])
            self.assertEqual(2000, aggregate["timestamp"])
            self.assertTrue(manager._account_cache_dirty)

        with self.subTest(msg="ws_partial_to_filled_uses_cumulative_order_aggregate"):
            self._reset_temp()
            manager = self._order_manager([], 0)
            base = {
                "s": "BTCUSDC",
                "i": 7,
                "x": "TRADE",
                "S": "BUY",
                "p": "100",
                "q": "2",
                "E": 1000,
                "t": 101,
            }
            partial = {
                **base, "X": "PARTIALLY_FILLED", "T": 1000,
                "l": "1", "L": "100", "z": "1", "Z": "100",
            }
            filled = {
                **base, "X": "FILLED", "T": 2000, "t": 102,
                "l": "1", "L": "120", "z": "2", "Z": "220",
            }
            with patch.object(cm, "get_cache_manager", return_value=manager):
                cm._upsert_order_from_execution_report(partial)
                cm._upsert_order_from_execution_report(filled)

            self.assertEqual(1, len(manager.cache["BTCUSDC"]))
            aggregate = manager.cache["BTCUSDC"][0]
            self.assertEqual(2.0, aggregate["quantity"])
            self.assertEqual(110.0, aggregate["price"])
            self.assertEqual("FILLED", aggregate["status"])
            self.assertTrue(manager._account_cache_dirty)

        with self.subTest(msg="rest_fill_is_deduplicated_after_concurrent_ws_update"):
            self._reset_temp()
            partial = {
                **_order(7, 1000),
                "price": 100.0,
                "quantity": 1.0,
                "_fillIds": ["101"],
            }
            manager = self._order_manager([partial], 1000)
            filled_event = {
                "s": "BTCUSDC", "i": 7, "t": 102, "x": "TRADE",
                "X": "FILLED", "S": "BUY", "T": 2000, "E": 2000,
                "p": "100", "q": "2", "l": "1", "L": "120",
                "z": "2", "Z": "220",
            }
            with patch.object(cm, "get_cache_manager", return_value=manager):
                cm._upsert_order_from_execution_report(filled_event)

            manager._persist_items("BTCUSDC", [{
                **_order(7, 2000),
                "price": 120.0,
                "quantity": 1.0,
                "_fillId": "102",
            }])
            aggregate = manager.cache["BTCUSDC"][0]
            self.assertEqual(2.0, aggregate["quantity"])
            self.assertEqual(110.0, aggregate["price"])
            self.assertEqual(["101", "102"], aggregate["_fillIds"])

        with self.subTest(msg="poll_overlap_is_idempotent_by_fill_id"):
            self._reset_temp()
            manager = self._order_manager([], 0)
            fill = {
                **_order(7, 1000),
                "price": 100.0,
                "quantity": 1.0,
                "_fillId": "101",
            }
            manager._persist_items("BTCUSDC", [fill])
            first_snapshot = [dict(item) for item in manager.cache["BTCUSDC"]]
            manager._persist_items("BTCUSDC", [fill])
            self.assertEqual(first_snapshot, manager.cache["BTCUSDC"])

        with self.subTest(msg="ws_reconnect_then_rest_backfill_does_not_double_count"):
            self._reset_temp()
            manager = self._order_manager([], 0)
            later_fill_event = {
                "s": "BTCUSDC", "i": 7, "t": 102, "x": "TRADE",
                "X": "FILLED", "S": "BUY", "T": 2000, "E": 2000,
                "p": "100", "q": "2", "l": "1", "L": "120",
                "z": "2", "Z": "220",
            }
            with patch.object(cm, "get_cache_manager", return_value=manager):
                cm._upsert_order_from_execution_report(later_fill_event)

            manager._persist_items("BTCUSDC", [
                {
                    **_order(7, 1000), "price": 100.0, "quantity": 1.0,
                    "_fillId": "101",
                },
                {
                    **_order(7, 2000), "price": 120.0, "quantity": 1.0,
                    "_fillId": "102",
                },
            ])

            aggregate = manager.cache["BTCUSDC"][0]
            self.assertEqual(2.0, aggregate["quantity"])
            self.assertEqual(110.0, aggregate["price"])
            self.assertEqual({"101", "102"}, set(aggregate["_fillIds"]))

    def test_malformed_rest_rows_abort_the_sync(self):
        trade_manager = self._trade_manager([_trade(1, 1000)], 1000)
        malformed_trade = {**_trade(2, 2000), "qty": "0"}
        with patch(
            "binance_api.bapi_allorders.paginate_my_trades",
            return_value=[malformed_trade],
        ):
            with self.assertRaises(RuntimeError):
                trade_manager.get_remote_items("BTCUSDC", 1001)

        order_manager = self._order_manager([_order(1, 1000)], 1000)
        malformed_fill = {**_trade(2, 2000), "isBuyer": "not-a-boolean"}
        with patch(
            "binance_api.bapi_allorders.paginate_my_trades",
            return_value=[malformed_fill],
        ):
            with self.assertRaises(RuntimeError):
                order_manager.get_remote_items("BTCUSDC", 1001)

    def test_malformed_ws_order_does_not_mutate_cache(self):
        manager = self._order_manager([], 0)
        malformed = {
            "s": "BTCUSDC", "i": 7, "x": "TRADE", "X": "FILLED",
            "S": "UNKNOWN", "T": 1000, "z": "1", "Z": "100",
            "l": "1", "L": "100", "t": 101,
        }
        with patch.object(cm, "get_cache_manager", return_value=manager):
            cm._upsert_order_from_execution_report(malformed)
        self.assertEqual([], manager.cache["BTCUSDC"])
        self.assertFalse(manager._account_cache_dirty)

    def test_trade_rest_fill_is_deduplicated_after_concurrent_ws_append(self):
        manager = self._trade_manager([_trade(1, 1000)], 1000)
        ws_fill = {**_trade(2, 2000), "id": "2"}
        rest_fill = {**_trade(2, 2000), "id": 2}
        with manager.lock:
            manager.cache["BTCUSDC"].append(ws_fill)
            manager._mark_account_cache_dirty_locked()

        manager._persist_items("BTCUSDC", [rest_fill])
        self.assertEqual(
            ["1", "2"],
            [str(item["id"]) for item in manager.cache["BTCUSDC"]],
        )

    def test_account_poll_cursor_uses_request_start_high_water(self):
        manager = self._trade_manager([_trade(1, 1000)], 1000)
        with (
            patch.object(manager, "get_remote_items", return_value=[]),
            patch.object(cm.time, "time", return_value=2.0),
        ):
            self.assertTrue(manager.query_remote_and_update_cache())
        self.assertEqual(1999, manager.fetchtime_time_per_symbol["BTCUSDC"])
        self.assertTrue(manager._account_cache_dirty)

    def test_wrong_symbol_trade_row_is_rejected(self):
        manager = self._trade_manager([_trade(1, 1000)], 1000)
        wrong_symbol = {**_trade(2, 2000), "symbol": "ETHUSDC"}
        with patch(
            "binance_api.bapi_allorders.paginate_my_trades",
            return_value=[wrong_symbol],
        ):
            with self.assertRaises(RuntimeError):
                manager.get_remote_items("BTCUSDC", 1001)

    def test_live_trade_duplicate_requires_same_financial_signature(self):
        first = _trade(2, 2000)
        manager = self._trade_manager([first], 2000)
        equivalent = {
            **first,
            "id": "2", "orderId": "2", "price": 100.0, "qty": "1.00",
        }
        manager._persist_items("BTCUSDC", [equivalent])
        self.assertEqual([first], manager.cache["BTCUSDC"])

        conflicting = {**equivalent, "qty": "2"}
        with self.assertRaises(RuntimeError):
            manager._persist_items("BTCUSDC", [conflicting])
        self.assertEqual([first], manager.cache["BTCUSDC"])

    def test_restart_recovers_previous_generation_after_compaction_crash(self):
        first = _trade(1, 1000)
        writer = self._trade_manager([first], 1000)
        writer.save_state = True
        self.assertTrue(writer.save_state_to_file())
        old_version = writer._persisted_data_version
        backup_path = writer._stage_previous_trade_generation()

        changed = {**first, "price": "200"}
        with cm.atomic_write(writer.filename) as handle:
            handle.write(json.dumps(
                {"s": "BTCUSDC", "i": changed},
                separators=(",", ":"),
            ) + "\n")
        self.assertTrue(os.path.exists(backup_path))

        restarted = cm.CacheTradeManager(
            3600, ["BTCUSDC"], writer.filename, api_client=cm.api
        )
        self.assertEqual([first], restarted.cache["BTCUSDC"])
        self.assertEqual(old_version, restarted._loaded_data_version)
        self.assertFalse(os.path.exists(backup_path))
        restarted.ensure_persisted_version(old_version)

    def test_reader_cannot_restore_previous_generation_during_compaction(self):
        first = _trade(1, 1000)
        writer = self._trade_manager([first], 1000)
        writer.save_state = True
        self.assertTrue(writer.save_state_to_file())
        old_version = writer._persisted_data_version
        changed = {**first, "price": "200"}
        with writer.lock:
            writer.cache["BTCUSDC"] = [changed]
            writer._mark_account_cache_dirty_locked()

        manifest_reached = threading.Event()
        allow_manifest = threading.Event()
        reader_finished = threading.Event()
        compact_result = []
        reader_result = []
        errors = []
        real_write_meta = writer._write_meta

        def paused_write_meta(metadata):
            manifest_reached.set()
            if not allow_manifest.wait(2):
                raise RuntimeError("test did not release manifest publication")
            return real_write_meta(metadata)

        def compact():
            try:
                with patch.object(
                        writer, "_write_meta", side_effect=paused_write_meta):
                    compact_result.append(writer.compact_jsonl())
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        def load_reader():
            try:
                reader_result.append(cm.CacheTradeManager(
                    3600, ["BTCUSDC"], writer.filename, api_client=cm.api
                ))
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)
            finally:
                reader_finished.set()

        compact_thread = threading.Thread(target=compact)
        compact_thread.start()
        self.assertTrue(manifest_reached.wait(2))
        self.assertEqual(old_version, writer._loaded_data_version)

        reader_thread = threading.Thread(target=load_reader)
        reader_thread.start()
        self.assertFalse(
            reader_finished.wait(0.1),
            "reader crossed an in-progress Trade generation publication",
        )

        allow_manifest.set()
        compact_thread.join(2)
        reader_thread.join(2)
        self.assertFalse(compact_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())
        self.assertEqual([], errors)
        self.assertEqual([True], compact_result)
        self.assertEqual(1, len(reader_result))
        self.assertEqual([changed], reader_result[0].cache["BTCUSDC"])
        self.assertNotEqual(
            old_version, reader_result[0]._loaded_data_version
        )
        self.assertFalse(os.path.exists(writer.filename + ".previous"))

    def test_late_legacy_migrator_cannot_replace_winner_or_new_append(self):
        path = os.path.join(self.tmp.name, "migration-race.jsonl")
        legacy_path = path[:-1]
        first = _trade(1, 1000)
        second = _trade(2, 2000)
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump({
                "items": {"BTCUSDC": [first]},
                "fetchtime": {"BTCUSDC": 1000},
            }, handle)

        with patch.object(
                cm.CacheManagerInterface, "load_state", return_value=None):
            winner = cm.CacheTradeManager(
                3600, ["BTCUSDC"], path, api_client=cm.api
            )
            late = cm.CacheTradeManager(
                3600, ["BTCUSDC"], path, api_client=cm.api
            )
        self.assertFalse(os.path.exists(path))

        winner_locked = threading.Event()
        allow_winner = threading.Event()
        late_finished = threading.Event()
        results = []
        errors = []

        def migrate_and_append():
            try:
                with winner._trade_generation_lock():
                    winner_locked.set()
                    if not allow_winner.wait(2):
                        raise RuntimeError("test did not release migration")
                    winner._migrate_legacy_json_generation_locked()
                    with winner.lock:
                        winner.cache["BTCUSDC"].append(second)
                        winner.fetchtime_time_per_symbol["BTCUSDC"] = 2000
                        winner._mark_account_cache_dirty_locked()
                    results.append(
                        winner._save_account_trade_append_generation_locked()
                    )
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        def late_migrate():
            try:
                late._migrate_legacy_json_if_needed()
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)
            finally:
                late_finished.set()

        winner_thread = threading.Thread(target=migrate_and_append)
        winner_thread.start()
        self.assertTrue(winner_locked.wait(2))
        late_thread = threading.Thread(target=late_migrate)
        late_thread.start()
        self.assertFalse(
            late_finished.wait(0.1),
            "late migrator crossed the winner's generation lock",
        )

        allow_winner.set()
        winner_thread.join(2)
        late_thread.join(2)
        self.assertFalse(winner_thread.is_alive())
        self.assertFalse(late_thread.is_alive())
        self.assertEqual([], errors)
        self.assertEqual([True], results)

        restarted = cm.CacheTradeManager(
            3600, ["BTCUSDC"], path, api_client=cm.api
        )
        self.assertEqual(
            [first, second], restarted.cache["BTCUSDC"]
        )
        with open(path + ".meta", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual({"BTCUSDC": 2}, metadata["counts"])

    def test_legacy_trade_data_handling(self):
        with self.subTest(msg="corrupt_legacy_trade_row_fails_closed"):
            self._reset_temp()
            path = os.path.join(self.tmp.name, "corrupt-legacy.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"s": "BTCUSDC", "i": _trade(1, 1000)},
                    separators=(",", ":"),
                ) + "\n")
                handle.write("{broken\n")
            with open(path + ".meta", "w", encoding="utf-8") as handle:
                json.dump({"fetchtime": {"BTCUSDC": 1000}}, handle)

            with self.assertRaises(health.AccountCacheNotReady):
                cm.CacheTradeManager(
                    3600, ["BTCUSDC"], path, api_client=cm.api
                )

        with self.subTest(msg="uncommitted_tail_is_recovered_by_exact_counts"):
            self._reset_temp()
            first = _trade(1, 1000)
            path = os.path.join(self.tmp.name, "legacy-tail.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"s": "BTCUSDC", "i": first},
                    separators=(",", ":"),
                ) + "\n")
                handle.write("{partial")
            with open(path + ".meta", "w", encoding="utf-8") as handle:
                json.dump({
                    "fetchtime": {"BTCUSDC": 1000},
                    "counts": {"BTCUSDC": 1},
                }, handle)

            manager = cm.CacheTradeManager(
                3600, ["BTCUSDC"], path, api_client=cm.api
            )
            self.assertEqual([first], manager.cache["BTCUSDC"])
            self.assertTrue(manager._legacy_jsonl_needs_rewrite)
            manager.save_state = True
            self.assertTrue(manager.save_state_to_file())
            self.assertEqual(os.path.getsize(path), manager._persisted_committed_bytes)

        with self.subTest(msg="equivalent_duplicate_ids_collapse"):
            self._reset_temp()
            path = os.path.join(self.tmp.name, "legacy-duplicates.jsonl")
            first = {
                **_trade("2", 2000),
                "orderId": "2", "price": "100.0", "qty": "1.00",
            }
            equivalent = {
                **_trade(2, 2000),
                "orderId": 2, "price": 100, "qty": 1,
            }
            with open(path, "w", encoding="utf-8") as handle:
                for item in (first, equivalent):
                    handle.write(json.dumps(
                        {"s": "BTCUSDC", "i": item},
                        separators=(",", ":"),
                    ) + "\n")
            with open(path + ".meta", "w", encoding="utf-8") as handle:
                json.dump({
                    "fetchtime": {"BTCUSDC": 2000},
                    "counts": {"BTCUSDC": 2},
                }, handle)

            manager = cm.CacheTradeManager(
                3600, ["BTCUSDC"], path, api_client=cm.api
            )
            self.assertEqual([first], manager.cache["BTCUSDC"])
            self.assertTrue(manager._legacy_jsonl_needs_rewrite)
            manager.save_state = True
            self.assertTrue(manager.save_state_to_file())
            with open(path + ".meta", encoding="utf-8") as handle:
                self.assertEqual(
                    {"BTCUSDC": 1}, json.load(handle)["counts"]
                )

        with self.subTest(msg="conflicting_duplicate_id_fails_closed"):
            self._reset_temp()
            path = os.path.join(self.tmp.name, "legacy-conflict.jsonl")
            first = _trade(2, 2000)
            conflicting = {**first, "qty": "2"}
            with open(path, "w", encoding="utf-8") as handle:
                for item in (first, conflicting):
                    handle.write(json.dumps(
                        {"s": "BTCUSDC", "i": item},
                        separators=(",", ":"),
                    ) + "\n")
            with open(path + ".meta", "w", encoding="utf-8") as handle:
                json.dump({
                    "fetchtime": {"BTCUSDC": 2000},
                    "counts": {"BTCUSDC": 1},
                }, handle)

            with self.assertRaises(health.AccountCacheNotReady):
                cm.CacheTradeManager(
                    3600, ["BTCUSDC"], path, api_client=cm.api
                )

        with self.subTest(msg="semantically_invalid_legacy_tail_is_not_recoverable"):
            self._reset_temp()
            path = os.path.join(self.tmp.name, "semantic-tail.jsonl")
            valid = _trade(1, 1000)
            invalid = {**_trade(2, 2000), "qty": "0"}
            with open(path, "w", encoding="utf-8") as handle:
                for item in (valid, invalid):
                    handle.write(json.dumps(
                        {"s": "BTCUSDC", "i": item},
                        separators=(",", ":"),
                    ) + "\n")
            with open(path + ".meta", "w", encoding="utf-8") as handle:
                json.dump({
                    "fetchtime": {"BTCUSDC": 1000},
                    "counts": {"BTCUSDC": 1},
                }, handle)

            with self.assertRaises(health.AccountCacheNotReady):
                cm.CacheTradeManager(
                    3600, ["BTCUSDC"], path, api_client=cm.api
                )

        with self.subTest(msg="invalid_legacy_full_json_migration_fails_closed"):
            self._reset_temp()
            jsonl_path = os.path.join(self.tmp.name, "legacy-trades.jsonl")
            legacy_path = jsonl_path[:-1]
            valid = _trade(1, 1000)
            invalid = {**_trade(2, 2000), "qty": "0"}
            with open(legacy_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "items": {"BTCUSDC": [valid, invalid]},
                    "fetchtime": {"BTCUSDC": 9_999_999},
                }, handle)
            with open(legacy_path, "rb") as handle:
                legacy_before = handle.read()

            with patch.object(
                    cm.account_cache_health, "record_persisted_version") as publish:
                with self.assertRaisesRegex(
                        RuntimeError, "Cannot migrate legacy cache"):
                    cm.CacheTradeManager(
                        3600, ["BTCUSDC"], jsonl_path, api_client=cm.api
                    )

            publish.assert_not_called()
            with open(legacy_path, "rb") as handle:
                self.assertEqual(legacy_before, handle.read())
            self.assertFalse(os.path.exists(jsonl_path))
            self.assertFalse(os.path.exists(jsonl_path + ".meta"))

    def test_legacy_order_cutoff_survives_new_fill_and_restart(self):
        legacy = {
            **_order(7, 2000), "price": 100.0, "quantity": 2.0,
        }
        manager = self._order_manager([legacy], 2000)
        manager._persist_items("BTCUSDC", [{
            **_order(7, 3000), "price": 130.0, "quantity": 1.0,
            "_fillId": "103",
        }])
        aggregate = manager.cache["BTCUSDC"][0]
        self.assertEqual(3.0, aggregate["quantity"])
        self.assertEqual(110.0, aggregate["price"])
        self.assertEqual(2000, aggregate["_legacyCoveredThrough"])
        manager.save_state = True
        self.assertTrue(manager.save_state_to_file())

        reloaded = cm.CacheOrderManager(
            3600, ["BTCUSDC"], manager.filename, api_client=cm.api
        )
        reloaded._persist_items("BTCUSDC", [
            {
                **_order(7, 1000), "price": 80.0, "quantity": 1.0,
                "_fillId": "101",
            },
            {
                **_order(7, 2000), "price": 120.0, "quantity": 1.0,
                "_fillId": "102",
            },
            {
                **_order(7, 3000), "price": 130.0, "quantity": 1.0,
                "_fillId": "103",
            },
        ])
        aggregate = reloaded.cache["BTCUSDC"][0]
        self.assertEqual(3.0, aggregate["quantity"])
        self.assertEqual(110.0, aggregate["price"])
        self.assertEqual(["103"], aggregate["_fillIds"])

    def test_rest_trade_id_zero_is_a_valid_immutable_fill(self):
        first_trade = {**_trade(0, 1000), "orderId": 1}
        manager = self._trade_manager([first_trade], 1000)

        self.assertTrue(manager._is_valid_trade(first_trade))
        self.assertEqual([first_trade], manager.cache["BTCUSDC"])

    def test_websocket_trade_id_zero_updates_both_account_caches(self):
        order_manager = self._order_manager([], 0)
        trade_manager = self._trade_manager([_trade(1, 500)], 500)
        with trade_manager.lock:
            trade_manager.cache["BTCUSDC"] = []
        event = {
            "e": "executionReport",
            "x": "TRADE",
            "X": "FILLED",
            "s": "BTCUSDC",
            "S": "BUY",
            "i": 99,
            "t": 0,
            "l": "0.1",
            "L": "100",
            "q": "0.1",
            "p": "100",
            "T": 1000,
            "E": 1000,
        }

        def manager_for(cache_name, *args, **kwargs):
            if cache_name == "Order":
                return order_manager
            if cache_name == "Trade":
                return trade_manager
            raise AssertionError(f"unexpected cache lookup: {cache_name}")

        with patch.object(cm, "get_cache_manager", side_effect=manager_for):
            cm._upsert_order_from_execution_report(event)
            cm._append_trade_from_execution_report(event)

        self.assertEqual(
            ["0"], order_manager.cache["BTCUSDC"][0]["_fillIds"]
        )
        self.assertEqual(
            "0", str(trade_manager.cache["BTCUSDC"][0]["id"])
        )

    def test_multi_symbol_health_time_uses_earliest_request_start(self):
        from types import SimpleNamespace

        manager = self._trade_manager([_trade(1, 1000)], 1000)
        manager.symbols = ["BTCUSDC", "ETHUSDC"]
        manager.fetchtime_time_per_symbol["ETHUSDC"] = 1000
        clock = SimpleNamespace(now=2.0)

        def fetch(*_args, **_kwargs):
            clock.now = 13.0
            return []

        with (
            patch.object(manager, "get_remote_items", side_effect=fetch),
            # Replace only this module's clock, not the logging module's time source.
            patch.object(cm, "time", SimpleNamespace(time=lambda: clock.now)),
        ):
            self.assertTrue(manager.query_remote_and_update_cache())

        self.assertEqual(2000, manager._last_complete_sync_at_ms)
        self.assertEqual(1999, manager.fetchtime_time_per_symbol["BTCUSDC"])
        self.assertEqual(12999, manager.fetchtime_time_per_symbol["ETHUSDC"])

        marker = os.path.join(self.tmp.name, "slow-cycle-health.json")
        with patch.object(
                health, "_pid_started_at_ms", return_value=500):
            health.enable_writer(path=marker, pid=123, now_ms=1000)
        try:
            self.assertTrue(health.record_successful_sync(
                "Order", "order-v1",
                now_ms=manager._last_complete_sync_at_ms))
            self.assertTrue(health.record_successful_sync(
                "Trade", "trade-v1",
                now_ms=manager._last_complete_sync_at_ms))
            status = health.inspect_health(
                path=marker, now_ms=13000, max_age_sec=10,
                pid_is_alive=lambda _pid: True,
                pid_started_at_ms=lambda _pid: 500,
            )
            self.assertFalse(status.ready)
            self.assertEqual("order_cache_stale", status.reason)
        finally:
            health.disable_writer(now_ms=13000)

    def test_failed_multi_symbol_cycle_preserves_last_health_marker(self):
        manager = self._trade_manager([_trade(1, 1000)], 1000)
        manager.symbols = ["BTCUSDC", "ETHUSDC"]
        manager.fetchtime_time_per_symbol["ETHUSDC"] = 1000
        manager._last_complete_sync_at_ms = 2000
        marker = os.path.join(self.tmp.name, "failed-cycle-health.json")
        with patch.object(
                health, "_pid_started_at_ms", return_value=500):
            health.enable_writer(path=marker, pid=123, now_ms=1000)
        health.record_successful_sync("Order", "order-v1", now_ms=2000)
        health.record_successful_sync("Trade", "trade-v1", now_ms=2000)
        with open(marker, "rb") as handle:
            marker_before = handle.read()
        try:
            with (
                patch.object(
                    manager, "get_remote_items",
                    side_effect=[[], RuntimeError("second symbol failed")]),
                # The cycle reads time.time() more than once per symbol (the request
                # high-water plus the empty-result log's timestamp formatter). Values
                # are irrelevant here: the cycle raises on the second symbol, so this
                # only needs to never exhaust before that RuntimeError.
                patch.object(cm.time, "time", side_effect=lambda: 3.0),
            ):
                with self.assertRaisesRegex(RuntimeError, "second symbol failed"):
                    manager.query_remote_and_update_cache()
            self.assertEqual(2000, manager._last_complete_sync_at_ms)
            with open(marker, "rb") as handle:
                self.assertEqual(marker_before, handle.read())
        finally:
            health.disable_writer(now_ms=4000)

    def test_main_claims_single_writer_before_health_and_cache_touch(self):
        with open(cm.__file__, encoding="utf-8") as handle:
            source = handle.read()
        main = source.split('if __name__ == "__main__":', 1)[1]
        lock_position = main.index('single_instance("cacheManager")')
        health_position = main.index("account_cache_health.enable_writer()")
        cache_position = main.index('get_cache_manager(_account_cache_name')
        self.assertLess(lock_position, health_position)
        self.assertLess(lock_position, cache_position)
    def test_invalid_legacy_order_cannot_receive_a_durable_version(self):
        invalid = {**_order(1, 1000), "side": "UNKNOWN"}
        manager = self._order_manager([invalid], 1000)
        manager.save_state = True
        with patch.object(
            cm.account_cache_health, "record_persisted_version"
        ) as publish:
            self.assertFalse(manager.save_state_to_file())
        publish.assert_not_called()
        self.assertEqual("", manager._persisted_data_version)

    def test_duplicate_order_aggregate_cannot_receive_durable_version(self):
        first = _order(7, 1000)
        duplicate = {**_order(7, 2000), "price": "110"}
        manager = self._order_manager([first, duplicate], 2000)
        manager.save_state = True
        with patch.object(
                cm.account_cache_health, "record_persisted_version") as publish:
            self.assertFalse(manager.save_state_to_file())

        publish.assert_not_called()
        self.assertEqual("", manager._persisted_data_version)

    def test_fill_id_cannot_belong_to_two_order_aggregates(self):
        first = {
            **_order(7, 1000),
            "_fillIds": ["101"],
        }
        conflicting = {
            **_order(8, 2000),
            "_fillIds": ["101"],
        }
        manager = self._order_manager([first, conflicting], 2000)
        manager.save_state = True
        with patch.object(
                cm.account_cache_health, "record_persisted_version") as publish:
            self.assertFalse(manager.save_state_to_file())

        publish.assert_not_called()
        self.assertEqual("", manager._persisted_data_version)

    def test_running_writer_cannot_be_demoted_by_reader_style_start(self):
        manager = self._order_manager([_order(1, 1000)], 1000)
        manager._first_sleep = True
        try:
            writer_thread = manager.periodic_sync(3600, True)
            reader_thread = manager.periodic_sync(3600, False)
            self.assertIs(writer_thread, reader_thread)
            self.assertTrue(manager.save_state)
        finally:
            manager.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
