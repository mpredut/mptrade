from __future__ import annotations

import json
import inspect
import os
import tempfile
import unittest
from unittest import mock

import cacheManager as cm
import tradeall_price_archiver as archiver


class _Current:
    def __init__(self, events):
        self.events = events

    def unsubscribe_price(self, cache):
        self.events.append(("unsubscribe", cache))

    def shutdown(self):
        self.events.append(("current_shutdown", None))


class _Cache:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append(("cache_shutdown", self))

    def save_state_to_file(self):
        self.events.append(("flush", self))


class _WsManager:
    def __init__(self, events):
        self.events = events

    def stop(self):
        self.events.append(("ws_stop", None))


class _WsModule:
    def __init__(self, events):
        self.bapi_ws_manager = _WsManager(events)


class TestArchiverLifecycle(unittest.TestCase):
    def test_main_never_starts_the_private_account_stream(self):
        source = inspect.getsource(archiver.main)
        self.assertNotIn("enable_real_ws_event_sync", source)

    def test_symbols_are_normalized_deduplicated_and_validated(self):
        self.assertEqual(archiver._symbols(" btcusdc,TAOUSDC,BTCUSDC "),
                         ["BTCUSDC", "TAOUSDC"])
        with self.assertRaises(ValueError):
            archiver._symbols("../escape")
        with self.assertRaises(ValueError):
            archiver._symbols("")

    def test_shutdown_unsubscribes_then_stops_and_flushes_before_ws_close(self):
        events = []
        cache = _Cache(events)
        archiver._shutdown([cache], _Current(events), _WsModule(events))
        self.assertEqual([name for name, _ in events], [
            "unsubscribe", "cache_shutdown", "flush", "current_shutdown", "ws_stop"])

    def test_jsonl_restart_loads_existing_history_and_appends_only_new_tick(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"s": "BTCUSDC", "i": [1000, 10.0]}) + "\n")
            manager = cm.Cache365PriceManager(60, ["BTCUSDC"], path)
            manager.KEEP_HOURS = 10**9
            manager.on_price_update("BTCUSDC", 2000, 11.0)
            manager.save_state_to_file()
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(rows, [
                {"s": "BTCUSDC", "i": [1000, 10.0]},
                {"s": "BTCUSDC", "i": [2000, 11.0]},
            ])
        finally:
            for suffix in ("", ".meta"):
                try:
                    os.remove(path + suffix)
                except OSError:
                    pass

    def test_long_archive_trimming_never_compacts_from_the_memory_tail(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"s": "BTCUSDC", "i": [1, 10.0]}) + "\n")
            manager = cm.Cache365PriceManager(60, ["BTCUSDC"], path)
            manager.KEEP_HOURS = 1
            manager.cache = {"BTCUSDC": [[1, 10.0]]}
            manager._last_long_trim = 100.0
            with mock.patch("cacheManager.time.monotonic", side_effect=[101.0, 100.0 + 86401]), \
                    mock.patch.object(manager, "compact_jsonl") as compact:
                manager._trim_old_data("BTCUSDC")
                compact.assert_not_called()
                manager._trim_old_data("BTCUSDC")
                compact.assert_not_called()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_long_archive_memory_is_bounded_without_truncating_disk_history(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        total = cm.CM_LONG_ARCHIVE_MEMORY_ROWS + 25
        try:
            with open(path, "w", encoding="utf-8") as handle:
                for index in range(total):
                    handle.write(json.dumps({"s": "BTCUSDC", "i": [index + 1, 10.0]}) + "\n")
            manager = cm.Cache365PriceManager(60, ["BTCUSDC"], path)
            manager.KEEP_HOURS = 10**9
            self.assertEqual(len(manager.cache["BTCUSDC"]), cm.CM_LONG_ARCHIVE_MEMORY_ROWS)
            manager.on_price_update("BTCUSDC", total + 1, 11.0)
            manager.save_state_to_file()
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), total + 1)
        finally:
            for suffix in ("", ".meta"):
                try:
                    os.remove(path + suffix)
                except OSError:
                    pass

    def test_legacy_json_migration_succeeds_for_cache365_price_manager(self):
        tmp_dir = tempfile.mkdtemp()
        legacy_json = os.path.join(tmp_dir, "cache_365_price_BTCUSDC.json")
        target_jsonl = os.path.join(tmp_dir, "cache_365_price_BTCUSDC.jsonl")
        try:
            with open(legacy_json, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "items": {"BTCUSDC": [[1000, 10.0], [2000, 11.0]]},
                        "fetchtime": {"BTCUSDC": 2000},
                    },
                    handle,
                )
            manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
            self.assertTrue(os.path.exists(target_jsonl))
            self.assertEqual(manager.cache.get("BTCUSDC"), [[1000, 10.0], [2000, 11.0]])
            with open(target_jsonl, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(
                rows,
                [
                    {"s": "BTCUSDC", "i": [1000, 10.0]},
                    {"s": "BTCUSDC", "i": [2000, 11.0]},
                ],
            )
        finally:
            for p in (legacy_json, target_jsonl, target_jsonl + ".meta"):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def test_cache365_rotation_failure_restores_active_file_and_memory(self):
        tmp_dir = tempfile.mkdtemp()
        target_jsonl = os.path.join(tmp_dir, "cache_365_price_BTCUSDC.jsonl")
        try:
            now_ms = int(cm.time.time() * 1000)
            with open(target_jsonl, "w", encoding="utf-8") as handle:
                for i in range(20):
                    handle.write(
                        json.dumps({"s": "BTCUSDC", "i": [now_ms + i, 10.0 + i]}) + "\n"
                    )
            manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
            manager.save_state = True
            manager.MAX_FILE_BYTES = 1
            manager.ROTATE_KEEP_FRACTION = 0.25
            with mock.patch.object(manager, "_write_meta", return_value=False):
                self.assertFalse(manager.maintain_append_persist())
            self.assertTrue(os.path.exists(target_jsonl))
            with open(target_jsonl, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(manager.cache["BTCUSDC"]), 20)
            self.assertEqual(manager._persisted_counts.get("BTCUSDC"), 20)
            self.assertEqual(
                [f for f in os.listdir(tmp_dir) if ".archive" in f],
                [],
            )
        finally:
            for name in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, name))
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def test_cache365_maintenance_preserves_unpersisted_memory_tail_and_fetchtime(self):
        tmp_dir = tempfile.mkdtemp()
        target_jsonl = os.path.join(tmp_dir, "cache_365_price_BTCUSDC.jsonl")
        try:
            now_ms = int(cm.time.time() * 1000)
            with open(target_jsonl, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"s": "BTCUSDC", "i": [now_ms - 2000, 10.0]}) + "\n"
                )
            manager = cm.Cache365PriceManager(60, ["BTCUSDC"], target_jsonl)
            manager.save_state = True
            manager.KEEP_HOURS = 10**9
            manager.on_price_update("BTCUSDC", now_ms, 12.5)
            self.assertTrue(manager.maintain_append_persist())
            self.assertEqual(
                manager.cache.get("BTCUSDC"),
                [[now_ms - 2000, 10.0], [now_ms, 12.5]],
            )
            self.assertEqual(manager._persisted_counts.get("BTCUSDC"), 2)
            self.assertGreaterEqual(manager.fetchtime_per_symbol.get("BTCUSDC", 0), now_ms)
            with open(target_jsonl, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(
                rows,
                [
                    {"s": "BTCUSDC", "i": [now_ms - 2000, 10.0]},
                    {"s": "BTCUSDC", "i": [now_ms, 12.5]},
                ],
            )
            # Direct _rotate_disk_archive preserves unpersisted in-memory tail items
            manager.on_price_update("BTCUSDC", now_ms + 1000, 15.0)
            self.assertTrue(manager._rotate_disk_archive())
            self.assertEqual(manager.cache.get("BTCUSDC")[-1], [now_ms + 1000, 15.0])
            self.assertEqual(manager._persisted_counts.get("BTCUSDC"), len(manager.cache.get("BTCUSDC", [])))
            # Non-dict .meta does not crash _load_jsonl and rebuilds fetchtimes from loaded tail
            with open(target_jsonl + ".meta", "w", encoding="utf-8") as handle:
                handle.write("[]")
            manager._load_jsonl()
            self.assertGreaterEqual(manager.fetchtime_per_symbol.get("BTCUSDC", 0), now_ms + 1000)
            self.assertEqual(cm.Cache365PriceManager._tail_lines(target_jsonl, True), [])
            with open(target_jsonl, "a", encoding="utf-8") as handle:
                handle.write("\n\n")
            tail_rows = cm.Cache365PriceManager._tail_lines(target_jsonl, 1, block_size=7)
            self.assertEqual(len(tail_rows), 1)
            self.assertEqual(json.loads(tail_rows[0])["i"], [now_ms + 1000, 15.0])
        finally:
            for name in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, name))
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
