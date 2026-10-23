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


if __name__ == "__main__":
    unittest.main()
