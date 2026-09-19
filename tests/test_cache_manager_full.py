"""Comprehensive tests for cacheManager.py, a project-wide critical module.

Coverage:
  - CacheManagerInterface (via ConcreteTestManager)
  - CacheTradeManager
  - CacheOrderManager
  - CacheSparsePriceManager (formerly CachePriceManager)
  - Cache24PriceManager
  - CachePriceLongTrendManager
  - CacheAssetValueManager
  - CacheCurrentPriceManager
  - CacheFactory / get_cache_manager
  - get_current_price_manager (singleton)
  - WebSocket health functions
"""
import gzip, os, sys, json, time, tempfile, threading, unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

# -- Mock bapi before any imports. ---------------------------------------------
mock_api = MagicMock()
mock_api.get_current_price = MagicMock(return_value=50000.0)
mock_api.client = MagicMock()
mock_api.client.get_symbol_ticker = MagicMock(return_value={"price": "50000.0"})
sys.modules.setdefault("bapi", mock_api)
sys.modules.setdefault("bapi_trades", MagicMock())
sys.modules.setdefault("bapi_allorders", MagicMock())

import cacheManager as cm


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete helper for testing the abstract interface.
# ═══════════════════════════════════════════════════════════════════════════════
class ConcreteTestManager(cm.CacheManagerInterface):
    """Provide a minimal CacheManagerInterface implementation for tests."""
    def __init__(self, sync_ts, symbols, filename, append_mode=True, api_client=None,
                 remote_items=None, append_persist=False):
        self._remote_items = remote_items or {}   # {symbol: [items]}
        super().__init__(sync_ts, symbols, filename, append_mode=append_mode,
                         api_client=api_client or MagicMock(), append_persist=append_persist)

    def rebuild_fetchtime_times(self):
        return {}

    def get_remote_items(self, symbol, startTime):
        return self._remote_items.get(symbol, [])


def _tmp_file(tmp_dir, name="cache_test.json"):
    return os.path.join(tmp_dir, name)


def _write_cache_file(fname, items_dict, fetchtime_dict=None):
    with open(fname, "w") as f:
        json.dump({"items": items_dict, "fetchtime": fetchtime_dict or {}}, f)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CacheManagerInterface
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheManagerInterface(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    # ── load_state ────────────────────────────────────────────────────────────

    def test_load_state_behaviors(self):
        with self.subTest("load from existing file"):
            fname = _tmp_file(self.tmp)
            _write_cache_file(fname, {"SYM": [[1000, 50.0]]}, {"SYM": 1000})
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            with mgr.lock:
                self.assertEqual(mgr.cache["SYM"], [[1000, 50.0]])

        with self.subTest("missing cache uses default start time"):
            fname = _tmp_file(self.tmp, "missing.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items={"SYM": []})
            mgr.cache = {}
            mgr.fetchtime_time_per_symbol = {}
            mgr.fallback_time_default = 123_456
            with patch.object(cm.os.path, "exists", return_value=False):
                result = mgr._CacheManagerInterface__rebuild_fetchtime_times()
            self.assertEqual(result, {"SYM": 123_456})

        with self.subTest("missing file calls remote"):
            fname = _tmp_file(self.tmp, "nonexistent.json")
            remote = {"SYM": [[int(time.time()*1000), 100.0]]}
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items=remote)
            with mgr.lock:
                self.assertIn("SYM", mgr.cache)

        with self.subTest("corrupt file calls remote"):
            fname = _tmp_file(self.tmp)
            with open(fname, "w") as f:
                f.write("NOT_JSON{{{")
            remote = {"SYM": [[int(time.time()*1000), 77.0]]}
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items=remote)
            with mgr.lock:
                prices = [e[1] for e in mgr.cache.get("SYM", [])]
            self.assertIn(77.0, prices)

    # ── save_state_to_file_if_enabled ─────────────────────────────────────────

    def test_save_state_behaviors(self):
        with self.subTest("save disabled by default"):
            fname = _tmp_file(self.tmp, "no_save.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            mgr.cache["SYM"] = [[999, 1.0]]
            mgr.save_state_to_file_if_enabled()
            self.assertFalse(os.path.exists(fname))

        with self.subTest("save enabled writes file"):
            fname = _tmp_file(self.tmp, "save_test.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            mgr.enable_save_state_to_file()
            mgr.cache["SYM"] = [[999, 42.0]]
            mgr.save_state_to_file_if_enabled()
            self.assertTrue(os.path.exists(fname))
            with open(fname) as f:
                data = json.load(f)
            self.assertEqual(data["items"]["SYM"], [[999, 42.0]])

        with self.subTest("save uses tmp then replace"):
            fname = _tmp_file(self.tmp, "atomic.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            mgr.enable_save_state_to_file()
            mgr.cache["SYM"] = [[1, 2.0]]
            mgr.save_state_to_file_if_enabled()
            self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname + ".tmp"))

    # ── update_cache_per_symbol ───────────────────────────────────────────────

    def test_update_cache_per_symbol_behaviors(self):
        with self.subTest("append mode extends"):
            fname = _tmp_file(self.tmp, "append.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=True)
            mgr.cache["SYM"] = [[1, 10.0]]
            mgr.update_cache_per_symbol("SYM", [[2, 20.0]])
            with mgr.lock:
                self.assertEqual(len(mgr.cache["SYM"]), 2)
                self.assertEqual(mgr.cache["SYM"][1][1], 20.0)

        with self.subTest("snapshot mode replaces"):
            fname = _tmp_file(self.tmp, "snap.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=False)
            mgr.cache["SYM"] = [[1, 10.0]]
            mgr.update_cache_per_symbol("SYM", [[2, 20.0]])
            with mgr.lock:
                self.assertEqual(mgr.cache["SYM"], [[2, 20.0]])

        with self.subTest("creates symbol if missing"):
            fname = _tmp_file(self.tmp, "new_sym.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=True)
            mgr.update_cache_per_symbol("SYM", [[1, 5.0]])
            with mgr.lock:
                self.assertIn("SYM", mgr.cache)

        with self.subTest("deduplicates in append mode"):
            fname = _tmp_file(self.tmp, "dedup.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=True)
            mgr.cache["SYM"] = [[1, 10.0]]
            mgr.update_cache_per_symbol("SYM", [[1, 10.0]])
            with mgr.lock:
                self.assertEqual(len(mgr.cache["SYM"]), 1)

        with self.subTest("sets fetchtime"):
            fname = _tmp_file(self.tmp, "ft.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=True)
            mgr.update_cache_per_symbol("SYM", [[int(time.time()*1000), 1.0]])
            self.assertIn("SYM", mgr.fetchtime_time_per_symbol)

    # ── filter_new_items ──────────────────────────────────────────────────────

    def test_filter_new_items_behaviors(self):
        with self.subTest("removes duplicates"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            existing = [[1, 10.0], [2, 20.0]]
            new = [[2, 20.0], [3, 30.0]]
            self.assertEqual(mgr.filter_new_items(existing, new), [[3, 30.0]])

        with self.subTest("all new"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            self.assertEqual(len(mgr.filter_new_items([], [[1, 1.0], [2, 2.0]])), 2)

    def test_query_remote_and_update_behaviors(self):
        with self.subTest("fetches and stores"):
            fname = _tmp_file(self.tmp)
            ts = int(time.time() * 1000)
            remote = {"SYM": [[ts, 99.0]]}
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items=remote)
            mgr.cache = {}
            mgr.fetchtime_time_per_symbol = {}
            mgr.query_remote_and_update_cache()
            with mgr.lock:
                self.assertIn(99.0, [e[1] for e in mgr.cache.get("SYM", [])])

        with self.subTest("skips empty"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items={"SYM": []})
            mgr.cache = {}
            mgr.fetchtime_time_per_symbol = {}
            mgr.query_remote_and_update_cache()
            with mgr.lock:
                self.assertEqual(mgr.cache.get("SYM", []), [])

        with self.subTest("continues on empty symbol"):
            fname = _tmp_file(self.tmp)
            ts = int(time.time() * 1000)
            remote = {"SYM1": [], "SYM2": [[ts, 5.0]]}
            mgr = ConcreteTestManager(9999, ["SYM1", "SYM2"], fname, remote_items=remote)
            mgr.cache = {}
            mgr.fetchtime_time_per_symbol = {}
            mgr.query_remote_and_update_cache()
            with mgr.lock:
                self.assertIn(5.0, [e[1] for e in mgr.cache.get("SYM2", [])])
                
    def test_on_items_update_stores_entry(self):
        fname = _tmp_file(self.tmp)
        mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=True, remote_items={"SYM": []})
        mgr.on_items_update("SYM", [[int(time.time()*1000), 123.0]])
        with mgr.lock:
            self.assertIn(123.0, [e[1] for e in mgr.cache.get("SYM", [])])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cache persistence durability
# ═══════════════════════════════════════════════════════════════════════════════
class TestCachePersistenceDurability(unittest.TestCase):
    """Keep durable rows, cursors, and metadata on one coherent snapshot."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_manager(self, *, append_persist):
        suffix = "jsonl" if append_persist else "json"
        manager = ConcreteTestManager(
            9999,
            ["SYM"],
            _tmp_file(self.tmp.name, f"durable.{suffix}"),
            append_mode=True,
            append_persist=append_persist,
            remote_items={"SYM": []},
        )
        manager.cache = {"SYM": [[1000, 1.0]]}
        manager.fetchtime_time_per_symbol = {"SYM": 1000}
        manager._persisted_counts = {}
        return manager

    @staticmethod
    def _read_jsonl(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_partial_jsonl_write_rolls_back_cursor_and_retries_all_rows(self):
        manager = self._make_manager(append_persist=True)
        manager.cache = {
            "AAA": [[1000, 1.0]],
            "BBB": [[2000, 2.0]],
        }
        manager.fetchtime_time_per_symbol = {"AAA": 1000, "BBB": 2000}
        real_dumps = cm.json.dumps
        calls = 0

        def fail_second_dump(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated partial write failure")
            return real_dumps(*args, **kwargs)

        with patch.object(cm.json, "dumps", side_effect=fail_second_dump):
            self.assertFalse(manager.save_state_to_file())

        self.assertEqual(manager._persisted_counts, {})
        self.assertEqual(os.path.getsize(manager.filename), 0)
        self.assertTrue(manager.save_state_to_file())
        rows = self._read_jsonl(manager.filename)
        self.assertEqual([row["s"] for row in rows], ["AAA", "BBB"])
        self.assertEqual(manager._persisted_counts, {"AAA": 1, "BBB": 1})

    def test_jsonl_fsync_failure_rolls_back_and_retry_persists_every_row(self):
        manager = self._make_manager(append_persist=True)

        with patch.object(
                cm.os, "fsync", side_effect=[OSError("simulated fsync failure"), None]):
            self.assertFalse(manager.save_state_to_file())

        self.assertEqual(manager._persisted_counts, {})
        self.assertEqual(os.path.getsize(manager.filename), 0)
        self.assertTrue(manager.save_state_to_file())
        rows = self._read_jsonl(manager.filename)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["i"], [1000, 1.0])

    def test_metadata_failure_does_not_duplicate_durable_jsonl_rows_on_retry(self):
        manager = self._make_manager(append_persist=True)

        with patch.object(manager, "_write_meta", return_value=False):
            self.assertFalse(manager.save_state_to_file())

        self.assertEqual(manager._persisted_counts, {"SYM": 1})
        self.assertEqual(len(self._read_jsonl(manager.filename)), 1)
        self.assertTrue(manager.save_state_to_file())
        self.assertEqual(len(self._read_jsonl(manager.filename)), 1)
        with open(manager.filename + ".meta", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["counts"], {"SYM": 1})

    def _save_while_update_attempts_to_cross_metadata(self, manager):
        real_atomic_write_json = cm.atomic_write_json
        mutation_started = threading.Event()
        mutation_finished = threading.Event()
        mutation_threads = []
        blocked_during_metadata = []

        def mutate_cache():
            mutation_started.set()
            manager.on_items_update("SYM", [[2000, 2.0]])
            mutation_finished.set()

        def observe_metadata_write(path, payload, indent=None):
            if path == manager.filename + ".meta":
                mutation_thread = threading.Thread(target=mutate_cache)
                mutation_threads.append(mutation_thread)
                mutation_thread.start()
                self.assertTrue(mutation_started.wait(1))
                blocked_during_metadata.append(not mutation_finished.wait(0.1))
            return real_atomic_write_json(path, payload, indent=indent)

        with patch.object(
                cm, "atomic_write_json", side_effect=observe_metadata_write):
            self.assertTrue(manager.save_state_to_file())

        self.assertEqual(blocked_during_metadata, [True])
        for mutation_thread in mutation_threads:
            mutation_thread.join(timeout=2)
            self.assertFalse(mutation_thread.is_alive())
        self.assertTrue(mutation_finished.is_set())

    def test_jsonl_metadata_cannot_advance_past_durable_rows(self):
        manager = self._make_manager(append_persist=True)
        self._save_while_update_attempts_to_cross_metadata(manager)

        rows = self._read_jsonl(manager.filename)
        with open(manager.filename + ".meta", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(len(rows), 1)
        self.assertEqual(metadata["counts"], {"SYM": 1})
        self.assertEqual(metadata["fetchtime"], {"SYM": 1000})
        self.assertEqual(len(manager.cache["SYM"]), 2)

    def test_full_snapshot_metadata_cannot_advance_past_durable_data(self):
        manager = self._make_manager(append_persist=False)
        self._save_while_update_attempts_to_cross_metadata(manager)

        with open(manager.filename, encoding="utf-8") as handle:
            data = json.load(handle)
        with open(manager.filename + ".meta", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(data["items"], {"SYM": [[1000, 1.0]]})
        self.assertEqual(data["fetchtime"], metadata["fetchtime"])
        self.assertEqual(metadata["counts"], {"SYM": 1})
        self.assertEqual(manager.cache["SYM"], [[1000, 1.0], [2000, 2.0]])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CacheTradeManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheTradeManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self):
        api_mock = MagicMock()
        api_mock.client.get_my_trades.return_value = []
        fname = _tmp_file(self.tmp, "cache_trade.json")
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]):
            return cm.CacheTradeManager(9999, ["BTC"], fname, api_client=api_mock)

    def test_trade_manager_behaviors(self):
        with self.subTest(msg="trade_validation"):
            mgr = self._make()
            valid = {'symbol': 'BTC', 'id': 1, 'orderId': 2, 'price': '100',
                     'qty': '1', 'time': 123, 'isBuyer': True}
            invalid = {'symbol': 'BTC', 'id': 1}
            for label, trade, expected in (("complete", valid, True), ("missing-key", invalid, False)):
                with self.subTest(case=label):
                    self.assertEqual(mgr._is_valid_trade(trade), expected)

        with self.subTest(msg="rebuild_fetchtime_returns_none"):
            mgr = self._make()
            self.assertIsNone(mgr.rebuild_fetchtime_times())

        with self.subTest(msg="append_mode_true"):
            mgr = self._make()
            self.assertTrue(mgr.append_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CacheOrderManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheOrderManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self):
        api_mock = MagicMock()
        fname = _tmp_file(self.tmp, "cache_order.json")
        with patch("binance_api.bapi_allorders.get_filled_orders", return_value=[]):
            return cm.CacheOrderManager(9999, ["BTC"], fname, api_client=api_mock)

    def test_order_manager_behaviors(self):
        with self.subTest(msg="order_validation"):
            mgr = self._make()
            valid = {'orderId': 1, 'price': '100', 'quantity': '1',
                     'timestamp': 123, 'side': 'BUY'}
            for label, order, expected in (
                ("complete", valid, True), ("missing-key", {'orderId': 1}, False)
            ):
                with self.subTest(case=label):
                    self.assertEqual(mgr._is_valid_trade(order), expected)

        with self.subTest(msg="rebuild_fetchtime_returns_none"):
            mgr = self._make()
            self.assertIsNone(mgr.rebuild_fetchtime_times())

        with self.subTest(msg="get_all_symbols_from_cache"):
            mgr = self._make()
            mgr.cache = {"BTC": [], "ETH": []}
            self.assertEqual(set(mgr.get_all_symbols_from_cache()), {"BTC", "ETH"})

        with self.subTest(msg="mutable_order_update_is_persisted"):
            mgr = self._make()
            mgr.enable_save_state_to_file()
            order = {
                "orderId": 7, "price": "100", "quantity": "0.5",
                "timestamp": int(time.time() * 1000), "side": "BUY",
                "status": "PARTIALLY_FILLED",
            }
            mgr.cache = {"BTC": [order]}
            mgr.save_state_to_file()
            order.update({"quantity": "1.0", "status": "FILLED"})
            mgr.save_state_to_file()
            reloaded = self._make()
            self.assertEqual(reloaded.cache["BTC"][0]["status"], "FILLED")
            self.assertEqual(reloaded.cache["BTC"][0]["quantity"], "1.0")


class TestAccountCacheHealthPublication(unittest.TestCase):
    """Publish freshness only after a complete, durable account-cache sync."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown()
        self.tmp.cleanup()

    def _make_manager(self, manager_class, symbols):
        filename = _tmp_file(
            self.tmp.name, f"{manager_class.__name__}_health.json")
        if manager_class is cm.CacheTradeManager:
            with open(filename, "w", encoding="utf-8") as handle:
                for index, symbol in enumerate(symbols, start=1):
                    handle.write(json.dumps({
                        "s": symbol,
                        "i": {
                            "symbol": symbol,
                            "id": index,
                            "orderId": index,
                            "price": "1",
                            "qty": "1",
                            "time": 1,
                            "isBuyer": True,
                        },
                    }) + "\n")
            with open(filename + ".meta", "w", encoding="utf-8") as handle:
                json.dump({
                    "fetchtime": {symbol: 1 for symbol in symbols},
                    "counts": {symbol: 1 for symbol in symbols},
                }, handle)
        else:
            _write_cache_file(
                filename, {symbol: [] for symbol in symbols},
                {symbol: 1 for symbol in symbols})
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]), \
             patch("binance_api.bapi_allorders.get_filled_orders", return_value=[]):
            manager = manager_class(
                3600, symbols, filename, api_client=MagicMock())
        manager._first_sleep = False
        self.managers.append(manager)
        return manager

    def _run_one_iteration(self, manager, *, canonical_symbols,
                           query_result=True, query_error=None,
                           persisted=True, should_poll=True):
        iteration_complete = threading.Event()

        def query_once():
            if query_error is not None:
                iteration_complete.set()
                raise query_error
            if query_result:
                manager._last_complete_sync_at_ms = 123456
            return query_result
        def persist_once():
            iteration_complete.set()
            if persisted:
                manager._persisted_data_version = "test-cache-version"
            return persisted

        query = MagicMock(side_effect=query_once)
        persist = MagicMock(side_effect=persist_once)
        persist_method = (
            "_save_account_trade_append_if_current_generation_locked"
            if manager.cls_name == "CacheTradeManager"
            else "save_state_to_file_if_enabled"
        )
        with patch.object(manager, "_should_poll", return_value=should_poll), \
             patch.object(manager, "query_remote_and_update_cache", query), \
             patch.object(manager, persist_method, persist), \
             patch.object(cm.sym, "symbols", canonical_symbols), \
             patch.object(
                 cm.account_cache_health, "record_successful_sync") as publish:
            manager.periodic_sync(sync_ts=3600, save_state=True)
            self.assertTrue(
                iteration_complete.wait(2),
                "cache-manager synchronization iteration did not complete")
            self.assertTrue(manager.shutdown())
        return query, persist, publish

    def test_order_and_trade_publish_after_complete_persisted_sync(self):
        symbols = ["BTC", "ETH"]
        for manager_class in (cm.CacheOrderManager, cm.CacheTradeManager):
            with self.subTest(manager=manager_class.__name__):
                manager = self._make_manager(manager_class, symbols)
                query, persist, publish = self._run_one_iteration(
                    manager, canonical_symbols=symbols)
                query.assert_called_once_with()
                persist.assert_called_once_with()
                publish.assert_called_once_with(
                    manager_class.__name__,
                    "test-cache-version",
                    now_ms=123456,
                )

    def test_failed_remote_sync_does_not_publish(self):
        manager = self._make_manager(cm.CacheOrderManager, ["BTC"])
        query, persist, publish = self._run_one_iteration(
            manager,
            canonical_symbols=["BTC"],
            query_error=RuntimeError("remote sync failed"),
        )
        query.assert_called_once_with()
        persist.assert_not_called()
        publish.assert_not_called()

    def test_incomplete_remote_sync_does_not_publish(self):
        manager = self._make_manager(cm.CacheTradeManager, ["BTC"])
        _, persist, publish = self._run_one_iteration(
            manager, canonical_symbols=["BTC"], query_result=False)
        persist.assert_called_once_with()
        publish.assert_not_called()

    def test_persistence_failure_does_not_publish(self):
        manager = self._make_manager(cm.CacheOrderManager, ["BTC"])
        _, persist, publish = self._run_one_iteration(
            manager, canonical_symbols=["BTC"], persisted=False)
        persist.assert_called_once_with()
        publish.assert_not_called()

    def test_restricted_symbol_manager_does_not_publish(self):
        manager = self._make_manager(cm.CacheTradeManager, ["BTC"])
        _, persist, publish = self._run_one_iteration(
            manager, canonical_symbols=["BTC", "ETH"])
        persist.assert_called_once_with()
        publish.assert_not_called()

    def test_non_polling_iteration_does_not_publish(self):
        manager = self._make_manager(cm.CacheOrderManager, ["BTC"])
        query, persist, publish = self._run_one_iteration(
            manager, canonical_symbols=["BTC"], should_poll=False)
        query.assert_not_called()
        persist.assert_called_once_with()
        publish.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CacheSparsePriceManager (renamed from CachePriceManager, 21 Jul)
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheSparsePriceManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cm._current_price_instance = None

    def tearDown(self):
        cm._current_price_instance = None

    def _make(self, price=50000.0):
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = price
        fname = _tmp_file(self.tmp, "cache_price_BTC.json")
        # Patch get_current_price_manager to use this test's mock. get_price(),
        # rather than get_price_value(), preserves the real timestamp.
        cur_mgr = MagicMock()
        cur_mgr.get_price.return_value = [int(time.time() * 1000), price]
        with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
            mgr = cm.CacheSparsePriceManager(9999, ["BTC"], fname, api_client=api_mock)
        self._cur_mgr_mock = cur_mgr
        return mgr

    def test_rebuild_fetchtime_from_cache(self):
        mgr = self._make()
        ts = int(time.time() * 1000)
        mgr.cache = {"BTC": [[ts - 1000, 100.0], [ts, 200.0]]}
        result = mgr.rebuild_fetchtime_times()
        self.assertEqual(result["BTC"], ts)

    def test_rebuild_fetchtime_empty_cache(self):
        mgr = self._make()
        mgr.cache = {}
        result = mgr.rebuild_fetchtime_times()
        self.assertEqual(result, {})

    def test_get_remote_items_returns_price_entry(self):
        fname = _tmp_file(self.tmp, "cp.json")
        api_mock = MagicMock()
        cur_mgr = MagicMock()
        ts = int(time.time() * 1000)
        cur_mgr.get_price.return_value = [ts, 55000.0]
        with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
            mgr = cm.CacheSparsePriceManager(9999, ["BTC"], fname, api_client=api_mock)
            result = mgr.get_remote_items("BTC", 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 55000.0)
        self.assertEqual(result[0][0], ts)   # Real observation timestamp, not time.time().

    def test_get_remote_items_none_price_returns_empty(self):
        fname = _tmp_file(self.tmp, "cp_none.json")
        api_mock = MagicMock()
        cur_mgr = MagicMock()
        cur_mgr.get_price.return_value = None
        with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
            mgr = cm.CacheSparsePriceManager(9999, ["BTC"], fname, api_client=api_mock)
            result = mgr.get_remote_items("BTC", 0)
        self.assertEqual(result, [])

    def test_get_remote_items_uses_real_timestamp_not_wallclock(self):
        """Record a stale price with its real observation time, not wall time."""
        fname = _tmp_file(self.tmp, "cp_frozen.json")
        api_mock = MagicMock()
        cur_mgr = MagicMock()
        frozen_ts = int((time.time() - 27 * 60) * 1000)
        cur_mgr.get_price.return_value = [frozen_ts, 12345.6]
        with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
            mgr = cm.CacheSparsePriceManager(9999, ["BTC"], fname, api_client=api_mock)
            result = mgr.get_remote_items("BTC", 0)
        self.assertEqual(result[0][0], frozen_ts)
        self.assertNotAlmostEqual(result[0][0], int(time.time() * 1000), delta=5000)

    def test_append_mode_true(self):
        mgr = self._make()
        self.assertTrue(mgr.append_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Cache24PriceManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCache24PriceManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cm._current_price_instance = None

    def tearDown(self):
        cm._current_price_instance = None

    def _make(self):
        api_mock = MagicMock()
        fname = _tmp_file(self.tmp, "cache_24price_BTC.json")
        cur_mgr = MagicMock()
        # get_remote_items uses get_price() for timestamp and price rather than
        # the legacy get_price_value(), which would fabricate the timestamp.
        cur_mgr.get_price.return_value = [int(time.time() * 1000), 50000.0]
        with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
            mgr = cm.Cache24PriceManager(9999, ["BTC"], fname, api_client=api_mock)
        return mgr

    def test_on_price_update_appends_entry(self):
        mgr = self._make()
        ts = int(time.time() * 1000)
        mgr.on_price_update("BTC", ts, 60000.0)
        with mgr.lock:
            prices = [e[1] for e in mgr.cache.get("BTC", [])]
        self.assertIn(60000.0, prices)

    def test_on_price_update_multiple_entries_all_kept(self):
        mgr = self._make()
        base_ts = int(time.time() * 1000)
        for i in range(5):
            mgr.on_price_update("BTC", base_ts + i*1000, 50000.0 + i)
        with mgr.lock:
            self.assertGreaterEqual(len(mgr.cache.get("BTC", [])), 5)

    def test_trim_removes_entries_older_than_keep_hours(self):
        mgr = self._make()
        old_ts  = int((time.time() - (mgr.KEEP_HOURS + 1) * 3600) * 1000)
        fresh_ts = int(time.time() * 1000)
        with mgr.lock:
            mgr.cache["BTC"] = [[old_ts, 1.0], [fresh_ts, 2.0]]
        mgr._trim_old_data("BTC")
        with mgr.lock:
            entries = mgr.cache["BTC"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][1], 2.0)

    def test_trim_keeps_all_fresh_entries(self):
        mgr = self._make()
        now = int(time.time() * 1000)
        with mgr.lock:
            mgr.cache["BTC"] = [[now - 100, 1.0], [now, 2.0]]
        mgr._trim_old_data("BTC")
        with mgr.lock:
            self.assertEqual(len(mgr.cache["BTC"]), 2)

    def test_rebuild_fetchtime_max_timestamp(self):
        mgr = self._make()
        mgr.cache = {"BTC": [[100, 1.0], [500, 2.0], [300, 3.0]]}
        result = mgr.rebuild_fetchtime_times()
        self.assertEqual(result["BTC"], 500)

    def test_no_polling_only_saves(self):
        mgr = self._make()
        with patch.object(mgr, "query_remote_and_update_cache") as mock_poll:
            time.sleep(0.1)
            mock_poll.assert_not_called()

    def test_append_mode_true(self):
        mgr = self._make()
        self.assertTrue(mgr.append_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CachePriceLongTrendManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCachePriceLongTrendManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self):
        api_mock = MagicMock()
        fname = _tmp_file(self.tmp, "cache_price_long_trend.json")
        return cm.CachePriceLongTrendManager(9999, ["BTC"], fname, api_client=api_mock)

    def test_rebuild_fetchtime_from_dict_cache(self):
        mgr = self._make()
        mgr.cache = {
            "BTC": [{"timestamp": 100}, {"timestamp": 500}, {"timestamp": 300}]
        }
        result = mgr.rebuild_fetchtime_times()
        # max minus offset 60s
        self.assertEqual(result["BTC"], max(0, 500 * 1000 - 60_000))

    def test_rebuild_fetchtime_empty(self):
        mgr = self._make()
        mgr.cache = {}
        result = mgr.rebuild_fetchtime_times()
        self.assertEqual(result, {})

    def test_get_remote_items_missing_file(self):
        mgr = self._make()
        # A missing priceanalysis.json yields an empty list.
        with patch("os.path.exists", return_value=False):
            result = mgr.get_remote_items("BTC", 0)
        self.assertEqual(result, [])

    def test_get_remote_items_symbol_not_in_data(self):
        mgr = self._make()
        fake_data = {"ETH": {"trend": "up"}}
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(fake_data))):
            with patch("os.path.exists", return_value=True):
                result = mgr.get_remote_items("BTC", 0)
        self.assertEqual(result, [])

    def test_explicit_neutral_trend_clears_the_snapshot(self):
        mgr = self._make()
        with patch("builtins.open", unittest.mock.mock_open(
                read_data=json.dumps({"BTC": None}))):
            with patch("os.path.exists", return_value=True):
                self.assertEqual(mgr.get_remote_items("BTC", 0), [None])

        mgr.cache = {"BTC": [{"timestamp": 500, "direction": "up"}]}
        mgr.update_cache_per_symbol("BTC", [None])
        self.assertEqual(mgr.cache["BTC"], [None])
        self.assertEqual(mgr.rebuild_fetchtime_times(), {})

    def test_rebuild_skips_neutral_and_malformed_trend_entries(self):
        mgr = self._make()
        mgr.cache = {
            "BTC": [
                None,
                "not-a-trend",
                {"timestamp": None},
                {"timestamp": "invalid"},
                {"timestamp": 500},
            ],
        }
        self.assertEqual(
            mgr.rebuild_fetchtime_times(),
            {"BTC": 440_000},
        )

    def test_get_remote_items_returns_symbol_data(self):
        mgr = self._make()
        fake_data = {"BTC": {"trend": "up", "score": 0.9}}
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(fake_data))):
            with patch("os.path.exists", return_value=True):
                result = mgr.get_remote_items("BTC", 0)
        self.assertEqual(result, [{"trend": "up", "score": 0.9}])

    def test_append_mode_false(self):
        mgr = self._make()
        self.assertFalse(mgr.append_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CacheAssetValueManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheAssetValueManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self, total_value=1000.0):
        api_mock = MagicMock()
        api_mock.get_total_assets_value_usdc.return_value = total_value
        fname = _tmp_file(self.tmp, "cache_asset_value.json")
        return cm.CacheAssetValueManager(9999, ["TOTAL"], fname, api_client=api_mock)

    def test_asset_value_manager_behaviors(self):
        with self.subTest(msg="rebuild_fetchtime_from_timestamp_field"):
            mgr = self._make()
            mgr.cache = {
                "TOTAL": [{"timestamp": 200, "total_value_usdc": 1000.0},
                          {"timestamp": 500, "total_value_usdc": 1100.0}]
            }
            result = mgr.rebuild_fetchtime_times()
            self.assertEqual(result["TOTAL"], max(0, 500 * 1000 - 60_000))

        with self.subTest(msg="get_remote_items_returns_snapshot"):
            mgr = self._make(total_value=2500.0)
            result = mgr.get_remote_items("TOTAL", 0)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0]["total_value_usdc"], 2500.0)
            self.assertIn("timestamp", result[0])
            self.assertIn("datetime_local", result[0])

        with self.subTest(msg="get_remote_items_invalid_values_return_empty"):
            for value in (None, 0):
                with self.subTest(value=value):
                    mgr = self._make()
                    mgr.api_client.get_total_assets_value_usdc.return_value = value
                    self.assertEqual(mgr.get_remote_items("TOTAL", 0), [])

        with self.subTest(msg="append_mode_true"):
            mgr = self._make()
            self.assertTrue(mgr.append_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CacheCurrentPriceManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheCurrentPriceManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cm._current_price_instance = None

    def tearDown(self):
        cm._current_price_instance = None

    def _make(self, price=50000.0, provider_names=None):
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = price
        fname = _tmp_file(self.tmp, "cache_currentprice.json")
        # Inject market_api so HTTP fetching uses the testable market-data facade.
        return cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTC"], filename=fname,
            ws_manager=None, api_client=api_mock, market_api=api_mock,
            provider_names=provider_names,
        ), api_mock

    # ── on_items_update ───────────────────────────────────────────────────────

    def test_on_items_update_behaviors(self):
        with self.subTest("stores price and marks ws event"):
            mgr, _ = self._make()
            before = time.time()
            mgr.on_items_update("BTC", [55000.0])
            with mgr.lock:
                entries = mgr.cache.get("BTC", [])
            self.assertTrue(entries)
            self.assertEqual(entries[0][1], 55000.0)
            self.assertGreaterEqual(mgr._ws_last_event_ts, before)

        with self.subTest("ignores none price"):
            mgr, _ = self._make()
            with mgr.lock:
                mgr.cache.clear()
            mgr.on_items_update("BTC", [])
            with mgr.lock:
                entries = mgr.cache.get("BTC", [])
            self.assertFalse(entries)

        with self.subTest("snapshot mode replaces"):
            mgr, _ = self._make()
            mgr.on_items_update("BTC", [50000.0])
            mgr.on_items_update("BTC", [60000.0])
            with mgr.lock:
                entries = mgr.cache["BTC"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0][1], 60000.0)

    # ── get_price / get_price_value ───────────────────────────────────────────

    def test_fresh_price_read_apis_use_cached_value(self):
        mgr, api_mock = self._make()
        mgr.on_items_update("BTC", [55000.0])
        api_mock.get_current_price.reset_mock()
        entry = mgr.get_price("BTC")
        self.assertIsNotNone(entry)
        self.assertEqual(entry[1], 55000.0)
        self.assertEqual(len(entry), 2)
        self.assertIsInstance(entry[0], int)
        self.assertIsInstance(entry[1], float)
        value = mgr.get_price_value("BTC")
        self.assertIsInstance(value, float)
        self.assertEqual(value, 55000.0)
        api_mock.get_current_price.assert_not_called()

    def test_get_price_stale_forces_http(self):
        mgr, api_mock = self._make()
        api_mock.get_current_price.return_value = 65000.0
        with mgr.lock:
            mgr.cache["BTC"] = [[0, 50000.0]]   # A zero timestamp is always stale.
        api_mock.get_current_price.reset_mock()
        entry = mgr.get_price("BTC")
        api_mock.get_current_price.assert_called()
        self.assertEqual(entry[1], 65000.0)

    def test_get_price_missing_forces_http(self):
        mgr, api_mock = self._make()
        api_mock.get_current_price.return_value = 70000.0
        with mgr.lock:
            mgr.cache = {}
        api_mock.get_current_price.reset_mock()
        entry = mgr.get_price("BTC")
        api_mock.get_current_price.assert_called()
        self.assertEqual(entry[1], 70000.0)

    def test_get_price_value_none_if_unavailable(self):
        mgr, api_mock = self._make()
        api_mock.get_current_price.return_value = None
        with mgr.lock:
            mgr.cache = {}
        val = mgr.get_price_value("BTC")
        self.assertIsNone(val)

    def test_get_price_does_not_return_stale_row_after_http_failure(self):
        mgr, api_mock = self._make()
        api_mock.get_current_price.return_value = None
        stale = [0, 50000.0]
        with mgr.lock:
            mgr.cache["BTC"] = [stale]

        self.assertIsNone(mgr.get_price("BTC"))
        api_mock.get_current_price.assert_called()
        with mgr.lock:
            self.assertEqual([stale], mgr.cache["BTC"])

    def test_unbound_remote_price_keeps_implicit_provider_routing(self):
        mgr, api_mock = self._make(price=51000.0)
        api_mock.get_current_price.reset_mock()

        rows = mgr.get_remote_items("BTC", None)

        self.assertEqual(rows[0][1], 51000.0)
        call = api_mock.get_current_price.call_args
        self.assertEqual(call.kwargs["symbol"], "BTC")
        self.assertIsNone(call.kwargs.get("provider_name"))
        self.assertIsNone(mgr.provider_name_for("BTC"))

    def test_bound_remote_price_uses_explicit_provider(self):
        mgr, api_mock = self._make(
            price=85.49,
            provider_names={"BTC": "Kraken"},
        )
        api_mock.get_current_price.reset_mock()

        rows = mgr.get_remote_items("BTC", None)

        self.assertEqual(rows[0][1], 85.49)
        api_mock.get_current_price.assert_called_once_with(
            symbol="BTC",
            provider_name="Kraken",
        )
        self.assertEqual(mgr.provider_name_for("BTC"), "Kraken")

    def test_conflicting_provider_binding_is_rejected(self):
        mgr, _ = self._make(provider_names={"BTC": "Kraken"})

        mgr.bind_provider("BTC", "kraken")
        with self.assertRaisesRegex(ValueError, r"conflicting provider"):
            mgr.bind_provider("BTC", "Hyperliquid")

    def test_cached_price_observation_never_fetches(self):
        mgr, api_mock = self._make()
        mgr.on_items_update("BTC", [55000.0])
        api_mock.get_current_price.reset_mock()

        observed_at, price = mgr.cached_price_observation(
            "BTC", max_age_sec=10.0)

        self.assertAlmostEqual(observed_at, time.time(), delta=1.0)
        self.assertEqual(price, 55000.0)
        api_mock.get_current_price.assert_not_called()

        with mgr.lock:
            mgr.cache["BTC"][0][0] = int((time.time() - 20.0) * 1000)
        self.assertIsNone(
            mgr.cached_price_observation("BTC", max_age_sec=10.0))
        api_mock.get_current_price.assert_not_called()

    # ── subscribe_price / unsubscribe_price ───────────────────────────────────

    def test_subscribe_price_behaviors(self):
        with self.subTest("registration lifecycle"):
            mgr, _ = self._make()
            sub = MagicMock()
            mgr.subscribe_price(sub)
            with mgr.lock:
                self.assertIn(sub, mgr._price_subscribers)
            mgr.subscribe_price(sub)
            with mgr.lock:
                self.assertEqual(mgr._price_subscribers.count(sub), 1)
            mgr.unsubscribe_price(sub)
            with mgr.lock:
                self.assertNotIn(sub, mgr._price_subscribers)

        with self.subTest("ws update notifies subscriber"):
            mgr, _ = self._make()
            sub = MagicMock()
            mgr.subscribe_price(sub)
            mgr.on_items_update("BTC", [62000.0])
            sub.on_price_update.assert_called_once()
            args = sub.on_price_update.call_args[0]
            self.assertEqual(args[0], "BTC")
            self.assertAlmostEqual(args[2], 62000.0)

        with self.subTest("http fetch notifies subscriber"):
            mgr, api_mock = self._make()
            api_mock.get_current_price.return_value = 63000.0
            sub = MagicMock()
            mgr.subscribe_price(sub)
            with mgr.lock:
                mgr.cache["BTC"] = [[0, 1.0]]   # stale
            mgr.get_price("BTC")
            sub.on_price_update.assert_called_once()

        with self.subTest("subscriber exception doesnt block others"):
            mgr, _ = self._make()
            bad = MagicMock()
            bad.on_price_update.side_effect = RuntimeError("crash")
            good = MagicMock()
            mgr.subscribe_price(bad)
            mgr.subscribe_price(good)
            mgr.on_items_update("BTC", [1.0])
            good.on_price_update.assert_called_once()

        with self.subTest("unsubscribed not notified"):
            mgr, _ = self._make()
            sub = MagicMock()
            mgr.subscribe_price(sub)
            mgr.unsubscribe_price(sub)
            mgr.on_items_update("BTC", [1.0])
            sub.on_price_update.assert_not_called()

    # ── WS health ─────────────────────────────────────────────────────────────

    def test_ws_health_from_event_age(self):
        mgr, _ = self._make()
        for label, timestamp, expected in (("recent", time.time(), True), ("old", 0.0, False)):
            with self.subTest(case=label):
                mgr._ws_last_event_ts = timestamp
                self.assertEqual(mgr._ws_is_healthy(), expected)

    # -- Persistence. ----------------------------------------------------------

    def test_persistence_reload(self):
        mgr, api_mock = self._make()
        mgr.enable_save_state_to_file()
        mgr.on_items_update("BTC", [58000.0])
        mgr.save_state_to_file_if_enabled()

        mgr2 = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTC"],
            filename=mgr.filename, api_client=api_mock
        )
        with mgr2.lock:
            entries = mgr2.cache.get("BTC", [])
        self.assertTrue(entries)
        self.assertEqual(entries[0][1], 58000.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WS health functions
# ═══════════════════════════════════════════════════════════════════════════════
class TestWsHealthFunctions(unittest.TestCase):

    def setUp(self):
        cm._ws_available = False
        cm._ws_last_event_ts = 0.0
        cm._ws_is_healthy = False

    def test_mark_ws_available(self):
        for available in (True, False):
            with self.subTest(available=available):
                cm._mark_ws_available(available)
                with cm._ws_health_lock:
                    self.assertEqual(cm._ws_available, available)

    def test_ws_event_and_unhealthy_lifecycle(self):
        cm._mark_ws_event_received()
        with cm._ws_health_lock:
            self.assertTrue(cm._ws_is_healthy)
            self.assertGreater(cm._ws_last_event_ts, 0)
        cm._mark_ws_unhealthy()
        with cm._ws_health_lock:
            self.assertFalse(cm._ws_is_healthy)

    def test_polling_fallback_policies(self):
        cases = (
            ("ws-mode-off", False, True, "CacheOrderManager"),
            ("unmanaged", True, True, "CacheSparsePriceManager"),
            ("ws-unavailable", True, False, "CacheOrderManager"),
        )
        try:
            for label, ws_only, available, manager_name in cases:
                with self.subTest(case=label):
                    cm.WS_ONLY_MODE = ws_only
                    cm._ws_available = available
                    self.assertTrue(cm._should_poll_for_manager(manager_name))
        finally:
            cm.WS_ONLY_MODE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CacheFactory / get_cache_manager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheFactory(unittest.TestCase):

    def setUp(self):
        # Reset singletons.
        cm.CacheFactory.shutdown_all()
        if cm._current_price_instance is not None:
            cm._current_price_instance.shutdown()
        cm._current_price_instance = None

    def tearDown(self):
        cm.CacheFactory.shutdown_all()
        if cm._current_price_instance is not None:
            cm._current_price_instance.shutdown()
        cm._current_price_instance = None

    def test_read_only_creation_can_be_promoted_to_periodic_sync(self):
        with patch.object(cm.CacheTradeManager, "periodic_sync") as periodic, \
             patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]):
            first = cm.CacheFactory.get("Trade", symbols=["BTC"], start_sync=False)
            periodic.assert_not_called()
            second = cm.CacheFactory.get("Trade", symbols=["BTC"], start_sync=True)
            self.assertIs(first, second)
            periodic.assert_called_once()
        cm._current_price_instance = None

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            cm.CacheFactory.get("NonExistent")

    def test_returns_same_instance_twice(self):
        with patch("cacheManager.get_current_price_manager", return_value=MagicMock()):
            i1 = cm.CacheFactory.get("CurrentPrice", symbols=["BTC"])
            i2 = cm.CacheFactory.get("CurrentPrice", symbols=["BTC"])
        self.assertIs(i1, i2)

    def test_factory_and_current_price_getter_share_identity_factory_first(self):
        with patch.object(
                cm.CacheCurrentPriceManager, "load_state", return_value=None):
            factory_manager = cm.CacheFactory.get(
                "CurrentPrice", symbols=["BTC"], start_sync=False)
            singleton_manager = cm.get_current_price_manager(
                symbols=["BTC"], start_sync=False)
        self.assertIs(factory_manager, singleton_manager)
        self.assertEqual(
            singleton_manager.sync_ts,
            cm.CURRENTPRICE_SYNC_INTERVAL_SEC,
        )

    def test_factory_and_current_price_getter_share_identity_getter_first(self):
        with patch.object(
                cm.CacheCurrentPriceManager, "load_state", return_value=None), \
             patch.object(
                cm.CacheCurrentPriceManager, "periodic_sync") as periodic:
            singleton_manager = cm.get_current_price_manager(
                symbols=["BTC"], sync_ts=0.8, start_sync=False)
            factory_manager = cm.CacheFactory.get(
                "CurrentPrice", symbols=["BTC"])
        self.assertIs(factory_manager, singleton_manager)
        self.assertEqual(factory_manager.sync_ts, 0.8)
        periodic.assert_called_once_with(0.8, False)

    def test_price_factories_return_per_symbol_managers_and_filenames(self):
        cur_mock = MagicMock()
        cur_mock.get_price.return_value = [int(time.time() * 1000), 50000.0]
        cases = (
            ("Price", ["BTC", "ETH"], "cache_price_BTC.json"),
            ("Price24", ["BTC"], "cache_24price_BTC.json"),
        )
        for name, symbols, expected_filename in cases:
            with self.subTest(factory=name):
                cm.CacheFactory.remove(name)
                with patch("cacheManager.get_current_price_manager", return_value=cur_mock):
                    result = cm.CacheFactory.get(name, symbols=symbols)
                self.assertIsInstance(result, dict)
                for symbol in symbols:
                    self.assertIn(symbol, result)
                self.assertIn(expected_filename, result["BTC"].filename)

    def test_factory_returns_expected_manager_classes(self):
        cases = (
            ("Trade", cm.CacheTradeManager),
            ("Order", cm.CacheOrderManager),
            ("CurrentPrice", cm.CacheCurrentPriceManager),
        )
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]), \
             patch("binance_api.bapi_allorders.get_filled_orders", return_value=[]), \
             patch.object(cm.CacheCurrentPriceManager, "load_state", return_value=None):
            for name, expected_class in cases:
                with self.subTest(factory=name):
                    result = cm.CacheFactory.get(
                        name, symbols=["BTC"], start_sync=False)
                    self.assertIsInstance(result, expected_class)

    def test_get_cache_manager_delegates_to_factory(self):
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]):
            r1 = cm.get_cache_manager("Trade", symbols=["BTC"])
            r2 = cm.CacheFactory.get("Trade", symbols=["BTC"])
        self.assertIs(r1, r2)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. get_current_price_manager — singleton
# ═══════════════════════════════════════════════════════════════════════════════
class TestGetCurrentPriceManagerSingleton(unittest.TestCase):

    def setUp(self):
        if cm._current_price_instance is not None:
            cm._current_price_instance.shutdown()
        cm._current_price_instance = None

    def tearDown(self):
        if cm._current_price_instance is not None:
            cm._current_price_instance.shutdown()
        cm._current_price_instance = None

    def test_returns_single_cache_current_price_manager(self):
        # This characterizes singleton identity, not live polling.
        m1 = cm.get_current_price_manager(symbols=["BTC"], start_sync=False)
        m2 = cm.get_current_price_manager(symbols=["BTC"], start_sync=False)
        self.assertIsInstance(m1, cm.CacheCurrentPriceManager)
        self.assertIs(m1, m2)

    def test_ws_manager_subscribed_if_provided(self):
        ws = MagicMock()
        mgr = cm.get_current_price_manager(
            ws_manager=ws, symbols=["BTC"], start_sync=False,
        )
        ws.subscribe.assert_called_once_with(mgr)


class TestRefreshSymbolInCache(unittest.TestCase):
    """Test the WebSocket handler helper that refreshes one symbol."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_refresh_single_symbol(self):
        fname = _tmp_file(self.tmp)
        remote = {"SYM": [[int(time.time()*1000), 42.0]]}
        mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items=remote)
        with mgr.lock:
            mgr.cache["SYM"] = []
        cm._refresh_symbol_in_cache(mgr, "SYM")
        with mgr.lock:
            self.assertIn(42.0, [e[1] for e in mgr.cache.get("SYM", [])])

    def test_refresh_missing_symbol_no_crash(self):
        fname = _tmp_file(self.tmp, "x.json")
        mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items={})
        cm._refresh_symbol_in_cache(mgr, "NOPE")   # Must not raise.


class TestFactorySingletonWarning(unittest.TestCase):
    """The named singleton ignores different symbols on later calls."""

    def setUp(self):
        cm.CacheFactory.remove("AssetValue")

    def tearDown(self):
        cm.CacheFactory.remove("AssetValue")

    def test_same_instance_returned_and_warns(self):
        m1 = cm.get_cache_manager("AssetValue", symbols=["TOTAL"])
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m2 = cm.get_cache_manager("AssetValue", symbols=["OTHER"])
        self.assertIs(m1, m2)                       # Same instance.
        self.assertIn("IGNORED", buf.getvalue())    # Warning was emitted.

    def test_no_warning_same_symbols(self):
        cm.get_cache_manager("AssetValue", symbols=["TOTAL"])
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cm.get_cache_manager("AssetValue", symbols=["TOTAL"])
        self.assertNotIn("IGNORED", buf.getvalue())


class TestAppendJsonlPersist(unittest.TestCase):
    """Test JSONL append persistence for append-only Trade/AssetValue caches."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_append_writes_only_delta(self):
        fname = os.path.join(self.tmp, "t.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        with m.lock:
            m.cache["SYM"] = [{"id": 1}, {"id": 2}]
        m.save_state_to_file_if_enabled()
        n1 = sum(1 for _ in open(fname))
        self.assertEqual(n1, 2)
        # Add one item and write only its one-line delta.
        with m.lock:
            m.cache["SYM"].append({"id": 3})
        m.save_state_to_file_if_enabled()
        n2 = sum(1 for _ in open(fname))
        self.assertEqual(n2, 3)   # Two existing lines plus one, not a rewrite.

    def test_load_jsonl_rebuilds_cache(self):
        fname = os.path.join(self.tmp, "t2.jsonl")
        w = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        w.save_state = True
        with w.lock:
            w.cache["SYM"] = [{"id": 1}, {"id": 2}]
        w.save_state_to_file_if_enabled()
        # Another manager loads the JSONL at startup.
        r = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        self.assertEqual(r.cache.get("SYM"), [{"id": 1}, {"id": 2}])

    def test_compact_dedups(self):
        fname = os.path.join(self.tmp, "dedup.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        with m.lock:
            m.cache["SYM"] = [{"id": 1}, {"id": 1}, {"id": 2}, {"id": 2}, {"id": 3}]
        m.save_state_to_file_if_enabled()
        m.compact_jsonl()
        with m.lock:
            self.assertEqual(m.cache["SYM"], [{"id": 1}, {"id": 2}, {"id": 3}])  # Deduplicate memory.
        n = sum(1 for _ in open(fname))
        self.assertEqual(n, 3)   # dedup pe disc

    def test_save_to_file_unconditional_vs_if_enabled(self):
        fname = _tmp_file(self.tmp, "split.json")
        m = ConcreteTestManager(9999, ["SYM"], fname)
        m.save_state = False
        with m.lock:
            m.cache["SYM"] = [[1, 1.0]]
        m.save_state_to_file_if_enabled()       # save_state=False does not write.
        self.assertFalse(os.path.exists(fname))
        m.save_state_to_file()                  # Unconditional write.
        self.assertTrue(os.path.exists(fname))

    def test_compact_rewrites(self):
        fname = os.path.join(self.tmp, "t3.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        with m.lock:
            m.cache["SYM"] = [{"id": i} for i in range(5)]
        m.save_state_to_file_if_enabled()
        m.compact_jsonl()
        n = sum(1 for _ in open(fname))
        self.assertEqual(n, 5)
        r = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        self.assertEqual(len(r.cache.get("SYM")), 5)

    def test_maintain_prunes_old_entries(self):
        fname = os.path.join(self.tmp, "p.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        m.RETENTION_DAYS = 730
        now_ms = int(time.time() * 1000)
        old_ms = now_ms - 800 * 24 * 3600 * 1000   # More than two years old.
        with m.lock:
            m.cache["SYM"] = [[old_ms, 1.0], [now_ms, 2.0]]
        m.save_state_to_file_if_enabled()
        m.maintain_append_persist()
        with m.lock:
            self.assertEqual(m.cache["SYM"], [[now_ms, 2.0]])   # Old entry removed.

    def test_retention_normalizes_unix_seconds(self):
        now_sec = int(time.time())
        self.assertEqual(
            ConcreteTestManager._entry_timestamp_ms({"timestamp": now_sec}),
            now_sec * 1000,
        )

    def test_rotation_archives_and_keeps_latest(self):
        fname = os.path.join(self.tmp, "r.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        m.MAX_FILE_BYTES = 1          # Force rotation.
        m.ROTATE_KEEP_FRACTION = 0.10
        now_ms = int(time.time() * 1000)
        with m.lock:
            m.cache["SYM"] = [[now_ms + i, float(i)] for i in range(100)]
        m.save_state_to_file_if_enabled()
        m.maintain_append_persist()
        # Create an archive and retain the newest ten percent in memory.
        archives = [f for f in os.listdir(self.tmp) if ".archive" in f]
        self.assertTrue(archives)
        with m.lock:
            self.assertEqual(len(m.cache["SYM"]), 10)
            self.assertEqual(m.cache["SYM"][-1], [now_ms + 99, 99.0])

    # -- Safety: rotation and maintenance do not lose data. --------------------

    def _count_lines(self, path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def test_rotation_archive_has_FULL_history(self):
        """The archive contains every original record, so no data is lost."""
        fname = os.path.join(self.tmp, "full.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        m.MAX_FILE_BYTES = 1
        now = int(time.time() * 1000)
        with m.lock:
            m.cache["SYM"] = [[now + i, float(i)] for i in range(100)]
        m.save_state_to_file_if_enabled()
        self.assertEqual(self._count_lines(fname), 100)
        m.maintain_append_persist()
        archive = [os.path.join(self.tmp, f) for f in os.listdir(self.tmp) if ".archive" in f][0]
        self.assertEqual(self._count_lines(archive), 100)   # Complete history.
        self.assertEqual(self._count_lines(fname), 10)       # Newest ten percent.
        # The compressed archive remains a complete, independently recoverable JSONL stream.
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[-1]["i"], [now + 99, 99.0])

    def test_legacy_full_json_is_migrated_without_deleting_source(self):
        legacy = os.path.join(self.tmp, "cache_trade.json")
        target = legacy + "l"
        now = int(time.time() * 1000)
        _write_cache_file(legacy, {"SYM": [[now, 1.0]]}, {"SYM": now})
        manager = ConcreteTestManager(9999, ["SYM"], target, append_persist=True)
        self.assertTrue(os.path.exists(legacy))
        self.assertTrue(os.path.exists(target))
        self.assertEqual(manager.cache["SYM"], [[now, 1.0]])

    def test_maintain_noop_leaves_file_intact(self):
        """Maintenance leaves a small recent file intact and creates no archive."""
        fname = os.path.join(self.tmp, "intact.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        now = int(time.time() * 1000)
        with m.lock:
            m.cache["SYM"] = [[now, 1.0], [now + 1, 2.0]]
        m.save_state_to_file_if_enabled()
        before = open(fname).read()
        m.maintain_append_persist()
        self.assertEqual(open(fname).read(), before)         # File unchanged.
        self.assertEqual([f for f in os.listdir(self.tmp) if ".archive" in f], [])  # No archive.
        self.assertEqual(len(m.cache["SYM"]), 2)             # Data intact.

    def test_maintain_missing_file_no_crash(self):
        fname = os.path.join(self.tmp, "nofile.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.maintain_append_persist()   # A missing file must not crash.
        self.assertEqual([f for f in os.listdir(self.tmp) if ".archive" in f], [])

    def test_maintain_noop_when_not_append_persist(self):
        """Fresh data makes generalized append maintenance a true no-op.

        Maintenance also runs when append_persist is false so retention covers
        Trade, Order, and AssetValue. The former timestamp=1 fixture was stale and
        made old gating appear to be a no-op only because it skipped maintenance.
        """
        fname = os.path.join(self.tmp, "fullrw.json")
        now_ms = int(time.time() * 1000)
        _write_cache_file(fname, {"SYM": [[now_ms, 1.0]]})
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=False)
        before = open(fname).read()
        m.maintain_append_persist()   # Fresh data leaves nothing to prune.
        self.assertEqual(open(fname).read(), before)
        self.assertEqual(len(m.cache["SYM"]), 1)

    def test_maintain_prunes_old_entries_even_when_not_append_persist(self):
        """Apply retention to full-JSON classes where append_persist is false.

        This covers the discovered unbounded Trade, Order, and AssetValue growth
        and explicitly exercises save_state_to_file after pruning.
        """
        fname = os.path.join(self.tmp, "fullrw_prune.json")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=False)
        m.RETENTION_DAYS = 730
        now_ms = int(time.time() * 1000)
        old_ms = now_ms - 800 * 24 * 3600 * 1000   # More than two years old.
        with m.lock:
            m.cache["SYM"] = [[old_ms, 1.0], [now_ms, 2.0]]
        m.save_state_to_file_if_enabled()
        m.maintain_append_persist()
        with m.lock:
            self.assertEqual(m.cache["SYM"], [[now_ms, 2.0]])   # Old entry removed from memory.
        m.enable_save_state_to_file()
        m.save_state_to_file_if_enabled()
        with open(fname) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["items"]["SYM"], [[now_ms, 2.0]])   # Also on disk.

    def test_prune_keeps_file_and_recent(self):
        fname = os.path.join(self.tmp, "prune.jsonl")
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        m.save_state = True
        now = int(time.time() * 1000)
        old = now - 800 * 24 * 3600 * 1000
        with m.lock:
            m.cache["SYM"] = [[old, 1.0], [now, 2.0]]
        m.save_state_to_file_if_enabled()
        m.maintain_append_persist()
        self.assertTrue(os.path.exists(fname))               # File still exists.
        r = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        self.assertEqual(r.cache["SYM"], [[now, 2.0]])       # Retain recent, remove old.

    def test_skips_corrupt_lines(self):
        fname = os.path.join(self.tmp, "t4.jsonl")
        with open(fname, "w") as f:
            f.write(json.dumps({"s": "SYM", "i": {"id": 1}}) + "\n")
            f.write("{partial broken line\n")   # Corrupt line from a crash during append.
            f.write(json.dumps({"s": "SYM", "i": {"id": 2}}) + "\n")
        r = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
        self.assertEqual(r.cache.get("SYM"), [{"id": 1}, {"id": 2}])   # Skip corrupt line.


class TestMemFileResync(unittest.TestCase):
    """Test stale-overwrite protection and memory-to-file reconciliation."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _mgr(self, fname, append_persist=False):
        m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=append_persist)
        m.save_state = True
        return m

    def test_refuses_overwrite_with_older(self):
        fname = _tmp_file(self.tmp, "g.json")
        # Writer A writes new data with a high fetch time.
        a = self._mgr(fname)
        with a.lock:
            a.cache["SYM"] = [[1, 1.0]]
            a.fetchtime_time_per_symbol["SYM"] = 2000   # New.
        a.save_state_to_file_if_enabled()
        # Writer B has stale data and must not overwrite it.
        b = ConcreteTestManager(9999, ["SYM"], fname)
        b.save_state = True
        with b.lock:
            b.cache["SYM"] = [[9, 9.0]]
            b.fetchtime_time_per_symbol["SYM"] = 1000   # Older.
        b.save_state_to_file_if_enabled()
        # The file retains writer A's data.
        data = json.load(open(fname))
        self.assertEqual(data["items"]["SYM"], [[1, 1.0]])

    def test_allows_overwrite_with_newer(self):
        fname = _tmp_file(self.tmp, "g2.json")
        a = self._mgr(fname)
        with a.lock:
            a.cache["SYM"] = [[1, 1.0]]
            a.fetchtime_time_per_symbol["SYM"] = 1000
        a.save_state_to_file_if_enabled()
        b = ConcreteTestManager(9999, ["SYM"], fname)
        b.save_state = True
        with b.lock:
            b.cache["SYM"] = [[9, 9.0]]
            b.fetchtime_time_per_symbol["SYM"] = 3000   # Newer, so overwrite.
        b.save_state_to_file_if_enabled()
        data = json.load(open(fname))
        self.assertEqual(data["items"]["SYM"], [[9, 9.0]])

    def test_resync_reloads_when_file_newer(self):
        fname = _tmp_file(self.tmp, "r.json")
        # Process A writes new data to disk.
        a = self._mgr(fname)
        with a.lock:
            a.cache["SYM"] = [[5, 5.0]]
            a.fetchtime_time_per_symbol["SYM"] = 5000
        a.save_state_to_file_if_enabled()
        # Process B has stale memory, so resync must reload it.
        b = ConcreteTestManager(9999, ["SYM"], fname)
        with b.lock:
            b.cache["SYM"] = [[1, 1.0]]
            b.fetchtime_time_per_symbol["SYM"] = 1000
        b.resync_mem_file()
        with b.lock:
            self.assertEqual(b.cache["SYM"], [[5, 5.0]])   # Reloaded from the file.


class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_atomic_write_json_roundtrip(self):
        p = os.path.join(self.tmp, "a.json")
        cm.atomic_write_json(p, {"x": 1, "y": [2, 3]})
        with open(p) as f:
            self.assertEqual(json.load(f), {"x": 1, "y": [2, 3]})
        self.assertFalse(os.path.exists(p + ".tmp"))   # Temporary file cleaned up.

    def test_atomic_write_cleanup_on_error(self):
        p = os.path.join(self.tmp, "b.json")
        with self.assertRaises(ValueError):
            with cm.atomic_write(p) as f:
                f.write("partial")
                raise ValueError("boom")
        self.assertFalse(os.path.exists(p))            # Target file untouched.
        self.assertFalse(os.path.exists(p + ".tmp"))   # Temporary file removed.

    def test_atomic_write_preserves_old_on_error(self):
        p = os.path.join(self.tmp, "c.json")
        cm.atomic_write_json(p, {"v": "old"})
        with self.assertRaises(ValueError):
            with cm.atomic_write(p) as f:
                f.write("new-incomplete")
                raise ValueError("boom")
        with open(p) as f:
            self.assertEqual(json.load(f), {"v": "old"})   # Previous content remains intact.


if __name__ == "__main__":
    unittest.main(verbosity=2)
