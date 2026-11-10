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

    def rebuild_fetchtimes(self):
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
            mgr.fetchtime_per_symbol = {}
            mgr.fallback_time_default = 123_456
            with patch.object(cm.os.path, "exists", return_value=False):
                result = mgr._rebuild_fetchtimes()
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
            mgr.save_state = True
            mgr.cache["SYM"] = [[999, 42.0]]
            mgr.save_state_to_file_if_enabled()
            self.assertTrue(os.path.exists(fname))
            with open(fname) as f:
                data = json.load(f)
            self.assertEqual(data["items"]["SYM"], [[999, 42.0]])

        with self.subTest("save uses tmp then replace"):
            fname = _tmp_file(self.tmp, "atomic.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            mgr.save_state = True
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
            self.assertIn("SYM", mgr.fetchtime_per_symbol)

        with self.subTest("on_items_update stores entry"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname, append_mode=True, remote_items={"SYM": []})
            mgr.on_items_update("SYM", [[int(time.time()*1000), 123.0]])
            with mgr.lock:
                self.assertIn(123.0, [e[1] for e in mgr.cache.get("SYM", [])])

    # ── filter_new_items and query_remote ─────────────────────────────────────

    def test_filter_and_remote_query_behaviors(self):
        with self.subTest("filter_new_items removes duplicates"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            existing = [[1, 10.0], [2, 20.0]]
            new = [[2, 20.0], [3, 30.0]]
            self.assertEqual(mgr.filter_new_items(existing, new), [[3, 30.0]])

        with self.subTest("filter_new_items all new"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname)
            self.assertEqual(len(mgr.filter_new_items([], [[1, 1.0], [2, 2.0]])), 2)

        with self.subTest("query_remote fetches and stores"):
            fname = _tmp_file(self.tmp)
            ts = int(time.time() * 1000)
            remote = {"SYM": [[ts, 99.0]]}
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items=remote)
            mgr.cache = {}
            mgr.fetchtime_per_symbol = {}
            mgr.query_remote_and_update_cache()
            with mgr.lock:
                self.assertIn(99.0, [e[1] for e in mgr.cache.get("SYM", [])])

        with self.subTest("query_remote skips empty"):
            fname = _tmp_file(self.tmp)
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items={"SYM": []})
            mgr.cache = {}
            mgr.fetchtime_per_symbol = {}
            mgr.query_remote_and_update_cache()
            with mgr.lock:
                self.assertEqual(mgr.cache.get("SYM", []), [])

        with self.subTest("query_remote continues on empty symbol"):
            fname = _tmp_file(self.tmp)
            ts = int(time.time() * 1000)
            remote = {"SYM1": [], "SYM2": [[ts, 5.0]]}
            mgr = ConcreteTestManager(9999, ["SYM1", "SYM2"], fname, remote_items=remote)
            mgr.cache = {}
            mgr.fetchtime_per_symbol = {}
            mgr.query_remote_and_update_cache()
            with mgr.lock:
                self.assertIn(5.0, [e[1] for e in mgr.cache.get("SYM2", [])])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cache persistence durability
# ═══════════════════════════════════════════════════════════════════════════════
class TestCachePersistenceDurability(unittest.TestCase):
    """Keep durable rows, cursors, and metadata on one coherent snapshot."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_manager(self, *, append_persist, filename=None):
        suffix = "jsonl" if append_persist else "json"
        if filename is None:
            filename = _tmp_file(
                self.tmp.name, f"durable_{len(os.listdir(self.tmp.name))}_{time.time_ns()}.{suffix}")
        manager = ConcreteTestManager(
            9999,
            ["SYM"],
            filename,
            append_mode=True,
            append_persist=append_persist,
            remote_items={"SYM": []},
        )
        manager.cache = {"SYM": [[1000, 1.0]]}
        manager.fetchtime_per_symbol = {"SYM": 1000}
        manager._persisted_counts = {}
        return manager

    @staticmethod
    def _read_jsonl(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_jsonl_write_failures_rollback_and_retry(self):
        with self.subTest("partial write rolls back cursor and retries all rows"):
            manager = self._make_manager(append_persist=True)
            manager.cache = {
                "AAA": [[1000, 1.0]],
                "BBB": [[2000, 2.0]],
            }
            manager.fetchtime_per_symbol = {"AAA": 1000, "BBB": 2000}
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

        with self.subTest("fsync failure rolls back and retry persists every row"):
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

        with self.subTest("metadata failure does not duplicate durable rows on retry"):
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

    def test_metadata_cannot_advance_past_durable_data(self):
        for append_persist in (True, False):
            with self.subTest(append_persist=append_persist):
                manager = self._make_manager(append_persist=append_persist)
                self._save_while_update_attempts_to_cross_metadata(manager)

                with open(manager.filename + ".meta", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                self.assertEqual(metadata["counts"], {"SYM": 1})
                self.assertEqual(len(manager.cache["SYM"]), 2)

                if append_persist:
                    rows = self._read_jsonl(manager.filename)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(metadata["fetchtime"], {"SYM": 1000})
                else:
                    with open(manager.filename, encoding="utf-8") as handle:
                        data = json.load(handle)
                    self.assertEqual(data["items"], {"SYM": [[1000, 1.0]]})
                    self.assertEqual(data["fetchtime"], metadata["fetchtime"])


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
                    self.assertEqual(mgr._is_valid_item(trade), expected)

        with self.subTest(msg="rebuild_fetchtime_returns_none"):
            mgr = self._make()
            self.assertIsNone(mgr.rebuild_fetchtimes())

        with self.subTest(msg="append_mode_true"):
            mgr = self._make()
            self.assertTrue(mgr.append_mode)

        with self.subTest(msg="kraken_venue_remote_items_and_ws_normalization"):
            fname = _tmp_file(self.tmp, "cache_trade_kraken.json")
            kraken_api = MagicMock()
            kraken_api._private.return_value = {
                "ledger": {
                    "L1": {"refid": "T100", "type": "trade", "asset": "HYPE", "amount": "2.0", "fee": "0.0", "time": 1700000000.0},
                    "L2": {"refid": "T100", "type": "trade", "asset": "ZUSD", "amount": "-50.0", "fee": "0.1", "time": 1700000000.0},
                }
            }
            mgr = cm.CacheTradeManager(
                9999, ["HYPEUSD"], fname, api_client=kraken_api, venue="kraken",
            )
            items = mgr.get_remote_items("HYPEUSD", 0)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["id"], "T100")
            self.assertEqual(items[0]["symbol"], "HYPEUSD")
            self.assertTrue(items[0]["isBuyer"])
            from providers.kraken_provider import KrakenProvider

            ws_item = KrakenProvider.normalize_ws_trade(
                "200",
                {"ordertxid": "O200", "pair": "HYPEUSD", "price": 26.0, "vol": 1.5, "type": "sell", "time": 1700000100.0},
            )
            self.assertEqual(ws_item["id"], "200")
            self.assertEqual(ws_item["symbol"], "HYPEUSD")
            self.assertFalse(ws_item["isBuyer"])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CacheOrderManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheOrderManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self):
        api_mock = MagicMock()
        fname = _tmp_file(self.tmp, "cache_order.json")
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
                    self.assertEqual(mgr._is_valid_item(order), expected)

        with self.subTest(msg="rebuild_fetchtime_returns_none"):
            mgr = self._make()
            self.assertIsNone(mgr.rebuild_fetchtimes())

        with self.subTest(msg="mutable_order_update_is_persisted"):
            mgr = self._make()
            mgr.save_state = True
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
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]):
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

    def test_non_publishing_failure_and_skip_scenarios(self):
        with self.subTest("failed remote sync"):
            manager = self._make_manager(cm.CacheOrderManager, ["BTC"])
            query, persist, publish = self._run_one_iteration(
                manager,
                canonical_symbols=["BTC"],
                query_error=RuntimeError("remote sync failed"),
            )
            query.assert_called_once_with()
            persist.assert_not_called()
            publish.assert_not_called()

        with self.subTest("incomplete remote sync"):
            manager = self._make_manager(cm.CacheTradeManager, ["BTC"])
            _, persist, publish = self._run_one_iteration(
                manager, canonical_symbols=["BTC"], query_result=False)
            persist.assert_called_once_with()
            publish.assert_not_called()

        with self.subTest("persistence failure"):
            manager = self._make_manager(cm.CacheOrderManager, ["BTC"])
            _, persist, publish = self._run_one_iteration(
                manager, canonical_symbols=["BTC"], persisted=False)
            persist.assert_called_once_with()
            publish.assert_not_called()

        with self.subTest("restricted symbol manager"):
            manager = self._make_manager(cm.CacheTradeManager, ["BTC"])
            _, persist, publish = self._run_one_iteration(
                manager, canonical_symbols=["BTC", "ETH"])
            persist.assert_called_once_with()
            publish.assert_not_called()

        with self.subTest("non polling iteration"):
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
        cm.CacheFactory.remove("CurrentPrice")

    def tearDown(self):
        cm.CacheFactory.remove("CurrentPrice")

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

    def test_sparse_price_manager_rebuild_and_remote_behaviors(self):
        with self.subTest("rebuild_fetchtime_from_cache"):
            mgr = self._make()
            ts = int(time.time() * 1000)
            mgr.cache = {"BTC": [[ts - 1000, 100.0], [ts, 200.0]]}
            result = mgr.rebuild_fetchtimes()
            self.assertEqual(result["BTC"], ts)

        with self.subTest("rebuild_fetchtime_empty_cache"):
            mgr = self._make()
            mgr.cache = {}
            result = mgr.rebuild_fetchtimes()
            self.assertEqual(result, {})

        with self.subTest("get_remote_items_returns_price_entry"):
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

        with self.subTest("get_remote_items_none_price_returns_empty"):
            fname = _tmp_file(self.tmp, "cp_none.json")
            api_mock = MagicMock()
            cur_mgr = MagicMock()
            cur_mgr.get_price.return_value = None
            with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
                mgr = cm.CacheSparsePriceManager(9999, ["BTC"], fname, api_client=api_mock)
                result = mgr.get_remote_items("BTC", 0)
            self.assertEqual(result, [])

        with self.subTest("get_remote_items_uses_real_timestamp_not_wallclock"):
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

        with self.subTest("append_mode_true"):
            mgr = self._make()
            self.assertTrue(mgr.append_mode)

        with self.subTest("get_remote_items_rejects_out_of_order_and_invalid_price"):
            mgr = self._make()
            now_ms = int(time.time() * 1000)
            with mgr.lock:
                mgr.cache["BTC"] = [[now_ms, 50000.0]]
            cur_mgr = MagicMock()
            with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
                cur_mgr.get_price.return_value = [now_ms - 1, 51000.0]
                self.assertEqual(mgr.get_remote_items("BTC", 0), [])
                cur_mgr.get_price.return_value = [now_ms + 1, 0.0]
                self.assertEqual(mgr.get_remote_items("BTC", 0), [])
                cur_mgr.get_price.return_value = [now_ms + 1, 52000.0]
                self.assertEqual(mgr.get_remote_items("BTC", 0), [[now_ms + 1, 52000.0]])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Cache24PriceManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCache24PriceManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cm.CacheFactory.remove("CurrentPrice")

    def tearDown(self):
        cm.CacheFactory.remove("CurrentPrice")

    def _make(self):
        api_mock = MagicMock()
        fname = _tmp_file(self.tmp, "cache_24_price_BTC.json")
        cur_mgr = MagicMock()
        # get_remote_items uses get_price() for timestamp and price rather than
        # the legacy get_price_value(), which would fabricate the timestamp.
        cur_mgr.get_price.return_value = [int(time.time() * 1000), 50000.0]
        with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
            mgr = cm.Cache24PriceManager(9999, ["BTC"], fname, api_client=api_mock)
        return mgr

    def test_cache24_price_manager_lifecycle_and_trimming(self):
        with self.subTest("on_price_update_appends_entry"):
            mgr = self._make()
            ts = int(time.time() * 1000)
            mgr.on_price_update("BTC", ts, 60000.0)
            with mgr.lock:
                prices = [e[1] for e in mgr.cache.get("BTC", [])]
            self.assertIn(60000.0, prices)

        with self.subTest("on_price_update_multiple_entries_all_kept"):
            mgr = self._make()
            base_ts = int(time.time() * 1000)
            for i in range(5):
                mgr.on_price_update("BTC", base_ts + i*1000, 50000.0 + i)
            with mgr.lock:
                self.assertGreaterEqual(len(mgr.cache.get("BTC", [])), 5)

        with self.subTest("on_price_update_rejects_older_timestamp"):
            mgr = self._make()
            base_ts = int(time.time() * 1000) + 10_000
            mgr.on_price_update("BTC", base_ts, 60000.0)
            mgr.on_price_update("BTC", base_ts - 1000, 50000.0)
            with mgr.lock:
                self.assertEqual(mgr.cache["BTC"][-1], [base_ts, 60000.0])

        with self.subTest("trim_removes_entries_older_than_keep_hours"):
            mgr = self._make()
            old_ts = int((time.time() - (mgr.KEEP_HOURS + 1) * 3600) * 1000)
            fresh_ts = int(time.time() * 1000)
            with mgr.lock:
                mgr.cache["BTC"] = [[old_ts, 1.0], [fresh_ts, 2.0]]
            mgr._trim_old_data("BTC")
            with mgr.lock:
                entries = mgr.cache["BTC"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0][1], 2.0)

        with self.subTest("trim_keeps_all_fresh_entries"):
            mgr = self._make()
            now = int(time.time() * 1000)
            with mgr.lock:
                mgr.cache["BTC"] = [[now - 100, 1.0], [now, 2.0]]
            mgr._trim_old_data("BTC")
            with mgr.lock:
                self.assertEqual(len(mgr.cache["BTC"]), 2)

        with self.subTest("rebuild_fetchtime_max_timestamp"):
            mgr = self._make()
            mgr.cache = {"BTC": [[100, 1.0], [500, 2.0], [300, 3.0]]}
            result = mgr.rebuild_fetchtimes()
            self.assertEqual(result["BTC"], 500)

        with self.subTest("no_polling_only_saves"):
            mgr = self._make()
            with patch.object(mgr, "query_remote_and_update_cache") as mock_poll:
                time.sleep(0.1)
                mock_poll.assert_not_called()

        with self.subTest("append_mode_true"):
            mgr = self._make()
            self.assertTrue(mgr.append_mode)

        with self.subTest("get_remote_items_rejects_out_of_order_and_invalid_price"):
            mgr = self._make()
            now_ms = int(time.time() * 1000)
            with mgr.lock:
                mgr.cache["BTC"] = [[now_ms, 50000.0]]
            cur_mgr = MagicMock()
            with patch("cacheManager.get_current_price_manager", return_value=cur_mgr):
                cur_mgr.get_price.return_value = [now_ms - 1, 51000.0]
                self.assertEqual(mgr.get_remote_items("BTC", 0), [])
                cur_mgr.get_price.return_value = [now_ms + 1, 0.0]
                self.assertEqual(mgr.get_remote_items("BTC", 0), [])
                cur_mgr.get_price.return_value = [now_ms + 1, 52000.0]
                self.assertEqual(mgr.get_remote_items("BTC", 0), [[now_ms + 1, 52000.0]])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CacheLongTrendManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheLongTrendManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self):
        api_mock = MagicMock()
        fname = _tmp_file(self.tmp, "cache_long_trend.json")
        return cm.CacheLongTrendManager(9999, ["BTC"], fname, api_client=api_mock)

    def test_price_long_trend_manager_behaviors(self):
        with self.subTest("rebuild_fetchtime_from_dict_cache"):
            mgr = self._make()
            mgr.cache = {
                "BTC": [{"timestamp": 100}, {"timestamp": 500}, {"timestamp": 300}]
            }
            result = mgr.rebuild_fetchtimes()
            self.assertEqual(result["BTC"], max(0, 500 * 1000 - 60_000))

        with self.subTest("rebuild_fetchtime_empty"):
            mgr = self._make()
            mgr.cache = {}
            result = mgr.rebuild_fetchtimes()
            self.assertEqual(result, {})

        with self.subTest("get_remote_items_missing_file"):
            mgr = self._make()
            with patch("os.path.exists", return_value=False):
                result = mgr.get_remote_items("BTC", 0)
            self.assertEqual(result, [])

        with self.subTest("get_remote_items_symbol_not_in_data"):
            mgr = self._make()
            fake_data = {"ETH": {"trend": "up"}}
            with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(fake_data))):
                with patch("os.path.exists", return_value=True):
                    result = mgr.get_remote_items("BTC", 0)
            self.assertEqual(result, [])

        with self.subTest("explicit_neutral_trend_clears_the_snapshot"):
            mgr = self._make()
            with patch("builtins.open", unittest.mock.mock_open(
                    read_data=json.dumps({"BTC": None}))):
                with patch("os.path.exists", return_value=True):
                    self.assertEqual(mgr.get_remote_items("BTC", 0), [None])

            mgr.cache = {"BTC": [{"timestamp": 500, "direction": "up"}]}
            mgr.update_cache_per_symbol("BTC", [None])
            self.assertEqual(mgr.cache["BTC"], [None])
            self.assertEqual(mgr.rebuild_fetchtimes(), {})

        with self.subTest("rebuild_skips_neutral_and_malformed_trend_entries"):
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
                mgr.rebuild_fetchtimes(),
                {"BTC": 440_000},
            )

        with self.subTest("get_remote_items_returns_symbol_data"):
            mgr = self._make()
            fake_data = {"BTC": {"trend": "up", "score": 0.9}}
            with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(fake_data))):
                with patch("os.path.exists", return_value=True):
                    result = mgr.get_remote_items("BTC", 0)
            self.assertEqual(result, [{"trend": "up", "score": 0.9}])

        with self.subTest("append_mode_false"):
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
            result = mgr.rebuild_fetchtimes()
            self.assertEqual(result["TOTAL"], max(0, 500 * 1000 - 60_000))

        with self.subTest(msg="get_remote_items_returns_snapshot"):
            mgr = self._make(total_value=2500.0)
            result = mgr.get_remote_items("TOTAL", 0)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0]["total_value_usdc"], 2500.0)
            self.assertIn("timestamp", result[0])
            self.assertIn("datetime_local", result[0])

        with self.subTest(msg="get_remote_items_invalid_values_return_empty"):
            for value in (None, 0, -5.0, True, False, float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=value):
                    mgr = self._make()
                    mgr.api_client.get_total_assets_value_usdc.return_value = value
                    self.assertEqual(mgr.get_remote_items("TOTAL", 0), [])

        with self.subTest(msg="append_mode_true"):
            mgr = self._make()
            self.assertTrue(mgr.append_mode)

        with self.subTest(msg="usdt_to_usdc_migration_persists_to_jsonl_on_disk"):
            fname = _tmp_file(self.tmp, "cache_asset_value_mig.jsonl")
            with open(fname, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"s": "TOTAL", "i": {"timestamp": 500, "total_value_usdt": 1234.5}}) + "\n"
                )
            api_mock = MagicMock()
            api_mock.get_total_assets_value_usdc.return_value = 1234.5
            mgr = cm.CacheAssetValueManager(9999, ["TOTAL"], fname, api_client=api_mock)
            with open(fname, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["i"].get("total_value_usdc"), 1234.5)
            self.assertNotIn("total_value_usdt", rows[0]["i"])

            with open(fname, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"s": "TOTAL", "i": {"timestamp": 600, "total_value_usdt": 2345.6}}) + "\n"
                )
            mgr._reload_from_disk()
            self.assertEqual(mgr.cache["TOTAL"][-1].get("total_value_usdc"), 2345.6)
            self.assertNotIn("total_value_usdt", mgr.cache["TOTAL"][-1])

        with self.subTest(msg="multi_venue_balances_enriched_in_snapshot"):
            orders_dir = os.path.join(self.tmp, "orders")
            state_dir = os.path.join(self.tmp, "state")
            os.makedirs(orders_dir, exist_ok=True)
            os.makedirs(state_dir, exist_ok=True)
            vb_path = os.path.join(state_dir, "venue_balances_state.json")
            with open(vb_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cash:kraken:USD": {"venue": "kraken", "asset": "USD", "free": 500.0, "locked_quote_usd": 100.0},
                        "pos:kraken:HYPEUSD": {"venue": "kraken", "symbol": "HYPEUSD", "base": "HYPE", "value_usd": 300.0, "locked_base_usd": 50.0},
                        "pos:kraken:HYPEUSDC": {"venue": "kraken", "symbol": "HYPEUSDC", "base": "HYPE", "value_usd": 300.0, "locked_base_usd": 0.0},
                        "cash:t212:USD": {"venue": "t212", "asset": "USD", "free_quote_usd": 14.0},
                        "pos:t212:NVDA_US_EQ": {"venue": "t212", "symbol": "NVDA_US_EQ", "free_base": 0.43, "price": 200.0},
                    },
                    handle,
                )
            api_mock = MagicMock()
            api_mock.get_total_assets_value_usdc.return_value = 2000.0
            av_file = os.path.join(orders_dir, "cache_asset_value.jsonl")
            mgr = cm.CacheAssetValueManager(9999, ["TOTAL"], av_file, api_client=api_mock)
            items = mgr.get_remote_items("TOTAL", 0)
            self.assertEqual(len(items), 1)
            snap = items[0]
            self.assertEqual(snap["total_value_usdc"], 2000.0)
            self.assertEqual(snap["venues_usd"], {"binance": 2000.0, "kraken": 950.0, "t212": 100.0})
            self.assertEqual(snap["portfolio_total_usd"], 3050.0)

            prices_dir = os.path.join(self.tmp, "prices")
            os.makedirs(prices_dir, exist_ok=True)
            cp_path = os.path.join(prices_dir, "cache_current_price.json")
            with open(cp_path, "w", encoding="utf-8") as handle:
                json.dump({"items": {"NVDA_US_EQ": [[1700000000000, 250.0]]}}, handle)
            items_live = mgr.get_remote_items("TOTAL", 0)
            self.assertEqual(items_live[0]["venues_usd"]["t212"], 121.5)
            self.assertEqual(items_live[0]["portfolio_total_usd"], 3071.5)

            from cachepaths import compute_venue_nav_breakdown
            with open(vb_path, encoding="utf-8") as handle:
                vb_loaded = json.load(handle)
            eq_only = compute_venue_nav_breakdown(venue_balances_state=vb_loaded, scope="equities")
            self.assertEqual(eq_only["venues_usd"], {"t212": 100.0})
            cr_only = compute_venue_nav_breakdown(venue_balances_state=vb_loaded, scope="crypto")
            self.assertEqual(cr_only["venues_usd"], {"kraken": 950.0})
            cr_remarked = compute_venue_nav_breakdown(
                positions=[
                    {
                        "venue": "kraken",
                        "symbol": "XBTUSD",
                        "free_base": 0.01,
                        "value_usd": 800.0,
                        "locked_base_qty": 0.005,
                        "locked_base_usd": 400.0,
                    }
                ],
                live_prices={"BTCUSDC": 100000.0},
            )
            self.assertEqual(cr_remarked["venues_usd"], {"kraken": 1500.0})


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CacheCurrentPriceManager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheCurrentPriceManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cm.CacheFactory.remove("CurrentPrice")

    def tearDown(self):
        cm.CacheFactory.remove("CurrentPrice")

    def _make(self, price=50000.0, provider_names=None):
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = price
        fname = _tmp_file(self.tmp, "cache_current_price.json")
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
            mgr.on_items_update("BTC", [[int(before * 1000), 55000.0]])
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
            self.assertEqual(mgr._ws_last_event_ts, 0.0)
            self.assertFalse(mgr._ws_is_healthy("BTC"))
            mgr.on_items_update("BTC", [[int(time.time() * 1000), 0]])
            self.assertEqual(mgr._ws_last_event_ts, 0.0)
            self.assertFalse(mgr._ws_is_healthy("BTC"))

        with self.subTest("snapshot mode replaces"):
            mgr, _ = self._make()
            now_ms = int(time.time() * 1000)
            mgr.on_items_update("BTC", [[now_ms, 50000.0]])
            mgr.on_items_update("BTC", [[now_ms + 1, 60000.0]])
            with mgr.lock:
                entries = mgr.cache["BTC"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0][1], 60000.0)

        with self.subTest("out of order delayed ws frame does not overwrite newer timestamp"):
            mgr, _ = self._make()
            now_ms = int(time.time() * 1000)
            mgr.on_items_update("BTC", [[now_ms, 60000.0]])
            mgr.on_items_update("BTC", [[now_ms - (mgr.STALE_THRESHOLD_MS + 1000), 49000.0]])
            with mgr.lock:
                entries = mgr.cache["BTC"]
            self.assertEqual(entries[0], [now_ms, 60000.0])

    # ── get_price / get_price_value ───────────────────────────────────────────

    def test_fresh_price_read_apis_use_cached_value(self):
        mgr, api_mock = self._make()
    def test_get_price_and_value_cache_and_http_fallback(self):
        with self.subTest("fresh price read apis use cached value"):
            mgr, api_mock = self._make()
            mgr.on_items_update("BTC", [[int(time.time() * 1000), 55000.0]])
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

        with self.subTest("get_price stale forces http"):
            mgr, api_mock = self._make()
            api_mock.get_current_price.return_value = 65000.0
            with mgr.lock:
                mgr.cache["BTC"] = [[0, 50000.0]]   # A zero timestamp is always stale.
            api_mock.get_current_price.reset_mock()
            entry = mgr.get_price("BTC")
            api_mock.get_current_price.assert_called()
            self.assertEqual(entry[1], 65000.0)

        with self.subTest("get_price missing forces http"):
            mgr, api_mock = self._make()
            api_mock.get_current_price.return_value = 70000.0
            with mgr.lock:
                mgr.cache = {}
            api_mock.get_current_price.reset_mock()
            entry = mgr.get_price("BTC")
            api_mock.get_current_price.assert_called()
            self.assertEqual(entry[1], 70000.0)

        with self.subTest("get_price_value none if unavailable"):
            mgr, api_mock = self._make()
            api_mock.get_current_price.return_value = None
            with mgr.lock:
                mgr.cache = {}
            val = mgr.get_price_value("BTC")
            self.assertIsNone(val)

        with self.subTest("get_price does not return stale row after http failure"):
            mgr, api_mock = self._make()
            for bad_return in (None, -1.0, 0.0, float("nan"), float("inf"), True):
                api_mock.get_current_price.return_value = bad_return
                stale = [0, 50000.0]
                with mgr.lock:
                    mgr.cache["BTC"] = [stale]

                self.assertIsNone(mgr.get_price("BTC"))
                api_mock.get_current_price.assert_called()
                with mgr.lock:
                    self.assertEqual([stale], mgr.cache["BTC"])
            with self.assertRaises(ValueError):
                mgr.get_cached_price_observation("BTC", True)
            with mgr.lock:
                mgr.cache["BTC"] = [[int(time.time() * 1000), True]]
            self.assertIsNone(mgr.get_cached_price_observation("BTC", 60.0))

    def test_provider_binding_and_cached_observation(self):
        with self.subTest("unbound remote price keeps implicit provider routing"):
            mgr, api_mock = self._make(price=51000.0)
            api_mock.get_current_price.reset_mock()
            rows = mgr.get_remote_items("BTC", None)
            self.assertEqual(rows[0][1], 51000.0)
            call = api_mock.get_current_price.call_args
            self.assertEqual(call.kwargs["symbol"], "BTC")
            self.assertIsNone(call.kwargs.get("provider_name"))
            self.assertIsNone(mgr.provider_name_for("BTC"))

        with self.subTest("bound remote price uses explicit provider"):
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

        with self.subTest("conflicting provider binding is rejected"):
            mgr, _ = self._make(provider_names={"BTC": "Kraken"})
            mgr.bind_provider("BTC", "kraken")
            with self.assertRaisesRegex(ValueError, r"conflicting provider"):
                mgr.bind_provider("BTC", "Hyperliquid")

        with self.subTest("cached price observation never fetches"):
            mgr, api_mock = self._make()
            mgr.on_items_update("BTC", [[int(time.time() * 1000), 55000.0]])
            api_mock.get_current_price.reset_mock()

            observed_at, price = mgr.get_cached_price_observation(
                "BTC", max_age_sec=10.0)

            self.assertAlmostEqual(observed_at, time.time(), delta=1.0)
            self.assertEqual(price, 55000.0)
            api_mock.get_current_price.assert_not_called()

            with mgr.lock:
                mgr.cache["BTC"][0][0] = int((time.time() - 20.0) * 1000)
            self.assertIsNone(
                mgr.get_cached_price_observation("BTC", max_age_sec=10.0))
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
            mgr.on_items_update("BTC", [[int(time.time() * 1000), 62000.0]])
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
            mgr.on_items_update("BTC", [[int(time.time() * 1000), 1.0]])
            good.on_price_update.assert_called_once()

        with self.subTest("unsubscribed not notified"):
            mgr, _ = self._make()
            sub = MagicMock()
            mgr.subscribe_price(sub)
            mgr.unsubscribe_price(sub)
            mgr.on_items_update("BTC", [[int(time.time() * 1000), 1.0]])
            sub.on_price_update.assert_not_called()

    # ── WS health ─────────────────────────────────────────────────────────────

    def test_ws_health_from_event_age(self):
        mgr, _ = self._make()
        for label, timestamp, expected in (("recent", time.time(), True), ("old", 0.0, False)):
            with self.subTest(case=label):
                mgr._ws_last_event_ts = timestamp
                self.assertEqual(mgr._ws_is_healthy(), expected)

        with self.subTest(case="per_symbol_ws_health_polls_only_stale_symbols"):
            api_mock = MagicMock()
            api_mock.get_current_price.side_effect = (
                lambda symbol, **kwargs: 3200.0 if symbol == "ETH" else 60000.0
            )
            fname = _tmp_file(self.tmp, "cache_current_price_multi.json")
            multi = cm.CacheCurrentPriceManager(
                sync_ts=9999,
                symbols=["BTC", "ETH"],
                filename=fname,
                ws_manager=None,
                api_client=api_mock,
                market_api=api_mock,
            )
            api_mock.get_current_price.reset_mock()
            now_ms = int(time.time() * 1000)
            multi.on_items_update("BTC", [[now_ms, 59000.0]])
            self.assertTrue(multi._ws_is_healthy("BTC"))
            self.assertFalse(multi._ws_is_healthy("ETH"))
            self.assertFalse(multi._ws_is_healthy())
            self.assertTrue(multi._should_poll())

            multi.query_remote_and_update_cache()
            api_mock.get_current_price.assert_called_once_with(symbol="ETH")
            self.assertEqual(multi.cache["BTC"][0][1], 59000.0)
            self.assertEqual(multi.cache["ETH"][0][1], 3200.0)

            multi.on_items_update("ETH", [[now_ms + 1, 3210.0]])
            self.assertTrue(multi._ws_is_healthy("ETH"))
            self.assertTrue(multi._ws_is_healthy())
            self.assertFalse(multi._should_poll())

    # -- Persistence. ----------------------------------------------------------

    def test_persistence_reload(self):
        mgr, api_mock = self._make()
        mgr.save_state = True
        mgr.on_items_update("BTC", [[int(time.time() * 1000), 58000.0]])
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
        from binance_api import bapi_ws_bridge

        self.bws = bapi_ws_bridge
        self.bws._ws_available = False
        self.bws._ws_last_event_ts = 0.0
        self.bws._ws_is_healthy = False

    def test_ws_health_lifecycle_and_policies(self):
        with self.subTest("mark ws available"):
            for available in (True, False):
                with self.subTest(available=available):
                    self.bws.mark_ws_available(available)
                    with self.bws._ws_health_lock:
                        self.assertEqual(self.bws._ws_available, available)

        with self.subTest("ws event and unhealthy lifecycle"):
            self.bws.mark_ws_event_received()
            with self.bws._ws_health_lock:
                self.assertTrue(self.bws._ws_is_healthy)
                self.assertGreater(self.bws._ws_last_event_ts, 0)
            self.bws.mark_ws_unhealthy()
            with self.bws._ws_health_lock:
                self.assertFalse(self.bws._ws_is_healthy)

        with self.subTest("polling fallback policies"):
            cases = (
                ("ws-mode-off", False, True, "CacheOrderManager"),
                ("unmanaged", True, True, "CacheSparsePriceManager"),
                ("ws-unavailable", True, False, "CacheOrderManager"),
            )
            try:
                for label, ws_only, available, manager_name in cases:
                    with self.subTest(case=label):
                        self.bws.WS_ONLY_MODE = ws_only
                        self.bws._ws_available = available
                        self.assertTrue(self.bws.should_poll_for_manager(manager_name))
            finally:
                self.bws.WS_ONLY_MODE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CacheFactory / get_cache_manager
# ═══════════════════════════════════════════════════════════════════════════════
class TestCacheFactory(unittest.TestCase):

    def setUp(self):
        # Reset singletons.
        cm.CacheFactory.shutdown_all()

    def tearDown(self):
        cm.CacheFactory.shutdown_all()

    def test_factory_lifecycle_promotion_and_identity(self):
        def _reset():
            cm.CacheFactory.shutdown_all()

        with self.subTest("read only creation can be promoted to periodic sync"):
            _reset()
            with patch.object(cm.CacheTradeManager, "periodic_sync") as periodic, \
                 patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]):
                first = cm.CacheFactory.get("BinanceTrade", symbols=["BTC"])
                periodic.assert_not_called()
                second = cm.CacheFactory.get("BinanceTrade", symbols=["BTC"], data_sync_mode="live_reader")
                self.assertIs(first, second)
                periodic.assert_called_once()
            _reset()

        with self.subTest("unknown name raises"):
            _reset()
            with self.assertRaises(ValueError):
                cm.CacheFactory.get("NonExistent")

        with self.subTest("returns same instance twice"):
            _reset()
            with patch("cacheManager.get_current_price_manager", return_value=MagicMock()):
                i1 = cm.CacheFactory.get("CurrentPrice", symbols=["BTC"])
                i2 = cm.CacheFactory.get("CurrentPrice", symbols=["BTC"])
            self.assertIs(i1, i2)

        with self.subTest("factory and current price getter share identity factory first"):
            _reset()
            with patch.object(
                    cm.CacheCurrentPriceManager, "load_state", return_value=None):
                factory_manager = cm.CacheFactory.get(
                    "CurrentPrice", symbols=["BTC"])
                singleton_manager = cm.get_current_price_manager(
                    symbols=["BTC"])
            self.assertIs(factory_manager, singleton_manager)
            self.assertEqual(
                singleton_manager.sync_ts,
                cm.CURRENT_PRICE_SYNC_INTERVAL_SEC,
            )

        with self.subTest("factory and current price getter share identity getter first"):
            _reset()
            with patch.object(
                    cm.CacheCurrentPriceManager, "load_state", return_value=None), \
                 patch.object(
                    cm.CacheCurrentPriceManager, "periodic_sync") as periodic:
                singleton_manager = cm.get_current_price_manager(
                    symbols=["BTC"], sync_ts=0.8)
                factory_manager = cm.CacheFactory.get(
                    "CurrentPrice", symbols=["BTC"])
                periodic.assert_not_called()
                promoted_manager = cm.CacheFactory.get(
                    "CurrentPrice", symbols=["BTC"], data_sync_mode="live_reader")
            self.assertIs(factory_manager, singleton_manager)
            self.assertIs(promoted_manager, singleton_manager)
            self.assertEqual(factory_manager.sync_ts, 0.8)
            periodic.assert_called_once_with(0.8, False)
            _reset()

        with self.subTest("concurrent first get constructs once and promotes sync"):
            _reset()
            barrier = threading.Barrier(2)
            results = []
            load_calls = 0

            def slow_load_state(_self):
                nonlocal load_calls
                load_calls += 1
                time.sleep(0.03)

            def worker(sync_mode):
                barrier.wait(timeout=2.0)
                results.append(
                    cm.CacheFactory.get(
                        "BinanceOrder", symbols=["BTC"], data_sync_mode=sync_mode
                    )
                )

            with patch.object(
                    cm.CacheOrderManager, "load_state", autospec=True,
                    side_effect=slow_load_state), \
                 patch.object(
                    cm.CacheOrderManager, "periodic_sync") as periodic:
                threads = [
                    threading.Thread(target=worker, args=("disk_reader",)),
                    threading.Thread(target=worker, args=("live_reader",)),
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5.0)
            self.assertEqual(len(results), 2)
            self.assertIs(results[0], results[1])
            self.assertEqual(load_calls, 1)
            periodic.assert_called_once_with(cm.ORDER_SYNC_INTERVAL_SEC, False)
            _reset()

        with self.subTest("disk_reader_never_queries_remote_when_cache_missing_or_empty"):
            _reset()
            empty_dir = tempfile.mkdtemp()
            with patch("utils.cache_path", side_effect=lambda f: os.path.join(empty_dir, f)), \
                 patch.object(cm.CacheTradeManager, "query_remote_and_update_cache") as trade_remote, \
                 patch.object(cm.CacheOrderManager, "query_remote_and_update_cache") as order_remote, \
                 patch.object(cm.CacheCurrentPriceManager, "query_remote_and_update_cache") as price_remote:
                trade_mgr = cm.CacheFactory.get("BinanceTrade", symbols=["BTC"])
                order_mgr = cm.CacheFactory.get("BinanceOrder", symbols=["BTC"])
                price_mgr = cm.CacheFactory.get("CurrentPrice", symbols=["BTC"])
                self.assertTrue(trade_mgr.read_only)
                self.assertTrue(order_mgr.read_only)
                self.assertTrue(price_mgr.read_only)
                trade_remote.assert_not_called()
                order_remote.assert_not_called()
                price_remote.assert_not_called()
            _reset()

    def test_failed_sync_start_retry_reuses_published_managers(self):
        """A periodic_sync failure must not make the next call build duplicate managers."""
        with patch.object(cm.CacheOrderManager, "load_state", return_value=None), \
             patch.object(
                 cm.CacheOrderManager, "periodic_sync",
                 side_effect=[RuntimeError("can't start new thread"), None],
             ) as periodic:
            with self.assertRaises(RuntimeError):
                cm.CacheFactory.get("BinanceOrder", symbols=["BTC"], data_sync_mode="live_reader")
            first = cm.CacheFactory._instances.get("BinanceOrder")
            self.assertIsNotNone(first)
            self.assertNotIn("BinanceOrder", cm.CacheFactory._sync_started)
            second = cm.CacheFactory.get("BinanceOrder", symbols=["BTC"], data_sync_mode="live_reader")
        self.assertIs(first, second)
        self.assertEqual(periodic.call_count, 2)
        self.assertIn("BinanceOrder", cm.CacheFactory._sync_started)

    def test_price_factories_return_per_symbol_managers_and_filenames(self):
        cur_mock = MagicMock()
        cur_mock.get_price.return_value = [int(time.time() * 1000), 50000.0]
        cases = (
            ("SparsePrice", ["BTC", "ETH"], "cache_sparse_price_BTC.jsonl"),
            ("24Price", ["BTC"], "cache_24_price_BTC.json"),
            ("365Price", ["BTC"], "cache_365_price_BTC.jsonl"),
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
            ("BinanceTrade", cm.CacheTradeManager),
            ("BinanceOrder", cm.CacheOrderManager),
            ("CurrentPrice", cm.CacheCurrentPriceManager),
        )
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]), \
             patch.object(cm.CacheCurrentPriceManager, "load_state", return_value=None):
            for name, expected_class in cases:
                with self.subTest(factory=name):
                    result = cm.CacheFactory.get(name, symbols=["BTC"])
                    self.assertIsInstance(result, expected_class)

    def test_get_cache_manager_delegates_to_factory(self):
        with patch("binance_api.bapi_allorders.paginate_my_trades", return_value=[]):
            r1 = cm.get_cache_manager("BinanceTrade", symbols=["BTC"])
            r2 = cm.CacheFactory.get("BinanceTrade", symbols=["BTC"])
        self.assertIs(r1, r2)

    def test_factory_default_symbols_use_ssot_price_tracking(self):
        tracked = (["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"], ["HYPEUSD"], {"HYPEUSD": "kraken"})
        with patch.object(cm, "tracked_price_instruments", return_value=tracked), \
             patch.object(cm.CacheCurrentPriceManager, "load_state", return_value=None), \
             patch.object(cm.CacheLongTrendManager, "load_state", return_value=None):
            cm.CacheFactory.remove("CurrentPrice")
            cm.CacheFactory.remove("LongTrend")
            cpm = cm.CacheFactory.get("CurrentPrice")
            plt = cm.CacheFactory.get("LongTrend")
            self.assertEqual(cpm.symbols, ["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"])
            self.assertEqual(cpm._provider_names.get("HYPEUSD"), "kraken")
            self.assertEqual(plt.symbols, ["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"])


# ═══════════════════════════════════════════════════════════════════════════════
# 11. get_current_price_manager — singleton
# ═══════════════════════════════════════════════════════════════════════════════
class TestGetCurrentPriceManagerSingleton(unittest.TestCase):

    def setUp(self):
        cm.CacheFactory.remove("CurrentPrice")

    def tearDown(self):
        cm.CacheFactory.remove("CurrentPrice")

    def test_get_current_price_manager_singleton_and_ws_wiring(self):
        with self.subTest("returns single cache current price manager"):
            m1 = cm.get_current_price_manager(symbols=["BTC"])
            m2 = cm.get_current_price_manager(symbols=["BTC"])
            self.assertIsInstance(m1, cm.CacheCurrentPriceManager)
            self.assertIs(m1, m2)

        with self.subTest("ws manager subscribed if provided"):
            ws = MagicMock()
            mgr = cm.get_current_price_manager(
                ws_manager=ws, symbols=["BTC"],
            )
            ws.subscribe.assert_called_once_with(mgr)

        with self.subTest("price series get_remote_items does not start sync on disk_reader singleton"):
            mgr = cm.get_current_price_manager(symbols=["BTC"])
            self.assertIsNone(mgr.thread)
            now_ms = int(time.time() * 1000)
            with mgr.lock:
                mgr.cache["BTC"] = [[now_ms, 95000.0]]
            price_base = object.__new__(cm.CachePriceManagerBase)
            price_base.lock = cm.threading.RLock()
            price_base.cache = {}
            items = price_base.get_remote_items("BTC", 0)
            self.assertEqual(items, [[now_ms, 95000.0]])
            self.assertIsNone(mgr.thread)

        with self.subTest("initialized singletons and periodic_sync use lock-free fast path"):
            class _ForbidLock:
                def __enter__(self):
                    raise AssertionError("fast path must not acquire lock")

                def __exit__(self, *args):
                    return False

            previous_short_trend = cm.CacheFactory._instances.get("ShortTrend")
            cm.CacheFactory._instances.pop("ShortTrend", None)
            with patch.object(cm.CacheCurrentPriceManager, "load_state", return_value=None), \
                 patch.object(cm.CacheTradeManager, "load_state", return_value=None):
                mgr = cm.get_current_price_manager(symbols=["BTC"], data_sync_mode="live_reader")
                short_mgr = cm.get_short_trend_manager(symbols=["BTC"])
                trade_mgr = cm.CacheFactory.get("BinanceTrade", symbols=["BTC"])
                try:
                    with patch.object(cm.CacheFactory, "_lock", _ForbidLock()), \
                         patch.object(mgr, "lock", _ForbidLock()):
                        self.assertIs(cm.get_current_price_manager(), mgr)
                        self.assertIs(mgr.periodic_sync(save_state=False), mgr.thread)
                        self.assertIs(cm.get_short_trend_manager(), short_mgr)
                        self.assertIs(
                            cm.CacheFactory.get("BinanceTrade"),
                            trade_mgr,
                        )
                finally:
                    mgr.shutdown()
                    cm.CacheFactory.remove("BinanceTrade")
                    short_mgr.shutdown()
                    if previous_short_trend is not None:
                        cm.CacheFactory._instances["ShortTrend"] = previous_short_trend
                    else:
                        cm.CacheFactory._instances.pop("ShortTrend", None)

    def test_every_factory_manager_accepts_read_only(self):
        """CacheFactory passes read_only to every class it builds, including PriceMulti."""
        import inspect
        from market_monitor.pricefetcher import CacheAllPriceFetcherManager

        classes = [config["class"] for config in cm.CacheFactory._CONFIG.values()]
        classes.append(CacheAllPriceFetcherManager)
        for manager_class in classes:
            with self.subTest(manager=manager_class.__name__):
                self.assertIn(
                    "read_only",
                    inspect.signature(manager_class.__init__).parameters,
                )


class TestRefreshSymbolInCache(unittest.TestCase):
    """Test the WebSocket handler helper that refreshes one symbol."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_refresh_symbol_in_cache(self):
        from binance_api import bapi_ws_bridge

        with self.subTest("refresh single symbol"):
            fname = _tmp_file(self.tmp)
            remote = {"SYM": [[int(time.time()*1000), 42.0]]}
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items=remote)
            with mgr.lock:
                mgr.cache["SYM"] = []
            bapi_ws_bridge.refresh_symbol_in_cache(mgr, "SYM")
            with mgr.lock:
                self.assertIn(42.0, [e[1] for e in mgr.cache.get("SYM", [])])

        with self.subTest("refresh missing symbol no crash"):
            fname = _tmp_file(self.tmp, "x.json")
            mgr = ConcreteTestManager(9999, ["SYM"], fname, remote_items={})
            bapi_ws_bridge.refresh_symbol_in_cache(mgr, "NOPE")   # Must not raise.


class TestFactorySingletonWarning(unittest.TestCase):
    """The named singleton ignores different symbols on later calls."""

    def setUp(self):
        cm.CacheFactory.remove("AssetValue")

    def tearDown(self):
        cm.CacheFactory.remove("AssetValue")

    def test_factory_singleton_warning_behavior(self):
        with self.subTest("same instance returned and warns"):
            m1 = cm.get_cache_manager("AssetValue", symbols=["TOTAL"])
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m2 = cm.get_cache_manager("AssetValue", symbols=["OTHER"])
            self.assertIs(m1, m2)                       # Same instance.
            self.assertIn("IGNORED", buf.getvalue())    # Warning was emitted.

        with self.subTest("no warning same symbols"):
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

    def _count_lines(self, path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def test_append_jsonl_lifecycle_and_dedup(self):
        with self.subTest("append writes only delta"):
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

        with self.subTest("load jsonl rebuilds cache"):
            fname = os.path.join(self.tmp, "t2.jsonl")
            w = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            w.save_state = True
            with w.lock:
                w.cache["SYM"] = [{"id": 1}, {"id": 2}]
            w.save_state_to_file_if_enabled()
            # Another manager loads the JSONL at startup.
            r = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            self.assertEqual(r.cache.get("SYM"), [{"id": 1}, {"id": 2}])

        with self.subTest("compact dedups"):
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

        with self.subTest("save to file unconditional vs if enabled"):
            fname = _tmp_file(self.tmp, "split.json")
            m = ConcreteTestManager(9999, ["SYM"], fname)
            m.save_state = False
            with m.lock:
                m.cache["SYM"] = [[1, 1.0]]
            m.save_state_to_file_if_enabled()       # save_state=False does not write.
            self.assertFalse(os.path.exists(fname))
            m.save_state_to_file()                  # Unconditional write.
            self.assertTrue(os.path.exists(fname))

        with self.subTest("compact rewrites"):
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

        with self.subTest("compact rollback when write_meta fails and shortened append compacts"):
            fname = os.path.join(self.tmp, "meta_fail.jsonl")
            m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            m.save_state = True
            with m.lock:
                m.cache["SYM"] = [{"id": 1}, {"id": 2}]
            m.save_state_to_file_if_enabled()
            self.assertEqual(self._count_lines(fname), 2)
            self.assertEqual(m._persisted_counts.get("SYM"), 2)

            with m.lock:
                m.cache["SYM"] = [{"id": 1}, {"id": 1}, {"id": 2}]
            with patch.object(m, "_write_meta", return_value=False):
                self.assertFalse(m.compact_jsonl())
            self.assertEqual(self._count_lines(fname), 2)
            self.assertEqual(m._persisted_counts.get("SYM"), 2)
            self.assertEqual(len(m.cache["SYM"]), 3)

            with m.lock:
                m.cache["SYM"] = [{"id": 2}]
            self.assertTrue(m._save_jsonl_append())
            self.assertEqual(self._count_lines(fname), 1)
            self.assertEqual(m._persisted_counts.get("SYM"), 1)

    def test_maintain_pruning_and_rotation(self):
        tmp = tempfile.mkdtemp(dir=self.tmp)
        with self.subTest("maintain prunes old entries"):
            fname = os.path.join(tmp, "p.jsonl")
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

        with self.subTest("retention normalizes unix seconds"):
            now_sec = int(time.time())
            self.assertEqual(
                ConcreteTestManager._entry_timestamp_ms({"timestamp": now_sec}),
                now_sec * 1000,
            )

        with self.subTest("rotation archives and keeps latest"):
            fname = os.path.join(tmp, "r.jsonl")
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
            archives = [f for f in os.listdir(tmp) if "r.jsonl" in f and ".archive" in f]
            self.assertTrue(archives)
            with m.lock:
                self.assertEqual(len(m.cache["SYM"]), 10)
                self.assertEqual(m.cache["SYM"][-1], [now_ms + 99, 99.0])

        with self.subTest("rotation archive has FULL history"):
            fname = os.path.join(tmp, "full.jsonl")
            m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            m.save_state = True
            m.MAX_FILE_BYTES = 1
            now = int(time.time() * 1000)
            with m.lock:
                m.cache["SYM"] = [[now + i, float(i)] for i in range(100)]
            m.save_state_to_file_if_enabled()
            self.assertEqual(self._count_lines(fname), 100)
            m.maintain_append_persist()
            archive = [os.path.join(tmp, f) for f in os.listdir(tmp) if "full.jsonl" in f and ".archive" in f][0]
            self.assertEqual(self._count_lines(archive), 100)   # Complete history.
            self.assertEqual(self._count_lines(fname), 10)       # Newest ten percent.
            # The compressed archive remains a complete, independently recoverable JSONL stream.
            with gzip.open(archive, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[-1]["i"], [now + 99, 99.0])

        with self.subTest("rotation failure restores active file and memory state"):
            fname = os.path.join(tmp, "rot_fail.jsonl")
            m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            m.save_state = True
            m.MAX_FILE_BYTES = 1
            m.ROTATE_KEEP_FRACTION = 0.10
            now = int(time.time() * 1000)
            with m.lock:
                m.cache["SYM"] = [[now + i, float(i)] for i in range(50)]
            m.save_state_to_file_if_enabled()
            self.assertEqual(self._count_lines(fname), 50)
            with patch.object(m, "compact_jsonl", return_value=False):
                m.maintain_append_persist()
            self.assertTrue(os.path.exists(fname))
            self.assertEqual(self._count_lines(fname), 50)
            self.assertEqual(len(m.cache["SYM"]), 50)
            self.assertEqual(m._persisted_counts.get("SYM"), 50)
            self.assertEqual(
                [f for f in os.listdir(tmp) if "rot_fail.jsonl" in f and ".archive" in f],
                [],
            )

        with self.subTest("legacy full json is migrated without deleting source"):
            legacy = os.path.join(tmp, "cache_trade.json")
            target = legacy + "l"
            now = int(time.time() * 1000)
            _write_cache_file(legacy, {"SYM": [[now, 1.0]]}, {"SYM": now})
            manager = ConcreteTestManager(9999, ["SYM"], target, append_persist=True)
            self.assertTrue(os.path.exists(legacy))
            self.assertTrue(os.path.exists(target))
            self.assertEqual(manager.cache["SYM"], [[now, 1.0]])

    def test_maintain_edge_cases_and_corruption(self):
        tmp = tempfile.mkdtemp(dir=self.tmp)
        with self.subTest("maintain noop leaves file intact"):
            fname = os.path.join(tmp, "intact.jsonl")
            m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            m.save_state = True
            now = int(time.time() * 1000)
            with m.lock:
                m.cache["SYM"] = [[now, 1.0], [now + 1, 2.0]]
            m.save_state_to_file_if_enabled()
            before = open(fname).read()
            m.maintain_append_persist()
            self.assertEqual(open(fname).read(), before)         # File unchanged.
            self.assertEqual([f for f in os.listdir(tmp) if ".archive" in f], [])  # No archive.
            self.assertEqual(len(m.cache["SYM"]), 2)             # Data intact.

        with self.subTest("maintain missing file no crash"):
            fname = os.path.join(tmp, "nofile.jsonl")
            m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=True)
            m.maintain_append_persist()   # A missing file must not crash.
            self.assertEqual([f for f in os.listdir(tmp) if ".archive" in f], [])

        with self.subTest("maintain noop when not append persist"):
            fname = os.path.join(tmp, "fullrw.json")
            now_ms = int(time.time() * 1000)
            _write_cache_file(fname, {"SYM": [[now_ms, 1.0]]})
            m = ConcreteTestManager(9999, ["SYM"], fname, append_persist=False)
            before = open(fname).read()
            m.maintain_append_persist()   # Fresh data leaves nothing to prune.
            self.assertEqual(open(fname).read(), before)
            self.assertEqual(len(m.cache["SYM"]), 1)

        with self.subTest("maintain prunes old entries even when not append persist"):
            fname = os.path.join(tmp, "fullrw_prune.json")
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
            m.save_state = True
            m.save_state_to_file_if_enabled()
            with open(fname) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["items"]["SYM"], [[now_ms, 2.0]])   # Also on disk.

        with self.subTest("prune keeps file and recent"):
            fname = os.path.join(tmp, "prune.jsonl")
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

        with self.subTest("skips corrupt lines"):
            fname = os.path.join(tmp, "t4.jsonl")
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

    def test_mem_file_resync_and_overwrite_rules(self):
        with self.subTest("refuses overwrite with older"):
            fname = _tmp_file(self.tmp, "g.json")
            # Writer A writes new data with a high fetch time.
            a = self._mgr(fname)
            with a.lock:
                a.cache["SYM"] = [[1, 1.0]]
                a.fetchtime_per_symbol["SYM"] = 2000   # New.
            a.save_state_to_file_if_enabled()
            # Writer B has stale data and must not overwrite it.
            b = ConcreteTestManager(9999, ["SYM"], fname)
            b.save_state = True
            with b.lock:
                b.cache["SYM"] = [[9, 9.0]]
                b.fetchtime_per_symbol["SYM"] = 1000   # Older.
            b.save_state_to_file_if_enabled()
            # The file retains writer A's data.
            data = json.load(open(fname))
            self.assertEqual(data["items"]["SYM"], [[1, 1.0]])

        with self.subTest("allows overwrite with newer"):
            fname = _tmp_file(self.tmp, "g2.json")
            a = self._mgr(fname)
            with a.lock:
                a.cache["SYM"] = [[1, 1.0]]
                a.fetchtime_per_symbol["SYM"] = 1000
            a.save_state_to_file_if_enabled()
            b = ConcreteTestManager(9999, ["SYM"], fname)
            b.save_state = True
            with b.lock:
                b.cache["SYM"] = [[9, 9.0]]
                b.fetchtime_per_symbol["SYM"] = 3000   # Newer, so overwrite.
            b.save_state_to_file_if_enabled()
            data = json.load(open(fname))
            self.assertEqual(data["items"]["SYM"], [[9, 9.0]])

        with self.subTest("resync reloads when file newer"):
            fname = _tmp_file(self.tmp, "r.json")
            # Process A writes new data to disk.
            a = self._mgr(fname)
            with a.lock:
                a.cache["SYM"] = [[5, 5.0]]
                a.fetchtime_per_symbol["SYM"] = 5000
            a.save_state_to_file_if_enabled()
            # Process B has stale memory, so resync must reload it.
            b = ConcreteTestManager(9999, ["SYM"], fname)
            with b.lock:
                b.cache["SYM"] = [[1, 1.0]]
                b.fetchtime_per_symbol["SYM"] = 1000
            b.resync_mem_file()
            with b.lock:
                self.assertEqual(b.cache["SYM"], [[5, 5.0]])   # Reloaded from the file.


class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_atomic_write_behaviors(self):
        with self.subTest("atomic write json roundtrip"):
            p = os.path.join(self.tmp, "a.json")
            cm.atomic_write_json(p, {"x": 1, "y": [2, 3]})
            with open(p) as f:
                self.assertEqual(json.load(f), {"x": 1, "y": [2, 3]})
            self.assertFalse(os.path.exists(p + ".tmp"))   # Temporary file cleaned up.

        with self.subTest("atomic write cleanup on error"):
            p = os.path.join(self.tmp, "b.json")
            with self.assertRaises(ValueError):
                with cm.atomic_text_writer(p) as f:
                    f.write("partial")
                    raise ValueError("boom")
            self.assertFalse(os.path.exists(p))            # Target file untouched.
            self.assertFalse(os.path.exists(p + ".tmp"))   # Temporary file removed.

        with self.subTest("atomic write preserves old on error"):
            p = os.path.join(self.tmp, "c.json")
            cm.atomic_write_json(p, {"v": "old"})
            with self.assertRaises(ValueError):
                with cm.atomic_text_writer(p) as f:
                    f.write("new-incomplete")
                    raise ValueError("boom")
            with open(p) as f:
                self.assertEqual(json.load(f), {"v": "old"})   # Previous content remains intact.

        with self.subTest("legacy filename migration and sidecars"):
            mig_dir = tempfile.mkdtemp(dir=self.tmp)
            cases = [
                ("cache_currentprice.json", "cache_current_price.json"),
                ("cache_instant_trend.json", "cache_short_trend.json"),
                ("cache_price_long_trend.json", "cache_long_trend.json"),
                ("cache_24price_BTCUSDC.json", "cache_24_price_BTCUSDC.json"),
                ("cache_365price_BTCUSDC.jsonl", "cache_365_price_BTCUSDC.jsonl"),
                ("cache_price_sparse_BTCUSDC.jsonl", "cache_sparse_price_BTCUSDC.jsonl"),
            ]
            for old_name, new_name in cases:
                old_p = os.path.join(mig_dir, old_name)
                new_p = os.path.join(mig_dir, new_name)
                with open(old_p, "w", encoding="utf-8") as fh:
                    fh.write('{"migrated": true}\n')
                with open(old_p + ".meta", "w", encoding="utf-8") as fh:
                    fh.write('{"meta": true}\n')
                cm.migrate_legacy_filename_if_needed(new_p)
                self.assertTrue(os.path.exists(new_p), f"Expected {new_name} to exist")
                self.assertTrue(os.path.exists(new_p + ".meta"), f"Expected {new_name}.meta to exist")
                self.assertFalse(os.path.exists(old_p), f"Expected {old_name} to be moved")

    def test_factory_shutdown_all_clears_short_trend_and_ws_bridge_rollback_on_start_error(self):
        from binance_api import bapi_ws, bapi_ws_bridge

        st = cm.CacheFactory.get("ShortTrend", symbols=["BTCUSDC"])
        self.assertIs(cm.CacheFactory._instances.get("ShortTrend"), st)
        self.assertTrue(cm.CacheFactory.shutdown_all())
        self.assertIsNone(cm.CacheFactory._instances.get("ShortTrend"))

        bapi_ws_bridge._ws_bridge = None
        failing_stream = MagicMock()
        failing_stream.start.side_effect = RuntimeError("stream start failed")
        ok_stream = MagicMock()
        with patch.object(
            bapi_ws,
            "BinanceAccountStream",
            side_effect=[failing_stream, ok_stream],
        ):
            with self.assertRaises(RuntimeError):
                bapi_ws_bridge.enable_real_ws_event_sync()
            self.assertIsNone(bapi_ws_bridge._ws_bridge)
            started = bapi_ws_bridge.enable_real_ws_event_sync()
            self.assertIs(started, ok_stream)
            ok_stream.start.assert_called_once()
        bapi_ws_bridge._ws_bridge = None

        # Non-dict JSON root in primary file recovers from .previous on load_state / _reload_from_disk
        p = os.path.join(self.tmp, "cache_non_dict_root.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("[]")
        with open(p + ".previous", "w", encoding="utf-8") as fh:
            json.dump({"items": {"BTC": [[1000, 50000.0]]}, "fetchtime": None}, fh)
        c24 = cm.Cache24PriceManager(60, ["BTC"], p, read_only=True)
        self.assertEqual(c24.cache.get("BTC"), [[1000, 50000.0]])
        self.assertEqual(c24.fetchtime_per_symbol, {})
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("null")
        with open(p + ".previous", "w", encoding="utf-8") as fh:
            json.dump({"items": {"BTC": [[2000, 51000.0]]}, "fetchtime": None}, fh)
        c24._reload_from_disk()
        self.assertEqual(c24.cache.get("BTC"), [[2000, 51000.0]])
        self.assertEqual(c24.fetchtime_per_symbol, {})

        # bapi_ws_normalize rejects boolean IDs and falls back when l/L are NaN/Inf
        from binance_api import bapi_ws_normalize
        import cacheintegrity

        self.assertIsNone(
            bapi_ws_normalize.parse_order_execution_report(
                {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": True, "i": 10, "l": "1", "L": "50000", "S": "BUY"}
            )
        )
        self.assertIsNone(
            bapi_ws_normalize.parse_trade_execution_report(
                {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": True, "i": 10, "l": "1", "L": "50000", "S": "BUY"}
            )
        )
        sym_o, ord_it = bapi_ws_normalize.parse_order_execution_report(
            {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": 7, "i": 10, "l": "nan", "z": "0.5", "L": "nan", "Y": "25000", "S": "BUY", "T": 1000}
        )
        self.assertEqual(sym_o, "BTCUSDC")
        self.assertAlmostEqual(ord_it["quantity"], 0.5)
        self.assertAlmostEqual(ord_it["price"], 50000.0)
        sym_t, trd_it = bapi_ws_normalize.parse_trade_execution_report(
            {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": 7, "i": 10, "l": "nan", "z": "0.5", "L": "nan", "p": "50000", "S": "BUY", "T": 1000}
        )
        self.assertEqual(sym_t, "BTCUSDC")
        self.assertEqual(trd_it["qty"], "0.5")
        self.assertEqual(trd_it["price"], "50000")

        # read_trade_records_range_strict rejects start_byte > 0 not aligned to newline
        t_path = os.path.join(self.tmp, "trade_align.jsonl")
        row_bytes = b'{"s":"BTCUSDC","i":{"symbol":"BTCUSDC","id":1,"orderId":10,"price":"50000","qty":"0.1","time":1000,"isBuyer":true}}\n'
        with open(t_path, "wb") as fh:
            fh.write(row_bytes * 2)
        with self.assertRaises(cm.account_cache_health.AccountCacheNotReady):
            cacheintegrity.read_trade_records_range_strict(
                t_path,
                5,
                len(row_bytes) * 2,
                is_valid_item_fn=cm.CacheTradeManager._is_valid_item,
                reader_reason="trade_cache_reader_not_current",
            )
        with self.assertRaises(cm.account_cache_health.AccountCacheNotReady):
            cacheintegrity.read_trade_records_range_strict(
                t_path,
                10,
                5,
                is_valid_item_fn=cm.CacheTradeManager._is_valid_item,
                reader_reason="trade_cache_reader_not_current",
            )
        with self.assertRaises(ValueError):
            cacheintegrity.trade_semantic_signature(
                {"symbol": "BTCUSDC", "id": 1, "orderId": 10, "price": "NaN", "qty": "0.1", "time": 1000, "isBuyer": True}
            )
        with self.assertRaises(ValueError):
            cacheintegrity.trade_semantic_signature(
                {"symbol": "BTCUSDC", "id": 1, "orderId": 10, "price": "Infinity", "qty": "0.1", "time": 1000, "isBuyer": True}
            )
        with self.assertRaises(ValueError):
            cacheintegrity.trade_semantic_signature(
                {"symbol": "BTCUSDC", "id": True, "orderId": 10, "price": "50000", "qty": "0.1", "time": 1000, "isBuyer": True}
            )
        from orchestratorOS.livecheck import os_resources_watchdog
        self.assertIsNone(
            os_resources_watchdog._usage({"ticks": 10}, {"ticks": 20}, 100, 200, 0)
        )
        self.assertIsNone(
            os_resources_watchdog._usage({"ticks": 10}, {"ticks": 20}, 100, 200, False)
        )

        # CacheAllPriceFetcherManager overwrite guard, null symbol_metadata resilience, and fetchtime cleanup
        from market_monitor import pricefetcher

        pf_path = os.path.join(self.tmp, "cache_watcher_all_price.json")
        with open(pf_path, "w", encoding="utf-8") as fh:
            json.dump({"items": {"BTC": [[1000, 60000.0]]}, "fetchtime": {"BTC": 1000}, "symbol_metadata": None}, fh)
        with patch.object(pricefetcher, "PricePlatformFactory") as pf_cls:
            pf_cls.return_value.check_symbol_support.return_value = {"Binance": True}
            pf_mgr = pricefetcher.CacheAllPriceFetcherManager(60, ["BTC"], pf_path, read_only=True)
            pf_mgr.fetchtime_per_symbol["OLD"] = 500
            pf_mgr._rest_sync_cursor_per_symbol["OLD"] = 500
            pf_mgr.cache["OLD"] = [[1, 1.0]]
            pf_mgr.cleanup_old_prices()
            self.assertNotIn("OLD", pf_mgr.cache)
            self.assertNotIn("OLD", pf_mgr.fetchtime_per_symbol)
            self.assertNotIn("OLD", pf_mgr._rest_sync_cursor_per_symbol)
            with open(pf_path + ".meta", "w", encoding="utf-8") as fh:
                json.dump({"max_ts": 9999999999999}, fh)
            self.assertFalse(pf_mgr.save_state_to_file())

        # CacheCurrentPriceManager.extract_price_and_ts selects newest entry from multi-entry list
        cp_path = os.path.join(self.tmp, "cache_current_multi.json")
        now_ms = int(time.time() * 1000)
        with open(cp_path, "w", encoding="utf-8") as fh:
            json.dump({"items": {"BTCUSDC": [[now_ms - 1000, 50000.0], [now_ms, 60000.0]]}, "fetchtime": {"BTCUSDC": now_ms}}, fh)
        cp_mgr = cm.CacheCurrentPriceManager(60, ["BTCUSDC"], cp_path, read_only=True)
        self.assertEqual(cp_mgr.extract_price_and_ts([[now_ms - 1000, 50000.0], [now_ms, 60000.0]]), (60000.0, now_ms))
        self.assertEqual(cp_mgr.get_cached_price_observation("BTCUSDC", max_age_sec=60.0), (now_ms / 1000.0, 60000.0))

        # CacheShortTrendManager.get_window and get_analyzer resolve symbol variants (e.g. BTC/USDC -> BTCUSDC)
        st_path = os.path.join(self.tmp, "cache_short_variant.json")
        st_mgr = cm.CacheShortTrendManager(["BTCUSDC"], filename=st_path, read_only=True)
        self.assertIs(st_mgr.get_window("BTC/USDC"), st_mgr.get_window("BTCUSDC"))
        self.assertIs(st_mgr.get_analyzer("BTC/USDC"), st_mgr.get_analyzer("BTCUSDC"))

        # Read-only venue trade caches (KrakenTrade, HyperliquidTrade, T212Trade) do not pass duplicate read_only or start sync,
        # and resolve venue-specific filenames and default symbols
        self.assertEqual(cm.CacheFactory._CONFIG["KrakenTrade"]["filename"], "cache_trade_kraken.json")
        self.assertEqual(cm.CacheFactory._CONFIG["HyperliquidTrade"]["filename"], "cache_trade_hyperliquid.json")
        self.assertEqual(cm.CacheFactory._CONFIG["T212Trade"]["filename"], "cache_trade_t212.json")
        cm.CacheFactory._instances.pop("KrakenTrade", None)
        cm.CacheFactory._sync_started.discard("KrakenTrade")
        kt_mgr = cm.get_cache_manager("KrakenTrade", symbols=["SOLUSD"], data_sync_mode="producer")
        self.assertTrue(kt_mgr.read_only)
        self.assertNotIn("KrakenTrade", cm.CacheFactory._sync_started)
        cm.CacheFactory._instances.pop("KrakenTrade", None)

        hl_trd_path = os.path.join(self.tmp, "cache_trade_hyperliquid.json")
        with open(hl_trd_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "items": {
                        "HYPEUSDC": [
                            {"coin": "@107", "px": "88.0", "sz": "5.68", "side": "A", "dir": "Sell", "time": 1000, "tid": 101, "oid": 201},
                            {"coin": "@107", "px": "83.8", "sz": "5.75", "side": "B", "dir": "Buy", "time": 2000, "tid": 102, "oid": 202},
                        ]
                    },
                    "fetchtime": {"HYPEUSDC": 2000},
                },
                fh,
            )
        hl_san = cm.CacheTradeManager.load_sanitized_trade_snapshot(hl_trd_path, venue="hyperliquid")
        self.assertEqual(len(hl_san["items"].get("HYPEUSDC", [])), 2)
        hl_mgr = cm.CacheTradeManager(filename=hl_trd_path, venue="hyperliquid", stream_format="json", read_only=True)
        self.assertEqual(hl_mgr.get_last_opposite_fill_price("HYPEUSDC", "BUY"), 88.0)
        self.assertEqual(hl_mgr.get_last_opposite_fill_price("HYPEUSDC", "SELL"), 83.8)

        # tradeall.sanitize_pipe_field and rtrade revision/config validation
        import tradeall
        import rtrade
        self.assertEqual(tradeall.sanitize_pipe_field("a|b\rc\nd"), "a/b c d")
        self.assertEqual(tradeall.sanitize_pipe_field(None), "")
        with self.assertRaises(ValueError):
            rtrade._LivePairVenue._limit_kind(True)
        with self.assertRaises(ValueError):
            rtrade._LivePairVenue._limit_kind(1.5)
        with self.assertRaises(ValueError):
            rtrade._validate_intent_recovery_config(poll_sec=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_cachelongtrendmanager_get_remote_items_resolves_taousd_taousdc_from_priceanalysis_json_iter6_s2(tmp_path, monkeypatch):
    """[ITER6 #2] CacheLongTrendManager.get_remote_items resolves TAOUSD -> TAOUSDC from priceanalysis.json."""
    import json
    import cacheManager as cm
    # 2. CacheLongTrendManager.get_remote_items resolves TAOUSD -> TAOUSDC from priceanalysis.json
    pa_dir = tmp_path / "pa"
    pa_dir.mkdir()
    (pa_dir / "priceanalysis.json").write_text(
        json.dumps({"TAOUSDC": {"timestamp": 1700000000, "direction": "up", "duration_seconds": 86400}}),
        encoding="utf-8",
    )
    lt = cm.CacheLongTrendManager(60, ["TAOUSD"], str(pa_dir / "cache_long_trend.json"), read_only=True)
    monkeypatch.setattr(cm, "_CONFIG_ROOT", str(pa_dir))
    assert lt.get_remote_items("TAOUSD", 0) == [{"timestamp": 1700000000, "direction": "up", "duration_seconds": 86400}]

def test_normalize_cache_base_asset_preserves_usdcusdt_and_strips_ron_gbp_quotes_iter13_s2():
    """[ITER13 #2] normalize_cache_base_asset preserves USDCUSDT and strips RON/GBP quotes."""
    from cachepaths import normalize_cache_base_asset
    # 2. normalize_cache_base_asset preserves USDCUSDT and strips RON/GBP quotes
    assert normalize_cache_base_asset("USDCUSDT") == "USDCUSDT"
    assert normalize_cache_base_asset("USDC/USDT") == "USDCUSDT"
    assert normalize_cache_base_asset("BTCRON") == "BTC"
    assert normalize_cache_base_asset("ETHGBP") == "ETH"

def test_cachetrend__short_trend_symbol_variants_preserves_usdcusdt_intact_and_strips_ron_iter14_s4():
    """[ITER14 #4] cachetrend._short_trend_symbol_variants preserves USDCUSDT intact and strips RON/GBP."""
    from cachetrend import _short_trend_symbol_variants
    # 4. cachetrend._short_trend_symbol_variants preserves USDCUSDT intact and strips RON/GBP
    assert _short_trend_symbol_variants("USDCUSDT") == ["USDCUSDT"]
    assert "BTCUSDC" in _short_trend_symbol_variants("BTCRON")
    assert "ETHUSDC" in _short_trend_symbol_variants("ETHGBP")

def test_cacheshorttrendmanager_slash_sanitization_cross_venue_dynamic_window_resolution_iter23_s4(tmp_path):
    """[ITER23 #4] CacheShortTrendManager slash sanitization & cross-venue dynamic window resolution."""
    import json
    import os
    from cachetrend import CacheShortTrendManager
    # 4. CacheShortTrendManager slash sanitization & cross-venue dynamic window resolution
    st_file = tmp_path / "cache_short_trend.json"
    (tmp_path / "24trend").mkdir(parents=True, exist_ok=True)
    stm = CacheShortTrendManager(symbols=["TAOUSDC"], filename=str(st_file), data_sync_mode="producer")
    assert "/" not in os.path.basename(stm._live_trend_file("TAO/USD"))
    c24_path = tmp_path / "cache_24_price_TAOUSDC.json"
    c24_path.write_text(
        json.dumps({
            "items": {
                "TAOUSDC": [
                    [1_700_000_000_000, 100.0],
                    [1_700_000_010_000, 101.0],
                    [1_700_000_020_000, 102.0],
                    [1_700_000_030_000, 103.0],
                ]
            }
        }),
        encoding="utf-8",
    )
    dyn_res = stm.get_short_trend_for_window("TAO/USD", 30, now=1_700_000_030.0)
    assert dyn_res is not None
    assert dyn_res["growth_coefficient"] > 0

def test_cacheshorttrendmanager_should_wait_gt_s_with_none_growth_coefficient_gradient_recent_fast_iter27_s1():
    """[ITER27 #1] CacheShortTrendManager.should_wait (GT_S) with None growth_coefficient / gradient_recent_fast."""
    from cachetrend import CacheShortTrendManager
    # 1. CacheShortTrendManager.should_wait (GT_S) with None growth_coefficient / gradient_recent_fast
    mgr = CacheShortTrendManager.__new__(CacheShortTrendManager)
    mgr.window_seconds = (1800,)
    mgr._epsilon = lambda snap: 1e-6
    mgr.get_fresh_snapshot = lambda sym, now=None: {
        "growth_coefficient": None,
        "gradient_recent_fast": None,
        "gradient_recent": -0.005,
    }
    assert mgr.should_wait("BUY", "BTCUSDC", fast=False, use_noise_gate=False) is True
    assert mgr.should_wait("BUY", "BTCUSDC", fast=True, use_noise_gate=False) is True
    assert mgr.should_wait("SELL", "BTCUSDC", fast=False, use_noise_gate=False) is False

    mgr.get_short_trend_for_window = lambda sym, win, now=None: {
        "growth_coefficient": None,
        "gradient_recent": 0.004,
        "epsilon": 1e-6,
    }
    assert mgr.should_wait("SELL", "BTCUSDC", window_seconds=900, use_noise_gate=True) is True
    assert mgr.should_wait("BUY", "BTCUSDC", window_seconds=900, use_noise_gate=True) is False

def test_priceanalysis_get_weight_for_cash_permission_at_quant_time_iter27_s2(monkeypatch):
    """[ITER27 #2] priceAnalysis.get_weight_for_cash_permission_at_quant_time."""
    import time
    from types import SimpleNamespace
    import cacheManager as cm
    import priceAnalysis as pa
    # 2. priceAnalysis.get_weight_for_cash_permission_at_quant_time
    fake_lt_cache = {
        "TAOUSD": [
            {
                "duration_seconds": 86400 * 7,
                "direction": "up",
                "start_timestamp": 1_700_000_000.0,
                "timestamp": 1_700_604_800.0,
            }
        ]
    }
    monkeypatch.setattr(
        cm,
        "get_cache_manager",
        lambda name: SimpleNamespace(cache=fake_lt_cache),
    )
    pa.last_timestamp.clear()
    pa.last_w.clear()
    pa.last_calc_time.clear()
    pa.last_duration.clear()

    # Slash symbol TAO/USD resolves to TAOUSD and computes aligned BUY weight (~0.95 at mid-T)
    w_up = pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY")
    assert w_up is not None and w_up > 0.8

    # Flipping direction to 'down' at the same timestamp must NOT return the cached 'up' weight
    fake_lt_cache["TAOUSD"][0]["direction"] = "down"
    w_down = pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY")
    assert w_down is not None and w_down < 0.2

    # Future last_calc_time (clock skew) forces recalculation instead of using stale last_w
    memo_key = ("TAO/USD", "BUY", 14.0, "down")
    pa.last_calc_time[memo_key] = time.time() + 10_000.0
    pa.last_w[memo_key] = [0.999]
    w_recalc = pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY")
    assert w_recalc is not None and w_recalc < 0.2

    # NaN duration_seconds returns None cleanly without ValueError
    fake_lt_cache["TAOUSD"][0]["duration_seconds"] = float("nan")
    assert pa.get_weight_for_cash_permission_at_quant_time("TAO/USD", T_quanta=14, order_type="BUY") is None

def test_cacheintegrity_trade_data_version_trade_semantic_signature_build_account_meta_snapshot_iter71_s1():
    """[ITER71 #1] cacheintegrity.trade_data_version & trade_semantic_signature & build_account_meta_snapshot."""
    import pytest
    import cacheintegrity
    # 1. cacheintegrity.trade_data_version & trade_semantic_signature & build_account_meta_snapshot
    with pytest.raises(ValueError):
        cacheintegrity.trade_data_version("", 1)
    with pytest.raises(ValueError):
        cacheintegrity.trade_data_version("stream1", 1.5)
    with pytest.raises(ValueError):
        cacheintegrity.trade_data_version("stream1", -1)
    assert cacheintegrity.trade_data_version("stream1", 2) == "stream1:2"

    with pytest.raises(ValueError):
        cacheintegrity.trade_semantic_signature({
            "symbol": "BTCUSDC", "id": 10.5, "orderId": 20,
            "price": "100", "qty": "1", "time": 1000, "isBuyer": True,
        })
    with pytest.raises(ValueError):
        cacheintegrity.build_account_meta_snapshot(
            data_version="s:1", fetchtime={}, counts={},
            stream_id="s", revision=1.5, committed_bytes=10, content_digest="chain:" + "00" * 16,
        )

def test_priceanalysis_format_duration_format_timestamp_get_trade_weight_get_weight_for_cash_permission_at_quant_time_iter102_s2():
    """[ITER102 #2] priceAnalysis: format_duration, format_timestamp, get_trade_weight, get_weight_for_cash_permission_at_quant_time."""
    import pytest
    import priceAnalysis as pa
    # 2. priceAnalysis: format_duration, format_timestamp, get_trade_weight, get_weight_for_cash_permission_at_quant_time
    assert pa.format_duration(None) == "0m"
    assert pa.format_duration(True) == "0m"
    assert pa.format_duration(-100) == "0m"
    assert pa.format_duration(float("nan")) == "0m"
    assert pa.format_duration(3660) == "1h 1m"
    assert isinstance(pa.format_timestamp(None), str)
    assert isinstance(pa.format_timestamp(True), str)
    assert isinstance(pa.format_timestamp(float("inf")), str)

    t_b, w_b = pa.get_trade_weight(T=True, trend_len=5.0, trend="up", order_type="BUY")
    assert len(w_b) == 0
    t_inv, w_inv = pa.get_trade_weight(T=14.0, trend_len=5.0, trend="up", order_type="INVALID")
    assert len(w_inv) == 0
    t_bull, w_bull = pa.get_trade_weight(T=14.0, trend_len=7.0, trend=" BULL ", order_type=" b ")
    assert len(w_bull) > 0 and float(w_bull[0]) == pytest.approx(0.95, abs=0.02)
    t_bear, w_bear = pa.get_trade_weight(T=14.0, trend_len=7.0, trend=" BEAR ", order_type=" s ")
    assert len(w_bear) > 0 and float(w_bear[0]) == pytest.approx(0.95, abs=0.02)

    assert pa.get_weight_for_cash_permission_at_quant_time(None, "BUY") is None
    assert pa.get_weight_for_cash_permission_at_quant_time("BTCUSDC", None) is None
    assert pa.get_weight_for_cash_permission_at_quant_time("BTCUSDC", True) is None
    assert pa.get_weight_for_cash_permission_at_quant_time("BTCUSDC", "BUY", T_quanta=True) is None
    assert pa.get_weight_for_cash_permission_at_quant_time("BTCUSDC", "BUY", quant_seconds=False) is None

def test_cachepaths_migrate_legacy_filename_if_needed_and_cache_path_with_pathlike_bool_none_iter110_s2(tmp_path):
    """[ITER110 #2] cachepaths: migrate_legacy_filename_if_needed and cache_path with PathLike/bool/None."""
    import cachepaths as cp
    # 2. cachepaths: migrate_legacy_filename_if_needed and cache_path with PathLike/bool/None
    cp.migrate_legacy_filename_if_needed(None)
    cp.migrate_legacy_filename_if_needed(True)
    cp.migrate_legacy_filename_if_needed(tmp_path / "cache_short_trend.json")
    assert cp.cache_path(True) is True
    assert cp.cache_path(123) == 123

def test_priceanalysis_detect_long_term_trend_with_int_strings_mk_alpha_string_bool_iter163_s2():
    """[ITER163 #2] priceAnalysis: detect_long_term_trend with .0 int strings, mk_alpha string/bool/NaN, and lindy_plateau='false'."""
    import priceAnalysis as pa
    # 2. priceAnalysis: detect_long_term_trend with .0 int strings, mk_alpha string/bool/NaN, and lindy_plateau="false"
    ts = [float(i * 3600) for i in range(1, 100)]
    px = [100.0 + i * 0.5 for i in range(1, 100)]
    assert pa.detect_long_term_trend(ts, px, min_consecutive_blocks=3.5) is None
    assert pa.detect_long_term_trend(ts, px, mk_alpha=True) is None
    assert pa.detect_long_term_trend(ts, px, mk_alpha=float("nan")) is None
    assert pa.detect_long_term_trend(ts, px, mk_alpha=1.5) is None
    res_trend = pa.detect_long_term_trend(
        ts,
        px,
        min_consecutive_blocks="3.0",
        noise_tolerance="2.0",
        min_points_per_window="4.0",
        mk_alpha="0.05",
    )
    assert res_trend is not None and res_trend["direction"] == "up"
    _, w_plateau = pa.get_trade_weight(14, 10, "up", "BUY", lindy_plateau="true")
    _, w_decay = pa.get_trade_weight(14, 10, "up", "BUY", lindy_plateau="false")
    assert w_plateau[0] > w_decay[0]

def test_cachetrademanager_string_symbols_normalization_read_only_true_flag_and_iter165_s2(tmp_path, monkeypatch):
    """[ITER165 #2] CacheTradeManager: string symbols normalization, read_only='true' flag, and 'B'/'S' side aliases in get_last_opposite_fill_price."""
    import pytest
    import cacheManager as cm
    # 2. CacheTradeManager: string symbols normalization, read_only="true" flag, and "B"/"S" side aliases in get_last_opposite_fill_price
    monkeypatch.setattr("utils.cache_path", lambda fn: str(tmp_path / fn))
    tm = cm.CacheTradeManager(60, "BTCUSDC", "test_trades.jsonl", read_only="true")
    assert tm.symbols == ["BTCUSDC"]
    assert tm.read_only is True
    monkeypatch.setattr(tm, "sync_account_reader_if_needed", lambda: None)
    tm.cache["BTCUSDC"] = [
        {"symbol": "BTCUSDC", "id": 1, "orderId": 10, "price": "65000.0", "qty": "0.1", "time": 1700000000000, "isBuyer": True},
        {"symbol": "BTCUSDC", "id": 2, "orderId": 11, "price": "66000.0", "qty": "0.1", "time": 1700000005000, "isBuyer": False},
    ]
    assert tm.get_last_opposite_fill_price("BTCUSDC", "b") == pytest.approx(66000.0)
    assert tm.get_last_opposite_fill_price("BTCUSDC", "s") == pytest.approx(65000.0)

def test_cacheshorttrendmanager_string_symbols_writer_false_window_seconds_as_scalar_iter165_s3(tmp_path):
    """[ITER165 #3] CacheShortTrendManager: string symbols, writer='false', window_seconds as scalar, and should_wait with ' buy '/' sell '/'b'/'s' + use_noise_gate='false'."""
    import cachetrend as ct
    # 3. CacheShortTrendManager: string symbols, writer="false", window_seconds as scalar, and should_wait with " buy "/" sell "/"b"/"s" + use_noise_gate="false"
    st_file = str(tmp_path / "test_short_trend.json")
    stm = ct.CacheShortTrendManager(
        "BTCUSDC",
        filename=st_file,
        data_sync_mode="disk_reader",
        window_seconds="120.0",
        read_only="true",
    )
    assert stm.symbols == ["BTCUSDC"]
    assert stm.writer is False
    assert stm.read_only is True
    assert stm.window_seconds == [120.0]

    now_ts = 1700000100.0
    stm._set_mem(
        "BTCUSDC",
        growth_coefficient=-0.5,
        gradient_recent=-0.5,
        gradient_recent_fast=-0.5,
        epsilon=0.01,
        current_price=65000.0,
        ts=now_ts,
    )
    assert stm.should_wait(" buy ", "BTCUSDC", now=now_ts, use_noise_gate="false") is True
    assert stm.should_wait("b", "BTCUSDC", now=now_ts, use_noise_gate="false") is True
    assert stm.should_wait(" sell ", "BTCUSDC", now=now_ts, use_noise_gate="false") is False
    assert stm.should_wait("s", "BTCUSDC", now=now_ts, use_noise_gate="false") is False


def test_nonbinance_trade_cache_snapshot_resync_and_fetchtime_sanitization(tmp_path):
    """Verify non-Binance JSON trade caches reload external updates without .meta and strip foreign fetchtime keys."""
    from state_io import save_json_state

    trd_file = str(tmp_path / "cache_trade_kraken.json")
    save_json_state(
        trd_file,
        {
            "items": {
                "HYPEUSD": [
                    {
                        "id": "101",
                        "orderId": "101",
                        "symbol": "HYPEUSD",
                        "price": 40.0,
                        "qty": 1.0,
                        "time": 1700000000000,
                        "isBuyer": True,
                    }
                ]
            },
            "fetchtime": {
                "HYPEUSD": 1700000000000.0,
                "BTCUSDC": 1600000000000.0,
            },
        },
    )
    san = cm.CacheTradeManager.load_sanitized_trade_snapshot(trd_file, venue="kraken")
    assert list(san["items"].keys()) == ["HYPEUSD"]
    assert list(san["fetchtime"].keys()) == ["HYPEUSD"]

    mgr = cm.CacheTradeManager(
        venue="kraken",
        symbols=["HYPEUSD"],
        filename=trd_file,
        read_only=True,
    )
    assert len(mgr.cache.get("HYPEUSD", [])) == 1

    # Simulate external producer updating JSON snapshot without touching .meta
    save_json_state(
        trd_file,
        {
            "items": {
                "HYPEUSD": [
                    {
                        "id": "101",
                        "orderId": "101",
                        "symbol": "HYPEUSD",
                        "price": 40.0,
                        "qty": 1.0,
                        "time": 1700000000000,
                        "isBuyer": True,
                    },
                    {
                        "id": "102",
                        "orderId": "102",
                        "symbol": "HYPEUSD",
                        "price": 41.5,
                        "qty": 2.0,
                        "time": 1700000060000,
                        "isBuyer": False,
                    },
                ]
            },
            "fetchtime": {"HYPEUSD": 1700000060000.0},
        },
    )
    assert mgr._persisted_max_ts() == 1700000060000.0
    mgr.sync_account_reader_if_needed()
    assert len(mgr.cache.get("HYPEUSD", [])) == 2


def test_decoupled_account_and_price_ws_daemon_lifecycle_helpers(monkeypatch):
    """Verify cacheManager delegates account/price WS lifecycle to cacheaccount and cacheprice without direct binance_api imports."""
    import ast
    from types import SimpleNamespace
    import cacheaccount
    import cacheprice

    with open(cm.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=cm.__file__)
    direct_binance_imports = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("binance")
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("binance") for alias in node.names)
        )
    ]
    assert direct_binance_imports == []
    assert not hasattr(cacheaccount, "_bapi_trades")
    assert not hasattr(cacheaccount, "clear_binance_trade_cache_hook")
    assert not hasattr(cm, "clear_binance_trade_cache_hook")
    assert not hasattr(cm, "_current_price_instance")
    assert not hasattr(cm, "_short_trend_instance")
    assert not hasattr(cm.CacheFactory, "register_teardown_hook")
    assert not hasattr(cm.CacheFactory, "_teardown_hooks")
    assert not hasattr(cm.CacheManagerInterface, "_is_account_cache")
    assert not hasattr(cm.CacheManagerInterface, "_is_canonical_account_cache")
    assert hasattr(cm.CacheAccountManagerBase, "_is_account_cache")
    assert hasattr(cm.CacheAccountManagerBase, "_is_canonical_account_cache")

    now_ms = int(time.time() * 1000)
    recorded_syncs = []
    init_called = []
    fake_ach = SimpleNamespace(
        MAX_AGE_SEC=60.0,
        enable_writer=lambda: recorded_syncs.append(("enable_writer",)),
        disable_writer=lambda: recorded_syncs.append(("disable_writer",)),
        record_successful_sync=lambda name, ver, now_ms=0: recorded_syncs.append(
            ("sync", name, ver, now_ms)
        ),
    )
    monkeypatch.setattr(cm, "account_cache_health", fake_ach)
    monkeypatch.setattr(
        cacheaccount._bapi_ws_bridge,
        "initialize_ws_bridge_once",
        lambda **_kw: init_called.append(True),
    )

    sync_set = set()
    cacheaccount.start_account_cache_daemon(
        cache_getter=lambda name: SimpleNamespace(
            get_persisted_sync_state=lambda: (f"v-{name}", now_ms - 1000)
        ),
        sync_started_set=sync_set,
    )
    assert sync_set == {"BinanceOrder", "BinanceTrade"}
    assert init_called == [True]
    assert ("enable_writer",) in recorded_syncs
    assert ("sync", "BinanceOrder", "v-BinanceOrder", now_ms - 1000) in recorded_syncs
    assert ("sync", "BinanceTrade", "v-BinanceTrade", now_ms - 1000) in recorded_syncs

    bridge_events = []
    fake_bridge = SimpleNamespace(
        request_stop=lambda: bridge_events.append("request_stop"),
        stop=lambda timeout=0.0: bridge_events.append(("stop", round(timeout, 3))),
    )
    monkeypatch.setattr(cacheaccount._bapi_ws_bridge, "_ws_bridge", fake_bridge)
    cacheaccount.request_stop_account_ws_streams()
    cacheaccount.disable_account_cache_daemon_writer()
    cacheaccount.stop_account_ws_streams(time.monotonic() + 1.0)
    assert "request_stop" in bridge_events
    assert any(isinstance(ev, tuple) and ev[0] == "stop" for ev in bridge_events)
    assert ("disable_writer",) in recorded_syncs
    assert not hasattr(cm, "_ws_bridge")
    assert not hasattr(cm, "_ws_available")
    assert hasattr(cacheprice.CachePriceManagerBase, "subscribe_price")
    assert "subscribe_price" not in cm.CacheManagerInterface.__dict__
    assert "bind_runtime_services" not in cm.CacheManagerInterface.__dict__

    from binance_api import bapi_ws

    monkeypatch.setattr(bapi_ws, "get_ws_manager", lambda: "fake-ws-mgr")
    cpm_calls = []
    res = cacheprice.attach_default_price_ws_stream(
        cpm_getter=lambda **kw: cpm_calls.append(kw) or "cpm-inst",
        sync_ts=0.8,
    )
    assert res == "cpm-inst"
    assert (
        len(cpm_calls) == 1
        and cpm_calls[0]["sync_ts"] == 0.8
        and cpm_calls[0]["data_sync_mode"] == "disk_reader"
    )
    assert cpm_calls[0]["ws_manager"] == "fake-ws-mgr"


def test_cache24pricemanager_external_ohlc_reload_and_merge(tmp_path):
    """Verify Cache24PriceManager reloads newer external OHLC series when idle and merges when live ticks arrive."""
    p24_file = str(tmp_path / "cache_24_price_NVDA_US_EQ.json")
    t0 = 1700000000
    cm.Cache24PriceManager.persist_external_ohlc_series(
        p24_file,
        "NVDA_US_EQ",
        [[t0 * 1000, 175.0, 10.0]],
        float(t0),
    )
    mgr = cm.Cache24PriceManager(
        600,
        ["NVDA_US_EQ"],
        p24_file,
        read_only=False,
    )
    mgr.save_state = True
    assert len(mgr.cache["NVDA_US_EQ"]) == 1

    # External collector writes newer pre-market bar while manager has no new in-memory ticks
    t1 = t0 + 300
    cm.Cache24PriceManager.persist_external_ohlc_series(
        p24_file,
        "NVDA_US_EQ",
        [[t0 * 1000, 175.0, 10.0], [t1 * 1000, 176.5, 25.0]],
        float(t1),
    )
    assert mgr._persist_periodic_state(False) is True
    assert len(mgr.cache["NVDA_US_EQ"]) == 2
    assert mgr.cache["NVDA_US_EQ"][-1] == [t1 * 1000, 176.5, 25.0]

    # Now live tick arrives in memory AND external collector writes an intermediate OHLC bar
    t2 = t1 + 60
    t3 = t1 + 120
    mgr.on_price_update("NVDA_US_EQ", t2 * 1000, 177.0)
    cm.Cache24PriceManager.persist_external_ohlc_series(
        p24_file,
        "NVDA_US_EQ",
        [
            [t0 * 1000, 175.0, 10.0],
            [t1 * 1000, 176.5, 25.0],
            [t3 * 1000, 177.2, 30.0],
        ],
        float(t3),
    )
    assert mgr.save_state_to_file() is True
    assert len(mgr.cache["NVDA_US_EQ"]) == 4


def test_consumer_safe_defaults_and_auto_resync(tmp_path):
    """Verify the 3 cache operating models (producer, live_reader, disk_reader) via data_sync_mode."""
    import inspect

    for fn in (cm.get_cache_manager, cm.CacheFactory.get, cm.get_current_price_manager, cm.get_short_trend_manager):
        params = inspect.signature(fn).parameters
        assert params["data_sync_mode"].default in (cm.CacheManagerInterface.MODE_DISK_READER, None)
        assert "start_sync" not in params
        assert "sync_to_file" not in params
        assert "mode" not in params
        assert "writer" not in params

    cm.CacheFactory.remove("24Price")
    try:
        with patch("utils.cache_path", side_effect=lambda f: str(tmp_path / os.path.basename(f))):
            p24_path = str(tmp_path / "cache_24_price_BTCUSDC.json")
            t0 = int(time.time()) - 300
            cm.Cache24PriceManager.persist_external_ohlc_series(
                p24_path,
                "BTCUSDC",
                [[t0 * 1000, 90000.0, 1.0]],
                float(t0),
            )
            # 1. Model 3 (default): disk_reader -> starts from disk, never increments on its own, auto-resyncs from disk
            managers = cm.get_cache_manager("24Price", symbols=["BTCUSDC"])
            mgr = managers["BTCUSDC"]
            assert mgr.data_sync_mode == cm.CacheManagerInterface.MODE_DISK_READER
            assert mgr.read_only is True
            assert mgr.save_state is False
            assert mgr.thread is None
            assert "24Price" not in cm.CacheFactory._sync_started
            assert mgr.cache["BTCUSDC"] == [[t0 * 1000, 90000.0, 1.0]]

            # Producer persists newer data on disk; next get_cache_manager() call auto-resyncs
            t1 = t0 + 60
            cm.Cache24PriceManager.persist_external_ohlc_series(
                p24_path,
                "BTCUSDC",
                [[t0 * 1000, 90000.0, 1.0], [t1 * 1000, 90500.0, 2.0]],
                float(t1),
            )
            managers_again = cm.get_cache_manager("24Price")
            assert managers_again["BTCUSDC"] is mgr
            assert mgr.cache["BTCUSDC"][-1] == [t1 * 1000, 90500.0, 2.0]

            # 2. Model 2: live_reader -> starts from disk, increments in memory, never writes to disk or reloads from disk
            cm.CacheFactory.remove("24Price")
            with patch.object(cm.Cache24PriceManager, "periodic_sync") as live_periodic:
                live_managers = cm.get_cache_manager(
                    "24Price",
                    symbols=["BTCUSDC"],
                    data_sync_mode="live_reader",
                )
                live_periodic.assert_called_once_with(cm.PRICE_24_SYNC_INTERVAL_SEC, False)
            live_mgr = live_managers["BTCUSDC"]
            assert live_mgr.data_sync_mode == cm.CacheManagerInterface.MODE_LIVE_READER
            assert live_mgr.save_state is False
            t_live = t1 + 15
            live_mgr.on_price_update("BTCUSDC", t_live * 1000, 90800.0)
            assert live_mgr.cache["BTCUSDC"][-1] == [t_live * 1000, 90800.0]

            # Even if disk advances, live_reader does not reload from disk and lose its in-memory state
            t2 = t1 + 120
            cm.Cache24PriceManager.persist_external_ohlc_series(
                p24_path,
                "BTCUSDC",
                [[t0 * 1000, 90000.0, 1.0], [t2 * 1000, 91000.0, 3.0]],
                float(t2),
            )
            live_mgr.resync_mem_file()
            cm.get_cache_manager("24Price")
            assert [pt[1] for pt in live_mgr.cache["BTCUSDC"]] == [90000.0, 90500.0, 90800.0]

            # 3. Model 1: producer (data_sync_mode='producer') -> persists state to disk
            with patch.object(cm.Cache24PriceManager, "periodic_sync") as periodic_mock:
                producer_managers = cm.get_cache_manager(
                    "24Price",
                    symbols=["BTCUSDC"],
                    data_sync_mode="producer",
                )
                producer_mgr = producer_managers["BTCUSDC"]
                assert producer_mgr.data_sync_mode == cm.CacheManagerInterface.MODE_PRODUCER
                assert producer_mgr.save_state is True
                assert producer_mgr.read_only is False
                periodic_mock.assert_called_once_with(cm.PRICE_24_SYNC_INTERVAL_SEC, True)
    finally:
        cm.CacheFactory.remove("24Price")


def test_ast_only_producers_enable_cache_sync_or_writer():
    """Verify via AST that no consumer module passes data_sync_mode='producer' or writer=True."""
    import ast

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_getters = {
        "get_cache_manager",
        "get_current_price_manager",
        "get_short_trend_manager",
        "attach_default_price_ws_stream",
    }
    legacy_aliases = {"start_sync", "sync_to_file", "mode", "writer"}
    violations = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in {"myenv", "venv", "__pycache__", "tests", "cachedb", "logs", "logger", "offline", "archive", "media", "models"}
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), repo_root)
            if rel_path == "cacheManager.py":
                continue
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=rel_path)
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func_name = None
                is_cache_factory_get = False
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    if (
                        func_name == "get"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "CacheFactory"
                    ):
                        is_cache_factory_get = True
                    elif (
                        func_name == "get"
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr == "CacheFactory"
                    ):
                        is_cache_factory_get = True
                if func_name in ("enable_real_ws_event_sync", "start_account_cache_daemon") and rel_path not in {
                    "cacheaccount.py",
                    os.path.join("binance_api", "bapi_ws_bridge.py"),
                }:
                    violations.append(f"{rel_path}:{node.lineno} ({func_name})")
                if is_cache_factory_get or func_name in cache_getters:
                    for kw in node.keywords:
                        if kw.arg in legacy_aliases:
                            violations.append(f"{rel_path}:{node.lineno} ({func_name} legacy kwarg {kw.arg})")
                        elif (
                            kw.arg == "data_sync_mode"
                            and isinstance(kw.value, ast.Constant)
                            and str(kw.value.value).strip().lower() == "producer"
                        ):
                            violations.append(f"{rel_path}:{node.lineno} ({func_name} data_sync_mode='producer')")
                elif func_name == "get_short_trend_manager":
                    for kw in node.keywords:
                        if (
                            kw.arg == "writer"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            violations.append(f"{rel_path}:{node.lineno} (get_short_trend_manager writer=True)")
    assert violations == [], f"Consumer modules must not enable producer mode or use legacy aliases: {violations}"


def test_venue_agnostic_trade_cache_and_t212_ssot_price_routing(tmp_path):
    """Verify CacheTradeManager delegates non-append trade fetches via MarketDataProvider and T212Provider consults SSOT caches."""
    from providers.base import MarketDataProvider
    from providers.t212_provider import T212Provider

    # 1. cacheManager no longer exposes unused apiorders
    assert not hasattr(cm, "apiorders")

    # 2. CacheTradeManager has no venue-specific parsing methods
    for removed_attr in (
        "fetch_kraken_ledger_into_snapshot",
        "merge_kraken_ledger_into_snapshot",
        "apply_hyperliquid_ws_user_fills",
        "touch_hyperliquid_ws_fetchtimes",
        "touch_kraken_ws_fetchtimes",
        "normalize_kraken_ws_trade",
        "kraken_ledger_to_fills",
    ):
        assert not hasattr(cm.CacheTradeManager, removed_attr)

    # 3. Polymorphic MarketDataProvider.resolve_venue_provider_class covers binance, kraken, hyperliquid, t212
    assert MarketDataProvider.resolve_venue_provider_class("binance").__name__ == "BinanceProvider"
    assert MarketDataProvider.resolve_venue_provider_class("kraken").__name__ == "KrakenProvider"
    assert MarketDataProvider.resolve_venue_provider_class("hyperliquid").__name__ == "HyperliquidProvider"
    assert MarketDataProvider.resolve_venue_provider_class("t212").__name__ == "T212Provider"
    assert not hasattr(MarketDataProvider, "_is_binance")
    from providers.quantity import venue_adjust_partial_sell_quantity
    adj_q, adj_m = venue_adjust_partial_sell_quantity("t212", "SELL", 0.5, 0.8, 10.0, min_notional_usd=2.0)
    assert adj_m in ("leave_min_6_remainder", "avoid_sub_6_dust")
    assert adj_q > 0
    assert venue_adjust_partial_sell_quantity("binance", "SELL", 0.5, 0.8, 10.0) == (0.5, None)

    # 4. CacheTradeManager.get_remote_items routes polymorphically for t212 without venue branches
    fake_t212_client = MagicMock()
    fake_t212_client.refresh_order_history.return_value = [
        {
            "order": {"id": 501, "ticker": "NVDA_US_EQ", "side": "BUY", "status": "FILLED"},
            "fill": {
                "id": 9001,
                "quantity": 2.0,
                "price": 120.5,
                "filledAt": "2026-01-15T14:30:00Z",
                "walletImpact": {"taxes": []},
            },
        }
    ]
    t212_trade_mgr = cm.CacheTradeManager(
        sync_ts=9999,
        symbols=["NVDA_US_EQ"],
        filename=str(tmp_path / "cache_t212_trades.json"),
        api_client=fake_t212_client,
        data_sync_mode="disk_reader",
        venue="t212",
    )
    assert t212_trade_mgr.append_persist is False
    rows = t212_trade_mgr.get_remote_items("NVDA_US_EQ", 0)
    assert len(rows) == 1
    assert rows[0]["orderId"] == "501"
    assert rows[0]["side"] == "BUY"
    assert rows[0]["price"] == 120.5
    assert rows[0]["qty"] == 2.0

    # 5. T212Provider consults SSOT price and 24h history caches first via MarketDataProvider Template Method
    tp = T212Provider(client=fake_t212_client, live_enabled=False)
    with patch.object(tp, "_cached_current_price", return_value=135.25) as mock_ssot_px:
        assert tp.get_current_price("NVDA_US_EQ") == 135.25
        mock_ssot_px.assert_called_once_with("NVDA_US_EQ")
    with patch.object(tp, "_cached_price_history", return_value=[(1700000000.0, 130.0), (1700000060.0, 135.25)]) as mock_ssot_hist:
        hist = tp.get_price_history("NVDA_US_EQ", 1.0)
        assert hist == [(1700000000.0, 130.0), (1700000060.0, 135.25)]
        mock_ssot_hist.assert_called_once_with("NVDA_US_EQ", 1.0)


def test_ssot_state_io_hygiene_and_unified_cache_factory_getters():
    """Verify cache modules use state_io/cacheintegrity instead of raw open/json calls and unify factory getters."""
    import ast
    from providers.kraken_provider import KrakenProvider

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checked_files = (
        "cacheInterface.py",
        "cacheaccount.py",
        "cacheprice.py",
        "cachetrend.py",
        "cacheManager.py",
    )
    raw_violations = []
    for rel_path in checked_files:
        full_path = os.path.join(repo_root, rel_path)
        with open(full_path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=rel_path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                raw_violations.append(f"{rel_path}:{node.lineno} raw open()")
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr in {"load", "dump", "loads", "dumps"}
            ):
                raw_violations.append(f"{rel_path}:{node.lineno} raw json.{node.func.attr}()")
    assert raw_violations == [], f"Unexpected raw open/json calls in cache layer: {raw_violations}"

    # Verify KrakenProvider.extract_ws_own_trades extracts and groups ownTrades WS frames
    ws_frame = [
        [
            {
                "TX-101": {
                    "pair": "HYPE/USD",
                    "type": "buy",
                    "price": "24.5",
                    "vol": "2.0",
                    "time": 1700000005.0,
                    "ordertxid": "ORD-101",
                }
            }
        ],
        "ownTrades",
        {"sequence": 1},
    ]
    extracted = KrakenProvider.extract_ws_own_trades(ws_frame, pairs=["HYPEUSD"])
    assert extracted is not None
    assert "HYPEUSD" in extracted
    assert len(extracted["HYPEUSD"]) == 1
    assert extracted["HYPEUSD"][0]["id"] == "TX-101"
    assert extracted["HYPEUSD"][0]["orderId"] == "ORD-101"
    assert extracted["HYPEUSD"][0]["side"] == "BUY"
    assert KrakenProvider.extract_ws_own_trades({"event": "heartbeat"}, pairs=["HYPEUSD"]) is None


def test_kraken_account_trade_bridge_poll_lifecycle(tmp_path, monkeypatch):
    """Verify _KrakenAccountTradeBridge starts in poll mode, persists fills, and stops cleanly."""
    import cacheaccount
    from providers.kraken_provider import KrakenProvider
    from state_io import read_json_dict

    cache_file = str(tmp_path / "cache_trade_kraken.json")
    pulled_event = threading.Event()

    class _FakeKrakenClient:
        def _private(self, method, data=None, fresh=False):
            pulled_event.set()
            return {
                "ledger": {
                    "L1": {"refid": "R10", "asset": "HYPE", "amount": "2.0", "type": "trade", "time": 1700000010.0},
                    "L2": {"refid": "R10", "asset": "ZUSD", "amount": "-60.0", "type": "trade", "time": 1700000010.0},
                }
            }

    monkeypatch.setattr(
        KrakenProvider,
        "resolve_cache_daemon_config",
        classmethod(
            lambda _cls: {
                "pairs": ["HYPEUSD"],
                "poll_interval": 10.0,
                "backoff_init": 1.0,
                "backoff_max": 5.0,
                "mode": "poll",
                "lookback_sec": 3600.0,
                "ws_url": "wss://ws-auth.kraken.com/",
                "cache_file": cache_file,
            }
        ),
    )
    monkeypatch.setattr(
        KrakenProvider,
        "build_cache_client",
        classmethod(lambda _cls: (_FakeKrakenClient(), "Kraken CACHE")),
    )

    bridge = cacheaccount._KrakenAccountTradeBridge()
    assert bridge.start() is True
    try:
        assert pulled_event.wait(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not os.path.exists(cache_file):
            time.sleep(0.02)
        saved = read_json_dict(cache_file)
        assert isinstance(saved, dict)
        assert len(saved["items"]["HYPEUSD"]) == 1
        assert saved["items"]["HYPEUSD"][0]["id"] == "R10"
        assert saved["items"]["HYPEUSD"][0]["price"] == 30.0
    finally:
        bridge.stop(timeout=1.0)
    assert not bridge.is_running


def test_cache_modules_delegate_venue_ws_protocols_to_providers():
    """Enforce zero concrete provider imports or wss:// URLs in cache*.py and verify provider WS helpers."""
    import ast
    from pathlib import Path
    import cacheprice as cp
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider

    repo_root = Path(__file__).resolve().parent.parent
    concrete_provider_mods = {
        "providers.binance_provider",
        "providers.kraken_provider",
        "providers.hyperliquid_provider",
        "providers.t212_provider",
    }
    for mod_name in (
        "cacheInterface.py",
        "cacheaccount.py",
        "cacheprice.py",
        "cachetrend.py",
        "cacheManager.py",
    ):
        src = (repo_root / mod_name).read_text(encoding="utf-8")
        assert "wss://" not in src, f"{mod_name} must not hardcode venue wss:// URLs"
        tree = ast.parse(src, filename=mod_name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in concrete_provider_mods:
                raise AssertionError(f"{mod_name} imports concrete provider {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in concrete_provider_mods:
                        raise AssertionError(f"{mod_name} imports concrete provider {alias.name}")

    # Declarative venue price stream registry on _VenuePriceStreamBase
    registered_venues = {
        venue_key
        for venue_key, _ws_name, _builder in cp._VenuePriceStreamBase.iter_registered_venue_streams()
    }
    assert {"hyperliquid", "kraken"}.issubset(registered_venues)

    # Provider-owned WS protocol builders & parsers
    assert HyperliquidProvider.build_ws_user_fills_subscriptions(None) == ()
    hl_sub = json.loads(HyperliquidProvider.build_ws_user_fills_subscriptions("0xabc")[0])
    assert hl_sub == {"method": "subscribe", "subscription": {"type": "userFills", "user": "0xabc"}}
    assert json.loads(HyperliquidProvider.build_ws_all_mids_subscription()) == {
        "method": "subscribe",
        "subscription": {"type": "allMids"},
    }
    assert HyperliquidProvider.extract_ws_all_mids_prices(
        {"channel": "allMids", "data": {"mids": {"@107": "33.5"}}},
        ["HYPEUSDC"],
    ) == {"HYPEUSDC": 33.5}

    assert KrakenProvider.classify_ws_control_event({"event": "heartbeat"}) == ("keepalive", "")
    assert KrakenProvider.classify_ws_control_event(
        {"event": "subscriptionStatus", "status": "error", "errorMessage": "bad token"}
    ) == ("error", "bad token")
    assert KrakenProvider.parse_ws_public_ticker_message({"event": "heartbeat"}) == (
        "keepalive",
        None,
        None,
    )








