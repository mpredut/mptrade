"""
Tests the behaviour of the fallback thread in CacheCurrentPriceManager.

Scenarios:
  1. At startup with a cache loaded from file, the thread does NOT overwrite immediately.
  2. If the WS is healthy, the thread does NOT poll over HTTP.
  3. If the WS is dead, the thread DOES poll over HTTP after sync_ts.
  4. If the WS comes back, the thread stops polling.

Isolation: each test uses a local MagicMock for api_client, avoiding the
cross-contamination caused by daemon threads left running.
"""
import os, sys, json, time, tempfile, unittest
from unittest.mock import MagicMock

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

# A minimal mock for importing the module.
_module_mock = MagicMock()
_module_mock.get_current_price = MagicMock(return_value=50000.0)
sys.modules.setdefault("bapi", _module_mock)

import cacheManager as cm

SYMBOLS = ["BTCUSDC"]


def _saved_file(tmp_dir, price, ts_ms=None):
    ts_ms = ts_ms or int(time.time() * 1000)
    fname = os.path.join(tmp_dir, "cache_currentprice.json")
    with open(fname, "w") as f:
        json.dump({
            "items":     {"BTCUSDC": [[ts_ms, price]]},
            "fetchtime": {"BTCUSDC": ts_ms},
        }, f)
    return fname


def _make_mgr(filename, sync_ts, api_mock):
    """Create a manager with its own mock — isolated from the other tests."""
    return cm.CacheCurrentPriceManager(
        sync_ts=sync_ts,
        symbols=SYMBOLS,
        filename=filename,
        ws_manager=None,
        api_client=api_mock,
        market_api=api_mock,   # the HTTP fetch goes through the market-data facade (injectable)
    )


class TestFallbackThreadBehavior(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cm._current_price_instance = None
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown()
        cm._current_price_instance = None

    def _manager(self, filename, sync_ts, api_mock):
        manager = _make_mgr(filename, sync_ts, api_mock)
        self.managers.append(manager)
        manager.periodic_sync(save_state=False)
        return manager

    # ── Test 1: the cache from file is NOT overwritten immediately ───────────

    def test_loaded_cache_not_overwritten_immediately(self):
        """
        A fresh manager with a large sync_ts must not overwrite the cache
        loaded from file before the first sleep.
        """
        fname = _saved_file(self.tmp, price=99000.0)
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = 50000.0

        mgr = self._manager(fname, sync_ts=9999, api_mock=api_mock)

        with mgr.lock:
            entries = mgr.cache.get("BTCUSDC", [])

        self.assertTrue(entries, "empty cache after loading from file")
        self.assertEqual(
            entries[0][1], 99000.0,
            f"The cached price is {entries[0][1]}, not 99000.0 — the thread overwrote it immediately!"
        )
        api_mock.get_current_price.assert_not_called()

    # ── Test 2: healthy WS -> NO HTTP poll ──────────────────────────────────

    def test_ws_healthy_skips_http_poll(self):
        """
        If the WS is healthy (_ws_last_event_ts recent),
        the thread does NOT poll over HTTP for the whole test.
        """
        fname = _saved_file(self.tmp, price=55000.0)
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = 55000.0

        mgr = self._manager(fname, sync_ts=1, api_mock=api_mock)
        mgr._ws_last_event_ts = time.time()   # fresh WS

        time.sleep(2.5)   # 2 cicluri complete

        api_mock.get_current_price.assert_not_called()

    # ── Test 3: dead WS -> it polls over HTTP ───────────────────────────────

    def test_ws_dead_triggers_http_poll(self):
        """
        With _ws_last_event_ts = 0 (the WS never active),
        the thread must poll over HTTP after sync_ts seconds.
        """
        fname = _saved_file(self.tmp, price=55000.0)
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = 55000.0

        mgr = self._manager(fname, sync_ts=1, api_mock=api_mock)
        mgr._ws_last_event_ts = 0.0   # The WS is explicitly dead.

        time.sleep(2.5)

        api_mock.get_current_price.assert_called()

    # ── Test 4: the WS returns -> polling stops ─────────────────────────────

    def test_ws_recovery_stops_polling(self):
        """
        If the WS was dead and comes back, the thread stops polling.
        """
        fname = _saved_file(self.tmp, price=55000.0)
        api_mock = MagicMock()
        api_mock.get_current_price.return_value = 55000.0

        mgr = self._manager(fname, sync_ts=1, api_mock=api_mock)
        mgr._ws_last_event_ts = 0.0   # The WS is dead -> it will poll.

        time.sleep(1.5)
        self.assertGreater(
            api_mock.get_current_price.call_count, 0,
            "no poll with a dead WS — the thread is not working"
        )

        # WS revine
        api_mock.get_current_price.reset_mock()
        mgr._ws_last_event_ts = time.time()

        time.sleep(2.5)   # 2 cycles — it should no longer poll
        self.assertEqual(
            api_mock.get_current_price.call_count, 0,
            "the thread keeps polling after the WS came back"
        )

    def test_shutdown_stops_waiting_thread(self):
        fname = _saved_file(self.tmp, price=55000.0)
        manager = self._manager(fname, sync_ts=9999, api_mock=MagicMock())
        self.assertTrue(manager.thread.is_alive())
        self.assertTrue(manager.shutdown(timeout=1.0))
        self.assertIsNone(manager.thread)


if __name__ == "__main__":
    unittest.main(verbosity=2)
