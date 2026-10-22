"""
Event-driven WebSocket-to-cache bridge (renamed and cleaned up from the
ambiguously named test_session_changes.py on July 28).

Coverage kept here because it is unique:
  1. The bapi_ws subscriber pattern (subscribe/unsubscribe/_notify_subscribers).
  2. End-to-end propagation from BinanceWebSocketManager through
     CacheCurrentPriceManager to Cache24PriceManager.

The isolated manager tests were removed because test_cache_manager_full.py
covers the same behavior and additional cases. No coverage is lost.
"""
import os, sys, tempfile, unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

# ─── mock bapi before the import ────────────────────────────────────────────
mock_bapi = MagicMock()
mock_bapi.get_current_price = MagicMock(return_value=50000.0)
mock_bapi.client = MagicMock()
sys.modules.setdefault("bapi", mock_bapi)

from binance_api.bapi_ws import BinanceWebSocketManager  # noqa: E402
import cacheManager as cm  # noqa: E402

SYMBOLS = ["BTCUSDC"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. bapi_ws subscriber pattern (unique and not covered in test_bapi_ws.py)
# ═══════════════════════════════════════════════════════════════════════════════
class TestBapiWsSubscriber(unittest.TestCase):

    def setUp(self):
        self.ws = BinanceWebSocketManager()

    def test_subscribe_adds_subscriber(self):
        sub = MagicMock()
        self.ws.subscribe(sub)
        self.assertIn(sub, self.ws._subscribers)

    def test_subscribe_no_duplicates(self):
        sub = MagicMock()
        self.ws.subscribe(sub)
        self.ws.subscribe(sub)
        self.assertEqual(self.ws._subscribers.count(sub), 1)

    def test_unsubscribe_removes_subscriber(self):
        sub = MagicMock()
        self.ws.subscribe(sub)
        self.ws.unsubscribe(sub)
        self.assertNotIn(sub, self.ws._subscribers)

    def test_notify_calls_on_items_update(self):
        sub = MagicMock()
        self.ws.subscribe(sub)
        self.ws._notify_subscribers("BTCUSDC", [50000.0])
        sub.on_items_update.assert_called_once_with("BTCUSDC", [50000.0])

    def test_notify_multiple_subscribers(self):
        subs = [MagicMock() for _ in range(3)]
        for s in subs:
            self.ws.subscribe(s)
        self.ws._notify_subscribers("TAOUSDC", [300.0])
        for s in subs:
            s.on_items_update.assert_called_once_with("TAOUSDC", [300.0])

    def test_notify_subscriber_exception_doesnt_stop_others(self):
        bad = MagicMock()
        bad.on_items_update.side_effect = RuntimeError("boom")
        good = MagicMock()
        self.ws.subscribe(bad)
        self.ws.subscribe(good)
        self.ws._notify_subscribers("BTCUSDC", [1.0])   # must not raise
        good.on_items_update.assert_called_once()


# ─── Integration helpers ──────────────────────────────────────────────────────
def _make_current_price_manager(tmp_dir, price=50000.0):
    mock_bapi.get_current_price.return_value = price
    mgr = cm.CacheCurrentPriceManager(
        sync_ts=9999, symbols=SYMBOLS,
        filename=os.path.join(tmp_dir, "cache_current_price.json"),
        ws_manager=None, api_client=mock_bapi, market_api=mock_bapi,
    )
    mgr.enable_save_state_to_file()
    return mgr


def _make_cache24_manager(tmp_dir, current_price_manager):
    with patch.object(
        cm,
        "get_current_price_manager",
        return_value=current_price_manager,
    ):
        mgr = cm.Cache24PriceManager(
            sync_ts=9999, symbols=SYMBOLS,
            filename=os.path.join(tmp_dir, "cache_24_price_BTCUSDC.json"),
            api_client=mock_bapi,
        )
    mgr.enable_save_state_to_file()
    return mgr


# ═══════════════════════════════════════════════════════════════════════════════
# 2. End-to-end integration: WS -> CurrentPrice -> Cache24. This is not covered
#    by test_cache_manager_full.py, which tests the managers in isolation.
# ═══════════════════════════════════════════════════════════════════════════════
class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        mock_bapi.get_current_price.return_value = 50000.0
        self.ws = BinanceWebSocketManager(symbols=["BTCUSDC"])
        self.cur = _make_current_price_manager(self.tmp)
        self.h24 = _make_cache24_manager(self.tmp, self.cur)
        self.ws.subscribe(self.cur)
        self.cur.subscribe_price(self.h24)

    def test_ws_event_updates_current_price(self):
        now_ms = int(cm.time.time() * 1000)
        self.ws._notify_subscribers("BTCUSDC", [[now_ms, 55000.0]])
        self.assertEqual(self.cur.get_price_value("BTCUSDC"), 55000.0)

    def test_ws_event_propagates_to_cache24(self):
        now_ms = int(cm.time.time() * 1000)
        self.ws._notify_subscribers("BTCUSDC", [[now_ms, 56000.0]])
        with self.h24.lock:
            entries = self.h24.cache.get("BTCUSDC", [])
        self.assertIn(56000.0, [e[1] for e in entries])

    def test_multiple_ws_events_all_recorded_in_cache24(self):
        prices = [50000.0, 50100.0, 50200.0, 50300.0]
        base_ms = int(cm.time.time() * 1000)
        for idx, p in enumerate(prices):
            self.ws._notify_subscribers("BTCUSDC", [[base_ms + idx, p]])
        with self.h24.lock:
            recorded = [e[1] for e in self.h24.cache.get("BTCUSDC", [])]
        for p in prices:
            self.assertIn(p, recorded)

    def test_current_price_snapshot_not_history(self):
        """CurrentPrice is a snapshot while Cache24 retains price history."""
        now_ms = int(cm.time.time() * 1000)
        self.ws._notify_subscribers("BTCUSDC", [[now_ms, 57000.0]])
        with self.cur.lock:
            self.assertEqual(len(self.cur.cache.get("BTCUSDC", [])), 1)

    def test_persistence_currentprice(self):
        """Load the cache from its file after a simulated restart."""
        self.cur.enable_save_state_to_file()
        now_ms = int(cm.time.time() * 1000)
        self.ws._notify_subscribers("BTCUSDC", [[now_ms, 58000.0]])
        self.cur.save_state_to_file_if_enabled()
        mgr2 = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=SYMBOLS,
            filename=self.cur.filename, api_client=mock_bapi, market_api=mock_bapi,
        )
        with mgr2.lock:
            entries = mgr2.cache.get("BTCUSDC", [])
        self.assertTrue(entries, "empty cache after load")
        self.assertEqual(entries[0][1], 58000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
