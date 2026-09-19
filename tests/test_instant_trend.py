"""Tests for cacheManager.CachePriceShortTrendManager:
  - store cross-process (file-backed) + merge snapshot
  - opportunistic gate (should_wait / wait_for_favorable_entry) and epsilon
  - calculation API and fast on_price_update channel
"""
import os, sys, json, time, tempfile, unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

mock_api = MagicMock()
mock_api.get_current_price = MagicMock(return_value=60000.0)
mock_api.client = MagicMock()
sys.modules.setdefault("bapi", mock_api)
sys.modules.setdefault("bapi_trades", MagicMock())
sys.modules.setdefault("bapi_allorders", MagicMock())

with patch("cacheManager._initialize_once", return_value=None):
    import cacheManager as cm


def _make_cache24(symbol, entries, tmp):
    fname = os.path.join(tmp, f"c24_{symbol}.json")
    with open(fname, "w") as f:
        json.dump({"items": {symbol: entries}, "fetchtime": {}}, f)
    return cm.Cache24PriceManager(sync_ts=9999, symbols=[symbol], filename=fname, api_client=mock_api)


def _entries_now(n=60, interval_ms=800, start=60000.0, delta=10.0):
    now = int(time.time() * 1000)
    start_ts = now - n * interval_ms
    return [[start_ts + i * interval_ms, start + i * delta] for i in range(n)]


def _mgr(tmp, name="trend.json"):
    return cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(tmp, name))


# ═══════════════════════════════════════════════════════════════════════════
# Store + merge
# ═══════════════════════════════════════════════════════════════════════════
class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.m = _mgr(self.tmp)

    def test_basic_snapshot_operations(self):
        with self.subTest(msg="update_and_get"):
            self.m.update_snapshot("BTCUSDT", gradient_recent=-0.5, current_price=60000.0)
            snap = self.m.get_snapshot("BTCUSDT")
            self.assertEqual(snap["gradient_recent"], -0.5)
            self.assertEqual(snap["symbol"], "BTCUSDT")
        
        with self.subTest(msg="merge_preserves_fields"):
            self.m.update_snapshot("BTCUSDT", slope_big=5.0, gradient_recent=0.1)
            self.m.update_snapshot("BTCUSDT", gradient_recent=-0.9)
            snap = self.m.get_snapshot("BTCUSDT")
            self.assertEqual(snap["gradient_recent"], -0.9)
            self.assertEqual(snap["slope_big"], 5.0)

        with self.subTest(msg="get_unknown_none"):
            self.assertIsNone(self.m.get_snapshot("NOPE"))

    def test_writer_behavior(self):
        with self.subTest(msg="non-writer does not write"):
            fname = os.path.join(self.tmp, "nw.json")
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=False)
            m.update_snapshot("BTCUSDT", gradient_recent=0.5)
            self.assertFalse(os.path.exists(fname))     # A non-writer does not write.
            self.assertIsNotNone(m.get_snapshot("BTCUSDT"))  # It still retains memory state.
        
        with self.subTest(msg="writer writes file"):
            fname = os.path.join(self.tmp, "w.json")
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=True)
            m.update_snapshot("BTCUSDT", gradient_recent=0.5)
            self.assertTrue(os.path.exists(fname))      # A writer writes.

    def test_is_snapshot_fresh(self):
        with self.subTest(msg="fresh"):
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "f.json"), writer=True)
            m.update_snapshot(
                "BTCUSDT", gradient_recent=0.1,
                current_price=60000.0, ts=time.time())
            self.assertTrue(m.is_snapshot_fresh("BTCUSDT", max_age_sec=10))
            m.update_snapshot(
                "BTCUSDT", gradient_recent=0.1,
                current_price=60000.0, ts=time.time() - 100)
            self.assertFalse(m.is_snapshot_fresh("BTCUSDT", max_age_sec=10))

        with self.subTest(msg="zero price not fresh"):
            m = cm.CachePriceShortTrendManager(
                ["BTCUSDT"], os.path.join(self.tmp, "zero.json"), writer=True)
            m.update_snapshot(
                "BTCUSDT", gradient_recent=0.1,
                current_price=0.0, ts=time.time())
            self.assertFalse(m.is_snapshot_fresh("BTCUSDT", max_age_sec=10))
            self.assertIsNone(m.fresh_snapshot("BTCUSDT"))

        with self.subTest(msg="no data not fresh"):
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "g.json"))
            self.assertFalse(m.is_snapshot_fresh("BTCUSDT", max_age_sec=10))

        with self.subTest(msg="become writer failover"):
            fname = os.path.join(self.tmp, "fail.json")
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=False)
            m.update_snapshot("BTCUSDT", gradient_recent=0.5)
            self.assertFalse(os.path.exists(fname))   # A non-writer does not write.
            m.become_writer()                          # Take over writing.
            m.update_snapshot("BTCUSDT", gradient_recent=0.6)
            self.assertTrue(os.path.exists(fname))     # It now writes.

    def test_resilient_snapshot(self):
        with self.subTest(msg="uses file when fresh"):
            fname = os.path.join(self.tmp, "res1.json")
            writer = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=True)
            writer.update_snapshot(
                "BTCUSDT", gradient_recent=-0.4,
                current_price=60000.0, ts=time.time())
            reader = cm.CachePriceShortTrendManager(["BTCUSDT"], fname)
            snap = reader.get_snapshot_resilient("BTCUSDT", max_age_sec=10)
            self.assertEqual(snap["gradient_recent"], -0.4)
            self.assertFalse(reader._computing)   # A fresh file does not start computation.

        with self.subTest(msg="failover when stale"):
            fname = os.path.join(self.tmp, "res2.json")
            writer = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=True)
            writer.update_snapshot("BTCUSDT", gradient_recent=-0.4, ts=time.time() - 100)  # Stale.
            reader = cm.CachePriceShortTrendManager(["BTCUSDT"], fname)
            reader.get_snapshot_resilient(
                "BTCUSDT", max_age_sec=10,
                cache24_managers={"BTCUSDT": self._cache24()}, current_price_mgr=self._cpm())
            self.assertTrue(reader._computing)   # Started local failover computation.
            self.assertTrue(reader.writer)       # Became the writer.

    def _cache24(self):
        return _make_cache24("BTCUSDT", _entries_now(60), self.tmp)

    def _cpm(self):
        fname = os.path.join(self.tmp, "cp_res.json")
        c = cm.CacheCurrentPriceManager(sync_ts=9999, symbols=["BTCUSDT"],
                                        filename=fname, ws_manager=None, api_client=mock_api,
                                        market_api=mock_api)
        c.on_items_update("BTCUSDT", [60000.0])
        return c

    def test_cross_process_behavior(self):
        with self.subTest(msg="prime from file loads initial"):
            fname = os.path.join(self.tmp, "shared2.json")
            writer = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=True)
            writer.update_snapshot("BTCUSDT", gradient_recent=-0.3, slope_small=2.0)
            # A new reader seeds itself from the file at startup.
            reader = cm.CachePriceShortTrendManager(["BTCUSDT"], fname)   # writer=False
            n = reader.prime_from_file()
            self.assertEqual(n, 1)
            snap = reader.get_snapshot("BTCUSDT")
            self.assertEqual(snap["gradient_recent"], -0.3)
            self.assertEqual(snap["slope_small"], 2.0)
            # After seeding, the reader retains data in _mem for later local calculation.
            reader.update_snapshot("BTCUSDT", gradient_recent=0.9)  # writer=False updates only _mem.
            self.assertEqual(reader.get_snapshot("BTCUSDT")["gradient_recent"], 0.9)
            self.assertFalse(os.path.exists(fname + ".reader"))     # No additional file was written.

        with self.subTest(msg="reader sees writer"):
            fname = os.path.join(self.tmp, "shared.json")
            writer = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=True)
            reader = cm.CachePriceShortTrendManager(["BTCUSDT"], fname)
            writer.update_snapshot("BTCUSDT", gradient_recent=-0.7, current_price=60000.0)
            snap = reader.get_snapshot("BTCUSDT")
            self.assertIsNotNone(snap)
            self.assertEqual(snap["gradient_recent"], -0.7)

        with self.subTest(msg="rapid updates"):
            # The reader sees both same-second updates through mtime_ns.
            fname = os.path.join(self.tmp, "rapid.json")
            writer = cm.CachePriceShortTrendManager(["BTCUSDT"], fname, writer=True)
            reader = cm.CachePriceShortTrendManager(["BTCUSDT"], fname)
            writer.update_snapshot("BTCUSDT", gradient_recent=0.1, current_price=60000.0)
            reader.get_snapshot("BTCUSDT")
            writer.update_snapshot("BTCUSDT", gradient_recent=-0.9)
            self.assertEqual(reader.get_snapshot("BTCUSDT")["gradient_recent"], -0.9)


# ═══════════════════════════════════════════════════════════════════════════
# Gate + epsilon
# ═══════════════════════════════════════════════════════════════════════════
class TestGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.m = _mgr(self.tmp)

    def _pub(self, **f):
        f.setdefault("ts", time.time())
        f.setdefault("current_price", 60000.0)
        f.setdefault("epsilon", 1e-12)   # Direction tests use a negligible noise floor.
        self.m.update_snapshot("BTCUSDT", **f)

    def test_should_wait_conditions(self):
        with self.subTest(msg="mode gradient vs full"):
            self.m._mem.clear()
            self._pub(gradient_recent=-0.5, growth_coefficient=0.5)
            self.assertTrue(self.m.should_wait("BUY", "BTCUSDT", fast=True))
            self.assertFalse(self.m.should_wait("BUY", "BTCUSDT", fast=False))

        with self.subTest(msg="buy waits falling"):
            self.m._mem.clear()
            self._pub(gradient_recent=-0.5)
            self.assertTrue(self.m.should_wait("BUY", "BTCUSDT"))

        with self.subTest(msg="buy places rising"):
            self.m._mem.clear()
            self._pub(gradient_recent=0.5)
            self.assertFalse(self.m.should_wait("BUY", "BTCUSDT"))

        with self.subTest(msg="sell waits rising"):
            self.m._mem.clear()
            self._pub(gradient_recent=0.5)
            self.assertTrue(self.m.should_wait("SELL", "BTCUSDT"))

        with self.subTest(msg="no snapshot"):
            self.m._mem.clear()
            self.assertFalse(self.m.should_wait("BUY", "BTCUSDT"))

        with self.subTest(msg="stale"):
            self.m._mem.clear()
            self._pub(gradient_recent=-0.5, ts=time.time() - cm.CachePriceShortTrendManager.TREND_STALE_SEC - 5)
            self.assertFalse(self.m.should_wait("BUY", "BTCUSDT"))

        with self.subTest(msg="noise waits for clarity"):
            self.m._mem.clear()
            self._pub(gradient_recent=0.4, epsilon=1.0)
            self.assertTrue(self.m.should_wait("BUY", "BTCUSDT"))

        with self.subTest(msg="informed epsilon clear up places"):
            self.m._mem.clear()
            self._pub(gradient_recent=5.0, epsilon=1.0)
            self.assertFalse(self.m.should_wait("BUY", "BTCUSDT"))

    def test_wait_for_favorable_entry(self):
        with self.subTest(msg="returns immediately when favorable"):
            self._pub(gradient_recent=0.5)   # A rising trend does not delay BUY.
            calls = []
            waited = self.m.wait_for_favorable_entry("BUY", "BTCUSDT", max_wait_sec=10,
                                                     sleep_fn=lambda s: calls.append(s))
            self.assertEqual(waited, 0.0)
            self.assertEqual(calls, [])
            
        with self.subTest(msg="stops when flips"):
            self._pub(gradient_recent=-0.5, epsilon=1e-12)
            st = {"n": 0}
            def fake_sleep(_):
                st["n"] += 1
                if st["n"] >= 2:
                    self._pub(gradient_recent=0.5, epsilon=1e-12)
            waited = self.m.wait_for_favorable_entry("BUY", "BTCUSDT", max_wait_sec=60,
                                                     poll_sec=1.0, sleep_fn=fake_sleep)
            self.assertGreaterEqual(waited, 2.0)
            self.assertLess(waited, 60.0)


# ═══════════════════════════════════════════════════════════════════════════
# Calculation API and fast channel (start_computation / on_price_update).
# ═══════════════════════════════════════════════════════════════════════════
class TestComputation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache24 = _make_cache24("BTCUSDT", _entries_now(60), self.tmp)
        fname = os.path.join(self.tmp, "cp.json")
        self.cpm = cm.CacheCurrentPriceManager(sync_ts=9999, symbols=["BTCUSDT"],
                                               filename=fname, ws_manager=None, api_client=mock_api,
                                               market_api=mock_api)
        self.m = _mgr(self.tmp)
        self.m.start_computation({"BTCUSDT": self.cache24}, self.cpm)

    def test_computation_window_and_trend(self):
        with self.subTest(msg="windows built"):
            self.assertIsNotNone(self.m.get_window("BTCUSDT"))
            self.assertIsNotNone(self.m.get_window("BTCUSDT", self.m.window_big_sec))
            self.assertIsNotNone(self.m.get_analyzer("BTCUSDT"))
            
        with self.subTest(msg="get instant trend"):
            ft, gc, sf, gr = self.m.get_instant_trend("BTCUSDT")
            self.assertIn(ft, (-1, 0, 1))

        with self.subTest(msg="on price update publishes fast"):
            ts_ms = int(time.time() * 1000)
            self.m.on_price_update("BTCUSDT", ts_ms, 60500.0)
            snap = self.m.get_snapshot("BTCUSDT")
            self.assertIsNotNone(snap)
            self.assertIn("gradient_recent_fast", snap)
            self.assertIn("epsilon", snap)
            self.assertEqual(snap["current_price"], 60500.0)
            self.assertEqual(snap["ts_fast"], ts_ms / 1000.0)
            
        with self.subTest(msg="cache24 tick updates window and snapshot"):
            win = self.m.get_window("BTCUSDT")
            n_before = len(win.prices)
            self.cache24.on_price_update("BTCUSDT", int(time.time() * 1000), 61234.0)
            self.assertIn(61234.0, win.prices)
            self.assertGreaterEqual(len(win.prices), n_before)
            self.assertIsNotNone(self.m.get_snapshot("BTCUSDT"))

    def test_configurable_window_durations(self):
        m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "cfg.json"),
                                        window_seconds=[120, 3600])
        self.assertEqual(m.window_small_sec, 120)   # Smallest.
        self.assertEqual(m.window_big_sec, 3600)    # Largest.
        m.start_computation({"BTCUSDT": self.cache24}, self.cpm)
        # The primary small window reflects the configured Cache24 duration.
        self.assertIsNotNone(m.get_window("BTCUSDT"))
        self.assertIsNotNone(m.get_window("BTCUSDT", 3600))

    def test_n_windows_list(self):
        # Sort an N-window list ascending and choose the smallest as primary.
        m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "n.json"),
                                        window_seconds=[3600, 60, 600])
        self.assertEqual(m.window_seconds, [60.0, 600.0, 3600.0])
        self.assertEqual(m.window_small_sec, 60.0)
        self.assertEqual(m.window_big_sec, 3600.0)
        m.start_computation({"BTCUSDT": self.cache24}, self.cpm)
        self.cpm._push_price("BTCUSDT", 50000.0)
        for sec in (60, 600, 3600):
            self.assertIsNotNone(m.get_window("BTCUSDT", sec))
        m.evaluate_full("BTCUSDT")
        snap = m.get_snapshot("BTCUSDT")
        # Compatibility aliases map slope_small to smallest and slope_big to largest.
        self.assertIn("slope_small", snap)
        self.assertIn("slope_big", snap)
        # Generic slope for each window, keyed by seconds.
        for sec in (60, 600, 3600):
            self.assertIn(str(sec), snap["slopes"])
        self.assertIn("gradient_recent", snap)   # From the primary window.

    def test_threshold_configurations(self):
        with self.subTest(msg="default per window"):
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "thd.json"),
                                            window_seconds=[60, 3600])
            self.assertEqual(m.threshold_for("BTCUSDT", 60), m.PRICE_CHANGE_THRESHOLD_SMALL)
            self.assertEqual(m.threshold_for("BTCUSDT", 3600), m.PRICE_CHANGE_THRESHOLD_BIG)

        with self.subTest(msg="per window dict"):
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "thw.json"),
                                            window_seconds=[60, 3600],
                                            thresholds={60: 0.3, 3600: 1.5})
            self.assertEqual(m.threshold_for("BTCUSDT", 60), 0.3)
            self.assertEqual(m.threshold_for("BTCUSDT", 3600), 1.5)

        with self.subTest(msg="per symbol"):
            m = cm.CachePriceShortTrendManager(["BTCUSDT", "TAOUSDT"], os.path.join(self.tmp, "ths.json"),
                                            window_seconds=[60, 3600],
                                            thresholds={"BTCUSDT": {60: 0.3}, "TAOUSDT": {60: 1.2}})
            self.assertEqual(m.threshold_for("BTCUSDT", 60), 0.3)
            self.assertEqual(m.threshold_for("TAOUSDT", 60), 1.2)
            self.assertEqual(m.threshold_for("TAOUSDT", 3600), m.PRICE_CHANGE_THRESHOLD_BIG)

        with self.subTest(msg="callable"):
            m = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "thc.json"),
                                            window_seconds=[60, 3600],
                                            thresholds=lambda sym, sec: 0.9 if sec < 100 else 2.0)
            self.assertEqual(m.threshold_for("BTCUSDT", 60), 0.9)
            self.assertEqual(m.threshold_for("BTCUSDT", 3600), 2.0)

    def test_start_computation_idempotent(self):
        w1 = self.m.get_window("BTCUSDT")
        self.m.start_computation({"BTCUSDT": self.cache24}, self.cpm)
        self.assertIs(self.m.get_window("BTCUSDT"), w1)

    def test_evaluate_full_writes_complete_snapshot(self):
        # Full calculation without trading logic yields every snapshot metric.
        self.cpm._push_price("BTCUSDT", 50000.0)
        self.m.evaluate_full("BTCUSDT")
        snap = self.m.get_snapshot("BTCUSDT")
        self.assertIsNotNone(snap)
        for key in ("final_trend", "slope_full", "gradient_recent",
                    "slope_small", "slope_big", "slope_max_min", "pos", "epsilon"):
            self.assertIn(key, snap)

    def test_failed_full_eval_preserves_primed_stale_snapshot(self):
        filename = os.path.join(self.tmp, "stale_trend.json")
        stale_ts = time.time() - 120.0
        stale = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "current_price": 60400.0,
                "gradient_recent": -0.25,
                "ts": stale_ts,
            },
        }
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(stale, handle)
        cpm = MagicMock()
        cpm.cached_price_observation.return_value = None
        cpm.get_price_value.return_value = None
        manager = cm.CachePriceShortTrendManager(
            ["BTCUSDT"], filename, writer=True)
        self.addCleanup(manager.shutdown)
        manager.start_computation({"BTCUSDT": self.cache24}, cpm)

        manager.evaluate_full("BTCUSDT")
        snapshot = manager.get_snapshot("BTCUSDT")

        self.assertEqual(snapshot["current_price"], 60400.0)
        self.assertNotEqual(snapshot["current_price"], 0.0)
        self.assertEqual(snapshot["gradient_recent"], -0.25)
        self.assertEqual(snapshot["ts"], stale_ts)
        cpm.cached_price_observation.assert_called_with(
            "BTCUSDT", manager.TREND_STALE_SEC)
        cpm.get_price_value.assert_not_called()

    def test_valid_push_full_eval_uses_price_observation_timestamp(self):
        self.cpm._push_price("BTCUSDT", 61234.5)
        observed_at_ms = int((time.time() - 2.0) * 1000)
        with self.cpm.lock:
            self.cpm.cache["BTCUSDT"][0][0] = observed_at_ms

        self.m.evaluate_full("BTCUSDT")
        snapshot = self.m.get_snapshot("BTCUSDT")

        self.assertEqual(snapshot["current_price"], 61234.5)
        self.assertEqual(snapshot["ts"], observed_at_ms / 1000.0)

    def test_full_eval_loop_thread_started(self):
        m2 = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "t2.json"))
        m2.start_computation({"BTCUSDT": self.cache24}, self.cpm, run_full_eval=True)
        self.assertIsNotNone(m2._full_eval_thread)
        self.assertTrue(m2._full_eval_thread.is_alive())


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic window: get_instant_trend_for_window, is_trend_up_for_window, and
# should_wait(window_seconds=...) calculate on demand from raw Cache24 data
# instead of using the two fixed precomputed windows.
# ═══════════════════════════════════════════════════════════════════════════
def _entries_trending(n, interval_sec, start, delta, end_offset_sec=0.0):
    """Build a strictly monotonic price series ending ``end_offset_sec`` ago.

    Positive delta rises and negative delta falls. Zero ends now; a positive
    offset simulates stale data.
    """
    now_ms = int(time.time() * 1000)
    end_ms = now_ms - int(end_offset_sec * 1000)
    step_ms = int(interval_sec * 1000)
    start_ts = end_ms - (n - 1) * step_ms
    return [[start_ts + i * step_ms, start + i * delta] for i in range(n)]


class TestDynamicWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _cache24_up(self, n=120, interval_sec=1.0, delta=5.0, end_offset_sec=0.0):
        entries = _entries_trending(n, interval_sec, 60000.0, delta, end_offset_sec)
        return _make_cache24("BTCUSDT", entries, self.tmp)

    def _mgr_with_cache24(self, cache24):
        m = _mgr(self.tmp, name=f"dyn_{id(cache24)}.json")
        cpm_fname = os.path.join(self.tmp, f"cp_{id(cache24)}.json")
        cpm = cm.CacheCurrentPriceManager(sync_ts=9999, symbols=["BTCUSDT"],
                                          filename=cpm_fname, ws_manager=None, api_client=mock_api,
                                          market_api=mock_api)
        m.start_computation({"BTCUSDT": cache24}, cpm)
        return m

    def test_dynamic_window_edge_cases(self):
        with self.subTest(msg="unavailable without start computation"):
            m = _mgr(self.tmp)
            self.assertIsNone(m.get_instant_trend_for_window("BTCUSDT", 60))
            self.assertFalse(m.is_trend_up_for_window("BTCUSDT", 60))
            self.assertFalse(m.should_wait("BUY", "BTCUSDT", window_seconds=60))

        with self.subTest(msg="unknown symbol returns none"):
            m = self._mgr_with_cache24(self._cache24_up())
            self.assertIsNone(m.get_instant_trend_for_window("ETHUSDT", 60))

        with self.subTest(msg="stale returns none"):
            stale_offset = cm.CachePriceShortTrendManager.TREND_STALE_SEC + 30
            m = self._mgr_with_cache24(self._cache24_up(end_offset_sec=stale_offset))
            self.assertIsNone(m.get_instant_trend_for_window("BTCUSDT", 60))
            self.assertFalse(m.should_wait("BUY", "BTCUSDT", window_seconds=60))

        with self.subTest(msg="too few samples returns none"):
            m = self._mgr_with_cache24(self._cache24_up(n=20, interval_sec=10.0))
            self.assertIsNone(m.get_instant_trend_for_window("BTCUSDT", 14))

        with self.subTest(msg="window clamped to configured bounds"):
            m = self._mgr_with_cache24(self._cache24_up(n=300, interval_sec=1.0))
            below = m.get_instant_trend_for_window("BTCUSDT", 1.0)
            self.assertIsNotNone(below)
            self.assertEqual(below["window_seconds"], cm.CM_DYNAMIC_WINDOW_MIN_SEC)
            above = m.get_instant_trend_for_window("BTCUSDT", 999999.0)
            self.assertIsNotNone(above)
            self.assertEqual(above["window_seconds"], cm.CM_DYNAMIC_WINDOW_MAX_SEC)

    def test_dynamic_window_trend_detection(self):
        with self.subTest(msg="uptrend detected"):
            m = self._mgr_with_cache24(self._cache24_up(delta=5.0))
            dyn = m.get_instant_trend_for_window("BTCUSDT", 60)
            self.assertIsNotNone(dyn)
            self.assertEqual(dyn["final_trend"], 1)
            self.assertGreater(dyn["growth_coefficient"], 0)
            self.assertTrue(m.is_trend_up_for_window("BTCUSDT", 60))

        with self.subTest(msg="downtrend detected"):
            m = self._mgr_with_cache24(self._cache24_up(delta=-5.0))
            dyn = m.get_instant_trend_for_window("BTCUSDT", 60)
            self.assertIsNotNone(dyn)
            self.assertEqual(dyn["final_trend"], -1)
            self.assertLess(dyn["growth_coefficient"], 0)
            self.assertFalse(m.is_trend_up_for_window("BTCUSDT", 60))

    def test_dynamic_window_waiting(self):
        with self.subTest(msg="uses dynamic path not snapshot"):
            m = self._mgr_with_cache24(self._cache24_up(delta=-5.0))
            m.update_snapshot("BTCUSDT", growth_coefficient=5.0, ts=time.time())
            self.assertFalse(m.should_wait("BUY", "BTCUSDT"))
            self.assertTrue(m.should_wait("BUY", "BTCUSDT", window_seconds=60, use_noise_gate=False))
            
        with self.subTest(msg="dynamic sell side"):
            m = self._mgr_with_cache24(self._cache24_up(delta=5.0))
            self.assertTrue(m.should_wait("SELL", "BTCUSDT", window_seconds=60, use_noise_gate=False))
            self.assertFalse(m.should_wait("BUY", "BTCUSDT", window_seconds=60, use_noise_gate=False))

        with self.subTest(msg="wait forwards window seconds"):
            m = self._mgr_with_cache24(self._cache24_up(delta=-5.0))
            calls = []
            waited = m.wait_for_favorable_entry(
                "BUY", "BTCUSDT", max_wait_sec=2, poll_sec=0.5, window_seconds=60,
                sleep_fn=lambda s: calls.append(s))
            self.assertGreater(waited, 0.0)
            self.assertTrue(calls)


class TestCache24RecentEntriesBisect(unittest.TestCase):
    """Verify bisect produces the former O(n) filter's exact output.

    The optimized complexity is O(log n + k) for every cutoff.
    """
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_bisect_recent_entries(self):
        with self.subTest(msg="matches naive filter"):
            entries = _entries_trending(500, 0.2, 100.0, 0.01)
            c24 = _make_cache24("BTCUSDT", entries, self.tmp)
            for last_seconds in (1, 5, 20, 60, 90, 1000):
                cutoff_ms = int((time.time() - last_seconds) * 1000)
                expected = [e for e in entries if e[0] >= cutoff_ms]
                got = c24.get_recent_entries("BTCUSDT", last_seconds=last_seconds)
                self.assertEqual(got, expected, f"mismatch la last_seconds={last_seconds}")

        with self.subTest(msg="empty symbol"):
            c24 = _make_cache24("BTCUSDT", [], self.tmp)
            self.assertEqual(c24.get_recent_entries("BTCUSDT", last_seconds=60), [])

        with self.subTest(msg="unknown symbol"):
            c24 = _make_cache24("BTCUSDT", _entries_trending(10, 1.0, 100.0, 1.0), self.tmp)
            self.assertEqual(c24.get_recent_entries("NOPE", last_seconds=60), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
