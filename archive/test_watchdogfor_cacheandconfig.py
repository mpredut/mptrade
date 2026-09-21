import os, sys, json, time, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verify_tools"))
import watchdogfor_cacheandconfig as wd


def _write_cache(path, fetchtime_ms, mtime_sec=None):
    json.dump({"items": {"HYPE": [[fetchtime_ms, 65.0]]},
               "fetchtime": {"HYPE": fetchtime_ms}}, open(path, "w"))
    if mtime_sec is not None:
        os.utime(path, (mtime_sec, mtime_sec))


class TestWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "cache_prices_multi.json")
        wd._CACHE_DIR = Path(self.tmp)
        wd.STATE_FILE = os.path.join(self.tmp, ".state.json")
        wd.STALE_MINUTES = 20
        wd.COOLDOWN_MINUTES = 60

    def test_normalize_ts(self):
        self.assertAlmostEqual(wd._normalize_ts_seconds(1779829664000), 1779829664.0)  # ms
        self.assertAlmostEqual(wd._normalize_ts_seconds(1779829664), 1779829664.0)      # sec
        self.assertEqual(wd._normalize_ts_seconds(0), 0.0)

    def test_cache_alerts_and_staleness(self):
        now = time.time()
        with self.subTest("fresh cache no alert"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            _write_cache(self.cache, int(now * 1000))   # fresh
            with patch.object(wd.wc, "send_ntfy") as ntfy, patch.object(wd.wc, "send_email") as email:
                self.assertFalse(wd.check_once(now=now))
                ntfy.assert_not_called()
                email.assert_not_called()

        with self.subTest("stale cache alerts"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            old = now - 60 * 60   # an hour ago -> stale
            _write_cache(self.cache, int(old * 1000), mtime_sec=old)
            with patch.object(wd.wc, "send_ntfy", return_value=True) as ntfy, \
                 patch.object(wd.wc, "send_email", return_value=True) as email:
                self.assertTrue(wd.check_once(now=now))
                ntfy.assert_called_once()
                email.assert_called_once()

        with self.subTest("cooldown suppresses second alert"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            _write_cache(self.cache, int((now - 3600) * 1000), mtime_sec=now - 3600)
            with patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                self.assertTrue(wd.check_once(now=now))            # the first one -> alert
                self.assertFalse(wd.check_once(now=now + 60))      # in cooldown -> no
                self.assertTrue(wd.check_once(now=now + 3700))     # after the cooldown -> yes

        with self.subTest("missing cache is stale"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            if os.path.exists(self.cache):
                os.remove(self.cache)
            with patch.object(wd.wc, "send_ntfy", return_value=True) as ntfy, \
                 patch.object(wd.wc, "send_email", return_value=True):
                self.assertTrue(wd.check_once(now=now))
                ntfy.assert_called_once()


class TestEventDrivenGating(unittest.TestCase):
    """Gating flota-vie pt cache-urile event-driven (order/trade) — 28 iul.
    A stale fill cache must NOT alarm while the fleet is demonstrably
    alive (a fast price cache is fresh)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        wd._CACHE_DIR = Path(self.tmp)
        wd.STATE_FILE = os.path.join(self.tmp, ".state.json")
        wd.STALE_MINUTES = 20
        wd.COOLDOWN_MINUTES = 60
        self.price = os.path.join(self.tmp, "cache_prices_multi.json")   # fleet-alive
        self.order = os.path.join(self.tmp, "cache_order.json")          # event-driven

    def test_event_driven_gating_behaviors(self):
        now = time.time()
        
        with self.subTest("stale but fleet alive -> no alarm"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            _write_cache(self.price, int(now * 1000))                        # A FRESH price -> the fleet is alive.
            _write_cache(self.order, int((now - 90 * 3600) * 1000),          # A fill 90h old (>the 72h threshold).
                         mtime_sec=now - 90 * 3600)
            with patch.object(wd.wc, "send_ntfy") as ntfy, patch.object(wd.wc, "send_email") as email:
                self.assertFalse(wd.check_once(now=now), "a live fleet -> a stale fill is benign, no alarm")
                ntfy.assert_not_called()
                email.assert_not_called()

        with self.subTest("stale and fleet dead -> alarms"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            # BOTH stale: an old price (a dead fleet) plus an old fill -> an alarm (fail-safe).
            _write_cache(self.price, int((now - 3600) * 1000), mtime_sec=now - 3600)
            _write_cache(self.order, int((now - 90 * 3600) * 1000), mtime_sec=now - 90 * 3600)
            with patch.object(wd.wc, "send_ntfy", return_value=True) as ntfy, \
                 patch.object(wd.wc, "send_email", return_value=True):
                self.assertTrue(wd.check_once(now=now), "flota moarta -> alarma trece")
                ntfy.assert_called_once()

        with self.subTest("beyond hard ceiling -> alarms even if fleet alive"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            _write_cache(self.price, int(now * 1000))                        # flota vie
            # A fill older than the hard cap (30 days) -> an alarm regardless.
            old = now - (wd._EVENT_DRIVEN_HARD_CEILING_MIN + 60) * 60
            _write_cache(self.order, int(old * 1000), mtime_sec=old)
            with patch.object(wd.wc, "send_ntfy", return_value=True) as ntfy, \
                 patch.object(wd.wc, "send_email", return_value=True):
                self.assertTrue(wd.check_once(now=now), "above the hard cap -> an alarm even with a live fleet")
                ntfy.assert_called_once()


class TestFastPriceThreshold(unittest.TestCase):
    """A dedicated tight threshold (5min) plus restart eligibility ONLY for the
    fast price caches (~1s), written by cacheManager.py itself. The sparse .jsonl archive
    stays on the general threshold and does NOT trigger a restart (28 Jul)."""

    def test_cache_classification(self):
        with self.subTest("fast price caches classified"):
            for n in ("cache_currentprice.json",
                      "cache_instant_trend.json", "cache_24price_HYPEUSD.json",
                      "cache_24price_BTCUSDC.json"):
                self.assertTrue(wd._is_fast_price_cache(n), n)
                self.assertEqual(wd._threshold_for(n), wd._FAST_PRICE_THRESHOLD_MIN, n)

        with self.subTest("sparse and slow not fast"):
            # sparse .jsonl / archiver plus slow, event-driven ones are NOT fast (no restart)
            for n in ("cache_price_BTCUSDC.jsonl", "cache_24price_long_BTCUSDC.jsonl",
                      "cache_price_long_trend.json", "cache_order.json", "cache_asset_value.json"):
                self.assertFalse(wd._is_fast_price_cache(n), n)
                
        with self.subTest("cache prices multi reclassified slow override"):
            self.assertFalse(wd._is_fast_price_cache("cache_prices_multi.json"))
            self.assertEqual(wd._threshold_for("cache_prices_multi.json"), 8)
            self.assertNotEqual(wd._threshold_for("cache_prices_multi.json"),
                               wd._FAST_PRICE_THRESHOLD_MIN)

    def test_active_migrated_cache_hides_frozen_legacy_sibling(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        original = wd._CACHE_DIR
        try:
            wd._CACHE_DIR = Path(tmp)
            for name in (
                    "cache_order.json", "cache_order.jsonl",
                    "cache_trade.json", "cache_trade.jsonl",
                    "cache_asset_value.json", "cache_asset_value.jsonl"):
                (wd._CACHE_DIR / name).write_text("{}", encoding="utf-8")
            names = {path.name for path in wd._cache_files()}
            self.assertIn("cache_order.json", names)
            self.assertIn("cache_trade.jsonl", names)
            self.assertIn("cache_asset_value.jsonl", names)
            self.assertNotIn("cache_order.jsonl", names)
            self.assertNotIn("cache_trade.json", names)
            self.assertNotIn("cache_asset_value.json", names)
        finally:
            wd._CACHE_DIR = original

    def test_sparse_archive_stall_does_not_restart(self):
        """A stale sparse .jsonl cache (not fast) -> alarm, but NO restart."""
        import tempfile
        tmp = tempfile.mkdtemp()
        wd._CACHE_DIR = Path(tmp)
        wd.STATE_FILE = os.path.join(tmp, ".state.json")
        wd.AUTO_RESTART = True
        now = time.time()
        jf = os.path.join(tmp, "cache_price_BTCUSDC.jsonl")
        with open(jf, "w") as f:
            f.write(json.dumps({"s": "BTCUSDC", "i": [int((now - 3600) * 1000), 65000.0]}) + "\n")
        os.utime(jf, (now - 3600, now - 3600))
        with patch.object(wd, "_do_restart", return_value=True) as restart, \
             patch.object(wd.wc, "send_ntfy", return_value=True), \
             patch.object(wd.wc, "send_email", return_value=True):
            wd.check_once(now=now)
            restart.assert_not_called()


class TestAutoRestart(unittest.TestCase):
    """Auto-restart on a REAL price stall (28 Jul). _do_restart is mocked —
    NO real process is touched in the tests."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        wd._CACHE_DIR = Path(self.tmp)
        wd.STATE_FILE = os.path.join(self.tmp, ".state.json")
        wd.STALE_MINUTES = 20
        wd.COOLDOWN_MINUTES = 60
        wd.AUTO_RESTART = True
        wd.AUTO_RESTART_COOLDOWN_MIN = 15
        wd.AUTO_RESTART_MAX = 3
        wd.AUTO_RESTART_WINDOW_H = 6
        # 30 Jul: cache_currentprice.json (not cache_prices_multi.json — see
        # test_cache_prices_multi_is_not_restart_eligible mai jos, acela e al
        # market_alerts.py, not cacheManager.py, so it no longer triggers a restart).
        self.price = os.path.join(self.tmp, "cache_currentprice.json")   # The FAST cache, written by cacheManager.
        self.order = os.path.join(self.tmp, "cache_order.json")          # event-driven (in overrides)

    def _stale_price(self, now):
        _write_cache(self.price, int((now - 3600) * 1000), mtime_sec=now - 3600)  # 1h stale

    def test_restart_eligibility_behaviors(self):
        now = time.time()
        
        with self.subTest("fast cache stall triggers restart"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            self._stale_price(now)
            with patch.object(wd, "_do_restart", return_value=True) as restart, \
                 patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                self.assertTrue(wd.check_once(now=now))
                restart.assert_called_once()

        with self.subTest("disabled flag no restart"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            wd.AUTO_RESTART = False
            self._stale_price(now)
            with patch.object(wd, "_do_restart", return_value=True) as restart, \
                 patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                wd.check_once(now=now)
                restart.assert_not_called()
            wd.AUTO_RESTART = True

        with self.subTest("slow cache stall does not restart"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            if os.path.exists(self.price): os.remove(self.price)
            _write_cache(self.order, int((now - 100 * 3600) * 1000), mtime_sec=now - 100 * 3600)
            with patch.object(wd, "_do_restart", return_value=True) as restart, \
                 patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                wd.check_once(now=now)
                restart.assert_not_called()

        with self.subTest("cache prices multi is not restart eligible"):
            if os.path.exists(wd.STATE_FILE): os.remove(wd.STATE_FILE)
            if os.path.exists(self.price): os.remove(self.price)
            if os.path.exists(self.order): os.remove(self.order)
            multi = os.path.join(self.tmp, "cache_prices_multi.json")
            _write_cache(multi, int((now - 3600) * 1000), mtime_sec=now - 3600)  # 1h stale
            with patch.object(wd, "_do_restart", return_value=True) as restart, \
                 patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                self.assertTrue(wd.check_once(now=now), "it must still alarm")
                restart.assert_not_called()

    def test_restart_rate_limits(self):
        now = time.time()
        
        with self.subTest("cooldown blocks second restart"):
            if os.path.exists(wd.STATE_FILE):
                os.remove(wd.STATE_FILE)
            self._stale_price(now)
            with patch.object(wd, "_do_restart", return_value=True) as restart, \
                 patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                wd.check_once(now=now)                       # restart 1
                wd.check_once(now=now + 5 * 60)              # +5min < the 15min cooldown -> NO.
                self.assertEqual(restart.call_count, 1, "the cooldown must block the 2nd restart")
                wd.check_once(now=now + 20 * 60)             # +20min > cooldown -> restart 2
                self.assertEqual(restart.call_count, 2)

        with self.subTest("max per window then escalates"):
            if os.path.exists(wd.STATE_FILE):
                os.remove(wd.STATE_FILE)
            self._stale_price(now)
            with patch.object(wd, "_do_restart", return_value=True) as restart, \
                 patch.object(wd.wc, "send_ntfy", return_value=True), \
                 patch.object(wd.wc, "send_email", return_value=True):
                wd.check_once(now=now)                   # 1
                wd.check_once(now=now + 16 * 60)         # 2 (past the cooldown)
                wd.check_once(now=now + 32 * 60)         # 3
                wd.check_once(now=now + 48 * 60)         # The 4th: THE CAP -> no restart.
                self.assertEqual(restart.call_count, 3, "the cap of 3 per window must be honoured")


class TestConfigWatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = os.path.join(self.tmp, "fake.conf")
        self._root0, self._owners0, self._cr0 = wd._ROOT, wd._CONFIG_OWNERS, wd.CONFIG_RESTART
        wd._ROOT = Path(self.tmp)
        wd.STATE_FILE = os.path.join(self.tmp, ".state.json")
        wd._CONFIG_OWNERS = {"fake.conf": ["fakeproc.py"]}
        wd.CONFIG_RESTART = True
        wd.CONFIG_RESTART_COOLDOWN_MIN = 0   # No cooldown in the test.
        with open(self.cfg, "w") as f:
            f.write("val = 1\n")

    def tearDown(self):
        wd._ROOT, wd._CONFIG_OWNERS, wd.CONFIG_RESTART = self._root0, self._owners0, self._cr0

    def test_config_watch_behaviors(self):
        with self.subTest("baseline then change then debounce"):
            with patch.object(wd, "_do_restart") as restart, \
                 patch.object(wd.wc, "send_ntfy"), patch.object(wd.wc, "send_email"):
                # 1) first sighting = baseline only, NO restart
                self.assertEqual(wd.check_configs_once(), [])
                restart.assert_not_called()
                # 2) continut schimbat -> restart proprietarul
                with open(self.cfg, "w") as f:
                    f.write("val = 2\n")
                self.assertEqual(wd.check_configs_once(), ["fakeproc.py"])
                restart.assert_called_once_with("fakeproc.py")
                # 3) unchanged -> debounce, no second restart
                restart.reset_mock()
                self.assertEqual(wd.check_configs_once(), [])
                restart.assert_not_called()

        with self.subTest("mtime touch without content change no restart"):
            # reset config and state for this subtest
            with open(self.cfg, "w") as f:
                f.write("val = 1\n")
            if os.path.exists(wd.STATE_FILE):
                os.remove(wd.STATE_FILE)
            with patch.object(wd, "_do_restart") as restart, \
                 patch.object(wd.wc, "send_ntfy"), patch.object(wd.wc, "send_email"):
                wd.check_configs_once()                       # baseline
                os.utime(self.cfg, (time.time() + 100, time.time() + 100))  # Only mtime, identical content.
                self.assertEqual(wd.check_configs_once(), [])  # An identical hash -> nothing.
                restart.assert_not_called()

        with self.subTest("kill switch off detects but no restart"):
            wd.CONFIG_RESTART = False
            # reset state
            if os.path.exists(wd.STATE_FILE):
                os.remove(wd.STATE_FILE)
            with patch.object(wd, "_do_restart") as restart, \
                 patch.object(wd.wc, "send_ntfy"), patch.object(wd.wc, "send_email"):
                wd.check_configs_once()                       # baseline
                with open(self.cfg, "w") as f:
                    f.write("val = 3\n")
                self.assertEqual(wd.check_configs_once(), [])  # It detects but does NOT restart.
                restart.assert_not_called()
            wd.CONFIG_RESTART = True

if __name__ == "__main__":
    unittest.main(verbosity=2)
