"""
Tests for the bounded wait used by non-Binance price polling.

NonBinanceTrendPoller once froze when get_current_price(HYPEUSD) blocked
during DNS resolution. The fetch now runs in a separate worker, while
future.result(timeout) prevents the poller itself from waiting indefinitely.
The provider client remains responsible for ending the network call.
"""
import os
import sys
import time
import unittest
import concurrent.futures as futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import cacheManager as cm


class _FakeApi:
    """A fake market_api: get_current_price either returns quickly or BLOCKS
    (simulating a DNS/network hang), depending on the flag."""
    def __init__(self, hang=False, price=55.5):
        self.hang = hang
        self.price = price
        self.calls = 0
        self.provider_names = []

    def get_current_price(self, symbol=None, *, provider_name=None):
        self.calls += 1
        self.provider_names.append(provider_name)
        # A TRANSIENT blockage: it stays blocked while the flag is True, and unblocks
        # when the network "comes back" (hang=False) — like a DNS/socket recovering.
        waited = 0.0
        while self.hang and waited < 30:
            time.sleep(0.05); waited += 0.05
        return self.price


class TestFetchDeadline(unittest.TestCase):
    def setUp(self):
        self.pool = futures.ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        if hasattr(self, "blocked_api"):
            self.blocked_api.hang = False
        self.pool.shutdown(wait=True, cancel_futures=True)

    def test_normal_fetch_returns_price(self):
        api = _FakeApi(hang=False, price=55.5)
        p = cm._fetch_price_with_deadline(api, "HYPEUSD", self.pool, deadline_sec=2)
        self.assertEqual(p, 55.5)

    def test_explicit_provider_is_forwarded_inside_deadline(self):
        api = _FakeApi(hang=False, price=85.49)
        p = cm._fetch_price_with_deadline(
            api, "HYPEUSD", self.pool, deadline_sec=2,
            provider_name="Kraken")
        self.assertEqual(p, 85.49)
        self.assertEqual(api.provider_names, ["Kraken"])

    def test_hung_fetch_raises_timeout_not_blocks(self):
        api = _FakeApi(hang=True)
        self.blocked_api = api
        t0 = time.time()
        with self.assertRaises(futures.TimeoutError):
            cm._fetch_price_with_deadline(api, "HYPEUSD", self.pool, deadline_sec=1)
        # The poller wait returns near its deadline without waiting out the worker.
        self.assertLess(
            time.time() - t0,
            5,
            "the poll wait must return in about 1s, not wait for the 30s hang",
        )

    def test_poller_requires_a_provider_for_every_symbol(self):
        api = _FakeApi()

        class _Cpm:
            market_api = api

            def __init__(self):
                self.bind_called = False

            def bind_providers(self, _provider_names):
                self.bind_called = True

        invalid_bindings = (
            None,
            {"HYPEUSD": "Kraken"},
            {"HYPEUSD": "Kraken", "TAOUSD": "Binance"},
        )
        for provider_names in invalid_bindings:
            with self.subTest(provider_names=provider_names):
                cpm = _Cpm()
                with self.assertRaisesRegex(
                    ValueError, "explicit provider bindings are required"
                ):
                    cm._start_nonbinance_trend_poller(
                        cpm,
                        ["HYPEUSD", "TAOUSD"],
                        provider_names=provider_names,
                    )
                self.assertFalse(cpm.bind_called)
                self.assertEqual(api.calls, 0)

    def test_poller_survives_a_hang_and_recovers(self):
        """The poller, with a fetch that hangs and then recovers, must NOT
        freeze: after a hang, on a later cycle with a good fetch, it pushes the price."""
        pushed = []
        api = _FakeApi(hang=True)

        class _Cpm:
            market_api = api

            def __init__(self):
                self.bindings = {}

            def bind_providers(self, provider_names):
                self.bindings.update(provider_names or {})

            def provider_name_for(self, symbol):
                return self.bindings.get(symbol)

            def _push_price(self, s, p):
                pushed.append((s, p))

        cpm = _Cpm()
        # A small interval plus a short deadline, so we can observe the timeout and the recovery in the test.
        poller = cm._start_nonbinance_trend_poller(
            cpm, ["HYPEUSD"], interval_sec=0.3, fetch_deadline_sec=0.5,
            provider_names={"HYPEUSD": "Kraken"})
        self.addCleanup(poller.stop)
        time.sleep(2)            # while the fetch is blocked -> timeouts, NOT pushes
        self.assertEqual(pushed, [], "while the fetch is blocked nothing must be pushed")
        api.hang = False         # The network comes back -> the blocked fetches unblock.
        time.sleep(3)            # A few cycles -> it should push the price.
        self.assertTrue(any(s == "HYPEUSD" for s, _ in pushed),
                        "once the network returns, the poller must recover and push the price")
        self.assertEqual(cpm.bindings, {"HYPEUSD": "Kraken"})
        self.assertTrue(api.provider_names)
        self.assertEqual(set(api.provider_names), {"Kraken"})

    def test_poller_marks_nonbinance_symbol_healthy_on_cpm(self):
        import tempfile
        api = _FakeApi(hang=False, price=76.5)
        tmp_dir = tempfile.mkdtemp()
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999,
            symbols=["HYPEUSD"],
            filename=os.path.join(tmp_dir, "cache_current_price.json"),
            ws_manager=None,
            api_client=api,
            market_api=api,
            provider_names={"HYPEUSD": "Hyperliquid"},
            read_only=True,
        )
        self.assertFalse(cpm._ws_is_healthy("HYPEUSD"))
        poller = cm._start_nonbinance_trend_poller(
            cpm,
            ["HYPEUSD"],
            interval_sec=0.2,
            fetch_deadline_sec=1.0,
            provider_names={"HYPEUSD": "Hyperliquid"},
        )
        self.addCleanup(poller.stop)
        deadline = time.time() + 2.0
        while time.time() < deadline and not cpm._ws_is_healthy("HYPEUSD"):
            time.sleep(0.05)
        self.assertTrue(cpm._ws_is_healthy("HYPEUSD"))


if __name__ == "__main__":
    unittest.main()
