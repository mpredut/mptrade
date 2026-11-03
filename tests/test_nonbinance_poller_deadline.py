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

        # Non-finite or boolean prices must never be pushed or mark the symbol healthy
        cpm._ws_last_event_ts_per_symbol.clear()
        cpm.cache.clear()
        for bad_price in (float("nan"), float("inf"), float("-inf"), True, False, 0.0, -1.0):
            cpm._push_price("HYPEUSD", bad_price)
        self.assertEqual(cpm.cache, {})

    def test_hyperliquid_and_kraken_ws_streams_feed_cpm_and_suppress_rest_polling(self):
        import json
        import tempfile

        api = _FakeApi(hang=False, price=99.9)
        tmp_dir = tempfile.mkdtemp()
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999,
            symbols=["HYPEUSDC", "HYPEUSD", "TAOUSD", "ADAUSD"],
            filename=os.path.join(tmp_dir, "cache_current_price.json"),
            ws_manager=None,
            api_client=api,
            market_api=api,
            provider_names={
                "HYPEUSDC": "hyperliquid",
                "HYPEUSD": "kraken",
                "TAOUSD": "kraken",
                "ADAUSD": "kraken",
            },
            read_only=True,
        )
        hl_stream = cm.HyperliquidPriceStream(["HYPEUSDC"], market_api=api)
        krk_stream = cm.KrakenPriceStream(["HYPEUSD", "TAOUSD", "ADAUSD"])
        cpm.attach_ws_manager(hl_stream)
        cpm.attach_ws_manager(krk_stream)
        self.assertEqual(hl_stream.running_symbols, {"HYPEUSDC"})
        self.assertEqual(krk_stream.running_symbols, {"HYPEUSD", "TAOUSD", "ADAUSD"})

        now_ms = int(time.time() * 1000) - 2000
        hl_stream._process_message(
            json.dumps({"channel": "allMids", "data": {"mids": {"@107": "42.125"}}}),
            now_ms=now_ms,
        )
        krk_stream._process_message(
            json.dumps([42, {"c": ["42.05", "1.0"], "b": ["42.00", "1"], "a": ["42.10", "1"]}, "ticker", "HYPE/USD"]),
            now_ms=now_ms,
        )
        krk_stream._process_message(
            json.dumps([43, {"c": ["355.0", "1.0"], "b": ["354.5", "1"], "a": ["355.5", "1"]}, "ticker", "TAO/USD"]),
            now_ms=now_ms,
        )
        krk_stream._process_message(
            json.dumps([44, {"c": ["0.80", "1.0"], "b": ["0.74", "1"], "a": ["0.76", "1"]}, "ticker", "ADA/USD"]),
            now_ms=now_ms,
        )

        # Verify Kraken heartbeat refreshes known prices after >=1000ms so get_price never sees false staleness
        hb_ms = now_ms + 1500
        krk_stream._process_message(json.dumps({"event": "heartbeat"}), now_ms=hb_ms)
        self.assertEqual(cpm.cache["TAOUSD"][0][0], hb_ms)
        self.assertAlmostEqual(hl_stream.get_price("HYPEUSDC"), 42.125)
        self.assertAlmostEqual(krk_stream.get_price("ADAUSD"), 0.75)
        self.assertEqual(set(krk_stream.get_all_prices().keys()), {"HYPEUSD", "TAOUSD", "ADAUSD"})

        # Verify WS ticks updated cpm cache and marked real WS stream healthy
        self.assertTrue(cpm._real_ws_is_healthy("HYPEUSDC"))
        self.assertTrue(cpm._real_ws_is_healthy("HYPEUSD"))
        self.assertTrue(cpm._real_ws_is_healthy("TAOUSD"))
        self.assertTrue(cpm._real_ws_is_healthy("ADAUSD"))
        self.assertAlmostEqual(cpm.get_cached_price_observation("HYPEUSDC", 5.0)[1], 42.125)
        self.assertAlmostEqual(cpm.get_cached_price_observation("HYPEUSD", 5.0)[1], 42.05)
        self.assertAlmostEqual(cpm.get_cached_price_observation("TAOUSD", 5.0)[1], 355.0)
        # ADA last trade 0.80 was outside [0.74, 0.76] spread -> clamped to midpoint 0.75
        self.assertAlmostEqual(cpm.get_cached_price_observation("ADAUSD", 5.0)[1], 0.75)

        # While real WS is healthy, _start_nonbinance_trend_poller must NOT make any REST calls
        poller = cm._start_nonbinance_trend_poller(
            cpm,
            ["HYPEUSDC", "HYPEUSD", "TAOUSD", "ADAUSD"],
            interval_sec=0.15,
            fetch_deadline_sec=0.5,
            provider_names={
                "HYPEUSDC": "hyperliquid",
                "HYPEUSD": "kraken",
                "TAOUSD": "kraken",
                "ADAUSD": "kraken",
            },
            start_ws=False,
        )
        self.addCleanup(poller.stop)
        time.sleep(0.35)
        self.assertEqual(api.calls, 0, "REST poller must not fire while WebSocket stream is healthy")

        # Simulate WS stream going stale for HYPEUSDC only -> poller falls back to REST only for HYPEUSDC
        cpm._ws_stream_event_ts_per_symbol["HYPEUSDC"] = time.time() - 60.0
        time.sleep(0.35)
        self.assertGreaterEqual(api.calls, 1, "REST poller must fall back when WebSocket stream goes stale")
        self.assertEqual(set(api.provider_names), {"hyperliquid"})

    def test_hyperliquid_ws_user_fills_updates_ssot_trade_cache(self):
        import json
        import tempfile
        from providers.hyperliquid_provider import HyperliquidProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cache_trade_hyperliquid.json")
            hl_ws = cm.HyperliquidPriceStream(
                ["HYPEUSDC"],
                user_address="0x1234567890abcdef1234567890abcdef12345678",
                fills_cache_path=cache_file,
            )
            extra = hl_ws._build_extra_subscribe_payloads()
            self.assertEqual(len(extra), 1)
            self.assertEqual(
                json.loads(extra[0]),
                {
                    "method": "subscribe",
                    "subscription": {
                        "type": "userFills",
                        "user": "0x1234567890abcdef1234567890abcdef12345678",
                    },
                },
            )
            now_ms = int(time.time() * 1000)
            hl_ws._process_message(
                json.dumps(
                    {
                        "channel": "userFills",
                        "data": {
                            "isSnapshot": True,
                            "user": "0x1234567890abcdef1234567890abcdef12345678",
                            "fills": [
                                {"coin": "@107", "oid": 101, "tid": 1, "side": "B", "px": "42.0", "sz": "2.0", "time": now_ms - 5000},
                                {"coin": "HYPE", "oid": 999, "tid": 2, "side": "A", "px": "99.0", "sz": "10.0", "time": now_ms - 4000},
                            ],
                        },
                    }
                ),
                now_ms=now_ms,
            )
            hl_ws._process_message(
                json.dumps(
                    {
                        "channel": "userFills",
                        "data": {
                            "isSnapshot": False,
                            "user": "0x1234567890abcdef1234567890abcdef12345678",
                            "fills": [
                                {"coin": "@107", "oid": 102, "tid": 3, "side": "A", "px": "44.5", "sz": "2.0", "time": now_ms - 1000},
                            ],
                        },
                    }
                ),
                now_ms=now_ms + 100,
            )
            prov = HyperliquidProvider(token="HYPE")
            prov._ssot_fills_cache_path = cache_file
            prov._force_ssot_cache = True
            orders = prov.get_orders("HYPEUSDC", None, 3600)
            self.assertEqual(len(orders), 2)
            self.assertEqual([o["orderId"] for o in orders], ["101", "102"])
            self.assertEqual([o["side"] for o in orders], ["BUY", "SELL"])


if __name__ == "__main__":
    unittest.main()
