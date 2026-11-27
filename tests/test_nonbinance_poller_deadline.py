"""
Tests for unified multi-venue price polling and non-blocking REST slot tracking.

CacheCurrentPriceManager once delegated non-Binance symbols to a separate poller
that froze when get_current_price(HYPEUSD) blocked during DNS resolution. All
venues now share unified CacheCurrentPriceManager WS streams and NonBlockingRestSlotTracker
with WS preemption and hard-timeout eviction/reset.
"""

import cacheManager as cm
from tests.conftest import TempDirTestCase
import cacheprice
import concurrent.futures as futures
import json
import os
import time
from binance_api.bapi_ws import BinancePriceStream, PublicPriceStreamBase
from botcore import (
    NonBlockingRestSlotTracker,
    clamp_price_to_bbo_mid,
    compute_rest_error_backoff_sec,
    is_rate_limit_error,
    shutdown_daemon_executor,
)
import cacheaccount
from intelligence.sentiment.collectors.portfolio_collector import _load_open_order_timestamps
from intelligence.sentiment.evaluators.price_series_evaluator import recalculate_live_price_geometry
from providers.execution_audit import find_duplicate_ai_open_limit_order
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import KrakenProvider
from providers.order_ohlc_utils import extract_order_fee, validate_ohlc_candle_bar
from providers.price_poller import VenuePriceStreamBase
from trailing_core import TrailingAdapterBase
from types import SimpleNamespace




class _StubCpm:
    def __init__(self, market_api, *, pushed=None):
        self.market_api = market_api
        self.bind_called = False
        self.bindings = {}
        self._pushed = pushed

    def bind_providers(self, provider_names):
        self.bind_called = True
        self.bindings.update(provider_names or {})

    def provider_name_for(self, symbol):
        return self.bindings.get(symbol)

    def push_price(self, s, p):
        if self._pushed is not None:
            self._pushed.append((s, p))


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
        # when the network "comes back" (hang=False) -- like a DNS/socket recovering.
        waited = 0.0
        while self.hang and waited < 30:
            time.sleep(0.05); waited += 0.05
        return self.price


class TestFetchDeadline(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.pool = futures.ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        if hasattr(self, "blocked_api"):
            self.blocked_api.hang = False
        self.pool.shutdown(wait=True, cancel_futures=True)
        super().tearDown()

    def test_normal_fetch_returns_price(self):
        api = _FakeApi(hang=False, price=55.5)
        p = cacheprice._fetch_price_with_deadline(api, "HYPEUSD", self.pool, deadline_sec=2)
        self.assertEqual(p, 55.5)

    def test_explicit_provider_is_forwarded_inside_deadline(self):
        api = _FakeApi(hang=False, price=85.49)
        p = cacheprice._fetch_price_with_deadline(
            api, "HYPEUSD", self.pool, deadline_sec=2,
            provider_name="Kraken")
        self.assertEqual(p, 85.49)
        self.assertEqual(api.provider_names, ["Kraken"])

    def test_hung_fetch_raises_timeout_not_blocks(self):
        api = _FakeApi(hang=True)
        self.blocked_api = api
        t0 = time.time()
        with self.assertRaises(futures.TimeoutError):
            cacheprice._fetch_price_with_deadline(api, "HYPEUSD", self.pool, deadline_sec=1)
        # The poller wait returns near its deadline without waiting out the worker.
        self.assertLess(
            time.time() - t0,
            5,
            "the poll wait must return in about 1s, not wait for the 30s hang",
        )

    def test_poller_requires_a_provider_for_every_symbol(self):
        """CacheCurrentPriceManager.bind_providers rejects empty or conflicting venue bindings."""
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=1.0,
            symbols=["HYPEUSD", "TAOUSD"],
            filename=os.path.join(self.tmp, "cpm_bindings.json"),
            provider_names={"HYPEUSD": "Kraken"},
            read_only=True,
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            cpm.bind_provider("TAOUSD", "")
        with self.assertRaisesRegex(ValueError, "conflicting providers"):
            cpm.bind_provider("HYPEUSD", "Hyperliquid")
        self.assertEqual(cpm.provider_name_for("HYPEUSD"), "Kraken")

    def _empty_cp_file(self, name: str) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"items": {}, "fetchtime": {}}, fh)
        return path

    def test_poller_survives_a_hang_and_recovers(self):
        """Unified CacheCurrentPriceManager with a fetch that hangs and then recovers must NOT
        freeze: while hung it stays non-blocking, and once unblocked it harvests and pushes the price."""
        api = _FakeApi(hang=True, price=55.5)
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=0.05,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cpm_hang_recover.json"),
            market_api=api,
            provider_names={"HYPEUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm._is_live_daemon_poll_mode = lambda: True
        cpm._is_live_network_mode = lambda: True
        self.addCleanup(lambda: cpm.shutdown(timeout=2.0))
        try:
            t0 = time.monotonic()
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertLess(time.monotonic() - t0, 0.25)
            self.assertNotIn("HYPEUSD", cpm.cache)
            api.hang = False
            cpm._in_flight_http_futs["HYPEUSD"].future.result(timeout=2.0)
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertAlmostEqual(cpm.cache["HYPEUSD"][-1][1], 55.5)
            self.assertEqual(set(api.provider_names), {"Kraken"})
        finally:
            api.hang = False

    def test_poller_marks_nonbinance_symbol_healthy_on_cpm(self):
        api = _FakeApi(hang=False, price=76.5)
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=0.05,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cache_current_price_hl.json"),
            ws_manager=None,
            api_client=api,
            market_api=api,
            provider_names={"HYPEUSD": "Hyperliquid"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm._is_live_daemon_poll_mode = lambda: True
        cpm._is_live_network_mode = lambda: True
        self.addCleanup(lambda: cpm.shutdown(timeout=2.0))
        self.assertFalse(cpm._ws_is_healthy("HYPEUSD"))
        self.assertTrue(cpm.query_remote_and_update_cache())
        if "HYPEUSD" in cpm._in_flight_http_futs:
            cpm._in_flight_http_futs["HYPEUSD"].future.result(timeout=2.0)
            self.assertTrue(cpm.query_remote_and_update_cache())
        self.assertAlmostEqual(cpm.cache["HYPEUSD"][-1][1], 76.5)
        # Non-finite or boolean prices must never be pushed or mark the symbol healthy
        cpm._ws_last_event_ts_per_symbol.clear()
        cpm.cache.clear()
        for bad_price in (float("nan"), float("inf"), float("-inf"), True, False, 0.0, -1.0):
            cpm.push_price("HYPEUSD", bad_price)
        self.assertEqual(cpm.cache, {})

    def test_hyperliquid_and_kraken_ws_streams_feed_cpm_and_suppress_rest_polling(self):
        api = _FakeApi(hang=False, price=99.9)
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=0.05,
            symbols=["HYPEUSDC", "HYPEUSD", "TAOUSD", "ADAUSD"],
            filename=self._empty_cp_file("cache_current_price_ws.json"),
            ws_manager=None,
            api_client=api,
            market_api=api,
            provider_names={
                "HYPEUSDC": "hyperliquid",
                "HYPEUSD": "kraken",
                "TAOUSD": "kraken",
                "ADAUSD": "kraken",
            },
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm._is_live_daemon_poll_mode = lambda: True
        cpm._is_live_network_mode = lambda: True
        self.addCleanup(lambda: cpm.shutdown(timeout=2.0))
        hl_stream = cm.HyperliquidPriceStream(["HYPEUSDC"], market_api=api)
        krk_stream = cm.KrakenPriceStream(["HYPEUSD", "TAOUSD", "ADAUSD"])
        cpm.attach_ws_manager(hl_stream)
        cpm.attach_ws_manager(krk_stream)
        self.assertEqual(hl_stream.running_symbols, {"HYPEUSDC"})
        self.assertEqual(krk_stream.running_symbols, {"HYPEUSD", "TAOUSD", "ADAUSD"})

        now_ms = int(time.time() * 1000) - 1200
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
        hb_ms = now_ms + 1150
        hl_stream._process_message(
            json.dumps({"channel": "allMids", "data": {"mids": {"@107": "42.125"}}}),
            now_ms=hb_ms,
        )
        krk_stream._process_message(json.dumps({"event": "heartbeat"}), now_ms=hb_ms)
        self.assertEqual(cpm.cache["TAOUSD"][0][0], hb_ms)
        self.assertAlmostEqual(hl_stream.get_price("HYPEUSDC"), 42.125)
        self.assertAlmostEqual(krk_stream.get_price("ADAUSD"), 0.75)
        self.assertEqual(set(krk_stream.get_all_prices().keys()), {"HYPEUSD", "TAOUSD", "ADAUSD"})

        # Verify WS ticks updated cpm cache and marked real WS stream healthy
        self.assertTrue(cpm.is_real_ws_healthy("HYPEUSDC"))
        self.assertTrue(cpm.is_real_ws_healthy("HYPEUSD"))
        self.assertTrue(cpm.is_real_ws_healthy("TAOUSD"))
        self.assertTrue(cpm.is_real_ws_healthy("ADAUSD"))
        self.assertAlmostEqual(cpm.get_cached_price_observation("HYPEUSDC", 5.0)[1], 42.125)
        self.assertAlmostEqual(cpm.get_cached_price_observation("HYPEUSD", 5.0)[1], 42.05)
        self.assertAlmostEqual(cpm.get_cached_price_observation("TAOUSD", 5.0)[1], 355.0)
        # ADA last trade 0.80 was outside [0.74, 0.76] spread -> clamped to midpoint 0.75
        self.assertAlmostEqual(cpm.get_cached_price_observation("ADAUSD", 5.0)[1], 0.75)

        # While real WS is healthy, unified CacheCurrentPriceManager must NOT make any REST calls
        self.assertFalse(cpm._should_poll())
        self.assertTrue(cpm.query_remote_and_update_cache())
        self.assertEqual(api.calls, 0, "REST fallback must not fire while WebSocket stream is healthy")

        # Simulate WS stream going stale for HYPEUSDC only -> falls back to REST ONLY for HYPEUSDC
        cpm._ws_stream_event_ts_per_symbol["HYPEUSDC"] = time.time() - 60.0
        cpm._ws_last_event_ts_per_symbol["HYPEUSDC"] = time.time() - 60.0
        self.assertTrue(cpm._should_poll())
        self.assertTrue(cpm.query_remote_and_update_cache())
        if "HYPEUSDC" in cpm._in_flight_http_futs:
            cpm._in_flight_http_futs["HYPEUSDC"].future.result(timeout=2.0)
            self.assertTrue(cpm.query_remote_and_update_cache())
        self.assertEqual(api.calls, 1, "REST fallback must poll only the stale symbol")
        self.assertEqual(set(api.provider_names), {"hyperliquid"})

    def test_hyperliquid_ws_user_fills_updates_ssot_trade_cache(self):

        cache_file = os.path.join(self.tmp, "cache_trade_hyperliquid.json")
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

        # Invalidating SSOT fills cache must not be resurrected by an allMids tick before a new fill arrives
        from unittest.mock import patch
        prov.invalidate_ssot_fills_cache("HYPEUSDC", cache_file)
        self.assertIsNone(prov.read_ssot_fills_cache("HYPEUSDC", cache_file, max_stale_sec=90.0))
        hl_ws._last_fills_cache_touch_ms = 0
        with patch.object(
            cacheaccount._HyperliquidAccountTradeBridge,
            "_schedule_invalidated_fills_resync",
            return_value=False,
        ):
            hl_ws._touch_fills_cache_if_needed(now_ms + 70_000)
        self.assertIsNone(prov.read_ssot_fills_cache("HYPEUSDC", cache_file, max_stale_sec=90.0))

        # Reconnect resets _fills_ws_seeded until a new isSnapshot=True message arrives
        _ = hl_ws._build_extra_subscribe_payloads()
        self.assertFalse(hl_ws._fills_ws_seeded)

    def test_validate_ohlc_close_ts_sentinel_and_duplicate_ai_order_and_trailing_lookup(self):

        # 1. Explicit close_ts=None must be rejected, whereas omitted close_ts is accepted
        self.assertIsNone(
            validate_ohlc_candle_bar(
                1700000000, 100.0, 105.0, 99.0, 102.0, close_ts=None
            )
        )
        self.assertIsNotNone(
            validate_ohlc_candle_bar(
                1700000000, 100.0, 105.0, 99.0, 102.0
            )
        )

        # 2. find_duplicate_ai_open_limit_order normalizes lowercase side and case-insensitive clientOrderId
        dup_oid = find_duplicate_ai_open_limit_order(
            SimpleNamespace(
                open_orders=lambda: [
                    {
                        "orderId": "9001",
                        "side": "SELL",
                        "type": "LIMIT",
                        "price": "50.0",
                        "origQty": "1.6",
                        "executedQty": "0.0",
                        "clientOrderId": "AI_ABC123DEF",
                    }
                ]
            ),
            side="sell",
            limit_price=50.0,
            notional_usd=100.0,
            intents=[
                {
                    "intent_id": "abc123def",
                    "suggested_notional_usd": 100.0,
                    "clamped_notional_usd": 80.0,
                }
            ],
        )
        self.assertEqual(dup_oid, "9001")

        # 3. TrailingAdapterBase._lookup_symbol_entry with predicate skips non-matching pair entry
        class _DummyTrailing(TrailingAdapterBase):
            _VENUE_SLUG = "kraken"
            _LOOKUP_BY_BASE_ASSET = True

        val = _DummyTrailing._lookup_symbol_entry(
            {"BTCUSD": "invalid", "BTC": "on"},
            "BTCUSD",
            predicate=lambda v: v in ("on", "off"),
        )
        self.assertEqual(val, "on")

    def test_ws_stream_dynamic_add_remove_and_provider_ssot_cache_helpers(self):
        # 1. Dynamic add_symbol / remove_symbol on KrakenPriceStream and HyperliquidPriceStream
        krk_stream = cm.KrakenPriceStream(["HYPEUSD"])
        self.assertTrue(krk_stream.add_symbol("BTC/USD"))
        self.assertIn("BTC/USD", json.loads(krk_stream._build_subscribe_payload())["pair"])
        now_ms = int(time.time() * 1000)
        krk_stream._process_message(
            json.dumps([99, {"c": ["95000.0", "0.1"], "b": ["94990.0", "1"], "a": ["95010.0", "1"]}, "ticker", "XBT/USD"]),
            now_ms=now_ms,
        )
        self.assertAlmostEqual(krk_stream.get_price("BTCUSD"), 95000.0)
        self.assertTrue(krk_stream.remove_symbol("BTCUSD"))
        self.assertIsNone(krk_stream.get_price("BTCUSD"))
        self.assertIsNone(krk_stream._resolve_symbol("XBT/USD"))

        hl_stream = cm.HyperliquidPriceStream(["HYPEUSDC"])
        _ = hl_stream._candidate_mids_keys("HYPEUSDC")
        self.assertIn("HYPEUSDC", hl_stream._resolved_spot_keys)
        self.assertTrue(hl_stream.remove_symbol("HYPEUSDC"))
        self.assertNotIn("HYPEUSDC", hl_stream._resolved_spot_keys)

        # 2. clamp_price_to_bbo_mid must not fabricate mid from wide (>5%) spread when price is None
        self.assertIsNone(clamp_price_to_bbo_mid(None, "10.0", "20.0"))
        self.assertAlmostEqual(clamp_price_to_bbo_mid(None, "100.0", "101.0"), 100.5)

        # 3. is_real_ws_healthy must not fall through to _ws_default_event_ts when symbol ts is 0.0
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999,
            symbols=["HYPEUSD"],
            filename=os.path.join(self.tmp, "cache_current_price.json"),
            ws_manager=None,
            api_client=_FakeApi(),
            market_api=_FakeApi(),
            provider_names={"HYPEUSD": "kraken"},
            read_only=True,
        )
        cpm._ws_last_event_ts = time.time()
        cpm._ws_stream_event_ts_per_symbol["HYPEUSD"] = 0.0
        self.assertFalse(cpm.is_real_ws_healthy("HYPEUSD"))

        # 4. read_ssot_fills_cache skips stale alias key and returns fresh matching key

        kp = KrakenProvider(client=object())
        fills_file = os.path.join(self.tmp, "cache_trade_kraken.json")
        with open(fills_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "items": {
                        "XBTUSD": [{"ordertxid": "O1", "type": "buy", "price": 90000.0, "vol": 0.1, "time": now_ms / 1000.0}],
                        "BTCUSD": [{"ordertxid": "O2", "type": "buy", "price": 95000.0, "vol": 0.2, "time": now_ms / 1000.0}],
                    },
                    "fetchtime": {
                        "XBTUSD": now_ms - 600_000,
                        "BTCUSD": now_ms,
                    },
                },
                fh,
            )
        rows = kp.read_ssot_fills_cache("BTCUSD", fills_file, max_stale_sec=90.0)
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ordertxid"], "O2")
        kp.invalidate_ssot_fills_cache("BTCUSD", fills_file)
        self.assertIsNone(kp.read_ssot_fills_cache("BTCUSD", fills_file, max_stale_sec=90.0))

        # 5. PublicPriceStreamBase shared inheritance and cross-venue position_cost_basis symmetry

        self.assertTrue(issubclass(BinancePriceStream, PublicPriceStreamBase))
        self.assertTrue(issubclass(cm.HyperliquidPriceStream, PublicPriceStreamBase))
        self.assertTrue(issubclass(cm.KrakenPriceStream, PublicPriceStreamBase))

        kr_cost_file = os.path.join(self.tmp, "kr_cost_cache.json")
        with open(kr_cost_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "items": {
                        "HYPEUSD": [
                            {
                                "id": "T1",
                                "ordertxid": "O10",
                                "side": "BUY",
                                "price": 40.0,
                                "qty": 2.0,
                                "timestamp": now_ms - 10_000,
                                "fee": 0.20,
                                "feeAsset": "USD",
                            },
                            {
                                "id": "T2",
                                "ordertxid": "O11",
                                "side": "BUY",
                                "price": 50.0,
                                "qty": 2.0,
                                "timestamp": now_ms - 5_000,
                                "fee": 0.20,
                                "feeAsset": "USD",
                            },
                        ]
                    },
                    "fetchtime": {"HYPEUSD": now_ms - 1_000},
                },
                fh,
            )
        kp._ssot_fills_cache_path = kr_cost_file
        # Total cost = (40*2 + 0.20) + (50*2 + 0.20) = 180.40 for 4.0 HYPE -> 45.10
        self.assertAlmostEqual(kp.position_cost_basis("HYPEUSD", 4.0, 3600.0), 45.10, places=4)

        hl_cost_file = os.path.join(self.tmp, "hl_cost_cache.json")
        with open(hl_cost_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "items": {
                        "HYPEUSDC": [
                            {
                                "tid": 1,
                                "oid": 201,
                                "side": "B",
                                "px": "30.0",
                                "sz": "3.0",
                                "time": now_ms - 8_000,
                                "fee": "0.30",
                                "feeToken": "USDC",
                            }
                        ]
                    },
                    "fetchtime": {"HYPEUSDC": now_ms - 1_000},
                },
                fh,
            )
        hlp = HyperliquidProvider(token="HYPE")
        hlp._ssot_fills_cache_path = hl_cost_file
        # Total cost = 30*3 + 0.30 = 90.30 for 3.0 HYPE -> 30.10
        self.assertAlmostEqual(hlp.position_cost_basis("HYPEUSDC", 3.0, 3600.0), 30.10, places=4)

        # 6. Kraken fee asset aliases (XXBT / ZUSD), live price geometry 15m/6h updates, and open order ts lookup from list intents

        fee_btc = extract_order_fee({"fee": 0.001, "feeAsset": "XXBT"}, base_asset="BTC", quote_asset="USD", fill_price=90000.0)
        self.assertIsNotNone(fee_btc)
        self.assertAlmostEqual(fee_btc["base_fee"], 0.001)
        self.assertAlmostEqual(fee_btc["fee_in_quote"], 90.0)
        fee_usd = extract_order_fee({"fee": 1.25, "feeAsset": "ZUSD"}, base_asset="BTC", quote_asset="USD", fill_price=90000.0)
        self.assertIsNotNone(fee_usd)
        self.assertAlmostEqual(fee_usd["quote_fee"], 1.25)

        pinfo = {
            "low_15m": 100.0,
            "high_15m": 102.0,
            "prior_6h_low_px": 100.0,
            "recent_1h_low_px": 100.5,
            "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
            "higher_low_vs_prior_pct": 0.5,
        }
        recalculate_live_price_geometry(pinfo, 99.5)
        self.assertEqual(pinfo["bottom_structure_6h"], "FRESH_6H_LOW_TEST")
        self.assertNotIn("higher_low_vs_prior_pct", pinfo)
        self.assertAlmostEqual(pinfo["low_15m"], 99.5)
        self.assertAlmostEqual(pinfo["pos_15m_pct"], 0.0)
        self.assertAlmostEqual(pinfo["span_15m_pct"], 2.51, places=2)

        intents_file = os.path.join(self.tmp, "autonomous_trade_intents_state.json")
        with open(intents_file, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {
                        "intent_id": "krk-hype-buy-1700000123",
                        "timestamp": 1700000123.5,
                        "execution_payload": {"txid": ["O-KRK-777"]},
                    }
                ],
                fh,
            )
        ts_map = _load_open_order_timestamps(self.tmp, {})
        self.assertAlmostEqual(ts_map.get("O-KRK-777"), 1700000123.5)
        self.assertAlmostEqual(ts_map.get("ai_krk-hype-buy-1700000123"), 1700000123.5)

    def test_poller_deduplicates_in_flight_hangs_and_backs_off_on_rate_limit_errors(self):
        """Unified CacheCurrentPriceManager must not pile duplicate hung requests into the pool and must back off on 429 errors."""
        import threading

        release_hang = threading.Event()
        calls_by_sym = {"HYPEUSD": 0, "SPCXxUSD": 0}

        class _MixedApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                calls_by_sym[symbol] = calls_by_sym.get(symbol, 0) + 1
                if symbol == "HYPEUSD":
                    release_hang.wait(timeout=5.0)
                    return 62.5
                return 129.5

        cpm = cm.CacheCurrentPriceManager(
            sync_ts=0.05,
            symbols=["HYPEUSD", "SPCXxUSD"],
            filename=self._empty_cp_file("cpm_mixed_dedup.json"),
            market_api=_MixedApi(),
            provider_names={"HYPEUSD": "Kraken", "SPCXxUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm._is_live_daemon_poll_mode = lambda: True
        cpm._is_live_network_mode = lambda: True
        try:
            for _ in range(3):
                cpm._last_http_poll_ts_per_symbol.pop("SPCXxUSD", None)
                self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertEqual(calls_by_sym["HYPEUSD"], 1)
            self.assertGreaterEqual(calls_by_sym["SPCXxUSD"], 3)
            self.assertAlmostEqual(cpm.cache["SPCXxUSD"][-1][1], 129.5)
            self.assertNotIn("HYPEUSD", cpm.cache)
            release_hang.set()
            cpm._in_flight_http_futs["HYPEUSD"].future.result(timeout=2.0)
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertAlmostEqual(cpm.cache["HYPEUSD"][-1][1], 62.5)
        finally:
            release_hang.set()
            cpm.shutdown(timeout=2.0)

        # 429 rate-limit exception triggers monotonic exponential backoff instead of hammering every cycle.
        rl_calls = [0]

        class _RateLimitApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                rl_calls[0] += 1
                raise RuntimeError("HTTP 429 rate limit exceeded")

        cpm_rl = cm.CacheCurrentPriceManager(
            sync_ts=0.04,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cpm_429_rl.json"),
            market_api=_RateLimitApi(),
            provider_names={"HYPEUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm_rl._is_live_daemon_poll_mode = lambda: True
        cpm_rl._is_live_network_mode = lambda: True
        try:
            self.assertTrue(cpm_rl.query_remote_and_update_cache())
            if "HYPEUSD" in cpm_rl._in_flight_http_futs:
                cpm_rl._in_flight_http_futs["HYPEUSD"].future.result(timeout=2.0)
                self.assertTrue(cpm_rl.query_remote_and_update_cache())
            self.assertGreater(cpm_rl._err_backoff_until_mono.get("HYPEUSD", 0.0), time.monotonic() + 10.0)
            self.assertFalse(cpm_rl._should_poll())
            self.assertTrue(cpm_rl.query_remote_and_update_cache())
            self.assertEqual(rl_calls[0], 1)
        finally:
            cpm_rl.shutdown(timeout=2.0)

    def test_cpm_fallback_nonblocking_dedup_ws_preemption_and_hard_timeout_reset(self):
        """CacheCurrentPriceManager live fallback must never block on slow HTTP, never apply false backoff while in-flight, let WS preempt stale REST, and reset wedged slots after hard timeout."""
        import threading

        release_hang = threading.Event()
        btc_started = threading.Event()
        calls_by_sym = {"BTCUSDC": 0, "ETHUSDC": 0}
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=1.0,
            symbols=["BTCUSDC", "ETHUSDC"],
            filename=os.path.join(self.tmp, "cache_current_price_dedup.json"),
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm._is_live_daemon_poll_mode = lambda: True
        cpm._is_live_network_mode = lambda: True

        def _fake_remote(symbol, _start_time):
            calls_by_sym[symbol] = calls_by_sym.get(symbol, 0) + 1
            if symbol == "BTCUSDC":
                btc_started.set()
                release_hang.wait(timeout=5.0)
                return [[int(time.time() * 1000), 60000.0]]
            return [[int(time.time() * 1000), 1700.0]]

        cpm.get_remote_items = _fake_remote
        try:
            # 1. Non-blocking micro-wait (~0.05s): fast ETHUSDC is harvested immediately in the same call,
            #    while slow BTCUSDC stays deduplicated in flight with ZERO premature backoff.
            t0 = time.monotonic()
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertLess(time.monotonic() - t0, 0.25)
            self.assertTrue(btc_started.wait(timeout=2.0))
            self.assertEqual(calls_by_sym["BTCUSDC"], 1)
            self.assertEqual(calls_by_sym["ETHUSDC"], 1)
            self.assertAlmostEqual(cpm.cache["ETHUSDC"][0][1], 1700.0)
            self.assertIn("BTCUSDC", cpm._in_flight_http_futs)
            self.assertNotIn("ETHUSDC", cpm._in_flight_http_futs)
            self.assertNotIn("BTCUSDC", cpm._empty_http_backoff_ts)
            self.assertTrue(cpm._should_poll())

            # Immediate revisit while BTCUSDC is still hanging must NOT submit a second BTCUSDC call.
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertEqual(calls_by_sym["BTCUSDC"], 1)

            # 2. Once BTCUSDC finishes, _next_sync_wait_sec() wakes immediately and the next cycle harvests its price without re-polling.
            release_hang.set()
            cpm._in_flight_http_futs["BTCUSDC"].future.result(timeout=2.0)
            self.assertLessEqual(cpm._next_sync_wait_sec(), 0.03)
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertEqual(calls_by_sym["BTCUSDC"], 1)
            self.assertNotIn("BTCUSDC", cpm._in_flight_http_futs)
            self.assertNotIn("BTCUSDC", cpm._empty_http_backoff_ts)
            self.assertAlmostEqual(cpm.cache["BTCUSDC"][0][1], 60000.0)

            # 3. WS preemption: if a real WS tick arrives while REST is in flight, WS price wins and older REST is discarded.
            release_ws_race = threading.Event()
            cpm._last_http_poll_ts_per_symbol.clear()
            cpm.get_remote_items = lambda sym, _st: (
                release_ws_race.wait(timeout=5.0),
                [[int(time.time() * 1000), 58000.0]],
            )[1] if sym == "BTCUSDC" else [[int(time.time() * 1000), 1705.0]]
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertIn("BTCUSDC", cpm._in_flight_http_futs)
            ws_now_ms = int(time.time() * 1000) + 50
            cpm.on_items_update("BTCUSDC", [[ws_now_ms, 61500.0]])
            release_ws_race.set()
            cpm._in_flight_http_futs["BTCUSDC"].future.result(timeout=2.0)
            self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertAlmostEqual(cpm.cache["BTCUSDC"][0][1], 61500.0)
            self.assertNotIn("BTCUSDC", cpm._in_flight_http_futs)

            # 4. Hard-timeout reset: a truly wedged REST call exceeding _rest_hard_timeout_sec is evicted and retried fresh.
            wedged_ev = threading.Event()
            retry_calls = [0]

            def _wedged_then_ok(sym, _st):
                if sym != "BTCUSDC":
                    return [[int(time.time() * 1000), 1710.0]]
                retry_calls[0] += 1
                if retry_calls[0] == 1:
                    wedged_ev.wait(timeout=5.0)
                    return [[int(time.time() * 1000), 50000.0]]
                return [[int(time.time() * 1000), 62200.0]]

            cpm._ws_last_event_ts_per_symbol.clear()
            cpm._ws_stream_event_ts_per_symbol.clear()
            cpm._last_http_poll_ts_per_symbol.clear()
            cpm._rest_hard_timeout_sec = 0.08
            cpm.get_remote_items = _wedged_then_ok
            try:
                self.assertTrue(cpm.query_remote_and_update_cache())
                self.assertIn("BTCUSDC", cpm._in_flight_http_futs)
                time.sleep(0.11)
                # Exceeded hard timeout -> evicts wedged slot #1 and immediately submits & harvests fresh retry #2.
                self.assertTrue(cpm.query_remote_and_update_cache())
                if "BTCUSDC" in cpm._in_flight_http_futs:
                    cpm._in_flight_http_futs["BTCUSDC"].future.result(timeout=2.0)
                    self.assertTrue(cpm.query_remote_and_update_cache())
                self.assertEqual(retry_calls[0], 2)
                self.assertAlmostEqual(cpm.cache["BTCUSDC"][0][1], 62200.0)
            finally:
                wedged_ev.set()
        finally:
            release_hang.set()
            cpm._shutdown_fallback_pool()

    def test_poller_ws_preemption_and_hard_timeout_reset(self):
        """Unified CacheCurrentPriceManager.start_venue_ws_streams attaches all registered venue streams uniformly, discards WS-preempted REST, and resets wedged slots on hard timeout."""
        import threading
        from unittest.mock import patch

        # 1. Unified start_venue_ws_streams attaches registered streams for configured venues idempotently and stops owned streams on shutdown
        started_calls = []
        stopped_calls = []

        class _FakeWs:
            def __init__(self, label, syms):
                self.label = label
                self.syms = list(syms)
                self.is_running = False
                self.subs = []

            def subscribe(self, sub):
                self.subs.append(sub)

            def start(self, name=None, daemon=True):
                self.is_running = True
                started_calls.append((self.label, tuple(self.syms), name))

            def request_stop(self):
                pass

            def stop(self, timeout=None):
                self.is_running = False
                stopped_calls.append((self.label, timeout))

        fake_specs = (
            ("binance", "BinancePriceWS", lambda syms, cpm=None: _FakeWs("binance", syms)),
            ("hyperliquid", "HyperliquidPriceWS", lambda syms, cpm=None: _FakeWs("hyperliquid", syms)),
            ("kraken", "KrakenPriceWS", lambda syms, cpm=None: _FakeWs("kraken", syms)),
        )
        cpm_ws = cm.CacheCurrentPriceManager(
            sync_ts=0.2,
            symbols=["BTCUSDC", "HYPEUSDC", "HYPEUSD", "NVDA_US_EQ"],
            filename=self._empty_cp_file("cpm_unified_ws.json"),
            provider_names={
                "BTCUSDC": "binance",
                "HYPEUSDC": "hyperliquid",
                "HYPEUSD": "kraken",
                "NVDA_US_EQ": "t212",
            },
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        try:
            with patch.object(VenuePriceStreamBase, "iter_registered_venue_streams", return_value=fake_specs):
                streams = cpm_ws.start_venue_ws_streams(start_ws=True, startup_grace_sec=0.0)
                self.assertEqual(len(streams), 3)
                self.assertEqual(
                    started_calls,
                    [
                        ("binance", ("BTCUSDC",), "BinancePriceWS"),
                        ("hyperliquid", ("HYPEUSDC",), "HyperliquidPriceWS"),
                        ("kraken", ("HYPEUSD",), "KrakenPriceWS"),
                    ],
                )
                # Idempotent second call must not spawn duplicate streams
                self.assertEqual(cpm_ws.start_venue_ws_streams(start_ws=True), [])
        finally:
            cpm_ws.shutdown(timeout=1.0)
        # Non-primary owned streams (hyperliquid, kraken) are stopped cleanly by shutdown()
        self.assertEqual({lbl for lbl, _ in stopped_calls}, {"hyperliquid", "kraken"})

        # 2. Hard-timeout reset on periodic_sync loop for a Kraken symbol: wedged call #1 evicted after hard_timeout_sec, call #2 recovers
        wedged_ev = threading.Event()
        ht_calls = [0]

        class _WedgedOnceApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                ht_calls[0] += 1
                if ht_calls[0] == 1:
                    wedged_ev.wait(timeout=5.0)
                    return 10.0
                return 77.5

        cpm_ht = cm.CacheCurrentPriceManager(
            sync_ts=0.04,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cpm_ht_loop.json"),
            market_api=_WedgedOnceApi(),
            provider_names={"HYPEUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm_ht._is_live_daemon_poll_mode = lambda: True
        cpm_ht._is_live_network_mode = lambda: True
        cpm_ht._rest_hard_timeout_sec = 0.12
        cpm_ht._first_sleep = False
        try:
            cpm_ht.periodic_sync(0.04, save_state=True)
            deadline = time.time() + 2.0
            while time.time() < deadline and "HYPEUSD" not in cpm_ht.cache:
                time.sleep(0.03)
            self.assertGreaterEqual(ht_calls[0], 2)
            self.assertAlmostEqual(cpm_ht.cache["HYPEUSD"][-1][1], 77.5)
        finally:
            wedged_ev.set()
            cpm_ht.shutdown(timeout=2.0)

    def test_poller_harvests_late_valid_price_and_late_429_backoff(self):
        """Slow responses exceeding the 50ms micro-wait must wake _sync_wake_event, apply valid prices, and record 429 backoff on late exceptions."""
        slow_calls = [0]

        class _SlowOkApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                slow_calls[0] += 1
                time.sleep(0.11)
                return 88.25

        cpm_ok = cm.CacheCurrentPriceManager(
            sync_ts=0.25,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cpm_slow_ok.json"),
            market_api=_SlowOkApi(),
            provider_names={"HYPEUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm_ok._is_live_daemon_poll_mode = lambda: True
        cpm_ok._is_live_network_mode = lambda: True
        cpm_ok._first_sleep = False
        try:
            cpm_ok.periodic_sync(0.25, save_state=True)
            deadline = time.time() + 2.0
            while time.time() < deadline and "HYPEUSD" not in cpm_ok.cache:
                time.sleep(0.02)
            self.assertIn("HYPEUSD", cpm_ok.cache)
            self.assertAlmostEqual(cpm_ok.cache["HYPEUSD"][-1][1], 88.25)
        finally:
            cpm_ok.shutdown(timeout=2.0)

        late_rl_calls = [0]

        class _SlowRateLimitApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                late_rl_calls[0] += 1
                time.sleep(0.11)
                raise RuntimeError("HTTP 429 too many requests")

        cpm_rl = cm.CacheCurrentPriceManager(
            sync_ts=0.04,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cpm_slow_rl.json"),
            market_api=_SlowRateLimitApi(),
            provider_names={"HYPEUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm_rl._is_live_daemon_poll_mode = lambda: True
        cpm_rl._is_live_network_mode = lambda: True
        cpm_rl._first_sleep = False
        try:
            cpm_rl.periodic_sync(0.04, save_state=True)
            time.sleep(0.35)
            self.assertEqual(late_rl_calls[0], 1)
        finally:
            cpm_rl.shutdown(timeout=2.0)

    def test_pass12_poller_late_empty_response_triggers_empty_backoff(self):
        """Pass #12 invariant & cross-check: late-completing empty/None equity fetches must record monotonic empty backoff."""
        late_empty_calls = [0]

        class _SlowEmptyApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                late_empty_calls[0] += 1
                time.sleep(0.09)
                return None

        cpm_empty = cm.CacheCurrentPriceManager(
            sync_ts=0.04,
            symbols=["NVDA_US_EQ"],
            filename=self._empty_cp_file("cpm_slow_empty.json"),
            market_api=_SlowEmptyApi(),
            provider_names={"NVDA_US_EQ": "t212"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm_empty._is_live_daemon_poll_mode = lambda: True
        cpm_empty._is_live_network_mode = lambda: True
        cpm_empty._enforce_rth_gate = True
        cpm_empty._first_sleep = False
        try:
            cpm_empty.periodic_sync(0.04, save_state=True)
            time.sleep(0.35)
            # After the first slow None finishes late, empty backoff prevents re-polling every cycle.
            self.assertEqual(late_empty_calls[0], 1)
            self.assertIn("NVDA_US_EQ", cpm_empty._empty_backoff_mono)
        finally:
            cpm_empty.shutdown(timeout=2.0)

    def test_pass34_future_ws_ts_does_not_preempt_rest_or_block_ws_and_shutdown_submit_safe(self):
        """Pass #34 pure invariants + dependent cross-checks:
        1. Corrupt far-future _ws_stream_event_ts_per_symbol must NOT falsely preempt REST in unified CacheCurrentPriceManager.
        2. Corrupt far-future cached entry must NOT cause on_items_update to drop real incoming WS ticks.
        3. NonBlockingRestSlotTracker.submit returns None cleanly when executor is shut down; _next_sync_wait_sec honors hard-timeout remaining.
        """
        # 1. Pure invariant: NonBlockingRestSlotTracker.submit handles shut-down executor safely
        tracker = NonBlockingRestSlotTracker()
        dead_pool = futures.ThreadPoolExecutor(max_workers=1)
        dead_pool.shutdown(wait=True)
        self.assertIsNone(
            tracker.submit("BTCUSDC", dead_pool, lambda: 1.0, now_mono=time.monotonic(), now_wall_ts=time.time())
        )

        # 2. Pure invariants on CacheCurrentPriceManager
        now_s = time.time()
        future_s = now_s + 30 * 86400.0
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=1.0,
            symbols=["BTCUSDC"],
            filename=os.path.join(self.tmp, "cache_current_price_p34.json"),
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm._is_live_daemon_poll_mode = lambda: True
        cpm._is_live_network_mode = lambda: True
        cpm._ws_stream_event_ts_per_symbol["BTCUSDC"] = future_s
        self.assertFalse(cpm._was_ws_updated_since("BTCUSDC", now_s - 2.0, now_wall_ts=now_s))
        cpm._ws_stream_event_ts_per_symbol["BTCUSDC"] = now_s - 0.5
        self.assertTrue(cpm._was_ws_updated_since("BTCUSDC", now_s - 2.0, now_wall_ts=now_s))

        # Corrupt far-future cached entry must be healed by a real incoming WS tick
        cpm.cache["BTCUSDC"] = [[int(future_s * 1000), 999999.0]]
        cpm.on_items_update("BTCUSDC", [[int(now_s * 1000), 61800.0]])
        self.assertAlmostEqual(cpm.cache["BTCUSDC"][0][1], 61800.0)

        # 3. Dependent cross-check: CPM REST fallback recovers even when _ws_stream_event_ts_per_symbol is corrupt far-future
        cpm._ws_stream_event_ts_per_symbol["BTCUSDC"] = future_s
        cpm._ws_last_event_ts_per_symbol.clear()
        cpm._last_http_poll_ts_per_symbol.clear()
        cpm.get_remote_items = lambda sym, _st: [[int(time.time() * 1000), 62100.0]]
        try:
            self.assertFalse(cpm.is_real_ws_healthy("BTCUSDC"))
            self.assertTrue(cpm.query_remote_and_update_cache())
            if "BTCUSDC" in cpm._in_flight_http_futs:
                cpm._in_flight_http_futs["BTCUSDC"].future.result(timeout=2.0)
                self.assertTrue(cpm.query_remote_and_update_cache())
            self.assertAlmostEqual(cpm.cache["BTCUSDC"][0][1], 62100.0)

            # _next_sync_wait_sec respects imminent hard-timeout on an active slot
            import threading
            hold_ev = threading.Event()
            cpm._rest_hard_timeout_sec = 0.05
            cpm._ws_stream_event_ts_per_symbol.clear()
            cpm._ws_last_event_ts_per_symbol.clear()
            cpm._last_http_poll_ts_per_symbol.clear()
            cpm.get_remote_items = lambda sym, _st: (hold_ev.wait(timeout=2.0), [[int(time.time() * 1000), 62200.0]])[1]
            try:
                self.assertTrue(cpm.query_remote_and_update_cache())
                self.assertLessEqual(cpm._next_sync_wait_sec(), 0.08)
            finally:
                hold_ev.set()
        finally:
            cpm._shutdown_fallback_pool()

        # 4. Dependent cross-check: unified CacheCurrentPriceManager ignores corrupt far-future _ws_stream_event_ts_per_symbol for Kraken HYPEUSD
        cpm_kr = cm.CacheCurrentPriceManager(
            sync_ts=0.04,
            symbols=["HYPEUSD"],
            filename=self._empty_cp_file("cpm_p34_kr.json"),
            market_api=_FakeApi(hang=False, price=66.5),
            provider_names={"HYPEUSD": "Kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        cpm_kr._is_live_daemon_poll_mode = lambda: True
        cpm_kr._is_live_network_mode = lambda: True
        cpm_kr._ws_stream_event_ts_per_symbol["HYPEUSD"] = future_s
        try:
            self.assertTrue(cpm_kr.query_remote_and_update_cache())
            if "HYPEUSD" in cpm_kr._in_flight_http_futs:
                cpm_kr._in_flight_http_futs["HYPEUSD"].future.result(timeout=2.0)
                self.assertTrue(cpm_kr.query_remote_and_update_cache())
            self.assertAlmostEqual(cpm_kr.cache["HYPEUSD"][-1][1], 66.5)
        finally:
            cpm_kr.shutdown(timeout=2.0)

    def test_slot_tracker_submit_thread_dedup_hard_timeout_and_late_completion(self):
        """NonBlockingRestSlotTracker.submit_thread deduplicates in-flight work, evicts wedged threads on hard timeout, and tolerates late completion."""
        import threading

        tracker = NonBlockingRestSlotTracker()
        release_wedged = threading.Event()
        started = threading.Event()
        t0 = time.monotonic()

        def _wedged():
            started.set()
            release_wedged.wait(timeout=5.0)
            return "stale"

        try:
            slot1 = tracker.submit_thread(
                "kraken",
                _wedged,
                thread_name="test-slot-thread",
                now_mono=t0,
                now_wall_ts=time.time(),
            )
            self.assertIsNotNone(slot1)
            self.assertTrue(started.wait(timeout=2.0))
            # While in-flight before hard_timeout, duplicate submit is rejected and inspect reports IN_FLIGHT
            self.assertIsNone(
                tracker.submit_thread(
                    "kraken",
                    lambda: "dup",
                    thread_name="test-slot-thread",
                    now_mono=t0 + 0.05,
                    now_wall_ts=time.time(),
                )
            )
            st, val, _ = tracker.inspect_and_harvest("kraken", now_mono=t0 + 0.05, hard_timeout_sec=0.2)
            self.assertEqual(st, "IN_FLIGHT")
            self.assertIsNone(val)

            # Once hard_timeout elapses, wedged slot is evicted and a fresh retry succeeds
            st_ht, _, _ = tracker.inspect_and_harvest("kraken", now_mono=t0 + 0.25, hard_timeout_sec=0.2)
            self.assertEqual(st_ht, "HARD_TIMEOUT")
            done_ev = threading.Event()
            slot2 = tracker.submit_thread(
                "kraken",
                lambda: "fresh",
                thread_name="test-slot-thread",
                now_mono=t0 + 0.25,
                now_wall_ts=time.time(),
                on_done=lambda _f: done_ev.set(),
            )
            self.assertIsNotNone(slot2)
            self.assertTrue(done_ev.wait(timeout=2.0))
            st_ok, val_ok, _ = tracker.inspect_and_harvest("kraken", now_mono=t0 + 0.3, hard_timeout_sec=0.2)
            self.assertEqual((st_ok, val_ok), ("DONE_OK", "fresh"))
        finally:
            release_wedged.set()
            tracker.clear()

    def test_shared_http_rate_limit_backoff_pool_and_fills_bundle_ssot(self):
        """Shared core/http primitives (rate-limit detection, exponential backoff, tracker pool/batch wait)
        and CacheAccountManagerBase multi-venue fills bundle helpers behave deterministically."""
        from unittest.mock import patch

        # 1. is_rate_limit_error + compute_rest_error_backoff_sec
        for msg in (
            RuntimeError("HTTP 429 Too Many Requests"),
            "APIError(code=-1003): Too much request weight used",
            "EGeneral:Temporary lockout",
            "rate limit exceeded",
        ):
            self.assertTrue(is_rate_limit_error(msg), f"expected rate-limit match for {msg!r}")
        for msg in (None, "", RuntimeError("read timed out"), "DNS failure"):
            self.assertFalse(is_rate_limit_error(msg), f"unexpected rate-limit match for {msg!r}")

        self.assertEqual(
            compute_rest_error_backoff_sec("timeout", 1, base_sec=2.0, max_sec=30.0), 2.0
        )
        self.assertEqual(
            compute_rest_error_backoff_sec("timeout", 3, base_sec=2.0, max_sec=30.0), 8.0
        )
        self.assertEqual(
            compute_rest_error_backoff_sec("timeout", 10, base_sec=2.0, max_sec=30.0), 30.0
        )
        self.assertEqual(
            compute_rest_error_backoff_sec("HTTP 429", 1, base_sec=2.0, max_sec=30.0, rate_limit_base_sec=15.0, rate_limit_max_sec=120.0),
            15.0,
        )
        self.assertEqual(
            compute_rest_error_backoff_sec("HTTP 429", 4, base_sec=2.0, max_sec=30.0, rate_limit_base_sec=15.0, rate_limit_max_sec=120.0),
            120.0,
        )

        # 2. NonBlockingRestSlotTracker.ensure_pool / submit_batch_and_wait / shutdown_pool
        shutdown_daemon_executor(None)
        tracker = NonBlockingRestSlotTracker()
        pool = tracker.ensure_pool(max_workers=2, thread_name_prefix="SharedTestPool")
        self.assertIs(tracker.pool, pool)
        self.assertIs(tracker.ensure_pool(max_workers=2, thread_name_prefix="SharedTestPool"), pool)
        try:
            done_keys = tracker.submit_batch_and_wait(
                [("BTCUSDC", lambda: 111.0, "m1"), ("ETHUSDC", lambda: 222.0, "m2")],
                pool=pool,
                now_mono=time.monotonic(),
                now_wall_ts=time.time(),
                micro_wait_sec=1.0,
            )
            self.assertEqual(set(done_keys), {"BTCUSDC", "ETHUSDC"})
            st_btc, val_btc, slot_btc = tracker.inspect_and_harvest("BTCUSDC", now_mono=time.monotonic(), hard_timeout_sec=5.0)
            self.assertEqual((st_btc, val_btc, slot_btc.meta), ("DONE_OK", 111.0, "m1"))
        finally:
            tracker.shutdown_pool()
            self.assertIsNone(tracker.pool)
            self.assertFalse(tracker.has_active())

        # 3. CacheAccountManagerBase.publish_venue_fills_bundle & touch_venue_fills_bundle
        tr_path = os.path.join(self.tmp, "cache_trade_kraken.json")
        fo_path = os.path.join(self.tmp, "cache_order_filled_kraken.json")
        t0_ms = 1_795_770_000_000
        fill_1 = {
            "id": "F1",
            "orderId": "ORD-1",
            "symbol": "HYPEUSD",
            "side": "BUY",
            "price": 65.0,
            "qty": 2.0,
            "time": t0_ms - 1000,
            "isBuyer": True,
        }
        fill_2 = {
            "id": "F2",
            "orderId": "ORD-1",
            "symbol": "HYPEUSD",
            "side": "BUY",
            "price": 66.0,
            "qty": 3.0,
            "time": t0_ms - 500,
            "isBuyer": True,
        }
        health_calls = []
        refresh_calls = []
        with (
            patch.object(
                cacheaccount,
                "_record_venue_health_sync",
                side_effect=lambda v, s, *, now_ms=None: health_calls.append((v, s, now_ms)) or True,
            ),
            patch.object(
                cacheaccount,
                "request_venue_account_refresh",
                side_effect=lambda v, *, include_open_orders=False, blocking=False: refresh_calls.append((v, include_open_orders)) or True,
            ),
        ):
            sig_holder = [None]
            merged_tr, merged_fo, tr_miss, fo_miss = cacheaccount.CacheAccountManagerBase.publish_venue_fills_bundle(
                "kraken",
                tr_path,
                {"HYPEUSD": [fill_1]},
                now_ms=t0_ms,
                touch_symbols=("HYPEUSD",),
                touch_create_missing=True,
                sanitize_venue="kraken",
                indent=1,
                force_save=True,
                refresh_mode="on_change",
                prev_sig_holder=sig_holder,
            )
            self.assertEqual((tr_miss, fo_miss), ([], []))
            self.assertEqual(len(merged_tr["items"]["HYPEUSD"]), 1)
            self.assertEqual(len(merged_fo["items"]["HYPEUSD"]), 1)
            self.assertEqual(refresh_calls, [])
            self.assertIn(("kraken", "trades", t0_ms), health_calls)
            self.assertIn(("kraken", "filled_orders", t0_ms), health_calls)

            # Same fills repeated -> no refresh; adding fill_2 partial upgrade -> triggers refresh
            cacheaccount.CacheAccountManagerBase.publish_venue_fills_bundle(
                "kraken",
                tr_path,
                {"HYPEUSD": [fill_1]},
                now_ms=t0_ms + 5_000,
                touch_symbols=("HYPEUSD",),
                touch_create_missing=True,
                sanitize_venue="kraken",
                indent=1,
                refresh_mode="on_change",
                prev_sig_holder=sig_holder,
            )
            self.assertEqual(refresh_calls, [])

            cacheaccount.CacheAccountManagerBase.publish_venue_fills_bundle(
                "kraken",
                tr_path,
                {"HYPEUSD": [fill_1, fill_2]},
                now_ms=t0_ms + 10_000,
                touch_symbols=("HYPEUSD",),
                touch_create_missing=True,
                sanitize_venue="kraken",
                indent=1,
                refresh_mode="on_change",
                prev_sig_holder=sig_holder,
            )
            self.assertEqual(refresh_calls, [("kraken", True)])

            # Keepalive dual-touch updates both trade and filled_orders fetchtime & health
            health_calls.clear()
            tr_t, fo_t, tr_m, fo_m = cacheaccount.CacheAccountManagerBase.touch_venue_fills_bundle(
                "kraken",
                tr_path,
                now_ms=t0_ms + 25_000,
                symbols=("HYPEUSD",),
                create_missing=False,
                indent=1,
            )
            self.assertEqual((tr_t, fo_t, tr_m, fo_m), (True, True, [], []))
            self.assertEqual(
                health_calls,
                [("kraken", "trades", t0_ms + 25_000), ("kraken", "filled_orders", t0_ms + 25_000)],
            )
            self.assertTrue(os.path.exists(fo_path))

    def test_pass37_unified_ws_stream_order_isolation_cleanup_and_slot_thread_failure(self):
        """Pass #37 pure invariants + dependent cross-checks:
        1. NonBlockingRestSlotTracker.submit_thread cleans up slot and does not fire on_done when thread start raises.
        2. CacheCurrentPriceManager.start_venue_ws_streams starts Binance primary WS even when non-primary venues
           are registered before binance (the real runtime import order), isolates failing sibling venues with
           unsubscribe rollback, and detaches stopped owned streams cleanly on shutdown.
        """
        import threading
        from unittest.mock import patch

        # 1. Pure invariant: submit_thread RuntimeError/OSError cleans slot without spurious on_done callback
        tracker = NonBlockingRestSlotTracker()
        cb_fired = []

        class _FailingThread:
            def __init__(self, **_kw):
                pass

            def start(self):
                raise RuntimeError("can't start new thread")

        slot_fail = tracker.submit_thread(
            "hl",
            lambda: "never",
            thread_name="fail-thread",
            now_mono=time.monotonic(),
            now_wall_ts=time.time(),
            thread_cls=_FailingThread,
            on_done=lambda f: cb_fired.append(f),
        )
        self.assertIsNone(slot_fail)
        self.assertEqual(cb_fired, [])
        self.assertFalse(tracker.has_active())

        done_ev = threading.Event()
        slot_ok = tracker.submit_thread(
            "hl",
            lambda: "ok",
            thread_name="ok-thread",
            now_mono=time.monotonic(),
            now_wall_ts=time.time(),
            on_done=lambda f: (cb_fired.append(f), done_ev.set()),
        )
        self.assertIsNotNone(slot_ok)
        self.assertTrue(done_ev.wait(timeout=2.0))
        self.assertEqual(len(cb_fired), 1)
        st_ok, val_ok, _ = tracker.inspect_and_harvest("hl", now_mono=time.monotonic(), hard_timeout_sec=5.0)
        self.assertEqual((st_ok, val_ok), ("DONE_OK", "ok"))

        # 2. Pure invariant & cross-check: non-primary venues registered BEFORE binance must NOT suppress BinancePriceWS
        started_calls = []
        stopped_calls = []
        unsub_calls = []

        class _FakeWs:
            def __init__(self, label, syms, *, fail_start=False):
                self.label = label
                self.syms = list(syms)
                self.fail_start = fail_start
                self.is_running = False
                self.subs = []

            def subscribe(self, sub):
                self.subs.append(sub)

            def unsubscribe(self, sub):
                unsub_calls.append(self.label)
                if sub in self.subs:
                    self.subs.remove(sub)

            def start(self, name=None, daemon=True):
                if self.fail_start:
                    raise RuntimeError(f"{self.label} start boom")
                self.is_running = True
                started_calls.append((self.label, tuple(self.syms), name))

            def request_stop(self):
                pass

            def stop(self, timeout=None):
                self.is_running = False
                stopped_calls.append(self.label)

        runtime_order_specs = (
            ("hyperliquid", "HyperliquidPriceWS", lambda syms, cpm=None: _FakeWs("hyperliquid", syms)),
            ("kraken", "KrakenPriceWS", lambda syms, cpm=None: _FakeWs("kraken", syms)),
            ("binance", "BinancePriceWS", lambda syms, cpm=None: _FakeWs("binance", syms)),
        )
        cpm_order = cm.CacheCurrentPriceManager(
            sync_ts=0.2,
            symbols=["BTCUSDC", "HYPEUSDC", "HYPEUSD"],
            filename=self._empty_cp_file("cpm_p37_order.json"),
            provider_names={"BTCUSDC": "binance", "HYPEUSDC": "hyperliquid", "HYPEUSD": "kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        try:
            with patch.object(VenuePriceStreamBase, "iter_registered_venue_streams", return_value=runtime_order_specs):
                streams = cpm_order.start_venue_ws_streams(start_ws=True, startup_grace_sec=0.0)
                self.assertEqual(len(streams), 3)
                self.assertEqual([lbl for lbl, _, _ in started_calls], ["hyperliquid", "kraken", "binance"])
                self.assertEqual(getattr(cpm_order._ws_manager, "label", None), "binance")
        finally:
            cpm_order.shutdown(timeout=1.0)
        # Shutdown detaches stopped owned streams from _ws_managers while leaving external binance singleton intact
        self.assertEqual({lbl for lbl in stopped_calls}, {"hyperliquid", "kraken"})
        self.assertEqual([getattr(w, "label", None) for w in cpm_order._ws_managers], ["binance"])

        # 3. Dependent cross-check: failing venue start rolls back attachment and does NOT block sibling venues;
        #    pre-attached external non-owned ws_manager still suppresses duplicate binance WS spawn.
        started_calls.clear()
        unsub_calls.clear()
        ext_binance = _FakeWs("ext_binance", ["BTCUSDC"])
        failing_specs = (
            ("hyperliquid", "HyperliquidPriceWS", lambda syms, cpm=None: _FakeWs("hyperliquid", syms, fail_start=True)),
            ("kraken", "KrakenPriceWS", lambda syms, cpm=None: _FakeWs("kraken", syms)),
            ("binance", "BinancePriceWS", lambda syms, cpm=None: _FakeWs("binance", syms)),
        )
        cpm_iso = cm.CacheCurrentPriceManager(
            sync_ts=0.2,
            symbols=["BTCUSDC", "HYPEUSDC", "HYPEUSD"],
            filename=self._empty_cp_file("cpm_p37_iso.json"),
            ws_manager=ext_binance,
            provider_names={"BTCUSDC": "binance", "HYPEUSDC": "hyperliquid", "HYPEUSD": "kraken"},
            read_only=False,
            data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
        )
        try:
            with patch.object(VenuePriceStreamBase, "iter_registered_venue_streams", return_value=failing_specs):
                streams = cpm_iso.start_venue_ws_streams(start_ws=True, startup_grace_sec=0.0)
                self.assertEqual([getattr(w, "label", None) for w in streams], ["kraken"])
                self.assertEqual([lbl for lbl, _, _ in started_calls], ["kraken"])
                self.assertIn("hyperliquid", unsub_calls)
                self.assertEqual([getattr(w, "label", None) for w in cpm_iso._ws_managers], ["ext_binance", "kraken"])
        finally:
            cpm_iso.shutdown(timeout=1.0)
        self.assertEqual([getattr(w, "label", None) for w in cpm_iso._ws_managers], ["ext_binance"])

    def test_symmetric_explicit_provider_ws_and_fail_closed_invariants(self):
        """Pure invariants + dependent cross-checks for symmetric explicit-provider architecture:
        1. Symbols missing explicit provider_name fail closed under require_providers=True and never
           silently bind to Binance in start_venue_ws_streams().
        2. VenuePriceStreamBase registry specs declaratively declare owned=False + lifecycle hooks for
           external shared WS singletons (binance) vs owned=True for per-manager streams (kraken, hyperliquid).
        3. attach_ws_manager populates _ws_by_venue by venue cleanly, and stop_price_ws_streams dispatches
           through registered hooks without venue-specific special cases.
        """
        from unittest.mock import patch

        # 1. Pure invariant: require_providers=True fails closed when any symbol has no provider
        with self.assertRaisesRegex(ValueError, "missing explicit provider_name"):
            cm.CacheCurrentPriceManager(
                sync_ts=1.0,
                symbols=["BTCUSDC", "ORPHANUSDC"],
                filename=self._empty_cp_file("cpm_req_prov.json"),
                provider_names={"BTCUSDC": "binance"},
                read_only=True,
                require_providers=True,
            )

        # 2. Pure invariant: unmapped symbol must never implicitly launch BinancePriceWS
        spawned_syms: list[tuple[str, tuple[str, ...]]] = []

        class _StubWs:
            def __init__(self, label, syms):
                self.label = label
                self.syms = tuple(syms)
                self.is_running = False
                self._VENUE_NAME = label

            def subscribe(self, _sub):
                pass

            def start(self, name=None, daemon=True):
                self.is_running = True
                spawned_syms.append((self.label, self.syms))

            def stop(self, timeout=None):
                self.is_running = False

        cpm_unmapped = cm.CacheCurrentPriceManager(
            sync_ts=1.0,
            symbols=["UNMAPPED_SYM"],
            filename=self._empty_cp_file("cpm_unmapped.json"),
            provider_names={},
            read_only=False,
        )
        fake_specs = (
            ("binance", "BinancePriceWS", lambda syms, cpm=None: _StubWs("binance", syms)),
            ("kraken", "KrakenPriceWS", lambda syms, cpm=None: _StubWs("kraken", syms)),
        )
        try:
            with patch.object(VenuePriceStreamBase, "iter_registered_venue_streams", return_value=fake_specs):
                started = cpm_unmapped.start_venue_ws_streams(start_ws=True, startup_grace_sec=0.0)
                self.assertEqual(started, [])
                self.assertEqual(spawned_syms, [])
                with self.assertRaisesRegex(ValueError, "missing explicit provider_name"):
                    cpm_unmapped.start_venue_ws_streams(
                        start_ws=True, startup_grace_sec=0.0, require_providers=True
                    )
        finally:
            cpm_unmapped.shutdown(timeout=1.0)

        # 3. Pure invariant: VenuePriceStreamBase declares owned + hooks without is_primary hardcoding
        bn_spec = VenuePriceStreamBase.get_registered_venue_stream_spec("binance")
        kr_spec = VenuePriceStreamBase.get_registered_venue_stream_spec("kraken")
        hl_spec = VenuePriceStreamBase.get_registered_venue_stream_spec("hyperliquid")
        self.assertIsNotNone(bn_spec)
        self.assertFalse(bn_spec.owned)
        self.assertTrue(callable(bn_spec.default_ws_getter))
        self.assertTrue(callable(bn_spec.request_stop_hook))
        self.assertTrue(callable(bn_spec.stop_hook))
        self.assertIsNotNone(kr_spec)
        self.assertTrue(kr_spec.owned)
        self.assertIsNotNone(hl_spec)
        self.assertTrue(hl_spec.owned)

        # 4. Dependent cross-check: _ws_by_venue prevents duplicate spawn for same venue even with multiple streams
        cpm_multi = cm.CacheCurrentPriceManager(
            sync_ts=1.0,
            symbols=["BTCUSDC", "HYPEUSD"],
            filename=self._empty_cp_file("cpm_by_venue.json"),
            provider_names={"BTCUSDC": "binance", "HYPEUSD": "kraken"},
            read_only=False,
        )
        ext_kr = _StubWs("kraken", ["HYPEUSD"])
        cpm_multi.attach_ws_manager(ext_kr)
        self.assertIs(cpm_multi._ws_by_venue.get("kraken"), ext_kr)
        try:
            with patch.object(VenuePriceStreamBase, "iter_registered_venue_streams", return_value=fake_specs):
                started = cpm_multi.start_venue_ws_streams(start_ws=True, startup_grace_sec=0.0)
                self.assertEqual([(s.label, s.syms) for s in started], [("binance", ("BTCUSDC",))])
                self.assertIs(cpm_multi._ws_by_venue.get("kraken"), ext_kr)
                self.assertEqual(cpm_multi._ws_by_venue["binance"].label, "binance")
        finally:
            cpm_multi.shutdown(timeout=1.0)


    def test_account_trade_poll_loop_and_event_refresh_hard_timeout_and_dedup(self):
        """_run_account_trade_poll_loop evicts wedged polls on hard_timeout and stops promptly on shutdown;
        request_venue_account_refresh deduplicates in-flight async refreshes and resets wedged slots on hard_timeout."""
        import threading
        from unittest.mock import patch

        # 1. _run_account_trade_poll_loop: wedged call #1 evicted at hard_timeout_sec, call #2 recovers
        wedged_ev = threading.Event()
        stop_ev = threading.Event()
        calls = [0]
        hb_calls = [0]

        def _sync_once():
            calls[0] += 1
            if calls[0] == 1:
                wedged_ev.wait(timeout=5.0)
                return {"items": {"NVDA_US_EQ": []}}
            stop_ev.set()
            return {"items": {"NVDA_US_EQ": [{"id": "t1"}]}}

        try:
            t0 = time.monotonic()
            cacheaccount._run_account_trade_poll_loop(
                sync_once_fn=_sync_once,
                log_tag="t212_test",
                poll_interval=0.01,
                default_poll_sec=0.01,
                backoff_init=0.02,
                default_backoff_init=0.02,
                backoff_max=0.05,
                default_backoff_max=0.05,
                heartbeat_fn=lambda **_kw: hb_calls.__setitem__(0, hb_calls[0] + 1),
                stop_event=stop_ev,
                hard_timeout_sec=0.10,
            )
            self.assertGreaterEqual(calls[0], 2)
            self.assertGreaterEqual(hb_calls[0], 3)
            self.assertLess(time.monotonic() - t0, 2.0)
        finally:
            wedged_ev.set()

        # 2. _run_account_trade_poll_loop: stop_event interrupts a wedged poll quickly without waiting out 30s hard_timeout
        wedged_ev2 = threading.Event()
        entered2 = threading.Event()
        stop_ev2 = threading.Event()

        def _wedged_forever():
            entered2.set()
            wedged_ev2.wait(timeout=5.0)
            return {"items": {}}

        loop_th = threading.Thread(
            target=cacheaccount._run_account_trade_poll_loop,
            kwargs={
                "sync_once_fn": _wedged_forever,
                "log_tag": "t212_stop_test",
                "poll_interval": 1.0,
                "default_poll_sec": 1.0,
                "backoff_init": 0.1,
                "default_backoff_init": 0.1,
                "backoff_max": 1.0,
                "default_backoff_max": 1.0,
                "stop_event": stop_ev2,
                "hard_timeout_sec": 30.0,
            },
            daemon=True,
        )
        try:
            loop_th.start()
            self.assertTrue(entered2.wait(timeout=1.0))
            t_stop = time.monotonic()
            stop_ev2.set()
            loop_th.join(timeout=1.5)
            self.assertFalse(loop_th.is_alive())
            self.assertLess(time.monotonic() - t_stop, 1.0)
        finally:
            stop_ev2.set()
            wedged_ev2.set()
            loop_th.join(timeout=1.0)

        # 3. request_venue_account_refresh (async): dedup in-flight balance while allowing open_orders escalation,
        #    then reset wedged balance slot on hard_timeout_sec
        bal_entered = threading.Event()
        bal_release = threading.Event()
        bal_done2 = threading.Event()
        oo_done = threading.Event()
        bal_calls = [0]
        oo_calls = [0]

        def _bal_sync(_v):
            bal_calls[0] += 1
            if bal_calls[0] == 1:
                bal_entered.set()
                bal_release.wait(timeout=5.0)
                return True
            bal_done2.set()
            return True

        def _oo_sync(_v):
            oo_calls[0] += 1
            oo_done.set()
            return True

        mono_now = [100.0]
        try:
            with (
                patch.object(cacheaccount.account_cache_health, "is_writer_enabled", return_value=True),
                patch.object(cacheaccount, "_monotonic_time", side_effect=lambda: mono_now[0]),
                patch.object(cacheaccount, "_venue_refresh_last_ts", {}),
                patch.object(cacheaccount, "_venue_balances_bridge", SimpleNamespace(sync_venue_once=_bal_sync)),
                patch.object(cacheaccount, "_venue_open_orders_bridge", SimpleNamespace(sync_venue_once=_oo_sync)),
            ):
                cacheaccount._venue_refresh_tracker.clear()
                self.assertTrue(
                    cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=False, hard_timeout_sec=5.0)
                )
                self.assertTrue(bal_entered.wait(timeout=1.0))

                # Past 1.0s debounce, but balance slot is still IN_FLIGHT (<5.0s) -> balance deduped, open_orders still launches
                mono_now[0] = 102.0
                self.assertFalse(
                    cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=False, hard_timeout_sec=5.0)
                )
                self.assertEqual(bal_calls[0], 1)
                self.assertTrue(
                    cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=True, hard_timeout_sec=5.0)
                )
                self.assertTrue(oo_done.wait(timeout=1.0))
                self.assertEqual((bal_calls[0], oo_calls[0]), (1, 1))

                # Past hard_timeout_sec (5.0s), wedged balance slot is evicted and call #2 succeeds
                mono_now[0] = 106.0
                self.assertTrue(
                    cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=False, hard_timeout_sec=5.0)
                )
                self.assertTrue(bal_done2.wait(timeout=1.0))
                self.assertEqual(bal_calls[0], 2)
        finally:
            bal_release.set()
            cacheaccount._venue_refresh_tracker.clear()



