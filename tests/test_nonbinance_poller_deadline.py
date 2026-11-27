"""
Tests for the bounded wait used by non-Binance price polling.

NonBinanceTrendPoller once froze when get_current_price(HYPEUSD) blocked
during DNS resolution. The fetch now runs in a separate worker, while
future.result(timeout) prevents the poller itself from waiting indefinitely.
The provider client remains responsible for ending the network call.
"""

import cacheManager as cm
from tests.conftest import TempDirTestCase
import cacheprice
import concurrent.futures as futures
import json
import os
import time
from binance_api.bapi_ws import BinancePriceStream, PublicPriceStreamBase
from botcore import clamp_price_to_bbo_mid
from intelligence.sentiment.collectors.portfolio_collector import _load_open_order_timestamps
from intelligence.sentiment.evaluators.price_series_evaluator import recalculate_live_price_geometry
from providers.execution_audit import find_duplicate_ai_open_limit_order
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import KrakenProvider
from providers.order_ohlc_utils import extract_order_fee, validate_ohlc_candle_bar
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
        api = _FakeApi()
        invalid_bindings = (
            None,
            {"HYPEUSD": "Kraken"},
            {"HYPEUSD": "Kraken", "TAOUSD": "Binance"},
        )
        for provider_names in invalid_bindings:
            with self.subTest(provider_names=provider_names):
                cpm = _StubCpm(api)
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
        cpm = _StubCpm(api, pushed=pushed)
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
        api = _FakeApi(hang=False, price=76.5)
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999,
            symbols=["HYPEUSD"],
            filename=os.path.join(self.tmp, "cache_current_price.json"),
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
            cpm.push_price("HYPEUSD", bad_price)
        self.assertEqual(cpm.cache, {})

    def test_hyperliquid_and_kraken_ws_streams_feed_cpm_and_suppress_rest_polling(self):

        api = _FakeApi(hang=False, price=99.9)
        cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999,
            symbols=["HYPEUSDC", "HYPEUSD", "TAOUSD", "ADAUSD"],
            filename=os.path.join(self.tmp, "cache_current_price.json"),
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
        self.assertTrue(cpm.is_real_ws_healthy("HYPEUSDC"))
        self.assertTrue(cpm.is_real_ws_healthy("HYPEUSD"))
        self.assertTrue(cpm.is_real_ws_healthy("TAOUSD"))
        self.assertTrue(cpm.is_real_ws_healthy("ADAUSD"))
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
        prov.invalidate_ssot_fills_cache("HYPEUSDC", cache_file)
        self.assertIsNone(prov.read_ssot_fills_cache("HYPEUSDC", cache_file, max_stale_sec=90.0))
        hl_ws._last_fills_cache_touch_ms = 0
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
        """NonBinanceTrendPoller must not pile duplicate hung requests into the pool and must back off on 429 errors."""
        import threading

        release_hang = threading.Event()
        calls_by_sym = {"HYPEUSD": 0, "SPCXxUSD": 0}
        pushed = []

        class _MixedApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                calls_by_sym[symbol] = calls_by_sym.get(symbol, 0) + 1
                if symbol == "HYPEUSD":
                    release_hang.wait(timeout=5.0)
                    return 62.5
                return 129.5

        cpm = _StubCpm(_MixedApi(), pushed=pushed)
        poller = cm._start_nonbinance_trend_poller(
            cpm,
            ["HYPEUSD", "SPCXxUSD"],
            interval_sec=0.05,
            fetch_deadline_sec=0.08,
            provider_names={"HYPEUSD": "Kraken", "SPCXxUSD": "Kraken"},
            start_ws=False,
        )
        try:
            time.sleep(0.45)
            # HYPEUSD is hung in-flight -> at most 1 concurrent call submitted, while healthy SPCXxUSD keeps polling!
            self.assertEqual(calls_by_sym["HYPEUSD"], 1)
            self.assertGreaterEqual(calls_by_sym["SPCXxUSD"], 3)
            self.assertTrue(any(s == "SPCXxUSD" for s, _ in pushed))
            self.assertFalse(any(s == "HYPEUSD" for s, _ in pushed))
            # Unblocking the hung fetch lets the next cycle poll fresh and recover HYPEUSD.
            release_hang.set()
            deadline = time.time() + 2.0
            while time.time() < deadline and not any(s == "HYPEUSD" for s, _ in pushed):
                time.sleep(0.05)
            self.assertTrue(any(s == "HYPEUSD" for s, _ in pushed))
        finally:
            release_hang.set()
            poller.stop(timeout=2.0)

        # 429 rate-limit exception triggers backoff instead of hammering every interval_sec.
        rl_calls = [0]

        class _RateLimitApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                rl_calls[0] += 1
                raise RuntimeError("HTTP 429 rate limit exceeded")

        cpm_rl = _StubCpm(_RateLimitApi(), pushed=[])
        poller_rl = cm._start_nonbinance_trend_poller(
            cpm_rl,
            ["HYPEUSD"],
            interval_sec=0.04,
            fetch_deadline_sec=0.2,
            provider_names={"HYPEUSD": "Kraken"},
            start_ws=False,
        )
        try:
            time.sleep(0.35)
            self.assertEqual(rl_calls[0], 1)
        finally:
            poller_rl.stop(timeout=2.0)

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
        """NonBinanceTrendPoller must discard stale REST results preempted by WS and reset genuinely wedged REST slots on hard_timeout_sec."""
        import threading

        # 1. WS preemption: while REST is in flight, real WS updates HYPEUSD -> slow REST result must NOT overwrite WS.
        release_rest = threading.Event()
        ws_healthy = [False]
        pushed = []

        class _SlowRestApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                release_rest.wait(timeout=5.0)
                return 40.0

        cpm_ws = _StubCpm(_SlowRestApi(), pushed=pushed)
        cpm_ws.is_real_ws_healthy = lambda _s: ws_healthy[0]
        cpm_ws._ws_stream_event_ts_per_symbol = {}
        poller_ws = cm._start_nonbinance_trend_poller(
            cpm_ws,
            ["HYPEUSD"],
            interval_sec=0.04,
            provider_names={"HYPEUSD": "Kraken"},
            start_ws=False,
        )
        try:
            time.sleep(0.06)
            ws_healthy[0] = True
            cpm_ws._ws_stream_event_ts_per_symbol["HYPEUSD"] = time.time()
            release_rest.set()
            time.sleep(0.15)
            self.assertEqual(pushed, [])
        finally:
            release_rest.set()
            poller_ws.stop(timeout=2.0)

        # 2. Hard-timeout reset: wedged call #1 is evicted after hard_timeout_sec and fresh call #2 recovers HYPEUSD.
        wedged_ev = threading.Event()
        pushed_ht = []
        ht_calls = [0]

        class _WedgedOnceApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                ht_calls[0] += 1
                if ht_calls[0] == 1:
                    wedged_ev.wait(timeout=5.0)
                    return 10.0
                return 77.5

        cpm_ht = _StubCpm(_WedgedOnceApi(), pushed=pushed_ht)
        poller_ht = cm._start_nonbinance_trend_poller(
            cpm_ht,
            ["HYPEUSD"],
            interval_sec=0.04,
            hard_timeout_sec=0.12,
            provider_names={"HYPEUSD": "Kraken"},
            start_ws=False,
        )
        try:
            deadline = time.time() + 2.0
            while time.time() < deadline and not pushed_ht:
                time.sleep(0.03)
            self.assertGreaterEqual(ht_calls[0], 2)
            self.assertTrue(any(s == "HYPEUSD" and p == 77.5 for s, p in pushed_ht))
        finally:
            wedged_ev.set()
            poller_ht.stop(timeout=2.0)

    def test_poller_harvests_late_valid_price_and_late_429_backoff(self):
        """Slow responses exceeding fetch_deadline_sec must still apply valid prices and record 429 backoff on late exceptions."""
        pushed = []
        slow_calls = [0]

        class _SlowOkApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                slow_calls[0] += 1
                time.sleep(0.11)
                return 88.25

        cpm_ok = _StubCpm(_SlowOkApi(), pushed=pushed)
        poller_ok = cm._start_nonbinance_trend_poller(
            cpm_ok,
            ["HYPEUSD"],
            interval_sec=0.04,
            fetch_deadline_sec=0.05,
            provider_names={"HYPEUSD": "Kraken"},
            start_ws=False,
        )
        try:
            deadline = time.time() + 2.0
            while time.time() < deadline and not pushed:
                time.sleep(0.03)
            self.assertTrue(any(s == "HYPEUSD" and p == 88.25 for s, p in pushed))
        finally:
            poller_ok.stop(timeout=2.0)

        late_rl_calls = [0]

        class _SlowRateLimitApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                late_rl_calls[0] += 1
                time.sleep(0.11)
                raise RuntimeError("HTTP 429 too many requests")

        cpm_rl = _StubCpm(_SlowRateLimitApi(), pushed=[])
        poller_rl = cm._start_nonbinance_trend_poller(
            cpm_rl,
            ["HYPEUSD"],
            interval_sec=0.04,
            fetch_deadline_sec=0.05,
            provider_names={"HYPEUSD": "Kraken"},
            start_ws=False,
        )
        try:
            time.sleep(0.45)
            # Late 429 exception after deadline must trigger backoff instead of hammering every cycle.
            self.assertEqual(late_rl_calls[0], 1)
        finally:
            poller_rl.stop(timeout=2.0)

    def test_pass12_poller_late_empty_response_triggers_empty_backoff(self):
        """Pass #12 invariant & cross-check: late-completing empty/None fetches must record empty_backoff_ts."""
        late_empty_calls = [0]

        class _SlowEmptyApi:
            def get_current_price(self, symbol=None, *, provider_name=None, **_kw):
                late_empty_calls[0] += 1
                time.sleep(0.11)
                return None

        cpm_empty = _StubCpm(_SlowEmptyApi(), pushed=[])
        cpm_empty._enforce_rth_gate = True
        poller_empty = cm._start_nonbinance_trend_poller(
            cpm_empty,
            ["HYPEUSD"],
            interval_sec=0.04,
            fetch_deadline_sec=0.05,
            provider_names={"HYPEUSD": "Kraken"},
            start_ws=False,
        )
        try:
            time.sleep(0.45)
            # After the first slow None finishes late, 60s empty_backoff_ts must prevent re-polling every cycle.
            self.assertEqual(late_empty_calls[0], 1)
        finally:
            poller_empty.stop(timeout=2.0)


