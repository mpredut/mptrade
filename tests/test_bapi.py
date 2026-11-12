"""Unified unit tests for Binance API (bapi, bapi_ws, bapi_trades, bapi_allorders, bapi_placeorder)."""

import cacheManager as cm
import instrument_registry as sym_mod
import json
import monitortrades as mt_mod
import order_guard
import pricewindow as pw
import providers.market_api as market_api_mod
import pytest
import sys
import threading
import time
import types
import unittest
from binance_api import (
    bapi,
    bapi_allorders,
    bapi_client,
    bapi_placeorder,
    bapi_trades,
    bapi_ws,
    bapi_ws_bridge as ws_bridge,
    bapi_ws_normalize,
    market_data as b_md,
    trailing_stop,
)
from binance_api.bapi_ws import (
    BinanceAccountStream,
    BinancePriceStream,
    WS_USERDATA_KEEPALIVE_SEC,
    WS_USERDATA_LOSS_TIMEOUT_SEC,
)
from binance_api.bapi_ws_normalize import parse_order_execution_report, parse_trade_execution_report
from binance_api.trailing_stop import TrailingStop
from decimal import Decimal
from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics
from order_lifecycle import extract_order_restore_params
from providers.base import extract_order_price, extract_timestamp_ms, extract_timestamp_s
from providers.market_api import BinanceProvider, MarketApi
from providers.order_ohlc_utils import (
    extract_order_qty,
    extract_order_side,
    extract_remaining_qty,
    normalize_order,
)
from providers.quantity import (
    BinanceFilterError,
    BinanceOrderRules,
    _decimal,
    _parse_bool_flag,
    balance_cap_quantity,
    decimal_places,
    resolve_assets,
)
from providers.replay_provider import ReplayMarketDataProvider
from providers.strategy_executor import ProviderError, SubmissionRefused
from tests.conftest import DummySpotExecutionProvider, MockHttpResponse
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch




SYMBOL = "BTCUSDC"


# --- WebSocket & UserData Stream --------------------------------------------

class BapiWsShutdownHandlingTests(unittest.TestCase):
    def test_shutdown_errors_are_detected(self):
        manager = BinancePriceStream()
        self.assertTrue(
            manager._is_shutdown_error(
                RuntimeError("cannot schedule new futures after shutdown")
            )
        )
        self.assertFalse(manager._is_shutdown_error(RuntimeError("boom")))

    def test_process_message_clamps_stale_last_trade_outside_bid_ask_to_mid(self):
        manager = BinancePriceStream()
        received = []
        manager.subscribe(types.SimpleNamespace(on_items_update=lambda sym, items: received.append((sym, items))))
        # Stale last trade c=233.20 while live orderbook moved to 234.40 / 234.60 -> mid 234.50
        manager._process_message(json.dumps({
            "stream": "taousdc@ticker",
            "data": {"s": "TAOUSDC", "c": "233.20", "b": "234.40", "a": "234.60", "E": 1700000005000},
        }))
        self.assertEqual(manager.get_price("TAOUSDC"), 234.5)
        self.assertEqual(received[-1], ("TAOUSDC", [[1700000005000, 234.5]]))
        # Active last trade c=234.45 inside [234.40, 234.60] -> keeps exact trade price 234.45
        manager._process_message(json.dumps({
            "stream": "taousdc@ticker",
            "data": {"s": "TAOUSDC", "c": "234.45", "b": "234.40", "a": "234.60", "E": 1700000006000},
        }))
        self.assertEqual(manager.get_price("TAOUSDC"), 234.45)


class TestUserDataClassify(unittest.TestCase):
    """_classify decides between ping / command reply / real event."""

    C = staticmethod(BinanceAccountStream._classify)

    def test_control_messages(self):
        for message, expected_kind in (
            ({"id": "ping", "status": 200}, "ping"),
            ({"id": "sub", "status": 200}, "response"),
        ):
            with self.subTest(kind=expected_kind):
                kind, _ = self.C(message)
                self.assertEqual(kind, expected_kind)
                self.assertNotEqual(kind, "event")

    def test_event_shapes(self):
        event = {"e": "executionReport", "s": "BTCUSDT", "i": 42}
        for label, message in (("wrapped", {"event": event}), ("bare", event)):
            with self.subTest(shape=label):
                kind, payload = self.C(message)
                self.assertEqual(kind, "event")
                self.assertEqual(payload, event)


class TestWSBaseAndUserData(unittest.TestCase):
    def test_market_manager_no_autostart_on_import(self):
        self.assertFalse(bapi_ws.bapi_ws_manager.is_running)

    def test_userdata_callbacks_default_noop(self):
        s = BinanceAccountStream(on_event=lambda payload: None)
        s.on_available(True)
        s.on_healthy()
        s.on_unhealthy()

    def test_userdata_health_marks_call_callbacks(self):
        seen = {}
        s = BinanceAccountStream(
            on_event=lambda p: None,
            on_available=lambda v: seen.__setitem__("avail", v),
            on_healthy=lambda: seen.__setitem__("healthy", True),
            on_unhealthy=lambda: seen.__setitem__("unhealthy", True),
        )
        s._mark_available(True)
        s._mark_event()
        s._mark_unhealthy()
        self.assertEqual(seen.get("avail"), True)
        self.assertTrue(seen.get("healthy"))
        self.assertTrue(seen.get("unhealthy"))


# --- Trades Reader ----------------------------------------------------------

class _SnapshotLock:
    def __init__(self, manager):
        self.manager = manager
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.held = False
        self.manager.cache[SYMBOL].clear()
        return False


class BapiTradesReaderTest(unittest.TestCase):
    def test_bapi_trades_reader_behaviors(self):
        with self.subTest(msg="lazy manager is created without private synchronization"):
            manager = types.SimpleNamespace()
            get_cache_manager = mock.Mock(return_value=manager)
            cache_module = types.SimpleNamespace(get_cache_manager=get_cache_manager)

            with mock.patch.dict(sys.modules, {"cacheManager": cache_module}):
                actual = bapi_trades.init_cache_trade_manager()

            self.assertIs(actual, manager)
            get_cache_manager.assert_called_once_with("BinanceTrade")

        with self.subTest(msg="recent trade reader copies one locked generation"):
            trade = {
                "symbol": SYMBOL,
                "id": 11,
                "orderId": 22,
                "price": "100.5",
                "qty": "0.01",
                "time": 2_000_000,
                "isBuyer": True,
            }
            manager = types.SimpleNamespace(cache={SYMBOL: [trade]})
            manager.lock = _SnapshotLock(manager)

            with (
                mock.patch.object(bapi_trades, "init_cache_trade_manager", return_value=manager),
                mock.patch.object(bapi_trades.apiorders.sym, "validate_ordertype"),
                mock.patch.object(bapi_trades.apiorders.sym, "validate_symbols"),
                mock.patch.object(bapi_trades.time, "time", return_value=2_001.0),
            ):
                result = bapi_trades.get_trade_orders("BUY", SYMBOL, 10)

            self.assertFalse(manager.lock.held)
            self.assertEqual(manager.cache[SYMBOL], [])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 11)
            self.assertEqual(result[0]["price"], 100.5)


# --- REST Price Cache -------------------------------------------------------

class BinanceRestPriceCacheTest(unittest.TestCase):
    def setUp(self):
        bapi.cprice.clear()
        bapi.cprice_time.clear()

    def tearDown(self):
        bapi.cprice.clear()
        bapi.cprice_time.clear()

    def test_failed_refresh_does_not_keep_an_old_price_fresh(self):
        ttl = bapi.BINANCE_REST_PRICE_CACHE_TTL_SEC
        client = mock.Mock()
        client.get_symbol_ticker.return_value = {"price": "100.25"}
        with (
            mock.patch.object(bapi, "client", client),
            mock.patch.object(bapi.time, "time", side_effect=[10.0, 10.0]),
        ):
            self.assertEqual(100.25, bapi.get_current_price(SYMBOL))
        self.assertEqual(10.0, bapi.cprice_time[SYMBOL])

        client.get_symbol_ticker.side_effect = ConnectionError("ticker unavailable")
        with (
            mock.patch.object(bapi, "client", client),
            mock.patch.object(bapi.time, "time", side_effect=lambda: 10.0 + ttl + 1),
        ):
            self.assertIsNone(bapi.get_current_price(SYMBOL))
            self.assertIsNone(bapi.get_current_price(SYMBOL))

        self.assertEqual(3, client.get_symbol_ticker.call_count)
        self.assertEqual(10.0, bapi.cprice_time[SYMBOL])
        self.assertEqual(100.25, bapi.cprice[SYMBOL])

    def test_invalid_ticker_is_never_cached(self):
        for invalid in ("0", "-1", "nan", "inf"):
            with self.subTest(price=invalid):
                client = mock.Mock()
                client.get_symbol_ticker.return_value = {"price": invalid}
                with mock.patch.object(bapi, "client", client):
                    self.assertIsNone(bapi.update_price(SYMBOL))
                self.assertNotIn(SYMBOL, bapi.cprice)
                self.assertNotIn(SYMBOL, bapi.cprice_time)

    def test_clock_rollback_forces_a_refresh(self):
        bapi.cprice[SYMBOL] = 100.0
        bapi.cprice_time[SYMBOL] = 20.0
        client = mock.Mock()
        client.get_symbol_ticker.return_value = {"price": "101"}
        with (
            mock.patch.object(bapi, "client", client),
            mock.patch.object(bapi.time, "time", side_effect=[10.0, 10.0]),
        ):
            self.assertEqual(101.0, bapi.get_current_price(SYMBOL))


# --- AllOrders & Trade Pagination -------------------------------------------

class TestPaginateMyTrades(unittest.TestCase):
    def test_single_page_one_call(self):
        client = MagicMock()
        client.get_my_trades.return_value = [{"id": i, "time": 1} for i in range(10)]
        out = bapi_allorders.paginate_my_trades(client, SYMBOL, 0, limit=1000)
        self.assertEqual(len(out), 10)
        self.assertEqual(client.get_my_trades.call_count, 1)
        _, kw = client.get_my_trades.call_args_list[0]
        self.assertEqual(kw.get("startTime"), 0)
        self.assertNotIn("fromId", kw)

    def test_multi_page_uses_fromid(self):
        client = MagicMock()
        page1 = [{"id": i, "time": 1} for i in range(1000)]
        page2 = [{"id": 1000 + i, "time": 1} for i in range(500)]
        client.get_my_trades.side_effect = [page1, page2]
        out = bapi_allorders.paginate_my_trades(client, SYMBOL, 0, limit=1000)
        self.assertEqual(len(out), 1500)
        self.assertEqual(client.get_my_trades.call_count, 2)
        _, kw2 = client.get_my_trades.call_args_list[1]
        self.assertEqual(kw2.get("fromId"), 1000)
        self.assertNotIn("startTime", kw2)

    def test_empty(self):
        client = MagicMock()
        client.get_my_trades.return_value = []
        self.assertEqual(bapi_allorders.paginate_my_trades(client, "X", 0), [])

    def test_exact_multiple_of_limit_stops_on_empty(self):
        client = MagicMock()
        page1 = [{"id": i, "time": 1} for i in range(2)]
        client.get_my_trades.side_effect = [page1, []]
        out = bapi_allorders.paginate_my_trades(client, "X", 0, limit=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(client.get_my_trades.call_count, 2)

    def test_none_response_remains_empty_for_legacy_readers(self):
        client = MagicMock()
        client.get_my_trades.return_value = None
        self.assertEqual(bapi_allorders.paginate_my_trades(client, "X", 0), [])

    def test_none_response_fails_a_strict_reconciliation(self):
        client = MagicMock()
        client.get_my_trades.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Binance returned no trade page"):
            bapi_allorders.paginate_my_trades(client, "X", 0, strict=True)


class TestCachedOrders(unittest.TestCase):
    def test_reader_does_not_start_a_duplicate_polling_loop(self):
        manager = MagicMock(cache={SYMBOL: []})
        with patch("cacheManager.get_cache_manager", return_value=manager) as get_manager:
            self.assertEqual(bapi_allorders.get_trade_orders("BUY", SYMBOL, 60), [])
        get_manager.assert_called_once_with("BinanceOrder")

    def test_reader_copies_mutable_order_rows_under_manager_lock(self):
        now_ms = int(time.time() * 1000)
        manager = MagicMock(cache={
            SYMBOL: [{"orderId": 7, "price": 100.0, "quantity": 0.5, "timestamp": now_ms, "side": "BUY"}],
        })

        class MutatingLock:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                manager.cache[SYMBOL][0]["price"] = 999.0

        manager.lock = MutatingLock()
        with patch("cacheManager.get_cache_manager", return_value=manager):
            result = bapi_allorders.get_trade_orders("BUY", SYMBOL, 60)

        self.assertEqual(result[0]["price"], 100.0)
        self.assertEqual(manager.cache[SYMBOL][0]["price"], 999.0)


# --- PlaceOrder Helpers & WS Price Validation -------------------------------

class SafeOrderMarketRegimeBoundaryTest(unittest.TestCase):
    def test_lowercase_order_type_and_aggregate_trades_and_ws_price_validation(self):
        now_ms = int(time.time() * 1000)
        order_mgr = MagicMock(cache={
            SYMBOL: [{"orderId": 1, "price": 100.0, "quantity": 1.0, "timestamp": now_ms, "side": "BUY"}],
        })
        order_mgr.lock = threading.Lock()
        with patch("cacheManager.get_cache_manager", return_value=order_mgr):
            res_ao = bapi_allorders.get_trade_orders("buy", SYMBOL, 60)
        self.assertEqual(len(res_ao), 1)
        self.assertEqual(res_ao[0]["side"], "BUY")

        trade_mgr = MagicMock(cache={
            SYMBOL: [{
                "symbol": SYMBOL, "id": 10, "orderId": 1, "price": "100.0",
                "qty": "1.0", "time": now_ms, "isBuyer": True,
            }],
        })
        trade_mgr.lock = threading.Lock()
        with patch.object(bapi_trades, "init_cache_trade_manager", return_value=trade_mgr):
            res_bt = bapi_trades.get_trade_orders("buy", SYMBOL, 60)
        self.assertEqual(len(res_bt), 1)
        self.assertTrue(res_bt[0]["isBuyer"])

        stream = bapi_ws.BinancePriceStream(symbols=[SYMBOL])
        stream._process_message('{"data": {"s": "BTCUSDC", "c": "0", "E": 1000}}')
        self.assertIsNone(stream.get_price(SYMBOL))
        stream._process_message('{"data": {"s": "BTCUSDC", "c": "123.45", "E": 1000}}')
        self.assertEqual(stream.get_price(SYMBOL), 123.45)

    def test_short_trend_dynamic_window_uses_now_val_and_snapshot_fresh_guards_none_ts(self):
        mgr = cm.CacheShortTrendManager([SYMBOL], filename="test_short_trend_tmp.json")
        hist_now = 1_700_000_000.0
        fake_c24 = types.SimpleNamespace(
            lock=threading.Lock(),
            cache={
                SYMBOL: [
                    [int((hist_now - 120) * 1000), 100.0],
                    [int((hist_now - 60) * 1000), 101.0],
                    [int((hist_now - 10) * 1000), 102.0],
                    [int(hist_now * 1000), 103.0],
                ]
            },
        )
        mgr._cache24_managers = {SYMBOL: fake_c24}
        dyn = mgr.get_short_trend_for_window(SYMBOL, 180.0, now=hist_now)
        self.assertIsNotNone(dyn)
        self.assertEqual(dyn["current_price"], 103.0)
        self.assertEqual(dyn["ts"], hist_now)

        mgr._mem[SYMBOL] = {"symbol": SYMBOL, "ts_fast": time.time(), "ts": None}
        self.assertIsNone(mgr.get_fresh_snapshot(SYMBOL))

    def test_pricewindow_edge_cases(self):
        w = pw.PriceWindow("BTCUSDC", 5, initial_prices=[100.0, 101.0, 102.0])
        self.assertEqual(len(w.prices), 3)
        w.sample_rate_sec = 0
        self.assertEqual(w.recent_n, 2)
        trend, _, _, _ = pw.instant_trend_from_prices([100.0, 101.0, 102.0], None)
        self.assertEqual(trend, 1)

    def test_canonical_order_normalization_and_bapi_trade_unification(self):
        # 1. Canonical timestamp normalization (seconds vs ms, multiple field names)
        self.assertEqual(extract_timestamp_ms({"timestamp": 1_700_000_000}), 1_700_000_000_000)
        self.assertEqual(extract_timestamp_ms({"time": 1_700_000_000_500}), 1_700_000_000_500)
        self.assertEqual(extract_timestamp_ms({"updateTime": 1_700_000_001}), 1_700_000_001_000)
        self.assertEqual(extract_timestamp_ms({"opentm": 1_700_000_002.25}), 1_700_000_002_250)

        # 2. Canonical side, price, qty extraction and normalize_order contract
        norm = normalize_order({"isBuyer": False, "px": "250.5", "sz": "2.0", "updateTime": 1_700_000_000})
        self.assertEqual(norm, {
            "side": "SELL",
            "price": 250.5,
            "qty": 2.0,
            "timestamp": 1_700_000_000_000,
        })
        self.assertEqual(extract_order_side({"side": "B"}), "BUY")
        self.assertEqual(extract_order_side({"side": "A"}), "SELL")
        self.assertEqual(extract_order_price({"limitPx": "99.5"}), 99.5)
        self.assertEqual(extract_order_qty({"vol": "3.25"}), 3.25)

        # 5. bapi_allorders.get_trade_orders and bapi_trades.get_trade_orders emit both timestamp/time and qty/quantity
        now_ms = int(time.time() * 1000)
        fake_order_mgr = types.SimpleNamespace(
            lock=threading.Lock(),
            cache={SYMBOL: [
                {"orderId": 77, "symbol": SYMBOL, "side": "BUY", "price": "105.0", "executedQty": "1.5", "updateTime": now_ms - 5000}
            ]},
        )
        with mock.patch.object(cm, "get_cache_manager", return_value=fake_order_mgr):
            all_orders_res = bapi_allorders.get_trade_orders("BUY", SYMBOL, 60)
        self.assertEqual(len(all_orders_res), 1)
        self.assertEqual(all_orders_res[0]["time"], now_ms - 5000)
        self.assertEqual(all_orders_res[0]["timestamp"], now_ms - 5000)
        self.assertEqual(all_orders_res[0]["qty"], 1.5)
        self.assertEqual(all_orders_res[0]["quantity"], 1.5)
        self.assertTrue(all_orders_res[0]["isBuyer"])

        fake_trade_mgr = types.SimpleNamespace(
            lock=threading.Lock(),
            cache={SYMBOL: [
                {"orderId": 88, "symbol": SYMBOL, "side": "SELL", "price": "110.0", "quantity": "2.5", "timestamp": now_ms - 5000}
            ]},
        )
        with mock.patch.object(bapi_trades, "init_cache_trade_manager", return_value=fake_trade_mgr):
            tr_orders_res = bapi_trades.get_trade_orders("SELL", SYMBOL, 60)
        self.assertEqual(len(tr_orders_res), 1)
        self.assertEqual(tr_orders_res[0]["time"], now_ms - 5000)
        self.assertEqual(tr_orders_res[0]["timestamp"], now_ms - 5000)
        self.assertEqual(tr_orders_res[0]["qty"], 2.5)
        self.assertEqual(tr_orders_res[0]["quantity"], 2.5)
        self.assertFalse(tr_orders_res[0]["isBuyer"])

        # 6. bapi.cancel_expired_orders handles millisecond timestamps properly
        expired_ms = int((time.time() - 7200) * 1000)
        fresh_ms = int((time.time() - 60) * 1000)
        open_orders_map = {
            1: {"orderId": 1, "symbol": SYMBOL, "timestamp": expired_ms},
            2: {"orderId": 2, "symbol": SYMBOL, "time": fresh_ms},
        }
        with (
            mock.patch.object(bapi, "get_open_orders", return_value=open_orders_map),
            mock.patch.object(bapi, "cancel_order", return_value=True) as mock_cancel,
        ):
            bapi.cancel_expired_orders("BUY", SYMBOL, 3600)
            mock_cancel.assert_called_once_with(SYMBOL, 1)

        # 7. extract_timestamp_s, extract_remaining_qty, and order_rollback.extract_order_restore_params
        self.assertEqual(extract_timestamp_s({"timestamp": 1_700_000_000_000}), 1_700_000_000.0)
        self.assertEqual(extract_timestamp_s({"updateTime": 1_700_000_000}), 1_700_000_000.0)
        self.assertEqual(extract_remaining_qty({"origQty": 2.5, "executedQty": 0.5, "price": 60.25}), 2.0)
        self.assertEqual(extract_remaining_qty({"remainingQty": 1.25, "origQty": 2.5, "executedQty": 0.5}), 1.25)

        restore_params = extract_order_restore_params({
            "orderId": "OPEN-1",
            "side": "SELL",
            "price": 60.25,
            "origQty": 2.5,
            "executedQty": 0.5,
            "status": "OPEN",
        })
        self.assertEqual(restore_params["qty"], 2.0)
        self.assertEqual(restore_params["price"], 60.25)
        self.assertEqual(restore_params["side"], "SELL")

        # 8. order_guard.daily_limit_guard enforces symmetric 24h daily cap
        now_s = int(time.time())
        fake_provider = types.SimpleNamespace(
            name="binance",
            get_orders=lambda sym, side, window: [
                {"price": 100.0, "qty": 1.0, "time": now_s - 2},
            ],
        )
        ok, reason = order_guard.daily_limit_guard(
            fake_provider, SYMBOL, "BUY", max_daily_trades=1,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_limit")

        # 9. monitortrades.get_relevant_trade normalizes epoch seconds and alternate price/timestamp keys
        rel_px, rel_ts, _ = mt_mod.get_relevant_trade(
            [{"px": "105.5", "opentm": 1_700_000_000}], "BUY"
        )
        self.assertEqual(rel_px, 105.5)
        self.assertEqual(rel_ts, 1_700_000_000.0)

        # 10. extract_order_price resolves market order VWAP from quote cost or fills
        self.assertAlmostEqual(
            extract_order_price({"price": "0.00000000", "cummulativeQuoteQty": "650.0", "executedQty": "0.01"}),
            65000.0,
        )
        self.assertAlmostEqual(
            extract_order_price({"price": "0", "cost": "150.0", "vol_exec": "2.5"}),
            60.0,
        )
        self.assertAlmostEqual(
            extract_order_price({"filledValue": -300.0, "filledQuantity": -2.0}),
            150.0,
        )
        self.assertAlmostEqual(
            extract_order_price({
                "price": "0.00000000",
                "fills": [
                    {"price": "100.0", "qty": "1.0"},
                    {"price": "130.0", "qty": "2.0"},
                ],
            }),
            120.0,
        )

        # 11. binance_api.market_data.fetch_ohlc_candles paginates across 1000-candle pages even when limit_per_batch > 1000

        page1 = [[1_700_000_000_000 + i * 60_000, "100", "101", "99", "100", "10", 1_700_000_059_999 + i * 60_000] for i in range(1000)]
        page2 = [[1_700_000_000_000 + (1000 + i) * 60_000, "100", "101", "99", "100", "10", 1_700_000_059_999 + (1000 + i) * 60_000] for i in range(200)]
        calls = []

        def _fake_urlopen(req, timeout=20):
            calls.append(req.full_url)
            return MockHttpResponse(page1 if len(calls) == 1 else page2)

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            candles = b_md.fetch_ohlc_candles(
                "BTCUSDC",
                1,
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
                limit_per_batch=1500,
                drop_incomplete=False,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(candles), 1200)

        # 13. bapi.update_price and get_free_balance reject boolean/NaN/negative payloads
        fake_client = MagicMock()
        fake_client.get_symbol_ticker.return_value = {"price": True}
        fake_client.get_asset_balance.return_value = {"free": True}
        with mock.patch.object(bapi, "client", fake_client):
            self.assertIsNone(bapi.update_price(SYMBOL))
            self.assertIsNone(bapi.get_free_balance("USDC"))
            fake_client.get_asset_balance.return_value = {"free": "-5.0"}
            self.assertIsNone(bapi.get_free_balance("USDC"))

        # 14. compute_24h_price_metrics normalizes Unix-second timestamps
        sec_base = 1_700_000_000.0
        sec_series = [
            [sec_base - 180 * 60, 100.0],
            [sec_base - 60 * 60, 105.0],
            [sec_base - 30 * 60, 108.0],
            [sec_base - 15 * 60, 109.0],
            [sec_base, 110.0],
        ]
        m24 = compute_24h_price_metrics(sec_series)
        self.assertEqual(m24.get("chg_3h_pct"), 10.0)
        self.assertEqual(m24.get("chg_1h_pct"), 4.76)

        # 15. BybitLiquidationCollector normalizes symbol case and rejects NaN/non-positive notionals
        bybit_col = BybitLiquidationCollector(symbols=["BTCUSDT"])
        bybit_col._ingest(json.dumps({
            "topic": "allLiquidation.BTCUSDT",
            "data": [
                {"s": "btcusdt", "S": "Sell", "v": "2.0", "p": "100.0"},
                {"s": "BTCUSDT", "S": "Sell", "v": "inf", "p": "100.0"},
            ],
        }))
        bybit_col.record_event("btcusdt", "SHORT", float("nan"))
        bybit_col.record_event("btcusdt", "SHORT", -50.0)
        summ = bybit_col.get_summary("BTCUSDT")
        self.assertEqual(summ.short_liq_usd, 200.0)

        # 16. instrument_registry.validate_params rejects booleans, NaN, and inf; bapi.get_account_assets_balances rejects non-finite/negative/bool
        for bad_val in (True, False, float("nan"), float("inf"), float("-inf"), 0, -1):
            with self.assertRaises(ValueError):
                sym_mod.validate_params("BUY", SYMBOL, price=bad_val, qty=1.0)
            with self.assertRaises(ValueError):
                sym_mod.validate_params("BUY", SYMBOL, price=100.0, qty=bad_val)

        fake_client.get_account.return_value = {"balances": [{"asset": "BTC", "free": True, "locked": "0.0"}]}
        with mock.patch.object(bapi, "client", fake_client):
            self.assertEqual(bapi.get_account_assets_balances(), [])
            fake_client.get_account.return_value = {"balances": [{"asset": "BTC", "free": "nan", "locked": "0.0"}]}
            self.assertEqual(bapi.get_account_assets_balances(), [])
            fake_client.get_account.return_value = {"balances": [{"asset": "BTC", "free": "-1.0", "locked": "0.0"}]}
            self.assertEqual(bapi.get_account_assets_balances(), [])

        with mock.patch.object(bapi_placeorder.api, "get_current_price", return_value=float("inf")):
            self.assertEqual(
                bapi_placeorder.adjust_price_and_cancel_opposite("SELL", SYMBOL, 100.0, cancel_opposite=False),
                100.0,
            )
            self.assertEqual(
                bapi_placeorder.order_candidate_price("BUY", 100.0, True),
                100.0,
            )

        # 18. _convert_to_usdc supports inverse USDCUSDT quote and stablecoin peg fallback
        with mock.patch.object(bapi, "_get_symbol_price_safe", side_effect=lambda s: 1.001 if s == "USDCUSDT" else None):
            self.assertAlmostEqual(bapi._convert_to_usdc("USDT", 100.1), 100.0, places=4)
            self.assertEqual(bapi._convert_to_usdc("FDUSD", 50.0), 50.0)
            self.assertEqual(bapi._convert_to_usdc("USD", 25.0), 25.0)
        self.assertEqual(bapi._normalize_side("long"), "BUY")
        self.assertEqual(bapi._normalize_side("short"), "SELL")


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_bapi_allorders_get_trade_orders_bapi_trades_get_trade_orders_and_aggregate_trades_iter28_s2(monkeypatch):
    """[ITER28 #2] bapi_allorders.get_trade_orders, bapi_trades.get_trade_orders, and aggregate_trades."""
    tm = cm.CacheTradeManager.__new__(cm.CacheTradeManager)
    tm.lock = threading.RLock()
    tm.sync_account_reader_if_needed = lambda: None
    tm.cache = {
        "BTCUSDC": [
            {
                "symbol": "BTCUSDC",
                "id": 101,
                "orderId": 201,
                "price": "88500.0",
                "qty": "0.1",
                "time": 1_700_000_000_000,
                "isBuyer": False,
            }
        ]
    }
    assert tm.get_last_opposite_fill_price("BTC/USDC", "BUY") == 88500.0
    assert tm.get_last_opposite_fill_price("btcusdc", "BUY") == 88500.0
    cpm = cm.CacheCurrentPriceManager.__new__(cm.CacheCurrentPriceManager)
    cpm.cls_name = "CacheCurrentPriceManager"
    cpm.lock = threading.RLock()
    cpm.symbols = ["TAOUSD"]
    cpm._provider_names = {"TAOUSD": "kraken"}
    now_ms = int(time.time() * 1000)
    cpm.cache = {"TAOUSD": [[now_ms, 420.5]]}
    assert cpm.provider_name_for("tao/usd") == "kraken"
    obs = cpm.get_cached_price_observation("TAO/USD", 10.0)
    assert obs is not None and obs[1] == 420.5
    assert cpm.get_price_value("tao/usd") == 420.5
    # 2. bapi_allorders.get_trade_orders, bapi_trades.get_trade_orders, and aggregate_trades
    monkeypatch.setattr(bapi_allorders.sym, "validate_symbols", lambda s: None)
    future_ts = now_ms + 3_600_000  # 1 hour in the future (clock skew)
    valid_ts = now_ms - 10_000
    fake_order_mgr = SimpleNamespace(
        lock=threading.RLock(),
        cache={
            "BTCUSDC": [
                {"orderId": 1, "price": 90000.0, "quantity": 0.1, "timestamp": future_ts, "side": "SELL"},
                {"orderId": 2, "price": 0.0, "quantity": 0.1, "timestamp": valid_ts, "side": "SELL"},
                {"orderId": 3, "price": 85000.0, "quantity": 0.2, "timestamp": valid_ts, "side": "SELL"},
            ]
        },
    )
    monkeypatch.setattr(cm, "get_cache_manager", lambda name, **kw: fake_order_mgr)
    orders_out = bapi_allorders.get_trade_orders("SELL", "BTC/USDC", 300)
    assert len(orders_out) == 1 and orders_out[0]["orderId"] == 3

    fake_trade_mgr = SimpleNamespace(
        lock=threading.RLock(),
        cache={
            "BTCUSDC": [
                {"symbol": "BTCUSDC", "id": 1, "orderId": 1, "price": 90000.0, "qty": 0.1, "time": future_ts, "isBuyer": False},
                {"symbol": "BTCUSDC", "id": 2, "orderId": 2, "price": 0.0, "qty": 0.1, "time": valid_ts, "isBuyer": False},
                {"symbol": "BTCUSDC", "id": 3, "orderId": 3, "price": 86000.0, "qty": 0.3, "time": valid_ts, "isBuyer": False},
            ]
        },
    )
    monkeypatch.setattr(bapi_trades, "init_cache_trade_manager", lambda: fake_trade_mgr)
    trades_out = bapi_trades.get_trade_orders("SELL", "BTC/USDC", 300)
    assert len(trades_out) == 1 and trades_out[0]["id"] == 3

def test_bapi_cancel_expired_orders_and_bapi_cancel_recent_orders_with_nan_bool_iter28_s3(monkeypatch):
    """[ITER28 #3] bapi.cancel_expired_orders and bapi.cancel_recent_orders with NaN/bool/non-positive bounds & future order_time."""
    now_ms = int(time.time() * 1000)
    # 3. bapi.cancel_expired_orders and bapi.cancel_recent_orders with NaN/bool/non-positive bounds & future order_time
    canceled_ids = []
    monkeypatch.setattr(bapi.sym, "validate_params", lambda ot, s: None)
    monkeypatch.setattr(
        bapi,
        "get_open_orders",
        lambda ot, s, **kw: {
            501: {"time": now_ms - 5_000, "price": 80000.0, "qty": 0.1},
            502: {"time": now_ms + 60_000, "price": 80000.0, "qty": 0.1},  # future timestamp
        },
    )
    monkeypatch.setattr(bapi, "cancel_order", lambda s, oid: (canceled_ids.append(oid) or True))

    bapi.cancel_expired_orders("BUY", "BTCUSDC", 0)
    bapi.cancel_expired_orders("BUY", "BTCUSDC", -10)
    bapi.cancel_expired_orders("BUY", "BTCUSDC", float("nan"))
    bapi.cancel_expired_orders("BUY", "BTCUSDC", True)
    assert canceled_ids == []

def test_bapi_placeorder_rejects_boolean_qty_price_cancel_opposite_requested_price_iter37_s2():
    """[ITER37 #2] bapi_placeorder rejects boolean qty/price/cancel_opposite_requested_price."""
    # 2. bapi_placeorder rejects boolean qty/price/cancel_opposite_requested_price
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder.apply_weight_limit("BTCUSDC", "BUY", True, 1.0, 1.0)
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder.apply_weight_limit("BTCUSDC", "BUY", 60000.0, True, 1.0)
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder.apply_weight_limit("BTCUSDC", "BUY", 60000.0, 1.0, True)
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder.issue_account_cache_submit_permit("BTCUSDC", "BUY", True, price=60000.0)
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder.issue_account_cache_submit_permit("BTCUSDC", "BUY", 1.0, price=True)
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order("BUY", "BTCUSDC", 1.0, price=60000.0, cancel_opposite_requested_price=True)

def test__require_submit_permit_headroom_refuses_nan_negative_or_boolean_cache_ages_iter38_s1():
    """[ITER38 #1] _require_submit_permit_headroom refuses NaN, negative, or boolean cache ages."""
    # 1. _require_submit_permit_headroom refuses NaN, negative, or boolean cache ages
    for bad_status in (
        SimpleNamespace(order_age_sec=0.5, trade_age_sec=float("nan")),
        SimpleNamespace(order_age_sec=float("nan"), trade_age_sec=0.5),
        SimpleNamespace(order_age_sec=True, trade_age_sec=0.5),
        SimpleNamespace(order_age_sec=-1.0, trade_age_sec=0.5),
    ):
        with pytest.raises(bapi_placeorder.SubmissionRefused):
            bapi_placeorder._require_submit_permit_headroom(bad_status)

def test_cancel_opposite_orders_ignores_zero_missing_price_on_buy_symmetrically_iter38_s2(monkeypatch):
    """[ITER38 #2] cancel_opposite_orders ignores zero/missing price on BUY symmetrically with SELL and rejects bool requested_price."""
    # 2. cancel_opposite_orders ignores zero/missing price on BUY symmetrically with SELL and rejects bool requested_price
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder.cancel_opposite_orders("BUY", "BTCUSDC", True)

    canceled_ids = []
    monkeypatch.setattr(
        bapi_placeorder.api,
        "get_open_orders",
        lambda side, sym, **kw: {
            "oid_zero": {"price": "0.00000000"},
            "oid_adverse": {"price": "59000.0" if side == "SELL" else "61000.0"},
        },
    )
    monkeypatch.setattr(bapi_placeorder.api, "cancel_order", lambda sym, oid: canceled_ids.append(oid) or True)
    bapi_placeorder.cancel_opposite_orders("BUY", "BTCUSDC", 60000.0)
    assert canceled_ids == ["oid_adverse"]
    canceled_ids.clear()
    bapi_placeorder.cancel_opposite_orders("SELL", "BTCUSDC", 60000.0)
    assert canceled_ids == ["oid_adverse"]

def test_binance_filters__decimal_rejects_bool_and__parse_bool_flag_handles_string_iter39_s1():
    """[ITER39 #1] binance_filters._decimal rejects bool and _parse_bool_flag handles string 'false'."""
    # 1. binance_filters._decimal rejects bool and _parse_bool_flag handles string "false"
    with pytest.raises(BinanceFilterError):
        _decimal(True, name="qty")
    assert _parse_bool_flag("false") is False
    assert _parse_bool_flag("true") is True
    rules = BinanceOrderRules.from_symbol_info({
        "baseAsset": "BTC",
        "quoteAsset": "USDC",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "1", "maxPrice": "1000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100"},
            {"filterType": "NOTIONAL", "minNotional": "10", "maxNotional": "100000", "applyMinToMarket": "false"},
        ],
    })
    assert rules.min_notional_market is False

def test_binanceprovider_position_cost_basis_order_status_cancel_order_reject_boolean_inputs_iter39_s3():
    """[ITER39 #3] BinanceProvider.position_cost_basis / order_status / cancel_order reject boolean inputs."""
    # 3. BinanceProvider.position_cost_basis / order_status / cancel_order reject boolean inputs
    bp_prov = BinanceProvider()
    with pytest.raises(ValueError):
        bp_prov.position_cost_basis("BTCUSDC", 1.0, True)
    with pytest.raises(ProviderError):
        bp_prov.order_status("BTCUSDC", True)
    with pytest.raises(ProviderError):
        bp_prov.cancel_order("BTCUSDC", True)

def test_bapi_get_current_price_ignores_nan_bool_cached_values_and_iter41_s2(monkeypatch):
    """[ITER41 #2] bapi.get_current_price ignores NaN/bool cached values and cancel_order rejects bool order_id."""
    # 2. bapi.get_current_price ignores NaN/bool cached values and cancel_order rejects bool order_id
    monkeypatch.setitem(bapi.cprice, "BTCUSDC", float("nan"))
    monkeypatch.setitem(bapi.cprice_time, "BTCUSDC", time.time())
    monkeypatch.setattr(bapi, "update_price", lambda sym: 61234.5)
    assert bapi.get_current_price("BTCUSDC") == pytest.approx(61234.5)
    assert bapi.cancel_order("BTCUSDC", True) is False
    assert bapi._convert_to_usdc("USDC", True) == 0.0

def test_bapi_allorders_get_total_traded_stats_ignores_malformed_nan_bool_trades_and_iter41_s3(monkeypatch):
    """[ITER41 #3] bapi_allorders.get_total_traded_stats ignores malformed/NaN/bool trades and get_filled_orders keeps unkeyed fills separate."""
    # 3. bapi_allorders.get_total_traded_stats ignores malformed/NaN/bool trades and get_filled_orders keeps unkeyed fills separate
    monkeypatch.setattr(
        bapi_allorders,
        "get_trade_orders",
        lambda order_type, symbol, max_age_seconds: [
            {"side": "BUY", "price": 100.0, "quantity": 2.0},
            {"side": "BUY", "price": float("nan"), "quantity": 5.0},
            {"side": "SELL", "price": True, "quantity": 3.0},
            "malformed",
        ],
    )
    st = bapi_allorders.get_total_traded_stats("BTCUSDC", period_seconds=True)
    assert st["BUY"]["total_quantity"] == pytest.approx(2.0)
    assert st["BUY"]["total_value"] == pytest.approx(200.0)
    assert st["SELL"]["trade_count"] == 0

def test_bapi_ws_normalize_now_ms_true_nan_falls_back_to_current_iter42_s2():
    """[ITER42 #2] bapi_ws_normalize: now_ms=True / NaN falls back to current epoch ms; boolean commission ignored."""
    # 2. bapi_ws_normalize: now_ms=True / NaN falls back to current epoch ms; boolean commission ignored
    ev = {
        "s": "BTCUSDC",
        "x": "TRADE",
        "X": "FILLED",
        "S": "BUY",
        "i": 1001,
        "t": 2002,
        "l": "0.5",
        "L": "60000.0",
        "n": True,
        "N": False,
    }
    sym1, ord_row = bapi_ws_normalize.parse_order_execution_report(ev, now_ms=True)
    assert sym1 == "BTCUSDC"
    assert ord_row["timestamp"] > 1_700_000_000_000
    sym2, ord_row2 = bapi_ws_normalize.parse_order_execution_report(ev, now_ms=float("nan"))
    assert sym2 == "BTCUSDC"
    assert ord_row2["timestamp"] > 1_700_000_000_000
    sym3, tr_row = bapi_ws_normalize.parse_trade_execution_report(ev, now_ms=True)
    assert sym3 == "BTCUSDC"
    assert tr_row["time"] > 1_700_000_000_000
    assert "commission" not in tr_row
    assert "commissionAsset" not in tr_row

def test_bapi_ws_health_mark_ws_event_received_ignores_bool_nan_and_falls_back_iter43_s1():
    """[ITER43 #1] bapi_ws_health.mark_ws_event_received ignores bool/NaN and falls back to wall clock."""
    # 1. bapi_ws_health.mark_ws_event_received ignores bool/NaN and falls back to wall clock
    ts_bool = ws_bridge.mark_ws_event_received(True)
    assert ts_bool > 1_700_000_000.0
    ts_nan = ws_bridge.mark_ws_event_received(float("nan"))
    assert ts_nan > 1_700_000_000.0

def test_market_data_fetch_ohlc_candles_rejects_boolean_window_parameters_iter43_s2():
    """[ITER43 #2] market_data.fetch_ohlc_candles rejects boolean window parameters."""
    # 2. market_data.fetch_ohlc_candles rejects boolean window parameters
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", start_ms=True)
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", lookback_days=True)

def test_trailingstop_rejects_boolean_price_balance_trend_iter43_s5():
    """[ITER43 #5] TrailingStop rejects boolean price/balance/trend."""
    # 5. TrailingStop rejects boolean price/balance/trend
    dummy_api = SimpleNamespace(
        get_current_price=lambda pair: True,
        get_account_assets_balances=lambda: [{"asset": "BTC", "free": True, "locked": False}],
    )
    dummy_po = SimpleNamespace(position_cost_basis=lambda *a, **k: 50000.0)
    dummy_sym = SimpleNamespace(symbols=["BTCUSDC"])
    ts = trailing_stop.TrailingStop(dummy_api, dummy_po, dummy_sym, enabled=False)
    ts._balances = [{"asset": "BTC", "free": True, "locked": True}]
    assert ts.price("BTCUSDC") is None
    assert ts.free_qty("BTC") == 0.0
    assert ts.total_qty("BTC") == 0.0
    assert ts.cost_basis("BTCUSDC") is None

def test_bapi_placeorder__submit_binance_order_rejects_bool_nan_inf_even_when_iter51_s1():
    """[ITER51 #1] bapi_placeorder._submit_binance_order rejects bool/NaN/Inf/<=0 even when cache_permit is None."""
    # 1. bapi_placeorder._submit_binance_order rejects bool/NaN/Inf/<=0 even when cache_permit is None
    dummy_bclient = SimpleNamespace(
        order_limit_buy=lambda **kw: {"orderId": 1},
        order_market_buy=lambda **kw: {"orderId": 2},
    )
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order(
            "BUY", "BTCUSDC", True, price=50000.0, market=False, api_client=dummy_bclient
        )
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order(
            "BUY", "BTCUSDC", 0.01, price=float("nan"), market=False, api_client=dummy_bclient
        )
    with pytest.raises(bapi_placeorder.SubmissionRefused):
        bapi_placeorder._submit_binance_order(
            "BUY", "BTCUSDC", float("nan"), price=None, market=True, api_client=dummy_bclient
        )
    with pytest.raises(ValueError):
        bapi_placeorder.order_candidate_price("BUY", True, 100.0)
    assert bapi_placeorder.order_candidate_price("BUY", 100.0, True) == 100.0
    assert bapi_placeorder.order_candidate_price("BUY", 100.0, float("nan")) == 100.0

def test_bapi_allorders_paginate_my_trades_rejects_bool_start_time_ms_normalizes_bool_limit_iter51_s2():
    """[ITER51 #2] bapi_allorders.paginate_my_trades rejects bool start_time_ms & normalizes bool limit to 1000."""
    # 2. bapi_allorders.paginate_my_trades rejects bool start_time_ms & normalizes bool limit to 1000
    assert bapi_allorders.paginate_my_trades(SimpleNamespace(), "BTCUSDC", start_time_ms=True) == []
    seen_limits = []
    assert (
        bapi_allorders.paginate_my_trades(
            SimpleNamespace(get_my_trades=lambda **kw: seen_limits.append(kw["limit"]) or []),
            "BTCUSDC",
            start_time_ms=1000,
            limit=True,
        )
        == []
    )
    assert seen_limits == [1000]

def test_trailingstop_reconcile_pending_handles_nan_bool_filled_qty_and_req_qty_iter51_s3():
    """[ITER51 #3] TrailingStop.reconcile_pending handles NaN/bool filled_qty and req_qty safely."""
    # 3. TrailingStop.reconcile_pending handles NaN/bool filled_qty and req_qty safely
    ts = trailing_stop.TrailingStop(
        api=SimpleNamespace(), po=SimpleNamespace(), sym=SimpleNamespace(symbols=[]), log=lambda _m: None
    )
    ts.order_lifecycle = SimpleNamespace(
        reconcile=lambda pending, persist: SimpleNamespace(
            outcome="terminal",
            status=SimpleNamespace(status="closed", filled_qty=float("nan")),
        )
    )
    st_dict = {"retry_counts": {}}
    persist_fn = lambda _val: None
    persist_fn.st = st_dict
    pending_obj = {"symbol": "BTCUSDC", "action": "SELL", "submitted_qty": float("nan"), "attempt": True}
    res = ts.reconcile_pending(pending_obj, persist_fn)
    assert res.outcome == "terminal"
    assert st_dict["retry_counts"]["SELL"] == 1

def test_binance_api_market_data_parse_interval_minutes_fetch_ohlc_candles_reject_float_inf_nan_iter52_s1():
    """[ITER52 #1] binance_api/market_data: parse_interval_minutes & fetch_ohlc_candles reject float/Inf/NaN/bool."""
    # 1. binance_api/market_data: parse_interval_minutes & fetch_ohlc_candles reject float/Inf/NaN/bool
    with pytest.raises(ValueError):
        b_md.parse_interval_minutes(float("inf"))
    with pytest.raises(ValueError):
        b_md.parse_interval_minutes(60.5)
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, timeout=True)
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, timeout=float("nan"))
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, end_ms=float("inf"))
    with pytest.raises(ValueError):
        b_md.fetch_ohlc_candles("BTCUSDC", 60, start_ms=-5, end_ms=1000)

def test_binance_trailingstop__sync_retry_count_from_state__order_intent_iter58_s1():
    """[ITER58 #1] Binance TrailingStop._sync_retry_count_from_state & _order_intent."""
    # 1. Binance TrailingStop._sync_retry_count_from_state & _order_intent
    ts = object.__new__(TrailingStop)
    ts._retry_counts = {}
    ts.order_lifecycle = MagicMock()
    ts.order_lifecycle.new_intent.side_effect = lambda **kw: kw
    dummy_persist = type("P", (), {"st": {"retry_counts": {"SELL": True}}})()
    ts._sync_retry_count_from_state("BTCUSDC", "SELL", dummy_persist)
    assert ts._retry_counts.get(("BTCUSDC", "SELL"), 0) == 0
    intent = ts._order_intent("SELL", "BTCUSDC", "BTC", "BTCUSDC", 0.1, 60000.0, anchor=None)
    assert ":60000.00000000" in intent["intent_id"]

def test_binanceprovider_open_orders_order_status_reject_bool_nan_iter58_s3(monkeypatch):
    """[ITER58 #3] BinanceProvider.open_orders & order_status reject bool/NaN."""
    # 3. BinanceProvider.open_orders & order_status reject bool/NaN
    bp = BinanceProvider()
    fake_bapi = MagicMock()
    fake_bapi.client.get_open_orders.return_value = [{"orderId": 1, "price": float("nan"), "origQty": 1.0, "executedQty": 0.0}]
    monkeypatch.setattr("providers.market_api._get_bapi", lambda: fake_bapi)
    with pytest.raises(ProviderError):
        bp.open_orders("BTCUSDC")

    fake_bapi.client.get_order.return_value = {"status": "NEW", "executedQty": True, "cummulativeQuoteQty": 0.0, "origQty": 1.0}
    with pytest.raises(ProviderError):
        bp.order_status("BTCUSDC", "123")

def test_parse_trade_execution_report_derives_price_from_y_z_when_l_iter61_s1():
    """[ITER61 #1] parse_trade_execution_report derives price from Y/z when L, p, l are NaN/0."""
    # 1. parse_trade_execution_report derives price from Y/z when L, p, l are NaN/0
    sym_t, trd = bapi_ws_normalize.parse_trade_execution_report(
        {
            "s": "BTCUSDC",
            "x": "TRADE",
            "X": "FILLED",
            "t": 9,
            "i": 11,
            "l": "nan",
            "z": "0.5",
            "L": "nan",
            "p": "0",
            "Y": "25000",
            "S": "BUY",
            "T": 1000,
        }
    )
    assert sym_t == "BTCUSDC"
    assert float(trd["qty"]) == pytest.approx(0.5)
    assert float(trd["price"]) == pytest.approx(50000.0)

def test_bapi_client_sync_time_rejects_boolean_invalid_servertime_without_corrupting_iter61_s4(monkeypatch):
    """[ITER61 #4] bapi_client.sync_time rejects boolean/invalid serverTime without corrupting timestamp_offset."""
    # 4. bapi_client.sync_time rejects boolean/invalid serverTime without corrupting timestamp_offset
    fake_bclient = MagicMock()
    fake_bclient.timestamp_offset = 0
    fake_bclient.get_server_time.return_value = {"serverTime": True}
    monkeypatch.setattr(bapi_client, "_client", fake_bclient)
    assert bapi_client.sync_time(safety_margin_ms=float("nan")) is None
    assert fake_bclient.timestamp_offset == 0

def test_binancepricestream__process_message_rejects_non_integer_e_timestamps_and_iter67_s1():
    """[ITER67 #1] BinancePriceStream._process_message rejects non-integer E timestamps and accepts valid int E."""
    # 1. BinancePriceStream._process_message rejects non-integer E timestamps and accepts valid int E
    ps = BinancePriceStream(symbols=["BTCUSDC"])
    ps._process_message(json.dumps({"data": {"s": "BTCUSDC", "c": "64600.25", "E": "not-an-int"}}))
    assert ps.get_price("BTCUSDC") is None
    ps._process_message(json.dumps({"data": {"s": "BTCUSDC", "c": "64500.5", "E": 1700000000000}}))
    assert ps.get_price("BTCUSDC") == 64500.5

def test_binanceaccountstream__watchdog_loop_handles__last_event_ts_age_inf_without_overflowerror_iter67_s2():
    """[ITER67 #2] BinanceAccountStream._watchdog_loop handles _last_event_ts == 0.0 (age == inf) without OverflowError."""
    # 2. BinanceAccountStream._watchdog_loop handles _last_event_ts == 0.0 (age == inf) without OverflowError
    unhealthy_called = []
    acc_stream = BinanceAccountStream(
        on_event=lambda _e: None,
        on_unhealthy=lambda: (unhealthy_called.append(True), acc_stream._stop_event.set()),
        loss_timeout_sec=0.01,
    )
    acc_stream._available = True
    acc_stream._healthy = True
    acc_stream._last_event_ts = 0.0
    acc_stream._watchdog_loop()
    assert unhealthy_called == [True]

def test_bapi_ws_normalize_strips_uppercases_symbols_rejects_whitespace_symbols_and_iter82_s2():
    """[ITER82 #2] bapi_ws_normalize strips/uppercases symbols, rejects whitespace symbols, and replaces negative timestamps with fallback."""
    # 2. bapi_ws_normalize strips/uppercases symbols, rejects whitespace symbols, and replaces negative timestamps with fallback
    assert bapi_ws_normalize.parse_order_execution_report({"s": "   ", "x": "TRADE", "X": "FILLED", "t": 1}) is None
    res_order = bapi_ws_normalize.parse_order_execution_report(
        {"s": " btcusdc ", "x": "TRADE", "X": "FILLED", "t": 42, "i": 100, "L": "65000", "l": "0.01", "T": -50},
        now_ms=1700000000000,
    )
    assert res_order is not None
    sym, order_item = res_order
    assert sym == "BTCUSDC"
    assert order_item["symbol"] == "BTCUSDC"
    assert order_item["timestamp"] == 1700000000000

    assert bapi_ws_normalize.parse_trade_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "S": "BUY", "t": "   ", "i": None}
    ) is None

def test_providers_binance_filters__parse_bool_flag_handles_nan_and_binanceorderrules_rejects_iter85_s3():
    """[ITER85 #3] providers.binance_filters._parse_bool_flag handles NaN and BinanceOrderRules rejects boolean baseAsset/quoteAsset."""
    # 3. providers.binance_filters._parse_bool_flag handles NaN and BinanceOrderRules rejects boolean baseAsset/quoteAsset
    assert _parse_bool_flag(float("nan"), default=False) is False
    assert _parse_bool_flag("unknown", default=True) is True
    with pytest.raises(BinanceFilterError, match="baseAsset and quoteAsset must not be boolean"):
        BinanceOrderRules.from_symbol_info({
            "baseAsset": True,
            "quoteAsset": "USDC",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01", "maxPrice": "100000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
            ],
        })

def test_binanceprovider_submit_order_rejects_invalid_sides_instead_of_mapping_iter86_s2(monkeypatch):
    """[ITER86 #2] BinanceProvider.submit_order rejects invalid sides instead of mapping them to SELL or BUY & BinanceProvider.order_status normalizes lowercase/whitespace venue status."""
    # 2. BinanceProvider.submit_order rejects invalid sides instead of mapping them to SELL or BUY
    bp = BinanceProvider()
    for bad_side in ("", None, "HOLD", "bad", True):
        with pytest.raises(SubmissionRefused, match="invalid_side"):
            bp.submit_order("BTCUSDC", bad_side, 0.01, price=65000.0)


    # 3. BinanceProvider.order_status normalizes lowercase/whitespace venue status
    monkeypatch.setattr(
        market_api_mod,
        "_get_bapi",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                get_order=lambda symbol, orderId: {
                    "status": " filled ",
                    "executedQty": "0.0",
                    "cummulativeQuoteQty": "0.0",
                    "origQty": "1.0",
                }
            )
        ),
    )
    st = bp.order_status("BTCUSDC", "12345")
    assert st.status == "closed"
    assert st.venue_status == "FILLED"

def test_marketdataprovider_last_opposite_fill_bool_invalid_order_type_fractional_decimals_in_iter96_s3():
    """[ITER96 #3] MarketDataProvider.last_opposite_fill bool/invalid order_type & fractional decimals in round_quantity/round_price & ReplayMarketDataProvider fractional advance, case-insensitive/whitespace free_balance, and get_orders side validation & resolve_assets & balance_cap_quantity bool rejection & BinanceProvider.supports_symbol & non-integer float order_id rejection + MarketApi.latest_fill_price."""
    # 3. MarketDataProvider.last_opposite_fill bool/invalid order_type & fractional decimals in round_quantity/round_price
    dp = DummySpotExecutionProvider("dummy")
    dp.get_orders = lambda symbol, side, since_s: [{"side": side or "BUY", "price": 100.0, "qty": 1.0, "timestamp": 1000}]
    dp.pair_precision = lambda symbol: SimpleNamespace(volume_decimals=2.5, price_decimals=3.5, order_min=0.01)
    assert dp.last_opposite_fill("BTCUSDC", True) is None
    assert dp.last_opposite_fill("BTCUSDC", "INVALID") is None
    assert dp.last_opposite_fill("BTCUSDC", "BUY") == pytest.approx(100.0)
    assert dp.round_quantity("BTCUSDC", 1.23456) == pytest.approx(1.23456)
    assert dp.round_price("BTCUSDC", 1.23456) == pytest.approx(1.23456)


    # 4. ReplayMarketDataProvider fractional advance, case-insensitive/whitespace free_balance, and get_orders side validation
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0), (2000.0, 51000.0)]})
    assert rp.advance("BTCUSDC", 1.5) is None
    assert rp.advance("BTCUSDC", 1) == pytest.approx(50000.0)
    rp.place_order("BTCUSDC", "BUY", 50000.0, 1.5)
    assert rp.free_balance(" btc ") == pytest.approx(1.5)
    assert rp.free_balance(True) == 0.0
    assert rp.get_orders("BTCUSDC", True, 3600.0) == []
    assert rp.get_orders("BTCUSDC", "INVALID", 3600.0) == []
    assert len(rp.get_orders("BTCUSDC", "BUY", 3600.0)) == 1


    # 5. resolve_assets & balance_cap_quantity bool rejection
    assert resolve_assets("BTCUSDC", base=True, quote=False) == ("BTC", "USDC")
    with pytest.raises(ValueError, match="side must be BUY or SELL"):
        balance_cap_quantity(lambda _a: 100.0, "BTCUSDC", True, 100.0)


    # 6. BinanceProvider.supports_symbol & non-integer float order_id rejection + MarketApi.latest_fill_price
    bp = BinanceProvider()
    assert bp.supports_symbol(None) is False
    assert bp.supports_symbol(True) is False
    assert bp.supports_symbol(" btcusdc ") is True
    assert bp.supports_symbol(" hypeusdc ") is False
    with pytest.raises(ProviderError, match="invalid order id"):
        bp.order_status("BTCUSDC", 123.7)
    with pytest.raises(ProviderError, match="invalid order id"):
        bp.cancel_order("BTCUSDC", 123.7)

    ma = MarketApi([dp])
    with pytest.raises(ValueError, match="side must be BUY or SELL"):
        ma.latest_fill_price("BTCUSDC", True, 60.0)

def test_bapi_placeorder_order_candidate_price_supports_b_s_and_rejects_bool_iter106_s4():
    """[ITER106 #4] bapi_placeorder: order_candidate_price supports 'b'/'s' and rejects bool."""
    # 4. bapi_placeorder: order_candidate_price supports 'b'/'s' and rejects bool
    assert bapi_placeorder.order_candidate_price(" b ", 100.0, 95.0) == 95.0
    assert bapi_placeorder.order_candidate_price(" s ", 100.0, 105.0) == 105.0

def test_bapi_ws_normalize_parse_order_execution_report_rejects_non_integer_float_orderid_or_iter108_s2():
    """[ITER108 #2] bapi_ws_normalize: parse_order_execution_report rejects non-integer float orderId or boolean side."""
    # 2. bapi_ws_normalize: parse_order_execution_report rejects non-integer float orderId or boolean side
    assert bapi_ws_normalize.parse_order_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": 100, "i": float("nan"), "S": "BUY", "l": "1.0", "L": "100.0"}
    ) is None
    assert bapi_ws_normalize.parse_order_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": 100, "i": 200, "S": True, "l": "1.0", "L": "100.0"}
    ) is None

    # parse_trade_execution_report sanitizes fallback trade_id and commission fields
    res_trade = bapi_ws_normalize.parse_trade_execution_report(
        {
            "s": "BTCUSDC",
            "x": "TRADE",
            "S": "BUY",
            "t": None,
            "i": 555,
            "T": True,
            "L": "100.0",
            "l": "1.0",
            "n": "nan",
            "N": 12345,
        },
        now_ms=1700000000000,
    )
    assert res_trade is not None
    _sym, trade_row = res_trade
    assert trade_row["id"] == "555-1700000000000"
    assert "commission" not in trade_row
    assert "commissionAsset" not in trade_row

def test_bapi_ws_bridge_handle_binance_ws_event_ignores_unhashable_or_boolean_event_types_iter108_s3():
    """[ITER108 #3] bapi_ws_bridge: handle_binance_ws_event ignores unhashable or boolean event types."""
    # 3. bapi_ws_bridge: handle_binance_ws_event ignores unhashable or boolean event types
    ws_bridge.handle_binance_ws_event({"e": ["unhashable"]})
    ws_bridge.handle_binance_ws_event({"e": True})

def test_binance_api_trailing_stop_string_false_enabled_flag_and_boolean_iter109_s4(tmp_path):
    """[ITER109 #4] binance_api.trailing_stop: string 'false' enabled flag and boolean symbol guards."""
    # 4. binance_api.trailing_stop: string 'false' enabled flag and boolean symbol guards
    dummy_api = SimpleNamespace()
    dummy_po = SimpleNamespace()
    dummy_sym = SimpleNamespace(symbols=["BTCUSDC"])
    ts = trailing_stop.TrailingStop(dummy_api, dummy_po, dummy_sym, enabled="false", state_file=str(tmp_path / "trail.json"))
    assert ts.enabled is False
    assert ts.rebuy_enabled_for(True) is False
    assert ts.rebuy_configured_for(True) is False

def test_binanceprovider_open_orders_rejects_missing_bool_orderid_and_bool_iter112_s1(monkeypatch):
    """[ITER112 #1] BinanceProvider.open_orders rejects missing/bool orderId and bool status/side."""
    # 1. BinanceProvider.open_orders rejects missing/bool orderId and bool status/side
    b_prov = BinanceProvider()
    monkeypatch.setattr(
        market_api_mod,
        "_get_bapi",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                get_open_orders=lambda symbol: [{"orderId": None, "price": "100", "origQty": "1", "executedQty": "0"}]
            )
        ),
    )
    with pytest.raises(ProviderError):
        b_prov.open_orders("BTCUSDC")

def test_bapi_placeorder__normalize_side_and__last_opposite_fill_price_api_iter115_s1():
    """[ITER115 #1] bapi_placeorder._normalize_side and _last_opposite_fill_price_api."""
    # 1. bapi_placeorder._normalize_side and _last_opposite_fill_price_api
    assert bapi_placeorder._normalize_side(" b ") == "BUY"
    assert bapi_placeorder._normalize_side(" s ") == "SELL"
    with pytest.raises( bapi_placeorder.SubmissionRefused ):
        bapi_placeorder.apply_weight_limit("BTCUSDC", True, 100.0, 1.0, 2.0)
    with pytest.raises( bapi_placeorder.SubmissionRefused ):
        bapi_placeorder.apply_weight_limit("BTCUSDC", "INVALID", 100.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        bapi_placeorder._last_opposite_fill_price_api("BTCUSDC", "INVALID")
    with pytest.raises(ValueError):
        bapi_placeorder._last_opposite_fill_price_api("BTCUSDC", True)

def test_binancewsbase_stop_timeout_and_binanceaccountstream_keepalive_sec_loss_timeout_sec_iter117_s3():
    """[ITER117 #3] BinanceWSBase.stop timeout and BinanceAccountStream keepalive_sec / loss_timeout_sec."""
    # 3. BinanceWSBase.stop timeout and BinanceAccountStream keepalive_sec / loss_timeout_sec
    ps = BinancePriceStream(["BTCUSDC"])
    ps.stop(timeout=float("nan"))
    ps.stop(timeout=True)
    acc_stream = BinanceAccountStream(
        on_event=lambda *a: None,
        keepalive_sec=True,
        loss_timeout_sec=-5.0,
    )
    assert acc_stream.keepalive_sec == float(WS_USERDATA_KEEPALIVE_SEC)
    assert acc_stream.loss_timeout_sec == float(WS_USERDATA_LOSS_TIMEOUT_SEC)

def test_parse_order_execution_report_rejects_none_or_orderid_iter117_s4():
    """[ITER117 #4] parse_order_execution_report rejects None or <= 0 orderId."""
    # 4. parse_order_execution_report rejects None or <= 0 orderId
    base_exec = {
        "e": "executionReport",
        "s": "BTCUSDC",
        "x": "TRADE",
        "X": "FILLED",
        "t": 101,
        "S": "BUY",
        "l": "1.0",
        "z": "1.0",
        "L": "50000.0",
        "Z": "50000.0",
        "T": 1700000000000,
    }
    assert parse_order_execution_report({**base_exec, "i": None}) is None
    assert parse_order_execution_report({**base_exec, "i": 0}) is None
    assert parse_order_execution_report({**base_exec, "i": "-5"}) is None
    parsed_exec = parse_order_execution_report({**base_exec, "i": 42})
    assert parsed_exec is not None and parsed_exec[1]["orderId"] == 42

def test_bapi_allorders_paginate_my_trades_symbol_validation_and_limit_cap_at_iter120_s1():
    """[ITER120 #1] bapi_allorders.paginate_my_trades symbol validation and limit cap at 1000."""
    # 1. bapi_allorders.paginate_my_trades symbol validation and limit cap at 1000
    assert bapi_allorders.paginate_my_trades(object(), "", 1000) == []
    assert bapi_allorders.paginate_my_trades(object(), True, 1000) == []
    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        bapi_allorders.paginate_my_trades(object(), "  ", 1000, strict=True)
    captured_kw = {}
    dummy_api = SimpleNamespace(get_my_trades=lambda **kw: captured_kw.update(kw) or [])
    bapi_allorders.paginate_my_trades(dummy_api, " btcusdc ", 1000, limit=5000)
    assert captured_kw["symbol"] == "BTCUSDC"
    assert captured_kw["limit"] == 1000

def test_parse_trade_execution_report_does_not_leak_invalid_event_i_into_iter120_s2():
    """[ITER120 #2] parse_trade_execution_report does not leak invalid event['i'] into trade_item['orderId']."""
    # 2. parse_trade_execution_report does not leak invalid event["i"] into trade_item["orderId"]
    res = parse_trade_execution_report(
        {
            "s": "BTCUSDC",
            "x": "TRADE",
            "S": "BUY",
            "t": 999,
            "i": float("nan"),
            "L": "50000.0",
            "l": "0.1",
            "T": 1700000000000,
        }
    )
    assert res is not None
    assert res[1]["orderId"] is None
    res2 = parse_trade_execution_report(
        {
            "s": "BTCUSDC",
            "x": "TRADE",
            "S": "BUY",
            "t": 999,
            "i": -5,
            "L": "50000.0",
            "l": "0.1",
            "T": 1700000000000,
        }
    )
    assert res2 is not None and res2[1]["orderId"] is None

def test_bapi_trades__valid_fill_id_rejects_containers_non_integer_floats_and_iter130_s1():
    """[ITER130 #1] bapi_trades._valid_fill_id rejects containers, non-integer floats, and sentinel strings while normalizing integer floats."""
    # 1. bapi_trades._valid_fill_id rejects containers, non-integer floats, and sentinel strings while normalizing integer floats
    assert bapi_trades._valid_fill_id(777.0) == 777
    assert bapi_trades._valid_fill_id(12.5) is None
    assert bapi_trades._valid_fill_id("nan") is None
    assert bapi_trades._valid_fill_id("null") is None
    assert bapi_trades._valid_fill_id({"a": 1}) is None
    assert bapi_trades._valid_fill_id([123]) is None

def test_bapi_placeorder__submit_binance_order_replaces_boolean_blank_client_order_id_and_tolerates_iter130_s4(monkeypatch):
    """[ITER130 #4] bapi_placeorder._submit_binance_order replaces boolean/blank client_order_id and tolerates order dict without 'orderId' key."""
    # 4. bapi_placeorder._submit_binance_order replaces boolean/blank client_order_id and tolerates order dict without 'orderId' key
    captured = {}

    def _fake_limit_buy(**kw):
        captured.update(kw)
        return {"id": "ord-999"}

    fake_api = SimpleNamespace(order_limit_buy=_fake_limit_buy)
    monkeypatch.setattr(bapi_placeorder.sym, "validate_symbols", lambda _s: None)
    monkeypatch.setattr(bapi_placeorder.cfg, "is_trade_enabled", lambda: True)
    monkeypatch.setattr(bapi_placeorder, "require_account_cache_for_submit", lambda: None)
    monkeypatch.setattr(bapi_placeorder.rc, "create_client_order_id", lambda: "AUTO_CID_1")
    res = bapi_placeorder._submit_binance_order(
        "BUY", "BTCUSDC", 1.0, price=100.0, client_order_id=True, api_client=fake_api
    )
    assert res == {"id": "ord-999"}
    assert captured["newClientOrderId"] == "AUTO_CID_1"

def test_bapi_ws_normalize_rejects_sentinel_strings_nan_null_and_containers_iter131_s2():
    """[ITER131 #2] bapi_ws_normalize rejects sentinel strings ('0', 'nan', 'null') and containers for order/trade IDs."""
    # 2. bapi_ws_normalize rejects sentinel strings ("0", "nan", "null") and containers for order/trade IDs
    assert (
        bapi_ws_normalize.parse_order_execution_report(
            {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "S": "BUY", "i": "nan", "t": 10, "l": "1", "L": "100"}
        )
        is None
    )
    assert (
        bapi_ws_normalize.parse_order_execution_report(
            {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "S": "BUY", "i": "0", "t": 10, "l": "1", "L": "100"}
        )
        is None
    )
    assert (
        bapi_ws_normalize.parse_trade_execution_report(
            {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "S": "BUY", "i": "null", "t": "nan", "l": "1", "L": "100"}
        )
        is None
    )
    assert (
        bapi_ws_normalize.parse_trade_execution_report(
            {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "S": "BUY", "i": [1], "t": {"a": 1}, "l": "1", "L": "100"}
        )
        is None
    )

def test_bapi_get_current_price_normalizes_symbol_casing_whitespace_to_hit_iter133_s1():
    """[ITER133 #1] bapi.get_current_price normalizes symbol casing/whitespace to hit cprice cache."""
    # 1. bapi.get_current_price normalizes symbol casing/whitespace to hit cprice cache
    bapi.cprice["BTCUSDC"] = 64000.0
    bapi.cprice_time["BTCUSDC"] = bapi.time.time()
    assert bapi.get_current_price(" btcusdc ") == pytest.approx(64000.0)

def test_bapi_get_open_orders_rejects_sentinel_container_orderid_and_normalizes_iter133_s2(monkeypatch):
    """[ITER133 #2] bapi.get_open_orders rejects sentinel/container orderId and normalizes integer float orderId."""
    # 2. bapi.get_open_orders rejects sentinel/container orderId and normalizes integer float orderId
    monkeypatch.setattr(bapi.sym, "validate_params", lambda *_a: None)
    monkeypatch.setattr(
        bapi,
        "client",
        SimpleNamespace(
            get_open_orders=lambda symbol: [
                {"orderId": "nan", "side": "BUY", "price": "100", "origQty": "1"},
                {"orderId": {"bad": 1}, "side": "BUY", "price": "100", "origQty": "1"},
                {"orderId": 444.0, "side": "BUY", "price": "100", "origQty": "1"},
            ],
            cancel_order=lambda **kw: (_ for _ in ()).throw(AssertionError(f"Should not cancel {kw}")),
        ),
    )
    open_ords = bapi.get_open_orders("BUY", "BTCUSDC")
    assert list(open_ords.keys()) == [444]

def test_bapi_cancel_order_rejects_sentinel_strings_nan_null_and_iter133_s3():
    """[ITER133 #3] bapi.cancel_order rejects sentinel strings ('nan', 'null') and containers."""
    # 3. bapi.cancel_order rejects sentinel strings ("nan", "null") and containers
    assert bapi.cancel_order("BTCUSDC", "nan") is False
    assert bapi.cancel_order("BTCUSDC", "null") is False
    assert bapi.cancel_order("BTCUSDC", {"a": 1}) is False

def test_bapi_client_stop_periodic_resync_tolerates_boolean_nan_negative_timeout_iter133_s4():
    """[ITER133 #4] bapi_client.stop_periodic_resync tolerates boolean/NaN/negative timeout."""
    # 4. bapi_client.stop_periodic_resync tolerates boolean/NaN/negative timeout
    assert bapi_client.stop_periodic_resync(timeout=True) is True
    assert bapi_client.stop_periodic_resync(timeout=float("nan")) is True
    assert bapi_client.stop_periodic_resync(timeout=-5.0) is True

def test_providers_binance_filters_py_binanceorderrules_normalize_string_boolean_market_iter154_s4():
    """[ITER154 #4] providers/binance_filters.py: BinanceOrderRules.normalize string boolean market='false'."""
    # 4. providers/binance_filters.py: BinanceOrderRules.normalize string boolean market="false"
    rules = BinanceOrderRules.from_symbol_info({
        "baseAsset": "BTC",
        "quoteAsset": "USDC",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "1.0", "maxPrice": "1000000.0"},
            {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "1000.0"},
        ],
    })
    norm_qty, norm_px = rules.normalize(quantity="1.234", price="100.56", market="false")  # type: ignore[arg-type]
    assert norm_qty == Decimal("1.23")
    assert norm_px == Decimal("100.50")

def test_bapi_cancel_order_check_order_filled_and_get_open_orders_reject_negative_orderids_iter159_s3(monkeypatch):
    """[ITER159 #3] bapi: cancel_order, check_order_filled, and get_open_orders reject '0'/negative orderIds and parse bool flags."""
    # 3. bapi: cancel_order, check_order_filled, and get_open_orders reject "0"/negative orderIds and parse bool flags
    monkeypatch.setattr(
        bapi,
        "client",
        SimpleNamespace(
            cancel_order=lambda **kw: (_ for _ in ()).throw(AssertionError("should not call cancel")),
            get_open_orders=lambda symbol: [
                {"orderId": 0, "side": "BUY", "price": "60000", "origQty": "0.01", "executedQty": "0", "time": 1000},
                {"orderId": "0.0", "side": "BUY", "price": "60000", "origQty": "0.01", "executedQty": "0", "time": 1000},
                {"orderId": "777.0", "side": "BUY", "price": "60000", "origQty": "0.01", "executedQty": "0", "time": 1000, "clientOrderId": "None"},
            ],
        ),
    )
    assert bapi.cancel_order("BTCUSDC", 0) is False
    assert bapi.cancel_order("BTCUSDC", "0.0") is False
    assert bapi.cancel_order("BTCUSDC", -5) is False

    open_map = bapi.get_open_orders("BUY", "BTCUSDC", strict="false")
    assert list(open_map.keys()) == ["777"]
    assert open_map["777"]["clientOrderId"] == ""

def test_bapi_placeorder_order_candidate_price_and__submit_binance_order_handle_market_false_and_iter159_s4(monkeypatch):
    """[ITER159 #4] bapi_placeorder: order_candidate_price and _submit_binance_order handle market='false' and client_order_id='0'."""
    # 4. bapi_placeorder: order_candidate_price and _submit_binance_order handle market="false" and client_order_id="0"
    assert bapi_placeorder.order_candidate_price("BUY", 60000.0, 60100.0, market="false") == 60000.0
    captured_submit = {}
    monkeypatch.setattr(bapi_placeorder.cfg, "is_trade_enabled", lambda: True)
    monkeypatch.setattr(bapi_placeorder, "require_account_cache_for_submit", lambda: None)
    res_order = bapi_placeorder._submit_binance_order(
        "BUY",
        "BTCUSDC",
        0.01,
        price=60000.0,
        market="false",
        client_order_id="0",
        api_client=SimpleNamespace(order_limit_buy=lambda **kw: captured_submit.update(kw) or {"orderId": 888}),
    )
    assert res_order == {"orderId": 888}
    assert captured_submit["newClientOrderId"] != "0"

def test_bapi_allorders_paginate_my_trades_accepts_tail_id_and_get_filled_orders_does_iter159_s5(monkeypatch):
    """[ITER159 #5] bapi_allorders: paginate_my_trades accepts '100.0' tail id and get_filled_orders does not group orderId=0 fills."""
    # 5. bapi_allorders: paginate_my_trades accepts "100.0" tail id and get_filled_orders does not group orderId=0 fills
    pages = [
        [{"id": "100.0", "orderId": 0, "price": "100.0", "qty": "1.0", "time": int(time.time() * 1000), "isBuyer": True}],
        [{"id": "101", "orderId": "0.0", "price": "200.0", "qty": "1.0", "time": int(time.time() * 1000), "isBuyer": True}],
    ]
    pager = SimpleNamespace(get_my_trades=lambda **kw: pages.pop(0) if pages else [])
    monkeypatch.setattr(bapi_allorders.time, "sleep", lambda _s: None)
    all_trades = bapi_allorders.paginate_my_trades(pager, "BTCUSDC", int(time.time() * 1000) - 10000, limit=1, strict="false")
    assert len(all_trades) == 2

def test_providers_market_api_binanceprovider_ohlc_series_with_marketapi_provider_by_name_with_iter164_s2(monkeypatch):
    """[ITER164 #2] providers.market_api: BinanceProvider.ohlc_series with '15.0', MarketApi.provider_by_name with provider object, and market_regime_context(use_runtime_snapshot='false')."""
    # 2. providers.market_api: BinanceProvider.ohlc_series with "15.0", MarketApi.provider_by_name with provider object, and market_regime_context(use_runtime_snapshot="false")
    bp = BinanceProvider()
    fake_klines = [[0, "1", "2", "0.5", "100.0", "10", 1700000000000] for _ in range(25)]
    monkeypatch.setattr(
        market_api_mod,
        "_get_bapi",
        lambda: SimpleNamespace(client=SimpleNamespace(get_klines=lambda **kw: fake_klines)),
    )
    series = bp.ohlc_series("BTCUSDC", "15.0")
    assert series.interval_min == 15 and len(series.closes) == 24
    assert market_api_mod.api.provider_by_name(bp) is not None

    snap_called = []
    custom_api = MarketApi(
        [bp],
        regime_snapshot_resolver=lambda sym, now=None: (snap_called.append(sym), None)[1],
    )
    custom_api.market_regime_context("BTCUSDC", use_runtime_snapshot="false", allow_fallback="false", now=1700000000.0)
    assert snap_called == []

def test_bapi_ws_health__parse_bool_flag_value_default_bapi_ws_bridge_handle_binance_ws_event_log_enabled_true_iter166_s4():
    """[ITER166 #4] bapi_ws_health._parse_bool_flag(value, default=...) & bapi_ws_bridge.handle_binance_ws_event(log_enabled=True)."""
    # 4. bapi_ws_health._parse_bool_flag(value, default=...) & bapi_ws_bridge.handle_binance_ws_event(log_enabled=True)
    assert ws_bridge._parse_bool_flag("unknown", default=True) is True
    assert ws_bridge._parse_bool_flag("false", default=True) is False
    persisted = []
    ws_bridge.handle_binance_ws_event(
        {"e": "executionReport", "s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "S": "BUY", "i": 123},
        log_enabled=True,
        upsert_order_fn=lambda ev: True,
        append_trade_fn=lambda ev: True,
        persist_fn=lambda et: persisted.append(et),
    )
    assert persisted == ["executionReport"]

def test_binancepricestream_with_scalar_string_symbols_and_slash_formatted_iter167_s3():
    """[ITER167 #3] BinancePriceStream with scalar string symbols and slash-formatted symbol methods."""
    # 3. BinancePriceStream with scalar string symbols and slash-formatted symbol methods
    ps = bapi_ws.BinancePriceStream("btc/usdc")
    assert ps.running_symbols == {"BTCUSDC"}
    assert ps.add_symbol("tao/usdc") is True
    assert "TAOUSDC" in ps.running_symbols
    ps._prices["TAOUSDC"] = 450.0
    assert ps.get_price(" tao/usdc ") == pytest.approx(450.0)
    assert ps.remove_symbol("tao/usdc") is True
    assert "TAOUSDC" not in ps.running_symbols

def test_binance_filters_decimal_places_with_decimal_str_float_and_invalid_iter168_s1():
    """[ITER168 #1] binance_filters.decimal_places with Decimal, str, float, and invalid inputs."""
    # 1. binance_filters.decimal_places with Decimal, str, float, and invalid inputs
    assert decimal_places(Decimal("0.001")) == 3
    assert decimal_places("0.0001") == 4
    assert decimal_places(0.01) == 2
    assert _parse_bool_flag("", default=True) is True
    assert _parse_bool_flag("y", default=False) is True
    assert _parse_bool_flag("n", default=True) is False
    with pytest.raises(BinanceFilterError):
        decimal_places(True)
    with pytest.raises(BinanceFilterError):
        decimal_places(0)

def test_bapi_allorders_snapshot_cache_symbol_rows_copies_rows_safely_under_lock_and_iter185_s2():
    """[ITER185 #2] bapi_allorders.snapshot_cache_symbol_rows copies rows safely under lock and resolves variants."""
    # 2. bapi_allorders.snapshot_cache_symbol_rows copies rows safely under lock and resolves variants
    sync_calls = []
    mgr = SimpleNamespace(
        lock=threading.Lock(),
        cache={"BTCUSDC": [{"orderId": 100, "status": "FILLED"}]},
        sync_account_reader_if_needed=lambda: sync_calls.append(True),
    )
    rows = bapi_allorders.snapshot_cache_symbol_rows(mgr, "BTC/USDC")
    assert sync_calls == [True]
    assert rows == [{"orderId": 100, "status": "FILLED"}]
    # Ensure rows are defensive copies (mutating snapshot does not mutate cache)
    rows[0]["status"] = "CANCELED"
    assert mgr.cache["BTCUSDC"][0]["status"] == "FILLED"

def test_market_data_fetch_ohlc_candles_discards_candles_with_inverted_timestamps_close_time_iter185_s4(monkeypatch):
    """[ITER185 #4] market_data.fetch_ohlc_candles discards candles with inverted timestamps (close_time < open_time)."""
    # 4. market_data.fetch_ohlc_candles discards candles with inverted timestamps (close_time < open_time)
    fake_klines = [
        [1000, "10.0", "15.0", "9.0", "12.0", "1.0", 900, "12.0", 1, "0.5", "6.0", "0"],
        [2000, "12.0", "18.0", "11.0", "17.0", "2.0", 2500, "34.0", 2, "1.0", "17.0", "0"],
    ]
    monkeypatch.setattr(b_md, "fetch_public_json", lambda *a, **kw: fake_klines)
    candles = b_md.fetch_ohlc_candles("BTCUSDC", "1h", start_ms=1000, end_ms=3000)
    assert len(candles) == 1
    assert candles[0]["timestamp"] == 2
    assert candles[0]["close"] == 17.0
