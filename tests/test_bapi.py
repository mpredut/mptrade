"""Unified unit tests for Binance API (bapi, bapi_ws, bapi_trades, bapi_allorders, bapi_placeorder)."""

import os
import sys
import time
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from binance_api import bapi, bapi_allorders as ao, bapi_placeorder as po, bapi_trades, bapi_ws
from binance_api.bapi_ws import BinanceWebSocketManager
from market_regime import MarketRegimeContext, MarketRegimeDecision

SYMBOL = "BTCUSDC"
TAO_SYMBOL = "TAOUSDC"


# ─── WebSocket & UserData Stream ────────────────────────────────────────────

class BapiWsShutdownHandlingTests(unittest.TestCase):
    def test_shutdown_errors_are_detected(self):
        manager = BinanceWebSocketManager()
        self.assertTrue(
            manager._is_shutdown_error(
                RuntimeError("cannot schedule new futures after shutdown")
            )
        )
        self.assertFalse(manager._is_shutdown_error(RuntimeError("boom")))


class TestUserDataClassify(unittest.TestCase):
    """_classify decides between ping / command reply / real event."""

    C = staticmethod(bapi_ws.BinanceUserDataStream._classify)

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
        s = bapi_ws.BinanceUserDataStream(on_event=lambda payload: None)
        s.on_available(True)
        s.on_healthy()
        s.on_unhealthy()

    def test_userdata_health_marks_call_callbacks(self):
        seen = {}
        s = bapi_ws.BinanceUserDataStream(
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


# ─── Trades Reader ──────────────────────────────────────────────────────────

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
    def tearDown(self):
        bapi_trades.cache_trade_manager = None

    def test_bapi_trades_reader_behaviors(self):
        with self.subTest(msg="lazy manager is created without private synchronization"):
            manager = types.SimpleNamespace()
            get_cache_manager = mock.Mock(return_value=manager)
            cache_module = types.SimpleNamespace(get_cache_manager=get_cache_manager)

            with mock.patch.dict(sys.modules, {"cacheManager": cache_module}):
                actual = bapi_trades.init_cache_trade_manager()

            self.assertIs(actual, manager)
            get_cache_manager.assert_called_once_with("Trade", start_sync=False)

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
                mock.patch.object(bapi_trades.sym, "validate_ordertype"),
                mock.patch.object(bapi_trades.sym, "validate_symbols"),
                mock.patch.object(bapi_trades.time, "time", return_value=2_001.0),
            ):
                result = bapi_trades.get_trade_orders("BUY", SYMBOL, 10)

            self.assertFalse(manager.lock.held)
            self.assertEqual(manager.cache[SYMBOL], [])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 11)
            self.assertEqual(result[0]["price"], 100.5)


# ─── REST Price Cache ───────────────────────────────────────────────────────

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


# ─── AllOrders & Trade Pagination ───────────────────────────────────────────

class TestPaginateMyTrades(unittest.TestCase):
    def test_single_page_one_call(self):
        client = MagicMock()
        client.get_my_trades.return_value = [{"id": i, "time": 1} for i in range(10)]
        out = ao.paginate_my_trades(client, SYMBOL, 0, limit=1000)
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
        out = ao.paginate_my_trades(client, SYMBOL, 0, limit=1000)
        self.assertEqual(len(out), 1500)
        self.assertEqual(client.get_my_trades.call_count, 2)
        _, kw2 = client.get_my_trades.call_args_list[1]
        self.assertEqual(kw2.get("fromId"), 1000)
        self.assertNotIn("startTime", kw2)

    def test_empty(self):
        client = MagicMock()
        client.get_my_trades.return_value = []
        self.assertEqual(ao.paginate_my_trades(client, "X", 0), [])

    def test_exact_multiple_of_limit_stops_on_empty(self):
        client = MagicMock()
        page1 = [{"id": i, "time": 1} for i in range(2)]
        client.get_my_trades.side_effect = [page1, []]
        out = ao.paginate_my_trades(client, "X", 0, limit=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(client.get_my_trades.call_count, 2)

    def test_none_response_remains_empty_for_legacy_readers(self):
        client = MagicMock()
        client.get_my_trades.return_value = None
        self.assertEqual(ao.paginate_my_trades(client, "X", 0), [])

    def test_none_response_fails_a_strict_reconciliation(self):
        client = MagicMock()
        client.get_my_trades.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Binance returned no trade page"):
            ao.paginate_my_trades(client, "X", 0, strict=True)


class TestCachedOrders(unittest.TestCase):
    def test_reader_does_not_start_a_duplicate_polling_loop(self):
        manager = MagicMock(cache={SYMBOL: []})
        with patch("cacheManager.get_cache_manager", return_value=manager) as get_manager:
            self.assertEqual(ao.get_trade_orders("BUY", SYMBOL, 60), [])
        get_manager.assert_called_once_with("Order", start_sync=False)

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
            result = ao.get_trade_orders("BUY", SYMBOL, 60)

        self.assertEqual(result[0]["price"], 100.0)
        self.assertEqual(manager.cache[SYMBOL][0]["price"], 999.0)


# ─── PlaceOrder Legacy Adapters, Daily Limit & Market Regime ────────────────

def test_place_safe_order_delegates_to_common_pipeline_without_smart_repricing():
    expected = {"orderId": 101}
    with patch.object(po, "_guarded_market_place", return_value=expected) as place:
        result = po.place_safe_order(
            "buy", TAO_SYMBOL, 225.0, 2.0,
            safeback_seconds=123, force=True, cancelorders=True,
            hours=0.5, bypass_guards={"profit_guard"},
        )

    assert result == expected
    place.assert_called_once_with(
        TAO_SYMBOL, "BUY", 225.0, 2.0,
        smart=False, safeback_seconds=123, force=True,
        cancelorders=True, hours=0.5, bypass_guards={"profit_guard"},
    )


def test_place_order_smart_delegates_to_common_pipeline_with_motivation():
    expected = {"orderId": 202}
    with patch.object(po, "_guarded_market_place", return_value=expected) as place:
        result = po.place_order_smart(
            "sell", TAO_SYMBOL, 230.0, 1.5,
            safeback_seconds=456, force=False, cancelorders=False,
            hours=1.25, pair=True, motivation="legacy-test",
        )

    assert result == expected
    place.assert_called_once_with(
        TAO_SYMBOL, "SELL", 230.0, 1.5,
        smart=True, safeback_seconds=456, force=False,
        cancelorders=False, hours=1.25, motivation="legacy-test",
    )


def test_legacy_adapter_preserves_none_quantity_for_common_decision():
    with (patch.object(po, "_resolve_qty", return_value=3.25) as resolve,
          patch.object(po, "_guarded_market_place", return_value=None) as place):
        po.place_safe_order("BUY", TAO_SYMBOL, 225.0, None)

    resolve.assert_not_called()
    assert place.call_args.args[:4] == (TAO_SYMBOL, "BUY", 225.0, None)


def _trades(n, side="BUY", age_sec=4000.0, price=100.0):
    ts = (time.time() - age_sec) * 1000
    return [{"orderId": i, "price": price, "qty": 1.0, "quantity": 1.0,
             "timestamp": ts, "side": side} for i in range(n)]


class IfPlaceSafeOrderDailyLimitTestCase(unittest.TestCase):
    def _run(self, same_side_trades, opposite_side_trades=None, *, side="BUY"):
        opposite_side_trades = opposite_side_trades or []

        def fake_get_trade_orders(order_type, symbol, max_age_seconds):
            return same_side_trades if order_type == side else opposite_side_trades

        with patch("binance_api.bapi_allorders.get_trade_orders", side_effect=fake_get_trade_orders), \
             patch("binance_api.bapi.get_current_price", return_value=100.0), \
             patch("providers.market_api.api.market_regime_context", return_value=None):
            return po.if_place_safe_order(
                side, SYMBOL, 100.0, 1.0,
                time_back_in_seconds=48 * 3600 + 60,
                bypass_guards={"profit_guard"},
            )

    def test_no_trades_allowed(self):
        with patch.object(po.order_guard, "get_window_cfg") as get_win, \
             patch.object(po.order_guard, "profit_guard", return_value=True) as pg:
            ok, reason = self._run(same_side_trades=[])
        self.assertTrue(ok)
        self.assertIsNone(reason)
        get_win.assert_not_called()
        pg.assert_called_once()
        self.assertEqual(pg.call_args.kwargs.get("bypass_guards"), {"profit_guard"})

    def test_daily_limit_blocks(self):
        # 25 filled orders in 24h reaches the 25/24h cap
        ok, reason = self._run(same_side_trades=_trades(25, age_sec=4000.0))
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_limit")

    def test_daily_limit_blocks_sell_symmetrically(self):
        ok, reason = self._run(same_side_trades=_trades(25, side="SELL", age_sec=4000.0), side="SELL")
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_limit")

    def test_antispam_5s_blocks_and_after_5s_allows(self):
        ok_recent, reason_recent = self._run(same_side_trades=_trades(1, age_sec=2.0))
        self.assertFalse(ok_recent)
        self.assertEqual(reason_recent, "daily_limit")
        ok_buy, reason_buy = self._run(same_side_trades=_trades(1, age_sec=6.0))
        self.assertTrue(ok_buy)
        self.assertIsNone(reason_buy)
        ok_sell, reason_sell = self._run(same_side_trades=_trades(1, side="SELL", age_sec=6.0), side="SELL")
        self.assertTrue(ok_sell)
        self.assertIsNone(reason_sell)


class SafeOrderMarketRegimeBoundaryTest(unittest.TestCase):
    @staticmethod
    def _fresh_context():
        decision = MarketRegimeDecision(
            regime="bull", gradient=0.5, epsilon=0.1,
            strength=5.0, fresh=True, reason="directional_signal",
        )
        return MarketRegimeContext.from_decision(decision, evaluated_at=1_700_000_000.0)

    def _exercise_buy(self, *, regime_context=None, pass_context=False):
        resolved_context = self._fresh_context()
        kwargs = {"regime_context": regime_context} if pass_context else {}
        with (
            mock.patch.object(po.order_guard, "daily_limit_guard", return_value=(True, None)),
            mock.patch.object(po.api, "get_current_price", return_value=100.0),
            mock.patch("binance_api.bapi_allorders.get_trade_orders", return_value=[]),
            mock.patch.object(po.order_guard, "get_window_cfg", return_value=8 * 3600.0) as get_window_cfg,
            mock.patch.object(po.order_guard, "get_margin_cfg", return_value=1.15),
            mock.patch.object(po.order_guard, "profit_guard", return_value=True) as profit_guard,
            mock.patch("providers.market_api.api.market_regime_context", return_value=resolved_context) as resolve_context,
        ):
            result = po.if_place_safe_order(
                "BUY", SYMBOL, 100.0, 1.0, 48 * 3600 + 60, **kwargs,
            )
        return result, resolved_context, resolve_context, get_window_cfg, profit_guard

    def test_explicit_context_is_reused_without_runtime_resolution(self):
        explicit_context = self._fresh_context()
        result, _, resolve_context, get_window_cfg, profit_guard = self._exercise_buy(
            regime_context=explicit_context, pass_context=True,
        )
        self.assertEqual(result, (True, None))
        resolve_context.assert_not_called()
        self.assertIs(get_window_cfg.call_args.kwargs["regime_context"], explicit_context)
        self.assertIs(profit_guard.call_args.kwargs["regime_context"], explicit_context)

    def test_missing_context_is_resolved_once_at_market_api_boundary(self):
        result, resolved_context, resolve_context, get_window_cfg, profit_guard = self._exercise_buy()
        self.assertEqual(result, (True, None))
        self.assertTrue(resolved_context.fresh)
        resolve_context.assert_called_once_with(SYMBOL, provider_name="binance")
        self.assertIs(get_window_cfg.call_args.kwargs["regime_context"], resolved_context)
        self.assertIs(profit_guard.call_args.kwargs["regime_context"], resolved_context)

    def test_if_place_safe_order_preserves_subdollar_precision_and_handles_none_current_price(self):
        with (
            mock.patch.object(po.order_guard, "daily_limit_guard", return_value=(True, None)),
            mock.patch.object(po.api, "get_current_price", return_value=None),
            mock.patch("binance_api.bapi_allorders.get_trade_orders", return_value=[]),
            mock.patch.object(po.order_guard, "get_window_cfg", return_value=3600.0),
            mock.patch.object(po.order_guard, "get_margin_cfg", return_value=1.01),
            mock.patch.object(po.order_guard, "profit_guard", return_value=True) as profit_guard,
            mock.patch("providers.market_api.api.market_regime_context", return_value=self._fresh_context()),
        ):
            ok, reason = po.if_place_safe_order("BUY", SYMBOL, 0.4567, 0.00015, 3600)
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertAlmostEqual(profit_guard.call_args.args[3], 0.4567)
        self.assertAlmostEqual(profit_guard.call_args.kwargs["qty"], 0.00015)

    def test_lowercase_order_type_and_aggregate_trades_and_ws_price_validation(self):
        import threading
        now_ms = int(time.time() * 1000)
        order_mgr = MagicMock(cache={
            SYMBOL: [{"orderId": 1, "price": 100.0, "quantity": 1.0, "timestamp": now_ms, "side": "BUY"}],
        })
        order_mgr.lock = threading.Lock()
        with patch("cacheManager.get_cache_manager", return_value=order_mgr):
            res_ao = ao.get_trade_orders("buy", SYMBOL, 60)
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

        agg = bapi_trades.aggregate_trades([{
            "id": 1, "orderId": 99, "symbol": SYMBOL, "price": "100.0",
            "qty": "0.5", "quoteQty": "50.0", "commission": "0.05",
            "commissionAsset": "USDC", "time": now_ms,
            "isBuyer": True, "isMaker": False, "isBestMatch": True,
        }])
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0]["orderId"], 99)

        stream = bapi_ws.BinancePriceStream(symbols=[SYMBOL])
        stream._process_message('{"data": {"s": "BTCUSDC", "c": "0", "E": 1000}}')
        self.assertIsNone(stream.get_price(SYMBOL))
        stream._process_message('{"data": {"s": "BTCUSDC", "c": "123.45", "E": 1000}}')
        self.assertEqual(stream.get_price(SYMBOL), 123.45)

    def test_short_trend_dynamic_window_uses_now_val_and_snapshot_fresh_guards_none_ts(self):
        import threading
        import cacheManager as cm
        mgr = cm.CacheShortTrendManager([SYMBOL], filename="test_short_trend_tmp.json", writer=False)
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
        self.assertFalse(mgr.is_snapshot_fresh(symbol=None))

    def test_if_place_safe_order_fetches_full_ref_window_and_pricewindow_edge_cases(self):
        import pricewindow as pw
        fetched_ages = []
        now_ms = time.time() * 1000.0

        def fake_get_trade_orders(order_type, symbol, max_age_seconds):
            fetched_ages.append((order_type, max_age_seconds))
            return [
                {"price": 100.0, "timestamp": now_ms - 7200 * 1000.0},
                {"price": 102.0, "timestamp": now_ms - 5400 * 1000.0},
            ]

        with (
            mock.patch.object(po.order_guard, "daily_limit_guard", return_value=(True, None)),
            mock.patch.object(po.api, "get_current_price", return_value=100.0),
            mock.patch("binance_api.bapi_allorders.get_trade_orders", side_effect=fake_get_trade_orders),
            mock.patch.object(po.order_guard, "get_window_cfg", return_value=86400.0),
            mock.patch.object(po.order_guard, "get_margin_cfg", return_value=1.01),
            mock.patch.object(po.order_guard, "profit_guard", return_value=True),
            mock.patch("providers.market_api.api.market_regime_context", return_value=self._fresh_context()),
        ):
            ok, reason = po.if_place_safe_order("BUY", SYMBOL, 98.0, 1.0, 3600)
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(fetched_ages, [("SELL", 86400.0), ("BUY", 86400.0)])

        w = pw.PriceWindow.from_existing_window([100.0, 101.0, 102.0], 5, symbol="BTCUSDC")
        self.assertEqual(w.current_window_size(), 3)
        w.sample_rate_sec = 0
        self.assertEqual(w.recent_n, 2)
        trend, _, _, _ = pw.instant_trend_from_prices([100.0, 101.0, 102.0], None)
        self.assertEqual(trend, 1)

    def test_canonical_order_normalization_and_bapi_trade_unification(self):
        from providers.base import (
            extract_order_price,
            extract_order_qty,
            extract_order_side,
            extract_timestamp_ms,
            normalize_order,
        )
        import binance_api.bapi as bapi_mod
        import binance_api.bapi_allorders as bapi_all
        import binance_api.bapi_trades as bapi_tr

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

        # 3. bapi_trades.aggregate_trades computes VWAP on multi-fill orders and emits unified fields
        raw_fills = [
            {"id": 10, "orderId": 500, "symbol": SYMBOL, "isBuyer": True, "price": "100.0", "qty": "1.0", "quoteQty": "100.0", "commission": "0.1", "commissionAsset": "USDC", "time": 1_700_000_000_000},
            {"id": 11, "orderId": 500, "symbol": SYMBOL, "isBuyer": True, "price": "130.0", "qty": "2.0", "quoteQty": "260.0", "commission": "0.2", "commissionAsset": "USDC", "time": 1_700_000_005_000},
        ]
        agg = bapi_tr.aggregate_trades(raw_fills)
        self.assertEqual(len(agg), 1)
        self.assertAlmostEqual(float(agg[0]["price"]), 120.0)
        self.assertAlmostEqual(float(agg[0]["qty"]), 3.0)
        self.assertAlmostEqual(float(agg[0]["quantity"]), 3.0)
        self.assertEqual(agg[0]["side"], "BUY")
        self.assertTrue(agg[0]["isBuyer"])
        self.assertEqual(agg[0]["time"], 1_700_000_005_000)
        self.assertEqual(agg[0]["timestamp"], 1_700_000_005_000)

        # 4. bapi_trades.get_my_trades_simple preserves distinct partial fill ids under same orderId
        with mock.patch.object(bapi_all, "paginate_my_trades", return_value=raw_fills):
            deduped = bapi_tr.get_my_trades_simple("BUY", SYMBOL, backdays=1)
        self.assertEqual(len(deduped), 2)

        # 5. bapi_allorders.get_trade_orders and bapi_trades.get_trade_orders emit both timestamp/time and qty/quantity
        now_ms = int(time.time() * 1000)
        import threading
        import types
        import cacheManager as cm_mod
        fake_order_mgr = types.SimpleNamespace(
            lock=threading.Lock(),
            cache={SYMBOL: [
                {"orderId": 77, "symbol": SYMBOL, "side": "BUY", "price": "105.0", "executedQty": "1.5", "updateTime": now_ms - 5000}
            ]},
        )
        with mock.patch.object(cm_mod, "get_cache_manager", return_value=fake_order_mgr):
            all_orders_res = bapi_all.get_trade_orders("BUY", SYMBOL, 60)
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
        with mock.patch.object(bapi_tr, "init_cache_trade_manager", return_value=fake_trade_mgr):
            tr_orders_res = bapi_tr.get_trade_orders("SELL", SYMBOL, 60)
        self.assertEqual(len(tr_orders_res), 1)
        self.assertEqual(tr_orders_res[0]["time"], now_ms - 5000)
        self.assertEqual(tr_orders_res[0]["timestamp"], now_ms - 5000)
        self.assertEqual(tr_orders_res[0]["qty"], 2.5)
        self.assertEqual(tr_orders_res[0]["quantity"], 2.5)
        self.assertFalse(tr_orders_res[0]["isBuyer"])

        # 6. bapi.cancel_expired_orders and cancel_recent_orders handle millisecond timestamps properly
        expired_ms = int((time.time() - 7200) * 1000)
        fresh_ms = int((time.time() - 60) * 1000)
        open_orders_map = {
            1: {"orderId": 1, "symbol": SYMBOL, "timestamp": expired_ms},
            2: {"orderId": 2, "symbol": SYMBOL, "time": fresh_ms},
        }
        with (
            mock.patch.object(bapi_mod, "get_open_orders", return_value=open_orders_map),
            mock.patch.object(bapi_mod, "cancel_order", return_value=True) as mock_cancel,
        ):
            bapi_mod.cancel_expired_orders("BUY", SYMBOL, 3600)
            mock_cancel.assert_called_once_with(SYMBOL, 1)
            mock_cancel.reset_mock()
            bapi_mod.cancel_recent_orders("BUY", SYMBOL, 3600)
            mock_cancel.assert_called_once_with(SYMBOL, 2)

        # 7. extract_timestamp_s, extract_remaining_qty, and order_rollback.extract_order_restore_params
        from providers.base import extract_remaining_qty, extract_timestamp_s
        from order_rollback import extract_order_restore_params
        import monitortrades as mt_mod

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

        # 8. order_guard.daily_limit_guard enforces symmetric 24h daily cap and 5s anti-spam
        now_s = int(time.time())
        fake_provider = types.SimpleNamespace(
            name="binance",
            get_orders=lambda sym, side, window: [
                {"price": 100.0, "qty": 1.0, "time": now_s - 2},
            ],
        )
        ok, reason = po.order_guard.daily_limit_guard(
            fake_provider, SYMBOL, "BUY", max_daily_trades=10,
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
        from providers.base import extract_order_price
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
        import json
        from binance_api import market_data as b_md
        import utils as u_mod

        page1 = [[1_700_000_000_000 + i * 60_000, "100", "101", "99", "100", "10", 1_700_000_059_999 + i * 60_000] for i in range(1000)]
        page2 = [[1_700_000_000_000 + (1000 + i) * 60_000, "100", "101", "99", "100", "10", 1_700_000_059_999 + (1000 + i) * 60_000] for i in range(200)]
        calls = []

        class _FakeResp:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_urlopen(req, timeout=20):
            calls.append(req.full_url)
            return _FakeResp(page1 if len(calls) == 1 else page2)

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

        # 12. utils.exponential_decrease and asymptotic_decrease handle zero expired_duration safely
        self.assertEqual(u_mod.exponential_decrease(1.5, 0, 5), 1.5)
        self.assertEqual(u_mod.asymptotic_decrease(1.5, 0, 5), 1.5)
        self.assertEqual(u_mod.exponential_decrease(1.5, 60, 5, half_life_duration=0), 0.0)

        # 13. bapi.update_price and get_free_balance reject boolean/NaN/negative payloads
        fake_client = MagicMock()
        fake_client.get_symbol_ticker.return_value = {"price": True}
        fake_client.get_asset_balance.return_value = {"free": True}
        with mock.patch.object(bapi_mod, "client", fake_client):
            self.assertIsNone(bapi_mod.update_price(SYMBOL))
            self.assertIsNone(bapi_mod.get_free_balance("USDC"))
            fake_client.get_asset_balance.return_value = {"free": "-5.0"}
            self.assertIsNone(bapi_mod.get_free_balance("USDC"))

        # 14. compute_24h_price_metrics normalizes Unix-second timestamps
        from intelligence.sentiment.guards.high_stake_guard import compute_24h_price_metrics
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

        # 15. liquidation_signal and BybitLiquidationCollector normalize symbol case and reject NaN/non-positive notionals
        import liquidation_signal as liq_mod
        from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector
        liq_col = liq_mod.LiquidationSignal(symbols=["BTCUSDT"])
        liq_col._ingest(json.dumps({
            "topic": "allLiquidation.BTCUSDT",
            "data": [
                {"s": "btcusdt", "S": "Buy", "v": "1.5", "p": "100.0"},
                {"s": "BTCUSDT", "S": "Buy", "v": "nan", "p": "100.0"},
                {"s": "BTCUSDT", "S": "Buy", "v": "-1.0", "p": "100.0"},
            ],
        }))
        sig = liq_col.get_signal("BTCUSDT")
        self.assertEqual(sig["long_liq_usd"], 150.0)

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

        # 16. symbols.validate_params rejects booleans, NaN, and inf; bapi.get_account_assets_balances rejects non-finite/negative/bool
        import symbols as sym_mod
        for bad_val in (True, False, float("nan"), float("inf"), float("-inf"), 0, -1):
            with self.assertRaises(ValueError):
                sym_mod.validate_params("BUY", SYMBOL, price=bad_val, qty=1.0)
            with self.assertRaises(ValueError):
                sym_mod.validate_params("BUY", SYMBOL, price=100.0, qty=bad_val)

        fake_client.get_account.return_value = {"balances": [{"asset": "BTC", "free": True, "locked": "0.0"}]}
        with mock.patch.object(bapi_mod, "client", fake_client):
            self.assertEqual(bapi_mod.get_account_assets_balances(), [])
            fake_client.get_account.return_value = {"balances": [{"asset": "BTC", "free": "nan", "locked": "0.0"}]}
            self.assertEqual(bapi_mod.get_account_assets_balances(), [])
            fake_client.get_account.return_value = {"balances": [{"asset": "BTC", "free": "-1.0", "locked": "0.0"}]}
            self.assertEqual(bapi_mod.get_account_assets_balances(), [])

        with mock.patch.object(po.api, "get_current_price", return_value=float("inf")):
            self.assertEqual(
                po.adjust_price_and_cancel_opposite("SELL", SYMBOL, 100.0, cancel_opposite=False),
                100.0,
            )
            self.assertEqual(
                po.order_candidate_price("BUY", 100.0, True),
                100.0,
            )

        # 17. symbols.get_quantity_precision returns positive decimal places for sub-unit stepSize
        fake_sym_client = MagicMock()
        fake_sym_client.get_symbol_info.return_value = {
            "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.00100000"}]
        }
        with mock.patch.object(sym_mod, "_get_client", return_value=fake_sym_client):
            self.assertEqual(sym_mod.get_quantity_precision(SYMBOL), 3)
            fake_sym_client.get_symbol_info.return_value = {
                "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.0"}]
            }
            self.assertEqual(sym_mod.get_quantity_precision(SYMBOL), 8)

        # 18. _convert_to_usdc supports inverse USDCUSDT quote and stablecoin peg fallback
        with mock.patch.object(bapi_mod, "_get_symbol_price_safe", side_effect=lambda s: 1.001 if s == "USDCUSDT" else None):
            self.assertAlmostEqual(bapi_mod._convert_to_usdc("USDT", 100.1), 100.0, places=4)
            self.assertEqual(bapi_mod._convert_to_usdc("FDUSD", 50.0), 50.0)
            self.assertEqual(bapi_mod._convert_to_usdc("USD", 25.0), 25.0)
        self.assertEqual(bapi_mod._normalize_side("long"), "BUY")
        self.assertEqual(bapi_mod._normalize_side("short"), "SELL")



if __name__ == "__main__":
    unittest.main()
