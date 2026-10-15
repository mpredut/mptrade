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
            hours=0.5, bypass_profit_guard=True,
        )

    assert result == expected
    place.assert_called_once_with(
        TAO_SYMBOL, "BUY", 225.0, 2.0,
        smart=False, safeback_seconds=123, force=True,
        cancelorders=True, hours=0.5, bypass_profit_guard=True,
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
                bypass_profit_guard=True,
            )

    def test_no_trades_allowed(self):
        ok, reason = self._run(same_side_trades=[])
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_daily_limit_blocks(self):
        # backdays = math.ceil((48h+60s)/86400) = 3 -> threshold 25*3=75; 90 exceeds it.
        ok, reason = self._run(same_side_trades=_trades(90, age_sec=4000.0))
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_limit")

    def test_recent_transaction_blocks(self):
        ok, reason = self._run(same_side_trades=_trades(1, age_sec=5.0))
        self.assertFalse(ok)
        self.assertEqual(reason, "recent_transaction")

    def test_recent_transaction_does_not_block_sell(self):
        ok, reason = self._run(same_side_trades=_trades(1, side="SELL", age_sec=5.0), side="SELL")
        self.assertTrue(ok)
        self.assertIsNone(reason)


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
            mock.patch.object(po.order_guard, "window_for", return_value=8 * 3600.0) as window_for,
            mock.patch.object(po.order_guard, "margin_for", return_value=1.15),
            mock.patch.object(po.order_guard, "profit_guard", return_value=True) as profit_guard,
            mock.patch("providers.market_api.api.market_regime_context", return_value=resolved_context) as resolve_context,
        ):
            result = po.if_place_safe_order(
                "BUY", SYMBOL, 100.0, 1.0, 48 * 3600 + 60, **kwargs,
            )
        return result, resolved_context, resolve_context, window_for, profit_guard

    def test_explicit_context_is_reused_without_runtime_resolution(self):
        explicit_context = self._fresh_context()
        result, _, resolve_context, window_for, profit_guard = self._exercise_buy(
            regime_context=explicit_context, pass_context=True,
        )
        self.assertEqual(result, (True, None))
        resolve_context.assert_not_called()
        self.assertIs(window_for.call_args.kwargs["regime_context"], explicit_context)
        self.assertIs(profit_guard.call_args.kwargs["regime_context"], explicit_context)

    def test_missing_context_is_resolved_once_at_market_api_boundary(self):
        result, resolved_context, resolve_context, window_for, profit_guard = self._exercise_buy()
        self.assertEqual(result, (True, None))
        self.assertTrue(resolved_context.fresh)
        resolve_context.assert_called_once_with(SYMBOL, provider_name="binance")
        self.assertIs(window_for.call_args.kwargs["regime_context"], resolved_context)
        self.assertIs(profit_guard.call_args.kwargs["regime_context"], resolved_context)


if __name__ == "__main__":
    unittest.main()
