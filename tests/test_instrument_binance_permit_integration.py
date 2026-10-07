"""End-to-end regressions for Instrument's Binance submit-permit plumbing."""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_api import bapi_placeorder as placeorder
from binance_cache_health import CacheHealthStatus
from instrument import Instrument
from lock import trade_cooldown
import instrument as instrument_module
import order_guard
import order_retry
import order_retry_worker
from providers import market_api
from providers.market_api import BinanceProvider, MarketApi


class _FakeClient:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def get_symbol_info(self, symbol):
        return {
            "baseAsset": "BTC",
            "quoteAsset": "USDC",
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "tickSize": "0.01",
                    "minPrice": "0.01",
                    "maxPrice": "1000000",
                },
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "0.001",
                    "minQty": "0.001",
                    "maxQty": "1000",
                },
            ],
        }

    def order_limit_buy(self, **kwargs):
        self.events.append("submit_buy")
        self.calls.append(("BUY", kwargs))
        return {"orderId": 501, "status": "NEW"}

    def order_limit_sell(self, **kwargs):
        self.events.append("submit_sell")
        self.calls.append(("SELL", kwargs))
        return {"orderId": 502, "status": "NEW"}


class _FakeBapi:
    def __init__(self, client, events):
        self.client = client
        self.events = events
        self.canceled = False

    def get_current_price(self, symbol):
        return 100.0

    def get_free_balance(self, asset):
        return 1_000.0 if asset == "USDC" else 10.0

    def get_open_orders(self, order_type, symbol, *, strict=False):
        price = 90.0 if order_type == "SELL" else 110.0
        return {"old-order": {"price": price}}

    def cancel_order(self, symbol, order_id):
        self.events.append("cancel")
        self.canceled = True
        return True


class InstrumentBinancePermitIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_state = trade_cooldown.STATE_FILE
        self.old_lock = trade_cooldown.LOCK_FILE
        trade_cooldown.STATE_FILE = os.path.join(
            self.temp.name, "cooldown.json")
        trade_cooldown.LOCK_FILE = os.path.join(
            self.temp.name, "cooldown.lock")
        self.old_retry_queue = order_retry.QUEUE_FILE
        self.old_retry_lock = order_retry.LOCK_FILE
        self.old_retry_enabled = order_retry.RETRY_ENABLED
        self.old_retry_dedup = order_retry.RETRY_DEDUP
        order_retry.QUEUE_FILE = os.path.join(
            self.temp.name, "order_retry_queue.jsonl")
        order_retry.LOCK_FILE = os.path.join(
            self.temp.name, "order_retry_queue.lock")
        order_retry.RETRY_ENABLED = True
        order_retry.RETRY_DEDUP = True

    def tearDown(self):
        trade_cooldown.STATE_FILE = self.old_state
        trade_cooldown.LOCK_FILE = self.old_lock
        order_retry.QUEUE_FILE = self.old_retry_queue
        order_retry.LOCK_FILE = self.old_retry_lock
        order_retry.RETRY_ENABLED = self.old_retry_enabled
        order_retry.RETRY_DEDUP = self.old_retry_dedup
        self.temp.cleanup()

    def _place(self, side, qty, *, smart, legacy_safe_adapter=False,
               expire_permit_before_dispatch=False):
        events = []
        client = _FakeClient(events)
        fake_bapi = _FakeBapi(client, events)
        provider = BinanceProvider()
        api = MarketApi([provider])
        instrument = Instrument(
            name="BTC", symbol="BTCUSDC", provider="Binance",
            base="BTC", quote="USDC", api=api)
        status = CacheHealthStatus(
            ready=True, reason="ok", order_age_sec=1.0,
            trade_age_sec=1.0,
            order_cache_version="orders-v1",
            trade_cache_version="trades-v1",
        )
        health_calls = []

        def health_gate():
            if fake_bapi.canceled:
                raise AssertionError(
                    "a valid permit must not reread cache health after cancel")
            health_calls.append("health")
            return status

        real_issue = placeorder.issue_account_cache_submit_permit
        issued = []

        def issue(*args, **kwargs):
            permit = real_issue(*args, **kwargs)
            issued.append(permit)
            return permit

        real_consume = placeorder._consume_account_cache_submit_permit

        def consume(*args, **kwargs):
            if not expire_permit_before_dispatch:
                return real_consume(*args, **kwargs)
            permit = args[0]
            with patch.object(
                    placeorder.time, "monotonic",
                    return_value=permit.expires_at + 1.0):
                return real_consume(*args, **kwargs)

        def routed_place(symbol, order_side, order_price, order_qty, **kwargs):
            self.assertEqual(symbol, "BTCUSDC")
            return instrument.place(
                order_side, order_price, order_qty, **kwargs)

        original_price = 105.0 if side == "BUY" else 95.0
        with (
            patch.object(market_api, "_bapi", fake_bapi),
            patch.object(placeorder, "api", fake_bapi),
            patch.object(placeorder, "client", client),
            patch.object(
                placeorder, "require_account_cache_for_submit",
                side_effect=health_gate),
            patch.object(
                placeorder, "issue_account_cache_submit_permit",
                side_effect=issue),
            patch.object(
                placeorder, "_consume_account_cache_submit_permit",
                side_effect=consume),
            patch.object(
                placeorder.cfg, "is_trade_enabled", return_value=True),
            patch.object(
                provider, "profit_guard_window_ref", return_value=None),
            patch.object(
                provider, "policy_cap_quantity",
                side_effect=lambda _symbol, _side, _price, _qty,
                                   available_qty, **_kwargs:
                    min(2.0, available_qty)),
            patch.object(
                provider, "fee_cap_quantity",
                side_effect=lambda _symbol, _side, _price,
                                   available_qty: available_qty),
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(order_guard, "profit_guard", return_value=True),
            # Isolate the placement from the live short-trend cache: the legacy
            # place_safe_order path does not pass wait_for_trend=False, so without
            # this it defers on whatever the production trend state happens to be.
            patch("cacheManager.get_short_trend_manager",
                  return_value=Mock(should_wait=Mock(return_value=False))),
            patch.object(
                instrument_module._outcomes_log, "log_order_outcome"),
            patch.object(
                market_api.api, "place", side_effect=routed_place),
            patch.object(
                placeorder, "_submit_binance_order",
                wraps=placeorder._submit_binance_order) as low_level,
        ):
            if legacy_safe_adapter:
                result = placeorder.place_safe_order(
                    side, "BTCUSDC", original_price, qty=qty)
            else:
                result = instrument.place(
                    side, original_price, qty, smart=smart,
                    wait_for_trend=False, caller_owns_retry=True,
                    kind=f"integration_{side.lower()}")

        return {
            "result": result,
            "events": events,
            "client": client,
            "issued": issued,
            "low_level": low_level,
            "health_calls": health_calls,
        }

    def test_legacy_safe_adapter_preserves_qty_none_until_finite_decision(self):
        observed = self._place(
            "BUY", None, smart=False, legacy_safe_adapter=True)

        self.assertEqual(observed["result"]["orderId"], 501)
        self.assertEqual(observed["low_level"].call_args.args[2], 2.0)
        self.assertEqual(observed["client"].calls[0][1]["quantity"], 2.0)
        self.assertEqual(len(observed["issued"]), 1)
        self.assertIs(
            observed["low_level"].call_args.kwargs["cache_permit"],
            observed["issued"][0])

    def test_smart_buy_uses_one_permit_across_cancel_and_submit(self):
        observed = self._place("BUY", 2.0, smart=True)

        self.assertEqual(observed["events"], ["cancel", "submit_buy"])
        self.assertEqual(observed["result"]["orderId"], 501)
        self.assertEqual(
            observed["low_level"].call_args.kwargs[
                "permit_requested_price"],
            99.9)
        self.assertIs(
            observed["low_level"].call_args.kwargs["cache_permit"],
            observed["issued"][0])

    def test_smart_sell_uses_one_permit_across_cancel_and_submit(self):
        observed = self._place("SELL", 2.0, smart=True)

        self.assertEqual(observed["events"], ["cancel", "submit_sell"])
        self.assertEqual(observed["result"]["orderId"], 502)
        self.assertEqual(
            observed["low_level"].call_args.kwargs[
                "permit_requested_price"],
            100.1)
        self.assertIs(
            observed["low_level"].call_args.kwargs["cache_permit"],
            observed["issued"][0])

    def test_expired_permit_refuses_before_smart_cancellation(self):
        observed = self._place(
            "BUY", 2.0, smart=True,
            expire_permit_before_dispatch=True)

        self.assertIsNone(observed["result"])
        self.assertEqual(observed["events"], [])
        self.assertEqual(len(observed["issued"]), 1)

    def test_disabled_binance_gate_leaves_no_future_live_intent(self):
        events = []
        client = _FakeClient(events)
        fake_bapi = _FakeBapi(client, events)
        provider = BinanceProvider()
        api = MarketApi([provider])
        instrument = Instrument(
            name="BTC", symbol="BTCUSDC", provider="Binance",
            base="BTC", quote="USDC", api=api)

        with (
            patch.object(market_api, "_bapi", fake_bapi),
            patch.object(placeorder.cfg, "is_trade_enabled", return_value=False),
            patch.object(provider, "place_order", wraps=provider.place_order) as submit,
        ):
            result = instrument.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False)

        self.assertIsNone(result)
        self.assertEqual(order_retry.load_all(), [])
        submit.assert_not_called()

        order_retry.enqueue(
            "BTCUSDC", "BUY", 1.0, {"smart": False},
            requested_price=100.0, failure_reason="execution_disabled",
            provider_name="Binance", now=1000.0)
        with (
            patch.object(market_api, "_bapi", fake_bapi),
            patch.object(placeorder.cfg, "is_trade_enabled", return_value=False),
            patch.object(provider, "place_order", wraps=provider.place_order) as retry_submit,
        ):
            stats = order_retry_worker.process_once(api, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(order_retry.load_all(), [])
        retry_submit.assert_not_called()

        with (
            patch.object(market_api, "_bapi", fake_bapi),
            patch.object(placeorder.cfg, "is_trade_enabled", return_value=True),
            patch.object(provider, "place_order", wraps=provider.place_order) as later_submit,
        ):
            later = order_retry_worker.process_once(api, now=1800.0)
        self.assertEqual(later["attempted"], 0)
        later_submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
