"""The StrategyExecutor contract for Trading212, without network or real keys."""

import os
import sys
import time
import unittest
from unittest import mock
import importlib
import pytest
import tempfile
from cachepaths import canonical_base_asset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T212_DIR = os.path.join(ROOT, "212trading")
if T212_DIR not in sys.path:
    sys.path.insert(0, T212_DIR)

import t212_client  # noqa: E402

from providers.strategy_executor import (
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    StrategyExecutor,
)
from providers.t212_provider import T212Provider
from botcore import load_local_module
from tests.conftest import load_venue_modules
from types import SimpleNamespace
from cacheaccount import CacheTradeManager


class FakeT212Client:
    def __init__(self):
        self.calls = []
        self.portfolio_result = [{
            "ticker": "NVDA_US_EQ", "quantity": 1.25,
            "averagePrice": 100.0, "currentPrice": 120.0,
        }]
        self.place_result = (200, {"id": 712, "status": "NEW"})
        self.status_result = {
            "id": 712,
            "ticker": "NVDA_US_EQ",
            "status": "PARTIALLY_FILLED",
            "filledQuantity": 0.4,
            "filledValue": 48.0,
        }
        self.cancel_result = True
        self.active_orders_result = []

    def get_portfolio(self):
        return self.portfolio_result

    def list_active_orders(self):
        return self.active_orders_result

    def place_limit_order(self, ticker, quantity, limit_price, validity="DAY"):
        self.calls.append(("limit", ticker, quantity, limit_price, validity))
        return self.place_result

    def place_market_order(self, ticker, quantity, extended_hours=False):
        self.calls.append(("market", ticker, quantity, extended_hours))
        return self.place_result

    def get_order_status(self, order_id):
        self.calls.append(("status", order_id))
        return self.status_result

    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))
        return self.cancel_result


class T212ExecutorContractTest(unittest.TestCase):
    def setUp(self):
        self.previous_live = os.environ.get("T212_LIVE_ORDERS")
        os.environ["T212_LIVE_ORDERS"] = "true"
        self.fake = FakeT212Client()
        self.provider = T212Provider(order_validity="DAY")
        self.provider._cli = self.fake

    def tearDown(self):
        if self.previous_live is None:
            os.environ.pop("T212_LIVE_ORDERS", None)
        else:
            os.environ["T212_LIVE_ORDERS"] = self.previous_live

    def test_the_contract_balance_and_precision(self):
        self.assertIsInstance(self.provider, StrategyExecutor)
        self.assertEqual(self.provider.free_balance("NVDA_US_EQ"), 1.25)
        self.assertEqual(
            self.provider.pair_precision("NVDA_US_EQ"),
            PairPrecision(2, 2, 0.01, "NVDA_US_EQ"),
        )
        self.assertEqual(self.provider.ohlc_closes("NVDA_US_EQ", 240), [])

        self.fake.portfolio_result = None
        self.assertIsNone(self.provider.free_balance("NVDA_US_EQ"))

    def test_get_current_price_portfolio_and_fallback(self):
        # 1. Position held in portfolio: returns currentPrice (120.0)
        self.assertEqual(self.provider.get_current_price("NVDA_US_EQ"), 120.0)

        # 2. Position NOT held in portfolio: fallback to get_price_usd via t212_to_yahoo
        self.fake.portfolio_result = []
        md = importlib.import_module("212trading.market_data")
        with mock.patch.object(md, "get_price_usd", return_value=125.50) as mock_feed:
            price = self.provider.get_current_price("NVDA_US_EQ")
            self.assertEqual(price, 125.50)
            mock_feed.assert_called_once_with("NVDA")

    def test_submit_order_behaviors(self):
        with self.subTest(msg="submit_limit_market_and_the_live_gate"):
            oid = self.provider.submit_order("NVDA_US_EQ", "buy", 0.5, price=119.25)
            self.assertEqual(oid, "712")
            self.assertEqual(self.fake.calls[-1], ("limit", "NVDA_US_EQ", 0.5, 119.25, "DAY"))

            self.provider.submit_order("NVDA_US_EQ", "sell", 0.25, market=True)
            self.assertEqual(self.fake.calls[-1], ("market", "NVDA_US_EQ", -0.25, False))

            os.environ["T212_LIVE_ORDERS"] = "false"
            with self.assertRaisesRegex(ProviderError, "blocked"):
                self.provider.submit_order("NVDA_US_EQ", "buy", 0.5, price=119.25)

            os.environ["T212_LIVE_ORDERS"] = "true"
            with self.assertRaisesRegex(ProviderError, "quantity"):
                self.provider.submit_order("NVDA_US_EQ", "buy", 0.001, price=119.25)

        with self.subTest(msg="the_live_gate_can_be_injected_by_the_standalone_launcher"):
            os.environ["T212_LIVE_ORDERS"] = "false"
            explicit_live = T212Provider(
                client=self.fake, live_enabled=True, order_validity="DAY",
            )
            self.assertEqual(
                explicit_live.submit_order("NVDA_US_EQ", "buy", 0.5, price=119.25),
                "712",
            )

            os.environ["T212_LIVE_ORDERS"] = "true"
            explicit_paper = T212Provider(client=self.fake, live_enabled=False)
            with self.assertRaisesRegex(ProviderError, "blocked"):
                explicit_paper.submit_order("NVDA_US_EQ", "buy", 0.5, price=119.25)

        with self.subTest(msg="the_limit_validity_can_be_injected_by_the_profile"):
            provider = T212Provider(
                client=self.fake, live_enabled=True,
                order_validity="GOOD_TILL_CANCEL",
            )

            provider.submit_order("NVDA_US_EQ", "buy", 0.5, price=119.25)

            self.assertEqual(self.fake.calls[-1][-1], "GOOD_TILL_CANCEL")

        with self.subTest(msg="a_rejected_submit_or_one_without_an_id_is_an_error"):
            for result in ((429, {"error": "rate limit"}), (200, {"status": "NEW"})):
                with self.subTest(result=result):
                    self.fake.place_result = result
                    with self.assertRaises(ProviderError):
                        self.provider.submit_order("NVDA_US_EQ", "buy", 0.5, price=119.25)

    def test_status_partial_terminal_and_fail_closed(self):
        self.assertEqual(
            self.provider.order_status("NVDA_US_EQ", "712"),
            OrderStatus(
                "open", filled_qty=0.4, cost=48.0, fee=0.0,
                venue_status="PARTIALLY_FILLED"),
        )

        cases = [
            ("FILLED", "closed"),
            ("CANCELLED", "canceled"),
            ("REJECTED", "expired"),
        ]
        for venue_status, expected in cases:
            with self.subTest(status=venue_status):
                self.fake.status_result = {
                    "ticker": "NVDA_US_EQ", "status": venue_status,
                    "filledQuantity": 1.0, "filledValue": 120.0,
                }
                observed = self.provider.order_status("NVDA_US_EQ", "712")
                self.assertEqual(observed.status, expected)
                self.assertEqual(observed.venue_status, venue_status)

        with self.subTest(msg="negative_ordered_quantity_for_sell_is_normalized"):
            self.fake.status_result = {
                "ticker": "nvda_us_eq", "status": "FILLED",
                "filledQuantity": -2.83, "filledValue": 339.6,
                "orderedQuantity": -2.83,
            }
            observed = self.provider.order_status("NVDA_US_EQ", "712")
            self.assertEqual(observed.status, "closed")
            self.assertEqual(observed.filled_qty, 2.83)
            self.assertEqual(observed.orig_qty, 2.83)
            observed_short = self.provider.order_status("NVDA", "712")
            self.assertEqual(observed_short.status, "closed")
            with self.assertRaisesRegex(ProviderError, "ticker"):
                self.provider.order_status("SPCX_US_EQ", "712")

        self.fake.status_result = {
            "ticker": "NVDA_US_EQ", "status": "FILLED", "filledQuantity": 1.0,
        }
        with self.assertRaisesRegex(ProviderError, "without an executed cost"):
            self.provider.order_status("NVDA_US_EQ", "712")

        self.fake.status_result = {"ticker": "NVDA_US_EQ", "status": "MYSTERY"}
        with self.assertRaisesRegex(ProviderError, "unknown"):
            self.provider.order_status("NVDA_US_EQ", "712")

        self.fake.status_result = {
            "ticker": "NVDA_US_EQ", "status": "FILLED", "currency": "USD",
            "filledQuantity": 1.0, "filledValue": 120.0, "fee": 0.2,
            "_feeCurrencies": ["RON"],
        }
        with self.assertRaisesRegex(ProviderError, "order currency"):
            self.provider.order_status("NVDA_US_EQ", "712")

    def test_cancel_confirmed_idempotent_terminal_and_unconfirmed(self):
        self.provider.cancel_order("NVDA_US_EQ", "712")
        self.assertEqual(self.fake.calls[-1], ("cancel", "712"))

        self.fake.cancel_result = False
        self.fake.status_result = {
            "ticker": "NVDA_US_EQ", "status": "FILLED",
            "filledQuantity": 1.0, "filledValue": 120.0,
        }
        self.provider.cancel_order("NVDA_US_EQ", "712")

        self.fake.status_result = {
            "ticker": "NVDA_US_EQ", "status": "NEW",
            "filledQuantity": 0.0, "filledValue": 0.0,
        }
        with self.assertRaisesRegex(ProviderError, "still open"):
            self.provider.cancel_order("NVDA_US_EQ", "712")

    def test_free_balance_deducts_active_sell_orders(self):
        self.fake.portfolio_result = [{
            "ticker": "RGNT_US_EQ",
            "quantity": 90.23,
            "averagePrice": 1.20,
            "currentPrice": 1.38,
        }]
        # 1. No active orders: full position is available
        self.assertEqual(self.provider.free_balance("RGNT_US_EQ"), 90.23)

        # 2. Active sell order committed on exchange: deducted from free balance
        self.fake.active_orders_result = [{
            "id": 59857476538,
            "ticker": "RGNT_US_EQ",
            "quantity": -72.46,
            "filledQuantity": 0.0,
            "side": "SELL",
            "status": "NEW",
        }]
        self.assertEqual(self.provider.free_balance("RGNT_US_EQ"), 17.77)

        # 3. Partially executed sell order: only unfilled remaining qty is deducted
        self.fake.active_orders_result[0]["filledQuantity"] = -20.0
        self.assertEqual(self.provider.free_balance("RGNT_US_EQ"), 37.77)

        # 4. Unrelated active BUY order on same ticker: doesn't deduct stock balance
        self.fake.active_orders_result.append({
            "id": 59857476539,
            "ticker": "RGNT_US_EQ",
            "quantity": 10.0,
            "filledQuantity": 0.0,
            "side": "BUY",
            "status": "NEW",
        })
        self.assertEqual(self.provider.free_balance("RGNT_US_EQ"), 37.77)

    def test_open_orders_resolves_nested_instrument_ticker_and_skips_terminal_statuses(self):
        self.fake.portfolio_result = [
            {"ticker": "RGNT_US_EQ", "quantity": 100.0, "averagePrice": 1.20, "currentPrice": 1.38}
        ]
        self.fake.active_orders_result = [
            {
                "id": 901,
                "instrument": {"ticker": "RGNT_US_EQ"},
                "orderedQuantity": -30.0,
                "filledQuantity": 0.0,
                "limitPrice": 1.45,
                "status": "WORKING",
            },
            {
                "id": 902,
                "instrument": {"ticker": "RGNT_US_EQ"},
                "orderedQuantity": -20.0,
                "filledQuantity": 20.0,
                "limitPrice": 1.40,
                "status": "FILLED",
            },
            {
                "id": 903,
                "ticker": "RGNT_US_EQ",
                "orderedQuantity": -15.0,
                "filledQuantity": 0.0,
                "limitPrice": 1.50,
                "status": "CANCELLED",
            },
        ]
        open_ords = self.provider.open_orders("RGNT_US_EQ")
        self.assertEqual([o["id"] for o in open_ords], ["901"])
        self.assertEqual(open_ords[0]["side"], "SELL")
        self.assertEqual(open_ords[0]["origQty"], 30.0)
        self.assertEqual(self.provider.free_balance("RGNT_US_EQ"), 70.0)

    def test_place_order_raises_submission_refused_on_definitive_error(self):
        # 400 Bad Request (e.g. InsufficientQuantity) is a definitive rejection
        self.fake.portfolio_result = [{"ticker": "RGNT_US_EQ", "quantity": 100.0, "averagePrice": 1.20, "currentPrice": 1.38}]
        self.fake.place_result = (400, {"code": "InsufficientQuantity", "message": "Not enough shares"})
        with self.assertRaises(SubmissionRefused) as ctx:
            self.provider.place_order("RGNT_US_EQ", "SELL", 1.38, 72.46)
        self.assertIn("HTTP 400", str(ctx.exception))

        # Instrument integration verification: definitive rejection marks state as 'refused'
        from instrument import Instrument
        from providers.market_api import MarketApi
        import order_guard
        inst = Instrument("RGNT", "RGNT_US_EQ", "t212", api=MarketApi([self.provider]))
        outcome = {}
        with mock.patch.object(order_guard, "daily_limit_guard", return_value=(True, None)):
            placed = inst.place(
                "SELL", 1.38, 72.46, smart=False, wait_for_trend=False,
                caller_owns_retry=True, _outcome_context=outcome,
            )
        self.assertIsNone(placed)
        self.assertEqual(outcome.get("state"), "refused")
        self.assertIn("HTTP 400", str(outcome.get("reason") or ""))

    def test_market_buy_retries_on_insufficient_free_cash_buffer(self):
        responses = [
            (400, {"type": "/api-errors/insufficient-free-for-stocks-buy", "detail": "Insufficient funds"}),
            (200, {"id": 9901, "status": "NEW"}),
        ]
        with mock.patch.object(self.fake, "place_market_order", side_effect=responses) as mock_mkt:
            res = self.provider.place_order("NVDA_US_EQ", "BUY", 201.73, 0.44, market=True)
        self.assertIsNotNone(res)
        self.assertEqual(res.get("orderId"), "9901")
        self.assertEqual(mock_mkt.call_count, 2)
        self.assertEqual(mock_mkt.call_args_list[0].args, ("NVDA_US_EQ", 0.44))
        self.assertEqual(mock_mkt.call_args_list[1].args, ("NVDA_US_EQ", 0.41))


class T212ClientTransportTest(unittest.TestCase):
    def test_the_market_payload_and_the_history_fallback(self):
        client = t212_client.T212Client(
            "dummy", "dummy", env="demo", min_gap_sec=0.0,
            portfolio_ttl_sec=6.0, order_min_gap_sec=0.0,
        )
        client._min_gap = 0.0
        client._pf_cache = (time.monotonic(), [{"ticker": "NVDA_US_EQ", "quantity": 1.0}])
        with mock.patch.object(
            t212_client, "http_post_json", return_value=(200, b'{"id": 55}')
        ) as post:
            status, data = client.place_market_order("NVDA_US_EQ", -0.25)
        self.assertIsNone(client._pf_cache)
        self.assertEqual((status, data), (200, {"id": 55}))
        self.assertTrue(post.call_args.args[0].endswith("/equity/orders/market"))
        self.assertEqual(
            post.call_args.kwargs["payload"],
            {"ticker": "NVDA_US_EQ", "quantity": -0.25, "extendedHours": False},
        )

        history = (
            b'{"items":[{"order":{"id":55,"ticker":"NVDA_US_EQ",'
            b'"status":"FILLED"},"fill":{"quantity":0.25,"price":120.0,'
            b'"walletImpact":{"taxes":[{"name":"CURRENCY_CONVERSION_FEE",'
            b'"quantity":0.045}]}}}]}'
        )
        with mock.patch.object(
            t212_client, "http_get", side_effect=[(404, b""), (200, history)]
        ):
            order = client.get_order_status(55)
        self.assertEqual(order["status"], "FILLED")
        self.assertEqual(order["filledQuantity"], 0.25)
        self.assertEqual(order["filledValue"], 30.0)
        self.assertEqual(order["fee"], 0.045)

    def test_rate_limit_backoff_and_stale_cache_fallback(self):
        client = t212_client.T212Client(
            "dummy", "dummy", env="demo", min_gap_sec=0.0,
            portfolio_ttl_sec=1.0, order_min_gap_sec=0.0,
        )
        client._min_gap = 0.0

        # Step 1: Successful initial fetch populates cache
        with mock.patch.object(
            t212_client, "http_get", return_value=(200, b'[{"ticker":"SPCX_US_EQ","quantity":2.92}]')
        ) as get_mock:
            pf = client.get_portfolio()
            self.assertEqual(len(pf), 1)
            self.assertEqual(get_mock.call_count, 1)

        # Step 2: Simulate TTL expiry and 429 rate limit
        client._pf_cache = (time.monotonic() - 2.0, pf)  # Expire TTL
        with mock.patch.object(
            t212_client, "http_get", return_value=(429, b"Too Many Requests")
        ) as get_mock:
            pf_stale = client.get_portfolio()
            # Stale cache returned rather than None
            self.assertEqual(len(pf_stale), 1)
            self.assertEqual(pf_stale[0]["ticker"], "SPCX_US_EQ")
            # Backoff until is set in the future
            self.assertGreater(client._backoff_until, time.monotonic())

            # Step 3: Second call within negative cache TTL does NOT call HTTP again
            pf_stale_2 = client.get_portfolio()
            self.assertEqual(len(pf_stale_2), 1)
            self.assertEqual(get_mock.call_count, 1)  # No second stampede!

    def test_order_pacing_and_cache_preservation_on_429(self):
        client = t212_client.T212Client(
            "dummy", "dummy", env="demo", min_gap_sec=0.0,
            portfolio_ttl_sec=6.0, order_min_gap_sec=2.1,
        )
        client._min_gap = 0.0

        # Prime the active orders cache
        cached_orders = [{"id": 1, "ticker": "SPCX_US_EQ"}]
        client._ord_cache = (time.monotonic(), cached_orders)

        sleep_calls = []
        with mock.patch("time.sleep", side_effect=sleep_calls.append), \
             mock.patch.object(t212_client, "http_post_json", return_value=(200, b'{"id": 100}')):
            # First order: no sleep needed
            status, _ = client.place_limit_order("SPCX_US_EQ", 0.5, 100.0)
            self.assertEqual(status, 200)
            self.assertEqual(len(sleep_calls), 0)
            # 200 OK invalidates the orders cache for fresh tracking
            self.assertIsNone(client._ord_cache)

            # Re-prime cache
            client._ord_cache = (time.monotonic(), cached_orders)

            # Second order placed immediately: should sleep ~2.1s
            status, _ = client.place_limit_order("SPCX_US_EQ", 0.5, 105.0)
            self.assertEqual(status, 200)
            self.assertEqual(len(sleep_calls), 1)
            self.assertAlmostEqual(sleep_calls[0], 2.1, delta=0.1)

        # Test 429 preserving cache
        client._ord_cache = (time.monotonic(), cached_orders)
        with mock.patch("time.sleep"), \
             mock.patch.object(t212_client, "http_post_json", return_value=(429, b'{"code": "BusinessException"}')):
            status, _ = client.place_limit_order("SPCX_US_EQ", 0.5, 110.0)
            self.assertEqual(status, 429)
            # On 429, the cache is preserved to avoid immediate cascading 429 on active orders lookup
            self.assertIsNotNone(client._ord_cache)
            self.assertEqual(client._ord_cache[1], cached_orders)

    def test_runtime_env_fallback_and_post_order_endpoint_pacing(self):
        import providers.t212_provider as t212_prov_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "runtime.env"), "w", encoding="utf-8") as fh:
                fh.write("T212_LIVE_ORDERS=true\nT212_ORDER_VALIDITY=DAY\n")
            fake = FakeT212Client()
            prov = T212Provider(client=fake)
            with mock.patch.object(t212_prov_mod, "_T212_DIR", tmpdir), \
                 mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("T212_LIVE_ORDERS", None)
                os.environ.pop("T212_ORDER_VALIDITY", None)
                self.assertTrue(prov.execution_enabled())
                oid = prov.submit_order("SPCX_US_EQ", "buy", 1.0, price=140.0)
                self.assertEqual(oid, "712")
                self.assertEqual(fake.calls[-1], ("limit", "SPCX_US_EQ", 1.0, 140.0, "DAY"))

        client = t212_client.T212Client(
            "dummy", "dummy", env="demo", min_gap_sec=0.0,
            portfolio_ttl_sec=6.0, order_min_gap_sec=2.1,
        )
        sleep_calls = []
        with mock.patch("time.sleep", side_effect=sleep_calls.append), \
             mock.patch.object(t212_client, "http_get", return_value=(200, b'{"free": 500.0}')), \
             mock.patch.object(t212_client, "http_post_json", return_value=(200, b'{"id": 101}')):
            self.assertEqual(client.get_cash(), {"free": 500.0})
            self.assertEqual(len(sleep_calls), 0)
            client.place_market_order("RGNT_US_EQ", 10.0)
            self.assertIsNone(client._cash_cache)
            # Immediate second get_cash() after cache invalidation must respect _order_min_gap
            self.assertEqual(client.get_cash(), {"free": 500.0})
            self.assertEqual(len(sleep_calls), 1)
            self.assertAlmostEqual(sleep_calls[0], 2.1, delta=0.2)

        # Verify get_historical_order caches the 50-item history page across multiple order lookups
        # and ignores 0-amount tax currencies so order_status does not fail on zero-fee currency mismatch
        history_payload = (
            b'{"items":['
            b'{"order":{"id":201,"ticker":"RGNT_US_EQ","status":"FILLED","currency":"USD"},'
            b'"fill":{"quantity":72.14,"price":1.43,"walletImpact":{"taxes":['
            b'{"quantity":0.0,"currency":"RON"},{"quantity":0.05,"currency":"USD"}]}}},'
            b'{"order":{"id":202,"ticker":"SPCX_US_EQ","status":"FILLED"},"fill":{"quantity":0.71,"price":140.51}}'
            b']}'
        )
        with mock.patch("time.sleep"), \
             mock.patch.object(
                 t212_client,
                 "http_get",
                 side_effect=[(404, b""), (200, history_payload), (404, b"")],
             ) as hist_get_mock:
            st1 = client.get_order_status(201)
            st2 = client.get_order_status(202)
            self.assertEqual(st1["status"], "FILLED")
            self.assertAlmostEqual(st1["filledQuantity"], 72.14)
            self.assertEqual(st1["_feeCurrencies"], ["USD"])
            self.assertEqual(st2["status"], "FILLED")
            self.assertAlmostEqual(st2["filledQuantity"], 0.71)
            # Only 3 HTTP GETs total: (404 on 201, 200 on history, 404 on 202 -> served from _hist_cache)
            self.assertEqual(hist_get_mock.call_count, 3)

        # Verify _pace(endpoint_key) does not double-sleep when both _min_gap and _order_min_gap are positive
        client_both_gaps = t212_client.T212Client(
            "dummy", "dummy", env="demo", min_gap_sec=1.2,
            portfolio_ttl_sec=6.0, order_min_gap_sec=2.1,
        )
        both_sleeps = []
        with mock.patch("time.sleep", side_effect=both_sleeps.append), \
             mock.patch.object(t212_client, "http_get", return_value=(200, b'{"free": 500.0}')), \
             mock.patch.object(t212_client, "http_post_json", return_value=(200, b'{"id": 101}')):
            self.assertEqual(client_both_gaps.get_cash(), {"free": 500.0})
            self.assertEqual(len(both_sleeps), 0)
            client_both_gaps.place_market_order("RGNT_US_EQ", 10.0)
            both_sleeps.clear()
            self.assertEqual(client_both_gaps.get_cash(), {"free": 500.0})
            self.assertEqual(len(both_sleeps), 1)
            self.assertAlmostEqual(both_sleeps[0], 2.1, delta=0.2)

    def test_list_instruments_shared_cache_and_429_negative_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = os.path.join(tmpdir, "shared_t212.json")
            c1 = t212_client.T212Client(
                "demo-key",
                "demo-secret",
                env="demo",
                min_gap_sec=0.0,
                portfolio_ttl_sec=6.0,
                order_min_gap_sec=0.0,
            )
            c1._shared_cache_path = shared_path
            c1._enable_shared_disk_cache = True
            calls = []

            def fake_get(url, headers=None):
                calls.append(url)
                return 200, b'[{"ticker": "NVDA_US_EQ", "isin": "US67066G1040"}]'

            with mock.patch.object(t212_client, "http_get", side_effect=fake_get), \
                 mock.patch.object(t212_client, "http_post_json", return_value=(200, b'{"id": 99}')):
                res1 = c1.list_instruments()
                self.assertEqual(res1, [{"ticker": "NVDA_US_EQ", "isin": "US67066G1040"}])
                self.assertEqual(len(calls), 1)
                # Placing an order invalidates order/portfolio caches but preserves _instr_cache on disk
                c1.place_market_order("NVDA_US_EQ", 1.0)

            c2 = t212_client.T212Client(
                "demo-key",
                "demo-secret",
                env="demo",
                min_gap_sec=0.0,
                portfolio_ttl_sec=6.0,
                order_min_gap_sec=0.0,
            )
            c2._shared_cache_path = shared_path
            c2._enable_shared_disk_cache = True
            with mock.patch.object(t212_client, "http_get", side_effect=AssertionError("should use shared disk cache")):
                res2 = c2.list_instruments()
                self.assertEqual(res2, [{"ticker": "NVDA_US_EQ", "isin": "US67066G1040"}])

            # Negative cache on 429 prevents multi-thread stampede when cold
            c3 = t212_client.T212Client(
                "demo-key",
                "demo-secret",
                env="demo",
                min_gap_sec=0.0,
                portfolio_ttl_sec=6.0,
                order_min_gap_sec=0.0,
            )
            c3._enable_shared_disk_cache = False
            calls_429 = []

            def fake_429(url, headers=None):
                calls_429.append(url)
                return 429, b'{"error": "rate limit"}'

            with mock.patch.object(t212_client, "http_get", side_effect=fake_429):
                self.assertIsNone(c3.list_instruments())
                self.assertIsNone(c3.list_instruments())
                self.assertEqual(len(calls_429), 1)

            # Verify _instr_cache lives in dedicated _instruments.json sidecar, stale .tmp files are cleaned,
            # and _hist_cache syncs FILLED orders into _trade_cache_path (cache_trade_t212.json)
            instr_sidecar = c1._instr_shared_cache_path()
            self.assertTrue(os.path.exists(instr_sidecar))
            from cachepaths import STALE_STAGING_MAX_AGE_SEC
            stale_tmp = os.path.join(tmpdir, ".shared_t212.json.orphan123.tmp")
            with open(stale_tmp, "w", encoding="utf-8") as fh:
                fh.write("{}")
            old_ts = time.time() - (STALE_STAGING_MAX_AGE_SEC + 60.0)
            os.utime(stale_tmp, (old_ts, old_ts))

            trade_cache_file = os.path.join(tmpdir, "cache_trade_t212.json")
            c1._trade_cache_path = trade_cache_file
            hist_body = (
                b'{"items":['
                b'{"order":{"id":777,"ticker":"SPCX_US_EQ","status":"FILLED","orderedQuantity":4.01,"type":"LIMIT"},'
                b'"fill":{"quantity":4.01,"price":156.47,"filledAt":"2026-10-13T15:37:40.000Z"}},'
                b'{"order":{"id":778,"ticker":"SPCX_US_EQ","status":"CANCELLED","orderedQuantity":-1.0,"limitPrice":180.0}}'
                b']}'
            )
            with mock.patch.object(t212_client, "http_get", side_effect=[(404, b""), (200, hist_body)]):
                ord_res = c1.get_order_status(777)
            self.assertEqual(ord_res["status"], "FILLED")
            self.assertFalse(os.path.exists(stale_tmp))
            self.assertTrue(os.path.exists(trade_cache_file))
            from state_io import read_json_dict
            saved_trades = read_json_dict(trade_cache_file, default={})
            self.assertIn("SPCX_US_EQ", saved_trades.get("items", {}))
            self.assertEqual(len(saved_trades["items"]["SPCX_US_EQ"]), 1)
            row = saved_trades["items"]["SPCX_US_EQ"][0]
            self.assertEqual(row["orderId"], "777")
            self.assertEqual(row["side"], "BUY")
            self.assertAlmostEqual(row["price"], 156.47)
            self.assertAlmostEqual(row["qty"], 4.01)

            latest_buy = c1.latest_filled_order_for_ticker("SPCX_US_EQ", side="BUY", refresh=False)
            self.assertIsNotNone(latest_buy)
            self.assertEqual(latest_buy["orderId"], "777")
            self.assertAlmostEqual(latest_buy["price"], 156.47)
            self.assertIsNone(c1.latest_filled_order_for_ticker("SPCX_US_EQ", side="SELL", refresh=False))

            # Verify T212Provider.get_orders reads real BUY/SELL fills from cache_trade_t212.json
            now_ms = int(time.time() * 1000)
            from state_io import save_json_state
            save_json_state(
                trade_cache_file,
                {
                    "items": {
                        "SPCX_US_EQ": [
                            {"orderId": "801", "symbol": "SPCX_US_EQ", "side": "BUY", "price": 138.0, "qty": 1.0, "timestamp": now_ms - 1800_000},
                            {"orderId": "802", "symbol": "SPCX_US_EQ", "side": "SELL", "price": 145.0, "qty": 0.5, "timestamp": now_ms - 900_000},
                        ]
                    },
                    "fetchtime": {"SPCX_US_EQ": now_ms},
                },
            )
            fake_prov = T212Provider(client=FakeT212Client())
            fake_prov._ssot_fills_cache_path = trade_cache_file
            buys = fake_prov.get_orders("SPCX_US_EQ", "BUY", 3600.0)
            sells = fake_prov.get_orders("SPCX_US_EQ", "SELL", 3600.0)
            self.assertEqual(len(buys), 1)
            self.assertEqual(buys[0]["orderId"], "801")
            self.assertAlmostEqual(buys[0]["price"], 138.0)
            self.assertEqual(len(sells), 1)
            self.assertEqual(sells[0]["orderId"], "802")
            self.assertAlmostEqual(sells[0]["price"], 145.0)
            # When window is narrower than cached fills, BUY falls back to synthetic portfolio averagePrice
            fake_prov._cli.portfolio_result = [{"ticker": "SPCX_US_EQ", "quantity": 2.0, "averagePrice": 140.5, "currentPrice": 142.0}]
            synth_buys = fake_prov.get_orders("SPCX_US_EQ", "BUY", 60.0)
            self.assertEqual(len(synth_buys), 1)
            self.assertAlmostEqual(synth_buys[0]["price"], 140.5)
            self.assertEqual(fake_prov.get_orders("SPCX_US_EQ", "SELL", 60.0), [])

            # Order lifecycle mutations invalidate fetchtime and trigger refresh_order_history on next get_orders
            fake_prov._live_enabled = True
            fake_prov._order_validity = "DAY"
            fake_prov.submit_order("SPCX_US_EQ", "BUY", 1.0, 140.0)
            invalidated = read_json_dict(trade_cache_file, default={})
            self.assertNotIn("SPCX_US_EQ", invalidated.get("fetchtime", {}))

            refresh_calls = []

            def _fake_refresh(*, max_age_sec=None):
                refresh_calls.append(max_age_sec)
                CacheTradeManager.merge_venue_fills_snapshot(
                    trade_cache_file,
                    {
                        "SPCX_US_EQ": [
                            {"orderId": "901", "symbol": "SPCX_US_EQ", "side": "BUY", "price": 140.0, "qty": 1.0, "timestamp": now_ms - 10_000},
                        ]
                    },
                )
                return []

            fake_prov._cli.refresh_order_history = _fake_refresh
            refreshed_buys = fake_prov.get_orders("SPCX_US_EQ", "BUY", 3600.0)
            self.assertEqual(refresh_calls, [0.0])
            self.assertEqual([o["orderId"] for o in refreshed_buys], ["801", "901"])

            # Verify 10-minute (600s) 429 shared-cache fallback and cross-process fail_ts backoff
            save_json_state(
                shared_path,
                {
                    "_ord_cache": {
                        "ts": time.time() - 300.0,
                        "data": [{"id": 501, "ticker": "SPCX_US_EQ", "quantity": 1.0, "limitPrice": 120.0}],
                    }
                },
            )
            c_p1 = t212_client.T212Client(
                "demo-key", "demo-secret", env="demo",
                min_gap_sec=0.0, portfolio_ttl_sec=6.0, order_min_gap_sec=0.0,
            )
            c_p1._shared_cache_path = shared_path
            c_p1._enable_shared_disk_cache = True
            c_p2 = t212_client.T212Client(
                "demo-key", "demo-secret", env="demo",
                min_gap_sec=0.0, portfolio_ttl_sec=6.0, order_min_gap_sec=0.0,
            )
            c_p2._shared_cache_path = shared_path
            c_p2._enable_shared_disk_cache = True
            p1_calls = []

            def fake_429_orders(url, headers=None):
                p1_calls.append(url)
                return 429, b'{"error": "rate limit"}'

            with mock.patch.object(t212_client, "http_get", side_effect=fake_429_orders):
                ords_p1 = c_p1.list_active_orders()
                self.assertEqual(len(ords_p1), 1)
                self.assertEqual(ords_p1[0]["id"], 501)
                self.assertEqual(len(p1_calls), 1)
                # Process 2 reads the fail_ts-stamped shared cache entry without making a duplicate 429 request
                ords_p2 = c_p2.list_active_orders()
                self.assertEqual(len(ords_p2), 1)
                self.assertEqual(ords_p2[0]["id"], 501)
                self.assertEqual(len(p1_calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_t212provider_get_orders_and_order_status_iter29_s2():
    """[ITER29 #2] T212Provider.get_orders and order_status."""
    # 2. T212Provider.get_orders and order_status
    fake_cli = SimpleNamespace(
        get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "averagePrice": 120.0, "quantity": 5.0}],
        get_order_status=lambda oid: {
            "ticker": "NVDA_US_EQ",
            "status": "FILLED",
            "filledQuantity": 5.0,
            "filledValue": 600.0,
            "fee": 0.0,
            "orderedQuantity": float("nan"),
        },
    )
    t212 = T212Provider(client=fake_cli, live_enabled=False)
    assert t212.get_orders("NVDA_US_EQ", "BUY", float("nan")) == []
    assert t212.get_orders("NVDA_US_EQ", "BUY", True) == []
    assert t212.get_orders("NVDA_US_EQ", "BUY", -10.0) == []

    # Case-insensitive ticker match and synthetic timestamp inside 600s window
    now_ms = int(time.time() * 1000)
    short_win_orders = t212.get_orders("nvda_us_eq", "BUY", 600.0)
    assert len(short_win_orders) == 1
    assert now_ms - short_win_orders[0]["timestamp"] <= 600_000

    st = t212.order_status("nvda_us_eq", "ord-1")
    assert st.status == "closed" and st.orig_qty is None

def test_t212provider_rejects_boolean_qty_price_order_id_fill_fields_iter41_s1():
    """[ITER41 #1] T212Provider rejects boolean qty/price/order_id/fill fields."""
    # 1. T212Provider rejects boolean qty/price/order_id/fill fields
    tp = T212Provider(
        client=SimpleNamespace(
            get_portfolio=lambda: [{"ticker": "AAPL_US_EQ", "averagePrice": True, "quantity": 1.0, "currentPrice": True}],
            get_cash=lambda: {"free": True},
            get_order_status=lambda oid: {"ticker": "AAPL_US_EQ", "status": "FILLED", "filledQuantity": True, "filledValue": 100.0},
        ),
        live_enabled=True,
        order_validity="DAY",
    )
    assert tp.free_balance("RON") is None
    assert tp.free_balance("AAPL_US_EQ") == 1.0
    with pytest.raises(ProviderError):
        tp.get_orders("AAPL_US_EQ", "BUY", 3600.0)
    with pytest.raises(ProviderError):
        tp._signed_qty("BUY", True)
    with pytest.raises(ProviderError):
        tp._send_order("AAPL_US_EQ", "BUY", 1.0, True, market=False)
    with pytest.raises(ProviderError):
        tp.order_status("AAPL_US_EQ", True)
    with pytest.raises(ProviderError):
        tp.order_status("AAPL_US_EQ", "OID1")
    with pytest.raises(ProviderError):
        tp.cancel_order("AAPL_US_EQ", True)

def test_t212client_place_limit_order_place_market_order_reject_bool_nan_inf_before_iter50_s1():
    """[ITER50 #1] T212Client: place_limit_order & place_market_order reject bool/NaN/Inf/0 before HTTP."""
    t212c = load_venue_modules("212trading", "t212_client")
    # 1. T212Client: place_limit_order & place_market_order reject bool/NaN/Inf/0 before HTTP
    cli = t212c.T212Client("k", "s", env="demo", min_gap_sec=True, portfolio_ttl_sec=float("nan"), order_min_gap_sec=-1.0)
    assert cli._min_gap == 0.0
    assert cli._pf_ttl == 6.0
    assert cli._order_min_gap == 0.0
    with pytest.raises(ValueError):
        cli.place_limit_order("NVDA_US_EQ", True, 100.0)
    with pytest.raises(ValueError):
        cli.place_limit_order("NVDA_US_EQ", 1.0, float("nan"))
    with pytest.raises(ValueError):
        cli.place_limit_order("NVDA_US_EQ", 0.001, 100.0)  # rounds to 0.0
    with pytest.raises(ValueError):
        cli.place_market_order("NVDA_US_EQ", False)
    with pytest.raises(ValueError):
        cli.place_market_order("NVDA_US_EQ", float("inf"))

def test_212trading_strategy_generate_tp_ladder_adapt_ladder_rungs__avg_cost__portfolio_position_step_iter50_s2():
    """[ITER50 #2] 212trading/strategy: generate_tp_ladder, adapt_ladder_rungs, _avg_cost, _portfolio_position, step."""
    t212_strat = load_venue_modules("212trading", "strategy")
    # 2. 212trading/strategy: generate_tp_ladder, adapt_ladder_rungs, _avg_cost, _portfolio_position, step
    with pytest.raises(ValueError):
        t212_strat.generate_tp_ladder(True, 10.0, 5)
    with pytest.raises(ValueError):
        t212_strat.generate_tp_ladder(2.0, 10.0, True)
    assert t212_strat.adapt_ladder_rungs([(2.0, 0.5), (4.0, 0.5)], float("nan")) == []
    assert t212_strat.adapt_ladder_rungs([(2.0, 0.5), (4.0, 0.5)], True) == []

    params = t212_strat.StratParams(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.5,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=5.0,
        takeprofit_pct=3.0, max_budget=500.0, max_dca_buys=5,
        validity="GOOD_TILL_CANCEL", enable_takeprofit=True,
        order_ttl_min=60.0, stop_loss_pct=10.0, yahoo_sym="NVDA",
    )
    dummy_cli = SimpleNamespace(
        get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": float("nan"), "averagePrice": 120.0}],
        list_active_orders=lambda: [],
    )
    st = t212_strat.Strategy(dummy_cli, "NVDA_US_EQ", params, dry_run=True, fx_to_usd=1.0, clock=lambda: 1000.0)
    assert st._portfolio_position() is None
    st.s["qty"] = 2.0
    st.s["cost_usd"] = float("nan")
    assert st._avg() is None
    assert st.reconcile(float("nan")) is False
    assert st.reconcile(True) is False
    st.step(float("nan"))
    st.step(True)

def test_212trading_replay__fx_rates__ols_slope_pct_reject_bool_nan_inf_iter50_s3():
    """[ITER50 #3] 212trading/replay: _fx_rates & _ols_slope_pct reject bool/NaN/Inf."""
    t212_rep = load_venue_modules("212trading", "replay")
    # 3. 212trading/replay: _fx_rates & _ols_slope_pct reject bool/NaN/Inf
    with pytest.raises(ValueError):
        t212_rep._fx_rates(True, 3)
    with pytest.raises(ValueError):
        t212_rep._fx_rates(float("nan"), 3)
    with pytest.raises(ValueError):
        t212_rep._fx_rates([1.0, float("inf"), 1.0], 3)
    assert t212_rep._md.compute_ols_slope_pct([True] * 10, bars=12) is None
    assert t212_rep._md.compute_ols_slope_pct([float("nan")] * 10, bars=12) is None

def test_t212provider_order_status_rejects_nan_inf_filledquantity_filledvalue_fee_iter51_s6(monkeypatch):
    """[ITER51 #6] T212Provider: order_status rejects NaN/Inf filledQuantity, filledValue, fee, fillPrice."""
    # 6. T212Provider: order_status rejects NaN/Inf filledQuantity, filledValue, fee, fillPrice
    tp = T212Provider()
    fake_t212 = SimpleNamespace(
        get_order_status=lambda oid: {
            "ticker": "NVDA_US_EQ",
            "status": "FILLED",
            "filledQuantity": "nan",
            "filledValue": 100.0,
            "fee": 0.0,
        }
    )
    monkeypatch.setattr(tp, "_client", lambda: fake_t212)
    with pytest.raises(ProviderError):
        tp.order_status("NVDA_US_EQ", "ORD1")

def test_212trading_order_manager__portfolio_qty__matching_active_orders__reconcile_record_iter53_s1():
    """[ITER53 #1] 212trading/order_manager: _portfolio_qty, _matching_active_orders, _reconcile_record."""
    # 1. 212trading/order_manager: _portfolio_qty, _matching_active_orders, _reconcile_record
    t212_om = load_venue_modules("212trading", "order_manager")

    cli_nan_pf = SimpleNamespace(get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": float("nan")}])
    assert t212_om._portfolio_qty(cli_nan_pf, "NVDA_US_EQ") is None
    cli_bool_pf = SimpleNamespace(get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": True}])
    assert t212_om._portfolio_qty(cli_bool_pf, "NVDA_US_EQ") is None

    cli_orders = SimpleNamespace(
        list_active_orders=lambda: [
            {"ticker": "NVDA_US_EQ", "quantity": True, "limitPrice": 100.0},
            {"ticker": "NVDA_US_EQ", "quantity": 1.0, "limitPrice": float("nan")},
        ]
    )
    assert t212_om._matching_active_orders(cli_orders, {"ticker": "NVDA_US_EQ", "qty": 1.0, "limit_price": 100.0}) == []
    assert t212_om._matching_active_orders(cli_orders, {"ticker": "NVDA_US_EQ", "qty": True, "limit_price": 100.0}) is None

def test_212trading_strategy_py_generate_tp_ladder__active_order_identity_iter57_s4():
    """[ITER57 #4] 212trading/strategy.py generate_tp_ladder & _active_order_identity."""
    # 4. 212trading/strategy.py generate_tp_ladder & _active_order_identity
    strat_mod = load_venue_modules("212trading", "strategy")

    with pytest.raises(ValueError, match="min_pct > -100"):
        strat_mod.generate_tp_ladder(-100.0, 10.0, 3, curve="geometric")
    with pytest.raises(ValueError, match="min_pct > -100"):
        strat_mod.generate_tp_ladder(-150.0, 10.0, 3, curve="geometric")

    ticker, side, qty, limit = strat_mod.Strategy._active_order_identity({
        "ticker": "AAPL",
        "quantity": True,
        "limitPrice": float("nan"),
    })
    assert ticker == "AAPL"
    assert qty is None
    assert limit is None

def test_212trading_strategy_py_stratparams_from_env_rejects_nan_in_iter58_s6():
    """[ITER58 #6] 212trading/strategy.py StratParams.from_env rejects NaN in STRAT_TP_LADDER."""
    # 6. 212trading/strategy.py StratParams.from_env rejects NaN in STRAT_TP_LADDER
    strat_mod = load_venue_modules("212trading", "strategy")

    base_env = {
        "STRATEGY_MODE": "avg_tp",
        "STRAT_TP_LADDER": "5:nan,10:50",
        "STRAT_MAX_BUDGET": "1000",
        "STRAT_ENTRY_PCT": "10",
        "STRAT_DCA_PCT": "10",
        "STRAT_MAX_DCA_BUYS": "5",
    }
    with pytest.raises(ValueError, match="Invalid STRAT_TP_LADDER"):
        strat_mod.StratParams.from_env(base_env)

def test_t212provider_free_balance_eur_converts_ron_using_usd_ron_eur_usd_iter61_s2(monkeypatch):
    """[ITER61 #2] T212Provider.free_balance('EUR') converts RON using usd_ron * eur_usd, and order_status rejects negative fee."""
    from unittest.mock import MagicMock
    # 2. T212Provider.free_balance("EUR") converts RON using usd_ron * eur_usd, and order_status rejects negative fee
    fake_cli = MagicMock()
    fake_cli.get_cash.return_value = {"free": 496.8}
    tp = T212Provider(client=fake_cli, live_enabled=False)
    fake_md = MagicMock()
    fake_md.get_usd_ron.return_value = 4.60
    fake_md.get_eur_usd.return_value = 1.08
    monkeypatch.setattr("importlib.import_module", lambda name: fake_md if name == "212trading.market_data" else __import__(name))
    assert tp.free_balance("USD") == pytest.approx(108.0)
    assert tp.free_balance("EUR") == pytest.approx(100.0)

    fake_cli.get_order_status.return_value = {
        "status": "FILLED",
        "ticker": "NVDA_US_EQ",
        "filledQuantity": 1.0,
        "filledValue": 100.0,
        "fee": -1.5,
    }
    with pytest.raises(ProviderError, match="invalid fill values"):
        tp.order_status("NVDA_US_EQ", "ord-1")

def test_212trading_market_data_212trading_strategy_generate_tp_ladder_iter69_s4():
    """[ITER69 #4] 212trading.market_data & 212trading.strategy.generate_tp_ladder."""
    t212_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "212trading")
    t212_md = load_local_module(t212_dir, "market_data.py", module_key="_t212_market_data_69", reload=True)
    t212_strat = load_local_module(t212_dir, "strategy.py", module_key="_t212_strategy_69", reload=True)
    # 4. 212trading.market_data & 212trading.strategy.generate_tp_ladder
    with pytest.raises(ValueError):
        t212_md.parse_interval_minutes(None)
    assert canonical_base_asset(None) == ""
    with pytest.raises(ValueError):
        t212_strat.generate_tp_ladder(1.0, 10.0, steps=2.5)
    ladder = t212_strat.generate_tp_ladder(1.0, 5.0, steps=2.0)
    assert len(ladder) == 2

def test_212trading_t212_bot_verify_instrument_isin_tolerates_non_dict_items_in_iter70_s2():
    """[ITER70 #2] 212trading.t212_bot.verify_instrument_isin tolerates non-dict items in instruments list."""
    t212_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "212trading")
    t212_bot = load_local_module(t212_dir, "t212_bot.py", module_key="_t212_bot_70", reload=True)
    # 2. 212trading.t212_bot.verify_instrument_isin tolerates non-dict items in instruments list
    dummy_t212 = SimpleNamespace(list_instruments=lambda: ["bad-item", {"ticker": "NVDA_US_EQ", "isin": "US67066G1040"}])
    assert t212_bot.verify_instrument_isin(dummy_t212, "NVDA_US_EQ", "US67066G1040") is True

def test_t212_continuous_grid_report_g_safely_rejects_non_dict_sub_iter75_s3():
    """[ITER75 #3] t212_continuous_grid.report.g safely rejects non-dict sub-entries, bools, and NaN."""
    from offline.research.t212_continuous_grid import report as t_rep
    # 3. t212_continuous_grid.report.g safely rejects non-dict sub-entries, bools, and NaN
    assert t_rep.g({"1h": "corrupt"}, "1h", "ret") is None
    assert t_rep.g({"1h": {"ret": True}}, "1h", "ret") is None
    assert t_rep.g({"1h": {"ret": float("nan")}}, "1h", "ret") is None
    assert t_rep.g({"1h": {"ret": 4.5}}, "1h", "ret") == 4.5

def test_t212provider__definitive_submit_failure_and_pair_precision_safety_iter87_s3():
    """[ITER87 #3] T212Provider._definitive_submit_failure and pair_precision safety."""
    from providers.t212_provider import _definitive_submit_failure
    # 3. T212Provider._definitive_submit_failure and pair_precision safety
    assert _definitive_submit_failure(None, "order rejected by venue") is True
    assert _definitive_submit_failure(True, "timeout") is False
    tp = T212Provider()
    assert tp.pair_precision("   ") is None
    assert tp.pair_precision(True) is None
    assert tp.pair_precision("  AAPL_US_EQ  ").base_asset == "AAPL_US_EQ"

def test_t212client_validates_env_ticker_timevalidity_and_get_historical_order_rejects_iter90_s3():
    """[ITER90 #3] T212Client validates env/ticker/timeValidity and get_historical_order rejects None order_id."""
    # 3. T212Client validates env/ticker/timeValidity and get_historical_order rejects None order_id
    t212_mod = importlib.import_module("212trading.t212_client")
    T212Client = t212_mod.T212Client
    tc = T212Client("k", "s", env=" DEMO ", min_gap_sec=0.0, portfolio_ttl_sec=1.0, order_min_gap_sec=0.0)
    assert tc.env == "demo"
    with pytest.raises(ValueError, match="invalid T212 ticker"):
        tc.place_limit_order("   ", 1.0, 100.0, validity="DAY")
    with pytest.raises(ValueError, match="invalid T212 timeValidity"):
        tc.place_limit_order("AAPL_US_EQ", 1.0, 100.0, validity="GTC")
    assert tc.get_historical_order(None) is None
    assert tc.get_historical_order(True) is None

def test_212trading_market_data_validation_future_timestamp_protection_iter91_s2(monkeypatch):
    """[ITER91 #2] 212trading.market_data validation & future-timestamp protection."""
    # 2. 212trading.market_data validation & future-timestamp protection
    t212_md = importlib.import_module("212trading.market_data")
    assert canonical_base_asset(True) == ""
    assert canonical_base_asset(" nvda_us_eq ") == "NVDA"
    with pytest.raises(ValueError, match="invalid symbol"):
        t212_md.fetch_yahoo_candles("  ", "5d", "5m")
    with pytest.raises(ValueError, match="invalid range"):
        t212_md.fetch_yahoo_candles("NVDA", True, "5m")

    # Future timestamp (>60s ahead) in check_market must not trigger fresh_series / really_trading
    import time as _time
    future_ts = _time.time() + 86400.0
    monkeypatch.setattr(
        t212_md,
        "_chart",
        lambda sym, rng="1d", interval="5m": (
            {"marketState": 12345, "regularMarketPrice": 100.0, "regularMarketVolume": 0, "regularMarketTime": future_ts},
            [(future_ts, 100.0, 500.0), (future_ts + 60, 102.0, 500.0)],
        ),
    )
    cm_res = t212_md.check_market("SPCX")
    assert cm_res["state"] == "?"
    assert cm_res["launched"] is False
    assert cm_res["trading"] is False

def test_212trading_order_manager_ticker_validity_validation_iter91_s3():
    """[ITER91 #3] 212trading.order_manager ticker/validity validation."""
    # 3. 212trading.order_manager ticker/validity validation
    t212_om = importlib.import_module("212trading.order_manager")
    assert t212_om.place_order_with_retry(None, "", 1.0, 100.0, "DAY", True) is False
    assert t212_om.place_order_with_retry(None, "NVDA_US_EQ", 1.0, 100.0, None, True) is False

def test_t212client_get_order_status_cancel_order_212trading_t212_bot_relative_import_isolation_iter94_s1(monkeypatch):
    """[ITER94 #1] T212Client get_order_status / cancel_order & 212trading.t212_bot relative import isolation & verify_instrument_isin validation."""
    import importlib.util
    # 1. T212Client get_order_status / cancel_order
    t212_client_mod = importlib.import_module("212trading.t212_client")
    cli = t212_client_mod.T212Client(
        "k", "s", env="demo", min_gap_sec=0.0, portfolio_ttl_sec=1.0, order_min_gap_sec=0.0
    )
    assert cli.get_order_status("") is None
    assert cli.get_order_status(True) is None
    assert cli.cancel_order("  ") is False
    assert cli.cancel_order(False) is False


    # 2. 212trading.t212_bot relative import isolation & verify_instrument_isin validation
    monkeypatch.setitem(sys.modules, "strategy", object())
    t212_bot_mod = importlib.import_module("212trading.t212_bot")
    assert t212_bot_mod.verify_instrument_isin(cli, "", "US123") is False
    assert t212_bot_mod.verify_instrument_isin(cli, True, "US123") is False
    assert t212_bot_mod.verify_instrument_isin(cli, "SPCX_US_EQ", "") is True

def test_t212provider_string_live_enabled_empty_symbol_in__position_and_iter95_s1():
    """[ITER95 #1] T212Provider string live_enabled, empty symbol in _position, and side validation in get_orders."""
    # 1. T212Provider string live_enabled, empty symbol in _position, and side validation in get_orders
    t212_p = T212Provider(
        client=type("C", (), {"get_portfolio": lambda self: [{"ticker": "", "quantity": 10.0, "averagePrice": 50.0}]})(),
        live_enabled="false",
    )
    assert t212_p.execution_enabled() is False
    assert t212_p._position("") is None
    with pytest.raises(ProviderError, match="invalid symbol"):
        t212_p._position("  ", strict=True)
    assert t212_p.get_orders("SPCX_US_EQ", True, 3600.0) == []
    assert t212_p.get_orders("SPCX_US_EQ", "INVALID", 3600.0) == []

def test_212trading_ipo_import_works_even_when_212trading_is_iter103_s5():
    """[ITER103 #5] 212trading.ipo import works even when 212trading is not at sys.path[0]."""
    # 5. 212trading.ipo import works even when 212trading is not at sys.path[0]
    mod_ipo = importlib.import_module("212trading.ipo")
    assert hasattr(mod_ipo, "wait_for_launch")

def test_t212provider_iter111_s3():
    """[ITER111 #3] T212Provider."""
    import providers.t212_provider as tp
    # 3. T212Provider
    t_prov = tp.T212Provider.__new__(tp.T212Provider)
    assert t_prov.free_balance(True) is None
    assert t_prov.free_balance("   ") is None
    with pytest.raises(ProviderError):
        t_prov._signed_qty(True, 1.0)
    assert t_prov.pair_precision(123) is None
    assert t_prov.pair_precision("   ") is None

def test_212trading_market_data__chart_url_encodes_symbol_range_and_iter117_s5(monkeypatch):
    """[ITER117 #5] 212trading.market_data._chart URL-encodes symbol, range, and interval."""
    # 5. 212trading.market_data._chart URL-encodes symbol, range, and interval
    t212_md = importlib.import_module("212trading.market_data")
    called_urls = []
    monkeypatch.setattr(t212_md, "http_get", lambda url, **_kw: called_urls.append(url) or (0, b""))
    t212_md._chart("BRK/B?x=1", rng="5d&foo=1", interval="5m#bar")
    assert len(called_urls) == 1
    assert "BRK%2FB%3Fx%3D1" in called_urls[0]
    assert "range=5d%26foo%3D1" in called_urls[0]
    assert "interval=5m%23bar" in called_urls[0]

def test_212trading_order_manager__persist_marker_and_integer_sanitization_iter117_s6(monkeypatch, tmp_path):
    """[ITER117 #6] 212trading.order_manager._persist_marker and integer sanitization."""
    # 6. 212trading.order_manager._persist_marker and integer sanitization
    t212_om = importlib.import_module("212trading.order_manager")
    marker_path = tmp_path / "marker.json"
    monkeypatch.setattr(t212_om, "_marker_path", lambda _t, test=False: str(marker_path))
    t212_om._persist_marker({"version": 2, "ticker": "SPCX_US_EQ"}, str(marker_path))
    assert marker_path.exists()

def test_periods_per_year_rejects_nan_inf_bool_and_non_positive_iter136_s1():
    """[ITER136 #1] periods_per_year rejects NaN, inf, bool, and non-positive values."""
    from offline.runners.t212_walk_forward_baseline import periods_per_year
    # 1. periods_per_year rejects NaN, inf, bool, and non-positive values
    for bad_im in (float("nan"), float("inf"), -10, 0, True, "bad"):
        with pytest.raises(ValueError, match="interval_minutes must be positive"):
            periods_per_year(bad_im)  # type: ignore[arg-type]
    assert periods_per_year(1440) == pytest.approx(252.0)

def test_fx_quote_normalizes_whitespace_casing_and_rejects_empty_boolean_iter136_s2():
    """[ITER136 #2] fx_quote normalizes whitespace/casing and rejects empty/boolean currency."""
    from offline.runners.t212_walk_forward_baseline import fx_quote
    # 2. fx_quote normalizes whitespace/casing and rejects empty/boolean currency
    assert fx_quote(" ron ") == ("USDRON=X", True)
    assert fx_quote(" eur ") == ("EURUSD=X", False)
    for bad_ccy in ("", "   ", None, True):
        with pytest.raises(ValueError, match="currency must be a non-empty string"):
            fx_quote(bad_ccy)  # type: ignore[arg-type]

def test_t212_client_mod_clean_order_id_normalizes_integer_floats_and_rejects_sentinel_iter138_s1():
    """[ITER138 #1] t212_client_mod.clean_order_id normalizes integer floats and rejects sentinel strings / containers."""
    t212_client_mod = importlib.import_module("212trading.t212_client")
    T212Client = t212_client_mod.T212Client
    # 1. t212_client_mod.clean_order_id normalizes integer floats and rejects sentinel strings / containers
    assert t212_client_mod.clean_order_id(12345.0) == "12345"
    assert t212_client_mod.clean_order_id(" 999 ") == "999"
    for bad in ("nan", "null", "none", "inf", 12.5, ["123"], {"id": 1}, True, None):
        assert t212_client_mod.clean_order_id(bad) is None

    cli = T212Client("test-key", env="demo", min_gap_sec=0.0, portfolio_ttl_sec=1.0, order_min_gap_sec=0.0)
    assert cli.get_order_status("nan") is None
    assert cli.get_order_status(["123"]) is None
    assert cli.cancel_order("null") is False
    assert cli.cancel_order({"id": 1}) is False

def test_t212provider_order_status_and_cancel_order_accept_integer_floats_while_iter138_s2():
    """[ITER138 #2] T212Provider.order_status and cancel_order accept integer floats while rejecting sentinel strings / containers."""
    # 2. T212Provider.order_status and cancel_order accept integer floats while rejecting sentinel strings / containers
    seen_ids = []
    fake_cli = SimpleNamespace(
        get_order_status=lambda oid: (
            seen_ids.append(("status", oid))
            or {"id": oid, "ticker": "NVDA_US_EQ", "status": "FILLED", "filledQuantity": 1.0, "filledValue": 100.0}
        ),
        cancel_order=lambda oid: seen_ids.append(("cancel", oid)) or True,
        place_limit_order=lambda ticker, qty, px, validity="DAY": (200, {"id": 777.0, "status": "NEW"}),
    )
    prov = T212Provider(client=fake_cli, live_enabled=True, order_validity="DAY")
    st = prov.order_status("NVDA_US_EQ", 555.0)  # type: ignore[arg-type]
    assert st.status == "closed"
    prov.cancel_order("NVDA_US_EQ", 555.0)  # type: ignore[arg-type]
    assert seen_ids == [("status", "555"), ("cancel", "555")]

    for bad_oid in ("nan", "null", ["555"], {"id": 555}, True, None):
        with pytest.raises(ProviderError):
            prov.order_status("NVDA_US_EQ", bad_oid)  # type: ignore[arg-type]
        with pytest.raises(ProviderError):
            prov.cancel_order("NVDA_US_EQ", bad_oid)  # type: ignore[arg-type]

    placed = prov.place_order("NVDA_US_EQ", "BUY", 100.0, 1.0)
    assert placed["orderId"] == "777"

def test__sanitize_t212_state_parses_string_boolean_flags_false_false_and_iter140_s1():
    """[ITER140 #1] _sanitize_t212_state parses string boolean flags ('false'/'0' -> False) and cleans order IDs & levels."""
    strat_mod = importlib.import_module("212trading.strategy")
    _sanitize_t212_state = strat_mod._sanitize_t212_state
    # 1. _sanitize_t212_state parses string boolean flags ("false"/"0" -> False) and cleans order IDs & levels
    raw_state = {
        "tr_alerted": "false",
        "tr_armed": "0",
        "sl_alerted": "false",
        "sl_pending": "0",
        "pending_submit": {
            "side": "BUY",
            "qty": 1.0,
            "limit": 100.0,
            "intent_id": "intent-1",
            "market": "false",
            "level": "5.0",
            "portfolio_fill_observed": "0",
        },
        "orders": [
            {
                "id": 12345.0,
                "side": "SELL",
                "qty": 2.0,
                "limit": 110.0,
                "market": "false",
                "level": "7.5",
                "cancel_requested": "false",
                "dca_counted": "0",
                "intent_id": "nan",
            },
            {"id": "nan", "side": "BUY", "qty": 1.0, "limit": 95.0},
            {"id": ["bad"], "side": "BUY", "qty": 1.0, "limit": 95.0},
        ],
    }
    clean = _sanitize_t212_state(raw_state)
    assert clean["tr_alerted"] is False
    assert clean["tr_armed"] is False
    assert clean["sl_alerted"] is False
    assert clean["sl_pending"] is False
    assert clean["pending_submit"]["market"] is False
    assert clean["pending_submit"]["level"] == pytest.approx(5.0)
    assert clean["pending_submit"]["portfolio_fill_observed"] is False
    assert len(clean["orders"]) == 1
    ord0 = clean["orders"][0]
    assert ord0["id"] == "12345"
    assert ord0["market"] is False
    assert ord0["level"] == pytest.approx(7.5)
    assert ord0["cancel_requested"] is False
    assert ord0["dca_counted"] is False
    assert ord0["intent_id"] is None

    # Sentinel intent_id in pending_submit invalidates the corrupted pending_submit
    bad_ps = _sanitize_t212_state(
        {"pending_submit": {"side": "BUY", "qty": 1.0, "limit": 100.0, "intent_id": "null"}}
    )
    assert bad_ps["pending_submit"] is None

def test_212trading_analyze_volatility_py_pct_pullbacks_iter150_s2():
    """[ITER150 #2] 212trading/analyze_volatility.py pct & pullbacks."""
    # 2. 212trading/analyze_volatility.py pct & pullbacks
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    av_mod = load_local_module(os.path.join(root, "212trading"), "analyze_volatility", module_key="t212_av_t150", reload=True)

    assert av_mod.pct(True, 100.0) == 0.0
    assert av_mod.pct(100.0, float("nan")) == 0.0
    assert av_mod.pct(100.0, -5.0) == 0.0
    assert av_mod.pullbacks([True, float("nan"), -10.0, 100.0, 80.0, 105.0]) == [20.0]

def test_t212provider__orders_live_rejects_nan_false_containers_iter152_s3():
    """[ITER152 #3] T212Provider _orders_live rejects NaN / 'false' / containers."""
    from providers.t212_provider import _parse_bool_flag as t212_bool
    # 3. T212Provider _orders_live rejects NaN / "false" / containers
    assert t212_bool("false", True) is False
    assert t212_bool(float("nan"), False) is False
    assert T212Provider(live_enabled="false").execution_enabled() is False  # type: ignore[arg-type]
    assert T212Provider(live_enabled=float("nan")).execution_enabled() is False  # type: ignore[arg-type]
    assert T212Provider(live_enabled="true").execution_enabled() is True  # type: ignore[arg-type]

def test_t212provider__position_none_strict_false_returns_none_without_iter164_s4():
    """[ITER164 #4] T212Provider: _position(None, strict='false') returns None without raising ProviderError."""
    # 4. T212Provider: _position(None, strict="false") returns None without raising ProviderError
    tp = T212Provider(client=SimpleNamespace(get_portfolio=lambda: []))
    assert tp._position(None, strict="false") is None

def test_t212provider__signed_qty_aliases_and_positional_force_in_place_order_iter169_s4():
    """[ITER169 #4] T212Provider _signed_qty aliases and positional force in place_order."""
    from providers import t212_provider as t212p
    # 4. T212Provider _signed_qty aliases and positional force in place_order
    assert t212p.T212Provider._signed_qty("BID", 1.234) == pytest.approx(1.23)
    assert t212p.T212Provider._signed_qty("ASK", 1.234) == pytest.approx(-1.23)

    class _FakeT212Client:
        def place_market_order(self, symbol, signed_qty, extended_hours=False):
            return 200, {"id": 98765, "quantity": signed_qty}

    t212 = t212p.T212Provider(client=_FakeT212Client(), live_enabled="y")
    res_t212 = t212.place_order("AAPL_US_EQ", "LONG", 180.0, 1.5, True)
    assert res_t212["orderId"] == "98765" and res_t212["quantity"] == pytest.approx(1.5)
