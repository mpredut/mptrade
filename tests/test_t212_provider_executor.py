"""The StrategyExecutor contract for Trading212, without network or real keys."""

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T212_DIR = os.path.join(ROOT, "212trading")
if T212_DIR not in sys.path:
    sys.path.insert(0, T212_DIR)

import t212_client  # noqa: E402

from providers.strategy_executor import (
    OrderStatus,
    PairPrecision,
    ProviderError,
    StrategyExecutor,
)
from providers.t212_provider import T212Provider


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

    def get_portfolio(self):
        return self.portfolio_result

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


class T212ClientTransportTest(unittest.TestCase):
    def test_the_market_payload_and_the_history_fallback(self):
        client = t212_client.T212Client(
            "dummy", "dummy", env="demo", min_gap_sec=0.0,
            portfolio_ttl_sec=6.0,
        )
        client._min_gap = 0.0
        with mock.patch.object(
            t212_client, "http_post_json", return_value=(200, b'{"id": 55}')
        ) as post:
            status, data = client.place_market_order("NVDA_US_EQ", -0.25)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
