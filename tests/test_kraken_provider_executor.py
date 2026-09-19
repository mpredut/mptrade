"""Kraken executor-contract tests with an injected offline fake client."""
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import instrument as instrument_module  # noqa: E402
import order_guard  # noqa: E402
from instrument import Instrument  # noqa: E402
from providers import kraken_provider  # noqa: E402
from providers.kraken_provider import (  # noqa: E402
    KrakenProvider,
    kraken_strategy_dry_run,
)
from providers.market_api import MarketApi  # noqa: E402
from providers.strategy_executor import (  # noqa: E402
    StrategyExecutor, OrderStatus, PairPrecision, ProviderError)


class FakeClient:
    def __init__(self):
        self.calls = []

    def add_order(self, pair, side, volume, price=None, ordertype="limit", validate=False,
                  cl_ord_id=None):
        self.calls.append((
            "add_order", pair, side, volume, price, ordertype, validate, cl_ord_id,
        ))
        return {"txid": ["OABC-123"], "descr": {}}

    def query_orders(self, txids):
        return {txids: {"status": "closed", "vol_exec": "2.5", "cost": "150.0", "fee": "0.39"}}

    def open_orders(self):
        return {
            "OPEN-1": {
                "cl_ord_id": "0123456789abcdef0123456789abcdef",
                "status": "open", "vol": "2.5", "vol_exec": "0.5",
                "descr": {
                    "pair": "HYPEUSD", "type": "buy", "price": "60.25",
                },
            },
            "OTHER-1": {
                "cl_ord_id": "11111111111111111111111111111111",
                "status": "open", "vol": "1", "vol_exec": "0",
                "descr": {
                    "pair": "ADAUSD", "type": "sell", "price": "1.25",
                },
            },
        }

    def closed_orders(self):
        return {
            "CLOSED-1": {
                "cl_ord_id": "fedcba9876543210fedcba9876543210",
                "status": "closed", "descr": {"pair": "HYPEUSD"},
            },
        }

    def cancel_order(self, txid):
        self.calls.append(("cancel_order", txid))
        return {"count": 1}

    def pair_info(self, pair):
        return {
            "pair_decimals": 2, "lot_decimals": 8,
            "ordermin": "0.1", "base": "HYPE",
        }

    def balance(self):
        return {"HYPE": "2.5", "ZUSD": "1000"}

    def last_price(self, pair):
        return 60.0

    def ohlc_closes(self, pair, interval):
        return [10.0, 11.0, 12.0]


def _provider(fake):
    p = KrakenProvider()
    p._cli = fake                    # It short-circuits _client() (no keys, no network).
    return p


class KrakenExecutorContractTest(unittest.TestCase):
    def setUp(self):
        self.previous_live = os.environ.get("KRAKEN_LIVE_ORDERS")
        os.environ["KRAKEN_LIVE_ORDERS"] = "true"
        self.fake = FakeClient()
        self.p = _provider(self.fake)

    def tearDown(self):
        if self.previous_live is None:
            os.environ.pop("KRAKEN_LIVE_ORDERS", None)
        else:
            os.environ["KRAKEN_LIVE_ORDERS"] = self.previous_live

    @staticmethod
    def _slot_context():
        class AllowedSlot:
            allowed = True
            info = {}

            def commit(self, _order_id=None):
                return None

        class SlotContext:
            def __enter__(self):
                return AllowedSlot()

            def __exit__(self, *_args):
                return False

        return SlotContext()

    def _instrument_place(self, *, force=False, client_order_id=None):
        os.environ["KRAKEN_LIVE_ORDERS"] = "true"
        instrument = Instrument(
            "HYPE_KRAKEN", "HYPEUSD", "Kraken", base="HYPE", quote="USD",
            api=MarketApi([self.p]))
        with (
            patch.object(order_guard, "daily_limit_guard",
                         return_value=(True, None)),
            patch("instrument.trade_cooldown.trade_slot",
                  return_value=self._slot_context()),
            patch.object(instrument_module._outcomes_log, "log_order_outcome"),
        ):
            return instrument.place(
                "BUY", 60.0, 1.0, smart=False, wait_for_trend=False,
                caller_owns_retry=True, bypass_profit_guard=True,
                force=force, client_order_id=client_order_id)

    def test_satisfies_protocol(self):
        self.assertIsInstance(self.p, StrategyExecutor)

    def test_submit_order_behaviors(self):
        with self.subTest(msg="limit_submit_returns_order_id"):
            oid = self.p.submit_order("HYPEUSD", "buy", 2.5, price=60.0)
            self.assertEqual(oid, "OABC-123")
            self.assertEqual(self.fake.calls[-1],
                             ("add_order", "HYPEUSD", "buy", 2.5, 60.0, "limit", False, None))

        with self.subTest(msg="propagates_client_order_id"):
            client_id = "0123456789abcdef0123456789abcdef"
            self.p.submit_order(
                "HYPEUSD", "buy", 2.5, price=60.0, client_order_id=client_id,
            )
            self.assertEqual(self.fake.calls[-1][-1], client_id)

        with self.subTest(msg="market_without_a_price"):
            self.p.submit_order("HYPEUSD", "sell", 1.0, price=59.0, market=True)
            c = self.fake.calls[-1]
            self.assertEqual(c[5], "market")
            self.assertIsNone(c[4])

        with self.subTest(msg="refuses_when_live_execution_is_disabled"):
            os.environ["KRAKEN_LIVE_ORDERS"] = "false"
            calls_before = len(self.fake.calls)
            with self.assertRaisesRegex(ProviderError, "real execution is disabled"):
                self.p.submit_order("HYPEUSD", "buy", 1.0, price=60.0)
            self.assertEqual(len(self.fake.calls), calls_before)
            os.environ["KRAKEN_LIVE_ORDERS"] = "true"

        with self.subTest(msg="without_a_txid_raises"):
            with patch.object(self.fake, "add_order", return_value={"descr": {}}):
                with self.assertRaises(ProviderError):
                    self.p.submit_order("HYPEUSD", "buy", 1.0, price=60.0)

        with self.subTest(msg="venue_submit_error_becomes_provider_error"):
            def boom(*a, **k):
                raise RuntimeError("Kraken: Insufficient funds")
            with patch.object(self.fake, "add_order", side_effect=boom):
                with self.assertRaises(ProviderError):
                    self.p.submit_order("HYPEUSD", "buy", 1.0, price=60.0)

    def test_live_instrument_place_behaviors(self):
        with self.subTest(msg="forwards_the_deterministic_client_id"):
            raw_client_id = "OR_0123456789abcdef01234567_0"
            expected = kraken_provider._kraken_client_order_id(raw_client_id)
            order = self._instrument_place(client_order_id=raw_client_id)
            self.assertEqual(order["txid"], ["OABC-123"])
            self.assertEqual(self.fake.calls[-1][-1], expected)

            self.p.submit_order(
                "HYPEUSD", "buy", 1.0, price=60.0,
                client_order_id=raw_client_id)
            self.assertEqual(self.fake.calls[-1][-1], expected)

        with self.subTest(msg="force_path_places_a_market_order_without_price"):
            order = self._instrument_place(force=True)
            self.assertEqual(order["txid"], ["OABC-123"])
            call = self.fake.calls[-1]
            self.assertEqual(call[5], "market")
            self.assertIsNone(call[4])
            self.assertFalse(call[6])

    def test_client_id_lookup_is_not_advertised_as_authoritative(self):
        self.assertFalse(
            self.p.reconciliation_capabilities().lookup_by_client_order_id)

    def test_strategy_orchestration_requires_both_execution_flags(self):
        for strategy_execute, provider_execute, expected_dry in (
                (False, False, True),
                (False, True, True),
                (True, False, True),
                (True, True, False)):
            with self.subTest(
                    strategy_execute=strategy_execute,
                    provider_execute=provider_execute):
                os.environ["KRAKEN_LIVE_ORDERS"] = (
                    "true" if provider_execute else "false")
                self.assertEqual(
                    kraken_strategy_dry_run(False, strategy_execute),
                    expected_dry)
                self.assertTrue(
                    kraken_strategy_dry_run(True, strategy_execute))

    def test_preflight_behaviors(self):
        with self.subTest(msg="sell_refuses_quantity_above_balance"):
            with self.assertRaisesRegex(ProviderError, "insufficient funds SELL"):
                self.p.preflight_order("HYPEUSD", "sell", 2.50000001, price=60.0)
            self.assertFalse(any(call[0] == "add_order" for call in self.fake.calls))

        with self.subTest(msg="sell_accepts_reconciled_balance"):
            self.p.preflight_order("HYPEUSD", "sell", 2.5, price=60.0)

        with self.subTest(msg="buy_leaves_fee_and_slippage_to_venue"):
            def boom():
                return (_ for _ in ()).throw(AssertionError("BUY preflight must not read balance"))
            with patch.object(self.fake, "balance", side_effect=boom):
                self.p.preflight_order("HYPEUSD", "buy", 100.0, price=60.0)

        with self.subTest(msg="final_submit_gate_observes_a_switch_flip"):
            states = iter((True, False))
            original_preflight = self.p.preflight_order
            def preflight_then_disable(*args, **kwargs):
                result = original_preflight(*args, **kwargs)
                self.assertTrue(next(states))
                os.environ["KRAKEN_LIVE_ORDERS"] = "false"
                return result

            # We use manual patching here like the original code did because we modify states
            self.p.preflight_order = preflight_then_disable
            self.p.preflight_order(
                "HYPEUSD", "buy", 1.0, price=60.0, market=False, kind="DCA")
            self.assertFalse(next(states))

            with self.assertRaisesRegex(ProviderError, "real execution is disabled"):
                self.p.submit_order("HYPEUSD", "buy", 1.0, price=60.0)
            
            self.p.preflight_order = original_preflight
            os.environ["KRAKEN_LIVE_ORDERS"] = "true"

    def test_cancel_order_behaviors(self):
        with self.subTest(msg="cancel_delegates"):
            self.p.cancel_order("HYPEUSD", "OABC-123")
            self.assertEqual(self.fake.calls[-1], ("cancel_order", "OABC-123"))

        with self.subTest(msg="cancel_is_idempotent_on_an_unknown_order"):
            def boom(txid):
                raise RuntimeError("EOrder:Unknown order")
            with patch.object(self.fake, "cancel_order", side_effect=boom):
                self.p.cancel_order("HYPEUSD", "GONE")

        with self.subTest(msg="unconfirmed_cancel_raises"):
            with patch.object(self.fake, "cancel_order", return_value={"count": 0}):
                with self.assertRaises(ProviderError):
                    self.p.cancel_order("HYPEUSD", "OABC-123")

    def test_order_query_behaviors(self):
        with self.subTest(msg="order_status_mapping"):
            st = self.p.order_status("HYPEUSD", "OABC-123")
            self.assertIsInstance(st, OrderStatus)
            self.assertEqual(st.status, "closed")
            self.assertEqual(st.filled_qty, 2.5)
            self.assertEqual(st.cost, 150.0)
            self.assertEqual(st.fee, 0.39)

        with self.subTest(msg="missing_order_status_raises"):
            with patch.object(self.fake, "query_orders", return_value={}):
                with self.assertRaises(ProviderError):
                    self.p.order_status("HYPEUSD", "NOPE")

        with self.subTest(msg="order_by_client_id_searches_open_and_closed"):
            self.assertEqual(
                self.p.order_by_client_id(
                    "HYPEUSD", "0123456789abcdef0123456789abcdef"),
                {"orderId": "OPEN-1", "status": "open"},
            )
            self.assertEqual(
                self.p.order_by_client_id(
                    "HYPEUSD", "fedcba9876543210fedcba9876543210"),
                {"orderId": "CLOSED-1", "status": "closed"},
            )
            self.assertIsNone(
                self.p.order_by_client_id("HYPEUSD", "0" * 32))

        with self.subTest(msg="open_orders_normalises_and_filters_by_symbol"):
            self.assertEqual(self.p.open_orders("HYPEUSD"), [{
                "orderId": "OPEN-1",
                "clientOrderId": "0123456789abcdef0123456789abcdef",
                "side": "BUY",
                "price": 60.25,
                "origQty": 2.5,
                "executedQty": 0.5,
                "status": "OPEN",
            }])

        with self.subTest(msg="ambiguous_open_orders_payload_fails_closed"):
            with patch.object(self.fake, "open_orders", return_value={"BROKEN": {"status": "open", "vol": "1", "vol_exec": "0"}}):
                with self.assertRaisesRegex(ProviderError, "without a pair"):
                    self.p.open_orders("HYPEUSD")

    def test_pair_info_and_ohlc_behaviors(self):
        with self.subTest(msg="pair_precision_mapping"):
            pp = self.p.pair_precision("HYPEUSD")
            self.assertEqual(pp, PairPrecision(
                price_decimals=2, volume_decimals=8,
                order_min=0.1, base_asset="HYPE",
            ))

        with self.subTest(msg="unlisted_pair_precision_returns_none"):
            with patch.object(self.fake, "pair_info", return_value=None):
                self.assertIsNone(self.p.pair_precision("NEWX"))

        with self.subTest(msg="ohlc_closes_delegates"):
            self.assertEqual(self.p.ohlc_closes("HYPEUSD", 240), [10.0, 11.0, 12.0])

            series = self.p.ohlc_series("HYPEUSD", 240)
            self.assertEqual(series.closes, (10.0, 11.0, 12.0))
            self.assertIsNone(series.last_closed_at)


if __name__ == "__main__":
    unittest.main(verbosity=2)
