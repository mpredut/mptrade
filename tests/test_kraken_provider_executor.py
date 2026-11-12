"""Kraken executor-contract tests with an injected offline fake client."""
import base64
import importlib
import json
import os
import time as _time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from cacheaccount import CacheTradeManager
import instrument as instrument_module
from instrument import Instrument
from kraken import kraken_bot as kb, kraken_client as kc, kraken_xstock_watch as kxw, shadow_live as k_sl
from kraken.kraken_client import KrakenClient, KrakenError
import kraken.market_data as kmd
from kraken.market_data import pair_precision
from offline.research.kraken_continuous_grid import grid as k_grid
from offline.research.t212_continuous_grid import grid as t_grid
import order_guard
from providers import kraken_provider
from providers.base import aggregate_fills_by_order_id
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import (
    KrakenProvider,
    _normalize_kraken_fill,
    _parse_bool_flag as krk_bool,
    kraken_pair_matches,
)
from providers.market_api import MarketApi, api
from providers.order_ohlc_utils import clean_order_id, require_clean_order_id
from providers.strategy_executor import (
    OrderStatus,
    PairPrecision,
    ProviderError,
    StrategyExecutor,
    SubmissionRefused,
)
from providers.t212_provider import T212Provider
from tests.conftest import load_venue_modules



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

    def _private(self, endpoint, data=None):
        return {"trades": {}}

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
    p._fills_from_cache = lambda _s: None
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

    def _instrument_place(self, *, force=False, client_order_id=None, outcome_context=None):
        os.environ["KRAKEN_LIVE_ORDERS"] = "true"
        instrument = Instrument(
            "HYPE_KRAKEN", "HYPEUSD", "Kraken", base="HYPE", quote="USD",
            api=MarketApi([self.p]))
        with (
            patch.object(order_guard, "daily_limit_guard",
                         return_value=(True, None)),
            patch.object(instrument_module.order_lifecycle, "log_order_outcome"),
        ):
            return instrument.place(
                "BUY", 60.0, 1.0, smart=False, wait_for_trend=False,
                caller_owns_retry=True,
                force=force, client_order_id=client_order_id,
                _outcome_context=outcome_context,
                bypass_guards={"all"})

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
            with patch.object(self.fake, "add_order", side_effect=RuntimeError("Kraken: EOrder:Insufficient funds")):
                with self.assertRaises(kraken_provider.SubmissionRefused):
                    self.p.submit_order("HYPEUSD", "buy", 1.0, price=60.0)
            with patch.object(self.fake, "add_order", side_effect=RuntimeError("network timeout")):
                with self.assertRaises(ProviderError) as ctx:
                    self.p.submit_order("HYPEUSD", "buy", 1.0, price=60.0)
                self.assertNotIsInstance(ctx.exception, kraken_provider.SubmissionRefused)

    def test_live_instrument_place_behaviors(self):
        with self.subTest(msg="forwards_the_deterministic_client_id"):
            raw_client_id = "OR_0123456789abcdef01234567_0"
            expected = kraken_provider.KrakenProvider.derive_client_order_id(raw_client_id)
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

        with self.subTest(msg="definitive_venue_refusal_in_place_order_marks_refused_not_unknown"):
            ctx_refused = {}
            with patch.object(self.fake, "add_order", side_effect=RuntimeError("KrakenError: EOrder:Insufficient funds")):
                order = self._instrument_place(outcome_context=ctx_refused)
            self.assertIsNone(order)
            self.assertEqual(ctx_refused.get("state"), "refused")
            self.assertIn("Insufficient funds", str(ctx_refused.get("reason") or ""))

            ctx_unknown = {}
            with patch.object(self.fake, "add_order", side_effect=RuntimeError("network timeout")):
                order = self._instrument_place(outcome_context=ctx_unknown)
            self.assertIsNone(order)
            self.assertEqual(ctx_unknown.get("state"), "unknown")
            self.assertEqual(ctx_unknown.get("reason"), "response_without_order_id")

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
                    KrakenProvider.strategy_dry_run(False, strategy_execute),
                    expected_dry)
                self.assertTrue(
                    KrakenProvider.strategy_dry_run(True, strategy_execute))

    def test_preflight_behaviors(self):
        with self.subTest(msg="sell_refuses_quantity_above_balance"):
            with self.assertRaisesRegex(ProviderError, "insufficient funds SELL"):
                self.p.preflight_order("HYPEUSD", "sell", 2.50000001, price=60.0)
            self.assertFalse(any(call[0] == "add_order" for call in self.fake.calls))

        with self.subTest(msg="sell_accepts_reconciled_balance"):
            self.p.preflight_order("HYPEUSD", "sell", 2.5, price=60.0)

        with self.subTest(msg="buy_leaves_fee_and_slippage_to_venue"):
            with patch.object(self.fake, "balance", side_effect=AssertionError("BUY preflight must not read balance")):
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
            with patch.object(self.fake, "cancel_order", side_effect=RuntimeError("EOrder:Unknown order")):
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

        with self.subTest(msg="order_status_normalizes_pending_and_cancelled"):
            with patch.object(self.fake, "query_orders", return_value={
                "OPEND-1": {"status": "pending", "vol_exec": "0", "cost": "0", "fee": "0"}
            }):
                st_pending = self.p.order_status("HYPEUSD", "OPEND-1")
                self.assertEqual(st_pending.status, "open")
            with patch.object(self.fake, "query_orders", return_value={
                "OCANC-1": {"status": "cancelled", "vol_exec": "0", "cost": "0", "fee": "0"}
            }):
                st_canc = self.p.order_status("HYPEUSD", "OCANC-1")
                self.assertEqual(st_canc.status, "canceled")

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

        with self.subTest(msg="get_price_history_snaps_to_valid_kraken_ohlc_intervals"):
            now_s = int(_time.time())
            observed_intervals = []
            def _fake_public(endpoint, params=None):
                self.assertEqual(endpoint, "OHLC")
                observed_intervals.append(params["interval"])
                return {"ADAUSD": [[now_s - 60, "0.24", "0.25", "0.24", "0.245", "0.244", "100", 5]], "last": now_s - 60}
            self.fake._public = _fake_public
            hist_24h = self.p.get_price_history("ADAUSD", 24.0)
            hist_7d = self.p.get_price_history("ADAUSD", 168.0)
            self.assertEqual(observed_intervals, [5, 15])
            self.assertEqual(len(hist_24h), 1)
            self.assertAlmostEqual(hist_24h[0]["price"], 0.245)
            self.assertEqual(len(hist_7d), 1)
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

    def test_usd_equivalent_free_balance_and_auto_funding(self):
        with self.subTest(msg="free_balance_usd_sums_fiat_and_1to1_stablecoins"):
            with patch.object(self.fake, "balance", return_value={
                "ZUSD": "0.0092",
                "USDC": "2520.03719900",
                "USDG": "1.75000000",
            }):
                self.assertAlmostEqual(
                    self.p.free_balance("USD"),
                    0.0092 + 2520.03719900 + 1.75000000,
                    places=6,
                )
                self.assertEqual(
                    self.p.usd_cash_breakdown(),
                    {"USD": 0.01, "USDC": 2520.04, "USDG": 1.75},
                )
                # Direct asset query still returns only that specific bucket.
                self.assertAlmostEqual(
                    self.p.free_balance("USDC"), 2520.03719900, places=6)

        with self.subTest(msg="corrupt_usd_bucket_fails_closed"):
            with patch.object(self.fake, "balance", return_value={
                "ZUSD": "10.0",
                "USDC": True,
            }):
                self.assertIsNone(self.p.free_balance("USD"))

        with self.subTest(msg="buy_on_usd_pair_auto_converts_usdc_when_zusd_short"):
            self.fake.calls.clear()
            with patch.object(self.fake, "balance", return_value={
                "ZUSD": "0.01",
                "USDC": "2520.04",
            }):
                oid = self.p.submit_order("HYPEUSD", "buy", 2.0, price=50.0)
                self.assertEqual(oid, "OABC-123")
                self.assertEqual(len(self.fake.calls), 2)
                conv_call, buy_call = self.fake.calls
                self.assertEqual(conv_call[0], "add_order")
                self.assertEqual(conv_call[1], "USDCUSD")
                self.assertEqual(conv_call[2], "sell")
                self.assertGreaterEqual(conv_call[3], 100.0)
                self.assertEqual(conv_call[5], "market")
                self.assertEqual(buy_call[1], "HYPEUSD")
                self.assertEqual(buy_call[2], "buy")

        with self.subTest(msg="balance_ex_subtracts_hold_trade_and_triggers_usdc_conversion"):
            self.fake.calls.clear()
            with patch.object(self.fake, "balance_ex", create=True, return_value={
                "ZUSD.F": {"balance": "808.8325", "hold_trade": "800.0000"},
                "USDC.F": {"balance": "2049.5800", "hold_trade": "0.0"},
                "HYPE": {"balance": "4.9386", "hold_trade": "4.8769"},
            }):
                self.assertAlmostEqual(self.p.free_balance("HYPE"), 0.0617, places=4)
                self.assertEqual(
                    self.p.usd_cash_breakdown(),
                    {"USD": 8.83, "USDC": 2049.58},
                )
                oid = self.p.submit_order("HYPEUSD", "buy", 9.81595, price=81.50)
                self.assertEqual(oid, "OABC-123")
                self.assertEqual(len(self.fake.calls), 2)
                conv_call, buy_call = self.fake.calls
                self.assertEqual(conv_call[0], "add_order")
                self.assertEqual(conv_call[1], "USDCUSD")
                self.assertEqual(conv_call[2], "sell")
                self.assertGreaterEqual(conv_call[3], 790.0)
                self.assertEqual(buy_call[1], "HYPEUSD")
                self.assertEqual(buy_call[2], "buy")

    def test_ledger_to_fills_ignores_dustsweeping_and_multi_asset_groups(self):
        ledger = {
            "L1": {
                "type": "trade", "subtype": "dustsweeping", "refid": "SWEEP-1",
                "asset": "HYPE", "amount": "-0.0002315", "fee": "0", "time": 1793288936.0,
            },
            "L2": {
                "type": "trade", "subtype": "dustsweeping", "refid": "SWEEP-1",
                "asset": "PEPE", "amount": "-5094.26", "fee": "0", "time": 1793288936.0,
            },
            "L3": {
                "type": "trade", "subtype": "dustsweeping", "refid": "SWEEP-1",
                "asset": "ZUSD", "amount": "1.2900", "fee": "0.0398", "time": 1793288936.0,
            },
            "L4": {
                "type": "trade", "subtype": "", "refid": "REAL-TRADE-1",
                "asset": "HYPE", "amount": "-27.2", "fee": "0", "time": 1793288800.0,
            },
            "L5": {
                "type": "trade", "subtype": "", "refid": "REAL-TRADE-1",
                "asset": "ZUSD", "amount": "2537.76", "fee": "6.598", "time": 1793288800.0,
            },
        }
        fills = KrakenProvider.ledger_to_fills(ledger)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["symbol"], "HYPEUSD")
        self.assertAlmostEqual(fills[0]["price"], 93.3, places=4)
        self.assertEqual(fills[0]["side"], "SELL")



class _SequentialPairInfoClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def pair_info(self, symbol):
        self.calls += 1
        r = self._responses.pop(0) if self._responses else {}
        if isinstance(r, Exception):
            raise r
        return r


class MinOrderQtyCacheTest(unittest.TestCase):
    def test_failed_lookup_not_cached_then_recovers(self):
        for label, first_response in (("empty", {}), ("exception", RuntimeError("DNS blip"))):
            with self.subTest(case=label):
                fake = _SequentialPairInfoClient([first_response, {"ordermin": "0.1"}])
                p = _provider(fake)
                self.assertEqual(p.min_order_qty("HYPEUSD"), 0.0)
                self.assertNotIn("HYPEUSD", p._minqty)
                self.assertEqual(p.min_order_qty("HYPEUSD"), 0.1)
                self.assertEqual(p._minqty.get("HYPEUSD"), 0.1)

    def test_positive_cached_no_refetch(self):
        fake = _SequentialPairInfoClient([{"ordermin": "0.1"}])
        p = _provider(fake)
        self.assertEqual(p.min_order_qty("HYPEUSD"), 0.1)
        self.assertEqual(p.min_order_qty("HYPEUSD"), 0.1)
        self.assertEqual(fake.calls, 1)


class AddOrderRoundingTest(unittest.TestCase):
    def _client(self, pair_decimals):
        c = KrakenClient.__new__(KrakenClient)
        c.pair_info = lambda pair: {"pair_decimals": pair_decimals}
        sent = {}
        c._private = lambda method, data: sent.update(data) or {"ok": True}
        return c, sent

    def test_limit_price_respects_pair_decimals(self):
        cases = (
            (2, "HYPEUSD", "sell", 1.23456789, 53.839 * 0.995, "53.57"),
            (5, "XBTUSD", "buy", 1.0, 0.123456789, "0.12346"),
        )
        for decimals, pair, side, volume, price, expected in cases:
            with self.subTest(pair=pair, decimals=decimals):
                c, sent = self._client(decimals)
                c.add_order(pair, side, volume, price, ordertype="limit")
                self.assertEqual(sent["price"], expected)

    def test_price_decimals_fallback(self):
        for label, pair_info in (
            ("missing", lambda pair: None),
            ("exception", Mock(side_effect=RuntimeError("API down"))),
        ):
            with self.subTest(case=label):
                c = KrakenClient.__new__(KrakenClient)
                c.pair_info = pair_info
                self.assertEqual(c.price_decimals("WHATEVERUSD"), 2)

    def test_market_order_no_price_field(self):
        c, sent = self._client(2)
        c.add_order("HYPEUSD", "sell", 1.0, None, ordertype="market")
        self.assertNotIn("price", sent)

    def test_client_order_id_is_validated_and_sent(self):
        c, sent = self._client(2)
        client_id = "0123456789ABCDEF0123456789ABCDEF"
        c.add_order(
            "HYPEUSD", "buy", 1.0, 60.0, ordertype="limit",
            cl_ord_id=client_id,
        )
        self.assertEqual(sent["cl_ord_id"], client_id.lower())

        with self.assertRaisesRegex(ValueError, "128-bit hex UUID"):
            c.add_order("HYPEUSD", "buy", 1.0, 60.0, cl_ord_id="too-short")


_KRAKEN_EMPTY = (200, b'{"error":[],"result":{}}')
_KRAKEN_FULL = (200, b'{"error":[],"result":{"HYPEUSD":{"ordermin":"0.1"}}}')


class PublicCacheTest(unittest.TestCase):
    def setUp(self):
        kc._CACHE.clear()

    def test_empty_not_cached_refetches_then_caches_full(self):
        with patch.object(kc, "http_get") as mget:
            mget.side_effect = [_KRAKEN_EMPTY, _KRAKEN_FULL]
            client = kc.KrakenClient()
            r1 = client._public("AssetPairs", {"pair": "HYPEUSD"})
            self.assertEqual(r1, {})
            r2 = client._public("AssetPairs", {"pair": "HYPEUSD"})
            self.assertIn("HYPEUSD", r2)
            self.assertEqual(mget.call_count, 2)
            r3 = client._public("AssetPairs", {"pair": "HYPEUSD"})
            self.assertIn("HYPEUSD", r3)
            self.assertEqual(mget.call_count, 2)

    def test_pair_info_none_on_empty_then_recovers(self):
        with patch.object(kc, "http_get") as mget:
            mget.side_effect = [_KRAKEN_EMPTY, _KRAKEN_FULL]
            client = kc.KrakenClient()
            self.assertIsNone(client.pair_info("HYPEUSD"))
            info = client.pair_info("HYPEUSD")
            self.assertEqual(info.get("ordermin"), "0.1")

    def test_ohlc_closes_excludes_potentially_incomplete_last_candle(self):
        response = {
            "HYPEUSD": [
                [1, "9", "11", "8", "10", "10", "1", 1],
                [14401, "10", "12", "9", "11", "11", "1", 1],
                [28801, "11", "99", "1", "42", "42", "1", 1],
            ],
            "last": 28801,
        }
        client = kc.KrakenClient()
        with patch.object(client, "_public", return_value=response):
            closes, last_closed_at = client.ohlc_closes_with_timestamp("HYPEUSD", 240)
            all_closes, timestamps = client.ohlc_closes_with_timestamps("HYPEUSD", 240)

        self.assertEqual(closes, [10.0, 11.0])
        self.assertEqual(last_closed_at, 14401 + 240 * 60)
        self.assertEqual(all_closes, closes)
        self.assertEqual(timestamps, (14401.0, 28801.0))

    def test_ohlc_cache_ttl_is_interval_aware(self):
        self.assertEqual(kc._read_ttl("OHLC", {"interval": 1}), 30.0)
        self.assertEqual(kc._read_ttl("OHLC", {"interval": 5}), 150.0)
        self.assertEqual(kc._read_ttl("OHLC", {"interval": 240}), 900.0)

    def test_cache_prunes_expired_and_enforces_hard_cap(self):
        with patch.object(kc, "_CACHE_MAX", 2), \
             patch.object(kc.time, "time", return_value=100.0):
            kc._CACHE[("Ticker", (("pair", "OLD"),))] = (99.0, {"old": 1})
            kc._cache_put("Ticker", {"pair": "A"}, 3.0, {"a": 1})
            kc._cache_put("Ticker", {"pair": "B"}, 4.0, {"b": 1})
            kc._cache_put("Ticker", {"pair": "C"}, 5.0, {"c": 1})
        self.assertEqual(len(kc._CACHE), 2)
        self.assertNotIn(("Ticker", (("pair", "OLD"),)), kc._CACHE)

    def test_private_read_retries_on_invalid_nonce_and_succeeds(self):
        client = kc.KrakenClient(api_key="key", api_secret=base64.b64encode(b"secret").decode())
        fail_nonce = (200, b'{"error":["EAPI:Invalid nonce"],"result":{}}')
        success_bal = (200, b'{"error":[],"result":{"HYPE":"7.54"}}')
        with patch.object(kc, "http_post_form") as mpost, \
             patch.object(kc.time, "sleep") as msleep:
            mpost.side_effect = [fail_nonce, success_bal]
            res = client._private("Balance")
            self.assertEqual(res, {"HYPE": "7.54"})
            self.assertEqual(mpost.call_count, 2)
            self.assertEqual(msleep.call_count, 1)

    def test_private_write_does_not_retry_on_invalid_nonce(self):
        client = kc.KrakenClient(api_key="key", api_secret=base64.b64encode(b"secret").decode())
        fail_nonce = (200, b'{"error":["EAPI:Invalid nonce"],"result":{}}')
        with patch.object(kc, "http_post_form") as mpost:
            mpost.return_value = fail_nonce
            with self.assertRaises(kc.KrakenError) as ctx:
                client._private("AddOrder", {"pair": "HYPEUSD"})
            self.assertIn("Invalid nonce", str(ctx.exception))
            self.assertEqual(mpost.call_count, 1)

    def test_kraken_cachemanager_loads_mandatory_env_stack(self):
        cfg = KrakenProvider.resolve_cache_daemon_config()
        self.assertTrue(len(cfg["pairs"]) > 0)
        self.assertIn("HYPEUSD", cfg["pairs"])
        self.assertGreater(cfg["poll_interval"], 0.0)
        self.assertGreater(cfg["lookback_sec"], 0.0)

    def test_wait_for_listing_suppresses_alert_on_transient_network_recovery(self):
        client = kc.KrakenClient()
        with patch.object(kc, "http_get", side_effect=[(0, b""), _KRAKEN_FULL]), \
             patch.object(kb.time, "sleep"), \
             patch.object(kb, "notify") as mnotify:
            ok = kb._wait_for_listing(client, "HYPEUSD", "Kraken", 15, False)
            self.assertTrue(ok)
            mnotify.assert_not_called()

        kc._CACHE.clear()
        unknown_pair = (200, b'{"error":["EQuery:Unknown asset pair"],"result":{}}')
        ticker_ok = (200, b'{"error":[],"result":{"HYPEUSD":{"c":["42.5","1.0"]}}}')
        with patch.object(kc, "http_get", side_effect=[unknown_pair, _KRAKEN_FULL, ticker_ok]), \
             patch.object(kb.time, "sleep"), \
             patch.object(kb, "notify") as mnotify:
            ok = kb._wait_for_listing(client, "HYPEUSD", "Kraken", 15, False)
            self.assertTrue(ok)
            mnotify.assert_called_once()
            self.assertEqual(mnotify.call_args.kwargs["title"], "HYPEUSD listed on Kraken!")


# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_aggregate_fills_by_order_id_window_reference_krakenprovider_get_orders_hyperliquidprovider_get_orders_with_nan_iter26_s4():
    """[ITER26 #4] aggregate_fills_by_order_id, window_reference, KrakenProvider.get_orders, HyperliquidProvider.get_orders with NaN."""
    # 4. aggregate_fills_by_order_id, window_reference, KrakenProvider.get_orders, HyperliquidProvider.get_orders with NaN
    rows = [{"orderId": "1", "side": "SELL", "price": 100.0, "qty": 1.0, "timestamp": 1_700_000_000_000}]
    assert aggregate_fills_by_order_id(rows, cutoff_ms=float("nan")) == []

    called_get_orders = []
    dummy_prov = SimpleNamespace(
        name="kraken",
        get_orders=lambda sym, side, win: (called_get_orders.append(win), rows)[-1],
    )
    assert order_guard.window_reference(dummy_prov, "TAOUSD", "BUY", float("nan")) is None
    assert called_get_orders == []

    kp = KrakenProvider()
    assert kp.get_orders("TAOUSD", "BUY", float("nan")) == []
    assert kp.get_orders("TAOUSD", "BUY", -10.0) == []

    hp = HyperliquidProvider()
    assert hp.get_orders("HYPEUSDC", "BUY", float("nan")) == []
    assert hp.get_orders("HYPEUSDC", "BUY", -5.0) == []

def test_krakenprovider_min_order_qty_free_balance_preflight_order_cancel_order_by_id_pair_precision_iter40_s3():
    """[ITER40 #3] KrakenProvider min_order_qty, free_balance, preflight_order, cancel_order_by_id, pair_precision."""
    # 3. KrakenProvider min_order_qty, free_balance, preflight_order, cancel_order_by_id, pair_precision
    kp = KrakenProvider(client=SimpleNamespace(
        pair_info=lambda sym: {"ordermin": True, "pair_decimals": True, "lot_decimals": 2, "base": "HYPE"},
        balance=lambda: {"HYPE": True},
        cancel_order=lambda oid: {"count": True},
    ))
    assert kp.min_order_qty("HYPEUSD") == 0.0
    assert kp.free_balance("HYPE") is None
    with pytest.raises(ProviderError):
        kp.preflight_order("HYPEUSD", "SELL", True, price=20.0)
    with pytest.raises(ProviderError):
        kp.cancel_order("HYPEUSD", "OID1")
    with pytest.raises(ProviderError):
        kp.pair_precision("HYPEUSD")

def test_kraken_client__read_ttl_last_price_price_decimals_add_order_ohlc_closes_with_timestamps_iter48_s1(monkeypatch):
    """[ITER48 #1] kraken_client: _read_ttl, last_price, price_decimals, add_order, ohlc_closes_with_timestamps."""
    # 1. kraken_client: _read_ttl, last_price, price_decimals, add_order, ohlc_closes_with_timestamps
    assert kc._read_ttl("OHLC", {"interval": float("nan")}) == 30.0
    assert kc._read_ttl("OHLC", {"interval": True}) == 30.0
    cli = kc.KrakenClient.public()
    monkeypatch.setattr(cli, "ticker", lambda _pair: {"c": [float("nan"), "1.0"]})
    assert cli.last_price("HYPEUSD") is None
    monkeypatch.setattr(cli, "ticker", lambda _pair: {"c": [True, "1.0"]})
    assert cli.last_price("HYPEUSD") is None
    monkeypatch.setattr(cli, "pair_info", lambda _pair: {"pair_decimals": True})
    assert cli.price_decimals("HYPEUSD") == 2
    with pytest.raises(ValueError):
        cli.add_order("HYPEUSD", "buy", True, 20.0)
    with pytest.raises(ValueError):
        cli.add_order("HYPEUSD", "buy", 1.0, float("nan"), ordertype="limit")
    with pytest.raises(ValueError):
        cli.ohlc_closes_with_timestamps("HYPEUSD", True)

def test_kraken_cachemanager__ledger_to_fills_rejects_nan_bool_zero_amounts_and_iter48_s2():
    """[ITER48 #2] KrakenProvider.ledger_to_fills rejects NaN/bool/zero amounts and fees."""
    # 2. KrakenProvider.ledger_to_fills rejects NaN/bool/zero amounts and fees
    bad_ledger = {
        "L1": {"refid": "R1", "type": "trade", "asset": "HYPE", "amount": float("nan"), "time": 1700000000},
        "L2": {"refid": "R1", "type": "trade", "asset": "ZUSD", "amount": "-100.0", "time": 1700000000},
        "L3": {"refid": "R2", "type": "trade", "asset": "HYPE", "amount": True, "time": 1700000000},
        "L4": {"refid": "R2", "type": "trade", "asset": "ZUSD", "amount": "-100.0", "time": 1700000000},
        "L5": {"refid": "R3", "type": "trade", "asset": "HYPE", "amount": "2.0", "fee": True, "time": 1700000000},
        "L6": {"refid": "R3", "type": "trade", "asset": "ZUSD", "amount": "-40.0", "fee": float("nan"), "time": 1700000000},
    }
    fills = KrakenProvider.ledger_to_fills(bad_ledger)
    assert len(fills) == 1
    assert fills[0]["id"] == "R3"
    assert fills[0]["price"] == 20.0
    assert "fee" not in fills[0]

def test_kraken_xstock_watch__bot_alive_rejects_boolean_true_pid_systemd_and_iter48_s3():
    """[ITER48 #3] kraken_xstock_watch: _bot_alive rejects boolean True (PID 1 systemd) and <= 1."""
    # 3. kraken_xstock_watch: _bot_alive rejects boolean True (PID 1 systemd) and <= 1
    assert kxw._bot_alive(True) is False
    assert kxw._bot_alive(1) is False
    assert kxw._bot_alive(-1) is False
    assert kxw._bot_alive(0) is False

def test_krakenprovider_get_price_history_open_orders_order_status_iter51_s5(monkeypatch):
    """[ITER51 #5] KrakenProvider: get_price_history, open_orders, order_status."""
    # 5. KrakenProvider: get_price_history, open_orders, order_status
    kp = KrakenProvider()
    fake_kcli = SimpleNamespace(
        _public=lambda ep, params: {"XXBTZUSD": [[1700000000, "1", "1", "1", "nan"], [True, "1", "1", "1", "50000"]]},
        open_orders=lambda: {
            "O1": {"descr": {"pair": "XBTUSD", "type": "buy", "price": True}, "vol": "1.0", "vol_exec": "0.0"}
        },
        query_orders=lambda oid: {
            oid: {"status": "closed", "vol_exec": "nan", "cost": "100.0", "fee": "0.1", "vol": "1.0"}
        },
    )
    monkeypatch.setattr(kp, "_client", lambda: fake_kcli)
    assert kp.get_price_history("BTCUSD", True) is None
    assert kp.get_price_history("BTCUSD", float("nan")) is None
    with pytest.raises(ProviderError):
        kp.open_orders("BTCUSD")
    with pytest.raises(ProviderError):
        kp.order_status("BTCUSD", "O1")

def test_kraken_market_data_fetch_ohlc_candles_rejects_bool_nan_inf_interval_iter52_s2():
    """[ITER52 #2] kraken/market_data: fetch_ohlc_candles rejects bool/NaN/Inf/<=0 interval and timeout."""
    # 2. kraken/market_data: fetch_ohlc_candles rejects bool/NaN/Inf/<=0 interval and timeout
    with pytest.raises(ValueError):
        kmd.fetch_ohlc_candles("HYPEUSD", True)
    with pytest.raises(ValueError):
        kmd.fetch_ohlc_candles("HYPEUSD", 60, timeout=True)
    with pytest.raises(ValueError):
        kmd.fetch_ohlc_candles("HYPEUSD", float("nan"))
    with pytest.raises(ValueError):
        kmd.fetch_ohlc_candles("HYPEUSD", 0)

def test_kraken_kraken_bot__cmd_balance_ignores_bool_nan_inf_balances_iter52_s4(monkeypatch):
    """[ITER52 #4] kraken/kraken_bot._cmd_balance ignores bool/NaN/Inf balances without crashing."""
    # 4. kraken/kraken_bot._cmd_balance ignores bool/NaN/Inf balances without crashing
    logged_lines = []
    monkeypatch.setattr(kb, "log", lambda msg: logged_lines.append(msg))
    fake_kcli = SimpleNamespace(
        balance=lambda: {"ZUSD": "100.5", "BAD_BOOL": True, "BAD_NAN": "nan", "BAD_INF": "inf", "ZERO": "0"}
    )
    assert kb._cmd_balance(fake_kcli) == 0
    assert any("ZUSD" in ln for ln in logged_lines)
    assert not any("BAD_" in ln for ln in logged_lines)

def test_kraken_kraken_xstock_watch_check_levels_ignores_bool_nan_inf_alloc_price_iter52_s5():
    """[ITER52 #5] kraken/kraken_xstock_watch.check_levels ignores bool/NaN/Inf alloc_price, tp_pct, sl_pct, price."""
    # 5. kraken/kraken_xstock_watch.check_levels ignores bool/NaN/Inf alloc_price, tp_pct, sl_pct, price
    xw_st = {"allocated": {"asset": "SPCX", "qty": True}, "pair": "SPCXUSD", "alerted_tp": False, "alerted_sl": False}
    kxw.check_levels(
        SimpleNamespace(last_price=lambda _p: 150.0),
        xw_st,
        alloc_price=float("nan"),
        tp_pct=10.0,
        sl_pct=10.0,
        yahoo_sym="",
        desktop=False,
    )
    assert xw_st["alerted_tp"] is False

def test_kraken_shadow_live__load_history_filters_out_bool_nan_inf_iter52_s6(monkeypatch, tmp_path):
    """[ITER52 #6] kraken/shadow_live._load_history filters out bool/NaN/Inf/non-positive bars."""
    # 6. kraken/shadow_live._load_history filters out bool/NaN/Inf/non-positive bars
    monkeypatch.setattr(k_sl, "LOG_DIR", str(tmp_path))
    hist_file = tmp_path / "HYPEUSD_240m.ohlc.json"
    hist_file.write_text(
        json.dumps([
            [1700000000, 20.0, 21.0, 19.5, 20.5],
            [True, 20.0, 21.0, 19.5, 20.5],
            [1700014400, -1.0, 21.0, 19.5, 20.5],
        ]),
        encoding="utf-8",
    )
    loaded_bars = k_sl._load_history("HYPEUSD", 240)
    assert loaded_bars == [(1700000000, 20.0, 21.0, 19.5, 20.5)]

def test_krakenprovider_pair_precision_rejects_nan_non_positive_ordermin_iter58_s4():
    """[ITER58 #4] KrakenProvider.pair_precision rejects NaN/non-positive ordermin."""
    # 4. KrakenProvider.pair_precision rejects NaN/non-positive ordermin
    kp = KrakenProvider(client=MagicMock())
    kp._cli.pair_info.return_value = {"pair_decimals": 2, "lot_decimals": 4, "ordermin": float("nan"), "base": "HYPE"}
    with pytest.raises(ProviderError):
        kp.pair_precision("HYPEUSD")

def test_krakenclient_ohlc_closes_with_timestamps_rejects_inf_nan_interval_cleanly_with_iter59_s4():
    """[ITER59 #4] KrakenClient.ohlc_closes_with_timestamps rejects inf/nan interval cleanly with ValueError."""
    # 4. KrakenClient.ohlc_closes_with_timestamps rejects inf/nan interval cleanly with ValueError
    kc = KrakenClient.__new__(KrakenClient)
    with pytest.raises(ValueError, match="positive integer"):
        kc.ohlc_closes_with_timestamps("XBTUSD", float("inf"))

def test_kraken_xstock_watch__load_state_merges_partial_json_onto_default_keys_iter60_s3(tmp_path, monkeypatch):
    """[ITER60 #3] kraken_xstock_watch._load_state merges partial JSON onto default keys without KeyError."""
    # 3. kraken_xstock_watch._load_state merges partial JSON onto default keys without KeyError
    state_file = tmp_path / "xstock_partial.json"
    state_file.write_text(json.dumps({"known_assets": "corrupt"}), encoding="utf-8")
    monkeypatch.setenv("XSTOCK_STATE_FILE", str(state_file))
    loaded = kxw._load_state()
    assert loaded["known_assets"] == {}
    assert loaded["alerted_pair"] is False
    assert loaded["alerted_tp"] is False

def test_krakenclient__parse_rejects_non_dict_json_payloads_cleanly_iter68_s1():
    """[ITER68 #1] KrakenClient._parse rejects non-dict JSON payloads cleanly and add_order rejects None/non-numeric volume/price with ValueError."""
    # 1. KrakenClient._parse rejects non-dict JSON payloads cleanly and add_order rejects None/non-numeric volume/price with ValueError
    with pytest.raises(KrakenError):
        KrakenClient._parse(200, b"[1, 2, 3]")
    kc = KrakenClient("k", "c2VjcmV0")
    with pytest.raises(ValueError):
        kc.add_order("XBTUSD", "buy", None, 50000.0)
    with pytest.raises(ValueError):
        kc.add_order("XBTUSD", "buy", 0.1, "not-a-price", ordertype="limit")
    with pytest.raises(ValueError):
        kc.ohlc_closes_with_timestamps("XBTUSD", 15.5)

def test_kraken_market_data_pair_precision_handles_float_strings_and_overflowerror_iter68_s2():
    """[ITER68 #2] kraken.market_data.pair_precision handles float strings and OverflowError."""
    # 2. kraken.market_data.pair_precision handles float strings and OverflowError
    assert pair_precision({"pair_decimals": "2.0", "lot_decimals": "4.0", "ordermin": "0.01"}) == (2, 4, 0.01)
    assert pair_precision({"pair_decimals": 10**400, "lot_decimals": 10**400, "ordermin": 10**400}) == (2, 8, 0.0)

def test__ledger_to_fills_tolerates_non_dict_ledger_or_non_dict_iter68_s3():
    """[ITER68 #3] KrakenProvider.ledger_to_fills tolerates non-dict ledger or non-dict entries."""
    # 3. KrakenProvider.ledger_to_fills tolerates non-dict ledger or non-dict entries
    assert KrakenProvider.ledger_to_fills(None) == []
    assert KrakenProvider.ledger_to_fills({"L1": "not-a-dict"}) == []

def test_kraken_xstock_watch_check_pairs_tolerates_non_dict_entries_and_missing_iter70_s3():
    """[ITER70 #3] kraken_xstock_watch.check_pairs tolerates non-dict entries and missing alerted_pair in state dict."""
    k_watch = load_venue_modules("kraken", "kraken_xstock_watch")
    # 3. kraken_xstock_watch.check_pairs tolerates non-dict entries and missing alerted_pair in state dict
    k_watch.notify = lambda **kw: None
    dummy_k = SimpleNamespace(
        asset_pairs=lambda: {"BAD": "not-a-dict", "SPCXXUSD": {"wsname": "SPCXx/USD", "base": "SPCXx"}},
        last_price=lambda k: 50.0,
    )
    st_watch = {}
    k_watch.check_pairs(dummy_k, st_watch, "SPCX", desktop=False, quote="USD")
    assert st_watch.get("pair") == "SPCXXUSD"
    assert st_watch.get("alerted_pair") is True

def test_kraken_t212_continuous_grid_load_handles_float_timestamps_and_iter74_s4(tmp_path):
    """[ITER74 #4] kraken/t212 continuous_grid.load handles float timestamps and skips non-positive/NaN OHLC."""
    # 4. kraken/t212 continuous_grid.load handles float timestamps and skips non-positive/NaN OHLC
    k_csv = tmp_path / "XXBTZUSD.csv"
    k_csv.write_text("timestamp,open,high,low,close,volume\n1000.0,100,105,95,102,1\n1060.0,0,105,95,102,1\n", encoding="utf-8")
    assert len(k_grid.load(str(k_csv))) == 1

    t_csv = tmp_path / "VUSA_EQ_1h.csv"
    t_lines = ["timestamp,open,high,low,close,volume"] + [
        f"{2000 + i * 60}.0,50,55,48,52,1" for i in range(35)
    ] + ["5000.0,50,nan,48,52,1"]
    t_csv.write_text("\n".join(t_lines) + "\n", encoding="utf-8")
    orig_t_dir = t_grid.DATA
    try:
        t_grid.DATA = str(tmp_path)
        t_grid._LOAD_CACHE.pop(("VUSA_EQ", "1h"), None)
        assert len(t_grid.load("VUSA_EQ", "1h")) == 35
    finally:
        t_grid.DATA = orig_t_dir

def test_krakenprovider_rejects_boolean_client_order_id_and_invalid_order_sides_iter87_s1():
    """[ITER87 #1] KrakenProvider rejects boolean client_order_id and invalid order sides in place_order / submit_order & HyperliquidProvider rejects boolean client_order_id and invalid order sides in place_order / submit_order."""
    # 1. KrakenProvider rejects boolean client_order_id and invalid order sides in place_order / submit_order
    with pytest.raises(ProviderError, match="client_order_id"):
        KrakenProvider.derive_client_order_id(True)
    kp = KrakenProvider()
    for bad_side in ("", None, "HOLD", "bad", True):
        assert kp.place_order("HYPEUSD", bad_side, 25.0, 1.0) is None
        with pytest.raises(SubmissionRefused, match="invalid side"):
            kp.submit_order("HYPEUSD", bad_side, 1.0, price=25.0)


    # 2. HyperliquidProvider rejects boolean client_order_id and invalid order sides in place_order / submit_order
    with pytest.raises(ProviderError, match="client_order_id"):
        HyperliquidProvider.derive_client_order_id(False)
    hp = HyperliquidProvider(live_enabled=True)
    for bad_side in ("", None, "HOLD", "bad", True):
        assert hp.place_order("HYPEUSDC", bad_side, 25.0, 1.0) is None
        with pytest.raises(SubmissionRefused, match="invalid side"):
            hp.submit_order("HYPEUSDC", bad_side, 1.0, price=25.0)

def test_krakenclient_add_order_validates_side_ordertype_and_mandatory_limit_iter90_s1():
    """[ITER90 #1] KrakenClient.add_order validates side, ordertype, and mandatory limit price."""
    # 1. KrakenClient.add_order validates side, ordertype, and mandatory limit price
    kc = KrakenClient("key", "c2VjcmV0")
    with pytest.raises(ValueError, match="invalid Kraken order side"):
        kc.add_order("HYPEUSD", "HOLD", 1.0, price=25.0)
    with pytest.raises(ValueError, match="invalid Kraken ordertype"):
        kc.add_order("HYPEUSD", "buy", 1.0, price=25.0, ordertype="stop")
    with pytest.raises(ValueError, match="invalid Kraken limit price"):
        kc.add_order("HYPEUSD", "buy", 1.0, price=None, ordertype="limit")

def test_kraken_cachemanager__normalize_does_not_fabricate_sell_when_trade_iter90_s2():
    """[ITER90 #2] KrakenProvider.normalize_ws_trade does not fabricate SELL when trade side is missing."""
    # 2. KrakenProvider.normalize_ws_trade does not fabricate SELL when trade side is missing
    norm = KrakenProvider.normalize_ws_trade("T1", {"pair": "HYPEUSD", "price": "25.0", "vol": "1.0", "time": 1700000000})
    assert norm["side"] is None
    assert norm["isBuyer"] is None

def test_kraken_market_data_fetch_ohlc_candles_check_pair_listing_pair_precision_iter91_s1():
    """[ITER91 #1] kraken.market_data.fetch_ohlc_candles, check_pair_listing, pair_precision."""
    # 1. kraken.market_data.fetch_ohlc_candles, check_pair_listing, pair_precision
    with pytest.raises(ValueError, match="invalid pair"):
        kmd.fetch_ohlc_candles("", 5)
    with pytest.raises(ValueError, match="invalid pair"):
        kmd.fetch_ohlc_candles(True, 5)

    info, unlisted = kmd.check_pair_listing(
        SimpleNamespace(pair_info=lambda pair: {"status": None, "pair_decimals": 3.5, "lot_decimals": -2, "ordermin": "0.1"}),
        "HYPEUSD",
    )
    assert info is not None and unlisted is False
    p_dec, v_dec, omin = kmd.pair_precision(info)
    assert (p_dec, v_dec, omin) == (2, 8, 0.1)

def test_kraken_kraken_xstock_watch_check_pairs_skips_non_tradable_status_iter92_s4():
    """[ITER92 #4] kraken.kraken_xstock_watch.check_pairs skips non-tradable status."""
    # 4. kraken.kraken_xstock_watch.check_pairs skips non-tradable status
    xw = importlib.import_module("kraken.kraken_xstock_watch")
    st_xw = xw._default_state()
    fake_k = type(
        "FK",
        (),
        {
            "asset_pairs": lambda self: {
                "SPCXXUSD": {"wsname": "SPCXx/USD", "base": "SPCXX", "status": "cancel_only"}
            },
            "last_price": lambda self, p: 200.0,
        },
    )()
    xw.check_pairs(fake_k, st_xw, "SPCX", False, quote="USD")
    assert st_xw["pair"] is None
    assert st_xw["alerted_pair"] is False

def test_krakenclient_validation_on_pair_txid_and_price_decimals_non_iter94_s3(monkeypatch):
    """[ITER94 #3] KrakenClient validation on pair, txid, and price_decimals non-integer rejection."""
    # 3. KrakenClient validation on pair, txid, and price_decimals non-integer rejection

    kc = KrakenClient("k", "s")
    assert kc.pair_info("") is None
    assert kc.pair_info(True) is None
    assert kc.ticker("  ") is None
    monkeypatch.setattr(kc, "pair_info", lambda _p: {"pair_decimals": 3.5})
    assert kc.price_decimals("XXBTZUSD") == 2
    with pytest.raises(ValueError, match="invalid Kraken pair"):
        kc.add_order("  ", "buy", 1.0, price=100.0)
    with pytest.raises(ValueError, match="invalid Kraken txid"):
        kc.cancel_order("")
    with pytest.raises(ValueError, match="invalid Kraken txid"):
        kc.query_orders(True)

def test_krakenprovider_kraken_pair_matches_free_balance_normalization_negative_fee_rejection_non_iter95_s2():
    """[ITER95 #2] KrakenProvider kraken_pair_matches, free_balance normalization, negative fee rejection, non-integer ohlc_series."""
    # 2. KrakenProvider kraken_pair_matches, free_balance normalization, negative fee rejection, non-integer ohlc_series
    assert kraken_pair_matches("", "---") is False
    assert kraken_pair_matches(True, "BTCUSD") is False
    kp = KrakenProvider(
        client=type(
            "KC",
            (),
            {
                "balance": lambda self: {"XXBT": "1.25"},
                "query_orders": lambda self, oid: {
                    oid: {"status": "closed", "vol_exec": "1.0", "cost": "100.0", "fee": "-0.5", "vol": "1.0"}
                },
            },
        )()
    )
    assert kp.free_balance("") is None
    assert kp.free_balance(True) is None
    assert kp.free_balance(" btc ") == pytest.approx(1.25)
    assert kp.free_balance("xbt") == pytest.approx(1.25)
    with pytest.raises(ProviderError, match="invalid fill values"):
        kp.order_status("XXBTZUSD", "OID-1")
    with pytest.raises(ProviderError, match="invalid interval_min"):
        kp.ohlc_series("XXBTZUSD", 60.5)

def test_krakenprovider_iter111_s1():
    """[ITER111 #1] KrakenProvider."""
    # 1. KrakenProvider
    assert kraken_provider.kraken_pair_key(True) == ""
    assert kraken_provider.kraken_pair_key(None) == ""
    norm = kraken_provider._normalize_kraken_fill({"side": "BUY", "price": 100.0, "qty": 1.0, "timestamp": 1000, "ordertxid": True})
    assert "orderId" not in norm

    k_prov = kraken_provider.KrakenProvider.__new__(kraken_provider.KrakenProvider)
    k_prov.preflight_order("XXBTZUSD", True, 1.0)
    with pytest.raises(ProviderError):
        k_prov.order_status("XXBTZUSD", True)
    with pytest.raises(ProviderError):
        k_prov.order_status("XXBTZUSD", "   ")
    with pytest.raises(ProviderError):
        k_prov.cancel_order("XXBTZUSD", True)
    with pytest.raises(ProviderError):
        k_prov.cancel_order("XXBTZUSD", "")

    k_prov._client = lambda: SimpleNamespace(
        pair_info=lambda _s: {"pair_decimals": 2, "lot_decimals": 4, "ordermin": 0.01, "base": "   "}
    )
    with pytest.raises(ProviderError):
        k_prov.pair_precision("XXBTZUSD")

def test_krakenclient_ohlc_closes_with_timestamps_and_kraken_market_data_iter115_s2():
    """[ITER115 #2] KrakenClient.ohlc_closes_with_timestamps and kraken.market_data."""
    # 2. KrakenClient.ohlc_closes_with_timestamps and kraken.market_data
    kc = KrakenClient(api_key="", api_secret="")
    with pytest.raises(ValueError):
        kc.ohlc_closes_with_timestamps("   ", 60)
    with pytest.raises(ValueError):
        kc.ohlc_closes_with_timestamps("XBTUSD", 60.0)
    with pytest.raises(ValueError):
        kc.ohlc_closes_with_timestamps("XBTUSD", "60.0")
    with pytest.raises(ValueError):
        kmd.fetch_ohlc_candles("XBTUSD", 60.0)
    assert kmd.pair_precision({"pair_decimals": 999, "lot_decimals": 999, "ordermin": "0.1"}) == (2, 8, 0.1)

def test_kraken_cachemanager__normalize__load_initial_cache_and__ledger_to_fills_iter115_s3(tmp_path):
    """[ITER115 #3] KrakenProvider.normalize_ws_trade, CacheTradeManager.load_sanitized_trade_snapshot, and KrakenProvider.ledger_to_fills."""
    # 3. KrakenProvider.normalize_ws_trade, CacheTradeManager.load_sanitized_trade_snapshot, and KrakenProvider.ledger_to_fills
    norm = KrakenProvider.normalize_ws_trade("tx1", {"pair": "XBTUSD", "ordertxid": True, "order_id": "OID-1", "price": "100", "vol": "1", "time": 1700000000})
    assert norm["orderId"] == "OID-1"
    cache_file = tmp_path / "kraken_trades.json"
    cache_file.write_text('{"items":{"XBTUSD":[{"id":true,"time":100},{"id":"   ","time":100},{"id":"t1","time":0},{"id":" t2 ","time":1700000000000}]},"fetchtime":{"XBTUSD":100}}')
    loaded = CacheTradeManager.load_sanitized_trade_snapshot(str(cache_file), venue="kraken")
    assert [r["id"] for r in loaded["items"]["XBTUSD"]] == ["t2"]
    fills = KrakenProvider.ledger_to_fills({
        "l1": {"refid": True, "asset": "XXBT", "type": "trade", "amount": "1", "time": 1700000000},
        "l2": {"refid": True, "asset": "ZUSD", "type": "trade", "amount": "-100", "time": 1700000000},
        "l3": {"refid": " R1 ", "asset": "XXBT", "type": " TRADE ", "amount": "0.5", "time": 1700000000},
        "l4": {"refid": "R1", "asset": "ZUSD", "type": "TRADE", "amount": "-50", "time": 1700000000},
    })
    assert len(fills) == 1 and fills[0]["orderId"] == "R1"

def test_krakenclient_pair_info_ticker_and_add_order_validate_false_iter116_s1(monkeypatch):
    """[ITER116 #1] KrakenClient.pair_info, ticker, and add_order validate='false'."""
    # 1. KrakenClient.pair_info, ticker, and add_order validate="false"
    kc = KrakenClient(api_key="k", api_secret="c2VjcmV0")
    assert kc.pair_info(123) is None
    assert kc.ticker(123) is None
    captured_data = {}
    monkeypatch.setattr(kc, "_private", lambda ep, data=None: captured_data.update(data or {}) or {"txid": ["T1"]})
    monkeypatch.setattr(kc, "price_decimals", lambda _p: 2)
    kc.add_order("XBTUSD", "buy", 1.0, price=100.0, ordertype="limit", validate="false")
    assert "validate" not in captured_data
    captured_data.clear()
    kc.add_order("XBTUSD", "buy", 1.0, price=100.0, ordertype="limit", validate="true")
    assert captured_data.get("validate") == "true"

def test_kraken_cachemanager__match_configured_pair__normalize_and__ledger_to_fills_iter116_s2():
    """[ITER116 #2] KrakenProvider.match_configured_pair, normalize_ws_trade, and ledger_to_fills."""
    pairs = KrakenProvider.resolve_cache_daemon_config()["pairs"]
    # 2. KrakenProvider.match_configured_pair, normalize_ws_trade, and ledger_to_fills
    assert KrakenProvider.match_configured_pair(True, pairs=pairs) is None
    assert KrakenProvider.match_configured_pair(123, pairs=pairs) is None
    assert KrakenProvider.normalize_ws_trade(None, {"pair": "XBTUSD"}) == {}
    assert KrakenProvider.normalize_ws_trade(True, {"pair": "XBTUSD"}) == {}
    assert KrakenProvider.normalize_ws_trade("tx1", "not-a-dict") == {}
    assert KrakenProvider.normalize_ws_trade(" tx1 ", {"pair": True})["symbol"] is None
    assert KrakenProvider.ledger_to_fills({
        "l1": {"refid": "R1", "asset": True, "type": "trade", "amount": "1", "time": 1700000000},
        "l2": {"refid": "R1", "asset": "ZUSD", "type": "trade", "amount": "-100", "time": 1700000000},
    }) == []

def test_kraken_market_data_fetch_ohlc_candles_rejects_float_interval_with_invalid_iter117_s1():
    """[ITER117 #1] kraken.market_data.fetch_ohlc_candles rejects float interval with 'invalid interval'."""
    # 1. kraken.market_data.fetch_ohlc_candles rejects float interval with "invalid interval"
    with pytest.raises(ValueError, match="invalid interval"):
        kmd.fetch_ohlc_candles("XXBTZUSD", 1.5)

def test_marketapi_krakenprovider_and_t212provider_reject_nan_inf_non_iter122_s3():
    """[ITER122 #3] MarketApi, KrakenProvider, and T212Provider reject NaN/Inf/non-integer order_id."""
    # 3. MarketApi, KrakenProvider, and T212Provider reject NaN/Inf/non-integer order_id
    for bad_id in (float("nan"), float("inf"), "nan", "INF", "null", 12.5, {"bad": 1}):
        with pytest.raises(ProviderError):
            api.order_status("BTCUSDC", bad_id, provider_name="binance")
        with pytest.raises(ProviderError):
            api.cancel_order("BTCUSDC", bad_id, provider_name="binance")
    kp = KrakenProvider()
    tp = T212Provider()
    for bad_str in ("nan", "inf", "none", "null"):
        with pytest.raises(ProviderError):
            kp.order_status("HYPEUSD", bad_str)
        with pytest.raises(ProviderError):
            kp.cancel_order("HYPEUSD", bad_str)
        with pytest.raises(ProviderError):
            tp.order_status("AAPL_US_EQ", bad_str)
        with pytest.raises(ProviderError):
            tp.cancel_order("AAPL_US_EQ", bad_str)

def test_krakenclient_cancel_order_and_query_orders_reject_sentinel_strings_nan_iter135_s1():
    """[ITER135 #1] KrakenClient.cancel_order and query_orders reject sentinel strings ('nan', 'null') and containers/floats."""
    # 1. KrakenClient.cancel_order and query_orders reject sentinel strings ("nan", "null") and containers/floats
    kc = KrakenClient("k", "c2VjcmV0")
    for bad_txid in ("nan", "null", "none", "inf", float("nan"), ["OX123"], {"a": 1}, True, None):
        with pytest.raises(ValueError, match="invalid Kraken txid"):
            kc.cancel_order(bad_txid)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="invalid Kraken txid"):
            kc.query_orders(bad_txid)  # type: ignore[arg-type]

def test_kraken_cachemanager__normalize_rejects_sentinel_txid_and_normalizes_integer_iter135_s2():
    """[ITER135 #2] KrakenProvider.normalize_ws_trade rejects sentinel txid and normalizes integer float ordertxid."""
    # 2. KrakenProvider.normalize_ws_trade rejects sentinel txid and normalizes integer float ordertxid
    assert KrakenProvider.normalize_ws_trade("nan", {"type": "buy", "price": "10", "vol": "1", "time": 1700000000}) == {}
    assert KrakenProvider.normalize_ws_trade(["T1"], {"type": "buy", "price": "10", "vol": "1", "time": 1700000000}) == {}
    norm = KrakenProvider.normalize_ws_trade(
        " TX-100 ",
        {"type": "buy", "price": "10", "vol": "1", "time": 1700000000, "ordertxid": "nan", "id": 999.0},
    )
    assert norm["id"] == "TX-100"
    assert norm["orderId"] == "999"

def test_kraken_cachemanager__load_initial_cache_filters_out_sentinel_container_fill_ids_iter135_s3(tmp_path):
    """[ITER135 #3] CacheTradeManager.load_sanitized_trade_snapshot filters out sentinel/container fill IDs."""
    # 3. CacheTradeManager.load_sanitized_trade_snapshot filters out sentinel/container fill IDs
    cache_file = str(tmp_path / "cache_trade_kraken.json")
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "items": {
                    "HYPEUSD": [
                        {"id": "nan", "time": 1700000000000},
                        {"id": ["bad"], "time": 1700000000000},
                        {"id": " VALID-1 ", "time": 1700000005000.0},
                    ]
                },
                "fetchtime": {"HYPEUSD": 1700000005000},
            },
            fh,
        )
    loaded = CacheTradeManager.load_sanitized_trade_snapshot(cache_file, venue="kraken")
    assert [r["id"] for r in loaded["items"]["HYPEUSD"]] == ["VALID-1"]

def test_kraken_cachemanager__ledger_to_fills_ignores_entries_with_sentinel_container_refid_iter135_s4():
    """[ITER135 #4] KrakenProvider.ledger_to_fills ignores entries with sentinel/container refid."""
    # 4. KrakenProvider.ledger_to_fills ignores entries with sentinel/container refid
    fills = KrakenProvider.ledger_to_fills(
        {
            "L1": {"refid": "nan", "asset": "HYPE", "amount": "2.0", "type": "trade", "time": 1700000000.0},
            "L2": {"refid": "nan", "asset": "ZUSD", "amount": "-100.0", "type": "trade", "time": 1700000000.0},
            "L3": {"refid": "REF-OK", "asset": "HYPE", "amount": "2.0", "type": "trade", "time": 1700000000.0},
            "L4": {"refid": "REF-OK", "asset": "ZUSD", "amount": "-100.0", "type": "trade", "time": 1700000000.0},
        }
    )
    assert [f["id"] for f in fills] == ["REF-OK"]

def test_derive_client_order_id_normalize_kraken_fill_and_krakenprovider_order_status_whitespace_handling_iter141_s2():
    """[ITER141 #2] derive_client_order_id, _normalize_kraken_fill, and KrakenProvider.order_status whitespace handling."""
    # 2. derive_client_order_id, _normalize_kraken_fill, and KrakenProvider.order_status whitespace handling
    for bad_cid in ("nan", "null", "none", "inf", ["abc"], {"a": 1}, True, None, ""):
        with pytest.raises(ProviderError):
            KrakenProvider.derive_client_order_id(bad_cid)  # type: ignore[arg-type]
        with pytest.raises(ProviderError):
            HyperliquidProvider.derive_client_order_id(bad_cid)  # type: ignore[arg-type]

    nf_bad = _normalize_kraken_fill({"ordertxid": "nan", "price": 20.0, "qty": 1.0, "side": "BUY", "timestamp": 1000})
    assert "orderId" not in nf_bad
    nf_good = _normalize_kraken_fill({"ordertxid": 444.0, "price": 20.0, "qty": 1.0, "side": "BUY", "timestamp": 1000})
    assert nf_good["orderId"] == "444"

    k_cli = SimpleNamespace(
        query_orders=lambda txid: {txid.strip(): {"status": "closed", "vol_exec": "2.0", "cost": "40.0", "fee": "0.1", "vol": "2.0"}},
    )
    kp = KrakenProvider(client=k_cli)
    st = kp.order_status("HYPEUSD", "  OUN337-ABC  ")
    assert st.status == "closed"
    assert st.filled_qty == pytest.approx(2.0)

def test_hyperliquid_strategy_py__sanitize_state_and__clean_oid_kraken_xstock_watch__load_state_iter146_s2(tmp_path, monkeypatch):
    """[ITER146 #2] hyperliquid/strategy.py _sanitize_state and _clean_oid & kraken_xstock_watch._load_state & _bot_alive & 212trading/price_alert.py _parse_armed_flag."""
    # 2. hyperliquid/strategy.py _sanitize_state and _clean_oid
    hl_mod = load_venue_modules("hyperliquid", "strategy")

    for bad_oid in ("nan", "null", "none", "0", 0, -3, ["1"], {"id": 1}, True, None, ""):
        assert hl_mod.clean_order_id(bad_oid) is None
    assert hl_mod.clean_order_id(555.0) == "555"
    assert hl_mod.clean_order_id("555.0") == "555"

    sanitized_hl = hl_mod._sanitize_state(
        {
            "orders": [
                {"oid": "nan", "role": "open", "px": 20.0, "sz": 1.0},
                {"oid": 0, "role": "open", "px": 20.0, "sz": 1.0},
                {"oid": 555.0, "role": "open", "px": 20.0, "sz": 1.0, "dca_counted": "false"},
            ]
        }
    )
    assert len(sanitized_hl["orders"]) == 1
    assert sanitized_hl["orders"][0]["oid"] == 555
    assert sanitized_hl["orders"][0]["dca_counted"] is False


    # 3. kraken_xstock_watch._load_state & _bot_alive
    st_file = tmp_path / "xstock_state.json"
    st_file.write_text(
        json.dumps(
            {
                "known_assets": {"SPCX": "10.5", "BAD": "nan", "ZERO": 0},
                "allocated": "invalid_not_dict",
                "pair": "nan",
                "alerted_pair": "false",
                "alerted_tp": "0",
                "alerted_tp2": "no",
                "alerted_sl": "nan",
                "alerted_need_price": "false",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XSTOCK_STATE_FILE", str(st_file))
    loaded_xw = kxw._load_state()
    assert loaded_xw["known_assets"] == {"SPCX": 10.5}
    assert loaded_xw["allocated"] is None
    assert loaded_xw["pair"] is None
    assert loaded_xw["alerted_pair"] is False
    assert loaded_xw["alerted_tp"] is False
    assert loaded_xw["alerted_tp2"] is False
    assert loaded_xw["alerted_sl"] is False
    assert loaded_xw["alerted_need_price"] is False
    assert kxw._parse_bool_flag(float("nan")) is False
    assert kxw._bot_alive("1234.5") is False
    assert kxw._bot_alive(float("nan")) is False
    assert kxw._bot_alive(["1234"]) is False


    # 4. 212trading/price_alert.py _parse_armed_flag
    pa_mod = load_venue_modules("212trading", "price_alert")
    assert pa_mod._parse_armed_flag("0.0") is False
    assert pa_mod._parse_armed_flag("false") is False
    assert pa_mod._parse_armed_flag("null") is False
    assert pa_mod._parse_armed_flag(float("nan"), default=True) is True
    assert pa_mod._parse_armed_flag("true") is True

def test_kraken_cachemanager_clean_order_id__normalize_and__load_initial_cache_iter147_s1(tmp_path):
    """[ITER147 #1] clean_order_id, KrakenProvider.normalize_ws_trade, and CacheTradeManager.load_sanitized_trade_snapshot."""
    # 1. clean_order_id, KrakenProvider.normalize_ws_trade, and CacheTradeManager.load_sanitized_trade_snapshot
    for bad_id in ("nan", "null", "none", "0", 0, -5, 12.5, ["123"], {"id": 1}, True, None, ""):
        assert clean_order_id(bad_id) is None
        assert KrakenProvider.normalize_ws_trade(bad_id, {"price": "20.0", "qty": "1.0", "side": "BUY", "time": 1700000000}) == {}

    assert clean_order_id(777.0) == "777"
    assert clean_order_id("777.0") == "777"
    norm_ok = KrakenProvider.normalize_ws_trade(777.0, {"price": "20.0", "qty": "1.0", "side": "BUY", "time": 1700000000})
    assert norm_ok["id"] == "777"

    cache_file = tmp_path / "cache_trade_kraken.json"
    cache_file.write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSD": [
                        {"id": "0", "time": 1700000000000, "isBuyer": "false"},
                        {"id": "888.0", "time": 1700000001000, "isBuyer": "false"},
                    ]
                },
                "fetchtime": {"HYPEUSD": 1700000001000},
            }
        ),
        encoding="utf-8",
    )
    loaded_cache = CacheTradeManager.load_sanitized_trade_snapshot(str(cache_file), venue="kraken")
    assert len(loaded_cache["items"]["HYPEUSD"]) == 1
    assert loaded_cache["items"]["HYPEUSD"][0]["id"] == "888"
    assert loaded_cache["items"]["HYPEUSD"][0]["isBuyer"] is False

def test_kraken_shadow_live__get_anchor_handles_integral_float_strings_and_iter147_s2(tmp_path):
    """[ITER147 #2] kraken/shadow_live._get_anchor handles integral float strings and non-finite default_ts."""
    # 2. kraken/shadow_live._get_anchor handles integral float strings and non-finite default_ts
    orig_log_dir = k_sl.LOG_DIR
    k_sl.LOG_DIR = str(tmp_path / "shadow_logs")
    try:
        os.makedirs(k_sl.LOG_DIR, exist_ok=True)
        anchor_p = os.path.join(k_sl.LOG_DIR, f"{k_sl._safe_pair('HYPEUSD')}_240m.anchor")
        with open(anchor_p, "w", encoding="utf-8") as fh:
            fh.write("1700000000.0\n")
        assert k_sl._get_anchor("HYPEUSD", 240, 1600000000) == 1700000000
        assert k_sl._get_anchor("ADAUSD", 240, float("nan")) == 0  # type: ignore[arg-type]
    finally:
        k_sl.LOG_DIR = orig_log_dir

def test_krakenprovider__clean_kraken_order_id_strategy_dry_run_iter152_s2():
    """[ITER152 #2] require_clean_order_id(str_only=True) & KrakenProvider.strategy_dry_run."""
    # 2. require_clean_order_id(str_only=True) & KrakenProvider.strategy_dry_run
    assert krk_bool("false", True) is False
    assert KrakenProvider.strategy_dry_run(False, "false") is True  # type: ignore[arg-type]
    assert KrakenProvider.strategy_dry_run(False, float("nan")) is True  # type: ignore[arg-type]
    for bad_oid in ("", "0", "0.0", "nan", "null", "none", "inf", True, 123, ["O-1"], None):
        with pytest.raises(ProviderError):
            require_clean_order_id(bad_oid, f"order_status {bad_oid}", str_only=True)
    assert require_clean_order_id(" O-KRK-123 ", "order_status O-KRK-123", str_only=True) == "O-KRK-123"

    krk = KrakenProvider()
    with pytest.raises(ProviderError):
        krk.order_status("HYPEUSD", "0")
    with pytest.raises(ProviderError):
        krk.cancel_order("HYPEUSD", "0")

def test_krakenprovider__fills_from_api_none_and_cancel_order_with_count_iter164_s3():
    """[ITER164 #3] KrakenProvider: _fills_from_api(None) and cancel_order with count='1.0'."""
    # 3. KrakenProvider: _fills_from_api(None) and cancel_order with count="1.0"
    kp = KrakenProvider(client=SimpleNamespace(cancel_order=lambda oid: {"count": "1.0"}))
    assert kp._fills_from_api(None) == []
    assert kp._fills_from_api(True) == []
    kp.cancel_order("HYPEUSD", "OQCLML-BW3P3-BUCMWZ")

def test_krakenprovider_open_orders_side_aliases_preflight_order_ask_alias_and_iter169_s3(monkeypatch):
    """[ITER169 #3] KrakenProvider open_orders side aliases, preflight_order ASK alias, and positional force."""
    # 3. KrakenProvider open_orders side aliases, preflight_order ASK alias, and positional force
    fake_cli = SimpleNamespace(
        open_orders=lambda: {
            "OID1": {
                "descr": {"pair": "HYPEUSD", "type": " b ", "price": "20.0"},
                "vol": "1.5",
                "vol_exec": "0.5",
            }
        },
        add_order=lambda symbol, side, qty, price, **kw: {
            "txid": ["OID2"],
            "ordertype": kw.get("ordertype"),
            "validate": kw.get("validate"),
        },
    )
    kr = kraken_provider.KrakenProvider(client=fake_cli)
    monkeypatch.setattr(kr, "execution_enabled", lambda: False)
    monkeypatch.setattr(kr, "round_quantity", lambda sym, q: float(q))
    monkeypatch.setattr(
        kr,
        "pair_precision",
        lambda sym: PairPrecision(price_decimals=2, volume_decimals=2, order_min=0.1, base_asset="HYPE"),
    )
    monkeypatch.setattr(kr, "free_balance", lambda asset: 0.5)
    oo = kr.open_orders("HYPEUSD")
    assert len(oo) == 1 and oo[0]["side"] == "BUY"
    with pytest.raises(ProviderError, match="insufficient funds SELL"):
        kr.preflight_order("HYPEUSD", "ASK", 2.0)
    res_kr = kr.place_order("HYPEUSD", "BID", 20.0, 1.0, True)
    assert res_kr["ordertype"] == "market" and res_kr["validate"] is True
