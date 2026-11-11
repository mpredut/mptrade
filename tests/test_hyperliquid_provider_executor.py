"""Hyperliquid StrategyExecutor contract tests with fake SDK clients."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import pytest
from offline.runners import fetch_hyperliquid_candles as fhc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_HL_DIR = os.path.join(ROOT, "hyperliquid")
if _HL_DIR not in sys.path:
    sys.path.insert(0, _HL_DIR)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from instrument import Instrument  # noqa: E402
import instrument as instrument_module  # noqa: E402
import order_guard  # noqa: E402
from providers import hyperliquid_provider as hl_provider  # noqa: E402
from providers.hyperliquid_provider import HyperliquidProvider  # noqa: E402
from providers.market_api import MarketApi  # noqa: E402
from providers.strategy_executor import (  # noqa: E402
    StrategyExecutor, PairPrecision, ProviderError)
from botcore import load_local_module
from tests.conftest import TempDirTestCase, load_venue_modules
from types import SimpleNamespace


class FakeInfo:
    def __init__(self, fills, status="open"):
        self._fills = fills
        self._status = status

    def user_fills(self, addr):
        return self._fills

    def query_order_by_oid(self, addr, oid):
        if self._status == "unknownOid":
            return {"status": "unknownOid"}
        filled = 0.0
        for item in self._fills:
            try:
                item_oid = int(item.get("oid") or -1)
            except (TypeError, ValueError):
                continue
            if item_oid == oid:
                filled += float(item.get("sz") or 0.0)
        original = max(1.0, filled)
        remaining = 0.0 if self._status == "filled" else original - filled
        return {
            "status": "order",
            "order": {
                "status": self._status,
                "order": {"origSz": str(original), "sz": str(remaining)},
            },
        }


class FakeRead:
    def __init__(self, fills=None, opens=None, status="open", balances=None):
        self.info = FakeInfo(fills or [], status=status)
        self.info.spot_user_state = lambda addr: {
            "balances": balances if balances is not None else [
                {"coin": "USDC", "total": "10000", "hold": "0"},
            ],
        }
        self._opens = opens or []

    def sz_decimals(self, coin):
        return 2

    def open_orders(self, pair):
        return self._opens

    def spot_mid(self, pair):
        return 60.0

    def candles(self, pair, iv, lh):
        return [{"c": "10"}, {"c": "11"}, {"c": "12"}]


class FakeSigner:
    def __init__(self, order_res=(True, 12345, "resting"), cancel_res=True):
        self._o, self._c, self.calls = order_res, cancel_res, []
        self.kwargs_calls = []

    def sz_decimals(self, coin):
        return 2

    def spot_order(self, pair, is_buy, sz, px, sz_decimals=2, cloid=None, **kwargs):
        self.calls.append(("spot_order", pair, is_buy, sz, px, cloid))
        self.kwargs_calls.append(kwargs)
        return self._o

    def cancel(self, pair, oid):
        self.calls.append(("cancel", pair, oid))
        return self._c


def _provider(read, signer):
    p = HyperliquidProvider(token="HYPE")
    p._client = read
    p._client_tried = True
    p._spot_pair = "@107"
    p._signer = lambda: signer
    return p


class HLExecutorContractTest(unittest.TestCase):
    def setUp(self):
        os.environ["HL_ACCOUNT_ADDRESS"] = "0xABC"
        os.environ.pop("HL_LIVE_ORDERS", None)
        self.signer = FakeSigner()
        self.p = _provider(FakeRead(), self.signer)

    def tearDown(self):
        os.environ.pop("HL_LIVE_ORDERS", None)
        os.environ.pop("HL_SECRET_KEY", None)

    def test_satisfies_protocol(self):
        self.assertIsInstance(self.p, StrategyExecutor)

    def test_pair_precision(self):
        pp = self.p.pair_precision("HYPE")
        # spot: price_decimals = 8 - szDecimals(2) = 6
        self.assertEqual(pp, PairPrecision(price_decimals=6, volume_decimals=2,
                                           order_min=0.0, base_asset="HYPE"))

    def test_ohlc_behaviors(self):
        with self.subTest(msg="closes exclude forming bar"):
            self.assertEqual(self.p.ohlc_closes("HYPE", 240), [10.0, 11.0])

        with self.subTest(msg="series preserves last completed timestamp"):
            self.p._client.candles = lambda *_args: [
                {"c": "10", "T": 1_000},
                {"c": "11", "T": 2_000},
                {"c": "12", "T": 3_000},
            ]
            series = self.p.ohlc_series("HYPE", 240)
            self.assertEqual(series.closes, (10.0, 11.0))
            self.assertEqual(series.last_closed_at, 2.0)
            self.assertEqual(series.timestamps, (1.0, 2.0))

        with self.subTest(msg="rejects non hype symbol before reading cached pair"):
            calls = []
            self.p._client.candles = lambda *args: calls.append(args)
            for symbol in ("BTCUSDC", "HYPERUSDC", "HYPEFAKE", "HYPEUSD"):
                with self.subTest(symbol=symbol):
                    with self.assertRaisesRegex(ProviderError, "unsupported"):
                        self.p.ohlc_closes(symbol, 240)
            self.assertEqual(calls, [])

    def test_symbol_support_behaviors(self):
        with self.subTest(msg="exact hyperliquid spot aliases are supported"):
            for symbol in (
                    "HYPE", "hype", "HYPEUSDC", "HYPE/USDC", "HYPE-USDC"):
                with self.subTest(symbol=symbol):
                    self.assertTrue(self.p.supports_symbol(symbol))
            for symbol in (
                    "HYPEUSD", "HYPEFAKE", "H/Y/P/E", "HY-PE-US-DC",
                    "HYPE//USDC", "HYPE USDC"):
                with self.subTest(symbol=symbol):
                    self.assertFalse(self.p.supports_symbol(symbol))

            purr = HyperliquidProvider(token="PURR")
            self.assertTrue(purr.supports_symbol("PURR/USDC"))
            self.assertFalse(purr.supports_symbol("HYPE/USDC"))

        with self.subTest(msg="pair scoped entrypoints reject a mismatched symbol"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            calls = {
                "price": lambda: self.p.get_current_price("BTCUSDC"),
                "history": lambda: self.p.get_price_history("BTCUSDC", 1),
                "orders": lambda: self.p.get_orders("BTCUSDC", "BUY", 60),
                "trades": lambda: self.p.get_trades("BTCUSDC", 60),
                "open_orders": lambda: self.p.open_orders("BTCUSDC"),
                "place": lambda: self.p.place_order(
                    "BTCUSDC", "BUY", 60.0, 1.0),
                "precision": lambda: self.p.pair_precision("BTCUSDC"),
                "preflight_buy": lambda: self.p.preflight_order(
                    "BTCUSDC", "BUY", 1.0, 60.0),
                "preflight_sell": lambda: self.p.preflight_order(
                    "BTCUSDC", "SELL", 1.0, 60.0),
                "submit_limit": lambda: self.p.submit_order(
                    "BTCUSDC", "buy", 1.0, price=60.0),
                "submit_market": lambda: self.p.submit_order(
                    "BTCUSDC", "sell", 1.0, market=True),
                "lookup": lambda: self.p.order_by_client_id(
                    "BTCUSDC", "client-id"),
                "status": lambda: self.p.order_status("BTCUSDC", "1"),
                "cancel": lambda: self.p.cancel_order("BTCUSDC", "1"),
            }
            for name, call in calls.items():
                with self.subTest(entrypoint=name):
                    with self.assertRaisesRegex(ProviderError, "unsupported"):
                        call()
            self.assertEqual(self.signer.calls, [])

        with self.subTest(msg="instrument and explicit facade validate the symbol"):
            market_api = MarketApi([self.p])
            with self.assertRaisesRegex(ProviderError, "unsupported"):
                Instrument(
                    "BTC", "BTCUSDC", "Hyperliquid",
                    base="BTC", quote="USDC", api=market_api)
            with self.assertRaisesRegex(ProviderError, "unsupported"):
                market_api.get_current_price(
                    "BTCUSDC", provider_name="Hyperliquid")

    def test_submit_order_behaviors(self):
        with self.subTest(msg="is gated by hl_live_orders"):
            # by default HL_LIVE_ORDERS is missing -> a refusal (DN co-mingling safety)
            with self.assertRaises(ProviderError):
                self.p.submit_order("HYPE", "buy", 1.0, price=60.0)

        with self.subTest(msg="execution_enabled lazily loads hyperliquid env when untried"):
            fresh_p = HyperliquidProvider(token="HYPE")
            with patch.object(fresh_p, "_load_env", side_effect=lambda: os.environ.__setitem__("HL_LIVE_ORDERS", "true")) as load_mock:
                os.environ.pop("HL_LIVE_ORDERS", None)
                self.assertTrue(fresh_p.execution_enabled())
                load_mock.assert_called_once()
            os.environ.pop("HL_LIVE_ORDERS", None)

        with self.subTest(msg="live submit returns order id"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            oid = self.p.submit_order("HYPE", "buy", 1.0, price=60.0)
            self.assertEqual(oid, "12345")
            self.assertEqual(self.signer.calls[-1][:3], ("spot_order", "@107", True))

        with self.subTest(msg="propagates cloid"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            cloid = "0x0123456789abcdef0123456789abcdef"
            self.p.submit_order(
                "HYPE", "buy", 1.0, price=60.0, client_order_id=cloid,
            )
            self.assertEqual(self.signer.calls[-1][-1], cloid)

        with self.subTest(msg="market crosses the price"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            self.p.submit_order("HYPE", "sell", 1.0, price=None, market=True)
            px = self.signer.calls[-1][4]
            self.assertAlmostEqual(
                px, 60.0 * 0.95)  # A market sell crosses below mid for immediate fill.

        with self.subTest(msg="rejection raises"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            original_signer = self.p._signer
            self.p._signer = lambda: FakeSigner(order_res=(False, None, "Insufficient"))
            with self.assertRaises(ProviderError):
                self.p.submit_order("HYPE", "buy", 1.0, price=60.0)
            self.p._signer = original_signer

        with self.subTest(msg="an underfunded buy is refused before the signer"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            original_client = self.p._client
            self.p._client = FakeRead(balances=[
                {"coin": "USDC", "total": "10", "hold": "0"},
            ])
            self.signer.calls.clear()
            with self.assertRaisesRegex(
                    ProviderError, "insufficient USDC balance"):
                self.p.submit_order("HYPE", "buy", 1.0, price=60.0, kind="DCA")
            self.assertEqual(self.signer.calls, [])
            self.p._client = original_client

        with self.subTest(msg="a sell is not blocked by the quote balance"):
            os.environ["HL_LIVE_ORDERS"] = "true"
            original_client = self.p._client
            self.p._client = FakeRead(balances=[])
            self.assertEqual(
                self.p.submit_order("HYPE", "sell", 1.0, price=60.0), "12345",
            )
            self.p._client = original_client

    def test_place_order_market_and_rounding(self):
        os.environ["HL_LIVE_ORDERS"] = "true"
        os.environ["HL_SECRET_KEY"] = "fake-key"
        self.p._new_client = lambda _s=None: self.signer

        with self.subTest("place_order_market_sends_ioc_with_fresh_market_price_slippage"):
            self.signer.calls.clear()
            self.signer.kwargs_calls.clear()
            res = self.p.place_order("HYPE", "BUY", 100.0, 1.0, market=True)
            self.assertIsNotNone(res)
            self.assertEqual(res["orderId"], 12345)
            call = self.signer.calls[-1]
            kw = self.signer.kwargs_calls[-1]
            self.assertEqual(call[1], "@107")
            self.assertTrue(call[2])  # is_buy
            self.assertAlmostEqual(call[4], 63.0)  # fresh spot_mid 60.0 * 1.05
            self.assertTrue(kw.get("is_market"))
            self.assertEqual(kw.get("tif"), "Ioc")

        with self.subTest("place_order_force_treated_as_market_with_fresh_price"):
            self.signer.calls.clear()
            self.signer.kwargs_calls.clear()
            res = self.p.place_order("HYPE", "SELL", 100.0, 1.0, force=True)
            self.assertIsNotNone(res)
            call = self.signer.calls[-1]
            kw = self.signer.kwargs_calls[-1]
            self.assertFalse(call[2])  # is_buy is False
            self.assertAlmostEqual(call[4], 57.0)  # fresh spot_mid 60.0 * 0.95
            self.assertTrue(kw.get("is_market"))
            self.assertEqual(kw.get("tif"), "Ioc")

        with self.subTest("place_order_market_falls_back_to_input_price_when_current_price_unavailable"):
            self.signer.calls.clear()
            self.signer.kwargs_calls.clear()
            with patch.object(self.p, "get_current_price", return_value=None):
                res = self.p.place_order("HYPE", "BUY", 100.0, 1.0, market=True)
            self.assertIsNotNone(res)
            call = self.signer.calls[-1]
            self.assertAlmostEqual(call[4], 105.0)  # fallback 100.0 * 1.05

        with self.subTest("place_order_limit_sends_gtc"):
            self.signer.calls.clear()
            self.signer.kwargs_calls.clear()
            res = self.p.place_order("HYPE", "BUY", 100.0, 1.0)
            self.assertIsNotNone(res)
            call = self.signer.calls[-1]
            kw = self.signer.kwargs_calls[-1]
            self.assertAlmostEqual(call[4], 100.0)
            self.assertFalse(kw.get("is_market"))
            self.assertEqual(kw.get("tif"), "Gtc")

        with self.subTest("round_price_enforces_5_significant_figures"):
            self.assertAlmostEqual(self.p.round_price("HYPE", 1.234568), 1.2346)
            self.assertAlmostEqual(self.p.round_price("HYPE", 1234.567), 1234.6)
            self.assertAlmostEqual(self.p.round_price("HYPE", 0.01234567), 0.012346)
            self.assertAlmostEqual(self.p.round_price("HYPE", 0.001234567), 0.001235)

    def test_client_id_behaviors(self):
        with self.subTest(msg="order by client id uses authoritative cloid query"):
            cloid = "0x0123456789abcdef0123456789abcdef"
            self.p._client.info.query_order_by_cloid = lambda _addr, _cloid: {
                "status": "order",
                "order": {"status": "open", "order": {"oid": 42}},
            }
            self.assertEqual(
                self.p.order_by_client_id("HYPE", cloid),
                {"orderId": "42", "status": "open"},
            )

        with self.subTest(msg="order by client id converts hex string to sdk cloid"):
            cloid = "0x0123456789abcdef0123456789abcdef"
            observed = []
            self.p._client.info.query_order_by_cloid = lambda _addr, value: (
                observed.append(value) or {"status": "unknownOid"}
            )
            self.assertIsNone(self.p.order_by_client_id("HYPE", cloid))
            self.assertEqual(str(observed[0]), cloid)
            self.assertTrue(hasattr(observed[0], "to_raw"))

    def test_instrument_place_submit_and_lookup_share_deterministic_cloid(self):
        os.environ["HL_LIVE_ORDERS"] = "true"
        os.environ["HL_SECRET_KEY"] = "test-only-secret"
        raw_client_id = "OR_0123456789abcdef01234567_0"
        expected = hl_provider.HyperliquidProvider.derive_client_order_id(raw_client_id)
        observed_lookup = []
        self.p._new_client = lambda _secret=None: self.signer
        self.p._client.info.query_order_by_cloid = lambda _addr, value: (
            observed_lookup.append(str(value)) or {
                "status": "order",
                "order": {"status": "open", "order": {"oid": 12345}},
            }
        )
        instrument = Instrument(
            "HYPE", "HYPE", "Hyperliquid", base="HYPE", quote="USDC",
            api=MarketApi([self.p]))

        with (
            patch.object(order_guard, "daily_limit_guard",
                         return_value=(True, None)),
            patch.object(order_guard, "get_margin_cfg", return_value=0.01),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(self.p, "policy_cap_quantity",
                         side_effect=lambda _s, _o, _p, qty, _a, **_k: qty),
            patch.object(self.p, "fee_cap_quantity",
                         side_effect=lambda _s, _o, _p, available: available),
            patch.object(instrument_module.order_lifecycle, "log_order_outcome"),
        ):
            placed = instrument.place(
                "BUY", 60.0, 1.0, smart=False, wait_for_trend=False,
                caller_owns_retry=True, client_order_id=raw_client_id)

        self.assertEqual(placed["orderId"], 12345)
        self.assertEqual(self.signer.calls[-1][-1], expected)
        self.p.submit_order(
            "HYPE", "buy", 1.0, price=60.0,
            client_order_id=raw_client_id)
        self.assertEqual(self.signer.calls[-1][-1], expected)
        self.assertEqual(
            self.p.order_by_client_id("HYPE", raw_client_id)["orderId"],
            "12345")
        self.assertEqual(observed_lookup, [expected])

    def test_order_status_behaviors(self):
        with self.subTest(msg="open"):
            self.p._client = FakeRead(status="open")
            st = self.p.order_status("HYPE", "999")
            self.assertEqual(st.status, "open")

        with self.subTest(msg="includes partial fill"):
            self.p._client = FakeRead(
                fills=[{"oid": 999, "sz": "0.4", "px": "60", "fee": "0.02"}],
                status="open",
            )
            st = self.p.order_status("HYPE", "999")
            self.assertEqual(st.status, "open")
            self.assertAlmostEqual(st.filled_qty, 0.4)
            self.assertAlmostEqual(st.cost, 24.0)
            self.assertAlmostEqual(st.fee, 0.02)

        with self.subTest(msg="aggregates fills"):
            self.p._client = FakeRead(
                fills=[
                    {"oid": 5, "sz": "1.5", "px": "60", "fee": "0.1"},
                    {"oid": 5, "sz": "0.5", "px": "62", "fee": "0.05"},
                ],
                status="filled",
            )
            st = self.p.order_status("HYPE", "5")
            self.assertEqual(st.status, "closed")
            self.assertAlmostEqual(st.filled_qty, 2.0)
            self.assertAlmostEqual(st.cost, 1.5 * 60 + 0.5 * 62)
            self.assertAlmostEqual(st.fee, 0.15)

        with self.subTest(msg="converts the fee from hype to usdc"):
            self.p._client = FakeRead(
                fills=[{
                    "oid": 5, "sz": "2", "px": "75", "fee": "0.001",
                    "feeToken": "HYPE",
                }],
                status="filled",
            )
            st = self.p.order_status("HYPE", "5")
            self.assertAlmostEqual(st.fee, 0.075)

        with self.subTest(msg="canceled from the dedicated endpoint"):
            self.p._client = FakeRead(status="canceled")
            st = self.p.order_status("HYPE", "77")
            self.assertEqual(st.status, "canceled")

        with self.subTest(msg="unknown order status remains indeterminate"):
            self.p._client = FakeRead(status="unknownOid")
            with self.assertRaises(ProviderError):
                self.p.order_status("HYPE", "77")

        with self.subTest(msg="a terminal order waits for complete fills"):
            read = FakeRead(status="filled")
            read.info.query_order_by_oid = lambda addr, oid: {
                "status": "order",
                "order": {
                    "status": "filled",
                    "order": {"origSz": "1", "sz": "0"},
                },
            }
            self.p._client = read
            with self.assertRaisesRegex(ProviderError, "fills incomplete"):
                self.p.order_status("HYPE", "77")

    def test_cancel_behaviors(self):
        with self.subTest(msg="delegates to signer"):
            self.p.cancel_order("HYPE", "5")
            self.assertIn(("cancel", "@107", 5), self.signer.calls)

        with self.subTest(msg="unconfirmed cancel raises"):
            self.p._signer = lambda: FakeSigner(cancel_res=False)
            with self.assertRaises(ProviderError):
                self.p.cancel_order("HYPE", "5")

    def test_open_orders_standard_fields_for_recovery(self):
        self.p._client = FakeRead(opens=[{
            "coin": "@107",
            "side": "B",
            "limitPx": "55.5",
            "sz": "1.5",
            "origSz": "2.0",
            "oid": 8888,
            "cloid": "0x1234567890abcdef1234567890abcdef",
            "timestamp": 123456789,
        }])
        orders = self.p.open_orders("HYPE")
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertEqual(o["orderId"], "8888")
        self.assertEqual(o["clientOrderId"], "0x1234567890abcdef1234567890abcdef")
        self.assertEqual(o["side"], "BUY")
        self.assertAlmostEqual(o["price"], 55.5)
        self.assertAlmostEqual(o["qty"], 1.5)
        self.assertAlmostEqual(o["origQty"], 2.0)
        self.assertAlmostEqual(o["executedQty"], 0.5)
        self.assertEqual(o["status"], "OPEN")
        self.assertEqual(o["timestamp"], 123456789)

    def test_order_status_resilience_and_fee_conversion(self):
        with self.subTest(msg="safe with null oid and invalid user_fills type"):
            # A fill with None oid should be gracefully skipped instead of raising TypeError
            read = FakeRead(
                fills=[
                    {"oid": None, "sz": "0.1", "px": "50"},
                    {"oid": "not_an_int", "sz": "0.1", "px": "50"},
                    {"oid": 999, "sz": "0.5", "px": "60", "fee": "0.01"},
                ],
                status="open",
            )
            self.p._client = read
            st = self.p.order_status("HYPE", "999")
            self.assertAlmostEqual(st.filled_qty, 0.5)

            # user_fills returning dict (e.g. rate limit error) raises ProviderError fail-closed
            read.info.user_fills = lambda _addr: {"error": "rate limited"}
            with self.assertRaisesRegex(ProviderError, "unexpected type"):
                self.p.order_status("HYPE", "999")

        with self.subTest(msg="converts hype fee for non-hype token"):
            purr_p = HyperliquidProvider(token="PURR")
            purr_client = FakeRead(
                fills=[{
                    "oid": 10, "sz": "100", "px": "0.5", "fee": "0.1",
                    "feeToken": "HYPE",
                }],
                status="filled",
            )
            purr_client.resolve_spot_pair = lambda token: "@107" if token == "HYPE" else "@150"
            purr_client.spot_mid = lambda pair: 50.0 if pair == "@107" else 0.5
            purr_p._client = purr_client
            purr_p._client_tried = True
            purr_p._spot_pair = "@150"
            st = purr_p.order_status("PURR", "10")
            # 0.1 HYPE fee * $50 mid = $5.0 USDC fee
            self.assertAlmostEqual(st.fee, 5.0)

        with self.subTest(msg="retries transient 500 ServerError on order_status"):
            flaky_read = FakeRead(status="open")
            calls = {"q": 0}
            orig_q = flaky_read.info.query_order_by_oid

            def _flaky_query(addr, oid):
                calls["q"] += 1
                if calls["q"] == 1:
                    raise RuntimeError(500, "null")
                return orig_q(addr, oid)

            flaky_read.info.query_order_by_oid = _flaky_query
            self.p._client = flaky_read
            with patch("providers.hyperliquid_provider.time.sleep", return_value=None):
                st = self.p.order_status("HYPE", "999")
            self.assertEqual(st.status, "open")
            self.assertEqual(calls["q"], 2)

    def test_hl_client_cancel_inspects_statuses_error(self):
        mod = load_local_module(os.path.join(ROOT, "hyperliquid"), "hl_client", module_key="hl_client_under_test", reload=True)
        HLClient = mod.HLClient
        client = HLClient(secret_key="0x" + "1" * 64, account_address="0xABC")
        client.exchange = FakeSigner()
        # Mock exchange cancel returning status ok but error status inside data
        client.exchange.cancel = lambda coin, oid: {
            "status": "ok",
            "response": {"type": "cancel", "data": {"statuses": [{"error": "Order already filled"}]}},
        }
        self.assertFalse(client.cancel("HYPE", 123))

        # Successful cancel
        client.exchange.cancel = lambda coin, oid: {
            "status": "ok",
            "response": {"type": "cancel", "data": {"statuses": ["success"]}},
        }
        self.assertTrue(client.cancel("HYPE", 123))

    def test_strategy_apply_close_cancels_resting_orders_on_cycle_end(self):
        mod = load_local_module(os.path.join(ROOT, "hyperliquid"), "strategy", module_key="hl_strategy_under_test", register=False)
        Strategy, StratParams = mod.Strategy, mod.StratParams
        client = FakeSigner()
        params = StratParams(direction="long", entry_amount=50.0)
        strat = Strategy(client, "HYPE", params, dry_run=False, replay_mode=False)
        strat.s["qty"] = 1.0
        strat.s["cost"] = 60.0
        strat.s["orders"] = [
            {"oid": 101, "role": "open", "sz": 1.0, "px": 55.0},
            {"oid": 102, "role": "close", "sz": 1.0, "px": 65.0},
        ]
        strat._apply_close(avg=60.0, price=65.0, sz=1.0)
        self.assertEqual(strat.s["qty"], 0.0)
        # Verify cancel was called for both resting orders
        canceled_oids = [call[2] for call in client.calls if call[0] == "cancel"]
        self.assertIn(101, canceled_oids)
        self.assertIn(102, canceled_oids)


# ─── Hyperliquid Launcher, Client Timeout, Shadow Longterm & Candles ────────

class HLLauncherTest(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_local_module(os.path.join(ROOT, "hyperliquid"), "hl_bot", module_key="repo_hl_bot", reload=True)

    def setUp(self):
        super().setUp()
        self.original_here = self.module._HERE
        self.module._HERE = self.tempdir.name

    def tearDown(self):
        self.module._HERE = self.original_here
        super().tearDown()

    def test_live_state_is_hyperliquid_scoped(self):
        self.assertEqual(self.module.state_dir_for(False), self.tempdir.name)

    def test_paper_state_is_created_and_separate_from_live(self):
        paper_dir = self.module.state_dir_for(True)
        self.assertEqual(paper_dir, os.path.join(self.tempdir.name, ".paper_state"))
        self.assertTrue(os.path.isdir(paper_dir))
        self.assertNotEqual(paper_dir, self.module.state_dir_for(False))


class _TimeoutSession:
    def __init__(self):
        self.timeouts = []

    def request(self, *_args, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return kwargs.get("timeout")


class _TimeoutApi:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.session = _TimeoutSession()


class HyperliquidTimeoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_local_module(
            os.path.join(ROOT, "hyperliquid"), "hl_client", module_key="hl_client_timeout_under_test", reload=True
        )

    def test_timeouts_are_passed_before_sdk_constructors_run(self):
        info = _TimeoutApi(timeout=self.module.INFO_TIMEOUT_SECONDS)
        exchange = _TimeoutApi(timeout=self.module.EXCHANGE_TIMEOUT_SECONDS)
        wallet = object()

        with (
            patch.object(self.module, "Info", return_value=info) as info_constructor,
            patch.object(self.module, "Exchange", return_value=exchange) as exchange_constructor,
            patch.object(self.module.eth_account.Account, "from_key", return_value=wallet),
        ):
            client = self.module.HLClient(
                secret_key="secret", account_address="0xaccount", mainnet=True,
            )

        info_constructor.assert_called_once_with(
            self.module.constants.MAINNET_API_URL,
            skip_ws=True,
            timeout=self.module.INFO_TIMEOUT_SECONDS,
        )
        exchange_constructor.assert_called_once_with(
            wallet,
            self.module.constants.MAINNET_API_URL,
            account_address="0xaccount",
            timeout=self.module.EXCHANGE_TIMEOUT_SECONDS,
        )
        self.assertIs(client.info, info)
        self.assertIs(client.exchange, exchange)

    def test_replaces_sdk_timeout_none_on_api_and_request(self):
        api = _TimeoutApi(timeout=None)
        self.module._force_timeout(api, seconds=10)
        self.assertEqual(api.timeout, 10)
        self.assertEqual(api.session.request("POST", "/info", timeout=None), 10)

    def test_preserves_explicit_shorter_timeout(self):
        api = _TimeoutApi(timeout=3)
        self.module._force_timeout(api, seconds=10)
        self.assertEqual(api.timeout, 3)
        self.assertEqual(api.session.request("POST", "/info", timeout=2), 2)

    def test_connect_only_retries_are_mounted_for_dns_blips(self):
        import requests
        api = _TimeoutApi(timeout=None)
        api.session = requests.Session()
        self.module._force_timeout(api, seconds=10)
        retries = api.session.get_adapter("https://api.hyperliquid.xyz").max_retries
        self.assertEqual((retries.connect, retries.read, retries.status), (3, 0, 0))

    def test_info_requests_retry_transient_500_without_retrying_exchange(self):

        class _FlakySession:
            def __init__(self, statuses):
                self.statuses = list(statuses)
                self.calls = 0

            def request(self, *_args, **_kwargs):
                self.calls += 1
                code = self.statuses.pop(0) if self.statuses else 200
                return SimpleNamespace(status_code=code)

        info_api = SimpleNamespace(timeout=10, session=_FlakySession([500, 200]))
        exch_api = SimpleNamespace(timeout=30, session=_FlakySession([500, 200]))
        with patch.object(self.module.time, "sleep", return_value=None):
            self.module._force_timeout(info_api, seconds=10)
            self.module._force_timeout(exch_api, seconds=30)
            info_resp = info_api.session.request("POST", "https://api.hyperliquid.xyz/info")
            exch_resp = exch_api.session.request("POST", "https://api.hyperliquid.xyz/exchange")

        self.assertEqual(info_resp.status_code, 200)
        self.assertEqual(info_api.session.calls, 2)
        self.assertEqual(exch_resp.status_code, 500)
        self.assertEqual(exch_api.session.calls, 1)


class HyperliquidLongtermShadowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in (os.path.join(ROOT, "hyperliquid"), os.path.join(ROOT, "kraken")):
            if p not in sys.path:
                sys.path.insert(0, p)
        cls.module = load_local_module(
            os.path.join(ROOT, "hyperliquid"), "shadow_longterm", module_key="hl_shadow_longterm_under_test", reload=True
        )

    def test_variants_are_fixed_and_paper_only_parameters(self):
        variants = self.module._variants(240)
        self.assertEqual(list(variants), [
            "current", "tp_regime_gate", "long_tp3_trail3", "reentry4",
            "trail_profit_floor_sl18", "overlay650t8_regime_v2",
            "B_dcabrake_regime_v2",
        ])
        candidate = variants["long_tp3_trail3"]
        self.assertTrue(variants["tp_regime_gate"].tp_regime_gate)
        self.assertEqual(candidate.takeprofit_pct, 3.0)
        self.assertTrue(candidate.tp_trend_hold)
        self.assertFalse(candidate.tp_trail_adaptive)
        self.assertEqual(candidate.tp_trail_pct, 3.0)
        self.assertEqual(variants["reentry4"].reentry_drop_pct, 4.0)
        floor = variants["trail_profit_floor_sl18"]
        self.assertEqual(floor.tp_trail_profit_floor_pct, 1.0)
        self.assertEqual(floor.stop_loss_pct, 18.0)
        overlay = variants["overlay650t8_regime_v2"]
        self.assertTrue(overlay.trend_overlay)
        self.assertEqual(overlay.trend_topup, 650.0)
        self.assertEqual(overlay.trend_trail_pct, 8.0)
        self.assertFalse(overlay.trend_exit_break)
        self.assertTrue(variants["B_dcabrake_regime_v2"].dca_trend_brake)

    def test_rejects_non_native_interval(self):
        with self.assertRaisesRegex(ValueError, "240m"):
            self.module._variants(60)

    def test_public_fetch_constructs_client_without_secret_and_drops_open_bar(self):
        fake = MagicMock()
        fake.candles.return_value = [
            {"t": 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.5"},
            {"t": 2000, "o": "1.5", "h": "3", "l": "1", "c": "2.5"},
        ]
        with patch("hl_client.HLClient.public", return_value=fake) as constructor:
            rows = self.module._fetch_with_ts("ignored", 240)
        constructor.assert_called_once_with()
        self.assertEqual(rows, [(1, 1.0, 2.0, 0.5, 1.5)])

    def test_snapshot_injects_isolated_log_dir_and_public_fetch(self):
        observed = {}

        def _fake_snapshot(*_args, **_kwargs):
            observed["log_dir"] = self.module.shadow.LOG_DIR
            observed["fetch"] = self.module.shadow._fetch_with_ts
            return {"ok": True}

        with patch.object(self.module.shadow, "snapshot", side_effect=_fake_snapshot) as call:
            result = self.module.snapshot(quiet=True)
        self.assertEqual(result, {"ok": True})
        call.assert_called_once_with("HYPE-HL", 240, 0.07, quiet=True)
        self.assertEqual(observed["log_dir"], self.module.LOG_DIR)
        self.assertIs(observed["fetch"], self.module._fetch_with_ts)


class HyperliquidDatasetTest(unittest.TestCase):
    def test_spot_pair_normalization_and_continuity(self):
        from offline.runners.fetch_hyperliquid_candles import (
            _hl_md,
            resolve_spot_pair,
            validate_continuity,
        )

        meta = {
            "tokens": [{"name": "USDC", "index": 0}, {"name": "HYPE", "index": 150}],
            "universe": [{"name": "@107", "tokens": [150, 0]}],
        }
        self.assertEqual(resolve_spot_pair("hype", "usdc", meta), "@107")

        rows = [
            {"t": 2_000, "T": 2_999, "o": "2", "h": "4", "l": "1", "c": "3"},
            {"t": 1_000, "T": 1_999, "o": "1", "h": "3", "l": "0.5", "c": "2"},
            {"t": 2_000, "T": 2_999, "o": "2", "h": "5", "l": "1", "c": "4"},
            {"t": 3_000, "T": 3_999, "o": "4", "h": "6", "l": "3", "c": "5"},
        ]
        records = _hl_md.normalize_closed_candles(rows, now_ms=3_500)
        self.assertEqual([row["timestamp"] for row in records], [1, 2])
        self.assertEqual((records[-1]["high"], records[-1]["close"]), (5.0, 4.0))

        with self.assertRaisesRegex(ValueError, "discontinuous dataset"):
            validate_continuity([{"timestamp": 0}, {"timestamp": 3_600}, {"timestamp": 10_800}], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_hyperliquidprovider_preflight_order_order_status_cancel_order_reject_nan_bool_iter40_s4(monkeypatch):
    """[ITER40 #4] HyperliquidProvider preflight_order, order_status, cancel_order reject NaN/bool."""
    # 4. HyperliquidProvider preflight_order, order_status, cancel_order reject NaN/bool
    hp = HyperliquidProvider()
    monkeypatch.setattr(hp, "free_balance", lambda asset: 1000.0)
    with pytest.raises(ProviderError):
        hp.preflight_order("HYPEUSDC", "BUY", float("nan"), price=20.0)
    with pytest.raises(ProviderError):
        hp.preflight_order("HYPEUSDC", "BUY", 1.0, price=float("nan"))
    with pytest.raises(ProviderError):
        hp.preflight_order("HYPEUSDC", "BUY", True, price=20.0)
    with pytest.raises(ProviderError):
        hp.order_status("HYPEUSDC", True)
    with pytest.raises(ProviderError):
        hp.cancel_order("HYPEUSDC", True)

def test_hl_client__round_px_mid_spot_mid_position_strict_spot_balance_strict_iter49_s1():
    """[ITER49 #1] hl_client: _round_px, mid, spot_mid, position_strict, spot_balance_strict."""
    hlc = load_venue_modules("hyperliquid", "hl_client")
    # 1. hl_client: _round_px, mid, spot_mid, position_strict, spot_balance_strict
    with pytest.raises(ValueError):
        hlc._round_px(True, 2)
    with pytest.raises(ValueError):
        hlc._round_px(20.0, True)

    dummy = object.__new__(hlc.HLClient)
    dummy.address = "0x1234"
    dummy.info = SimpleNamespace(
        all_mids=lambda: {"HYPE": float("nan"), "BTC": True, "ETH": "2500.5"},
        user_state=lambda _addr: {
            "assetPositions": [{"position": {"coin": "HYPE", "szi": float("nan"), "entryPx": 20.0}}],
            "withdrawable": True,
        },
        spot_user_state=lambda _addr: {
            "balances": [{"coin": "HYPE", "total": True}]
        },
    )
    assert dummy.mid("HYPE") is None
    assert dummy.mid("BTC") is None
    assert dummy.mid("ETH") == 2500.5
    with pytest.raises(hlc.HLError):
        dummy.position_strict("HYPE")

def test_hyperliquid_market_data_normalize_closed_candles_rejects_bool_and_invalid_ohlc_iter49_s2():
    """[ITER49 #2] hyperliquid/market_data: normalize_closed_candles rejects bool and invalid OHLC bounds."""
    hl_md = load_venue_modules("hyperliquid", "market_data")
    # 2. hyperliquid/market_data: normalize_closed_candles rejects bool and invalid OHLC bounds
    rows = [
        {"t": True, "T": 2000, "o": 10, "h": 12, "l": 9, "c": 11},
        {"t": 1000, "T": 2000, "o": 10, "h": 8, "l": 9, "c": 11},   # h < l
        {"t": 1000, "T": 2000, "o": 10, "h": 12, "l": 9, "c": 11},  # valid
    ]
    norm = hl_md.normalize_closed_candles(rows, now_ms=5000, drop_incomplete=True)
    assert len(norm) == 1
    assert norm[0]["close"] == 11.0

def test_hyperliquid_price_analysis_detect_long_term_trend_ignores_nan_bool_points_iter49_s3():
    """[ITER49 #3] hyperliquid/price_analysis: detect_long_term_trend ignores NaN/bool points."""
    hl_pa = load_venue_modules("hyperliquid", "signals")
    # 3. hyperliquid/price_analysis: detect_long_term_trend ignores NaN/bool points
    assert hl_pa.detect_long_term_trend([1000.0, 2000.0, 3000.0], [float("nan"), float("nan"), float("nan")]) is None
    assert hl_pa.detect_long_term_trend([True, False], [10.0, 11.0]) is None

def test_hyperliquid_strategy_replay_step_ignores_nan_bool_price_iter49_s4():
    """[ITER49 #4] hyperliquid/strategy & replay: step ignores NaN/bool price and funding_at sanitizes NaN/bool funding rate."""
    hl_strat, hl_rep = load_venue_modules("hyperliquid", "strategy", "replay")
    # 4. hyperliquid/strategy & replay: step ignores NaN/bool price and funding_at sanitizes NaN/bool funding rate
    st_inst = hl_strat.Strategy(
        client=None, coin="HYPE", params=hl_strat.StratParams(direction="long"), dry_run=True, replay_mode=True
    )
    st_inst.step(float("nan"))
    st_inst.step(True)
    assert st_inst.s["orders"] == []

    assert hl_rep.funding_at([(1000, float("nan"))], 1000) == 0.0
    assert hl_rep.funding_at([(1000, True)], 1000) == 0.0
    assert hl_rep.funding_at([(1000, 0.0001)], True) == 0.0

def test_hyperliquidprovider_free_balance_get_price_history_ohlc_series_order_status_iter51_s4(monkeypatch):
    """[ITER51 #4] HyperliquidProvider: free_balance, get_price_history, ohlc_series, order_status."""
    # 4. HyperliquidProvider: free_balance, get_price_history, ohlc_series, order_status
    hlp = HyperliquidProvider()
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
    fake_hl = SimpleNamespace(
        info=SimpleNamespace(
            spot_user_state=lambda addr: {
                "balances": [{"coin": "USDC", "total": "nan", "hold": "0.0"}]
            },
            candles_snapshot=lambda pair, interval, start, end: [
                {"t": 1700000000000, "c": "nan"},
                {"t": True, "c": "25.0"},
                {"t": 1700000060000, "c": "25.5"},
            ],
            query_order_by_oid=lambda addr, oid: {
                "status": "order",
                "order": {"status": "open", "order": {"origSz": "nan", "sz": "1.0"}},
            },
            user_fills=lambda addr: [],
        ),
        candles=lambda pair, interval, lookback_h: [
            {"c": "nan", "T": 1700000000000},
            {"c": True, "T": 1700000060000},
            {"c": "25.5", "T": 1700000120000},
            {"c": "26.0", "T": 1700000180000},
        ],
    )
    monkeypatch.setattr(hlp, "_hl", lambda: fake_hl)
    monkeypatch.setattr(hlp, "_pair", lambda: "@107")
    assert hlp.free_balance("USDC") is None
    assert hlp.get_price_history("HYPEUSDC", True) is None
    assert hlp.get_price_history("HYPEUSDC", float("nan")) is None
    hist = hlp.get_price_history("HYPEUSDC", 1.0)
    assert hist == [{"timestamp": 1700000060000, "price": 25.5}]
    with pytest.raises(ProviderError):
        hlp.ohlc_series("HYPEUSDC", True)
    series = hlp.ohlc_series("HYPEUSDC", 15)
    assert series.closes == (25.5,)
    with pytest.raises(ProviderError):
        hlp.order_status("HYPEUSDC", "101")

def test_hyperliquidprovider_submit_order_market_rejects_nan_mid_price_iter58_s5(monkeypatch):
    """[ITER58 #5] HyperliquidProvider.submit_order market rejects NaN mid price."""
    # 5. HyperliquidProvider.submit_order market rejects NaN mid price
    hp = HyperliquidProvider(token="HYPE", live_enabled=True)
    monkeypatch.setattr(hp, "_pair", lambda: "@107")
    monkeypatch.setattr(hp, "get_current_price", lambda s: float("nan"))
    with pytest.raises(ProviderError, match="price unavailable"):
        hp.submit_order("HYPEUSDC", "BUY", 1.0, price=None, market=True)

def test_hlclient_metadata_and_order_placement_reject_boolean_nan_iter59_s5():
    """[ITER59 #5] HLClient metadata and order placement reject boolean/NaN/non-positive inputs."""
    hl_mod = load_venue_modules("hyperliquid", "hl_client")
    HLClient, HLError = hl_mod.HLClient, hl_mod.HLError
    # 5. HLClient metadata and order placement reject boolean/NaN/non-positive inputs
    hl = HLClient.__new__(HLClient)
    hl._meta_cache = {"HYPE": {"name": "HYPE", "szDecimals": True, "maxLeverage": False}}
    hl.info = MagicMock()
    hl.info.spot_meta.return_value = {"tokens": []}
    with pytest.raises(HLError):
        hl.sz_decimals("HYPE")

    hl._meta_cache = {"HYPE": {"name": "HYPE", "szDecimals": 2, "maxLeverage": 5}}
    hl.exchange = MagicMock()
    ok, oid, msg = hl.place_limit("HYPE", True, float("nan"), 20.0)
    assert ok is False and oid is None
    ok2, oid2, msg2 = hl.spot_order("@107", True, 0.00001, 20.0, sz_decimals=2)
    assert ok2 is False and oid2 is None
    hl.exchange.order.assert_not_called()

def test_hl_strategy_normalizes_uppercase_long_to_sign_and_iter60_s1():
    """[ITER60 #1] HL Strategy normalizes uppercase 'LONG' to sign=+1 and rejects invalid direction."""
    hl_strat_mod = load_venue_modules("hyperliquid", "strategy")
    # 1. HL Strategy normalizes uppercase "LONG" to sign=+1 and rejects invalid direction
    p_long = hl_strat_mod.StratParams(direction="LONG")
    st_long = hl_strat_mod.Strategy(
        None, "HYPE", p_long, dry_run=True, replay_mode=True,
        initial_state={"qty": None, "cost": "nan", "orders": [None, {"role": "open", "px": True, "sz": 1.0}]},
    )
    assert st_long.sign == 1
    assert st_long.open_side == "buy"
    assert st_long.s["qty"] == 0.0
    assert st_long.s["cost"] == 0.0
    assert st_long.s["orders"] == []

    with pytest.raises(ValueError, match="Invalid Strategy direction"):
        hl_strat_mod.Strategy(None, "HYPE", hl_strat_mod.StratParams(direction="sideways"), dry_run=True, replay_mode=True)

def test_hl_price_analysis_detect_long_term_trend_rejects_step_hours_preventing_infinite_loop_iter60_s2():
    """[ITER60 #2] HL price_analysis.detect_long_term_trend rejects step_hours <= 0 (preventing infinite loop) and sorts unsorted timestamps."""
    hl_pa_mod = load_venue_modules("hyperliquid", "signals")
    # 2. HL price_analysis.detect_long_term_trend rejects step_hours <= 0 (preventing infinite loop) and sorts unsorted timestamps
    assert hl_pa_mod.detect_long_term_trend([1000, 2000, 3000], [10.0, 11.0, 12.0], window_hours=1, step_hours=0) is None
    ts_unsorted = [float(i * 3600) for i in range(1, 60)][::-1]
    px_unsorted = [10.0 + 0.5 * i for i in range(1, 60)][::-1]
    res = hl_pa_mod.detect_long_term_trend(ts_unsorted, px_unsorted, window_hours=12, step_hours=4, min_consecutive_blocks=3)
    assert res is not None and res["direction"] == "up"

def test_t212client_get_historical_order_does_not_dilute_fillprice_when_a_iter60_s4(monkeypatch):
    """[ITER60 #4] T212Client.get_historical_order does not dilute fillPrice when a sub-fill has price=0."""
    # 4. T212Client.get_historical_order does not dilute fillPrice when a sub-fill has price=0
    t212_mod = load_venue_modules("212trading", "t212_client")

    cli = t212_mod.T212Client("k", "s", env="demo", min_gap_sec=0.0, portfolio_ttl_sec=1.0, order_min_gap_sec=0.0)
    payload = {
        "items": [
            {"order": {"id": "77", "status": "FILLED"}, "fill": {"quantity": 2.0, "price": 50.0}},
            {"order": {"id": "77", "status": "FILLED"}, "fill": {"quantity": 2.0, "price": 0.0}},
        ]
    }
    monkeypatch.setattr(t212_mod, "http_get", lambda url, headers=None: (200, json.dumps(payload).encode("utf-8")))
    hist = cli.get_historical_order("77")
    assert hist["filledQuantity"] == pytest.approx(4.0)
    assert hist["fillPrice"] == pytest.approx(50.0)

def test__round_px_hlclient_cancel_candles_reject_inf_bool_invalid_iter68_s4():
    """[ITER68 #4] _round_px & HLClient.cancel / candles reject inf/bool/invalid parameters."""
    from hl_client import _round_px, HLClient
    # 4. _round_px & HLClient.cancel / candles reject inf/bool/invalid parameters
    with pytest.raises(ValueError):
        _round_px(25.0, float("inf"))
    dummy_hl = SimpleNamespace(exchange=object(), info=SimpleNamespace(candles_snapshot=lambda *a: [{"ok": 1}]))
    assert HLClient.cancel(dummy_hl, "HYPE", None) is False
    assert HLClient.cancel(dummy_hl, "HYPE", True) is False
    assert HLClient.candles(dummy_hl, "HYPE", lookback_hours=True) == []
    assert HLClient.candles(dummy_hl, "HYPE", lookback_hours=-5) == []

def test_hyperliquid_market_data_normalize_closed_candles_float_string_timestamps_fetch_ohlc_candles_interval_iter69_s1():
    """[ITER69 #1] hyperliquid.market_data: normalize_closed_candles float-string timestamps & fetch_ohlc_candles interval/lookback checks."""
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    hl_md = load_local_module(hl_dir, "market_data.py", module_key="_hl_market_data_69", reload=True)
    # 1. hyperliquid.market_data: normalize_closed_candles float-string timestamps & fetch_ohlc_candles interval/lookback checks
    rows = [{"t": "1700000000000.0", "T": "1700003599000.0", "o": "10", "h": "12", "l": "9", "c": "11"}]
    norm = hl_md.normalize_closed_candles(rows, now_ms=1700005000000)
    assert len(norm) == 1 and norm[0]["timestamp"] == 1700000000
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", interval=15.5)
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", interval=60, lookback_hours=True)
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", interval=60, lookback_hours=-1)

def test_hyperliquid_signals__file_signal_catches_overflowerror_on_huge_ts_iter69_s2(tmp_path):
    """[ITER69 #2] hyperliquid.signals._file_signal: catches OverflowError on huge ts/confidence."""
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    hl_sig = load_local_module(hl_dir, "signals.py", module_key="_hl_signals_69", reload=True)
    # 2. hyperliquid.signals._file_signal: catches OverflowError on huge ts/confidence
    sig_file = tmp_path / "signal.json"
    sig_file.write_text(json.dumps({"trend": "up", "confidence": 10**400, "ts": 10**400}), encoding="utf-8")
    os.environ.setdefault("SIGNAL_MAX_AGE_MIN", "60")
    res_sig = hl_sig._file_signal(str(sig_file))
    assert res_sig["trend"] == "neutral"

def test_hyperliquid_strategy_stratparams_from_dict_strategy__place_validation_iter69_s3():
    """[ITER69 #3] hyperliquid.strategy.StratParams.from_dict & Strategy._place validation."""
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    hl_strat = load_local_module(hl_dir, "strategy.py", module_key="_hl_strategy_69", reload=True)
    # 3. hyperliquid.strategy.StratParams.from_dict & Strategy._place validation
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"entry": True})
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"entry": float("nan")})
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"maxdca": 2.5})
    sp = hl_strat.StratParams.from_dict({"maxdca": "10.0"})
    assert sp.max_dca_buys == 10
    s_inst = hl_strat.Strategy(client=None, coin="HYPE", params=sp, dry_run=True, replay_mode=True)
    s_inst._place("open", 1.0, -5.0, "ENTRY", 50.0)
    assert s_inst.s["orders"] == []

def test_fetch_hyperliquid_candles_resolve_spot_pair_validate_continuity_iter72_s5():
    """[ITER72 #5] fetch_hyperliquid_candles.resolve_spot_pair & validate_continuity."""
    # 5. fetch_hyperliquid_candles.resolve_spot_pair & validate_continuity
    with pytest.raises(ValueError):
        fhc.resolve_spot_pair("HYPE", "USDC", "not-a-dict")
    meta = {
        "tokens": ["bad-item", {"name": "HYPE", "index": 1}, {"name": "USDC", "index": 0}],
        "universe": ["bad-item", {"tokens": [1, 0], "name": "@107"}],
    }
    assert fhc.resolve_spot_pair("hype", "usdc", meta) == "@107"
    with pytest.raises(ValueError):
        fhc.validate_continuity([], 0)

def test_iteration76_kraken_adaptive_thresholds_and_sys_path_isolation():
    import importlib
    from offline.runners import hyperliquid_walk_forward_baseline as hl_wf
    from offline.runners import t212_walk_forward_baseline as t212_wf

    while str(hl_wf.HL_DIR) in sys.path:
        sys.path.remove(str(hl_wf.HL_DIR))
    while str(t212_wf.T212_DIR) in sys.path:
        sys.path.remove(str(t212_wf.T212_DIR))
    importlib.reload(hl_wf)
    importlib.reload(t212_wf)
    importlib.reload(fhc)
    assert str(hl_wf.HL_DIR) not in sys.path
    assert str(t212_wf.T212_DIR) not in sys.path
    assert str(fhc.HL_DIR) not in sys.path

def test_hlclient_spot_order_place_limit_reject_non_bool_is_buy_and_iter92_s1(monkeypatch):
    """[ITER92 #1] HLClient spot_order / place_limit reject non-bool is_buy and fractional szDecimals."""
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    hl_client_mod = load_local_module(hl_dir, "hl_client", module_key="_it92_hl_client", reload=True)
    # 1. HLClient spot_order / place_limit reject non-bool is_buy and fractional szDecimals
    hl = object.__new__(hl_client_mod.HLClient)
    hl.exchange = object()
    hl._meta_cache = {"HYPE": {"szDecimals": 2.5, "maxLeverage": 3.5}}
    hl.info = type("I", (), {"spot_meta": lambda self: {}})()
    with pytest.raises(hl_client_mod.HLError, match="Missing size precision"):
        hl.sz_decimals("HYPE")
    hl._meta_cache = {"HYPE": {"szDecimals": 2, "maxLeverage": 5}}
    assert hl.spot_order("@107", "SELL", 1.0, 20.0) == (False, None, "invalid non-boolean is_buy")
    assert hl.spot_order("  ", True, 1.0, 20.0) == (False, None, "invalid pair")
    assert hl.place_limit("HYPE", "SELL", 1.0, 20.0) == (False, None, "invalid non-boolean is_buy or reduce_only")
    assert hl.place_limit("", True, 1.0, 20.0) == (False, None, "invalid coin")

def test_hyperliquid_market_data_validation_iter92_s2(monkeypatch):
    """[ITER92 #2] hyperliquid.market_data validation."""
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    hl_md_mod = load_local_module(hl_dir, "market_data", module_key="_it92_hl_md", reload=True)
    # 2. hyperliquid.market_data validation
    assert hl_md_mod.normalize_closed_candles(None, now_ms=1000) == []
    with pytest.raises(ValueError, match="invalid coin"):
        hl_md_mod.fetch_ohlc_candles("  ", 60)
    with pytest.raises(ValueError, match="invalid coin"):
        hl_md_mod.fetch_ohlc_candles(True, 60)
    assert hl_md_mod.fetch_funding_history("", 1000) == []
    assert hl_md_mod.fetch_funding_history("HYPE", -1) == []

def test_hyperliquid_strategy_stratparams_from_dict_bool_validation__check_stop_loss_avoids_iter92_s3(monkeypatch):
    """[ITER92 #3] hyperliquid.strategy: StratParams.from_dict bool validation & _check_stop_loss avoids duplicate close on close cancel failure."""
    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    hl_strat_mod = load_local_module(hl_dir, "strategy", module_key="_it92_hl_strat", reload=True)
    # 3. hyperliquid.strategy: StratParams.from_dict bool validation & _check_stop_loss avoids duplicate close on close cancel failure
    with pytest.raises(ValueError, match="invalid numeric boolean"):
        hl_strat_mod.StratParams.from_dict({"enable_takeprofit": 2})

    class _CloseFailCancelClient:
        exchange = None
        placed = 0

        def sz_decimals(self, _coin):
            return 2

        def cancel(self, _coin, _oid):
            return False

        def place_limit(self, _coin, _is_buy, _sz, _px, reduce_only=False):
            self.placed += 1
            return True, f"SL-{self.placed}", "ok"

    cli = _CloseFailCancelClient()
    strat = hl_strat_mod.Strategy(
        client=cli,
        coin="HYPE",
        params=hl_strat_mod.StratParams(direction="long", stop_loss_pct=5.0),
        dry_run=False,
        replay_mode=False,
        initial_state={"qty": 1.0, "cost": 100.0, "spent": 100.0, "orders": [{"oid": "TP-1", "role": "close", "px": 105.0, "sz": 1.0}]},
    )
    strat._save = lambda: None
    assert strat._check_stop_loss(94.0) is True
    assert cli.placed == 0
    assert [o["oid"] for o in strat.s["orders"]] == ["TP-1"]

def test_hyperliquid_replay_validation_212trading_strategy_ladder_validation_owned_iter93_s3(monkeypatch):
    """[ITER93 #3] hyperliquid.replay validation & 212trading.strategy ladder validation & owned parsing."""
    # 3. hyperliquid.replay validation
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hl_replay = load_local_module(os.path.join(root_dir, "hyperliquid"), "replay", module_key="hl_replay_iter93", reload=True)
    with pytest.raises(ValueError, match="Unsupported Hyperliquid replay engine"):
        hl_replay.make_hyperliquid_replay_adapter({}, engine="futures")
    with pytest.raises(ValueError, match="fee_pct"):
        hl_replay.run_replay([(100.0, 101.0, 99.0, 100.0, 1000)], fee_pct=-0.1)


    # 4. 212trading.strategy ladder validation & owned parsing
    t212_strat = load_local_module(os.path.join(root_dir, "212trading"), "strategy", module_key="t212_strategy_iter93", reload=True)

    with pytest.raises(ValueError, match="generate_tp_ladder requires max_pct >= min_pct"):
        t212_strat.generate_tp_ladder(steps=3, min_pct=5.0, max_pct=2.0)
    with pytest.raises(ValueError, match="unsupported generate_tp_ladder curve"):
        t212_strat.generate_tp_ladder(steps=3, min_pct=1.0, max_pct=5.0, curve="cubic")

    with pytest.raises(ValueError, match="fractions exceed 100%"):
        t212_strat.StratParams.from_env({"STRATEGY_MODE": "avg_tp", "STRAT_TP_LADDER": "2.0:60,5.0:60"})

    with pytest.raises(ValueError, match="Invalid STRAT_TP_LADDER"):
        t212_strat.StratParams.from_env({"STRATEGY_MODE": "avg_tp", "STRAT_TP_LADDER": "0.0:50"})

def test_priceanalysis_hyperliquid_price_analysis_flat_window_and_invalid_step_hours_iter94_s4(monkeypatch):
    """[ITER94 #4] priceAnalysis & hyperliquid.price_analysis flat-window and invalid step_hours protection."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 4. priceAnalysis & hyperliquid.price_analysis flat-window and invalid step_hours protection
    import priceAnalysis as pa

    ts_flat = [1000.0 + i * 3600.0 for i in range(60)]
    px_flat = [100.0 for _ in range(60)]
    assert pa.detect_long_term_trend(ts_flat, px_flat, window_hours=24, step_hours=8) is None
    assert pa.detect_long_term_trend(ts_flat, px_flat, window_hours=24, step_hours=0) is None
    assert pa.detect_long_term_trend(ts_flat, px_flat, window_hours=24, step_hours=-4) is None

    # Unsorted timestamps should still detect valid uptrend and blocks copy must not mutate cache
    px_up = [100.0 + i * 1.5 for i in range(60)]
    res1 = pa.detect_long_term_trend(list(reversed(ts_flat)), list(reversed(px_up)), window_hours=24, step_hours=8)
    assert res1 is not None and res1["direction"] == "up"
    res1["blocks"].clear()
    res2 = pa.detect_long_term_trend(ts_flat, px_up, window_hours=24, step_hours=8)
    assert res2 is not None and len(res2["blocks"]) >= 3

    hl_pa = load_local_module(os.path.join(root_dir, "hyperliquid"), "signals", module_key="hl_pa_iter94", reload=True)
    assert hl_pa.detect_long_term_trend(ts_flat, px_flat, window_hours=24, step_hours=8) is None

def test_hyperliquidprovider_token_live_enabled_order_id_ohlc_series_validation_iter95_s3():
    """[ITER95 #3] HyperliquidProvider token/live_enabled/order_id/ohlc_series validation."""
    from providers.hyperliquid_provider import _spot_symbol_alias
    # 3. HyperliquidProvider token/live_enabled/order_id/ohlc_series validation
    assert _spot_symbol_alias(True, "HYPE") is False
    with pytest.raises(ValueError, match="invalid Hyperliquid token"):
        HyperliquidProvider(token=True)
    hl_p = HyperliquidProvider(token=" hype ", live_enabled="false")
    assert hl_p._token == "HYPE"
    assert hl_p.execution_enabled() is False
    assert hl_p.free_balance("") is None
    assert hl_p.free_balance(True) is None
    with pytest.raises(ProviderError, match="invalid interval_min"):
        hl_p.ohlc_series("HYPEUSDC", 15.5)
    with pytest.raises(ProviderError, match="invalid order id"):
        hl_p.order_status("HYPEUSDC", 123.7)
    with pytest.raises(ProviderError, match="invalid order id"):
        hl_p.cancel_order("HYPEUSDC", 123.7)

def test_hyperliquidprovider_iter111_s2():
    """[ITER111 #2] HyperliquidProvider."""
    # 2. HyperliquidProvider
    hl_prov = hl_provider.HyperliquidProvider.__new__(hl_provider.HyperliquidProvider)
    hl_prov._token = "PURR"
    hl_prov._symbol = "PURR"
    hl_prov.validate_symbol = lambda _s: None
    hl_prov.preflight_order("PURR", True, 1.0)
    hl_prov._hl = lambda: SimpleNamespace(sz_decimals=lambda _t: True)
    with pytest.raises(ProviderError):
        hl_prov.pair_precision("PURR")
    hl_prov._hl = lambda: SimpleNamespace(sz_decimals=lambda _t: -1)
    with pytest.raises(ProviderError):
        hl_prov.pair_precision("PURR")

def test_hlclient_strict_position_spot_balance_market_data_guards_iter115_s4():
    """[ITER115 #4] HLClient strict position/spot_balance & market_data guards."""
    hlc, hlmd = load_venue_modules("hyperliquid", "hl_client", "market_data")
    HLClient, HLError = hlc.HLClient, hlc.HLError
    # 4. HLClient strict position/spot_balance & market_data guards
    hl = HLClient.__new__(HLClient)
    hl.address = "0x123"
    hl.exchange = None
    hl._user_state = lambda: {"assetPositions": [{"position": {"coin": "HYPE", "szi": "bad", "entryPx": "10"}}]}
    with pytest.raises(HLError):
        hl.position_strict("HYPE")
    hl.info = SimpleNamespace(
        candles_snapshot=lambda *_a: ["not-a-dict", {"t": 1}],
    )
    assert hl.candles("", "1h", 10) == []
    assert hl.candles("HYPE", "", 10) == []
    assert hl.candles("HYPE", "1h", 10) == [{"t": 1}]
    assert hlmd.normalize_closed_candles("not-a-list", now_ms=1000) == []
    assert hlmd.normalize_closed_candles([{"t": 1, "T": 2, "o": 1, "h": 2, "l": 1, "c": 1.5}], now_ms=True) == []
    with pytest.raises(ValueError):
        hlmd.fetch_ohlc_candles("HYPE", 60.0)
    assert hlmd.fetch_funding_history("HYPE", 1000, timeout=0) == []

def test_hl_and_t212_strategy_state_sanitization_iter115_s5():
    """[ITER115 #5] HL and T212 strategy state sanitization."""
    hl_strat = load_venue_modules("hyperliquid", "strategy")
    t212_strat = load_venue_modules("212trading", "strategy")
    # 5. HL and T212 strategy state sanitization
    hl_clean = hl_strat._sanitize_state({
        "orders": [
            {"role": "open", "px": 10.0, "sz": 1.0},  # missing oid
            {"role": "open", "oid": True, "px": 10.0, "sz": 1.0},  # bool oid
            {"role": "open", "oid": "123", "px": 10.0, "sz": 1.0},
        ]
    })
    assert len(hl_clean["orders"]) == 1 and hl_clean["orders"][0]["oid"] == "123"

    t212_clean = t212_strat._sanitize_t212_state({
        "pending_submit": {"side": "INVALID", "qty": 1.0, "limit": 10.0, "intent_id": "id1"},
        "orders": [
            {"side": "BUY", "qty": 1.0, "limit": 10.0},  # missing id
            {"id": "o1", "side": "INVALID", "qty": 1.0, "limit": 10.0},
            {"id": " o2 ", "side": " buy ", "qty": "2.5", "limit": "15.0"},
        ],
    })
    assert t212_clean["pending_submit"] is None
    assert len(t212_clean["orders"]) == 1
    assert t212_clean["orders"][0]["id"] == "o2"
    assert t212_clean["orders"][0]["side"] == "BUY"

def test_hlclient_spot_order_is_market_tif_validation_iter116_s3():
    """[ITER116 #3] HLClient.spot_order is_market & tif validation."""
    HLClient = load_venue_modules("hyperliquid", "hl_client").HLClient
    # 3. HLClient.spot_order is_market & tif validation
    hl = HLClient.__new__(HLClient)
    hl.exchange = SimpleNamespace(order=lambda *a, **kw: {"status": "ok"})
    assert hl.spot_order("@107", True, 1.0, 10.0, is_market="false")[0] is False
    assert hl.spot_order("@107", True, 1.0, 10.0, tif=True)[0] is False
    assert hl.spot_order("@107", True, 1.0, 10.0, tif="   ")[0] is False

def test_t212client___init___place_market_order_extended_hours_false_and_get_historical_order_currency_iter116_s4(monkeypatch):
    """[ITER116 #4] T212Client.__init__, place_market_order extended_hours='false', and get_historical_order currency=True."""
    t212c = load_venue_modules("212trading", "t212_client")
    T212Client = t212c.T212Client
    # 4. T212Client.__init__, place_market_order extended_hours="false", and get_historical_order currency=True
    with pytest.raises(ValueError):
        T212Client(api_key="", env="demo", min_gap_sec=0, portfolio_ttl_sec=1, order_min_gap_sec=0)
    with pytest.raises(ValueError):
        T212Client(api_key=True, env="demo", min_gap_sec=0, portfolio_ttl_sec=1, order_min_gap_sec=0)
    with pytest.raises(ValueError):
        T212Client(api_key="key", api_secret=True, env="demo", min_gap_sec=0, portfolio_ttl_sec=1, order_min_gap_sec=0)
    tc = T212Client(api_key="key", env="demo", min_gap_sec=0, portfolio_ttl_sec=1, order_min_gap_sec=0)
    sent_payload = {}
    monkeypatch.setattr(tc, "_place_order", lambda ep, payload: sent_payload.update(payload) or (200, {"id": 1}))
    tc.place_market_order("AAPL_US_EQ", 1.0, extended_hours="false")
    assert sent_payload["extendedHours"] is False
    monkeypatch.setattr(
        t212c,
        "http_get",
        lambda *_a, **_kw: (
            200,
            b'{"items":[{"order":{"id":"O1"},"fill":{"quantity":1,"price":10,"walletImpact":{"taxes":[{"quantity":0.1,"currency":true}]}}}]}',
        ),
    )
    hist = tc.get_historical_order("O1")
    assert "_feeCurrencies" not in hist

def test_hl_stratparams_from_dict_rejects_direction_false_and_out_iter118_s1():
    """[ITER118 #1] HL StratParams.from_dict rejects direction=False and out-of-range sz_decimals; sanitizes currency=True."""
    hl_strat = load_venue_modules("hyperliquid", "strategy")
    # 1. HL StratParams.from_dict rejects direction=False and out-of-range sz_decimals; sanitizes currency=True
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"direction": False})
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"sz_decimals": -1})
    with pytest.raises(ValueError):
        hl_strat.StratParams.from_dict({"sz_decimals": 25})
    assert hl_strat.StratParams.from_dict({"currency": True}).currency == "USDC"

def test_hl_strategy__apply_close_preserves_avg_cost_and_updates_iter118_s2(monkeypatch):
    """[ITER118 #2] HL Strategy._apply_close preserves avg cost and updates spent when real_avg=0 on partial close."""
    hl_strat = load_venue_modules("hyperliquid", "strategy")
    # 2. HL Strategy._apply_close preserves avg cost and updates spent when real_avg=0 on partial close
    monkeypatch.setattr(hl_strat, "notify", lambda *a, **kw: None)
    st_hl = hl_strat.Strategy(
        client=None,
        coin="HYPE",
        params=hl_strat.StratParams(sz_decimals=99),
        dry_run=True,
        replay_mode=True,
    )
    assert st_hl.sz_dec == 18
    st_hl.s["qty"] = 10.0
    st_hl.s["cost"] = 100.0
    st_hl.s["spent"] = 100.0
    st_hl._apply_close(10.0, 12.0, 4.0, real_qty=6.0, real_avg=0.0)
    assert st_hl.s["qty"] == 6.0
    assert st_hl.s["cost"] == 60.0
    assert st_hl.s["spent"] == 60.0

def test_hl_replay_validates_ohlc_envelope_and_accepts_tuple_iter118_s3():
    """[ITER118 #3] HL replay validates OHLC envelope and accepts 5-tuple bars (o, h, l, c, ts)."""
    hl_replay = load_venue_modules("hyperliquid", "replay")
    # 3. HL replay validates OHLC envelope and accepts 5-tuple bars (o, h, l, c, ts)
    res_5t = hl_replay.run_replay([(100.0, 101.0, 99.0, 100.0, 1000)])
    assert res_5t["initial_capital"] == 500.0
    with pytest.raises(ValueError, match="has non-finite or invalid prices"):
        hl_replay.run_replay([(105.0, 101.0, 99.0, 100.0)])

def test_t212_strategy__reconcile_real_falls_back_to_order_limit_iter118_s4(monkeypatch):
    """[ITER118 #4] T212 Strategy._reconcile_real falls back to order limit/real_avg when marginal_px <= 0."""
    t212_strat = load_venue_modules("212trading", "strategy")
    # 4. T212 Strategy._reconcile_real falls back to order limit/real_avg when marginal_px <= 0
    monkeypatch.setattr(t212_strat, "notify", lambda *a, **kw: None)
    dummy_client = SimpleNamespace(
        get_portfolio=lambda: [{"ticker": "TEST_US_EQ", "quantity": 1.01, "averagePrice": 99.0}],
        list_active_orders=lambda: [],
    )
    t_params = t212_strat.StratParams(
        currency="USD", entry_amount=100.0, entry_discount_pct=1.0, dca_amount=10.0,
        dca_drop_pct=2.0, check_minutes=5.0, takeprofit_pct=2.0, max_budget=1000.0,
        max_dca_buys=5, validity="GOOD_TILL_CANCEL", enable_takeprofit=True,
        order_ttl_min=60.0, stop_loss_pct=5.0,
    )
    t_eng = t212_strat.Strategy(
        dummy_client,
        "TEST_US_EQ",
        t_params,
        dry_run=False,
        initial_state={
            "qty": 1.0,
            "cost_usd": 100.0,
            "spent_cash": 100.0,
            "entry_price": 100.0,
            "last_buy_price": 100.0,
            "orders": [{"id": "B1", "side": "BUY", "qty": 0.01, "limit": 95.0, "kind": "DCA"}],
        },
        fx_to_usd=1.0,
        clock=lambda: 1000.0,
    )
    t_eng._save = lambda: None
    t_eng._tracked_order_status = lambda _o, _c: None
    assert t_eng._reconcile_real(96.0) is True
    assert t_eng.s["last_buy_price"] == 95.0

def test_hyperliquid_backtest_precompute_signal_validates_signal_source_and_normalizes_iter119_s4():
    """[ITER119 #4] hyperliquid.backtest.precompute_signal validates signal source and normalizes case/whitespace."""
    hl_bt = load_venue_modules("hyperliquid", "backtest")
    # 4. hyperliquid.backtest.precompute_signal validates signal source and normalizes case/whitespace
    assert hl_bt.precompute_signal([100.0, 101.0], [1, 2], " OFF ", 2, 4, 0.01, 24, 8) == ["__any__", "__any__"]
    with pytest.raises(ValueError):
        hl_bt.precompute_signal([100.0, 101.0], [1, 2], "invalid_source", 2, 4, 0.01, 24, 8)
    with pytest.raises(ValueError):
        hl_bt.precompute_signal([100.0, 101.0], [1, 2], True, 2, 4, 0.01, 24, 8)

def test_canonical_venue_and__detect_venue_handle_bool_provider_objects_and_iter134_s3():
    """[ITER134 #3] canonical_venue and _detect_venue handle bool, provider objects, and empty/whitespace strings."""
    from offline.runners.walk_forward import canonical_venue
    from offline.runners.walk_forward_compare import _detect_venue
    # 3. canonical_venue and _detect_venue handle bool, provider objects, and empty/whitespace strings
    assert canonical_venue(True) is None  # type: ignore[arg-type]
    assert canonical_venue("   ") is None
    assert canonical_venue(SimpleNamespace(name=" Trading212 ")) == "t212"  # type: ignore[arg-type]
    assert _detect_venue({"venue": "   ", "purpose": "Binance walk-forward"}, cli_venue="  ") == "binance"
    assert _detect_venue({"venue": True, "purpose": "Hyperliquid"}, cli_venue=False) == "hyperliquid"  # type: ignore[arg-type]

def test_intent_client_order_id_and_auditedstrategyexecutor_reject_sentinel_strings_and_containers_iter141_s1(tmp_path):
    """[ITER141 #1] intent_client_order_id and AuditedStrategyExecutor reject sentinel strings and containers."""
    from providers.execution_audit import AuditedStrategyExecutor, ExecutionAudit, intent_client_order_id
    # 1. intent_client_order_id and AuditedStrategyExecutor reject sentinel strings and containers
    for bad_intent in ("nan", "null", "none", "inf", ["abc"], {"id": 1}, True, None, ""):
        assert intent_client_order_id("binance", bad_intent) is None  # type: ignore[arg-type]
        assert intent_client_order_id("kraken", bad_intent) is None  # type: ignore[arg-type]
        assert intent_client_order_id("hyperliquid", bad_intent) is None  # type: ignore[arg-type]

    from providers.execution_audit import clean_order_id_str
    assert clean_order_id_str("nan") == ""
    assert clean_order_id_str(["123"]) == ""
    assert clean_order_id_str(12345.0) == "12345"

    seen_submits = []
    fake_exec = SimpleNamespace(
        name="Binance",
        submit_order=lambda sym, side, qty, px, **kw: seen_submits.append(kw.get("client_order_id")) or "98765",
    )
    audited = AuditedStrategyExecutor(fake_exec, audit=ExecutionAudit(directory=str(tmp_path)))
    oid = audited.submit_order_with_intent(
        "binance-BTCUSDC-entry-0123456789abcdef0123456789abcdef",
        "BTCUSDC",
        "BUY",
        1.0,
        50000.0,
        client_order_id="nan",
    )
    assert oid == "98765"
    assert seen_submits == ["SD_0123456789abcdef0123456789abcdef"]

def test_hyperliquidprovider_open_orders_skips_entries_with_missing_zero_sentinel_iter141_s3():
    """[ITER141 #3] HyperliquidProvider.open_orders skips entries with missing/zero/sentinel orderId."""
    # 3. HyperliquidProvider.open_orders skips entries with missing/zero/sentinel orderId
    hl = HyperliquidProvider(token="HYPE", live_enabled=False)
    hl._spot_pair = "@107"
    hl._client = SimpleNamespace(
        open_orders=lambda pair: [
            {"oid": "nan", "side": "B", "limitPx": "25.0", "sz": "1.0", "origSz": "1.0"},
            {"oid": 0, "side": "B", "limitPx": "25.0", "sz": "1.0", "origSz": "1.0"},
            {"oid": 888.0, "side": "B", "limitPx": "25.0", "sz": "1.0", "origSz": "1.0"},
        ]
    )
    hl_open = hl.open_orders("HYPEUSDC")
    assert len(hl_open) == 1
    assert hl_open[0]["orderId"] == "888"

def test_hyperliquid_signals__file_signal_guards_directory_bool_empty_paths_iter147_s3(tmp_path):
    """[ITER147 #3] hyperliquid/signals._file_signal guards directory/bool/empty paths."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 3. hyperliquid/signals._file_signal guards directory/bool/empty paths
    hl_sig_mod = load_local_module(os.path.join(root, "hyperliquid"), "signals", module_key="hl_sig_t147", register=False)

    assert hl_sig_mod._file_signal("")["source"] == "file(missing)"
    assert hl_sig_mod._file_signal(True)["source"] == "file(missing)"  # type: ignore[arg-type]
    assert hl_sig_mod._file_signal(str(tmp_path))["source"] == "file(missing)"

def test_hyperliquidprovider_execution_enabled_rejects_nan_false_containers_iter152_s1():
    """[ITER152 #1] HyperliquidProvider execution_enabled rejects NaN / 'false' / containers."""
    from providers.hyperliquid_provider import _parse_bool_flag as hl_bool
    # 1. HyperliquidProvider execution_enabled rejects NaN / "false" / containers
    assert hl_bool("false", True) is False
    assert hl_bool(float("nan"), False) is False
    assert HyperliquidProvider(live_enabled="false").execution_enabled() is False  # type: ignore[arg-type]
    assert HyperliquidProvider(live_enabled=float("nan")).execution_enabled() is False  # type: ignore[arg-type]
    assert HyperliquidProvider(live_enabled="true").execution_enabled() is True  # type: ignore[arg-type]

def test_hyperliquid_shadow_longterm__fetch_with_ts_deduplicates_identical_timestamps_iter153_s2(monkeypatch):
    """[ITER153 #2] hyperliquid/shadow_longterm._fetch_with_ts deduplicates identical timestamps."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    hl_sl = load_local_module(os.path.join(root, "hyperliquid"), "shadow_longterm.py", module_key="hl_shadow_longterm_it153", reload=True)
    # 2. hyperliquid/shadow_longterm._fetch_with_ts deduplicates identical timestamps
    class DummyPubClient:
        def candles(self, token, interval, lookback_hours=0):
            return [
                {"t": 1700000000000, "o": "10", "h": "12", "l": "9", "c": "11"},
                {"t": 1700000000000, "o": "10", "h": "13", "l": "9", "c": "12"},
                {"t": 1700014400000, "o": "12", "h": "14", "l": "11", "c": "13"},
            ]

    import hl_client as hlc_mod
    monkeypatch.setattr(hlc_mod.HLClient, "public", staticmethod(lambda: DummyPubClient()))
    rows = hl_sl._fetch_with_ts("HYPE-HL", 240)
    assert len(rows) == 1
    assert rows[0] == (1700000000, 10.0, 13.0, 9.0, 12.0)

def test_hyperliquid_archive_delta_neutral_funding_scan_py_rank__spot_tokens_safety_iter153_s3():
    """[ITER153 #3] hyperliquid/archive/delta_neutral/funding_scan.py rank & _spot_tokens safety."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    f_scan = load_local_module(
        os.path.join(root, "archive/hyperliquid/delta_neutral"),
        "funding_scan.py",
        module_key="hl_funding_scan_it153",
        reload=True,
    )
    # 3. hyperliquid/archive/delta_neutral/funding_scan.py rank & _spot_tokens safety
    ranked = f_scan.rank(
        [None, {"name": "BAD_BOOL"}, {"name": "NAN_F"}, {"name": "HYPE"}],
        [
            None,
            {"funding": True, "dayNtlVlm": 50_000_000, "markPx": 20, "openInterest": 1000},
            {"funding": "nan", "dayNtlVlm": 50_000_000, "markPx": 20, "openInterest": 1000},
            {"funding": "0.0005", "dayNtlVlm": "25000000", "markPx": "25.0", "openInterest": "10000"},
        ],
        min_vol_usd=float("nan"),
        top=True,
    )
    assert len(ranked) == 1 and ranked[0]["coin"] == "HYPE"
    bad_spot_client = SimpleNamespace(info=SimpleNamespace(spot_meta=lambda: {"tokens": [None], "universe": [None]}))
    assert f_scan._spot_tokens(bad_spot_client) == set()

def test_hyperliquid_archive_delta_neutral_delta_neutral_py_state_path_for__load_sanitization_iter153_s4(tmp_path, monkeypatch):
    """[ITER153 #4] hyperliquid/archive/delta_neutral/delta_neutral.py state_path_for & _load sanitization."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dn_mod = load_local_module(
        os.path.join(root, "archive/hyperliquid/delta_neutral"),
        "delta_neutral.py",
        module_key="hl_delta_neutral_it153",
        reload=True,
    )
    # 4. hyperliquid/archive/delta_neutral/delta_neutral.py state_path_for & _load sanitization
    assert dn_mod.state_path_for("").endswith(".state_dn_UNKNOWN.json")
    assert dn_mod.state_path_for(True).endswith(".state_dn_UNKNOWN.json")  # type: ignore[arg-type]
    params = dn_mod.DNParams(
        coin="HYPE", spot_pair="@107", spot_token="HYPE", notional=100.0,
        entry_funding_hr=0.0001, exit_funding_hr=0.0, funding_window_h=4.0,
        min_hold_h=1.0, rebalance_pct=5.0, check_minutes=5.0, sz_decimals=2,
        liq_alert_pct=15.0, auto_protect=True, reduce_pct=25.0, perp_leverage=2,
        allow_scale_up=False, fee_pct=0.05,
    )
    corrupt_state = tmp_path / ".state_dn_HYPE.json"
    corrupt_state.write_text(json.dumps({
        "status": "invalid_status",
        "target_sz": "nan",
        "fees_paid": True,
        "liq_alerted": "false",
        "orphan_count": -5,
        "opened_ts": "nan",
        "funding_hist": ["bad", [True, 0.01], [1700000000.0, "nan"], [1700000000.0, 0.0002]],
    }), encoding="utf-8")
    monkeypatch.setattr(dn_mod, "state_path_for", lambda c: str(corrupt_state))
    dn = dn_mod.DeltaNeutral(client=SimpleNamespace(), params=params, dry_run=True)
    assert dn.s["status"] == "flat"
    assert dn.s["target_sz"] == 0.0
    assert dn.s["fees_paid"] == 0.0
    assert dn.s["liq_alerted"] is False
    assert dn.s["orphan_count"] == 0
    assert dn.s["opened_ts"] is None
    assert dn.s["funding_hist"] == [[1700000000.0, 0.0002]]

def test_simulate_sys_modules_market_data_polluted_by_hyperliquid_market_data_iter154_s1(monkeypatch):
    """[ITER154 #1] Simulate sys.modules['market_data'] polluted by hyperliquid/market_data.py."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # 1. Simulate sys.modules["market_data"] polluted by hyperliquid/market_data.py
    hl_md = load_local_module(os.path.join(root, "hyperliquid"), "market_data.py", module_key="hl_market_data_polluter", reload=True)
    monkeypatch.setitem(sys.modules, "market_data", hl_md)
    om_mod = load_local_module(os.path.join(root, "212trading"), "order_manager.py", module_key="t212_om_it154", reload=True)
    assert callable(getattr(om_mod, "get_usd_ron", None))

def test_hyperliquidprovider_underscore_alias_and_long_ask_side_normalization_iter169_s2(monkeypatch):
    """[ITER169 #2] HyperliquidProvider underscore alias and LONG/ASK side normalization."""
    # 2. HyperliquidProvider underscore alias and LONG/ASK side normalization
    hl = hl_provider.HyperliquidProvider(token="HYPE", live_enabled=False)
    assert hl.supports_symbol("HYPE_USDC") is True
    monkeypatch.setattr(hl, "free_balance", lambda asset: 10.0)
    with pytest.raises(ProviderError, match="insufficient USDC"):
        hl.preflight_order("HYPE_USDC", "LONG", 2.0, price=25.0)
