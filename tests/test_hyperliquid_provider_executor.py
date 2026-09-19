"""Hyperliquid StrategyExecutor contract tests with fake SDK clients."""
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from instrument import Instrument  # noqa: E402
import instrument as instrument_module  # noqa: E402
import order_guard  # noqa: E402
from providers import hyperliquid_provider as hl_provider  # noqa: E402
from providers.hyperliquid_provider import HyperliquidProvider  # noqa: E402
from providers.market_api import MarketApi  # noqa: E402
from providers.strategy_executor import (  # noqa: E402
    StrategyExecutor, OrderStatus, PairPrecision, ProviderError)


class FakeInfo:
    def __init__(self, fills, status="open"):
        self._fills = fills
        self._status = status

    def user_fills(self, addr):
        return self._fills

    def query_order_by_oid(self, addr, oid):
        if self._status == "unknownOid":
            return {"status": "unknownOid"}
        filled = sum(float(item.get("sz") or 0.0) for item in self._fills
                     if int(item.get("oid", -1)) == oid)
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

    def sz_decimals(self, coin):
        return 2

    def spot_order(self, pair, is_buy, sz, px, sz_decimals=2, cloid=None):
        self.calls.append(("spot_order", pair, is_buy, sz, px, cloid))
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
        expected = hl_provider._cloid_for_client_order_id(raw_client_id)
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

        with (
            patch.object(order_guard, "daily_limit_guard",
                         return_value=(True, None)),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(self.p, "policy_cap_quantity",
                         side_effect=lambda _s, _o, _p, qty, _a, **_k: qty),
            patch.object(self.p, "fee_cap_quantity",
                         side_effect=lambda _s, _o, _p, available: available),
            patch("instrument.trade_cooldown.trade_slot",
                  return_value=SlotContext()),
            patch.object(instrument_module._outcomes_log, "log_order_outcome"),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
