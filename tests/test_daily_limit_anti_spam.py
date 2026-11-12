"""Unit tests for symmetric GO1 (30 filled orders / 24h) and Kraken multi-fill aggregation."""

import json
from tests.conftest import TempDirTestCase
import order_guard
import order_retry
import os
import threading
import time
import unittest
from instrument import Instrument
from providers.base import aggregate_fills_by_order_id
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import KrakenProvider
from providers.market_api import MarketApi, MarketDataProvider
from providers.strategy_executor import ProviderError
from types import SimpleNamespace
from unittest.mock import patch


class FakeProvider(MarketDataProvider):
    """Minimal in-memory provider whose guards are handled by Instrument."""

    def __init__(self, name="FakeVenue", price=100.0, *, price_history=None, biz_min=False, observed_flags=None):
        self._name = name
        self._price = price
        self._orders = []
        self.placed = []
        self._price_history = list(price_history or [])
        self._biz_min = biz_min
        self._observed_flags = observed_flags

    @property
    def name(self):
        return self._name

    def get_current_price(self, symbol):
        return self._price

    def supports_symbol(self, symbol):
        return False

    def supports_business_minimum(self) -> bool:
        return self._biz_min

    def get_price_history(self, symbol: str, lookback_sec: float):
        return list(self._price_history)

    def order_filter_refusal(self, symbol, side, price, qty, *, market=False, enforce_business_minimum=True):
        if self._observed_flags is not None:
            self._observed_flags.append(("filter", enforce_business_minimum))
        if self._biz_min and enforce_business_minimum and (price * qty) < 25.0:
            return "filter_business_min_notional"
        return super().order_filter_refusal(
            symbol, side, price, qty, market=market, enforce_business_minimum=enforce_business_minimum
        )

    def free_balance(self, asset):
        return 1_000_000.0

    def get_orders(self, symbol, side, since_s):
        cutoff_ms = time.time() * 1000 - since_s * 1000
        out = [o for o in self._orders
               if o.get("symbol") == symbol and o.get("timestamp", 0) >= cutoff_ms]
        if side:
            out = [o for o in out if o.get("side") == side.upper()]
        return out

    def place_order(self, symbol, side, price, qty, **kwargs):
        if self._observed_flags is not None:
            self._observed_flags.append(("place", kwargs.get("enforce_business_minimum")))
        payload = {"orderId": len(self.placed) + 1, "symbol": symbol, "side": side, "price": price, "qty": qty}
        self.placed.append(payload)
        return payload

    def guards_internally(self):
        return False

    def seed_trade(self, side, age_sec=0.0, price=100.0, qty=1.0, symbol="BTCUSDC"):
        self._orders.append({
            "orderId": len(self._orders) + 1,
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "qty": qty,
            "timestamp": (time.time() - age_sec) * 1000.0
        })


class TestSymmetricDailyLimitGuard(TempDirTestCase):
    def setUp(self):
        super().setUp()
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def tearDown(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0
        super().tearDown()

    def test_recent_trades_below_daily_cap_are_allowed_symmetrically(self):
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            prov.seed_trade(side, age_sec=2.0, symbol="BTCUSDC")
            ok_same, reason_same = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=30)
            self.assertTrue(ok_same)
            self.assertIsNone(reason_same)

            ok_other, reason_other = order_guard.daily_limit_guard(
                prov, "TAOUSDC", side, max_daily_trades=30)
            self.assertTrue(ok_other)
            self.assertIsNone(reason_other)

    def test_daily_limit_30_orders_in_24h_blocks_buy_and_sell_symmetrically(self):
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            for _ in range(30):
                prov.seed_trade(side, age_sec=3600.0)
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=30)
            self.assertFalse(ok)
            self.assertEqual(reason, "daily_limit")

            # 29 orders in 24h is allowed
            prov_29 = FakeProvider()
            for _ in range(29):
                prov_29.seed_trade(side, age_sec=3600.0)
            ok_29, reason_29 = order_guard.daily_limit_guard(
                prov_29, "BTCUSDC", side, max_daily_trades=30)
            self.assertTrue(ok_29)
            self.assertIsNone(reason_29)

    def test_bypass_go1_or_daily_limit_or_all_skips_cap(self):
        prov = FakeProvider()
        for _ in range(35):
            prov.seed_trade("BUY", age_sec=3600.0)
        for bypass in ({"GO1"}, {"daily_limit"}, {"all"}):
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", "BUY", max_daily_trades=30,
                bypass_guards=bypass)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_daily_limit_is_non_failure_deferral_in_order_retry(self):
        self.assertTrue(order_retry.is_non_failure_deferral("daily_limit"))

    def test_kraken_get_orders_groups_multi_fill_orders_by_order_id(self):
        kp = KrakenProvider()
        now_ms = int(time.time() * 1000)
        cache_path = os.path.join(self.tmp, "trades.json")
        with open(cache_path, "w", encoding="utf-8") as tf:
            json.dump({
                "items": {
                    "HYPEUSD": [
                        {
                            "symbol": "HYPEUSD",
                            "id": "T1",
                            "orderId": "O-PARENT-1",
                            "side": "BUY",
                            "price": 40.0,
                            "qty": 2.0,
                            "timestamp": now_ms - 10_000,
                        },
                        {
                            "symbol": "HYPEUSD",
                            "id": "T2",
                            "orderId": "O-PARENT-1",
                            "side": "BUY",
                            "price": 43.0,
                            "qty": 1.0,
                            "timestamp": now_ms - 9_000,
                        },
                        {
                            "symbol": "HYPEUSD",
                            "id": "T3",
                            "orderId": "O-PARENT-2",
                            "side": "BUY",
                            "price": 44.0,
                            "qty": 1.5,
                            "timestamp": now_ms - 5_000,
                        },
                    ]
                },
                "fetchtime": {"HYPEUSD": now_ms},
            }, tf)
        with patch("providers.kraken_provider._KRAKEN_CACHE_FILE", cache_path):
            orders = kp.get_orders("HYPEUSD", "BUY", 3600.0)
        self.assertEqual(len(orders), 2)
        # First order is VWAP of (40*2 + 43*1) / 3 = 41.0, qty=3.0, max timestamp
        self.assertAlmostEqual(orders[0]["price"], 41.0)
        self.assertAlmostEqual(orders[0]["qty"], 3.0)
        self.assertEqual(orders[0]["timestamp"], now_ms - 9_000)
        self.assertEqual(orders[0]["side"], "BUY")
        # Second order
        self.assertAlmostEqual(orders[1]["price"], 44.0)
        self.assertAlmostEqual(orders[1]["qty"], 1.5)

    def test_clock_skew_future_fill_timestamp_does_not_block_below_daily_cap(self):
        """A single fill whose exchange timestamp is slightly ahead of local host time (e.g. +200ms)
        remains allowed when below the 24h trade cap."""
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            prov.seed_trade(side, age_sec=-0.2, symbol="BTCUSDC")
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=30)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_hyperliquid_multi_fill_aggregation_by_oid(self):
        now_ms = int(time.time() * 1000)
        raw_fills = [
            {"coin": "@107", "side": "B", "px": "40.0", "sz": "2.0", "time": now_ms - 4000, "oid": 999001},
            {"coin": "@107", "side": "B", "px": "43.0", "sz": "1.0", "time": now_ms - 3900, "oid": 999001},
            {"coin": "@107", "side": "B", "px": "45.0", "sz": "1.0", "time": now_ms - 1000, "oid": 999002},
        ]
        orders = aggregate_fills_by_order_id(raw_fills, cutoff_ms=now_ms - 60_000, side="BUY")
        self.assertEqual(len(orders), 2)
        self.assertAlmostEqual(orders[0]["price"], 41.0)
        self.assertAlmostEqual(orders[0]["qty"], 3.0)
        self.assertEqual(orders[0]["timestamp"], now_ms - 3900)
        self.assertEqual(orders[1]["orderId"], "999002")

    def test_instrument_place_unified_kind_and_motivation(self):
        prov = FakeProvider(name="kraken")
        api = MarketApi([prov])
        inst = Instrument(name="KRAKEN_BTC", symbol="BTCUSDC", provider="kraken",
                          base="BTC", quote="USDC", api=api)

        tmpdir = self.tmp
        qfile = os.path.join(tmpdir, "order_retry_queue.jsonl")
        lfile = os.path.join(tmpdir, "order_retry.lock")
        with patch.object(order_retry, "QUEUE_FILE", qfile), \
             patch.object(order_retry, "LOCK_FILE", lfile), \
             patch.object(order_guard, "profit_guard", return_value=True):
            res = inst.place("SELL", 100.0, 1.0, bypass_guards={"all"}, kind="STOP")
            self.assertIsNotNone(res)

    def test_daily_limit_deferral_consolidates_when_retry_dedup_disabled(self):
        """When RETRY_DEDUP=False, repeated GO1 'daily_limit' blocks for the same
        (provider, symbol, side, kind) must consolidate into a single pending retry
        record instead of piling up duplicates."""
        tmpdir = self.tmp
        qfile = os.path.join(tmpdir, "order_retry_queue.jsonl")
        lfile = os.path.join(tmpdir, "order_retry.lock")
        with patch.object(order_retry, "QUEUE_FILE", qfile), \
             patch.object(order_retry, "LOCK_FILE", lfile), \
             patch.object(order_retry, "RETRY_DEDUP", False):
            rec1 = order_retry.enqueue(
                "HYPEUSD",
                "BUY",
                1.0,
                requested_price=40.0,
                failure_reason="daily_limit",
                provider_name="kraken",
                kind="ENTRY",
            )
            rec2 = order_retry.enqueue(
                "HYPEUSD",
                "BUY",
                1.2,
                requested_price=41.5,
                failure_reason="daily_limit",
                provider_name="kraken",
                kind="ENTRY",
            )
            self.assertIsNotNone(rec1)
            self.assertIsNotNone(rec2)
            pending = order_retry._read_nolock()
            self.assertEqual(len(pending), 1)
            self.assertAlmostEqual(pending[0]["requested_price"], 41.5)
            self.assertAlmostEqual(pending[0]["qty"], 1.2)

    def test_hyperliquid_transient_startup_timeout_does_not_permanently_disable_client(self):
        """A transient network timeout during HyperliquidProvider._new_client() must not
        permanently set _client_tried=True; subsequent calls must retry initialization."""

        hp = HyperliquidProvider.__new__(HyperliquidProvider)
        hp._lock = threading.Lock()
        hp._env_loaded = True
        hp._client = None
        hp._client_tried = False

        call_count = {"n": 0}
        fake_sdk_client = object()

        def flaky_new_client():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TimeoutError("HTTPSConnectionPool(host='api.hyperliquid.xyz', port=443): Read timed out.")
            return fake_sdk_client

        hp._new_client = flaky_new_client
        # First call fails transiently -> returns None, _client_tried remains False
        self.assertIsNone(hp._hl())
        self.assertFalse(hp._client_tried)
        # Second call succeeds and caches the client
        self.assertIs(hp._hl(), fake_sdk_client)
        self.assertTrue(hp._client_tried)

    def test_market_api_get_orders_and_price_history_explicit_provider_routing(self):
        """MarketApi.get_orders and get_price_history must honor explicit provider_name
        even when the provider's supports_symbol() returns False (e.g. KrakenProvider)."""
        kraken_prov = FakeProvider(name="kraken", price_history=[(1000, 42.5)])
        default_prov = FakeProvider(name="binance")
        kraken_prov.seed_trade("BUY", age_sec=10.0, symbol="HYPEUSD")

        api = MarketApi([default_prov, kraken_prov])
        orders = api.get_orders("HYPEUSD", "BUY", 3600.0, provider_name="kraken")
        self.assertEqual(len(orders), 1)
        hist = api.get_price_history("HYPEUSD", 3600.0, provider_name="kraken")
        self.assertEqual(hist, [(1000, 42.5)])

    def test_instrument_place_honors_caller_enforce_business_minimum_false(self):
        """Instrument.place(..., enforce_business_minimum=False) must pass
        enforce_business_minimum=False to quantity_decision and place_order even
        on BUY orders (e.g. AutonomousIntentGateway $15 orders below $25 business min)."""
        observed_flags = []
        prov = FakeProvider(name="kraken", price=15.0, biz_min=True, observed_flags=observed_flags)
        api = MarketApi([prov])
        inst = Instrument(name="KRAKEN_TAO", symbol="TAOUSD", provider="kraken",
                          base="TAO", quote="USD", api=api)

        tmpdir = self.tmp
        qfile = os.path.join(tmpdir, "order_retry_queue.jsonl")
        lfile = os.path.join(tmpdir, "order_retry.lock")
        with patch.object(order_retry, "QUEUE_FILE", qfile), \
             patch.object(order_retry, "LOCK_FILE", lfile), \
             patch.object(order_guard, "profit_guard", return_value=True):
            # $15 notional BUY (price=15, qty=1.0) with enforce_business_minimum=False must succeed
            res = inst.place(
                "BUY",
                15.0,
                1.0,
                market=True,
                enforce_business_minimum=False,
                bypass_guards={"all"},
            )
            self.assertIsNotNone(res)
            self.assertIn(("filter", False), observed_flags)
            self.assertIn(("place", False), observed_flags)

    def test_hyperliquid_order_status_does_not_use_unrelated_fill_price_for_hype_fee(self):
        """When self._token != 'HYPE' and feeToken == 'HYPE', order_status must not
        fall back to the non-HYPE token's fill_price if HYPE mid price is unavailable."""

        fake_hl_info = SimpleNamespace(
            query_order_by_oid=lambda addr, oid: {
                "status": "order",
                "order": {
                    "status": "filled",
                    "order": {"origSz": "100.0", "sz": "0.0"},
                },
            },
            user_fills=lambda addr: [
                {
                    "oid": 777,
                    "sz": "100.0",
                    "px": "0.10",
                    "fee": "0.05",
                    "feeToken": "HYPE",
                }
            ],
        )
        fake_hl_client = SimpleNamespace(
            address="0x123",
            info=fake_hl_info,
            resolve_spot_pair=lambda token: None,
            spot_mid=lambda pair: None,
            mid=lambda token: None,
        )

        hp = HyperliquidProvider.__new__(HyperliquidProvider)
        hp._lock = threading.Lock()
        hp._env_loaded = True
        hp._token = "PURR"
        hp._spot_pair = "@107"
        hp._client = fake_hl_client
        hp._client_tried = True

        with patch.dict(os.environ, {"HL_ACCOUNT_ADDRESS": "0x123"}):
            with self.assertRaises(ProviderError) as ctx:
                hp.order_status("PURRUSDC", "777")
        self.assertIn("cannot value HYPE fee in USDC", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
