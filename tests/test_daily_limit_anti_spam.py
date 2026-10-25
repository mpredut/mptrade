"""Unit tests for symmetric GO1 (25 filled orders / 24h + 5s per-symbol anti-spam) and Kraken multi-fill aggregation."""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers.market_api import MarketApi, MarketDataProvider
from providers.kraken_provider import KrakenProvider
import order_guard
import order_retry
from instrument import Instrument


class FakeProvider(MarketDataProvider):
    """Minimal in-memory provider whose guards are handled by Instrument."""

    def __init__(self, name="FakeVenue", price=100.0):
        self._name = name
        self._price = price
        self._orders = []
        self.placed = []

    @property
    def name(self):
        return self._name

    def get_current_price(self, symbol):
        return self._price

    def supports_symbol(self, symbol):
        return False

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


class TestSymmetricDailyLimitGuard(unittest.TestCase):
    def setUp(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def tearDown(self):
        order_guard._MARGINS = None
        order_guard._MARGINS_FILE_MTIME = 0.0

    def test_5s_antispam_per_symbol_symmetric(self):
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            # Fill on BTCUSDC 2s ago (< 5s) blocks BTCUSDC
            prov.seed_trade(side, age_sec=2.0, symbol="BTCUSDC")
            ok_same, reason_same = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=25)
            self.assertFalse(ok_same)
            self.assertEqual(reason_same, "daily_limit")

            # Another symbol (TAOUSDC) in the exact same second is NOT blocked
            ok_other, reason_other = order_guard.daily_limit_guard(
                prov, "TAOUSDC", side, max_daily_trades=25)
            self.assertTrue(ok_other)
            self.assertIsNone(reason_other)

            # After 5s (>= 5s), BTCUSDC is allowed
            prov_after = FakeProvider()
            prov_after.seed_trade(side, age_sec=6.0, symbol="BTCUSDC")
            ok_after, reason_after = order_guard.daily_limit_guard(
                prov_after, "BTCUSDC", side, max_daily_trades=25)
            self.assertTrue(ok_after)
            self.assertIsNone(reason_after)

    def test_daily_limit_25_orders_in_24h_blocks_buy_and_sell_symmetrically(self):
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            for _ in range(25):
                prov.seed_trade(side, age_sec=3600.0)
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=25)
            self.assertFalse(ok)
            self.assertEqual(reason, "daily_limit")

            # 24 orders in 24h is allowed
            prov_24 = FakeProvider()
            for _ in range(24):
                prov_24.seed_trade(side, age_sec=3600.0)
            ok_24, reason_24 = order_guard.daily_limit_guard(
                prov_24, "BTCUSDC", side, max_daily_trades=25)
            self.assertTrue(ok_24)
            self.assertIsNone(reason_24)

    def test_bypass_go1_or_daily_limit_or_all_skips_cap(self):
        prov = FakeProvider()
        for _ in range(30):
            prov.seed_trade("BUY", age_sec=3600.0)
        for bypass in ({"GO1"}, {"daily_limit"}, {"all"}):
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", "BUY", max_daily_trades=25,
                bypass_guards=bypass)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_daily_limit_is_non_failure_deferral_in_order_retry(self):
        self.assertTrue(order_retry.is_non_failure_deferral("daily_limit"))

    def test_kraken_get_orders_groups_multi_fill_orders_by_order_id(self):
        kp = KrakenProvider()
        now_ms = int(time.time() * 1000)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            cache_path = tf.name
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
        try:
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
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def test_5s_antispam_clock_skew_future_fill_timestamp_still_blocks(self):
        """A fill whose exchange timestamp is slightly ahead of local host time (e.g. +200ms)
        must still trigger the 5-second anti-spam pause instead of slipping through."""
        for side in ("BUY", "SELL"):
            prov = FakeProvider()
            prov.seed_trade(side, age_sec=-0.2, symbol="BTCUSDC")
            ok, reason = order_guard.daily_limit_guard(
                prov, "BTCUSDC", side, max_daily_trades=25)
            self.assertFalse(ok)
            self.assertEqual(reason, "daily_limit")

    def test_hyperliquid_multi_fill_aggregation_by_oid(self):
        from providers.base import aggregate_fills_by_order_id
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

        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
        import threading
        from providers.hyperliquid_provider import HyperliquidProvider

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
        class ExplicitOnlyProvider(FakeProvider):
            def supports_symbol(self, symbol: str) -> bool:
                return False

            def get_price_history(self, symbol: str, lookback_sec: float):
                return [(1000, 42.5)]

        kraken_prov = ExplicitOnlyProvider(name="kraken")
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

        class BizMinProvider(FakeProvider):
            def supports_business_minimum(self) -> bool:
                return True

            def order_filter_refusal(self, symbol, side, price, qty, *, market=False, enforce_business_minimum=True):
                observed_flags.append(("filter", enforce_business_minimum))
                if enforce_business_minimum and (price * qty) < 25.0:
                    return "filter_business_min_notional"
                return None

            def place_order(self, symbol, side, price, qty, **kwargs):
                observed_flags.append(("place", kwargs.get("enforce_business_minimum")))
                return super().place_order(symbol, side, price, qty, **kwargs)

        prov = BizMinProvider(name="kraken", price=15.0)
        api = MarketApi([prov])
        inst = Instrument(name="KRAKEN_TAO", symbol="TAOUSD", provider="kraken",
                          base="TAO", quote="USD", api=api)

        with tempfile.TemporaryDirectory() as tmpdir:
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
        import threading
        from providers.hyperliquid_provider import HyperliquidProvider
        from providers.strategy_executor import ProviderError

        class FakeHLInfo:
            def query_order_by_oid(self, addr, oid):
                return {
                    "status": "order",
                    "order": {
                        "status": "filled",
                        "order": {"origSz": "100.0", "sz": "0.0"},
                    },
                }

            def user_fills(self, addr):
                return [
                    {
                        "oid": 777,
                        "sz": "100.0",
                        "px": "0.10",
                        "fee": "0.05",
                        "feeToken": "HYPE",
                    }
                ]

        class FakeHLClient:
            address = "0x123"
            info = FakeHLInfo()

            def resolve_spot_pair(self, token):
                return None

            def spot_mid(self, pair):
                return None

            def mid(self, token):
                return None

        hp = HyperliquidProvider.__new__(HyperliquidProvider)
        hp._lock = threading.Lock()
        hp._env_loaded = True
        hp._token = "PURR"
        hp._spot_pair = "@107"
        hp._client = FakeHLClient()
        hp._client_tried = True

        with patch.dict(os.environ, {"HL_ACCOUNT_ADDRESS": "0x123"}):
            with self.assertRaises(ProviderError) as ctx:
                hp.order_status("PURRUSDC", "777")
        self.assertIn("cannot value HYPE fee in USDC", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
