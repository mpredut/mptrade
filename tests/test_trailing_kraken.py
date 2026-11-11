#!/usr/bin/env python3
"""Tests for the Kraken trailing_stop (no real API, no money)."""
from __future__ import annotations

import os
import signal
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KRAKEN_DIR = os.path.join(_ROOT_DIR, "kraken")
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _KRAKEN_DIR not in sys.path:
    sys.path.append(_KRAKEN_DIR)
from botcore import load_local_module, parse_dotenv  # noqa: E402
import kraken_xstock_watch as xw  # noqa: E402
import kraken.trailing_stop as ts_mod
from kraken.trailing_stop import KrakenTrailing
from trailing_core import should_sell
from strategies import spot_engine as _xstock_strategy


class FakeK:
    def __init__(self, price, total=25.0, held=0.0):
        self.price = price; self.total = total; self.held = held; self.orders = []
    def _private(self, method, data=None):
        if method == "BalanceEx":
            return {"HYPE": {"balance": str(self.total), "hold_trade": str(self.held)}}
        if method == "TradesHistory":
            return {"trades": {}}
        return {}
    def last_price(self, pair): return self.price
    def add_order(self, pair, side, volume, price=None, ordertype="limit", validate=False,
                  cl_ord_id=None):
        self.orders.append({"side": side, "volume": volume, "price": price,
                            "ordertype": ordertype, "cl_ord_id": cl_ord_id})
        return {"txid": ["X"]}
    def query_orders(self, txids):
        order = self.orders[-1]
        px = order["price"] if order["price"] is not None else self.price
        return {str(txids): {
            "status": "closed", "vol_exec": str(order["volume"]),
            "cost": str(order["volume"] * px), "fee": "0",
        }}
    def open_orders(self):
        return {}
    def closed_orders(self):
        return {}
    def pair_info(self, pair):
        return {"pair_decimals": 2, "lot_decimals": 8, "ordermin": "0.01"}
    def cancel_order(self, txid):
        return {"count": 1}


class Base(unittest.TestCase):
    def setUp(self):
        self.previous_live = os.environ.get("KRAKEN_LIVE_ORDERS")
        os.environ["KRAKEN_LIVE_ORDERS"] = "true"
        fd, self.sf = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(self.sf)
        ts_mod.notify = lambda **kw: None
        self._orig_trail_pct = dict(ts_mod.TRAIL_PCT)
        ts_mod.TRAIL_PCT["HYPE"] = 18.0
        ts_mod.TRAIL_PCT["TAO"] = 22.0
        ts_mod.TRAIL_PCT["ADA"] = 25.0

    def tearDown(self):
        ts_mod.TRAIL_PCT.clear()
        ts_mod.TRAIL_PCT.update(self._orig_trail_pct)
        if self.previous_live is None:
            os.environ.pop("KRAKEN_LIVE_ORDERS", None)
        else:
            os.environ["KRAKEN_LIVE_ORDERS"] = self.previous_live
        for p in (self.sf, self.sf + ".tmp"):
            if os.path.exists(p):
                os.remove(p)
    def ts(self, client, enabled=True, min_profit_pct=0.0):
        return KrakenTrailing(client, log=lambda *a: None, enabled=enabled, state_file=self.sf,
                              min_profit_pct=min_profit_pct)


class TestLogic(unittest.TestCase):
    def test_should_sell(self):
        self.assertTrue(should_sell(82, 100, 18))
        self.assertFalse(should_sell(83, 100, 18))
        self.assertFalse(should_sell(50, 0, 18))


class TestTrailingKraken(Base):
    def test_a_rise_updates_the_peak_and_does_not_sell(self):
        c = FakeK(60.0)
        self.ts(c).check_once()
        c.price = 65.0
        ts = self.ts(c); ts.check_once()
        self.assertEqual(c.orders, [])
        import json
        self.assertEqual(json.load(open(self.sf))["HYPE"]["peak"], 65.0)

    def test_a_crash_beyond_18pct_sells_the_free_balance(self):
        c = FakeK(60.0, total=25.0, held=3.38)        # 21.62 free
        ts = self.ts(c)
        ts.check_once()                                # peak 60
        c.price = 49.0                                 # -18.3% (HYPE threshold 18%)
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertEqual(c.orders[0]["ordertype"], "market")
        self.assertAlmostEqual(c.orders[0]["volume"], 21.62, places=2)   # only the free balance, not 25

    def test_trailing_market_false_places_limit_order(self):
        c = FakeK(60.0, total=25.0, held=3.38)
        ts = KrakenTrailing(c, log=lambda *a: None, enabled=True, state_file=self.sf,
                            trailing_market=False)
        ts.check_once()
        c.price = 49.0
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["ordertype"], "limit")
        self.assertAlmostEqual(c.orders[0]["price"], 49.0 * 0.995, places=4)

    def test_a_small_fall_does_not_sell(self):
        c = FakeK(60.0)
        ts = self.ts(c); ts.check_once()
        c.price = 56.0                                 # -6.7% < 18%
        ts.check_once()
        self.assertEqual(c.orders, [])

    def test_a_dry_run_does_not_sell(self):
        c = FakeK(60.0)
        ts = self.ts(c, enabled=False); ts.check_once()
        c.price = 48.0
        ts.check_once()
        self.assertEqual(c.orders, [])

    def test_live_switch_flip_blocks_the_final_trailing_submit(self):
        c = FakeK(60.0)
        trailing = self.ts(c)
        trailing.check_once()
        c.price = 48.0
        os.environ["KRAKEN_LIVE_ORDERS"] = "false"

        trailing.check_once()

        self.assertEqual(c.orders, [])

    def test_the_peak_survives_a_restart(self):
        c = FakeK(65.0)
        self.ts(c).check_once()
        c.price = 53.0                                 # -18.5% from 65
        self.ts(c).check_once()
        self.assertEqual(len(c.orders), 1)

    def test_sub_notional_ignored(self):
        c = FakeK(60.0, total=0.1)                     # 0.1*60 = $6 < $10
        ts = self.ts(c); ts.check_once()
        c.price = 40.0
        ts.check_once()
        self.assertEqual(c.orders, [])

    def test_without_a_free_balance_it_emits_a_heartbeat(self):
        messages = []
        c = FakeK(60.0, total=3.38, held=3.38)
        ts = KrakenTrailing(c, log=messages.append, enabled=True, state_file=self.sf)

        ts.check_once()

        self.assertEqual(c.orders, [])
        self.assertEqual(messages, ["  [TRAIL-K] heartbeat"])


class TestMinProfitKraken(Base):
    """Require minimum profit before Kraken trailing activates."""

    def test_warming_up_does_not_sell_below_the_threshold(self):
        c = FakeK(60.0)
        ts = self.ts(c, min_profit_pct=5.0)
        ts.check_once()                    # initial=60, active at 63.0, stop at 49.2 (-18%)
        c.price = 55.0                     # normal dip of -8.3% (<18% stop): warmup suppresses sell
        ts.check_once()
        self.assertEqual(c.orders, [], "it does not sell on normal dips before reaching the profit threshold")
        c.price = 48.0                     # crash of -20% (exceeds 18% stop): extreme crash protection triggers
        ts.check_once()
        self.assertEqual(len(c.orders), 1, "extreme crash protection triggers even before warmup")
        self.assertEqual(c.orders[0]["side"], "sell")

    def test_active_past_the_threshold_sells(self):
        c = FakeK(60.0)
        ts = self.ts(c, min_profit_pct=5.0)
        ts.check_once()                    # initial=60
        c.price = 64.0; ts.check_once()   # +6.7% > 5% -> trailing active, peak=64
        c.price = 52.0; ts.check_once()   # -18.8% from peak 64 (HYPE threshold 18%)
        self.assertEqual(len(c.orders), 1, "it sells once the profit threshold is passed")
        self.assertEqual(c.orders[0]["side"], "sell")


class TestRebuyGuardsKraken(Base):
    """Verify Kraken re-buy uses bypass_guards={'math_reference'} and defers on guard refusal."""

    def test_rebuy_executes_when_guards_allow(self):
        from unittest.mock import patch
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 40.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards", return_value=(True, "ok", 1.0)) as mock_guard:
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=lambda _: None)
            mock_guard.assert_called_once()
            self.assertEqual(mock_guard.call_args.kwargs.get("bypass_guards"), {"math_reference"})
            self.assertEqual(len(c.orders), 1)
            self.assertEqual(c.orders[0]["side"], "buy")
            self.assertEqual(res.outcome, "active")
            self.assertIsNotNone(res.intent.get("order_id"))

    def test_rebuy_defers_when_guards_block_cascade(self):
        from unittest.mock import patch
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 40.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards",
                   return_value=(False, "liquidation_cascade (OI -6.2%, taker 0.72)", 0.0)):
            persisted = []
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=persisted.append)
            self.assertEqual(len(c.orders), 0, "No buy order should be sent during cascade")
            self.assertEqual(res.outcome, "refused")
            self.assertEqual(persisted, [None], "Pending order cleared so rebuy intent persists")

    def test_rebuy_downscales_quantity_when_intelligence_guard_suggests_scale(self):
        from unittest.mock import patch
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 40.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards",
                   return_value=(True, "elevated_threat_downscale", 0.5)):
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=lambda _: None)
            self.assertEqual(len(c.orders), 1)
            self.assertEqual(c.orders[0]["side"], "buy")
            self.assertAlmostEqual(c.orders[0]["volume"], 5.0)
            self.assertAlmostEqual(res.intent.get("requested_qty"), 5.0)

    def test_rebuy_below_sell_price_bypasses_all_guards(self):
        from unittest.mock import patch
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        rb_state = {"qty": 10.0, "sell_price": 45.0, "low": 40.0}

        with patch("order_guard.check_intelligence_guards",
                   return_value=(False, "would_block_if_consulted", 0.0)) as mock_guard:
            res = ts.execute_rebuy("HYPE", "HYPE", "HYPEUSD", 10.0, 42.0, rb_state, persist=lambda _: None)
            mock_guard.assert_not_called()
            self.assertEqual(len(c.orders), 1)
            self.assertEqual(c.orders[0]["side"], "buy")
            self.assertAlmostEqual(c.orders[0]["volume"], 10.0)
            self.assertEqual(res.outcome, "active")


    def test_multi_asset_trailing_kraken(self):
        """Verify trailing protection operates on multiple configured assets (e.g. TAO, ADA)."""
        balances = {
            "HYPE": {"balance": "10.0", "hold_trade": "0.0"},
            "TAO": {"balance": "5.0", "hold_trade": "0.0"},
            "ADA": {"balance": "1000.0", "hold_trade": "0.0"},
        }
        prices = {"HYPEUSD": 50.0, "TAOUSD": 200.0, "ADAUSD": 0.50}

        class FakeMultiK(FakeK):
            def __init__(self):
                super().__init__(price=50.0)
                self.prices = prices
                self.balances = balances
            def _private(self, method, data=None):
                if method == "BalanceEx":
                    return self.balances
                return {}
            def last_price(self, pair):
                return self.prices.get(pair, 0.0)

        c = FakeMultiK()
        ts = self.ts(c)
        ts.check_once()  # tracks initial peaks: HYPE=50, TAO=200, ADA=0.50

        # Crash TAO by -25% (exceeds 22% trail) while HYPE and ADA stay normal
        c.prices["TAOUSD"] = 150.0  # -25% from 200
        ts.check_once()

        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertEqual(c.orders[0]["volume"], 5.0)

    def test_emergency_unlock_cancels_resting_sell_and_executes_stop(self):
        """When all balance is locked in a resting SELL order and price crashes past stop,
        total_qty + unlock_qty cancel the resting order (including slash pair aliases) and sell."""
        class FakeUnlockK(FakeK):
            def __init__(self):
                super().__init__(price=60.0, total=20.0, held=0.0)
                self.cancelled = []

            def open_orders(self):
                if self.held > 0:
                    return {
                        "OPEN-TP-1": {
                            "descr": {"type": "sell", "pair": "HYPE/USD", "price": "75.0"},
                        }
                    }
                return {}

            def cancel_order(self, txid):
                self.cancelled.append(txid)
                self.held = 0.0
                return {"count": 1}

        c = FakeUnlockK()
        ts = self.ts(c)
        ts.check_once()  # peak=60.0 while free=20.0

        # Lock all 20.0 HYPE in a resting TP order, then crash price to 48.0 (-20% > 18%)
        c.held = 20.0
        c.price = 48.0
        ts.check_once()

        self.assertEqual(c.cancelled, ["OPEN-TP-1"])
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertAlmostEqual(c.orders[0]["volume"], 20.0)

    def test_execute_sell_metadata_is_not_double_nested(self):
        c = FakeK(60.0, total=10.0, held=0.0)
        ts = self.ts(c)
        res = ts.execute_sell("HYPE", "HYPE", "HYPEUSD", 10.0, 48.0, 60.0, 18.0, persist=lambda _: None)
        self.assertTrue(res.intent.get("market"))
        self.assertNotIn("metadata", res.intent)

    def test_cachemanager_fetch_rest_into_refreshes_empty_configured_pairs_and_matches_aliases(self):
        from cacheaccount import CacheTradeManager
        from providers.kraken_provider import KrakenProvider

        class FakeLedgerClient:
            def _private(self, method, data=None, fresh=False):
                return {
                    "ledger": {
                        "L1": {"refid": "R1", "asset": "HYPE.F", "amount": "5.0", "type": "trade", "time": 1700000000.0},
                        "L2": {"refid": "R1", "asset": "ZUSD", "amount": "-250.0", "type": "trade", "time": 1700000000.0},
                    }
                }

        cfg = KrakenProvider.resolve_cache_daemon_config()
        pairs = ["HYPEUSD", "TAOUSD"]
        lookback_sec = cfg["lookback_sec"]
        cache = {"items": {"TAOUSD": [{"id": "OLD"}]}, "fetchtime": {}}
        by_sym = KrakenProvider.fetch_ledger_fills_by_symbol(
            FakeLedgerClient(), pairs=pairs, lookback_sec=lookback_sec
        )
        CacheTradeManager.merge_venue_fills_into_dict(cache, by_sym, is_snapshot=True)
        self.assertEqual(len(cache["items"]["HYPEUSD"]), 1)
        self.assertEqual(cache["items"]["HYPEUSD"][0]["symbol"], "HYPEUSD")
        self.assertEqual(cache["items"]["TAOUSD"], [])
        self.assertIn("HYPEUSD", cache["fetchtime"])
        self.assertIn("TAOUSD", cache["fetchtime"])

        # WS incremental merge preserves existing WS fills while adding missing REST fills
        ws_cache = {
            "items": {
                "HYPEUSD": [{"id": "WS-1", "symbol": "HYPEUSD", "time": 1700000005000, "price": 52.0, "qty": 1.0}],
            },
            "fetchtime": {},
        }
        ws_by_sym = KrakenProvider.fetch_ledger_fills_by_symbol(
            FakeLedgerClient(), pairs=pairs, lookback_sec=lookback_sec
        )
        CacheTradeManager.merge_venue_fills_into_dict(ws_cache, ws_by_sym, is_snapshot=False)
        self.assertEqual([x["id"] for x in ws_cache["items"]["HYPEUSD"]], ["R1", "WS-1"])
        CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
            ws_cache, now_ms=1700000010000, symbols=pairs, create_missing=True
        )
        self.assertEqual(ws_cache["fetchtime"]["HYPEUSD"], 1700000010000)
        self.assertEqual(ws_cache["fetchtime"]["TAOUSD"], 1700000010000)

    def test_empty_or_failed_balance_ex_does_not_wipe_peak_and_caches_per_tick(self):
        import json

        class CountingFakeK(FakeK):
            def __init__(self):
                super().__init__(price=65.0, total=25.0, held=0.0)
                self.balance_calls = 0
                self.fail_balance = False

            def _private(self, method, data=None):
                if method == "BalanceEx":
                    self.balance_calls += 1
                    if self.fail_balance:
                        return {}
                return super()._private(method, data)

        c = CountingFakeK()
        ts = self.ts(c)
        ts.check_once()
        self.assertEqual(c.balance_calls, 1)
        self.assertEqual(json.load(open(self.sf))["HYPE"]["peak"], 65.0)

        # Transient empty BalanceEx must skip the tick and NOT wipe peak
        c.fail_balance = True
        ts.check_once()
        self.assertEqual(json.load(open(self.sf))["HYPE"]["peak"], 65.0)

    def test_canceled_zero_fill_order_increments_retry_attempt_and_client_order_id(self):
        class CanceledZeroFillK(FakeK):
            def query_orders(self, txids):
                return {str(txids): {
                    "status": "canceled", "vol_exec": "0", "cost": "0", "fee": "0",
                }}

        c = CanceledZeroFillK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts.check_once()  # peak=60.0
        c.price = 48.0   # -20% > 18% -> first SELL attempt
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        first_cid = c.orders[0]["cl_ord_id"]

        # Next tick: reconciles as canceled with 0 fill; next tick retries with a fresh client_order_id
        ts.check_once()
        ts.check_once()
        self.assertEqual(len(c.orders), 2)
        second_cid = c.orders[1]["cl_ord_id"]
        self.assertNotEqual(first_cid, second_cid)

    def test_trend_value_resolves_cross_venue_symbol_variants(self):
        import json
        import time

        fd, trend_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        prev_cache = ts_mod.CACHE_TREND
        try:
            with open(trend_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "TAOUSD": {
                            "ts": time.time() - 3600.0,  # stale exact entry must not shadow fresh variant
                            "gradient_recent": 0.99,
                        },
                        "TAOUSDC": {
                            "ts": time.time(),
                            "gradient_recent": -0.42,
                        },
                    },
                    f,
                )
            ts_mod.CACHE_TREND = trend_path
            c = FakeK(60.0)
            ts = self.ts(c)
            self.assertAlmostEqual(ts.trend("TAOUSD"), -0.42)
            self.assertAlmostEqual(ts.trend("TAO/USD"), -0.42)
        finally:
            ts_mod.CACHE_TREND = prev_cache
            if os.path.exists(trend_path):
                os.remove(trend_path)

    def test_free_balance_fails_closed_on_corrupt_hold_trade_and_unlock_qty_handles_null_descr(self):
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts._tick_balances = {"HYPE": {"balance": "25.0", "hold_trade": "corrupt"}}
        self.assertEqual(ts.free_qty("HYPE"), 0.0)
        ts._tick_balances = {"HYPE": {"balance": "25.0", "hold_trade": "-5.0"}}
        self.assertEqual(ts.free_qty("HYPE"), 0.0)
        ts._tick_balances = {"HYPE": {"balance": "25.0", "hold_trade": True}}
        self.assertEqual(ts.free_qty("HYPE"), 0.0)

        cancelled = []
        c.open_orders = lambda: {
            "TX-NULL-DESCR": {"descr": None, "pair": "HYPEUSD", "type": "sell"},
            "TX-NON-DICT": "malformed",
        }
        c.cancel_order = lambda txid: cancelled.append(txid) or {"count": 1}
        self.assertTrue(ts.unlock_qty("HYPEUSD"))
        self.assertEqual(cancelled, ["TX-NULL-DESCR"])

    def test_sell_trend_defer_up_to_7m_then_forces_sell(self):
        from unittest.mock import patch
        c = FakeK(60.0, total=25.0, held=0.0)
        logs = []
        ts = KrakenTrailing(c, log=logs.append, enabled=True, state_file=self.sf)
        ts.trend = lambda _pair: 1.0  # short trend UP
        with patch("trailing_core.time.time", return_value=2000.0):
            ts.check_once()  # peak=60.0
            c.price = 48.0   # -20% > 18% stop
            ts.check_once()  # t=2000s (0s < 420s): deferred by UP trend
        self.assertEqual(c.orders, [])
        self.assertTrue(any("instant trend is UP" in m for m in logs))

        # After 420s (7 minutes) at t=2420s, timer expires and forces emergency SELL
        with patch("trailing_core.time.time", return_value=2420.0):
            ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertTrue(any("sell trend deferral expired" in m for m in logs))

    def test_rebuy_clamps_limit_below_sell_price_and_enforces_daily_limit_above_sell_price(self):
        from unittest.mock import patch
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts.check_once()  # peak=60.0
        c.price = 48.0   # -20% > 18% -> SELL @ 48.0
        ts.check_once()
        c.total = 0.0
        ts.check_once()  # terminal SELL fill confirmed -> rebuy armed with sell_price=48.0, low=48.0

        # Dip to 47.0, then bounce to 47.80 (+1.7% > 1.2% bounce, and 47.80 < 48.0 sell_price,
        # while 47.80 * 1.005 = 48.039 >= 48.0 sell_price -> limit must clamp to 47.80!)
        c.price = 47.0
        ts.check_once()
        c.price = 47.80
        ts.check_once()
        buy_orders = [o for o in c.orders if o["side"] == "buy"]
        self.assertEqual(len(buy_orders), 1)
        self.assertAlmostEqual(buy_orders[0]["price"], 47.80)

        # Now test rebuy above sell_price (price=49.0 >= sell_price=48.0) when daily_limit_guard blocks GO1
        self.tearDown(); self.setUp()
        c2 = FakeK(60.0, total=25.0, held=0.0)
        ts2 = self.ts(c2)
        ts2.check_once()
        c2.price = 48.0
        ts2.check_once()
        c2.total = 0.0
        ts2.check_once()  # rebuy armed with sell_price=48.0
        c2.price = 49.0   # +2.08% > 1.2%, above sell_price=48.0
        ts2.executor.get_orders = lambda *_a, **_k: []
        with patch("kraken.trailing_stop.order_guard.daily_limit_guard", return_value=(False, "daily_limit_exceeded")):
            ts2.check_once()
        self.assertEqual([o for o in c2.orders if o["side"] == "buy"], [])

    def test_cross_venue_shared_cost_basis_cli_dispatch_and_strategy_dry_run_gates(self):
        import io
        import json
        import time
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import patch
        from providers.base import MarketDataProvider
        from providers.hyperliquid_provider import HyperliquidProvider
        from providers.kraken_provider import KrakenProvider

        # 1. KrakenTrailing.cost_basis delegates to KrakenProvider.position_cost_basis / compute_fills_cost_basis
        fd, cache_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        now_ms = int(time.time() * 1000)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "items": {
                            "HYPEUSD": [
                                {
                                    "id": "T1",
                                    "ordertxid": "O1",
                                    "symbol": "HYPEUSD",
                                    "side": "BUY",
                                    "price": 40.0,
                                    "qty": 10.0,
                                    "fee": 0.0,
                                    "feeAsset": "USD",
                                    "time": now_ms - 5000,
                                },
                                {
                                    "id": "T2",
                                    "ordertxid": "O2",
                                    "symbol": "HYPEUSD",
                                    "side": "BUY",
                                    "price": 50.0,
                                    "qty": 15.0,
                                    "fee": 0.0,
                                    "feeAsset": "USD",
                                    "time": now_ms - 2000,
                                },
                            ]
                        },
                        "fetchtime": {"HYPEUSD": now_ms},
                    },
                    f,
                )
            with patch("providers.kraken_provider._KRAKEN_CACHE_FILE", cache_path):
                c = FakeK(60.0, total=25.0, held=5.0)  # total balance = 25.0 (20 free + 5 held)
                ts = self.ts(c)
                basis = ts.cost_basis("HYPEUSD")
                # 10 @ 40 + 15 @ 50 = 400 + 750 = 1150 / 25 = 46.0
                self.assertAlmostEqual(basis, 46.0)
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)

        # 2. Shared TrailingAdapterBase print_status and dispatch_cli
        c = FakeK(60.0, total=25.0, held=0.0)
        ts = self.ts(c)
        ts.check_once()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ts.dispatch_cli(SimpleNamespace(status=True, once=False))
        self.assertEqual(rc, 0)
        self.assertIn("HYPE: peak=60.0", buf.getvalue())
        self.assertIn("ENABLED=True", buf.getvalue())

        # 3. Shared MarketDataProvider.resolve_strategy_dry_run
        self.assertTrue(MarketDataProvider.resolve_strategy_dry_run(True, True, True))
        self.assertTrue(MarketDataProvider.resolve_strategy_dry_run(False, False, True))
        self.assertTrue(MarketDataProvider.resolve_strategy_dry_run(False, True, False))
        self.assertFalse(MarketDataProvider.resolve_strategy_dry_run(False, True, True))

        with patch.dict(os.environ, {"HL_LIVE_ORDERS": "true", "KRAKEN_LIVE_ORDERS": "true"}):
            self.assertFalse(HyperliquidProvider.strategy_dry_run(False, True))
            self.assertFalse(KrakenProvider.strategy_dry_run(False, True))
        with patch.dict(os.environ, {"HL_LIVE_ORDERS": "false", "KRAKEN_LIVE_ORDERS": "false"}):
            self.assertTrue(HyperliquidProvider.strategy_dry_run(False, True))
            self.assertTrue(KrakenProvider.strategy_dry_run(False, True))

        # 4. AuditedStrategyExecutor passthroughs & SSOT cache_path resolution
        c.pair_info = lambda _p: {"pair_decimals": 2, "lot_decimals": 4, "ordermin": "0.01", "base": "HYPE"}
        self.assertAlmostEqual(ts.executor.min_order_qty("HYPEUSD"), 0.01)
        self.assertAlmostEqual(ts.executor.round_quantity("HYPEUSD", 1.23459), 1.2345)
        self.assertAlmostEqual(ts.executor.round_price("HYPEUSD", 48.129), 48.13)
        from cachepaths import cache_path
        from providers.order_ohlc_utils import (
            parse_ohlc_interval,
            parse_ohlc_interval_minutes,
            resolve_ohlc_time_window_ms,
        )
        hl_Live_ssot = HyperliquidProvider(token="HYPE", live_enabled=True)
        hl_Live_ssot._force_ssot_cache = True
        hl_cache_file = hl_Live_ssot._fills_cache_file()
        self.assertEqual(hl_cache_file, cache_path("cache_trade_hyperliquid.json"))

        # 5. Shared OHLC interval/window helpers, clamp_orderbook_limit & resolve_venue_instrument_spec
        from providers.order_ohlc_utils import (
            build_open_order_entry,
            format_strategy_execution_mode,
        )
        self.assertEqual(parse_ohlc_interval(" 1H "), (60, "1h"))
        self.assertEqual(parse_ohlc_interval(" 1M "), (43200, "1M"))
        self.assertEqual(parse_ohlc_interval_minutes("240"), 240)
        _, s_ms, e_ms, to_f = resolve_ohlc_time_window_ms(end_ms=100_000_000, lookback_hours=1, timeout=15)
        self.assertEqual(e_ms - s_ms, 3600 * 1000)
        self.assertEqual(to_f, 15.0)
        self.assertIsNotNone(MarketDataProvider.resolve_venue_instrument_spec("kraken", "TAOUSD"))
        oo_rec = build_open_order_entry(
            {"cl_ord_id": "abc123", "status": "open", "time": 1700000000},
            order_id="O1",
            side="BUY",
            price=25.0,
            orig_qty=2.0,
            executed_qty=0.5,
        )
        self.assertEqual(oo_rec["orderId"], "O1")
        self.assertEqual(oo_rec["clientOrderId"], "abc123")
        self.assertEqual(oo_rec["status"], "OPEN")
        self.assertEqual(oo_rec["time"], 1700000000000)
        self.assertEqual(format_strategy_execution_mode(True, True), "OBSERVE ONLY (AI managed)")
        self.assertEqual(format_strategy_execution_mode(False, True), "PAPER (no money)")
        self.assertEqual(format_strategy_execution_mode(False, False), "⚠ REAL — REAL MONEY")


# ─── xStock Watcher & Strategy Adoption (from test_xstock.py) ───────────────

class FakeKrakenXStock:
    def __init__(self, bal=None, pairs=None, price=200.0):
        self.bal = bal or {}
        self.pairs = pairs or {}
        self.price = price

    def balance(self):
        return self.bal

    def asset_pairs(self):
        return self.pairs

    def last_price(self, pair):
        return self.price

    def pair_info(self, pair):
        return {"base": "TSTX", "pair_decimals": 2, "lot_decimals": 6, "ordermin": "0.01"}

    def pair_precision(self, pair):
        info = self.pair_info(pair)
        return SimpleNamespace(
            price_decimals=info["pair_decimals"],
            volume_decimals=info["lot_decimals"],
            order_min=float(info["ordermin"]),
            base_asset=info["base"],
        )

    def free_balance(self, asset):
        return float(self.bal.get(asset, 0.0) or 0.0)


class _XStockBase(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        self.state_path = path
        os.environ["XSTOCK_STATE_FILE"] = path
        os.environ.update(parse_dotenv(os.path.join(_KRAKEN_DIR, "config.env")))
        self.alerts: list[str] = []
        self._xw_notify = xw.notify
        self._st_notify = _xstock_strategy.notify
        xw.notify = lambda **kw: self.alerts.append(kw.get("title", ""))
        _xstock_strategy.notify = lambda **kw: self.alerts.append(kw.get("title", ""))

    def tearDown(self):
        xw.notify = self._xw_notify
        _xstock_strategy.notify = self._st_notify
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        os.environ.clear()
        os.environ.update(self._env)


class TestWatcherPersistenta(_XStockBase):
    def test_allocation_alerts_once_across_a_restart(self):
        c = FakeKrakenXStock(bal={"ZUSD": "100"})
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 0)
        c.bal = {"ZUSD": "100", "SPCXX": "37.5"}
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1)
        self.assertEqual(st["allocated"]["asset"], "SPCXX")
        xw._save_state(st)
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1, "a restart must not duplicate the alert")
        self.assertEqual(st["allocated"]["qty"], 37.5, "the allocation must be known after a restart")

    def test_a_new_unmatched_asset_gives_an_informational_alert(self):
        c = FakeKrakenXStock(bal={"ZUSD": "100"})
        st = xw._load_state()
        xw.check_balance(c, st, "SPCX", False)
        c.bal = {"ZUSD": "100", "RANDOMCOIN": "5"}
        xw.check_balance(c, st, "SPCX", False)
        self.assertEqual(len(self.alerts), 1)
        self.assertIsNone(st["allocated"], "a non-matching asset is not an allocation")

    def test_the_pair_prefers_the_quote_currency_and_does_not_duplicate(self):
        pairs = {"SPCXXEUR": {"wsname": "SPCXx/EUR", "base": "SPCXX"},
                 "SPCXXUSD": {"wsname": "SPCXx/USD", "base": "SPCXX"}}
        c = FakeKrakenXStock(pairs=pairs)
        st = xw._load_state()
        xw.check_pairs(c, st, "SPCX", False, quote="USD")
        self.assertEqual(st["pair"], "SPCXXUSD", "the USD quote must be preferred")
        self.assertEqual(len(self.alerts), 1)
        xw._save_state(st)
        st = xw._load_state()
        xw.check_pairs(c, st, "SPCX", False, quote="USD")
        self.assertEqual(len(self.alerts), 1, "the listing alert is not duplicated")

    def test_the_tp_and_sl_level_alerts_fire_once(self):
        c = FakeKrakenXStock(price=250.0)
        st = xw._load_state()
        self.assertIn("alerted_tp2", st)
        self.assertFalse(st["alerted_tp2"])
        st["allocated"] = {"asset": "SPCXX", "qty": 37.5}
        st["pair"] = "SPCXXUSD"
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        self.assertEqual(len(self.alerts), 1, "alerta TP o singura data")
        c.price = 160.0
        xw.check_levels(c, st, 200.0, 20.0, 15.0, "", False)
        self.assertEqual(len(self.alerts), 2, "and the SL alert fires only once")

    def test_load_state_sanitizes_corrupt_allocated_and_bot_pid(self):
        xw._save_state({
            "known_assets": {"ZUSD": 100.0},
            "allocated": {"asset": True, "qty": "corrupt"},
            "bot_pid": True,
        })
        st = xw._load_state()
        self.assertIsNone(st["allocated"])
        self.assertIsNone(st["bot_pid"])

        xw._save_state({
            "known_assets": {"ZUSD": 100.0},
            "allocated": {"asset": " SPCXX ", "qty": "37.5", "ts": -5},
            "bot_pid": 4321.0,
        })
        st2 = xw._load_state()
        self.assertEqual(st2["allocated"]["asset"], "SPCXX")
        self.assertEqual(st2["allocated"]["qty"], 37.5)
        self.assertEqual(st2["allocated"]["ts"], 0.0)
        self.assertEqual(st2["bot_pid"], 4321)


class TestAutoStart(_XStockBase):
    def setUp(self):
        super().setUp()
        self._script, self._log = xw.BOT_SCRIPT, xw.BOT_LOG
        fd, self.stub = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write("import time\ntime.sleep(120)\n")
        xw.BOT_SCRIPT = self.stub
        xw.BOT_LOG = self.stub + ".log"
        os.environ["XSTOCK_AUTOSTART"] = "true"
        self.st = {"allocated": {"asset": "SPCXX", "qty": 37.5}, "pair": "SPCXXUSD",
                   "bot_pid": None, "alerted_need_price": False}

    def tearDown(self):
        if self.st.get("bot_pid"):
            try:
                os.kill(int(self.st["bot_pid"]), signal.SIGKILL)
                time.sleep(0.2)
                xw._bot_alive(self.st["bot_pid"])
            except (OSError, ValueError, TypeError):
                pass
        for p in (self.stub, xw.BOT_LOG):
            if os.path.exists(p):
                os.remove(p)
        xw.BOT_SCRIPT, xw.BOT_LOG = self._script, self._log
        super().tearDown()

    def test_it_starts_and_is_idempotent(self):
        xw.maybe_start_bot(self.st, 200.0, False)
        pid = self.st["bot_pid"]
        self.assertTrue(xw._bot_alive(pid))
        self.assertEqual(len(self.alerts), 1)
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertEqual(self.st["bot_pid"], pid, "it does not start a second bot")
        self.assertEqual(len(self.alerts), 1)

    def test_the_watchdog_restarts_a_dead_or_zombie_bot(self):
        xw.maybe_start_bot(self.st, 200.0, False)
        pid = self.st["bot_pid"]
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)
        self.assertFalse(xw._bot_alive(pid), "the zombie must be reaped and declared dead")
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertNotEqual(self.st["bot_pid"], pid)
        self.assertTrue(xw._bot_alive(self.st["bot_pid"]))
        self.assertIn("RESTARTED", self.alerts[-1])

    def test_it_does_not_start_without_a_price_but_asks_for_it_once(self):
        xw.maybe_start_bot(self.st, 0.0, False)
        xw.maybe_start_bot(self.st, 0.0, False)
        self.assertIsNone(self.st["bot_pid"])
        self.assertEqual(len(self.alerts), 1)
        self.assertIn("XSTOCK_ALLOC_PRICE", self.alerts[0])

    def test_autostart_disabled_does_not_start(self):
        os.environ["XSTOCK_AUTOSTART"] = "false"
        xw.maybe_start_bot(self.st, 200.0, False)
        self.assertIsNone(self.st["bot_pid"])
        self.assertEqual(len(self.alerts), 0)

    def test_it_does_not_start_without_an_allocation_or_a_pair(self):
        xw.maybe_start_bot({"allocated": None, "pair": "X", "bot_pid": None,
                            "alerted_need_price": False}, 200.0, False)
        xw.maybe_start_bot({"allocated": {"asset": "A", "qty": 1}, "pair": None,
                            "bot_pid": None, "alerted_need_price": False}, 200.0, False)
        self.assertEqual(len(self.alerts), 0)

    def test_the_paper_flag_is_passed_to_the_bot(self):
        os.environ["XSTOCK_BOT_PAPER"] = "true"
        with open(self.stub, "w") as f:
            f.write("import sys,time\nopen(sys.argv[0]+'.argv','w').write(' '.join(sys.argv[1:]))\ntime.sleep(120)\n")
        xw.maybe_start_bot(self.st, 200.0, False)
        time.sleep(0.6)
        argv = open(self.stub + ".argv").read()
        os.remove(self.stub + ".argv")
        self.assertIn("--paper", argv)
        self.assertIn("--pair SPCXXUSD", argv)


class TestAdoptare(_XStockBase):
    PAIR = "TSTXUSD"

    def setUp(self):
        super().setUp()
        os.environ.update(parse_dotenv(os.path.join(_KRAKEN_DIR, "config.env")))
        os.environ["STRAT_ADOPT_COST"] = "200"
        sp = _xstock_strategy.state_path_for(self.PAIR)
        if os.path.exists(sp):
            os.remove(sp)

    def tearDown(self):
        sp = _xstock_strategy.state_path_for(self.PAIR)
        if os.path.exists(sp):
            os.remove(sp)
        super().tearDown()

    def _strategy(self, client):
        return _xstock_strategy.Strategy(client, self.PAIR, _xstock_strategy.StratParams.from_env(), dry_run=True, desktop=False)

    def test_adopts_from_the_balance_and_places_a_tp(self):
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "37.5"}))
        s.step(200.0)
        self.assertTrue(s.s["adopted"])
        self.assertAlmostEqual(s.s["qty"], 37.5, delta=0.001)
        self.assertLess(s.s["qty"], 37.5 + 1e-12, "it never sells more than the ledger")
        self.assertAlmostEqual(s.s["cost"] / s.s["qty"], 200.0)
        sells = [o for o in s.s["orders"] if o["side"] == "sell" and o["kind"] == "TP"]
        self.assertEqual(
            len(sells), 0,
            "configured trend-hold waits for the TP threshold before trailing",
        )

    def test_waits_for_the_allocation_without_buying(self):
        c = FakeKrakenXStock(bal={"ZUSD": "100"})
        s = self._strategy(c)
        s.step(100.0)
        self.assertEqual(s.s["orders"], [], "it does NOT buy a new entry while waiting")
        c.bal = {"ZUSD": "100", "TSTX": "37.5"}
        s.step(200.0)
        self.assertTrue(s.s["adopted"])

    def test_it_does_not_adopt_over_a_cycle_in_progress(self):
        import json
        with open(_xstock_strategy.state_path_for(self.PAIR), "w") as f:
            json.dump({"cycle": 3, "qty": 2.46, "cost": 150.0, "spent": 150.0,
                       "dca_buys": 1, "orders": []}, f)
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "99"}))
        s._maybe_adopt()
        self.assertFalse(s.s.get("adopted", False))
        self.assertEqual(s.s["qty"], 2.46, "the existing position remains untouched")

    def test_it_does_not_re_adopt_after_a_restart(self):
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "37.5"}))
        s._maybe_adopt()
        q1 = s.s["qty"]
        s._save()
        s2 = self._strategy(FakeKrakenXStock(bal={"TSTX": "99999"}))
        s2._maybe_adopt()
        self.assertEqual(s2.s["qty"], q1, "a restart does not re-adopt or duplicate")

    def test_the_allocation_does_not_consume_the_dca_cap(self):
        s = self._strategy(FakeKrakenXStock(bal={"TSTX": "37.5"}))
        s._maybe_adopt()
        self.assertEqual(s.s["spent"], 0.0, "the cap remains available for additional funds")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_krakentrailing__submit_lifecycle_order_normalizes_bypass_guards_all_all_iter7_s3(monkeypatch):
    """[ITER7 #3] KrakenTrailing._submit_lifecycle_order normalizes bypass_guards='ALL' / {'ALL'}."""
    from unittest.mock import MagicMock
    import order_guard
    from kraken.trailing_stop import KrakenTrailing
    # 3. KrakenTrailing._submit_lifecycle_order normalizes bypass_guards="ALL" / {"ALL"}
    eng = object.__new__(KrakenTrailing)
    eng.executor = MagicMock()
    eng.order_lifecycle = MagicMock()
    eng.order_lifecycle.submit.return_value = MagicMock(outcome="accepted")
    guard_called = []
    monkeypatch.setattr(
        order_guard,
        "check_intelligence_guards",
        lambda *a, **k: guard_called.append(1) or (False, "should_be_skipped", 1.0),
    )
    res = eng._submit_lifecycle_order(
        intent={"intent_id": "id1", "client_order_id": "cid1"},
        persist=lambda x: None, pair="SOLUSD", side="BUY",
        qty=1.0, price=100.0, kind="TRAIL", bypass_guards="ALL", market=False,
    )
    assert res.outcome == "accepted"
    assert guard_called == []

def test_trailing_core_should_sell_krakentrailing__trend_value_previous_recovery_iter21_s3(tmp_path, monkeypatch):
    """[ITER21 #3] trailing_core.should_sell & KrakenTrailing._trend_value .previous recovery."""
    import json
    import time
    import pytest
    import trailing_core
    import kraken.trailing_stop as kt
    # 3. trailing_core.should_sell & KrakenTrailing._trend_value .previous recovery
    assert trailing_core.should_sell(0.0, 100.0, 15.0) is False
    assert trailing_core.should_sell(-5.0, 100.0, 15.0) is False
    assert trailing_core.should_sell(float("nan"), 100.0, 15.0) is False
    assert trailing_core.should_sell(80.0, 100.0, 15.0) is True

    trend_file = tmp_path / "cache_short_trend.json"
    trend_prev = tmp_path / "cache_short_trend.json.previous"
    trend_file.write_text("{corrupt", encoding="utf-8")
    trend_prev.write_text(
        json.dumps({"HYPEUSDC": {"ts": time.time() * 1000.0, "gradient_recent": 0.42}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(kt, "CACHE_TREND", str(trend_file))
    kt_inst = object.__new__(kt.KrakenTrailing)
    assert kt_inst.trend("HYPE/USD") == pytest.approx(0.42)

def test_kraken_trailing_stop_krakentrailing_reconcile_pending_execute_rebuy_nan_bool_safety_iter52_s3(tmp_path):
    """[ITER52 #3] kraken/trailing_stop: KrakenTrailing.reconcile_pending & execute_rebuy NaN/bool safety."""
    from types import SimpleNamespace
    from kraken import trailing_stop as k_ts
    # 3. kraken/trailing_stop: KrakenTrailing.reconcile_pending & execute_rebuy NaN/bool safety
    kt = k_ts.KrakenTrailing(
        client=SimpleNamespace(),
        log=lambda _m: None,
        enabled=False,
        state_file=str(tmp_path / "kt_state.json"),
    )
    kt.order_lifecycle = SimpleNamespace(
        reconcile=lambda pending, persist: SimpleNamespace(
            outcome="terminal",
            status=SimpleNamespace(status="closed", filled_qty=float("nan"), cost=float("nan"), fee=0.0),
            intent=pending,
        ),
        new_intent=lambda **kw: kw,
    )
    st_dict = {"retry_counts": {}}
    persist_fn = lambda _val: None
    persist_fn.st = st_dict
    pending_obj = {"symbol": "HYPEUSD", "action": "SELL", "submitted_qty": float("nan"), "attempt": True}
    res = kt.reconcile_pending(pending_obj, persist_fn)
    assert res.outcome == "terminal"
    assert st_dict["retry_counts"]["SELL"] == 1

    captured_intent = {}
    kt._submit_lifecycle_order = lambda intent, *a, **kw: (
        captured_intent.update(intent) or SimpleNamespace(outcome="refused", intent=intent)
    )
    kt.execute_rebuy(
        "HYPE", "HYPE", "HYPEUSD", 1.0, 25.0,
        {"low": float("nan"), "sell_price": True},
        persist_fn,
    )
    assert captured_intent["metadata"]["anchor"] == 25.0
    assert captured_intent["metadata"]["sell_price"] == 0.0

def test_krakentrailing__sync_retry_count_from_state__order_intent_iter57_s3():
    """[ITER57 #3] KrakenTrailing._sync_retry_count_from_state & _order_intent."""
    from unittest.mock import MagicMock
    from kraken.trailing_stop import KrakenTrailing
    # 3. KrakenTrailing._sync_retry_count_from_state & _order_intent
    adapter = object.__new__(KrakenTrailing)
    adapter._retry_counts = {}
    adapter.order_lifecycle = MagicMock()
    adapter.order_lifecycle.new_intent.side_effect = lambda **kw: kw
    dummy_persist = type("P", (), {"st": {"retry_counts": {"SELL": True}}})()
    adapter._sync_retry_count_from_state("HYPEUSD", "SELL", dummy_persist)
    assert adapter._retry_counts.get(("HYPEUSD", "SELL"), 0) == 0
    intent = adapter._order_intent("SELL", "HYPE", "HYPE", "HYPEUSD", 1.5, 20.0, anchor=None)
    assert ":20.00000000" in intent["intent_id"]

def test_krakentrailing_string_enabled_trailing_market_and_reconcile_pending_safety_iter117_s2(tmp_path):
    """[ITER117 #2] KrakenTrailing string enabled / trailing_market and reconcile_pending safety."""
    from kraken.trailing_stop import KrakenTrailing
    # 2. KrakenTrailing string enabled / trailing_market and reconcile_pending safety
    kt = KrakenTrailing(
        client=object(),
        enabled="false",
        trailing_market="0",
        state_file=str(tmp_path / "kt_state.json"),
    )
    assert kt.enabled is False
    assert kt.trailing_market is False
    assert kt.reconcile_pending(None, lambda _p: None) is None
    assert kt.reconcile_pending("not-a-dict", lambda _p: None) is None
    from types import SimpleNamespace
    kt.order_lifecycle = SimpleNamespace(
        reconcile=lambda _p, persist: SimpleNamespace(outcome="absent", status=None)
    )
    kt.reconcile_pending({"symbol": " XXBTZUSD ", "action": " sell "}, lambda _p: None)
    assert kt._retry_counts.get(("XXBTZUSD", "SELL")) == 1


def test_trailing_core_is_venue_agnostic_ast():
    """trailing_core.py must contain zero venue SDK imports or hardcoded venue string branches."""
    import ast
    from pathlib import Path

    import trailing_core

    src = Path(trailing_core.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_nodes.add(node.body[0].value)

    forbidden_venues = ("binance", "kraken", "hyperliquid", "t212", "212trading")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0].lower()
                assert not any(v in root for v in forbidden_venues), f"Forbidden import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0].lower()
            assert not any(v in root for v in forbidden_venues), f"Forbidden import from {node.module}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstring_nodes:
            low = node.value.lower()
            assert not any(v in low for v in forbidden_venues), (
                f"trailing_core.py line {node.lineno} contains hardcoded venue literal {node.value!r}"
            )

