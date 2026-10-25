#!/usr/bin/env python3
"""Tests for the Kraken trailing_stop (no real API, no money)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

_KRAKEN_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_KRAKEN_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _KRAKEN_DIR not in sys.path:
    sys.path.append(_KRAKEN_DIR)
import kraken.trailing_stop as ts_mod
from kraken.trailing_stop import KrakenTrailing, should_sell


class FakeK:
    def __init__(self, price, total=25.0, held=0.0):
        self.price = price; self.total = total; self.held = held; self.orders = []
    def _private(self, method, data=None):
        if method == "BalanceEx":
            return {"HYPE": {"balance": str(self.total), "hold_trade": str(self.held)}}
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
    def tearDown(self):
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
        c.prices["TAOUSD"] = 150.0  # -25% de la 200
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
        import kraken_cachemanager as kcm

        class FakeLedgerClient:
            def _private(self, method, data=None, fresh=False):
                return {
                    "ledger": {
                        "L1": {"refid": "R1", "asset": "HYPE.F", "amount": "5.0", "type": "trade", "time": 1700000000.0},
                        "L2": {"refid": "R1", "asset": "ZUSD", "amount": "-250.0", "type": "trade", "time": 1700000000.0},
                    }
                }

        cache = {"items": {"TAOUSD": [{"id": "OLD"}]}, "fetchtime": {}}
        saved_pairs = list(kcm.PAIRS)
        try:
            kcm.PAIRS[:] = ["HYPEUSD", "TAOUSD"]
            kcm._fetch_rest_into(FakeLedgerClient(), cache)
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
            kcm._merge_rest_into(FakeLedgerClient(), ws_cache)
            self.assertEqual([x["id"] for x in ws_cache["items"]["HYPEUSD"]], ["R1", "WS-1"])
            kcm._touch_ws_fetchtimes(ws_cache, 1700000010000)
            self.assertEqual(ws_cache["fetchtime"]["HYPEUSD"], 1700000010000)
            self.assertEqual(ws_cache["fetchtime"]["TAOUSD"], 1700000010000)
        finally:
            kcm.PAIRS[:] = saved_pairs

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

