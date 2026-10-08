#!/usr/bin/env python3
"""Tests for the Kraken trailing_stop (no real API, no money)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trailing_stop as ts_mod
from trailing_stop import KrakenTrailing, should_sell


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
        return {str(txids): {
            "status": "closed", "vol_exec": str(order["volume"]),
            "cost": str(order["volume"] * order["price"]), "fee": "0",
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


class TestLogica(unittest.TestCase):
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
        c = FakeK(60.0, total=25.0, held=3.38)        # 21.62 liber
        ts = self.ts(c)
        ts.check_once()                                # varf 60
        c.price = 49.0                                 # -18.3% (prag HYPE 18%)
        ts.check_once()
        self.assertEqual(len(c.orders), 1)
        self.assertEqual(c.orders[0]["side"], "sell")
        self.assertAlmostEqual(c.orders[0]["volume"], 21.62, places=2)   # only the free balance, not 25

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
        c.price = 53.0                                 # -18.5% de la 65
        self.ts(c).check_once()
        self.assertEqual(len(c.orders), 1)

    def test_sub_notional_ignora(self):
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
        ts.check_once()                    # initial=60, activ la 63.0
        c.price = 48.0                     # A 20% crash is below the activation threshold.
        ts.check_once()
        self.assertEqual(c.orders, [], "it does not sell before reaching the profit threshold")

    def test_active_past_the_threshold_sells(self):
        c = FakeK(60.0)
        ts = self.ts(c, min_profit_pct=5.0)
        ts.check_once()                    # initial=60
        c.price = 64.0; ts.check_once()   # +6.7% > 5% -> trailing activ, peak=64
        c.price = 52.0; ts.check_once()   # -18.8% de la peak 64 (prag HYPE 18%)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

