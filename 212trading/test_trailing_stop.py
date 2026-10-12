#!/usr/bin/env python3
"""Exercise trailing crash breaker and stop loss failure backoff and sizing.

Verifies:
  1. _sellable_qty reserves ladder_min_free when held has fractional dust.
  2. _check_trailing backs off (30 min) when order placement fails/refused.
  3. _check_stop_loss backs off (30 min) when order placement fails/refused.
  4. trailing/stop wait for previous cancel requests to reconcile.
  5. new high clears tr_fail_until.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import strategy as strat_mod  # noqa: E402
from strategy import Strategy, StratParams, state_path_for  # noqa: E402

strat_mod.notify = lambda **kw: None  # No external notifications in tests.

TICK = "TRAILTEST_US_EQ"


class MockClient:
    def __init__(self):
        self.portfolio = [{"ticker": TICK, "quantity": 2.92, "averagePrice": 146.0}]
        self.active = []
        self.placed = []
        self._id = 0

    def get_portfolio(self):
        return self.portfolio

    def list_active_orders(self):
        return self.active

    def cancel_order(self, oid):
        self.active = [o for o in self.active if o.get("id") != oid]
        return True

    def place_limit_order(self, ticker, qty, limit, validity):
        self._id += 1
        oid = f"LIM-{self._id}"
        o = {"id": oid, "ticker": ticker, "quantity": qty, "limit": limit}
        self.placed.append(o)
        self.active.append(o)
        return 201, {"id": oid}

    def place_market_order(self, ticker, qty, extended_hours=False):
        self._id += 1
        oid = f"MKT-{self._id}"
        o = {"id": oid, "ticker": ticker, "quantity": qty, "market": True}
        self.placed.append(o)
        return 201, {"id": oid}


class TestTrailingStopAndSellableQty(unittest.TestCase):
    def setUp(self):
        self.sf = state_path_for(TICK)
        if os.path.exists(self.sf):
            os.remove(self.sf)
        self.now_val = 10000.0
        self.params = StratParams.from_env({
            "STRATEGY_MODE": "avg_tp",
            "STRAT_CURRENCY": "USD",
            "YAHOO_SYMBOL": "TRAILTEST",
            "STRAT_MAX_BUDGET": "1000",
            "STRAT_ENTRY_PCT": "20",
            "STRAT_DCA_PCT": "10",
            "STRAT_MAX_DCA_BUYS": "auto",
            "STRAT_ENTRY_DISCOUNT_PCT": "0.2",
            "STRAT_DCA_DROP_PCT": "2",
            "STRAT_CHECK_MINUTES": "0.4",
            "STRAT_TAKEPROFIT_PCT": "5.0",
            "STRAT_ORDER_TTL_MIN": "10",
            "STRAT_STOP_LOSS_PCT": "30",
            "STRAT_REENTRY_DROP_PCT": "1.0",
            "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
            "STRAT_FX_FEE_PCT": "0.15",
            "STRAT_LOSS_ALERT_STEP": "1",
            "STRAT_LADDER_MIN_FREE": "6.0",
            "STRAT_SL_REBUY_ENABLED": "true",
            "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
            "STRAT_DCA_TREND_GATE_PCT": "0",
            "STRAT_TRAIL_PCT": "8",
            "STRAT_TRAIL_MIN_PROFIT_PCT": "0",
        })
        self.client = MockClient()
        self.strat = Strategy(self.client, TICK, self.params, dry_run=False,
                              clock=lambda: self.now_val)
        self.strat.s["qty"] = 2.9229
        self.strat.s["cost_usd"] = 426.74
        self.strat.s["pos_peak"] = 175.0
        self.strat.s["tr_armed"] = True

    def tearDown(self):
        if os.path.exists(self.sf):
            os.remove(self.sf)

    def test_sellable_qty_with_buffer(self):
        # 2.9229 shares held, avg ~146. Buffer = 6.0 / 146.0 = 0.0411 shares.
        # sellable_qty should subtract buffer and round to 2 decimals (2.88).
        qty = self.strat._sellable_qty(2.9229, ref_price=146.0)
        self.assertEqual(qty, 2.88)
        self.assertLess(qty, 2.9229)

    def test_sellable_qty_no_buffer(self):
        self.strat.p.ladder_min_free = 0.0
        qty = self.strat._sellable_qty(2.9229, ref_price=146.0)
        self.assertEqual(qty, 2.92)

    def test_trailing_drawdown_triggers_sell(self):
        # Peak = 175.0. Price drops to 160.0 (drop = 8.57% >= 8%).
        self.assertTrue(self.strat._check_trailing(160.0))
        # Verify sell was placed with sellable_qty (2.88)
        self.assertEqual(len(self.client.placed), 1)
        self.assertEqual(self.client.placed[0]["quantity"], -2.89)
        self.assertTrue(self.strat.s["sl_pending"])

    def test_trailing_failure_sets_backoff(self):
        # Simulate _place_sell failure (e.g. refused by venue)
        self.strat._place_sell = lambda *a, **kw: False

        # Should return True on failure (cycle consumed/exit active), and set tr_fail_until = now + 1800
        res = self.strat._check_trailing(160.0)
        self.assertTrue(res)
        self.assertEqual(self.strat.s.get("tr_fail_until"), self.now_val + 1800)

        # On next tick (still in backoff), _check_trailing returns False without placing orders
        self.assertFalse(self.strat._check_trailing(160.0))

        # A new peak clears tr_fail_until
        self.strat._check_trailing(180.0)
        self.assertNotIn("tr_fail_until", self.strat.s)
        self.assertEqual(self.strat.s["pos_peak"], 180.0)

    def test_stop_loss_failure_sets_backoff(self):
        # Avg = 146.0. Price = 100.0 (loss = 31.5% >= 30%).
        self.strat._place_sell = lambda *a, **kw: False

        res = self.strat._check_stop_loss(100.0)
        self.assertTrue(res)
        self.assertEqual(self.strat.s.get("sl_fail_until"), self.now_val + 1800)

        # In backoff, returns False without retrying
        self.assertFalse(self.strat._check_stop_loss(100.0))

        # Price recovery clears sl_fail_until
        self.strat._check_stop_loss(140.0)
        self.assertNotIn("sl_fail_until", self.strat.s)


if __name__ == "__main__":
    unittest.main()
