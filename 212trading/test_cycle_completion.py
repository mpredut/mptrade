#!/usr/bin/env python3
"""Exercise cycle completion, broker dust handling, and reentry unblocking.

Verifies:
  1. _is_dust identifies negligible residual broker dust below min holding.
  2. _sellable_qty returns 0.0 when held position is negligible dust.
  3. Live TP ladder completion leaving residual broker dust properly closes cycle N
     and advances to cycle N+1 with flat state and recorded exit price.
  4. Residual broker dust in portfolio is not erroneously adopted into a new position.
  5. Real positions (> dust threshold) are properly adopted.
  6. Paper mode properly closes cycle when TP ladder leaves dust buffer.
  7. Reentry evaluation in step() is enabled after cycle completion with residual dust.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

for _mod in ("strategy", "market_data", "notify"):
    sys.modules.pop(_mod, None)

import strategy as strat_mod  # noqa: E402
from strategy import Strategy, StratParams, state_path_for  # noqa: E402

strat_mod.notify = lambda **kw: None  # No external notifications in tests.

TICK = "CYCLE_TEST_US_EQ"


class MockClient:
    def __init__(self, initial_qty=2.9229, initial_avg=150.0):
        self.portfolio = [{"ticker": TICK, "quantity": initial_qty, "averagePrice": initial_avg}]
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


class TestCycleCompletionAndDust(unittest.TestCase):
    def setUp(self):
        self.sf = state_path_for(TICK)
        if os.path.exists(self.sf):
            os.remove(self.sf)
        self.now_val = 10000.0
        self.params = StratParams.from_env({
            "STRATEGY_MODE": "avg_tp",
            "STRAT_CURRENCY": "USD",
            "YAHOO_SYMBOL": "CYCLETEST",
            "STRAT_MAX_BUDGET": "1000",
            "STRAT_ENTRY_PCT": "20",
            "STRAT_DCA_PCT": "10",
            "STRAT_MAX_DCA_BUYS": "auto",
            "STRAT_ENTRY_DISCOUNT_PCT": "0.2",
            "STRAT_DCA_DROP_PCT": "2",
            "STRAT_CHECK_MINUTES": "0.4",
            "STRAT_TAKEPROFIT_PCT": "11.0",
            "STRAT_TP_LADDER": "11:33,20:33,30:34",
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
            "STRAT_TRAIL_PCT": "0",
            "STRAT_TRAIL_MIN_PROFIT_PCT": "0",
            "STRAT_REENTRY_HYBRID_ENABLED": "false",
            "STRAT_REENTRY_PEAK_RELATIVE": "false",
            "STRAT_REENTRY_PULLBACK_PCT": "1.5",
            "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false",
            "STRAT_REENTRY_PULLBACK_K": "1.0",
            "STRAT_REENTRY_PULLBACK_MIN": "0.8",
            "STRAT_REENTRY_PULLBACK_MAX": "4.0",
            "STRAT_REENTRY_TTL_HOURS": "0.0",
            "STRAT_SL_MHDSL_ENABLED": "false",
            "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
            "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0",
            "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
            "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
            "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
            "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true",
            "STRAT_TP_MDHTP_ENABLED": "false",
            "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
        })
        self.client = MockClient(initial_qty=2.9229, initial_avg=150.0)
        self.strat = Strategy(self.client, TICK, self.params, dry_run=False,
                              clock=lambda: self.now_val)

    def tearDown(self):
        if os.path.exists(self.sf):
            os.remove(self.sf)

    def test_is_dust(self):
        # 0.0 is always dust
        self.assertTrue(self.strat._is_dust(0.0))
        self.assertTrue(self.strat._is_dust(1e-10))

        # 0.0429 shares @ 150 = $6.43 USD (<= max(5.0, 6.0 * 1.5) = $9.0) -> dust
        self.assertTrue(self.strat._is_dust(0.0429, ref_price=150.0))

        # 2.0 shares @ 150 = $300 USD -> NOT dust
        self.assertFalse(self.strat._is_dust(2.0, ref_price=150.0))

        # When ladder_min_free is 0, non-zero qty is never dust
        self.strat.p.ladder_min_free = 0.0
        self.assertFalse(self.strat._is_dust(0.0429, ref_price=150.0))

    def test_sellable_qty_dust_returns_zero(self):
        # 0.0429 shares held is dust -> _sellable_qty returns 0.0
        qty = self.strat._sellable_qty(0.0429, ref_price=150.0)
        self.assertEqual(qty, 0.0)

    def test_live_tp_ladder_cycle_closure_with_dust(self):
        # Set up holding in state
        self.strat.s["qty"] = 2.9229
        self.strat.s["cost_usd"] = 438.435
        self.strat.s["spent_cash"] = 438.44
        self.strat.s["entry_price"] = 150.0
        self.strat.s["tp_sold_levels"] = [11.0, 20.0]  # Levels 11 and 20 already sold

        # Level 30 order is active
        order_30 = {
            "id": "ORD-30",
            "side": "SELL",
            "qty": 2.88,
            "limit": 195.0,
            "level": 30.0,
            "kind": "TP",
        }
        self.strat.s["orders"] = [order_30]

        # Simulate level 30 executing: portfolio quantity drops to 0.0429 (dust buffer)
        self.client.portfolio = [{"ticker": TICK, "quantity": 0.0429, "averagePrice": 150.0}]
        self.client.active = []  # Order ORD-30 is no longer active

        # Tracked status mock
        mock_status = MagicMock()
        mock_status.status = "closed"
        mock_status.filled_qty = 2.88
        mock_status.cost = 2.88 * 195.0
        mock_status.fee = 0.5
        self.strat._tracked_order_status = lambda o, c: mock_status

        # Reconcile real
        self.strat._reconcile_real(195.0)

        # Assert cycle 1 closed and cycle 2 started!
        self.assertEqual(self.strat.s["cycle"], 2)
        self.assertEqual(self.strat.s["qty"], 0.0)
        self.assertAlmostEqual(self.strat.s["last_sell_price"], 195.0, places=2)
        self.assertGreater(self.strat.s["realized_pnl_usd"], 0.0)

        # On the next tick, portfolio still has 0.0429 dust. It must NOT be adopted!
        self.strat._reconcile_real(195.0)
        self.assertEqual(self.strat.s["qty"], 0.0)
        self.assertEqual(len(self.strat.s["orders"]), 0)

        # Step method evaluates flat position and allows reentry
        placed_buys = []
        self.strat._place_buy = lambda amt, lim, kind: placed_buys.append((amt, lim, kind))
        # Ensure reentry criteria are met (price dropped below last_sell_price - drop_pct)
        # Drop by 2% from 195.0 = 191.1
        self.strat.step(190.0)
        self.assertEqual(len(placed_buys), 1)
        self.assertEqual(placed_buys[0][2], "ENTRY")

    def test_real_position_adoption_not_blocked(self):
        # State starts flat
        self.strat.s["qty"] = 0.0
        self.strat.s["orders"] = []

        # Broker has a real position (5.0 shares @ $150 = $750 USD)
        self.client.portfolio = [{"ticker": TICK, "quantity": 5.0, "averagePrice": 150.0}]
        self.strat._tracked_order_status = lambda o, c: None

        self.strat._reconcile_real(150.0)

        # The real position MUST be adopted!
        self.assertEqual(self.strat.s["qty"], 5.0)
        self.assertEqual(self.strat.s["entry_price"], 150.0)

    def test_paper_mode_cycle_closure_with_ladder_dust(self):
        # Create paper strategy
        paper_strat = Strategy(self.client, TICK, self.params, dry_run=True,
                               clock=lambda: self.now_val)
        paper_strat.s["qty"] = 2.9229
        paper_strat.s["cost_usd"] = 438.435
        paper_strat.s["tp_sold_levels"] = [11.0, 20.0, 30.0]

        # Apply fill that sells 2.88, leaving 0.0429 dust
        paper_order = {"side": "SELL", "qty": 2.88, "kind": "TP", "level": 30.0}
        paper_strat._apply_fill(paper_order, 2.88, 195.0)

        # In paper mode, cycle should advance to cycle 2 and qty should be 0.0
        self.assertEqual(paper_strat.s["cycle"], 2)
        self.assertEqual(paper_strat.s["qty"], 0.0)
        self.assertEqual(paper_strat.s["last_sell_price"], 195.0)

    def test_absent_order_released_after_confirmation_passes(self):
        # Order is tracked, but absent from active orders, and status is confirmed absent ("not_found")
        order = {"id": "GHOST-1", "side": "BUY", "qty": 0.03, "limit": 230.0, "ts": 1000.0}
        self.strat.s["orders"] = [order]
        self.client.active = []  # Not in active orders
        self.strat._tracked_order_status = lambda o, c: "not_found"  # Confirmed absent from venue

        # Pass 1: misses becomes 1, order kept
        self.strat._reconcile_real(230.0)
        self.assertEqual(len(self.strat.s["orders"]), 1)
        self.assertEqual(order.get("lookup_misses"), 1)

        # Pass 2: misses becomes 2, order kept
        self.strat._reconcile_real(230.0)
        self.assertEqual(len(self.strat.s["orders"]), 1)
        self.assertEqual(order.get("lookup_misses"), 2)

        # Pass 3: misses becomes 3, order released from tracking!
        self.strat._reconcile_real(230.0)
        self.assertEqual(len(self.strat.s["orders"]), 0)

    def test_transient_status_error_keeps_order_tracked(self):
        # Order is tracked, absent from active list, but status lookup fails (returns None)
        order = {"id": "TRANSIENT-1", "side": "BUY", "qty": 0.03, "limit": 230.0, "ts": 1000.0}
        self.strat.s["orders"] = [order]
        self.client.active = []
        self.strat._tracked_order_status = lambda o, c: None  # Transient error / rate limit

        # Passes 1..5: order must remain tracked and lookup_misses must NOT increment
        for _ in range(5):
            self.strat._reconcile_real(230.0)
            self.assertEqual(len(self.strat.s["orders"]), 1)
            self.assertEqual(order.get("lookup_misses", 0), 0)

    def test_absent_cancel_requested_released_after_two_passes(self):
        order = {"id": "CANCEL-1", "side": "SELL", "qty": 0.08, "limit": 250.0,
                 "cancel_requested": True, "cancel_ts": 1000.0, "ts": 900.0}
        self.strat.s["orders"] = [order]
        self.client.active = []
        self.strat._tracked_order_status = lambda o, c: "not_found"

        # Pass 1: misses = 1
        self.strat._reconcile_real(250.0)
        self.assertEqual(len(self.strat.s["orders"]), 1)

        # Pass 2: cancel_requested releases after 2 misses
        self.strat._reconcile_real(250.0)
        self.assertEqual(len(self.strat.s["orders"]), 0)

    def test_strategy_run_graceful_stop(self):
        import threading
        stop_event = threading.Event()
        stop_event.set()  # Already stopped
        # Strategy.run should exit immediately without hanging
        self.strat.run(stop=stop_event)


if __name__ == "__main__":
    unittest.main()
