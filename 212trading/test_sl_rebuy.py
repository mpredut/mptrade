#!/usr/bin/env python3
"""Exercise catastrophe-stop rebuy through the real execution path.

The network-free stub reproduces the live scenario with ``dry_run=False``:
  1. the catastrophe stop sells the complete position and closes the cycle;
  2. reconciliation sees zero holdings plus ``sl_pending`` and arms the rebuy;
  3. a deeper fall updates the low without buying;
  4. a bounce below ``bounce_pct`` still does not buy;
  5. a sufficient bounce from the low places one BUY ENTRY tranche;
  6. consuming the arm prevents another rebuy.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import strategy as strat_mod  # noqa: E402
from strategy import Strategy, StratParams, state_path_for  # noqa: E402

strat_mod.notify = lambda **kw: None   # No telegram or desktop notification in the test.

TICK = "SLREBUYTEST_US_EQ"


class StubClient:
    """Minimal T212 client: portfolio plus orders controllable from the test."""
    def __init__(self):
        self.portfolio = [{"ticker": TICK, "quantity": 1.0, "averagePrice": 100.0}]
        self.active = []
        self.placed = []
        self._id = 0

    def get_portfolio(self):
        return self.portfolio

    def list_active_orders(self):
        return self.active

    def cancel_order(self, oid):
        self.active = [o for o in self.active if o.get("id") != oid]

    def place_limit_order(self, ticker, qty, limit, validity):
        self._id += 1
        oid = f"O{self._id}"
        o = {"id": oid, "ticker": ticker, "quantity": qty, "limit": limit}
        self.placed.append(o)
        self.active.append(o)
        return 201, {"id": oid}


def main() -> int:
    # clean state
    sf = state_path_for(TICK)
    if os.path.exists(sf):
        os.remove(sf)

    params = StratParams.from_env({
        "STRAT_CURRENCY": "USD",          # 1:1 -> no FX network call.
        "STRATEGY_MODE": "avg_tp",
        "YAHOO_SYMBOL": "SLREBUY",
        "STRAT_ENTRY_PCT": "10",
        "STRAT_DCA_PCT": "10",
        "STRAT_DCA_DROP_PCT": "2",
        "STRAT_TAKEPROFIT_PCT": "3",
        "STRAT_MAX_DCA_BUYS": "3",
        "STRAT_MAX_BUDGET": "1000",
        "STRAT_FX_FEE_PCT": "0.15",
        "STRAT_ENTRY_DISCOUNT_PCT": "0.2",
        "STRAT_CHECK_MINUTES": "5",
        "STRAT_ORDER_TTL_MIN": "10",
        "STRAT_STOP_LOSS_PCT": "30",
        "STRAT_SL_REBUY_ENABLED": "true",
        "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
        "STRAT_REENTRY_DROP_PCT": "1.0",  # profit guard: it would block re-entry above last_sell if sl_rebuy did not take priority
        "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
        "STRAT_LOSS_ALERT_STEP": "1",
        "STRAT_LADDER_MIN_FREE": "6",
        "STRAT_DCA_TREND_GATE_PCT": "0",
        "STRAT_TRAIL_PCT": "0",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
    })
    client = StubClient()
    s = Strategy(client, TICK, params, dry_run=False)

    fails = []

    def check(cond, msg):
        print(("  OK  " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # --- The starting state: we hold 1.0 @ 100, the SL has just been placed (catastrophe) ---
    s.s["qty"] = 1.0
    s.s["cost_usd"] = 100.0
    s.s["entry_price"] = 100.0
    s.s["last_buy_price"] = 100.0
    s.s["spent_cash"] = 100.0
    s.s["sl_pending"] = True            # Marked before the stop executes.

    # 1+2. The SL executed -> the portfolio is 0; reconciliation sees the catastrophe close.
    client.portfolio = [{"ticker": TICK, "quantity": 0.0, "averagePrice": 0.0}]
    client.active = []
    s._reconcile_real(94.0)
    rb = s.s.get("sl_rebuy")
    check(rb is not None, "sl_rebuy ARMED after the stop-loss close")
    check(abs(s.s.get("last_sell_price", 0) - 94.0) < 1e-9, "last_sell_price = the sale price (94)")
    check(s.s["qty"] <= 1e-9, "the position is closed (qty 0) after the SL")
    n_after_arm = len(client.placed)

    # 3. A deeper fall tracks the low and does not buy.
    s.step(92.0)
    check(s.s.get("sl_rebuy") is not None, "still armed during the deep fall (92)")
    check(abs(s.s["sl_rebuy"]["low"] - 92.0) < 1e-9, "minimul urmarit cobora la 92")
    check(len(client.placed) == n_after_arm, "NO buy during the deep fall (it does not catch the knife)")

    # 4. a small pullback (+0.5% from 92) < the 1.2% threshold -> it does not buy yet
    s.step(92.0 * 1.005)
    check(len(client.placed) == n_after_arm, "a pullback of +0.5% < 1.2% -> still no buy")
    check(s.s.get("sl_rebuy") is not None, "still armed after an insufficient pullback")

    # 5. A pullback >= 1.2% from the low of 92 -> it places a BUY ENTRY.
    s.step(92.0 * 1.013)
    check(len(client.placed) == n_after_arm + 1, "a pullback of +1.3% >= 1.2% -> a BUY ENTRY is placed")
    if len(client.placed) == n_after_arm + 1:
        last = client.placed[-1]
        check(last["quantity"] > 0, "the order placed is a BUY (qty>0)")

    # 6. The consumed arm prevents another buy on the next bounce.
    s.step(92.0 * 1.02)
    check(s.s.get("sl_rebuy") is None, "the arming is CONSUMED (1 tranche) after the re-buy")
    check(len(client.placed) == n_after_arm + 1, "no second buy (the arming is consumed)")

    if os.path.exists(sf):
        os.remove(sf)

    print()
    if fails:
        print(f"=== {len(fails)} CHECKS FAILED ===")
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


def test_sl_rebuy_workflow():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
