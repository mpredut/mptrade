#!/usr/bin/env python3
"""
Test suite for delta-neutral bot edge cases (server autonomy).
NO real API, NO money: a fake client, captured notifications, state in temp files.

  python test_dn.py -v
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import delta_neutral as dn  # noqa: E402
from delta_neutral import DeltaNeutral, DNParams, state_path_for  # noqa: E402

COIN = "TSTDN"


def params(**over) -> DNParams:
    base = dict(coin=COIN, spot_pair="@999", spot_token=COIN, notional=100.0,
                entry_funding_hr=0.0, exit_funding_hr=-0.00005, funding_window_h=4.0,
                min_hold_h=6.0, rebalance_pct=5.0, check_minutes=5.0, sz_decimals=2,
                liq_alert_pct=20.0, auto_protect=True, reduce_pct=25.0, perp_leverage=1,
                allow_scale_up=False, fee_pct=0.035)
    base.update(over)
    return DNParams(**base)


class FakeClient:
    """Client fals: inregistreaza ordinele, simuleaza erori la cerere."""
    exchange = None

    def __init__(self):
        self.orders: list[tuple] = []
        self.fail_orders = False
        self.raise_account = False
        self.spot_bal = 2.0
        self.perp_szi = -2.0
        self.liq_px = 0.0
        self.free = 100000.0

    def withdrawable(self):
        return self.free

    def spot_balance(self, token):
        return self.free if token == "USDC" else self.spot_bal

    def spot_mid(self, pair): return 50.0
    def mid(self, coin): return 50.0
    def funding_rate(self, coin): return 0.0000125

    def spot_balance_strict(self, token):
        if self.raise_account: raise RuntimeError("API down")
        return self.spot_bal

    def position_strict(self, coin):
        if self.raise_account: raise RuntimeError("API down")
        return self.perp_szi, 50.0

    def position_full(self, coin):
        return {"szi": self.perp_szi, "liquidationPx": self.liq_px}

    def spot_order(self, pair, is_buy, sz, px, szd):
        if self.fail_orders: return False, None, "err: margin"
        self.orders.append(("spot", "buy" if is_buy else "sell", sz))
        return True, 1, "ok"

    def place_limit(self, coin, is_buy, sz, px, reduce_only=False):
        if self.fail_orders: return False, None, "err: margin"
        self.orders.append(("perp", "buy" if is_buy else "sell", sz))
        return True, 2, "ok"

    def set_leverage(self, coin, lev): pass
    def open_orders(self, coin=None): return []
    def cancel(self, coin, oid): return True


def L(spot_qty=2.0, perp_szi=-2.0, funding=0.0000125):
    return {"spot_px": 50.0, "perp_px": 50.0, "funding": funding,
            "spot_qty": spot_qty, "perp_szi": perp_szi}


class Base(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        self.alerts: list[str] = []
        self._notify = dn.notify
        dn.notify = lambda **kw: self.alerts.append(kw.get("title", ""))
        self._cleanup_state()
        self.c = FakeClient()

    def tearDown(self):
        dn.notify = self._notify
        self._cleanup_state()
        os.environ.clear(); os.environ.update(self._env)

    @staticmethod
    def _cleanup_state():
        for suf in ("", ".lock", ".tmp"):
            p = state_path_for(COIN) + suf
            if os.path.exists(p):
                os.remove(p)

    def make(self, dry=False, **over) -> DeltaNeutral:
        d = DeltaNeutral(self.c, params(**over), dry_run=dry, desktop=False)
        return d

    def opened(self, d: DeltaNeutral, target=2.0):
        d.s["status"] = "open"; d.s["target_sz"] = target
        d.s["opened_ts"] = time.time()


# ---------------------------------------------------------------------------
class TestFailedReads(Base):
    def test_an_account_api_error_does_not_guess_and_does_not_trade(self):
        d = self.make()
        self.c.raise_account = True
        self.assertIsNone(d.legs(), "a failed read -> None, not a false 0")
        self.assertEqual(self.c.orders, [])

    def test_a_missing_price_gives_none(self):
        d = self.make()
        self.c.spot_mid = lambda pair: None
        self.assertIsNone(d.legs())


class TestPiciorOrfan(Base):
    def test_a_single_suspicious_tick_does_not_act(self):
        d = self.make(); self.opened(d)
        d.tick(L(spot_qty=2.0, perp_szi=0.0))      # The perp is absent in one reading.
        self.assertEqual(self.c.orders, [], "anti-glitch: it does not act on the first reading")
        self.assertEqual(d.s["orphan_count"], 1)

    def test_a_liquidated_short_closes_the_spot_after_confirmation(self):
        d = self.make(); self.opened(d)
        d.tick(L(spot_qty=2.0, perp_szi=0.0))
        d.tick(L(spot_qty=2.0, perp_szi=0.0))      # confirmat
        sells = [o for o in self.c.orders if o[0] == "spot" and o[1] == "sell"]
        self.assertEqual(len(sells), 1, "sell the remaining spot leg to de-risk")
        self.assertEqual(d.s["status"], "flat")
        self.assertGreater(d.s["cooldown_until"], time.time(), "cooldown anti-thrash")
        self.assertTrue(any("a leg is gone" in a for a in self.alerts))

    def test_a_recovered_glitch_resets_the_counter(self):
        d = self.make(); self.opened(d)
        d.tick(L(spot_qty=2.0, perp_szi=0.0))      # citire gresita o data
        d.tick(L())                                 # revine normal
        self.assertEqual(d.s["orphan_count"], 0)
        self.assertEqual(d.s["status"], "open")

    def test_both_legs_gone_goes_flat_without_orders(self):
        d = self.make(); self.opened(d)
        d.tick(L(spot_qty=0.0, perp_szi=0.0))
        d.tick(L(spot_qty=0.0, perp_szi=0.0))
        self.assertEqual(self.c.orders, [], "there is nothing to close")
        self.assertEqual(d.s["status"], "flat")

    def test_without_auto_protect_it_only_alerts(self):
        d = self.make(auto_protect=False); self.opened(d)
        d.tick(L(spot_qty=2.0, perp_szi=0.0))
        d.tick(L(spot_qty=2.0, perp_szi=0.0))
        self.assertEqual(self.c.orders, [])
        self.assertTrue(any("MANUAL INTERVENTION" in a for a in self.alerts))


class TestDriftSiDust(Base):
    def test_a_large_drift_needs_confirmation_over_2_ticks(self):
        d = self.make(); self.opened(d)
        d.tick(L(spot_qty=0.8, perp_szi=-2.0))     # The spot is at 40% of the target (>50% drift).
        self.assertEqual(self.c.orders, [], "the first tick: it only observes")
        d.tick(L(spot_qty=0.8, perp_szi=-2.0))     # confirmat -> corecteaza
        buys = [o for o in self.c.orders if o[0] == "spot" and o[1] == "buy"]
        self.assertEqual(len(buys), 1)

    def test_a_small_drift_is_corrected_immediately(self):
        d = self.make(); self.opened(d)
        d.tick(L(spot_qty=1.6, perp_szi=-2.0))     # -20% drift: above tolerance, below 50%, above $10
        buys = [o for o in self.c.orders if o[0] == "spot" and o[1] == "buy"]
        self.assertEqual(len(buys), 1, "normal drift is corrected without delay")

    def test_dust_below_the_minimum_sends_no_order(self):
        d = self.make(); self.opened(d, target=2.0)
        d._buy_spot(0.1, 50.0)                     # $5 < $10.5
        self.assertEqual(self.c.orders, [])


class TestFailedOrders(Base):
    def test_alerts_after_3_consecutive_failures(self):
        d = self.make(); self.opened(d)
        self.c.fail_orders = True
        for _ in range(3):
            d._buy_spot(1.0, 50.0)
        self.assertTrue(any("3 consecutive failed orders" in a for a in self.alerts))
        self.c.fail_orders = False
        d._buy_spot(1.0, 50.0)
        self.assertEqual(d.s["order_fails"], 0, "succesul reseteaza contorul")


class TestCooldownSiIntrare(Base):
    def test_the_cooldown_blocks_reopening(self):
        d = self.make()
        d.s["cooldown_until"] = time.time() + 600
        d.tick(L(spot_qty=0.0, perp_szi=0.0, funding=0.001))   # funding excelent
        self.assertEqual(d.s["status"], "flat")
        self.assertEqual(self.c.orders, [])

    def test_after_the_cooldown_it_opens(self):
        d = self.make()
        d.s["cooldown_until"] = time.time() - 1
        d.tick(L(spot_qty=0.0, perp_szi=0.0, funding=0.001))
        self.assertEqual(d.s["status"], "open")
        self.assertEqual(len(self.c.orders), 2, "ambele picioare plasate")


class TestIesireInteligenta(Base):
    def _force_avg(self, d, val):
        d.s["funding_hist"] = [[time.time(), val]] * 5

    def test_negative_funding_held_briefly_does_not_close(self):
        d = self.make(); self.opened(d)
        self._force_avg(d, -0.0002)
        d.tick(L(funding=-0.0002))
        self.assertEqual(d.s["status"], "open", "min_hold is not met yet")

    def test_negative_funding_held_long_enough_closes(self):
        d = self.make(); self.opened(d)
        d.s["opened_ts"] = time.time() - 7 * 3600   # tinut 7h > 6h
        self._force_avg(d, -0.0002)
        d.tick(L(funding=-0.0002))
        self.assertEqual(d.s["status"], "flat")
        self.assertEqual(len(self.c.orders), 2, "it sells the spot and covers the perp")


class TestInfrastructura(Base):
    def test_the_second_instance_is_refused_by_the_lock(self):
        d1 = self.make()
        self.assertTrue(d1._acquire_lock())
        d2 = self.make()
        self.assertFalse(d2._acquire_lock(), "lacatul previne dublarea ordinelor")
        d1._lock_fh.close()
        d3 = self.make()
        self.assertTrue(d3._acquire_lock(), "after the stop the lock is released")
        d3._lock_fh.close()

    def test_the_save_is_atomic_and_valid(self):
        d = self.make(); self.opened(d)
        d._save()
        with open(d.state_file) as f:
            st = json.load(f)
        self.assertEqual(st["status"], "open")
        self.assertFalse(os.path.exists(d.state_file + ".tmp"))

    def test_it_adopts_the_existing_position_on_a_restart(self):
        d = self.make()                              # stare proaspata (flat)
        d.tick(L(spot_qty=1.7, perp_szi=-1.71))
        self.assertEqual(d.s["status"], "open", "it adopts instead of opening a double position")
        self.assertAlmostEqual(d.s["target_sz"], 1.705, places=3)
        self.assertEqual(self.c.orders, [], "adoption places no new orders")


class TestLiquidationProtection(Base):
    def test_auto_protect_reduces_both_legs(self):
        d = self.make(); self.opened(d)
        self.c.liq_px = 55.0                         # A price of 50, liquidation at 55 -> 10% < 20%.
        d.tick(L())
        covers = [o for o in self.c.orders if o[0] == "perp" and o[1] == "buy"]
        sells = [o for o in self.c.orders if o[0] == "spot" and o[1] == "sell"]
        self.assertEqual((len(covers), len(sells)), (1, 1), "reduce ambele picioare")
        self.assertTrue(any("LIQUIDATION" in a or "reduced" in a for a in self.alerts))


class TestScaleUp(Base):
    def test_disabled_does_nothing(self):
        d = self.make(dry=True, allow_scale_up=False, notional=400.0); self.opened(d, target=2.0)
        d._maybe_scale_up(L())
        self.assertEqual(d.s["target_sz"], 2.0)

    def test_it_bumps_the_target_to_the_notional(self):
        d = self.make(dry=True, allow_scale_up=True, notional=400.0); self.opened(d, target=2.0)
        d._maybe_scale_up(L())                 # perp_px 50 -> want 400/50 = 8.0
        self.assertAlmostEqual(d.s["target_sz"], 8.0)

    def test_it_does_not_grow_past_the_target(self):
        d = self.make(dry=True, allow_scale_up=True, notional=400.0); self.opened(d, target=8.0)
        d._maybe_scale_up(L())
        self.assertEqual(d.s["target_sz"], 8.0)

    def test_partial_when_the_collateral_is_small(self):
        d = self.make(dry=False, allow_scale_up=True, notional=400.0); self.opened(d, target=2.0)
        self.c.free = 100.0                    # Only $100 free, spot_px 50 -> +~1.9.
        d._maybe_scale_up(L())
        self.assertGreater(d.s["target_sz"], 2.0)
        self.assertLess(d.s["target_sz"], 8.0)

    def test_a_tick_executes_the_scale_up_and_buys_both_legs(self):
        d = self.make(dry=False, allow_scale_up=True, notional=400.0); self.opened(d, target=2.0)
        d.tick(L()); d.tick(L())               # 2 tick-uri (confirmarea anti-glitch a rebalansului)
        buys_spot = [o for o in self.c.orders if o[0] == "spot" and o[1] == "buy"]
        buys_perp = [o for o in self.c.orders if o[0] == "perp" and o[1] == "sell"]
        self.assertTrue(buys_spot and buys_perp, "the scale-up buys spot and shorts perp (staying neutral)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
