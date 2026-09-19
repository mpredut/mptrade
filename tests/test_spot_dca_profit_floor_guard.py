"""Intent-aware profit-floor guard in the spot DCA engine.

A new non-STOP limit SELL below known average cost is refused at placement.
MARKET exits bypass this particular guard, not every execution safeguard.
This is a gross-price check, not a net-profit or fill guarantee.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

KRAKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kraken")
ROOT = os.path.dirname(KRAKEN_DIR)
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_dca as strat  # noqa: E402
from providers.strategy_executor import PairPrecision  # noqa: E402


def _make_strategy(**overrides):
    client = MagicMock()
    client.pair_precision.return_value = PairPrecision(2, 6, 0.01, "TEST")
    defaults = dict(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2, dca_amount=50.0,
        dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=1.9, max_budget=1000.0,
        max_dca_buys=10, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=0.0,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2, reentry_tolerance_pct=0.05,
        reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )
    defaults.update(overrides)
    params = strat.StratParams(**defaults)
    s = strat.Strategy(client, "TESTPAIR_GUARD", params, dry_run=True,
                       initial_state=strat._new_state())
    # avg = cost / qty = 1000 / 10 = 100.0
    s.s["cost"] = 1000.0
    s.s["qty"] = 10.0
    return s


class ProfitFloorGuardTest(unittest.TestCase):
    def test_profit_floor_guard_behaviors(self):
        with self.subTest(msg="equal_cost_is_allowed_without_a_fee_profit_guarantee"):
            self.assertTrue(_make_strategy()._place("sell", 1.0, 100.0, kind="TP"))

        with self.subTest(msg="unknown_cost_is_not_a_proven_profitable_exit"):
            s = _make_strategy()
            s.s["qty"] = 0
            self.assertIsNone(s._avg())
            self.assertTrue(s._place("sell", 1.0, 95.0, kind="TP"))

        with self.subTest(msg="stop_exemption_is_by_intent_even_for_limit_orders"):
            self.assertTrue(_make_strategy()._place("sell", 1.0, 90.0, kind="STOP"))

        with self.subTest(msg="rounding_can_turn_a_nominally_profitable_tp_into_equal_cost"):
            s = _make_strategy()
            self.assertTrue(s._place("sell", 1.0, 100.004, kind="TP"))
            self.assertEqual(s.s["orders"][-1]["price"], 100.0)

        with self.subTest(msg="avg_is_the_reference"):
            self.assertEqual(_make_strategy()._avg(), 100.0)

        with self.subTest(msg="limit_sell_below_avg_is_refused"):
            self.assertFalse(_make_strategy()._place("sell", 1.0, 95.0, kind="TP"))

        with self.subTest(msg="limit_sell_above_avg_is_allowed"):
            self.assertTrue(_make_strategy()._place("sell", 1.0, 105.0, kind="TP"))

        with self.subTest(msg="stop_below_avg_is_exempt"):
            # A protective STOP must always run, even at a loss.
            self.assertTrue(_make_strategy()._place("sell", 1.0, 90.0, kind="STOP", market=True))

        with self.subTest(msg="market_trailing_exit_below_avg_is_exempt"):
            # An urgent MARKET trailing exit must not be blocked by the floor.
            self.assertTrue(_make_strategy()._place("sell", 1.0, 90.0, kind="TP", market=True))

        with self.subTest(msg="buy_below_avg_is_unaffected"):
            self.assertTrue(_make_strategy()._place("buy", 1.0, 90.0, kind="DCA"))


class DcaPartialFillTest(unittest.TestCase):
    """DCA spends what is available instead of skipping when underfunded."""

    def _dca_ready(self, free_quote):
        s = _make_strategy(dca_amount=50.0, dca_drop_pct=2.0, stop_loss_pct=0.0,
                           takeprofit_pct=1.9, max_budget=1000.0, max_dca_buys=10)
        s.s["qty"] = 10.0; s.s["cost"] = 1000.0            # avg 100
        s.s["last_buy_price"] = 100.0
        s.s["dca_buys"] = 0; s.s["spent"] = 0.0
        s.client.free_balance.return_value = free_quote
        s.client.ohlc_closes.return_value = []             # regime unavailable -> no brake
        return s

    def _dcas(self, s):
        return [o for o in s.s["orders"] if o.get("kind") == "DCA"]

    def test_dca_budget_and_balance_caps(self):
        with self.subTest(msg="full_dca_when_funded"):
            s = self._dca_ready(500.0)                         # >= 50
            s.step(97.0, timestamp=0.0)                        # dip 97 <= 100*(1-2%)
            self.assertEqual(len(self._dcas(s)), 1)
            self.assertAlmostEqual(self._dcas(s)[-1]["amount"], 50.0)

        with self.subTest(msg="dca_capped_to_available_when_short"):
            s = self._dca_ready(20.0)                          # < 50 -> spend 20, not skip
            s.step(97.0, timestamp=0.0)
            self.assertEqual(len(self._dcas(s)), 1)
            self.assertAlmostEqual(self._dcas(s)[-1]["amount"], 20.0)

        with self.subTest(msg="dca_skipped_only_when_available_is_effectively_zero"):
            s = self._dca_ready(0.0)
            s.step(97.0, timestamp=0.0)
            self.assertEqual(len(self._dcas(s)), 0)            # qty rounds to 0 -> skip

        with self.subTest(msg="unknown_or_invalid_balance_does_not_authorize_a_buy"):
            for balance in (None, MagicMock(), True, -1, float("nan"), float("inf")):
                with self.subTest(balance=balance):
                    s = self._dca_ready(balance)
                    s.step(97.0, timestamp=0.0)
                    self.assertEqual(self._dcas(s), [])

        with self.subTest(msg="balance_outage_does_not_submit_or_crash"):
            s = self._dca_ready(50.0)
            s.client.free_balance.side_effect = RuntimeError("offline")
            s.step(97.0, timestamp=0.0)
            self.assertEqual(self._dcas(s), [])

        with self.subTest(msg="remaining_cycle_budget_can_fund_a_smaller_dca"):
            s = self._dca_ready(500.0)
            s.s["spent"] = 985.0
            s.step(97.0, timestamp=0.0)
            order = self._dcas(s)[0]
            self.assertEqual(order["amount"], 15.0)
            self.assertLessEqual(order["vol"] * order["price"], 15.0)

        with self.subTest(msg="rounding_does_not_exceed_capped_balance"):
            s = self._dca_ready(20.0)
            s.vol_dec = 2
            s.step(97.0, timestamp=0.0)
            order = self._dcas(s)[0]
            self.assertLessEqual(order["vol"] * order["price"], 20.0)

        with self.subTest(msg="exact_nominal_balance_is_still_binding_after_price_rounding"):
            s = self._dca_ready(50.0)
            s.step(97.0, timestamp=0.0)
            order = self._dcas(s)[0]
            self.assertLessEqual(order["vol"] * order["price"], 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
