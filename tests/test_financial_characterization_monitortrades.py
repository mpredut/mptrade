"""Characterization tests for monitortrades financial behaviour.

These tests intentionally record the CURRENT decision semantics before the order
pipeline is redesigned.  They use no network, real cache, credentials or exchange
client.  A failure during refactoring means that the financial behaviour changed
and the change must be reviewed explicitly.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import monitortrades as mt
from providers.base import MarketDataProvider


NOW = 2_000_000_000.0
SYMBOL = "CHARUSD"


def _fill(side: str, price: float, qty: float, age_s: float = 4 * 3600) -> dict:
    return {
        "side": side,
        "price": float(price),
        "qty": float(qty),
        "timestamp": (NOW - age_s) * 1000,
    }


class FakeProvider:
    """Only the account-history contract used by get_position_stats."""

    def __init__(self, fills):
        self.fills = list(fills)

    def get_orders(self, symbol, side, since_s):
        return [dict(x) for x in self.fills if x["side"] == side]


class FakeInstrument(mt._Instrument):
    """Instrument-shaped test double that records final financial intents."""

    def __init__(self, *, price, free, fills, params=None, min_qty=0.0,
                 place_result=None):
        # Deliberately do not call Instrument.__init__: no provider registry/API.
        self.symbol = SYMBOL
        self.name = "CHAR"
        self.isolation = "own_ledger"
        self._price = float(price)
        self._free = float(free)
        self._fills = list(fills)
        self._params = dict(params or {})
        self._min_qty = float(min_qty)
        self._provider = FakeProvider(self._fills)
        self.placed = []
        self._place_result = place_result

    def param(self, consumer, key, default=None, cast=None):
        value = self._params.get(f"{consumer}.{key}", default)
        if cast is None:
            return value
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    def orders(self, side, since_s):
        return [dict(x) for x in self._fills if x["side"] == side]

    def price(self):
        return self._price

    def free(self):
        return self._free

    def min_qty(self):
        return self._min_qty

    def place(self, side, price, qty, **kwargs):
        intent = {
            "side": side,
            "price": price,
            "qty": qty,
            "kwargs": dict(kwargs),
        }
        self.placed.append(intent)
        if self._place_result is not None:
            return self._place_result
        return {"orderId": f"fake-{len(self.placed)}"}


class RefusingInstrument(FakeInstrument):
    """Records an attempted placement while simulating a guarded refusal."""

    def place(self, side, price, qty, **kwargs):
        super().place(side, price, qty, **kwargs)
        return None


class UnavailableBalanceApi:
    """Provider account read failed: this is distinct from a real zero balance."""

    @staticmethod
    def free_balance(asset):
        return None


class PipelineProvider(MarketDataProvider):
    """Minimal provider for testing monitortrades through real Instrument guards."""

    def __init__(self, *, price, free, fills):
        self._price = float(price)
        self._free = float(free)
        self._fills = list(fills)
        self.submitted = []

    @property
    def name(self):
        return "PipelineTest"

    def supports_symbol(self, symbol):
        return True

    def get_current_price(self, symbol):
        return self._price

    def free_balance(self, asset):
        return self._free

    def get_orders(self, symbol, side, since_s):
        cutoff = (time.time() - since_s) * 1000
        return [dict(fill) for fill in self._fills
                if fill["timestamp"] >= cutoff
                and (side is None or fill["side"] == side)]

    def policy_cap_quantity(self, symbol, side, price, qty, available_qty, **kwargs):
        return min(float(qty), float(available_qty))

    def place_order(self, symbol, side, price, qty, **kwargs):
        self.submitted.append({
            "symbol": symbol,
            "side": side,
            "price": price,
            "qty": qty,
            "kwargs": dict(kwargs),
        })
        return {"orderId": f"pipeline-{len(self.submitted)}"}


class PipelineApi:
    def __init__(self, provider):
        self.provider = provider

    def provider_by_name(self, name):
        return self.provider if name.lower() == self.provider.name.lower() else None


class MonitorTradesFinancialCharacterization(unittest.TestCase):
    def setUp(self):
        mt._hard_tp_last.clear()
        self._hard_tp_enabled = mt.HARD_TP_ENABLED
        mt.HARD_TP_ENABLED = True

    def tearDown(self):
        mt.HARD_TP_ENABLED = self._hard_tp_enabled
        mt._hard_tp_last.clear()

    def run_tick(self, inst, *, trend_up=False, gain=0.10, loss=0.05):
        with patch.object(mt, "is_trend_up", return_value=trend_up):
            mt.monitor_price_and_trade(
                inst,
                sbs=12 * 24 * 3600 + 60,
                maxage_trade_s=10 * 24 * 3600,
                gain_threshold=gain,
                lost_threshold=loss,
                now_fn=lambda: NOW,
            )

    def test_position_stats_are_gross_averages_and_net_quantity(self):
        provider = FakeProvider([
            _fill("BUY", 100, 2),
            _fill("BUY", 120, 1),
            _fill("SELL", 130, 0.5),
        ])
        got = mt.get_position_stats(SYMBOL, 10 * 24 * 3600, api=provider)
        self.assertEqual(got["buy_qty"], 3.0)
        self.assertEqual(got["sell_qty"], 0.5)
        self.assertEqual(got["net_qty"], 2.5)
        self.assertAlmostEqual(got["average_buy_price"], 320 / 3)
        self.assertEqual(got["average_sell_price"], 130.0)

    def test_hard_take_profit_behaviors(self):
        mt._hard_tp_last.clear()
        with self.subTest(msg="sells_configured_fraction_and_stops_the_tick"):
            inst = FakeInstrument(
                price=118,
                free=10,
                fills=[_fill("BUY", 100, 10)],
                params={"mt.hardtp": 17, "mt.hardtp_fraction": 0.5},
            )
            self.run_tick(inst, trend_up=True)  # hard TP is independent of uptrend
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual(order["side"], "SELL")
            self.assertEqual(order["qty"], 5.0)
            self.assertTrue(order["kwargs"]["force"])
            self.assertNotIn("caller_owns_retry", order["kwargs"])

        mt._hard_tp_last.clear()
        with self.subTest(msg="fractional_hard_tp_below_minimum_falls_through_to_full_normal_sell"):
            inst = FakeInstrument(
                price=118,
                free=1,
                fills=[_fill("BUY", 100, 1)],
                params={"mt.hardtp": 17, "mt.hardtp_fraction": 0.5},
                min_qty=0.6,
            )
            self.run_tick(inst, trend_up=False)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]),
                             ("SELL", 1.0))
            self.assertFalse(inst.placed[0]["kwargs"]["force"])

        mt._hard_tp_last.clear()
        with self.subTest(msg="hard_tp_cooldown_prevents_a_second_attempt_in_same_window"):
            inst = FakeInstrument(
                price=118,
                free=10,
                fills=[_fill("BUY", 100, 10)],
                params={
                    "mt.hardtp": 17,
                    "mt.hardtp_fraction": 0.5,
                    "mt.hardtp_cooldown_h": 6,
                },
            )
            self.run_tick(inst, trend_up=True)
            self.run_tick(inst, trend_up=True)
            self.assertEqual(len(inst.placed), 1)

        mt._hard_tp_last.clear()
        with self.subTest(msg="regression_refused_hard_tp_must_not_start_cooldown"):
            inst = RefusingInstrument(
                price=118,
                free=10,
                fills=[_fill("BUY", 100, 10)],
                params={"mt.hardtp": 17, "mt.hardtp_fraction": 0.5},
            )
            self.run_tick(inst, trend_up=True)
            self.assertNotIn(SYMBOL, mt._hard_tp_last)

        mt._hard_tp_last.clear()
        with self.subTest(msg="regression_hard_tp_fraction_must_not_exceed_free_balance"):
            inst = FakeInstrument(
                price=118,
                free=10,
                fills=[_fill("BUY", 100, 10)],
                params={"mt.hardtp": 17, "mt.hardtp_fraction": 1.5},
            )
            self.run_tick(inst, trend_up=True)
            self.assertEqual(inst.placed, [])

    def test_normal_take_profit_behaviors(self):
        with self.subTest(msg="normal_take_profit_sells_all_free_balance_when_trend_not_up"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(price=112, free=7.25, fills=[_fill("BUY", 100, 8)])
            self.run_tick(inst, trend_up=False, gain=0.10)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["price"], order["qty"]),
                             ("SELL", 112.0, 7.25))
            self.assertFalse(order["kwargs"]["force"])
            self.assertNotIn("caller_owns_retry", order["kwargs"])

        with self.subTest(msg="uptrend_blocks_normal_take_profit"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(price=112, free=7.25, fills=[_fill("BUY", 100, 8)])
            self.run_tick(inst, trend_up=True, gain=0.10)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="loss_exit_bypasses_profit_guard_and_uses_shared_outbox"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(price=93, free=4, fills=[_fill("BUY", 100, 4)])
            self.run_tick(inst, trend_up=False, loss=0.05)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertTrue(order["kwargs"]["bypass_profit_guard"])
            self.assertNotIn("caller_owns_retry", order["kwargs"])

        with self.subTest(msg="average_reference_can_trigger_sell_when_last_buy_does_not"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(
                price=116,
                free=4,
                fills=[
                    _fill("BUY", 120, 1, age_s=4 * 3600),  # latest buy
                    _fill("BUY", 100, 3, age_s=5 * 3600),
                ],
                params={"mt.ref": "average"},
            )
            # Average is 105: +10.47% triggers. Last buy is 120 and would not trigger TP.
            self.run_tick(inst, trend_up=False, gain=0.10, loss=0.20)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "SELL")

        with self.subTest(msg="regression_own_ledger_exit_must_not_sell_unattributed_free_holdings"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(
                price=112,
                free=10,
                fills=[_fill("BUY", 100, 1)],
            )
            self.run_tick(inst, trend_up=False, gain=0.10)
            self.assertLessEqual(inst.placed[0]["qty"], 1.0)

    def test_buyback_behaviors(self):
        with self.subTest(msg="buyback_uses_budget_divided_by_market_price_and_adds_offset"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(
                price=90,
                free=2,
                fills=[_fill("SELL", 100, 2)],
                params={"mt.buy_budget": 225, "mt.max_budget": 1_000},
            )
            self.run_tick(inst, trend_up=True, gain=0.09)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual(order["side"], "BUY")
            self.assertEqual(order["qty"], 2.5)
            self.assertEqual(order["price"], 90 + mt.MT_BUY_PRICE_OFFSET)

        with self.subTest(msg="buyback_is_blocked_when_existing_free_position_reaches_budget"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(
                price=90,
                free=10,
                fills=[_fill("SELL", 100, 2)],
                params={"mt.buy_budget": 225, "mt.max_budget": 900},
            )
            self.run_tick(inst, trend_up=True, gain=0.09)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="regression_buyback_must_include_proposed_buy_in_exposure_cap"):
            inst = FakeInstrument(
                price=90,
                free=9,
                fills=[_fill("SELL", 100, 2)],
                params={"mt.buy_budget": 225, "mt.max_budget": 900},
            )
            # Existing exposure is 810, but the proposed 225 buy would reach 1,035.
            self.run_tick(inst, trend_up=True, gain=0.09)
            self.assertEqual(inst.placed, [])

    def test_guards_and_blocking_behaviors(self):
        with self.subTest(msg="no_recent_fills_produces_no_financial_intent"):
            inst = FakeInstrument(price=100, free=10, fills=[])
            self.run_tick(inst, trend_up=False)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="minimum_quantity_blocks_placement_before_provider_call"):
            inst = FakeInstrument(
                price=118,
                free=1,
                fills=[_fill("BUY", 100, 1)],
                params={"mt.hardtp": 17, "mt.hardtp_fraction": 0.5},
                min_qty=2,
            )
            # Both fractional hard TP and the normal full-balance sell are below min_qty.
            self.run_tick(inst, trend_up=False)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="regression_global_recent_trade_gate_must_block_every_new_order"):
            inst = FakeInstrument(
                price=112,
                free=7,
                fills=[_fill("BUY", 100, 7, age_s=30 * 60)],
            )
            mt.HARD_TP_ENABLED = False
            with (
                patch.object(mt, "MT_RECENT_TRADE_BLOCK_SEC", 5 * 60),
                patch.object(mt, "MT_ALL_TRADES_BLOCK_SEC", 60 * 60),
            ):
                self.run_tick(inst, trend_up=False, gain=0.10)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="regression_non_finite_market_price_must_not_reach_executor"):
            inst = FakeInstrument(
                price=float("inf"),
                free=10,
                fills=[_fill("BUY", 100, 10)],
            )
            self.run_tick(inst, trend_up=False)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="regression_refused_order_must_not_be_reported_as_placed"):
            inst = RefusingInstrument(
                price=118,
                free=10,
                fills=[_fill("BUY", 100, 10)],
            )
            placed = mt._place_guarded(
                inst,
                "SELL",
                118,
                5,
                0,
                force=True,
            )
            self.assertFalse(placed)

    def test_unavailable_balance_remains_distinct_from_real_zero(self):
        self.assertIsNone(
            mt.get_available_qty(SYMBOL, api=UnavailableBalanceApi()),
        )

    def test_regression_loss_exit_must_reach_provider_through_real_guard_pipeline(self):
        """The strategy's loss exit is ineffective if profit guard blocks the SELL."""
        real_now = time.time()
        provider = PipelineProvider(
            price=93,
            free=4,
            fills=[{
                "side": "BUY",
                "price": 100.0,
                "qty": 4.0,
                "timestamp": (real_now - 4 * 3600) * 1000,
            }],
        )
        pipeline_symbol = "CHARPIPEUSD"
        inst = mt._Instrument(
            name="PIPELINE",
            symbol=pipeline_symbol,
            provider=provider.name,
            base="CHAR",
            quote="USD",
            api=PipelineApi(provider),
        )
        mt.HARD_TP_ENABLED = False
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(mt, "is_trend_up", return_value=False),
                patch("order_retry.RETRY_ENABLED", False),
                patch("lock.trade_cooldown.STATE_FILE", os.path.join(temp_dir, "cooldown.json")),
                patch("lock.trade_cooldown.LOCK_FILE", os.path.join(temp_dir, "cooldown.lock")),
                patch("lock.trade_cooldown._cd", None),
            ):
                mt.monitor_price_and_trade(
                    inst,
                    sbs=12 * 24 * 3600,
                    maxage_trade_s=10 * 24 * 3600,
                    gain_threshold=0.10,
                    lost_threshold=0.05,
                    now_fn=lambda: real_now,
                )
        self.assertEqual(len(provider.submitted), 1)
        self.assertEqual(provider.submitted[0]["side"], "SELL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
