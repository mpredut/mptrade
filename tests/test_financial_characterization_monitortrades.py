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
                 place_result=None, open_orders=None):
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
        self._open_orders = list(open_orders or [])
        self.cancelled = []

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

    def open_orders(self):
        return [dict(x) for x in self._open_orders]

    def cancel_order(self, order_id):
        self.cancelled.append(str(order_id))
        self._open_orders = [o for o in self._open_orders if str(o.get("orderId")) != str(order_id)]
        # Unlocking simulation: when TP order is cancelled, restore free balance from holdings
        if self._free == 0.0 and self._fills:
            buy_qty = sum(float(x.get("qty", 0.0)) for x in self._fills if x.get("side") == "BUY")
            self._free = buy_qty
        return None

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
        mt._SL_BOUNCE_STATE.clear()
        mt._TP_RIDE_STATE.clear()
        mt.clear_position_stats_cache()
        self._hard_tp_enabled = mt.HARD_TP_ENABLED
        mt.HARD_TP_ENABLED = True

    def tearDown(self):
        mt.HARD_TP_ENABLED = self._hard_tp_enabled
        mt._hard_tp_last.clear()
        mt._SL_BOUNCE_STATE.clear()
        mt._TP_RIDE_STATE.clear()
        mt.clear_position_stats_cache()

    def run_tick(self, inst, *, trend_up=False, gain=0.10, loss=0.05, now=None):
        ts = now if now is not None else NOW
        with (
            patch.object(mt, "is_trend_up", return_value=trend_up),
            patch.object(mt, "_apply_post_stop_ai_guard", side_effect=lambda sym, px, spx, st, snap, bb, bq, q, **_kwargs: (True, bb, bq, q)),
        ):
            mt.monitor_price_and_trade(
                inst,
                sbs=12 * 24 * 3600 + 60,
                maxage_trade_s=10 * 24 * 3600,
                gain_threshold=gain,
                lost_threshold=loss,
                now_fn=lambda: ts,
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

        with self.subTest(msg="regression_early_sell_of_pre_window_holdings_does_not_cancel_later_buys"):
            provider_aged = FakeProvider([
                _fill("SELL", 110, 5.0, age_s=8 * 24 * 3600),
                _fill("BUY", 100, 2.0, age_s=3 * 24 * 3600),
                _fill("SELL", 105, 0.5, age_s=2 * 24 * 3600),
                _fill("BUY", 95, 1.0, age_s=1 * 24 * 3600),
            ])
            got_aged = mt.get_position_stats(SYMBOL, 10 * 24 * 3600, api=provider_aged)
            self.assertEqual(got_aged["buy_qty"], 3.0)
            self.assertEqual(got_aged["sell_qty"], 5.5)
            self.assertAlmostEqual(got_aged["net_qty"], 2.5)

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
            mt._TP_RIDE_STATE.clear()
            inst = FakeInstrument(price=93, free=4, fills=[_fill("BUY", 100, 4)])
            self.run_tick(inst, trend_up=False, loss=0.05)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertEqual(order["kwargs"]["bypass_guards"], {"all"})
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

        with self.subTest(msg="regression_own_ledger_stop_loss_not_blocked_by_earlier_sell_of_aged_out_buys"):
            mt.HARD_TP_ENABLED = False
            inst = FakeInstrument(
                price=93,
                free=0.06625,
                fills=[
                    _fill("BUY", 100, 0.02205, age_s=1 * 3600),
                    _fill("BUY", 100, 0.04420, age_s=3 * 3600),
                    _fill("SELL", 105, 0.02462, age_s=4 * 24 * 3600),
                    _fill("BUY", 102, 0.02462, age_s=5 * 24 * 3600),
                    _fill("SELL", 110, 0.09205, age_s=8 * 24 * 3600),
                ],
                min_qty=0.00001,
            )
            self.run_tick(inst, trend_up=False, loss=0.05)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "SELL")
            self.assertAlmostEqual(inst.placed[0]["qty"], 0.06625, places=6)
            self.assertEqual(inst.placed[0]["kwargs"]["bypass_guards"], {"all"})

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

        with self.subTest(msg="post_stop_bounce_recovery_triggers_reentry"):
            mt._REENTRY_STATE.clear()
            mt._LAST_EXIT_KIND.clear()
            mt._LAST_EXIT_KIND["CHARUSD"] = "STOP"
            inst = FakeInstrument(
                price=90,
                free=0,
                fills=[_fill("SELL", 100, 2)],
                params={"mt.buy_budget": 225, "mt.max_budget": 1_000},
            )
            # Tick 1 at 90: establishes trough, bounce is 0% -> waiting
            self.run_tick(inst, trend_up=True)
            self.assertEqual(inst.placed, [])

            # Tick 2 at 92.5: bounce is (92.5-90)/90 = 2.78% >= 1.5% -> enters
            inst._price = 92.5
            self.run_tick(inst, trend_up=True)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "BUY")

        with self.subTest(msg="post_stop_bounce_blocked_if_trend_down"):
            mt._REENTRY_STATE.clear()
            mt._LAST_EXIT_KIND.clear()
            mt._LAST_EXIT_KIND["CHARUSD"] = "STOP"
            inst = FakeInstrument(
                price=90,
                free=0,
                fills=[_fill("SELL", 100, 2)],
                params={"mt.buy_budget": 225, "mt.max_budget": 1_000},
            )
            self.run_tick(inst, trend_up=False)
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="post_stop_bounce_state_preserved_if_budget_cap_exceeded"):
            mt._REENTRY_STATE.clear()
            mt._LAST_EXIT_KIND.clear()
            mt._LAST_EXIT_KIND["CHARUSD"] = "STOP"
            inst = FakeInstrument(
                price=90,
                free=0,
                fills=[_fill("SELL", 100, 2)],
                params={"mt.buy_budget": 225, "mt.max_budget": 100},
            )
            # Tick 1 at 90: establishes trough
            self.run_tick(inst, trend_up=True)
            self.assertEqual(inst.placed, [])
            self.assertIn("CHARUSD", mt._REENTRY_STATE)
            self.assertEqual(mt._REENTRY_STATE["CHARUSD"]["trough"], 90)

            # Tick 2 at 92.5: bounce is 2.78% >= 1.5% -> would enter, but blocked by max_budget
            inst._price = 92.5
            self.run_tick(inst, trend_up=True)
            self.assertEqual(inst.placed, [])
            # Crucial: _REENTRY_STATE must NOT be wiped out!
            self.assertIn("CHARUSD", mt._REENTRY_STATE)
            self.assertEqual(mt._REENTRY_STATE["CHARUSD"]["trough"], 90)

            # Tick 3 at 92.5: budget cap lifted -> places buy using original trough (90), not resetting to 92.5
            inst._params["mt.max_budget"] = 1_000
            self.run_tick(inst, trend_up=True)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "BUY")
            self.assertNotIn("CHARUSD", mt._REENTRY_STATE)

    def test_tp_ride_state_preserved_when_sell_placement_fails(self):
        mt.HARD_TP_ENABLED = False
        mt._TP_RIDE_STATE.clear()
        inst = FakeInstrument(
            price=110,
            free=10,
            fills=[_fill("BUY", 100, 10)],
            params={"mt.gain": 7},
        )
        t0 = NOW
        # Tick 1 at t0: initiates trend ride watch
        self.run_tick(inst, trend_up=True, now=t0)
        self.assertEqual(inst.placed, [])
        self.assertIn("CHARUSD", mt._TP_RIDE_STATE)
        self.assertEqual(mt._TP_RIDE_STATE["CHARUSD"]["last_peak_time"], t0)

        # Tick 2 at t0 + 1085s: stagnation timeout triggers (1085s >= 1080s)
        # Simulate placement failure
        with patch.object(mt, "_place_guarded", return_value=False):
            self.run_tick(inst, trend_up=True, now=t0 + 1085)

        # State must be preserved, not wiped!
        self.assertIn("CHARUSD", mt._TP_RIDE_STATE)
        self.assertEqual(mt._TP_RIDE_STATE["CHARUSD"]["last_peak_time"], t0)

        # Tick 3 at t0 + 1090s: placement succeeds, so state is popped
        self.run_tick(inst, trend_up=True, now=t0 + 1090)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "SELL")
        self.assertNotIn("CHARUSD", mt._TP_RIDE_STATE)

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

    def test_mhdsl_multi_horizon_stop_loss_tiers(self):
        """Verify MHDSL decision tree:
        1. Normal dip defers if trend_up is True (micro-trend bounce protection).
        2. Disaster floor breach executes unconditionally even if trend_up is True (emergency circuit breaker).
        3. Disaster floor breach executes unconditionally when trend_up is False.
        4. Active crash cascade defers if trend_up is True (bounce protection respected).
        5. Active crash cascade executes unconditionally when trend_up is False.
        """
        mt.HARD_TP_ENABLED = False

        # Subtest 1: Normal dip (3% loss vs 2.5% threshold, disaster floor = 3.25%), trend_up=True -> DEFER
        with self.subTest(msg="normal_dip_deferred_when_trend_up"):
            inst = FakeInstrument(price=97.0, free=4, fills=[_fill("BUY", 100, 4)])
            self.run_tick(inst, trend_up=True, loss=0.025)
            self.assertEqual(inst.placed, [])

        # Subtest 2: Disaster floor breach (4.0% loss >= 2.5% + 0.75% = 3.25%), trend_up=True -> EXECUTE (emergency circuit breaker overrides trend)
        with self.subTest(msg="disaster_floor_executes_even_when_trend_up"):
            inst = FakeInstrument(price=96.0, free=4, fills=[_fill("BUY", 100, 4)])
            self.run_tick(inst, trend_up=True, loss=0.025)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertEqual(order["kwargs"]["bypass_guards"], {"all"})

        # Subtest 3: Disaster floor breach, trend_up=False -> EXECUTE
        with self.subTest(msg="disaster_floor_executes_when_trend_not_up"):
            inst = FakeInstrument(price=96.0, free=4, fills=[_fill("BUY", 100, 4)])
            self.run_tick(inst, trend_up=False, loss=0.025)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertEqual(order["kwargs"]["bypass_guards"], {"all"})

        # Subtest 4: Active crash cascade, trend_up=True -> DEFER (bounce protection)
        with self.subTest(msg="active_cascade_deferred_when_trend_up"):
            inst = FakeInstrument(price=97.0, free=4, fills=[_fill("BUY", 100, 4)])
            with patch("intelligence.external.guards.smart_stop_loss_guard.evaluate_smart_stop_loss",
                       return_value=(True, "active_crash_cascade", {})):
                self.run_tick(inst, trend_up=True, loss=0.025)
            self.assertEqual(inst.placed, [])

        # Subtest 5: Active crash cascade, trend_up=False -> EXECUTE
        with self.subTest(msg="active_cascade_executes_when_trend_not_up"):
            inst = FakeInstrument(price=97.0, free=4, fills=[_fill("BUY", 100, 4)])
            with patch("intelligence.external.guards.smart_stop_loss_guard.evaluate_smart_stop_loss",
                       return_value=(True, "active_crash_cascade", {})):
                self.run_tick(inst, trend_up=False, loss=0.025)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertEqual(order["kwargs"]["bypass_guards"], {"all"})


    def test_stop_loss_unlocks_resting_tp_orders(self):
        """Verify that when stop-loss triggers:
        1. Resting TP sell orders on the exchange are automatically cancelled to unlock captive coins.
        2. Recent trade fills do not impede stop-loss execution.
        3. If placement fails, cancelled TP order is restored.
        """
        mt.HARD_TP_ENABLED = False

        # Subtest 1: Coins locked in open TP order (free=0, but open_orders has SELL) -> cancels TP, unlocks free, executes SELL
        with self.subTest(msg="cancels_open_tp_order_and_sells_unlocked_qty"):
            inst = FakeInstrument(
                price=94.0,  # 6% loss vs 5% threshold
                free=0.0,   # free balance is 0 because all 4 coins are locked in TP
                fills=[_fill("BUY", 100, 4)],
                open_orders=[{"orderId": "tp-order-123", "side": "SELL", "price": 102.0, "origQty": 4.0}],
            )
            self.run_tick(inst, trend_up=False, loss=0.05)
            # Verify TP was cancelled
            self.assertIn("tp-order-123", inst.cancelled)
            # Verify Stop-Loss SELL was placed
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertEqual(order["kwargs"]["bypass_guards"], {"all"})

        # Subtest 2: Recent trade does not block Stop-Loss
        with self.subTest(msg="recent_trade_does_not_block_stop_loss"):
            inst = FakeInstrument(
                price=94.0,
                free=4.0,
                fills=[_fill("BUY", 100, 4, age_s=20 * 60)],  # bought only 20 min ago
            )
            self.run_tick(inst, trend_up=False, loss=0.05)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "SELL")

        # Subtest 3: If placement fails, cancelled TP order is rolled back / restored
        with self.subTest(msg="restores_cancelled_tp_order_if_placement_fails"):
            inst = RefusingInstrument(
                price=94.0,
                free=0.0,
                fills=[_fill("BUY", 100, 4)],
                open_orders=[{"orderId": "tp-order-456", "side": "SELL", "price": 102.0, "origQty": 4.0}],
            )
            self.run_tick(inst, trend_up=False, loss=0.05)
            self.assertIn("tp-order-456", inst.cancelled)
            # The refusing instrument rejected the SL, so a restore TP placement was attempted
            restore_orders = [o for o in inst.placed if o.get("kwargs", {}).get("kind") == "TP"]
            self.assertEqual(len(restore_orders), 1)
            self.assertEqual(restore_orders[0]["price"], 102.0)
            self.assertEqual(restore_orders[0]["qty"], 4.0)

        # Subtest 4: If STOP placement outcome is unknown (in-flight / timeout), do NOT restore cancelled TP order!
        with self.subTest(msg="does_not_restore_tp_when_stop_outcome_unknown"):
            class UnknownOutcomeInstrument(FakeInstrument):
                def place(self, side, price, qty, **kwargs):
                    super().place(side, price, qty, **kwargs)
                    outcome_ctx = kwargs.get("_outcome_context")
                    if outcome_ctx is not None and isinstance(outcome_ctx, dict):
                        outcome_ctx.update(accepted=False, reason="timeout", state="unknown")
                    return None

            inst = UnknownOutcomeInstrument(
                price=94.0,
                free=0.0,
                fills=[_fill("BUY", 100, 4)],
                open_orders=[{"orderId": "tp-order-789", "side": "SELL", "price": 102.0, "origQty": 4.0}],
            )
            self.run_tick(inst, trend_up=False, loss=0.05)
            self.assertIn("tp-order-789", inst.cancelled)
            # TP must NOT be restored while STOP outcome is unknown/ambiguous
            restore_orders = [o for o in inst.placed if o.get("kwargs", {}).get("kind") == "TP"]
            self.assertEqual(restore_orders, [], "TP must NOT be restored after unknown/ambiguous STOP outcome")

    def test_mhdsl_bounce_timeout_forces_stop_loss(self):
        """Proposal 2: Time-bounded bounce deferral.
        If trend_up stays True past MT_SL_TREND_DEFER_MAX_SEC (220s),
        the deferral timer expires and Stop-Loss executes.
        """
        inst = FakeInstrument(price=97.0, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol
        # Tick 1 at t=0: initial dip (3% loss > 2.5% threshold), trend_up=True -> defers
        self.run_tick(inst, trend_up=True, loss=0.025, now=NOW)
        self.assertEqual(inst.placed, [])
        self.assertIn(sym, mt._SL_BOUNCE_STATE)

        # Tick 2 at t=100s: still in window (< 220s), trend_up=True -> still defers
        self.run_tick(inst, trend_up=True, loss=0.025, now=NOW + 100)
        self.assertEqual(inst.placed, [])

        # Tick 3 at t=230s: window expired (230s >= 220s), PositionExitManager triggers Stop-Loss,
        # while wait_for_trend remains True (monitortrades never bypasses GT_S).
        self.run_tick(inst, trend_up=True, loss=0.025, now=NOW + 230)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 4.0))
        self.assertIs(inst.placed[0]["kwargs"].get("wait_for_trend"), True)
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)

        # If GT_S short_trend gate is active on live venue, _execute_stop_loss still defers
        # and keeps resting TP orders intact even after bounce timeout.
        inst_live = FakeInstrument(
            price=97.0,
            free=4.0,
            fills=[_fill("BUY", 100, 4)],
            open_orders=[{"orderId": "tp-keep-1", "side": "SELL", "price": 105.0, "origQty": 4.0}],
        )
        with patch.object(mt, "_is_trend_gate_deferred", return_value=True):
            self.run_tick(inst_live, trend_up=True, loss=0.025, now=NOW)
            self.run_tick(inst_live, trend_up=True, loss=0.025, now=NOW + 230)
        self.assertEqual(inst_live.placed, [])
        self.assertEqual(inst_live.cancelled, [])

    def test_mhdsl_bounce_rollover_forces_stop_loss(self):
        """Proposal 3: Dynamic micro-trailing on bounce rollover.
        If price bounces significantly (>= min_recovery_pct) and then drops
        from the bounce peak (>= pullback_pct), Stop-Loss executes without
        waiting for the timeout.
        """
        inst = FakeInstrument(price=94.5, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol
        # Threshold: 5.0% loss. Disaster floor: 5.0% + 1.5% = 6.5%.
        # Tick 1 at t=0: price=94.5 (dip 5.5% between SL 5.0% and floor 6.5%), trend_up=True -> arms bounce state
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2 at t=10s: price bounces to 94.9 (loss 5.1%, recovery (94.9-94.5)/94.5 = +0.42% >= 0.4%)
        inst._price = 94.9
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW + 10)
        self.assertEqual(inst.placed, [])
        self.assertEqual(mt._SL_BOUNCE_STATE[sym]["peak_bounce_price"], 94.9)

        # Tick 3 at t=20s: price rolls over to 94.55 (pulled back 0.37% from peak 94.9 >= 0.3% threshold)
        inst._price = 94.55
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW + 20)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 4.0))
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)

    def test_mhdsl_bounce_full_recovery_clears_state(self):
        """When price recovers above the lost_threshold, bounce tracking state is cleared."""
        inst = FakeInstrument(price=97.0, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol
        # Tick 1: dip into SL territory (3% loss > 2.5% threshold), trend_up=True -> arms
        self.run_tick(inst, trend_up=True, loss=0.025, now=NOW)
        self.assertIn(sym, mt._SL_BOUNCE_STATE)

        # Tick 2: price recovers to 99.0 (loss 1.0% <= threshold 2.5%) -> cleared!
        inst._price = 99.0
        self.run_tick(inst, trend_up=True, loss=0.025, now=NOW + 10)
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)
        self.assertEqual(inst.placed, [])

    def test_mhdsl_parallel_volatility_scaling_disaster_floor(self):
        """Verify parallel disaster buffer resolution (Variant A proportional + Variant B realized volatility)."""
        # Case 1: No volatility data -> Variant A dominates (30% of lost_threshold)
        with patch.object(mt, "get_symbol_volatility_1h", return_value=None):
            buf, meta = mt.compute_disaster_buffer("BTCUSDT", lost_threshold=0.02925)
            self.assertAlmostEqual(buf, 0.02925 * 0.30, places=5)
            self.assertEqual(meta["source"], "variant_a_proportional")

        # Case 2: High realized volatility (2.0%) -> Variant B dominates (1.5 * 2.0% = 3.0% > 0.8775%)
        with patch.object(mt, "get_symbol_volatility_1h", return_value=0.020):
            buf, meta = mt.compute_disaster_buffer("BTCUSDT", lost_threshold=0.02925)
            self.assertAlmostEqual(buf, 0.020 * 1.5, places=5)
            self.assertEqual(meta["source"], "variant_b_vol1h")

        # Case 3: Very low threshold & low volatility -> Bounded by MT_SL_DISASTER_BUFFER_MIN_PCT (0.5%)
        with patch.object(mt, "get_symbol_volatility_1h", return_value=0.001):
            buf, meta = mt.compute_disaster_buffer("BTCUSDT", lost_threshold=0.005)
            self.assertAlmostEqual(buf, mt.MT_SL_DISASTER_BUFFER_MIN_PCT, places=5)

    def test_get_symbol_volatility_1h_explicit_percentage_to_ratio_conversion(self):
        """Verify get_symbol_volatility_1h explicitly converts percentage to decimal ratio (no 100x blowup for vol < 0.10%)."""
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_data = {
                "BTCUSDT": {"vol_1h_pct": 2.5},     # 2.5% normal vol -> ratio 0.025
                "ETHUSDT": {"vol_1h_pct": 0.05},    # 0.05% low vol -> ratio 0.0005 (NOT 0.05!)
                "USDCUSDT": {"vol_1h_pct": 0.0031}, # 0.0031% very low vol -> ratio 0.000031
            }
            with open(os.path.join(tmp_dir, "cache_kalman_trend.json"), "w", encoding="utf-8") as f:
                json.dump(state_data, f)

            with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": tmp_dir}):
                # 1. Normal volatility: 2.5% -> 0.025
                self.assertAlmostEqual(mt.get_symbol_volatility_1h("BTCUSDT"), 0.025, places=6)
                # 2. Low volatility < 0.10%: 0.05% -> 0.0005, MUST NOT return 0.05
                vol_eth = mt.get_symbol_volatility_1h("ETHUSDT")
                self.assertAlmostEqual(vol_eth, 0.0005, places=6)
                self.assertNotAlmostEqual(vol_eth, 0.05, places=3)
                # 3. Very low volatility: 0.0031% -> 0.000031
                self.assertAlmostEqual(mt.get_symbol_volatility_1h("USDCUSDT"), 0.000031, places=6)
                # 4. Unknown symbol returns None
                self.assertIsNone(mt.get_symbol_volatility_1h("UNKNOWN"))

    def test_mdhtp_trend_riding_and_peak_tracking(self):
        """Verify MDHTP arms ride state and tracks higher highs while trend_up is True."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=108.0, free=5.0, fills=[_fill("BUY", 100, 5)])
        sym = inst.symbol

        # Tick 1 at t=NOW: price 108 (+8.0% > 7.0% threshold), trend_up=True -> defers TP, arms ride state
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW)
        self.assertEqual(inst.placed, [])
        self.assertIn(sym, mt._TP_RIDE_STATE)
        self.assertEqual(mt._TP_RIDE_STATE[sym]["peak_price"], 108.0)
        self.assertEqual(mt._TP_RIDE_STATE[sym]["last_peak_time"], NOW)

        # Tick 2 at t=NOW+100: price climbs to 112.0 (+12.0%), trend_up=True -> updates peak and resets last_peak_time
        inst._price = 112.0
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW + 100)
        self.assertEqual(inst.placed, [])
        self.assertEqual(mt._TP_RIDE_STATE[sym]["peak_price"], 112.0)
        self.assertEqual(mt._TP_RIDE_STATE[sym]["last_peak_time"], NOW + 100)

    def test_mdhtp_micro_trailing_peak_pullback_triggers_sell(self):
        """Verify that when price pulls back >= MT_TP_PULLBACK_PCT (0.6%) from the ride peak, TP executes."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=110.0, free=5.0, fills=[_fill("BUY", 100, 5)])
        sym = inst.symbol

        # Tick 1: price=110.0 (+10% gain), trend_up=True -> arms ride state
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2: price pulls back to 109.2 (pullback = (110 - 109.2)/110 = 0.727% >= 0.60%)
        inst._price = 109.2
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW + 30)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 5.0))
        self.assertNotIn(sym, mt._TP_RIDE_STATE)

    def test_mdhtp_stagnation_timeout_triggers_sell(self):
        """Verify that when no new peak is made for >= MT_TP_STAGNATION_MAX_SEC (1080s), TP executes."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=110.0, free=5.0, fills=[_fill("BUY", 100, 5)])
        sym = inst.symbol

        # Tick 1: price=110.0, arms at t=NOW
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2 at t=NOW+500s (< 1080s): slight dip to 109.8 (pullback 0.18% < 0.6%) -> still riding
        inst._price = 109.8
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW + 500)
        self.assertEqual(inst.placed, [])

        # Tick 3 at t=NOW+1100s (>= 1080s stagnation): still at 109.8 -> stagnation timer expired, forces TP!
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW + 1100)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 5.0))
        self.assertNotIn(sym, mt._TP_RIDE_STATE)

    def test_mdhtp_guaranteed_profit_floor_breached_triggers_sell(self):
        """Verify that if price slides below guaranteed profit floor, TP executes immediately to protect gains."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=108.0, free=5.0, fills=[_fill("BUY", 100, 5)])
        sym = inst.symbol

        # Threshold=0.07 (7%), floor = max(0.07 * 0.5, 0.008) = 0.035 (3.5%)
        # Tick 1: price=108.0 (+8.0%), trend_up=True -> arms ride state
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2: price drops to 103.0 (+3.0% gain < floor 3.5%), trend_up=True -> floor breached, executes!
        inst._price = 103.0
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW + 30)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 5.0))
        self.assertNotIn(sym, mt._TP_RIDE_STATE)

    def test_mdhtp_trend_reversal_triggers_sell(self):
        """Verify that when trend_up flips to False during an active ride, TP executes immediately."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=109.0, free=5.0, fills=[_fill("BUY", 100, 5)])
        sym = inst.symbol

        # Tick 1: price=109.0 (+9.0%), trend_up=True -> arms ride state
        self.run_tick(inst, trend_up=True, gain=0.07, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2: price=109.0, trend_up=False -> trend reversal, executes!
        self.run_tick(inst, trend_up=False, gain=0.07, now=NOW + 15)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 5.0))
        self.assertNotIn(sym, mt._TP_RIDE_STATE)

    def test_mhdsl_structural_break_triggers_sell_when_enabled(self):
        """Variant PROPOSED: When an attempted bounce breaks below the established support low, exit immediately."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=94.5, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol

        # Tick 1: dip to 94.5 (loss 5.5% > 5%), trend_up=True -> arms bounce state with lowest=94.5
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW)
        self.assertEqual(inst.placed, [])
        self.assertIn(sym, mt._SL_BOUNCE_STATE)

        # Tick 2: bounce to 94.9 (+0.42% recovery >= 0.40% min recovery)
        inst._price = 94.9
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW + 10)
        self.assertEqual(inst.placed, [])

        # Tick 3: drops to 94.4 (below lowest_px 94.5) -> structural support break triggers immediate exit!
        inst._price = 94.4
        with patch.object(mt, "MT_SL_STRUCTURAL_BREAK_ENABLED", True):
            self.run_tick(inst, trend_up=True, loss=0.05, now=NOW + 20)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 4.0))
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)

    def test_mhdsl_structural_break_disabled_resets_trough_variant_baseline(self):
        """Variant BASELINE: When MT_SL_STRUCTURAL_BREAK_ENABLED is False, a new low resets lowest_price and continues deferring."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=94.5, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol

        # Tick 1: dip to 94.5
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2: bounce to 94.9
        inst._price = 94.9
        self.run_tick(inst, trend_up=True, loss=0.05, now=NOW + 10)
        self.assertEqual(inst.placed, [])

        # Tick 3: drops to 94.4 with structural break disabled -> resets trough to 94.4 and defers
        inst._price = 94.4
        with patch.object(mt, "MT_SL_STRUCTURAL_BREAK_ENABLED", False):
            self.run_tick(inst, trend_up=True, loss=0.05, now=NOW + 20)
        self.assertEqual(inst.placed, [])
        self.assertIn(sym, mt._SL_BOUNCE_STATE)
        self.assertEqual(mt._SL_BOUNCE_STATE[sym]["lowest_price"], 94.4)

    def test_mhdsl_bounce_ratchet_tightens_pullback(self):
        """Verify dynamic ratchet tightens pullback threshold when bounce recovery is strong."""
        mt.HARD_TP_ENABLED = False
        inst = FakeInstrument(price=91.5, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol

        # Threshold: 7.0% loss. Disaster floor: 7.0% + 2.1% = 9.1%.
        # Tick 1: dip to 91.5 (loss 8.5% between 7.0% and 9.1%)
        self.run_tick(inst, trend_up=True, loss=0.07, now=NOW)
        self.assertEqual(inst.placed, [])

        # Tick 2: strong bounce to 92.8 (+1.42% recovery >> 0.4%, loss 7.2% > 7.0% threshold)
        inst._price = 92.8
        self.run_tick(inst, trend_up=True, loss=0.07, now=NOW + 10)
        self.assertEqual(inst.placed, [])

        # Tick 3: price pulls back to 92.6 (pullback (92.8 - 92.6)/92.8 = 0.215%, loss 7.4%).
        # Standard pullback threshold is 0.30% (which would NOT trigger at 0.215%).
        # With ratchet enabled, effective pullback is tightened to 0.15%, so 0.215% TRIGGERS exit!
        inst._price = 92.6
        with patch.object(mt, "MT_SL_BOUNCE_RATCHET_ENABLED", True):
            self.run_tick(inst, trend_up=True, loss=0.07, now=NOW + 20)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 4.0))
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)

    def test_clear_position_stats_cache_resets_all_exit_state(self):
        mt._position_stats_cache["foo"] = "bar"
        mt._REENTRY_STATE["BTCUSDC"] = {"peak": 100.0}
        mt._LAST_EXIT_KIND["BTCUSDC"] = "STOP"
        mt._SL_BOUNCE_STATE["BTCUSDC"] = {"peak_bounce_price": 95.0}
        mt._TP_RIDE_STATE["BTCUSDC"] = {"peak_price": 105.0}
        mt._hard_tp_last["BTCUSDC"] = 12345678
        mt._notified_trade_ids.add("order_1")

        mt.clear_position_stats_cache()

        self.assertEqual(len(mt._position_stats_cache), 0)
        self.assertEqual(len(mt._REENTRY_STATE), 0)
        self.assertEqual(len(mt._LAST_EXIT_KIND), 0)
        self.assertEqual(len(mt._SL_BOUNCE_STATE), 0)
        self.assertEqual(len(mt._TP_RIDE_STATE), 0)
        self.assertEqual(len(mt._hard_tp_last), 0)
        self.assertEqual(len(mt._notified_trade_ids), 0)

    def test_hard_tp_unknown_outcome_sets_cooldown_and_returns(self):
        """When Hard-TP submit has unknown/ambiguous outcome (timeout), cooldown is set and tick returns."""
        class UnknownOutcomeInstrument(FakeInstrument):
            def place(self, side, price, qty, **kwargs):
                super().place(side, price, qty, **kwargs)
                ctx = kwargs.get("_outcome_context")
                if isinstance(ctx, dict):
                    ctx.update(accepted=False, reason="timeout", state="unknown")
                return None

        inst = UnknownOutcomeInstrument(price=125.0, free=4.0, fills=[_fill("BUY", 100, 4)])
        sym = inst.symbol
        # Hard TP is triggered at 25% gain (>= hardtp 17%)
        with patch.object(mt, "HARD_TP_ENABLED", True):
            self.run_tick(inst, trend_up=True, now=NOW)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "SELL")
        self.assertEqual(mt._hard_tp_last.get(sym), NOW)
        self.assertEqual(mt._LAST_EXIT_KIND.get(sym), "TP")

    def test_reentry_unknown_outcome_clears_reentry_state(self):
        """When Reentry BUY submit has unknown/ambiguous outcome, reentry state is cleared to prevent repeated spam."""
        class UnknownOutcomeInstrument(FakeInstrument):
            def place(self, side, price, qty, **kwargs):
                super().place(side, price, qty, **kwargs)
                ctx = kwargs.get("_outcome_context")
                if isinstance(ctx, dict):
                    ctx.update(accepted=False, reason="timeout", state="unknown")
                return None

        inst = UnknownOutcomeInstrument(
            price=85.0,
            free=0.0,
            fills=[_fill("SELL", 100.0, 4.0)],
        )
        sym = inst.symbol
        mt._LAST_EXIT_KIND[sym] = "STOP"
        mt._REENTRY_STATE[sym] = {
            "trough": 80.0,
            "peak": 100.0,
            "peak_bounce": 85.0,
            "started_at": NOW - 300,
        }
        self.run_tick(inst, trend_up=True, now=NOW)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "BUY")
        self.assertNotIn(sym, mt._REENTRY_STATE)
        self.assertNotIn(sym, mt._LAST_EXIT_KIND)

    def test_stop_loss_unknown_outcome_defers_duplicate_submission(self):
        """When Stop-Loss submit has unknown/ambiguous outcome, in-flight state is tracked and duplicate submission deferred."""
        class UnknownOutcomeInstrument(FakeInstrument):
            def place(self, side, price, qty, **kwargs):
                super().place(side, price, qty, **kwargs)
                ctx = kwargs.get("_outcome_context")
                if isinstance(ctx, dict):
                    ctx.update(accepted=False, reason="timeout", state="unknown")
                return None

        # Position bought at 100, now dropped to 95 (-5% > lost_threshold 3.3%)
        inst = UnknownOutcomeInstrument(
            price=95.0,
            free=4.0,
            fills=[_fill("BUY", 100.0, 4.0)],
        )
        sym = inst.symbol

        # Tick 1: Trend down -> Stop loss triggers and attempts SELL with unknown outcome
        self.run_tick(inst, trend_up=False, now=NOW)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "SELL")
        self.assertIn(sym, mt._PENDING_STOP_EXIT)
        self.assertEqual(mt._LAST_EXIT_KIND.get(sym), "STOP")

        # Tick 2: Trend still down, next tick 5 seconds later. Should DEFER duplicate submission!
        self.run_tick(inst, trend_up=False, now=NOW + 5)
        self.assertEqual(len(inst.placed), 1, "Duplicate STOP order must not be submitted while pending reconciliation")

        # Tick 3: Position closed (free=0.0). Pending stop is resolved cleanly.
        inst._free = 0.0
        self.run_tick(inst, trend_up=False, now=NOW + 10)
        self.assertNotIn(sym, mt._PENDING_STOP_EXIT)

    def test_bounce_recovery_triggers_take_profit_on_same_tick(self):
        """When price recovers from bounce and crosses into TP territory, TP must execute on that same tick without a 1-tick delay."""
        inst = FakeInstrument(
            price=95.0,
            free=4.0,
            fills=[_fill("BUY", 100.0, 4.0)],
        )
        sym = inst.symbol

        # Tick 1: Dip to 95 (-5%) with trend up -> enters bounce watch
        self.run_tick(inst, trend_up=True, now=NOW)
        self.assertEqual(len(inst.placed), 0, "Stop-Loss should be deferred by upward trend")
        self.assertIn(sym, mt._SL_BOUNCE_STATE)

        # Tick 2: Price surges to 110 (+10% > gain_threshold 7%). Trend down (so TP executes immediately rather than riding).
        inst._price = 110.0
        self.run_tick(inst, trend_up=False, now=NOW + 10)
        self.assertEqual(len(inst.placed), 1, "Take-Profit must trigger immediately on the recovery tick without a 1-tick delay")
        self.assertEqual(inst.placed[0]["side"], "SELL")
        self.assertEqual(inst.placed[0]["price"], 110.0)
        self.assertEqual(mt._LAST_EXIT_KIND.get(sym), "TP")
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)

    def test_mt_state_persists_across_restart(self):
        """In-flight MHDSL bounce, MDHTP ride, and exit kinds must survive monitor restart via persistent state file."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_state_file = tf.name
        try:
            with patch.object(mt, "_MT_STATE_FILE", temp_state_file):
                sym = "TESTSYM"
                mt._SL_BOUNCE_STATE[sym] = {"started_at": NOW - 50, "peak_bounce_price": 98.0, "disaster_floor": 0.04}
                mt._TP_RIDE_STATE[sym] = {"started_at": NOW - 30, "peak_price": 115.0, "peak_gain": 0.15}
                mt._LAST_EXIT_KIND[sym] = "TP"
                mt._save_mt_state()

                self.assertTrue(os.path.exists(temp_state_file))

                # Simulate monitor restart: wipe in-memory structures
                mt._SL_BOUNCE_STATE.clear()
                mt._TP_RIDE_STATE.clear()
                mt._LAST_EXIT_KIND.clear()

                # Reload from disk
                mt._load_mt_state()
                self.assertIn(sym, mt._SL_BOUNCE_STATE)
                self.assertEqual(mt._SL_BOUNCE_STATE[sym]["peak_bounce_price"], 98.0)
                self.assertIn(sym, mt._TP_RIDE_STATE)
                self.assertEqual(mt._TP_RIDE_STATE[sym]["peak_price"], 115.0)
                self.assertEqual(mt._LAST_EXIT_KIND.get(sym), "TP")
        finally:
            if os.path.exists(temp_state_file):
                try:
                    os.remove(temp_state_file)
                except OSError:
                    pass

    def test_smart_sl_grace_cleared_on_exit(self):
        """Smart-SL grace tracker entries must be cleaned up when an exit executes."""
        from intelligence.external.guards.smart_stop_loss_guard import _GRACE_TRACKER
        inst = FakeInstrument(
            price=95.0,
            free=4.0,
            fills=[_fill("BUY", 100.0, 4.0)],
        )
        sym = inst.symbol
        _GRACE_TRACKER[sym] = {
            "start_ts": NOW - 10,
            "initial_loss_pct": 5.0,
            "lost_threshold_pct": 3.3,
            "provider": "binance",
        }

        # Trend down -> STOP loss executes
        self.run_tick(inst, trend_up=False, now=NOW)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "SELL")
        self.assertNotIn(sym, _GRACE_TRACKER, "Active Smart-SL grace must be cleared after protective STOP placement")

    def test_closed_position_skips_ghost_stop_loss_and_infers_stop_reentry(self):
        """After a STOP exit closes the position (sell_time > buy_time, free < min_qty), Step 3 must not
        fire ghost Stop-Loss evaluations, and Step 4 must infer STOP exit kind even if state was cleared."""
        mt.HARD_TP_ENABLED = False
        mt._LAST_EXIT_KIND.clear()
        mt._REENTRY_STATE.clear()
        inst = FakeInstrument(
            price=90.0,
            free=0.000005,
            min_qty=0.00001,
            fills=[
                _fill("BUY", 100.0, 2.0, age_s=6 * 3600),
                _fill("SELL", 95.0, 1.999995, age_s=4 * 3600),
            ],
            params={"mt.buy_budget": 200, "mt.max_budget": 1000},
        )
        sym = inst.symbol
        # Tick 1 at 90.0: skips Step 3 (no ghost SELL), infers STOP, records trough=90.0
        self.run_tick(inst, trend_up=True, now=NOW)
        self.assertEqual(inst.placed, [])
        self.assertEqual(mt._LAST_EXIT_KIND.get(sym), "STOP")
        self.assertEqual(mt._REENTRY_STATE[sym]["trough"], 90.0)

        # Tick 2 at 92.0 (+2.22% bounce >= 1.5%, below sell_price 95.0): places Post-STOP re-entry BUY bypassing all guards (GT_S remains active)
        inst._price = 92.0
        self.run_tick(inst, trend_up=True, now=NOW + 60)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "BUY")
        self.assertEqual(inst.placed[0]["kwargs"]["bypass_guards"], {"all"})

    def test_stop_loss_does_not_fall_through_and_resets_stale_reentry_and_min_notional_dust(self):
        """Regression for ARBUSDC incident:
        1. Executing STOP SELL must clear stale _REENTRY_STATE and return immediately without evaluating Step 4 on the same tick.
        2. On subsequent ticks where sell_time > buy_time and remaining balance is below min_notional (even if >= min_qty),
           position_closed_by_sell must treat the remainder as dust and skip ghost Stop-Loss evaluations.
        3. Stale _REENTRY_STATE from an older sell_time is discarded and re-anchored to the new SELL.
        """
        mt.HARD_TP_ENABLED = False
        mt._LAST_EXIT_KIND.clear()
        mt._REENTRY_STATE.clear()
        sym = SYMBOL

        # Seed a stale _REENTRY_STATE from 2 days ago with an obsolete low trough
        mt._REENTRY_STATE[sym] = {
            "trough": 0.1719,
            "peak": 0.1950,
            "peak_bounce": 0.1846,
            "started_at": NOW - 2 * 86400,
        }

        # Tick 1: Position bought 1 day ago at 0.1930, older sell 2 days ago at 0.1846.
        # Current price drops to 0.1884 (-2.38% > 2.2% loss threshold), trend_up=True but disaster floor breached -> STOP executes.
        inst = FakeInstrument(
            price=0.1884,
            free=9496.6,
            min_qty=0.1,
            fills=[
                _fill("SELL", 0.1846, 5000.0, age_s=2 * 86400),
                _fill("BUY", 0.1930, 9496.6, age_s=86400),
            ],
            params={"mt.buy_budget": 250, "mt.max_budget": 2500},
        )
        self.run_tick(inst, trend_up=True, loss=0.015, now=NOW)
        # Only the STOP SELL must be placed on Tick 1; must NOT fall through to Step 4 BUY on the same tick!
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual(inst.placed[0]["side"], "SELL")
        self.assertNotIn(sym, mt._REENTRY_STATE)
        self.assertEqual(mt._LAST_EXIT_KIND.get(sym), "STOP")

        # Tick 2 (1000s later, after cooldown): The STOP SELL at 0.1884 filled, leaving 9.5 ARB (~$1.79 < $5 min_notional)
        # which is above min_qty (0.1) but refused by provider order_filter_refusal as below_min_notional.
        inst.placed.clear()
        inst._free = 9.5
        inst._price = 0.1884
        inst._fills = [
            _fill("SELL", 0.1884, 9487.1, age_s=-10),  # timestamp = NOW + 10
            _fill("BUY", 0.1930, 9496.6, age_s=86400),
        ]
        inst._provider.fills = list(inst._fills)
        inst._provider.order_filter_refusal = (
            lambda _sym, _side, px, q, **kw: "below_min_notional" if (q * px) < 5.0 else None
        )
        # Even if stale _REENTRY_STATE were restored externally, sell_time mismatch resets it to trough=0.1884
        mt._REENTRY_STATE[sym] = {
            "trough": 0.1719,
            "peak": 0.1950,
            "peak_bounce": 0.1884,
            "started_at": NOW - 2 * 86400,
        }
        self.run_tick(inst, trend_up=True, loss=0.022, now=NOW + 1000)
        # Must NOT fire ghost Stop-Loss SELL, and must NOT fire false Post-STOP BUY (bounce from 0.1884 is 0% < 1.5%)
        self.assertEqual(inst.placed, [])
        self.assertEqual(mt._REENTRY_STATE[sym]["trough"], 0.1884)
        self.assertEqual(mt._REENTRY_STATE[sym]["sell_time"], NOW + 10)

    def test_active_cycle_cost_basis_and_reentry_guards(self):
        """Regression tests for:
        1. mt.ref='average' uses active_average_buy_price (ignoring closed higher cycles and <=2% fee dust).
        2. Profitable TP sell on a lower cycle after a closed higher cycle is NOT falsely inferred as STOP.
        3. Step 4 does not re-buy against an older SELL after a newer BUY has already re-entered (buy_time > sell_time).
        4. Step 4 checks max_budget (including owned_qty when coins are locked in an open order) before invoking AI guard.
        """
        mt.HARD_TP_ENABLED = False

        with self.subTest(msg="active_average_buy_price_excludes_closed_cycle_with_fee_dust"):
            # Cycle 1: BUY 10 @ 120 -> SELL 9.99 @ 130 (0.1% fee dust left). Cycle 2: BUY 10 @ 90.
            fills = [
                _fill("BUY", 90.0, 10.0, age_s=1 * 3600),
                _fill("SELL", 130.0, 9.99, age_s=3 * 3600),
                _fill("BUY", 120.0, 10.0, age_s=5 * 3600),
            ]
            stats = mt.get_position_stats(SYMBOL, 10 * 24 * 3600, api=FakeProvider(fills))
            self.assertAlmostEqual(stats["average_buy_price"], 105.0, places=4)
            self.assertAlmostEqual(stats["active_average_buy_price"], 90.0, places=4)

            # With mt.ref='average', current price 100.0 is +11.1% above active avg 90.0 (not -4.76% below gross 105.0)
            inst = FakeInstrument(
                price=100.0,
                free=10.0,
                fills=fills,
                params={"mt.ref": "average"},
            )
            self.run_tick(inst, trend_up=False, gain=0.10, loss=0.04)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "SELL")
            self.assertIsNone(inst.placed[0]["kwargs"].get("bypass_guards"))

        with self.subTest(msg="profitable_tp_on_lower_cycle_not_inferred_as_stop"):
            mt._LAST_EXIT_KIND.clear()
            mt._REENTRY_STATE.clear()
            # Cycle 1 closed at 120 -> 115. Cycle 2 bought at 90 and sold for profit at 98 (+8.9%).
            fills = [
                _fill("SELL", 98.0, 10.0, age_s=1 * 3600),
                _fill("BUY", 90.0, 10.0, age_s=2 * 3600),
                _fill("SELL", 115.0, 10.0, age_s=5 * 3600),
                _fill("BUY", 120.0, 10.0, age_s=6 * 3600),
            ]
            inst = FakeInstrument(
                price=96.0,
                free=0.0,
                min_qty=0.1,
                fills=fills,
                params={"mt.buy_budget": 200, "mt.max_budget": 1000},
            )
            self.run_tick(inst, trend_up=True, gain=0.10, loss=0.05)
            # Must NOT infer STOP because 98.0 > 90.0 (Cycle 2 cost basis), even though gross avg is 105.0
            self.assertNotEqual(mt._LAST_EXIT_KIND.get(SYMBOL), "STOP")
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="step4_skipped_when_newer_buy_already_reentered_after_sell"):
            mt._LAST_EXIT_KIND.clear()
            mt._REENTRY_STATE.clear()
            # Older SELL at 100 (4h ago), already re-entered by BUY at 90 (2h ago). Price is still 90.
            fills = [
                _fill("BUY", 90.0, 2.0, age_s=2 * 3600),
                _fill("SELL", 100.0, 2.0, age_s=4 * 3600),
            ]
            inst = FakeInstrument(
                price=90.0,
                free=2.0,
                fills=fills,
                params={"mt.buy_budget": 200, "mt.max_budget": 1000},
            )
            self.run_tick(inst, trend_up=True, gain=0.08, loss=0.05)
            self.assertEqual(inst.placed, [], "Must not place duplicate buyback against an older SELL after a newer BUY")

        with self.subTest(msg="max_budget_checked_before_ai_guard_and_counts_owned_qty"):
            mt._LAST_EXIT_KIND.clear()
            mt._REENTRY_STATE.clear()
            mt._LAST_EXIT_KIND[SYMBOL] = "STOP"
            mt._REENTRY_STATE[SYMBOL] = {
                "trough": 88.0,
                "peak": 100.0,
                "peak_bounce": 90.0,
                "started_at": NOW - 600,
                "sell_time": NOW - 3600,
            }
            # Partial sell left 10 owned coins, all locked in an open order (free=0.0).
            # At price=90.0, owned exposure = 10 * 90 = $900 == max_budget ($900).
            fills = [
                _fill("SELL", 100.0, 2.0, age_s=3600),
                _fill("BUY", 100.0, 12.0, age_s=7200),
            ]
            inst = FakeInstrument(
                price=90.0,
                free=0.0,
                fills=fills,
                params={"mt.buy_budget": 200, "mt.max_budget": 900},
            )
            with (
                patch.object(mt, "is_trend_up", return_value=True),
                patch.object(mt, "_apply_post_stop_ai_guard") as mock_ai_guard,
            ):
                mt.monitor_price_and_trade(
                    inst,
                    sbs=12 * 24 * 3600 + 60,
                    maxage_trade_s=10 * 24 * 3600,
                    gain_threshold=0.10,
                    lost_threshold=0.20,
                    now_fn=lambda: NOW,
                )
            mock_ai_guard.assert_not_called()
            self.assertEqual(inst.placed, [])

        with self.subTest(msg="tp_ride_gap_down_into_sl_territory_forces_immediate_exit"):
            mt._TP_RIDE_STATE.clear()
            mt._SL_BOUNCE_STATE.clear()
            inst = FakeInstrument(
                price=108.0,
                free=4.0,
                fills=[_fill("BUY", 100.0, 4.0)],
            )
            # Tick 1 at 108 (+8% > 7%): initiates TP ride
            self.run_tick(inst, trend_up=True, gain=0.07, loss=0.05, now=NOW)
            self.assertIn(SYMBOL, mt._TP_RIDE_STATE)
            self.assertEqual(inst.placed, [])

            # Tick 2: flash crash from 108 to 94.5 (-5.5% loss, inside SL bounce corridor with trend_up=True).
            # Must NOT defer via SL bounce watch; must exit immediately via mdhtp_profit_floor_breached!
            inst._price = 94.5
            self.run_tick(inst, trend_up=True, gain=0.07, loss=0.05, now=NOW + 10)
            self.assertEqual(len(inst.placed), 1)
            self.assertEqual(inst.placed[0]["side"], "SELL")
            self.assertNotIn(SYMBOL, mt._TP_RIDE_STATE)

        with self.subTest(msg="reentry_peak_bounce_tracked_while_trend_down_detects_rollover"):
            mt._REENTRY_STATE.clear()
            mt._LAST_EXIT_KIND.clear()
            mt._LAST_EXIT_KIND[SYMBOL] = "STOP"
            inst = FakeInstrument(
                price=90.0,
                free=0.0,
                fills=[_fill("SELL", 100.0, 2.0)],
                params={"mt.buy_budget": 200, "mt.max_budget": 1000},
            )
            # Tick 1 (trend_up=False) at 90.0: establishes trough=90.0, peak_bounce=90.0
            self.run_tick(inst, trend_up=False, now=NOW)
            # Tick 2 (trend_up=False) at 94.0: bounces +4.4% while trend is still down -> updates peak_bounce=94.0
            inst._price = 94.0
            self.run_tick(inst, trend_up=False, now=NOW + 10)
            self.assertEqual(mt._REENTRY_STATE[SYMBOL]["peak_bounce"], 94.0)

            # Tick 3 (trend_up=True) at 91.5: still +1.67% above trough 90.0 (>= 1.5%),
            # but rolled over -2.66% from peak_bounce 94.0 (> 0.75% allowed rollover) -> blocked!
            inst._price = 91.5
            self.run_tick(inst, trend_up=True, now=NOW + 20)
            self.assertEqual(inst.placed, [], "Rolled-over bounce must be blocked by ratchet")

        with self.subTest(msg="hard_tp_last_persisted_across_restart"):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                temp_state_file = tf.name
            try:
                with patch.object(mt, "_MT_STATE_FILE", temp_state_file):
                    mt._hard_tp_last["TESTSYM"] = NOW - 120
                    mt._save_mt_state(ts=NOW)
                    mt._hard_tp_last.clear()
                    mt._load_mt_state(now_curr=NOW)
                    self.assertEqual(mt._hard_tp_last.get("TESTSYM"), NOW - 120)
            finally:
                if os.path.exists(temp_state_file):
                    os.remove(temp_state_file)

        with self.subTest(msg="tp_unknown_outcome_defers_duplicate_submission_on_next_tick"):
            mt._PENDING_STOP_EXIT.clear()
            mt._TP_RIDE_STATE.clear()

            class UnknownTpInstrument(FakeInstrument):
                def place(self, side, price, qty, **kwargs):
                    super().place(side, price, qty, **kwargs)
                    ctx = kwargs.get("_outcome_context")
                    if isinstance(ctx, dict):
                        ctx.update(accepted=False, reason="timeout", state="unknown")
                    return None

            inst = UnknownTpInstrument(
                price=110.0,
                free=4.0,
                fills=[_fill("BUY", 100.0, 4.0)],
            )
            # Tick 1: trend_up=False, +10% gain -> TP attempts SELL with unknown outcome
            self.run_tick(inst, trend_up=False, gain=0.07, now=NOW)
            self.assertEqual(len(inst.placed), 1)
            self.assertIn(SYMBOL, mt._PENDING_STOP_EXIT)

            # Tick 2 (5s later): even if price stays at 110 or drops to 101 (recovery zone),
            # pending exit check at top of Step 3 must defer and preserve _PENDING_STOP_EXIT!
            inst._price = 101.0
            self.run_tick(inst, trend_up=False, gain=0.07, now=NOW + 5)
            self.assertEqual(len(inst.placed), 1)
            self.assertIn(SYMBOL, mt._PENDING_STOP_EXIT)

        with self.subTest(msg="pending_exit_canceled_on_venue_clears_immediately"):
            mt._PENDING_STOP_EXIT.clear()
            inst = FakeInstrument(
                price=94.0,
                free=4.0,
                fills=[_fill("BUY", 100.0, 4.0)],
            )
            inst._provider.order_by_client_id = lambda _sym, _cid: {"orderId": "999", "status": "CANCELED"}
            mt._PENDING_STOP_EXIT[SYMBOL] = {
                "ts": NOW - 5,
                "client_order_id": "MTSL_CHARUSD_old",
                "qty": 4.0,
            }
            self.assertFalse(mt._is_pending_exit_active(SYMBOL, inst, 4.0, 0.1, NOW))
            self.assertNotIn(SYMBOL, mt._PENDING_STOP_EXIT)

        with self.subTest(msg="trades_monitor_warmup_retries_failed_symbol_without_spam"):
            mt.reset_trades_monitor_state_for_test()
            call_counts = {"SYM1": 0, "SYM2": 0}

            def _fake_get_trade_orders(_client, symbol, _maxage):
                call_counts[symbol] = call_counts.get(symbol, 0) + 1
                if symbol == "SYM2" and call_counts[symbol] == 1:
                    raise RuntimeError("transient API timeout")
                return [{"id": f"{symbol}_hist_1", "orderId": f"ord_{symbol}", "side": "BUY", "price": "100", "qty": "1", "quoteQty": "100", "time": 1000}]

            with (
                patch.object(mt.apitrades, "get_trade_orders", side_effect=_fake_get_trade_orders),
                patch("alertnotifiers.notify") as mock_notify,
            ):
                # Run 1: SYM1 warms up, SYM2 fails
                mt.check_and_notify_executed_trades(symbols_list=["SYM1", "SYM2"])
                self.assertTrue(mt._trades_monitor_initialized)
                self.assertIn("SYM2", mt._trades_monitor_warmup_failed_symbols)
                mock_notify.assert_not_called()

                # Run 2: SYM2 succeeds warmup without sending historical fill notification
                mt.check_and_notify_executed_trades(symbols_list=["SYM1", "SYM2"])
                self.assertNotIn("SYM2", mt._trades_monitor_warmup_failed_symbols)
                self.assertIn("SYM2_hist_1", mt._notified_trade_ids)
                mock_notify.assert_not_called()

        with self.subTest(msg="post_stop_ai_guard_scales_policy_capped_qty_and_defers_below_min_notional"):
            from types import SimpleNamespace
            snap = mt.ReentrySnapshot(
                symbol="ARBUSDC",
                current_price=0.18,
                last_sell_price=0.19,
                last_exit_kind="STOP",
                post_sell_peak=0.19,
                post_sell_trough=0.176,
                current_time_s=NOW,
                last_sell_time_s=NOW - 3600,
                trend_up=True,
                trend_down=False,
                vol_1h=0.01,
                vol_source="1h_cache",
            )

            class _LivePolicyProvider:
                name = "binance"

                @staticmethod
                def free_balance(_asset):
                    return 1000.0

                @staticmethod
                def quantity_decision(_sym, _side, _px, _req_qty, **_kw):
                    # 24h weight cap limits 250 USDC (~1388.88 ARB) down to 406.0 ARB (~73.08 USDC)
                    return SimpleNamespace(final_qty=406.0, refuse_reason=None, balance_asset="USDC")

                @staticmethod
                def round_amount(_sym, q):
                    return round(q, 1)

                @staticmethod
                def order_filter_refusal(_sym, _side, px, q, **_kw):
                    return "below_min_notional" if (px * q) < 50.0 else None

            live_inst = SimpleNamespace(
                symbol="ARBUSDC",
                base="ARB",
                quote="USDC",
                provider_name="binance",
                provider=_LivePolicyProvider(),
                _provider=_LivePolicyProvider(),
                min_qty=lambda: 1.0,
            )
            fake_guard = SimpleNamespace(
                check=lambda **_kw: SimpleNamespace(
                    allowed=True, suggested_scale=0.5, reason="weak bounce half size"
                )
            )
            with (
                patch.object(mt, "_is_simulated_env_or_inst", return_value=False),
                patch(
                    "intelligence.sentiment.guards.post_stop_reentry_guard.get_post_stop_reentry_guard",
                    return_value=fake_guard,
                ),
            ):
                # Below last_sell (0.18 < 0.19) -> does not consult AI guard, allows immediately
                allowed_below, _, _, _ = mt._apply_post_stop_ai_guard(
                    "ARBUSDC", 0.18, 0.19, NOW - 3600, snap, 250.0, None, 1.0, inst=live_inst
                )
                self.assertTrue(allowed_below)
                # Above last_sell (0.18 >= 0.178): 406.0 ARB * 0.5 = 203.0 ARB ($36.54 < $50 min_notional) -> must defer (allowed=False)
                allowed, bb, bq, q = mt._apply_post_stop_ai_guard(
                    "ARBUSDC", 0.18, 0.178, NOW - 3600, snap, 250.0, None, 1.0, inst=live_inst
                )
                self.assertFalse(allowed)

        with self.subTest(msg="pending_exit_timeout_restores_tp_and_sl_state_for_immediate_retry"):
            sym = SYMBOL
            mt._PENDING_STOP_EXIT.clear()
            mt._TP_RIDE_STATE.clear()
            mt._SL_BOUNCE_STATE.clear()
            saved_tp = {"started_at": NOW - 300, "peak_price": 115.0, "peak_gain": 0.15, "last_peak_time": NOW - 300, "buy_price": 100.0}
            mt._record_pending_exit(
                sym, None, "TP", NOW - 70, 5.0, 104.0,
                {"reason": "timeout", "client_order_id": "cid_tp_1"},
                default_intent_id="intent_tp_1",
                prev_tp_state=saved_tp,
            )
            inst_retry = FakeInstrument(price=104.0, free=5.0, fills=[_fill("BUY", 100, 5)])
            # At NOW (70s >= 60s timeout), _is_pending_exit_active must return False and restore _TP_RIDE_STATE
            active = mt._is_pending_exit_active(sym, inst_retry, 5.0, 0.001, NOW, current_price=104.0)
            self.assertFalse(active)
            self.assertIn(sym, mt._TP_RIDE_STATE)
            self.assertEqual(mt._TP_RIDE_STATE[sym]["peak_price"], 115.0)
            mt._PENDING_STOP_EXIT.clear()
            mt._TP_RIDE_STATE.clear()

        with self.subTest(msg="cancel_and_restore_resting_tp_orders_respect_cancel_result_and_remaining_qty"):
            sym = SYMBOL
            cancelled_ids = []
            resolved_ids = []
            placed_calls = []

            class _FakeTPInst:
                @staticmethod
                def open_orders():
                    return [
                        {"orderId": "tp_ok", "side": "SELL", "price": "110.0", "origQty": "10.0", "executedQty": "4.0", "remainingQty": "6.0"},
                        {"orderId": "tp_fail", "side": "SELL", "price": "112.0", "origQty": "5.0"},
                    ]

                @staticmethod
                def cancel_order(oid):
                    cancelled_ids.append(oid)
                    return False if oid == "tp_fail" else {"orderId": oid, "status": "CANCELED"}

                @staticmethod
                def place(side, px, q, **kw):
                    placed_calls.append((side, px, q, kw))
                    return {"orderId": "tp_restored"}

            import order_retry
            with patch.object(order_retry, "resolve_accepted_order", side_effect=lambda s, o: resolved_ids.append((s, o)) or True):
                cancelled = mt._cancel_resting_tp_orders(_FakeTPInst(), sym)
            self.assertEqual(len(cancelled), 1)
            self.assertEqual(cancelled[0]["orderId"], "tp_ok")
            self.assertEqual(resolved_ids, [(sym, "tp_ok")])

            mt._restore_resting_tp_orders(_FakeTPInst(), sym, cancelled, 6.0, "refused")
            self.assertEqual(len(placed_calls), 1)
            self.assertEqual(placed_calls[0][:3], ("SELL", 110.0, 6.0))
            self.assertEqual(placed_calls[0][3].get("kind"), "TP")

        with self.subTest(msg="timed_out_pending_exit_cleared_when_tp_decision_is_hold"):
            sym = SYMBOL
            mt._PENDING_STOP_EXIT.clear()
            mt._TP_RIDE_STATE.clear()
            mt._record_pending_exit(
                sym, None, "TP", NOW - 75, 5.0, 108.0,
                {"reason": "timeout", "client_order_id": "cid_tp_hold"},
                default_intent_id="intent_tp_hold",
            )
            inst_hold = FakeInstrument(
                price=108.0,
                free=5.0,
                fills=[_fill("BUY", 100.0, 5.0)],
            )
            with patch.object(mt, "is_trend_up", return_value=True):
                mt.monitor_price_and_trade(inst_hold, 3600, 86400, 0.05, 0.03, now_fn=lambda: NOW)
            self.assertNotIn(sym, mt._PENDING_STOP_EXIT)
            self.assertIn(sym, mt._TP_RIDE_STATE)
            mt._TP_RIDE_STATE.clear()

        with self.subTest(msg="forced_take_profit_disables_wait_for_trend_and_stale_sl_bounce_buy_price_not_logged"):
            sym = SYMBOL
            snap_tp = mt.PositionSnapshot(
                symbol=sym,
                current_price=108.0,
                avg_buy_price=100.0,
                sellable_qty=4.0,
                current_time_s=NOW,
                trend_up=True,
            )
            inst_tp = FakeInstrument(price=108.0, free=4.0, fills=[_fill("BUY", 100.0, 4.0)])
            forced_dec = mt.ExitDecision(
                action=mt.ExitAction.SELL,
                should_exit=True,
                reason="mdhtp_peak_pullback",
                forced=True,
            )
            sold = mt._execute_take_profit(inst_tp, snap_tp, forced_dec, 3600, 0.001, {}, 0.05)
            self.assertTrue(sold)
            self.assertEqual(len(inst_tp.placed), 1)
            self.assertTrue(inst_tp.placed[0]["kwargs"].get("wait_for_trend", True))

            mt._SL_BOUNCE_STATE[sym] = {"buy_price": 120.0, "started_at": NOW - 30}
            printed_lines = []
            with patch("builtins.print", side_effect=lambda *a, **k: printed_lines.append(" ".join(str(x) for x in a))):
                mt._handle_position_recovery(inst_tp, snap_tp, 0.03)
            self.assertNotIn(sym, mt._SL_BOUNCE_STATE)
            self.assertFalse(any("Stop-Loss bounce successfully recovered" in line for line in printed_lines))

        with self.subTest(msg="ai_or_partial_profit_sell_not_misclassified_as_stop_and_no_double_reentry_buy"):
            sym = SYMBOL
            mt.clear_position_stats_cache()
            # Multi-tranche buys: 3.16 @ 262.10, then dip buy 1.54 @ 246.30 (avg ~ 256.92),
            # followed by an AI profit trim SELL of 1.50 @ 251.14 (above 246.30 last buy, below 256.92 avg).
            fills_ai_trim = [
                {
                    "side": "SELL",
                    "price": 251.14,
                    "qty": 1.50,
                    "timestamp": (NOW - 600) * 1000,
                    "clientOrderId": "AI_intent_1792941034_sell_tao",
                },
                {"side": "BUY", "price": 246.30, "qty": 1.54, "timestamp": (NOW - 3600) * 1000},
                {"side": "BUY", "price": 262.10, "qty": 3.16, "timestamp": (NOW - 7200) * 1000},
            ]
            inst_trim = FakeInstrument(
                price=251.0,
                free=3.20,
                fills=fills_ai_trim,
                params={"mt.buy_budget": 200, "mt.max_budget": 2000},
            )
            self.run_tick(inst_trim, trend_up=True, gain=0.05, loss=0.03, now=NOW)
            self.assertNotEqual(mt._LAST_EXIT_KIND.get(sym), "STOP")
            self.assertEqual(inst_trim.placed, [])

            # Genuine STOP exit followed by re-entry BUY must record _REENTRY_DONE_SELL_TIME
            # and prevent a second re-entry BUY for the same sell_time on the next tick.
            mt.clear_position_stats_cache()
            stop_sell_ts = NOW - 3600
            fills_stop = [
                {"side": "SELL", "price": 94.0, "qty": 4.0, "timestamp": stop_sell_ts * 1000},
                {"side": "BUY", "price": 100.0, "qty": 4.0, "timestamp": (NOW - 7200) * 1000},
            ]
            mt._LAST_EXIT_KIND[sym] = "STOP"
            mt._REENTRY_STATE[sym] = {
                "trough": 90.0,
                "peak": 94.0,
                "peak_bounce": 92.0,
                "started_at": NOW - 600,
                "sell_time": float(stop_sell_ts),
            }
            inst_reentry = FakeInstrument(
                price=92.0,
                free=0.0,
                fills=fills_stop,
                params={"mt.buy_budget": 200, "mt.max_budget": 1000},
            )
            self.run_tick(inst_reentry, trend_up=True, gain=0.05, loss=0.03, now=NOW)
            self.assertEqual(len(inst_reentry.placed), 1)
            self.assertEqual(inst_reentry.placed[0]["side"], "BUY")
            self.assertEqual(mt._REENTRY_DONE_SELL_TIME.get(sym), float(stop_sell_ts))
            # Second tick with same fills (resting limit BUY not yet filled in trade history):
            self.run_tick(inst_reentry, trend_up=True, gain=0.05, loss=0.03, now=NOW + 60)
            self.assertEqual(len(inst_reentry.placed), 1)

            # Verify sub-sell re-entry clamps order_price when current_price + _buy_offset >= sell_price:
            inst_clamp = FakeInstrument(
                price=99.5,
                free=0.0,
                fills=[],
                params={"mt.buy_price_offset": 1.0},
            )
            placed_clamp = mt._execute_reentry_buy(
                inst_clamp, sym, 99.5, 2.0, 0.001, 60.0,
                "STOP", {"sell_time": float(stop_sell_ts)},
                sell_price=100.0,
            )
            self.assertTrue(placed_clamp)
            self.assertEqual(len(inst_clamp.placed), 1)
            self.assertAlmostEqual(inst_clamp.placed[0]["price"], 99.5)
            self.assertEqual(inst_clamp.placed[0]["kwargs"].get("bypass_guards"), {"all"})



if __name__ == "__main__":
    unittest.main(verbosity=2)

