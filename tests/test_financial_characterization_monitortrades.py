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
        with patch.object(mt, "is_trend_up", return_value=trend_up):
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
            self.assertTrue(order["kwargs"]["bypass_profit_guard"])
            self.assertEqual(order["kwargs"]["bypass_guards"], {"all"})

        # Subtest 3: Disaster floor breach, trend_up=False -> EXECUTE
        with self.subTest(msg="disaster_floor_executes_when_trend_not_up"):
            inst = FakeInstrument(price=96.0, free=4, fills=[_fill("BUY", 100, 4)])
            self.run_tick(inst, trend_up=False, loss=0.025)
            self.assertEqual(len(inst.placed), 1)
            order = inst.placed[0]
            self.assertEqual((order["side"], order["qty"]), ("SELL", 4.0))
            self.assertTrue(order["kwargs"]["bypass_profit_guard"])
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
            self.assertTrue(order["kwargs"]["bypass_profit_guard"])
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
            self.assertTrue(order["kwargs"]["bypass_profit_guard"])
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

        # Tick 3 at t=230s: window expired (230s >= 220s), trend_up=True -> forces Stop-Loss!
        self.run_tick(inst, trend_up=True, loss=0.025, now=NOW + 230)
        self.assertEqual(len(inst.placed), 1)
        self.assertEqual((inst.placed[0]["side"], inst.placed[0]["qty"]), ("SELL", 4.0))
        self.assertNotIn(sym, mt._SL_BOUNCE_STATE)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
