"""Trading 212 replay runs the live engine and preserves partial accounting."""

import importlib.util
import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T212_DIR = os.path.join(ROOT, "212trading")
sys.path.insert(0, T212_DIR)
sys.path.insert(0, ROOT)

from providers.execution_audit import ExecutionAudit  # noqa: E402

_COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
_PRELOADED = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
try:
    spec = importlib.util.spec_from_file_location(
        "t212_replay_under_test", os.path.join(T212_DIR, "replay.py"),
    )
    replay = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = replay
    spec.loader.exec_module(replay)
    strategy = replay._strat
finally:
    for name in _COLLIDING:
        sys.modules.pop(name, None)
    sys.modules.update(_PRELOADED)


def _params(**overrides):
    config = {
        "STRAT_CURRENCY": "USD",
        "YAHOO_SYMBOL": "TEST",
        "STRATEGY_MODE": "avg_tp",
        "STRAT_ENTRY_PCT": "20",
        "STRAT_DCA_PCT": "10",
        "STRAT_ENTRY_DISCOUNT_PCT": "0.2",
        "STRAT_DCA_DROP_PCT": "2",
        "STRAT_TAKEPROFIT_PCT": "3",
        "STRAT_MAX_DCA_BUYS": "3",
        "STRAT_MAX_BUDGET": "500",
        "STRAT_FX_FEE_PCT": "0.15",
        "STRAT_STOP_LOSS_PCT": "20",
        "STRAT_CHECK_MINUTES": "5",
        "STRAT_ORDER_TTL_MIN": "10",
        "STRAT_REENTRY_DROP_PCT": "0",
        "STRAT_REENTRY_TOLERANCE_PCT": "0",
        "STRAT_LOSS_ALERT_STEP": "1",
        "STRAT_LADDER_MIN_FREE": "6",
        "STRAT_SL_REBUY_ENABLED": "false",
        "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
        "STRAT_DCA_TREND_GATE_PCT": "0",
        "STRAT_TRAIL_PCT": "0",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
    }
    config.update({key: str(value) for key, value in overrides.items()})
    return strategy.StratParams.from_env(config)


class T212ReplayTest(unittest.TestCase):
    def test_replay_core_mechanics(self):
        with self.subTest(msg="partial_fill_remains_open_for_later_bars"):
            params = _params()
            bars = [(100, 101, 99, 100), (100, 101, 99, 100)]
            full = replay.run_replay(bars, params, bar_minutes=1440)
            partial = replay.run_replay(
                bars, params, bar_minutes=1440,
                execution=replay.ExecutionModel(partial_fill_ratio=0.5),
            )
            self.assertEqual(full["fills"], 1)
            self.assertEqual(partial["fills"], 1)
            self.assertAlmostEqual(partial["open_qty"], full["open_qty"] / 2)

        with self.subTest(msg="historical_fx_changes_position_sizing_at_decision_time"):
            params = _params(STRAT_CURRENCY="RON")
            bars = [(100, 101, 99, 100), (100, 101, 99, 100)]
            one_to_one = replay.run_replay(bars, params, bar_minutes=1440, fx_to_usd=1.0)
            historical = replay.run_replay(
                bars, params, bar_minutes=1440, fx_to_usd=[0.5, 0.5],
            )
            self.assertAlmostEqual(historical["open_qty"], one_to_one["open_qty"] / 2)
            self.assertEqual(historical["account_currency"], "RON")

        with self.subTest(msg="worst_case_reports_ambiguous_buy_and_sell_paths"):
            params = _params()
            bars = [
                (100, 101, 99, 100),   # decide ENTRY
                (100, 101, 99, 100),   # fill ENTRY, decide TP
                (97, 101, 96.9, 97),   # decides DCA; the TP stays open
                (100, 104, 96, 100),   # hits both the DCA BUY and the TP SELL
            ]
            result = replay.run_replay(
                bars, params, bar_minutes=1440,
                execution=replay.ExecutionModel(intrabar_policy="worst_case"),
            )
            self.assertGreaterEqual(result["ambiguous_bars"], 1)
            self.assertIn(result["intrabar_policy_selected"], {"buy_first", "sell_first"})
            scenarios = result["intrabar_scenarios"]
            self.assertNotEqual(
                scenarios["buy_first"]["return_pct"],
                scenarios["sell_first"]["return_pct"],
            )

        with self.subTest(msg="order_decided_at_close_fills_only_in_next_bar"):
            params = _params()
            first = (100.0, 200.0, 1.0, 100.0)
            one = replay.run_replay([first], params, bar_minutes=1440)
            self.assertEqual(one["fills"], 0)
            self.assertEqual(one["open_qty"], 0.0)

            second = (100.0, 101.0, 99.0, 100.0)
            two = replay.run_replay([first, second], params, bar_minutes=1440)
            self.assertEqual(two["fills"], 1)
            self.assertGreater(two["open_qty"], 0.0)
            self.assertAlmostEqual(two["net_pnl"], two["total"])

        with self.subTest(msg="market_stop_fills_at_next_open_and_slippage_is_adverse"):
            params = _params(STRAT_STOP_LOSS_PCT="20")
            bars = [
                (100, 101, 99, 100),  # decide ENTRY
                (100, 101, 99, 100),  # fill ENTRY
                (70, 71, 69, 70),     # decide STOP MARKET la close
                (65, 66, 64, 65),     # A STOP fill at the open, right after the gap.
            ]
            base = replay.run_replay(bars, params, bar_minutes=1440)
            stressed = replay.run_replay(
                bars, params, bar_minutes=1440,
                execution=replay.ExecutionModel(market_slippage_bps=100),
            )
            self.assertEqual(base["open_qty"], 0.0)
            self.assertEqual(base["cycles"], 1)
            self.assertLess(stressed["total"], base["total"])

        with self.subTest(msg="partial_fill_worst_case_handles_fractional_dust"):
            params = _params(STRAT_ENTRY="1", STRAT_TAKEPROFIT_PCT="1")
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 102.0, 99.0, 101.0),
                (101.0, 103.0, 100.0, 102.0),
            ]

            result = replay.run_replay(
                bars, params, bar_minutes=1440,
                execution=replay.ExecutionModel(
                    partial_fill_ratio=0.75, intrabar_policy="worst_case",
                ),
            )

            self.assertGreaterEqual(result["fills"], 1)

        with self.subTest(msg="trend_gate_refuses_wrong_cadence"):
            params = _params(STRAT_DCA_TREND_GATE_PCT="0.1")
            with self.assertRaisesRegex(ValueError, "5-minute bars"):
                replay.run_replay([(100, 101, 99, 100)], params, bar_minutes=1440)

    def test_strategy_engine_direct_actions(self):
        with self.subTest(msg="partial_sell_reduces_remaining_cost_basis"):
            params = _params()
            engine = strategy.Strategy(
                MagicMock(), "TEST_US_EQ", params, dry_run=True,
                initial_state=strategy._new_state(), fx_to_usd=1.0,
            )
            engine.s.update({"qty": 2.0, "cost_usd": 200.0, "spent_cash": 200.0})
            strategy.notify = lambda **_kwargs: None
            engine._apply_fill(
                {"side": "SELL", "kind": "TP", "qty": 1.0, "limit": 110.0},
                1.0, 110.0,
            )
            self.assertEqual(engine.s["qty"], 1.0)
            self.assertAlmostEqual(engine.s["cost_usd"], 100.0)
            self.assertAlmostEqual(engine._avg_cost(), 100.0)

        with self.subTest(msg="dust_sell_does_not_create_zero_quantity_order"):
            params = _params()
            engine = strategy.Strategy(
                MagicMock(), "TEST_US_EQ", params, dry_run=True,
                initial_state=strategy._new_state(), fx_to_usd=1.0,
            )

            self.assertFalse(engine._place_sell(0.004, 110.0))
            self.assertEqual(engine.s["orders"], [])

        with self.subTest(msg="paper_stop_arms_same_rebuy_as_real_path"):
            params = _params(STRAT_SL_REBUY_ENABLED="true", STRAT_SL_REBUY_BOUNCE_PCT="1.2")
            engine = strategy.Strategy(
                MagicMock(), "TEST_US_EQ", params, dry_run=True,
                initial_state=strategy._new_state(), fx_to_usd=1.0,
            )
            engine.s.update({
                "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
                "sl_pending": True,
            })
            strategy.notify = lambda **_kwargs: None
            engine._apply_fill(
                {"side": "SELL", "kind": "SL", "qty": 1.0, "limit": 70.0},
                1.0, 70.0,
            )
            self.assertEqual(engine.s["last_sell_price"], 70.0)
            self.assertEqual(engine.s["sl_rebuy"], {"low": 70.0, "sell_price": 70.0})


class T212StaleBuyTest(unittest.TestCase):
    """One TTL rule for a resting BUY, shared by live reconciliation and replay."""

    def _engine(self, **overrides):
        now = [1_000_000.0]
        engine = strategy.Strategy(
            MagicMock(), "TEST_US_EQ", _params(**overrides), dry_run=True,
            initial_state=strategy._new_state(), fx_to_usd=1.0, clock=lambda: now[0],
        )
        return engine, now

    def test_buy_is_stale_compares_the_order_a_fresh_placement_would_make(self):
        engine, now = self._engine(STRAT_ENTRY_DISCOUNT_PCT="5", STRAT_ORDER_TTL_MIN="3")
        order = {"side": "BUY", "limit": 95.0, "ts": now[0]}
        now[0] += 4 * 60
        # Unchanged price: a new order would be the same 95.00, so nothing to replace.
        # The old rule (price > limit * 1.003) re-placed it on every TTL.
        self.assertFalse(engine._buy_is_stale(order, 100.0))
        self.assertTrue(engine._buy_is_stale(order, 100.5))
        with self.subTest(msg="younger_than_ttl"):
            order_young = {"side": "BUY", "limit": 95.0, "ts": now[0] - 60}
            self.assertFalse(engine._buy_is_stale(order_young, 110.0))
        with self.subTest(msg="sells_and_market_orders_are_never_stale_buys"):
            self.assertFalse(engine._buy_is_stale({"side": "SELL", "limit": 95.0, "ts": 0}, 110.0))
            self.assertFalse(engine._buy_is_stale(
                {"side": "BUY", "limit": 95.0, "ts": 0, "market": True}, 110.0))

    def test_replay_replaces_an_entry_left_below_a_rally(self):
        # The entry sits 0.2% under 100; the next bars rally without touching it, then
        # dip to 110.5. Live re-places the stale entry near the price, so it must fill.
        bars = [(100.0, 100.5, 99.9, 100.0)]
        bars += [(p, p * 1.004, p * 0.9995, p) for p in (103.0, 106.0, 109.0, 111.0)]
        bars += [(111.0, 111.2, 110.5, 111.0)]
        result = replay.run_replay(bars, _params(), bar_minutes=1440)
        self.assertGreater(result["open_qty"], 0.0)

    def test_log_lines_carry_the_asset_thread_name(self):
        import threading
        lines = []
        with patch.object(strategy, "_log", lines.append):
            strategy.log("main")
            worker = threading.Thread(target=strategy.log, args=("tick",), name="nvda")
            worker.start()
            worker.join()
        self.assertEqual(lines, ["main", "[nvda] tick"])


class T212StatePersistenceTest(unittest.TestCase):
    @staticmethod
    def _client():
        return MagicMock()

    def test_state_persistence_behaviors(self):
        with self.subTest(msg="corrupt_state_fails_closed_live_but_may_reset_in_paper"):
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "state.json")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("not-json")
                with patch.object(strategy, "state_path_for", return_value=path):
                    with self.assertRaisesRegex(RuntimeError, "stare T212 invalida"):
                        strategy.Strategy(
                            self._client(), "TEST_US_EQ", _params(), dry_run=False,
                            fx_to_usd=1.0,
                        )
                    paper = strategy.Strategy(
                        self._client(), "TEST_US_EQ", _params(), dry_run=True,
                        fx_to_usd=1.0,
                    )
                self.assertEqual(paper.s, strategy._new_state())

        with self.subTest(msg="live_save_is_atomic"):
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "state.json")
                engine = strategy.Strategy(
                    self._client(), "TEST_US_EQ", _params(), dry_run=False,
                    initial_state=strategy._new_state(), fx_to_usd=1.0,
                )
                engine.state_file = path
                engine.s["qty"] = 1.25

                engine._save()

                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle)["qty"], 1.25)
                self.assertEqual(os.listdir(directory), ["state.json"])

        with self.subTest(msg="failed_save_marks_state_dirty_and_live_fails_closed"):
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "state.json")
                for dry_run in (True, False):
                    with self.subTest(dry_run=dry_run):
                        engine = strategy.Strategy(
                            self._client(), "TEST_US_EQ", _params(), dry_run=dry_run,
                            initial_state=strategy._new_state(), fx_to_usd=1.0,
                        )
                        engine.state_file = path
                        with patch(
                            "strategies.state_store.os.replace", side_effect=OSError("disk")
                        ):
                            if dry_run:
                                engine._save()
                            else:
                                with self.assertRaisesRegex(RuntimeError, "persisting the"):
                                    engine._save()
                        self.assertTrue(engine._state_write_failed)


class _FillClient:
    def __init__(self):
        self.portfolio = [{"ticker": "TEST_US_EQ", "quantity": 0.5, "averagePrice": 100.0}]
        self.active = [{"id": "SELL-1", "ticker": "TEST_US_EQ"}]
        self.status = {
            "id": "SELL-1", "ticker": "TEST_US_EQ", "status": "PARTIALLY_FILLED",
            "filledQuantity": 0.5, "filledValue": 55.0,
        }
        self.cancel_calls = []

    def get_portfolio(self):
        return self.portfolio

    def list_active_orders(self):
        return self.active

    def get_order_status(self, order_id):
        return self.status

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        return True


class T212ExactFillReconciliationTest(unittest.TestCase):
    def _engine(self, client, audit_dir):
        state = strategy._new_state()
        state.update({
            "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
            "entry_price": 100.0, "last_buy_price": 100.0,
            "orders": [{
                "id": "SELL-1", "side": "SELL", "qty": 1.0,
                "limit": 108.0, "kind": "TP", "level": None,
                "intent_id": "t212-test-sell", "ts": 0.0,
            }],
        })
        engine = strategy.Strategy(
            client, "TEST_US_EQ", _params(STRAT_FX_FEE_PCT="0"), dry_run=False,
            initial_state=state, fx_to_usd=1.0,
            execution_audit=ExecutionAudit(audit_dir),
        )
        engine._save = lambda: None
        return engine

    def test_reconciliation_behaviors(self):
        with self.subTest(msg="partial_and_terminal_sell_use_cumulative_fill_prices_not_poll_price"):
            client = _FillClient()
            with tempfile.TemporaryDirectory() as audit_dir:
                engine = self._engine(client, audit_dir)
                with patch.object(strategy, "notify"):
                    engine._reconcile_real(150.0)  # poll-ul e deliberat departe de fill-ul 110

                self.assertAlmostEqual(engine.s["qty"], 0.5)
                self.assertAlmostEqual(engine.s["realized_pnl_usd"], 5.0)
                self.assertAlmostEqual(engine.s["last_sell_price"], 110.0)
                self.assertAlmostEqual(engine.s["orders"][0]["applied_fill_qty"], 0.5)

                client.portfolio = [{"ticker": "TEST_US_EQ", "quantity": 0.0, "averagePrice": 0.0}]
                client.active = []
                client.status = {
                    "id": "SELL-1", "ticker": "TEST_US_EQ", "status": "FILLED",
                    "filledQuantity": 1.0, "filledValue": 112.0,
                }
                with patch.object(strategy, "notify"):
                    engine._reconcile_real(160.0)

                # The second half executed at 114; the gross total = 5 + 7, not at 150/160.
                self.assertAlmostEqual(engine.s["realized_pnl_usd"], 12.0)
                self.assertAlmostEqual(engine.s["last_sell_price"], 114.0)
                self.assertEqual(engine.s["qty"], 0.0)

        with self.subTest(msg="canceled_unfilled_ladder_order_is_not_marked_as_sold"):
            client = _FillClient()
            client.portfolio = [{"ticker": "TEST_US_EQ", "quantity": 1.0, "averagePrice": 100.0}]
            client.active = []
            client.status = {
                "id": "SELL-1", "ticker": "TEST_US_EQ", "status": "CANCELLED",
                "filledQuantity": 0.0, "filledValue": 0.0,
            }
            with tempfile.TemporaryDirectory() as audit_dir:
                engine = self._engine(client, audit_dir)
                engine.s["orders"][0]["level"] = 10.0
                with patch.object(strategy, "notify"):
                    engine._reconcile_real(100.0)
            self.assertEqual(engine.s["orders"], [])
            self.assertEqual(engine.s["tp_sold_levels"], [])

        with self.subTest(msg="partial_dca_counts_one_buy_across_multiple_reconciliations"):
            client = _FillClient()
            client.active = [{"id": "BUY-1", "ticker": "TEST_US_EQ"}]
            client.portfolio = [{
                "ticker": "TEST_US_EQ", "quantity": 1.5,
                "averagePrice": 145.0 / 1.5,
            }]
            client.status = {
                "id": "BUY-1", "ticker": "TEST_US_EQ", "status": "PARTIALLY_FILLED",
                "filledQuantity": 0.5, "filledValue": 45.0,
            }
            with tempfile.TemporaryDirectory() as audit_dir:
                engine = self._engine(client, audit_dir)
                engine.s["orders"] = [{
                    "id": "BUY-1", "side": "BUY", "qty": 1.0,
                    "limit": 90.0, "amount": 90.0, "kind": "DCA",
                    "intent_id": "t212-test-dca", "ts": 0.0,
                }]
                with patch.object(strategy, "notify"):
                    engine._reconcile_real(90.0)
                self.assertEqual(engine.s["dca_buys"], 1)

                client.portfolio = [{
                    "ticker": "TEST_US_EQ", "quantity": 1.75,
                    "averagePrice": 167.5 / 1.75,
                }]
                client.status.update(filledQuantity=0.75, filledValue=67.5)
                with patch.object(strategy, "notify"):
                    engine._reconcile_real(90.0)
            self.assertEqual(engine.s["dca_buys"], 1)

        with self.subTest(msg="fill_racing_with_accepted_cancel_is_still_reconciled_exactly"):
            client = _FillClient()
            client.portfolio = [{
                "ticker": "TEST_US_EQ", "quantity": 1.0, "averagePrice": 100.0,
            }]
            client.active = [{"id": "SELL-1", "ticker": "TEST_US_EQ"}]
            client.status = {
                "id": "SELL-1", "ticker": "TEST_US_EQ", "status": "CONFIRMED",
                "filledQuantity": 0.0, "filledValue": 0.0,
            }
            with tempfile.TemporaryDirectory() as audit_dir:
                engine = self._engine(client, audit_dir)
                order = engine.s["orders"][0]

                self.assertTrue(engine._cancel_specific(order))
                self.assertIn(order, engine.s["orders"])

                # One half executes in the very race with the cancellation. The terminal status
                # must be read before we forget the order, otherwise the P&L would use the poll price.
                client.portfolio = [{
                    "ticker": "TEST_US_EQ", "quantity": 0.5, "averagePrice": 100.0,
                }]
                client.active = []
                client.status = {
                    "id": "SELL-1", "ticker": "TEST_US_EQ", "status": "CANCELLED",
                    "filledQuantity": 0.5, "filledValue": 55.0,
                }
                with patch.object(strategy, "notify"):
                    engine._reconcile_real(150.0)

            self.assertEqual(engine.s["orders"], [])
            self.assertAlmostEqual(engine.s["qty"], 0.5)
            self.assertAlmostEqual(engine.s["realized_pnl_usd"], 5.0)
            self.assertAlmostEqual(engine.s["last_sell_price"], 110.0)


class _CancelClient:
    def __init__(self, cancel_result=False):
        self.cancel_result = cancel_result
        self.limit_result = None
        self.market_result = None
        self.cancel_calls = []
        self.place_calls = []
        self.portfolio = []
        self.active = []
        self.status = None

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        if isinstance(self.cancel_result, Exception):
            raise self.cancel_result
        return self.cancel_result

    def place_limit_order(self, ticker, quantity, limit, validity):
        self.place_calls.append(("limit", ticker, quantity, limit, validity))
        if self.limit_result is not None:
            return self.limit_result
        return 200, {"id": f"NEW-{len(self.place_calls)}"}

    def place_market_order(self, ticker, quantity, extended_hours=False):
        self.place_calls.append(("market", ticker, quantity, extended_hours))
        if self.market_result is not None:
            return self.market_result
        return 200, {"id": f"NEW-{len(self.place_calls)}"}

    def get_portfolio(self):
        return self.portfolio

    def list_active_orders(self):
        return self.active

    def get_order_status(self, order_id):
        return self.status


class T212CancellationLifecycleTest(unittest.TestCase):
    @staticmethod
    def _engine(client, **param_overrides):
        engine = strategy.Strategy(
            client, "TEST_US_EQ", _params(**param_overrides), dry_run=False,
            initial_state=strategy._new_state(), fx_to_usd=1.0,
        )
        engine._save = lambda: None
        return engine

    @staticmethod
    def _order(**overrides):
        order = {
            "id": "OLD-1", "side": "SELL", "qty": 1.0,
            "limit": 110.0, "kind": "TP", "ts": 0.0,
        }
        order.update(overrides)
        return order

    def test_failed_or_raised_cancel_keeps_order_tracked(self):
        for result in (False, RuntimeError("timeout")):
            with self.subTest(result=result):
                client = _CancelClient(result)
                engine = self._engine(client)
                order = self._order()
                engine.s["orders"] = [order]

                self.assertFalse(engine._cancel_specific(order))
                self.assertEqual(engine.s["orders"], [order])

    def test_ladder_does_not_replace_an_order_whose_cancel_failed(self):
        for old in (self._order(level=10.0), self._order()):
            with self.subTest(order=old):
                client = _CancelClient(False)
                engine = self._engine(client, STRAT_TP_LADDER="10:100")
                engine.s["orders"] = [old]

                engine._manage_tp_ladder(held=1.0, avg=100.0)

                self.assertEqual(engine.s["orders"], [old])
                self.assertEqual(client.place_calls, [])

    def test_stop_waits_for_confirmed_cancels_then_places_one_exit(self):
        client = _CancelClient(False)
        engine = self._engine(client)
        old = self._order(side="BUY", kind="DCA", limit=90.0)
        engine.s.update({
            "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
            "orders": [old],
        })

        with patch.object(strategy, "notify"):
            self.assertTrue(engine._check_stop_loss(70.0))
        self.assertEqual(engine.s["orders"], [old])
        self.assertEqual(client.place_calls, [])

        client.cancel_result = True
        with patch.object(strategy, "notify"):
            self.assertTrue(engine._check_stop_loss(70.0))
        self.assertEqual(len(client.place_calls), 1)
        self.assertEqual(client.place_calls[0][0], "market")
        self.assertEqual(len(engine.s["orders"]), 2)
        self.assertTrue(old["cancel_requested"])
        stop = next(o for o in engine.s["orders"] if o.get("kind") == "SL")
        self.assertTrue(stop["market"])

    def test_accepted_cancel_stays_tracked_and_is_not_submitted_twice(self):
        client = _CancelClient(True)
        engine = self._engine(client)
        order = self._order()
        engine.s["orders"] = [order]

        self.assertTrue(engine._cancel_specific(order))
        self.assertTrue(engine._cancel_specific(order))

        self.assertEqual(engine.s["orders"], [order])
        self.assertTrue(order["cancel_requested"])
        self.assertEqual(client.cancel_calls, ["OLD-1"])

    def test_ladder_waits_for_terminal_cancel_before_replacement(self):
        client = _CancelClient(True)
        engine = self._engine(client, STRAT_TP_LADDER="10:100")
        old = self._order(level=10.0, qty=0.5, limit=109.0)
        engine.s["orders"] = [old]

        engine._manage_tp_ladder(held=1.0, avg=100.0)

        self.assertTrue(old["cancel_requested"])
        self.assertEqual(engine.s["orders"], [old])
        self.assertEqual(client.place_calls, [])

    def test_live_submit_and_cancel_are_persisted_immediately(self):
        client = _CancelClient(True)
        engine = self._engine(client)
        engine._save = MagicMock()

        self.assertTrue(engine._place_sell(1.0, 110.0))
        self.assertEqual(engine._save.call_count, 2)  # pending pre-submit, then the order id

        engine._save.reset_mock()
        order = engine.s["orders"][0]
        self.assertTrue(engine._cancel_specific(order))
        engine._save.assert_called_once()

    def test_live_submit_is_durable_before_external_post(self):
        client = _CancelClient(True)
        engine = self._engine(client)
        snapshots = []

        def save_snapshot():
            snapshots.append(copy.deepcopy(engine.s))

        engine._save = save_snapshot
        self.assertTrue(engine._place_sell(1.0, 110.0))

        self.assertGreaterEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["pending_submit"]["side"], "SELL")
        self.assertEqual(snapshots[0]["orders"], [])
        self.assertIsNone(engine.s["pending_submit"])
        self.assertEqual(engine.s["orders"][0]["id"], "NEW-1")

    def test_ambiguous_submit_stays_pending_and_blocks_second_post(self):
        client = _CancelClient(True)
        client.market_result = (500, {"error": "gateway timeout"})
        engine = self._engine(client)

        self.assertTrue(engine._place_sell(1.0, 110.0, kind="SL", market=True))
        first_intent = dict(engine.s["pending_submit"])
        self.assertEqual(engine.s["orders"], [])

        self.assertFalse(engine._place_sell(1.0, 109.0, kind="SL", market=True))
        self.assertEqual(len(client.place_calls), 1)
        self.assertEqual(engine.s["pending_submit"]["intent_id"], first_intent["intent_id"])

    def test_response_lost_submit_is_recovered_from_unique_active_order(self):
        client = _CancelClient(True)
        client.limit_result = (500, {"error": "gateway timeout"})
        client.portfolio = [{
            "ticker": "TEST_US_EQ", "quantity": 1.0, "averagePrice": 100.0,
        }]
        engine = self._engine(client)
        engine.s.update({
            "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
        })
        self.assertTrue(engine._place_sell(1.0, 110.0))
        client.active = [{
            "id": "RECOVERED-1", "ticker": "TEST_US_EQ",
            "quantity": -1.0, "limitPrice": 110.0,
        }]

        with patch.object(strategy, "notify"):
            engine._reconcile_real(110.0)

        self.assertIsNone(engine.s["pending_submit"])
        self.assertEqual(engine.s["orders"][0]["id"], "RECOVERED-1")
        self.assertEqual(len(client.place_calls), 1)

    def test_two_confirmed_absences_release_ambiguous_submit_for_retry(self):
        client = _CancelClient(True)
        client.limit_result = (500, {"error": "gateway timeout"})
        engine = self._engine(client)
        engine._place_buy(100.0, 100.0, "ENTRY")

        self.assertEqual(engine._recover_pending_submit(0.0, []), "waiting")
        self.assertIsNotNone(engine.s["pending_submit"])
        self.assertEqual(engine._recover_pending_submit(0.0, []), "retryable")
        self.assertIsNone(engine.s["pending_submit"])

    def test_trailing_exit_is_market_and_is_not_replaced_while_pending(self):
        client = _CancelClient(True)
        engine = self._engine(
            client, STRAT_TRAIL_PCT="5", STRAT_TRAIL_MIN_PROFIT_PCT="0",
        )
        engine.s.update({
            "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
            "pos_peak": 120.0, "tr_armed": True,
        })

        with patch.object(strategy, "notify"):
            self.assertTrue(engine._check_trailing(110.0))
            self.assertTrue(engine._check_trailing(105.0))

        self.assertEqual(len(client.place_calls), 1)
        self.assertEqual(client.place_calls[0][0], "market")
        self.assertEqual(engine.s["orders"][0]["kind"], "TR")
        self.assertTrue(engine.s["orders"][0]["market"])

    def test_rejected_market_exit_does_not_arm_rebuy_or_claim_success(self):
        for check, overrides, state in (
            ("_check_stop_loss", {}, {}),
            (
                "_check_trailing",
                {"STRAT_TRAIL_PCT": "5", "STRAT_TRAIL_MIN_PROFIT_PCT": "0"},
                {"pos_peak": 120.0, "tr_armed": True},
            ),
        ):
            with self.subTest(check=check):
                client = _CancelClient(True)
                client.market_result = (500, {"error": "rejected"})
                engine = self._engine(client, **overrides)
                engine.s.update({
                    "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
                    **state,
                })

                with patch.object(strategy, "notify") as notify:
                    self.assertTrue(getattr(engine, check)(70.0))

                self.assertFalse(engine.s.get("sl_pending", False))
                self.assertEqual(engine.s["orders"], [])
                notify.assert_not_called()

    def test_ambiguous_not_owned_error_never_erases_local_position(self):
        client = _CancelClient(True)
        client.limit_result = (400, {"code": "selling-equity-not-owned"})
        engine = self._engine(client)
        engine.s.update({
            "qty": 1.0, "cost_usd": 100.0, "spent_cash": 100.0,
        })

        self.assertFalse(engine._place_sell(1.0, 110.0))

        self.assertEqual(engine.s["qty"], 1.0)
        self.assertEqual(engine.s["cost_usd"], 100.0)
        self.assertEqual(engine.s["spent_cash"], 100.0)

    def test_limit_orders_keep_the_profile_validity(self):
        client = _CancelClient(True)
        engine = self._engine(client)

        self.assertTrue(engine._place_sell(1.0, 110.0))

        self.assertEqual(client.place_calls[-1][-1], "GOOD_TILL_CANCEL")


if __name__ == "__main__":
    unittest.main()
