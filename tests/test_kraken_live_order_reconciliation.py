"""Invariants for reconciling real Kraken orders.

Kraken reports execution cumulatively through QueryOrders. The live engine must
apply only the new delta and keep the order locally until the exchange marks it
raporteaza terminal (closed/canceled/expired).
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_engine as strat  # noqa: E402
from providers.strategy_executor import OrderStatus, PairPrecision, ProviderError  # noqa: E402
from botcore import parse_dotenv  # noqa: E402

_CONFIG_ENV = parse_dotenv(os.path.join(ROOT, "kraken", "config.env"))


def _params(**overrides):
    values = dict(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=2.0,
        takeprofit_pct=5.0, max_budget=1000.0, max_dca_buys=10,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2,
        reentry_tolerance_pct=0.0, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )
    values.update(overrides)
    return strat.StratParams(**values)


def _strategy(*, state=None, **param_overrides):
    client = MagicMock()
    client.pair_precision.return_value = PairPrecision(2, 6, 0.01, "TEST")
    engine = strat.Strategy(
        client, "TESTPAIR_LIVE_RECONCILE", _params(**param_overrides),
        dry_run=False, initial_state=state or strat._new_state(),
    )
    engine._save = lambda: None
    return engine


def _buy_order(kind="ENTRY"):
    return {
        "txid": "BUY-1", "side": "buy", "vol": 1.0, "price": 100.0,
        "amount": 100.0, "kind": kind, "market": False, "ts": 0.0,
    }


class KrakenLiveReconciliationTest(unittest.TestCase):
    def test_open_partial_fill_applies_only_new_cumulative_delta(self):
        state = strat._new_state()
        state["orders"].append(_buy_order())
        engine = _strategy(state=state)
        engine.client.order_status.return_value = OrderStatus(
            "open", filled_qty=0.4, cost=40.0, fee=0.10,
        )

        with patch.object(strat, "notify"):
            engine.reconcile(101.0)
            engine.reconcile(101.0)  # the same cumulative reply is not reapplied

        self.assertAlmostEqual(engine.s["qty"], 0.4)
        self.assertAlmostEqual(engine.s["cost"], 40.0)
        self.assertAlmostEqual(engine.s["spent"], 40.0)
        self.assertAlmostEqual(engine.s["fees_total"], 0.10)
        order = engine._find_open("buy")
        self.assertIsNotNone(order)
        self.assertAlmostEqual(order["applied_vol"], 0.4)

        engine.client.order_status.return_value = OrderStatus(
            "open", filled_qty=0.7, cost=71.0, fee=0.18,
        )
        with patch.object(strat, "notify"):
            engine.reconcile(102.0)

        self.assertAlmostEqual(engine.s["qty"], 0.7)
        self.assertAlmostEqual(engine.s["cost"], 71.0)
        self.assertAlmostEqual(engine.s["spent"], 70.0)
        self.assertAlmostEqual(engine.s["fees_total"], 0.18)

    def test_canceled_partial_buy_is_accounted_then_removed_once(self):
        state = strat._new_state()
        state["qty"] = 1.0
        state["cost"] = 100.0
        state["spent"] = 100.0
        state["last_buy_price"] = 100.0
        state["orders"].append(_buy_order(kind="DCA"))
        engine = _strategy(state=state)
        engine.client.order_status.return_value = OrderStatus(
            "canceled", filled_qty=0.25, cost=24.5, fee=0.07,
        )

        with patch.object(strat, "notify"):
            engine.reconcile(98.0)

        self.assertFalse(engine._has_open("buy"))
        self.assertAlmostEqual(engine.s["qty"], 1.25)
        self.assertAlmostEqual(engine.s["cost"], 124.5)
        self.assertAlmostEqual(engine.s["spent"], 125.0)
        self.assertEqual(engine.s["dca_buys"], 1)
        self.assertAlmostEqual(engine.s["fees_total"], 0.07)

    def test_negative_maker_fee_is_accounted_as_rebate(self):
        state = strat._new_state()
        state["orders"].append(_buy_order())
        engine = _strategy(state=state)
        engine.client.order_status.return_value = OrderStatus(
            "closed", filled_qty=1.0, cost=100.0, fee=-0.02,
        )

        with patch.object(strat, "notify"):
            engine.reconcile(100.0)

        self.assertAlmostEqual(engine.s["fees_total"], -0.02)
        self.assertAlmostEqual(engine.s["realized_net"], 0.02)

    def test_incremental_sell_closes_cycle_only_when_order_is_terminal(self):
        state = strat._new_state()
        state.update(
            qty=1.0, cost=100.0, spent=100.0,
            entry_price=100.0, last_buy_price=100.0,
        )
        state["orders"].append({
            "txid": "SELL-1", "side": "sell", "vol": 1.0,
            "price": 110.0, "amount": 0.0, "kind": "TP",
            "market": False, "ts": 0.0,
        })
        engine = _strategy(state=state)
        engine.client.order_status.return_value = OrderStatus(
            "open", filled_qty=0.4, cost=44.0, fee=0.10,
        )

        with patch.object(strat, "notify"):
            engine.reconcile(110.0)

        self.assertEqual(engine.s["cycle"], 1)
        self.assertAlmostEqual(engine.s["qty"], 0.6)
        self.assertAlmostEqual(engine.s["cost"], 60.0)
        self.assertTrue(engine._has_open("sell"))

        engine.client.order_status.return_value = OrderStatus(
            "closed", filled_qty=1.0, cost=112.0, fee=0.28,
        )
        with patch.object(strat, "notify"):
            engine.reconcile(113.0)

        self.assertEqual(engine.s["cycle"], 2)
        self.assertEqual(engine.s["qty"], 0.0)
        self.assertFalse(engine._has_open("sell"))
        self.assertAlmostEqual(engine.s["realized_gross"], 12.0)
        self.assertAlmostEqual(engine.s["realized_net"], 11.72)

    def test_cancel_failure_keeps_order_tracked(self):
        state = strat._new_state()
        state["orders"].append(_buy_order())
        engine = _strategy(state=state)
        engine.client.cancel_order.side_effect = ProviderError("timeout")

        self.assertFalse(engine._cancel_open("buy"))
        self.assertIsNotNone(engine._find_open("buy"))
        self.assertNotIn("cancel_requested", engine._find_open("buy"))

    def test_accepted_cancel_waits_for_terminal_exchange_status(self):
        state = strat._new_state()
        state["orders"].append(_buy_order())
        engine = _strategy(state=state)
        engine.client.cancel_order.return_value = None

        self.assertTrue(engine._cancel_open("buy"))
        self.assertTrue(engine._find_open("buy")["cancel_requested"])

        engine.client.order_status.return_value = OrderStatus(
            "canceled", filled_qty=0.0, cost=0.0, fee=0.0,
        )
        engine.reconcile(101.0)
        self.assertFalse(engine._has_open("buy"))

    def test_stop_does_not_create_conflicting_exit_when_cancel_fails(self):
        state = strat._new_state()
        state.update(
            qty=1.0, cost=100.0, spent=100.0,
            entry_price=100.0, last_buy_price=100.0,
        )
        state["orders"].append(_buy_order(kind="DCA"))
        engine = _strategy(state=state, stop_loss_pct=10.0)
        engine.client.cancel_order.side_effect = ProviderError("timeout")

        with patch.object(strat, "notify"):
            engine.step(80.0)

        engine.client.submit_order.assert_not_called()
        self.assertIsNotNone(engine._find_open("buy"))

    def test_failed_stop_order_is_not_announced_as_executing(self):
        state = strat._new_state()
        state.update(
            qty=1.0, cost=100.0, spent=100.0,
            entry_price=100.0, last_buy_price=100.0,
        )
        engine = _strategy(state=state, stop_loss_pct=10.0)
        engine.client.submit_order.side_effect = ProviderError("insufficient funds")

        with patch.object(strat, "notify") as notify_mock:
            engine.step(80.0)

        notify_mock.assert_not_called()
        self.assertFalse(engine._has_pending_market_exit())


class KrakenConfigParsingTest(unittest.TestCase):
    def test_explicit_zero_disables_stop_reentry_bounce(self):
        env = {**_CONFIG_ENV, "STRAT_REENTRY_SL_BOUNCE_PCT": "0"}
        with patch.dict(os.environ, env, clear=True):
            params = strat.StratParams.from_env()
        self.assertEqual(params.reentry_sl_bounce_pct, 0.0)

    def test_trailing_profit_floor_is_explicit_and_clamped_non_negative(self):
        with patch.dict(os.environ, {
            **_CONFIG_ENV, "STRAT_TP_TRAIL_PROFIT_FLOOR_PCT": "1.0",
        }, clear=True):
            self.assertEqual(strat.StratParams.from_env().tp_trail_profit_floor_pct, 1.0)
        with patch.dict(os.environ, {
            **_CONFIG_ENV, "STRAT_TP_TRAIL_PROFIT_FLOOR_PCT": "-2.0",
        }, clear=True):
            self.assertEqual(strat.StratParams.from_env().tp_trail_profit_floor_pct, 0.0)


class KrakenStatePersistenceTest(unittest.TestCase):
    @staticmethod
    def _client():
        client = MagicMock()
        client.pair_precision.return_value = PairPrecision(2, 6, 0.01, "TEST")
        return client

    def test_corrupt_state_fails_closed_in_real_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "state.json")
            with open(state_file, "w", encoding="utf-8") as handle:
                handle.write('{"qty":')
            with patch.object(strat, "state_path_for", return_value=state_file):
                with self.assertRaisesRegex(RuntimeError, "stare.*invalida"):
                    strat.Strategy(self._client(), "PAIR", _params(), dry_run=False)

    def test_corrupt_state_may_reset_only_in_paper_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "state.json")
            with open(state_file, "w", encoding="utf-8") as handle:
                handle.write("not-json")
            with patch.object(strat, "state_path_for", return_value=state_file):
                engine = strat.Strategy(self._client(), "PAIR", _params(), dry_run=True)
        self.assertEqual(engine.s, strat._new_state())

    def test_save_replaces_state_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "state.json")
            engine = strat.Strategy(
                self._client(), "PAIR", _params(), dry_run=False,
                initial_state=strat._new_state(),
            )
            engine.state_file = state_file
            engine.s["qty"] = 1.25
            engine._save()

            with open(state_file, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["qty"], 1.25)
            self.assertEqual(os.listdir(tmp), ["state.json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
