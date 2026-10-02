"""Characterisation for moving the spot engine out of the venue into the neutral package."""

import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from strategies import spot_engine, spot_dca  # noqa: E402
from providers.strategy_executor import PairPrecision  # noqa: E402


class _Executor:
    def pair_precision(self, _symbol):
        return PairPrecision(2, 4, 0.01, "ASSET")


def _params():
    return spot_engine.StratParams(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=2.0,
        takeprofit_pct=5.0, max_budget=1000.0, max_dca_buys=10,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2,
        reentry_tolerance_pct=0.0, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )


class SpotModuleBoundaryTest(unittest.TestCase):
    def test_spot_dca_shim_matches_spot_engine_canonical(self):
        self.assertIs(spot_dca.Strategy, spot_engine.Strategy)
        self.assertIs(spot_dca.StratParams, spot_engine.StratParams)
        self.assertEqual(spot_dca._new_state(), spot_engine._new_state())
        self.assertEqual(spot_dca.state_path_for("HYPEUSD"), spot_engine.state_path_for("HYPEUSD"))

    def test_missing_venue_precision_fails_before_strategy_start(self):
        executor = _Executor()
        executor.pair_precision = lambda _symbol: None
        with self.assertRaisesRegex(RuntimeError, "no pair metadata"):
            spot_engine.Strategy(
                executor, "ASSETUSD", _params(), dry_run=False,
                initial_state=spot_engine._new_state(),
            )

    def test_legacy_kraken_module_reexports_canonical_engine(self):
        path = os.path.join(ROOT, "kraken", "strategy.py")
        spec = importlib.util.spec_from_file_location("legacy_kraken_strategy", path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

        self.assertIs(legacy.Strategy, spot_engine.Strategy)
        self.assertIs(legacy.StratParams, spot_engine.StratParams)
        self.assertEqual(legacy._new_state(), spot_engine._new_state())

    def test_default_state_path_stays_in_legacy_kraken_directory(self):
        expected = os.path.join(ROOT, "kraken", ".state_HYPEUSD.json")
        self.assertEqual(spot_engine.state_path_for("HYPE/USD"), expected)

    def test_state_directory_is_injectable_for_another_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = spot_engine.Strategy(
                _Executor(), "ASSET/USD", _params(), dry_run=True,
                initial_state=spot_engine._new_state(), state_dir=directory,
            )
            self.assertEqual(engine.state_file, os.path.join(directory, ".state_ASSETUSD.json"))

    def test_notification_sink_and_source_are_injectable(self):
        events = []
        engine = spot_engine.Strategy(
            _Executor(), "ASSETUSD", _params(), dry_run=True,
            initial_state=spot_engine._new_state(), notifier=lambda **event: events.append(event),
            notification_source="paper-venue",
        )
        order = {
            "side": "buy", "kind": "ENTRY", "amount": 100.0,
            "vol": 1.0, "price": 100.0,
        }

        engine._apply_fill(order, 1.0, 100.0, 0.0)

        self.assertEqual(events[0]["source"], "paper-venue")
        self.assertEqual(events[0]["price"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
