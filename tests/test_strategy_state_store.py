"""The shared persistence contract for the financial engines."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from strategies.state_store import (
    JsonStateStore,
    StatePersistenceError,
    safe_state_token,
    state_path_for,
)


def _defaults():
    return {"cycle": 1, "qty": 0.0, "orders": []}


class JsonStateStoreTest(unittest.TestCase):
    def _store(self, path, *, fail_closed):
        messages = []
        store = JsonStateStore(
            path, _defaults, label="TEST", logger=messages.append,
            fail_closed=fail_closed,
        )
        return store, messages

    def test_load_merges_schema_and_rejects_corruption_by_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"qty": 2.5}, handle)
            store, _messages = self._store(path, fail_closed=True)
            self.assertEqual(store.load(), {"cycle": 1, "qty": 2.5, "orders": []})

            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"qty":')
            with self.assertRaisesRegex(StatePersistenceError, "invalid TEST state"):
                store.load()

            paper, messages = self._store(path, fail_closed=False)
            self.assertEqual(paper.load(), _defaults())
            self.assertIn("reset allowed only in PAPER", messages[-1])

    def test_save_replaces_atomically_without_temporary_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            store, _messages = self._store(path, fail_closed=True)

            self.assertTrue(store.save({"cycle": 2, "qty": 1.25, "orders": []}))

            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["qty"], 1.25)
            self.assertEqual(os.listdir(directory), ["state.json"])

    def test_failed_replace_cleans_temporary_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            store, _messages = self._store(path, fail_closed=True)
            with patch("state_io.os.replace", side_effect=OSError("disk")):
                with self.assertRaisesRegex(StatePersistenceError, "persisting the"):
                    store.save(_defaults())
            self.assertEqual(os.listdir(directory), [])

    def test_paper_save_failure_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            store, messages = self._store(path, fail_closed=False)
            with patch("state_io.os.replace", side_effect=OSError("disk")):
                self.assertFalse(store.save(_defaults()))
            self.assertIn("cannot save the state", messages[-1])
            self.assertEqual(os.listdir(directory), [])

    def test_safe_state_token_and_state_path_for_sanitize_and_fallback(self):
        self.assertEqual(safe_state_token("HYPE/USD", "unknown"), "HYPEUSD")
        self.assertEqual(safe_state_token(" LONG ", "long", lower=True), "long")
        self.assertEqual(safe_state_token(True, "unknown"), "unknown")
        self.assertEqual(safe_state_token("...", "unknown"), "unknown")
        self.assertEqual(
            state_path_for("TAO/USD", None, default_dir="/tmp/kraken").replace("\\", "/"),
            "/tmp/kraken/.state_TAOUSD.json",
        )

    def test_base_spot_mixin_and_step_sub_stages_present_on_both_engines(self):
        import importlib.util
        import sys
        from strategies.base_spot_mixin import BaseSpotMixin
        from strategies.spot_engine import Strategy as SpotStrategy

        spec = importlib.util.spec_from_file_location(
            "t212_strategy_mod",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "212trading", "strategy.py"),
        )
        t212_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = t212_mod
        spec.loader.exec_module(t212_mod)
        T212Strategy = t212_mod.Strategy

        self.assertTrue(issubclass(SpotStrategy, BaseSpotMixin))
        self.assertTrue(issubclass(T212Strategy, BaseSpotMixin))
        for cls in (SpotStrategy, T212Strategy):
            for method_name in (
                "_step_sync_market",
                "_step_evaluate_exits",
                "_step_evaluate_reentry",
                "_step_dispatch_orders",
                "_evaluate_mdhtp_tp",
                "_dispatch_take_profit",
                "_dispatch_dca_buy",
                "_apply_buy_fill_accounting",
                "_release_sell_cost_basis",
                "_mark_order_cancel_requested",
                "_recover_state_write_or_skip",
            ):
                self.assertTrue(callable(getattr(cls, method_name, None)), f"Missing {method_name} on {cls}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
