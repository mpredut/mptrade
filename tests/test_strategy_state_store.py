"""The shared persistence contract for the financial engines."""

import ast
import json
import os
import pytest
import tempfile
import unittest
from assetguardian import AssetGuardianState
from pathlib import Path
from providers.base import MarketDataProvider
from state_io import StateReadError
from strategies.base_spot_mixin import BaseSpotMixin
from strategies.spot_engine import Strategy as SpotStrategy
from strategies.state_store import (
    JsonStateStore,
    StatePersistenceError,
    safe_state_token,
    state_path_for,
)
from tests.conftest import load_venue_modules
from trailing_core import TrailingCore
from unittest.mock import patch


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
        self.assertEqual(safe_state_token("CL=F", "unknown", allow_equals=True), "CL=F")
        self.assertEqual(safe_state_token("CL=F", "unknown", allow_equals=False), "CLF")
        self.assertEqual(safe_state_token(" LONG ", "long", lower=True), "long")
        self.assertEqual(safe_state_token(True, "unknown"), "unknown")
        self.assertEqual(safe_state_token("...", "unknown"), "unknown")
        self.assertEqual(
            state_path_for("TAO/USD", None, default_dir="/tmp/kraken").replace("\\", "/"),
            "/tmp/kraken/.state_TAOUSD.json",
        )

    def test_base_spot_mixin_and_step_sub_stages_present_on_both_engines(self):

        T212Strategy = load_venue_modules("212trading", "strategy").Strategy

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


def test_assetguardian_initializes_its_state_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for load_first in (False, True):
        for relative in (False, True):
            sub = tmp_path / f"dir_{load_first}_{relative}"
            sub.mkdir(parents=True, exist_ok=True)
            monkeypatch.chdir(sub)
            path = "assetguardian.json" if relative else str(sub / "nested" / "state.json")
            state = AssetGuardianState(path)
            if load_first:
                assert state.load() == {}
            value = {"version": 2, "symbols": {"TEST": {"pending": True}}}
            state.save(value)
            assert state.load() == value


def test_assetguardian_rejects_corrupt_live_state(tmp_path):
    path = tmp_path / "assetguardian.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(StateReadError, match="AssetGuardian"):
        AssetGuardianState(str(path)).load()


def test_assetguardian_reset_requires_explicit_non_live_policy(tmp_path):
    path = tmp_path / "assetguardian.json"
    path.write_text("[]", encoding="utf-8")
    assert AssetGuardianState(str(path), fail_closed=False).load() == {}


def test_live_trailing_rejects_corrupt_state_but_paper_can_reset(tmp_path):
    path = tmp_path / "trailing.json"
    path.write_text("{broken", encoding="utf-8")
    common = dict(
        adapter=object(), log=lambda _message: None, state_file=str(path),
        min_notional=1.0, rebuy_enabled=False, rebuy_bounce_pct=1.0,
        rebuy_skip_if_trend_down=False, sell_skip_if_trend_up=False,
    )
    live = TrailingCore(enabled=True, **common)
    with pytest.raises(StateReadError, match="live trailing"):
        live.load()
    assert TrailingCore(enabled=False, **common).load() == {}


def test_hyperliquid_live_engines_declare_fail_closed_reads():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for relative in ("hyperliquid/strategy.py",):
        text = open(os.path.join(root, relative), encoding="utf-8").read()
        assert "fail_closed=not self.dry_run" in text


def test_strategies_core_modules_dag_and_venue_purity():
    """Enforce DAG layering and zero hardcoded venue/mode branches across strategies/."""

    assert MarketDataProvider.resolve_venue_strategy_heartbeat("Hyperliquid", "HYPE") == "HL-bot"
    assert MarketDataProvider.resolve_venue_strategy_heartbeat("Kraken", "HYPEUSD") == "Kraken-HYPE"
    assert MarketDataProvider.resolve_venue_strategy_heartbeat("Kraken", "TAOUSD") == "Kraken-TAOUSD"
    assert MarketDataProvider.resolve_venue_strategy_heartbeat("T212", "NVDA_US_EQ") == "T212-bot"

    strat_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "strategies"
    forbidden_import_roots = {
        "binance_api", "binance", "kraken", "hyperliquid", "212trading",
        "hl_client", "orchestratorOS", "orchestratorTrade",
        "monitortrades", "assetguardian", "rtrade", "tradeall",
    }
    venue_tokens = ("binance", "kraken", "hyperliquid", "t212", "212trading")
    pure_venue_files = {
        "base_spot_mixin.py",
        "mt_position_accounting.py",
        "position_exit_manager.py",
        "rtrade_pair.py",
        "spot_replay.py",
        "spot_rules.py",
        "state_store.py",
    }

    for py_file in sorted(strat_dir.glob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        assert "_exit_manager_mode" not in src, f"{py_file.name} still references _exit_manager_mode"
        tree = ast.parse(src)

        docstring_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    docstring_nodes.add(node.body[0].value)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden_import_roots, f"{py_file.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in forbidden_import_roots, f"{py_file.name} imports from {node.module}"
            elif (
                py_file.name in pure_venue_files
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node not in docstring_nodes
            ):
                low = node.value.lower()
                assert not any(tok in low for tok in venue_tokens), (
                    f"{py_file.name}:{node.lineno} contains hardcoded venue literal {node.value!r}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
