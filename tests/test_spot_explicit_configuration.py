"""Live financial policy is explicit; offline profiles do not inherit process state, and SpotEngine module boundaries are preserved."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest

from botcore import parse_dotenv
from providers.strategy_executor import PairPrecision
from strategies import spot_dca, spot_engine
from strategies.spot_engine import StratParams


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {venue: parse_dotenv(str(ROOT / venue / "config.env"))
            for venue in ("kraken", "hyperliquid")}
REQUIRED_KEYS = sorted(key for key in PROFILES["kraken"]
                       if key.startswith("STRAT_") and key != "STRAT_EXECUTE") + ["STRATEGY_MODE", "SHADOW_K_REENTRY"]


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


def test_every_live_strategy_setting_must_exist(monkeypatch):
    for venue in PROFILES:
        for key in REQUIRED_KEYS:
            env = PROFILES[venue].copy()
            monkeypatch.setenv(key, env.pop(key))
            with pytest.raises(ValueError, match=key):
                StratParams.from_env(env)


def test_invalid_settings_never_disable_or_default_policy():
    for venue in PROFILES:
        for key in REQUIRED_KEYS:
            env = {**PROFILES[venue], key: "not-a-setting"}
            if key == "STRAT_CURRENCY":
                env[key] = ""
            with pytest.raises(ValueError, match=key):
                StratParams.from_env(env)


def test_versioned_profiles_preserve_effective_values_and_environment():
    for venue in PROFILES:
        env = PROFILES[venue]
        before = dict(os.environ)
        params = StratParams.from_env(env)
        assert dict(os.environ) == before
        with patch.dict(os.environ, env, clear=True):
            assert StratParams.from_env() == params
        assert params.stop_loss_pct == float(env["STRAT_STOP_LOSS_PCT"])
        assert params.reentry_peak_relative is False
        assert params.reentry_pullback_pct == float(env["STRAT_REENTRY_PULLBACK_PCT"])
        assert params.reentry_hybrid_enabled is True
        assert params.surge_gain_pct == float(env["STRAT_SURGE_GAIN_PCT"])
        assert params.trend_sma_n == 30
        assert params.tp_trend_min_pct == 0.5
        assert params.tp_trail_pct == (3 if venue == "kraken" else 2)


def test_versioned_profile_is_isolated_from_the_calling_environment():
    from offline.runners.spot_profile_review import revision_params
    config = (ROOT / "hyperliquid/config.env").read_text()
    with patch.dict(os.environ, STRAT_ENTRY="999999"), patch(
            "offline.runners.spot_profile_review.subprocess.run",
            return_value=SimpleNamespace(stdout=config)):
        params = revision_params("test:hyperliquid/config.env")
        assert params.entry_amount == 350
        assert os.environ["STRAT_ENTRY"] == "999999"


def test_nonfinite_or_empty_new_policy_is_rejected():
    for value in ["nan", "inf", "-inf", ""]:
        with pytest.raises(ValueError, match="STRAT_SURGE_GAIN_PCT"):
            StratParams.from_env({**PROFILES["kraken"], "STRAT_SURGE_GAIN_PCT": value})


def test_malformed_tranches_cannot_silently_become_single_tp():
    for value in ["garbage", "3:40", "3:100,junk", "nan:100", "3:-50,6:150"]:
        with pytest.raises(ValueError, match="STRAT_TP_TRANCHES"):
            StratParams.from_env({**PROFILES["kraken"], "STRAT_TP_TRANCHES": value})


def test_empty_and_valid_tranches_are_supported():
    assert StratParams.from_env(PROFILES["kraken"]).tp_tranches == []
    params = StratParams.from_env({**PROFILES["kraken"], "STRAT_TP_TRANCHES": "3:50,6:50"})
    assert params.tp_tranches == [(3, 50), (6, 50)]


def test_sweep_uses_selected_profile_not_cwd_or_shell(tmp_path, monkeypatch):
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRAT_ENTRY", "999999")
    params = sweep.load_base_params(ROOT / "hyperliquid/config.env")
    assert Path(sweep.ROOT) == ROOT
    assert params == StratParams.from_env(PROFILES["hyperliquid"])
    assert os.environ["STRAT_ENTRY"] == "999999"
    candidates = sweep.build_candidate_grid(params)
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    assert next(c["params"] for c in candidates if c["id"] == "CURRENT_CONFIG") == params
    control = next(c["params"] for c in candidates if c["id"] == "BASELINE_SURGE_OFF")
    assert control == replace(params, surge_guard=False, surge_dynamic=False)
    assert all(c["params"].effective_max_budget() == 1050 for c in candidates)


def test_sweep_threads_cash_and_explicit_fees_through_all_replays(monkeypatch):
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    params = StratParams.from_env(PROFILES["hyperliquid"])
    candidate = sweep.build_candidate_grid(params)[0]
    calls = []

    def replay(**kwargs):
        calls.append(kwargs)
        return {"total": 5, "max_drawdown_pct": 2, "cycles": 1}

    monkeypatch.setattr(sweep, "run_replay", replay)
    result = sweep.evaluate_single_candidate(candidate, {"HYPE": [(100, 101, 99, 100)] * 80}, 1050, 0.4)
    assert len(calls) == 3
    assert all(call["initial_cash"] == 1050 and call["fee_pct"] == 0.4 for call in calls)
    assert result["total_profit_usd"] == 5
    assert "calmar_ratio" not in result


def test_sweep_refuses_missing_datasets(tmp_path, monkeypatch):
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    monkeypatch.setattr("sys.argv", ["sweep", "--fee-pct", "0.4", "--data-dir", str(tmp_path),
                                     "--out-dir", str(tmp_path / "output")])
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 2
    assert not (tmp_path / "output/sweep_results.json").exists()


@pytest.mark.parametrize("fail_control", [False, True])
def test_sweep_report_is_complete_and_records_effective_policy(tmp_path, monkeypatch, fail_control):
    from concurrent.futures import ThreadPoolExecutor
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as sweep

    profile = ROOT / "hyperliquid/config.env"
    params = sweep.load_base_params(profile)
    candidates = sweep.build_candidate_grid(params)[:2]
    (tmp_path / "TEST_240m.csv").touch()
    monkeypatch.setattr(sweep, "ASSETS", ["TEST"])
    monkeypatch.setattr(sweep, "load_ohlc_csv", lambda _: [(100, 101, 99, 100)] * 80)
    monkeypatch.setattr(sweep, "build_candidate_grid", lambda _: candidates)
    monkeypatch.setattr(sweep.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)

    def evaluate(candidate, asset_data, initial_cash, fee_pct):
        if fail_control and candidate["category"] == "Baseline":
            raise RuntimeError("Synthetic baseline failure")
        assert list(asset_data) == ["TEST"]
        assert initial_cash == params.effective_max_budget()
        assert fee_pct == 0.4
        return {
            "id": candidate["id"], "category": candidate["category"], "desc": candidate["desc"],
            "total_profit_usd": 5, "worst_maxdd_pct": 2, "pnl_per_drawdown_point": 2.5,
            "h1_profit_usd": 2, "h2_profit_usd": 3, "asset_profits": {"TEST": 5},
            "asset_maxdds": {"TEST": 2},
        }

    monkeypatch.setattr(sweep, "evaluate_single_candidate", evaluate)
    monkeypatch.setattr("sys.argv", [
        "sweep", "--config", str(profile), "--fee-pct", "0.4",
        "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "output"),
    ])
    report_path = tmp_path / "output/sweep_results.json"
    if fail_control:
        with pytest.raises(RuntimeError, match="Incomplete sweep"):
            sweep.main()
        assert not report_path.exists()
    else:
        sweep.main()
        report = json.loads(report_path.read_text())
        assert report["schema_version"] == 2
        assert report["strategy_params"]["surge_gain_pct"] == params.surge_gain_pct
        assert report["initial_cash_per_asset"] == params.effective_max_budget()
        assert report["fee_pct_per_leg"] == 0.4
        assert report["total_candidates"] == 2
        assert len(report["limitations"]) == 4
        text = (tmp_path / "output/DEEP_SURGE_SWEEP_REPORT.md").read_text()
        assert "no live promotion is justified" in text
        assert "Calmar" not in text
