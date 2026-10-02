"""Live financial policy is explicit; offline profiles do not inherit process state."""

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from botcore import parse_dotenv
from strategies.spot_dca import StratParams


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {venue: parse_dotenv(str(ROOT / venue / "config.env"))
            for venue in ("kraken", "hyperliquid")}
REQUIRED_KEYS = sorted(key for key in PROFILES["kraken"]
                       if key.startswith("STRAT_") and key != "STRAT_EXECUTE") + ["STRATEGY_MODE"]


@pytest.mark.parametrize("venue", PROFILES)
@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_every_live_strategy_setting_must_exist(venue, key, monkeypatch):
    env = PROFILES[venue].copy()
    monkeypatch.setenv(key, env.pop(key))
    with pytest.raises(ValueError, match=key):
        StratParams.from_env(env)


@pytest.mark.parametrize("venue", PROFILES)
@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_invalid_settings_never_disable_or_default_policy(venue, key):
    env = {**PROFILES[venue], key: "not-a-setting"}
    if key == "STRAT_CURRENCY":
        env[key] = ""
    with pytest.raises(ValueError, match=key):
        StratParams.from_env(env)


@pytest.mark.parametrize("venue", PROFILES)
def test_versioned_profiles_preserve_effective_values_and_environment(venue):
    env = PROFILES[venue]
    before = dict(os.environ)
    params = StratParams.from_env(env)
    assert dict(os.environ) == before
    with patch.dict(os.environ, env, clear=True):
        assert StratParams.from_env() == params
    assert params.stop_loss_pct == 0
    assert params.reentry_peak_relative is False
    assert params.reentry_pullback_pct == 1.5
    assert params.reentry_hybrid_enabled is True
    assert params.surge_gain_pct == float(env["STRAT_SURGE_GAIN_PCT"])
    assert params.trend_sma_n == 30
    assert params.tp_trend_min_pct == 0.5
    assert params.tp_trail_pct == (3 if venue == "kraken" else 2)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", ""])
def test_nonfinite_or_empty_new_policy_is_rejected(value):
    with pytest.raises(ValueError, match="STRAT_SURGE_GAIN_PCT"):
        StratParams.from_env({**PROFILES["kraken"], "STRAT_SURGE_GAIN_PCT": value})


@pytest.mark.parametrize("value", ["garbage", "3:40", "3:100,junk", "nan:100", "3:-50,6:150"])
def test_malformed_tranches_cannot_silently_become_single_tp(value):
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
    import json
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
