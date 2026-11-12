"""Regression tests for Wave 15 backtest, replay, promotion, ML forecast, and runner fixes."""

import ast
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from offline.backtests.datasets import save_dataset
from offline.backtests.execution import ExecutionModel, FeeModel
from offline.backtests.promotion import evaluate_dual_promotion, evaluate_risk_adjusted_promotion
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
from offline.research.ml_forecast import forecast as ml_fc, vol_chronos as ml_vc
from offline.research.monitortrades_backtest import scheduled_pilot as sp
from offline.runners import (
    apply_proposals,
    binance_walk_forward_baseline as bwfb,
    hyperliquid_walk_forward_baseline as hwfb,
    walk_forward_compare as wfc,
)
from state_io import atomic_write_text, read_json_file, read_text_file
from strategies import spot_engine as strat, spot_replay
from tests.conftest import load_venue_modules

ROOT = Path(__file__).resolve().parents[1]
KR_TRAIL = load_venue_modules("kraken", "backtest_trail")

_WAVE15_PY_FILES = (
    "strategies/spot_replay.py",
    "offline/backtests/promotion.py",
    "offline/research/t212_continuous_grid/grid.py",
    "offline/research/kraken_continuous_grid/grid.py",
    "offline/runners/walk_forward_compare.py",
    "offline/runners/binance_walk_forward_baseline.py",
    "offline/runners/hyperliquid_walk_forward_baseline.py",
    "offline/research/monitortrades_backtest/scheduled_pilot.py",
    "offline/runners/apply_proposals.py",
    "offline/research/ml_forecast/forecast.py",
    "offline/research/ml_forecast/vol_chronos.py",
    "kraken/backtest_trail.py",
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
    "tests/test_wave15_backtest_fixes.py",
)

_WAVE15_SH_FILES = (
    "offline/runners/load_dev_backtest_env.sh",
    "offline/runners/run_dev_regression.sh",
)


def _make_promo_report(
    *,
    mean: float = 1.0,
    worst: float = -2.0,
    drawdown: float = 6.0,
    returns: tuple[float | None, ...] = (0.5,) * 20,
    drawdowns: tuple[float | None, ...] = (6.0,) * 20,
    calmar: float | None = 0.0,
    sortino: float | None = 0.0,
    cvar: float | None = -3.0,
    exposure: float | None = 60.0,
) -> dict:
    windows = [
        {
            "key": f"w{idx:02d}",
            "return_pct": r_val,
            "max_drawdown_pct": dd_val,
        }
        for idx, (r_val, dd_val) in enumerate(zip(returns, drawdowns), 1)
    ]
    return {
        "venue": "kraken",
        "initial_capital_usd": 3900.0,
        "dataset": {"sha256": "abc123def456", "bars": 2000},
        "walk_forward": {
            "interval_minutes": 240,
            "train": 720,
            "validation": 180,
            "test": 90,
            "step": 90,
            "warmup": 40,
            "anchored_train": False,
        },
        "scenarios": {
            "central": {
                "assumptions": {"fee_pct": 0.26},
                "windows": windows,
                "aggregate": {
                    "mean_return_pct": mean,
                    "worst_return_pct": worst,
                    "worst_max_drawdown_pct": drawdown,
                    "worst_cvar_95_pct": cvar,
                    "mean_exposure_pct": exposure,
                    "median_calmar": calmar,
                    "median_sortino": sortino,
                },
            }
        },
    }


def test_wave15_ascii_ast_and_shell_hygiene() -> None:
    """All Wave 15 touched files obey 100% ASCII, zero raw casts/open/json, and valid bash syntax."""
    forbidden_calls = {"float", "int", "open"}
    for rel in (*_WAVE15_PY_FILES, *_WAVE15_SH_FILES):
        src = read_text_file(str(ROOT / rel))
        assert all(ord(ch) < 128 for ch in src), f"Non-ASCII character in {rel}"

    for rel in _WAVE15_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls, (
                        f"Forbidden raw call {node.func.id}() in {rel}:{node.lineno}"
                    )
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    assert not (
                        node.func.value.id == "json" and node.func.attr in {"load", "loads", "dump", "dumps"}
                    ), f"Forbidden json.{node.func.attr}() in {rel}:{node.lineno}"

    for rel in _WAVE15_SH_FILES:
        proc = subprocess.run(["bash", "-n", str(ROOT / rel)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"bash -n failed for {rel}: {proc.stderr}"


def test_spot_replay_finite_cash_dust_clamp_and_limit_buy_clamping() -> None:
    """spot_replay clamps float-dust cash to 0.0 and clamps non-market BUY fills cleanly."""
    params = spot_replay.prepare_spot_interval_params(
        strat.StratParams(
            currency="USD",
            entry_amount=100.0,
            entry_discount_pct=0.0,
            dca_amount=50.0,
            dca_drop_pct=2.0,
            check_minutes=60.0,
            takeprofit_pct=10.0,
            max_budget=500.0,
            max_dca_buys=4,
            enable_takeprofit=True,
            order_ttl_min=60.0,
            stop_loss_pct=0.0,
        ),
        60.0,
    )
    bars = [(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)]
    res = spot_replay.run_replay(
        bars,
        params,
        fee_pct=0.26,
        bar_minutes=60.0,
        initial_cash=100.26 - 8e-10,
    )
    assert res["funding"]["final_cash"] >= 0.0
    assert res["funding"]["minimum_cash"] >= 0.0
    assert res["funding"]["final_cash"] == pytest.approx(0.0, abs=1e-12)

    # Same-bar market BUY gap-up + resting limit BUY with tight cash must clamp cleanly without crashing
    orig_step = strat.Strategy.step

    def _inject_dual_buys(self, price, timestamp=None, high=None):
        if not self.s.get("_injected_test_orders"):
            self.s["_injected_test_orders"] = True
            self.s["orders"].append(
                {"id": "m1", "side": "buy", "vol": 0.5, "price": 100.0, "amount": 50.0, "market": True, "kind": "ENTRY"}
            )
            self.s["orders"].append(
                {"id": "l1", "side": "buy", "vol": 0.5, "price": 100.0, "amount": 50.0, "market": False, "kind": "DCA"}
            )
            return
        return orig_step(self, price, timestamp=timestamp, high=high)

    strat.Strategy.step = _inject_dual_buys
    try:
        gap_bars = [
            (100.0, 101.0, 99.0, 100.0),
            (120.0, 125.0, 99.0, 105.0),
        ]
        res_gap = spot_replay.run_replay(
            gap_bars,
            params,
            bar_minutes=60.0,
            fee_model=FeeModel(limit_fee_pct=0.26, market_fee_pct=0.26),
            execution=ExecutionModel(intrabar_policy="worst_case"),
            initial_cash=100.26,
        )
    finally:
        strat.Strategy.step = orig_step

    assert res_gap["fills"] == 2
    assert res_gap["funding"]["final_cash"] >= 0.0
    assert res_gap["funding"]["minimum_cash"] >= 0.0


def test_promotion_defensive_gate_zero_baseline_calmar_and_sortino_continuity() -> None:
    """evaluate_risk_adjusted_promotion handles median_calmar==0.0 and median_sortino==0.0 continuously."""
    base_zero = _make_promo_report(
        mean=0.0,
        worst=-4.0,
        drawdown=6.0,
        returns=(0.0,) * 20,
        drawdowns=(6.0,) * 16 + (3.0,) * 4,
        calmar=0.0,
        sortino=0.0,
        cvar=-3.0,
        exposure=60.0,
    )
    cand_strong = _make_promo_report(
        mean=0.8,
        worst=-1.5,
        drawdown=3.5,
        returns=(0.05,) * 20,
        drawdowns=(3.0,) * 16 + (3.5,) * 4,
        calmar=1.1,
        sortino=1.4,
        cvar=-1.2,
        exposure=45.0,
    )
    dual = evaluate_dual_promotion(base_zero, cand_strong)
    assert dual["promote"] is True
    assert "DEFENSIVE" in dual["promotion_paths"]
    checks = dual["defensive_gate"]["scenarios"]["central"]["checks"]
    assert checks["calmar_available"] is True
    assert checks["material_calmar_improvement"] is True
    assert checks["sortino_preserved"] is True

    cand_tiny = _make_promo_report(
        mean=0.8,
        worst=-1.5,
        drawdown=3.5,
        returns=(0.05,) * 20,
        drawdowns=(3.0,) * 16 + (3.5,) * 4,
        calmar=0.05,
        sortino=1.4,
        cvar=-1.2,
        exposure=45.0,
    )
    res_tiny = evaluate_risk_adjusted_promotion(base_zero, cand_tiny)
    assert res_tiny["promote"] is False
    assert res_tiny["scenarios"]["central"]["checks"]["material_calmar_improvement"] is False

    cand_nan = _make_promo_report(
        mean=0.8,
        worst=-1.5,
        drawdown=3.5,
        returns=(0.05,) * 20,
        drawdowns=(3.0,) * 16 + (3.5,) * 4,
        calmar=math.nan,
        sortino=1.4,
        cvar=-1.2,
        exposure=45.0,
    )
    res_nan = evaluate_risk_adjusted_promotion(base_zero, cand_nan)
    assert res_nan["promote"] is False
    assert res_nan["scenarios"]["central"]["checks"]["calmar_available"] is False


def test_continuous_grid_canon_aliases_avoid_parent_recompute_crash() -> None:
    """t212 and kraken continuous grid collectors look up canon_aliases without parent recomputation."""
    t212_src = read_text_file(str(ROOT / "offline/research/t212_continuous_grid/grid.py"))
    kr_src = read_text_file(str(ROOT / "offline/research/kraken_continuous_grid/grid.py"))
    assert "canon_aliases" in t212_src
    assert "canon_aliases" in kr_src
    t212_loop_tail = t212_src.split("pool.imap(task, canon_todo):", 1)[1]
    assert "params_for(" not in t212_loop_tail
    kr_loop_tail = kr_src.split("pool.imap_unordered(task, canon_todo):", 1)[1]
    assert "CONFIGS[" not in kr_loop_tail


def test_load_dev_backtest_env_and_run_dev_regression_hardening(tmp_path: Path) -> None:
    """load_dev_backtest_env initializes ROOT even with exported env_get; run_dev_regression guards SSH 255 & args."""
    env_script = ROOT / "offline/runners/load_dev_backtest_env.sh"
    # 1. Direct execution with missing profile exits 1 cleanly without bash 'return' error
    bad_prof = subprocess.run(
        ["bash", str(env_script)],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "DEV_BACKTEST_ENV": str(tmp_path / "missing.env")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_prof.returncode == 1
    assert "can only `return'" not in bad_prof.stderr
    assert "can only 'return'" not in bad_prof.stderr

    # 2. Exported env_get in parent shell without ROOT still initializes ROOT cleanly under set -u
    exported_fn_cmd = (
        "env_get() { printf '%s' \"${2:-}\"; }; export -f env_get; "
        "unset ROOT PYTHON_BIN; "
        f"source {str(env_script)!r} && printf 'ROOT=%s\\n' \"$ROOT\""
    )
    ok_env = subprocess.run(
        ["bash", "-c", exported_fn_cmd],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_env.returncode == 0, ok_env.stderr
    assert f"ROOT={ROOT}" in ok_env.stdout

    # 3. run_dev_regression.sh rejects bad/extra flags with exit 2 and does not report RUNNING on SSH 255
    reg_script = ROOT / "offline/runners/run_dev_regression.sh"
    for bad_args in (["trigger", "--bad"], ["run", "--extra"], ["check", "--extra"]):
        proc = subprocess.run(["bash", str(reg_script), *bad_args], capture_output=True, text=True, check=False)
        assert proc.returncode == 2

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    atomic_write_text(str(fake_ssh), "#!/usr/bin/env bash\nexit 255\n")
    fake_ssh.chmod(0o755)
    fake_scp = fake_bin / "scp"
    atomic_write_text(str(fake_scp), "#!/usr/bin/env bash\nexit 1\n")
    fake_scp.chmod(0o755)

    check_env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "DEV_REGRESSION_LOCK": str(tmp_path / "isolated_dev_reg.lock"),
    }
    chk = subprocess.run(["bash", str(reg_script), "check"], env=check_env, capture_output=True, text=True, check=False)
    assert chk.returncode == 0
    assert "RUNNING:" not in chk.stdout
    assert "currently RUNNING" not in chk.stdout


def test_walk_forward_fee_stress_keys_and_supplied_dataset_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """walk_forward_compare preserves >3-decimal fee keys; baseline runners fail closed on mismatched --dataset."""
    records = [
        {
            "timestamp": 1700000000 + i * 14400,
            "open": 20.0 + (i % 5) * 0.2,
            "high": 20.8 + (i % 5) * 0.2,
            "low": 19.4 + (i % 5) * 0.2,
            "close": 20.1 + ((i + 2) % 5) * 0.2,
            "volume": 1000.0,
        }
        for i in range(1100)
    ]
    ds_csv = tmp_path / "hype_240m.csv"
    save_dataset(records, ds_csv)
    short_csv = tmp_path / "short_240m.csv"
    save_dataset(records[:5], short_csv)

    monkeypatch.setattr(bwfb, "fetch_ohlc_candles", lambda *a, **kw: [])
    monkeypatch.setattr(hwfb, "fetch_ohlc_candles", lambda *a, **kw: [])

    # Supplied --dataset with <20 bars or mismatched interval must fail closed (return 1)
    assert bwfb.main(["--intervals", "240", "--dataset", f"240={short_csv}", "--output-dir", str(tmp_path / "bn_s")]) == 1
    assert hwfb.main(["--intervals", "240", "--dataset", f"240={short_csv}", "--output-dir", str(tmp_path / "hl_s")]) == 1
    assert bwfb.main(["--intervals", "240", "--dataset", f"60={ds_csv}", "--output-dir", str(tmp_path / "bn")]) == 1
    assert hwfb.main(["--intervals", "240", "--dataset", f"60={ds_csv}", "--output-dir", str(tmp_path / "hl")]) == 1

    # Generate valid Binance baseline and run walk_forward_compare with sub-millipoint fee tiers
    bn_out = tmp_path / "bn_ok"
    assert (
        bwfb.main(["--symbol", "BTCUSDT", "--intervals", "240", "--dataset", f"240={ds_csv}", "--output-dir", str(bn_out)])
        == 0
    )
    base_json = next(bn_out.glob("BTCUSDT_240m_baseline.json"))
    orig_cands = wfc.default_candidates()
    monkeypatch.setattr(wfc, "default_candidates", lambda: [c for c in orig_cands if c.name == "live"])
    cmp_dir = tmp_path / "cmp_out"
    rc_cmp = wfc.main(
        [
            str(base_json),
            "--fee-stress",
            "0.01,0.0104,0.012,0.0125,0.075,0.26",
            "--output-dir",
            str(cmp_dir),
        ]
    )
    assert rc_cmp == 0
    cmp_json = next(cmp_dir.glob("*comparison*.json"))
    cmp_payload = read_json_file(str(cmp_json))
    assert set(cmp_payload["fee_stress"].keys()) == {"0.01", "0.0104", "0.012", "0.0125", "0.075", "0.26"}


def test_pilot_grid_validation_ml_hourly_contiguity_and_cli_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate <100% pilot stop/trailing bounds, ML internal timestamp gap detection, and CLI guards."""
    assert sp._valid_pilot_grid_value("mt.lost", 100.0) is None
    assert sp._valid_pilot_grid_value("trailing.pct", 100.0) is None
    assert sp._valid_pilot_grid_value("mt.lost", 5.5) == pytest.approx(5.5)

    audits: list[dict] = []
    monkeypatch.delenv("APPLY_DISABLED", raising=False)
    monkeypatch.setenv("APPLY_PROPOSALS_LOCK", str(tmp_path / "apply.lock"))
    monkeypatch.setattr(apply_proposals, "_ALLOWED_PILOT_PARAM_NAMES", frozenset({"mt.lost"}))
    monkeypatch.setattr(apply_proposals, "load_env_stack", lambda: None)
    monkeypatch.setattr(
        apply_proposals,
        "_read_proposals",
        lambda: [
            {
                "full_key": "BINANCE_BTC.mt.lost",
                "section": "BINANCE_BTC",
                "key": "mt.lost",
                "symbol": "BTCUSDC",
                "winner_value": 100.0,
                "dev_commit": "abc1234",
            }
        ],
    )
    monkeypatch.setattr(apply_proposals.sp, "_current_value", lambda _s, _k: 5.0)
    monkeypatch.setattr(apply_proposals.sp, "_append_audit", lambda rec: audits.append(dict(rec)))
    apply_proposals.main(["--dry-run"])
    assert len(audits) == 1
    assert audits[0]["action"] == "skipped_invalid_value"

    # Internal duplicate+jump timestamp corruption keeping outer span unchanged must be caught!
    px = np.linspace(100.0, 125.0, ml_fc.WARMUP + ml_fc.HORIZON_H + 4)
    ts_corrupt = np.array([1700000000.0 + i * 3600.0 for i in range(len(px))], dtype=float)
    # Duplicate one candle timestamp while preserving outer endpoints -> step 0s then step 7200s!
    mid = ml_fc.WARMUP // 2
    ts_corrupt[mid] = ts_corrupt[mid - 1]
    X_bad, _, _ = ml_fc.build_dataset(px, ts=ts_corrupt)
    assert len(X_bad) == 0

    ts_live_corrupt = np.array([1700000000.0 + i * 3600.0 for i in range(len(px))], dtype=float)
    ts_live_corrupt[-10] = ts_live_corrupt[-11]
    lf_bad = ml_fc.live_forecast(
        px,
        np.empty((0, len(ml_fc.FEATURES))),
        np.empty(0, dtype=int),
        np.empty(0, dtype=float),
        {},
        "TAOUSDC",
        ts=ts_live_corrupt,
    )
    assert lf_bad.get("gap_detected") is True

    # vol_chronos masks cross-gap returns and skips gapped horizon windows
    px_vc = np.linspace(100.0, 120.0, 40)
    ts_vc = np.array([1700000000.0 + i * 3600.0 for i in range(len(px_vc))], dtype=float)
    ts_vc[15:] += 48.0 * 3600.0
    rv_gapped = ml_vc.realized_vol_series(px_vc, win=5, ts=ts_vc)
    assert math.isnan(rv_gapped[16])
    assert math.isfinite(rv_gapped[25])
    wf_gapped = ml_vc.walk_forward_vol(np.full(40, 0.2), horizon=10, warmup=10, stride=1, ts=ts_vc[:1] * 40)
    assert wf_gapped["n_test"] == 0

    # CLI validators on kraken/backtest_trail.py and run_deep_surge_sweep.py
    fetched = []
    monkeypatch.setattr(KR_TRAIL, "fetch_candles", lambda *a, **kw: fetched.append(1) or [])
    monkeypatch.setattr(KR_TRAIL, "fetch_with_time", lambda *a, **kw: fetched.append(1) or [])
    for bad_argv in (["--wf", "../BTCUSD"], ["--intraday", "../BTCUSD"], ["--wf", " "]):
        with pytest.raises(SystemExit):
            KR_TRAIL.main(bad_argv)
    assert fetched == []

    monkeypatch.setattr("sys.argv", ["run_deep_surge_sweep.py", "--fee-pct", "100"])
    with pytest.raises(SystemExit):
        rdss.main()
