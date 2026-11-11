"""Regression tests for Wave 13 backtesting, forecast, pilot, sweep, and CLI fixes."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from offline.backtests import tradeall as bt_tradeall
from offline.backtests.datasets import dataset_metadata, save_dataset
from offline.research import comprehensive_strategy_sweep as css
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
from offline.research.hybrid_reentry import harness as hrh
from offline.research.ml_forecast import forecast as ml_fc, vol_chronos as ml_vc
from offline.research.monitortrades_backtest import scheduled_pilot as sp
from offline.runners import (
    apply_proposals,
    kraken_walk_forward_baseline as kwfb,
    spot_profile_review as spr,
)
from state_io import atomic_write_json, atomic_write_text, read_json_file, read_text_file
from tests.conftest import load_venue_modules

ROOT = Path(__file__).resolve().parents[1]
_HL_BT = load_venue_modules("hyperliquid", "backtest")
_KR_TRAIL = load_venue_modules("kraken", "backtest_trail")

WAVE13_PY_FILES = (
    "offline/runners/spot_profile_review.py",
    "offline/research/ml_forecast/forecast.py",
    "offline/research/ml_forecast/vol_chronos.py",
    "offline/runners/kraken_walk_forward_baseline.py",
    "offline/research/monitortrades_backtest/scheduled_pilot.py",
    "offline/runners/apply_proposals.py",
    "offline/research/comprehensive_strategy_sweep.py",
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
    "offline/research/hybrid_reentry/harness.py",
    "offline/research/hybrid_reentry/run_equities.py",
    "hyperliquid/backtest.py",
    "kraken/backtest_trail.py",
    "offline/backtests/tradeall.py",
    "tests/test_wave13_backtest_fixes.py",
)


def test_wave13_ascii_and_ast_hygiene() -> None:
    """Enforce 100% ASCII, zero raw float/int/open calls, and zero json imports."""
    for rel in WAVE13_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "json", f"Direct json import in {rel}"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "json", f"Direct from json import in {rel}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in {"float", "int", "open"}, (
                        f"Raw {node.func.id}() call in {rel}:{node.lineno}"
                    )
                for kw in node.keywords:
                    if kw.arg == "type" and isinstance(kw.value, ast.Name):
                        assert kw.value.id not in {"float", "int"}, (
                            f"Raw type={kw.value.id} in {rel}:{node.lineno}"
                        )


def test_spot_profile_review_historical_backfill_and_subbar_scaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """revision_params loads pre-MHDPA profiles cleanly while failing fast on missing core keys."""
    legacy_env = tmp_path / "legacy_hl.env"
    atomic_write_text(
        str(legacy_env),
        "\n".join(
            [
                "STRATEGY_MODE=avg_tp",
                "STRAT_CURRENCY=USDC",
                "STRAT_ENTRY=500",
                "STRAT_ENTRY_DISCOUNT_PCT=0.2",
                "STRAT_DCA=250",
                "STRAT_DCA_DROP_PCT=2.0",
                "STRAT_TAKEPROFIT_PCT=5.0",
                "STRAT_CHECK_MINUTES=1",
                "STRAT_MAX_BUDGET=3000",
                "STRAT_MAX_DCA_BUYS=8",
                "STRAT_ORDER_TTL_MIN=10",
                "STRAT_STOP_LOSS_PCT=7",
                "STRAT_REENTRY_TOLERANCE_PCT=0.05",
                "STRAT_TP_TREND_HOLD=true",
                "STRAT_TP_TRAIL_ADAPTIVE=true",
                "STRAT_TP_TRAIL_K=2.0",
                "STRAT_TP_TRAIL_MIN=1.5",
                "STRAT_TP_TRAIL_MAX=8.0",
            ]
        )
        + "\n",
    )
    loaded = spr.revision_params(str(legacy_env))
    assert loaded.entry_amount == pytest.approx(500.0)
    assert loaded.stop_loss_pct == pytest.approx(7.0)
    assert loaded.tp_trend_hold is True
    assert loaded.tp_trail_k == pytest.approx(2.0)
    assert loaded.surge_guard is False
    assert loaded.sl_mhdsl_enabled is False
    assert loaded.reentry_hybrid_enabled is False
    assert loaded.tp_mdhtp_enabled is False

    incomplete_env = tmp_path / "incomplete_hl.env"
    atomic_write_text(str(incomplete_env), "STRAT_ENTRY=500\nSTRAT_MAX_BUDGET=3000\n")
    with pytest.raises(ValueError):
        spr.revision_params(str(incomplete_env))

    # Verify review() scales sub-bar timeouts via prepare_spot_interval_params
    records = [
        {
            "timestamp": 1700000000 + i * 14400,
            "open": 20.0 + (i % 7) * 0.1,
            "high": 20.4 + (i % 7) * 0.1,
            "low": 19.6 + (i % 7) * 0.1,
            "close": 20.1 + (i % 7) * 0.1,
            "volume": 1000.0,
        }
        for i in range(1000)
    ]
    ds_csv = tmp_path / "ds_240m.csv"
    save_dataset(records, ds_csv)
    meta = dataset_metadata(records, interval_minutes=240)
    manifest_p = tmp_path / "manifest.json"
    atomic_write_json(str(manifest_p), {"datasets": {"240": {"sha256": meta["sha256"]}}})
    monkeypatch.setattr(spr, "DEFAULT_DATASET", ds_csv)
    monkeypatch.setattr(spr, "DEFAULT_MANIFEST", manifest_p)

    rep = spr.review(str(ROOT / "hyperliquid" / "config.env"), str(legacy_env), 1000.0)
    prev_sp = rep["profiles"]["previous"]["strategy_params"]
    assert prev_sp["check_minutes"] == pytest.approx(240.0)
    assert prev_sp["tp_stagnation_max_sec"] > 1080.0


def test_ml_forecast_and_vol_chronos_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """live_forecast detects terminal gaps, forecast.main preserves --out, and vol_chronos avoids NaN."""
    px = np.linspace(100.0, 130.0, 420)
    contig_ts = np.array([1700000000.0 + i * 3600.0 for i in range(len(px))], dtype=float)
    X, y_dir, y_mag = ml_fc.build_dataset(px, ts=contig_ts)
    assert len(X) > 0
    rep = {"lindy": 0.5, "logit": 0.55, "boost": 0.6}

    ok_fc = ml_fc.live_forecast(px, X, y_dir, y_mag, rep, "TAOUSDC", ts=contig_ts)
    assert "gap_detected" not in ok_fc

    gapped_ts = contig_ts.copy()
    gapped_ts[-1] += 72.0 * 3600.0
    gap_fc = ml_fc.live_forecast(px, X, y_dir, y_mag, rep, "TAOUSDC", ts=gapped_ts)
    assert gap_fc.get("gap_detected") is True
    assert gap_fc["proba_up"] == pytest.approx(0.5)
    assert gap_fc["expected_move_pct"] == pytest.approx(0.0)

    out_path = tmp_path / "forecast.json"
    atomic_write_json(str(out_path), {"sentinel": "keep"})
    monkeypatch.setattr(ml_fc, "fetch_klines", lambda _sym, _days: (gapped_ts, px))
    rc = ml_fc.main(["--symbol", "TAOUSDC", "--days", "30", "--forecast", "--out", str(out_path)])
    assert rc == 1
    assert read_json_file(str(out_path)) == {"sentinel": "keep"}

    with pytest.raises(SystemExit):
        ml_fc.main(["--symbol", "../bad"])
    with pytest.raises(SystemExit):
        ml_fc.main(["--loop", "-1"])

    empty_vc = ml_vc.walk_forward_vol(np.ones(20, dtype=float), horizon=24, warmup=10, stride=6)
    assert empty_vc["corr_model_vs_actual"] is None
    for v in empty_vc.values():
        if isinstance(v, float):
            assert math.isfinite(v)

    monkeypatch.setattr(ml_vc, "fetch_klines", lambda _sym, _days: (contig_ts[:50], px[:50]))
    assert ml_vc.main(["--symbol", "TAOUSDC", "--days", "5", "--eval"]) == 1


def test_kraken_walk_forward_baseline_slash_pair_and_subbar_scaling(tmp_path: Path) -> None:
    """kraken_walk_forward_baseline sanitizes slash pairs in output filenames and rejects traversal."""
    records = [
        {
            "timestamp": 1700000000 + i * 14400,
            "open": 20.0 + (i % 5) * 0.2,
            "high": 20.5 + (i % 5) * 0.2,
            "low": 19.5 + (i % 5) * 0.2,
            "close": 20.1 + (i % 5) * 0.2,
            "volume": 500.0,
        }
        for i in range(120)
    ]
    ds_csv = tmp_path / "hype_240m.csv"
    save_dataset(records, ds_csv)
    out_dir = tmp_path / "wf_out"

    with pytest.raises(SystemExit):
        kwfb.main(["--pair", "../HYPEUSD", "--intervals", "240", "--dataset", f"240={ds_csv}"])

    rc = kwfb.main(
        [
            "--pair",
            "HYPE/USD",
            "--intervals",
            "240",
            "--dataset",
            f"240={ds_csv}",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    saved_reports = list(out_dir.glob("baseline_HYPE_USD_*.json"))
    assert len(saved_reports) == 1
    payload = read_json_file(str(saved_reports[0]))
    assert payload["runtime"]["pair"] == "HYPE/USD"


def test_scheduled_pilot_and_apply_proposals_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scheduled_pilot filters invalid grid candidates, restores print, and refuses all-error overwrite."""
    assert sp._valid_pilot_grid_value("mt.gain", "5.0") == pytest.approx(5.0)
    assert sp._valid_pilot_grid_value("mt.gain", "0") is None
    assert sp._valid_pilot_grid_value("mt.gain", "-2.0") is None
    assert sp._valid_pilot_grid_value("mt.hardtp_fraction", "0.8") == pytest.approx(0.8)
    assert sp._valid_pilot_grid_value("mt.hardtp_fraction", "1.5") is None

    propose_out = tmp_path / "proposals.json"
    atomic_write_json(str(propose_out), [{"full_key": "keep"}])
    monkeypatch.delenv("PILOT_DISABLED", raising=False)
    monkeypatch.setattr(sp, "single_instance", lambda *_a, **_kw: None)
    monkeypatch.setattr(sp, "PILOT_KEYS", {"SEC.mt.gain": ("BTCUSDC", "BTC", "mt.gain")})
    monkeypatch.setattr(sp, "_split_series", lambda _sym: ([], []))
    monkeypatch.setattr(sp, "scan_backtest_ranges", lambda _p: {})
    monkeypatch.setattr(sp, "_append_audit", lambda _e: None)

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(sp, "evaluate_key", _boom)
    monkeypatch.setattr(
        "sys.argv",
        ["scheduled_pilot.py", "--propose", "--propose-out", str(propose_out)],
    )
    with pytest.raises(SystemExit) as exc_info:
        sp.main()
    assert exc_info.value.code == 1
    assert builtins.print is not sp.log._dummy_print
    assert read_json_file(str(propose_out)) == [{"full_key": "keep"}]

    # Also verify finally restores builtins.print when an unhandled error occurs
    monkeypatch.setattr(sp, "scan_backtest_ranges", lambda _p: (_ for _ in ()).throw(RuntimeError("io crash")))
    with pytest.raises(RuntimeError, match="io crash"):
        sp.main()
    assert builtins.print is not sp.log._dummy_print

    # apply_proposals dynamic env branch resolution + FETCH_HEAD priority
    monkeypatch.setattr(apply_proposals, "BRANCH", apply_proposals._DEFAULT_BRANCH)
    monkeypatch.setenv("BACKTEST_PROPOSALS_BRANCH", "custom-proposals-branch")
    assert apply_proposals._configured_branch() == "custom-proposals-branch"

    monkeypatch.setattr(apply_proposals, "BRANCH", "monkeypatched-branch")
    assert apply_proposals._configured_branch() == "monkeypatched-branch"

    git_cmds: list[list[str]] = []

    def _fake_git_run(cmd, **_kw):
        git_cmds.append(list(cmd))
        if "fetch" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "FETCH_HEAD:backtest_proposals.json" in cmd:
            return SimpleNamespace(returncode=0, stdout='[{"full_key": "A.mt.gain"}]', stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="missing ref")

    monkeypatch.setattr(apply_proposals.subprocess, "run", _fake_git_run)
    got_props = apply_proposals._read_proposals("backtest-proposals")
    assert got_props == [{"full_key": "A.mt.gain"}]
    assert any("FETCH_HEAD:backtest_proposals.json" in c for c in git_cmds)


def test_sweeps_hybrid_reentry_and_cli_backtest_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate fail-closed sweeps, ISO/partial CSV timestamps, and CLI validators."""
    # 1. comprehensive_strategy_sweep fails closed when both WF and continuous are empty
    monkeypatch.setattr(css, "load_dataset", lambda _p: [])
    monkeypatch.setattr(css, "run_walk_forward_evaluation", lambda *_a, **_kw: {})
    monkeypatch.setattr(css, "run_continuous_628d_replay", lambda *_a, **_kw: {})
    monkeypatch.setattr(css, "run_live_forward_shadow", lambda *_a, **_kw: {})
    with pytest.raises(SystemExit) as exc_css:
        css.main()
    assert exc_css.value.code == 1

    # 2. run_deep_surge_sweep.load_base_params scales sub-bar timeouts for 240m bars
    surge_base = rdss.load_base_params(ROOT / "kraken" / "config.env")
    assert surge_base.check_minutes == pytest.approx(240.0)
    assert surge_base.tp_stagnation_max_sec > 1080.0

    # 3. hybrid_reentry.harness.load_ohlc_with_timestamps_csv parses ISO dates & skips bad ts row
    iso_csv = tmp_path / "equity_iso.csv"
    atomic_write_text(
        str(iso_csv),
        "date,open,high,low,close\n"
        "2025-01-02T00:00:00Z,100.0,105.0,99.0,102.0\n"
        "bad_ts,102.0,106.0,101.0,104.0\n"
        "2025-01-03,104.0,108.0,103.0,107.0\n",
    )
    rows, ts_list = hrh.load_ohlc_with_timestamps_csv(str(iso_csv))
    assert len(rows) == 2
    assert len(ts_list) == 2
    assert ts_list[1] > ts_list[0]

    # 4. hyperliquid/backtest.py validates parameters BEFORE fetch_ohlc_candles
    fetch_called = []
    monkeypatch.setattr(_HL_BT, "fetch_ohlc_candles", lambda *a, **kw: fetch_called.append(1) or [])
    for bad_argv in (
        ["--budget", "0"],
        ["--coin", "../BTC"],
        ["--disc", "100"],
        ["--fee", "100"],
    ):
        with pytest.raises(SystemExit):
            _HL_BT.main(bad_argv)
    assert fetch_called == []

    # 5. kraken/backtest_trail.py and offline/backtests/tradeall.py CLI validators
    with pytest.raises(SystemExit):
        _KR_TRAIL.main(["--wf", "HYPEUSD", "--interval", "1.5"])

    for bad_ta_argv in (
        ["tradeall.py", "--symbol", "../BTCUSDC", "--start", "2026-06-01"],
        ["tradeall.py", "--symbol", "BTCUSDC", "--start", "2026-06-01", "--run-id", "../escape"],
        ["tradeall.py", "--symbol", "BTCUSDC", "--start", "2026-06-01", "--trend-gate-timeout-sec", "-1"],
        ["tradeall.py", "--symbol", "BTCUSDC", "--start", "2026-06-01", "--trend-gate-max-wait-sec", "-5"],
    ):
        monkeypatch.setattr("sys.argv", bad_ta_argv)
        with pytest.raises(SystemExit):
            bt_tradeall.main()
