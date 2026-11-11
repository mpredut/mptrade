"""Regression tests for Wave 14 backtesting, replay, ML forecast, grid, and runner fixes."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from botcore import load_local_module
from offline.backtests import evaluation as bt_eval
from offline.research import short_trend_estimator_benchmark as st_bench
from offline.research import slice_dense_archive as sda
from offline.research.kraken_continuous_grid import (
    grid as kcg_grid,
    report as kcg_report,
)
from offline.research.ml_forecast import priceprediction as pp_mod
from offline.research.monitortrades_backtest import compare_sl_variants as cslv
from offline.research.t212_continuous_grid import (
    grid as t212_grid,
    report as t212_report,
)
from offline.runners import (
    binance_walk_forward_baseline as bwf,
    fetch_hyperliquid_candles as fhc,
    hyperliquid_walk_forward_baseline as hwf,
    t212_walk_forward_baseline as twf,
    walk_forward_compare as wfc,
)
from state_io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_text_file
from strategies.replay_metrics import validate_bt_cli_token

ROOT = Path(__file__).resolve().parents[1]
HL_REPLAY = bt_eval.load_venue_replay_module("hyperliquid", "test_wave14_hl_replay")
T212_BT = load_local_module(str(ROOT / "212trading"), "backtest", module_key="test_wave14_t212_bt")
KRAKEN_BT = load_local_module(str(ROOT / "kraken"), "backtest", module_key="test_wave14_kraken_bt")

WAVE14_PY_FILES = (
    "strategies/replay_metrics.py",
    "hyperliquid/replay.py",
    "offline/research/ml_forecast/priceprediction.py",
    "offline/research/t212_continuous_grid/fetch.py",
    "offline/research/t212_continuous_grid/grid.py",
    "offline/research/t212_continuous_grid/report.py",
    "offline/research/kraken_continuous_grid/grid.py",
    "offline/research/kraken_continuous_grid/report.py",
    "offline/research/monitortrades_backtest/compare_sl_variants.py",
    "offline/research/short_trend_estimator_benchmark.py",
    "offline/backtests/evaluation.py",
    "offline/runners/walk_forward_compare.py",
    "offline/runners/kraken_walk_forward_baseline.py",
    "offline/runners/hyperliquid_walk_forward_baseline.py",
    "offline/runners/binance_walk_forward_baseline.py",
    "offline/runners/t212_walk_forward_baseline.py",
    "offline/runners/fetch_hyperliquid_candles.py",
    "offline/research/slice_dense_archive.py",
    "212trading/backtest.py",
    "kraken/backtest.py",
    "tests/test_wave14_backtest_fixes.py",
)

WAVE14_SH_FILES = (
    "offline/runners/run_backtest_cycle.sh",
    "offline/runners/run_backtest_parallel.sh",
    "offline/runners/sync_cache_to.sh",
    "offline/research/kraken_continuous_grid/run_all.sh",
)


def test_wave14_ascii_ast_and_shell_hygiene() -> None:
    """Enforce 100% ASCII, zero raw float/int/open calls, and clean bash -n syntax."""
    for rel in WAVE14_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"float", "int", "open"}, (
                    f"Forbidden raw {node.func.id}() in {rel}:{node.lineno}"
                )
    for sh_rel in WAVE14_SH_FILES:
        sh_path = ROOT / sh_rel
        sh_src = read_text_file(str(sh_path))
        assert sh_src.isascii(), f"Non-ASCII content in {sh_rel}"
        proc = subprocess.run(
            ["bash", "-n", str(sh_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"bash -n failed for {sh_rel}: {proc.stderr}"


def test_run_backtest_cycle_pilot_disabled_ignores_held_lock(tmp_path: Path) -> None:
    """PILOT_DISABLED=true exits 0 immediately even if BACKTEST_CYCLE_LOCK is held by another process."""
    script = ROOT / "offline" / "runners" / "run_backtest_cycle.sh"
    lock_file = tmp_path / "held_cycle.lock"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'exec 8>"$1"; flock -x 8; PILOT_DISABLED=true BACKTEST_CYCLE_LOCK="$1" bash "$2" local',
            "_",
            str(lock_file),
            str(script),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "PILOT_DISABLED" in proc.stdout


def test_hyperliquid_replay_small_epoch_funding_and_step_sl_turnover() -> None:
    """Small synthetic epoch timestamps scale without double-multiplying, and close-step SL records turnover."""
    assert HL_REPLAY.funding_at([(1000, 0.00025)], 1000) == pytest.approx(0.00025)
    # Multi-hour bar starting at epoch 0 sums distinct hourly funding rates (0h, 1h, 2h, 3h)
    ohlc_f = [
        (100.0, 100.2, 99.8, 100.0),
        (100.0, 100.2, 99.8, 100.0),
        (100.0, 100.2, 99.8, 100.0),
    ]
    ts_f = [0, 14400, 28800]
    fund_f = [(0, 0.001), (28800, 0.001), (32400, 0.002), (36000, 0.003), (39600, 0.004)]
    p_fund = HL_REPLAY.StratParams(
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=50.0,
        dca_drop_pct=10.0,
        max_dca_buys=0,
        max_budget=100.0,
        takeprofit_pct=50.0,
        stop_loss_pct=50.0,
        enable_takeprofit=True,
        fee_pct=0.0,
    )
    res_f = HL_REPLAY.run_replay(
        ohlc_f,
        ts=ts_f,
        fund=fund_f,
        sig=["LONG", "LONG", "LONG"],
        params=p_fund,
        direction="long",
        bar_minutes=240.0,
    )
    # Bar 2 accrues 0.001 + 0.002 + 0.003 + 0.004 = 0.01 * 100 notional = -1.0
    assert res_f["funding"] == pytest.approx(-1.0, rel=1e-4)

    # Bar 1 fills entry at 100, then DCA fills at 90 on bar 2 so avg=94.7368; close=89.9 triggers step() SL
    p_step_sl = HL_REPLAY.StratParams(
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=100.0,
        dca_drop_pct=10.0,
        max_dca_buys=1,
        max_budget=200.0,
        takeprofit_pct=20.0,
        stop_loss_pct=5.0,
        enable_takeprofit=True,
        fee_pct=0.0,
    )
    # Bar 0: places entry at 100
    # Bar 1: fills entry at 100 (low=96 > SL=95), close=89 places DCA at 89? Wait: close=89 triggers step() SL!
    # Because low=96 did NOT breach intrabar SL=95? Wait: OHLC envelope requires low <= close!
    # Make entry fill + DCA fill on same bar so avg drops, or partial fill / slippage!
    # With market_slippage_bps=0: Bar 0 places entry at 100. Bar 1 fills entry at open=94.0!
    # Wait: if open=94.0, entry fills at min(100, 94)=94, SL is 94*0.95 = 89.3.
    # How can step(c) trigger SL without intrabar low triggering SL?
    # Intrabar SL checks: if loss_pct(low) >= sl_pct - 1e-9 -> executes intrabar SL.
    # What if Strategy.step(c) triggers via custom subclass / step hook? Verify turnover_notional tracks held_before_step * c:
    orig_check = HL_REPLAY.Strategy._check_stop_loss

    def _deferred_sl(self, price: float) -> bool:
        if math.isclose(price, 95.0):
            return False
        return orig_check(self, price)

    HL_REPLAY.Strategy._check_stop_loss = _deferred_sl
    try:
        res_sl = HL_REPLAY.run_replay(
            [(100.0, 100.5, 99.5, 100.0), (100.0, 100.2, 99.8, 100.0), (100.0, 100.0, 90.0, 92.0)],
            sig=["LONG", "LONG", "LONG"],
            params=p_step_sl,
            direction="long",
            bar_minutes=60.0,
        )
    finally:
        HL_REPLAY.Strategy._check_stop_loss = orig_check
    assert res_sl["cycles"] == 1
    assert res_sl["fills"] == 2
    assert res_sl["turnover_pct"] == pytest.approx(96.0, rel=1e-4)


def test_priceprediction_float64_scaling_and_nonfinite_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """PricePrediction scales extreme float64 prices without float32 overflow and rejects non-finite weights."""
    pred = pp_mod.PricePrediction(window_size=4, epochs=5)
    big_prices = [1.0e39 * (1.0 + 0.02 * i) for i in range(12)]
    pred.process_prices(big_prices)
    assert pred.trained is True
    out = pred.predict_next_price()
    assert out is not None and math.isfinite(out) and out > 0.0

    # Force non-finite lstsq weights in fallback path -> must fail closed (trained=False, predict=None)
    monkeypatch.setattr(pp_mod, "torch", None)
    pred_fb = pp_mod.PricePrediction(window_size=4, epochs=3)
    monkeypatch.setattr(
        pp_mod.np.linalg,
        "lstsq",
        lambda *_a, **_kw: (np.full((5, 1), math.nan), np.empty(0), 0, np.empty(0)),
    )
    pred_fb.process_prices([100.0 + i for i in range(12)])
    assert pred_fb.trained is False
    assert pred_fb.predict_next_price() is None


def test_parallel_and_sync_cache_shell_guards(tmp_path: Path) -> None:
    """run_backtest_parallel.sh resolves caller-relative paths & rejects empty files; sync_cache_to.sh rejects bogus flags & missing tars."""
    par_sh = ROOT / "offline" / "runners" / "run_backtest_parallel.sh"
    missing_proc = subprocess.run(
        ["bash", str(par_sh), str(tmp_path / "does_not_exist.cmds")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_proc.returncode == 2

    empty_cmds = tmp_path / "comments_only.cmds"
    atomic_write_text(str(empty_cmds), "# comment only\n   \n")
    empty_proc = subprocess.run(
        ["bash", str(par_sh), "comments_only.cmds"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert empty_proc.returncode == 1
    assert "0 executable commands" in empty_proc.stderr

    sync_sh = ROOT / "offline" / "runners" / "sync_cache_to.sh"
    bad_flag = subprocess.run(
        ["bash", str(sync_sh), "wsl", "--unknown-flag"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_flag.returncode == 2

    missing_tar_env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "BACKUP_TAR": str(tmp_path / "missing_secrets.tar.gz"),
        "SYNC_CACHE_WSL_LOCK": str(tmp_path / "wsl.lock"),
    }
    missing_tar = subprocess.run(
        ["bash", str(sync_sh), "wsl", "--unpack-backup"],
        env=missing_tar_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_tar.returncode == 1
    assert "backup archive not found" in missing_tar.stderr


def test_t212_and_kraken_grid_report_fail_closed_and_non_overwrite(tmp_path: Path) -> None:
    """Empty/error-only grid reports preserve existing REPORT.md/summary.json and CLI exits 1; _row_matches_current_spec requires ts bounds."""
    # 1. t212 report preserves existing REPORT.md when 0 valid rows exist
    t_dir = tmp_path / "t212_exp"
    t_dir.mkdir()
    existing_t_md = "# Existing Good T212 Report\n"
    atomic_write_text(str(t_dir / "REPORT.md"), existing_t_md)
    atomic_write_jsonl(
        str(t_dir / "results.jsonl"),
        [{"profile": "nvda", "variant": "live", "asset": "NVDA", "error": "boom"}],
    )
    ret_md = t212_report.generate_report(str(t_dir))
    assert "0 runs, 1 errors" in ret_md
    assert read_text_file(str(t_dir / "REPORT.md")) == existing_t_md

    # CLI exits 1 on 0 valid rows
    cli_t = subprocess.run(
        [sys.executable, str(ROOT / "offline/research/t212_continuous_grid/report.py")],
        env={"PATH": "/usr/bin:/bin", "GRID_OUT": str(t_dir)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli_t.returncode == 1

    # 2. kraken report preserves existing REPORT.md & summary.json when 0 valid rows exist
    k_dir = tmp_path / "kraken_exp"
    k_dir.mkdir()
    existing_k_md = "# Existing Good Kraken Report\n"
    atomic_write_text(str(k_dir / "REPORT.md"), existing_k_md)
    atomic_write_json(str(k_dir / "summary.json"), [{"cfg": "live"}], indent=1)
    atomic_write_jsonl(
        str(k_dir / "results.jsonl"),
        [{"asset": "BTC", "cfg": "live", "error": "network timeout"}],
    )
    k_lines = kcg_report.generate_report(str(k_dir))
    assert any("network timeout" in ln for ln in k_lines)
    assert read_text_file(str(k_dir / "REPORT.md")) == existing_k_md

    # 3. kraken grid _row_matches_current_spec rejects rows missing start_ts / end_ts
    live_p, live_pk = kcg_grid.CONFIGS["live"]
    fake_bars = [(1700000000 + i * 14400, 10.0, 10.2, 9.8, 10.0) for i in range(50)]
    kcg_grid._ASSET_CACHE[(str(Path(kcg_grid.DATA).resolve()), "SYNTH")] = (fake_bars, [b[1:] for b in fake_bars], [])
    try:
        stale_no_ts = {
            "asset": "SYNTH",
            "cfg": "live",
            "bars": 50,
            "params_repr": repr(live_p),
            "peak": bool(live_pk),
            "full": {"ret": 1.0, "dd": 1.0},
            "w30": [],
            "w90": [],
        }
        assert kcg_grid._row_matches_current_spec(stale_no_ts, {"SYNTH"}) is False
        valid_with_ts = {
            **stale_no_ts,
            "start_ts": fake_bars[0][0],
            "end_ts": fake_bars[-1][0],
        }
        assert kcg_grid._row_matches_current_spec(valid_with_ts, {"SYNTH"}) is True
    finally:
        kcg_grid._ASSET_CACHE.pop((str(Path(kcg_grid.DATA).resolve()), "SYNTH"), None)

    # 4. t212 fetch CLI rejects unknown flags
    bad_fetch = subprocess.run(
        [sys.executable, str(ROOT / "offline/research/t212_continuous_grid/fetch.py"), "--bogus"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_fetch.returncode == 2
    assert hasattr(t212_grid, "HOME")


def test_short_trend_npz_validation_and_compare_sl_variants_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """short_trend_estimator_benchmark rejects corrupt .npz & traversal symbols; compare_sl_variants exits 1 on 0 valid symbols."""
    # Path traversal symbol returns empty series immediately
    ts_bad, px_bad = st_bench.load_series("../etc/passwd")
    assert len(ts_bad) == 0 and len(px_bad) == 0

    # Corrupt 2D npz returns empty series cleanly instead of propagating invalid shape
    monkeypatch.setattr(st_bench, "D", str(tmp_path))
    monkeypatch.setattr(st_bench, "resolve_cachedb_filepath", lambda p: Path(p))
    np.savez(tmp_path / "cache_365_price_CORRUPT2D.npz", ts=np.zeros((2, 2)), px=np.ones((2, 2)))
    ts_c, px_c = st_bench.load_series("CORRUPT2D")
    assert len(ts_c) == 0 and len(px_c) == 0

    # Non-positive / NaN entries in 1D npz are filtered out cleanly
    np.savez(
        tmp_path / "cache_365_price_FILTERED.npz",
        ts=np.array([100.0, math.nan, 101.0, 102.0], dtype=np.float64),
        px=np.array([10.0, 11.0, -5.0, 12.0], dtype=np.float64),
    )
    ts_ok, px_ok = st_bench.load_series("FILTERED")
    assert ts_ok.tolist() == [100.0, 102.0]
    assert px_ok.tolist() == [10.0, 12.0]

    # _safe_nanmean produces math.nan without RuntimeWarning on all-NaN slices
    assert math.isnan(st_bench._safe_nanmean([math.nan, math.nan]))
    assert st_bench._safe_nanmean([2.0, math.nan, 4.0]) == pytest.approx(3.0)

    # compare_sl_variants exits 1 when all configured symbols lack archive data
    monkeypatch.setattr(cslv, "load_archive_series", lambda *_a, **_kw: [])
    with pytest.raises(SystemExit) as exc_info:
        cslv.main()
    assert exc_info.value.code == 1


def test_cli_tokens_fee_bounds_and_walk_forward_compare_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """validate_bt_cli_token, --fee >= 100, empty --fee-stress, invalid fx_model, and T212/Kraken CLI bounds."""
    assert validate_bt_cli_token("EURUSD=X", "fx") == "EURUSD=X"
    assert validate_bt_cli_token("BRK-B", "sym") == "BRK-B"
    assert validate_bt_cli_token("HYPE/USD", "pair", allow_slash=True) == "HYPE/USD"
    for bad in ("", "  ", "-flag", "../secret", "a/b", "a\\b", "BTC USDT", "BTC\nUSDT"):
        with pytest.raises(ValueError):
            validate_bt_cli_token(bad, "sym")
    for bad_pair in ("/USD", "HYPE/", "A/B/C"):
        with pytest.raises(ValueError):
            validate_bt_cli_token(bad_pair, "pair", allow_slash=True)

    # build_execution_model_from_args rejects fee >= 100
    ap = argparse.ArgumentParser()
    bt_eval.add_walk_forward_cli_args(ap, default_fee=0.1)
    with pytest.raises(SystemExit):
        bt_eval.parse_walk_forward_cli_args(ap, ["--fee", "100"])

    # walk_forward_compare rejects invalid fx_model.usd_per_unit and fee_pct_per_leg >= 100
    with pytest.raises(ValueError, match="fee_pct_per_leg"):
        wfc._build_venue_evaluator("kraken", {"fee_pct_per_leg": 100.0, "strategy_params": {}})
    t212_params = dataclasses.asdict(twf.StratParams.from_env(
        bt_eval.parse_dotenv(str(ROOT / "212trading/config.nvda.env"))
        if hasattr(bt_eval, "parse_dotenv")
        else T212_BT._load_isolated_t212_cfg(
            str(ROOT / "212trading/.env"),
            profile_file=str(ROOT / "212trading/config.nvda.env"),
        )
    ))
    with pytest.raises(ValueError, match="usd_per_unit"):
        wfc._build_venue_evaluator(
            "t212",
            {"strategy_params": t212_params, "fx_model": {"usd_per_unit": -1.0}},
        )
    # Empty fx_model still defaults cleanly to 1.0
    _, base_fee, _ = wfc._build_venue_evaluator("t212", {"strategy_params": t212_params, "fx_model": {}})
    assert base_fee >= 0.0

    # Baseline CLI token guards reject whitespace / leading dash
    with pytest.raises(SystemExit):
        hwf.main(["--coin", "HYPE BAD", "--output-dir", str(tmp_path / "hl")])
    with pytest.raises(SystemExit):
        bwf.main(["--symbol", "BTC BAD", "--output-dir", str(tmp_path / "bn")])
    with pytest.raises(SystemExit):
        fhc.main(["--token", "HYPE BAD", "--output-dir", str(tmp_path / "fhc")])
    with pytest.raises(SystemExit):
        sda.main(["-bad", "7"])

    # 212trading/backtest.py and kraken/backtest.py reject fee >= 100 and invalid entry_pct/dca_pct before network I/O
    with pytest.raises(ValueError, match="entry_pct"):
        T212_BT.simulate([(10.0, 10.5, 9.5, 10.0)] * 25, {"budget": 500.0, "entry_pct": -5.0})
    monkeypatch.setattr(sys, "argv", ["212trading/backtest.py", "--fee", "100"])
    assert T212_BT.main() == 1
    monkeypatch.setattr(sys, "argv", ["212trading/backtest.py", "--profile", "../bad"])
    assert T212_BT.main() == 1
    monkeypatch.setattr(sys, "argv", ["kraken/backtest.py", "--fee", "100"])
    assert KRAKEN_BT.main() == 1
    monkeypatch.setattr(sys, "argv", ["kraken/backtest.py", "--pair", "../bad"])
    assert KRAKEN_BT.main() == 1
