"""Regression tests for Wave 12 backtesting, calibration, grid, and runner fixes."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest

from offline.backtests.execution_calibration import calibrate_execution_events
from offline.research import chop_weight_backtest as chop_bt
from offline.research import compare_mhdsl_reentry_replay as cmp_mhdsl
from offline.research.kraken_continuous_grid import (
    fetch_data as kcg_fetch,
    grid as kcg_grid,
    grid2 as kcg_grid2,
    grid3 as kcg_grid3,
    report2 as kcg_report2,
    report3 as kcg_report3,
)
from offline.research.monitortrades_backtest import (
    run_replay_backtest as rrb,
    sweep_mt_models as smm,
)
from offline.runners import kraken_financial_benchmark as kfb
from state_io import atomic_write_jsonl, atomic_write_text, read_text_file
from strategies import spot_engine as strat

ROOT = Path(__file__).resolve().parents[1]

WAVE12_PY_FILES = (
    "offline/runners/kraken_financial_benchmark.py",
    "offline/backtests/execution_calibration.py",
    "offline/research/compare_mhdsl_reentry_replay.py",
    "offline/research/intelligence_backtest.py",
    "offline/research/chop_weight_backtest.py",
    "offline/research/kraken_continuous_grid/fetch_data.py",
    "offline/research/kraken_continuous_grid/grid2.py",
    "offline/research/kraken_continuous_grid/grid3.py",
    "offline/research/kraken_continuous_grid/report2.py",
    "offline/research/kraken_continuous_grid/report3.py",
    "offline/research/monitortrades_backtest/run_replay_backtest.py",
    "offline/research/monitortrades_backtest/sweep_mt_models.py",
    "offline/research/monitortrades_backtest/sweep_multiwindow_mt.py",
    "offline/research/verify_optimizations_parity.py",
    "tests/test_wave12_backtest_fixes.py",
)

WAVE12_SH_FILES = (
    "offline/runners/run_backtest_cycle.sh",
)


def test_wave12_ascii_ast_and_shell_hygiene() -> None:
    """Enforce 100% ASCII, zero raw float/int/open calls, and clean bash -n syntax."""
    for rel in WAVE12_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"float", "int", "open"}, (
                    f"Forbidden raw {node.func.id}() in {rel}:{node.lineno}"
                )
    for sh_rel in WAVE12_SH_FILES:
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


def test_kraken_financial_benchmark_scales_subbar_timeouts_and_validates_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_scenario_windows scales sub-bar timeouts via prepare_spot_interval_params; CLI rejects non-finite values."""
    captured: list[strat.StratParams] = []
    orig_make = kfb.kraken_replay.make_spot_replay_adapter

    def _spy_make(params, **kwargs):
        captured.append(params)
        return orig_make(params, **kwargs)

    monkeypatch.setattr(kfb.kraken_replay, "make_spot_replay_adapter", _spy_make)
    records = [
        {"timestamp": 1700000000 + i * 14400, "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0}
        for i in range(40)
    ]
    raw_params = dataclasses.replace(
        kcg_grid.LIVE,
        check_minutes=2.0,
        sl_trend_defer_max_sec=220.0,
        tp_stagnation_max_sec=1080.0,
    )
    central, _ = kfb.default_scenarios()
    windows = kfb._scenario_windows(
        records,
        raw_params,
        central,
        interval=240,
        train=10,
        validation=10,
        test=10,
        step=10,
        warmup=5,
    )
    assert len(windows) == 2
    assert len(captured) == 1
    assert captured[0].check_minutes == pytest.approx(240.0)
    assert captured[0].sl_trend_defer_max_sec == pytest.approx(220.0 * 120.0)
    assert captured[0].tp_stagnation_max_sec == pytest.approx(1080.0 * 120.0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("~/env"))
    parser.add_argument("--config-file", type=Path, default=Path("~/cfg"))
    kfb.add_financial_benchmark_common_args(parser)
    parser.add_argument("--verify", type=Path, default=Path("~/ver.json"))
    with pytest.raises(SystemExit):
        parser.parse_args(["--regime-threshold", "nan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--train", "12.5"])
    parsed = parser.parse_args([])
    kfb.validate_financial_benchmark_args(
        parser, parsed, path_attrs=("env_file", "config_file", "verify")
    )
    assert parsed.verify.is_absolute()
    assert parsed.env_file.is_absolute()
    assert parsed.config_file.is_absolute()


def test_execution_calibration_missing_ts_partial_cost_and_false_cid() -> None:
    """Missing timestamps do not fabricate 0s latency, partial multi-leg cost/fee returns None, and False cid is invalid."""
    events = [
        # Intent 1: retry leg B missing cost -> avg_fill and fee_bps must be None (no fake -6000 bps slippage)
        {
            "ts": 100.0,
            "event": "submit_requested",
            "intent_id": "partial-cost",
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "side": "BUY",
            "qty": 1.0,
            "price": 100.0,
            "market": False,
            "client_order_id": False,
        },
        {"ts": None, "event": "submit_accepted", "intent_id": "partial-cost", "order_id": "OA"},
        {
            "ts": 105.0,
            "event": "order_terminal_retry",
            "intent_id": "partial-cost",
            "order_id": "OA",
            "filled_qty": 0.4,
            "cost": 40.0,
            "fee": 0.04,
        },
        {"ts": 106.0, "event": "submit_accepted", "intent_id": "partial-cost", "order_id": "OB"},
        {
            "ts": 110.0,
            "event": "order_terminal",
            "intent_id": "partial-cost",
            "order_id": "OB",
            "filled_qty": 0.6,
            "cost": None,
            "fee": 0.06,
        },
        # Intent 2: first leg cancelled with 0 fill & None fee, second leg fully filled with known cost & fee
        {
            "ts": 200.0,
            "event": "submit_requested",
            "intent_id": "unfilled-leg-1",
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "side": "BUY",
            "qty": 1.0,
            "price": 100.0,
            "market": False,
            "client_order_id": "a" * 32,
        },
        {"ts": 201.0, "event": "submit_accepted", "intent_id": "unfilled-leg-1", "order_id": "OC"},
        {
            "ts": 202.0,
            "event": "order_terminal_retry",
            "intent_id": "unfilled-leg-1",
            "order_id": "OC",
            "filled_qty": 0.0,
            "cost": 0.0,
            "fee": None,
        },
        {"ts": 203.0, "event": "submit_accepted", "intent_id": "unfilled-leg-1", "order_id": "OD"},
        {
            "ts": 205.0,
            "event": "order_terminal",
            "intent_id": "unfilled-leg-1",
            "order_id": "OD",
            "filled_qty": 1.0,
            "cost": 100.0,
            "fee": 0.26,
        },
    ]
    rep = calibrate_execution_events(events)
    by_intent = {o["intent_id"]: o for o in rep["orders"]}

    p_ord = by_intent["partial-cost"]
    assert p_ord["filled_qty"] == pytest.approx(1.0)
    assert p_ord["first_fill_latency_s"] is None
    assert p_ord["limit_fill_deviation_bps"] is None
    assert p_ord["fee_bps"] is None
    assert p_ord["client_order_id_valid"] is False

    u_ord = by_intent["unfilled-leg-1"]
    assert u_ord["filled_qty"] == pytest.approx(1.0)
    assert u_ord["limit_fill_deviation_bps"] == pytest.approx(0.0)
    assert u_ord["fee_bps"] == pytest.approx(26.0)
    assert u_ord["first_fill_latency_s"] == pytest.approx(4.0)

    cid_val = rep["client_order_id_validation"]
    assert cid_val["supported_accepted_orders"] == 2
    assert cid_val["with_client_order_id"] == 2
    assert cid_val["valid_client_order_ids"] == 1
    assert cid_val["invalid_client_order_ids"] == 1
    assert cid_val["missing_client_order_ids"] == 0


def test_compare_mhdsl_reentry_replay_scales_check_minutes_and_defer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compare_mhdsl_reentry_replay scales check_minutes=2.0 -> 240.0 and sl_trend_defer_max_sec -> 43200s."""
    csv_p = tmp_path / "hype.csv"
    lines = ["timestamp,open,high,low,close"]
    for i in range(70):
        ts = 1700000000 + i * 14400
        lines.append(f"{ts},10.0,10.2,9.8,10.0")
    atomic_write_text(str(csv_p), "\n".join(lines) + "\n")

    seen_params: list[strat.StratParams] = []
    orig_replay = cmp_mhdsl.run_replay

    def _spy_replay(ohlc, params, **kwargs):
        seen_params.append(params)
        return orig_replay(ohlc, params, **kwargs)

    monkeypatch.setattr(cmp_mhdsl, "run_replay", _spy_replay)
    monkeypatch.setattr(cmp_mhdsl, "prewarm_trend_series", lambda *a, **kw: None)
    res = cmp_mhdsl.run_comparison(str(csv_p))
    assert len(res) == 4
    prop_runs = [p for p in seen_params if p.sl_mhdsl_enabled]
    assert len(prop_runs) == 4
    assert prop_runs[0].check_minutes == pytest.approx(240.0)
    assert prop_runs[0].sl_trend_defer_max_sec == pytest.approx(43200.0)


def test_grid_fetch_report2_and_report3_metadata_and_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch_data validates CSV bar count, report2 unions assets & dedups select/wave1, report3 filters stale specs."""
    bad_csv = tmp_path / "BAD_240m.csv"
    atomic_write_text(str(bad_csv), "timestamp,open,high,low,close\n" + ("not,a,valid,csv,row\n" * 80))
    assert kcg_fetch._has_usable_csv(str(bad_csv)) is False

    short_csv = tmp_path / "SHORT_240m.csv"
    short_lines = ["timestamp,open,high,low,close"] + [
        f"{1700000000 + i * 14400},10.0,10.5,9.5,10.1" for i in range(10)
    ]
    atomic_write_text(str(short_csv), "\n".join(short_lines) + "\n")
    assert kcg_fetch._has_usable_csv(str(short_csv)) is False

    good_csv = tmp_path / "GOOD_240m.csv"
    good_lines = ["timestamp,open,high,low,close"] + [
        f"{1700000000 + i * 14400},10.0,10.5,9.5,10.1" for i in range(50)
    ]
    atomic_write_text(str(good_csv), "\n".join(good_lines) + "\n")
    assert kcg_fetch._has_usable_csv(str(good_csv)) is True

    # report2: union of assets across wave1 and wave2 + deduplicated select & wave1 last-wins
    exp_dir = tmp_path / "exp2"
    exp_dir.mkdir()
    live_p, live_pk = kcg_grid2.ALL["live"]
    atomic_write_jsonl(
        str(exp_dir / "results.jsonl"),
        [
            {
                "asset": "BTC",
                "cfg": "live",
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "full": {"ret": -99.0, "dd": 50.0},
                "w90": [[1.0, 2.0, 0.5, 1]],
            },
            {
                "asset": "BTC",
                "cfg": "live",
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "full": {"ret": 12.0, "dd": 5.0},
                "w90": [[2.0, 2.0, 0.5, 1]],
            },
            {
                "asset": "ETH",
                "cfg": "live",
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "full": {"ret": 8.0, "dd": 4.0},
                "w90": [[1.5, 2.0, 0.5, 1]],
            },
        ],
    )
    ov_p, ov_pk = kcg_grid2.ALL["ov500_t7"]
    atomic_write_jsonl(
        str(exp_dir / "results2.jsonl"),
        [
            # Partial wave2 config (only BTC, missing ETH) -> must NOT appear in complete summary table
            {
                "kind": "views",
                "fee": "std",
                "asset": "BTC",
                "cfg": "ov500_t7",
                "params_repr": repr(ov_p),
                "peak": bool(ov_pk),
                "full": {"ret": 20.0, "dd": 6.0},
                "w90": [[3.0, 2.0, 0.5, 1]],
            },
            # Duplicate select rows for BTC -> last wins, not summed twice
            {
                "kind": "select",
                "asset": "BTC",
                "steps": [{"t": 1, "pick": "live", "pick_ret": 100.0, "oos": {"live": 100.0}}],
            },
            {
                "kind": "select",
                "asset": "BTC",
                "steps": [{"t": 1, "pick": "live", "pick_ret": 5.0, "oos": {"live": 5.0}}],
            },
        ],
    )
    r2_lines = kcg_report2.generate_report(str(exp_dir))
    r2_text = "\n".join(r2_lines)
    assert "| live | std | 0/2 | +10.0 |" in r2_text
    assert "| ov500_t7 | std |" not in r2_text
    assert "| SELECTOR | +5.0 |" in r2_text

    # report3: rejects stale params_repr, never prints -1e300, and handles non-numeric per-asset cell
    exp3 = tmp_path / "exp3"
    exp3.mkdir()
    atomic_write_jsonl(
        str(exp3 / "results3.jsonl"),
        [
            {
                "asset": "BTC",
                "cfg": "live",
                "params_repr": repr(live_p),
                "peak": bool(live_pk),
                "0.26": {"ret": 15.0, "dd": 5.0, "refused": 0, "min_cash": 3500.0},
                "0.40": {"ret": 12.0, "dd": 6.0, "refused": 0, "min_cash": 3400.0},
            },
            {
                "asset": "BTC",
                "cfg": "sl0",
                "params_repr": "STALE_PARAMS_REPR",
                "peak": False,
                "0.26": {"ret": 99.0, "dd": 1.0, "refused": 0, "min_cash": 3900.0},
                "0.40": {"ret": 95.0, "dd": 1.0, "refused": 0, "min_cash": 3900.0},
            },
        ],
    )
    r3_lines = kcg_report3.generate_report(str(exp3))
    r3_text = "\n".join(r3_lines)
    assert "| live |" in r3_text
    assert "sl0" not in r3_text
    assert "-1e+300" not in r3_text and "-1e300" not in r3_text

    # grid3.task stamps metadata fields
    monkeypatch.setattr(
        kcg_grid,
        "load_asset",
        lambda _a: ([(1700000000 + i * 14400, 10.0, 10.2, 9.8, 10.0) for i in range(45)], [], []),
    )
    kcg_grid3._TASK3_CACHE.clear()
    g3_row = kcg_grid3.task(("SYNTH", "live"))
    assert g3_row["bars"] == 45
    assert g3_row["start_ts"] == 1700000000
    assert g3_row["params_repr"] == repr(live_p)
    assert g3_row["peak"] is bool(live_pk)


def test_run_backtest_cycle_invalid_mode_and_pilot_disabled(tmp_path: Path) -> None:
    """run_backtest_cycle.sh rejects unknown modes with exit 2 and exits 0 cleanly when PILOT_DISABLED=true."""
    script = ROOT / "offline" / "runners" / "run_backtest_cycle.sh"
    bad_proc = subprocess.run(
        ["bash", str(script), "bogus_mode"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_proc.returncode == 2
    assert "invalid mode" in bad_proc.stderr

    env = dict(**{"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PILOT_DISABLED": "true"})
    disabled_proc = subprocess.run(
        ["bash", str(script), "local"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert disabled_proc.returncode == 0
    assert "PILOT_DISABLED" in disabled_proc.stdout


def test_monitortrades_npz_validation_worker_cap_and_chop_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """load_fresh_365_npz_arrays filters non-finite/non-positive rows; worker cap honors low RAM; chop CLI rejects floats."""
    npz_file = tmp_path / "cache_365_price_TESTSYM.npz"
    np.savez(
        npz_file,
        ts=np.array([1000.0, math.nan, 1002.0, 1003.0], dtype=np.float64),
        px=np.array([50.0, 51.0, 0.0, 52.5], dtype=np.float64),
    )
    monkeypatch.setattr(smm, "cache_path", lambda name: str(tmp_path / name))
    loaded = smm.load_fresh_365_npz_arrays("TESTSYM")
    assert loaded is not None
    ts_arr, px_arr = loaded
    assert ts_arr.tolist() == [1000.0, 1003.0]
    assert px_arr.tolist() == [50.0, 52.5]

    # 2D corrupt npz returns None cleanly
    bad_npz = tmp_path / "cache_365_price_BAD2D.npz"
    np.savez(bad_npz, ts=np.zeros((2, 2)), px=np.ones((2, 2)))
    assert smm.load_fresh_365_npz_arrays("BAD2D") is None

    # print_mt_buy_hold_footer handles None / non-numeric entries and negative col_width
    smm.print_mt_buy_hold_footer(["BTCUSDC"], {"BTCUSDC": None}, "hdr", col_width=0)
    out = capsys.readouterr().out
    assert "+0.00" in out

    # Low-RAM worker cap in run_replay_backtest
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.setattr(rrb.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(rrb, "read_text_file", lambda *_a, **_kw: "MemTotal:        3959808 kB\n")
    assert rrb._resolve_replay_worker_cap(False) == 2

    # parse_cli_int used in chop_weight_backtest rejects float string
    with pytest.raises(argparse.ArgumentTypeError):
        chop_bt.parse_cli_int("15.5")
