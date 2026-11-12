"""Targeted regression tests for Wave 11 backtest, replay, promotion, and grid fixes."""

import argparse
import ast
import dataclasses
import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from offline.backtests import comparison, promotion
from offline.backtests.datasets import save_dataset
from offline.backtests.hype_candidates import Candidate
from offline.research.hybrid_reentry import report as hr_report
from offline.research.kraken_continuous_grid import grid as kcg_grid
from offline.research.t212_continuous_grid import (
    grid as t212_grid,
    report as t212_report,
)
from offline.runners import (
    calibrate_execution_audit as cea,
    financial_promotion_gate as fpg,
    kraken_financial_compare as kfc,
    t212_walk_forward_baseline as t212_wf,
)
from state_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json_dict,
    read_text_file,
)
from strategies.replay_metrics import (
    add_common_backtest_cli_args,
    parse_bt_cli_float,
    parse_bt_cli_int,
)
from tests.conftest import load_venue_modules

ROOT = Path(__file__).resolve().parents[1]
_T212_BT = load_venue_modules("212trading", "backtest")
_KR_BT = load_venue_modules("kraken", "backtest")

WAVE11_PY_FILES = (
    "212trading/backtest.py",
    "kraken/backtest.py",
    "offline/backtests/comparison.py",
    "offline/backtests/promotion.py",
    "offline/research/hybrid_reentry/report.py",
    "offline/research/kraken_continuous_grid/grid.py",
    "offline/research/kraken_continuous_grid/report.py",
    "offline/research/t212_continuous_grid/fetch.py",
    "offline/research/t212_continuous_grid/grid.py",
    "offline/research/t212_continuous_grid/report.py",
    "offline/runners/calibrate_execution_audit.py",
    "offline/runners/financial_promotion_gate.py",
    "offline/runners/kraken_financial_compare.py",
    "offline/runners/t212_walk_forward_baseline.py",
    "strategies/replay_metrics.py",
    "tests/test_wave11_backtest_fixes.py",
)

WAVE11_SH_FILES = (
    "offline/runners/run_overnight_master_suite.sh",
)


def test_wave11_ascii_ssot_and_layer_hygiene():
    """Verify 100% ASCII, zero raw float/int/open/json calls, DAG isolation, and bash -n."""
    forbidden_calls = {"float", "int", "open"}
    forbidden_json_attrs = {"load", "loads", "dump", "dumps"}
    forbidden_venue_pkgs = {"kraken", "hyperliquid", "binance_api"}

    for rel in (*WAVE11_PY_FILES, *WAVE11_SH_FILES):
        text = read_text_file(ROOT / rel)
        bad_chars = [ch for ch in text if ord(ch) >= 128]
        assert not bad_chars, f"Non-ASCII characters in {rel}: {bad_chars[:5]}"

    for rel in WAVE11_PY_FILES:
        tree = ast.parse(read_text_file(ROOT / rel), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls, (
                        f"{rel}:{node.lineno} uses raw {node.func.id}()"
                    )
                elif (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"
                ):
                    assert node.func.attr not in forbidden_json_attrs, (
                        f"{rel}:{node.lineno} uses raw json.{node.func.attr}()"
                    )
            elif (
                isinstance(node, ast.keyword)
                and node.arg == "type"
                and isinstance(node.value, ast.Name)
                and node.value.id in {"float", "int"}
            ):
                raise AssertionError(f"{rel}:{node.value.lineno} uses raw type={node.value.id}")
            elif rel.startswith("offline/"):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    assert top not in forbidden_venue_pkgs, (
                        f"{rel}:{node.lineno} imports {node.module} directly"
                    )
            elif rel.startswith("strategies/"):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("offline"), (
                        f"{rel}:{node.lineno} upward-imports {node.module}"
                    )

    for rel in WAVE11_SH_FILES:
        res = subprocess.run(
            ["bash", "-n", str(ROOT / rel)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr


def test_resolve_grid_workers_respects_low_ram_and_env(monkeypatch):
    """Verify resolve_grid_workers caps workers to 2 on <=4.5 GiB VMs unless WORKERS is set."""
    monkeypatch.delenv("WORKERS", raising=False)
    with patch.object(kcg_grid, "read_text_file", return_value="MemTotal:        3959808 kB\n"), patch(
        "os.cpu_count", return_value=8
    ):
        assert kcg_grid.resolve_grid_workers() == 2

    with patch.object(kcg_grid, "read_text_file", return_value="MemTotal:       16384000 kB\n"), patch(
        "os.cpu_count", return_value=8
    ):
        assert kcg_grid.resolve_grid_workers() == 4

    monkeypatch.setenv("WORKERS", "3")
    assert kcg_grid.resolve_grid_workers() == 3


def test_t212_grid_spec_matching_empty_row_filter_and_fetch_threshold(tmp_path, monkeypatch):
    """Verify tuple spec_key matching, params_repr stamping, empty-row filtering, and >32 fetch check."""
    results_path = tmp_path / "results.jsonl"
    valid_params, valid_peak = t212_grid.params_for("nvda", "live")
    atomic_write_jsonl(
        results_path,
        [
            {
                "profile": "nvda",
                "variant": "live",
                "asset": "NVDA",
                "params_repr": repr(valid_params),
                "peak": bool(valid_peak),
                "1d": {"ret": 15.0, "dd": 5.0},
            },
            {
                "profile": "nvda",
                "variant": "tp5",
                "asset": "NVDA",
                "params_repr": "STALE_REPR",
                "peak": False,
                "1d": {"ret": 99.0, "dd": 1.0},
            },
            {
                "profile": "nvda",
                "variant": "tp8",
                "asset": "NVDA",
            },
            {
                "profile": "nvda",
                "variant": "tp3",
                "asset": "AAPL",
                "1d": {"ret": 8.0, "dd": 2.0},
            },
        ],
    )
    rep_md = t212_report.generate_report(str(tmp_path))
    assert "2 runs, 0 errors." in rep_md
    assert "| live |" in rep_md
    assert "| tp5 |" not in rep_md
    assert "| tp8 |" not in rep_md
    assert "| tp3 |" in rep_md

    # Verify task() stamps params_repr and peak even when no 1h/1d bars exist
    monkeypatch.setattr(t212_grid, "load", lambda asset, itv: None)
    t212_grid._TASK_CACHE.clear()
    out = t212_grid.task(("nvda", "live", "EMPTY_SYM"))
    assert out["params_repr"] == repr(valid_params)
    assert out["peak"] is bool(valid_peak)

    # Verify fetch.py > 32 bar threshold matches grid.py WARM + 12 (32)
    fetch_src = read_text_file(ROOT / "offline/research/t212_continuous_grid/fetch.py")
    assert ") > 32" in fetch_src


def test_hybrid_reentry_report_validates_test_id_and_fails_closed(tmp_path, monkeypatch):
    """Verify _valid_ablation_row requires non-empty test_id and main() exits 1 on missing/empty data."""
    assert hr_report._valid_ablation_row({"asset": "BTC", "test_id": "Test 0"}) is True
    assert hr_report._valid_ablation_row({"asset": "BTC"}) is False
    assert hr_report._valid_ablation_row({"asset": "BTC", "test_id": "  "}) is False

    out_dir = tmp_path / "hybrid_reentry"
    out_dir.mkdir(parents=True)
    report_file = out_dir / "HYBRID_REENTRY_ABLATION_REPORT.md"
    atomic_write_text(report_file, "EXISTING_VALID_REPORT")

    monkeypatch.setattr(hr_report, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(hr_report, "CRYPTO_JSON", str(out_dir / "ablation_results.json"))
    monkeypatch.setattr(hr_report, "SUMMARY_JSON", str(out_dir / "ablation_summary.json"))
    monkeypatch.setattr(hr_report, "EQUITIES_JSON", str(out_dir / "equities_ablation_results.json"))

    with pytest.raises(SystemExit) as exc_missing:
        hr_report.main()
    assert exc_missing.value.code == 1
    assert read_text_file(report_file) == "EXISTING_VALID_REPORT"

    atomic_write_json(out_dir / "ablation_results.json", [{"asset": "BTC", "error": "boom"}])
    with pytest.raises(SystemExit) as exc_invalid:
        hr_report.main()
    assert exc_invalid.value.code == 1
    assert read_text_file(report_file) == "EXISTING_VALID_REPORT"


def test_t212_walk_forward_baseline_cli_guards_and_default_warmup(tmp_path):
    """Verify --fx-to-usd rejects NaN, --range rejects traversal, and nvda defaults warmup to 12."""
    with pytest.raises(SystemExit) as exc_nan:
        t212_wf.main(["--fx-to-usd", "nan"])
    assert exc_nan.value.code == 2

    with pytest.raises(SystemExit) as exc_rng:
        t212_wf.main(["--range", "../bad"])
    assert exc_rng.value.code == 2

    csv_path = tmp_path / "nvda_1d.csv"
    records = [
        {
            "timestamp": 1700000000 + i * 86400,
            "open": 100.0 + (i % 5),
            "high": 102.0 + (i % 5),
            "low": 98.0 + (i % 5),
            "close": 100.5 + (i % 5),
            "volume": 1000.0,
        }
        for i in range(90)
    ]
    save_dataset(records, csv_path)
    out_dir = tmp_path / "wf_out"
    rc = t212_wf.main(
        [
            "--profile", "nvda",
            "--interval", "1d",
            "--dataset", str(csv_path),
            "--train", "30",
            "--validation", "15",
            "--test", "15",
            "--step", "15",
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    saved = list((out_dir / "nvda").glob("baseline_NVDA_*.json"))
    assert len(saved) == 1
    payload = read_json_dict(saved[0])
    assert payload["window_sizes"]["signal_warmup_bars"] == 12


def test_promotion_sanitizes_nonfinite_deltas_and_gate_writes_clean_json(tmp_path):
    """Verify non-finite aggregate deltas serialize as None instead of crashing atomic_write_json."""
    def _make_report(mean_ret):
        return {
            "dataset": {"sha256": "deadbeef" * 8, "bars": 100},
            "walk_forward": {"train": 10, "validation": 5, "test": 5, "step": 5},
            "scenarios": {
                "central": {
                    "windows": [
                        {"key": i, "return_pct": 1.0, "max_drawdown_pct": 2.0}
                        for i in range(12)
                    ],
                    "aggregate": {
                        "mean_return_pct": mean_ret,
                        "worst_return_pct": None,
                        "worst_max_drawdown_pct": None,
                        "median_calmar": 1.0,
                        "median_sortino": 1.0,
                    },
                }
            },
        }

    base_rep = _make_report(1.0)
    cand_rep = _make_report(None)
    dual = promotion.evaluate_dual_promotion(base_rep, cand_rep)
    ret_scen = dual["return_gate"]["scenarios"]["central"]
    def_scen = dual["defensive_gate"]["scenarios"]["central"]
    assert ret_scen["mean_return_delta_pp"] is None
    assert ret_scen["worst_return_delta_pp"] is None
    assert ret_scen["worst_drawdown_delta_pp"] is None
    assert def_scen["mean_return_delta_pp"] is None
    assert def_scen["worst_return_delta_pp"] is None
    assert def_scen["worst_drawdown_improvement_pp"] is None
    assert dual["promote"] is False

    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    out_file = tmp_path / "gate_out.json"
    atomic_write_json(base_file, base_rep)
    atomic_write_json(cand_file, cand_rep)
    rc = fpg.main([str(base_file), str(cand_file), "--output", str(out_file)])
    assert rc == 1
    written = read_json_dict(out_file)
    assert written["return_gate"]["scenarios"]["central"]["mean_return_delta_pp"] is None


def test_kraken_financial_compare_evaluates_live_first_regardless_of_candidate_order(tmp_path):
    """Verify build_comparison always compares candidates against regenerated live_report."""
    baseline_path = tmp_path / "baseline.json"

    stale_scen = {
        "windows": [{"key": i, "return_pct": -50.0, "max_drawdown_pct": 40.0} for i in range(12)],
        "aggregate": {
            "window_count": 12,
            "mean_return_pct": -50.0,
            "worst_return_pct": -50.0,
            "worst_max_drawdown_pct": 40.0,
            "median_calmar": -1.0,
            "median_sortino": -1.0,
        },
    }
    atomic_write_json(
        baseline_path,
        {
            "dataset": {"sha256": "cafebabe" * 8, "bars": 100},
            "walk_forward": {"train": 10, "validation": 5, "test": 5, "step": 5},
            "strategy_params": dataclasses.asdict(kcg_grid.LIVE),
            "scenarios": {"central": stale_scen},
        },
    )

    regen_scen = {
        "windows": [{"key": i, "return_pct": 10.0, "max_drawdown_pct": 5.0} for i in range(12)],
        "aggregate": {
            "window_count": 12,
            "mean_return_pct": 10.0,
            "worst_return_pct": 10.0,
            "worst_max_drawdown_pct": 5.0,
            "median_calmar": 2.0,
            "median_sortino": 2.0,
        },
    }

    def _fake_build_report(_args, *, params, candidate_name):
        return {
            "code": {"commit": "test"},
            "dataset": {"sha256": "cafebabe" * 8, "bars": 100},
            "walk_forward": {"train": 10, "validation": 5, "test": 5, "step": 5},
            "scenarios": {"central": regen_scen},
        }

    args = argparse.Namespace(baseline=baseline_path, candidate=None, strict_baseline=False)
    reordered = [
        Candidate("cand_first", "Candidate before live", {"takeprofit_pct": 3.0}),
        Candidate("live", "Live profile", {}),
    ]
    with patch.object(kfc, "financial_priority_candidates", return_value=reordered), patch.object(
        kfc.benchmark, "build_report", side_effect=_fake_build_report
    ):
        comp = kfc.build_comparison(args)

    assert list(comp["candidates"].keys()) == ["cand_first", "live"]
    cand_gate = comp["candidates"]["cand_first"]["promotion_gate"]["return_gate"]["scenarios"]["central"]
    assert cand_gate["mean_return_delta_pp"] == pytest.approx(0.0)


def test_calibrate_execution_audit_streaming_and_intent_matching(tmp_path):
    """Verify streaming JSONL load, stripped intent_id matching, blank venue follow-up, and bool rejection."""
    audit_file = tmp_path / "execution_audit_w11.jsonl"
    atomic_write_jsonl(
        audit_file,
        [
            {"venue": " Kraken ", "intent_id": " i1 ", "event": "submit_accepted"},
            {"venue": "", "intent_id": "i1", "event": "order_terminal"},
            {"venue": None, "intent_id": " i1 ", "event": "fill"},
            {"venue": True, "intent_id": "i1", "event": "bogus_bool_venue"},
        ],
    )
    loaded = cea._load([audit_file], "kraken")
    assert [ev["event"] for ev in loaded] == ["submit_accepted", "order_terminal", "fill"]

    bad_file = tmp_path / "bad.jsonl"
    atomic_write_text(bad_file, '{"venue": "kraken"}\nnot-json\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        cea._load([bad_file], None)


def test_comparison_markdown_coerces_malformed_counts():
    """Verify format_comparison_markdown coerces malformed count fields via safe_nonneg_int."""
    md = comparison.format_comparison_markdown(
        {
            "candidates": {
                "live": {
                    "summary": {
                        "mean_return_pct": 2.5,
                        "median_return_pct": 2.0,
                        "worst_return_pct": -1.0,
                        "worst_max_drawdown_pct": 3.0,
                        "positive_windows": "bad",
                        "window_count": None,
                        "total_cycles": math.nan,
                    },
                    "relative_to_live": {
                        "wins_vs_live": True,
                        "ties_vs_live": -4,
                        "losses_vs_live": "oops",
                    },
                }
            },
            "fee_stress": {
                "0.40": {
                    "summary": {
                        "mean_return_pct": 1.0,
                        "worst_return_pct": -2.0,
                        "worst_max_drawdown_pct": 4.0,
                        "positive_windows": None,
                        "window_count": "bad",
                    }
                }
            },
        }
    )
    assert "| 0/0 | 0/0/0 | 0 |" in md
    assert "| 0.40 | +1.000 | -2.000 | 4.000 | 0/0 |" in md


def test_backtest_cli_parsers_and_t212_kraken_subbar_scaling(monkeypatch):
    """Verify finite/exact CLI parsers, explicit entry/dca validation, and sub-bar timeout scaling."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bt_cli_float("nan")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bt_cli_float("inf")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bt_cli_int("3.5")
    assert parse_bt_cli_float("1.25") == pytest.approx(1.25)
    assert parse_bt_cli_int("240") == 240

    ap = argparse.ArgumentParser()
    add_common_backtest_cli_args(ap)
    with pytest.raises(SystemExit):
        ap.parse_args(["--tp", "nan"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--maxdca", "2.5"])

    bars = [(100.0, 101.0, 99.0, 100.2) for _ in range(25)]
    with pytest.raises(ValueError, match="entry must be finite and positive"):
        _T212_BT.simulate(bars, {"entry": 0, "budget": 500.0, "bar_minutes": 60})
    with pytest.raises(ValueError, match="dca must be finite and positive"):
        _T212_BT.simulate(bars, {"dca": -10, "budget": 500.0, "bar_minutes": 60})

    captured_t212 = {}

    def _spy_t212_replay(ohlc, params, **kwargs):
        captured_t212["params"] = params
        captured_t212["kwargs"] = kwargs
        return {"total": 0.0, "net": 0.0, "cycles": 0, "wins": 0}

    monkeypatch.setattr(_T212_BT, "run_replay", _spy_t212_replay)
    _T212_BT.simulate(
        bars,
        {"budget": 500.0, "bar_minutes": 60, "timestamps": [1700000000 + i * 3600 for i in range(25)]},
    )
    assert captured_t212["params"].check_minutes == pytest.approx(60.0)
    assert captured_t212["params"].sl_trend_defer_max_sec == pytest.approx(220.0 * 60.0)
    assert len(captured_t212["kwargs"]["timestamps"]) == 25

    # Verify kraken/backtest.py keeps _bt_params unscaled while simulate() scales sub-bar timeouts
    unscaled_kr = _KR_BT._bt_params({"interval": 240})
    assert unscaled_kr.check_minutes == pytest.approx(kcg_grid.LIVE.check_minutes)
    assert unscaled_kr.check_minutes < 240.0

    captured_kr = {}

    def _spy_kr_replay(ohlc, params, **kwargs):
        captured_kr["params"] = params
        captured_kr["kwargs"] = kwargs
        return {"total": 0.0, "net": 0.0, "cycles": 0, "wins": 0}

    monkeypatch.setattr(_KR_BT.rp, "run_replay", _spy_kr_replay)
    _KR_BT.simulate(
        bars,
        {"interval": 240, "timestamps": [1700000000 + i * 14400 for i in range(25)]},
    )
    assert captured_kr["params"].check_minutes == pytest.approx(240.0)
    assert captured_kr["params"].sl_trend_defer_max_sec == pytest.approx(
        unscaled_kr.sl_trend_defer_max_sec * (240.0 / unscaled_kr.check_minutes)
    )
    assert len(captured_kr["kwargs"]["timestamps"]) == 25
