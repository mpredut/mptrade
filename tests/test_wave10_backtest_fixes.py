"""Targeted regression tests for Wave 10 backtest, replay, walk-forward, and sweep fixes."""

import ast
import dataclasses
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from offline.backtests.datasets import dataset_sha256, save_dataset
from offline.backtests.evaluation import (
    ReplayContext,
    load_venue_module,
    load_venue_replay_module,
)
from offline.research import comprehensive_strategy_sweep as css
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
from offline.research.hybrid_reentry import harness as hr_harness
from offline.research.kraken_continuous_grid import grid as kcg_grid, report as kcg_report
from offline.research.monitortrades_backtest import compare_sl_variants as csv_mod
from offline.runners import (
    binance_walk_forward_baseline as bwf,
    hyperliquid_walk_forward_baseline as hwf,
    kraken_walk_forward_baseline as kwf,
    walk_forward_compare as wfc,
)
from state_io import atomic_write_jsonl, atomic_write_text, read_text_file
from strategies import spot_replay
from strategies.spot_engine import StratParams as SpotStratParams

ROOT = Path(__file__).resolve().parents[1]
_HL_MD = load_venue_module("hyperliquid", "market_data.py", "test_w10_hl_md")
_HL_REPLAY = load_venue_replay_module("hyperliquid", "test_w10_hl_replay")
_HL_BT = load_venue_module("hyperliquid", "backtest.py", "test_w10_hl_bt")
_T212_REPLAY = load_venue_replay_module("t212", "test_w10_t212_replay")

WAVE10_PY_FILES = (
    "212trading/replay.py",
    "hyperliquid/backtest.py",
    "hyperliquid/market_data.py",
    "hyperliquid/replay.py",
    "offline/backtests/evaluation.py",
    "offline/research/comprehensive_strategy_sweep.py",
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
    "offline/research/hybrid_reentry/harness.py",
    "offline/research/kraken_continuous_grid/report.py",
    "offline/research/monitortrades_backtest/compare_sl_variants.py",
    "offline/runners/binance_walk_forward_baseline.py",
    "offline/runners/hyperliquid_walk_forward_baseline.py",
    "offline/runners/kraken_walk_forward_baseline.py",
    "offline/runners/walk_forward_compare.py",
    "strategies/spot_replay.py",
    "tests/test_wave10_backtest_fixes.py",
)

WAVE10_TEXT_FILES = (
    "212trading/config.nvda.env",
    "212trading/config.rgnt.env",
    "212trading/config.spcx.env",
    "offline/research/kraken_continuous_grid/run_all.sh",
    "offline/runners/README.md",
)


def test_wave10_ascii_and_ssot_ast_hygiene():
    """Verify 100% ASCII, zero raw float/int/open/json calls, and bash -n syntax on Wave 10 targets."""
    forbidden_calls = {"float", "int", "open"}
    forbidden_json_attrs = {"load", "loads", "dump", "dumps"}
    forbidden_venue_pkgs = {"kraken", "hyperliquid", "binance_api"}

    for rel in (*WAVE10_PY_FILES, *WAVE10_TEXT_FILES):
        path = ROOT / rel
        text = read_text_file(path)
        bad_chars = [ch for ch in text if ord(ch) >= 128]
        assert not bad_chars, f"Non-ASCII characters in {rel}: {bad_chars[:5]}"

    for rel in WAVE10_PY_FILES:
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

    res = subprocess.run(
        ["bash", "-n", str(ROOT / "offline/research/kraken_continuous_grid/run_all.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_fetch_funding_history_paginates_beyond_500_hours():
    """fetch_funding_history must paginate forward past 500 hourly entries and respect end_ms."""
    calls: list[dict] = []

    def fake_post_info(payload, timeout=25):
        calls.append(payload)
        start = payload["startTime"]
        if start == 1000:
            return [{"time": 1000 + idx * 10, "fundingRate": "0.0001"} for idx in range(500)]
        return [{"time": 6000 + idx * 10, "fundingRate": "0.0002"} for idx in range(120)]

    with patch.object(_HL_MD, "_post_info", side_effect=fake_post_info):
        rows = _HL_MD.fetch_funding_history("HYPE", 1000, end_ms=7050)

    assert len(calls) == 2
    assert calls[0]["startTime"] == 1000
    assert calls[0]["endTime"] == 7050
    assert calls[1]["startTime"] == 5991
    assert len(rows) == 606
    assert rows[0] == (1000, pytest.approx(0.0001))
    assert rows[-1] == (7050, pytest.approx(0.0002))


def test_scale_subbar_timeout_params_is_idempotent_and_applied_in_t212_adapter():
    """scale_subbar_timeout_params must be idempotent and scale T212 adapter sub-bar timeouts."""
    base = SpotStratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.2,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        check_minutes=2.0,
        takeprofit_pct=4.0,
        max_budget=500.0,
        max_dca_buys=5,
        enable_takeprofit=True,
        order_ttl_min=10.0,
        stop_loss_pct=15.0,
        sl_trend_defer_max_sec=220.0,
        tp_stagnation_max_sec=1080.0,
    )
    once = spot_replay.scale_subbar_timeout_params(base, 240.0)
    twice = spot_replay.scale_subbar_timeout_params(once, 240.0)
    assert once.check_minutes == pytest.approx(240.0)
    assert once.sl_trend_defer_max_sec == pytest.approx(220.0 * 120.0)
    assert once.tp_stagnation_max_sec == pytest.approx(1080.0 * 120.0)
    assert twice == once

    t212_params = _T212_REPLAY._strat.StratParams.from_dict(
        {
            "currency": "USD",
            "yahoo_sym": "NVDA",
            "entry_amount": 50.0,
            "dca_amount": 30.0,
            "entry_discount_pct": 0.2,
            "dca_drop_pct": 2.0,
            "check_minutes": 0.5,
            "takeprofit_pct": 5.0,
            "max_budget": 300.0,
            "max_dca_buys": 4,
            "order_ttl_min": 10.0,
            "stop_loss_pct": 20.0,
            "fx_fee_pct": 0.15,
            "sl_trend_defer_max_sec": 220.0,
            "tp_stagnation_max_sec": 1080.0,
        }
    )
    with patch.object(_T212_REPLAY, "run_replay", wraps=_T212_REPLAY.run_replay) as spy_run:
        adapter = _T212_REPLAY.make_t212_replay_adapter(t212_params, bar_minutes=60.0)
        bars = [(100.0, 101.0, 99.0, 100.5) for _ in range(10)]
        ts = tuple(1_700_000_000 + i * 3600 for i in range(10))
        adapter(bars, (), ReplayContext(timestamps=ts, warmup_timestamps=()))
        eff = spy_run.call_args.args[1]
        assert eff.check_minutes == pytest.approx(60.0)
        assert eff.sl_trend_defer_max_sec == pytest.approx(220.0 * 120.0)
        assert eff.tp_stagnation_max_sec == pytest.approx(1080.0 * 120.0)


def test_hyperliquid_replay_signal_gate_fails_closed_when_sig_omitted():
    """Perp replay with signal_gate=True and sig=None must fail closed instead of bypassing."""
    ohlc = [(100.0, 100.5, 98.0, 99.5) for _ in range(12)]
    base_cfg = {
        "currency": "USDC",
        "entry": 100.0,
        "dca": 50.0,
        "disc": 0.5,
        "drop": 2.0,
        "tp": 3.0,
        "maxdca": 3,
        "budget": 250.0,
        "fee": 0.035,
        "sl": 15.0,
    }
    gated = _HL_REPLAY.StratParams.from_dict({**base_cfg, "signal_gate": True}, direction="long")
    ungated = dataclasses.replace(gated, signal_gate=False)

    res_gated = _HL_REPLAY.run_replay(ohlc, None, None, None, gated, direction="long", coin="HYPE")
    res_ungated = _HL_REPLAY.run_replay(ohlc, None, None, None, ungated, direction="long", coin="HYPE")
    assert res_gated["fills"] == 0
    assert res_ungated["fills"] > 0


def test_hyperliquid_walk_forward_baseline_routes_spot_vs_perp_pair():
    """fetch_ohlc_candles in hyperliquid_walk_forward_baseline resolves @107 for spot HYPE."""
    with patch.object(hwf.mkt, "fetch_ohlc_candles", return_value=[]) as mock_mkt:
        hwf.fetch_ohlc_candles("HYPE", 240, engine="spot", quote="USDC", start_ms=0)
        assert mock_mkt.call_args.args[0] == "@107"

        hwf.fetch_ohlc_candles("HYPE", 240, engine="perp", quote="USDC", start_ms=0)
        assert mock_mkt.call_args.args[0] == "HYPE"


def test_walk_forward_compare_fx_integrity_and_anchored_train(tmp_path):
    """walk_forward_compare enforces historical_asof FX file/hash and forwards anchored_train."""
    base_report = {
        "strategy_params": {
            "currency": "USD",
            "yahoo_sym": "NVDA",
            "entry_amount": 50.0,
            "dca_amount": 30.0,
            "entry_discount_pct": 0.2,
            "dca_drop_pct": 2.0,
            "check_minutes": 5.0,
            "takeprofit_pct": 5.0,
            "max_budget": 300.0,
            "max_dca_buys": 4,
            "order_ttl_min": 10.0,
            "stop_loss_pct": 15.0,
            "fx_fee_pct": 0.15,
        },
        "fx_model": {"mode": "historical_asof"},
    }
    with pytest.raises(ValueError, match="requires frozen_file"):
        wfc._build_venue_evaluator("t212", base_report, baseline_dir=tmp_path)

    fx_csv = tmp_path / "fx.csv"
    fx_rows = [
        {"timestamp": 1_700_000_000 + i * 86400, "open": 4.5, "high": 4.6, "low": 4.4, "close": 4.55}
        for i in range(5)
    ]
    save_dataset(fx_rows, fx_csv)
    bad_hash_report = {
        **base_report,
        "fx_model": {"mode": "historical_asof", "frozen_file": str(fx_csv), "sha256": "deadbeef"},
    }
    with pytest.raises(ValueError, match="sha256 mismatch"):
        wfc._build_venue_evaluator("t212", bad_hash_report, baseline_dir=tmp_path)

    good_hash_report = {
        **base_report,
        "fx_model": {
            "mode": "historical_asof",
            "frozen_file": str(fx_csv),
            "sha256": dataset_sha256(fx_rows),
        },
    }
    _, fee, _ = wfc._build_venue_evaluator("t212", good_hash_report, baseline_dir=tmp_path)
    assert fee == pytest.approx(0.15)

    records = [
        {"timestamp": 1_700_000_000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.2}
        for i in range(25)
    ]
    seen_starts: list[str] = []

    def stub_seg(seg_records, *_args, **_kwargs):
        seen_starts.append(str(seg_records[0]["timestamp"]))
        return {
            "start_utc": "2023-11-14T00:00:00+00:00",
            "end_utc": "2023-11-14T02:00:00+00:00",
            "buy_hold_return_pct": 0.0,
            "metrics": {"return_pct": 0.5, "max_drawdown_pct": 0.1, "cycles": 1, "fills": 2},
        }

    datasets = {
        60: (
            records,
            {"train": 8, "validation": 3, "test": 2, "step": 2, "anchored_train": True},
            0,
        )
    }
    windows = wfc._candidate_windows(datasets, None, 0.1, None, segment_fn=stub_seg)
    assert len(windows) == 7
    assert seen_starts[0] == str(records[11]["timestamp"])
    assert seen_starts[1] == str(records[13]["timestamp"])


def test_binance_walk_forward_baseline_handles_missing_vs_incomplete_strat_keys(tmp_path):
    """binance_walk_forward_baseline falls back when no STRAT_* keys exist, and errors cleanly on incomplete STRAT_*."""
    empty_cfg = tmp_path / "empty.env"
    empty_env = tmp_path / "dot.env"
    atomic_write_text(empty_cfg, "SYMBOL=BTCUSDT\n")
    atomic_write_text(empty_env, "")

    with patch.object(bwf, "fetch_ohlc_candles", return_value=[]):
        rc = bwf.main(
            [
                "--config-file",
                str(empty_cfg),
                "--env-file",
                str(empty_env),
                "--intervals",
                "240",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert rc == 0

    incomplete_cfg = tmp_path / "incomplete.env"
    atomic_write_text(incomplete_cfg, "SYMBOL=BTCUSDT\nSTRAT_ENTRY=100\n")
    with pytest.raises(SystemExit):
        bwf.main(
            [
                "--config-file",
                str(incomplete_cfg),
                "--env-file",
                str(empty_env),
                "--intervals",
                "240",
                "--output-dir",
                str(tmp_path / "out2"),
            ]
        )


def test_kraken_grid_report_filters_stale_spec_rows_and_sweep_passes_timestamps(tmp_path):
    """kraken_continuous_grid/report.py filters stale spec rows and comprehensive_strategy_sweep passes timestamps."""
    exp_dir = tmp_path / "kcg_exp"
    exp_dir.mkdir()
    live_params, live_peak = kcg_grid.CONFIGS["live"]
    results_path = exp_dir / "results.jsonl"
    atomic_write_jsonl(
        results_path,
        [
            {
                "cfg": "live",
                "asset": "BTC",
                "params_repr": repr(live_params),
                "peak": live_peak,
                "full": {"ret": 12.5, "dd": 4.2, "expo": 35.0},
                "w90": [[3.0, 2.0, 5.0]],
                "w30": [[1.0, 1.0, 2.0]],
                "year": {"2025": {"ret": 12.5}},
            },
            {
                "cfg": "tp4",
                "asset": "BTC",
                "params_repr": "STALE_SPEC_REPR",
                "peak": False,
                "full": {"ret": 99.0, "dd": 1.0, "expo": 10.0},
            },
        ],
    )
    lines = kcg_report.generate_report(str(exp_dir))
    joined = "\n".join(lines)
    assert "`live`" in joined
    assert "`tp4`" not in joined

    records = [
        {"timestamp": 1_700_000_000 + i * 14400, "open": 20.0, "high": 20.5, "low": 19.5, "close": 20.1}
        for i in range(12)
    ]
    with patch.object(css.kraken_replay, "run_replay", wraps=css.kraken_replay.run_replay) as spy_rp:
        out = css.run_continuous_628d_replay({"00_LIVE_CURRENT": {}}, records)
        assert "00_LIVE_CURRENT" in out
        assert spy_rp.call_args.kwargs["timestamps"] == [r["timestamp"] for r in records]
        scaled_p = spy_rp.call_args.args[1]
        assert scaled_p.check_minutes == pytest.approx(240.0)

    # Verify symbol imports remain wired cleanly across Wave 10 sweep/backtest modules.
    assert callable(hr_harness.evaluate_asset)
    assert callable(rdss.load_base_params)
    assert callable(kwf.evaluate_walk_forward)
    assert callable(csv_mod.main)
    assert callable(_HL_BT.main)
