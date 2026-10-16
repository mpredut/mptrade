#!/usr/bin/env python3
"""tests/test_research_and_backtest_robustness.py — Regression tests for research runners, backtesting, and reporting robustness."""

from __future__ import annotations

import json
import os
import pytest

from offline.backtests import execution_calibration, financial_benchmark
from offline.research.chop_weight_backtest import run_single_simulation
from offline.research.hybrid_reentry import harness as hybrid_harness
from offline.research.hybrid_reentry import report as hybrid_report
from offline.research.t212_continuous_grid import report as t212_report
from strategies import spot_engine


@pytest.fixture(autouse=True)
def restore_strategy_step():
    orig = spot_engine.Strategy.step
    yield
    spot_engine.Strategy.step = orig


def test_t212_report_empty_directory(tmp_path):
    """Calling generate_report on an empty directory returns clean markdown without crashing."""
    out = t212_report.generate_report(str(tmp_path))
    assert "# Trading 212 grid" in out
    assert "No results found" in out


def test_t212_report_with_missing_live_baseline_and_none_metrics(tmp_path):
    """Reporting handles data where live baseline is missing or contains None metrics."""
    results_file = tmp_path / "results.jsonl"
    data = [
        {"profile": "nvda", "variant": "tp5", "asset": "NVDA", "1h": {"ret": 10.5, "dd": 2.1}, "w90": [[1.2, 0.5, 3.0]]},
        {"profile": "nvda", "variant": "tp5", "asset": "SPY", "1h": {"ret": 5.0, "dd": 1.2}, "1d": {"ret": None, "dd": None}},
        {"profile": "spcx", "variant": "tp8", "asset": "SPCX", "1h": None, "1d": None},
        {"profile": "rgnt", "variant": "live", "asset": "RGNT", "error": "test error trace\nValueError: test"},
    ]
    with open(results_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    out = t212_report.generate_report(str(tmp_path))
    assert "# Trading 212 grid" in out
    assert "Profile `nvda`" in out
    assert os.path.exists(tmp_path / "REPORT.md")


def test_hybrid_reentry_report_none_metrics(tmp_path, monkeypatch):
    """Hybrid reentry report handles None metric entries without format string TypeError."""
    crypto_json = tmp_path / "ablation_results.json"
    data = [
        {
            "asset": "BTC",
            "test_id": "Test 0",
            "variant": "Baseline",
            "total_return_pct": None,
            "max_drawdown_pct": None,
            "exposure_pct": None,
        }
    ]
    with open(crypto_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    monkeypatch.setattr(hybrid_report, "CRYPTO_JSON", str(crypto_json))
    monkeypatch.setattr(hybrid_report, "SUMMARY_JSON", str(tmp_path / "missing_summary.json"))
    monkeypatch.setattr(hybrid_report, "EQUITIES_JSON", str(tmp_path / "missing_equities.json"))
    rep = hybrid_report.generate_markdown_report()
    assert "Hybrid Re-Entry Architecture" in rep
    assert "**BTC**" in rep
    assert "N/A" in rep


def test_hybrid_harness_empty_ohlc():
    """Hybrid harness returns empty list cleanly when given empty OHLC."""
    from offline.research.hybrid_reentry.run_suite import BASE_PARAMS
    res = hybrid_harness.evaluate_asset("BTC", [], BASE_PARAMS)
    assert res == []


def test_chop_weight_backtest_empty_sampled():
    """Chop weight simulation handles empty sampled list without crashing or IndexError."""
    stats = run_single_simulation("TEST", [], None, "CHOP_0.03")
    assert stats.symbol == "TEST"
    assert stats.total_trades == 0
    assert stats.net_profit_usd == 0.0


def test_financial_benchmark_none_net_pnl():
    """aggregate_financial_windows handles windows whose metrics have None net_pnl."""
    windows = [
        {
            "key": "test_win",
            "return_pct": 2.5,
            "max_drawdown_pct": 1.0,
            "buy_hold_return_pct": 1.5,
            "cycles": 1,
            "fills": 2,
            "metrics": {
                "net_pnl": None,
                "max_drawdown_abs": 10.0,
            },
        }
    ]
    agg = financial_benchmark.aggregate_financial_windows(windows, initial_capital=1000.0)
    assert agg["sum_reset_net_pnl_usd"] == 0.0
    assert agg["mean_net_pnl_usd_per_window"] == 0.0


def test_execution_calibration_distribution_none_handling():
    """distribution() handles None values in list without TypeError."""
    dist = execution_calibration.distribution([1.0, None, 2.0, None, 3.0])
    assert dist["count"] == 3
    assert dist["min"] == 1.0
    assert dist["max"] == 3.0
    assert dist["mean"] == 2.0


def test_scheduled_pilot_run_one_delegates_to_run_symbol():
    """scheduled_pilot._run_one delegates directly to run_replay_backtest.run_symbol(..., series=...)."""
    from offline.research.monitortrades_backtest import run_replay_backtest as rb
    from offline.research.monitortrades_backtest import scheduled_pilot as sp

    cfg = rb.SYMBOLS["BTCUSDC"]
    series = [(1_700_000_000.0 + i * 60.0, 50_000.0 + (i % 5) * 100.0) for i in range(30)]
    full_res = rb.run_symbol("BTCUSDC", cfg["params"], cfg["base"], quiet=True, series=series)
    pilot_res = sp._run_one("BTCUSDC", cfg["base"], cfg["params"], series)
    assert full_res is not None
    assert pilot_res == {"net": full_res["net"], "buy_hold": full_res["buy_hold"]}
    assert sp._run_one("BTCUSDC", cfg["base"], cfg["params"], []) is None


def test_spot_replay_adapters_ssot():
    """All spot replay adapters delegate to strategies.spot_replay.make_spot_replay_adapter."""
    import importlib.util
    from pathlib import Path
    from binance_api.replay import make_binance_replay_adapter
    from kraken.replay import make_kraken_replay_adapter
    from strategies.spot_engine import StratParams
    from strategies.spot_replay import make_spot_replay_adapter

    hl_path = Path(__file__).resolve().parents[1] / "hyperliquid" / "replay.py"
    spec = importlib.util.spec_from_file_location("hl_replay_ssot_test", str(hl_path))
    hl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl_mod)
    make_hyperliquid_spot_replay_adapter = hl_mod.make_hyperliquid_spot_replay_adapter

    params = StratParams(
        currency="USDC",
        entry_amount=100.0,
        entry_discount_pct=0.5,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        takeprofit_pct=2.0,
        max_budget=300.0,
        max_dca_buys=2,
        stop_loss_pct=0.0,
        check_minutes=1.0,
        enable_takeprofit=True,
        order_ttl_min=60.0,
    )
    ohlc = [(100.0, 101.0, 99.0, 100.0), (100.0, 102.0, 98.0, 101.0)]
    for factory in (
        make_spot_replay_adapter,
        make_kraken_replay_adapter,
        make_binance_replay_adapter,
        make_hyperliquid_spot_replay_adapter,
    ):
        adapter = factory(params, bar_minutes=60.0)
        res = adapter(ohlc, None)
        assert "return_pct" in res
        assert "max_drawdown_pct" in res


def test_sim_clock_value_and_ts_alias():
    """SimClock supports both .ts and .value property access and optional initial_ts."""
    from providers.replay_clock import SimClock

    clock = SimClock(0.0)
    assert clock.ts == 0.0
    assert clock.value == 0.0
    assert clock() == 0.0
    clock.value = 123.5
    assert clock.ts == 123.5
    assert clock() == 123.5


def test_walk_forward_cli_shared_helpers(tmp_path):
    """Shared walk-forward CLI helpers parse intervals, dataset specs, and window sizes."""
    import argparse
    from offline.backtests.evaluation import (
        add_walk_forward_cli_args,
        build_execution_model_from_args,
        parse_dataset_specs,
        parse_intervals,
        resolve_window_sizes,
    )

    assert parse_intervals("60, 240,1440") == [60, 240, 1440]
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
    specs = parse_dataset_specs([f"60={csv_file}"])
    assert specs[60] == csv_file.resolve()

    parser = argparse.ArgumentParser()
    add_walk_forward_cli_args(parser, include_warmup=True, default_warmup=0)
    args = parser.parse_args(["--train", "50", "--validation", "10", "--test", "15", "--spread-bps", "5"])
    model = build_execution_model_from_args(parser, args)
    assert model.spread_bps == 5.0
    assert resolve_window_sizes(args, 200) == (50, 10, 15, 15)
