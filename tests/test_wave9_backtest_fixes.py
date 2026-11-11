"""Targeted regression tests for Wave 9 backtest, replay, simulation, runner, and research fixes."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from botcore import load_local_module
from offline.backtests import tradeall as bt_tradeall
from offline.backtests.evaluation import load_venue_replay_module
from offline.research import chop_weight_backtest as cwb, intelligence_backtest as ib
from offline.research.kraken_continuous_grid import (
    grid as kcg_grid,
    grid2 as kcg_grid2,
    grid3 as kcg_grid3,
    report as kcg_report,
    report2 as kcg_report2,
)
from offline.research.monitortrades_backtest import backtest_wait_and_regime as bwr
from offline.runners import apply_proposals
from offline.simulations.sim import PriceWindow
from providers.replay_provider import backtest_isolation_context, load_price_series
from state_io import atomic_write_jsonl, read_text_file
from strategies.replay_execution import ExecutionModel, split_order_fill
from strategies.replay_metrics import calculate_performance_metrics

ROOT = Path(__file__).resolve().parents[1]
_HL_REPLAY = load_venue_replay_module("hyperliquid", "test_w9_hl_replay")
_KRAKEN_TRAIL = load_local_module(str(ROOT / "kraken"), "backtest_trail", module_key="test_w9_kraken_trail")

WAVE9_PY_FILES = (
    "hyperliquid/replay.py",
    "kraken/backtest_trail.py",
    "offline/backtests/tradeall.py",
    "offline/research/chop_weight_backtest.py",
    "offline/research/compare_mhdsl_reentry_replay.py",
    "offline/research/hybrid_reentry/run_suite.py",
    "offline/research/intelligence_backtest.py",
    "offline/research/kraken_continuous_grid/grid.py",
    "offline/research/kraken_continuous_grid/grid2.py",
    "offline/research/kraken_continuous_grid/grid3.py",
    "offline/research/kraken_continuous_grid/report.py",
    "offline/research/kraken_continuous_grid/report2.py",
    "offline/research/monitortrades_backtest/backtest_wait_and_regime.py",
    "offline/research/monitortrades_backtest/sweep_mt_models.py",
    "offline/research/verify_optimizations_parity.py",
    "offline/runners/apply_proposals.py",
    "offline/runners/fetch_hyperliquid_candles.py",
    "offline/runners/spot_profile_review.py",
    "offline/simulations/sim.py",
    "providers/replay_provider.py",
    "strategies/replay_execution.py",
    "strategies/replay_metrics.py",
    "tests/test_wave9_backtest_fixes.py",
)

WAVE9_SH_FILES = (
    "offline/runners/run_backtest_cycle.sh",
    "offline/runners/run_dev_regression.sh",
)


def test_wave9_ascii_and_ast_hygiene() -> None:
    """Enforce 100% ASCII, zero raw float/int/open calls, and venue-import isolation on Wave 9 files."""
    for rel in WAVE9_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"float", "int", "open"}, (
                    f"Forbidden raw {node.func.id}() in {rel}:{node.lineno}"
                )
            if rel.startswith("offline/") and isinstance(node, (ast.Import, ast.ImportFrom)):
                mod_name = getattr(node, "module", "") or ""
                assert not mod_name.startswith(("kraken.", "hyperliquid.", "binance_api.")), (
                    f"Forbidden direct venue import {mod_name!r} in {rel}:{node.lineno}"
                )
    for rel in WAVE9_SH_FILES:
        path = ROOT / rel
        src = read_text_file(str(path))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"bash -n failed for {rel}: {proc.stderr}"


def test_wave9_replay_metrics_float_jitter_and_split_order_fill_bounds() -> None:
    """Constant-growth geometric curves must not produce 1e16 Sharpe from float jitter; ratio > 1 fails."""
    curve = [100.0 * (1.1 ** i) for i in range(10)]
    summary = calculate_performance_metrics(curve, initial_capital=100.0, trade_pnls=[159.37])
    assert summary["sharpe"] is None
    assert summary["sortino"] is None

    with pytest.raises(ValueError, match="ratio must be within"):
        split_order_fill({"sz": 10.0, "amount": 100.0}, quantity_key="sz", amount_key="amount", ratio=1.5)


def test_wave9_hyperliquid_replay_gap_sl_dca_guard_price_improvement_and_spot_adapter() -> None:
    """Gap-down past SL blocks resting DCA, gap opens improve resting limits, and spot adapter aligns."""
    with pytest.raises(ValueError, match="ts is required when fund is provided"):
        _HL_REPLAY.run_replay(
            [(100.0, 101.0, 99.0, 100.0)],
            ts=None,
            fund=[(1700000000000, 0.0001)],
        )

    # Gap-down open improves resting BUY ENTRY limit (limit ~99.9, bar 1 opens at 95.0 -> fills at 95.0)
    params_gap = _HL_REPLAY.StratParams(
        direction="long",
        max_budget=1000.0,
        entry_amount=100.0,
        entry_discount_pct=0.1,
        dca_amount=100.0,
        dca_drop_pct=5.0,
        max_dca_buys=2,
        takeprofit_pct=20.0,
        stop_loss_pct=10.0,
        fee_pct=0.0,
    )
    bars_gap_entry = [
        (100.0, 100.5, 100.0, 100.0),  # Bar 0 close places resting ENTRY buy near 99.9
        (95.0, 96.0, 94.5, 95.0),      # Bar 1 gaps down to 95.0 -> must fill ENTRY at 95.0, then place DCA ~90.25
        (80.0, 92.0, 79.0, 82.0),      # Bar 2 gaps open to 80.0 (< SL 85.5) -> must NOT fill DCA at 90.25!
    ]
    res = _HL_REPLAY.run_replay(
        bars_gap_entry,
        params=params_gap,
        bar_minutes=60.0,
        execution=ExecutionModel(intrabar_policy="worst_case"),
    )
    assert res["cycles"] == 1
    assert res["fills"] == 2  # 1 ENTRY + 1 SL exit (resting DCA blocked by open gap past SL)
    # Entry ~100/99.9 qty filled at 95.0, stopped out at gap open 80.0 => loss ~ -15.015, NOT extra DCA loss
    assert res["net"] == pytest.approx(-(100.0 / 99.9) * (95.0 - 80.0), rel=1e-3)

    spot_adapter = _HL_REPLAY.make_hyperliquid_replay_adapter(
        params={"max_budget": 1000.0, "entry_amount": 200.0, "dca_amount": 200.0, "fee": 0.1},
        bar_minutes=60.0,
        engine="spot",
    )
    spot_res = spot_adapter([(100.0, 102.0, 99.0, 101.0), (101.0, 103.0, 100.0, 102.0)])
    assert spot_res["initial_capital"] == pytest.approx(1000.0)


def test_wave9_kraken_backtest_trail_isolation_and_reentry_drop_validation() -> None:
    """Isolated trail comparison disables reentry TTL/bounce and validates reentry_fallback."""
    assert _KRAKEN_TRAIL._ISOLATED_COMPARISON_OVERRIDES["reentry_ttl_hours"] == 0.0
    assert _KRAKEN_TRAIL._ISOLATED_COMPARISON_OVERRIDES["reentry_bear_bounce_pct"] == 0.0
    assert _KRAKEN_TRAIL._resolve_reentry_drop({"reentry_fallback": 3.5}) == pytest.approx(3.5)
    assert _KRAKEN_TRAIL._resolve_reentry_drop({"reentry_drop": 2.0}) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        _KRAKEN_TRAIL._resolve_reentry_drop({"reentry_fallback": True})


def test_wave9_sim_price_window_small_max_index_normalization() -> None:
    """PriceWindow with window_size >= max_index normalizes cleanly without stuck high indices."""
    pw = PriceWindow(window_size=5, max_index=3)
    for idx in range(25):
        pw.process_price(100.0 + (idx % 4))
    assert len(pw.prices) == 5
    assert pw.current_index <= 7
    assert pw.get_min()[0] == pytest.approx(100.0)
    assert pw.get_max()[0] == pytest.approx(103.0)


def test_wave9_tradeall_broker_budget_cap_and_replay_provider_isolation(tmp_path: Path) -> None:
    """BacktestBroker enforces max_quote_budget and backtest_isolation_context restores builtins.print."""
    clock = SimpleNamespace(ts=1000.0)
    broker = bt_tradeall.BacktestBroker(
        str(tmp_path),
        lambda: clock.ts,
        max_quote_budget=500.0,
    )
    broker.current_tick_price = 100.0
    r1 = broker.place_order_smart("BUY", "BTCUSDC", 100.0, qty=4.0, motivation="entry")
    assert r1 is not None
    assert broker.pos_cost == pytest.approx(400.0)
    # Second BUY requests 300 USD, capped to remaining 100 USD budget
    r2 = broker.place_order_smart("BUY", "BTCUSDC", 100.0, qty=3.0, motivation="dca")
    assert r2 is not None
    assert broker.pos_cost == pytest.approx(500.0)
    assert broker.peak_deployed_cost == pytest.approx(500.0)
    # Third BUY when budget is exhausted is refused
    r3 = broker.place_order_smart("BUY", "BTCUSDC", 100.0, qty=1.0, motivation="overflow")
    assert r3 is None
    assert broker.n_buy == 2

    jsonl_path = tmp_path / "ticks.jsonl"
    atomic_write_jsonl(
        str(jsonl_path),
        [{"s": "BTCUSDC", "i": [1000 * (i + 1), 100.0 + i]} for i in range(10)],
    )
    limited = load_price_series(str(jsonl_path), "BTCUSDC", max_samples=4)
    assert len(limited) == 4
    assert limited[0] == (7.0, 106.0)
    assert limited[-1] == (10.0, 109.0)

    prev_print = print
    with backtest_isolation_context(quiet=True):
        pass
    assert print is prev_print


def test_wave9_intelligence_survival_one_bar_trends_and_regime_short_span(tmp_path: Path) -> None:
    """1-bar BULL/BEAR trends record nonzero elapsed duration; evaluate_regime_900s handles short & full spans."""
    tape = []
    t = 1_700_000_000.0
    for idx in range(12):
        trend = "BULL" if idx % 2 == 0 else "FLAT"
        tape.append({"t": t + idx * 86400.0, "current_trend": trend, "trend_duration_sec": 0.0})
    sched = ib._compute_causal_survival_schedule(
        tape, warmup_median_days=3.0, warmup_p90_days=7.0, min_completed_trends=3
    )
    # After 3 completed 1-day BULL trends (at indices 1, 3, 5), schedule switches from (3.0, 7.0) to ~1.0 day
    assert sched[0] == (3.0, 7.0)
    assert sched[-1][0] == pytest.approx(1.0)

    # Short synthetic span (< 900s) evaluates remaining samples; long span (>= 900s) drops unmatured tail
    short_ts = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    short_px = np.array([100.0, 100.1, 100.2, 100.3])
    short_sig = {"gc": np.ones_like(short_px), "eps": np.ones_like(short_px), "scale_inv": np.ones_like(short_px)}
    short_res, short_raw, _ = bwr.evaluate_regime_900s(
        short_ts, short_px, short_sig, horizon_sec=900.0, stride=1
    )
    assert len(short_res) == 6
    assert len(short_raw) == 3

    long_ts = np.arange(0.0, 2000.0, 100.0)
    long_px = 100.0 + 0.01 * long_ts
    long_sig = {"gc": np.ones_like(long_px), "eps": np.ones_like(long_px), "scale_inv": np.ones_like(long_px)}
    long_res, long_raw, _ = bwr.evaluate_regime_900s(
        long_ts, long_px, long_sig, horizon_sec=900.0, stride=1
    )
    # Last timestamp is 1900.0, so samples with t0 + 900 <= 1900 => t0 <= 1000 => 11 samples (0..1000)
    assert len(long_res) == 6
    assert len(long_raw) == 11

    assert hasattr(cwb, "run_chop_benchmark")


def test_wave9_kraken_continuous_grid_classic_variants_and_report_spec_filter(tmp_path: Path) -> None:
    """Classic reentry variants disable TTL/bounce; report filters stale params_repr rows but keeps legacy rows."""
    for cfg_name in ("reentry0", "reentry1", "reentry4", "reentry_adaptive"):
        params, _ = kcg_grid.CONFIGS[cfg_name]
        assert params.reentry_ttl_hours == 0.0
        assert params.reentry_bear_bounce_pct == 0.0
    ov_params, _ = kcg_grid2.FINE["ov650_t8_reentry4"]
    assert ov_params.reentry_ttl_hours == 0.0
    assert ov_params.reentry_bear_bounce_pct == 0.0
    assert "live" in kcg_grid3.CFGS

    results_path = tmp_path / "results.jsonl"
    live_params, live_peak = kcg_grid.CONFIGS["live"]
    atomic_write_jsonl(
        str(results_path),
        [
            {"cfg": "live", "asset": "BTC", "params_repr": "STALE_REPR", "peak": False, "full": {"ret": 99.0, "dd": 1.0}},
            {"cfg": "live", "asset": "ETH", "params_repr": repr(live_params), "peak": bool(live_peak), "full": {"ret": 12.0, "dd": 4.0}},
            {"cfg": "live", "asset": "SOL", "full": {"ret": 8.0, "dd": 3.0}},
            {"cfg": "live", "asset": "XRP", "kind": "error", "error": "boom"},
        ],
    )
    valid_rows, errors, by, assets = kcg_report._load_grid_results_by_cfg(
        str(results_path), spec_map=kcg_grid.CONFIGS
    )
    assert len(errors) == 1
    assert assets == ["ETH", "SOL"]
    assert "BTC" not in by["live"]
    assert kcg_report2._rank_key({"full_med": 10.0, "calmar_med": 2.0, "dd_med": 5.0})[0] is True


def test_wave9_apply_proposals_rolls_back_on_failed_git_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """When git commit fails in apply_proposals.main, sp._apply_config_change rolls back to current."""
    calls: list[tuple[str, str, float, float]] = []
    audits: list[dict] = []

    monkeypatch.delenv("APPLY_DISABLED", raising=False)
    monkeypatch.setattr(apply_proposals, "_ALLOWED_PILOT_PARAM_NAMES", frozenset({"mt.lost"}))
    monkeypatch.setattr(apply_proposals, "load_env_stack", lambda: None)
    monkeypatch.setattr(
        apply_proposals,
        "_read_proposals",
        lambda: [{
            "full_key": "crypto:BTCUSDC.mt.lost",
            "section": "crypto:BTCUSDC",
            "key": "mt.lost",
            "symbol": "BTCUSDC",
            "winner_value": 2.0,
            "dev_commit": "abc1234",
        }],
    )
    monkeypatch.setattr(apply_proposals.sp, "_current_value", lambda _s, _k: 1.0)
    monkeypatch.setattr(apply_proposals.sp, "is_rate_limited", lambda _fk: (False, None))
    monkeypatch.setattr(
        apply_proposals.sp,
        "_apply_config_change",
        lambda sec, k, old_v, new_v: calls.append((sec, k, old_v, new_v)),
    )
    monkeypatch.setattr(apply_proposals.sp, "_append_audit", lambda rec: audits.append(dict(rec)))
    monkeypatch.setattr(
        apply_proposals.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="lock error"),
    )
    monkeypatch.setattr("sys.argv", ["apply_proposals.py"])

    apply_proposals.main()
    assert calls == [
        ("crypto:BTCUSDC", "mt.lost", 1.0, 1.5),
        ("crypto:BTCUSDC", "mt.lost", 1.5, 1.0),
    ]
    assert audits and audits[0]["action"] == "commit_failed"
