"""Targeted regression tests for Wave 6 backtest, replay, research, and runner fixes."""
import os
from pathlib import Path
import subprocess

import pytest

import kraken.backtest as k_bt
from offline.backtests.datasets import align_previous_values, load_ohlc_tuples_csv, save_dataset
from offline.backtests.evaluation import (
    assess_decision_evidence,
    evaluate_walk_forward,
    extract_cfg_int,
    load_layered_env_config,
    load_venue_module,
    to_ohlc,
)
from offline.backtests.execution import ExecutionModel, choose_intrabar_scenario
from offline.research import slice_dense_archive as sda
from offline.research.kraken_continuous_grid import report as k_rep
from offline.research.t212_continuous_grid.grid import _extract_asset_name
from offline.runners import financial_promotion_gate as fpg
from providers.order_ohlc_utils import drop_incomplete_last_bar
from state_io import atomic_write_json, atomic_write_jsonl
from strategies.spot_engine import StratParams
from strategies.spot_replay import run_replay as run_spot_replay
from strategies.spot_rules import check_fast_profit_reversal

ROOT = Path(__file__).resolve().parents[1]

WAVE6_PURE_ASCII_FILES = (
    # Wave 6A (excluding 212trading/strategy.py which intentionally keeps notify_engine markers)
    "strategies/spot_rules.py",
    "strategies/spot_replay.py",
    "strategies/spot_engine.py",
    "212trading/replay.py",
    "212trading/backtest.py",
    "kraken/backtest.py",
    "offline/backtests/datasets.py",
    "providers/ohlc_utils.py",
    "providers/order_ohlc_utils.py",
    "offline/backtests/evaluation.py",
    "offline/backtests/execution.py",
    # Wave 6B
    "offline/research/monitortrades_backtest/run_replay_backtest.py",
    "offline/research/monitortrades_backtest/replay_trend_source.py",
    "offline/research/monitortrades_backtest/backtest_wait_and_regime.py",
    "offline/research/monitortrades_backtest/compare_sl_variants.py",
    "offline/research/monitortrades_backtest/sweep_mt_models.py",
    "offline/research/monitortrades_backtest/sweep_multiwindow_mt.py",
    # Wave 6C
    "offline/research/hybrid_reentry/harness.py",
    "offline/research/hybrid_reentry/report.py",
    "offline/research/hybrid_reentry/run_equities.py",
    "offline/research/comprehensive_strategy_sweep.py",
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
    "offline/research/kraken_continuous_grid/report.py",
    "offline/research/kraken_continuous_grid/report2.py",
    "offline/research/kraken_continuous_grid/run_all.sh",
    "offline/research/t212_continuous_grid/grid.py",
    "offline/research/t212_continuous_grid/report.py",
    "offline/research/slice_dense_archive.py",
    "offline/research/ml_forecast/forecast.py",
    "offline/research/ml_forecast/priceprediction.py",
    "offline/research/ml_forecast/vol_chronos.py",
    # Wave 6D
    "offline/runners/load_dev_backtest_env.sh",
    "offline/runners/run_overnight_master_suite.sh",
    "offline/runners/run_dev_regression.sh",
    "offline/runners/sync_cache_to.sh",
    "offline/runners/calibrate_execution_audit.py",
    "offline/runners/financial_promotion_gate.py",
    "offline/runners/kraken_walk_forward_baseline.py",
    "offline/runners/kraken_walk_forward_compare.py",
    "offline/runners/walk_forward_compare.py",
    "offline/runners/walk_forward.py",
    "hyperliquid/shadow_longterm.py",
    "kraken/shadow_live.py",
    "offline/legacy_tools/clean_order_cache.py",
    "offline/legacy_tools/migrate_price_cache.py",
    "offline/legacy_tools/move.py",
    "offline/legacy_tools/test.py",
    "offline/legacy_tools/test2.py",
    "tests/test_wave6_backtest_fixes.py",
)


def test_wave6_ascii_and_shell_syntax_hygiene():
    for rel in WAVE6_PURE_ASCII_FILES:
        path = ROOT / rel
        assert path.is_file(), f"Missing file: {rel}"
        text = path.read_text(encoding="utf-8")
        bad = [
            (idx + 1, ch)
            for idx, line in enumerate(text.splitlines())
            if not (rel == "providers/order_ohlc_utils.py" and "REAL MONEY" in line)
            for ch in line
            if ord(ch) >= 128
        ]
        assert not bad, f"Non-ASCII characters in {rel}: {bad[:5]}"
        if rel.endswith(".sh"):
            proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
            assert proc.returncode == 0, f"bash -n failed for {rel}: {proc.stderr}"


def test_wave6a_spot_rules_fast_profit_window_and_params_validation():
    # Stale 4h shadow ticks older than 5-minute window_sec=300 must NOT false-trigger
    triggered_stale, gain_stale, drop_stale = check_fast_profit_reversal(
        current_price=112.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(1000.0, 120.0), (15400.0, 118.0)],
        window_sec=300.0,
        drop_pct=1.0,
        current_time=29800.0,
    )
    assert triggered_stale is False
    assert gain_stale == pytest.approx(12.0)
    assert drop_stale == 0.0

    # Future ticks (> current_time) and non-positive window_sec must NOT trigger
    triggered_future, _, _ = check_fast_profit_reversal(
        current_price=112.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(30100.0, 125.0)],
        window_sec=300.0,
        drop_pct=1.0,
        current_time=29800.0,
    )
    assert triggered_future is False

    triggered_zero_win, _, _ = check_fast_profit_reversal(
        current_price=112.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(29750.0, 120.0)],
        window_sec=0.0,
        drop_pct=1.0,
        current_time=29800.0,
    )
    assert triggered_zero_win is False

    # Genuine in-window micro-gradient drop triggers cleanly
    triggered_ok, _, drop_ok = check_fast_profit_reversal(
        current_price=112.0,
        avg_cost=100.0,
        base_tp_pct=5.0,
        mult=2.0,
        shadow_prices=[(29600.0, 115.0), (29800.0, 112.0)],
        window_sec=300.0,
        drop_pct=1.0,
        current_time=29800.0,
    )
    assert triggered_ok is True
    assert drop_ok == pytest.approx((115.0 - 112.0) / 115.0 * 100.0)

    # StratParams validates positive sizing/cadence and tranche percentage sums
    base_kw = dict(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        check_minutes=5.0,
        takeprofit_pct=5.0,
        max_budget=300.0,
        max_dca_buys=2,
        enable_takeprofit=True,
        order_ttl_min=30.0,
    )
    with pytest.raises(ValueError):
        StratParams(**{**base_kw, "entry_amount": 0.0})
    with pytest.raises(ValueError):
        StratParams(**{**base_kw, "tp_tranches": [(3.0, 40.0), (6.0, 40.0)]})


def test_wave6a_partial_fill_tp_tranches_and_t212_ladder_no_zeno_paradox():
    exec_half = ExecutionModel(partial_fill_ratio=0.5)

    # 1. Spot multi-tranche TP under partial_fill_ratio=0.5 completes in finite bars
    spot_params = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=50.0,
        dca_drop_pct=10.0,
        check_minutes=60.0,
        takeprofit_pct=10.0,
        max_budget=200.0,
        max_dca_buys=0,
        enable_takeprofit=True,
        order_ttl_min=600.0,
        tp_tranches=[(3.0, 50.0), (6.0, 50.0)],
        reentry_drop_pct=50.0,
    )
    ohlc = [
        (100.0, 100.2, 99.8, 100.0),  # bar 0 places entry at 100
        (100.0, 100.2, 99.8, 100.0),  # bar 1 fills 50% entry
        (100.0, 100.2, 99.8, 100.0),  # bar 2 fills remaining 50% entry -> dispatches 2 TP tranches (103, 106)
        (102.0, 104.0, 101.5, 103.5), # bar 3 fills 50% of tranche 0 (103)
        (103.5, 104.5, 103.0, 104.0), # bar 4 completes remaining 50% of tranche 0 (no Zeno halving!)
        (104.0, 107.0, 103.8, 106.5), # bar 5 fills 50% of tranche 1 (106) without re-placing tranche 0!
        (106.5, 108.0, 106.0, 107.0), # bar 6 completes remaining 50% of tranche 1 -> cycle closed!
    ]
    spot_res = run_spot_replay(ohlc, spot_params, fee_pct=0.0, bar_minutes=60, execution=exec_half)
    assert spot_res["cycles"] == 1
    assert spot_res["open_qty"] == pytest.approx(0.0, abs=1e-9)
    assert spot_res["fills"] == 6  # 2 entry partial fills + 2 tranche0 partial fills + 2 tranche1 partial fills

    # 2. Trading212 multi-rung ladder TP under partial_fill_ratio=0.5 completes cleanly
    t212_rp = load_venue_module("212trading", "replay.py", "test_w6_t212_rp")
    t212_bt = load_venue_module("212trading", "backtest.py", "test_w6_t212_bt")
    t212_env = {
        **t212_bt._SIMULATE_ENV_DEFAULTS,
        "STRATEGY_MODE": "avg_tp",
        "STRAT_CURRENCY": "USD",
        "YAHOO_SYMBOL": "TEST",
        "STRAT_ENTRY_PCT": "100",
        "STRAT_DCA_PCT": "50",
        "STRAT_ENTRY_DISCOUNT_PCT": "0.0",
        "STRAT_DCA_DROP_PCT": "10.0",
        "STRAT_TAKEPROFIT_PCT": "10.0",
        "STRAT_MAX_DCA_BUYS": "0",
        "STRAT_MAX_BUDGET": "100",
        "STRAT_FX_FEE_PCT": "0.0",
        "STRAT_STOP_LOSS_PCT": "0.0",
        "STRAT_CHECK_MINUTES": "60",
        "STRAT_ORDER_TTL_MIN": "600",
        "STRAT_TP_LADDER": "3:50,6:50",
        "STRAT_REENTRY_DROP_PCT": "50.0",
    }
    t212_params = t212_rp._strat.StratParams.from_env(t212_env)
    t212_res = t212_rp.run_replay(ohlc, t212_params, bar_minutes=60.0, execution=exec_half)
    assert t212_res["cycles"] == 1
    assert t212_res["open_qty"] == pytest.approx(0.0, abs=1e-9)
    assert t212_res["fills"] == 6


def test_wave6a_t212_and_kraken_backtest_isolation_and_replay_guards(monkeypatch):
    t212_rp = load_venue_module("212trading", "replay.py", "test_w6_t212_rp2")
    t212_bt = load_venue_module("212trading", "backtest.py", "test_w6_t212_bt2")

    # 1. Poison ambient STRAT_* in os.environ and verify load_layered_env_config & backtesters ignore it
    monkeypatch.setenv("STRAT_TAKEPROFIT_PCT", "999.0")
    monkeypatch.setenv("STRAT_TP_LADDER", "99:100")

    layered = load_layered_env_config(ROOT / "kraken" / "config.env", ROOT / "nonexistent.env")
    assert layered["STRAT_TAKEPROFIT_PCT"] != "999.0"

    iso_cfg = t212_bt._load_isolated_t212_cfg(
        str(ROOT / "212trading" / ".env"),
        profile_file=str(ROOT / "212trading" / "config.nvda.env"),
    )
    assert iso_cfg.get("STRAT_TAKEPROFIT_PCT") != "999.0"
    assert os.environ.get("YAHOO_SYMBOL") != "NVDA"

    # 2. T212 simulate() handles both percentage (50,50) and fraction (0.5,0.5) tp_ladder tuples
    bars = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 112.0, 99.5, 111.0),
    ]
    res_pct = t212_bt.simulate(bars, {"entry": 100.0, "budget": 100.0, "disc": 0.0, "maxdca": 0, "tp_ladder": [(5.0, 50.0), (10.0, 50.0)]})
    res_frac = t212_bt.simulate(bars, {"entry": 100.0, "budget": 100.0, "disc": 0.0, "maxdca": 0, "tp_ladder": [(5.0, 0.5), (10.0, 0.5)]})
    assert res_pct["total"] == pytest.approx(res_frac["total"])
    assert res_pct["cycles"] == 1

    # 3. T212 replay validates timestamps strictly increasing and non-negative
    p_base = t212_rp._strat.StratParams.from_env(
        {
            **t212_bt._SIMULATE_ENV_DEFAULTS,
            "STRATEGY_MODE": "avg_tp",
            "STRAT_CURRENCY": "USD",
            "YAHOO_SYMBOL": "TEST",
            "STRAT_ENTRY_PCT": "50",
            "STRAT_DCA_PCT": "50",
            "STRAT_ENTRY_DISCOUNT_PCT": "0.0",
            "STRAT_DCA_DROP_PCT": "5.0",
            "STRAT_TAKEPROFIT_PCT": "5.0",
            "STRAT_MAX_DCA_BUYS": "1",
            "STRAT_MAX_BUDGET": "200",
        }
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        t212_rp.run_replay(bars, p_base, timestamps=[100, 100, 200])
    with pytest.raises(ValueError, match="non-negative"):
        t212_rp.run_replay(bars, p_base, timestamps=[-10, 0, 10])

    # 4. Kraken _bt_params unmasks tp_tranches & tp_dynamic_flat when explicit tp is supplied
    kp = k_bt._bt_params({"tp": 4.25, "entry": 200.0, "dca": 100.0, "budget": 500.0})
    assert kp.takeprofit_pct == pytest.approx(4.25)
    assert kp.tp_tranches == []
    assert kp.tp_dynamic_flat is False
    assert kp.entry_pct == 0.0
    assert kp.dca_pct == 0.0


def test_wave6a_datasets_order_ohlc_evaluation_and_worst_case_tiebreak(tmp_path):
    # 1. datasets: save_dataset validates OHLC invariants & ordering; load_ohlc_tuples_csv deduplicates/sorts
    with pytest.raises(ValueError):
        save_dataset([{"timestamp": 1700000000, "open": 100.0, "high": 95.0, "low": 98.0, "close": 99.0}], tmp_path / "bad.csv")

    csv_path = tmp_path / "unsorted.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close\n"
        "1700000120,101,103,100,102\n"
        "1700000060,100,102,99,101\n"
        "1700000060,100,104,98,103\n",
        encoding="utf-8",
    )
    tuples = load_ohlc_tuples_csv(csv_path)
    assert len(tuples) == 2
    assert tuples[0] == (1700000060, 100.0, 104.0, 98.0, 103.0)
    assert tuples[1] == (1700000120, 101.0, 103.0, 100.0, 102.0)

    with pytest.raises(ValueError, match="non-decreasing"):
        align_previous_values([200, 100], [{"timestamp": 50, "close": 1.2}])

    # 2. drop_incomplete_last_bar drops multiple future/open-session daily tail bars and rejects bool interval
    with pytest.raises(ValueError):
        drop_incomplete_last_bar([], interval_minutes=True, now_timestamp=1000)
    daily_rows = [
        {"timestamp": 1700000000, "open": 100, "high": 101, "low": 99, "close": 100},
        {"timestamp": 1700086400, "open": 100, "high": 101, "low": 99, "close": 100},
        {"timestamp": 1700172800, "open": 100, "high": 101, "low": 99, "close": 100},
    ]
    trimmed = drop_incomplete_last_bar(
        daily_rows,
        interval_minutes=1440,
        now_timestamp=1700050000,
        regular_session_start=1700086400,
        regular_session_end=1700110000,
    )
    assert len(trimmed) == 1
    assert trimmed[0]["timestamp"] == 1700000000

    # 3. evaluation: <-100% leveraged windows clamp at ruin instead of flipping negative*negative -> positive
    records = [
        {"timestamp": 1700000000 + i * 60, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(20)
    ]
    wf = evaluate_walk_forward(
        records,
        lambda _o, _w, _c: {"return_pct": -120.0, "max_drawdown_pct": 100.0, "cycles": 1, "fills": 2},
        train_size=8,
        validation_size=3,
        test_size=2,
        step_size=2,
    )
    assert wf["aggregate_test"]["compounded_reset_return_pct"] == pytest.approx(-100.0)

    # Millisecond timestamps in assess_decision_evidence normalize to seconds
    ev_ms = assess_decision_evidence(
        [{"timestamp": 1_700_000_000_000}, {"timestamp": 1_700_000_000_000 + 100 * 86_400 * 1000}],
        {"fold_count": 25, "total_cycles": 5, "worst_return_pct": 1.0},
    )
    assert ev_ms["history_days"] == pytest.approx(100.0)
    assert ev_ms["verdict"] == "decision_grade"

    with pytest.raises(ValueError):
        extract_cfg_int({"N": "3.5"}, "N", 1)
    with pytest.raises(ValueError):
        to_ohlc([{"timestamp": 1, "open": float("nan"), "high": 10.0, "low": 9.0, "close": 9.5}])

    # 4. choose_intrabar_scenario worst_case breaks return_pct ties by larger max_drawdown_pct
    em_worst = ExecutionModel(intrabar_policy="worst_case")
    chosen = choose_intrabar_scenario(
        em_worst,
        lambda m: {
            "return_pct": 0.0,
            "max_drawdown_pct": 3.0 if m.intrabar_policy == "buy_first" else 9.5,
            "fills": 1,
            "ambiguous_bars": 2,
        },
    )
    assert chosen["max_drawdown_pct"] == pytest.approx(9.5)
    assert chosen["intrabar_policy_selected"] == "sell_first"


def test_wave6bcd_monitortrades_grids_archive_and_promotion_gate(tmp_path):
    # 1. t212_continuous_grid preserves multi-underscore asset stems
    assert _extract_asset_name("VUSA_EQ_1d.csv") == "VUSA_EQ"
    assert _extract_asset_name("NVDA_US_EQ_1h.csv") == "NVDA_US_EQ"
    assert _extract_asset_name("README.txt") is None

    # 2. kraken_continuous_grid report ranks all-negative candidates accurately (no 0.0 masking)
    exp_dir = tmp_path / "kgrid"
    exp_dir.mkdir()
    atomic_write_jsonl(
        exp_dir / "results.jsonl",
        [
            {
                "fee": "std",
                "asset": "BTC",
                "cfg": "live",
                "full": {"ret": -5.0, "dd": 10.0, "cyc": 2, "upnl": 0.0},
                "h1": {"ret": -2.5, "dd": 6.0},
                "h2": {"ret": -2.5, "dd": 6.0},
                "w30": [[-1.0, 0.0, 3.0]],
                "w60": [[-2.0, 0.0, 4.0]],
                "w90": [[-3.0, 0.0, 5.0]],
            },
            {
                "fee": "std",
                "asset": "BTC",
                "cfg": "cand_less_neg",
                "full": {"ret": -1.5, "dd": 4.0, "cyc": 2, "upnl": 0.0},
                "h1": {"ret": -0.8, "dd": 3.0},
                "h2": {"ret": -0.7, "dd": 3.0},
                "w30": [[-0.5, 0.0, 2.0]],
                "w60": [[-1.0, 0.0, 2.5]],
                "w90": [[-1.2, 0.0, 3.0]],
            },
        ],
    )
    lines = k_rep.generate_report(str(exp_dir))
    joined = "\n".join(lines)
    assert "cand_less_neg" in joined
    assert joined.index("cand_less_neg") < joined.index("live ")

    # 3. slice_dense_archive normalizes mixed ms and second records
    assert sda._ts_sec({"i": [1_700_000_000_000, 100.0]}) == pytest.approx(1_700_000_000.0)
    assert sda._ts_sec({"t": 1_700_000_000.0}) == pytest.approx(1_700_000_000.0)

    # 4. financial_promotion_gate exits cleanly (SystemExit(2)) on corrupt/non-object JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    ok_json = tmp_path / "ok.json"
    atomic_write_json(ok_json, {"schema_version": 1})
    with pytest.raises(SystemExit) as exc_info:
        fpg.main([str(bad_json), str(ok_json)])
    assert exc_info.value.code == 2
