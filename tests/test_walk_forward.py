import argparse
from tests.conftest import TempDirTestCase
import dataclasses
import json
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import kraken.backtest as k_bt
from kraken import backtest_trail as k_btt
from offline.backtests.comparison import (
    dominates,
    format_comparison_markdown,
    format_number,
    relative_to_baseline,
)
from offline.backtests.evaluation import (
    VENUE_REPLAY_DIRS,
    assess_decision_evidence,
    evaluate_segment,
    evaluate_walk_forward,
    load_venue_module,
    load_venue_replay_module,
    parse_dataset_specs,
    parse_intervals,
)
from offline.backtests.execution import ExecutionModel, split_order_fill
from offline.backtests.execution_calibration import _percentile, calibrate_execution_events
from offline.backtests.financial_benchmark import (
    VENUE_BENCHMARK_SCENARIOS,
    aggregate_financial_windows,
    default_scenarios,
)
from offline.backtests.hype_candidates import hype_240_candidates
from offline.backtests.metrics import format_performance_summary
from offline.backtests.promotion import _identity, _one_sided_sign_pvalue, evaluate_promotion
from offline.backtests.tradeall import BacktestBroker, run_backtest
from offline.backtests.walk_forward import summarize_test_windows, walk_forward_splits
from offline.research import (
    backtest_ranges as br,
    chop_weight_backtest as cwb,
    comprehensive_strategy_sweep as css,
    intelligence_backtest as ibt,
    slice_dense_archive as sda,
)
from offline.research.chop_weight_backtest import _precompute_chop_bar_tape, detect_regime_slope
from offline.research.compare_mhdsl_reentry_replay import (
    run_comparison,
    safe_float as mhdsl_safe_f,
    safe_nonneg_int as mhdsl_safe_int,
)
from offline.research.comprehensive_strategy_sweep import (
    _summarize_replay,
    define_candidate_grid,
)
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
from offline.research.comprehensive_surge_sweep.run_deep_surge_sweep import (
    safe_float as surge_safe_f,
    safe_nonneg_int as surge_safe_int,
)
from offline.research.hybrid_reentry import (
    harness as hrh,
    run_equities as req,
)
from offline.research.hybrid_reentry.harness import format_ablation_result, load_ohlc_csv
from offline.research.hybrid_reentry.report import _fmt_pct
from offline.research.hybrid_reentry.run_suite import (
    safe_float as rs_safe_f,
    safe_nonneg_int as rs_safe_int,
)
from offline.research.kraken_continuous_grid import (
    grid as k_grid,
    grid2 as k_grid2,
    report as k_rep,
)
from offline.research.kraken_continuous_grid.grid import load as k_grid_load
from offline.research.kraken_continuous_grid.report import q as k_report_q
from offline.research.kraken_continuous_grid.report2 import generate_report as gen_report2
from offline.research.kraken_continuous_grid.report3 import safe_float as k_rep3_safe_f
from offline.research.ml_forecast.forecast import (
    safe_float as fc_safe_f,
    safe_pos_int as fc_safe_int,
)
from offline.research.ml_forecast.priceprediction import PricePrediction
from offline.research.ml_forecast.vol_chronos import safe_pos_int as vc_safe_int
from offline.research.t212_continuous_grid import grid as t_grid
from offline.runners import (
    binance_walk_forward_baseline as bwf,
    hyperliquid_walk_forward_baseline as hwf,
    kraken_walk_forward_compare as kraken_compare,
    spot_profile_review as spr,
    walk_forward_compare,
)
from offline.runners.calibrate_execution_audit import format_number as cea_fmt
from offline.runners.financial_promotion_gate import _fmt_calmar_pct
from offline.runners.kraken_financial_benchmark import (
    format_number as kfb_fmt,
    markdown_report as hype_md_report,
    strat_params_from_dict,
)
from offline.runners.kraken_financial_compare import markdown_report
from offline.runners.t212_walk_forward_baseline import periods_per_year
from offline.runners.walk_forward import (
    SUPPORTED_MODES,
    SUPPORTED_VENUES,
    VENUE_RUNNER_MODULES,
    canonical_venue,
    dispatch,
    main,
    run_baseline,
    run_compare,
)
from offline.runners.walk_forward_compare import (
    DEFAULT_FEE_STRESS,
    VENUE_EVALUATOR_BUILDERS,
    _build_venue_evaluator,
    _detect_venue,
    _load_report,
    _markdown,
    _resolve_candidate_overrides,
    default_candidates,
)
from strategies.spot_engine import StratParams as SpotParams


class WalkForwardSplitsTest(unittest.TestCase):
    def test_rolling_windows_are_strictly_temporal(self):
        folds = walk_forward_splits(
            20, train_size=8, validation_size=3, test_size=2, step_size=2,
        )
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0].train, slice(0, 8))
        self.assertEqual(folds[0].validation, slice(8, 11))
        self.assertEqual(folds[0].test, slice(11, 13))
        self.assertEqual(folds[1].train, slice(2, 10))

        for fold in folds:
            self.assertEqual(fold.train.stop, fold.validation.start)
            self.assertEqual(fold.validation.stop, fold.test.start)
            self.assertLessEqual(fold.test.stop, 20)

    def test_anchored_train_expands_without_leaking_future_data(self):
        folds = walk_forward_splits(
            20, train_size=8, validation_size=3, test_size=2,
            step_size=2, anchored_train=True,
        )
        self.assertEqual(folds[0].train, slice(0, 8))
        self.assertEqual(folds[1].train, slice(0, 10))
        self.assertEqual(folds[1].validation, slice(10, 13))
        self.assertEqual(folds[1].test, slice(13, 15))

    def test_insufficient_history_returns_no_fold(self):
        self.assertEqual(
            walk_forward_splits(12, train_size=8, validation_size=3, test_size=2),
            [],
        )

    def test_invalid_sizes_fail_fast(self):
        with self.assertRaises(ValueError):
            walk_forward_splits(20, train_size=0, validation_size=3, test_size=2)
        with self.assertRaises(ValueError):
            walk_forward_splits(
                20, train_size=8, validation_size=3, test_size=2, step_size=-1,
            )

    def test_summary_preserves_worst_case_and_market_regimes(self):
        summary = summarize_test_windows([
            {"return_pct": 2.0, "max_drawdown_pct": 1.0,
             "buy_hold_return_pct": 5.0, "cycles": 2, "fills": 4},
            {"return_pct": -3.0, "max_drawdown_pct": 4.0,
             "buy_hold_return_pct": -10.0, "cycles": 1, "fills": 5},
            {"return_pct": 1.0, "max_drawdown_pct": 2.0,
             "buy_hold_return_pct": -1.0, "cycles": 0, "fills": 1},
        ])
        self.assertEqual(summary["window_count"], 3)
        self.assertEqual(summary["worst_return_pct"], -3.0)
        self.assertEqual(summary["worst_max_drawdown_pct"], 4.0)
        self.assertEqual(summary["positive_windows"], 2)
        self.assertEqual(summary["mean_return_up_market_pct"], 2.0)
        self.assertEqual(summary["mean_return_down_market_pct"], -1.0)
        self.assertEqual(summary["total_fills"], 10)

    def test_generic_evaluator_separates_signal_warmup_from_test_capital(self):
        records = [
            {"timestamp": index * 60, "open": 100, "high": 101, "low": 99, "close": 100 + index}
            for index in range(20)
        ]
        calls = []

        def replay(_ohlc, warmup, context):
            calls.append(len(warmup))
            self.assertEqual(len(context.timestamps), len(_ohlc))
            return {
                "return_pct": 1.0, "max_drawdown_pct": 2.0,
                "cycles": 1, "fills": 2, "sortino": None,
            }

        result = evaluate_walk_forward(
            records, replay, train_size=8, validation_size=3, test_size=2,
            step_size=2, warmup_bars=4,
        )
        self.assertEqual(result["aggregate_test"]["fold_count"], 4)
        self.assertIn(4, calls)
        self.assertTrue(all(fold["test"]["bars"] == 2 for fold in result["folds"]))

    def test_evidence_gate_separates_short_history_from_negative_worst_fold(self):
        records = [
            {"timestamp": index * 86_400, "open": 100, "high": 101,
             "low": 99, "close": 100}
            for index in range(60)
        ]
        evidence = assess_decision_evidence(records, {
            "fold_count": 3, "total_cycles": 0, "worst_return_pct": -0.62,
        })

        self.assertEqual(evidence["verdict"], "characterization_only_with_risk_flags")
        self.assertFalse(evidence["decision_grade_data"])
        self.assertEqual(len(evidence["data_issues"]), 3)
    def test_contract_validation_enforces_canonical_metrics(self):
        records = [
            {"timestamp": index * 60, "open": 100, "high": 101, "low": 99, "close": 100 + index}
            for index in range(5)
        ]

        def minimal_replay(_ohlc, _warmup, _ctx):
            return {
                "return_pct": 1.0, "max_drawdown_pct": 2.0,
                "cycles": 1, "fills": 2,
            }

        # validate_contract=False succeeds with minimal metrics
        res = evaluate_segment(records, minimal_replay, validate_contract=False)
        self.assertEqual(res["metrics"]["return_pct"], 1.0)

        # validate_contract=True fails when canonical metrics are missing
        with self.assertRaises(AssertionError):
            evaluate_segment(records, minimal_replay, validate_contract=True)

        def full_replay(_ohlc, _warmup, _ctx):
            return {
                "return_pct": 1.0,
                "max_drawdown_pct": 2.0,
                "cycles": 1,
                "fills": 2,
                "initial_capital": 1000.0,
                "account_currency": "USD",
                "net_pnl": 10.0,
                "realized": 10.0,
                "total": 10.0,
                "fees": 0.5,
                "ambiguous_bars": 0,
            }

        # validate_contract=True succeeds with full canonical metrics
        res_full = evaluate_segment(records, full_replay, validate_contract=True)
        self.assertEqual(res_full["metrics"]["total"], 10.0)


class WalkForwardCompareTest(TempDirTestCase):
    def test_detect_venue_explicit_field(self):
        self.assertEqual(_detect_venue({"venue": "binance"}), "binance")
        self.assertEqual(_detect_venue({"venue": "Hyperliquid"}), "hyperliquid")
        self.assertEqual(_detect_venue({"venue": "Trading212"}), "trading212")
        self.assertEqual(_detect_venue({"venue": "kraken"}), "kraken")

    def test_detect_venue_cli_override(self):
        self.assertEqual(
            _detect_venue({"venue": "kraken"}, cli_venue="binance"),
            "binance",
        )

    def test_detect_venue_purpose_heuristics(self):
        self.assertEqual(
            _detect_venue({"purpose": "Binance live-engine fixed-config walk-forward baseline"}),
            "binance",
        )
        self.assertEqual(
            _detect_venue({"purpose": "Hyperliquid spot walk-forward baseline"}),
            "hyperliquid",
        )
        self.assertEqual(
            _detect_venue({"purpose": "Trading 212 equity walk-forward baseline"}),
            "trading212",
        )
        self.assertEqual(
            _detect_venue({"purpose": "fixed-live-config walk-forward baseline; no parameter selection"}),
            "kraken",
        )

    def test_default_candidates_not_empty(self):
        candidates = default_candidates()
        self.assertTrue(len(candidates) >= 10)
        by_name = {c.name: c for c in candidates}
        self.assertIn("live", by_name)
        self.assertIn("classic_tp", by_name)
        self.assertIn("dca_drop_1_5", by_name)
        for fixed_trail_name in ("trail_2", "trail_4", "trail_5", "confirm_dca_1_5_trail_2"):
            self.assertIs(by_name[fixed_trail_name].overrides.get("tp_trail_adaptive"), False)
            self.assertIs(by_name[fixed_trail_name].overrides.get("tp_mdhtp_enabled"), False)

    def test_default_fee_stress_schedules(self):
        for venue in ("kraken", "binance", "hyperliquid", "trading212"):
            self.assertIn(venue, DEFAULT_FEE_STRESS)
            fees = [float(f) for f in DEFAULT_FEE_STRESS[venue].split(",")]
            self.assertTrue(all(f >= 0.0 for f in fees))
            self.assertEqual(fees, sorted(fees))

    def test_markdown_renders_title_with_venue(self):
        mock_report = {
            "source_baseline_report": "/tmp/test.json",
            "method": {
                "candidate_style": "standard",
                "comparison_dimensions": ["mean_return_pct"],
                "window_weighting": "equal",
                "execution_model": {},
            },
            "candidates": {
                "live": {
                    "summary": {
                        "mean_return_pct": 1.0,
                        "worst_return_pct": -5.0,
                        "worst_max_drawdown_pct": 8.0,
                    },
                    "relative_to_live": {
                        "delta_mean_return_pct": 0.0,
                        "delta_worst_return_pct": 0.0,
                        "delta_worst_drawdown_pct": 0.0,
                        "wins_vs_live": 0,
                        "ties_vs_live": 1,
                        "losses_vs_live": 0,
                    },
                    "dominates_live": False,
                    "by_interval": {},
                    "description": "live config",
                }
            },
            "pareto_frontier": ["live"],
            "dominates_live": [],
            "fee_stress": {},
        }
        md = _markdown(mock_report, venue="binance")
        self.assertIn("Binance walk-forward robustness comparison", md)
        self.assertIn("Pareto", md)

    def test_kraken_shim_compatibility(self):
        self.assertIs(kraken_compare.main, walk_forward_compare.main)

    def test_load_report_supports_both_schemas(self):
        report_path = self.tmp / "report.json"
        for payload in (
            {"schema_version": 1, "intervals": {"240": {}}},
            {"schema_version": 1, "dataset": {}, "windows": {}},
            {"schema_version": 1, "dataset": {}, "window_sizes": {}},
        ):
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            rep = _load_report(report_path)
            self.assertEqual(rep["schema_version"], 1)

        report_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_report(report_path)

    def test_detect_venue_trading212_no_space_and_ticker(self):
        self.assertEqual(
            _detect_venue({"purpose": "Trading212 live-engine fixed-config walk-forward baseline"}),
            "trading212",
        )
        self.assertEqual(
            _detect_venue({"runtime": {"ticker": "NVDA_US_EQ"}}),
            "trading212",
        )

    def test_format_number_non_finite(self):
        self.assertEqual(format_number(None), "n/a")
        self.assertEqual(format_number(float("nan")), "n/a")
        self.assertEqual(format_number(float("inf")), "n/a")
        self.assertEqual(format_number(-float("inf")), "n/a")
        self.assertEqual(format_number(1.23456), "1.235")
        self.assertEqual(format_number(1.23456, signed=True), "+1.235")

    def test_summarize_test_windows_handles_none_buy_hold(self):
        windows = [
            {"return_pct": 5.0, "max_drawdown_pct": 2.0, "buy_hold_return_pct": None, "cycles": 1, "fills": 2},
            {"return_pct": -3.0, "max_drawdown_pct": 4.0, "buy_hold_return_pct": 1.5, "cycles": 1, "fills": 3},
        ]
        summary = summarize_test_windows(windows)
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["up_market_windows"], 1)
        self.assertEqual(summary["down_market_windows"], 0)

    def test_build_venue_evaluator_hyperliquid_spot(self):
        source_report = {
            "venue": "hyperliquid",
            "runtime": {"engine": "spot", "coin": "HYPE"},
            "strategy_params": {
                "pair": "HYPEUSDC",
                "entry_amount": 50.0,
                "takeprofit_pct": 5.0,
                "tp_mdhtp_enabled": True,
            },
            "fee_pct_per_leg": 0.035,
        }
        base_params, base_fee, segment_fn = _build_venue_evaluator("hyperliquid", source_report)
        self.assertEqual(base_fee, 0.035)
        self.assertEqual(base_params.currency, "USDC")
        self.assertTrue(base_params.tp_mdhtp_enabled)
        self.assertTrue(callable(segment_fn))

    def test_runners_and_backtest_broker_guards(self):
        for fn in (kfb_fmt, cea_fmt):
            self.assertEqual(fn(None), "n/a")
            self.assertEqual(fn(True), "n/a")
            self.assertEqual(fn(float("nan")), "n/a")
            self.assertEqual(fn(float("inf")), "n/a")

        with self.assertRaises(ValueError):
            periods_per_year(0)
        with self.assertRaises(ValueError):
            periods_per_year(False)

        broker = BacktestBroker(self.tmp, lambda: 1000.0)
        self.assertIsNone(broker._execute("BUY", "BTCUSDC", 0.0))
        self.assertIsNone(broker._execute("BUY", "BTCUSDC", float("nan")))
        self.assertIsNone(broker._execute("BUY", "BTCUSDC", 100.0, qty=0.0))
        self.assertIsNone(broker._execute("BUY", "BTCUSDC", True))

        with self.assertRaises(ValueError):
            run_backtest("BTCUSDC", 0.0, 1.0, "fast", "", "history")
        with self.assertRaises(ValueError):
            run_backtest("BTCUSDC", 0.0, 1.0, "fast", "..", "history")


class TestWalkForwardFacade(unittest.TestCase):
    def test_canonical_venue_normalization(self):
        self.assertEqual(canonical_venue("kraken"), "kraken")
        self.assertEqual(canonical_venue("Kraken"), "kraken")
        self.assertEqual(canonical_venue("  binance  "), "binance")
        self.assertEqual(canonical_venue("hyperliquid"), "hyperliquid")
        self.assertEqual(canonical_venue("t212"), "t212")
        self.assertEqual(canonical_venue("trading212"), "t212")
        self.assertEqual(canonical_venue("Trading212"), "t212")
        self.assertIsNone(canonical_venue(None))

    def test_supported_registries(self):
        for v in ("kraken", "binance", "hyperliquid", "t212", "trading212"):
            self.assertIn(v, SUPPORTED_VENUES)
            self.assertIn(v, VENUE_RUNNER_MODULES)
        for m in ("baseline", "compare"):
            self.assertIn(m, SUPPORTED_MODES)

    def test_run_baseline_unsupported_venue(self):
        with self.assertRaises(ValueError) as ctx:
            run_baseline("unknown_venue", [])
        self.assertIn("Unsupported baseline venue", str(ctx.exception))

    @patch("offline.runners.binance_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_binance(self, mock_binance_main):
        ret = run_baseline("binance", ["--symbol", "BTCUSDT", "--fee", "0.08"])
        self.assertEqual(ret, 0)
        mock_binance_main.assert_called_once_with(["--symbol", "BTCUSDT", "--fee", "0.08"])

    @patch("offline.runners.kraken_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_kraken(self, mock_kraken_main):
        ret = run_baseline("kraken", ["--pair", "XBT/EUR", "--fee", "0.26"])
        self.assertEqual(ret, 0)
        mock_kraken_main.assert_called_once_with(["--pair", "XBT/EUR", "--fee", "0.26"])

    @patch("offline.runners.hyperliquid_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_hyperliquid(self, mock_hl_main):
        ret = run_baseline("hyperliquid", ["--coin", "HYPE", "--intervals", "240"])
        self.assertEqual(ret, 0)
        mock_hl_main.assert_called_once_with(["--coin", "HYPE", "--intervals", "240"])

    @patch("offline.runners.t212_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_t212(self, mock_t212_main):
        ret = run_baseline("t212", ["--profile", "spcx", "--interval", "5m"])
        self.assertEqual(ret, 0)
        mock_t212_main.assert_called_once_with(["--profile", "spcx", "--interval", "5m"])

    @patch("offline.runners.t212_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_trading212_alias(self, mock_t212_main):
        ret = run_baseline("trading212", ["--profile", "nvda"])
        self.assertEqual(ret, 0)
        mock_t212_main.assert_called_once_with(["--profile", "nvda"])

    @patch("offline.runners.walk_forward_compare.main", return_value=0)
    def test_run_compare(self, mock_compare_main):
        ret = run_compare(["sample_report.json", "--fee-stress", "0.1,0.2"])
        self.assertEqual(ret, 0)
        mock_compare_main.assert_called_once_with(["sample_report.json", "--fee-stress", "0.1,0.2"])

    @patch("offline.runners.walk_forward_compare.main", return_value=0)
    def test_run_compare_with_venue_override(self, mock_compare_main):
        ret = run_compare(["sample_report.json"], venue="trading212")
        self.assertEqual(ret, 0)
        mock_compare_main.assert_called_once_with(["sample_report.json", "--venue", "t212"])

    @patch("offline.runners.binance_walk_forward_baseline.main", return_value=0)
    def test_dispatch_baseline_mode(self, mock_binance_main):
        code = dispatch(["--venue", "binance", "--symbol", "ETHUSDT"])
        self.assertEqual(code, 0)
        mock_binance_main.assert_called_once_with(["--symbol", "ETHUSDT"])

    @patch("offline.runners.walk_forward_compare.main", return_value=0)
    def test_dispatch_compare_mode(self, mock_compare_main):
        code = dispatch(["--mode", "compare", "report.json", "--candidate-set", "standard"])
        self.assertEqual(code, 0)
        mock_compare_main.assert_called_once_with(["report.json", "--candidate-set", "standard"])

    def test_dispatch_baseline_requires_venue(self):
        with self.assertRaises(SystemExit):
            dispatch(["--mode", "baseline", "--symbol", "ETHUSDT"])

    def test_dispatch_top_level_help(self):
        code = dispatch(["--help"])
        self.assertEqual(code, 0)

    @patch("offline.runners.walk_forward.dispatch", return_value=42)
    def test_main_entrypoint(self, mock_dispatch):
        res = main(["--venue", "kraken"])
        self.assertEqual(res, 42)
        mock_dispatch.assert_called_once_with(["--venue", "kraken"])


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_intelligence_backtest_validates_step_seconds_initial_balance_fee_pct_iter72_s1():
    """[ITER72 #1] intelligence_backtest validates step_seconds, initial_balance, fee_pct."""
    # 1. intelligence_backtest validates step_seconds, initial_balance, fee_pct
    with pytest.raises(ValueError):
        ibt._precompute_intelligence_tape("BTCUSDC", [], 0, 3.5)
    with pytest.raises(ValueError):
        ibt.run_intelligence_backtest("BTCUSDC", step_seconds=0)
    with pytest.raises(ValueError):
        ibt.run_intelligence_backtest("BTCUSDC", initial_balance=0.0)
    with pytest.raises(ValueError):
        ibt.run_intelligence_backtest("BTCUSDC", fee_pct=1.0)

def test_chop_weight_backtest_filters_non_finite_non_positive_prices_and_iter72_s2():
    """[ITER72 #2] chop_weight_backtest filters non-finite/non-positive prices and validates params."""
    # 2. chop_weight_backtest filters non-finite/non-positive prices and validates params
    reg, slp = cwb.detect_regime_slope(np.array([float("nan"), -5.0, 100.0, 101.0]))
    assert reg == "CHOP" and slp == 0.0
    with pytest.raises(ValueError):
        cwb._precompute_chop_bar_tape("BTCUSDC", [], None, 0)
    with pytest.raises(ValueError):
        cwb.run_single_simulation("BTCUSDC", [], None, "CHOP_0.03", initial_balance=0.0)
    with pytest.raises(ValueError):
        cwb.run_single_simulation("BTCUSDC", [], None, "CHOP_0.03", fee_pct=-0.1)

def test_slice_dense_archive_ts_of_handles_overflowerror_iter72_s3():
    """[ITER72 #3] slice_dense_archive.ts_of handles OverflowError."""
    # 3. slice_dense_archive.ts_of handles OverflowError
    assert sda.ts_of({"t": 10**400}) == 0.0
    assert sda.ts_of({"i": [1700000000000, 100.0]}) == 1700000000000.0

def test_comprehensive_strategy_sweep__summarize_replay_handles_nan_bool_non_positive_budget_iter72_s4():
    """[ITER72 #4] comprehensive_strategy_sweep._summarize_replay handles NaN/bool/non-positive budget."""
    # 4. comprehensive_strategy_sweep._summarize_replay handles NaN/bool/non-positive budget
    summary = css._summarize_replay({"net": 50.0, "total": 60.0}, float("nan"))
    assert math.isfinite(summary["net_pct"])

def test_hybrid_reentry_harness_load_ohlc_csv_rejects_inverted_high_low_bars_iter73_s4(tmp_path):
    """[ITER73 #4] hybrid_reentry.harness.load_ohlc_csv rejects inverted high < low bars and format_ablation_result guards capital."""
    # 4. hybrid_reentry.harness.load_ohlc_csv rejects inverted high < low bars and format_ablation_result guards capital
    csv_f = tmp_path / "ohlc.csv"
    csv_f.write_text("timestamp,open,high,low,close\n1,100,90,95,92\n2,100,105,98,103\n", encoding="utf-8")
    loaded_rows = hrh.load_ohlc_csv(str(csv_f))
    assert loaded_rows == [(100.0, 105.0, 98.0, 103.0)]
    fmt_res = hrh.format_ablation_result("BTC", "Test 0", "Control", {"total": 100.0}, float("inf"))
    assert fmt_res["total_return_pct"] == 0.0

def test_run_deep_surge_sweep__win_move_at_ts_rejects_negative_timestamps_iter73_s5():
    """[ITER73 #5] run_deep_surge_sweep._win_move_at_ts rejects negative timestamps."""
    # 5. run_deep_surge_sweep._win_move_at_ts rejects negative timestamps
    assert rdss._win_move_at_ts([100.0] * 50, 30, -14400.0, 48.0) is None

def test_kraken_continuous_grid_report_q_and_win_stats_filter_nan_and_iter74_s5():
    """[ITER74 #5] kraken_continuous_grid.report.q and win_stats filter NaN and bool."""
    # 5. kraken_continuous_grid.report.q and win_stats filter NaN and bool
    assert k_rep.q([float("nan"), True, 10.0, 20.0], 0.5) == 10.0
    st = k_rep.win_stats({"w30": [(float("nan"), 0.0, 5.0), (5.0, 1.0, 15.0)]}, "w30")
    assert st["mean"] == 5.0 and st["pos"] == 100.0

def test_spot_profile_review_revision_params_rejects_empty_or_flag_like_specs_iter75_s1():
    """[ITER75 #1] spot_profile_review.revision_params rejects empty or flag-like specs."""
    # 1. spot_profile_review.revision_params rejects empty or flag-like specs
    with pytest.raises(ValueError, match="Invalid git revision spec"):
        spr.revision_params("")
    with pytest.raises(ValueError, match="Invalid git revision spec"):
        spr.revision_params("--help")

def test_backtest_ranges_scan_backtest_ranges_handles_export_key_val_crlf_and_iter75_s2(tmp_path):
    """[ITER75 #2] backtest_ranges.scan_backtest_ranges handles export KEY=val, CRLF, and inline comments."""
    # 2. backtest_ranges.scan_backtest_ranges handles export KEY=val, CRLF, and inline comments
    cfg_f = tmp_path / "test.env"
    cfg_f.write_bytes(b"# BACKTEST: 1.0, 2.0, 3.0 # comment\r\nexport MY_PARAM = 2.0\r\n")
    ranges = br.scan_backtest_ranges(str(cfg_f))
    assert ranges == {"MY_PARAM": ["1.0", "2.0", "3.0"]}

def test_hybrid_reentry_run_equities__import_t212_restores_sys_path_and_sys_iter75_s4():
    """[ITER75 #4] hybrid_reentry.run_equities._import_t212 restores sys.path and sys.modules."""
    # 4. hybrid_reentry.run_equities._import_t212 restores sys.path and sys.modules
    path_before = list(sys.path)
    strat_mod, rp_mod = req._import_t212()
    assert hasattr(strat_mod, "StratParams")
    assert hasattr(rp_mod, "run_replay")
    assert sys.path == path_before

def test_kraken_backtest__bt_params_skips_none_fallback_keys_and_iter119_s3():
    """[ITER119 #3] kraken.backtest._bt_params skips None fallback keys and rejects boolean values/intervals."""
    # 3. kraken.backtest._bt_params skips None fallback keys and rejects boolean values/intervals
    bt_p = k_bt._bt_params({"drop_fallback": None, "drop": 3.5, "reentry_fallback": None, "reentry_drop": 1.25})
    assert bt_p.dca_drop_pct == 3.5
    assert bt_p.reentry_drop_pct == 1.25
    with pytest.raises(ValueError):
        k_bt._bt_params({"entry": True})
    with pytest.raises(ValueError):
        k_bt._bt_params({"interval": True})

def test_executionmodel_limit_touched_market_price_and_split_order_fill_iter129_s2():
    """[ITER129 #2] ExecutionModel.limit_touched / market_price and split_order_fill."""
    # 2. ExecutionModel.limit_touched / market_price and split_order_fill
    em = ExecutionModel()
    assert em.limit_touched(" BUY ", high=105.0, low=95.0, limit=100.0) is True
    with pytest.raises(ValueError, match="side invalid"):
        em.limit_touched(None, high=105.0, low=95.0, limit=100.0)
    with pytest.raises(ValueError, match="side invalid"):
        em.market_price(True, 100.0)
    with pytest.raises(ValueError, match="ratio must be positive and finite"):
        split_order_fill({"vol": 10.0}, quantity_key="vol", ratio=0.0)
    with pytest.raises(ValueError, match="ratio must be positive and finite"):
        split_order_fill({"vol": 10.0}, quantity_key="vol", ratio=True)

def test_format_performance_summary_renders_nan_bool_non_numeric_metrics_as_iter129_s3():
    """[ITER129 #3] format_performance_summary renders NaN/bool/non-numeric metrics as n/a."""
    # 3. format_performance_summary renders NaN/bool/non-numeric metrics as n/a
    lines = format_performance_summary(
        {"max_drawdown_pct": float("nan"), "sharpe": True, "sortino": "bad", "exposure_pct": 42.5}
    )
    assert "maxDD n/a%" in lines[0]
    assert "Sharpe n/a" in lines[0]
    assert "Sortino n/a" in lines[0]
    assert "exposure 42.50%" in lines[1]

def test_parse_intervals_and_parse_dataset_specs_reject_non_numeric_non_positive_iter129_s4():
    """[ITER129 #4] parse_intervals and parse_dataset_specs reject non-numeric / non-positive / empty path specs."""
    # 4. parse_intervals and parse_dataset_specs reject non-numeric / non-positive / empty path specs
    with pytest.raises(argparse.ArgumentTypeError):
        parse_intervals("60,abc")
    with pytest.raises(ValueError, match="invalid dataset spec"):
        parse_dataset_specs(["0=/tmp/data.json"])
    with pytest.raises(ValueError, match="invalid dataset spec"):
        parse_dataset_specs(["60=   "])

def test_default_scenarios_handles_bool_provider_objects_whitespace_strings_iter132_s1():
    """[ITER132 #1] default_scenarios handles bool / provider objects / whitespace strings."""
    # 1. default_scenarios handles bool / provider objects / whitespace strings
    s_bool = default_scenarios(True)  # type: ignore[arg-type]
    assert s_bool[0].fees.limit_fee_pct == pytest.approx(0.16)
    s_obj = default_scenarios(SimpleNamespace(name=" BINANCE "))  # type: ignore[arg-type]
    assert s_obj[0].fees.limit_fee_pct == pytest.approx(0.075)

def test_aggregate_financial_windows_sanitizes_nan_bool_net_pnl_and_float_string_iter132_s2():
    """[ITER132 #2] aggregate_financial_windows sanitizes NaN/bool net_pnl and float-string trade_count/bars."""
    # 2. aggregate_financial_windows sanitizes NaN/bool net_pnl and float-string trade_count/bars
    agg = aggregate_financial_windows(
        [
            {
                "key": "w1",
                "return_pct": 2.0,
                "max_drawdown_pct": 1.0,
                "buy_hold_return_pct": 4.0,
                "cycles": 1,
                "fills": 2,
                "bars": "120.0",
                "metrics": {
                    "net_pnl": float("nan"),
                    "trade_count": "3.0",
                    "max_underwater_periods": "15.0",
                },
            }
        ],
        initial_capital=1000.0,
    )
    assert agg["mean_net_pnl_usd_per_window"] == pytest.approx(0.0)
    assert agg["total_trade_count"] == 3
    assert agg["worst_underwater_bars"] == 15
    assert agg["total_test_bars"] == 120

def test__percentile_clamps_out_of_bounds_percentiles_and_filters_iter132_s3():
    """[ITER132 #3] _percentile clamps out-of-bounds percentiles and filters non-finite/bool values."""
    # 3. _percentile clamps out-of-bounds percentiles and filters non-finite/bool values
    assert _percentile([10.0, 20.0, 30.0], -10.0) == pytest.approx(10.0)
    assert _percentile([10.0, 20.0, 30.0], 150.0) == pytest.approx(30.0)
    assert _percentile([10.0, float("nan"), True, 30.0], 50.0) == pytest.approx(20.0)  # type: ignore[list-item]
    assert _percentile([10.0, 20.0], True) is None  # type: ignore[arg-type]

def test_calibrate_execution_events_normalizes_string_false_for_market_flag_iter132_s4():
    """[ITER132 #4] calibrate_execution_events normalizes string 'false' for market flag."""
    # 4. calibrate_execution_events normalizes string "false" for market flag
    cal = calibrate_execution_events(
        [
            {
                "intent_id": "int-1",
                "event": "submit_requested",
                "ts": 100.0,
                "venue": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 1.0,
                "price": 50000.0,
                "reference_price": 50000.0,
                "market": "false",
            },
            {
                "intent_id": "int-1",
                "event": "order_terminal",
                "ts": 101.0,
                "filled_qty": 1.0,
                "cost": 50050.0,
                "fee": 0.5,
            },
        ]
    )
    assert len(cal["orders"]) == 1
    assert cal["orders"][0]["market"] is False
    assert cal["orders"][0]["market_execution_shortfall_bps"] is None
    assert cal["summary"]["limit"]["orders"] == 1
    assert cal["summary"]["market"]["orders"] == 0

def test_comparison_tolerance_sanitization_and_format_comparison_markdown_non_mapping_candidate_iter132_s5():
    """[ITER132 #5] comparison tolerance sanitization and format_comparison_markdown non-Mapping candidate resilience."""
    # 5. comparison tolerance sanitization and format_comparison_markdown non-Mapping candidate resilience
    rel = relative_to_baseline(
        [{"key": "w1", "return_pct": 1.0}],
        [{"key": "w1", "return_pct": 1.0}],
        tolerance=-5.0,
    )
    assert rel["ties_vs_live"] == 1
    assert (
        dominates(
            {"mean_return_pct": 2.0, "worst_return_pct": 1.0, "worst_max_drawdown_pct": 1.0},
            {"mean_return_pct": 1.0, "worst_return_pct": 1.0, "worst_max_drawdown_pct": 1.0},
            tolerance=float("nan"),
        )
        is True
    )
    md = format_comparison_markdown(
        {
            "candidates": {"live": None, "cand1": {"summary": None, "relative_to_live": None}},
            "fee_stress": {"0.10": None},
        }
    )
    assert "| `live` | n/a |" in md
    assert "| `cand1` | n/a |" in md

def test__one_sided_sign_pvalue_does_not_overflow_for_active_and_handles_iter134_s1():
    """[ITER134 #1] _one_sided_sign_pvalue does not overflow for active >= 1024 and handles negative/bool inputs."""
    # 1. _one_sided_sign_pvalue does not overflow for active >= 1024 and handles negative/bool inputs
    assert 0.0 <= _one_sided_sign_pvalue(600, 600) <= 1.0
    assert _one_sided_sign_pvalue(-5, 0) == pytest.approx(1.0)
    assert _identity({"dataset": None, "walk_forward": None}) == (None,) * 9

    # Duplicate window keys in scenario are rejected by evaluate_promotion
    base_rep = {
        "dataset": {"sha256": "h1", "bars": 100},
        "walk_forward": {"train": 10, "validation": 5, "test": 5, "step": 5, "warmup": 2},
        "scenarios": {
            "central": {
                "assumptions": {},
                "aggregate": {},
                "windows": [{"key": "w1", "return_pct": 1.0}, {"key": "w1", "return_pct": 2.0}],
            }
        },
    }
    cand_rep = {
        "dataset": {"sha256": "h1", "bars": 100},
        "walk_forward": {"train": 10, "validation": 5, "test": 5, "step": 5, "warmup": 2},
        "scenarios": {
            "central": {
                "assumptions": {},
                "aggregate": {},
                "windows": [{"key": "w1", "return_pct": 1.5}],
            }
        },
    }
    with pytest.raises(ValueError, match="different windows"):
        evaluate_promotion(base_rep, cand_rep)

def test_kraken_financial_compare_markdown_report_renders_nan_calmar_delta_as_n_a_iter136_s3():
    """[ITER136 #3] kraken_financial_compare.markdown_report renders NaN calmar_delta as 'n/a' instead of 'n/a%'."""
    # 3. kraken_financial_compare.markdown_report renders NaN calmar_delta as 'n/a' instead of 'n/a%'
    md = markdown_report(
        {
            "candidates": {
                "live": {
                    "scenarios": {
                        "central": {"mean_return_pct": 1.0, "worst_return_pct": -1.0},
                    },
                    "promotion_gate": {
                        "return_gate": {
                            "scenarios": {
                                "central": {
                                    "mean_return_delta_pp": 0.0,
                                    "worst_return_delta_pp": 0.0,
                                    "worst_drawdown_delta_pp": 0.0,
                                    "wins_ties_losses": {"wins": 0, "ties": 1, "losses": 0},
                                    "passed": False,
                                }
                            }
                        },
                        "defensive_gate": {
                            "scenarios": {
                                "central": {
                                    "drawdown_wins_ties_losses": {"wins": 0, "ties": 1, "losses": 0},
                                    "median_calmar_improvement_ratio": float("nan"),
                                    "passed": False,
                                }
                            }
                        },
                    },
                }
            },
            "promotable": [],
        }
    )
    assert "n/a%" not in md
    assert "| n/a |" in md

def test_run_intelligence_backtest_rejects_invalid_parabolic_surge_pct_iter139_s2():
    """[ITER139 #2] run_intelligence_backtest rejects invalid parabolic_surge_pct."""
    # 2. run_intelligence_backtest rejects invalid parabolic_surge_pct
    for bad_surge in (0.0, -2.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="parabolic_surge_pct"):
            ibt.run_intelligence_backtest(parabolic_surge_pct=bad_surge)  # type: ignore[arg-type]

def test_chop_weight_backtest_filters_non_finite_bool_bars_and_sanitizes_iter139_s3():
    """[ITER139 #3] chop_weight_backtest filters non-finite/bool bars and sanitizes mk_alpha."""
    # 3. chop_weight_backtest filters non-finite/bool bars and sanitizes mk_alpha
    reg, slope = detect_regime_slope(np.linspace(100.0, 110.0, 30), mk_alpha=float("nan"))
    assert reg in ("BULL", "CHOP")
    assert math.isfinite(slope)
    tape = _precompute_chop_bar_tape(
        "BTCUSDC",
        [(1000.0, float("nan")), (1900.0, -5.0), (2800.0, True), (3700.0, 50000.0)],  # type: ignore[list-item]
        None,
        step_seconds=900,
    )
    assert len(tape) == 1
    assert tape[0]["p"] == pytest.approx(50000.0)

def test__summarize_replay_handles_nan_bool_metrics_and_float_string_iter139_s4():
    """[ITER139 #4] _summarize_replay handles NaN/bool metrics and float-string counts."""
    # 4. _summarize_replay handles NaN/bool metrics and float-string counts
    sum_m = _summarize_replay(
        {"net": float("nan"), "total": True, "max_drawdown_pct": "bad", "cycles": "4.0", "wins": "2.0", "fills": "8.0"},
        1000.0,
    )
    assert sum_m["net_pct"] == pytest.approx(0.0)
    assert sum_m["cycles"] == 4
    assert sum_m["wins"] == 2
    assert sum_m["fills"] == 8

def test_slice_dense_archive_ts_of_accepts_tuple_i_field_iter139_s5():
    """[ITER139 #5] slice_dense_archive.ts_of accepts tuple 'i' field."""
    # 5. slice_dense_archive.ts_of accepts tuple 'i' field
    assert sda.ts_of({"i": (1700000000000, 50000.0)}) == pytest.approx(1700000000000.0)

def test_load_ohlc_csv_format_ablation_result_iter142_s1(tmp_path):
    """[ITER142 #1] load_ohlc_csv & format_ablation_result."""
    # 1. load_ohlc_csv & format_ablation_result
    assert load_ohlc_csv("") == []
    assert load_ohlc_csv(True) == []
    assert load_ohlc_csv(str(tmp_path / "missing.csv")) == []

    row = format_ablation_result(
        "BTCUSDC",
        "Test 3",
        "Hybrid",
        {
            "total": float("nan"),
            "max_drawdown_pct": True,
            "exposure_pct": "12.5",
            "cycles": "4.0",
            "wins": "3.0",
            "win_rate_pct": float("nan"),
            "profit_factor": "bad",
            "fills": "8.0",
        },
        capital=3900.0,
    )
    assert row["net_profit_usd"] == 0.0
    assert row["max_drawdown_pct"] == 0.0
    assert row["exposure_pct"] == 12.5
    assert row["cycles"] == 4
    assert row["wins"] == 3
    assert row["win_rate"] == 75.0
    assert row["fills"] == 8

def test__fmt_pct_iter142_s2():
    """[ITER142 #2] _fmt_pct."""
    # 2. _fmt_pct
    assert _fmt_pct(float("nan"), signed=True) == "N/A"
    assert _fmt_pct(True, signed=True) == "N/A"
    assert _fmt_pct("bad", signed=True) == "N/A"
    assert _fmt_pct(4.5, decimals=2, signed=True, bold=True) == "**+4.50%**"

def test__fmt_calmar_pct_strat_params_from_dict_and_hype_md_report_iter142_s4():
    """[ITER142 #4] _fmt_calmar_pct, strat_params_from_dict, and hype_md_report."""
    # 4. _fmt_calmar_pct, strat_params_from_dict, and hype_md_report
    assert _fmt_calmar_pct(float("nan")) == "n/a"
    assert _fmt_calmar_pct(True) == "n/a"
    assert _fmt_calmar_pct(0.125) == "+12.5%"

    sp = strat_params_from_dict({
        "currency": "USD",
        "entry_amount": 350.0,
        "entry_discount_pct": 0.2,
        "dca_amount": 100.0,
        "dca_drop_pct": 2.0,
        "check_minutes": 5,
        "takeprofit_pct": 5.0,
        "max_budget": 1050.0,
        "max_dca_buys": 7,
        "enable_takeprofit": True,
        "order_ttl_min": 60,
        "legacy_removed_field": 123,
    })
    assert sp.sl_mhdsl_enabled is False
    md = hype_md_report({"candidate_name": "empty_test", "scenarios": {}, "walk_forward": {"interval_minutes": 240}})
    assert "0 TEST bars" in md
    assert "0 days without overlap" in md

def test_iteration143_hl_place_order_int_oid_and_research_trigger_gate_safety():
    # 1. compare_mhdsl_reentry_replay and run_deep_surge_sweep helpers
    assert run_comparison(True) == []
    assert mhdsl_safe_f(float("nan"), 1.5) == 1.5
    assert mhdsl_safe_int("4.0", 0) == 4
    assert surge_safe_f(True, 2.5) == 2.5
    assert surge_safe_int("7.0", 0) == 7

def test_kraken_continuous_grid_grid_load_report_q_iter145_s1(tmp_path):
    """[ITER145 #1] kraken_continuous_grid.grid.load & report.q."""
    # 1. kraken_continuous_grid.grid.load & report.q
    assert k_grid_load(True) == []
    csv_p = tmp_path / "test_240m.csv"
    csv_p.write_text(
        "timestamp,open,high,low,close\n"
        "1700000000,100.0,90.0,95.0,98.0\n"  # inverted high < open
        "1700014400,100.0,105.0,98.0,102.0\n",
        encoding="utf-8",
    )
    loaded = k_grid_load(str(csv_p))
    assert len(loaded) == 1
    assert loaded[0][0] == 1700014400

    assert math.isnan(k_report_q([1.0, 2.0, 3.0], float("nan")))
    assert math.isnan(k_report_q([1.0, 2.0, 3.0], True))
    assert k_report_q([1.0, 2.0, 3.0], -5.0) == 1.0
    assert k_report_q([1.0, 2.0, 3.0], 5.0) == 3.0

def test_kraken_continuous_grid_report3__safe_f_iter145_s2():
    """[ITER145 #2] kraken_continuous_grid.report3._safe_f."""
    # 2. kraken_continuous_grid.report3._safe_f
    assert k_rep3_safe_f(None, -1.0) == -1.0
    assert k_rep3_safe_f(True, -1.0) == -1.0
    assert k_rep3_safe_f(float("nan"), -1.0) == -1.0

def test_ml_forecast_helpers_priceprediction___init___iter145_s3():
    """[ITER145 #3] ml_forecast helpers & PricePrediction.__init__."""
    # 3. ml_forecast helpers & PricePrediction.__init__
    assert fc_safe_f("bad", 0.7) == 0.7
    assert fc_safe_f(True, 0.7) == 0.7
    assert fc_safe_int("168.0", 24) == 168
    assert vc_safe_int("bad", 24) == 24
    assert vc_safe_int(True, 24) == 24

    pp = PricePrediction(window_size="bad", hidden_size="bad", lr="bad", epochs="bad")
    assert pp.window_size == 10
    assert pp.epochs == 30

def test_report2_generate_report_with_none_bool_nan_values_iter148_s1(tmp_path):
    """[ITER148 #1] report2.generate_report with None/bool/NaN values."""
    # 1. report2.generate_report with None/bool/NaN values
    exp_dir = tmp_path / "grid2_exp"
    exp_dir.mkdir()
    r2_file = exp_dir / "results2.jsonl"
    r2_file.write_text(
        json.dumps(
            {
                "kind": "select",
                "asset": "BTC",
                "steps": [
                    {"pick": "ov650_t8", "pick_ret": None, "oos": {"live": None, "ov650_t8": 4.5}},
                    {"pick": "live", "pick_ret": 2.0, "oos": "corrupt"},
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "views",
                "fee": "std",
                "asset": "BTC",
                "cfg": "live",
                "full": {"ret": None, "dd": 10.0},
                "w90": [[1.0]],
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "views",
                "fee": "std",
                "asset": "BTC",
                "cfg": "ov650_t8",
                "full": {"ret": 12.0, "dd": 8.0},
                "w90": [[2.0, 5.0, 1.0, 2], ["bad"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lines = gen_report2(str(exp_dir))
    assert any("BTC" in line for line in lines)

def test_hybrid_reentry_run_suite_helpers_iter148_s3():
    """[ITER148 #3] hybrid_reentry.run_suite helpers."""
    # 3. hybrid_reentry.run_suite helpers
    assert rs_safe_f(True, -1.0) == -1.0
    assert rs_safe_int("4.0", 0) == 4
    assert rs_safe_int(True, 0) == 0

def test_walk_forward_compare__build_venue_evaluator_fee_ppy_sanitization_iter150_s1():
    """[ITER150 #1] walk_forward_compare._build_venue_evaluator fee & ppy sanitization."""
    # 1. walk_forward_compare._build_venue_evaluator fee & ppy sanitization
    _p, fee_k, _fn = _build_venue_evaluator("kraken", {"fee_pct_per_leg": True})
    assert fee_k == 0.26
    _p_t, fee_t, _fn_t = _build_venue_evaluator(
        "t212",
        {
            "strategy_params": {
                "currency": "USD", "yahoo_sym": "NVDA", "entry_pct": 10.0, "dca_pct": 10.0,
                "entry_discount_pct": 0.2, "dca_drop_pct": 2.0, "check_minutes": 5.0,
                "takeprofit_pct": 5.0, "max_budget": 1000.0, "max_dca_buys": 5,
                "order_ttl_min": 10.0, "stop_loss_pct": 10.0, "fx_fee_pct": 0.15,
            },
            "fee_pct_per_leg": float("nan"),
            "execution_model": {"calendar_periods_per_year": "bad"},
        },
    )
    assert fee_t == 0.0

def test_kraken_backtest__bt_params_float_string_maxdca_and_nan_iter150_s3():
    """[ITER150 #3] kraken.backtest._bt_params float-string maxdca and NaN rejection."""
    # 3. kraken.backtest._bt_params float-string maxdca and NaN rejection
    bt_p = k_bt._bt_params({"entry": "150.0", "maxdca": "8.0"})
    assert bt_p.entry_amount == 150.0
    assert bt_p.max_dca_buys == 8
    with pytest.raises(ValueError):
        k_bt._bt_params({"entry": float("nan")})


def test_wave2_cross_venue_candidate_resolution_and_unmasked_trailing_variants(tmp_path):
    """Verify cross-venue candidate mapping, baseline lookbacks, and unmasked trailing configs."""
    # 1. Kraken-scale params preserve historical 500/250 lower_sizing; smaller profiles scale 0.75x.
    kraken_base = SpotParams(
        currency="USD", entry_amount=650.0, entry_discount_pct=0.2, dca_amount=325.0,
        dca_drop_pct=1.25, check_minutes=1.0, takeprofit_pct=4.5, max_budget=3900.0,
        max_dca_buys=10, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=25.0,
    )
    spot_fields = {f.name for f in dataclasses.fields(SpotParams)}
    lower_cand = next(c for c in default_candidates() if c.name == "lower_sizing")
    k_over, k_desc = _resolve_candidate_overrides(lower_cand, kraken_base, spot_fields)
    assert k_over == {"entry_amount": 500.0, "dca_amount": 250.0}
    assert k_desc == "entry 500 + DCA 250 USD"

    hl_base = dataclasses.replace(
        kraken_base, currency="USDC", entry_amount=350.0, dca_amount=100.0, max_budget=1050.0,
    )
    hl_over, hl_desc = _resolve_candidate_overrides(lower_cand, hl_base, spot_fields)
    assert hl_over == {"entry_amount": 262.5, "dca_amount": 75.0}
    assert "0.75x live" in hl_desc

    # 2. Trading 212 StratParams resolves all 18 standard candidates cleanly and rejects adaptive k*sigma.
    t212_strat = load_venue_module("212trading", "strategy.py", "test_w2_t212_strat")
    t212_base = t212_strat.StratParams.from_dict({
        "currency": "EUR", "yahoo_sym": "NVDA", "entry_amount": 12.032, "dca_amount": 12.032,
        "entry_discount_pct": 0.2, "dca_drop_pct": 3.5, "check_minutes": 5.0,
        "takeprofit_pct": 8.0, "max_budget": 80.0, "max_dca_buys": 5,
        "order_ttl_min": 10.0, "stop_loss_pct": 20.0, "fx_fee_pct": 0.15,
        "trail_pct": 8.0, "trail_min_profit_pct": 5.0, "tp_mdhtp_enabled": True,
    })
    t212_fields = {f.name for f in dataclasses.fields(t212_base.__class__)}
    resolved_t212 = {
        c.name: _resolve_candidate_overrides(c, t212_base, t212_fields)
        for c in default_candidates()
    }
    assert all(v is not None for v in resolved_t212.values())
    assert len(resolved_t212) == 18
    assert resolved_t212["lower_sizing"][0] == {"entry_amount": 9.024, "dca_amount": 9.024}
    assert resolved_t212["classic_tp"][0] == {
        "tp_mdhtp_enabled": False, "trail_pct": 0.0, "tp_ladder": [],
    }
    assert resolved_t212["trail_2"][0] == {
        "tp_mdhtp_enabled": False, "enable_takeprofit": False, "trail_pct": 2.0,
    }
    assert resolved_t212["confirm_dca_1_5_trail_2"][0] == {
        "dca_drop_pct": 1.5, "tp_mdhtp_enabled": False, "enable_takeprofit": False, "trail_pct": 2.0,
    }
    adaptive_cand = next(c for c in hype_240_candidates() if c.name == "A_adaptive_trail")
    assert _resolve_candidate_overrides(adaptive_cand, t212_base, t212_fields) is None

    # 3. Hyperliquid and Binance baseline runners fetch full decision-grade lookbacks by default.
    with patch.object(hwf, "fetch_ohlc_candles", return_value=[]) as mock_hl_fetch:
        assert hwf.main(["--intervals", "240", "--output-dir", str(tmp_path / "hl")]) == 0
        assert mock_hl_fetch.call_args.kwargs.get("start_ms") == 0
    with patch.object(bwf, "fetch_ohlc_candles", return_value=[]) as mock_bn_fetch:
        assert bwf.main(["--intervals", "240", "--symbol", "BTCUSDC", "--output-dir", str(tmp_path / "bn")]) == 0
        assert mock_bn_fetch.call_args.kwargs.get("lookback_days") == 630
    with pytest.raises(SystemExit):
        hwf.main(["--lookback-days", "0"])
    with pytest.raises(SystemExit):
        bwf.main(["--lookback-days", "0"])

    # 4. Sweep combos, Kraken trail compare, and continuous grids explicitly disable tp_mdhtp_enabled.
    sweep_grid = define_candidate_grid()
    for combo_name in ("25_COMBO_smart_flow", "27_COMBO_high_calmar_runner", "29_COMBO_ultimate_balanced"):
        assert sweep_grid[combo_name]["tp_mdhtp_enabled"] is False
    assert k_grid.CONFIGS["abl_trail_fixed"][0].tp_mdhtp_enabled is False
    assert k_grid2.FINE["gateoff_trail5"][0].tp_mdhtp_enabled is False
    assert t_grid.VARIANTS["trail8"][0]["STRAT_TP_MDHTP_ENABLED"] == "false"
    assert t_grid.VARIANTS["trail_off"][0]["STRAT_TP_MDHTP_ENABLED"] == "false"

    bars = [
        [100.0, 106.0 if i % 4 == 2 else 101.0, 98.5, 105.5 if i % 4 == 2 else 100.0, 1700000000 + i * 14400]
        for i in range(40)
    ]
    with patch.object(k_btt.rp, "run_replay", wraps=k_btt.rp.run_replay) as spy_replay:
        k_btt.simulate_trail(bars, {**k_btt.base_params(), "interval": 240, "trail": 3.0})
        trail_params = spy_replay.call_args.args[1]
        assert trail_params.tp_trail_adaptive is False
        assert trail_params.tp_mdhtp_enabled is False
        assert trail_params.tp_trail_pct == 3.0

        k_btt.simulate_fixed_tp(bars, {**k_btt.base_params(), "interval": 240})
        fix_params = spy_replay.call_args.args[1]
        assert fix_params.tp_trend_hold is False
        assert fix_params.tp_mdhtp_enabled is False


def test_declarative_venue_replay_and_benchmark_registries():
    assert set(VENUE_REPLAY_DIRS) == {"hyperliquid", "t212"}
    assert set(VENUE_EVALUATOR_BUILDERS) == {"kraken", "binance", "hyperliquid", "t212"}
    assert set(VENUE_BENCHMARK_SCENARIOS) == {"kraken", "binance", "hyperliquid", "t212"}

    hl_rp = load_venue_replay_module("hyperliquid", "test_reg_hl_rp")
    t212_rp = load_venue_replay_module("trading212", "test_reg_t212_rp")
    assert hasattr(hl_rp, "make_hyperliquid_replay_adapter")
    assert hasattr(t212_rp, "make_t212_replay_adapter")
    with pytest.raises(ValueError, match="Unsupported replay venue"):
        load_venue_replay_module("unknown_venue")

    assert default_scenarios("trading212") == VENUE_BENCHMARK_SCENARIOS["t212"]
    assert default_scenarios("unknown_venue") == VENUE_BENCHMARK_SCENARIOS["kraken"]
    _, t212_fee, _ = _build_venue_evaluator("trading212", {"strategy_params": {"fx_fee_pct": 0.15}})
    assert t212_fee == pytest.approx(0.15)

