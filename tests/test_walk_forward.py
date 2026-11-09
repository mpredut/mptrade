import unittest
from pathlib import Path
from unittest.mock import patch

from offline.backtests.evaluation import assess_decision_evidence, evaluate_walk_forward
from offline.backtests.walk_forward import summarize_test_windows, walk_forward_splits
from offline.runners.walk_forward import (
    canonical_venue,
    dispatch,
    main,
    run_baseline,
    run_compare,
    SUPPORTED_VENUES,
    SUPPORTED_MODES,
    VENUE_RUNNER_MODULES,
)
from offline.runners.walk_forward_compare import (
    _detect_venue,
    _markdown,
    default_candidates,
    DEFAULT_FEE_STRESS,
)
import offline.runners.kraken_walk_forward_compare as kraken_compare


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
        from offline.backtests.evaluation import evaluate_segment

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


class WalkForwardCompareTest(unittest.TestCase):
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
        names = {c.name for c in candidates}
        self.assertIn("live", names)
        self.assertIn("classic_tp", names)
        self.assertIn("dca_drop_1_5", names)

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
        from offline.runners import walk_forward_compare

        self.assertIs(kraken_compare.main, walk_forward_compare.main)

    def test_load_report_supports_both_schemas(self):
        from offline.runners.walk_forward_compare import _load_report
        import tempfile
        import json

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 1, "intervals": {"240": {}}}, f)
            f.flush()
            rep = _load_report(Path(f.name))
            self.assertEqual(rep["schema_version"], 1)

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 1, "dataset": {}, "windows": {}}, f)
            f.flush()
            rep = _load_report(Path(f.name))
            self.assertEqual(rep["schema_version"], 1)

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 1, "dataset": {}, "window_sizes": {}}, f)
            f.flush()
            rep = _load_report(Path(f.name))
            self.assertEqual(rep["schema_version"], 1)

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 2}, f)
            f.flush()
            with self.assertRaises(ValueError):
                _load_report(Path(f.name))

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
        from offline.backtests.comparison import format_number
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
        from offline.runners.walk_forward_compare import _build_venue_evaluator
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
        from offline.backtests.tradeall import BacktestBroker, run_backtest
        from offline.runners.t212_walk_forward_baseline import periods_per_year
        from offline.runners.kraken_financial_benchmark import format_number as kfb_fmt
        from offline.runners.calibrate_execution_audit import format_number as cea_fmt
        import tempfile

        for fn in (kfb_fmt, cea_fmt):
            self.assertEqual(fn(None), "n/a")
            self.assertEqual(fn(True), "n/a")
            self.assertEqual(fn(float("nan")), "n/a")
            self.assertEqual(fn(float("inf")), "n/a")

        with self.assertRaises(ValueError):
            periods_per_year(0)
        with self.assertRaises(ValueError):
            periods_per_year(False)

        with tempfile.TemporaryDirectory() as tmp:
            broker = BacktestBroker(tmp, lambda: 1000.0)
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
