import unittest

from offline.backtests.evaluation import assess_decision_evidence, evaluate_walk_forward
from offline.backtests.walk_forward import summarize_test_windows, walk_forward_splits


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


if __name__ == "__main__":
    unittest.main()
