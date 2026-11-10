import copy
import json
from pathlib import Path
import unittest

from offline.backtests.financial_benchmark import (
    aggregate_financial_windows,
    default_scenarios,
)
from offline.backtests.promotion import (
    PromotionThresholds,
    evaluate_dual_promotion,
    evaluate_promotion,
    evaluate_risk_adjusted_promotion,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "offline" / "research" / "hype_dataset"
BASELINE = DATASET_DIR / "financial_baseline_v1.json"


def _window(key, strategy_return, buy_hold, drawdown, net_pnl):
    return {
        "key": key,
        "bars": 90,
        "return_pct": strategy_return,
        "max_drawdown_pct": drawdown,
        "buy_hold_return_pct": buy_hold,
        "cycles": 1,
        "fills": 3,
        "metrics": {
            "net_pnl": net_pnl,
            "max_drawdown_abs": drawdown * 10,
            "sortino": 1.5,
            "calmar": 2.0,
            "profit_factor": 1.8,
            "expectancy": net_pnl,
            "cvar_95_pct": -1.2,
            "exposure_pct": 50.0,
            "turnover_pct": 80.0,
            "trade_count": 1,
            "max_underwater_periods": 4,
        },
    }


class FinancialBenchmarkTest(unittest.TestCase):
    def test_default_scenarios_are_explicit_and_stress_is_more_adverse(self):
        central, stress = default_scenarios()
        self.assertEqual((central.name, stress.name), ("central", "stress"))
        self.assertGreater(stress.fees.limit_fee_pct, central.fees.limit_fee_pct)
        self.assertGreater(stress.fees.market_fee_pct, central.fees.market_fee_pct)
        self.assertGreater(stress.execution.spread_bps, central.execution.spread_bps)
        self.assertLess(stress.execution.partial_fill_ratio,
                        central.execution.partial_fill_ratio)
        self.assertFalse(central.calibrated)

    def test_aggregate_reports_usd_risk_and_bull_bear_sideways_separately(self):
        windows = [
            _window("bull", 2.0, 5.0, 1.0, 78.0),
            _window("bear", -3.0, -8.0, 4.0, -117.0),
            _window("sideways", 1.0, 1.0, 2.0, 39.0),
        ]

        result = aggregate_financial_windows(windows, initial_capital=3900.0)

        self.assertEqual(result["window_count"], 3)
        self.assertEqual(result["sum_reset_net_pnl_usd"], 0.0)
        self.assertEqual(result["mean_net_pnl_usd_per_window"], 0.0)
        self.assertEqual(result["mean_buy_hold_return_pct"], -2.0 / 3.0)
        self.assertEqual(result["worst_return_pct"], -3.0)
        self.assertEqual(result["worst_max_drawdown_pct"], 4.0)
        self.assertEqual(result["regimes"]["bull"]["windows"], 1)
        self.assertEqual(result["regimes"]["bear"]["windows"], 1)
        self.assertEqual(result["regimes"]["sideways"]["windows"], 1)
        self.assertEqual(result["total_test_bars"], 270)

    def test_missing_financial_fields_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aggregate_financial_windows([{"key": "x"}], initial_capital=3900.0)

    def test_multi_venue_default_scenarios(self):
        for venue in ("binance", "hyperliquid", "t212", "kraken"):
            central, stress = default_scenarios(venue)
            self.assertEqual((central.name, stress.name), ("central", "stress"))
            self.assertGreater(stress.fees.market_fee_pct, central.fees.limit_fee_pct)
            self.assertGreater(stress.execution.spread_bps, central.execution.spread_bps)
            self.assertLessEqual(stress.execution.partial_fill_ratio, central.execution.partial_fill_ratio)

    def test_aggregate_handles_none_buy_hold_gracefully(self):
        windows = [
            _window("w1", 2.0, None, 1.0, 78.0),
            _window("w2", -1.0, 4.0, 2.0, -39.0),
        ]
        result = aggregate_financial_windows(windows, initial_capital=3900.0)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["mean_buy_hold_return_pct"], 4.0)
        self.assertEqual(result["regimes"]["sideways"]["windows"], 1)
        self.assertEqual(result["regimes"]["bull"]["windows"], 1)


class FinancialBaselineArtifactTest(unittest.TestCase):
    def test_versioned_baseline_has_reproducible_contract(self):
        with BASELINE.open(encoding="utf-8") as handle:
            report = json.load(handle)
        with (DATASET_DIR / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["candidate_name"], "base_v2_live")
        self.assertEqual(
            report["dataset"]["sha256"],
            manifest["datasets"]["240"]["sha256"],
        )
        self.assertEqual(report["dataset"]["bars"], 3772)
        self.assertEqual(report["walk_forward"]["parameter_selection"],
                         "none; fixed profile")
        self.assertFalse(report["walk_forward"]["shuffle"])
        self.assertFalse(report["code"]["worktree_dirty"])
        self.assertRegex(report["code"]["commit"], r"^[0-9a-f]{40}$")

        self.assertEqual(set(report["scenarios"]), {"central", "stress"})
        for name, scenario in report["scenarios"].items():
            with self.subTest(scenario=name):
                self.assertFalse(
                    scenario["assumptions"]["calibrated_from_real_fills"]
                )
                self.assertEqual(
                    scenario["assumptions"]["execution"]["intrabar_policy"],
                    "worst_case",
                )
                self.assertEqual(len(scenario["windows"]), 31)
                self.assertEqual(scenario["aggregate"]["window_count"], 31)
                self.assertEqual(scenario["aggregate"]["total_test_bars"], 2790)


def _report(
    mean=1.0, worst=-2.0, drawdown=4.0, returns=(1.0,) * 20, *,
    drawdowns=None, calmar=1.0, sortino=2.0, cvar=-2.0, exposure=50.0,
):
    drawdowns = drawdowns or (drawdown,) * len(returns)
    windows = [
        {
            "key": f"240m/fold-{index:02d}",
            "return_pct": value,
            "max_drawdown_pct": drawdowns[index - 1],
        }
        for index, value in enumerate(returns, start=1)
    ]
    scenario = {
        "aggregate": {
            "mean_return_pct": mean,
            "worst_return_pct": worst,
            "worst_max_drawdown_pct": drawdown,
            "median_calmar": calmar,
            "median_sortino": sortino,
            "worst_cvar_95_pct": cvar,
            "mean_exposure_pct": exposure,
        },
        "windows": windows,
    }
    return {
        "dataset": {"sha256": "abc", "bars": 3772},
        "walk_forward": {
            "train": 720, "validation": 180, "test": 90,
            "step": 90, "warmup": 40,
        },
        "scenarios": {"central": copy.deepcopy(scenario),
                      "stress": copy.deepcopy(scenario)},
    }


class PromotionGateTest(unittest.TestCase):
    def test_candidate_must_improve_mean_without_worsening_tail_or_drawdown(self):
        baseline = _report(returns=(0.0,) * 20)
        candidate = _report(
            mean=1.2, worst=-1.9, drawdown=3.8,
            returns=(0.2,) * 16 + (-0.1,) * 4,
        )

        result = evaluate_promotion(baseline, candidate)

        self.assertTrue(result["promote"])
        self.assertTrue(result["scenarios"]["central"]["passed"])
        self.assertEqual(
            result["scenarios"]["stress"]["wins_ties_losses"],
            {"wins": 16, "ties": 0, "losses": 4},
        )
        self.assertLessEqual(
            result["scenarios"]["stress"]["one_sided_sign_pvalue"], 0.10,
        )

    def test_sparse_effect_fails_even_when_all_active_windows_win(self):
        baseline = _report(returns=(0.0,) * 20)
        candidate = _report(
            mean=1.2, worst=-1.9, drawdown=3.8,
            returns=(0.2,) * 6 + (0.0,) * 14,
        )

        result = evaluate_promotion(baseline, candidate)
        central = result["scenarios"]["central"]

        self.assertFalse(result["promote"])
        self.assertEqual(central["active_windows"], 6)
        self.assertFalse(central["checks"]["enough_active_windows"])
        self.assertTrue(central["checks"]["sign_test_support"])

    def test_one_failed_stress_scenario_blocks_promotion(self):
        baseline = _report(returns=(0.0,) * 20)
        candidate = _report(
            mean=1.2, worst=-1.9, drawdown=3.8, returns=(0.2,) * 20,
        )
        candidate["scenarios"]["stress"]["aggregate"]["worst_max_drawdown_pct"] = 4.5

        result = evaluate_promotion(baseline, candidate)

        self.assertFalse(result["promote"])
        self.assertFalse(result["scenarios"]["stress"]["checks"]["drawdown_preserved"])

    def test_same_strategy_is_not_a_promotion(self):
        baseline = _report()
        result = evaluate_promotion(baseline, copy.deepcopy(baseline))
        self.assertFalse(result["promote"])
        self.assertFalse(
            result["scenarios"]["central"]["checks"]["material_mean_improvement"]
        )

    def test_different_dataset_is_rejected(self):
        baseline = _report()
        candidate = _report()
        candidate["dataset"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "dataset"):
            evaluate_promotion(baseline, candidate, PromotionThresholds())


class RiskAdjustedPromotionGateTest(unittest.TestCase):
    def test_material_robust_risk_reduction_passes_defensive_path(self):
        baseline = _report(returns=(0.0,) * 20, drawdowns=(4.0,) * 20)
        candidate = _report(
            mean=0.90, worst=-1.80, drawdown=2.50,
            returns=(0.0,) * 20,
            drawdowns=(3.0,) * 16 + (4.1,) * 4,
            calmar=1.30, sortino=2.10, cvar=-1.50, exposure=45.0,
        )

        result = evaluate_risk_adjusted_promotion(baseline, candidate)

        self.assertTrue(result["promote"])
        self.assertTrue(result["scenarios"]["central"]["passed"])
        self.assertEqual(
            result["scenarios"]["central"]["drawdown_wins_ties_losses"],
            {"wins": 16, "ties": 0, "losses": 4},
        )

    def test_aggregate_return_over_worst_dd_is_not_accepted_as_calmar(self):
        baseline = _report(calmar=1.0)
        candidate = _report(
            mean=0.91, drawdown=2.5, calmar=0.80,
            sortino=1.8, cvar=-1.5, exposure=40.0,
            drawdowns=(2.5,) * 20,
        )

        result = evaluate_risk_adjusted_promotion(baseline, candidate)

        self.assertFalse(result["promote"])
        self.assertFalse(
            result["scenarios"]["central"]["checks"][
                "material_calmar_improvement"
            ]
        )

    def test_sparse_drawdown_effect_fails_defensive_path(self):
        baseline = _report(drawdowns=(4.0,) * 20)
        candidate = _report(
            mean=0.90, drawdown=2.5, calmar=1.3, sortino=2.1,
            cvar=-1.5, exposure=45.0,
            drawdowns=(3.0,) * 6 + (4.0,) * 14,
        )

        result = evaluate_risk_adjusted_promotion(baseline, candidate)

        self.assertFalse(result["promote"])
        self.assertFalse(
            result["scenarios"]["central"]["checks"]["enough_active_windows"]
        )

    def test_dual_gate_preserves_promotion_objective(self):
        baseline = _report(returns=(0.0,) * 20, drawdowns=(4.0,) * 20)
        candidate = _report(
            mean=0.90, worst=-1.8, drawdown=2.5,
            returns=(0.0,) * 20,
            drawdowns=(3.0,) * 16 + (4.1,) * 4,
            calmar=1.3, sortino=2.1, cvar=-1.5, exposure=45.0,
        )

        result = evaluate_dual_promotion(baseline, candidate)

        self.assertTrue(result["promote"])
        self.assertEqual(result["promotion_paths"], ["DEFENSIVE"])

    def test_different_execution_assumptions_are_rejected(self):
        baseline = _report()
        candidate = _report()
        baseline["scenarios"]["central"]["assumptions"] = {"spread_bps": 10}
        candidate["scenarios"]["central"]["assumptions"] = {"spread_bps": 0}

        with self.assertRaisesRegex(ValueError, "execution assumptions"):
            evaluate_dual_promotion(baseline, candidate)

    def test_non_finite_metrics_never_pass_promotion(self):
        baseline = _report(returns=(0.0,) * 20)
        candidate = _report(
            mean=float("inf"), worst=float("inf"), drawdown=float("-inf"),
            returns=(float("inf"),) * 20,
        )
        res = evaluate_promotion(baseline, candidate)
        self.assertFalse(res["promote"])

        candidate_none = _report(
            mean=None, worst=None, drawdown=None,
            returns=(None,) * 20,
            drawdowns=(None,) * 20,
        )
        dual_none = evaluate_dual_promotion(baseline, candidate_none)
        self.assertFalse(dual_none["promote"])
        self.assertEqual(dual_none["promotion_paths"], [])

    def test_spot_profile_review_rejects_invalid_cash_and_snapshot_handles_non_dict_json(self):
        import tempfile
        from unittest.mock import patch
        from offline.runners.spot_profile_review import review
        from verify_tools import portfolio_snapshot as ps

        for bad_cash in (0, -100, float("nan"), float("inf"), True):
            with self.assertRaisesRegex(ValueError, "cash must be a finite positive number"):
                review("HEAD:kraken/config.env", "HEAD:kraken/config.env", bad_cash)

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / ".state_HYPEUSD.json"
            state_file.write_text("[]", encoding="utf-8")
            alert_file = Path(tmp) / "alerts.json"
            alert_file.write_text("[]", encoding="utf-8")
            with patch.object(ps, "KRAKEN_DIR", tmp), patch.object(ps, "ALERT_STATE", str(alert_file)):
                self.assertIsNone(ps._load_state("HYPEUSD"))
                self.assertEqual(ps.run_alerts([]), [])

    def test_median_profit_factor_includes_zero_loss_winning_folds_and_negative_base_defensive_gate(self):
        win_fold_1 = _window("w1", 3.0, 2.0, 0.5, 117.0)
        win_fold_1["metrics"]["profit_factor"] = None
        win_fold_1["metrics"]["gross_profit"] = 117.0
        win_fold_1["metrics"]["gross_loss"] = 0.0
        win_fold_2 = _window("w2", 2.5, 1.0, 0.4, 97.5)
        win_fold_2["metrics"]["profit_factor"] = None
        win_fold_2["metrics"]["gross_profit"] = 97.5
        win_fold_2["metrics"]["gross_loss"] = 0.0
        loss_fold = _window("w3", -1.0, -5.0, 2.0, -39.0)
        loss_fold["metrics"]["profit_factor"] = 0.2
        loss_fold["metrics"]["gross_profit"] = 9.75
        loss_fold["metrics"]["gross_loss"] = 48.75

        agg = aggregate_financial_windows([win_fold_1, win_fold_2, loss_fold], initial_capital=3900.0)
        self.assertGreater(agg["median_profit_factor"], 1.0)
        self.assertAlmostEqual(agg["aggregate_profit_factor"], (117.0 + 97.5 + 9.75) / 48.75)

        baseline = _report(mean=-0.5, worst=-4.0, drawdown=6.0, returns=(-0.5,) * 20, drawdowns=(6.0,) * 20, calmar=-0.3, sortino=-0.4, cvar=-3.0, exposure=60.0)
        candidate = _report(mean=0.8, worst=-1.5, drawdown=3.0, returns=(0.8,) * 20, drawdowns=(3.0,) * 20, calmar=1.1, sortino=1.4, cvar=-1.2, exposure=45.0)
        res = evaluate_risk_adjusted_promotion(baseline, candidate)
        self.assertTrue(res["promote"])


if __name__ == "__main__":
    unittest.main()
