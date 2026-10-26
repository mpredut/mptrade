import copy
import unittest

from offline.backtests.promotion import (
    PromotionThresholds,
    evaluate_dual_promotion,
    evaluate_promotion,
    evaluate_risk_adjusted_promotion,
)


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

    def test_spot_profile_review_rejects_invalid_cash_and_snapshot_handles_non_dict_json(self):
        import tempfile
        from pathlib import Path
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
                self.assertEqual(ps._alert_state(), {})


if __name__ == "__main__":
    unittest.main()
