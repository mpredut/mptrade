"""Conservative promotion gate for two financial benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PromotionThresholds:
    min_mean_improvement_pp: float = 0.10
    max_worst_return_degradation_pp: float = 0.25
    max_drawdown_increase_pp: float = 0.25
    min_windows: int = 20
    min_active_windows: int = 10
    max_one_sided_sign_pvalue: float = 0.10
    require_more_wins_than_losses: bool = True


@dataclass(frozen=True)
class RiskAdjustedThresholds:
    """Thresholds for a defensive candidate, without confusing it with a return one."""

    min_median_calmar_improvement_ratio: float = 0.15
    max_median_sortino_degradation_ratio: float = 0.05
    max_mean_return_degradation_pp: float = 0.15
    max_worst_return_degradation_pp: float = 0.25
    min_worst_drawdown_improvement_pp: float = 1.0
    min_worst_cvar_improvement_pp: float = 0.25
    max_mean_exposure_increase_pp: float = 0.0
    min_windows: int = 20
    min_active_windows: int = 10
    max_one_sided_drawdown_sign_pvalue: float = 0.10
    require_more_drawdown_wins_than_losses: bool = True


def _identity(report: dict) -> tuple:
    if not isinstance(report, dict):
        return (None,) * 9
    raw_ds = report.get("dataset")
    dataset = raw_ds if isinstance(raw_ds, dict) else {}
    raw_wf = report.get("walk_forward")
    walk = raw_wf if isinstance(raw_wf, dict) else {}
    return (
        dataset.get("sha256"), dataset.get("bars"),
        report.get("initial_capital_usd"), walk.get("interval_minutes"),
        walk.get("train"), walk.get("validation"), walk.get("test"),
        walk.get("step"), walk.get("warmup"),
    )


def _one_sided_sign_pvalue(wins: int, losses: int) -> float:
    """P(X >= wins), X~Binomial(wins+losses, 0.5), ignoring ties."""
    w = max(0, int(wins)) if not isinstance(wins, bool) else 0
    l = max(0, int(losses)) if not isinstance(losses, bool) else 0
    active = w + l
    if active == 0:
        return 1.0
    total_comb = sum(math.comb(active, value) for value in range(w, active + 1))
    return float(total_comb / (1 << active))


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _diff(left, right) -> float:
    lv = _finite(left)
    rv = _finite(right)
    if lv is None or rv is None:
        return float("nan")
    return lv - rv


def _relative_change(baseline, candidate) -> float | None:
    """Relative change only for a positive, comparable base."""
    base = _finite(baseline)
    cand = _finite(candidate)
    if base is None or cand is None or base <= 0:
        return None
    return (cand - base) / base


def evaluate_promotion(
    baseline: dict,
    candidate: dict,
    thresholds: PromotionThresholds | None = None,
) -> dict:
    """The candidate passes only if it is robust in ALL shared scenarios."""
    limits = thresholds or PromotionThresholds()
    if _identity(baseline) != _identity(candidate):
        raise ValueError("baseline and candidate use different datasets or windows")

    base_scenarios = baseline.get("scenarios") or {}
    candidate_scenarios = candidate.get("scenarios") or {}
    if not base_scenarios or set(base_scenarios) != set(candidate_scenarios):
        raise ValueError("baseline and candidate must have the same scenarios")

    scenario_results = {}
    for name in sorted(base_scenarios):
        base = base_scenarios[name]
        cand = candidate_scenarios[name]
        if base.get("assumptions") != cand.get("assumptions"):
            raise ValueError(f"different execution assumptions in scenario {name}")
        base_win_list = base.get("windows") if isinstance(base.get("windows"), list) else []
        cand_win_list = cand.get("windows") if isinstance(cand.get("windows"), list) else []
        base_windows = {
            window["key"]: window
            for window in base_win_list
            if isinstance(window, dict) and "key" in window
        }
        cand_windows = {
            window["key"]: window
            for window in cand_win_list
            if isinstance(window, dict) and "key" in window
        }
        if (
            len(base_windows) != len(base_win_list)
            or len(cand_windows) != len(cand_win_list)
            or set(base_windows) != set(cand_windows)
        ):
            raise ValueError(f"different windows in scenario {name}")
        deltas = [
            _diff(cand_windows[key].get("return_pct"), base_windows[key].get("return_pct"))
            for key in sorted(base_windows)
        ]
        all_deltas_finite = all(
            _finite(cand_windows[key].get("return_pct")) is not None
            and _finite(base_windows[key].get("return_pct")) is not None
            and math.isfinite(d)
            for key, d in zip(sorted(base_windows), deltas)
        )
        tolerance = 1e-10
        wins = sum(math.isfinite(delta) and delta > tolerance for delta in deltas)
        ties = sum(math.isfinite(delta) and abs(delta) <= tolerance for delta in deltas)
        losses = sum(math.isfinite(delta) and delta < -tolerance for delta in deltas)
        active_windows = wins + losses
        sign_pvalue = _one_sided_sign_pvalue(wins, losses)
        base_aggregate = base.get("aggregate") if isinstance(base.get("aggregate"), dict) else {}
        candidate_aggregate = cand.get("aggregate") if isinstance(cand.get("aggregate"), dict) else {}
        mean_delta = _diff(
            candidate_aggregate.get("mean_return_pct"),
            base_aggregate.get("mean_return_pct"),
        )
        worst_delta = _diff(
            candidate_aggregate.get("worst_return_pct"),
            base_aggregate.get("worst_return_pct"),
        )
        drawdown_delta = _diff(
            candidate_aggregate.get("worst_max_drawdown_pct"),
            base_aggregate.get("worst_max_drawdown_pct"),
        )
        checks = {
            "enough_windows": all_deltas_finite and len(deltas) >= limits.min_windows,
            "enough_active_windows": active_windows >= limits.min_active_windows,
            "material_mean_improvement": (
                math.isfinite(mean_delta)
                and mean_delta >= limits.min_mean_improvement_pp
            ),
            "worst_return_preserved": (
                math.isfinite(worst_delta)
                and worst_delta >= -limits.max_worst_return_degradation_pp
            ),
            "drawdown_preserved": (
                math.isfinite(drawdown_delta)
                and drawdown_delta <= limits.max_drawdown_increase_pp
            ),
            "pairwise_robust": (
                wins > losses if limits.require_more_wins_than_losses else True
            ),
            "sign_test_support": (
                sign_pvalue <= limits.max_one_sided_sign_pvalue
            ),
        }
        scenario_results[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "mean_return_delta_pp": mean_delta,
            "worst_return_delta_pp": worst_delta,
            "worst_drawdown_delta_pp": drawdown_delta,
            "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
            "active_windows": active_windows,
            "active_window_rate_pct": (
                active_windows / len(deltas) * 100.0 if deltas else 0.0
            ),
            "one_sided_sign_pvalue": sign_pvalue,
        }

    return {
        "promote": all(item["passed"] for item in scenario_results.values()),
        "objective": "RETURN",
        "thresholds": asdict(limits),
        "scenarios": scenario_results,
    }


def evaluate_risk_adjusted_promotion(
    baseline: dict,
    candidate: dict,
    thresholds: RiskAdjustedThresholds | None = None,
) -> dict:
    """Evaluate a defensive track on the real per-fold metrics.

    We do not use ``mean_return / worst_drawdown`` as Calmar: the numerator and
    denominator could come from different folds. The benchmark computes Calmar on
    each fold's curve, and the gate compares the medians of those values.
    """
    limits = thresholds or RiskAdjustedThresholds()
    if _identity(baseline) != _identity(candidate):
        raise ValueError("baseline and candidate use different datasets or windows")

    base_scenarios = baseline.get("scenarios") or {}
    candidate_scenarios = candidate.get("scenarios") or {}
    if not base_scenarios or set(base_scenarios) != set(candidate_scenarios):
        raise ValueError("baseline and candidate must have the same scenarios")

    scenario_results = {}
    tolerance = 1e-10
    for name in sorted(base_scenarios):
        base = base_scenarios[name]
        cand = candidate_scenarios[name]
        if base.get("assumptions") != cand.get("assumptions"):
            raise ValueError(f"different execution assumptions in scenario {name}")
        base_win_list = base.get("windows") if isinstance(base.get("windows"), list) else []
        cand_win_list = cand.get("windows") if isinstance(cand.get("windows"), list) else []
        base_windows = {
            window["key"]: window
            for window in base_win_list
            if isinstance(window, dict) and "key" in window
        }
        cand_windows = {
            window["key"]: window
            for window in cand_win_list
            if isinstance(window, dict) and "key" in window
        }
        if (
            len(base_windows) != len(base_win_list)
            or len(cand_windows) != len(cand_win_list)
            or set(base_windows) != set(cand_windows)
        ):
            raise ValueError(f"different windows in scenario {name}")

        drawdown_improvements = [
            _diff(base_windows[key].get("max_drawdown_pct"), cand_windows[key].get("max_drawdown_pct"))
            for key in sorted(base_windows)
        ]
        all_dd_finite = all(
            _finite(base_windows[key].get("max_drawdown_pct")) is not None
            and _finite(cand_windows[key].get("max_drawdown_pct")) is not None
            and math.isfinite(d)
            for key, d in zip(sorted(base_windows), drawdown_improvements)
        )
        wins = sum(math.isfinite(delta) and delta > tolerance for delta in drawdown_improvements)
        ties = sum(math.isfinite(delta) and abs(delta) <= tolerance for delta in drawdown_improvements)
        losses = sum(math.isfinite(delta) and delta < -tolerance for delta in drawdown_improvements)
        active_windows = wins + losses
        sign_pvalue = _one_sided_sign_pvalue(wins, losses)

        base_aggregate = base.get("aggregate") if isinstance(base.get("aggregate"), dict) else {}
        candidate_aggregate = cand.get("aggregate") if isinstance(cand.get("aggregate"), dict) else {}
        cand_mean_return = _finite(candidate_aggregate.get("mean_return_pct"))
        mean_delta = _diff(
            candidate_aggregate.get("mean_return_pct"),
            base_aggregate.get("mean_return_pct"),
        )
        worst_return_delta = _diff(
            candidate_aggregate.get("worst_return_pct"),
            base_aggregate.get("worst_return_pct"),
        )
        worst_drawdown_improvement = _diff(
            base_aggregate.get("worst_max_drawdown_pct"),
            candidate_aggregate.get("worst_max_drawdown_pct"),
        )
        base_cvar = _finite(base_aggregate.get("worst_cvar_95_pct"))
        candidate_cvar = _finite(candidate_aggregate.get("worst_cvar_95_pct"))
        cvar_improvement = (
            candidate_cvar - base_cvar
            if base_cvar is not None and candidate_cvar is not None else None
        )
        base_exposure = _finite(base_aggregate.get("mean_exposure_pct"))
        candidate_exposure = _finite(candidate_aggregate.get("mean_exposure_pct"))
        exposure_delta = (
            candidate_exposure - base_exposure
            if base_exposure is not None and candidate_exposure is not None else None
        )
        calmar_improvement = _relative_change(
            base_aggregate.get("median_calmar"),
            candidate_aggregate.get("median_calmar"),
        )
        sortino_change = _relative_change(
            base_aggregate.get("median_sortino"),
            candidate_aggregate.get("median_sortino"),
        )

        checks = {
            "enough_windows": all_dd_finite and len(drawdown_improvements) >= limits.min_windows,
            "enough_active_windows": active_windows >= limits.min_active_windows,
            "positive_return": cand_mean_return is not None and cand_mean_return > 0,
            "return_noninferior": (
                math.isfinite(mean_delta)
                and mean_delta >= -limits.max_mean_return_degradation_pp
            ),
            "worst_return_preserved": (
                math.isfinite(worst_return_delta)
                and worst_return_delta >= -limits.max_worst_return_degradation_pp
            ),
            "calmar_available": calmar_improvement is not None,
            "material_calmar_improvement": (
                calmar_improvement is not None
                and calmar_improvement >= limits.min_median_calmar_improvement_ratio
            ),
            "sortino_preserved": (
                sortino_change is not None
                and sortino_change >= -limits.max_median_sortino_degradation_ratio
            ),
            "material_drawdown_improvement": (
                math.isfinite(worst_drawdown_improvement)
                and worst_drawdown_improvement
                >= limits.min_worst_drawdown_improvement_pp
            ),
            "material_cvar_improvement": (
                cvar_improvement is not None
                and cvar_improvement >= limits.min_worst_cvar_improvement_pp
            ),
            "exposure_not_increased": (
                exposure_delta is not None
                and exposure_delta <= limits.max_mean_exposure_increase_pp + tolerance
            ),
            "drawdown_pairwise_robust": (
                wins > losses
                if limits.require_more_drawdown_wins_than_losses else True
            ),
            "drawdown_sign_test_support": (
                sign_pvalue <= limits.max_one_sided_drawdown_sign_pvalue
            ),
        }
        scenario_results[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "mean_return_delta_pp": mean_delta,
            "worst_return_delta_pp": worst_return_delta,
            "worst_drawdown_improvement_pp": worst_drawdown_improvement,
            "worst_cvar_improvement_pp": cvar_improvement,
            "mean_exposure_delta_pp": exposure_delta,
            "median_calmar_improvement_ratio": calmar_improvement,
            "median_sortino_change_ratio": sortino_change,
            "drawdown_wins_ties_losses": {
                "wins": wins, "ties": ties, "losses": losses,
            },
            "active_windows": active_windows,
            "active_window_rate_pct": (
                active_windows / len(drawdown_improvements) * 100.0
                if drawdown_improvements else 0.0
            ),
            "one_sided_drawdown_sign_pvalue": sign_pvalue,
        }

    return {
        "promote": all(item["passed"] for item in scenario_results.values()),
        "objective": "DEFENSIVE",
        "thresholds": asdict(limits),
        "scenarios": scenario_results,
    }


def evaluate_dual_promotion(
    baseline: dict,
    candidate: dict,
    return_thresholds: PromotionThresholds | None = None,
    risk_thresholds: RiskAdjustedThresholds | None = None,
) -> dict:
    """The candidate is eligible through RETURN or DEFENSIVE, with the label kept."""
    return_gate = evaluate_promotion(baseline, candidate, return_thresholds)
    defensive_gate = evaluate_risk_adjusted_promotion(
        baseline, candidate, risk_thresholds,
    )
    paths = []
    if return_gate["promote"]:
        paths.append("RETURN")
    if defensive_gate["promote"]:
        paths.append("DEFENSIVE")
    return {
        "promote": bool(paths),
        "promotion_paths": paths,
        "return_gate": return_gate,
        "defensive_gate": defensive_gate,
    }
