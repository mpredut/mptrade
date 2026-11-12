"""Conservative promotion gate for two financial benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from botcore import _parse_bool_flag, exact_pos_int, safe_float, safe_int


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
    return tuple(
        ("__bool__", v) if isinstance(v, bool) else v
        for v in (
            dataset.get("sha256"), dataset.get("bars"),
            report.get("initial_capital_usd"), walk.get("interval_minutes"),
            walk.get("train"), walk.get("validation"), walk.get("test"),
            walk.get("step"), walk.get("warmup"),
        )
    )


def _one_sided_sign_pvalue(wins: int, losses: int) -> float:
    """P(X >= wins), X~Binomial(wins+losses, 0.5), ignoring ties."""
    w = max(0, safe_int(wins, 0))
    l = max(0, safe_int(losses, 0))
    active = w + l
    if active == 0:
        return 1.0
    total_comb = sum(math.comb(active, value) for value in range(w, active + 1))
    return total_comb / (1 << active)


def _diff(left, right) -> float:
    lv = safe_float(left, default=None)
    rv = safe_float(right, default=None)
    if lv is None or rv is None:
        return math.nan
    return lv - rv


def _relative_change(baseline, candidate) -> float | None:
    """Signed relative change ``(cand - base) / |base|`` when ``|base| > 1e-12``."""
    base = safe_float(baseline, default=None)
    cand = safe_float(candidate, default=None)
    if base is None or cand is None or abs(base) <= 1e-12:
        return None
    return (cand - base) / abs(base)


def _assumption_signature(assumptions: object) -> object:
    """Normalize scenario assumptions while ignoring human-readable description prose."""
    if isinstance(assumptions, dict):
        return {k: v for k, v in assumptions.items() if k != "description"}
    return assumptions


def _validate_promotion_reports(baseline: dict, candidate: dict) -> tuple[dict, dict]:
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("baseline and candidate must be dictionaries")
    if _identity(baseline) != _identity(candidate):
        raise ValueError("baseline and candidate use different datasets or windows")
    base_wf = baseline.get("walk_forward") if isinstance(baseline.get("walk_forward"), dict) else {}
    cand_wf = candidate.get("walk_forward") if isinstance(candidate.get("walk_forward"), dict) else {}
    if _parse_bool_flag(base_wf.get("anchored_train"), False) != _parse_bool_flag(cand_wf.get("anchored_train"), False):
        raise ValueError("baseline and candidate use different datasets or windows")
    base_venue = "" if isinstance(baseline.get("venue"), bool) else str(baseline.get("venue") or "").strip().lower()
    cand_venue = "" if isinstance(candidate.get("venue"), bool) else str(candidate.get("venue") or "").strip().lower()
    if base_venue != cand_venue:
        raise ValueError("baseline and candidate use different datasets or windows")
    raw_base_scen = baseline.get("scenarios")
    raw_cand_scen = candidate.get("scenarios")
    base_scenarios = raw_base_scen if isinstance(raw_base_scen, dict) else {}
    candidate_scenarios = raw_cand_scen if isinstance(raw_cand_scen, dict) else {}
    if not base_scenarios or set(base_scenarios) != set(candidate_scenarios):
        raise ValueError("baseline and candidate must have the same scenarios")
    for name in sorted(base_scenarios):
        if not isinstance(base_scenarios[name], dict) or not isinstance(candidate_scenarios[name], dict):
            raise ValueError(f"invalid scenario structure in scenario {name}")
    for label, rep in (("baseline", baseline), ("candidate", candidate)):
        ds = rep.get("dataset") if isinstance(rep.get("dataset"), dict) else {}
        sha = ds.get("sha256")
        bars = exact_pos_int(ds.get("bars"), default=None)
        if not isinstance(sha, str) or not sha.strip() or bars is None:
            raise ValueError(f"{label} dataset metadata (sha256, bars) is missing or invalid")
    return base_scenarios, candidate_scenarios


def _extract_scenario_windows(name: str, base: dict, cand: dict) -> tuple[dict, dict, dict, dict]:
    if not isinstance(base, dict) or not isinstance(cand, dict):
        raise ValueError(f"invalid scenario structure in scenario {name}")
    if _assumption_signature(base.get("assumptions")) != _assumption_signature(cand.get("assumptions")):
        raise ValueError(f"different execution assumptions in scenario {name}")
    base_win_list = base.get("windows") if isinstance(base.get("windows"), list) else []
    cand_win_list = cand.get("windows") if isinstance(cand.get("windows"), list) else []
    base_windows = {
        window["key"]: window
        for window in base_win_list
        if isinstance(window, dict) and isinstance(window.get("key"), (str, int)) and not isinstance(window.get("key"), bool)
    }
    cand_windows = {
        window["key"]: window
        for window in cand_win_list
        if isinstance(window, dict) and isinstance(window.get("key"), (str, int)) and not isinstance(window.get("key"), bool)
    }
    if (
        len(base_windows) != len(base_win_list)
        or len(cand_windows) != len(cand_win_list)
        or set(base_windows) != set(cand_windows)
    ):
        raise ValueError(f"different windows in scenario {name}")
    base_aggregate = base.get("aggregate") if isinstance(base.get("aggregate"), dict) else {}
    candidate_aggregate = cand.get("aggregate") if isinstance(cand.get("aggregate"), dict) else {}
    return base_windows, cand_windows, base_aggregate, candidate_aggregate


def _count_delta_outcomes(deltas: list[float], tolerance: float = 1e-10) -> tuple[int, int, int, int, float]:
    wins = sum(math.isfinite(delta) and delta > tolerance for delta in deltas)
    ties = sum(math.isfinite(delta) and abs(delta) <= tolerance for delta in deltas)
    losses = sum(math.isfinite(delta) and delta < -tolerance for delta in deltas)
    active_windows = wins + losses
    return wins, ties, losses, active_windows, _one_sided_sign_pvalue(wins, losses)


def _iter_validated_scenario_windows(baseline: dict, candidate: dict):
    base_scenarios, candidate_scenarios = _validate_promotion_reports(baseline, candidate)
    for name in sorted(base_scenarios):
        base_windows, cand_windows, base_aggregate, candidate_aggregate = _extract_scenario_windows(
            name, base_scenarios[name], candidate_scenarios[name]
        )
        yield name, base_windows, cand_windows, base_aggregate, candidate_aggregate


def evaluate_promotion(
    baseline: dict,
    candidate: dict,
    thresholds: PromotionThresholds | None = None,
) -> dict:
    """The candidate passes only if it is robust in ALL shared scenarios."""
    limits = thresholds or PromotionThresholds()
    scenario_results = {}
    for name, base_windows, cand_windows, base_aggregate, candidate_aggregate in _iter_validated_scenario_windows(
        baseline, candidate
    ):
        deltas = [
            _diff(cand_windows[key].get("return_pct"), base_windows[key].get("return_pct"))
            for key in sorted(base_windows, key=lambda k: (0, k) if isinstance(k, int) and not isinstance(k, bool) else (1, str(k)))
        ]
        all_deltas_finite = all(math.isfinite(d) for d in deltas)
        wins, ties, losses, active_windows, sign_pvalue = _count_delta_outcomes(deltas)
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
            "mean_return_delta_pp": mean_delta if math.isfinite(mean_delta) else None,
            "worst_return_delta_pp": worst_delta if math.isfinite(worst_delta) else None,
            "worst_drawdown_delta_pp": drawdown_delta if math.isfinite(drawdown_delta) else None,
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
    scenario_results = {}
    tolerance = 1e-10
    for name, base_windows, cand_windows, base_aggregate, candidate_aggregate in _iter_validated_scenario_windows(
        baseline, candidate
    ):

        drawdown_improvements = [
            _diff(base_windows[key].get("max_drawdown_pct"), cand_windows[key].get("max_drawdown_pct"))
            for key in sorted(base_windows, key=lambda k: (0, k) if isinstance(k, int) and not isinstance(k, bool) else (1, str(k)))
        ]
        all_dd_finite = all(math.isfinite(d) for d in drawdown_improvements)
        wins, ties, losses, active_windows, sign_pvalue = _count_delta_outcomes(
            drawdown_improvements, tolerance
        )
        cand_mean_return = safe_float(candidate_aggregate.get("mean_return_pct"), default=None)
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
        base_cvar = safe_float(base_aggregate.get("worst_cvar_95_pct"), default=None)
        candidate_cvar = safe_float(candidate_aggregate.get("worst_cvar_95_pct"), default=None)
        cvar_improvement = (
            candidate_cvar - base_cvar
            if base_cvar is not None and candidate_cvar is not None else None
        )
        base_exposure = safe_float(base_aggregate.get("mean_exposure_pct"), default=None)
        candidate_exposure = safe_float(candidate_aggregate.get("mean_exposure_pct"), default=None)
        exposure_delta = (
            candidate_exposure - base_exposure
            if base_exposure is not None and candidate_exposure is not None else None
        )
        base_calmar = safe_float(base_aggregate.get("median_calmar"), default=None)
        cand_calmar = safe_float(candidate_aggregate.get("median_calmar"), default=None)
        calmar_improvement = _relative_change(base_calmar, cand_calmar)
        base_sortino = safe_float(base_aggregate.get("median_sortino"), default=None)
        cand_sortino = safe_float(candidate_aggregate.get("median_sortino"), default=None)
        sortino_change = _relative_change(base_sortino, cand_sortino)
        zero_base_calmar = base_calmar is not None and abs(base_calmar) <= 1e-12
        zero_base_sortino = base_sortino is not None and abs(base_sortino) <= 1e-12

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
            "calmar_available": base_calmar is not None and cand_calmar is not None,
            "material_calmar_improvement": (
                cand_calmar is not None
                and cand_calmar > 0.0
                and (
                    (cand_calmar - base_calmar) >= limits.min_median_calmar_improvement_ratio
                    if zero_base_calmar
                    else (
                        calmar_improvement is not None
                        and calmar_improvement >= limits.min_median_calmar_improvement_ratio
                    )
                )
            ),
            "sortino_preserved": (
                cand_sortino is not None
                and cand_sortino > 0.0
                and (
                    True
                    if zero_base_sortino
                    else (
                        sortino_change is not None
                        and sortino_change >= -limits.max_median_sortino_degradation_ratio
                    )
                )
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
            "mean_return_delta_pp": mean_delta if math.isfinite(mean_delta) else None,
            "worst_return_delta_pp": worst_return_delta if math.isfinite(worst_return_delta) else None,
            "worst_drawdown_improvement_pp": (
                worst_drawdown_improvement if math.isfinite(worst_drawdown_improvement) else None
            ),
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
