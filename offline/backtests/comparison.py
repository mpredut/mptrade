#!/usr/bin/env python3
"""offline/backtests/comparison.py — Multi-criteria candidate comparison and Pareto dominance.

Provides venue-agnostic comparison math for walk-forward evaluation windows:
- Conservative Pareto dominance (better or equal on all primary objectives, strictly better on at least one)
- Non-dominated set extraction (Pareto frontier)
- Relative win/loss/tie scoring against a control baseline
- Standard Markdown robustness tables
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def relative_to_baseline(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    key_field: str = "key",
    metric_field: str = "return_pct",
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compute relative win, loss, tie counts and mean return delta against baseline."""
    base_by_key = {
        window[key_field]: window
        for window in baseline
        if isinstance(window, Mapping) and key_field in window
    }
    deltas = []
    for window in candidate:
        if not isinstance(window, Mapping) or window.get(key_field) not in base_by_key:
            continue
        c_val = _finite_float(window.get(metric_field))
        b_val = _finite_float(base_by_key[window[key_field]].get(metric_field))
        if c_val is not None and b_val is not None:
            deltas.append(c_val - b_val)
    if not deltas:
        return {
            "mean_return_delta_pct": 0.0,
            "wins_vs_live": 0,
            "ties_vs_live": 0,
            "losses_vs_live": 0,
        }
    return {
        "mean_return_delta_pct": sum(deltas) / len(deltas),
        "wins_vs_live": sum(delta > tolerance for delta in deltas),
        "ties_vs_live": sum(abs(delta) <= tolerance for delta in deltas),
        "losses_vs_live": sum(delta < -tolerance for delta in deltas),
    }


def _summary_metrics(summary: Mapping[str, Any]) -> tuple[float, float, float] | None:
    if not isinstance(summary, Mapping):
        return None
    mean_ret = _finite_float(summary.get("mean_return_pct"))
    worst_ret = _finite_float(summary.get("worst_return_pct"))
    worst_dd = _finite_float(summary.get("worst_max_drawdown_pct"))
    if mean_ret is None or worst_ret is None or worst_dd is None:
        return None
    return mean_ret, worst_ret, worst_dd


def dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Conservative Pareto dominance: better/equal mean and worst return, drawdown no larger.

    Strictly better in at least one dimension.
    """
    left_vals = _summary_metrics(left)
    right_vals = _summary_metrics(right)
    if left_vals is None or right_vals is None:
        return False
    l_mean, l_worst, l_dd = left_vals
    r_mean, r_worst, r_dd = right_vals
    no_worse = (
        l_mean >= r_mean - tolerance
        and l_worst >= r_worst - tolerance
        and l_dd <= r_dd + tolerance
    )
    strictly_better = (
        l_mean > r_mean + tolerance
        or l_worst > r_worst + tolerance
        or l_dd < r_dd - tolerance
    )
    return no_worse and strictly_better


def pareto_frontier(
    results: Mapping[str, Mapping[str, Any]],
    *,
    summary_key: str = "summary",
) -> list[str]:
    """Return sorted names of non-dominated candidates from results dict."""
    valid_items = {
        name: candidate
        for name, candidate in results.items()
        if isinstance(candidate, Mapping)
        and _summary_metrics(candidate.get(summary_key, candidate)) is not None
    }
    pool = valid_items if valid_items else results
    names = []
    for name, candidate in pool.items():
        summary = candidate.get(summary_key, candidate) if isinstance(candidate, Mapping) else {}
        if not any(
            other_name != name
            and dominates(
                other.get(summary_key, other) if isinstance(other, Mapping) else {},
                summary,
            )
            for other_name, other in pool.items()
        ):
            names.append(name)
    return sorted(names)


def format_number(value: float | None, signed: bool = False) -> str:
    """Format floating point numbers consistently across report tables."""
    fval = _finite_float(value)
    if fval is None:
        return "n/a"
    return f"{fval:+.3f}" if signed else f"{fval:.3f}"


def format_comparison_markdown(
    report: dict[str, Any],
    title: str = "Walk-Forward Robustness Comparison",
) -> str:
    """Render canonical Markdown table of candidate comparison results."""
    lines = [
        f"# {title}",
        "",
        f"Generated UTC: {report.get('generated_at_utc', 'n/a')}",
        "",
        "Every candidate uses the same frozen TEST windows. No candidate is activated.",
        "",
        "| Candidate | Mean % | Median % | Worst % | Worst DD % | Positive | W/T/L vs live | Cycles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    candidates = report.get("candidates", {})
    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            item[0] != "live",
            -(_finite_float((item[1].get("summary") or {}).get("mean_return_pct")) or 0.0),
            item[0],
        ),
    )
    for name, item in ordered:
        summary = item.get("summary", {})
        relative = item.get("relative_to_live", {})
        lines.append(
            f"| `{name}` | {format_number(summary.get('mean_return_pct'), True)} | "
            f"{format_number(summary.get('median_return_pct'), True)} | "
            f"{format_number(summary.get('worst_return_pct'), True)} | "
            f"{format_number(summary.get('worst_max_drawdown_pct'))} | "
            f"{summary.get('positive_windows', 0)}/{summary.get('window_count', 0)} | "
            f"{relative.get('wins_vs_live', 0)}/{relative.get('ties_vs_live', 0)}/"
            f"{relative.get('losses_vs_live', 0)} | {summary.get('total_cycles', 0)} |"
        )
    pareto = report.get("pareto_frontier", [])
    pareto_str = ", ".join(f"`{name}`" for name in pareto) if pareto else "none"
    lines.extend([
        "",
        f"Pareto frontier: {pareto_str}",
        "",
    ])

    fee_stress = report.get("fee_stress", {})
    if fee_stress:
        lines.extend([
            "## Fee stress — live configuration",
            "",
            "| Fee per leg % | Mean % | Worst % | Worst DD % | Positive |",
            "|---:|---:|---:|---:|---:|",
        ])
        for fee, item in fee_stress.items():
            summary = item.get("summary", {})
            lines.append(
                f"| {fee} | {format_number(summary.get('mean_return_pct'), True)} | "
                f"{format_number(summary.get('worst_return_pct'), True)} | "
                f"{format_number(summary.get('worst_max_drawdown_pct'))} | "
                f"{summary.get('positive_windows', 0)}/{summary.get('window_count', 0)} |"
            )
        lines.append("")
    return "\n".join(lines)
