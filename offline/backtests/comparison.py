#!/usr/bin/env python3
"""offline/backtests/comparison.py — Multi-criteria candidate comparison and Pareto dominance.

Provides venue-agnostic comparison math for walk-forward evaluation windows:
- Conservative Pareto dominance (better or equal on all primary objectives, strictly better on at least one)
- Non-dominated set extraction (Pareto frontier)
- Relative win/loss/tie scoring against a control baseline
- Standard Markdown robustness tables
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def relative_to_baseline(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    key_field: str = "key",
    metric_field: str = "return_pct",
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compute relative win, loss, tie counts and mean return delta against baseline."""
    base_by_key = {window[key_field]: window for window in baseline}
    deltas = [
        float(window[metric_field]) - float(base_by_key[window[key_field]][metric_field])
        for window in candidate
        if window[key_field] in base_by_key
    ]
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


def dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Conservative Pareto dominance: better/equal mean and worst return, drawdown no larger.

    Strictly better in at least one dimension.
    """
    no_worse = (
        float(left["mean_return_pct"]) >= float(right["mean_return_pct"]) - tolerance
        and float(left["worst_return_pct"]) >= float(right["worst_return_pct"]) - tolerance
        and float(left["worst_max_drawdown_pct"]) <= (
            float(right["worst_max_drawdown_pct"]) + tolerance
        )
    )
    strictly_better = (
        float(left["mean_return_pct"]) > float(right["mean_return_pct"]) + tolerance
        or float(left["worst_return_pct"]) > float(right["worst_return_pct"]) + tolerance
        or float(left["worst_max_drawdown_pct"]) < (
            float(right["worst_max_drawdown_pct"]) - tolerance
        )
    )
    return no_worse and strictly_better


def pareto_frontier(
    results: Mapping[str, Mapping[str, Any]],
    *,
    summary_key: str = "summary",
) -> list[str]:
    """Return sorted names of non-dominated candidates from results dict."""
    names = []
    for name, candidate in results.items():
        summary = candidate.get(summary_key, candidate)
        if not any(
            other_name != name
            and dominates(other.get(summary_key, other), summary)
            for other_name, other in results.items()
        ):
            names.append(name)
    return sorted(names)


import math


def format_number(value: float | None, signed: bool = False) -> str:
    """Format floating point numbers consistently across report tables."""
    if value is None or (isinstance(value, (int, float)) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


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
            -float(item[1].get("summary", {}).get("mean_return_pct", 0.0)),
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
