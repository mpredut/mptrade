#!/usr/bin/env python3
"""tests/test_comparison.py — Unit tests for offline.backtests.comparison."""

from __future__ import annotations

import pytest

from offline.backtests.comparison import (
    dominates,
    format_comparison_markdown,
    format_number,
    pareto_frontier,
    relative_to_baseline,
)


def test_relative_to_baseline_scoring():
    baseline = [
        {"key": "w1", "return_pct": 5.0},
        {"key": "w2", "return_pct": 10.0},
        {"key": "w3", "return_pct": -2.0},
    ]

    candidate = [
        {"key": "w1", "return_pct": 7.0},   # +2 (win)
        {"key": "w2", "return_pct": 10.0},  # +0 (tie)
        {"key": "w3", "return_pct": -4.0},  # -2 (loss)
    ]

    res = relative_to_baseline(candidate, baseline)
    assert res["wins_vs_live"] == 1
    assert res["ties_vs_live"] == 1
    assert res["losses_vs_live"] == 1
    assert pytest.approx(res["mean_return_delta_pct"]) == 0.0


def test_relative_to_baseline_empty():
    res = relative_to_baseline([], [])
    assert res["wins_vs_live"] == 0
    assert res["mean_return_delta_pct"] == 0.0


def test_dominates():
    a = {"mean_return_pct": 10.0, "worst_return_pct": -2.0, "worst_max_drawdown_pct": 5.0}
    b = {"mean_return_pct": 8.0, "worst_return_pct": -3.0, "worst_max_drawdown_pct": 6.0}
    c = {"mean_return_pct": 10.0, "worst_return_pct": -2.0, "worst_max_drawdown_pct": 5.0}
    d = {"mean_return_pct": 12.0, "worst_return_pct": -4.0, "worst_max_drawdown_pct": 4.0}

    # a dominates b in all dimensions
    assert dominates(a, b) is True
    # b does not dominate a
    assert dominates(b, a) is False

    # Identical performance: no strictly better dimension, so neither dominates the other
    assert dominates(a, c) is False
    assert dominates(c, a) is False

    # d has higher mean, but worse worst_return_pct (-4 vs -2), so does not dominate a
    assert dominates(d, a) is False
    assert dominates(a, d) is False


def test_pareto_frontier():
    results = {
        "A": {"summary": {"mean_return_pct": 10.0, "worst_return_pct": -2.0, "worst_max_drawdown_pct": 5.0}},
        "B": {"summary": {"mean_return_pct": 8.0, "worst_return_pct": -3.0, "worst_max_drawdown_pct": 6.0}},  # dominated by A
        "C": {"summary": {"mean_return_pct": 12.0, "worst_return_pct": -5.0, "worst_max_drawdown_pct": 4.0}}, # trade-off (higher mean, lower DD, worse tail)
    }

    frontier = pareto_frontier(results)
    assert "A" in frontier
    assert "C" in frontier
    assert "B" not in frontier


def test_format_number():
    assert format_number(None) == "n/a"
    assert format_number(2.3456, signed=False) == "2.346"
    assert format_number(2.3456, signed=True) == "+2.346"
    assert format_number(-1.2, signed=True) == "-1.200"


def test_format_comparison_markdown():
    report = {
        "generated_at_utc": "2026-10-14T21:00:00Z",
        "candidates": {
            "live": {
                "summary": {
                    "mean_return_pct": 5.0, "median_return_pct": 4.5,
                    "worst_return_pct": -1.0, "worst_max_drawdown_pct": 3.0,
                    "positive_windows": 8, "window_count": 10, "total_cycles": 15,
                },
                "relative_to_live": {"wins_vs_live": 0, "ties_vs_live": 10, "losses_vs_live": 0},
            },
            "candidate_a": {
                "summary": {
                    "mean_return_pct": 6.5, "median_return_pct": 5.5,
                    "worst_return_pct": -0.5, "worst_max_drawdown_pct": 2.5,
                    "positive_windows": 9, "window_count": 10, "total_cycles": 18,
                },
                "relative_to_live": {"wins_vs_live": 7, "ties_vs_live": 2, "losses_vs_live": 1},
            },
        },
        "pareto_frontier": ["candidate_a"],
        "fee_stress": {
            "0.26": {
                "summary": {
                    "mean_return_pct": 5.0, "worst_return_pct": -1.0,
                    "worst_max_drawdown_pct": 3.0, "positive_windows": 8, "window_count": 10,
                }
            }
        },
    }

    md = format_comparison_markdown(report)
    assert "# Walk-Forward Robustness Comparison" in md
    assert "| `candidate_a` | +6.500 | +5.500 | -0.500 | 2.500 | 9/10 | 7/2/1 | 18 |" in md
    assert "Pareto frontier: `candidate_a`" in md
    assert "## Fee stress — live configuration" in md


def test_format_comparison_markdown_none_mean_return():
    report = {
        "candidates": {
            "empty_cand": {
                "summary": {"mean_return_pct": None, "worst_return_pct": None},
                "relative_to_live": {},
            }
        }
    }
    md = format_comparison_markdown(report)
    assert "| `empty_cand` | n/a | n/a | n/a |" in md


def test_dominates_and_pareto_frontier_non_finite_and_bool():
    valid = {"mean_return_pct": 5.0, "worst_return_pct": -1.0, "worst_max_drawdown_pct": 3.0}
    nan_summary = {"mean_return_pct": float("nan"), "worst_return_pct": -1.0, "worst_max_drawdown_pct": 3.0}
    none_summary = {"mean_return_pct": None, "worst_return_pct": None, "worst_max_drawdown_pct": None}
    assert dominates(valid, nan_summary) is False
    assert dominates(nan_summary, valid) is False
    assert dominates(valid, none_summary) is False
    assert format_number(True) == "n/a"
    assert format_number(False) == "n/a"

    frontier = pareto_frontier({
        "valid": {"summary": valid},
        "nan_cand": {"summary": nan_summary},
        "none_cand": {"summary": none_summary},
    })
    assert frontier == ["valid"]


