#!/usr/bin/env python3
"""tests/test_kraken_continuous_grid_reporting.py — Unit tests for kraken continuous grid reporting and error isolation."""

from __future__ import annotations

import math
import pytest
from offline.research.kraken_continuous_grid import grid3, report, report2, report3


def test_win_stats_empty():
    res = report.win_stats({"w90": []}, "w90")
    assert math.isnan(res["mean"])
    assert math.isnan(res["pos"])
    assert math.isnan(res["p10"])
    assert math.isnan(res["down"])
    assert math.isnan(res["up"])


def test_win_stats_with_data():
    sample = {
        "w30": [
            [5.0, 2.0, 15.0, 1],   # up market (>10%)
            [-2.0, 3.0, -12.0, 0], # down market (<-10%)
            [3.0, 1.0, 2.0, 1],    # sideways
        ]
    }
    res = report.win_stats(sample, "w30")
    assert pytest.approx(res["mean"]) == 2.0
    assert pytest.approx(res["pos"]) == 66.66666666666667
    assert pytest.approx(res["down"]) == -2.0
    assert pytest.approx(res["up"]) == 5.0


def test_grid3_task_error_isolation():
    # Calling task with a non-existent asset must return an error dict, not raise an uncaught exception
    res = grid3.task(("NON_EXISTENT_ASSET", "live"))
    assert res["asset"] == "NON_EXISTENT_ASSET"
    assert res["cfg"] == "live"
    assert "error" in res
    assert "FileNotFoundError" in res["error"] or "No such file" in res["error"]


def test_reports_generate_empty_dir(tmp_path):
    # Calling generate_report with an empty directory must create the report without crashing
    l1 = report.generate_report(str(tmp_path / "exp1"))
    assert any("Overnight Kraken spot-DCA grid" in line for line in l1)

    l2 = report2.generate_report(str(tmp_path / "exp2"))
    assert any("Overnight wave 2" in line for line in l2)

    l3 = report3.generate_report(str(tmp_path / "exp3"))
    assert any("Finite-cash continuous runs" in line for line in l3)
