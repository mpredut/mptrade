import unittest

from offline.backtests.metrics import calculate_performance_metrics


class PerformanceMetricsTest(unittest.TestCase):
    def test_return_drawdown_exposure_and_trade_metrics(self):
        result = calculate_performance_metrics(
            [100.0, 110.0, 105.0, 120.0],
            initial_capital=100.0,
            periods_per_year=365.0,
            exposure=[False, True, True],
            trade_pnls=[10.0, -5.0, 20.0],
            turnover_notional=250.0,
        )

        self.assertAlmostEqual(result["return_pct"], 20.0)
        self.assertAlmostEqual(result["max_drawdown_abs"], 5.0)
        self.assertAlmostEqual(result["max_drawdown_pct"], 100 * 5 / 110)
        self.assertEqual(result["max_underwater_periods"], 1)
        self.assertAlmostEqual(result["exposure_pct"], 200 / 3)
        self.assertEqual(result["trade_count"], 3)
        self.assertAlmostEqual(result["win_rate_pct"], 200 / 3)
        self.assertAlmostEqual(result["profit_factor"], 6.0)
        self.assertAlmostEqual(result["expectancy"], 25 / 3)
        self.assertAlmostEqual(result["turnover_pct"], 250.0)
        self.assertIsNotNone(result["sharpe"])
        self.assertIsNotNone(result["sortino"])
        self.assertIsNotNone(result["calmar"])

    def test_constant_losses_have_negative_sortino_not_zero(self):
        result = calculate_performance_metrics(
            [100.0, 99.0, 98.0, 97.0],
            initial_capital=100.0,
            periods_per_year=252.0,
        )
        self.assertLess(result["sortino"], 0.0)
        self.assertLess(result["cvar_95_pct"], 0.0)

    def test_frequency_dependent_metrics_are_not_invented(self):
        result = calculate_performance_metrics(
            [100.0, 101.0, 102.0], initial_capital=100.0,
        )
        self.assertIsNone(result["annualized_return_pct"])
        self.assertIsNone(result["sharpe"])
        self.assertIsNone(result["sortino"])
        self.assertIsNone(result["calmar"])

    def test_rejects_curve_without_initial_capital_anchor(self):
        with self.assertRaises(ValueError):
            calculate_performance_metrics([101.0, 102.0], initial_capital=100.0)

    def test_format_performance_summary(self):
        from offline.backtests.metrics import format_performance_summary
        m = {
            "max_drawdown_pct": 5.25,
            "sharpe": 1.5,
            "sortino": 2.1,
            "calmar": 3.0,
            "cvar_95_pct": -4.2,
            "exposure_pct": 45.0,
            "turnover_pct": 120.0,
            "fills": 12,
            "profit_factor": 1.8,
            "expectancy": 2.5,
        }
        lines = format_performance_summary(m, indent="  ")
        self.assertEqual(len(lines), 2)
        self.assertIn("RISC: maxDD 5.25%  Sharpe 1.50  Sortino 2.10  Calmar 3.00  CVaR95 -4.20%", lines[0])
        self.assertIn("EXECUTIE: expunere 45.00%  turnover 120.00%  fills 12  profit-factor 1.80  expectancy $2.50", lines[1])

    def test_validate_replay_metrics(self):
        from offline.backtests.metrics import validate_replay_metrics
        valid = {
            "return_pct": 10.5,
            "max_drawdown_pct": 2.1,
            "cycles": 3,
            "fills": 6,
            "initial_capital": 500.0,
            "account_currency": "USD",
            "net_pnl": 52.5,
            "realized": 52.5,
            "total": 52.5,
            "fees": 1.2,
        }
        validate_replay_metrics(valid)

        invalid_missing = dict(valid)
        del invalid_missing["fills"]
        with self.assertRaises(AssertionError):
            validate_replay_metrics(invalid_missing)

        invalid_currency = dict(valid, account_currency="")
        with self.assertRaises(AssertionError):
            validate_replay_metrics(invalid_currency)


import configparser
import math
import os
import shutil
import sys
import tempfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from offline.research.backtest_ranges import scan_backtest_ranges  # noqa: E402
from strategies import spot_engine  # noqa: E402
from offline.research.kraken_continuous_grid import grid3, report, report2, report3  # noqa: E402

_INI_CONFIGS = [os.path.join(ROOT, "instruments.conf")]


class TestScanBacktestRanges(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="backtest_ranges_")
        self._tmp = os.path.join(self._tmpdir, "backtest_ranges.conf")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, content):
        with open(self._tmp, "w", encoding="utf-8") as f:
            f.write(content)

    def test_supported_assignment_styles(self):
        cases = (
            (
                "ini",
                "# BACKTEST: 5.0, 6.0, 7.0, 8.0, 9.0\nmt.gain = 7.0\n",
                {"mt.gain": ["5.0", "6.0", "7.0", "8.0", "9.0"]},
            ),
            (
                "env",
                "# BACKTEST: 3, 4, 5.1, 6.5, 8\nTRADEALL_SLOPE_EXTREME_THRESHOLD=5.1\n",
                {"TRADEALL_SLOPE_EXTREME_THRESHOLD": ["3", "4", "5.1", "6.5", "8"]},
            ),
        )
        for label, content, expected in cases:
            with self.subTest(style=label):
                self._write(content)
                self.assertEqual(scan_backtest_ranges(self._tmp), expected)

    def test_intervening_line_breaks_association(self):
        for label, separator in (("blank", "\n"), ("comment", "# another explanatory comment\n")):
            with self.subTest(separator=label):
                self._write("# BACKTEST: 1, 2, 3\n" + separator + "mt.gain = 7.0\n")
                self.assertEqual(scan_backtest_ranges(self._tmp), {})

    def test_multiple_annotations_in_same_file(self):
        self._write(
            "# BACKTEST: 5.0, 7.0, 9.0\n"
            "mt.gain = 7.0\n"
            "mt.lost = 3.3\n"
            "# BACKTEST: 2.3, 3.3, 4.3\n"
            "mt.lost2 = 3.3\n"
        )
        result = scan_backtest_ranges(self._tmp)
        self.assertEqual(result, {"mt.gain": ["5.0", "7.0", "9.0"], "mt.lost2": ["2.3", "3.3", "4.3"]})

    def test_ini_sections_disambiguate_same_key_name(self):
        self._write(
            "[BINANCE_BTC]\n"
            "# BACKTEST: 5.0, 7.0, 8.0, 9.0\n"
            "mt.gain = 7.0\n"
            "\n"
            "[BINANCE_TAO]\n"
            "# BACKTEST: 7.5, 9.2, 10.5, 12.0\n"
            "mt.gain = 9.2\n"
        )
        result = scan_backtest_ranges(self._tmp)
        self.assertEqual(result, {
            "BINANCE_BTC.mt.gain": ["5.0", "7.0", "8.0", "9.0"],
            "BINANCE_TAO.mt.gain": ["7.5", "9.2", "10.5", "12.0"],
        })

    def test_file_without_sections_keeps_bare_key(self):
        self._write("# BACKTEST: 3, 4, 5\nSOME_KEY=4\n")
        self.assertEqual(scan_backtest_ranges(self._tmp), {"SOME_KEY": ["3", "4", "5"]})

    def test_missing_or_unannotated_returns_empty(self):
        cases = (
            ("missing", os.path.join(self._tmpdir, "does_not_exist.conf"), None),
            ("unannotated", self._tmp, "mt.gain = 7.0\nmt.lost = 3.3\n"),
        )
        for label, path, content in cases:
            with self.subTest(case=label):
                if content is not None:
                    self._write(content)
                self.assertEqual(scan_backtest_ranges(path), {})


class TestBacktestAnnotations(unittest.TestCase):
    def _current_ini_value(self, path, full_key):
        section, key = full_key.split(".", 1)
        cp = configparser.ConfigParser()
        cp.read(path)
        return float(cp[section][key])

    def test_annotations_well_formed_and_live_in_range(self):
        seen_any = False
        for path in _INI_CONFIGS:
            ranges = scan_backtest_ranges(path)
            for full_key, grid in ranges.items():
                seen_any = True
                with self.subTest(config=os.path.basename(path), key=full_key):
                    self.assertGreaterEqual(len(grid), 2, f"{full_key}: grid has < 2 values: {grid}")
                    try:
                        vals = [float(v) for v in grid]
                    except ValueError as e:
                        self.fail(f"{full_key}: grid has non-numeric values ({grid}): {e}")
                    current = self._current_ini_value(path, full_key)
                    lo, hi = min(vals), max(vals)
                    self.assertTrue(
                        lo <= current <= hi,
                        f"{full_key}: LIVE value {current} left grid interval [{lo}, {hi}] {grid} — stale grid?",
                    )
        self.assertTrue(seen_any, "no # BACKTEST annotation found — is scan or config broken?")


@pytest.fixture(autouse=True)
def restore_strategy_step():
    orig = spot_engine.Strategy.step
    yield
    spot_engine.Strategy.step = orig


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
            [5.0, 2.0, 15.0, 1],
            [-2.0, 3.0, -12.0, 0],
            [3.0, 1.0, 2.0, 1],
        ]
    }
    res = report.win_stats(sample, "w30")
    assert pytest.approx(res["mean"]) == 2.0
    assert pytest.approx(res["pos"]) == 66.66666666666667
    assert pytest.approx(res["down"]) == -2.0
    assert pytest.approx(res["up"]) == 5.0


def test_grid3_task_error_isolation():
    res = grid3.task(("NON_EXISTENT_ASSET", "live"))
    assert res["asset"] == "NON_EXISTENT_ASSET"
    assert res["cfg"] == "live"
    assert "error" in res
    assert "FileNotFoundError" in res["error"] or "No such file" in res["error"]


def test_reports_generate_empty_dir(tmp_path):
    l1 = report.generate_report(str(tmp_path / "exp1"))
    assert any("Overnight Kraken spot-DCA grid" in line for line in l1)
    l2 = report2.generate_report(str(tmp_path / "exp2"))
    assert any("Overnight wave 2" in line for line in l2)
    l3 = report3.generate_report(str(tmp_path / "exp3"))
    assert any("Finite-cash continuous runs" in line for line in l3)


def test_old_baseline_disables_post_0923_promotions():
    from offline.research.kraken_continuous_grid import grid as kgrid
    assert kgrid.OLD.trend_overlay is False
    assert kgrid.OLD.dca_trend_brake is False
    assert kgrid.OLD.surge_guard is False
    assert kgrid.OLD.fast_profit_guard is False
    assert kgrid.OLD.tp_dynamic_flat is False
    assert kgrid.CONFIGS["old_live_pre0923"][0] != kgrid.CONFIGS["ov650_t8_oldlive"][0]


def test_t212_grid_peak_reentry_fallback_for_zero_drop_profile():
    from offline.research.t212_continuous_grid import grid as tgrid
    orig_nvda = dict(tgrid.PROFILES["nvda"])
    try:
        p_live, pk_live = tgrid.params_for("nvda", "live")
        p_peak, pk_peak = tgrid.params_for("nvda", "reentry_peak")
        assert pk_live == 0
        assert pk_peak == 1
        assert p_live.reentry_drop_pct == 0.0
        assert p_peak.reentry_drop_pct == 1.5
        tgrid.PROFILES["nvda"] = {**orig_nvda, "STRAT_REENTRY_PULLBACK_PCT": "0"}
        p_peak_zero, _ = tgrid.params_for("nvda", "reentry_peak")
        assert p_peak_zero.reentry_drop_pct == 2.0
    finally:
        tgrid.PROFILES["nvda"] = orig_nvda
        tgrid.strat.Strategy.step = tgrid._orig_step


if __name__ == "__main__":
    unittest.main()
