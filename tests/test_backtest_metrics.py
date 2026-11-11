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
        self.assertIn("RISK: maxDD 5.25%  Sharpe 1.50  Sortino 2.10  Calmar 3.00  CVaR95 -4.20%", lines[0])
        self.assertIn("EXECUTION: exposure 45.00%  turnover 120.00%  fills 12  profit-factor 1.80  expectancy $2.50", lines[1])

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


# ---------------------------------------------------------------------------
# Consolidated from tests/test_comparison.py & tests/test_backtest_datasets.py
# ---------------------------------------------------------------------------

import json
from pathlib import Path
import tempfile

from offline.backtests.comparison import (
    dominates,
    format_comparison_markdown,
    format_number,
    pareto_frontier,
    relative_to_baseline,
)
from offline.backtests.datasets import (
    align_previous_values,
    dataset_metadata,
    drop_incomplete_last_bar,
    load_dataset,
    merge_datasets,
    save_dataset,
    validate_dataset,
)
from offline.backtests.execution import (
    ExecutionModel,
    FeeModel,
    choose_intrabar_scenario,
    split_order_fill,
)
from offline.backtests.hype_candidates import (
    financial_priority_candidates,
    hype_240_candidates,
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

    assert dominates(a, b) is True
    assert dominates(b, a) is False
    assert dominates(a, c) is False
    assert dominates(c, a) is False
    assert dominates(d, a) is False
    assert dominates(a, d) is False


def test_pareto_frontier():
    results = {
        "A": {"summary": {"mean_return_pct": 10.0, "worst_return_pct": -2.0, "worst_max_drawdown_pct": 5.0}},
        "B": {"summary": {"mean_return_pct": 8.0, "worst_return_pct": -3.0, "worst_max_drawdown_pct": 6.0}},
        "C": {"summary": {"mean_return_pct": 12.0, "worst_return_pct": -5.0, "worst_max_drawdown_pct": 4.0}},
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
    report_data = {
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

    md = format_comparison_markdown(report_data)
    assert "# Walk-Forward Robustness Comparison" in md
    assert "| `candidate_a` | +6.500 | +5.500 | -0.500 | 2.500 | 9/10 | 7/2/1 | 18 |" in md
    assert "Pareto frontier: `candidate_a`" in md
    assert "## Fee stress - live configuration" in md


def test_format_comparison_markdown_none_mean_return():
    report_data = {
        "candidates": {
            "empty_cand": {
                "summary": {"mean_return_pct": None, "worst_return_pct": None},
                "relative_to_live": {},
            }
        }
    }
    md = format_comparison_markdown(report_data)
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


def _dataset_rows():
    return [
        {"timestamp": 0, "open": 10, "high": 12, "low": 9, "close": 11},
        {"timestamp": 3600, "open": 11, "high": 13, "low": 10, "close": 12},
    ]


class OhlcDatasetContractTest(unittest.TestCase):
    def test_drops_only_the_incomplete_intraday_bar(self):
        rows = _dataset_rows()
        self.assertEqual(
            drop_incomplete_last_bar(rows, interval_minutes=60, now_timestamp=7199),
            rows[:1],
        )
        self.assertEqual(
            drop_incomplete_last_bar(rows, interval_minutes=60, now_timestamp=7200),
            rows,
        )

    def test_drops_multiple_incomplete_yahoo_tail_rows(self):
        rows = _dataset_rows() + [
            {"timestamp": 3900, "open": 12, "high": 13, "low": 11, "close": 12},
        ]
        self.assertEqual(
            drop_incomplete_last_bar(rows, interval_minutes=60, now_timestamp=3700),
            rows[:1],
        )

    def test_daily_bar_uses_regular_session_end(self):
        rows = [{"timestamp": 100, "open": 10, "high": 12, "low": 9, "close": 11}]
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=1440, now_timestamp=150,
                regular_session_start=100, regular_session_end=200,
            ),
            [],
        )
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=1440, now_timestamp=200,
                regular_session_start=100, regular_session_end=200,
            ),
            rows,
        )

    def test_asof_alignment_never_uses_a_future_fx_value(self):
        fx = [
            {"timestamp": 10, "close": 0.20},
            {"timestamp": 20, "close": 0.25},
        ]
        self.assertEqual(
            align_previous_values([10, 19, 20, 30], fx),
            [0.20, 0.20, 0.25, 0.25],
        )
        with self.assertRaisesRegex(ValueError, "extend the FX dataset"):
            align_previous_values([9], fx)

    def test_round_trip_is_canonical_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            save_dataset(_dataset_rows(), path)
            loaded = load_dataset(path)
            self.assertEqual(loaded, validate_dataset(_dataset_rows(), interval_minutes=60))
            metadata = dataset_metadata(loaded, interval_minutes=60)
            self.assertEqual(metadata["bars"], 2)
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_merge_overlapping_windows_preserves_history_and_newer_value(self):
        older = _dataset_rows()
        newer = [
            {**_dataset_rows()[1], "close": 12.5, "high": 13.5},
            {"timestamp": 7200, "open": 12.5, "high": 14, "low": 12, "close": 13},
        ]
        merged = merge_datasets(older, newer)

        self.assertEqual([row["timestamp"] for row in merged], [0, 3600, 7200])
        self.assertEqual(merged[1]["close"], 12.5)

    def test_rejects_gap_duplicate_and_invalid_ohlc(self):
        with self.assertRaisesRegex(ValueError, "cadence"):
            validate_dataset([_dataset_rows()[0], {**_dataset_rows()[1], "timestamp": 7200}], interval_minutes=60)
        with self.assertRaisesRegex(ValueError, "duplicat"):
            validate_dataset([_dataset_rows()[0], {**_dataset_rows()[1], "timestamp": 0}])
        with self.assertRaisesRegex(ValueError, "OHLC"):
            validate_dataset([{"timestamp": 0, "open": 10, "high": 9, "low": 8, "close": 10}])

    def test_versioned_hype_manifest_matches_frozen_files(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "offline/research/hype_dataset"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for interval, expected in manifest["datasets"].items():
            records = load_dataset(directory / expected["file"])
            actual = dataset_metadata(records, interval_minutes=int(interval))
            for key in ("sha256", "bars", "start_utc", "end_utc"):
                self.assertEqual(actual[key], expected[key])

    def test_versioned_t212_manifest_matches_frozen_files(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "offline/research/t212_dataset"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for expected in manifest["datasets"].values():
            records = load_dataset(directory / expected["file"])
            actual = dataset_metadata(records)
            for key in ("sha256", "bars", "start_utc", "end_utc"):
                self.assertEqual(actual[key], expected[key])


class ExecutionModelTest(unittest.TestCase):
    def test_fee_model_separates_limit_and_market_costs(self):
        fees = FeeModel(limit_fee_pct=0.16, market_fee_pct=0.26)
        self.assertEqual(fees.rate_pct(market=False), 0.16)
        self.assertEqual(fees.rate_pct(market=True), 0.26)

        for invalid in (-0.01, float("nan"), 100.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    FeeModel(limit_fee_pct=invalid, market_fee_pct=0.26)

    def test_spread_changes_limit_touch_and_market_slippage_is_adverse(self):
        plain = ExecutionModel()
        stressed = ExecutionModel(spread_bps=20, market_slippage_bps=30)
        self.assertTrue(plain.limit_touched("buy", high=101, low=99.95, limit=100))
        self.assertFalse(stressed.limit_touched("buy", high=101, low=99.95, limit=100))
        self.assertAlmostEqual(stressed.market_price("buy", 100), 100.4)
        self.assertAlmostEqual(stressed.market_price("sell", 100), 99.6)

    def test_partial_fill_uses_original_quantity_and_prorates_amount(self):
        order = {"side": "BUY", "kind": "DCA", "qty": 10.0, "amount": 100.0}
        first, qty1, done1 = split_order_fill(order, quantity_key="qty", amount_key="amount", ratio=0.4)
        second, qty2, done2 = split_order_fill(order, quantity_key="qty", amount_key="amount", ratio=0.4)
        third, qty3, done3 = split_order_fill(order, quantity_key="qty", amount_key="amount", ratio=0.4)
        self.assertEqual((qty1, qty2, qty3), (4.0, 4.0, 2.0))
        self.assertEqual((done1, done2, done3), (False, False, True))
        self.assertAlmostEqual(first["amount"] + second["amount"] + third["amount"], 100)
        self.assertEqual(first["kind"], "DCA")
        self.assertEqual(second["kind"], "DCA_PARTIAL")

    def test_worst_case_selects_lower_return(self):
        result = choose_intrabar_scenario(
            ExecutionModel(intrabar_policy="worst_case"),
            lambda model: {
                "return_pct": 2.0 if model.intrabar_policy == "sell_first" else -1.0,
                "max_drawdown_pct": 3.0, "fills": 1, "ambiguous_bars": 1,
            },
        )
        self.assertEqual(result["intrabar_policy_selected"], "buy_first")
        self.assertEqual(result["return_pct"], -1.0)
        self.assertEqual(set(result["intrabar_scenarios"]), {"buy_first", "sell_first"})

    def test_invalid_assumptions_are_rejected(self):
        with self.assertRaises(ValueError):
            ExecutionModel(partial_fill_ratio=0)
        with self.assertRaises(ValueError):
            ExecutionModel(spread_bps=-1)


class HypeCandidatesTest(unittest.TestCase):
    def test_priority_set_is_unique_fixed_and_starts_with_live(self):
        candidates = financial_priority_candidates()
        names = [candidate.name for candidate in candidates]

        self.assertEqual(names[0], "live")
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            names,
            [
                "live", "reentry4", "tp4", "dca15", "dca_progressive025", "dca_vol_m1",
                "A_trail", "B_dcabrake", "overlay650t8",
                "trail_profit_floor_sl18", "trail_profit_floor_sl125",
            ],
        )

    def test_tp4_and_dca15_are_one_factor_candidates(self):
        by_name = {candidate.name: candidate for candidate in financial_priority_candidates()}
        self.assertEqual(by_name["tp4"].overrides, {"takeprofit_pct": 4.0})
        self.assertEqual(by_name["reentry4"].overrides, {"reentry_drop_pct": 4.0})
        self.assertEqual(by_name["dca15"].overrides, {"dca_drop_pct": 1.5})
        self.assertEqual(by_name["dca_progressive025"].overrides, {"dca_spacing_growth_pct": 0.25})
        self.assertEqual(
            by_name["dca_vol_m1"].overrides,
            {"dca_vol_scale_k": -1.0, "dca_vol_ref": 2.0, "dca_vol_interval": 240},
        )
        self.assertEqual(
            by_name["trail_profit_floor_sl18"].overrides,
            {"tp_trail_profit_floor_pct": 1.0, "stop_loss_pct": 18.0},
        )
        self.assertEqual(
            by_name["trail_profit_floor_sl125"].overrides,
            {"tp_trail_profit_floor_pct": 1.0},
        )

    def test_walk_forward_registry_keeps_historical_report_names(self):
        self.assertEqual(
            [candidate.name for candidate in hype_240_candidates()],
            [
                "live", "overlay_orig", "overlay650t8", "A_adaptive_trail",
                "B_dca_brake", "tp_4", "dca_drop_1_5",
            ],
        )


def test_backtest_ranges_accepts_empty_valued_key(tmp_path):
    cfg = tmp_path / "empty.env"
    cfg.write_text("# BACKTEST: 2:50,4:50, 3:50,6:50\nSTRAT_TP_TRANCHES=\n", encoding="utf-8")
    assert scan_backtest_ranges(str(cfg)) == {"STRAT_TP_TRANCHES": ["2:50", "4:50", "3:50", "6:50"]}


def test_short_trend_benchmark_sparse_fallback_does_not_poison_365_npz(tmp_path, monkeypatch):
    import numpy as np
    import offline.research.short_trend_estimator_benchmark as st_bench

    sparse_file = tmp_path / "cache_sparse_price_SPARSECOIN.jsonl"
    sparse_file.write_text(
        '{"s": "SPARSECOIN", "i": [1700000000000, 100.0]}\n'
        '{"s": "SPARSECOIN", "i": [1700000300000, 101.5]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(st_bench, "D", str(tmp_path))
    monkeypatch.setattr(st_bench, "resolve_cachedb_filepath", lambda p: p)
    ts, px = st_bench.load_series("SPARSECOIN")
    assert len(ts) == 2 and len(px) == 2
    assert not (tmp_path / "cache_365_price_SPARSECOIN.npz").exists()
    assert (tmp_path / "cache_sparse_price_SPARSECOIN.npz").exists()

    # IS-only Platt scaling ignores OOS extreme spikes
    sig = np.array([0.05, -0.05, 0.08, -0.08, 1000.0, -1000.0], dtype=np.float64)
    gated_is = st_bench.platt_scale(sig, thr=0.60, fit_end=4)
    assert gated_is[2] != 0.0


def test_chop_weight_backtest_zero_trade_pf_and_asof_btc_lookup():
    from offline.research import chop_weight_backtest as cwb

    empty_stats = cwb.run_single_simulation("TAOUSDC", [], None, "CHOP_0.03")
    assert empty_stats.profit_factor == 0.0

    # Unaligned BTC timestamps (offset by 500s across 900s bucket boundary) still resolve as-of BTC regime
    sampled_alt = [(10_000.0 + i * 900.0, 100.0 + (i % 3) * 0.1) for i in range(40)]
    sampled_btc = [(9_800.0 + i * 900.0, 50_000.0 + i * 250.0) for i in range(40)]
    tape = cwb._precompute_chop_bar_tape("TAOUSDC", sampled_alt, sampled_btc, step_seconds=900)
    assert any(bar["btc_regime"] == "BULL" for bar in tape)


if __name__ == "__main__":
    unittest.main()
