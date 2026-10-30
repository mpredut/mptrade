#!/usr/bin/env python3
"""Tests for the canonical multi-venue walk-forward comparison runner."""
from __future__ import annotations

import unittest
from pathlib import Path

from offline.runners.walk_forward_compare import (
    _detect_venue,
    _markdown,
    default_candidates,
    DEFAULT_FEE_STRESS,
)
import offline.runners.kraken_walk_forward_compare as kraken_compare


class WalkForwardCompareTest(unittest.TestCase):
    def test_detect_venue_explicit_field(self):
        self.assertEqual(_detect_venue({"venue": "binance"}), "binance")
        self.assertEqual(_detect_venue({"venue": "Hyperliquid"}), "hyperliquid")
        self.assertEqual(_detect_venue({"venue": "Trading212"}), "trading212")
        self.assertEqual(_detect_venue({"venue": "kraken"}), "kraken")

    def test_detect_venue_cli_override(self):
        self.assertEqual(
            _detect_venue({"venue": "kraken"}, cli_venue="binance"),
            "binance",
        )

    def test_detect_venue_purpose_heuristics(self):
        self.assertEqual(
            _detect_venue({"purpose": "Binance live-engine fixed-config walk-forward baseline"}),
            "binance",
        )
        self.assertEqual(
            _detect_venue({"purpose": "Hyperliquid spot walk-forward baseline"}),
            "hyperliquid",
        )
        self.assertEqual(
            _detect_venue({"purpose": "Trading 212 equity walk-forward baseline"}),
            "trading212",
        )
        self.assertEqual(
            _detect_venue({"purpose": "fixed-live-config walk-forward baseline; no parameter selection"}),
            "kraken",
        )

    def test_default_candidates_not_empty(self):
        candidates = default_candidates()
        self.assertTrue(len(candidates) >= 10)
        names = {c.name for c in candidates}
        self.assertIn("live", names)
        self.assertIn("classic_tp", names)
        self.assertIn("dca_drop_1_5", names)

    def test_default_fee_stress_schedules(self):
        for venue in ("kraken", "binance", "hyperliquid", "trading212"):
            self.assertIn(venue, DEFAULT_FEE_STRESS)
            fees = [float(f) for f in DEFAULT_FEE_STRESS[venue].split(",")]
            self.assertTrue(all(f >= 0.0 for f in fees))
            self.assertEqual(fees, sorted(fees))

    def test_markdown_renders_title_with_venue(self):
        mock_report = {
            "source_baseline_report": "/tmp/test.json",
            "method": {
                "candidate_style": "standard",
                "comparison_dimensions": ["mean_return_pct"],
                "window_weighting": "equal",
                "execution_model": {},
            },
            "candidates": {
                "live": {
                    "summary": {
                        "mean_return_pct": 1.0,
                        "worst_return_pct": -5.0,
                        "worst_max_drawdown_pct": 8.0,
                    },
                    "relative_to_live": {
                        "delta_mean_return_pct": 0.0,
                        "delta_worst_return_pct": 0.0,
                        "delta_worst_drawdown_pct": 0.0,
                        "wins_vs_live": 0,
                        "ties_vs_live": 1,
                        "losses_vs_live": 0,
                    },
                    "dominates_live": False,
                    "by_interval": {},
                    "description": "live config",
                }
            },
            "pareto_frontier": ["live"],
            "dominates_live": [],
            "fee_stress": {},
        }
        md = _markdown(mock_report, venue="binance")
        self.assertIn("Binance walk-forward robustness comparison", md)
        self.assertIn("Pareto", md)

    def test_kraken_shim_compatibility(self):
        self.assertIs(kraken_compare.default_candidates, default_candidates)
        self.assertTrue(hasattr(kraken_compare, "main"))
        self.assertTrue(hasattr(kraken_compare, "_candidate_windows"))

    def test_load_report_supports_both_schemas(self):
        from offline.runners.walk_forward_compare import _load_report
        import tempfile
        import json

        # Multi-interval format (Kraken)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 1, "intervals": {"240": {}}}, f)
            f.flush()
            rep = _load_report(Path(f.name))
            self.assertEqual(rep["schema_version"], 1)

        # Single-interval format (Binance/Hyperliquid)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 1, "dataset": {}, "windows": {}}, f)
            f.flush()
            rep = _load_report(Path(f.name))
            self.assertEqual(rep["schema_version"], 1)

        # Single-interval format with window_sizes (Trading 212)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 1, "dataset": {}, "window_sizes": {}}, f)
            f.flush()
            rep = _load_report(Path(f.name))
            self.assertEqual(rep["schema_version"], 1)

        # Invalid format
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump({"schema_version": 2}, f)
            f.flush()
            with self.assertRaises(ValueError):
                _load_report(Path(f.name))

    def test_detect_venue_trading212_no_space_and_ticker(self):
        self.assertEqual(
            _detect_venue({"purpose": "Trading212 live-engine fixed-config walk-forward baseline"}),
            "trading212",
        )
        self.assertEqual(
            _detect_venue({"runtime": {"ticker": "NVDA_US_EQ"}}),
            "trading212",
        )

    def test_format_number_non_finite(self):
        from offline.backtests.comparison import format_number
        self.assertEqual(format_number(None), "n/a")
        self.assertEqual(format_number(float("nan")), "n/a")
        self.assertEqual(format_number(float("inf")), "n/a")
        self.assertEqual(format_number(-float("inf")), "n/a")
        self.assertEqual(format_number(1.23456), "1.235")
        self.assertEqual(format_number(1.23456, signed=True), "+1.235")

    def test_summarize_test_windows_handles_none_buy_hold(self):
        from offline.backtests.walk_forward import summarize_test_windows
        windows = [
            {"return_pct": 5.0, "max_drawdown_pct": 2.0, "buy_hold_return_pct": None, "cycles": 1, "fills": 2},
            {"return_pct": -3.0, "max_drawdown_pct": 4.0, "buy_hold_return_pct": 1.5, "cycles": 1, "fills": 3},
        ]
        summary = summarize_test_windows(windows)
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["up_market_windows"], 1)
        self.assertEqual(summary["down_market_windows"], 0)

    def test_build_venue_evaluator_hyperliquid_spot(self):
        from offline.runners.walk_forward_compare import _build_venue_evaluator
        source_report = {
            "venue": "hyperliquid",
            "runtime": {"engine": "spot", "coin": "HYPE"},
            "strategy_params": {
                "pair": "HYPEUSDC",
                "entry_amount": 50.0,
                "takeprofit_pct": 5.0,
                "tp_mdhtp_enabled": True,
            },
            "fee_pct_per_leg": 0.035,
        }
        base_params, base_fee, segment_fn = _build_venue_evaluator("hyperliquid", source_report)
        self.assertEqual(base_fee, 0.035)
        self.assertEqual(base_params.currency, "USDC")
        self.assertTrue(base_params.tp_mdhtp_enabled)
        self.assertTrue(callable(segment_fn))

    def test_runners_and_backtest_broker_guards(self):
        from offline.backtests.tradeall import BacktestBroker, run_backtest
        from offline.runners.t212_walk_forward_baseline import periods_per_year
        from offline.runners.kraken_financial_benchmark import _format as kfb_fmt
        from offline.runners.kraken_financial_compare import _fmt as kfc_fmt
        from offline.runners.calibrate_execution_audit import _fmt as cea_fmt
        import tempfile

        for fn in (kfb_fmt, kfc_fmt, cea_fmt):
            self.assertEqual(fn(None), "n/a")
            self.assertEqual(fn(True), "n/a")
            self.assertEqual(fn(float("nan")), "n/a")
            self.assertEqual(fn(float("inf")), "n/a")

        with self.assertRaises(ValueError):
            periods_per_year(0)
        with self.assertRaises(ValueError):
            periods_per_year(False)

        with tempfile.TemporaryDirectory() as tmp:
            broker = BacktestBroker(tmp, lambda: 1000.0)
            self.assertIsNone(broker._execute("BUY", "BTCUSDC", 0.0))
            self.assertIsNone(broker._execute("BUY", "BTCUSDC", float("nan")))
            self.assertIsNone(broker._execute("BUY", "BTCUSDC", 100.0, qty=0.0))
            self.assertIsNone(broker._execute("BUY", "BTCUSDC", True))

        with self.assertRaises(ValueError):
            run_backtest("BTCUSDC", 0.0, 1.0, "fast", "", "history")
        with self.assertRaises(ValueError):
            run_backtest("BTCUSDC", 0.0, 1.0, "fast", "..", "history")


if __name__ == "__main__":
    unittest.main()
