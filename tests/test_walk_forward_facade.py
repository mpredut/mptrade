#!/usr/bin/env python3
"""Tests for the canonical unified walk-forward runner dispatcher facade."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from offline.runners.walk_forward import (
    canonical_venue,
    dispatch,
    main,
    run_baseline,
    run_compare,
    SUPPORTED_VENUES,
    SUPPORTED_MODES,
    VENUE_RUNNER_MODULES,
)


class TestWalkForwardFacade(unittest.TestCase):
    def test_canonical_venue_normalization(self):
        self.assertEqual(canonical_venue("kraken"), "kraken")
        self.assertEqual(canonical_venue("Kraken"), "kraken")
        self.assertEqual(canonical_venue("  binance  "), "binance")
        self.assertEqual(canonical_venue("hyperliquid"), "hyperliquid")
        self.assertEqual(canonical_venue("t212"), "t212")
        self.assertEqual(canonical_venue("trading212"), "t212")
        self.assertEqual(canonical_venue("Trading212"), "t212")
        self.assertIsNone(canonical_venue(None))

    def test_supported_registries(self):
        for v in ("kraken", "binance", "hyperliquid", "t212", "trading212"):
            self.assertIn(v, SUPPORTED_VENUES)
            self.assertIn(v, VENUE_RUNNER_MODULES)
        for m in ("baseline", "compare"):
            self.assertIn(m, SUPPORTED_MODES)

    def test_run_baseline_unsupported_venue(self):
        with self.assertRaises(ValueError) as ctx:
            run_baseline("unknown_venue", [])
        self.assertIn("Unsupported baseline venue", str(ctx.exception))

    @patch("offline.runners.binance_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_binance(self, mock_binance_main):
        ret = run_baseline("binance", ["--symbol", "BTCUSDT", "--fee", "0.08"])
        self.assertEqual(ret, 0)
        mock_binance_main.assert_called_once_with(["--symbol", "BTCUSDT", "--fee", "0.08"])

    @patch("offline.runners.kraken_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_kraken(self, mock_kraken_main):
        ret = run_baseline("kraken", ["--pair", "XBT/EUR", "--fee", "0.26"])
        self.assertEqual(ret, 0)
        mock_kraken_main.assert_called_once_with(["--pair", "XBT/EUR", "--fee", "0.26"])

    @patch("offline.runners.hyperliquid_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_hyperliquid(self, mock_hl_main):
        ret = run_baseline("hyperliquid", ["--coin", "HYPE", "--intervals", "240"])
        self.assertEqual(ret, 0)
        mock_hl_main.assert_called_once_with(["--coin", "HYPE", "--intervals", "240"])

    @patch("offline.runners.t212_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_t212(self, mock_t212_main):
        ret = run_baseline("t212", ["--profile", "spcx", "--interval", "5m"])
        self.assertEqual(ret, 0)
        mock_t212_main.assert_called_once_with(["--profile", "spcx", "--interval", "5m"])

    @patch("offline.runners.t212_walk_forward_baseline.main", return_value=0)
    def test_run_baseline_trading212_alias(self, mock_t212_main):
        ret = run_baseline("trading212", ["--profile", "nvda"])
        self.assertEqual(ret, 0)
        mock_t212_main.assert_called_once_with(["--profile", "nvda"])

    @patch("offline.runners.walk_forward_compare.main", return_value=0)
    def test_run_compare(self, mock_compare_main):
        ret = run_compare(["sample_report.json", "--fee-stress", "0.1,0.2"])
        self.assertEqual(ret, 0)
        mock_compare_main.assert_called_once_with(["sample_report.json", "--fee-stress", "0.1,0.2"])

    @patch("offline.runners.walk_forward_compare.main", return_value=0)
    def test_run_compare_with_venue_override(self, mock_compare_main):
        ret = run_compare(["sample_report.json"], venue="trading212")
        self.assertEqual(ret, 0)
        mock_compare_main.assert_called_once_with(["sample_report.json", "--venue", "t212"])

    @patch("offline.runners.binance_walk_forward_baseline.main", return_value=0)
    def test_dispatch_baseline_mode(self, mock_binance_main):
        code = dispatch(["--venue", "binance", "--symbol", "ETHUSDT"])
        self.assertEqual(code, 0)
        mock_binance_main.assert_called_once_with(["--symbol", "ETHUSDT"])

    @patch("offline.runners.walk_forward_compare.main", return_value=0)
    def test_dispatch_compare_mode(self, mock_compare_main):
        code = dispatch(["--mode", "compare", "report.json", "--candidate-set", "standard"])
        self.assertEqual(code, 0)
        mock_compare_main.assert_called_once_with(["report.json", "--candidate-set", "standard"])

    def test_dispatch_baseline_requires_venue(self):
        with self.assertRaises(SystemExit):
            dispatch(["--mode", "baseline", "--symbol", "ETHUSDT"])

    def test_dispatch_top_level_help(self):
        code = dispatch(["--help"])
        self.assertEqual(code, 0)

    @patch("offline.runners.walk_forward.dispatch", return_value=42)
    def test_main_entrypoint(self, mock_dispatch):
        res = main(["--venue", "kraken"])
        self.assertEqual(res, 42)
        mock_dispatch.assert_called_once_with(["--venue", "kraken"])


if __name__ == "__main__":
    unittest.main()
