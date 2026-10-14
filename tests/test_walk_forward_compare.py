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


if __name__ == "__main__":
    unittest.main()
