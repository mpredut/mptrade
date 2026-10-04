"""Unit test for offline intelligence backtest engine across all 4 pillars."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from offline.research.intelligence_backtest import BacktestStats, run_intelligence_backtest


class TestIntelligenceBacktest(unittest.TestCase):
    """Verifies backtest simulation engine and guard statistics across all 4 pillars."""

    def test_run_intelligence_backtest_synthetic_series(self):
        # Generate 60 synthetic bars stepped by 300 seconds
        synthetic_series = []
        base_t = 1_700_000_000.0
        p = 100.0
        for i in range(60):
            # Upward momentum followed by a pullback
            if i < 30:
                p += 0.5
            elif i < 45:
                p -= 0.8
            else:
                p += 0.3
            synthetic_series.append((base_t + (i * 300), p))

        with patch("offline.research.intelligence_backtest.load_price_series", return_value=synthetic_series):
            res = run_intelligence_backtest(symbol="TESTPAIR", cache_path="/dummy/path")

        self.assertIn("BASELINE_UNGUARDED", res)
        self.assertIn("PILLAR1_GUARDED", res)
        self.assertIn("PILLAR1_PLUS_PILLAR2", res)
        self.assertIn("FULL_COMPOSITE", res)

        for mode, stats in res.items():
            self.assertIsInstance(stats, BacktestStats)
            self.assertEqual(stats.symbol, "TESTPAIR")
            self.assertGreaterEqual(stats.initial_balance, 1000.0)
            self.assertGreaterEqual(stats.final_balance, 0.0)
            self.assertGreaterEqual(stats.win_rate_pct, 0.0)
            self.assertLessEqual(stats.win_rate_pct, 100.0)
            self.assertGreaterEqual(stats.max_drawdown_pct, 0.0)
            self.assertLessEqual(stats.max_drawdown_pct, 100.0)

    def test_backtest_stats_fields_complete(self):
        stats = BacktestStats(
            symbol="BTCUSDC",
            mode="FULL_COMPOSITE",
            initial_balance=10000.0,
            final_balance=10500.0,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate_pct=70.0,
            net_profit_usd=500.0,
            net_return_pct=5.0,
            max_drawdown_pct=2.5,
            profit_factor=2.1,
            parabolic_vetoes=1,
            weibull_downscales=2,
            noise_vetoes=1,
            whale_vetoes=1,
            orderbook_vetoes=1,
            cascade_vetoes=1,
            funding_downscales=1,
            geopolitical_vetoes=1,
            sentiment_downscales=1,
        )
        self.assertEqual(stats.symbol, "BTCUSDC")
        self.assertEqual(stats.mode, "FULL_COMPOSITE")
        self.assertEqual(stats.geopolitical_vetoes, 1)
        self.assertEqual(stats.funding_downscales, 1)
