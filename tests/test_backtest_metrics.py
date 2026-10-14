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


if __name__ == "__main__":
    unittest.main()
