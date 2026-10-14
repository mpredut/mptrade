import unittest

from offline.backtests.financial_benchmark import (
    aggregate_financial_windows,
    default_scenarios,
)


def _window(key, strategy_return, buy_hold, drawdown, net_pnl):
    return {
        "key": key,
        "bars": 90,
        "return_pct": strategy_return,
        "max_drawdown_pct": drawdown,
        "buy_hold_return_pct": buy_hold,
        "cycles": 1,
        "fills": 3,
        "metrics": {
            "net_pnl": net_pnl,
            "max_drawdown_abs": drawdown * 10,
            "sortino": 1.5,
            "calmar": 2.0,
            "profit_factor": 1.8,
            "expectancy": net_pnl,
            "cvar_95_pct": -1.2,
            "exposure_pct": 50.0,
            "turnover_pct": 80.0,
            "trade_count": 1,
            "max_underwater_periods": 4,
        },
    }


class FinancialBenchmarkTest(unittest.TestCase):
    def test_default_scenarios_are_explicit_and_stress_is_more_adverse(self):
        central, stress = default_scenarios()
        self.assertEqual((central.name, stress.name), ("central", "stress"))
        self.assertGreater(stress.fees.limit_fee_pct, central.fees.limit_fee_pct)
        self.assertGreater(stress.fees.market_fee_pct, central.fees.market_fee_pct)
        self.assertGreater(stress.execution.spread_bps, central.execution.spread_bps)
        self.assertLess(stress.execution.partial_fill_ratio,
                        central.execution.partial_fill_ratio)
        self.assertFalse(central.calibrated)

    def test_aggregate_reports_usd_risk_and_bull_bear_sideways_separately(self):
        windows = [
            _window("bull", 2.0, 5.0, 1.0, 78.0),
            _window("bear", -3.0, -8.0, 4.0, -117.0),
            _window("sideways", 1.0, 1.0, 2.0, 39.0),
        ]

        result = aggregate_financial_windows(windows, initial_capital=3900.0)

        self.assertEqual(result["window_count"], 3)
        self.assertEqual(result["sum_reset_net_pnl_usd"], 0.0)
        self.assertEqual(result["mean_net_pnl_usd_per_window"], 0.0)
        self.assertEqual(result["mean_buy_hold_return_pct"], -2.0 / 3.0)
        self.assertEqual(result["worst_return_pct"], -3.0)
        self.assertEqual(result["worst_max_drawdown_pct"], 4.0)
        self.assertEqual(result["regimes"]["bull"]["windows"], 1)
        self.assertEqual(result["regimes"]["bear"]["windows"], 1)
        self.assertEqual(result["regimes"]["sideways"]["windows"], 1)
        self.assertEqual(result["total_test_bars"], 270)

    def test_missing_financial_fields_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aggregate_financial_windows([{"key": "x"}], initial_capital=3900.0)

    def test_multi_venue_default_scenarios(self):
        for venue in ("binance", "hyperliquid", "t212", "kraken"):
            central, stress = default_scenarios(venue)
            self.assertEqual((central.name, stress.name), ("central", "stress"))
            self.assertGreater(stress.fees.market_fee_pct, central.fees.limit_fee_pct)
            self.assertGreater(stress.execution.spread_bps, central.execution.spread_bps)
            self.assertLessEqual(stress.execution.partial_fill_ratio, central.execution.partial_fill_ratio)


if __name__ == "__main__":
    unittest.main()
