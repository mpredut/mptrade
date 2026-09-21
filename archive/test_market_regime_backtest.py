import unittest

from market_regime import MarketRegimeEvaluator, MarketRegimeService
from market_regime_backtest import evaluate_forward_classifications


class MarketRegimeBacktestTest(unittest.TestCase):
    def test_walk_forward_metrics_are_deterministic(self):
        evaluator = MarketRegimeEvaluator()
        service = MarketRegimeService()
        bull = evaluator.evaluate({"gradient_recent": 0.6, "epsilon": 0.1})
        bear = evaluator.evaluate({"gradient_recent": -0.6, "epsilon": 0.1})
        up = service.compose(bull, bull)
        down = service.compose(bear, bear)
        report = evaluate_forward_classifications([
            (up, 0.03), (down, -0.02), (up, -0.01),
        ])
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["directional"], 3)
        self.assertAlmostEqual(report["directional_accuracy"], 2 / 3)
        self.assertAlmostEqual(report["mean_signed_forward_return"], 0.04 / 3)

    def test_backtest_rejects_nonfinite_forward_return(self):
        evaluator = MarketRegimeEvaluator()
        bull = evaluator.evaluate({"gradient_recent": 0.6, "epsilon": 0.1})
        decision = MarketRegimeService().compose(bull, bull)
        with self.assertRaises(ValueError):
            evaluate_forward_classifications([(decision, float("nan"))])


if __name__ == "__main__":
    unittest.main()
