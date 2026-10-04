"""Market-regime context wiring for the legacy Binance safety wrapper."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from binance_api import bapi_placeorder as placeorder
from market_regime import MarketRegimeContext, MarketRegimeDecision


SYMBOL = "BTCUSDC"


class SafeOrderMarketRegimeBoundaryTest(unittest.TestCase):
    @staticmethod
    def _fresh_context():
        decision = MarketRegimeDecision(
            regime="bull",
            gradient=0.5,
            epsilon=0.1,
            strength=5.0,
            fresh=True,
            reason="directional_signal",
        )
        return MarketRegimeContext.from_decision(
            decision, evaluated_at=1_700_000_000.0,
        )

    def _exercise_buy(self, *, regime_context=None, pass_context=False):
        resolved_context = self._fresh_context()
        kwargs = {"regime_context": regime_context} if pass_context else {}
        with (
            mock.patch.object(
                placeorder.order_guard,
                "daily_limit_guard",
                return_value=(True, None),
            ),
            mock.patch.object(
                placeorder.api, "get_current_price", return_value=100.0,
            ),
            mock.patch(
                "binance_api.bapi_allorders.get_trade_orders", return_value=[],
            ),
            mock.patch.object(
                placeorder.order_guard, "window_for", return_value=8 * 3600.0,
            ) as window_for,
            mock.patch.object(
                placeorder.order_guard, "margin_for", return_value=1.15,
            ),
            mock.patch.object(
                placeorder.order_guard, "profit_guard", return_value=True,
            ) as profit_guard,
            mock.patch(
                "providers.market_api.api.market_regime_context",
                return_value=resolved_context,
            ) as resolve_context,
        ):
            result = placeorder.if_place_safe_order(
                "BUY", SYMBOL, 100.0, 1.0, 48 * 3600 + 60, **kwargs,
            )

        return (
            result,
            resolved_context,
            resolve_context,
            window_for,
            profit_guard,
        )

    def test_explicit_context_is_reused_without_runtime_resolution(self):
        explicit_context = self._fresh_context()
        result, _, resolve_context, window_for, profit_guard = self._exercise_buy(
            regime_context=explicit_context,
            pass_context=True,
        )

        self.assertEqual(result, (True, None))
        resolve_context.assert_not_called()
        self.assertIs(
            window_for.call_args.kwargs["regime_context"], explicit_context,
        )
        self.assertIs(
            profit_guard.call_args.kwargs["regime_context"], explicit_context,
        )

    def test_missing_context_is_resolved_once_at_market_api_boundary(self):
        result, resolved_context, resolve_context, window_for, profit_guard = (
            self._exercise_buy()
        )

        self.assertEqual(result, (True, None))
        self.assertTrue(resolved_context.fresh)
        resolve_context.assert_called_once_with(
            SYMBOL, provider_name="binance",
        )
        self.assertIs(
            window_for.call_args.kwargs["regime_context"], resolved_context,
        )
        self.assertIs(
            profit_guard.call_args.kwargs["regime_context"], resolved_context,
        )


if __name__ == "__main__":
    unittest.main()
