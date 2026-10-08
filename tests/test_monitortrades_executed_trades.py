import unittest
from unittest.mock import patch, MagicMock

import monitortrades as mt


class TestMonitortradesExecutedTrades(unittest.TestCase):
    def setUp(self):
        mt.reset_trades_monitor_state_for_test()

    def test_warmup_does_not_notify_historical_trades(self):
        historical_trades = [
            {
                "symbol": "BTCUSDC",
                "id": "1001",
                "orderId": "50001",
                "price": 85000.0,
                "qty": 0.1,
                "time": 1700000000000,
                "isBuyer": True,
            },
            {
                "symbol": "BTCUSDC",
                "id": "1002",
                "orderId": "50002",
                "price": 86000.0,
                "qty": 0.05,
                "time": 1700000010000,
                "isBuyer": False,
            },
        ]
        with patch.object(mt.apitrades, "get_trade_orders", return_value=historical_trades), \
             patch("alertnotifiers.notify") as mock_notify:
            mt.check_and_notify_executed_trades(symbols_list=["BTCUSDC"])
            mock_notify.assert_not_called()

    def test_new_trade_triggers_notification_and_deduplicates(self):
        initial_trades = [
            {
                "symbol": "TAOUSDC",
                "id": "2001",
                "orderId": "60001",
                "price": 280.0,
                "qty": 1.0,
                "time": 1700000000000,
                "isBuyer": True,
            }
        ]
        new_trade = {
            "symbol": "TAOUSDC",
            "id": "2002",
            "orderId": "707115262",
            "price": 283.92,
            "qty": 0.5,
            "time": 1700000050000,
            "isBuyer": True,
        }

        with patch.object(mt.apitrades, "get_trade_orders", side_effect=[initial_trades, initial_trades + [new_trade], initial_trades + [new_trade]]), \
             patch("alertnotifiers.notify") as mock_notify:
            # 1. Warm-up
            mt.check_and_notify_executed_trades(symbols_list=["TAOUSDC"])
            mock_notify.assert_not_called()

            # 2. Cycle with new trade
            mt.check_and_notify_executed_trades(symbols_list=["TAOUSDC"])
            mock_notify.assert_called_once_with(
                title="🎉 BUY TAOUSDC @ 283.92",
                body="Order 707115262 executed! Filled: 0.5000 TAOUSDC (~141.96 USDC)",
                source="monitortrades",
                symbol="TAOUSDC",
                price=283.92,
            )

            # 3. Subsequent cycle with same trade -> No duplicate notification
            mock_notify.reset_mock()
            mt.check_and_notify_executed_trades(symbols_list=["TAOUSDC"])
            mock_notify.assert_not_called()

    def test_partial_fills_grouped_by_order_id(self):
        initial_trades = []
        partial_fills = [
            {
                "symbol": "BTCUSDC",
                "id": "3001",
                "orderId": "88888",
                "price": 80000.0,
                "qty": 0.4,
                "time": 1700000001000,
                "isBuyer": False,
            },
            {
                "symbol": "BTCUSDC",
                "id": "3002",
                "orderId": "88888",
                "price": 80100.0,
                "qty": 0.6,
                "time": 1700000002000,
                "isBuyer": False,
            },
        ]
        with patch.object(mt.apitrades, "get_trade_orders", side_effect=[initial_trades, partial_fills]), \
             patch("alertnotifiers.notify") as mock_notify:
            # Warm-up with empty
            mt.check_and_notify_executed_trades(symbols_list=["BTCUSDC"])
            mock_notify.assert_not_called()

            # New batch with 2 partial fills for same order
            mt.check_and_notify_executed_trades(symbols_list=["BTCUSDC"])
            # Total qty = 1.0, weighted avg price = (0.4*80000 + 0.6*80100)/1.0 = 80060.0
            mock_notify.assert_called_once_with(
                title="🎉 SELL BTCUSDC @ 80060.00",
                body="Order 88888 executed! Filled: 1.0000 BTCUSDC (~80060.00 USDC) (2 fills)",
                source="monitortrades",
                symbol="BTCUSDC",
                price=80060.0,
            )

    def test_api_error_handled_gracefully(self):
        with patch.object(mt.apitrades, "get_trade_orders", side_effect=RuntimeError("API timeout")), \
             patch("alertnotifiers.notify") as mock_notify:
            # Should not raise
            mt.check_and_notify_executed_trades(symbols_list=["BTCUSDC"])
            mock_notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
