import unittest
from unittest.mock import patch

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

    def test_fifo_trade_id_tracker_prunes_in_strict_fifo_order(self):
        tracker = mt._FifoTradeIdTracker(maxsize=10)
        for i in range(11):
            tracker.add(f"trade_{i}")
        # When exceeding 10, oldest 6 (trade_0 through trade_5) are pruned down to 5 items (50% maxsize)
        self.assertEqual(len(tracker), 5)
        for i in range(6):
            self.assertNotIn(f"trade_{i}", tracker)
        # Most recent (trade_6 through trade_10) are strictly retained in FIFO order
        for i in range(6, 11):
            self.assertIn(f"trade_{i}", tracker)

    def test_order_timestamp_ms_normalizes_time_and_timestamp_keys(self):
        orders = [
            {"price": 100.0, "qty": 1.0, "time": 1700000001000},
            {"price": 105.0, "qty": 2.0, "timestamp": 1700000005000},
        ]
        px, ts_sec, ok = mt.get_relevant_trade(list(orders), "BUY", symbol="BTCUSDC")
        self.assertTrue(ok)
        self.assertEqual(px, 105.0)
        self.assertAlmostEqual(ts_sec, 1700000005.0, places=3)
        tot_q, tot_v, sig = mt._summarize_orders(orders)
        self.assertEqual(tot_q, 3.0)
        self.assertEqual(tot_v, 310.0)
        self.assertEqual(sig[3], 1700000005000)

    def test_execute_take_profit_honors_partial_sell_fraction(self):
        from strategies.position_exit_manager import ExitAction, ExitDecision, PositionSnapshot
        snap = PositionSnapshot(
            symbol="BTCUSDC",
            current_price=110.0,
            avg_buy_price=100.0,
            sellable_qty=2.0,
            current_time_s=1700000000.0,
        )
        decision = ExitDecision(
            action=ExitAction.PARTIAL_SELL,
            should_exit=True,
            sell_fraction=0.4,
            reason="partial_tp_test",
        )
        with patch.object(mt, "_place_guarded", return_value={"orderId": "1"}) as mock_place, \
             patch.object(mt, "_resolve_exit"):
            ok = mt._execute_take_profit(
                inst=object(),
                snapshot=snap,
                tp_decision=decision,
                sbs=60.0,
                min_qty=0.01,
                prev_tp_state={},
                gain_threshold=0.05,
            )
            self.assertTrue(ok)
            self.assertAlmostEqual(mock_place.call_args.args[3], 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
