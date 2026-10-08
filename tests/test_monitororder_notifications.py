import unittest
from unittest.mock import patch, MagicMock

import monitororder


class TestMonitororderNotifications(unittest.TestCase):
    def setUp(self):
        monitororder.reset_monitororder_state_for_test()

    def tearDown(self):
        monitororder.reset_monitororder_state_for_test()

    def test_warmup_does_not_notify_historical_open_orders(self):
        open_orders = {
            1001: {"price": 80000.0, "quantity": 0.1, "remainingQty": 0.1},
            1002: {"price": 81000.0, "quantity": 0.2, "remainingQty": 0.2},
        }
        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch("alertnotifiers.notify") as mock_notify:
            monitororder.warmup_open_orders()
            mock_notify.assert_not_called()
            # Verify orders are marked as notified
            self.assertIn("1001", monitororder._notified_placed_order_ids)
            self.assertIn("1002", monitororder._notified_placed_order_ids)

    def test_newly_launched_order_triggers_notification_and_deduplicates(self):
        # 1. Warm up with empty orders
        with patch.object(monitororder.api, "get_open_orders", return_value={}):
            monitororder.warmup_open_orders()

        # 2. A new order appears
        open_orders = {
            2001: {"price": 283.92, "quantity": 0.5, "remainingQty": 0.5}
        }
        with patch.object(monitororder.api, "get_open_orders", return_value=open_orders), \
             patch.object(monitororder.api, "get_current_price", return_value=250.0), \
             patch("alertnotifiers.notify") as mock_notify:
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_notify.assert_called_once_with(
                title="📝 BUY TAOUSDC @ 283.92",
                body="Order 2001 placed on venue. Qty: 0.5000 TAOUSDC (~141.96 USDC)",
                source="monitororder",
                symbol="TAOUSDC",
                price=283.92,
            )

            # 3. Next cycle with same order -> deduplicated, no second notification
            mock_notify.reset_mock()
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_notify.assert_not_called()

    def test_derived_replacement_order_does_not_trigger_notification(self):
        # 1. Warm up
        with patch.object(monitororder.api, "get_open_orders", return_value={}):
            monitororder.warmup_open_orders()

        # 2. An initial order appears and is close to market price -> replaced
        initial_order = {
            3001: {"price": 100.0, "quantity": 1.0, "remainingQty": 1.0}
        }
        replacement_order = {"orderId": 9999}

        with patch.object(monitororder.api, "get_open_orders", return_value=initial_order), \
             patch.object(monitororder.api, "get_current_price", return_value=100.1), \
             patch.object(monitororder.mkt, "preflight_order", return_value=object()), \
             patch.object(monitororder.api, "cancel_order", return_value=True), \
             patch.object(monitororder.mkt, "order_status", return_value=MagicMock(venue_status="CANCELED", filled_qty=0.0)), \
             patch.object(monitororder.mkt, "place", return_value=replacement_order), \
             patch.object(monitororder.accepted_order_persistence, "complete_accepted_claim", return_value=True), \
             patch("alertnotifiers.notify") as mock_notify:
            monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")
            # The initial order 3001 notified once when first seen
            self.assertEqual(mock_notify.call_count, 1)
            self.assertIn("3001", mock_notify.call_args[1]["body"])

            # Verify 9999 is recorded as a derived replacement
            self.assertIn("9999", monitororder._derived_replacement_order_ids)

            # 3. Next cycle: Binance returns replacement order 9999 as open
            mock_notify.reset_mock()
            open_after_replace = {
                9999: {"price": 101.2, "quantity": 1.0, "remainingQty": 1.0}
            }
            with patch.object(monitororder.api, "get_open_orders", return_value=open_after_replace), \
                 patch.object(monitororder.api, "get_current_price", return_value=90.0):  # not close, won't replace
                monitororder.monitor_open_orders_by_type("BTCUSDC", "SELL")
                # Replacement order must NOT trigger any new notification!
                mock_notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
