import unittest

from offline.backtests.execution_calibration import calibrate_execution_events


class ExecutionCalibrationTest(unittest.TestCase):
    def test_correlates_fees_latency_partial_fills_and_limit_deviation(self):
        events = [
            {
                "ts": 10, "event": "submit_requested", "intent_id": "limit-1",
                "venue": "Kraken", "symbol": "HYPEUSD", "side": "buy",
                "qty": 10, "price": 100, "market": False,
            },
            {"ts": 11, "event": "submit_accepted", "intent_id": "limit-1"},
            {
                "ts": 14, "event": "order_status", "intent_id": "limit-1",
                "status": "open", "filled_qty": 4, "cost": 396, "fee": 0.396,
            },
            {
                "ts": 20, "event": "order_status", "intent_id": "limit-1",
                "status": "closed", "filled_qty": 10, "cost": 990, "fee": 0.99,
            },
            {
                "ts": 21, "event": "order_status", "intent_id": "limit-1",
                "status": "settled", "filled_qty": 10, "cost": 990, "fee": 1.98,
            },
            {
                "ts": 30, "event": "submit_requested", "intent_id": "market-1",
                "venue": "Kraken", "symbol": "HYPEUSD", "side": "sell",
                "qty": 5, "price": None, "reference_price": 102,
                "market": True,
            },
            {"ts": 31, "event": "submit_accepted", "intent_id": "market-1"},
            {
                "ts": 32, "event": "order_status", "intent_id": "market-1",
                "status": "closed", "filled_qty": 5, "cost": 500, "fee": 1.3,
            },
        ]

        report = calibrate_execution_events(events)

        self.assertEqual(report["summary"]["all"]["orders"], 2)
        self.assertEqual(report["summary"]["all"]["filled"], 2)
        self.assertEqual(report["summary"]["limit"]["ever_partial"], 1)
        self.assertAlmostEqual(report["summary"]["limit"]["fee_bps"]["p50"], 20.0)
        self.assertAlmostEqual(
            report["summary"]["market"]["fee_bps"]["p50"], 26.0,
        )
        self.assertEqual(
            report["summary"]["limit"]["first_fill_latency_s"]["p50"], 3.0,
        )
        self.assertLess(
            report["summary"]["limit"]["limit_fill_deviation_bps"]["p50"], 0,
        )
        self.assertAlmostEqual(
            report["summary"]["market"]
            ["market_execution_shortfall_bps"]["p50"],
            (1.0 - 100.0 / 102.0) * 10_000,
        )
        self.assertTrue(
            report["calibration_readiness"]
            ["has_market_execution_shortfall_samples"],
        )
        self.assertFalse(
            report["calibration_readiness"]["can_calibrate_market_slippage"],
        )

    def test_validates_first_natural_client_order_id_without_placing_orders(self):
        events = [
            {
                "ts": 10, "event": "submit_requested", "intent_id": "k-1",
                "venue": "Kraken", "symbol": "HYPEUSD", "qty": 1,
                "client_order_id": "0123456789abcdef0123456789abcdef",
            },
            {
                "ts": 11, "event": "submit_accepted", "intent_id": "k-1",
                "venue": "Kraken", "symbol": "HYPEUSD", "order_id": "K-7",
                "client_order_id": "0123456789abcdef0123456789abcdef",
            },
            {
                "ts": 20, "event": "submit_requested", "intent_id": "b-1",
                "venue": "Binance", "symbol": "BTCUSDC", "qty": 1,
                "client_order_id": "gresit",
            },
            {
                "ts": 21, "event": "submit_accepted", "intent_id": "b-1",
                "order_id": "B-7", "client_order_id": "gresit",
            },
            {
                "ts": 30, "event": "submit_requested", "intent_id": "t-1",
                "venue": "T212", "symbol": "NVDA_US_EQ", "qty": 1,
            },
            {
                "ts": 31, "event": "submit_accepted", "intent_id": "t-1",
                "order_id": "T-7",
            },
        ]

        validation = calibrate_execution_events(events)[
            "client_order_id_validation"
        ]

        self.assertEqual(validation["supported_accepted_orders"], 2)
        self.assertEqual(validation["with_client_order_id"], 2)
        self.assertEqual(validation["valid_client_order_ids"], 1)
        self.assertEqual(validation["invalid_client_order_ids"], 1)
        self.assertEqual(validation["missing_client_order_ids"], 0)
        self.assertEqual(
            validation["first_valid_by_venue"]["Kraken"]["order_id"], "K-7",
        )
        self.assertNotIn("T212", validation["first_valid_by_venue"])

    def test_correlates_order_terminal_and_order_terminal_retry_fills(self):
        events = [
            {
                "ts": 100, "event": "submit_requested", "intent_id": "outbox-1",
                "venue": "binance", "symbol": "BTCUSDC", "side": "buy",
                "qty": 1.0, "price": 63000.0, "market": False,
            },
            {"ts": 101, "event": "submit_accepted", "intent_id": "outbox-1", "order_id": "B-1"},
            {
                "ts": 105, "event": "order_terminal", "intent_id": "outbox-1",
                "status": "closed", "venue_status": "FILLED",
                "filled_qty": 1.0, "cost": 63000.0, "fee": 1.0,
            },
            {
                "ts": 200, "event": "submit_requested", "intent_id": "outbox-2",
                "venue": "binance", "symbol": "BTCUSDC", "side": "buy",
                "qty": 1.0, "price": 100.0, "market": False,
            },
            {"ts": 201, "event": "submit_accepted", "intent_id": "outbox-2", "order_id": "B-2"},
            {
                "ts": 203, "event": "order_terminal_retry", "intent_id": "outbox-2",
                "status": "expired", "venue_status": "EXPIRED",
                "filled_qty": 0.4, "cost": 40.0, "fee": 0.02,
                "remaining_qty": 0.6,
            },
            {
                "ts": 300, "event": "submit_requested", "intent_id": "outbox-3",
                "venue": "binance", "symbol": "BTCUSDC", "side": "sell",
                "qty": 1.0, "price": 100.0, "market": False,
            },
            {
                "ts": 301, "event": "submit_refused", "intent_id": "outbox-3",
                "reason": "guard refusal",
            },
        ]

        report = calibrate_execution_events(events)

        self.assertEqual(report["summary"]["all"]["orders"], 3)
        self.assertEqual(report["summary"]["all"]["filled"], 2)
        self.assertEqual(report["summary"]["all"]["rejected"], 1)
        self.assertEqual(report["summary"]["limit"]["filled"], 2)
        self.assertEqual(report["summary"]["limit"]["ever_partial"], 1)
        self.assertAlmostEqual(report["summary"]["all"]["fee_bps"]["p50"], (1.0 / 63000.0 * 10000 + 0.02 / 40.0 * 10000) / 2.0, places=3)
        self.assertEqual(report["summary"]["all"]["first_fill_latency_s"]["min"], 2.0)
        self.assertEqual(report["summary"]["all"]["first_fill_latency_s"]["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
