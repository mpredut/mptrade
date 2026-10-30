#!/usr/bin/env python3
"""tests/test_order_rollback.py — Unit tests for unified order rollback and restoration."""

import unittest
from order_rollback import (
    extract_order_restore_params,
    restore_cancelled_orders,
    AtomicOrderRollbackGuard,
)


class TestOrderRollback(unittest.TestCase):
    def test_extract_order_restore_params_binance_and_kraken(self):
        # Binance format
        b_ord = {
            "orderId": "12345",
            "side": "SELL",
            "price": "150.25",
            "origQty": "2.5",
        }
        params = extract_order_restore_params(b_ord)
        self.assertEqual(params["side"], "SELL")
        self.assertEqual(params["limit"], 150.25)
        self.assertEqual(params["qty"], 2.5)
        self.assertEqual(params["kind"], "TP")

        # Kraken format
        k_ord = {
            "id": "OG-KRAKEN",
            "side": "sell",
            "price": 105.0,
            "vol": 10.0,
        }
        params_k = extract_order_restore_params(k_ord)
        self.assertEqual(params_k["side"], "SELL")
        self.assertEqual(params_k["limit"], 105.0)
        self.assertEqual(params_k["qty"], 10.0)

    def test_extract_order_restore_params_t212(self):
        t_ord = {
            "id": "58805932163",
            "side": "SELL",
            "qty": 0.19,
            "limit": 176.31,
            "level": 11.0,
            "kind": "TP",
        }
        params = extract_order_restore_params(t_ord)
        self.assertEqual(params["side"], "SELL")
        self.assertEqual(params["limit"], 176.31)
        self.assertEqual(params["qty"], 0.19)
        self.assertEqual(params["level"], 11.0)
        self.assertEqual(params["kind"], "TP")

    def test_restore_cancelled_orders_skips_invalid_and_calls_restore(self):
        cancelled = [
            {"side": "SELL", "limit": 100.0, "qty": 1.0},
            {"side": "SELL", "limit": 0.0, "qty": 1.0},  # invalid limit
            {"side": "SELL", "limit": 120.0, "qty": 0.0},  # invalid qty
            {"side": "BUY", "limit": 90.0, "qty": 2.0, "kind": "ENTRY"},
        ]
        restored_calls = []

        def mock_restore(p):
            restored_calls.append(p)
            return True

        logs = []
        count = restore_cancelled_orders(
            cancelled,
            restore_fn=mock_restore,
            log_fn=logs.append,
            reason="test run",
        )
        self.assertEqual(count, 2)
        self.assertEqual(len(restored_calls), 2)
        self.assertEqual(restored_calls[0]["side"], "SELL")
        self.assertEqual(restored_calls[0]["limit"], 100.0)
        self.assertEqual(restored_calls[1]["side"], "BUY")
        self.assertEqual(restored_calls[1]["limit"], 90.0)

    def test_restore_handles_callback_exceptions_gracefully(self):
        cancelled = [
            {"side": "SELL", "limit": 100.0, "qty": 1.0},
            {"side": "SELL", "limit": 110.0, "qty": 1.0},
        ]
        calls = []

        def failing_restore(p):
            calls.append(p)
            if len(calls) == 1:
                raise RuntimeError("venue connection dropped")
            return True

        logs = []
        count = restore_cancelled_orders(
            cancelled,
            restore_fn=failing_restore,
            log_fn=logs.append,
            reason="test failure",
        )
        self.assertEqual(count, 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("failed restoring order" in l for l in logs))

    def test_atomic_rollback_guard_committed(self):
        cancelled = [{"side": "SELL", "limit": 100.0, "qty": 1.0}]
        restored = []

        with AtomicOrderRollbackGuard(cancelled, restore_fn=restored.append) as guard:
            # Placement succeeds
            guard.commit()

        self.assertEqual(len(restored), 0)

    def test_atomic_rollback_guard_aborted_triggers_rollback(self):
        cancelled = [{"side": "SELL", "limit": 100.0, "qty": 1.0}]
        restored = []

        try:
            with AtomicOrderRollbackGuard(cancelled, restore_fn=restored.append, reason="crash"):
                # Simulated failure / exception before commit
                raise ValueError("placement exploded")
        except ValueError:
            pass

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["limit"], 100.0)

    def test_extract_order_restore_params_prefers_remaining_qty(self):
        b_ord = {
            "orderId": "999",
            "side": "SELL",
            "price": "100.0",
            "origQty": "10.0",
            "remainingQty": "3.5",
        }
        params = extract_order_restore_params(b_ord)
        self.assertEqual(params["qty"], 3.5)

    def test_atomic_rollback_guard_explicit_rollback_not_duplicated_on_exit(self):
        cancelled = [{"side": "SELL", "limit": 100.0, "qty": 1.0}]
        restored = []

        try:
            with AtomicOrderRollbackGuard(cancelled, restore_fn=restored.append, reason="explicit") as guard:
                guard.rollback()
                raise RuntimeError("abort after explicit rollback")
        except RuntimeError:
            pass

        self.assertEqual(len(restored), 1)


if __name__ == "__main__":
    unittest.main()
