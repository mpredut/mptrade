"""Contractul auditului observational al StrategyExecutor."""
import json
import os
import tempfile
import unittest
from unittest import mock

import providers.execution_audit as execution_audit_module
from order_retry import StrategyExecutorLifecycleApi

from providers.execution_audit import (
    AuditedStrategyExecutor,
    ExecutionAudit,
    intent_client_order_id,
)
from providers.strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    StrategyExecutor,
)


class _Executor:
    name = "TEST"

    def __init__(self):
        self.calls = []
        self.status = OrderStatus("open", 0.25, 25.0, 0.05)
        self.submit_error = None

    def get_current_price(self, symbol):
        return 100.0

    def reconciliation_capabilities(self):
        return OrderReconciliationCapabilities(True, True, True, False)

    def submit_order(self, symbol, side, qty, price=None, *, market=False, kind=None,
                     client_order_id=None):
        self.calls.append((
            "submit", symbol, side, qty, price, market, kind, client_order_id,
        ))
        if self.submit_error:
            raise self.submit_error
        return "OID-7"

    def order_status(self, symbol, order_id):
        self.calls.append(("status", symbol, order_id))
        return self.status

    def order_by_client_id(self, symbol, client_order_id):
        self.calls.append(("lookup", symbol, client_order_id))
        return {"orderId": "OID-7", "status": "open"}

    def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel", symbol, order_id))

    def pair_precision(self, symbol):
        return PairPrecision(2, 2, 0.01, symbol)

    def free_balance(self, asset):
        return 3.0

    def ohlc_closes(self, symbol, interval_min):
        return [99.0, 100.0]


class ExecutionAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.executor = _Executor()
        self.audit = ExecutionAudit(self.tmp.name, clock=lambda: 1_700_000_000.0)
        self.wrapped = AuditedStrategyExecutor(self.executor, self.audit, venue="Kraken")

    def tearDown(self):
        self.tmp.cleanup()

    def _events(self):
        paths = [os.path.join(self.tmp.name, name) for name in os.listdir(self.tmp.name)]
        rows = []
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        return rows

    def test_ohlc_series_falls_back_to_legacy_closes(self):
        series = self.wrapped.ohlc_series("ABC", 240)
        self.assertEqual(series.closes, (99.0, 100.0))
        self.assertEqual(series.interval_min, 240)
        self.assertIsNone(series.last_closed_at)

    def test_lifecycle_uses_one_intent_and_suppresses_identical_status_polls(self):
        self.assertIsInstance(self.wrapped, StrategyExecutor)
        oid = self.wrapped.submit_order_with_intent(
            "intent-fixed", "ABC", "buy", 0.25, 100.0, kind="ENTRY",
        )
        self.wrapped.order_status_with_intent("intent-fixed", "ABC", oid)
        self.wrapped.order_status_with_intent("intent-fixed", "ABC", oid)
        self.wrapped.cancel_order_with_intent("intent-fixed", "ABC", oid)

        rows = self._events()
        self.assertEqual(
            [row["event"] for row in rows],
            ["submit_requested", "submit_accepted", "order_status",
             "cancel_requested", "cancel_accepted"],
        )
        self.assertEqual({row["intent_id"] for row in rows}, {"intent-fixed"})
        self.assertEqual(rows[2]["filled_qty"], 0.25)
        client_id = intent_client_order_id("Kraken", "intent-fixed")
        self.assertEqual(rows[0]["client_order_id"], client_id)
        self.assertEqual(self.executor.calls[0][-1], client_id)

    def test_correlation_caches_are_lru_bounded(self):
        with mock.patch.object(execution_audit_module, "_CACHE_MAX", 2):
            for index in range(3):
                order_id = f"OID-{index}"
                self.wrapped._remember("ABC", order_id, f"intent-{index}")
                self.wrapped.order_status_with_intent(
                    f"intent-{index}", "ABC", order_id)
        self.assertEqual(len(self.wrapped._intent_by_order), 2)
        self.assertEqual(len(self.wrapped._last_status), 2)
        self.assertNotIn(("ABC", "OID-0"), self.wrapped._intent_by_order)
        self.assertNotIn(("ABC", "OID-0"), self.wrapped._last_status)

    def test_submit_error_is_audited_as_unknown_and_compatibility_raises(self):
        self.executor.submit_error = ProviderError("venue down")
        with self.assertRaisesRegex(ProviderError, "venue down"):
            self.wrapped.submit_order_with_intent(
                "intent-error", "ABC", "sell", 1.0, market=True, kind="STOP",
                reference_price=99.5,
            )
        rows = self._events()
        self.assertEqual([row["event"] for row in rows],
                         ["submit_requested", "submit_unknown"])
        self.assertEqual(rows[-1]["submission_state"], "unknown")
        self.assertIn("ProviderError: venue down", rows[-1]["reason"])
        self.assertTrue(rows[-1]["market"])
        self.assertEqual(rows[-1]["reference_price"], 99.5)
        submit = [call for call in self.executor.calls if call[0] == "submit"]
        self.assertEqual(len(submit), 1)
        self.assertEqual(submit[0][1:7], ("ABC", "sell", 1.0, None, True, "STOP"))
        self.assertEqual(
            submit[0][-1], intent_client_order_id("Kraken", "intent-error"),
        )

    def test_client_order_id_formats_preserve_full_intent_uuid(self):
        intent = "kraken-HYPEUSD-entry-0123456789abcdef0123456789abcdef"
        self.assertEqual(
            intent_client_order_id("Kraken", intent),
            "0123456789abcdef0123456789abcdef",
        )
        self.assertEqual(
            intent_client_order_id("Binance", intent),
            "SD_0123456789abcdef0123456789abcdef",
        )
        self.assertEqual(
            intent_client_order_id("Hyperliquid", intent),
            "0x0123456789abcdef0123456789abcdef",
        )
        self.assertIsNone(intent_client_order_id("T212", intent))

    def test_order_by_client_id_is_delegated(self):
        result = self.wrapped.order_by_client_id("ABC", "client-7")
        self.assertEqual(result["orderId"], "OID-7")
        self.assertEqual(self.executor.calls[-1], ("lookup", "ABC", "client-7"))

    def test_reconciliation_capabilities_survive_audit_decoration(self):
        self.assertEqual(
            self.wrapped.reconciliation_capabilities(),
            self.executor.reconciliation_capabilities(),
        )
        lifecycle_api = StrategyExecutorLifecycleApi(self.wrapped)
        self.assertEqual(
            lifecycle_api.reconciliation_capabilities(),
            self.executor.reconciliation_capabilities(),
        )

    def test_audit_ohlc_series_and_strategy_executor_validation(self):
        from providers.strategy_executor import candle_interval

        for bad_itv in (True, False, None, 0, -5, 1.5, float("nan"), float("inf"), "bad"):
            with self.assertRaises(ProviderError):
                self.wrapped.ohlc_series("ABC", bad_itv)

        series_f = self.wrapped.ohlc_series("ABC", 240.0)
        self.assertEqual(series_f.interval_min, 240)

        self.wrapped._remember(" ABC ", " OID-99 ", "intent-99")
        self.assertEqual(self.wrapped._intent("ABC", "OID-99"), "intent-99")

        pp = PairPrecision("2.0", 4.0, "0.01", " ABC ")
        self.assertEqual(pp.price_decimals, 2)
        self.assertEqual(pp.volume_decimals, 4)
        self.assertEqual(pp.base_asset, "ABC")
        with self.assertRaises(ValueError):
            PairPrecision("2.5", 4, 0.01, "ABC")

        self.assertEqual(candle_interval(60.0), "1h")
        self.assertEqual(candle_interval("60"), "1h")
        with self.assertRaises(ProviderError):
            candle_interval("60.5")

        with self.assertRaisesRegex(ValueError, "invalid orig_qty"):
            OrderStatus("open", 0.0, 0.0, 0.0, orig_qty="not_a_number")


if __name__ == "__main__":
    unittest.main()
