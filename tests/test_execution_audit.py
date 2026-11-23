"""Observational audit contract for StrategyExecutor."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from intelligence.sentiment.collectors.microstructure_collector import (
    load_microstructure_cycle_history,
    save_microstructure_cycle_history,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    MicrostructureCycleDeltaEvaluator,
    evaluate_microstructure_cycle_deltas,
)
from offline.backtests.execution_calibration import calibrate_execution_events
from order_lifecycle import StrategyExecutorLifecycleApi
from providers import (
    execution_audit as execution_audit_module,
    hyperliquid_provider as hlp_mod,
    intent_execution,
    intent_state,
    order_ohlc_utils,
)
from providers.execution_audit import (
    AuditedStrategyExecutor,
    ExecutionAudit,
    _parse_bool_flag as ea_parse_bool,
    intent_client_order_id,
)
from providers.market_api import (
    MarketApi,
    _parse_bool_flag as ma_parse_bool,
    require_int_order_id,
)
from providers.strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    StrategyExecutor,
    candle_interval,
)
from tests.conftest import TempDirTestCase


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


class ExecutionAuditTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.executor = _Executor()
        self.audit = ExecutionAudit(self.tmp, clock=lambda: 1_700_000_000.0)
        self.wrapped = AuditedStrategyExecutor(self.executor, self.audit, venue="Kraken")

    def _events(self):
        paths = [os.path.join(self.tmp, name) for name in os.listdir(self.tmp)]
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
        with patch.object(execution_audit_module, "_CACHE_MAX", 2):
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




class ExecutionCalibrationTest(TempDirTestCase):
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
        self.assertAlmostEqual(report["summary"]["market"]["fee_bps"]["p50"], 26.0)
        self.assertEqual(report["summary"]["limit"]["first_fill_latency_s"]["p50"], 3.0)
        self.assertLess(report["summary"]["limit"]["limit_fill_deviation_bps"]["p50"], 0)
        self.assertAlmostEqual(
            report["summary"]["market"]["market_execution_shortfall_bps"]["p50"],
            (1.0 - 100.0 / 102.0) * 10_000,
        )
        self.assertTrue(report["calibration_readiness"]["has_market_execution_shortfall_samples"])
        self.assertFalse(report["calibration_readiness"]["can_calibrate_market_slippage"])

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

        validation = calibrate_execution_events(events)["client_order_id_validation"]

        self.assertEqual(validation["supported_accepted_orders"], 2)
        self.assertEqual(validation["with_client_order_id"], 2)
        self.assertEqual(validation["valid_client_order_ids"], 1)
        self.assertEqual(validation["invalid_client_order_ids"], 1)
        self.assertEqual(validation["missing_client_order_ids"], 0)
        self.assertEqual(validation["first_valid_by_venue"]["Kraken"]["order_id"], "K-7")
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

    def test_provider_import_cycle_cache_decoupling_and_intent_split(self):
        # 1. Fresh Python process imports providers.strategy_executor first without circular ImportError
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import providers.strategy_executor as se, providers.order_ohlc_utils as ou; "
                    "assert se.ProviderError is ou.ProviderError"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIs(ProviderError, order_ohlc_utils.ProviderError)

        # 2. MicrostructureCycleDeltaEvaluator.evaluate is pure over (snapshot, history) and never writes disk
        tmp_cdir = os.path.join(self.tmp, "cdir")
        os.makedirs(tmp_cdir, exist_ok=True)
        hist_file = os.path.join(tmp_cdir, "microstructure_cycle_history.json")
        hist = {
            "BTCUSDC": [
                {
                    "ts": 1_700_000_000.0,
                    "ob_imbalance": 0.40,
                    "cvd_15m_usd": 100_000.0,
                    "whale_long_pct": 0.50,
                    "oi_usd": 1_000_000.0,
                    "taker_buy_sell_ratio": 0.90,
                }
            ]
        }
        micro = {
            "BTCUSDC": {
                "ob_imbalance": 0.55,
                "cvd_15m_usd": 250_000.0,
                "whale_long_pct": 0.62,
                "oi_usd": 1_100_000.0,
                "taker_buy_sell_ratio": 1.15,
            }
        }
        ev_res = MicrostructureCycleDeltaEvaluator().evaluate(
            micro,
            history=hist,
            cache_dir=tmp_cdir,
            now_ts=1_700_000_400.0,
        )
        self.assertFalse(os.path.exists(hist_file))
        self.assertEqual(ev_res["BTCUSDC"]["cycle_delta"]["ob_imbalance_delta"], 0.15)
        self.assertEqual(len(hist["BTCUSDC"]), 2)

        # Producer explicitly persists and reloads history
        self.assertTrue(save_microstructure_cycle_history(tmp_cdir, hist))
        self.assertTrue(os.path.exists(hist_file))
        reloaded = load_microstructure_cycle_history(tmp_cdir)
        self.assertEqual(len(reloaded["BTCUSDC"]), 2)
        _, _, dirty_same = evaluate_microstructure_cycle_deltas(
            micro, reloaded, now_ts=1_700_000_410.0
        )
        self.assertFalse(dirty_same)

        # 3. Hyperliquid get_orders() / _fills_from_api() does not mutate SSOT cache unless publish_cache=True
        tmp_hl = os.path.join(self.tmp, "hl")
        os.makedirs(tmp_hl, exist_ok=True)
        fills_cache = os.path.join(tmp_hl, "hl_trades.json")
        prov = hlp_mod.HyperliquidProvider("HYPE")
        prov._spot_pair = "@107"
        prov._fills_cache_file = lambda: fills_cache
        fake_client = MagicMock()
        fake_client.info.user_fills.return_value = [
            {
                "coin": "@107",
                "oid": 901,
                "side": "B",
                "px": "25.0",
                "sz": "2.0",
                "time": int(time.time() * 1000) - 5000,
            }
        ]
        prov._hl = lambda: fake_client
        with patch.dict(os.environ, {"HL_ACCOUNT_ADDRESS": "0xabc"}, clear=False):
            orders = prov.get_orders("HYPEUSDC", "BUY", 3600.0)
            self.assertEqual(len(orders), 1)
            self.assertFalse(os.path.exists(fills_cache))

            # Explicit cache publication via publish_cache=True
            published = prov.get_orders("HYPEUSDC", "BUY", 3600.0, authoritative=True, publish_cache=True)
            self.assertEqual(len(published), 1)
            self.assertTrue(os.path.exists(fills_cache))

            # Authoritative read bypasses cache without mutating it unless publish_cache=True
            fake_client.info.user_fills.return_value = [
                {
                    "coin": "@107",
                    "oid": 902,
                    "side": "S",
                    "px": "26.5",
                    "sz": "1.5",
                    "time": int(time.time() * 1000) - 1000,
                }
            ]
            auth_orders = prov.get_orders("HYPEUSDC", "SELL", 3600.0, authoritative=True)
            self.assertEqual(len(auth_orders), 1)
            self.assertEqual(str(auth_orders[0].get("orderId")), "902")

        # 4. Intent state & execution split and direct botcore.clean_json_value usage
        tmp_int = os.path.join(self.tmp, "int")
        os.makedirs(tmp_int, exist_ok=True)
        intents_path = os.path.join(tmp_int, "intents.json")
        self.assertTrue(
            intent_state.save_durable_intents_state(
                intents_path,
                [{"intent_id": "i-1", "confidence": float("nan"), "status": "PENDING"}],
            )
        )
        loaded_intents = intent_state.load_durable_intents_state(intents_path)
        self.assertEqual(loaded_intents, [{"intent_id": "i-1", "confidence": None, "status": "PENDING"}])
        self.assertEqual(
            intent_execution.resolve_intent_topic_tag({"created_by": "auto_trade_executor"}, "binance"),
            "[auto_cryptotrade_executor]",
        )
        self.assertEqual(
            intent_execution.resolve_intent_topic_tag({"created_by": "trade_executor"}, "t212"),
            "[auto_sharetrade_executor]",
        )
        self.assertEqual(
            intent_execution.resolve_intent_topic_tag({"created_by": "guard_high_stake"}, "binance"),
            "[auto_guard_large_order]",
        )
        self.assertEqual(
            intent_execution.resolve_intent_topic_tag({}, "t212"),
            "[auto_sharetrade_executor]",
        )

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_executionaudit_record_strips_nan_inf_fields_strict_json_iter43_s3(tmp_path):
    """[ITER43 #3] ExecutionAudit.record strips NaN/Inf fields (strict JSON) and handles NaN/bool clock."""
    # 3. ExecutionAudit.record strips NaN/Inf fields (strict JSON) and handles NaN/bool clock
    audit = ExecutionAudit(directory=str(tmp_path), clock=lambda: float("nan"))
    ok = audit.record(
        "submit_requested",
        intent_id="id-1",
        venue="binance",
        symbol="BTCUSDC",
        price=float("nan"),
        reference_price=float("inf"),
        qty=1.25,
    )
    assert ok is True
    files = list(tmp_path.glob("execution_audit_*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["qty"] == 1.25
    assert "price" not in row
    assert "reference_price" not in row

    aud_exec = AuditedStrategyExecutor(
        SimpleNamespace(
            name="binance",
            order_by_client_id=lambda s, cid: {"orderId": 1},
        ),
        audit=audit,
    )
    with pytest.raises(ProviderError):
        aud_exec.order_by_client_id("BTCUSDC", True)
    aud_exec._remember("BTCUSDC", True, "intent-bool")
    assert ("BTCUSDC", "True") not in aud_exec._intent_by_order

def test_executionaudit_handles_out_of_range_clock_timestamps_and_iter59_s3(tmp_path):
    """[ITER59 #3] ExecutionAudit handles out-of-range clock timestamps and set/frozenset fields."""
    # 3. ExecutionAudit handles out-of-range clock timestamps and set/frozenset fields
    audit = ExecutionAudit(directory=str(tmp_path), clock=lambda: 1e30)
    assert audit.record(
        "submit",
        intent_id="id-1",
        venue="binance",
        symbol="BTCUSDC",
        tags={"a", "b"},
        nested={"nan_val": float("nan"), "ok": 1},
    ) is True

def test_execution_audit__parse_cache_max_parses_float_strings_iter84_s3(monkeypatch):
    """[ITER84 #3] execution_audit._parse_cache_max parses float strings."""
    # 3. execution_audit._parse_cache_max parses float strings
    monkeypatch.setenv("EXECUTION_AUDIT_CACHE_MAX", "2500.0 # inline comment")
    assert execution_audit_module._parse_cache_max() == 2500

def test_providers_execution_audit_intent_client_order_id_rejects_none_bool_empty_iter106_s2():
    """[ITER106 #2] providers.execution_audit: intent_client_order_id rejects None / bool / empty."""
    # 2. providers.execution_audit: intent_client_order_id rejects None / bool / empty
    assert execution_audit_module.intent_client_order_id("kraken", None) is None
    assert execution_audit_module.intent_client_order_id("kraken", True) is None
    assert execution_audit_module.intent_client_order_id("kraken", "   ") is None

def test_execution_audit_market_api__parse_bool_flag_and__parse_binance_order_id__clean_order_id_arg_iter151_s3():
    """[ITER151 #3] execution_audit & market_api _parse_bool_flag and _parse_binance_order_id / _clean_order_id_arg."""
    # 3. execution_audit & market_api _parse_bool_flag and _parse_binance_order_id / _clean_order_id_arg
    assert ea_parse_bool("false", True) is False
    assert ea_parse_bool(float("nan"), False) is False
    assert ma_parse_bool("false", True) is False
    assert ma_parse_bool("true", False) is True

    assert require_int_order_id("12345.0", "ctx") == 12345
    assert require_int_order_id(12345.0, "ctx") == 12345
    for bad_oid in (0, -1, "0", "-5", "nan", "null", True, ["1"], None):
        with pytest.raises(ProviderError):
            require_int_order_id(bad_oid, "ctx")
        with pytest.raises(ProviderError):
            MarketApi._clean_order_id_arg(bad_oid, "Binance")
    assert MarketApi._clean_order_id_arg("98765.0", "Binance") == "98765"


def test_execution_calibration_aggregates_partial_fills_across_retries():
    events = [
        {
            "ts": 100, "event": "submit_requested", "intent_id": "retry-intent-1",
            "venue": "binance", "symbol": "BTCUSDC", "side": "buy",
            "qty": 1.0, "price": 100.0, "market": False,
        },
        {"ts": 101, "event": "submit_accepted", "intent_id": "retry-intent-1", "order_id": "OID-A"},
        {
            "ts": 103, "event": "order_terminal_retry", "intent_id": "retry-intent-1",
            "order_id": "OID-A", "status": "expired", "filled_qty": 0.4, "cost": 40.0, "fee": 0.04,
        },
        {"ts": 104, "event": "submit_accepted", "intent_id": "retry-intent-1", "order_id": "OID-B"},
        {
            "ts": 106, "event": "order_terminal", "intent_id": "retry-intent-1",
            "order_id": "OID-B", "status": "closed", "filled_qty": 0.6, "cost": 60.6, "fee": 0.06,
        },
    ]
    rep = calibrate_execution_events(events)
    assert rep["summary"]["all"]["orders"] == 1
    assert rep["summary"]["all"]["filled"] == 1
    assert rep["summary"]["limit"]["ever_partial"] == 1
    assert rep["orders"][0]["filled_qty"] == 1.0
    assert rep["orders"][0]["final_fill_ratio"] == 1.0
    assert abs(rep["orders"][0]["limit_fill_deviation_bps"] - 60.0) < 1e-9
    assert abs(rep["orders"][0]["fee_bps"] - (0.10 / 100.6 * 10_000)) < 1e-9


def test_audited_strategy_executor_forwards_get_orders_keyword_flags(tmp_path):
    """AuditedStrategyExecutor.get_orders forwards authoritative and publish_cache while supporting legacy 3-arg stubs."""
    seen_calls = []

    def _modern_get_orders(symbol, side=None, lookback=None, *, authoritative=False, publish_cache=False):
        seen_calls.append((symbol, side, lookback, authoritative, publish_cache))
        return [{"orderId": "1"}]

    aud = AuditedStrategyExecutor(
        SimpleNamespace(name="t212", get_orders=_modern_get_orders),
        audit=ExecutionAudit(directory=str(tmp_path)),
    )
    res = aud.get_orders("NVDA_US_EQ", "BUY", 3600.0, authoritative=True, publish_cache=False)
    assert res == [{"orderId": "1"}]
    assert seen_calls == [("NVDA_US_EQ", "BUY", 3600.0, True, False)]

    legacy_aud = AuditedStrategyExecutor(
        SimpleNamespace(name="stub", get_orders=lambda sym, side=None, lb=None: [(sym, side, lb)]),
        audit=ExecutionAudit(directory=str(tmp_path)),
    )
    assert legacy_aud.get_orders("BTCUSDC", "SELL", 60.0) == [("BTCUSDC", "SELL", 60.0)]


def test_order_status_ssot_lifecycle_predicates_and_prompt_mapping():
    """Verify CanonicalOrderState classification, status predicates, and prompt display mapping."""
    from providers.order_status_ssot import (
        CanonicalOrderState,
        classify_order_lifecycle_state,
        format_prompt_intent_status,
        is_active_or_filled_status,
        is_cancel_or_expire_status,
        is_non_fill_order_status,
        is_open_working_status,
        is_terminal_audit_status,
    )

    assert classify_order_lifecycle_state("NEW", None, is_resting_in_book=True) == CanonicalOrderState.OPEN_WORKING
    assert (
        classify_order_lifecycle_state("NEW", None, filled_qty=0.4, orig_qty=1.0, is_resting_in_book=True)
        == CanonicalOrderState.PARTIALLY_FILLED
    )
    assert classify_order_lifecycle_state("FILLED", None, filled_qty=1.0, orig_qty=1.0) == CanonicalOrderState.FILLED
    assert (
        classify_order_lifecycle_state("CANCELED", None, filled_qty=0.25, orig_qty=1.0)
        == CanonicalOrderState.PARTIALLY_FILLED
    )
    assert classify_order_lifecycle_state("CANCELED", None) == CanonicalOrderState.CANCELED
    assert classify_order_lifecycle_state("EXPIRED", None) == CanonicalOrderState.EXPIRED
    assert classify_order_lifecycle_state("REJECTED", None) == CanonicalOrderState.REJECTED
    assert classify_order_lifecycle_state("REPLACED", None) == CanonicalOrderState.REPLACED
    assert (
        classify_order_lifecycle_state(None, "refused_before_submit", submission_state="refused")
        == CanonicalOrderState.REFUSED_PRE_SUBMIT
    )
    assert classify_order_lifecycle_state("SUBMIT_UNKNOWN", None) == CanonicalOrderState.SUBMIT_UNKNOWN

    assert is_open_working_status("NEW") is True
    assert is_open_working_status("resting") is True
    assert is_open_working_status("", allow_empty=False) is False
    assert is_open_working_status("", allow_empty=True) is True
    assert is_open_working_status(True, allow_empty=True) is False

    assert is_cancel_or_expire_status("CANCELED_OR_EXPIRED", include_composite=False) is False
    assert is_cancel_or_expire_status("CANCELED_OR_EXPIRED", include_composite=True) is True
    assert is_cancel_or_expire_status(None, "expired") is True

    assert is_non_fill_order_status("SUBMIT_UNKNOWN") is True
    assert is_non_fill_order_status("FILLED", "refused") is True
    assert is_non_fill_order_status("FILLED", "submitted") is False

    assert is_active_or_filled_status("open", 0.0) is True
    assert is_active_or_filled_status("canceled", 0.5) is True
    assert is_active_or_filled_status("canceled", 0.0) is False

    assert is_terminal_audit_status("CLOSED", None) is True
    assert is_terminal_audit_status("NEW", "open") is False

    assert (
        format_prompt_intent_status("EXECUTED", "NEW", in_open_book=True, section="ai_trade_intents")
        == "RESTING_IN_BOOK"
    )
    assert (
        format_prompt_intent_status("EXECUTED", "PARTIALLY_FILLED", in_open_book=True, section="ai_trade_intents")
        == "PARTIALLY_FILLED_IN_BOOK"
    )
    assert (
        format_prompt_intent_status("EXECUTED", "PARTIALLY_FILLED", in_open_book=False, section="ai_trade_intents")
        == "EXECUTED"
    )
    assert (
        format_prompt_intent_status("EXECUTED", "NEW", in_open_book=True, filled_qty=0.0, section="last_cycle_execution_status")
        == "OPEN_RESTING"
    )
    assert (
        format_prompt_intent_status("EXECUTED", "NEW", in_open_book=True, filled_qty=0.5, section="last_cycle_execution_status")
        == "PARTIALLY_FILLED"
    )
    assert (
        format_prompt_intent_status("OBSERVED_SHADOW", "", section="last_cycle_execution_status")
        == "SHADOW_SIMULATED"
    )


def test_active_intents_autonomous_ai_indexing_and_unified_trace(tmp_path):
    """Verify autonomous_ai active intent filtering and end-to-end trace_order_or_intent correlation."""
    from active_intents import build_active_intent_index, trace_order_or_intent
    from providers.order_id_context import (
        all_correlatable_client_order_ids,
        derive_ai_intent_client_order_id,
        derive_executor_intent_client_order_id,
    )
    from state_io import atomic_write_json

    intent_id = "intent_1795099235_sell_hypeusd"
    ai_cid = derive_ai_intent_client_order_id(intent_id)
    krk_cid = derive_executor_intent_client_order_id("kraken", intent_id)
    bn_cid = derive_executor_intent_client_order_id("binance", intent_id)
    hl_cid = derive_executor_intent_client_order_id("hyperliquid", intent_id)
    assert ai_cid is not None and ai_cid.startswith("AI_") and len(ai_cid) <= 36
    assert krk_cid is not None and len(krk_cid) == 32
    assert bn_cid is not None and bn_cid.startswith("SD_") and len(bn_cid) == 35
    assert hl_cid is not None and hl_cid.startswith("0x") and len(hl_cid) == 34
    assert derive_executor_intent_client_order_id("t212", intent_id) is None

    corr_cids = all_correlatable_client_order_ids(intent_id)
    assert {ai_cid.lower(), krk_cid.lower(), bn_cid.lower(), hl_cid.lower()}.issubset(corr_cids)

    state_dir = tmp_path / "cachedb" / "state"
    orders_dir = tmp_path / "cachedb" / "orders"
    audit_dir = tmp_path / "cachedb" / "audit"
    state_dir.mkdir(parents=True)
    orders_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    ai_intents_payload = [
        {
            "intent_id": intent_id,
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "side": "SELL",
            "order_type": "LIMIT",
            "requested_qty": 9.35,
            "requested_price": 69.49,
            "status": "EXECUTED",
            "created_by": "auto_cryptotrade_executor",
            "execution_payload": {
                "orderId": "O7KKV6-45DBY-G5QHOT",
                "clientOrderId": krk_cid,
                "status": "NEW",
                "executedQty": 0.0,
            },
        },
        {
            "intent_id": "intent_terminal_filled",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "111", "status": "FILLED", "executedQty": 0.01},
        },
        {
            "intent_id": "intent_terminal_market",
            "venue": "kraken",
            "symbol": "TAOUSD",
            "side": "SELL",
            "order_type": "MARKET",
            "status": "EXECUTED",
            "execution_payload": {"txid": ["OTX-MKT-1"], "executedQty": 1.25},
        },
        {
            "intent_id": "intent_reconciled_cancel",
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "side": "SELL",
            "status": "CANCELED_OR_EXPIRED",
            "reconciled_terminal": True,
        },
    ]
    atomic_write_json(str(state_dir / "autonomous_trade_intents_state.json"), ai_intents_payload)
    atomic_write_json(
        str(orders_dir / "cache_order_open_kraken.json"),
        {
            "items": {
                "HYPEUSD": [
                    {
                        "orderId": "O7KKV6-45DBY-G5QHOT",
                        "clientOrderId": krk_cid,
                        "symbol": "HYPEUSD",
                        "side": "SELL",
                        "status": "NEW",
                        "price": 69.49,
                        "origQty": 9.35,
                        "executedQty": 0.0,
                    }
                ]
            },
            "fetchtime": {"HYPEUSD": 1795099237000},
            "verified_ts": 1795099237.0,
        },
    )
    atomic_write_json(
        str(orders_dir / "cache_order_filled_binance.json"),
        {
            "items": {
                "BTCUSDC": {
                    "111": {
                        "orderId": "111",
                        "symbol": "BTCUSDC",
                        "side": "BUY",
                        "status": "FILLED",
                        "origQty": 0.01,
                        "executedQty": 0.01,
                    }
                }
            }
        },
    )
    audit = ExecutionAudit(directory=str(audit_dir), clock=lambda: 1795099237.0)
    audit.record(
        "submit_accepted",
        intent_id=intent_id,
        order_id="O7KKV6-45DBY-G5QHOT",
        client_order_id=krk_cid,
        venue="kraken",
        symbol="HYPEUSD",
        side="sell",
        qty=9.35,
        price=69.49,
    )

    idx = build_active_intent_index(str(tmp_path))
    ai_active = [r for r in idx["intents"] if r.get("owner") == "autonomous_ai"]
    assert len(ai_active) == 1
    assert ai_active[0]["intent_id"] == intent_id
    assert ai_active[0]["order_id"] == "O7KKV6-45DBY-G5QHOT"

    for query in (intent_id, "O7KKV6-45DBY-G5QHOT", krk_cid):
        tr = trace_order_or_intent(query, root=str(tmp_path), audit_dir=str(audit_dir))
        assert tr["found"] is True
        assert tr["canonical_state"] == "OPEN_WORKING"
        assert tr["is_resting_in_book"] is True
        assert intent_id.lower() in tr["correlated_ids"]
        assert "o7kkv6-45dby-g5qhot" in tr["correlated_ids"]
        assert len(tr["ai_intents"]) == 1
        assert len(tr["open_orders"]) == 1
        assert len(tr["audit_events"]) == 1

    tr_filled = trace_order_or_intent("intent_terminal_filled", root=str(tmp_path), audit_dir=str(audit_dir))
    assert tr_filled["found"] is True
    assert tr_filled["canonical_state"] == "FILLED"
    assert tr_filled["is_resting_in_book"] is False


def test_evaluate_tradable_position_prompt_view_parity():
    """Verify evaluate_tradable_position_prompt_view separates math from _format_tradable_position_line."""
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_tradable_position_prompt_view
    from orchestratorOS.supervisor.prompt_sections import _format_tradable_position_line

    raw_pos = {
        "venue": "binance",
        "symbol": "BTCUSDC",
        "base": "BTC",
        "quote": "USDC",
        "price": 65000.0,
        "free_base": 0.05,
        "value_usd": 3250.0,
        "free_quote": 2500.0,
        "fee_tier": "VIP0_BNB",
        "maker_fee_bps": 7.5,
        "taker_fee_bps": 7.12,
        "effective_max_buy_usd": 2500.0,
        "effective_max_sell_usd": 3250.0,
        "active_avg_buy_px": 64000.0,
        "open_lots": [
            {"lot_id": "L1", "qty": 0.03, "entry_px": 64500.0},
            {"lot_id": "L2", "qty": 0.01, "entry_px": 62000.0},
        ],
    }
    pv = evaluate_tradable_position_prompt_view(raw_pos, max_usd=2000.0)
    assert pv["_is_tradable_pos_view"] is True
    assert ("max_buy=$", 2000.0) in pv["limit_items"]
    assert ("max_sell=$", 2000.0) in pv["limit_items"]
    assert pv["profitable_lifo"] is not None
    assert pv["older_core_qty"] == 0.01
    assert _format_tradable_position_line(pv, max_usd=2000.0) == _format_tradable_position_line(raw_pos, max_usd=2000.0)


def test_post_phase4_bugfixes_intent_dedup_symbol_case_sqlite_stderr_and_replay(tmp_path, monkeypatch):
    """Verify log-audit bugfixes: tokenized intent symbol case, zombie/mirror dedup, SQLite busy_timeout, stderr line buffering, and shared replay primitives."""
    import sqlite3
    import time as _time
    import log as _log_mod
    from active_intents import build_active_intent_index
    from orchestratorOS.agy.summary_store import connect_sqlite_db, read_summary_row
    from providers.intent_reconciliation import reconcile_intents_from_execution_audit
    from providers.intent_state import _clean_intent_symbol
    from state_io import atomic_write_json
    from strategies.replay_execution import (
        ExecutionModel,
        clamp_sell_fill_to_held,
        iter_eligible_orders_in_side_order,
        resolve_order_fill_price,
    )

    # 1. _clean_intent_symbol preserves canonical mixed-case xStock storage keys & reconciler heals SPCXXUSD
    assert _clean_intent_symbol("SPCXXUSD") == "SPCXxUSD"
    assert _clean_intent_symbol("spcxxusd") == "SPCXxUSD"
    assert _clean_intent_symbol("ANTXUSD") == "ANTxUSD"
    assert _clean_intent_symbol("btcusdc") == "BTCUSDC"
    intents_doc = [{"intent_id": "i-tok", "symbol": "SPCXXUSD", "status": "IGNORED_HOLD"}]
    assert reconcile_intents_from_execution_audit(intents_doc, str(tmp_path / "missing_audit")) is True
    assert intents_doc[0]["symbol"] == "SPCXxUSD"

    # 2. active_intents prunes zombie SUBMIT_UNKNOWN / closed historical EXECUTED intents, parses 3-segment open_orders:kraken:SPCXxUSD, and deduplicates global_outbox mirrors
    now_s = _time.time()
    state_dir = tmp_path / "cachedb" / "state"
    orders_dir = tmp_path / "cachedb" / "orders"
    state_dir.mkdir(parents=True)
    orders_dir.mkdir(parents=True)
    atomic_write_json(
        str(orders_dir / "cache_order_open_kraken.json"),
        {
            "items": {
                "SPCXxUSD": [
                    {"orderId": "OMGUV7-LIVE", "symbol": "SPCXxUSD", "side": "SELL", "status": "NEW", "price": 140.88, "origQty": 5.0, "executedQty": 0.0}
                ]
            },
            "fetchtime": {"SPCXxUSD": int(now_s * 1000)},
            "verified_ts": now_s,
        },
    )
    atomic_write_json(
        str(state_dir / "autonomous_trade_intents_state.json"),
        [
            {
                "intent_id": "intent_live_spcx",
                "venue": "kraken",
                "symbol": "SPCXXUSD",
                "side": "SELL",
                "order_type": "LIMIT",
                "requested_qty": 5.0,
                "requested_price": 140.88,
                "timestamp": now_s - 600.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "OMGUV7-LIVE", "status": "NEW"},
            },
            {
                "intent_id": "intent_zombie_closed",
                "venue": "kraken",
                "symbol": "HYPEUSD",
                "side": "BUY",
                "order_type": "LIMIT",
                "requested_qty": 10.0,
                "requested_price": 80.0,
                "timestamp": now_s - 86400.0 * 10,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "OLD-CLOSED-OID", "status": "NEW"},
            },
            {
                "intent_id": "intent_zombie_unknown",
                "venue": "kraken",
                "symbol": "HYPEUSD",
                "side": "BUY",
                "requested_qty": 8.0,
                "requested_price": 82.0,
                "timestamp": now_s - 86400.0 * 5,
                "status": "SUBMIT_UNKNOWN",
            },
        ],
    )
    (orders_dir / "order_retry_queue.jsonl").write_text(
        json.dumps(
            {
                "intent_id": "intent_live_spcx",
                "provider_name": "kraken",
                "symbol": "SPCXxUSD",
                "side": "SELL",
                "qty": 5.0,
                "requested_price": 140.88,
                "order_id": "OMGUV7-LIVE",
                "lifecycle": "accepted",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    idx = build_active_intent_index(str(tmp_path))
    assert len(idx["intents"]) == 1
    assert idx["intents"][0]["intent_id"] == "intent_live_spcx"
    assert idx["intents"][0]["symbol"] == "SPCXxUSD"
    assert idx["summary"]["pending_notional"]["total_sell"] == pytest.approx(704.40)

    # 3. connect_sqlite_db sets busy_timeout and read_summary_row handles sqlite3.Error gracefully
    db_file = tmp_path / "conversation_summaries.db"
    conn = connect_sqlite_db(str(db_file), timeout=7.5)
    try:
        bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert bt == 7500
    finally:
        conn.close()
    assert read_summary_row(str(tmp_path), "missing-cid") is None
    monkeypatch.setattr(
        "orchestratorOS.agy.summary_store.connect_sqlite_db",
        lambda *_a, **_kw: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    assert read_summary_row(str(tmp_path), "locked-cid") is None

    # 4. log._patched_stderr_write buffers partial caret chunks until newline
    logged_lines: list[str] = []
    monkeypatch.setattr(_log_mod, "_original_stderr_write", lambda s: len(s))
    monkeypatch.setattr(_log_mod._file_logger, "info", lambda msg: logged_lines.append(msg))
    _log_mod._stderr_line_buf.buf = ""
    for ch in ("    ", "^", "^", "^", "\n"):
        _log_mod._patched_stderr_write(ch)
    assert len(logged_lines) == 1
    assert logged_lines[0].endswith("[STDERR]     ^^^")

    # 5. Shared replay_execution intrabar helpers
    em = ExecutionModel(spread_bps=0.0, market_slippage_bps=10.0, intrabar_policy="sell_first")
    assert clamp_sell_fill_to_held(5.0, 4.9999999999995, False) == (4.9999999999995, True)
    assert clamp_sell_fill_to_held(1e-15, 0.0, False) == (0.0, True)
    b_ord = {"side": "BUY", "price": 99.0}
    s_ord = {"side": "SELL", "price": 105.0}
    dispatched = list(iter_eligible_orders_in_side_order([b_ord, s_ord], [b_ord, s_ord], em.side_order()))
    assert dispatched == [("sell", s_ord), ("buy", b_ord)]
    assert resolve_order_fill_price(s_ord, "sell", 100.0, em, limit_key="price") == (False, 105.0)


