"""Exercise live strategy paths through the StrategyExecutor contract.

Golden tests cover replay and dry-run decisions. This module covers the adapter
calls from place, reconcile, and cancel without accessing the network.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_dca as strat  # noqa: E402
from order_retry import OrderSubmissionRefused  # noqa: E402
from providers.strategy_executor import (  # noqa: E402
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
)
from providers.kraken_provider import KrakenProvider  # noqa: E402


class FakeExecutor:
    """Implement the StrategyExecutor contract and record every call."""
    def __init__(self):
        self.calls = []
        self.next_status = OrderStatus("open", 0.0, 0.0, 0.0)
        self._seq = 0
        self.preflight_error = None
        self.client_orders = {}
        self.on_submit = None

    def get_current_price(self, symbol):
        return 60.0

    def reconciliation_capabilities(self):
        return OrderReconciliationCapabilities(True, True, True, False)

    def submit_order(self, symbol, side, qty, price=None, *, market=False, kind=None,
                     client_order_id=None):
        self._seq += 1
        self.calls.append((
            "submit_order", symbol, side, qty, price, market, kind,
            client_order_id,
        ))
        if self.on_submit:
            self.on_submit(client_order_id)
        return f"OID-{self._seq}"

    def order_by_client_id(self, symbol, client_order_id):
        self.calls.append(("order_by_client_id", symbol, client_order_id))
        order_id = self.client_orders.get(client_order_id)
        return None if order_id is None else {"orderId": order_id, "status": "open"}

    def order_status(self, symbol, order_id):
        self.calls.append(("order_status", symbol, order_id))
        return self.next_status

    def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel_order", symbol, order_id))

    def pair_precision(self, symbol):
        return PairPrecision(price_decimals=2, volume_decimals=8, order_min=0.0, base_asset="HYPE")

    def free_balance(self, asset):
        return 0.0

    def ohlc_closes(self, symbol, interval_min):
        return []

    def preflight_order(self, symbol, side, qty, price=None, *, market=False, kind=None):
        self.calls.append(("preflight_order", symbol, side, qty, price, market, kind))
        if self.preflight_error:
            raise self.preflight_error


def _strategy(fake, **over):
    p = dict(currency="USD", entry_amount=650.0, entry_discount_pct=0.8, dca_amount=325.0,
             dca_drop_pct=1.25, check_minutes=2.0, takeprofit_pct=5.0, max_budget=3900.0,
             max_dca_buys=10, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
             adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2, reentry_tolerance_pct=0.05,
             reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[],
             tp_trend_hold=True, tp_trail_pct=3.0)
    p.update(over)
    s = strat.Strategy(fake, "HYPEUSD", strat.StratParams(**p),
                       dry_run=False, initial_state=strat._new_state())
    s._save = lambda *a, **k: None
    return s


class ProviderLivePathTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeExecutor()
        self._orig_notify = strat.notify
        strat.notify = lambda *a, **k: None      # No push notifications in the tests.
        self.s = _strategy(self.fake)

    def tearDown(self):
        strat.notify = self._orig_notify

    def test_precision_and_dust_safety(self):
        with self.subTest(msg="init_reads_the_precision_through_the_contract"):
            self.assertEqual((self.s.price_dec, self.s.vol_dec), (2, 8))

        with self.subTest(msg="dust_safe_leaves_one_tick_at_low_precision"):
            self.s.vol_dec = 2
            self.assertEqual(self.s._dust_safe_qty(13.4), 13.39)

    def test_place_behaviors(self):
        with self.subTest(msg="place_calls_submit_order_and_stores_the_order_id"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            self.s._save = MagicMock()
            self.s._place("buy", 1.0, 60.0, kind="ENTRY", amount=650.0)
            sub = [c for c in self.fake.calls if c[0] == "submit_order"]
            self.assertEqual(len(sub), 1)
            self.assertEqual(sub[0][1:4], ("HYPEUSD", "buy", 1.0))
            self.assertTrue(sub[0][-1])
            self.assertEqual(self.s.s["orders"][-1]["txid"], "OID-1")
            self.assertGreaterEqual(self.s._save.call_count, 2)
            self.assertIsNone(self.s.s["pending_intent"])

    def test_spot_dca_provider_path_releases_intent_when_live_gate_is_off(self):
        class KrakenClient:
            def __init__(self):
                self.calls = []

            def pair_info(self, _pair):
                return {
                    "pair_decimals": 2, "lot_decimals": 8,
                    "ordermin": "0.01", "base": "HYPE",
                }

            def add_order(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {"txid": ["unexpected-live-order"]}

        client = KrakenClient()
        provider = KrakenProvider(client=client)
        with patch.dict(os.environ, {"KRAKEN_LIVE_ORDERS": "false"}):
            strategy = _strategy(provider)
            strategy._save = MagicMock()
            placed = strategy._place(
                "buy", 1.0, 60.0, kind="ENTRY", amount=60.0)

        self.assertFalse(placed)
        self.assertEqual(client.calls, [])
        self.assertIsNone(strategy.s["pending_intent"])

        with self.subTest(msg="place_market_propagates_flag"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            self.s.s["qty"] = 5.0
            self.s._place("sell", 5.0, 59.0, kind="STOP", market=True)
            sub = [c for c in self.fake.calls if c[0] == "submit_order"][-1]
            self.assertTrue(sub[5])                  # market=True is propagated.

        with self.subTest(msg="the_intent_is_persisted_before_the_submit"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            observed = []
            self.fake.on_submit = lambda client_id: observed.append(
                dict(self.s.s.get("pending_intent") or {}))
            self.assertTrue(self.s._place("buy", 1.0, 60.0, kind="ENTRY", amount=60.0))
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["client_order_id"], self.fake.calls[-1][-1])
            self.assertNotIn("order_id", observed[0])

    def test_intent_recovery_and_refusal_behaviors(self):
        with self.subTest(msg="a_lost_response_recovers_without_a_second_submit"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            original = self.fake.submit_order

            def lose_response(*args, **kwargs):
                client_id = kwargs.get("client_order_id")
                self.fake.client_orders[client_id] = "OID-RECOVERED"
                original(*args, **kwargs)
                raise ProviderError("a timeout after acceptance")

            self.fake.submit_order = lose_response
            self.assertFalse(
                self.s._place("buy", 1.0, 60.0, kind="ENTRY", amount=60.0))
            self.assertIsNotNone(self.s.s["pending_intent"])
            self.s.reconcile(60.0)
            submits = [call for call in self.fake.calls if call[0] == "submit_order"]
            self.assertEqual(len(submits), 1)
            self.assertEqual(self.s.s["orders"][0]["txid"], "OID-RECOVERED")
            self.assertIsNone(self.s.s["pending_intent"])

        with self.subTest(msg="definitive_venue_refusal_releases_pending_intent"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            self.fake.submit_order = MagicMock(
                side_effect=OrderSubmissionRefused("Insufficient spot balance"),
            )
            self.assertFalse(
                self.s._place("sell", 1.0, 60.0, kind="TP", market=True)
            )
            self.assertIsNone(self.s.s["pending_intent"])
            self.assertTrue(self.s.s["ledger_reconcile_required"])

    def test_audited_market_submit_receives_observational_reference_price(self):
        class IntentExecutor(FakeExecutor):
            def submit_order_with_intent(
                self, intent_id, symbol, side, qty, price=None, *, market=False,
                kind=None, reference_price=None, client_order_id=None,
            ):
                self._seq += 1
                self.calls.append((
                    "submit_with_intent", symbol, side, qty, price, market, kind,
                    reference_price, client_order_id,
                ))
                return f"OID-{self._seq}"

        executor = IntentExecutor()
        strategy = _strategy(executor)
        strategy._place("sell", 5.0, 59.0, kind="STOP", market=True)

        call = next(item for item in executor.calls if item[0] == "submit_with_intent")
        self.assertIsNone(call[4])
        self.assertTrue(call[5])
        self.assertEqual(call[7], 59.0)

    def test_dca_volatility_behaviors(self):
        with self.subTest(msg="volatility_scaled_dca_amount_and_fail_safe"):
            self.fake = FakeExecutor()
            fixed = _strategy(self.fake, dca_vol_scale_k=0.0)
            self.assertEqual(fixed._effective_dca_amount(), 325.0)

            aggressive = _strategy(
                self.fake, dca_vol_scale_k=-1.0, dca_vol_ref=2.0,
            )
            aggressive._dca_vol_1h = lambda: 4.0
            self.assertEqual(aggressive._effective_dca_amount(), 650.0)
            aggressive._dca_vol_1h = lambda: 20.0
            self.assertEqual(aggressive._effective_dca_amount(), 975.0)

            defensive = _strategy(
                self.fake, dca_vol_scale_k=1.0, dca_vol_ref=2.0,
            )
            defensive._dca_vol_1h = lambda: 4.0
            self.assertEqual(defensive._effective_dca_amount(), 162.5)

            for bad_reference in (-1.0, float("nan")):
                invalid = _strategy(
                    self.fake, dca_vol_scale_k=-1.0,
                    dca_vol_ref=bad_reference,
                )
                invalid._dca_vol_1h = lambda: 4.0
                self.assertEqual(invalid._effective_dca_amount(), 325.0)

            aggressive._dca_vol_1h = lambda: None
            self.assertEqual(aggressive._effective_dca_amount(), 325.0)

        with self.subTest(msg="dca_volatility_uses_closed_ohlc_in_live_mode"):
            self.fake = FakeExecutor()
            strategy = _strategy(
                self.fake, dca_vol_scale_k=-1.0, dca_vol_interval=240,
            )
            closes = [100.0 + (index % 3) for index in range(20)]
            strategy.client.ohlc_closes = MagicMock(return_value=closes)

            self.assertIsNotNone(strategy._dca_vol_1h())
            strategy.client.ohlc_closes.assert_called_once_with("HYPEUSD", 240)

    def test_provider_error_and_backoff_handling(self):
        with self.subTest(msg="place_ProviderError_does_not_store_the_order"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            def boom(*a, **k):
                raise ProviderError("Insufficient funds")
            self.fake.submit_order = boom
            self.s._place("buy", 1.0, 60.0, kind="ENTRY", amount=650.0)
            self.assertEqual(self.s.s["orders"], [])   # A failure -> no order stored.
            self.assertIn("buy:ENTRY", self.s.s["placement_backoffs"])

        with self.subTest(msg="insufficient_funds_preflight_has_persistent_exponential_backoff"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            self.fake.preflight_error = ProviderError("Insufficient funds")
            with patch.object(strat.time, "time", return_value=1000.0):
                self.assertFalse(
                    self.s._place("sell", 1.0, 60.0, kind="TP"))
            record = self.s.s["placement_backoffs"]["sell:TP"]
            self.assertEqual(record["attempts"], 1)
            self.assertEqual(record["until"], 1120.0)
            preflight_calls = len([
                call for call in self.fake.calls if call[0] == "preflight_order"
            ])

            with patch.object(strat.time, "time", return_value=1060.0):
                self.assertFalse(
                    self.s._place("sell", 1.0, 60.0, kind="TP"))
            self.assertEqual(len([
                call for call in self.fake.calls if call[0] == "preflight_order"
            ]), preflight_calls)

            with patch.object(strat.time, "time", return_value=1121.0):
                self.assertFalse(
                    self.s._place("sell", 1.0, 60.0, kind="TP"))
            record = self.s.s["placement_backoffs"]["sell:TP"]
            self.assertEqual(record["attempts"], 2)
            self.assertEqual(record["until"], 1361.0)

        with self.subTest(msg="market_protection_bypasses_funds_backoff"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            self.s.s["placement_backoffs"]["sell:STOP"] = {
                "attempts": 4, "until": 9999999999.0,
                "reason": "Insufficient funds",
            }
            self.assertTrue(
                self.s._place("sell", 1.0, 59.0, kind="STOP", market=True))
            self.assertTrue(any(
                call[0] == "submit_order" for call in self.fake.calls
            ))

        with self.subTest(msg="a_refused_preflight_creates_no_intent_or_order"):
            self.fake = FakeExecutor()
            self.s = _strategy(self.fake)
            self.fake.preflight_error = ProviderError("Insufficient funds")
            placed = self.s._place("buy", 10.0, 60.0, kind="DCA", amount=600.0)
            self.assertFalse(placed)
            self.assertTrue(any(c[0] == "preflight_order" for c in self.fake.calls))
            self.assertFalse(any(c[0] == "submit_order" for c in self.fake.calls))
            self.assertEqual(self.s.s["orders"], [])

    def test_reconcile_fills_on_closed_through_order_status(self):
        self.s.s["orders"] = [{"txid": "OID-9", "side": "buy", "vol": 2.0, "price": 60.0,
                               "amount": 120.0, "kind": "ENTRY", "ts": 0}]
        self.fake.next_status = OrderStatus("closed", filled_qty=2.0, cost=120.0, fee=0.31)
        self.s.reconcile(60.0)
        self.assertTrue(any(c[0] == "order_status" for c in self.fake.calls))
        self.assertAlmostEqual(self.s.s["qty"], 2.0)     # The fill was applied.
        self.assertEqual(self.s.s["orders"], [])         # The order was consumed.

    def test_cancel_open_calls_cancel_order(self):
        self.s._save = MagicMock()
        self.s.s["orders"] = [{"txid": "OID-7", "side": "buy", "vol": 1.0, "price": 60.0,
                               "amount": 60.0, "kind": "ENTRY", "ts": 0}]
        self.s._cancel_open("buy")
        self.assertIn(("cancel_order", "HYPEUSD", "OID-7"), self.fake.calls)
        # The order stays tracked until a terminal status, so we do not lose
        # a fill concurrent with the cancellation.
        self.assertTrue(self.s.s["orders"][0]["cancel_requested"])
        self.s._save.assert_called_once()


class PercentageSizingTest(unittest.TestCase):
    def test_percentage_sizing_behaviors(self):
        with self.subTest(msg="default_off_preserves_legacy_amounts_exactly"):
            engine = _strategy(FakeExecutor())
            self.assertFalse(engine._pct_sizing_on())
            self.assertEqual(engine._effective_entry_amount(), 650.0)
            self.assertEqual(engine._base_dca_amount(), 325.0)
            self.assertEqual(engine._effective_max_budget(), 3900.0)

        with self.subTest(msg="equivalent_percentages_preserve_hype_proportions"):
            engine = _strategy(
                FakeExecutor(), total_budget=10_000.0, alloc_pct=39.0,
                entry_pct=100.0 / 6.0, dca_pct=100.0 / 12.0,
            )
            self.assertTrue(engine._pct_sizing_on())
            self.assertAlmostEqual(engine._effective_max_budget(), 3900.0)
            self.assertAlmostEqual(engine._effective_entry_amount(), 650.0)
            self.assertAlmostEqual(engine._base_dca_amount(), 325.0)

        with self.subTest(msg="partial_or_non_finite_percentage_config_fails_closed"):
            invalid = (
                {"entry_pct": 10.0},
                {"total_budget": 10_000.0, "alloc_pct": 39.0, "entry_pct": 0.0, "dca_pct": 8.0},
                {"total_budget": 10_000.0, "alloc_pct": 101.0, "entry_pct": 10.0, "dca_pct": 8.0},
                {"total_budget": float("nan"), "alloc_pct": 39.0, "entry_pct": 10.0, "dca_pct": 8.0},
            )
            for overrides in invalid:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    _strategy(FakeExecutor(), **overrides)


class CorePercentageValidationTest(unittest.TestCase):
    def test_core_validation_behaviors(self):
        with self.subTest(msg="valid_disabled_boundaries_remain_supported"):
            engine = _strategy(
                FakeExecutor(), entry_discount_pct=0, stop_loss_pct=0,
                enable_takeprofit=False, takeprofit_pct=0,
            )
            self.assertEqual(engine.p.entry_discount_pct, 0.0)
            self.assertEqual(engine.p.stop_loss_pct, 0.0)
            self.assertEqual(engine.p.takeprofit_pct, 0.0)

        with self.subTest(msg="order_percentages_reject_invalid_values"):
            invalid = (
                ("entry_discount_pct", float("nan")),
                ("entry_discount_pct", -0.01),
                ("entry_discount_pct", 100.0),
                ("dca_drop_pct", float("inf")),
                ("dca_drop_pct", 0.0),
                ("dca_drop_pct", -1.0),
                ("takeprofit_pct", float("nan")),
                ("takeprofit_pct", 0.0),
                ("takeprofit_pct", -1.0),
                ("stop_loss_pct", float("inf")),
                ("stop_loss_pct", -1.0),
            )
            for field, value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _strategy(FakeExecutor(), **{field: value})

        with self.subTest(msg="numeric_values_are_normalized_to_float"):
            engine = _strategy(
                FakeExecutor(), entry_discount_pct="0.8", dca_drop_pct="1.25",
                takeprofit_pct="5", stop_loss_pct="12.5", trend_topup="650",
            )
            self.assertEqual(
                (engine.p.entry_discount_pct, engine.p.dca_drop_pct,
                 engine.p.takeprofit_pct, engine.p.stop_loss_pct, engine.p.trend_topup),
                (0.8, 1.25, 5.0, 12.5, 650.0),
            )

        with self.subTest(msg="regime_warmup_must_fit_the_canonical_window"):
            for invalid in (2, 43):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "STRAT_REGIME_MIN_SAMPLES",
                ):
                    _strategy(FakeExecutor(), regime_min_samples=invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
