"""Regression coverage for the shared account-cache submission preflight."""

import glob
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from instrument import Instrument
from binance_api import bapi_placeorder as placeorder
import order_guard
import order_outcomes_log
import order_retry
import order_retry_worker
import accepted_order_persistence
from lock import trade_cooldown
from providers.base import MarketDataProvider
from providers.strategy_executor import SubmissionRefused


class _Api:
    def __init__(self, provider):
        self.provider = provider

    def provider_by_name(self, name):
        if name.lower() == self.provider.name.lower():
            return self.provider
        return None


class _StaleCacheProvider(MarketDataProvider):
    def __init__(self):
        self.preflight_calls = []
        self.price_adjustment_calls = []
        self.submission_calls = []
        self.market_price_calls = []

    @property
    def name(self):
        return "StaleCacheVenue"

    def supports_symbol(self, symbol):
        return symbol == "TESTUSDC"

    def get_current_price(self, symbol):
        self.market_price_calls.append(symbol)
        return 100.0

    def preflight_order(self, symbol, side, qty, price=None, *, market=False,
                        kind=None):
        self.preflight_calls.append((symbol, side, qty, price, market, kind))
        raise SubmissionRefused("account_cache_not_fresh")

    def adjust_order_price(self, symbol, side, price, cancel_opposite=True):
        self.price_adjustment_calls.append(
            (symbol, side, price, cancel_opposite))
        return price

    def place_order(self, symbol, side, price, qty, **kwargs):
        self.submission_calls.append((symbol, side, price, qty, kwargs))
        return {"orderId": "unexpected"}


class _BinancePermitProvider(MarketDataProvider):
    def __init__(self):
        self.events = []
        self.state = ("orders-v1", "trades-v1")
        self.permit = object()
        self.final_qty = 1.25

    @property
    def name(self):
        return "Binance"

    def supports_symbol(self, symbol):
        return symbol == "TESTUSDC"

    def get_current_price(self, symbol):
        return 100.0

    def prepare_order_state(self):
        self.events.append(("prepare", self.state))
        return self.state

    def validate_order_state(self, expected_state):
        self.events.append(("validate", expected_state, self.state))
        if tuple(expected_state) != self.state:
            raise SubmissionRefused("account_cache_snapshot_changed")
        return self.state

    def preflight_order(self, symbol, side, qty, price=None, *, market=False,
                        kind=None):
        self.events.append(("preflight", side, qty, price, market, kind))
        return self.permit

    def adjust_order_price(self, symbol, side, price, cancel_opposite=True):
        self.events.append(("adjust", side, price, cancel_opposite))
        return price - 1.0 if side == "BUY" else price + 1.0

    def quantity_decision(self, symbol, side, price, qty, **kwargs):
        self.events.append((
            "quantity", side, price, qty, kwargs.get("market"),
            kwargs.get("enforce_business_minimum")))
        return SimpleNamespace(
            final_qty=self.final_qty,
            refuse_reason=None,
            balance_asset="USDC" if side == "BUY" else "TEST",
        )

    def cancel_opposite_orders(self, symbol, side, requested_price):
        self.events.append(("cancel", side, requested_price))

    def place_order(self, symbol, side, price, qty, **kwargs):
        cancel_requested_price = kwargs.get(
            "_cancel_opposite_requested_price")
        if cancel_requested_price is not None:
            self.events.append(
                ("cancel", side, cancel_requested_price))
        self.events.append(("submit", side, price, qty, kwargs))
        return {"orderId": "accepted"}


class _WeightedBinanceProvider(_BinancePermitProvider):
    """Use the real Binance allocation policy behind the shared pipeline."""

    def free_balance(self, asset):
        return 1_000.0 if asset == "USDC" else 10.0

    def quantity_decision(self, symbol, side, price, qty, **kwargs):
        return MarketDataProvider.quantity_decision(
            self, symbol, side, price, qty, **kwargs)

    def policy_cap_quantity(self, symbol, side, price, qty,
                            available_qty, **kwargs):
        return placeorder.apply_weight_limit(
            symbol, side, price,
            None if qty == float("inf") else qty,
            available_qty)

    def fee_cap_quantity(self, symbol, side, price, available_qty):
        return available_qty


class InstrumentCachePreflightTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        root = self._temp_dir.name

        self._old_cooldown_state = trade_cooldown.STATE_FILE
        self._old_cooldown_lock = trade_cooldown.LOCK_FILE
        trade_cooldown.STATE_FILE = os.path.join(root, "trade_cooldown.json")
        trade_cooldown.LOCK_FILE = os.path.join(root, "trade_cooldown.lock")

        self._old_outcome_dir = order_outcomes_log.ORDER_OUTCOMES_LOG_DIR
        order_outcomes_log.ORDER_OUTCOMES_LOG_DIR = os.path.join(root, "outcomes")

        self._old_queue_file = order_retry.QUEUE_FILE
        self._old_queue_lock = order_retry.LOCK_FILE
        self._old_retry_enabled = order_retry.RETRY_ENABLED
        order_retry.QUEUE_FILE = os.path.join(root, "order_retry_queue.jsonl")
        order_retry.LOCK_FILE = os.path.join(root, "order_retry_queue.lock")
        order_retry.RETRY_ENABLED = True

    def tearDown(self):
        trade_cooldown.STATE_FILE = self._old_cooldown_state
        trade_cooldown.LOCK_FILE = self._old_cooldown_lock
        order_outcomes_log.ORDER_OUTCOMES_LOG_DIR = self._old_outcome_dir
        order_retry.QUEUE_FILE = self._old_queue_file
        order_retry.LOCK_FILE = self._old_queue_lock
        order_retry.RETRY_ENABLED = self._old_retry_enabled
        self._temp_dir.cleanup()

    def test_stale_cache_refusal_precedes_mutation_submission_and_enqueue(self):
        provider = _StaleCacheProvider()
        instrument = Instrument(
            name="TEST",
            symbol="TESTUSDC",
            provider=provider.name.lower(),
            base="TEST",
            quote="USDC",
            api=_Api(provider),
        )
        outcome = {}

        with patch.object(order_retry, "enqueue") as enqueue:
            result = instrument.place(
                "BUY",
                100.0,
                2.5,
                motivation="cache-preflight-test",
                _outcome_context=outcome,
            )

        self.assertIsNone(result)
        self.assertEqual(
            provider.preflight_calls,
            [("TESTUSDC", "BUY", 2.5, 100.0, False,
              "cache-preflight-test")],
        )
        self.assertEqual(provider.price_adjustment_calls, [])
        self.assertEqual(provider.submission_calls, [])
        self.assertEqual(provider.market_price_calls, [])
        enqueue.assert_not_called()
        self.assertEqual(
            outcome,
            {
                "accepted": False,
                "reason": "account_cache_not_fresh",
                "state": "refused",
            },
        )

        log_files = glob.glob(
            os.path.join(order_outcomes_log.ORDER_OUTCOMES_LOG_DIR,
                         "order_outcomes_*.log"))
        self.assertEqual(len(log_files), 1)
        with open(log_files[0], encoding="utf-8") as log_file:
            fields = log_file.read().strip().split("|")
        self.assertEqual(fields[1:7], [
            "TESTUSDC", "BUY", "100.0", "2.5", "refused",
            "account_cache_not_fresh",
        ])
        self.assertFalse(os.path.exists(order_retry.QUEUE_FILE))

    def _binance_instrument(self, provider):
        return Instrument(
            name="TEST",
            symbol="TESTUSDC",
            provider="binance",
            base="TEST",
            quote="USDC",
            api=_Api(provider),
        )

    def test_qty_none_is_resolved_before_exact_permit_issuance(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
        ):
            result = instrument.place(
                "BUY", 100.0, None, smart=False, wait_for_trend=False,
                caller_owns_retry=True, kind="tradeall")

        self.assertEqual(result["orderId"], "accepted")
        quantity = next(event for event in provider.events
                        if event[0] == "quantity")
        preflight = next(event for event in provider.events
                         if event[0] == "preflight")
        submit = next(event for event in provider.events
                      if event[0] == "submit")
        self.assertIsNone(quantity[3])
        self.assertEqual(preflight[2], 1.25)
        self.assertEqual(submit[3], 1.25)
        self.assertIs(submit[4]["cache_permit"], provider.permit)

    def test_reader_sync_precedes_policy_and_snapshot_change_refuses_submit(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)

        def daily_guard(*args, **kwargs):
            self.assertEqual(provider.events[0][0], "prepare")
            provider.state = ("orders-v2", "trades-v2")
            return True, None

        with (
            patch.object(
                order_guard, "daily_limit_guard", side_effect=daily_guard),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
        ):
            result = instrument.place(
                "BUY", 100.0, 2.0, smart=True, wait_for_trend=False,
                caller_owns_retry=True, kind="snapshot-race")

        self.assertIsNone(result)
        self.assertIn(
            ("validate", ("orders-v1", "trades-v1"),
             ("orders-v2", "trades-v2")),
            provider.events,
        )
        self.assertFalse(any(event[0] == "cancel" for event in provider.events))
        self.assertFalse(any(event[0] == "submit" for event in provider.events))

    def test_invalid_weight_inputs_and_stats_fail_before_provider_submit(self):
        cases = (
            ("policy-error", RuntimeError("weight policy failed"),
             {"BUY": {"total_value": 0.0}}, None,
             "weight_policy_unavailable"),
            ("nan-weight", float("nan"),
             {"BUY": {"total_value": 0.0}}, None,
             "invalid_weight_policy_weight"),
            ("stats-error", 0.03, None, RuntimeError("cache read failed"),
             "trade_stats_unavailable"),
        )
        for label, weight, stats, stats_error, expected_reason in cases:
            with self.subTest(label=label):
                provider = _WeightedBinanceProvider()
                instrument = self._binance_instrument(provider)
                outcome = {}
                stats_effect = stats_error if stats_error is not None else stats
                weight_effect = weight if isinstance(weight, Exception) else None
                weight_return = None if isinstance(weight, Exception) else weight
                with (
                    patch.object(
                        order_guard, "daily_limit_guard",
                        return_value=(True, None)),
                    patch.object(order_guard, "profit_guard", return_value=True),
                    patch.object(order_guard, "margin_for", return_value=0.01),
                    patch.object(
                        placeorder.pa,
                        "get_weight_for_cash_permission_at_quant_time",
                        side_effect=weight_effect,
                        return_value=weight_return),
                    patch(
                        "binance_api.bapi_allorders.get_total_traded_stats",
                        side_effect=(stats_error if stats_error is not None
                                     else None),
                        return_value=stats),
                ):
                    result = instrument.place(
                        "BUY", 100.0, None, smart=False,
                        wait_for_trend=False, caller_owns_retry=True,
                        _outcome_context=outcome)

                self.assertIsNone(result)
                self.assertFalse(any(
                    event[0] in {"cancel", "submit"}
                    for event in provider.events))
                self.assertEqual(outcome["reason"], expected_reason)

    def test_missing_trend_falls_back_to_proxy_or_default_weight(self):
        provider = _WeightedBinanceProvider()
        instrument = self._binance_instrument(provider)
        outcome = {}
        with (
            patch.object(
                order_guard, "daily_limit_guard",
                return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(
                placeorder.pa,
                "get_weight_for_cash_permission_at_quant_time",
                return_value=None),
            patch(
                "binance_api.bapi_allorders.get_total_traded_stats",
                return_value={"BUY": {"total_value": 0.0}}),
        ):
            result = instrument.place(
                "BUY", 100.0, None, smart=False,
                wait_for_trend=False, caller_owns_retry=True,
                _outcome_context=outcome)

        self.assertIsNotNone(result)
        self.assertEqual(result["orderId"], "accepted")
        self.assertTrue(any(
            event[0] == "submit" for event in provider.events))

    def test_invalid_weight_policy_price_is_refused(self):
        for label, price, qty in (
                ("invalid-price", float("nan"), 1.0),
                ("explicit-infinite-quantity", 100.0, float("inf"))):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                        SubmissionRefused, "invalid_weight_policy_inputs"):
                    placeorder.apply_weight_limit(
                        "TESTUSDC", "BUY", price, qty, 10.0)

    def test_policy_downtime_fails_open_for_sell_but_not_buy(self):
        # Policy DOWNTIME (the weight fetch or the trade-stats source raises) must
        # never trap a SELL/exit: it fails open to the balance cap so a position can
        # always leave. A BUY still fails closed (do not add unverified exposure).
        available = 5.0
        for label, reason, weight_patch, stats_patch in (
                ("weight-policy-down", "weight_policy_unavailable",
                 patch.object(placeorder.pa,
                              "get_weight_for_cash_permission_at_quant_time",
                              side_effect=RuntimeError("pa down")),
                 patch("binance_api.bapi_allorders.get_total_traded_stats",
                       return_value={"SELL": {"total_value": 0.0},
                                     "BUY": {"total_value": 0.0}})),
                ("trade-stats-down", "trade_stats_unavailable",
                 patch.object(placeorder.pa,
                              "get_weight_for_cash_permission_at_quant_time",
                              return_value=0.03),
                 patch("binance_api.bapi_allorders.get_total_traded_stats",
                       side_effect=RuntimeError("stats down"))),
        ):
            with self.subTest(label=label):
                with weight_patch, stats_patch:
                    self.assertEqual(
                        placeorder.apply_weight_limit(
                            "TAOUSDC", "SELL", 260.0, 1.5827, available),
                        available)
                    with self.assertRaisesRegex(SubmissionRefused, reason):
                        placeorder.apply_weight_limit(
                            "TAOUSDC", "BUY", 260.0, 1.5827, available)

    def test_qty_none_remains_supported_by_valid_real_weight_policy(self):
        provider = _WeightedBinanceProvider()
        instrument = self._binance_instrument(provider)
        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(
                placeorder.pa,
                "get_weight_for_cash_permission_at_quant_time",
                return_value=0.03),
            patch(
                "binance_api.bapi_allorders.get_total_traded_stats",
                return_value={"BUY": {"total_value": 0.0}}),
        ):
            result = instrument.place(
                "BUY", 100.0, None, smart=False,
                wait_for_trend=False, caller_owns_retry=True)

        self.assertEqual(result["orderId"], "accepted")
        preflight = next(
            event for event in provider.events if event[0] == "preflight")
        submit = next(
            event for event in provider.events if event[0] == "submit")
        self.assertAlmostEqual(preflight[2], 0.3)
        self.assertAlmostEqual(submit[3], 0.3)

    def test_market_profit_is_revalidated_at_final_dispatch_boundary(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        market_prices = iter((110.0, 80.0))
        outcome = {}

        def profit_guard(_provider, _symbol, _side, checked_price,
                         _margin, **_kwargs):
            return float(checked_price) > 90.0

        with (
            patch.object(
                provider, "get_current_price",
                side_effect=lambda _symbol: next(market_prices)),
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", side_effect=profit_guard),
            patch.object(order_guard, "margin_for", return_value=0.01),
        ):
            result = instrument.place(
                "SELL", 100.0, 2.0, force=True, smart=False,
                wait_for_trend=False, caller_owns_retry=True,
                _outcome_context=outcome)

        self.assertIsNone(result)
        quantity = next(event for event in provider.events
                        if event[0] == "quantity")
        self.assertTrue(quantity[4])
        self.assertTrue(quantity[5])
        self.assertEqual(quantity[2], 110.0)
        self.assertTrue(any(event[0] == "preflight" for event in provider.events))
        self.assertFalse(any(event[0] == "submit" for event in provider.events))
        self.assertEqual(outcome["reason"], "profit_guard")

    def test_final_market_price_unavailable_precedes_smart_cancellation(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        market_prices = iter((110.0, None))
        outcome = {}

        with (
            patch.object(
                provider, "get_current_price",
                side_effect=lambda _symbol: next(market_prices)),
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
        ):
            result = instrument.place(
                "SELL", 100.0, 2.0, force=True, smart=True,
                wait_for_trend=False, caller_owns_retry=True,
                _outcome_context=outcome)

        self.assertIsNone(result)
        self.assertEqual(outcome["reason"], "market_price_unavailable")
        self.assertFalse(any(
            event[0] in {"cancel", "submit"} for event in provider.events))

    def test_lost_producer_claim_precedes_smart_cancellation(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        outcome = {}

        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(
                order_retry, "begin_claimed_submit", return_value=None),
        ):
            result = instrument.place(
                "BUY", 100.0, 2.0, smart=True,
                wait_for_trend=False, _outcome_context=outcome)

        self.assertIsNone(result)
        self.assertEqual(outcome["reason"], "producer_claim_lost")
        self.assertFalse(any(
            event[0] in {"cancel", "submit"} for event in provider.events))

    def test_durable_producer_marker_precedes_smart_cancellation(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        real_begin = order_retry.begin_claimed_submit

        def begin_and_observe(claim, *args, **kwargs):
            refreshed = real_begin(claim, *args, **kwargs)
            provider.events.append(("producer_marker", refreshed))
            return refreshed

        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(
                order_retry, "begin_claimed_submit",
                side_effect=begin_and_observe),
        ):
            result = instrument.place(
                "BUY", 100.0, 2.0, smart=True, wait_for_trend=False)

        self.assertEqual(result["orderId"], "accepted")
        event_names = [event[0] for event in provider.events]
        self.assertLess(
            event_names.index("producer_marker"),
            event_names.index("cancel"))
        marker = next(
            event[1] for event in provider.events
            if event[0] == "producer_marker")
        self.assertEqual(marker["submission_state"], "producer_claimed")
        self.assertTrue(dict(marker["place_kwargs"])["smart"])

    def test_transient_accepted_tracking_failure_is_retried_exactly(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        real_complete = order_retry.complete_claim
        attempts = []

        def transient_complete(*args, **kwargs):
            attempts.append((args, kwargs))
            if len(attempts) == 1:
                raise OSError("transient fsync failure")
            return real_complete(*args, **kwargs)

        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(
                order_retry, "complete_claim",
                side_effect=transient_complete),
            patch.object(
                accepted_order_persistence.time, "sleep") as retry_sleep,
            patch.object(
                accepted_order_persistence.alert, "notify") as critical_alert,
        ):
            result = instrument.place(
                "BUY", 100.0, 2.0, smart=False, wait_for_trend=False)

        self.assertEqual(result["orderId"], "accepted")
        self.assertEqual(len(attempts), 2)
        retry_sleep.assert_called_once_with(
            accepted_order_persistence.
            ACCEPTED_TRACKING_PERSIST_RETRY_DELAY_SEC)
        critical_alert.assert_not_called()
        tracked = order_retry.load_all()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["lifecycle"], "accepted")
        self.assertEqual(tracked[0]["order_id"], "accepted")

    def test_exhausted_accepted_tracking_retries_alert_and_retain_claim(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)

        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(
                order_retry, "complete_claim", return_value=False) as complete,
            patch.object(accepted_order_persistence.time, "sleep"),
            patch.object(
                accepted_order_persistence.alert, "notify") as critical_alert,
        ):
            result = instrument.place(
                "BUY", 100.0, 2.0, smart=False, wait_for_trend=False)

        self.assertEqual(result["orderId"], "accepted")
        self.assertEqual(
            complete.call_count,
            accepted_order_persistence.ACCEPTED_TRACKING_PERSIST_ATTEMPTS)
        critical_alert.assert_called_once()
        self.assertTrue(critical_alert.call_args.kwargs["email"])
        retained = order_retry.load_all()
        self.assertEqual(len(retained), 1)
        self.assertEqual(
            retained[0]["submission_state"], "producer_claimed")
        self.assertIn("claim_token", retained[0])

    def test_retry_market_tolerance_must_be_below_one(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        outcome = {}

        result = instrument.place(
            "BUY", 100.0, 2.0, force=True, smart=False,
            wait_for_trend=False, caller_owns_retry=True,
            _retry_requested_price=100.0,
            _retry_price_tolerance=1.0,
            _outcome_context=outcome)

        self.assertIsNone(result)
        self.assertEqual(outcome["reason"], "invalid_retry_price_constraint")
        self.assertFalse(any(
            event[0] in {"cancel", "submit"} for event in provider.events))

    def test_caller_supplied_smart_permit_is_refused_before_any_cancellation(self):
        for label in ("expired", "reused", "wrong-kind"):
            with self.subTest(label=label):
                provider = _BinancePermitProvider()
                instrument = self._binance_instrument(provider)
                result = instrument.place(
                    "BUY", 100.0, 2.0, smart=True,
                    wait_for_trend=False, caller_owns_retry=True,
                    kind="smart", cache_permit=object())
                self.assertIsNone(result)
                self.assertFalse(any(
                    event[0] in {"adjust", "cancel", "submit"}
                    for event in provider.events))

    def test_binance_permit_cannot_reach_internally_guarded_other_provider(self):
        provider = _StaleCacheProvider()
        provider.guards_internally = lambda: True
        instrument = Instrument(
            name="TEST",
            symbol="TESTUSDC",
            provider=provider.name,
            base="TEST",
            quote="USDC",
            api=_Api(provider),
        )
        outcome = {}

        result = instrument.place(
            "BUY", 100.0, 2.0, cache_permit=object(),
            _outcome_context=outcome)

        self.assertIsNone(result)
        self.assertEqual(provider.submission_calls, [])
        self.assertEqual(
            outcome["reason"], "invalid_cache_permit_provider")

    def test_blocked_producer_remains_single_owner_beyond_full_lease(self):
        provider = _BinancePermitProvider()
        instrument = self._binance_instrument(provider)
        entered_provider = threading.Event()
        release_provider = threading.Event()
        results = []

        def blocked_place(symbol, side, price, qty, **kwargs):
            entered_provider.set()
            if not release_provider.wait(3.0):
                raise RuntimeError("test provider was not released")
            return {"orderId": "producer-order"}

        class UnsupportedRecoveryMarket:
            def __init__(self):
                self.place_calls = []

            def get_current_price(self, symbol):
                return 100.0

            def place(self, *args, **kwargs):
                self.place_calls.append((args, kwargs))
                return {"orderId": "duplicate"}

        worker_market = UnsupportedRecoveryMarket()
        with (
            patch.object(
                order_guard, "daily_limit_guard", return_value=(True, None)),
            patch.object(order_guard, "profit_guard", return_value=True),
            patch.object(order_guard, "margin_for", return_value=0.01),
            patch.object(provider, "place_order", side_effect=blocked_place),
            patch.object(
                order_retry_worker.alert, "notify") as quarantine_alert,
        ):
            producer = threading.Thread(
                target=lambda: results.append(instrument.place(
                    "BUY", 100.0, 2.0, smart=False,
                    wait_for_trend=False)),
                daemon=True,
            )
            producer.start()
            self.assertTrue(entered_provider.wait(2.0))
            record = order_retry.load_all()[0]
            self.assertEqual(record["submission_state"], "producer_claimed")
            after_full_lease = max(
                float(record["claim_until"]) + 1.0,
                float(record["created_ts"])
                + order_retry.RETRY_INTERVAL_SEC + 1.0,
            )
            try:
                stats = order_retry_worker.process_once(
                    worker_market, now=after_full_lease)
                self.assertEqual(stats["attempted"], 0)
                self.assertEqual(stats["quarantined"], 1)
                self.assertEqual(worker_market.place_calls, [])
                quarantine_alert.assert_called_once()
            finally:
                release_provider.set()
                producer.join(3.0)

        self.assertFalse(producer.is_alive())
        self.assertEqual(results[0]["orderId"], "producer-order")
        tracked = order_retry.load_all()[0]
        self.assertEqual(tracked["lifecycle"], "accepted")
        self.assertEqual(tracked["order_id"], "producer-order")
