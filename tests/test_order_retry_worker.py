"""Tests for order_retry_worker.process_once — the pure queue-draining logic, with a
FAKE `mkt` (no network). The queue is isolated in tmp."""
import os
import subprocess
import sys
import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import order_retry as oq
import order_retry_worker as worker
from providers.strategy_executor import (
    OrderReconciliationCapabilities, OrderStatus,
)
from providers.base import MarketDataProvider
from providers.kraken_provider import KrakenProvider
from providers.market_api import MarketApi


class FakeMkt:
    def __init__(self, price=100.0, succeed=True):
        self.price = price
        self.succeed = succeed
        self.calls = []
        self.status = OrderStatus("open", 0.0, 0.0, 0.0, "NEW")
        self.status_calls = []
        self.lookup_calls = []
    def get_current_price(self, symbol):
        return self.price
    def place(self, symbol, side, price, qty, **kw):
        self.calls.append({"symbol": symbol, "side": side, "price": price, "qty": qty, "kw": kw})
        return {"orderId": 1} if self.succeed else None
    def order_status(self, symbol, order_id, provider_name=None):
        self.status_calls.append((symbol, str(order_id), provider_name))
        return self.status
    def order_by_client_id(self, symbol, client_order_id, provider_name=None):
        self.lookup_calls.append((symbol, client_order_id, provider_name))
        # Explicit absence authorizes a fresh submit; missing lookup support now
        # remains ambiguous and correctly defers production retries.
        return None
    def reconciliation_capabilities(self, symbol, provider_name=None):
        return OrderReconciliationCapabilities(
            True, True, True, True,
            not_found_reliable_for_seconds=90 * 24 * 60 * 60)


class ProcessOnceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        oq.QUEUE_FILE = os.path.join(self.tmp, "q.jsonl")
        oq.LOCK_FILE = os.path.join(self.tmp, "q.lock")
        oq.RETRY_ENABLED = True
        oq.RETRY_INTERVAL_SEC = 300.0
        oq.RETRY_TTL_SEC = 86400.0
        oq.RETRY_MAX_ATTEMPTS = 0
        oq.RETRY_PRICE_TOL = 0.002
        oq.RETRY_DEDUP = True
        oq.RETRY_MAX_QUEUE = 500
        oq.RETRY_CLAIM_LEASE_SEC = 120.0
        oq.RETRY_NOT_FOUND_MAX_AGE_SEC = 30 * 24 * 60 * 60
        self.alerts = []
        original_notify = worker.alert.notify
        worker.alert.notify = lambda **kwargs: self.alerts.append(kwargs)
        self.addCleanup(
            lambda: setattr(worker.alert, "notify", original_notify))
        worker._ALERTED_QUEUE_CORRUPTIONS.clear()
        worker._ALERTED_PRODUCER_QUARANTINES.clear()
        worker._ALERTED_RECONCILIATION_QUARANTINES.clear()

    def test_acceptance_stays_tracked_until_fill(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {"safeback_seconds": 9, "smart": False},
                   requested_price=63000.0, now=1000.0)   # BUY: the current price <= the requested one -> the gate passes.
        mkt = FakeMkt(price=63000.0, succeed=True)
        stats = worker.process_once(mkt, now=1000.0 + 400)   # due (>300s)
        self.assertEqual(stats["succeeded"], 1)
        tracked = oq.load_all()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["lifecycle"], "accepted")
        self.assertEqual(tracked[0]["order_id"], "1")
        # it was resumed with caller_owns_retry=True, at the CURRENT price, the kwargs preserved
        self.assertEqual(len(mkt.calls), 1)
        self.assertTrue(mkt.calls[0]["kw"]["caller_owns_retry"])
        self.assertEqual(mkt.calls[0]["price"], 63000.0)
        self.assertEqual(mkt.calls[0]["kw"]["smart"], False)

        mkt.status = OrderStatus("closed", 1.0, 63000.0, 1.0, "FILLED")
        terminal = worker.process_once(mkt, now=1700.0)
        self.assertEqual(terminal["filled"], 1)
        self.assertEqual(oq.load_all(), [])

    def test_failure_keeps_and_increments_attempts(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        mkt = FakeMkt(succeed=False)   # price=100 default, BUY gate ok (100<=100)
        worker.process_once(mkt, now=1000.0 + 400)
        q = oq.load_all()
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["attempts"], 1)
        self.assertAlmostEqual(q[0]["last_attempt_ts"], 1000.0 + 400)

    def test_retry_passes_persisted_provider_name_to_market_facade(self):
        oq.enqueue(
            "ABCUSD", "BUY", 1.0, {}, requested_price=100.0,
            provider_name="Second", now=1000.0)
        mkt = FakeMkt(price=100.0)

        worker.process_once(mkt, now=1400.0)
        self.assertEqual(
            [call[2] for call in mkt.lookup_calls], ["Second"])

        self.assertEqual(mkt.calls[0]["kw"]["provider_name"], "Second")
        self.assertEqual(
            mkt.calls[0]["kw"]["_retry_requested_price"], 100.0)
        self.assertEqual(
            mkt.calls[0]["kw"]["_retry_price_tolerance"],
            oq.RETRY_PRICE_TOL)

        worker.process_once(mkt, now=1700.0)
        self.assertEqual(
            mkt.status_calls,
            [("ABCUSD", "1", "Second")])

    def test_market_retry_rechecks_price_after_worker_gate_before_submit(self):
        class MovingMarketProvider(MarketDataProvider):
            name = "Moving"

            def __init__(self):
                self.prices = iter((100.0, 100.0, 101.0))
                self.submit_calls = []

            def supports_symbol(self, symbol):
                return symbol == "BTCUSDC"

            def get_current_price(self, symbol):
                return next(self.prices)

            def profit_guard_window_ref(self, symbol, side, safeback_sec):
                return None

            def quantity_decision(self, symbol, side, price, qty, **kwargs):
                return SimpleNamespace(
                    final_qty=float(qty), refuse_reason=None,
                    balance_asset="USDC")

            def place_order(self, symbol, side, price, qty, **kwargs):
                self.submit_calls.append((symbol, side, price, qty, kwargs))
                return {"orderId": "unexpected-duplicate"}

        class AllowedSlot:
            allowed = True
            info = {}

            def commit(self, order_id=None):
                return None

        class SlotContext:
            def __enter__(self):
                return AllowedSlot()

            def __exit__(self, exc_type, exc, traceback):
                return False

        provider = MovingMarketProvider()
        market = MarketApi([provider])
        oq.enqueue(
            "BTCUSDC", "BUY", 1.0,
            {"force": True, "smart": False, "wait_for_trend": False},
            requested_price=100.0, failure_reason="profit_guard",
            provider_name="Moving", now=1000.0)

        with (
            patch(
                "instrument.order_guard.daily_limit_guard",
                return_value=(True, None)),
            patch("instrument.order_guard.margin_for", return_value=0.01),
            patch("instrument.order_guard.profit_guard", return_value=True),
            patch("instrument.trade_cooldown.trade_slot",
                  return_value=SlotContext()),
        ):
            stats = worker.process_once(market, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(provider.submit_calls, [])
        record = oq.load_all()[0]
        self.assertEqual(record["attempts"], 0)
        self.assertEqual(
            record["last_failure_reason"], "retry_price_unfavorable")

    def test_protective_market_retry_reconstructs_venue_only_minimum_policy(self):
        class ProtectiveProvider(MarketDataProvider):
            name = "Protective"

            def __init__(self):
                self.business_minimum_flags = []
                self.submit_calls = []

            def supports_symbol(self, symbol):
                return symbol == "BTCUSDC"

            def get_current_price(self, symbol):
                return 100.0

            def free_balance(self, asset):
                return 1.0

            def policy_cap_quantity(self, *args, **kwargs):
                raise AssertionError("protective SELL must bypass quantity policy")

            def order_filter_refusal(
                    self, symbol, side, price, qty, *, market=False,
                    enforce_business_minimum=True):
                self.business_minimum_flags.append(
                    enforce_business_minimum)
                return None

            def place_order(self, symbol, side, price, qty, **kwargs):
                self.submit_calls.append((symbol, side, price, qty, kwargs))
                return {"orderId": "protective-retry"}

        class AllowedSlot:
            allowed = True
            info = {}

            def commit(self, order_id=None):
                return None

        class SlotContext:
            def __enter__(self):
                return AllowedSlot()

            def __exit__(self, exc_type, exc, traceback):
                return False

        provider = ProtectiveProvider()
        market = MarketApi([provider])
        oq.enqueue(
            "BTCUSDC", "SELL", 0.2,
            {"force": True, "smart": False, "wait_for_trend": False,
             "bypass_profit_guard": True},
            requested_price=100.0, failure_reason="profit_guard",
            provider_name="Protective", now=1000.0)

        with (
            patch(
                "instrument.order_guard.daily_limit_guard",
                return_value=(True, None)),
            patch(
                "instrument.trade_cooldown.trade_slot",
                return_value=SlotContext()),
        ):
            stats = worker.process_once(market, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["succeeded"], 1)
        self.assertEqual(provider.business_minimum_flags, [False])
        self.assertEqual(len(provider.submit_calls), 1)
        self.assertTrue(provider.submit_calls[0][4]["force"])

    def test_trend_refusal_is_retried_without_attempt_or_ttl_consumption(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)

        class TrendDeferredMkt(FakeMkt):
            def place(self, symbol, side, price, qty, **kw):
                self.calls.append({
                    "symbol": symbol, "side": side, "price": price,
                    "qty": qty, "kw": kw,
                })
                kw["_outcome_context"].update(
                    accepted=False, reason="trend_deferred")
                return None

        now = 1000.0 + 400
        stats = worker.process_once(TrendDeferredMkt(), now=now)
        rec = oq.load_all()[0]
        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["succeeded"], 0)
        self.assertEqual(rec["attempts"], 0)
        self.assertEqual(rec["last_failure_reason"], "trend_deferred")
        self.assertFalse(oq.is_expired(rec, now=now + 20 * 86400))

    def test_stale_account_cache_refusal_is_a_non_failure_deferral(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)

        class StaleAccountCacheMkt(FakeMkt):
            def place(self, symbol, side, price, qty, **kw):
                self.calls.append({
                    "symbol": symbol, "side": side, "price": price,
                    "qty": qty, "kw": kw,
                })
                kw["_outcome_context"].update(
                    accepted=False, state="refused",
                    reason="account_cache_not_fresh")
                return None

        now = 1000.0 + 400
        stats = worker.process_once(StaleAccountCacheMkt(), now=now)
        rec = oq.load_all()[0]
        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["succeeded"], 0)
        self.assertEqual(rec["attempts"], 0)
        self.assertEqual(rec["last_failure_reason"], "account_cache_not_fresh")
        self.assertEqual(rec["submission_state"], "refused")
        self.assertEqual(rec["ttl_started_ts"], 1000.0)
        self.assertFalse(oq.is_expired(rec, now=now + 20 * 86400))

    def test_snapshot_change_is_a_non_failure_deferral(self):
        oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {},
            requested_price=100.0, now=1000.0)

        class SnapshotChangedMkt(FakeMkt):
            def place(self, symbol, side, price, qty, **kw):
                kw["_outcome_context"].update(
                    accepted=False, state="refused",
                    reason="account_cache_snapshot_changed")
                return None

        now = 1400.0
        worker.process_once(SnapshotChangedMkt(), now=now)
        rec = oq.load_all()[0]
        self.assertEqual(rec["attempts"], 0)
        self.assertEqual(
            rec["last_failure_reason"], "account_cache_snapshot_changed")
        self.assertFalse(oq.is_expired(rec, now=now + 20 * 86400))

    def test_awaiting_cancel_behaviors(self):
        with self.subTest(msg="active_order_never_submits_or_expires"):
            oq.rewrite([])
            oq.enqueue(
                "BTCUSDC", "SELL", 1.25, {"smart": False},
                requested_price=101.0, now=1000.0,
                lifecycle="awaiting_cancel", replaces_order_id="old-1",
                replaces_original_qty=2.0)
            mkt = FakeMkt(price=101.0)
            mkt.status = OrderStatus("open", 0.75, 75.0, 0.01, "PARTIALLY_FILLED")

            worker.process_once(mkt, now=1400.0)

            rec = oq.load_all()[0]
            self.assertEqual(rec["lifecycle"], "awaiting_cancel")
            self.assertEqual(rec["replaces_order_id"], "old-1")
            self.assertEqual(mkt.calls, [])
            self.assertFalse(oq.is_expired(rec, now=1400.0 + 20 * 86400))

        with self.subTest(msg="activates_only_after_confirmed_cancellation"):
            oq.rewrite([])
            oq.enqueue(
                "BTCUSDC", "SELL", 1.25, {"smart": False},
                requested_price=101.0, now=1000.0,
                lifecycle="awaiting_cancel", replaces_order_id="old-2",
                replaces_original_qty=2.0)
            mkt = FakeMkt(price=101.0)
            mkt.status = OrderStatus("canceled", 0.75, 75.0, 0.01, "CANCELED")

            first = worker.process_once(mkt, now=1400.0)
            activated = oq.load_all()[0]
            self.assertEqual(first["attempted"], 0)
            self.assertEqual(activated["lifecycle"], "submit_pending")
            self.assertEqual(mkt.calls, [])

            second = worker.process_once(mkt, now=1401.0)
            self.assertEqual(second["attempted"], 1)
            self.assertEqual(len(mkt.calls), 1)
            self.assertEqual(mkt.calls[0]["qty"], 1.25)

        with self.subTest(msg="terminal_order_resolves_without_replacement"):
            oq.rewrite([])
            oq.enqueue(
                "BTCUSDC", "SELL", 1.25, {"smart": False},
                requested_price=101.0, now=1000.0,
                lifecycle="awaiting_cancel", replaces_order_id="old-3",
                replaces_original_qty=2.0)
            mkt = FakeMkt(price=101.0)
            mkt.status = OrderStatus("closed", 2.0, 200.0, 0.02, "FILLED")

            worker.process_once(mkt, now=1400.0)

            self.assertEqual(oq.load_all(), [])
            self.assertEqual(mkt.calls, [])

        with self.subTest(msg="rejected_order_never_activates_replacement"):
            oq.rewrite([])
            oq.enqueue(
                "BTCUSDC", "SELL", 2.0, {"smart": False},
                requested_price=101.0, now=1000.0,
                lifecycle="awaiting_cancel", replaces_order_id="old-rejected",
                replaces_original_qty=2.0)
            mkt = FakeMkt(price=101.0)
            mkt.status = OrderStatus("canceled", 0.0, 0.0, 0.0, "REJECTED")

            worker.process_once(mkt, now=1400.0)

            self.assertEqual(oq.load_all(), [])
            self.assertEqual(mkt.calls, [])

        with self.subTest(msg="lookup_error_stays_non_submittable"):
            oq.rewrite([])
            oq.enqueue(
                "BTCUSDC", "SELL", 1.25, {"smart": False},
                requested_price=101.0, now=1000.0,
                lifecycle="awaiting_cancel", replaces_order_id="old-4",
                replaces_original_qty=2.0)

            class BrokenStatusMkt(FakeMkt):
                def order_status(self, symbol, order_id, provider_name=None):
                    raise RuntimeError("status unavailable")

            worker.process_once(BrokenStatusMkt(), now=1400.0)

            rec = oq.load_all()[0]
            self.assertEqual(rec["lifecycle"], "awaiting_cancel")
            self.assertIn("status unavailable", rec["status_error"])

    def test_not_due_skipped(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0)
        mkt = FakeMkt(succeed=True)
        stats = worker.process_once(mkt, now=1000.0 + 100)   # < the interval of 300 -> not due.
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(len(oq.load_all()), 1)              # retained

    def test_expired_dropped_and_alerted(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0)
        queued = oq.load_all()
        queued[0]["submission_state"] = "refused"
        oq.rewrite(queued)
        mkt = FakeMkt(succeed=True)
        alerts = []
        orig = worker.alert.notify
        worker.alert.notify = lambda **kw: alerts.append(kw)
        try:
            stats = worker.process_once(mkt, now=1000.0 + 86400 + 10)   # past the TTL.
        finally:
            worker.alert.notify = orig
        self.assertEqual(stats["expired"], 1)
        self.assertEqual(oq.load_all(), [])          # removed
        self.assertEqual(len(mkt.calls), 0)          # NOT retried (expired).
        self.assertEqual(len(alerts), 1)             # give-up alert

    def test_giveup_behaviors(self):
        from pathlib import Path
        with self.subTest(msg="batch_audited_individually_and_summary_flushes_empty_queue"):
            self.alerts.clear()
            oq.rewrite([])
            Path(self.tmp, "order_retry_giveup_alerts.json").unlink(missing_ok=True)
            oq.RETRY_DEDUP = False
            mkt = FakeMkt()
            with patch.object(worker._AUDIT, "record") as audit:
                for created, count, now in ((1000, 3, 87401), (1030, 2, 87431)):
                    for _ in range(count):
                        oq.enqueue("TAOUSDC", "SELL", 1.0, {}, now=created,
                                   failure_reason="weight_policy_unavailable")
                    stats = worker.process_once(mkt, now=now)
                    self.assertEqual(stats["expired"], count)
                    self.assertEqual(oq.load_all(), [])
                self.assertEqual(len(self.alerts), 1)
                self.assertIn("3 intent(s) expired", self.alerts[0]["body"])
                self.assertEqual(audit.call_count, 5)
                self.assertTrue(all(c.args[0] == "retry_giveup" for c in audit.call_args_list))
                self.assertEqual(len({c.kwargs["intent_id"] for c in audit.call_args_list}), 5)
                worker.process_once(mkt, now=88301)
                self.assertEqual(len(self.alerts), 2)
                self.assertIn("summary", self.alerts[1]["title"])
                self.assertIn("2 intent(s) expired", self.alerts[1]["body"])
                self.assertEqual(audit.call_count, 5)
            self.assertEqual(mkt.calls, [])
            self.assertEqual(mkt.lookup_calls, [])
            oq.RETRY_DEDUP = True

        with self.subTest(msg="digest_corruption_does_not_abort_financial_processing"):
            self.alerts.clear()
            oq.rewrite([])
            Path(self.tmp, "order_retry_giveup_alerts.json").write_text("broken")
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000,
                       requested_price=100.0)
            mkt = FakeMkt()
            self.assertEqual(worker.process_once(mkt, now=1400)["succeeded"], 1)
            self.assertEqual(len(mkt.calls), 1)
            self.assertEqual(self.alerts, [])

        with self.subTest(msg="delivery_error_does_not_hide_another_incident"):
            self.alerts.clear()
            oq.rewrite([])
            Path(self.tmp, "order_retry_giveup_alerts.json").unlink(missing_ok=True)
            for symbol in ("TAOUSDC", "BTCUSDC"):
                oq.enqueue(symbol, "BUY", 1.0, {}, now=1000,
                           failure_reason="weight_policy_unavailable")
            with patch.object(worker.alert, "notify", side_effect=[OSError("offline"), None]) as notify:
                stats = worker.process_once(FakeMkt(), now=87401)
            self.assertEqual(stats["expired"], 2)
            self.assertEqual(notify.call_count, 2)

        with self.subTest(msg="attempt_limit_is_not_mislabeled_as_ttl"):
            self.alerts.clear()
            oq.rewrite([])
            Path(self.tmp, "order_retry_giveup_alerts.json").unlink(missing_ok=True)
            oq.RETRY_MAX_ATTEMPTS = 1
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000,
                       failure_reason="weight_policy_unavailable")
            records = oq.load_all()
            records[0]["attempts"] = 1
            oq.rewrite(records)
            mkt = FakeMkt()
            self.assertEqual(worker.process_once(mkt, now=1001)["expired"], 1)
            self.assertIn("expiry=attempt_limit", self.alerts[0]["body"])
            self.assertEqual(mkt.calls, [])
            oq.RETRY_MAX_ATTEMPTS = 0

    def test_semantically_invalid_record_fails_closed_without_submit(self):
        oq.rewrite([{
            "id": "legacy", "symbol": "BTCUSDC", "side": "BUY", "qty": None,
            "place_kwargs": {}, "requested_price": 100.0, "created_ts": 1000.0,
            "attempts": 2, "last_attempt_ts": 0.0,
        }])
        with open(oq.QUEUE_FILE, "rb") as queue_file:
            before = queue_file.read()
        mkt = FakeMkt(price=100.0)
        with self.assertRaises(oq.RetryQueueCorruptionError):
            worker.process_once(mkt, now=1400.0)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(mkt.lookup_calls, [])
        with open(oq.QUEUE_FILE, "rb") as queue_file:
            self.assertEqual(queue_file.read(), before)

    def test_price_gate_behaviors(self):
        with self.subTest(msg="none_leaves_in_queue"):
            oq.rewrite([])
            oq.enqueue("BTCUSDC", "BUY", 1.0, {}, now=1000.0)
            mkt = FakeMkt(price=None, succeed=True)
            stats = worker.process_once(mkt, now=1000.0 + 400)
            self.assertEqual(stats["attempted"], 0)
            self.assertEqual(len(oq.load_all()), 1)

        with self.subTest(msg="skips_unfavorable"):
            oq.rewrite([])
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=100.0, now=1000.0)
            mkt = FakeMkt(price=90.0, succeed=True)
            stats = worker.process_once(mkt, now=1000.0 + 400)
            self.assertEqual(stats["attempted"], 0)
            self.assertEqual(stats["skipped_price"], 1)
            self.assertEqual(len(mkt.calls), 0)
            q = oq.load_all()
            self.assertEqual(len(q), 1)
            self.assertEqual(q[0]["attempts"], 0)

        with self.subTest(msg="allows_favorable"):
            oq.rewrite([])
            oq.enqueue("BTCUSDC", "SELL", 1.0, {}, requested_price=100.0, now=1000.0)
            mkt = FakeMkt(price=101.0, succeed=True)
            stats = worker.process_once(mkt, now=1000.0 + 400)
            self.assertEqual(stats["succeeded"], 1)
            self.assertEqual(len(mkt.calls), 1)
            self.assertEqual(mkt.calls[0]["price"], 101.0)
            self.assertEqual(oq.load_all()[0]["lifecycle"], "accepted")

    def test_leased_in_queue_during_place_and_not_due(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        seen = {}

        class CheckMkt(FakeMkt):
            def place(self, symbol, side, price, qty, **kw):
                queued = oq.load_all()
                seen["in_queue_during_place"] = len(queued)
                seen["due_during_place"] = oq.is_due(queued[0], now=1400.0)
                return super().place(symbol, side, price, qty, **kw)

        mkt = CheckMkt(price=100.0, succeed=True)
        worker.process_once(mkt, now=1000.0 + 400)
        self.assertEqual(seen["in_queue_during_place"], 1)
        self.assertFalse(seen["due_during_place"])
        self.assertEqual(oq.load_all()[0]["lifecycle"], "accepted")

    def test_truthy_response_without_order_id_is_failure(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)

        class AmbiguousMkt(FakeMkt):
            def place(self, *args, **kwargs):
                return {"status": "unknown"}

        stats = worker.process_once(AmbiguousMkt(price=100.0), now=1400.0)
        self.assertEqual(stats["succeeded"], 0)
        self.assertEqual(oq.load_all()[0]["attempts"], 1)

    def test_lookup_error_defers_ambiguous_record_without_submit(self):
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="submit_ambiguous",
            provider_name="Second", now=1000.0)

        class BrokenLookupMkt(FakeMkt):
            def order_by_client_id(
                    self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(
                    (symbol, client_order_id, provider_name))
                raise RuntimeError("lookup outage")

        mkt = BrokenLookupMkt(price=100.0)
        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["quarantined"], 1)
        self.assertEqual(len(self.alerts), 1)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(
            [call[2] for call in mkt.lookup_calls], ["Second"])
        record = oq.get(record_id)
        self.assertEqual(record["attempts"], 0)
        self.assertNotIn("claim_token", record)

    def test_unsupported_lookup_never_resubmits_ambiguous_record(self):
        record_id = oq.enqueue(
            "AAPLUSD", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="response_without_order_id",
            provider_name="T212", now=1000.0, attempts=10)
        oq.RETRY_TTL_SEC = 1.0
        oq.RETRY_MAX_ATTEMPTS = 1

        class UnsupportedLookupMkt(FakeMkt):
            order_by_client_id = None

        mkt = UnsupportedLookupMkt(price=100.0)
        stats = worker.process_once(mkt, now=100000.0)

        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(stats["expired"], 0)
        self.assertEqual(stats["quarantined"], 1)
        self.assertEqual(oq.get(record_id)["attempts"], 10)

        self.assertEqual(len(self.alerts), 1)

    def test_expired_kraken_producer_claim_is_quarantined_without_submit(self):
        class KrakenClient:
            def __init__(self):
                self.calls = []

            def add_order(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                raise AssertionError("an expired Kraken producer claim must not submit")

        client = KrakenClient()
        provider = KrakenProvider(client=client)
        market = MarketApi([provider])
        claimed = oq.enqueue_claimed(
            "HYPEUSD", "BUY", 1.0, {}, requested_price=100.0,
            provider_name="Kraken", now=1000.0, lease_sec=301.0)

        with patch.object(
                oq, "producer_claim_owner_state", return_value="dead"):
            stats = worker.process_once(market, now=1400.0)

        self.assertIsNotNone(claimed)
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["quarantined"], 1)
        self.assertEqual(client.calls, [])
        record = oq.get(claimed["id"])
        self.assertEqual(record["submission_state"], "producer_claimed")
        self.assertEqual(record["attempts"], 0)

    def test_active_producer_lease_is_not_stolen_while_owner_is_alive(self):
        claimed = oq.enqueue_claimed(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            provider_name="Binance", now=1000.0, lease_sec=301.0)
        mkt = FakeMkt(price=100.0)

        stats = worker.process_once(mkt, now=1100.0)

        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["quarantined"], 0)
        self.assertEqual(mkt.lookup_calls, [])
        self.assertEqual(mkt.calls, [])
        durable = oq.get(claimed["id"])
        self.assertEqual(durable["claim_token"], claimed["claim_token"])
        self.assertEqual(durable["producer_pid"], os.getpid())

    def test_expired_live_producer_recovers_acceptance_after_completion_write_failure(self):
        claimed = oq.enqueue_claimed(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            provider_name="Binance", now=1000.0, lease_sec=301.0)
        accepted_order = {
            "orderId": 88,
            "status": "NEW",
            "clientOrderId": claimed["place_kwargs"]["client_order_id"],
        }
        with patch.object(
                oq, "_write_nolock",
                side_effect=OSError("accepted state write failed")):
            with self.assertRaisesRegex(
                    OSError, "accepted state write failed"):
                oq.complete_claim(
                    claimed, "accepted", now=1001.0,
                    order=accepted_order, provider_name="Binance")

        class ExistingOrderMkt(FakeMkt):
            def order_by_client_id(
                    self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(
                    (symbol, client_order_id, provider_name))
                return accepted_order

        mkt = ExistingOrderMkt(price=100.0)
        stats = worker.process_once(mkt, now=1302.0)

        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["succeeded"], 1)
        self.assertEqual(stats["reconciled"], 1)
        self.assertEqual(stats["quarantined"], 0)
        self.assertEqual(len(mkt.lookup_calls), 1)
        self.assertEqual(mkt.calls, [])
        durable = oq.get(claimed["id"])
        self.assertEqual(durable["lifecycle"], "accepted")
        self.assertEqual(durable["order_id"], "88")
        self.assertNotIn("claim_token", durable)

    def test_expired_live_producer_absence_or_outage_never_submits(self):
        class LiveProducerLookupMkt(FakeMkt):
            def __init__(self, lookup_error):
                super().__init__(price=100.0)
                self.lookup_error = lookup_error

            def order_by_client_id(
                    self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(
                    (symbol, client_order_id, provider_name))
                if self.lookup_error:
                    raise RuntimeError("lookup outage")
                return None

        for lookup_error in (False, True):
            with self.subTest(lookup_error=lookup_error):
                oq.rewrite([])
                self.alerts.clear()
                worker._ALERTED_PRODUCER_QUARANTINES.clear()
                claimed = oq.enqueue_claimed(
                    "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                    provider_name="Binance", now=1000.0, lease_sec=301.0)
                mkt = LiveProducerLookupMkt(lookup_error)

                first = worker.process_once(mkt, now=1302.0)
                second = worker.process_once(mkt, now=1303.0)

                self.assertEqual(first["attempted"], 0)
                self.assertEqual(first["quarantined"], 1)
                self.assertEqual(second["attempted"], 0)
                self.assertEqual(second["quarantined"], 1)
                self.assertEqual(len(mkt.lookup_calls), 2)
                self.assertEqual(mkt.calls, [])
                self.assertEqual(len(self.alerts), 1)
                durable = oq.get(claimed["id"])
                self.assertEqual(
                    durable["claim_token"], claimed["claim_token"])
                self.assertEqual(
                    durable["claim_revision"], claimed["claim_revision"])
                self.assertEqual(
                    durable["submission_state"], "producer_claimed")
                self.assertEqual(durable["attempts"], 0)

    def test_dead_or_reused_producer_identity_recovers_by_lookup_only(self):
        class ExistingOrderMkt(FakeMkt):
            def order_by_client_id(
                    self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(
                    (symbol, client_order_id, provider_name))
                return {
                    "orderId": 72,
                    "status": "NEW",
                    "clientOrderId": client_order_id,
                }

        for owner_state in ("dead", "mismatched"):
            with self.subTest(owner_state=owner_state):
                oq.rewrite([])
                claimed = oq.enqueue_claimed(
                    "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                    provider_name="Binance", now=1000.0, lease_sec=301.0)
                mkt = ExistingOrderMkt(price=100.0)

                with patch.object(
                        oq, "producer_claim_owner_state",
                        return_value=owner_state):
                    stats = worker.process_once(mkt, now=1400.0)

                self.assertEqual(stats["attempted"], 0)
                self.assertEqual(stats["reconciled"], 1)
                self.assertEqual(len(mkt.lookup_calls), 1)
                self.assertEqual(mkt.calls, [])
                self.assertEqual(
                    oq.get(claimed["id"])["lifecycle"], "accepted")

    def test_crash_after_begin_submit_recovers_without_second_place(self):
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="profit_guard", now=1000.0)
        oq.mark_failure(
            record_id, "profit_guard", now=1000.0,
            submission_state="refused")
        claimed = oq.claim([record_id], now=1400.0)[0]
        dispatched = oq.begin_claimed_submit(claimed, now=1400.0)

        class AcceptedDuringCrashMkt(FakeMkt):
            def order_by_client_id(
                    self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(
                    (symbol, client_order_id, provider_name))
                return {
                    "orderId": 73,
                    "status": "NEW",
                    "clientOrderId": client_order_id,
                }

        mkt = AcceptedDuringCrashMkt(price=100.0)
        with patch.object(
                oq, "producer_claim_owner_state", return_value="dead"):
            stats = worker.process_once(mkt, now=1600.0)

        self.assertIsNotNone(dispatched)
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["reconciled"], 1)
        self.assertEqual(len(mkt.lookup_calls), 1)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(oq.get(record_id)["order_id"], "73")

    def test_old_not_found_is_quarantined_beyond_operator_horizon(self):
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="submit_ambiguous", provider_name="Binance",
            now=1000.0)
        mkt = FakeMkt(price=100.0)

        stats = worker.process_once(
            mkt, now=1000.0 + oq.RETRY_NOT_FOUND_MAX_AGE_SEC + 1.0)

        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["quarantined"], 1)
        self.assertEqual(len(mkt.lookup_calls), 1)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(oq.get(record_id)["attempts"], 0)

    def test_not_found_at_operator_horizon_boundary_can_submit(self):
        oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="submit_ambiguous", provider_name="Binance",
            now=1000.0)
        mkt = FakeMkt(price=100.0)

        stats = worker.process_once(
            mkt, now=1000.0 + oq.RETRY_NOT_FOUND_MAX_AGE_SEC)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["succeeded"], 1)
        self.assertEqual(len(mkt.lookup_calls), 1)
        self.assertEqual(len(mkt.calls), 1)

    def test_missing_or_invalid_provider_horizon_fails_closed(self):
        class UnboundedAbsenceMkt(FakeMkt):
            def __init__(self, horizon):
                super().__init__(price=100.0)
                self.horizon = horizon

            def reconciliation_capabilities(
                    self, symbol, provider_name=None):
                return SimpleNamespace(
                    lookup_by_client_order_id=True,
                    not_found_reliable_for_seconds=self.horizon,
                )

        for horizon in (None, float("nan"), float("inf"), 0.0):
            with self.subTest(horizon=horizon):
                oq.rewrite([])
                record_id = oq.enqueue(
                    "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
                    failure_reason="submit_ambiguous",
                    provider_name="Unbounded", now=1000.0)
                mkt = UnboundedAbsenceMkt(horizon)

                stats = worker.process_once(mkt, now=1400.0)

                self.assertEqual(stats["attempted"], 0)
                self.assertEqual(stats["quarantined"], 1)
                self.assertEqual(len(mkt.lookup_calls), 1)
                self.assertEqual(mkt.calls, [])
                self.assertEqual(oq.get(record_id)["attempts"], 0)

    def test_expiry_cleanup_preserves_concurrently_refreshed_revision(self):
        oq.RETRY_TTL_SEC = 1.0
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="profit_guard", now=1000.0)
        oq.mark_failure(
            record_id, "profit_guard", now=1000.0,
            submission_state="refused")
        original_discard = oq.discard_expired

        def refresh_before_cleanup(snapshots, now):
            oq.enqueue(
                "BTCUSDC", "BUY", 2.0, {}, requested_price=99.0,
                failure_reason="profit_guard", now=1400.0)
            return original_discard(snapshots, now)

        with patch.object(
                oq, "discard_expired", side_effect=refresh_before_cleanup):
            stats = worker.process_once(FakeMkt(), now=1400.0)

        durable = oq.get(record_id)
        self.assertEqual(stats["expired"], 0)
        self.assertIsNotNone(durable)
        self.assertEqual(durable["revision"], 1)
        self.assertEqual(durable["qty"], 2.0)

    def test_execution_disabled_refusal_is_terminal_for_existing_retry(self):
        oq.enqueue(
            "DRYUSD", "BUY", 1.0, {}, requested_price=100.0,
            provider_name="Dry", now=1000.0)

        class DisabledMkt(FakeMkt):
            def place(self, symbol, side, price, qty, **kw):
                self.calls.append((symbol, side, price, qty))
                kw["_outcome_context"].update(
                    accepted=False, state="refused",
                    reason="execution_disabled")
                return None

        market = DisabledMkt(price=100.0)
        stats = worker.process_once(market, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["terminal_failed"], 1)
        self.assertEqual(len(market.calls), 1)
        self.assertEqual(oq.load_all(), [])

    def test_filter_refusal_is_terminal_for_existing_retry(self):
        for refusal_reason in (
                "below_min_notional",
                "binance_filter_refused:quantity below lot size"):
            with self.subTest(refusal_reason=refusal_reason):
                oq.rewrite([])
                oq.enqueue(
                    "BTCUSDC", "BUY", 0.01, {}, requested_price=100.0,
                    failure_reason="profit_guard", now=1000.0)

                class FilterRefusedMkt(FakeMkt):
                    def place(self, symbol, side, price, qty, **kw):
                        self.calls.append((symbol, side, price, qty))
                        kw["_outcome_context"].update(
                            accepted=False, state="refused",
                            reason=refusal_reason)
                        return None

                market = FilterRefusedMkt(price=100.0)
                stats = worker.process_once(market, now=1400.0)

                self.assertEqual(stats["attempted"], 1)
                self.assertEqual(stats["terminal_failed"], 1)
                self.assertEqual(len(market.calls), 1)
                self.assertEqual(oq.load_all(), [])

    def test_confirmed_client_id_absence_allows_exactly_one_submit(self):
        oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="submit_ambiguous", now=1000.0)
        mkt = FakeMkt(price=100.0)

        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(len(mkt.lookup_calls), 1)
        self.assertEqual(len(mkt.calls), 1)
        self.assertEqual(oq.load_all()[0]["lifecycle"], "accepted")

    def test_known_pre_submit_refusal_can_retry_without_lookup_support(self):
        oq.enqueue(
            "AAPLUSD", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="profit_guard",
            provider_name="T212", now=1000.0)

        class UnsupportedLookupMkt(FakeMkt):
            order_by_client_id = None

        mkt = UnsupportedLookupMkt(price=100.0)
        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(len(mkt.calls), 1)
        self.assertEqual(
            mkt.calls[0]["kw"]["provider_name"], "T212")

    def test_existing_client_order_is_reconciled_without_resubmit(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)

        class ReconcileMkt(FakeMkt):
            def order_by_client_id(self, symbol, client_order_id,
                                   provider_name=None):
                return {"orderId": 77, "status": "FILLED", "clientOrderId": client_order_id}

        mkt = ReconcileMkt(price=100.0)
        stats = worker.process_once(mkt, now=1400.0)
        self.assertEqual(stats["reconciled"], 1)
        self.assertEqual(stats["succeeded"], 1)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(oq.load_all()[0]["order_id"], "77")

    def test_reconciliation_ignores_unfavorable_price_for_existing_order(self):
        oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            failure_reason="submit_ambiguous", now=1000.0)

        class ExistingOrderMkt(FakeMkt):
            def get_current_price(self, symbol):
                raise AssertionError(
                    "price must not gate reconciliation of an existing order")

            def order_by_client_id(
                    self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(
                    (symbol, client_order_id, provider_name))
                return {
                    "orderId": 77,
                    "status": "NEW",
                    "clientOrderId": client_order_id,
                }

        mkt = ExistingOrderMkt(price=1000.0)
        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(stats["reconciled"], 1)
        self.assertEqual(mkt.calls, [])
        self.assertEqual(oq.load_all()[0]["lifecycle"], "accepted")

    def test_ambiguous_submit_is_reconciled_after_response_loss(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)

        class LostResponseMkt(FakeMkt):
            def __init__(self):
                super().__init__(price=100.0, succeed=False)
                self.lookups = 0

            def order_by_client_id(self, symbol, client_order_id,
                                   provider_name=None):
                self.lookups += 1
                if self.lookups == 1:
                    return None
                return {"orderId": 88, "status": "NEW", "clientOrderId": client_order_id}

        mkt = LostResponseMkt()
        stats = worker.process_once(mkt, now=1400.0)
        self.assertEqual(len(mkt.calls), 1)
        self.assertEqual(stats["reconciled"], 1)
        self.assertEqual(oq.load_all()[0]["order_id"], "88")

    def test_open_partial_is_observed_without_resubmit(self):
        rid = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        oq.mark_accepted(rid, {"orderId": 41, "status": "NEW"}, now=1100.0)
        mkt = FakeMkt(price=100.0)
        mkt.status = OrderStatus(
            "open", 0.4, 39.5, 0.02, "PARTIALLY_FILLED")

        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["observed"], 1)
        self.assertEqual(mkt.calls, [])
        rec = oq.load_all()[0]
        self.assertEqual(rec["lifecycle"], "accepted")
        self.assertEqual(rec["filled_qty"], 0.4)
        self.assertEqual(rec["last_status"], "open")

    def test_expired_partial_retries_only_remainder_with_new_client_id(self):
        rid = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        first_cid = oq.get(rid)["place_kwargs"]["client_order_id"]
        oq.mark_accepted(rid, {"orderId": 42, "status": "NEW"}, now=1100.0)
        mkt = FakeMkt(price=99.0)
        mkt.status = OrderStatus("expired", 0.4, 39.5, 0.02, "EXPIRED")

        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["terminal_retried"], 1)
        self.assertEqual(mkt.calls, [])
        rec = oq.load_all()[0]
        self.assertEqual(rec["lifecycle"], "submit_pending")
        self.assertAlmostEqual(rec["qty"], 0.6)
        self.assertEqual(rec["delivered_qty"], 0.4)
        self.assertEqual(rec["revision"], 1)
        self.assertNotEqual(rec["place_kwargs"]["client_order_id"], first_cid)
        self.assertEqual(rec["order_history"][-1]["venue_status"], "EXPIRED")

        retried = worker.process_once(mkt, now=1700.0)
        self.assertEqual(retried["attempted"], 1)
        self.assertEqual(mkt.calls[0]["qty"], 0.6)

    def test_native_rejected_status_is_retried_but_canceled_is_not(self):
        rejected_id = oq.enqueue(
            "BTCUSDC", "SELL", 2.0, {}, requested_price=100.0, now=1000.0)
        oq.mark_accepted(
            rejected_id, {"orderId": 50, "status": "NEW"}, now=1100.0)
        mkt = FakeMkt(price=101.0)
        mkt.status = OrderStatus("canceled", 0.0, 0.0, 0.0, "REJECTED")

        stats = worker.process_once(mkt, now=1400.0)
        self.assertEqual(stats["terminal_retried"], 1)
        self.assertEqual(oq.load_all()[0]["lifecycle"], "submit_pending")

        # A separate intentional cancel is terminal and removed without submit.
        oq.rewrite([])
        canceled_id = oq.enqueue(
            "BTCUSDC", "SELL", 2.0, {}, requested_price=100.0, now=2000.0)
        oq.mark_accepted(
            canceled_id, {"orderId": 51, "status": "NEW"}, now=2100.0)
        mkt.status = OrderStatus("canceled", 0.5, 50.0, 0.02, "CANCELED")
        alerts = []
        original_notify = worker.alert.notify
        worker.alert.notify = lambda **kwargs: alerts.append(kwargs)
        try:
            canceled = worker.process_once(mkt, now=2400.0)
        finally:
            worker.alert.notify = original_notify

        self.assertEqual(canceled["terminal_failed"], 1)
        self.assertEqual(oq.load_all(), [])
        self.assertEqual(len(mkt.calls), 0)
        self.assertEqual(len(alerts), 1)

    def test_status_error_is_rate_limited_by_observation_interval(self):
        rid = oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        oq.mark_accepted(rid, {"orderId": 60}, now=1100.0)

        class BrokenStatusMkt(FakeMkt):
            def order_status(self, symbol, order_id, provider_name=None):
                self.status_calls.append((symbol, str(order_id), provider_name))
                raise RuntimeError("venue unavailable")

        mkt = BrokenStatusMkt()
        first = worker.process_once(mkt, now=1400.0)
        second = worker.process_once(mkt, now=1500.0)

        self.assertEqual(first["observed"], 1)
        self.assertEqual(second["observed"], 0)
        self.assertEqual(len(mkt.status_calls), 1)
        self.assertIn("venue unavailable", oq.load_all()[0]["status_error"])

    def test_failure_reenqueues_preserving_age(self):
        oq.enqueue("BTCUSDC", "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        mkt = FakeMkt(price=100.0, succeed=False)   # The BUY gate passes (100<=100), but it fails.
        worker.process_once(mkt, now=1000.0 + 400)
        q = oq.load_all()
        self.assertEqual(len(q), 1)                  # Re-added after the failure.
        self.assertEqual(q[0]["created_ts"], 1000.0) # The age is PRESERVED (the TTL does not reset).
        self.assertEqual(q[0]["attempts"], 1)        # attempts+1

    def test_corrupt_queue_aborts_before_market_calls_or_mutation(self):
        oq.enqueue(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            now=1000.0)
        with open(oq.QUEUE_FILE, "ab") as queue_file:
            queue_file.write(b"{broken json\n")
        with open(oq.QUEUE_FILE, "rb") as queue_file:
            before = queue_file.read()
        mkt = FakeMkt(price=100.0)

        with self.assertRaises(oq.RetryQueueCorruptionError):
            worker.process_once(mkt, now=1400.0)

        self.assertEqual(mkt.calls, [])
        self.assertEqual(mkt.lookup_calls, [])
        self.assertEqual(mkt.status_calls, [])
        with open(oq.QUEUE_FILE, "rb") as queue_file:
            self.assertEqual(queue_file.read(), before)

    def test_corruption_alerts_once_per_distinct_fingerprint(self):
        worker._ALERTED_QUEUE_CORRUPTIONS.clear()
        alerts = []
        original_notify = worker.alert.notify
        worker.alert.notify = lambda **kwargs: alerts.append(kwargs)
        first = oq.RetryQueueCorruptionError(
            oq.QUEUE_FILE, 2, "{bad\n", "bad JSON")
        same = oq.RetryQueueCorruptionError(
            oq.QUEUE_FILE, 2, "{bad\n", "bad JSON")
        distinct = oq.RetryQueueCorruptionError(
            oq.QUEUE_FILE, 3, "{other\n", "bad JSON")
        try:
            self.assertTrue(worker._alert_queue_corruption(first))
            self.assertFalse(worker._alert_queue_corruption(same))
            self.assertTrue(worker._alert_queue_corruption(distinct))
        finally:
            worker.alert.notify = original_notify
            worker._ALERTED_QUEUE_CORRUPTIONS.clear()
        self.assertEqual(len(alerts), 2)

    def test_import_rejects_unsafe_worker_poll_interval(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_env = os.environ.copy()
        base_env.update({
            "RETRY_ENABLED": "true",
            "RETRY_INTERVAL_SEC": "300",
            "RETRY_TTL_SEC": "86400",
            "RETRY_MAX_ATTEMPTS": "0",
            "RETRY_PRICE_TOL": "0.002",
            "RETRY_DEDUP": "false",
            "RETRY_MAX_QUEUE": "500",
            "RETRY_CLAIM_LEASE_SEC": "120",
            "RETRY_NOT_FOUND_MAX_AGE_SEC": "2592000",
        })
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value):
                env = base_env.copy()
                env["RETRY_WORKER_POLL_SEC"] = value
                completed = subprocess.run(
                    [sys.executable, "-c", "import order_retry_worker"],
                    cwd=root, env=env, capture_output=True, text=True,
                    check=False)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "RETRY_WORKER_POLL_SEC",
                    completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
