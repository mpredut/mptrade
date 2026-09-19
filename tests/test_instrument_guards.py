"""Tests for the provider-agnostic safeguards in Instrument.place.

Coverage includes the daily cap and anti-spam guard, rapid-fire cooldown,
instantaneous trend gate, and fleet-wide outcome journal. These checks apply to
providers whose guards_internally() returns False. Providers that explicitly own
their guard chain bypass this shared layer.

The minimal in-memory fake provider is fully isolated from the network and uses
the default guards_internally() == False contract.
"""
import os
import sys
import time
import glob
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers.market_api import MarketApi, MarketDataProvider
from providers.strategy_executor import (
    OrderReconciliationCapabilities, SubmissionRefused,
)
from instrument import Instrument
import order_outcomes_log as outcomes_log
from lock import trade_cooldown as tc

SYMBOL = "ZZZFAKEUSD"


class _FakeProvider(MarketDataProvider):
    """Minimal in-memory provider whose guards are handled by Instrument."""

    def __init__(self, name="FakeVenue", price=100.0):
        self._name = name
        self._price = price
        self._orders = []   # [{"side","price","qty","timestamp"(ms)}]
        self.placed = []    # REAL calls to place_order (after every gate)
        self.status_calls = []

    @property
    def name(self):
        return self._name

    def get_current_price(self, symbol):
        return self._price

    def supports_symbol(self, symbol):
        return False

    def free_balance(self, asset):
        return 1_000_000.0   # Enough that it never limits qty artificially in the tests.

    def get_orders(self, symbol, side, since_s):
        cutoff_ms = time.time() * 1000 - since_s * 1000
        out = [o for o in self._orders
              if o["symbol"] == symbol and o["timestamp"] >= cutoff_ms]
        if side:
            out = [o for o in out if o["side"] == side.upper()]
        return out

    def place_order(self, symbol, side, price, qty, **kwargs):
        self.placed.append((symbol, side, price, qty, kwargs))
        return {"orderId": len(self.placed)}

    def order_status(self, symbol, order_id):
        self.status_calls.append((symbol, order_id))
        raise AssertionError("Instrument.place must not poll terminal status")

    def guards_internally(self):
        return False

    def seed_trade(self, side, age_sec=0.0, price=100.0, qty=1.0, symbol=SYMBOL):
        self._orders.append({"symbol": symbol, "side": side.upper(), "price": price,
                             "qty": qty, "timestamp": (time.time() - age_sec) * 1000})


class _GuardsInternallyProvider(_FakeProvider):
    """Provider with its own guard chain; Instrument must skip shared guards."""
    def guards_internally(self):
        return True


class InstrumentGuardsTestCase(unittest.TestCase):
    def setUp(self):
        # Isolate cooldown state and locking so the test never touches live state.
        self._tmp = tempfile.mkdtemp()
        tc.STATE_FILE = os.path.join(self._tmp, "trade_cooldown.json")
        tc.LOCK_FILE = os.path.join(self._tmp, "trade_cooldown.lock")
        # Isolate the outcome journal from the production logger directory.
        self._log_tmp = tempfile.mkdtemp()
        self._orig_log_dir = outcomes_log.ORDER_OUTCOMES_LOG_DIR
        outcomes_log.ORDER_OUTCOMES_LOG_DIR = self._log_tmp
        # The re-placement queue: isolated — Instrument.place() enqueues on failure, and
        # these tests trigger many expected failures (cooldown/daily-limit) -> they must NOT
        # pollute the real cachedb/order_retry_queue.jsonl.
        import order_retry as _oq
        _oq.QUEUE_FILE = os.path.join(self._tmp, "order_retry_queue.jsonl")
        _oq.LOCK_FILE = os.path.join(self._tmp, "order_retry_queue.lock")
        # Explicit pin — the tests must not depend on the kill switch in the live config
        _oq.RETRY_ENABLED = True
        _oq.RETRY_DEDUP = True
        self._clear_state()

    def _clear_state(self):
        import order_retry as _oq
        open(tc.STATE_FILE, 'w').close()
        open(_oq.QUEUE_FILE, 'w').close()
        for f in glob.glob(os.path.join(self._log_tmp, "order_outcomes_*.log")):
            os.remove(f)

    def tearDown(self):
        outcomes_log.ORDER_OUTCOMES_LOG_DIR = self._orig_log_dir

    def _inst(self, provider):
        api = MarketApi([provider])
        return Instrument(name="ZZZFAKE", symbol=SYMBOL, provider=provider.name.lower(),
                          base="ZZZFAKE", quote="USD", api=api)

    def _log_lines(self):
        files = glob.glob(os.path.join(self._log_tmp, "order_outcomes_*.log"))
        lines = []
        for f in files:
            with open(f) as fh:
                lines.extend(fh.read().splitlines())
        return lines

    def test_explicit_balance_lookup_does_not_route_by_bare_asset(self):
        first = _FakeProvider(name="First")
        second = _FakeProvider(name="Second")
        first.free_balance = lambda _asset: 11.0
        second.free_balance = lambda _asset: 22.0
        api = MarketApi([first, second])

        self.assertEqual(api.free_balance_for("second", "USDC"), 22.0)
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            api.free_balance_for("missing", "USDC")

    def test_disabled_execution_never_creates_a_later_live_retry(self):
        import order_retry as oq
        import order_retry_worker as worker

        class SwitchableProvider(_FakeProvider):
            live = False

            def execution_enabled(self):
                return self.live

        provider = SwitchableProvider()
        instrument = self._inst(provider)

        self.assertIsNone(instrument.place("BUY", 100.0, 1.0))
        self.assertEqual(provider.placed, [])
        self.assertEqual(oq.load_all(), [])

        provider.live = True
        stats = worker.process_once(MarketApi([provider]), now=time.time() + 1000.0)
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(provider.placed, [])
        self.assertEqual(oq.load_all(), [])

    def test_live_producer_retains_sole_submit_after_lease_expiry(self):
        import order_retry as oq
        import order_retry_worker as worker

        entered_provider = threading.Event()
        release_provider = threading.Event()

        class BlockingRecoverableProvider(_FakeProvider):
            def reconciliation_capabilities(self):
                return OrderReconciliationCapabilities(
                    lookup_by_client_order_id=True,
                    status_by_order_id=True,
                    cancel_by_order_id=True,
                    list_open_orders=True,
                )

            def place_order(self, symbol, side, price, qty, **kwargs):
                entered_provider.set()
                if not release_provider.wait(timeout=2.0):
                    raise TimeoutError("test did not release the provider")
                self.placed.append((symbol, side, price, qty, kwargs))
                return {"orderId": "producer-only"}

        provider = BlockingRecoverableProvider(name="Recoverable")
        instrument = self._inst(provider)
        result = {}
        errors = []

        def produce():
            try:
                result["order"] = instrument.place(
                    "BUY", 100.0, 1.0, smart=False,
                    wait_for_trend=False, bypass_profit_guard=True)
            except BaseException as exc:  # noqa: BLE001 - Captured for the assertion.
                errors.append(exc)

        producer = threading.Thread(target=produce, name="instrument-producer")
        producer.start()
        self.assertTrue(entered_provider.wait(timeout=1.0))
        pending = oq.load_all()[0]

        # Advance beyond the complete producer lease without sleeping. The PID and
        # process-start identity still prove that the original producer is alive,
        # so the worker must not perform a second external submission.
        stats = worker.process_once(
            MarketApi([provider]),
            now=max(
                float(pending["claim_until"]) + 1.0,
                float(pending["created_ts"]) + oq.RETRY_INTERVAL_SEC + 1.0,
            ))
        self.assertEqual(stats["attempted"], 0)
        self.assertEqual(provider.placed, [])

        release_provider.set()
        producer.join(timeout=2.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result["order"]["orderId"], "producer-only")
        self.assertEqual(len(provider.placed), 1)
        tracked = oq.load_all()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["lifecycle"], "accepted")
        self.assertEqual(tracked[0]["order_id"], "producer-only")

    # -- Daily cap and anti-spam. ---------------------------------------------
    def test_daily_cap_and_safeback_behaviors(self):
        with self.subTest("daily_limit_blocks_after_threshold"):
            p = _FakeProvider()
            inst = self._inst(p)
            # An explicit safeback (48h) so the test is independent of the default in config.
            # backdays = ceil(48h/86400) = 3, so the threshold is 25*3=75;
            # 90 old trades (>3min, under the anti-spam threshold) exceed it.
            for _ in range(90):
                p.seed_trade("BUY", age_sec=4000.0)
            order = inst.place("BUY", 100.0, 1.0, safeback_seconds=48 * 3600 + 60)
            self.assertIsNone(order)
            self.assertEqual(p.placed, [])
            lines = self._log_lines()
            self.assertTrue(any("|refused|daily_limit|" in l for l in lines), lines)

        with self.subTest("safeback_seconds_default_window_misses_old_trades"):
            # 30 Jul, a fix: monitortrades.py (sbs=MT_GUARD_WINDOW_DAYS, 12 DAYS by default) and
            # tradeall.py (14 days) overwrite safeback_seconds on every real call — the default
            # from config (48h) is almost never actually used. instruments.conf already has
            # [KRAKEN_HYPE] enabled=yes under "mt", so the same sbs applies there too.
            p = _FakeProvider()
            inst = self._inst(p)
            for _ in range(60):
                p.seed_trade("BUY", age_sec=5 * 24 * 3600)   # 5 days ago -> OUTSIDE the default 48h.
            order = inst.place("BUY", 100.0, 1.0)   # No override -> the default (48h) does not see them
            self.assertIsNotNone(order)

        with self.subTest("safeback_seconds_override_sees_older_trades_and_blocks"):
            p = _FakeProvider()
            inst = self._inst(p)
            # backdays = ceil((14 days + 60 seconds) / 86400) = 15, so the
            # threshold is 25*15=375; 400 trades exceed it.
            for _ in range(400):
                p.seed_trade("BUY", age_sec=5 * 24 * 3600)   # Five days ago.
            # An explicit 14-day override (identical to tradeall.py: d=14, h=24) -> NOW it sees them -> blocked.
            order = inst.place("BUY", 100.0, 1.0, safeback_seconds=14 * 24 * 3600 + 60)
            self.assertIsNone(order)

        with self.subTest("recent_transaction_blocks"):
            p = _FakeProvider()
            p.seed_trade("BUY", age_sec=5.0)   # 5s ago, under the default threshold of 180s.
            inst = self._inst(p)
            order = inst.place("BUY", 100.0, 1.0)
            self.assertIsNone(order)
            lines = self._log_lines()
            self.assertTrue(any("|refused|recent_transaction|" in l for l in lines), lines)

        with self.subTest("bypass_profit_guard_does_not_skip_daily_limit"):
            p = _FakeProvider()
            for _ in range(90):   # See the 48-hour threshold of 75 tested above.
                p.seed_trade("BUY", age_sec=4000.0)
            inst = self._inst(p)
            order = inst.place("BUY", 100.0, 1.0, safeback_seconds=48 * 3600 + 60, bypass_profit_guard=True)
            self.assertIsNone(order)   # The daily cap stays active even with the bypass.

    def test_reference_only_bypass_behaviors(self):
        with self.subTest("keeps_quantity_policy_active"):
            quantity_calls = []

            class _QuantityCapProvider(_FakeProvider):
                def policy_cap_quantity(self, symbol, side, price, qty, available_qty,
                                        **kwargs):
                    quantity_calls.append((symbol, side, price, qty, available_qty))
                    return 0.25

            p = _QuantityCapProvider()
            p.seed_trade("SELL", age_sec=400.0, price=100.0)
            inst = self._inst(p)

            blocked = inst.place(
                "BUY", 101.0, 1.0, smart=False, caller_owns_retry=True)
            allowed = inst.place(
                "BUY", 101.0, 1.0, smart=False, caller_owns_retry=True,
                bypass_profit_reference=True)

            self.assertIsNone(blocked, "the normal guard must block a BUY above a SELL")
            self.assertIsNotNone(allowed)
            self.assertEqual(len(quantity_calls), 1)
            self.assertEqual(p.placed[0][3], 0.25)

        with self.subTest("does_not_skip_daily_limit"):
            p = _FakeProvider()
            for _ in range(90):
                p.seed_trade("BUY", age_sec=4000.0)
            order = self._inst(p).place(
                "BUY", 100.0, 1.0,
                safeback_seconds=48 * 3600 + 60,
                bypass_profit_reference=True,
                caller_owns_retry=True)
            self.assertIsNone(order)
            self.assertEqual(p.placed, [])

        with self.subTest("is_ignored_for_sell"):
            p = _FakeProvider()
            p.seed_trade("BUY", age_sec=400.0, price=100.0)

            order = self._inst(p).place(
                "SELL", 99.0, 1.0,
                smart=False,
                bypass_profit_reference=True,
                caller_owns_retry=True)

            self.assertIsNone(order, "a SELL below the BUY reference must be blocked")
            self.assertEqual(p.placed, [])
            lines = self._log_lines()
            self.assertTrue(any("|refused|profit_guard|" in line for line in lines), lines)

    def test_quantity_policy_bypass_behaviors(self):
        with self.subTest("is_sell_only_and_keeps_profit_guard"):
            self._clear_state()
            policy_calls = []

            class _ZeroPolicyProvider(_FakeProvider):
                def policy_cap_quantity(self, *args, **kwargs):
                    policy_calls.append((args, kwargs))
                    return 0.0

            p = _ZeroPolicyProvider()
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            inst = self._inst(p)

            loss = inst.place(
                "SELL", 99.0, 1.0, smart=False, caller_owns_retry=True,
                bypass_quantity_policy=True)
            profit = inst.place(
                "SELL", 102.0, 1.0, smart=False, caller_owns_retry=True,
                bypass_quantity_policy=True)

            self.assertIsNone(loss, "the quantity bypass must not skip the profit guard")
            self.assertIsNotNone(profit)
            self.assertEqual(policy_calls, [], "the weight policy must be skipped only on a SELL")
            self.assertEqual(p.placed[0][3], 1.0)

        with self.subTest("is_ignored_for_buy"):
            self._clear_state()
            class _ZeroPolicyProvider(_FakeProvider):
                def policy_cap_quantity(self, *args, **kwargs):
                    return 0.0

            p = _ZeroPolicyProvider()
            p.seed_trade("SELL", age_sec=400.0, price=100.0)
            order = self._inst(p).place(
                "BUY", 98.0, 1.0, smart=False, caller_owns_retry=True,
                bypass_quantity_policy=True)

            self.assertIsNone(order)
            self.assertEqual(p.placed, [])

        with self.subTest("keeps_balance_and_fee_caps"):
            self._clear_state()
            class _FeeCapProvider(_FakeProvider):
                def free_balance(self, _asset):
                    return 1.0

                def policy_cap_quantity(self, *args, **kwargs):
                    raise AssertionError("policy must be bypassed")

                def fee_cap_quantity(self, symbol, side, price, available_qty):
                    return 0.9

            p = _FeeCapProvider()
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            order = self._inst(p).place(
                "SELL", 102.0, 2.0, smart=False, caller_owns_retry=True,
                bypass_quantity_policy=True)

            self.assertIsNotNone(order)
            self.assertEqual(p.placed[0][3], 0.9)

    def test_first_order_allowed_and_logged(self):
        p = _FakeProvider()
        inst = self._inst(p)
        order = inst.place("BUY", 100.0, 1.0)
        self.assertIsNotNone(order)
        self.assertEqual(len(p.placed), 1)
        lines = self._log_lines()
        self.assertTrue(any("|accepted|" in l and SYMBOL in l for l in lines), lines)

    def test_intent_is_persisted_with_same_client_id_before_provider_submit(self):
        import order_retry as oq

        class _InspectProvider(_FakeProvider):
            def place_order(self, symbol, side, price, qty, **kwargs):
                queued = oq.load_all()
                self.persisted_during_submit = copy = dict(queued[0])
                self.submitted_client_id = kwargs.get("client_order_id")
                return {"orderId": "persisted-1"}

        p = _InspectProvider()
        order = self._inst(p).place("BUY", 100.0, 1.0)

        self.assertEqual(order["orderId"], "persisted-1")
        self.assertEqual(
            p.persisted_during_submit["place_kwargs"]["client_order_id"],
            p.submitted_client_id)
        tracked = oq.load_all()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["lifecycle"], "accepted")
        self.assertEqual(tracked[0]["order_id"], "persisted-1")

    def test_truthy_payload_without_order_id_is_unknown_and_remains_queued(self):
        import order_retry as oq

        class _AmbiguousProvider(_FakeProvider):
            def place_order(self, symbol, side, price, qty, **kwargs):
                self.placed.append((symbol, side, price, qty, kwargs))
                return {"status": "UNKNOWN"}

        p = _AmbiguousProvider()
        order = self._inst(p).place("BUY", 100.0, 1.0)

        self.assertIsNone(order)
        queued = oq.load_all()
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["last_failure_reason"],
                         "response_without_order_id")
        self.assertTrue(any("|unknown|response_without_order_id|" in line
                            for line in self._log_lines()))

    def test_submit_response_loss_keeps_pre_submit_intent_with_same_client_id(self):
        import order_retry as oq

        class _LostResponseProvider(_FakeProvider):
            def place_order(self, symbol, side, price, qty, **kwargs):
                self.submitted_client_id = kwargs.get("client_order_id")
                raise TimeoutError("response lost after possible venue acceptance")

        p = _LostResponseProvider()
        order = self._inst(p).place("BUY", 100.0, 1.0)

        self.assertIsNone(order)
        queued = oq.load_all()
        self.assertEqual(len(queued), 1)
        self.assertEqual(
            queued[0]["place_kwargs"]["client_order_id"],
            p.submitted_client_id)
        self.assertEqual(queued[0]["last_failure_reason"], "submit_ambiguous")

    # ── cooldown anti-rapid-fire ────────────────────────────────────────────────
    def test_cooldown_behaviors(self):
        with self.subTest("cooldown_blocks_second_order"):
            self._clear_state()
            p = _FakeProvider()
            inst = self._inst(p)
            first = inst.place("BUY", 100.0, 1.0)
            self.assertIsNotNone(first)
            second = inst.place("SELL", 101.0, 1.0)   # < cooldown_sec after the first one.
            self.assertIsNone(second)
            self.assertEqual(len(p.placed), 1)   # Only the first one reached the provider.
            lines = self._log_lines()
            self.assertTrue(any("|refused|cooldown|" in l for l in lines), lines)

        with self.subTest("cooldown_independent_per_symbol"):
            self._clear_state()
            p = _FakeProvider()
            inst_a = Instrument(name="A", symbol="ZZZFAKEUSD_A", provider=p.name.lower(),
                                base="A", quote="USD", api=MarketApi([p]))
            inst_b = Instrument(name="B", symbol="ZZZFAKEUSD_B", provider=p.name.lower(),
                                base="B", quote="USD", api=MarketApi([p]))
            self.assertIsNotNone(inst_a.place("BUY", 100.0, 1.0))
            self.assertIsNotNone(
                inst_b.place("BUY", 100.0, 1.0))  # A different symbol is unaffected.

        with self.subTest("pair_id_allows_only_the_opposite_leg_through_cooldown"):
            self._clear_state()
            p = _FakeProvider()
            inst = self._inst(p)

            buy = inst.place(
                "BUY", 99.0, 1.0, smart=False, wait_for_trend=False,
                bypass_profit_guard=True,
                cooldown_pair_id="pair-1", caller_owns_retry=True)
            sell = inst.place(
                "SELL", 101.0, 1.0, smart=False, wait_for_trend=False,
                bypass_profit_guard=True,
                cooldown_pair_id="pair-1", caller_owns_retry=True)
            duplicate = inst.place(
                "SELL", 102.0, 1.0, smart=False, wait_for_trend=False,
                bypass_profit_guard=True,
                cooldown_pair_id="pair-1", caller_owns_retry=True)

            self.assertIsNotNone(buy)
            self.assertIsNotNone(sell)
            self.assertIsNone(duplicate)
            self.assertEqual(len(p.placed), 2)

    def test_facade_place_routes_through_pipeline(self):
        # MarketApi.place is the single guarded proxy replacing place_order_smart.
        # It builds an ephemeral Instrument and runs the pipeline (the cooldown blocks the 2nd).
        p = _FakeProvider()
        mkt = MarketApi([p])
        first = mkt.place(SYMBOL, "BUY", 100.0, 1.0)
        self.assertIsNotNone(first)
        self.assertEqual(len(p.placed), 1)
        second = mkt.place(SYMBOL, "SELL", 101.0, 1.0)   # < cooldown -> blocked.
        self.assertIsNone(second)
        self.assertEqual(len(p.placed), 1)

    def test_failed_order_enqueued_for_retry(self):
        import order_retry as _oq
        p = _FakeProvider()
        p.seed_trade("BUY", age_sec=5.0)   # Anti-spam refusal.
        inst = self._inst(p)
        order = inst.place("BUY", 100.0, 1.0)
        self.assertIsNone(order)
        q = _oq.load_all()
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["symbol"], SYMBOL)
        self.assertEqual(q[0]["side"], "BUY")

    def test_retry_flag_prevents_reenqueue(self):
        import order_retry as _oq
        p = _FakeProvider()
        p.seed_trade("BUY", age_sec=5.0)   # Expected refusal.
        inst = self._inst(p)
        order = inst.place("BUY", 100.0, 1.0, caller_owns_retry=True)
        self.assertIsNone(order)
        self.assertEqual(_oq.load_all(), [])   # It is NOT re-enqueued (no recursion)

    def test_success_remains_tracked_until_terminal(self):
        import order_retry as _oq
        p = _FakeProvider()
        inst = self._inst(p)
        order = inst.place("BUY", 100.0, 1.0)
        self.assertIsNotNone(order)
        tracked = _oq.load_all()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["lifecycle"], "accepted")
        self.assertEqual(tracked[0]["order_id"], "1")
        self.assertEqual(p.status_calls, [])

    def test_success_does_not_remove_an_independent_same_side_intent(self):
        import order_retry as _oq
        _oq.RETRY_DEDUP = False
        _oq.enqueue(SYMBOL, "BUY", 1.0, {}, requested_price=100.0, now=1000.0)
        _oq.enqueue(SYMBOL, "SELL", 1.0, {}, requested_price=101.0, now=1001.0)

        p = _FakeProvider()
        order = self._inst(p).place("BUY", 100.0, 1.0)

        self.assertIsNotNone(order)
        remaining = _oq.load_all()
        self.assertEqual(len(remaining), 3)
        buys = [row for row in remaining if row["side"] == "BUY"]
        sells = [row for row in remaining if row["side"] == "SELL"]
        self.assertEqual(len(buys), 2)
        self.assertEqual(len(sells), 1)
        self.assertEqual(
            sorted(row["lifecycle"] for row in buys),
            ["accepted", "submit_pending"],
        )

    def test_smart_flag_gates_price_adjust(self):
        # Smart placement adjusts price; safe placement does not. This preserves
        # the former place_safe_order behavior for lifecycle-owning callers.
        calls = []

        class _SpyProvider(_FakeProvider):
            def adjust_order_price(self, symbol, side, price, cancel_opposite=True):
                calls.append((symbol, side, cancel_opposite))
                return price

        p = _SpyProvider()
        inst = Instrument(name="ZZZFAKE", symbol=SYMBOL, provider=p.name.lower(),
                          base="ZZZFAKE", quote="USD", api=MarketApi([p]))
        inst.place("BUY", 100.0, 1.0, smart=True)
        self.assertEqual(len(calls), 1, "smart=True must call adjust_order_price")
        calls.clear()
        # The cooldown would block the 2nd one on the same symbol -> a different symbol for smart=False.
        inst2 = Instrument(name="ZZZFAKE2", symbol="ZZZFAKEUSD2", provider=p.name.lower(),
                           base="ZZZFAKE2", quote="USD", api=MarketApi([p]))
        inst2.place("BUY", 100.0, 1.0, smart=False)
        self.assertEqual(calls, [], "smart=False must NOT call adjust_order_price")

    def test_cancelorders_and_hours_reach_quantity_hook(self):
        calls = []

        class _QuantitySpyProvider(_FakeProvider):
            def policy_cap_quantity(self, symbol, side, price, qty, available_qty,
                                    base=None, quote=None, cancelorders=False, hours=5):
                calls.append((cancelorders, hours))
                return qty

        p = _QuantitySpyProvider()
        order = self._inst(p).place(
            "BUY", 100.0, 1.0, smart=False, cancelorders=True, hours=2.7)

        self.assertIsNotNone(order)
        self.assertEqual(calls, [(True, 2.7)])

    # -- Financial fidelity for market orders. -------------------------------
    def test_financial_fidelity_and_profit_guard_behaviors(self):
        with self.subTest("market_order_profit_guard_uses_current_price_not_ignored_target"):
            self._clear_state()
            p = _FakeProvider(price=100.0)
            # The last BUY is the SELL reference. The declared target of 102 passes the margin of
            # 1.15%, but MARKET would fill at 100 and only produce costs.
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            order = self._inst(p).place("SELL", 102.0, 1.0, force=True, smart=False)
            self.assertIsNone(order)
            self.assertEqual(p.placed, [])
            self.assertTrue(any("|refused|profit_guard|" in line for line in self._log_lines()))

        with self.subTest("limit_order_keeps_profitable_target_price"):
            self._clear_state()
            p = _FakeProvider(price=100.0)
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            order = self._inst(p).place("SELL", 102.0, 1.0, force=False, smart=False)
            self.assertIsNotNone(order)
            self.assertEqual(p.placed[0][2], 102.0)

        with self.subTest("protective_market_bypass_remains_explicitly_allowed"):
            self._clear_state()
            p = _FakeProvider(price=90.0)
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            order = self._inst(p).place(
                "SELL", 102.0, 1.0, force=True, smart=False,
                bypass_profit_guard=True)
            self.assertIsNotNone(order)
            self.assertTrue(p.placed[0][4]["force"])

        with self.subTest("protective_bypass_still_applies_terminal_venue_filter"):
            self._clear_state()
            import order_retry as oq

            business_minimum_flags = []

            class _DustProvider(_FakeProvider):
                def order_filter_refusal(self, *_args, **kwargs):
                    business_minimum_flags.append(
                        kwargs["enforce_business_minimum"])
                    return "below_min_notional"

            p = _DustProvider(price=90.0)
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            outcome = {}
            order = self._inst(p).place(
                "SELL", 102.0, 0.01, force=True, smart=False,
                bypass_profit_guard=True, _outcome_context=outcome)

            self.assertIsNone(order)
            self.assertEqual(p.placed, [])
            self.assertEqual(oq.load_all(), [])
            self.assertEqual(outcome["reason"], "below_min_notional")
            self.assertEqual(business_minimum_flags, [False])

        with self.subTest("profit_bypass_buy_keeps_the_business_minimum"):
            self._clear_state()
            business_minimum_flags = []

            class _PolicySpyProvider(_FakeProvider):
                def order_filter_refusal(self, *_args, **kwargs):
                    business_minimum_flags.append(
                        kwargs["enforce_business_minimum"])
                    return None

            p = _PolicySpyProvider(price=90.0)
            order = self._inst(p).place(
                "BUY", 90.0, 0.5, force=True, smart=False,
                bypass_profit_guard=True, wait_for_trend=False,
                caller_owns_retry=True)

            self.assertIsNotNone(order)
            self.assertEqual(business_minimum_flags, [True])

        with self.subTest("terminal_filter_refusal_after_prequeue_removes_claim"):
            self._clear_state()
            import order_retry as oq

            class _LateDustProvider(_FakeProvider):
                def place_order(self, *_args, **_kwargs):
                    raise SubmissionRefused(
                        "binance_filter_refused:quantity below lot size")

            p = _LateDustProvider()
            order = self._inst(p).place(
                "BUY", 100.0, 0.01, smart=False, wait_for_trend=False)

            self.assertIsNone(order)
            self.assertEqual(oq.load_all(), [])

        with self.subTest("market_order_allowed_only_when_current_price_meets_margin"):
            self._clear_state()
            p = _FakeProvider(price=102.0)
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            order = self._inst(p).place("SELL", 102.0, 1.0, force=True, smart=False)
            self.assertIsNotNone(order)
            self.assertTrue(p.placed[0][4]["force"])

    def test_trend_deferral_behaviors(self):
        with self.subTest("returns_immediately_without_wait_loop"):
            class _Deferred:
                @staticmethod
                def should_wait(_side, _symbol):
                    return True

                @staticmethod
                def wait_for_favorable_entry(*_args, **_kwargs):
                    raise AssertionError("Instrument.place must not enter a wait loop")

            p = _FakeProvider(price=100.0)
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            with patch("cacheManager.get_short_trend_manager", return_value=_Deferred()):
                order = self._inst(p).place("SELL", 102.0, 1.0, force=False, smart=False)

            self.assertIsNone(order)
            self.assertEqual(p.placed, [])
            self.assertTrue(any("|refused|trend_deferred|" in line
                                for line in self._log_lines()))

        with self.subTest("auto_quantity_enters_outbox_with_numeric_qty"):
            import order_retry as oq
            open(oq.QUEUE_FILE, 'w').close()
            class _Deferred2:
                @staticmethod
                def should_wait(_side, _symbol):
                    return True

            p = _FakeProvider(price=100.0)
            with patch("cacheManager.get_short_trend_manager", return_value=_Deferred2()):
                order = self._inst(p).place("BUY", 100.0, None, smart=False)

            self.assertIsNone(order)
            queued = oq.load_all()
            self.assertEqual(len(queued), 1)
            self.assertGreater(float(queued[0]["qty"]), 0.0)
            self.assertEqual(queued[0]["last_failure_reason"], "trend_deferred")

        with self.subTest("tracked_caller_can_disable_instantaneous_trend_gate"):
            p = _FakeProvider(price=102.0)
            p.seed_trade("BUY", age_sec=400.0, price=100.0)
            with patch("cacheManager.get_short_trend_manager") as manager:
                order = self._inst(p).place(
                    "SELL", 102.0, 1.0, force=False, smart=False,
                    wait_for_trend=False)

            self.assertIsNotNone(order)
            manager.assert_not_called()

    # -- guards_internally skips the complete provider-agnostic layer. --------
    def test_guards_internally_provider_bypasses_new_gates(self):
        p = _GuardsInternallyProvider()
        inst = self._inst(p)
        # A seed that WOULD trip the daily limit if it applied -> it must not block
        for _ in range(60):
            p.seed_trade("BUY", age_sec=4000.0)
        order = inst.place("BUY", 100.0, 1.0)
        self.assertIsNotNone(order)
        self.assertEqual(len(p.placed), 1)
        # No new FLEET-WIDE log (Binance logged for itself, so as not to duplicate).
        self.assertEqual(self._log_lines(), [])
        # The second immediate placement is NOT blocked by the cooldown (guards_internally skips it)
        order2 = inst.place("SELL", 101.0, 1.0)
        self.assertIsNotNone(order2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
