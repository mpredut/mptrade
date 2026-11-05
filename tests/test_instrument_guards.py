"""Tests for the provider-agnostic safeguards in Instrument.place.

Coverage includes the daily cap and anti-spam guard, rapid-fire cooldown,
instantaneous trend gate, and fleet-wide outcome journal. These checks apply to
providers whose guards_internally() returns False. Providers that explicitly own
their guard chain bypass this shared layer.

The minimal in-memory fake provider is fully isolated from the network and uses
the default guards_internally() == False contract.
"""
import json
import os
import sys
import time
import glob
import tempfile
import threading
import unittest
from unittest import mock
from unittest.mock import patch, ANY

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers.market_api import MarketApi, MarketDataProvider
from providers.strategy_executor import (
    OrderReconciliationCapabilities, SubmissionRefused,
)
from instrument import Instrument
import order_guard
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
        # Isolate the outcome journal and execution audit from the production logger directory.
        self._log_tmp = tempfile.mkdtemp()
        self._orig_log_dir = outcomes_log.ORDER_OUTCOMES_LOG_DIR
        outcomes_log.ORDER_OUTCOMES_LOG_DIR = self._log_tmp
        import instrument as _inst_mod
        self._orig_exec_audit_dir = _inst_mod._EXECUTION_AUDIT.directory
        _inst_mod._EXECUTION_AUDIT.directory = self._log_tmp
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
        import instrument as _inst_mod
        _inst_mod._EXECUTION_AUDIT.directory = self._orig_exec_audit_dir

    def _inst(self, provider, params=None):
        api = MarketApi([provider])
        return Instrument(name="ZZZFAKE", symbol=SYMBOL, provider=provider.name.lower(),
                          base="ZZZFAKE", quote="USD", params=params, api=api)

    def _log_lines(self):
        files = glob.glob(os.path.join(self._log_tmp, "order_outcomes_*.log"))
        lines = []
        for f in files:
            with open(f) as fh:
                lines.extend(fh.read().splitlines())
        return lines

    def test_provider_balance_routing_and_disabled_execution(self):
        with self.subTest("explicit_balance_lookup_does_not_route_by_bare_asset"):
            first = _FakeProvider(name="First")
            second = _FakeProvider(name="Second")
            first.free_balance = lambda _asset: 11.0
            second.free_balance = lambda _asset: 22.0
            api = MarketApi([first, second])

            self.assertEqual(api.free_balance_for("second", "USDC"), 22.0)
            with self.assertRaisesRegex(ValueError, "Unknown provider"):
                api.free_balance_for("missing", "USDC")

        with self.subTest("disabled_execution_never_creates_a_later_live_retry"):
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
                if not release_provider.wait(timeout=5.0):
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
                    wait_for_trend=False, bypass_guards={"profit_guard"})
            except BaseException as exc:  # noqa: BLE001 - Captured for the assertion.
                errors.append(exc)

        producer = threading.Thread(target=produce, name="instrument-producer")
        producer.start()
        self.assertTrue(entered_provider.wait(timeout=5.0))
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
        with self.subTest("daily_limit_blocks_after_25_orders_in_24h"):
            p = _FakeProvider()
            inst = self._inst(p)
            for _ in range(25):
                p.seed_trade("BUY", age_sec=4000.0)
            order = inst.place("BUY", 100.0, 1.0)
            self.assertIsNone(order)
            self.assertEqual(p.placed, [])
            lines = self._log_lines()
            self.assertTrue(any("|refused|daily_limit|" in l for l in lines), lines)

        with self.subTest("daily_limit_24h_window_ignores_older_trades_even_with_safeback_seconds"):
            p = _FakeProvider()
            inst = self._inst(p)
            for _ in range(60):
                p.seed_trade("BUY", age_sec=5 * 24 * 3600)   # 5 days ago -> OUTSIDE 24h GO1 window.
            order = inst.place("BUY", 100.0, 1.0, safeback_seconds=14 * 24 * 3600 + 60)
            self.assertIsNotNone(order)

        with self.subTest("antispam_5s_blocks_same_symbol_and_allows_after_5s"):
            self._clear_state()
            for side in ("BUY", "SELL"):
                p_recent = _FakeProvider()
                p_recent.seed_trade(side, age_sec=2.0)
                inst_recent = self._inst(p_recent)
                self.assertIsNone(inst_recent.place(side, 100.0, 1.0, bypass_guards={"profit_guard"}))

                p_ok = _FakeProvider()
                p_ok.seed_trade(side, age_sec=6.0)
                inst_ok = self._inst(p_ok)
                order = inst_ok.place(side, 100.0, 1.0, bypass_guards={"profit_guard"})
                self.assertIsNotNone(order)
                self.assertEqual(len(p_ok.placed), 1)

        with self.subTest("bypass_profit_guard_does_not_skip_daily_limit"):
            p = _FakeProvider()
            for _ in range(25):
                p.seed_trade("BUY", age_sec=4000.0)
            inst = self._inst(p)
            order = inst.place("BUY", 100.0, 1.0, bypass_guards={"profit_guard"})
            self.assertIsNone(order)   # The daily cap stays active even with profit_guard bypass.

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
                bypass_guards={"G_PROF"})

            self.assertIsNone(blocked, "the normal guard must block a BUY above a SELL")
            self.assertIsNotNone(allowed)
            self.assertEqual(len(quantity_calls), 1)
            self.assertEqual(p.placed[0][3], 0.25)

        with self.subTest("does_not_skip_daily_limit"):
            p = _FakeProvider()
            for _ in range(25):
                p.seed_trade("BUY", age_sec=4000.0)
            order = self._inst(p).place(
                "BUY", 100.0, 1.0,
                bypass_guards={"G_PROF"},
                caller_owns_retry=True)
            self.assertIsNone(order)
            self.assertEqual(p.placed, [])

        with self.subTest("is_ignored_for_sell"):
            p = _FakeProvider()
            p.seed_trade("BUY", age_sec=400.0, price=100.0)

            order = self._inst(p).place(
                "SELL", 99.0, 1.0,
                smart=False,
                bypass_guards={"G_PROF"},
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
                bypass_guards={"G_QLIMIT1"})
            profit = inst.place(
                "SELL", 102.0, 1.0, smart=False, caller_owns_retry=True,
                bypass_guards={"G_QLIMIT1"})

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
                bypass_guards={"G_QLIMIT1"})

            self.assertIsNone(order)
            self.assertEqual(p.placed, [])

        with self.subTest("is_allowed_for_assetguardian_buy"):
            self._clear_state()
            class _ZeroPolicyProvider(_FakeProvider):
                def policy_cap_quantity(self, *args, **kwargs):
                    return 0.0

            p = _ZeroPolicyProvider()
            p.seed_trade("SELL", age_sec=400.0, price=100.0)
            order = self._inst(p).place(
                "BUY", 98.0, 1.0, smart=False, caller_owns_retry=True,
                bypass_guards={"G_QLIMIT1"}, module="assetguardian")

            self.assertIsNotNone(order)
            self.assertEqual(len(p.placed), 1)
            self.assertEqual(p.placed[0][3], 1.0)

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
                bypass_guards={"G_QLIMIT1"})

            self.assertIsNotNone(order)
            self.assertEqual(p.placed[0][3], 0.9)

    def test_intent_lifecycle_persistence_and_response_loss(self):
        with self.subTest("first_order_allowed_and_logged"):
            self._clear_state()
            p = _FakeProvider()
            inst = self._inst(p)
            order = inst.place("BUY", 100.0, 1.0)
            self.assertIsNotNone(order)
            self.assertEqual(len(p.placed), 1)
            lines = self._log_lines()
            self.assertTrue(any("|accepted|" in l and SYMBOL in l for l in lines), lines)

        with self.subTest("intent_persisted_with_same_client_id_before_submit"):
            self._clear_state()
            import order_retry as oq

            class _InspectProvider(_FakeProvider):
                def place_order(self, symbol, side, price, qty, **kwargs):
                    queued = oq.load_all()
                    self.persisted_during_submit = dict(queued[0])
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

        with self.subTest("truthy_payload_without_order_id_is_unknown"):
            self._clear_state()
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

        with self.subTest("submit_response_loss_keeps_pre_submit_intent"):
            self._clear_state()
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

    # ── rapid-fire order placement without trade_cooldown lock ──────────────────
    def test_cooldown_behaviors(self):
        with self.subTest("orders_not_blocked_by_cooldown_lock"):
            self._clear_state()
            p = _FakeProvider()
            inst = self._inst(p)
            first = inst.place("BUY", 100.0, 1.0)
            self.assertIsNotNone(first)
            second = inst.place("BUY", 101.0, 1.0)
            self.assertIsNotNone(second)
            self.assertEqual(len(p.placed), 2)

        with self.subTest("orders_independent_per_symbol"):
            self._clear_state()
            p = _FakeProvider()
            inst_a = Instrument(name="A", symbol="ZZZFAKEUSD_A", provider=p.name.lower(),
                                base="A", quote="USD", api=MarketApi([p]))
            inst_b = Instrument(name="B", symbol="ZZZFAKEUSD_B", provider=p.name.lower(),
                                base="B", quote="USD", api=MarketApi([p]))
            self.assertIsNotNone(inst_a.place("BUY", 100.0, 1.0))
            self.assertIsNotNone(
                inst_b.place("BUY", 100.0, 1.0))

        with self.subTest("pair_id_orders_allowed"):
            self._clear_state()
            p = _FakeProvider()
            inst = self._inst(p)

            buy = inst.place(
                "BUY", 99.0, 1.0, smart=False, wait_for_trend=False,
                bypass_guards={"profit_guard"},
                cooldown_pair_id="pair-1", caller_owns_retry=True)
            sell = inst.place(
                "SELL", 101.0, 1.0, smart=False, wait_for_trend=False,
                bypass_guards={"profit_guard"},
                cooldown_pair_id="pair-1", caller_owns_retry=True)
            buy2 = inst.place(
                "BUY", 98.0, 1.0, smart=False, wait_for_trend=False,
                bypass_guards={"profit_guard"},
                cooldown_pair_id="pair-1", caller_owns_retry=True)

            self.assertIsNotNone(buy)
            self.assertIsNotNone(sell)
            self.assertIsNotNone(buy2)
            self.assertEqual(len(p.placed), 3)

    def test_facade_and_placement_flags_routing(self):
        with self.subTest("facade_place_routes_through_pipeline"):
            self._clear_state()
            p = _FakeProvider()
            p.seed_trade("SELL", age_sec=5.0, price=90.0)  # Profit guard trips on BUY above recent SELL
            mkt = MarketApi([p])
            order = mkt.place(SYMBOL, "BUY", 100.0, 1.0)
            self.assertIsNone(order)
            self.assertEqual(len(p.placed), 0)

        with self.subTest("smart_flag_gates_price_adjust"):
            self._clear_state()
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
            inst2 = Instrument(name="ZZZFAKE2", symbol="ZZZFAKEUSD2", provider=p.name.lower(),
                               base="ZZZFAKE2", quote="USD", api=MarketApi([p]))
            inst2.place("BUY", 100.0, 1.0, smart=False)
            self.assertEqual(calls, [], "smart=False must NOT call adjust_order_price")

        with self.subTest("cancelorders_and_hours_reach_quantity_hook"):
            self._clear_state()
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

    def test_retry_enqueue_and_tracking_lifecycle(self):
        with self.subTest("failed_order_enqueued_for_retry"):
            self._clear_state()
            import order_retry as _oq
            p = _FakeProvider()
            p.seed_trade("SELL", age_sec=5.0, price=90.0)
            inst = self._inst(p)
            order = inst.place("BUY", 100.0, 1.0)
            self.assertIsNone(order)
            q = _oq.load_all()
            self.assertEqual(len(q), 1)
            self.assertEqual(q[0]["symbol"], SYMBOL)
            self.assertEqual(q[0]["side"], "BUY")

        with self.subTest("retry_flag_prevents_reenqueue"):
            self._clear_state()
            import order_retry as _oq
            p = _FakeProvider()
            p.seed_trade("SELL", age_sec=5.0, price=90.0)
            inst = self._inst(p)
            order = inst.place("BUY", 100.0, 1.0, caller_owns_retry=True)
            self.assertIsNone(order)
            self.assertEqual(_oq.load_all(), [])

        with self.subTest("success_remains_tracked_until_terminal"):
            self._clear_state()
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

        with self.subTest("success_does_not_remove_independent_same_side_intent"):
            self._clear_state()
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
                bypass_guards={"profit_guard"})
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
                bypass_guards={"profit_guard"}, _outcome_context=outcome)

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
                bypass_guards={"profit_guard"}, wait_for_trend=False,
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

    def test_instrument_place_reuses_single_regime_context_and_logs_telemetry(self):
        p = _FakeProvider(price=100.0)
        p.seed_trade("BUY", age_sec=400.0, price=100.0)
        inst = self._inst(p)

        mock_decision = order_guard.MarketRegimeDecision(
            regime="bull", gradient=0.75, epsilon=0.15, strength=5.0,
            fresh=True, reason="directional_signal", source="mock_src",
        )
        ctx = order_guard.MarketRegimeContext.from_decision(mock_decision, evaluated_at=time.time())

        with patch.object(inst._api, "market_regime_context", return_value=ctx) as mock_ctx, \
                patch.object(p, "profit_guard_window_ref", return_value=None) as mock_window_ref, \
                patch("order_guard.profit_guard", return_value=True) as mock_profit_guard:
            order = inst.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False)

        self.assertIsNotNone(order)
        mock_ctx.assert_called_once_with(inst.symbol, provider_name=inst.provider_name)
        mock_window_ref.assert_called_once_with(inst.symbol, "BUY", None, regime_context=ctx)
        mock_profit_guard.assert_called_once_with(
            p, inst.symbol, "BUY", 100.0, ANY, window_ref=None, regime_context=ctx, qty=1.0
        )
        import order_retry as oq
        queued = oq.load_all()
        self.assertEqual(len(queued), 1)
        self.assertNotIn("regime_context", queued[0]["place_kwargs"])
        self.assertNotIn("regime_context", p.placed[0][4])
        json.dumps(queued[0], allow_nan=False)
        lines = self._log_lines()
        self.assertTrue(any("|bull|5.0000|mock_src|True|False|directional_signal" in l for l in lines), lines)

    def test_market_dispatch_reuses_context_even_when_final_guard_refuses(self):
        ctx = order_guard.MarketRegimeContext.from_decision(
            order_guard.MarketRegimeDecision(
                regime="bull", gradient=0.75, epsilon=0.15, strength=5.0,
                fresh=True, reason="directional_signal"))
        for final_allowed in (True, False):
            with self.subTest(final_allowed=final_allowed):
                self._clear_state()
                provider = _FakeProvider(price=100.0)
                inst = self._inst(provider)
                with patch.object(
                        inst._api, "market_regime_context", return_value=ctx) as resolve, \
                        patch.object(
                            provider, "profit_guard_window_ref", return_value=None) as window, \
                        patch("order_guard.profit_guard",
                              side_effect=[True, final_allowed]) as guard:
                    result = inst.place(
                        "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                        force=True, caller_owns_retry=True)
                resolve.assert_called_once_with(
                    SYMBOL, provider_name=inst.provider_name)
                window.assert_called_once_with(
                    SYMBOL, "BUY", None, regime_context=ctx)
                self.assertEqual(guard.call_count, 2)
                for call in guard.call_args_list:
                    self.assertIs(call.kwargs["regime_context"], ctx)
                self.assertEqual(result is not None, final_allowed)
                self.assertEqual(len(provider.placed), int(final_allowed))

    def test_profit_guard_not_called_when_quantity_refused(self):
        """Ensure post-sizing profit_guard (and LLM HighStakeGuard) never runs when 24h weight policy refuses qty."""
        self._clear_state()

        class _ZeroWeightProvider(_FakeProvider):
            def policy_cap_quantity(self, *args, **kwargs):
                return 0.0

        provider = _ZeroWeightProvider(price=100.0)
        inst = self._inst(provider)
        with patch("order_guard.profit_guard", return_value=True) as guard:
            result = inst.place("BUY", 100.0, 10.0, smart=False, wait_for_trend=False, caller_owns_retry=True)
        self.assertIsNone(result)
        guard.assert_not_called()

    def test_window_reference_hook_compatibility_and_internal_type_error(self):
        provider = _FakeProvider()
        inst = self._inst(provider)
        context = object()
        calls = []

        def legacy(symbol, side, seconds):
            calls.append((symbol, side, seconds, None))
            return 42.0

        def keyword_only(symbol, side, seconds, *, regime_context=None):
            calls.append((symbol, side, seconds, regime_context))
            return 42.0

        def keywords(symbol, side, seconds, **kwargs):
            calls.append((symbol, side, seconds, kwargs["regime_context"]))
            return 42.0

        def positional_only(symbol, side, seconds, regime_context=None, /):
            calls.append((symbol, side, seconds, regime_context))
            return 42.0

        def broken(symbol, side, seconds, *, regime_context=None):
            calls.append((symbol, side, seconds, regime_context))
            raise TypeError("provider hook failed internally")

        class OpaqueLegacyHook:
            __signature__ = object()

            def __call__(self, symbol, side, seconds):
                return legacy(symbol, side, seconds)

        cases = (
            ("legacy", legacy, None),
            ("keyword_only", keyword_only, context),
            ("kwargs", keywords, context),
            ("positional_only", positional_only, None),
            ("opaque_legacy", OpaqueLegacyHook(), None),
            ("internal_error", broken, context),
        )
        for name, hook, expected in cases:
            with self.subTest(name):
                calls.clear()
                with patch.object(provider, "profit_guard_window_ref", hook):
                    if name == "internal_error":
                        with self.assertRaisesRegex(TypeError, "failed internally"):
                            inst._call_profit_guard_window_ref(
                                SYMBOL, "BUY", 300, regime_context=context)
                    else:
                        self.assertEqual(inst._call_profit_guard_window_ref(
                            SYMBOL, "BUY", 300, regime_context=context), 42.0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][:3], (SYMBOL, "BUY", 300))
                self.assertIs(calls[0][3], expected)

    def test_supplied_context_stays_ephemeral_and_retry_recomputes_it(self):
        import order_retry as oq
        import order_retry_worker as worker

        old = order_guard.MarketRegimeContext.from_decision(
            order_guard.MarketRegimeDecision(
                regime="bull", gradient=0.5, epsilon=0.1, strength=5.0,
                fresh=True, reason="original_decision"))
        new = order_guard.MarketRegimeContext.from_decision(
            order_guard.MarketRegimeDecision(
                regime="bear", gradient=-0.5, epsilon=0.1, strength=5.0,
                fresh=True, reason="retry_decision"))
        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                self._clear_state()
                provider = _FakeProvider()
                inst = self._inst(provider)
                with patch.object(inst._api, "market_regime_context") as resolve, \
                        patch("order_guard.profit_guard", return_value=accepted) as guard:
                    result = inst.place(
                        "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                        regime_context=old)
                resolve.assert_not_called()
                self.assertIs(guard.call_args.kwargs["regime_context"], old)
                self.assertEqual(result is not None, accepted)
                queued = oq.load_all()
                self.assertEqual(len(queued), 1)
                self.assertNotIn("regime_context", queued[0]["place_kwargs"])
                json.dumps(queued[0], allow_nan=False)
                if not accepted:
                    with patch.object(
                            inst._api, "market_regime_context", return_value=new) as resolve, \
                            patch.object(provider, "profit_guard_window_ref") as window, \
                            patch("order_guard.profit_guard", return_value=True) as guard:
                        stats = worker.process_once(inst._api, now=time.time() + 1000.0)
                    self.assertEqual(stats["attempted"], 1)
                    self.assertEqual(stats["succeeded"], 1)
                    resolve.assert_called_once_with(
                        SYMBOL, provider_name=provider.name)
                    self.assertIs(window.call_args.kwargs["regime_context"], new)
                    self.assertIs(guard.call_args.kwargs["regime_context"], new)
                    self.assertEqual(len(oq.load_all()), 1)
                    self.assertEqual(oq.load_all()[0]["lifecycle"], "accepted")


    def test_bypass_guards_hierarchical_resolution(self):
        """Verify bypass_guards resolves from call kwargs, instrument params, and order_guard.conf."""
        self._clear_state()
        provider = _FakeProvider(name="FakeVenue")

        # 1. Kwargs priority
        inst = self._inst(provider, params={"bypass_guards": "math_reference"})
        with patch("order_guard.profit_guard", return_value=True) as guard:
            inst.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False, bypass_guards="GO1")
            self.assertIn("bypass_guards", guard.call_args.kwargs)
            self.assertIn("daily_limit", guard.call_args.kwargs["bypass_guards"])

        # 2. Instrument params fallback
        inst_params = self._inst(provider, params={"bypass_guards": "G_QLIMIT2,GI_OBOOK"})
        with patch("order_guard.profit_guard", return_value=True) as guard:
            inst_params.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False)
            self.assertIn("bypass_guards", guard.call_args.kwargs)
            self.assertIn("g_qlimit2", guard.call_args.kwargs["bypass_guards"])
            self.assertIn("gi_obook", guard.call_args.kwargs["bypass_guards"])

        # 3. order_guard.conf venue fallback
        inst_conf = self._inst(provider, params={})
        with patch.dict(order_guard._load_margins(), {"fakevenue_bypass_guards": "GLLM1"}):
            with patch("order_guard.profit_guard", return_value=True) as guard:
                inst_conf.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False)
                self.assertIn("bypass_guards", guard.call_args.kwargs)
                self.assertIn("gllm1", guard.call_args.kwargs["bypass_guards"])

    def test_instrument_profit_margin_override_and_allow_math_reference_sell(self):
        """Verify custom profit_margin and allow_math_reference_sell on SELL."""
        self._clear_state()
        provider = _FakeProvider(name="FakeVenue")
        inst = self._inst(provider)

        # 1. Custom profit_margin passed to place()
        with patch("order_guard.profit_guard", return_value=True) as guard:
            inst.place("SELL", 100.0, 1.0, smart=False, wait_for_trend=False, profit_margin=0.15)
            # The 5th positional argument to profit_guard is profit_margin
            self.assertEqual(guard.call_args[0][4], 0.15)

        # 2. Standard SELL with math_reference bypass is discarded without permission
        with patch("order_guard.profit_guard", return_value=True) as guard:
            inst.place("SELL", 100.0, 1.0, smart=False, wait_for_trend=False, bypass_guards=["math_reference"])
            self.assertNotIn("math_reference", guard.call_args.kwargs.get("bypass_guards", set()))

        # 3. SELL with allow_math_reference_sell=True retains math_reference in bypass_guards
        with patch("order_guard.profit_guard", return_value=True) as guard:
            inst.place(
                "SELL", 100.0, 1.0, smart=False, wait_for_trend=False,
                bypass_guards=["math_reference"], allow_math_reference_sell=True,
            )
            self.assertIn("math_reference", guard.call_args.kwargs["bypass_guards"])

    def test_composite_profit_guard_bypass_does_not_bypass_intelligence(self):
        """Verify bypass_guards={'profit_guard'} bypasses only G_PROF & G_QLIMIT1, not intelligence."""
        from instrument import normalize_bypassed_guards
        bypassed = normalize_bypassed_guards("profit_guard")
        self.assertIn("math_reference", bypassed)
        self.assertIn("math_profit", bypassed)
        self.assertIn("profit_margin", bypassed)
        self.assertIn("quantity_policy", bypassed)
        self.assertNotIn("all", bypassed)

    def test_emergency_stop_loss_bypasses_cooldown(self):
        """Instrument.place no longer blocks on trade_cooldown for BUY or SELL."""
        prov = _FakeProvider()
        inst = self._inst(prov)

        class BlockedSlot:
            allowed = False
            info = {"side": "BUY", "timestamp": time.time() - 5}
            def commit(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("lock.trade_cooldown.trade_slot", return_value=BlockedSlot()) as mock_slot, \
             patch("order_guard.daily_limit_guard", return_value=(True, "ok")), \
             patch("order_guard.profit_guard", return_value=True):
            res_buy = inst.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False)
            self.assertIsNotNone(res_buy)

            res_normal = inst.place("SELL", 100.0, 1.0, smart=False, wait_for_trend=False)
            self.assertIsNotNone(res_normal)

            res_stop = inst.place("SELL", 100.0, 1.0, smart=False, wait_for_trend=False, kind="STOP")
            self.assertIsNotNone(res_stop)

            res_all = inst.place("SELL", 100.0, 1.0, smart=False, wait_for_trend=False, bypass_guards={"all"})
            self.assertIsNotNone(res_all)

            mock_slot.assert_not_called()

    def test_fine_grained_bypass_guards_granularity(self):
        """Verify fine-grained bypass combinations and ensure non-bypassed guards apply by default."""
        from instrument import normalize_bypassed_guards, GUARD_CATALOG

        # 0. GUARD_CATALOG completeness (new canonical codes exist, legacy G1..G10 removed)
        canonical_codes = {
            "GO1", "GT_S", "GT_M", "GT_L",
            "G_PROF", "G_QLIMIT1", "G_QLIMIT2",
            "GI_OBOOK", "GI_WHALE", "GI_FUNDING",
            "GLLM1", "GLLM2", "GLLM3",
        }
        legacy_codes = {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"}
        self.assertTrue(canonical_codes.issubset(set(GUARD_CATALOG.keys())))
        self.assertTrue(legacy_codes.isdisjoint(set(GUARD_CATALOG.keys())))

        # 1. Default (None) bypasses nothing
        none_bypassed = normalize_bypassed_guards(None)
        self.assertEqual(none_bypassed, set())

        # 2. Canonical code resolution (e.g. GO1 + GT_S + G_PROF)
        g_bypassed = normalize_bypassed_guards(["GO1", "GT_S", "G_PROF"])
        self.assertIn("go1", g_bypassed)
        self.assertIn("daily_limit", g_bypassed)
        self.assertIn("gt_s", g_bypassed)
        self.assertIn("short_trend", g_bypassed)
        self.assertIn("g_prof", g_bypassed)
        self.assertIn("math_profit", g_bypassed)
        self.assertIn("math_reference", g_bypassed)
        self.assertNotIn("quantity_policy", g_bypassed)
        self.assertNotIn("g_qlimit1", g_bypassed)
        self.assertNotIn("medium_trend", g_bypassed)
        self.assertNotIn("gt_m", g_bypassed)
        self.assertNotIn("g_qlimit2", g_bypassed)

        # 3. Specific opt-out: only daily_limit and short_trend
        dt_bypassed = normalize_bypassed_guards("daily_limit, short_trend")
        self.assertIn("daily_limit", dt_bypassed)
        self.assertIn("go1", dt_bypassed)
        self.assertIn("short_trend", dt_bypassed)
        self.assertNotIn("medium_trend", dt_bypassed)
        self.assertNotIn("math_reference", dt_bypassed)
        self.assertNotIn("quantity_policy", dt_bypassed)
        self.assertNotIn("g_qlimit2", dt_bypassed)
        self.assertNotIn("gi_obook", dt_bypassed)
        self.assertNotIn("gllm1", dt_bypassed)
        self.assertNotIn("gllm3", dt_bypassed)

        # 4. Unrecognized tokens are ignored without crashing or breaking valid guards
        mixed_bypassed = normalize_bypassed_guards(["GO1", "invalid_foobar_token", "short_trend"])
        self.assertIn("daily_limit", mixed_bypassed)
        self.assertIn("short_trend", mixed_bypassed)
        self.assertNotIn("invalid_foobar_token", mixed_bypassed)
        self.assertNotIn("math_profit", mixed_bypassed)

        # 5. Specific opt-out: only G_QLIMIT2 and GLLM1
        p_bypassed = normalize_bypassed_guards(["G_QLIMIT2", "GLLM1"])
        self.assertIn("g_qlimit2", p_bypassed)
        self.assertIn("internal_quant_limit", p_bypassed)
        self.assertIn("quantitative", p_bypassed)
        self.assertIn("gllm1", p_bypassed)
        self.assertIn("llm_high_stake", p_bypassed)
        self.assertIn("llm", p_bypassed)
        self.assertNotIn("gi_obook", p_bypassed)
        self.assertNotIn("gllm3", p_bypassed)
        self.assertNotIn("daily_limit", p_bypassed)
        self.assertNotIn("math_reference", p_bypassed)

        # 6. 'all' or '*' bypasses every gate
        all_bypassed = normalize_bypassed_guards("all")
        self.assertIn("all", all_bypassed)
        self.assertIn("daily_limit", all_bypassed)
        self.assertIn("short_trend", all_bypassed)
        self.assertIn("medium_trend", all_bypassed)
        self.assertIn("long_trend", all_bypassed)
        self.assertIn("math_profit", all_bypassed)
        self.assertIn("math_reference", all_bypassed)
        self.assertIn("quantity_policy", all_bypassed)
        self.assertIn("g_qlimit2", all_bypassed)
        self.assertIn("gi_obook", all_bypassed)
        self.assertIn("gi_whale", all_bypassed)
        self.assertIn("gi_funding", all_bypassed)
        self.assertIn("gllm1", all_bypassed)
        self.assertIn("gllm2", all_bypassed)
        self.assertIn("gllm3", all_bypassed)

        # 7. Runtime execution: non-bypassed guards apply by default in Instrument.place
        prov = _FakeProvider()
        inst = self._inst(prov)

        with patch("order_guard.daily_limit_guard", return_value=(True, None)) as mock_daily, \
             patch("order_guard.profit_guard", return_value=True) as mock_profit:

            # 7a. Default: both daily_limit and profit_guard are called
            inst.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False)
            mock_daily.assert_called_once()
            mock_profit.assert_called()

        with patch("order_guard.daily_limit_guard", return_value=(True, None)) as mock_daily, \
             patch("order_guard.profit_guard", return_value=True) as mock_profit:

            # 7b. Bypassing only GO1 (daily_limit): daily_limit is NOT called, profit_guard IS called
            inst.place("BUY", 100.0, 1.0, smart=False, wait_for_trend=False, bypass_guards={"GO1"})
            mock_daily.assert_not_called()
            mock_profit.assert_called()

    def test_guard_hierarchy_and_categories(self):
        """Verify object-oriented guard hierarchy, category grouping, and is_bypassed behavior."""
        from guard_hierarchy import (
            BaseGuard,
            OperationalGuard,
            TrendGuard,
            ProfitGuard,
            QuantLimitGuard,
            FinancialGuard,
            IntelligenceGuard,
            LLMGuard,
            GUARD_REGISTRY,
            GUARD_CATEGORIES,
            normalize_bypassed_guards,
        )

        # Hierarchy validation
        self.assertIsInstance(GUARD_REGISTRY["GO1"], OperationalGuard)
        self.assertIsInstance(GUARD_REGISTRY["GT_S"], TrendGuard)
        self.assertIsInstance(GUARD_REGISTRY["GT_M"], TrendGuard)
        self.assertIsInstance(GUARD_REGISTRY["GT_L"], TrendGuard)
        self.assertIsInstance(GUARD_REGISTRY["G_PROF"], ProfitGuard)
        self.assertIsInstance(GUARD_REGISTRY["G_PROF"], FinancialGuard)
        self.assertIsInstance(GUARD_REGISTRY["G_QLIMIT1"], QuantLimitGuard)
        self.assertIsInstance(GUARD_REGISTRY["G_QLIMIT2"], QuantLimitGuard)
        self.assertIsInstance(GUARD_REGISTRY["GI_OBOOK"], IntelligenceGuard)
        self.assertIsInstance(GUARD_REGISTRY["GI_WHALE"], IntelligenceGuard)
        self.assertIsInstance(GUARD_REGISTRY["GI_FUNDING"], IntelligenceGuard)
        self.assertIs(GUARD_REGISTRY["GI_CROWD"], GUARD_REGISTRY["GI_FUNDING"])
        self.assertIsInstance(GUARD_REGISTRY["GLLM1"], LLMGuard)
        self.assertIsInstance(GUARD_REGISTRY["GLLM2"], LLMGuard)
        self.assertIsInstance(GUARD_REGISTRY["GLLM3"], LLMGuard)

        for g in GUARD_REGISTRY.values():
            self.assertIsInstance(g, BaseGuard)
            self.assertTrue(g.code.startswith("G"))
            self.assertTrue(bool(g.description))
            self.assertTrue(bool(g.canonical))

        # Categories mapping
        self.assertEqual(len(GUARD_CATEGORIES["trend"]), 3)
        self.assertEqual(len(GUARD_CATEGORIES["profit"]), 1)
        self.assertEqual(len(GUARD_CATEGORIES["qlimit"]), 2)
        self.assertEqual(len(GUARD_CATEGORIES["intelligence"]), 3)
        self.assertEqual(len(GUARD_CATEGORIES["llm"]), 3)

        # Category bypass: 'trend' bypasses GT_S, GT_M & GT_L, but not G_PROF or G_QLIMIT2
        trend_bypassed = normalize_bypassed_guards("trend")
        self.assertTrue(GUARD_REGISTRY["GT_S"].is_bypassed(trend_bypassed))
        self.assertTrue(GUARD_REGISTRY["GT_M"].is_bypassed(trend_bypassed))
        self.assertTrue(GUARD_REGISTRY["GT_L"].is_bypassed(trend_bypassed))
        self.assertFalse(GUARD_REGISTRY["G_PROF"].is_bypassed(trend_bypassed))
        self.assertFalse(GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(trend_bypassed))

        # Category bypass: 'qlimit' (or 'g_qlimit') bypasses G_QLIMIT1 & G_QLIMIT2
        qlim_bypassed = normalize_bypassed_guards("g_qlimit")
        self.assertFalse(GUARD_REGISTRY["GO1"].is_bypassed(qlim_bypassed))
        self.assertFalse(GUARD_REGISTRY["G_PROF"].is_bypassed(qlim_bypassed))
        self.assertTrue(GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(qlim_bypassed))
        self.assertTrue(GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(qlim_bypassed))

        # Category bypass: 'intelligence' (or 'gi') bypasses GI_OBOOK, GI_WHALE, GI_FUNDING
        intel_bypassed = normalize_bypassed_guards("gi")
        self.assertFalse(GUARD_REGISTRY["GO1"].is_bypassed(intel_bypassed))
        self.assertFalse(GUARD_REGISTRY["G_PROF"].is_bypassed(intel_bypassed))
        self.assertTrue(GUARD_REGISTRY["GI_OBOOK"].is_bypassed(intel_bypassed))
        self.assertTrue(GUARD_REGISTRY["GI_WHALE"].is_bypassed(intel_bypassed))
        self.assertTrue(GUARD_REGISTRY["GI_FUNDING"].is_bypassed(intel_bypassed))
        self.assertFalse(GUARD_REGISTRY["GLLM1"].is_bypassed(intel_bypassed))

        # Category bypass: 'llm' (or 'gllm') bypasses GLLM1, GLLM2, GLLM3
        llm_bypassed = normalize_bypassed_guards("gllm")
        self.assertTrue(GUARD_REGISTRY["GLLM1"].is_bypassed(llm_bypassed))
        self.assertTrue(GUARD_REGISTRY["GLLM2"].is_bypassed(llm_bypassed))
        self.assertTrue(GUARD_REGISTRY["GLLM3"].is_bypassed(llm_bypassed))

    def test_medium_trend_guard_runtime(self):
        """Verify GT_M (medium_trend) gate runtime actions (defer, scale, bypass)."""
        from market_regime import MarketRegimeContext, MarketRegimeDecision

        bear_ctx = MarketRegimeContext.from_decision(
            MarketRegimeDecision(
                regime="bear", gradient=-0.8, epsilon=0.1, strength=5.0,
                fresh=True, reason="directional_bear",
            )
        )
        bull_ctx = MarketRegimeContext.from_decision(
            MarketRegimeDecision(
                regime="bull", gradient=0.8, epsilon=0.1, strength=5.0,
                fresh=True, reason="directional_bull",
            )
        )

        prov = _FakeProvider()
        inst = self._inst(prov)

        with patch("order_guard.daily_limit_guard", return_value=(True, None)), \
             patch("order_guard.profit_guard", return_value=True):

            # 1. Action = 'defer' on BUY during bear: returns None with reason 'medium_trend_deferred'
            res = inst.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                regime_context=bear_ctx, medium_trend_action="defer",
            )
            self.assertIsNone(res)

            # 2. Action = 'defer' on BUY during bear but bypassed with GT_M: passes through!
            res_bypassed = inst.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                regime_context=bear_ctx, medium_trend_action="defer",
                bypass_guards={"GT_M"},
            )
            self.assertIsNotNone(res_bypassed)

            # 3. Action = 'defer' on BUY during bull: passes through
            res_bull = inst.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                regime_context=bull_ctx, medium_trend_action="defer",
            )
            self.assertIsNotNone(res_bull)

            # 4. Action = 'scale' on BUY during bear: scales quantity down by medium_trend_scale (0.5)
            res_scaled = inst.place(
                "BUY", 100.0, 2.0, smart=False, wait_for_trend=False,
                regime_context=bear_ctx, medium_trend_action="scale",
                medium_trend_scale=0.5,
            )
            self.assertIsNotNone(res_scaled)
            self.assertEqual(prov.placed[-1][3], 1.0)

            # 5. Action = 'defer' on SELL without explicit regime_context resolves context lazily
            import order_retry as oq
            self.assertTrue(oq.is_non_failure_deferral("medium_trend_deferred"))
            outcome = {}
            with patch.object(inst._api, "market_regime_context", return_value=bull_ctx) as mock_ctx:
                res_sell_defer = inst.place(
                    "SELL", 105.0, 1.0, smart=False, wait_for_trend=False,
                    medium_trend_action="defer", _outcome_context=outcome,
                )
            self.assertIsNone(res_sell_defer)
            self.assertEqual(outcome.get("reason"), "medium_trend_deferred")
            mock_ctx.assert_called_once_with(inst.symbol, provider_name=inst.provider_name)

    def test_aborted_by_caller_and_invalid_price_never_enqueued_and_ttl_sec_forwarded(self):
        import order_retry as oq
        prov = _FakeProvider()
        inst = self._inst(prov)

        with patch("order_guard.daily_limit_guard", return_value=(True, None)), \
             patch("order_guard.profit_guard", return_value=True):
            # 1. abort_if triggers on first check (before prequeued_claim) -> never enqueued
            oq.rewrite([])
            out1 = {}
            res1 = inst.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                abort_if=lambda: True, _outcome_context=out1,
            )
            self.assertIsNone(res1)
            self.assertEqual(out1.get("reason"), "aborted_by_caller")
            self.assertEqual(oq.load_all(), [])

            # 2. abort_if triggers on second check (after prequeued_claim) -> claim completed as terminal success, removed from queue
            calls = {"n": 0}
            def abort_on_second():
                calls["n"] += 1
                return calls["n"] >= 2

            out2 = {}
            res2 = inst.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                abort_if=abort_on_second, _outcome_context=out2,
            )
            self.assertIsNone(res2)
            self.assertEqual(out2.get("reason"), "aborted_by_caller")
            self.assertEqual(oq.load_all(), [])

            # 3. invalid_price on BUY -> never enqueued
            out3 = {}
            res3 = inst.place(
                "BUY", -5.0, 1.0, smart=False, wait_for_trend=False,
                _outcome_context=out3,
            )
            self.assertIsNone(res3)
            self.assertEqual(out3.get("reason"), "invalid_price")
            self.assertEqual(oq.load_all(), [])

        # 4. Pre-claim fallback enqueue forwards ttl_sec
        with patch("order_guard.daily_limit_guard", return_value=(True, None)), \
             patch("order_guard.profit_guard", return_value=False):
            oq.rewrite([])
            res4 = inst.place(
                "BUY", 100.0, 1.0, smart=False, wait_for_trend=False,
                ttl_sec=1234.0,
            )
            self.assertIsNone(res4)
            queued = oq.load_all()
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0].get("ttl_sec"), 1234.0)

        # 5. Pre-sizing refusal (Stage 1 G_PROF or GO1 daily_limit) with auto-qty (qty=None)
        # symmetrically on BUY and SELL must not raise UnboundLocalError on applied_scale.
        import contextlib
        import io
        for side, blocked_px, seed_side, seed_px in (
            ("SELL", 98.0, "BUY", 100.0),
            ("BUY", 102.0, "SELL", 100.0),
        ):
            self._clear_state()
            p_sym = _FakeProvider(price=blocked_px)
            p_sym.seed_trade(seed_side, age_sec=400.0, price=seed_px)
            inst_sym = self._inst(p_sym)
            buf = io.StringIO()
            out_sym = {}
            with contextlib.redirect_stdout(buf):
                res_sym = inst_sym.place(
                    side, blocked_px, None, smart=False, wait_for_trend=False,
                    applied_scale=0.5, _outcome_context=out_sym,
                )
            stdout_text = buf.getvalue()
            self.assertIsNone(res_sym)
            self.assertEqual(out_sym.get("reason"), "profit_guard")
            self.assertNotIn("enqueue retry failed", stdout_text)
            self.assertNotIn("applied_scale", stdout_text)
            self.assertEqual(oq.load_all(), [])

    def test_gt_s_ignores_bypass_guards_and_stop_loss_module_owns_timer(self):
        """GT_S cannot be bypassed by bypass_guards ('GT_S' or 'all') in Instrument.place,
        and Instrument.place contains no is_stop timer logic (managed by PositionExitManager)."""
        from unittest.mock import MagicMock
        from strategies.position_exit_manager import PositionExitManager, PositionExitConfig, PositionSnapshot, ExitAction
        self._clear_state()
        p = _FakeProvider(price=100.0)
        inst = self._inst(p)

        fake_stm = MagicMock()
        fake_stm.should_wait.return_value = True

        with patch("cacheManager.get_short_trend_manager", return_value=fake_stm), \
             patch("order_guard.daily_limit_guard", return_value=(True, None)), \
             patch("order_guard.profit_guard", return_value=True):
            # 1. Caller requests bypass_guards={"GT_S"} or {"all"} -> ignored for GT_S, waits while trend is adverse
            out_buy = {}
            res_buy = inst.place(
                "BUY", 100.0, 1.0, smart=False,
                bypass_guards={"GT_S", "all"}, caller_owns_retry=True,
                _outcome_context=out_buy,
            )
            self.assertIsNone(res_buy)
            self.assertEqual(out_buy.get("reason"), "trend_deferred")

            out_sell = {}
            res_sell = inst.place(
                "SELL", 100.0, 1.0, smart=False,
                bypass_guards={"GT_S", "all"}, caller_owns_retry=True,
                _outcome_context=out_sell,
            )
            self.assertIsNone(res_sell)
            self.assertEqual(out_sell.get("reason"), "trend_deferred")

            # 2. Stop-Loss module (monitortrades) always passes wait_for_trend=True even on timeout;
            # only Trailing Stop sets wait_for_trend=False after its 7-minute (420s) crash timer expires.
            pem = PositionExitManager(PositionExitConfig(lost_threshold=0.05, sl_trend_defer_max_sec=300.0, smart_sl_enabled=False))
            t0 = 1000.0
            snap_init = PositionSnapshot(symbol=SYMBOL, current_price=94.0, avg_buy_price=100.0, net_qty=1.0, sellable_qty=1.0, current_time_s=t0, trend_up=True)
            dec_init = pem.evaluate_stop_loss(snap_init)
            self.assertEqual(dec_init.action, ExitAction.HOLD)

            # Still ascending at 120s (< 300s) -> HOLD
            snap_mid = PositionSnapshot(symbol=SYMBOL, current_price=94.1, avg_buy_price=100.0, net_qty=1.0, sellable_qty=1.0, current_time_s=t0 + 120.0, trend_up=True)
            dec_mid = pem.evaluate_stop_loss(snap_mid)
            self.assertEqual(dec_mid.action, ExitAction.HOLD)

            # Still ascending at 305s (>= 300s) -> PositionExitManager returns timeout SELL,
            # but monitortrades keeps wait_for_trend=True so GT_S continues to defer while trend is up!
            snap_timeout = PositionSnapshot(symbol=SYMBOL, current_price=94.2, avg_buy_price=100.0, net_qty=1.0, sellable_qty=1.0, current_time_s=t0 + 305.0, trend_up=True)
            dec_timeout = pem.evaluate_stop_loss(snap_timeout)
            self.assertEqual(dec_timeout.action, ExitAction.SELL)
            self.assertEqual(dec_timeout.metadata.get("sub_reason"), "timeout")
            out_mt_sl = {}
            res_mt_sl = inst.place(
                "SELL", 94.2, 1.0, smart=False,
                bypass_guards={"all"}, wait_for_trend=True,
                caller_owns_retry=True, _outcome_context=out_mt_sl,
            )
            self.assertIsNone(res_mt_sl)
            self.assertEqual(out_mt_sl.get("reason"), "trend_deferred")

            # Trailing Stop (wait_for_trend=False after 7m timer expires) releases emergency SELL
            res_ts_released = inst.place(
                "SELL", 94.2, 1.0, smart=False,
                bypass_guards={"all"}, wait_for_trend=False,
                caller_owns_retry=True,
            )
            self.assertIsNotNone(res_ts_released)
            self.assertEqual(len(p.placed), 1)


from order_lifecycle import (
    AtomicOrderRollbackGuard,
    extract_order_restore_params,
    restore_cancelled_orders,
)


class TestOrderRollback(unittest.TestCase):
    def test_extract_order_restore_params_binance_and_kraken(self):
        b_ord = {"orderId": "12345", "side": "SELL", "price": "150.25", "origQty": "2.5"}
        params = extract_order_restore_params(b_ord)
        self.assertEqual(params["side"], "SELL")
        self.assertEqual(params["limit"], 150.25)
        self.assertEqual(params["qty"], 2.5)
        self.assertEqual(params["kind"], "TP")

        k_ord = {"id": "OG-KRAKEN", "side": "sell", "price": 105.0, "vol": 10.0}
        params_k = extract_order_restore_params(k_ord)
        self.assertEqual(params_k["side"], "SELL")
        self.assertEqual(params_k["limit"], 105.0)
        self.assertEqual(params_k["qty"], 10.0)

    def test_extract_order_restore_params_t212(self):
        t_ord = {"id": "58805932163", "side": "SELL", "qty": 0.19, "limit": 176.31, "level": 11.0, "kind": "TP"}
        params = extract_order_restore_params(t_ord)
        self.assertEqual(params["side"], "SELL")
        self.assertEqual(params["limit"], 176.31)
        self.assertEqual(params["qty"], 0.19)
        self.assertEqual(params["level"], 11.0)
        self.assertEqual(params["kind"], "TP")

    def test_restore_cancelled_orders_skips_invalid_and_calls_restore(self):
        cancelled = [
            {"side": "SELL", "limit": 100.0, "qty": 1.0},
            {"side": "SELL", "limit": 0.0, "qty": 1.0},
            {"side": "SELL", "limit": 120.0, "qty": 0.0},
            {"side": "BUY", "limit": 90.0, "qty": 2.0, "kind": "ENTRY"},
        ]
        restored_calls = []

        def mock_restore(p):
            restored_calls.append(p)
            return True

        logs = []
        count = restore_cancelled_orders(cancelled, restore_fn=mock_restore, log_fn=logs.append, reason="test run")
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
        count = restore_cancelled_orders(cancelled, restore_fn=failing_restore, log_fn=logs.append, reason="test failure")
        self.assertEqual(count, 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("failed restoring order" in l for l in logs))

    def test_atomic_rollback_guard_committed(self):
        cancelled = [{"side": "SELL", "limit": 100.0, "qty": 1.0}]
        restored = []
        with AtomicOrderRollbackGuard(cancelled, restore_fn=restored.append) as guard:
            guard.commit()
        self.assertEqual(len(restored), 0)

    def test_atomic_rollback_guard_aborted_triggers_rollback(self):
        cancelled = [{"side": "SELL", "limit": 100.0, "qty": 1.0}]
        restored = []
        try:
            with AtomicOrderRollbackGuard(cancelled, restore_fn=restored.append, reason="crash"):
                raise ValueError("placement exploded")
        except ValueError:
            pass
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["limit"], 100.0)

    def test_extract_order_restore_params_prefers_remaining_qty(self):
        b_ord = {"orderId": "999", "side": "SELL", "price": "100.0", "origQty": "10.0", "remainingQty": "3.5"}
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


# ---------------------------------------------------------------------------
# Consolidated from tests/test_trade_cooldown.py & tests/test_active_intents.py
# ---------------------------------------------------------------------------

import multiprocessing as mp
from active_intents import build_active_intent_index, format_summary


class _PipePutAdapter:
    """Semaphore-free multiprocessing pipe adapter immune to systemd-logind RemoveIPC /dev/shm wipes."""

    def __init__(self, conn):
        self._conn = conn

    def put(self, item):
        try:
            self._conn.send(item)
        finally:
            self._conn.close()


def _proc_attempt(state_file, lock_file, q):
    tc.STATE_FILE = state_file
    tc.LOCK_FILE = lock_file
    ok, _ = tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
    q.put(ok)


class TestTradeCooldown(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        tc.STATE_FILE = os.path.join(self.tmp, "trade_cooldown.json")
        tc.LOCK_FILE = os.path.join(self.tmp, "trade_cooldown.lock")

    def test_first_allowed_second_blocked(self):
        ok, _ = tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
        self.assertTrue(ok)
        ok2, last = tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=180)
        self.assertFalse(ok2)
        self.assertEqual(last["side"], "BUY")

    def test_per_symbol_independent(self):
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])
        self.assertTrue(tc.reserve_trade("BUY", "TAOUSDC", cooldown_sec=180)[0])

    def test_allowed_after_cooldown(self):
        with mock.patch("lock.cooldown.time.time") as now:
            now.return_value = 1_000.0
            self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=1)[0])
            self.assertFalse(tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=1)[0])
            now.return_value = 1_001.1
            self.assertTrue(tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=1)[0])

    def test_release_unblocks(self):
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])
        tc.release_trade("BTCUSDC")
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_update_binance_order_id(self):
        tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
        tc.update_binance_order_id("BTCUSDC", 12345)
        self.assertIn("12345", tc.describe_last_trade("BTCUSDC"))

    def test_concurrent_threads_single_winner(self):
        results = []
        barrier = threading.Barrier(20)

        def attempt():
            barrier.wait()
            ok, _ = tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
            results.append(ok)

        threads = [threading.Thread(target=attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(1 for r in results if r), 1)

    def test_concurrent_processes_single_winner(self):
        ctx = mp.get_context("spawn")
        pipes = [ctx.Pipe(duplex=False) for _ in range(10)]
        procs = [
            ctx.Process(
                target=_proc_attempt,
                args=(tc.STATE_FILE, tc.LOCK_FILE, _PipePutAdapter(w_conn)),
            )
            for _, w_conn in pipes
        ]
        for p, (_, w_conn) in zip(procs, pipes):
            p.start()
            w_conn.close()
        for p in procs:
            p.join(timeout=15)
        results = []
        for r_conn, _ in pipes:
            try:
                self.assertTrue(r_conn.poll(5.0), "spawned worker did not report result")
                results.append(r_conn.recv())
            finally:
                r_conn.close()
        self.assertEqual(sum(1 for r in results if r), 1)

    def test_slot_commit_keeps_reservation(self):
        with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=180) as slot:
            self.assertTrue(slot.allowed)
            slot.commit(999)
        self.assertFalse(tc.reserve_trade("SELL", "BTCUSDC", cooldown_sec=180)[0])

    def test_slot_no_commit_auto_releases(self):
        with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=180) as slot:
            self.assertTrue(slot.allowed)
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_slot_exception_auto_releases(self):
        with self.assertRaises(RuntimeError):
            with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=180) as slot:
                self.assertTrue(slot.allowed)
                raise RuntimeError("placement crashed")
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_slot_blocked_does_not_release_existing(self):
        self.assertTrue(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])
        with tc.trade_slot("SELL", "BTCUSDC", cooldown_sec=180) as slot:
            self.assertFalse(slot.allowed)
        self.assertFalse(tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)[0])

    def test_get_last_trade_age(self):
        self.assertIsNone(tc.get_last_trade_age("BTCUSDC"))
        tc.reserve_trade("BUY", "BTCUSDC", cooldown_sec=180)
        age = tc.get_last_trade_age("BTCUSDC")
        self.assertIsNotNone(age)
        self.assertLess(age, 5)

    def test_pair_allows_one_buy_and_one_sell_inside_same_cooldown(self):
        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as buy:
            self.assertTrue(buy.allowed)
            buy.commit(101)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as sell:
            self.assertTrue(sell.allowed)
            sell.commit(102)

        state = tc._cooldown().get("TAOUSDC")
        self.assertCountEqual(state["group_committed"], ["BUY", "SELL"])
        self.assertEqual(state["group_results"]["BUY"]["binance_order_id"], 101)
        self.assertEqual(state["group_results"]["SELL"]["binance_order_id"], 102)

    def test_pair_blocks_duplicate_side_and_unrelated_group(self):
        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as buy:
            self.assertTrue(buy.allowed)
            buy.commit(101)

        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as duplicate:
            self.assertFalse(duplicate.allowed)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-2") as outsider:
            self.assertFalse(outsider.allowed)
        self.assertFalse(tc.reserve_trade("SELL", "TAOUSDC", cooldown_sec=180)[0])

    def test_failed_second_leg_rolls_back_only_that_leg(self):
        with tc.trade_slot("BUY", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as buy:
            buy.commit(101)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as sell:
            self.assertTrue(sell.allowed)

        state = tc._cooldown().get("TAOUSDC")
        self.assertEqual(state["group_members"], ["BUY"])
        self.assertEqual(state["group_committed"], ["BUY"])
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as retry:
            self.assertTrue(retry.allowed)

    def test_legacy_reservation_still_blocks_pair(self):
        self.assertTrue(tc.reserve_trade("BUY", "TAOUSDC", cooldown_sec=180)[0])
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as leg:
            self.assertFalse(leg.allowed)

    def test_canceled_pair_leg_can_be_replaced_without_opening_other_groups(self):
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as sell:
            sell.commit(102)
        self.assertTrue(tc.release_pair_leg("TAOUSDC", "pair-1", "SELL"))

        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-2") as outsider:
            self.assertFalse(outsider.allowed)
        with tc.trade_slot("SELL", "TAOUSDC", cooldown_sec=180, pair_id="pair-1") as replacement:
            self.assertTrue(replacement.allowed)
            replacement.commit(103)


def _write_intent_file(path, payload, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("\n".join(json.dumps(row) for row in payload) + "\n"
            if lines else json.dumps(payload))
    path.write_text(text, encoding="utf-8")


def test_normalizes_multiple_owners_without_writing(tmp_path):
    outbox = tmp_path / "cachedb/order_retry_queue.jsonl"
    t212 = tmp_path / "212trading/.state_NVDA_US_EQ.json"
    rtrade = tmp_path / "cachedb/rtrade_pairs_state.json"
    _write_intent_file(outbox, [{
        "intent_id": "out-1", "provider_name": "Binance", "symbol": "TAOUSDC",
        "side": "BUY", "qty": 2, "lifecycle": "submit_pending",
    }], lines=True)
    _write_intent_file(t212, {
        "pending_submit": {"intent_id": "t-1", "side": "SELL", "qty": 1,
                           "limit": 150, "submission_outcome": "unknown"},
    })
    _write_intent_file(rtrade, {"pairs": {"pair-1": {
        "symbol": "BTCUSDC", "terminal": False,
        "intents": {"limit:BUY": {
            "intent_id": "r-1", "side": "BUY", "requested_qty": 0.1,
            "client_order_id": "CID-R", "order_id": "77",
        }},
    }}})
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (outbox, t212, rtrade)}

    result = build_active_intent_index(str(tmp_path))

    assert result["read_only"] is True
    assert {row["intent_id"] for row in result["intents"]} == {"out-1", "t-1", "r-1"}
    assert next(row for row in result["intents"] if row["intent_id"] == "t-1")["status"] == "unknown"
    assert next(row for row in result["intents"] if row["intent_id"] == "r-1")["symbol"] == "BTCUSDC"
    for path, snapshot in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == snapshot


def test_excludes_terminal_intents_and_reports_corrupt_sources(tmp_path):
    _write_intent_file(tmp_path / "cachedb/rtrade_pairs_state.json", {"pairs": {
        "done": {"terminal": True, "intents": {"x": {
            "intent_id": "done", "side": "SELL", "qty": 1,
        }}},
    }})
    corrupt = tmp_path / "cachedb/assetguardian_state.json"
    corrupt.write_text("{broken", encoding="utf-8")

    result = build_active_intent_index(str(tmp_path))

    assert result["intents"] == []
    assert result["errors"][0]["source"] == "cachedb/assetguardian_state.json"


def test_summary_calculates_pending_notional_and_net_exposures(tmp_path):
    outbox = tmp_path / "cachedb/order_retry_queue.jsonl"
    _write_intent_file(outbox, [
        {"intent_id": "buy-1", "symbol": "BTCUSDC", "side": "BUY", "qty": 0.5, "requested_price": 60000.0, "lifecycle": "pending"},
        {"intent_id": "sell-1", "symbol": "ETHUSDC", "side": "SELL", "qty": 2.0, "executed_qty": 0.5, "requested_price": 3000.0, "lifecycle": "pending"},
    ], lines=True)

    kraken = tmp_path / "kraken/.state_BTCUSDC.json"
    _write_intent_file(kraken, {
        "qty": 1.5, "cost": 90000.0, "entry_price": 60000.0, "entry_ts": 1700000000.0,
    })

    result = build_active_intent_index(str(tmp_path))

    summary = result["summary"]
    notional = summary["pending_notional"]
    assert notional["total_buy"] == 30000.0
    assert notional["total_sell"] == 4500.0
    assert notional["by_symbol"]["BTCUSDC"]["buy"] == 30000.0
    assert notional["by_symbol"]["ETHUSDC"]["sell"] == 4500.0

    exposures = summary["net_exposures"]
    assert exposures["total_cost"] == 90000.0
    assert len(exposures["positions"]) == 1
    assert exposures["positions"][0]["symbol"] == "BTCUSDC"
    assert exposures["positions"][0]["qty"] == 1.5

    formatted = format_summary(result)
    assert "Pending BUY notional:  $30000.00" in formatted
    assert "Pending SELL notional: $4500.00" in formatted
    assert "Total cost basis:      $90000.00" in formatted
    assert "BTCUSDC: 1.5 ($90000.00) [kraken]" in formatted


def test_filters_backup_files_and_recognizes_id_fallback(tmp_path):
    bak = tmp_path / "hyperliquid/.state_HYPE.pre_fee_reconcile.json"
    _write_intent_file(bak, {"qty": 999.0, "cost": 999.0})

    t212 = tmp_path / "212trading/.state_TEST_US_EQ.json"
    _write_intent_file(t212, {
        "qty": 10.0,
        "cost_usd": 500.0,
        "entry_price": 50.0,
        "orders": [
            {"id": "t212-order-99", "side": "SELL", "qty": 10.0, "limit": 60.0}
        ]
    })

    result = build_active_intent_index(str(tmp_path))
    assert not any(exp["symbol"] == "HYPE.pre_fee_reconcile" for exp in result["exposures"])
    assert any(it["intent_id"] == "t212-order-99" and it["requested_price"] == 60.0 for it in result["intents"])
    assert any(exp["symbol"] == "TEST_US_EQ" and exp["qty"] == 10.0 for exp in result["exposures"])


def test_normalizes_kraken_vol_t212_limit_price_and_nested_outbox_client_order_id(tmp_path):
    kraken = tmp_path / "kraken/.state_BTCUSDC.json"
    t212_one_shot = tmp_path / "212trading/.t212_order.NVDA_US_EQ.json"
    outbox = tmp_path / "cachedb/order_retry_queue.jsonl"
    _write_intent_file(kraken, {
        "orders": [{
            "txid": "O-KRK-1",
            "intent_id": "krk-intent-1",
            "side": "buy",
            "vol": 0.25,
            "exec_vol": 0.10,
            "price": 60000.0,
        }],
    })
    _write_intent_file(t212_one_shot, {
        "intent_id": "t212-one-shot-NVDA_US_EQ-1",
        "ticker": "NVDA_US_EQ",
        "side": "BUY",
        "qty": 2.5,
        "limit_price": 120.0,
        "applied_fill_qty": 0.5,
        "lifecycle": "accepted",
        "order_id": "777",
    })
    _write_intent_file(outbox, [{
        "symbol": "SOLUSDC",
        "side": "BUY",
        "qty": 3.0,
        "ref_price": 150.0,
        "place_kwargs": {"client_order_id": "CID-NESTED-1"},
    }], lines=True)

    result = build_active_intent_index(str(tmp_path))
    by_id = {row["intent_id"]: row for row in result["intents"]}
    assert by_id["krk-intent-1"]["requested_qty"] == 0.25
    assert by_id["krk-intent-1"]["executed_qty"] == 0.10
    assert by_id["t212-one-shot-NVDA_US_EQ-1"]["requested_price"] == 120.0
    assert by_id["t212-one-shot-NVDA_US_EQ-1"]["executed_qty"] == 0.5
    assert by_id["CID-NESTED-1"]["client_order_id"] == "CID-NESTED-1"
    assert by_id["CID-NESTED-1"]["requested_price"] == 150.0


def test_excludes_inactive_tickets_in_rtrade_pairs(tmp_path):
    rtrade = tmp_path / "cachedb/rtrade_pairs_state.json"
    _write_intent_file(rtrade, {
        "pairs": {
            "round-1": {
                "symbol": "ETHUSDC",
                "terminal": False,
                "tickets": [
                    {"order_id": "101", "side": "BUY", "price": 3000.0, "qty": 0.1, "active": False},
                    {"order_id": "102", "side": "SELL", "price": 3100.0, "qty": 0.1, "active": True},
                ],
            }
        }
    })
    result = build_active_intent_index(str(tmp_path))
    ids = {row["intent_id"] for row in result["intents"]}
    assert ids == {"102"}


def test_normalizes_px_sz_filled_sz_side_and_venue_aliases(tmp_path):
    hl_state = tmp_path / "hyperliquid/.state_HYPE.json"
    _write_intent_file(hl_state, {
        "venue": "HL",
        "orders": [{
            "id": 9001.0,
            "side": "OPEN LONG",
            "sz": 10.0,
            "filled_sz": 2.0,
            "px": 25.0,
        }],
    })
    result = build_active_intent_index(str(tmp_path))
    assert len(result["intents"]) == 1
    it = result["intents"][0]
    assert it["intent_id"] == "9001"
    assert it["venue"] == "hyperliquid"
    assert it["side"] == "BUY"
    assert it["requested_qty"] == 10.0
    assert it["executed_qty"] == 2.0
    assert it["requested_price"] == 25.0
    assert result["summary"]["pending_notional"]["total_buy"] == 200.0


def test_iter67_terminal_shadowing_refused_status_negative_exec_qty_and_zero_notional_summary(tmp_path):
    outbox = tmp_path / "cachedb/order_retry_queue.jsonl"
    _write_intent_file(
        outbox,
        [
            {
                "intent_id": "term-shadow-1",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 1.0,
                "requested_price": 60000.0,
                "submission_outcome": "accepted",
                "last_status": "canceled",
            },
            {
                "intent_id": "refused-1",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 1.0,
                "requested_price": 60000.0,
                "submission_outcome": "refused",
                "submit_status": "refused_before_submit",
            },
            {
                "intent_id": "active-partial-1",
                "symbol": "ETHUSDC",
                "side": "BUY",
                "qty": 2.0,
                "executed_qty": -1.0,
                "requested_price": 3000.0,
                "submission_outcome": "accepted",
                "last_status": "partial",
            },
            {
                "intent_id": "active-noprice-1",
                "symbol": "SOLUSDC",
                "side": "SELL",
                "qty": 5.0,
                "lifecycle": "pending",
            },
        ],
        lines=True,
    )
    result = build_active_intent_index(str(tmp_path))
    by_id = {row["intent_id"]: row for row in result["intents"]}
    assert set(by_id.keys()) == {"active-partial-1", "active-noprice-1"}
    assert by_id["active-partial-1"]["status"] == "partial"
    assert result["summary"]["pending_notional"]["by_symbol"]["ETHUSDC"]["buy"] == 6000.0
    formatted = format_summary(result)
    assert "SOLUSDC: $0.00" in formatted


if __name__ == "__main__":
    unittest.main(verbosity=2)
