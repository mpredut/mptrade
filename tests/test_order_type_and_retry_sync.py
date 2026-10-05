"""Tests for order type normalization, retry worker quantity synchronization, and zombie recovery."""
import math
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import order_retry as oq
import order_retry_worker as worker
from providers.kraken_provider import KrakenProvider
from providers.strategy_executor import OrderReconciliationCapabilities, OrderStatus
from binance_api import bapi_placeorder as po
from instrument import Instrument
from providers.market_api import MarketApi


class OrderTypeAndRetrySyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_queue_file = oq.QUEUE_FILE
        self.orig_lock_file = oq.LOCK_FILE
        oq.QUEUE_FILE = os.path.join(self.tmp, "q.jsonl")
        oq.LOCK_FILE = os.path.join(self.tmp, "q.lock")
        oq.RETRY_ENABLED = True
        oq.RETRY_INTERVAL_SEC = 300.0
        oq.RETRY_TTL_SEC = 86400.0
        oq.RETRY_MAX_ATTEMPTS = 0
        oq.RETRY_PRICE_TOL = 0.002

    def tearDown(self):
        oq.QUEUE_FILE = self.orig_queue_file
        oq.LOCK_FILE = self.orig_lock_file

    def test_kraken_order_type_normalization(self):
        """Verify KrakenProvider respects market=True, force=True, and default limit."""
        class FakeKrakenClient:
            def __init__(self):
                self.calls = []

            def add_order(self, pair, side, volume, price=None, ordertype="limit", validate=False, cl_ord_id=None):
                self.calls.append({
                    "pair": pair, "side": side, "volume": volume,
                    "price": price, "ordertype": ordertype, "validate": validate,
                    "cl_ord_id": cl_ord_id,
                })
                return {"txid": ["TX-123"]}

        client = FakeKrakenClient()
        provider = KrakenProvider(client=client)

        # 1. market=True without force=True
        provider.place_order("HYPEUSD", "buy", 100.0, 1.5, market=True)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["ordertype"], "market")
        self.assertIsNone(client.calls[0]["price"])

        # 2. force=True without market=True
        provider.place_order("HYPEUSD", "sell", 100.0, 1.5, force=True)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["ordertype"], "market")
        self.assertIsNone(client.calls[1]["price"])

        # 3. Neither (limit order)
        provider.place_order("HYPEUSD", "buy", 100.0, 1.5)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[2]["ordertype"], "limit")
        self.assertEqual(client.calls[2]["price"], 100.0)

    def test_binance_place_order_mechanics_market_flag(self):
        """Verify place_order_mechanics handles market=True equivalently to force=True."""
        with patch.object(po.api, "get_free_balance", return_value=100.0), \
             patch.object(po.api, "get_current_price", return_value=50.0), \
             patch("providers.binance_filters.BinanceOrderRules.from_symbol_info") as mock_from_info, \
             patch.object(po, "_submit_binance_order", return_value={"orderId": 999}) as mock_submit:

            mock_rules = MagicMock()
            mock_rules.normalize.return_value = ("1.0", None)
            mock_from_info.return_value = mock_rules

            # Test with market=True, force=False
            res = po.place_order_mechanics("BUY", "BTCUSDC", 50.0, 1.0, force=False, market=True)
            self.assertEqual(res, {"orderId": 999})
            mock_submit.assert_called_once()
            _, kwargs = mock_submit.call_args
            self.assertTrue(kwargs.get("market"))
            self.assertIsNone(kwargs.get("price"))

    def test_instrument_normalizes_market_and_force(self):
        """Verify Instrument.place sets both market and force consistently in provider_kwargs."""
        received_kwargs = {}
        from providers.base import MarketDataProvider
        class MockProvider(MarketDataProvider):
            name = "mock"
            def validate_symbol(self, s): pass
            def get_current_price(self, s): return 100.0
            def free_balance(self, a): return 100.0
            def free_balance_for(self, name, a): return 100.0
            def supports_symbol(self, s): return True
            def execution_enabled(self): return True
            def guards_internally(self): return False
            def get_orders(self, *args, **kwargs): return []
            def place_order(self, symbol, side, price, qty, **kwargs):
                received_kwargs.update(kwargs)
                return {"orderId": 1234}

        provider = MockProvider()
        api = MarketApi([provider])
        inst = Instrument("TESTUSDC", symbol="TESTUSDC", provider="mock", api=api)

        # Calling with market=True sets both market=True and force=True
        inst.place("BUY", 100.0, 1.0, market=True, bypass_profit_guard=True)
        self.assertTrue(received_kwargs.get("market"))
        self.assertTrue(received_kwargs.get("force"))

    def test_retry_worker_quantity_sync_on_late_scale(self):
        """Verify that when worker dispatches and late scaling reduces quantity, the queue is updated before submit."""
        oq.rewrite([])
        record_id = oq.enqueue(
            "BTCUSDC", "BUY", 4.0, {}, requested_price=100.0,
            failure_reason="profit_guard", now=1000.0)
        oq.mark_failure(
            record_id, "profit_guard", now=1000.0,
            submission_state="refused")

        class ScalingMkt:
            def __init__(self):
                self.calls = []

            def get_current_price(self, symbol, provider_name=None):
                return 100.0

            def order_by_client_id(self, symbol, client_order_id, provider_name=None):
                return None

            def reconciliation_capabilities(self, symbol, provider_name=None):
                return OrderReconciliationCapabilities(
                    True, True, True, True, not_found_reliable_for_seconds=3600.0)

            def place(self, symbol, side, price, qty, **kwargs):
                self.calls.append({"qty": qty, "kwargs": kwargs})
                # Simulate Instrument scaling late intelligence:
                # Update claim on disk with reduced quantity 1.0
                retry_claim = kwargs.get("_retry_claim")
                if retry_claim is not None:
                    oq.update_claimed_qty(retry_claim, 1.0)
                outcome = kwargs.get("_outcome_context")
                if outcome is not None:
                    outcome["accepted"] = True
                    outcome["submitted_qty"] = 1.0
                    outcome["submitted_price"] = price
                return {"orderId": 777}

        mkt = ScalingMkt()
        stats = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["succeeded"], 1)
        record = oq.get(record_id)
        self.assertEqual(record["lifecycle"], "accepted")
        self.assertEqual(record["order_id"], "777")
        self.assertEqual(record["qty"], 1.0)
        self.assertEqual(record["requested_qty_total"], 1.0)

        # Now simulate order expiration with partial fill 0.4: remainder must be 0.6 (1.0 - 0.4), NOT 3.6 (4.0 - 0.4)
        status = OrderStatus(
            status="canceled", filled_qty=0.4, cost=40.0, fee=0.04,
            venue_status="EXPIRED")
        claimed = oq.claim([record_id], now=2000.0)[0]
        transition = oq.advance_claimed_status(claimed, status, now=2000.0)
        self.assertEqual(transition.action, "retry_terminal")
        self.assertAlmostEqual(transition.remaining_qty, 0.6, places=6)

    def test_zombie_order_recovery_unknown_vs_dead_owner(self):
        """Verify that unknown owner state is quarantined fail-closed, while a real dead zombie is retried."""
        oq.rewrite([])
        claimed = oq.enqueue_claimed(
            "BTCUSDC", "BUY", 1.0, {}, requested_price=100.0,
            provider_name="Binance", now=1000.0, lease_sec=301.0)

        class ReconcilingMkt:
            def __init__(self):
                self.lookup_calls = []
                self.place_calls = []

            def get_current_price(self, symbol, provider_name=None):
                return 100.0

            def order_by_client_id(self, symbol, client_order_id, provider_name=None):
                self.lookup_calls.append(client_order_id)
                return None  # Confirmed absent from venue

            def reconciliation_capabilities(self, symbol, provider_name=None):
                return OrderReconciliationCapabilities(
                    True, True, True, True, not_found_reliable_for_seconds=3600.0)

            def place(self, symbol, side, price, qty, **kwargs):
                self.place_calls.append({"symbol": symbol, "side": side, "price": price, "qty": qty})
                return {"orderId": 888}

        mkt = ReconcilingMkt()
        # 1. Unknown owner state: identity unverifiable -> must quarantine fail-closed to avoid duplicate orders
        with patch.object(oq, "producer_claim_owner_state", return_value="unknown"):
            stats_unknown = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats_unknown["quarantined"], 1)
        self.assertEqual(stats_unknown["attempted"], 0)
        self.assertEqual(len(mkt.lookup_calls), 0)
        self.assertEqual(len(mkt.place_calls), 0)
        self.assertEqual(oq.get(claimed["id"])["submission_state"], "producer_claimed")

        # 2. Real dead zombie: owner confirmed dead -> safely reconciled and retried
        with patch.object(oq, "producer_claim_owner_state", return_value="dead"):
            stats_dead = worker.process_once(mkt, now=1400.0)

        self.assertEqual(stats_dead["quarantined"], 0)
        self.assertEqual(stats_dead["attempted"], 1)
        self.assertEqual(stats_dead["succeeded"], 1)
        self.assertEqual(len(mkt.lookup_calls), 1)
        self.assertEqual(len(mkt.place_calls), 1)
        durable = oq.get(claimed["id"])
        self.assertEqual(durable["lifecycle"], "accepted")
        self.assertEqual(durable["order_id"], "888")


if __name__ == "__main__":
    unittest.main(verbosity=2)
