"""Comprehensive tests for Pillar 2: Cross-Venue File-Based Order Ledger SSoT."""
from __future__ import annotations

import concurrent.futures
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import active_intents
from cacheaccount import CacheOrderFilledManager, CacheOrderOpenManager
from cachepaths import (
    CACHE_BINANCE_ORDER_FILLED_FILENAME,
    CACHE_BINANCE_ORDER_OPEN_FILENAME,
    CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
    CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
    CACHE_KRAKEN_ORDER_FILLED_FILENAME,
    CACHE_KRAKEN_ORDER_OPEN_FILENAME,
    ORDER_LEDGER_ACTIVE_FILENAME,
    order_ledger_active_path,
    order_ledger_closed_path,
    order_ledger_events_path,
)
from order_ledger_worker import (
    OrderLedgerReconciler,
    derive_intent_stale_sec,
    derive_reconcile_grace_sec,
)
from providers.execution_audit import ExecutionAudit
from providers.market_api import MarketApi
from providers.order_id_context import derive_venue_128bit_client_order_id
from providers.order_ledger import (
    OrderEventType,
    OrderLedgerReader,
    OrderLedgerState,
    OrderLedgerWriter,
    get_default_ledger_writer,
    utc_day_for_ts,
    utc_month_for_ts,
)
from providers.strategy_executor import OrderReconciliationCapabilities


def test_order_ledger_multi_producer_tail_overlay_and_zero_ms_visibility(tmp_path):
    cdb = str(tmp_path / "cachedb")
    os.makedirs(os.path.join(cdb, "orders"), exist_ok=True)
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    t0 = 1_770_000_100.0

    # Concurrent multi-producer appends across venues/bots must never corrupt JSONL
    def _emit(idx: int) -> bool:
        return writer.record_event(
            OrderEventType.INTENT,
            venue="binance" if idx % 2 == 0 else "kraken",
            symbol="BTCUSDC" if idx % 2 == 0 else "SPCXxUSD",
            ts=t0 + idx * 0.01,
            owner="tradeall" if idx % 2 == 0 else "ai_executor",
            side="BUY",
            order_type="LIMIT",
            qty=0.05,
            px=65000.0,
            iid=f"intent-{idx}",
            cid=f"TA_cid_{idx}" if idx % 2 == 0 else f"AI_cid_{idx}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_emit, range(12)))
    assert all(results)

    # Even BEFORE OrderLedgerReconciler has ever run, tail-overlay reader sees all 12 intents with 0ms lag
    active_all = reader.list_active_orders(apply_tail=True)
    assert len(active_all) == 12
    assert len(reader.list_active_orders(venue="binance")) == 6
    assert len(reader.list_active_orders(venue="kraken", symbol="SPCXxUSD")) == 6
    assert len(reader.list_active_orders(owner="ai_executor")) == 6
    # Passive reader must NOT create orders_active.json on disk
    assert not os.path.exists(order_ledger_active_path(cachedb_dir=cdb))


def test_reconciler_converges_open_partial_filled_canceled_failed_and_orphans_cross_venue(tmp_path):
    cdb = str(tmp_path / "cachedb")
    orders_dir = tmp_path / "cachedb" / "orders"
    orders_dir.mkdir(parents=True)
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(
        cachedb_dir=cdb,
        use_singletons=False,
        reconcile_grace_sec=30.0,
        intent_stale_sec=120.0,
    )
    t0 = 1_770_100_000.0

    # 1. Binance order: INTENT -> SUBMITTED -> matched in CacheOrderOpenManager -> PARTIAL -> FILLED
    writer.record_event(
        OrderEventType.INTENT,
        venue="Binance",  # mixed casing must normalize to lowercase "binance"
        symbol="BTCUSDC",
        ts=t0,
        owner="tradeall",
        side="BUY",
        order_type="LIMIT",
        qty=0.10,
        px=64000.0,
        iid="bn-iid-1",
        cid="TA_bn_1",
    )
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="BTCUSDC",
        ts=t0 + 0.5,
        iid="bn-iid-1",
        cid="TA_bn_1",
    )

    # 2. Hyperliquid order: SUBMITTED with cloid correlation (128-bit hex derived from cid)
    hl_cid = "AI_hl_order_1"
    hl_cloid = derive_venue_128bit_client_order_id("hyperliquid", hl_cid)
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0 + 1.0,
        owner="ai_executor",
        side="SELL",
        order_type="LIMIT",
        qty=5.0,
        px=62.0,
        iid="hl-iid-1",
        cid=hl_cid,
    )

    # 3. Kraken order: SUBMIT_UNKNOWN that never arrives on exchange -> closes as FAILED after verified grace
    writer.record_event(
        OrderEventType.SUBMIT_UNKNOWN,
        venue="kraken",
        symbol="HYPEUSD",
        ts=t0 + 1.0,
        owner="spot_dca",
        side="BUY",
        order_type="LIMIT",
        qty=2.0,
        px=60.0,
        iid="kr-iid-lost",
        cid="SD_kr_lost",
        reason="ReadTimeout",
    )

    # 4. Guard-blocked (non-deferred) decision -> immediately archived as REJECTED, never pollutes active orders
    writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="binance",
        symbol="ETHUSDC",
        ts=t0 + 2.0,
        owner="tradeall",
        side="BUY",
        qty=0.5,
        px=3100.0,
        iid="bn-iid-blocked",
        reason="profit_guard",
        deferred=False,
    )

    # Seed live venue open-order snapshots (Binance has TA_bn_1 + manual orphan; HL has hl_cloid; Kraken empty)
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME),
        {
            "BTCUSDC": [
                {
                    "orderId": "9001",
                    "clientOrderId": "TA_bn_1",
                    "side": "BUY",
                    "price": 64000.0,
                    "origQty": 0.10,
                    "executedQty": 0.04,
                    "status": "PARTIALLY_FILLED",
                    "timestamp": int((t0 + 2) * 1000),
                },
                {
                    "orderId": "9999",
                    "clientOrderId": "web_manual_1",
                    "side": "SELL",
                    "price": 70000.0,
                    "origQty": 0.02,
                    "executedQty": 0.0,
                    "status": "NEW",
                    "timestamp": int((t0 + 2) * 1000),
                },
            ]
        },
        now_ms=int((t0 + 5) * 1000),
        verified_ts=t0 + 5.0,
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {
            "HYPEUSDC": [
                {
                    "orderId": "88001",
                    "clientOrderId": hl_cloid,
                    "side": "SELL",
                    "price": 62.0,
                    "origQty": 5.0,
                    "executedQty": 0.0,
                    "status": "OPEN",
                    "timestamp": int((t0 + 3) * 1000),
                }
            ]
        },
        now_ms=int((t0 + 5) * 1000),
        verified_ts=t0 + 5.0,
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_KRAKEN_ORDER_OPEN_FILENAME),
        {"HYPEUSD": []},
        now_ms=int((t0 + 5) * 1000),
        verified_ts=t0 + 5.0,
    )

    # First pass at t0 + 6s (before 30s grace): bn-iid-1 -> PARTIAL (vid=9001), hl-iid-1 -> OPEN (vid=88001),
    # orphan 9999 adopted (owner=manual), kr-iid-lost still SUBMIT_UNKNOWN (within grace), blocked archived.
    st1 = reconciler.reconcile_once(now_ts=t0 + 6.0)
    assert st1["orphans_adopted"] == 1
    assert st1["reconciled_open"] == 2
    assert st1["reconciled_closed"] == 1  # bn-iid-blocked

    active1 = {r["key"]: r for r in reader.list_active_orders(apply_tail=False)}
    assert active1["binance:iid:bn-iid-1"]["state"] == OrderLedgerState.PARTIAL.value
    assert active1["binance:iid:bn-iid-1"]["vid"] == "9001"
    assert active1["binance:iid:bn-iid-1"]["fq"] == pytest.approx(0.04)
    assert active1["hyperliquid:iid:hl-iid-1"]["state"] == OrderLedgerState.OPEN.value
    assert active1["hyperliquid:iid:hl-iid-1"]["vid"] == "88001"
    assert active1["binance:cid:web_manual_1"]["owner"] == "manual"
    assert active1["binance:cid:web_manual_1"]["vid"] == "9999"
    assert active1["kraken:iid:kr-iid-lost"]["state"] == OrderLedgerState.SUBMIT_UNKNOWN.value

    # Second pass at t0 + 45s (past 30s grace):
    # - Binance TA_bn_1 fills completely (removed from open, appears in filled cache)
    # - Hyperliquid 88001 disappears from open book with no fill -> CANCELED
    # - Kraken kr-iid-lost still absent from freshly verified open+filled -> FAILED
    # - Orphan 9999 remains OPEN
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME),
        {
            "BTCUSDC": [
                {
                    "orderId": "9999",
                    "clientOrderId": "web_manual_1",
                    "side": "SELL",
                    "price": 70000.0,
                    "origQty": 0.02,
                    "executedQty": 0.0,
                    "status": "NEW",
                    "timestamp": int((t0 + 2) * 1000),
                }
            ]
        },
        now_ms=int((t0 + 45) * 1000),
        verified_ts=t0 + 45.0,
    )
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(orders_dir / CACHE_BINANCE_ORDER_FILLED_FILENAME),
        {
            "BTCUSDC": [
                {
                    "orderId": 9001,
                    "id": "fill-9001",
                    "price": 64010.0,
                    "quantity": 0.10,
                    "timestamp": int((t0 + 20) * 1000),
                    "side": "BUY",
                    "clientOrderId": "TA_bn_1",
                    "status": "FILLED",
                    "fee": 0.0005,
                    "feeAsset": "BNB",
                }
            ]
        },
        venue="binance",
        now_ms=int((t0 + 45) * 1000),
        is_snapshot=True,
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {"HYPEUSDC": []},
        now_ms=int((t0 + 45) * 1000),
        verified_ts=t0 + 45.0,
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_KRAKEN_ORDER_OPEN_FILENAME),
        {"HYPEUSD": []},
        now_ms=int((t0 + 45) * 1000),
        verified_ts=t0 + 45.0,
    )

    st2 = reconciler.reconcile_once(now_ts=t0 + 45.0)
    assert st2["reconciled_filled"] == 1
    assert st2["reconciled_closed"] == 3  # bn filled + hl canceled + kr failed
    assert st2["active_count"] == 1  # only manual orphan 9999 remains open!

    # Active ledger is now 1:1 with venue open book
    active2 = reader.list_active_orders(apply_tail=True)
    assert [r["vid"] for r in active2] == ["9999"]

    # Trace by venue orderId, cid, and intent_id all correlate seamlessly
    bn_trace = reader.trace_order("9001")
    assert bn_trace["found"] is True
    assert bn_trace["active_order"] is None
    assert len(bn_trace["closed_orders"]) == 1
    assert bn_trace["closed_orders"][0]["state"] == OrderLedgerState.FILLED.value
    assert bn_trace["closed_orders"][0]["avg_px"] == pytest.approx(64010.0)
    assert bn_trace["closed_orders"][0]["fee"] == pytest.approx(0.0005)
    assert bn_trace["closed_orders"][0]["fee_asset"] == "BNB"
    assert "ta_bn_1" in bn_trace["correlated_ids"]

    hl_trace = reader.trace_order(hl_cid)
    assert hl_trace["closed_orders"][0]["state"] == OrderLedgerState.CANCELED.value

    kr_trace = reader.trace_order("kr-iid-lost")
    assert kr_trace["closed_orders"][0]["state"] == OrderLedgerState.FAILED.value

    # Closed monthly archive file exists in cachedb/orders/
    closed_file = order_ledger_closed_path(utc_month_for_ts(t0), cachedb_dir=cdb)
    assert os.path.isfile(closed_file)
    events_file = order_ledger_events_path(utc_day_for_ts(t0), cachedb_dir=cdb)
    assert os.path.isfile(events_file)


def test_execution_audit_market_api_cancel_and_active_intents_trace_integration(tmp_path, monkeypatch):
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(cdb))

    # 1. ExecutionAudit.record mirrors automatically into OrderLedgerWriter
    audit = ExecutionAudit(directory=str(tmp_path / "audit"))
    assert audit.record(
        "submit_requested",
        intent_id="aud-iid-42",
        venue="Kraken",
        symbol="SPCXxUSD",
        side="buy",
        qty=2.0,
        price=140.0,
        client_order_id="AI_spcx_42",
    )
    assert audit.record(
        "submit_accepted",
        intent_id="aud-iid-42",
        venue="Kraken",
        symbol="SPCXxUSD",
        order_id="OKRK-42",
        client_order_id="AI_spcx_42",
    )

    # 2. MarketApi.cancel_order records CANCEL_REQ + CANCEL_ACK (covering monitororder & technical_repair)
    fake_prov = MagicMock()
    fake_prov.name = "kraken"
    fake_prov.supports_symbol.return_value = True
    fake_prov.reconciliation_capabilities.return_value = OrderReconciliationCapabilities(
        lookup_by_client_order_id=True,
        status_by_order_id=True,
        cancel_by_order_id=True,
        list_open_orders=True,
    )
    fake_prov.cancel_order.return_value = None
    mkt = MarketApi([fake_prov])
    mkt.cancel_order("SPCXxUSD", "OKRK-42", provider_name="kraken")

    # 3. Reconcile & trace via active_intents.trace_order_or_intent
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdb), use_singletons=False)
    reconciler.reconcile_once()

    trace = active_intents.trace_order_or_intent("OKRK-42", root=str(tmp_path), audit_dir=str(tmp_path / "audit"))
    assert trace["found"] is True
    assert trace["ledger_order"] is not None
    assert trace["ledger_order"]["state"] == OrderLedgerState.CANCELED.value
    assert trace["ledger_order"]["v"] == "kraken"
    assert trace["ledger_order"]["sym"] == "SPCXxUSD"
    ev_types = [e["ev"] for e in trace["ledger_events"]]
    assert OrderEventType.SUBMITTED.value in ev_types
    assert OrderEventType.ACKED.value in ev_types
    assert OrderEventType.CANCEL_REQ.value in ev_types
    assert OrderEventType.CANCEL_ACK.value in ev_types


def test_formulaic_reconcile_grace_and_day_rollover_cursor(tmp_path):
    assert derive_reconcile_grace_sec(15.0) == pytest.approx(45.0)
    assert derive_intent_stale_sec(15.0) == pytest.approx(150.0)

    cdb = str(tmp_path / "cachedb")
    os.makedirs(os.path.join(cdb, "orders"), exist_ok=True)
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False)

    day1_ts = 1_770_076_790.0  # 23:59:50 UTC
    day2_ts = day1_ts + 30.0   # 00:00:20 UTC next day
    assert utc_day_for_ts(day1_ts) != utc_day_for_ts(day2_ts)

    writer.record_event(
        OrderEventType.INTENT,
        venue="binance",
        symbol="BTCUSDC",
        ts=day1_ts,
        iid="rollover-1",
        cid="TA_roll_1",
        qty=0.01,
        px=65000.0,
    )
    reconciler.reconcile_once(now_ts=day1_ts + 5.0)

    # Event on next UTC day links vid to the prior-day intent across file boundary
    writer.record_event(
        OrderEventType.ACKED,
        venue="binance",
        symbol="BTCUSDC",
        ts=day2_ts,
        iid="rollover-1",
        cid="TA_roll_1",
        vid="555001",
    )
    stats = reconciler.reconcile_once(now_ts=day2_ts + 2.0)
    assert stats["events_applied"] == 1
    assert stats["active_count"] == 1

    active = OrderLedgerReader(cachedb_dir=cdb).list_active_orders(apply_tail=False)
    assert len(active) == 1
    assert active[0]["vid"] == "555001"
    assert active[0]["state"] == OrderLedgerState.OPEN.value
    # Confirm orders_active.json lives under cachedb/orders/
    assert os.path.basename(order_ledger_active_path(cachedb_dir=cdb)) == ORDER_LEDGER_ACTIVE_FILENAME
    assert get_default_ledger_writer() is not None


def test_order_ledger_worker_once_cli(tmp_path, monkeypatch):
    import sys
    import order_ledger_worker

    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(cdb))
    get_default_ledger_writer().record_event(
        OrderEventType.INTENT,
        venue="binance",
        symbol="BTCUSDC",
        owner="tradeall",
        side="BUY",
        qty=0.02,
        px=64500.0,
        iid="worker-cli-1",
        cid="TA_worker_1",
    )
    monkeypatch.setattr(sys, "argv", ["order_ledger_worker.py", "--once"])
    assert order_ledger_worker.main() == 0
    active_p = order_ledger_active_path(cachedb_dir=str(cdb))
    assert os.path.isfile(active_p)
    doc = json.loads(open(active_p, encoding="utf-8").read())
    assert "binance:iid:worker-cli-1" in doc["orders"]
