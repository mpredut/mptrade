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
from order_ledger_reconciler_worker import (
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
    # orphan 9999 adopted (owner=manual), kr-iid-lost still SUBMIT_UNKNOWN (within grace);
    # pre-submit GUARD_BLOCKED (never submitted) stays traceable in events only, NOT in orders_closed.
    st1 = reconciler.reconcile_once(now_ts=t0 + 6.0)
    assert st1["orphans_adopted"] == 1
    assert st1["reconciled_open"] == 2
    assert st1["reconciled_closed"] == 0  # pre-submit GUARD_BLOCKED is never archived as a closed order
    blocked_trace = reader.trace_order("bn-iid-blocked")
    assert blocked_trace["found"] is True
    assert blocked_trace["closed_orders"] == []
    assert any(e["ev"] == OrderEventType.GUARD_BLOCKED.value for e in blocked_trace["events"])

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


def test_order_ledger_reconciler_worker_once_cli(tmp_path, monkeypatch):
    import sys
    import order_ledger_reconciler_worker

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
    monkeypatch.setattr(sys, "argv", ["order_ledger_reconciler_worker.py", "--once"])
    assert order_ledger_reconciler_worker.main() == 0
    active_p = order_ledger_active_path(cachedb_dir=str(cdb))
    assert os.path.isfile(active_p)
    doc = json.loads(open(active_p, encoding="utf-8").read())
    assert "binance:iid:worker-cli-1" in doc["orders"]


def test_prod_parity_no_stale_open_readoption_clean_owner_and_order_terminal(tmp_path, monkeypatch):
    """Verify PROD-discovered invariants: no stale-open re-adoption loop, clean owner, order_terminal bridge, no anon guard spam."""
    cdb = str(tmp_path / "cachedb")
    orders_dir = tmp_path / "cachedb" / "orders"
    orders_dir.mkdir(parents=True)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", cdb)
    t0 = 1_795_448_920.0

    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)
    hl_cid = "AI_intent_1795448914_buy_hypeusdc"
    hl_cloid = derive_venue_128bit_client_order_id("hyperliquid", hl_cid)

    # 1. Wrapper owner='market_api.py' with AI_ cid must resolve to 'ai_executor', never 'market_api_py'.
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0,
        owner="market_api.py",
        side="BUY",
        qty=8.31,
        px=66.15,
        iid="intent_hl_1",
        cid=hl_cid,
    )
    writer.record_event(
        OrderEventType.ACKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0 + 4.0,
        iid="intent_hl_1",
        cid=hl_cid,
        vid="614160779028",
    )
    # Anonymous GUARD_BLOCKED (no iid/cid/vid) must NOT pollute orders_closed archive
    writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="binance",
        symbol="TAOUSDC",
        ts=t0 + 5.0,
        owner="rtrade",
        side="BUY",
        qty=2.5,
        px=195.0,
        reason="profit_guard",
        deferred=False,
    )

    # Fill arrives in CacheOrderFilledManager BEFORE CacheOrderOpenManager drops the order (10s poll lag!)
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {
            "HYPEUSDC": [{
                "orderId": "614160779028",
                "clientOrderId": hl_cloid,
                "side": "BUY",
                "price": 66.15,
                "origQty": 8.31,
                "executedQty": 0.0,
                "status": "OPEN",
                "timestamp": int((t0 + 4) * 1000),
            }]
        },
        now_ms=int((t0 + 10) * 1000),
        verified_ts=t0 + 10.0,
    )
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(orders_dir / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME),
        {
            "HYPEUSDC": [{
                "orderId": "614160779028",
                "id": "fill-hl-1",
                "clientOrderId": hl_cloid,
                "side": "BUY",
                "price": 66.15,
                "quantity": 8.31,
                "status": "FILLED",
                "timestamp": int((t0 + 12) * 1000),
            }]
        },
        venue="hyperliquid",
        now_ms=int((t0 + 12) * 1000),
        is_snapshot=True,
    )

    st1 = reconciler.reconcile_once(now_ts=t0 + 13.0)
    assert st1["reconciled_filled"] == 1
    assert st1["reconciled_closed"] == 1  # ONLY the real HL order; anonymous GUARD_BLOCKED skipped!
    assert st1["active_count"] == 0

    # Subsequent passes while open-order cache is STILL stale MUST NOT re-adopt or re-close!
    st2 = reconciler.reconcile_once(now_ts=t0 + 15.0)
    st3 = reconciler.reconcile_once(now_ts=t0 + 17.0)
    assert st2["orphans_adopted"] == 0 and st2["reconciled_filled"] == 0 and st2["reconciled_closed"] == 0
    assert st3["orphans_adopted"] == 0 and st3["reconciled_filled"] == 0 and st3["reconciled_closed"] == 0

    # Even a fresh reconciler instance (empty RAM memory) seeing stale open + complete fill MUST skip adoption!
    fresh_rec = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)
    st4 = fresh_rec.reconcile_once(now_ts=t0 + 19.0)
    assert st4["orphans_adopted"] == 0 and st4["reconciled_filled"] == 0

    trace = OrderLedgerReader(cachedb_dir=cdb).trace_order("614160779028")
    assert len(trace["closed_orders"]) == 1
    assert trace["closed_orders"][0]["owner"] == "ai_executor"

    # 2. Late ExecutionAudit 'order_terminal' arriving AFTER reconciler already closed the order
    #    must NOT resurrect the order or append a duplicate row into orders_closed_YYYY-MM.jsonl!
    audit = ExecutionAudit(directory=str(tmp_path / "audit"))
    assert audit.record(
        "order_terminal",
        intent_id="intent_hl_1",
        venue="hyperliquid",
        symbol="HYPEUSDC",
        order_id="614160779028",
        client_order_id=hl_cid,
        side="buy",
        filled_qty=8.31,
        cost=549.7065,
        status="closed",
        venue_status="FILLED",
    )
    st_late = fresh_rec.reconcile_once(now_ts=t0 + 25.0)
    assert st_late["reconciled_closed"] == 0 and st_late["active_count"] == 0
    assert len(OrderLedgerReader(cachedb_dir=cdb).trace_order("614160779028")["closed_orders"]) == 1

    # 3. Unsubmitted tracked GUARD_BLOCKED (even with iid + cid) & unsubmitted stale INTENT never enter orders_closed
    writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="binance",
        symbol="TAOUSDC",
        ts=t0 + 26.0,
        owner="ai_executor",
        side="SELL",
        qty=4.18,
        px=195.2,
        iid="intent_unsubmitted_guard",
        cid="AI_intent_unsubmitted_guard",
        reason="insufficient_funds",
        deferred=False,
    )
    writer.record_event(
        OrderEventType.INTENT,
        venue="binance",
        symbol="TAOUSDC",
        ts=t0 + 26.0,
        owner="tradeall",
        side="BUY",
        qty=1.0,
        px=190.0,
        iid="intent_unsubmitted_stale",
    )
    st_unsub = reconciler.reconcile_once(now_ts=t0 + 300.0)
    assert st_unsub["reconciled_closed"] == 0 and st_unsub["active_count"] == 0
    assert OrderLedgerReader(cachedb_dir=cdb).trace_order("intent_unsubmitted_guard")["closed_orders"] == []
    assert OrderLedgerReader(cachedb_dir=cdb).trace_order("intent_unsubmitted_stale")["closed_orders"] == []

    # 4. CANCEL_FAILED reverts CANCELING back to OPEN so a transient network error never strands CANCELING
    writer.record_event(OrderEventType.ACKED, venue="binance", symbol="BTCUSDC", ts=t0 + 301.0, iid="cf-1", vid="55501", side="BUY", qty=0.01, px=62000.0)
    writer.record_event(OrderEventType.CANCEL_REQ, venue="binance", symbol="BTCUSDC", ts=t0 + 302.0, vid="55501")
    writer.record_event(OrderEventType.CANCEL_FAILED, venue="binance", symbol="BTCUSDC", ts=t0 + 303.0, vid="55501", reason="ReadTimeout")
    act_cf = OrderLedgerReader(cachedb_dir=cdb).list_active_orders(venue="binance", apply_tail=True)
    assert len(act_cf) == 1 and act_cf[0]["state"] == OrderLedgerState.OPEN.value

    # 5. ExecutionAudit 'order_terminal' for an unclosed order mirrors cleanly into OrderLedger as STATUS -> FILLED
    assert audit.record(
        "order_terminal",
        intent_id="aud-term-1",
        venue="binance",
        symbol="BTCUSDC",
        order_id="10944715108",
        client_order_id="AI_aud_term_1",
        side="buy",
        filled_qty=0.01283,
        cost=799.95,
        status="closed",
        venue_status="FILLED",
    )
    tr_term = OrderLedgerReader(cachedb_dir=cdb).trace_order("10944715108")
    assert tr_term["found"] is True
    assert any(e["ev"] == OrderEventType.STATUS.value and e.get("vst") == "FILLED" for e in tr_term["events"])

    # 6. Orphan Hyperliquid order with hex cloid (no prior fill) gets owner='external', NEVER 'alt_0x...'
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {
            "HYPEUSDC": [{
                "orderId": "999888777",
                "clientOrderId": "0x99998888777766665555444433332222",
                "side": "SELL",
                "price": 75.0,
                "origQty": 2.0,
                "executedQty": 0.0,
                "status": "OPEN",
                "timestamp": int((t0 + 305) * 1000),
            }]
        },
        now_ms=int((t0 + 305) * 1000),
        verified_ts=t0 + 305.0,
    )
    st5 = reconciler.reconcile_once(now_ts=t0 + 306.0)
    assert st5["orphans_adopted"] == 1
    act = OrderLedgerReader(cachedb_dir=cdb).list_active_orders(venue="hyperliquid", apply_tail=False)
    assert len(act) == 1 and act[0]["owner"] == "external"

    # 7. KrakenProvider.open_orders_by_symbol fetches ALL pairs in ONE single private API call
    from providers.kraken_provider import KrakenProvider

    kp = KrakenProvider()
    fake_cli = MagicMock()
    fake_cli.open_orders.return_value = {
        "OKRK-1": {"descr": {"pair": "HYPEUSD", "type": "buy", "price": "60.0"}, "vol": "2.0", "vol_exec": "0.0"},
        "OKRK-2": {"descr": {"pair": "TAOUSD", "type": "sell", "price": "210.0"}, "vol": "1.0", "vol_exec": "0.0"},
    }
    kp._query_with_bot_fallback = lambda fn, _lbl: fn(fake_cli)
    by_sym = kp.open_orders_by_symbol(["HYPEUSD", "TAOUSD", "ADAUSD", "SPCXxUSD"])
    assert fake_cli.open_orders.call_count == 1
    assert len(by_sym["HYPEUSD"]) == 1 and len(by_sym["TAOUSD"]) == 1 and by_sym["ADAUSD"] == [] and by_sym["SPCXxUSD"] == []

    # 8. Outbox local pre-submit guard refusal (trend_deferred / profit_guard) stays in events ONLY, never orders_closed;
    #    real venue rejection (HTTP 400 / exchange error) DOES enter orders_closed.
    writer.record_event(OrderEventType.SUBMITTED, venue="binance", symbol="BTCUSDC", ts=t0 + 310.0, iid="outbox-local-1", cid="TA_OR_local_1", side="BUY", qty=0.1, px=61600.0)
    writer.record_event(OrderEventType.SUBMIT_REFUSED, venue="binance", symbol="BTCUSDC", ts=t0 + 310.1, iid="outbox-local-1", cid="TA_OR_local_1", reason="trend_deferred")
    writer.record_event(OrderEventType.SUBMITTED, venue="t212", symbol="NVDA_US_EQ", ts=t0 + 311.0, iid="t212-rej-1", side="BUY", qty=1.0, px=165.0)
    writer.record_event(OrderEventType.SUBMIT_REFUSED, venue="t212", symbol="NVDA_US_EQ", ts=t0 + 311.1, iid="t212-rej-1", reason="T212 HTTP 400: insufficient buying power")
    st_ref = reconciler.reconcile_once(now_ts=t0 + 312.0)
    assert st_ref["reconciled_closed"] == 1  # ONLY the genuine T212 HTTP 400 rejection!
    assert OrderLedgerReader(cachedb_dir=cdb).trace_order("outbox-local-1")["closed_orders"] == []
    assert len(OrderLedgerReader(cachedb_dir=cdb).trace_order("t212-rej-1")["closed_orders"]) == 1

    # 9. KrakenClient._private must NEVER tight-retry 'EAPI:Rate limit exceeded' (single attempt, immediate raise)
    from kraken import kraken_client

    kcli = kraken_client.KrakenClient("k", "c2VjcmV0")
    calls = []
    with patch.object(kraken_client, "http_post_form", side_effect=lambda *_a, **_kw: calls.append(1) or (200, '{"error":["EAPI:Rate limit exceeded"]}')):
        with pytest.raises(kraken_client.KrakenError, match="Rate limit"):
            kcli._private("Ledgers", {"start": 0}, fresh=True)
    assert len(calls) == 1


def test_migrated_audit_readers_use_order_ledger_first_with_legacy_fallback(tmp_path):
    """Verify all 4 migrated ExecutionAudit readers operate purely from OrderLedger and stitch pre-ledger history cleanly."""
    from intelligence.sentiment.collectors.portfolio_collector import (
        _load_audit_terminal_orders,
        _load_recent_audit_origins,
    )
    from providers.ai_attribution import load_and_evaluate_ai_attribution
    from providers.execution_audit import find_order_audit_record
    from providers.intent_reconciliation import reconcile_intents_from_execution_audit

    ws_dir = tmp_path / "ws"
    cdb_dir = ws_dir / "cachedb"
    orders_dir = cdb_dir / "orders"
    state_dir = cdb_dir / "state"
    audit_dir = ws_dir / "logger" / "execution_audit"
    orders_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    # Day 1 = historical pre-ledger ExecutionAudit only; Day 2 = boundary day (pre-ledger morning + OrderLedger afternoon)!
    day1_ts = 1_795_200_000.0  # 2026-11-20 UTC
    day2_ts = day1_ts + 2 * 86400.0 + 36000.0  # 2026-11-22 afternoon UTC
    day1_str = utc_day_for_ts(day1_ts)
    day2_str = utc_day_for_ts(day2_ts)

    # Seed Day 1 legacy execution_audit (1 historical BOT BUY on ETHUSDC)
    (audit_dir / f"execution_audit_{day1_str}.jsonl").write_text(
        json.dumps({
            "ts": day1_ts,
            "event": "order_terminal",
            "intent_id": "legacy-bot-buy-1",
            "venue": "binance",
            "symbol": "ETHUSDC",
            "order_id": "70001",
            "client_order_id": "TA_legacy_1",
            "side": "buy",
            "filled_qty": 2.0,
            "price": 3000.0,
            "cost": 6000.0,
            "status": "closed",
            "venue_status": "FILLED",
        }) + "\n",
        encoding="utf-8",
    )
    # Seed Day 2 boundary execution_audit: ONE pre-ledger morning AI BUY (MUST be kept!) + ONE post-ledger duplicate (MUST NOT double-count!)
    (audit_dir / f"execution_audit_{day2_str}.jsonl").write_text(
        json.dumps({
            "ts": day2_ts - 7200.0,
            "event": "order_terminal",
            "intent_id": "ai-morning-pre-ledger",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_id": "70000",
            "client_order_id": "AI_morning_pre_ledger",
            "side": "buy",
            "filled_qty": 0.01,
            "price": 61000.0,
            "cost": 610.0,
            "status": "closed",
            "venue_status": "FILLED",
        }) + "\n"
        + json.dumps({
            "ts": day2_ts + 25.0,
            "event": "order_terminal",
            "intent_id": "ai-hl-1",
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "order_id": "81001",
            "client_order_id": "AI_hl_migrated_1",
            "side": "buy",
            "filled_qty": 5.0,
            "price": 60.0,
            "cost": 300.0,
            "status": "closed",
            "venue_status": "FILLED",
        }) + "\n",
        encoding="utf-8",
    )

    writer = OrderLedgerWriter(cachedb_dir=str(cdb_dir))
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdb_dir), use_singletons=False, reconcile_grace_sec=30.0)

    # Day 2 OrderLedger records (ZERO Day 2 execution_audit files exist!):
    # A. BOT SELL closing Day 1 ETHUSDC BUY with native BNB fee (must NOT be misread as USD!)
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="ETHUSDC",
        ts=day2_ts,
        owner="tradeall",
        side="SELL",
        order_type="LIMIT",
        qty=2.0,
        px=3100.0,
        iid="led-bot-sell-1",
        cid="TA_led_sell_1",
        kind="trend_take_profit",
    )
    writer.record_event(
        OrderEventType.RECONCILED_FILLED,
        venue="binance",
        symbol="ETHUSDC",
        ts=day2_ts + 5.0,
        owner="tradeall",
        side="SELL",
        order_type="LIMIT",
        qty=2.0,
        px=3100.0,
        fq=2.0,
        avg_px=3100.0,
        fee=0.005,
        fee_asset="BNB",
        iid="led-bot-sell-1",
        cid="TA_led_sell_1",
        vid="70002",
        vst="FILLED",
        src="manual_test",
    )

    # B. Hyperliquid AI BUY filled (with 128-bit derived cloid) + pending intent with submit_requested only
    hl_cid = "AI_hl_migrated_1"
    hl_cloid = derive_venue_128bit_client_order_id("hyperliquid", hl_cid)
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=day2_ts + 10.0,
        owner="ai_executor",
        side="BUY",
        order_type="LIMIT",
        qty=5.0,
        px=60.0,
        iid="ai-hl-1",
        cid=hl_cid,
        kind="ai_entry",
    )
    writer.record_event(
        OrderEventType.STATUS,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=day2_ts + 20.0,
        owner="ai_executor",
        side="BUY",
        order_type="LIMIT",
        qty=5.0,
        px=60.0,
        fq=5.0,
        avg_px=60.0,
        fee=0.12,
        fee_asset="USDC",
        iid="ai-hl-1",
        cid=hl_cid,
        vid="81001",
        vst="FILLED",
    )
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="BTCUSDC",
        ts=day2_ts + 30.0,
        owner="ai_executor",
        side="BUY",
        order_type="LIMIT",
        qty=0.01,
        px=62000.0,
        iid="ai-pend-led",
        cid="AI_pend_led",
    )
    reconciler.reconcile_once(now_ts=day2_ts + 40.0)

    # Backfill Day 1 + Day 2 morning legacy execution_audit into OrderLedger, verify idempotency, then delete legacy audit_dir!
    import shutil
    from verify_tools.backfill_order_ledger_from_audit import backfill_order_ledger_from_audit

    dry_stats = backfill_order_ledger_from_audit(str(audit_dir), str(cdb_dir), dry_run=True)
    assert dry_stats["events_converted"] == 2  # Day 1 BUY + Day 2 morning AI BUY (post-live duplicate excluded)
    assert dry_stats["closed_rows_planned"] == 2

    bf_stats = backfill_order_ledger_from_audit(str(audit_dir), str(cdb_dir))
    assert bf_stats["event_days_written"] == 2
    assert bf_stats["boundary_day_prepended"] == 1
    assert bf_stats["closed_rows_added"] == 2

    # Idempotency: second run writes zero files and adds zero closed rows.
    bf_again = backfill_order_ledger_from_audit(str(audit_dir), str(cdb_dir))
    assert bf_again["event_days_written"] == 0
    assert bf_again["closed_rows_added"] == 0

    # Cursor offset on boundary day was shifted cleanly so reconciler does NOT replay backfilled events.
    rec_after = reconciler.reconcile_once(now_ts=day2_ts + 50.0)
    assert rec_after["events_applied"] == 0

    # Remove legacy execution_audit directory completely: ALL migrated readers must work 100% from OrderLedger!
    shutil.rmtree(audit_dir)

    # 1. find_order_audit_record resolves both live and backfilled orders purely from OrderLedger
    found = find_order_audit_record("70002", symbol="ETHUSDC", cachedb_dir=str(cdb_dir))
    assert found is not None
    assert found["client_order_id"] == "TA_led_sell_1"
    assert found["kind"] == "trend_take_profit"
    assert "fee" not in found and found.get("fee_native") == pytest.approx(0.005)  # BNB fee not misread as USD!
    found_bf = find_order_audit_record("70001", symbol="ETHUSDC", cachedb_dir=str(cdb_dir))
    assert found_bf is not None and found_bf["client_order_id"] == "TA_legacy_1"

    # 2. portfolio_collector helpers resolve origins (including HL cloid) & filled terminal orders purely from OrderLedger
    origins = _load_recent_audit_origins(str(cdb_dir))
    assert origins.get("70002") == ("tradeall", "trend_take_profit")
    assert origins.get("TA_led_sell_1") == ("tradeall", "trend_take_profit")
    assert hl_cloid in origins and origins[hl_cloid][1] == "ai_entry"

    hl_terms = _load_audit_terminal_orders(str(cdb_dir), "hyperliquid")
    assert "HYPEUSDC" in hl_terms and len(hl_terms["HYPEUSDC"]) == 1
    assert hl_terms["HYPEUSDC"][0]["orderId"] == "81001"
    assert hl_terms["HYPEUSDC"][0]["executedQty"] == pytest.approx(5.0)

    # 3. load_and_evaluate_ai_attribution reads backfilled Day 1 + Day 2 morning + live Day 2 purely from OrderLedger
    (state_dir / "autonomous_trade_intents_state.json").write_text("[]", encoding="utf-8")
    attr_rep = load_and_evaluate_ai_attribution(str(ws_dir), lookback_days=7.0, now_ts=day2_ts + 60.0)
    bot_cmp = attr_rep["execution_audit_ai_vs_bot"]["by_actor"]["BOT"]
    assert bot_cmp["filled_orders"] == 2  # Backfilled Day 1 BUY + live Day 2 SELL
    assert bot_cmp["matched_round_trips"] == 1
    assert bot_cmp["realized_net_usd"] > 180.0  # +$200 gross minus realistic ~20bps RT fee (NOT broken BNB fee)
    assert attr_rep["execution_audit_ai_vs_bot"]["by_actor"]["AI"]["filled_orders"] == 2  # Backfilled Day 2 morning + live Day 2 HL

    # 4. reconcile_intents_from_execution_audit reconciles both FILLED and PENDING_EXECUTION purely from OrderLedger
    intents_to_rec = [
        {
            "intent_id": "ai-hl-1",
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "decision": "BUY",
            "status": "FAILED",
            "timestamp": day2_ts + 5.0,
        },
        {
            "intent_id": "ai-pend-led",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "decision": "BUY",
            "status": "PENDING_EXECUTION",
            "timestamp": day2_ts - 1000.0,
        },
    ]
    changed = reconcile_intents_from_execution_audit(
        intents_to_rec, str(tmp_path / "nonexistent_audit_dir"), cachedb_dir=str(cdb_dir)
    )
    assert changed is True
    assert intents_to_rec[0]["status"] == "EXECUTED"
    assert intents_to_rec[0]["execution_payload"]["orderId"] == "81001"
    assert intents_to_rec[1]["status"] == "SUBMIT_UNKNOWN"
    assert intents_to_rec[1]["execution_payload"]["clientOrderId"] == "AI_pend_led"


def test_execution_audit_prod_mode_writes_only_order_ledger_and_6h_recently_closed_ttl(tmp_path, monkeypatch):
    """Production ExecutionAudit() writes exclusively to OrderLedger (no dual JSONL), and recently_closed prunes at 6h."""
    from providers import execution_audit as ea_mod
    from providers.order_ledger import update_recently_closed

    monkeypatch.delenv("EXECUTION_AUDIT_DIR", raising=False)
    monkeypatch.setenv("CACHEDB_DIR", str(tmp_path / "cachedb"))
    fake_default_audit = tmp_path / "default_legacy_audit"
    monkeypatch.setattr(ea_mod, "_DEFAULT_AUDIT_DIR", str(fake_default_audit))

    prod_audit = ea_mod.ExecutionAudit()
    prod_audit.record(
        "submit_requested",
        intent_id="prod-only-1",
        venue="binance",
        symbol="ETHUSDC",
        side="BUY",
        qty=0.5,
        price=3000.0,
    )
    assert not fake_default_audit.exists()
    reader = OrderLedgerReader(cachedb_dir=str(tmp_path / "cachedb"))
    assert len(reader.list_event_files()) == 1

    # Explicit directory still writes legacy JSONL for isolated test suites.
    explicit_dir = tmp_path / "explicit_audit"
    test_audit = ea_mod.ExecutionAudit(directory=str(explicit_dir))
    test_audit.record("submit_requested", intent_id="test-only-1", venue="binance", symbol="ETHUSDC", side="BUY", qty=0.2)
    assert any(p.name.startswith("execution_audit_") for p in explicit_dir.iterdir())

    # Default recently_closed TTL is 6 hours (21600s).
    now_t = 1_795_500_000.0
    pruned = update_recently_closed(
        {"old": now_t - 22_000.0, "fresh": now_t - 3_600.0},
        (),
        eff_now=now_t,
    )
    assert "old" not in pruned and "fresh" in pruned


def test_late_fill_supersedes_premature_canceled_and_closed_archives_survive_event_pruning(tmp_path):
    """Late authoritative FILLED updates are never dropped after premature CANCELED, and >60d attribution survives event pruning."""
    from providers.order_ledger import apply_order_event, update_recently_closed

    cdb_dir = tmp_path / "cachedb"
    writer = OrderLedgerWriter(cachedb_dir=str(cdb_dir))
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdb_dir), use_singletons=False, reconcile_grace_sec=30.0)
    t0 = 1_795_457_276.0

    # 1. Simulate premature RECONCILED_CANCELED (absent_from_verified_open_book) followed 112s later by authoritative FILLED.
    orders: dict = {}
    closed_rows: list = []
    rc: dict = {}
    apply_order_event(
        orders,
        {"ts": t0, "ev": "ACKED", "v": "binance", "sym": "ARBUSDC", "iid": "iid-arb-1", "cid": "AI_arb_1", "vid": "568040423", "side": "SELL", "qty": 4634.3, "px": 0.1153},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    apply_order_event(
        orders,
        {"ts": t0 + 100.0, "ev": "RECONCILED_CANCELED", "v": "binance", "sym": "ARBUSDC", "iid": "iid-arb-1", "cid": "AI_arb_1", "vid": "568040423", "reason": "absent_from_verified_open_book"},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert len(closed_rows) == 1 and closed_rows[0]["state"] == "CANCELED"
    rc = update_recently_closed(rc, closed_rows, eff_now=t0 + 100.0)

    # Late STATUS NEW (non-fill) is still ignored cleanly.
    assert (
        apply_order_event(
            orders,
            {"ts": t0 + 150.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "iid-arb-1", "cid": "AI_arb_1", "vid": "568040423", "vst": "NEW"},
            closed_out=closed_rows,
            recently_closed=rc,
        )
        is None
    )

    # Late authoritative STATUS vst=FILLED MUST NOT be dropped; it archives a FILLED record!
    apply_order_event(
        orders,
        {"ts": t0 + 212.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "iid-arb-1", "cid": "AI_arb_1", "vid": "568040423", "side": "SELL", "vst": "FILLED", "fq": 4634.3, "avg_px": 0.1153, "fee": 0.40},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert len(closed_rows) == 2 and closed_rows[1]["state"] == "FILLED" and closed_rows[1]["fq"] == pytest.approx(4634.3)
    rc = update_recently_closed(rc, closed_rows[1:], eff_now=t0 + 212.0)

    # Duplicate FILLED arriving AFTER the authoritative FILLED IS ignored.
    assert (
        apply_order_event(
            orders,
            {"ts": t0 + 220.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "iid-arb-1", "cid": "AI_arb_1", "vid": "568040423", "vst": "FILLED", "fq": 4634.3, "avg_px": 0.1153},
            closed_out=closed_rows,
            recently_closed=rc,
        )
        is None
    )

    # 2. iter_audit_compat_records still serves permanent orders_closed_*.jsonl even after aged order_events_*.jsonl are pruned (>60d).
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="ETHUSDC",
        ts=t0,
        owner="ai_executor",
        side="BUY",
        qty=1.0,
        px=3000.0,
        iid="old-eth-1",
        cid="AI_old_eth_1",
    )
    writer.record_event(
        OrderEventType.STATUS,
        venue="binance",
        symbol="ETHUSDC",
        ts=t0 + 5.0,
        owner="ai_executor",
        side="BUY",
        qty=1.0,
        px=3000.0,
        fq=1.0,
        avg_px=3000.0,
        iid="old-eth-1",
        cid="AI_old_eth_1",
        vid="999001",
        vst="FILLED",
    )
    reconciler.reconcile_once(now_ts=t0 + 10.0)
    reader = OrderLedgerReader(cachedb_dir=str(cdb_dir))
    for ev_file in reader.list_event_files():
        os.unlink(ev_file)
    assert not reader.list_event_files() and len(reader.list_closed_files()) == 1

    day_str = utc_day_for_ts(t0)
    recs, has_ledger = reader.iter_audit_compat_records(min_day=day_str, max_day=day_str)
    assert has_ledger is True
    assert any(r.get("order_id") == "999001" and r.get("filled_qty") == pytest.approx(1.0) for r in recs)


def test_section_c_monitororder_cooldown_manual_trace_actor_and_macro_ntfy_dedup(tmp_path):
    """Verify C1 (monitororder refusal cooldown), C2 (Kraken WS async REST sync), C3 (MANUAL trace actor), and C4 (macro ntfy stance dedup)."""
    import sys
    import threading
    from types import SimpleNamespace
    import active_intents
    import cacheaccount
    import monitororder
    from intelligence.macro_analyzer import MacroAnalyzer
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisorAssessment

    # C1: monitororder applies 300s backoff on policy refusal and 30s on transient refusal.
    monitororder.reset_monitororder_state_for_test()
    t0 = 1_795_560_000.0
    monitororder._record_reprice_refusal_cooldown("568296122", "qty_zero_after_policy", now_ts=t0)
    cd_pol = monitororder._active_reprice_refusal_cooldown("568296122", now_ts=t0 + 60.0)
    assert cd_pol is not None and cd_pol[0] == pytest.approx(240.0) and cd_pol[1] == "qty_zero_after_policy"
    assert monitororder._active_reprice_refusal_cooldown("568296122", now_ts=t0 + 301.0) is None

    monitororder._record_reprice_refusal_cooldown("568296123", "account_cache_not_fresh", now_ts=t0)
    cd_tr = monitororder._active_reprice_refusal_cooldown("568296123", now_ts=t0 + 10.0)
    assert cd_tr is not None and cd_tr[0] == pytest.approx(20.0)
    assert monitororder._active_reprice_refusal_cooldown("568296123", now_ts=t0 + 31.0) is None
    monitororder.reset_monitororder_state_for_test()

    # C2: Kraken WS loop uses ping_interval=30, ping_timeout=20 and dispatches periodic REST Ledgers sync off the WS reader thread.
    ws_params: dict = {}
    rest_thread_names: list[str] = []
    rest_done = threading.Event()
    stop_ev = threading.Event()

    clock_ms = [1_000_000]

    class _FakeWsApp:
        def __init__(self, url, on_open=None, on_message=None, on_error=None):
            self.on_open = on_open
            self.on_message = on_message
            self.on_error = on_error

        def send(self, _msg):
            pass

        def run_forever(self, **kwargs):
            ws_params.update(kwargs)
            self.on_open(self)
            clock_ms[0] = 1_400_000
            # Trigger periodic REST sync inside keepalive message; on_message must return immediately without blocking.
            self.on_message(self, '{"event":"heartbeat"}')
            rest_done.wait(timeout=3.0)
            stop_ev.set()

    class _FakeKrakenProv:
        PRIVATE_WS_URL = "wss://fake"

        @staticmethod
        def fetch_ledger_fills_by_symbol(_client, *, pairs, lookback_sec):
            rest_thread_names.append(threading.current_thread().name)
            if len(rest_thread_names) >= 2:
                rest_done.set()
            return {"HYPEUSD": []}

        @staticmethod
        def build_ws_own_trades_subscription(_client):
            return '{"event":"subscribe"}'

        @staticmethod
        def extract_ws_own_trades(_data, *, pairs):
            return None

        @staticmethod
        def classify_ws_control_event(_data):
            return "keepalive", ""

    kr_cache = tmp_path / "cachedb" / "orders" / "cache_trade_kraken.json"
    kr_cache.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(cacheaccount.MarketDataProvider, "resolve_venue_provider_class", return_value=_FakeKrakenProv), \
         patch.dict(sys.modules, {"websocket": SimpleNamespace(WebSocketApp=_FakeWsApp)}), \
         patch.object(cacheaccount, "resolve_now_ms", side_effect=lambda v=None: v if v is not None else clock_ms[0]):
        cacheaccount.run_kraken_trade_ws_loop(
            object(),
            cache_file=str(kr_cache),
            pairs=["HYPEUSD"],
            poll_interval=30.0,
            backoff_init=5.0,
            backoff_max=60.0,
            lookback_sec=3600.0,
            stop_event=stop_ev,
        )
    assert ws_params.get("ping_interval") == 30 and ws_params.get("ping_timeout") == 20
    assert "kraken-ws-rest-sync" in rest_thread_names

    # C3: trace_order_or_intent resolves MANUAL actor for Binance Android manual orders (and_*).
    ws_dir = tmp_path / "trace_ws"
    cdb_dir = ws_dir / "cachedb"
    writer = OrderLedgerWriter(cachedb_dir=str(cdb_dir))
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdb_dir), use_singletons=False, reconcile_grace_sec=30.0)
    writer.record_event(
        OrderEventType.ORPHAN_ADOPTED,
        venue="binance",
        symbol="TAOUSDC",
        ts=t0,
        owner="manual",
        side="BUY",
        order_type="LIMIT",
        qty=11.0,
        px=174.2,
        cid="and_6cacf9b64fd049878f47206773d2f93b",
        vid="729245677",
        vst="NEW",
    )
    reconciler.reconcile_once(now_ts=t0 + 5.0)
    tr = active_intents.trace_order_or_intent("729245677", root=str(ws_dir))
    assert tr["found"] is True
    assert tr["actor"] == "MANUAL"

    # C4: MacroAnalyzer suppresses repeated ntfy alerts when stance/regime is unchanged, but notifies immediately on change.
    ma_dir = tmp_path / "macro_cache"
    ma_dir.mkdir(parents=True, exist_ok=True)
    analyzer = MacroAnalyzer(cache_dir=str(ma_dir), advisor_interval_sec=1800.0, macro_llm_interval_sec=3600.0)
    analyzer.shadow_notify = True
    adv1 = SentimentAdvisorAssessment("CAUTION", "MODERATE", 0.7, "Steady.", [], "HOLD", ts=t0)
    adv2_same = SentimentAdvisorAssessment("CAUTION", "MODERATE", 0.72, "Still steady.", [], "HOLD", ts=t0 + 3700.0)
    adv3_chg = SentimentAdvisorAssessment("BEARISH", "HIGH", 0.85, "Risk-off shift.", [], "DEFENSIVE", ts=t0 + 5500.0)
    geo1 = GeopoliticalThreatAssessment("NORMAL", 0.1, "Calm.", "NONE", 3, ts=t0, is_fallback=False)
    geo2_same = GeopoliticalThreatAssessment("NORMAL", 0.12, "Still calm.", "NONE", 3, ts=t0 + 7200.0, is_fallback=False)
    geo3_chg = GeopoliticalThreatAssessment("ELEVATED", 0.65, "Escalation.", "DOWNSCALE_50", 5, ts=t0 + 9000.0, is_fallback=False)

    with patch("notify_engine.alertnotifiers.notify") as mock_notify:
        with patch.object(analyzer.sentiment_advisor, "review", return_value=adv1):
            analyzer.assess_market_advisor(force=False, now=t0)
        with patch.object(analyzer.sentiment_advisor, "review", return_value=adv2_same):
            analyzer.assess_market_advisor(force=False, now=t0 + 3700.0)
        assert mock_notify.call_count == 1  # Unchanged HOLD (CAUTION) after 1h suppressed!
        with patch.object(analyzer.sentiment_advisor, "review", return_value=adv3_chg):
            analyzer.assess_market_advisor(force=False, now=t0 + 5500.0)
        assert mock_notify.call_count == 2  # Genuine shift to DEFENSIVE (BEARISH) notified immediately!

        with patch.object(analyzer.geopolitical_analyzer, "assess", return_value=geo1):
            analyzer.assess_geopolitical_threat(force=False, now=t0)
        with patch.object(analyzer.geopolitical_analyzer, "assess", return_value=geo2_same):
            analyzer.assess_geopolitical_threat(force=False, now=t0 + 7200.0)
        assert mock_notify.call_count == 3  # Unchanged NONE (NORMAL) after 2h suppressed!
        with patch.object(analyzer.geopolitical_analyzer, "assess", return_value=geo3_chg):
            analyzer.assess_geopolitical_threat(force=False, now=t0 + 9000.0)
        assert mock_notify.call_count == 4  # Escalation to DOWNSCALE_50 (ELEVATED) notified immediately!
