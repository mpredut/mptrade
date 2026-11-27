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
    from providers.ai_attribution_store import load_and_evaluate_ai_attribution
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

    # C1: monitororder classifies refusal cooldowns into 3 tiers (900s quota/structural, 120s market guard, 30s transient)
    # and Instrument.place honors G_QLIMIT1 bypass on BUY for monitor_order_replace without weakening normal BUY guards.
    monitororder.reset_monitororder_state_for_test()
    t0 = 1_795_560_000.0
    monitororder._record_reprice_refusal_cooldown("568296122", "qty_zero_after_policy", now_ts=t0)
    cd_pol = monitororder._active_reprice_refusal_cooldown("568296122", now_ts=t0 + 60.0)
    assert cd_pol is not None and cd_pol[0] == pytest.approx(840.0) and cd_pol[1] == "qty_zero_after_policy" and cd_pol[2] == "quota_or_structural"
    assert monitororder._active_reprice_refusal_cooldown("568296122", now_ts=t0 + 901.0) is None

    monitororder._record_reprice_refusal_cooldown("568296124", "trend_deferred", now_ts=t0)
    cd_mkt = monitororder._active_reprice_refusal_cooldown("568296124", now_ts=t0 + 20.0)
    assert cd_mkt is not None and cd_mkt[0] == pytest.approx(100.0) and cd_mkt[2] == "market_guard"
    assert monitororder._active_reprice_refusal_cooldown("568296124", now_ts=t0 + 121.0) is None

    monitororder._record_reprice_refusal_cooldown("568296123", "account_cache_not_fresh", now_ts=t0)
    cd_tr = monitororder._active_reprice_refusal_cooldown("568296123", now_ts=t0 + 10.0)
    assert cd_tr is not None and cd_tr[0] == pytest.approx(20.0) and cd_tr[2] == "transient"
    assert monitororder._active_reprice_refusal_cooldown("568296123", now_ts=t0 + 31.0) is None
    monitororder.reset_monitororder_state_for_test()

    # Verify Instrument._init_and_gate_pre_pipeline preserves G_QLIMIT1 bypass on BUY for monitor_order_replace only.
    from instrument import Instrument

    fake_prov = SimpleNamespace(
        name="Binance",
        execution_enabled=True,
        validate_symbol=lambda _s: None,
        requires_cache_permit=lambda: False,
        defers_smart_cancellation=lambda: False,
        supports_business_minimum=lambda: True,
        supports_post_only=lambda: False,
        supports_order_validity=lambda: False,
        guards_internally=lambda: False,
    )
    inst = Instrument("ARBUSDC", "ARBUSDC", "Binance", api=SimpleNamespace(provider_by_name=lambda _n: fake_prov))
    ctx_rep, err_rep = inst._init_and_gate_pre_pipeline(
        "BUY", 0.1095, 1805.0,
        {"kind": "monitor_order_replace", "bypass_guards": ["G_PROF", "GT_S", "GLLM1", "G_QLIMIT1"], "replaces_order_id": "568296122"},
    )
    assert err_rep is None and ctx_rep is not None and ctx_rep.bypass_quantity_policy is True
    ctx_new, err_new = inst._init_and_gate_pre_pipeline(
        "BUY", 0.1095, 1805.0,
        {"kind": "entry", "bypass_guards": ["G_QLIMIT1"]},
    )
    assert err_new is None and ctx_new is not None and ctx_new.bypass_quantity_policy is False

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

    with patch("notify_engine.ntfyapi.notify") as mock_notify:
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


def test_bug2_intelligence_guard_reason_and_qlimit_bypass_log_and_bug3_caller_owns_retry_acked_and_hl_ai_owner(tmp_path, capsys):
    import instrument as inst_mod
    import order_guard
    from cachepaths import AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME

    # Bug #2: When G_PROF + G_QLIMIT1 are both bypassed and Stage-2 intelligence guard blocks,
    # Instrument logs that BOTH were bypassed and reports reason="intelligence_guard" (not "profit_guard").
    fake_prov = MagicMock()
    fake_prov.name = "Binance"
    fake_prov.guards_internally.return_value = False
    fake_prov.requires_cache_permit.return_value = False
    fake_prov.defers_smart_cancel.return_value = False
    fake_prov.free_balance.return_value = 1000.0
    fake_prov.get_current_price.return_value = 0.12
    fake_prov.round_price.side_effect = lambda _s, px: px
    fake_prov.quantity_decision.return_value = SimpleNamespace(
        final_qty=100.0, refuse_reason=None, balance_asset="USDC",
        scale_applied=False, applied_scale=1.0, wallet_constrained=False,
    )
    fake_prov.place_order.return_value = {"orderId": "568399001", "status": "NEW"}
    fake_api = MagicMock()
    fake_api.market_regime_context.return_value = None
    inst = inst_mod.Instrument(
        name="ARB", symbol="ARBUSDC", provider="Binance", base="ARB", quote="USDC", api=fake_api
    )
    inst._provider = fake_prov

    outcome_ctx: dict = {}
    with (
        patch.object(order_guard, "daily_limit_guard", return_value=(True, None)),
        patch.object(order_guard, "get_margin_cfg", return_value=0.01),
        patch.object(order_guard, "check_intelligence_guards", return_value=(False, "severe_taker_selling_dominated", 1.0)),
        patch.object(inst_mod.order_lifecycle, "log_order_outcome"),
    ):
        res = inst.place(
            "BUY",
            0.12,
            100.0,
            smart=False,
            wait_for_trend=False,
            caller_owns_retry=True,
            kind="monitor_order_replace",
            bypass_guards=["G_PROF", "GT_S", "GLLM1", "G_QLIMIT1"],
            _outcome_context=outcome_ctx,
        )
    assert res is None
    assert outcome_ctx["reason"] == "intelligence_guard"
    out = capsys.readouterr().out
    assert "the historical price reference and quantity/weight guard were explicitly bypassed" in out
    assert "quantity/weight guard remains active" not in out

    # Bug #3a: caller_owns_retry=True without intent_id (e.g. monitororder MO_MN_OR_*) emits ACKED to OrderLedger.
    cdb = str(tmp_path / "cachedb")
    os.makedirs(os.path.join(cdb, "orders"), exist_ok=True)
    os.makedirs(os.path.join(cdb, "state"), exist_ok=True)
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    with (
        patch.object(order_guard, "daily_limit_guard", return_value=(True, None)),
        patch.object(order_guard, "get_margin_cfg", return_value=0.01),
        patch.object(order_guard, "check_intelligence_guards", return_value=(True, "ok", 1.0)),
        patch.object(inst_mod.order_lifecycle, "log_order_outcome"),
        patch.object(inst_mod, "get_default_ledger_writer", return_value=writer),
    ):
        placed = inst.place(
            "BUY",
            0.12,
            100.0,
            smart=False,
            wait_for_trend=False,
            caller_owns_retry=True,
            client_order_id="MO_MN_OR_TEST_01",
            bypass_guards=["G_PROF", "GT_S", "GLLM1", "G_QLIMIT1"],
        )
    assert placed is not None
    active_after_ack = reader.list_active_orders(apply_tail=True)
    assert any(r["cid"] == "MO_MN_OR_TEST_01" and r["vid"] == "568399001" for r in active_after_ack)

    # Bug #3b: OrderLedgerReconciler resolves known local AI 128-bit hex cloid on Hyperliquid as owner="ai_executor"
    # while keeping genuinely foreign 128-bit hex cloids as owner="external".
    t0 = 1_795_570_000.0
    known_cid = "AI_intent_1795570000_buy_hypeusdc"
    known_cloid = derive_venue_128bit_client_order_id("hyperliquid", known_cid)
    foreign_cloid = derive_venue_128bit_client_order_id("hyperliquid", "AI_intent_foreign_host_order")
    with open(os.path.join(cdb, "state", AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "intents": [
                    {
                        "intent_id": "intent_1795570000_buy_hypeusdc",
                        "venue": "hyperliquid",
                        "symbol": "HYPEUSDC",
                        "client_order_id": known_cid,
                        "status": "OPEN",
                    }
                ]
            },
            fh,
        )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        os.path.join(cdb, "orders", CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {
            "HYPEUSDC": [
                {"orderId": "700001", "clientOrderId": known_cloid, "side": "BUY", "price": 58.0, "origQty": 10.0, "executedQty": 0.0, "status": "OPEN", "timestamp": int(t0 * 1000)},
                {"orderId": "700002", "clientOrderId": foreign_cloid, "side": "BUY", "price": 58.1, "origQty": 10.0, "executedQty": 0.0, "status": "OPEN", "timestamp": int(t0 * 1000)},
            ]
        },
        now_ms=int((t0 + 2) * 1000),
        verified_ts=t0 + 2.0,
    )
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)
    reconciler.reconcile_once(now_ts=t0 + 3.0)
    by_vid = {r["vid"]: r for r in reader.list_active_orders(apply_tail=False)}
    assert by_vid["700001"]["owner"] == "ai_executor"
    assert by_vid["700002"]["owner"] == "external"


def test_retry_giveup_audit_event_expires_deferred_intent_from_active_ledger(tmp_path):
    """Verify retry_giveup mirrors into RECONCILED_EXPIRED so given-up outbox intents do not linger 72h."""
    cdb = str(tmp_path / "cachedb")
    t0 = 1_795_620_000.0
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)

    assert writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0,
        owner="tradeall",
        side="SELL",
        order_type="LIMIT",
        iid="outbox-deadbeef01",
        kind="accel_plunge_down",
        reason="intelligence_guard",
        qty=10.61,
        px=57.633,
        deferred=True,
        src="instrument_guard",
    )
    reconciler.reconcile_once(now_ts=t0 + 5.0)
    assert any(r.get("iid") == "outbox-deadbeef01" for r in reader.list_active_orders(apply_tail=False))

    assert writer.record_audit_event(
        "retry_giveup",
        intent_id="outbox-deadbeef01",
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0 + 60.0,
        fields={
            "side": "sell",
            "qty": 10.61,
            "expiry_reason": "active_ttl",
            "last_failure_reason": "intelligence_guard",
        },
    )
    reconciler.reconcile_once(now_ts=t0 + 65.0)
    assert not any(r.get("iid") == "outbox-deadbeef01" for r in reader.list_active_orders(apply_tail=False))
    # Unsubmitted deferred intent must not pollute closed-trade history
    assert reader.list_closed_files() == []


def test_compound_terminal_non_fill_statuses_and_ai_orphan_iid_adoption(tmp_path):
    """Compound terminal non-fill venue statuses classify as terminal (never open), and AI orphan adoption populates iid."""
    from providers.order_ledger import OrderEventType, ledger_row_to_audit_record
    from providers.order_status_ssot import is_cancel_or_expire_status, is_terminal_audit_status

    for vst, exp_local in (
        ("MINTRADENTLREJECTED", "rejected"),
        ("BADALOPXREJECTED", "rejected"),
        ("ARCHIVED_CANCELED", "canceled"),
        ("EXPIRED_IN_MATCH", "expired"),
    ):
        assert is_cancel_or_expire_status(vst) is True
        assert is_terminal_audit_status(vst) is True
        rec = ledger_row_to_audit_record(
            {
                "ts": 1_795_500_000.0,
                "ev": OrderEventType.STATUS.value,
                "v": "hyperliquid",
                "sym": "HYPEUSDC",
                "vid": "999",
                "vst": vst,
            }
        )
        assert rec is not None
        assert rec["event"] == "order_terminal"
        assert rec["status"] == exp_local
    # Composite status stays gated by include_composite
    assert is_cancel_or_expire_status("CANCELED_OR_EXPIRED", include_composite=False) is False
    assert is_cancel_or_expire_status("CANCELED_OR_EXPIRED", include_composite=True) is True

    # AI orphan adoption populates iid from auto_order_intents_state.json (and AI_intent_ cid prefix)
    from cachepaths import (
        AUTO_ORDER_INTENTS_STATE_FILENAME,
        CACHE_BINANCE_ORDER_OPEN_FILENAME,
        resolve_cachedb_filepath,
    )

    t0 = 1_795_500_100.0
    cdir = tmp_path / "cachedb"
    intent_id = "intent_1795500000_ab12"
    intents_path = resolve_cachedb_filepath(os.path.join(str(cdir), AUTO_ORDER_INTENTS_STATE_FILENAME))
    os.makedirs(os.path.dirname(intents_path), exist_ok=True)
    with open(intents_path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "intent_id": intent_id,
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "decision": "BUY",
                    "status": "SUBMIT_UNKNOWN",
                    "execution_payload": {"orderId": "888777"},
                }
            ],
            fh,
        )
    oo_path = resolve_cachedb_filepath(os.path.join(str(cdir), CACHE_BINANCE_ORDER_OPEN_FILENAME))
    os.makedirs(os.path.dirname(oo_path), exist_ok=True)
    with open(oo_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "verified_ts": t0,
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "888777",
                            "clientOrderId": f"AI_{intent_id}",
                            "symbol": "BTCUSDC",
                            "side": "BUY",
                            "type": "LIMIT",
                            "origQty": "0.01",
                            "executedQty": "0",
                            "price": "70000",
                            "status": "NEW",
                        }
                    ]
                },
            },
            fh,
        )
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdir), use_singletons=False)
    reader = OrderLedgerReader(cachedb_dir=str(cdir))
    stats = reconciler.reconcile_once(now_ts=t0 + 2.0)
    assert stats["orphans_adopted"] == 1
    active = reader.list_active_orders(apply_tail=False)
    assert len(active) == 1
    assert active[0]["owner"] == "ai_executor"
    assert active[0]["iid"] == intent_id


def test_superseded_outbox_deferred_intent_expires_once_absent_from_retry_queue(tmp_path):
    """Deferred outbox-* GUARD_BLOCKED intents stay active while queued, then expire once consolidated/evicted from order_retry_queue.jsonl."""
    from state_stores.order_retry_queue import OrderRetryQueueStore

    cdb = str(tmp_path / "cachedb")
    t0 = 1_795_620_000.0
    q_path = OrderRetryQueueStore.resolve_path(cdb)
    os.makedirs(os.path.dirname(q_path), exist_ok=True)
    OrderRetryQueueStore.write_records_nolock(
        q_path,
        [
            {
                "id": "7032f7b37bd64dfeb7a6271fec455265",
                "intent_id": "outbox-7032f7b37bd64dfeb7a6271fec455265",
                "provider_name": "hyperliquid",
                "symbol": "HYPEUSDC",
                "side": "SELL",
                "qty": 10.61,
                "lifecycle": "submit_pending",
                "last_failure_reason": "intelligence_guard",
            }
        ],
    )

    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)

    assert writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0,
        owner="tradeall",
        side="SELL",
        order_type="LIMIT",
        iid="outbox-7032f7b37bd64dfeb7a6271fec455265",
        kind="accel_plunge_down",
        reason="intelligence_guard",
        qty=10.61,
        px=57.633,
        deferred=True,
        src="instrument_guard",
    )

    # Still present in order_retry_queue.jsonl past reconcile_grace_sec -> remains active in orders_active.json
    reconciler.reconcile_once(now_ts=t0 + 120.0)
    assert any(
        r.get("iid") == "outbox-7032f7b37bd64dfeb7a6271fec455265"
        for r in reader.list_active_orders(apply_tail=False)
    )

    # Consolidated/evicted from order_retry_queue.jsonl without submit -> expires cleanly on next reconcile pass
    OrderRetryQueueStore.write_records_nolock(
        q_path,
        [
            {
                "id": "f835d2062cb840c08a1a432a1a302eee",
                "intent_id": "intent_1795673120_buy_hypeusdc",
                "provider_name": "hyperliquid",
                "symbol": "HYPEUSDC",
                "side": "BUY",
                "qty": 8.29,
                "lifecycle": "accepted",
            }
        ],
    )
    reconciler.reconcile_once(now_ts=t0 + 125.0)
    assert not any(
        r.get("iid") == "outbox-7032f7b37bd64dfeb7a6271fec455265"
        for r in reader.list_active_orders(apply_tail=False)
    )
    trace = reader.trace_order("outbox-7032f7b37bd64dfeb7a6271fec455265")
    assert any(
        e.get("ev") == OrderEventType.RECONCILED_EXPIRED.value
        and e.get("reason") == "absent_from_retry_outbox"
        for e in trace["events"]
    )
    assert reader.list_closed_files() == []


def test_terminal_closed_ledger_normalizes_contradictory_non_terminal_vst(tmp_path):
    """Absence-reconciled and historical closed ledger rows never preserve contradictory NEW/OPEN vst."""
    from providers.order_ledger import (
        OrderEventType,
        OrderLedgerState,
        apply_order_event,
        build_order_event,
        ledger_row_to_audit_record,
    )

    # 1. apply_order_event normalizes stale NEW/OPEN vst when archiving a terminal closed summary
    orders: dict = {}
    closed: list = []
    t0 = 1_795_720_000.0
    apply_order_event(
        orders,
        build_order_event(
            OrderEventType.ACKED.value,
            ts=t0,
            venue="binance",
            symbol="ARBUSDC",
            owner="ai_executor",
            side="SELL",
            order_type="LIMIT",
            qty=2353.9,
            px=0.115,
            iid="intent_test_cancel_vst",
            vid="568763607",
            vst="NEW",
        ),
        closed_out=closed,
    )
    apply_order_event(
        orders,
        build_order_event(
            OrderEventType.RECONCILED_CANCELED.value,
            ts=t0 + 120.0,
            venue="binance",
            symbol="ARBUSDC",
            owner="ai_executor",
            side="SELL",
            iid="intent_test_cancel_vst",
            vid="568763607",
            reason="absent_from_verified_open_book",
        ),
        closed_out=closed,
    )
    assert len(closed) == 1
    assert closed[0]["state"] == OrderLedgerState.CANCELED.value
    assert closed[0]["vst"] == OrderLedgerState.CANCELED.value

    # 2. Historical already-archived closed rows with stale NEW/OPEN vst normalize cleanly on audit adaptation
    adapted_cancel = ledger_row_to_audit_record(
        {
            "v": "hyperliquid",
            "sym": "HYPEUSDC",
            "side": "SELL",
            "state": "CANCELED",
            "vst": "OPEN",
            "vid": "617365492660",
            "iid": "intent_1795710217_sell_hypeusdc",
            "closed_ts": t0 + 180.0,
        },
        is_closed=True,
    )
    assert adapted_cancel is not None
    assert adapted_cancel["event"] == "order_terminal"
    assert adapted_cancel["status"] == "canceled"
    assert adapted_cancel["venue_status"] == "CANCELED"

    # Compound terminal venue statuses (e.g. EXPIRED_IN_MATCH / MINTRADENTLREJECTED) stay preserved
    adapted_compound = ledger_row_to_audit_record(
        {
            "v": "hyperliquid",
            "sym": "HYPEUSDC",
            "side": "SELL",
            "state": "REJECTED",
            "vst": "MINTRADENTLREJECTED",
            "vid": "617365492661",
            "closed_ts": t0 + 185.0,
        },
        is_closed=True,
    )
    assert adapted_compound is not None
    assert adapted_compound["venue_status"] == "MINTRADENTLREJECTED"

    # 3. OrderLedgerReconciler absence-confirmation emits terminal vst on the ledger event and closed summary
    from cachepaths import CACHE_BINANCE_ORDER_OPEN_FILENAME, resolve_cachedb_filepath

    cdb = str(tmp_path / "cachedb")
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)
    assert writer.record_event(
        OrderEventType.ACKED,
        venue="binance",
        symbol="ARBUSDC",
        ts=t0,
        owner="ai_executor",
        side="SELL",
        order_type="LIMIT",
        qty=100.0,
        px=0.12,
        iid="intent_abs_reconcile",
        vid="999001",
        vst="NEW",
    )
    oo_path = resolve_cachedb_filepath(os.path.join(cdb, CACHE_BINANCE_ORDER_OPEN_FILENAME))
    os.makedirs(os.path.dirname(oo_path), exist_ok=True)
    with open(oo_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "verified_ts": t0 + 90.0,
                "symbol_verified_ts": {"ARBUSDC": t0 + 90.0},
                "items": {"ARBUSDC": []},
            },
            fh,
        )
    reconciler.reconcile_once(now_ts=t0 + 95.0)
    trace = reader.trace_order("999001")
    assert any(
        e.get("ev") == OrderEventType.RECONCILED_CANCELED.value and e.get("vst") == "CANCELED"
        for e in trace["events"]
    )
    assert trace["closed_orders"] and trace["closed_orders"][-1]["vst"] == "CANCELED"


def test_order_ledger_pass5_invariants_and_reload_fallback(tmp_path, monkeypatch):
    """Pass 5 invariants: strip non-positive snapshot ts, never pin failed reload sigs, and report error heartbeats."""
    import threading
    import order_ledger_reconciler_worker as worker_mod
    from cacheaccount import CacheOrderFilledManager, CacheOrderOpenManager
    from cachepaths import CACHE_BINANCE_ORDER_FILLED_FILENAME, CACHE_BINANCE_ORDER_OPEN_FILENAME, resolve_cachedb_filepath

    t0 = 1_795_800_000.0
    cdb = str(tmp_path / "cachedb")
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)

    # 1. Invariant: read_active_snapshot strips <= 0 / corrupt timestamps from verified maps and recently_closed.
    writer.write_active_snapshot(
        {
            "schema_version": 1,
            "updated_ts": t0,
            "journal_cursor": {"file": "", "offset": 0},
            "venue_verified_ts": {"binance": t0, "kraken": 0, "hyperliquid": "bad", "t212": -5},
            "symbol_verified_ts": {"binance:btcusdc": t0, "binance:ethusdc": 0.0, "kraken:hypeusd": None},
            "recently_closed": {"binance:vid:1": t0, "binance:vid:0": 0.0, "binance:vid:bad": "nan"},
            "orders": {},
        }
    )
    snap = reader.read_active_snapshot()
    assert snap["venue_verified_ts"] == {"binance": t0}
    assert snap["symbol_verified_ts"] == {"binance:btcusdc": t0}
    assert snap["recently_closed"] == {"binance:vid:1": t0}

    # 2. Dependent cross-check: failed _reload_snapshot_from_disk MUST NOT pin _file_sigs and MUST fall back
    # to disk fills so a filled order absent from open_orders is reconciled as FILLED, never CANCELED.
    assert writer.record_event(
        OrderEventType.ACKED,
        venue="binance",
        symbol="BTCUSDC",
        ts=t0,
        owner="ai_executor",
        side="BUY",
        order_type="LIMIT",
        qty=0.5,
        px=60000.0,
        iid="iid_reload_race",
        vid="777001",
        vst="NEW",
    )
    oo_path = resolve_cachedb_filepath(os.path.join(cdb, CACHE_BINANCE_ORDER_OPEN_FILENAME))
    of_path = resolve_cachedb_filepath(os.path.join(cdb, CACHE_BINANCE_ORDER_FILLED_FILENAME))
    os.makedirs(os.path.dirname(oo_path), exist_ok=True)
    with open(oo_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "verified_ts": t0 + 90.0,
                "symbol_verified_ts": {"BTCUSDC": t0 + 90.0},
                "items": {"BTCUSDC": []},
            },
            fh,
        )
    with open(of_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "fetchtime": {"BTCUSDC": int((t0 + 90.0) * 1000)},
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "777001",
                            "price": 60000.0,
                            "quantity": 0.5,
                            "timestamp": int((t0 + 60.0) * 1000),
                            "side": "BUY",
                            "status": "FILLED",
                        }
                    ]
                },
            },
            fh,
        )

    reload_attempts = {"filled": 0}
    fail_first = {"flag": True}

    class _FlakyFilledSingleton:
        filename = of_path
        lock = threading.Lock()
        cache = {}  # Stale RAM missing order 777001

        def _reload_snapshot_from_disk(self) -> bool:
            reload_attempts["filled"] += 1
            if fail_first["flag"]:
                return False  # Simulates order_cache_reader_not_current
            self.cache = {
                "BTCUSDC": [
                    {
                        "orderId": "777001",
                        "price": 60000.0,
                        "quantity": 0.5,
                        "timestamp": int((t0 + 60.0) * 1000),
                        "side": "BUY",
                        "status": "FILLED",
                    }
                ]
            }
            return True

    class _FreshOpenSingleton:
        filename = oo_path
        lock = threading.Lock()
        cache = {"BTCUSDC": []}
        verified_ts = t0 + 90.0
        symbol_verified_ts = {"BTCUSDC": t0 + 90.0}

        def _reload_snapshot_from_disk(self) -> bool:
            return True

    flaky_filled = _FlakyFilledSingleton()
    fresh_open = _FreshOpenSingleton()

    rec = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=True, reconcile_grace_sec=30.0)
    rec._explicit_cachedb_dir = None  # Exercise singleton path in test
    monkeypatch.setattr(worker_mod, "get_order_filled_manager", lambda v: flaky_filled if v == "binance" else None)
    monkeypatch.setattr(worker_mod, "get_order_open_manager", lambda v: fresh_open if v == "binance" else None)

    stats1 = rec.reconcile_once(now_ts=t0 + 95.0)
    assert reload_attempts["filled"] == 1
    assert of_path not in rec._file_sigs  # Failed reload MUST NOT pin signature!
    assert stats1["reconciled_filled"] == 1
    trace = reader.trace_order("777001")
    assert trace["closed_orders"] and trace["closed_orders"][-1]["state"] == "FILLED"

    # Next tick retries reload (since signature wasn't pinned), succeeds, and caches signature.
    fail_first["flag"] = False
    rec.reconcile_once(now_ts=t0 + 100.0)
    assert reload_attempts["filled"] == 2
    assert of_path in rec._file_sigs
    rec.reconcile_once(now_ts=t0 + 105.0)
    assert reload_attempts["filled"] == 2  # Unchanged file now skipped after clean reload

    # 3. Direct manager invariant: _reload_snapshot_from_disk returns False on missing/error reload.
    from cacheaccount import CacheBalanceManager

    bad_open = CacheOrderOpenManager(60, ["BTCUSDC"], str(tmp_path / "missing_open.json"), read_only=True, venue="binance")
    assert bad_open._reload_snapshot_from_disk() is False
    bad_bal = CacheBalanceManager(60, ["BTC"], str(tmp_path / "missing_bal.json"), read_only=True, venue="binance")
    assert bad_bal._reload_snapshot_from_disk() is False
    bad_filled = CacheOrderFilledManager(
        60, ["BTCUSDC"], str(tmp_path / "corrupt_filled.json"), read_only=True, venue="binance"
    )
    monkeypatch.setattr(bad_filled, "_load_versioned_account_meta", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert bad_filled._reload_snapshot_from_disk() is False

    # 4. Heartbeat invariant: main() never reports {"status": "ok"} when reconcile_once() raises.
    hb_calls: list[dict] = []

    class _ExplodingReconciler:
        reconcile_grace_sec = 30.0
        intent_stale_sec = 60.0

        def reconcile_once(self):
            raise RuntimeError("disk_io_timeout")

    monkeypatch.setattr(worker_mod, "OrderLedgerReconciler", _ExplodingReconciler)
    monkeypatch.setattr(worker_mod, "single_instance", lambda *_a, **_kw: None)
    monkeypatch.setattr(worker_mod, "_touch_ledger_hb", lambda st, **kw: hb_calls.append(dict(st)))
    monkeypatch.setattr(worker_mod.sys, "argv", ["order_ledger_reconciler_worker.py"])
    monkeypatch.setattr(worker_mod.time, "sleep", lambda _s: (_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit):
        worker_mod.main()
    assert len(hb_calls) >= 2
    assert hb_calls[-1].get("status") == "error"
    assert "disk_io_timeout" in str(hb_calls[-1].get("error") or "")


def test_pass19_order_ledger_truthy_zero_ts_invariants_and_crosscheck():
    """Pass 19 invariant & cross-check: make_closed_summary and ledger_row_to_audit_record reject truthy '0' timestamps instead of shadowing valid created_ts/closed_ts."""
    from providers.order_ledger import ledger_row_to_audit_record, make_closed_summary
    t_created = 1_795_000_100.0
    t_closed = 1_795_000_250.5

    # 1. Pure invariant: make_closed_summary replaces corrupt "0" / negative created_ts with closed_ts
    summary_bad = make_closed_summary(
        "k1",
        {"v": "binance", "sym": "BTCUSDC", "state": "FILLED", "created_ts": "0"},
        closed_ts=t_closed,
    )
    assert summary_bad["created_ts"] == pytest.approx(t_closed)

    summary_neg = make_closed_summary(
        "k2",
        {"v": "binance", "sym": "BTCUSDC", "state": "CANCELED", "created_ts": -10.0},
        closed_ts=t_closed,
    )
    assert summary_neg["created_ts"] == pytest.approx(t_closed)

    summary_ok = make_closed_summary(
        "k3",
        {"v": "binance", "sym": "BTCUSDC", "state": "FILLED", "created_ts": t_created},
        closed_ts=t_closed,
    )
    assert summary_ok["created_ts"] == pytest.approx(t_created)

    # 2. Dependent cross-check: ledger_row_to_audit_record recovers created_ts when updated_ts is truthy "0"
    adapted = ledger_row_to_audit_record(
        {
            "v": "hyperliquid",
            "sym": "HYPEUSDC",
            "side": "BUY",
            "state": "FILLED",
            "vst": "FILLED",
            "vid": "617441855947",
            "fq": 2.5,
            "avg_px": 65.0,
            "updated_ts": "0",
            "created_ts": t_created,
        },
        is_closed=True,
    )
    assert adapted is not None
    assert adapted["ts"] == pytest.approx(t_created)
    assert adapted["event"] == "order_terminal"
    assert adapted["status"] == "closed"


def test_pass22_pnl_report_facade_and_backfill_invariants_and_crosschecks(tmp_path, monkeypatch, capsys):
    """Pass 22 pure invariants & cross-checks: pnl_report ignores unfilled canceled orders, verify_account_facade fails closed when all symbols skip, and backfill guards future live boundary/mtime."""
    import time
    from types import SimpleNamespace
    from verify_tools import backfill_order_ledger_from_audit as bfl
    from verify_tools import pnl_report
    from verify_tools import verify_account_facade as vaf

    # 1. Pure invariant & cross-check: pnl_report.per_bot must NEVER book canceled/unfilled origQty as filled P&L
    canceled_order = {
        "orderId": 1,
        "clientOrderId": "AI_cancel_1",
        "side": "BUY",
        "status": "CANCELED",
        "origQty": "1.5",
        "executedQty": "0.00000000",
        "cummulativeQuoteQty": "0.00000000",
        "price": "60000.0",
    }
    partial_canceled = {
        "orderId": 2,
        "clientOrderId": "AI_part_2",
        "side": "BUY",
        "status": "CANCELED",
        "origQty": "1.0",
        "executedQty": "0.25",
        "cummulativeQuoteQty": "15000.0",
        "price": "60000.0",
    }
    filled_sell = {
        "orderId": 3,
        "clientOrderId": "AI_sell_3",
        "side": "SELL",
        "status": "FILLED",
        "origQty": "0.25",
        "executedQty": "0.25",
        "cummulativeQuoteQty": "15500.0",
        "price": "62000.0",
    }
    assert pnl_report._extract_attributed_fill_qty(canceled_order, src_kind="orders") == 0.0
    assert pnl_report._extract_attributed_fill_qty(partial_canceled, src_kind="orders") == pytest.approx(0.25)
    assert pnl_report._extract_attributed_fill_qty({"qty": "0.5", "price": "60000.0"}, src_kind="fills") == pytest.approx(0.5)

    fake_prov = SimpleNamespace(fetch_recent_orders=lambda _s, limit=1000: [canceled_order, partial_canceled, filled_sell])
    monkeypatch.setattr(pnl_report.mkt, "provider_by_name", lambda _v: fake_prov)
    monkeypatch.setattr(pnl_report, "get_accepted_order_owner_map", lambda: {})
    pnl_report.per_bot("BTCUSDC", provider_name="binance")
    out_pnl = capsys.readouterr().out
    assert "15000" in out_pnl and "15500" in out_pnl
    assert "+500.00" in out_pnl  # 0.25 * (62000 - 60000), NOT poisoned by 1.5 BTC canceled order!
    assert "105000" not in out_pnl

    # 2. Pure invariant & cross-check: verify_account_facade exits non-zero when ALL symbols skip due to unavailable balances
    monkeypatch.setattr(vaf, "SYMBOLS", ["BTCUSDC", "ETHUSDC"])
    monkeypatch.setattr(vaf, "free_direct", lambda _b: 0.0)
    monkeypatch.setattr(vaf.mkt, "provider_by_name", lambda _v: SimpleNamespace(free_balance=lambda _b: None))
    with pytest.raises(SystemExit) as exc_info:
        vaf.main()
    assert exc_info.value.code == 1
    out_vaf = capsys.readouterr().out
    assert "inconclusive" in out_vaf.lower()

    # 3. Pure invariant & cross-check: backfill_order_ledger picks earliest non-future live boundary & ignores far-future source mtime
    t_now = time.time()
    audit_dir = tmp_path / "audit"
    cdb_dir = tmp_path / "cachedb"
    audit_dir.mkdir(parents=True)
    reader = OrderLedgerReader(cachedb_dir=str(cdb_dir))
    day_live = bfl.utc_day_for_ts(t_now - 3600.0)
    ev_live_path = reader.events_path(day_live)
    os.makedirs(os.path.dirname(ev_live_path), exist_ok=True)
    # Unsorted live events + poisoned far-future live row
    with open(ev_live_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": t_now + 86400.0, "src": "live", "ev": "STATUS"}) + "\n")
        fh.write(json.dumps({"ts": t_now - 100.0, "src": "live", "ev": "STATUS"}) + "\n")
        fh.write(json.dumps({"ts": t_now - 3600.0, "src": "live", "ev": "STATUS"}) + "\n")
    b_day, b_ts = bfl._find_first_live_event_boundary(reader, now_ts=t_now)
    assert b_day == day_live
    assert b_ts == pytest.approx(t_now - 3600.0)

    # Audit file with corrupt far-future mtime must NOT copy future mtime onto created order_events_*.jsonl
    day_old = bfl.utc_day_for_ts(t_now - 2 * 86400.0)
    audit_file = audit_dir / f"execution_audit_{day_old}.jsonl"
    audit_file.write_text(
        json.dumps(
            {
                "event": "order_terminal",
                "status": "FILLED",
                "ts": t_now - 2 * 86400.0,
                "venue": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "filled_qty": 0.01,
                "price": 60000.0,
                "order_id": "9001",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(audit_file, (t_now + 86400.0, t_now + 86400.0))
    stats = bfl.backfill_order_ledger_from_audit(str(audit_dir), str(cdb_dir), dry_run=False)
    assert stats["event_days_written"] == 1
    ev_old_path = reader.events_path(day_old)
    assert os.path.getmtime(ev_old_path) <= time.time() + 60.0


def test_pass25_order_ledger_future_verified_ts_recently_closed_and_worker_hb(tmp_path, monkeypatch):
    """[Pass 25] Far-future symbol_verified_ts/recently_closed cannot prematurely close orders, and worker main forces HB transitions."""
    import order_ledger_reconciler_worker as olw
    from providers.order_ledger import update_recently_closed

    t0 = 1_750_000_000.0

    # 1. Pure invariant: update_recently_closed prunes far-future tokens and clamps far-future closed_ts
    rc_out = update_recently_closed(
        {"binance:btcusdc:vid:bad": t0 + 99999.0, "binance:btcusdc:vid:ok": t0 - 10.0},
        [{"v": "binance", "sym": "BTCUSDC", "vid": "new1", "state": "FILLED", "closed_ts": t0 + 50000.0}],
        eff_now=t0,
        prune=True,
    )
    assert "binance:btcusdc:vid:bad" not in rc_out
    assert rc_out["binance:btcusdc:vid:ok"] == pytest.approx(t0 - 10.0)
    assert rc_out["binance:btcusdc:vid:new1"] == pytest.approx(t0)

    # 2. Dependent cross-check: poisoned far-future symbol_verified_ts on disk/venue snapshot must NOT close a fresh SUBMITTED order
    cdb = tmp_path / "cachedb"
    writer = OrderLedgerWriter(cachedb_dir=str(cdb))
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdb), use_singletons=False, reconcile_grace_sec=30.0)
    reader = OrderLedgerReader(cachedb_dir=str(cdb))

    # Seed orders_active.json with a poisoned far-future symbol_verified_ts
    reconciler.write_active_snapshot(
        {
            "schema_version": 1,
            "updated_ts": t0 - 5.0,
            "journal_cursor": {"file": "", "offset": 0, "offsets": {}},
            "venue_verified_ts": {"binance": t0 + 86400.0},
            "symbol_verified_ts": {"binance:btcusdc": t0 + 86400.0},
            "recently_closed": {"binance:btcusdc:vid:ghost": t0 + 86400.0},
            "orders": {},
        }
    )
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="BTCUSDC",
        ts=t0,
        owner="tradeall",
        side="BUY",
        qty=0.01,
        px=60000.0,
        iid="iid-p25-fresh",
        cid="TA_p25_fresh",
    )
    # Also inject a corrupt far-future venue open snapshot
    monkeypatch.setattr(
        reconciler,
        "_read_venue_open_snapshot",
        lambda v: ({}, t0 + 99999.0, {"BTCUSDC": t0 + 99999.0}) if v == "binance" else ({}, 0.0, {}),
    )
    reconciler.reconcile_once(now_ts=t0 + 1.0)

    active = {r["key"]: r for r in reader.list_active_orders(apply_tail=False)}
    assert "binance:iid:iid-p25-fresh" in active
    assert active["binance:iid:iid-p25-fresh"]["state"] == OrderLedgerState.SUBMITTED.value
    snap_after = reader.read_active_snapshot()
    assert "binance:btcusdc" not in snap_after["symbol_verified_ts"]
    assert "binance:btcusdc:vid:ghost" not in snap_after["recently_closed"]

    # 3. Pure invariant: worker main() passes force=True on ok->error->ok transitions and force=False on steady ticks
    monkeypatch.setattr(olw, "resolve_worker_poll_sec", lambda: 5.0)
    monkeypatch.setattr(olw, "single_instance", lambda *_a, **_kw: None)
    monkeypatch.setattr(olw.sys, "argv", ["order_ledger_reconciler_worker.py"])

    outcomes = iter([
        {"events_applied": 0, "active_count": 0, "reconciled_open": 0, "reconciled_filled": 0, "reconciled_closed": 0, "orphans_adopted": 0},
        RuntimeError("disk_io_err"),
        RuntimeError("still_err"),
        {"events_applied": 0, "active_count": 0, "reconciled_open": 0, "reconciled_filled": 0, "reconciled_closed": 0, "orphans_adopted": 0},
    ])

    class _FakeRec:
        reconcile_grace_sec = 30.0
        intent_stale_sec = 100.0

        def reconcile_once(self):
            nxt = next(outcomes)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

    hb_calls: list[tuple[dict, bool]] = []
    sleeps = [0]

    def _fake_sleep(_s):
        sleeps[0] += 1
        if sleeps[0] >= 4:
            raise SystemExit(0)

    monkeypatch.setattr(olw, "OrderLedgerReconciler", lambda: _FakeRec())
    monkeypatch.setattr(
        olw,
        "_touch_ledger_hb",
        lambda st, **kw: hb_calls.append((dict(st), bool(kw.get("force", False)))),
    )
    monkeypatch.setattr(olw.time, "sleep", _fake_sleep)
    with pytest.raises(SystemExit):
        olw.main()
    # startup force=True, then ok(False), error(True), error(False), ok(True)
    assert [force for _st, force in hb_calls] == [True, False, True, False, True]
    assert hb_calls[2][0]["status"] == "error"


def test_pass30_backfill_rejects_future_audit_rows_normalizes_ms_and_reads_venue_exec_qty(tmp_path):
    """[Pass 30] backfill_order_ledger_from_audit rejects corrupt far-future audit timestamps (no 2099 files),
    normalizes epoch-ms audit timestamps, and preserves venue-shaped executedQty/vol_exec partial fills."""
    from verify_tools import backfill_order_ledger_from_audit as bfl

    t_now = 1_796_250_000.0
    audit_dir = tmp_path / "logger" / "execution_audit"
    cdb_dir = tmp_path / "cachedb"
    audit_dir.mkdir(parents=True)
    day_ok = bfl.utc_day_for_ts(t_now - 86400.0)
    audit_file = audit_dir / f"execution_audit_{day_ok}.jsonl"

    # Pure invariant on _convert_audit_row_to_ledger_event: epoch-ms ts normalized + venue executedQty extracted
    conv = bfl._convert_audit_row_to_ledger_event(
        {
            "event": "order_terminal",
            "status": "PARTIALLY_FILLED",
            "ts": (t_now - 86400.0) * 1000.0,
            "venue": "binance",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "qty": 0.05,
            "executedQty": 0.02,
            "price": 60000.0,
            "order_id": "bn-p30-ms",
        }
    )
    assert conv is not None
    assert conv["ts"] == pytest.approx(t_now - 86400.0)
    assert conv["fq"] == pytest.approx(0.02)

    # Dependent cross-check: backfill skips far-future row (no future order_events_*.jsonl created) & writes valid ms row
    rows = [
        {
            "event": "order_terminal",
            "status": "FILLED",
            "ts": t_now + 365.0 * 86400.0,  # corrupt 1 year in the future
            "venue": "binance",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "qty": 0.01,
            "filled_qty": 0.01,
            "price": 99999.0,
            "order_id": "bn-p30-future",
        },
        {
            "event": "order_terminal",
            "status": "FILLED",
            "ts": (t_now - 86400.0) * 1000.0,
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "side": "SELL",
            "qty": 2.0,
            "vol_exec": 2.0,
            "price": 66.0,
            "order_id": "kr-p30-valid",
        },
    ]
    audit_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    stats = bfl.backfill_order_ledger_from_audit(str(audit_dir), str(cdb_dir), dry_run=False, now_ts=t_now)
    assert stats["events_converted"] == 1
    assert stats["event_days_written"] == 1
    reader = OrderLedgerReader(cachedb_dir=str(cdb_dir))
    ev_files = reader.list_event_files()
    assert len(ev_files) == 1
    assert day_ok in os.path.basename(ev_files[0])
    written_rows = [json.loads(ln) for ln in open(ev_files[0], encoding="utf-8") if ln.strip()]
    assert len(written_rows) == 1
    assert written_rows[0]["vid"] == "kr-p30-valid"
    assert written_rows[0]["fq"] == pytest.approx(2.0)








