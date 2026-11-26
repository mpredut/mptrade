"""Regression tests for Bug-Hunting Cycle #5 fixes across providers, order ledger, core loops, and infra."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from providers.execution_audit import find_order_audit_record
from providers.intent_execution import cancel_existing_ai_open_orders
from providers.intent_reconciliation import reconcile_intents_from_execution_audit
from providers.intent_telemetry import summarize_active_ai_trade_plans
from providers.order_ledger import (
    OrderEventType,
    OrderLedgerReader,
    OrderLedgerState,
    OrderLedgerWriter,
    apply_order_event,
    build_order_event,
    ledger_row_to_audit_record,
    update_recently_closed,
)
from providers.order_ohlc_utils import extract_order_fee
from providers.ws_stream import PublicPriceStreamBase


def test_cycle5_late_submit_unknown_does_not_resurrect_closed_order():
    orders = {}
    closed = []
    ev_ack = build_order_event(
        OrderEventType.ACKED,
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_000.0,
        iid="iid-1",
        cid="cid-1",
        vid="9001",
        side="BUY",
        qty=0.1,
    )
    apply_order_event(orders, ev_ack, closed_out=closed)
    ev_fill = build_order_event(
        OrderEventType.RECONCILED_FILLED,
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_001.0,
        iid="iid-1",
        cid="cid-1",
        vid="9001",
        side="BUY",
        qty=0.1,
        fq=0.1,
        avg_px=60000.0,
    )
    apply_order_event(orders, ev_fill, closed_out=closed)
    assert orders == {}
    assert len(closed) == 1
    recently_closed = update_recently_closed({}, closed, eff_now=1_700_000_002.0)

    ev_late_unknown = build_order_event(
        OrderEventType.SUBMIT_UNKNOWN,
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_003.0,
        iid="iid-1",
        cid="cid-1",
        side="BUY",
        qty=0.1,
        reason="timeout",
    )
    res_key = apply_order_event(orders, ev_late_unknown, closed_out=closed, recently_closed=recently_closed)
    assert res_key is None
    assert orders == {}
    assert len(closed) == 1


def test_cycle5_order_terminal_retry_preserves_retry_and_reconciles_replacement_fill(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    writer = OrderLedgerWriter(cachedb_dir=str(cache_dir))

    # 1. Initial attempt accepted, then expired with retryable order_terminal_retry.
    writer.record_audit_event(
        "submit_accepted",
        intent_id="ai-retry-1",
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_000.0,
        fields={"order_id": "1001", "client_order_id": "AI_1", "side": "BUY", "qty": 0.01},
    )
    writer.record_audit_event(
        "order_terminal_retry",
        intent_id="ai-retry-1",
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_060.0,
        fields={
            "order_id": "1001",
            "client_order_id": "AI_1",
            "side": "BUY",
            "qty": 0.01,
            "venue_status": "EXPIRED",
            "status": "expired",
        },
    )

    intents = [
        {
            "intent_id": "ai-retry-1",
            "status": "EXECUTED",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "decision": "BUY",
            "timestamp": 1_700_000_000.0,
            "execution_payload": {"orderId": "1001", "status": "NEW"},
        }
    ]
    changed_1 = reconcile_intents_from_execution_audit(intents, str(tmp_path / "audit"), cachedb_dir=str(cache_dir))
    assert changed_1 is True
    ep1 = intents[0]["execution_payload"]
    assert ep1.get("retry_pending") is True
    assert not ep1.get("reconciled_terminal")

    # 2. Replacement attempt submits and fills under the same intent_id.
    writer.record_audit_event(
        "submit_accepted",
        intent_id="ai-retry-1",
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_070.0,
        fields={"order_id": "1002", "client_order_id": "AI_2", "side": "BUY", "qty": 0.01},
    )
    writer.record_audit_event(
        "order_terminal",
        intent_id="ai-retry-1",
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_080.0,
        fields={
            "order_id": "1002",
            "client_order_id": "AI_2",
            "side": "BUY",
            "qty": 0.01,
            "filled_qty": 0.01,
            "cost": 600.0,
            "venue_status": "FILLED",
            "status": "closed",
        },
    )
    changed_2 = reconcile_intents_from_execution_audit(intents, str(tmp_path / "audit"), cachedb_dir=str(cache_dir))
    assert changed_2 is True
    ep2 = intents[0]["execution_payload"]
    assert ep2["status"] == "FILLED"
    assert ep2["orderId"] == "1002"
    assert float(ep2["executedQty"]) == pytest.approx(0.01)
    assert ep2["reconciled_terminal"] is True
    assert "retry_pending" not in ep2


def test_cycle5_extract_order_fee_preserves_mixed_case_xstock_base_fee():
    fee_info = extract_order_fee(
        {"fee": "0.01", "fee_asset": "SPCXx", "price": "100.0"},
        base_asset="SPCXx",
        quote_asset="USD",
    )
    assert fee_info is not None
    assert fee_info["fee_asset"] == "SPCXx"
    assert fee_info["base_fee"] == pytest.approx(0.01)
    assert fee_info["quote_fee"] == pytest.approx(0.0)
    assert fee_info["fee_in_quote"] == pytest.approx(1.0)


def test_cycle5_only_bot_orders_blocks_targeted_ai_order_cancel():
    canceled = []

    class _FakeInst:
        symbol = "BTCUSDC"
        provider_name = "binance"

        def open_orders(self, **_kw):
            return [
                {"orderId": "AI-101", "clientOrderId": "AI_BTC_1", "side": "BUY", "price": 60000.0},
                {"orderId": "BOT-202", "clientOrderId": "RT_BTC_1", "side": "BUY", "price": 59500.0},
            ]

        def cancel_order(self, oid):
            canceled.append(str(oid))
            return True

    failed_ids = []
    canceled_ids = cancel_existing_ai_open_orders(
        _FakeInst(),
        "BUY",
        include_bot_orders=True,
        only_bot_orders=True,
        target_order_ids=["AI-101", "BOT-202"],
        failed_order_ids=failed_ids,
    )
    assert canceled_ids == ["BOT-202"]
    assert canceled == ["BOT-202"]
    assert failed_ids == ["AI-101"]


def test_cycle5_has_newer_buy_fill_respects_venue_isolation():
    base_ts = 1_700_000_000.0
    intents = [
        {
            "intent_id": "sell-krk-1",
            "status": "EXECUTED",
            "venue": "kraken",
            "symbol": "BTCUSD",
            "decision": "SELL",
            "timestamp": base_ts,
            "target_price": 58000.0,
            "invalidation_price": 64000.0,
            "execution_payload": {"status": "FILLED", "executedQty": "0.05", "price": 61000.0},
        }
    ]
    positions = [{"venue": "kraken", "symbol": "BTCUSD", "value_usd": 500.0}]
    # Newer Binance BUY must NOT drop Kraken's active SELL plan.
    ro_binance_only = {
        "BTCUSDC": {"venue": "binance", "last_buy": {"ts": base_ts + 300.0, "price": 60500.0}}
    }
    plans_keep = summarize_active_ai_trade_plans(
        intents,
        "crypto",
        base_ts + 600.0,
        lambda _i, **_kw: 3050.0,
        tradable_positions=positions,
        recent_orders=ro_binance_only,
    )
    assert len(plans_keep) == 1
    assert plans_keep[0]["venue"] == "kraken"

    # Newer Kraken BUY stored under "kraken:BTCUSD" MUST supersede Kraken's SELL plan.
    ro_kraken_buy = {
        "kraken:BTCUSD": {"venue": "kraken", "last_buy": {"ts": base_ts + 300.0, "price": 59000.0}}
    }
    plans_drop = summarize_active_ai_trade_plans(
        intents,
        "crypto",
        base_ts + 600.0,
        lambda _i, **_kw: 3050.0,
        tradable_positions=positions,
        recent_orders=ro_kraken_buy,
    )
    assert plans_drop == []


def test_cycle5_ws_stream_norm_symbol_preserves_mixed_case_xstocks():
    assert PublicPriceStreamBase._norm_ws_symbol("NVDAxUSD") == "NVDAxUSD"
    assert PublicPriceStreamBase._norm_ws_symbol("SPCXx/USD") == "SPCXxUSD"
    assert PublicPriceStreamBase._norm_ws_symbol("btc/usdc") == "BTCUSDC"
    assert PublicPriceStreamBase._norm_ws_symbol("BTC/") is None


def test_cycle5_order_ledger_load_terminal_filled_orders_preserves_xstock_case(tmp_path):
    cache_dir = tmp_path / "cachedb"
    reader = OrderLedgerReader(cachedb_dir=str(cache_dir))
    cpath = Path(reader.closed_path(1_700_000_000.0))
    cpath.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "v": "kraken",
        "sym": "SPCXxUSD",
        "state": OrderLedgerState.FILLED.value,
        "side": "SELL",
        "vid": "O-SPCX-1",
        "fq": 2.0,
        "avg_px": 14.5,
        "closed_ts": 1_700_000_000.0,
    }
    cpath.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = reader.load_terminal_filled_orders("kraken", max_months=2)
    assert "SPCXxUSD" in loaded
    assert "SPCXXUSD" not in loaded
    assert loaded["SPCXxUSD"][0]["symbol"] == "SPCXxUSD"


def test_cycle5_find_order_audit_record_ignores_guard_blocked_events(tmp_path):
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    writer = OrderLedgerWriter(cachedb_dir=str(cache_dir))
    writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="binance",
        symbol="BTCUSDC",
        ts=1_700_000_000.0,
        cid="TA_gb1",
        side="BUY",
        reason="profit_guard",
    )
    rec = find_order_audit_record(
        client_order_id="TA_gb1",
        symbol="BTCUSDC",
        cachedb_dir=str(cache_dir),
    )
    assert rec is None
