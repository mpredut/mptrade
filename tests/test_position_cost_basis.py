"""Inventory-reconciled acquisition cost; no exchange requests or real state writes."""
import math
from types import SimpleNamespace
from threading import RLock
from unittest.mock import Mock

import pytest

from providers.quantity import remaining_average_cost


def fill(i, side="BUY", qty=1, price=100, **values):
    return dict(id=i, timestamp=i, side=side, qty=qty, price=price,
                base_fee=0, quote_fee=0, **values)


def test_closed_cycle_does_not_pollute_the_new_position():
    rows = [fill(1, price=10), fill(2, "SELL", price=12), fill(3, price=100)]
    assert remaining_average_cost(rows, 1) == 100


def test_partial_sale_then_new_buy_uses_remaining_average_cost():
    rows = [fill(1, qty=2, price=20), fill(2, "SELL", price=30), fill(3)]
    assert remaining_average_cost(rows, 2) == 60


def test_ambiguous_invalid_or_unreconciled_history_is_unknown():
    cases = [
        ([], 1), ([fill(1)], 2), ([fill(1, "SELL"), fill(2)], 1),
        ([fill(1), fill(1)], 2), ([fill(2), fill(1)], 2),
        ([fill(1), fill(2, "SELL", qty=2)], 1),
        ([fill(1, price=float("inf"))], 1), ([fill(1, qty=float("nan"))], 1),
        ([fill(1, qty=-1)], 1), ([fill(1, qty=True)], 1),
        ([fill(1, "UNKNOWN")], 1), ([fill(1)], float("nan")), ([fill(1)], True),
    ]
    for rows, held in cases:
        assert remaining_average_cost(rows, held) is None


def test_base_and_quote_fees_are_accounted_for_without_guessing():
    row = fill(1)
    row.update(base_fee=0.001, quote_fee=0.1)
    assert math.isclose(remaining_average_cost([row], 0.999), 100.1 / 0.999)
    row.pop("base_fee")
    assert remaining_average_cost([row], 1) is None


def test_negative_fee_is_unknown():
    row = fill(1)
    row["quote_fee"] = -0.1
    assert remaining_average_cost([row], 1) is None


@pytest.fixture
def binance_history(monkeypatch):
    import binance_cache_health as health
    import cacheManager as cm
    from providers.market_api import api

    rows = []
    manager = SimpleNamespace(cache={"ARBUSDC": rows}, lock=RLock(),
                              _is_valid_trade=cm.CacheTradeManager._is_valid_trade)
    current = Mock(return_value=SimpleNamespace(order_cache_version="o", trade_cache_version="t"))
    ensure = Mock()
    get_cache = Mock(return_value=manager)
    monkeypatch.setattr(health, "require_fresh_account_cache", current)
    monkeypatch.setattr(cm, "ensure_account_cache_readers", ensure)
    monkeypatch.setattr(cm, "get_cache_manager", get_cache)
    return api, rows, current, ensure, get_cache


def native(i, side="BUY", qty=1, price=100, **extra):
    import time
    return dict(symbol="ARBUSDC", id=i, orderId=i, time=int(time.time()*1000)-1000+i,
                isBuyer=side == "BUY", qty=str(qty), price=str(price),
                commission="0", commissionAsset="USDC", **extra)


def test_binance_uses_immutable_fills_and_current_durable_versions(binance_history):
    api, rows, current, ensure, get_cache = binance_history
    # The cache can receive REST overlap out of order; venue time/ID orders the fills.
    rows.extend([native(3), native(1, price=10), native(2, "SELL", price=12)])
    assert api.position_cost_basis("ARBUSDC", 1, 3600, provider_name="binance") == 100
    current.assert_called_once()
    ensure.assert_called_once_with(current.return_value)
    get_cache.assert_called_once_with("Trade", start_sync=False)
    assert api.position_cost_basis("ARBUSDC", 2, 3600, provider_name="binance") is None


def test_binance_rejects_stale_history_before_reading(binance_history):
    import binance_cache_health as health
    api, rows, current, ensure, get_cache = binance_history
    rows.append(native(1))
    current.side_effect = health.AccountCacheNotReady("stale")
    with pytest.raises(health.AccountCacheNotReady):
        api.position_cost_basis("ARBUSDC", 1, 3600, provider_name="binance")
    ensure.assert_not_called()
    get_cache.assert_not_called()


def test_binance_base_fee_matches_actual_inventory(binance_history):
    api, rows, *_ = binance_history
    row = native(1)
    row.update(commission="0.001", commissionAsset="ARB")
    rows.append(row)
    assert math.isclose(api.position_cost_basis("ARBUSDC", 0.999, 3600), 100 / 0.999)


def test_provider_without_inventory_cost_support_reports_unknown():
    from providers.market_api import api
    assert api.position_cost_basis("HYPEUSD", 1, 3600, provider_name="kraken") is None


def test_binance_missing_commission_returns_none_instead_of_keyerror(binance_history):
    api, rows, *_ = binance_history
    row = native(1)
    row.pop("commission")
    row.pop("commissionAsset")
    rows.append(row)
    assert api.position_cost_basis("ARBUSDC", 1, 3600, provider_name="binance") is None


def test_extract_order_fee_canonical_across_venues():
    from providers.base import extract_order_fee

    # Binance REST trade shape
    info = extract_order_fee(
        {"commission": "0.02", "commissionAsset": "ARB"},
        base_asset="ARB", quote_asset="USDC", fill_price=1.5,
    )
    assert info is not None
    assert info["fee"] == 0.02
    assert info["fee_asset"] == "ARB"
    assert info["base_fee"] == 0.02
    assert math.isclose(info["fee_in_quote"], 0.03)

    # Binance WS executionReport shape (n / N)
    info = extract_order_fee(
        {"n": "0.05", "N": "USDC"},
        base_asset="ARB", quote_asset="USDC", fill_price=1.5,
    )
    assert info is not None
    assert info["fee"] == 0.05
    assert info["fee_asset"] == "USDC"
    assert info["quote_fee"] == 0.05
    assert math.isclose(info["fee_in_quote"], 0.05)

    # Hyperliquid fill shape (fee / feeToken)
    info = extract_order_fee(
        {"fee": "0.01", "feeToken": "HYPE"},
        base_asset="HYPE", quote_asset="USDC", fill_price=25.0,
    )
    assert info is not None
    assert info["fee"] == 0.01
    assert info["fee_asset"] == "HYPE"
    assert math.isclose(info["fee_in_quote"], 0.25)

    # Kraken fill shape (fee without explicit asset defaults to quote unless require_explicit_asset=True)
    info = extract_order_fee(
        {"fee": "0.12"},
        base_asset="HYPE", quote_asset="USD", fill_price=25.0,
    )
    assert info is not None
    assert info["fee"] == 0.12
    assert info["fee_asset"] == "USD"
    assert math.isclose(info["fee_in_quote"], 0.12)
    assert extract_order_fee(
        {"fee": "0.12"},
        base_asset="HYPE", quote_asset="USD", fill_price=25.0,
        require_explicit_asset=True,
    ) is None


def test_ws_execution_report_preserves_commission_and_zero_price_fallback(monkeypatch):
    import cacheManager as cm

    persisted = []

    class DummyTradeCache:
        _is_valid_trade = staticmethod(cm.CacheTradeManager._is_valid_trade)

        def _persist_items(self, symbol, items):
            persisted.extend(items)

    monkeypatch.setattr(cm, "get_cache_manager", lambda name: DummyTradeCache())
    cm._append_trade_from_execution_report({
        "x": "TRADE",
        "X": "FILLED",
        "s": "ARBUSDC",
        "S": "BUY",
        "i": 1001,
        "t": 5001,
        "L": "0.00000000",
        "p": "1.25000000",
        "l": "0.00000000",
        "q": "10.00000000",
        "n": "0.01000000",
        "N": "ARB",
        "T": 1700000000000,
    })
    assert len(persisted) == 1
    assert persisted[0]["price"] == "1.25000000"
    assert persisted[0]["qty"] == "10.00000000"
    assert persisted[0]["commission"] == "0.01000000"
    assert persisted[0]["commissionAsset"] == "ARB"


def test_ws_and_rest_reconciliation_preserves_rest_cursor_and_merges_missed_fills(tmp_path, monkeypatch):
    import cacheManager as cm
    from unittest.mock import MagicMock

    trade_file = str(tmp_path / "cache_trade.jsonl")
    rest_calls = []

    def fake_paginate(_client, symbol, start_time_ms, limit=1000, *, strict=False):
        rest_calls.append((symbol, start_time_ms))
        if len(rest_calls) == 1:
            return []
        # On the second REST poll, return both the missed fill (t=1500) during WS outage
        # and the fill that already arrived via WS (t=2000).
        return [
            {
                "symbol": "ARBUSDC",
                "id": 501,
                "orderId": 9001,
                "price": "1.10",
                "qty": "5.0",
                "time": 1500,
                "isBuyer": True,
                "commission": "0.005",
                "commissionAsset": "USDC",
            },
            {
                "symbol": "ARBUSDC",
                "id": 502,
                "orderId": 9002,
                "price": "1.20",
                "qty": "10.0",
                "time": 2000,
                "isBuyer": True,
                "commission": "0.010",
                "commissionAsset": "USDC",
            },
        ]

    monkeypatch.setattr("binance_api.bapi_allorders.paginate_my_trades", fake_paginate)
    mgr = cm.CacheTradeManager(9999, ["ARBUSDC"], trade_file, api_client=MagicMock())
    mgr.fetchtime_time_per_symbol["ARBUSDC"] = 1000
    mgr._rest_sync_cursor_per_symbol["ARBUSDC"] = 1000

    # 1. WS disconnects briefly (fill 501 at t=1500 is missed on WS), then WS reconnects
    #    and delivers fill 502 at t=2000 BEFORE REST polling runs:
    cm._mark_ws_available(True)
    cm._mark_ws_unhealthy()
    cm._mark_ws_event_received()
    mgr._persist_items("ARBUSDC", [{
        "symbol": "ARBUSDC",
        "id": "502",
        "orderId": 9002,
        "price": "1.20",
        "qty": "10.0",
        "time": 2000,
        "isBuyer": True,
    }])

    # Even if WS_ONLY_MODE is True and WS is healthy again, _should_poll_for_manager
    # must force a REST reconciliation poll after the unhealthy period:
    monkeypatch.setattr(cm, "WS_ONLY_MODE", True)
    assert cm._should_poll_for_manager("CacheTradeManager") is True

    # 2. Now REST poll runs: it must query from startTime=1000 (NOT 2000!),
    #    backfill missed fill 501, deduplicate fill 502 (and enrich its commission),
    #    and sort the unpersisted fills chronologically [501, 502]:
    mgr.query_remote_and_update_cache()
    assert rest_calls[-1] == ("ARBUSDC", 1000)
    bucket = mgr.cache["ARBUSDC"]
    assert [int(r["id"]) for r in bucket] == [501, 502]
    assert bucket[1].get("commission") == "0.010"
    assert bucket[1].get("commissionAsset") == "USDC"
    assert cm._should_poll_for_manager("CacheTradeManager") is False


def test_extract_remaining_qty_explicit_zero_and_tp_restore_normalization():
    from providers.base import extract_remaining_qty
    import instrument
    import order_retry
    import monitortrades as mt
    from binance_api import bapi_trades

    # 1. Explicit remainingQty=0 must return 0.0 instead of falling back to origQty/quantity
    assert extract_remaining_qty({"remainingQty": 0.0, "origQty": 1.5, "quantity": 1.5}) == 0.0
    assert extract_remaining_qty({"remainingQty": "0.00000000", "origQty": "2.0"}) == 0.0
    assert extract_remaining_qty({"remainingQty": -1.0, "origQty": 2.0, "executedQty": 0.5}) == pytest.approx(1.5)

    # 2. instrument._accepted_order_payload and _order_id_from_payload support oid/order_id and reject bool
    assert instrument._accepted_order_payload({"oid": 778899, "status": "resting"}) is True
    assert instrument._order_id_from_payload({"oid": 778899}) == "778899"
    assert instrument._accepted_order_payload({"order_id": "ord-42"}) is True
    assert instrument._accepted_order_payload({"orderId": True}) is False
    assert instrument._order_id_from_payload({"orderId": False}) is None

    # 3. order_retry.infer_record_owner recognizes accel_plunge_down as tradeall
    assert order_retry.infer_record_owner({"kind": "accel_plunge_down"}) == "tradeall"

    # 4. monitortrades._restore_resting_tp_orders restores orders with origQty/executedQty (no 'qty' key)
    placed_calls = []

    class DummyInst:
        def place(self, side, price, qty, **kw):
            placed_calls.append((side, price, qty, kw))
            return {"orderId": "restored-1"}

    mt._restore_resting_tp_orders(
        DummyInst(),
        "HYPEUSD",
        [{"orderId": "OB1", "side": "SELL", "price": 25.5, "origQty": 4.0, "executedQty": 1.5}],
        sellable_qty=2.5,
        sl_state="refused",
    )
    assert len(placed_calls) == 1
    assert placed_calls[0][0] == "SELL"
    assert placed_calls[0][1] == pytest.approx(25.5)
    assert placed_calls[0][2] == pytest.approx(2.5)

    # 5. bapi_trades.aggregate_trades handles price="0.00000000" (via quoteQty/qty) and alternate fee keys
    agg = bapi_trades.aggregate_trades([
        {
            "id": 11,
            "orderId": 555,
            "symbol": "BTCUSDC",
            "price": "0.00000000",
            "qty": "0.5",
            "quoteQty": "30000.0",
            "fee": "15.0",
            "feeAsset": "USDC",
            "time": 1700000000000,
            "side": "BUY",
        }
    ])
    assert len(agg) == 1
    assert float(agg[0]["price"]) == pytest.approx(60000.0)
    assert float(agg[0]["commission"]) == pytest.approx(15.0)
    assert agg[0]["commissionAsset"] == "USDC"


def test_cross_venue_and_ws_extract_normalization_edges():
    from providers.base import (
        extract_order_id,
        extract_order_price,
        extract_order_qty,
        extract_order_side,
        extract_remaining_qty,
        extract_timestamp_ms,
    )
    from binance_api.order_id_context import resolve_order_owner

    # 1. Binance WS executionReport: remainingQty from q - z, price from L/Z, timestamp from T/E, side from S
    ws_event = {
        "e": "executionReport",
        "E": 1700000005000,
        "T": 1700000004000,
        "i": 987654,
        "S": "BUY",
        "q": "10.00000000",
        "z": "4.00000000",
        "l": "2.00000000",
        "p": "0.00000000",
        "L": "250.50",
        "Z": "1002.00",
    }
    assert extract_remaining_qty(ws_event) == pytest.approx(6.0)
    assert extract_order_price(ws_event) == pytest.approx(250.50)
    assert extract_order_qty(ws_event) == pytest.approx(2.0)
    assert extract_order_side(ws_event) == "BUY"
    assert extract_timestamp_ms(ws_event) == 1700000004000
    assert extract_order_id(ws_event) == "987654"

    # 2. Raw Hyperliquid open_orders payload: origSz is original size, sz is remaining size
    hl_open_order = {"oid": 112233, "side": "A", "limitPx": "35.5", "origSz": "5.0", "sz": "2.0"}
    assert extract_remaining_qty(hl_open_order) == pytest.approx(2.0)
    assert extract_order_side(hl_open_order) == "SELL"

    # 3. Order ID precedence: Kraken trade (ordertxid over trade id) and Hyperliquid fill (oid over trade id)
    assert extract_order_id({"id": "TRADE-1", "ordertxid": "ORDER-99"}) == "ORDER-99"
    assert extract_order_id({"id": "FILL-7", "oid": 554433}) == "554433"

    # 4. Raw Kraken open order (nested descr.type and descr.price) and cost/vol price fallback
    kraken_raw_open = {"vol": "3.0", "vol_exec": "1.0", "descr": {"type": "sell", "price": "42.75"}}
    assert extract_order_side(kraken_raw_open) == "SELL"
    assert extract_order_price(kraken_raw_open) == pytest.approx(42.75)
    assert extract_order_price({"price": "0.00000", "cost": "150.0", "vol": "3.0"}) == pytest.approx(50.0)

    # 5. T212 signed orderedQuantity side inference
    assert extract_order_side({"ticker": "NVDA_US_EQ", "orderedQuantity": -2.5}) == "SELL"
    assert extract_order_side({"ticker": "NVDA_US_EQ", "orderedQuantity": 2.5}) == "BUY"

    # 6. Composite prefix resolution without trailing underscore & AssetGuardian AGS/AGB prefixes
    assert resolve_order_owner("MO_TA") == "tradeall"
    assert resolve_order_owner("MO_MN") == "manual"
    assert resolve_order_owner("MO_RT") == "rtrade"
    assert resolve_order_owner("TA_OR") == "tradeall"
    assert resolve_order_owner("AGS1234abcd") == "assetguardian"
    assert resolve_order_owner("AGB1234abcd") == "assetguardian"




