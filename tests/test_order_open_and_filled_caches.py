"""Comprehensive cross-venue tests for CacheOrderFilledManager and CacheOrderOpenManager."""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cacheManager as cm
from binance_api import bapi_ws_bridge, bapi_ws_normalize
from cacheaccount import (
    CacheAccountManagerBase,
    CacheOrderFilledManager,
    CacheOrderOpenManager,
    _HyperliquidAccountTradeBridge,
    get_order_filled_manager,
    get_order_open_manager,
)
from cachepaths import (
    CACHE_BINANCE_ORDER_FILLED_FILENAME,
    CACHE_BINANCE_ORDER_OPEN_FILENAME,
    CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
    CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
    CACHE_KRAKEN_ORDER_FILLED_FILENAME,
    CACHE_KRAKEN_ORDER_OPEN_FILENAME,
    CACHE_T212_ORDER_FILLED_FILENAME,
    CACHE_T212_ORDER_OPEN_FILENAME,
    legacy_filenames_for,
)
from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector


def test_cross_venue_order_filled_and_open_filenames_and_factory_registration():
    expected_filled = {
        "binance": CACHE_BINANCE_ORDER_FILLED_FILENAME,
        "kraken": CACHE_KRAKEN_ORDER_FILLED_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
        "t212": CACHE_T212_ORDER_FILLED_FILENAME,
    }
    expected_open = {
        "binance": CACHE_BINANCE_ORDER_OPEN_FILENAME,
        "kraken": CACHE_KRAKEN_ORDER_OPEN_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
        "t212": CACHE_T212_ORDER_OPEN_FILENAME,
    }
    for venue, fname in expected_filled.items():
        assert CacheOrderFilledManager.disk_filename(venue) == fname
    for venue, fname in expected_open.items():
        assert CacheOrderOpenManager.disk_filename(venue) == fname
    assert issubclass(CacheOrderFilledManager, CacheAccountManagerBase)
    assert issubclass(CacheOrderOpenManager, CacheAccountManagerBase)

    assert "cache_order_binance.json" in legacy_filenames_for(CACHE_BINANCE_ORDER_FILLED_FILENAME)
    assert "cache_order.json" in legacy_filenames_for(CACHE_BINANCE_ORDER_FILLED_FILENAME)

    for reg_key, expected_cls in (
        ("BinanceOrderFilled", CacheOrderFilledManager),
        ("KrakenOrderFilled", CacheOrderFilledManager),
        ("HyperliquidOrderFilled", CacheOrderFilledManager),
        ("T212OrderFilled", CacheOrderFilledManager),
        ("BinanceOrderOpen", CacheOrderOpenManager),
        ("KrakenOrderOpen", CacheOrderOpenManager),
        ("HyperliquidOrderOpen", CacheOrderOpenManager),
        ("T212OrderOpen", CacheOrderOpenManager),
    ):
        assert reg_key in cm.CacheFactory._CONFIG
        assert cm.CacheFactory._CONFIG[reg_key]["class"] is expected_cls

    for bad_venue in ("null", "unknown", True, 123):
        with pytest.raises(ValueError, match="Unsupported filled-order cache venue"):
            get_order_filled_manager(bad_venue)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unsupported open-order cache venue"):
            get_order_open_manager(bad_venue)  # type: ignore[arg-type]


def test_order_filled_manager_merges_vwap_cid_status_type_and_fee_across_venues(tmp_path):
    # 1. Binance multi-fill aggregation preserves clientOrderId, status, type, fee, and feeAsset
    bn_path = str(tmp_path / CACHE_BINANCE_ORDER_FILLED_FILENAME)
    bn_mgr = CacheOrderFilledManager(3600, ["BTCUSDC"], bn_path, read_only=True, venue="binance")
    bn_mgr.save_state = True
    bn_mgr.read_only = False
    bn_mgr._persist_items(
        "BTCUSDC",
        [
            {
                "orderId": 1001,
                "price": 60000.0,
                "quantity": 0.1,
                "timestamp": 1700000001000,
                "side": "BUY",
                "clientOrderId": "AI_intent_1",
                "status": "PARTIALLY_FILLED",
                "type": "LIMIT",
                "fee": 0.0001,
                "feeAsset": "BNB",
                "_fillId": "501",
            },
            {
                "orderId": 1001,
                "price": 60300.0,
                "quantity": 0.2,
                "timestamp": 1700000002000,
                "side": "BUY",
                "status": "FILLED",
                "fee": 0.0002,
                "feeAsset": "BNB",
                "_fillId": "502",
            },
            # Duplicate fill 502 must be ignored idempotently
            {
                "orderId": 1001,
                "price": 60300.0,
                "quantity": 0.2,
                "timestamp": 1700000002000,
                "side": "BUY",
                "status": "FILLED",
                "fee": 0.0002,
                "feeAsset": "BNB",
                "_fillId": "502",
            },
        ],
    )
    rows = bn_mgr.cache["BTCUSDC"]
    assert len(rows) == 1
    row = rows[0]
    assert row["orderId"] == 1001
    assert row["quantity"] == pytest.approx(0.3)
    assert row["price"] == pytest.approx(60200.0)
    assert row["clientOrderId"] == "AI_intent_1"
    assert row["status"] == "FILLED"
    assert row["type"] == "LIMIT"
    assert row["fee"] == pytest.approx(0.0003)
    assert row["feeAsset"] == "BNB"
    assert row["_fillIds"] == ["501", "502"]
    assert bn_mgr.save_state_to_file() is True

    # 2. Kraken alphanumeric orderId + fill IDs aggregate cleanly cross-venue
    kr_path = str(tmp_path / CACHE_KRAKEN_ORDER_FILLED_FILENAME)
    kr_doc = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        kr_path,
        {
            "HYPEUSD": [
                {
                    "id": "TK1-AAA",
                    "orderId": "OZKX3K-SCPNP-TNHPAL",
                    "side": "SELL",
                    "price": 63.0,
                    "qty": 4.0,
                    "time": 1700000010000,
                    "fee": 0.4,
                    "feeAsset": "USD",
                    "cl_ord_id": "AI_krk_1",
                },
                {
                    "id": "TK2-BBB",
                    "orderId": "OZKX3K-SCPNP-TNHPAL",
                    "side": "SELL",
                    "price": 64.0,
                    "qty": 6.0,
                    "time": 1700000020000,
                    "fee": 0.6,
                    "feeAsset": "USD",
                },
            ]
        },
        venue="kraken",
        now_ms=1700000025000,
        is_snapshot=True,
    )
    kr_rows = kr_doc["items"]["HYPEUSD"]
    assert len(kr_rows) == 1
    assert kr_rows[0]["orderId"] == "OZKX3K-SCPNP-TNHPAL"
    assert kr_rows[0]["quantity"] == pytest.approx(10.0)
    assert kr_rows[0]["price"] == pytest.approx(63.6)
    assert kr_rows[0]["fee"] == pytest.approx(1.0)
    assert kr_rows[0]["feeAsset"] == "USD"
    assert kr_rows[0]["clientOrderId"] == "AI_krk_1"
    assert kr_rows[0]["_fillIds"] == ["TK1-AAA", "TK2-BBB"]


def test_order_open_manager_replace_all_ws_transitions_and_cross_venue_reads(tmp_path):
    # 1. Binance WS executionReport updates open orders in real time: NEW -> PARTIALLY_FILLED -> FILLED/CANCELED
    open_path = str(tmp_path / CACHE_BINANCE_ORDER_OPEN_FILENAME)
    open_mgr = CacheOrderOpenManager(15.0, ["BTCUSDC"], open_path, read_only=False, venue="binance")

    ev_new = {
        "e": "executionReport",
        "s": "BTCUSDC",
        "i": 777001,
        "c": "TA_limit_1",
        "S": "BUY",
        "o": "LIMIT",
        "x": "NEW",
        "X": "NEW",
        "p": "61000.0",
        "q": "0.05",
        "z": "0.0",
        "O": 1700000100000,
        "E": 1700000100050,
    }
    parsed_new = bapi_ws_normalize.parse_open_order_execution_report(ev_new)
    assert parsed_new is not None
    sym_n, act_n, item_n = parsed_new
    assert (sym_n, act_n) == ("BTCUSDC", "upsert")
    assert open_mgr.apply_ws_order_update(sym_n, act_n, item_n) is True
    assert len(open_mgr.cache["BTCUSDC"]) == 1
    assert open_mgr.cache["BTCUSDC"][0]["clientOrderId"] == "TA_limit_1"
    assert open_mgr.cache["BTCUSDC"][0]["status"] == "NEW"

    ev_partial = {
        **ev_new,
        "x": "TRADE",
        "X": "PARTIALLY_FILLED",
        "l": "0.02",
        "L": "61000.0",
        "z": "0.02",
        "t": 9901,
        "n": "0.00005",
        "N": "BNB",
    }
    sym_p, act_p, item_p = bapi_ws_normalize.parse_open_order_execution_report(ev_partial)
    assert (sym_p, act_p) == ("BTCUSDC", "upsert")
    assert open_mgr.apply_ws_order_update(sym_p, act_p, item_p) is True
    assert open_mgr.cache["BTCUSDC"][0]["status"] == "PARTIALLY_FILLED"
    assert open_mgr.cache["BTCUSDC"][0]["executedQty"] == pytest.approx(0.02)

    # Filled-order parser also extracts clientOrderId, type, fee, and feeAsset from the WS TRADE report
    sym_f, fill_item = bapi_ws_normalize.parse_order_execution_report(ev_partial)
    assert sym_f == "BTCUSDC"
    assert fill_item["clientOrderId"] == "TA_limit_1"
    assert fill_item["type"] == "LIMIT"
    assert fill_item["fee"] == pytest.approx(0.00005)
    assert fill_item["feeAsset"] == "BNB"

    ev_cancel = {**ev_new, "x": "CANCELED", "X": "CANCELED"}
    sym_c, act_c, item_c = bapi_ws_normalize.parse_open_order_execution_report(ev_cancel)
    assert (sym_c, act_c) == ("BTCUSDC", "remove")
    assert open_mgr.apply_ws_order_update(sym_c, act_c, item_c) is True
    assert open_mgr.cache["BTCUSDC"] == []

    # 2. Empty remote snapshot clears previously open orders (replace-all semantics)
    open_mgr.replace_symbol_open_orders("BTCUSDC", [item_n])
    assert len(open_mgr.cache["BTCUSDC"]) == 1
    with patch.object(open_mgr, "get_remote_items", return_value=[]):
        open_mgr._fetch_and_apply_remote_items("BTCUSDC", 0)
    assert open_mgr.cache["BTCUSDC"] == []
    assert open_mgr.save_state_to_file() is True

    # 3. Cross-venue snapshot replace & passive read_all_venues_open_orders
    kr_open_path = str(tmp_path / CACHE_KRAKEN_ORDER_OPEN_FILENAME)
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        kr_open_path,
        {
            "SPCXxUSD": [
                {
                    "orderId": "OKRK-1",
                    "clientOrderId": "AI_spcx_1",
                    "side": "SELL",
                    "price": 145.0,
                    "origQty": 3.0,
                    "executedQty": 0.0,
                    "status": "OPEN",
                    "timestamp": 1700000200000,
                },
                # Terminal FILLED row must be rejected by open-order normalization
                {
                    "orderId": "OKRK-TERM",
                    "side": "BUY",
                    "price": 140.0,
                    "origQty": 1.0,
                    "executedQty": 1.0,
                    "status": "FILLED",
                },
            ]
        },
        now_ms=1700000205000,
        verified_ts=1700000205.0,
    )
    flat, covered, verified_map = CacheOrderOpenManager.read_all_venues_open_orders(str(tmp_path))
    assert "binance" in covered and "kraken" in covered
    assert verified_map["kraken"] == pytest.approx(1700000205.0)
    assert len(flat) == 1
    assert flat[0]["venue"] == "kraken"
    assert flat[0]["symbol"] == "SPCXxUSD"
    assert flat[0]["orderId"] == "OKRK-1"
    assert flat[0]["clientOrderId"] == "AI_spcx_1"


def test_hyperliquid_ws_bridge_mirrors_fills_to_order_filled_cache(tmp_path):
    hl_trade_file = str(tmp_path / "cache_trade_hyperliquid.json")
    ws_stream = SimpleNamespace(
        _lock=MagicMock(__enter__=lambda self: None, __exit__=lambda self, *a: None),
        _symbols=["HYPEUSDC"],
        _candidate_mids_keys=lambda s: [s, "@107"],
    )
    bridge = _HyperliquidAccountTradeBridge(
        ws_stream,
        user_address="0x1111111111111111111111111111111111111111",
        fills_cache_path=hl_trade_file,
        resolve_env_user_address=False,
    )
    fake_hl_fills = (
        True,
        {
            "HYPEUSDC": [
                {"oid": 88801, "tid": 1, "side": "B", "px": "60.0", "sz": "2.0", "time": 1700000500000, "fee": "0.06", "cloid": "0xabc1"},
                {"oid": 88801, "tid": 2, "side": "B", "px": "61.0", "sz": "3.0", "time": 1700000501000, "fee": "0.09", "cloid": "0xabc1"},
            ]
        },
    )
    with patch.object(bridge.hl_prov_cls, "extract_ws_user_fills", return_value=fake_hl_fills):
        bridge.handle_user_fills({"isSnapshot": True, "fills": []}, ts_ms=1700000505000)

    assert os.path.isfile(hl_trade_file)
    hl_filled_file = str(tmp_path / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME)
    assert os.path.isfile(hl_filled_file)
    with open(hl_filled_file, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = doc["items"]["HYPEUSDC"]
    assert len(rows) == 1
    assert rows[0]["orderId"] == "88801"
    assert rows[0]["quantity"] == pytest.approx(5.0)
    assert rows[0]["price"] == pytest.approx(60.6)
    assert rows[0]["fee"] == pytest.approx(0.15)
    assert rows[0]["clientOrderId"] == "0xabc1"

    # Regression: a subsequent reconnect WS snapshot carrying only newer fills must NOT wipe
    # older fills from cache_trade_hyperliquid.json (keeping 1:1 parity with cache_order_filled_hyperliquid.json).
    reconnect_hl_fills = (
        True,
        {
            "HYPEUSDC": [
                {"oid": 88802, "tid": 3, "side": "A", "px": "65.0", "sz": "5.0", "time": 1700001500000, "fee": "0.10", "cloid": "0xabc2"},
            ]
        },
    )
    with patch.object(bridge.hl_prov_cls, "extract_ws_user_fills", return_value=reconnect_hl_fills):
        bridge.handle_user_fills({"isSnapshot": True, "fills": []}, ts_ms=1700001505000)

    with open(hl_trade_file, encoding="utf-8") as fh:
        tr_doc = json.load(fh)
    with open(hl_filled_file, encoding="utf-8") as fh:
        fi_doc = json.load(fh)
    assert [r["tid"] for r in tr_doc["items"]["HYPEUSDC"]] == [1, 2, 3]
    assert [r["orderId"] for r in fi_doc["items"]["HYPEUSDC"]] == ["88801", "88802"]

    # Bidirectional verifier confirms 0 diffs when in sync, and catches missing trade rows + realized_net drift!
    from verify_tools import sync_order_filled_from_trade_cache as sync_tool

    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(exist_ok=True)
    os.replace(hl_trade_file, str(orders_dir / os.path.basename(hl_trade_file)))
    os.replace(hl_filled_file, str(orders_dir / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME))
    res_ok = sync_tool._sync_json_venue(
        str(tmp_path), "hyperliquid", os.path.basename(hl_trade_file), CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME, apply_changes=False
    )
    assert res_ok["diffs"] == []

    # Simulate pre-fix trade cache truncation (missing order 88801 -> realized_net & spent drift)
    trunc_tr = dict(tr_doc)
    trunc_tr["items"] = {"HYPEUSDC": [r for r in tr_doc["items"]["HYPEUSDC"] if r["oid"] == 88802]}
    with open(orders_dir / os.path.basename(hl_trade_file), "w", encoding="utf-8") as fh:
        json.dump(trunc_tr, fh)
    res_drift = sync_tool._sync_json_venue(
        str(tmp_path), "hyperliquid", os.path.basename(hl_trade_file), CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME, apply_changes=False
    )
    assert any("only_in_filled=1" in d for d in res_drift["diffs"])
    assert any("pos_realized_net_diff" in d for d in res_drift["diffs"])
    assert any("pos_spent_diff" in d for d in res_drift["diffs"])


def test_kraken_poll_loop_preserves_older_fills_outside_lookback_window(tmp_path):
    """Windowed lookback REST Ledgers polling must union-merge rather than wiping older fills."""
    import threading
    import cacheaccount
    from providers.base import MarketDataProvider
    import providers.kraken_provider  # noqa: F401

    kr_trade = tmp_path / "cache_trade_kraken.json"
    old_fill = {"id": "old_1", "orderId": "oid_old", "side": "BUY", "time": 1_700_000_000_000, "price": 50.0, "qty": 2.0}
    new_fill = {"id": "new_2", "orderId": "oid_new", "side": "SELL", "time": 1_700_900_000_000, "price": 55.0, "qty": 2.0}
    cacheaccount.CacheTradeManager.merge_venue_fills_snapshot(
        str(kr_trade), {"HYPEUSD": [old_fill]}, now_ms=1_700_000_005_000, is_snapshot=True
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(tmp_path / "cache_order_filled_kraken.json"),
        {"HYPEUSD": [old_fill]},
        venue="kraken",
        now_ms=1_700_000_005_000,
        is_snapshot=True,
    )

    stop_ev = threading.Event()
    kr_cls = MarketDataProvider.resolve_venue_provider_class("kraken")

    def _windowed_ledger(_client, *, pairs, lookback_sec):
        stop_ev.set()
        return {"HYPEUSD": [new_fill]}

    with patch.object(kr_cls, "fetch_ledger_fills_by_symbol", side_effect=_windowed_ledger):
        cacheaccount.run_kraken_trade_poll_loop(
            SimpleNamespace(),
            cache_file=str(kr_trade),
            pairs=["HYPEUSD"],
            poll_interval=0.01,
            backoff_init=0.01,
            backoff_max=0.05,
            lookback_sec=3600.0,
            stop_event=stop_ev,
        )

    tr_after = json.loads(kr_trade.read_text(encoding="utf-8"))
    fi_after = json.loads((tmp_path / "cache_order_filled_kraken.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in tr_after["items"]["HYPEUSD"]] == ["old_1", "new_2"]
    assert [r["orderId"] for r in fi_after["items"]["HYPEUSD"]] == ["oid_old", "oid_new"]


def test_sidecarless_account_caches_resync_from_disk_and_never_corrupt_balance_cache(tmp_path):
    """Regression: CacheBalanceManager, CacheOrderOpenManager, and non-Binance CacheOrderFilledManager
    persist JSON without a .meta sidecar; resync_mem_file() must detect newer disk snapshots via
    CacheAccountManagerBase._persisted_max_ts() and reload cleanly (no NO-OP, no KeyError: 0)."""
    import cacheaccount

    t1_s = 1_795_000_000.0
    t2_s = t1_s + 30.0
    t1_ms = int(t1_s * 1000)
    t2_ms = int(t2_s * 1000)

    # 1. CacheBalanceManager: resyncs updated balances on get_balance()/get_all_balances() without KeyError
    bal_p = tmp_path / "cache_balance_kraken.json"
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_p),
        {"USD": (1000.0, 0.0), "SPCXx": (5.0, 0.0)},
        venue="kraken",
        now_s=t1_s,
        verified_ts=t1_s,
        breakdown={"USD": {"USD": 1000.0}},
    )
    bal_mgr = cacheaccount.CacheBalanceManager(60, None, str(bal_p), read_only=True, venue="kraken")
    assert bal_mgr._persisted_max_ts() == t1_ms
    assert bal_mgr._mem_max_ts() == t1_ms
    assert bal_mgr.get_balance("USD")["free"] == pytest.approx(1000.0)

    # Update disk after reader is instantiated -> get_balance()/get_breakdown() must auto-reload cleanly
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_p),
        {"USD": (1250.5, 50.0), "SPCXx": (12.0, 0.0)},
        venue="kraken",
        now_s=t2_s,
        verified_ts=t2_s,
        breakdown={"USD": {"USD": 1250.5}},
    )
    assert bal_mgr._persisted_max_ts() == t2_ms
    assert bal_mgr.get_balance("USD")["free"] == pytest.approx(1250.5)
    assert bal_mgr.get_balance("SPCXx")["free"] == pytest.approx(12.0)
    assert bal_mgr.get_breakdown("USD") == {"USD": 1250.5}
    assert bal_mgr.get_all_balances()["USD"]["locked"] == pytest.approx(50.0)

    # 2. CacheOrderOpenManager: resync_mem_file() reloads new open orders across venues without .meta
    open_p = tmp_path / "cache_order_open_hyperliquid.json"
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(open_p),
        {"HYPEUSDC": [{"orderId": "101", "side": "BUY", "price": 50.0, "origQty": 2.0, "status": "OPEN", "timestamp": t1_ms}]},
        now_ms=t1_ms,
        verified_ts=t1_s,
    )
    open_mgr = cacheaccount.CacheOrderOpenManager(30, None, str(open_p), read_only=True, venue="hyperliquid")
    assert open_mgr._persisted_max_ts() == t1_ms
    assert [r["orderId"] for r in open_mgr.cache["HYPEUSDC"]] == ["101"]

    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(open_p),
        {"HYPEUSDC": [{"orderId": "202", "side": "SELL", "price": 60.0, "origQty": 3.0, "status": "OPEN", "timestamp": t2_ms}]},
        now_ms=t2_ms,
        verified_ts=t2_s,
    )
    open_mgr.resync_mem_file()
    assert [r["orderId"] for r in open_mgr.cache["HYPEUSDC"]] == ["202"]

    # 3. Non-Binance CacheOrderFilledManager: resync_mem_file() reloads new fills without .meta
    filled_p = tmp_path / "cache_order_filled_kraken.json"
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(filled_p),
        {"HYPEUSD": [{"id": "f1", "orderId": "o1", "side": "BUY", "price": 50.0, "qty": 1.0, "time": t1_ms}]},
        venue="kraken",
        now_ms=t1_ms,
        is_snapshot=False,
    )
    filled_mgr = cacheaccount.CacheOrderFilledManager(60, None, str(filled_p), read_only=True, venue="kraken")
    assert filled_mgr._persisted_max_ts() == t1_ms
    assert [r["orderId"] for r in filled_mgr.cache["HYPEUSD"]] == ["o1"]

    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(filled_p),
        {"HYPEUSD": [{"id": "f2", "orderId": "o2", "side": "SELL", "price": 55.0, "qty": 1.0, "time": t2_ms}]},
        venue="kraken",
        now_ms=t2_ms,
        is_snapshot=False,
    )
    filled_mgr.resync_mem_file()
    assert [r["orderId"] for r in filled_mgr.cache["HYPEUSD"]] == ["o1", "o2"]


def test_portfolio_collector_strips_open_orders_from_venue_balances_and_persists_to_order_open_cache(tmp_path):
    now_s = time.time()
    state_dir = tmp_path / "state"
    orders_dir = tmp_path / "orders"
    state_dir.mkdir()
    orders_dir.mkdir()
    # Seed legacy venue_balances_state.json containing stale open_orders:* keys
    vb_path = state_dir / "venue_balances_state.json"
    vb_path.write_text(
        json.dumps({
            "binance:BTC": 0.1,
            "open_orders:binance:BTCUSDC": [{"orderId": "legacy-1"}],
            "open_orders_verified_ts:binance": now_s - 60.0,
        }),
        encoding="utf-8",
    )

    fake_provider = SimpleNamespace(
        free_balance=lambda asset: 0.2 if asset == "BTC" else 500.0,
        get_current_price=lambda sym: 65000.0,
        usd_cash_breakdown=lambda: None,
    )
    fake_inst = SimpleNamespace(
        symbol="BTCUSDC",
        provider_name="binance",
        base="BTC",
        quote="USDC",
        params={},
        _provider=fake_provider,
        open_orders=lambda: [
            {
                "orderId": "900123",
                "clientOrderId": "AI_open_test",
                "side": "SELL",
                "price": 68000.0,
                "origQty": 0.05,
                "executedQty": 0.0,
                "status": "NEW",
                "time": int(now_s * 1000),
            }
        ],
    )
    collector = TradablePositionsAndOrdersCollector()
    verified: set[str] = set()
    _, open_orders, _, _ = collector.collect(
        str(tmp_path),
        inst_map={"BIN_BTC": fake_inst},
        prices_and_trends={"BTCUSDC": {"price": 65000.0, "trend": 0, "growth_coefficient": 0.0}},
        verified_open_venues=verified,
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_s,
    )
    assert "binance" in verified
    assert len(open_orders) == 1
    assert open_orders[0]["order_id"] == "900123"

    # Split persisted files & unified reader must NOT contain any open_orders:* or open_orders_verified_ts:* keys
    from providers.portfolio_state_store import load_venue_balances_state_dict

    bal_cache_path = orders_dir / "cache_balance_binance.json"
    pos_state_path = state_dir / "portfolio_positions_state.json"
    assert bal_cache_path.is_file() and pos_state_path.is_file()
    saved_pos = json.loads(pos_state_path.read_text(encoding="utf-8"))
    assert all(k.startswith("pos:") for k in saved_pos)
    saved_vb = load_venue_balances_state_dict(str(tmp_path))
    assert not any(k.startswith(("open_orders:", "open_orders_verified_ts:")) for k in saved_vb)

    # cache_order_open_binance.json must contain the live open order
    open_cache_path = orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME
    assert open_cache_path.is_file()
    saved_open = json.loads(open_cache_path.read_text(encoding="utf-8"))
    assert len(saved_open["items"]["BTCUSDC"]) == 1
    assert saved_open["items"]["BTCUSDC"][0]["orderId"] == "900123"
    assert saved_open["verified_ts"] == pytest.approx(now_s, abs=1.0)


def test_order_filled_manager_symmetric_passive_readers_and_fallbacks(tmp_path):
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir()

    # 1. Binance legacy cache_order_binance.json fallback + legacy dict-bucket normalization
    (orders_dir / "cache_order_binance.json").write_text(
        json.dumps({
            "items": {
                "BTCUSDC": {
                    "101": {
                        "orderId": 101,
                        "price": 62000.0,
                        "quantity": 0.05,
                        "timestamp": 1700001000000,
                        "side": "BUY",
                        "status": "FILLED",
                    }
                }
            },
            "fetchtime": {"BTCUSDC": 1700001005000},
        }),
        encoding="utf-8",
    )
    # 2. Kraken canonical cache_order_filled_kraken.json
    (orders_dir / CACHE_KRAKEN_ORDER_FILLED_FILENAME).write_text(
        json.dumps({
            "items": {
                "ETHUSD": [
                    {
                        "orderId": "OKRK-FILLED-1",
                        "price": 3200.0,
                        "quantity": 0.5,
                        "timestamp": 1700002000000,
                        "side": "SELL",
                        "status": "FILLED",
                    }
                ]
            },
            "fetchtime": {"ETHUSD": 1700002005000},
        }),
        encoding="utf-8",
    )
    # 3. Hyperliquid reference cache_trade_hyperliquid.json fallback (opt-in via allow_trade_fallback=True)
    (orders_dir / "cache_trade_hyperliquid.json").write_text(
        json.dumps({
            "items": {
                "HYPEUSDC": [
                    {
                        "orderId": "HL-99",
                        "price": 55.0,
                        "quantity": 2.0,
                        "timestamp": 1700003000000,
                        "side": "BUY",
                    }
                ]
            },
            "fetchtime": {"HYPEUSDC": 1700003005000},
        }),
        encoding="utf-8",
    )

    # Single-venue passive read with legacy fallback
    bn_items, bn_ft = CacheOrderFilledManager.read_venue_filled_orders_snapshot(
        str(orders_dir / CACHE_BINANCE_ORDER_FILLED_FILENAME),
        venue="binance",
    )
    assert len(bn_items["BTCUSDC"]) == 1
    assert bn_items["BTCUSDC"][0]["orderId"] == 101
    assert bn_ft["BTCUSDC"] == 1700001005000
    # Passive read must NOT create or migrate the canonical file on disk
    assert not (orders_dir / CACHE_BINANCE_ORDER_FILLED_FILENAME).exists()

    # Cross-venue passive read without trade fallback covers binance (legacy) + kraken (canonical)
    flat_no_trd, covered_no_trd, ft_by_venue = CacheOrderFilledManager.read_all_venues_filled_orders(str(tmp_path))
    assert covered_no_trd == {"binance", "kraken"}
    assert len(flat_no_trd) == 2
    assert {r["venue"] for r in flat_no_trd} == {"binance", "kraken"}
    assert ft_by_venue["kraken"]["ETHUSD"] == 1700002005000

    # Cross-venue passive read with trade fallback also includes hyperliquid reference trade cache
    flat_with_trd, covered_with_trd, _ = CacheOrderFilledManager.read_all_venues_filled_orders(
        str(tmp_path),
        allow_trade_fallback=True,
    )
    assert covered_with_trd == {"binance", "kraken", "hyperliquid"}
    assert len(flat_with_trd) == 3
    assert any(r["venue"] == "hyperliquid" and r["orderId"] == "HL-99" for r in flat_with_trd)


def test_order_filled_coherence_kraken_subsumption_t212_fee_upgrade_and_binance_last_opposite(tmp_path):
    from providers.portfolio_state_store import resolve_kraken_trade_cache_position
    from providers.quantity import summarize_fills_position_and_cost

    # 1. Kraken Bug K1: provisional Ledger fills (orderId == fillId) must be subsumed by the real order
    #    both when real order arrives later and when Ledger fill arrives after the real order.
    kr_path = str(tmp_path / CACHE_KRAKEN_ORDER_FILLED_FILENAME)
    ledger_fills = [
        {"id": "TAIDKY-1", "orderId": "TAIDKY-1", "side": "SELL", "price": 135.42, "qty": 4.91815, "time": 1795467222749, "fee": 0.0},
        {"id": "TJJTVY-2", "orderId": "TJJTVY-2", "side": "SELL", "price": 135.42, "qty": 4.91815, "time": 1795467222750, "fee": 0.0},
        {"id": "TAVULX-3", "orderId": "TAVULX-3", "side": "SELL", "price": 135.42, "qty": 2.500535, "time": 1795467222751, "fee": 0.0},
    ]
    own_trades_fills = [
        {"id": "BUY-0", "orderId": "OBUY-000", "side": "BUY", "price": 134.85, "qty": 24.472723, "time": 1795400000000, "fee": 0.0, "feeAsset": "USD"},
        {"id": "TAIDKY-1", "orderId": "OFMFTT-REAL", "side": "SELL", "price": 135.42, "qty": 4.91815, "time": 1795467222749, "fee": 0.0, "feeAsset": "USD"},
        {"id": "TJJTVY-2", "orderId": "OFMFTT-REAL", "side": "SELL", "price": 135.42, "qty": 4.91815, "time": 1795467222750, "fee": 0.0, "feeAsset": "USD"},
        {"id": "TAVULX-3", "orderId": "OFMFTT-REAL", "side": "SELL", "price": 135.42, "qty": 2.500535, "time": 1795467222751, "fee": 0.0, "feeAsset": "USD"},
    ]
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        kr_path, {"SPCXxUSD": ledger_fills}, venue="kraken", now_ms=1795467230000, is_snapshot=False
    )
    doc = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        kr_path, {"SPCXxUSD": own_trades_fills}, venue="kraken", now_ms=1795467240000, is_snapshot=False
    )
    assert [r["orderId"] for r in doc["items"]["SPCXxUSD"]] == ["OBUY-000", "OFMFTT-REAL"]
    # Re-sending provisional Ledger fills after real order must NOT re-add duplicates
    doc2 = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        kr_path, {"SPCXxUSD": ledger_fills}, venue="kraken", now_ms=1795467250000, is_snapshot=False
    )
    assert [r["orderId"] for r in doc2["items"]["SPCXxUSD"]] == ["OBUY-000", "OFMFTT-REAL"]
    pos_filled = resolve_kraken_trade_cache_position("SPCXxUSD", str(tmp_path))
    pos_ref = summarize_fills_position_and_cost(own_trades_fills, base_asset="SPCXx", quote_asset="USD")
    assert pos_filled is not None and pos_ref is not None
    assert pos_filled["qty"] == pytest.approx(pos_ref["qty"], abs=1e-6)
    assert pos_filled["cost"] == pytest.approx(pos_ref["cost"], abs=1e-4)

    # 2. T212 Bug T1: settled fee 0.0 -> 1.24 upgrades on equal-qty/ts merge and never downgrades back to 0.0.
    t212_path = str(tmp_path / CACHE_T212_ORDER_FILLED_FILENAME)
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        t212_path,
        {"NVDA_US_EQ": [{"orderId": "60903496656", "side": "BUY", "price": 182.5, "quantity": 5.0, "timestamp": 1795530903000, "fee": 0.0}]},
        venue="t212",
        now_ms=1795530910000,
    )
    doc_t1 = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        t212_path,
        {"NVDA_US_EQ": [{"orderId": "60903496656", "side": "BUY", "price": 182.5, "quantity": 5.0, "timestamp": 1795530903000, "fee": 1.24, "feeAsset": "USD"}]},
        venue="t212",
        now_ms=1795530920000,
    )
    assert doc_t1["items"]["NVDA_US_EQ"][0]["fee"] == pytest.approx(1.24)
    assert doc_t1["items"]["NVDA_US_EQ"][0]["feeAsset"] == "USD"
    doc_t2 = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        t212_path,
        {"NVDA_US_EQ": [{"orderId": "60903496656", "side": "BUY", "price": 182.5, "quantity": 5.0, "timestamp": 1795530903000, "fee": 0.0}]},
        venue="t212",
        now_ms=1795530930000,
        is_snapshot=True,
    )
    assert doc_t2["items"]["NVDA_US_EQ"][0]["fee"] == pytest.approx(1.24)

    # 3. Binance Bugs B1 & B2: _persist_items backfills fee/feeAsset/type/clientOrderId on known _fillId
    #    and get_last_opposite_fill_price returns latest opposite fill price.
    bn_path = str(tmp_path / CACHE_BINANCE_ORDER_FILLED_FILENAME)
    bn_mgr = CacheOrderFilledManager(3600, ["BTCUSDC"], bn_path, read_only=True, venue="binance")
    bn_mgr.read_only = False
    bn_mgr._persist_items(
        "BTCUSDC",
        [
            {"orderId": 100, "price": 60000.0, "quantity": 0.1, "timestamp": 1700000001000, "side": "BUY", "_fillId": "1"},
            {"orderId": 101, "price": 61500.0, "quantity": 0.1, "timestamp": 1700000005000, "side": "SELL", "_fillId": "2"},
        ],
    )
    bn_mgr._persist_items(
        "BTCUSDC",
        [
            {
                "orderId": 100,
                "price": 60000.0,
                "quantity": 0.1,
                "timestamp": 1700000001000,
                "side": "BUY",
                "clientOrderId": "TA_backfill_1",
                "type": "LIMIT",
                "status": "FILLED",
                "fee": 0.0004,
                "feeAsset": "BNB",
                "_fillId": "1",
            }
        ],
    )
    row_100 = next(r for r in bn_mgr.cache["BTCUSDC"] if r["orderId"] == 100)
    assert row_100["fee"] == pytest.approx(0.0004)
    assert row_100["feeAsset"] == "BNB"
    assert row_100["clientOrderId"] == "TA_backfill_1"
    assert row_100["type"] == "LIMIT"
    assert bn_mgr.get_last_opposite_fill_price("BTCUSDC", "BUY") == pytest.approx(61500.0)
    assert bn_mgr.get_last_opposite_fill_price("BTCUSDC", "SELL") == pytest.approx(60000.0)


def test_final_audit_partial_sell_fee_proration_previous_recovery_and_non_destructive_ssot_write(tmp_path):
    """Regression tests for:
    1. summarize_fills_position_and_cost prorates quote_fee on partial-history sells so split vs aggregated fills match.
    2. resolve_kraken_trade_cache_position & CacheAccountManagerBase._persisted_max_ts recover from .previous swaps.
    3. ProviderSsotBridgeMixin.write_ssot_fills_cache union-merges without wiping older fills.
    """
    import json
    from cacheaccount import CacheTradeManager
    from cachepaths import CACHE_KRAKEN_ORDER_FILLED_FILENAME, CACHE_KRAKEN_TRADE_FILENAME
    from providers.portfolio_state_store import resolve_kraken_trade_cache_position
    from providers.quantity import summarize_fills_position_and_cost
    from providers.ssot_bridge import ProviderSsotBridgeMixin
    from state_io import previous_generation_path

    # 1. Partial-history sell: tracked BUY 5.475079 @ 79.306557, SELL 17.88 @ 93.3 (2 partial fills vs 1 order row).
    buy_row = {"id": "B1", "orderId": "OB1", "side": "BUY", "price": 79.306557, "quantity": 5.475079, "timestamp": 1792000000000, "fee": 0.0}
    sell_split = [
        {"id": "S1", "orderId": "OTWER3", "side": "SELL", "price": 93.3, "quantity": 6.1252, "timestamp": 1793287711316, "fee": 1.25726},
        {"id": "S2", "orderId": "OTWER3", "side": "SELL", "price": 93.3, "quantity": 11.7548, "timestamp": 1793287711318, "fee": 2.41279},
    ]
    sell_agg = [
        {"orderId": "OTWER3", "side": "SELL", "price": 93.3, "quantity": 17.88, "timestamp": 1793287711318, "fee": 3.67005},
    ]
    pos_split = summarize_fills_position_and_cost([buy_row, *sell_split], base_asset="HYPE", quote_asset="USD")
    pos_agg = summarize_fills_position_and_cost([buy_row, *sell_agg], base_asset="HYPE", quote_asset="USD")
    assert pos_split is not None and pos_agg is not None
    assert pos_split["qty"] == pytest.approx(0.0, abs=1e-9)
    assert pos_agg["qty"] == pytest.approx(0.0, abs=1e-9)
    assert pos_split["realized_net"] == pytest.approx(pos_agg["realized_net"], abs=1e-4)
    # Normal fully-matched sell (qty >= removed) still deducts 100% of quote_fee.
    full_buy = {"id": "FB1", "orderId": "OFB1", "side": "BUY", "price": 100.0, "quantity": 10.0, "timestamp": 1792000000000, "fee": 1.0}
    full_sell = {"id": "FS1", "orderId": "OFS1", "side": "SELL", "price": 110.0, "quantity": 10.0, "timestamp": 1792000005000, "fee": 2.0}
    pos_full = summarize_fills_position_and_cost([full_buy, full_sell], base_asset="HYPE", quote_asset="USD")
    assert pos_full is not None
    assert pos_full["realized_net"] == pytest.approx(10.0 * (110.0 - 100.1) - 2.0, abs=1e-6)

    # 2. .previous-only atomic swap recovery for resolve_kraken_trade_cache_position & _persisted_max_ts.
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    kr_filled_primary = orders_dir / CACHE_KRAKEN_ORDER_FILLED_FILENAME
    kr_filled_prev = previous_generation_path(str(kr_filled_primary))
    with open(kr_filled_prev, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "items": {
                    "HYPEUSD": [
                        {"orderId": "O1", "side": "BUY", "price": 80.0, "quantity": 2.0, "timestamp": 1795000001000, "fee": 0.32}
                    ]
                },
                "fetchtime": {"HYPEUSD": 1795000005000},
            },
            fh,
        )
    pos_prev = resolve_kraken_trade_cache_position("HYPEUSD", str(tmp_path))
    assert pos_prev is not None
    assert pos_prev["qty"] == pytest.approx(2.0, abs=1e-6)

    kr_trade_primary = orders_dir / CACHE_KRAKEN_TRADE_FILENAME
    kr_trade_prev = previous_generation_path(str(kr_trade_primary))
    with open(kr_trade_prev, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "items": {"HYPEUSD": [buy_row]},
                "fetchtime": {"HYPEUSD": 1795000099000},
            },
            fh,
        )
    tm = CacheTradeManager(3600, ["HYPEUSD"], str(kr_trade_primary), read_only=True, venue="kraken")
    assert tm._persisted_max_ts() == 1795000099000

    # 3. write_ssot_fills_cache union-merges without wiping historical fills in trade & mirrored filled caches.
    class _BridgeProbe(ProviderSsotBridgeMixin):
        name = "hyperliquid"

    probe = _BridgeProbe()
    hl_dir = tmp_path / "hl_bridge"
    hl_dir.mkdir(parents=True, exist_ok=True)
    hl_trade = hl_dir / "cache_trade_hyperliquid.json"
    hl_filled = hl_dir / "cache_order_filled_hyperliquid.json"
    older_fill = {"coin": "@107", "oid": 1001, "tid": 11, "side": "B", "px": "80.0", "sz": "1.0", "time": 1793700000000}
    newer_fill = {"coin": "@107", "oid": 1002, "tid": 12, "side": "B", "px": "82.0", "sz": "2.0", "time": 1795600000000}
    assert probe.write_ssot_fills_cache("HYPEUSDC", str(hl_trade), [older_fill]) is True
    assert probe.write_ssot_fills_cache("HYPEUSDC", str(hl_trade), [newer_fill]) is True
    assert probe.write_ssot_fills_cache("HYPEUSDC", str(hl_trade), []) is True

    tr_doc = json.loads(hl_trade.read_text(encoding="utf-8"))
    fi_doc = json.loads(hl_filled.read_text(encoding="utf-8"))
    assert {str(r["oid"]) for r in tr_doc["items"]["HYPEUSDC"]} == {"1001", "1002"}
    assert {str(r["orderId"]) for r in fi_doc["items"]["HYPEUSDC"]} == {"1001", "1002"}
    assert probe.write_ssot_fills_cache("SOLUSDC", str(hl_trade), []) is True
    tr_doc2 = json.loads(hl_trade.read_text(encoding="utf-8"))
    assert tr_doc2["items"]["SOLUSDC"] == []
    assert tr_doc2["fetchtime"]["SOLUSDC"] > 0


def test_ws_outbound_account_position_updates_balance_cache_and_stale_lock_guard(tmp_path):
    """Regression for Step 73 phantom -$966 NAV dip after ADAUSDC fill:
    1. Binance outboundAccountPosition WS event atomically updates CacheBalanceManager (USDC + ADA) without clobbering untouched assets or allowing stale out-of-order events.
    2. Read-only CacheBalanceManager rejects WS writes outside the sole cacheManager producer.
    3. _venue_balance_cache_has_stale_order_lock detects when newer open_orders shows fewer locked funds/base than cache_balance_<venue>.
    """
    import cacheaccount
    from binance_api import bapi_ws_bridge
    from intelligence.sentiment.collectors.portfolio_collector import _venue_balance_cache_has_stale_order_lock
    from providers.portfolio_valuation import BalanceObservation

    t0 = 1_795_716_850.0
    t_fill = t0 + 48.0
    bal_path = tmp_path / "cache_balance_binance.json"

    # Seed pre-fill full baseline snapshot (BTC $899.97 + ADA $966.03 locked; ADA only partially filled 226.49)
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path),
        {
            "USDC": (7361.97, 1866.00),
            "ADA": (226.4909, 0.0),
            "ARB": (2677.64511, 0.0),
        },
        venue="binance",
        now_s=t0,
        verified_ts=t0,
    )
    ro_mgr = cacheaccount.CacheBalanceManager(60, None, str(bal_path), read_only=True, venue="binance")
    prod_mgr = cacheaccount.CacheBalanceManager(60, None, str(bal_path), read_only=False, venue="binance")

    ws_event = {
        "e": "outboundAccountPosition",
        "E": int(t_fill * 1000),
        "u": int(t_fill * 1000),
        "B": [
            {"a": "USDC", "f": "7361.96724542", "l": "899.97440000"},
            {"a": "ADA", "f": "6680.09090000", "l": "0.00000000"},
        ],
    }

    # Read-only consumer refuses WS write
    persisted_ro: list[str] = []
    bapi_ws_bridge.handle_binance_ws_event(
        ws_event,
        cache_getter=lambda name: ro_mgr if name == "BinanceBalance" else None,
        log_enabled=False,
        persist_fn=persisted_ro.append,
    )
    assert persisted_ro == []
    assert ro_mgr.get_balance("ADA")["free"] == pytest.approx(226.4909)

    # Producer applies outboundAccountPosition atomically, advances verified_ts, and preserves untouched ARB
    persisted_prod: list[str] = []
    bapi_ws_bridge.handle_binance_ws_event(
        ws_event,
        cache_getter=lambda name: prod_mgr if name == "BinanceBalance" else None,
        log_enabled=False,
        persist_fn=persisted_prod.append,
    )
    assert persisted_prod == ["outboundAccountPosition"]
    assert prod_mgr.get_balance("ADA")["free"] == pytest.approx(6680.0909)
    assert prod_mgr.get_balance("USDC")["locked"] == pytest.approx(899.9744)
    assert prod_mgr.get_balance("ARB")["free"] == pytest.approx(2677.64511)
    assert prod_mgr.verified_ts == pytest.approx(t_fill, abs=1e-3)

    # Stale out-of-order WS event older than t_fill must NOT regress ADA back to 226.49
    stale_ws_event = {
        "e": "outboundAccountPosition",
        "E": int((t0 + 5.0) * 1000),
        "u": int((t0 + 5.0) * 1000),
        "B": [{"a": "ADA", "f": "226.49090000", "l": "0.00000000"}],
    }
    bapi_ws_bridge.handle_binance_ws_event(
        stale_ws_event,
        cache_getter=lambda name: prod_mgr if name == "BinanceBalance" else None,
        log_enabled=False,
    )
    assert prod_mgr.get_balance("ADA")["free"] == pytest.approx(6680.0909)

    # Cross-venue stale-lock detector catches pre-fill balance cache vs post-fill open_orders (both BUY & SELL)
    obs_stale_buy = {
        ("binance", "USDC"): BalanceObservation(free=7361.97, ts=t0, priority=2, reflected_locked=1866.00),
        ("binance", "ADA"): BalanceObservation(free=226.49, ts=t0, priority=2, reflected_locked=0.0),
    }
    open_after_ada_fill = {
        ("binance", "BTCUSDC"): [{"orderId": "10976647800", "side": "BUY", "price": 59680.0, "origQty": 0.01508, "executedQty": 0.0}],
    }
    assert _venue_balance_cache_has_stale_order_lock(
        "binance", t0, obs_stale_buy, open_after_ada_fill, {"binance": t_fill}, {}
    ) is True
    # When balance cache is fresh (>= open_orders verified_ts), detector stays cache-first (False)
    assert _venue_balance_cache_has_stale_order_lock(
        "binance", t_fill, obs_stale_buy, open_after_ada_fill, {"binance": t_fill}, {}
    ) is False
    # Symmetric SELL fill case: reflected_locked base TAO=5.5387 > open_orders base locked 0.0
    obs_stale_sell = {
        ("binance", "TAO"): BalanceObservation(free=0.005, ts=t0, priority=2, reflected_locked=5.5387),
    }
    assert _venue_balance_cache_has_stale_order_lock(
        "binance", t0, obs_stale_sell, {}, {"binance": t_fill}, {}
    ) is True


def test_cross_venue_event_refresh_balance_update_ws_fetchtime_and_immediate_fill_guard(tmp_path, monkeypatch):
    """Verify cross-venue symmetry (Binance, Kraken, Hyperliquid, T212):
    1. request_venue_account_refresh is producer-only, debounced, refreshes balances + open_orders on non-primary venues and balances-only on Binance.
    2. Binance balanceUpdate WS event triggers BinanceBalance.request_ws_balance_refresh().
    3. Immediate fill (reflected_locked=0) in cache_order_filled_<venue>.json invalidates older cache_balance_<venue>.json via _read_latest_filled_ts_by_venue.
    4. Binance WS open-order removal advances open_latest_ts_by_venue (fetchtime) so stale-lock guard fires before the next REST open-order poll.
    """
    import account_cache_health
    import cacheaccount
    from binance_api import bapi_ws_bridge
    from intelligence.sentiment.collectors.portfolio_collector import (
        _read_cached_open_orders_by_venue_sym,
        _read_latest_filled_ts_by_venue,
        _venue_balance_cache_has_stale_order_lock,
    )
    from providers.portfolio_valuation import BalanceObservation

    # 1. Producer-only event-driven refresh across Kraken/Hyperliquid/T212/Binance + debounce
    calls: list[tuple[str, str]] = []

    class _FakeBalBridge:
        def sync_venue_once(self, v_key: str, *, now_s=None) -> bool:
            calls.append(("bal", v_key))
            return True

    class _FakeOpenBridge:
        def sync_venue_once(self, v_key: str, *, now_ms=None) -> bool:
            calls.append(("open", v_key))
            return True

    monkeypatch.setattr(cacheaccount, "_venue_balances_bridge", _FakeBalBridge())
    monkeypatch.setattr(cacheaccount, "_venue_open_orders_bridge", _FakeOpenBridge())
    monkeypatch.setattr(cacheaccount, "_venue_refresh_last_ts", {})
    monkeypatch.setattr(account_cache_health, "is_writer_enabled", lambda: False)
    assert cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=True, blocking=True) is False
    assert calls == []

    monkeypatch.setattr(account_cache_health, "is_writer_enabled", lambda: True)
    assert cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=True, blocking=True) is True
    assert calls == [("bal", "hyperliquid"), ("open", "hyperliquid")]
    # Second call within 1s debounce window is suppressed
    assert cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=True, blocking=True) is False
    # Binance refresh triggers balances only (open orders already real-time via WS executionReport)
    assert cacheaccount.request_venue_account_refresh("binance", include_open_orders=True, blocking=True) is True
    assert ("bal", "binance") in calls and ("open", "binance") not in calls

    # 2. Binance balanceUpdate WS event routes to BinanceBalance.request_ws_balance_refresh()
    bal_path = tmp_path / "cache_balance_binance.json"
    bal_mgr = cacheaccount.CacheBalanceManager(60, None, str(bal_path), read_only=False, venue="binance")
    monkeypatch.setattr(cacheaccount, "_venue_refresh_last_ts", {})
    persisted_bu: list[str] = []
    bapi_ws_bridge.handle_binance_ws_event(
        {"e": "balanceUpdate", "E": 1795717000000, "a": "BNB", "d": "0.1"},
        cache_getter=lambda name: bal_mgr if name == "BinanceBalance" else None,
        log_enabled=False,
        persist_fn=persisted_bu.append,
    )
    assert persisted_bu == ["balanceUpdate"]

    # 3. Immediate fill (no resting lock ever seen) detected by _read_latest_filled_ts_by_venue on Hyperliquid/Kraken/Binance/T212
    t0 = 1_795_717_000.0
    t_fill = t0 + 12.0
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir()
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(orders_dir / "cache_order_filled_hyperliquid.json"),
        {"HYPEUSDC": [{"orderId": "hl-99", "side": "BUY", "price": 60.5, "qty": 10.0, "time": int(t_fill * 1000), "status": "FILLED"}]},
        venue="hyperliquid",
        now_ms=int(t_fill * 1000),
    )
    latest_fills = _read_latest_filled_ts_by_venue(str(tmp_path))
    assert latest_fills["hyperliquid"] == pytest.approx(t_fill, abs=1e-3)
    assert latest_fills["hyperliquid"] > t0 + 0.5

    # 4. Binance WS open-order removal updates fetchtime (without verified_ts) -> open_latest_ts_by_venue triggers stale-lock check
    open_bn = orders_dir / "cache_order_open_binance.json"
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(open_bn),
        {"ADAUSDC": [{"orderId": "1706227251", "side": "BUY", "price": 0.1497, "origQty": 6680.0, "status": "NEW", "timestamp": int((t0 - 10) * 1000)}]},
        now_ms=int(t0 * 1000),
        verified_ts=t0,
    )
    oo_mgr = cacheaccount.CacheOrderOpenManager(
        15,
        ["ADAUSDC"],
        str(open_bn),
        read_only=False,
        venue="binance",
        data_sync_mode=cacheaccount.CacheManagerInterface.MODE_PRODUCER,
    )
    assert oo_mgr.apply_ws_order_update("ADAUSDC", "remove", {"orderId": "1706227251"}, now_ms=int(t_fill * 1000)) is True
    oo_mgr.save_state_to_file_if_enabled()

    latest_open_ts: dict[str, float] = {}
    by_vs, ver_v, sym_ver = _read_cached_open_orders_by_venue_sym(str(tmp_path), out_latest_ts=latest_open_ts)
    assert ver_v.get("binance", 0.0) == pytest.approx(t0, abs=1e-3)  # verified_ts did NOT advance on WS delta
    assert latest_open_ts["binance"] == pytest.approx(t_fill, abs=1e-3)  # fetchtime DID advance!
    obs_stale_ws = {("binance", "USDC"): BalanceObservation(free=7361.97, ts=t0, priority=2, reflected_locked=1000.0)}
    assert _venue_balance_cache_has_stale_order_lock(
        "binance", t0, obs_stale_ws, by_vs, ver_v, sym_ver, cached_open_latest_ts=latest_open_ts
    ) is True


def test_portfolio_collector_multi_venue_remaining_qty_and_future_ts_invariants(tmp_path):
    """Pure invariants + dependent cross-checks on stale-lock detection, audit order dedup, and runtime balance views."""
    from intelligence.sentiment.collectors.portfolio_collector import (
        TradablePositionsAndOrdersCollector,
        _merge_audit_orders,
        _venue_balance_cache_has_stale_order_lock,
        derive_runtime_portfolio_view,
    )
    from providers.portfolio_valuation import BalanceObservation

    t0 = 1_795_720_000.0
    t_open = t0 + 15.0

    # 1. Pure invariant: Kraken vol/vol_exec and Hyperliquid origSz/sz are honored accurately in stale-lock check
    obs_kr = {("kraken", "HYPE"): BalanceObservation(free=1.0, ts=t0, priority=2, reflected_locked=10.0)}
    # Order partially filled (vol=10, vol_exec=6 -> remaining=4 < reflected_locked=10) -> stale lock detected!
    open_kr_partial = {
        ("kraken", "HYPEUSD"): [{"orderId": "KR-1", "side": "SELL", "price": 60.0, "vol": "10.0", "vol_exec": "6.0"}]
    }
    assert _venue_balance_cache_has_stale_order_lock(
        "kraken", t0, obs_kr, open_kr_partial, {"kraken": t_open}, {}, now_ts=t_open + 1.0
    ) is True
    # Order completely unfilled (vol=10, vol_exec=0 -> remaining=10 == reflected_locked=10) -> NOT stale!
    open_kr_full = {
        ("kraken", "HYPEUSD"): [{"orderId": "KR-1", "side": "SELL", "price": 60.0, "vol": "10.0", "vol_exec": "0.0"}]
    }
    assert _venue_balance_cache_has_stale_order_lock(
        "kraken", t0, obs_kr, open_kr_full, {"kraken": t_open}, {}, now_ts=t_open + 1.0
    ) is False

    # 2. Pure invariant: epoch-ms open_orders ts normalizes cleanly, and corrupt far-future bal_verified_ts fails closed (True)
    assert _venue_balance_cache_has_stale_order_lock(
        "kraken", t0, obs_kr, open_kr_partial, {"kraken": t_open * 1000.0}, {}, now_ts=t_open + 1.0
    ) is True
    assert _venue_balance_cache_has_stale_order_lock(
        "kraken", t_open + 3600.0, obs_kr, open_kr_full, {"kraken": t_open}, {}, now_ts=t_open + 1.0
    ) is True

    # 3. Pure invariant: _merge_audit_orders deduplicates Kraken vol_exec fills against audit executedQty without double-counting
    kr_primary = [{"orderId": "K-PRIMARY", "side": "BUY", "vol_exec": "4.25", "time": int(t0 * 1000)}]
    kr_audit = [{"orderId": "K-AUDIT-ALT", "side": "BUY", "executedQty": "4.25", "timestamp": int((t0 + 2.0) * 1000)}]
    merged = _merge_audit_orders(kr_primary, kr_audit)
    assert len(merged) == 1

    # 4. Dependent cross-check: corrupt far-future _balance_ts on cash/pos does not disable synthetic lock or persist future ts
    positions = [
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "base": "BTC",
            "quote": "USDC",
            "price": 60000.0,
            "free_base": 0.0,
            "free_quote": 2000.0,
            "_balance_ts": t_open + 9999.0,
        }
    ]
    venue_cash = {
        "binance:USDC": {
            "venue": "binance",
            "asset": "USDC",
            "free": 2000.0,
            "_balance_ts": t_open + 9999.0,
        }
    }
    updated_bals = {
        "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 2000.0, "ts": t_open + 9999.0},
        "pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "base": "BTC", "quote": "USDC", "free_base": 0.0, "free_quote": 2000.0, "ts": t_open + 9999.0},
    }
    derive_runtime_portfolio_view(
        [{"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 60000.0, "remaining_qty": 0.01, "created_ts": t_open * 1000.0}],
        positions,
        venue_cash,
        {},
        {"BTCUSDC": {"price": 60000.0}},
        updated_bals,
        {},
        now_ts_oo=t_open + 5.0,
        gateway_max_usd=2000.0,
    )
    assert venue_cash["binance:USDC"]["locked_quote_usd"] == pytest.approx(600.0)
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(exist_ok=True)
    TradablePositionsAndOrdersCollector._persist_venue_balances_locked(
        str(tmp_path),
        {},
        updated_bals,
        set(),
        now_ts=t_open + 5.0,
        rj_bals=lambda _p, **_k: {},
    )
    saved_bal = json.loads((orders_dir / "cache_balance_binance.json").read_text(encoding="utf-8"))
    assert saved_bal["items"]["USDC"]["ts"] <= t_open + 5.0 + 1.0



