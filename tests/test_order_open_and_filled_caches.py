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
    from providers.portfolio_valuation import load_venue_balances_state_dict

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
    from providers.quantity import resolve_kraken_trade_cache_position, summarize_fills_position_and_cost

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
