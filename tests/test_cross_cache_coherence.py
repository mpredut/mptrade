"""Cross-cache event fan-out and reflection coherence regression tests.

Verifies that multiplexed order/balance events stay mutually coherent across all six
account reflection surfaces (open orders, filled orders, trade mirror, balances,
order ledger, and account health) across binance, kraken, hyperliquid, and t212,
and that every drift class is deterministically caught by the coherence evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import account_cache_health as health
import cacheaccount
from binance_api.bapi_ws_bridge import (
    apply_open_order_from_execution_report,
    handle_binance_ws_event,
)
from order_ledger_reconciler_worker import OrderLedgerReconciler
from providers.base import MarketDataProvider
from state_io import atomic_write_json
from verify_tools import check_cache_coherence as ccc
from verify_tools.account_cache_coherence import (
    AccountCacheCoherenceSnapshot,
    VenueCoherenceEvidence,
    audit_account_cache_coherence,
    evaluate_account_cache_coherence,
)


def _write_health_doc(
    path: Path,
    *,
    now_ms: int,
    started_ms: int,
    venue_streams: dict[str, dict[str, int]],
) -> None:
    venues_doc: dict[str, dict[str, Any]] = {}
    for v_key, streams in venue_streams.items():
        entry: dict[str, Any] = {}
        for s_name, ts_ms in streams.items():
            entry[f"{s_name}_sync_at_ms"] = int(ts_ms)
            entry[f"{s_name}_version"] = f"v-{s_name}"
        venues_doc[v_key] = entry
    bn = venue_streams.get("binance", {})
    doc = {
        "schema_version": health.SCHEMA_VERSION,
        "writer_pid": 4321,
        "writer_started_at_ms": started_ms,
        "published_at_ms": now_ms,
        "generation": 5,
        "stopping": False,
        "order_sync_at_ms": int(bn.get("filled_orders", now_ms)),
        "trade_sync_at_ms": int(bn.get("trades", now_ms)),
        "order_cache_version": "v-filled_orders",
        "trade_cache_version": "v-trades",
        "venues": venues_doc,
    }
    atomic_write_json(str(path), doc, indent=None)


def _append_jsonl_trade(path: Path, symbol: str, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"s": symbol, "i": item}) + "\n")


def test_binance_ws_event_fanout_lifecycle_stays_coherent(tmp_path, monkeypatch):
    """End-to-end Binance WS fan-out: NEW -> PARTIAL -> FILLED and NEW -> CANCELED keep all 6 caches coherent."""
    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    t0 = 1_795_720_000.0
    t0_ms = int(t0 * 1000)

    oo_path = acct_dir / "cache_order_open_binance.json"
    fo_path = acct_dir / "cache_order_filled_binance.json"
    tr_path = acct_dir / "cache_trade_binance.jsonl"
    bal_path = acct_dir / "cache_balance_binance.json"
    health_path = acct_dir / "cache_account_health.json"

    # Seed empty baseline caches at t0.
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path), {"ADAUSDC": []}, now_ms=t0_ms, verified_ts=t0
    )
    atomic_write_json(str(fo_path), {"items": {"ADAUSDC": []}, "fetchtime": {"ADAUSDC": t0_ms}})
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path),
        {"USDC": (2000.0, 0.0), "ADA": (0.0, 0.0)},
        venue="binance",
        now_s=t0,
        verified_ts=t0,
    )

    oo_mgr = cacheaccount.CacheOrderOpenManager(
        15,
        ["ADAUSDC"],
        str(oo_path),
        read_only=False,
        venue="binance",
        data_sync_mode=cacheaccount.CacheManagerInterface.MODE_PRODUCER,
    )
    bal_mgr = cacheaccount.CacheBalanceManager(
        60,
        ["USDC", "ADA"],
        str(bal_path),
        read_only=False,
        venue="binance",
        data_sync_mode=cacheaccount.CacheManagerInterface.MODE_PRODUCER,
    )
    monkeypatch.setattr(cacheaccount.account_cache_health, "is_writer_enabled", lambda: True)
    monkeypatch.setattr(cacheaccount, "_record_venue_health_sync", lambda *a, **k: True)

    def _apply_filled_row(ev: dict[str, Any]) -> bool:
        if ev.get("x") != "TRADE":
            return False
        sym = str(ev.get("s") or "ADAUSDC")
        oid = str(ev.get("i") or "")
        qty = float(ev.get("z") or 0.0)
        quote_qty = float(ev.get("Z") or 0.0)
        avg_px = (quote_qty / qty) if qty > 0 else float(ev.get("p") or 0.0)
        row = {
            "orderId": oid,
            "symbol": sym,
            "side": str(ev.get("S") or "BUY"),
            "status": str(ev.get("X") or ""),
            "origQty": float(ev.get("q") or 0.0),
            "executedQty": qty,
            "quantity": qty,
            "price": avg_px,
            "timestamp": int(ev.get("T") or ev.get("E") or t0_ms),
            "_fillIds": [str(ev.get("t"))] if ev.get("t") else [],
        }
        cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
            str(fo_path),
            {sym: [row]},
            venue="binance",
            now_ms=int(ev.get("E") or t0_ms),
            is_snapshot=False,
            touch_symbols=[sym],
            touch_create_missing=True,
        )
        return True

    def _append_trade(ev: dict[str, Any]) -> bool:
        if ev.get("x") != "TRADE":
            return False
        _append_jsonl_trade(
            tr_path,
            str(ev.get("s") or "ADAUSDC"),
            {
                "symbol": str(ev.get("s") or "ADAUSDC"),
                "id": str(ev.get("t")),
                "orderId": str(ev.get("i")),
                "price": str(ev.get("L") or ev.get("p")),
                "qty": str(ev.get("l")),
                "time": int(ev.get("T") or ev.get("E") or t0_ms),
                "isBuyer": str(ev.get("S")) == "BUY",
            },
        )
        return True

    def _getter(name: str):
        if name == "BinanceOrderOpen":
            return oo_mgr
        if name == "BinanceBalance":
            return bal_mgr
        return None

    reconciler = OrderLedgerReconciler(cachedb_dir=str(tmp_path), use_singletons=False)

    def _assert_coherent(now_s: float) -> None:
        now_ms_i = int(now_s * 1000)
        cacheaccount.CacheOrderFilledManager.touch_venue_filled_orders_snapshot(
            str(fo_path), now_ms=now_ms_i, symbols=["ADAUSDC"], create_missing=True
        )
        _, bal_ver, _ = cacheaccount.CacheBalanceManager.read_venue_balances_snapshot(str(bal_path))
        _, oo_ft, oo_ver, _ = cacheaccount.CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(oo_path))
        oo_lat = max([oo_ver, *(float(v) / 1000.0 if float(v) > 1e12 else float(v) for v in (oo_ft or {}).values())])
        _write_health_doc(
            health_path,
            now_ms=now_ms_i,
            started_ms=t0_ms - 5_000,
            venue_streams={
                "binance": {
                    "balances": int(bal_ver * 1000),
                    "open_orders": int(oo_lat * 1000),
                    "filled_orders": now_ms_i,
                    "trades": now_ms_i,
                }
            },
        )
        reconciler.reconcile_once(now_ts=now_s)
        viols = audit_account_cache_coherence(
            str(tmp_path),
            now_s=now_s,
            venues=("binance",),
            health_path=str(health_path),
            explicit_ledger_dir=str(tmp_path),
        )
        assert viols == [], [v.format_line() for v in viols]

    def _dispatch_exec(ev: dict[str, Any]) -> None:
        handle_binance_ws_event(
            ev,
            cache_getter=_getter,
            upsert_order_fn=_apply_filled_row,
            append_trade_fn=_append_trade,
            apply_open_order_fn=lambda e: apply_open_order_from_execution_report(e, cache_getter=_getter),
            persist_fn=lambda _et: oo_mgr.save_state_to_file_if_enabled(),
        )

    def _sync_open_ver(ts_s: float) -> None:
        cur_items, _, _ = cacheaccount.CacheOrderOpenManager.read_venue_open_orders_snapshot(str(oo_path))
        cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
            str(oo_path),
            {"ADAUSDC": list(cur_items.get("ADAUSDC") or [])},
            now_ms=int(ts_s * 1000),
            verified_ts=ts_s,
        )
        oo_mgr._load_snapshot_state()

    # Step 1: LIMIT BUY NEW (locks 1000 USDC).
    t1 = t0 + 10.0
    _dispatch_exec(
        {
            "e": "executionReport",
            "E": int(t1 * 1000),
            "T": int(t1 * 1000),
            "s": "ADAUSDC",
            "S": "BUY",
            "o": "LIMIT",
            "q": "6680.0",
            "p": "0.1497",
            "x": "NEW",
            "X": "NEW",
            "i": 9001,
            "z": "0.0",
            "Z": "0.0",
        }
    )
    handle_binance_ws_event(
        {
            "e": "outboundAccountPosition",
            "E": int(t1 * 1000),
            "B": [{"a": "USDC", "f": "1000.0", "l": "1000.0"}, {"a": "ADA", "f": "0.0", "l": "0.0"}],
        },
        cache_getter=_getter,
    )
    # Refresh open_orders verified_ts so symbol verification advances past t1.
    _sync_open_ver(t1 + 1.0)
    _assert_coherent(t1 + 2.0)

    # Step 2: PARTIALLY_FILLED (2000/6680 filled -> 4680 remaining, ~700.60 USDC still locked).
    t2 = t1 + 30.0
    ev_partial = {
        "e": "executionReport",
        "E": int(t2 * 1000),
        "T": int(t2 * 1000),
        "s": "ADAUSDC",
        "S": "BUY",
        "o": "LIMIT",
        "q": "6680.0",
        "p": "0.1497",
        "x": "TRADE",
        "X": "PARTIALLY_FILLED",
        "i": 9001,
        "t": 50001,
        "l": "2000.0",
        "L": "0.1497",
        "z": "2000.0",
        "Z": "299.4",
    }
    _dispatch_exec(ev_partial)
    handle_binance_ws_event(
        {
            "e": "outboundAccountPosition",
            "E": int(t2 * 1000),
            "B": [{"a": "USDC", "f": "1000.0", "l": "700.6"}, {"a": "ADA", "f": "2000.0", "l": "0.0"}],
        },
        cache_getter=_getter,
    )
    _sync_open_ver(t2 + 1.0)
    _assert_coherent(t2 + 25.0)

    # Step 3: FULL FILL (6680/6680 -> removed from open_orders, 0 USDC locked, 6680 ADA free).
    t3 = t2 + 40.0
    ev_full = {
        "e": "executionReport",
        "E": int(t3 * 1000),
        "T": int(t3 * 1000),
        "s": "ADAUSDC",
        "S": "BUY",
        "o": "LIMIT",
        "q": "6680.0",
        "p": "0.1497",
        "x": "TRADE",
        "X": "FILLED",
        "i": 9001,
        "t": 50002,
        "l": "4680.0",
        "L": "0.1497",
        "z": "6680.0",
        "Z": "1000.0",
    }
    _dispatch_exec(ev_full)
    handle_binance_ws_event(
        {
            "e": "outboundAccountPosition",
            "E": int(t3 * 1000),
            "B": [{"a": "USDC", "f": "1000.0", "l": "0.0"}, {"a": "ADA", "f": "6680.0", "l": "0.0"}],
        },
        cache_getter=_getter,
    )
    _sync_open_ver(t3 + 1.0)
    _assert_coherent(t3 + 50.0)

    # Step 4: LIMIT SELL NEW -> CANCELED (ADA locked then cleanly unlocked).
    t4 = t3 + 60.0
    _dispatch_exec(
        {
            "e": "executionReport",
            "E": int(t4 * 1000),
            "T": int(t4 * 1000),
            "s": "ADAUSDC",
            "S": "SELL",
            "o": "LIMIT",
            "q": "3000.0",
            "p": "0.1600",
            "x": "NEW",
            "X": "NEW",
            "i": 9002,
            "z": "0.0",
            "Z": "0.0",
        }
    )
    handle_binance_ws_event(
        {
            "e": "outboundAccountPosition",
            "E": int(t4 * 1000),
            "B": [{"a": "ADA", "f": "3680.0", "l": "3000.0"}],
        },
        cache_getter=_getter,
    )
    _sync_open_ver(t4 + 1.0)
    _assert_coherent(t4 + 2.0)

    t5 = t4 + 20.0
    _dispatch_exec(
        {
            "e": "executionReport",
            "E": int(t5 * 1000),
            "T": int(t5 * 1000),
            "s": "ADAUSDC",
            "S": "SELL",
            "o": "LIMIT",
            "q": "3000.0",
            "p": "0.1600",
            "x": "CANCELED",
            "X": "CANCELED",
            "i": 9002,
            "z": "0.0",
            "Z": "0.0",
        }
    )
    handle_binance_ws_event(
        {
            "e": "outboundAccountPosition",
            "E": int(t5 * 1000),
            "B": [{"a": "ADA", "f": "6680.0", "l": "0.0"}],
        },
        cache_getter=_getter,
    )
    _sync_open_ver(t5 + 1.0)
    _assert_coherent(t5 + 50.0)


def test_non_binance_open_order_delta_triggers_balance_refresh_and_fanout_coherence(tmp_path, monkeypatch):
    """Open-order placement/cancel deltas trigger immediate balance refresh on kraken/hyperliquid/t212 and keep caches coherent."""
    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    t0 = 1_795_725_000.0

    # 1. Verify _VenueOpenOrdersBridge.sync_venue_once triggers balance refresh ONLY on open-order deltas (not seed/steady).
    state_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = {
        "kraken": {"SPCXxUSD": []},
        "hyperliquid": {"HYPEUSDC": []},
        "t212": {"NVDA_US_EQ": []},
    }
    refresh_calls: list[tuple[str, bool]] = []

    class _FakeProv:
        def __init__(self, venue: str):
            self._v = venue

        def open_orders_by_symbol(self, syms):
            return {s: list(state_by_venue[self._v].get(s, [])) for s in syms}

    monkeypatch.setattr(cacheaccount.sym, "symbols_for", lambda v: list(state_by_venue[v].keys()))
    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: (lambda: _FakeProv(v)) if v in state_by_venue else None),
    )
    monkeypatch.setattr(cacheaccount, "cache_path", lambda fname: str(acct_dir / fname))
    monkeypatch.setattr(cacheaccount, "_record_venue_health_sync", lambda *a, **k: True)
    monkeypatch.setattr(
        cacheaccount,
        "request_venue_account_refresh",
        lambda v, *, include_open_orders=False, blocking=False: refresh_calls.append((v, include_open_orders)) or True,
    )

    bridge = cacheaccount._VenueOpenOrdersBridge()
    # Initial seed: no refresh triggered.
    for v in ("kraken", "hyperliquid", "t212"):
        assert bridge.sync_venue_once(v, now_ms=int(t0 * 1000)) is True
    assert refresh_calls == []

    # Steady-state poll (unchanged): no refresh triggered.
    for v in ("kraken", "hyperliquid", "t212"):
        assert bridge.sync_venue_once(v, now_ms=int((t0 + 15) * 1000)) is True
    assert refresh_calls == []

    # Order placed on each venue -> delta triggers immediate balance refresh!
    state_by_venue["kraken"]["SPCXxUSD"] = [
        {"orderId": "KR-1", "symbol": "SPCXxUSD", "side": "SELL", "origQty": 2.0, "executedQty": 0.0, "price": 135.0, "timestamp": int((t0 + 20) * 1000)}
    ]
    state_by_venue["hyperliquid"]["HYPEUSDC"] = [
        {"orderId": "HL-1", "symbol": "HYPEUSDC", "side": "BUY", "origQty": 5.0, "executedQty": 0.0, "price": 60.0, "timestamp": int((t0 + 20) * 1000)}
    ]
    state_by_venue["t212"]["NVDA_US_EQ"] = [
        {"orderId": "T212-1", "symbol": "NVDA_US_EQ", "side": "SELL", "origQty": 1.5, "executedQty": 0.0, "price": 150.0, "timestamp": int((t0 + 20) * 1000)}
    ]
    for v in ("kraken", "hyperliquid", "t212"):
        assert bridge.sync_venue_once(v, now_ms=int((t0 + 20) * 1000)) is True
    assert refresh_calls == [("kraken", False), ("hyperliquid", False), ("t212", False)]

    # When called with trigger_balance_refresh=False (from fill refresh), no duplicate balance refresh is fired.
    refresh_calls.clear()
    state_by_venue["kraken"]["SPCXxUSD"] = []
    assert bridge.sync_venue_once("kraken", now_ms=int((t0 + 30) * 1000), trigger_balance_refresh=False) is True
    assert refresh_calls == []

    # 2. Verify coherent multi-venue reflection across kraken (mixed-case SPCXx), hyperliquid (HYPEUSDC), t212 (NVDA_US_EQ).
    t_now = t0 + 40.0
    t_now_ms = int(t_now * 1000)
    # Restore Kraken open SELL so all 3 venues have active orders + matching locked balances + recent fills.
    state_by_venue["kraken"]["SPCXxUSD"] = [
        {"orderId": "KR-1", "symbol": "SPCXxUSD", "side": "SELL", "origQty": 2.0, "executedQty": 0.0, "price": 135.0, "timestamp": int((t_now - 5) * 1000)}
    ]
    for v in ("kraken", "hyperliquid", "t212"):
        bridge.sync_venue_once(v, now_ms=t_now_ms, trigger_balance_refresh=False)

    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(acct_dir / "cache_balance_kraken.json"),
        {"USD": (500.0, 0.0), "SPCXx": (1.0, 2.0)},
        venue="kraken",
        now_s=t_now,
        verified_ts=t_now,
    )
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(acct_dir / "cache_balance_hyperliquid.json"),
        {"USDC": (400.0, 300.0), "HYPE": (10.0, 0.0)},
        venue="hyperliquid",
        now_s=t_now,
        verified_ts=t_now,
    )
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(acct_dir / "cache_balance_t212.json"),
        {"USD": (800.0, 0.0), "NVDA_US_EQ": (0.5, 1.5)},
        venue="t212",
        now_s=t_now,
        verified_ts=t_now,
    )
    for v, sym, oid in (("kraken", "SPCXxUSD", "KR-F1"), ("hyperliquid", "HYPEUSDC", "HL-F1"), ("t212", "NVDA_US_EQ", "T212-F1")):
        fill_ts_ms = int((t_now - 25.0) * 1000)
        atomic_write_json(
            str(acct_dir / f"cache_order_filled_{v}.json"),
            {
                "items": {sym: [{"orderId": oid, "symbol": sym, "side": "BUY", "status": "FILLED", "quantity": 1.0, "price": 100.0, "timestamp": fill_ts_ms}]},
                "fetchtime": {sym: t_now_ms},
            },
        )
        atomic_write_json(
            str(acct_dir / f"cache_trade_{v}.json"),
            {
                "items": {sym: [{"orderId": oid, "symbol": sym, "side": "BUY", "qty": 1.0, "price": 100.0, "time": fill_ts_ms}]},
                "fetchtime": {sym: t_now_ms},
            },
        )

    health_path = acct_dir / "cache_account_health.json"
    _write_health_doc(
        health_path,
        now_ms=t_now_ms,
        started_ms=t_now_ms - 5_000,
        venue_streams={
            v: {"balances": t_now_ms, "open_orders": t_now_ms, "filled_orders": t_now_ms, "trades": t_now_ms}
            for v in ("kraken", "hyperliquid", "t212")
        },
    )

    viols = audit_account_cache_coherence(
        str(tmp_path),
        now_s=t_now,
        venues=("kraken", "hyperliquid", "t212"),
        health_path=str(health_path),
        include_ledger=False,
    )
    assert viols == [], [v.format_line() for v in viols]


def test_cross_cache_auditor_catches_every_drift_class_across_venues():
    """Pure evaluator catches all 10 cross-cache drift rules and honors propagation grace windows."""
    t_now = 1_795_730_000.0

    # 1. causal_fill_to_balance (caught past grace, ignored inside grace)
    ev_in_grace = VenueCoherenceEvidence(
        venue="hyperliquid",
        filled_items={"HYPEUSDC": [{"orderId": "10", "symbol": "HYPEUSDC", "status": "FILLED", "timestamp": int((t_now - 4.0) * 1000)}]},
        filled_file_exists=True,
        balance_items={"USDC": {"free": 500.0, "locked": 0.0, "ts": t_now - 40.0}},
        balance_verified_ts=t_now - 40.0,
        balance_file_exists=True,
    )
    assert evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"hyperliquid": ev_in_grace}), sync_grace_sec=15.0) == []

    ev_past_grace = VenueCoherenceEvidence(
        venue="hyperliquid",
        filled_items={"HYPEUSDC": [{"orderId": "10", "symbol": "HYPEUSDC", "status": "FILLED", "timestamp": int((t_now - 20.0) * 1000)}]},
        filled_file_exists=True,
        balance_items={"USDC": {"free": 500.0, "locked": 0.0, "ts": t_now - 50.0}},
        balance_verified_ts=t_now - 50.0,
        balance_file_exists=True,
    )
    v_causal = evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"hyperliquid": ev_past_grace}), sync_grace_sec=15.0)
    assert [v.rule for v in v_causal] == ["causal_fill_to_balance"]

    # 2. balance_phantom_lock (quote & base) + open_order_missing_balance_lock (quote & base)
    ev_phantom = VenueCoherenceEvidence(
        venue="binance",
        open_items={"ADAUSDC": []},
        open_verified_ts=t_now - 20.0,
        open_latest_ts=t_now - 20.0,
        open_file_exists=True,
        balance_items={
            "USDC": {"free": 100.0, "locked": 966.0, "ts": t_now - 55.0},
            "ADA": {"free": 0.0, "locked": 500.0, "ts": t_now - 55.0},
        },
        balance_verified_ts=t_now - 55.0,
        balance_file_exists=True,
        configured_base_assets=frozenset({"ADA"}),
    )
    v_phantom = evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"binance": ev_phantom}), sync_grace_sec=15.0)
    assert [v.rule for v in v_phantom] == ["balance_phantom_lock", "balance_phantom_lock"]

    ev_missing_lock = VenueCoherenceEvidence(
        venue="kraken",
        open_items={
            "HYPEUSD": [{"orderId": "K1", "symbol": "HYPEUSD", "side": "BUY", "origQty": 10.0, "executedQty": 0.0, "price": 60.0}],
            "SPCXxUSD": [{"orderId": "K2", "symbol": "SPCXxUSD", "side": "SELL", "origQty": 3.0, "executedQty": 0.0, "price": 130.0}],
        },
        open_verified_ts=t_now - 10.0,
        open_latest_ts=t_now - 10.0,
        open_file_exists=True,
        balance_items={
            "USD": {"free": 900.0, "locked": 0.0, "ts": t_now - 5.0},
            "SPCXx": {"free": 3.0, "locked": 0.0, "ts": t_now - 5.0},
        },
        balance_verified_ts=t_now - 5.0,
        balance_file_exists=True,
        configured_base_assets=frozenset({"HYPE", "SPCXx"}),
    )
    v_miss_lock = evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_missing_lock}), sync_grace_sec=15.0)
    assert [v.rule for v in v_miss_lock] == ["open_order_missing_balance_lock", "open_order_missing_balance_lock"]

    # Unknown locked=None must NOT falsely flag missing balance lock.
    ev_none_lock = VenueCoherenceEvidence(
        venue="t212",
        open_items={"NVDA_US_EQ": [{"orderId": "T1", "symbol": "NVDA_US_EQ", "side": "SELL", "origQty": 2.0, "executedQty": 0.0, "price": 150.0}]},
        open_verified_ts=t_now - 10.0,
        open_latest_ts=t_now - 10.0,
        open_file_exists=True,
        balance_items={"NVDA_US_EQ": {"free": 2.0, "locked": None, "ts": t_now - 5.0}},
        balance_verified_ts=t_now - 5.0,
        balance_file_exists=True,
        configured_base_assets=frozenset({"NVDA_US_EQ"}),
    )
    assert evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"t212": ev_none_lock}), sync_grace_sec=15.0) == []

    # 3. open_order_already_filled (terminal caught, partial allowed)
    ev_ghost_open = VenueCoherenceEvidence(
        venue="binance",
        open_items={
            "BTCUSDC": [
                {"orderId": "B1", "symbol": "BTCUSDC", "side": "BUY", "origQty": 0.1, "executedQty": 0.0, "price": 60000.0},
                {"orderId": "B2", "symbol": "BTCUSDC", "side": "BUY", "origQty": 0.2, "executedQty": 0.1, "price": 60000.0},
            ]
        },
        open_verified_ts=t_now - 5.0,
        open_latest_ts=t_now - 5.0,
        open_file_exists=True,
        filled_items={
            "BTCUSDC": [
                {"orderId": "B1", "symbol": "BTCUSDC", "status": "FILLED", "quantity": 0.1, "timestamp": int((t_now - 30.0) * 1000)},
                {"orderId": "B2", "symbol": "BTCUSDC", "status": "PARTIALLY_FILLED", "quantity": 0.1, "timestamp": int((t_now - 30.0) * 1000)},
            ]
        },
        filled_file_exists=True,
    )
    v_ghost = evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"binance": ev_ghost_open}), sync_grace_sec=15.0)
    assert [v.rule for v in v_ghost] == ["open_order_already_filled"]
    assert "B1" in v_ghost[0].detail

    # 4. trade_missing_in_filled
    ev_tr_gap = VenueCoherenceEvidence(
        venue="kraken",
        filled_items={"HYPEUSD": [{"orderId": "OK1", "symbol": "HYPEUSD", "timestamp": int((t_now - 60.0) * 1000)}]},
        filled_file_exists=True,
        trade_items={
            "HYPEUSD": [
                {"orderId": "OK1", "symbol": "HYPEUSD", "time": int((t_now - 60.0) * 1000)},
                {"orderId": "MISS1", "symbol": "HYPEUSD", "time": int((t_now - 45.0) * 1000)},
            ]
        },
    )
    v_tr = evaluate_account_cache_coherence(AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_tr_gap}), sync_grace_sec=15.0)
    assert [v.rule for v in v_tr] == ["trade_missing_in_filled"]
    assert "MISS1" in v_tr[0].detail

    # 5. Ledger rules: open_order_missing_from_ledger, ledger_stuck_filled_order (incl. status=None venue fills), ledger_orphan_active_order
    ev_ledger = VenueCoherenceEvidence(
        venue="binance",
        open_items={"TAOUSDC": [{"orderId": "REST-9", "symbol": "TAOUSDC", "side": "SELL", "origQty": 1.0, "executedQty": 0.0, "price": 190.0, "timestamp": int((t_now - 90.0) * 1000)}]},
        open_verified_ts=t_now - 2.0,
        open_latest_ts=t_now - 2.0,
        open_sym_verified_ts={"taousdc": t_now - 2.0, "adausdc": t_now - 2.0},
        open_file_exists=True,
        filled_items={
            "TAOUSDC": [{"orderId": "FILLED-8", "symbol": "TAOUSDC", "status": "FILLED", "quantity": 2.0, "timestamp": int((t_now - 80.0) * 1000)}],
            "ARBUSDC": [{"orderId": "FILL-NOSTAT-6", "symbol": "ARBUSDC", "quantity": 321.3, "timestamp": int((t_now - 80.0) * 1000)}],
        },
        filled_file_exists=True,
    )
    snap_ledger = AccountCacheCoherenceSnapshot(
        now_s=t_now,
        venues={"binance": ev_ledger},
        ledger_present=True,
        ledger_updated_ts=t_now - 1.0,
        ledger_active_orders={
            "binance:FILLED-8": {"v": "binance", "sym": "TAOUSDC", "vid": "FILLED-8", "state": "OPEN", "qty": 2.0, "updated_ts": t_now - 100.0},
            "binance:FILL-NOSTAT-6": {"v": "binance", "sym": "ARBUSDC", "vid": "FILL-NOSTAT-6", "state": "SUBMITTED", "qty": 321.3, "updated_ts": t_now - 100.0},
            "binance:ORPHAN-7": {"v": "binance", "sym": "ADAUSDC", "vid": "ORPHAN-7", "state": "OPEN", "updated_ts": t_now - 100.0},
        },
    )
    v_led = evaluate_account_cache_coherence(snap_ledger, sync_grace_sec=15.0, ledger_grace_sec=45.0)
    assert sorted(v.rule for v in v_led) == [
        "ledger_orphan_active_order",
        "ledger_stuck_filled_order",
        "ledger_stuck_filled_order",
        "open_order_missing_from_ledger",
    ]

    # 6. health_disk_missing & health_disk_lag
    ev_health = VenueCoherenceEvidence(
        venue="hyperliquid",
        open_file_exists=False,
        balance_file_exists=True,
        balance_verified_ts=t_now - 90.0,
        filled_file_exists=True,
        filled_latest_ft_ts=t_now - 2.0,
        health_sync_ts_by_stream={"open_orders": t_now - 2.0, "balances": t_now - 2.0, "filled_orders": t_now - 2.0},
    )
    v_hlth = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=t_now, venues={"hyperliquid": ev_health}, health_present=True),
        sync_grace_sec=15.0,
    )
    assert sorted(v.rule for v in v_hlth) == ["health_disk_lag", "health_disk_missing"]


def test_check_cache_coherence_cli_fails_on_cross_cache_drift_and_passes_when_coherent(tmp_path, monkeypatch, capsys):
    """check_cache_coherence.main() fails on stale balance after fill even when health ages are < 600s."""
    coh_dir = tmp_path / "coh"
    acct_dir = coh_dir / "account"
    acct_dir.mkdir(parents=True)

    now_s = 1_795_735_000.0
    now_ms = int(now_s * 1000)
    started_ms = now_ms - 60_000

    atomic_write_json(
        str(coh_dir / "cache_short_trend.json"),
        {"items": {"BTCUSDC": {"ts": now_s - 2.0, "current_price": 60000.0, "gradient_recent": 0.01}}},
    )
    atomic_write_json(
        str(coh_dir / "cache_current_price.json"),
        {"items": {"BTCUSDC": [[(now_s - 0.5) * 1000.0, 60000.0]]}},
    )

    # Populate all 4 venues coherently first.
    streams_ok: dict[str, dict[str, int]] = {}
    for v_key in health.SUPPORTED_HEALTH_VENUES:
        cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
            str(acct_dir / cacheaccount.CacheOrderOpenManager.VENUE_FILENAMES[v_key]),
            {"BTCUSDC": []},
            now_ms=now_ms - 2_000,
            verified_ts=now_s - 2.0,
        )
        atomic_write_json(
            str(acct_dir / cacheaccount.CacheOrderFilledManager.VENUE_FILENAMES[v_key]),
            {
                "items": {"BTCUSDC": [{"orderId": f"{v_key}-1", "symbol": "BTCUSDC", "status": "FILLED", "timestamp": now_ms - 40_000}]},
                "fetchtime": {"BTCUSDC": now_ms - 2_000},
            },
        )
        cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
            str(acct_dir / cacheaccount.CacheBalanceManager.VENUE_FILENAMES[v_key]),
            {"USDC": (1000.0, 0.0), "USD": (1000.0, 0.0)},
            venue=v_key,
            now_s=now_s - 2.0,
            verified_ts=now_s - 2.0,
        )
        streams_ok[v_key] = {
            "balances": now_ms - 2_000,
            "open_orders": now_ms - 2_000,
            "filled_orders": now_ms - 2_000,
            "trades": now_ms - 2_000,
        }
    health_file = acct_dir / "cache_account_health.json"
    _write_health_doc(health_file, now_ms=now_ms, started_ms=started_ms, venue_streams=streams_ok)

    monkeypatch.setattr(ccc, "CACHEDB", str(coh_dir))
    monkeypatch.setattr(ccc, "expected_symbols", lambda: ["BTCUSDC"])
    monkeypatch.setattr(ccc.time, "time", lambda: now_s)
    monkeypatch.setattr(health, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(health, "_pid_started_at_ms", lambda _pid: started_ms)

    assert ccc.main() == 0
    out_ok = capsys.readouterr().out
    assert "all venue reflections (open/filled/trade/balance/ledger/health) -> OK" in out_ok

    # Now inject the exact ADAUSDC-class stale-balance bug: health says age=55s (< 600s -> health OK!),
    # open_orders is empty at now-20s, filled has a fill at now-25s, balance is stuck at now-55s with locked=966!
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(acct_dir / "cache_balance_binance.json"),
        {"USDC": (100.0, 966.0)},
        venue="binance",
        now_s=now_s - 55.0,
        verified_ts=now_s - 55.0,
    )
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(acct_dir / "cache_order_open_binance.json"),
        {"ADAUSDC": []},
        now_ms=now_ms - 20_000,
        verified_ts=now_s - 20.0,
    )
    atomic_write_json(
        str(acct_dir / "cache_order_filled_binance.json"),
        {
            "items": {"ADAUSDC": [{"orderId": "bn-fill-99", "symbol": "ADAUSDC", "status": "FILLED", "timestamp": now_ms - 25_000}]},
            "fetchtime": {"ADAUSDC": now_ms - 2_000},
        },
    )
    streams_ok["binance"]["balances"] = now_ms - 55_000
    streams_ok["binance"]["open_orders"] = now_ms - 20_000
    _write_health_doc(health_file, now_ms=now_ms, started_ms=started_ms, venue_streams=streams_ok)

    assert ccc.main() == 1
    out_fail = capsys.readouterr().out
    assert "binance:causal_fill_to_balance" in out_fail
    assert "binance:balance_phantom_lock" in out_fail


def test_hyperliquid_ws_invalidated_fetchtime_does_not_lie_in_health_and_recovers_via_rest(tmp_path, monkeypatch):
    """After order_status/cancel invalidates HYPEUSDC fetchtime, HL WS bridge must not advance health until REST catch-up restores disk fetchtime."""
    import threading

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    tr_path = acct_dir / "cache_trade_hyperliquid.json"
    fo_path = acct_dir / "cache_order_filled_hyperliquid.json"
    health_path = acct_dir / "cache_account_health.json"

    t0 = 1_795_740_000.0
    t0_ms = int(t0 * 1000)
    fill_row = {
        "coin": "@107",
        "px": "62.06",
        "sz": "19.33",
        "side": "B",
        "time": t0_ms - 60_000,
        "oid": 617441855947,
        "tid": 734038162456393,
        "fee": "0.007732",
        "feeToken": "HYPE",
    }
    cacheaccount.CacheTradeManager.merge_venue_fills_snapshot(
        str(tr_path), {"HYPEUSDC": [fill_row]}, now_ms=t0_ms, is_snapshot=True, touch_symbols=("HYPEUSDC",), touch_create_missing=True
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path), {"HYPEUSDC": [fill_row]}, venue="hyperliquid", now_ms=t0_ms, is_snapshot=True, touch_symbols=("HYPEUSDC",), touch_create_missing=True
    )

    # Simulate HyperliquidProvider.order_status/cancel_order expiring fetchtime["HYPEUSDC"].
    cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(tr_path))
    cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(fo_path))

    health_calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        cacheaccount,
        "_record_venue_health_sync",
        lambda v, s, *, now_ms=None: health_calls.append((v, s, int(now_ms or 0))) or True,
    )

    class _DummyWs:
        _lock = threading.Lock()
        _symbols = ["HYPEUSDC"]
        _fills_ws_seeded = True
        _last_fills_touch_ms = t0_ms

    # 1. With a failing REST resync, touch_fills_cache_if_needed must NOT lie by advancing health.
    class _FailingProv:
        def __init__(self, *, token=None):
            self.token = token

        def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0):
            raise RuntimeError("transient hl error")

        @staticmethod
        def resolve_ws_user_address(addr, resolve_env=None):
            return addr or "0xabc"

    monkeypatch.setattr(MarketDataProvider, "resolve_venue_provider_class", classmethod(lambda cls, v: _FailingProv if v == "hyperliquid" else None))
    ws = _DummyWs()
    bridge = cacheaccount._HyperliquidAccountTradeBridge(ws, user_address="0xabc", fills_cache_path=str(tr_path))
    ws._fills_ws_seeded = True
    ws._last_fills_touch_ms = t0_ms
    # Intercept async spawn to run synchronously in test.
    orig_sched = bridge._schedule_invalidated_fills_resync
    monkeypatch.setattr(bridge, "_schedule_invalidated_fills_resync", lambda syms, ff, tms: orig_sched(syms, ff, tms, blocking=True))
    bridge.touch_fills_cache_if_needed(t0_ms + 20_000)
    assert health_calls == []

    # 2. With a healthy REST provider, touch_fills_cache_if_needed restores disk fetchtime & records health coherently.
    class _OkProv(_FailingProv):
        def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0):
            return [fill_row]

    bridge.hl_prov_cls = _OkProv
    t_rec_ms = t0_ms + 40_000
    monkeypatch.setattr(cacheaccount, "resolve_now_ms", lambda v=None: t_rec_ms if v is None else int(v))
    bridge.touch_fills_cache_if_needed(t_rec_ms)
    assert ("hyperliquid", "trades", t_rec_ms) in health_calls
    assert ("hyperliquid", "filled_orders", t_rec_ms) in health_calls

    _, fo_ft = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(fo_path), venue="hyperliquid")
    assert fo_ft.get("HYPEUSDC") == t_rec_ms

    # Cross-cache coherence audit confirms zero health_disk_lag!
    _write_health_doc(
        health_path,
        now_ms=t_rec_ms,
        started_ms=t0_ms - 5_000,
        venue_streams={"hyperliquid": {"filled_orders": t_rec_ms, "trades": t_rec_ms}},
    )
    viols = audit_account_cache_coherence(
        str(tmp_path),
        now_s=t_rec_ms / 1000.0,
        venues=("hyperliquid",),
        health_path=str(health_path),
        include_ledger=False,
    )
    assert viols == [], [v.format_line() for v in viols]


def test_multi_symbol_ws_invalidated_fetchtime_triggers_targeted_rest_recovery(tmp_path, monkeypatch):
    """Invalidating one symbol in a multi-symbol venue must still report missing_out and trigger targeted REST recovery."""
    import threading

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    tr_path = acct_dir / "cache_trade_hyperliquid.json"
    fo_path = acct_dir / "cache_order_filled_hyperliquid.json"
    health_path = acct_dir / "cache_account_health.json"

    t0 = 1_795_750_000.0
    t0_ms = int(t0 * 1000)
    hype_fill = {
        "coin": "@107",
        "px": "62.06",
        "sz": "19.33",
        "side": "B",
        "time": t0_ms - 60_000,
        "oid": 617441855947,
        "tid": 734038162456393,
        "fee": "0.007732",
        "feeToken": "HYPE",
    }
    sol_fill = {
        "coin": "@156",
        "px": "71.20",
        "sz": "5.0",
        "side": "B",
        "time": t0_ms - 50_000,
        "oid": 617441900001,
        "tid": 734038162456999,
        "fee": "0.002",
        "feeToken": "SOL",
    }
    cacheaccount.CacheTradeManager.merge_venue_fills_snapshot(
        str(tr_path),
        {"HYPEUSDC": [hype_fill], "SOLUSDC": [sol_fill]},
        now_ms=t0_ms,
        is_snapshot=True,
        touch_symbols=("HYPEUSDC", "SOLUSDC"),
        touch_create_missing=True,
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path),
        {"HYPEUSDC": [hype_fill], "SOLUSDC": [sol_fill]},
        venue="hyperliquid",
        now_ms=t0_ms,
        is_snapshot=True,
        touch_symbols=("HYPEUSDC", "SOLUSDC"),
        touch_create_missing=True,
    )

    # Invalidate only HYPEUSDC while SOLUSDC remains fresh.
    cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(tr_path))
    cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(fo_path))

    # Direct touch reports True (for SOLUSDC) AND populates missing_out=["HYPEUSDC"].
    tr_missing: list[str] = []
    fo_missing: list[str] = []
    assert cacheaccount.CacheTradeManager.touch_venue_fills_snapshot(
        str(tr_path), t0_ms + 10_000, symbols=("HYPEUSDC", "SOLUSDC"), create_missing=False, missing_out=tr_missing
    ) is True
    assert tr_missing == ["HYPEUSDC"]
    assert cacheaccount.CacheOrderFilledManager.touch_venue_filled_orders_snapshot(
        str(fo_path), t0_ms + 10_000, symbols=("HYPEUSDC", "SOLUSDC"), create_missing=False, missing_out=fo_missing
    ) is True
    assert fo_missing == ["HYPEUSDC"]

    health_calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        cacheaccount,
        "_record_venue_health_sync",
        lambda v, s, *, now_ms=None: health_calls.append((v, s, int(now_ms or 0))) or True,
    )
    fetched_syms: list[str] = []

    class _DummyWs:
        _lock = threading.Lock()
        _symbols = ["HYPEUSDC", "SOLUSDC"]
        _fills_ws_seeded = True
        _last_fills_touch_ms = t0_ms

    class _TrackedProv:
        def __init__(self, *, token=None):
            self.token = token

        def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0):
            fetched_syms.append(symbol)
            return [hype_fill] if symbol == "HYPEUSDC" else [sol_fill]

        @staticmethod
        def resolve_ws_user_address(addr, resolve_env=None):
            return addr or "0xabc"

    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: _TrackedProv if v == "hyperliquid" else None),
    )
    ws = _DummyWs()
    bridge = cacheaccount._HyperliquidAccountTradeBridge(ws, user_address="0xabc", fills_cache_path=str(tr_path))
    ws._fills_ws_seeded = True
    ws._last_fills_touch_ms = t0_ms
    orig_sched = bridge._schedule_invalidated_fills_resync
    monkeypatch.setattr(
        bridge, "_schedule_invalidated_fills_resync", lambda syms, ff, tms: orig_sched(syms, ff, tms, blocking=True)
    )

    t_rec_ms = t0_ms + 25_000
    monkeypatch.setattr(cacheaccount, "resolve_now_ms", lambda v=None: t_rec_ms if v is None else int(v))
    bridge.touch_fills_cache_if_needed(t_rec_ms)

    # Only the missing symbol HYPEUSDC was fetched via REST (SOLUSDC was already fresh).
    assert fetched_syms == ["HYPEUSDC"]
    assert ("hyperliquid", "trades", t_rec_ms) in health_calls
    assert ("hyperliquid", "filled_orders", t_rec_ms) in health_calls

    _, fo_ft = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(fo_path), venue="hyperliquid")
    assert fo_ft.get("HYPEUSDC") == t_rec_ms
    assert fo_ft.get("SOLUSDC") == t_rec_ms

    _write_health_doc(
        health_path,
        now_ms=t_rec_ms,
        started_ms=t0_ms - 5_000,
        venue_streams={"hyperliquid": {"filled_orders": t_rec_ms, "trades": t_rec_ms}},
    )
    viols = audit_account_cache_coherence(
        str(tmp_path),
        now_s=t_rec_ms / 1000.0,
        venues=("hyperliquid",),
        health_path=str(health_path),
        include_ledger=False,
    )
    assert viols == [], [v.format_line() for v in viols]


def test_cross_cache_per_symbol_fetchtime_invariants_across_venues():
    """Pure invariant check: per-symbol filled_orders fetchtime detects single-symbol invalidation/lag without false-flagging retired symbols or Binance WS-vs-REST skew."""
    t_now = 1_795_755_000.0

    # 1. Kraken multi-symbol: HYPEUSD missing fetchtime while SPCXxUSD & ANTxUSD are fresh -> caught despite max(ft)==fresh; retired TAOUSD ignored.
    ev_krk_missing = VenueCoherenceEvidence(
        venue="kraken",
        filled_file_exists=True,
        filled_items={"HYPEUSD": [], "SPCXxUSD": [], "ANTxUSD": [], "TAOUSD": []},
        filled_sym_ft_ts={"SPCXxUSD": t_now - 2.0, "ANTxUSD": t_now - 2.0, "TAOUSD": t_now - 90_000.0},
        filled_latest_ft_ts=t_now - 2.0,
        health_sync_ts_by_stream={"filled_orders": t_now - 2.0},
        configured_symbols=frozenset({"HYPEUSD", "SPCXxUSD", "ANTxUSD"}),
    )
    v_miss = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_krk_missing}, health_present=True),
        sync_grace_sec=15.0,
    )
    assert [v.rule for v in v_miss] == ["health_symbol_fetchtime_missing"]
    assert "HYPEUSD" in v_miss[0].detail

    # 2. Kraken multi-symbol: HYPEUSD lags health by 30s (> 15s sync_grace) -> caught; retired TAOUSD still ignored.
    ev_krk_lag = VenueCoherenceEvidence(
        venue="kraken",
        filled_file_exists=True,
        filled_items={"HYPEUSD": [], "SPCXxUSD": [], "ANTxUSD": [], "TAOUSD": []},
        filled_sym_ft_ts={
            "HYPEUSD": t_now - 32.0,
            "SPCXxUSD": t_now - 2.0,
            "ANTxUSD": t_now - 2.0,
            "TAOUSD": t_now - 90_000.0,
        },
        filled_latest_ft_ts=t_now - 2.0,
        health_sync_ts_by_stream={"filled_orders": t_now - 2.0},
        configured_symbols=frozenset({"HYPEUSD", "SPCXxUSD", "ANTxUSD"}),
    )
    v_lag = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_krk_lag}, health_present=True),
        sync_grace_sec=15.0,
    )
    assert [v.rule for v in v_lag] == ["health_symbol_fetchtime_lag"]
    assert "HYPEUSD" in v_lag[0].detail

    # 3. Kraken healthy: all configured symbols fresh, retired TAOUSD ancient -> 0 violations.
    ev_krk_ok = VenueCoherenceEvidence(
        venue="kraken",
        filled_file_exists=True,
        filled_items={"HYPEUSD": [], "SPCXxUSD": [], "ANTxUSD": [], "TAOUSD": []},
        filled_sym_ft_ts={
            "HYPEUSD": t_now - 2.0,
            "SPCXxUSD": t_now - 2.0,
            "ANTxUSD": t_now - 2.0,
            "TAOUSD": t_now - 90_000.0,
        },
        filled_latest_ft_ts=t_now - 2.0,
        health_sync_ts_by_stream={"filled_orders": t_now - 2.0},
        configured_symbols=frozenset({"HYPEUSD", "SPCXxUSD", "ANTxUSD"}),
    )
    assert (
        evaluate_account_cache_coherence(
            AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_krk_ok}, health_present=True),
            sync_grace_sec=15.0,
        )
        == []
    )

    # 4. Binance (primary stream venue): 22s skew after an ADAUSDC WS fill is inside ORDER_SYNC_INTERVAL_SEC + sync_grace (39s) -> OK;
    #    50s skew exceeds bound -> health_symbol_fetchtime_lag; missing fetchtime -> health_symbol_fetchtime_missing.
    ev_bn_skew_ok = VenueCoherenceEvidence(
        venue="binance",
        filled_file_exists=True,
        filled_items={"BTCUSDC": [], "ADAUSDC": [], "BNBUSDC": []},
        filled_sym_ft_ts={"BTCUSDC": t_now - 24.0, "ADAUSDC": t_now - 2.0, "BNBUSDC": t_now - 500_000.0},
        filled_latest_ft_ts=t_now - 2.0,
        health_sync_ts_by_stream={"filled_orders": t_now - 2.0},
        configured_symbols=frozenset({"BTCUSDC", "ADAUSDC"}),
    )
    assert (
        evaluate_account_cache_coherence(
            AccountCacheCoherenceSnapshot(now_s=t_now, venues={"binance": ev_bn_skew_ok}, health_present=True),
            sync_grace_sec=15.0,
        )
        == []
    )

    ev_bn_bad = VenueCoherenceEvidence(
        venue="binance",
        filled_file_exists=True,
        filled_items={"BTCUSDC": [], "ADAUSDC": [], "SOLUSDC": []},
        filled_sym_ft_ts={"BTCUSDC": t_now - 55.0, "ADAUSDC": t_now - 2.0},
        filled_latest_ft_ts=t_now - 2.0,
        health_sync_ts_by_stream={"filled_orders": t_now - 2.0},
        configured_symbols=frozenset({"BTCUSDC", "ADAUSDC", "SOLUSDC"}),
    )
    v_bn = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=t_now, venues={"binance": ev_bn_bad}, health_present=True),
        sync_grace_sec=15.0,
    )
    assert sorted((v.rule, v.detail.split()[0]) for v in v_bn) == [
        ("health_symbol_fetchtime_lag", "BTCUSDC"),
        ("health_symbol_fetchtime_missing", "SOLUSDC"),
    ]


def test_hyperliquid_dependent_crosscheck_alo_resting_perp_fill_backoff_recovery_and_cost_basis(tmp_path, monkeypatch):
    """Dependent cross-check across HyperliquidProvider, _HyperliquidAccountTradeBridge, position_cost_basis, and coherence audit."""
    import threading
    from types import SimpleNamespace
    import providers.quantity as providers_qty
    from providers.hyperliquid_provider import HyperliquidProvider

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    tr_path = acct_dir / "cache_trade_hyperliquid.json"
    fo_path = acct_dir / "cache_order_filled_hyperliquid.json"
    oo_path = acct_dir / "cache_order_open_hyperliquid.json"
    bal_path = acct_dir / "cache_balance_hyperliquid.json"
    health_path = acct_dir / "cache_account_health.json"

    t0 = 1_795_760_000.0
    t0_ms = int(t0 * 1000)
    fill_1 = {
        "coin": "@107",
        "px": "62.06",
        "sz": "10.0",
        "side": "B",
        "time": t0_ms - 60_000,
        "oid": 617441855947,
        "tid": 734038162456393,
        "fee": "0.004",
        "feeToken": "HYPE",
    }
    fill_2 = {
        "coin": "@107",
        "px": "64.06",
        "sz": "10.0",
        "side": "B",
        "time": t0_ms + 30_000,
        "oid": 617441999999,
        "tid": 734038162459999,
        "fee": "0.004",
        "feeToken": "HYPE",
    }

    cacheaccount.CacheTradeManager.merge_venue_fills_snapshot(
        str(tr_path), {"HYPEUSDC": [fill_1]}, now_ms=t0_ms, is_snapshot=True, touch_symbols=("HYPEUSDC",), touch_create_missing=True
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path), {"HYPEUSDC": [fill_1]}, venue="hyperliquid", now_ms=t0_ms, is_snapshot=True, touch_symbols=("HYPEUSDC",), touch_create_missing=True
    )
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path), {"HYPEUSDC": []}, now_ms=t0_ms, verified_ts=t0
    )
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path), {"USDC": (1000.0, 0.0), "HYPE": (9.996, 0.0)}, venue="hyperliquid", now_s=t0, verified_ts=t0
    )

    now_ref = [t0]
    monkeypatch.setattr(cacheaccount, "_now_time", lambda: now_ref[0])
    monkeypatch.setattr(cacheaccount, "resolve_now_ms", lambda v=None: int(now_ref[0] * 1000) if v is None else int(v))
    monkeypatch.setattr(providers_qty.time, "time", lambda: now_ref[0])

    hl = HyperliquidProvider(token="HYPE", live_enabled=True)
    hl._ssot_fills_cache_path = str(tr_path)
    monkeypatch.setattr(hl, "_filled_orders_cache_file", lambda: str(fo_path))
    monkeypatch.setattr(hl, "_pair", lambda: "@107")
    monkeypatch.setattr(hl, "require_account_cache_fresh_for_submit", lambda *_a, **_kw: None)
    monkeypatch.setattr(hl, "round_price", lambda _s, px: round(float(px), 2))
    monkeypatch.setattr(hl, "round_quantity", lambda _s, q: round(float(q), 2))

    # 1. Baseline: position_cost_basis is fresh & exact (net 9.996 HYPE after 0.004 base fee).
    assert hl.position_cost_basis("HYPEUSDC", 9.996, 86400.0) == pytest.approx(62.06 / 0.9996, rel=1e-3)

    # 2. Placing a post-only Alo resting order MUST preserve fetchtime and keep position_cost_basis available!
    fake_signer = SimpleNamespace(
        sz_decimals=lambda _t: 2,
        spot_order=lambda pair, is_buy, qty, px, **kw: (True, 617441888888, "resting"),
    )
    monkeypatch.setattr(hl, "_signer", lambda: fake_signer)
    res_alo = hl.place_order("HYPEUSDC", "BUY", 61.50, 5.0, post_only=True)
    assert res_alo and res_alo["orderId"] == 617441888888
    assert hl.position_cost_basis("HYPEUSDC", 9.996, 86400.0) == pytest.approx(62.06 / 0.9996, rel=1e-3)
    _, fo_ft = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(fo_path), venue="hyperliquid")
    assert fo_ft.get("HYPEUSDC") == t0_ms

    # 3. Placing a crossing GTC order DOES invalidate fetchtime (conservative).
    res_gtc = hl.place_order("HYPEUSDC", "BUY", 64.06, 10.0, post_only=False)
    assert res_gtc and res_gtc["orderId"] == 617441888888
    _, fo_ft_inv = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(fo_path), venue="hyperliquid")
    assert "HYPEUSDC" not in fo_ft_inv

    # 4. While HYPEUSDC is invalidated, a perp fill on the co-mingled wallet (coin="ETH") arrives over WS:
    #    must NOT fabricate spot health and must NOT reset _last_fills_touch_ms!
    health_calls: list[tuple[str, str, int]] = []
    refresh_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        cacheaccount,
        "_record_venue_health_sync",
        lambda v, s, *, now_ms=None: health_calls.append((v, s, int(now_ms or 0))) or True,
    )
    monkeypatch.setattr(
        cacheaccount,
        "request_venue_account_refresh",
        lambda v, *, include_open_orders=False, blocking=False: refresh_calls.append((v, include_open_orders)) or True,
    )

    rest_state = {"calls": 0, "fail": True}

    class _DynamicHLProv(HyperliquidProvider):
        def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0, *, client=None):
            rest_state["calls"] += 1
            if rest_state["fail"]:
                raise RuntimeError("HTTP 429 rate limit")
            return [fill_1, fill_2]

    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: _DynamicHLProv if v == "hyperliquid" else None),
    )

    class _DummyWs:
        _lock = threading.Lock()
        _symbols = ["HYPEUSDC"]
        _fills_ws_seeded = True
        _last_fills_touch_ms = t0_ms

        @staticmethod
        def _candidate_mids_keys(s):
            return ("@107", "HYPE/USDC") if s == "HYPEUSDC" else ()

    ws = _DummyWs()
    bridge = cacheaccount._HyperliquidAccountTradeBridge(ws, user_address="0xabc", fills_cache_path=str(tr_path))
    ws._fills_ws_seeded = True
    ws._last_fills_touch_ms = t0_ms
    orig_sched = bridge._schedule_invalidated_fills_resync
    monkeypatch.setattr(
        bridge, "_schedule_invalidated_fills_resync", lambda syms, ff, tms: orig_sched(syms, ff, tms, blocking=True)
    )

    # Perp fill arrives at t0 + 10s -> completely ignored for spot health & touch timer.
    now_ref[0] = t0 + 10.0
    bridge.handle_user_fills({"isSnapshot": False, "fills": [{"coin": "ETH", "oid": 999, "sz": "1.0", "px": "2000.0"}]}, ts_ms=t0_ms + 10_000)
    assert health_calls == []
    assert ws._last_fills_touch_ms == t0_ms

    # 5. At t0 + 20s, WS tick triggers REST resync -> 429 engages 30s backoff; health stays silent!
    now_ref[0] = t0 + 20.0
    bridge.touch_fills_cache_if_needed(t0_ms + 20_000)
    assert rest_state["calls"] == 1
    assert health_calls == []
    assert bridge._rest_backoff_sec == pytest.approx(30.0)

    # At t0 + 36s (inside 30s backoff window), second tick must NOT hammer REST!
    now_ref[0] = t0 + 36.0
    bridge.touch_fills_cache_if_needed(t0_ms + 36_000)
    assert rest_state["calls"] == 1
    assert health_calls == []

    # 6. At t0 + 55s (backoff expired), REST recovers fill_2 -> triggers balance/open-order refresh, restores fetchtime, resets backoff,
    #    updates position_cost_basis to blended 63.06, and passes full coherence audit!
    rest_state["fail"] = False
    now_ref[0] = t0 + 55.0
    t_rec_ms = t0_ms + 55_000
    bridge.touch_fills_cache_if_needed(t_rec_ms)
    assert rest_state["calls"] == 2
    assert bridge._rest_backoff_sec == 0.0
    assert ("hyperliquid", True) in refresh_calls
    assert ("hyperliquid", "filled_orders", t_rec_ms) in health_calls
    assert ("hyperliquid", "trades", t_rec_ms) in health_calls
    assert hl.position_cost_basis("HYPEUSDC", 19.992, 86400.0) == pytest.approx(63.06 / 0.9996, rel=1e-3)

    # Simulate the triggered balance/open-orders refresh settling at t0 + 56s.
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path), {"USDC": (359.4, 0.0), "HYPE": (19.992, 0.0)}, venue="hyperliquid", now_s=t0 + 56.0, verified_ts=t0 + 56.0
    )
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path), {"HYPEUSDC": []}, now_ms=t0_ms + 56_000, verified_ts=t0 + 56.0
    )
    _write_health_doc(
        health_path,
        now_ms=t0_ms + 56_000,
        started_ms=t0_ms - 5_000,
        venue_streams={
            "hyperliquid": {
                "balances": t0_ms + 56_000,
                "open_orders": t0_ms + 56_000,
                "filled_orders": t_rec_ms,
                "trades": t_rec_ms,
            }
        },
    )

    # 7. Late order_status poll for already-persisted fill_2 preserves fetchtime & cost basis and keeps coherence clean!
    fake_cli = SimpleNamespace(
        info=SimpleNamespace(
            query_order_by_oid=lambda _addr, _oid: {
                "status": "order",
                "order": {"status": "filled", "order": {"coin": "@107", "origSz": "10.0", "sz": "0.0"}},
            },
            user_fills=lambda _addr: [fill_1, fill_2],
        )
    )
    monkeypatch.setattr(hl, "_require_hl_client_and_pair", lambda *_a, **_kw: (fake_cli, "@107", "0xabc"))
    st_late = hl.order_status("HYPEUSDC", "617441999999")
    assert st_late.status == "closed" and st_late.filled_qty == pytest.approx(10.0)
    assert hl.position_cost_basis("HYPEUSDC", 19.992, 86400.0) == pytest.approx(63.06 / 0.9996, rel=1e-3)

    viols = audit_account_cache_coherence(
        str(tmp_path),
        now_s=t0 + 57.0,
        venues=("hyperliquid",),
        health_path=str(health_path),
        include_ledger=False,
    )
    assert viols == [], [v.format_line() for v in viols]


def test_kraken_multi_symbol_ws_fill_preserves_invalidation_and_cli_cross_cache_catches_drift(tmp_path, capsys):
    """Dependent cross-check: Kraken WS fill on SPCXxUSD must report missing HYPEUSD via missing_out and --check-cross-cache catches false health."""
    from botcore import resolve_now_ts
    from verify_tools import audit_prompt_invariants as audit_cli

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    tr_path = acct_dir / "cache_trade_kraken.json"
    fo_path = acct_dir / "cache_order_filled_kraken.json"
    oo_path = acct_dir / "cache_order_open_kraken.json"
    bal_path = acct_dir / "cache_balance_kraken.json"
    health_path = acct_dir / "cache_account_health.json"

    t0 = resolve_now_ts() - 60.0
    t0_ms = int(t0 * 1000)
    f_hype = {"symbol": "HYPEUSD", "id": "T-H1", "orderId": "O-H1", "side": "BUY", "price": 62.0, "qty": 2.0, "quantity": 2.0, "time": t0_ms - 50_000}
    f_spcx1 = {"symbol": "SPCXxUSD", "id": "T-S1", "orderId": "O-S1", "side": "BUY", "price": 125.0, "qty": 1.0, "quantity": 1.0, "time": t0_ms - 40_000}
    f_tao_retired = {"symbol": "TAOUSD", "id": "T-T0", "orderId": "O-T0", "side": "BUY", "price": 180.0, "qty": 1.0, "quantity": 1.0, "time": t0_ms - 900_000}

    pairs = ("HYPEUSD", "SPCXxUSD")
    cacheaccount.CacheTradeManager._merge_venue_fills_snapshot_locked(
        str(tr_path),
        {"HYPEUSD": [f_hype], "SPCXxUSD": [f_spcx1], "TAOUSD": [f_tao_retired]},
        now_ms=t0_ms,
        is_snapshot=False,
        touch_symbols=(*pairs, "TAOUSD"),
        touch_create_missing=True,
        sanitize_venue="kraken",
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path),
        {"HYPEUSD": [f_hype], "SPCXxUSD": [f_spcx1], "TAOUSD": [f_tao_retired]},
        venue="kraken",
        now_ms=t0_ms,
        is_snapshot=False,
        touch_symbols=(*pairs, "TAOUSD"),
        touch_create_missing=True,
    )
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path), {"HYPEUSD": [], "SPCXxUSD": []}, now_ms=t0_ms + 20_000, verified_ts=t0 + 20.0
    )
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path), {"USD": (500.0, 0.0), "HYPE": (2.0, 0.0), "SPCXx": (2.0, 0.0)}, venue="kraken", now_s=t0 + 20.0, verified_ts=t0 + 20.0
    )

    # Invalidate ONLY HYPEUSD; then a live WS fill arrives for SPCXxUSD 20s later (< old 60s throttle).
    cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSD", str(tr_path))
    cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSD", str(fo_path))

    t1_ms = t0_ms + 20_000
    f_spcx2 = {"symbol": "SPCXxUSD", "id": "T-S2", "orderId": "O-S2", "side": "BUY", "price": 126.0, "qty": 1.0, "quantity": 1.0, "time": t1_ms - 1000}
    tr_missing: list[str] = []
    fo_missing: list[str] = []
    cacheaccount.CacheTradeManager._merge_venue_fills_snapshot_locked(
        str(tr_path),
        {"SPCXxUSD": [f_spcx2]},
        now_ms=t1_ms,
        is_snapshot=False,
        touch_symbols=pairs,
        touch_create_missing=False,
        sanitize_venue="kraken",
        missing_out=tr_missing,
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path),
        {"SPCXxUSD": [f_spcx2]},
        venue="kraken",
        now_ms=t1_ms,
        is_snapshot=False,
        touch_symbols=pairs,
        touch_create_missing=False,
        missing_out=fo_missing,
    )
    assert tr_missing == ["HYPEUSD"]
    assert fo_missing == ["HYPEUSD"]
    _, fo_ft_after = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(fo_path), venue="kraken")
    assert fo_ft_after.get("SPCXxUSD") == t1_ms
    assert "HYPEUSD" not in fo_ft_after

    # If health had falsely advanced while HYPEUSD was missing, CLI --check-cross-cache catches kraken:health_symbol_fetchtime_missing!
    _write_health_doc(
        health_path,
        now_ms=t1_ms,
        started_ms=t0_ms - 5_000,
        venue_streams={
            "kraken": {
                "balances": t1_ms,
                "open_orders": t1_ms,
                "filled_orders": t1_ms,
                "trades": t1_ms,
            }
        },
    )
    rc_bad = audit_cli.main(["--check-cross-cache", "--scope", "crypto", "--cache-dir", str(tmp_path)])
    err_bad = capsys.readouterr().err
    assert rc_bad == 1
    assert "CROSS_CACHE_COHERENCE" in err_bad
    assert "kraken:health_symbol_fetchtime_missing" in err_bad
    assert "HYPEUSD" in err_bad

    # After REST catch-up restores ONLY filled_orders, missing trades fetchtime on HYPEUSD must STILL fail coherence.
    t2_ms = t0_ms + 30_000
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path),
        {"HYPEUSD": [f_hype], "SPCXxUSD": [f_spcx1, f_spcx2]},
        venue="kraken",
        now_ms=t2_ms,
        is_snapshot=False,
        touch_symbols=pairs,
        touch_create_missing=True,
    )
    _write_health_doc(
        health_path,
        now_ms=t2_ms,
        started_ms=t0_ms - 5_000,
        venue_streams={
            "kraken": {
                "balances": t2_ms,
                "open_orders": t2_ms,
                "filled_orders": t2_ms,
                "trades": t2_ms,
            }
        },
    )
    cacheaccount.CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path), {"USD": (500.0, 0.0), "HYPE": (2.0, 0.0), "SPCXx": (2.0, 0.0)}, venue="kraken", now_s=t0 + 30.0, verified_ts=t0 + 30.0
    )
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path), {"HYPEUSD": [], "SPCXxUSD": []}, now_ms=t2_ms, verified_ts=t0 + 30.0
    )
    rc_still_bad = audit_cli.main(["--check-cross-cache", "--scope", "crypto", "--cache-dir", str(tmp_path)])
    err_still_bad = capsys.readouterr().err
    assert rc_still_bad == 1
    assert "kraken:health_symbol_fetchtime_missing" in err_still_bad
    assert "trades items" in err_still_bad
    assert "HYPEUSD" in err_still_bad

    # After REST catch-up restores cache_trade_kraken.json as well, coherence passes cleanly even though retired TAOUSD still has old t0_ms fetchtime.
    cacheaccount.CacheTradeManager._merge_venue_fills_snapshot_locked(
        str(tr_path),
        {"HYPEUSD": [f_hype], "SPCXxUSD": [f_spcx1, f_spcx2]},
        now_ms=t2_ms,
        is_snapshot=False,
        touch_symbols=pairs,
        touch_create_missing=True,
        sanitize_venue="kraken",
    )
    rc_ok = audit_cli.main(["--check-cross-cache", "--scope", "crypto", "--cache-dir", str(tmp_path)])
    assert rc_ok == 0


def test_touch_venue_ws_fetchtimes_does_not_resurrect_nonpositive_fetchtime():
    """Zero or null fetchtime entries must be reported in missing_out and never stamped fresh by WS heartbeats."""
    t0_ms = 1_795_750_000_000
    cache_obj = {
        "items": {"HYPEUSDC": [], "SOLUSDC": []},
        "fetchtime": {"HYPEUSDC": 0, "SOLUSDC": t0_ms - 5_000},
    }
    missing: list[str] = []
    touched = cacheaccount.CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
        cache_obj,
        t0_ms,
        symbols=("HYPEUSDC", "SOLUSDC"),
        create_missing=False,
        missing_out=missing,
    )
    assert touched is True
    assert missing == ["HYPEUSDC"]
    assert cache_obj["fetchtime"]["HYPEUSDC"] == 0
    assert cache_obj["fetchtime"]["SOLUSDC"] == t0_ms


def test_hyperliquid_rest_resync_triggers_refresh_on_same_oid_partial_fill_upgrade(tmp_path, monkeypatch):
    """REST recovery must trigger balance/open-order refresh when an existing orderId receives an additional partial fill slice, and skip refresh when fills are unchanged."""
    import threading

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    tr_path = acct_dir / "cache_trade_hyperliquid.json"
    fo_path = acct_dir / "cache_order_filled_hyperliquid.json"

    t0_ms = 1_795_750_000_000
    slice_1 = {
        "coin": "@107",
        "px": "62.00",
        "sz": "5.0",
        "side": "B",
        "time": t0_ms - 30_000,
        "oid": 617441855947,
        "tid": 734038162456301,
        "fee": "0.002",
        "feeToken": "HYPE",
    }
    slice_2 = {
        "coin": "@107",
        "px": "62.10",
        "sz": "14.33",
        "side": "B",
        "time": t0_ms - 5_000,
        "oid": 617441855947,
        "tid": 734038162456302,
        "fee": "0.005732",
        "feeToken": "HYPE",
    }

    cacheaccount.CacheTradeManager.merge_venue_fills_snapshot(
        str(tr_path), {"HYPEUSDC": [slice_1]}, now_ms=t0_ms, is_snapshot=True, touch_symbols=("HYPEUSDC",), touch_create_missing=True
    )
    cacheaccount.CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        str(fo_path), {"HYPEUSDC": [slice_1]}, venue="hyperliquid", now_ms=t0_ms, is_snapshot=True, touch_symbols=("HYPEUSDC",), touch_create_missing=True
    )

    refresh_calls: list[tuple[str, bool]] = []
    health_calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        cacheaccount,
        "request_venue_account_refresh",
        lambda v, *, include_open_orders=False, blocking=False: refresh_calls.append((v, include_open_orders)) or True,
    )
    monkeypatch.setattr(
        cacheaccount,
        "_record_venue_health_sync",
        lambda v, s, *, now_ms=None: health_calls.append((v, s, int(now_ms or 0))) or True,
    )

    remote_rows = [slice_1]

    class _FakeHLProv:
        def __init__(self, *, token=None):
            self.token = token

        def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0):
            return list(remote_rows)

        @staticmethod
        def resolve_ws_user_address(addr, resolve_env=None):
            return addr or "0xabc"

    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: _FakeHLProv if v == "hyperliquid" else None),
    )

    class _DummyWs:
        _lock = threading.Lock()
        _symbols = ["HYPEUSDC"]
        _fills_ws_seeded = True
        _last_fills_touch_ms = t0_ms

    ws = _DummyWs()
    bridge = cacheaccount._HyperliquidAccountTradeBridge(ws, user_address="0xabc", fills_cache_path=str(tr_path))

    # 1. Unchanged fills (only slice_1): no spurious balance/open-order refresh, deterministic sync_ms == ts_ms.
    cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(tr_path))
    cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(fo_path))
    t1_ms = t0_ms + 20_000
    assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(tr_path), t1_ms, blocking=True) is True
    assert refresh_calls == []
    assert ("hyperliquid", "filled_orders", t1_ms) in health_calls

    # 2. Same orderId upgraded with slice_2 (order row count stays 1, but qty 5.0 -> 19.33 & _fillIds 1 -> 2):
    #    MUST trigger request_venue_account_refresh("hyperliquid", include_open_orders=True)!
    remote_rows.append(slice_2)
    cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(tr_path))
    cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSDC", str(fo_path))
    t2_ms = t0_ms + 40_000
    assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(tr_path), t2_ms, blocking=True) is True
    assert refresh_calls == [("hyperliquid", True)]
    fo_items, fo_ft = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(str(fo_path), venue="hyperliquid")
    assert len(fo_items["HYPEUSDC"]) == 1
    assert fo_items["HYPEUSDC"][0]["quantity"] == pytest.approx(19.33)
    assert fo_ft["HYPEUSDC"] == t2_ms


def test_t212_and_kraken_rest_sync_trigger_account_refresh_on_partial_fill_upgrades(tmp_path, monkeypatch):
    """T212 REST poll, Kraken REST poll, and Kraken WS REST catch-up must trigger balance/open-order refresh on same-orderId partial-fill upgrades."""
    import sys
    import threading
    import types

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)

    t0_ms = 1_795_760_000_000
    now_ref = [t0_ms]
    refresh_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(cacheaccount, "resolve_now_ms", lambda v=None: now_ref[0] if v is None else int(v))
    monkeypatch.setattr(
        cacheaccount,
        "request_venue_account_refresh",
        lambda v, *, include_open_orders=False, blocking=False: refresh_calls.append((v, include_open_orders)) or True,
    )
    monkeypatch.setattr(cacheaccount, "_record_venue_health_sync", lambda *_a, **_kw: True)

    # 1. Trading 212 REST poll loop: same-orderId partial-fill upgrade (0.4 -> 1.0, row count stays 1).
    t212_batches = [
        {"NVDA": [{"symbol": "NVDA", "orderId": "T212-100", "side": "BUY", "price": 120.0, "qty": 0.4, "quantity": 0.4, "status": "PARTIALLY_FILLED", "time": t0_ms - 10_000}]},
        {"NVDA": [{"symbol": "NVDA", "orderId": "T212-100", "side": "BUY", "price": 120.0, "qty": 0.4, "quantity": 0.4, "status": "PARTIALLY_FILLED", "time": t0_ms - 10_000}]},
        {"NVDA": [{"symbol": "NVDA", "orderId": "T212-100", "side": "BUY", "price": 120.5, "qty": 1.0, "quantity": 1.0, "status": "FILLED", "time": t0_ms + 5_000}]},
    ]
    t212_idx = [0]

    class _FakeT212Prov:
        @staticmethod
        def fetch_history_fills_by_symbol(_client, *, symbols, publish_shared_cache=True):
            i = t212_idx[0]
            t212_idx[0] += 1
            return t212_batches[i]

    def _run_three_polls(*, sync_once_fn, **_kw):
        for step in range(3):
            now_ref[0] = t0_ms + step * 30_000
            sync_once_fn()

    monkeypatch.setattr(cacheaccount, "_run_account_trade_poll_loop", _run_three_polls)
    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: _FakeT212Prov if v == "t212" else None),
    )
    cacheaccount.run_t212_trade_poll_loop(
        object(),
        cache_file=str(acct_dir / "cache_trade_t212.json"),
        symbols=("NVDA",),
        poll_interval=60.0,
        backoff_init=10.0,
        backoff_max=120.0,
    )
    assert refresh_calls == [("t212", True)]

    # 2. Kraken REST poll loop: same-orderId partial-fill slice upgrade (1.0 -> 2.5, filled-order row count stays 1).
    refresh_calls.clear()
    k_slice_1 = {"symbol": "HYPEUSD", "id": "TK-1", "orderId": "OK-1", "side": "BUY", "price": 62.0, "qty": 1.0, "quantity": 1.0, "time": t0_ms - 20_000}
    k_slice_2 = {"symbol": "HYPEUSD", "id": "TK-2", "orderId": "OK-1", "side": "BUY", "price": 62.5, "qty": 1.5, "quantity": 1.5, "time": t0_ms + 10_000}
    kr_batches = [
        {"HYPEUSD": [k_slice_1]},
        {"HYPEUSD": [k_slice_1]},
        {"HYPEUSD": [k_slice_1, k_slice_2]},
    ]
    kr_idx = [0]

    class _FakeKrakenProv:
        PRIVATE_WS_URL = "wss://example.invalid"

        @staticmethod
        def fetch_ledger_fills_by_symbol(_client, *, pairs, lookback_sec):
            i = kr_idx[0]
            kr_idx[0] += 1
            return kr_batches[i]

        @staticmethod
        def build_ws_own_trades_subscription(_client):
            return "{}"

        @staticmethod
        def extract_ws_own_trades(_data, *, pairs):
            return None

        @staticmethod
        def classify_ws_control_event(_data):
            return ("keepalive", "")

    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: _FakeKrakenProv if v == "kraken" else None),
    )
    cacheaccount.run_kraken_trade_poll_loop(
        object(),
        cache_file=str(acct_dir / "cache_trade_kraken_poll.json"),
        pairs=("HYPEUSD",),
        poll_interval=30.0,
        backoff_init=5.0,
        backoff_max=60.0,
        lookback_sec=3600.0,
    )
    assert refresh_calls == [("kraken", True)]

    # 3. Kraken WS loop _merge_rest recovery: no refresh on startup seed or unchanged recovery; triggers on same-orderId slice upgrade.
    refresh_calls.clear()
    k_slice_3 = {"symbol": "HYPEUSD", "id": "TK-3", "orderId": "OK-1", "side": "BUY", "price": 63.0, "qty": 0.5, "quantity": 0.5, "time": t0_ms + 60_000}
    kr_batches[:] = [
        {"HYPEUSD": [k_slice_1]},
        {"HYPEUSD": [k_slice_1]},
        {"HYPEUSD": [k_slice_1, k_slice_3]},
    ]
    kr_idx[0] = 0
    ws_tr_path = acct_dir / "cache_trade_kraken_ws.json"
    ws_fo_path = acct_dir / "cache_order_filled_kraken_ws.json"
    monkeypatch.setattr(cacheaccount, "_sibling_order_filled_path", lambda _p, _v: str(ws_fo_path))

    spawned_threads: list[threading.Thread] = []
    real_thread = threading.Thread

    class _SyncThread:
        def __init__(self, *, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(cacheaccount.threading, "Thread", _SyncThread)
    stop_ev = threading.Event()

    class _FakeWSApp:
        def __init__(self, _url, *, on_open, on_message, on_error):
            self._on_open = on_open
            self._on_message = on_message
            self.keep_running = True

        def send(self, _payload):
            pass

        def run_forever(self, **_kw):
            self._on_open(self)
            # Invalidate HYPEUSD so keepalive triggers _merge_rest with unchanged fills -> no refresh.
            cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSD", str(ws_tr_path))
            cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSD", str(ws_fo_path))
            now_ref[0] = t0_ms + 20_000
            self._on_message(self, '{"event":"heartbeat"}')
            assert refresh_calls == []
            # Invalidate again and return same-orderId partial-fill upgrade k_slice_3 -> MUST trigger refresh.
            cacheaccount.CacheTradeManager.invalidate_venue_fills_snapshot("HYPEUSD", str(ws_tr_path))
            cacheaccount.CacheOrderFilledManager.invalidate_venue_fills_snapshot("HYPEUSD", str(ws_fo_path))
            now_ref[0] = t0_ms + 40_000
            self._on_message(self, '{"event":"heartbeat"}')
            stop_ev.set()

    fake_ws_mod = types.SimpleNamespace(WebSocketApp=_FakeWSApp)
    monkeypatch.setitem(sys.modules, "websocket", fake_ws_mod)
    now_ref[0] = t0_ms
    cacheaccount.run_kraken_trade_ws_loop(
        object(),
        cache_file=str(ws_tr_path),
        pairs=("HYPEUSD",),
        poll_interval=30.0,
        backoff_init=5.0,
        backoff_max=60.0,
        lookback_sec=3600.0,
        stop_event=stop_ev,
    )
    assert refresh_calls == [("kraken", True)]
    assert spawned_threads == [] and real_thread is not None


def test_trades_stream_per_symbol_fetchtime_coherence_invariant():
    """Pure invariant: trades stream health must detect stream lag, missing per-symbol fetchtime, and per-symbol lag while respecting retired symbols and primary-stream cadence."""
    now_s = 1_795_760_000.0
    sync_g = 15.0

    # 1. Missing HYPEUSDC fetchtime in cache_trade_hyperliquid.json while sibling SOLUSDC is fresh.
    ev_missing = VenueCoherenceEvidence(
        venue="hyperliquid",
        trade_items={"HYPEUSDC": [], "SOLUSDC": []},
        trade_latest_ft_ts=now_s - 2.0,
        trade_sym_ft_ts={"SOLUSDC": now_s - 2.0},
        trade_file_exists=True,
        trade_has_fetchtime_surface=True,
        health_sync_ts_by_stream={"trades": now_s - 1.0},
        configured_symbols=frozenset({"HYPEUSDC", "SOLUSDC"}),
    )
    v_missing = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=now_s, venues={"hyperliquid": ev_missing}, health_present=True),
        sync_grace_sec=sync_g,
    )
    assert [(v.venue, v.rule) for v in v_missing] == [("hyperliquid", "health_symbol_fetchtime_missing")]
    assert "HYPEUSDC" in v_missing[0].detail and "trades" in v_missing[0].detail

    # 2. Lagging HYPEUSDC fetchtime in trades while retired OLDUSDC is ignored.
    ev_lag = VenueCoherenceEvidence(
        venue="hyperliquid",
        trade_items={"HYPEUSDC": [], "SOLUSDC": [], "OLDUSDC": []},
        trade_latest_ft_ts=now_s - 2.0,
        trade_sym_ft_ts={"HYPEUSDC": now_s - 45.0, "SOLUSDC": now_s - 2.0, "OLDUSDC": now_s - 900.0},
        trade_file_exists=True,
        trade_has_fetchtime_surface=True,
        health_sync_ts_by_stream={"trades": now_s - 1.0},
        configured_symbols=frozenset({"HYPEUSDC", "SOLUSDC"}),
    )
    v_lag = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=now_s, venues={"hyperliquid": ev_lag}, health_present=True),
        sync_grace_sec=sync_g,
    )
    assert [(v.venue, v.rule) for v in v_lag] == [("hyperliquid", "health_symbol_fetchtime_lag")]
    assert "HYPEUSDC" in v_lag[0].detail and "OLDUSDC" not in v_lag[0].detail

    # 3. Stream-level trades disk lag (all symbols stale relative to health).
    ev_stream_lag = VenueCoherenceEvidence(
        venue="kraken",
        trade_items={"HYPEUSD": []},
        trade_latest_ft_ts=now_s - 50.0,
        trade_sym_ft_ts={"HYPEUSD": now_s - 50.0},
        trade_file_exists=True,
        trade_has_fetchtime_surface=True,
        health_sync_ts_by_stream={"trades": now_s - 1.0},
        configured_symbols=frozenset({"HYPEUSD"}),
    )
    v_stream_lag = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=now_s, venues={"kraken": ev_stream_lag}, health_present=True),
        sync_grace_sec=sync_g,
    )
    assert [(v.venue, v.rule) for v in v_stream_lag] == [("kraken", "health_disk_lag")]

    # 4. Primary-stream venue (Binance) respects TRADE_SYNC_INTERVAL_SEC + sync_grace boundary.
    tr_cadence = cacheaccount.TRADE_SYNC_INTERVAL_SEC
    ev_bin_ok = VenueCoherenceEvidence(
        venue="binance",
        trade_items={"BTCUSDC": [], "ETHUSDC": []},
        trade_latest_ft_ts=now_s - 5.0,
        trade_sym_ft_ts={"BTCUSDC": now_s - 5.0, "ETHUSDC": now_s - (tr_cadence + sync_g - 5.0)},
        trade_file_exists=True,
        trade_has_fetchtime_surface=True,
        health_sync_ts_by_stream={"trades": now_s - 1.0},
        configured_symbols=frozenset({"BTCUSDC", "ETHUSDC"}),
    )
    assert evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=now_s, venues={"binance": ev_bin_ok}, health_present=True),
        sync_grace_sec=sync_g,
    ) == []

    ev_bin_bad = VenueCoherenceEvidence(
        venue="binance",
        trade_items={"BTCUSDC": [], "ETHUSDC": []},
        trade_latest_ft_ts=now_s - 5.0,
        trade_sym_ft_ts={"BTCUSDC": now_s - 5.0, "ETHUSDC": now_s - (tr_cadence + sync_g + 10.0)},
        trade_file_exists=True,
        trade_has_fetchtime_surface=True,
        health_sync_ts_by_stream={"trades": now_s - 1.0},
        configured_symbols=frozenset({"BTCUSDC", "ETHUSDC"}),
    )
    v_bin_bad = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=now_s, venues={"binance": ev_bin_bad}, health_present=True),
        sync_grace_sec=sync_g,
    )
    assert [(v.venue, v.rule) for v in v_bin_bad] == [("binance", "health_symbol_fetchtime_lag")]
    assert "ETHUSDC" in v_bin_bad[0].detail

    # 5. Synthetic snapshot without fetchtime surface (e.g. raw jsonl without .meta) does not false-fail.
    ev_no_surface = VenueCoherenceEvidence(
        venue="binance",
        trade_items={"BTCUSDC": []},
        trade_file_exists=True,
        trade_has_fetchtime_surface=False,
        health_sync_ts_by_stream={"trades": now_s - 1.0},
        configured_symbols=frozenset({"BTCUSDC"}),
    )
    assert evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=now_s, venues={"binance": ev_no_surface}, health_present=True),
        sync_grace_sec=sync_g,
    ) == []


def test_nonpositive_fetchtimes_dropped_across_account_cache_readers_and_cursors(tmp_path):
    """Invariant: zero/null/malformed fetchtimes must never pollute account cache dicts, spoof venue coverage, or trigger epoch-0 REST scans."""
    import json

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)

    # 1. load_sanitized_trade_snapshot strips zero/negative/null fetchtime entries.
    tr_path = acct_dir / "cache_trade_kraken.json"
    tr_path.write_text(
        json.dumps({
            "items": {"HYPEUSD": [], "SPCXxUSD": []},
            "fetchtime": {"HYPEUSD": 0, "SPCXxUSD": 1_795_760_000_000, "BADUSD": None},
        }),
        encoding="utf-8",
    )
    san = cacheaccount.CacheTradeManager.load_sanitized_trade_snapshot(str(tr_path), venue="kraken")
    assert san["fetchtime"] == {"SPCXxUSD": 1_795_760_000_000.0}

    # 2. read_venue_filled_orders_snapshot & read_all_venues_filled_orders drop non-positive fetchtimes (no false covered venue).
    fo_path = acct_dir / "cache_order_filled_hyperliquid.json"
    fo_path.write_text(
        json.dumps({
            "items": {},
            "fetchtime": {"HYPEUSDC": 0, "SOLUSDC": None, "BAD": "nan"},
        }),
        encoding="utf-8",
    )
    fo_items, fo_ft = cacheaccount.CacheOrderFilledManager.read_venue_filled_orders_snapshot(
        str(fo_path), venue="hyperliquid", allow_legacy=False, allow_trade_fallback=False
    )
    assert fo_items == {} and fo_ft == {}
    _, covered_fo, ft_by_v = cacheaccount.CacheOrderFilledManager.read_all_venues_filled_orders(
        str(acct_dir), allow_legacy=False, allow_trade_fallback=False
    )
    assert "hyperliquid" not in covered_fo and "hyperliquid" not in ft_by_v

    # 3. CacheOrderOpenManager snapshot readers/writers drop non-positive fetchtimes (no false covered venue).
    oo_path = acct_dir / "cache_order_open_hyperliquid.json"
    oo_path.write_text(
        json.dumps({
            "items": {},
            "fetchtime": {"HYPEUSDC": 0, "SOLUSDC": None, "BAD": -5},
            "verified_ts": None,
        }),
        encoding="utf-8",
    )
    oo_items, oo_ft, oo_ver, oo_sver = cacheaccount.CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(
        str(oo_path)
    )
    assert oo_items == {} and oo_ft == {} and oo_ver == 0.0 and oo_sver == {}
    _, covered_oo, _ = cacheaccount.CacheOrderOpenManager.read_all_venues_open_orders(str(acct_dir))
    assert "hyperliquid" not in covered_oo

    # 4. _hydrate_fetchtimes_from_mapping_locked & _resolve_remote_query_start_time reject 0/None/corrupt cursors.
    mgr = object.__new__(cacheaccount.CacheOrderFilledManager)
    mgr.symbols = ["BTCUSDC", "ETHUSDC"]
    mgr.fallback_time_default = 1_790_000_000_000
    mgr._hydrate_fetchtimes_from_mapping_locked({
        "fetchtime": {"BTCUSDC": 0, "ETHUSDC": 1_795_760_000_000, "BAD": None, "STR": "corrupt"},
        "rest_sync_cursor": {"BTCUSDC": 0, "ETHUSDC": 1_795_759_999_000, "BAD": False},
    })
    assert mgr.fetchtime_per_symbol == {"ETHUSDC": 1_795_760_000_000}
    assert mgr._rest_sync_cursor_per_symbol == {"ETHUSDC": 1_795_759_999_000}
    # Even if an external mutation injects 0 or None, _resolve_remote_query_start_time never crashes or returns 0.
    mgr._rest_sync_cursor_per_symbol["BTCUSDC"] = 0
    assert mgr._resolve_remote_query_start_time("BTCUSDC", None) == 1_790_000_000_000
    assert mgr._resolve_remote_query_start_time("BTCUSDC", 1_795_760_000_000) == 1_795_760_000_000
    assert mgr._resolve_remote_query_start_time("ETHUSDC", 1_795_760_000_000) == 1_795_759_999_000


def test_open_order_stale_overwrite_readonly_ws_guard_and_refresh_escalation(tmp_path, monkeypatch):
    """Cross-check: CacheOrderOpenManager refuses stale overwrites and read-only WS mutations, and request_venue_account_refresh does not drop include_open_orders escalation."""
    from types import SimpleNamespace
    import account_cache_health

    acct_dir = tmp_path / "account"
    acct_dir.mkdir(parents=True)
    oo_path = acct_dir / "cache_order_open_binance.json"
    t0 = 1_795_760_000.0
    t0_ms = int(t0 * 1000)

    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path),
        {"BTCUSDC": [{"orderId": "101", "side": "BUY", "price": 60000.0, "origQty": 0.01, "status": "NEW", "timestamp": t0_ms - 5_000}]},
        now_ms=t0_ms,
        verified_ts=t0,
    )
    stale_mgr = cacheaccount.CacheOrderOpenManager(15.0, ["BTCUSDC"], str(oo_path), read_only=False, venue="binance")
    ro_mgr = cacheaccount.CacheOrderOpenManager(15.0, ["BTCUSDC"], str(oo_path), read_only=True, venue="binance")

    # 1. Read-only consumer refuses WS open-order mutation when writer is disabled.
    monkeypatch.setattr(account_cache_health, "is_writer_enabled", lambda: False)
    assert ro_mgr.apply_ws_order_update("BTCUSDC", "remove", {"orderId": "101"}, now_ms=t0_ms + 5_000) is False
    assert len(ro_mgr.cache["BTCUSDC"]) == 1

    # 2. Newer snapshot written to disk at t0 + 20s must not be overwritten by stale_mgr still at t0.
    t1 = t0 + 20.0
    t1_ms = int(t1 * 1000)
    cacheaccount.CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(oo_path),
        {"BTCUSDC": []},
        now_ms=t1_ms,
        verified_ts=t1,
    )
    assert stale_mgr.save_state_to_file() is False
    disk_items, _, disk_ver, _ = cacheaccount.CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(oo_path))
    assert disk_items["BTCUSDC"] == []
    assert disk_ver == pytest.approx(t1, abs=1e-3)

    # 3. request_venue_account_refresh: balance-only refresh followed within <1s by fill with include_open_orders=True
    #    must still execute the open_orders refresh instead of dropping it behind the balance debounce window.
    calls: list[tuple[str, str]] = []
    mono_now = [100.0]
    monkeypatch.setattr(cacheaccount, "_monotonic_time", lambda: mono_now[0])
    monkeypatch.setattr(account_cache_health, "is_writer_enabled", lambda: True)
    monkeypatch.setattr(cacheaccount, "_venue_refresh_last_ts", {})
    monkeypatch.setattr(
        cacheaccount,
        "_venue_balances_bridge",
        SimpleNamespace(sync_venue_once=lambda v: calls.append(("bal", v)) or True),
    )
    monkeypatch.setattr(
        cacheaccount,
        "_venue_open_orders_bridge",
        SimpleNamespace(sync_venue_once=lambda v: calls.append(("open", v)) or True),
    )

    assert cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=False, blocking=True) is True
    assert calls == [("bal", "hyperliquid")]

    mono_now[0] = 100.2  # < 1.0s debounce gap
    assert cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=True, blocking=True) is True
    assert calls == [("bal", "hyperliquid"), ("open", "hyperliquid")]

    # Repeat include_open_orders=True within 1.0s is cleanly debounced for both streams.
    mono_now[0] = 100.4
    assert cacheaccount.request_venue_account_refresh("hyperliquid", include_open_orders=True, blocking=True) is False
    assert calls == [("bal", "hyperliquid"), ("open", "hyperliquid")]


def test_short_trend_save_drops_nonpositive_fetchtimes(tmp_path):
    """Invariant: CacheShortTrendManager.save_state_to_file must never persist zero/invalid fetchtime entries into .meta."""
    import json
    import cachetrend

    st_path = tmp_path / "cache_short_trend.json"
    mgr = cachetrend.CacheShortTrendManager(
        ["BTCUSDC", "ETHUSDC"],
        str(st_path),
        data_sync_mode=cacheaccount.CacheManagerInterface.MODE_PRODUCER,
    )
    t_ok = 1_795_760_000.0
    mgr._mem["BTCUSDC"] = {"symbol": "BTCUSDC", "current_price": 60500.0, "ts": t_ok}
    mgr._mem["ETHUSDC"] = {"symbol": "ETHUSDC", "current_price": 1700.0}  # missing ts
    mgr.fetchtime_per_symbol["ETHUSDC"] = 0
    assert mgr.save_state_to_file() is True
    meta = json.loads((tmp_path / "cache_short_trend.json.meta").read_text(encoding="utf-8"))
    assert "BTCUSDC" in meta["fetchtime"] and meta["fetchtime"]["BTCUSDC"] > 0
    assert "ETHUSDC" not in meta["fetchtime"]


def test_pass15_future_timestamp_coherence_invariants_and_cli_crosscheck(tmp_path, monkeypatch, capsys):
    """Pass #15 invariants & cross-checks: corrupt far-future timestamps (> now + 60s) must never silently pass
    as fresh/negative-lag in account_cache_coherence or on closed-market symbols in check_cache_coherence."""
    t_now = 1_795_765_000.0

    # 1. Pure invariants: far-future balance_verified_ts triggers causal_fill_to_balance + health_disk_future_ts;
    #    far-future per-symbol filled_orders/trades fetchtime triggers health_disk_future_ts + health_symbol_fetchtime_future.
    ev_future = VenueCoherenceEvidence(
        venue="kraken",
        balance_file_exists=True,
        balance_verified_ts=t_now + 7200.0,
        filled_file_exists=True,
        filled_items={
            "HYPEUSD": [{"orderId": "kr-fut-1", "symbol": "HYPEUSD", "status": "FILLED", "timestamp": int((t_now - 30.0) * 1000)}],
            "SPCXxUSD": [],
        },
        filled_sym_ft_ts={"HYPEUSD": t_now + 3600.0, "SPCXxUSD": t_now - 2.0},
        filled_latest_ft_ts=t_now + 3600.0,
        trade_file_exists=True,
        trade_has_fetchtime_surface=True,
        trade_items={"HYPEUSD": [], "SPCXxUSD": []},
        trade_sym_ft_ts={"HYPEUSD": t_now + 3600.0, "SPCXxUSD": t_now - 2.0},
        trade_latest_ft_ts=t_now + 3600.0,
        health_sync_ts_by_stream={
            "balances": t_now - 2.0,
            "filled_orders": t_now - 2.0,
            "trades": t_now - 2.0,
        },
        configured_symbols=frozenset({"HYPEUSD", "SPCXxUSD"}),
    )
    viols = evaluate_account_cache_coherence(
        AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_future}, health_present=True),
        sync_grace_sec=15.0,
    )
    rules = [v.rule for v in viols]
    assert "causal_fill_to_balance" in rules
    assert rules.count("health_disk_future_ts") == 3
    assert rules.count("health_symbol_fetchtime_future") == 2

    # 2. Dependent cross-check: check_cache_coherence.main() rejects far-future timestamps even when market is closed
    coh_dir = tmp_path / "coh_fut"
    coh_dir.mkdir(parents=True)
    atomic_write_json(
        str(coh_dir / "cache_short_trend.json"),
        {"items": {"NVDA_US_EQ": {"ts": t_now + 3600.0, "current_price": 180.0, "gradient_recent": 0.0}}},
    )
    atomic_write_json(
        str(coh_dir / "cache_current_price.json"),
        {"items": {"NVDA_US_EQ": [[(t_now + 3600.0) * 1000.0, 180.0]]}},
    )
    monkeypatch.setattr(ccc, "CACHEDB", str(coh_dir))
    monkeypatch.setattr(ccc, "expected_symbols", lambda: ["NVDA_US_EQ"])
    monkeypatch.setattr(ccc.time, "time", lambda: t_now)
    monkeypatch.setattr(ccc, "is_symbol_market_open", lambda _s, _n=None: False)
    assert ccc.main() == 1
    out_txt = capsys.readouterr().out
    assert "short_trend NVDA_US_EQ" in out_txt
    assert "currentprice NVDA_US_EQ" in out_txt


def test_pass21_account_coherence_future_balance_and_open_ts_without_fills_invariant_and_crosscheck():
    """Pass 21 pure invariant & cross-check: Rule 2 explicitly flags far-future balance_verified_ts and open_latest_ts even with zero fills/open orders."""
    from verify_tools.account_cache_coherence import (
        AccountCacheCoherenceSnapshot,
        VenueCoherenceEvidence,
        evaluate_account_cache_coherence,
    )

    t_now = 1_795_000_000.0

    # 1. Corrupt far-future balance_verified_ts & open_latest_ts with zero fills -> BOTH explicit errors emitted
    ev_future = VenueCoherenceEvidence(
        venue="kraken",
        balance_file_exists=True,
        balance_verified_ts=t_now + 3600.0,
        balance_items={"USD": {"free": 100.0, "locked": 0.0}},
        open_file_exists=True,
        open_latest_ts=t_now + 1800.0,
        open_items={"HYPEUSD": []},
    )
    snap_fut = AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_future})
    viols_fut = evaluate_account_cache_coherence(snap_fut)
    rules_fut = {v.rule for v in viols_fut}
    assert "balance_verified_ts_future" in rules_fut
    assert "open_orders_ts_future" in rules_fut

    # 2. Cross-check: normal past timestamps with aligned balances/open orders emit zero violations
    ev_ok = VenueCoherenceEvidence(
        venue="kraken",
        balance_file_exists=True,
        balance_verified_ts=t_now - 10.0,
        balance_items={"USD": {"free": 100.0, "locked": 0.0}},
        open_file_exists=True,
        open_latest_ts=t_now - 10.0,
        open_items={"HYPEUSD": []},
    )
    snap_ok = AccountCacheCoherenceSnapshot(now_s=t_now, venues={"kraken": ev_ok})
    viols_ok = evaluate_account_cache_coherence(snap_ok)
    assert not any(v.rule in ("balance_verified_ts_future", "open_orders_ts_future") for v in viols_ok)


def test_pass29_future_row_timestamps_never_blind_coherence_rules_invariant_and_crosscheck():
    """Pass 29 pure invariants & cross-checks: corrupt far-future row timestamps in filled_orders, trades, or
    open_orders must never blind Rules 1, 3, 4, 5a, or 5b and must emit explicit future-timestamp violations."""
    t_now = 1_796_250_000.0

    # 1. Pure invariants: one corrupt far-future filled row must NOT blind causal_fill_to_balance for a valid lagging fill,
    #    nor hide open_order_already_filled / ledger_stuck_filled_order; far-future trade emits trade_ts_future;
    #    far-future open order timestamp still triggers open_order_missing_from_ledger via snapshot fallback.
    ev_poisoned = VenueCoherenceEvidence(
        venue="hyperliquid",
        balance_file_exists=True,
        balance_verified_ts=t_now - 120.0,
        balance_items={"USDC": {"free": 1000.0, "locked": 0.0}},
        open_file_exists=True,
        open_latest_ts=t_now - 120.0,
        open_verified_ts=t_now - 120.0,
        open_items={
            "HYPEUSDC": [
                {
                    "orderId": "ord-already-filled",
                    "symbol": "HYPEUSDC",
                    "side": "BUY",
                    "price": 40.0,
                    "origQty": 1.0,
                    "executedQty": 0.0,
                    "timestamp": int((t_now - 60.0) * 1000),
                },
                {
                    "orderId": "ord-untracked-fut",
                    "symbol": "HYPEUSDC",
                    "side": "BUY",
                    "price": 39.0,
                    "origQty": 1.0,
                    "executedQty": 0.0,
                    "timestamp": int((t_now + 86400.0) * 1000),
                },
            ]
        },
        filled_file_exists=True,
        filled_items={
            "HYPEUSDC": [
                {
                    "orderId": "ord-valid-lag",
                    "symbol": "HYPEUSDC",
                    "status": "FILLED",
                    "origQty": 1.0,
                    "executedQty": 1.0,
                    "timestamp": int((t_now - 45.0) * 1000),
                },
                {
                    "orderId": "ord-already-filled",
                    "symbol": "HYPEUSDC",
                    "status": "FILLED",
                    "origQty": 1.0,
                    "executedQty": 1.0,
                    "timestamp": int((t_now + 86400.0) * 1000),
                },
                {
                    "orderId": "ord-ledger-stuck",
                    "symbol": "HYPEUSDC",
                    "status": "FILLED",
                    "origQty": 1.0,
                    "executedQty": 1.0,
                    "timestamp": int((t_now + 86400.0) * 1000),
                },
            ]
        },
        trade_file_exists=True,
        trade_items={
            "HYPEUSDC": [
                {
                    "id": "tr-fut-1",
                    "orderId": "ord-tr-fut",
                    "symbol": "HYPEUSDC",
                    "timestamp": int((t_now + 7200.0) * 1000),
                }
            ]
        },
    )
    snap_poisoned = AccountCacheCoherenceSnapshot(
        now_s=t_now,
        venues={"hyperliquid": ev_poisoned},
        ledger_present=True,
        ledger_updated_ts=t_now - 10.0,
        ledger_active_orders={
            "k1": {
                "v": "hyperliquid",
                "sym": "HYPEUSDC",
                "vid": "ord-already-filled",
                "state": "OPEN",
                "qty": 1.0,
                "updated_ts": t_now - 60.0,
            },
            "k2": {
                "v": "hyperliquid",
                "sym": "HYPEUSDC",
                "vid": "ord-ledger-stuck",
                "state": "OPEN",
                "qty": 1.0,
                "updated_ts": t_now - 60.0,
            },
        },
    )
    viols = evaluate_account_cache_coherence(snap_poisoned, sync_grace_sec=15.0, ledger_grace_sec=30.0)
    rules = [v.rule for v in viols]
    assert "filled_order_ts_future" in rules
    assert "causal_fill_to_balance" in rules
    assert "open_order_already_filled" in rules
    assert "trade_ts_future" in rules
    assert "open_order_missing_from_ledger" in rules
    assert "ledger_stuck_filled_order" in rules

    # 2. Dependent cross-check: valid bounded timestamps with aligned balance, trades, and ledger produce zero violations
    ev_clean = VenueCoherenceEvidence(
        venue="hyperliquid",
        balance_file_exists=True,
        balance_verified_ts=t_now - 5.0,
        balance_items={"USDC": {"free": 1000.0, "locked": 39.0}},
        open_file_exists=True,
        open_latest_ts=t_now - 5.0,
        open_verified_ts=t_now - 5.0,
        open_items={
            "HYPEUSDC": [
                {
                    "orderId": "ord-live-1",
                    "symbol": "HYPEUSDC",
                    "side": "BUY",
                    "price": 39.0,
                    "origQty": 1.0,
                    "executedQty": 0.0,
                    "timestamp": int((t_now - 60.0) * 1000),
                }
            ]
        },
        filled_file_exists=True,
        filled_items={
            "HYPEUSDC": [
                {
                    "orderId": "ord-valid-lag",
                    "symbol": "HYPEUSDC",
                    "status": "FILLED",
                    "origQty": 1.0,
                    "executedQty": 1.0,
                    "timestamp": int((t_now - 45.0) * 1000),
                }
            ]
        },
        trade_file_exists=True,
        trade_items={
            "HYPEUSDC": [
                {
                    "id": "tr-ok-1",
                    "orderId": "ord-valid-lag",
                    "symbol": "HYPEUSDC",
                    "timestamp": int((t_now - 45.0) * 1000),
                }
            ]
        },
    )
    snap_clean = AccountCacheCoherenceSnapshot(
        now_s=t_now,
        venues={"hyperliquid": ev_clean},
        ledger_present=True,
        ledger_updated_ts=t_now - 5.0,
        ledger_active_orders={
            "k1": {
                "v": "hyperliquid",
                "sym": "HYPEUSDC",
                "vid": "ord-live-1",
                "state": "OPEN",
                "qty": 1.0,
                "updated_ts": t_now - 60.0,
            }
        },
    )
    assert evaluate_account_cache_coherence(snap_clean, sync_grace_sec=15.0, ledger_grace_sec=30.0) == []


def test_account_multi_venue_bridges_nonblocking_isolation_and_hard_timeout_reset(monkeypatch):
    """_VenueOpenOrdersBridge and _VenueBalancesBridge must isolate hung venues (zero head-of-line blocking) and evict/retry wedged venues after hard_timeout_sec."""
    import threading
    import time

    # 1. _VenueOpenOrdersBridge: T212 hangs while Kraken and Hyperliquid keep syncing freely;
    #    after hard_timeout_sec, T212 slot is evicted and retried fresh without stalling siblings.
    release_t212_oo = threading.Event()
    oo_calls: dict[str, int] = {"kraken": 0, "hyperliquid": 0, "t212": 0}
    oo_ok: dict[str, int] = {"kraken": 0, "hyperliquid": 0, "t212": 0}
    oo_lock = threading.Lock()

    bridge_oo = cacheaccount._VenueOpenOrdersBridge(hard_timeout_sec=0.12, poll_interval_sec=0.03)

    def _fake_oo_sync(v_key: str, *, now_ms=None, trigger_balance_refresh=True) -> bool:
        with oo_lock:
            oo_calls[v_key] = oo_calls.get(v_key, 0) + 1
            call_n = oo_calls[v_key]
        if v_key == "t212" and call_n == 1:
            release_t212_oo.wait(timeout=5.0)
            return False
        with oo_lock:
            oo_ok[v_key] = oo_ok.get(v_key, 0) + 1
        return True

    monkeypatch.setattr(bridge_oo, "sync_venue_once", _fake_oo_sync)
    assert bridge_oo.start() is True
    try:
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            with oo_lock:
                if oo_ok["kraken"] >= 3 and oo_ok["hyperliquid"] >= 3 and oo_ok["t212"] >= 1:
                    break
            time.sleep(0.02)
        with oo_lock:
            assert oo_ok["kraken"] >= 3, f"Kraken stalled behind hung T212: {oo_ok}"
            assert oo_ok["hyperliquid"] >= 3, f"Hyperliquid stalled behind hung T212: {oo_ok}"
            assert oo_calls["t212"] >= 2 and oo_ok["t212"] >= 1, f"T212 failed to evict & retry: {oo_calls} / {oo_ok}"
    finally:
        release_t212_oo.set()
        bridge_oo.stop(timeout=2.0)

    # 2. _VenueBalancesBridge: hung T212 does not stall Binance/Kraken/Hyperliquid; after hard timeout,
    #    T212 recovers and once seeded during closed US hours it is skipped while crypto venues keep polling.
    release_t212_bal = threading.Event()
    bal_calls: dict[str, int] = {"binance": 0, "kraken": 0, "hyperliquid": 0, "t212": 0}
    bal_ok: dict[str, int] = {"binance": 0, "kraken": 0, "hyperliquid": 0, "t212": 0}
    bal_lock = threading.Lock()
    monkeypatch.setattr(cacheaccount, "is_symbol_market_open", lambda _s, _ts=None: False)

    bridge_bal = cacheaccount._VenueBalancesBridge(hard_timeout_sec=0.12, poll_interval_sec=0.03)

    def _fake_bal_sync(v_key: str, *, now_s=None) -> bool:
        with bal_lock:
            bal_calls[v_key] = bal_calls.get(v_key, 0) + 1
            call_n = bal_calls[v_key]
        if v_key == "t212" and call_n == 1:
            release_t212_bal.wait(timeout=5.0)
            return False
        with bal_lock:
            bal_ok[v_key] = bal_ok.get(v_key, 0) + 1
        return True

    monkeypatch.setattr(bridge_bal, "sync_venue_once", _fake_bal_sync)
    assert bridge_bal.start() is True
    try:
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            with bal_lock:
                if (
                    bal_ok["binance"] >= 3
                    and bal_ok["kraken"] >= 3
                    and bal_ok["hyperliquid"] >= 3
                    and bal_ok["t212"] >= 1
                ):
                    break
            time.sleep(0.02)
        time.sleep(0.12)
        with bal_lock:
            assert bal_ok["binance"] >= 3
            assert bal_ok["kraken"] >= 3
            assert bal_ok["hyperliquid"] >= 3
            assert bal_calls["t212"] == 2 and bal_ok["t212"] == 1
    finally:
        release_t212_bal.set()
        bridge_bal.stop(timeout=2.0)


def test_hl_and_kraken_ws_bg_rest_resync_hard_timeout_eviction_and_recovery(tmp_path, monkeypatch):
    """Wedged background REST resync threads on Hyperliquid and Kraken WS bridges must evict after hard timeout instead of locking resync forever."""
    import sys
    import threading
    import time
    import types

    acct_dir = tmp_path / "account_ht"
    acct_dir.mkdir(parents=True)
    monkeypatch.setattr(cacheaccount, "_record_venue_health_sync", lambda *_a, **_kw: True)
    monkeypatch.setattr(cacheaccount, "request_venue_account_refresh", lambda *_a, **_kw: True)

    # 1. Hyperliquid _HyperliquidAccountTradeBridge._schedule_invalidated_fills_resync
    hl_tr = acct_dir / "cache_trade_hyperliquid.json"
    hl_fo = acct_dir / "cache_order_filled_hyperliquid.json"
    monkeypatch.setattr(cacheaccount, "_sibling_order_filled_path", lambda _p, _v: str(hl_fo))
    release_hl = threading.Event()
    hl_started = threading.Event()
    hl_done = threading.Event()
    hl_calls = [0]
    t0_ms = 1_795_760_000_000

    class _FakeHlProv:
        def __init__(self, token=None):
            self.token = token

        def fetch_remote_trades(self, sym_s, _start):
            hl_calls[0] += 1
            if hl_calls[0] == 1:
                hl_started.set()
                release_hl.wait(timeout=5.0)
                raise RuntimeError("stale wedged timeout")
            rows = [{"symbol": sym_s, "id": "HL-REC-1", "orderId": "HL-O1", "side": "BUY", "price": 65.0, "qty": 1.0, "quantity": 1.0, "status": "FILLED", "time": t0_ms + 1000}]
            hl_done.set()
            return rows

        @staticmethod
        def resolve_ws_user_address(addr, resolve_env=None):
            return addr

    monkeypatch.setattr(
        cacheaccount.MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda _cls, v: _FakeHlProv if v == "hyperliquid" else None),
    )

    class _DummyWs:
        _lock = threading.Lock()
        _symbols = ("HYPEUSDC",)

    bridge = cacheaccount._HyperliquidAccountTradeBridge(_DummyWs(), user_address="0xabc", fills_cache_path=str(hl_tr))
    bridge._rest_resync_hard_timeout_sec = 0.08
    try:
        assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(hl_tr), t0_ms, blocking=False) is True
        assert hl_started.wait(timeout=2.0)
        # Immediate duplicate before hard timeout must be deduplicated
        assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(hl_tr), t0_ms + 1000, blocking=False) is False
        time.sleep(0.11)
        # After hard timeout, wedged slot is evicted and fresh retry runs & recovers cache
        assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(hl_tr), t0_ms + 2000, blocking=False) is True
        assert hl_done.wait(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not hl_fo.exists():
            time.sleep(0.02)
        assert hl_calls[0] == 2
        assert hl_fo.exists()
        # Late completion of wedged call #1 must NOT poison backoff after fresh call #2 succeeded
        release_hl.set()
        time.sleep(0.08)
        assert bridge._rest_backoff_sec == 0.0
        assert bridge._next_rest_resync_ms == 0
    finally:
        release_hl.set()

    # 2. Kraken run_kraken_trade_ws_loop periodic background REST Ledgers resync
    kr_tr = acct_dir / "cache_trade_kraken.json"
    kr_fo = acct_dir / "cache_order_filled_kraken.json"
    monkeypatch.setattr(cacheaccount, "_sibling_order_filled_path", lambda _p, _v: str(kr_fo))
    release_kr = threading.Event()
    kr_wedged_started = threading.Event()
    kr_recovered = threading.Event()
    kr_calls = [0]
    now_ref = [t0_ms]
    monkeypatch.setattr(cacheaccount, "resolve_now_ms", lambda v=None: now_ref[0] if v is None else int(v))

    class _FakeKrProv:
        PRIVATE_WS_URL = "wss://ws-auth.kraken.com/"

        @staticmethod
        def fetch_ledger_fills_by_symbol(_client, *, pairs, lookback_sec):
            kr_calls[0] += 1
            if kr_calls[0] == 1:
                return {"HYPEUSD": []}  # initial seed
            if kr_calls[0] == 2:
                kr_wedged_started.set()
                release_kr.wait(timeout=5.0)
                raise RuntimeError("stale wedged kraken call")
            kr_recovered.set()
            return {"HYPEUSD": [{"symbol": "HYPEUSD", "id": "KR-REC-1", "orderId": "KR-O1", "side": "BUY", "price": 66.0, "qty": 1.0, "quantity": 1.0, "status": "FILLED", "time": t0_ms + 5000}]}

        @staticmethod
        def build_ws_own_trades_subscription(_c):
            return '{"event":"subscribe"}'

        @staticmethod
        def extract_ws_own_trades(_d, *, pairs):
            return None

        @staticmethod
        def classify_ws_control_event(_d):
            return "keepalive", ""

    monkeypatch.setattr(
        cacheaccount.MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda _cls, v: _FakeKrProv if v == "kraken" else None),
    )
    stop_ev = threading.Event()

    class _FakeKrWs:
        def __init__(self, _url, *, on_open, on_message, on_error):
            self._on_open = on_open
            self._on_message = on_message
            self.keep_running = True

        def send(self, _p):
            pass

        def run_forever(self, **_kw):
            self._on_open(self)
            # Trigger wedged periodic REST sync #2
            now_ref[0] = t0_ms + 320_000
            self._on_message(self, '{"event":"heartbeat"}')
            assert kr_wedged_started.wait(timeout=2.0)
            # Keepalive before hard timeout must be deduplicated (no 3rd call yet)
            now_ref[0] = t0_ms + 340_000
            self._on_message(self, '{"event":"heartbeat"}')
            assert kr_calls[0] == 2
            # Wait past rest_hard_timeout_sec -> next keepalive evicts wedged slot and runs call #3
            time.sleep(0.11)
            now_ref[0] = t0_ms + 360_000
            self._on_message(self, '{"event":"heartbeat"}')
            assert kr_recovered.wait(timeout=2.0)
            stop_ev.set()

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(WebSocketApp=_FakeKrWs))
    try:
        cacheaccount.run_kraken_trade_ws_loop(
            object(),
            cache_file=str(kr_tr),
            pairs=("HYPEUSD",),
            poll_interval=30.0,
            backoff_init=5.0,
            backoff_max=60.0,
            lookback_sec=3600.0,
            stop_event=stop_ev,
            rest_hard_timeout_sec=0.08,
        )
        assert kr_calls[0] == 3
    finally:
        release_kr.set()


def test_pass37_account_bridge_merge_failure_backoff_gen_guard_and_hook_isolation(tmp_path, monkeypatch):
    """Pass #37 pure invariants + dependent cross-checks:
    1. Hyperliquid WS invalidated-fills REST resync backs off and omits health sync when a cache merge fails.
    2. Kraken WS late stale pre-timeout REST sync does not wipe a newer generation's rate-limit backoff,
       and failed filled_orders merge on ownTrades does not advance last_flush_ms.
    3. _MultiVenueNonBlockingBridgeBase isolates should_skip_fn / on_success_fn errors per venue so sibling venues survive.
    """
    import sys
    import threading
    import time
    import types
    import cacheaccount

    t0_ms = 1_795_800_000_000

    # 1. Pure invariant: Hyperliquid invalidated-fills resync must back off & skip failed-cache health on merge error
    hl_tr = tmp_path / "cache_trade_hyperliquid_p37.json"
    health_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cacheaccount,
        "_record_venue_health_sync",
        lambda venue, domain, now_ms=None: health_calls.append((venue, domain)),
    )

    class _OkHlProv:
        def __init__(self, token=None):
            self.token = token

        @staticmethod
        def resolve_ws_user_address(addr, resolve_env=None):
            return addr

        def fetch_remote_trades(self, sym_s, start_ms):
            return [{"symbol": sym_s, "id": "T1", "orderId": "O1", "side": "BUY", "price": 40.0, "qty": 1.0, "time": t0_ms}]

    monkeypatch.setattr(
        cacheaccount.MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda _cls, v: _OkHlProv if v == "hyperliquid" else None),
    )
    monkeypatch.setattr(
        cacheaccount.CacheTradeManager,
        "merge_venue_fills_snapshot",
        staticmethod(lambda *a, **kw: True),
    )
    monkeypatch.setattr(
        cacheaccount.CacheOrderFilledManager,
        "merge_venue_filled_orders_snapshot",
        staticmethod(lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))),
    )
    ws_stub = types.SimpleNamespace(_lock=threading.Lock(), _symbols=["HYPEUSDC"])
    bridge = cacheaccount._HyperliquidAccountTradeBridge(
        ws_stub, user_address="0xabc", fills_cache_path=str(hl_tr)
    )
    assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(hl_tr), t0_ms, blocking=True) is True
    assert ("hyperliquid", "trades") in health_calls
    assert ("hyperliquid", "filled_orders") not in health_calls
    assert bridge._rest_backoff_sec >= 15.0
    assert bridge._next_rest_resync_ms > t0_ms
    # Backoff blocks immediate retry
    assert bridge._schedule_invalidated_fills_resync(("HYPEUSDC",), str(hl_tr), t0_ms + 1_000, blocking=True) is False

    # 2. Dependent cross-check: Kraken stale pre-timeout gen=1 completion must NOT erase gen=2 rate-limit backoff
    kr_tr = tmp_path / "cache_trade_kraken_p37.json"
    now_ref = [t0_ms]
    monkeypatch.setattr(cacheaccount, "resolve_now_ms", lambda now_ms=None: now_ref[0] if now_ms is None else now_ms)
    monkeypatch.setattr(
        cacheaccount.CacheTradeManager,
        "_merge_venue_fills_snapshot_locked",
        staticmethod(lambda *a, **kw: {"items": {"HYPEUSD": []}, "fetchtime": {"HYPEUSD": now_ref[0]}}),
    )
    monkeypatch.setattr(
        cacheaccount.CacheOrderFilledManager,
        "merge_venue_filled_orders_snapshot",
        staticmethod(lambda *a, **kw: {"items": {"HYPEUSD": []}, "fetchtime": {"HYPEUSD": now_ref[0]}}),
    )
    monkeypatch.setattr(
        cacheaccount.CacheTradeManager,
        "touch_venue_fills_snapshot",
        staticmethod(lambda *a, **kw: False),
    )
    monkeypatch.setattr(
        cacheaccount.CacheOrderFilledManager,
        "touch_venue_filled_orders_snapshot",
        staticmethod(lambda *a, **kw: False),
    )
    gen1_started = threading.Event()
    release_gen1 = threading.Event()
    gen1_done = threading.Event()
    gen2_failed = threading.Event()
    kr_calls = [0]

    class _RaceKrProv:
        PRIVATE_WS_URL = "wss://ws-auth.kraken.com/"

        @staticmethod
        def fetch_ledger_fills_by_symbol(_client, *, pairs, lookback_sec):
            kr_calls[0] += 1
            c = kr_calls[0]
            if c == 1:
                return {"HYPEUSD": []}  # initial seed
            if c == 2:
                gen1_started.set()
                release_gen1.wait(timeout=5.0)
                return {"HYPEUSD": []}  # late stale completion after hard timeout
            if c == 3:
                try:
                    raise RuntimeError("EAPI:Rate limit exceeded")
                finally:
                    gen2_failed.set()
            return {"HYPEUSD": []}

        @staticmethod
        def build_ws_own_trades_subscription(_c):
            return '{"event":"subscribe"}'

        @staticmethod
        def extract_ws_own_trades(_d, *, pairs):
            return None

        @staticmethod
        def classify_ws_control_event(_d):
            return "keepalive", ""

    monkeypatch.setattr(
        cacheaccount.MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda _cls, v: _RaceKrProv if v == "kraken" else None),
    )
    stop_ev = threading.Event()

    class _RaceKrWs:
        def __init__(self, _url, *, on_open, on_message, on_error):
            self._on_open = on_open
            self._on_message = on_message
            self.keep_running = True

        def send(self, _p):
            pass

        def run_forever(self, **_kw):
            try:
                self._on_open(self)
                # Launch gen=1 (call #2) which hangs past hard timeout
                now_ref[0] = t0_ms + 320_000
                self._on_message(self, '{"event":"heartbeat"}')
                assert gen1_started.wait(timeout=2.0)
                time.sleep(0.11)
                # Evict gen=1 and launch gen=2 (call #3) which hits 30s rate limit -> next_rest_sync_ms = t0 + 390s
                now_ref[0] = t0_ms + 360_000
                self._on_message(self, '{"event":"heartbeat"}')
                assert gen2_failed.wait(timeout=2.0)
                time.sleep(0.05)
                # Now release stale gen=1 so it finishes AFTER gen=2 recorded rate-limit backoff
                release_gen1.set()
                time.sleep(0.08)
                gen1_done.set()
                # Keepalive 20s later (< 30s rate-limit backoff) with invalidated cache must NOT launch call #4
                # because stale gen=1 must not wipe gen=2's rate-limit backoff.
                now_ref[0] = t0_ms + 380_000
                self._on_message(self, '{"event":"heartbeat"}')
                time.sleep(0.05)
            finally:
                stop_ev.set()

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(WebSocketApp=_RaceKrWs))
    try:
        cacheaccount.run_kraken_trade_ws_loop(
            object(),
            cache_file=str(kr_tr),
            pairs=("HYPEUSD",),
            poll_interval=30.0,
            backoff_init=5.0,
            backoff_max=60.0,
            lookback_sec=3600.0,
            stop_event=stop_ev,
            rest_hard_timeout_sec=0.08,
        )
        assert gen1_done.is_set()
        assert kr_calls[0] == 3
    finally:
        release_gen1.set()

    # 3. Dependent cross-check: _MultiVenueNonBlockingBridgeBase isolates throwing hooks so sibling venues continue
    class _HookBridge(cacheaccount._MultiVenueNonBlockingBridgeBase):
        LOG_TAG = "test_hook_bridge"

        def start(self) -> bool:
            return True

    hb = _HookBridge(hard_timeout_sec=1.0, poll_interval_sec=0.05)
    synced: list[str] = []
    hl_synced_ev = threading.Event()

    def _sync_fn(v_key, _ns, _nm):
        synced.append(v_key)
        if v_key == "hyperliquid":
            hl_synced_ev.set()
        return True

    def _bad_skip(v_key, _ns):
        if v_key == "kraken":
            raise RuntimeError("kraken skip boom")
        return False

    def _bad_success(v_key, _res):
        if v_key == "t212":
            raise RuntimeError("t212 on_success boom")

    th = threading.Thread(
        target=hb._run_multi_venue_slot_loop,
        args=(("kraken", "t212", "hyperliquid"),),
        kwargs={
            "interval_sec": 0.05,
            "sync_fn": _sync_fn,
            "should_skip_fn": _bad_skip,
            "on_success_fn": _bad_success,
            "pool_prefix": "TestHookIso",
        },
        daemon=True,
    )
    th.start()
    try:
        assert hl_synced_ev.wait(timeout=2.0)
        assert "t212" in synced
        assert "hyperliquid" in synced
        assert "kraken" not in synced
    finally:
        hb.request_stop()
        th.join(timeout=2.0)
        hb.stop(timeout=1.0)











