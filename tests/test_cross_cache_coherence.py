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

    # 5. Ledger rules: open_order_missing_from_ledger, ledger_stuck_filled_order, ledger_orphan_active_order
    ev_ledger = VenueCoherenceEvidence(
        venue="binance",
        open_items={"TAOUSDC": [{"orderId": "REST-9", "symbol": "TAOUSDC", "side": "SELL", "origQty": 1.0, "executedQty": 0.0, "price": 190.0, "timestamp": int((t_now - 90.0) * 1000)}]},
        open_verified_ts=t_now - 2.0,
        open_latest_ts=t_now - 2.0,
        open_sym_verified_ts={"taousdc": t_now - 2.0, "adausdc": t_now - 2.0},
        open_file_exists=True,
        filled_items={"TAOUSDC": [{"orderId": "FILLED-8", "symbol": "TAOUSDC", "status": "FILLED", "quantity": 2.0, "timestamp": int((t_now - 80.0) * 1000)}]},
        filled_file_exists=True,
    )
    snap_ledger = AccountCacheCoherenceSnapshot(
        now_s=t_now,
        venues={"binance": ev_ledger},
        ledger_present=True,
        ledger_updated_ts=t_now - 1.0,
        ledger_active_orders={
            "binance:FILLED-8": {"v": "binance", "sym": "TAOUSDC", "vid": "FILLED-8", "state": "OPEN", "updated_ts": t_now - 100.0},
            "binance:ORPHAN-7": {"v": "binance", "sym": "ADAUSDC", "vid": "ORPHAN-7", "state": "OPEN", "updated_ts": t_now - 100.0},
        },
    )
    v_led = evaluate_account_cache_coherence(snap_ledger, sync_grace_sec=15.0, ledger_grace_sec=45.0)
    assert sorted(v.rule for v in v_led) == [
        "ledger_orphan_active_order",
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
