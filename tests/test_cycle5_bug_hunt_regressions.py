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


def test_cycle5_backfill_boundary_day_preserves_live_byte_offsets(tmp_path):
    from verify_tools.backfill_order_ledger_from_audit import backfill_order_ledger_from_audit
    import state_io as sio

    cache_dir = tmp_path / "cachedb"
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    reader = OrderLedgerReader(cachedb_dir=str(cache_dir))

    # Write 5 compact live ledger events on 2026-11-25, cursor positioned after event #3.
    day_key = "2026-11-25"
    ev_path = Path(reader.events_path(day_key))
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    live_rows = [
        {
            "ts": 1_795_568_400.0 + idx * 10.0,
            "ev": "SUBMITTED",
            "v": "binance",
            "sym": "BTCUSDC",
            "cid": f"LIVE_{idx}",
            "side": "BUY",
            "qty": 0.01,
        }
        for idx in range(5)
    ]
    live_lines = [sio.encode_json_payload(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in live_rows]
    ev_path.write_bytes("".join(live_lines).encode("utf-8"))
    prev_off = len("".join(live_lines[:3]).encode("utf-8"))

    active_p = Path(reader.active_path())
    active_p.parent.mkdir(parents=True, exist_ok=True)
    sio.save_json_state(
        str(active_p),
        {
            "orders": {},
            "recently_closed": {},
            "journal_cursor": {
                "file": ev_path.name,
                "offset": prev_off,
                "offsets": {ev_path.name: prev_off},
            },
        },
    )

    # Historical audit file strictly earlier than the first live event on the same UTC day.
    hist_audit = [
        {
            "ts": 1_795_564_900.0 + idx,
            "event": "submit_requested",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "client_order_id": f"HIST_{idx}",
            "side": "BUY",
            "qty": 0.02,
        }
        for idx in range(2)
    ]
    (audit_dir / "execution_audit_2026-11-25.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in hist_audit),
        encoding="utf-8",
    )

    stats = backfill_order_ledger_from_audit(str(audit_dir), str(cache_dir), dry_run=False)
    assert stats["boundary_day_prepended"] == 2

    snap = reader.read_active_snapshot()
    shifted_off = snap["journal_cursor"]["offsets"][ev_path.name]
    unread, _ = sio.read_jsonl_dicts_from_offset(str(ev_path), shifted_off)
    assert [r["cid"] for r in unread] == ["LIVE_3", "LIVE_4"]


def test_cycle5_gemini_cli_timeout_returns_unhandled_for_rest_fallback(monkeypatch):
    import subprocess
    from intelligence.sentiment import gemini_client as gc

    cli = gc.GeminiClient.__new__(gc.GeminiClient)
    cli._cli_runner = "agy"
    cli._model = "gemini-2.5-pro"
    cli._JSON_MODE_LOCAL = SimpleNamespace(include_repo_rules=False)
    cli._QUOTA_MEM_CACHE = {}
    cli._resolve_cli_model_candidates = lambda *_a, **_kw: ["gemini-2.5-pro"]
    cli._resolve_model_pools = lambda **_kw: ((), (), ())
    cli._build_cli_cmd = lambda *_a, **_kw: ["agy", "run"]
    cli._invoke_cli = lambda *_a, **_kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["agy"], timeout=5))

    monkeypatch.setattr(gc, "resolve_bot_cli_base", lambda: "agy")
    handled, out = cli._query_cli_runner(
        "hello",
        selected_model="gemini-2.5-pro",
        timeout_sec=5.0,
        cache_ttl_sec=60.0,
        now=1_700_000_000.0,
        thread_title=None,
        topic_key=None,
        conversation_id=None,
        resume_prompt=None,
        include_repo_rules=False,
        require_json=False,
        cli_mocked=True,
    )
    assert handled is False
    assert out is None


def test_cycle5_intelligence_cli_finds_equity_and_xstock_pillar2_telemetry(tmp_path):
    from intelligence.cli import inspect_symbol_intelligence
    from cachepaths import telemetry_cache_file_path
    import state_io as sio

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sym in ("NVDA_US_EQ", "SPCXxUSD"):
        p = telemetry_cache_file_path(str(cache_dir), "orderbook_depth", sym)
        assert p is not None
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        sio.atomic_write_json(p, {"ts": 1_700_000_000.0, "bid_depth_usd": 12345.0})

    st_eq = inspect_symbol_intelligence("NVDA_US_EQ", cache_dir=str(cache_dir))
    assert st_eq["pillar2_external"]["orderbook"] is not None
    assert st_eq["pillar2_external"]["orderbook"]["bid_depth_usd"] == 12345.0

    st_xs = inspect_symbol_intelligence("SPCXxUSD", cache_dir=str(cache_dir))
    assert st_xs["pillar2_external"]["orderbook"] is not None
    assert st_xs["pillar2_external"]["orderbook"]["bid_depth_usd"] == 12345.0


def test_cycle5_macro_analyzer_does_not_double_scale_derived_intervals(monkeypatch, tmp_path):
    import intelligence.macro_analyzer as ma_mod

    monkeypatch.setattr(ma_mod, "_load_margins", lambda: {})
    monkeypatch.setattr(
        ma_mod,
        "resolve_derived_telemetry_intervals",
        lambda: {
            "crypto_executor_interval_sec": 600.0,
            "default_geopolitical_interval_sec": 27000.0,
            "default_sentiment_interval_sec": 18000.0,
        },
    )
    monkeypatch.setattr(ma_mod._geo_analyzer, "GeopoliticalThreatAnalyzer", lambda **_kw: SimpleNamespace())
    monkeypatch.setattr(ma_mod._sentiment_advisor, "SentimentAdvisor", lambda **_kw: SimpleNamespace())

    analyzer = ma_mod.MacroAnalyzer(cache_dir=str(tmp_path))
    assert analyzer.macro_llm_interval_sec == pytest.approx(27000.0)
    assert analyzer.advisor_interval_sec == pytest.approx(18000.0)


def test_cycle5_inspect_prod_fleet_distinct_ssh_paths_and_flat_breadth(monkeypatch, capsys):
    from verify_tools import inspect_prod_fleet as ipf

    probed = []

    def _fake_ssh_cat(p):
        probed.append(p)
        return None

    monkeypatch.setattr(ipf, "ssh_cat", _fake_ssh_cat)
    ipf._ssh_cat_cachedb_candidates("cache_balance_binance.json", ("account", "orders", ""))
    assert len(probed) == 3
    assert len(set(probed)) == 3
    assert probed[0].endswith("/cachedb/account/cache_balance_binance.json")
    assert probed[1].endswith("/cachedb/orders/cache_balance_binance.json")
    assert probed[2].endswith("/cachedb/cache_balance_binance.json")


def test_cycle5_monitor_night_no_deleted_healthcheck_sh():
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "verify_tools" / "monitor_night.py").read_text(encoding="utf-8")
    assert "healthcheck.sh" not in src
    assert "trading-orchestrator.service" in src
    assert "__RESTART_OK__" in src


def test_cycle5_rtrade_pair_store_skips_unpersisted_terminal_checkpoint(tmp_path):
    from rtrade_pair_store import RTradePairStore

    store = RTradePairStore(str(tmp_path / "rtrade_pairs.json"))
    assert store.has_pair("never-persisted") is False
    # Terminal checkpoint on unpersisted failed start must NOT raise KeyError.
    store.checkpoint("never-persisted", {"phase": "failed", "reason": "buy_insufficient_funds:USDC"}, terminal=True)
    # Non-terminal checkpoint on unknown pair still raises KeyError so state corruption is never hidden.
    with pytest.raises(KeyError):
        store.checkpoint("never-persisted", {"phase": "working"}, terminal=False)


def test_cycle5_rtrade_loop_isolates_per_coordinator_step_error():
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "rtrade.py").read_text(encoding="utf-8")
    loop_block = src.split("survivors, checkpoints = [], []", 1)[1].split("pair_store.checkpoint_many(checkpoints)", 1)[0]
    assert "except Exception as step_exc:" in loop_block
    assert "survivors.append(coordinator)" in loop_block
    assert "continue" in loop_block


def test_cycle5_llm_order_guard_classifies_kraken_xstocks_as_equities_for_decoupling(monkeypatch, tmp_path):
    import llm_order_guard as log_mod

    now_ts = 1_700_000_000.0
    monkeypatch.setattr(log_mod, "is_intelligence_master_off", lambda: False)
    monkeypatch.setattr(
        log_mod,
        "_load_margins",
        lambda: {"geopolitical_guard_mode": "enforce"},
    )
    geo_assessment = SimpleNamespace(
        threat_level="ELEVATED",
        risk_score=0.72,
        recommended_brake="DOWNSCALE_50",
        summary="Regional escalation",
        ts=now_ts - 2400.0,
    )
    monkeypatch.setattr(
        log_mod._geo_analyzer,
        "GeopoliticalThreatAnalyzer",
        lambda start_sync=False: SimpleNamespace(
            _cached_assessment=geo_assessment,
            _effective_cache_dir=lambda: str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        log_mod._mtc,
        "load_cached_macro_benchmarks",
        lambda _cdir: {
            "ts": now_ts - 60.0,
            "cross_asset_regime": "CRYPTO_BULLISH_DECOUPLING",
        },
    )
    # Under CRYPTO_BULLISH_DECOUPLING (equities selling off, crypto bullish), SPCXxUSD MUST NOT absorb the threat!
    allowed, reason, scale = log_mod.check_geopolitical_order_guards(
        "SPCXxUSD",
        "BUY",
        margins={"geopolitical_guard_mode": "enforce"},
        now=now_ts,
    )
    assert allowed is True
    assert scale < 1.0
    assert "absorbed" not in reason


def test_cycle6_cache24_compact_jsonl_acquires_generation_lock_before_self_lock(tmp_path):
    from contextlib import contextmanager
    import cacheprice as cp_mod

    order: list[str] = []

    class _TrackedLock:
        def __enter__(self):
            order.append("self.lock")
            return self

        def __exit__(self, *_a):
            return False

    @contextmanager
    def _tracked_gen_lock():
        order.append("gen_lock")
        yield

    mgr = cp_mod.Cache24PriceManager.__new__(cp_mod.Cache24PriceManager)
    mgr.append_persist = True
    mgr.filename = str(tmp_path / "cache_24h_price.jsonl")
    mgr.lock = _TrackedLock()
    mgr._generation_lock = _tracked_gen_lock
    mgr._filter_fetchtimes_to_symbols = False
    mgr.symbols = []
    mgr.cache = {}
    mgr.fetchtime = {}
    mgr._jsonl_needs_compact = False
    mgr._jsonl_trimmed_disk_rows = 0
    mgr._last_24h_compact_mono = 0.0
    mgr._base_compact_noop = False

    # Stub CacheManagerInterface.compact_jsonl to just verify lock order.
    orig_super = cp_mod.CacheManagerInterface.compact_jsonl
    try:
        cp_mod.CacheManagerInterface.compact_jsonl = lambda self: (self._generation_lock().__enter__(), True)[1]
        assert mgr.compact_jsonl() is True
    finally:
        cp_mod.CacheManagerInterface.compact_jsonl = orig_super
    assert order[:2] == ["gen_lock", "self.lock"]
