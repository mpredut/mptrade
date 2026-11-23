"""Dedicated negative-invariant regression suite covering all 11 audited order, LLM, and financial bugs."""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import active_intents
from cacheaccount import CacheOrderFilledManager, CacheOrderOpenManager
from cachepaths import (
    CACHE_BINANCE_ORDER_FILLED_FILENAME,
    CACHE_BINANCE_ORDER_OPEN_FILENAME,
)
from core import config as cfg
from intelligence.sentiment.llm_quota_router import is_degraded_trading_model, normalize_model_identity
from intelligence.sentiment.openai_compat_rest import (
    OpenAICompatConfig,
    extract_openai_compat_text,
    query_openai_compat_chat,
)
import monitororder
from order_ledger_reconciler_worker import OrderLedgerReconciler
from providers.ai_attribution import _evaluate_execution_audit_actors, evaluate_ai_attribution
from providers.ai_survival_guard import match_filled_intent_rows_lifo
from providers.execution_audit import ExecutionAudit
from providers.intent_execution import cancel_existing_ai_open_orders
from providers.intent_reconciliation import reconcile_intents_from_execution_audit
from providers.intent_state import (
    collect_known_ai_order_ids,
    load_durable_intents_state,
    upsert_durable_intent,
)
from providers.nav_drawdown import append_nav_history_point, evaluate_nav_drawdown_metrics
from providers.order_id_context import resolve_order_actor
from providers.order_ledger import OrderEventType, OrderLedgerReader, OrderLedgerState, OrderLedgerWriter
from state_io import StateReadError


def test_p1_cross_venue_order_id_collision_never_cancels_manual_binance_order():
    """Bug #1: Historical Kraken AI intent with orderId=42 must NOT classify Binance manual orderId=42 as AI."""
    intents = [
        {
            "intent_id": "krk-ai-42",
            "venue": "kraken",
            "symbol": "HYPEUSD",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "42", "status": "FILLED"},
        }
    ]
    scoped_all = collect_known_ai_order_ids(intents)
    # Same numeric ID on Binance must stay BOT, never AI
    assert resolve_order_actor({"orderId": "42", "clientOrderId": "web_manual_42"}, scoped_all, venue="binance", symbol="BTCUSDC") == "BOT"
    # Same venue + symbol matches AI
    assert resolve_order_actor({"orderId": "42"}, scoped_all, venue="kraken", symbol="HYPEUSD") == "AI"
    # Same venue, DIFFERENT symbol on Binance-like venues must stay BOT
    assert resolve_order_actor({"orderId": "42"}, scoped_all, venue="kraken", symbol="ETHUSD") == "BOT"

    bn_inst = SimpleNamespace(
        provider_name="binance",
        symbol="BTCUSDC",
        open_orders=MagicMock(return_value=[{"orderId": "42", "clientOrderId": "web_manual_42", "side": "BUY", "price": "60000"}]),
        cancel_order=MagicMock(return_value=True),
    )
    canceled = cancel_existing_ai_open_orders(
        bn_inst,
        "BUY",
        intents=intents,
        include_bot_orders=False,
    )
    assert canceled == []
    bn_inst.cancel_order.assert_not_called()


def test_p2_active_intents_and_order_ledger_scope_terminal_order_ids_by_symbol(tmp_path):
    """Bug #2: ETHUSDC FILLED #17 must NOT drop BTCUSDC NEW #17 from active_intents or OrderLedger."""
    orders_dir = tmp_path / "cachedb" / "orders"
    state_dir = tmp_path / "cachedb" / "state"
    orders_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    now_s = time.time()

    # ETHUSDC #17 is FILLED; BTCUSDC #17 is resting NEW open order
    (orders_dir / CACHE_BINANCE_ORDER_FILLED_FILENAME).write_text(
        json.dumps({
            "items": {
                "ETHUSDC": [{"orderId": 17, "price": 3100.0, "quantity": 0.5, "timestamp": int(now_s * 1000), "side": "BUY", "status": "FILLED"}],
            },
            "fetchtime": {"ETHUSDC": int(now_s * 1000)},
        }),
        encoding="utf-8",
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME),
        {
            "BTCUSDC": [{"orderId": "17", "clientOrderId": "AI_btc_17", "side": "BUY", "price": 65000.0, "origQty": 0.02, "executedQty": 0.0, "status": "NEW", "timestamp": int(now_s * 1000)}],
        },
        now_ms=int(now_s * 1000),
        verified_ts=now_s,
    )
    (state_dir / "autonomous_trade_intents_state.json").write_text(
        json.dumps([
            {
                "intent_id": "ai-btc-17",
                "venue": "binance",
                "symbol": "BTCUSDC",
                "decision": "BUY",
                "order_type": "LIMIT",
                "status": "EXECUTED",
                "timestamp": now_s - 10.0,
                "execution_payload": {"orderId": "17", "clientOrderId": "AI_btc_17", "status": "NEW", "origQty": "0.02"},
            }
        ]),
        encoding="utf-8",
    )

    idx = active_intents.build_active_intent_index(str(tmp_path))
    btc_rows = [r for r in idx["intents"] if r.get("symbol") == "BTCUSDC" and str(r.get("order_id")) == "17"]
    assert len(btc_rows) == 1

    # Also verify OrderLedgerReconciler keeps BTCUSDC #17 OPEN despite ETHUSDC #17 being FILLED!
    cdb = str(tmp_path / "cachedb")
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    writer.record_event(OrderEventType.ACKED, venue="binance", symbol="BTCUSDC", ts=now_s - 5.0, iid="ai-btc-17", cid="AI_btc_17", vid="17", side="BUY", qty=0.02, px=65000.0)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False)
    reconciler.reconcile_once(now_ts=now_s)
    active = OrderLedgerReader(cachedb_dir=cdb).list_active_orders(apply_tail=False)
    assert len(active) == 1
    assert active[0]["sym"] == "BTCUSDC"
    assert active[0]["state"] == OrderLedgerState.OPEN.value


def test_p1_targeted_cancel_already_canceled_never_sweeps_unrelated_mo_orders():
    """Bug #3: Missing targeted canceled order must NEVER sweep unrelated MO_ orders on the same side."""
    inst = SimpleNamespace(
        provider_name="binance",
        symbol="BTCUSDC",
        open_orders=MagicMock(return_value=[
            {"orderId": "mo_unrelated_1", "clientOrderId": "MO_TA_1", "side": "SELL", "price": "69000"},
            {"orderId": "mo_unrelated_2", "clientOrderId": "MO_AG_2", "side": "SELL", "price": "71000"},
        ]),
        cancel_order=MagicMock(return_value=True),
        order_status=MagicMock(return_value=SimpleNamespace(status="canceled", filled_qty=0.0, side="SELL")),
    )
    canceled = cancel_existing_ai_open_orders(
        inst,
        "SELL",
        include_bot_orders=True,
        target_order_ids=["already_canceled_999"],
        verify_not_filled=True,
        failed_order_ids=[],
    )
    assert canceled == []
    inst.cancel_order.assert_not_called()


def test_p1_monitororder_native_replace_unknown_with_lookup_timeout_never_double_submits(monkeypatch):
    """Bug #4a: Native replace returning unknown + client_order_id lookup timeout must NOT call place() twice."""
    claim = {"id": "claim-1", "symbol": "BTCUSDC", "side": "SELL", "qty": 1.0}
    complete_calls = []

    fake_mkt = MagicMock()
    fake_mkt.place.side_effect = TimeoutError("socket read timeout")
    fake_mkt.order_by_client_id.side_effect = TimeoutError("lookup timeout")
    fake_api = MagicMock()

    monkeypatch.setattr(monitororder, "mkt", fake_mkt)
    monkeypatch.setattr(monitororder, "api", fake_api)
    monkeypatch.setattr(monitororder, "_native_replace_enabled_for", lambda _s, _p: True)
    monkeypatch.setattr(monitororder.order_retry, "begin_claimed_submit", lambda c: c)
    monkeypatch.setattr(
        monitororder.order_retry,
        "complete_claim",
        lambda c, outcome, **kw: complete_calls.append((outcome, kw)) or True,
    )

    monitororder._dispatch_native_or_cancel_replace(
        symbol="BTCUSDC",
        order_type="SELL",
        order_id="old-1",
        oid_str="old-1",
        price="65000",
        price_f=65000.0,
        new_price=64900.0,
        quantity=1.0,
        original_quantity=1.0,
        provider_name="binance",
        replacement_claim=claim,
        client_order_id="MO_TA_child_1",
        cache_permit=None,
        preflight_qty=1.0,
        replace_bypass_guards=[],
        replace_wait_for_trend=False,
        orders={},
        initial_prices={},
    )
    assert fake_mkt.place.call_count == 1
    fake_api.cancel_order.assert_not_called()
    assert len(complete_calls) == 1
    assert complete_calls[0][0] == "failure"
    assert complete_calls[0][1].get("submission_state") == "unknown"


def test_p1_trade_enabled_hot_reload_turns_false_despite_root_loaded_environ(tmp_path, monkeypatch):
    """Bug #5: Changing TRADE_ENABLED=true -> false in config.env must hot-reload to False even after load_dotenv()."""
    cfg_file = tmp_path / "config.env"
    txt_file = tmp_path / "config.txt"
    cfg_file.write_text("TRADE_ENABLED=true\n", encoding="utf-8")

    import botcore

    monkeypatch.delenv("TRADE_ENABLED", raising=False)
    monkeypatch.setattr(cfg, "CONFIG_ENV_PATH", str(cfg_file))
    monkeypatch.setattr(cfg, "CONFIG_TXT_PATH", str(txt_file))
    monkeypatch.setattr(cfg, "_ROOT_CONFIG_PATH", str(cfg_file))
    monkeypatch.setattr(botcore, "CONFIG_ENV_PATH", str(cfg_file), raising=False)
    monkeypatch.setattr(botcore, "CONFIG_TXT_PATH", str(txt_file), raising=False)
    monkeypatch.setattr(botcore, "_ROOT_CONFIG_PATH", str(cfg_file), raising=False)
    monkeypatch.setattr(cfg, "_ROOT_CONFIG_KEYS", set())
    monkeypatch.setattr(cfg, "_ROOT_CONFIG_VALUES", {})

    cfg.load_dotenv(str(cfg_file))
    cfg.load_config()
    assert os.environ.get("TRADE_ENABLED") == "true"
    assert cfg.is_trade_enabled() is True

    # Flip TRADE_ENABLED=false in file and reload
    cfg_file.write_text("TRADE_ENABLED=false\n", encoding="utf-8")
    cfg.load_config()
    assert cfg.is_trade_enabled() is False

    # Explicit external override in os.environ STILL wins over file
    monkeypatch.setenv("TRADE_ENABLED", "true")
    cfg.load_config()
    assert cfg.is_trade_enabled() is True


def test_p1_pending_execution_with_submit_requested_becomes_submit_unknown_not_abandoned(tmp_path):
    """Bug #6: PENDING_EXECUTION > 900s with submit_requested in audit must become SUBMIT_UNKNOWN, never FAILED/abandoned."""
    audit_dir = tmp_path / "audit"
    audit = ExecutionAudit(directory=str(audit_dir))
    old_ts = time.time() - 901.0
    assert audit.record(
        "submit_requested",
        intent_id="ai-pend-901",
        venue="binance",
        symbol="BTCUSDC",
        side="buy",
        qty=0.05,
        price=64000.0,
        client_order_id="AI_pend_901",
    )

    intents = [
        {
            "intent_id": "ai-pend-901",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "decision": "BUY",
            "status": "PENDING_EXECUTION",
            "timestamp": old_ts,
        }
    ]
    changed = reconcile_intents_from_execution_audit(intents, str(audit_dir))
    assert changed is True
    assert intents[0]["status"] == "SUBMIT_UNKNOWN"
    assert intents[0]["execution_payload"]["clientOrderId"] == "AI_pend_901"
    assert intents[0]["execution_payload"]["submission_state"] == "unknown"


def test_p1_corrupt_intents_queue_fails_closed_without_overwriting_existing_file(tmp_path):
    """Bug #7: Existing truncated/corrupt durable intents file must fail closed on write and never be wiped."""
    q_path = tmp_path / "autonomous_trade_intents_state.json"

    # Absent file still initializes cleanly as empty list
    assert load_durable_intents_state(str(q_path)) == []

    # Truncated existing file with no .previous backup must raise StateReadError on writable load
    q_path.write_text('[{"intent_id": "old-1"', encoding="utf-8")
    with pytest.raises(StateReadError):
        load_durable_intents_state(str(q_path))

    # Upsert must refuse (return False) and preserve original corrupt bytes untouched for forensics
    assert upsert_durable_intent(str(q_path), {"intent_id": "new-2", "status": "PENDING"}) is False
    assert q_path.read_text(encoding="utf-8") == '[{"intent_id": "old-1"'


def test_p1_degraded_model_classification_normalizes_vendor_prefixes():
    """Bug #8: openai_compat:gemini-2.5-flash and openai_compat:google/gemini-2.5-flash must share the same policy."""
    assert normalize_model_identity("openai_compat:google/gemini-2.5-flash") == "gemini-2.5-flash"
    assert normalize_model_identity("openai_compat:gemini-2.5-flash") == "gemini-2.5-flash"
    assert is_degraded_trading_model("openai_compat:gemini-2.5-flash") is True
    assert is_degraded_trading_model("openai_compat:google/gemini-2.5-flash") is True
    assert is_degraded_trading_model("openai_compat:google/gemini-2.5-pro") is True
    assert is_degraded_trading_model("openai_compat:anthropic/claude-sonnet-5-5-medium") is False


def test_p2_openai_compat_rejects_truncated_length_and_unclosed_or_only_think_blocks():
    """Bug #9: Reject finish_reason='length', unclosed <think>, and <think>-only JSON; accept valid post-think JSON."""
    cfg_oai = OpenAICompatConfig(url="https://example.invalid/v1/chat/completions", api_key="k", models=("m1",))

    # 1. Truncated response with JSON only inside <think>...</think>
    trunc_resp = {
        "choices": [{
            "finish_reason": "length",
            "message": {"content": '<think>Maybe {"decision":"BUY","symbol":"BTCUSDC"}</think>'},
        }]
    }
    assert extract_openai_compat_text(trunc_resp, require_complete=True) == ""
    assert extract_openai_compat_text(trunc_resp, require_complete=False) == ""

    # 2. Unclosed <think> block with JSON inside
    unclosed_resp = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": '<think>Considering {"decision":"BUY","symbol":"BTCUSDC"}'},
        }]
    }
    assert extract_openai_compat_text(unclosed_resp, require_complete=True) == ""

    # 3. Valid complete response with final JSON after </think>
    valid_resp = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": '<think>Draft {"decision":"BUY"}</think>\n{"decision":"HOLD"}'},
        }]
    }
    assert extract_openai_compat_text(valid_resp, require_complete=True) == '{"decision":"HOLD"}'
    txt, used = query_openai_compat_chat(
        "test", cfg_oai, require_json=True, post_fn=lambda *_a, **_kw: valid_resp
    )
    assert txt == '{"decision":"HOLD"}' and used == "openai_compat:m1"
    txt_bad, used_bad = query_openai_compat_chat(
        "test", cfg_oai, require_json=True, post_fn=lambda *_a, **_kw: trunc_resp
    )
    assert txt_bad is None and used_bad is None


def test_p2_lifo_partial_lot_matching_computes_exact_190_usd_for_ai_and_non_ai(tmp_path):
    """Bug #10: BUY 10x100, SELL 1x110, SELL 9x120 (0 fees) MUST compute exact +$190 gross in BOTH AI and fleet reports."""
    rows = [
        {"ts": 10.0, "inst_key": ("binance", "ETH", ("USD",)), "side": "BUY", "px": 100.0, "qty": 10.0, "usd": 1000.0},
        {"ts": 20.0, "inst_key": ("binance", "ETH", ("USD",)), "side": "SELL", "px": 110.0, "qty": 1.0, "usd": 110.0},
        {"ts": 30.0, "inst_key": ("binance", "ETH", ("USD",)), "side": "SELL", "px": 120.0, "qty": 9.0, "usd": 1080.0},
    ]
    closed, open_b, open_s = match_filled_intent_rows_lifo(rows)
    assert len(closed) == 2
    assert open_b == [] and open_s == []
    gross_total = sum(e["usd"] * ((x["px"] - e["px"]) / e["px"]) for _, e, x in closed)
    assert gross_total == pytest.approx(190.0)

    # Full build_ai_attribution_report with 0 fees
    now_s = time.time()
    intents = [
        {"intent_id": "b1", "timestamp": now_s - 300, "decision": "BUY", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1000.0, "execution_payload": {"orderId": "1", "status": "FILLED", "avg_fill_price": 100.0, "executedQty": "10"}},
        {"intent_id": "s1", "timestamp": now_s - 200, "decision": "SELL", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 110.0, "execution_payload": {"orderId": "2", "status": "FILLED", "avg_fill_price": 110.0, "executedQty": "1"}},
        {"intent_id": "s2", "timestamp": now_s - 100, "decision": "SELL", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1080.0, "execution_payload": {"orderId": "3", "status": "FILLED", "avg_fill_price": 120.0, "executedQty": "9"}},
    ]
    report = evaluate_ai_attribution(
        intents,
        now_ts=now_s,
        fee_resolver=lambda *_a: 0.0,
    )
    assert report["ai_summary"]["realized_gross_usd"] == pytest.approx(190.0)
    assert report["ai_summary"]["realized_net_usd"] == pytest.approx(190.0)

    # Non-AI fleet audit comparison uses the exact same partial-lot LIFO primitive -> exact +$190
    audit_recs = [
        {"event": "order_status", "ts": now_s - 300, "intent_id": "bot-b1", "venue": "binance", "symbol": "ETHUSDC", "order_id": "101", "client_order_id": "TA_101", "side": "buy", "status": "closed", "venue_status": "FILLED", "filled_qty": 10.0, "price": 100.0, "cost": 1000.0},
        {"event": "order_status", "ts": now_s - 200, "intent_id": "bot-s1", "venue": "binance", "symbol": "ETHUSDC", "order_id": "102", "client_order_id": "TA_102", "side": "sell", "status": "closed", "venue_status": "FILLED", "filled_qty": 1.0, "price": 110.0, "cost": 110.0},
        {"event": "order_status", "ts": now_s - 100, "intent_id": "bot-s2", "venue": "binance", "symbol": "ETHUSDC", "order_id": "103", "client_order_id": "TA_103", "side": "sell", "status": "closed", "venue_status": "FILLED", "filled_qty": 9.0, "price": 120.0, "cost": 1080.0},
    ]
    cmp_rep = _evaluate_execution_audit_actors(
        audit_recs,
        known_ai_order_ids=set(),
        cutoff_ts=now_s - 86400.0,
        eff_now=now_s + 10.0,
        scope="all",
        fee_resolver=lambda *_a: 0.0,
    )
    assert cmp_rep["by_actor"]["BOT"]["realized_net_usd"] == pytest.approx(190.0)


def test_p2_nav_drawdown_never_hides_market_loss_after_large_buy():
    """Bug #11: A $2,000 BUY followed by a $600 market loss ($10,000 -> $9,400) MUST score -6% DD, not 0% cashflow."""
    t0 = 1_770_000_000.0
    hist: list[dict] = []
    hist, _ = append_nav_history_point(
        hist,
        {"ts": t0, "nav_usd": 10000.0, "invested_usd": 0.0, "free_cash_usd": 10000.0, "venues_usd": {"binance": 10000.0}},
    )
    hist, _ = append_nav_history_point(
        hist,
        {"ts": t0 + 3600.0, "nav_usd": 9400.0, "invested_usd": 1400.0, "free_cash_usd": 8000.0, "venues_usd": {"binance": 9400.0}},
    )
    assert not hist[-1].get("external_cashflow")
    metrics = evaluate_nav_drawdown_metrics(hist, now_ts=t0 + 3600.0)
    assert metrics["external_cashflow_events"] == 0
    assert metrics["daily_nav_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)
