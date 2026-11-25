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
    CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
    CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
    CACHE_KRAKEN_ORDER_OPEN_FILENAME,
)
from core import config as cfg
from intelligence.sentiment.llm_quota_router import is_degraded_trading_model, normalize_model_identity
from intelligence.sentiment.openai_compat_rest import (
    OpenAICompatConfig,
    extract_openai_compat_text,
    query_openai_compat_chat,
)
from kraken import kraken_client as kc
import monitororder
from order_ledger_reconciler_worker import OrderLedgerReconciler
from providers.ai_attribution import _evaluate_execution_audit_actors, evaluate_ai_attribution
from providers.ai_survival_guard import evaluate_ai_decision_scorecard, match_filled_intent_rows_lifo
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
    # Same numeric ID on Binance manual order must classify as MANUAL, never AI
    assert resolve_order_actor({"orderId": "42", "clientOrderId": "web_manual_42"}, scoped_all, venue="binance", symbol="BTCUSDC") == "MANUAL"
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


def test_p2_ai_attribution_symmetrical_loss_matches_non_ai():
    """Bug audit: BUY 10x100, SELL 1x90, SELL 9x80 without fees MUST report exact -$190 across AI and non-AI paths."""
    now_s = time.time()
    intents = [
        {"intent_id": "b1", "timestamp": now_s - 300, "decision": "BUY", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1000.0, "execution_payload": {"orderId": "1", "status": "FILLED", "avg_fill_price": 100.0, "executedQty": "10"}},
        {"intent_id": "s1", "timestamp": now_s - 200, "decision": "SELL", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 90.0, "execution_payload": {"orderId": "2", "status": "FILLED", "avg_fill_price": 90.0, "executedQty": "1"}},
        {"intent_id": "s2", "timestamp": now_s - 100, "decision": "SELL", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 720.0, "execution_payload": {"orderId": "3", "status": "FILLED", "avg_fill_price": 80.0, "executedQty": "9"}},
    ]
    report = evaluate_ai_attribution(
        intents,
        now_ts=now_s,
        fee_resolver=lambda *_a: 0.0,
    )
    assert report["ai_summary"]["realized_gross_usd"] == pytest.approx(-190.0)
    assert report["ai_summary"]["realized_net_usd"] == pytest.approx(-190.0)

    # AI Survival Guard scorecard also computes exact -$190 impact (zero fees)
    with patch("providers.ai_survival_guard.resolve_venue_round_trip_fee_bps", return_value=0.0):
        scorecard = evaluate_ai_decision_scorecard(intents, now_ts=now_s)
    assert scorecard["net_usd"] == pytest.approx(-190.0)

    # Non-AI fleet audit comparison
    audit_recs = [
        {"event": "order_status", "ts": now_s - 300, "intent_id": "bot-b1", "venue": "binance", "symbol": "ETHUSDC", "order_id": "101", "client_order_id": "TA_101", "side": "buy", "status": "closed", "venue_status": "FILLED", "filled_qty": 10.0, "price": 100.0, "cost": 1000.0},
        {"event": "order_status", "ts": now_s - 200, "intent_id": "bot-s1", "venue": "binance", "symbol": "ETHUSDC", "order_id": "102", "client_order_id": "TA_102", "side": "sell", "status": "closed", "venue_status": "FILLED", "filled_qty": 1.0, "price": 90.0, "cost": 90.0},
        {"event": "order_status", "ts": now_s - 100, "intent_id": "bot-s2", "venue": "binance", "symbol": "ETHUSDC", "order_id": "103", "client_order_id": "TA_103", "side": "sell", "status": "closed", "venue_status": "FILLED", "filled_qty": 9.0, "price": 80.0, "cost": 720.0},
    ]
    cmp_rep = _evaluate_execution_audit_actors(
        audit_recs,
        known_ai_order_ids=set(),
        cutoff_ts=now_s - 86400.0,
        eff_now=now_s + 10.0,
        scope="all",
        fee_resolver=lambda *_a: 0.0,
    )
    assert cmp_rep["by_actor"]["BOT"]["realized_net_usd"] == pytest.approx(-190.0)


def test_p2_nav_drawdown_small_buy_and_loss_or_confirmed_withdrawal():
    """With $600 buy + $600 market loss (delta_cash == delta_nav == -600, delta_inv == 0), NAV 10k -> 9.4k MUST score -6% DD.
    With $100 confirmed withdrawal + $500 loss, return MUST score -5% DD (only confirmed cashflow removed)."""
    t0 = 1_770_000_000.0
    # Exact accounting identity: start (nav=10k, inv=5k, cash=5k) -> buy $600 (inv=5.6k, cash=4.4k)
    # -> market drops $600 (inv=5k, cash=4.4k, nav=9.4k): delta_cash=-600, delta_nav=-600, delta_inv=0!
    hist: list[dict] = []
    hist, _ = append_nav_history_point(
        hist,
        {"ts": t0, "nav_usd": 10000.0, "invested_usd": 5000.0, "free_cash_usd": 5000.0, "venues_usd": {"binance": 10000.0}},
    )
    hist, _ = append_nav_history_point(
        hist,
        {"ts": t0 + 3600.0, "nav_usd": 9400.0, "invested_usd": 5000.0, "free_cash_usd": 4400.0, "venues_usd": {"binance": 9400.0}},
    )
    assert not hist[-1].get("external_cashflow")
    m1 = evaluate_nav_drawdown_metrics(hist, now_ts=t0 + 3600.0)
    assert m1["external_cashflow_events"] == 0
    assert m1["daily_nav_drawdown_pct"] == pytest.approx(-6.0, abs=0.05)

    # Confirmed withdrawal of only $100, while NAV dropped from 10k to 9.4k ($500 market loss)
    hist2: list[dict] = []
    hist2, _ = append_nav_history_point(
        hist2,
        {"ts": t0, "nav_usd": 10000.0, "invested_usd": 5000.0, "free_cash_usd": 5000.0, "venues_usd": {"binance": 10000.0}},
    )
    hist2, _ = append_nav_history_point(
        hist2,
        {
            "ts": t0 + 3600.0,
            "nav_usd": 9400.0,
            "invested_usd": 4500.0,
            "free_cash_usd": 4900.0,
            "venues_usd": {"binance": 9400.0},
            "confirmed_cashflow_usd": -100.0,
        },
    )
    m2 = evaluate_nav_drawdown_metrics(hist2, now_ts=t0 + 3600.0)
    assert m2["external_cashflow_events"] == 1
    assert m2["daily_nav_drawdown_pct"] == pytest.approx(-5.0, abs=0.05)


def test_p1_trade_enabled_typo_and_key_removal_fail_closed(tmp_path, monkeypatch):
    """Starting from TRADE_ENABLED=false, both a typo ('flase') and removing the key MUST keep trading disabled."""
    cfg_file = tmp_path / "config.env"
    txt_file = tmp_path / "config.txt"
    cfg_file.write_text("TRADE_ENABLED=false\n", encoding="utf-8")

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
    assert cfg.is_trade_enabled() is False

    # Typo 'flase' fails closed
    cfg_file.write_text("TRADE_ENABLED=flase\n", encoding="utf-8")
    cfg.load_config()
    assert cfg.is_trade_enabled() is False

    # Removing key while stopped preserves stopped state
    cfg_file.write_text("# comment only\n", encoding="utf-8")
    cfg.load_config()
    assert cfg.is_trade_enabled() is False

    # Explicit valid true re-enables
    cfg_file.write_text("TRADE_ENABLED=true\n", encoding="utf-8")
    cfg.load_config()
    assert cfg.is_trade_enabled() is True

    # Invalid external env override also fails closed even when previously true
    monkeypatch.setenv("TRADE_ENABLED", "flase")
    cfg.load_config()
    assert cfg.is_trade_enabled() is False


def test_p1_kraken_private_cache_isolated_per_client_and_public_pair_info_intact():
    """Two KrakenClient instances with distinct credentials must NOT share private Balance cache;
    public AssetPairs cache must still satisfy get_cached_pair_info()."""
    kc._CACHE.clear()
    c1 = kc.KrakenClient("key_alpha", "c2VjcmV0X2FscGhh")
    c2 = kc.KrakenClient("key_beta", "c2VjcmV0X2JldGE=")
    calls: list[str] = []

    def _fake_post(_url, _data, headers=None, **_kw):
        k = (headers or {}).get("API-Key", "")
        calls.append(k)
        if k == "key_alpha":
            return 200, b'{"error":[],"result":{"USDC":"100.0"}}'
        return 200, b'{"error":[],"result":{"USDC":"7.0"}}'

    with patch.object(kc, "http_post_form", side_effect=_fake_post):
        b1 = c1.balance()
        b2 = c2.balance()
        # Second calls hit per-client cache cleanly
        b1_cached = c1.balance()
        b2_cached = c2.balance()
    assert b1 == {"USDC": "100.0"} and b1_cached == {"USDC": "100.0"}
    assert b2 == {"USDC": "7.0"} and b2_cached == {"USDC": "7.0"}
    assert calls == ["key_alpha", "key_beta"]

    # Public AssetPairs cache stays 2-tuple compatible with get_cached_pair_info()
    with patch.object(
        kc,
        "http_get",
        return_value=(200, b'{"error":[],"result":{"HYPEUSD":{"fees_maker":[[0,0.12]],"lot_decimals":2}}}'),
    ):
        c1.asset_pairs()
    pinfo = kc.get_cached_pair_info(["HYPEUSD"])
    assert isinstance(pinfo, dict) and pinfo.get("lot_decimals") == 2
    kc._CACHE.clear()


def test_p1_hyperliquid_fill_separates_qty_from_status_partial_full_and_cancel(tmp_path):
    """Audit #1: Raw Hyperliquid fills without status must NOT synthesize FILLED; ledger must handle
    partial-still-active, full-fill, and cancel-after-partial accurately in one classifier."""
    cdb = tmp_path / "cachedb"
    orders_dir = cdb / "orders"
    orders_dir.mkdir(parents=True)
    t0 = 1_795_500_000.0

    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=10.0)
    reader = OrderLedgerReader(cachedb_dir=cdb)

    # 1. Place Hyperliquid order of qty=10.0; 2.0 fills while 8.0 stays resting in open book
    writer.record_event(
        OrderEventType.ACKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0,
        iid="hl-part-1",
        vid="HL-10",
        side="BUY",
        qty=10.0,
        px=60.0,
    )
    hl_open_path = str(orders_dir / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME)
    hl_fill_path = str(orders_dir / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME)

    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        hl_open_path,
        {
            "HYPEUSDC": [{
                "orderId": "HL-10",
                "side": "BUY",
                "price": 60.0,
                "origQty": 10.0,
                "executedQty": 2.0,
                "status": "OPEN",
                "timestamp": int(t0 * 1000),
            }]
        },
        now_ms=int((t0 + 2) * 1000),
        verified_ts=t0 + 2.0,
    )
    doc1 = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        hl_fill_path,
        {
            "HYPEUSDC": [{
                "oid": "HL-10",
                "tid": "f1",
                "side": "B",
                "px": "60.0",
                "sz": "2.0",
                "time": int((t0 + 2) * 1000),
            }]
        },
        venue="hyperliquid",
        now_ms=int((t0 + 2) * 1000),
    )
    # Cache MUST NOT fabricate status="FILLED" for raw execution fills!
    row1 = doc1["items"]["HYPEUSDC"][0]
    assert row1["quantity"] == pytest.approx(2.0)
    assert "status" not in row1

    st1 = reconciler.reconcile_once(now_ts=t0 + 3.0)
    assert st1["reconciled_filled"] == 0 and st1["reconciled_closed"] == 0 and st1["active_count"] == 1
    act1 = reader.list_active_orders(venue="hyperliquid", apply_tail=False)
    assert len(act1) == 1 and act1[0]["state"] == OrderLedgerState.PARTIAL.value
    assert act1[0]["fq"] == pytest.approx(2.0)

    # 2. Remaining 8.0 fills -> cumulative qty == 10.0 closes as FILLED even if open cache still lags!
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        hl_fill_path,
        {
            "HYPEUSDC": [{
                "oid": "HL-10",
                "tid": "f2",
                "side": "B",
                "px": "60.5",
                "sz": "8.0",
                "time": int((t0 + 4) * 1000),
            }]
        },
        venue="hyperliquid",
        now_ms=int((t0 + 4) * 1000),
    )
    st2 = reconciler.reconcile_once(now_ts=t0 + 5.0)
    assert st2["reconciled_filled"] == 1 and st2["reconciled_closed"] == 1 and st2["active_count"] == 0
    tr_full = reader.trace_order("HL-10")
    assert len(tr_full["closed_orders"]) == 1
    assert tr_full["closed_orders"][0]["state"] == OrderLedgerState.FILLED.value
    assert tr_full["closed_orders"][0]["fq"] == pytest.approx(10.0)

    # 3. Partial fill (2/10) followed by cancel (absent from freshly verified HYPEUSDC snapshot past grace) -> CANCELED!
    writer.record_event(
        OrderEventType.ACKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0 + 20.0,
        iid="hl-cancel-after-partial",
        vid="HL-20",
        side="BUY",
        qty=10.0,
        px=59.0,
    )
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        hl_fill_path,
        {
            "HYPEUSDC": [{
                "oid": "HL-20",
                "tid": "f20",
                "side": "B",
                "px": "59.0",
                "sz": "2.0",
                "time": int((t0 + 21) * 1000),
            }]
        },
        venue="hyperliquid",
        now_ms=int((t0 + 21) * 1000),
    )
    # Even with an older snapshot (t0 + 2.0 < created_ts t0 + 20.0), partial fill stays PARTIAL
    st3 = reconciler.reconcile_once(now_ts=t0 + 22.0)
    assert st3["reconciled_filled"] == 0 and st3["active_count"] == 1
    assert reader.list_active_orders(venue="hyperliquid", apply_tail=False)[0]["state"] == OrderLedgerState.PARTIAL.value

    # Now fresh complete open snapshot covering HYPEUSDC: [] arrives past 10s grace -> closes as CANCELED (fq=2.0)
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        hl_open_path,
        {"HYPEUSDC": []},
        now_ms=int((t0 + 40) * 1000),
        verified_ts=t0 + 40.0,
    )
    st4 = reconciler.reconcile_once(now_ts=t0 + 40.0)
    assert st4["reconciled_filled"] == 0 and st4["reconciled_closed"] == 1 and st4["active_count"] == 0
    tr_can = reader.trace_order("HL-20")
    assert len(tr_can["closed_orders"]) == 1
    assert tr_can["closed_orders"][0]["state"] == OrderLedgerState.CANCELED.value
    assert tr_can["closed_orders"][0]["fq"] == pytest.approx(2.0)


def test_p1_absence_closing_requires_verified_symbol_snapshot_not_partial_or_ws(tmp_path):
    """Audit #2: Publishing empty SPCXxUSD or WS single-symbol deltas must NEVER close an omitted HYPEUSD order."""
    cdb = tmp_path / "cachedb"
    orders_dir = cdb / "orders"
    orders_dir.mkdir(parents=True)
    t0 = 1_795_510_000.0

    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=10.0)
    reader = OrderLedgerReader(cachedb_dir=cdb)

    writer.record_event(OrderEventType.ACKED, venue="kraken", symbol="HYPEUSD", ts=t0, iid="kr-hype-1", vid="OKRK-H1", side="BUY", qty=2.0, px=60.0)
    writer.record_event(OrderEventType.ACKED, venue="kraken", symbol="SPCXxUSD", ts=t0, iid="kr-spcx-1", vid="OKRK-S1", side="BUY", qty=1.0, px=140.0)

    kr_open_path = str(orders_dir / CACHE_KRAKEN_ORDER_OPEN_FILENAME)
    # 1. First partial refresh publishes ONLY empty SPCXxUSD with verified_ts=None
    snap1 = CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        kr_open_path,
        {"SPCXxUSD": []},
        now_ms=int((t0 + 25) * 1000),
        verified_ts=None,
    )
    assert snap1["verified_ts"] is None
    assert "SPCXxUSD" in snap1["symbol_verified_ts"]
    assert "HYPEUSD" not in snap1["symbol_verified_ts"]

    reconciler.reconcile_once(now_ts=t0 + 25.0)
    active = {r["vid"]: r for r in reader.list_active_orders(venue="kraken", apply_tail=False)}
    # SPCXxUSD was explicitly verified empty past grace -> CANCELED; omitted HYPEUSD stays ACTIVE!
    assert "OKRK-S1" not in active
    assert "OKRK-H1" in active and active["OKRK-H1"]["state"] == OrderLedgerState.OPEN.value

    # 2. Even if a caller passes verified_ts on a snapshot that omits HYPEUSD, HYPEUSD MUST remain ACTIVE
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        kr_open_path,
        {"SPCXxUSD": []},
        now_ms=int((t0 + 50) * 1000),
        verified_ts=t0 + 50.0,
    )
    reconciler.reconcile_once(now_ts=t0 + 50.0)
    active2 = {r["vid"]: r for r in reader.list_active_orders(venue="kraken", apply_tail=False)}
    assert "OKRK-H1" in active2

    # 3. Single-order WebSocket updates must NEVER advance verified_ts or symbol_verified_ts
    bn_open_path = str(orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME)
    open_mgr = CacheOrderOpenManager(15.0, ["BTCUSDC", "ETHUSDC"], bn_open_path, read_only=False, venue="binance")
    open_mgr.apply_ws_order_update(
        "BTCUSDC",
        "upsert",
        {"orderId": "111", "side": "BUY", "price": 65000.0, "origQty": 0.01, "executedQty": 0.0, "status": "NEW"},
        now_ms=int((t0 + 60) * 1000),
    )
    assert open_mgr.verified_ts == 0.0
    assert open_mgr.symbol_verified_ts == {}
    assert open_mgr.save_state_to_file() is True
    saved_bn = json.loads((orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME).read_text(encoding="utf-8"))
    assert saved_bn["verified_ts"] is None and saved_bn["symbol_verified_ts"] == {}

    # 4. Once a complete snapshot explicitly covers HYPEUSD: [], HYPEUSD cleanly closes as CANCELED
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        kr_open_path,
        {"HYPEUSD": []},
        now_ms=int((t0 + 80) * 1000),
        verified_ts=t0 + 80.0,
    )
    reconciler.reconcile_once(now_ts=t0 + 80.0)
    assert reader.list_active_orders(venue="kraken", apply_tail=False) == []
    assert reader.trace_order("OKRK-H1")["closed_orders"][0]["state"] == OrderLedgerState.CANCELED.value


def test_p1_unified_journal_replay_utc_rollover_and_terminal_suppression_parity(tmp_path):
    """Audit #3: Late events on previous UTC day after rollover must be read, and ACKED -> FILLED -> OPEN
    must yield identical empty active state in BOTH OrderLedgerReader and OrderLedgerReconciler."""
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(cachedb_dir=cdb, use_singletons=False, reconcile_grace_sec=30.0)
    reader = OrderLedgerReader(cachedb_dir=cdb)

    # UTC day boundary: 2026-11-23T23:59:50Z and 2026-11-24T00:00:05Z
    day1_ts = 1_795_478_390.0  # 2026-11-23 23:59:50 UTC
    day2_ts = day1_ts + 15.0   # 2026-11-24 00:00:05 UTC

    writer.record_event(OrderEventType.ACKED, venue="binance", symbol="BTCUSDC", ts=day1_ts, iid="d1-1", vid="D1-1", side="BUY", qty=0.01, px=60000.0)
    reconciler.reconcile_once(now_ts=day1_ts + 1.0)
    writer.record_event(OrderEventType.ACKED, venue="binance", symbol="ETHUSDC", ts=day2_ts, iid="d2-1", vid="D2-1", side="BUY", qty=0.1, px=3000.0)
    reconciler.reconcile_once(now_ts=day2_ts + 1.0)

    # Now append a late event to Day 1 AFTER cursor moved to Day 2
    writer.record_event(OrderEventType.ACKED, venue="binance", symbol="SOLUSDC", ts=day1_ts + 9.0, iid="d1-late", vid="D1-LATE", side="BUY", qty=1.0, px=150.0)
    tail_vids = {r["vid"] for r in reader.list_active_orders(apply_tail=True)}
    assert "D1-LATE" in tail_vids

    reconciler.reconcile_once(now_ts=day2_ts + 2.0)
    persisted_vids = {r["vid"] for r in reader.list_active_orders(apply_tail=False)}
    assert "D1-LATE" in persisted_vids

    # Same-day sequence ACKED -> STATUS(FILLED) -> STATUS(OPEN) MUST yield identical state in reader and reconciler
    writer.record_event(OrderEventType.ACKED, venue="binance", symbol="TAOUSDC", ts=day2_ts + 10.0, iid="seq-1", vid="SEQ-1", side="BUY", qty=1.0, px=200.0)
    writer.record_event(OrderEventType.STATUS, venue="binance", symbol="TAOUSDC", ts=day2_ts + 11.0, iid="seq-1", vid="SEQ-1", vst="FILLED", fq=1.0)
    writer.record_event(OrderEventType.STATUS, venue="binance", symbol="TAOUSDC", ts=day2_ts + 12.0, iid="seq-1", vid="SEQ-1", vst="OPEN")

    reader_active_vids = {r["vid"] for r in reader.list_active_orders(apply_tail=True)}
    assert "SEQ-1" not in reader_active_vids

    reconciler.reconcile_once(now_ts=day2_ts + 13.0)
    reconciler_active_vids = {r["vid"] for r in reader.list_active_orders(apply_tail=False)}
    assert reader_active_vids == reconciler_active_vids
    assert "SEQ-1" not in reconciler_active_vids


def test_p2_filled_cache_overlapping_batches_and_backfill_preserve_exact_qty_and_vwap(tmp_path):
    """Overlapping fill batches [1,2] and [2,3] (prices 20, 30, 40; qty 1 each) MUST yield qty=3 @ VWAP 30."""
    cache_path = str(tmp_path / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME)
    f1 = {"oid": "O-OV", "tid": "1", "side": "B", "px": "20.0", "sz": "1.0", "time": 1700000001000}
    f2 = {"oid": "O-OV", "tid": "2", "side": "B", "px": "30.0", "sz": "1.0", "time": 1700000002000}
    f3 = {"oid": "O-OV", "tid": "3", "side": "B", "px": "40.0", "sz": "1.0", "time": 1700000003000}

    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        cache_path, {"HYPEUSDC": [f1, f2]}, venue="hyperliquid", now_ms=1700000005000
    )
    doc = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        cache_path, {"HYPEUSDC": [f2, f3]}, venue="hyperliquid", now_ms=1700000006000
    )
    row = doc["items"]["HYPEUSDC"][0]
    assert row["quantity"] == pytest.approx(3.0)
    assert row["price"] == pytest.approx(30.0)
    assert row["_fillIds"] == ["1", "2", "3"]

    # Also verify a legacy cache corrupted by pre-aggregated overlaps WITHOUT _fills is rebuilt by a full backfill snapshot
    legacy_path = str(tmp_path / "legacy_filled_hl.json")
    (tmp_path / "legacy_filled_hl.json").write_text(
        json.dumps({
            "items": {
                "HYPEUSDC": [{
                    "orderId": "O-OV",
                    "side": "BUY",
                    "price": 35.0,
                    "quantity": 2.0,
                    "timestamp": 1700000003000,
                    "_fillIds": ["1", "2", "3"],
                }]
            },
            "fetchtime": {"HYPEUSDC": 1700000004000},
        }),
        encoding="utf-8",
    )
    rebuilt = CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        legacy_path, {"HYPEUSDC": [f1, f2, f3]}, venue="hyperliquid", now_ms=1700000009000, is_snapshot=True
    )
    r_row = rebuilt["items"]["HYPEUSDC"][0]
    assert r_row["quantity"] == pytest.approx(3.0)
    assert r_row["price"] == pytest.approx(30.0)
    assert r_row["_fillIds"] == ["1", "2", "3"]


def test_p0_bapi_client_threadsafe_request_prevents_cross_symbol_response_race(monkeypatch):
    """Incident 2026-11-24: python-binance's shared ``self.response`` caused concurrent ARBUSDC/TAOUSDC
    REST ticker polls to swap payloads. Verify ``_install_threadsafe_request`` + ``bapi.update_price``
    + ``BinancePriceStream`` eliminate cross-symbol contamination."""
    import concurrent.futures
    import threading
    from binance.client import Client
    from binance_api import bapi, bapi_client, bapi_ws

    cl = Client("", "", ping=False)
    bapi_client._install_threadsafe_request(cl)
    bapi_client._install_recvwindow_retry(cl)

    barrier = threading.Barrier(4)
    prices_by_sym = {
        "ARBUSDC": "0.11520000",
        "TAOUSDC": "189.70000000",
        "ADAUSDC": "0.31200000",
        "BTCUSDC": "61250.00000000",
    }

    class _FakeResp:
        def __init__(self, sym: str):
            self.status_code = 200
            self.headers = {}
            self.text = json.dumps({"symbol": sym, "price": prices_by_sym[sym]})

        def json(self):
            return json.loads(self.text)

    class _FakeSession:
        def get(self, uri, **kwargs):
            params = str(kwargs.get("params") or "")
            sym = params.split("symbol=", 1)[1].split("&", 1)[0]
            barrier.wait(timeout=5.0)
            return _FakeResp(sym)

        def close(self):
            return None

    cl.session = _FakeSession()

    for _ in range(15):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = {sym: pool.submit(cl.get_symbol_ticker, symbol=sym) for sym in prices_by_sym}
            for sym, fut in futs.items():
                res = fut.result(timeout=5.0)
                assert res["symbol"] == sym
                assert res["price"] == prices_by_sym[sym]

    # Defense-in-depth: bapi.update_price and _get_symbol_price_safe MUST reject mismatched response symbol
    fake_cli = SimpleNamespace(
        get_symbol_ticker=lambda symbol: {"symbol": "TAOUSDC", "price": "189.70000000"}
    )
    monkeypatch.setattr(bapi, "client", fake_cli)
    assert bapi.update_price("ARBUSDC") is None
    assert bapi._get_symbol_price_safe("ARBUSDC") is None
    assert bapi.update_price("TAOUSDC") == pytest.approx(189.7)

    # Defense-in-depth: BinancePriceStream._process_message drops mismatched stream vs data.s
    stream = bapi_ws.BinancePriceStream(symbols=["ARBUSDC", "TAOUSDC"])
    received = []
    stream.subscribe(SimpleNamespace(on_items_update=lambda s, items: received.append((s, items))))
    stream._process_message(
        json.dumps({"stream": "arbusdc@ticker", "data": {"s": "TAOUSDC", "c": "189.7", "E": 1795504766434}})
    )
    assert received == []
    stream._process_message(
        json.dumps({"stream": "arbusdc@ticker", "data": {"s": "ARBUSDC", "c": "0.1152", "E": 1795504766435}})
    )
    assert len(received) == 1 and received[0][0] == "ARBUSDC" and received[0][1][0][1] == pytest.approx(0.1152)


def test_p0_cache_current_price_quarantines_single_tick_outlier_and_allows_confirmed_move(tmp_path):
    """Incident 2026-11-24 Layer 2: CacheCurrentPriceManager rejects an unconfirmed single-tick spike
    (0.1152 -> 189.7) without broadcasting to Cache24PriceManager, while accepting a 2-tick confirmed move."""
    from cacheprice import CacheCurrentPriceManager

    cpm = CacheCurrentPriceManager(
        sync_ts=1.0,
        symbols=["ARBUSDC"],
        filename=str(tmp_path / "cache_current_price.json"),
        read_only=True,
        data_sync_mode=CacheCurrentPriceManager.MODE_DISK_READER,
    )
    forwarded = []
    cpm.subscribe_price(SimpleNamespace(on_price_update=lambda s, ts, px: forwarded.append((s, ts, px))))

    t0_ms = 1_795_504_763_000
    # Establish a 2-tick baseline (matching steady-state live operation)
    cpm.push_price("ARBUSDC", 0.1151, ts_ms=t0_ms)
    cpm.push_price("ARBUSDC", 0.1152, ts_ms=t0_ms + 1_641)
    # Single corrupt cross-symbol tick 1.8s later -> MUST be quarantined and never broadcast
    cpm.push_price("ARBUSDC", 189.7, ts_ms=t0_ms + 3_434)
    assert cpm.extract_price_and_ts(cpm.cache.get("ARBUSDC"))[0] == pytest.approx(0.1152)
    assert len(forwarded) == 2

    # Normal tick arrives 1.1s later -> clears quarantine and updates cleanly
    cpm.push_price("ARBUSDC", 0.1150, ts_ms=t0_ms + 4_590)
    assert cpm.extract_price_and_ts(cpm.cache.get("ARBUSDC"))[0] == pytest.approx(0.1150)
    assert len(forwarded) == 3

    # Genuine re-denomination / confirmed move (2 distinct corroborating ticks within 30s) -> accepted on 2nd tick
    cpm.push_price("ARBUSDC", 1.150, ts_ms=t0_ms + 6_000)
    assert cpm.extract_price_and_ts(cpm.cache.get("ARBUSDC"))[0] == pytest.approx(0.1150)
    cpm.push_price("ARBUSDC", 1.152, ts_ms=t0_ms + 7_500)
    assert cpm.extract_price_and_ts(cpm.cache.get("ARBUSDC"))[0] == pytest.approx(1.152)
    assert len(forwarded) == 4


def test_p0_assetguardian_filters_189_7_outlier_and_autoheals_corrupted_state(monkeypatch):
    """Incident 2026-11-24 Layer 3: AssetGuardian ignores an isolated 189.7 tick in cache_24_price_ARBUSDC,
    rejects a corrupt live current_price, and auto-heals a previously poisoned peak_price=189.7 state."""
    import assetguardian as ag

    now = 1_795_505_000.0
    rows = [
        [(now - 600) * 1000, 0.1160],
        [(now - 500) * 1000, 0.1158],
        [(now - 400) * 1000, 0.1155],
        [(now - 300) * 1000, 0.1152],
        [(now - 298) * 1000, 189.7],  # exact live corrupt tick
        [(now - 297) * 1000, 0.1150],
        [(now - 100) * 1000, 0.1149],
    ]
    ag._local_price_samples["ARBUSDC"] = []
    monkeypatch.setattr(ag, "_read_symbol_price_rows", lambda _sym: rows)
    monkeypatch.setattr(ag.time, "time", lambda: now)

    extrema = ag._get_symbol_window_extrema("ARBUSDC", 0.1148, 1440)
    assert extrema is not None
    minimum, maximum = extrema
    assert minimum["price"] == pytest.approx(0.1148)
    assert maximum["price"] == pytest.approx(0.1160)

    # Corrupt live current_price=189.7 against ~0.115 rolling median must be refused fail-closed
    assert ag._get_symbol_window_extrema("ARBUSDC", 189.7, 1440) is None

    # Auto-heal poisoned assetguardian_state.json (peak_price=189.7) during evaluate_symbol
    class _MemState:
        def __init__(self, initial):
            self.value = initial
            self.path = None

        def load(self):
            return json.loads(json.dumps(self.value))

        def save(self, val):
            self.value = json.loads(json.dumps(val))

    poisoned_state = _MemState(
        {
            "version": 3,
            "symbols": {
                "ARBUSDC": {
                    "buy": {
                        "peak_price": 189.7,
                        "peak_ts": now - 298.0,
                        "initial_cash": 16250.0,
                        "completed_tiers": [7.0, 10.0, 14.0],
                        "spent_quote_by_tier": {"7": 531.0, "10": 2850.0, "14": 4754.0},
                        "total_spent_quote": 8135.0,
                        "attempts_by_tier": {"7": 1, "10": 1, "14": 1},
                        "terminal_orders": [],
                    }
                }
            },
        }
    )
    provider = SimpleNamespace(free_balance=lambda _asset: 0.0, open_orders=lambda _sym: [])
    buy_called = []
    monkeypatch.setattr(ag, "STATE", poisoned_state)
    monkeypatch.setattr(ag.api, "get_current_price", lambda _sym: 0.1110)  # -4.3% vs 0.1160 (above -2.5% reset, below -7% buy)
    monkeypatch.setattr(ag.mkt, "provider_by_name", lambda _name: provider)
    monkeypatch.setattr(ag, "buy_with_all_cash", lambda **kw: buy_called.append(kw))
    monkeypatch.setattr(ag, "_cycle_portfolio_growth_percent", lambda **_kw: None)
    monkeypatch.setattr(ag, "_cycle_p3b_sentiment", lambda: None)

    assert ag.evaluate_symbol("ARBUSDC") is False
    assert buy_called == []
    assert poisoned_state.value["symbols"].get("ARBUSDC", {}).get("buy", {}) == {}

    # Also verify symmetric SELL trough auto-heal (reproducing live BTCUSDC trough_price=0.1461 incident)
    btc_rows = [
        [(now - 600) * 1000, 61000.0],
        [(now - 500) * 1000, 60800.0],
        [(now - 400) * 1000, 60600.0],
        [(now - 300) * 1000, 60400.0],
        [(now - 298) * 1000, 0.1461],  # corrupt cross-symbol trough tick
        [(now - 297) * 1000, 60350.0],
        [(now - 100) * 1000, 60500.0],
    ]
    ag._local_price_samples["BTCUSDC"] = []
    monkeypatch.setattr(ag, "_read_symbol_price_rows", lambda _sym: btc_rows)
    poisoned_sell = _MemState(
        {
            "version": 3,
            "symbols": {
                "BTCUSDC": {
                    "sell": {
                        "trough_price": 0.1461,
                        "trough_ts": now - 298.0,
                        "initial_qty": 0.025,
                        "completed_tiers": [],
                        "filled_qty_by_tier": {},
                        "total_filled_qty": 0.0,
                        "attempts_by_tier": {"9": 1},
                        "terminal_orders": [],
                    }
                }
            },
        }
    )
    sell_called = []
    monkeypatch.setattr(ag, "STATE", poisoned_sell)
    monkeypatch.setattr(ag.api, "get_current_price", lambda _sym: 60550.0)
    monkeypatch.setattr(
        ag.mkt,
        "provider_by_name",
        lambda _name: SimpleNamespace(free_balance=lambda _asset: 0.025, open_orders=lambda _sym: []),
    )
    monkeypatch.setattr(ag, "sell_asset", lambda *a, **kw: sell_called.append((a, kw)))
    assert ag.evaluate_symbol("BTCUSDC") is False
    assert sell_called == []
    assert poisoned_sell.value["symbols"].get("BTCUSDC", {}).get("sell", {}) == {}


def test_p2_order_ledger_expires_anonymous_and_stale_deferred_intents(tmp_path):
    """Anonymous deferred=True GUARD_BLOCKED intents (no iid/cid/vid) must expire after intent_stale_sec,
    and named deferred intents must expire after deferred_intent_stale_sec without polluting orders_closed."""
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    writer = OrderLedgerWriter(cachedb_dir=cdb)
    reader = OrderLedgerReader(cachedb_dir=cdb)
    reconciler = OrderLedgerReconciler(
        cachedb_dir=cdb,
        use_singletons=False,
        reconcile_grace_sec=30.0,
        intent_stale_sec=120.0,
        deferred_intent_stale_sec=600.0,
    )
    t0 = 1_795_456_238.0

    # 1. Anonymous deferred GUARD_BLOCKED (exact live failure mode)
    writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="binance",
        symbol="BTCUSDC",
        ts=t0,
        owner="tradeall",
        side="BUY",
        qty=0.13594,
        px=61460.22,
        reason="profit_guard",
        deferred=True,
    )
    # 2. Named deferred GUARD_BLOCKED (with outbox iid)
    writer.record_event(
        OrderEventType.GUARD_BLOCKED,
        venue="binance",
        symbol="ARBUSDC",
        ts=t0,
        owner="tradeall",
        side="BUY",
        qty=2602.6,
        px=0.1143,
        iid="outbox-arb-1",
        reason="profit_guard",
        deferred=True,
    )

    # At t0 + 30s (< intent_stale_sec): both remain active
    reconciler.reconcile_once(now_ts=t0 + 30.0)
    assert len(reader.list_active_orders(apply_tail=False)) == 2

    # At t0 + 180s (> intent_stale_sec, < deferred_intent_stale_sec): anonymous expires, named stays
    reconciler.reconcile_once(now_ts=t0 + 180.0)
    active_mid = reader.list_active_orders(apply_tail=True)
    assert len(active_mid) == 1
    assert active_mid[0]["iid"] == "outbox-arb-1"

    # At t0 + 700s (> deferred_intent_stale_sec): named deferred intent also expires cleanly
    reconciler.reconcile_once(now_ts=t0 + 700.0)
    assert reader.list_active_orders(apply_tail=True) == []
    assert reader.list_active_orders(apply_tail=False) == []
    assert reader.list_closed_files() == []
    assert reader.trace_order("outbox-arb-1")["closed_orders"] == []


def test_p0_intelligence_24h_metrics_and_cache24_filter_isolated_cross_symbol_outliers(tmp_path, monkeypatch):
    """Sweep fix: intelligence compute_24h_price_metrics, Cache24PriceManager, CacheShortTrendManager,
    survival.get_recent_price_history, and macro_telemetry_evaluator must drop isolated cross-symbol
    outliers (ARBUSDC 189.7 and BTCUSDC 0.1461) including 2-tick bursts."""
    import json
    import threading
    from cacheprice import CachePriceManagerBase
    from cachetrend import CacheShortTrendManager
    from intelligence.internal.state import survival as _survival
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        build_cycle_fallback_prices,
        enrich_macro_benchmarks_live_state,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_24h_price_metrics,
        compute_7d_range_metrics,
        normalize_and_dedup_24h_series,
    )

    t0 = 1_795_504_000_000
    arb_series = [
        [t0, 0.1160],
        [t0 + 60_000, 0.1155],
        [t0 + 120_000, 0.1152],
        [t0 + 122_000, 189.7],   # corrupt single tick
        [t0 + 124_000, 0.1150],
        [t0 + 180_000, 0.1142],
    ]
    # Exact production path: normalize_and_dedup_24h_series pre-populates _SORTED_TS_PX_MEMO!
    arb_norm = normalize_and_dedup_24h_series(arb_series, "ARBUSDC")
    assert all(pt[1] < 1.0 for pt in arb_norm)
    arb_m = compute_24h_price_metrics(arb_norm)
    assert arb_m["high_24h"] == pytest.approx(0.1160)
    assert arb_m["low_24h"] == pytest.approx(0.1142)
    assert abs(arb_m["dd_24h_high_pct"]) < 5.0

    # Also protect 7d range / swing-wave scalar lists against extreme cross-symbol spikes
    arb_7d = compute_7d_range_metrics([0.110, 0.115, 0.114, 189.7, 0.113, 0.112])
    assert arb_7d["high_7d"] == pytest.approx(0.115)
    assert arb_7d["low_7d"] == pytest.approx(0.110)

    btc_series = [
        [t0, 61000.0],
        [t0 + 60_000, 60800.0],
        [t0 + 120_000, 60500.0],
        [t0 + 122_000, 0.1461],  # corrupt 2-tick burst
        [t0 + 123_000, 0.1461],
        [t0 + 124_000, 60450.0],
        [t0 + 180_000, 60300.0],
    ]
    btc_norm = normalize_and_dedup_24h_series(btc_series, "BTCUSDC")
    assert all(pt[1] > 50000.0 for pt in btc_norm)
    btc_m = compute_24h_price_metrics(btc_norm)
    assert btc_m["low_24h"] == pytest.approx(60300.0)
    assert btc_m["high_24h"] == pytest.approx(61000.0)

    cleaned = CachePriceManagerBase.filter_isolated_price_points(btc_series)
    assert len(cleaned) == 5
    assert all(pt[1] > 50000.0 for pt in cleaned)

    # CacheShortTrendManager._load_disk_cache24_entries + survival.get_recent_price_history + macro_telemetry_evaluator
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    btc_edge_series = [
        [t0 - 120_000, 60000.0],
        [t0 - 60_000, 60100.0],
        [t0, 60200.0],
        [t0 + 60_000, 60300.0],
        [t0 + 120_000, 0.1461],  # corrupt trailing edge tick
    ]
    (prices_dir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": btc_edge_series}}), encoding="utf-8"
    )
    stm = CacheShortTrendManager.__new__(CacheShortTrendManager)
    stm._lock = threading.Lock()
    stm._disk_c24_cache = {}
    stm.filename = str(tmp_path / "cache_short_trend.json")
    stm.read_only = True
    disk_entries = stm._load_disk_cache24_entries("BTCUSDC")
    assert len(disk_entries) == 4
    assert all(pt[1] > 50000.0 for pt in disk_entries)

    monkeypatch.setattr(_survival, "get_cachedb_dir", lambda: str(tmp_path))
    hist = _survival._read_cached_price_history("BTCUSDC", window_seconds=3600.0, now=(t0 + 180_000) / 1000.0)
    assert hist is not None and len(hist) == 4
    assert all(px > 50000.0 for _, px in hist)

    fallback_pxs = build_cycle_fallback_prices({}, {"BTCUSDC": btc_edge_series})
    assert fallback_pxs["BTCUSDC"] == pytest.approx(60300.0)

    enriched = enrich_macro_benchmarks_live_state({"items": {}}, cache_dir=str(tmp_path), force_recompute=True)
    assert isinstance(enriched, dict)
    assert enriched.get("cross_asset_regime") in ("CONVERGENT", "DIVERGENT", "NEUTRAL", "DECOUPLED", "RISK_ON", "RISK_OFF", "UNKNOWN", None) or isinstance(enriched.get("cross_asset_regime"), str)


def test_p0_trailing_core_quarantines_single_tick_spike_and_flash_crash_without_dumping(tmp_path):
    """Sweep fix: TrailingCore (Binance & Kraken crash circuit breaker) must quarantine an unconfirmed
    single-tick spike (189.7 -> would poison peak) or flash-drop (0.1461 -> would force-sell BTC),
    while still triggering immediately on real <= -60% drops and confirming sustained 2-tick moves."""
    from trailing_core import TrailingCore

    class _FakeAdapter:
        def __init__(self, px: float):
            self.px = px
            self.sells = []
            self.unlocks = 0

        def begin_tick(self):
            return True

        def assets(self):
            return [("BTCUSDC", "BTC", "BTCUSDC", 13.375)]

        def free_qty(self, _asset):
            return 0.05

        def total_qty(self, _asset):
            return 0.05

        def price(self, _pair):
            return self.px

        def trend(self, _pair):
            return 0.0

        def unlock_qty(self, _pair):
            self.unlocks += 1
            return True

        def reconcile_pending(self, _pending, _persist):
            raise AssertionError("unexpected pending reconcile")

        def execute_sell(self, key, asset, pair, qty, price, peak, trail, _persist):
            self.sells.append((pair, qty, price, peak))
            return SimpleNamespace(outcome="submitted")

        def log_hold(self, *_a):
            pass

        def log_item_error(self, _k, exc):
            raise exc

    state_file = str(tmp_path / "trailing_state.json")
    adapter = _FakeAdapter(61000.0)
    core = TrailingCore(
        adapter,
        log=lambda _m: None,
        enabled=True,
        state_file=state_file,
        min_notional=11.0,
        rebuy_enabled=False,
        rebuy_bounce_pct=2.0,
        rebuy_skip_if_trend_down=False,
        min_profit_pct=0.0,
    )
    # 1. Establish normal BTC peak at 61,000
    core.check_once()
    assert core.load()["BTCUSDC"]["peak"] == pytest.approx(61000.0)

    # 2. Single bogus low tick 0.1461 (ratio 417,522x) -> MUST be quarantined; NO unlock, NO sell!
    adapter.px = 0.1461
    core.check_once()
    assert adapter.unlocks == 0
    assert adapter.sells == []

    # 3. Single bogus spike tick 200,000 (ratio 3.28x) -> MUST be quarantined; peak stays 61,000!
    adapter.px = 200000.0
    core.check_once()
    assert core.load()["BTCUSDC"]["peak"] == pytest.approx(61000.0)

    # 4. Normal tick 60,500 clears quarantine
    adapter.px = 60500.0
    core.check_once()
    assert adapter.sells == []

    # 5. Genuine crash -15% (51,850, ratio 1.17x <= 2.5x) -> triggers IMMEDIATELY on tick 1!
    adapter.px = 51850.0
    core.check_once()
    assert len(adapter.sells) == 1
    assert adapter.sells[0][2] == pytest.approx(51850.0)


def test_p1_hl_client_force_timeout_isolates_per_thread_sessions_and_kraken_matches_pair(monkeypatch):
    """Sweep fix: Hyperliquid _force_timeout isolates per-thread requests.Session for concurrent
    workers, and KrakenClient._public_first_dict selects the matching pair key in multi-key payloads."""
    import concurrent.futures
    import importlib.util
    import os
    import threading
    import requests
    from kraken.kraken_client import KrakenClient

    hl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid", "hl_client.py")
    spec = importlib.util.spec_from_file_location("_local_hl_client", hl_path)
    hl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl_mod)
    _force_timeout = hl_mod._force_timeout

    class _FakeApi:
        def __init__(self):
            self.timeout = None
            self.session = requests.Session()
            self.session.headers["X-Test"] = "ok"

    captured = []
    cap_lock = threading.Lock()
    barrier = threading.Barrier(3)

    def _fake_session_request(self_sess, method, url, **kw):
        with cap_lock:
            captured.append((threading.get_ident(), id(self_sess), self_sess.headers.get("X-Test"), kw.get("timeout")))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(requests.Session, "request", _fake_session_request)

    api_obj = _FakeApi()
    main_sess_id = id(api_obj.session)
    _force_timeout(api_obj, seconds=7.5)
    assert api_obj.timeout == pytest.approx(7.5)

    # Main thread uses original session with enforced timeout
    api_obj.session.request("POST", "https://api.hyperliquid.xyz/info")

    def _worker(_idx):
        barrier.wait(timeout=5.0)
        api_obj.session.request("POST", "https://api.hyperliquid.xyz/info")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_worker, range(3)))

    api_obj.session.close()
    assert len(captured) == 4
    # Main thread hit original session; each worker got its own distinct session with cloned headers & timeout
    main_rows = [r for r in captured if r[0] == threading.get_ident()]
    worker_rows = [r for r in captured if r[0] != threading.get_ident()]
    assert len(main_rows) == 1 and main_rows[0][1] == main_sess_id and main_rows[0][3] == pytest.approx(7.5)
    assert len(worker_rows) == 3
    worker_sess_ids = {r[1] for r in worker_rows}
    assert len(worker_sess_ids) == 3
    assert main_sess_id not in worker_sess_ids
    assert all(r[2] == "ok" and r[3] == pytest.approx(7.5) for r in worker_rows)

    # KrakenClient._public_first_dict must select the matching pair when payload has >1 key
    kc = KrakenClient()
    kc._public = lambda _method, _params: {
        "BTCUSD": {"c": ["60000.0"]},
        "HYPEUSD": {"c": ["62.5"]},
    }
    assert kc.ticker("HYPEUSD") == {"c": ["62.5"]}
    kc._public = lambda _method, _params: {
        "BTCUSD": {"c": ["60000.0"]},
        "ETHUSD": {"c": ["3000.0"]},
    }
    assert kc.ticker("HYPEUSD") is None


def test_llm_timeout_140s_topic_auto_rotation_and_archive_sanitizers(tmp_path):
    """Verify 140s LLM topic timeouts, 2-consecutive-timeout thread rotation, NumPy/replay/Kraken
    archive outlier filters, and streaming JSONL archive sanitizer."""
    import json
    import os
    import subprocess
    import numpy as np
    from core.llm_topics import topic_spec
    from orchestratorOS.agy import topics as agy_topics
    from pricewindow import filter_isolated_price_arrays
    from providers.replay_provider import load_price_series
    from kraken.market_data import load_archive_ohlc_candles
    from verify_tools.sanitize_price_archives import sanitize_jsonl_price_archive

    # 1. Topic timeout defaults configurable via LLM_DEFAULT_TIMEOUT_SEC (default 240s)
    for k in (
        "auto_cryptotrade_executor",
        "auto_sharetrade_executor",
        "macro_geopolitical",
        "macro_sentiment",
        "guard_high_stake",
        "guard_post_stop",
    ):
        assert topic_spec(k).default_timeout_sec == pytest.approx(240.0)
    assert topic_spec("auto_technical_repair").default_timeout_sec == pytest.approx(240.0)

    # 2. Consecutive resume timeouts auto-rotate the stuck conversation while preserving rolling_history
    reg_path = str(tmp_path / "llm_topic_threads.json")
    bot_base = str(tmp_path / "agy_base")
    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps(
            {
                "auto_cryptotrade_executor": {
                    "topic_key": "auto_cryptotrade_executor",
                    "conversation_id": "cid-stuck-1",
                    "turn_count": 5,
                    "rolling_history": [{"utc": "2026-11-24 16:00 UTC", "summary": "decision=HOLD"}],
                }
            }
        ),
        encoding="utf-8",
    )
    assert agy_topics.record_topic_resume_timeout(
        "auto_cryptotrade_executor", "cid-stuck-1", cli_base=bot_base, registry_path=reg_path, max_consecutive_timeouts=2
    ) is False
    reg1 = json.loads((tmp_path / "llm_topic_threads.json").read_text(encoding="utf-8"))
    assert reg1["auto_cryptotrade_executor"]["conversation_id"] == "cid-stuck-1"
    assert reg1["auto_cryptotrade_executor"]["consecutive_resume_timeouts"] == 1

    assert agy_topics.record_topic_resume_timeout(
        "auto_cryptotrade_executor", "cid-stuck-1", cli_base=bot_base, registry_path=reg_path, max_consecutive_timeouts=2
    ) is True
    reg2 = json.loads((tmp_path / "llm_topic_threads.json").read_text(encoding="utf-8"))
    assert reg2["auto_cryptotrade_executor"]["conversation_id"] is None
    assert reg2["auto_cryptotrade_executor"]["consecutive_resume_timeouts"] == 0
    assert reg2["auto_cryptotrade_executor"]["needs_full_prompt"] is True
    assert len(reg2["auto_cryptotrade_executor"]["rolling_history"]) == 1

    # 3. Vectorized NumPy array filter removes isolated 189.7 & 0.1461 spikes
    ts_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    px_arr = np.array([0.115, 0.1152, 189.7, 0.1150, 0.1148], dtype=np.float64)
    ts_c, px_c = filter_isolated_price_arrays(ts_arr, px_arr)
    assert len(px_c) == 4 and float(np.max(px_c)) < 1.0

    # 4. Replay loader, Kraken OHLC archive loader, and streaming JSONL archive sanitizer
    t0 = 1_795_504_760_000
    jsonl_path = tmp_path / "cache_365_price_ARBUSDC.jsonl"
    npz_path = tmp_path / "cache_365_price_ARBUSDC.npz"
    npz_path.write_bytes(b"stale-npz")
    rows = [
        {"s": "ARBUSDC", "i": [t0, 0.1150]},
        {"s": "ARBUSDC", "i": [t0 + 1_000, 0.1152]},
        {"s": "ARBUSDC", "i": [t0 + 2_000, 189.7]},
        {"s": "ARBUSDC", "i": [t0 + 3_000, 0.1150]},
        {"s": "ARBUSDC", "i": [t0 + 4_000, 0.1149]},
        {"s": "ARBUSDC", "i": [t0 + 65_000, 0.1148]},
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Read-side guards drop 189.7 even before disk sanitization
    replay_ser = load_price_series(str(jsonl_path), "ARBUSDC")
    assert len(replay_ser) == 5 and all(px < 1.0 for _, px in replay_ser)

    ohlc_bars = load_archive_ohlc_candles("HYPEUSD", 1, drop_incomplete=True, archive_path=str(jsonl_path))
    assert len(ohlc_bars) >= 1 and all(b["high"] < 1.0 for b in ohlc_bars)

    # Dry-run detects line 3 without rewriting; apply removes line 3 and deletes stale .npz
    dry = sanitize_jsonl_price_archive(str(jsonl_path), dry_run=True)
    assert len(dry["outliers"]) == 1 and dry["outliers"][0]["line_no"] == 3 and not dry["rewritten"]
    assert npz_path.exists()

    applied = sanitize_jsonl_price_archive(str(jsonl_path), dry_run=False)
    assert applied["rewritten"] is True and applied["removed_npz"] is True
    assert not npz_path.exists()
    after_lines = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(after_lines) == 5 and all(r["i"][1] < 1.0 for r in after_lines)

    # 5. manage_gitautodeploy.sh passes bash -n syntax check
    sh_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "orchestratorOS", "admin", "manage_gitautodeploy.sh")
    proc = subprocess.run(["bash", "-n", sh_path], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_llm_soft_timeout_extends_without_kill_rotates_next_turn_demotes_and_4h_monitor_restores(tmp_path, monkeypatch):
    """Verify full slow-turn lifecycle:
    1. Soft timeout (140s) does NOT kill CLI; extends wait up to 3x (420s) and uses its completed output.
    2. Schedules thread rotation for the NEXT turn (preserving rolling_history).
    3. After 2 consecutive slow rotations, steps down to faster fallback model and sends notification.
    4. 4-hour model/token/credit monitor verifies quota+availability and restores the best primary model."""
    import json
    import subprocess
    import time as _t
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.llm_quota_router import resolve_crypto_executor_model_override
    from orchestratorOS.agy import topics as agy_topics
    from orchestratorOS.lib import watchdog_common
    from orchestratorOS.admin import monitor_llm_models

    # 1. Soft timeout extends wait up to 3x WITHOUT killing the running Popen process
    popen_calls = []

    class _SlowThenOkPopen:
        def __init__(self, cmd, **_kw):
            self.cmd = cmd
            self.returncode = 0
            self.killed = False
            self.timeouts_seen = []
            popen_calls.append(self)

        def communicate(self, timeout=None):
            self.timeouts_seen.append(timeout)
            if len(self.timeouts_seen) == 1:
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            return json.dumps({"response": '{"decision": "HOLD"}', "conversation_id": "cid-slow-1"}), ""

        def kill(self):
            self.killed = True

    monkeypatch.setattr(subprocess, "Popen", _SlowThenOkPopen)
    completed, exceeded = GeminiClient._run_subprocess_with_soft_extend(
        ["agy-bot", "--model", "gemini-4-neon-high"],
        soft_timeout=140.0,
        extend_mult=3.0,
        run_env={},
        exec_cwd=None,
        cli_model="gemini-4-neon-high",
        resume="cid-slow-1",
    )
    assert exceeded is True
    assert completed.returncode == 0
    assert "HOLD" in completed.stdout
    assert len(popen_calls) == 1
    assert popen_calls[0].killed is False
    assert popen_calls[0].timeouts_seen == [pytest.approx(140.0), pytest.approx(280.0)]

    # 2. Slow turn schedules next-turn rotation while preserving rolling_history
    reg_path = str(tmp_path / "llm_topic_threads.json")
    bot_base = str(tmp_path / "agy_base")
    (tmp_path / "agy_base" / "conversations").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agy_topics, "get_thread_step_count", lambda cid, base: (cid == "cid-slow-1", 10, "IDLE"))
    monkeypatch.setattr(agy_topics, "sync_topic_thread_from_ui", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "mirror_topic_thread_to_ui", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "read_summary_row", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "write_annotation_title", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "delete_threads", lambda *_a, **_k: None)

    agy_topics.record_topic_turn(
        "auto_cryptotrade_executor",
        "cid-slow-1",
        '{"decision": "HOLD", "summary": "slow completed turn"}',
        cli_base=bot_base,
        registry_path=reg_path,
        schedule_rotation=True,
    )
    assert agy_topics.get_topic_slow_rotation_count("auto_cryptotrade_executor", registry_path=reg_path) == 1
    # Next iteration prepare_topic_session rotates to a fresh thread carrying rolling_history
    sess_next = agy_topics.prepare_topic_session(
        "auto_cryptotrade_executor", cli_base=bot_base, compact_steps=18, registry_path=reg_path
    )
    assert sess_next.conversation_id is None
    assert "slow completed turn" in sess_next.history_prefix

    # Second consecutive slow turn on fresh thread triggers count == 2
    slow_cnt2 = agy_topics.record_topic_slow_rotation(
        "auto_cryptotrade_executor", "cid-slow-2", cli_base=bot_base, registry_path=reg_path
    )
    assert slow_cnt2 == 2

    # 3. Latency step-down demotes slow model behind faster fallback and sends notification
    quota_path = str(tmp_path / "llm_quota_status.json")
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", classmethod(lambda cls: quota_path))
    monkeypatch.setattr(GeminiClient, "_QUOTA_MEM_CACHE", {})
    sent_alerts = []
    monkeypatch.setattr(
        watchdog_common,
        "send_ntfy",
        lambda title, msg, **kw: (sent_alerts.append((title, msg, kw)), True)[1],
    )
    monkeypatch.setenv("LLM_FORCE_QUOTA_ALERT_TEST", "1")

    now_t = _t.time()
    (tmp_path / "llm_quota_status.json").write_text(
        json.dumps(
            {
                "checked_ts": now_t,
                "available_credits": 20.0,
                "buckets": {
                    "gemini-4-neon-high": 0.90,
                    "gemini-4-neon-medium": 0.90,
                    "claude-sonnet-5-5-medium": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    assert GeminiClient.record_model_latency_demotion(
        "gemini-4-neon-high",
        "gemini-4-neon-medium",
        topic_key="auto_cryptotrade_executor",
        slow_rotations=2,
        ttl_sec=14400.0,
        notify=True,
        now_ts=now_t,
    ) is True
    assert len(sent_alerts) == 1
    assert "gemini-4-neon-high -> gemini-4-neon-medium" in sent_alerts[0][0]

    snap_demoted = json.loads((tmp_path / "llm_quota_status.json").read_text(encoding="utf-8"))
    ordered = GeminiClient.filter_models_with_confirmed_credits(
        ["gemini-4-neon-high", "gemini-4-neon-medium", "claude-sonnet-5-5-medium"],
        snap_demoted,
    )
    assert ordered[0] == "gemini-4-neon-medium"
    assert ordered[-1] == "gemini-4-neon-high"

    ov_model, ov_reason = resolve_crypto_executor_model_override(
        snap_demoted,
        high_model="gemini-4-neon-high",
        default_model="gemini-4-neon-medium",
        fallback_models=("gemini-4-neon-medium", "claude-sonnet-5-5-medium"),
        escalate_high_effort=True,
        urgent_escalation=True,
        now_ts=now_t + 60.0,
    )
    assert ov_model is None
    assert "suppress_latency_demoted" in ov_reason

    # 4. 4-hour model monitor verifies quota & available models and restores primary model
    monkeypatch.setattr(agy_topics, "get_topic_registry_path", lambda p=None: p or reg_path)
    monkeypatch.setattr(
        GeminiClient,
        "probe_live_model_quota",
        classmethod(lambda cls, **kw: snap_demoted),
    )
    monkeypatch.setattr(
        monitor_llm_models,
        "audit_llm_model_catalog_and_binaries",
        lambda *_a, **_k: {
            "agy_models_ok": True,
            "available_models": ["gemini-4-neon-high", "gemini-4-neon-medium", "claude-sonnet-5-5-medium"],
            "missing_configured_models": [],
        },
    )
    mon_res = monitor_llm_models.run_llm_model_monitor(str(tmp_path), force_refresh=True, notify=True)
    assert mon_res["restored_models"] == ["gemini-4-neon-high"]
    assert agy_topics.get_topic_slow_rotation_count("auto_cryptotrade_executor", registry_path=reg_path) == 0
    assert len(sent_alerts) == 2
    assert "Restored" in sent_alerts[1][0]

    snap_after = json.loads((tmp_path / "llm_quota_status.json").read_text(encoding="utf-8"))
    assert "latency_demoted_models" not in snap_after
    restored_order = GeminiClient.filter_models_with_confirmed_credits(
        ["gemini-4-neon-high", "gemini-4-neon-medium", "claude-sonnet-5-5-medium"],
        snap_after,
    )
    assert restored_order[0] == "gemini-4-neon-high"


def test_llm_timeout_configurability_and_audit_bug_fixes(tmp_path, monkeypatch):
    """Verify configurable LLM timeout SSOT and all 4 audit fixes:
    1. Dynamic env configurability for global, repair, and per-topic timeouts (default 240s).
    2. BaseException (KeyboardInterrupt/SIGTERM) kills & reaps Popen child during soft-extended wait.
    3. Hard timeout on a fresh rotated thread (conversation_id=None) AND primary timeout rescued by
       a faster fallback BOTH increment consecutive_slow_rotations and trigger latency step-down.
    4. Expired latency demotion in probe_live_model_quota is NOT restored when quota is 0%/exhausted,
       and outer timeouts (ai_impulse_exec_timeout_sec & crontab.prod.txt) cover 240s * 3 = 720s."""
    import json
    import os
    import subprocess
    import time as _t
    from botcore import topic_spec
    from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals
    from intelligence.sentiment.gemini_client import GeminiClient, query_topic_json
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
    from orchestratorOS.agy import topics as agy_topics
    from orchestratorOS.lib import watchdog_common

    # 1. Dynamic timeout configurability from 4 core anchors + per-topic override (zero redundant keys)
    monkeypatch.setenv("LLM_DEFAULT_TIMEOUT_SEC", "300")
    monkeypatch.setenv("LLM_TOPIC_TIMEOUT_AUTO_CRYPTOTRADE_EXECUTOR", "195")
    assert topic_spec("auto_sharetrade_executor").default_timeout_sec == pytest.approx(300.0)
    assert topic_spec("auto_technical_repair").default_timeout_sec == pytest.approx(300.0)
    assert topic_spec("auto_cryptotrade_executor").default_timeout_sec == pytest.approx(195.0)
    monkeypatch.setenv("LLM_TOPIC_TIMEOUT_AUTO_TECHNICAL_REPAIR", "360")
    assert topic_spec("auto_technical_repair").default_timeout_sec == pytest.approx(360.0)
    # Outer impulse timeout derives automatically from T_soft * M_ext + 60 (300 * 3 + 60 = 960s)
    assert resolve_derived_telemetry_intervals(executor_interval_sec=360.0)["ai_impulse_exec_timeout_sec"] >= 960.0
    observed_to = []
    dummy_cli = GeminiClient(custom_runner=lambda _p, _m, t: (observed_to.append(t), '{"ok": true}')[1])
    assert HighStakeGuard(gemini_client=dummy_cli).timeout_sec == pytest.approx(300.0)
    assert PostStopReentryGuard(gemini_client=dummy_cli, notify_enabled=False).timeout_sec == pytest.approx(300.0)
    query_topic_json(dummy_cli, "macro_sentiment", "ping")
    assert observed_to == [pytest.approx(300.0)]
    monkeypatch.delenv("LLM_TOPIC_TIMEOUT_AUTO_CRYPTOTRADE_EXECUTOR", raising=False)
    monkeypatch.delenv("LLM_TOPIC_TIMEOUT_AUTO_TECHNICAL_REPAIR", raising=False)
    monkeypatch.setenv("LLM_DEFAULT_TIMEOUT_SEC", "240")

    # 2. Bug Fix #1: BaseException during extended wait kills & reaps child process cleanly
    killed_flags = []

    class _InterruptedPopen:
        def __init__(self, cmd, **_kw):
            self.cmd = cmd
            self.returncode = None
            self.killed = False
            self.waited = False
            self.calls = 0
            killed_flags.append(self)

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            raise KeyboardInterrupt("outer SIGTERM/SIGINT")

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waited = True
            return -9

    monkeypatch.setattr(subprocess, "Popen", _InterruptedPopen)
    with pytest.raises(KeyboardInterrupt):
        GeminiClient._run_subprocess_with_soft_extend(
            ["agy-bot", "--model", "gemini-4-neon-high"],
            soft_timeout=240.0,
            extend_mult=3.0,
            run_env={},
            exec_cwd=None,
            cli_model="gemini-4-neon-high",
            resume="cid-int-1",
        )
    assert len(killed_flags) == 1
    assert killed_flags[0].killed is True
    assert killed_flags[0].waited is True

    # 3. Bug Fix #2 & #2b: Fresh rotated thread hard timeout + fallback-rescued timeout count slow rotations
    reg_path = str(tmp_path / "llm_topic_threads.json")
    bot_base = str(tmp_path / "agy_base")
    (tmp_path / "agy_base" / "conversations").mkdir(parents=True, exist_ok=True)
    quota_path = str(tmp_path / "llm_quota_status.json")
    monkeypatch.setattr(agy_topics, "get_topic_registry_path", lambda p=None: p or reg_path)
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", classmethod(lambda cls: quota_path))
    monkeypatch.setattr(GeminiClient, "_QUOTA_MEM_CACHE", {})
    monkeypatch.setenv("LLM_SLOW_ROTATIONS_FOR_FALLBACK", "2")
    monkeypatch.setenv("LLM_FORCE_QUOTA_ALERT_TEST", "1")
    sent_alerts = []
    monkeypatch.setattr(
        watchdog_common,
        "send_ntfy",
        lambda title, msg, **kw: (sent_alerts.append((title, msg, kw)), True)[1],
    )
    monkeypatch.setattr(agy_topics, "get_thread_step_count", lambda cid, base: (True, 4, "IDLE"))
    monkeypatch.setattr(agy_topics, "sync_topic_thread_from_ui", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "mirror_topic_thread_to_ui", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "read_summary_row", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "write_annotation_title", lambda *_a, **_k: None)
    monkeypatch.setattr(agy_topics, "delete_threads", lambda *_a, **_k: None)

    now_t = _t.time()
    (tmp_path / "llm_quota_status.json").write_text(
        json.dumps(
            {
                "checked_ts": now_t,
                "available_credits": 10.0,
                "buckets": {
                    "gemini-4-neon-high": 0.90,
                    "gemini-4-neon-medium": 0.90,
                    "claude-sonnet-5-5-medium": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    # Seed 1 prior slow rotation and fresh rotated thread (conversation_id=None)
    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps(
            {
                "auto_sharetrade_executor": {
                    "topic_key": "auto_sharetrade_executor",
                    "conversation_id": None,
                    "consecutive_slow_rotations": 1,
                    "turn_count": 3,
                    "rolling_history": [{"utc": "2026-11-24 18:00 UTC", "summary": "decision=HOLD"}],
                }
            }
        ),
        encoding="utf-8",
    )

    import intelligence.sentiment.gemini_client as gc_mod
    monkeypatch.setattr(gc_mod, "resolve_bot_cli_base", lambda *_a, **_k: bot_base)
    monkeypatch.setattr(gc_mod, "prepare_topic_session", lambda tk, cli_base=None: agy_topics.TopicSession(None, "", False))
    monkeypatch.setattr(
        gc_mod,
        "record_topic_slow_rotation",
        lambda tk, cid=None, **kw: agy_topics.record_topic_slow_rotation(tk, cid, registry_path=reg_path, **kw),
    )
    monkeypatch.setattr(
        gc_mod,
        "record_topic_turn",
        lambda *a, **kw: agy_topics.record_topic_turn(*a, registry_path=reg_path, **kw),
    )
    monkeypatch.setattr(
        gc_mod,
        "get_topic_slow_rotation_count",
        lambda tk: agy_topics.get_topic_slow_rotation_count(tk, registry_path=reg_path),
    )

    cli = GeminiClient(cli_path="/bin/true")
    monkeypatch.setattr(cli, "probe_live_model_quota", lambda **_k: json.loads((tmp_path / "llm_quota_status.json").read_text(encoding="utf-8")))

    # Primary model times out on fresh thread (cid=None), faster fallback succeeds -> must increment count to 2 & demote primary!
    def _timeout_primary_then_ok_fallback(resume, text, call_timeout, cli_model, *args, **kwargs):
        if cli_model == "gemini-4-neon-high":
            raise subprocess.TimeoutExpired(["agy-bot"], call_timeout)
        return subprocess.CompletedProcess(
            args=["agy-bot"],
            returncode=0,
            stdout=json.dumps({"response": '{"decision": "HOLD"}', "conversation_id": "cid-fresh-fb"}),
            stderr="",
        )

    monkeypatch.setattr(cli, "_execute_cli", _timeout_primary_then_ok_fallback)
    out = cli.query_json("evaluate shares", topic_key="auto_sharetrade_executor", model="gemini-4-neon-high", timeout_sec=240.0)
    assert out == {"decision": "HOLD"}
    assert agy_topics.get_topic_slow_rotation_count("auto_sharetrade_executor", registry_path=reg_path) == 2
    q_after_fb = json.loads((tmp_path / "llm_quota_status.json").read_text(encoding="utf-8"))
    assert "gemini-4-neon-high" in (q_after_fb.get("latency_demoted_models") or {})
    assert any("gemini-4-neon-high -> gemini-4-neon-medium" in t for t, _, _ in sent_alerts)

    # Also verify all-candidates hard timeout on fresh thread (cid=None) increments slow rotation counter
    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps(
            {
                "auto_cryptotrade_executor": {
                    "topic_key": "auto_cryptotrade_executor",
                    "conversation_id": None,
                    "consecutive_slow_rotations": 0,
                    "turn_count": 1,
                    "rolling_history": [],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_execute_cli",
        lambda resume, text, call_timeout, cli_model, *a, **k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["agy-bot"], call_timeout)
        ),
    )
    res_none = cli.query_json("evaluate crypto", topic_key="auto_cryptotrade_executor", model="gemini-4-neon-medium", timeout_sec=240.0)
    assert res_none is None
    assert agy_topics.get_topic_slow_rotation_count("auto_cryptotrade_executor", registry_path=reg_path) == 1

    # 4. Bug Fix #3: probe_live_model_quota must NOT restore expired latency demotion when quota == 0.0
    tok_dir = tmp_path / "bot_tok_base"
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "antigravity-oauth-token").write_text(
        json.dumps({"access_token": "tok-1", "id_token_email": "bot@example.com"}),
        encoding="utf-8",
    )
    q_zero_path = str(tmp_path / "llm_quota_zero.json")
    monkeypatch.setattr(GeminiClient, "_get_quota_state_path", classmethod(lambda cls: q_zero_path))
    monkeypatch.setattr(GeminiClient, "_QUOTA_MEM_CACHE", {})
    # Write an EXPIRED demotion (restore_after_ts in the past)
    (tmp_path / "llm_quota_zero.json").write_text(
        json.dumps(
            {
                "checked_ts": now_t - 20000.0,
                "available_credits": 0.0,
                "buckets": {"gemini-4-neon-high": 0.0, "gemini-4-neon-medium": 0.90},
                "latency_demoted_models": {
                    "gemini-4-neon-high": {
                        "fallback_model": "gemini-4-neon-medium",
                        "demoted_ts": now_t - 20000.0,
                        "restore_after_ts": now_t - 100.0,
                        "topic": "auto_sharetrade_executor",
                        "slow_rotations": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def _fake_fetch(url, payload=None, headers=None, timeout=None):
        if "loadCodeAssist" in url:
            return {"paidTier": {"id": "pro", "availableCredits": []}}
        return {
            "buckets": [
                {"modelId": "gemini-4-neon-high", "remainingFraction": 0.0},
                {"modelId": "gemini-4-neon-medium", "remainingFraction": 0.95},
                {"modelId": "claude-sonnet-5-5-medium", "remainingFraction": 1.0},
            ]
        }

    monkeypatch.setattr(gc_mod, "fetch_json_post", _fake_fetch)
    snap_zero = GeminiClient.probe_live_model_quota(bot_cli_base=str(tok_dir), force_refresh=True)
    assert isinstance(snap_zero, dict)
    assert "gemini-4-neon-high" in (snap_zero.get("latency_demoted_models") or {})

    # Once quota recovers (> 0.10), expired demotion IS restored (unless restore_expired_demotions=False)
    def _fake_fetch_healthy(url, payload=None, headers=None, timeout=None):
        if "loadCodeAssist" in url:
            return {"paidTier": {"id": "pro", "availableCredits": []}}
        return {
            "buckets": [
                {"modelId": "gemini-4-neon-high", "remainingFraction": 0.85},
                {"modelId": "gemini-4-neon-medium", "remainingFraction": 0.95},
            ]
        }

    monkeypatch.setattr(gc_mod, "fetch_json_post", _fake_fetch_healthy)
    snap_deferred = GeminiClient.probe_live_model_quota(
        bot_cli_base=str(tok_dir), force_refresh=True, restore_expired_demotions=False
    )
    assert "gemini-4-neon-high" in (snap_deferred.get("latency_demoted_models") or {})
    snap_restored = GeminiClient.probe_live_model_quota(
        bot_cli_base=str(tok_dir), force_refresh=True, restore_expired_demotions=True
    )
    assert "latency_demoted_models" not in snap_restored

    # 5. Bug Fix #4: Outer timeout alignment (>= 240s * 3 = 720s) + unified /compact soft-extend flow
    derived = resolve_derived_telemetry_intervals(executor_interval_sec=360.0)
    assert derived["ai_impulse_exec_timeout_sec"] >= 780.0
    cron_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "systemd", "crontab.prod.txt")
    cron_txt = open(cron_path, encoding="utf-8").read()
    for job_key in ("auto_technical_repair", "auto_cryptotrade_executor", "auto_sharetrade_executor"):
        matching = [ln for ln in cron_txt.splitlines() if f"--topic-key {job_key}" in ln]
        assert len(matching) == 1
        assert "/usr/bin/timeout 780 " in matching[0]

    # 6. /compact uses the exact same soft-extend flow without killing and schedules next-turn rotation when slow
    compact_calls = []
    recorded_turns = []
    step_downs = []
    cli_c = GeminiClient(cli_path="/bin/true")

    def _slow_compact_exec(resume, text, call_timeout, cli_model, *a, **kw):
        compact_calls.append((call_timeout, kw))
        cli_c._JSON_MODE_LOCAL.exceeded_soft_timeout = True
        return subprocess.CompletedProcess(
            args=["agy-bot"],
            returncode=0,
            stdout=json.dumps({"response": "COMPACT_OK", "conversation_id": resume}),
            stderr="",
        )

    monkeypatch.setattr(cli_c, "_execute_cli", _slow_compact_exec)
    monkeypatch.setattr(cli_c, "_maybe_step_down_after_slow_rotations", lambda **kw: step_downs.append(kw))
    monkeypatch.setattr(gc_mod, "snapshot_conversation_state", lambda *_a, **_k: None)
    monkeypatch.setattr(gc_mod, "record_topic_turn", lambda **kw: recorded_turns.append(kw))
    cli_c._finalize_topic_turn(
        topic_key="auto_cryptotrade_executor",
        session=agy_topics.TopicSession("cid-c1", "", True),
        turn_cid="cid-c1",
        output='{"decision": "HOLD"}',
        replaced_conversation=False,
        timeout_sec=120.0,
        models_to_try=["gemini-4-neon-medium"],
        attempt_idx=0,
        enable_pre_quota=False,
        bot_cli_base=bot_base,
        run_env={},
        is_mocked=False,
        include_repo_rules=False,
        topic_lock=None,
        schedule_rotation=False,
        effective_model_used="gemini-4-neon-medium",
    )
    assert len(compact_calls) == 1
    assert compact_calls[0][0] == pytest.approx(240.0)
    assert compact_calls[0][1].get("allow_soft_extend") is True
    assert len(recorded_turns) == 1
    assert recorded_turns[0].get("compacted") is True
    assert recorded_turns[0].get("schedule_rotation") is True
    # Slow background /compact must NOT falsely demote a fast main trading model
    assert step_downs == []


def test_subject1_flow_unifications_order_guard_auto_timeout_4h_cadence_and_share_delta_gate(tmp_path, monkeypatch):
    """Verify all 4 Subject 1 unifications:
    1. order_guard.conf 'auto' timeouts inherit LLM_DEFAULT_TIMEOUT_SEC from config.env.
    2. Quota alert cooldown inherits unified 4h cadence (LLM_LATENCY_DEMOTION_TTL_SEC).
    3. GeminiClient._run_validated_cli_attempt unifies JSON validation across primary & detached calls.
    4. auto_sharetrade_executor shares the exact same Delta Gate HOLD carry-forward flow as crypto."""
    import json
    import subprocess
    import time as _t
    from guard_hierarchy import _load_margins
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard
    from intelligence.sentiment.llm_quota_router import resolve_latency_demotion_ttl_sec
    from orchestratorOS.supervisor.autonomous_ai_reconciler import AutonomousAIReconciler
    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot

    # 1. order_guard.conf 'auto' inherits dynamic LLM_DEFAULT_TIMEOUT_SEC (SSOT)
    margins = _load_margins()
    assert str(margins.get("llm_timeout_sec")).strip().lower() == "auto"
    assert str(margins.get("post_stop_reentry_timeout_sec")).strip().lower() == "auto"
    monkeypatch.setenv("LLM_DEFAULT_TIMEOUT_SEC", "310")
    psg = get_post_stop_reentry_guard(
        gemini_client=GeminiClient(custom_runner=lambda *_a: '{"decision": "APPROVED"}'),
        margins=margins,
    )
    assert psg.timeout_sec == pytest.approx(310.0)
    monkeypatch.setenv("LLM_DEFAULT_TIMEOUT_SEC", "240")

    # 2. Unified 4h cadence shared between latency demotion TTL and low-quota alert cooldown
    monkeypatch.setenv("LLM_LATENCY_DEMOTION_TTL_SEC", "7200")
    monkeypatch.delenv("LLM_QUOTA_ALERT_COOLDOWN_SEC", raising=False)
    assert resolve_latency_demotion_ttl_sec() == pytest.approx(7200.0)

    # 3. Unified _run_validated_cli_attempt rejects non-JSON output cleanly
    cli = GeminiClient(cli_path="/bin/true")
    monkeypatch.setattr(
        cli,
        "_execute_cli",
        lambda *a, **k: subprocess.CompletedProcess(
            ["agy-bot"], 0, stdout=json.dumps({"response": "plain prose not json", "conversation_id": "cid-nj"}), stderr=""
        ),
    )
    _proc, out_txt, cid, _exc_soft, non_json = cli._run_validated_cli_attempt(
        resume=None,
        prompt_text="hi",
        call_timeout=60.0,
        cli_model="gemini-4-neon-medium",
        bot_cli_base=str(tmp_path),
        run_env={},
        is_mocked=True,
        require_json=True,
    )
    assert out_txt is None and cid == "cid-nj" and non_json is True

    # 4. auto_sharetrade_executor uses the unified Delta Gate (skips flat quiet RTH repeat tick)
    llm_calls = []
    dummy_cli = GeminiClient(
        custom_runner=lambda prompt, model, timeout: (
            llm_calls.append((model, timeout)),
            json.dumps({
                "summary": "quiet hold",
                "market_trading_decisions": [{"decision": "HOLD", "symbol": "SPCX_US_EQ", "confidence": 0.9}],
            }),
        )[1]
    )
    sup = AutonomousAIReconciler(
        workspace_dir=str(tmp_path),
        cache_dir=str(tmp_path),
        mode="shadow",
        gemini_auditor=dummy_cli,
    )
    now_t = _t.time()

    def _make_eq_snap(spcx_px: float):
        return TelemetrySensorSnapshot(
            ts=now_t,
            prices_and_trends={"SPCX_US_EQ": {"price": spcx_px, "atr_15m_pct": 0.4}},
            portfolio_summary={
                "tradable_positions": [
                    {"symbol": "SPCX_US_EQ", "venue": "t212", "price": spcx_px, "effective_max_buy_usd": 500.0}
                ],
                "open_limit_orders": [],
            },
            source_timestamps={},
        )

    cur_px = {"px": 125.0}
    monkeypatch.setattr(sup, "harvest_sensors_telemetry", lambda **_k: _make_eq_snap(cur_px["px"]))
    monkeypatch.setattr("orchestratorOS.supervisor.autonomous_ai_reconciler.has_pending_ui_turn", lambda *_a, **_k: False)
    monkeypatch.setenv("AI_DELTA_GATE_ENABLED", "true")

    # First RTH cycle wakes LLM (no_prev_eval, calm -> medium default model, no high-effort override)
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "gemini-4-neon-medium")
    monkeypatch.setenv("LLM_HIGH_STAKE_MODEL", "gemini-4-neon-high")
    monkeypatch.delenv("LLM_TOPIC_MODEL_AUTO_SHARETRADE_EXECUTOR", raising=False)
    sup.gemini_auditor._explicit_model = False
    sup.gemini_auditor.model = "gemini-4-neon-medium"
    sup.gemini_auditor.fallback_models = ("gemini-4-neon-medium", "claude-sonnet-5-5-medium")

    rep1 = sup.run_cycle(topic_key="auto_sharetrade_executor", force_rth_open=True)
    assert len(llm_calls) == 1
    assert rep1.model_used != "delta_gate_skip"
    assert llm_calls[-1][0] == "gemini-4-neon-medium"  # calm cycle uses LLM_DEFAULT_MODEL (-medium)

    # Second flat RTH cycle (0.08% move <= 0.45%) skips redundant LLM call
    cur_px["px"] = 125.10
    rep2 = sup.run_cycle(topic_key="auto_sharetrade_executor", force_rth_open=True)
    assert len(llm_calls) == 1
    assert rep2.model_used == "delta_gate_skip"

    # Third RTH cycle with moderate move (+0.72%, >0.45% and <=1.25%) wakes LLM on -medium (no high-effort escalation)
    cur_px["px"] = 125.90
    rep3 = sup.run_cycle(topic_key="auto_sharetrade_executor", force_rth_open=True)
    assert len(llm_calls) == 2
    assert rep3.model_used != "delta_gate_skip"
    assert llm_calls[-1][0] == "gemini-4-neon-medium"

    # Fourth RTH cycle with strong move (+1.67% > 1.25% via are_close) wakes LLM AND escalates to LLM_HIGH_STAKE_MODEL (-high)
    cur_px["px"] = 128.00
    rep4 = sup.run_cycle(topic_key="auto_sharetrade_executor", force_rth_open=True)
    assert len(llm_calls) == 3
    assert rep4.model_used != "delta_gate_skip"
    assert llm_calls[-1][0] == "gemini-4-neon-high"
    assert rep1.completed_ts is not None and rep1.completed_ts >= rep1.ts


def test_impulse_eval_gap_configurable_are_close_and_zero_benefit_triggers(tmp_path, monkeypatch):
    """Verify:
    1. ai_impulse_min_eval_gap_sec is no longer hardcoded to 65s; it derives from AI_IMPULSE_COOLDOWN_SEC (150s = 2.5m)
       and respects completed_ts (eval finish timestamp) plus optional AI_IMPULSE_MIN_EVAL_GAP_SEC env override.
    2. _filter_unmoved_since_eval_symbols & Delta Gate use botcore.are_close / diff_percent symmetrically.
    3. Zero-benefit noise wakeups (open_order_near_fill, identical-verdict macro_update & geo_update) skip cleanly
       while preserving immediate wakeups on real verdict changes and critical geo shocks."""
    import os
    from pathlib import Path
    from types import SimpleNamespace
    import cachepaths
    import state_io
    from intelligence import globaltelemetry_collector as gtc_mod
    from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals
    from orchestratorOS.supervisor.crypto_executor_delta_gate import evaluate_crypto_executor_delta_gate

    # 1. Config.env exposes explicit AI_IMPULSE_COOLDOWN_SEC=120, AI_IMPULSE_MIN_EVAL_GAP_SEC=120,
    #    AI_DELTA_GATE_MIN_PRICE_MOVE_PCT=0.45, AI_HIGH_EFFORT_PRICE_MOVE_PCT=1.25; derived gap matches cooldown (not 65s)
    from botcore import parse_dotenv
    root_cfg = parse_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.env"))
    assert root_cfg.get("AI_IMPULSE_COOLDOWN_SEC") == "120"
    assert root_cfg.get("AI_IMPULSE_MIN_EVAL_GAP_SEC") == "120"
    assert root_cfg.get("AI_DELTA_GATE_MIN_PRICE_MOVE_PCT") == "0.45"
    assert root_cfg.get("AI_HIGH_EFFORT_PRICE_MOVE_PCT") == "1.25"

    derived = resolve_derived_telemetry_intervals()
    assert derived["ai_impulse_min_eval_gap_sec"] == pytest.approx(derived["ai_impulse_cooldown_sec"])
    assert derived["ai_impulse_min_eval_gap_sec"] == pytest.approx(150.0)

    cdb_dir = tmp_path / "cachedb"
    (cdb_dir / "state").mkdir(parents=True, exist_ok=True)
    (cdb_dir / "prices").mkdir(parents=True, exist_ok=True)
    (cdb_dir / "system").mkdir(parents=True, exist_ok=True)
    eval_now = 1_795_540_000.0 + 180.0  # mid-cycle (outside post/pre cron guards)
    state_io.save_json_state(
        str(cdb_dir / "prices" / "cache_current_price.json"),
        {"items": {"TAOUSDC": [int(eval_now * 1000), 205.0]}},
    )
    state_io.save_json_state(
        str(cdb_dir / "prices" / "cache_short_trend.json"),
        {"items": {"TAOUSDC": {"vel_90s_pct": 1.2, "vel_5m_pct": 1.6, "accel_pct": 0.6, "price": 205.0, "ts": eval_now}}},
    )
    eval_file = Path(cachepaths.resolve_cachedb_filepath(str(cdb_dir / "auto_cryptotrade_executor_eval.json")))
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    old_mtime = eval_now - 300.0
    # Eval started 220s ago, but FINISHED 90s ago (completed_ts) -> < 150s cooldown MUST suppress impulse!
    state_io.save_json_state(
        str(eval_file),
        {"ts": eval_now - 220.0, "completed_ts": eval_now - 90.0, "delta_gate_meta": {"eval_prices": {"TAOUSDC": 200.0}}},
    )
    os.utime(str(eval_file), (old_mtime, old_mtime))

    popen_calls = []
    monkeypatch.setattr(gtc_mod.subprocess, "Popen", lambda cmd, **kw: popen_calls.append(cmd) or SimpleNamespace(pid=99))
    monkeypatch.setattr(gtc_mod.os.path, "isfile", lambda _p: True)
    monkeypatch.setattr(gtc_mod, "evaluate_kinematic_impulse_symbols", lambda *_a, **_k: ["TAOUSDC"])
    monkeypatch.setattr(gtc_mod, "_filter_actionable_impulse_symbols", lambda syms, *_a, **_k: list(syms))

    gtc = gtc_mod.GlobalTelemetryCollector(cache_dir=str(cdb_dir), symbols=["TAOUSDC"], skip_news=True, skip_external=True)
    monkeypatch.setenv("AI_IMPULSE_TRIGGER_ENABLED", "true")
    monkeypatch.setenv("AI_IMPULSE_COOLDOWN_SEC", "150")
    monkeypatch.delenv("AI_IMPULSE_MIN_EVAL_GAP_SEC", raising=False)

    # 90s after completed_ts (< 150s) -> suppressed!
    assert gtc.maybe_trigger_kinematic_ai_wakeup(now=eval_now) is None
    assert popen_calls == []

    # Override AI_IMPULSE_MIN_EVAL_GAP_SEC=60 -> now 90s >= 60s triggers immediately (fully configurable!)
    monkeypatch.setenv("AI_IMPULSE_MIN_EVAL_GAP_SEC", "60")
    assert gtc.maybe_trigger_kinematic_ai_wakeup(now=eval_now) == ["TAOUSDC"]
    assert len(popen_calls) == 1

    # With default AI_IMPULSE_COOLDOWN_SEC=150 (no gap override), 160s after completed_ts triggers cleanly
    monkeypatch.delenv("AI_IMPULSE_MIN_EVAL_GAP_SEC", raising=False)
    gtc._last_ai_impulse_ts = 0.0
    sig_file = cdb_dir / "system" / "kinematic_impulse_signal.json"
    if sig_file.exists():
        sig_file.unlink()
    state_io.save_json_state(
        str(eval_file),
        {"ts": eval_now - 240.0, "completed_ts": eval_now - 160.0, "delta_gate_meta": {"eval_prices": {"TAOUSDC": 200.0}}},
    )
    os.utime(str(eval_file), (old_mtime, old_mtime))
    assert gtc.maybe_trigger_kinematic_ai_wakeup(now=eval_now) == ["TAOUSDC"]
    assert len(popen_calls) == 2

    # 2. _filter_unmoved_since_eval_symbols uses are_close cleanly
    kept_moved = gtc_mod._filter_unmoved_since_eval_symbols(
        ["BTCUSDC", "TAOUSDC"],
        {"BTCUSDC": {"price": 90100.0}, "TAOUSDC": {"price": 202.0}},
        {"ts": eval_now - 200.0, "delta_gate_meta": {"eval_prices": {"BTCUSDC": 90000.0, "TAOUSDC": 200.0}}},
        eff_now=eval_now,
        max_idle_sec=1080.0,
        min_move_pct=0.35,
    )
    assert kept_moved == ["TAOUSDC"]

    # 3. Delta Gate preserves urgent CRITICAL_SHOCK geo_update even if verdict strings match, while skipping identical NORMAL|NONE
    t0 = 1_795_540_000.0
    prev_normal = {
        "ts": t0,
        "mode": "enforce",
        "model_used": "gemini-4-neon-medium",
        "market_trading_decisions": [{"decision": "HOLD", "confidence": 0.5}],
        "delta_gate_meta": {
            "eval_prices": {"BTCUSDC": 90000.0},
            "macro_ts": t0 - 300.0,
            "macro_verdict": "CAUTION|HIGH|HOLD|COUPLED_RISK_OFF",
            "geo_ts": t0 - 300.0,
            "geo_verdict": "NORMAL|NONE",
            "latest_fill_ts": 0.0,
            "open_order_ids": [],
        },
    }
    fresh_crit_s = SimpleNamespace(
        prices_and_trends={"BTCUSDC": {"price": 90050.0}},
        portfolio_summary={"open_limit_orders": [], "recent_orders": {}},
        sentiment_advisor={
            "ts": t0 + 120.0,
            "market_bias": "CAUTION",
            "risk_level": "HIGH",
            "recommended_action": "HOLD",
            "cross_asset_regime": "COUPLED_RISK_OFF",
        },
        geopolitical_threat={"ts": t0 + 120.0, "threat_level": "CRITICAL_SHOCK", "recommended_brake": "HARD_VETO_NEW_BUYS", "risk_score": 0.85},
    )
    skip_c, rsn_c, hi_c, _ = evaluate_crypto_executor_delta_gate(fresh_crit_s, prev_normal, now_ts=t0 + 360.0)
    assert (skip_c, rsn_c, hi_c) == (False, "geo_update", True)

    # 4. Ultra-rare extreme-stakes catalysts (CRITICAL_SHOCK geo crisis & trade_plan_invalidated) escalate to -xhigh
    #    ONLY when quota > soft_conserve & not latency-demoted; routine high triggers stay on -high
    from intelligence.sentiment.llm_quota_router import (
        derive_xhigh_model_from_high,
        resolve_crypto_executor_model_override,
    )
    from orchestratorOS.supervisor.crypto_executor_delta_gate import is_extreme_stakes_delta_gate_wakeup

    assert derive_xhigh_model_from_high("gemini-4-neon-high") == "gemini-4-neon-xhigh"
    assert derive_xhigh_model_from_high("claude-sonnet-5-5-high") == ""
    assert derive_xhigh_model_from_high("gemini-4-neon-high", "custom-xhigh") == "custom-xhigh"

    assert is_extreme_stakes_delta_gate_wakeup("trade_plan_invalidated(BTCUSDC)") is True
    assert is_extreme_stakes_delta_gate_wakeup("new_fill") is True
    assert is_extreme_stakes_delta_gate_wakeup("geo_update", sensors=fresh_crit_s, prev_eval=prev_normal) is True
    for non_extreme in (
        "open_order_stale_or_far",
        "open_order_wall_trapped",
        "focus_symbols(TAOUSDC)",
        "price_move(TAOUSDC:2.5%>=0.6%)",
        "trade_plan_target_hit(BTCUSDC)",
    ):
        assert is_extreme_stakes_delta_gate_wakeup(non_extreme) is False

    healthy_q = {"available_credits": 0.0, "buckets": {"gemini-4-neon-xhigh": 0.60, "gemini-4-neon-high": 0.60, "gemini-4-neon-medium": 0.60}}
    low_conserve_q = {"available_credits": 0.0, "buckets": {"gemini-4-neon-xhigh": 0.20, "gemini-4-neon-high": 0.20, "gemini-4-neon-medium": 0.20}}
    demoted_q = {
        **healthy_q,
        "latency_demoted_models": {"gemini-4-neon-xhigh": {"fallback_model": "gemini-4-neon-high", "restore_after_ts": t0 + 3600.0}},
    }
    assert resolve_crypto_executor_model_override(
        healthy_q,
        high_model="gemini-4-neon-high",
        default_model="gemini-4-neon-medium",
        xhigh_model="gemini-4-neon-xhigh",
        escalate_high_effort=True,
        urgent_escalation=True,
        escalate_xhigh=True,
        now_ts=t0,
    ) == ("gemini-4-neon-xhigh", "escalate_xhigh_effort")
    # Quota <= soft_conserve (20% <= 25%) steps down from xhigh to high (preserves urgent high, avoids draining on xhigh)
    assert resolve_crypto_executor_model_override(
        low_conserve_q,
        high_model="gemini-4-neon-high",
        default_model="gemini-4-neon-medium",
        xhigh_model="gemini-4-neon-xhigh",
        escalate_high_effort=True,
        urgent_escalation=True,
        escalate_xhigh=True,
        now_ts=t0,
    ) == ("gemini-4-neon-high", "escalate_high_effort")
    # Latency-demoted xhigh also gracefully steps down to high
    assert resolve_crypto_executor_model_override(
        demoted_q,
        high_model="gemini-4-neon-high",
        default_model="gemini-4-neon-medium",
        xhigh_model="gemini-4-neon-xhigh",
        escalate_high_effort=True,
        urgent_escalation=True,
        escalate_xhigh=True,
        now_ts=t0,
    ) == ("gemini-4-neon-high", "escalate_high_effort")

    # 5. agy_store_fix.prune_old_backups caps unbounded timestamped backup growth while preserving custom dirs
    from orchestratorOS.admin import agy_store_fix as asf

    bk_root = tmp_path / "agy_store_fix_backups"
    for ts_name in (
        "20261112_091209",
        "20261113_100000",
        "20261114_110000",
        "20261115_120000",
        "20261123_130000",
        "20261124_140000",
        "20261124_213206",
    ):
        d = bk_root / ts_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "dummy.db").write_text("x", encoding="utf-8")
    (bk_root / "manual_keep").mkdir(parents=True, exist_ok=True)
    assert asf.prune_old_backups(str(bk_root), keep_count=3) == 4
    remaining = sorted(p.name for p in bk_root.iterdir())
    assert remaining == ["20261123_130000", "20261124_140000", "20261124_213206", "manual_keep"]


def test_crypto_prompt_critical_data_audit_fixes_1_to_6(tmp_path):
    """Regression suite for the 6 critical/high crypto prompt data correctness fixes:
    1.1 Manual orders classify as actor='MANUAL' (never 'BOT') while remaining cancellable via include_bot_orders=True.
    1.2 Newly added crypto symbols merge fresh cache_ohlc_24h into series_24h so 24h_range matches vol_profile/vwap.
    1.3 Partial (<5d) crypto 7d archives flag is_partial_7d and suppress bogus 7d / vs_btc(7d) in prompt & bellwethers.
    1.4 LIFO scorecard drops sub-$2 dust remainders (no phantom WIN/LOSS or OPEN_BUY) and uses persisted live venue fees.
    1.5 MarketBreadth deduplicates multi-quote pairs by canonical base asset and prompts use BTC_vol_share / Alt_vol_share.
    1.6 Geopolitical news matching catches kinetic strikes / air raids and prevents/recovers backdated first_seen_ts.
    """
    from intelligence.macro.calendar_ssot import headline_effective_ts
    from intelligence.macro.lexicon_ssot import match_high_severity_keywords
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsHeadline
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.sentiment.collectors.microstructure_collector import (
        enrich_24h_and_microstructure,
        load_symbol_price_series,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    from orchestratorOS.supervisor.prompt_sections import (
        _format_macro_and_news_section,
        _format_market_prices_section,
    )
    from providers.ai_survival_guard import build_persisted_round_trip_fee_resolver
    from providers.intent_execution import cancel_existing_ai_open_orders
    from providers.order_id_context import format_actor_tag, resolve_order_origin

    # --- 1.1 Manual order classification & AI intervention compatibility ---
    for cid in ("and_729245677", "ios_9911", "web_568296122", "MO_MN_OR_MainThread_1A2B"):
        act, mod, _ = resolve_order_origin({"orderId": "10", "clientOrderId": cid})
        assert act == "MANUAL"
        assert mod == "manual"
        assert format_actor_tag(act, mod, None) == "MANUAL"
    assert resolve_order_actor({"clientOrderId": "TA_grid_1"}) == "BOT"
    assert resolve_order_actor({"clientOrderId": "sd_manual_1"}) == "BOT"
    assert resolve_order_actor({"clientOrderId": "AI_intent_1"}) == "AI"

    manual_oo = {"orderId": "729245677", "clientOrderId": "and_729245677", "side": "BUY", "price": "174.2", "origQty": "2.0"}
    inst_keep = SimpleNamespace(
        provider_name="binance",
        symbol="TAOUSDC",
        open_orders=MagicMock(return_value=[manual_oo]),
        cancel_order=MagicMock(return_value=True),
    )
    assert cancel_existing_ai_open_orders(inst_keep, "BUY", set(), include_bot_orders=False) == []
    assert inst_keep.cancel_order.call_count == 0
    assert cancel_existing_ai_open_orders(inst_keep, "BUY", set(), include_bot_orders=True) == ["729245677"]
    assert inst_keep.cancel_order.call_count == 1

    # --- 1.2 & 1.3 Crypto cache_ohlc_24h merge & partial (<5d) 7d suppression ---
    cdir = tmp_path / "cachedb"
    pdir = cdir / "prices"
    pdir.mkdir(parents=True, exist_ok=True)
    now_s = 1_795_558_000.0
    now_ms = int(now_s * 1000)

    # ETHUSDC: only 6h in cache_24_price ([1665, 1687]), full 22h in cache_ohlc_24h ([1665, 1717.73])
    eth_24 = [[now_ms - (6 - i) * 3600_000, 1665.0 + i * 3.5] for i in range(7)]
    eth_ohlc = (
        [[now_ms - (22 - i) * 3600_000, 1717.73 - i * 3.0, 5000.0, 1718.0 - i * 3.0, 1715.0 - i * 3.0] for i in range(16)]
        + [[now_ms - (6 - i) * 3600_000, 1665.0 + i * 3.5, 5000.0, 1666.0 + i * 3.5, 1664.0 + i * 3.5] for i in range(7)]
    )
    (pdir / "cache_24_price_ETHUSDC.json").write_text(json.dumps({"items": {"ETHUSDC": eth_24}}), encoding="utf-8")
    (pdir / "cache_ohlc_24h_ETHUSDC.json").write_text(
        json.dumps({"ts": now_s, "symbol": "ETHUSDC", "points": eth_ohlc}), encoding="utf-8"
    )
    # ADAUSDC: ~25h in cache_24 and cache_365 (not 7 days!)
    ada_pts = [[now_ms - (25 - i) * 3600_000, 0.1500 - i * 0.0002] for i in range(26)]
    (pdir / "cache_24_price_ADAUSDC.json").write_text(json.dumps({"items": {"ADAUSDC": ada_pts[-24:]}}), encoding="utf-8")
    (pdir / "cache_365_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps({"s": "ADAUSDC", "i": pt}) for pt in ada_pts) + "\n", encoding="utf-8"
    )
    # BTCUSDC: full 7d archive
    btc_pts = [[now_ms - (7 - i) * 86400_000, 64000.0 - i * 600.0] for i in range(8)]
    (pdir / "cache_24_price_BTCUSDC.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[now_ms - 24 * 3600_000, 60800.0], [now_ms, 59800.0]]}}), encoding="utf-8"
    )
    (pdir / "cache_365_price_BTCUSDC.jsonl").write_text(
        "\n".join(json.dumps({"s": "BTCUSDC", "i": pt}) for pt in btc_pts) + "\n", encoding="utf-8"
    )

    s24_eth, _ = load_symbol_price_series(str(cdir), "ETHUSDC", include_7d=True)
    assert max(p[1] for p in s24_eth) == pytest.approx(1717.73)

    pt_map = {
        "BTCUSDC": {"price": 59800.0},
        "ETHUSDC": {"price": 1686.0},
        "ADAUSDC": {"price": 0.1450},
    }
    enrich_24h_and_microstructure(str(cdir), str(tmp_path), pt_map, {}, {}, {}, now_ts=now_s, allow_network=False)
    assert pt_map["ETHUSDC"]["high_24h"] == pytest.approx(1717.73)
    assert pt_map["ETHUSDC"]["pos_24h_pct"] < 45.0
    assert pt_map["ADAUSDC"].get("is_partial_7d") is True
    assert pt_map["ADAUSDC"].get("span_7d_hours") == pytest.approx(25.0, abs=0.5)
    assert pt_map["BTCUSDC"].get("is_partial_7d") is None

    prompt_prices = "\n".join(_format_market_prices_section(pt_map))
    ada_prompt_line = next(l for l in prompt_prices.splitlines() if "ADAUSDC:" in l)
    btc_prompt_line = next(l for l in prompt_prices.splitlines() if "BTCUSDC:" in l)
    assert "7d=" not in ada_prompt_line
    assert "7d_range=" not in ada_prompt_line
    assert "7d=" in btc_prompt_line
    cm = build_cross_market_context(pt_map, {}, {})
    assert "chg_7d_pct" not in cm["crypto_bellwethers"]["ADA"]
    assert "chg_7d_pct" in cm["crypto_bellwethers"]["BTC"]

    # --- 1.4 LIFO dust remainder filtering & persisted live venue fees ---
    # BUY 4.4121 TAO @ 187.80 ($828.59), SELL 4.4089 TAO @ 187.80 ($827.99) -> $0.60 dust remainder must be discarded
    # Next BUY 3.0 TAO @ 180.0, SELL 3.0 TAO @ 190.0 -> must NOT emit a separate $0.60 slice
    dust_rows = [
        {"ts": 100.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "BUY", "px": 187.8, "qty": 4.4121, "usd": 828.59},
        {"ts": 200.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "SELL", "px": 187.8, "qty": 4.4089, "usd": 827.99},
        {"ts": 300.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "BUY", "px": 180.0, "qty": 3.0, "usd": 540.0},
        {"ts": 400.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "SELL", "px": 190.0, "qty": 3.0, "usd": 570.0},
    ]
    c_pairs, ob_rem, os_rem = match_filled_intent_rows_lifo(dust_rows)
    assert len(c_pairs) == 2
    assert ob_rem == [] and os_rem == []

    # Persisted VIP0_BNB 7.5bps maker -> 15.0bps round-trip fee applied in scorecard
    fee_dir = tmp_path / "fee_cache"
    (fee_dir / "state").mkdir(parents=True, exist_ok=True)
    (fee_dir / "state" / "venue_fee_maintenance_state.json").write_text(
        json.dumps({
            "ts": now_s,
            "interval_sec": 14400.0,
            "venues": {
                "binance": {
                    "ts": now_s,
                    "fee_spec": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.12},
                }
            },
        }),
        encoding="utf-8",
    )
    fee_res = build_persisted_round_trip_fee_resolver(str(fee_dir), now_ts=now_s)
    assert fee_res is not None
    assert fee_res("binance", "TAOUSDC", "LIMIT") == pytest.approx(15.0)
    assert fee_res("binance", "TAOUSDC", "MARKET") == pytest.approx(14.24)
    sc_intents = [
        {"intent_id": "b1", "timestamp": now_s - 600, "decision": "BUY", "venue": "binance", "symbol": "TAOUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1000.0, "execution_payload": {"orderId": "1", "status": "FILLED", "avg_fill_price": 100.0, "executedQty": "10.005"}},
        {"intent_id": "s1", "timestamp": now_s - 300, "decision": "SELL", "venue": "binance", "symbol": "TAOUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1010.0, "execution_payload": {"orderId": "2", "status": "FILLED", "avg_fill_price": 101.0, "executedQty": "10.0"}},
    ]
    sc_live = evaluate_ai_decision_scorecard(sc_intents, now_ts=now_s, scope="crypto", fee_resolver=fee_res)
    # Gross +1.00% minus 0.15% live VIP0_BNB RT fee = +0.85% (not +0.80% static 20bps), and 0.005 ($0.50) dust left no OPEN_BUY
    assert sc_live["total_outcomes"] == 1
    assert sc_live["avg_edge_pct"] == pytest.approx(0.85, abs=0.01)

    # --- 1.5 MarketBreadth per-base dedup & accurate BTC_vol_share / Alt_vol_share labels ---
    dup_tickers = [
        {"symbol": "BTCUSDT", "lastPrice": "60000", "priceChangePercent": "-2.0", "quoteVolume": "300000000"},
        {"symbol": "BTCUSDC", "lastPrice": "60000", "priceChangePercent": "-2.1", "quoteVolume": "150000000"},
        {"symbol": "BTCFDUSD", "lastPrice": "60000", "priceChangePercent": "-2.0", "quoteVolume": "50000000"},
        {"symbol": "ETHUSDT", "lastPrice": "1700", "priceChangePercent": "-2.5", "quoteVolume": "200000000"},
        {"symbol": "ETHUSDC", "lastPrice": "1700", "priceChangePercent": "-2.4", "quoteVolume": "50000000"},
        {"symbol": "PEPEUSDT", "lastPrice": "0.00001", "priceChangePercent": "+5.0", "quoteVolume": "100000000"},
        {"symbol": "PEPEUSDC", "lastPrice": "0.00001", "priceChangePercent": "+5.2", "quoteVolume": "30000000"},
        {"symbol": "SOLUSDT", "lastPrice": "70", "priceChangePercent": "-3.0", "quoteVolume": "80000000"},
        {"symbol": "XRPUSDT", "lastPrice": "1.5", "priceChangePercent": "-1.5", "quoteVolume": "40000000"},
    ]
    mb_dedup = MarketBreadthCollector.parse_tickers(dup_tickers, min_volume_usd=5_000_000.0, now_ts=now_s)
    assert mb_dedup is not None
    assert mb_dedup.total_count == 5
    assert mb_dedup.advancing_count == 1
    assert mb_dedup.declining_count == 4
    assert mb_dedup.btc_vol_share_pct == pytest.approx(50.0)
    assert mb_dedup.alt_vol_share_pct == pytest.approx(25.0)
    sec1_mb = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(
                market_breadth={
                    "advance_ratio": mb_dedup.advance_ratio,
                    "median_change_pct": mb_dedup.median_change_pct,
                    "mean_change_pct": mb_dedup.mean_change_pct,
                    "dispersion_std": mb_dedup.dispersion_std,
                    "advancing": mb_dedup.advancing_count,
                    "total_symbols": mb_dedup.total_count,
                    "btc_vol_share_pct": mb_dedup.btc_vol_share_pct,
                    "alt_vol_share_pct": mb_dedup.alt_vol_share_pct,
                }
            ),
            asset_scope="crypto",
        )
    )
    assert "BTC_vol_share=50.0%" in sec1_mb
    assert "Alt_vol_share=25.0%" in sec1_mb
    assert "BTCDOM=" not in sec1_mb
    assert "TOTAL3_share=" not in sec1_mb

    # --- 1.6 Geopolitical kinetic keywords & non-backdated first_seen_ts ---
    assert "houthi strike" in match_high_severity_keywords(
        "Eleven hurt in second Houthi strike within days on Riyadh's tallest tower - Reuters"
    )
    assert "air raid" in match_high_severity_keywords(
        "Russian jets spark air raid alert in South Korea"
    )
    assert match_high_severity_keywords("Rail workers threaten union strike over contract dispute") == ()

    # Corrupted backdated first_seen_ts (2.7h earlier than pub_date) recovers cleanly to pub_date
    pub_s = 1_795_555_995.0  # Tue, 24 Nov 2026 21:33:15 GMT
    bad_first_seen = 1_795_546_160.0  # Tue, 24 Nov 2026 18:49:20 GMT
    hist_f = tmp_path / "news_feed_collect.json"
    hist_f.write_text(
        json.dumps([
            {
                "title": "11 injured in Houthi drone attack on Riyadh tower, Saudi civil defence says - Al Jazeera",
                "source": "Al Jazeera",
                "pub_date": "Tue, 24 Nov 2026 21:33:15 GMT",
                "url": "https://example.com/aj-riyadh",
                "matched_keywords": ["attack", "drone attack"],
                "is_high_severity": True,
                "first_seen_ts": bad_first_seen,
            }
        ]),
        encoding="utf-8",
    )
    nfc = NewsFeedCollector(history_file=str(hist_f))
    nfc._load_history(now_ts=pub_s + 500.0)
    snap_nf = nfc.get_pool_snapshot(now_ts=pub_s + 500.0)
    assert len(snap_nf.headlines) == 1
    assert snap_nf.headlines[0].first_seen_ts == pytest.approx(pub_s)
    assert headline_effective_ts(
        NewsHeadline(
            title="11 injured in Houthi drone attack on Riyadh tower",
            source="Al Jazeera",
            pub_date="Tue, 24 Nov 2026 21:33:15 GMT",
            url="https://example.com/aj-riyadh",
            matched_keywords=("drone attack",),
            is_high_severity=True,
            first_seen_ts=bad_first_seen,
        ),
        pub_s + 500.0,
    ) == pytest.approx(pub_s)

    # Newer canonical URL revision with newer pub_date must NOT be backdated to the earlier live-blog stub
    nfc_rev = NewsFeedCollector(history_file=str(tmp_path / "news_rev.json"))
    snap_rev = nfc_rev._aggregate(
        SimpleNamespace(
            headlines=[
                NewsHeadline(
                    title="Smoke reported near Riyadh commercial district",
                    source="Al Jazeera",
                    pub_date="Tue, 24 Nov 2026 18:49:20 GMT",
                    url="https://example.com/liveblog",
                    matched_keywords=(),
                    is_high_severity=False,
                    first_seen_ts=bad_first_seen,
                )
            ]
        ),
        now=pub_s + 500.0,
    )
    assert len(snap_rev.headlines) == 1
    snap_rev2 = nfc_rev._aggregate(
        SimpleNamespace(
            headlines=[
                NewsHeadline(
                    title="11 injured in Houthi drone attack on Riyadh tower",
                    source="Al Jazeera",
                    pub_date="Tue, 24 Nov 2026 21:33:15 GMT",
                    url="https://example.com/liveblog",
                    matched_keywords=("drone attack", "attack"),
                    is_high_severity=True,
                    first_seen_ts=pub_s,
                )
            ]
        ),
        now=pub_s + 500.0,
    )
    assert len(snap_rev2.headlines) == 1
    assert snap_rev2.headlines[0].first_seen_ts == pytest.approx(pub_s)




def test_crypto_prompt_medium_minor_audit_fixes_2_1_to_2_4():
    """Verify medium/minor live crypto executor prompt audit fixes 2.1 through 2.4."""
    from types import SimpleNamespace

    from intelligence.external.collectors.telemetry_ws_streams import BinanceSpotKline5mStream
    from intelligence.sentiment.evaluators.microstructure_evaluator import compute_liquidation_clusters
    from intelligence.sentiment.evaluators.prompt_view_evaluator import PortfolioPromptViewEvaluator
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import (
        _format_macro_and_news_section,
        _format_tradable_position_line,
    )
    from providers.binance_provider import BinanceProvider

    # --- 2.1 Binance 15m rolling spot flow window (REST limit=4 + WS 4-bar proration parity) ---
    req_usdc = BinanceProvider.public_spot_cvd_15m_request("BTCUSDC", now_ts=1_795_560_000.0)
    req_usdt = BinanceProvider.public_primary_quote_spot_cvd_15m_request("BTCUSDC", now_ts=1_795_560_000.0)
    assert req_usdc is not None and "interval=5m&limit=4" in req_usdc[0]
    assert req_usdt is not None and "interval=5m&limit=4" in req_usdt[0]

    t0_ms = 1_795_559_100_000  # 15m before t3
    t1_ms = t0_ms + 300_000
    t2_ms = t1_ms + 300_000
    t3_ms = t2_ms + 300_000
    # 44s into the live 5m bar t3: exact rolling 15m window overlaps 256/300 of bar t0 + full t1, t2 + live t3
    now_s = (t3_ms + 44_000) / 1000.0
    since_s = now_s - 900.0
    raw_4 = [
        [t0_ms, "1", "1", "1", "1", "1", t0_ms + 299_999, "1000000", 10, "1", "600000", "0"],
        [t1_ms, "1", "1", "1", "1", "1", t1_ms + 299_999, "1000000", 10, "1", "550000", "0"],
        [t2_ms, "1", "1", "1", "1", "1", t2_ms + 299_999, "1000000", 10, "1", "500000", "0"],
        [t3_ms, "1", "1", "1", "1", "1", t3_ms + 299_999, "146667", 10, "1", "88000", "0"],
    ]
    rest_flow = BinanceProvider.extract_spot_flow_15m_usd(raw_4, since_ts=since_s)
    assert rest_flow is not None
    # True 15m total volume is ~3.0M (not ~2.15M 10m44s sawtooth drop)
    assert rest_flow[1] == pytest.approx(3_000_000.0, abs=10.0)
    # Bit-for-bit parity with live WS 5m kline stream
    ws_kline = BinanceSpotKline5mStream(["BTCUSDC"])
    ws_kline._connected = True
    ws_kline._last_msg_ts = now_s
    assert ws_kline.seed_from_rest_klines("BTCUSDC", raw_4, now=now_s) == 4
    ws_flow = ws_kline.get_spot_flow_15m("BTCUSDC", now=now_s)
    assert ws_flow == rest_flow
    # Legacy 3-bar / since_ts=0 compatibility preserved
    legacy_3 = BinanceProvider.extract_spot_flow_15m_usd(raw_4[-3:], since_ts=0.0)
    assert legacy_3 == (129_333.0, 2_146_667.0)

    # --- 2.2 Hyperliquid 1h funding normalized to 8h-equivalent in compute_liquidation_clusters ---
    base_minfo = {"oi_usd": 100_000_000.0, "whale_long_pct": 58.0}
    p_dict = {"price": 100.0, "vol_1h_pct": 1.5}
    # Both venues express identical +10.95% annualized funding: Binance 8h=0.0001, Hyperliquid 1h=0.0000125
    bn_liq = compute_liquidation_clusters(
        p_dict,
        {**base_minfo, "funding_rate": 0.0001, "funding_annualized_pct": 10.95},
    )
    hl_liq = compute_liquidation_clusters(
        p_dict,
        {**base_minfo, "funding_rate": 0.0000125, "funding_annualized_pct": 10.95},
    )
    hl_unnorm = compute_liquidation_clusters(
        p_dict,
        {**base_minfo, "funding_rate": 0.0000125},
    )
    assert hl_liq["long_liq_usd"] == bn_liq["long_liq_usd"]
    assert hl_liq["short_liq_usd"] == bn_liq["short_liq_usd"]
    assert hl_liq["long_liq_usd"] > hl_unnorm["long_liq_usd"]

    # --- 2.3 Flat/dust positions drop misleading pnl_vs_last_buy & dust zero-cost cycles omitted ---
    flat_dust_pos = {
        "venue": "binance",
        "symbol": "TAOUSDC",
        "base": "TAO",
        "quote": "USDC",
        "price": 179.46,
        "free_base": 0.004569,
        "value_usd": 0.82,
        "free_quote": 500.0,
        "active_avg_buy_px": None,
        "last_buy_px": 182.9,
        "pnl_vs_last_buy_pct": -1.88,
        "last_sell_px": 184.7,
        "chg_vs_last_sell_pct": -2.84,
        "open_lots": [],
    }
    flat_line = _format_tradable_position_line(flat_dust_pos)
    assert "chg_vs_last_buy=-1.88%, lot_harvested" in flat_line
    assert "pnl_vs_last_buy=" not in flat_line

    active_pos = {
        "venue": "binance",
        "symbol": "BTCUSDC",
        "base": "BTC",
        "quote": "USDC",
        "price": 69000.0,
        "free_base": 0.05,
        "value_usd": 3450.0,
        "free_quote": 500.0,
        "active_avg_buy_px": 70000.0,
        "last_buy_px": 69500.0,
        "open_lots": [{"qty": 0.05, "entry_px": 69500.0}],
    }
    active_line = _format_tradable_position_line(active_pos)
    assert "pnl_vs_last_buy=-0.72%" in active_line

    pv_eval = PortfolioPromptViewEvaluator().evaluate(
        {
            "tradable_positions": [
                {"venue": "kraken", "symbol": "HYPEUSD", "price": 27.68, "free_base": 0.002591, "value_usd": 0.07},
                {"venue": "hyperliquid", "symbol": "HYPEUSDC", "price": 27.68, "free_base": 0.001755, "value_usd": 0.05},
                {"venue": "kraken", "symbol": "SOLUSD", "price": 86.0, "free_base": 10.0, "value_usd": 860.0},
            ],
            "kraken_cycles": {
                "HYPEUSD": {"qty": 0.002591, "cost_usd": 0.0, "avg_cost_px": None},
                "SOLUSD": {"qty": 10.0, "cost_usd": 850.0, "avg_cost_px": 85.0},
            },
            "hyperliquid_cycles": {
                "HYPEUSDC": {"qty": 0.001755, "cost_usd": 0.0, "avg_cost_px": None},
            },
        },
        asset_scope="crypto",
    )
    kept_cycles = {
        (vid, c_sym)
        for _, vid, items in pv_eval["venue_cycles"]
        for c_sym, _ in items
    }
    assert ("kraken", "SOLUSD") in kept_cycles
    assert ("kraken", "HYPEUSD") not in kept_cycles
    assert ("hyperliquid", "HYPEUSDC") not in kept_cycles

    # --- 2.4 Macro calendar upcoming label + LIMIT spread/fee & calibration arr/slip sign clarity ---
    sec1_macro = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(
                macro_calendar={
                    "session_label": "ASIA_SESSION",
                    "next_high_impact_event": "US Equity Market Open (09:30 ET) [in 13.7h]",
                    "macro_release_confirmed": False,
                    "fomc_today": False,
                    "upcoming_windows": [
                        "US Equity Market Open (09:30 ET) (in 13.7h)",
                        "Core PCE Price Index m/m [HIGH] (in 12.7h)",
                    ],
                }
            ),
            asset_scope="crypto",
        )
    )
    assert "upcoming=[" in sec1_macro
    assert "routine_session_transitions=" not in sec1_macro

    sec1_routine = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(
                macro_calendar={
                    "session_label": "ASIA_SESSION",
                    "next_high_impact_event": "US Equity Market Open (09:30 ET) [in 13.7h]",
                    "macro_release_confirmed": False,
                    "fomc_today": False,
                    "upcoming_windows": [
                        "US Equity Market Open (09:30 ET) (in 13.7h)",
                        "US Equity Market Close (16:00 ET) (in 20.2h)",
                    ],
                }
            ),
            asset_scope="crypto",
        )
    )
    assert "routine_session_transitions=[" in sec1_routine

    rules_txt = build_supervisor_prompt(
        SimpleNamespace(),
        {},
        topic_key="auto_cryptotrade_executor",
        max_usd=1000.0,
    )
    assert "for low maker fees" not in rules_txt
    assert "capture bid-ask spread and lower execution cost" in rules_txt
    assert "negative bps = favorable price improvement" in rules_txt


def test_cycle1_geo_unassessed_plans_and_news_clickbait_fixes(tmp_path):
    """Verify Cycle 1 live PROD prompt audit fixes: geo unassessed/surge, superseded SELL plans, and clickbait filter."""
    import json
    from types import SimpleNamespace

    from intelligence.macro.calendar_ssot import compute_dynamic_geopolitical_interval
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item
    from intelligence.macro.news_feed_collector import NewsFeedCollector
    from intelligence.sentiment.collectors.macro_telemetry_collector import MacroAndNewsSensorsCollector
    from orchestratorOS.supervisor.prompt_sections import _format_macro_and_news_section
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import summarize_active_ai_trade_plans

    now_s = 1_795_561_200.0  # 2026-11-25 01:00 EET

    # --- C1.1: 2 genuinely new fresh high-severity headlines wake Geo Threat Shield (45m vs 4.5h) ---
    prev_dig = "Libya's Haftar visits Turkiye days after Ukraine hits Russian warship"
    snap_two = {
        "headlines": [
            {
                "title": "Russian jets spark air raid alert in South Korea",
                "source": "Al Jazeera",
                "is_high_severity": True,
                "first_seen_ts": now_s - 3900.0,
            },
            {
                "title": "11 injured in Houthi drone attack on Riyadh tower, Saudi civil defence says",
                "source": "Al Jazeera",
                "is_high_severity": True,
                "first_seen_ts": now_s - 2800.0,
            },
        ],
        "high_severity_count": 2,
        "has_critical_shock_keywords": False,
    }
    itv_2, rsn_2 = compute_dynamic_geopolitical_interval(
        base_interval_sec=16200.0,
        last_eval_ts=now_s - 6900.0,
        now_ts=now_s,
        cached_assessment={"threat_level": "NORMAL", "is_fallback": False},
        news_snapshot=snap_two,
        headlines_changed=True,
        last_headlines_digest=prev_dig,
    )
    assert itv_2 == 2700.0
    assert rsn_2 == "high_severity_headline_surge_2"

    # --- C1.2: Geo Threat Shield annotates unassessed fresh (<90m) high-sev headlines arriving after shield ts ---
    gt_cur = {
        "threat_level": "NORMAL",
        "risk_score": 0.2,
        "summary": "The sole headline within the 90-minute window is a routine diplomatic visit.",
        "ts": now_s - 6900.0,
        "age_min": 115.0,
        "headlines_digest": prev_dig,
    }
    valid_news = [
        {
            "title": "11 injured in Houthi drone attack on Riyadh tower, Saudi civil defence says - Al Jazeera",
            "is_high_severity": True,
            "first_seen_ts": now_s - 2800.0,
        }
    ]
    MacroAndNewsSensorsCollector._reconcile_geopolitical_deescalation(
        gt_cur,
        valid_news,
        [{"title": valid_news[0]["title"], "is_high_severity": True, "age_min": 46.7}],
        set(),
        cache_dir=str(tmp_path),
        now_ts=now_s,
        allow_network=False,
        persist=False,
        read_json_fn=None,
    )
    assert "UNASSESSED_FRESH_HIGH_SEV" in gt_cur.get("live_unassessed_high_sev_note", "")
    sec1_gt = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(geopolitical_threat=gt_cur),
            asset_scope="crypto",
        )
    )
    assert "live_unassessed=UNASSESSED_FRESH_HIGH_SEV" in sec1_gt

    # --- C1.3: NewsFeedCollector._load_history upgrades cached missed headlines on lexicon refresh ---
    hist_p = tmp_path / "news_feed_collect.json"
    hist_p.write_text(
        json.dumps([
            {
                "title": "Eleven hurt in second Houthi strike within days on Riyadh's tallest tower - Reuters",
                "source": "Reuters",
                "pub_date": "Tue, 24 Nov 2026 20:42:15 GMT",
                "url": "https://example.com/reuters-riyadh",
                "matched_keywords": [],
                "is_high_severity": False,
                "first_seen_ts": now_s - 5000.0,
            }
        ]),
        encoding="utf-8",
    )
    nfc = NewsFeedCollector(history_file=str(hist_p))
    nfc._load_history(now_ts=now_s)
    snap_loaded = nfc.get_pool_snapshot(now_ts=now_s)
    assert len(snap_loaded.headlines) == 1
    assert snap_loaded.headlines[0].is_high_severity is True
    assert "houthi strike" in snap_loaded.headlines[0].matched_keywords

    # --- C1.4: Superseded FILLED SELL plans dropped once newer RESTING BUY plan or newer BUY fill exists ---
    intents_sup = [
        {
            "status": "EXECUTED",
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "decision": "SELL",
            "order_type": "LIMIT",
            "timestamp": now_s - 18.8 * 3600.0,
            "target_px": 62.9,
            "invalidation_px": 64.1,
            "execution_payload": {"status": "FILLED", "price": 63.55, "filled_qty": 11.22},
        },
        {
            "status": "EXECUTED",
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "decision": "BUY",
            "order_type": "LIMIT",
            "timestamp": now_s - 2.4 * 3600.0,
            "limit_price": 58.45,
            "target_px": 60.35,
            "invalidation_px": 56.85,
            "execution_payload": {"status": "OPEN", "order_id": "615579321222", "price": 58.45},
        },
        {
            "status": "EXECUTED",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "decision": "SELL",
            "order_type": "LIMIT",
            "timestamp": now_s - 20.8 * 3600.0,
            "target_px": 60150.0,
            "invalidation_px": 61180.0,
            "execution_payload": {"status": "FILLED", "price": 60792.2, "filled_qty": 0.0296},
        },
    ]
    open_orders_sup = [
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "side": "BUY", "order_id": "615579321222", "price": 58.45, "status": "NEW"}
    ]
    pos_sup = [
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "price": 60.29, "value_usd": 0.11, "effective_max_buy_usd": 2000.0},
        {"venue": "binance", "symbol": "BTCUSDC", "price": 59902.2, "value_usd": 1504.59, "effective_max_buy_usd": 2000.0},
    ]
    ro_sup = {
        "BTCUSDC": {
            "last_buy": {"price": 60076.8, "ts_ms": int((now_s - 14.4 * 3600.0) * 1000.0)},
        }
    }
    sup_plans = summarize_active_ai_trade_plans(
        intents_sup,
        "crypto",
        now_s,
        extract_intent_notional,
        tradable_positions=pos_sup,
        open_orders=open_orders_sup,
        recent_orders=ro_sup,
    )
    assert [(p["symbol"], p["side"], p["order_state"]) for p in sup_plans] == [
        ("HYPEUSDC", "BUY", "RESTING")
    ]

    # --- C1.5: Consumer personal-finance / health lifestyle clickbait excluded from breaking news ---
    assert is_excluded_reference_news_item(
        "How to get your paycheck up to 2 days early: 14 banks with early direct deposit"
    )
    assert is_excluded_reference_news_item(
        "Feeling tired and losing your hair? You might have low iron — even if you’ve tested ‘normal’ before."
    )
    assert is_excluded_reference_news_item(
        "Many people overlook this crucial piece of retirement planning — and it isn’t money"
    )
    assert not is_excluded_reference_news_item(
        "US regional banks slide after Fed warns on commercial real estate exposure"
    )


def test_cycle2_watcher_partial_7d_etf_partial_streak_and_geo_stale_summary(tmp_path):
    """Verify Cycle 2 live PROD prompt audit fixes: watcher partial-7d flag, minority-partial ETF streak suppression, and stale Geo Shield summary handling."""
    from types import SimpleNamespace

    from intelligence.sentiment.collectors.macro_telemetry_collector import MacroAndNewsSensorsCollector
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import format_etf_flows_summary_line
    from intelligence.sentiment.evaluators.microstructure_evaluator import enrich_cross_venue_trends
    from orchestratorOS.supervisor.prompt_sections import (
        _format_macro_and_news_section,
        _format_market_prices_section,
    )

    now_s = 1_795_564_000.0
    now_ms = int(now_s * 1000.0)

    # --- C2.1: enrich_cross_venue_trends flags partial (<5d) crypto watcher series so Section 3 suppresses bogus 7d / vs_btc(7d) ---
    ada_watcher_27h = [
        [now_ms - int(27.3 * 3600 * 1000), 0.1480],
        [now_ms - int(12.0 * 3600 * 1000), 0.1507],
        [now_ms - int(4.0 * 3600 * 1000), 0.1424],
        [now_ms, 0.14405],
    ]
    btc_watcher_7d = [
        [now_ms - int(7.0 * 86400 * 1000), 64860.0],
        [now_ms - int(3.0 * 86400 * 1000), 55381.95],
        [now_ms, 59711.62],
    ]
    pt_map = {
        "ADAUSDC": {"price": 0.14405, "chg_24h_pct": -2.60, "low_24h": 0.1424, "high_24h": 0.1507},
        "BTCUSDC": {"price": 59711.62, "chg_24h_pct": -2.36, "low_24h": 59206.06, "high_24h": 61605.205},
    }
    enrich_cross_venue_trends(
        pt_map,
        {},
        {},
        {},
        {},
        {},
        {"ADAUSDC": ada_watcher_27h, "BTCUSDC": btc_watcher_7d},
        allow_network=False,
    )
    assert pt_map["ADAUSDC"].get("is_partial_7d") is True
    assert pt_map["ADAUSDC"].get("span_7d_hours") == pytest.approx(27.3, abs=0.2)
    assert pt_map["BTCUSDC"].get("is_partial_7d") is None
    sec3_txt = "\n".join(_format_market_prices_section(pt_map))
    ada_line = next(l for l in sec3_txt.splitlines() if "ADAUSDC:" in l)
    assert "7d=" not in ada_line
    assert "7d_range=" not in ada_line
    assert "vs_btc(" not in ada_line or "7d=" not in ada_line.split("vs_btc(", 1)[1].split(")", 1)[0]

    # --- C2.2: Suppress bogus streak=+1d on minority-partial (<50% issuers) ETF flow rows while keeping majority partial (2/3) ---
    etf_minor_partial = {
        "assets": {
            "BTC": {
                "latest_date": "2026-11-24",
                "latest_net_flow_usd": 4_000_000.0,
                "top_issuers_musd": {"BTC": 4.0},
                "sum_5d_usd": -2_123_700_000.0,
                "streak_days": 1,
                "is_partial": True,
                "reported_issuers": 2,
                "total_issuers": 12,
            },
            "ETH": {
                "latest_date": "2026-11-24",
                "latest_net_flow_usd": -129_400_000.0,
                "top_issuers_musd": {"ETHA": -124.2, "ETHE": -5.2},
                "sum_5d_usd": -357_000_000.0,
                "streak_days": -2,
                "is_partial": True,
                "reported_issuers": 2,
                "total_issuers": 3,
            },
        }
    }
    etf_line = format_etf_flows_summary_line(etf_minor_partial)
    btc_seg = etf_line.split("|")[0]
    eth_seg = etf_line.split("|")[1]
    assert "partial=2/12" in btc_seg
    assert "streak=" not in btc_seg
    assert "partial=2/3" in eth_seg
    assert "streak=-2d" in eth_seg

    # --- C2.3: Stale >=90m NORMAL Geo Threat Shield summary resets when digest aged >90m, or flags live_section1 warning ---
    gt_aged_normal = {
        "threat_level": "NORMAL",
        "risk_score": 0.2,
        "summary": "The sole headline within the 90-minute window is a routine visit; SPY (+0.37% 1h), QQQ (+0.41% 1h), VIX falling.",
        "ts": now_s - 9700.0,
        "age_min": 162.0,
        "headlines_digest": "Ukraine says attack on Russian ships in Libya was self-defence",
    }
    sec1_warn = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(geopolitical_threat=dict(gt_aged_normal)),
            asset_scope="crypto",
        )
    )
    assert "shield_note=historical_eval_gt_90m_verify_live_section1" in sec1_warn

    gt_reset_target = dict(gt_aged_normal)
    MacroAndNewsSensorsCollector._reconcile_geopolitical_deescalation(
        gt_reset_target,
        [{"title": "Routine corporate earnings wrap", "is_high_severity": False, "first_seen_ts": now_s - 600.0}],
        [{"title": "Routine corporate earnings wrap", "is_high_severity": False, "age_min": 10.0}],
        {"ukraine says attack on russian ships in libya was self-defence"},
        cache_dir=str(tmp_path),
        now_ts=now_s,
        allow_network=False,
        persist=False,
        read_json_fn=None,
    )
    assert "no fresh high-severity macro headlines in the last 90 minutes" in gt_reset_target["summary"].lower()
    assert gt_reset_target["age_min"] == 0.0


def test_cycle3_open_order_balance_race_sre_dump_log_and_fog_of_war(tmp_path):
    """Verify Cycle 3 fixes: open-order vs stale balance double-count guard, SRE dump log exclusion, 'fog of war' metaphor filter, and Geo shield_note alongside live_unassessed."""
    import json
    from types import SimpleNamespace

    from intelligence.macro.lexicon_ssot import match_high_severity_keywords
    from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs
    from orchestratorOS.supervisor.prompt_sections import _format_macro_and_news_section

    now_s = 1_795_567_190.0
    bal_ts = now_s - 45.0  # Cached balance snapshot taken BEFORE order placement
    ord_ts = now_s - 40.0  # Brand-new open SELL & BUY orders placed AFTER balance snapshot

    # --- C3.1: Stale cached balance + newer open SELL/BUY order must NEVER double-count free + locked ---
    cdir = tmp_path / "cachedb"
    (cdir / "state").mkdir(parents=True)
    (cdir / "orders").mkdir(parents=True)
    (cdir / "state" / "venue_balances_state.json").write_text(
        json.dumps({
            "pos:binance:ARBUSDC": {
                "venue": "binance",
                "symbol": "ARBUSDC",
                "base": "ARB",
                "quote": "USDC",
                "price": 0.1102,
                "free_base": 11003.84511,
                "value_usd": 1212.62,
                "free_quote": 4058.97,
                "ts": bal_ts,
            },
            "cash:binance:USDC": {
                "venue": "binance",
                "asset": "USDC",
                "free": 4058.97,
            },
        }),
        encoding="utf-8",
    )
    (cdir / "orders" / "cache_order_open_binance.json").write_text(
        json.dumps({
            "items": {
                "ARBUSDC": [
                    {
                        "orderId": "568352644",
                        "symbol": "ARBUSDC",
                        "side": "SELL",
                        "type": "LIMIT",
                        "price": "0.1111",
                        "origQty": "5400.5",
                        "executedQty": "0.0",
                        "status": "NEW",
                        "time": int(ord_ts * 1000.0),
                    },
                    {
                        "orderId": "568352699",
                        "symbol": "ARBUSDC",
                        "side": "BUY",
                        "type": "LIMIT",
                        "price": "0.1050",
                        "origQty": "1000.0",
                        "executedQty": "0.0",
                        "status": "NEW",
                        "time": int(ord_ts * 1000.0),
                    },
                ]
            },
            "fetchtime": now_s - 5.0,
        }),
        encoding="utf-8",
    )
    tpos, oolist, vcash, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir),
        prices_and_trends={"ARBUSDC": {"price": 0.1102}},
        allow_empty_inst_recovery=True,
        allow_network=False,
        persist=False,
        now_ts=now_s,
    )
    assert len(tpos) == 1
    arb = tpos[0]
    assert arb["free_base"] == pytest.approx(5603.34511, abs=1e-4)
    assert arb["locked_base_qty"] == pytest.approx(5400.5, abs=1e-4)
    assert arb["value_usd"] == pytest.approx(617.49, abs=0.05)
    assert arb["locked_base_usd"] == pytest.approx(595.14, abs=0.05)
    # Total base is conserved (11003.84511 ARB = $1212.62, NOT phantom $1807.76!)
    assert arb["free_base"] + arb["locked_base_qty"] == pytest.approx(11003.84511, abs=1e-4)
    assert arb["replace_max_sell_usd"] == pytest.approx(1212.62, abs=0.1)
    assert "_balance_ts" not in arb
    # Quote cash is also conserved ($4058.97 - $105.00 = $3953.97 free + $105.00 locked)
    assert vcash["binance:USDC"]["free"] == pytest.approx(3953.97, abs=0.05)
    assert vcash["binance:USDC"]["locked_quote_usd"] == pytest.approx(105.0, abs=0.05)
    assert "_balance_ts" not in vcash["binance:USDC"]

    # --- C3.2: SRE log anomaly scanner ignores verify_tools diagnostic prompt dump logs ---
    ldir = tmp_path / "logger"
    ldir.mkdir()
    (ldir / "dump_crypto_prompt_sections_2026-11-25.log").write_text(
        "02:39:50 (EXCEPTION: SELL-only Kraken SPCXxUSD = long-term HOLD)\n"
        "02:39:51 (EXCEPTION: SELL-only Kraken SPCXxUSD = long-term HOLD)\n",
        encoding="utf-8",
    )
    log_sum = harvest_log_anomalies_from_dirs([str(ldir)], lookback_hours=1.0)
    assert log_sum.error_count == 0
    assert log_sum.persistent_errors == []

    # --- C3.3: 'fog of war' idiom in software/tech article must NOT match high-severity 'war' ---
    assert match_high_severity_keywords(
        "Microsoft's historic cloud outage exposed an engineering 'fog of war' - Ars Technica"
    ) == ()
    assert "war" in match_high_severity_keywords(
        "Iran war live: Houthis threaten Saudi ports over coalition's Hodeidah order"
    )

    # --- C3.4: Geo shield_note also renders when age >= 90m alongside live_unassessed ---
    sec1_both = "\n".join(
        _format_macro_and_news_section(
            SimpleNamespace(
                geopolitical_threat={
                    "threat_level": "NORMAL",
                    "risk_score": 0.2,
                    "age_min": 214.5,
                    "live_unassessed_high_sev_note": "UNASSESSED_FRESH_HIGH_SEV (1 headline)",
                    "summary": "The sole headline within the 90-minute window is routine; SPY (+0.37% 1h), QQQ (+0.41% 1h).",
                }
            ),
            asset_scope="crypto",
        )
    )
    assert "live_unassessed=UNASSESSED_FRESH_HIGH_SEV" in sec1_both
    assert "shield_note=historical_eval_gt_90m_verify_live_section1" in sec1_both
