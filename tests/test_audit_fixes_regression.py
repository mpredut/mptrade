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
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics

    t0 = 1_795_504_000_000
    arb_series = [
        [t0, 0.1160],
        [t0 + 60_000, 0.1155],
        [t0 + 120_000, 0.1152],
        [t0 + 122_000, 189.7],   # corrupt single tick
        [t0 + 124_000, 0.1150],
        [t0 + 180_000, 0.1142],
    ]
    arb_m = compute_24h_price_metrics(arb_series)
    assert arb_m["high_24h"] == pytest.approx(0.1160)
    assert arb_m["low_24h"] == pytest.approx(0.1142)
    assert abs(arb_m["dd_24h_high_pct"]) < 5.0

    btc_series = [
        [t0, 61000.0],
        [t0 + 60_000, 60800.0],
        [t0 + 120_000, 60500.0],
        [t0 + 122_000, 0.1461],  # corrupt 2-tick burst
        [t0 + 123_000, 0.1461],
        [t0 + 124_000, 60450.0],
        [t0 + 180_000, 60300.0],
    ]
    btc_m = compute_24h_price_metrics(btc_series)
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
